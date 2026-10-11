"""最终对照：用真实 dataloader + model，逐项控制变量。

已知：单点测试都是 ~400ms，但真实训练是 1006ms。
本脚本用真实 DataLoader（不同 num_workers）迭代 + 模型前向，测端到端。
"""
import copy
import time

import numpy as np
import torch
from mmengine.config import Config
from mmengine.optim import build_optim_wrapper
from mmengine.registry import init_default_scope
from mmengine.runner import Runner

init_default_scope('mmpose')

from mmpose.registry import MODELS  # noqa: E402

OUT = []


def L(s=''):
    print(s)
    OUT.append(str(s))


def main():
    cfg = Config.fromfile('config.py')
    model = MODELS.build(cfg.model).cuda()
    model.train()
    ow = build_optim_wrapper(model, cfg.optim_wrapper)

    def bench(label, dl_cfg, n=12, do_model=True):
        loader = Runner.build_dataloader(dl_cfg, seed=0)
        it = iter(loader)
        for _ in range(2):
            try:
                b = next(it)
            except StopIteration:
                it = iter(loader)
                b = next(it)
            if do_model:
                model.train_step(b, ow)
        torch.cuda.synchronize()

        ts = []
        for _ in range(n):
            t0 = time.perf_counter()
            try:
                b = next(it)
            except StopIteration:
                it = iter(loader)
                b = next(it)
            t_load = time.perf_counter() - t0
            t1 = time.perf_counter()
            if do_model:
                model.train_step(b, ow)
                torch.cuda.synchronize()
            ts.append((t_load, time.perf_counter() - t1))
        ts = np.array(ts)
        L(f'  {label:<30} 加载{ts[:, 0].mean()*1000:>7.1f} + '
          f'模型{ts[:, 1].mean()*1000:>7.1f} = '
          f'{ts[:, 0].mean()*1000 + ts[:, 1].mean()*1000:>7.1f} ms')

    L('=' * 78)
    L('端到端每迭代耗时（真实 DataLoader + 模型）')
    L('=' * 78)

    base = copy.deepcopy(cfg.train_dataloader)
    base['num_workers'] = 0
    base['persistent_workers'] = False
    bench('workers=0 (主进程)', base)

    for w in (2, 4, 8):
        d = copy.deepcopy(cfg.train_dataloader)
        d['num_workers'] = w
        d['persistent_workers'] = True
        bench(f'workers={w} persistent', d)

    L()
    L('仅 DataLoader（不跑模型）:')
    for w in (0, 4):
        d = copy.deepcopy(cfg.train_dataloader)
        d['num_workers'] = w
        d['persistent_workers'] = w > 0
        bench(f'workers={w}', d, n=12, do_model=False)

    L()
    L('=' * 78)
    L('汇总')
    L('=' * 78)
    L('  纯 GPU 计算（固定输入）     ~400 ms')
    L('  完整 train_step（CPU 输入） ~407 ms')
    L('  真实训练日志                1006 ms')

    with open('_prof2_out.txt', 'w', encoding='utf-8') as f:
        f.write('\n'.join(OUT))


if __name__ == '__main__':
    main()
