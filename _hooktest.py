"""决定性实验：为什么 run_train.py (1.0s/it) 比 tools/train.py (0.42s/it) 慢 2.4 倍？

唯一差别是 run_train.py 注册了：
  1. ProgressHook（自定义，含 runner.optim_wrapper.get_lr() + logger.info）
  2. GpuMonitor（后台 subprocess nvidia-smi）

用真实的 Runner.train() 循环，逐个组合测 60 iter：提高对比可信度。
"""
import copy
import io
import os
import re
import subprocess
import sys
import threading
import time

from mmengine.config import Config
from mmengine.registry import init_default_scope
from mmengine.runner import Runner

init_default_scope('mmpose')

OUT = []


def L(s=''):
    print(s)
    OUT.append(str(s))


def main():
    def run_case(label, use_progress, use_gpumon, max_iters=60):
        cfg = Config.fromfile('config.py')
        # 用极少数据把 1 epoch 压到很少 iter
        cfg.train_dataloader['dataset']['indices'] = max_iters * 4
        cfg.train_dataloader['dataset']['serialize_data'] = False
        cfg.train_dataloader['num_workers'] = 4
        cfg.val_dataloader['dataset']['indices'] = 8
        cfg.val_dataloader['dataset']['serialize_data'] = False
        cfg.train_cfg = dict(by_epoch=True, max_epochs=1, val_interval=100)
        cfg.default_hooks['checkpoint'] = dict(
            type='CheckpointHook', interval=1000)
        cfg.default_hooks['visualization']['enable'] = False
        cfg.default_hooks['logger']['interval'] = 20
        cfg.work_dir = 'work_dirs/_hooktest'
        cfg.log_level = 'ERROR'          # 避免日志本身干扰
        cfg.load_from = None             # 不加载，避免 IO

        runner = Runner.from_cfg(cfg)

        hooks = []
        if use_progress:
            sys.path.insert(0, '.')
            from run_train import ProgressHook
            hooks.append(ProgressHook(
                interval=20, gpu_monitor=None, total_epochs=1,
                iters_per_epoch=max_iters))
        if hooks:
            runner.register_custom_hooks(hooks)

        gm = None
        if use_gpumon:
            from run_train import GpuMonitor
            gm = GpuMonitor(interval=2.0)
            gm.start()

        # 计时：只统计前 N 个 iter（用 IterTimerHook 的统计不便，直接墙钟）
        t0 = time.time()
        try:
            runner.train()
        except Exception as e:
            L(f'  [{label}] 训练异常: {type(e).__name__}: {e}')
        dt = time.time() - t0
        if gm:
            gm.stop()

        # 从日志文件读每个 iter 的时间
        logf = os.path.join(cfg.work_dir, '*/[0-9]*.log')
        import glob
        logs = glob.glob(logf)
        per_iter = None
        if logs:
            txt = io.open(sorted(logs)[-1], encoding='utf-8',
                          errors='replace').read()
            hits = re.findall(r'time:\s*([\d.]+)\s+data_time:\s*([\d.]+)', txt)
            if hits:
                ts = [float(a) for a, _ in hits]
                per_iter = sum(ts) / len(ts)
        L(f'  {label:<40} 墙钟 {dt:6.1f}s  '
          f'每iter={per_iter if per_iter is None else round(per_iter, 3)}')
        return per_iter

    L('=' * 82)
    L('组合实验：定位 run_train.py 的额外开销（每次跑 1 epoch ≈ 60 iter）')
    L('=' * 82)
    run_case('① 无自定义 hook（= tools/train.py）', False, False)
    run_case('② 只加 ProgressHook', True, False)
    run_case('③ 只加 GpuMonitor', False, True)
    run_case('④ 两者都加（= run_train.py）', True, True)

    L()
    L('=' * 82)
    L('说明')
    L('=' * 82)
    L('  对比 ① 与 ④ 即可知道额外开销来自哪个组件。')

    with open('_hooktest_out.txt', 'w', encoding='utf-8') as f:
        f.write('\n'.join(OUT))


if __name__ == '__main__':
    main()
