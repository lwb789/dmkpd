"""验证 batch size 对吞吐的影响 —— 判断能否用增大 batch 来提速。

核心问题：GTX 1650 只有 896 个 CUDA 核心，
bs=4 时 GPU 可能未被喂满（实测 train_step bs=1 要 170ms，
而 bs=4 纯 GPU 只要 400ms -> 说明小 batch 效率低）。
本脚本测 bs=2/4/6/8 的单迭代耗时与"每样本耗时"。
"""
import time

import torch
from mmengine.config import Config
from mmengine.registry import init_default_scope

init_default_scope('mmpose')

from mmpose.registry import MODELS  # noqa: E402

OUT = []


def L(s=''):
    print(s)
    OUT.append(str(s))


cfg = Config.fromfile('config.py')
model = MODELS.build(cfg.model).cuda()
opt = torch.optim.Adam(model.parameters(), lr=1e-4)
crit = MODELS.build(dict(type='KeypointMSELoss', use_target_weight=True))

L('=' * 74)
L('batch size 对吞吐的影响（输入 256x256，HRNet-W48）')
L('=' * 74)
L(f'  {"bs":>4} {"单迭代":>10} {"每样本":>10} {"显存":>9}')
L('  ' + '-' * 40)

results = {}
for bs in (2, 4, 6, 8):
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    x = torch.randn(bs, 3, 256, 256, device='cuda')
    hm = torch.rand(bs, 4, 64, 64, device='cuda')
    tw = torch.ones(bs, 4, device='cuda')

    def step():
        feats = model.extract_feat(x)
        pred = model.head.forward(feats)
        loss = crit(pred, hm, tw)
        opt.zero_grad(); loss.backward(); opt.step()

    try:
        for _ in range(3):
            step()
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        n = 12
        for _ in range(n):
            step()
        torch.cuda.synchronize()
        ms = (time.perf_counter() - t0) / n * 1000
    except torch.cuda.OutOfMemoryError:
        L(f'  {bs:>4}  OOM')
        continue

    mem = torch.cuda.max_memory_allocated() / 1024**3
    results[bs] = (ms, ms / bs, mem)
    L(f'  {bs:>4} {ms:>8.1f}ms {ms/bs:>8.1f}ms {mem:>7.2f}GB')

L('  ' + '-' * 40)
L()
L('  每样本耗时（越小越划算，基线 = bs=4）:')
per4 = results[4][1]
for bs, (ms, per, mem) in results.items():
    L(f'    bs={bs}: {per:6.1f} ms/样本   = bs=4 的 {per4/per:.2f}x 吞吐')
L()

L('=' * 74)
L('对 20494 实例 / 1 epoch 的推算（纯 GPU，不含 loader）')
L('=' * 74)
L(f'  {"bs":>4} {"iter数":>8} {"样本/秒":>10} {"1 epoch":>11}')
L('  ' + '-' * 42)
for bs, (ms, per, mem) in results.items():
    iters = 20494 / bs
    sps = 1000.0 / per
    h = 20494 / sps / 3600
    L(f'  {bs:>4} {iters:>8.0f} {sps:>9.1f} {h:>9.2f} h')

L()
L('  注：真实训练还要加上 optimizer 状态、数据预处理等开销，')
L('      实测 bs=4 真实 1006ms vs 纯 GPU 400ms，比值约 2.5x。')

with open('_bench_out.txt', 'w', encoding='utf-8') as f:
    f.write('\n'.join(OUT))
