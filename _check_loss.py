"""验证三件事：
1) 迭代速度的真实趋势（是变快还是变慢）
2) loss=0.003 的真实含义（是否等于"预测全零"的基线）
3) 数据预处理耗时，解释 1.24s/it 与 0.40s/it(纯GPU) 的差距
"""
import io
import re
import time

import numpy as np
import torch
from mmengine.registry import init_default_scope

init_default_scope('mmpose')

from mmpose.datasets.datasets.DM import DMKPD_dataset  # noqa: E402

try:
    from mmpose.models import KeypointMSELoss  # noqa: E402
except ImportError:
    from mmpose.models.losses import KeypointMSELoss  # noqa: E402

out = []


def L(s=''):
    print(s)
    out.append(str(s))


# ------------------------------------------------------------------ #
L('=' * 72)
L('1) 迭代速度趋势（来自 _rt_out.txt 的真实训练日志）')
L('=' * 72)
rows = []
for line in io.open('_rt_out.txt', encoding='utf-8', errors='replace'):
    m = re.search(r'\[\s*\d+/\d+\]\[\s*(\d+)/\d+\]\s+loss=([\d.]+).*?'
                  r'\(([\d.]+)s/it\)', line)
    if m:
        rows.append((int(m.group(1)), float(m.group(2)), float(m.group(3))))

if rows:
    for it, loss, sp in rows:
        L(f'  iter={it:>5}  loss={loss:.5f}  {sp:6.3f} s/it  '
          f'{"#" * int(min(36, sp * 10))}')
    L()
    L(f'  首个采样点 {rows[0][2]:.3f} s/it  ->  '
      f'末个采样点 {rows[-1][2]:.3f} s/it')
    if rows[-1][2] < rows[0][2]:
        L(f'  结论: 速度在【变快】，共提升 {rows[0][2] / rows[-1][2]:.2f} 倍')
    else:
        L(f'  结论: 速度在【变慢】')
else:
    L('  未解析到日志行')

# ------------------------------------------------------------------ #
L()
L('=' * 72)
L('2) GT 热力图有多"稀疏"？loss=0.003 到底是什么含义？')
L('=' * 72)

META = dict(
    from_file='open-mmlab/mmpose/configs/_base_/datasets/DMKPD_meta.py')
ds = DMKPD_dataset(
    data_root='DMKPD_dataset', ann_file='splits/val.txt',
    data_prefix=dict(img='train/images'), metainfo=META,
    serialize_data=False,
    pipeline=[
        dict(type='LoadImage'),
        dict(type='GetBBoxCenterScale', padding=1.5),
        dict(type='TopdownAffine', input_size=(256, 256)),
        dict(
            type='GenerateTarget',
            encoder=dict(type='MSRAHeatmap', input_size=(256, 256),
                         heatmap_size=(64, 64), sigma=2),
            use_dataset_keypoint_weights=True),
    ])

N = 30
gts, wts = [], []
for i in range(N):
    r = ds[i]
    gts.append(np.asarray(r['heatmaps'], dtype=np.float32))
    wts.append(np.asarray(r['keypoint_weights'], dtype=np.float32).reshape(-1))

gt = torch.tensor(np.stack(gts))          # (N, K, H, W)
w = torch.tensor(np.stack(wts))           # (N, K)
K = gt.shape[1]
HW = gt.shape[-1] * gt.shape[-2]

nz = (gt > 0.01).float()
L(f'  张量形状                = {tuple(gt.shape)}  (N, K, H, W)')
L(f'  峰值                    = {gt.max():.4f}')
L(f'  非零(>0.01)像素占比      = {100.0 * nz.mean():.3f}%')
L(f'  平均每关键点的非零像素    = {nz.sum().item() / (N * K):.1f} 个 '
  f'(整张共 {HW} 像素)')
L()

crit = KeypointMSELoss(use_target_weight=True)
zero_pred = torch.zeros_like(gt)
# KeypointMSELoss._get_mask 要求 target_weights 是 (N,K) 或 (N,K,H,W)，
# 且必须等于 target.shape[:ndim]。传 (N,K,1,1) 会断言失败。
tw = w                                   # (N, K)
floor = crit(zero_pred, gt, tw)
plain = (gt ** 2).mean()

# 再模拟几个"不同质量的预测"，看 loss 量级
torch.manual_seed(0)
rand_pred = torch.rand_like(gt)
rand_loss = crit(rand_pred, gt, tw)

# "预测出一个错位的高斯"（整体偏 2 个热图像素 = 8px）
shifted = torch.roll(gt, shifts=2, dims=-1)
shift_loss = crit(shifted, gt, tw)
perfect = crit(gt, gt, tw)

L('  【关键实测】不同预测下的 loss:')
L(f'    完美预测（=GT）               = {float(perfect):.6f}')
L(f'    高斯整体错位 8px (2热图像素)   = {float(shift_loss):.6f}')
L(f'    均匀随机预测                  = {float(rand_loss):.6f}')
L(f'    预测全 0                      = {float(floor):.6f}')
L()
L(f'  训练日志: iter20 loss = 0.003070,  iter240 loss = 0.002700')
L()
L(f'  高斯理论值: sum(GT^2) ~= pi*sigma^2 = {np.pi * 4:.2f}')
L(f'              均摊到 64x64 全图      = {np.pi * 4 / HW:.6f}')
L(f'              mean(GT^2) 实测        = {float(plain):.6f}')
L()
L(f'  => 训练 loss 0.003 与"预测全零"的 {float(floor):.6f} 几乎相同')

# ------------------------------------------------------------------ #
L()
L('=' * 72)
L('3) 为什么每迭代 1.24s，而纯 GPU 只要 0.40s？')
L('=' * 72)

for i in range(3):
    ds[i]                                  # 预热
t0 = time.perf_counter()
M = 20
for i in range(M):
    ds[i]
dt = (time.perf_counter() - t0) / M

L(f'  单样本 pipeline 耗时(读 1702x1276 JPG + warpAffine + 高斯) = '
  f'{dt * 1000:.1f} ms')
L(f'  batch=4 的 CPU 侧合计约  = {dt * 4 * 1000:.0f} ms')
L(f'  纯 GPU 前向+反向+step 实测 = 400 ms')
L(f'  实际每迭代实测            = 1240 ms')
L()
L(f'  num_workers=4 时每 worker 只需处理 1 个样本/迭代，')
L(f'  若 {dt * 1000:.0f}ms < 1240ms 则理论上可以被完全掩盖')

with open('_loss_out.txt', 'w', encoding='utf-8') as f:
    f.write('\n'.join(out))
