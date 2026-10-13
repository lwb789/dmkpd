# -*- coding: utf-8 -*-
"""画 24 轮检测器训练曲线（来自 work_dirs/dmkpd_det/20261012_173701/scalars.json）。

scalars.json 是 mmengine 的日志，每行一个 JSON，**不含 mode 字段**：
  训练行: {"loss": 2.58, "loss_cls": ..., "epoch": 1, "iter": 50,
           "step": 50, "lr": ..., "data_time": ..., "time": ...}
  验证行: {"coco/bbox_mAP": 0.928, ..., "step": 24}   ← 此处的 step 是 epoch
所以用「有没有 coco/bbox_mAP」来区分两类。
"""
import json
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

matplotlib.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei']
matplotlib.rcParams['axes.unicode_minus'] = False

SRC = r'work_dirs/dmkpd_det/20261012_173701/scalars.json'
OUT = r'vis_out2/det_curve.jpg'

train, val = [], []
with open(SRC, encoding='utf-8') as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        try:
            d = json.loads(line)
        except json.JSONDecodeError:
            continue
        if 'coco/bbox_mAP' in d:
            val.append(d)
        elif 'loss' in d:
            train.append(d)

print(f'train 记录 {len(train)} 条, val 记录 {len(val)} 条')
if val:
    print('val 首条键:', sorted(val[0].keys()))

# 训练记录里 step 是全局 step，val 记录里 step 是 epoch，各自排序
train.sort(key=lambda x: x.get('step', 0))
val.sort(key=lambda x: x.get('step', 0))

fig, axes = plt.subplots(1, 3, figsize=(19, 5.2))
fig.suptitle('RTMDet-tiny 检测器训练曲线（24 epoch, RTX 4060 Laptop）',
             fontsize=13, fontweight='bold')

# ---- (1) 训练 loss ---- #
ax = axes[0]
loss_keys = ['loss', 'loss_cls', 'loss_bbox', 'loss_giou']
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
for k, c in zip(loss_keys, colors):
    if not train or k not in train[0]:
        continue
    xs = [d['step'] for d in train]
    ys = [d.get(k) for d in train]
    # 滑动平均让曲线可读
    w = 20
    if len(ys) > w:
        ys_s = np.convolve(ys, np.ones(w) / w, mode='valid')
        xs_s = xs[w // 2: w // 2 + len(ys_s)]
    else:
        ys_s, xs_s = ys, xs
    lw = 2.2 if k == 'loss' else 1.3
    ax.plot(xs_s, ys_s, color=c, lw=lw, label=k)
ax.set_xlabel('global step')
ax.set_ylabel('loss')
ax.set_title('(a) 训练损失（滑窗均值 w=20）')
ax.grid(alpha=0.3)
ax.legend(fontsize=9)

# ---- (2) 学习率 ---- #
ax = axes[1]
if train and 'lr' in train[0]:
    xs = [d['step'] for d in train]
    ys = [d['lr'] for d in train]
    ax.plot(xs, ys, color='#9467bd', lw=1.8)
    ax.set_yscale('log')
    ax.set_xlabel('global step')
    ax.set_ylabel('lr (log)')
    ax.set_title('(b) 学习率调度')
    ax.grid(alpha=0.3, which='both')
else:
    ax.text(0.5, 0.5, '无 lr 记录', ha='center', transform=ax.transAxes)

# ---- (3) 验证 mAP ---- #
ax = axes[2]
keys = [('coco/bbox_mAP', 'mAP', '#1f77b4', 'o-', 2.4),
        ('coco/bbox_mAP_50', 'mAP@50', '#2ca02c', 's--', 1.6),
        ('coco/bbox_mAP_75', 'mAP@75', '#ff7f0e', '^--', 1.6),
        ('coco/bbox_mAP_s', 'mAP(小)', '#d62728', 'v:', 1.8),
        ('coco/bbox_mAP_m', 'mAP(中)', '#8c564b', 'D:', 1.3),
        ('coco/bbox_mAP_l', 'mAP(大)', '#7f7f7f', 'x:', 1.3)]
for k, lab, c, m, lw in keys:
    if not val or k not in val[0]:
        continue
    xs = [d['step'] for d in val]
    ys = [d.get(k) for d in val]
    ax.plot(xs, ys, m, color=c, lw=lw, ms=5, label=lab)
if val:
    bi = int(np.argmax([d.get('coco/bbox_mAP', -1) for d in val]))
    bx = val[bi]['step']
    by = val[bi]['coco/bbox_mAP']
    ax.annotate(f'最优 {by:.4f} (ep{bx})', (bx, by),
                textcoords='offset points', xytext=(4, -16),
                fontsize=9, color='#1f77b4', fontweight='bold')
    ax.axvline(bx, color='#1f77b4', ls=':', alpha=0.5)
ax.set_xlabel('epoch')
ax.set_ylabel('mAP')
ax.set_title('(c) 验证集 mAP（每 3 epoch 一次）')
ax.grid(alpha=0.3)
ax.legend(fontsize=8.5, loc='lower right')
ax.set_ylim(0.3, 1.02)

plt.tight_layout()
os.makedirs(os.path.dirname(OUT), exist_ok=True)
plt.savefig(OUT, dpi=140, bbox_inches='tight')
plt.close()
print(f'已保存 {OUT}')

# ---- 打印 epoch 级 val 表 ---- #
print()
print('=' * 70)
print(f'{"epoch":>8}{"mAP":>9}{"mAP50":>9}{"mAP75":>9}{"mAP_s":>9}'
      f'{"mAP_m":>9}{"mAP_l":>9}')
for d in val:
    print(f'{d["step"]:>8}{d.get("coco/bbox_mAP", 0):>9.4f}'
          f'{d.get("coco/bbox_mAP_50", 0):>9.4f}'
          f'{d.get("coco/bbox_mAP_75", 0):>9.4f}'
          f'{d.get("coco/bbox_mAP_s", 0):>9.4f}'
          f'{d.get("coco/bbox_mAP_m", 0):>9.4f}'
          f'{d.get("coco/bbox_mAP_l", 0):>9.4f}')
