# -*- coding: utf-8 -*-
"""提交文件格式自检。

对照比赛规格逐条核对：
  1. 每张测试图对应一个 .txt（**没有目标的图必须是空文件**）
  2. 每行格式：`DPM conf x1 y1 x2 y2 x3 y3 x4 y4`（1 + 1 + 8 = 10 个字段）
  3. conf ∈ [0,1]，且 >= 0.3（低于 0.3 不参与评测）
  4. 坐标为**原图像素**，不归一化；应在 [0, W] x [0, H] 内
  5. 顺序 TL→TR→BR→BL（推荐；评测会 normalize，但保持一致可自检）
  6. 四个角点应构成**凸四边形**、顺序应为顺时针、面积合理
"""
import glob
import os
import os.path as osp

import cv2
import numpy as np

SUB = 'submission'
TEST = 'DMKPD_dataset/test'

files = sorted(glob.glob(osp.join(TEST, '*.jpg')))
txts = sorted(glob.glob(osp.join(SUB, '*.txt')))

print('=' * 78)
print('1. 文件覆盖检查')
print('=' * 78)
stems_img = set(osp.splitext(osp.basename(p))[0] for p in files)
stems_txt = set(osp.splitext(osp.basename(p))[0] for p in txts)
print(f'  测试图片   {len(stems_img)}')
print(f'  提交 txt   {len(stems_txt)}')
print(f'  缺 txt     {sorted(stems_img - stems_txt)[:10] or "无"}')
print(f'  多余 txt   {sorted(stems_txt - stems_img)[:10] or "无"}')

empty = [p for p in txts if osp.getsize(p) == 0]
print(f'  空 txt（无目标，允许） {len(empty)}')

print()
print('=' * 78)
print('2. 逐行格式检查')
print('=' * 78)
n_line = 0
n_bad_field = 0
n_bad_conf = 0
n_low_conf = 0
n_oob = 0
n_bad_poly = 0
n_notcw = 0
confs = []
areas = []
sides = []
bad_examples = []

for p in txts:
    with open(p, encoding='utf-8') as f:
        lines = [ln.rstrip('\n') for ln in f if ln.strip()]
    if not lines:
        continue
    img = cv2.imread(osp.join(TEST, osp.basename(p).replace('.txt', '.jpg')))
    if img is None:
        bad_examples.append((p, '找不到对应图片'))
        continue
    H, W = img.shape[:2]
    for ln in lines:
        n_line += 1
        parts = ln.split()
        if len(parts) != 10 or parts[0] != 'DPM':
            n_bad_field += 1
            if len(bad_examples) < 5:
                bad_examples.append((p, f'字段数={len(parts)} 首字段={parts[0]}'))
            continue
        try:
            conf = float(parts[1])
            xy = np.array([float(v) for v in parts[2:]], dtype=np.float64)
        except ValueError:
            n_bad_field += 1
            continue
        confs.append(conf)
        if not (0.0 <= conf <= 1.0):
            n_bad_conf += 1
        if conf < 0.3:
            n_low_conf += 1

        pts = xy.reshape(4, 2)
        # 越界检查
        if pts[:, 0].min() < -0.5 or pts[:, 0].max() > W + 0.5 or \
           pts[:, 1].min() < -0.5 or pts[:, 1].max() > H + 0.5:
            n_oob += 1
            if len(bad_examples) < 8:
                bad_examples.append(
                    (p, f'越界 x[{pts[:,0].min():.0f},{pts[:,0].max():.0f}] '
                        f'y[{pts[:,1].min():.0f},{pts[:,1].max():.0f}] '
                        f'图 {W}x{H}'))
        # 凸性 + 有向面积（图像坐标 y 向下，顺时针 => 面积为正）
        x, y = pts[:, 0], pts[:, 1]
        ar = 0.5 * (np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))
        if ar <= 0:
            n_notcw += 1
        areas.append(abs(ar))
        sides.append(np.linalg.norm(np.roll(pts, -1, axis=0) - pts, axis=1))
        # 凸性：每个内角同号
        v = np.roll(pts, -1, axis=0) - pts
        v2 = np.roll(v, -1, axis=0)
        cross = v[:, 0] * v2[:, 1] - v[:, 1] * v2[:, 0]
        if not (np.all(cross > 0) or np.all(cross < 0)):
            n_bad_poly += 1
            if len(bad_examples) < 12:
                bad_examples.append((p, '非凸四边形'))

print(f'  总行数（目标实例）      {n_line}')
print(f'  字段格式错              {n_bad_field}')
print(f'  conf 越界 [0,1]         {n_bad_conf}')
print(f'  conf < 0.3（不该出现）   {n_low_conf}')
print(f'  坐标越出原图             {n_oob}')
print(f'  非凸四边形               {n_bad_poly}')
print(f'  非顺时针（有向面积<=0）  {n_notcw}')
if bad_examples:
    print()
    print('  问题样例：')
    for p, m in bad_examples[:8]:
        print(f'    {osp.basename(p)}: {m}')

print()
print('=' * 78)
print('3. 数值分布')
print('=' * 78)
c = np.array(confs)
print(f'  conf   min={c.min():.4f} p05={np.percentile(c,5):.4f} '
      f'中位={np.median(c):.4f} p95={np.percentile(c,95):.4f} max={c.max():.4f}')
a = np.array(areas)
print(f'  四边形面积  中位={np.median(a):.0f}px^2  '
      f'等效边长中位={np.sqrt(np.median(a)):.1f}px')
s = np.concatenate(sides)
print(f'  边长   中位={np.median(s):.1f}px  '
      f'p05={np.percentile(s,5):.1f} p95={np.percentile(s,95):.1f}')
print(f'  每图实例数  平均={n_line/len(stems_img):.2f}  最多='
      f'{max(len(open(p,encoding="utf-8").read().strip().splitlines() or [1]) if osp.getsize(p) else 0 for p in txts)}')

print()
print('=' * 78)
print('4. 提交目录内容')
print('=' * 78)
print(f'  路径   {osp.abspath(SUB)}')
print(f'  txt 数 {len(txts)}   总大小 '
      f'{sum(osp.getsize(p) for p in txts)/1024:.1f} KB')
sg = [p for p in txts if osp.getsize(p) > 0]
print(f'  非空   {len(sg)}   空 {len(txts)-len(sg)}')
if sg:
    print()
    print(f'  --- 示例 {osp.basename(sg[0])} ---')
    print('  ' + open(sg[0], encoding='utf-8').read().strip()
          .replace('\n', '\n  '))
    print()
    print(f'  --- 示例 {osp.basename(sg[-1])} ---')
    print('  ' + open(sg[-1], encoding='utf-8').read().strip()
          .replace('\n', '\n  '))
