# -*- coding: utf-8 -*-
"""诊断提交文件里越界 / 非凸的实例。"""
import os
import os.path as osp

import cv2
import numpy as np

SUB = 'submission'
TEST = 'DMKPD_dataset/test'
KP = [(0, 0, 255), (0, 255, 0), (255, 0, 0), (0, 255, 255)]


def load(stem):
    p = osp.join(SUB, stem + '.txt')
    out = []
    with open(p, encoding='utf-8') as f:
        for ln in f:
            if not ln.strip():
                continue
            a = ln.split()
            out.append((float(a[1]),
                        np.array([float(v) for v in a[2:]]).reshape(4, 2)))
    return out


# 找出所有越界与非凸的实例
oob, nonconvex = [], []
for fn in sorted(os.listdir(SUB)):
    if not fn.endswith('.txt'):
        continue
    stem = fn[:-4]
    img = cv2.imread(osp.join(TEST, stem + '.jpg'))
    if img is None:
        continue
    H, W = img.shape[:2]
    for i, (conf, pts) in enumerate(load(stem)):
        if pts[:, 0].min() < 0 or pts[:, 0].max() > W - 1 or \
           pts[:, 1].min() < 0 or pts[:, 1].max() > H - 1:
            oob.append((stem, i, conf, pts.copy()))
        v = np.roll(pts, -1, axis=0) - pts
        v2 = np.roll(v, -1, axis=0)
        cr = v[:, 0] * v2[:, 1] - v[:, 1] * v2[:, 0]
        if not (np.all(cr > 0) or np.all(cr < 0)):
            nonconvex.append((stem, i, conf, pts.copy()))

print(f'越界 {len(oob)} 个，非凸 {len(nonconvex)} 个')
print()
print('--- 越界明细 ---')
for stem, i, c, pts in oob:
    img = cv2.imread(osp.join(TEST, stem + '.jpg'))
    H, W = img.shape[:2]
    base = np.array([pts[:, 0].min(), pts[:, 1].min()])
    print(f'  {stem} #{i} conf={c:.3f} 图{W}x{H}')
    print(f'    pred {np.round(pts - base, 1).tolist()}')
    print(f'    绝对 x[{pts[:,0].min():.0f},{pts[:,0].max():.0f}] '
          f'y[{pts[:,1].min():.0f},{pts[:,1].max():.0f}]')

print()
print('--- 非凸明细 ---')
for stem, i, c, pts in nonconvex:
    print(f'  {stem} #{i} conf={c:.3f}')
    print(f'    {np.round(pts, 1).tolist()}')
    v = np.roll(pts, -1, axis=0) - pts
    v2 = np.roll(v, -1, axis=0)
    print(f'    cross={np.round(v[:,0]*v2[:,1]-v[:,1]*v2[:,0], 1).tolist()}')

# 画出来
CELL = 640
panels = []
for stem, i, c, pts in (oob[:4] + nonconvex):
    img = cv2.imread(osp.join(TEST, stem + '.jpg'))
    H, W = img.shape[:2]
    pad = 60
    x1 = max(0, int(pts[:, 0].min()) - pad)
    x2 = min(W, int(pts[:, 0].max()) + pad)
    y1 = max(0, int(pts[:, 1].min()) - pad)
    y2 = min(H, int(pts[:, 1].max()) + pad)
    x1 = min(x1, max(0, int(pts[:, 0].min())))   # 保证可见边缘
    y1 = min(y1, max(0, int(pts[:, 1].min())))
    crop = np.ascontiguousarray(img[y1:y2, x1:x2])
    for j in range(4):
        p = tuple(np.round(pts[j] - [x1, y1]).astype(int))
        cv2.circle(crop, p, 10, KP[j], -1)
        cv2.circle(crop, p, 10, (255, 255, 255), 2)
        cv2.putText(crop, str(j), (p[0] + 12, p[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 3)
        cv2.putText(crop, str(j), (p[0] + 12, p[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, KP[j], 2)
    for j in range(4):
        cv2.line(crop, tuple(np.round(pts[j] - [x1, y1]).astype(int)),
                 tuple(np.round(pts[(j + 1) % 4] - [x1, y1]).astype(int)),
                 (255, 255, 0), 3)
    h, w = crop.shape[:2]
    sc = min(CELL / w, CELL / h)
    rs = cv2.resize(crop, (max(1, int(w * sc)), max(1, int(h * sc))))
    cell = np.full((CELL, CELL, 3), 20, np.uint8)
    oy, ox = (CELL - rs.shape[0]) // 2, (CELL - rs.shape[1]) // 2
    cell[oy:oy + rs.shape[0], ox:ox + rs.shape[1]] = rs
    bar = np.full((36, CELL, 3), 45, np.uint8)
    cv2.putText(bar, f'{stem} #{i} conf={c:.2f}', (8, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    panels.append(np.vstack([bar, cell]))

while len(panels) % 3:
    panels.append(np.full((CELL + 36, CELL, 3), 20, np.uint8))
rows = [np.hstack(panels[i:i + 3]) for i in range(0, len(panels), 3)]
grid = np.vstack(rows)
cv2.imwrite('vis_out2/sub_issues.jpg', grid,
            [cv2.IMWRITE_JPEG_QUALITY, 92])
print()
print('-> vis_out2/sub_issues.jpg', grid.shape[1], 'x', grid.shape[0])
