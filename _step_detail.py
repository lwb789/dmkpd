# -*- coding: utf-8 -*-
"""最终对照图：原图窗口 -> 256裁剪 -> 热力图叠加（含GT vs 峰值偏差）"""
import numpy as np
import copy
import cv2

from mmengine.config import Config
from mmengine.registry import init_default_scope
from mmpose.registry import TRANSFORMS

IDX, IN, HMS, CELL = 22, 256, 64, 620
KP_COLORS = [(0, 0, 255), (0, 255, 0), (255, 0, 0), (0, 255, 255)]
KP_NAMES = ['0=TL', '1=TR', '2=BR', '3=BL']

cfg = Config.fromfile('config.py')
init_default_scope('mmpose')
from mmpose.datasets.datasets.DM.registry import DMKPD_dataset

ds = DMKPD_dataset(data_root='DMKPD_dataset', ann_file='splits/val.txt',
                   data_prefix=dict(img='train/images'),
                   metainfo=cfg.metainfo, serialize_data=False)
info = ds.get_data_info(IDX)


def run(res, spec, record=False):
    t = TRANSFORMS.build(spec)
    if record:
        t.record = True
    return t(res)


r = run(copy.deepcopy(info), dict(type='GetBBoxCenterScale', padding=1.5))
c = np.asarray(r['bbox_center']).reshape(-1, 2)[0]
s = np.asarray(r['bbox_scale']).reshape(-1, 2)[0]
side = max(s)
W, H = info['width'], info['height']
x0 = int(round(c[0] - side / 2))
y0 = int(round(c[1] - side / 2))
side_i = int(round(side))
img = cv2.imread(info['img_path'])
win = np.zeros((side_i, side_i, 3), np.uint8)
sx0, sy0 = max(0, x0), max(0, y0)
sx1, sy1 = min(W, x0 + side_i), min(H, y0 + side_i)
win[sy0 - y0:sy1 - y0, sx0 - x0:sx1 - x0] = img[sy0:sy1, sx0:sx1]
print(f'窗口 side={side_i}  source rect=({sx0},{sy0})-({sx1},{sy1}) '
      f'-> 窗口内偏移 ({sx0 - x0},{sy0 - y0})  黑边 '
      f'右{side_i - (sx1 - x0) if sx1>=x0 else 0}px/下{side_i - (sy1 - y0)}px')

# 2) 完整管线（重新从原始 info 开始，避免 padding 叠加）
res = copy.deepcopy(info)
for spec in [dict(type='LoadImage'),
             dict(type='GetBBoxCenterScale', padding=1.5),
             dict(type='RandomBBoxTransform', shift_prob=0.0, scale_prob=0.0,
                  rotate_prob=0.0, scale_factor=(1.0, 1.0)),
             dict(type='TopdownAffine', input_size=(IN, IN))]:
    res = run(res, spec)
crop = res['img']
tk = np.asarray(res['transformed_keypoints']).reshape(-1, 2)
print('crop', crop.shape, 'tk', np.round(tk, 1).tolist())

hm_res = run(res, dict(type='GenerateTarget', encoder=dict(
    type='MSRAHeatmap', input_size=(IN, IN), heatmap_size=(HMS, HMS),
    sigma=2)), record=True)
hm = np.asarray(hm_res['heatmaps'])[:4].astype(np.float32)
print('hm', hm.shape)

# 3) 面板构建
def fit(im, cell=CELL):
    h, w = im.shape[:2]
    sc = min(cell / w, cell / h)
    nw, nh = max(1, int(w * sc)), max(1, int(h * sc))
    rs = cv2.resize(im, (nw, nh), interpolation=cv2.INTER_AREA)
    out = np.full((cell, cell, 3), 22, np.uint8)
    oy, ox = (cell - nh) // 2, (cell - nw) // 2
    out[oy:oy + nh, ox:ox + nw] = rs
    return out


# 面板A：原图缩略 + 窗口框 + 关键点
full = img.copy()
for k in range(4):
    p = tuple(np.round(info['keypoints'][0][k]).astype(int))
    cv2.circle(full, p, 12, KP_COLORS[k], -1)
    cv2.circle(full, p, 12, (255, 255, 255), 2)
    cv2.putText(full, str(k), (p[0] + 16, p[1] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 1.6, (255, 255, 255), 4)
    cv2.putText(full, str(k), (p[0] + 16, p[1] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 1.6, KP_COLORS[k], 2)
cv2.rectangle(full, (x0, y0), (x0 + side_i, y0 + side_i), (255, 0, 255), 6)
bx = np.asarray(info['bbox'][0]).astype(int)
cv2.rectangle(full, (bx[0], bx[1]), (bx[2], bx[3]), (0, 165, 255), 5)
A = fit(full)

# 面板B：窗口内容
B = win.copy()
for k in range(4):
    p = np.asarray(info['keypoints'][0][k]) - np.array([x0, y0])
    cv2.circle(B, tuple(np.round(p).astype(int)), 7, KP_COLORS[k], -1)
    cv2.circle(B, tuple(np.round(p).astype(int)), 7, (255, 255, 255), 2)
B = fit(B)

# 面板C：256 裁剪 + GT
C = crop.copy()
for k in range(4):
    p = tuple(np.round(tk[k]).astype(int))
    cv2.circle(C, p, 5, KP_COLORS[k], -1)
    cv2.circle(C, p, 5, (255, 255, 255), 1)
    cv2.putText(C, str(k), (p[0] + 8, p[1] - 6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
C = fit(C)

# 面板D：热力图叠加（4通道取max，放大到256）
hm_max = hm.max(axis=0)
heat = cv2.applyColorMap((hm_max * 255).astype(np.uint8), cv2.COLORMAP_JET)
heat = cv2.resize(heat, (IN, IN), interpolation=cv2.INTER_NEAREST)
over = cv2.addWeighted(crop, 0.45, heat, 0.55, 0)
for k in range(4):
    p = tuple(np.round(tk[k]).astype(int))
    cv2.circle(over, p, 5, KP_COLORS[k], -1)
    cv2.circle(over, p, 5, (255, 255, 255), 1)
D = fit(over)

# 面板E：峰值 vs GT 偏差表
E = np.full((CELL, CELL, 3), 22, np.uint8)
lines = ['heatmap peak vs GT  (in 64x64 heatmap grid)',
         'k   GT/4        peak    d(px in 256)']
for k in range(4):
    yx = np.unravel_index(np.argmax(hm[k]), hm[k].shape)
    gt = tk[k] / 4.0
    d = np.hypot((yx[1] - gt[0]) * 4, (yx[0] - gt[1]) * 4)
    lines.append(f'{k}  ({gt[0]:5.2f},{gt[1]:5.2f})  '
                 f'({yx[1]:2d},{yx[0]:2d})   {d:5.2f}')
for i, t in enumerate(lines):
    cv2.putText(E, t, (18, 60 + i * 48), cv2.FONT_HERSHEY_SIMPLEX,
                0.72 if i else 0.78, (255, 255, 255) if i else (120, 220, 255),
                2 if i else 2)

titles = ['A. original 1702x1276 + window(pad1.5)',
          'B. window content (side=2151)',
          'C. TopdownAffine -> 256x256 + GT',
          'D. heatmap(argmax over 4ch) overlay',
          'E. peak vs GT error']
panels = []
for im, t in zip([A, B, C, D, E], titles):
    bar = np.full((46, CELL, 3), 45, np.uint8)
    cv2.putText(bar, t, (10, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.78,
                (255, 255, 255), 2)
    panels.append(np.vstack([bar, im]))

row = np.hstack(panels)
cv2.imwrite('vis_out2/step_detail.png', row, [cv2.IMWRITE_PNG_COMPRESSION, 6])
print('-> vis_out2/step_detail.png', row.shape[1], 'x', row.shape[0])
