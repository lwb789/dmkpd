# -*- coding: utf-8 -*-
"""热力图放大：每通道单独放一张 64x64 -> 512x512。"""
import os
import numpy as np
import cv2

HMS, IN = 64, 256
IDX = 22

# 直接复用 _step_vis 的管线，取 GenerateTarget 的 heatmaps
import sys
sys.argv = [sys.argv[0]]
import importlib.util

spec = importlib.util.spec_from_file_location('sv', r'_step_vis.py')
# 不执行它的绘图部分，改为手动重跑管线
from mmengine.config import Config
from mmengine.registry import init_default_scope
import mmpose  # noqa
from mmpose.registry import DATASETS, TRANSFORMS
from mmengine.dataset import Compose

cfg = Config.fromfile('config.py')
init_default_scope('mmpose')
from mmpose.datasets.datasets.DM.registry import DMKPD_dataset
ds = DMKPD_dataset(
    data_root='DMKPD_dataset', ann_file='splits/val.txt',
    data_prefix=dict(img='train/images'), metainfo=cfg.metainfo,
    serialize_data=False)
raw = ds.get_data_info(IDX)
data_info = raw if isinstance(raw, dict) else raw[0]
print('sample img_id =', data_info.get('img_id'))

steps = [
    dict(type='LoadImage'),
    dict(type='GetBBoxCenterScale', padding=1.5),
    dict(type='RandomBBoxTransform', shift_prob=0.0, scale_prob=0.0,
         rotate_prob=0.0, scale_factor=(1.0, 1.0)),
    dict(type='TopdownAffine', input_size=(IN, IN)),
    dict(type='GenerateTarget', encoder=dict(
        type='MSRAHeatmap', input_size=(IN, IN),
        heatmap_size=(HMS, HMS), sigma=2)),
]
res = data_info
for s in steps:
    t = TRANSFORMS.build(s)
    res = t(res)
hm = np.asarray(res['heatmaps'])[:4].astype(np.float32)
print('heatmaps', hm.shape)

tiles = []
for k in range(4):
    m = hm[k]
    cam = cv2.applyColorMap((m * 255).astype(np.uint8), cv2.COLORMAP_JET)
    big = cv2.resize(cam, (512, 512), interpolation=cv2.INTER_NEAREST)
    # 画峰值坐标
    yx = np.unravel_index(np.argmax(m), m.shape)
    cy, cx = int((yx[0] + 0.5) * 512 / HMS), int((yx[1] + 0.5) * 512 / HMS)
    cv2.drawMarker(big, (cx, cy), (255, 255, 255), cv2.MARKER_CROSS, 26, 2)
    cv2.putText(big, f'ch{k} peak=({yx[1]},{yx[0]}) v={m.max():.3f}',
                (8, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 2)
    tiles.append(big)
    print(f'  ch{k}: peak row={yx[0]} col={yx[1]} peak_val={m.max():.4f} '
          f'nonzero={int((m>0.01).sum())}')

top = np.hstack(tiles[:2])
bot = np.hstack(tiles[2:])
grid = np.vstack([top, bot])
cv2.imwrite(r'vis_out2/heat_channels.png', grid)
print('-> vis_out2/heat_channels.png', grid.shape[1], 'x', grid.shape[0])

# 单通道归一化后的等高线视图（看高斯形状）
m0 = hm[0]
print('ch0 5x5 around peak:')
yx = np.unravel_index(np.argmax(m0), m0.shape)
r, c = yx
print(np.round(m0[max(0, r-2):r+3, max(0, c-2):c+3], 4))
