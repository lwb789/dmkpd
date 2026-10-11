"""验证：大码的误差是"单位换算效应"还是"真实定位能力差"。

核心检验：把误差归一化成"占码边长的百分比"。
若各组相对误差接近 => 模型相对精度一致，绝对像素差只是码大导致。
"""
import numpy as np
from mmengine.config import Config
from mmengine.registry import init_default_scope

init_default_scope('mmpose')

from mmpose.apis import inference_topdown, init_model  # noqa: E402
from mmpose.datasets.datasets.DM import DMKPD_dataset  # noqa: E402

cfg = Config.fromfile('config.py')
m = init_model(cfg, 'work_dirs/dmkpd/best_coco_AP_epoch_5.pth',
               device='cuda:0')
ds = DMKPD_dataset(data_root='DMKPD_dataset', ann_file='splits/val.txt',
                   data_prefix=dict(img='train/images'),
                   metainfo=cfg.metainfo, serialize_data=False)

S, E, MX = [], [], []
for i in range(len(ds)):
    info = ds.get_data_info(i)
    r = inference_topdown(m, info['img_path'], bboxes=info['bbox'],
                          bbox_format='xyxy')
    p = r[0].pred_instances.keypoints[0]
    g = info['keypoints'][0]
    e = np.linalg.norm(p - g, axis=1)
    S.append(np.sqrt(info['area']))
    E.append(e.mean())
    MX.append(e.max())
S, E, MX = np.array(S), np.array(E), np.array(MX)

print('=' * 78)
print('绝对误差 vs 相对误差（按码大小分组）')
print('=' * 78)
print(f'{"分组":<16}{"n":>5}{"绝对EPE均值":>12}{"绝对中位":>10}'
      f'{"相对均值":>10}{"相对中位":>10}')
print('-' * 66)
for lo, hi, name in [(0, 60, '小码<60px'), (60, 120, '中60~120'),
                     (120, 260, '大120~260'), (260, 1e9, '超大>260')]:
    k = (S >= lo) & (S < hi)
    if k.sum():
        rel = E[k] / S[k]
        print(f'{name:<16}{k.sum():>5}{E[k].mean():>12.2f}'
              f'{np.median(E[k]):>10.2f}{rel.mean() * 100:>9.2f}%'
              f'{np.median(rel) * 100:>9.2f}%')

print()
print('=' * 78)
print('理论量化下限（热力图 64x64，即 1 热图像素 = 4 输入像素）')
print('=' * 78)
print(f'{"分组":<16}{"缩放倍率":>10}{"1热图像素":>12}{"占码边长":>10}')
print('-' * 50)
for lo, hi, name in [(0, 60, '小码<60px'), (60, 120, '中60~120'),
                     (120, 260, '大120~260'), (260, 1e9, '超大>260')]:
    k = (S >= lo) & (S < hi)
    if k.sum():
        sc = S[k] * 1.5 / 256
        q = 4 * sc
        print(f'{name:<16}{np.median(sc):>9.2f}x{np.median(q):>11.2f}px'
              f'{np.median(q) / np.median(S[k]) * 100:>9.2f}%')

print()
print('=' * 78)
print('换算成 OKS 口径（sigma=0.1，4 点等权）')
print('=' * 78)
print(f'{"分组":<16}{"相对均值":>10}{"换算OKS":>10}{"OKS>=0.75占比":>14}')
print('-' * 52)
for lo, hi, name in [(0, 60, '小码<60px'), (60, 120, '中60~120'),
                     (120, 260, '大120~260'), (260, 1e9, '超大>260')]:
    k = (S >= lo) & (S < hi)
    if k.sum():
        rel = E[k] / S[k]
        oks = np.exp(-rel ** 2 * 12.5)
        print(f'{name:<16}{rel.mean():>10.4f}{oks.mean():>10.4f}'
              f'{(oks >= 0.75).mean() * 100:>13.1f}%')

print()
print('=' * 78)
print('结论判据')
print('=' * 78)
rel_all = E / S
print(f'  全体相对误差 均值={rel_all.mean():.4f}  中位={np.median(rel_all):.4f}')
print('  若各分组"相对中位"接近(同一量级) => 相对精度一致，')
print('  绝对像素差只是码尺寸放大的结果，不是模型能力问题。')
