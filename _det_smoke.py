# -*- coding: utf-8 -*-
"""检测数据管线冒烟测试：确认路径解析、标注对齐、pack 后的张量形状。"""
import numpy as np
from mmengine.config import Config
from mmengine.registry import init_default_scope

cfg = Config.fromfile('config_det.py')
init_default_scope('mmdet')

from mmengine.dataset import Compose
from mmdet.registry import DATASETS

print('=' * 78)
print('构建数据集')
print('=' * 78)
for name, ann in [('train', 'det_dataset/annotations/instances_train.json'),
                  ('val', 'det_dataset/annotations/instances_val.json'),
                  ('test', 'det_dataset/annotations/instances_test.json')]:
    ds = DATASETS.build(dict(
        type='CocoDataset', metainfo=dict(classes=('DPM', )), data_root='.',
        ann_file=ann, data_prefix=dict(img=''),
        pipeline=cfg.test_pipeline_noann if name == 'test'
        else cfg.test_pipeline,
        test_mode=(name != 'train')))
    print(f'  {name:<6} len={len(ds):<6} '
          f'classes={ds.metainfo.get("classes")} '
          f'cat_ids={list(ds.cat2label.items())[:3]}')

print()
print('=' * 78)
print('逐样本检查（train 前 3 个）')
print('=' * 78)
ds = DATASETS.build(dict(
    type='CocoDataset', metainfo=dict(classes=('DPM', )), data_root='.',
    ann_file='det_dataset/annotations/instances_train.json',
    data_prefix=dict(img=''), pipeline=cfg.train_pipeline,
    serialize_data=False))
for i in range(3):
    raw = ds.get_data_info(i)
    print(f'  [{i}] img={raw["img_path"]}')
    import os
    print(f'      存在={os.path.isfile(raw["img_path"])}  '
          f'width={raw["width"]} height={raw["height"]}')
    # gt_bboxes 由 pipeline 里的 LoadAnnotations 加上，原始 data_info 里没有
    print(f'      raw keys={sorted(k for k in raw if "gt" in k or "inst" in k)}')
    if 'instances' in raw:
        print(f'      instances = {len(raw["instances"])} 个')
        for it in raw['instances'][:3]:
            print(f'        bbox={np.round(it["bbox"], 1).tolist()} '
                  f'label={it["bbox_label"]}')

print()
print('=' * 78)
print('完整 pipeline 走一遍（含 Resize+Pad+Pack）')
print('=' * 78)
pipe = Compose(cfg.train_pipeline)
r = pipe(ds.get_data_info(0))
inp = r['inputs']
print(f'  inputs          shape={tuple(inp.shape)} dtype={inp.dtype}')
print(f'  归一化后范围     [{inp.min():.2f}, {inp.max():.2f}] '
      f'(mean=0/std=1 附近)')
ds_meta = r['data_samples']
print(f'  metainfo keys   {list(ds_meta.metainfo.keys())}')
print(f'  img_shape       {ds_meta.metainfo.get("img_shape")}')
print(f'  scale_factor    {ds_meta.metainfo.get("scale_factor")}')
g = ds_meta.gt_instances
print(f'  gt bboxes       shape={tuple(g.bboxes.shape)}  '
      f'type={type(g.bboxes).__name__}')
for b in g.bboxes.tensor:
    print(f'      {np.round(b.tolist(), 1)}')
print(f'  gt labels       {g.labels.tolist()}')

# 检查框是否在 1024 内
bx = g.bboxes.tensor
print()
print(f'  框坐标范围      x [{bx[:, 0].min():.1f}, {bx[:, 2].max():.1f}]  '
      f'y [{bx[:, 1].min():.1f}, {bx[:, 3].max():.1f}]')
w = (bx[:, 2] - bx[:, 0])
print(f'  缩放后边长      {np.round(w.tolist(), 1)}')

# 反推：验证缩放映射自洽
sf = ds_meta.metainfo['scale_factor']
print(f'  scale_factor={tuple(round(v,4) for v in sf)}')
print(f'  原始 1702x1276 长边 -> 1024,  1024/1702 = {1024/1702:.4f}')
bw = float(w[0])
print(f'  缩放后边长 {bw:.1f}px  ->  反算原图 {bw/sf[0]:.1f}px')
print(f'  raw instances bbox 宽 = '
      f'{r["data_samples"].metainfo["img_shape"]} (应为 1024x1024)')

# 抽查图形内容确实不是全黑（证明图片真的读进来了）
print()
print(f'  图像非零像素占比 {(inp.abs().sum(0) > 1e-6).float().mean():.3f}')
print(f'  左上/中心/右下像素均值 '
      f'{inp[:, :64, :64].mean():.2f} / {inp[:, 480:544, 480:544].mean():.2f} /'
      f' {inp[:, -64:, -64:].mean():.2f}')
