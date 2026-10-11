# -*- coding: utf-8 -*-
"""环境自查：mmdet / mmcv / mmengine / mmpose 版本与 API 兼容性。

关键点：RTMDet 的 config 里用到了一些在新版本 mmdet 中已改名的类
（如 `Pad` vs `Pad`、`CachedMosaic`、`mmdet.CocoDataset`），
以及 mmengine 的 `BaseTransform`、`DetDataPreprocessor`。先确认能不能
把官方 rtmdet_tiny 配置成功地 build 起来。
"""
import os
import sys

import mmcv
import mmengine
import mmdet

print('python      ', sys.version.split()[0])
print('mmcv        ', mmcv.__version__)
print('mmengine    ', mmengine.__version__)
print('mmdet       ', mmdet.__version__, os.path.dirname(mmdet.__file__))

import torch
print('torch       ', torch.__version__, 'cuda', torch.cuda.is_available())
if torch.cuda.is_available():
    print('gpu         ', torch.cuda.get_device_name(0))

# 注册 mmdet scope 并尝试 build 官方 rtmdet_tiny
from mmengine.config import Config
from mmengine.registry import init_default_scope

CFG = os.path.join(os.path.dirname(mmdet.__file__),
                   '.mim', 'configs', 'rtmdet',
                   'rtmdet_tiny_8xb32-300e_coco.py')
print()
print('cfg exists  ', os.path.isfile(CFG))
cfg = Config.fromfile(CFG)
print('cfg loaded  ', 'model' in cfg, 'train_pipeline' in cfg)

init_default_scope('mmdet')
from mmdet.registry import MODELS

try:
    model = MODELS.build(cfg.model)
    n = sum(p.numel() for p in model.parameters())
    print('model build OK, params = %.2fM' % (n / 1e6))
    for k, v in model.named_children():
        print('   ', k, type(v).__name__)
except Exception as e:
    import traceback
    traceback.print_exc()

# 检查 pipeline 里用到的 transform 是否都能 build
print()
from mmdet.registry import TRANSFORMS
print('pipeline steps:', len(cfg.train_pipeline))
bad = []
for step in cfg.train_pipeline:
    t = step.get('type')
    try:
        TRANSFORMS.get(t)
    except Exception:
        bad.append(t)
print('unbuildable:', bad or 'none')

print()
print('test_pipeline:')
for step in cfg.test_pipeline:
    print('   ', step.get('type'))
print('test_cfg:', cfg.model.test_cfg)
print('num_classes:', cfg.model.bbox_head.num_classes)
print('input_size:', cfg.model.data_preprocessor.get('size'))
