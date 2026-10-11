# -*- coding: utf-8 -*-
"""迁移后环境自检 —— 把「会不会报错」提前 10 秒暴露出来。

用途：在新机器（如 RTX 4060 笔记本）重建环境后，**跑训练前**先跑这个。
它覆盖 7 类最常见的迁移失败点：

  1. Python 版本（必须 3.10.x，否则 mmcv 的 cp310 编译扩展不能用）
  2. GPU / CUDA / sm 架构（torch 编译列表是否含目标 GPU 的 sm）
  3. mmcv 编译扩展能否导入（_ext.cp310-win_amd64.pyd）
  4. mmpose 是否以 editable 方式指向**本项目源码**（而非旧机器路径）
  5. pkg_resources 是否存在（setuptools 必须 < 81）
  6. mmdet junction 是否可解析
  7. 配置能否加载、模型能否构建、数据管线能否真正跑通一个样本

退出码：0 = 全通过（可以训练）；1 = 有失败项（先修好）。
"""
import os
import platform
import sys

OK = True


def chk(name, cond, detail=''):
    global OK
    OK &= bool(cond)
    tag = 'OK  ' if cond else 'FAIL'
    print(f'  [{tag}] {name}' + (f'  ->  {detail}' if detail else ''))


def sep(t):
    print()
    print('=' * 78)
    print(t)
    print('=' * 78)


# ------------------------------------------------------------------ #
sep('1. 基础环境')
chk('Python 3.10.x（mmcv 的 cp310 编译扩展要求）',
    sys.version.startswith('3.10'), sys.version.split()[0])
chk('Windows x64', platform.machine() == 'AMD64', platform.machine())
print(f'       解释器路径 {sys.executable}')

import torch
chk('torch 可导入', True, torch.__version__)
cuda_ok = torch.cuda.is_available()
chk('CUDA 可用', cuda_ok,
    torch.cuda.get_device_name(0) if cuda_ok else 'NO CUDA -> 检查是否装了 CPU 版 torch')
if cuda_ok:
    sm = torch.cuda.get_device_capability(0)
    arch = torch.cuda.get_arch_list()
    tag = f'sm_{sm[0]}{sm[1]}'
    chk(f'GPU 计算能力 {tag} 在 torch 编译列表内',
        tag in arch,
        f'arch_list={arch}')
    if tag not in arch:
        print('       注：缺 sm 不代表不能跑 —— CUDA 的 binary compatibility')
        print('           允许同 major 内低 minor 的 cubin 在高 minor 上运行。')
        print('           只是少了架构专属优化，正确性无问题。')
    print(f'       cudnn {torch.backends.cudnn.version()}')

# ------------------------------------------------------------------ #
sep('2. 框架版本')
import mmcv
import mmengine
import mmdet
chk('mmcv == 2.1.0', mmcv.__version__ == '2.1.0', mmcv.__version__)
chk('mmengine == 0.10.7', mmengine.__version__ == '0.10.7', mmengine.__version__)
chk('mmdet == 3.2.0', mmdet.__version__ == '3.2.0', mmdet.__version__)

try:
    import mmcv.ops  # 编译扩展
    chk('mmcv.ops（编译扩展）可导入', True)
except Exception as e:
    chk('mmcv.ops（编译扩展）可导入', False, str(e)[:100])
    print('       修法：确认 Python 是 3.10.x；否则 mmcv 的 .pyd 与解释器不符')

import mmpose
chk('mmpose == 1.3.2', mmpose.__version__ == '1.3.2', mmpose.__version__)
mp = mmpose.__file__.replace('\\', '/')
chk('mmpose 指向**本项目**源码（而非旧机器路径）',
    'open-mmlab/mmpose' in mp, mp)

try:
    import pkg_resources
    chk('pkg_resources 可导入（setuptools < 81）', True)
except Exception as e:
    chk('pkg_resources 可导入（setuptools < 81）', False, str(e)[:80])
    print('       修法：pip install "setuptools==80.9.0"')

chk('setuptools < 81',
    tuple(int(v) for v in __import__('setuptools').__version__.split('.')[:2]) < (81, 0),
    __import__('setuptools').__version__)

# ------------------------------------------------------------------ #
sep('3. 自定义数据集模块')
try:
    from mmpose.datasets.datasets.DM.registry import DMKPD_dataset  # noqa
    chk('DMKPD_dataset 可导入', True)
except Exception as e:
    chk('DMKPD_dataset 可导入', False, str(e)[:120])

try:
    from mmpose.datasets.datasets.DM.transforms import (  # noqa
        CheckKeypointOutOfBounds)
    chk('CheckKeypointOutOfBounds 可导入', True)
except Exception as e:
    chk('CheckKeypointOutOfBounds 可导入', False, str(e)[:120])

# ------------------------------------------------------------------ #
sep('4. 数据与权重文件')
FILES = [
    ('划分清单 train', 'DMKPD_dataset/splits/train.txt'),
    ('划分清单 val', 'DMKPD_dataset/splits/val.txt'),
    ('检测标注 train', 'det_dataset/annotations/instances_train.json'),
    ('检测标注 val', 'det_dataset/annotations/instances_val.json'),
    ('检测标注 test', 'det_dataset/annotations/instances_test.json'),
    ('姿态元信息', 'open-mmlab/mmpose/configs/_base_/datasets/DMKPD_meta.py'),
    ('姿态预训练权重',
     'modelbase_test/td-hm_hrnet-w48_8xb32-210e_coco-'
     '256x192-0e67c616_20220913.pth'),
    ('检测预训练权重',
     'modelbase_test/rtmdet_tiny_8xb32-300e_coco_'
     '20220902_112414-78e30dcc.pth'),
]
for name, p in FILES:
    chk(name, os.path.isfile(p), p if not os.path.isfile(p) else '')

chk('mmdet junction 可解析',
    os.path.isfile('mmdet/.mim/configs/rtmdet/rtmdet_tiny_8xb32-300e_coco.py'))
print('       注：junction 用 `New-Item -ItemType Junction -Path mmdet '
      '-Target "$PWD\\.venv\\Lib\\site-packages\\mmdet"` 创建')

# 图片抽查（真正读一张）
try:
    import cv2
    n_img = len([f for f in os.listdir('DMKPD_dataset/train/images')
                 if f.endswith('.jpg')])
    im = cv2.imread('DMKPD_dataset/train/images/00003.jpg')
    chk('训练图片可读', im is not None,
        f'{n_img} 张，抽查 shape={None if im is None else im.shape}')
    n_test = len([f for f in os.listdir('DMKPD_dataset/test')
                  if f.endswith('.jpg')])
    chk('测试图片', n_test == 300, f'{n_test} 张')
except Exception as e:
    chk('训练图片可读', False, str(e)[:100])

# ------------------------------------------------------------------ #
sep('5. 配置加载 + 模型构建')
from mmengine.config import Config
from mmengine.registry import init_default_scope

try:
    cfg = Config.fromfile('config.py')
    init_default_scope('mmpose')
    from mmpose.registry import MODELS as PM
    m = PM.build(cfg.model)
    n = sum(p.numel() for p in m.parameters()) / 1e6
    chk('姿态模型可构建（HRNet-W48，期望 ~63.6M）',
        abs(n - 63.60) < 0.6, f'{n:.2f}M 参数')
    print(f'       输入 {cfg.input_size} / 热力图 {cfg.heatmap_size} / '
          f'输出通道 {cfg.model["head"]["out_channels"]}')
except Exception as e:
    chk('姿态模型可构建', False, str(e)[:150])

cfgd = None
try:
    cfgd = Config.fromfile('config_det.py')
    init_default_scope('mmdet')
    from mmdet.registry import MODELS as DM
    md = DM.build(cfgd.model)
    n = sum(p.numel() for p in md.parameters()) / 1e6
    chk('检测模型可构建（RTMDet-tiny，期望 ~4.87M）',
        abs(n - 4.87) < 0.25, f'{n:.2f}M 参数')
    print(f'       输入 {cfgd.train_pipeline[2]["scale"]} / '
          f'类别数 {cfgd.model["bbox_head"]["num_classes"]}')
except Exception as e:
    chk('检测模型可构建', False, str(e)[:150])

# ------------------------------------------------------------------ #
sep('6. 数据管线真正前向（读图 -> 变换 -> pack）')
if cfgd is not None:
    try:
        from mmengine.dataset import Compose
        from mmdet.registry import DATASETS as DD
        init_default_scope('mmdet')
        ds = DD.build(dict(
            type='CocoDataset', metainfo=dict(classes=('DPM', )),
            data_root='.', data_prefix=dict(img=''),
            ann_file='det_dataset/annotations/instances_val.json',
            pipeline=cfgd.test_pipeline, test_mode=True,
            serialize_data=False))
        r = Compose(cfgd.test_pipeline)(ds.get_data_info(0))
        shp = tuple(r['inputs'].shape)
        chk('检测管线前向', shp == (3, 1024, 1024), f'inputs shape={shp}')
        g = r['data_samples'].gt_instances
        chk('标注解析正确', len(g.bboxes) > 0,
            f'{len(g.bboxes)} 个框，labels={g.labels.tolist()[:5]}')
        sf = r['data_samples'].metainfo['scale_factor']
        chk('scale_factor 合理', 0.4 < sf[0] < 0.9,
            f'{tuple(round(v, 4) for v in sf)}（1702 -> 1024 应约 0.6016）')
    except Exception as e:
        chk('检测管线前向', False, str(e)[:150])

try:
    from mmengine.dataset import Compose as C2
    from mmpose.registry import DATASETS as PD
    init_default_scope('mmpose')
    ds2 = PD.build(dict(
        type='DMKPD_dataset', data_root='DMKPD_dataset',
        ann_file='splits/val.txt', data_prefix=dict(img='train/images'),
        metainfo=cfg.metainfo, serialize_data=False))
    chk('姿态数据集可构建', len(ds2) > 0, f'{len(ds2)} 个实例')
    r2 = C2(cfg.val_pipeline)(ds2.get_data_info(0))
    shp = tuple(r2['inputs'].shape)
    chk('姿态管线前向', shp == (3, 256, 256), f'inputs shape={shp}')
except Exception as e:
    chk('姿态数据集/管线', False, str(e)[:150])

# ------------------------------------------------------------------ #
sep('7. GPU 冒烟（前向 + 反向，验证 cudnn/算子在目标卡上真能跑）')
if cuda_ok:
    try:
        import time
        init_default_scope('mmpose')
        from mmpose.registry import MODELS as PM2
        mm = PM2.build(cfg.model).cuda().train()
        torch.cuda.reset_peak_memory_stats()
        x = torch.randn(2, 3, 256, 256, device='cuda')
        t0 = time.time()
        for _ in range(3):
            out = mm.extract_feat(x)
            loss = sum(o.float().pow(2).mean() for o in out
                       if torch.is_tensor(o))
            loss.backward()
        torch.cuda.synchronize()
        dt = (time.time() - t0) / 3
        peak = torch.cuda.max_memory_allocated() / 1024**3
        chk('姿态模型 GPU 前反向可跑', True,
            f'{dt*1000:.0f} ms/iter(bs=2), 峰值 {peak:.2f} GB')
        print(f'       GPU {torch.cuda.get_device_name(0)}  '
              f'显存 {torch.cuda.get_device_properties(0).total_memory/1024**3:.1f} GB')
    except Exception as e:
        chk('姿态模型 GPU 前反向可跑', False, str(e)[:150])
else:
    print('  [跳过] 无 CUDA')

# ------------------------------------------------------------------ #
sep('结论')
if OK:
    print('  全部通过 —— 环境就绪，可以开始训练。')
    print()
    print('  启动命令：')
    print('    # 先测显存/速度（可选）')
    print('    & ".\\\\.venv\\\\Scripts\\\\python.exe" _det_bench2.py')
    print('    # 训练检测器')
    print('    & ".\\\\.venv\\\\Scripts\\\\python.exe" run_train_det.py '
          '--epochs 24 --val-interval 3 --no-resume')
else:
    print('  有 FAIL 项 —— 先按上面的提示修好再训练！')
    print('  常见修法见 MIGRATE_TO_4060.md 的「常见报错对照表」。')
sys.exit(0 if OK else 1)
