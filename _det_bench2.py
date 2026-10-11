# -*- coding: utf-8 -*-
"""RTMDet-tiny 真实训练速度实测（带预热，用 model.train_step）。

上一次测的 5-9s/iter 明显被 cuDNN autotune 和重复 build 污染了，
这里改成：build 一次 -> 预热 5 iter -> 计 20 iter 的平均。
"""
import os
import time

import torch
from mmengine.config import Config
from mmengine.registry import init_default_scope
from mmengine.structures import InstanceData
from mmdet.structures import DetDataSample
import mmdet  # noqa

CFG = os.path.join(os.path.dirname(mmdet.__file__), '.mim', 'configs',
                   'rtmdet', 'rtmdet_tiny_8xb32-300e_coco.py')
cfg = Config.fromfile(CFG)
cfg.model.bbox_head.num_classes = 1
init_default_scope('mmdet')
from mmdet.registry import MODELS
from mmengine.optim import build_optim_wrapper

dev = 'cuda'
torch.backends.cudnn.benchmark = True


def make_batch(bs, S, n_gt=3):
    x = torch.randn(bs, 3, S, S, device=dev)
    samples = []
    for b in range(bs):
        s = DetDataSample()
        s.set_metainfo(dict(batch_input_shape=(S, S), img_shape=(S, S),
                            ori_shape=(S, S), pad_shape=(S, S),
                            scale_factor=(1.0, 1.0)))
        g = InstanceData()
        cx = torch.rand(n_gt) * S * 0.6 + S * 0.2
        cy = torch.rand(n_gt) * S * 0.6 + S * 0.2
        w = torch.rand(n_gt) * S * 0.08 + S * 0.04
        h = w * (0.9 + torch.rand(n_gt) * 0.2)
        g.bboxes = torch.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2],
                               1)
        g.labels = torch.zeros(n_gt, dtype=torch.long)
        s.gt_instances = g
        samples.append(s)
    return dict(inputs=list(x), data_samples=samples)


print('=' * 84)
print('RTMDet-tiny 真实训练速度（预热 5 iter + 计 20 iter）')
print('=' * 84)
print(f'{"输入":>12}{"bs":>4}{"峰值显存":>11}{"s/iter":>10}{"iter/s":>9}'
      f'{"epoch(9110图)":>14}')

for S, bs in [(640, 4), (768, 4), (1024, 2), (1024, 4), (1280, 2)]:
    try:
        model = MODELS.build(cfg.model).to(dev).train()
        ow = build_optim_wrapper(
            model, dict(optimizer=dict(type='AdamW', lr=0.001)))
        data = make_batch(bs, S)
        # 预热
        for _ in range(5):
            ow.zero_grad()
            model.train_step(data, ow)
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        t0 = time.time()
        for _ in range(20):
            ow.zero_grad()
            model.train_step(data, ow)
        torch.cuda.synchronize()
        dt = (time.time() - t0) / 20
        peak = torch.cuda.max_memory_allocated() / 1024**3
        ep = dt * (9110 / bs) / 60
        print(f'{str(S)+"x"+str(S):>12}{bs:>4}{peak:>10.2f}G{dt:>10.3f}'
              f'{1/dt:>9.2f}{ep:>12.1f}min')
    except RuntimeError as e:
        msg = 'OOM' if 'out of memory' in str(e).lower() else str(e)[:40]
        print(f'{str(S)+"x"+str(S):>12}{bs:>4}{"":>11}{"":>10}{"":>9}'
              f'{"  " + msg:>14}')
    finally:
        try:
            del model
        except Exception:
            pass
        torch.cuda.empty_cache()
