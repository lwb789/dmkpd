# -*- coding: utf-8 -*-
"""RTMDet-tiny 在 GTX 1650 (4GB) 上的显存/速度实测，用于选输入分辨率。

只测 fwd+bwd 的峰值显存（训练时的主要开销），不跑完整训练。
"""
import os
import time

import torch
from mmengine.config import Config
from mmengine.registry import init_default_scope
import mmdet  # noqa

CFG = os.path.join(os.path.dirname(mmdet.__file__), '.mim', 'configs',
                   'rtmdet', 'rtmdet_tiny_8xb32-300e_coco.py')
cfg = Config.fromfile(CFG)
cfg.model.bbox_head.num_classes = 1
# 关掉 AMP 以外的花活，纯粹测算力
init_default_scope('mmdet')
from mmdet.registry import MODELS

dev = 'cuda'
torch.backends.cudnn.benchmark = True

print('=' * 84)
print('RTMDet-tiny fwd+bwd 显存与耗时（1 类，输入按测试集长宽比 pad 到 S）')
print('=' * 84)
print(f'{"输入(HxW)":>14}{"bs":>5}{"峰值显存":>12}{"耗时/iter":>12}{"是否OOM":>10}')

for S in (640, 768, 1024):
    # 1702x1276 -> 短边缩放到 S，pad 成 S x S（就是 mmdet 的 Resize+Pad）
    for bs in (1, 2, 4):
        model = MODELS.build(cfg.model).to(dev).train()
        try:
            x = torch.randn(bs, 3, S, S, device=dev)
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            t0 = time.time()
            losses = model.backbone(x)
            # 直接调用 loss 需要 batch_data_samples，这里只近似测到 neck+head
            # 用一次完整前向 + 假 loss 反向
            feat = model.neck(losses)
            out = model.bbox_head(feat)
            loss = sum(o[0].sum() + o[1].sum() + o[2].sum()
                       for o in out if isinstance(o, tuple))
            loss.backward()
            torch.cuda.synchronize()
            dt = time.time() - t0
            peak = torch.cuda.max_memory_allocated() / 1024**3
            print(f'{str(S)+"x"+str(S):>14}{bs:>5}{peak:>11.2f}G'
                  f'{dt:>11.2f}s{"":>10}')
        except RuntimeError as e:
            if 'out of memory' in str(e).lower():
                print(f'{str(S)+"x"+str(S):>14}{bs:>5}{"-":>12}{"-":>12}'
                      f'{"OOM":>10}')
            else:
                print('  ERR', e)
        finally:
            del model
            torch.cuda.empty_cache()
