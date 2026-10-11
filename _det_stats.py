# -*- coding: utf-8 -*-
"""检测器输入尺寸的设计依据：按图片来源分组统计码的绝对/相对尺寸。

决定性问题：测试集 300 张里 299 张是 1702x1276，而训练集同尺寸的只有
2659/9591。所以必须分开看「1702 档」和「512 档」的码尺寸分布，才能定
输入分辨率，并且判断两档能否混合训练。
"""
import json
import os
import os.path as osp

import numpy as np

C = 'det_dataset/annotations/instances_train.json'
d = json.load(open(C, encoding='utf-8'))
imgs = {im['id']: im for im in d['images']}

groups = {}
for a in d['annotations']:
    im = imgs[a['image_id']]
    w, h = im['width'], im['height']
    fn = osp.basename(im['file_name'])
    if fn.startswith('adv'):
        src = 'adv_final (512)'
    else:
        src = 'numeric (1702x1276)' if w > 1400 else f'other'
    g = groups.setdefault(src, dict(n=0, side=[], rel=[], per_img={},
                                    sizes=set()))
    g['n'] += 1
    bw, bh = a['bbox'][2], a['bbox'][3]
    g['side'].append((bw * bh) ** 0.5)
    g['rel'].append(((bw * bh) ** 0.5) / min(w, h))
    g['sizes'].add((w, h))
    g['per_img'][a['image_id']] = g['per_img'].get(a['image_id'], 0) + 1

print('=' * 88)
print('按来源分组：码的边长 sqrt(area) 统计（像素）')
print('=' * 88)
print(f'{"来源":<22}{"实例数":>8}{"min":>8}{"p05":>8}{"p25":>8}{"中位":>8}'
      f'{"p75":>8}{"p95":>8}{"max":>9}')
for k, g in sorted(groups.items()):
    s = np.array(g['side'])
    print(f'{k:<22}{g["n"]:>8}{s.min():>8.1f}'
          + ''.join(f'{np.percentile(s,q):>8.1f}'
                    for q in (5, 25, 50, 75, 95))
          + f'{s.max():>9.1f}')

print()
print('=' * 88)
print('按来源分组：码的边长 / 图短边（相对尺寸，无量纲）')
print('=' * 88)
print(f'{"来源":<22}{"实例数":>8}{"中位":>9}{"p05":>9}{"p95":>9}'
      f'{"图尺寸集合":>34}')
for k, g in sorted(groups.items()):
    r = np.array(g['rel'])
    szs = sorted(g['sizes'])
    print(f'{k:<22}{g["n"]:>8}{100*np.median(r):>8.2f}%'
          f'{100*np.percentile(r,5):>8.2f}%{100*np.percentile(r,95):>8.2f}%'
          f'{str(szs[:4]):>34}')

print()
print('=' * 88)
print('关键推论：如果统一 resize 到 S x S，码在输入上会有多大（像素）')
print('=' * 88)
print(f'{"来源":<22}' + ''.join(f'{"S=" + str(s):>14}' for s in (512, 640,
                                                                 768, 1024)))
for k, g in sorted(groups.items()):
    s = np.array(g['side'])
    row = f'{k:<22}'
    for S in (512, 640, 768, 1024):
        # 按短边等比缩放
        sc = min(S / min(x) for x in g['sizes'])
        row += f'{np.median(s)*sc:>14.1f}'
    print(row)
print()
print('（上表 = 中位码长在缩放后的像素数；RTMDet stride=8，'
      '码长 <16px 时只剩 2 个特征格，检测极难）')

# 每图实例数（1702 档）
print()
print('=' * 88)
print('每图实例数（1702 档）')
print('=' * 88)
for k, g in sorted(groups.items()):
    cnt = list(g['per_img'].values())
    print(f'{k:<22} 图数={len(cnt):>5}  平均={np.mean(cnt):.2f}  '
          f'最多={max(cnt)}  分布={dict(sorted(__import__("collections").Counter(cnt).items()))}')
