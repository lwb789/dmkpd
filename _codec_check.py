# -*- coding: utf-8 -*-
"""对比 `MSRAHeatmap` 解码选项对角点定位误差（EPE）的影响。

===================================================================
背景
===================================================================
`config.py` 用 `MSRAHeatmap(sigma=2)`，**没有开** `unbiased`。
mmpose 的 decode 有两条精修路径：

  `unbiased=False`（当前）→ `refine_keypoints`
      Taylor 展开，把峰值往次大方向挪 0.25 像素
  `unbiased=True`        → `refine_keypoints_dark`
      DarkPose 高斯模糊调制（`blur_kernel_size` 必须匹配热力图 σ）

热力图 64x64，`argmax` 是整数坐标 → 量化误差上限 ±0.5 热力图像素
= ±2 输入像素（画布 256），换回原图还要乘 `1.5*bbox/256`。

===================================================================
做法
===================================================================
✅ 通过**替换 `head.decoder`** 来切换解码，然后正常走
   `model.test_step()` —— 这样坐标还原完全复用 mmpose 的逻辑。
❌ 不要手动跑 backbone+head 自己映射坐标（需要 TopdownAffine 的
   逆变换矩阵，第一版就是这么写的，结果 EPE 完全不对）。

用法：
    python _codec_check.py --split splits/holdout1702.txt --limit 120
"""
import argparse
import time
from collections import defaultdict

import cv2
import numpy as np
from mmengine.config import Config
from mmengine.registry import init_default_scope
from mmengine.runner import load_checkpoint

import mmpose  # noqa
from mmpose.registry import KEYPOINT_CODECS
from mmpose.registry import MODELS as POSE_MODELS

from _ap_holdout import PoseOnBoxes  # noqa

SIGMA = 0.1

MODES = [
    ('taylor(当前)', dict(unbiased=False)),
    ('dark(ks=7)', dict(unbiased=True, blur_kernel_size=7)),
    ('dark(ks=9)', dict(unbiased=True, blur_kernel_size=9)),
    ('dark(ks=11)', dict(unbiased=True, blur_kernel_size=11)),
    ('dark(ks=13)', dict(unbiased=True, blur_kernel_size=13)),
    ('dark(ks=17)', dict(unbiased=True, blur_kernel_size=17)),
]


def swap_decoder(model, **kw):
    d = model.head.decoder
    cfg = dict(
        type='MSRAHeatmap',
        input_size=tuple(int(v) for v in np.asarray(d.input_size).tolist()),
        heatmap_size=tuple(int(v)
                           for v in np.asarray(d.heatmap_size).tolist()),
        sigma=float(d.sigma),
        unbiased=False)
    cfg.update(kw)
    model.head.decoder = KEYPOINT_CODECS.build(cfg)
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pose-config', default='config.py')
    ap.add_argument('--pose-ckpt',
                    default='work_dirs/dmkpd/best_coco_AP_epoch_5.pth')
    ap.add_argument('--split', default='splits/holdout1702.txt')
    ap.add_argument('--limit', type=int, default=120)
    args = ap.parse_args()

    cfg = Config.fromfile(args.pose_config)

    runners = {}
    for name, kw in MODES:
        init_default_scope('mmpose')
        m = POSE_MODELS.build(cfg.model).cuda().eval()
        load_checkpoint(m, args.pose_ckpt, map_location='cpu', strict=False)
        m.cfg = cfg
        swap_decoder(m, **kw)
        runners[name] = PoseOnBoxes(m, cfg)

    from mmpose.datasets.datasets.DM.registry import DMKPD_dataset
    ds = DMKPD_dataset(
        data_root='DMKPD_dataset', ann_file=args.split,
        data_prefix=dict(img='train/images'), metainfo=cfg.metainfo,
        serialize_data=False)
    by_img = defaultdict(list)
    for i in range(len(ds)):
        info = ds.get_data_info(i)
        by_img[info['img_id']].append((i, info))
    img_ids = sorted(by_img)[:args.limit]
    print(f'跑 {len(img_ids)} 张图（GT 框条件，隔离出纯解码影响）')

    epe_px = {n: [] for n, _ in MODES}
    epe_nm = {n: [] for n, _ in MODES}
    t0 = time.time()
    for k, iid in enumerate(img_ids):
        gts = by_img[iid]
        img = cv2.imread(gts[0][1]['img_path'])
        boxes = np.stack([np.asarray(g['bbox'][0], np.float64)
                          for _, g in gts])
        gt_kp = np.stack([np.asarray(g['keypoints'][0], np.float64)
                          for _, g in gts])
        areas = np.array([float(g['area']) for _, g in gts])
        for name, runner in runners.items():
            pred, _ = runner(img, boxes)
            d = np.linalg.norm(pred - gt_kp, axis=2).ravel()   # (N*4,)
            for i in range(len(boxes)):
                s = np.sqrt(areas[i])
                dd = np.linalg.norm(pred[i] - gt_kp[i], axis=1)
                epe_px[name].extend(dd.tolist())
                epe_nm[name].extend((dd / s).tolist())
        if (k + 1) % 30 == 0:
            el = time.time() - t0
            print(f'  [{k+1}/{len(img_ids)}] {el:.0f}s')

    print()
    print('=' * 84)
    print('角点定位误差（GT 框条件，纯解码差异）')
    print('=' * 84)
    print(f'{"解码方式":<16}{"EPE(px)均值":>12}{"EPE(px)中位":>13}'
          f'{"归一化均值":>12}{"OKS代理":>10}')
    for name, _ in MODES:
        px = np.asarray(epe_px[name])
        nm = np.asarray(epe_nm[name])
        eff = float(np.mean(np.exp(-nm ** 2 / (2 * SIGMA ** 2))))
        print(f'{name:<16}{px.mean():>12.2f}{np.median(px):>13.2f}'
              f'{nm.mean():>12.4f}{eff:>10.4f}')
    print()
    base_px = np.asarray(epe_px['taylor(当前)'])
    base_nm = np.asarray(epe_nm['taylor(当前)'])
    base_eff = np.mean(np.exp(-base_nm ** 2 / (2 * SIGMA ** 2)))
    print('  相对 taylor(当前) 的变化：')
    for name, _ in MODES:
        if name == 'taylor(当前)':
            continue
        px = np.asarray(epe_px[name])
        nm = np.asarray(epe_nm[name])
        eff = np.mean(np.exp(-nm ** 2 / (2 * SIGMA ** 2)))
        print(f'    {name:<16} EPE中位 {np.median(px)-np.median(base_px):+8.2f}px'
              f'   归一化 {nm.mean()-base_nm.mean():+.5f}'
              f'   OKS代理 {eff-base_eff:+.5f}')
    print()
    print('  ⚠️ OKS代理只是「exp(-e^2/2σ²) 的均值」外推，')
    print('     真实 AP 还要经过贪心匹配 + 101 点插值 —— 见 `_codec_ap.py` 的实测。')


if __name__ == '__main__':
    main()
