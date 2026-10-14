# -*- coding: utf-8 -*-
"""实测 `unbiased=True` 对端到端 AP 的收益（无需重训）。

===================================================================
做法
===================================================================
`MSRAHeatmap` 的 `unbiased` 会同时影响 encode 与 decode：
  encode: generate_gaussian_heatmaps -> generate_unbiased_gaussian_heatmaps
  decode: refine_keypoints         -> refine_keypoints_dark

**推理时 encode 根本不会被调用**（不需要造 target 热力图）。
所以「把 head 的 decoder 换成 `unbiased=True` 的实例」就等价于
「不重训、只换解码」。

❌ 不要手动跑 backbone+head 再自己映射坐标：热力图坐标 -> 画布坐标
   -> 原图坐标 需要 TopdownAffine 的逆变换矩阵，手写极易出错。
   （我第一版就是这么写的，结果 AP 全是 0。）
✅ 直接换 decoder，走 `model.test_step()`，完全复用 mmpose 的还原逻辑。

用法：
    python _codec_ap.py --limit 200
"""
import argparse
import glob
import time
from collections import defaultdict

import cv2
import numpy as np
from mmengine.config import Config
from mmengine.registry import init_default_scope
from mmengine.runner import load_checkpoint

import mmdet  # noqa
import mmpose  # noqa
from mmdet.apis import inference_detector
from mmdet.registry import MODELS as DET_MODELS
from mmpose.registry import KEYPOINT_CODECS
from mmpose.registry import MODELS as POSE_MODELS

from _ap_holdout import PoseOnBoxes, coco_ap  # noqa


def swap_decoder(model, unbiased, blur_kernel_size=11):
    """把 head.decoder 换成相同参数但 `unbiased` 不同的新实例。"""
    d = model.head.decoder
    cfg = dict(
        type='MSRAHeatmap',
        input_size=tuple(int(v) for v in np.asarray(d.input_size).tolist()),
        heatmap_size=tuple(int(v)
                           for v in np.asarray(d.heatmap_size).tolist()),
        sigma=float(d.sigma),
        unbiased=bool(unbiased),
        blur_kernel_size=int(blur_kernel_size))
    model.head.decoder = KEYPOINT_CODECS.build(cfg)
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pose-config', default='config.py')
    ap.add_argument('--pose-ckpt', default=None,
                    help='默认自动找 work_dirs/dmkpd/best_coco_AP*.pth 中'
                         'mtime 最新的')
    ap.add_argument('--det-config', default='config_det.py')
    ap.add_argument('--det-ckpt', default=None)
    ap.add_argument('--split', default='splits/holdout1702.txt')
    ap.add_argument('--score-thr', type=float, default=0.30)
    ap.add_argument('--limit', type=int, default=200)
    args = ap.parse_args()

    if args.det_ckpt is None:
        c = sorted(glob.glob('work_dirs/dmkpd_det/best_coco_bbox_mAP*.pth'))
        args.det_ckpt = c[-1]
    if args.pose_ckpt is None:
        c = glob.glob('work_dirs/dmkpd/best_coco_AP*.pth')
        args.pose_ckpt = max(c, key=osp.getmtime) if c else None
    if not args.pose_ckpt or not osp.isfile(args.pose_ckpt):
        raise SystemExit(f'找不到姿态权重：{args.pose_ckpt}')

    if args.det_ckpt is None:
        c = sorted(glob.glob('work_dirs/dmkpd_det/best_coco_bbox_mAP*.pth'))
        args.det_ckpt = c[-1]

    print('=' * 78)
    print('解码精修对比（同一份权重，只换 head.decoder）')
    print('=' * 78)

    cfg_p = Config.fromfile(args.pose_config)

    models, runners = {}, {}
    for mode, ub in (('taylor', False), ('dark', True)):
        init_default_scope('mmpose')
        m = POSE_MODELS.build(cfg_p.model).cuda().eval()
        load_checkpoint(m, args.pose_ckpt, map_location='cpu', strict=False)
        m.cfg = cfg_p
        swap_decoder(m, unbiased=ub, blur_kernel_size=11)
        models[mode] = m
        runners[mode] = PoseOnBoxes(m, cfg_p)
        print(f'  {mode:<8} unbiased={ub}  '
              f'decoder={type(m.head.decoder).__name__}')

    cfg_d = Config.fromfile(args.det_config)
    init_default_scope('mmdet')
    det = DET_MODELS.build(cfg_d.model).cuda().eval()
    load_checkpoint(det, args.det_ckpt, map_location='cpu', strict=False)
    det.cfg = cfg_d

    from mmpose.datasets.datasets.DM.registry import DMKPD_dataset
    ds = DMKPD_dataset(
        data_root='DMKPD_dataset', ann_file=args.split,
        data_prefix=dict(img='train/images'), metainfo=cfg_p.metainfo,
        serialize_data=False)
    by_img = defaultdict(list)
    for i in range(len(ds)):
        info = ds.get_data_info(i)
        by_img[info['img_id']].append((i, info))
    img_ids = sorted(by_img)[:args.limit]
    print(f'  图 {len(img_ids)} 张')

    ends = {'taylor': [], 'dark': []}
    t0 = time.time()
    for k, iid in enumerate(img_ids):
        gts = by_img[iid]
        img = cv2.imread(gts[0][1]['img_path'])
        H, W = img.shape[:2]
        gt_kps = np.stack([np.asarray(g['keypoints'][0], np.float64)
                           for _, g in gts])
        gt_area = np.array([float(g['area']) for _, g in gts])

        init_default_scope('mmdet')
        r = inference_detector(det, img)
        pi = r.pred_instances
        bx = pi.bboxes
        bx = bx.tensor if hasattr(bx, 'tensor') else bx
        bx = bx.detach().cpu().numpy() if hasattr(bx, 'detach') \
            else np.asarray(bx)
        sc = pi.scores
        sc = sc.detach().cpu().numpy() if hasattr(sc, 'detach') \
            else np.asarray(sc)
        if len(bx):
            bx = bx.copy()
            bx[:, [0, 2]] = np.clip(bx[:, [0, 2]], 0, W)
            bx[:, [1, 3]] = np.clip(bx[:, [1, 3]], 0, H)
            w = bx[:, 2] - bx[:, 0]
            h = bx[:, 3] - bx[:, 1]
            keep = (sc >= args.score_thr) & (np.minimum(w, h) >= 12)
            bx, sc = bx[keep], sc[keep]

        for mode in ('taylor', 'dark'):
            kps, scs = runners[mode](img, bx)
            conf = sc * scs if len(bx) else np.zeros(0)
            ends[mode].append(dict(pred_kps=kps, pred_conf=conf,
                                   gt_kps=gt_kps, gt_area=gt_area))
        if (k + 1) % 50 == 0:
            el = time.time() - t0
            print(f'  [{k+1}/{len(img_ids)}] {el:.0f}s  '
                  f'ETA {el/(k+1)*(len(img_ids)-k-1):.0f}s')

    print()
    print('=' * 78)
    print('端到端 AP 对比（COCO 101 点插值）')
    print('=' * 78)
    res = {}
    for mode in ('taylor', 'dark'):
        res[mode] = coco_ap(ends[mode])
        r = res[mode]
        print(f'  {mode:<8}AP_50_95 {r["AP"]*100:>7.4f}   '
              f'AP_50 {r["AP50"]*100:>7.4f}   AP_75 {r["AP75"]*100:>7.4f}')
    t, d = res['taylor'], res['dark']
    print()
    print(f'  {"收益":<8}AP_50_95 {(d["AP"]-t["AP"])*100:>+7.4f}   '
          f'AP_50 {(d["AP50"]-t["AP50"])*100:>+7.4f}   '
          f'AP_75 {(d["AP75"]-t["AP75"])*100:>+7.4f}')
    print()
    print('  各阈值明细（端到端）：')
    print(f'  {"thr":>6}{"taylor":>10}{"dark":>10}{"差":>10}')
    for thr in sorted(t['detail'], key=float):
        a, b = t['detail'][thr] * 100, d['detail'][thr] * 100
        print(f'  {float(thr):>6.2f}{a:>10.3f}{b:>10.3f}{b-a:>+10.3f}')


if __name__ == '__main__':
    main()
