# -*- coding: utf-8 -*-
"""判决实验：本地评测是否漏掉了「点序归一化」？

===================================================================
发现过程
===================================================================
`_ar_miss_detail.py` 打印出 17 个「最佳 OKS < 0.5」的实例，
其中 00086 的四点距离是 584.6 / 576.1 / 573.0 / 566.3 —— **四边几乎等长**，
这在"定位错了"的情况下极不自然。手工比对坐标发现：

    GT  P0 (575.3, 277.7)   pred P1 (577.8, 268.8)   ~10px
    GT  P1 (1026.7,625.0)   pred P2 (1032.2,632.3)   ~9px
    GT  P2 (630.7,1031.8)   pred P3 (618.2,1046.3)   ~18px
    GT  P3 (164.5, 691.2)   pred P0 (163.8, 692.9)   ~2px

=> **模型找对了 4 个角点，只是点序循环移位了！**（pred[i] ≈ GT[i+1]）

===================================================================
这意味着什么
===================================================================
比赛说明原文：「推荐顺序 TL→TR→BR→BL，**但评测脚本会 normalize 点顺序**」
=> 评测方是**按点集**比对的，点序不影响得分。

而我的 `_ap_holdout.py` 里的 `oks_matrix` 是**逐点直接求距离**，
**没有做任何点序归一化** —— 所以把这类"顺序错但位置对"的实例
全部误判成了漏检（OKS≈0）。

本地 AP_50_95=92.52 / AR=93.95，榜单是 97.67 / 98.35，
差距 ~4.4 点，而这类实例恰好占 3.34% —— **数量级吻合**。

===================================================================
本脚本做什么
===================================================================
对同一批预测，用三种匹配方式分别算 AP/AR：
  ① 直接逐点（当前实现，可能低估）
  ② 4 种循环移位取最优（对应"起点不同"）
  ③ 8 种（4 移位 × 2 方向，对应"起点+绕向不同"）
  ④ 24 种全排列（对应严格的"点集"语义）

哪一档与榜单吻合，就说明评测方的口径是那一档。

用法：
    python _codec_ap.py 无关；直接：
    python _order_test.py
"""
import glob
import json
import os
import os.path as osp
from collections import defaultdict
from itertools import permutations

import cv2
import numpy as np
from mmengine.config import Config
from mmengine.registry import init_default_scope
from mmengine.runner import load_checkpoint

import mmdet  # noqa
import mmpose  # noqa
from mmdet.registry import MODELS as DET_MODELS
from mmpose.registry import MODELS as POSE_MODELS

from _ar_analyze import coco_ar
from _ap_holdout import THRS, ap_from_oks, coco_ap, PoseOnBoxes, SIGMA

CACHE = '_ar_det_cache.json'
COMP_THR = 0.30

# ---- 四种点序对齐方式 ---- #
CYCLIC4 = [list(np.roll(np.arange(4), s)) for s in range(4)]
CYCLIC8 = CYCLIC4 + [list(np.roll(np.arange(4)[::-1], s))
                     for s in range(4)]
PERM24 = [list(p) for p in permutations(range(4))]


def oks_aligned(pred, gt, area, perms):
    """按给定排列集合取 OKS 最优。

    pred: (P,4,2)  gt: (G,4,2)  area: (G,)
    perms: list[list[int]]，每个是"pred 的第 i 点对应 gt 的第 perms[i] 点"
    """
    P, G = len(pred), len(gt)
    if P == 0 or G == 0:
        return np.zeros((P, G))
    best = np.zeros((P, G))
    for pm in perms:
        d2 = ((pred[:, None, np.asarray(pm), :]
               - gt[None, :, :, :]) ** 2).sum(-1)      # (P,G,4)
        oks = np.exp(-d2 / (2 * SIGMA ** 2 * area[None, :, None])).mean(-1)
        best = np.maximum(best, oks)
    return best


def measure(ends, perms, tag):
    """用指定对齐方式算 AP / AR。"""
    per_img = []
    for e in ends:
        keep = e['bs'] >= COMP_THR
        if len(e['bx']):
            w = e['bx'][:, 2] - e['bx'][:, 0]
            h = e['bx'][:, 3] - e['bx'][:, 1]
            keep &= (np.minimum(w, h) >= 12.0)
        conf = (e['bs'] * e['ksc'])[keep]
        m = conf >= COMP_THR
        per_img.append(dict(pred_kps=e['kps'][keep][m], pred_conf=conf[m],
                            gt_kps=e['gt_kps'], gt_area=e['gt_area']))
    # 替换 oks_matrix 的行为：这里直接手算
    n_gt = sum(len(e['gt_kps']) for e in per_img)
    aps, ars = [], []
    for thr in THRS:
        S, T, F = [], [], []
        for e in per_img:
            if len(e['gt_kps']) == 0 or len(e['pred_kps']) == 0:
                continue
            ok = oks_aligned(e['pred_kps'], e['gt_kps'], e['gt_area'], perms)
            sc, tp, fp = ap_from_oks(ok, e['pred_conf'], thr)
            if len(sc) == 0:
                continue
            if len(sc) > 100:
                k = np.argsort(-sc)[:100]
                sc, tp, fp = sc[k], tp[k], fp[k]
            S.append(sc); T.append(tp); F.append(fp)
        if not S:
            aps.append(0.0); ars.append(0.0)
            continue
        S = np.concatenate(S); T = np.concatenate(T); F = np.concatenate(F)
        o = np.argsort(-S)
        tpc = np.cumsum(T[o]); fpc = np.cumsum(F[o])
        rec = tpc / max(n_gt, 1)
        prec = tpc / np.maximum(tpc + fpc, 1e-9)
        q = np.linspace(0, 1, 101)
        idx = np.searchsorted(rec, q, side='left')
        p_int = np.zeros(101)
        ok_ = idx < len(prec)
        p_int[ok_] = prec[idx[ok_]]
        p_int = np.maximum.accumulate(p_int[::-1])[::-1]
        aps.append(float(p_int.mean()))
        ars.append(float(rec[-1]))
    aps = np.array(aps); ars = np.array(ars)
    return dict(AP=float(np.nanmean(aps)), AP50=float(aps[0]),
                AP75=float(aps[5]), AR=float(np.nanmean(ars)),
                detail=aps, ar_detail=ars)


IDENT = [[0, 1, 2, 3]]


def main():
    cfgd = Config.fromfile('config_det.py')
    init_default_scope('mmdet')
    det = DET_MODELS.build(cfgd.model).cuda().eval()
    load_checkpoint(det, sorted(glob.glob(
        'work_dirs/dmkpd_det/best_coco_bbox_mAP*.pth'))[-1],
        map_location='cpu', strict=False)
    det.cfg = cfgd
    cfgp = Config.fromfile('config.py')
    init_default_scope('mmpose')
    pose = POSE_MODELS.build(cfgp.model).cuda().eval()
    load_checkpoint(pose, max(glob.glob('work_dirs/dmkpd/best_coco_AP*.pth'),
                              key=osp.getmtime), map_location='cpu',
                    strict=False)
    pose.cfg = cfgp
    runner = PoseOnBoxes(pose, cfgp)

    from mmpose.datasets.datasets.DM.registry import DMKPD_dataset
    ds = DMKPD_dataset(data_root='DMKPD_dataset',
                       ann_file='splits/holdout1702.txt',
                       data_prefix=dict(train='train/images'),
                       metainfo=cfgp.metainfo, serialize_data=False) \
        if False else DMKPD_dataset(
            data_root='DMKPD_dataset', ann_file='splits/holdout1702.txt',
            data_prefix=dict(img='train/images'), metainfo=cfgp.metainfo,
            serialize_data=False)
    by_img = defaultdict(list)
    for i in range(len(ds)):
        info = ds.get_data_info(i)
        by_img[info['img_id']].append(info)

    det_map = json.load(open(CACHE, encoding='utf-8'))
    ends = []
    for iid in sorted(by_img):
        gts = by_img[iid]
        img = cv2.imread(gts[0]['img_path'])
        gt_kps = np.stack([np.asarray(g['keypoints'][0], np.float64)
                           for g in gts])
        gt_area = np.array([float(g['area']) for g in gts])
        dm = det_map.get(iid, dict(boxes=[], scores=[]))
        bx = np.asarray(dm['boxes'], np.float64).reshape(-1, 4)
        bs = np.asarray(dm['scores'], np.float64).reshape(-1)
        kps, ksc = (runner(img, bx) if len(bx)
                    else (np.zeros((0, 4, 2)), np.zeros(0)))
        ends.append(dict(bx=bx, bs=bs, kps=kps, ksc=ksc,
                         gt_kps=gt_kps, gt_area=gt_area))

    print('=' * 88)
    print('本地评测口径 vs 榜单（holdout1702，400 张现场图）')
    print('=' * 88)
    print('  榜单（300 张测试图，24轮检测+12轮姿态）：'
          'AP_50_95 97.6694  AP_50 97.7970  AP_75 97.7970  AR 98.3544')
    print()
    print(f'  {"对齐方式":<40}{"AP_50_95":>10}{"AP_50":>9}{"AP_75":>9}'
          f'{"AR_50_95":>11}')
    for tag, perms in (('① 逐点（当前实现）', IDENT),
                       ('② 4 种循环移位', CYCLIC4),
                       ('③ 8 种（移位×绕向）', CYCLIC8),
                       ('④ 24 种全排列', PERM24)):
        r = measure(ends, perms, tag)
        print(f'  {tag:<40}{r["AP"]*100:>10.4f}{r["AP50"]*100:>9.4f}'
              f'{r["AP75"]*100:>9.4f}{r["AR"]*100:>11.4f}')

    print()
    print('  ❗判读：哪一档最接近榜单 97.67 / 97.80 / 97.80 / 98.35，')
    print('     就说明评测方用的是那一档口径。')

    # 逐阈值对比
    print()
    print('=' * 88)
    print('逐 OKS 阈值的 recall 对比')
    print('=' * 88)
    print(f'  {"thr":>6}{"逐点":>10}{"4移位":>10}{"8种":>10}{"24种":>10}')
    r1 = measure(ends, IDENT, 'a')
    r2 = measure(ends, CYCLIC4, 'b')
    r3 = measure(ends, CYCLIC8, 'c')
    r4 = measure(ends, PERM24, 'd')
    for i, t in enumerate(THRS):
        print(f'  {t:>6.2f}{r1["ar_detail"][i]*100:>10.3f}'
              f'{r2["ar_detail"][i]*100:>10.3f}'
              f'{r3["ar_detail"][i]*100:>10.3f}'
              f'{r4["ar_detail"][i]*100:>10.3f}')

    json.dump(dict(ident=dict(AP=r1['AP'], AR=r1['AR']),
                   cyc4=dict(AP=r2['AP'], AR=r2['AR']),
                   cyc8=dict(AP=r3['AP'], AR=r3['AR']),
                   perm24=dict(AP=r4['AP'], AR=r4['AR'])),
              open('_order_test_result.json', 'w', encoding='utf-8'),
              ensure_ascii=False, indent=1)
    print()
    print('  已保存 _order_test_result.json')


if __name__ == '__main__':
    main()
