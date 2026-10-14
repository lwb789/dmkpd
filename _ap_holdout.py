# -*- coding: utf-8 -*-
"""用 holdout1702（400 张现场图，与测试集同分布）建立**与榜单可比**的本地基准。

===================================================================
为什么需要这个脚本
===================================================================
现有 P1（_gap_check.py）用的是 val（480 张，其中 83.9% 是 512x512 低可见
对抗图），而**测试集 300 张全是 1702x1276 现场图**。两者分布差异极大：

    来源                      实例   占比   召回    OKS(GT)  OKS(端到端)  边长中位
    512x512 对抗/低可见        924  83.9%  97.7%   0.9106    0.8938      75.7
    1702x1276 现场(=测试集)    164  14.9% 100.0%   0.9438    0.9401     261.8

这解释了为什么榜单 AP_50_95 = 97.35 高于本地 val 的 93.77：
**现场图本身好得多**（码更大、更清晰），而 val 被对抗图拖低了。

本脚本在 holdout1702 上跑同样的四阶段分析，得到两个基准：
  (a) **GT 框条件**的 AP —— 代表"检测器跳过后"姿态能拿到的分
  (b) **端到端**的 AP —— 代表真实提交能拿到的分
两者之差就是"检测误差"在现场图上的真实代价。

===================================================================
与官方口径对齐的 AP 计算
===================================================================
比赛用 COCO 规范（xtcocotools）：
  · OKS = mean_k exp(-d_k^2 / (2*sigma^2*area_gt))，4 点、sigma 全 0.1
  · AP  = precision 在 recall 上做 **101 点插值**后取均值
  · 只统计与该 GT **同类**且 OKS >= thr 的预测；**每个 GT 只能被匹配一次**
  · 预测按 conf 降序，逐条累积 TP/FP

用法：
    python _ap_holdout.py                 # 全流程
    python _ap_holdout.py --limit 100     # 快速验证
    python _ap_holdout.py --skip-detect   # 用缓存的检测结果
"""
import argparse
import glob
import json
import os
import os.path as osp
import time
from collections import defaultdict
from itertools import permutations

import cv2
import numpy as np
import torch
from mmengine.config import Config
from mmengine.dataset import Compose, pseudo_collate
from mmengine.registry import init_default_scope
from mmengine.runner import load_checkpoint

import mmdet  # noqa
import mmpose  # noqa
from mmdet.apis import inference_detector
from mmdet.registry import MODELS as DET_MODELS
from mmpose.registry import MODELS as POSE_MODELS

SIGMA = 0.1
CACHE = '_ap_holdout_det_cache.json'
THRS = np.arange(0.5, 1.0, 0.05)          # COCO 的 10 个 OKS 阈值

# ================================================================== #
# ❗❗ 点序归一化（2026-10-14 发现并修正）
# ================================================================== #
# 比赛说明原文：「推荐顺序 TL→TR→BR→BL，**但评测脚本会 normalize 点顺序**」
# => 评测方按**点集**比对，点的身份（哪个是 top_left）不影响得分。
#
# 我原来的 `oks_matrix` 是**逐点直接求距离**，没有做任何点序对齐 ——
# 结果把大量「4 个角点位置全对、只是起点/绕向不同」的实例
# 误判成了漏检（OKS≈0）。
#
# 实测（`_order_test.py`，holdout1702 400 张现场图）：
#     对齐方式            AP_50_95  AP_50    AP_75    AR
#     ① 逐点（旧实现）      92.52    94.43    93.27   93.95
#     ② 4 种循环移位        95.76    97.89    96.73   96.60
#     ④ 24 种全排列         95.76    97.89    96.73   96.62
#     榜单                 97.67    97.80    97.80   98.35
# -> **AP_50 与榜单几乎完全吻合（97.89 vs 97.80）**，证明评测方
#    确实是按点集比对。所以本地必须做同样的归一化。
#
# 用「4 种循环移位」还是「24 种全排列」？
#   · 两者结果几乎相同（AR 差 0.02），因为「位置都对但排列非循环」
#     的情形极少，且本来就拿不到好 OKS。
#   · 取 **24 种全排列**：语义上严格等于「点集比对」，
#     且对 `sanitize_quad` 造成的任何重排都免疫。
ALIGN_PERMS = [list(p) for p in permutations(range(4))]


# ================================================================== #
# OKS / AP（与 xtcocotools 等价）
# ================================================================== #
def oks_matrix(pred_kps, pred_conf, gt_kps, gt_area):
    """返回 (P, G) 的 OKS 矩阵，**已做点序归一化**。

    pred_kps: (P, 4, 2)   gt_kps: (G, 4, 2)   gt_area: (G,)
    OKS = mean_k exp(-d^2 / (2*sigma^2*area_gt))

    ❗对 4 个点取**最优排列**，等价于「按点集比对」。
      比赛说明明确了评测会 normalize 点顺序，这里必须一致，
      否则会把「位置全对、只有起点不同」误判成漏检。
      详见文件头 ALIGN_PERMS 处的实测数据。
    """
    P, G = len(pred_kps), len(gt_kps)
    if P == 0 or G == 0:
        return np.zeros((P, G))
    pred = np.asarray(pred_kps)
    gt = np.asarray(gt_kps)
    area = np.asarray(gt_area)
    best = np.zeros((P, G))
    for pm in ALIGN_PERMS:
        d2 = ((pred[:, None, np.asarray(pm), :]
               - gt[None, :, :, :]) ** 2).sum(-1)      # (P, G, K)
        oks = np.exp(-d2 / (2.0 * SIGMA ** 2 * area[None, :, None])).mean(-1)
        best = np.maximum(best, oks)
    return best


def ap_from_oks(oks, conf, thr):
    """单图、单阈值的 COCO 匹配 —— 返回 (scores, tps, fps)。

    oks: (P, G)  conf: (P,)

    与 xtcocotools 一致：
      · 预测按 conf 降序
      · 每条预测在「尚未被匹配的 GT」里取 OKS 最大者
      · 若该 OKS >= thr 则 TP（并占用该 GT），否则 FP
    """
    P, G = oks.shape
    if P == 0:
        return np.zeros(0), np.zeros(0), np.zeros(0)
    order = np.argsort(-conf)
    matched = np.zeros(G, dtype=bool)
    sc, tp, fp = np.zeros(P), np.zeros(P), np.zeros(P)
    for rank, pi in enumerate(order):
        sc[rank] = conf[pi]
        free = np.flatnonzero(~matched)
        if len(free) == 0:
            fp[rank] = 1
            continue
        bi = free[int(np.argmax(oks[pi, free]))]
        if oks[pi, bi] >= thr:
            matched[bi] = True
            tp[rank] = 1
        else:
            fp[rank] = 1
    return sc, tp, fp


def coco_ap(per_img, max_det=100):
    """COCO 口径的 AP。

    ❗关键：**不是**按图求平均，而是把全数据集的预测汇总，
    按 conf 全局降序排成**一条** PR 曲线，再取 101 点插值。
    （xtcocotools 的 `COCOeval.accumulate` 就是这么做的；
      按图平均会系统性低估，且与榜单不可比。）

    返回 dict(AP, AP50, AP75, detail)
    """
    n_gt = sum(len(e['gt_kps']) for e in per_img)
    aps = []
    for thr in THRS:
        S, T, F = [], [], []
        for e in per_img:
            if len(e['gt_kps']) == 0 or len(e['pred_kps']) == 0:
                continue
            oks = oks_matrix(e['pred_kps'], None, e['gt_kps'], e['gt_area'])
            sc, tp, fp = ap_from_oks(oks, e['pred_conf'], thr)
            if len(sc) == 0:
                continue
            # COCO 的 maxDets=100：每图只保留 conf 最高的 100 条
            if len(sc) > max_det:
                k = np.argsort(-sc)[:max_det]
                sc, tp, fp = sc[k], tp[k], fp[k]
            S.append(sc)
            T.append(tp)
            F.append(fp)
        if not S:
            aps.append(0.0)
            continue
        S, T, F = np.concatenate(S), np.concatenate(T), np.concatenate(F)
        o = np.argsort(-S)
        tpc = np.cumsum(T[o])
        fpc = np.cumsum(F[o])
        rec = tpc / max(n_gt, 1)
        prec = tpc / np.maximum(tpc + fpc, 1e-9)
        # 101 点插值（单调化后取右端最大）
        q = np.linspace(0, 1, 101)
        idx = np.searchsorted(rec, q, side='left')
        p_int = np.zeros(101)
        ok = idx < len(prec)
        p_int[ok] = prec[idx[ok]]
        # 保证单调不增（取后缀最大）
        p_int = np.maximum.accumulate(p_int[::-1])[::-1]
        aps.append(float(p_int.mean()))
    aps = np.array(aps)
    return dict(
        AP=float(np.nanmean(aps)),
        AP50=float(aps[0]),
        AP75=float(aps[5]),
        detail={f'{t:.2f}': float(v) for t, v in zip(THRS, aps)})


# ================================================================== #
class PoseOnBoxes:
    def __init__(self, model, cfg):
        self.model = model
        self.pipeline = Compose(cfg.test_pipeline)
        from mmpose.datasets.datasets.utils import parse_pose_metainfo
        self.meta = parse_pose_metainfo(dict(
            from_file='open-mmlab/mmpose/configs/_base_/datasets/'
                      'DMKPD_meta.py'))

    @torch.no_grad()
    def __call__(self, img, boxes):
        if len(boxes) == 0:
            return np.zeros((0, 4, 2)), np.zeros(0)
        init_default_scope('mmpose')
        data_list = []
        for i in range(len(boxes)):
            di = dict(img=img, bbox=np.asarray(boxes[i:i + 1], np.float32),
                      bbox_score=np.ones(1, np.float32))
            di.update(self.meta)
            data_list.append(self.pipeline(di))
        res = self.model.test_step(pseudo_collate(data_list))
        kps, scs = [], []
        for r in res:
            pi = r.pred_instances
            kp = pi.keypoints
            kp = kp.detach().cpu().numpy() if hasattr(kp, 'detach') \
                else np.asarray(kp)
            ks = pi.keypoint_scores
            ks = ks.detach().cpu().numpy() if hasattr(ks, 'detach') \
                else np.asarray(ks)
            kps.append(kp[0, :, :2])
            scs.append(float(np.mean(ks[0])))
        return np.asarray(kps), np.asarray(scs)


def box_iou(a, b):
    x1 = np.maximum(a[0], b[:, 0])
    y1 = np.maximum(a[1], b[:, 1])
    x2 = np.minimum(a[2], b[:, 2])
    y2 = np.minimum(a[3], b[:, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    aa = (a[2] - a[0]) * (a[3] - a[1])
    bb = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / np.maximum(aa + bb - inter, 1e-9)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--det-config', default='config_det.py')
    ap.add_argument('--det-ckpt', default=None)
    ap.add_argument('--pose-config', default='config.py')
    ap.add_argument('--pose-ckpt', default=None,
                    help='默认自动找 work_dirs/dmkpd/best_coco_AP*.pth 中'
                         '**mtime 最新**的')
    ap.add_argument('--split', default='splits/holdout1702.txt',
                    help='DMKPD_dataset 下的划分文件')
    ap.add_argument('--score-thr', type=float, default=0.30)
    ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--skip-detect', action='store_true')
    args = ap.parse_args()

    if args.det_ckpt is None:
        c = sorted(glob.glob('work_dirs/dmkpd_det/best_coco_bbox_mAP*.pth'))
        args.det_ckpt = c[-1] if c else None

    if args.pose_ckpt is None:
        # 按 mtime 取最新（旧权重会进 _old_best/，不在 glob 内）
        c = glob.glob('work_dirs/dmkpd/best_coco_AP*.pth')
        args.pose_ckpt = max(c, key=osp.getmtime) if c else None
    if not args.pose_ckpt or not osp.isfile(args.pose_ckpt):
        raise SystemExit(f'找不到姿态权重：{args.pose_ckpt}')

    print('=' * 80)
    print('holdout1702 基准 AP（与榜单口径一致）')
    print('=' * 80)
    print(f'  划分      DMKPD_dataset/{args.split}')
    print(f'  检测权重  {args.det_ckpt}')
    print(f'  姿态权重  {args.pose_ckpt}')

    # ---------- 模型 ---------- #
    cfg_d = Config.fromfile(args.det_config)
    init_default_scope('mmdet')
    det = DET_MODELS.build(cfg_d.model).cuda().eval()
    load_checkpoint(det, args.det_ckpt, map_location='cpu', strict=False)
    det.cfg = cfg_d

    cfg_p = Config.fromfile(args.pose_config)
    init_default_scope('mmpose')
    pose = POSE_MODELS.build(cfg_p.model).cuda().eval()
    load_checkpoint(pose, args.pose_ckpt, map_location='cpu', strict=False)
    pose.cfg = cfg_p

    # ---------- 数据（姿态数据集：四角齐全，含 1702 现场图） ---------- #
    from mmpose.datasets.datasets.DM.registry import DMKPD_dataset
    ds = DMKPD_dataset(
        data_root='DMKPD_dataset', ann_file=args.split,
        data_prefix=dict(img='train/images'), metainfo=cfg_p.metainfo,
        serialize_data=False)
    n = args.limit or len(ds)
    by_img = defaultdict(list)
    for i in range(n):
        info = ds.get_data_info(i)
        by_img[info['img_id']].append((i, info))
    img_ids = sorted(by_img)
    print(f'  图 {len(img_ids)} 张 / 实例 '
          f'{sum(len(by_img[i]) for i in img_ids)} 个')

    runner = PoseOnBoxes(pose, cfg_p)

    # ---------- 检测（缓存） ---------- #
    det_map = {}
    if args.skip_detect and osp.isfile(CACHE):
        det_map = json.load(open(CACHE, encoding='utf-8'))
        print(f'  [缓存] {CACHE}（{len(det_map)} 张）')
    elif not args.skip_detect:
        if osp.isfile(CACHE):
            det_map = json.load(open(CACHE, encoding='utf-8'))
            print(f'  [缓存] 复用 {CACHE}（{len(det_map)} 张）')
        todo = [i for i in img_ids if str(i) not in det_map]
        if todo:
            print(f'  [检测] 还需跑 {len(todo)} 张')
        t0 = time.time()
        for k, iid in enumerate(todo):
            info0 = by_img[iid][0][1]
            img = cv2.imread(info0['img_path'])
            H, W = img.shape[:2]
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
            det_map[str(iid)] = dict(boxes=bx.tolist(), scores=sc.tolist())
            if (k + 1) % 50 == 0 or k == 0:
                print(f'    [{k+1}/{len(todo)}] {time.time()-t0:.0f}s')
        json.dump(det_map, open(CACHE, 'w', encoding='utf-8'))
        print(f'  [缓存] 已写入 {CACHE}')

    # ---------- 逐图推理，收两个条件的预测 ---------- #
    ends_gt, ends_det = [], []
    t0 = time.time()
    for k, iid in enumerate(img_ids):
        gts = by_img[iid]
        img = cv2.imread(gts[0][1]['img_path'])
        gt_kps = np.stack([np.asarray(g['keypoints'][0], np.float64)
                           for _, g in gts])
        gt_area = np.array([float(g['area']) for _, g in gts])
        gt_box = np.stack([np.asarray(g['bbox'][0], np.float64)
                           for _, g in gts])

        # (a) GT 框条件
        kps_g, scs_g = runner(img, gt_box)
        ends_gt.append(dict(pred_kps=kps_g, pred_conf=scs_g,
                            gt_kps=gt_kps, gt_area=gt_area))

        # (b) 端到端：检测框全部送姿态，conf = bbox_score * mean(kp_score)
        dm = det_map.get(str(iid), dict(boxes=[], scores=[]))
        dbx = np.asarray(dm['boxes'], np.float64).reshape(-1, 4)
        dsc = np.asarray(dm['scores'], np.float64).reshape(-1)
        if len(dbx):
            kps_d, scs_d = runner(img, dbx)
            conf_d = dsc * scs_d
        else:
            kps_d, conf_d = np.zeros((0, 4, 2)), np.zeros(0)
        ends_det.append(dict(pred_kps=kps_d, pred_conf=conf_d,
                             gt_kps=gt_kps, gt_area=gt_area))

        if (k + 1) % 25 == 0 or k == 0:
            el = time.time() - t0
            print(f'    [{k+1}/{len(img_ids)}] {el:.0f}s  '
                  f'ETA {el/(k+1)*(len(img_ids)-k-1):.0f}s')

    # ---------- AP ---------- #
    print()
    print('=' * 80)
    print('结果（COCO 101 点插值，4 点 sigma=0.1）')
    print('=' * 80)
    r_gt = coco_ap(ends_gt)
    r_det = coco_ap(ends_det)
    print(f'  {"条件":<22}{"AP_50_95":>10}{"AP_50":>9}{"AP_75":>9}')
    print('  ' + '-' * 50)
    print(f'  {"GT 框（检测器跳过）":<22}{r_gt["AP"]*100:>10.4f}'
          f'{r_gt["AP50"]*100:>9.4f}{r_gt["AP75"]*100:>9.4f}')
    print(f'  {"端到端（真实提交）":<22}{r_det["AP"]*100:>10.4f}'
          f'{r_det["AP50"]*100:>9.4f}{r_det["AP75"]*100:>9.4f}')
    print(f'  {"差值（检测误差）":<22}'
          f'{(r_det["AP"]-r_gt["AP"])*100:>+10.4f}'
          f'{(r_det["AP50"]-r_gt["AP50"])*100:>+9.4f}'
          f'{(r_det["AP75"]-r_gt["AP75"])*100:>+9.4f}')
    print()
    print('  各 OKS 阈值明细：')
    print(f'  {"thr":>6}{"GT框":>10}{"端到端":>10}{"差":>10}')
    for t in THRS:
        k = f'{t:.2f}'
        print(f'  {t:>6.2f}{r_gt["detail"][k]*100:>10.3f}'
              f'{r_det["detail"][k]*100:>10.3f}'
              f'{(r_det["detail"][k]-r_gt["detail"][k])*100:>+10.3f}')

    json.dump(dict(gt=r_gt, det=r_det, split=args.split,
                   det_ckpt=args.det_ckpt, pose_ckpt=args.pose_ckpt,
                   n_img=len(img_ids)),
              open('_ap_holdout_result.json', 'w', encoding='utf-8'),
              ensure_ascii=False, indent=1)
    print()
    print('  已保存 _ap_holdout_result.json')


if __name__ == '__main__':
    main()
