# -*- coding: utf-8 -*-
"""AR 归因分析：召回到底丢在哪里。

===================================================================
背景：榜单（300 张测试图）
===================================================================
    版本                 AP_50_95   AP_50     AP_75     AR_50_95
    24轮检测 + 5轮姿态    97.3524   97.7762   97.7328   98.2278
    24轮检测 + 12轮姿态   97.6694   97.7970   97.7970   98.3544

❗两个决定性观察
  1. **AP_75 == AP_50（都是 97.7970）**
     -> 没有任何「已匹配」实例的 OKS 落在 [0.5, 0.75)
     -> 即：**凡是被检出的实例，定位精度都已 ≥ 0.75**
     -> 结论：**定位已不是瓶颈**，继续抠精度对 AP 无益
  2. **AR(98.3544) > AP(97.6694)**
     -> 召回高于精度，说明丢分的主因是「少数实例完全没检出」
        而不是「检出了但定位不准」

=> 唯一还有空间的方向：**提高召回**。

===================================================================
本脚本做什么
===================================================================
   A. 逐 OKS 阈值打印 AP 与 recall —— 看损失集中在哪个阈值
   B. 每个 GT 的「最佳 OKS」分布（用全部预测算，不受 conf 过滤影响）
        <0.5        -> 完全没被"够到"（真漏检）
        [0.5, 0.75) -> 检出了但不够准
        >=0.75      -> 已达标
   C. 各种 conf 方案的对比，量化每条通道能捞回多少召回：
        ① 现状：conf = bbox_score × mean(kp_score)，检测阈值 0.3
        ② conf 只用 bbox_score（不乘 kp 分数）
        ③ 降低检测阈值 / 最小边长
        ④ 放宽 NMS

⚠️ 比赛约束：**conf < 0.3 的行不参与评测**。
   所以「放宽到 0.3 以下」是无效的；能做的只有
   「改变 conf 的计算方式，让更多真实框跨过 0.3」以及
   「让检测器多产出一些框」。

用法：
    python _ar_analyze.py                 # 全流程（含检测，约 2 分钟）
    python _ar_analyze.py --skip-detect   # 复用缓存
"""
import argparse
import glob
import json
import os
import os.path as osp
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
from mmpose.registry import MODELS as POSE_MODELS

from _ap_holdout import (THRS, ap_from_oks, coco_ap, oks_matrix,
                         PoseOnBoxes)

CACHE = '_ar_det_cache.json'
COMP_THR = 0.30          # 比赛硬约束：conf < 0.3 不计分


# ================================================================== #
def coco_ar(per_img, max_det=100):
    """AR：每个 OKS 阈值下的 recall@maxDets，再对阈值平均。"""
    n_gt = sum(len(e['gt_kps']) for e in per_img)
    if n_gt == 0:
        return np.zeros(len(THRS))
    out = []
    for thr in THRS:
        tp = 0
        for e in per_img:
            if len(e['gt_kps']) == 0 or len(e['pred_kps']) == 0:
                continue
            oks = oks_matrix(e['pred_kps'], None, e['gt_kps'], e['gt_area'])
            sc, t, _ = ap_from_oks(oks, e['pred_conf'], thr)
            if len(sc) > max_det:
                k = np.argsort(-sc)[:max_det]
                t = t[k]
            tp += int(t.sum())
        out.append(tp / n_gt)
    return np.array(out)


def gt_best_oks(per_img):
    """每个 GT 的「最佳 OKS」（跨全部预测取最大）——衡量"可达召回"。"""
    vals = []
    for e in per_img:
        g = len(e['gt_kps'])
        if g == 0:
            continue
        if len(e['pred_kps']) == 0:
            vals.extend([0.0] * g)
            continue
        oks = oks_matrix(e['pred_kps'], None, e['gt_kps'], e['gt_area'])
        vals.extend(oks.max(0).tolist())
    return np.array(vals)


def nms_np(boxes, scores, thr):
    """给本次对比用的轻量 NMS（与 mmdet 的略有差异，仅用于相对比较）。"""
    if len(boxes) == 0:
        return np.zeros(0, bool)
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = np.maximum(x2 - x1, 0) * np.maximum(y2 - y1, 0)
    order = scores.argsort()[::-1]
    keep = np.zeros(len(boxes), bool)
    while order.size:
        i = order[0]
        keep[i] = True
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        inter = np.clip(xx2 - xx1, 0, None) * np.clip(yy2 - yy1, 0, None)
        iou = inter / np.maximum(areas[i] + areas[order[1:]] - inter, 1e-9)
        order = order[1:][iou <= thr]
    return keep


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--det-config', default='config_det.py')
    ap.add_argument('--det-ckpt', default=None)
    ap.add_argument('--pose-config', default='config.py')
    ap.add_argument('--pose-ckpt', default=None)
    ap.add_argument('--split', default='splits/holdout1702.txt')
    ap.add_argument('--det-thr', type=float, default=0.01,
                    help='检测保留阈值（故意放低以做分析）')
    ap.add_argument('--min-side', type=float, default=2.0)
    ap.add_argument('--skip-detect', action='store_true')
    ap.add_argument('--limit', type=int, default=0)
    args = ap.parse_args()

    if args.det_ckpt is None:
        c = sorted(glob.glob('work_dirs/dmkpd_det/best_coco_bbox_mAP*.pth'))
        args.det_ckpt = c[-1] if c else None
    if args.pose_ckpt is None:
        c = glob.glob('work_dirs/dmkpd/best_coco_AP*.pth')
        args.pose_ckpt = max(c, key=osp.getmtime) if c else None

    print('=' * 82)
    print('AR 归因分析（holdout1702，400 张现场图）')
    print('=' * 82)
    print(f'  检测权重 {args.det_ckpt}')
    print(f'  姿态权重 {args.pose_ckpt}')

    # ---------- 模型 ---------- #
    cfgd = Config.fromfile(args.det_config)
    init_default_scope('mmdet')
    det = DET_MODELS.build(cfgd.model).cuda().eval()
    load_checkpoint(det, args.det_ckpt, map_location='cpu', strict=False)
    det.cfg = cfgd

    cfgp = Config.fromfile(args.pose_config)
    init_default_scope('mmpose')
    pose = POSE_MODELS.build(cfgp.model).cuda().eval()
    load_checkpoint(pose, args.pose_ckpt, map_location='cpu', strict=False)
    pose.cfg = cfgp
    runner = PoseOnBoxes(pose, cfgp)

    # ---------- 数据 ---------- #
    from mmpose.datasets.datasets.DM.registry import DMKPD_dataset
    ds = DMKPD_dataset(
        data_root='DMKPD_dataset', ann_file=args.split,
        data_prefix=dict(img='train/images'), metainfo=cfgp.metainfo,
        serialize_data=False)
    by_img = defaultdict(list)
    for i in range(len(ds)):
        info = ds.get_data_info(i)
        by_img[info['img_id']].append((i, info))
    img_ids = sorted(by_img)
    if args.limit:
        img_ids = img_ids[:args.limit]
    n_gt = sum(len(by_img[i]) for i in img_ids)
    print(f'  {len(img_ids)} 图 / {n_gt} 实例')

    # ---------- 检测（低阈值 + 缓存） ---------- #
    det_map = {}
    if args.skip_detect and osp.isfile(CACHE):
        det_map = json.load(open(CACHE, encoding='utf-8'))
        print(f'  [缓存] {CACHE}  {len(det_map)} 图')
    else:
        if osp.isfile(CACHE):
            det_map = json.load(open(CACHE, encoding='utf-8'))
        todo = [i for i in img_ids if str(i) not in det_map]
        print(f'  [检测] 需跑 {len(todo)} 张（阈值 {args.det_thr}）')
        t0 = time.time()
        for k, iid in enumerate(todo):
            img = cv2.imread(by_img[iid][0][1]['img_path'])
            H, W = img.shape[:2]
            init_default_scope('mmdet')
            pi = inference_detector(det, img).pred_instances
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
                keep = (sc >= args.det_thr) & (np.minimum(w, h) >= args.min_side)
                bx, sc = bx[keep], sc[keep]
            det_map[str(iid)] = dict(boxes=bx.tolist(), scores=sc.tolist())
            if (k + 1) % 100 == 0:
                print(f'    [{k+1}/{len(todo)}] {time.time()-t0:.0f}s')
        json.dump(det_map, open(CACHE, 'w', encoding='utf-8'))

    # ---------- 姿态（跑全部框，保留 kp 分数） ---------- #
    ends = []
    t0 = time.time()
    for k, iid in enumerate(img_ids):
        gts = by_img[iid]
        img = cv2.imread(gts[0][1]['img_path'])
        gt_kps = np.stack([np.asarray(g['keypoints'][0], np.float64)
                           for _, g in gts])
        gt_area = np.array([float(g['area']) for _, g in gts])
        dm = det_map.get(str(iid), dict(boxes=[], scores=[]))
        bx = np.asarray(dm['boxes'], np.float64).reshape(-1, 4)
        bs = np.asarray(dm['scores'], np.float64).reshape(-1)
        if len(bx):
            kps, ksc = runner(img, bx)
        else:
            kps, ksc = np.zeros((0, 4, 2)), np.zeros(0)
        ends.append(dict(iid=iid, bx=bx, bs=bs, kps=kps, ksc=ksc,
                         gt_kps=gt_kps, gt_area=gt_area))
        if (k + 1) % 100 == 0:
            print(f'    [{k+1}/{len(img_ids)}] {time.time()-t0:.0f}s')

    # ============================================================== #
    def build(conf_mode='mul', det_thr=0.30, min_side=12.0,
              nms_thr=None, comp_thr=COMP_THR):
        """按给定方案产出 per_img（已按比赛约束 conf>=0.3 过滤）。"""
        out = []
        for e in ends:
            bx, bs, kps, ksc = e['bx'], e['bs'], e['kps'], e['ksc']
            keep = bs >= det_thr
            if min_side > 0 and len(bx):
                w = bx[:, 2] - bx[:, 0]
                h = bx[:, 3] - bx[:, 1]
                keep &= (np.minimum(w, h) >= min_side)
            if nms_thr is not None and len(bx):
                kn = nms_np(bx, bs, nms_thr)
                keep &= kn
            kp = kps[keep]
            b = bs[keep]
            ks = ksc[keep]
            if conf_mode == 'mul':
                conf = b * ks
            else:                       # 'bbox'
                conf = b
            m = conf >= comp_thr
            out.append(dict(pred_kps=kp[m], pred_conf=conf[m],
                            gt_kps=e['gt_kps'], gt_area=e['gt_area']))
        return out

    def report(name, per_img):
        r = coco_ap(per_img)
        ar = coco_ar(per_img)
        n_pred = sum(len(e['pred_kps']) for e in per_img)
        print(f'  {name:<34}{r["AP"]*100:>10.4f}{r["AP50"]*100:>9.4f}'
              f'{r["AP75"]*100:>9.4f}{ar.mean()*100:>11.4f}'
              f'{n_pred:>8}')
        return r, ar

    # ---------- A. 逐阈值明细（现状） ---------- #
    print()
    print('=' * 82)
    print('A. 现状方案（conf = bbox×kp，检测阈 0.3，最小边 12）')
    print('=' * 82)
    cur = build('mul', 0.30, 12.0)
    r_cur = coco_ap(cur)
    ar_cur = coco_ar(cur)
    print(f'  {"thr":>6}{"AP":>10}{"recall":>10}')
    for t, a, rc in zip(THRS, [r_cur['detail'][f'{x:.2f}'] for x in THRS],
                        ar_cur):
        print(f'  {t:>6.2f}{a*100:>10.3f}{rc*100:>10.3f}')
    print(f'  {"均值":>6}{r_cur["AP"]*100:>10.4f}{ar_cur.mean()*100:>10.4f}')

    # ---------- B. 每个 GT 的最佳 OKS 分布 ---------- #
    print()
    print('=' * 82)
    print('B. 每个 GT 的「最佳 OKS」分布（用全部预测算，不受 conf 过滤）')
    print('=' * 82)
    bx_all = build('mul', args.det_thr, args.min_side, comp_thr=-1.0)
    best = gt_best_oks(bx_all)
    tot = len(best)
    for lo, hi, tag in ((0.0, 0.5, '完全没够到（真漏检）'),
                        (0.5, 0.75, '检出了但不够准'),
                        (0.75, 1.01, '已达标(>=0.75)')):
        s = ((best >= lo) & (best < hi)).sum()
        print(f'  [{lo:.2f}, {hi if hi <= 1 else 1:.2f})  {s:>5} 个  '
              f'{100*s/tot:>6.2f}%   {tag}')
    print()
    print(f'  若「完全没够到」= 0，则 recall 上限可达 100%')
    print(f'  当前「完全没够到」{((best < 0.5).sum())} 个 '
          f'= {100*(best<0.5).sum()/tot:.2f}%  <- 这就是 AR 的天花板所在')

    # ---------- C. 各方案对比 ---------- #
    print()
    print('=' * 82)
    print('C. 各方案对比（全部已按 conf>=0.3 过滤，符合比赛约束）')
    print('=' * 82)
    print(f'  {"方案":<34}{"AP_50_95":>10}{"AP_50":>9}{"AP_75":>9}'
          f'{"AR_50_95":>11}{"框数":>8}')
    cfgs = [
        ('① 现状 bbox×kp, 检测0.3',      dict(conf_mode='mul', det_thr=0.30,
                                              min_side=12.0)),
        ('② conf 只用 bbox',             dict(conf_mode='bbox', det_thr=0.30,
                                              min_side=12.0)),
        ('③ 检测阈 0.10, bbox×kp',       dict(conf_mode='mul', det_thr=0.10,
                                              min_side=12.0)),
        ('④ 检测阈 0.10, 最小边 6',      dict(conf_mode='mul', det_thr=0.10,
                                              min_side=6.0)),
        ('⑤ 检测阈 0.05, 最小边 4',      dict(conf_mode='mul', det_thr=0.05,
                                              min_side=4.0)),
        ('⑥ 检测阈 0.05 + NMS 0.75',     dict(conf_mode='mul', det_thr=0.05,
                                              min_side=4.0, nms_thr=0.75)),
        ('⑦ 检测阈 0.30 + NMS 0.75',     dict(conf_mode='mul', det_thr=0.30,
                                              min_side=12.0, nms_thr=0.75)),
    ]
    rows = []
    for name, kw in cfgs:
        r, ar = report(name, build(**kw))
        rows.append((name, r, ar))

    print()
    print('  ❗相对现状(①)的变化：')
    base = rows[0]
    for name, r, ar in rows[1:]:
        print(f'    {name:<30} '
              f'AP_50_95 {(r["AP"]-base[1]["AP"])*100:>+7.4f}  '
              f'AR_50_95 {(ar.mean()-base[2].mean())*100:>+7.4f}')

    # ---------- D. conf 公式扫描 ---------- #
    # ❗关键发现：③~⑦ 与 ① 结果**完全相同**（都是 517 框），说明
    #   降低检测阈值无效 —— 因为 conf = bbox × kp ≤ bbox，
    #   框的 bbox_score 即使有 0.29，乘上 kp 后必然 < 0.3 而被竞赛约束剔除。
    #   => **真正的约束是 conf 的构造方式，不是检测阈值。**
    #   所以要在这里做文章：找一个既保留高分排序、又不把 conf 压太低的公式。
    print()
    print('=' * 82)
    print('D. conf 公式扫描（都在 conf>=0.3 的竞赛约束下）')
    print('=' * 82)
    print(f'  {"公式":<30}{"AP_50_95":>10}{"AP_50":>9}{"AP_75":>9}'
          f'{"AR_50_95":>11}{"框数":>8}')
    formulas = [
        ('bbox × kp     （现状）', lambda b, k: b * k),
        ('bbox × kp^0.8', lambda b, k: b * k ** 0.8),
        ('bbox × kp^0.6', lambda b, k: b * k ** 0.6),
        ('bbox × kp^0.5', lambda b, k: b * np.sqrt(k)),
        ('bbox × kp^0.3', lambda b, k: b * k ** 0.3),
        ('bbox          （不乘 kp）', lambda b, k: b),
        ('min(bbox, kp)', lambda b, k: np.minimum(b, k)),
        ('sqrt(bbox×kp)', lambda b, k: np.sqrt(b * k)),
        ('bbox×(0.5+0.5kp)', lambda b, k: b * (0.5 + 0.5 * k)),
    ]
    rows2 = []
    for name, fn in formulas:
        per = []
        for e in ends:
            keep = e['bs'] >= 0.30
            if len(e['bx']):
                w = e['bx'][:, 2] - e['bx'][:, 0]
                h = e['bx'][:, 3] - e['bx'][:, 1]
                keep &= (np.minimum(w, h) >= 12.0)
            b, k = e['bs'][keep], e['ksc'][keep]
            conf = fn(b, k) if len(b) else np.zeros(0)
            m = conf >= COMP_THR
            per.append(dict(pred_kps=e['kps'][keep][m], pred_conf=conf[m],
                            gt_kps=e['gt_kps'], gt_area=e['gt_area']))
        r, ar = report(name, per)
        rows2.append((name, r, ar))
    print()
    print('  ❗相对现状的变化：')
    b0 = rows2[0]
    for name, r, ar in rows2[1:]:
        print(f'    {name:<30} '
              f'AP_50_95 {(r["AP"]-b0[1]["AP"])*100:>+7.4f}  '
              f'AR_50_95 {(ar.mean()-b0[2].mean())*100:>+7.4f}')

    # ---------- E. 「完全没够到」的 17 个 GT 归因 ---------- #
    print()
    print('=' * 82)
    print('E. 「完全没够到」的 GT 归因（检测 miss 还是姿态 miss？）')
    print('=' * 82)
    miss_det, miss_pose = [], []
    for e in ends:
        if len(e['gt_kps']) == 0:
            continue
        if len(e['kps']) == 0:
            for gi in range(len(e['gt_kps'])):
                miss_det.append(dict(iid=e['iid'], side=float(
                    np.sqrt(e['gt_area'][gi])), best=0.0,
                    det_iou=0.0, kp=0.0))
            continue
        oks = oks_matrix(e['kps'], None, e['gt_kps'], e['gt_area'])
        best_per_gt = oks.max(0)
        for gi in range(len(e['gt_kps'])):
            if best_per_gt[gi] >= 0.5:
                continue
            pi = int(oks[:, gi].argmax())
            # 该 GT 与「所有检测框」的最大 IoU（不管分数）
            x1 = np.maximum(e['gt_kps'][gi].min(0)[0],
                            e['bx'][:, 0]) if len(e['bx']) else np.zeros(0)
            gb = np.array([e['gt_kps'][gi][:, 0].min(),
                           e['gt_kps'][gi][:, 1].min(),
                           e['gt_kps'][gi][:, 0].max(),
                           e['gt_kps'][gi][:, 1].max()])
            diou = 0.0
            if len(e['bx']):
                xx1 = np.maximum(gb[0], e['bx'][:, 0])
                yy1 = np.maximum(gb[1], e['bx'][:, 1])
                xx2 = np.minimum(gb[2], e['bx'][:, 2])
                yy2 = np.minimum(gb[3], e['bx'][:, 3])
                inter = np.clip(xx2 - xx1, 0, None) * np.clip(
                    yy2 - yy1, 0, None)
                ga = max((gb[2] - gb[0]) * (gb[3] - gb[1]), 1e-9)
                ba = np.maximum(e['bx'][:, 2] - e['bx'][:, 0], 0) * \
                    np.maximum(e['bx'][:, 3] - e['bx'][:, 1], 0)
                diou = float((inter / np.maximum(ga + ba - inter, 1e-9)).max())
            rec = dict(iid=e['iid'],
                       side=float(np.sqrt(e['gt_area'][gi])),
                       best=float(best_per_gt[gi]), det_iou=diou,
                       kp=float(e['ksc'][pi]) if len(e['ksc']) else 0.0,
                       bx_score=float(e['bs'][pi]) if len(e['bs']) else 0.0)
            (miss_pose if diou >= 0.3 else miss_det).append(rec)

    print(f'  {"类型":<44}{"数量":>6}{"占比":>9}')
    print(f'  {"A. 检测没框住（最佳检测 IoU < 0.3）":<44}'
          f'{len(miss_det):>6}{100*len(miss_det)/tot:>8.2f}%')
    print(f'  {"B. 框住了但姿态定位错（IoU>=0.3 却 OKS<0.5）":<44}'
          f'{len(miss_pose):>6}{100*len(miss_pose)/tot:>8.2f}%')
    if miss_det or miss_pose:
        print()
        print(f'  {"img_id":<10}{"边长px":>9}{"最佳OKS":>10}{"检测IoU":>10}'
              f'{"kp分":>8}{"框分":>8}')
        for r in sorted(miss_det + miss_pose, key=lambda x: x['side']):
            print(f'  {r["iid"]:<10}{r["side"]:>9.1f}{r["best"]:>10.4f}'
                  f'{r["det_iou"]:>10.3f}{r["kp"]:>8.3f}{r["bx_score"]:>8.3f}')
    print()
    print('  ❗结论指引：')
    print('    · 若 A 为主 -> 瓶颈在**检测器**（漏检），需提升检测或放低门槛')
    print('    · 若 B 为主 -> 瓶颈在**姿态**（定位错），但这些实例的 kp 分数通常很低')
    print('    · 无论哪种，都可用「这些实例的边长」判断是否是小目标问题')

    json.dump(dict(
        thr=[float(t) for t in THRS],
        ap_cur=[float(r_cur['detail'][f'{x:.2f}']) for x in THRS],
        ar_cur=[float(v) for v in ar_cur],
        best_oks=best.tolist(),
        n_gt=int(tot),
        miss_det=miss_det, miss_pose=miss_pose,
    ), open('_ar_analyze_result.json', 'w', encoding='utf-8'),
        ensure_ascii=False)
    print()
    print('  已保存 _ar_analyze_result.json')


if __name__ == '__main__':
    main()
