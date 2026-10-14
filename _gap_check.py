# -*- coding: utf-8 -*-
"""P1：量化「用检测框替代 GT 框」给姿态精度带来的损失（gap）。

===================================================================
要回答的问题
===================================================================
姿态模型现有的 AP 0.9377 是**用 GT bbox 评的**（val_pipeline 里的
`GetBBoxCenterScale` 直接消费标注里的 bbox）。
但端到端跑测试集时，bbox 来自检测器 —— 检测器有定位误差，
姿态模型会基于一个"偏了的框"去裁切和定位。

**这个偏差到底让精度掉多少？** 这个数从来没测过。

===================================================================
评估口径（已核实，与官方完全一致）
===================================================================
`xtcocotools/cocoeval.py:computeOks()` 的公式：

    OKS = (1/K) * Σ_k exp( -d_k^2 / (2 * σ_k^2 * area_gt) )

  · K = 4（四个角点），σ 全为 0.1（本数据集元信息）
  · d_k = 预测点与实际点的欧氏距离（**原图像素**）
  · area_gt = **GT 实例的 `area` 字段**（= 标注框 (x2-x1)*(y2-y1)）

❗关键：`use_area=True` 时**分母只用 GT 的 area**，不含检测框面积。
  （`oks_iou()` 里那个 `(a_g + a_d)/2` 是给 OKS-NMS 用的，不是评估用的。）
  所以检测框误差**不直接进入 OKS 分母**，而是通过
  「裁切窗口偏移 → 姿态输入变形 → 预测点偏移」间接起作用。

===================================================================
四个阶段
===================================================================
  Phase 1  检测-标注匹配分析：IoU 分布、召回率、误检数
  Phase 2  OKS 对比：同一实例，GT 框 vs 检测框，逐实例配对比较
  Phase 3  灵敏度扫描：把 GT 框人为扰动（模拟"检测器变好"），
           画出 IoU -> OKS 曲线，用来预测 24 epoch 检测器能带来多少收益
  Phase 4  结论与优化建议

用法：
    python _gap_check.py                    # 全部阶段
    python _gap_check.py --limit 100        # 先用少量图快速验证
    python _gap_check.py --skip-detect      # 跳过检测器（只用已缓存的框）
"""
import argparse
import json
import os
import os.path as osp
import time
from collections import defaultdict

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

SIGMA = 0.1          # 本数据集四个角点等 sigma（DMKPD_meta.py）
CACHE = '_gap_det_cache.json'


# ================================================================== #
# OKS（与 xtcocotools 的 computeOks 等价，K=4，sigma=0.1）
# ================================================================== #
def oks(pred_kps, gt_kps, area):
    """单实例 OKS。

    pred_kps, gt_kps: (4, 2) 原图像素坐标
    area: GT 实例面积（像素²）

    公式：OKS = mean_k( exp(-d_k^2 / (2 * sigma^2 * area)) )
    注意 xtcocotools 里 vars = (sigma*2)^2 = 4*sigma^2，
    再除以 2 -> 等价于 2*sigma^2，与下面写法一致。
    """
    d2 = np.sum((np.asarray(pred_kps, np.float64)
                 - np.asarray(gt_kps, np.float64)) ** 2, axis=1)
    return float(np.mean(np.exp(-d2 / (2.0 * SIGMA ** 2 * area))))


def box_iou(a, b):
    """a: (4,) xyxy;  b: (N,4) xyxy -> (N,) IoU"""
    x1 = np.maximum(a[0], b[:, 0])
    y1 = np.maximum(a[1], b[:, 1])
    x2 = np.minimum(a[2], b[:, 2])
    y2 = np.minimum(a[3], b[:, 3])
    iw = np.clip(x2 - x1, 0, None)
    ih = np.clip(y2 - y1, 0, None)
    inter = iw * ih
    aa = (a[2] - a[0]) * (a[3] - a[1])
    bb = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / np.maximum(aa + bb - inter, 1e-9)


def perturb(box, dx, dy, scale):
    """把框按 (dx,dy) 平移、按 scale 缩放（绕框中心）。"""
    cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
    w = (box[2] - box[0]) * scale / 2
    h = (box[3] - box[1]) * scale / 2
    return np.array([cx - w + dx, cy - h + dy, cx + w + dx, cy + h + dy])


# ================================================================== #
class PoseOnBoxes:
    """把「一张图 + 一批框」喂进姿态模型，返回每个框的四角（原图坐标）。"""

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
            return []
        init_default_scope('mmpose')
        data_list = []
        for i in range(len(boxes)):
            di = dict(img=img)
            di['bbox'] = np.asarray(boxes[i:i + 1], dtype=np.float32)
            di['bbox_score'] = np.ones(1, dtype=np.float32)
            di.update(self.meta)
            data_list.append(self.pipeline(di))
        res = self.model.test_step(pseudo_collate(data_list))
        out = []
        for r in res:
            pi = r.pred_instances
            kp = pi.keypoints
            kp = kp.detach().cpu().numpy() if hasattr(kp, 'detach') \
                else np.asarray(kp)
            out.append(kp[0, :, :2].astype(np.float64))
        return out


# ================================================================== #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--det-config', default='config_det.py')
    ap.add_argument('--det-ckpt', default=None)
    ap.add_argument('--pose-config', default='config.py')
    ap.add_argument('--pose-ckpt', default=None,
                    help='默认自动找 work_dirs/dmkpd/best_coco_AP*.pth 中'
                         'mtime 最新的')
    ap.add_argument('--score-thr', type=float, default=0.30)
    ap.add_argument('--match-iou', type=float, default=0.5,
                    help='检测框与 GT 框匹配的 IoU 阈值')
    ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--skip-detect', action='store_true')
    ap.add_argument('--no-cache', action='store_true')
    args = ap.parse_args()

    if args.det_ckpt is None:
        import glob
        c = sorted(glob.glob('work_dirs/dmkpd_det/best_coco_bbox_mAP*.pth'))
        args.det_ckpt = c[-1] if c else None

    print('=' * 78)
    print('P1：检测框 vs GT 框 对姿态精度的影响')
    print('=' * 78)
    print(f'  检测权重  {args.det_ckpt}')
    print(f'  姿态权重  {args.pose_ckpt}')
    print(f'  匹配 IoU  {args.match_iou}   conf 阈值 {args.score_thr}')

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

    # ---------- 数据集（按图分组） ---------- #
    from mmpose.datasets.datasets.DM.registry import DMKPD_dataset
    ds = DMKPD_dataset(
        data_root='DMKPD_dataset', ann_file='splits/val.txt',
        data_prefix=dict(img='train/images'), metainfo=cfg_p.metainfo,
        serialize_data=False)
    n = args.limit or len(ds)
    by_img = defaultdict(list)
    for i in range(n):
        info = ds.get_data_info(i)
        by_img[info['img_id']].append((i, info))
    img_ids = sorted(by_img)
    n_inst = sum(len(v) for v in by_img.values())
    print(f'  实例 {n_inst} 个 / 图 {len(img_ids)} 张')

    runner = PoseOnBoxes(pose, cfg_p)

    # ---------- Phase 1：检测 + 匹配 ---------- #
    print()
    print('=' * 78)
    print('Phase 1  检测-标注匹配')
    print('=' * 78)

    det_map = {}
    if not args.skip_detect and os.path.isfile(CACHE) and not args.no_cache:
        det_map = json.load(open(CACHE, encoding='utf-8'))
        print(f'  [缓存] 从 {CACHE} 读入 {len(det_map)} 张图的检测结果')
    if not det_map:
        t0 = time.time()
        for k, iid in enumerate(img_ids):
            info = by_img[iid][0][1]
            img = cv2.imread(info['img_path'])
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
            # 与 infer_e2e.py 同样的后处理顺序：先 clip 再过滤
            if len(bx):
                bx = bx.copy()
                bx[:, [0, 2]] = np.clip(bx[:, [0, 2]], 0, W)
                bx[:, [1, 3]] = np.clip(bx[:, [1, 3]], 0, H)
                w = bx[:, 2] - bx[:, 0]
                h = bx[:, 3] - bx[:, 1]
                keep = (sc >= args.score_thr) & (np.minimum(w, h) >= 12)
                bx, sc = bx[keep], sc[keep]
            det_map[iid] = dict(boxes=bx.tolist(), scores=sc.tolist())
            if (k + 1) % 50 == 0 or k == 0:
                el = time.time() - t0
                print(f'    [{k+1}/{len(img_ids)}] {el:.0f}s  '
                      f'ETA {el/(k+1)*(len(img_ids)-k-1):.0f}s')
        if not args.no_cache:
            json.dump(det_map, open(CACHE, 'w', encoding='utf-8'))
            print(f'  [缓存] 已写入 {CACHE}')

    n_gt = n_matched = n_fp = 0
    ious = []
    # (img_id, gt_order) -> (det_box, iou)
    match = {}
    for iid in img_ids:
        gts = by_img[iid]
        dets = np.asarray(det_map.get(iid, {}).get('boxes', []),
                          dtype=np.float64).reshape(-1, 4)
        n_gt += len(gts)
        used = set()
        for gi, (idx, info) in enumerate(gts):
            gb = np.asarray(info['bbox'][0], dtype=np.float64)
            if len(dets) == 0:
                continue
            # 贪心：对每个 GT 找 IoU 最大且未被占用的检测框
            best, bi = 0.0, -1
            for di, db in enumerate(dets):
                if di in used:
                    continue
                v = float(box_iou(gb, db[None])[0])
                if v > best:
                    best, bi = v, di
            if best >= args.match_iou:
                used.add(bi)
                match[(iid, gi)] = (dets[bi], best)
                n_matched += 1
                ious.append(best)
        n_fp += len(dets) - len(used)

    ious = np.array(ious) if ious else np.zeros(1)
    print()
    print(f'  GT 实例        {n_gt}')
    print(f'  成功匹配       {n_matched}  ({100*n_matched/max(1,n_gt):.1f}%)'
          f'   ← 这是端到端的**召回上限**')
    print(f'  未匹配(漏检)   {n_gt - n_matched}  '
          f'({100*(n_gt-n_matched)/max(1,n_gt):.1f}%)')
    print(f'  误检框(无对应) {n_fp}')
    if n_matched:
        print(f'  匹配框 IoU     中位 {np.median(ious):.3f}  '
              f'均值 {ious.mean():.3f}  '
              f'p05 {np.percentile(ious,5):.3f}  '
              f'p95 {np.percentile(ious,95):.3f}')

    # ---------- Phase 2：OKS 对比 ---------- #
    print()
    print('=' * 78)
    print('Phase 2  OKS 对比（同一实例，GT 框 vs 检测框）')
    print('=' * 78)

    oks_gt, oks_det = [], []
    oks_gt_all = []      # 全部 GT（未匹配的按 0 计，模拟真实端到端）
    oks_det_all = []
    per_inst = []
    t0 = time.time()

    for k, iid in enumerate(img_ids):
        gts = by_img[iid]
        info0 = gts[0][1]
        img = cv2.imread(info0['img_path'])
        gt_boxes = np.stack([np.asarray(g['bbox'][0], np.float64)
                             for _, g in gts])
        gt_kps = np.stack([np.asarray(g['keypoints'][0], np.float64)
                           for _, g in gts])
        areas = np.array([float(g['area']) for _, g in gts])

        # --- 姿态跑 GT 框（复现 0.9377 的条件）--- #
        pred_gt = runner(img, gt_boxes)

        # --- 姿态跑检测框（端到端真实条件）--- #
        m_boxes, m_gi = [], []
        for gi in range(len(gts)):
            if (iid, gi) in match:
                m_boxes.append(match[(iid, gi)][0])
                m_gi.append(gi)
        pred_det = runner(img, np.asarray(m_boxes)) if m_boxes else []

        # --- CSV--- #
        for gi in range(len(gts)):
            o_gt = oks(pred_gt[gi], gt_kps[gi], areas[gi])
            oks_gt.append(o_gt)
            oks_gt_all.append(o_gt)
            # 该实例若没被检出，端到端下直接是"未命中"
            if (iid, gi) in match:
                j = m_gi.index(gi)
                o_det = oks(pred_det[j], gt_kps[gi], areas[gi])
                o_gt_m = o_gt
            else:
                o_det = 0.0
                o_gt_m = o_gt
            oks_det.append(o_det)
            oks_det_all.append(o_det)
            per_inst.append(dict(
                img_id=iid, gi=gi, area=float(areas[gi]),
                side=float(np.sqrt(areas[gi])),
                iou=float(match[(iid, gi)][1]) if (iid, gi) in match else -1,
                oks_gt=float(o_gt_m), oks_det=float(o_det),
                matched=bool((iid, gi) in match)))

        if (k + 1) % 40 == 0 or k == 0:
            el = time.time() - t0
            print(f'    [{k+1}/{len(img_ids)}] {el:.0f}s  '
                  f'ETA {el/(k+1)*(len(img_ids)-k-1):.0f}s')

    og = np.array(oks_gt)
    od = np.array(oks_det)
    print()
    print(f'  实例数 {len(og)}')

    def stat(name, arr):
        return (f'  {name:<26} 均值 {arr.mean():.4f}  '
                f'中位 {np.median(arr):.4f}  '
                f'OKS>=0.5 {100*(arr>=0.5).mean():5.1f}%  '
                f'>=0.75 {100*(arr>=0.75).mean():5.1f}%  '
                f'>=0.9 {100*(arr>=0.9).mean():5.1f}%')

    print(stat('GT 框（现有评测条件）', og))
    print(stat('检测框（端到端真实）', od))
    print(f'  {"差值":<26} 均值 {od.mean()-og.mean():+.4f}  '
          f'中位 {np.median(od)-np.median(og):+.4f}  '
          f'>=0.5 {100*((od>=0.5).mean()-(og>=0.5).mean()):+5.1f}%  '
          f'>=0.75 {100*((od>=0.75).mean()-(og>=0.75).mean()):+5.1f}%  '
          f'>=0.9 {100*((od>=0.9).mean()-(og>=0.9).mean()):+5.1f}%')

    # 只在"成功匹配"的子集上比（隔离出"框偏移"这一个变量）
    sub = [p for p in per_inst if p['matched']]
    if sub:
        a = np.array([p['oks_gt'] for p in sub])
        b = np.array([p['oks_det'] for p in sub])
        print()
        print(f'  【隔离变量】只看成功匹配的 {len(sub)} 个实例')
        print(stat('  GT 框', a))
        print(stat('  检测框', b))
        print(f'  {"  差值（纯框偏移影响）":<26} '
              f'均值 {b.mean()-a.mean():+.4f}  '
              f'中位 {np.median(b)-np.median(a):+.4f}')

    # ---------- Phase 3：灵敏度扫描 ---------- #
    print()
    print('=' * 78)
    print('Phase 3  灵敏度扫描（人为扰动 GT 框，模拟"更好的检测器"）')
    print('=' * 78)
    print('  目的：回答「把检测器训到 24 epoch，姿态 AP 能回来多少」')
    print()
    pert = [(0.00, 0.00, 1.00), (0.02, 0.02, 1.00), (0.05, 0.05, 1.00),
            (0.10, 0.10, 1.00), (0.05, 0.05, 1.10), (0.00, 0.00, 1.20)]
    print(f'  {"扰动(dx,dy,scale)":<22}{"平均IoU":>9}{"OKS均值":>10}'
          f'{"中位":>9}{">=0.75":>9}{">=0.9":>8}')
    # 用子样本加速
    step = max(1, len(img_ids) // 60)
    sub_ids = img_ids[::step]
    for (rx, ry, sc) in pert:
        rows = []
        for iid in sub_ids:
            gts = by_img[iid]
            info0 = gts[0][1]
            img = cv2.imread(info0['img_path'])
            H, W = img.shape[:2]
            gb = np.stack([np.asarray(g['bbox'][0], np.float64)
                           for _, g in gts])
            gk = np.stack([np.asarray(g['keypoints'][0], np.float64)
                           for _, g in gts])
            ar = np.array([float(g['area']) for _, g in gts])
            pb = np.stack([perturb(b, rx * np.sqrt(a), ry * np.sqrt(a), sc)
                           for b, a in zip(gb, ar)])
            pb[:, [0, 2]] = np.clip(pb[:, [0, 2]], 0, W)
            pb[:, [1, 3]] = np.clip(pb[:, [1, 3]], 0, H)
            pr = runner(img, pb)
            for j in range(len(gts)):
                rows.append((float(box_iou(gb[j], pb[j][None])[0]),
                             oks(pr[j], gk[j], ar[j])))
        r = np.array(rows)
        print(f'  {str((rx, ry, sc)):<22}{r[:,0].mean():>9.3f}'
              f'{r[:,1].mean():>10.4f}{np.median(r[:,1]):>9.4f}'
              f'{100*(r[:,1]>=0.75).mean():>8.1f}%'
              f'{100*(r[:,1]>=0.9).mean():>7.1f}%')

    # ---------- 保存 ---------- #
    json.dump(dict(
        n_gt=n_gt, n_matched=n_matched, n_fp=n_fp,
        iou_median=float(np.median(ious)), iou_mean=float(ious.mean()),
        oks_gt_mean=float(og.mean()), oks_det_mean=float(od.mean()),
        per_inst=per_inst),
        open('_gap_result.json', 'w', encoding='utf-8'), indent=1)
    print()
    print(f'  明细 -> _gap_result.json')

    # ---------- Phase 4 ---------- #
    print()
    print('=' * 78)
    print('Phase 4  结论')
    print('=' * 78)
    rec = n_matched / max(1, n_gt)
    print(f'  ① 检测召回 {100*rec:.1f}%  '
          f'-> 端到端时 {(1-rec)*100:.1f}% 的码**根本没机会被姿态看到**')
    if sub:
        d = b.mean() - a.mean()
        print(f'  ② 框偏移导致的 OKS 损失（隔离变量）= {d:+.4f}')
        if abs(d) < 0.01:
            print('     -> 姿态对框偏移**不敏感**；瓶颈在检测召回，不在框精度')
        else:
            print('     -> 框偏移影响明显，检测框精度值得优化')
    print(f'  ③ 端到端综合 OKS 均值 {od.mean():.4f} '
          f'(vs GT 框条件 {og.mean():.4f}，差 {od.mean()-og.mean():+.4f})')
    print()
    print('  优化优先级建议：')
    if rec < 0.90:
        print('    ★ 优先提升**检测召回**（训更多 epoch / 降 conf 阈值 /')
        print('      调 NMS IoU）—— 漏检是端到端最大损失来源')
    else:
        print('    ★ 检测召回已足够，可转向**框精度**（TTA / 框外扩）')
    print()

    json.dump(dict(n_gt=n_gt, n_matched=n_matched, n_fp=n_fp,
                   recall=rec,
                   oks_gt=og.mean(), oks_det=od.mean()),
              open('_gap_summary.json', 'w', encoding='utf-8'), indent=1)


if __name__ == '__main__':
    main()
