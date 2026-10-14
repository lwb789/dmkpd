# -*- coding: utf-8 -*-
"""端到端推理：检测 -> 逐个 DM 码裁切 -> 姿态回归四角 -> 导出比赛格式。

===================================================================
数据流（这正是比赛提交所要求的「原图像素坐标」）
===================================================================
  原图 (1702x1276)
    │
    ├─[检测器 RTMDet-tiny]──────────────────────────────────────┐
    │   Resize(1024, keep_ratio) + Pad -> 1024x1024             │
    │   预测框 -> 用 scale_factor 还原到原图坐标                  │
    │   输出: N 个 bbox (x1,y1,x2,y2) 原图坐标, + score          │
    ↓                                                            │
    └─[姿态 HRNet-W48 heatmap]  逐个 bbox 裁切 ─────────────────┤
        GetBBoxCenterScale(padding=1.5)                         │
        -> TopdownAffine(256x256)                               │
        -> 模型前向 -> 4 通道 64x64 热力图                        │
        -> codec.decode -> 256 画布上的 4 角点坐标                 │
        -> add_pred_to_datasample 用 input_center/input_scale    │
           把坐标反变换回**原图像素坐标**  ← 关键，mmpose 自动做了 │
    ↓                                                            │
  输出: N 个实例, 每个 4 个角点(原图坐标) + conf

===================================================================
conf 怎么取（与 COCO AP 口径一致的乘法分解）
===================================================================
    conf = bbox_score * mean(keypoint_scores)

  - `bbox_score`: 检测器给的框置信度（该框真的是个 DM 码的概率）
  - `keypoint_scores`: 4 个角点热力图峰值的均值（四角都定位得准的概率）

  乘法符合概率语义（「是个码」且「四角都准」），也与 COCO 评测里
  「检测框分数用于实例匹配、关键点分数用于 OKS」的分解一致。
  官方 mmpose 的 `TopDownAffineInferencer` 只把 bbox_score 当 conf，
  但比赛要求单值 conf，且 conf<0.3 会被剔除 —— 用乘法更保守、更安全。

===================================================================
用法
===================================================================
    # 先看几个样本的对不对（画图）
    python infer_e2e.py --pose-ckpt work_dirs/dmkpd/best_coco_AP_epoch_12.pth \
                        --det-ckpt  work_dirs/dmkpd_det/best_coco_bbox_mAP_epoch_60.pth \
                        --vis 12

    # 正式导出提交文件
    python infer_e2e.py --pose-ckpt ... --det-ckpt ... \
                        --out submission --export
"""
import argparse
import os
import os.path as osp
import time

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

KP_COLORS = [(0, 0, 255), (0, 255, 0), (255, 0, 0), (0, 255, 255)]


def _np(x):
    """统一把 Tensor / ndarray 转成 numpy ndarray。"""
    if hasattr(x, 'cpu'):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def postprocess_dets(bx, bs, W, H, score_thr=0.3, min_side=12,
                     max_per_img=100):
    """检测后处理：裁剪到图内 -> 过滤退化框 -> 按分数取前 N。

    ❗这里有个**必须先 clip 再过滤**的顺序要求（实测踩过）：
      推理管线是 `Resize(1024, keep_ratio) + Pad`。1702x1276 的图会先
      缩到 1024x767，**下方 pad 256 行**再补成 1024x1024。
      检测器会在「图像与 pad 的交界」产生虚假框，坐标还原后会落到
      原图的 y>1276（图外）。若先按尺寸过滤再 clip，这些框
      `[223, 1323, 535, 1700]` 尺寸看着正常（高 377），clip 后却
      退化成 `[223, 1276, 535, 1276]`（高 0）—— 于是漏进了提交文件，
      变成「图外的假码」。正确做法是 **clip 之后再重新算尺寸过滤**。
    """
    if len(bx) == 0:
        return bx, bs
    bx = bx.copy()
    bx[:, [0, 2]] = np.clip(bx[:, [0, 2]], 0, W)
    bx[:, [1, 3]] = np.clip(bx[:, [1, 3]], 0, H)
    w = bx[:, 2] - bx[:, 0]
    h = bx[:, 3] - bx[:, 1]
    keep = (bs >= score_thr) & (np.minimum(w, h) >= min_side)
    if not keep.any():
        return bx[:0], bs[:0]
    order = np.argsort(-bs[keep])[:max_per_img]
    return bx[keep][order], bs[keep][order]


# ------------------------------------------------------------------ #
# 检测
# ------------------------------------------------------------------ #
def build_detector(cfg_path, ckpt):
    cfg = Config.fromfile(cfg_path)
    init_default_scope('mmdet')
    model = DET_MODELS.build(cfg.model).cuda().eval()
    load_checkpoint(model, ckpt, map_location='cpu', strict=False)
    # ❗`mmdet.apis.inference_detector` 会读 `model.cfg.test_dataloader
    #   .dataset.pipeline` 来构造预处理管线；不挂 cfg 会报
    #   AttributeError: 'RTMDet' object has no attribute 'cfg'
    model.cfg = cfg
    return model, cfg


# ------------------------------------------------------------------ #
# 姿态
# ------------------------------------------------------------------ #
def build_pose(cfg_path, ckpt):
    cfg = Config.fromfile(cfg_path)
    init_default_scope('mmpose')
    model = POSE_MODELS.build(cfg.model).cuda().eval()
    if ckpt:
        load_checkpoint(model, ckpt, map_location='cpu', strict=False)
    # 官方 `inference_topdown` 就是这么做的：把 test_dataloader 的
    # pipeline 存到 model.cfg 上，供推理时复用
    model.cfg = cfg
    return model, cfg


class PoseRunner:
    """批量把检测框喂进姿态模型，拿回**原图像素坐标**的四角。

    ❗为什么直接用官方 pipeline + `model.test_step`，而不是自己
      手工还原坐标：
      1. 预处理完全一致（BGR->RGB、/255、减均值除标准差），
         不会有"我漏了 /255"这类静默错误；
      2. 坐标反变换由 `TopdownPoseEstimator.add_pred_to_datasample`
         用 metainfo 里的 `input_center` / `input_scale` / `input_size`
         自动完成，直接把 256 画布坐标映射回**原图像素坐标**
         （实现见 models/pose_estimators/topdown.py:152-156）：
             kp = kp / input_size * input_scale + input_center - 0.5*input_scale
      3. `bbox_score` 会被自动搬进 `pred_instances.bbox_scores`，
         不用手动凑 `gt_instances`。

    配置里 test_pipeline 是：
        LoadImage -> GetBBoxCenterScale(padding=1.5) -> TopdownAffine(256)
        -> PackPoseInputs
    与训练时的裁切完全同构（训练只是多了一次 RandomBBoxTransform）。
    """

    def __init__(self, model, cfg, padding=1.5):
        self.model = model
        self.pipeline = Compose(cfg.test_pipeline)
        # pipeline 里的 PackPoseInputs 需要 num_keypoints / sigmas 等
        # dataset_meta（与数据集类走同一套解析，保证口径一致）
        from mmpose.datasets.datasets.utils import parse_pose_metainfo
        self.meta = parse_pose_metainfo(dict(
            from_file='open-mmlab/mmpose/configs/_base_/datasets/'
                      'DMKPD_meta.py'))

    @torch.no_grad()
    def __call__(self, img_bgr, bboxes, bbox_scores):
        """bboxes: (N,4) xyxy 原图像素; 返回 list of dict(每实例)。"""
        if len(bboxes) == 0:
            return []
        # ❗全局默认作用域会被检测器改成 mmdet，必须显式设回 mmpose，
        #   否则 pipeline 里的 PackPoseInputs 会去 mmdet 注册表找，报
        #   KeyError: 'PackPoseInputs is not in the mmdet::transform registry'
        init_default_scope('mmpose')
        data_list = []
        for i in range(len(bboxes)):
            di = dict(img=img_bgr)
            di['bbox'] = bboxes[i:i + 1].astype(np.float32)   # (1,4)
            di['bbox_score'] = np.array([bbox_scores[i]],
                                        dtype=np.float32)     # (1,)
            di.update(self.meta)
            data_list.append(self.pipeline(di))
        batch = pseudo_collate(data_list)
        results = self.model.test_step(batch)
        out = []
        for res in results:
            pi = res.pred_instances
            # keypoints 可能是 Tensor 也可能是 ndarray，统一转 numpy
            kps = _np(pi.keypoints)[0, :, :2]     # 原图像素坐标
            scs = _np(pi.keypoint_scores)[0]
            out.append(dict(keypoints=kps.astype(np.float32),
                            keypoint_scores=scs.astype(np.float32)))
        return out


# ------------------------------------------------------------------ #
def _quad_metrics(p):
    """返回 (是否凸, 最小叉积, 面积)。"""
    v = np.roll(p, -1, axis=0) - p
    v2 = np.roll(v, -1, axis=0)
    cr = v[:, 0] * v2[:, 1] - v[:, 1] * v2[:, 0]
    convex = bool(np.all(cr > 0) or np.all(cr < 0))
    x, y = p[:, 0], p[:, 1]
    area = abs(0.5 * (float(np.dot(x, np.roll(y, -1)))
                      - float(np.dot(y, np.roll(x, -1)))))
    return convex, float(cr.min()), float(area)


def sanitize_quad(kps, W, H):
    """把四角修正成「图内 + 凸 + 顺时针 + TL 起点」的合法四边形。

    ❗为什么需要这一步（两次实测数据）：
      · 1 epoch 检测器：437 个预测里有 **8 个角点越界 1~8px**
        （贴边的真实码），**1 个是自相交的蝴蝶结**。
      · 21 epoch 检测器：411 个预测里 **1 个非凸**（`004353` 第 2 个实例，
        两个角只差 **3.7px**，几乎重合）。

      这些都不影响 AP（OKS 只比较点的位置，且评测方会 normalize
      点的顺序），但会让提交文件在严格的格式校验下不合法。

    处理策略（**只在病理情况下动手，正常样本一个像素都不改**）：
      1. 坐标 clip 到 [0, W-1] x [0, H-1]。
      2. 若四边形非凸：**枚举 4 个点的全部 24 种排列**，取
         「凸 + 面积最大」的那个作为解，并统一为顺时针、TL 起点。

    ❗为什么不能只用「绕质心极角排序」（我第一版的做法，实测不够）：
      极角排序**只在点集处于凸位置时**才能还原凸多边形顺序。
      当两个点几乎重合（本例相差 3.7px）时，点集退化成三角形，
      **任何排列都构不成凸四边形** —— 此时无解，只能退回近似。
      这种情况是姿态模型的真实失败（等于只预测出 3 个不同位置），
      OKS 本来就会很低，与丢弃等价。

    Returns:
        (kps, repaired, fixed): 修正后的 (4,2)、是否触发修复、
                                是否真正修成了凸四边形
    """
    out = np.clip(kps, [0, 0], [W - 1, H - 1]).astype(np.float64)
    convex, _, _ = _quad_metrics(out)
    if convex:
        return out.astype(np.float32), False, True

    import itertools

    def _canon(p):
        """统一为顺时针 + TL(x+y 最小) 起点。"""
        x, y = p[:, 0], p[:, 1]
        if 0.5 * (float(np.dot(x, np.roll(y, -1)))
                  - float(np.dot(y, np.roll(x, -1)))) < 0:
            p = p[::-1]
        s = p[:, 0] + p[:, 1]
        st = int(np.lexsort((p[:, 0], s))[0])
        return np.roll(p, -st, axis=0)

    best = None          # 凸解：按面积最大
    best_any = None      # 兜底：任意解里面积最大
    for perm in itertools.permutations(range(4)):
        p = out[list(perm)]
        cv, mincr, area = _quad_metrics(p)
        if best_any is None or area > best_any[1]:
            best_any = (_canon(p), area)
        if cv and (best is None or area > best[1]):
            best = (_canon(p), area)

    if best is not None:
        return best[0].astype(np.float32), True, True
    # 退化（点数重合）：无凸解，退回面积最大的排列，仅作外观修正
    return best_any[0].astype(np.float32), True, False


def draw(img, bboxes, bsc, poses, title=''):
    vis = img.copy()
    for i, (b, s) in enumerate(zip(bboxes, bsc)):
        cv2.rectangle(vis, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])),
                      (0, 200, 255), 3)
        cv2.putText(vis, f'{s:.2f}', (int(b[0]) + 4, int(b[1]) + 26),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 4)
        cv2.putText(vis, f'{s:.2f}', (int(b[0]) + 4, int(b[1]) + 26),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 200, 255), 2)
    for p in poses:
        k = p['keypoints']
        for j in range(4):
            cv2.circle(vis, tuple(np.round(k[j]).astype(int)), 11,
                       KP_COLORS[j], -1)
            cv2.circle(vis, tuple(np.round(k[j]).astype(int)), 11,
                       (255, 255, 255), 2)
        for j in range(4):
            a, b2 = k[j], k[(j + 1) % 4]
            cv2.line(vis, tuple(np.round(a).astype(int)),
                     tuple(np.round(b2).astype(int)), (255, 255, 0), 3)
    if title:
        cv2.putText(vis, title, (14, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.3,
                    (0, 0, 0), 7)
        cv2.putText(vis, title, (14, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.3,
                    (255, 255, 255), 3)
    return vis


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--det-config', default='config_det.py')
    ap.add_argument('--det-ckpt', default=None)
    ap.add_argument('--pose-config', default='config.py')
    ap.add_argument('--pose-ckpt', default=None,
                    help='默认自动找 work_dirs/dmkpd/best_coco_AP*.pth中'
                         '**mtime 最新**的那个；旧权重会被挪进 _old_best/，'
                         '所以这里按 mtime 取而不是按名字排序的最后一个')
    ap.add_argument('--img-dir', default='DMKPD_dataset/test')
    ap.add_argument('--score-thr', type=float, default=0.30)
    ap.add_argument('--no-nms-thr', type=float, default=0.65)
    ap.add_argument('--max-per-img', type=int, default=100)
    ap.add_argument('--out', default='submission')
    ap.add_argument('--vis', type=int, default=0,
                    help='可视化前 N 张（不导出）')
    ap.add_argument('--export', action='store_true')
    ap.add_argument('--limit', type=int, default=0)
    args = ap.parse_args()

    import glob
    import os.path as _osp

    if args.det_ckpt is None:
        # 自动找 best
        c = sorted(glob.glob('work_dirs/dmkpd_det/best_coco_bbox_mAP*.pth'))
        if not c:
            c = sorted(glob.glob('work_dirs/dmkpd_det/epoch_*.pth'))
        args.det_ckpt = c[-1] if c else None

    if args.pose_ckpt is None:
        # ⚠️ 按 **mtime** 取最新，而不是按文件名排序。
        #   因为旧权重会被挪进 `_old_best/`（不在 glob 范围内），
        #   但若将来有多个 best 并存（如 ep5 与 ep12），
        #   靠名字排序会选错（`epoch_12` < `epoch_5` 的字典序陷阱）。
        c = glob.glob('work_dirs/dmkpd/best_coco_AP*.pth')
        if not c:
            c = glob.glob('work_dirs/dmkpd/epoch_*.pth')
        args.pose_ckpt = (max(c, key=_osp.getmtime) if c else None)

    print('det  ckpt:', args.det_ckpt)
    print('pose ckpt:', args.pose_ckpt)
    if args.pose_ckpt is None or not _osp.isfile(args.pose_ckpt):
        raise SystemExit(f'❌ 找不到姿态权重：{args.pose_ckpt}\n'
                         '   检查 work_dirs/dmkpd/ 下有没有 best_coco_AP*.pth')

    det, det_cfg = build_detector(args.det_config, args.det_ckpt)
    pose, pose_cfg = build_pose(args.pose_config, args.pose_ckpt)
    runner = PoseRunner(pose, pose_cfg, padding=1.5)

    files = sorted(f for f in os.listdir(args.img_dir)
                   if f.lower().endswith('.jpg'))
    if args.limit:
        files = files[:args.limit]
    print(f'共 {len(files)} 张测试图')

    if args.export:
        os.makedirs(args.out, exist_ok=True)

    t0 = time.time()
    stats = dict(n_det=0, n_keep=0, n_img=0, confs=[], n_repair=0,
                 n_repair_ok=0, n_degenerate=0)
    vis_rows = []

    for idx, fn in enumerate(files):
        p = osp.join(args.img_dir, fn)
        img = cv2.imread(p)
        H, W = img.shape[:2]

        # ---- 检测 ---- #
        # ❗同理：姿态那边会把默认作用域设成 mmpose，这里要设回 mmdet
        init_default_scope('mmdet')
        det_res = inference_detector(det, img)
        pi = det_res.pred_instances
        # inference_detector 返回的 bboxes 可能是 Tensor 或 HorizontalBoxes
        bx = _np(pi.bboxes.tensor if hasattr(pi.bboxes, 'tensor')
                 else pi.bboxes)
        # 单类，第 0 列就是分数
        bs = _np(pi.scores)
        # Post-processing 命令：clip -> 过滤退化框 -> 取前 N
        # （顺序很关键，见 postprocess_dets 的注释）
        bx, bs = postprocess_dets(bx, bs, W, H,
                                  score_thr=args.score_thr,
                                  max_per_img=args.max_per_img)

        stats['n_det'] += len(bx)

        # ---- 姿态 ---- #
        poses = runner(img, bx, bs) if len(bx) else []
        recs = []
        for b, s, po in zip(bx, bs, poses):
            kp, repaired, fixed = sanitize_quad(po['keypoints'], W, H)
            stats['n_repair'] += int(repaired)
            stats['n_repair_ok'] += int(repaired and fixed)
            stats['n_degenerate'] += int(repaired and not fixed)
            ksc = po['keypoint_scores']
            conf = float(s) * float(np.mean(ksc))
            recs.append((b, s, kp, ksc, conf))

        kept = [r for r in recs if r[4] >= args.score_thr]
        stats['n_keep'] += len(kept)
        stats['n_img'] += 1
        stats['confs'].extend([r[4] for r in kept])

        # ---- 导出 ---- #
        if args.export:
            lines = []
            for _, _, kp, _, conf in kept:
                xy = ' '.join(f'{v:.2f}' for v in kp.reshape(-1))
                lines.append(f'DPM {conf:.4f} {xy}')
            with open(osp.join(args.out, osp.splitext(fn)[0] + '.txt'),
                      'w', encoding='utf-8') as f:
                f.write('\n'.join(lines) + ('\n' if lines else ''))

        # ---- 可视化 ---- #
        if args.vis and idx < args.vis:
            t = f'{fn}  det={len(bx)} kept={len(kept)}'
            v = draw(img, bx, bs, [dict(keypoints=r[2],
                                        keypoint_scores=r[3]) for r in kept],
                     t)
            sc = 700 / max(v.shape[:2])
            v = cv2.resize(v, None, fx=sc, fy=sc)
            vis_rows.append(v)

        if (idx + 1) % 20 == 0 or idx == 0:
            el = time.time() - t0
            eta = el / (idx + 1) * (len(files) - idx - 1)
            print(f'  [{idx+1}/{len(files)}] {fn}  det={len(bx)} '
                  f'keep={len(kept)}  elapsed {el:.0f}s  ETA {eta:.0f}s')

    dt = time.time() - t0
    print()
    print('=' * 78)
    print(f'完成 {stats["n_img"]} 张，用时 {dt:.1f}s '
          f'({dt/max(1,stats["n_img"]):.2f}s/张)')
    print(f'  检测框总数   {stats["n_det"]}')
    print(f'  通过 conf>={args.score_thr}  {stats["n_keep"]} '
          f'({100*stats["n_keep"]/max(1,stats["n_det"]):.1f}%)')
    print(f'  平均 {stats["n_keep"]/max(1,stats["n_img"]):.2f} 个/图')
    print(f'  四边形修复   {stats["n_repair"]} 个'
          f' ({100*stats["n_repair"]/max(1,stats["n_keep"]):.2f}%)'
          f'  其中修成凸 {stats["n_repair_ok"]} 个'
          f' / 退化无法修 {stats["n_degenerate"]} 个')
    if stats['n_degenerate']:
        print('       注：退化（角点几乎重合）是姿态模型的真实失败，')
        print('           OKS 本就很低，与丢弃等价，不影响评分')
    if stats['confs']:
        c = np.array(stats['confs'])
        print(f'  conf 分布  p05={np.percentile(c,5):.3f} '
              f'中位={np.median(c):.3f} p95={np.percentile(c,95):.3f} '
              f'min={c.min():.3f} max={c.max():.3f}')
    if args.export:
        print(f'  -> 已导出到 {osp.abspath(args.out)}')

    if vis_rows:
        n = len(vis_rows)
        cols = 3
        rows = (n + cols - 1) // cols
        ch = max(v.shape[0] for v in vis_rows)
        cw = max(v.shape[1] for v in vis_rows)
        grid = np.full((rows * ch, cols * cw, 3), 18, np.uint8)
        for i, v in enumerate(vis_rows):
            r, c = i // cols, i % cols
            grid[r * ch:r * ch + v.shape[0], c * cw:c * cw + v.shape[1]] = v
        cv2.imwrite('vis_out2/e2e_samples.jpg', grid)
        print(f'  可视化 -> vis_out2/e2e_samples.jpg  '
              f'({grid.shape[1]}x{grid.shape[0]})')


if __name__ == '__main__':
    main()
