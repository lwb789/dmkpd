# -*- coding: utf-8 -*-
"""放大看端到端预测质量：把每个检测框区域裁出来放大，叠加预测四角。

用**预测四角的外接矩形**来裁（而不是检测框），这样能直接看出
四角预测是否贴合 DM 码的真实边缘。
"""
import glob
import os.path as osp
import sys

import cv2
import numpy as np

sys.path.insert(0, '.')
import importlib.util
spec = importlib.util.spec_from_file_location('ie', 'infer_e2e.py')
ie = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ie)   # __name__ != '__main__'，不会跑 main()

from mmengine.registry import init_default_scope

c = sorted(glob.glob('work_dirs/dmkpd_det/best_coco_bbox_mAP*.pth'))
DET_CKPT = c[-1]
POSE_CKPT = 'work_dirs/dmkpd/best_coco_AP_epoch_5.pth'
print('det :', DET_CKPT)
print('pose:', POSE_CKPT)

init_default_scope('mmdet')
det, det_cfg = ie.build_detector('config_det.py', DET_CKPT)
init_default_scope('mmpose')
pose, pose_cfg = ie.build_pose('config.py', POSE_CKPT)
runner = ie.PoseRunner(pose, pose_cfg, padding=1.5)

CELL = 620
files = sorted(glob.glob('DMKPD_dataset/test/*.jpg'))[:4]
rows = []
for fn in files:
    img = cv2.imread(fn)
    H, W = img.shape[:2]
    init_default_scope('mmdet')
    res = ie.inference_detector(det, img)
    pi = res.pred_instances
    bx = ie._np(pi.bboxes.tensor if hasattr(pi.bboxes, 'tensor')
                else pi.bboxes)
    bs = ie._np(pi.scores)
    order = np.argsort(-bs)[:4]
    bx, bs = bx[order], bs[order]
    init_default_scope('mmpose')
    poses = runner(img, bx, bs) if len(bx) else []

    panels = []
    for i, (b, s, po) in enumerate(zip(bx, bs, poses)):
        kp = po['keypoints']
        x1 = max(0, int(min(b[0], kp[:, 0].min())) - 30)
        x2 = min(W, int(max(b[2], kp[:, 0].max())) + 30)
        y1 = max(0, int(min(b[1], kp[:, 1].min())) - 30)
        y2 = min(H, int(max(b[3], kp[:, 1].max())) + 30)
        crop = np.ascontiguousarray(img[y1:y2, x1:x2])
        k = kp - np.array([x1, y1])
        for j in range(4):
            p = tuple(np.round(k[j]).astype(int))
            cv2.circle(crop, p, 9, ie.KP_COLORS[j], -1)
            cv2.circle(crop, p, 9, (255, 255, 255), 2)
            cv2.putText(crop, str(j), (p[0] + 14, p[1] - 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 3)
            cv2.putText(crop, str(j), (p[0] + 14, p[1] - 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, ie.KP_COLORS[j], 2)
        for j in range(4):
            cv2.line(crop, tuple(np.round(k[j]).astype(int)),
                     tuple(np.round(k[(j + 1) % 4]).astype(int)),
                     (255, 255, 0), 2)
        h, w = crop.shape[:2]
        if h < 4 or w < 4:
            print(f'  跳过退化框 {osp.basename(fn)} #{i}: '
                  f'bbox={np.round(b,1).tolist()} kp={np.round(kp,1).tolist()}')
            continue
        sc = min(CELL / w, CELL / h)
        rs = cv2.resize(crop, (max(1, int(w * sc)), max(1, int(h * sc))),
                        interpolation=cv2.INTER_AREA)
        cell = np.full((CELL, CELL, 3), 20, np.uint8)
        oy, ox = (CELL - rs.shape[0]) // 2, (CELL - rs.shape[1]) // 2
        cell[oy:oy + rs.shape[0], ox:ox + rs.shape[1]] = rs
        bar = np.full((38, CELL, 3), 45, np.uint8)
        cv2.putText(bar,
                    f'{osp.basename(fn)} #{i} det={s:.3f} '
                    f'meanK={np.mean(po["keypoint_scores"]):.3f}',
                    (8, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.62,
                    (255, 255, 255), 2)
        panels.append(np.vstack([bar, cell]))
    while len(panels) < 4:
        panels.append(np.full((CELL + 38, CELL, 3), 20, np.uint8))
    rows.append(np.hstack(panels[:4]))

grid = np.vstack(rows)
cv2.imwrite('vis_out2/e2e_zoom.jpg', grid, [cv2.IMWRITE_JPEG_QUALITY, 92])
print('-> vis_out2/e2e_zoom.jpg', grid.shape[1], 'x', grid.shape[0])
