# -*- coding: utf-8 -*-
"""深挖「检测框正确、但姿态 OKS ≈ 0」的实例（AR 的真正天花板）。

`_ar_analyze.py` 发现 17 个 GT 的最佳 OKS < 0.5（占 3.34%），
且 **检测 miss = 0**（全部 17 个的最大检测 IoU 都 >= 0.79）。

其中最可疑的一组：**框分很高（0.91~0.98）但 OKS 恰为 0.0000**。
框是对的，姿态却给出 200+px 的错点 —— 这不像"模型不行"，
更像有系统性原因（贴边截断？大码缩放？透视？标注问题？）。

本脚本把这 5 个（以及全部 17 个）逐个摊开：
  · 该 GT 与**所有**检测框的 IoU 及其分数
  · 取 IoU 最大的那个框，单独跑姿态，打印预测四点 vs GT 四点
  · 逐点误差、归一化误差（这才是 OKS 的输入）
  · 检查窗口是否越出图像边界（padding=1.5）
  · 输出可视化到 vis_out2/

用法：
    python _ar_miss_detail.py
"""
import glob
import json
import os
import os.path as osp

import cv2
import numpy as np
from mmengine.config import Config
from mmengine.registry import init_default_scope
from mmengine.runner import load_checkpoint

import mmdet  # noqa
import mmpose  # noqa
from mmdet.registry import MODELS as DET_MODELS
from mmpose.registry import MODELS as POSE_MODELS

from _ap_holdout import SIGMA, PoseOnBoxes, oks_matrix

CACHE = '_ar_det_cache.json'
OUT = 'vis_out2'


def box_iou_one(gb, boxes):
    if len(boxes) == 0:
        return np.zeros(0)
    xx1 = np.maximum(gb[0], boxes[:, 0])
    yy1 = np.maximum(gb[1], boxes[:, 1])
    xx2 = np.minimum(gb[2], boxes[:, 2])
    yy2 = np.minimum(gb[3], boxes[:, 3])
    inter = np.clip(xx2 - xx1, 0, None) * np.clip(yy2 - yy1, 0, None)
    ga = max((gb[2] - gb[0]) * (gb[3] - gb[1]), 1e-9)
    ba = np.maximum(boxes[:, 2] - boxes[:, 0], 0) * \
        np.maximum(boxes[:, 3] - boxes[:, 1], 0)
    return inter / np.maximum(ga + ba - inter, 1e-9)


def main():
    res = json.load(open('_ar_analyze_result.json', encoding='utf-8'))
    miss = res['miss_det'] + res['miss_pose']
    ids = sorted({m['iid'] for m in miss})
    print(f'待分析 {len(ids)} 个实例（{len(miss)} 个 GT）')

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
                              key=osp.getmtime),
                    map_location='cpu', strict=False)
    pose.cfg = cfgp
    runner = PoseOnBoxes(pose, cfgp)

    # 从姿态数据集拿 GT（按图）
    from mmpose.datasets.datasets.DM.registry import DMKPD_dataset
    ds = DMKPD_dataset(data_root='DMKPD_dataset',
                       ann_file='splits/holdout1702.txt',
                       data_prefix=dict(img='train/images'),
                       metainfo=cfgp.metainfo, serialize_data=False)
    per_img = {}
    for i in range(len(ds)):
        info = ds.get_data_info(i)
        per_img.setdefault(info['img_id'], []).append(info)

    det_map = json.load(open(CACHE, encoding='utf-8'))
    os.makedirs(OUT, exist_ok=True)

    print()
    print('=' * 100)
    print('逐个实例：最佳 IoU 框单独跑姿态，看预测点偏到哪去了')
    print('=' * 100)

    for iid in ids:
        gts = per_img.get(iid)
        if not gts:
            continue
        img = cv2.imread(gts[0]['img_path'])
        H, W = img.shape[:2]
        dm = det_map.get(iid, dict(boxes=[], scores=[]))
        bx = np.asarray(dm['boxes'], np.float64).reshape(-1, 4)
        bs = np.asarray(dm['scores'], np.float64).reshape(-1)

        for gi, g in enumerate(gts):
            gk = np.asarray(g['keypoints'][0], np.float64)
            ga = float(g['area'])
            side = float(np.sqrt(ga))
            gb = np.array([gk[:, 0].min(), gk[:, 1].min(),
                           gk[:, 0].max(), gk[:, 1].max()])
            iou = box_iou_one(gb, bx)
            if len(iou) == 0:
                print(f'\n{iid} GT{gi} side={side:.0f}  ❗没有任何检测框')
                continue
            bi = int(iou.argmax())
            # 单独跑姿态（只这一个框，和端到端一致）
            b1 = bx[bi:bi + 1]
            kps, ksc = runner(img, b1)
            pk = np.asarray(kps[0], np.float64)
            d = np.linalg.norm(pk - gk, axis=1)
            oks = float(oks_matrix(kps, None, gk[None], np.array([ga]))[0, 0])
            norm = d / np.sqrt(ga)

            # padding=1.5 的采样窗口是否越界
            cx, cy = (b1[0, 0] + b1[0, 2]) / 2, (b1[0, 1] + b1[0, 3]) / 2
            s = max(b1[0, 2] - b1[0, 0], b1[0, 3] - b1[0, 1]) * 1.5
            x0, y0 = cx - s / 2, cy - s / 2
            over = max(0, -x0) + max(0, -y0) + max(0, x0 + s - W) + \
                max(0, y0 + s - H)
            trunc = over / s

            print()
            print('-' * 100)
            print(f'{iid}  GT{gi}   边长 {side:>6.1f}px   '
                  f'最佳框 IoU {iou[bi]:.3f}  框分 {bs[bi]:.4f}  '
                  f'kp分 {float(ksc[0]):.3f}   OKS {oks:.4f}')
            print(f'    采样窗口 {s:.0f}px  越界比例 {trunc*100:.1f}%   '
                  f'{"⚠️ 贴边/出界" if trunc > 0.02 else ""}')
            print(f'    {"点":<4}{"GT":>22}{"预测":>22}{"距离px":>10}'
                  f'{"归一化":>10}')
            order = np.argsort(-d)
            for j in order:
                print(f'    P{j:<3}({gk[j,0]:>9.1f},{gk[j,1]:>9.1f})'
                      f'({pk[j,0]:>9.1f},{pk[j,1]:>9.1f})'
                      f'{d[j]:>10.1f}{norm[j]:>10.4f}')
            print(f'    平均距离 {d.mean():.1f}px   最大 {d.max():.1f}px'
                  f'   （OKS 需要归一化误差远小于 sigma={SIGMA}）')

            ov = img.copy()
            cv2.polylines(ov, [gk.astype(np.int32).reshape(-1, 1, 2)],
                          True, (0, 200, 0), 4, cv2.LINE_AA)
            cv2.polylines(ov, [pk.astype(np.int32).reshape(-1, 1, 2)],
                          True, (0, 0, 255), 4, cv2.LINE_AA)
            for j in range(4):
                cv2.circle(ov, tuple(gk[j].astype(int)), 12, (0, 200, 0), -1)
                cv2.circle(ov, tuple(pk[j].astype(int)), 12, (0, 0, 255), -1)
            for j in range(4):
                cv2.line(ov, tuple(gk[j].astype(int)),
                         tuple(pk[j].astype(int)), (0, 180, 255), 3,
                         cv2.LINE_AA)
            cv2.putText(ov, f'GREEN=GT  RED=pred  OKS={oks:.3f}',
                        (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.6,
                        (0, 0, 0), 6, cv2.LINE_AA)
            cv2.putText(ov, f'GREEN=GT  RED=pred  OKS={oks:.3f}',
                        (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.6,
                        (255, 255, 255), 2, cv2.LINE_AA)
            p = osp.join(OUT, f'armiss_{iid}_g{gi}.jpg')
            cv2.imwrite(p, np.ascontiguousarray(ov))
            print(f'    -> {p}')


if __name__ == '__main__':
    main()
