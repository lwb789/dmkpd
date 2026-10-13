# -*- coding: utf-8 -*-
"""判决性实验：姿态模型在「训练时见过的 holdout」与「从没见过的 val-1702」
上，精度有区别吗？

===================================================================
为什么要做这个实验
===================================================================
事实：`DMKPD_dataset/splits/train.txt`(9111) **包含** holdout1702 的 400 张，
而姿态模型就是用 `train.txt` 训的。

于是有两种可能，结论完全不同：
  (A) 姿态模型**记住了**这 400 张 → `_ap_holdout.py` 的 AP 被高估
  (B) 姿态模型**没有记住**（热量回归看不到明确监督，数据只当增广）
      → 泄漏对姿态 AP 几乎无影响

**不能用「holdout AP vs val AP」直接比** —— 两者分布不同
（val 83.9% 是 512x512 对抗图，holdout 100% 是 1702x1276 现场图）。

必须在**同一分布**上比：
   · 组 A：holdout 的 400 张现场图（训练时**见过**）
   · 组 B：val 里的 1702 档 123 张现场图（训练时**没见过**）
两组同为 1702x1276、同经 Resize(1024)、同用 GT 框，
**唯一差别就是有没有在训练集里**。

用法：
    python _leak_test.py
"""
import json
import os
import os.path as osp
from collections import defaultdict

import cv2
import numpy as np
import torch
from mmengine.config import Config
from mmengine.dataset import Compose, pseudo_collate
from mmengine.registry import init_default_scope
from mmengine.runner import load_checkpoint

import mmpose  # noqa
from mmpose.registry import MODELS as POSE_MODELS

SIGMA = 0.1


class PoseOnGT:
    def __init__(self, model, cfg):
        self.model = model
        self.pipeline = Compose(cfg.test_pipeline)
        from mmpose.datasets.datasets.utils import parse_pose_metainfo
        self.meta = parse_pose_metainfo(dict(
            from_file='open-mmlab/mmpose/configs/_base_/datasets/'
                      'DMKPD_meta.py'))

    @torch.no_grad()
    def __call__(self, img, bboxes):
        init_default_scope('mmpose')
        dl = []
        for i in range(len(bboxes)):
            di = dict(img=img,
                      bbox=np.asarray(bboxes[i:i + 1], np.float32),
                      bbox_score=np.ones(1, np.float32))
            di.update(self.meta)
            dl.append(self.pipeline(di))
        res = self.model.test_step(pseudo_collate(dl))
        out = []
        for r in res:
            kp = r.pred_instances.keypoints
            kp = kp.detach().cpu().numpy() if hasattr(kp, 'detach') \
                else np.asarray(kp)
            out.append(kp[0, :, :2])
        return np.asarray(out)


def read_split(p):
    s = set()
    for l in open(p, encoding='utf-8'):
        l = l.strip().replace('\\', '/')
        if l and not l.startswith('#'):
            s.add(osp.splitext(osp.basename(l))[0])
    return s


def collect(split, data_root='DMKPD_dataset',
            ann_dir='train/annotations', img_dir='train/images'):
    """按图收集 GT（keypoints / bbox / area / 原图尺寸）。"""
    from mmpose.datasets.datasets.DM.registry import DMKPD_dataset
    cfg = Config.fromfile('config.py')
    ds = DMKPD_dataset(
        data_root=data_root, ann_file=split,
        data_prefix=dict(img=img_dir), metainfo=cfg.metainfo,
        serialize_data=False)
    by = defaultdict(list)
    for i in range(len(ds)):
        info = ds.get_data_info(i)
        by[info['img_id']].append(info)
    return cfg, by


def run(cfg, by, runner, tag):
    oks_all, epe_all, rel_all, sizes = [], [], [], []
    for iid, gts in by.items():
        img = cv2.imread(gts[0]['img_path'])
        bx = np.stack([np.asarray(g['bbox'][0], np.float64) for g in gts])
        gk = np.stack([np.asarray(g['keypoints'][0], np.float64)
                       for g in gts])
        ar = np.array([float(g['area']) for g in gts])
        pred = runner(img, bx)
        for i in range(len(gts)):
            d2 = ((pred[i] - gk[i]) ** 2).sum(1)
            oks_all.append(float(np.mean(np.exp(
                -d2 / (2 * SIGMA ** 2 * ar[i])))))
            epe_all.append(float(np.linalg.norm(pred[i] - gk[i], axis=1)
                                 .mean()))
            rel_all.append(float(np.linalg.norm(pred[i] - gk[i], axis=1)
                                 .mean() / np.sqrt(ar[i])))
            sizes.append(float(np.sqrt(ar[i])))
    return dict(tag=tag, n_img=len(by), n=len(oks_all),
                oks=np.array(oks_all), epe=np.array(epe_all),
                rel=np.array(rel_all), side=np.array(sizes))


def show(r):
    print(f'  {r["tag"]:<34}{r["n_img"]:>4} 图{r["n"]:>5} 实例'
          f'{r["oks"].mean():>10.4f}{np.median(r["oks"]):>9.4f}'
          f'{r["epe"].mean():>9.2f}{np.median(r["epe"]):>8.2f}'
          f'{r["rel"].mean():>10.4f}{np.median(r["side"]):>9.1f}')


def main():
    cfg, by_ho = collect('splits/holdout1702.txt')
    _, by_va = collect('splits/val.txt')

    # val 里只留 1702 档（与 holdout 同分布）
    import cv2 as _cv
    va_big = {}
    for iid, gts in by_va.items():
        im = _cv.imread(gts[0]['img_path'])
        if im is not None and max(im.shape[:2]) > 1400:
            va_big[iid] = gts
    print(f'holdout1702  {len(by_ho)} 图（训练时**见过**）')
    print(f'val 的 1702 档 {len(va_big)} 图（训练时**没见过**）')

    init_default_scope('mmpose')
    pose = POSE_MODELS.build(cfg.model).cuda().eval()
    load_checkpoint(pose, 'work_dirs/dmkpd/best_coco_AP_epoch_5.pth',
                    map_location='cpu', strict=False)
    pose.cfg = cfg
    runner = PoseOnGT(pose, cfg)

    r_ho = run(cfg, by_ho, runner, 'holdout（见过）')
    r_va = run(cfg, va_big, runner, 'val-1702（没见过）')

    print()
    print('=' * 92)
    print('姿态精度：见过 vs 没见过（GT 框条件，同为 1702x1276 现场图）')
    print('=' * 92)
    print(f'  {"组":<34}{"图":>6}{"实例":>7}{"OKS均值":>10}{"OKS中位":>9}'
          f'{"EPE均值":>9}{"EPE中位":>8}{"相对EPE均":>10}{"边长中位":>9}')
    show(r_ho)
    show(r_va)
    print()
    d_oks = r_ho['oks'].mean() - r_va['oks'].mean()
    print(f'  {"差值（见过 − 没见过）":<34}'
          f'{"":>6}{"":>7}{d_oks:>+10.4f}'
          f'{np.median(r_ho["oks"])-np.median(r_va["oks"]):>+9.4f}'
          f'{r_ho["epe"].mean()-r_va["epe"].mean():>+9.2f}'
          f'{np.median(r_ho["epe"])-np.median(r_va["epe"]):>+8.2f}')
    print()
    # 按尺寸分层，排除"两组尺寸分布不同"的干扰
    bins = [(0, 60), (60, 120), (120, 260), (260, 512), (512, 1e9)]
    print('  按码尺寸分层（若泄漏有害，见过组应系统性更好）：')
    print(f'  {"尺寸":<16}{"见过 n":>8}{"见过 OKS":>11}'
          f'{"没见过 n":>10}{"没见过 OKS":>12}{"差":>9}')
    for lo, hi in bins:
        m1 = (r_ho['side'] >= lo) & (r_ho['side'] < hi)
        m2 = (r_va['side'] >= lo) & (r_va['side'] < hi)
        if m1.sum() < 5 or m2.sum() < 5:
            continue
        a, b = r_ho['oks'][m1].mean(), r_va['oks'][m2].mean()
        print(f'  [{lo:>4},{hi if hi<1e9 else "inf":>4})'
              f'{m1.sum():>8}{a:>11.4f}{m2.sum():>10}{b:>12.4f}{a-b:>+9.4f}')
    print()
    print('  ⚠️ 判读要点：')
    print('     · 差值 ≈ 0        -> 姿态**没有记住**，泄漏对它无害（结论 B）')
    print('     · 见过组明显更好  -> 存在记忆，holdout AP 被高估（结论 A）')
    print('     · 注意两组样本量差 5x，OKS 均值本身有 ±0.01 量级波动')


if __name__ == '__main__':
    main()
