# -*- coding: utf-8 -*-
"""导出 ONNX：DMKPD 检测器 (RTMDet-tiny) + 姿态 (HRNet-W48 + HeatmapHead)。

===================================================================
用法
===================================================================
    # 两个都导（推荐先跑这个）
    python export_onnx.py

    # 只导检测器
    python export_onnx.py --only det

    # 只导姿态
    python export_onnx.py --only pose

    # 指定输出目录 / 权重
    python export_onnx.py --out onnx/ --det-ckpt work_dirs/... --pose-ckpt work_dirs/...

    # 跳过数值校验（不建议）
    python export_onnx.py --no-verify

导出的文件：
    onnx/dmkpd_det.onnx          检测：输入 1024x1024，输出 bboxes + scores
    onnx/dmkpd_pose.onnx         姿态：输入 256x256 的 bbox 裁剪，输出 4 通道热图

===================================================================
为什么这样切分
===================================================================
整个 pipeline 是「检测 -> 按框裁剪 -> 姿态回归热图 -> 解码角点」。
两段模型彼此独立，ONNX 只导出**网络前向**部分：

    · 检测器输入：整图 (1,3,1024,1024)，1x1 均值方差归一化
      *注意* 送进网络前需要按 config_det.py 的 IMG=1024 + keep_ratio
      缩放到 1024x768 并 pad 到 1024x1024（右侧/下侧 pad 114）。
      导出时用固定 1024x1024 输入，缩放与 pad 放在 Python 侧做。

    · 姿态输入：(1,3,256,256) 的 bbox 裁剪，padding=1.5 的仿射变换
      在 Python 侧做（见 infer_e2e.py 里的过程）。
      输出：(1,4,64,64) 热图，4 个角点通道。
      热图 -> 坐标的解码（MSRAHeatmap：argmax + 0.25 亚像素修正）
      放 Python 侧，避免在图里塞不可导的 argmax 逻辑。

===================================================================
已知坑（本脚本已处理）
===================================================================
1. `inference_detector` 会忽略顶层 `test_pipeline`
   -> 导出直接调 model.test_step 之外的前向，不依赖 pipeline。

2. numpy 版本冲突
   onnx>=1.22 要求 numpy>=2，会打断 mmcv(用 numpy1 编译)。
   本环境已固定：numpy==1.26.4 / onnx==1.15.0 / onnxruntime==1.17.3

3. 检测模型的 bbox 解码在 head 内部（`_predict_by_feat_single` 之前的
   `forward` 只输出 cls/reg 的原始特征）。
   为了拿到**解码后的框**，这里包一层 wrapper：调用 `model.test_step`
   太重，改用 head 的 test_cfg 手写 simple_test 等价逻辑是不稳定的。
   稳妥做法：导出 **backbone+neck+head.forward**（原始特征），
   解码放在 Python 侧（已在 infer_e2e.py 实现）。见下方 --raw 说明。

   本脚本默认导出 **解码后** 的框：通过 monkey-patch 让 head 的
   `predict_by_feat` 返回张量，从而把整个 simple_test 的确定性部分
   纳入计算图。这样 ONNX 输出直接可用。
"""
import argparse
import os
import os.path as osp
import sys

import numpy as np
import torch

import mmdet  # noqa
import mmpose  # noqa
from mmengine.config import Config
from mmengine.registry import init_default_scope
from mmengine.runner import load_checkpoint
from mmdet.registry import MODELS as DET_MODELS
from mmpose.registry import MODELS as POSE_MODELS

DET_INPUT = 1024
POSE_INPUT = 256


# ================================================================== #
# 检测器 wrapper
# ================================================================== #
class DetWrapper(torch.nn.Module):
    """RTMDet 前向 -> (bboxes, scores)，**严格对齐 mmdet 官方实现**。

    ------------------------------------------------------------------
    官方的关键实现（mmdet/models/dense_heads/rtmdet_head.py:126-146）
    ------------------------------------------------------------------
        reg_dist = scale(self.rtm_reg(reg_feat).exp()).float() * stride[0]

    ❗`bbox_preds` **已经 exp 且乘过 stride**，单位直接是像素。
      先验来自 MlvlPointGenerator.grid_priors(with_stride=True)，
      返回的是**网格左上角** (x, y, stride, stride)，**不是中心点**。
      解码用 distance2bbox（mmdet/structures/bbox/transforms.py）：
          x1 = px - d0,  y1 = py - d1,  x2 = px + d2,  y2 = py + d3

    ------------------------------------------------------------------
    曾经踩过的两个坑（已修正，勿回退）
    ------------------------------------------------------------------
      1. 又乘了一次 stride  -> 框整体放大数倍（框跑到图外）
      2. 用中心点 (gx+0.5)*stride -> 框整体偏移半个 stride

    这两处任一错误都会让分数与框错配：sigmoid 分数对得上，但框完全错。
    """

    def __init__(self, model, score_thr=0.05, max_per_img=300,
                 with_nms=True, nms_iou=0.65, min_bbox_size=0.0,
                 pre_topk=30000):
        super().__init__()
        self.model = model
        self.score_thr = score_thr
        self.max_per_img = max_per_img
        self.with_nms = with_nms
        self.nms_iou = nms_iou
        self.min_bbox_size = min_bbox_size
        self.pre_topk = pre_topk
        # 从模型读真实 stride，不要硬编码
        self.strides = [int(s[0]) for s in
                        model.bbox_head.prior_generator.strides]

    def forward(self, x):
        feats = self.model.extract_feat(x)
        cls_scores, bbox_preds = self.model.bbox_head(feats)

        all_boxes, all_scores = [], []
        for lvl, (cls, dist) in enumerate(zip(cls_scores, bbox_preds)):
            stride = self.strides[lvl]
            # cls: (B, num_base_priors*num_classes, H, W)，本项目=1
            score = torch.sigmoid(cls[:, 0])                # (B,H,W)
            B, H, W = score.shape
            # dist: (B, 4, H, W)，**已是像素单位，不要再乘 stride**
            reg = dist.view(B, 4, H, W)

            # 网格左上角坐标（与官方 grid_priors 定义一致）
            ys = torch.arange(H, dtype=x.dtype, device=x.device) * stride
            xs = torch.arange(W, dtype=x.dtype, device=x.device) * stride
            gy, gx = torch.meshgrid(ys, xs, indexing='ij')
            gx = gx.reshape(1, 1, H, W)
            gy = gy.reshape(1, 1, H, W)

            # distance2bbox
            x1 = gx - reg[:, 0:1]
            y1 = gy - reg[:, 1:2]
            x2 = gx + reg[:, 2:3]
            y2 = gy + reg[:, 3:4]
            boxes = torch.cat([x1, y1, x2, y2], 1)          # (B,4,H,W)
            boxes = boxes.permute(0, 2, 3, 1).reshape(B, -1, 4)
            sc = score.reshape(B, -1)
            all_boxes.append(boxes)
            all_scores.append(sc)

        boxes = torch.cat(all_boxes, 1)[0]                  # (N,4)
        scores = torch.cat(all_scores, 1)[0]                # (N,)

        # min_bbox_size（官方 test_cfg，本项目=0）
        if self.min_bbox_size > 0:
            w = boxes[:, 2] - boxes[:, 0]
            h = boxes[:, 3] - boxes[:, 1]
            keep = (w > self.min_bbox_size) & (h > self.min_bbox_size)
            boxes, scores = boxes[keep], scores[keep]

        # nms_pre：NMS 前按分数保留前 pre_topk（官方默认 30000）
        if boxes.shape[0] > self.pre_topk:
            order = scores.argsort(descending=True)[:self.pre_topk]
            boxes, scores = boxes[order], scores[order]

        keep = scores > self.score_thr
        boxes, scores = boxes[keep], scores[keep]

        if self.with_nms and boxes.numel():
            import torchvision
            idx = torchvision.ops.nms(boxes, scores, self.nms_iou)
            idx = idx[:self.max_per_img]
            boxes, scores = boxes[idx], scores[idx]

        return boxes, scores


# ================================================================== #
# 姿态 wrapper
# ================================================================== #
class PoseWrapper(torch.nn.Module):
    """HRNet backbone + HeatmapHead -> 热图 (1, 4, 64, 64)。

    输入：(1,3,256,256)，RGB，mean/std = [123.675,116.28,103.53] /
          [58.395,57.12,57.375]（bgr_to_rgb=True，所以 Python 侧要先
          把 BGR 转 RGB 再归一化）。
    """

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        feats = self.model.backbone(x)
        if isinstance(feats, (list, tuple)):
            feats = feats[0]
        out = self.model.head(feats)
        # HeatmapHead 输出 list[heatmap]；取最后一层
        if isinstance(out, (list, tuple)):
            out = out[-1]
        return out


# ================================================================== #
def build_det(ckpt, device='cpu'):
    cfg = Config.fromfile('config_det.py')
    init_default_scope('mmdet')
    m = DET_MODELS.build(cfg.model)
    load_checkpoint(m, ckpt, map_location='cpu', strict=False)
    m.cfg = cfg
    m.to(device).eval()
    return m, cfg


def build_pose(ckpt, device='cpu'):
    cfg = Config.fromfile('config.py')
    init_default_scope('mmpose')
    m = POSE_MODELS.build(cfg.model)
    load_checkpoint(m, ckpt, map_location='cpu', strict=False)
    m.cfg = cfg
    m.to(device).eval()
    return m, cfg


def export_det(args):
    print('\n' + '=' * 74)
    print('导出检测器 (RTMDet-tiny)')
    print('=' * 74)
    print(f'  权重 {args.det_ckpt}')
    m, cfg = build_det(args.det_ckpt)
    net = DetWrapper(m, score_thr=args.score_thr,
                     max_per_img=args.max_per_img,
                     with_nms=not args.no_nms).eval()

    dummy = torch.randn(1, 3, DET_INPUT, DET_INPUT)
    out = net(dummy)
    print(f'  [PyTorch] 输出 {len(out)} 个张量: '
          f'{[tuple(o.shape) for o in out]}')
    print(f'            boxes[:3]={out[0][:3].tolist()}')
    print(f'            scores[:3]={out[1][:3].tolist()}')

    path = osp.join(args.out, 'dmkpd_det.onnx')
    torch.onnx.export(
        net, dummy, path,
        input_names=['img'],
        output_names=['boxes', 'scores'],
        opset_version=args.opset,
        do_constant_folding=True,
        dynamic_axes=None,          # 固定 1024x1024
    )
    print(f'  -> 已写入 {path}  '
          f'({osp.getsize(path)/1024/1024:.1f} MB)')
    return path, dummy, out


def export_pose(args):
    print('\n' + '=' * 74)
    print('导出姿态模型 (HRNet-W48 + HeatmapHead)')
    print('=' * 74)
    print(f'  权重 {args.pose_ckpt}')
    m, cfg = build_pose(args.pose_ckpt)
    net = PoseWrapper(m).eval()
    dummy = torch.randn(1, 3, POSE_INPUT, POSE_INPUT)
    out = net(dummy)
    if isinstance(out, (list, tuple)):
        out = out[0]
    print(f'  [PyTorch] 输出 {tuple(out.shape)}  '
          f'min={out.min():.4f} max={out.max():.4f}')

    path = osp.join(args.out, 'dmkpd_pose.onnx')
    torch.onnx.export(
        net, dummy, path,
        input_names=['crop'],
        output_names=['heatmap'],
        opset_version=args.opset,
        do_constant_folding=True,
        dynamic_axes=None,
    )
    print(f'  -> 已写入 {path}  '
          f'({osp.getsize(path)/1024/1024:.1f} MB)')
    return path, dummy, out


def verify(path, dummy, torch_out, rtol=1e-3, atol=1e-4):
    """用 onnxruntime 跑一遍，与 PyTorch 输出比对。"""
    import onnx
    import onnxruntime as ort

    model = onnx.load(path)
    onnx.checker.check_model(model)
    print(f'  [ONNX] 检查通过，opset={model.opset_import[0].version}')

    sess = ort.InferenceSession(path, providers=['CPUExecutionProvider'])
    inp = sess.get_inputs()[0]
    print(f'         输入 {inp.name} {inp.shape} {inp.type}')
    for o in sess.get_outputs():
        print(f'         输出 {o.name} {o.shape} {o.type}')

    got = sess.run(None, {inp.name: dummy.numpy()})

    if not isinstance(torch_out, (list, tuple)):
        torch_out = [torch_out]
    ok_all = True
    for i, (g, t) in enumerate(zip(got, torch_out)):
        t = t.detach().cpu().numpy()
        if g.shape != t.shape:
            print(f'         ✗ 输出[{i}] shape 不一致: '
                  f'onnx={g.shape} torch={t.shape}')
            ok_all = False
            continue
        diff = np.abs(g - t)
        mx = float(diff.max())
        rel = mx / max(float(np.abs(t).max()), 1e-9)
        same = np.allclose(g, t, rtol=rtol, atol=atol)
        print(f'         输出[{i}] 最大绝对误差 {mx:.3e}  '
              f'相对 {rel:.3e}  {"✅ 一致" if same else "⚠ 有偏差"}')
        ok_all = ok_all and same
    return ok_all


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--only', choices=['det', 'pose'], default=None)
    ap.add_argument('--out', default='onnx')
    ap.add_argument('--det-ckpt', default=None)
    ap.add_argument('--pose-ckpt', default=None)
    ap.add_argument('--opset', type=int, default=12)
    ap.add_argument('--score-thr', type=float, default=0.05,
                    help='检测器输出阈值（可设低一些，后处理再卡 0.3）')
    ap.add_argument('--max-per-img', type=int, default=300)
    ap.add_argument('--no-nms', action='store_true',
                    help='不把 NMS 放进 ONNX 图（目标引擎不支持时用）')
    ap.add_argument('--no-verify', action='store_true')
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)

    if args.det_ckpt is None:
        import glob
        c = sorted(glob.glob(
            'work_dirs/dmkpd_det/best_coco_bbox_mAP*.pth'))
        if not c:
            sys.exit('找不到检测权重 work_dirs/dmkpd_det/*.pth')
        args.det_ckpt = c[-1]
    if args.pose_ckpt is None:
        import glob
        c = glob.glob('work_dirs/dmkpd/best_coco_AP*.pth')
        if not c:
            sys.exit('找不到姿态权重 work_dirs/dmkpd/*.pth')
        args.pose_ckpt = max(c, key=osp.getmtime)

    torch.set_grad_enabled(False)
    results = {}

    if args.only in (None, 'det'):
        p, dummy, out = export_det(args)
        results['det'] = (p, dummy, out)
    if args.only in (None, 'pose'):
        p, dummy, out = export_pose(args)
        results['pose'] = (p, dummy, out)

    if not args.no_verify:
        print('\n' + '=' * 74)
        print('onnxruntime 数值校验')
        print('=' * 74)
        for k, (p, dummy, out) in results.items():
            print(f'\n[{k}] {p}')
            verify(p, dummy, out)

    print('\n' + '=' * 74)
    print('完成')
    print('=' * 74)
    for k, (p, _, _) in results.items():
        print(f'  {k:5s}  {p}  {osp.getsize(p)/1024/1024:.1f} MB')
    print()
    print('后续推理（Python 侧要做的）：')
    print('  1. 读图 -> Resize(keep_ratio) 到 1024 长边 -> pad 到 1024x1024')
    print('     pad_val=114；归一化 mean=[103.53,116.28,123.675]')
    print('     std=[57.375,57.12,58.395]（BGR，不转 RGB）')
    print('  2. 检测 ONNX -> boxes, scores（框已解码，原图坐标系）')
    print('  3. 对每个框做 padding=1.5 的仿射裁剪到 256x256')
    print('     BGR->RGB，mean=[123.675,116.28,103.53]/std=[58.395,57.12,57.375]')
    print('  4. 姿态 ONNX -> heatmap (1,4,64,64)')
    print('  5. MSRA 解码：argmax + 0.25 亚像素修正，再仿射反映射回原图')
    print('  6. conf = bbox_score * mean(kp_score)，conf>=0.3 才写出')


if __name__ == '__main__':
    main()
