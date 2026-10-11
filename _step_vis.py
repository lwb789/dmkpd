"""对真实样本 001850 逐步展示 pipeline 每一步的产物（可视化 + 数值）。

001850 是 val 里 idx=22 的实例：
    原图 1702x1276，码 1434x549（边长 887px，长短边比 2.61）
    之前被列为"误差最大"的样本之一（maxEPE=1125px）

本脚本逐步执行 train_pipeline（**关闭随机增强**，便于对照），
把每一步的中间产物画出来，并输出数值。

产出：
    vis_out2/step_flow.jpg    主流程可视化
    _step_out.txt             每步数值
"""
import os

import cv2
import numpy as np
from mmengine.config import Config
from mmengine.registry import init_default_scope

init_default_scope('mmpose')

from mmpose.datasets.datasets.DM import DMKPD_dataset  # noqa: E402
from mmpose.registry import TRANSFORMS  # noqa: E402

IDX = 22
IN = 256
HMS = 64
CELL = 400
OUT = []

KP_COLORS = [(0, 0, 255), (0, 255, 0), (255, 0, 0), (0, 255, 255)]
KP_NAMES = ['0=TL', '1=TR', '2=BR', '3=BL']


def L(s=''):
    print(s)
    OUT.append(str(s))


def tp(p):
    return (int(round(float(p[0]))), int(round(float(p[1]))))


def fit_cell(im, cell=CELL, interp=cv2.INTER_LINEAR):
    """把图等比缩放进 cell x cell 的黑底方块。"""
    im = np.asarray(im)
    if im.ndim == 2:
        im = cv2.cvtColor(im, cv2.COLOR_GRAY2BGR)
    h, w = im.shape[:2]
    s = min(cell / h, cell / w)
    r = cv2.resize(im, (max(1, int(w * s)), max(1, int(h * s))),
                   interpolation=interp)
    canvas = np.full((cell, cell, 3), 18, np.uint8)
    y = (cell - r.shape[0]) // 2
    x = (cell - r.shape[1]) // 2
    canvas[y:y + r.shape[0], x:x + r.shape[1]] = r
    return canvas


def title_bar(texts, cell=CELL, h=30):
    b = np.zeros((h, cell * len(texts), 3), np.uint8)
    for i, t in enumerate(texts):
        cv2.putText(b, t, (i * cell + 6, 21), cv2.FONT_HERSHEY_SIMPLEX,
                    0.46, (255, 255, 255), 1)
    return b


def note_bar(text, width, h=34, color=(60, 60, 60)):
    b = np.full((h, width, 3), color, np.uint8)
    cv2.putText(b, text, (8, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (0, 255, 255), 1)
    return b


def draw_kpts(img, pts, r=6, label=True, link=True, tag='', tag_col=(255, 255, 255)):
    v = img.copy()
    if link:
        for i in range(len(pts)):
            cv2.line(v, tp(pts[i]), tp(pts[(i + 1) % len(pts)]),
                     (140, 140, 140), 2)
    for i, p in enumerate(pts):
        c = KP_COLORS[i]
        cv2.circle(v, tp(p), r, c, -1)
        cv2.circle(v, tp(p), r, (255, 255, 255), 2)
        if label:
            pos = (int(p[0]) + r + 3, int(p[1]) - r - 3)
            cv2.putText(v, str(i), pos, cv2.FONT_HERSHEY_SIMPLEX, 0.75,
                        (0, 0, 0), 4)
            cv2.putText(v, str(i), pos, cv2.FONT_HERSHEY_SIMPLEX, 0.75, c, 2)
    if tag:
        cv2.putText(v, tag, (6, v.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (0, 0, 0), 3)
        cv2.putText(v, tag, (6, v.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, tag_col, 1)
    return v


def heat_vis(hm, cell=CELL):
    h = np.asarray(hm, np.float32)
    if h.ndim == 3:
        h = h[0]
    if h.max() > 0:
        h = h / h.max()
    bgr = cv2.applyColorMap((h * 255).astype(np.uint8), cv2.COLORMAP_JET)
    return fit_cell(bgr, cell, interp=cv2.INTER_NEAREST)


def window_view(img, rects, colors, labels, pad=25):
    """把原图放到更大的画布上，叠加若干矩形（用于展示裁切窗口）。"""
    H, W = img.shape[:2]
    xs, ys = [0, W], [0, H]
    for (x0, y0, x1, y1) in rects:
        xs += [x0, x1]
        ys += [y0, y1]
    X0, X1 = int(min(xs)) - pad, int(max(xs)) + pad
    Y0, Y1 = int(min(ys)) - pad, int(max(ys)) + pad
    cw, ch = X1 - X0, Y1 - Y0
    canvas = np.full((ch, cw, 3), 35, np.uint8)
    ox, oy = -X0, -Y0
    canvas[oy:oy + H, ox:ox + W] = img
    # 画网格（每 200px 一条淡线，帮助判断尺度）
    for gx in range(0, cw, 200):
        cv2.line(canvas, (gx, 0), (gx, ch), (55, 55, 55), 1)
    for gy in range(0, ch, 200):
        cv2.line(canvas, (0, gy), (cw, gy), (55, 55, 55), 1)
    for (x0, y0, x1, y1), col, lab in zip(rects, colors, labels):
        p1 = (int(x0 - X0), int(y0 - Y0))
        p2 = (int(x1 - X0), int(y1 - Y0))
        cv2.rectangle(canvas, p1, p2, col, 3)
        if lab:
            ty = max(16, p1[1] - 8)
            cv2.putText(canvas, lab, (p1[0] + 4, ty),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3)
            cv2.putText(canvas, lab, (p1[0] + 4, ty),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, col, 1)
    return canvas


# =================================================================== #
os.makedirs('vis_out2', exist_ok=True)
cfg = Config.fromfile('config.py')
ds = DMKPD_dataset(
    data_root='DMKPD_dataset', ann_file='splits/val.txt',
    data_prefix=dict(img='train/images'), metainfo=cfg.metainfo,
    serialize_data=False)

info = ds.get_data_info(IDX)

L('=' * 84)
L('逐步拆解样本 001850（val idx=22）')
L('=' * 84)
L()
L('  [Step 0] 数据集解析出的原始 data_info（坐标 = 原图像素）')
L(f'    img_path     = {info["img_path"]}')
L(f'    img_id       = {info["img_id"]}')
L(f'    原图尺寸      = {info["width"]} x {info["height"]}')
L(f'    bbox (xyxy)  = {np.round(info["bbox"][0], 1).tolist()}')
for k in range(4):
    L(f'    keypoint[{k}]  = {np.round(info["keypoints"][0][k], 1).tolist()}'
      f'   ({KP_NAMES[k]})')
L(f'    group_id     = {info["group_id"]}')
L(f'    visibility   = {info["instance_visibility"]:.3f}')
L()

# ---- 逐步执行 ---- #
stages = [
    ('LoadImage', dict(type='LoadImage')),
    ('GetBBoxCenterScale', dict(type='GetBBoxCenterScale', padding=1.5)),
    ('RandomBBoxTransform(恒等)',
     dict(type='RandomBBoxTransform', shift_prob=0.0, scale_prob=0.0,
          rotate_prob=0.0, scale_factor=(1.0, 1.0))),
    ('TopdownAffine', dict(type='TopdownAffine', input_size=(IN, IN))),
    ('CheckKeypointOutOfBounds', dict(type='CheckKeypointOutOfBounds')),
    ('GenerateTarget',
     dict(type='GenerateTarget',
          encoder=dict(type='MSRAHeatmap', input_size=(IN, IN),
                       heatmap_size=(HMS, HMS), sigma=2),
          use_dataset_keypoint_weights=True)),
    ('PackPoseInputs', dict(type='PackPoseInputs')),
]

TRACK = ('img', 'bbox', 'bbox_center', 'bbox_scale', 'bbox_rotation',
         'input_size', 'input_center', 'input_scale', 'keypoints',
         'transformed_keypoints', 'keypoints_visible', 'heatmaps',
         'keypoint_weights')

trace = {}
r = dict(info)
for name, c in stages:
    r = TRANSFORMS.build(c)(r)
    snap = {}
    for k in TRACK:
        if k in r:
            v = r[k]
            if hasattr(v, 'numpy'):        # torch.Tensor
                v = v.numpy()
            if isinstance(v, np.ndarray):
                v = v.copy()
            snap[k] = v
    if isinstance(r, dict) and 'data_samples' in r:
        snap['inputs'] = r['inputs']
        snap['data_samples'] = r['data_samples']
    trace[name] = snap

    L(f'  [{name}] 得到:')
    for k in TRACK:
        if k not in r:
            continue
        v = r[k]
        if hasattr(v, 'numpy'):
            v = v.numpy()
        if isinstance(v, np.ndarray):
            if k == 'img':
                L(f'    {k:<22} shape={tuple(v.shape)} dtype={v.dtype}')
            else:
                L(f'    {k:<22} shape={str(tuple(v.shape)):<14} '
                  f'{np.round(v.ravel()[:8], 2).tolist()}')
        else:
            L(f'    {k:<22} {v}')
    L()

# =================================================================== #
# 绘图
# =================================================================== #
orig = trace['LoadImage']['img']
gt_k = trace['LoadImage']['keypoints'][0]
bbox = trace['LoadImage']['bbox'][0]

bc = trace['GetBBoxCenterScale']['bbox_center'][0]
bs = trace['GetBBoxCenterScale']['bbox_scale'][0]
bs_sq = trace['TopdownAffine']['bbox_scale'][0]
tk = trace['TopdownAffine']['transformed_keypoints'][0]
warped = trace['TopdownAffine']['img']

# --- (1) 原图 + bbox + GT 点 ---
p1 = draw_kpts(orig, gt_k, r=11, tag='Step0/1: original 1702x1276', tag_col=KP_COLORS[0])
cv2.rectangle(p1, tp(bbox[:2]), tp(bbox[2:]), (255, 255, 255), 4)
cv2.rectangle(p1, tp(bbox[:2]), tp(bbox[2:]), (0, 0, 255), 2)

# --- (2) 加 padding 窗口 ---
pad_rect = (bc[0] - bs[0] / 2, bc[1] - bs[1] / 2,
            bc[0] + bs[0] / 2, bc[1] + bs[1] / 2)
sq_rect = (bc[0] - bs_sq[0] / 2, bc[1] - bs_sq[1] / 2,
           bc[0] + bs_sq[0] / 2, bc[1] + bs_sq[1] / 2)
p2 = window_view(
    orig, [tuple(bbox), pad_rect],
    [(0, 0, 255), (0, 255, 0)],
    [f'GT bbox {bbox[2]-bbox[0]:.0f}x{bbox[3]-bbox[1]:.0f}',
     f'pad=1.5  window {bs[0]:.0f}x{bs[1]:.0f}'])
p2 = draw_kpts(p2, gt_k, r=9, label=False, link=False)

# --- (3) 加正方形窗口 ---
p3 = window_view(
    orig, [tuple(bbox), pad_rect, sq_rect],
    [(0, 0, 255), (0, 255, 0), (255, 128, 0)],
    [f'GT bbox {bbox[2]-bbox[0]:.0f}x{bbox[3]-bbox[1]:.0f}',
     f'pad=1.5  {bs[0]:.0f}x{bs[1]:.0f}',
     f'after TopdownAffine  {bs_sq[0]:.0f}x{bs_sq[1]:.0f}'])
p3 = draw_kpts(p3, gt_k, r=9, label=False, link=False)

# --- (4) 正方形窗口内实际裁到的内容 ---
X0 = int(max(0, sq_rect[0])); Y0 = int(max(0, sq_rect[1]))
X1 = int(min(orig.shape[1], sq_rect[2]))
Y1 = int(min(orig.shape[0], sq_rect[3]))
crop_full = np.full((int(sq_rect[3] - sq_rect[1]), int(sq_rect[2] - sq_rect[0]), 3),
                    0, np.uint8)
cx0 = int(X0 - sq_rect[0]); cy0 = int(Y0 - sq_rect[1])
crop_full[cy0:cy0 + (Y1 - Y0), cx0:cx0 + (X1 - X0)] = orig[Y0:Y1, X0:X1]
p4 = draw_kpts(crop_full, gt_k - np.array([sq_rect[0], sq_rect[1]]),
               r=max(6, int(crop_full.shape[0] / 90)), label=False)
p4 = cv2.resize(p4, (CELL, CELL))

# --- (5) TopdownAffine 输出 256x256 ---
p5 = draw_kpts(warped, tk, r=8,
               tag='Step4: after TopdownAffine 256x256',
               tag_col=(0, 255, 0))

# --- (6) 热力图 ---
hm = trace['GenerateTarget']['heatmaps']
if hm.ndim == 4:
    hm = hm[0]
h_cells = [heat_vis(hm[k]) for k in range(4)]
h_sum = heat_vis(hm.sum(axis=0))

# --- (7) inputs 张量 ---
inp = trace['PackPoseInputs']['inputs']
if hasattr(inp, 'numpy'):
    inp = inp.numpy()
inp_vis = inp.transpose(1, 2, 0)              # (3,H,W) -> (H,W,3)
if inp_vis.dtype != np.uint8:
    a = inp_vis.astype(np.float32)
    a = (a - a.min()) / max(1e-6, a.max() - a.min())
    inp_vis = (a * 255).astype(np.uint8)

# =================================================================== #
# 组装
# =================================================================== #
rows = []
rows.append(np.vstack([
    title_bar(['(1) 原始数据: 图 + GT bbox + 4 角点',
               '(2) Step2: GetBBoxCenterScale(padding=1.5)',
               '(3) Step4 准备: _fix_aspect_ratio 拉成正方形']),
    np.hstack([fit_cell(p1), fit_cell(p2), fit_cell(p3)]),
]))
rows.append(note_bar(
    f'GetBBoxCenterScale: bbox_scale = ({bs[0]:.0f}, {bs[1]:.0f})  (长方形)'
    f'  ->  TopdownAffine 用 _fix_aspect_ratio 拉成 ({bs_sq[0]:.0f}, {bs_sq[1]:.0f}) (正方形)',
    CELL * 3))

rows.append(np.vstack([
    title_bar(['(4) 正方形窗口内实际裁到的内容 (含黑边)',
               '(5) Step4: TopdownAffine 输出 256x256',
               '(6) Step4 关键点坐标: 原图 -> 256 画布']),
    np.hstack([fit_cell(p4), fit_cell(p5), fit_cell(p5)]),
]))

# 坐标映射说明行
mapping = np.full((CELL, CELL * 3, 3), 25, np.uint8)
cv2.putText(mapping, 'Coordinate mapping (same warp matrix for image & points)',
            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
cv2.putText(mapping, f'bbox_center = ({bc[0]:.1f}, {bc[1]:.1f})',
            (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)
cv2.putText(mapping, f'bbox_scale  = ({bs_sq[0]:.1f}, {bs_sq[1]:.1f})',
            (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)
cv2.putText(mapping, 'scale factor (orig -> 256) =',
            (10, 145), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)
cv2.putText(mapping, f'  {bs_sq[0]:.1f} / 256 = {bs_sq[0]/256:.3f}',
            (10, 175), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 1)
cv2.putText(mapping, 'code size after warp =',
            (10, 220), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)
cv2.putText(mapping,
            f'  {bbox[2]-bbox[0]:.0f} / {bs_sq[0]/256:.3f} = '
            f'{(bbox[2]-bbox[0])/(bs_sq[0]/256):.1f}px',
            (10, 250), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 1)
cv2.putText(mapping, '= 256 / 1.5 = 170.7px  (constant!)',
            (10, 280), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 1)
cv2.putText(mapping, '', (10, 320), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
            (200, 200, 200), 1)
cv2.putText(mapping, 'transformed_keypoints (in 256):',
            (10, 330), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)
for k in range(4):
    cv2.putText(mapping,
                f'  k{k}: {np.round(tk[k], 1).tolist()}',
                (10, 360 + k * 28), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                KP_COLORS[k], 1)

rows.append(np.vstack([
    title_bar(['(7) Step6: GenerateTarget -> 4 heatmaps 64x64',
               '(8) 热力图叠加 (sum)',
               '(9) Step6 坐标映射明细']),
    np.hstack([fit_cell(h_cells[0]), fit_cell(h_sum), fit_cell(mapping)]),
]))

rows.append(np.vstack([
    title_bar(['(10) heatmap 0=TL', '(11) heatmap 1=TR',
               '(12) heatmap 2=BR']),
    np.hstack([fit_cell(h_cells[0]), fit_cell(h_cells[1]),
               fit_cell(h_cells[2])]),
]))
rows.append(np.vstack([
    title_bar(['(13) heatmap 3=BL', '(14) 4 通道求和',
               '(15) Step7: PackPoseInputs 的 inputs (3,256,256)']),
    np.hstack([fit_cell(h_cells[3]), fit_cell(h_sum), fit_cell(inp_vis)]),
]))

canvas = np.vstack(rows)
cv2.imwrite('vis_out2/step_flow.jpg', canvas)
L(f'可视化 -> vis_out2/step_flow.jpg  ({canvas.shape[1]}x{canvas.shape[0]})')

# 再存一张热力图细节（放大看像素）
hm_big = np.hstack([heat_vis(hm[k], 320) for k in range(4)] +
                   [heat_vis(hm.sum(axis=0), 320)])
hm_big = np.vstack([
    title_bar(['heatmap[0]=top_left', 'heatmap[1]=top_right',
               'heatmap[2]=bottom_right', 'heatmap[3]=bottom_left',
               'sum of 4'], cell=320),
    hm_big])
cv2.imwrite('vis_out2/step_heatmap.jpg', hm_big)
L('热力图细节 -> vis_out2/step_heatmap.jpg')

# 热力图统计
nz = (hm > 0.01).sum()
L()
L('  [Step6] 热力图统计')
L(f'    形状              {hm.shape} = {hm[0].size} 像素/通道')
L(f'    非零(>0.01)像素    {nz} = {100.0 * nz / hm[0].size:.2f}%')
L(f'    峰值              {hm.max():.4f}')
L(f'    mean(GT^2)        {float((hm ** 2).mean()):.6f}  '
  f'<- 这就是"输出全零"时的 loss 基线')

with open('_step_out.txt', 'w', encoding='utf-8') as f:
    f.write('\n'.join(OUT))
