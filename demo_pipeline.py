"""逐阶段展示 DMKPD 的 top-down 数据流水线。

刻意**不**一次性跑完 Compose，而是把每个 stage 单独 apply，
这样每个阶段的中间产物（bbox_center/scale、warp 后的图与点、
热力图、pack 后的张量）都能被检查与可视化。

流程：
    LoadImage -> GetBBoxCenterScale -> [RandomBBoxTransform]
    -> TopdownAffine -> CheckKeypointOutOfBounds
    -> GenerateTarget(MSRAHeatmap) -> PackPoseInputs

产出：
    vis_out2/stage_val.jpg         无增强时各阶段可视化
    vis_out2/pipeline_aug_*.jpg    随机增强样例
    _pipeline_out.txt              各阶段数值
"""
import os

import cv2
import numpy as np
import torch
from mmengine.registry import init_default_scope

init_default_scope('mmpose')

from mmpose.datasets.datasets.DM import DMKPD_dataset  # noqa: E402
from mmpose.registry import TRANSFORMS  # noqa: E402

INPUT_SIZE = (256, 256)
HEATMAP_SIZE = (64, 64)
KP_NAMES = ['top_left', 'top_right', 'bottom_right', 'bottom_left']
KP_COLORS = [(0, 0, 255), (0, 255, 0), (255, 0, 0), (0, 255, 255)]
SCALE = INPUT_SIZE[0] / HEATMAP_SIZE[0]

META = dict(
    from_file='open-mmlab/mmpose/configs/_base_/datasets/DMKPD_meta.py')
ROOT = dict(
    data_root='DMKPD_dataset',
    ann_file='train/annotations',
    data_prefix=dict(img='train/images'),
    metainfo=META)

lines = []


def log(s=''):
    print(s)
    lines.append(s)


def build_stages(augment: bool, oob_out_dir=None):
    cfg = [('LoadImage', dict(type='LoadImage')),
           ('GetBBoxCenterScale',
            dict(type='GetBBoxCenterScale', padding=1.5))]
    if augment:
        cfg.append(
            ('RandomBBoxTransform',
             dict(
                 type='RandomBBoxTransform',
                 shift_factor=0.05,
                 shift_prob=0.3,
                 scale_factor=(0.9, 1.1),
                 scale_prob=1.0,
                 rotate_factor=30,
                 rotate_prob=1.0)))
    cfg += [
        ('TopdownAffine',
         dict(type='TopdownAffine', input_size=INPUT_SIZE, use_udp=False)),
        ('CheckKeypointOutOfBounds',
         dict(type='CheckKeypointOutOfBounds', out_dir=oob_out_dir)),
        ('GenerateTarget',
         dict(
             type='GenerateTarget',
             encoder=dict(
                 type='MSRAHeatmap',
                 input_size=INPUT_SIZE,
                 heatmap_size=HEATMAP_SIZE,
                 sigma=2),
             use_dataset_keypoint_weights=True)),
        ('PackPoseInputs', dict(type='PackPoseInputs')),
    ]
    return [(n, TRANSFORMS.build(c)) for n, c in cfg]


TRACK_KEYS = ('bbox_center', 'bbox_scale', 'bbox_rotation',
              'transformed_keypoints', 'keypoints', 'keypoints_visible',
              'heatmaps', 'keypoint_weights')


def run_stages(stages, results, record=False):
    trace = {}
    for name, t in stages:
        results = t(results)
        if results is None:
            log(f'  !! stage {name} 返回 None')
            break
        if record:
            snap = {'img': results.get('img')}
            for k in TRACK_KEYS:
                if k in results:
                    v = results[k]
                    if isinstance(v, torch.Tensor):
                        v = v.detach().cpu().numpy()
                    snap[k] = v.copy() if isinstance(v, np.ndarray) else v
            trace[name] = snap
    return results, trace


def show(name, results):
    log(f'--- [{name}] 之后 ---')
    for k in ['img', 'img_shape', 'ori_shape', 'bbox', 'bbox_center',
              'bbox_scale', 'bbox_rotation', 'input_size', 'input_center',
              'input_scale', 'keypoints', 'transformed_keypoints',
              'keypoints_visible', 'heatmaps', 'keypoint_weights']:
        if k not in results:
            continue
        v = results[k]
        if isinstance(v, torch.Tensor):
            v = v.numpy()
        if isinstance(v, np.ndarray):
            log(f'  {k:<24} shape={str(v.shape):<12} {str(v.dtype):<8}'
                f' 前几个={np.round(v.ravel()[:6], 2).tolist()}')
        else:
            log(f'  {k:<24} {v}')
    log()


def draw_points(img, pts, labels=True, radius=4, link=True):
    vis = img.copy()
    if link:
        for i in range(len(pts)):
            cv2.line(vis, tuple(np.round(pts[i]).astype(int)),
                     tuple(np.round(pts[(i + 1) % len(pts)]).astype(int)),
                     (180, 180, 180), 1)
    for i, (x, y) in enumerate(pts):
        c = KP_COLORS[i % len(KP_COLORS)]
        cv2.circle(vis, (int(round(x)), int(round(y))), radius, c, -1)
        cv2.circle(vis, (int(round(x)), int(round(y))), radius,
                   (255, 255, 255), 1)
        if labels:
            cv2.putText(vis, str(i), (int(x) + 6, int(y) - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3)
            cv2.putText(vis, str(i), (int(x) + 6, int(y) - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, c, 1)
    return vis


def hm_bgr(hm):
    h = np.asarray(hm, dtype=np.float32)
    if h.ndim == 3:
        h = h[0]
    if h.max() > 0:
        h = h / h.max()
    return cv2.applyColorMap((h * 255).astype(np.uint8), cv2.COLORMAP_JET)


def fit(im, size=256):
    return cv2.resize(np.asarray(im, dtype=np.uint8), (size, size))


def grid(panels, titles, cols, path, cell=256):
    rows = []
    for i in range(0, len(panels), cols):
        chunk, ttl = panels[i:i + cols], titles[i:i + cols]
        while len(chunk) < cols:
            chunk.append(np.zeros((cell, cell, 3), np.uint8))
            ttl.append('')
        row = np.hstack([fit(p, cell) for p in chunk])
        bar = np.zeros((28, row.shape[1], 3), np.uint8)
        for j, t in enumerate(ttl):
            cv2.putText(bar, t, (j * cell + 5, 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1)
        rows.append(np.vstack([bar, row]))
    cv2.imwrite(path, np.vstack(rows))
    log(f'  已保存 {path}')


def unwrap_hm(hm):
    a = np.asarray(hm)
    return a[0] if a.ndim == 4 else a


def unwrap_1d(v):
    a = np.asarray(v)
    return a[0] if a.ndim == 2 else a


os.makedirs('vis_out2', exist_ok=True)

# ====================================================================== #
log('=' * 74)
log('第 1 步：无随机增强，逐 stage 检查（数值可手算核对）')
log('=' * 74)

ds = DMKPD_dataset(serialize_data=False, **ROOT)
log(f'数据集实例数 = {len(ds)}\n')

raw_info = ds.get_data_info(0)
log('原始 data_info（坐标是【原图】像素）:')
log(f'  img_path     = {raw_info["img_path"]}')
log(f'  img_id       = {raw_info["img_id"]}')
log(f'  width/height = {raw_info["width"]} x {raw_info["height"]}')
log(f'  bbox         = {np.round(raw_info["bbox"], 2).tolist()}')
log(f'  keypoints    = {np.round(raw_info["keypoints"], 2).tolist()}')
log(f'  group_id     = {raw_info["group_id"]}   '
    f'instance_visibility = {raw_info["instance_visibility"]:.4f}')
log()

stages_val = build_stages(augment=False)
results, trace = run_stages(stages_val, dict(raw_info), record=True)
for name, _ in stages_val:
    show(name, trace[name])

orig = cv2.imread(raw_info['img_path'])
gt_orig = raw_info['keypoints'][0]
s0 = 256.0 / orig.shape[0]
orig_small = cv2.resize(orig, (int(orig.shape[1] * s0), 256))

bc = trace['GetBBoxCenterScale']['bbox_center'][0]
bs = trace['GetBBoxCenterScale']['bbox_scale'][0]
win = orig.copy()
cv2.rectangle(win, (int(bc[0] - bs[0] / 2), int(bc[1] - bs[1] / 2)),
              (int(bc[0] + bs[0] / 2), int(bc[1] + bs[1] / 2)),
              (0, 255, 0), 6)
win = draw_points(win, gt_orig, radius=10)

warped = np.asarray(trace['TopdownAffine']['img'])[..., ::-1].copy()
tk = trace['TopdownAffine']['transformed_keypoints'][0]
hm = unwrap_hm(trace['GenerateTarget']['heatmaps'])

panels = [orig_small, draw_points(orig_small, gt_orig * s0, radius=3),
          cv2.resize(win, (int(orig.shape[1] * s0), 256)),
          draw_points(warped, tk),
          hm_bgr(hm[0]), hm_bgr(hm[1]), hm_bgr(hm[2]), hm_bgr(hm[3]),
          hm_bgr(hm.sum(axis=0))]
titles = [f'original {orig.shape[1]}x{orig.shape[0]}',
          'orig + GT (0=TL red)',
          f'crop window pad=1.5 ({bs[0]:.0f}px)',
          'after TopdownAffine 256x256',
          'heatmap[top_left]', 'heatmap[top_right]',
          'heatmap[bottom_right]', 'heatmap[bottom_left]',
          'sum of 4 heatmaps']
grid(panels, titles, cols=3, path='vis_out2/stage_val.jpg')
log()

log('--- 热力图峰值 vs TopdownAffine 后的关键点（同在 256 画布）---')
ok = True
for k in range(4):
    ix = np.unravel_index(np.argmax(hm[k]), hm[k].shape)
    peak = np.array([ix[1] * SCALE, ix[0] * SCALE])
    d = float(np.linalg.norm(peak - tk[k]))
    flag = 'OK' if d <= 4.0 else 'MISMATCH'
    ok &= (flag == 'OK')
    log(f'  {KP_NAMES[k]:<13} 峰值={np.round(peak, 1).tolist()}  '
        f'关键点={np.round(tk[k], 1).tolist()}  误差={d:.2f}px  [{flag}]')
log(f'  => {"全部一致" if ok else "存在不一致!"}')
log()

packed = results   # run_stages 的返回值即 PackPoseInputs 的输出
dsp = packed['data_samples']
log('--- PackPoseInputs 之后的最终产物 ---')
log(f'  inputs (C,H,W)      = {tuple(packed["inputs"].shape)}  '
    f'{packed["inputs"].dtype}  值域=[{packed["inputs"].min():.0f}, '
    f'{packed["inputs"].max():.0f}]')
log(f'  data_sample 类型    = {type(dsp).__name__}')
log(f'  gt_fields.heatmaps  = {tuple(dsp.gt_fields.heatmaps.shape)}')
log(f'  gt_instance_labels  = {sorted(dsp.gt_instance_labels.keys())}')
log(f'  gt_instances        = {sorted(dsp.gt_instances.keys())}')
log(f'  关键点权重          = '
    f'{dsp.gt_instance_labels.keypoint_weights.numpy().tolist()}')
log()
log('注: gt_instances.keypoints 是【原图坐标】，供评估使用；')
log('    256 画布上的坐标叫 transformed_keypoints，被 GenerateTarget')
log('    消费后不再打包进最终 data_sample（所以上面没列出）。')
log()

# ====================================================================== #
log('=' * 74)
log('第 2 步：随机增强 shift=0.05 / scale=(0.9,1.1) / rotate=30°')
log('=' * 74)

OOB_DIR = 'work_dirs/pipeline_demo'
if os.path.exists(os.path.join(OOB_DIR, 'oob_keypoints_log.txt')):
    os.remove(os.path.join(OOB_DIR, 'oob_keypoints_log.txt'))
stages_aug = build_stages(augment=True, oob_out_dir=OOB_DIR)

n_check = 300
cnt_oob_img = cnt_oob_kpt = bad_peak = cnt_zero_weight = 0
rng = np.random.default_rng(0)

for _ in range(n_check):
    info = ds.get_data_info(int(rng.integers(0, len(ds))))
    # 用 record=True 取 trace：PackPoseInputs 之后顶层键会被打包进
    # data_sample，顶层就不再是 pipeline 的中间结果了
    _, tr = run_stages(stages_aug, dict(info), record=True)

    hm_i = unwrap_hm(tr['GenerateTarget']['heatmaps'])
    kp = tr['TopdownAffine']['transformed_keypoints'][0]
    vis = unwrap_1d(tr['CheckKeypointOutOfBounds']['keypoints_visible'])
    wgt = unwrap_1d(tr['GenerateTarget']['keypoint_weights'])

    n_oob = int((vis < 0.5).sum())
    cnt_oob_img += int(n_oob > 0)
    cnt_oob_kpt += n_oob
    cnt_zero_weight += int((wgt == 0).any())

    for k in range(4):
        if vis[k] < 0.5:
            continue
        ix = np.unravel_index(np.argmax(hm_i[k]), hm_i[k].shape)
        if np.linalg.norm(
                np.array([ix[1] * SCALE, ix[0] * SCALE]) - kp[k]) > 4.0:
            bad_peak += 1

log(f'随机抽样 {n_check} 个实例（种子固定可复现）:')
log(f'  含出界角点实例数           = {cnt_oob_img}  '
    f'({100.0*cnt_oob_img/n_check:.2f}%)')
log(f'  出界角点总数               = {cnt_oob_kpt}')
log(f'  存在权重0关键点的实例数    = {cnt_zero_weight}  （应≈上一行实例数）')
log(f'  可见点与热力图峰值不符次数 = {bad_peak}  （应为 0）')
p = os.path.join(OOB_DIR, 'oob_keypoints_log.txt')
if os.path.exists(p):
    log(f'  出界明细日志行数           = '
        f'{sum(1 for _ in open(p, encoding="utf-8"))} '
        f'（含"近界"记录，故可能多于出界实例数）')
log()

for t in range(3):
    info = ds.get_data_info(100 + t)
    _, tr = run_stages(stages_aug, dict(info), record=True)
    im = np.asarray(tr['TopdownAffine']['img'])[..., ::-1].copy()
    kp = tr['TopdownAffine']['transformed_keypoints'][0]
    vis = unwrap_1d(tr['CheckKeypointOutOfBounds']['keypoints_visible'])
    hm_j = unwrap_hm(tr['GenerateTarget']['heatmaps'])
    rot = tr['RandomBBoxTransform']['bbox_rotation'][0]
    bsc = tr['RandomBBoxTransform']['bbox_scale'][0]

    vis_im = draw_points(im, kp)
    for k in range(4):
        if vis[k] < 0.5:
            cv2.drawMarker(vis_im, tuple(np.round(kp[k]).astype(int)),
                           (128, 128, 128), cv2.MARKER_TILTED_CROSS, 14, 2)

    ps = [vis_im] + [hm_bgr(hm_j[k]) for k in range(4)] + \
         [hm_bgr(hm_j.sum(axis=0))]
    ts = [f'aug rot={rot:.0f}deg grayX=OOB'] + KP_NAMES + ['sum']
    grid(ps, ts, cols=6, path=f'vis_out2/pipeline_aug_{t}.jpg', cell=190)
    log(f'  样例{t}: visible={np.round(vis, 1).tolist()}  '
        f'rotation={rot:.1f}°  bbox_scale={np.round(bsc, 1).tolist()}')

log()
log('全部完成。数值日志 -> _pipeline_out.txt')

with open('_pipeline_out.txt', 'w', encoding='utf-8') as f:
    f.write('\n'.join(lines))
