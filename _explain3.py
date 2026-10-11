"""精确诊断：误差最大的实例到底是什么问题？

候选解释：
  A) 角点顺序混淆（预测的点对应到了错误的角）-> 用"最佳循环配对"检验
  B) 角点被裁出画面
  C) 长短边比过大导致有效像素浪费
"""
import io

import cv2
import numpy as np
from mmengine.config import Config
from mmengine.registry import init_default_scope

init_default_scope('mmpose')

from mmpose.apis import inference_topdown, init_model  # noqa: E402
from mmpose.datasets.datasets.DM import DMKPD_dataset  # noqa: E402
from mmpose.registry import TRANSFORMS  # noqa: E402

OUT = []


def L(s=''):
    print(s)
    OUT.append(str(s))


cfg = Config.fromfile('config.py')
model = init_model(cfg, 'work_dirs/dmkpd/best_coco_AP_epoch_5.pth',
                   device='cuda:0')
ds = DMKPD_dataset(
    data_root='DMKPD_dataset', ann_file='splits/val.txt',
    data_prefix=dict(img='train/images'), metainfo=cfg.metainfo,
    serialize_data=False)

BAD = [22, 100, 103, 21, 112, 145, 1096, 387]

recs = {}
for i in range(len(ds)):
    info = ds.get_data_info(i)
    r = inference_topdown(model, info['img_path'], bboxes=info['bbox'],
                          bbox_format='xyxy')
    pred = r[0].pred_instances.keypoints[0]
    gt = info['keypoints'][0]
    recs[i] = dict(pred=pred, gt=gt, info=info,
                   epe=np.linalg.norm(pred - gt, axis=1))

L('=' * 86)
L('Q2-精确诊断：误差最大的实例是什么问题？')
L('=' * 86)
L()
L(f'  {"idx":>5} {"img_id":<15} {"w x h":>12} {"比":>5} {"边长":>6} '
  f'{"逐点EPE (TL,TR,BR,BL)":>26} {"最佳循环配对":>14}')
L('  ' + '-' * 94)

for i in BAD:
    d = recs[i]
    info = d['info']
    x1, y1, x2, y2 = info['bbox'][0]
    w, h = x2 - x1, y2 - y1
    side = np.sqrt(info['area'])
    gt, pred = d['gt'], d['pred']
    best = min(
        (s, np.linalg.norm(np.roll(gt, s, axis=0) - pred, axis=1).mean())
        for s in range(4))
    L(f'  {i:>5} {info["img_id"]:<15} {w:>5.0f}x{h:<5.0f} '
      f'{max(w, h) / max(min(w, h), 1):>5.2f} {side:>6.0f} '
      f'{str(np.round(d["epe"], 0).astype(int).tolist()):<26} '
      f'{best[1]:>6.1f} (shift={best[0]})')

L()
L('  解读:')
L('    逐点EPE       = 按 TL,TR,BR,BL 一一对应的误差')
L('    最佳循环配对  = 允许把 GT 循环移位后取最小平均误差')
L('    若 最佳配对 << 逐点 => 【角点顺序混淆】')
L('    若 最佳配对 也很大 => 【真的定位错了】')
L()

n_better = 0
gains = []
for i in range(len(ds)):
    d = recs[i]
    base = d['epe'].mean()
    best = min(
        np.linalg.norm(np.roll(d['gt'], s, axis=0) - d['pred'],
                       axis=1).mean() for s in range(4))
    if best < base * 0.8:
        n_better += 1
    gains.append(base - best)

L(f'  全量统计（{len(ds)} 实例）:')
L(f'    循环移位后误差明显变小(<80%)的实例 = {n_better} '
  f'({100.0 * n_better / len(ds):.2f}%)')
L(f'    平均可降低误差 = {np.mean(gains):.2f} px')
L()

# ---------------- 变换后是否出界 + 有效像素 ---------------- #
stages_pre = [
    dict(type='LoadImage'),
    dict(type='GetBBoxCenterScale', padding=1.5),
    dict(type='TopdownAffine', input_size=(256, 256)),
]
trans = [TRANSFORMS.build(c) for c in stages_pre]

L('=' * 86)
L('这 8 个实例：变换后角点是否出界 / 短边有效像素')
L('=' * 86)
L(f'  {"idx":>5} {"img_id":<15} {"256内4点坐标":>42} {"出界":>6} {"短边占256":>10}')
L('  ' + '-' * 84)
for i in BAD:
    info = recs[i]['info']
    r = dict(info)
    for t in trans:
        r = t(r)
    tk = np.asarray(r['transformed_keypoints'])[0]
    oob = bool(((tk < 0) | (tk > 255)).any())
    x1, y1, x2, y2 = info['bbox'][0]
    w, h = x2 - x1, y2 - y1
    short256 = 256 / 1.5 * min(w, h) / max(w, h)
    coords = ','.join(f'({p[0]:.0f},{p[1]:.0f})' for p in tk)
    L(f'  {i:>5} {info["img_id"]:<15} {coords:>42} {str(oob):>6} '
      f'{short256:>9.0f}px')
L()

# ---------------- 可视化 ---------------- #
KP_COLORS = [(0, 0, 255), (0, 255, 0), (255, 0, 0), (0, 255, 255)]
panels, titles = [], []
for i in BAD[:8]:
    d = recs[i]
    info = d['info']
    raw = cv2.imread(info['img_path'])
    gt, pred = d['gt'], d['pred']
    x1, y1, x2, y2 = info['bbox'][0]
    cw, ch = x2 - x1, y2 - y1
    pad = 0.15
    X1 = int(max(0, x1 - cw * pad)); Y1 = int(max(0, y1 - ch * pad))
    X2 = int(min(raw.shape[1], x2 + cw * pad))
    Y2 = int(min(raw.shape[0], y2 + ch * pad))
    crop = raw[Y1:Y2, X1:X2].copy()
    gg = (gt - np.array([X1, Y1]))
    pp = (pred - np.array([X1, Y1]))
    hh, ww = crop.shape[:2]
    s = 400.0 / max(hh, ww)
    small = cv2.resize(crop, (int(ww * s), int(hh * s)))
    gg, pp = gg * s, pp * s

    v = small.copy()
    for k in range(4):
        cv2.circle(v, tuple(np.round(gg[k]).astype(int)), 10, KP_COLORS[k], 2)
        cv2.circle(v, tuple(np.round(pp[k]).astype(int)), 3, KP_COLORS[k], -1)
        cv2.putText(v, str(k), tuple(np.round(pp[k]).astype(int) + 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
    panels.append(v)
    titles.append(f'#{i} {info["img_id"]} w/h={max(cw, ch) / max(min(cw, ch), 1):.2f}')

rows = []
WMAX = 1600
for j in range(0, len(panels), 4):
    ch_, tt_ = panels[j:j + 4], titles[j:j + 4]
    hmax = max(c.shape[0] for c in ch_)
    # 统一每格宽度，避免 vstack 报错
    cell_w = WMAX // 4
    row = np.zeros((hmax, cell_w * len(ch_), 3), np.uint8)
    bar = np.zeros((26, cell_w * len(ch_), 3), np.uint8)
    for k, (c, t) in enumerate(zip(ch_, tt_)):
        cc = cv2.resize(c, (cell_w, hmax))
        row[:, k * cell_w:(k + 1) * cell_w] = cc
        cv2.putText(bar, t, (k * cell_w + 4, 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1)
    rows += [np.vstack([bar, row]), np.zeros((6, bar.shape[1], 3), np.uint8)]

cv2.imwrite('vis_out2/bad_cases.jpg', np.vstack(rows))
L('  可视化 -> vis_out2/bad_cases.jpg  (大圈=GT, 小实心=预测, 数字=点序号)')
L()

# ---------------- 几何合理性检查：4 点构成的形状像 DM 码吗？ ---------------- #
L('=' * 86)
L('几何合理性检查（DM 码投影后应近似平行四边形）')
L('=' * 86)
L('  对边长度比 应接近 1（透视会偏离但不会太夸张）；')
L('  若某组"对边"长度差很多，说明 4 个点可能不是同一个码。')
L()
L(f'  {"idx":>5} {"img_id":<15} {"4条边长":>30} {"对边长度比":>20}')
L('  ' + '-' * 74)
for i in BAD:
    gt = recs[i]['gt']
    e = [float(np.linalg.norm(gt[(k + 1) % 4] - gt[k])) for k in range(4)]
    r1 = e[0] / max(e[2], 1e-6)     # 上边 / 下边
    r2 = e[1] / max(e[3], 1e-6)     # 右边 / 左边
    L(f'  {i:>5} {recs[i]["info"]["img_id"]:<15} '
      f'{str([round(x) for x in e]):>30} '
      f'{r1:>8.2f} / {r2:<8.2f}')
L()

# ---------------- 分组统计 ---------------- #
sides = np.array([np.sqrt(recs[i]['info']['area']) for i in range(len(ds))])
epes = np.array([recs[i]['epe'].mean() for i in range(len(ds))])
maxepes = np.array([recs[i]['epe'].max() for i in range(len(ds))])
L('  按码大小分组（边长 sqrt(w*h)）:')
L(f'  {"分组":<18} {"n":>6} {"EPE均值":>9} {"EPE中位":>9} {"maxEPE中位":>11} '
  f'{"max<=20px":>10}')
L('  ' + '-' * 68)
for lo, hi, name in [(0, 60, '小码 <60px'), (60, 120, '中 60~120px'),
                     (120, 260, '大 120~260px'), (260, 1e9, '超大 >260px')]:
    m = (sides >= lo) & (sides < hi)
    if m.sum():
        L(f'  {name:<18} {m.sum():>6} {epes[m].mean():>9.2f} '
          f'{np.median(epes[m]):>9.2f} {np.median(maxepes[m]):>11.2f} '
          f'{100 * (maxepes[m] <= 20).mean():>9.1f}%')
L()

with open('_analyze2_out.txt', 'w', encoding='utf-8') as f:
    f.write('\n'.join(OUT))
