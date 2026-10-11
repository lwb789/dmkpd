# -*- coding: utf-8 -*-
"""P1 可视化：把 gap 分析画成图，便于一眼看懂结论。"""
import json

import cv2
import numpy as np

d = json.load(open('_gap_result.json', encoding='utf-8'))
per = d['per_inst']
n = len(per)
m = [p for p in per if p['matched']]
CW, CH = 640, 400          # 每个面板的画布
PAD_L, PAD_B, PAD_T, PAD_R = 62, 46, 52, 20


def canvas(title=''):
    c = np.full((CH, CW, 3), 26, np.uint8)
    if title:
        cv2.putText(c, title, (14, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.62,
                    (235, 235, 235), 2)
    # 边框
    cv2.rectangle(c, (PAD_L, PAD_T), (CW - PAD_R, CH - PAD_B),
                  (70, 70, 70), 1)
    return c


def bar_panel(title, pairs, xlabel='', fmt='{:.3f}', color=(90, 200, 120)):
    """pairs: [(label, value), ...]  横向条形图"""
    c = canvas(title)
    if not pairs:
        return c
    vmax = max(v for _, v in pairs) or 1.0
    plot_w = CW - PAD_L - PAD_R - 90
    plot_h = CH - PAD_T - PAD_B
    rowh = plot_h / len(pairs)
    for i, (lab, v) in enumerate(pairs):
        y0 = int(PAD_T + i * rowh + rowh * 0.18)
        y1 = int(PAD_T + (i + 1) * rowh - rowh * 0.18)
        w = int(plot_w * (v / vmax))
        col = color if i == len(pairs) - 1 else (150, 150, 150)
        cv2.rectangle(c, (PAD_L, y0), (PAD_L + w, y1), col, -1)
        cv2.putText(c, lab, (PAD_L - 6 - len(lab) * 8, y1 - 3),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (215, 215, 215), 1)
        cv2.putText(c, fmt.format(v), (PAD_L + w + 6, y1 - 3),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    if xlabel:
        cv2.putText(c, xlabel, (PAD_L, CH - 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (170, 170, 170), 1)
    return c


def hist_panel(title, vals, xlabel, nbins=40, vmin=None, vmax=None,
               color=(90, 170, 240), marks=None):
    c = canvas(title)
    vals = np.asarray(vals, float)
    vmin = vmin if vmin is not None else float(vals.min())
    vmax = vmax if vmax is not None else float(vals.max())
    if vmax <= vmin:
        vmax = vmin + 1
    h, _ = np.histogram(vals, bins=nbins, range=(vmin, vmax))
    plot_w = CW - PAD_L - PAD_R
    plot_h = CH - PAD_T - PAD_B
    hmax = h.max() or 1
    bw = plot_w / nbins
    for i, v in enumerate(h):
        if v == 0:
            continue
        x0 = int(PAD_L + i * bw)
        x1 = int(PAD_L + (i + 1) * bw) - 1
        bh = int(plot_h * v / hmax)
        cv2.rectangle(c, (x0, CH - PAD_B - bh), (x1, CH - PAD_B), color, -1)
    # 轴刻度
    for t in np.linspace(vmin, vmax, 6):
        x = int(PAD_L + plot_w * (t - vmin) / (vmax - vmin))
        cv2.line(c, (x, CH - PAD_B), (x, CH - PAD_B + 5), (110, 110, 110), 1)
        cv2.putText(c, f'{t:.2f}', (x - 18, CH - PAD_B + 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (185, 185, 185), 1)
    if marks:
        for mv, col, lab in marks:
            x = int(PAD_L + plot_w * (mv - vmin) / (vmax - vmin))
            cv2.line(c, (x, PAD_T), (x, CH - PAD_B), col, 2)
            cv2.putText(c, lab, (x + 4, PAD_T + 16),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, col, 1)
    cv2.putText(c, xlabel, (PAD_L, CH - 14),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (170, 170, 170), 1)
    cv2.putText(c, f'n={len(vals)}  中位 {np.median(vals):.3f}  '
                   f'均值 {vals.mean():.3f}',
                (PAD_L + 4, PAD_T + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                (200, 220, 200), 1)
    return c


oks_gt = np.array([p['oks_gt'] for p in per])
oks_det = np.array([p['oks_det'] for p in per])
m = [p for p in per if p['matched']]
iou = np.array([p['iou'] for p in m])
side = np.array([p['side'] for p in per])

panels = []

# 1) OKS 直方图对比
panels.append(hist_panel('OKS 分布：GT 框（现有评测条件）', oks_gt,
                         'OKS', vmin=0, vmax=1, color=(80, 190, 110),
                         marks=[(oks_gt.mean(), (0, 255, 255), 'mean')]))
panels.append(hist_panel('OKS 分布：检测框（端到端真实）', oks_det,
                         'OKS', vmin=0, vmax=1, color=(90, 140, 245),
                         marks=[(oks_det.mean(), (0, 255, 255), 'mean')]))

# 2) 损失分解
loss_miss = oks_gt[[not p['matched'] for p in per]].sum() / n
loss_off = (oks_gt[[p['matched'] for p in per]]
            - oks_det[[p['matched'] for p in per]]).sum() / n
panels.append(bar_panel(
    '损失分解（总 %.4f）' % (oks_gt.mean() - oks_det.mean()),
    [('漏检 8.6%\n(95个)', loss_miss), ('框偏移 26%', loss_off)],
    'OKS 损失贡献', '{:.4f}', color=(80, 120, 245)))

# 3) 匹配 IoU 直方图
panels.append(hist_panel('检测框 vs GT 框 的 IoU（1006 个匹配）', iou,
                         'IoU', vmin=0.5, vmax=1.0, color=(200, 150, 90),
                         marks=[(np.median(iou), (0, 255, 255), 'median')]))

# 4) 按尺寸的召回率
bins = [(0, 40), (40, 60), (60, 120), (120, 260), (260, 1e9)]
names = ['<40', '40-60', '60-120', '120-260', '>260']
pairs = []
for (lo, hi), nm in zip(bins, names):
    sel = [p for p in per if lo <= p['side'] < hi]
    if sel:
        rec = sum(1 for p in sel if p['matched']) / len(sel)
        pairs.append((f'{nm}px ({len(sel)})', rec))
panels.append(bar_panel('按码尺寸的检测召回率', pairs, 'recall',
                        '{:.1%}', color=(80, 190, 110)))

# 5) IoU -> OKS 关系（折线）
c = canvas('框 IoU 与 OKS 的关系（框精度的收益曲线）')
xs = np.linspace(0.5, 1.0, 11)
ys, cnt = [], []
for x in xs:
    sel = (iou >= x - 0.025) & (iou < x + 0.025)
    ys.append(np.mean([p['oks_det'] for p, s in zip(m, sel) if s])
              if sel.sum() else np.nan)
    cnt.append(int(sel.sum()))
pts = []
for x, y in zip(xs, ys):
    if np.isnan(y):
        continue
    px = int(PAD_L + (CW - PAD_L - PAD_R) * (x - 0.5) / 0.5)
    py = int(CH - PAD_B - (CH - PAD_T - PAD_B) * y)
    pts.append((px, py))
for i in range(len(pts) - 1):
    cv2.line(c, pts[i], pts[i + 1], (100, 220, 255), 3)
for x, y, cn in zip(xs, ys, cnt):
    if np.isnan(y):
        continue
    px = int(PAD_L + (CW - PAD_L - PAD_R) * (x - 0.5) / 0.5)
    py = int(CH - PAD_B - (CH - PAD_T - PAD_B) * y)
    cv2.circle(c, (px, py), 4, (255, 255, 255), -1)
    cv2.putText(c, f'{y:.2f}', (px - 16, py - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (220, 220, 220), 1)
for t in np.linspace(0.5, 1.0, 6):
    px = int(PAD_L + (CW - PAD_L - PAD_R) * (t - 0.5) / 0.5)
    cv2.putText(c, f'{t:.2f}', (px - 16, CH - PAD_B + 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.44, (185, 185, 185), 1)
cv2.putText(c, 'IoU', (CW - 60, CH - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
            (170, 170, 170), 1)
cv2.putText(c, 'OKS', (14, PAD_T + 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
            (170, 170, 170), 1)
cv2.putText(c, '当前中位 IoU=%.3f -> 提到 0.95 最多 +%.4f'
               % (np.median(iou),
                  abs(np.mean([p['oks_det'] for p in m])
                      - np.mean([p['oks_det'] for p in m
                                 if p['iou'] >= 0.95]))),
            (PAD_L + 4, CH - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.46,
            (0, 220, 255), 1)
panels.append(c)

# 6) 结论图
c = np.full((CH, CW, 3), 26, np.uint8)
lines = [
    ('P1 结论：端到端损失 0.0646 的来源', (0, 220, 255), 0.62),
    ('', None, 0),
    ('GT 框条件 OKS    0.9153', (205, 205, 205), 0.52),
    ('端到端 OKS       0.8507   (-0.0646)', (205, 205, 205), 0.52),
    ('', None, 0),
    ('[漏检 95个]      -0.0477   73.8%  主体', (90, 225, 130), 0.52),
    ('[框偏移]         -0.0169   26.2%', (155, 175, 245), 0.52),
    ('', None, 0),
    ('检测召回 91.4%：只有 91.4% 的码能被姿态看到',
     (255, 215, 120), 0.46),
    ('匹配 IoU 中位 0.931：已很高，框精度空间有限',
     (255, 215, 120), 0.46),
    ('掉幅>0.1 仅 64 个(6.4%)，多在 adv 低可见样本',
     (195, 195, 195), 0.46),
    ('', None, 0),
    ('优化优先级', (0, 220, 255), 0.56),
    ('1. 提升检测召回（1 -> 24 epoch，空间最大）',
     (90, 225, 130), 0.48),
    ('2. 框精度上限仅 +0.0513（优先级低）', (175, 175, 175), 0.48),
    ('3. 姿态 AP 0.938 已高，不是瓶颈', (175, 175, 175), 0.48),
]
y = 42
for txt, col, sc in lines:
    if txt and col:
        cv2.putText(c, txt, (18, y), cv2.FONT_HERSHEY_SIMPLEX, sc, col,
                    1 if sc < 0.55 else 2)
    y += 24 if sc < 0.55 else 30
panels.append(c)

rows = []
for i in range(0, len(panels), 2):
    chunk = panels[i:i + 2]
    if len(chunk) == 1:
        # 补齐到两个，保持行宽一致
        chunk.append(np.full((CH, CW, 3), 26, np.uint8))
    rows.append(np.hstack(chunk))
grid = np.vstack(rows)
cv2.imwrite('vis_out2/gap_analysis.jpg', grid,
            [cv2.IMWRITE_JPEG_QUALITY, 92])
print('-> vis_out2/gap_analysis.jpg', grid.shape[1], 'x', grid.shape[0])
