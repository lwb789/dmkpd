# -*- coding: utf-8 -*-
"""P1 补充分析：从 _gap_result.json 的明细里挖出「该优化什么」。

核心问题：
  Phase 2 显示端到端 OKS 均值比 GT 框条件低 0.0646。
  这 0.0646 要拆成两部分，才能定优化优先级：
    (a) 漏检（召回 91.4%）：未检出的实例 OKS 记 0，是巨大损失
    (b) 框偏移：检出的实例因框不准而 OKS 下降
  本脚本把两者分开，并按尺寸分组，找出漏检集中在哪。
"""
import json
from collections import Counter, defaultdict

import numpy as np

d = json.load(open('_gap_result.json', encoding='utf-8'))
per = d['per_inst']
n = len(per)
print('=' * 82)
print('P1 补充分析：损失分解与分组特征')
print('=' * 82)
print(f"  实例总数          {n}")
print(f"  成功匹配          {d['n_matched']}  ({100*d['n_matched']/n:.1f}%)")
print(f"  漏检              {n - d['n_matched']}  "
      f"({100*(n-d['n_matched'])/n:.1f}%)")
print(f"  误检框            {d['n_fp']}")

# ---------------- 1. 损失分解 ---------------- #
print()
print('=' * 82)
print('1. 端到端 OKS 损失分解（0.0646 到底来自哪里）')
print('=' * 82)
oks_gt = np.array([p['oks_gt'] for p in per])          # GT框条件（全量）
oks_det = np.array([p['oks_det'] for p in per])        # 端到端（漏检记0）
matched = np.array([p['matched'] for p in per])

# (a) 漏检造成的损失：这 95 个实例在端到端下 OKS=0，
#     但若用 GT 框它们本可拿到 oks_gt
miss_gt_sum = oks_gt[~matched].sum()          # 本可拿到但丢了的分
miss_n = int((~matched).sum())
loss_miss = miss_gt_sum / n                   # 摊到全部实例

# (b) 框偏移造成的损失：已匹配实例上，GT框 OKS - 检测框 OKS
loss_off = (oks_gt[matched] - oks_det[matched]).sum() / n

total = oks_gt.mean() - oks_det.mean()
print(f'  GT 框条件 OKS 均值      {oks_gt.mean():.4f}')
print(f'  端到端 OKS 均值         {oks_det.mean():.4f}')
print(f'  ─────────────────────────────────────────')
print(f'  总损失                  {total:.4f}   (100.0%)')
print(f'  其中 漏检 {miss_n} 个     {loss_miss:.4f}   '
      f'({100*loss_miss/total:5.1f}%)  ← 主体')
print(f'       框偏移            {loss_off:.4f}   '
      f'({100*loss_off/total:5.1f}%)')
print()
print(f'  ❗结论：{100*loss_miss/total:.0f}% 的损失来自"码没被检出"，')
print(f'         只有 {100*loss_off/total:.0f}% 来自"框不够准"。')

# ---------------- 2. 按尺寸分组 ---------------- #
print()
print('=' * 82)
print('2. 按码尺寸分组（sqrt(area)，原图像素）')
print('=' * 82)
bins = [(0, 40), (40, 60), (60, 120), (120, 260), (260, 512), (512, 1e9)]
print(f'{"尺寸区间":<16}{"实例":>6}{"占比":>7}{"召回":>8}{"匹配IoU中位":>12}'
      f'{"OKS(GT框)":>11}{"OKS(端到端)":>12}{"漏检数":>7}')
for lo, hi in bins:
    sel = [p for p in per if lo <= p['side'] < hi]
    if not sel:
        continue
    m = [p for p in sel if p['matched']]
    rec = len(m) / len(sel)
    iou = np.median([p['iou'] for p in m]) if m else float('nan')
    og = np.mean([p['oks_gt'] for p in sel])
    od = np.mean([p['oks_det'] for p in sel])
    tag = f'[{lo}, {hi if hi < 1e9 else "inf"})'
    print(f'{tag:<16}{len(sel):>6}{100*len(sel)/n:>6.1f}%{100*rec:>7.1f}%'
          f'{iou:>12.3f}{og:>11.4f}{od:>12.4f}{len(sel)-len(m):>7}')

# ---------------- 3. 漏检特征 ---------------- #
print()
print('=' * 82)
print('3. 漏检的 95 个实例长什么样')
print('=' * 82)
missed = [p for p in per if not p['matched']]
ms = np.array([p['side'] for p in missed])
allside = np.array([p['side'] for p in per])
print(f'  漏检实例的码尺寸: 中位 {np.median(ms):.1f}px  '
      f'均值 {ms.mean():.1f}px  min {ms.min():.1f}  max {ms.max():.1f}')
print(f'  全部实例的码尺寸: 中位 {np.median(allside):.1f}px  '
      f'均值 {allside.mean():.1f}px')
print()
print('  漏检在各尺寸区间的分布：')
for lo, hi in bins:
    tot = sum(1 for p in per if lo <= p['side'] < hi)
    mis = sum(1 for p in missed if lo <= p['side'] < hi)
    if tot == 0:
        continue
    bar = '#' * int(40 * mis / max(1, tot))
    print(f'    [{lo:>4}, {hi if hi < 1e9 else "inf":>4})  '
          f'漏 {mis:>3}/{tot:<3} = {100*mis/tot:>5.1f}%  {bar}')

# ---------------- 4. 框偏移的困难实例 ---------------- #
print()
print('=' * 82)
print('4. 框偏移影响最大的实例（已匹配，但 OKS 掉得多）')
print('=' * 82)
m = [p for p in per if p['matched']]
for p in m:
    p['drop'] = p['oks_gt'] - p['oks_det']
worst = sorted(m, key=lambda x: -x['drop'])[:10]
print(f'  {"img_id":<10}{"边长":>8}{"IoU":>8}{"OKS(GT)":>10}'
      f'{"OKS(det)":>10}{"掉幅":>9}')
for p in worst:
    print(f"  {p['img_id']:<10}{p['side']:>8.1f}{p['iou']:>8.3f}"
          f"{p['oks_gt']:>10.4f}{p['oks_det']:>10.4f}{p['drop']:>9.4f}")
drops = np.array([p['drop'] for p in m])
print()
print(f'  掉幅 >0.1 的实例: {(drops>0.1).sum()} 个 '
      f'({100*(drops>0.1).mean():.1f}%)，它们贡献了 '
      f'{100*drops[drops>0.1].sum()/max(1e-9,drops[drops>0].sum()):.0f}% 的下降')
print(f'  掉幅 <=0.01 的实例: {(drops<=0.01).sum()} 个 '
      f'({100*(drops<=0.01).mean():.1f}%)  ← 绝大多数几乎不受影响')

# ---------------- 5. IoU-OKS 相关性 ---------------- #
print()
print('=' * 82)
print('5. 框 IoU 与 OKS 的关系（决定"框精度的收益上限"）')
print('=' * 82)
iou_all = np.array([p['iou'] for p in m])
base = np.array([p['oks_det'] for p in m])
print(f'  {"IoU 区间":<18}{"实例":>7}{"OKS均值":>10}{"与IoU=1的差":>13}')
for lo, hi in [(0, 0.6), (0.6, 0.7), (0.7, 0.8), (0.8, 0.9),
               (0.9, 0.95), (0.95, 1.01)]:
    sel = (iou_all >= lo) & (iou_all < hi)
    if sel.sum() == 0:
        continue
    v = base[sel]
    print(f'  [{lo:.2f}, {hi:.2f}){"":<6}{int(sel.sum()):>6}'
          f'{v.mean():>10.4f}{v.mean()-base[iou_all>=0.95].mean():>13.4f}')
print()
print(f'  IoU>=0.95 的实例 OKS 均值 = {base[iou_all>=0.95].mean():.4f}')
print(f'  当前全部匹配实例的 IoU 中位 = {np.median(iou_all):.3f}')
print(f'  -> 若把 IoU 从 {np.median(iou_all):.2f} 提到 0.95+，'
      f'OKS 至多提升 '
      f'{abs(base.mean()-base[iou_all>=0.95].mean()):.4f}')
print('     （这是「框精度」的**收益上限**）')

# ---------------- 6. 汇总 ---------------- #
print()
print('=' * 82)
print('6. 结论与优化优先级')
print('=' * 82)
gain_miss_all = loss_miss                    # 全救回漏检的收益
gain_iou_all = abs(base.mean() - base[iou_all >= 0.95].mean()) \
    * len(m) / n
print(f'  乐观上限（假设全做到）：')
print(f'    召回 91.4% -> 100%   : OKS +{gain_miss_all:.4f}   '
      f'（当前检测器只训了 1 epoch，这最可能实现）')
print(f'    框 IoU -> 0.95+      : OKS +{gain_iou_all:.4f}   '
      f'（收益小，且已接近极限）')
print()
print('  => 优化优先级：')
print('     ① **提升检测召回**（收益最大、且当前只训 1 epoch，空间大）')
print('     ② 框精度（收益上限仅约 %.4f，优先级低）' % gain_iou_all)
print('     ③ 姿态本身（AP 0.938 已高，且是 GT 框条件下的，不是瓶颈）')
