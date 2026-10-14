# -*- coding: utf-8 -*-
"""画出提交文件里「有问题」的实例，便于人工核对。

用途
----
`_sub_check.py` 只给数字（越界 N 个 / 退化 M 个 / 非凸 K 个），
本脚本把**具体是哪个文件、哪一行、四点画在哪**直接画出来，
不需要再去翻 txt 对照像素坐标。

用法：
    python _sub_vis.py                    # 只看退化
    python _sub_vis.py --kind all         # 退化 + 非凸 + 越界
    python _sub_vis.py --ids 001443 002969
"""
import argparse
import glob
import os
import os.path as osp

import cv2
import numpy as np

OUT_DIR = 'vis_out2'


def quad_diag(xy):
    """返回诊断信息：边长、叉积、面积。"""
    v = np.roll(xy, -1, axis=0) - xy
    v2 = np.roll(v, -1, axis=0)
    cross = v[:, 0] * v2[:, 1] - v[:, 1] * v2[:, 0]
    d = np.linalg.norm(v, axis=1)
    area = 0.5 * abs(float(cross.sum()))
    return d, cross, area


def find_img(img_id):
    for pat in (f'DMKPD_dataset/test/{img_id}.jpg',
                f'DMKPD_dataset/test/**/{img_id}.jpg'):
        c = glob.glob(pat, recursive=True)
        if c:
            return c[0]
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--sub-dir', default='submission')
    ap.add_argument('--kind', default='degenerate',
                    choices=['degenerate', 'badpoly', 'oob', 'all'])
    ap.add_argument('--ids', nargs='*', default=None,
                    help='只处理这些 img_id')
    ap.add_argument('--min-edge', type=float, default=5.0)
    ap.add_argument('--scale', type=float, default=1.0)
    args = ap.parse_args()

    files = sorted(glob.glob(osp.join(args.sub_dir, '*.txt')))
    if args.ids:
        ids = set(args.ids)
        files = [f for f in files if osp.splitext(osp.basename(f))[0] in ids]

    found = []      # (img_id, line_no, conf, xy, kind, detail)
    for f in files:
        img_id = osp.splitext(osp.basename(f))[0]
        for ln, line in enumerate(open(f, encoding='utf-8'), 1):
            v = line.split()
            if not v:
                continue
            conf = float(v[1])
            xy = np.array([float(x) for x in v[2:10]],
                          np.float64).reshape(4, 2)
            d, cross, area = quad_diag(xy)
            mn = float(d.min())
            convex = bool((cross > 0).all() or (cross < 0).all())
            if mn < args.min_edge:
                kind, detail = 'degenerate', f'最短边 {mn:.2f}px'
            elif not convex:
                kind, detail = 'badpoly', f'最短边 {mn:.1f}px 非凸'
            else:
                continue
            if args.kind in (kind, 'all'):
                found.append((img_id, ln, conf, xy, kind, detail))

    print(f'共 {len(found)} 个问题实例')
    for img_id, ln, conf, xy, kind, detail in found:
        d, cross, area = quad_diag(xy)
        print()
        print('=' * 74)
        print(f'{img_id}.txt  第 {ln} 行   conf={conf:.4f}   [{kind}] {detail}')
        for j, (x, y) in enumerate(xy):
            print(f'    P{j}  ({x:>8.2f}, {y:>8.2f})')
        print(f'    边长 {[round(float(t), 2) for t in d]}')
        print(f'    叉积 {[round(float(t), 1) for t in cross]}   面积 {area:.1f}')
        for a in range(4):
            for b in range(a + 1, 4):
                dd = float(np.linalg.norm(xy[a] - xy[b]))
                if dd < 30:
                    print(f'    ❗ P{a} 与 P{b} 仅相距 {dd:.2f}px')

        p = find_img(img_id)
        if not p:
            print('    （找不到图，跳过绘图）')
            continue
        im = cv2.imread(p)
        if args.scale != 1.0:
            im = cv2.resize(im, None, fx=args.scale, fy=args.scale,
                            interpolation=cv2.INTER_LINEAR)
            xy_d = xy * args.scale
        else:
            xy_d = xy
        ov = im.copy()
        pts = xy_d.astype(np.int32).reshape(-1, 1, 2)
        # 画四边形（红）——退化时会画不出可见边，所以额外画大圆点
        cv2.polylines(ov, [pts], True, (0, 0, 255), 3, cv2.LINE_AA)
        cols = [(255, 0, 0), (0, 255, 0), (0, 255, 255), (255, 0, 255)]
        for j, (x, y) in enumerate(xy_d):
            c = tuple(int(t) for t in cols[j])
            cv2.circle(ov, (int(x), int(y)), 14, c, -1, cv2.LINE_AA)
            cv2.circle(ov, (int(x), int(y)), 14, (0, 0, 0), 2, cv2.LINE_AA)
            cv2.putText(ov, f'P{j}', (int(x) + 16, int(y) - 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 4,
                        cv2.LINE_AA)
            cv2.putText(ov, f'P{j}', (int(x) + 16, int(y) - 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, c, 2, cv2.LINE_AA)
        # 放大到重合点附近
        cx, cy = xy_d.mean(0)
        r = 260
        x0, x1 = int(max(0, cx - r)), int(min(ov.shape[1], cx + r))
        y0, y1 = int(max(0, cy - r)), int(min(ov.shape[0], cy + r))
        crop = ov[y0:y1, x0:x1]

        os.makedirs(OUT_DIR, exist_ok=True)
        out = osp.join(OUT_DIR, f'sub_{kind}_{img_id}_L{ln}.jpg')
        cv2.imwrite(out, np.ascontiguousarray(ov))
        out2 = osp.join(OUT_DIR, f'sub_{kind}_{img_id}_L{ln}_zoom.jpg')
        cv2.imwrite(out2, np.ascontiguousarray(crop))
        print(f'    -> {out}')
        print(f'    -> {out2}')

    if not found:
        print('（没有符合条件的问题实例）')


if __name__ == '__main__':
    main()
