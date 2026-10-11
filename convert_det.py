# -*- coding: utf-8 -*-
"""把 LabelMe 标注转成检测用 COCO 格式，同时统计包围框尺寸分布。

为什么要单独做一份「检测标注」：
    姿态模型（topdown）需要外部给 bbox；测试集没有标注，
    所以必须训一个检测器来生产 bbox。检测只需要 bbox，不需要关键点，
    因此这里把实例的角点折成 min/max 外接矩形即可。

⚠️ 标注的两个坑（第一版转换器踩过，训练集丢了 6449 张图）：
    ① `polygon` 类型是 **5 个点**（4 角 + 首尾闭合重复点），
       `linestrip` 类型才是 4 个点。只认 4 点会丢掉全部 polygon
       —— 而 polygon 有 17896 条，占绝大多数。
    ② 一个 json 里的多个 shape **不一定是一个实例**，
       要靠 `group_id` 分组：同组 = 一个 DM 码（早期标注把一个四边形
       拆成多条 linestrip，或 polygon 与 linestrip 混用）。
    这两条与姿态数据集 `_extract_keypoints` / `_group_shapes` 完全一致，
    这里照抄以保持两份标注口径统一。

划分复用 splits/train.txt / val.txt（与姿态模型同一套图级划分），
避免检测器在姿态模型的 val 图上训练过，导致评估口径不诚实。

输出：
    det_dataset/annotations/instances_train.json
    det_dataset/annotations/instances_val.json
    det_dataset/annotations/instances_test.json   (只有 images，无标注)
"""
import json
import os
import os.path as osp
from collections import Counter

import numpy as np

ROOT = 'DMKPD_dataset'
OUT = 'det_dataset/annotations'
os.makedirs(OUT, exist_ok=True)


def group_shapes(shapes):
    """按 group_id 分组：一组 = 一个 DM 码实例。与姿态数据集同逻辑。"""
    g = {}
    for idx, s in enumerate(shapes):
        gid = s.get('group_id')
        if gid is None:
            gid = f'unlabeled_{idx}'
        g.setdefault(gid, []).append(s)
    return g


def instance_corners(group):
    """把一组的点整理成 (4,2)，去掉 polygon 的首尾闭合重复点。"""
    pts = [p for s in group for p in (s.get('points') or [])]
    a = np.asarray(pts, dtype=np.float32)
    if a.ndim != 2 or a.shape[0] < 4:
        return None
    if np.allclose(a[0], a[-1]):
        a = a[:-1]
    if a.shape[0] != 4:
        return None
    return a


def load_instances(json_path):
    """读一个 LabelMe json，返回 (w, h, [bbox_xyxy...], n_dropped)。"""
    with open(json_path, encoding='utf-8') as f:
        d = json.load(f)
    w, h = int(d['imageWidth']), int(d['imageHeight'])
    boxes = []
    dropped = 0
    for _, grp in group_shapes(d.get('shapes') or []).items():
        a = instance_corners(grp)
        if a is None:
            dropped += 1
            continue
        x1, y1 = float(a[:, 0].min()), float(a[:, 1].min())
        x2, y2 = float(a[:, 0].max()), float(a[:, 1].max())
        if x2 - x1 < 2 or y2 - y1 < 2:
            dropped += 1
            continue
        boxes.append([x1, y1, x2, y2])
    return w, h, boxes, dropped


def read_split(list_file):
    rels = []
    with open(list_file, encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#'):
                rels.append(line)
    return rels


def build(split_name, json_list):
    images, annotations = [], []
    img_id = ann_id = 0
    ws, hs, ratios, areas = [], [], [], []
    per_img = Counter()
    n_dropped = n_used = 0

    for rel in json_list:
        jp = osp.join(ROOT, rel)
        if not osp.isfile(jp):
            continue
        w, h, boxes, dropped = load_instances(jp)
        n_dropped += dropped

        stem = osp.splitext(osp.basename(jp))[0]
        jpg = None
        for ext in ('.jpg', '.jpeg', '.png', '.bmp'):
            cand = osp.join('train/images', stem + ext)
            if osp.isfile(osp.join(ROOT, cand)):
                # 路径相对**工作区根**（config 里 data_root='.'，
                # img_prefix 就是 '.'），所以必须带 DMKPD_dataset 前缀
                jpg = osp.join(ROOT, cand)
                break
        if jpg is None or not boxes:
            continue
        n_used += 1

        images.append(dict(id=img_id, file_name=jpg, width=w, height=h))
        per_img[len(boxes)] += 1
        for b in boxes:
            bw, bh = b[2] - b[0], b[3] - b[1]
            annotations.append(dict(
                id=ann_id, image_id=img_id, category_id=1,
                bbox=[b[0], b[1], bw, bh], area=bw * bh, iscrowd=0,
                segmentation=[], num_keypoints=0))
            ann_id += 1
            ws.append(bw)
            hs.append(bh)
            ratios.append(bw / bh)
            areas.append(bw * bh)
        img_id += 1

    coco = dict(images=images, annotations=annotations,
                categories=[dict(id=1, name='DPM', supercategory='code')])
    p = osp.join(OUT, f'instances_{split_name}.json')
    with open(p, 'w', encoding='utf-8') as f:
        json.dump(coco, f)
    print(f'[{split_name:>5}] 用到 json={n_used}/{len(json_list)}  '
          f'images={len(images)}  instances={len(annotations)}  '
          f'丢弃实例={n_dropped}')
    print(f'        -> {p}')
    return dict(ws=ws, hs=hs, ratios=ratios, areas=areas, per_img=per_img)


def report(stats_list, title='包围框尺寸分布（全部实例，单位=原图像素）'):
    ws = np.concatenate([np.array(s['ws']) for s in stats_list])
    hs = np.concatenate([np.array(s['hs']) for s in stats_list])
    ar = np.concatenate([np.array(s['areas']) for s in stats_list])
    rt = np.concatenate([np.array(s['ratios']) for s in stats_list])
    per = Counter()
    for s in stats_list:
        per.update(s['per_img'])
    print()
    print('=' * 78)
    print(title)
    print('=' * 78)
    print(f'{"统计量":<10}{"宽w":>12}{"高h":>12}{"sqrt(area)":>14}{"w/h":>10}')
    for q, lab in [(0, 'min'), (5, 'p05'), (25, 'p25'), (50, 'median'),
                   (75, 'p75'), (90, 'p90'), (95, 'p95'), (99, 'p99'),
                   (100, 'max')]:
        print(f'{lab:<10}{np.percentile(ws, q):>12.1f}'
              f'{np.percentile(hs, q):>12.1f}'
              f'{np.sqrt(np.percentile(ar, q)):>14.1f}'
              f'{np.percentile(rt, q):>10.3f}')
    side = np.sqrt(ar)
    print()
    print(f'共 {len(side)} 个实例，按 sqrt(area) 分组：')
    for lo, hi in [(0, 40), (40, 60), (60, 120), (120, 260), (260, 512),
                   (512, 1e9)]:
        m = (side >= lo) & (side < hi)
        print(f'  [{lo}, {hi if hi < 1e9 else "inf"})'
              f'{"":<8} {int(m.sum()):>6} 个 ({100*m.mean():5.1f}%)')
    tot = sum(per.values())
    print()
    print('每图实例数分布:', dict(sorted(per.items())))
    print(f'  平均 {sum(k*v for k,v in per.items())/tot:.2f} 个/图，'
          f'最多 {max(per)} 个')


def main():
    tr = read_split(osp.join(ROOT, 'splits', 'train.txt'))
    va = read_split(osp.join(ROOT, 'splits', 'val.txt'))
    print(f'划分清单：train {len(tr)} json / val {len(va)} json')
    s1 = build('train', tr)
    s2 = build('val', va)
    report([s1, s2])

    import cv2
    test_dir = osp.join(ROOT, 'test')
    imgs = sorted(f for f in os.listdir(test_dir) if f.lower().endswith('.jpg'))
    images = []
    for i, fn in enumerate(imgs):
        im = cv2.imread(osp.join(test_dir, fn), cv2.IMREAD_REDUCED_COLOR_8)
        h, w = im.shape[:2]
        images.append(dict(id=i, file_name=osp.join(test_dir, fn),
                           width=w * 8, height=h * 8))
    p = osp.join(OUT, 'instances_test.json')
    with open(p, 'w', encoding='utf-8') as f:
        json.dump(dict(images=images, annotations=[],
                       categories=[dict(id=1, name='DPM',
                                        supercategory='code')]), f)
    print()
    print(f'[ test] images={len(images)} (无标注) -> {p}')

    # ---------------------------------------------------------------- #
    # 变尺度留出集：从 train 的 1702 档里切出 400 张**保持原始尺寸**
    #
    # 为什么需要它：val.txt 里 1702 档只有 135 张；而验证时 pipeline 会
    # 把图缩到 1024 长边，码尺寸也随之缩小 —— 与测试集（同为 1702 档、
    # 同样缩到 1024）其实是一致的。但为了能单独回答「检测器在**真实尺度**
    # 的现场图上表现如何」，这里再切一份只含 1702 档的留出集，
    # 训练时把它排除，专门用于诊断。
    # ---------------------------------------------------------------- #
    import random
    dtr = json.load(open(osp.join(OUT, 'instances_train.json'),
                         encoding='utf-8'))
    big_ids = [im['id'] for im in dtr['images'] if im['width'] > 1400]
    rng = random.Random(42)
    rng.shuffle(big_ids)
    hold_ids = set(big_ids[:400])
    new_imgs, new_anns = [], []
    hold_imgs, hold_anns = [], []
    remap = {}
    for im in dtr['images']:
        if im['id'] in hold_ids:
            remap[im['id']] = len(hold_imgs)
            hold_imgs.append(dict(im, id=remap[im['id']]))
        else:
            remap[im['id']] = len(new_imgs)
            new_imgs.append(dict(im, id=remap[im['id']]))
    hid = 0
    for a in dtr['annotations']:
        if a['image_id'] in hold_ids:
            hold_anns.append(dict(a, image_id=remap[a['image_id']],
                                  id=hid))
            hid += 1
        else:
            new_anns.append(dict(a, image_id=remap[a['image_id']]))
    cats = dtr['categories']
    json.dump(dict(images=new_imgs, annotations=new_anns, categories=cats),
              open(osp.join(OUT, 'instances_train.json'), 'w',
                   encoding='utf-8'))
    json.dump(dict(images=hold_imgs, annotations=hold_anns, categories=cats),
              open(osp.join(OUT, 'instances_holdout1702.json'), 'w',
                   encoding='utf-8'))
    print(f'[train ] 去掉留出集后: images={len(new_imgs)} '
          f'instances={len(new_anns)}')
    print(f'[holdout1702] images={len(hold_imgs)} instances={len(hold_anns)} '
          f'(纯 1702x1276 现场图，仅用于诊断)')


if __name__ == '__main__':
    main()
