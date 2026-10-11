"""用真实 transform 实测：RandomBBoxTransform 的 scale/rotate 组合是否会把
角点挤出 TopdownAffine 的窗口（挤出 = 该角点被裁掉，模型看不到，标签作废）。

原理：get_warp_matrix(fix_aspect_ratio=True) 只用 src_w 定窗口，
窗口是边长 = padding*scale*max(w,h) 的正方形（绕 center 旋转 rot）。
角点偏移 (±w/2, ±h/2) 固定不动，因此约束为
    (w|cosq| + h|sinq|) / max(w,h) <= padding * s
最坏情况（正方形, q=45°）= sqrt(2) = 1.4142
"""
import json
import os.path as osp
import glob
import numpy as np

from mmpose.datasets.transforms import (
    GetBBoxCenterScale, RandomBBoxTransform, TopdownAffine)

ANN_DIR = 'DMKPD_dataset/train/annotations'
INPUT_SIZE = (256, 256)


def load_instances():
    """复刻 DMKPD_dataset 的解析：每个 group_id 一条实例，4 个角点。"""
    insts = []
    for f in sorted(glob.glob(osp.join(ANN_DIR, '*.json'))):
        d = json.load(open(f, encoding='utf-8'))
        groups = {}
        for sh in d.get('shapes', []) or []:
            gid = sh.get('group_id')
            gid = 1 if gid is None else gid
            groups.setdefault(gid, []).append(sh)
        for gid, shapes in groups.items():
            pts = []
            for sh in shapes:
                pts.extend(sh['points'])
            pts = np.asarray(pts, dtype=np.float32)
            if pts.ndim != 2 or pts.shape[0] < 4:
                continue
            if np.allclose(pts[0], pts[-1]):
                pts = pts[:-1]
            if pts.shape[0] != 4:
                continue
            c = pts.mean(axis=0)
            ang = np.arctan2(pts[:, 1] - c[1], pts[:, 0] - c[0])
            pts = pts[np.argsort(ang, kind='stable')]
            x, y = pts[:, 0], pts[:, 1]
            area = 0.5 * (np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))
            if area < 0:
                pts = pts[::-1]
            s = pts[:, 0] + pts[:, 1]
            start = int(np.lexsort((pts[:, 0], s))[0])
            pts = np.roll(pts, -start, axis=0)
            x1, y1 = pts[:, 0].min(), pts[:, 1].min()
            x2, y2 = pts[:, 0].max(), pts[:, 1].max()
            # 编码方向：以 top_left->top_right 向量的倾角表示（-45~45°）。
            # 这是判断"最大旋转该控多少度"的唯一依据：旋转超过它就会
            # 把 top_left 转出"左上"语义区。
            v = pts[1] - pts[0]
            direction = float(np.degrees(np.arctan2(v[1], v[0])))
            while direction > 45.:
                direction -= 90.
            while direction <= -45.:
                direction += 90.
            insts.append(dict(
                bbox=np.array([[x1, y1, x2, y2]], np.float32),
                keypoints=pts[None].astype(np.float32),
                direction=direction,
            ))
    return insts


def run(insts, padding, scale_factor, rotate_factor, shift_factor,
        n_trials=5, seed=0):
    np.random.seed(seed)
    center_scale = GetBBoxCenterScale(padding=padding)
    rbt = RandomBBoxTransform(
        shift_factor=shift_factor, shift_prob=0.3,
        scale_factor=scale_factor, scale_prob=1.0,
        rotate_factor=rotate_factor, rotate_prob=1.0)
    affine = TopdownAffine(input_size=INPUT_SIZE)

    W, H = INPUT_SIZE
    oob_o = oob = total = 0          # 出界角点数 / 总角点数
    warn = 0                          # 落入边界安全带的角点数
    margin = 4.0                      # 距边界<4px 记为"危险"
    for _ in range(n_trials):
        for it in insts:
            r = dict(bbox=it['bbox'].copy(),
                     keypoints=it['keypoints'].copy(),
                     keypoints_visible=np.ones((1, 4), np.float32),
                     img=np.zeros((8, 8, 3), np.uint8))
            r = center_scale(r)
            r = rbt(r)
            r = affine(r)
            k = r['transformed_keypoints'][0]
            oob_o = int(0)  # noop
            bad = (k[:, 0] < 0) | (k[:, 0] > W - 1) | \
                  (k[:, 1] < 0) | (k[:, 1] > H - 1)
            near = (~bad) & ((k[:, 0] < margin) | (k[:, 0] > W - 1 - margin) |
                             (k[:, 1] < margin) | (k[:, 1] > H - 1 - margin))
            oob += int(bad.sum())
            warn += int(near.sum())
            total += 4
    return oob / total, warn / total


if __name__ == '__main__':
    insts = load_instances()
    print(f'实例数 = {len(insts)}')

    wh = np.array([[i['bbox'][0, 2] - i['bbox'][0, 0],
                    i['bbox'][0, 3] - i['bbox'][0, 1]] for i in insts])
    ratio = np.maximum(wh[:, 0], wh[:, 1]) / np.minimum(wh[:, 0], wh[:, 1])
    print(f'长短边比 mean={ratio.mean():.3f} med={np.median(ratio):.3f} '
          f'>1.05 占比={np.mean(ratio > 1.05):.1%} '
          f'>1.10 占比={np.mean(ratio > 1.10):.1%}')

    # ---- 编码方向分布：决定"最大旋转角"的合理上限 ----
    d = np.array([i['direction'] for i in insts])
    print()
    print('--- 编码方向（top_left->top_right 相对水平轴的倾角, -45~45°）---')
    print(f'  mean={d.mean():.2f}°  std={d.std():.2f}°  '
          f'median={np.median(d):.2f}°')
    print(f'  |θ|<=5° 占比={np.mean(np.abs(d) <= 5):.1%}   '
          f'<=10° 占比={np.mean(np.abs(d) <= 10):.1%}   '
          f'<=15° 占比={np.mean(np.abs(d) <= 15):.1%}')
    print(f'  <=20° 占比={np.mean(np.abs(d) <= 20):.1%}   '
          f'<=30° 占比={np.mean(np.abs(d) <= 30):.1%}')
    print(f'  绝对值的 p90={np.percentile(np.abs(d), 90):.2f}°  '
          f'p95={np.percentile(np.abs(d), 95):.2f}°  '
          f'p99={np.percentile(np.abs(d), 99):.2f}°  '
          f'max={np.abs(d).max():.2f}°')
    print('  → 若"最大旋转"取到 p95/p99，就能覆盖绝大多数真实姿态')
    # 理论最坏需求 padding*s >= sqrt(2)=1.4142（正方形时）
    theo = np.sqrt(wh[:, 0]**2 + wh[:, 1]**2) / np.max(wh, axis=1)
    print(f'理论 P*s 下限: max={theo.max():.4f} '
          f'p99={np.percentile(theo, 99):.4f} '
          f'med={np.median(theo):.4f}')

    # ---- 理论判据（最坏情况：正方形 + 该旋转角 + 平移取到极值）----
    # 归一化推导（M = max(w,h)，a = w/M，b = h/M）：
    #   窗口半边 = padding*s*M/2
    #   角点偏移 = 0.5*(w|cos|+h|sin|)          ，平移 = shift_factor*padding*w
    #   => padding*(s - 2*shift_factor*a) >= a*|cos| + b*|sin|
    # 注意：平移在归一化后是 2*shift_factor/s（比直觉大一倍，因为没有 0.5）
    import math
    print()
    print('--- 理论最大需求（正方形，即 a=b=1）---')
    for deg in (15, 20, 30, 40, 45):
        need = math.cos(math.radians(deg)) + math.sin(math.radians(deg))
        print(f'  θ={deg:>2}° -> 需 padding*s >= {need:.4f}')
    print('  （含平移时右侧还要再加 2*shift_factor）')

    cases = [
        # (padding, scale_factor, rotate, shift)
        ('① 你的方案 P=1.5 s=(.9,1.1) r30', 1.5, (0.9, 1.1), 30., 0.10),
        ('② 同上 shift=0.05', 1.5, (0.9, 1.1), 30., 0.05),
        ('③ 收紧 P=1.5 s=(.95,1.05) r30', 1.5, (0.95, 1.05), 30., 0.10),
        ('④ 同上 shift=0.05', 1.5, (0.95, 1.05), 30., 0.05),
        ('⑤ 不缩小 P=1.5 s=(1.0,1.1) r30', 1.5, (1.0, 1.1), 30., 0.10),
        ('⑥ 不缩小 P=1.5 s=(1.0,1.1) r30 sh.05', 1.5, (1.0, 1.1), 30., 0.05),
        ('⑦ 你的方案但转45 P=1.5 s=(.9,1.1) r45', 1.5, (0.9, 1.1), 45., 0.10),
        ('⑧ 最稳 P=1.6 s=(.9,1.1) r30', 1.6, (0.9, 1.1), 30., 0.10),
        ('⑨ 对照 P=1.5 s=(.95,1.05) r45', 1.5, (0.95, 1.05), 45., 0.10),
    ]
    print()
    print(f"{'情形':<36}{'s中心':>7}{'训练u中心':>10}{'测试u':>7}"
          f"{'出界率':>9}{'近界率':>9}")
    for name, p, sf, rf, shf in cases:
        o, w = run(insts, p, sf, rf, shf, n_trials=3)
        mu = (sf[0] + sf[1]) / 2.0
        print(f'{name:<36}{mu:>7.3f}{p * mu:>10.3f}{p:>7.2f}'
              f'{o:>8.2%}{w:>9.2%}')
    print()
    print('说明: "训练u中心" != "测试u" 就是训练/测试分辨率失配；')
    print('      s中心=1.0 时两者相等（无失配）。')
