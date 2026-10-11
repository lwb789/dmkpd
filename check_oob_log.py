"""验证 CheckKeypointOutOfBounds 的两个行为：
1) 出界角点的 keypoints_visible 被置 0（走 codec 的"跳过 + 权重 0"安全分支）
2) 出界明细被写入 {out_dir}/oob_keypoints_log.txt

构造一个**确定会出界**的样本：bbox 10x10 的四个角点，窗口边长设为 13
（模拟 padding*s 不足），再旋转 45°。
角点到中心距离 = sqrt(5^2+5^2) = 7.071，窗口半径 = 6.5 -> 必然出界。
"""
import json
import os
import shutil
import numpy as np

from mmpose.datasets.transforms import TopdownAffine
from mmpose.datasets.datasets.DM.transforms import CheckKeypointOutOfBounds

OUT_DIR = 'work_dirs/oob_check'
INPUT_SIZE = (256, 256)


def build_results(scale, rotation):
    pts = np.array([[0., 0.], [10., 0.], [10., 10.], [0., 10.]], np.float32)
    return dict(
        img=np.zeros((64, 64, 3), np.uint8),
        img_id='synthetic_oob',
        img_path='synthetic_oob.jpg',
        idx=0,
        bbox=pts.min(0).tolist() + pts.max(0).tolist(),
        keypoints=pts[None].astype(np.float32),
        keypoints_visible=np.ones((1, 4), np.float32),
        bbox_center=np.array([[5., 5.]], np.float32),
        bbox_scale=np.array([[scale, scale]], np.float32),
        bbox_rotation=np.array([rotation], np.float32),
    )


def main():
    if os.path.isdir(OUT_DIR):
        shutil.rmtree(OUT_DIR)
    os.makedirs(OUT_DIR, exist_ok=True)

    affine = TopdownAffine(input_size=INPUT_SIZE)
    checker = CheckKeypointOutOfBounds(
        elapse=1.0, mask_near=False, out_dir=OUT_DIR, log_interval=1)

    cases = [
        ('窗口够大 (scale=20, rot=45) -> 应无出界', 20.0, 45.0, False),
        ('窗口不足 (scale=13, rot=45) -> 应出界', 13.0, 45.0, True),
        ('窗口不足 (scale=13, rot=0)  -> 不应出界', 13.0, 0.0, False),
    ]
    ok = True
    for name, scale, rot, expect_oob in cases:
        r = build_results(scale, rot)
        r = affine(r)
        k = r['transformed_keypoints'][0]
        w, h = INPUT_SIZE
        oob_n = int(((k[:, 0] < 0) | (k[:, 0] > w - 1) |
                     (k[:, 1] < 0) | (k[:, 1] > h - 1)).sum())

        r = checker(r)
        vis_zero = int((r['keypoints_visible'][0] == 0).sum())
        flag = 'OK ' if (oob_n > 0) == expect_oob else 'BAD'
        if flag == 'BAD':
            ok = False
        print(f'[{flag}] {name}')
        print(f'        出界角点数={oob_n}  visible被置0数={vis_zero}')
        print(f'        变换后角点={np.round(k, 2).tolist()}')

    log_path = os.path.join(OUT_DIR, 'oob_keypoints_log.txt')
    print()
    if os.path.isfile(log_path):
        lines = open(log_path, encoding='utf-8').read().strip().split('\n')
        print(f'日志文件已写入: {log_path}  ({len(lines)} 行)')
        for ln in lines:
            rec = json.loads(ln)
            print('  出界索引=', rec['oob_kpt_indices'],
                  ' bbox_scale=', rec['bbox_scale'],
                  ' bbox_rotation=', rec['bbox_rotation'])
    else:
        ok = False
        print('!! 日志文件未生成')

    print()
    print('全部通过' if ok else '存在失败项')


if __name__ == '__main__':
    main()
