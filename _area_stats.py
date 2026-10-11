"""临时脚本：统计 DM 码 bbox 等效边长 / 面积，映射到 COCO 的 area 分段。

COCO keypoints 的 areaRng（xtcocotools/cocoeval.py setKpParams）：
    all    : [0, 1e10]
    small  : 不存在（keypoints 只有 all/medium/large 三段）
    medium : [32^2, 96^2] = [1024, 9216]
    large  : [96^2, 1e10]
所以边界是等效边长 32px 和 96px。

注意 area 是 bbox 面积 w*h，等效边长 = sqrt(w*h)。
"""
import numpy as np

from mmpose.datasets.datasets.DM import DMKPD_dataset

# 注意：serialize_data 默认 True 时，full_init 里 _serialize_data() 会调用
# `self.data_list.clear()` 把数据转存到 data_bytes（省内存），
# 所以 ds.data_list 是空的。这里绝不能用 serialize_data=True。
ds = DMKPD_dataset(
    data_root='DMKPD_dataset',
    ann_file='train/annotations',
    data_prefix=dict(img='train/images'),
    metainfo=dict(
        from_file='open-mmlab/mmpose/configs/_base_/datasets/DMKPD_meta.py'),
    serialize_data=False,
)

areas = []
for info in ds.data_list:
    x1, y1, x2, y2 = info['bbox'][0]
    areas.append(float((x2 - x1) * (y2 - y1)))
areas = np.array(areas, dtype=np.float64)

side = np.sqrt(areas)          # 等效正方形边长
med_w = np.array([i['bbox'][0][2] - i['bbox'][0][0] for i in ds.data_list])
med_h = np.array([i['bbox'][0][3] - i['bbox'][0][1] for i in ds.data_list])

n = len(areas)
small = int((areas < 32 ** 2).sum())
medium = int(((areas >= 32 ** 2) & (areas < 96 ** 2)).sum())
large = int((areas >= 96 ** 2).sum())

lines = [
    f'实例数 = {n}',
    f'裸 bbox 尺寸: w  med={np.median(med_w):.1f}  '
    f'p10={np.percentile(med_w,10):.1f}  p90={np.percentile(med_w,90):.1f}',
    f'             h  med={np.median(med_h):.1f}  '
    f'p10={np.percentile(med_h,10):.1f}  p90={np.percentile(med_h,90):.1f}',
    f'等效边长 sqrt(area): med={np.median(side):.1f}  '
    f'p10={np.percentile(side,10):.1f}  p90={np.percentile(side,90):.1f}  '
    f'min={side.min():.1f}  max={side.max():.1f}',
    '',
    '--- COCO keypoints area 分段（边界：32^2=1024, 96^2=9216）---',
    f'  small  (<32px)   : {small:6d}  ({100.0*small/n:5.1f}%)  '
    f'→ keypoints 评测里**没有** small 段，会被并入 all',
    f'  medium (32~96px) : {medium:6d}  ({100.0*medium/n:5.1f}%)  → AP (M)',
    f'  large  (>=96px)  : {large:6d}  ({100.0*large/n:5.1f}%)  → AP (L)',
    '',
    '--- sigma=0.1 下，达到某个 OKS 阈值所允许的相对误差 r=d/L ---',
]

for t in (0.95, 0.90, 0.85, 0.75, 0.5):
    r = float(np.sqrt(-np.log(t) / 12.5))
    lines.append(f'  OKS >= {t:.2f}  <= r <= {r*100:5.1f}%   '
                 f'(边长 {np.median(side):.0f}px 的码 -> {r*np.median(side):5.2f}px)')

with open('_area_stats_out.txt', 'w', encoding='utf-8') as f:
    f.write('\n'.join(lines))
print('\n'.join(lines))
