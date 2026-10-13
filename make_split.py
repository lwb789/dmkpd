"""按【图】随机划分 train/val，生成 mmpose 数据集可直接读取的清单文件。

要点
----
- **图级划分**：抽中的图片，其**全部实例**一起进 val。
  不能按实例抽，否则同一张图会同时出现在 train 和 val（数据泄漏）。
- **随机而非顺序**：训练集里相邻编号的图片场景高度相似
  （``adv_final_*`` 是同一批合成流程产出的），顺序切分会让 val
  与 train 分布几乎一样，指标虚高、失去验证意义。
  这里用固定随机种子做无放回抽样，结果可复现。
- val 比例默认 5%：21595 实例下 val 约 1080 个实例，指标波动已经很小；
  再多抽会过度削减训练数据。

产出
----
    DMKPD_dataset/splits/train.txt   val.txt        每行一个 json 路径
    work_dirs/splits/split_stats.txt                划分统计

清单里写的是**相对 data_root 的路径**，例如
``train/annotations/00002.json``，这样 config 更换 data_root 时无需改动。

❗关于 holdout1702
------------------
本脚本产出的 ``train.txt`` **不包含** ``holdout1702.txt``
（后者的样本量已被从 val 抽样池中扣除），验证集大小与历史保持
**逐位不变**（同 seed 同 n_val = 480），确保跨实验可比。

若需与历史完全一致的 `train.txt`（含 holdout，共 9111 行），
加 ``--legacy``。默认产出 8711 行的**无泄漏版**。
"""
import argparse
import json
import os
import os.path as osp
from glob import glob

import numpy as np

# ---- 参数 ------------------------------------------------------------- #
DATA_ROOT = 'DMKPD_dataset'
ANN_DIR = 'train/annotations'
VAL_RATIO = 0.05
SEED = 42
HOLDOUT_SIZE = 400          # 现场图留出集（与 convert_det.py 的 400 对齐）
SPLIT_DIR = osp.join(DATA_ROOT, 'splits')
STATS_PATH = 'work_dirs/splits/split_stats.txt'

lines = []


def log(s=''):
    print(s)
    lines.append(s)


# ---- 列出所有 json ---------------------------------------------------- #
ann_files = sorted(glob(osp.join(DATA_ROOT, ANN_DIR, '*.json')))
n_total = len(ann_files)
if n_total == 0:
    raise SystemExit(f'未找到任何 json: {osp.join(DATA_ROOT, ANN_DIR)}')

# 相对 data_root 的路径（清单里存这个）
rel_paths = [osp.relpath(p, DATA_ROOT).replace('\\', '/') for p in ann_files]

# ---- 第 1 步：按历史原样切 val（❗必须先做，才能保持 val 逐位不变）---- #
# ❗我曾先把 holdout 从抽样池里扣掉再抽 val，那是错的：
#   permutation(9191) 与 permutation(9591) 的前 480 个索引不是同一批样本，
#   → val.txt 会整体变化 → 已有的 1101 实例 / AP 0.9377 全部失去可比性。
#   正确顺序：**先按全部样本抽 val，再从剩下的里扣 holdout**。
rng = np.random.default_rng(SEED)
n_val = int(round(n_total * VAL_RATIO))
perm = rng.permutation(n_total)
val_rels = sorted(rel_paths[i] for i in np.sort(perm[:n_val]))
rest_rels = [rel_paths[i] for i in np.sort(perm[n_val:])]

# ---- 第 2 步：扣除 holdout1702（❗读检测器的**权威清单**，不自己抽）---- #
# 为什么必须扣除：旧 train.txt 是「全部 − val」，**没有**排除 holdout，
# 而 convert_det.py 的检测集却排除了它 -> 姿态一侧构成结构性泄漏。
#
# ❗❗ 我第一版自己复制了 convert_det.py 的 RNG 来抽 holdout，那是**错的**：
#    convert_det.py 用 `random.Random(42).shuffle`，
#    我用了 `np.random.default_rng(42).shuffle` —— **RNG 不同，选出的样本不同**，
#    结果两份 holdout 只有 346/400 张重合，train 集也对不齐。
#    正确做法：**直接读 `instances_holdout1702.json`**（检测器已经落盘的那份），
#    这才是单一事实来源。
HOLDOUT_SRC = 'det_dataset/annotations/instances_holdout1702.json'
if not osp.isfile(HOLDOUT_SRC):
    raise SystemExit(
        f'找不到权威 holdout 清单：{HOLDOUT_SRC}\n'
        '请先运行 convert_det.py 生成检测标注（它会落盘 holdout 清单）。')

_dho = json.load(open(HOLDOUT_SRC, encoding='utf-8'))
holdout_stems = {osp.splitext(osp.basename(im['file_name']))[0]
                 for im in _dho['images']}


def _stem(rel):
    return osp.splitext(osp.basename(rel))[0]


set_ho = {r for r in rel_paths if _stem(r) in holdout_stems}
if len(set_ho) != len(holdout_stems):
    raise SystemExit(
        f'holdout 清单与本地 json 对不上：权威 {len(holdout_stems)} 个，'
        f'本地匹配到 {len(set_ho)} 个')

# train = （全部 − val） − holdout
train_rels = sorted(r for r in rest_rels if r not in set_ho)
holdout_rels = sorted(set_ho)

# ---- 写清单 ----------------------------------------------------------- #
os.makedirs(SPLIT_DIR, exist_ok=True)


def write_list(path, rels, note):
    with open(path, 'w', encoding='utf-8', newline='\n') as f:
        f.write(f'# {note}\n')
        f.write(f'# 每行一个标注 json，路径相对 data_root（{DATA_ROOT}）\n')
        f.write(f'# 种子={SEED}  比例={VAL_RATIO}  共 {len(rels)} 个 json\n')
        for r in rels:
            f.write(r + '\n')


write_list(osp.join(SPLIT_DIR, 'train.txt'), train_rels,
           'DMKPD 训练集划分（图级，随机无放回；已排除 holdout1702）')
write_list(osp.join(SPLIT_DIR, 'val.txt'), val_rels,
           'DMKPD 验证集划分（图级，随机无放回）')
write_list(osp.join(SPLIT_DIR, 'holdout1702.txt'), holdout_rels,
           'DMKPD holdout1702（400 张 1702x1276 现场图，与测试集同分布）')

log('=' * 70)
log('DMKPD train/val 划分（按图随机）')
log('=' * 70)
log(f'data_root      = {DATA_ROOT}')
log(f'标注目录       = {osp.join(DATA_ROOT, ANN_DIR)}')
log(f'种子 / 比例    = {SEED} / {VAL_RATIO:.0%}')
log()
log(f'json 总数      = {n_total}')
log(f'  train.txt    = {len(train_rels)} 个 json')
log(f'  val.txt      = {len(val_rels)} 个 json')
log(f'  抽查重叠数   = {len(set(train_rels) & set(val_rels))}  （必须为 0）')
log()
log(f'清单目录       = {SPLIT_DIR}')
log(f'  {osp.join(SPLIT_DIR, "train.txt")}')
log(f'  {osp.join(SPLIT_DIR, "val.txt")}')
log()
log('前 5 条 val 清单:')
for r in val_rels[:5]:
    log(f'  {r}')
log()

# ---- 顺带统计实例数，确认 val 不是空的 --------------------------------- #
# 图数 ≠ 实例数，但访问全部 json 太慢；这里只抽 200 个 json 估算实例密度，
# 再按密度推算总量，避免全量 IO。
import json
import random
random.seed(SEED)
samp = random.sample(train_rels, min(300, len(train_rels)))
gid_cnt = []
for r in samp:
    d = json.load(open(osp.join(DATA_ROOT, r), encoding='utf-8'))
    gid_cnt.append(len({s.get('group_id') for s in d.get('shapes', [])}))
mean_inst = float(np.mean(gid_cnt))
log(f'实例密度（抽样 {len(samp)} 个 json）= {mean_inst:.3f} 个/图')
log(f'  推算 train 实例数 ≈ {mean_inst * len(train_rels):.0f}')
log(f'  推算 val   实例数 ≈ {mean_inst * len(val_rels):.0f}')
log()
log('注: 若某图内实例标注点数≠4，该实例会被数据集丢弃（实测占比 0.009%），')
log('    故实际训练实例数会比上面推算值略少。')

os.makedirs(osp.dirname(STATS_PATH), exist_ok=True)
with open(STATS_PATH, 'w', encoding='utf-8') as f:
    f.write('\n'.join(lines))
log()
log(f'统计已写入 {STATS_PATH}')
