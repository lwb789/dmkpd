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
"""
import os
import os.path as osp
from glob import glob

import numpy as np
from mmengine.fileio import dump

# ---- 参数 ------------------------------------------------------------- #
DATA_ROOT = 'DMKPD_dataset'
ANN_DIR = 'train/annotations'
VAL_RATIO = 0.05
SEED = 42
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

# ---- 无放回随机抽样 --------------------------------------------------- #
rng = np.random.default_rng(SEED)
n_val = int(round(n_total * VAL_RATIO))
perm = rng.permutation(n_total)
val_idx = np.sort(perm[:n_val])
train_idx = np.sort(perm[n_val:])

val_rels = [rel_paths[i] for i in val_idx]
train_rels = [rel_paths[i] for i in train_idx]

# ---- 写清单 ----------------------------------------------------------- #
os.makedirs(SPLIT_DIR, exist_ok=True)
def write_list(path, rels, note):
    with open(path, 'w', encoding='utf-8') as f:
        f.write(f'# {note}\n')
        f.write(f'# 每行一个标注 json，路径相对 data_root（{DATA_ROOT}）\n')
        f.write(f'# 种子={SEED}  比例={VAL_RATIO}  共 {len(rels)} 个 json\n')
        for r in rels:
            f.write(r + '\n')

write_list(osp.join(SPLIT_DIR, 'train.txt'), train_rels,
           'DMKPD 训练集划分（图级，随机无放回）')
write_list(osp.join(SPLIT_DIR, 'val.txt'), val_rels,
           'DMKPD 验证集划分（图级，随机无放回）')

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
