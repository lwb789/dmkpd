"""精确定位：为什么每迭代 1.0s 而纯 GPU 只要 0.4s？

逐项拆解 model.train_step() 的开销：
  1. 纯 GPU 计算（extract_feat + forward + loss + backward + step）
  2. + ComputeAcc（head.loss 里的 to_numpy + CPU PCK）
  3. + 完整 train_step（含数据预处理、日志等）
  4. dataloader 端到端
"""
import time

import numpy as np
import torch
from mmengine.config import Config
from mmengine.registry import init_default_scope

init_default_scope('mmpose')

from mmpose.datasets.datasets.DM import DMKPD_dataset  # noqa: E402
from mmpose.registry import MODELS  # noqa: E402

OUT = []


def L(s=''):
    print(s)
    OUT.append(str(s))


cfg = Config.fromfile('config.py')
L('=' * 78)
L('0) 检查 head.loss 里的 compute_acc 默认值')
L('=' * 78)
head = cfg.model['head']
L(f'  head.type = {head["type"]}')
L(f'  train_cfg in cfg.model = {cfg.model.get("train_cfg")}')
L('  -> HeatmapHead.loss 里 `train_cfg.get("compute_acc", True)` '
  '默认为 True，')
L('     即每迭代都做 to_numpy(pred) + to_numpy(gt) + CPU 上算 PCK')
L()

# ---------------- 1) 纯 GPU ---------------- #
L('=' * 78)
L('1) 纯 GPU 计算（跳过 head.loss 的 compute_acc）')
L('=' * 78)
model = MODELS.build(cfg.model).cuda()
opt = torch.optim.Adam(model.parameters(), lr=1e-4)

BS = 4
x = torch.randn(BS, 3, 256, 256, device='cuda')
hm = torch.rand(BS, 4, 64, 64, device='cuda')
tw = torch.ones(BS, 4, device='cuda')
crit = MODELS.build(dict(type='KeypointMSELoss', use_target_weight=True))


def iter_plain():
    feats = model.extract_feat(x)
    pred = model.head.forward(feats)
    loss = crit(pred, hm, tw)
    opt.zero_grad(); loss.backward(); opt.step()


for _ in range(3):
    iter_plain()
torch.cuda.synchronize()
t0 = time.perf_counter()
n = 15
for _ in range(n):
    iter_plain()
torch.cuda.synchronize()
t_gpu = (time.perf_counter() - t0) / n
L(f'  extract_feat+forward+loss+backward+step = {t_gpu*1000:.1f} ms')
L()

# ---------------- 2) 加上 compute_acc ---------------- #
L('=' * 78)
L('2) 加上 head.loss 的 compute_acc（含 to_numpy + CPU PCK）')
L('=' * 78)
from mmpose.evaluation.functional import pose_pck_accuracy  # noqa: E402
from mmpose.utils.tensor_utils import to_numpy  # noqa: E402


def iter_with_acc():
    feats = model.extract_feat(x)
    pred = model.head.forward(feats)
    loss = crit(pred, hm, tw)
    # 复刻 HeatmapHead.loss 里的 compute_acc 部分
    _, avg_acc, _ = pose_pck_accuracy(
        output=to_numpy(pred),
        target=to_numpy(hm),
        mask=to_numpy(tw) > 0)
    opt.zero_grad(); loss.backward(); opt.step()


for _ in range(3):
    iter_with_acc()
torch.cuda.synchronize()
t0 = time.perf_counter()
for _ in range(n):
    iter_with_acc()
torch.cuda.synchronize()
t_acc = (time.perf_counter() - t0) / n
L(f'  + compute_acc 合计 = {t_acc*1000:.1f} ms')
L(f'  => compute_acc 多花了 {(t_acc-t_gpu)*1000:.1f} ms')
L()

# 单独测 to_numpy + PCK（无 GPU 计算）
pred_cpu_src = torch.rand(BS, 4, 64, 64, device='cuda')
hm_cpu_src = torch.rand(BS, 4, 64, 64, device='cuda')
tw_cpu_src = torch.ones(BS, 4, device='cuda')
for _ in range(3):
    pose_pck_accuracy(to_numpy(pred_cpu_src), to_numpy(hm_cpu_src),
                      to_numpy(tw_cpu_src) > 0)
torch.cuda.synchronize()
t0 = time.perf_counter()
for _ in range(n):
    pose_pck_accuracy(to_numpy(pred_cpu_src), to_numpy(hm_cpu_src),
                      to_numpy(tw_cpu_src) > 0)
t_pck = (time.perf_counter() - t0) / n
L(f'  单独 to_numpy x3 + pose_pck_accuracy = {t_pck*1000:.1f} ms/iter')
L()

# ---------------- 3) 完整 train_step ---------------- #
L('=' * 78)
L('3) 完整 model.train_step()（含数据预处理、CPU<->GPU 搬运）')
L('=' * 78)
from mmpose.structures import PoseDataSample  # noqa: E402

ds = DMKPD_dataset(
    data_root='DMKPD_dataset', ann_file='splits/val.txt',
    data_prefix=dict(img='train/images'), metainfo=cfg.metainfo,
    serialize_data=False, pipeline=cfg.train_pipeline)

samples = [ds[i] for i in range(12)]
L(f'  样本数 = {len(samples)}, inputs={tuple(samples[0]["inputs"].shape)}')

from mmengine.optim import build_optim_wrapper  # noqa: E402
ow = build_optim_wrapper(model, cfg.optim_wrapper)

# bs=1 手动前向（绕过 dataloader），看"含预处理"的完整开销
def step_one(s):
    return model.train_step(
        dict(inputs=s['inputs'].unsqueeze(0),
             data_samples=[s['data_samples']]), ow)


for _ in range(3):
    step_one(samples[0])
torch.cuda.synchronize()
t0 = time.perf_counter()
for i in range(8):
    step_one(samples[i])
torch.cuda.synchronize()
t_step = (time.perf_counter() - t0) / 8
L(f'  train_step（bs=1，含预处理，无 dataloader） = {t_step*1000:.1f} ms')
L()

L('=' * 78)
L('汇总')
L('=' * 78)
L(f'  纯 GPU (bs=4)                = {t_gpu*1000:7.1f} ms')
L(f'  + compute_acc (bs=4)         = {t_acc*1000:7.1f} ms')
L(f'  真实训练每迭代 (bs=4)         =  1006.0 ms  (来自日志)')
L(f'  train_step (bs=1, 无 loader) = {t_step*1000:7.1f} ms')
L()

with open('_prof_out.txt', 'w', encoding='utf-8') as f:
    f.write('\n'.join(OUT))
