# -*- coding: utf-8 -*-
"""验证 RTMDet 预训练权重是否真的加载进模型。

背景：本项目在姿态模型上踩过一次坑 —— `backbone.init_cfg=Pretrained`
只加载了 0/1752 个键，且 strict=False 静默跳过只打一条警告。
所以这里用**逐张量 torch.equal 比对**来证明权重真的进去了，
而不是只看 load_state_dict 返回的 missing/unexpected 数量。
"""
import os

import torch
from mmengine.config import Config
from mmengine.registry import init_default_scope
import mmdet  # noqa

cfg = Config.fromfile('config_det.py')
init_default_scope('mmdet')
from mmdet.registry import MODELS

model = MODELS.build(cfg.model)
model.init_weights()
sd_before = {k: v.clone() for k, v in model.state_dict().items()}

ckpt = cfg.load_from
print('checkpoint:', ckpt, os.path.isfile(ckpt))
raw = torch.load(ckpt, map_location='cpu')
sd_ck = raw.get('state_dict', raw)

# 逐键统计
tgt = model.state_dict()
matched, mismatched, missing = [], [], []
for k, v in sd_ck.items():
    if k in tgt and tgt[k].shape == v.shape:
        matched.append(k)
    elif k in tgt:
        mismatched.append((k, tuple(v.shape), tuple(tgt[k].shape)))
    else:
        missing.append(k)
not_in_ck = [k for k in tgt if k not in sd_ck]

print()
print('=' * 74)
print('权重比对（load_from 前的原始比对）')
print('=' * 74)
print(f'  checkpoint 键数           {len(sd_ck)}')
print(f'  模型 state_dict 键数      {len(tgt)}')
print(f'  形状完全一致可加载        {len(matched)}')
print(f'  形状不符（预期=输出层）   {len(mismatched)}')
print(f'  checkpoint 里有、模型没有 {len(missing)}')
print(f'  模型有、checkpoint 没有   {len(not_in_ck)}')

if mismatched:
    print()
    print('  形状不符明细：')
    for k, a, b in mismatched:
        print(f'    {k}\n       ckpt {a} vs model {b}')
if not_in_ck:
    print()
    print('  模型有但 ckpt 没有（将随机初始化）：')
    for k in not_in_ck[:20]:
        print(f'    {k} {tuple(tgt[k].shape)}')

# 真正加载 —— 必须走 mmengine 的 load_checkpoint，而不是
# torch.load_state_dict(strict=False)！
#
# 实测坑：torch 的 strict=False **只忽略「键」层面的缺失/多余**，
# 对「同名但形状不同」的键**依然抛 RuntimeError**。而我们的
# bbox_head.rtm_cls.* 正是同名不同形（80 类 -> 1 类）。
# mmengine.load_checkpoint 内部会先做形状过滤再加载，这才是
# Runner(load_from=...) 实际走的路径。
from mmengine.runner.checkpoint import load_checkpoint

before_keys = set(sd_before)
load_checkpoint(model, ckpt, map_location='cpu', strict=False)
sd_after = model.state_dict()

print()
print('=' * 74)
print('[逐张量 torch.equal 抽检] 证明真的写进去了')
print('=' * 74)
import random
rng = random.Random(0)
sample = rng.sample(matched, min(12, len(matched)))
n_equal = 0
for k in sample:
    same = torch.equal(sd_after[k], sd_ck[k].to(sd_after[k].dtype))
    n_equal += same
    print(f'  [{"OK" if same else "BAD"}] {k}')
print(f'  -> {n_equal}/{len(sample)} 完全相等')

# 抽检「没被加载」的层是否真的变了（初始随机 vs 加载后）
print()
print('  [反例检查] 形状不符的层应当**没有**被加载')
for k, _, _ in mismatched[:5]:
    changed = not torch.equal(sd_before[k], sd_after[k])
    print(f'    {k}: {"被改了(异常!)" if changed else "保持初始值(正确)"}')
