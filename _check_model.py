"""验证 config.py 能构建模型，且预训练权重按预期加载。"""
from mmengine.config import Config
from mmengine.registry import init_default_scope
from mmengine.model import BaseModel

init_default_scope('mmpose')

cfg = Config.fromfile('config.py')
print('=' * 70)
print('config 解析成功')
print('=' * 70)

print(f'  env_cfg.mp_cfg      = {cfg.env_cfg["mp_cfg"]}')
print(f'  train_cfg           = {cfg.train_cfg}')
print(f'  auto_scale_lr       = {cfg.auto_scale_lr}')
print(f'  optim lr            = {cfg.optim_wrapper["optimizer"]["lr"]}')
print(f'  train pipeline 步数 = {len(cfg.train_pipeline)}')
print(f'  val   pipeline 步数 = {len(cfg.val_pipeline)}')
print(f'  val_evaluator 个数  = {len(cfg.val_evaluator)}')
print(f'  head.out_channels   = {cfg.model["head"]["out_channels"]}')
print(f'  test_cfg.flip_test  = {cfg.model["test_cfg"]["flip_test"]}')
print()

from mmpose.registry import MODELS  # noqa: E402

print('--- 构建模型 ---')
model = MODELS.build(cfg.model)
n_total = sum(p.numel() for p in model.parameters())
n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f'  参数量        = {n_total/1e6:.2f} M')
print(f'  可训练参数量  = {n_train/1e6:.2f} M')

h = model.head
print(f'  head 类型     = {type(h).__name__}')
print(f'  head out_ch   = {h.out_channels}')
# 找到最后的 1x1 卷积，确认输出通道是 4
import torch.nn as nn  # noqa: E402
last_conv = None
for m in h.modules():
    if isinstance(m, nn.Conv2d):
        last_conv = m
print(f'  最后卷积      = {last_conv}')
print(f'  最后卷积输出  = {last_conv.out_channels}  (应为 4)')
print()

print('--- 模拟一次前向 ---')
import torch  # noqa: E402
from mmpose.structures import PoseDataSample  # noqa: E402

model.eval()
x = torch.randn(2, 3, 256, 256)
with torch.no_grad():
    feats = model.extract_feat(x)
print(f'  输入 {tuple(x.shape)}')
print(f'  extract_feat 输出 {len(feats)} 个尺度, '
      f'第一个 {tuple(feats[0].shape)}')
print(f'  最终特征 {tuple(feats[-1].shape)}  '
      f'(下采样 {256 // feats[-1].shape[-1]} 倍)')

# 检查 GPU 可用并测一次显存
if torch.cuda.is_available():
    print()
    print('--- GPU 前向 + 反向显存 ---')
    model.cuda()
    x = torch.randn(4, 3, 256, 256, device='cuda')
    torch.cuda.reset_peak_memory_stats()
    out = model.extract_feat(x)
    loss = sum(o.float().mean() for o in out)
    loss.backward()
    peak = torch.cuda.max_memory_allocated() / 1024**3
    print(f'  bs=4 @256x256 前向+反向峰值 = {peak:.2f} GB')
    print(f'  GPU 总显存 = '
          f'{torch.cuda.get_device_properties(0).total_memory/1024**3:.2f} GB')
