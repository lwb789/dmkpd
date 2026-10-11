"""最终验证：预训练权重是否真的写进了模型。

用 Runner 的真实加载路径（load_from），然后逐个张量比对数值。
关键：`MODELS.build` 只构造模块、不加载权重，必须走 Runner 或
显式 load_checkpoint。这里直接用 Runner 的加载逻辑。
"""
import torch
from mmengine.config import Config
from mmengine.registry import init_default_scope
from mmengine.runner import Runner

init_default_scope('mmpose')

CKPT = ('modelbase_test/'
        'td-hm_hrnet-w48_8xb32-210e_coco-256x192-0e67c616_20220913.pth')

out = []


def L(s=''):
    out.append(str(s))


cfg = Config.fromfile('config.py')
# Runner.from_cfg 需要 work_dir；这里只为验证加载，随便给一个
cfg.work_dir = 'work_dirs/_weight_check'
cfg.log_level = 'ERROR'   # 降低日志噪音
L(f'load_from         = {cfg.load_from}')
L(f'backbone.init_cfg = {cfg.model["backbone"].get("init_cfg")}')
L()

runner = Runner.from_cfg(cfg)
model = runner.model
# ❗ Runner.from_cfg **不会**自动加载 load_from；
# load_or_resume() 只在 train()/val()/test() 内部被调用。
# 这里显式调一次，模拟训练时的真实行为。
runner.load_or_resume()
L('Runner 构建完成 + 显式 load_or_resume()（与训练时一致）')
L()

ck = torch.load(CKPT, map_location='cpu')
ck_sd = {k: v.cpu() for k, v in ck.get('state_dict', ck).items()}
# 模型可能在 GPU 上，比对前统一移到 CPU
sd = {k: v.detach().cpu() for k, v in model.state_dict().items()}

matched = [k for k in ck_sd if k in sd and ck_sd[k].shape == sd[k].shape]
skipped = [(k, tuple(ck_sd[k].shape), tuple(sd[k].shape))
           for k in ck_sd if k in sd and ck_sd[k].shape != sd[k].shape]

n_ok = sum(1 for k in matched if torch.equal(sd[k], ck_sd[k]))
L('键比对:')
L(f'  可加载键数         = {len(matched)}')
L(f'  数值完全一致的键数 = {n_ok}')
L(f'  形状不符被跳过的键 = {len(skipped)}')
for k, a, b in skipped:
    L(f'    {k}: ckpt{a} -> model{b}')
L()

for grp in ('backbone.', 'head.'):
    ks = [k for k in matched if k.startswith(grp)]
    ok = sum(1 for k in ks if torch.equal(sd[k], ck_sd[k]))
    L(f'  {grp:<12} 可加载={len(ks):<5} 数值一致={ok}')
L()

L('抽查具体张量:')
for k in ('backbone.bn1.bias',
          'backbone.conv1.weight',
          'backbone.stage4.3.branches.3.2.conv2.weight'):
    if k in sd and k in ck_sd:
        L(f'  {k}')
        L(f'    ckpt  = {ck_sd[k].ravel()[:3].tolist()}')
        L(f'    model = {sd[k].ravel()[:3].tolist()}')
        L(f'    一致  = {torch.equal(sd[k], ck_sd[k])}')
L()

L('head.final_layer（应被跳过、重新初始化）:')
for k in sd:
    if 'final_layer' in k:
        L(f'  {k}  shape={tuple(sd[k].shape)}  '
          f'|sum|={sd[k].abs().sum().item():.4f}')

L()
if n_ok == len(matched) and len(skipped) == 2:
    L('结论: 权重加载【成功】—— 1752 个 backbone 键全部写入，'
      '仅 head.final_layer 的 2 个键按预期跳过')
else:
    L(f'结论: 异常！一致 {n_ok}/{len(matched)}，跳过 {len(skipped)}')

with open('_weight_final_out.txt', 'w', encoding='utf-8') as f:
    f.write('\n'.join(out))
print('done')
