"""最小化排查：为什么 init_weights 后数值仍不匹配。

把所有结果写进文件，避免 mmengine 的警告刷屏淹没结论。
"""
import torch
from mmengine.config import Config
from mmengine.registry import init_default_scope
init_default_scope('mmpose')
from mmpose.registry import MODELS

CKPT = ('modelbase_test/'
        'td-hm_hrnet-w48_8xb32-210e_coco-256x192-0e67c616_20220913.pth')
ck = torch.load(CKPT, map_location='cpu')
ck_sd = ck.get('state_dict', ck)

cfg = Config.fromfile('config.py')

out = []
def L(s=''):
    out.append(str(s))

L('cfg.model.backbone.init_cfg = ' + repr(cfg.model['backbone'].get('init_cfg')))
L('cfg.load_from = ' + repr(cfg.get('load_from')))
L()

m = MODELS.build(cfg.model)
L('构造后 backbone._is_init = ' + repr(getattr(m.backbone, '_is_init', None)))
L('构造后 backbone.init_cfg = ' + repr(m.backbone.init_cfg))
L()

k = 'backbone.bn1.bias'
v_ckpt = ck_sd[k]
v_build = m.state_dict()[k].clone()

L(f'key = {k}')
L(f'  checkpoint 前4   = {v_ckpt[:4].tolist()}')
L(f'  构造后      前4   = {v_build[:4].tolist()}')
L(f'  equal ? {torch.equal(v_build, v_ckpt)}')
L(f'  allclose ? {torch.allclose(v_build, v_ckpt)}')
L(f'  max|diff| = {(v_build - v_ckpt).abs().max().item():.6e}')
L(f'  dtype: ckpt={v_ckpt.dtype} model={v_build.dtype}')
L(f'  shape: ckpt={tuple(v_ckpt.shape)} model={tuple(v_build.shape)}')
L()

# 统计：构造后全模型有多少键与 checkpoint 完全一致
sd_now = m.state_dict()
same = sum(1 for kk in ck_sd
           if kk in sd_now and torch.equal(sd_now[kk], ck_sd[kk]))
L(f'构造后（未显式 init_weights）与 checkpoint 完全一致的键数 = {same}')
L()

# 显式 init_weights 再看
m.init_weights()
sd_after = m.state_dict()
same2 = sum(1 for kk in ck_sd
            if kk in sd_after and torch.equal(sd_after[kk], ck_sd[kk]))
L(f'显式 init_weights 后一致的键数 = {same2}')
L()

# 手动 load_state_dict 做对照
m2 = MODELS.build({kk: vv for kk, vv in cfg.model.items()})
m2.backbone.init_cfg = None
sd_m2 = m2.state_dict()
sub = {kk: vv for kk, vv in ck_sd.items()
       if kk in sd_m2 and sd_m2[kk].shape == vv.shape}
info = m2.load_state_dict(sub, strict=False)
L(f'手动 load 部分权重: 加载了 {len(sub)} 个键, '
  f'missing={len(info.missing_keys)} unexpected={len(info.unexpected_keys)}')
sd_m2b = m2.state_dict()
same3 = sum(1 for kk in sub if torch.equal(sd_m2b[kk], ck_sd[kk]))
L(f'手动加载后与 checkpoint 完全一致的键数 = {same3} / {len(sub)}')

with open('_dbg_weight_out.txt', 'w', encoding='utf-8') as f:
    f.write('\n'.join(out))
print('done -> _dbg_weight_out.txt')

