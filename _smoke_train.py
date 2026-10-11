"""小规模跑通训练：用极少量数据 + 2 epoch 验证全链路可训练。

目的不是精度，而是确认：
    预训练权重已生效 -> 能前向 -> 能反传 -> loss 在降
    -> val 能算出指标 -> 能存 checkpoint

与上一版的区别：这次**启用** load_from（上一版设为 None 做纯流程验证）。
判断"预训练是否生效"的依据：
  ① 权重比对：训练开始时 backbone 权重 == checkpoint
  ② loss 起点应明显低于随机初始化时的 0.0035
"""
import torch
from mmengine.config import Config
from mmengine.registry import init_default_scope
from mmengine.runner import Runner

init_default_scope('mmpose')

CKPT = ('modelbase_test/'
        'td-hm_hrnet-w48_8xb32-210e_coco-256x192-0e67c616_20220913.pth')


def main():
    cfg = Config.fromfile('config.py')

    # ---- 缩小规模，让一次跑完只要几分钟 ---- #
    cfg.train_cfg = dict(by_epoch=True, max_epochs=2, val_interval=1)
    cfg.train_dataloader['dataset']['indices'] = 100
    cfg.train_dataloader['dataset']['serialize_data'] = False
    cfg.train_dataloader['batch_size'] = 4
    cfg.train_dataloader['num_workers'] = 2

    cfg.val_dataloader['dataset']['indices'] = 40
    cfg.val_dataloader['dataset']['serialize_data'] = False
    cfg.val_dataloader['batch_size'] = 4
    cfg.val_dataloader['num_workers'] = 2

    cfg.default_hooks['visualization']['enable'] = False
    cfg.default_hooks['checkpoint'] = dict(
        type='CheckpointHook', interval=1, max_keep_ckpts=1)
    cfg.param_scheduler[0] = dict(
        type='LinearLR', begin=0, end=20, start_factor=0.001, by_epoch=False)
    cfg.param_scheduler[1] = dict(
        type='MultiStepLR', begin=0, end=2, milestones=[1], gamma=0.1,
        by_epoch=True)

    cfg.work_dir = 'work_dirs/_smoke'
    cfg.log_level = 'INFO'
    # ⭐ 这次启用预训练（config.py 里已设好，这里确保不被覆盖）
    assert cfg.load_from, 'cfg.load_from 应为预训练权重路径'

    print('=' * 70)
    print('小规模训练验证（启用预训练）')
    print(f'  load_from    = {cfg.load_from}')
    print(f'  train 实例数 = 100, val = 40, max_epochs = 2')
    print(f'  batch_size = 4, num_workers = 2')
    print('=' * 70)

    runner = Runner.from_cfg(cfg)

    # ---- ① 训练前：确认 backbone 权重 == checkpoint ---- #
    ck = torch.load(CKPT, map_location='cpu')
    ck_sd = ck.get('state_dict', ck)
    runner.load_or_resume()     # 显式加载（train() 内部也会调，已加载则跳过）
    sd = {k: v.detach().cpu() for k, v in runner.model.state_dict().items()}

    matched = [k for k in ck_sd
               if k in sd and ck_sd[k].shape == sd[k].shape]
    n_same = sum(1 for k in matched if torch.equal(sd[k], ck_sd[k]))
    print()
    print(f'[预训练校验] 可加载键 {len(matched)} 个，'
          f'其中与 checkpoint 逐值相等 {n_same} 个')
    w0 = sd['backbone.bn1.bias'].clone()
    print(f'[预训练校验] backbone.bn1.bias[:3] = '
          f'{w0.ravel()[:3].tolist()}')
    print(f'[预训练校验] 结论: '
          f'{"预训练权重已生效" if n_same == len(matched) else "未生效!"}')
    print()

    runner.train()

    # ---- ② 训练后：权重应已被更新（证明预训练是被"接着训"而非冻结）---- #
    sd1 = {k: v.detach().cpu()
           for k, v in runner.model.state_dict().items()}
    w1 = sd1['backbone.bn1.bias']
    print()
    print(f'[训练后校验] backbone.bn1.bias[:3] = {w1.ravel()[:3].tolist()}')
    print(f'[训练后校验] 与训练前相比是否已更新: '
          f'{not torch.equal(w0, w1)}')
    print(f'[训练后校验] 与预训练 checkpoint 相比是否已偏移: '
          f'{not torch.equal(w1, ck_sd["backbone.bn1.bias"])}')
    print()
    print('训练完成')


if __name__ == '__main__':
    main()
