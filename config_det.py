# -*- coding: utf-8 -*-
"""DMKPD 检测器配置：RTMDet-tiny，单类（DPM），检测提供 topdown 姿态的 bbox。

===================================================================
为什么需要检测器
===================================================================
姿态模型是 topdown（HRNet heatmap），它**不自带目标定位能力**：
输入必须已经是一张围绕某个 DM 码的裁剪图。测试集 300 张没有标注，
所以必须由检测器先把每个码框出来，再逐框裁剪送进姿态模型。
检测器只输出 bbox，不输出四角 —— 四个角的顺序由姿态模型负责。

===================================================================
关键设计决策（每条都有实测依据）
===================================================================

【1】输入 1024x1024，不是官方默认 640
  keep_ratio 缩放会把长边归一到 S，于是**不同来源的图在输入上尺寸趋同**：

    来源                    码边长中位    S=1024 时
    adv_final (512x512)       76 px       152 px
    numeric   (1702x1276)    255 px       205 px
    other     (~1280)        189 px       151 px

  两档差 152 vs 205，只差 1.35 倍 —— 说明**混合训练是合理的**，
  不像「绝对像素」视角看起来差 3.4 倍那么离谱。
  选 1024 而非 640：640 时中位码长只有 95-128px，小码（p05=36px）
  在 640 下只剩 64px；1024 下小码 65px、大码 205px，更安全。
  实测显存：1024x1024 bs=4 = **3.21 GB**（4GB 卡能放下），
  0.516 s/iter，9110 图约 19.6 min/epoch。

【2】删除 Mosaic / MixUp
  官方 640 配置依赖 mosaic 把「大图中的小目标」拼成 640 的密集场景。
  我们本来就是「大图 + 稀疏目标」（平均 2.25 个/图），
  而且 mosaic 会把小尺寸的 512 档图放大拼进来，破坏尺度统计。
  更关键的是：**删掉 mosaic 后管线变成纯几何的 resize+pad**，
  与测试集管线完全同构，杜绝训测失配。

【3】不做 RandomFlip
  DM 码不存在镜像形态，翻转是非物理的（与姿态配置里
  `flip_test=False` 的理由一致）。左右翻转还会把码内的
  L 形定位图案搬到错误角上，属于构造不存在的样本。

【4】不做 RandomCrop
  现场图里码可能贴边（p05 相对尺寸 6.78% 就有贴边的），
  随机裁剪会切掉目标或把码裁成残块，得到「半个码」的伪标签。
  用户也明确要求不裁剪大图以保持码结构完整 —— 这里同样遵守。

【5】步长 / 学习率随 bs 线性缩放
  官方 base_lr=0.004 @ bs=32。我们用 bs=4 -> 0.004*(4/32)=5e-4。
  RTMDet 用 AdamW，对 lr 线性缩放敏感度低于 SGD，
  但按比例缩放仍是最稳的起点。
  （注意与姿态配置不同：那里显式关掉了 auto_scale_lr，
   这里直接手算写入，逻辑更透明。）

【6】关掉 SyncBN
  官方配置 norm_cfg=SyncBN，那是为多卡 DDP 设计的。单卡跑 SyncBN
  虽然能 work（退化成普通 BN），但会带来不必要的通信包装与
  潜在的非确定性。改为普通 BN（mmdet 里不写 norm_cfg 即默认 BN）。

【7】评估只用 bbox AP，且**必须用真实尺度 val**
  val.txt 里 1702 档只有 135 张，且验证时缩到 1024，码会变小。
  为了回答「现场真实尺度下检测行不行」，额外准备了
  instances_holdout1702.json（400 张纯 1702 档），训练时不参与。
"""
_base_ = [
    'mmdet/.mim/configs/_base_/default_runtime.py',
    'mmdet/.mim/configs/_base_/schedules/schedule_1x.py'
]

# =================================================================== #
# ⚠️ 关于上面两行 _base_：路径里的 `mmdet/` 是一个**指向
#    `.venv/Lib/site-packages/mmdet` 的 junction**（Windows 目录联接）。
#    原因：mmdet 的官方 _base_ 配置只存在于已安装包内（`.mim/`），
#    不在工作区里；而 mmengine 的配置加载器要求 base 路径能从**工作区
#    根目录**解析。用 junction 可以让配置保持可读、可提交，
#    不必把这两百行 runtime/schedule 复制进仓库。
#
#    创建方式（PowerShell）：
#      New-Item -ItemType Junction -Path mmdet `
#        -Target "$PWD\.venv\Lib\site-packages\mmdet"
#
#    如果不想用 junction，等价做法是把 _base_ 换成绝对路径，
#    但那样配置文件就绑死在某个机器上了。
# =================================================================== #

# =================================================================== #
# 预训练权重：RTMDet-tiny 的 COCO 完整权重
# =================================================================== #
# 用**整个检测器**的 COCO 权重（不是只加载 ImageNet backbone），
# 因为 COCO 已包含大量「密集小目标」的检测先验；即便 COCO 没有 DM 码，
# 特征提取与分类头的初始化也远好于随机。
#
# ⚠️ 这里用 load_from 而不是 backbone.init_cfg：
#     与姿态配置踩过的坑完全相同 —— init_cfg 只作用于 backbone 子模块，
#     加载「整个模型」的 .pth 时键前缀对不上会静默跳过。
#     唯一会因类别数不符被跳过的，就是 bbox_head 的分类/回归输出层。
load_from = ('modelbase_test/'
             'rtmdet_tiny_8xb32-300e_coco_20220902_112414-78e30dcc.pth')

default_scope = 'mmdet'

# Windows 必须用 spawn（与姿态配置一致）
env_cfg = dict(
    cudnn_benchmark=False,
    mp_cfg=dict(mp_start_method='spawn', opencv_num_threads=0),
    dist_cfg=dict(backend='nccl'))

# =================================================================== #
# 输入尺寸
# =================================================================== #
IMG = (1024, 1024)
PAD_VAL = dict(img=(114, 114, 114))

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    # 短边/长边等比缩放到 1024，再 pad。**与测试管线同构**，无随机几何
    dict(type='Resize', scale=IMG, keep_ratio=True),
    dict(type='Pad', size=IMG, pad_val=PAD_VAL),
    # 只保留光照扰动（不改几何、不改尺度统计）
    dict(type='PhotoMetricDistortion'),
    dict(type='PackDetInputs'),
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', scale=IMG, keep_ratio=True),
    dict(type='Pad', size=IMG, pad_val=PAD_VAL),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='PackDetInputs',
         meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                    'scale_factor')),
]

# 测试集没有标注，LoadAnnotations 会报错，单独给一份
test_pipeline_noann = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', scale=IMG, keep_ratio=True),
    dict(type='Pad', size=IMG, pad_val=PAD_VAL),
    dict(type='PackDetInputs',
         meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                    'scale_factor')),
]

# =================================================================== #
# 数据集与 dataloader
# =================================================================== #
# data_root='.'：file_name 在 json 里就是相对工作区根的路径，干净
metainfo = dict(classes=('DPM', ))

train_dataloader = dict(
    batch_size=4,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type='CocoDataset',
        metainfo=metainfo,
        data_root='.',
        ann_file='det_dataset/annotations/instances_train.json',
        data_prefix=dict(img=''),
        pipeline=train_pipeline))

val_dataloader = dict(
    batch_size=4,
    num_workers=4,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type='CocoDataset',
        metainfo=metainfo,
        data_root='.',
        ann_file='det_dataset/annotations/instances_val.json',
        data_prefix=dict(img=''),
        pipeline=test_pipeline,
        test_mode=True))

test_dataloader = dict(
    batch_size=4,
    num_workers=4,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type='CocoDataset',
        metainfo=metainfo,
        data_root='.',
        ann_file='det_dataset/annotations/instances_test.json',
        data_prefix=dict(img=''),
        pipeline=test_pipeline_noann,
        test_mode=True))

val_evaluator = dict(
    type='CocoMetric',
    ann_file='det_dataset/annotations/instances_val.json',
    metric='bbox',
    format_only=False)
test_evaluator = dict(
    type='CocoMetric',
    ann_file='det_dataset/annotations/instances_test.json',
    metric='bbox',
    format_only=True,
    outfile_prefix='work_dirs/dmkpd_det/test_preds')

# =================================================================== #
# 模型
# =================================================================== #
# 从 tiny 的官方配置继承结构，只改必要项
model = dict(
    type='RTMDet',
    data_preprocessor=dict(
        type='DetDataPreprocessor',
        mean=[103.53, 116.28, 123.675],
        std=[57.375, 57.12, 58.395],
        bgr_to_rgb=False,
        batch_augments=None),
    backbone=dict(
        type='CSPNeXt',
        arch='P5',
        expand_ratio=0.5,
        deepen_factor=0.167,
        widen_factor=0.375,
        channel_attention=True,
        # 单卡：用普通 BN，不用官方为 DDP 准备的 SyncBN
        norm_cfg=dict(type='BN'),
        act_cfg=dict(type='SiLU', inplace=True),
        init_cfg=None),
    neck=dict(
        type='CSPNeXtPAFPN',
        in_channels=[96, 192, 384],
        out_channels=96,
        num_csp_blocks=1,
        expand_ratio=0.5,
        norm_cfg=dict(type='BN'),
        act_cfg=dict(type='SiLU', inplace=True)),
    bbox_head=dict(
        type='RTMDetSepBNHead',
        num_classes=1,
        in_channels=96,
        stacked_convs=2,
        feat_channels=96,
        anchor_generator=dict(
            type='MlvlPointGenerator', offset=0, strides=[8, 16, 32]),
        bbox_coder=dict(type='DistancePointBBoxCoder'),
        loss_cls=dict(
            type='QualityFocalLoss',
            use_sigmoid=True,
            beta=2.0,
            loss_weight=1.0),
        loss_bbox=dict(type='GIoULoss', loss_weight=2.0),
        with_objectness=False,
        exp_on_reg=False,
        share_conv=True,
        pred_kernel_size=1,
        norm_cfg=dict(type='BN'),
        act_cfg=dict(type='SiLU', inplace=True)),
    train_cfg=dict(
        assigner=dict(type='DynamicSoftLabelAssigner', topk=13),
        allowed_border=-1,
        pos_weight=-1,
        debug=False),
    test_cfg=dict(
        nms_pre=30000,
        min_bbox_size=0,
        score_thr=0.001,
        nms=dict(type='nms', iou_threshold=0.65),
        max_per_img=300))

# =================================================================== #
# 训练配置
# =================================================================== #
# 官方 300 epoch @ bs32，8 卡 = 1500 epoch-equivalent。
# 我们 bs=4 -> 每 epoch 的梯度步数是官方的 1/8，
# 但单卡总步数 = 8710/4 = 2178 iter/epoch，50 epoch = 108900 iter，
# 而官方 300 epoch * (118287/32) = 约 1.1M iter。我们约 1/10。
# 数据量也是 1/6（COCO 是 118k 图）。所以 50 epoch 规模相当。
max_epochs = 60
base_lr = 0.004 * (4 / 32)      # = 5e-4

# ⚠️ base（schedule_1x.py）里已有 `train_cfg = dict(type='EpochBasedTrainLoop',
#    max_epochs=12, val_interval=1)`。这里**只能改 max_epochs/val_interval**，
#    如果再写 `by_epoch=True` 会报
#      RuntimeError: Only one of `type` or `by_epoch` can exist in `loop_cfg`
#    （实测踩过）。要么显式写 type，要么写 by_epoch，不能同时出现。
train_cfg = dict(
    type='EpochBasedTrainLoop', max_epochs=max_epochs, val_interval=5)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

# ⚠️ 必须写 `_delete_=True`！
#    mmengine 对 dict 是**递归合并**而非替换。base（schedule_1x.py）里是
#      optimizer=dict(type='SGD', lr=0.02, momentum=0.9, weight_decay=1e-4)
#    如果只写 optimizer=dict(type='AdamW', lr=..., weight_decay=...)，
#    合并后 SGD 的 `momentum=0.9` 会残留下来，报
#      TypeError: AdamW.__init__() got an unexpected keyword argument 'momentum'
#    （实测踩过）。加 `_delete_=True` 让整个 optim_wrapper 被替换掉。
optim_wrapper = dict(
    _delete_=True,
    type='OptimWrapper',
    optimizer=dict(type='AdamW', lr=base_lr, weight_decay=0.05),
    paramwise_cfg=dict(
        norm_decay_mult=0, bias_decay_mult=0, bypass_duplicate=True))

param_scheduler = [
    # 前 500 iter 线性 warmup
    dict(type='LinearLR', start_factor=1.0e-5, by_epoch=False, begin=0,
         end=500),
    # 后半程 cosine 退火到 base_lr*0.05
    dict(type='CosineAnnealingLR', eta_min=base_lr * 0.05,
         begin=max_epochs // 2, end=max_epochs, T_max=max_epochs // 2,
         by_epoch=True, convert_to_iter_based=True),
]

default_hooks = dict(
    checkpoint=dict(
        interval=5,
        max_keep_ckpts=3,
        save_best='coco/bbox_mAP',
        rule='greater'),
    logger=dict(interval=50))
# ⚠️ 不要在这里写 `visualizer=...`！`visualizer` 不是 hook，
#    放进 default_hooks 会被当作 hook 去注册，报
#    `KeyError: 'DetLocalVisualizer is not in the mmdet::hook registry'`。
#    base（default_runtime.py）顶层已有 visualizer，无需重复。
#
# ⚠️ DetVisualizationHook 没有 `enable` 参数（与 mmpose 的不同），
#    传 enable=False 会 TypeError；传 visualization=None 会触发
#    mmengine 的 `assert hook is not None`。base 里未启用它，不写即可。

# EMA：RTMDet 官方必备。对检测的定位精度提升明显，
# 且我们 epoch 数远少于官方，EMA 能显著平滑后期波动
custom_hooks = [
    dict(type='EMAHook', ema_type='ExpMomentumEMA', momentum=0.0002,
         update_buffers=True, priority=49),
]
