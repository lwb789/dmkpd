# ===================================================================== #
# 运行时（继承官方 default_runtime 提供 hooks / visualizer / env_cfg）
# ===================================================================== #
_base_ = ['open-mmlab/mmpose/configs/_base_/default_runtime.py']

default_scope = 'mmpose'

# ⚠️ Windows 覆盖：default_runtime.py 里是 mp_start_method='fork'，
# 而 Windows 只有 spawn（没有 fork），不覆盖会导致 dataloader 多进程崩溃
env_cfg = dict(
    cudnn_benchmark=False,
    mp_cfg=dict(mp_start_method='spawn', opencv_num_threads=0),
    dist_cfg=dict(backend='nccl'))

# ===================================================================== #
# 预训练权重
# ===================================================================== #
# 【必须用 load_from，不能用 backbone 的 init_cfg】实测对比：
#
#   backbone.init_cfg=Pretrained   -> 加载 0 / 1752 个键  ❌
#     （init_cfg 作用在 backbone 子模块上，它拿带 `backbone.` 前缀的键
#       去匹配无前缀的 `bn1.bias`，全部失配；strict=False 静默跳过，
#       只打一条 "do not match exactly" 警告，非常容易被忽略）
#
#   load_from=<整个 .pth>          -> 加载 1752 个键，仅跳过 2 个  ✅
#     （Runner 级别加载，键与**整个模型**的 state_dict 比对，前缀能对上；
#      被跳过的 2 个正是 head.final_layer 的 weight/bias，
#      因为它是 COCO 的 17 通道，我们是 4 通道，符合预期）
load_from = ('modelbase_test/'
             'td-hm_hrnet-w48_8xb32-210e_coco-256x192-0e67c616_20220913.pth')

# ===================================================================== #
# 数据流水线（增强 -> 变换 -> 编码 -> 打包）
# ===================================================================== #
#
# 输入尺寸取正方形：DM 码本身接近正方形（实测 bbox 长短边比中位 1.05），
# 而 TopdownAffine 会把采样窗口拉成输入宽高比，故正方形输入能让目标
# 在画面上占据最大面积（200px 的码：256x256 占 44.4%，192x256 只占 33.3%）。
input_size = (256, 256)
# 热力图下采样 4 倍。峰值量化误差约 ±2px（输入尺度），实测峰值与关键点
# 最大偏差 2.45px，在 64x64 下属于正常量化误差。
heatmap_size = (64, 64)

# 编码器：把 256 画布上的关键点坐标编码成 64x64 的高斯热力图
codec = dict(
    type='MSRAHeatmap',
    input_size=input_size,
    heatmap_size=heatmap_size,
    sigma=2)

# 增强参数（取值依据）：
#   padding=1.5     采样窗口 = 1.5 倍 bbox。用于 ①容纳旋转 ②保留背景
#                   ③容忍 bbox 误差 ④给随机平移留余量
#   shift_factor=.05 平移幅度。DM 码在窗口内总是完整的，平移没有真实
#                   分布要模拟，属于「纯成本」，故取小值
#   scale=(.9,1.1)  缩放。区间中心 = 1.0，使训练时的有效窗口尺度
#                   u = padding*s 的中心 (1.5) 与测试时 (1.5) 一致，
#                   避免训测分辨率失配
#   rotate=30       旋转。用来打散「top_left = 图上最左上角」的位置
#                   捷径，迫使模型依赖编码方向而非绝对坐标。30° 足够，
#                   且比 45° 更省几何预算
#   实测：以上组合在 300 个真实样本上出界率 0%
train_pipeline = [
    dict(type='LoadImage'),
    dict(type='GetBBoxCenterScale', padding=1.5),
    dict(
        type='RandomBBoxTransform',
        shift_factor=0.05,
        shift_prob=0.3,
        scale_factor=(0.9, 1.1),
        scale_prob=1.0,
        rotate_factor=30,
        rotate_prob=1.0),
    dict(type='TopdownAffine', input_size=input_size),
    # 必须紧跟 TopdownAffine：它消费 transformed_keypoints / input_size
    # 两个键，放在前面会静默失效。作用是把被裁出画面的角点权重置 0，
    # 避免 codec 走出「target 全 0 但权重非 0」的负监督分支
    dict(type='CheckKeypointOutOfBounds', out_dir='work_dirs/dmkpd'),
    dict(
        type='GenerateTarget',
        encoder=codec,
        use_dataset_keypoint_weights=True),
    dict(type='PackPoseInputs'),
]

# 验证集：不做任何随机增强，只做同样的裁切
# **不含 GenerateTarget**：模型的 `predict()`（val/test 用）只用 inputs，
# 热力图只在 `loss()`（训练）时才需要。官方 COCO 配置的 val_pipeline
# 同样只有这几步。少算一次高斯热力图，val 更快。
val_pipeline = [
    dict(type='LoadImage'),
    dict(type='GetBBoxCenterScale', padding=1.5),
    dict(type='TopdownAffine', input_size=input_size),
    dict(type='PackPoseInputs'),
]

# 测试集：只裁切与打包，**不做编码**（没有 GT 关键点，GenerateTarget 会报错）
# 注意：进入本流水线前，每条 data_info 必须已经带有 bbox，
#       而测试集图片没有标注 -> bbox 需由外部检测器提供（见后续工作）
test_pipeline = [
    dict(type='LoadImage'),
    dict(type='GetBBoxCenterScale', padding=1.5),
    dict(type='TopdownAffine', input_size=input_size),
    dict(type='PackPoseInputs'),
]

# ===================================================================== #
# 数据集与 dataloader
# ===================================================================== #
dataset_type = 'DMKPD_dataset'
metainfo = dict(
    from_file='open-mmlab/mmpose/configs/_base_/datasets/DMKPD_meta.py')

# ann_file='splits/*.txt' 是**图级划分清单**（由 make_split.py 生成，
# 按图随机、无放回、种子固定）。清单里写相对 data_root 的 json 路径。
# 抽中的图片，其**全部实例**一起进同一个 split，避免同图泄漏。
train_dataloader = dict(
    batch_size=4,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        data_root='DMKPD_dataset',
        ann_file='splits/train.txt',
        data_prefix=dict(img='train/images'),
        metainfo=metainfo,
        pipeline=train_pipeline,
        # 被剔除的异常实例明细会写到该目录，便于事后核对标注质量
        out_dir='work_dirs/dmkpd'))

val_dataloader = dict(
    batch_size=4,
    num_workers=4,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root='DMKPD_dataset',
        ann_file='splits/val.txt',
        data_prefix=dict(img='train/images'),
        metainfo=metainfo,
        pipeline=val_pipeline,
        test_mode=True))

# 测试集只有 300 张图片、无标注，ann_file='' 时数据集会直接枚举图片目录
test_dataloader = dict(
    batch_size=4,
    num_workers=4,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root='DMKPD_dataset',
        ann_file='',
        data_prefix=dict(img='test'),
        metainfo=metainfo,
        pipeline=test_pipeline,
        test_mode=True))

# visualizer settings
# browse_dataset.py 等工具会读取 cfg.visualizer，这里补上
vis_backends = [dict(type='LocalVisBackend')]
visualizer = dict(
    name='visualizer',
    type='PoseLocalVisualizer',
    vis_backends=vis_backends)

# ===================================================================== #
# 模型
# ===================================================================== #
# 沿用官方 HRNet-W48 作为 COCO 人体预训练骨干（63.59M 参数）。
#
# 为什么用 W48 而不是更小的 W18/W32：
#   ① 有 COCO 预训练权重可直接复用（backbone 1754 个键全部匹配，
#      只有 head.final_layer 因输出通道 17->4 需要重新初始化）
#   ② 实测显存充足：256x256 + bs=4 前向反向仅 1.38 GB（GTX 1650 共 4 GB）
#   ③ 是官方基线，复现性好，出问题便于对照
# 已知风险：train 只有 20494 实例（COCO 是 15 万），W48 可能过拟合。
# 对策：先跑通拿基线；若 train loss 降而 val AP 不涨，再降 W32/W18 对照。
model = dict(
    type='TopdownPoseEstimator',
    data_preprocessor=dict(
        type='PoseDataPreprocessor',
        # 与 COCO 一致的标准化参数（通用图像统计量，与任务无关）
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=True),
    backbone=dict(
        type='HRNet',
        in_channels=3,
        extra=dict(
            stage1=dict(
                num_modules=1,
                num_branches=1,
                block='BOTTLENECK',
                num_blocks=(4, ),
                num_channels=(64, )),
            stage2=dict(
                num_modules=1,
                num_branches=2,
                block='BASIC',
                num_blocks=(4, 4),
                num_channels=(48, 96)),
            stage3=dict(
                num_modules=4,
                num_branches=3,
                block='BASIC',
                num_blocks=(4, 4, 4),
                num_channels=(48, 96, 192)),
            stage4=dict(
                num_modules=3,
                num_branches=4,
                block='BASIC',
                num_blocks=(4, 4, 4, 4),
                num_channels=(48, 96, 192, 384))),
        # ⚠️ 这里**不能**用 `init_cfg=dict(type='Pretrained', checkpoint=...)`！
        #
        # 原因（实测踩坑）：init_cfg 是作用在 **backbone 这个子模块**上的，
        # 它会把 checkpoint 的键与 `backbone.state_dict()` 的键比对。
        # 而我们手上的 .pth 是**整个 TopdownPoseEstimator 的权重**，
        # 键带 `backbone.` 前缀（如 `backbone.bn1.bias`），
        # 子模块的键却是无前缀的（`bn1.bias`）→ 全部失配，
        # 且 load_state_dict(strict=False) 会**静默跳过**（只打一条
        # "do not match exactly" 的警告，不报错，极易被忽略）。
        #
        # 正确做法是用 Runner 级别的 `load_from`（见文件末尾），
        # 它是把权重的键与**整个模型**的 state_dict 比对，前缀能对上。
        init_cfg=None),
    head=dict(
        type='HeatmapHead',
        in_channels=48,
        # ⭐ 关键改动：COCO 是 17 个关键点，我们是 4 个角点
        out_channels=4,
        deconv_out_channels=None,
        # 按 target_weight 加权。出界角点的 visible=0 -> 权重 0 ->
        # 该通道 loss 与梯度恰好为 0，是"缺标签"而非"错标签"
        loss=dict(type='KeypointMSELoss', use_target_weight=True),
        decoder=codec),
    test_cfg=dict(
        # ⭐ 必须关闭：DM 码不存在镜像形态，水平翻转是非物理的。
        # 若开启，翻转后的热力图会被强行翻转回来并平均，污染预测
        flip_test=False,
        flip_mode='heatmap',
        shift_heatmap=True))

# ===================================================================== #
# 训练配置
# ===================================================================== #
# 官方 COCO 用 210 epoch / 150k 张图。我们 train 约 9122 个实例（约 1/7）。
# 先定 50 epoch 跑通看曲线，不够再续训（resume）。
train_cfg = dict(by_epoch=True, max_epochs=50, val_interval=5)
val_cfg = dict()
test_cfg = dict()

# 官方配置 base_batch_size=512 是靠 8 卡 x 32 达到的；
# 我们单卡 bs=4，若不关掉自动缩放，lr 会被缩到 5e-4 * (4/512) ≈ 4e-6，
# 学习率过小会导致几乎不收敛。这里显式关掉，learning rate 由下面直接指定。
auto_scale_lr = dict(enable=False)

# Adam + 5e-4（沿用官方）。bs 从 32 降到 4，梯度噪声更大，
# 若发现 loss 震荡明显可再降；先用官方值作起点
optim_wrapper = dict(optimizer=dict(type='Adam', lr=5e-4))

param_scheduler = [
    # 前 500 iter 线性 warm-up，避免开局大 lr 破坏预训练权重
    dict(
        type='LinearLR',
        begin=0,
        end=500,
        start_factor=0.001,
        by_epoch=False),
    # 在第 40 个 epoch 降 lr（总 50 epoch 的前 80% 结束处）
    dict(
        type='MultiStepLR',
        begin=0,
        end=50,
        milestones=[40],
        gamma=0.1,
        by_epoch=True),
]

# 每 5 个 epoch 验证一次；保存 val AP 最好的模型
default_hooks = dict(
    checkpoint=dict(
        type='CheckpointHook',
        interval=5,
        save_best='coco/AP',
        rule='greater',
        max_keep_ckpts=2))

# ===================================================================== #
# 评估器
# ===================================================================== #
# 三轨并行：
#   CocoMetric      -> AP/AP.5/AP.75/AR，**与官方 leaderboard 口径完全一致**
#   PCKAccuracy     -> 按 bbox 尺寸归一化的通过率，直观可解释
#   EPE             -> 端点误差（像素），最直接的定位精度
#
# 注意 CocoMetric 的 ann_file=None + 数据集提供 raw_ann_info，
# 它会现场把 GT 转成 COCO json 再评测（不依赖额外的标注文件）
val_evaluator = [
    dict(type='CocoMetric', ann_file=None),
    # 两个 PCKAccuracy 都输出名为 'PCK' 的指标，会撞名报
    # "multiple evaluation results with the same metric name"，
    # 所以各给一个 prefix 来区分
    dict(type='PCKAccuracy', thr=0.05, norm_item='bbox', prefix='PCK@0.05'),
    dict(type='PCKAccuracy', thr=0.10, norm_item='bbox', prefix='PCK@0.10'),
    dict(type='EPE', prefix='EPE'),
]

# 测试集没有标注，无法算指标，只导出预测结果
test_evaluator = [
    dict(type='CocoMetric', ann_file=None, format_only=True,
         outfile_prefix='work_dirs/dmkpd/test_preds'),
]
