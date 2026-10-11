# DMKPD 项目状态（长安大学 AI 创新大赛）

> 最后更新：2026-10-10
> 本文件记录当前成果、文件清单、已知问题与下一步方向。

---

## 一、任务与方案

**任务**：Data Matrix（DM）码 **4 角关键点检测**。

**方案：两阶段（检测 → 姿态）**

```
测试图 1702x1276（含多个 DM 码，无标注）
   │
   ├─[RTMDet-tiny]  1 类 DPM，输入 1024x1024
   │    Resize(keep_ratio) + Pad -> 检测框 -> 坐标还原到原图
   │
   └─[HRNet-W48 + HeatmapHead(4ch)]  逐个框 topdown
        GetBBoxCenterScale(padding=1.5) -> TopdownAffine(256)
        -> 4 通道 64x64 热力图 -> decode -> 原图像素坐标的四角
   │
   └─ 导出：每图一个 txt，行格式 `DPM conf x1 y1 x2 y2 x3 y3 x4 y4`
```

**为什么必须两阶段**：mmpose 的 topdown 模型是 HRNet heatmap，**不自带目标定位能力**，
输入必须已经是围绕单个目标的裁剪图。而测试集 300 张无标注 → 必须有检测器提供 bbox。

---

## 二、当前指标

### 姿态（HRNet-W48，5 epoch）

val 1101 实例，用 **GT bbox** 裁切：

| 指标 | 值 |
|---|---|
| **coco/AP** | **0.9377** |
| coco/AP.5 | 0.9585 |
| coco/AP.75 | 0.9485 |
| coco/AR | 0.9428 |
| PCK@0.05 | 0.8812 |
| PCK@0.10 | 0.9353 |
| EPE | 5.41 px（中位 1.36 px） |

### 检测（RTMDet-tiny，**仅 1 epoch 冒烟**）

| 指标 | 值 |
|---|---|
| **coco/bbox_mAP** | **0.7840** |
| bbox_mAP_50 | 0.9400 |
| bbox_mAP_75 | 0.8820 |
| bbox_mAP_s（小目标） | 0.4770 ← 最弱 |
| bbox_mAP_m | 0.7800 |
| bbox_mAP_l | 0.8180 |

### 端到端（300 张测试图）

| 项 | 值 |
|---|---|
| 耗时 | 45 秒（0.15 s/张） |
| 检测框总数 | 492 |
| 通过 conf≥0.3 | 437（88.8%） |
| 平均实例数 | 1.46 个/图 |
| conf 中位 | 0.799（p05 0.423，max 0.932） |
| 边长中位 | 210.2 px |

---

## 三、文件清单

### 核心脚本（要保留）

| 文件 | 作用 |
|---|---|
| `config.py` | 姿态配置（HRNet-W48 + 4ch head + 7 步 pipeline） |
| `config_det.py` | 检测配置（RTMDet-tiny, 1 类, 输入 1024） |
| `run_train.py` | 姿态训练启动器（GPU 监控 + 滑动窗口 ETA + 续训） |
| `run_train_det.py` | 检测训练启动器 |
| `convert_det.py` | LabelMe → COCO 检测标注 + 尺寸统计 |
| `make_split.py` | train/val **图级**划分（seed 42, 5%） |
| `infer_e2e.py` | **端到端推理 + 导出提交文件** |
| `demo_pipeline.py` | 姿态 pipeline 演示 |
| `check_aug_geometry.py` | 增强参数几何校验 |
| `check_oob_log.py` | 出界日志分析 |

### 自定义数据集（mmpose 源码）

| 文件 | 作用 |
|---|---|
| `open-mmlab/mmpose/mmpose/datasets/datasets/DM/registry.py` | `DMKPD_dataset` |
| `open-mmlab/mmpose/mmpose/datasets/datasets/DM/transforms.py` | `CheckKeypointOutOfBounds` |
| `open-mmlab/mmpose/configs/_base_/datasets/DMKPD_meta.py` | 4 角点元信息 |

⚠️ 必须 editable 安装：`.\.venv\Scripts\python.exe -m pip install -e open-mmlab/mmpose --no-build-isolation`
⚠️ setuptools 锁 **80.9.0**（≥81 移除 `pkg_resources` 会让 mmpose 报错）

### 数据与权重

```
DMKPD_dataset/                   原始数据（9591 json = 21595 实例）
  splits/train.txt  (9111 json / 20494 实例)
  splits/val.txt    ( 480 json /  1101 实例)
  test/             (300 张 1702x1276，无标注)

det_dataset/annotations/          检测标注（COCO bbox 格式）
  instances_train.json       8710 图 / 19985 实例
  instances_val.json          480 图 /  1101 实例
  instances_holdout1702.json  400 图 /   509 实例（留出诊断用）
  instances_test.json         300 图（无标注）

modelbase_test/
  td-hm_hrnet-w48_...pth        256.7 MB  COCO 姿态预训练
  rtmdet_tiny_8xb32-...pth       54.9 MB  COCO 检测预训练

work_dirs/dmkpd/
  best_coco_AP_epoch_5.pth      245.5 MB  姿态最佳（AP 0.938）
  epoch_5.pth                   731.4 MB  含优化器状态

work_dirs/dmkpd_det/
  best_coco_bbox_mAP_epoch_1.pth 40.4 MB  检测最佳（mAP 0.784）
  epoch_1.pth                    77.8 MB  含优化器状态
  last_checkpoint                 → epoch_1.pth（下次会从这里续训！）

submission/                     300 个 txt（299 非空 / 1 空），437 实例
```

### 可视化产物（`vis_out2/`）

| 文件 | 内容 |
|---|---|
| `step_detail.png` | 样本 001850 全流程 5 联对照 |
| `heat_channels.png` | 4 通道热力图放大（峰值验证） |
| `e2e_zoom.jpg` | **端到端预测四角放大（质量最好的一张）** |
| `e2e_samples.jpg` | 端到端 9 张总览 |
| `sub_issues.jpg` | 越界/非凸实例诊断（已修复） |
| `det_test_samples.jpg` | 测试集外观（实拍现场图） |
| `train_curves.png` | 姿态训练曲线 |
| `error_dist.png` | EPE 误差分布 |
| `bad_cases.jpg` | 大误差案例 |

### 诊断脚本（`_` 开头，可删）

`_step_vis.py` `_step_detail.py` `_heat_zoom.py` `_step_split.py`
`_det_stats.py` `_det_bench.py` `_det_bench2.py` `_det_vis.py` `_det_smoke.py`
`_check_weight_det.py` `_sub_check.py` `_sub_issues.py` `_e2e_zoom.py`
`_rt_calc.py` `_explain3.py` `_area_stats.py` `_check_split.py` 等

其中**建议保留**（有复用价值）：
- `_check_weight_det.py` / `_check_weight.py` — 权重加载校验（逐张量比对）
- `_sub_check.py` — 提交格式自检
- `_e2e_zoom.py` — 预测质量可视化

---

## 四、关键设计决策（附依据）

### 姿态

| 决策 | 依据 |
|---|---|
| `padding=1.5` | 容纳旋转 + 保留背景 + 容忍 bbox 误差；实测出界率 0.04% |
| `scale=(0.9,1.1)` | 区间中心 **1.0** → 训练有效窗口 u 中心 1.5 = 测试 u 1.5，**无训测失配** |
| `rotate_factor=30, prob=1.0` | 打散「top_left = 图上最左上角」的位置捷径 |
| **禁用 RandomFlip** | DM 码无镜像形态，翻转是非物理增强 |
| `input_size=(256,256)` | 正方形使目标占画面面积最大（44.4% vs 192x256 的 33.3%） |
| `heatmap_size=(64,64)` | 量化误差 ±4px，实测峰值与 GT 最大偏差 2.45px |
| `flip_test=False` | 同上，镜像非物理 |
| `auto_scale_lr=False` | 否则 lr 被缩到 4e-6（bs=4 vs 官方 512） |

### 检测

| 决策 | 依据 |
|---|---|
| `input_size=(1024,1024)` | 两档数据在 1024 下码长 152px(512档) vs 205px(1702档)，**只差 1.35 倍** → 混合训练合理；实测显存 3.21GB |
| **删 Mosaic/MixUp** | 本来就是「大图+稀疏目标」（2.25 个/图），mosaic 会破坏尺度统计 |
| **禁用 RandomFlip** | 同姿态 |
| **禁用 RandomCrop** | 会切出「半个码」的伪标签；用户也要求不裁大图 |
| 管线与测试**同构** | `Resize(keep_ratio)+Pad+PhotoMetricDistortion` → 杜绝训测失配 |
| `norm_cfg=BN` | 官方 SyncBN 是给 DDP 的，单卡没必要 |
| `lr=0.004*(4/32)=5e-4` | bs 线性缩放 |

### 推理

| 决策 | 依据 |
|---|---|
| 用官方 `pipeline + model.test_step` | 预处理与坐标反变换全由框架负责，避免手工出错 |
| `conf = bbox_score × mean(keypoint_scores)` | 乘法符合概率语义（「是个码」且「四角都准」），与 COCO AP 分解口径一致 |
| clip → 过滤顺序 | **必须先 clip 再算尺寸**，否则 pad 边界的虚假框 clip 后退化成零高度 |
| `sanitize_quad` | 只修病理情况（越界 8 个 / 非凸 1 个），正常样本不动；修复率 0.46% |

---

## 五、踩过的坑（已修复）

### 数据

| 坑 | 现象 | 根因 |
|---|---|---|
| **polygon 是 5 点** | 第一版转换器丢掉 6449/9591 张图 | `polygon` 是 4 角 + 首尾闭合重复点；只认 4 点会丢掉全部 17896 条 |
| **多 shape ≠ 多实例** | 同上 | 必须按 `group_id` 分组 |
| 四点顺序 | 起点是 TL/TR/BR/BL 中任意一个 | 必须绕质心极角排序；`x+y`/`x-y` 极值法在 45° 摆放时退化 6.16% |
| 图片路径 | CocoDataset 静默找不到图（`len(ds)` 仍正常） | `file_name` 必须带 `DMKPD_dataset/` 前缀（config 里 `data_root='.'`） |

### 权重加载

| 坑 | 现象 | 解法 |
|---|---|---|
| `backbone.init_cfg=Pretrained` | **加载 0/1752 个键**，`strict=False` 只打警告不报错 | 用 Runner 级 `load_from` |
| `torch.load_state_dict(strict=False)` | 对**同名不同形状**仍抛 RuntimeError | 用 `mmengine.runner.load_checkpoint`（内部先过滤形状） |
| `MODELS.build()` 不跑 `init_cfg` | 权重不生效 | 显式 `model.init_weights()` |
| `Runner.from_cfg()` 不自动 `load_from` | 单独验证时误判加载失败 | 显式 `runner.load_or_resume()` |

### mmdet 配置

| 现象 | 根因 | 解法 |
|---|---|---|
| `Only one of type or by_epoch can exist in loop_cfg` | base 已有 `type='EpochBasedTrainLoop'` | 写 `type=` 就别写 `by_epoch=` |
| `AdamW got unexpected keyword 'momentum'` | **mmengine 对 dict 递归合并**，base 的 SGD `momentum=0.9` 残留 | `optim_wrapper=dict(_delete_=True, ...)` |
| `DetVisualizationHook got unexpected keyword 'enable'` | mmdet 的该 hook 无 `enable` 参数 | 不要传 |
| `AssertionError` @ `register_default_hooks` | `visualization=None` 触发 `assert hook is not None` | 整个键删掉 |
| `DetLocalVisualizer is not in the mmdet::hook registry` | `visualizer` **不是 hook** | 不能放 `default_hooks` |
| `IndexError: list index out of range` | `serialize_data=True` 时 data_list 被清空 | 传 `serialize_data=False` |
| `'RTMDet' object has no attribute 'cfg'` | `inference_detector` 要读 `model.cfg` | 构建后挂 `model.cfg = cfg` |
| `PackPoseInputs is not in the mmdet::transform registry` | `init_default_scope` 被两阶段互相覆盖 | 各自调用点显式设 scope |

### Windows

| 坑 | 解法 |
|---|---|
| `mp_start_method='fork'` 不存在 | `env_cfg` 覆盖为 `spawn` |
| spawn 多进程 | 必须 `if __name__ == '__main__':` |
| GBK 控制台 | **绝不 print emoji**（`UnicodeEncodeError`） |

### 其他

| 坑 | 解法 |
|---|---|
| ETA 虚高 3~9 倍 | 用**滑动窗口增量速度**，不用累计平均（启动开销会污染） |
| bench 速度虚高 | 必须**预热 5 iter** 再计时（cuDNN autotune） |
| Windows 任务管理器显示 GPU 低 | 它统计 3D 引擎；CUDA 走 Compute，看 `nvidia-smi` |
| `coco/AP (M): -1.0` | 该面积段无样本，正常 |

---

## 六、下一步方向（按优先级）

### P0 — 提升检测器（**已在关键路径上**）

当前 `submission/` 的检测器**只训了 1 个 epoch**（mAP 0.784）。
24 epoch 预计 mAP 可达 0.85~0.90。

```powershell
# 注意：last_checkpoint 指向 epoch_1.pth，会自动续训
& '.\.venv\Scripts\python.exe' run_train_det.py --epochs 24 --val-interval 3

# 想从 COCO 权重干净重跑：
& '.\.venv\Scripts\python.exe' run_train_det.py --epochs 24 --no-resume
```

耗时：**19.6 min/epoch × 24 ≈ 7.8 h**

训完重新导出：
```powershell
& '.\.venv\Scripts\python.exe' infer_e2e.py --export --out submission
```

### P1 — 用检测框验证姿态的真实性能（**已完成，2026-10-11**）

**结论：端到端比 GT 框条件低 0.0646 OKS，其中 74% 来自漏检。**

| 项 | 值 |
|---|---|
| GT 框条件 OKS 均值 | **0.9153**（复现了 AP 0.938 的条件） |
| 端到端 OKS 均值 | **0.8507** |
| **gap** | **-0.0646** |
| 检测召回 | **91.4%**（1006/1101 匹配成功） |
| 匹配框 IoU | 中位 **0.931**、均值 0.907、p05 0.752 |
| 误检框 | 87 个 |

**损失分解（关键）**：

```
总损失 0.0646 = 100%
  ├─ 漏检 95 个        -0.0477   73.8%   ← 主体
  └─ 框偏移            -0.0169   26.2%
```

**按尺寸分组的召回率**：

| 码边长 | 实例 | 占比 | 召回 | 匹配IoU中位 | OKS(GT框) | OKS(端到端) |
|---|---|---|---|---|---|---|
| <40px | 79 | 7.2% | **82.3%** | 0.862 | 0.8447 | 0.7314 |
| 40-60 | 213 | 19.3% | 94.8% | 0.912 | 0.9081 | 0.8525 |
| **60-120** | **458** | **41.6%** | **88.6%** | 0.940 | 0.9095 | 0.8347 |
| 120-260 | 257 | 23.3% | 94.6% | 0.940 | 0.9458 | 0.8990 |
| 260-512 | 73 | 6.6% | 94.5% | 0.940 | 0.9479 | 0.8959 |
| >512 | 21 | 1.9% | 100.0% | 0.937 | 0.8941 | 0.8830 |

**漏检集中在两处**：`<40px`（17.7%）和 **`60-120px`（11.4%，绝对数 52 个最多）**。

**灵敏度扫描**（人为扰动 GT 框，模拟"更好的检测器"）：

| 扰动 | 平均 IoU | OKS 均值 | 中位 | >=0.75 |
|---|---|---|---|---|
| 无 | 1.000 | 0.9280 | 0.9861 | 88.9% |
| ±2% 平移 | 0.924 | 0.9249 | 0.9875 | 88.1% |
| ±5% 平移 | 0.822 | 0.8982 | 0.9845 | 86.7% |
| ±10% 平移 | 0.681 | 0.8584 | 0.9806 | 82.2% |
| ±5% + 放大10% | 0.824 | 0.8917 | 0.9844 | 83.7% |
| 放大20% | 0.697 | 0.8677 | 0.9737 | 83.7% |

**IoU → OKS 分层**：

| IoU | 实例 | OKS 均值 |
|---|---|---|
| [0.00,0.60) | 6 | 0.2847 |
| [0.60,0.70) | 21 | 0.4937 |
| [0.70,0.80) | 61 | 0.7304 |
| [0.80,0.90) | 211 | 0.8945 |
| [0.90,0.95) | 428 | **0.9716** |
| [0.95,1.00) | 279 | **0.9871** |

**掉幅分析**：掉幅 >0.1 的只有 **64 个（6.4%）**，却贡献了 78% 的下降；
**79.9% 的实例掉幅 <=0.01**（几乎不受框偏移影响）。
且 **worst 20 全部是 `adv_final_*` 低可见度对抗样本**。

### ❗P1 的关键发现：姿态对框偏移的容忍度比预期高得多

| 扰动 | 平均IoU | OKS 均值 | 相对无扰动 |
|---|---|---|---|
| 无 | 1.000 | 0.9280 | — |
| ±2% | 0.924 | 0.9249 | **-0.0031** |
| ±5% | 0.822 | 0.8982 | -0.0298 |
| ±10% | 0.681 | 0.8584 | -0.0696 |
| 放大20% | 0.697 | 0.8677 | -0.0603 |

**IoU 掉到 0.82 只损失 0.030 OKS** —— 这是 `padding=1.5` 起了作用：
窗口比目标大 1.5 倍，框偏一点不会把角点裁出画面。

**对比实测检测框（IoU 中位 0.931）**：对应 OKS 约 0.97 —— 说明
**当前检测框的精度已经足够**，不是瓶颈。

### 优化优先级（P1 后的修正）

| 优先级 | 动作 | 依据 |
|---|---|---|
| **① 提升检测召回** | 1 epoch → 24 epoch | 74% 的损失来自漏检；且只训了 1 epoch，空间最大 |
| **② 补小目标** | 针对性处理 `<40px`（17.7% 漏检）和 `60-120px`（绝对数最多） | 漏检集中区 |
| **③ 框精度** | 收益上限约 +0.0513，且现 IoU 0.931 已接近极限 | 优先级低 |
| ④ 姿态本身 | AP 0.938 已高（GT 框条件），不是瓶颈 | — |

### P1 产出

| 文件 | 说明 |
|---|---|
| `_gap_check.py` | 主脚本（四阶段：匹配分析 / OKS 对比 / 灵敏度扫描 / 结论） |
| `_gap_analyze.py` | 损失分解与分组分析 |
| `_gap_vis.py` | 可视化 |
| `_gap_result.json` | 逐实例明细（250 KB） |
| `_gap_det_cache.json` | 检测结果缓存（124 KB，避免重跑检测） |
| `vis_out2/gap_analysis.jpg` | 6 面板可视化 |

运行方式：
```powershell
& '.\.venv\Scripts\python.exe' _gap_check.py        # ~3 分钟（含检测）
& '.\.venv\Scripts\python.exe' _gap_analyze.py      # 秒级
& '.\.venv\Scripts\python.exe' _gap_vis.py          # 秒级
```

---

### P2 — 姿态续训（可选）

5 epoch 已到 AP 0.938。续到 10-20 epoch 可能还有余量，
但注意 `run_train.py` 会在 80% 处降 lr（5 epoch 时是 ep4，所以 ep4 有回落）。

```powershell
& '.\.venv\Scripts\python.exe' run_train.py --epochs 12 --val-interval 2
```

### P3 — 提交前优化（可选）

| 优化 | 说明 |
|---|---|
| **TTA** | mmdet 有 `rtmdet_tta.py`（多尺度 TTA）；姿态可加 flip_test... **但 DM 码镜像非物理，不能翻转** |
| 框 NMS 调参 | 当前 `iou_threshold=0.65`；可扫 0.5/0.6/0.7 看哪个 AP 最高 |
| conf 阈值 | 当前 0.3（比赛要求）；可调高到 0.5 减少假阳（但会漏检） |
| 检测框外扩 | 把检测框按比例外扩再送姿态，可能更稳 |

### P4 — 代码整理（可选）

- 删除 `_` 开头的临时诊断脚本（保留 `_check_weight*` / `_sub_check` / `_e2e_zoom`）
- 把 `vis_out2/` 里残留的 per-sample 数据集图片（`00002.jpg` 等）清掉
- 写一个 `README.md`（本文件可改名充当）

---

## 七、复现步骤（从零开始）

```powershell
# 0. 环境（一次性）
& '.\.venv\Scripts\python.exe' -m pip install "setuptools==80.9.0"
& '.\.venv\Scripts\python.exe' -m pip install -e open-mmlab/mmpose --no-build-isolation

# 1. mmdet 配置路径（junction）
New-Item -ItemType Junction -Path mmdet -Target "$PWD\.venv\Lib\site-packages\mmdet"

# 2. 划分数据
& '.\.venv\Scripts\python.exe' make_split.py

# 3. 生成检测标注
& '.\.venv\Scripts\python.exe' convert_det.py

# 4. 训练姿态
& '.\.venv\Scripts\python.exe' run_train.py --epochs 5 --val-interval 1

# 5. 训练检测
& '.\.venv\Scripts\python.exe' run_train_det.py --epochs 24 --val-interval 3

# 6. 端到端导出
& '.\.venv\Scripts\python.exe' infer_e2e.py --export --out submission

# 7. 格式自检
& '.\.venv\Scripts\python.exe' _sub_check.py
```

---

## 八、提交规格（比赛官方）

- **每张图一个 txt**，文件名 = 图片名（无目标的图保持**空文件**）
- 每行：`DPM conf x1 y1 x2 y2 x3 y3 x4 y4`（10 个字段）
- `conf ∈ [0,1]`，**< 0.3 不参与评测**
- 坐标为**原图像素坐标**，不归一化
- 推荐顺序 TL→TR→BR→BL，**但评测脚本会 normalize 点顺序**
- 指标：`AP_50_95` / `AP_50` / `AP_75` / `AR_50_95`，OKS 用 4 个**等 sigma=0.1**
