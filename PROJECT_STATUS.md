# DMKPD 项目状态（长安大学 AI 创新大赛）

> 最后更新：**2026-10-12**（检测器 24 轮训练回传并重新评测）
> 本文件记录当前成果、文件清单、已知问题与下一步方向。

**当前完成度**：检测器 ✅（mAP 0.9280）→ 姿态 ✅（AP 0.9377）→
端到端 ✅（OKS 0.9007）→ `submission/` ✅（411 实例，300 个非空 txt）。
**流水线已可用；剩余优化空间小（端到端仅比 GT 框条件低 0.0146 OKS）。**

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

### 检测（RTMDet-tiny，**24 epoch 已训完**）

| 指标 | 1 epoch（冒烟） | **24 epoch（正式）** |
|---|---|---|
| **coco/bbox_mAP** | 0.7840 | **0.9280** |
| bbox_mAP_50 | 0.9400 | **0.9870** |
| bbox_mAP_75 | 0.8820 | **0.9690** |
| bbox_mAP_s（小目标） | 0.4770 | **0.8060** |
| bbox_mAP_m | 0.7800 | **0.9270** |
| bbox_mAP_l | 0.8180 | **0.9400** |

**逐次验证（每 3 epoch 一次）**：

| epoch | mAP | mAP50 | mAP75 | mAP_s | mAP_m | mAP_l |
|---|---|---|---|---|---|---|
| 3 | 0.8640 | 0.9730 | 0.9370 | 0.5990 | 0.8660 | 0.8850 |
| 6 | 0.9010 | 0.9840 | 0.9590 | 0.6910 | 0.9010 | 0.9200 |
| 9 | 0.9120 | 0.9850 | 0.9690 | 0.7520 | 0.9080 | 0.9290 |
| 12 | 0.9180 | 0.9870 | 0.9670 | 0.7600 | 0.9160 | 0.9320 |
| 15 | 0.9220 | 0.9870 | 0.9700 | 0.7730 | 0.9180 | 0.9370 |
| 18 | 0.9250 | 0.9880 | 0.9720 | 0.7890 | 0.9230 | 0.9410 |
| **21** | **0.9280** | 0.9870 | 0.9690 | 0.7870 | 0.9260 | 0.9430 |
| **24** | **0.9280** | 0.9870 | 0.9690 | **0.8060** | 0.9270 | 0.9400 |

> ⚠️ ep21 与 ep24 的 `bbox_mAP` 打平时四舍五入后相同，但 **`mAP_s` 在 ep24 更高
> （0.8060 vs 0.7870）**。`save_best` 用严格 `>`，故 `epoch_24.pth` 未覆盖 ep21。
> 若小码是重点，可从 `epoch_24.pth` 另取 `state_dict` 再导出。

**训练资源**（RTX 4060 Laptop 8GB，4:25:55）：GPU 利用率均值 65.1%，
温度均值 74.5℃ / 峰值 80℃（未撞 87℃ 墙），功耗均值 61.2W / 峰值 113W。
曲线图见 `vis_out2/det_curve.jpg`（`_curve_det.py` 生成）。

### 端到端（300 张测试图，**24 轮检测器**）

| 项 | 值 |
|---|---|
| 耗时 | 45 秒（0.15 s/张） |
| 检测框总数 | **417** |
| 通过 conf≥0.3 | **411（98.6%）** |
| 平均实例数 | 1.37 个/图 |
| conf 中位 | **0.864**（p05 0.530，max 0.951） |
| 边长中位 | 212.2 px |
| 空 txt 文件 | **0** |

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
  best_coco_bbox_mAP_epoch_21.pth 44.7 MB  检测最佳（mAP 0.9280）★ 当前使用
  train_log.txt                   由新机回传
  train_stdout.txt                由新机回传
  20261012_173701/scalars.json    训练曲线原始数据（`_curve_det.py` 消费）
  _old_best/                      1 epoch 冒烟产物（已归档，不再使用）
    best_coco_bbox_mAP_epoch_1.pth 40.4 MB（mAP 0.784）
    epoch_1.pth                    77.8 MB
    last_checkpoint.stale

submission/                     300 个 txt（全部非空），411 实例，98.6% 保留率
```

### 可视化产物（`vis_out2/`）

| 文件 | 内容 |
|---|---|
| `step_detail.png` | 样本 001850 全流程 5 联对照 |
| `heat_channels.png` | 4 通道热力图放大（峰值验证） |
| `e2e_zoom.jpg` | **端到端预测四角放大（质量最好的一张）** |
| `e2e_samples.jpg` | 端到端 9 张总览 |
| `det_curve.jpg` | **24 轮检测器训练曲线（loss / lr / mAP 三面板）** |
| `gap_analysis.jpg` | P1 gap 六面板（**24 轮检测器版**） |
| `gap_analysis_1epoch.jpg` | 同上，1 轮检测器版（留档对比） |
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

### P0 — 提升检测器（**✅ 已完成 2026-10-12**）

**结果：24 epoch，coco/bbox_mAP 0.9280**（预测值 0.85~0.90，实得更高）。

```powershell
# 已在新机（RTX 4060 Laptop）执行完毕：
#   & '.\.venv\Scripts\python.exe' run_train_det.py --epochs 24 --val-interval 3
# 耗时 4:25:55；权重回传后已 hash 校验一致：
#   DF995973A279BFAFD4EBD5E65BCA2C17C0540798B7D9449B5F2845ECEC6353B4
```

已重新导出：
```powershell
& '.\.venv\Scripts\python.exe' infer_e2e.py --export --out submission
# → 417 框 / 411 保留（98.6%）/ 300 个非空 txt
```

**结论：检测器不再是瓶颈**（端到端 OKS 仅比 GT 框条件低 0.0146），
且 21→24 轮 mAP 已持平，继续增加轮数无收益。
下一步重心转向 **P2 姿态续训** 与 `<40px` 小码专项。

### P1 — 用检测框验证姿态的真实性能（**已完成，2026-10-12 用 24 轮检测器重跑**）

**结论：端到端只比 GT 框条件低 0.0146 OKS，检测器已不再是瓶颈。**

> 本节数据全部来自 **24 轮训练** 的检测器（`best_coco_bbox_mAP_epoch_21.pth`，
> coco/bbox_mAP **0.9280**）。对比 1 轮检测器的旧数据见下方「改善对照」。

| 项 | 值 |
|---|---|
| GT 框条件 OKS 均值 | **0.9153**（复现了姿态 AP 0.938 的条件） |
| 端到端 OKS 均值 | **0.9007** |
| **gap** | **-0.0146** |
| 检测召回 | **98.1%**（1080/1101 匹配成功） |
| 匹配框 IoU | 中位 **0.969**、均值 0.951、p05 0.856 |
| 误检框 | 23 个 |

**损失分解（关键）**：

```
总损失 0.0146 = 100%
  ├─ 漏检 21 个        -0.0067   46.0%
  └─ 框偏移            -0.0079   54.0%   ← 已与漏检相当
```

**按尺寸分组的召回率**：

| 码边长 | 实例 | 占比 | 召回 | 匹配IoU中位 | OKS(GT框) | OKS(端到端) |
|---|---|---|---|---|---|---|
| <40px | 79 | 7.2% | **93.7%** | 0.938 | 0.8447 | 0.8025 |
| 40-60 | 213 | 19.3% | **99.5%** | 0.959 | 0.9081 | 0.8904 |
| **60-120** | **458** | **41.6%** | **97.4%** | 0.973 | 0.9095 | 0.8997 |
| 120-260 | 257 | 23.3% | **99.6%** | 0.973 | 0.9458 | 0.9362 |
| 260-512 | 73 | 6.6% | 97.3% | 0.966 | 0.9479 | 0.9217 |
| >512 | 21 | 1.9% | 100.0% | 0.972 | 0.8941 | 0.8901 |

**小码（`<40px`）是唯一仍明显偏弱的区间**（召回 93.7%，端到端 OKS 0.8025）；
其余五个区间的召回全部 ≥97%、端到端 OKS 与 GT 框条件差距 ≤0.026。

**1 轮 → 24 轮 改善对照**：

| 指标 | 1 轮检测器 | **24 轮检测器** | 变化 |
|---|---|---|---|
| 检测召回 | 91.4% | **98.1%** | **+6.7%** |
| 漏检 | 95 | **21** | -74 |
| 误检框 | 87 | **23** | -64 |
| 匹配 IoU 中位 | 0.931 | **0.969** | +0.038 |
| 端到端 OKS | 0.8507 | **0.9007** | **+0.050** |
| gap vs GT 框 | -0.0646 | **-0.0146** | **缩小 4.4×** |
| gap 来源 | 74% 漏检 / 26% 偏移 | **46% 漏检 / 54% 偏移** | 瓶颈转移 |
| `<40px` 召回 | 82.3% | **93.7%** | +11.4% |
| `60-120px` 召回 | 88.6% | **97.4%** | +8.8% |

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

**对比实测检测框（IoU 中位 0.969）**：对应 OKS 约 0.98 —— 说明
**当前检测框的精度已经足够**，不是瓶颈。

### 优化优先级（24 轮检测器后的修正）

| 优先级 | 动作 | 依据 |
|---|---|---|
| **① 补小目标** | 只针对 `<40px`（召回 93.7%、端到端 OKS 0.8025） | 唯一明显偏弱的区间；其余区间已 ≥97% |
| **② 框精度** | 收益上限约 +0.0146，且现 IoU 0.969 已接近极限 | 收益小 |
| ③ 检测召回 | 98.1%，仅剩 21 个漏检 | 空间已很小 |
| ④ 姿态本身 | AP 0.938（GT 框条件），不是瓶颈 | — |
| ⑤ 提高检测轮数 | 21→24 轮 mAP 已持平（0.9280） | 已收敛，继续训无收益 |

### P1 产出

| 文件 | 说明 |
|---|---|
| `_gap_check.py` | 主脚本（四阶段：匹配分析 / OKS 对比 / 灵敏度扫描 / 结论） |
| `_gap_analyze.py` | 损失分解与分组分析 |
| `_gap_vis.py` | 可视化 |
| `_gap_result.json` | 逐实例明细（250 KB） |
| `_gap_det_cache.json` | 检测结果缓存（124 KB，避免重跑检测） |
| `_gap_summary.json` | 汇总摘要（召回 / OKS / gap） |
| `vis_out2/gap_analysis.jpg` | 6 面板可视化（24 轮版） |
| `_curve_det.py` | 画 24 轮训练曲线 -> `vis_out2/det_curve.jpg` |
| `_sub_check.py` | 提交文件自检（区分「非凸众修」与「退化无解」） |

运行方式：
```powershell
& '.\\.venv\\Scripts\\python.exe' _gap_check.py        # ~3 分钟（含检测）
& '.\\.venv\\Scripts\\python.exe' _gap_analyze.py      # 秒级
& '.\\.venv\\Scripts\\python.exe' _gap_vis.py          # 秒级
& '.\\.venv\\Scripts\\python.exe' _sub_check.py        # 秒级（提交前必跑）
& '.\\.venv\\Scripts\\python.exe' _curve_det.py        # 秒级
```

#### `_sub_check.py` 的两类「非凸」

24 轮检测器导出后，只剩 **`004353.txt` 的 1 个非凸四边形**。
它属于 **退化**（两个预测角点相距 **3.72px**），而非顺序错误：
点集实际不足 4 个不同位置，**任何排列都不可能构成凸四边形**（数学无解）。

| 类型 | 判定 | 意义 |
|---|---|---|
| **退化** | 最短边长 < 5px | 姿态模型真实失败，OKS 本来就很低，**与丢弃等价**，不影响评分 |
| **可修** | 最短边 ≥ 5px 但仍非凸 | `sanitize_quad` 枚举 24 种排列本应修好；若报数 >0 说明修复逻辑有 bug |

当前自检输出：`非凸且非退化（应修复） 0` / `非凸但退化（角点重合） 1`✅

---

### P2 — 姿态续训（**当前最高优先级**）

5 epoch 已到 AP 0.938（GT 框条件 AP 0.9377 / AP.75 0.9485 /
PCK@0.10 0.9353 / EPE 5.41px）。

⚠️ 姿态已是检测→姿态全链条的瓶颈之后的下一个可攻点：
端到端 gap 已只剩 0.0146，而 **GT 框条件下的 OKS 也只有 0.9153** ——
说明剩余提升空间主要在姿态本身，而不是检测器。

建议先复制权重备份，再续训：
```powershell
# 备份当前 5 轮最佳（重要！续训会覆盖）
Copy-Item 'work_dirs\dmkpd\best_coco_AP_epoch_5.pth' `
          'work_dirs\dmkpd\best_coco_AP_epoch_5.bak.pth'

# 续训（注意 run_train.py 会在 80% 处降 lr）
& '.\\.venv\\Scripts\\python.exe' run_train.py --epochs 12 --val-interval 2
```

耗时参考：GTX 1650 约 8 min/epoch → 12 轮约 1.6 h；
若搬到 4060 会快很多（可两机并行：4060 训姿态，1650 做其他验证）。

### P3 — 提交前优化（可选）

| 优化 | 说明 |
|---|---|
| **小码专项（`<40px`）** | 唯一明显偏弱区间：召回 93.7%、端到端 OKS 0.8025。可用多尺度测试或降低检测 `score_thr` |
| **试 epoch_24 权重** | `mAP_s` 0.8060 > ep21 的 0.7870；若小码重要可另存导出对比 |
| **TTA** | ⚠️ DM 码镜像非物理，**不能翻转**；多尺度需验证 |
| 框 NMS 调参 | 当前 `iou_threshold=0.65`；可扫 0.5/0.6/0.7 看哪个 AP 最高 |
| conf 阈值 | 当前 0.3（比赛要求）；可调高到 0.5 减少假阳（但会漏检） |
| 检测框外扩 | 把检测框按比例外扩再送姿态 —— 但 P1 已证明 `padding=1.5` 已很容错，预期收益小 |

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
