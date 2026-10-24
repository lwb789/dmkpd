# DMKPD — Data Matrix 码关键点检测

2026 VLP 挑战赛参赛作品

---

## 一、成绩

| 评测 | AP_50_95 | AP_50 | AP_75 | AR |
|---|---|---|---|---|
| **线上榜单（最终提交）** | **97.6694** | 97.7970 | 97.7970 | **98.3544** |

---

## 二、方案

两阶段：**检测 → 姿态**。

```
测试图 1702×1276（含 1~4 个 DM 码，无标注）
   │
   ├─[1] RTMDet-tiny 检测器
   │     Resize(keep_ratio) 长边→1024 + Pad(114) 到 1024×1024
   │     → 每图 N 个 bbox + bbox_score
   │
   ├─[2] HRNet-W48 + HeatmapHead(4ch) 姿态
   │     逐 bbox 做 GetBBoxCenterScale(padding=1.5) → TopdownAffine(256)
   │     → 4 通道 64×64 热图 → MSRA 解码（argmax + 0.25 亚像素）
   │     → 仿射反变换回原图坐标，得到 4 个角点
   │
   └─[3] 导出：每图一个 txt
         行格式 DPM conf x1 y1 x2 y2 x3 y3 x4 y4
         conf = bbox_score × mean(4 个 kp_score)，仅写出 conf ≥ 0.3
```

**为什么必须两阶段**：mmpose 的 topdown heatmap 模型不具备目标定位能力，输入必须是
围绕单个目标的裁剪图；而测试集无标注，所以需要检测器先给出 bbox。

---

## 三、已知局限

- 指标已接近饱和：`AP_75 ≈ AP_50`，说明被检出的实例已定位到亚像素精度，
  剩余差距几乎全部来自少数未检出实例。

---

## 四、目录

| 路径 | 说明 |
|---|---|
| `config.py` / `config_det.py` | 姿态 / 检测配置 |
| `run_train.py` / `run_train_det.py` | 训练入口 |
| `make_split.py` / `convert_det.py` | 数据划分 / 生成检测标注 |
| `infer_e2e.py` | 端到端推理，产出 `submission/` |
| `export_onnx.py` | 导出 ONNX（检测 + 姿态） |
| `_ap_holdout.py` | 核心评测模块（COCO AP + 点序归一化 OKS） |
| `_ar_analyze.py` / `_gap_check.py` / `_order_test.py` | AR 归因 / 框偏移影响 / 点序验证 |
| `_preflight_check.py` / `_sub_check.py` | 训练前预检 / 提交格式自检 |
| `DMKPD_dataset/splits/*.txt` | 数据划分清单（seed=42），复现关键 |
| `det_dataset/annotations/*.json` | 检测标注（COCO 格式） |
| `patches/0001-dmkpd-dataset.patch` | 对 mmpose 的 5 处自定义改动 |

---

## 五、提交规格

- 每张测试图一个 `.txt`，文件名与图同名
- 每行 `DPM conf x1 y1 x2 y2 x3 y3 x4 y4`，坐标为原图像素
- `conf < 0.3` 的行不计入评测
- 评测方会 normalize 点顺序

产物：`submission/` 300 个 txt、412 行，格式自检全部通过。

---

## 六、引用

RTMDet · HRNet · OpenMMLab（mmdetection / mmpose / mmcv / mmengine）

------

## 七、其它链接

参赛及数据下载：http://vlp.chd.edu.cn/tasks/5

模型权重下载：https://pan.baidu.com/s/1m_yNYYP4m4DuLJWdJo_FwQ?pwd=rtix 提取码: rtix 

