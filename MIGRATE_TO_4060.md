# 迁移到 RTX 4060 笔记本 —— 无损迁移方案

> 目标：本机（GTX 1650）做测试/调参/优化/上报，**RTX 4060 笔记本只跑吃性能的训练**。
> 本文档是实测调研结果 + 可执行步骤。

---

## 一、实测调研结果（本机环境盘点）

### 1.1 环境版本

```
Python        3.10.1  (AMD64)
torch         2.1.0+cu118
CUDA runtime  11.8      cudnn 8700
compile arch  sm_37 sm_50 sm_60 sm_61 sm_70 sm_75 sm_80 sm_86 sm_90
本机 GPU      NVIDIA GeForce GTX 1650  (sm_75, 4 GB)
```

### 1.2 ❗核心发现：torch 缺 sm_89

RTX 4060 是 **Ada Lovelace，计算能力 sm_89**，而本地 torch 2.1.0+cu118
的编译列表里**没有 sm_89**（最高到 sm_86）。

**能不能跑？能。** CUDA 的 binary compatibility 规则：
> 同一 major 版本内，低 minor 的 cubin 可以在高 minor 的 GPU 上运行。

sm_86 与 sm_89 同属 major=8，89 ≥ 86 → **可以运行**。

**代价：不是最优性能。** 没有针对 sm_89 编译的 kernel，
少了 Ada 架构的专属优化（FP8 之类本就用不上，但调度/指令选择会有差异）。

**结论**：
- 如果只是**把训练跑完** → 直接搬，能跑，正确性无问题
- 如果**要榨满 4060 性能** → 建议在新机重装 torch 2.4+（带 cu121/cu124，有 sm_89）

### 1.3 ❗第二个发现：editable 安装含硬编码路径

```
.venv/Lib/site-packages/__editable___mmpose_1_3_2_finder.py

MAPPING = {'mmpose': 'D:\\Personal\\长安大学\\人工智能创新大赛\\DMKPD\\open-mmlab\\mmpose\\mmpose'}
                                 ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
                                 绝对路径，换机/换目录后就失效
```

这是**最典型也最致命**的迁移坑：`.venv` 能靠近距离搬家，
但只要有**一个**硬编码路径，就白搭。

### 1.4 其他机器相关项

| 项 | 状态 | 说明 |
|---|---|---|
| `mmdet` junction | ⚠️ **链接**，`Target` 指向 `.venv\Lib\site-packages\mmdet` | 换机必须重建 |
| mmcv `_ext.cp310-win_amd64.pyd` | ⚠️ 编译扩展 | 与 Python 版本 + 平台**强绑定**（但同为 Windows x64 + cp310 即可） |
| 脚本里的绝对路径 | ✅ **无** | 所有脚本都用相对路径，很好 |
| `.venv/pyvenv.cfg` 的 `home` | ⚠️ 指向 `D:\Softwares\Python\Python 3.10.1(64-bit)` | 换机需同版本 Python |
| `.git` | ❌ 不存在 | 无法用 git 同步，要用文件拷贝 |
| `python.exe` | 242 KB 真实文件（非 symlink） | ✅ 可直接搬 |

### 1.5 体积盘点

| 目录 | 体积 | 文件数 | 迁移策略 |
|---|---|---|---|
| `DMKPD_dataset` | **1.09 GB** | 19484 | **必须搬**（原数据） |
| `modelbase_test` | 0.30 GB | 3 | **必须搬**（预训练权重） |
| `open-mmlab` | 0.03 GB | 2289 | **必须搬**（mmpose 源码，editable） |
| `det_dataset` | 0.01 GB | 4 | **必须搬**（检测标注 json） |
| 根目录 `*.py` | < 0.01 GB | ~40 | **必须搬**（脚本） |
| `submission` | 0.00 GB | 300 | 可选 |
| `vis_out2` | 0.02 GB | 77 | 可选 |
| `work_dirs` | 2.50 GB | 100 | ❗**挑着搬**（见下） |
| `.venv` | **5.58 GB** | 39247 | ❌ **不搬**，新机重建 |
| `DMKPD_dataset/splits` | — | 2 | **必须搬**（划分清单，保证可复现） |

#### `work_dirs` 挑搬策略

里面大部分是**调试垃圾**：

```
731.4 MB  work_dirs/dmkpd/epoch_5.pth              ← 姿态，含优化器状态
729.4 MB  work_dirs/_smoke/epoch_2.pth             ← ❌ 冒烟测试残留，删
729.4 MB  work_dirs/_hooktest/epoch_1.pth          ← ❌ 调试残留，删
245.5 MB  work_dirs/dmkpd/best_coco_AP_epoch_5.pth ← ✅ 要搬（姿态最佳）
 77.8 MB  work_dirs/dmkpd_det/epoch_1.pth          ← 检测，含优化器状态
 40.4 MB  work_dirs/dmkpd_det/best_coco_bbox_mAP_epoch_1.pth ← ✅ 要搬
```

**最小必需**：只搬 `best_*.pth`（285 MB）—— 用于本机推理/提交。
**若要续训**：额外搬 `epoch_*.pth`（含优化器状态，续训可无缝衔接）。

---

## 二、推荐方案：**不要搬 .venv，在新机重建**

### 为什么

`.venv` 5.58 GB，其中约 95% 是 `torch` 的大体积 wheel 解包结果。
而**重建的代价只有一次 `pip install`**（约 10 分钟），却换来：

| 重建的收益 | 说明 |
|---|---|
| 拿到 sm_89 优化 | 可选装 torch 2.4+cu121，4060 上更快 |
| 无硬编码路径 | `__editable___mmpose_finder.py` 自动写入**新机**的正确路径 |
| 无 junction 问题 | 新机重新 `New-Item -ItemType Junction` |
| 环境自洽 | 不会残留本机 `home = D:\Softwares\Python\...` |
| 省带宽 | 只需传 1.5 GB 数据，而不是 7 GB 全部 |

**反过来说**：搬 `.venv` 反而要在新机做 4 处修补（改 finder、建 junction、
改 pyvenv.cfg、期望 Python 装在同版本同路径），**修错一处就报错**，
正是你想避免的"各种报错耽误训练"。

---

## 三、执行步骤

### 阶段 A：本机准备（在新机上跑之前做完）

#### A1. 清理调试垃圾（省 1.5 GB）

```powershell
Remove-Item 'work_dirs\_smoke'   -Recurse -Force -EA SilentlyContinue
Remove-Item 'work_dirs\_smoke2'  -Recurse -Force -EA SilentlyContinue
Remove-Item 'work_dirs\_hooktest'-Recurse -Force -EA SilentlyContinue
Remove-Item 'work_dirs\_temp_test'-Recurse -Force -EA SilentlyContinue
```

#### A2. 生成 requirements（钉死版本）

```powershell
& '.\.venv\Scripts\python.exe' -m pip freeze > requirements-lock.txt
```

⚠️ **必须手工修正 3 行**（否则新机装不上/装错）：

```
# 删掉这几行（editable 安装产物会被 freeze 成奇怪形式）
-e git+...            # 或 mmpose @ file:///...
mmpose==1.3.2         # 改成下面这样，标明要从本地源码装

# torch 三件套改成指定 index 的写法（保证 cu118 一致）
torch==2.1.0+cu118
torchvision==0.16.0+cu118
torchaudio==2.1.0+cu118
```

#### A3. 打迁移包

```powershell
# 只打包必要内容（约 1.5 GB）
$need = @(
  'DMKPD_dataset', 'modelbase_test', 'det_dataset', 'open-mmlab',
  'submission', 'config.py', 'config_det.py', 'run_train.py',
  'run_train_det.py', 'convert_det.py', 'make_split.py',
  'infer_e2e.py', 'PROJECT_STATUS.md', 'requirements-lock.txt',
  'MIGRATE_TO_4060.md', 'vis_out2'
)
# work_dirs 只挑 best 权重
New-Item -ItemType Directory -Force 'work_dirs\_migrate' | Out-Null
Copy-Item 'work_dirs\dmkpd\best_coco_AP_epoch_5.pth'          'work_dirs\_migrate\'
Copy-Item 'work_dirs\dmkpd_det\best_coco_bbox_mAP_epoch_1.pth' 'work_dirs\_migrate\'

Compress-Archive -Path $need -DestinationPath 'DMKPD_migrate.zip' -CompressionLevel Optimal
```

> 若要**续训**，额外把 `work_dirs\dmkpd\epoch_5.pth` 和
> `work_dirs\dmkpd_det\epoch_1.pth` 也放进 `_migrate`（多 800 MB）。

---

### 阶段 B：新机搭建（RTX 4060 笔记本）

#### B1. 装 Python 3.10（版本必须一致）

```powershell
# 方式一：winget
winget install Python.Python.3.10

# 方式二：从 python.org 下 3.10.x 安装包
# ⚠️ 版本必须 3.10.x —— mmcv 的 _ext.cp310-win_amd64.pyd 是 cp310 编译的
#    如果新机装 3.11/3.12，mmcv 的预编译扩展就不能用
```

验证：
```powershell
python --version    # 应输出 Python 3.10.x
```

#### B2. 解压项目

建议**放在相似短的路径**，避免中文+空格带来的潜在问题：

```powershell
# 例如
D:\DMKPD\
```

⚠️ **路径中含中文不是问题**（本机就是 `长安大学/人工智能创新大赛`，
全程正常），但**路径越短越省心**（Windows 路径长度限制 260 字符，
`site-packages` 深层嵌套容易触顶）。

#### B3. 建虚拟环境

```powershell
cd D:\DMKPD
python -m venv .venv
```

#### B4. 装 PyTorch（❗这里做选择）

**选项 1 —— 与训练机严格一致（推荐先跑通）**

```powershell
& '.\.venv\Scripts\python.exe' -m pip install `
  torch==2.1.0+cu118 torchvision==0.16.0+cu118 torchaudio==2.1.0+cu118 `
  --index-url https://download.pytorch.org/whl/cu118
```

优点：与已验证的环境**完全一致**，不会引入新变量。
缺点：无 sm_89 优化（但能跑，正确性无问题）。

**选项 2 —— 换带 sm_89 的 torch（要更快时再试）**

先查有哪些版本带 sm_89：
```powershell
& '.\.venv\Scripts\python.exe' -c "
import torch
print(torch.__version__, torch.cuda.get_arch_list())
"
```
然后用 cu121/cu124：
```powershell
& '.\.venv\Scripts\python.exe' -m pip install `
  torch==2.4.1 torchvision==0.19.1 --index-url https://download.pytorch.org/whl/cu121
```

⚠️ 换 torch 版本要**同步验证 mmcv 兼容**（mmcv 2.1.0 官方支持 torch 1.8~2.1，
2.2+ 需要 mmcv >= 2.2.0）。所以选项 2 的代价是**连 mmcv 一起升**。
**建议先用选项 1 跑通，再考虑升级。**

#### B5. 装其余依赖

```powershell
& '.\.venv\Scripts\python.exe' -m pip install -r requirements-lock.txt
```

❗**关键**：`setuptools` 必须锁在 **80.9.0**
（≥81 移除 `pkg_resources`，mmpose 的 `get_installed_path` 会报
`ModuleNotFoundError`）

```powershell
& '.\.venv\Scripts\python.exe' -m pip install "setuptools==80.9.0"
```

#### B6. editable 安装 mmpose（❗最关键一步）

```powershell
& '.\.venv\Scripts\python.exe' -m pip install -e open-mmlab/mmpose --no-build-isolation
```

这一步会**在新机重新生成** `__editable___mmpose_1_3_2_finder.py`，
把正确的**新机路径**写进 `MAPPING` —— 这就是"不搬 .venv"的核心收益。

验证：
```powershell
& '.\.venv\Scripts\python.exe' -c "
import mmpose
from mmpose.datasets.datasets.DM.registry import DMKPD_dataset
print('mmpose from:', mmpose.__file__)
print('DMKPD_dataset OK:', DMKPD_dataset)
"
```
若输出指向 `D:\DMKPD\open-mmlab\mmpose\...` 而非旧路径 → 成功。

#### B7. 重建 mmdet junction

```powershell
New-Item -ItemType Junction -Path mmdet `
  -Target "$PWD\.venv\Lib\site-packages\mmdet"
```

验证：
```powershell
Test-Path 'mmdet\.mim\configs\rtmdet\rtmdet_tiny_8xb32-300e_coco.py'  # 应 True
```

---

### 阶段 C：环境自检（❗跑训练前必做）

写一个 `_env_check.py`（放在项目根，直接跑）：

```python
# -*- coding: utf-8 -*-
"""迁移后环境自检：把"会不会报错"提前暴露。"""
import os, sys, platform
ok = True

def chk(name, cond, detail=''):
    global ok
    ok &= bool(cond)
    print(f'[{"OK " if cond else "FAIL"}] {name} {detail}')

print('=' * 76)
print('1. 基础')
print('=' * 76)
chk('Python 3.10.x', sys.version.startswith('3.10'), sys.version.split()[0])
chk('Windows x64', platform.machine() == 'AMD64', platform.machine())

import torch
chk('torch 可导入', True, torch.__version__)
chk('CUDA 可用', torch.cuda.is_available(),
    torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NO CUDA')
if torch.cuda.is_available():
    sm = torch.cuda.get_device_capability(0)
    arch = torch.cuda.get_arch_list()
    chk(f'GPU sm_{sm[0]}{sm[1]} 在编译列表', f'sm_{sm[0]}{sm[1]}' in arch,
        f'arch_list={arch}')

print()
print('=' * 76)
print('2. 框架')
print('=' * 76)
import mmcv, mmengine, mmdet
chk('mmcv', mmcv.__version__ == '2.1.0', mmcv.__version__)
chk('mmengine', mmengine.__version__ == '0.10.7', mmengine.__version__)
chk('mmdet', mmdet.__version__ == '3.2.0', mmdet.__version__)

try:
    import mmcv.ops          # 编译扩展
    chk('mmcv.ops（编译扩展）可导入', True)
except Exception as e:
    chk('mmcv.ops（编译扩展）可导入', False, str(e)[:80])

import mmpose
chk('mmpose', mmpose.__version__ == '1.3.2', mmpose.__version__)
chk('mmpose 指向本项目源码',
    'open-mmlab' in mmpose.__file__.replace('\\', '/'),
    mmpose.__file__)

import pkg_resources
chk('pkg_resources 存在（setuptools<81）', True,
    pkg_resources.__name__)

print()
print('=' * 76)
print('3. 自定义模块')
print('=' * 76)
try:
    from mmpose.datasets.datasets.DM.registry import DMKPD_dataset
    from mmpose.datasets.datasets.DM.transforms import CheckKeypointOutOfBounds
    chk('DMKPD_dataset 可导入', True)
    chk('CheckKeypointOutOfBounds 可导入', True)
except Exception as e:
    chk('DMKPD_dataset / transforms', False, str(e)[:100])

print()
print('=' * 76)
print('4. 数据与配置')
print('=' * 76)
for p in ['DMKPD_dataset/splits/train.txt', 'DMKPD_dataset/splits/val.txt',
          'det_dataset/annotations/instances_train.json',
          'det_dataset/annotations/instances_val.json',
          'modelbase_test/td-hm_hrnet-w48_8xb32-210e_coco-256x192-0e67c616_20220913.pth',
          'modelbase_test/rtmdet_tiny_8xb32-300e_coco_20220902_112414-78e30dcc.pth',
          'open-mmlab/mmpose/configs/_base_/datasets/DMKPD_meta.py']:
    chk(p.split('/')[-1], os.path.isfile(p))

chk('mmdet junction 可用',
    os.path.isfile('mmdet/.mim/configs/rtmdet/rtmdet_tiny_8xb32-300e_coco.py'))

print()
print('=' * 76)
print('5. 配置可加载 + 模型可构建')
print('=' * 76)
from mmengine.config import Config
from mmengine.registry import init_default_scope

try:
    cfg = Config.fromfile('config.py')
    init_default_scope('mmpose')
    from mmpose.registry import MODELS as PM
    m = PM.build(cfg.model)
    n = sum(p.numel() for p in m.parameters()) / 1e6
    chk('姿态模型可构建', abs(n - 63.60) < 0.5, f'{n:.2f}M 参数')
except Exception as e:
    chk('姿态模型可构建', False, str(e)[:120])

try:
    cfgd = Config.fromfile('config_det.py')
    init_default_scope('mmdet')
    from mmdet.registry import MODELS as DM
    md = DM.build(cfgd.model)
    n = sum(p.numel() for p in md.parameters()) / 1e6
    chk('检测模型可构建', abs(n - 4.87) < 0.2, f'{n:.2f}M 参数')
except Exception as e:
    chk('检测模型可构建', False, str(e)[:120])

print()
print('=' * 76)
print('6. 数据管线前向（真正读图 + pack）')
print('=' * 76)
try:
    from mmengine.dataset import Compose
    from mmdet.registry import DATASETS as DD
    init_default_scope('mmdet')
    ds = DD.build(dict(type='CocoDataset', metainfo=dict(classes=('DPM',)),
                       data_root='.', data_prefix=dict(img=''),
                       ann_file='det_dataset/annotations/instances_val.json',
                       pipeline=cfgd.test_pipeline, test_mode=True,
                       serialize_data=False))
    r = Compose(cfgd.test_pipeline)(ds.get_data_info(0))
    chk('检测管线前向', tuple(r['inputs'].shape) == (3, 1024, 1024),
        f'shape={tuple(r["inputs"].shape)}')
except Exception as e:
    chk('检测管线前向', False, str(e)[:120])

print()
print('=' * 76)
print(('全部通过 —— 可以开始训练' if ok else
       '有失败项 —— 先修好再训练！'))
print('=' * 76)
sys.exit(0 if ok else 1)
```

跑：
```powershell
& '.\.venv\Scripts\python.exe' _env_check.py
```

**这一步是「避免新机报错」的关键** —— 它把 7 类常见坑（Python 版本 /
mmcv 扩展 / editable 路径 / junction / pkg_resources / 配置 / 管线）
全部提前暴露，**10 秒内**给出结论，而不是等训练跑到第 100 iter 才崩。

---

### 阶段 D：性能验证（新机跑通后）

#### D1. 实测 4060 的显存与速度

```powershell
& '.\.venv\Scripts\python.exe' _det_bench2.py     # 检测（已有脚本）
```

姿态的显存基准（本机 4 GB 卡的数据，用于对比）：

```
256x256  bs=4   1.38 GB      本机 0.41 s/it
```

4060 有 **8 GB 显存**（比 1650 翻倍）→ **可以放大 batch size**：

| 位置 | 本机(1650 4GB) | 4060 8GB 可试 |
|---|---|---|
| 姿态 bs | 4 | **8 / 12 / 16** |
| 检测 bs @1024 | 3.21 GB | **6 / 8** |

batch size 翻倍后，**lr 要按比例缩放**：
`lr_new = lr_old × (bs_new / bs_old)`

#### D2. 注意 4060 笔记本的两个坑

| 坑 | 说明 |
|---|---|
| **功耗墙** | 笔记本 GPU 有 TGP 限制（多为 105-140W，而非台式 4060 的 115W+），且**必须插电**才满血；用电池会降到很低 |
| **散热降频** | 笔记本散热受限，长时间满载易触发温度墙。本机实测满载 82-86°C 不降频；笔记本建议监控 `clocks.sm`，若持续低于标称值就是降频了 |

`run_train_det.py` 里的 `GpuMonitor` 已经会记录
`util_gpu / temp / power / clock_sm`，**直接用它的输出判断**。

---

## 四、日常协作流程

迁移完成后，推荐的**职责划分**：

```
┌─────────────────────────┐        ┌──────────────────────────┐
│  本机（GTX 1650 4GB）    │        │  RTX 4060 笔记本          │
│                          │        │                           │
│  · 数据转换/统计          │        │  · 只跑训练               │
│  · 推理 / 端到端验证      │        │    run_train.py           │
│  · 导出 submission       │        │    run_train_det.py       │
│  · 调参（小规模快实验）    │        │                           │
│  · 可视化 / 报错排查      │        │                           │
└───────────┬─────────────┘        └────────────┬─────────────┘
            │                                    │
            │  ① 改配置/脚本，上传到新机          │
            └───────────────────────────────────>│
                                                 │
            ┌────────────────────────────────────┘
            │  ② 训练完，只回传 best_*.pth
            ↓   （40 ~ 250 MB，不是 700 MB 的 epoch_*.pth）
      本机推理 + 导出 submission
```

### 关键纪律

| 规则 | 原因 |
|---|---|
| **代码用文件同步，不用 git**（本项目无 .git） | 或先 `git init` 再推，更稳 |
| **只回传 `best_*.pth`** | `epoch_*.pth` 含优化器状态，体积 2-3 倍；本机不续训就不需要 |
| **改配置后同步给两边** | 否则两边跑的不是同一个实验，结果不可比 |
| **每次回传记录 commit/hash** | 便于回溯"这个 AP 是哪份权重跑的" |
| **本机保留 `submission/` 历史** | 每次导出存一份带时间戳的副本，便于对比 |

### 建议：给本机也装 git

```powershell
git init
# 建 .gitignore
@'
.venv/
work_dirs/
vis_out2/
submission/
modelbase_test/
DMKPD_dataset/
det_dataset/
__pycache__/
*.pth
*.zip
'@ | Out-File -Encoding utf8 .gitignore
git add -A
git commit -m "init: DMKPD 姿态+检测全流程"
```

这样**只提交代码**（< 1 MB），新机 `git clone` 即可，
改配置也能用 `git diff` 看清改了什么。

---

## 五、checklist（照抄执行）

### 本机

- [ ] 删 `work_dirs\_smoke` / `_smoke2` / `_hooktest` / `_temp_test`
- [ ] `pip freeze > requirements-lock.txt`，手工修正 3 处
- [ ] 拷 `best_*.pth` 到 `work_dirs\_migrate`
- [ ] 打 `DMKPD_migrate.zip`（约 1.5 GB）
- [ ] （可选）`git init` + 提交代码

### 新机

- [ ] 装 **Python 3.10.x**（不能是 3.11/3.12）
- [ ] 解压到**短路径**（如 `D:\DMKPD`）
- [ ] `python -m venv .venv`
- [ ] 装 torch 2.1.0+cu118 三件套（是否升级见阶段 B4）
- [ ] `pip install -r requirements-lock.txt`
- [ ] **`pip install "setuptools==80.9.0"`** ← 别忘
- [ ] **`pip install -e open-mmlab/mmpose --no-build-isolation`** ← 最关键
- [ ] 重建 `mmdet` junction
- [ ] 跑 `_env_check.py`，**必须全 OK**
- [ ] 插电，跑 `_det_bench2.py` 测显存/速度
- [ ] 启动训练

---

## 六、常见报错对照表（新机上遇到时查这里）

| 报错 | 根因 | 修法 |
|---|---|---|
| `ModuleNotFoundError: No module named 'pkg_resources'` | setuptools ≥ 81 | `pip install "setuptools==80.9.0"` |
| `No module named 'mmpose.datasets.datasets.DM'` | editable 未装 / 路径错 | 重跑 `pip install -e open-mmlab/mmpose --no-build-isolation` |
| `mmpose.__file__` 指向旧机器路径 | 搬了旧 `.venv` | 删 `.venv` 重建（**本文核心建议**） |
| `ImportError: DLL load failed` (mmcv) | mmcv 的 `.pyd` 与 Python 版本不符 | 确认 Python 是 **3.10.x** |
| `RuntimeError: Only one of type or by_epoch` | 见 `PROJECT_STATUS.md` 的坑清单 | 改 config |
| `KeyError: 'PackDetInputs is not in the mmpose::transform registry'` | 两阶段 scope 互相污染 | 各自调用点显式 `init_default_scope` |
| `CUDA out of memory` | 4060 是 8GB 但要留余量 | 降 batch size；检测 @1024 从 bs=4 起 |
| `torch.cuda.is_available()` 为 False | 装了 CPU 版 torch | 重装带 `+cu118` 的版本 |
| 训练极慢（比 1650 还慢） | 笔记本没插电 / 降频 | 插电；监控 `clocks.sm` |
| `FileNotFoundError: .../rtmdet_tiny_...py` | junction 没建 | `New-Item -ItemType Junction` |
| Windows `spawn` 多进程报错 | 缺 `if __name__ == '__main__':` | 见 `PROJECT_STATUS.md` |

---

## 七、一句话总结

> **不要搬 `.venv`。搬数据+源码+脚本（1.5 GB），在新机用
> `requirements-lock.txt` 重建环境（10 分钟），然后跑 `_env_check.py` 自检。**
>
> 唯一必须注意的是 **Python 版本锁 3.10.x**、**setuptools 锁 80.9.0**、
> **editable 装 mmpose** 这三件事。torch 的 sm_89 缺失不影响正确性，
> 想提速再单独升级。

这样做的迁移失败率接近零，因为**所有路径都在新机现场生成**，
不存在"路径还指向旧机器"这类隐蔽问题。
