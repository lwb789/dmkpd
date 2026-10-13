# 新机部署手册（RTX 4060 笔记本 · 只负责训练）

> **职责边界**（本次明确）
> ```
> ┌──────────────────────────────┐         ┌──────────────────────────────┐
> │ 本机（GTX 1650 4GB）          │         │ RTX 4060 笔记本               │
> │                              │  git    │                              │
> │ · 数据准备 / 统计             │ ──────> │ · **只跑训练**                │
> │ · 测试集推理                  │         │   run_train_det.py           │
> │ · 导出 submission            │         │   run_train.py（可选）        │
> │ · 调参 / 可视化 / 排错        │  weights│                              │
> │ · 提交上报                    │ <────── │ · **只回传 best_*.pth**        │
> └──────────────────────────────┘  网盘   └──────────────────────────────┘
> ```
> 新机**不需要**测试集、不需要 submission、不需要做推理。

---

## 目录

- [零、开始前的清单](#零开始前的清单)
- [一、本机：准备迁移包](#一本机准备迁移包)
- [二、新机：装 Python](#二新机装-python)
- [三、新机：用 git 拿代码](#三新机用-git-拿代码)
- [四、新机：拷数据与权重](#四新机拷数据与权重)
- [五、新机：建虚拟环境 + 装依赖](#五新机建虚拟环境--装依赖)
- [六、新机：环境自检（❗必做）](#六新机环境自检必做)
- [七、新机：开始训练](#七新机开始训练)
- [八、训练完：回传权重](#八训练完回传权重)
- [九、本机：接权重 + 推理 + 提交](#九本机接权重--推理--提交)
- [十、报错对照表](#十报错对照表)

---

## 零、开始前的清单

### 0.1 新机需要什么

| 项 | 要求 | 备注 |
|---|---|---|
| 操作系统 | Windows 10/11 x64 | 与开发机一致 |
| GPU | RTX 4060 (8 GB) | ✅ 已具备 |
| 磁盘空间 | **≥ 25 GB** | 数据 1.4 G + 环境 6 G + 训练产物 3 G + 余量 |
| Python | **3.10.x**（必须是这个版本） | ⚠️ 见 §2 |
| 网络 | 能访问 GitHub（或备用方案） | 见 §3.3 |
| 网络硬盘 | 用于传数据和回收权重 | U 盘 / 网盘 / 局域网共享 |

### 0.2 体积预算

```
本机要传出去的（约 1.39 GB）:
  DMKPD_dataset/train/images        0.67 GB   9591 张 jpg
  DMKPD_dataset/train/annotations   0.39 GB   9591 个 json
  modelbase_test                    0.30 GB   2 个预训练权重
  open-mmlab                        0.03 GB   mmpose 源码
                                  ─────────
                                  1.39 GB

走 git 的（约 5.8 MB）:
  代码 + splits + det_dataset 标注 + patches

❗不需要传的（新机用不上）:
  DMKPD_dataset/test    0.03 GB   测试集（推理在本机做）
  work_dirs             1.07 GB   训练产物（新机从零开始）
  submission                    本机产出
  vis_out2                      本机产出
  .venv                 5.58 GB   新机重建（见 §5）
```

---

## 一、本机：准备迁移包

### 1.1 确认本机代码已全部提交并推送

```powershell
cd D:\Personal\长安大学\人工智能创新大赛\DMKPD

git status -sb
# 期望：## master...origin/master    （无 ahead，说明已同步）
```

如果有未提交的改动：
```powershell
git add -A
git status --short        # 先看一眼
git commit -m "说明改了什么"
git push
```

### 1.2 打迁移包（数据与权重）

我用脚本帮你打包，避免漏文件：

```powershell
# 见 §1.3 的脚本
& '.\.venv\Scripts\python.exe' _make_transfer.py
```

### 1.3 脚本内容（我会生成 `_make_transfer.py`）

它会：
1. 检查必需文件是否齐全
2. 把它们复制到 `_transfer/` 目录（保持目录结构）
3. 报出总体积，便于确认

### 1.4 拷到移动硬盘

```powershell
# 假设 E: 是移动硬盘
New-Item -ItemType Directory -Force "E:\DMKPD_transfer" | Out-Null
Copy-Item _transfer\* "E:\DMKPD_transfer\" -Recurse -Force

# 核对（两边文件数应一致）
(Get-ChildItem _transfer -Recurse -File).Count
(Get-ChildItem "E:\DMKPD_transfer" -Recurse -File).Count
```

---

## 二、新机：装 Python

### 2.1 ❗版本必须是 3.10.x

原因：`mmcv` 的编译扩展是 **`_ext.cp310-win_amd64.pyd`**，与
Python 的 **ABI 版本 cp310** 绑定。装 3.11/3.12 会导致
`ImportError: DLL load failed`。

### 2.2 安装方式（任选）

**方式 A：winget（推荐）**
```powershell
winget install Python.Python.3.10
```

**方式 B：官网安装包**
1. 打开 https://www.python.org/downloads/release/python-31011/
2. 下载 **Windows installer (64-bit)**
3. ❗安装时**勾选** `Add python.exe to PATH`

### 2.3 验证

```powershell
python --version
# 必须输出 Python 3.10.x
```

⚠️ 若新机装了多个 Python，确认 `python` 指向 3.10：
```powershell
where.exe python
# 第一行应是 3.10 的路径；不是的话调整 PATH 顺序
```

---

## 三、新机：用 git 拿代码

### 3.1 安装 git（如果没装）

```powershell
winget install Git.Git
```

验证：`git --version`

### 3.2 clone 仓库

```powershell
# 建议放在短路径，避免 260 字符路径长度限制
cd D:\
git clone https://github.com/lwb789/dmkpd.git DMKPD
cd D:\DMKPD
```

**期望**：拉下 **52 个文件**，约 5.8 MB。

```powershell
git ls-files | Measure-Object -Line
# 期望 52
```

### 3.3 ❗如果 GitHub 连不上

新机网络环境可能不同。三种备选：

**备选 1：配代理**（若新机也有 Clash 之类）
```powershell
# 先确认代理端口在监听
Test-NetConnection 127.0.0.1 -Port 7897

# 配给 git
git config --global http.proxy  http://127.0.0.1:7897
git config --global https.proxy http://127.0.0.1:7897

# 用一次性参数测试（推荐先这样验证）
git -c http.proxy=http://127.0.0.1:7897 ls-remote https://github.com/lwb789/dmkpd.git
```

> ❗**不要用环境变量设代理**（`$env:HTTPS_PROXY=...`）。
> 实测踩过：它会残留在会话里，代理关掉后所有 git 命令都失败。
> 用 `git -c http.proxy=...` 一次性参数最安全。

**备选 2：从本机拷 `git archive` 包**
```powershell
# 本机执行
git archive -o "E:\dmkpd_code.zip" HEAD
# 新机解压
Expand-Archive "E:\dmkpd_code.zip" -DestinationPath D:\DMKPD -Force
```
⚠️ 这样**没有 .git 目录**，新机后续不能 pull。适合临时用。

**备选 3：改用 Gitee**
```powershell
# 本机先建 Gitee 私有仓库，然后
git remote add gitee https://gitee.com/<用户名>/dmkpd.git
git push gitee master
# 新机
git clone https://gitee.com/<用户名>/dmkpd.git D:\DMKPD
```

### 3.4 配 git 身份（否则 commit 报错）

```powershell
git config user.name  "lwb789"
git config user.email "abc1021004366@163.com"
```

### 3.5 ❗新机的纪律

> **新机只 `pull`，永远不直接改 config / 代码。**

理由：本机是"配置变更的唯一发起方"。如果新机也改，两边配置不一致
→ 实验不可比，而且 `git pull` 会冲突。

要改配置 → **告诉本机改**，然后新机 `git pull`。

---

## 四、新机：拷数据与权重

### 4.1 拷贝

把移动硬盘里的 `DMKPD_transfer` 内容拷到 `D:\DMKPD\`：

```powershell
# 假设移动硬盘是 E:
Copy-Item "E:\DMKPD_transfer\*" "D:\DMKPD\" -Recurse -Force
```

拷完目录结构应是：

```
D:\DMKPD\
├── DMKPD_dataset\
│   ├── train\
│   │   ├── images\          9591 张 jpg
│   │   └── annotations\     9591 个 json
│   └── splits\              ✅ 来自 git
├── modelbase_test\
│   ├── td-hm_hrnet-w48_....pth          257 MB
│   └── rtmdet_tiny_8xb32-....pth         55 MB
├── open-mmlab\              ❗**不进 git**，按 §5.5 重建 + 打 patch
├── det_dataset\             ✅ 来自 git
├── patches\                 ✅ 来自 git
├── config.py config_det.py run_train.py run_train_det.py ...
├── _env_check.py            ✅ 来自 git（环境体检）
└── _preflight_check.py      ✅ 来自 git（训练前置校验，见 §6.4）
```

### 4.2 核对

```powershell
cd D:\DMKPD

# 图片数（应为 9591）
(Get-ChildItem DMKPD_dataset\train\images -Filter *.jpg).Count

# 标注数（应为 9591）
(Get-ChildItem DMKPD_dataset\train\annotations -Filter *.json).Count

# 权重（应看到 2 个 .pth）
Get-ChildItem modelbase_test\*.pth | Select-Object Name

# mmpose 源码（应存在）
Test-Path open-mmlab\mmpose\mmpose\datasets\datasets\DM\registry.py
```

**四项都要符合预期**才继续。

### 4.3 关于 `DMKPD_dataset/test/`

**新机不需要**。它是推理用的，而推理在本机做。

但如果空间够（0.03 GB），拷过去也无害 —— 万一你想在新机顺手验证
`infer_e2e.py` 也能跑。

---

## 五、新机：建虚拟环境 + 装依赖

### 5.1 建 venv

```powershell
cd D:\DMKPD
python -m venv .venv
```

### 5.2 装 PyTorch（❗先做这一步）

```powershell
& '.\.venv\Scripts\python.exe' -m pip install --upgrade pip

& '.\.venv\Scripts\python.exe' -m pip install `
    torch==2.1.0+cu118 torchvision==0.16.0+cu118 torchaudio==2.1.0+cu118 `
    --index-url https://download.pytorch.org/whl/cu118
```

**为什么用 cu118 而不是更新的版本**：与开发机**完全一致**，
不引入新变量。mmcv 2.1.0 官方支持 torch ≤ 2.1。

> **关于 4060 的性能**：torch 2.1.0+cu118 的编译架构列表最高到
> `sm_86`，没有 `sm_89`（4060 是 Ada）。但 CUDA 的 binary
> compatibility 允许 sm_86 的 cubin 在 sm_89 上运行 →
> **能跑，正确性无问题**，只是少了 Ada 的专属优化。
>
> 想榨满性能可换 torch 2.4+cu121，但代价是 **mmcv 也要升到 2.2+**。
> **建议先用 cu118 跑通，稳定后再考虑升级。**

**验证**：
```powershell
& '.\.venv\Scripts\python.exe' -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
# 期望：2.1.0+cu118 True NVIDIA GeForce RTX 4060 ...
```

### 5.3 装其余依赖

```powershell
& '.\.venv\Scripts\python.exe' -m pip install -r requirements-lock.txt
```

⚠️ 这一步会尝试装 `torch==2.1.0+cu118` —— 已装过会被跳过，正常。

### 5.4 ❗❗ 装 setuptools 80.9.0（关键）

```powershell
& '.\.venv\Scripts\python.exe' -m pip install "setuptools==80.9.0"
```

**为什么必须锁这个版本**：
- ≥ 64：支持 PEP 660（editable 安装）
- < 81：保留 `pkg_resources`

setuptools ≥ 81 **移除了 `pkg_resources`**，会让 mmpose 的
`get_installed_path()` 报
`ModuleNotFoundError: No module named 'pkg_resources'`。

**验证**：
```powershell
& '.\.venv\Scripts\python.exe' -c "import setuptools, pkg_resources; print(setuptools.__version__)"
# 期望：80.9.0
```

### 5.5 ❗❗ editable 安装 mmpose（关键）

```powershell
& '.\.venv\Scripts\python.exe' -m pip install -e open-mmlab/mmpose --no-build-isolation
```

**这一步做什么**：生成
`.venv\Lib\site-packages\__editable___mmpose_1_3_2_finder.py`，
里面把 `mmpose` 映射到**新机的实际路径**。

> ❗**这就是「不搬 .venv」的核心收益**：
> 开发机的那个 finder 文件里硬编码了
> `D:\Personal\长安大学\...\open-mmlab\mmpose\mmpose`。
> 如果直接拷 .venv，新机上所有 import 都会指向不存在的路径。

**为什么加 `--no-build-isolation`**：避免 pip 在隔离环境里重新拉一个
setuptools（可能是 ≥81 的版本），破坏 §5.4 的约束。

**验证**：
```powershell
& '.\.venv\Scripts\python.exe' -c "
import mmpose
print('mmpose:', mmpose.__version__)
print('path  :', mmpose.__file__)
from mmpose.datasets.datasets.DM.registry import DMKPD_dataset
print('DMKPD_dataset OK')
"
```
**关键**：`path` 必须指向 `D:\DMKPD\open-mmlab\mmpose\...`（**新机路径**），
不能是旧机器路径。

### 5.6 重建 mmdet junction

`config_det.py` 里 `_base_` 引用了 `mmdet/.mim/configs/...`，
所以需要在项目根建一个 junction：

```powershell
cd D:\DMKPD
New-Item -ItemType Junction -Path mmdet `
    -Target "$PWD\.venv\Lib\site-packages\mmdet"
```

**验证**：
```powershell
Test-Path 'mmdet\.mim\configs\rtmdet\rtmdet_tiny_8xb32-300e_coco.py'
# 期望 True
```

### 5.7 ❗不要在项目里建 `.gitignore` 之外的干扰

如果 §3 用了 `git archive`（备选 2）而没有 `.git`，
后续 commit/pull 都不可用。建议尽量走 §3.2 的正常 clone。

---

## 六、新机：环境自检（❗必做）

### 6.1 跑自检脚本

```powershell
cd D:\DMKPD
& '.\.venv\Scripts\python.exe' _env_check.py
```

### 6.2 期望输出

脚本会输出 **7 大类约 30 项**检查：

```
1. 基础环境      Python 3.10.x / Windows x64 / CUDA / sm 架构
2. 框架版本      mmcv 2.1.0 / mmengine 0.10.7 / mmdet 3.2.0
                 mmcv.ops 编译扩展 / mmpose 指向本项目 / pkg_resources
3. 自定义模块    DMKPD_dataset / CheckKeypointOutOfBounds
4. 数据与权重    划分清单 / 检测标注 / 权重 / junction / 图片可读
5. 配置+构建     姿态 63.60M ✅  检测 4.87M ✅
6. 管线前向      检测 (3,1024,1024) / 姿态 (3,256,256)
7. GPU 冒烟      前向+反向可跑，报告耗时与峰值显存
```

**最后一行必须是**：
```
全部通过 —— 环境就绪，可以开始训练。
```

### 6.3 如果失败

| 失败项 | 修法 |
|---|---|
| Python 不是 3.10 | 重装（§2） |
| `mmcv.ops（编译扩展）可导入` FAIL | 确认 Python 是 3.10.x |
| `mmpose 指向本项目源码` FAIL | 重跑 §5.5 |
| `pkg_resources` FAIL | 重跑 §5.4 |
| `mmdet junction 可解析` FAIL | 重跑 §5.6 |
| 数据文件缺失 | 重做 §4.1 |
| `姿态/检测模型可构建` FAIL | 看具体报错，多半是依赖问题 |

### 6.4 ❗再跑一次「训练前置校验」

`_env_check.py` 只回答**「这台机器能不能跑」**；
`_preflight_check.py` 回答**「这次训练能不能按我预期的参数跑、要多久、有坑吗」**。

```powershell
# 我要从头训 12 轮姿态
& '.\.venv\Scripts\python.exe' _preflight_check.py --epochs 12 --val-interval 1
```

它检查 9 节：解释器依赖 / **open-mmlab 的 5 个 patch 文件** / 数据集完整性
/ 权重文件 / **work_dir 残留** / **调度器按 --epochs 重算的预览** /
数据管线前向 / **GPU 训练冒烟 + 耗时估算** / 结论。

#### ★ 它专门会抓的两个坑

**坑 1：续训陷阱**

`run_train.py` 的逻辑是「在 work_dir 下按 **mtime** 找任意 `*.pth`，
找到就把 `runner._resume = True`」。所以只要 `work_dirs/dmkpd/` 里
还留着 `epoch_5.pth`，**即使你写了 `--epochs 12` 也会从第 5 轮接着训**，
而不是从头训 12 轮。

脚本会打印 work_dir 下的所有权重并给出两种解法：

```powershell
# 解法 A（推荐，保持目录干净）
New-Item -ItemType Directory -Force work_dirs\dmkpd\_old_best
Move-Item work_dirs\dmkpd\*.pth work_dirs\dmkpd\_old_best\ -Force
Remove-Item work_dirs\dmkpd\last_checkpoint -Force -ErrorAction SilentlyContinue

# 解法 B（保留原目录，靠参数绕过）
#   命令里加 --no-resume
```

**坑 2：调度器随 `--epochs` 重算**

`run_train.py` 会把 `MultiStepLR.milestones` 重算为 `[int(epochs * 0.8)]`：

| `--epochs` | milestones | 含义 |
|---|---|---|
| 8 | `[6]` | 最后 2 轮微调 |
| 12 | `[9]` | 最后 3 轮微调 |
| 24 | `[19]` | 最后 5 轮微调 |

注意这**不是**官方 210 epoch 的 `168` 的等比缩放，属于我们自己的约定。
脚本会把它打印出来，避免「以为 lr 会降结果没降」。

#### `holdout1702 ⊂ train.txt` —— 结构性泄漏（**已于 2026-10-13 修复**）

**泄漏只发生在姿态一侧**：

| 模型 | 训练集 | 图数（修前 → 修后） | 与 holdout 交集 |
|---|---|---|---|
| **检测器** | `instances_train.json` | 8710 → 8710 | **0** ✅ |
| **姿态** | `splits/train.txt` | 9111 → **8711** | **400 → 0** ✅ |

修复方式：`make_split.py` 现在会**直接读检测器的权威清单
`instances_holdout1702.json`**，从其产出的 `train.txt` 里扣除这 400 张。

修复后三方核对：

```
train 8711 + val 480 + holdout 400 = 9591   ✓ 不重不漏
train ∩ holdout = 0    val ∩ holdout = 0    train ∩ val = 0   ✓
```

**`val.txt` 逐位不变**（480 行、集合完全相同）→ 已有基线可比。

##### ❗两个设计要点（都是我踩过的坑）

**坑 A：不能先扣 holdout 再抽 val**

我第一版把 holdout 从抽样池扣掉后才抽 val，`permutation(9191)` 与
`permutation(9591)` 的前 480 个索引**不是同一批样本**，会让 `val.txt`
整体变化、已有基线（1101 实例 / AP 0.9377）全部失去可比性。

**正确顺序：先按全部样本抽 val，再从剩下的里扣 holdout。**

**坑 B：不能自己复制 RNG 去抽 holdout**

`convert_det.py` 用 `random.Random(42).shuffle`，我用了
`np.random.default_rng(42).shuffle` —— **RNG 不同则选出的样本不同**，
两份 holdout 只有 346/400 重合，train 集也对不齐。

**正确做法：直接读 `instances_holdout1702.json` 作为单一事实来源。**

##### 实测确认：姿态并没有"记住"这些图（`_leak_test.py`）

在同一分布上对比「训练时见过 vs 没见过」：

| 组 | 图 | 实例 | OKS 均值 | EPE 中位 |
|---|---|---|---|---|
| holdout（见过） | 400 | 509 | 0.9416 | 4.27 |
| val-1702（没见过） | 123 | 164 | 0.9438 | 4.50 |
| **差** | | | **-0.0022** | -0.23 |

差值 ≈ 0（见过组甚至略差），按尺寸分层正负交替 —— 属噪声。
原因：`rotate_prob=1.0` 的随机旋转破坏了像素级记忆，且这 400 张只占 4.4%。

**结论：泄漏影响的是"把 holdout 当未见数据来评测"这件事，不是训练本身。**
- `splits/val.txt` —— `train ∩ val = 0`，**始终干净**，P1 gap 分析不受影响
- **榜单** —— 主办方自己的测试集，**始终不受影响**
- 只有 `_ap_holdout.py` 的自测绝对值需要这个补丁

⚠️ **`train.txt` 变了 → 旧 5 轮权重是用旧 train.txt 训的**，
与新的「从头训 12 轮」不可直接对比。要严格对比，必须在新划分上重训
（这正是计划中要做的事）。

---

## 七、新机：开始训练

### 7.1 ❗先测显存与速度（可选但推荐）

4060 有 **8 GB** 显存（开发机只有 4 GB），可以放大 batch size。
先测一下最佳配置：

```powershell
& '.\.venv\Scripts\python.exe' _det_bench2.py
```

**开发机的基准**（用于对比）：
```
输入        bs   峰值显存   s/iter   epoch(9110图)
1024x1024    2     1.75G    0.254      19.3min
1024x1024    4     3.21G    0.516      19.6min   ← 当前配置
```

**在 4060 上试**：`bs=6`、`bs=8` 看显存是否够。
如果 bs=8 能放下，训练速度会明显提升。

### 7.2 ❗batch size 变了必须改 lr

```python
# 当前（bs=4）
lr = 0.004 * (4/32) = 5e-4

# 如果改成 bs=8
lr = 0.004 * (8/32) = 1e-3
```

**改法**：这是配置变更，按纪律应该**在本机改后 push**。
临时方案：用启动器的命令行参数覆盖（如果支持），或先按 bs=4 跑。

> **建议**：第一次先用 **bs=4 原样跑通**，确认流程无误。
> 第二轮再优化 bs/lr。

### 7.3 启动训练

```powershell
cd D:\DMKPD

& '.\.venv\Scripts\python.exe' run_train_det.py `
    --epochs 24 --val-interval 3 --no-resume `
    > _det_train_out.txt 2>&1
```

**参数说明**：

| 参数 | 说明 |
|---|---|
| `--epochs 24` | 总 epoch（P1 建议值） |
| `--val-interval 3` | 每 3 epoch 验证一次并保存 best |
| `--no-resume` | ❗**从 COCO 预训练权重开始**，不续训 |
| 重定向到文件 | 避免控制台 GBK 编码问题 |

**为什么加 `--no-resume`**：启动器会读 `work_dirs/dmkpd_det/last_checkpoint`
并自动续训。新机没有 `work_dirs`，本来就相当于 no-resume；
但显式加上更清晰（若之前跑过一半想重来，也能干净开始）。

### 7.4 建议用 `nohup` 式后台运行

24 epoch 约 **6~8 小时**（4060 上可能更快）。建议：

```powershell
# 方式 1：Start-Process 后台
Start-Process -FilePath '.\.venv\Scripts\python.exe' `
    -ArgumentList 'run_train_det.py','--epochs','24','--val-interval','3','--no-resume' `
    -RedirectStandardOutput '_det_train_out.txt' `
    -RedirectStandardError '_det_train_err.txt' `
    -NoNewWindow

# 方式 2：直接在终端跑，别关窗口
```

### 7.5 训练中监控

```powershell
# 看日志尾部
Get-Content _det_train_out.txt -Tail 20

# 或看 mmengine 自己的日志
$d = Get-ChildItem work_dirs\dmkpd_det -Directory | Sort-Object Name | Select-Object -Last 1
Get-Content "$($d.FullName)\$($d.Name).log" -Tail 10
```

**关注指标**（P1 指出这两个最关键）：
```
coco/bbox_mAP_s   小目标（<40px）  ← P1 显示漏检最严重
coco/bbox_mAP_m   中目标（60-120px）← P1 显示漏检绝对数最多
```

### 7.6 ⚠️ 笔记本的两个坑

| 坑 | 说明 |
|---|---|
| **必须插电** | 电池模式下 GPU 会大幅降频。实测开发机 CPU 满载时功耗 40-47W，笔记本更依赖外接电源 |
| **散热降频** | 长时间满载（6-8 小时）容易撞温度墙。`run_train_det.py` 的 `GpuMonitor` 会记录 `clock_sm`，**如果它持续低于标称值就是降频了** |

**建议**：垫高机身 / 用散热垫；训练结束后看日志里的
`GPU 采样` 报告（会输出温度、功耗、频率的均值/最大值）。

---

## 八、训练完：回传权重

### 8.1 确认训练产物

```powershell
cd D:\DMKPD
Get-ChildItem work_dirs\dmkpd_det\*.pth |
    Select-Object Name, @{n='MB';e={[math]::Round($_.Length/1MB,1)}}
```

会看到：
```
best_coco_bbox_mAP_epoch_XX.pth    约 40 MB    ← ✅ 要回传这个
epoch_XX.pth                       约 78 MB    ← 含优化器状态，不必回传
```

### 8.2 只看 best

```powershell
$best = Get-ChildItem work_dirs\dmkpd_det\best_*.pth | Sort-Object LastWriteTime | Select-Object -Last 1
$best.Name
```

### 8.3 ❗记录指标（重要）

回传前把**最终指标**记下来，本机要用：

```powershell
# 从日志里抓最后的验证结果
Get-Content _det_train_out.txt | Select-String 'bbox_mAP' | Select-Object -Last 2

# 或在 mmengine 日志里
$d = Get-ChildItem work_dirs\dmkpd_det -Directory | Sort-Object Name | Select-Object -Last 1
Get-Content "$($d.FullName)\$($d.Name).log" | Select-String 'coco/bbox_mAP:' | Select-Object -Last 3
```

把这几行**贴给本机**（或记在 `_transfer_back.md` 里一起传回）。

### 8.4 算 hash（核对传输完整性）

```powershell
Get-FileHash $best.FullName -Algorithm SHA256 |
    Select-Object Hash, Path | Tee-Object -Append weights.sha256

# 或直接写文件
"$($best.Name)  $((Get-FileHash $best.FullName -Algorithm SHA256).Hash)" |
    Out-File -Append "E:\weights.sha256" -Encoding utf8
```

### 8.5 拷回硬盘

```powershell
New-Item -ItemType Directory -Force "E:\DMKPD_weights" | Out-Null
Copy-Item $best.FullName "E:\DMKPD_weights\"
Copy-Item "E:\weights.sha256" "E:\DMKPD_weights\" 2>$null
```

### 8.6 要不要回传训练日志？

**建议回传**（体积小、信息量大）：
```powershell
$d = Get-ChildItem work_dirs\dmkpd_det -Directory | Sort-Object Name | Select-Object -Last 1
Copy-Item "$($d.FullName)\$($d.Name).log" "E:\DMKPD_weights\train_log.txt"
Copy-Item "vis_data\*.json" "E:\DMKPD_weights\vis_data\" -Recurse -Force
```

有了日志，本机可以画训练曲线、分析收敛情况。

---

## 九、本机：接权重 + 推理 + 提交

### 9.1 接收并核对 hash

```powershell
cd D:\Personal\长安大学\人工智能创新大赛\DMKPD

# 拷入
Copy-Item "E:\DMKPD_weights\best_*.pth" work_dirs\dmkpd_det\ -Force

# 核对 hash（应与新机一致）
Get-FileHash work_dirs\dmkpd_det\best_*.pth -Algorithm SHA256
Get-Content "E:\DMKPD_weights\weights.sha256"
```

### 9.2 用新权重重新导出提交

```powershell
# 推理脚本会自动找 work_dirs/dmkpd_det/best_*.pth
& '.\.venv\Scripts\python.exe' infer_e2e.py --export --out submission

# 格式自检
& '.\.venv\Scripts\python.exe' _sub_check.py
```

### 9.3 复测 gap，看改善多少

**这是关键验证** —— 用 P1 的脚本量化新检测器的真实收益：

```powershell
# 清掉旧缓存，强制重新检测
Remove-Item _gap_det_cache.json -ErrorAction SilentlyContinue

& '.\.venv\Scripts\python.exe' _gap_check.py
& '.\.venv\Scripts\python.exe' _gap_analyze.py
& '.\.venv\Scripts\python.exe' _gap_vis.py
```

**对比基准**（1 epoch 检测器的结果）：
```
检测召回 91.4%  →  端到端 OKS 0.8507  （GT框条件 0.9153）
损失分解: 漏检 73.8% + 框偏移 26.2%
```

**期望**：召回提升到 95%+，端到端 OKS 提升到 0.88+。

### 9.4 记录到 git

```powershell
# 留一份带时间戳的提交存档
Copy-Item submission "submission_$(Get-Date -f yyyyMMdd_HHmm)"

# 空 commit 记录"接受了哪份权重、多少分"
git commit --allow-empty -m "chore: 接收 4060 训练的检测器（mAP_X.XXX@ep24），端到端 OKS 0.XXX"
git push
```

> **为什么要记这一笔**：权重不进 git，但"接受了哪份权重 + 它多少分"
> 这个**事实**该记录，将来回溯能对上。

---

## 十、报错对照表

| 报错 | 根因 | 修法 |
|---|---|---|
| `ModuleNotFoundError: No module named 'pkg_resources'` | setuptools ≥ 81 | `pip install "setuptools==80.9.0"` |
| `No module named 'mmpose.datasets.datasets.DM'` | editable 没装 / 路径错 | 重跑 §5.5 |
| `mmpose.__file__` 指向旧机器路径 | 搬了旧 `.venv` | 删 `.venv` 重建（§5） |
| `ImportError: DLL load failed` (mmcv) | Python 版本不对 | 装 **3.10.x** |
| `torch.cuda.is_available()` 为 False | 装了 CPU 版 torch | 重装带 `+cu118` |
| `FileNotFoundError: .../rtmdet_tiny_...py` | junction 没建 | 重跑 §5.6 |
| `KeyError: 'PackDetInputs is not in the mmpose::transform registry'` | 作用域被污染 | 检查脚本里的 `init_default_scope` |
| `RuntimeError: Only one of type or by_epoch` | config 冲突 | 见 `PROJECT_STATUS.md` 坑清单 |
| `CUDA out of memory` | batch size 太大 | 降 bs（4060 8GB 可试 bs=6） |
| 训练极慢（比 1650 还慢） | 没插电 / 降频 | 插电；看日志里的 `clock_sm` |
| `UnicodeEncodeError: 'gbk' codec` | 控制台编码（打印 emoji/中文出错） | 加 `$env:PYTHONIOENCODING='utf-8'`，或重定向到文件 |
| `Could not resolve host: github` | 域名写错 | 检查地址里有没有 `.com` |
| `Failed to connect ... over proxy 127.0.0.1` | 代理残留 | `Remove-Item Env:HTTPS_PROXY`（或开新终端） |
| `error: remote origin already exists` | 已配过 | `git remote set-url origin <地址>` |

---

## 附：完整命令速查（新机）

```powershell
# ============ 1. 装 Python 3.10 + git ============
winget install Python.Python.3.10
winget install Git.Git
python --version          # 必须 3.10.x

# ============ 2. 拿代码 ============
cd D:\
git clone https://github.com/lwb789/dmkpd.git DMKPD
cd D:\DMKPD
git config user.name  "lwb789"
git config user.email "abc1021004366@163.com"

# ============ 3. 拷数据与权重（从移动硬盘）============
Copy-Item "E:\DMKPD_transfer\*" "D:\DMKPD\" -Recurse -Force
(Get-ChildItem DMKPD_dataset\train\images -Filter *.jpg).Count   # 9591

# ============ 4. 建环境 ============
python -m venv .venv
& '.\.venv\Scripts\python.exe' -m pip install --upgrade pip
& '.\.venv\Scripts\python.exe' -m pip install torch==2.1.0+cu118 torchvision==0.16.0+cu118 torchaudio==2.1.0+cu118 --index-url https://download.pytorch.org/whl/cu118
& '.\.venv\Scripts\python.exe' -m pip install -r requirements-lock.txt
& '.\.venv\Scripts\python.exe' -m pip install "setuptools==80.9.0"          # ❗关键
& '.\.venv\Scripts\python.exe' -m pip install -e open-mmlab/mmpose --no-build-isolation   # ❗关键
New-Item -ItemType Junction -Path mmdet -Target "$PWD\.venv\Lib\site-packages\mmdet"

# ============ 5. 自检（必须全 OK）============
& '.\.venv\Scripts\python.exe' _env_check.py

# ============ 6. 训练 ============
& '.\.venv\Scripts\python.exe' run_train_det.py --epochs 24 --val-interval 3 --no-resume > _det_train_out.txt 2>&1

# ============ 7. 回传 ============
Get-ChildItem work_dirs\dmkpd_det\best_*.pth
Copy-Item work_dirs\dmkpd_det\best_*.pth "E:\DMKPD_weights\"
```

---

## 附：职责边界总结

| 任务 | 在哪台机器做 |
|---|---|
| 数据转换 / 统计 | 本机 |
| **训练检测器 / 姿态** | **新机** |
| 测试集推理 | 本机 |
| 导出 submission | 本机 |
| 提交格式自检 | 本机 |
| gap 量化分析 | 本机 |
| 调参实验（小规模） | 本机 |
| 可视化 / 排错 | 本机 |

**新机的角色**：一台纯计算设备。进去 → `git pull` → 跑训练 → 回传权重。
**不需要**测试集、不需要推理、不需要产出提交文件。
