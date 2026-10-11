# Git 双机协作流程

> 场景：本机（GTX 1650）做测试/调参/上报，RTX 4060 笔记本只跑训练。
> 问题：这种分工下**有必要用 git 吗？怎么用？**

---

## 一、结论：**有必要，而且价值很大**

### 1.1 不用 git 会遇到的四个具体问题

| 问题 | 不用 git 时的表现 | 后果 |
|---|---|---|
| **代码漂移** | 手工拷 `*.py` 到新机；改了一处忘了同步 | 两边跑的不是同一个实验，结果**不可比** |
| **配置变更无记录** | 想不起来"这个 AP 是 `padding=1.5` 还是 `1.75` 跑的" | 无法回溯，重复劳动 |
| **改坏了没法退** | 新机上调参改崩了，想回到昨天的版本 | 只能凭记忆手工改回，容易漏 |
| **mmpose 改动同步** | 我们改了 mmpose 5 个文件；换机要**手工重做** | 极易漏掉，然后报 `No module named 'DMKPD_dataset'` |

### 1.2 这个项目有一个**特殊难点**,让 git 更有必要

我实测发现:

```
open-mmlab/mmpose/  ← 是一个**独立的 git clone**
  .git/             HEAD = 759b39c1 (mmpose 官方 main 分支)
                    无 remote（本地仓库）

  我们改动的 5 个文件：
    A  configs/_base_/datasets/DMKPD_meta.py        （新增 48 行）
    A  mmpose/datasets/datasets/DM/__init__.py      （新增 4 行）
    A  mmpose/datasets/datasets/DM/registry.py      （新增 522 行）
    A  mmpose/datasets/datasets/DM/transforms.py    （新增，未跟踪）
    M  mmpose/datasets/datasets/__init__.py         （改 1 行）
```

**这是个隐藏的坑**:我们的自定义数据集代码**住在另一个 git 仓库里**。

如果直接在外层 `git init`:
- `open-mmlab/mmpose` 会被当成 **gitlink**(子模块引用)或被忽略
- 我们最核心的 575 行自定义代码**不会被跟踪**
- 换机时如果只 clone 外层仓库 → **mmpose 里什么都没有 → 全线报错**

**解法**:用 **patch** 把这 5 个文件导出成一个小文件(41 KB),放进外层仓库。
这是 git 处理"给第三方库打补丁"的标准做法。

---

## 二、方案设计

### 2.1 什么进 git,什么不进

```
✅ 进 git（约 1.3 MB）
   · 所有 *.py 脚本（40 个）
   · config.py / config_det.py
   · DMKPD_dataset/splits/*.txt         ← 划分清单（几 KB，实验可复现的关键）
   · det_dataset/annotations/*.json     ← 4 个 json（0.01 GB）
   · patches/                            ← mmpose 的 5 文件改动（41 KB）★
   · *.md 文档（PROJECT_STATUS / MIGRATE_TO_4060 / 本文件）
   · requirements-lock.txt
   · .gitignore

❌ 不进 git（约 7 GB，手动/网盘同步）
   · .venv/                5.58 GB   新机 rebuild
   · DMKPD_dataset/        1.09 GB   除 splits 外
   · work_dirs/            1.07 GB   训练产物
   · modelbase_test/       0.30 GB   预训练权重
   · open-mmlab/           0.03 GB   用 patches/ 代替 ★
   · submission/           300 txt   每次导出都变
   · vis_out2/             可视化    可重新生成
```

### 2.2 为什么 `splits/*.txt` 必须进 git

这是我们**做对的一个设计**。`make_split.py` 用固定种子 42 生成划分:

```
DMKPD_dataset/splits/train.txt   9111 行（每个 json 一行）
DMKPD_dataset/splits/val.txt      480 行
```

如果把这两行清单丢给别人/丢到新机重新生成,**万一代码或环境有细微差异,
划分就变了** —— 那么两次实验的 val AP 就**不可比**了。

只有几 KB,进 git 是必须的。

### 2.3 为什么用 patches/ 而不是跟踪整个 mmpose

| 方案 | 体积 | 换机成本 | 可读性 |
|---|---|---|---|
| **A. 跟踪整个 mmpose** | 0.03 GB / 2289 文件 | 直接可用 | ❌ 每次 commit 都淹没在 mmpose 的文件里 |
| **B. patches/（推荐）** | **41 KB / 1 文件** | 需一步 `git apply` | ✅ diff 一眼看清改了什么 |
| C. 什么都不做 | 0 | 手工重做 5 个文件 | ❌ 极易漏 |

选 B。

---

## 三、一次性初始化（本机执行）

### 步骤 1：导出 mmpose 改动为 patch

```powershell
# ① 让未跟踪的 transforms.py 也进 diff
& git -C open-mmlab/mmpose add -N mmpose/datasets/datasets/DM/transforms.py

# ② 生成 patch（--binary 以防有二进制内容）
New-Item -ItemType Directory -Force patches | Out-Null
& git -C open-mmlab/mmpose diff HEAD --binary |
    Out-File -Encoding utf8 patches/0001-dmkpd-dataset.patch

# ③ 撤销 intent-to-add（恢复原状态）
& git -C open-mmlab/mmpose reset -q mmpose/datasets/datasets/DM/transforms.py

# ④ 验证
Get-Item patches/0001-dmkpd-dataset.patch |
    Select-Object Name, Length
(Get-Content patches/0001-dmkpd-dataset.patch).Count
```

**期望结果**：约 41 KB / 1200 行。
patch 里应包含 5 个文件的 `diff --git` 段。

### 步骤 2：处理 mmpose 仓库里的杂项

```powershell
# vis_results.jpg 是实验残留，删掉（patch 里不该有它）
Remove-Item open-mmlab/mmpose/vis_results.jpg -Force -EA SilentlyContinue

# 给 mmpose 自己的仓库也加个忽略（可选，但它自带 .gitignore 通常够用）
```

### 步骤 3：初始化外层仓库

```powershell
git init
git add .gitignore
git commit -m "chore: 初始化 .gitignore"

git add -A
git status --short        # ← 检查被加入的文件，确认没有大文件/数据
git commit -m "feat: DMKPD 姿态+检测全流程

- 姿态: HRNet-W48 + HeatmapHead(4ch), AP 0.9377（5 epoch, GT bbox）
- 检测: RTMDet-tiny(1类, 1024), mAP 0.784（1 epoch 冒烟）
- 端到端: infer_e2e.py, 300 图 45s, 437 实例
- 提交: submission/ 格式自检全过
- patches/: mmpose 的 5 文件改动（DMKPD_dataset 注册等）"
```

### 步骤 4：❗关键检查 —— 确认 patches 被跟踪、mmpose 没被跟踪

```powershell
# 应该看到 patches/0001-dmkpd-dataset.patch
git ls-files patches/

# 应该**为空**（mmpose 已被忽略）
git ls-files open-mmlab/

# 确认仓库体积（应该 < 5 MB）
git count-objects -vH
```

---

## 四、日常流程

### 4.1 职责划分

```
┌─────────────────────────────┐         ┌──────────────────────────────┐
│ 本机（GTX 1650 4GB）         │         │ RTX 4060 笔记本（8GB）        │
│                             │         │                              │
│  · 数据转换 / 统计           │  push   │  · 只跑训练                   │
│  · 推理 / 端到端验证          │ ──────> │    run_train.py               │
│  · 导出 submission          │         │    run_train_det.py           │
│  · 小规模调参（快实验）       │  pull   │                              │
│  · 可视化 / 排错              │ <────── │  · 只回传 best_*.pth（网盘）   │
│  · **配置变更的发起方**       │ (代码)   │                              │
└─────────────────────────────┘         └──────────────────────────────┘

   代码走 git（同步）
   数据/权重走网盘/移动硬盘（git 装不下）
```

**核心纪律**:
- **配置变更只能在本机发起**,然后 push。新机只 `pull`,不直接在训练机上改 config
- 否则两边配置不一致 → 实验不可比,而且 git 会有冲突

### 4.2 本机：改配置并推送

```powershell
# ① 拉最新（防止新机那边有改动，虽然纪律上不该有）
git pull --rebase

# ② 改配置（例：把 padding 从 1.5 调到 1.75 做实验）
#    用编辑器改 config.py

# ③ 看改了什么（这一步**养成习惯**，避免误改）
git diff

# ④ 提交
git add config.py
git commit -m "exp: padding 1.5 -> 1.75（扩大窗口，容忍检测框误差）"

# ⑤ 推送
git push origin main
```

### 4.3 新机：拉代码并训练

```powershell
git pull

# 确认改动到位
git log --oneline -3
git diff HEAD~1 HEAD -- config.py     # 看上一次改了什么

# 跑训练
& '.\.venv\Scripts\python.exe' run_train_det.py --epochs 24 --val-interval 3 --no-resume
```

### 4.4 新机：训练完回传权重

```powershell
# 只回传 best（不含优化器状态，体积小）
# 本机 40 MB（检测） / 245 MB（姿态）
Copy-Item work_dirs/dmkpd_det/best_coco_bbox_mAP_epoch_24.pth `
          "\\网盘路径\DMKPD\weights\"
```

**本机接收后**:
```powershell
Copy-Item "\\网盘路径\DMKPD\weights\best_coco_bbox_mAP_epoch_24.pth" `
          work_dirs/dmkpd_det/

# 记一笔（重要：便于回溯"这个 AP 是哪份权重跑的"）
git commit --allow-empty -m "chore: 接收 4060 训练的检测器权重（mAP 0.87@ep24）"
```

> **为什么要记这一笔空 commit**:权重不进 git,但"接受了哪份权重 + 它多少分"
> 这个**事实**该记录。将来回溯时能对上。

### 4.5 回到本机做推理

```powershell
# 用新权重重新导出（注意 last_checkpoint 的影响，见下）
& '.\.venv\Scripts\python.exe' infer_e2e.py --export --out submission

# 格式自检
& '.\.venv\Scripts\python.exe' _sub_check.py

# 留一份带时间戳的存档（git 不跟踪 submission/）
Copy-Item submission "submission_$(Get-Date -f yyyyMMdd_HHmm)"
```

---

## 五、换机搭建流程（结合 MIGRATE_TO_4060.md）

git 只解决**代码**同步。数据/权重仍要手动/网盘。完整流程:

```powershell
# ============ 新机（RTX 4060）============

# 1. 装 Python 3.10.x，然后 clone 代码
git clone <仓库地址> D:\DMKPD
cd D:\DMKPD

# 2. 建虚拟环境 + 装依赖（见 MIGRATE_TO_4060.md 阶段 B）
python -m venv .venv
& '.\.venv\Scripts\python.exe' -m pip install `
    torch==2.1.0+cu118 torchvision==0.16.0+cu118 torchaudio==2.1.0+cu118 `
    --index-url https://download.pytorch.org/whl/cu118
& '.\.venv\Scripts\python.exe' -m pip install -r requirements-lock.txt
& '.\.venv\Scripts\python.exe' -m pip install "setuptools==80.9.0"

# 3. ❗拿到 mmpose（git 里没有它）
#    路径 A：网盘拷贝整个 open-mmlab/（0.03 GB，最省事，推荐）
#    路径 B：重新 clone + apply patch（见下）
& git clone --depth 1 https://github.com/open-mmlab/mmpose.git open-mmlab/mmpose
& git -C open-mmlab/mmpose checkout 759b39c1      # 对齐我们的基线 commit
& git -C open-mmlab/mmpose apply ..\..\patches\0001-dmkpd-dataset.patch
#    ⚠️ 注意 apply 时的路径基准

# 4. editable 装 mmpose（重建 finder 路径 —— 这就是不搬 .venv 的核心收益）
& '.\.venv\Scripts\python.exe' -m pip install -e open-mmlab/mmpose --no-build-isolation

# 5. 重建 mmdet junction
New-Item -ItemType Junction -Path mmdet -Target "$PWD\.venv\Lib\site-packages\mmdet"

# 6. 拷数据与权重（网盘/移动硬盘）
#    DMKPD_dataset/     1.09 GB
#    modelbase_test/    0.30 GB
#    det_dataset/       0.01 GB  ← 已在 git 里，可跳过
#    work_dirs/         可选（要续训就拷 epoch_*.pth）

# 7. ❗环境自检（必做）
& '.\.venv\Scripts\python.exe' _env_check.py
#    必须全 OK 才能开始训练
```

### 关于 mmpose 的两条路径

| 路径 | 命令 | 适用 |
|---|---|---|
| **A. 网盘拷 `open-mmlab/`** | 直接复制 0.03 GB | **推荐**。省事，一次搞定，不怕 patch 对不上 |
| **B. clone + apply patch** | `git clone` + `git apply` | 适合想从纯净基线重建、或 patch 有更新时 |

**我推荐 A**:`open-mmlab/` 只有 0.03 GB（比一张图大不了多少),
直接拷过去最稳,不会遇到 patch 上下文不匹配的问题。

`patches/` 的真正价值在于**记录"我们改了什么"** —— 让你随时能
`git diff` 看清这 575 行自定义代码,以及将来 mmpose 升级时知道要重打哪些补丁。

---

## 六、⚠️ 一个必须注意的坑：last_checkpoint

我在检查时发现:

```powershell
work_dirs/dmkpd_det/last_checkpoint  ->  ...\epoch_1.pth
```

`run_train_det.py` 会**自动读取 `last_checkpoint` 并续训**。
这在新机上会造成**混淆**:

| 场景 | 不加 `--no-resume` 的后果 |
|---|---|
| 新机刚拷完权重,想从头训 | 会**从 epoch_1 的权重开始**,而不是 COCO 预训练 |
| 本机想跑新实验 | 同上,实验基线被污染 |

**规则**:
- **想从 COCO 预训练干净开始** → 必须加 `--no-resume`
- **想接着上次训** → 不要加,但要确认 `last_checkpoint` 指向的正是你要续的那个

```powershell
# 检查指向哪里
Get-Content work_dirs/dmkpd_det/last_checkpoint

# 想作为"新实验"从 COCO 开始
& '.\.venv\Scripts\python.exe' run_train_det.py --epochs 24 --no-resume
```

> 这也是为什么 `work_dirs/` 不进 git —— 它带着"上次训练的状态",
> 而这种状态**不该跨实验继承**。每台机器自己管理自己的 `work_dirs/`。

---

## 七、可选的进阶做法

### 7.1 用分支做实验（推荐）

```powershell
# 主线的稳定版本
git branch -M main

# 开实验分支
git switch -c exp/padding-1.75
# ... 改配置、在新机训练 ...
git commit -am "exp: padding 1.75, mAP 0.79->0.86"

# 效果好就合并
git switch main
git merge exp/padding-1.75
git tag exp-padding175-ok        # 打个标签，便于回溯

# 效果差就丢弃（实验完全不留痕）
git switch main
git branch -D exp/padding-1.75
```

**好处**:多个超参实验并行探索时,不会互相污染;失败的实验一键丢弃。

### 7.2 给权重文件算个 hash

权重不进 git,但可以记 hash 来对账:

```powershell
# 新机导出
Get-FileHash work_dirs/dmkpd_det/best_*.pth -Algorithm SHA256 |
    Select-Object Hash, Path | Out-File weights.sha256

# 本机接收后核对（确认传输无误）
Get-FileHash work_dirs/dmkpd_det/best_*.pth -Algorithm SHA256
```

大文件通过网盘传输偶有损坏,核对 hash 是**低成本高收益**的习惯。

### 7.3 用远程私有仓库

```powershell
# Gitee / GitHub 私有仓库 / 学校 GitLab 均可
git remote add origin <私有仓库地址>
git push -u origin main
```

⚠️ **务必用私有仓库**:项目含比赛数据相关信息。

---

## 八、checklist

### 一次性初始化（本机）

- [ ] 导出 patch:
  `git -C open-mmlab/mmpose add -N mmpose/datasets/datasets/DM/transforms.py`
  然后 `git -C open-mmlab/mmpose diff HEAD --binary > patches/0001-dmkpd-dataset.patch`
- [ ] 撤销 intent-to-add:`git -C open-mmlab/mmpose reset -q mmpose/.../transforms.py`
- [ ] 验证 patch 含 **5 个文件**、约 41 KB
- [ ] 删 `open-mmlab/mmpose/vis_results.jpg`
- [ ] `git init`
- [ ] 放 `.gitignore`（已生成）
- [ ] `git add -A` 后**检查** `git status --short`：不含 GB 级文件
- [ ] ❗`git ls-files open-mmlab/` **必须为空**
- [ ] ❗`git ls-files patches/` **必须有 patch**
- [ ] 首次 commit
- [ ] （可选）配远程私有仓库并 push

### 每次改配置

- [ ] 本机改 → `git diff` 看一眼 → commit → push
- [ ] 新机 `git pull` → `git log --oneline -3` 确认
- [ ] **新机上绝不直接改 config**

### 每次训练完

- [ ] 只回传 `best_*.pth`
- [ ] 记 hash
- [ ] 本机空 commit 记录"接收了哪份权重、多少分"

---

## 九、一句话总结

> **用 git,而且必须用。**
>
> 因为本项目有一个隐藏难点:**核心的 575 行自定义代码住在
> `open-mmlab/mmpose` 这个独立 git 仓库里**,不用 patch 机制
> 就**无法跨机同步** —— 换机会直接报 `No module named 'DMKPD_dataset'`。
>
> 方案:**代码走 git(1.3 MB)、数据和权重走网盘(7 GB)、
> mmpose 改动走 patches(41 KB)**,三者分工明确。
>
> 全程只需一条纪律:**配置变更只在本机发起。**
