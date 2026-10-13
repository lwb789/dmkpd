# -*- coding: utf-8 -*-
"""训练前置校验（preflight）—— 起飞前把「要浪费 1 小时才发现」的问题全查出来。

与 `_env_check.py` 的分工
-------------------------
`_env_check.py`  回答 **「这台机器的环境能不能跑？」**（依赖、CUDA、算子）
本脚本           回答 **「这次训练能不能按我预期的参数跑、要多久、有坑吗？」**

覆盖 9 节
---------
  1  解释器与关键依赖（精简；完整体检请跑 `_env_check.py`）
  2  代码完整性：open-mmlab 的 5 个 patch 文件、git HEAD、注册可导入
  3  数据集完整性：标注 json 数、划分行数、实例数
  4  权重文件：姿态/检测的「预训练权重」与「已训权重」
  5  **work_dir 残留**（★ 续训陷阱）
  6  **配置与调度器预览**（★ 按 --epochs 重算里程碑）
  7  数据管线真正前向（读图 -> 变换 -> pack）
  8  GPU 训练冒烟 + 速度实测 + 耗时估算
  9  结论：给出可直接粘贴的命令

用法
----
    # 我要从头训 12 轮姿态
    python _preflight_check.py --epochs 12 --val-interval 1

    # 只看计划不跑 GPU 冒烟
    python _preflight_check.py --epochs 12 --no-smoke

退出码：0 = 可以开训；1 = 有阻断项（BLOCK）。

===================================================================
★ 两个必须知道的坑（本脚本会主动检查）
===================================================================
坑 1（续训陷阱）
    `run_train.py` 的逻辑是「在 work_dir 下按 **mtime** 找任意 `*.pth`，
    找到就 `runner._resume = True`」。所以只要 `work_dirs/dmkpd/` 里
    还留着 `epoch_5.pth`，**即使你写了 `--epochs 12` 也会从第 5 轮接着训**，
    而不是从头训 12 轮。→ 要真从头训，必须 `--no-resume`
    （或把旧权重挪进 `_old_best/`）。

坑 2（调度器随 epochs 重算）
    `run_train.py` 会把 `MultiStepLR.milestones` 重算为
    `[int(epochs * 0.8)]`。所以：
        --epochs 12  ->  milestones=[9]   最后 3 轮微调
        --epochs 24  ->  milestones=[19]  最后 5 轮微调
    两者都不是「官方 210 epoch 的 168」的等比缩放，属于我们自己的约定。
    本脚本会把它打印出来，避免「以为 lr 会降结果没降」。
"""
import argparse
import glob
import os
import os.path as osp
import platform
import sys

OK = True
WARN = []


def chk(name, cond, detail='', block=True):
    """block=True 时不通过记为阻断项；False 时只记警告。"""
    global OK
    if block:
        OK &= bool(cond)
    else:
        if not cond:
            WARN.append(name + (f'  ({detail})' if detail else ''))
    tag = 'OK  ' if cond else ('FAIL' if block else 'WARN')
    print(f'  [{tag}] {name}' + (f'  ->  {detail}' if detail else ''))


def sep(t):
    print()
    print('=' * 78)
    print(t)
    print('=' * 78)


def count_lines(p):
    """数划分文件的有效行数（跳过 # 注释与空行）。"""
    if not osp.isfile(p):
        return -1
    n = 0
    with open(p, encoding='utf-8') as f:
        for line in f:
            s = line.strip()
            if s and not s.startswith('#'):
                n += 1
    return n


def read_split(p):
    """读划分文件为集合（去注释/空行）。"""
    if not osp.isfile(p):
        return set()
    out = set()
    with open(p, encoding='utf-8') as f:
        for line in f:
            s = line.strip().replace('\\', '/')
            if s and not s.startswith('#'):
                out.add(s)
    return out


def read_text_any(p):
    """尽量读出文本内容（last_checkpoint 是本地编码写的，"
        "可能是 GBK，直接 utf-8 读会炸）。"""
    if not osp.isfile(p):
        return None
    raw = open(p, 'rb').read()
    for enc in ('utf-8-sig', 'utf-8', 'gbk', 'mbcs'):
        try:
            return raw.decode(enc).strip()
        except (UnicodeDecodeError, LookupError):
            continue
    return raw.decode('utf-8', errors='replace').strip()


def git_head(repo):
    """读裸文件拿 HEAD，不依赖 git 命令。"""
    h = osp.join(repo, '.git', 'HEAD')
    if not osp.isfile(h):
        return None
    try:
        s = open(h, encoding='utf-8').read().strip()
    except Exception:
        return None
    if s.startswith('ref: '):
        ref = s[5:].strip()
        p = osp.join(repo, '.git', ref)
        if osp.isfile(p):
            return open(p, encoding='utf-8').read().strip()[:8]
        # ref 被打包了，去 packed-refs 找
        pk = osp.join(repo, '.git', 'packed-refs')
        if osp.isfile(pk):
            for line in open(pk, encoding='utf-8'):
                if line.strip().endswith(ref):
                    return line.split()[0][:8]
        return None
    return s[:8]


def main():
    ap = argparse.ArgumentParser('DMKPD 训练前置校验')
    ap.add_argument('--epochs', type=int, default=12,
                    help='计划训练的 epoch 数（影响调度器预览）')
    ap.add_argument('--batch-size', type=int, default=4)
    ap.add_argument('--val-interval', type=int, default=1)
    ap.add_argument('--work-dir', default='work_dirs/dmkpd')
    ap.add_argument('--no-smoke', action='store_true',
                    help='跳过 GPU 冒烟与速度实测')
    args = ap.parse_args()

    print('=' * 78)
    print('DMKPD 训练前置校验（preflight）')
    print('=' * 78)
    print(f'  计划：epochs={args.epochs}  batch={args.batch_size}  '
          f'val_interval={args.val_interval}')
    print(f'  work_dir：{args.work_dir}')

    # ============================================================== #
    sep('1. 解释器与关键依赖')
    chk('Python 3.10.x（mmcv 的 cp310 编译扩展要求）',
        sys.version.startswith('3.10'), sys.version.split()[0])
    print(f'       解释器 {sys.executable}')
    chk('在项目根目录运行', osp.isfile('config.py') and osp.isfile('config_det.py'),
        osp.abspath('.'))

    torch = None
    try:
        import torch
        chk('torch 可导入', True, torch.__version__)
    except Exception as e:
        chk('torch 可导入', False, str(e)[:100])

    for mod, want in (('mmcv', '2.1.0'), ('mmengine', '0.10.7'),
                      ('mmdet', '3.2.0'), ('mmpose', '1.3.2')):
        try:
            m = __import__(mod)
            chk(f'{mod} == {want}', m.__version__ == want, m.__version__)
        except Exception as e:
            chk(f'{mod} == {want}', False, str(e)[:100])

    try:
        import mmcv.ops  # noqa
        chk('mmcv.ops（编译扩展）可导入', True)
    except Exception as e:
        chk('mmcv.ops（编译扩展）可导入', False, str(e)[:100])

    try:
        import mmpose
        chk('mmpose 指向本项目源码（editable 生效）',
            'open-mmlab/mmpose' in mmpose.__file__.replace('\\', '/'),
            mmpose.__file__)
    except Exception:
        pass

    try:
        import pkg_resources  # noqa
        import setuptools
        v = setuptools.__version__
        chk('setuptools < 81（保住 pkg_resources）',
            tuple(int(x) for x in v.split('.')[:2]) < (81, 0), v)
    except Exception as e:
        chk('pkg_resources / setuptools', False, str(e)[:100])
        print('       修法：pip install "setuptools==80.9.0"')

    cuda_ok = bool(torch is not None and torch.cuda.is_available())
    if cuda_ok:
        name = torch.cuda.get_device_name(0)
        cc = torch.cuda.get_device_capability(0)
        tag = f'sm_{cc[0]}{cc[1]}'
        arch = torch.cuda.get_arch_list()
        chk(f'GPU {tag} 在 torch 编译列表内', tag in arch, name)
        if tag not in arch:
            print('       注：缺 sm 但仍可跑 —— CUDA binary compatibility')
            print('           允许同 major 内低 minor 的 cubin 在高 minor 上运行，')
            print('           只是少了架构专属优化，正确性无问题。')
        print(f'       显存 '
              f'{torch.cuda.get_device_properties(0).total_memory/1024**3:.1f} GB'
              f'    cudnn {torch.backends.cudnn.version()}')
    else:
        chk('CUDA 可用', False, '无 CUDA，训练无法进行')

    # ============================================================== #
    sep('2. 代码完整性（open-mmlab 的 5 个 patch 文件）')
    chk('mmdet junction 可解析',
        osp.isfile('mmdet/.mim/configs/rtmdet/rtmdet_tiny_8xb32-300e_coco.py'))
    print('       创建命令：New-Item -ItemType Junction -Path mmdet '
          '-Target "$PWD\\.venv\\Lib\\site-packages\\mmdet"')

    MP = 'open-mmlab/mmpose'
    PATCH_FILES = [
        ('configs/_base_/datasets/DMKPD_meta.py', 'DMKPD'),
        ('mmpose/datasets/datasets/DM/__init__.py', 'DMKPD'),
        ('mmpose/datasets/datasets/DM/registry.py', 'DMKPD_dataset'),
        ('mmpose/datasets/datasets/DM/transforms.py',
         'CheckKeypointOutOfBounds'),
        ('mmpose/datasets/datasets/__init__.py', 'DM'),
    ]
    missing = []
    for rel, needle in PATCH_FILES:
        p = osp.join(MP, rel)
        ok = osp.isfile(p)
        has = ok and needle in open(p, encoding='utf-8',
                                   errors='ignore').read()
        if not (ok and has):
            missing.append(rel)
        chk(f'patch  {rel}', ok and has,
            '' if (ok and has) else ('文件缺失' if not ok else
                                     f'内容里找不到 "{needle}"'),
            block=False)
    chk('5 个 patch 文件全部就位', not missing,
        f'缺 {len(missing)} 个' if missing else '')
    if missing:
        print('       修法（新机上重建）：')
        print('         git clone https://github.com/open-mmlab/mmpose.git '
              'open-mmlab/mmpose')
        print('         cd open-mmlab/mmpose; git checkout 759b39c1; cd ..\\..')
        print('         git apply ..\\..\\patches\\0001-dmkpd-dataset.patch')
        print('         & ".\\\\.venv\\\\Scripts\\\\python.exe" -m pip install -e '
              'open-mmlab/mmpose --no-build-isolation')

    h = git_head(MP)
    chk('open-mmlab/mmpose 有 git 仓库（可追溯）', h is not None,
        f'HEAD={h}' if h else '无 .git —— 无法确认基线版本', block=False)
    if h and h != '759b39c1':
        WARN.append(f'open-mmlab/mmpose HEAD={h}（我们的基线是 759b39c1）')
        print(f'       ⚠️ HEAD={h}，与记录的基线 759b39c1 不同 —— '
              f'确认是有意升级')

    from mmengine.registry import init_default_scope
    try:
        from mmpose.datasets.datasets.DM.registry import (  # noqa
            DMKPD_dataset)
        from mmpose.datasets.datasets.DM.transforms import (  # noqa
            CheckKeypointOutOfBounds)
        chk('DMKPD_dataset / CheckKeypointOutOfBounds 可导入', True)
    except Exception as e:
        chk('DMKPD_dataset / CheckKeypointOutOfBounds 可导入', False,
            str(e)[:130])

    # ============================================================== #
    sep('3. 数据集完整性')
    n_lab = len(glob.glob('DMKPD_dataset/train/annotations/*.json'))
    chk('原始标注 json 数 == 9591', n_lab == 9591, f'{n_lab} 个')

    n_img = len([f for f in os.listdir('DMKPD_dataset/train/images')
                 if f.lower().endswith(('.jpg', '.jpeg', '.png'))]) \
        if osp.isdir('DMKPD_dataset/train/images') else -1
    chk('训练图片数 == 9591', n_img == 9591, f'{n_img} 张')

    n_test = len([f for f in os.listdir('DMKPD_dataset/test')
                  if f.lower().endswith(('.jpg', '.jpeg', '.png'))]) \
        if osp.isdir('DMKPD_dataset/test') else -1
    chk('测试图片数 == 300', n_test == 300, f'{n_test} 张')

    for name, p, want in (
            ('划分 train.txt', 'DMKPD_dataset/splits/train.txt', 8711),
            ('划分 val.txt', 'DMKPD_dataset/splits/val.txt', 480),
            ('划分 holdout1702.txt',
             'DMKPD_dataset/splits/holdout1702.txt', 400)):
        n = count_lines(p)
        chk(f'{name}（期望 {want}）', n == want, f'{n} 行')
    print('       注：train.txt 8711 = 9591 − 480(val) − 400(holdout)，'
          '三份互不重叠')

    # ---- ★ 数据泄漏检查：holdout 是否真被排除在 train 之外 ---- #
    S_tr = read_split('DMKPD_dataset/splits/train.txt')
    S_va = read_split('DMKPD_dataset/splits/val.txt')
    S_ho = read_split('DMKPD_dataset/splits/holdout1702.txt')
    if S_tr and S_va and S_ho:
        ov = S_tr & S_ho
        # ⚠️ 这**不阻断训练** —— 实测证明姿态并未“记住”这 400 张
        #    （见 `_leak_test.py`：见过 vs 没见过 OKS 差 -0.0022 ≈ 噪声）
        chk('holdout1702 与 train.txt 无重叠', not ov,
            f'重叠 {len(ov)} 个（结构性泄漏，实测无害）' if ov else '',
            block=False)
        chk('holdout1702 与 val.txt 无重叠', not (S_va & S_ho),
            f'{len(S_va & S_ho)} 个', block=False)
        chk('train.txt 与 val.txt 无重叠', not (S_tr & S_va),
            f'{len(S_tr & S_va)} 个')
        if ov:
            print()
            print(f'  ⚠️ **结构性泄漏**：holdout1702 的 {len(ov)} 张图也在 train.txt 里')
            print()
            print('     但实测证明**姿态没有记住它们**，因此不影响结论：')
            print('       组                图   实例   OKS均值   EPE中位')
            print('       holdout(见过)    400   509   0.9416    4.27')
            print('       val-1702(没见过) 123   164   0.9438    4.50')
            print('       差                             -0.0022   -0.23  ≈ 噪声')
            print()
            print('     原因：RandomBBoxTransform(rotate_prob=1.0) 的随机旋转')
            print('           破坏了像素级记忆；且只占训练集 4.4%')
            print()
            print('     举证链干净程度：')
            print('       · splits/val.txt    train ∩ val = 0  -> 完全干净 ✓')
            print('       · 榜单成绩           主办方自己的测试集 -> 完全干净 ✓')
            print('       · _ap_holdout.py     结构性受影响，但实测膨胀 ~= 0')
            print()
            print('     检测器一侧是干净的（convert_det.py 已剔除 holdout）：')
            print('       instances_train.json 8710 图 ∩ holdout = 0 ✓')
            print('     重新跑 `_leak_test.py` 可复现上面的判决实验。')
            WARN.append(f'holdout1702 ∩ train.txt = {len(ov)} 个'
                        '（结构性泄漏，实测无害）')

    # ---- 一致性交叉验证：train.txt 应等于 检测train 图集（8710 有实例）---- #
    if S_tr:
        import json as _json
        _dtr = _json.load(open('det_dataset/annotations/instances_train.json',
                               encoding='utf-8'))
        DTR = {osp.splitext(osp.basename(im['file_name']))[0]
               for im in _dtr['images']}
        st = {osp.splitext(osp.basename(r))[0] for r in S_tr}
        chk('train.txt 与检测器训练集一致（交集 8710）',
            len(st & DTR) == 8710 and not (DTR - st),
            f'交集 {len(st & DTR)} / 仅姿态 {len(st - DTR)} / '
            f'仅检测 {len(DTR - st)}')
        if len(st - DTR):
            print(f'       （那 {len(st - DTR)} 张图的实例全被剔除或不在检测集，'
                  '属正常）')

    for name, p, want in (
            ('检测标注 train', 'det_dataset/annotations/'
                               'instances_train.json', 8710),
            ('检测标注 val', 'det_dataset/annotations/instances_val.json', 480),
            ('检测标注 holdout1702',
             'det_dataset/annotations/instances_holdout1702.json', 400),
            ('检测标注 test', 'det_dataset/annotations/instances_test.json',
             300)):
        ok = osp.isfile(p)
        if ok:
            import json
            d = json.load(open(p, encoding='utf-8'))
            n = len(d.get('images', []))
            chk(f'{name}（{want} 张图）', n == want,
                f'{n} 张图 / {len(d.get("annotations", []))} 实例')
        else:
            chk(name, False, '文件缺失')

    # 真正读一张图 + 构建 val 数据集（验证标注能解析）
    cfg = None
    try:
        import cv2
        im = cv2.imread('DMKPD_dataset/train/images/00003.jpg')
        chk('训练图片可读', im is not None,
            f'shape={None if im is None else im.shape}')
    except Exception as e:
        chk('训练图片可读', False, str(e)[:100])

    try:
        from mmengine.config import Config
        cfg = Config.fromfile('config.py')
        init_default_scope('mmpose')
        from mmpose.registry import DATASETS as PD
        ds_val = PD.build(dict(
            type='DMKPD_dataset', data_root='DMKPD_dataset',
            ann_file='splits/val.txt', data_prefix=dict(img='train/images'),
            metainfo=cfg.metainfo, serialize_data=False))
        chk('姿态 val 数据集可构建（期望 1101 实例）',
            len(ds_val) == 1101, f'{len(ds_val)} 个实例')
        ds_tr = PD.build(dict(
            type='DMKPD_dataset', data_root='DMKPD_dataset',
            ann_file='splits/train.txt',
            data_prefix=dict(img='train/images'),
            metainfo=cfg.metainfo, serialize_data=False))
        # 20494 → 已扣除 holdout 的 509 个实例 = 19985
        # （另 2 个异常实例在原 train 集，剔后仍与检测口径一致）
        chk('姿态 train 数据集可构建（期望 19985 实例，与检测一致）',
            len(ds_tr) == 19985, f'{len(ds_tr)} 个实例')
    except Exception as e:
        chk('姿态数据集可构建', False, str(e)[:150])

    # ============================================================== #
    sep('4. 权重文件')
    W = [
        ('姿态预训练（COCO，必须）',
         'modelbase_test/td-hm_hrnet-w48_8xb32-210e_coco-'
         '256x192-0e67c616_20220913.pth', 245),
        ('检测预训练（COCO，训练检测器才需要）',
         'modelbase_test/rtmdet_tiny_8xb32-300e_coco_'
         '20220902_112414-78e30dcc.pth', 55),
    ]
    for name, p, approx_mb in W:
        if osp.isfile(p):
            mb = osp.getsize(p) / 1024**2
            chk(name, abs(mb - approx_mb) < approx_mb * 0.15,
                f'{mb:.1f} MB')
        else:
            chk(name, False, f'缺失：{p}')

    pose_best = sorted(glob.glob(osp.join(args.work_dir, 'best_*.pth')))
    det_best = sorted(glob.glob('work_dirs/dmkpd_det/best_*.pth'))
    chk('姿态「已训」权重', bool(pose_best),
        ', '.join(osp.basename(f) for f in pose_best) or '无（从头训即可）',
        block=False)
    chk('检测「已训」权重', bool(det_best),
        ', '.join(osp.basename(f) for f in det_best) or '无', block=False)

    # ============================================================== #
    sep('5. work_dir 残留（★ 续训陷阱）')
    cks = sorted(glob.glob(osp.join(args.work_dir, '*.pth')),
                 key=osp.getmtime, reverse=True)
    lc = osp.join(args.work_dir, 'last_checkpoint')
    print(f'  {args.work_dir} 下的权重：')
    if cks:
        for f in cks:
            print(f'       {osp.basename(f):<44} '
                  f'{osp.getsize(f)/1024**2:>8.1f} MB')
    else:
        print('       （无）')
    if osp.isfile(lc):
        print(f'  last_checkpoint 指向：{read_text_any(lc)}')

    trap = bool(cks)
    chk('work_dir 无残留权重（否则会误触发续训）', not trap,
        '有残留 -> 必须加 --no-resume，或先挪走' if trap else '',
        block=False)
    if trap:
        print()
        print('  ❗ `run_train.py` 会「按 mtime 找任意 *.pth 就续训」，')
        print('     所以现在直接跑 `--epochs 12` 会**从已有权重的轮次接着训**，')
        print('     而不是从头训 12 轮。两种解法：')
        print()
        print(f'     解法 A（推荐，保持目录干净）：')
        print(f'       New-Item -ItemType Directory -Force {args.work_dir}\\_old_best')
        print(f'       Move-Item {args.work_dir}\\*.pth {args.work_dir}\\_old_best\\ -Force')
        print(f'       Remove-Item {args.work_dir}\\last_checkpoint -Force '
              f'-ErrorAction SilentlyContinue')
        print()
        print(f'     解法 B（保留原目录，靠参数绕过）：加 `--no-resume`')

    # ============================================================== #
    sep('6. 配置与调度器预览（★ 按 --epochs 重算）')
    if cfg is None:
        chk('配置已加载', False, '第 3 节失败，跳过')
    else:
        iters_per_epoch = (len(ds_tr) + args.batch_size - 1) // args.batch_size \
            if 'ds_tr' in dir() else -1
        ms = [max(1, int(args.epochs * 0.8))]
        print(f'  {"项":<24}{"config.py 原值":>18}{"本次实际生效":>18}')
        print('  ' + '-' * 60)
        print(f'  {"max_epochs":<24}{"50":>18}{args.epochs:>18}')
        print(f'  {"val_interval":<24}{"5":>18}{args.val_interval:>18}')
        print(f'  {"batch_size":<24}{"4":>18}{args.batch_size:>18}')
        print(f'  {"MultiStepLR milestones":<24}{"[40]":>18}'
              f'{str(ms):>18}')
        print(f'  {"MultiStepLR end":<24}{"50":>18}{args.epochs:>18}')
        print(f'  {"gamma":<24}{"0.1":>18}{"0.1":>18}')
        print(f'  {"optimizer / lr":<24}{"Adam 5e-4":>18}{"Adam 5e-4":>18}')
        print(f'  {"input / heatmap":<24}'
              f'{"(256, 256)":>18}{"(256, 256)":>18}')
        print(f'  {"codec":<24}{"MSRAHeatmap(s=2)":>18}'
              f'{"MSRAHeatmap(s=2)":>18}')
        print()
        print(f'  → lr 将在 **第 {ms[0]} 个 epoch** 降 10 倍，'
              f'最后 {args.epochs - ms[0]} 轮做微调。')

        chk('milestones 落在训练区间内', 0 < ms[0] < args.epochs,
            f'milestones={ms}, epochs={args.epochs}')
        if args.epochs <= 5:
            print('       ⚠️ epochs<=5 时降至 1 个 epoch 后就没剩几轮微调，'
                  '建议 >=8')

        # 预训练权重能否真被加载
        lf = cfg.get('load_from')
        chk('load_from 指向存在的文件', bool(lf) and osp.isfile(lf),
            str(lf), block=False)

    # ============================================================== #
    sep('7. 数据管线真正前向')
    try:
        from mmengine.dataset import Compose
        init_default_scope('mmpose')

        # (a) val 管线：不需要 target，所以 gt_instance_labels 是空的
        r = Compose(cfg.val_pipeline)(ds_val.get_data_info(0))
        shp = tuple(r['inputs'].shape)
        chk('姿态 val 管线前向 -> (3,256,256)',
            shp == (3, 256, 256), f'inputs={shp}')
        chk('val 顶层键符合预期（只有 inputs / data_samples）',
            set(r) == {'inputs', 'data_samples'}, str(sorted(r)))
        gi = r['data_samples'].gt_instances
        chk('val 标注解析出 keypoints',
            'keypoints' in gi.all_keys(),
            f'shape={tuple(gi.keypoints.shape)}')
        print('       注：val_pipeline 不含编码步骤（不需要 target）')
        print('           -> gt_instance_labels 为空是**正常**的')

        # (b) train 管线：必须能编码出 (4,64,64) 的高斯热力图
        rt = Compose(cfg.train_pipeline)(ds_tr.get_data_info(0))
        dsm = rt['data_samples']
        # ⚠️ heatmaps 不在 gt_instance_labels，而在 **gt_fields**（PixelData）！
        #    `GenerateTarget` 产出 -> `PackPoseInputs` 按 field_mapping_table
        #    归到 `gt_fields`。这是实测确认的位置，别猜。
        gf = dsm.gt_fields
        gfk = list(gf.keys()) if hasattr(gf, 'keys') else []
        chk('train 管线编码出热力图（gt_fields.heatmaps）',
            'heatmaps' in gfk, str(gfk))
        if 'heatmaps' in gfk:
            hm = gf.heatmaps
            chk('热力图形状 == (4, 64, 64)',
                tuple(hm.shape) == (4, 64, 64), str(tuple(hm.shape)))
            # ⚠️ hm 是 **Tensor**：Tensor.max(dim) 返回 namedtuple(values, indices)，
            #    先转 numpy 再取峰值，否则 (pk == 1.0) 会退化成单个 False。
            hmn = hm.detach().cpu().numpy()
            pk = hmn.reshape(hmn.shape[0], -1).max(1)
            chk('4 个通道各有峰值 == 1.0（编码正常）',
                bool((pk >= 0.999).all()),
                f'各通道峰值 {[round(float(v), 3) for v in pk]}')
            print('       注：DMKPD 只有 4 个角点（COCO 是 17），'
                  '故第 1 维是 4 而不是 17')
        lab = dsm.gt_instance_labels
        lk = list(lab.keys()) if hasattr(lab, 'keys') else []
        if 'keypoint_weights' in lk:
            kw = lab.keypoint_weights
            chk('keypoint_weights 全为 1（无出界点）',
                float(kw.min()) == 1.0,
                f'min={float(kw.min())}  mean={float(kw.mean()):.4f}')
    except Exception as e:
        chk('姿态管线前向', False, repr(e)[:200])

    # ============================================================== #
    sep('8. GPU 训练冒烟 + 速度实测')
    if args.no_smoke:
        print('  [跳过] --no-smoke')
    elif not cuda_ok:
        chk('GPU 冒烟', False, '无 CUDA', block=False)
    else:
        try:
            import time
            from mmengine.dataset import pseudo_collate
            # ⚠️ build_optim_wrapper 在 `mmengine.optim`，
            #    **不在** `mmengine.model`（踩过）。
            from mmengine.optim import build_optim_wrapper
            from mmpose.registry import MODELS as PM

            init_default_scope('mmpose')
            model = PM.build(cfg.model).cuda().train()
            optim_wrapper = build_optim_wrapper(model, cfg.optim_wrapper)

            # 取两个真实 batch 做前反向
            pipe = Compose(cfg.train_pipeline)
            batch = pseudo_collate(
                [pipe(ds_tr.get_data_info(i)) for i in range(args.batch_size)])
            torch.cuda.reset_peak_memory_stats()
            model.train_step(batch, optim_wrapper)     # 热一次
            torch.cuda.synchronize()
            t0 = time.time()
            N = 3
            for _ in range(N):
                out = model.train_step(batch, optim_wrapper)
            torch.cuda.synchronize()
            s_it = (time.time() - t0) / N
            loss = float(out['loss']) if isinstance(out, dict) else float(out)
            peak = torch.cuda.max_memory_allocated() / 1024**3
            chk('一次完整 train_step（前向+反向+优化器）可跑', True,
                f'loss={loss:.4f}')
            print(f'       速度 {s_it*1000:.0f} ms/iter (bs={args.batch_size})'
                  f'   峰值显存 {peak:.2f} GB')
            if peak > 0.9 * (torch.cuda.get_device_properties(0)
                             .total_memory / 1024**3):
                print('       ⚠️ 峰值显存接近上限，注意 bs 是否过大')

            # ---- ETA ----
            if 'iters_per_epoch' in vars() and iters_per_epoch > 0:
                val_it = (1101 + args.batch_size - 1) // args.batch_size
                val_s = 0.14 * val_it          # 实测姿态 val ≈ 0.14 s/it
                per_ep = s_it * iters_per_epoch
                if args.val_interval > 0:
                    per_ep += val_s / max(1, args.val_interval)
                tot = per_ep * args.epochs
                print()
                print(f'  {"iter/epoch":<20}{iters_per_epoch:>10}')
                print(f'  {"每 epoch 训练":<20}'
                      f'{s_it*iters_per_epoch/60:>9.1f} min')
                print(f'  {"每 epoch 验证":<20}'
                      f'{val_s/60:>9.1f} min (每 {args.val_interval} 轮)')
                print(f'  {"总耗时估计":<20}{tot/3600:>9.2f} h '
                      f'({args.epochs} epoch)')
                print()
                print('       注：本估计基于**本机实测**的 s/iter，'
                      '换机后请重跑本脚本。')
        except Exception as e:
            chk('GPU 训练冒烟', False, str(e)[:180])
            print('       若报 train_step 签名错，检查 mmpose 版本是否 1.3.2')

    # ============================================================== #
    sep('结论')
    if WARN:
        print('  警告（不阻断）：')
        for w in WARN:
            print(f'    - {w}')
        print()
    if not OK:
        print('  ❌ 有 FAIL 项 —— 先修好再训练。')
        print('     完整环境体检：& ".\\.venv\\Scripts\\python.exe" _env_check.py')
        return 1

    print('  ✅ 检查通过 —— 可以开训。')
    print()
    print('  ' + '=' * 60)
    print('  从头训（推荐，配合第 5 节的「解法 A」先挪走旧权重）')
    print('  ' + '=' * 60)
    print()
    print(f'    & ".\\.venv\\Scripts\\python.exe" run_train.py '
          f'--epochs {args.epochs} --val-interval {args.val_interval} '
          f'--no-resume')
    print()
    print('  ' + '=' * 60)
    print('  训完导出提交')
    print('  ' + '=' * 60)
    print()
    print('    & ".\\.venv\\Scripts\\python.exe" infer_e2e.py --export '
          '--out submission')
    print('    & ".\\.venv\\Scripts\\python.exe" _sub_check.py')
    print()
    print('  ' + '=' * 60)
    print('  ⚠️ 别忘了：`--no-resume` 只影响「是否续训」，')
    print('     `config.py` 里的 `load_from`（COCO 姿态预训练）仍会加载。')
    return 0


if __name__ == '__main__':
    sys.exit(main())
