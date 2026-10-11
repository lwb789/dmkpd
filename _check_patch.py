# -*- coding: utf-8 -*-
"""验证 patches/0001-dmkpd-dataset.patch 能在干净仓库上完整还原 mmpose 改动。

为什么需要这个验证：
    迁移到新机时，mmpose 有两条获取路径：
      A. 网盘直接拷 `open-mmlab/`（推荐，省事）
      B. 重新 clone + apply patch（备用）
    路径 B 必须被**实测证明可行**，否则新机上会卡住。

本脚本做的是「干净复现」：
    1. 从 mmpose 的 .git clone 一个干净副本（HEAD 不含任何 DM 文件）
    2. apply 我们的 patch
    3. 逐文件比对本机实际文件（行数 + 内容哈希）
"""
import hashlib
import os
import os.path as osp
import shutil
import subprocess
import sys
import tempfile

ROOT = osp.abspath('.')
SRC = osp.join(ROOT, 'open-mmlab', 'mmpose')
PATCH = osp.join(ROOT, 'patches', '0001-dmkpd-dataset.patch')

FILES = [
    'configs/_base_/datasets/DMKPD_meta.py',
    'mmpose/datasets/datasets/DM/__init__.py',
    'mmpose/datasets/datasets/DM/registry.py',
    'mmpose/datasets/datasets/DM/transforms.py',
    'mmpose/datasets/datasets/__init__.py',
]

# 这 4 个是**新增**文件（官方 mmpose 没有），干净仓库里应不存在
NEW_FILES = FILES[:4]
# 这 1 个是**已有文件**，我们只改了 1 行（加 `from .DM import *`）
MODIFIED_FILES = FILES[4:]


def git(*args, cwd=None):
    return subprocess.run(['git'] + list(args), cwd=cwd,
                          capture_output=True, text=True)


def sha(path):
    if not osp.isfile(path):
        return None
    with open(path, 'rb') as f:
        return hashlib.sha256(f.read()).hexdigest()[:16]


def main():
    print('=' * 78)
    print('验证 mmpose patch 的可还原性')
    print('=' * 78)
    print(f'  源仓库   {SRC}')
    print(f'  patch    {PATCH}  ({osp.getsize(PATCH)} 字节)')

    # ---- 0. patch 编码检查（git apply 要求 LF，且不能有 BOM）---- #
    with open(PATCH, 'rb') as f:
        raw = f.read()
    has_bom = raw.startswith(b'\xef\xbb\xbf')
    n_crlf = raw.count(b'\r\n')
    n_lf = raw.count(b'\n') - n_crlf
    print()
    print('  [编码检查]')
    print(f'    BOM        {"有(错!)" if has_bom else "无(正确)"}')
    print(f'    CRLF 行尾  {n_crlf}  {"（错! git apply 会报 corrupt patch）" if n_crlf else "（正确）"}')
    print(f'    LF 行尾    {n_lf}')
    if has_bom or n_crlf:
        print('    -> 必须用 git 原生 --output 生成，不能用 PowerShell 重定向')
        return 1

    # ---- 1. 干净克隆 ---- #
    tmp = tempfile.mkdtemp(prefix='mmpose_patch_verify_')
    dst = osp.join(tmp, 'mmpose')
    print()
    print(f'  [干净克隆] -> {dst}')
    r = git('clone', '-q', '--no-hardlinks', '--depth', '1', SRC, dst)
    # --depth 1 默认拉默认分支；若 patch 基线是特定 commit，要显式 checkout
    head = git('rev-parse', '--short', 'HEAD', cwd=SRC).stdout.strip()
    print(f'    克隆 exit={r.returncode}')
    got = git('rev-parse', '--short', 'HEAD', cwd=dst).stdout.strip()
    print(f'    源仓库 HEAD {head}  克隆后 HEAD {got}')

    # 确认干净：新增文件都不存在；已有文件存在但**不含**我们的改动
    print()
    print('  [克隆后应保持官方原状]')
    clean_ok = True
    for f in NEW_FILES:
        exists = osp.isfile(osp.join(dst, f))
        clean_ok &= not exists
        print(f'    新增文件 {"存在(异常!)" if exists else "不存在(符合预期)"}  {f}')
    for f in MODIFIED_FILES:
        p = osp.join(dst, f)
        exists = osp.isfile(p)
        has_our_change = exists and ('from .DM import' in
                                     open(p, encoding='utf-8').read())
        clean_ok &= exists and not has_our_change
        print(f'    已有文件 存在={exists}  含我们的改动={has_our_change}  {f}')
    if not clean_ok:
        print('    -> 克隆不干净，无法继续验证')
        shutil.rmtree(tmp, ignore_errors=True)
        return 1

    # ---- 2. apply patch ---- #
    print()
    print('  [apply patch]')
    r = git('apply', '--verbose', PATCH, cwd=dst)
    print(f'    exit={r.returncode}')
    if r.returncode != 0:
        print('    stderr:', r.stderr.strip()[:500])
        shutil.rmtree(tmp, ignore_errors=True)
        return 1

    # ---- 3. 逐文件比对 ---- #
    print()
    print('  [逐文件比对：干净+patch  vs  本机实际]')
    print(f'    {"文件":<50}{"行数":>7}{"sha256":>10}  判定')
    all_ok = True
    for f in FILES:
        pa = osp.join(dst, f)
        pb = osp.join(SRC, f)
        n_a = len(open(pa, encoding='utf-8').readlines()) \
            if osp.isfile(pa) else 0
        n_b = len(open(pb, encoding='utf-8').readlines()) \
            if osp.isfile(pb) else 0
        s_a, s_b = sha(pa), sha(pb)
        same = (s_a is not None) and (s_a == s_b)
        all_ok &= same
        flag = 'OK' if same else 'MISMATCH'
        print(f'    {f:<50}{n_a:>4}/{n_b:<4} {str(s_a):>10}  {flag}')

    # ---- 4. 反向验证：apply 后再 reverse 应回到干净 ---- #
    print()
    print('  [反向验证] apply 后 --reverse --check 应通过')
    r2 = git('apply', '--reverse', '--check', PATCH, cwd=dst)
    print(f'    exit={r2.returncode}  '
          f'{"（patch 与文件内容完全吻合）" if r2.returncode == 0 else "（不吻合!）"}')

    shutil.rmtree(tmp, ignore_errors=True)

    print()
    print('=' * 78)
    if all_ok and r2.returncode == 0:
        print('  结论：patch 可完整还原全部 5 个文件的改动 —— 路径 B 可行')
        return 0
    print('  结论：验证未通过，请检查 patch')
    return 1


if __name__ == '__main__':
    sys.exit(main())
