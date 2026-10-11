# -*- coding: utf-8 -*-
"""打迁移包：把新机需要的文件复制到 `_transfer/`，便于拷到移动硬盘。

===========================================================================
新机的定位：**只跑训练**。所以它只需要：
    ① 训练数据（train 的 images + annotations）
    ② 预训练权重（2 个 .pth）
    ③ mmpose 源码（含我们的自定义改动）
    ④ 代码（走 git，不在这里打）

**不需要**：测试集、work_dirs、submission、vis_out2、.venv
===========================================================================
"""
import json
import os
import os.path as osp
import shutil
import sys

DST = '_transfer'

# (源路径, 是否必需, 说明)
ITEMS = [
    ('DMKPD_dataset/train/images', True, '9591 张训练 jpg（0.67 GB）'),
    ('DMKPD_dataset/train/annotations', True,
     '9591 个 LabelMe json（0.39 GB）'),
    ('modelbase_test', True, '2 个预训练权重（0.30 GB）'),
    ('open-mmlab', True, 'mmpose 源码（含 DMKPD 自定义改动）'),
    # 可选：本机推理要用，但新机不需要
    ('DMKPD_dataset/test', False, '测试集（新机不需要，仅本机推理用）'),
]

# 只搬这两个 .pth，避免 modelbase_test 里有其它大文件
WEIGHTS_WHITELIST = (
    'td-hm_hrnet-w48_8xb32-210e_coco-256x192-0e67c616_20220913.pth',
    'rtmdet_tiny_8xb32-300e_coco_20220902_112414-78e30dcc.pth',
)


def dsize(p):
    if not osp.exists(p):
        return 0, 0
    total = n = 0
    for root, _, files in os.walk(p):
        for f in files:
            try:
                total += osp.getsize(osp.join(root, f))
                n += 1
            except OSError:
                pass
    return total, n


def main():
    print('=' * 78)
    print('打迁移包（新机 = 只跑训练）')
    print('=' * 78)
    root = osp.abspath('.')

    # ---------- 0. 前端检查：代码是否已提交 ---------- #
    import subprocess
    r = subprocess.run(['git', 'status', '--porcelain'], capture_output=True,
                       text=True)
    dirty = [l for l in r.stdout.splitlines() if l.strip()]
    if dirty:
        print(f'  ⚠️ 有 {len(dirty)} 个未提交的改动（代码不会进迁移包，'
              f'建议先 commit + push）:')
        for l in dirty[:8]:
            print('     ', l)
        print()
    else:
        print('  ✅ 工作区干净（代码可通过 git 在新机获取）')

    r2 = subprocess.run(['git', 'status', '-sb'], capture_output=True,
                        text=True)
    first = r2.stdout.splitlines()[0] if r2.stdout else ''
    if 'ahead' in first:
        print(f'  ⚠️ 有未推送的 commit: {first}')
        print('     新机 git clone 拿不到它们，请先 push')
    print()

    # ---------- 1. 检查源文件 ---------- #
    print('-' * 78)
    print('1. 检查源文件')
    print('-' * 78)
    ok = True
    for src, required, note in ITEMS:
        exists = osp.exists(src)
        if required:
            ok &= exists
        tag = 'OK  ' if exists else ('MISS' if required else 'SKIP')
        size, n = dsize(src)
        print(f'  [{tag}] {src:<36} {size/1e6:>9.1f} MB  {n:>6} 文件')
        print(f'         {note}')
    if not ok:
        print()
        print('  ❌ 缺少必需文件，请先补齐再打包')
        return 1

    # ---------- 2. 复制 ---------- #
    print()
    print('-' * 78)
    print('2. 复制到 ' + DST)
    print('-' * 78)
    if osp.exists(DST):
        print(f'  清空已有的 {DST} ...')
        shutil.rmtree(DST)
    os.makedirs(DST, exist_ok=True)

    for src, required, note in ITEMS:
        if not osp.exists(src):
            continue
        dst = osp.join(DST, src)
        os.makedirs(osp.dirname(dst), exist_ok=True)
        if src == 'modelbase_test':
            # 只拷白名单里的权重，跳过其它大文件
            os.makedirs(dst, exist_ok=True)
            for f in os.listdir(src):
                if f in WEIGHTS_WHITELIST:
                    shutil.copy2(osp.join(src, f), osp.join(dst, f))
                    print(f'  {src}/{f}')
                else:
                    print(f'  {src}/{f}  (跳过，不在白名单)')
        else:
            shutil.copytree(src, dst, dirs_exist_ok=True)
            size, n = dsize(dst)
            print(f'  {src:<40} {size/1e6:>9.1f} MB  {n:>6} 文件')

    # ---------- 3. 汇总 ---------- #
    print()
    print('-' * 78)
    print('3. 汇总')
    print('-' * 78)
    total, n = dsize(DST)
    print(f'  {DST}/  合计 {total/1e9:.2f} GB  {n} 文件')
    print()
    print('  目录结构:')
    for root_dir, dirs, files in os.walk(DST):
        depth = root_dir.count(os.sep) - DST.count(os.sep)
        if depth > 2:
            dirs[:] = []
            continue
        indent = '    ' + '  ' * depth
        print(f'{indent}{osp.basename(root_dir)}/  ({len(files)} 文件)')

    # ---------- 4. 下一步提示 ---------- #
    print()
    print('=' * 78)
    print('下一步')
    print('=' * 78)
    print('  1) 拷到移动硬盘:')
    print(f'     Copy-Item {DST}\\* "E:\\DMKPD_transfer\\" -Recurse -Force')
    print()
    print('  2) 新机上:')
    print('     git clone https://github.com/lwb789/dmkpd.git D:\\DMKPD')
    print(f'     Copy-Item "E:\\DMKPD_transfer\\*" "D:\\DMKPD\\" -Recurse -Force')
    print()
    print('  3) 详细流程见 NEW_MACHINE_SETUP.md')
    print()
    print('  ⚠️ 别忘了：DMKPD_dataset/splits 和 det_dataset 走 git，')
    print('     不在这个包里（clone 后自动就有）')

    # 写个清单，方便对付
    manifest = []
    for root_dir, dirs, files in os.walk(DST):
        for f in files:
            p = osp.join(root_dir, f)
            manifest.append(dict(
                path=osp.relpath(p, DST).replace('\\', '/'),
                size=osp.getsize(p)))
    json.dump(dict(total=total, count=n, files=manifest),
              open('_transfer_manifest.json', 'w', encoding='utf-8'), indent=1)
    print(f'  清单 -> _transfer_manifest.json（{len(manifest)} 条）')
    return 0


if __name__ == '__main__':
    sys.exit(main())
