"""决定性验证：GPU 监控线程里的 nvidia-smi 调用是否会拖慢 CUDA 计算？

背景：对比两次实测
  tools/train.py（无 GPU 监控）  -> 0.421 s/it
  run_train.py（有 GPU 监控）    -> 1.006 s/it
两者数据/模型/参数完全相同，唯一差别就是 GPU 监控线程
（它每 2 秒 fork 一个 nvidia-smi 进程）。

Windows 上启动 nvidia-smi 进程开销大（约 100-300ms），
且可能短暂占用 GPU 驱动 -> 阻塞 CUDA 调用。
本脚本用同一段 GPU 计算，分别在"无监控"和"有监控"下计时。
"""
import subprocess
import threading
import time

import torch
from mmengine.config import Config
from mmengine.registry import init_default_scope

init_default_scope('mmpose')

from mmpose.registry import MODELS  # noqa: E402

OUT = []


def L(s=''):
    print(s)
    OUT.append(str(s))


cfg = Config.fromfile('config.py')
model = MODELS.build(cfg.model).cuda()
opt = torch.optim.Adam(model.parameters(), lr=1e-4)
crit = MODELS.build(dict(type='KeypointMSELoss', use_target_weight=True))

BS = 4
x = torch.randn(BS, 3, 256, 256, device='cuda')
hm = torch.rand(BS, 4, 64, 64, device='cuda')
tw = torch.ones(BS, 4, device='cuda')


def step():
    feats = model.extract_feat(x)
    pred = model.head.forward(feats)
    loss = crit(pred, hm, tw)
    opt.zero_grad(); loss.backward(); opt.step()


def bench(n=15, label=''):
    for _ in range(3):
        step()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n):
        step()
    torch.cuda.synchronize()
    ms = (time.perf_counter() - t0) / n * 1000
    L(f'  {label:<40} {ms:>8.1f} ms/iter')
    return ms


L('=' * 76)
L('0) nvidia-smi 单次调用耗时（Windows）')
L('=' * 76)
subprocess.run(['nvidia-smi', '-L'], capture_output=True)
ts = []
for _ in range(5):
    t0 = time.perf_counter()
    subprocess.run(
        ['nvidia-smi',
         '--query-gpu=utilization.gpu,temperature.gpu',
         '--format=csv,noheader,nounits'],
        capture_output=True, text=True)
    ts.append((time.perf_counter() - t0) * 1000)
L(f'  单次 nvidia-smi 调用 = {sum(ts)/len(ts):.0f} ms '
  f'(范围 {min(ts):.0f}~{max(ts):.0f} ms)')
L()

L('=' * 76)
L('1) 纯 GPU 计算，无任何后台线程')
L('=' * 76)
t_none = bench(15, '无监控')
L()

L('=' * 76)
L('2) 同样计算 + 后台每 2s 调一次 nvidia-smi（run_train.py 的行为）')
L('=' * 76)
stop = False
count = [0]


def mon(interval):
    while not stop:
        subprocess.run(
            ['nvidia-smi',
             '--query-gpu=utilization.gpu,utilization.memory,temperature.gpu,'
             'power.draw,clocks.sm,memory.used',
             '--format=csv,noheader,nounits'],
            capture_output=True, text=True)
        count[0] += 1
        time.sleep(interval)


th = threading.Thread(target=mon, args=(2.0,), daemon=True)
th.start()
t_mon = bench(15, '有监控（每 2.0s 采样）')
stop = True
th.join(timeout=3)
L(f'  监控期间采样 {count[0]} 次')
L()

L('=' * 76)
L('3) 放大验证：每 0.2s 采样一次')
L('=' * 76)
stop2 = False


def mon2():
    while not stop2:
        subprocess.run(
            ['nvidia-smi',
             '--query-gpu=utilization.gpu,temperature.gpu',
             '--format=csv,noheader,nounits'],
            capture_output=True, text=True)
        time.sleep(0.2)


th2 = threading.Thread(target=mon2, daemon=True)
th2.start()
t_fast = bench(15, '有监控（每 0.2s 采样）')
stop2 = True
th2.join(timeout=3)
L()

L('=' * 76)
L('结论')
L('=' * 76)
L(f'  无监控            = {t_none:7.1f} ms/iter')
L(f'  有监控(2s 间隔)   = {t_mon:7.1f} ms/iter  ({t_mon/t_none:.2f}x)')
L(f'  有监控(0.2s 间隔) = {t_fast:7.1f} ms/iter  ({t_fast/t_none:.2f}x)')
L()
if t_mon > t_none * 1.2:
    L('  => nvidia-smi 后台调用确实拖慢 CUDA 计算')
    L('     对策：去掉 GPU 监控，或大幅提高采样间隔')
else:
    L('  => nvidia-smi 后台调用影响可忽略，瓶颈另有原因')

with open('_prof2_out.txt', 'w', encoding='utf-8') as f:
    f.write('\n'.join(OUT))
