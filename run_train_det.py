# -*- coding: utf-8 -*-
"""DMKPD 检测器训练启动器（RTMDet-tiny）。

功能与 run_train.py（姿态）对齐：
  - 后台线程采样 nvidia-smi（利用率/温度/功耗/显存）
  - 进度条 + **滑动窗口** ETA（不用累计平均，见 run_train.py 的踩坑记录）
  - 自动续训（从 work_dir/last_checkpoint.pth）
  - 可调 epochs / batch-size / val-interval

用法：
    python run_train_det.py                      # 默认 60 epoch
    python run_train_det.py --epochs 30
    python run_train_det.py --smoke              # 100 iter 冒烟
"""
import argparse
import os
import os.path as osp
import subprocess
import sys
import threading
import time
from collections import deque

import torch

WORK_DIR = 'work_dirs/dmkpd_det'


# --------------------------------------------------------------------- #
# GPU 监控
# --------------------------------------------------------------------- #
class GpuMonitor:
    """后台线程采样 nvidia-smi。

    注意：实测过「每 2s 调用 nvidia-smi」对训练速度 **零影响**
    （0.2s 间隔也是 1.00x），所以放心开着。
    """

    FIELDS = ['utilization.gpu', 'utilization.memory', 'temperature.gpu',
              'power.draw', 'clocks.sm', 'memory.used']

    def __init__(self, interval=2.0):
        self.interval = interval
        self.samples = []
        self._stop = threading.Event()
        self._th = None

    def _loop(self):
        while not self._stop.is_set():
            try:
                out = subprocess.run(
                    ['nvidia-smi', f'--query-gpu={",".join(self.FIELDS)}',
                     '--format=csv,noheader,nounits'],
                    capture_output=True, text=True, timeout=5)
                if out.returncode == 0:
                    vals = [v.strip() for v in out.stdout.strip().split(',')]
                    self.samples.append([float(v) for v in vals])
            except Exception:
                pass
            self._stop.wait(self.interval)

    def start(self):
        self._th = threading.Thread(target=self._loop, daemon=True)
        self._th.start()

    def stop(self):
        self._stop.set()
        if self._th:
            self._th.join(timeout=3)

    def summary(self):
        if not self.samples:
            return None
        import numpy as np
        a = np.array(self.samples)
        return dict(
            n=len(a),
            util_gpu=(a[:, 0].mean(), a[:, 0].max()),
            util_mem=(a[:, 1].mean(), a[:, 1].max()),
            temp=(a[:, 2].mean(), a[:, 2].max()),
            power=(a[:, 3].mean(), a[:, 3].max()),
            clock=(a[:, 4].mean(), a[:, 4].max()),
            mem=(a[:, 5].mean(), a[:, 5].max()))


# --------------------------------------------------------------------- #
# 进度 + ETA
# --------------------------------------------------------------------- #
def fmt_time(sec):
    if sec is None or sec != sec or sec < 0:
        return '--:--'
    sec = int(sec)
    h, m, s = sec // 3600, (sec % 3600) // 60, sec % 60
    return f'{h:d}:{m:02d}:{s:02d}' if h else f'{m:02d}:{s:02d}'


class ProgressHook:
    """训练进度 + ETA。

    ❗ETA 必须用**滑动窗口增量速度**（Δ时间/Δ迭代），不能用累计平均
      「已用时间 / 已完成iteration」。累计平均会把启动开销
      （worker 启动、cuDNN autotune、首次读大图，约 70s）
      全摊到前几十个 iteration 上，让 ETA 虚高 3~9 倍。
      这是本项目在姿态训练时实际踩过的坑。
    """

    def __init__(self, total_iters_per_epoch, max_epochs, window=6):
        self.tpe = total_iters_per_epoch
        self.max_epochs = max_epochs
        self.total = total_iters_per_epoch * max_epochs
        self.done = 0
        self.t0 = None
        self._dq = deque(maxlen=window)

    def update(self, n=1):
        now = time.time()
        self.done += n
        self._dq.append((now, self.done))
        if self.t0 is None:
            self.t0 = now

    def incr_speed(self):
        if len(self._dq) < 2:
            return None
        (t0, i0), (t1, i1) = self._dq[0], self._dq[-1]
        di = i1 - i0
        return (t1 - t0) / di if di > 0 else None

    def render(self):
        sp = self.incr_speed()
        ep = self.done / max(1, self.tpe)
        left = (self.total - self.done) * sp if sp else None
        eta = fmt_time(left)
        el = fmt_time(time.time() - self.t0 if self.t0 else 0)
        pct = 100.0 * self.done / max(1, self.total)
        bar = ('#' * int(pct / 3)).ljust(33, '.')
        return (f'  [{bar}] {pct:5.1f}%  '
                f'iter {self.done}/{self.total}  '
                f'ep {ep:5.2f}/{self.max_epochs}  '
                f'{sp if sp else 0:.3f}s/it  '
                f'elapsed {el}  ETA {eta}')


class CliProgressHook:
    """mmengine hook：定期打一行进度。

    ❗不要用 `\\r` 覆盖式输出：mmengine 的 logger 也在写 stdout，
      两者交错会出现 `ETA 01:3410/10 17:37:52 - mmengine - INFO` 这种
      粘连（实测）。改成**每 N iter 打一行**，与 logger 的 interval 对齐，
      输出干净、可直接重定向到文件。
    """

    def __init__(self, pg, every=50):
        self.pg = pg
        self.every = every

    def after_train_iter(self, runner, batch_idx, data_batch=None,
                         outputs=None):
        self.pg.update(1)
        if runner.iter % self.every == 0:
            print(self.pg.render(), flush=True)


# --------------------------------------------------------------------- #
def find_latest_ckpt(work_dir):
    p = osp.join(work_dir, 'last_checkpoint')
    if osp.isfile(p):
        with open(p) as f:
            c = f.read().strip()
        if c and osp.isfile(c):
            return c
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', default='config_det.py')
    ap.add_argument('--work-dir', default=WORK_DIR)
    ap.add_argument('--epochs', type=int, default=None)
    ap.add_argument('--batch-size', type=int, default=4)
    ap.add_argument('--val-interval', type=int, default=5)
    ap.add_argument('--log-interval', type=int, default=50)
    ap.add_argument('--smoke', action='store_true',
                    help='100 iter 冒烟（1 epoch 内）')
    ap.add_argument('--no-resume', action='store_true')
    ap.add_argument('--no-gpu-monitor', action='store_true')
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args()

    from mmengine.config import Config
    from mmengine.runner import Runner

    cfg = Config.fromfile(args.config)
    cfg.work_dir = args.work_dir
    cfg.train_dataloader.batch_size = args.batch_size
    cfg.val_dataloader.batch_size = args.batch_size

    epochs = 2 if args.smoke else (args.epochs or cfg.train_cfg.max_epochs)
    cfg.train_cfg.max_epochs = epochs
    cfg.train_cfg.val_interval = 1 if args.smoke else args.val_interval

    # scheduler 随 epochs 缩放：cosine 退火区间取后半程
    for s in cfg.param_scheduler:
        if s.get('type') == 'CosineAnnealingLR':
            s['begin'] = epochs // 2
            s['end'] = epochs
            s['T_max'] = max(1, epochs - epochs // 2)
        if s.get('type') == 'LinearLR':
            s['end'] = min(500, 10 if args.smoke else 500)

    for h in cfg.default_hooks.get('checkpoint', {}).items() \
            if isinstance(cfg.default_hooks.get('checkpoint'), dict) else []:
        pass
    if isinstance(cfg.default_hooks.get('checkpoint'), dict):
        cfg.default_hooks['checkpoint']['interval'] = 1 if args.smoke \
            else max(1, epochs // 12)

    # 随机种子
    cfg.randomness = dict(seed=args.seed, deterministic=False)

    if args.smoke:
        cfg.train_cfg.max_epochs = 1
        cfg.default_hooks['logger']['interval'] = 10
        cfg.train_dataloader.num_workers = 2
        cfg.val_dataloader.num_workers = 2

    print('=' * 80)
    print('DMKPD 检测器训练（RTMDet-tiny, 1 类 DPM）')
    print('=' * 80)
    print(f'  work_dir     {cfg.work_dir}')
    print(f'  epochs       {cfg.train_cfg.max_epochs}')
    print(f'  batch_size   {cfg.train_dataloader.batch_size}')
    print(f'  val_interval {cfg.train_cfg.val_interval}')
    print(f'  img size     {cfg.train_pipeline[2].scale}')
    print(f'  lr           {cfg.optim_wrapper.optimizer.lr}')
    print(f'  load_from    {cfg.load_from}')
    print(f'  train ann    {cfg.train_dataloader.dataset.ann_file}')
    print(f'  GPU          {torch.cuda.get_device_name(0)}')

    # 续训
    if not args.no_resume:
        ck = find_latest_ckpt(cfg.work_dir)
        if ck:
            cfg.load_from = ck
            cfg.resume = False
            print(f'  [续训] 从 {ck} 的权重继续（不恢复优化器状态）')
        else:
            print('  [从头] 未找到 last_checkpoint，用预训练权重开始')
    print()

    runner = Runner.from_cfg(cfg)

    # 统计每 epoch iter 数
    tpe = len(runner.train_dataloader)
    print(f'  每 epoch = {tpe} iter；'
          f'总 {tpe * cfg.train_cfg.max_epochs} iter')
    sys.stdout.flush()

    gm = None
    if not args.no_gpu_monitor and torch.cuda.is_available():
        gm = GpuMonitor(interval=2.0)
        gm.start()

    # 注册进度 hook（挂到 runner 的 custom hooks 上）
    from mmengine.hooks import Hook
    from mmengine.registry import HOOKS

    @HOOKS.register_module(name='_CliProgress')
    class _H(Hook):
        priority = 100

        def __init__(self):
            self.pg = ProgressHook(tpe, cfg.train_cfg.max_epochs)
            self._hook = CliProgressHook(self.pg, every=args.log_interval)

        def after_train_iter(self, runner, batch_idx, data_batch=None,
                             outputs=None):
            self._hook.after_train_iter(runner, batch_idx, data_batch,
                                        outputs)

    runner.register_hook(_H(), priority=100)

    t_start = time.time()
    try:
        runner.train()
    except KeyboardInterrupt:
        print('\n  [中断] 用户中止，正在整理输出...')
    finally:
        if gm:
            gm.stop()
        dt = time.time() - t_start

    sys.stdout.write('\n')
    print()
    print('=' * 80)
    print(f'训练结束  墙钟 {fmt_time(dt)}')
    print('=' * 80)
    if gm:
        s = gm.summary()
        if s:
            print(f'  GPU 采样 {s["n"]} 次')
            print(f'  利用率  GPU {s["util_gpu"][0]:.1f}% (max '
                  f'{s["util_gpu"][1]:.0f}%)   '
                  f'显存带宽 {s["util_mem"][0]:.1f}%')
            print(f'  温度    {s["temp"][0]:.1f}C (max {s["temp"][1]:.0f}C)')
            print(f'  功耗    {s["power"][0]:.1f}W (max {s["power"][1]:.0f}W)')
            print(f'  频率    {s["clock"][0]:.0f}MHz '
                  f'(max {s["clock"][1]:.0f}MHz)')
            print(f'  显存    {s["mem"][0]:.0f}MB '
                  f'(max {s["mem"][1]:.0f}MB)')
    if torch.cuda.is_available():
        print(f'  torch 峰值显存 {torch.cuda.max_memory_allocated()/1024**3:.2f}GB')
    print(f'  权重目录 {osp.abspath(cfg.work_dir)}')


if __name__ == '__main__':
    main()
