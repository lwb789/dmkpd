"""DMKPD 训练启动脚本（Windows 本地直接运行）。

    .\\.venv\\Scripts\\python.exe run_train.py                # 默认 20 epoch
    .\\.venv\\Scripts\\python.exe run_train.py --epochs 50
    .\\.venv\\Scripts\\python.exe run_train.py --batch-size 8
    .\\.venv\\Scripts\\python.exe run_train.py --no-gpu-monitor

功能
----
1. **清晰的进度显示**：每 N 个 iter 打印一行，含
   epoch / iter / loss / lr / 已用 / 剩余 / 速度
2. **GPU 监控**：后台线程每 2 秒采一次 nvidia-smi，
   把 GPU 利用率 / 温度 / 功耗 记入日志
3. **断点续训**：work_dir 下有 checkpoint 时自动 --resume
4. **退出时汇总**：训练结束打印耗时、GPU 利用率与温度峰值

ⓘ 关于"GPU 利用率低"
----------------------
Windows 任务管理器的"GPU 利用率"默认统计 **3D 引擎**，
而 CUDA 计算走的是 **Compute 引擎**，所以任务管理器会显示很低（甚至 0%）。
**这不代表 GPU 没用上**。本脚本用 `nvidia-smi` 采的是真实利用率。
实测本任务训练时利用率 96~99%（已接近满载，属正常）。

ⓘ 关于训练时长
--------------
实测稳态速度 **0.41 s/it**（bs=4, 256x256, HRNet-W48, GTX 1650）：
    1 epoch = 5124 it x 0.41s ≈ 35 min
   20 epoch ≈ 12 h
**注意**：训练刚启动的前几十个 iter 会慢很多（dataloader worker 启动、
cuDNN 选算法、首次读大图），本脚本的 ETA 用滑动窗口算增量速度，
不会被这段启动开销污染。
"""
import argparse
import os
import subprocess
import sys
import threading
import time
from collections import deque

# --------------------------------------------------------------------- #
# GPU 监控（后台线程）
# --------------------------------------------------------------------- #
NV_QUERY = ('utilization.gpu,utilization.memory,temperature.gpu,'
            'power.draw,clocks.sm,memory.used')


class GpuMonitor:
    """后台采样 nvidia-smi，记录 GPU 利用率 / 温度 / 功耗。"""

    def __init__(self, interval: float = 2.0):
        self.interval = interval
        self._stop = False
        self._thread = None
        self.samples = []          # [(ts, util_gpu, util_mem, temp, power)]
        self.latest = None
        self.enabled = self._available()

    @staticmethod
    def _available() -> bool:
        try:
            r = subprocess.run(['nvidia-smi', '-L'], capture_output=True,
                               text=True, timeout=5)
            return r.returncode == 0
        except Exception:
            return False

    def _query(self):
        try:
            r = subprocess.run(
                ['nvidia-smi', f'--query-gpu={NV_QUERY}',
                 '--format=csv,noheader,nounits'],
                capture_output=True, text=True, timeout=5)
            if r.returncode != 0:
                return None
            parts = [p.strip() for p in r.stdout.strip().split(',')]
            if len(parts) < 6:
                return None
            return dict(
                util_gpu=float(parts[0]),
                util_mem=float(parts[1]),
                temp=float(parts[2]),
                power=float(parts[3]) if parts[3] not in ('[N/A]', 'N/A')
                else float('nan'),
                clock_sm=float(parts[4]) if parts[4] not in ('[N/A]', 'N/A')
                else float('nan'),
                mem_used=float(parts[5]) / 1024.0,   # MiB -> GiB
            )
        except Exception:
            return None

    def _loop(self):
        while not self._stop:
            s = self._query()
            if s is not None:
                s['ts'] = time.time()
                self.latest = s
                self.samples.append(s)
            time.sleep(self.interval)

    def start(self):
        if not self.enabled:
            return
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop = True
        if self._thread is not None:
            self._thread.join(timeout=5)

    # ---- 统计 ---- #
    def summary(self):
        if not self.samples:
            return None
        utils = [s['util_gpu'] for s in self.samples]
        temps = [s['temp'] for s in self.samples]
        powers = [p for p in (s['power'] for s in self.samples)
                  if p == p]  # 过滤 NaN
        return dict(
            n=len(self.samples),
            util_mean=sum(utils) / len(utils),
            util_min=min(utils),
            util_max=max(utils),
            temp_mean=sum(temps) / len(temps),
            temp_max=max(temps),
            power_mean=(sum(powers) / len(powers)) if powers else float('nan'),
            power_max=max(powers) if powers else float('nan'),
        )


# --------------------------------------------------------------------- #
# 进度 Hook
# --------------------------------------------------------------------- #
from mmengine.hooks import Hook  # noqa: E402


class ProgressHook(Hook):
    """打印简洁的进度行：epoch/iter + loss + lr + 已用 + 剩余。

    ❗关于 ETA 的计算方式（我踩过坑）
    -------------------------------
    不能用 `已用时间 / 已完成迭代` 这种**累计平均**，因为训练启动阶段
    （dataloader worker 启动、cuDNN autotune、首次读图）会产生数十秒的
    一次性开销，全部摊到前几十个 iter 上，导致 ETA 严重虚高。
    实测：累计平均显示 1.2s/it，而真实稳态只有 0.41s/it（差 3 倍）。

    正确做法是用**最近若干次迭代的增量速度**：记录 (时间, 迭代数) 的
    滑动窗口，用 `Δ时间 / Δ迭代` 估计瞬时速度。
    """

    def __init__(self, interval: int = 50, gpu_monitor: GpuMonitor = None,
                 total_epochs: int = 1, iters_per_epoch: int = 1,
                 speed_window: int = 5):
        self.interval = interval
        self.gpu = gpu_monitor
        self.total_epochs = total_epochs
        self.iters_per_epoch = iters_per_epoch
        self.speed_window = speed_window
        self.t_start = None
        self.t_epoch_start = None
        self.epoch = 0
        # 滑动窗口：存 (时间戳, 累计迭代数)
        self._samples = deque(maxlen=speed_window + 1)

    def before_run(self, runner):
        self.t_start = time.time()
        runner.logger.info('')
        runner.logger.info('=' * 96)
        runner.logger.info(
            f'训练开始 | 共 {self.total_epochs} epoch '
            f'x {self.iters_per_epoch} iter/epoch = '
            f'{self.total_epochs * self.iters_per_epoch} iter')
        if self.gpu and not self.gpu.enabled:
            runner.logger.info('GPU 监控: 不可用（未找到 nvidia-smi）')
        runner.logger.info('=' * 96)

    def before_train_epoch(self, runner):
        self.t_epoch_start = time.time()
        self.epoch = runner.epoch + 1
        # 跨 epoch 时清空窗口，避免上一个 epoch 末尾的速度被延续
        self._samples.clear()
        runner.logger.info(
            f'--- Epoch {self.epoch}/{self.total_epochs} 开始 ---')

    def _incr_speed(self, done_iters: int, now: float) -> float:
        """用滑动窗口算瞬时速度（秒/iter）。窗口不足时回退到累计平均。"""
        self._samples.append((now, done_iters))
        if len(self._samples) >= 2:
            (t0, i0), (t1, i1) = self._samples[0], self._samples[-1]
            d_it = i1 - i0
            d_t = t1 - t0
            if d_it > 0 and d_t > 0:
                return d_t / d_it
        # 回退：累计平均
        return (now - self.t_start) / max(1, done_iters)

    def after_train_iter(self, runner, batch_idx: int, data_batch=None,
                         outputs=None):
        if (batch_idx + 1) % self.interval != 0:
            return

        now = time.time()
        done_iters = runner.epoch * self.iters_per_epoch + batch_idx + 1
        total_iters = self.total_epochs * self.iters_per_epoch
        elapsed = now - self.t_start
        speed = self._incr_speed(done_iters, now)
        remain = speed * (total_iters - done_iters)

        # 取 loss 做展示
        loss_txt = ''
        if outputs:
            losses = {k: v for k, v in outputs.items()
                      if k.startswith('loss') and hasattr(v, 'item')}
            if losses:
                main = losses.get('loss_kpt', next(iter(losses.values())))
                loss_txt = f'loss={main.item():.5f}'

        lr = ''
        try:
            lr = f"lr={runner.optim_wrapper.get_lr()['lr'][0]:.2e}"
        except Exception:
            pass

        gpu_txt = ''
        if self.gpu and self.gpu.latest:
            g = self.gpu.latest
            p = '' if g['power'] != g['power'] else f" {g['power']:.0f}W"
            gpu_txt = (f" | GPU {g['util_gpu']:.0f}%"
                       f" {g['temp']:.0f}C{p}"
                       f" {g['mem_used']:.2f}G")

        runner.logger.info(
            f'  [{self.epoch}/{self.total_epochs}]'
            f'[{batch_idx + 1:>5}/{self.iters_per_epoch}] '
            f'{loss_txt:<18} {lr:<12}'
            f' 已用 {fmt_time(elapsed)} 剩余 {fmt_time(remain)}'
            f' ({speed:.3f}s/it){gpu_txt}')

    def after_val_epoch(self, runner, metrics=None):
        if not metrics:
            return
        keys = [k for k in metrics
                if k.startswith(('coco/AP ', 'coco/AP', 'PCK', 'EPE'))]
        show = {k: metrics[k] for k in keys
                if k in ('coco/AP', 'coco/AP .5', 'coco/AP .75',
                         'coco/AR', 'PCK@0.05/PCK', 'PCK@0.10/PCK', 'EPE/EPE')}
        txt = '  '.join(f'{k.split("/")[-1]}={v:.4f}' for k, v in show.items())
        runner.logger.info(f'  ★ 验证结果: {txt}')

    def after_run(self, runner):
        total = time.time() - self.t_start
        runner.logger.info('')
        runner.logger.info('=' * 96)
        runner.logger.info(f'训练结束 | 总耗时 {fmt_time(total)}')

        # 最佳指标
        try:
            best = runner.message_hub.get_info('best_score')
            runner.logger.info(f'最佳 coco/AP = {best:.4f}')
        except Exception:
            pass

        if self.gpu:
            s = self.gpu.summary()
            if s:
                runner.logger.info(
                    f'GPU 统计: 利用率 均值{s["util_mean"]:.0f}% '
                    f'(最低{s["util_min"]:.0f}% 最高{s["util_max"]:.0f}%) | '
                    f'温度 均值{s["temp_mean"]:.0f}C 峰值{s["temp_max"]:.0f}C | '
                    f'功耗 均值{s["power_mean"]:.0f}W 峰值{s["power_max"]:.0f}W')
        runner.logger.info('=' * 96)


def fmt_time(sec: float) -> str:
    sec = int(max(0, sec))
    h, m, s = sec // 3600, (sec % 3600) // 60, sec % 60
    if h > 0:
        return f'{h}h{m:02d}m'
    return f'{m}m{s:02d}s'


# --------------------------------------------------------------------- #
def parse_args():
    p = argparse.ArgumentParser('DMKPD 训练')
    p.add_argument('--config', default='config.py')
    p.add_argument('--work-dir', default='work_dirs/dmkpd')
    p.add_argument('--epochs', type=int, default=20,
                   help='总 epoch 数（默认 20）')
    p.add_argument('--batch-size', type=int, default=4,
                   help='batch size（显存 4GB 建议 4；8 需 ~3.1GB）')
    p.add_argument('--val-interval', type=int, default=2,
                   help='每多少 epoch 验证一次')
    p.add_argument('--num-workers', type=int, default=4)
    p.add_argument('--lr', type=float, default=None, help='覆盖学习率')
    p.add_argument('--log-interval', type=int, default=50,
                   help='每多少 iter 打印一次进度')
    p.add_argument('--gpu-monitor-interval', type=float, default=2.0)
    p.add_argument('--no-gpu-monitor', action='store_true')
    p.add_argument('--no-resume', action='store_true',
                   help='忽略已有 checkpoint，从头训练')
    p.add_argument('--no-pretrain', action='store_true',
                   help='不加载预训练权重')
    p.add_argument('--seed', type=int, default=0)
    return p.parse_args()


def find_latest_ckpt(work_dir: str):
    """在 work_dir 下找最新的 .pth（排除 last_checkpoint 之类的文本文件）。"""
    import glob
    cands = glob.glob(os.path.join(work_dir, '*.pth'))
    if not cands:
        return None
    return max(cands, key=os.path.getmtime)


def main():
    args = parse_args()

    from mmengine.config import Config
    from mmengine.registry import init_default_scope
    from mmengine.runner import Runner

    init_default_scope('mmpose')

    cfg = Config.fromfile(args.config)

    # ---- 覆盖训练规模 ---- #
    cfg.work_dir = args.work_dir
    cfg.train_cfg = dict(by_epoch=True,
                         max_epochs=args.epochs,
                         val_interval=args.val_interval)
    cfg.train_dataloader['batch_size'] = args.batch_size
    cfg.train_dataloader['num_workers'] = args.num_workers
    cfg.val_dataloader['num_workers'] = args.num_workers
    if args.lr is not None:
        cfg.optim_wrapper['optimizer']['lr'] = args.lr
    if args.no_pretrain:
        cfg.load_from = None

    # 让 param_scheduler 的 end 跟着 epoch 数走，
    # 否则改小 epoch 时 lr 永远降不下来（milestone 落在训练结束之后）
    sched = cfg.param_scheduler
    for s in sched:
        if s.get('by_epoch'):
            s['end'] = args.epochs
            s['milestones'] = [max(1, int(args.epochs * 0.8))]
    for s in sched:
        if not s.get('by_epoch'):
            s['end'] = min(s.get('end', 500), 500)

    # ---- 续训 ---- #
    resume = False
    if not args.no_resume:
        ckpt = find_latest_ckpt(args.work_dir)
        if ckpt:
            print(f'[续训] 发现 checkpoint: {ckpt}')
            resume = True
        else:
            print('[新训] 未发现 checkpoint，从头开始')
    else:
        print('[新训] --no-resume 指定，从头开始')

    # ---- GPU 监控 ---- #
    gpu = None
    if not args.no_gpu_monitor:
        gpu = GpuMonitor(interval=args.gpu_monitor_interval)
        if gpu.enabled:
            gpu.start()
            print(f'[GPU] 监控已启动（每 {args.gpu_monitor_interval}s 采样）')
        else:
            print('[GPU] 未找到 nvidia-smi，跳过监控')

    # ---- 构建 Runner ---- #
    runner = Runner.from_cfg(cfg)
    if resume:
        # 让 Runner 内部按 auto 续训（从 work_dir 找最新）
        runner._resume = True
        runner._load_from = None

    iters_per_epoch = len(runner.train_dataloader)
    total_iters = iters_per_epoch * args.epochs

    print()
    print('=' * 96)
    print(f'  数据: {len(runner.train_dataloader.dataset)} 训练实例 / '
          f'{len(runner.val_dataloader.dataset)} 验证实例')
    print(f'  规模: {args.epochs} epoch x {iters_per_epoch} iter = '
          f'{total_iters} iter (bs={args.batch_size})')
    print(f'  预训练: {cfg.load_from or "无"}')
    print(f'  日志/权重: {os.path.abspath(args.work_dir)}')
    print('=' * 96)
    print()

    hook = ProgressHook(interval=args.log_interval, gpu_monitor=gpu,
                        total_epochs=args.epochs,
                        iters_per_epoch=iters_per_epoch)
    runner.register_custom_hooks([hook])

    try:
        runner.train()
    except KeyboardInterrupt:
        print('\n[中断] 用户手动停止')
    finally:
        if gpu is not None:
            gpu.stop()
            s = gpu.summary()
            if s:
                print()
                print('=' * 96)
                print(f'GPU 统计（{s["n"]} 次采样）:')
                print(f'  利用率  均值 {s["util_mean"]:.1f}%  '
                      f'范围 [{s["util_min"]:.0f}%, {s["util_max"]:.0f}%]')
                print(f'  温度    均值 {s["temp_mean"]:.1f}C  '
                      f'峰值 {s["temp_max"]:.0f}C')
                print(f'  功耗    均值 {s["power_mean"]:.1f}W  '
                      f'峰值 {s["power_max"]:.0f}W')
                print('=' * 96)


if __name__ == '__main__':
    sys.exit(main())
