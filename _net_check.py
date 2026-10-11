# -*- coding: utf-8 -*-
"""网络诊断：定位 git push 失败的原因。

区分三种情况：
  A. 全局断网          -> 所有站点都不通
  B. 仅 GitHub 被阻断  -> github 不通，其他通（国内常见）
  C. 代理配置错误      -> 配了代理但代理没开
"""
import socket
import subprocess
import sys
import time

TARGETS = [
    ('github.com', 443, 'GitHub'),
    ('gitee.com', 443, 'Gitee（国内）'),
    ('www.baidu.com', 443, '百度（对照）'),
    ('gitlab.com', 443, 'GitLab'),
]


def tcp_test(host, port, timeout=6):
    """返回 (是否连通, 耗时秒, 错误信息 或 解析到的IP)"""
    t0 = time.time()
    try:
        ip = socket.gethostbyname(host)
    except Exception as e:
        return False, time.time() - t0, f'DNS 失败: {e}'
    try:
        s = socket.create_connection((ip, port), timeout=timeout)
        s.close()
        return True, time.time() - t0, ip
    except Exception as e:
        return False, time.time() - t0, f'{ip} -> {type(e).__name__}: {e}'


print('=' * 74)
print('TCP 连接测试（port 443）')
print('=' * 74)
print(f'{"站点":<20}{"结果":<8}{"耗时":>8}   详情')
results = {}
for host, port, name in TARGETS:
    ok, dt, info = tcp_test(host, port)
    results[name] = ok
    flag = 'OK' if ok else 'FAIL'
    print(f'{name:<20}{flag:<8}{dt:>7.2f}s   {info}')

print()
print('=' * 74)
print('git 代理配置')
print('=' * 74)
for key in ('http.proxy', 'https.proxy', 'http.sslVerify',
            'remote.origin.proxy', 'credential.helper'):
    r = subprocess.run(['git', 'config', '--global', key],
                       capture_output=True, text=True)
    v = r.stdout.strip()
    print(f'  {key:<28} {v if v else "(未设置)"}')

import os
print()
print('  环境变量代理:')
for k in ('HTTP_PROXY', 'HTTPS_PROXY', 'ALL_PROXY',
          'http_proxy', 'https_proxy'):
    v = os.environ.get(k)
    print(f'    {k:<14} {v if v else "(无)"}')

print()
print('=' * 74)
print('诊断结论')
print('=' * 74)
gh = results.get('GitHub', False)
ge = results.get('Gitee（国内）', False)
bd = results.get('百度（对照）', False)

if not bd:
    print('  ❌ 连百度都不通 -> 本机网络有问题（检查 WiFi / 网线 / 代理软件）')
elif gh:
    print('  ✅ GitHub 可达 -> 网络没问题；push 失败可能是仓库名/权限/凭据')
else:
    print('  ⚠️  百度通、GitHub 不通 -> 典型的 GitHub 被阻断（国内常见）')
    if ge:
        print('  ✅ 但 Gitee 可达 -> 建议改用 Gitee')
    else:
        print('  ❌ Gitee 也不通 -> 检查是否配了失效的代理')

print()
print('  下一步选择：')
if ge:
    print('    方案 A（推荐）：改用 Gitee')
    print('      git remote set-url origin https://gitee.com/<用户名>/dmkpd.git')
    print('      git push -u origin master')
print('    方案 B：给 git 配代理（需要本机有代理软件在运行）')
print('      git config --global http.proxy  http://127.0.0.1:<端口>')
print('      git config --global https.proxy http://127.0.0.1:<端口>')
print('    方案 C：用 U 盘 + git archive（完全不依赖网络）')
print('      git archive -o "E:\\code.zip" HEAD')
print('    方案 D：等网络恢复 / 换时间段重试')
