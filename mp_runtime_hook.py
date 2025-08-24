# mp_runtime_hook.py
import multiprocessing as mp
import sys

# 让 PyInstaller 打包后的子进程能正常启动
mp.freeze_support()

# 统一使用 spawn，避免子进程重复跑主模块顶层代码
try:
    mp.set_start_method("spawn")
except RuntimeError:
    # 已经设置过就忽略
    pass

# 在冻结环境中，让 multiprocessing 用当前 EXE 作为子进程可执行文件
if getattr(sys, "frozen", False):
    mp.set_executable(sys.executable)
