#!/usr/bin/env python3
"""
清理残留进程脚本 (Python版本)
用于清理MANGER训练过程中产生的残留Python和SC2进程
"""

import os
import subprocess
import sys
import time
import psutil
from pathlib import Path

class ProcessCleaner:
    def __init__(self):
        self.python_pattern = "python src/main.py"
        self.sc2_pattern = "SC2"
        self.temp_pattern = "/tmp/sc-*"
        
    def get_processes(self, pattern):
        """获取匹配模式的进程"""
        processes = []
        for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
            try:
                if proc.info['cmdline']:
                    cmdline = ' '.join(proc.info['cmdline'])
                    if pattern in cmdline:
                        processes.append(proc)
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
        return processes
    
    def kill_processes(self, processes, force=False):
        """终止进程"""
        killed_count = 0
        for proc in processes:
            try:
                if force:
                    proc.kill()
                else:
                    proc.terminate()
                killed_count += 1
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
        return killed_count
    
    def cleanup_temp_files(self):
        """清理临时文件"""
        import glob
        temp_dirs = glob.glob(self.temp_pattern)
        for temp_dir in temp_dirs:
            try:
                import shutil
                shutil.rmtree(temp_dir, ignore_errors=True)
            except Exception:
                pass
        return len(temp_dirs)
    
    def get_system_info(self):
        """获取系统信息"""
        # 内存信息
        memory = psutil.virtual_memory()
        memory_info = f"内存使用: {memory.percent}% ({memory.used // (1024**3)}GB / {memory.total // (1024**3)}GB)"
        
        # GPU信息
        gpu_info = "GPU: 不可用"
        try:
            result = subprocess.run(['nvidia-smi', '--query-gpu=memory.used,memory.total,utilization.gpu', 
                                   '--format=csv,noheader,nounits'], 
                                  capture_output=True, text=True, timeout=5)
            if result.returncode == 0:
                gpu_data = result.stdout.strip().split(',')
                gpu_info = f"GPU: 内存 {gpu_data[0]}MB/{gpu_data[1]}MB, 利用率 {gpu_data[2]}%"
        except (subprocess.TimeoutExpired, FileNotFoundError):
            pass
            
        return memory_info, gpu_info
    
    def run_cleanup(self, verbose=True):
        """运行清理"""
        print("🧹 开始清理残留进程...")
        
        # 获取当前进程
        python_procs = self.get_processes(self.python_pattern)
        sc2_procs = self.get_processes(self.sc2_pattern)
        
        if verbose:
            print(f"📊 发现进程: Python训练进程 {len(python_procs)} 个, SC2进程 {len(sc2_procs)} 个")
        
        if not python_procs and not sc2_procs:
            print("✅ 没有发现残留进程，系统很干净！")
            return True
        
        # 清理Python进程
        if python_procs:
            print("🔍 清理Python训练进程...")
            killed = self.kill_processes(python_procs)
            time.sleep(2)
            
            # 检查是否还有残留进程
            remaining_procs = self.get_processes(self.python_pattern)
            if remaining_procs:
                print(f"⚠️  强制杀死 {len(remaining_procs)} 个残留Python进程")
                self.kill_processes(remaining_procs, force=True)
            
            print("✅ Python训练进程清理完成")
        
        # 清理SC2进程
        if sc2_procs:
            print("🔍 清理SC2进程...")
            killed = self.kill_processes(sc2_procs)
            time.sleep(3)
            
            # 检查是否还有残留进程
            remaining_sc2_procs = self.get_processes(self.sc2_pattern)
            if remaining_sc2_procs:
                print(f"⚠️  强制杀死 {len(remaining_sc2_procs)} 个残留SC2进程")
                self.kill_processes(remaining_sc2_procs, force=True)
            
            print("✅ SC2进程清理完成")
        
        # 清理临时文件
        print("🧽 清理临时文件...")
        temp_count = self.cleanup_temp_files()
        if temp_count > 0:
            print(f"✅ 清理了 {temp_count} 个临时目录")
        else:
            print("✅ 没有临时文件需要清理")
        
        # 最终检查
        final_python = self.get_processes(self.python_pattern)
        final_sc2 = self.get_processes(self.sc2_pattern)
        
        if verbose:
            print(f"📊 清理后状态: Python进程 {len(final_python)} 个, SC2进程 {len(final_sc2)} 个")
            
            # 系统信息
            memory_info, gpu_info = self.get_system_info()
            print(f"💾 {memory_info}")
            print(f"💾 {gpu_info}")
        
        if not final_python and not final_sc2:
            print("🎉 清理完成！所有残留进程已成功清理")
            print("✅ 系统已准备好运行新的训练任务")
            return True
        else:
            print("⚠️  仍有部分进程未能清理，可能需要手动检查")
            return False

def main():
    cleaner = ProcessCleaner()
    
    # 检查命令行参数
    verbose = True
    if len(sys.argv) > 1 and sys.argv[1] == '--quiet':
        verbose = False
    
    success = cleaner.run_cleanup(verbose=verbose)
    
    if verbose:
        print("\n📝 使用说明:")
        print("  运行训练: python src/main.py --config=triq_sep --env-config=sc2")
        print("  清理进程: python cleanup_processes.py")
        print("  安静模式: python cleanup_processes.py --quiet")
        print("  检查进程: ps aux | grep -E '(python.*main.py|SC2)'")
    
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()
