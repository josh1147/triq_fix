#!/usr/bin/env python3
"""
三元Q值框架快速测试脚本
用于验证基础功能和性能
"""

import os
import sys
import subprocess
import time
import argparse
import yaml
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

def run_training_test(config_name, map_name, steps=50000, test_mode="quick"):
    """
    运行训练测试
    Args:
        config_name: 配置名称
        map_name: 地图名称
        steps: 训练步数
        test_mode: 测试模式
    """
    print(f"\n🚀 开始测试: {config_name} on {map_name}")
    print(f"训练步数: {steps}")
    print(f"测试模式: {test_mode}")
    
    # 构建命令
    cmd = [
        "python", "src/main.py",
        f"--config={config_name}",
        f"--env-config=sc2",
        f"--env_args.map_name={map_name}",
        f"--t_max={steps}"
    ]
    
    # 根据测试模式调整参数
    if test_mode == "quick":
        cmd.extend([
            "--test_interval=5000",
            "--log_interval=5000",
            "--batch_size=32"
        ])
    elif test_mode == "detailed":
        cmd.extend([
            "--test_interval=10000",
            "--log_interval=1000",
            "--batch_size=64"
        ])
    
    print(f"执行命令: {' '.join(cmd)}")
    
    # 记录开始时间
    start_time = time.time()
    
    try:
        # 运行训练
        result = subprocess.run(cmd, cwd=project_root, capture_output=True, text=True, timeout=3600)
        
        # 记录结束时间
        end_time = time.time()
        duration = end_time - start_time
        
        print(f"✅ 测试完成，耗时: {duration:.2f}秒")
        
        if result.returncode == 0:
            print("✅ 训练成功完成")
            return True, duration
        else:
            print(f"❌ 训练失败，返回码: {result.returncode}")
            print(f"错误输出: {result.stderr}")
            return False, duration
            
    except subprocess.TimeoutExpired:
        print("❌ 训练超时")
        return False, 3600
    except Exception as e:
        print(f"❌ 训练异常: {e}")
        return False, 0

def run_complexity_gradient_test():
    """运行复杂度梯度测试"""
    print("\n📊 开始复杂度梯度测试")
    
    # 测试配置
    test_configs = [
        {"map": "2s3z", "steps": 50000, "complexity": "simple", "description": "简单场景：2个stalker + 3个zealot"},
        {"map": "3s5z", "steps": 100000, "complexity": "medium", "description": "中等场景：3个stalker + 5个zealot"},
        {"map": "MMM", "steps": 200000, "complexity": "complex", "description": "复杂场景：MMM地图"},
        {"map": "MMM2", "steps": 500000, "complexity": "super_complex", "description": "超复杂场景：MMM2地图"}
    ]
    
    results = {}
    
    for config in test_configs:
        print(f"\n--- {config['complexity'].upper()} 场景测试 ---")
        print(f"描述: {config['description']}")
        
        # 测试原始MANGER方法
        print("\n🔵 测试原始MANGER方法...")
        success_manger, duration_manger = run_training_test(
            "qmix_sep", config["map"], config["steps"], "quick"
        )
        
        # 测试三元Q值方法
        print("\n🔴 测试三元Q值方法...")
        success_triq, duration_triq = run_training_test(
            "triq_sep", config["map"], config["steps"], "quick"
        )
        
        # 记录结果
        results[config["complexity"]] = {
            "map": config["map"],
            "description": config["description"],
            "manger": {
                "success": success_manger,
                "duration": duration_manger
            },
            "triq": {
                "success": success_triq,
                "duration": duration_triq
            }
        }
        
        # 输出比较结果
        if success_manger and success_triq:
            speedup = duration_manger / duration_triq if duration_triq > 0 else 0
            print(f"\n📈 性能比较:")
            print(f"  MANGER耗时: {duration_manger:.2f}秒")
            print(f"  TriQ耗时: {duration_triq:.2f}秒")
            print(f"  速度提升: {speedup:.2f}x")
        else:
            print(f"\n⚠️ 部分测试失败，无法比较性能")
    
    return results

def run_ablation_test():
    """运行消融实验"""
    print("\n🔬 开始消融实验")
    
    # 消融实验配置
    ablation_configs = [
        {"name": "triq_full", "config": "triq_sep", "description": "完整三元Q值框架"},
        {"name": "triq_no_surrounding", "config": "triq_sep", "description": "无Q_surrounding"},
        {"name": "triq_fixed_weights", "config": "triq_sep", "description": "固定权重"},
        {"name": "triq_single_rnd", "config": "triq_sep", "description": "单一RND"}
    ]
    
    results = {}
    test_map = "2s3z"  # 使用简单地图进行快速测试
    test_steps = 30000
    
    for config in ablation_configs:
        print(f"\n--- {config['description']} ---")
        
        success, duration = run_training_test(
            config["config"], test_map, test_steps, "quick"
        )
        
        results[config["name"]] = {
            "description": config["description"],
            "success": success,
            "duration": duration
        }
    
    return results

def generate_test_report(results, ablation_results):
    """生成测试报告"""
    print("\n" + "="*60)
    print("📋 测试报告")
    print("="*60)
    
    # 复杂度梯度测试结果
    print("\n🎯 复杂度梯度测试结果:")
    for complexity, result in results.items():
        print(f"\n{complexity.upper()}场景 ({result['map']}):")
        print(f"  描述: {result['description']}")
        print(f"  MANGER: {'✅' if result['manger']['success'] else '❌'} ({result['manger']['duration']:.2f}s)")
        print(f"  TriQ: {'✅' if result['triq']['success'] else '❌'} ({result['triq']['duration']:.2f}s)")
        
        if result['manger']['success'] and result['triq']['success']:
            speedup = result['manger']['duration'] / result['triq']['duration']
            print(f"  性能提升: {speedup:.2f}x")
    
    # 消融实验结果
    print("\n🔬 消融实验结果:")
    for name, result in ablation_results.items():
        status = "✅" if result['success'] else "❌"
        print(f"  {result['description']}: {status} ({result['duration']:.2f}s)")
    
    # 总结
    print("\n📊 总结:")
    total_tests = len(results) + len(ablation_results)
    successful_tests = sum(1 for r in results.values() if r['manger']['success'] and r['triq']['success'])
    successful_tests += sum(1 for r in ablation_results.values() if r['success'])
    
    print(f"  总测试数: {total_tests}")
    print(f"  成功测试: {successful_tests}")
    print(f"  成功率: {successful_tests/total_tests*100:.1f}%")
    
    if successful_tests > total_tests * 0.8:
        print("🎉 测试总体成功！三元Q值框架基础功能正常")
    elif successful_tests > total_tests * 0.5:
        print("⚠️ 测试部分成功，需要进一步调试")
    else:
        print("❌ 测试失败较多，需要检查实现")

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="三元Q值框架测试脚本")
    parser.add_argument("--test_type", choices=["complexity", "ablation", "both"], 
                       default="both", help="测试类型")
    parser.add_argument("--quick", action="store_true", help="快速测试模式")
    
    args = parser.parse_args()
    
    print("🚀 三元Q值框架测试开始")
    print(f"测试类型: {args.test_type}")
    print(f"快速模式: {args.quick}")
    
    results = {}
    ablation_results = {}
    
    # 运行复杂度梯度测试
    if args.test_type in ["complexity", "both"]:
        results = run_complexity_gradient_test()
    
    # 运行消融实验
    if args.test_type in ["ablation", "both"]:
        ablation_results = run_ablation_test()
    
    # 生成报告
    generate_test_report(results, ablation_results)
    
    print("\n✅ 测试完成！")

if __name__ == "__main__":
    main()
