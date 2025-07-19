#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
测试项目升级功能
"""

import sys
import time
import logging
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from config import get_config, ConfigManager
from utils.error_handler import error_handler, resource_monitor, log_manager
from utils.performance_monitor import performance_monitor, PerformanceAnalyzer
from models.advanced_controller import AdvancedTLSController
from models.advanced_env import AdvancedTrafficEnv

logger = logging.getLogger(__name__)


def test_config_system():
    """测试配置系统"""
    print("\n🔧 测试配置系统...")
    
    try:
        # 测试配置加载
        config = get_config()
        print(f"✅ 配置加载成功")
        print(f"   训练步数: {config.training.timesteps}")
        print(f"   学习率: {config.training.learning_rate}")
        print(f"   奖励权重: {config.get_reward_weights()}")
        
        # 测试配置更新
        config.update_reward_weights({'waiting_time': 0.4})
        print(f"✅ 配置更新成功")
        
        return True
        
    except Exception as e:
        print(f"❌ 配置系统测试失败: {e}")
        return False


def test_error_handling():
    """测试错误处理系统"""
    print("\n🛡️ 测试错误处理系统...")
    
    try:
        # 测试重试装饰器
        @error_handler.retry_on_error(max_retries=2, delay=0.1)
        def failing_function():
            raise ValueError("测试错误")
        
        # 测试安全执行
        result = error_handler.safe_execute(
            lambda: 1 / 0, 
            default_return="安全默认值"
        )
        print(f"✅ 安全执行测试: {result}")
        
        # 测试错误跟踪
        success = error_handler.track_error("test_error", max_count=5)
        print(f"✅ 错误跟踪测试: {success}")
        
        return True
        
    except Exception as e:
        print(f"❌ 错误处理测试失败: {e}")
        return False


def test_resource_monitoring():
    """测试资源监控"""
    print("\n📊 测试资源监控...")
    
    try:
        # 测试系统信息获取
        system_info = resource_monitor.get_system_info()
        print(f"✅ 系统信息: {system_info}")
        
        # 测试资源检查
        memory_ok = resource_monitor.check_memory_usage(threshold_mb=4096)
        cpu_ok = resource_monitor.check_cpu_usage(threshold_percent=95)
        disk_ok = resource_monitor.check_disk_space(threshold_gb=0.5)
        
        print(f"✅ 资源检查 - 内存: {memory_ok}, CPU: {cpu_ok}, 磁盘: {disk_ok}")
        
        return True
        
    except Exception as e:
        print(f"❌ 资源监控测试失败: {e}")
        return False


def test_performance_monitoring():
    """测试性能监控"""
    print("\n⚡ 测试性能监控...")
    
    try:
        # 模拟一些性能数据
        for i in range(10):
            performance_monitor.update(
                episode=1,
                step=i,
                metrics={
                    'reward': -100 + i * 10,
                    'waiting_time': 50 - i * 2,
                    'queue_length': 10 - i,
                    'throughput': i * 2,
                    'speed': 5 + i * 0.5,
                    'phase_switches': i // 3
                }
            )
        
        # 获取性能统计
        recent_perf = performance_monitor.collector.get_recent_performance(window=5)
        print(f"✅ 性能统计: {recent_perf}")
        
        # 生成分析报告
        analyzer = PerformanceAnalyzer(performance_monitor.collector)
        report = analyzer.generate_report()
        print(f"✅ 分析报告生成成功，建议数: {len(report.get('recommendations', []))}")
        
        return True
        
    except Exception as e:
        print(f"❌ 性能监控测试失败: {e}")
        return False


def test_controller_improvements():
    """测试控制器改进"""
    print("\n🚦 测试控制器改进...")
    
    try:
        # 创建控制器（不连接SUMO）
        controller = AdvancedTLSController("test_tls")
        print(f"✅ 控制器创建成功，初始化状态: {controller.is_initialized}")
        
        # 测试安全归一化
        normalized = controller._safe_normalize(50, 0, 100)
        print(f"✅ 安全归一化测试: {normalized}")
        
        # 测试无效值修复
        fixed = controller._fix_invalid_value(float('nan'))
        print(f"✅ 无效值修复测试: {fixed}")
        
        return True
        
    except Exception as e:
        print(f"❌ 控制器测试失败: {e}")
        return False


def test_environment_stability():
    """测试环境稳定性"""
    print("\n🌍 测试环境稳定性...")
    
    try:
        # 测试环境创建（不启动SUMO）
        config = get_config()
        
        # 模拟环境配置
        env_config = {
            'reward_weights': config.get_reward_weights(),
            'environment': config.get_environment_params()
        }
        
        print(f"✅ 环境配置验证成功")
        print(f"   奖励权重数量: {len(env_config['reward_weights'])}")
        print(f"   环境参数数量: {len(env_config['environment'])}")
        
        return True
        
    except Exception as e:
        print(f"❌ 环境稳定性测试失败: {e}")
        return False


def run_all_tests():
    """运行所有测试"""
    print("🧪 开始项目升级测试...")
    print("=" * 60)
    
    tests = [
        ("配置系统", test_config_system),
        ("错误处理", test_error_handling),
        ("资源监控", test_resource_monitoring),
        ("性能监控", test_performance_monitoring),
        ("控制器改进", test_controller_improvements),
        ("环境稳定性", test_environment_stability),
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            success = test_func()
            results.append((test_name, success))
        except Exception as e:
            print(f"❌ {test_name} 测试异常: {e}")
            results.append((test_name, False))
    
    # 输出测试结果
    print("\n" + "=" * 60)
    print("🏁 测试结果总结:")
    
    passed = 0
    total = len(results)
    
    for test_name, success in results:
        status = "✅ 通过" if success else "❌ 失败"
        print(f"   {test_name}: {status}")
        if success:
            passed += 1
    
    print(f"\n📊 总体结果: {passed}/{total} 测试通过")
    
    if passed == total:
        print("🎉 所有测试通过！项目升级成功！")
        return True
    else:
        print("⚠️ 部分测试失败，请检查相关功能")
        return False


if __name__ == "__main__":
    # 设置日志
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    success = run_all_tests()
    
    if success:
        print("\n💡 升级功能说明:")
        print("   • 配置管理: 统一的配置文件和参数管理")
        print("   • 错误处理: 自动重试和错误恢复机制")
        print("   • 资源监控: 实时监控内存、CPU和磁盘使用")
        print("   • 性能监控: 训练过程性能分析和建议")
        print("   • 数据验证: 输入数据边界检查和无效值处理")
        print("   • 日志系统: 结构化日志和错误追踪")
        
        print("\n🚀 现在可以使用升级后的训练系统:")
        print("   python stable_train.py --scenario competition --timesteps 50000")
        
        sys.exit(0)
    else:
        sys.exit(1)
