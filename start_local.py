想·#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
本地使用启动脚本
提供简单的交互式界面，方便本地使用
"""

import os
import sys
import subprocess
from pathlib import Path

def print_banner():
    """打印欢迎横幅"""
    print("=" * 60)
    print("🚦 智能交通信号灯控制系统")
    print("=" * 60)
    print("🏆 支持比赛模式 | 👤 支持个人学习")
    print("包含升级的多智能体协调和智能奖励机制")
    print("=" * 60)

def show_menu():
    """显示主菜单"""
    print("\n📋 请选择操作:")
    print("  1. 🚀 快速训练（后台模式，速度快）")
    print("  2. 👀 可视化训练（观察AI学习过程）")
    print("  3. 🏆 比赛模式训练（长时间高质量训练）")
    print("  4. 🧪 测试已训练的模型")
    print("  5. 📊 查看训练历史")
    print("  6. 🔧 高级训练选项")
    print("  7. ❓ 帮助说明")
    print("  0. 🚪 退出")

def get_latest_model():
    """获取最新的模型文件"""
    local_logs_dir = Path("local_logs")
    if not local_logs_dir.exists():
        return None
        
    model_files = []
    for log_dir in local_logs_dir.iterdir():
        if log_dir.is_dir():
            final_model = log_dir / "final_model.zip"
            if final_model.exists():
                model_files.append((final_model, log_dir.name))
                
    if not model_files:
        return None
        
    # 按时间排序，返回最新的
    model_files.sort(key=lambda x: x[1], reverse=True)
    return str(model_files[0][0])

def list_available_models():
    """列出可用的模型"""
    local_logs_dir = Path("local_logs")
    if not local_logs_dir.exists():
        print("❌ 还没有训练过的模型")
        return []
        
    models = []
    for log_dir in local_logs_dir.iterdir():
        if log_dir.is_dir():
            final_model = log_dir / "final_model.zip"
            if final_model.exists():
                models.append((str(final_model), log_dir.name))
                
    if not models:
        print("❌ 还没有训练过的模型")
        return []
        
    print("\n📁 可用的模型:")
    for i, (model_path, timestamp) in enumerate(models, 1):
        print(f"  {i}. {timestamp}")
        
    return models

def run_training(gui_mode=False, timesteps=200000):
    """运行训练"""
    print(f"\n🚀 开始训练...")
    print(f"   模式: {'可视化' if gui_mode else '后台'}")
    print(f"   训练步数: {timesteps:,}")
    print(f"   预计时间: {timesteps // 1000} 分钟")
    
    cmd = [
        sys.executable, "train_local.py",
        "--mode", "train",
        "--timesteps", str(timesteps)
    ]
    
    if gui_mode:
        cmd.append("--gui")
        
    try:
        result = subprocess.run(cmd, check=True)
        print("\n✅ 训练完成!")
        
        # 获取最新模型
        latest_model = get_latest_model()
        if latest_model:
            print(f"📦 模型已保存: {latest_model}")
            print(f"💡 测试命令: python train_local.py --mode test --model-path {latest_model} --gui")
            
    except subprocess.CalledProcessError as e:
        print(f"\n❌ 训练失败: {e}")
    except KeyboardInterrupt:
        print("\n⏹️ 训练被用户中断")

def run_test():
    """运行测试"""
    models = list_available_models()
    if not models:
        return
        
    print(f"\n请选择要测试的模型 (1-{len(models)}):")
    try:
        choice = int(input("输入选择: ")) - 1
        if 0 <= choice < len(models):
            model_path = models[choice][0]
            print(f"\n🧪 开始测试模型: {models[choice][1]}")
            
            cmd = [
                sys.executable, "train_local.py",
                "--mode", "test",
                "--model-path", model_path,
                "--gui"
            ]
            
            try:
                subprocess.run(cmd, check=True)
                print("\n✅ 测试完成!")
            except subprocess.CalledProcessError as e:
                print(f"\n❌ 测试失败: {e}")
            except KeyboardInterrupt:
                print("\n⏹️ 测试被用户中断")
        else:
            print("❌ 无效选择")
    except ValueError:
        print("❌ 请输入数字")

def show_training_history():
    """显示训练历史和详细评估报告"""
    local_logs_dir = Path("local_logs")
    if not local_logs_dir.exists():
        print("❌ 还没有训练历史")
        return

    # 获取所有训练记录
    training_records = []
    for log_dir in sorted(local_logs_dir.iterdir(), reverse=True):
        if log_dir.is_dir():
            training_records.append(log_dir)

    if not training_records:
        print("❌ 还没有训练历史")
        return

    print("\n📊 训练历史总览:")
    print("=" * 80)

    # 显示训练列表
    for i, log_dir in enumerate(training_records, 1):
        final_model = log_dir / "final_model.zip"
        status = "✅ 完成" if final_model.exists() else "❌ 未完成"

        # 获取基本信息
        basic_info = get_training_basic_info(log_dir)

        print(f"{i:2d}. 📅 {log_dir.name} - {status}")
        print(f"    📈 平均奖励: {basic_info['avg_reward']}")
        print(f"    🎯 效果评级: {basic_info['grade']}")
        print(f"    🔄 总回合数: {basic_info['episodes']}")
        print()

    # 让用户选择查看详细报告
    print("请选择操作:")
    print("  1-{}: 查看详细评估报告".format(len(training_records)))
    print("  0: 返回主菜单")

    try:
        choice = input("\n请输入选择: ").strip()

        if choice == '0':
            return

        choice_num = int(choice)
        if 1 <= choice_num <= len(training_records):
            selected_log_dir = training_records[choice_num - 1]
            show_detailed_evaluation_report(selected_log_dir)
        else:
            print("❌ 无效选择")

    except ValueError:
        print("❌ 请输入数字")

def get_training_basic_info(log_dir: Path) -> dict:
    """获取训练的基本信息"""
    monitor_file = log_dir / "monitor.csv"

    basic_info = {
        'avg_reward': "未知",
        'grade': "未评估",
        'episodes': "未知"
    }

    if monitor_file.exists():
        try:
            import pandas as pd
            df = pd.read_csv(monitor_file, comment='#')
            if not df.empty and 'r' in df.columns:
                avg_reward = df['r'].mean()
                total_episodes = len(df)

                basic_info['avg_reward'] = f"{avg_reward:.1f}"
                basic_info['episodes'] = str(total_episodes)

                # 评级
                if avg_reward >= 400:
                    basic_info['grade'] = "🏆 优秀"
                elif avg_reward >= 200:
                    basic_info['grade'] = "👍 良好"
                elif avg_reward >= 0:
                    basic_info['grade'] = "📈 一般"
                else:
                    basic_info['grade'] = "❌ 较差"
        except Exception as e:
            pass

    return basic_info

def show_detailed_evaluation_report(log_dir: Path):
    """显示详细的评估报告"""
    print(f"\n📊 详细评估报告 - {log_dir.name}")
    print("=" * 80)

    # 基本信息
    final_model = log_dir / "final_model.zip"
    monitor_file = log_dir / "monitor.csv"

    print(f"📅 训练时间: {log_dir.name}")
    print(f"💾 模型状态: {'✅ 已保存' if final_model.exists() else '❌ 未完成'}")
    print(f"📁 日志路径: {log_dir}")

    # 训练数据分析
    if monitor_file.exists():
        try:
            import pandas as pd
            df = pd.read_csv(monitor_file, comment='#')

            if not df.empty and 'r' in df.columns:
                print(f"\n📈 训练统计:")
                print(f"  • 总回合数: {len(df)}")
                print(f"  • 平均奖励: {df['r'].mean():.2f}")
                print(f"  • 最高奖励: {df['r'].max():.2f}")
                print(f"  • 最低奖励: {df['r'].min():.2f}")
                print(f"  • 奖励标准差: {df['r'].std():.2f}")

                # 训练趋势分析
                if len(df) >= 10:
                    recent_avg = df['r'].tail(10).mean()
                    early_avg = df['r'].head(10).mean()
                    improvement = recent_avg - early_avg

                    print(f"\n📊 学习趋势:")
                    print(f"  • 前期平均奖励: {early_avg:.2f}")
                    print(f"  • 后期平均奖励: {recent_avg:.2f}")
                    print(f"  • 改进幅度: {improvement:+.2f}")

                    if improvement > 50:
                        trend = "🚀 显著提升"
                    elif improvement > 0:
                        trend = "📈 稳步改善"
                    elif improvement > -50:
                        trend = "📊 基本稳定"
                    else:
                        trend = "📉 有所下降"
                    print(f"  • 学习效果: {trend}")

                # 效果评估
                avg_reward = df['r'].mean()
                print(f"\n🎯 效果评估:")

                if avg_reward >= 400:
                    print("  🏆 优秀模型 - 适合比赛使用")
                    print("  • 预期表现: 车辆流畅通行，等待时间短")
                    print("  • 建议: 可以直接使用或继续优化")
                elif avg_reward >= 200:
                    print("  👍 良好模型 - 日常使用效果不错")
                    print("  • 预期表现: 通行效率有明显改善")
                    print("  • 建议: 可以使用，或尝试更长时间训练")
                elif avg_reward >= 0:
                    print("  📈 一般模型 - 有一定效果")
                    print("  • 预期表现: 比固定信号灯略好")
                    print("  • 建议: 增加训练步数或调整参数")
                else:
                    print("  ❌ 较差模型 - 需要重新训练")
                    print("  • 预期表现: 可能比固定信号灯更差")
                    print("  • 建议: 检查参数设置，重新训练")

        except Exception as e:
            print(f"❌ 数据分析失败: {e}")
    else:
        print("\n❌ 没有找到训练数据文件")

    # 操作建议
    print(f"\n💡 操作建议:")
    if final_model.exists():
        print(f"  🧪 测试模型: python start_local.py (选择4)")
        print(f"  📊 查看曲线: tensorboard --logdir {log_dir}")
    print(f"  🗑️  删除记录: 手动删除 {log_dir} 文件夹")

    print("=" * 80)
    input("\n按回车键返回...")

def advanced_training():
    """高级训练选项"""
    print("\n🔧 高级训练选项")
    print("-" * 40)
    
    # 获取训练步数
    try:
        timesteps = input("训练步数 (默认200000): ").strip()
        timesteps = int(timesteps) if timesteps else 200000
    except ValueError:
        timesteps = 200000
        
    # 获取学习率
    try:
        lr = input("学习率 (默认3e-4): ").strip()
        lr = float(lr) if lr else 3e-4
    except ValueError:
        lr = 3e-4
        
    # 是否启用GUI
    gui_choice = input("启用可视化? (y/N): ").strip().lower()
    gui_mode = gui_choice in ['y', 'yes']
    
    print(f"\n配置确认:")
    print(f"  训练步数: {timesteps:,}")
    print(f"  学习率: {lr}")
    print(f"  可视化: {'是' if gui_mode else '否'}")
    
    confirm = input("\n确认开始训练? (y/N): ").strip().lower()
    if confirm in ['y', 'yes']:
        cmd = [
            sys.executable, "train_local.py",
            "--mode", "train",
            "--timesteps", str(timesteps),
            "--learning-rate", str(lr)
        ]
        
        if gui_mode:
            cmd.append("--gui")
            
        try:
            subprocess.run(cmd, check=True)
            print("\n✅ 训练完成!")
        except subprocess.CalledProcessError as e:
            print(f"\n❌ 训练失败: {e}")
        except KeyboardInterrupt:
            print("\n⏹️ 训练被用户中断")

def run_competition_training():
    """运行比赛模式训练"""
    print("\n🏆 比赛模式训练")
    print("=" * 50)
    print("专为比赛优化的高质量训练模式")
    print("• 训练步数: 500,000步（约5-8小时）")
    print("• 学习率: 3e-4（稳定优化）")
    print("• 评估标准: 比赛官方指标")
    print("• 自动生成比赛评估报告")
    print()

    print("🎯 比赛评估指标:")
    print("  • 平均车辆状态 (30%权重)")
    print("  • 平均车辆密度 (30%权重)")
    print("  • 平均车速 (40%权重)")
    print()

    print("💰 奖项设置:")
    print("  🥇 一等奖: 50,000元（冠军）+ 20,000元（亚军）+ 20,000元（季军）")
    print("  🥈 二等奖: 10,000元 × 3名")
    print("  🥉 三等奖: 5,000元 × 6名")
    print()

    # 选择训练模式
    print("请选择比赛训练模式:")
    print("  1. 🚀 后台训练（推荐，速度最快）")
    print("  2. 👀 可视化训练（观察学习过程）")

    mode_choice = input("\n请选择 (1-2): ").strip()

    if mode_choice not in ['1', '2']:
        print("❌ 无效选择")
        return

    gui_mode = (mode_choice == '2')

    print(f"\n🎯 开始比赛模式训练...")
    print(f"📊 训练步数: 500,000")
    print(f"🎮 可视化: {'开启' if gui_mode else '关闭'}")
    print(f"⏱️  预计时间: {'6-8小时' if gui_mode else '4-6小时'}")

    confirm = input("\n确认开始比赛训练? (y/N): ").strip().lower()
    if confirm in ['y', 'yes']:
        cmd = [
            sys.executable, "train_local.py",
            "--mode", "train",
            "--scenario", "competition",
            "--timesteps", "500000",
            "--learning-rate", "3e-4"
        ]

        if gui_mode:
            cmd.append("--gui")

        try:
            print("\n🚀 比赛训练开始...")
            print("💡 提示: 训练完成后会自动生成比赛评估报告")
            subprocess.run(cmd, check=True)
            print("\n🏆 比赛训练完成!")
            print("📊 请查看训练日志中的比赛评估报告")
        except subprocess.CalledProcessError as e:
            print(f"\n❌ 训练失败: {e}")
        except KeyboardInterrupt:
            print("\n⏹️ 训练被用户中断")

def show_help():
    """显示帮助信息"""
    print("\n❓ 帮助说明")
    print("=" * 50)
    print("🎯 使用流程:")
    print("  1. 首次使用建议选择'可视化训练'观察学习过程")
    print("  2. 参加比赛选择'比赛模式训练'获得最佳效果")
    print("  3. 训练完成后可以测试模型效果")
    print("  4. 如果效果不满意，可以调整参数重新训练")
    print()
    print("⚙️ 训练模式选择:")
    print("  • 快速训练: 50,000 - 100,000步（30分钟-1小时）")
    print("  • 可视化训练: 观察AI学习过程，适合学习")
    print("  • 比赛模式: 500,000步（4-8小时），冲击奖金")
    print()
    print("📊 性能指标:")
    print("  • Episode Reward: 每回合奖励（越高越好）")
    print("  • Average Waiting Time: 平均等待时间（越低越好）")
    print()
    print("🔧 故障排除:")
    print("  • 如果SUMO启动失败，检查SUMO_HOME环境变量")
    print("  • 如果训练中断，程序会自动保存检查点")
    print("  • 如果效果不好，尝试增加训练步数或调整学习率")
    print()
    print("🏆 比赛信息:")
    print("  • 比赛主题: 智能交通信号灯调度")
    print("  • 评估指标: 车辆状态、密度、车速")
    print("  • 奖金总额: 超过15万元")
    print()
    print("📁 文件说明:")
    print("  • local_logs/: 训练日志和模型保存目录")
    print("  • train_local.py: 核心训练脚本")
    print("  • 使用说明.md: 详细使用说明")

def main():
    """主函数"""
    print_banner()
    
    # 检查依赖
    try:
        import stable_baselines3
        import gymnasium
        import torch
    except ImportError as e:
        print(f"❌ 缺少依赖库: {e}")
        print("请运行: pip install stable-baselines3[extra] gymnasium torch")
        return
        
    # 检查SUMO
    try:
        import sumolib
        sumolib.checkBinary('sumo')
    except:
        print("⚠️ 警告: SUMO未正确安装或配置")
        print("请确保SUMO已安装并设置了SUMO_HOME环境变量")
        
    while True:
        show_menu()
        
        try:
            choice = input("\n请输入选择 (0-7): ").strip()

            if choice == '0':
                print("👋 再见!")
                break
            elif choice == '1':
                run_training(gui_mode=False)
            elif choice == '2':
                run_training(gui_mode=True)
            elif choice == '3':
                run_competition_training()
            elif choice == '4':
                run_test()
            elif choice == '5':
                show_training_history()
            elif choice == '6':
                advanced_training()
            elif choice == '7':
                show_help()
            else:
                print("❌ 无效选择，请输入0-7")
                
        except KeyboardInterrupt:
            print("\n👋 再见!")
            break
        except Exception as e:
            print(f"❌ 发生错误: {e}")

if __name__ == "__main__":
    main()
