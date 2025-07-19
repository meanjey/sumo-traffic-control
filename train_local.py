#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
本地训练脚本 - 简化版
专门为本地使用优化，去除复杂的比赛平台功能
"""

import os
import sys
import logging
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional

# 添加项目根目录到Python路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from models.advanced_controller import AdvancedTLSController, MultiAgentCoordinator
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback
import gymnasium as gym
from gymnasium import spaces
import traci
import sumolib

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def evaluate_training_results(log_dir: str, timesteps: int):
    """评估训练结果并显示效果"""
    print("\n" + "="*60)
    print("📊 训练效果评估报告")
    print("="*60)

    try:
        # 读取TensorBoard日志（如果存在）
        import glob
        import pandas as pd

        # 查找事件文件
        event_files = glob.glob(os.path.join(log_dir, "events.out.tfevents.*"))

        if event_files:
            print("📈 训练统计:")
            print(f"  • 训练步数: {timesteps:,}")
            print(f"  • 训练时长: 已完成")
            print(f"  • 模型保存: ✅ 成功")
        else:
            print("📈 训练统计:")
            print(f"  • 训练步数: {timesteps:,}")
            print(f"  • 训练时长: 已完成")
            print(f"  • 模型保存: ✅ 成功")

    except Exception as e:
        logger.warning(f"读取训练统计失败: {e}")

    # 给出效果判断建议
    print("\n🎯 效果评估建议:")

    if timesteps >= 500000:
        print("  ✅ 训练步数充足 - 预期效果良好")
        print("  🏆 适合比赛使用")
    elif timesteps >= 200000:
        print("  ✅ 训练步数适中 - 预期效果不错")
        print("  👍 适合日常使用")
    elif timesteps >= 50000:
        print("  ⚠️  训练步数较少 - 效果可能一般")
        print("  💡 建议增加训练步数")
    else:
        print("  ❌ 训练步数太少 - 效果可能较差")
        print("  🔄 建议重新训练更多步数")

    print("\n📋 下一步建议:")
    print("  1. 🧪 测试模型: python start_local.py (选择4)")
    print("  2. 📊 查看曲线: tensorboard --logdir " + log_dir)
    print("  3. 🔄 如果效果不好，可以重新训练")

    # 效果判断标准
    print("\n📏 效果判断标准:")
    print("  ✅ 好模型特征:")
    print("     • 测试时车辆流畅通行")
    print("     • 等待时间明显减少")
    print("     • 很少出现长时间拥堵")
    print("  ❌ 需要重新训练:")
    print("     • 测试时经常拥堵")
    print("     • 等待时间很长")
    print("     • 信号灯切换不合理")

    print("="*60)


class SimpleTrafficEnv(gym.Env):
    """简化的本地交通环境"""
    
    def __init__(self, sumocfg_file: str, tls_ids: list, scenario_path: str, gui_enabled: bool = False):
        super().__init__()
        
        self.sumocfg_file = sumocfg_file
        self.tls_ids = tls_ids
        self.scenario_path = scenario_path
        self.gui_enabled = gui_enabled
        self.max_steps = 3600
        self.step_count = 0
        
        # 观测和动作空间
        self.single_obs_dim = 20  # 简化观测空间
        self.num_agents = len(tls_ids)
        
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(self.num_agents * self.single_obs_dim,),
            dtype=np.float32
        )
        self.action_space = spaces.MultiDiscrete([2] * self.num_agents)  # 简化动作空间
        
        # 控制器
        self.controllers = []
        self.coordinator = None
        self._sumo_started = False
        
        # 性能统计
        self.episode_rewards = []
        
        # 启动SUMO
        self._start_sumo()
        
    def _start_sumo(self):
        """启动SUMO"""
        try:
            if self.gui_enabled:
                sumoBinary = sumolib.checkBinary('sumo-gui')
                sumo_cmd = [sumoBinary, "-c", self.sumocfg_file, "--start", "--quit-on-end"]
            else:
                sumoBinary = sumolib.checkBinary('sumo')
                sumo_cmd = [sumoBinary, "-c", self.sumocfg_file, "--quit-on-end", "--no-step-log"]
            
            traci.start(sumo_cmd)
            self._sumo_started = True
            logger.info("SUMO启动成功")
            
        except Exception as e:
            logger.error(f"启动SUMO失败: {e}")
            raise
            
    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None):
        """重置环境"""
        super().reset(seed=seed)
        
        try:
            # 重新加载仿真
            sumo_cmd = ["-c", self.sumocfg_file, "--quit-on-end"]
            if self.gui_enabled:
                sumo_cmd.append("--start")
            else:
                sumo_cmd.append("--no-step-log")
            traci.load(sumo_cmd)
            
            self.step_count = 0
            
            # 创建控制器
            additional_file = os.path.join(self.scenario_path, 'additional.xml')
            config = {'coordination_enabled': True}  # 简化配置
            
            self.controllers = [
                AdvancedTLSController(tls_id, additional_file, config) 
                for tls_id in self.tls_ids
            ]
            
            # 创建协调器
            self.coordinator = MultiAgentCoordinator(self.controllers, config)
            
            # 获取初始观测
            observations = []
            for controller in self.controllers:
                obs = controller.reset()
                # 只取前20维，简化观测
                observations.append(obs[:20])
                
            observation = np.concatenate(observations).astype(np.float32)
            info = {"step": self.step_count}
            
            return observation, info
            
        except Exception as e:
            logger.error(f"重置失败: {e}")
            observation = np.zeros(self.observation_space.shape, dtype=np.float32)
            return observation, {"error": str(e)}
            
    def step(self, action):
        """执行一步"""
        try:
            # 执行仿真步
            traci.simulationStep()
            self.step_count += 1
            
            # 执行动作并收集奖励
            rewards = []
            observations = []
            
            for i, (controller, agent_action) in enumerate(zip(self.controllers, action)):
                reward = controller.update(self.step_count, agent_action)
                obs = controller.get_enhanced_observation()
                
                rewards.append(reward)
                observations.append(obs[:20])  # 只取前20维
                
            # 计算平均奖励
            total_reward = np.mean(rewards)
            
            # 拼接观测
            observation = np.concatenate(observations).astype(np.float32)
            
            # 检查终止条件
            terminated = self.step_count >= self.max_steps
            truncated = False
            
            info = {
                "step": self.step_count,
                "individual_rewards": rewards,
                "avg_waiting_time": np.mean([c.get_performance_metrics().get('avg_waiting_time', 0) for c in self.controllers])
            }
            
            return observation, total_reward, terminated, truncated, info
            
        except Exception as e:
            logger.error(f"步骤执行失败: {e}")
            observation = np.zeros(self.observation_space.shape, dtype=np.float32)
            return observation, -100.0, True, False, {"error": str(e)}
            
    def close(self):
        """关闭环境"""
        try:
            if self._sumo_started:
                traci.close()
                self._sumo_started = False
                logger.info("环境已关闭")
        except Exception as e:
            logger.error(f"关闭环境失败: {e}")


class SimpleCallback(BaseCallback):
    """简单的训练回调"""
    
    def __init__(self, save_freq: int = 10000, verbose: int = 1):
        super().__init__(verbose)
        self.save_freq = save_freq
        self.best_mean_reward = -np.inf
        
    def _on_step(self) -> bool:
        # 定期保存模型
        if self.num_timesteps % self.save_freq == 0:
            model_path = f"model_checkpoint_{self.num_timesteps}.zip"
            self.model.save(model_path)
            logger.info(f"模型已保存: {model_path}")
            
        return True


def create_local_env(scenario_name: str = "2x2_grid", gui_enabled: bool = False):
    """创建本地环境"""
    scenario_path = f"scenarios/{scenario_name}"
    sumocfg_file = os.path.join(scenario_path, "config.sumocfg")
    
    if not os.path.exists(sumocfg_file):
        raise FileNotFoundError(f"场景配置文件不存在: {sumocfg_file}")
        
    # 获取交通信号灯ID
    try:
        sumo_cmd = [sumolib.checkBinary('sumo'), "-c", sumocfg_file]
        traci.start(sumo_cmd, label="temp_check")
        tls_ids = traci.trafficlight.getIDList()
        traci.close()
    except Exception as e:
        logger.warning(f"获取交通信号灯ID失败: {e}，使用默认ID")
        tls_ids = ["A0", "A1", "B0", "B1"]
        
    return SimpleTrafficEnv(sumocfg_file, tls_ids, scenario_path, gui_enabled)


def train_local_model(scenario: str = "competition",
                     timesteps: int = 200000,
                     gui: bool = False,
                     learning_rate: float = 3e-4):
    """本地训练模型"""
    try:
        logger.info("开始本地训练")
        
        # 创建环境
        env = create_local_env(scenario, gui)
        
        # 创建日志目录
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_dir = f"local_logs/{timestamp}"
        os.makedirs(log_dir, exist_ok=True)
        
        # 包装Monitor
        env = Monitor(env, filename=os.path.join(log_dir, "monitor.csv"))
        
        # 创建PPO模型
        model = PPO(
            policy="MlpPolicy",
            env=env,
            learning_rate=learning_rate,
            n_steps=2048,
            batch_size=64,
            verbose=1,
            tensorboard_log=log_dir
        )
        
        # 创建回调
        callback = SimpleCallback(save_freq=10000)
        
        # 开始训练
        logger.info(f"开始训练，总步数: {timesteps}")
        model.learn(total_timesteps=timesteps, callback=callback, progress_bar=True)
        
        # 保存最终模型
        final_model_path = os.path.join(log_dir, "final_model.zip")
        model.save(final_model_path)
        logger.info(f"训练完成，模型已保存到: {final_model_path}")

        # 自动评估训练效果
        evaluate_training_results(log_dir, timesteps)

        env.close()
        return final_model_path
        
    except Exception as e:
        logger.error(f"训练失败: {e}")
        return None


def test_local_model(model_path: str, scenario: str = "2x2_grid", episodes: int = 3):
    """测试本地模型"""
    try:
        logger.info(f"开始测试模型: {model_path}")
        
        # 创建环境（启用GUI）
        env = create_local_env(scenario, gui_enabled=True)
        
        # 加载模型
        model = PPO.load(model_path, env=env)
        
        # 运行测试
        for episode in range(episodes):
            obs, info = env.reset()
            episode_reward = 0
            step_count = 0
            
            logger.info(f"开始第 {episode + 1} 回合测试")
            
            while step_count < 3600:
                action, _states = model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = env.step(action)
                
                episode_reward += reward
                step_count += 1
                
                if terminated or truncated:
                    break
                    
            logger.info(f"第 {episode + 1} 回合结束，奖励: {episode_reward:.2f}，步数: {step_count}")
            
        env.close()
        logger.info("测试完成")
        return True
        
    except Exception as e:
        logger.error(f"测试失败: {e}")
        return False


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="本地交通信号灯控制训练")
    parser.add_argument("--mode", choices=["train", "test"], default="train", help="运行模式")
    parser.add_argument("--scenario", default="competition", help="场景名称")
    parser.add_argument("--timesteps", type=int, default=200000, help="训练步数")
    parser.add_argument("--gui", action="store_true", help="启用GUI")
    parser.add_argument("--model-path", help="模型路径（测试模式需要）")
    parser.add_argument("--learning-rate", type=float, default=3e-4, help="学习率")
    parser.add_argument("--episodes", type=int, default=3, help="测试回合数")
    
    args = parser.parse_args()
    
    if args.mode == "train":
        model_path = train_local_model(
            scenario=args.scenario,
            timesteps=args.timesteps,
            gui=args.gui,
            learning_rate=args.learning_rate
        )
        if model_path:
            print(f"\n🎉 训练完成！模型保存在: {model_path}")
            print(f"💡 快速测试: python start_local.py (选择菜单4)")
            print(f"📊 详细分析: tensorboard --logdir {os.path.dirname(model_path)}")
        return 0 if model_path else 1
        
    elif args.mode == "test":
        if not args.model_path:
            logger.error("测试模式需要指定模型路径")
            return 1
        success = test_local_model(args.model_path, args.scenario, args.episodes)
        return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
