#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
稳定训练脚本 - 带有重试和错误恢复机制
"""

import os
import sys
import time
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, Any

# 添加项目根目录到Python路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from models.advanced_env import AdvancedTrafficEnv
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class StableTrainingCallback(BaseCallback):
    """稳定训练回调，处理连接断开等问题"""
    
    def __init__(self, check_freq: int = 100):
        super().__init__()
        self.check_freq = check_freq
        self.connection_errors = 0
        self.max_connection_errors = 5
        
    def _on_step(self) -> bool:
        # 每隔一定步数检查连接状态
        if self.n_calls % self.check_freq == 0:
            try:
                # 检查环境是否正常
                if hasattr(self.training_env, 'envs'):
                    env = self.training_env.envs[0]
                else:
                    env = self.training_env
                    
                if hasattr(env, '_is_connection_alive'):
                    if not env._is_connection_alive():
                        logger.warning("检测到连接断开")
                        self.connection_errors += 1
                        
                        if self.connection_errors >= self.max_connection_errors:
                            logger.error("连接错误过多，停止训练")
                            return False
                            
            except Exception as e:
                logger.warning(f"连接检查失败: {e}")
                
        return True


def stable_train(scenario: str = "competition", 
                timesteps: int = 50000,
                gui: bool = False,
                max_retries: int = 3):
    """稳定训练函数，带重试机制"""
    
    for attempt in range(max_retries):
        logger.info(f"开始训练尝试 {attempt + 1}/{max_retries}")
        
        try:
            # 创建日志目录
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            log_dir = Path("local_logs") / f"stable_{timestamp}"
            log_dir.mkdir(parents=True, exist_ok=True)
            
            # 创建环境
            logger.info("创建环境...")
            base_env = AdvancedTrafficEnv(
                sumocfg_file=f"scenarios/{scenario}/config.sumocfg",
                tls_ids=["J_cross", "J_t"],  # 两个路口ID
                scenario_path=f"scenarios/{scenario}",
                render_mode='human' if gui else None,
                config={
                    'reward_weights': {
                        'waiting_time': 0.35,    # 增加等待时间权重
                        'queue_length': 0.25,    # 增加队列长度权重
                        'speed': 0.15,           # 减少速度权重
                        'throughput': 0.10,      # 减少通行量权重
                        'switch_penalty': 0.03,  # 减少切换惩罚
                        'balance': 0.05,         # 减少平衡权重
                        'trend': 0.02,           # 减少趋势权重
                        'emergency': 0.05        # 增加紧急情况权重
                    }
                }
            )

            # 用Monitor包装环境，生成评估所需的monitor.csv
            env = Monitor(base_env, str(log_dir / "monitor.csv"))
            
            logger.info("环境创建成功")
            
            # 创建模型
            logger.info("创建PPO模型...")
            model = PPO(
                "MlpPolicy",
                env,
                learning_rate=3e-4,
                n_steps=2048,
                batch_size=64,
                n_epochs=10,
                gamma=0.99,
                gae_lambda=0.95,
                clip_range=0.2,
                ent_coef=0.01,
                vf_coef=0.5,
                max_grad_norm=0.5,
                verbose=1,
                tensorboard_log=str(log_dir)
            )
            
            # 创建稳定训练回调
            callback = StableTrainingCallback(check_freq=100)
            
            # 开始训练
            logger.info(f"开始训练 {timesteps} 步...")
            start_time = time.time()
            
            model.learn(
                total_timesteps=timesteps,
                callback=callback,
                progress_bar=True
            )
            
            training_time = time.time() - start_time
            logger.info(f"训练完成，耗时: {training_time:.2f}秒")
            
            # 保存模型（同时保存为stable_model.zip和final_model.zip）
            stable_model_path = log_dir / "stable_model.zip"
            final_model_path = log_dir / "final_model.zip"

            model.save(str(stable_model_path))
            model.save(str(final_model_path))  # 为评估系统保存

            logger.info(f"模型已保存: {stable_model_path}")
            logger.info(f"评估模型已保存: {final_model_path}")

            # 关闭环境
            env.close()
            
            # 训练成功，返回
            return str(stable_model_path)
            
        except Exception as e:
            logger.error(f"训练尝试 {attempt + 1} 失败: {e}")
            
            # 清理资源
            try:
                if 'env' in locals():
                    env.close()
            except:
                pass
                
            # 如果不是最后一次尝试，等待一段时间再重试
            if attempt < max_retries - 1:
                wait_time = (attempt + 1) * 10  # 递增等待时间
                logger.info(f"等待 {wait_time} 秒后重试...")
                time.sleep(wait_time)
            else:
                logger.error("所有训练尝试都失败了")
                raise


def test_stable_model(model_path: str, gui: bool = True):
    """测试稳定训练的模型"""
    logger.info(f"测试模型: {model_path}")
    
    try:
        # 创建环境
        base_env = AdvancedTrafficEnv(
            sumocfg_file="scenarios/competition/config.sumocfg",
            tls_ids=["J_cross", "J_t"],
            scenario_path="scenarios/competition",
            render_mode='human' if gui else None
        )

        # 用Monitor包装环境
        env = Monitor(base_env)
        
        # 加载模型
        model = PPO.load(model_path, env=env)
        
        # 测试运行
        obs, _ = env.reset()
        total_reward = 0
        steps = 0
        
        logger.info("开始测试...")
        
        for step in range(1000):  # 测试1000步
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            steps += 1
            
            if terminated or truncated:
                logger.info(f"测试结束于步骤 {step}")
                break
                
            if step % 100 == 0:
                logger.info(f"测试步骤 {step}, 累计奖励: {total_reward:.2f}")
        
        logger.info(f"测试完成: {steps} 步, 总奖励: {total_reward:.2f}, 平均奖励: {total_reward/steps:.2f}")
        
        env.close()
        
    except Exception as e:
        logger.error(f"测试失败: {e}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="稳定训练脚本")
    parser.add_argument("--mode", choices=["train", "test"], default="train", help="模式")
    parser.add_argument("--scenario", default="competition", help="场景名称")
    parser.add_argument("--timesteps", type=int, default=50000, help="训练步数")
    parser.add_argument("--gui", action="store_true", help="使用GUI")
    parser.add_argument("--model-path", help="测试模型路径")
    parser.add_argument("--max-retries", type=int, default=3, help="最大重试次数")
    
    args = parser.parse_args()
    
    if args.mode == "train":
        try:
            model_path = stable_train(
                scenario=args.scenario,
                timesteps=args.timesteps,
                gui=args.gui,
                max_retries=args.max_retries
            )
            print(f"\n🎉 训练成功完成！")
            print(f"📁 模型路径: {model_path}")
            print(f"🧪 测试命令: python stable_train.py --mode test --model-path {model_path}")
            
        except Exception as e:
            print(f"\n❌ 训练失败: {e}")
            sys.exit(1)
            
    elif args.mode == "test":
        if not args.model_path:
            print("❌ 测试模式需要指定 --model-path")
            sys.exit(1)
            
        test_stable_model(args.model_path, args.gui)
