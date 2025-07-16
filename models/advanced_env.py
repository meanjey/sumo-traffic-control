#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
升级版多智能体交通环境
支持更复杂的观测空间、动作空间和协调机制
"""

import os
import sys
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import traci
import sumolib
from typing import Dict, List, Tuple, Any, Optional
import logging
from collections import deque
from models.advanced_controller import AdvancedTLSController, MultiAgentCoordinator

logger = logging.getLogger(__name__)


class AdvancedTrafficEnv(gym.Env):
    """升级版交通环境"""
    
    metadata = {'render_modes': ['human', 'rgb_array'], 'render_fps': 10}
    
    def __init__(self, 
                 sumocfg_file: str,
                 tls_ids: List[str],
                 scenario_path: str,
                 config: Dict[str, Any] = None,
                 render_mode: Optional[str] = None):
        super().__init__()
        
        self.sumocfg_file = sumocfg_file
        self.tls_ids = list(tls_ids)
        self.num_agents = len(self.tls_ids)
        self.scenario_path = scenario_path
        self.config = config or {}
        self.render_mode = render_mode
        
        # 环境配置
        self.max_steps = self.config.get('max_steps', 3600)
        self.step_count = 0
        self.episode_count = 0
        
        # 升级的观测和动作空间
        self.single_obs_dim = 25  # 20 + 5 (协调信号)
        self.num_actions = 4  # 扩展动作空间：保持、切换、延长、缩短
        
        # 定义观测空间（每个智能体25维观测）
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.num_agents * self.single_obs_dim,),
            dtype=np.float32
        )
        
        # 定义动作空间（每个智能体4个动作选择）
        self.action_space = spaces.MultiDiscrete([self.num_actions] * self.num_agents)
        
        # 控制器和协调器
        self.controllers = []
        self.coordinator = None
        self._sumo_started = False
        
        # 性能统计
        self.episode_rewards = []
        self.episode_metrics = []
        self.global_metrics = {
            'total_waiting_time': 0,
            'total_throughput': 0,
            'total_fuel_consumption': 0,
            'total_emissions': 0
        }
        
        # 自适应参数
        self.adaptive_params = {
            'reward_weights': {
                'waiting_time': -1.0,
                'queue_length': -0.5,
                'speed': 0.2,
                'throughput': 1.0,
                'coordination': 0.3
            },
            'min_phase_duration': 10,
            'max_phase_duration': 120
        }
        
        # 启动SUMO
        self._start_sumo()
        
    def _start_sumo(self):
        """启动SUMO仿真"""
        try:
            if self.render_mode == 'human':
                sumoBinary = sumolib.checkBinary('sumo-gui')
                sumo_cmd = [sumoBinary, "-c", self.sumocfg_file, "--start", "--quit-on-end", "--no-warnings"]
            else:
                sumoBinary = sumolib.checkBinary('sumo')
                sumo_cmd = [sumoBinary, "-c", self.sumocfg_file, "--quit-on-end", "--no-warnings", "--no-step-log"]
            
            traci.start(sumo_cmd)
            self._sumo_started = True
            logger.info("SUMO仿真启动成功")
            
        except Exception as e:
            logger.error(f"启动SUMO失败: {e}")
            raise
            
    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None) -> Tuple[np.ndarray, Dict]:
        """重置环境"""
        super().reset(seed=seed)
        
        try:
            # 重新加载仿真
            sumo_cmd = ["-c", self.sumocfg_file, "--quit-on-end", "--no-warnings"]
            if self.render_mode == 'human':
                sumo_cmd.extend(["--start"])
            else:
                sumo_cmd.append("--no-step-log")
            traci.load(sumo_cmd)
            
            # 重置计数器
            self.step_count = 0
            self.episode_count += 1
            
            # 获取交通信号灯ID
            available_tls = traci.trafficlight.getIDList()
            if available_tls:
                self.tls_ids = list(available_tls)[:self.num_agents]
                
            logger.info(f"第 {self.episode_count} 回合开始，管理 {len(self.tls_ids)} 个路口")
            
            # 创建升级版控制器
            additional_file = os.path.join(self.scenario_path, 'additional.xml')
            self.controllers = [
                AdvancedTLSController(tls_id, additional_file, self.config) 
                for tls_id in self.tls_ids
            ]
            
            # 创建协调器
            self.coordinator = MultiAgentCoordinator(self.controllers, self.config)
            
            # 重置所有控制器
            initial_observations = []
            for controller in self.controllers:
                obs = controller.reset()
                initial_observations.append(obs)
                
            # 获取协调信号并拼接观测
            enhanced_observations = self._enhance_observations_with_coordination(initial_observations)
            observation = np.concatenate(enhanced_observations).astype(np.float32)
            
            # 重置性能统计
            self.episode_rewards = []
            self.global_metrics = {
                'total_waiting_time': 0,
                'total_throughput': 0,
                'total_fuel_consumption': 0,
                'total_emissions': 0
            }
            
            info = self._get_info()
            return observation, info
            
        except Exception as e:
            logger.error(f"环境重置失败: {e}")
            # 返回安全的默认观测
            observation = np.zeros(self.observation_space.shape, dtype=np.float32)
            info = {"error": str(e)}
            return observation, info
            
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """执行一步"""
        try:
            # 执行仿真步
            traci.simulationStep()
            self.step_count += 1
            
            # 获取当前观测
            current_observations = []
            for controller in self.controllers:
                obs = controller.get_enhanced_observation()
                current_observations.append(obs)
                
            # 获取协调建议
            coordinated_actions = self.coordinator.suggest_coordinated_actions(current_observations)
            
            # 结合RL动作和协调建议
            final_actions = self._combine_actions(action, coordinated_actions)
            
            # 执行动作并收集奖励
            rewards = []
            for i, (controller, agent_action) in enumerate(zip(self.controllers, final_actions)):
                reward = controller.update(self.step_count, agent_action)
                rewards.append(reward)
                
            # 计算全局奖励
            individual_reward = np.mean(rewards)
            coordination_reward = self._calculate_coordination_reward()
            global_reward = individual_reward + coordination_reward
            
            # 更新全局指标
            self._update_global_metrics()
            
            # 获取新观测
            new_observations = []
            for controller in self.controllers:
                obs = controller.get_enhanced_observation()
                new_observations.append(obs)
                
            # 增强观测（添加协调信号）
            enhanced_observations = self._enhance_observations_with_coordination(new_observations)
            observation = np.concatenate(enhanced_observations).astype(np.float32)
            
            # 检查终止条件
            terminated = self.step_count >= self.max_steps
            truncated = False
            
            # 自适应调整参数
            if self.step_count % 100 == 0:
                self._adaptive_parameter_adjustment()
                
            # 记录奖励
            self.episode_rewards.append(global_reward)
            
            info = self._get_info()
            info.update({
                'individual_rewards': rewards,
                'coordination_reward': coordination_reward,
                'global_metrics': self.global_metrics.copy()
            })
            
            return observation, global_reward, terminated, truncated, info
            
        except Exception as e:
            logger.error(f"环境步骤执行失败: {e}")
            # 返回安全的默认值
            observation = np.zeros(self.observation_space.shape, dtype=np.float32)
            reward = -100.0
            terminated = True
            truncated = False
            info = {"error": str(e)}
            return observation, reward, terminated, truncated, info
            
    def _enhance_observations_with_coordination(self, observations: List[np.ndarray]) -> List[np.ndarray]:
        """用协调信号增强观测"""
        enhanced_observations = []
        
        for i, obs in enumerate(observations):
            # 获取协调信号
            coord_signal = self.coordinator.get_coordination_signal(i, obs)
            
            # 拼接原始观测和协调信号
            enhanced_obs = np.concatenate([obs, coord_signal])
            enhanced_observations.append(enhanced_obs)
            
        return enhanced_observations
        
    def _combine_actions(self, rl_actions: np.ndarray, coordinated_actions: List[int]) -> List[int]:
        """结合RL动作和协调建议"""
        final_actions = []
        
        for i in range(self.num_agents):
            rl_action = rl_actions[i] if i < len(rl_actions) else 0
            coord_action = coordinated_actions[i] if i < len(coordinated_actions) else 0
            
            # 简单的动作融合策略
            if coord_action == 1 and rl_action == 0:  # 协调建议切换但RL建议保持
                final_action = 1 if np.random.random() < 0.3 else 0  # 30%概率采用协调建议
            else:
                final_action = rl_action
                
            # 将扩展动作映射回基础动作
            if final_action >= 2:
                final_action = 1  # 延长和缩短都映射为切换
                
            final_actions.append(final_action)
            
        return final_actions
        
    def _calculate_coordination_reward(self) -> float:
        """计算协调奖励"""
        try:
            coordination_reward = 0.0
            
            # 获取所有路口的状态
            all_waiting_times = []
            all_vehicle_counts = []
            
            for controller in self.controllers:
                metrics = controller.get_performance_metrics()
                all_waiting_times.append(metrics.get('avg_waiting_time', 0))
                all_vehicle_counts.append(metrics.get('total_vehicles', 0))
                
            if len(all_waiting_times) > 1:
                # 奖励等待时间的均衡性
                waiting_std = np.std(all_waiting_times)
                coordination_reward -= waiting_std * 0.1
                
                # 奖励车辆分布的均衡性
                vehicle_std = np.std(all_vehicle_counts)
                coordination_reward -= vehicle_std * 0.05
                
            return coordination_reward * self.adaptive_params['reward_weights']['coordination']
            
        except Exception as e:
            logger.error(f"计算协调奖励失败: {e}")
            return 0.0
            
    def _update_global_metrics(self):
        """更新全局指标"""
        try:
            # 累计等待时间
            total_waiting = 0
            for controller in self.controllers:
                metrics = controller.get_performance_metrics()
                total_waiting += metrics.get('avg_waiting_time', 0)
            self.global_metrics['total_waiting_time'] += total_waiting
            
            # 累计通行量
            departed = traci.simulation.getDepartedNumber()
            arrived = traci.simulation.getArrivedNumber()
            self.global_metrics['total_throughput'] = arrived
            
            # 燃油消耗和排放（如果SUMO支持）
            try:
                fuel_consumption = traci.simulation.getFuelConsumption()
                co2_emission = traci.simulation.getCO2Emission()
                self.global_metrics['total_fuel_consumption'] += fuel_consumption
                self.global_metrics['total_emissions'] += co2_emission
            except:
                pass  # 某些SUMO版本可能不支持这些指标
                
        except Exception as e:
            logger.error(f"更新全局指标失败: {e}")
            
    def _adaptive_parameter_adjustment(self):
        """自适应参数调整"""
        try:
            # 基于最近的性能调整奖励权重
            if len(self.episode_rewards) >= 50:
                recent_rewards = self.episode_rewards[-50:]
                avg_reward = np.mean(recent_rewards)
                reward_trend = np.mean(np.diff(recent_rewards))
                
                # 如果奖励趋势下降，调整权重
                if reward_trend < -0.1:
                    # 增加等待时间惩罚权重
                    self.adaptive_params['reward_weights']['waiting_time'] *= 1.1
                    self.adaptive_params['reward_weights']['waiting_time'] = max(
                        self.adaptive_params['reward_weights']['waiting_time'], -2.0
                    )
                elif reward_trend > 0.1:
                    # 减少等待时间惩罚权重
                    self.adaptive_params['reward_weights']['waiting_time'] *= 0.95
                    self.adaptive_params['reward_weights']['waiting_time'] = min(
                        self.adaptive_params['reward_weights']['waiting_time'], -0.5
                    )
                    
                logger.debug(f"自适应调整奖励权重: {self.adaptive_params['reward_weights']}")
                
        except Exception as e:
            logger.error(f"自适应参数调整失败: {e}")
            
    def _get_info(self) -> Dict[str, Any]:
        """获取环境信息"""
        info = {
            "step_count": self.step_count,
            "episode_count": self.episode_count,
            "num_agents": self.num_agents,
            "traffic_light_ids": self.tls_ids,
            "episode_length": len(self.episode_rewards),
            "total_reward": sum(self.episode_rewards) if self.episode_rewards else 0.0,
            "average_reward": np.mean(self.episode_rewards) if self.episode_rewards else 0.0,
            "adaptive_params": self.adaptive_params.copy()
        }
        
        # 添加控制器性能指标
        if self.controllers:
            controller_metrics = []
            for controller in self.controllers:
                controller_metrics.append(controller.get_performance_metrics())
            info["controller_metrics"] = controller_metrics
            
        return info
        
    def render(self, mode: str = 'human'):
        """渲染环境"""
        if mode == 'human':
            # SUMO GUI自己处理渲染
            pass
        elif mode == 'rgb_array':
            # 返回RGB数组（如果需要）
            return np.zeros((600, 800, 3), dtype=np.uint8)
            
    def close(self):
        """关闭环境"""
        try:
            if self._sumo_started:
                traci.close()
                self._sumo_started = False
                logger.info("环境已关闭")
        except Exception as e:
            logger.error(f"关闭环境失败: {e}")
            
    def get_episode_summary(self) -> Dict[str, Any]:
        """获取回合总结"""
        summary = {
            "episode_count": self.episode_count,
            "total_steps": self.step_count,
            "total_reward": sum(self.episode_rewards) if self.episode_rewards else 0.0,
            "average_reward": np.mean(self.episode_rewards) if self.episode_rewards else 0.0,
            "global_metrics": self.global_metrics.copy()
        }
        
        if self.controllers:
            # 汇总所有控制器的性能
            total_switches = sum(c.get_performance_metrics().get('phase_switches', 0) for c in self.controllers)
            avg_waiting = np.mean([c.get_performance_metrics().get('avg_waiting_time', 0) for c in self.controllers])
            
            summary.update({
                "total_phase_switches": total_switches,
                "average_waiting_time": avg_waiting,
                "switches_per_step": total_switches / max(1, self.step_count)
            })
            
        return summary
