#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
项目配置管理
"""

import os
import yaml
from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import dataclass, asdict
import logging

logger = logging.getLogger(__name__)


@dataclass
class TrainingConfig:
    """训练配置"""
    # 基础参数
    timesteps: int = 200000
    learning_rate: float = 3e-4
    batch_size: int = 64
    n_steps: int = 2048
    n_epochs: int = 10
    
    # PPO参数
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    
    # 环境参数
    max_steps: int = 3600
    num_agents: int = 2
    
    # 保存和日志
    save_freq: int = 10000
    log_interval: int = 100
    eval_freq: int = 50000


@dataclass
class RewardConfig:
    """奖励函数配置"""
    # 权重配置
    waiting_time: float = 0.35
    queue_length: float = 0.25
    speed: float = 0.15
    throughput: float = 0.10
    switch_penalty: float = 0.03
    balance: float = 0.05
    trend: float = 0.02
    emergency: float = 0.05
    coordination: float = 0.10
    
    # 归一化参数
    max_waiting_time: float = 120.0
    max_queue_length: float = 20.0
    max_speed: float = 15.0
    
    # 惩罚阈值
    emergency_waiting_threshold: float = 120.0
    severe_waiting_threshold: float = 80.0
    moderate_waiting_threshold: float = 60.0


@dataclass
class EnvironmentConfig:
    """环境配置"""
    # SUMO配置
    sumo_binary: str = "sumo"
    sumo_gui_binary: str = "sumo-gui"
    step_length: float = 1.0
    
    # 连接管理
    connection_timeout: int = 30
    max_connection_retries: int = 5
    retry_delay: int = 10
    
    # 观测空间
    observation_dim: int = 25
    history_length: int = 10
    
    # 动作空间
    num_actions: int = 4
    min_phase_duration: int = 10
    max_phase_duration: int = 120


@dataclass
class ScenarioConfig:
    """场景配置"""
    name: str = "competition"
    path: str = "scenarios/competition"
    sumocfg_file: str = "config.sumocfg"
    additional_file: str = "additional.xml"
    network_file: str = "network.net.xml"
    route_file: str = "routes.rou.xml"
    
    # 交通信号灯ID
    tls_ids: list = None
    
    def __post_init__(self):
        if self.tls_ids is None:
            self.tls_ids = ["J_cross", "J_t"]


class ConfigManager:
    """配置管理器"""
    
    def __init__(self, config_file: Optional[str] = None):
        self.config_file = config_file or "config.yaml"
        self.config_path = Path(self.config_file)
        
        # 默认配置
        self.training = TrainingConfig()
        self.reward = RewardConfig()
        self.environment = EnvironmentConfig()
        self.scenario = ScenarioConfig()
        
        # 加载配置文件
        self.load_config()
    
    def load_config(self):
        """加载配置文件"""
        if self.config_path.exists():
            try:
                with open(self.config_path, 'r', encoding='utf-8') as f:
                    config_data = yaml.safe_load(f)
                
                if config_data:
                    # 更新配置
                    if 'training' in config_data:
                        self._update_dataclass(self.training, config_data['training'])
                    if 'reward' in config_data:
                        self._update_dataclass(self.reward, config_data['reward'])
                    if 'environment' in config_data:
                        self._update_dataclass(self.environment, config_data['environment'])
                    if 'scenario' in config_data:
                        self._update_dataclass(self.scenario, config_data['scenario'])
                
                logger.info(f"配置已从 {self.config_file} 加载")
                
            except Exception as e:
                logger.warning(f"加载配置文件失败: {e}，使用默认配置")
        else:
            logger.info("配置文件不存在，使用默认配置")
            self.save_config()  # 保存默认配置
    
    def save_config(self):
        """保存配置文件"""
        try:
            config_data = {
                'training': asdict(self.training),
                'reward': asdict(self.reward),
                'environment': asdict(self.environment),
                'scenario': asdict(self.scenario)
            }
            
            with open(self.config_path, 'w', encoding='utf-8') as f:
                yaml.dump(config_data, f, default_flow_style=False, allow_unicode=True)
            
            logger.info(f"配置已保存到 {self.config_file}")
            
        except Exception as e:
            logger.error(f"保存配置文件失败: {e}")
    
    def _update_dataclass(self, obj, data: Dict[str, Any]):
        """更新数据类对象"""
        for key, value in data.items():
            if hasattr(obj, key):
                setattr(obj, key, value)
    
    def get_reward_weights(self) -> Dict[str, float]:
        """获取奖励权重字典"""
        return {
            'waiting_time': self.reward.waiting_time,
            'queue_length': self.reward.queue_length,
            'speed': self.reward.speed,
            'throughput': self.reward.throughput,
            'switch_penalty': self.reward.switch_penalty,
            'balance': self.reward.balance,
            'trend': self.reward.trend,
            'emergency': self.reward.emergency,
            'coordination': self.reward.coordination
        }
    
    def get_training_params(self) -> Dict[str, Any]:
        """获取训练参数字典"""
        return asdict(self.training)
    
    def get_environment_params(self) -> Dict[str, Any]:
        """获取环境参数字典"""
        return asdict(self.environment)
    
    def update_reward_weights(self, weights: Dict[str, float]):
        """更新奖励权重"""
        for key, value in weights.items():
            if hasattr(self.reward, key):
                setattr(self.reward, key, value)
        self.save_config()
    
    def update_training_params(self, params: Dict[str, Any]):
        """更新训练参数"""
        for key, value in params.items():
            if hasattr(self.training, key):
                setattr(self.training, key, value)
        self.save_config()


# 全局配置实例
config_manager = ConfigManager()


def get_config() -> ConfigManager:
    """获取配置管理器实例"""
    return config_manager


def load_scenario_config(scenario_name: str) -> ScenarioConfig:
    """加载特定场景配置"""
    scenario_config = ScenarioConfig()
    scenario_config.name = scenario_name
    scenario_config.path = f"scenarios/{scenario_name}"
    
    # 根据场景名称设置特定参数
    if scenario_name == "competition":
        scenario_config.tls_ids = ["J_cross", "J_t"]
    elif scenario_name == "2x2_grid":
        scenario_config.tls_ids = ["A0", "A1", "B0", "B1"]
    
    return scenario_config
