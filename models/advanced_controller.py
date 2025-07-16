#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
升级版交通信号灯控制器
包含更强大的神经网络架构和智能决策机制
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Any, Optional
import logging
from collections import deque
import xml.etree.ElementTree as ET
import traci

logger = logging.getLogger(__name__)


class AttentionLayer(nn.Module):
    """注意力机制层"""
    
    def __init__(self, input_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        
        self.query = nn.Linear(input_dim, hidden_dim)
        self.key = nn.Linear(input_dim, hidden_dim)
        self.value = nn.Linear(input_dim, hidden_dim)
        self.output = nn.Linear(hidden_dim, input_dim)
        
    def forward(self, x):
        # x shape: (batch_size, seq_len, input_dim)
        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)
        
        # 计算注意力权重
        attention_weights = F.softmax(torch.matmul(Q, K.transpose(-2, -1)) / np.sqrt(self.hidden_dim), dim=-1)
        
        # 应用注意力
        attended = torch.matmul(attention_weights, V)
        output = self.output(attended)
        
        return output + x  # 残差连接


class TemporalEncoder(nn.Module):
    """时序编码器"""
    
    def __init__(self, input_dim: int, hidden_dim: int = 128, num_layers: int = 2):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, dropout=0.1)
        self.attention = AttentionLayer(hidden_dim)
        self.output_proj = nn.Linear(hidden_dim, input_dim)
        
    def forward(self, x):
        # x shape: (batch_size, seq_len, input_dim)
        lstm_out, _ = self.lstm(x)
        attended = self.attention(lstm_out)
        output = self.output_proj(attended[:, -1, :])  # 取最后一个时间步
        return output


class MultiIntersectionNetwork(nn.Module):
    """多路口协调网络"""
    
    def __init__(self, 
                 single_obs_dim: int = 20,  # 升级后的单路口观测维度
                 num_intersections: int = 4,
                 hidden_dim: int = 256,
                 num_actions: int = 4):  # 扩展动作空间
        super().__init__()
        
        self.single_obs_dim = single_obs_dim
        self.num_intersections = num_intersections
        self.hidden_dim = hidden_dim
        self.num_actions = num_actions
        
        # 单路口特征提取器
        self.intersection_encoder = nn.Sequential(
            nn.Linear(single_obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU()
        )
        
        # 路口间注意力机制
        self.inter_attention = AttentionLayer(hidden_dim // 2)
        
        # 全局特征融合
        self.global_encoder = nn.Sequential(
            nn.Linear(num_intersections * (hidden_dim // 2), hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # 动作头（每个路口独立的动作预测）
        self.action_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim + hidden_dim // 2, hidden_dim // 2),
                nn.ReLU(),
                nn.Linear(hidden_dim // 2, num_actions)
            ) for _ in range(num_intersections)
        ])
        
        # 价值函数头
        self.value_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )
        
    def forward(self, observations):
        batch_size = observations.shape[0]
        
        # 重塑观测为 (batch_size, num_intersections, single_obs_dim)
        obs_reshaped = observations.view(batch_size, self.num_intersections, self.single_obs_dim)
        
        # 编码每个路口的特征
        intersection_features = []
        for i in range(self.num_intersections):
            features = self.intersection_encoder(obs_reshaped[:, i, :])
            intersection_features.append(features)
        
        # 堆叠特征 (batch_size, num_intersections, hidden_dim//2)
        stacked_features = torch.stack(intersection_features, dim=1)
        
        # 应用路口间注意力
        attended_features = self.inter_attention(stacked_features)
        
        # 全局特征融合
        global_features = self.global_encoder(attended_features.view(batch_size, -1))
        
        # 为每个路口预测动作
        actions = []
        for i in range(self.num_intersections):
            # 结合全局特征和局部特征
            combined_features = torch.cat([global_features, attended_features[:, i, :]], dim=1)
            action_logits = self.action_heads[i](combined_features)
            actions.append(action_logits)
        
        # 预测状态价值
        value = self.value_head(global_features)
        
        return torch.stack(actions, dim=1), value


class AdvancedTLSController:
    """升级版交通信号灯控制器"""
    
    def __init__(self, tls_id: str, additional_file_path: str, config: Dict[str, Any] = None):
        self.tls_id = tls_id
        self.additional_file_path = additional_file_path
        self.config = config or {}
        
        # 历史数据存储
        self.history_length = self.config.get('history_length', 10)
        self.observation_history = deque(maxlen=self.history_length)
        self.action_history = deque(maxlen=self.history_length)
        self.reward_history = deque(maxlen=self.history_length)
        
        # 交通信号灯信息
        self.program = None
        self.num_phases = 0
        self.phase_durations = {}
        self.current_phase = 0
        self.time_in_phase = 0
        
        # 车道和检测器映射
        self.lane_to_detector_map = {}
        self.incoming_lanes = {}
        self.lane_data = {}
        self.direction_data = {}
        
        # 性能统计
        self.total_vehicles_at_intersection = 0
        self.cumulative_waiting_time = 0
        self.cumulative_throughput = 0
        self.phase_switch_count = 0
        
        # 初始化
        self._initialize()
        
    def _initialize(self):
        """初始化控制器"""
        try:
            # 获取信号灯程序逻辑
            self.program = traci.trafficlight.getAllProgramLogics(self.tls_id)[0]
            self.num_phases = len(self.program.phases)
            self.phase_durations = {i: phase.duration for i, phase in enumerate(self.program.phases)}
            
            # 解析检测器映射
            self._parse_detector_mapping()
            
            # 设置进口车道
            self._setup_incoming_lanes()
            
            logger.info(f"升级版控制器初始化完成，路口: {self.tls_id}, 相位数: {self.num_phases}")
            
        except Exception as e:
            logger.error(f"控制器初始化失败: {e}")
            
    def _parse_detector_mapping(self):
        """解析车道到检测器的映射"""
        try:
            tree = ET.parse(self.additional_file_path)
            root = tree.getroot()
            
            for detector in root.findall('laneAreaDetector'):
                det_id = detector.get('id')
                lane_id = detector.get('lane')
                if det_id and lane_id:
                    self.lane_to_detector_map[lane_id] = det_id
                    
        except Exception as e:
            logger.error(f"解析检测器映射失败: {e}")
            
    def _setup_incoming_lanes(self):
        """设置进口车道分组"""
        try:
            all_controlled_links = traci.trafficlight.getControlledLinks(self.tls_id)
            
            ns_incoming_lanes = []
            ew_incoming_lanes = []
            
            # 基于相位状态分析车道方向
            if self.num_phases > 0:
                # 相位0（南北绿灯）
                if len(self.program.phases[0].state) == len(all_controlled_links):
                    phase_0_state = self.program.phases[0].state
                    for i, link in enumerate(all_controlled_links):
                        if phase_0_state[i] == 'G':
                            lane_id = link[0][0]
                            detector_id = self.lane_to_detector_map.get(lane_id)
                            if detector_id:
                                ns_incoming_lanes.append((lane_id, detector_id))
                
                # 相位2（东西绿灯）
                if self.num_phases > 2:
                    phase_2_state = self.program.phases[2].state
                    for i, link in enumerate(all_controlled_links):
                        if phase_2_state[i] == 'G':
                            lane_id = link[0][0]
                            detector_id = self.lane_to_detector_map.get(lane_id)
                            if detector_id:
                                ew_incoming_lanes.append((lane_id, detector_id))
            
            self.incoming_lanes["north_south_in"] = list(set(ns_incoming_lanes))
            self.incoming_lanes["east_west_in"] = list(set(ew_incoming_lanes))
            
        except Exception as e:
            logger.error(f"设置进口车道失败: {e}")
            
    def get_enhanced_observation(self) -> np.ndarray:
        """获取增强的观测数据"""
        try:
            # 基础交通数据
            self._collect_traffic_data()
            
            # 构建增强观测向量（20维）
            observation = np.array([
                # 基础信号灯状态 (4维)
                self.current_phase,
                self.time_in_phase,
                self.num_phases,
                self.phase_switch_count,
                
                # 南北方向交通状态 (5维)
                self.direction_data.get("north_south", {}).get("queue_length", 0),
                self.direction_data.get("north_south", {}).get("waiting_time", 0),
                self.direction_data.get("north_south", {}).get("avg_speed", 0),
                self.direction_data.get("north_south", {}).get("vehicle_count", 0),
                self.direction_data.get("north_south", {}).get("density", 0),
                
                # 东西方向交通状态 (5维)
                self.direction_data.get("east_west", {}).get("queue_length", 0),
                self.direction_data.get("east_west", {}).get("waiting_time", 0),
                self.direction_data.get("east_west", {}).get("avg_speed", 0),
                self.direction_data.get("east_west", {}).get("vehicle_count", 0),
                self.direction_data.get("east_west", {}).get("density", 0),
                
                # 全局统计 (3维)
                self.total_vehicles_at_intersection,
                self.cumulative_waiting_time / max(1, self.time_in_phase),  # 平均等待时间
                self.cumulative_throughput,
                
                # 时序特征 (3维)
                len(self.observation_history),
                np.mean([obs[0] for obs in self.observation_history]) if self.observation_history else 0,  # 历史相位均值
                np.std([obs[1] for obs in self.observation_history]) if len(self.observation_history) > 1 else 0,  # 历史时间方差
            ], dtype=np.float32)
            
            # 添加到历史记录
            self.observation_history.append(observation.copy())
            
            return observation
            
        except Exception as e:
            logger.error(f"获取观测失败: {e}")
            return np.zeros(20, dtype=np.float32)
            
    def _collect_traffic_data(self):
        """收集交通数据"""
        try:
            self.lane_data = {}
            
            # 收集各方向数据
            for direction, lanes in self.incoming_lanes.items():
                total_queue = 0
                total_waiting = 0
                total_speed = 0
                total_vehicles = 0
                total_density = 0
                lane_count = len(lanes)
                
                for lane_id, detector_id in lanes:
                    try:
                        # 基础数据
                        queue_length = traci.lane.getLastStepHaltingNumber(lane_id)
                        waiting_time = traci.lane.getWaitingTime(lane_id)
                        avg_speed = traci.lane.getLastStepMeanSpeed(lane_id)
                        vehicle_count = traci.lane.getLastStepVehicleNumber(lane_id)
                        
                        # 计算密度
                        lane_length = traci.lane.getLength(lane_id)
                        density = vehicle_count / max(1, lane_length / 100)  # 每100米的车辆数
                        
                        # 累加
                        total_queue += queue_length
                        total_waiting += waiting_time
                        total_speed += avg_speed
                        total_vehicles += vehicle_count
                        total_density += density
                        
                        # 存储车道级数据
                        self.lane_data[lane_id] = {
                            "queue_length": queue_length,
                            "waiting_time": waiting_time,
                            "avg_speed": avg_speed,
                            "vehicle_count": vehicle_count,
                            "density": density
                        }
                        
                    except Exception as e:
                        logger.warning(f"收集车道 {lane_id} 数据失败: {e}")
                
                # 计算方向平均值
                if lane_count > 0:
                    self.direction_data[direction.replace("_in", "")] = {
                        "queue_length": total_queue,
                        "waiting_time": total_waiting,
                        "avg_speed": total_speed / lane_count,
                        "vehicle_count": total_vehicles,
                        "density": total_density / lane_count
                    }
                    
            # 更新全局统计
            ns_data = self.direction_data.get("north_south", {})
            ew_data = self.direction_data.get("east_west", {})
            
            self.total_vehicles_at_intersection = (
                ns_data.get("vehicle_count", 0) + ew_data.get("vehicle_count", 0)
            )
            
            self.cumulative_waiting_time += (
                ns_data.get("waiting_time", 0) + ew_data.get("waiting_time", 0)
            )
            
        except Exception as e:
            logger.error(f"收集交通数据失败: {e}")
            
    def calculate_advanced_reward(self, observation: np.ndarray, action: int) -> float:
        """计算高级奖励函数"""
        try:
            # 提取观测特征
            current_phase = observation[0]
            time_in_phase = observation[1]
            
            ns_queue = observation[4]
            ns_waiting = observation[5]
            ns_speed = observation[6]
            ns_vehicles = observation[7]
            ns_density = observation[8]
            
            ew_queue = observation[9]
            ew_waiting = observation[10]
            ew_speed = observation[11]
            ew_vehicles = observation[12]
            ew_density = observation[13]
            
            total_vehicles = observation[14]
            avg_waiting = observation[15]
            throughput = observation[16]
            
            # 基础奖励组件
            reward_components = {}
            
            # 1. 等待时间惩罚（非线性）
            waiting_penalty = -(ns_waiting + ew_waiting) * 0.1
            if ns_waiting > 60 or ew_waiting > 60:  # 严重拥堵额外惩罚
                waiting_penalty *= 2
            reward_components['waiting'] = waiting_penalty
            
            # 2. 队列长度惩罚（考虑密度）
            queue_penalty = -(ns_queue + ew_queue) * 0.5
            density_penalty = -(ns_density + ew_density) * 0.3
            reward_components['queue'] = queue_penalty + density_penalty
            
            # 3. 速度奖励（鼓励流畅通行）
            speed_reward = (ns_speed + ew_speed) * 0.2
            reward_components['speed'] = speed_reward
            
            # 4. 通行效率奖励
            efficiency_reward = throughput * 1.0
            reward_components['efficiency'] = efficiency_reward
            
            # 5. 相位切换惩罚（智能化）
            switch_penalty = 0
            if action == 1:  # 切换动作
                # 基础切换惩罚
                switch_penalty = -10.0
                
                # 如果切换过于频繁，额外惩罚
                if time_in_phase < 10:
                    switch_penalty -= 20.0
                
                # 如果当前方向没有车辆但要切换，额外惩罚
                current_direction_vehicles = ns_vehicles if current_phase in [0, 1] else ew_vehicles
                if current_direction_vehicles == 0:
                    switch_penalty -= 15.0
                    
            reward_components['switch'] = switch_penalty
            
            # 6. 平衡性奖励（鼓励各方向均衡通行）
            vehicle_imbalance = abs(ns_vehicles - ew_vehicles)
            balance_penalty = -vehicle_imbalance * 0.1
            reward_components['balance'] = balance_penalty
            
            # 7. 历史趋势奖励
            trend_reward = 0
            if len(self.observation_history) >= 2:
                prev_total_vehicles = self.observation_history[-2][14]
                if total_vehicles < prev_total_vehicles:  # 车辆总数减少
                    trend_reward = 5.0
                elif total_vehicles > prev_total_vehicles:  # 车辆总数增加
                    trend_reward = -2.0
            reward_components['trend'] = trend_reward
            
            # 8. 紧急情况处理
            emergency_penalty = 0
            if ns_waiting > 120 or ew_waiting > 120:  # 极度拥堵
                emergency_penalty = -50.0
            reward_components['emergency'] = emergency_penalty
            
            # 计算总奖励
            total_reward = sum(reward_components.values())
            
            # 记录奖励历史
            self.reward_history.append(total_reward)
            
            # 调试信息（可选）
            if self.config.get('debug_reward', False):
                logger.debug(f"路口 {self.tls_id} 奖励组件: {reward_components}, 总奖励: {total_reward:.2f}")
            
            return total_reward
            
        except Exception as e:
            logger.error(f"计算奖励失败: {e}")
            return -10.0  # 默认惩罚
            
    def update(self, step: int, action: int = None) -> float:
        """更新控制器状态"""
        try:
            # 更新时间
            self.time_in_phase += 1
            
            # 获取观测
            observation = self.get_enhanced_observation()
            
            # 计算奖励
            reward = self.calculate_advanced_reward(observation, action)
            
            # 执行动作
            if action is not None:
                self._execute_action(action, step)
                
            # 记录动作历史
            if action is not None:
                self.action_history.append(action)
                
            return reward
            
        except Exception as e:
            logger.error(f"更新控制器失败: {e}")
            return -10.0
            
    def _execute_action(self, action: int, step: int):
        """执行动作"""
        try:
            min_phase_duration = 10  # 最小相位持续时间
            
            if action == 1 and self.time_in_phase >= min_phase_duration:
                # 切换到下一相位
                next_phase = (self.current_phase + 1) % self.num_phases
                traci.trafficlight.setPhase(self.tls_id, next_phase)
                
                self.current_phase = next_phase
                self.time_in_phase = 0
                self.phase_switch_count += 1
                
                logger.debug(f"步骤 {step}: 路口 {self.tls_id} 切换到相位 {next_phase}")
                
        except Exception as e:
            logger.error(f"执行动作失败: {e}")
            
    def reset(self) -> np.ndarray:
        """重置控制器"""
        try:
            self.current_phase = 0
            self.time_in_phase = 0
            self.phase_switch_count = 0
            self.cumulative_waiting_time = 0
            self.cumulative_throughput = 0
            
            # 清空历史记录
            self.observation_history.clear()
            self.action_history.clear()
            self.reward_history.clear()
            
            # 重置交通数据
            self.lane_data = {}
            self.direction_data = {}
            self.total_vehicles_at_intersection = 0
            
            # 设置初始相位
            traci.trafficlight.setPhase(self.tls_id, self.current_phase)
            
            return self.get_enhanced_observation()
            
        except Exception as e:
            logger.error(f"重置控制器失败: {e}")
            return np.zeros(20, dtype=np.float32)

    def get_performance_metrics(self) -> Dict[str, float]:
        """获取性能指标"""
        return {
            "id": self.tls_id,
            "current_phase": self.current_phase,
            "time_in_phase": self.time_in_phase,
            "phase_switches": self.phase_switch_count,
            "total_vehicles": self.total_vehicles_at_intersection,
            "avg_waiting_time": self.cumulative_waiting_time / max(1, self.time_in_phase),
            "throughput": self.cumulative_throughput,
            "avg_reward": np.mean(self.reward_history) if self.reward_history else 0.0
        }


class MultiAgentCoordinator:
    """多智能体协调器"""

    def __init__(self, controllers: List[AdvancedTLSController], config: Dict[str, Any] = None):
        self.controllers = controllers
        self.config = config or {}
        self.coordination_enabled = self.config.get('coordination_enabled', True)
        self.coordination_radius = self.config.get('coordination_radius', 500)  # 协调半径（米）

        # 路口间距离矩阵（简化版，实际应该从路网拓扑计算）
        self.distance_matrix = self._calculate_distance_matrix()

        # 协调历史
        self.coordination_history = deque(maxlen=100)

    def _calculate_distance_matrix(self) -> np.ndarray:
        """计算路口间距离矩阵（简化版）"""
        num_controllers = len(self.controllers)
        distances = np.zeros((num_controllers, num_controllers))

        # 简化假设：2x2网格，路口间距离为300米
        for i in range(num_controllers):
            for j in range(num_controllers):
                if i != j:
                    distances[i][j] = 300.0  # 简化距离

        return distances

    def get_coordination_signal(self, controller_idx: int, observation: np.ndarray) -> np.ndarray:
        """获取协调信号"""
        if not self.coordination_enabled:
            return np.zeros(5)  # 返回空协调信号

        try:
            coordination_signal = np.zeros(5)

            # 获取邻近路口信息
            nearby_controllers = self._get_nearby_controllers(controller_idx)

            if nearby_controllers:
                # 计算邻近路口的平均状态
                nearby_phases = []
                nearby_vehicles = []
                nearby_waiting = []

                for nearby_idx in nearby_controllers:
                    nearby_obs = self.controllers[nearby_idx].get_enhanced_observation()
                    nearby_phases.append(nearby_obs[0])
                    nearby_vehicles.append(nearby_obs[14])
                    nearby_waiting.append(nearby_obs[15])

                # 构建协调信号
                coordination_signal[0] = np.mean(nearby_phases)  # 邻近路口平均相位
                coordination_signal[1] = np.mean(nearby_vehicles)  # 邻近路口平均车辆数
                coordination_signal[2] = np.mean(nearby_waiting)  # 邻近路口平均等待时间
                coordination_signal[3] = len(nearby_controllers)  # 邻近路口数量
                coordination_signal[4] = self._calculate_coordination_urgency(controller_idx, nearby_controllers)

            return coordination_signal

        except Exception as e:
            logger.error(f"获取协调信号失败: {e}")
            return np.zeros(5)

    def _get_nearby_controllers(self, controller_idx: int) -> List[int]:
        """获取邻近控制器索引"""
        nearby = []
        for i, distance in enumerate(self.distance_matrix[controller_idx]):
            if 0 < distance <= self.coordination_radius:
                nearby.append(i)
        return nearby

    def _calculate_coordination_urgency(self, controller_idx: int, nearby_controllers: List[int]) -> float:
        """计算协调紧急度"""
        try:
            current_obs = self.controllers[controller_idx].get_enhanced_observation()
            current_waiting = current_obs[15]

            # 如果当前路口等待时间很长，紧急度高
            urgency = min(current_waiting / 60.0, 1.0)  # 标准化到[0,1]

            # 考虑邻近路口的影响
            for nearby_idx in nearby_controllers:
                nearby_obs = self.controllers[nearby_idx].get_enhanced_observation()
                nearby_waiting = nearby_obs[15]

                # 如果邻近路口也很拥堵，增加紧急度
                if nearby_waiting > 30:
                    urgency += 0.1

            return min(urgency, 1.0)

        except Exception as e:
            logger.error(f"计算协调紧急度失败: {e}")
            return 0.0

    def suggest_coordinated_actions(self, observations: List[np.ndarray]) -> List[int]:
        """建议协调动作"""
        if not self.coordination_enabled:
            return [0] * len(self.controllers)  # 返回默认动作

        try:
            suggested_actions = []

            for i, obs in enumerate(observations):
                # 获取协调信号
                coord_signal = self.get_coordination_signal(i, obs)

                # 基于协调信号调整动作建议
                action_suggestion = self._calculate_coordinated_action(i, obs, coord_signal)
                suggested_actions.append(action_suggestion)

            # 记录协调历史
            self.coordination_history.append({
                'step': len(self.coordination_history),
                'actions': suggested_actions.copy(),
                'observations': [obs.copy() for obs in observations]
            })

            return suggested_actions

        except Exception as e:
            logger.error(f"建议协调动作失败: {e}")
            return [0] * len(self.controllers)

    def _calculate_coordinated_action(self, controller_idx: int, observation: np.ndarray, coord_signal: np.ndarray) -> int:
        """计算协调动作"""
        try:
            current_phase = observation[0]
            time_in_phase = observation[1]
            current_waiting = observation[15]

            nearby_avg_phase = coord_signal[0]
            nearby_avg_waiting = coord_signal[2]
            coordination_urgency = coord_signal[4]

            # 基础决策：如果等待时间过长且相位时间足够，建议切换
            if current_waiting > 30 and time_in_phase >= 10:
                base_action = 1  # 切换
            else:
                base_action = 0  # 保持

            # 协调调整
            if coordination_urgency > 0.7:
                # 高紧急度：优先考虑本路口
                return base_action
            elif coordination_urgency < 0.3 and nearby_avg_waiting > current_waiting:
                # 低紧急度且邻近路口更拥堵：保持当前相位
                return 0
            else:
                return base_action

        except Exception as e:
            logger.error(f"计算协调动作失败: {e}")
            return 0
