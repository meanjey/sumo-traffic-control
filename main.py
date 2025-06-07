#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import numpy as np
import traci
import sumolib
from sumolib import checkBinary
import time
import xml.etree.ElementTree as ET # Import for XML parsing
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO # 导入PPO
# import argparse # 不再需要命令行参数

# 确保SUMO_HOME环境变量已设置
if 'SUMO_HOME' not in os.environ:
    sys.path.append(os.path.join('c:', os.sep, 'Program Files (x86)', 'Eclipse', 'Sumo', 'tools'))
    os.environ['SUMO_HOME'] = os.path.join('c:', os.sep, 'Program Files (x86)', 'Eclipse', 'Sumo')

class AdaptiveTLSController:
    """自适应交通信号灯控制器"""
    
    def __init__(self, tls_id):
        """初始化控制器
        
        参数:
            tls_id (str): 交通信号灯ID
        """
        self.tls_id = tls_id
        
        # 获取信号灯的程序逻辑
        self.program = traci.trafficlight.getAllProgramLogics(tls_id)[0]
        self.num_phases = len(self.program.phases)
        
        # 从程序逻辑中读取真实的相位持续时间
        self.phase_durations = {i: phase.duration for i, phase in enumerate(self.program.phases)}

        self.min_phase_duration = min(self.phase_durations.values()) if self.phase_durations else 5
        self.max_phase_duration = max(self.phase_durations.values()) if self.phase_durations else 60
        
        print(f"路口 {tls_id} 的相位数量: {self.num_phases}")
        for i, phase in enumerate(self.program.phases):
            print(f"相位 {i}: {phase.state}, 从SUMO读取的持续时间: {self.phase_durations.get(i)}")
        
        # 当前相位和计时
        self.current_phase = 0
        self.time_in_phase = 0
        
        # 交通数据存储
        self.lane_data = {} # Stores queue_length, waiting_time, mean_speed, vehicle_count for each incoming lane in current step
        self.direction_data = {} # Aggregated data per direction (north_south, east_west)
        self.incoming_lanes = {} # Stores lane IDs grouped by direction
        self.lane_to_detector_map = {} # Map lane_id to detector_id
        self._parse_detector_mapping() # Load detector mappings
        self._setup_incoming_lanes() # Setup the lane groups
        
        # 性能指标 (for current step, for conceptual reward calculation)
        self.total_vehicles_at_intersection = 0 # Total vehicles currently at this intersection's incoming lanes

    def _parse_detector_mapping(self):
        """从 additional.xml 文件解析车道ID到检测器ID的映射"""
        additional_file = "new_additional.xml"
        tree = ET.parse(additional_file)
        root = tree.getroot()

        for detector in root.findall('laneAreaDetector'):
            det_id = detector.get('id')
            lane_id = detector.get('lane')
            if det_id and lane_id:
                self.lane_to_detector_map[lane_id] = det_id
        # print(f"DEBUG: 车道到检测器的映射: {self.lane_to_detector_map}") # Removed debug print

    def _setup_incoming_lanes(self):
        """根据交通灯ID设置其对应的进口车道，并按方向分组"""
        all_controlled_links = traci.trafficlight.getControlledLinks(self.tls_id)
        
        ns_incoming_lanes = []
        ew_incoming_lanes = []

        # Assuming Phase 0 is NS green and Phase 2 is EW green based on typical SUMO grid defaults
        # And that the links for a 'G' in phase 0 are NS incoming, and for phase 2 are EW incoming.
        if self.num_phases > 0:
            # Phase 0 (NS green)
            if len(self.program.phases[0].state) == len(all_controlled_links):
                phase_0_state = self.program.phases[0].state
                for i, link in enumerate(all_controlled_links):
                    if phase_0_state[i] == 'G':
                        lane_id = link[0][0]
                        detector_id = self.lane_to_detector_map.get(lane_id) # Get detector_id from map
                        if detector_id:
                            ns_incoming_lanes.append((lane_id, detector_id)) # Store as (lane_id, detector_id)
                        else:
                            print(f"警告: 未找到车道 {lane_id} 的检测器 ID。")

            # Phase 2 (EW green)
            if self.num_phases > 2 and len(self.program.phases[2].state) == len(all_controlled_links):
                phase_2_state = self.program.phases[2].state
                for i, link in enumerate(all_controlled_links):
                    if phase_2_state[i] == 'G':
                        lane_id = link[0][0]
                        detector_id = self.lane_to_detector_map.get(lane_id) # Get detector_id from map
                        if detector_id:
                            ew_incoming_lanes.append((lane_id, detector_id)) # Store as (lane_id, detector_id)
                        else:
                            print(f"警告: 未找到车道 {lane_id} 的检测器 ID。")
        
        self.incoming_lanes["north_south_in"] = list(set(ns_incoming_lanes)) # Use set to avoid duplicates
        self.incoming_lanes["east_west_in"] = list(set(ew_incoming_lanes))
    
    def update(self, step, action=None):
        """更新控制器状态
        
        参数:
            step (int): 当前仿真步
            action (int, optional): 强化学习智能体给出的动作。0表示保持当前相位，1表示切换到下一相位。
        """
        # 更新相位计时
        self.time_in_phase += 1
        
        # 收集交通数据 (这将作为RL的观测)
        observation = self.get_observation()
        
        # 计算概念奖励 (供调试和初步评估使用)
        reward = self._calculate_reward(observation)
        
        # 如果提供了动作，则根据动作决定是否切换相位
        if action is not None:
            # 动作 0: 保持当前相位
            # 动作 1: 切换到下一相位
            if action == 1:
                # 只有在当前相位持续时间达到最小阈值时才允许切换
                if self.time_in_phase >= self.min_phase_duration:
                    self._switch_to_next_phase(step)
            # 如果动作是0，或者动作是1但未达到最小持续时间，则保持当前相位
        else:
            # 如果没有提供RL动作，则继续使用启发式（仅用于兼容性，最终会被RL取代）
            # 检查是否需要切换相位
            current_duration = self.phase_durations.get(self.current_phase, self.min_phase_duration) # Use default for yellow
            if self.time_in_phase >= current_duration:
                self._switch_to_next_phase(step)

        return reward # 返回奖励给RL环境

    def get_observation(self):
        """收集交通数据，包括排队长度、等待时间、平均速度和车辆数量，并返回作为强化学习的观测。"""
        self.lane_data = {}  # Reset for current step

        ns_queue_length = 0
        ns_waiting_time = 0
        ns_vehicle_count = 0
        ns_speed_sum = 0

        ew_queue_length = 0
        ew_waiting_time = 0
        ew_vehicle_count = 0
        ew_speed_sum = 0

        # Collect data for north-south incoming lanes
        for lane_id, detector_id in self.incoming_lanes.get("north_south_in", []):
            queue_length = traci.lanearea.getJamLengthMeters(detector_id)  # 使用检测器获取排队长度
            waiting_time = traci.lane.getWaitingTime(lane_id)  # total waiting time on this lane in the last step
            mean_speed = traci.lane.getLastStepMeanSpeed(lane_id)
            vehicle_count = traci.lane.getLastStepVehicleNumber(lane_id)

            self.lane_data[lane_id] = {
                "queue_length": queue_length,
                "waiting_time": waiting_time,
                "mean_speed": mean_speed,
                "vehicle_count": vehicle_count
            }

            ns_queue_length += queue_length
            ns_waiting_time += waiting_time
            ns_vehicle_count += vehicle_count
            ns_speed_sum += (mean_speed * vehicle_count)

        # Collect data for east-west incoming lanes
        for lane_id, detector_id in self.incoming_lanes.get("east_west_in", []):
            queue_length = traci.lanearea.getJamLengthMeters(detector_id)  # 使用检测器获取排队长度
            waiting_time = traci.lane.getWaitingTime(lane_id)
            mean_speed = traci.lane.getLastStepMeanSpeed(lane_id)
            vehicle_count = traci.lane.getLastStepVehicleNumber(lane_id)

            self.lane_data[lane_id] = {
                "queue_length": queue_length,
                "waiting_time": waiting_time,
                "mean_speed": mean_speed,
                "vehicle_count": vehicle_count
            }

            ew_queue_length += queue_length
            ew_waiting_time += waiting_time
            ew_vehicle_count += vehicle_count
            ew_speed_sum += (mean_speed * vehicle_count)

        # Aggregate data per direction
        self.direction_data = {
            "north_south": {
                "queue_length": ns_queue_length,
                "waiting_time": ns_waiting_time,
                "vehicle_count": ns_vehicle_count,
                "avg_speed": ns_speed_sum / max(1, ns_vehicle_count)  # Avoid division by zero
            },
            "east_west": {
                "queue_length": ew_queue_length,
                "waiting_time": ew_waiting_time,
                "vehicle_count": ew_vehicle_count,
                "avg_speed": ew_speed_sum / max(1, ew_vehicle_count)  # Avoid division by zero
            }
        }
        # Total vehicles currently at this intersection's incoming lanes
        self.total_vehicles_at_intersection = ns_vehicle_count + ew_vehicle_count

        # 返回观测，包括当前相位、相位计时以及交通数据 (扁平化为NumPy数组)
        # 确保顺序一致，以保证观测空间的稳定性
        observation = np.array([
            self.current_phase,
            self.time_in_phase,
            self.direction_data["north_south"]["queue_length"],
            self.direction_data["north_south"]["waiting_time"],
            self.direction_data["north_south"]["avg_speed"],
            self.direction_data["north_south"]["vehicle_count"],
            self.direction_data["east_west"]["queue_length"],
            self.direction_data["east_west"]["waiting_time"],
            self.direction_data["east_west"]["avg_speed"],
            self.direction_data["east_west"]["vehicle_count"],
        ], dtype=np.float32)
        return observation

    def _switch_to_next_phase(self, step):
        """切换到下一个相位"""
        # 更新相位
        next_phase = (self.current_phase + 1) % self.num_phases
        
        # 如果下一个相位是绿灯，设置新的持续时间
        if next_phase == 0 or next_phase == 2:
            # 这里可以根据RL智能体的决策来设置持续时间，目前仍使用默认值
            traci.trafficlight.setPhase(self.tls_id, next_phase)
            traci.trafficlight.setPhaseDuration(self.tls_id, self.phase_durations.get(next_phase, self.min_phase_duration))
            print(f"步骤 {step}: 路口 {self.tls_id} 切换到相位 {next_phase}")
            # print(f"相位 {next_phase} ({'南北方向' if next_phase == 0 else '东西方向'}) 的调整后持续时间: {self.phase_durations.get(next_phase, self.min_phase_duration)}秒")
        else:
            # 黄灯相位，使用默认持续时间
            traci.trafficlight.setPhase(self.tls_id, next_phase)
            traci.trafficlight.setPhaseDuration(self.tls_id, self.program.phases[next_phase].duration)
            print(f"步骤 {step}: 路口 {self.tls_id} 切换到相位 {next_phase}")
        
        self.current_phase = next_phase
        self.time_in_phase = 0

    def _calculate_reward(self, observation):
        """
        计算当前步骤的奖励。
        这是一个初步的奖励函数，将根据比赛目标进一步优化。
        
        奖励目标: 最小化总等待时间，最小化排队长度，最大化平均速度。
        """
        # 获取当前观测中的数据
        # 注意: 观测空间索引与get_observation中的顺序一致
        current_phase = observation[0]
        time_in_phase = observation[1]
        ns_queue = observation[2]
        ns_waiting = observation[3]
        ns_speed = observation[4]
        ns_vehicles = observation[5]

        ew_queue = observation[6]
        ew_waiting = observation[7]
        ew_speed = observation[8]
        ew_vehicles = observation[9]

        # 负奖励：排队长度和等待时间越大，奖励越低
        # 正奖励：平均速度越高，奖励越高 (这里需要注意，如果车道空了，速度会是0，需要适当处理)
        
        # 权重可以根据比赛目标进行调整
        queue_penalty_weight = 0.3 # Adjusted
        waiting_penalty_weight = 0.05 # Adjusted
        speed_reward_weight = 0.1 # Adjusted

        # Penalty for vehicles currently at the intersection (encourage clearing)
        total_vehicles_penalty_weight = 0.02 # New penalty

        # 计算南北方向奖励
        ns_reward = -(ns_queue * queue_penalty_weight) - (ns_waiting * waiting_penalty_weight)
        if ns_vehicles > 0:
            ns_reward += (ns_speed * speed_reward_weight)
        
        # 计算东西方向奖励
        ew_reward = -(ew_queue * queue_penalty_weight) - (ew_waiting * waiting_penalty_weight)
        if ew_vehicles > 0:
            ew_reward += (ew_speed * speed_reward_weight)
        
        # 总奖励是两个方向奖励的和
        total_reward = ns_reward + ew_reward

        # Add penalty for total vehicles at intersection
        total_reward -= (ns_vehicles + ew_vehicles) * total_vehicles_penalty_weight

        # 如果路口没有车辆，不再给予少量奖励，避免干扰RL学习
        # if ns_vehicles == 0 and ew_vehicles == 0:
        #     total_reward = 0.1 # 小的探索奖励

        return total_reward

    def get_performance_metrics(self):
        """获取当前交通信号灯的性能指标"""
        return {
            "id": self.tls_id,
            "current_phase": self.current_phase,
            "time_in_phase": self.time_in_phase,
            "direction_data": self.direction_data,
            "total_vehicles_at_intersection": self.total_vehicles_at_intersection
        }

    def reset(self):
        """重置交通信号灯控制器状态，用于RL回合开始"""
        self.current_phase = 0
        self.time_in_phase = 0
        traci.trafficlight.setPhase(self.tls_id, self.current_phase)
        traci.trafficlight.setPhaseDuration(self.tls_id, self.program.phases[self.current_phase].duration)
        self.lane_data = {}
        self.direction_data = {}
        self.total_vehicles_at_intersection = 0
        return self.get_observation() # 返回初始观测

class SumoTrafficEnv(gym.Env):
    """
    自定义的SUMO交通信号灯控制环境，遵循Gymnasium接口。
    """
    metadata = {'render_modes': ['human'], 'render_fps': 10}

    def __init__(self, sumocfg_file, tls_ids, render_mode=None):
        super().__init__()
        self.sumocfg_file = sumocfg_file
        self.tls_ids = tls_ids
        self.render_mode = render_mode
        self.label = "TrafficLightControlEnv"
        
        # 初始化控制器 (目前我们只处理一个交通灯)
        # 未来可以扩展为多智能体环境
        if not self.tls_ids:
            raise ValueError("没有提供交通信号灯ID。")
        self.controller = None # 在reset中初始化控制器
        self._sumo_started = False # 跟踪SUMO是否已启动
        self.max_steps = 3600 # 在这里定义最大步数
        self.step_count = 0 # 在这里初始化step_count

        # 定义观测空间 (与AdaptiveTLSController.get_observation()的输出一致)
        # 例子: current_phase, time_in_phase, ns_queue_length, ns_waiting_time, ns_avg_speed, ns_vehicle_count,
        # ew_queue_length, ew_waiting_time, ew_avg_speed, ew_vehicle_count
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(10,), dtype=np.float32)
        
        # 定义动作空间: 0 (保持当前相位), 1 (切换到下一相位)
        self.action_space = spaces.Discrete(2) # 2个离散动作
        
        self.current_step = 0

    def _start_sumo(self):
        """启动SUMO仿真"""
        if self.render_mode == 'human':
            sumoBinary = sumolib.checkBinary('sumo-gui')
            sumo_cmd = [sumoBinary, "-c", self.sumocfg_file, "--step-length", "1", "--quit-on-end", "--no-warnings", "--start", "--verbose"]
        else:
            sumoBinary = sumolib.checkBinary('sumo')
            sumo_cmd = [sumoBinary, "-c", self.sumocfg_file, "--step-length", "1", "--quit-on-end", "--no-warnings", "--no-step-log", "--verbose"]
        
        print(f"启动SUMO仿真，使用二进制文件: {sumoBinary}")
        traci.start(sumo_cmd)
        self._sumo_started = True # 设置标志，表示SUMO已启动
        print("SUMO仿真已成功启动。")

    def _get_info(self):
        """获取环境信息 (可选，用于调试或额外指标)"""
        metrics = self.controller.get_performance_metrics()
        return {
            "tls_id": metrics["id"],
            "current_phase": metrics["current_phase"],
            "time_in_phase": metrics["time_in_phase"],
            "ns_queue_length": metrics["direction_data"]["north_south"]["queue_length"],
            "ew_queue_length": metrics["direction_data"]["east_west"]["queue_length"],
            "total_vehicles": metrics["total_vehicles_at_intersection"]
        }

    def step(self, action):
        """执行一个动作，推进仿真一步"""
        if not self._sumo_started:
            raise RuntimeError("仿真未启动，请先调用reset()")

        # 执行仿真步骤
        traci.simulationStep()
        self.step_count += 1

        # 更新控制器状态
        reward = self.controller.update(self.step_count, action)
        observation = self.controller.get_observation()
        info = self._get_info()

        # 检查是否应该结束
        terminated = False  # 暂时禁用错误的终止条件
        truncated = self.step_count >= self.max_steps

        # 打印调试信息
        # print(f"步骤 {self.step_count}: terminated={terminated}, truncated={truncated}")

        return observation, reward, terminated, truncated, info

    def reset(self, seed=None, options=None):
        """重置环境，开始新的回合"""
        super().reset(seed=seed) # 调用父类的reset

        if self._sumo_started: # 检查标志，避免重复关闭或关闭未启动的仿真
            print("关闭现有SUMO仿真...")
            traci.close()
            self._sumo_started = False
            time.sleep(1) # 给SUMO时间来完全关闭
            print("SUMO仿真已关闭。")
            
        self._start_sumo()
        self.step_count = 0 # 重置步数计数器

        # 重新获取交通信号灯ID，以防load重置
        # 由于main函数不再提前获取ID，我们在这里获取并确保控制器能够找到它们
        tls_ids_in_sim = traci.trafficlight.getIDList()
        if not tls_ids_in_sim:
            raise ValueError("重置后未找到交通信号灯ID。")
        
        # 确保我们传入环境的tls_ids列表与仿真中找到的第一个ID匹配
        if self.tls_ids[0] != tls_ids_in_sim[0]:
            print(f"警告: 环境初始化时提供的TLS ID ({self.tls_ids[0]}) 与仿真中首次找到的TLS ID ({tls_ids_in_sim[0]}) 不匹配。将使用仿真中的ID。")
            self.tls_ids = tls_ids_in_sim # 更新tls_ids以匹配仿真

        self.controller = AdaptiveTLSController(self.tls_ids[0]) # 重新初始化控制器

        observation = self.controller.reset() # 重置控制器并获取初始观测
        info = self._get_info()
        
        return observation, info

    def render(self):
        """渲染仿真 (如果render_mode是'human')"""
        pass # SUMO GUI会自己渲染，这里不需要额外操作

    def close(self):
        """关闭仿真环境"""
        if self._sumo_started:
            traci.close()
            self._sumo_started = False

def force_close_sumo():
    """强制关闭所有SUMO进程"""
    print("正在强制关闭所有SUMO进程...")
    try:
        # 在Windows上使用taskkill
        os.system("taskkill /F /IM sumo.exe")
        os.system("taskkill /F /IM sumo-gui.exe")
        print("SUMO进程已强制关闭。")
    except Exception as e:
        print(f"强制关闭SUMO时出错: {e}")

def main():
    # # 不再使用命令行参数
    # parser = argparse.ArgumentParser(description="使用PPO训练自适应交通灯")
    # parser.add_argument("--gui", action="store_true", help="使用SUMO-GUI进行可视化")
    # parser.add_argument("--test", action="store_true", help="加载并测试一个已训练的模型")
    # args = parser.parse_args()

    # 添加交互式菜单
    print("==========================================")
    print("     欢迎使用智能交通灯控制器")
    print("==========================================")
    print("请选择运行模式:")
    print("  1: 后台训练 (无GUI，速度最快)")
    print("  2: 可视化训练 (有GUI，方便观察)")
    print("  3: 测试已保存的模型 (有GUI)")
    
    choice = ""
    while choice not in ["1", "2", "3"]:
        choice = input("请输入选项 (1, 2, 或 3) 并按回车: ")

    test_mode = False
    render_mode = None

    if choice == '1':
        print("您选择了：[后台训练]")
        # render_mode 默认为 None
    elif choice == '2':
        print("您选择了：[可视化训练]")
        render_mode = 'human'
    elif choice == '3':
        print("您选择了：[测试已保存的模型]")
        test_mode = True
        render_mode = 'human'

    # 在开始前强制关闭任何残留的SUMO进程
    force_close_sumo()
    time.sleep(2)  # 等待进程完全关闭

    # SUMO配置文件路径
    sumocfg_file = "new_simulation.sumocfg"
    
    # 定义TensorBoard日志目录
    tensorboard_log_dir = "./ppo_tensorboard_logs/"
    os.makedirs(tensorboard_log_dir, exist_ok=True)

    # 不再在main中提前启动和关闭traci，由环境的reset方法处理
    # tls_ids 在这里可以是一个预定义的列表，或者在环境内部动态获取
    # 对于2x2网格，我们知道交通信号灯ID是A0, A1, B0, B1
    tls_ids = ["A0"] # 根据当前路网，只有一个信号灯

    if not tls_ids:
        print("错误: 未找到交通信号灯ID，无法初始化环境。")
        return

    # 初始化环境
    # 根据用户的选择直接设置render_mode
    env = SumoTrafficEnv(sumocfg_file, tls_ids, render_mode=render_mode) 

    # 如果是测试模式，加载模型并运行
    if test_mode:
        model_path = "ppo_traffic_light_controller.zip"
        if not os.path.exists(model_path):
            print(f"错误: 找不到模型文件 {model_path}。请先进行训练。")
            return
            
        print(f"--- 加载模型 {model_path} 并进入测试模式 ---")
        model = PPO.load(model_path, env=env)
        
        observation, info = env.reset()
        for _ in range(3600): # 运行一个完整的episode
            action, _states = model.predict(observation, deterministic=True) # 使用确定性动作
            observation, reward, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                break
        env.close()
        print("--- 测试结束 ---")
        return

    # 否则，进入训练模式
    # 初始化PPO智能体
    model = PPO("MlpPolicy", env, verbose=1, learning_rate=0.0003, n_steps=2048, batch_size=64, tensorboard_log=tensorboard_log_dir) # 添加tensorboard_log

    num_episodes = 50 # 增加回合数以便训练
    max_steps_per_episode = 3600 # 每个回合的最大仿真步数

    for episode in range(num_episodes):
        print(f"\n--- 回合 {episode + 1}/{num_episodes} 开始 ---")
        observation, info = env.reset()
        total_reward = 0
        
        for step in range(max_steps_per_episode):
            action, _states = model.predict(observation, deterministic=False) # 使用模型预测动作
            observation, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            
            # 在这里进行模型的学习 (例如，每N步或每个回合结束)
            # 对于PPO，通常在每个回合结束或达到一定步数后进行学习
            # 这里简化为在每个step之后learn，实际RL训练需要更复杂的缓冲区和学习策略
            # model.learn(total_timesteps=1) # 每次一步进行学习 (效率不高，仅作演示)
            
            if step % 100 == 0:
                print(f"步骤 {step}: 总奖励 {total_reward:.2f}")
                # print(f"路口 {info['tls_id']}: 相位 {info['current_phase']}, 车辆数 {info['total_vehicles']}, 排队长度 (NS/EW) {info['ns_queue_length']:.2f}/{info['ew_queue_length']:.2f}")
            
            if terminated or truncated:
                print(f"回合 {episode + 1} 在 {step + 1} 步结束。总奖励: {total_reward:.2f}")
                break
        
        # 每个回合结束后进行模型学习
        model.learn(total_timesteps=step + 1) # 让模型学习整个回合的数据

        print(f"--- 回合 {episode + 1}/{num_episodes} 结束 ---")
    
    # 保存训练好的模型
    model.save("ppo_traffic_light_controller")
    print("模型已保存为 ppo_traffic_light_controller.zip")

    env.close()
    print("已关闭SUMO仿真环境")

if __name__ == "__main__":
    main() 