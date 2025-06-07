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
from datetime import datetime # 导入datetime
from tensorboard.backend.event_processing import event_accumulator # 导入读取日志的工具
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
        
        # 在初始化时就启动SUMO，确保只启动一次
        self._start_sumo()

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
        super().reset(seed=seed)

        # 不再关闭和重启SUMO，而是使用traci.load来重置仿真状态
        sumo_cmd = ["-c", self.sumocfg_file, "--step-length", "1", "--quit-on-end", "--no-warnings"]
        if self.render_mode == 'human':
            sumo_cmd.extend(["--start", "--verbose"])
        else:
            sumo_cmd.append("--no-step-log")
        traci.load(sumo_cmd)
        
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

def get_final_reward_from_logs(log_path):
    """从TensorBoard日志文件中读取最终的平均奖励"""
    try:
        # 加载日志文件
        acc = event_accumulator.EventAccumulator(log_path, size_guidance={'scalars': 0})
        acc.Reload()
        
        # 检查'rollout/ep_rew_mean'标签是否存在
        if 'rollout/ep_rew_mean' in acc.Tags()['scalars']:
            events = acc.Scalars('rollout/ep_rew_mean')
            if events:
                # 返回最后一个事件的值
                return events[-1].value
    except Exception as e:
        # print(f"读取日志 {log_path} 时出错: {e}") # 可选的调试信息
        pass
    return None # 如果找不到或出错，返回None

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
    
    # --- 日志和模型保存路径设置 ---
    # 定义TensorBoard的根日志目录
    tensorboard_log_dir = "./ppo_tensorboard_logs/"
    os.makedirs(tensorboard_log_dir, exist_ok=True)
    
    # 创建保存模型的目录
    model_save_dir = "./saved_models/"
    os.makedirs(model_save_dir, exist_ok=True)
    
    # 为本次运行创建一个带时间戳的唯一名称
    run_name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_dir_for_this_run = os.path.join(tensorboard_log_dir, run_name)
    model_path_for_this_run = os.path.join(model_save_dir, f"{run_name}.zip")

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
        # 列出所有已保存的模型并获取它们的最终奖励
        model_infos = []
        saved_model_files = [f for f in os.listdir(model_save_dir) if f.endswith('.zip')]

        for model_file in saved_model_files:
            run_name_from_file = model_file.replace('.zip', '')
            log_path_for_model = os.path.join(tensorboard_log_dir, run_name_from_file)
            final_reward = get_final_reward_from_logs(log_path_for_model)
            model_infos.append({
                "file": model_file,
                "reward": final_reward if final_reward is not None else -float('inf') # 如果没有奖励，则排在最后
            })

        # 按奖励从高到低排序
        model_infos.sort(key=lambda x: x['reward'], reverse=True)

        if not model_infos:
            print(f"错误: 在 '{model_save_dir}' 目录下找不到任何已保存的模型 (.zip文件)。")
            return

        print("\n--- 请选择要加载的模型 (已按性能排序) ---")
        for i, info in enumerate(model_infos):
            reward_str = f"{info['reward']:.2f}" if info['reward'] > -float('inf') else "N/A"
            print(f"  {i + 1}: {info['file']} (最终奖励: {reward_str})")

        model_choice = -1
        while model_choice < 1 or model_choice > len(model_infos):
            try:
                raw_choice = input(f"请输入选项 (1-{len(model_infos)}): ")
                model_choice = int(raw_choice)
            except ValueError:
                pass
        
        chosen_model_path = os.path.join(model_save_dir, model_infos[model_choice - 1]['file'])
        
        print(f"--- 加载模型 {chosen_model_path} 并进入测试模式 ---")
        model = PPO.load(chosen_model_path, env=env)
        
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
    model = PPO("MlpPolicy", env, verbose=1, learning_rate=0.0003, n_steps=2048, batch_size=64, tensorboard_log=log_dir_for_this_run) # 使用带时间戳的日志路径

    # 定义总训练步数
    total_training_steps = 180000  # 大约等于 50个回合 (50 * 3600)
    
    try:
        # 移除手动的回合循环，直接调用learn()方法进行训练
        # stable-baselines3的learn()方法会内部处理环境的重置
        model.learn(total_timesteps=total_training_steps, progress_bar=True)
    
    finally:
        # 无论训练是正常结束还是被中断，都使用带时间戳的路径保存模型
        model.save(model_path_for_this_run)
        print(f"\n模型已保存到: {model_path_for_this_run}")

        env.close()
        print("已关闭SUMO仿真环境")

if __name__ == "__main__":
    main() 