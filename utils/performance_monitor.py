#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
性能监控和分析系统
"""

import time
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional
from collections import deque, defaultdict
from pathlib import Path
import json
import logging
from dataclasses import dataclass, asdict
from datetime import datetime

logger = logging.getLogger(__name__)


@dataclass
class PerformanceMetrics:
    """性能指标"""
    timestamp: float
    episode: int
    step: int
    reward: float
    waiting_time: float
    queue_length: float
    throughput: float
    speed: float
    phase_switches: int
    memory_mb: float
    cpu_percent: float


class MetricsCollector:
    """指标收集器"""
    
    def __init__(self, max_history: int = 10000):
        self.max_history = max_history
        self.metrics_history = deque(maxlen=max_history)
        self.episode_metrics = defaultdict(list)
        self.step_times = deque(maxlen=1000)
        
    def record_step_metrics(self, episode: int, step: int, metrics: Dict[str, Any]):
        """记录步骤指标"""
        try:
            from utils.error_handler import resource_monitor
            system_info = resource_monitor.get_system_info()
            
            perf_metrics = PerformanceMetrics(
                timestamp=time.time(),
                episode=episode,
                step=step,
                reward=metrics.get('reward', 0.0),
                waiting_time=metrics.get('waiting_time', 0.0),
                queue_length=metrics.get('queue_length', 0.0),
                throughput=metrics.get('throughput', 0.0),
                speed=metrics.get('speed', 0.0),
                phase_switches=metrics.get('phase_switches', 0),
                memory_mb=system_info.get('memory_mb', 0.0),
                cpu_percent=system_info.get('cpu_percent', 0.0)
            )
            
            self.metrics_history.append(perf_metrics)
            self.episode_metrics[episode].append(perf_metrics)
            
        except Exception as e:
            logger.error(f"记录指标失败: {e}")
    
    def record_step_time(self, duration: float):
        """记录步骤执行时间"""
        self.step_times.append(duration)
    
    def get_recent_performance(self, window: int = 100) -> Dict[str, float]:
        """获取最近的性能统计"""
        if len(self.metrics_history) < window:
            return {}
        
        recent_metrics = list(self.metrics_history)[-window:]
        
        return {
            'avg_reward': np.mean([m.reward for m in recent_metrics]),
            'avg_waiting_time': np.mean([m.waiting_time for m in recent_metrics]),
            'avg_queue_length': np.mean([m.queue_length for m in recent_metrics]),
            'avg_throughput': np.mean([m.throughput for m in recent_metrics]),
            'avg_speed': np.mean([m.speed for m in recent_metrics]),
            'avg_memory_mb': np.mean([m.memory_mb for m in recent_metrics]),
            'avg_cpu_percent': np.mean([m.cpu_percent for m in recent_metrics]),
            'avg_step_time': np.mean(list(self.step_times)[-window:]) if self.step_times else 0.0
        }
    
    def get_episode_summary(self, episode: int) -> Dict[str, Any]:
        """获取回合总结"""
        if episode not in self.episode_metrics:
            return {}
        
        episode_data = self.episode_metrics[episode]
        
        return {
            'episode': episode,
            'total_steps': len(episode_data),
            'total_reward': sum(m.reward for m in episode_data),
            'avg_reward': np.mean([m.reward for m in episode_data]),
            'max_waiting_time': max(m.waiting_time for m in episode_data),
            'avg_waiting_time': np.mean([m.waiting_time for m in episode_data]),
            'total_phase_switches': sum(m.phase_switches for m in episode_data),
            'avg_throughput': np.mean([m.throughput for m in episode_data]),
            'duration': episode_data[-1].timestamp - episode_data[0].timestamp if episode_data else 0
        }
    
    def export_to_csv(self, filepath: str):
        """导出到CSV文件"""
        try:
            data = [asdict(m) for m in self.metrics_history]
            df = pd.DataFrame(data)
            df.to_csv(filepath, index=False)
            logger.info(f"性能数据已导出到: {filepath}")
        except Exception as e:
            logger.error(f"导出CSV失败: {e}")


class PerformanceAnalyzer:
    """性能分析器"""
    
    def __init__(self, collector: MetricsCollector):
        self.collector = collector
    
    def analyze_training_progress(self) -> Dict[str, Any]:
        """分析训练进度"""
        if len(self.collector.metrics_history) < 100:
            return {"status": "insufficient_data"}
        
        # 获取最近的数据
        recent_100 = list(self.collector.metrics_history)[-100:]
        recent_500 = list(self.collector.metrics_history)[-500:] if len(self.collector.metrics_history) >= 500 else recent_100
        
        # 计算趋势
        rewards_100 = [m.reward for m in recent_100]
        rewards_500 = [m.reward for m in recent_500]
        
        trend_100 = np.polyfit(range(len(rewards_100)), rewards_100, 1)[0]
        trend_500 = np.polyfit(range(len(rewards_500)), rewards_500, 1)[0]
        
        # 性能评估
        avg_reward_100 = np.mean(rewards_100)
        avg_reward_500 = np.mean(rewards_500)
        
        # 稳定性评估
        reward_std_100 = np.std(rewards_100)
        reward_std_500 = np.std(rewards_500)
        
        return {
            "status": "analyzing",
            "recent_performance": {
                "avg_reward_100": avg_reward_100,
                "avg_reward_500": avg_reward_500,
                "trend_100": trend_100,
                "trend_500": trend_500,
                "stability_100": reward_std_100,
                "stability_500": reward_std_500
            },
            "assessment": self._assess_performance(avg_reward_100, trend_100, reward_std_100)
        }
    
    def _assess_performance(self, avg_reward: float, trend: float, stability: float) -> str:
        """评估性能"""
        if avg_reward > 0 and trend > 0 and stability < 100:
            return "excellent"
        elif avg_reward > -500 and trend > -1 and stability < 200:
            return "good"
        elif avg_reward > -1000 and trend > -5:
            return "fair"
        else:
            return "poor"
    
    def detect_anomalies(self, window: int = 50) -> List[Dict[str, Any]]:
        """检测异常"""
        if len(self.collector.metrics_history) < window * 2:
            return []
        
        anomalies = []
        recent_metrics = list(self.collector.metrics_history)[-window * 2:]
        
        # 计算基线
        baseline_rewards = [m.reward for m in recent_metrics[:window]]
        baseline_mean = np.mean(baseline_rewards)
        baseline_std = np.std(baseline_rewards)
        
        # 检测最近的异常
        recent_rewards = [m.reward for m in recent_metrics[window:]]
        
        for i, reward in enumerate(recent_rewards):
            z_score = abs(reward - baseline_mean) / max(baseline_std, 1.0)
            
            if z_score > 3.0:  # 3-sigma规则
                anomalies.append({
                    "type": "reward_anomaly",
                    "step": recent_metrics[window + i].step,
                    "value": reward,
                    "z_score": z_score,
                    "severity": "high" if z_score > 5.0 else "medium"
                })
        
        return anomalies
    
    def generate_report(self) -> Dict[str, Any]:
        """生成性能报告"""
        progress = self.analyze_training_progress()
        anomalies = self.detect_anomalies()
        recent_perf = self.collector.get_recent_performance()
        
        return {
            "timestamp": datetime.now().isoformat(),
            "total_steps": len(self.collector.metrics_history),
            "training_progress": progress,
            "recent_performance": recent_perf,
            "anomalies": anomalies,
            "recommendations": self._generate_recommendations(progress, anomalies, recent_perf)
        }
    
    def _generate_recommendations(self, progress: Dict, anomalies: List, recent_perf: Dict) -> List[str]:
        """生成建议"""
        recommendations = []
        
        if progress.get("status") == "insufficient_data":
            recommendations.append("继续训练以收集足够的数据进行分析")
            return recommendations
        
        assessment = progress.get("recent_performance", {}).get("assessment", "unknown")
        
        if assessment == "poor":
            recommendations.append("考虑调整学习率或奖励函数参数")
            recommendations.append("检查环境配置是否正确")
        elif assessment == "fair":
            recommendations.append("可以尝试增加训练步数")
            recommendations.append("考虑微调超参数")
        elif assessment == "good":
            recommendations.append("训练效果良好，可以继续当前配置")
        elif assessment == "excellent":
            recommendations.append("训练效果优秀，可以考虑保存当前模型")
        
        # 基于异常的建议
        if len(anomalies) > 5:
            recommendations.append("检测到多个异常，建议检查训练稳定性")
        
        # 基于资源使用的建议
        if recent_perf.get("avg_memory_mb", 0) > 2048:
            recommendations.append("内存使用较高，考虑减少批次大小")
        
        if recent_perf.get("avg_step_time", 0) > 1.0:
            recommendations.append("步骤执行时间较长，考虑优化环境或关闭GUI")
        
        return recommendations


class RealTimeMonitor:
    """实时监控器"""
    
    def __init__(self, update_interval: float = 10.0):
        self.update_interval = update_interval
        self.last_update = 0
        self.collector = MetricsCollector()
        self.analyzer = PerformanceAnalyzer(self.collector)
        
    def should_update(self) -> bool:
        """检查是否应该更新"""
        current_time = time.time()
        if current_time - self.last_update >= self.update_interval:
            self.last_update = current_time
            return True
        return False
    
    def update(self, episode: int, step: int, metrics: Dict[str, Any]):
        """更新监控数据"""
        self.collector.record_step_metrics(episode, step, metrics)
        
        if self.should_update():
            self._print_status()
    
    def _print_status(self):
        """打印状态信息"""
        recent_perf = self.collector.get_recent_performance()
        
        if recent_perf:
            logger.info(f"性能监控 - "
                       f"平均奖励: {recent_perf.get('avg_reward', 0):.2f}, "
                       f"等待时间: {recent_perf.get('avg_waiting_time', 0):.2f}, "
                       f"内存: {recent_perf.get('avg_memory_mb', 0):.1f}MB")


# 全局监控实例
performance_monitor = RealTimeMonitor()


def monitor_performance(func):
    """性能监控装饰器"""
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        duration = time.time() - start_time
        
        performance_monitor.collector.record_step_time(duration)
        
        return result
    
    return wrapper
