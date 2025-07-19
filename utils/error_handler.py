#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
错误处理和恢复系统
"""

import logging
import traceback
import time
import functools
from typing import Any, Callable, Optional, Dict
from pathlib import Path
import psutil
import os

logger = logging.getLogger(__name__)


class ErrorHandler:
    """错误处理器"""
    
    def __init__(self):
        self.error_counts = {}
        self.max_retries = 3
        self.retry_delay = 1.0
        
    def retry_on_error(self, max_retries: int = 3, delay: float = 1.0, 
                      exceptions: tuple = (Exception,)):
        """重试装饰器"""
        def decorator(func: Callable) -> Callable:
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                last_exception = None
                
                for attempt in range(max_retries + 1):
                    try:
                        return func(*args, **kwargs)
                    except exceptions as e:
                        last_exception = e
                        if attempt < max_retries:
                            logger.warning(f"函数 {func.__name__} 第 {attempt + 1} 次尝试失败: {e}")
                            time.sleep(delay * (attempt + 1))  # 递增延迟
                        else:
                            logger.error(f"函数 {func.__name__} 所有重试都失败")
                
                raise last_exception
            return wrapper
        return decorator
    
    def safe_execute(self, func: Callable, default_return: Any = None, 
                    log_errors: bool = True) -> Any:
        """安全执行函数"""
        try:
            return func()
        except Exception as e:
            if log_errors:
                logger.error(f"安全执行失败: {e}")
                logger.debug(traceback.format_exc())
            return default_return
    
    def track_error(self, error_key: str, max_count: int = 10) -> bool:
        """跟踪错误次数"""
        self.error_counts[error_key] = self.error_counts.get(error_key, 0) + 1
        
        if self.error_counts[error_key] >= max_count:
            logger.critical(f"错误 {error_key} 达到最大次数 {max_count}")
            return False
        
        return True


class ResourceMonitor:
    """资源监控器"""
    
    def __init__(self):
        self.process = psutil.Process()
        
    def check_memory_usage(self, threshold_mb: float = 2048) -> bool:
        """检查内存使用"""
        try:
            memory_mb = self.process.memory_info().rss / 1024 / 1024
            if memory_mb > threshold_mb:
                logger.warning(f"内存使用过高: {memory_mb:.1f}MB > {threshold_mb}MB")
                return False
            return True
        except:
            return True
    
    def check_cpu_usage(self, threshold_percent: float = 90.0) -> bool:
        """检查CPU使用"""
        try:
            cpu_percent = self.process.cpu_percent()
            if cpu_percent > threshold_percent:
                logger.warning(f"CPU使用过高: {cpu_percent:.1f}% > {threshold_percent}%")
                return False
            return True
        except:
            return True
    
    def check_disk_space(self, path: str = ".", threshold_gb: float = 1.0) -> bool:
        """检查磁盘空间"""
        try:
            disk_usage = psutil.disk_usage(path)
            free_gb = disk_usage.free / 1024 / 1024 / 1024
            if free_gb < threshold_gb:
                logger.warning(f"磁盘空间不足: {free_gb:.1f}GB < {threshold_gb}GB")
                return False
            return True
        except:
            return True
    
    def get_system_info(self) -> Dict[str, Any]:
        """获取系统信息"""
        try:
            return {
                'memory_mb': self.process.memory_info().rss / 1024 / 1024,
                'cpu_percent': self.process.cpu_percent(),
                'disk_free_gb': psutil.disk_usage('.').free / 1024 / 1024 / 1024,
                'pid': self.process.pid,
                'threads': self.process.num_threads(),
            }
        except Exception as e:
            logger.error(f"获取系统信息失败: {e}")
            return {}


class SUMOConnectionManager:
    """SUMO连接管理器"""
    
    def __init__(self):
        self.connection_attempts = 0
        self.max_attempts = 5
        self.last_error_time = 0
        self.error_cooldown = 30  # 30秒冷却时间
        
    def is_connection_healthy(self) -> bool:
        """检查连接健康状态"""
        try:
            import traci
            traci.simulation.getTime()
            return True
        except:
            return False
    
    def can_retry_connection(self) -> bool:
        """检查是否可以重试连接"""
        current_time = time.time()
        
        # 检查冷却时间
        if current_time - self.last_error_time < self.error_cooldown:
            return False
        
        # 检查重试次数
        if self.connection_attempts >= self.max_attempts:
            return False
        
        return True
    
    def record_connection_attempt(self, success: bool):
        """记录连接尝试"""
        if success:
            self.connection_attempts = 0
        else:
            self.connection_attempts += 1
            self.last_error_time = time.time()
    
    def reset_connection_state(self):
        """重置连接状态"""
        self.connection_attempts = 0
        self.last_error_time = 0


class LogManager:
    """日志管理器"""
    
    def __init__(self, log_dir: str = "logs"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)
        self.setup_logging()
    
    def setup_logging(self):
        """设置日志系统"""
        # 创建格式化器
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        # 文件处理器
        file_handler = logging.FileHandler(
            self.log_dir / "traffic_control.log",
            encoding='utf-8'
        )
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(formatter)
        
        # 错误文件处理器
        error_handler = logging.FileHandler(
            self.log_dir / "errors.log",
            encoding='utf-8'
        )
        error_handler.setLevel(logging.ERROR)
        error_handler.setFormatter(formatter)
        
        # 控制台处理器
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(formatter)
        
        # 配置根日志器
        root_logger = logging.getLogger()
        root_logger.setLevel(logging.INFO)
        
        # 清除现有处理器
        for handler in root_logger.handlers[:]:
            root_logger.removeHandler(handler)
        
        # 添加新处理器
        root_logger.addHandler(file_handler)
        root_logger.addHandler(error_handler)
        root_logger.addHandler(console_handler)
    
    def log_system_info(self):
        """记录系统信息"""
        monitor = ResourceMonitor()
        info = monitor.get_system_info()
        logger.info(f"系统状态: {info}")
    
    def log_training_start(self, config: Dict[str, Any]):
        """记录训练开始"""
        logger.info("=" * 50)
        logger.info("训练开始")
        logger.info(f"配置: {config}")
        self.log_system_info()
        logger.info("=" * 50)
    
    def log_training_end(self, success: bool, duration: float):
        """记录训练结束"""
        logger.info("=" * 50)
        status = "成功" if success else "失败"
        logger.info(f"训练结束: {status}, 耗时: {duration:.2f}秒")
        self.log_system_info()
        logger.info("=" * 50)


# 全局实例
error_handler = ErrorHandler()
resource_monitor = ResourceMonitor()
sumo_connection_manager = SUMOConnectionManager()
log_manager = LogManager()


def safe_sumo_call(func: Callable, default_return: Any = None) -> Any:
    """安全的SUMO调用"""
    if not sumo_connection_manager.is_connection_healthy():
        logger.warning("SUMO连接不健康，跳过调用")
        return default_return
    
    return error_handler.safe_execute(func, default_return)


def monitor_resources(func: Callable) -> Callable:
    """资源监控装饰器"""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        # 检查资源
        if not resource_monitor.check_memory_usage():
            logger.warning("内存使用过高，可能影响性能")
        
        if not resource_monitor.check_disk_space():
            logger.error("磁盘空间不足，停止执行")
            raise RuntimeError("磁盘空间不足")
        
        return func(*args, **kwargs)
    
    return wrapper
