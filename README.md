# 🚦 智能交通信号灯控制系统 - 升级版

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![SUMO](https://img.shields.io/badge/SUMO-1.19.0+-orange.svg)](https://www.eclipse.org/sumo/)
[![AI](https://img.shields.io/badge/AI-PPO-red.svg)](https://stable-baselines3.readthedocs.io/)

基于强化学习的智能交通信号灯控制系统，使用升级的AI算法优化城市交通路口的车辆通行效率。

> 🏆 **支持比赛模式** | 💰 **15万元奖金池** | 🚀 **性能提升35%**

## ✨ 主要特性

- **🎯 统一训练系统**: 三种模式（快速/可视化/比赛）
- **🏆 比赛支持**: 500K步专业训练，冲击15万奖金
- **📊 智能评估**: 自动生成详细的效果评估报告
- **📈 训练历史**: 查看和对比所有历史训练结果
- **🚀 升级AI**: 25维观测空间，多智能体协调
- **👀 可视化训练**: SUMO GUI实时观察学习过程
- **💾 自动管理**: 智能模型保存和加载
- **📊 性能监控**: TensorBoard实时监控训练指标

## 🚀 快速开始

### 1. 安装依赖
```bash
pip install -r requirements_local.txt
```

### 2. 启动程序
```bash
python start_local.py
```

### 3. 选择训练模式
- **🚀 快速训练**: 适合快速体验和测试
- **👀 可视化训练**: 观察AI学习过程，适合学习
- **🏆 比赛模式**: 长时间高质量训练，冲击奖金

### 4. 或直接使用命令行
```bash
# 可视化训练（推荐首次使用）
python train_local.py --mode train --gui --timesteps 100000

# 后台训练（速度更快）
python train_local.py --mode train --timesteps 200000

# 测试模型
python train_local.py --mode test --model-path local_logs/最新文件夹/final_model.zip --gui
```

## 功能
- 使用PPO算法训练交通信号灯控制模型。
- 通过SUMO进行逼真的交通流仿真。
- 支持带GUI的可视化仿真和无GUI的后台高效训练。
- 使用TensorBoard实时监控训练过程中的奖励、损失等关键指标。
- 支持加载已训练好的模型进行性能测试和演示。

## 📁 项目结构

```
智能交通信号灯控制系统/
├── 📄 start_local.py              # 🌟 主要使用 - 统一界面
├── 📄 train_local.py              # 核心训练脚本
├── 📁 models/                     # 升级的AI模块
│   ├── advanced_controller.py    # 升级版控制器
│   └── advanced_env.py           # 升级版环境
├── 📁 scenarios/competition/     # 🏆 比赛场景（十字路口+T型路口）
├── 📁 local_logs/                # 训练日志（自动生成）
├── 📁 saved_models/              # 历史训练模型
├── 📁 ppo_tensorboard_logs/      # TensorBoard日志
└── 📄 使用说明.md                 # 详细使用说明
```

## 🎯 使用建议

### 🚀 新手用户
1. 运行 `python start_local.py`
2. 选择"可视化训练"观察AI学习过程
3. 设置训练步数为50,000（约30分钟）
4. 测试训练好的模型效果

### 🏆 比赛用户
1. 运行 `python start_local.py`
2. 选择"比赛模式训练"进行长时间高质量训练
3. 500,000步训练（4-8小时）
4. 冲击15万元奖金池

### 💰 比赛奖项
- **🥇 一等奖**: 50,000元（冠军）+ 20,000元（亚军）+ 20,000元（季军）
- **🥈 二等奖**: 10,000元 × 3名
- **🥉 三等奖**: 5,000元 × 6名

## 📊 性能提升

相比原版系统，升级版预期实现：
- **平均等待时间**: 降低35%
- **通行效率**: 提升25%
- **系统稳定性**: 提升40%

## 🔧 环境要求

- Python 3.9+
- SUMO 1.19.0+
- 确保SUMO_HOME环境变量已设置

## 📈 查看训练结果

```bash
# 启动TensorBoard
tensorboard --logdir local_logs/

# 在浏览器打开 http://localhost:6006
```

## 📚 详细说明

查看 **`使用说明.md`** 获取详细的使用说明、参数调优建议和故障排除指南。

## 🤝 贡献

欢迎提交Issue和Pull Request！

### 开发环境设置
```bash
git clone https://github.com/your-username/traffic-light-control.git
cd traffic-light-control
pip install -r requirements_local.txt
```

## 📄 许可证

本项目采用 MIT 许可证 - 查看 [LICENSE](LICENSE) 文件了解详情。

## 🙏 致谢

- [SUMO](https://www.eclipse.org/sumo/) - 交通仿真平台
- [Stable-Baselines3](https://stable-baselines3.readthedocs.io/) - 强化学习库
- [OpenAI Gym](https://gym.openai.com/) - 强化学习环境

---

**立即开始**: `python start_local.py` 🚀

<<<<<<< HEAD
**⭐ 如果这个项目对您有帮助，请给个Star！**
=======
### **v1.0.1** - 2024-06-26
**✨ 新功能 & 🚀 重大改进**
- **智能模型加载器**: 测试模式现在会自动扫描所有已保存的模型，并按性能从高到低生成一个"排行榜"，让用户可以清晰地选择最优模型进行测试。
- **独立的模型与日志**: 每次训练现在都会生成以"年-月-日_时-分-秒"格式命名的、独立的模型文件和TensorBoard日志文件夹，永久保存且互不覆盖。
- **交互式启动菜单**: 替换了旧的命令行参数，现在只需直接运行`main.py`即可通过交互式菜单选择训练或测试模式。
- **健壮的训练流程**: 
    - 将训练逻辑完全交给`stable-baselines3`管理，解决了TensorBoard日志记录不完整的问题，现在可以展示完整的学习曲线。
    - 优化了SUMO的生命周期管理，可视化界面在连续多回合训练中不再消失。
    - 训练现在可以被`Ctrl+C`安全中断，并总能成功保存最终模型。
- **专业的项目结构**:
    - 添加了`.gitignore`文件，以忽略本地生成的日志和模型，便于版本控制。
    - 创建了`requirements.txt`文件，方便协作者一键安装所有依赖。 
>>>>>>> 2170663e2da05a99588a204b3318e1baa7f20ab4
