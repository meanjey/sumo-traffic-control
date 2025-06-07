# 智能交通灯强化学习项目

本项目使用强化学习算法 (PPO, Proximal Policy Optimization) 来训练一个智能交通信号灯控制器，旨在优化城市交通路口的车辆通行效率。该项目基于 [SUMO (Simulation of Urban MObility)](https://www.eclipse.org/sumo/) 交通仿真软件和 [Stable-Baselines3](https://stable-baselines3.readthedocs.io/) 强化学习库。

## 快速上手指南 (给队友)

你好！要运行这个项目，请按以下两个简单步骤操作：

**第一步：安装环境**

1.  **安装 SUMO**: 如果你电脑上没有，请先去[官网](https://www.eclipse.org/sumo/docs/Downloads.php)下载安装。
2.  **安装 Python 库**: 打开终端，进入这个项目文件夹，然后运行下面这行命令：
    ```bash
    pip install -r requirements.txt
    ```

**第二步：运行脚本**

直接运行主程序，然后根据菜单提示进行选择：
```bash
# 记得把 "你的SUMO安装路径" 换成你自己的
$env:SUMO_HOME="你的SUMO安装路径"; python main.py
```
脚本启动后，你会看到一个菜单，让你选择是**训练**还是**测试**模型，以及是否需要**显示图形界面**。

---

## 功能
- 使用PPO算法训练交通信号灯控制模型。
- 通过SUMO进行逼真的交通流仿真。
- 支持带GUI的可视化仿真和无GUI的后台高效训练。
- 使用TensorBoard实时监控训练过程中的奖励、损失等关键指标。
- 支持加载已训练好的模型进行性能测试和演示。

## 目录结构
```
.
├── main.py                    # 主程序：包含环境、控制器和训练逻辑
├── new_additional.xml         # SUMO附加文件，定义检测器
├── new_network.net.xml        # SUMO路网文件
├── new_routes.rou.xml         # SUMO车流定义文件
├── new_simulation.sumocfg     # SUMO仿真主配置文件
├── ppo_tensorboard_logs/      # (训练后生成) TensorBoard日志目录
├── ppo_traffic_light_controller.zip # (训练后生成) 保存的PPO模型
├── README.md                  # 本文档
└── requirements.txt           # Python依赖库
```

## 环境设置

### 1. 安装SUMO
请从[SUMO官方网站](https://www.eclipse.org/sumo/docs/Downloads.php)下载并安装SUMO。安装完成后，请确保将SUMO的`bin`目录添加到了系统的环境变量 `SUMO_HOME` 中。脚本会自动根据此环境变量寻找SUMO可执行文件。

例如，如果您的SUMO安装在 `K:\sumo`，请设置环境变量 `SUMO_HOME=K:\sumo`。

### 2. 安装Python依赖
本项目使用Python 3.8+。首先，建议创建一个虚拟环境。然后，在项目根目录下，通过`pip`安装所有必需的库：
```bash
pip install -r requirements.txt
```

## 使用方法

### 运行程序
要启动本项目，只需运行主脚本。请确保已设置 `SUMO_HOME` 环境变量。

```bash
# 在Windows (PowerShell)上:
$env:SUMO_HOME="你的SUMO安装路径"; python main.py

# 在Linux/macOS上:
export SUMO_HOME="你的SUMO安装路径" && python main.py
```

程序启动后，会显示一个菜单供您选择：
1.  **后台训练 (无GUI，效率最高):** 选择此项开始一个新的训练任务，过程在后台运行。
2.  **可视化训练 (有GUI，方便观察):** 选择此项开始训练，并实时显示SUMO仿真界面。
3.  **测试已保存的模型 (有GUI):** 选择此项加载已有的 `ppo_traffic_light_controller.zip` 模型，并在GUI模式下进行测试和演示。

### 监控训练过程
在模型训练的同时 (选择模式1或2后)，您可以打开**一个新的终端**来启动TensorBoard，实时查看训练指标。
1.  首先，确保您在新终端中也进入了项目根目录。
2.  运行以下命令：
    ```bash
    tensorboard --logdir ./ppo_tensorboard_logs/
    ```
3.  在浏览器中打开TensorBoard提供的URL (通常是 `http://localhost:6006`)。在`SCALARS`选项卡下，您可以找到 `rollout/ep_rew_mean` 图表，它显示了每个回合的平均奖励，这是判断模型是否在改进的关键指标。

### 测试已训练的模型
在您或您的队友训练好一个模型 (`ppo_traffic_light_controller.zip`) 之后，可以使用以下命令在GUI模式下加载并测试它：
```bash
# 在Windows (PowerShell)上:
$env:SUMO_HOME="你的SUMO安装路径"; python main.py --test

# 在Linux/macOS上:
export SUMO_HOME="你的SUMO安装路径" && python main.py --test
```
该命令会自动启动SUMO的GUI界面，并运行一个完整的仿真回合，让您可以直观地观察智能体的决策效果。

## 系统原理

自适应交通信号灯控制系统基于以下原理：

1. **交通流检测**：使用SUMO的API获取各个方向的车辆数量和等待时间
2. **相位时长计算**：根据交通流量和等待时间计算最优的相位持续时间
3. **相位切换**：根据计算结果动态调整信号灯相位
4. **性能评估**：收集和分析交通性能指标，评估控制效果

## 未来改进

1. 引入机器学习算法，如强化学习，进一步优化信号灯控制策略
2. 增加更多的交通场景和路网结构
3. 优化多路口协调控制算法
4. 增加可视化界面，直观展示控制效果和性能指标 

---

## 更新日志 (Changelog)

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