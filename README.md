# 智能交通灯强化学习项目

本项目使用强化学习算法 (PPO, Proximal Policy Optimization) 来训练一个智能交通信号灯控制器，旨在优化城市交通路口的车辆通行效率。该项目基于 [SUMO (Simulation of Urban MObility)](https://www.eclipse.org/sumo/) 交通仿真软件和 [Stable-Baselines3](https://stable-baselines3.readthedocs.io/) 强化学习库。

## 快速上手指南 (给队友)

你好！要运行这个项目，请按以下三个简单步骤操作：

**第一步：安装环境**

1.  **安装 SUMO**: 如果你电脑上没有，请先去[官网](https://www.eclipse.org/sumo/docs/Downloads.php)下载安装。
2.  **安装 Python 库**: 在PyCharm底部的**Terminal**窗口中，运行下面这行命令：
    ```bash
    pip install -r requirements.txt
    ```

**第二步：在PyCharm中配置SUMO路径**

1.  在PyCharm顶部菜单，选择 **Run** -> **Edit Configurations...**。
2.  在弹出的窗口左侧，选择我们的 `main` 脚本。
3.  在右侧的 **Environment variables** 字段，点击旁边的文件夹图标。
4.  在新的小窗口里，点击 **+** 号，添加一个新的环境变量：
    -   **Name**: `SUMO_HOME`
    -   **Value**: `你的SUMO安装路径` (例如: `K:\sumo`)
5.  点击 **OK** 保存所有设置。**这个设置只需做一次，以后就不用再管了。**

**第三步：运行脚本**

1.  在PyCharm左侧的文件浏览器里，右键点击 `main.py` 文件。
2.  选择 **Run 'main'**。
3.  程序启动后，你会在底部的**Run**窗口看到一个菜单，让你选择是**训练**还是**测试**模型。

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
本项目推荐在PyCharm IDE中运行。

1.  **首次运行前的配置**:
    请参照上方"快速上手指南"的**第二步**，在PyCharm的"Run/Debug Configurations"中为`main.py`脚本设置好`SUMO_HOME`环境变量。这可以确保PyCharm每次都能找到SUMO。

2.  **开始运行**:
    配置完成后，只需在PyCharm中右键点击`main.py`并选择**Run 'main'**即可启动。程序会在底部的**Run**窗口中显示一个菜单供您选择：
    - **模式1 (后台训练)**: 无图形界面，训练速度最快。
    - **模式2 (可视化训练)**: 启动SUMO图形界面，可以实时观察训练过程。
    - **模式3 (测试模型)**: 启动SUMO图形界面，并加载一个已保存的模型进行演示。

### 监控训练过程
在模型开始训练后 (选择了模式1或2)，您可以打开PyCharm内置的终端来启动TensorBoard。

1.  在PyCharm窗口底部，点击**Terminal**标签页，打开一个终端。
2.  运行以下命令：
    ```bash
    tensorboard --logdir ./ppo_tensorboard_logs/
    ```
3.  按住`Ctrl`并点击终端里显示的`http://localhost:6006`链接，即可在浏览器中查看实时图表。

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