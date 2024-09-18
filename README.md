# 基于深度强化学习的机器人无地图导航系统

这个项目实现了一个基于ROS (Robot Operating System)和深度强化学习的机器人无地图导航系统。该系统使用DDPG (Deep Deterministic Policy Gradient)算法来训练机器人在未知环境中进行导航。

## 主要特性

- 使用DDPG算法训练机器人导航策略
- 支持在Gazebo仿真环境中训练和评估
- 可以部署到实际的Turtlebot2机器人上
- 包含标准DDPG的实现
- 使用ROS进行机器人控制和传感器数据处理
- 支持激光雷达和里程计数据输入

## 系统要求

- ROS Kinetic 
- Gazebo 7或更高版本
- Python 3
- PyTorch
- NumPy
- Shapely

## 安装

1. 克隆此仓库到您的catkin工作空间的src目录:

2. 编译ROS包:

```
cd ~/catkin_ws
catkin_make
```

3. 安装Python依赖:

```
pip install torch numpy shapely
```

## 使用方法

1. 启动Gazebo仿真环境:

```
roslaunch turtlebot_gazebo turtlebot_world.launch
```

2. 运行DDPG训练脚本:


```13:22:scripts/train_ddpg/train_ddpg.py
def do_training(run_name="DDPG_R1", exp_name="Rand_R1", episode_num=(100, 200, 300, 400),
               iteration_num_start=(200, 300, 400, 500), iteration_num_step=(1, 2, 3, 4),
               iteration_num_max=(1000, 1000, 1000, 1000),
               max_speed=0.5, min_speed=0.05, save_steps=10000,
               env_epsilon=(0.9, 0.6, 0.6, 0.6), env_epsilon_decay=(0.999, 0.9999, 0.9999, 0.9999),
               laser_half_num=9, laser_min_dis=0.35, scan_overall_num=36, goal_dis_min_dis=0.3,
               obs_reward=-20, goal_reward=30, goal_dis_amp=15, goal_th=0.5, obs_th=0.35,
               state_num=22, action_num=2, is_pos_neg=False, is_poisson=False, poisson_win=50,
               memory_size=100000, batch_size=256, epsilon_end=0.1, rand_start=10000, rand_decay=0.999,
               rand_step=2, target_tau=0.01, target_step=1, use_cuda=True):
```


3. 评估训练好的模型:


```182:202:evaluation/eval_real_world/run_ddpg_eval_rw.py
if __name__ == "__main__":
    WEIGHT_FILE = '../saved_model/ddpg_poisson.pt'
    GOAL_LIST = [[7.3, 2.5], [7.7, -3], [5.3, -5], [0, -5.2], [2.0, -9.2], [11.4, -9.2],
                 [13, -6], [11, -4.5], [13.5, -10], [10, -13.5], [7, -17], [1, -17], [0.5, -15.2], [7, -13.2], [0, -12.7]]
    USE_CUDA = True
    IS_POS_NEG = True
    IS_POISSON = True
    STATE_NUM = 18 + 4
    if IS_POS_NEG:
        RESCALE_STATE_NUM = STATE_NUM + 2
    else:
        RESCALE_STATE_NUM = STATE_NUM
    ACTION_NUM = 2
    POISSON_WIN = 50
    RAND_END = 0.01
    agent = Agent(STATE_NUM, ACTION_NUM, RESCALE_STATE_NUM, poisson_window=POISSON_WIN, use_poisson=IS_POISSON,
                  epsilon_end=RAND_END, use_cuda=USE_CUDA)
    agent.load(WEIGHT_FILE)
    sim_time = 400
    ros_node = RosNode(agent, GOAL_LIST, sim_time, is_pos_neg=IS_POS_NEG)
    record_pos, record_time, record_dis = ros_node.run_ros()
```


## 项目结构

- ros/catkin_ws: ROS工作空间
  - turtlebot_description: Turtlebot机器人的3D模型描述
  - simple_laserscan: 处理激光扫描数据的ROS包
- scripts: 主要的训练和环境代码
  - train_ddpg: DDPG训练相关代码
  - environment.py: Gazebo环境交互代码
- evaluation: 评估相关代码