# 1. Hello World!
hello! 作为一个刚接触RL robotics的小白，你一定有很多的疑问，不知道从哪里开始学习，下面是整理好的一个pipeline，帮助你循序渐进的知道自己需要了解什么，该学什么，该干什么。俗话说兴趣是最好的老师，如果遇到困难也请不要放弃，多问问老师学长，总会突破的！（笔者入门是想复刻迪士尼机器人研究院的BD-X机器人才一步步攻坚克难的）

最好提前掌握的知识和技能（当然不会也没事儿，后面都能学会）：
- 深度学习基础
- 机器人学原理
- Python，Pytorch
- 了解ROS咋个通信方式等

## 1.1 为什么使用RL？
传统方法使用MPC进行base位置以及落足点位置的预测，然后使用WBC的方法将位置信号转化为各个关节的角度信息，
<img width="1212" height="516" alt="image" src="https://github.com/user-attachments/assets/69c797fa-eda1-4541-b370-9c2b7c93f58c" />

## 1.2 为什么大部分论文与项目都使用Isaac仿真？
Isaac 是什么见 [第 3 节 仿真环境](#section-3)。Isaac是英伟达开发的仿真环境，并行性好，收集数据效率高。在以Isaac为代表的GPU并行环境之前，大多使用CPU集群运行并行环境，例如habitat等，训练效率等不高。
<img width="1049" height="453" alt="image" src="https://github.com/user-attachments/assets/7e525fef-604c-4433-acd0-34fa39c230b9" />
而基于Isaac的项目Legged gym贡献：
1. 提供成熟的四足机器人封装，以及地形的生成代码

2. Physx引擎进行环境的Reset或者update速度很慢，legged gym创建一个巨大的地形，并按照行列划分，在行维度上生成不同类型的地形，在列维度上生成不同的难度

3. 对于不同的并行四足机器人环境来说，地形是共享的，但是不同环境之间并不会发生影响或碰撞


# 2. 强化学习理论
鲁迅说过：要用理论指导实践。一上来就做项目复现肯定是一头雾水的，所以需要先打好理论上的基础，知道什么是强化学习，怎么用强化学习去实现机器人的控制等等。对于理论知识部分，这里最推荐b站上西湖大学赵世钰老师的[链接-强化学习课程](https://www.bilibili.com/video/BV1sd4y167NS/?spm_id_from=333.337.search-card.all.click&vd_source=dbf0a5b21ee3e608136d2de2f7aa4035)

希望看视频的时候一定要拿出本认真推导和记录里面的知识点，但是赵老师的课就到Actor-Critic基本就结束了，但是相信看完了后你自己也有能力去互联网上搜索和学习比如TRPO、PPO、PPO2等主流算法，这里贴一个笔者的早期学PPO时记录的博客：[链接-PPO算法](https://www.cnblogs.com/myleaf/p/18595876)
写的比较详细，希望可以帮助到你。

<a id="section-3"></a>
# 3. 仿真环境
训练的仿真环境主要是Nvidia提供的两个，Isaac Gym和Isaac Sim。前者是最早推出的并行仿真训练环境，也是早期各种RL工作的基础，而后者相当于前者的一次迭代，将更多的函数实现封装成供用户灵活调用的API接口（如果你做过嵌入式设计，就会熟悉sim32的标准库以及CUBEMX的区别，这两个也类似），比如奖励函数的实现、雷达信息的调用等等。目前主流的使用一般是Isaac Lab，但是它上手一般来说会更难一些，所以作为初学者先从Isaac Gym开始。下面我会推荐几个比较适合入门的项目。

Isaac Gym安装包: [链接-Isaac Gym install](https://developer.nvidia.com/isaac-gym)
具体安装过程可参考[链接-具体安装](https://github.com/leggedrobotics/legged_gym?tab=readme-ov-file)

Isaac Lab中文文档: [链接-Isaac Lab install](https://docs.robotsfan.com/isaaclab/index.html)
具体安装过程可参考“上述文档”

# 4. 理论实践项目
鲁迅还说过：实践是检验真理的唯一标准。所以是时候检验下自己是否真的掌握了RL的理论知识了。这里主要推荐最经典也是最基础的工程项目就是[链接-legged gym](https://github.com/leggedrobotics/legged_gym)
这个是基于Isaac Gym实现的对四足机器人的兼容框架，由苏黎世联邦理工大学和Nvidia联合开源（ETH，后面会反复提到这所在Robotics Engineering领域国际顶尖的学校），这个训练框架是目前所有框架的重要基础，只有真正弄懂了里面的代码逻辑和工程实现才算真正入门了这个方向。

还有一个同样是ETH的RSL_RL库（**R**obotics **S**ystem **L**ab _ **R**einforcement **L**earning）：[链接-rsl_rl](https://github.com/leggedrobotics/rsl_rl)
这个库是对PPO的代码的具体实现，这里我们主要关注下面的几个文件即可：
```
rsl_rl/
├── algorithms/        %这个文件是PPO算法的实现部分，包括损失函数、actor critic的mlp的更新逻辑，包含了models的网络初始化
├── env/               %这个是环境env.py的封装，为算法库提供了比如step、get_observations等接口的调用
├── extensions/        %一些拓展，比如镜像设置、随机网络蒸馏等（雅加达游戏）
├── models/            %这个是模块的调用
├── modules/           %一些比如mlp、cnn等模块的代码实现
├── runners/on_policy_runner.py %主要在train.py中调用的函数文件，相当于是一个main.py，里面集合了algorithms、modules等所有的实现
├── storage/           %进行环境rollout和代码，定义了replay buffer
├── utils/             %一些工具
└── __init__.py
```

这里最好一定要对照着赵老师的强化学习课程和这个RSL-RL代码进行学习，这样才会明白每一步都具体是（What）怎么干什么的，（Why）为什么这么做以及（How）如何实现的。

# 5. 主流任务
目前主流的机器人有四足机器人、轮足机器人（包括四轮足、双轮足等）以及人形机器人等，这些机器人的RL运动控制任务主要有两类：

- 一类是locomotion的任务，也就是运动学的任务。这种任务基本有两种实现方法，一种是通过Lidar（激光雷达）、depth camera（深度相机）等设备来感知外界地形信息，比如说草丛楼梯等；另一种是不使用上述这些感知设备，也就是“无感知的盲人”，一般对于四足机器人就叫“盲狗”（其他机器人也可以代指），这种实现方法一般对于一些简单的任务比较实用，而且训练成本也比较低一些，但是对于地形更复杂的任务比如穿越雪地、草坪等，就表现的不是很好了。
  
- 另一类是loco-manipulation的任务，也就是运动加上肢控制的任务，这里上肢对于人形机器人就是字面意思，对于其他机器人一般指的是携带机械臂的操作性任务。
  
- 其实还应该有第三种，就是motion tracking的任务，这种任务对于人形机器人来说主要就是为了完成动作模仿任务，比方说跳舞、杂技等（详见2026宇树春节联欢晚会）、像人类一样的直膝行走等，而对于其他机器人主要就是各种步态的模仿，比如四足机器人希望走狼的步态、豹子的步态等等，使用的方法基本是基于AMP的动作对抗生成。

# 6. 经典论文推荐
这里为了便于了解和学习，我将一些比较经典的论文进行了整理和简单的说明，便于你们进行系统性的学习。
## 6.1 四足机器人
- **Learning Quadrupedal Locomotion over Challenging Terrain** 强化学习开山之作，盲狗
- RMA: Rapid Motor Adaptation for Legged Robots 师生网络蒸馏，使用更长时间的历史本体观测序列，历史观测编码器推理较慢无需实时
- Walk These Ways: Tuning Robot Control for Generalization with Multiplicity of Behavior 早期四足机器人多步态的探索
- CTS: Concurrent Teacher-Student Reinforcement Learning for Legged Locomotion 两阶段并为一阶段的训练方法
- DreamWaQ: Learning Robust Quadrupedal Locomotion With Implicit Terrain Imagination via Deep Reinforcement Learning 使用VAE进行特权信息估计
- Extreme Parkour with Legged Robots 使用height scan到深度相机的师生蒸馏策略
- Robot Parkour Learning 使用深度相机
- Hybrid Internal Model: Learning Agile Legged Locomotion with Simulated Robot Response 使用对比学习替换传统的loss回归
- Learning Robust and Agile Legged Locomotion Using Adversarial Motion Priors 使用AMP提供四足的步态模仿

## 6.2 （带臂）四足机器人
- Deep Whole-Body Control: Learning a Unified Policy for Manipulation and Locomotion 提出ROA这种师生网络交替更新的方法
- UMI on Legs: Making Manipulation Policies Mobile with Manipulation-Centric Whole-body Controllers 作者是UMI的提出者，用到了四足的WBC任务上
- Helpful DoggyBot: Open-World Object Fetching using Legged Robots and Vision-Language Models  用狗的leg或者mouth作为夹爪加持（比较fancy，不实用其实）
  
## 6.3 人形机器人 -- 动作跟踪模仿
- Learning Human-to-Humanoid Real-Time Whole-Body Teleoperation 何大佬的工作，人形动作模仿的开山之作
- OmniH2O: Universal and Dexterous Human-to-Humanoid Whole-Body Teleoperation and Learning 何大佬的工作，人形动作模仿的开山之作
- ASAP: Aligning Simulation and Real-World Physics for Learning Agile Humanoid Whole-Body Skills 2025年初实现科比跳投等高难度动作，采用real2sim2real的方法
- Retargeting Matters: General Motion Retargeting for Humanoid Motion Tracking  提供了人体重定向的方法，也就是生成机器人运动数据先验用于motion mimic的任务
- TWIST: Teleoperated Whole-Body Imitation System 和GMR同一个作者，方法继承实现动作跟踪
- Mobile-TeleVision: Predictive Motion Priors for Humanoid Whole-Body Control 将上下肢解耦控制 Exbody系列论文
- Expressive Whole-Body Control for Humanoid Robots  Exbody系列论文
- ExBody2: Advanced Expressive Humanoid Whole-Body Control  Exbody系列论文
- Learning Humanoid Standing-up Control across Diverse Postures 人形机器人分阶段起身，采用辅助力的策略比较有启发
- OmniRetarget: Interaction-Preserving Data Generation for Humanoid Whole-Body Loco-Manipulation and Scene Interaction  近期工作，与环境交互互动等
- OmniXtreme: Breaking the Generality Barrier in High-Dynamic Humanoid Control  宇树和高校联合的项目，实现机器人极端运动的跟踪和模仿，使用flow matching
- SONIC: Supersizing Motion Tracking for Natural Humanoid Whole-Body Control 英伟达出的人形机器人中的“通用模型框架”
- RobotDancing: Residual-Action Reinforcement Learning Enables Robust Long-Horizon Humanoid Motion Tracking 人形机器人出残差action辅助训练
- Track Any Motions under Any Disturbances 预测极端外力干扰，进行强鲁棒的跳舞任务
- Visual Imitation Enables Contextual Humanoid Control 从视频中学习跳舞动作
- CLONE: Closed-Loop Whole-Body Humanoid Teleoperation for Long-Horizon Tasks 北理工的VR动作跟踪，加里程计进行位置感知
- AMO: Adaptive Motion Optimization for Hyper-Dexterous Humanoid Whole-Body Control 动作跟踪，上下肢分离控制
- BeyondMimic: From Motion Tracking to Versatile Humanoid Control via Guided Diffusion 廖佬的开源动作跟踪框架，之前的MPC+WBC也是最初这位大佬开源的
- 
## 6.4 人形机器人 -- 全地形运动控制
- Humanoid Parkour Learning 人形感知跑酷
- Humanoid-Gym: Reinforcement Learning for Humanoid Robot with Zero-Shot Sim2Real Transfer 比较推荐的人形机器人初学的框架，能训练出来一个机器人，也可移植自己的机器人进去
- Perceptive Humanoid Parkour: Chaining Dynamic Human Skills via Motion Matching 使用原子技能库实现不同地形的跑酷，比如爬墙等
- Deep Whole-body Parkour 庄子文大佬的全身跑酷系列
- Attention-Based Map Encoding for Learning Generalized Legged Locomotion 使用注意力机制学习最优落足点
- BeamDojo: Learning Agile Humanoid Locomotion on Sparse Footholds 梅花桩行走的两阶段迁移任务，从平坦地形到复杂地形
 
## 6.5 VLA相关
to be continue~

# 7. 真机部署
```
"Sim2Sim2Real"
Isaac Gym\Lab -> Mujoco -> Real world
```

仿真到现实还是有一定差距的，因此可以现在mujoco中进行预部署，这个仿真环境相较于isaac系列会更接近真实世界（大概吧）。如果机器人用的是宇树的机器人的话（比如go2，go1或者g1），记住安全第一，记得先sim2sim到[链接-unitree mujoco](https://github.com/unitreerobotics/unitree_mujoco)，unitree mujoco中所有消息的收发方式、消息类型、关节顺序等与实机完全一致，在unitree mujoco中成功基本在实机中也能成功。

## 7.1 Q&A 现实与仿真之间的gap 
1. **现实的传感器存在误差**。    解决方法：在仿真中往观测数据里加噪声（legged gym/Isaac lab已实现）

2. **有些观测值没有传感器可以获得，比如线速度，周围地形的高度**。      解决方法：1.设计相关算法获得
这些数据，比如使用里程计估计线速度，使用SLAM建图估计周围高度    2.在仿真中使用蒸馏方法，利用现实中可以获取到的信息估计无法直接获取的信息（ legged gym/Isaac lab未实现）

3. **各种复杂地形**。    解决方法：仿真里的地形尽量全涵盖，只要让模型见过，基本就能正常通过

4. **机器人质量、关节摩擦、足部与地面的动静摩擦力不确定**。    解决方法：域随机化，也即仿真里并行跑的环境，设置不同的物理参数，让网络学会各中物理参数下应该怎么做，例如设置随机的摩擦力、kp kd的值、身体的质量等

