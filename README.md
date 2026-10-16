# S2A — State to Action

S2A 研究如何把世界模型预测的期望状态或状态增量，反解为固定机器人控制器可接收的 action，使执行后的实际状态接近期望状态。

```text
普通 VLA：观测/任务 → 模型 → action → 控制器 → 机器人/环境 → 实际 state

我们的路线：观测/任务 → 世界模型 → 期望 state / delta_state
                                  → S2A → action
                                  → 同一控制器 → 机器人/环境 → 期望接近的 state
```

将控制器及机器人/环境的前向响应记作 `F_C`，S2A 实现其有条件的近似逆 `G_C`：

```text
实际状态：s_next = F_C(s_now, action, history, dt)
反解命令：action = G_C(s_now, desired_state_or_trajectory, history, dt)
主要目标：F_C(s_now, G_C(...), history, dt) ≈ desired_state
```

`desired_state` 是世界模型给出的目标；`s_now` 是机器人实测反馈。相同状态变化可能对应多条命令，因此首先评估经过控制器后的状态跟踪效果，历史 action 的拟合误差作为辅助指标。第一阶段固定一款机器人及控制接口，冻结世界模型，用训练数据中的真实状态轨迹作为 oracle 目标隔离验证 S2A。

当前为项目基础结构与实验计划阶段，尚未实现转换器、拟合模型或完成闭环实验。详细数据预检、公式、对照组和验收标准见 [实施计划](plans/v0.0.1/001_state_to_action.md)。

待比较的方法包括：

| 方法 | 在 S2A 中的作用 |
| --- | --- |
| 解析几何与接口转换 | 解码 delta、生成目标轨迹、匹配 TCP/基座坐标系和控制单位 |
| 线性辨识与前馈校准 | 估计局部增益、耦合和延迟，构造近似逆 |
| 前馈 + P / PI / PD / PID | 根据目标与实测状态的误差及其历史进行闭环纠偏 |
| 有界增益调度 | 用少量参数表达状态相关的反馈增益 |
| 小型直接逆模型 / 残差网络 | 学习 state、目标与历史到命令的关系，或补偿解析/反馈模型残差 |

PID 是候选组件之一。当前基线保留原机器人底层控制器；S2A 输出其输入命令，包含必要的外层反馈。所有方法使用可比较的数据划分、控制配置和参考轨迹。

仓库结构：

```text
src/s2a/      可复用算法包；当前只有包入口
scripts/      后续数据抽取、标定和评估入口
configs/      版本管理中的实验配置和样本选择
tests/        后续单元与集成测试
plans/        实施计划
reports/      实验报告与精简结果
outputs/      生成的数据子集、模型、日志、曲线等；Git 忽略
```

后续实现、测试和产物均放在本仓库。Cosmos 仅为外部代码与数据语义参考；安装 S2A 不依赖 Cosmos、其训练 dataloader 或模型权重。源码证据与数据来源保存在计划中。

首次环境建立（Python 3.11+）：

```bash
cd /share/project/eai_pwm/home/hcr/repos/s2a
LD_LIBRARY_PATH='' python3 -m venv .venv
LD_LIBRARY_PATH='' .venv/bin/python -m pip install -e '.[dev]'
```

基础依赖为 NumPy、SciPy、PyArrow；开发依赖包括 pytest、Ruff 和 build。后续学习/仿真依赖随对应实现添加。开发环境的实际版本应随实验记录保存；当前未提供依赖锁文件。

```bash
.venv/bin/ruff check src tests scripts
.venv/bin/ruff format --check src tests scripts
LD_LIBRARY_PATH='' .venv/bin/python -m build --no-isolation --wheel
```

添加实际测试后，在本仓库执行 `LD_LIBRARY_PATH='' .venv/bin/python -m pytest`。当前没有算法测试，不把“未收集到测试”当作通过。

首个样本选择保存在 [configs/droid_episode0.json](configs/droid_episode0.json)：DROID / Franka Panda + Robotiq 2F85，第 0 个 episode 的 `[30,330)`，300 个有效位姿转移。配置是后续实验的输入约定，当前尚无可执行的数据抽取 CLI。迁移到其他机器时修改配置中的数据根路径；相对输出路径以本仓库根目录解析。

原始 R1/R2 数据只读，输出写入 `outputs/droid_episode0/`。必须区分 pose delta 与夹爪命令、实测物理时间与媒体时间、位姿重建与真实执行；原轨迹回放不能证明新命令的闭环效果。具体证据见实施计划的第 2、3、5.6 节。

开发约定见 [AGENTS.md](AGENTS.md)。
