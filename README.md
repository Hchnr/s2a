# S2A — State to Action

S2A 把世界模型给出的期望 state/delta_state 转成固定控制器可接收的 action，使执行后的实际状态接近目标。

```text
已生成的期望 delta chunk → 带时间的 SE(3) 参考轨迹
                                  ↓
当前实测状态 + 因果历史 → S2A → 固定控制器 → 机器人/环境 → 新实测状态
```

主要评价 `F_C(s_now, G_C(...))` 的状态跟踪误差；历史 action 误差仅作诊断。必须区分 measured state、desired state、observed delta 和 controller command。

已实现 DROID 单 episode 的只读抽取、几何转换、参考接口编码、局部响应辨识、在线有状态 PID adapter 和模型内扰动对照。当前实验使用真实未来状态形成 oracle 参考，未接入世界模型预测。完整设计见 [实施计划](plans/v0.0.1/001_state_to_action.md)，过程见 [实施记录](plans/v0.0.1/record_20261016.md)，本轮完成状态及限制见 [结论](plans/v0.0.1/conclusion_20261016.md)。

```text
src/s2a/geometry.py        column-rot6d、SE(3)、TCP/flange、SO(3) 误差
src/s2a/data.py            provenance join、物理时钟、masks、样本与指纹
src/s2a/interface.py       固定版本 DROID 参考接口与独立夹爪 position
src/s2a/trajectory.py      固定锚点参考、插值、chunk 衔接
src/s2a/control.py         P/PI/PD/PID、状态、滤波、限幅和 anti-windup
src/s2a/runtime.py         StateToActionAdapter（生成命令，无硬件 I/O）
src/s2a/identification.py  ridge 前向响应与正则逆
src/s2a/study.py           时序选择、日志留出评估
src/s2a/rollout.py         候选命令推进新状态的局部模型对照
src/s2a/reporting.py       Markdown、JSON 摘要及 Matplotlib 曲线
scripts/                  实验入口
configs/                  数据路径、划分、所有网格及扰动参数
plans/、reports/           Git 内维护的设计、记录及精简结果
outputs/                  数据子集、参数、完整指标及图（Git 忽略）
```

Python 3.11+，从本仓库执行：

```bash
LD_LIBRARY_PATH='' python3 -m venv .venv
LD_LIBRARY_PATH='' .venv/bin/python -m pip install -e '.[dev,report]'
.venv/bin/ruff check src tests scripts
.venv/bin/ruff format --check src tests scripts
LD_LIBRARY_PATH='' .venv/bin/python -m pytest
LD_LIBRARY_PATH='' .venv/bin/python -m build --no-isolation --wheel --outdir outputs/dist
```

当前 NGC 主机也可用 `/opt/venv/bin/python -m venv --system-site-packages .venv` 继承系统 runtime 包；只向本仓库 venv 安装开发依赖，不修改共享环境。实际运行版本保存于 manifest，版本快照见 [constraints](configs/constraints_20261016.txt)；它不是跨平台完整锁文件。基础依赖只有 NumPy/SciPy/PyArrow，绘图需 `report` extra。

首个样本由 [配置](configs/droid_episode0.json) 固定为 DROID / Franka Panda + Robotiq 2F85 episode 0，转移 `[30,330)`，保留 endpoint 和 3 帧历史。数据根和明确的 shard 路径均由配置传入；S2A 不依赖 Cosmos 训练 dataloader、工作目录或模型权重。

```bash
# 完整 P0–P3：只读数据、参数选择、模型内 rollout、报告与曲线
LD_LIBRARY_PATH='' .venv/bin/python scripts/run_experiment.py --config configs/droid_episode0.json --seed 42 --stage all --plots

# 可单独运行 extract / analytic / calibrate；报告重绘不重新拟合或打开 test
LD_LIBRARY_PATH='' .venv/bin/python scripts/run_experiment.py --seed 42 --stage report --plots

# 显式读取外部发布数据的集成测试
LD_LIBRARY_PATH='' .venv/bin/python -m pytest tests -m integration --data-config configs/droid_episode0.json
```

默认测试不读取外部数据，集成测试标为 deselected，不算通过。显式运行时，文件缺失或契约不符会失败。所有测试临时产物位于 `outputs/pytest/`。实验产物位于 `outputs/droid_episode0/`，包括 sample/manifest/audit、analytic commands、geometry/calibration/closed-loop metrics、adapter/PID 参数、完整 traces、report、figures 和 SHA256 清单。

fit `[30,210)`、validation `[220,260)`、test `[270,330)` 保持时序隔离；带历史/延迟的支持不得跨边界。所有标准化只在 fit 计算，模型及增益只在 validation 选择。每个候选、场景、评估区间单独初始化历史、延迟队列和 PID 状态，计分剔除 8 帧 warm-up；部署时跨 chunk 保留 PID 状态。

在线入口是 `s2a.runtime.StateToActionAdapter`：

```python
from s2a.control import PIDConfig
from s2a.runtime import StateToActionAdapter

adapter = StateToActionAdapter(H_FC, dt_command=planned_period, pid_config=PIDConfig())
adapter.set_chunk(measured_tcp, physical_delta_chunk, planned_knot_times)
result = adapter.update(current_measured_tcp, observation_time, now=decision_time)
command = result["command"]  # 6D arm + 绝对 gripper，gripper_action_space="position"
```

示例变量由调用方提供；`H_FC` 来自 tool contract。chunk 必须已反归一化、有有效 mask，时间轴必须是事先生成的执行参考；`now` 与观测时间使用同一物理时钟。`dt_command` 是计划时域，不读取未来观测才知道的 `transition_dt`。默认 PID 增益为零，实验搜索的参数不作为真机部署配置。

canonical TCP 的局部 delta 在 SE(3) 上右乘积分，再完整转换到 flange；DROID `cartesian_velocity` 是归一化接口量，不能当作 m/s、rad/s。夹爪 slot 9 是绝对 opening 命令，与 arm PID 分离。历史控制器版本和 command/state 时刻仍未确认，编码器属于固定公开版本的 reference-interface。

实验矩阵保留解析、线性、P/PI/PD/PID、增益调度、直接逆及残差模型。后三者需要验证同设备的多 episode 数据后进入 P4；独立仿真和真机属于 P4/P5，当前未执行。日志回放、局部模型自洽、独立仿真和真机证据分别记录。开发约定见 [AGENTS.md](AGENTS.md)。
