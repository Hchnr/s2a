# AGENTS.md — S2A

先读 `README.md` 和 `plans/v0.0.1/001_state_to_action.md`。本仓库研究：世界模型输出期望 state/delta_state，经 S2A 反解 action，再通过固定控制器和机器人得到期望状态。

- 后续实现、测试、配置、报告及实验产物都在 `/share/project/eai_pwm/home/hcr/repos/s2a` 下维护；命令从本仓库执行。
- `src/s2a/` 放可复用算法，`scripts/` 放实验入口，`tests/` 放测试，`configs/` 放显式配置，`plans/` 与 `reports/` 保留在 Git 范围内，生成产物写 `outputs/`。
- Cosmos 仓库与 R1/R2 发布数据是外部只读参考。S2A 不引入 Cosmos 训练 dataloader、也不要求从 Cosmos 工作目录运行。数据路径通过配置传入。
- 主要评价命令经过前向控制系统后的 state error；不要把 pose 重建精确或 action RMSE 小等同于闭环控制成功。
- 解析、线性、P/PI/PD/PID、增益调度、直接逆模型和残差模型均在对照范围内，不预设胜者。保持相同样本划分和控制接口，分别记录已执行与未执行项。
- 明确 measured state、desired state、delta 和 controller command 的区别。固定坐标系、TCP、单位、时间间隔及有效 mask；旋转采用 SO(3)/SE(3) 约定。
- 在线输入只能包含当前可用状态、历史和已生成的参考轨迹。记录日志回放、局部模型 rollout、独立仿真和真机结果各自的证据等级。
- 实验显式使用 seed（初始 42），保存数据/代码指纹及参数；时序划分避免泄漏，PID 积分和滤波状态在独立评估间隔离。
- 用实际需求验证几何、控制状态和因果性；不编写仅为凑数量的占位测试。基础结构阶段尚无算法测试。
- 回答代码或数据问题时给出 `file:line` 依据；不确定项明确记录。

环境建立与基础检查见 `README.md`。安装开发环境后使用：

```bash
.venv/bin/ruff check src tests scripts
.venv/bin/ruff format --check src tests scripts
CUDA_VISIBLE_DEVICES='' LD_LIBRARY_PATH='' .venv/bin/python -m pytest
```

当前 NGC 环境执行 Python 时使用 `LD_LIBRARY_PATH=''` 前缀。本机已有 `/opt/venv/bin/python` 可用于验证，避免修改外部共享环境。

## GPU 使用限制（用户于2026-10-20明确指定）

后续实现和测试只允许使用物理 GPU 4、5、6、7，禁止在 GPU 0–3 启动任何任务。模型进程显式设置 CUDA_VISIBLE_DEVICES 为允许卡；CPU测试和OSMesa仿真设置为空。恢复/排队任务也必须遵守此限制，不沿用旧8卡映射。
