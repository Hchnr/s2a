# S2A — State to Action

将世界模型给出的期望 state/delta_state 轨迹，结合当前实测状态与因果历史，转换为固定控制器可接收的 action。以同一前向控制系统后的实测状态误差、安全和任务稳定性评价，不能用 pose 重建或 action RMSE 代替控制成功。

[当前状态与待解决项](docs/status.md) · [结构设计](docs/architecture/repository_design.md) · [研究索引](docs/research_index.md) · [原始目标](plans/v0.0.1/goals/001_state_to_action.md) · [原验收门槛](plans/v0.0.1/goals/036_project_goal_20261023.md)

总目标尚未完成；原 18 条正确参考、12 条 WAM 开发和 24 条最终留出验收保持。最终稳定须至少 20/24 且各任务至少 4/8。局部数值 witness、测试、目录迁移或 dry 均不授予完整控制资格。GPU 在用户明确恢复前继续禁用。

## 代码与证据导航

| 位置 | 内容 |
| --- | --- |
| [src/s2a](src/s2a/) | state/command 合同、几何、因果参考、adapter、固定控制接口、前向响应和评价；研究实现单列 research |
| [scripts](scripts/README.md) | 按领域组织的入口及独立审核；统一 `scripts.*` 模块身份 |
| [tests](tests/README.md) | 单元、研究、集成与明确共享 fixture |
| [configs](configs/index.md) · [plans](plans/v0.0.1/index.md) · [reports](reports/index.md) | 原始配置、预登记和结果；正文及数值保持原字节 |
| [迁移记录](docs/migrations/README.md) | 历史声明、当前路径、SHA、测试发现及准入边界 |
| [旧导航原文](docs/history/index.md) | 基线 README 字节；历史链接按原声明位置解释 |

`outputs/` 保存生成产物、环境和原运行证据，本次不移动。外部 Cosmos、权重与发布数据只读，通过显式配置定位。

## 开发检查

Python 3.11+，所有命令从本仓库执行；仅向本仓库虚拟环境安装依赖，不修改共享环境。

```bash
CUDA_VISIBLE_DEVICES='' LD_LIBRARY_PATH='' python3 -m venv .venv
CUDA_VISIBLE_DEVICES='' LD_LIBRARY_PATH='' .venv/bin/python -m pip install -e '.[dev,report]'
.venv/bin/ruff check src tests scripts
.venv/bin/ruff format --check src tests scripts
CUDA_VISIBLE_DEVICES='' LD_LIBRARY_PATH='' .venv/bin/python -m pytest
```

软件/合成检查、保存产物核验、实际物理、模型响应和最终留出分别记录。具体输入、版本限制、已通过与未执行项见当前状态及对应报告。已有实验输出不可覆盖，迁移不自动重开任何任务。

CPU 回归默认用 32 个并发进程，最终全量检查按资源使用 64 个；每进程数值库单线程、空 CUDA/LD、独立 outputs 临时目录，逐子进程实际 wait/reap。D3c 安装的标准库入口为 `python -m scripts.maintenance.run_cpu_tests --jobs 32 --output outputs/qa/<新的独占目录>`，最终检查改 `--jobs 64`。普通 skip、xfail、collection 错误与实际通过数分别保存，不能把部分运行报告为全仓通过。
