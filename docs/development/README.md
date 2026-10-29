# 开发与验证

所有命令从仓库根目录执行。环境、实验缓存和临时文件放在 `outputs/`；
外部 Cosmos 与发布数据只读，数据路径由配置显式传入。

## 安装

Python 3.11+。新环境只在仓库内安装，不修改共享解释器：

```bash
CUDA_VISIBLE_DEVICES='' LD_LIBRARY_PATH='' python3 -m venv .venv
CUDA_VISIBLE_DEVICES='' LD_LIBRARY_PATH='' .venv/bin/python -m pip install -e '.[dev,report]'
```

基础依赖是 NumPy、SciPy、PyArrow。MuJoCo/robosuite、模型服务和外部数据属于
独立可选运行环境，安装基础包不会建立这些环境。运行版本应随实验 manifest 保存。

## CPU 检查

```bash
.venv/bin/ruff check src tests scripts
.venv/bin/ruff format --check src tests scripts
CUDA_VISIBLE_DEVICES='' LD_LIBRARY_PATH='' .venv/bin/python -m pytest
```

当前主机已有的数值回归环境为 `outputs/libero_env311/bin/python`。使用它时显式
设置空 CUDA/LD、单线程及独占 basetemp，例如：

```bash
CUDA_VISIBLE_DEVICES='' LD_LIBRARY_PATH='' PYTHONDONTWRITEBYTECODE=1 \
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
NUMEXPR_NUM_THREADS=1 LP_NUM_THREADS=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
outputs/libero_env311/bin/python -m pytest tests/unit/infrastructure \
  --basetemp=outputs/pytest_infrastructure_local
```

每轮改用新的 basetemp，不覆盖已有验证产物。该数值环境的 NumPy 1.26.4 与
当前 PyArrow 存在既有导入冲突；涉及 PyArrow 的 collection 错误单列，不能据专项
通过声称全仓测试通过。默认跳过 `integration`，外部数据或物理仿真验证须显式配置。

并发回归使用仓库内的标准库入口，不需要额外安装pytest插件：

```bash
CUDA_VISIBLE_DEVICES='' LD_LIBRARY_PATH='' PYTHONDONTWRITEBYTECODE=1 \
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
NUMEXPR_NUM_THREADS=1 LP_NUM_THREADS=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
PYTHONPATH=src:. outputs/libero_env311/bin/python -m \
  scripts.maintenance.run_cpu_tests --jobs 32 \
  --output outputs/qa/cpu_local_run001
```

每轮更换新的目录名；目录已存在会拒绝覆盖。默认收集原pytest
scope，再按完整node ID分片；每个子进程有独立临时目录和结果，全部实际wait/reap。
最终全量可用`--jobs 64`，数值库仍各单线程。结果单列实际pass、skip、xfail、
collection错误及来源前后是否变化，不把跳过或收集失败计作通过。可用
`--expected-nodeids`及`--expected-nodeids-sha`核精确节点/参数顺序，用`--nodeids`
及对应SHA选择明确子集；这些输入须保存在仓库outputs内。

该3.11环境另有4项原streaming严格数值测试失败，已在迁移前原字节中同样复现；
开发3.12环境的相同26项在原容差下通过。两个环境的Python/NumPy/OpenBLAS均
不同，不能归因于单个组件；不通过换环境追认旧运行资格或称单一环境全仓通过。
当前证据及未执行项见[状态](../status.md)与[迁移记录](../migrations/README.md)。

GPU 当前禁用。原始配置中的历史设备编号只是记录，不能用于启动或恢复任务；
限制与用户恢复后的 UUID 约束见 [AGENTS.md](../../AGENTS.md)。

## 修改与研究记录

- 可复用代码、实验入口、测试和研究证据分层维护，职责见[总体设计](../architecture/repository_design.md)。
- 测试验证几何、控制状态、因果性或真实失败边界；独立审核保持独立的数学实现。
- 使用 seed 42 起步，保存数据、源码、配置与环境指纹；每个评估间隔隔离 PID 和滤波状态。
- 已完成与失败证据不覆盖，历史路径/SHA 与当前执行源码分开登记，见[迁移说明](../migrations/README.md)。
- 完成可验证的小阶段及时提交；提交说明和[当前状态](../status.md)如实记录未执行项。

目录迁移不授予新控制、runtime、世界模型或留出集资格。
