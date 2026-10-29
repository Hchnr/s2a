# 迁移运行时的 CPU OSMesa 加载与有界 bootstrap

2026-10-29 执行前登记。此阶段独立补齐[文件快照](../../reports/native_runtime_path_snapshot_20261029.md)尚未实际加载的缺口。当前完整H12数值witness仍未获得，原3657响应独立复算在原句柄10388继续；本项不依赖其结果，也不授予物理pilot、轨迹或任务资格。GPU继续禁用，原18正确参考/12WAM开发/24最终留出门槛保持。

## 固定输入与允许差异

全部路径及SHA由[配置](../../configs/native_runtime_bootstrap_20261029.json)冻结。输入包括1090文件快照manifest、已完成独立v2文件核验及actual wait/reap0的execution、原快照输入配置、只替换3项路径的runtime manifest、原manifest、已登记cv2导入环境恢复合同与原controller.json。先重新核对完整快照、历史45份LIBERO/120份robosuite源码及当前依赖；不得读取旧`/share/project`路径，原声明仅作provenance保留。

任务固定libero_goal/task8、seed42、trial0、原79维initial_state及其SHA，原record XML、默认固定delta OSC_POSE、20Hz/2ms、camera_observations=False及restore后controller refresh均保持。只使用runtime manifest中的record，不读取HDF、expert动作、训练dataset、未来实际状态或新world-model参考。当前BDDL/default OSC JSON及系统库只有当前身份；即使成功，不能追认为完整历史二进制或行为等价。

冻结成功路径依据为`src/s2a/libero_simulation.py`（SHA `903a8616e26bc4ee5fb0c9575870ed0600e98f5a74229cbf5be9ff610ffe74c1`）。新增隔离入口复用其configure_simulator，按make_environment顺序进行构建、seed、reset、record XML恢复、initial state恢复和controller refresh，不修改已执行源码。唯一reset语义差异是外层`ControlEnv.reset`改为一次底层`env.env.reset()`：原包装器的while/finally:continue会重试并吞掉异常；本项成功路径仍调用同一个底层reset，首次失败立即归档，不自动重试。

所有Python配置、Numba/Matplotlib缓存、临时文件和新增产物在本仓库新输出目录内；快照、venv、外部只读来源不写入。不安装依赖，不改LD/RPATH或系统库。先按既有cv2合同恢复其已知导入LD变更；随后CUDA_VISIBLE_DEVICES与LD_LIBRARY_PATH保持空，五项线程变量为1，禁bytecode。OSMesa按快照绝对路径加载，MUJOCO_GL/PYOPENGL_PLATFORM显式为osmesa。

## 唯一执行与实际计数

新独占输出为`outputs/native_runtime_bootstrap_20261029`，独立owner子进程总限600秒，一次执行，无覆盖、自动重试或后续控制。constructor、一次显式hard reset、record XML恢复预期共3个模型；观察包装仅记录参数/结果，不增加forward，不改变控制运算。构建前保存实际XML及全部file引用，禁止旧路径、越出已登记asset根、未知include或symlink；所有实际读入资产逐项指纹并结束时重核。不同bootstrap模型与最终record模型分别记录，不能用最终96项资产代替全部bootstrap依赖。

预算明确区分attempted和completed：模型构建≤3，OSMesa context≤3，mj_step1≤2，mj_forward≤32，mj_resetData≤1，显式render≤1、read_pixels≤1。静态调用审计预计成功路径16次forward；实际次数以观察记录为准，不伪造静态数值。`mj_step1`是不积分的初始化运算，单独计数并检查time不推进，不能省略为“没有执行仿真”。禁止任何control env.step、mj_step或mj_step2，若尝试调用则在派发前记录并拒绝。无368命令前缀、S2A动作、参考消费、世界模型调用或fresh案例。

恢复后检查79维完整state及其字节SHA和原record一致，核对原控制器输入/输出范围、Kp/Kd、delta/fixed模式、无插值、控制频率、模型dt和TCP site。保存当前controller/state证据，不将恢复初态等同actual368。

仅在原初态显式渲染一帧agentview、256×256 RGB，用于验证OSMesa context和实际软件renderer，保持前后完整state/time相同。绑定实际加载OSMesa路径/SHA、OpenGL vendor/renderer/version、context类型及图像shape/dtype/指纹；不要求图像等于历史视频，也不把可渲染当作控制成功。实际模块文件和进程加载库保留当前指纹及身份边界。

## 准入、失败及结论

新core、CLI及两测试模块使用native Python3.11与合成依赖验证单次reset成功/异常、失败计数、预算在调用前拒绝、模型/资产路径、初态精确恢复、禁止控制step、渲染不推进状态，以及CLI QA/来源/独占/失败归档接线。复用快照与环境恢复相关测试，不重复无关数学搜索QA。正式QA绑定最终源/plan/config与日志；Ruff/format和实际CLI dry通过后才启动唯一运行时构建。

QA与owner均保存真实进程、actual wait/reap、命令、环境和退出码。失败保存已完成阶段、实际计数、未执行步骤、原始异常与可获取来源；已执行源和失败产物冻结，不把部分初始化标记为bootstrap成功。成功也须退出后重新核验原输入、新源、加载源码/资产与生命周期，才能记录runtime_loaded/bootstrap_verified；文件快照本身不追改。

本项只证明指定当前运行时能加载、恢复原初态并由软件OSMesa渲染。控制资格、任务资格、原H12witness及总目标均不由此成立。下一物理pilot仍须完整数值witness、全部独立复算及单独登记通过。
