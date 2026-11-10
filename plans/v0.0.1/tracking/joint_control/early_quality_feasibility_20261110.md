# 两条安全闭环的质量约束可行性

2026-11-10。[原细化器实际闭环](../../../reports/tracking/joint_control/early_refiner_feedback_20261110.md)在两案例共16周期均保持真实内环0.05 rad缓冲，但每条后四周期旋转误差超标。日志中这八个超标周期的原模型也报告首点精度比较失败，而被选中的 `interior_witness` 不强制该精度。本阶段在全部16个已完成分支状态上，先检验已有绝对首点质量约束求解器，未派发任何候选动作。

## 固定数据与候选

绑定原producer `outputs/early_refiner_feedback_20261110/producer/result.json` SHA `2e900f506f7bda364348822e2891d3c1be4191bb71703590cc8704a4300c669b` 和独立动作重放checker `outputs/early_refiner_feedback_20261110/checker/result.json` SHA `7062266a8a6eff29ff43fb339ddb578769150a4d35e279e6c2e5f496022f1404`。固定两条refiner分支所有8个决策前状态与该时刻实际已生成的原chunk；状态0取重放前缀终点，状态k>0取同分支已完成的第k周期state，上一动作和已完成历史也只取同分支过去。该周期原refiner已经生成的8点计划可作为同时刻的数值seed，不使用该周期后的真实state、动作或后续WAM输出作为候选输入。

每个状态按已验证的record XML与资产建立独立MuJoCo模型，以保存的q/qvel和对象状态做一次无时间推进的`mj_forward`恢复机器人M/J/site。模型、地址、时间及机器人q/qvel核对；记录其他qpos可能的浮点归一化差。使用该分支当前运行时记录的controller.initial_joint，不把它当作已保存的历史设备隐藏状态。先将原已选计划在同一冻结自由模型内rollout，比较首周期预测TCP误差/25个q与此前真实执行误差/内环q；比较使用真实后果仅作事后评分。

对每个状态固定两组 `quality_capped_inner_joint_plan.solve_interior_plan`：

1. `nominal`：原0.05 rad模型内环安全、box/rate、终端及现有0.1 s制动约束，强制首点位置≤0.01 m、SO(3)≤0.05 rad，不加入历史误差收紧。
2. `past_error`：相同全部限制，另以**此前已完成**周期的真实内环q与对应冻结模型预测q差的逐关节绝对误差，沿当前周期内时间取累计最大，再跨过去周期取最大，作为原求解器的非递减inner误差收紧。起点之前没有已保存的25内环预测—实测配对，因此初始profile明确为零，不冒称覆盖此前或未来误差。offset0的两组输入完全相同，只求解一次并标记复用。

每组保留原每次最多160优化响应、60迭代及原1e-10安全复核；无witness明确记录失败，不能放宽精度或将局部优化失败称为全局不可行。现有0.1 s制动是额外模型限制，不能将其导致的无解推广为原安全+跟踪问题无解；之前固定0.1 s制动的实际失败继续保留。对每个返回的witness，用逐周期独立构造的 `build_joint_response/predict_joint_response` 重算每2 ms关节状态、末端关节速度、制动/终端/box/rate及首点TCP误差；核对模型数值不代表实际控制安全。

## 预算与准入

16个CPU worker按案例×offset各使用独立模型及输出；每worker依次恢复最多 `offset+1` 个已保存状态，最多72次`mj_forward`及72次基线模型rollout，0模型物理积分、0软件/GPU渲染context。最多30次实际优化（offset0两组复用）、4800优化响应；逐witness独立关节复算最多6000个端点查询/78000次模型内转换步，并另列实际成本。owner实际wait/reap，线程库单线程、空CUDA/LD、不自动重试，输入/源码/资产SHA前后一致，0 WAM/GPU/fresh。先测试历史误差包络因果性、无未来记录、误差收紧及独立重算；32并发专项和零模型dry后冻结源码。

只能报告这16个相关开发状态的局部模型解及预测误差；若未找到witness不宣称机器人全局不可达，若找到也不得称实际达标。后续必须另登记相同起点的实际闭环、已完成历史在线更新和独立动作重放；原18正确参考、12 WAM开发、24最终留出及精度/关节标准均不变。

首次CPU QA在来源快照阶段失败，结果 `outputs/qa/early_quality_feasibility_20261110/result.json` SHA `a3d56cfe72ba6e5063150ef769771c26631851f0655132aa9c67921d09e74197`；因仅修改历史求解器docstring导致迁移清单SHA漂移，尚未收集测试或启动模型。已恢复该文件原字节并在新QA目录继续，原失败保留。历史docstring声称不强制首点精度，与代码中的绝对质量不等式及现有测试不一致；本文以代码和测试为准，不修改原迁移身份。

首次16 worker实际执行 `outputs/early_quality_feasibility_20261110/result.json` SHA `67ea65d74ac89f0cdaa220ae2c2c0407ff58fada5451138d8ab0b98f5b278d5f` 失败：每个worker在导入MuJoCo时找不到已指定的OSMesa符号，均在模型构建前退出；16个均实际wait/reap，新增模型/forward/优化/物理均为0，原结果不覆盖。独立CPU导入试验先按现有仿真运行时以绝对路径和RTLD_GLOBAL加载已核SHA的仓库内OSMesa，再导入MuJoCo成功，未建模型或context。现另行登记v2输出和v3 QA，固定原16状态、预算和所有门槛，加入独立干净子进程加载测试；这不是自动重试。
