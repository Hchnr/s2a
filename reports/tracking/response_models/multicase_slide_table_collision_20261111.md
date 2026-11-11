# 三任务开发参考的桌面碰撞条件检查

2026-11-11。为检查[单初态 30 个端点的碰撞证书](slide_range_table_collision_20261111.md)
是否适用于更广的开发记录，对已审核 v4 三任务全部 12 条开发轨迹逐查询检查其已生成的
`effective_targets_tcp`。每任务只有一个开发初态，各自两种 seed 和两个变体；12 条
轨迹及其 7,200 个目标高度相关，并非 12 个新初态，也不是最终留出或当前 v5 评估。
每条轨迹有 75 次查询、每次检查前 8 个目标，共 600 个目标；全部纳入，没有按结果筛选。

检查仅针对各次查询时恢复的 finger2 真实碰撞网格、已核验的单滑动关节和该案例固定
仿真桌面。若某材料顶点在允许的全部 TCP 位置 `0.01 m`、旋转 `0.05 rad` 误差和
指定滑动关节范围下仍严格落在桌面盒内，给出充分碰撞证书。下表的“当前”固定为
**查询时**测量的关节值；“整行程”允许目标关节在额定 `[-0.04, 0] m` 的任何值。
额外形变取零，几何与该单轴模型、固定桌面和数值界成立均为条件。

| 开发案例 | 目标数 | 当前关节证书 | 整行程证书 | 当前关节超额定查询数 / 75 |
| --- | ---: | ---: | ---: | ---: |
| task0 / seed42 / raw | 600 | 1 | 0 | 41 |
| task0 / seed42 / terminal058 | 600 | 0 | 0 | 12 |
| task0 / seed43 / raw | 600 | 5 | 0 | 1 |
| task0 / seed43 / terminal058 | 600 | 0 | 0 | 0 |
| task1 / seed42 / raw | 600 | 0 | 0 | 20 |
| task1 / seed42 / terminal058 | 600 | 0 | 0 | 3 |
| task1 / seed43 / raw | 600 | 0 | 0 | 8 |
| task1 / seed43 / terminal058 | 600 | 3 | 0 | 9 |
| task2 / seed42 / raw | 600 | 0 | 0 | 2 |
| task2 / seed42 / terminal058 | 600 | 0 | 0 | 19 |
| task2 / seed43 / raw | 600 | 0 | 0 | 1 |
| task2 / seed43 / terminal058 | 600 | 0 | 0 | 9 |
| **合计** | **7,200** | **9** | **0** | **125 / 900** |

三个任务的当前关节证书分别为 6/2,400、3/2,400、0/2,400，整行程均为 0/2,400。
9 个肯定结果来自 4 次查询：task0 seed42 raw 的 step216 cursor7；task0 seed43
raw 的 step72 cursor7、step80 cursor4–7；task1 seed43 terminal058 的 step136
cursor5–7。前 6 个所用查询关节值在额定范围内，后 3 个共享的查询关节值
`-0.040030116186254225 m` 略超额定范围；“当前”结果仍针对该实测值的假设，
不声称它满足额定限制。125 次越界查询是查询时的 finger2 值，不能与原报告中的
机械臂七关节越界次数混同。

**0 个整行程证书不说明其余目标无碰撞或可执行。** 充分证书失效可能是因目标确实
可避免碰撞，也可能是当前保守界不够紧。尤其此前 correct_08 的 30 个相关端点中有
11 个整行程证书，与此处 v4 原轨迹的查询时刻、状态和已生成参考不同；两批结果不
构成同分布对照，也不能据此推断碰撞随版本改变的因果效果。结果不覆盖其他夹爪
几何、任务允许接触、连续路径、动态物体、真实标定或真机；原 12 条轨迹 0/12 任务
成功也不能归因于此证书。

## 复核与开销

逐案例独占进程加载只读模型和 96 个资产快照，恢复该查询已保存的状态并静态执行
`mj_forward`；每次只使用当时已生成的前 8 个参考，未将未来实测状态输入计算。
12 个 worker 均实际 wait/reap、退出码 0；共 12 模型、900 次 forward、14,400 次
证书计算、0 次物理积分、0 次 WAM 请求、0 个新初态。允许 free/ball 四元数仅发生
同姿态的标准归一化，最大 qpos 变化 `2.220446049250313e-16`；其他 qpos、qvel
及仿真时间不变。

独立审核重新检查 900 个原始状态哈希、7,200 个原始目标及游标和全部 worker 收据，
采用另一套球面角度界重新验证全部 9 个肯定证据；其严格内部最小裕量
`0.0006346855256430112 m`，超过 `1e-9 m` 数值余量。该审核复用已保存的几何，
并非再次从模型独立重建几何。未重跑全仓测试：此前仿真环境全量结果为 4 项失败、
2 项跳过、5 个文件收集错误；不能称全量通过。

独立审核第一次专项误用默认 `.venv`，由于没有安装 MuJoCo，有 2 个测试文件收集
错误；已保留失败结果，切换仓库内 `outputs/libero_env311` 后 32 并发专项 10 项
通过、零跳过及收集错误。原分析运行前 32 并发专项 23 项通过，dry 无模型或物理工作。
更新前 `docs/status.md` 的 SHA256 为
`8a7aaa1ffe0f80f4bfc8ae71c4394353353e65c2a3d9d2ed9d53373f484c884c`，字节见
[历史副本](../../../docs/history/experiments/status_before_multicase_slide_7d15c10d.md.txt)。

| 仓库内证据 | SHA256 |
| --- | --- |
| [原分析运行前 32 并发专项](../../../outputs/qa/multicase_slide_table_collision_20261111/result.json) | `a3fc3b2214607ac2e2104b4f9b9aac0997caf8c57aa5161ab71963aab98c2611` |
| [零模型 dry](../../../outputs/multicase_slide_table_collision_dry_20261111/result.json) | `58094033ab30bc2bb372c25c034c96ef1ba332f22d621fff6efa7f35c88a72ff` |
| [全部案例和目标](../../../outputs/multicase_slide_table_collision_20261111/result.json) | `12295d9c6feaa309ebcb6fa0fa6fb0968d31a9b85d9d7de8f46425de1c41ee13` |
| [12 worker 生命周期](../../../outputs/multicase_slide_table_collision_20261111/workers.json) | `d7c4940793c4925391e1fcd775d4966a92503c7760072b4702a665f6f9da20ed` |
| [默认环境失败专项](../../../outputs/qa/multicase_slide_table_collision_audit_20261111/result.json) | `7dc7e51574b2d84b6c134fa8959bbd67f65eb8c3ce7bf314d91d7547c87854d1` |
| [仿真环境复核 32 并发专项](../../../outputs/qa/multicase_slide_table_collision_audit_libero_20261111/result.json) | `fc8b96fb400daf4d61174aae04a5db8183c000a699a3cd05b821e4a7e608e7d2` |
| [独立来源及肯定证据复核](../../../outputs/multicase_slide_table_collision_audit_20261111/result.json) | `475ac5c35e1044273487237c3ddc0e515e2a776f563704ea15fbd0233d6fd96b` |
