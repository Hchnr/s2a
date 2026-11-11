# 真实 WAM step125 停止点的静态模型诊断

2026-11-11，按[预登记](../../../plans/v0.0.1/tracking/joint_control/cached_live_stop125_feasibility_20261111.md)，
只使用已独立重放的 step124 实测状态、上一条命令、已完成的内环误差历史和
step120 实际 WAM 生成的参考，冻结一个局部 MuJoCo 模型。已完成计划移位形成
搜索种子；原 step125 失败时的初始计划未保存，因此这不是对原搜索的精确复现。
恢复后的当前参考位置误差为 0.026969 m、旋转误差为 0.059394 rad，与原停止
记录一致；最近关节余量为 0.784269 rad。

| 局部模型搜索 | 独立 witness | 搜索响应 | 说明 |
| --- | --- | ---: | --- |
| 绝对质量门 + 过去误差 | 无 | 58 | 优化器报告成功终止，候选未通过 witness 过滤 |
| 绝对质量门 + 零误差（诊断） | 无 | 58 | 同上；零误差不是实际历史保证 |
| 关节约束 + 过去误差 | 有 | 76 | 独立计算通过关节、命令速率及制动核验 |
| 质量门 + 过去误差，以关节 witness 为初值 | 无 | 65 | 优化器报告成功终止，候选未通过 witness 过滤 |

关节候选在冻结模型的第一周期末位置误差为 **0.014189 m**、旋转误差为
**0.039391 rad**；位置超过绝对质量门 0.01 m，旋转低于 0.05 rad。独立核验
显示最小误差收紧后关节余量约 0.717748 rad、最小制动位置余量约 0.782314 rad；
与求解器内模型的关节位置最大差 `2.78e-17`。关节分支的 `first_pose_caps` 是按
种子构造的对照值，不是绝对质量门；上述是否满足绝对门以独立核验为准。

三个质量搜索的线性约束均有候选、优化器均报告成功，但最终没有通过更严格的
witness 过滤。已保存的结果没有记录被拒候选的逐约束残差，暂无法判断是数值
容差、其他约束还是搜索路径造成；不得据此宣称全局不可行。这个冻结模型不含
真实接触、时变动力学及物理闭环，不授予动作派发或任务成功资格。

实际一次模型初始化、一次不推进时间的 forward、四次优化和 257 次局部模型
响应；关节 witness 独立核验使用 200 个端点查询、2,600 个模型内转换步。没有
物理积分、新 WAM/GPU 调用或新初态；子进程正常退出并实际 wait/reap。运行前
32 并发专项 33 项通过、无跳过或收集错误，全仓 Ruff 检查通过。此前真实 WAM
仍是 45 个新反馈周期、末尾连续 38 个任务成功端点，不是完整 48 周期或 40
端点稳定资格。

| 仓库内证据 | SHA256 |
| --- | --- |
| [零工作 dry](../../../outputs/cached_live_stop125_feasibility_dry_20261111/result.json) | `4e904306048c02a321cdfd7c18fb0aeba9dc575d72e63ddfd34f933036992658` |
| [32 并发专项](../../../outputs/qa/cached_live_stop125_feasibility_20261111/result.json) | `3fcff7b724d12ca7ed23d6ea865fad708a741a9902071e2571e3a7b22ae1c8aa` |
| [运行登记](../../../outputs/cached_live_stop125_feasibility_20261111/registration.json) | `1521ff6e8485e9e7cb5a0ac23f7a2f3695cca31424dbbf8f5c0e3a3c8a236646` |
| [owner 结果](../../../outputs/cached_live_stop125_feasibility_20261111/result.json) | `dd608c99912e8a2804a309756ec5ef911fa77dc7670b767243ef09ba802ce454` |
| [分支与独立核验结果](../../../outputs/cached_live_stop125_feasibility_20261111/worker/result.json) | `33ca81bddb311de56a1d46f7d37eef564e217e880c149b996052af0aa0fabdb1` |
| [进程收尾](../../../outputs/cached_live_stop125_feasibility_20261111/workers.json) | `23ab7b3a8067d96c949d286993be6c2369d92e7e808f4d7f2f2080466d81f500` |
