# 迁移 CPU 运行时的保存命令回归

2026-10-29：实现已提交 `bee872a`。隔离QA、四条实际准入和唯一实际运行时重放均已正常关闭：**377控制周期/9425积分步、378端点/225内环样本全部核对，state/q/qvel最大差异均0，actual368完整state逐字节一致。** 当前保存分支的行为回归通过；完整H12控制、任务及历史二进制等价仍未取得。首次执行在QA命令格式检查失败、0运行时/0物理的记录完整保留，五份实现来源始终不变。

本阶段依据[登记计划](../plans/v0.0.1/native_runtime_replay_20261029.md)和[固定配置](../configs/native_runtime_replay_20261029.json)，只重放已执行并独立核验的 H8 分支。它不重新规划、不产生新世界模型参考，不推进原 cursor9，也不取得完整 H12 控制、任务或历史二进制等价资格。GPU 继续禁用。

## 首轮 QA 与 metadata 准入

正式 native Python3.11.15、空 CUDA/LD、单线程及禁 bytecode 环境中，**27 passed，5.82秒**；原会话36291已实际exit0。QA子进程PID1212885实际wait/reap0，无signal或自动retry，5份新来源和1个测试模块保持不变。测试包含完整保存命令调度、内步中途失败计费、严格实际caps、控制合同变化、失败cleanup禁控制，以及archive异常不覆盖原派发异常。

实际准入原会话57859已exit0，三个子进程PID1217082/1217178/1217271分别完成Ruff、format和CLI metadata-only检查，均实际wait/reap0。metadata检查绑定2808份来源、实际物理步骤0，runtime_loaded/control_qualified均False。

| 证据 | SHA256 |
| --- | --- |
| [正式QA](../outputs/native_runtime_replay_qa_20261029/result.json) | `7550e8e135b28a34aac616f0b36efd4bc4468af2b5c2bf2ceb0fcea0b7fa9197` |
| [QA日志](../outputs/native_runtime_replay_qa_20261029/00.log) | `1186764ad12176f37a0a8c2c21ef959a364d12a41ef0ed37976184094f104581` |
| [实际准入](../outputs/native_runtime_replay_admission_20261029/result.json) | `16cde188f3b315688ebc8e3f0649e1f3e8489e48f9d777857313e05b70614846` |
| [准入metadata日志](../outputs/native_runtime_replay_admission_20261029/02.log) | `e2793354ebb61f956b8a910c28283b482ef8ee167185a52a795337c1068e16e8` |
| [首轮执行driver](../outputs/cpu_control_continuation_driver_v2_20261029.py) | `aac8bd7ecf1b23df2cff575246200c9cf3df18765fc095a8688867a7f829d2f9` |

## 固定分支与原预算

原H8分支已有377条命令、378端点，368..376的9个周期共225个内环q/qvel/time样本；producer与独立checker的完整state archive SHA一致。本次从原79维初态自然重放全部368条前缀，恢复控制器、夹爪和接触历史，再发送原保存的后9条命令，不直接恢复actual368。

唯一物理重放上限为**377控制周期、9425个2ms积分步**，含全部前缀；最多3模型/3contexts，禁止mj_step2。实际派发和积分的attempted/completed分别记账，失败立即停后续派发。原初态及controller全字段保持，actual368完整state逐字节一致及q/v逐元素精确一致单列；核对原M/J、旋转goal和initial_joint，另记录motor cache与completed367命令，用于定位迁移后的状态合同。

历史数值回归使用原atol1e-10/rtol0；实际位置.01m、SO(3) .05rad及内环关节裕量.05rad仍严格检查，不用比较容差放宽实际门槛。保留原current_observation/endpoint刷新及相机调度，软件OSMesa渲染；未来保存状态仅用于结果比较，不恢复仿真或选择新命令。

## 保留的首次执行准入失败

原会话55725已实际exit1，child PID1219262实际wait/reap1。QA receipt的命令使用分离的 `--basetemp` 与路径参数，而冻结CLI的执行QA gate只接受 `--basetemp=...` 尾项，因此报`Replay component test command`。metadata-only准入没有调用执行QA gate，先前准入成功不能覆盖此失败。

[失败owner](../outputs/native_runtime_replay_producer_20261029/result.json) SHA为`96ceab95ba806f51e8abb3887540f51d4fe2302ffdead8cc8f118fde1fa55b48`。失败发生在创建实际输出root及加载运行时之前，**运行时初始化0、控制派发0、积分0**；不存在部分成功重放或新的物理结果。原QA、准入和失败均保留。

后续使用独立`native_runtime_replay_v2` QA/owner，driver以单一`--basetemp=...`参数运行同一冻结组件测试，并在实际准入中增加真正的`verify_qa`调用。两次QA的五份来源SHA清单完全相同。失败时实际登记输出root尚未创建，因此首次真正运行时沿用原登记位置；没有覆盖既有物理产物，没有自动重试。

## 隔离准入与唯一实际重放

新QA原会话98514已实际exit0，child PID1227344实际wait/reap0，**27 passed，5.87秒**。新准入原会话96103已exit0，四个child PID1231281/1231374/1231467/1231891分别完成Ruff、format、metadata检查及实际`verify_qa`绑定；均wait/reap0，最后一项明确返回`QA receipt admitted`。这补齐了首轮metadata-only准入漏检的执行参数合同。

实际重放原会话11523已exit0，child PID1234541、parent PID1234427，owner实际wait/reap0，无signal或自动retry，登记timeout1800秒。CLI和core的complete/passed均True，runtime_loaded及behavioural_replay_verified均True；cleanup_attempted/complete均True、forbidden_control_attempts为0。CLI明确不自称掌握actual wait，退出事实由独立owner保存。

| 证据 | SHA256 |
| --- | --- |
| [隔离QA](../outputs/native_runtime_replay_v2_qa_20261029/result.json) | `16c672bfda5a29cf003b07e0c4885d5d76ad702d5d3e883262b30a76581f4652` |
| [隔离QA日志](../outputs/native_runtime_replay_v2_qa_20261029/00.log) | `b644b8a84c248ee4ed2b28957cec22a9fe6f597ab00283c795ac428a1875a32b` |
| [四条实际准入](../outputs/native_runtime_replay_v2_admission_20261029/result.json) | `fc370713d7c5fb7e2d9166b23e78aaae132c5ddaa2f4d706c6d0ef3ff0a01611` |
| [实际重放结果](../outputs/native_runtime_replay_20261029/result.json) | `3436345322baabe2b0473cfaf491a45060f7a7e49d95c721981ab7b10b2dc5f4` |
| [实际owner](../outputs/native_runtime_replay_v2_producer_20261029/result.json) | `aedd662e30743223bfb0ea30310a14003f11853786d966489a1ae9ca71ab6901` |
| [实际运行日志](../outputs/native_runtime_replay_v2_producer_20261029/00.log) | `855531039da7c9a7a7419c5f9fc85594ce79ab92569f2992447c70f296525f36` |
| [隔离driver](../outputs/cpu_control_continuation_driver_v3_20261029.py) | `1b0c8b112e3e4b2da8f20975bb37eb50571c005cb53a532caad98e9a9b20a710` |

实际成本均完整计入：configure 1/1、模型3/3、context 3/3、control_step **377/377**、mj_step **9425/9425**、mj_step2 **0/0**（均attempted/completed）。累计仿真时间推进18.850000000000293秒，最终端点time=19.100000000000293；这是物理时钟推进量，不是墙钟实时资格。首轮执行参数失败为0物理，两次执行入口合计仍377周期/9425积分步，不能将前缀或失败历史遗漏。

378个端点和225个新段内环样本均核对完成，最后检查endpoint377。完整state、inner q及qvel最大绝对差异均0；time与历史profile通过原容差检查。actual368完整state字节一致、q/v元素精确一致；原initial_joint精确一致、完整固定controller合同一致，当前M/J和旋转goal与原period0证据匹配。motor cache为`[1,-1]`，completed367命令及controller goal/cache保存在结果中；这些是历史状态回归证据，不是对未来未知动作的保证。

原cursor0..8的9个实际周期全部满足严格caps：最大TCP位置误差0.00417953568m、最大SO(3)误差0.04570233721rad，最小内环关节裕量0.08123685949rad；task/grasp均False。仍只到cursor8，未发送cursor9命令，没有新的任务推进或fresh案例。

实际renderer为Mesa `llvmpipe (LLVM 20.1.2, 256 bits)`，OSMesa context含`mujoco.osmesa.GLContext`，OpenGL为`4.5 (Compatibility Profile) Mesa 25.1.7-1ubuntu2~24.04.2`。原三阶段模型XML SHA与v3一致；登记OSMesa库实际mapped身份通过。保存153个加载库、357个运行时模块来源，均仅声明当前实际身份。2808份已绑定来源在运行前后保持；本报告只查看小receipt/log及已存9周期指标，没有重新执行模型或增加全来源核验阶段。

成功也仅能证明此保存分支在登记容差内复现；未知动作、其他初态及历史未绑定二进制身份仍无此证据。新的H12物理候选仍等待完整模型witness及独立核验；原18/12/24任务门槛、实际cursor8、task/grasp0和总目标保持。
