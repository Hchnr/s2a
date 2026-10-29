# 连续开度与闭合保持候选：独立仿真资格 — 2026-10-21

预注册的九条 CPU/OSMesa 资格全部完成：三种执行方式各 3/3 在末 40 个端点保持原生任务成功，合计 5400 个物理端点，无执行错误。新 `position_hold` 可以进入冻结协议下的 WAM 对照。本报告只证明三个旧初态上正确参考的执行资格，不证明 WAM 任务能力、未见初态泛化或实际握持力。

## 固定协议与执行映射

[004 计划](../plans/v0.0.1/004_gripper_position_hold_20261021.md)在实现和新物理执行前固定预算与参数。[配置](../configs/gripper_position_hold_qualification_20261021.json)使用 bowl、wine、cream 各 init0，seed 42，20Hz，每条 600 步，H40、执行 40、最长 80 个物理步重新获取参考。三个初态来自已有 fixtures，未增加初态、重跑失败或事后调整阈值。

参考来自已保存的成功 native state/opening，按当前物理步选择未来段，经 Action128 接口解码。这是使用未来真实参考的 **oracle 正控**。夹爪模块本身只接收当前观测开度、已知电机目标、上一实际命令、dt 和已生成参考；不读取任务标签、对象位置、接触或成功谓词。原生任务成功和对象接触仅在执行后记为评价指标。

[新模块](../src/s2a/gripper_position_hold.py)与[测试](../tests/test_gripper_position_hold.py)在 `f968221` 提交；[资格入口](../scripts/qualify_gripper_position_hold.py)、[独立核验器](../scripts/verify_gripper_position_hold.py)及配置在 `163309b` 提交。模块与配置在物理执行前冻结，28 个新旧夹爪测试通过，定向 Ruff 检查和格式检查通过。测试覆盖高开度微降、量化抖动、受载开度、部分释放后不立即重锁、时间证据与实例隔离。

新映射在未锁存时，根据参考与已知 motor opening 的误差发送原生 −1/0/+1，deadband 为 .07。仅当参考 ≤.60 且 `.02` 滞环判定仍为关闭方向时，才锁存完整关闭；持续 +1 到 motor opening≤.005，之后以 0 保留电机位置目标。释放继承 v2 的全部条件：打开方向、超过当前/记载开度 .04、相对关闭参考低点回升 .04、完整 .05 秒观察确认，最大有效证据间隔 .1 秒；释放后恢复位置跟踪。没有固定 .70 释放下限。

`.60` 关闭门槛覆盖正确 wine/cream 参考最低约 .529；固定 .50 会漏掉这两条合法参考。`.07` deadband 略大于当前接口一个物理周期原生 aperture 变化约 .126582 的一半。这些是当前 Panda/20Hz 合同下的工程参数，不是任务分别设定的阈值，也未由后续 WAM 成功瞬间调整。电机位置限位与 0 保持均不是力命令或抓稳证明。

三组均使用原 OSC response 反解、inverse_weight=.5、command rate=5/s；只有第三组加入已冻结 recovering governor。oracle 与后续 WAM 共用 [实际物理执行器](../src/s2a/stability_position_simulation.py)和[矩阵入口](../scripts/run_wam_position_matrix.py)，避免正控使用另一套控制循环。

## 完整结果

预注册要求每条完整 600 步、末 40 连续原生成功、canonical TCP RMSE≤10mm、SO(3) RMSE≤.05rad。没有增加必须双指抓取的事后成功条件。

| 执行方式 | 场景 | 完整步数 | 末40稳定 | TCP RMSE mm | SO(3) RMSE rad | 尾连续成功步数 |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| baseline_v2，无 governor | bowl | 600 | 是 | 1.834 | .003347 | 531 |
| baseline_v2，无 governor | wine | 600 | 是 | 1.301 | .006744 | 454 |
| baseline_v2，无 governor | cream | 600 | 是 | 1.547 | .002596 | 501 |
| position_hold，无 governor | bowl | 600 | 是 | 1.838 | .003339 | 531 |
| position_hold，无 governor | wine | 600 | 是 | 1.421 | .006744 | 454 |
| position_hold，无 governor | cream | 600 | 是 | 1.548 | .002595 | 501 |
| position_hold，recovering governor | bowl | 600 | 是 | 1.951 | .002903 | 524 |
| position_hold，recovering governor | wine | 600 | 是 | 1.311 | .006788 | 451 |
| position_hold，recovering governor | cream | 600 | 是 | 2.056 | .003485 | 500 |

三组各 3/3 通过；TCP 实际范围 1.301–2.056mm，SO(3) .002595–.006788rad。全部参考 opening 范围 .052066–1，观测 opening .052393–.999895，motor opening 0–1。物理 dt 范围 .04999999999997229–.0500000000000167 秒。5208 个原生 0 命令间隔的 motor target 保持精确，包含满开与闭合保持，不能全部解释成持握。

## 基线复现与独立核验

候选启动前，`baseline_v2` 三条与旧 `recovery_executor_qualification/simulation/oracle_intent_hold` 对应结果比较。每条 600 个端点的全部保存字段、15 个 chunk（仅排除 wall-clock `generation_s`）、controller JSON 和模型 XML 精确一致。由此确认新共享入口的 v2 分支复现原物理行为；这不是新候选优于基线的证据。

[独立核验器](../scripts/verify_gripper_position_hold.py)不导入候选 adapter，而用独立 `GripperReplay` 重算 v2/v3 状态机和每条命令；用既有独立 `RecoveryReplay` 重算 governor。另核 Action128 从正确参考生成的来源与物理索引、原始/有效参考、查询前后物理时钟及状态 SHA、OSC 命令、电机积分与实际 actuator control、600 步预算、原生成功和 TCP/SO(3) 指标。数据与执行环境均保存指纹；本轮没有启动模型。

冻结核验器原先已检查测量位姿、电机目标连续性、25 个子步的原生 motor 积分和 0 保持，但没有独立重算 qpos→opening。资格完成后仅新增 [只读 supplement](../outputs/wam_stability_20261021/gripper_position_hold_qualification/qualification_qpos_opening_supplement.json)，没有改原核验器或重新执行物理。它从全部 5400 端点独立重算：

- 观测开度：`clip((qpos[0]-qpos[1]-.001)/.079,0,1)`，分别检查 before/after。
- 电机目标位置：`[.02,-.02]+.02*motor_target`，再用同一几何公式重算 before/after motor opening。
- 5391 个跨步连接的 qpos、opening、motor target、motor opening 连续性。

以上关系的最大绝对差均为 **0**。supplement 重核 434 项原始结果、冻结输入及审查副本 SHA；记录的三个 coordinator 与九个 worker PID 均已退出，扫描未发现本资格入口残留进程。所有物理 worker 使用空 `CUDA_VISIBLE_DEVICES`、空 `LD_LIBRARY_PATH` 和 OSMesa，最大并发两个；无 GPU 或真机执行。

## 证据指纹与结论边界

产物根为 [gripper_position_hold_qualification](../outputs/wam_stability_20261021/gripper_position_hold_qualification/)，包含逐步原始数据和 25 份审查源码副本。启动前 manifest 固定 224 项实际输入；最后补核没有修改其内容。

| 产物 | SHA256 |
| --- | --- |
| [启动 manifest](../outputs/wam_stability_20261021/gripper_position_hold_qualification/qualification_launch_manifest.json) | `bdc2749dc09ae3551099ccdcedd247206cafb211296685cade52cda399323d28` |
| [baseline exact](../outputs/wam_stability_20261021/gripper_position_hold_qualification/baseline_exact_repetition.json) | `ad883022d8596582bed67dceaff0b4eff4b864c88ee6429816453d9c6d24b307` |
| [独立资格 gate](../outputs/wam_stability_20261021/gripper_position_hold_qualification/qualification_independent_verification.json) | `3c416a11e85d8b2493803b7dabd7ff1b9c8fe6ccebf1dab198f42c41fbce888e` |
| [qpos/连续性 supplement](../outputs/wam_stability_20261021/gripper_position_hold_qualification/qualification_qpos_opening_supplement.json) | `ae53688698c328e2f46d69fa2103a5d2f0b1f6930aa038ede539f32ffc549bbd` |

配置 SHA 为 `7f9ea201a15954e250781954b2692fb4a3d2beffd1c775654e270a0c86589edf`；新模块为 `d16875a69220d36cd5d94dca20289b275d2eb003e4731eb705f762a435d705c7`；实际执行器为 `8e807d0fe296899dbda139e143227c23439b579c8de725a2363608f9bee3b136`；实际矩阵入口为 `0c444210cadb50e232d658d93d8061f920e961073f2b09cf2c74fbeecc6476f7`。

本轮完整通过预注册执行资格，同时保持旧基线的精确复现。它没有证明任意部分开度都可精确跟踪，没有测得握持力，也没有证明真实 WAM 参考能稳定完成任务。后者须由另外冻结的 WAM 闭环协议评价，不能把本报告九条 oracle 次数并入模型成功率。
