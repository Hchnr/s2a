# 因果夹爪候选与真实仿真资格 — 2026-10-21

第一版没有通过资格，完整保留失败；第二版在三条正确oracle参考上通过预注册资格，可以进入后续WAM受控试验。**这不是WAM任务成功、稳定泛化或实际握持力的证明。** 本报告的12次新执行均为CPU/OSMesa、CUDA为空，未运行真机或新GPU模型，旧报告和原27条诊断保持原字节。

## 固定协议

旧六初态中的bowl、wine、cream各init0，seed42；两种夹爪各独立恢复环境，600步/20Hz。参考来自原native稳定成功的真实state/opening，经原H40/执行8的Action128 oracle接口；OSC参数保持inverse_weight=.5、command_rate=5/s。每次执行前后核源、代码与配置SHA，adapter不读取对象/contact、任务标签、arm位置或未来真实state。

两协议各6次、合计12次新物理执行/7200步，只有3个旧物理场景，原intent基线重复执行。历史源参考只读；这些次数不并入任何WAM成功率。初始current_action同旧oracle入口，由initial observed opening对齐为shape2，不冒称历史expert内部电机目标。

预注册资格要求六条均最后40端点稳定成功，且canonical TCP RMSE≤10mm、SO(3) RMSE≤.05rad。接触/位移只作指标，不事后增加必须抓取的成功条件。

## 第一版：滤波和观测开度目标出现回归

[模块](../src/s2a/gripper_tracking.py)、[配置](../configs/gripper_tracking_qualification_20261021.json)和[执行入口](../scripts/qualify_gripper_tracking.py)在3e0f5aa提交后冻结。21个新单测加原4个gripper测试25 passed；Ruff检查与格式通过。首次物理前修复了motor_deadband抵消preload的代码缺陷，参数数值未根据任务结果调整。

| 任务 | 旧intent稳定 | v1稳定 | v1 TCP RMSE mm | v1开度MAE |
| --- | ---: | ---: | ---: | ---: |
| bowl | 1/1 | 1/1 | 2.245 | .01826 |
| wine | 1/1 | 1/1 | 1.250 | .02992 |
| cream | 1/1 | 0/1 | 1.547 | .03364 |

cream基线2.50秒开始close并继续把motor推至0；v1从2.55秒开始，夹有0保持，motor停在约.4698。两者arm RMSE仅差约.45μm，baseline有41个双指接触端点并稳定成功，v1无接触且从未成功。滤波延迟、目标和压紧偏置同时改变，未测握持力，不能从该对照唯一分离三者。保存的完整时间顺序见[失败时序](../outputs/wam_stability_20261021/gripper_v1_failure_sequence.json)。v1 gate=false，未作为合格方案升级。

[v1原始汇总](../outputs/wam_stability_20261021/gripper_qualification/summary.json) SHA `0be91361f6aab08612178b60ab01ab7017c5a87f2d0ab04cebbabdff2ce5400c`；[独立审计](../outputs/wam_stability_20261021/gripper_qualification/independent_raw_audit_20261020T181222Z.json) SHA `bf683b85fd68674bae1b9bd815fb3fd183286b4d61f5963ad03505a28405d43b`。83项输入SHA复核、3对初态精确，1759个native0前后motor目标完全不变；实际执行器ctrl仿射核验误差为0。

## 第二版：保持原关闭起点，闭合电机到限位后才0

独立[新模块](../src/s2a/gripper_intent_hold.py)、[新配置](../configs/gripper_intent_hold_qualification_20261021.json)、[新测试](../tests/test_gripper_intent_hold.py)和[新入口](../scripts/qualify_gripper_intent_hold.py)在531cc1d提交后冻结；未修改v1字节。10个新测试加原4个gripper测试14 passed，Ruff与格式通过。

[原生关闭与保持](../src/s2a/gripper_intent_hold.py#L104)沿用OpeningIntent .02滞环的关闭起点，不给关闭参考再加EMA；持续+1直到motor opening≤.005，之后0保留闭合目标。observed aperture不再当自由电机目标。-1/+1是原生开/闭方向，0保持已存在位置目标，均不是力命令。

[释放资格](../src/s2a/gripper_intent_hold.py#L111)同时要求旧intent为open、参考超过当前/holding观测宽度+.04、超过闭合参考低点.04，并连续满足一个完整.05秒观察间隔；近全开有可达端点处理。margin/excursion来自历史滞环的两倍、确认时间为一个20Hz周期；没有用WAM成功标签选参数。断档或实际open使相应时间/宽度证据失效；电机到限位或宽度停滞都不等于抓稳。

| 任务 | 旧intent稳定 | v2稳定 | v2 TCP RMSE mm | v2 SO(3) RMSE rad | v2开度MAE | v2接触端点 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| bowl | 1/1 | 1/1 | 1.834 | .003347 | .00870 | 26 |
| wine | 1/1 | 1/1 | 1.301 | .006744 | .00951 | 66 |
| cream | 1/1 | 1/1 | 1.547 | .002596 | .00962 | 42 |

六条完整3600步，gate=true。候选尾连续成功分别531/454/501步。每条14个open、12个close、574个0；574个0包括满开保持和闭合保持，不全是持握。闭合motor hold分别30/59/35步，共124步，原始motor opening均实值0；不声称由力传感器认证保压。原intent与候选在这些专家参考上均只有3次非零方向反转，不能据此宣称WAM参考抖动已解决。

独立重算6条预算、20Hz、native成功、canonical误差和源成功参考；3对完整初始physics/pose/qpos/motor一致，3个intent逐值复现v1基线。候选600步normal_intent与旧intent一致，关闭episode起点精确为bowl[0,32]、wine[0,74]、cream[0,50]。1722个native0前后motor目标精确不变，全部3600步actualctrl公式一致。native每20Hz非饱和步的motor aperture最大变化约.126582，而非.253。

证据：[v2汇总](../outputs/wam_stability_20261021/gripper_qualification_v2/summary.json) SHA `1977d0b07bf90542e972ca668ad78a2e61794c27af0bb318f776acf01257cf50`；[独立raw审计](../outputs/wam_stability_20261021/gripper_qualification_v2/independent_raw_audit_20261020T181729Z.json) SHA `1c467320c6dfa4e42d5fc8f6c8d668dd90a481cc89062b7ea0aefcc3505d49ad`。87项输入SHA首尾不变，源码AST确认无object/contact/success输入。v2模块SHA `b6dba5c603a3f4856311b6bdffa9b0bad0bf958871e8b69e3930ea8ffbf99692`，v1仍为 `2de7e76762d2af114af95da4a52ecbf7bf1b4765a9d3f8dc3e0f7bfb13b6f901`。

## 接入与范围

同一instance跨chunk保留，独立trial重建；更新只给当前measured opening、当前motor目标、上一实际发送command、已生成opening及过去dt。`panda_opening_from_motor_target(gripper.current_action)`只读现有shape2电机目标；`GripperIntentHold.update(...).command`是原生三值。

本轮只认证三条旧oracle参考的执行资格；任意partial aperture、未见场景、真实抓力与WAM闭环尚未由此认证。v2改变了夹爪策略，后续需要冻结模型、arm和协议的WAM真实闭环对照；不能把oracle成功、参考反事实或学到的补偿当原checkpoint正确的证明。
