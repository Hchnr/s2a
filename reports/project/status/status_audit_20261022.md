# S2A 实现、目标与下一步核查（2026-10-22）

本次实际工作区为`/share-mixed/project/eai_pwm/home/hcr/repos/s2a`。用户最新指定只允许当前可见GPU2/3，按AGENTS中的两个UUID执行；旧目录与旧数字卡号仅保留为历史声明，不沿用启动任务。

## 原来的总目标及当前判断

原始目标是让世界模型预测期望state/delta_state，由S2A结合当前实测状态和因果历史反解action，经同一固定控制器与机器人执行后接近期望状态。判断依据是实际前向控制后的state error和任务结果，不是pose重建精确、action RMSE低或测试数量。依据：[001原始设计](../plans/v0.0.1/001_state_to_action.md)、[在线adapter](../src/s2a/runtime.py)、[轨迹接口](../src/s2a/trajectory.py)。

后续将任务验收具体化为：真实WAM仅以当前图像/任务生成参考，经固定S2A，在预留12个新初态×seed42/43的24条运行中稳定成功至少20/24，各任务至少4/8；稳定指末40个20Hz端点持续满足原生任务谓词。等待模型时暂停物理时钟，当前不考核实时推理。依据：[002稳定成功计划](../plans/v0.0.1/002_wam_stability_20261021.md)。

**完整总目标尚未达到。** 正确参考经过S2A的独立仿真正控已经建立，但真实WAM没有达到上述多任务稳定门槛，预留最终24条性能验收仍未打开，真机执行为0。几何/控制资格、数据资格、模型适配资格与任务成功是不同证据。

## 已经做到了什么

| 层次 | 已完成的工作与结果 | 当前边界 |
| --- | --- | --- |
| 表示与数据合同 | SO(3)/SE(3)、column-rot6d、TCP外参、physical time、mask、命令与测量区分；DROID和跨数据集几何验证 | 几何精确不等于控制精确 |
| S2A算法与软件 | 解析接口、线性响应/逆、P/PI/PD/PID、有状态adapter、限幅/anti-windup、时序参考和异步接入 | DROID局部模型上的反馈方案未通过全部扰动验收，未给真机增益 |
| 独立物理仿真 | LIBERO/MuJoCo实际执行与原生命令/oracle正控、夹爪/旋转/重规划资格及完整日志审计 | 2026-10-20统一24条正控native和oracle均24/24稳定；仅适用于已验证仿真合同 |
| 真实WAM闭环 | 多checkpoint、guidance、步数、执行窗口、重观测和夹爪/旋转对照 | 统一独立矩阵8400为0/24，8800为1/24稳定；后续旧开发场景raw8的seed42为1/3、seed43为0/3，未建立稳定重复性 |
| 模型侧诊断/适配 | 训练—推理数值桥接、teacher readout、真实完整生成和自由轨迹诊断 | 027 teacher曾改善57.45%，但028生成门槛失败；029第二noise seed退化51.42%，自由轨迹clean误差增加1.60%，不能推广 |
| 新同步训练数据 | 原030固定30条native、150当前图像、4980物理步完整采集审计；032已完成末端pose×motor对照 | 原030稳定28/30、全量训练资格False；motor保持未救回demo35，不作为默认方案 |
| 当前后续实现 | 新clean残差CPU原语、真实sampler桥接和流式ridge原语；034固定三十条收尾协议 | CPU合成测试不代替真实模型资格、真实拟合或任务结果 |

主要结果依据：[早期结论](../plans/v0.0.1/conclusion_20261016.md)、[独立24条报告](wam_independent_20261020.md)、[raw8第二seed报告](wam_raw8_seed43_20261021.md)、[029完整诊断](wam_free_trajectory_diagnostic_20261021.md)、[032末端motor对照](native_motor_hold_20261022.md)。原先“唯一模型仍运行”等文字是按时间保留的历史记录；029已完整结束，以其正式报告为准。

## 下一步的实际顺序

1. 完成034全30条新协议采集与独立审核。保留全部native前缀，末端锁存实测controller SITE pose并继续末gripper命令；40步settling后再检查40端点，五个anchor中最后一个移到N+40。新预算30环境/6180步/150当前canvas。原030的28/30 False保持不变。[034登记](../plans/v0.0.1/034_settled_native_training_20261022.md)
2. 完成035真实WAM桥接资格。固定解释器与隔离Cosmos源码，仅执行3 raw＋3 zero＋1合成非零请求，核全35节点FP32 sample/actual sigma、zero精确复现、真实返回velocity改变及正常GPU释放。0拟合/物理/fresh。[035登记](../plans/v0.0.1/035_clean_bridge_qualification_20261022.md)
3. 两项通过后，另登记有限真实模型训练：冻结8400、显式C0 clean残差、训练noise排除42/43、原始及固定两轮on-policy上下文、固定ridge与完整数据权重。使用全量流式fit，不能根据validation反复改阈值或选成功demo。
4. 统一候选先做完整生成state诊断和有限开发任务实测，再冻结到原预留24条最终验收。只有实际达到20/24和各任务4/8才能报告稳定目标达成。

不建议此刻继续扩大PID/夹爪/采样网格。已有正确参考正控和后缀毫米级跟踪却任务失败的结果，支持优先改善生成参考及训练上下文；它们不足以证明所有失败都只由WAM造成。新clean readout属于模型适配，潜在收益不能追认为原checkpoint或纯S2A转换已解决。

## 环境及工程现状

旧canonical目录被外部删除，关键50组产物已安全恢复到本工作区，历史JSON/path/SHA保留，经只读resolver定位，不改写历史证据。模拟器已在本工作区实际恢复运行，032的完整native前缀逐位回归通过。[迁移报告](workspace_relocation_20261022.md)

模型预检只读检查了8400的163个文件stat、三份小metadata SHA及实际依赖导入。Python3.13.14模型解释器可用；没有重新全量哈希大权重。外部share-mixed Cosmos在预检期间被更新，后续使用本工作区内固定源码副本。[预检证据](../outputs/model_runtime_preflight_20261022/README.md)

完整迁移QA已有826 passed、1 skipped、2 deselected，但还保留1 failed/12 errors，原因是旧测试中的canonical绝对路径或外部源缺失；不能写成全仓全绿。新阶段另执行相应模块测试与Ruff，精确结果随阶段收据保存。原git pull中的untracked覆盖保护与实验结果无关，本次未删除工作文件、强制pull或改写分支。

034已完整结束：[正式报告](settled_native_training_20261022.md)。新协议30/30稳定、每任务10/10，6180步/150当前canvas及完整native前缀/缓存/标签/图像独立核验通过。首个owner的历史task0封套检查False保留，经事前登记只读补充完成全部审计，不重复物理；这只获得后续训练协议登记资格，模型训练与fresh仍未授权。新增支持子集回归950 passed/1 skipped/3 deselected；补充56项定向检查通过，旧完整QA失败边界不变。

035仍在完成真实运行前的实现和独立检查，尚未初始化新模型或取得真实桥接资格。
