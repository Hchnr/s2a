# 真实WAM clean桥接资格 — 2026-10-23

**037的完整模型采集与039补充独立审计联合通过，`bridge_qualified=True`。** 完成一次新模型初始化、7个完整请求、245次denoise；没有新拟合、物理任务、fresh或真机结果。总目标尚未达到，下一阶段为[038有限训练和开发协议](../plans/v0.0.1/038_clean_residual_training_20261023.md)的实现与准备检查。

## 实际完成范围

固定8400非EMA、SDPA、H40/20Hz、35步、shift5、guidance1、active10，模型仅用允许UUID`GPU-ad74a609-3a39-43cb-1392-24cbd279ce7f`。外部Cosmos/权重/环境只读；实际执行隔离源码位于本仓库037输出，解释器为已登记Python3.13.14。163权重文件核stat，三份小metadata重新SHA；没有新全量大权重内容哈希。[037登记](../plans/v0.0.1/037_clean_bridge_continuation_20261023.md)

| 检查 | 独立结果 |
| --- | --- |
| 三份raw历史回归 | 三任务各一当前图像/任务输入，完整40×128原始和normalized输出全部逐值相同 |
| 三组raw/zero | 35节点在线张量、初噪声/条件及最终完整输出全部精确一致 |
| 固定非零head | active10常数clean bias `1e-5`，真实sampler返回的102个有效值改变，覆盖29/35节点 |
| 逐节点数学 | 245节点的实际FP32 sample、actual sigma、raw C0、残差/velocity转换、原dtype舍入及真实scheduler接收均通过 |
| 因果与mask | 当前vision首latent帧有条件，其余future vision无条件；有效10 action槽无条件，无效118槽固定零；未读取未来标签 |
| 完整预算 | 7请求/245 denoise/245 head，1次新模型初始化；0fit/0物理/0fresh |
| 模型与资源 | 原参数metadata identity前后一致；模型worker actual wait0/reap、无信号退出，之后GPU记录0MiB、compute apps空 |

039独立checker完整重算后，额外stdlib/NumPy peer再次读取252个NPZ、4431个数值数组，完整检查245节点；2423项实际消费源码/产物SHA前后不变。它确认同样的102值/29节点及完整联合生命周期；没有调用生产数学实现或重复模型。

## 原失败及修正边界

旧035只留下初始化日志、无结果/退出收据，记录PID已不存在；旧运行的退出原因和完整计算消耗未知。原目录与SHA未改，037使用独立输出和新增有限预算。[旧运行审计](../outputs/clean_bridge_continuation_20261023/old_run_audit.json)

037模型worker已完整成功，但其原独立checker错误要求整个40×128 action condition mask全0，因此原owner为False。已登记的新原生Cosmos在无效118槽先将reference清零，再将flat mask置1以固定缺失坐标；有效10槽仍为0。这不是未来action条件，也不是将mask方向反转。[冻结源码:2747](../outputs/wam_stability_20261021/clean_bridge_qualification_20261023/source/cosmos/cosmos_framework/model/generator/omni_mot_model.py#L2747)、[原检查:430](../scripts/verify_wam_clean_bridge.py#L430)

[039补充](../plans/v0.0.1/039_native_mask_audit_addendum_20261023.md)在训练或任务评分前登记，仅CPU复用已保存数据；新增严格的完整mask重建、当前latent绑定和无效零值检查，其余原独立数学/预算/来源检查保留。原037 owner False和错误日志不回写，新True只属于补充审计与原完整模型证据的联合资格。

039首次准备漏建preparation父目录，未启动checker；5份源码现场已完整归档至同级`clean_bridge_native_mask_audit_20261023_preparation_failed/`。修复后新CPU审计child正常wait0/reap。额外peer首启又因自身将Markdown引用按JSON读取而在0节点停止，False及旧脚本保留；修正入口后的完整成功结果另存。成功peer随后仅增显式来源别名，增补前内容保留于`final_peer_completed_before_ref_aliases.json`，最终引用以下表为准。这些准备/peer实现错误均不计作新模型失败或任务失败。

## 正式证据

以下根均在`outputs/wam_stability_20261021/`：

| 收据 | SHA256 |
| --- | --- |
| `clean_bridge_qualification_20261023/preparation/prepared.json` | `88dfed133bcbf073dd81b90f7ebf89c6829d1a191c3bce1891f7341488753c09` |
| `clean_bridge_qualification_20261023/qualification/result.json` | `e35e68f83fbe83ccbf6a1c13e09f7c16b413cfdde88380b5885fed8bc7cca4e5` |
| `clean_bridge_qualification_20261023/orchestration/result.json`（原False） | `44f5473d98909e16c627aee16da2ba1a97aafc03eacd192ac03579b8f716180f` |
| `clean_bridge_native_mask_audit_20261023/review/verified.json` | `1e7597ed4ee416d5155ab628ad2113b38a4c991fbd07657ab96f7f992603c66f` |
| `clean_bridge_native_mask_audit_20261023/orchestration/result.json` | `247f9c541dd4bcfa8c6bf7a5b00633a2427fc0ce90664875180cf4a254ffa788` |
| `clean_bridge_native_mask_audit_20261023/peer/final_peer_completed.json` | `cc78cf1aa825f2a893bf201c8cdaf3f67e3a29cc02aee8ff3fe7915bfabc8fe7` |

实际模型PID1212273/start566625580、037失败checker PID1227922和039成功checker PID1237788均保留其各自真实wait/reap记录。037外层工具session22375 exit1、039首次准备session37726 exit1、新039外层session39994 exit0，分别属于不同阶段，不能混记为所有入口均成功。模型日志末尾的NCCL退出提示保留；实际child exit0及后续显存释放已独立核验，不将提示隐藏为无告警运行。

## 软件与后续

新runner按SHA绑定独立config/plan，相关六模块CPU测试110 passed，413份当时Python源码Ruff通过；原生mask wrapper另有31项拒错/来源/输出路由测试通过。精简sink已有29项CPU测试，并从037第一raw保存的35点完成离线接入：1400行、C0独立误差0，所有消费SHA前后不变；这是复用已有采集的CPU验证，0新denoise。[mask QA](../outputs/clean_bridge_native_mask_qa_20261023/qa.json)、[精简接入](../outputs/wam_stability_20261021/clean_bridge_qualification_20261023/aux/compact_replay_probe/probe.json)

资格不等于训练或任务收益。038固定30条/150anchor、每anchor两训练noise，raw300→W1→W1轨迹300→全量union W2；仍须完成数据/源码准备、采样与fit/独立QR、完整生成诊断及成对开发任务接入。旧六validation的native初态同步未证实，继续用完整demo保守排除并明记未知，不伪造SHA认证。最终12新初态×seed42/43的24次性能验收未运行，稳定≥20/24、各任务≥4/8门槛不变。
