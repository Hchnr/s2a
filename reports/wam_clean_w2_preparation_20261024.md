# 047：W2双轮拟合接续准备

2026-10-24 10:22（北京时间）：**W2入口与准入实现已完成准备，142项相关CPU检查通过；实际W2拟合、QR和执行准入均未发生。** 同一D1模型仍存活，已闭合66/300请求，对应已完成请求至少2310次denoise；进行中请求的精确总调用数未断言。三层PID/birth匹配，原519源/快照全部不变。D1完整收尾仍未取得，W2不能使用部分数据。

本项推进[036总目标](../plans/v0.0.1/036_project_goal_20261023.md)，协议见[047](../plans/v0.0.1/047_clean_training_w2_runtime_fit_20261024.md)。最终仍为同一候选在原预留24条中稳定≥20/24、每任务≥4/8；尚无新增任务或留出成绩。

## 实现与固定范围

[元数据准入:65](../src/s2a/clean_training_w2_admission.py#L65)要求046 supervisor、wrapper的真实owner及worker、原D1 owner、300请求/10500节点完整collection和独立capture审查都合格；正常wait0/reap、PID birth、模型和审查的不同launch及实际解释器/源码、后代释放和GPU释放缺一不可。不能将partial ledger、活体进程或True字段单独当作完成证据。

[047完整准入:193](../src/s2a/clean_training_w2_admission.py#L193)另绑定原D0、045唯一W1、原519闭包、固定policy、最终真实配置及独立metadata envelope。[实际运行时:302](../src/s2a/clean_training_w2_admission.py#L302)逐次检查Python3.11.15/NumPy1.26.4/SciPy1.15.3，以及解释器和两个数值扩展文件SHA；生产/独立worker均只能使用实际合格sim解释器，CPU空CUDA/LD、三线程8。

[双轮worker:231](../scripts/execute_wam_clean_training_w2_fit.py#L231)只改变W2阶段、D0+D1两组collection/audit引用、840000行及新的输出/来源字段，仍调用原生产与独立数学函数，不改源码、ridge=.1、1290特征或参数比较rtol1e-9/atol1e-11。独立阶段绑定同一完整fit/params，不允许重试。[串行owner:141](../scripts/run_wam_clean_training_w2_fit.py#L141)保留真实Popen producer→wait/reap→independent的监督逻辑，两角色各7200秒，0新模型/denoise/physics/fresh。

新入口由原043固定SHA模板作有限变换，AST逐项匹配；阶段参数及数据/审查引用实际传给原函数，未复制数学求解函数。W2数据仍按原038固定两轮全局权重，D1特征用raw C0；clean协议本项最多新生产尝试1、QR1，累计生产尝试最多3（043失败W1、045成功W1、本项W2），不会重跑W1。

[不可执行policy](../configs/wam_clean_training_w2_policy_20261024.json)已固定，但未填造任何未来D1 SHA。准备脚本`runtime_w2_fit_20261024/preparation/bind.py`先要求真实完整046收尾链，再绑定实际D1结果引用、最终配置和扩展source/snapshot；目前没有最终配置、proposal、admitted或真实`fit/W2`。最终源码封存及实际admitted dry必须等完整D1资格，不把准备记录当执行许可。

## 已执行检查

- 新增接口/准入/真实CPU小子进程测试，以及原W2双轮流、raw C0、全局权重、streaming、readout、QR、source-proof相关回归：7模块142通过。真实小进程证明串行wait/reap及producer失败后不启动独立阶段；它们使用合成metadata，不是实际W2拟合。
- 原数学回归确实运行了合成求解；“0实际fit”特指0真实训练数据W2 fit，不将合成solver冒充训练，也不称全部检查没有数学调用。487份主仓库源码及3份准备脚本Ruff/format通过。
- 实际sim解释器的runtime组件检查通过，指纹与045一致；它只证明运行时检查组件，不是完整W2准入或dry。045原sim合成4项严格NumPy参考比较失败及Decimal诊断仍保留，未改数值容差，未以main环境测试代替未来真实sim拟合/独立QR。
- 对真实运行中的D1调用W2资格函数，明确拒绝“无完整supervisor结果”；没有导入NumPy、torch、MuJoCo或数学producer/checker。实际绑定CLI也在发布最终配置前拒绝，exit非零及原日志保留，0真实W2尝试。

## 证据与下一步

证据均位于原prepared.root下`runtime_w2_fit_20261024/`：

| 证据 | 路径 / SHA256 |
| --- | --- |
| 142项组合检查 | `qa/pytest_combined_v2.log` |
| 准备QA与被测源SHA | `qa/preflight_qa.json` / `6bc25825ef2ee6cbdd9d13b146c3c5e89635675cb23e0069cc5eac3bfe651dc8` |
| 实际sim运行时组件 | `preparation/runtime_component.json` / `f8d95a9a06fa0f993cfb20a3047fc495293691128a938e93a5dbad805db3656d` |
| 真实live D1被拒绝 | `preparation/pending_gate_probe.json` / `5b9d320c29bdfe415a6d56333a30e9b9bd6bde4238106a49f45c4473726b2bcf` |
| 实际绑定CLI拒绝及无发布 | `preparation/bind_pending_refusal.json` / `1aad336bc0070c4de525c055430be83ab0c9be7b38f60383b1ae9fede1ebfc16` |
| 10:22 D1活体与519源核验 | `preparation/live_d1_after_qa.json` / `95eca2018d97b5cf89582ff6f6dadb2bb93632158d576877a3ddacb5973397ce` |

D1仍是supervisor PID2010452/birth574805981→owner2010453/birth574806125→model2010513/birth574806584，同一模型只使用允许UUID（当前可见2/minor6）。本轮未重启模型，也没有新增GPU任务。[046报告](wam_clean_d1_continuation_20261024.md)保留前2请求/70节点的原独立CPU重算资格，不能扩称整轮已通过。

下一步先完成D1及其完整原独立审查、真实wait/reap/GPU释放，才执行047绑定→独立metadata审查→新admitted→实际两role dry→一次固定W2/独立QR。之后仍须新W2参数的完整生成诊断、固定开发任务门、统一候选及最终留出验收；当前不开放fresh、不宣称世界模型任务收益。
