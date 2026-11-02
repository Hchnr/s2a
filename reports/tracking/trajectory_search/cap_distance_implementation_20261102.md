# Cap-distance目标与共享驱动实现（2026-11-02，Asia/Shanghai）

## 实现与边界

新增`cap_distance_objective.py`，实现24组三维球外投影、完整径向/切向导数、
从rawJe构造Jr，以及B=I+Jr.T@Jr、g=Jr.T@r。预测下降显式使用无阻尼展开式，
pose的最小下降按同单位Lcap缩放；restore与原sum-q诊断字段保持原含义。
cap loss另存cap_loss字段，不用它覆盖state_loss或原caps判定。

新增`objective_search_driver.py`作为显式objective的共享状态机，保留原命令/
安全/证书/信赖域/原raw secant与反馈/预算/最后repeat机制。旧默认内核不修改。
新目标不会把投影后的零导数写回rawJe，因此当前球内组可以在后续重新激活。

本阶段仅实现和软件检查。尚未实现独立cap-distance ledger/fresh审核器，也没有
cap-distance研究运行、完整witness或物理资格；GPU继续禁用。技术门通过后再
冻结执行配置，不需要以软件测试代替研究证据。

## 显式算术补充

原设计直接sqrt差存在可复现浮点反例：原平移误差[.01,1e-10,0]对应
q=1.0000000000000002，caps失败，但float64 sqrt(q)-1舍入为0。固定使用实数
等价的有理化超限量(q-1)/(sqrt(q)+1)，且仅q>1时激活；不增加epsilon或新caps。
loss、投影系数及D的浮点求值在执行前固定，不能按结果切换。
这是对原直接表达式的明确数值修订，二者不宣称逐位相同。

补充计划为
`plans/v0.0.1/tracking/trajectory_search/cap_distance_arithmetic_20261102.md`。
原参考、尺度、控制接口、完整witness和物理验收门仍保持。

## 检查证据

58项专项使用统一CPU runner、jobs32，全部通过，无skip/collection错误，worker
实际回收、源码前后不变。覆盖球内/外/边界、完整径向/切向差分、浮点零集、
rawJe保留与未来泄漏拒绝、restore矩阵不变、无阻尼预测、近caps下降尺度、
不安全witness拒绝、有限预算/repeat及原dual-recovery算法/审核。

专项结果：`outputs/qa/cap_distance_objective_20261102/result.json`，SHA256
`09fb252003787528ffcb41f3d1a148faa2f91ee9806bcb58486f5e53146f8a1f`。
Ruff/format及Git空白检查通过；未运行全仓回归。

另外用已关闭sum-q基线的保存响应和QP返回值做纯软件等价回放：原内核和新驱动
sum-q模式各122次逻辑响应回调、55次逻辑QP回调。所有B/g/A/b、初始辅助变量及
solver选项同值，完整answer的序列化SHA也与闭合原基线完全相同：
`26c0fcd1286c08a503ca8899007416f5cefdab2e6fd0401a7da37402fa85a258`。

该回放没有调用minimize/NNLS、机器人模型或响应预测器。旧内核单次调用在隔离
命名空间中使用保存solver回调，未改动导入模块全局；新驱动显式接收qp_solver。
这是状态机保真检查，不是新搜索、额外研究响应或成功证据。
来源/代码SHA前后不变，输入旧bundle也不变；执行进程正常退出并已回收。

回放脚本及收据：`outputs/cap_distance_driver_implementation_20261102/replay_sumq.py`、
`sumq_equivalence.json`；收据SHA256
`c332e95f3e5da15c2a5a656f89849c6c2207cac311a86b805ed7966991be4de9`。
新增真实QP/NNLS/响应/查询/物理/WAM均0。

## 下一步

独立重建投影目标、QP约束/证书、raw因果更新和严格fresh决策，审核器不能调用
生产投影函数替代独立推导。随后固定唯一cap-distance执行配置，以已完成的新
sum-q为同预测器对照，保持60阶段/122响应及原安全和完整caps。结果仍需报告
前9/后3、全部q、原state error、完整成本和物理证据等级；没有完整witness就
不派发到实际控制，不以新loss下降宣布任务成功。
