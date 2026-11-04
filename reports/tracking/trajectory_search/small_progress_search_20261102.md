# 小进展阈值对照：独立补齐通过，完整跟踪未改善

2026-11-02。按预登记仅将两个minimum reduction从1e-4改为1e-5；其余
实际安全、reserve .9、原24caps、全局域、预算及审核容差保持。
生产搜索已完成，但首次独立审核没有接入显式配置分支，原结果
`outputs/small_progress_search_20261102/small_progress/result.json`
SHA `844c4e05a24b41dda33e36aac180bb421afa73e63f88f8eb70b213615f6e4c5a`
继续保持complete/passed=False，未覆盖或追认为原进程正常完成。

## 独立审核补齐

修复提交561292f后，按独立补齐计划只读取保留的生产账本，新建一个native
model重建全部36响应，没有重跑搜索或新增QP。默认配置审核仍拒绝新阈值；
显式分支先审核原冻结配置，只允许两字段准确为1e-5，其余key/value/type保持。
源码差异清单核实控制器、响应模型和算法数值源码与原执行相同，只有登记的
审核接线、失败计数及补齐文件变化。

新结果complete/passed=True；全部36响应逐值exact，原1e-10/torque1e-8和布尔
判断保持。生产ledger与独立metric替换后的ledger均通过，initial/history/repeat
exact，来源前后不变。此为局部数值响应证据，未执行物理积分或下发候选。

## 科学结果

对照为已关闭的历史safety-feedback运行，不声称本轮重跑。

| 指标 | 历史安全反馈 | 新小进展组 |
| --- | ---: | ---: |
| 初始cap loss | 2.0100673083 | 2.0100673083 |
| 最终cap loss | 2.0088178441 | 2.0088853694 |
| 最终reserve | .903607 | .920588 |
| 达标caps / 连续prefix | 12/24 / 2 | 12/24 / 2 |
| 完整witness | 0 | 0 |
| stage / 接纳 / 安全反馈 | 12 / 6 / 2 | 12 / 7 / 1 |
| 生产响应 | 20 | 36 |
| 停止 | radius下限 | radius下限 |

小进展组多接纳一步，但完整达标数量和prefix均不变，最终cap loss甚至略高。
终点位置误差.0169000m、旋转误差.131082rad，仍超过原.01m/.05rad。
第三点旋转.0501403rad已使prefix停在2；中段旋转和后段位置/旋转均未解决。
不能用reserve稍高或接纳次数更多宣称可靠跟踪或任务收益。

按预登记停止继续增加标量进展阈值变体。下一阶段先审查完整轨迹瓶颈：
逐时刻误差、原安全/命令约束的可用方向、局部模型对这些方向的可信度，以及
固定参考在原接口下是否存在可执行轨迹。必须区分局部搜索停滞与不可达证明；
当前结果不构成不可达证书。任何新方法需先设计、登记预算并独立验证，
不能借本轮剩余预算续跑，也不提前消费最终留出。

## 成本、验证与收据

原生产36响应/10800内步/11233查询/1model，12QP/12NNLS；旧失败头的0QP/
NNLS为未到完成分支的计数缺口，实际费用以完整保留producer ledger为据。
新增独立36响应/10800内步/11233查询/1model，0新QP/NNLS。累计72响应/
21600内步/22466查询/2model，仍在原登记预算内。0物理/GPU/WAM/fresh/
最终留出/实际参考消费。原失败记录未改。

94项专项jobs32全部通过，无skip，worker实际回收，来源不变；Ruff/format及
Git空白检查通过。没有声称全仓pytest通过。CPU空CUDA/LD，数值库线程1。
tool session91629退出0并实际wait/reap，PID266045不存在；无自动重试。

以下路径相对于outputs：

| 产物 | 路径 | SHA256 |
| --- | --- | --- |
| 独立补齐 | small_progress_native_repair_20261102/result.json | 52a41511b116e958a3bb7ba43adbfaf1485eb351be1e88a5349e4a79eae0ac90 |
| 对照分析 | small_progress_native_repair_20261102/comparison_analysis.json | 07a5cc8db5760203a0ae0cab9e3a9f9c40f5d2d1b37fbc93594a5d4420ef6609 |
| 进程终态 | small_progress_native_repair_20261102/owner_observation.json | 3f7ca082e773f412bf3b89c6e9f0d15561aed9fd94edcf19f748540a6b6b947d |
| 94项QA | qa/small_progress_native_repair_20261102/result.json | 6381c266a5397e0b6496b87728454cd0429f17c3dceb41388db87aeceb8a7f20 |

总目标仍未达成：当前资格cursor8/task0/grasp0，原18正确参考/12 WAM开发/
24最终留出、稳定≥20/24且各任务≥4/8门槛保持。GPU禁令继续有效。
