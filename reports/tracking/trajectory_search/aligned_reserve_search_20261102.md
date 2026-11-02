# reserve下限对齐的实际有限连续迭代

2026-11-02。预登记ef201a3、实现54343ac。全部新响应及两次独立ledger审核通过。
对齐后有更多局部接纳，但仍停滞且无完整witness，不能作为最终控制方案。

## 实际结果与明确的历史对照

同一anchor85、f4bc1804模型、原根域/pose ceiling与60stage/122response上限。
新aligned组实际执行；baseline与reserve列来自此前已关闭匹配对照，没有重跑。

| 指标 | 历史baseline | 历史reserve | 新aligned |
| --- | ---: | ---: | ---: |
| 初始cap loss | 2.0100673083 | 2.0100673083 | 2.0100673083 |
| 最终cap loss | 2.0076879552 | 2.0098115464 | 2.0091872116 |
| 最终sum q | 30.00497285 | 30.10544356 | 30.03698312 |
| caps / prefix | 12/24 / 2 | 12/24 / 2 | 12/24 / 2 |
| 完整witness | 0 | 0 | 0 |
| 最终reserve | .557286 | .928585 | .911682 |
| stage / 接纳 | 60 / 49 | 7 / 1 | 9 / 4 |
| restore stage | 24 | 0 | 0 |
| 生产 / 独立响应 | 91 / 91 | 28 / 28 | 22 / 22 |
| 停止 | stage预算耗尽 | radius下限 | radius下限 |

新组最终joint/buffer安全，min raw slack9.12681561e-6、row4332，末点旋转误差
.1310849741rad仍高于原.05rad。接纳次数及下降优于只加接纳门的旧reserve组，
但少于预算上限就停滞，完整loss仍不及旧baseline；单案例不能证明泛化或优胜。

## 停止机制：门槛对齐后仍有模型误差

新组9个stage均为pose，4个接纳、5个全拒绝，拒绝反馈更新0。
实际原因计数：accepted_reduction4、actual_reduction_insufficient5、
pose_reserve_below_entry11、predicted_reduction_insufficient11。

末个接纳后anchor reserve=.9116815612。stage5 alpha .25预测=.9087611709，
实际=.8901088035；stage6相同步长预测=.9087611709、实际=.8978810725。
stage8 alpha .5预测=.9058407806、实际=.8710254028。
这与旧半reserve QP已经预测低于.9的结构不一致不同：现在局部模型预测可保留
余量，但完整响应损失了更多余量，原接纳门正确拒绝，没有降低门来继续。

完整步的预测有约1e-9量级原QP数值偏差（约.899999999..），按原证书审核；它
远小于上述actual差距，不能用调epsilon消除真实模型偏差。更小步还会受原最小
预测下降门限制。数据支持下一步核查临界行误差与安全拒绝校正，不证明加入
任何反馈必然有效，也不证明原目标不可达。

## 实际计费与验证

新组1个独立预算，实际9QP/218迭代/9NNLS、22生产+22独立响应，共44响应/
13200内步/13730查询、2model。未使用的stage/响应不转移或自动恢复。
0新stencil拟合；4个accepted causal secant更新照原ledger记录，rejected update0。
0物理/GPU/WAM/fresh/最终留出/实际参考消费，所有候选未下发。

先独立重建生产ledger/GN/KKT/scaling/recovery，再逐响应完整native复算，原
1e-10/torque1e-8与布尔audit保持。全部response及decision metric逐值exact，
替换为独立metric后再次审核全部决策、反馈与更新；历史anchor/repeat exact，
源码/输入前后SHA不变。两次审核保留在result，全部native响应独立保存。

共享有限协议仅显式增加路径与策略参数，旧默认未全局替换；新entry只提供登记
参数。旧baseline的60个已保存QP、91响应完整软件回放答案SHA一致：
e3f6690e754d8595720e024ef1c5709cb28d8dcfddecf75d9270c34b7c17728b，
独立ledger也通过。该回放没有新增QP/NNLS/响应，旧数据比较不冒充本轮执行。

50项jobs32专项全部通过，0skip/collection错误，worker回收、源码前后不变；
Ruff/format与Git空白通过，未运行全仓回归。CPU空CUDA/LD、数值库线程1。
tool session18234正常退出0并实际wait/reap，PID226268确认不存在；1800秒TERM/
10秒KILL及phase alarm未触发，无重试。

| 产物 | outputs相对路径 | SHA256 |
| --- | --- | --- |
| 完整新结果 | aligned_reserve_search_20261102/aligned/result.json | 68a177b2acee1aed82bf527d85503f47b8376f0398a34fd1d76376983fb916cd |
| 新旧比较分析 | aligned_reserve_search_20261102/comparison_analysis.json | 1ee87f3e1810422ba07adce8d30ea3f2e48c9f3b3cd01a7cedcc1e7a1358eee4 |
| owner终态 | aligned_reserve_search_owner_20261102/observation.json | 295e97697fc9ca353ae20bae3e43199544a2172472ee27dd76095f19d021ce3d |
| 50项QA | qa/aligned_reserve_search_20261102/result.json | 46071ab113d16ba28211c27e2618fe9c4d0db016afaad1954aa6edc1b1fb6baa |
| 旧baseline完整回放 | aligned_reserve_search_implementation_20261102/baseline_replay.json | 3f7f9e1eba2362a1c12a7e0265049089e31a666fc62f65ac71207fbc2c7ce9ab |

下一步先以本轮已计费安全拒绝检查临界模型误差及校正效果，明确后验、未见数据
与新执行区别，不能直接挪用旧半reserve shadow结论。任何新反馈策略、约束裕量
或响应预算另登记并独立审核。原可靠跟踪和18/12/24稳定验收仍未完成。
