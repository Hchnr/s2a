# 终点局部排除结论依赖原算法搜索域

2026-11-02。按预登记，仅在保存的最终anchor23线性模型中，将root±.1算法域
替换为原控制器完整输入域。全部4369线性安全行、.25变化率及模型系数保留。
没有修改生产算法或执行任何新方向。

| floor | 原root域下界（历史） | 完整input域下界（新LP） |
| --- | ---: | ---: |
| 原buffer | 1.4519795340 | -6.5783629945 |
| reserve .9 | 1.4521963830 | -6.5781445496 |

两个新LP均通过原单位primal/dual/stationarity/gap审核和有限box残差扣除；
最大相对stationarity2.22e-16、gap0。原排除结论确实依赖算法域，不能当作固定
机器人的全域不可达证明。负支持下界也不证明球约束可行：方向可能从球旁边
经过，且其他23caps需要同一轨迹共同满足。LP点最大步1.57，严重超出原探针
尺度，结合前序大步非线性安全失败，不能将其作为可执行轨迹或直接扩大生产域。

## 对后续实现的约束

仓库已有完整input域、全24caps、每步重新差分的非线性SLSQP，不能把简单
恢复这些设置包装为新机制：旧4000响应/53迭代仍无完整witness，独立检查仅
完成500/4000，见[原报告](state_updated_trajectory_control_20261028.md:12)。
其安全阶段只在初始不安全时运行，旧起点安全所以未进入；当前qualified LP方向
则提供已独立确认、旋转更接近但安全失败的不同数值起点。

下一步采用明确的非线性最小距离可行性恢复试验：从该不安全起点重新计算完整
响应及安全Jacobian，恢复原buffer，而非只用原anchor的一阶切向方向或继续
放宽进展参数。先验证能否在保留轨迹改善的同时恢复安全，之后再登记连续搜索。
它不等于重跑旧4000响应，也不预设能解全部caps或证明任务收益。

## 验证、费用与来源

新增2LP，0响应/model/查询/内步/QP/NNLS/拟合/物理/GPU/WAM/fresh/最终留出/
实际参考消费。29项jobs32专项全部通过，无skip/collection错误，worker回收；
Ruff/format和Git空白检查通过。源码、输入SHA前后不变。CPU空CUDA/LD、
数值库线程1。tool session38973退出0并实际wait/reap，PID284193不存在，无重试。

以下路径相对outputs：

| 产物 | 路径 | SHA256 |
| --- | --- | --- |
| 完整结果 | input_domain_support_20261102/result.json | 1b42ac7584666840cdd82d769419e040c26211fade8e6d7b86ff1d7b5fd71e58 |
| buffer LP | input_domain_support_20261102/lp_0.0.json | 973ac79891dfe6dd679fb4106978cfea60e3ddaacb3e39c3d7fc4be0577b2bce |
| reserve LP | input_domain_support_20261102/lp_0.9.json | 704617b875b9bf8cd27dfe5d3c755d51ae84c5f143035e1b9e6f0ce002aceb6f |
| 29项QA | qa/input_domain_support_20261102/result.json | af4e26c312e6488703017ba8cde55b7d3bbc125b8da431d0d22538f3ceb48075 |

当前资格cursor8/task0/grasp0，原18/12/24和稳定验收未完成，GPU禁令保持。
