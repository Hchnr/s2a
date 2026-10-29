# 196–197 实测手指接触历史的因果性核验

新增finger_contact_history.py，三滞后×13维，共39维，包含当前相对过去的fingerq差、过去fingerqvel、wrench差、motor差及有效标记。显式episode重置，历史不足时mask0，禁止没有episode前缀的孤立anchor。3项需求测试通过，未来观测/额外label变化不影响既有prefix，跨episode不串历史。

outputs/measured_finger_history_20261025/verified.json与history.npz完整核验30轨迹6180行：独立标量索引重构1e-12一致，截断prefix逐bit一致，20Hz原时间合同通过；归一化仅用24fit4865行，不读holdout统计。无新响应fit、物理、WAM/fresh，也没有控制收益或状态充分性结论。

后续应把历史加入固定响应模型作单独对照，并继续原6holdout/6local门槛，不把可用性审查当局部资格。原完整18/18控制、WAM开发与最终24新初态目标保持。
