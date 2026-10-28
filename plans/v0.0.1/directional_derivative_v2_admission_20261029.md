# 固定方向诊断入口修正（v2，执行前登记）

延续[原固定80请求计划](directional_derivative_consistency_20261029.md)的原始anchor1819、stages24–26、13步长、相同actual368初态、完整原H12与所有计账/独立重放/资格约束。总目标、原18/12/24任务门槛与GPU禁用保持。没有改变数学core、差分、旧Jacobian、原caps、安全buffer或控制器。

v1正式QA503项通过，但实际producer dry在模型创建前失败。失败准入SHA `56827f760b11d25f913d3993d7441daa7fc068f94b217c52d62f6cd6e48dfbda`；前两项Ruff/format exit0，producer dry exit1，后两项未执行；全部已启动子进程实际wait/reap，无信号或重试，新响应/查询/物理/模型/fresh均0。原因是入口错误复用了旧solver warm裁剪字节身份守卫，超出了本诊断登记的command `1e-12` 容差。原anchor首行两坐标裁剪会分别改变`1.1102230246251565e-16`和`7.216449660063518e-16`，最大相邻rate为`0.25000000000000067`；按原合同仍合法，不能裁剪原命令来通过另一个合同。

保留v1全部source、QA、失败准入及日志原字节。新增隔离v2 producer/checker/owner/evidence/测试，本次修正只以原四组box/rate有符号slack及其`1e-12`容差接纳固定anchor，命令原字节保持；所有预测前也使用同一原命令守卫。超原容差、非finite、shape错误仍拒绝。实际调用仍从同一个实际状态开始，完整独立checker保持原严格joint/pose标签，原历史prefix9/witness0不变。

新source闭包包括全部旧62项加本修正6项，共68源、21测试模块。新增回归覆盖浮点ULP落在原容差内时保持signed-zero和完整命令字节、超原容差拒绝、后续周期rate拒绝及v2入口/QA/生命周期绑定。全部QA、Ruff/format、三个实际dry通过后，才允许唯一v2诊断；新输出使用`tracking_v5_directional_derivative_v2[_check][_execution]_20261029`，不覆盖v1目录，不自动重试失败步骤。producer600秒/checker1800秒及80调用/79unique/24000内步/24961查询上限保持。

该修正不是放宽原计划门槛，也不将组件QA升级为数值或控制成功。任何真实响应后保持执行source不变；结果仍须完整独立核验和来源/生命周期闭合。若出现完整witness，另行登记物理pilot，不能直接消费新参考或宣称任务达成。
