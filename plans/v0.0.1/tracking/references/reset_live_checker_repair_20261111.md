# 复位双窗口独立 checker 的端点收据格式修复

2026-11-11。按[原计划](reset_live_second_window_20261111.md)，实际 producer 已从
case8 原复位完成 16 周期，并在本次 step8 当前图像上实际请求一次 WAM、完成 35 次
去噪；模型与仿真进程 exit0、wait/reap、GPU 释放。producer SHA256 为
`a4634066e9a4ed8c7b93eeda89501b8f66b0454b0fe854c5976cf46b2c104d0f`。

原 checker 真实执行首 8 周期/200 内步后，在比较 step8 图像时失败并清理，worker
exit1、wait/reap、无信号；checker owner SHA256 为
`bbdcb22927763f9a7cb6bc460155fe70adc073cc7c3da1ed17017e9fad32c93a`，
partial branch SHA256 为 `554e49aa1cdd870b8b2e51f8cddfe23dae54885f3db55d51ad54c7df4527a850`。
只读比对显示实际 step、状态 SHA、物理时钟、双相机画布 RGB 和 PNG 哈希均一致；
失败原因是流式 producer 传给 renderer 的 endpoint 仅含 step/time/SHA 三字段，
checker 传入完整 endpoint（另含 dtype、shape、native success、motor 哈希），
原核验错误地要求两个字典字段集合完全相同。原失败不重新标为通过。

## 新预算和不变门

新增独立 checker 入口，保留现有渲染和比较实现，只在调用 renderer 前从 checker
现场完整 endpoint 提取 producer 同格式的 step/time/SHA；三个值必须原样来自本次
实测端点，不能使用 producer 的值填充。完整 endpoint 仍用于自身状态核验；图像
RGB/PNG SHA、状态 SHA、时间、相机角色、source pose、opening 和物理重放容差不变。
新增测试覆盖带额外字段的真实收据、状态或图像不一致拒绝。原有源码及失败产物
不覆盖；新的 QA、dry、worker 和 owner 各使用新路径，不自动重试。

新 checker 从同一 reset 最多重放实际已发送的 16 个周期/400 内步，使用 3 模型、
3 OSMesa context；不加载候选策略或模型、不请求新 WAM、不调用 GPU、不训练、不用
新初态。必须记录和核对原失败 8 周期/200 内步与新增实际开销。原 producer + 原失败
checker + 新 checker 的本试验累计上限 **40 周期/1,000 内步、9 模型/context、
1 次 WAM 请求/35 denoise**。完整重放后独立只读核查并分别保留各阶段生命周期、
实际差值及控制/任务资格限制。
