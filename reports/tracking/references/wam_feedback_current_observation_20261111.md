# WAM 参考反馈后的当前图像

2026-11-11。按[登记](../../../plans/v0.0.1/tracking/references/wam_feedback_current_observation_20261111.md)
及[回调修复](../../../plans/v0.0.1/tracking/references/wam_feedback_current_capture_repair_20261111.md)，
从相同初态重放前 80 条动作和已独立核验的 8 条 WAM 反馈动作；逐一核对原前缀
2000 个内环样本及新增 200 个内环样本、动作和端点。没有重新选择动作。

成功重放实际完成 88 周期、2200 内步，step80 和 step88 当前双相机画布均已保存，
渲染前后仿真状态 SHA 和时钟一致。step80 PNG 与此前 WAM 输入逐字节相同；step88
为 WAM 反馈实际到达状态，其状态 SHA 和画布像素 SHA 均不同于沿正确参考到达的
step88。worker exit0、实际 wait/reap、无信号，环境完成清理。

首次运行已完成相同 88 周期/2200 内步，但我在末端回调中漏传 observer，造成
TypeError，未生成 step88 图像；原失败及 exit1/wait 收据保留。回调直接调用测试
修复后，在新的 v2 目录完整重放，两次累计 176 周期、4400 内步；CPU 修复专项
18 项通过，合并前 64 路相关专项 18 项通过，均无跳过；全仓 Ruff 静态及格式检查通过。本次无新模型请求、训练或
新初态。原 step80 模型运行的 postflight 失败和初始噪声 ledger 缺失仍然有效；
仅靠当前图像不构成合格 WAM 闭环或最终验收。

| 仓库内证据 | SHA256 |
| --- | --- |
| 首次失败 producer | `d9a07980476754d03f4c134bc875e6f449c857897d34f880b90245da48d3f565` |
| 修复后 producer | `6a6dd806da9bb5e8cbb8e8212c8c110ec51b20fa654948ce9a8d954446ad1b27` |
| 修复后分支 | `058bbef1015318f9f28503271839af97dd06abce1c8635782c7a989eb9251584` |
| step80 PNG | `09986f11e165682c8eab81978b1f49c7f37ecbcae5ec1da1c74111ffbcbd1e5d` |
| WAM 执行后 step88 PNG | `93919e6dbba418d01d66a96f80cad05220add68c4405dff5e4a516c4284ba8fc` |
| CPU 修复专项 | `a575603b5f25757fb9d52afc12dd40352b73464c3576831750398962f00c2445` |
| 合并前 64 路 CPU 专项 | `51372a0cfe5b746129a39a35c7f177bfa03fc234e9c669d5f6a1825a10604747` |
