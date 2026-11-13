# 双 seed 自适应模型 worker

2026-11-13。新增独立 worker，将登记的 seed 42/43 同时用于模型初始化前
和每次请求前的随机数重置，并继续要求单张指定 UUID、检查点身份、无效槽
mask、受限请求预算及来源复核。原实验 worker 和请求循环的 SHA 未改变。

CPU 合成联测使用真实 Unix 帧通道，分别完成 42 和 43 的一条请求，核对模型
初始化及请求前的四个随机数接口、实际请求和返回 envelope 的 seed、两次
loader 复核及 35 个合成去噪调用。错误设备、seed、预算、任务、图像路径和
Python 环境均在模型初始化前拒绝；合成加载失败的成本也有记录。

32 并发相关回归 **108 项通过、0 skip、0 collection errors**，源码前后一致、
子进程全部实际 wait/reap；结果
`outputs/qa/adaptive_multiseed_worker_20261113/result.json` SHA-256
`ed8b5af9e7123f31aa2e58c1b4eaeea0581b6ce508f698c222b7cf0dc4c33a29`。
这些调用由合成服务和设备替身完成，没有初始化真实模型、使用 GPU、调用 WAM、
推进物理或使用最终留出初态；真实 seed 43 仍须另行登记并实际验证。

合并前 64 并发相关回归 **163 项通过、0 skip、0 collection errors**，源码
前后一致，worker 全部实际 wait/reap；结果
`outputs/qa/adaptive_multiseed_final_20261113/result.json` SHA-256
`42592adb8ff314137f328cc727c4180d5ac02e2a1ec283cd275a574c832ed4c4`。
全仓 Ruff check 和 format check 均通过；没有重跑全仓 pytest。
