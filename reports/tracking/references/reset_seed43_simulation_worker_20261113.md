# seed 43 复位反馈与独立重放 worker 接线

2026-11-13。仿真 worker 在接触物理环境前核验完整预检、登记输入、源码和
producer 模型 bundle；checker 只接受已关闭且释放 GPU 的 producer 及其精确
模型/轨迹收据。producer 接入已有逐周期历史首窗门禁和当前图像请求；checker
接入修正后的端点渲染协议，以免重复既有 checker 元数据失败。

CPU 合成测试覆盖 producer/checker 路由、seed 43、当前图像目录、登记篡改
及独立 checker 未加载候选策略；模拟物理异常时保留已尝试 3 周期和已完成
2 周期的部分收据。32 并发相关回归 **69 项通过、0 skip、0 collection
errors**，worker 均 wait/reap、来源前后一致。结果
`outputs/qa/reset_seed43_simulation_worker_20261113/result.json` SHA-256
`cd70525ee8b960f1300de1266ea8462808fc96dff07e90e312a0f76bbeb15ec5`。

这些是合成接线检查，未创建真实仿真模型、推进物理或发起 WAM 请求。
进程监督与真实两窗口执行仍待完成，不能作为任务成功或留出评估证据。
