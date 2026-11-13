# seed 43 当前图像模型 worker

2026-11-13。新 worker 将已登记的 seed 43 单请求服务接入模型生命周期：
加载前检查固定 GPU UUID、单线程环境、复位后第 8 端点、会话和请求预算；
模型初始化与实际请求均以 seed 43 重置随机数状态。图像目录可由仿真进程
在模型初始化后创建，模型只在完成的当前端点收到请求后读取图像。

合成模型通过真实本地 socket 验证一次请求/35 去噪及提前停止；另验证部分
去噪失败时保留 3 次已完成调用、不发送参考。导入探针确认准入前没有加载
OpenCV、PyTorch 或 Cosmos。32 并发相关 CPU 回归 **97 项通过、0 skip、
0 collection errors**，全部 worker 已 wait/reap，来源前后一致。结果
`outputs/qa/reset_seed43_model_worker_20261113/result.json` SHA-256
`94f9dd12a2655a4d9284430560dc86c9e6b18b0a9f1c0c49df5e0d396c5d7141`。

本阶段未启动真实模型或推进物理。仿真进程监督、独立重放、64 并发最终
准入和实际复位闭环仍待完成；这些 CPU 测试不代表任务成功。
