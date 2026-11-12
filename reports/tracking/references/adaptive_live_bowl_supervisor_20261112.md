# 连续重规划：试验监督与失败收据

2026-11-12。新增受登记限制的试验入口：分别准入 CPU 回归和零物理 dry run，
预检 GPU、模型源码和权重指纹，再启动一组隔离的仿真与模型 worker；独立 checker
只重放 producer 实际派发的动作。启动目录、输入和来源保持阶段隔离，工作进程
需要实际退出并由监督者回收。模型收据缺失时，相关开销记为未知，而不记为零。

32 并发专项回归 **56 项通过、0 skip、0 collection errors**；结果
`outputs/qa/adaptive_live_bowl_supervisor_20261112/result.json` SHA-256
`feb4b3453e87d79226239e2f172b651bc6b01159ec0afd980e5118c68cd5c76b`。
监督入口的 CPU 导入检查未加载 MuJoCo、robosuite、LIBERO、OpenCV、PyTorch 或
Cosmos。上述验证不包含新模型初始化、GPU 推理、物理步或留出案例；最终 64 并发
回归、实际预检与登记试验仍待完成。
