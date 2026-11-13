# 真实 seed 43 重放的输入准入

2026-11-13。[预登记](../../../plans/v0.0.1/tracking/references/adaptive_multiseed_real43_20261113.md)
的只读准入已核对 task0/init0 旧开发案例：首图两份来源的 SHA 一致，记录的
第 0 端点在渲染前后状态与时间相同；原 seed 42 响应与此前单次复现的两组
40×128 数组逐元素一致。记录端点仅供离线来源绑定，不表示本次现场观测。

历史 checkpoint 指纹和 163 项权重清单与已准备模型的清单一致，固定 Cosmos
源码清单为 1004 项。此处核对的是已登记清单身份；运行前仍须实际重算各文件
哈希、完成 OpenCV 探针并确认 GPU 空闲。

32 并发相关回归 **98 项通过、0 skip、0 collection errors**，源码前后一致、
子进程全部实际 wait/reap；结果
`outputs/qa/adaptive_multiseed_real43_admission_20261113/result.json` SHA-256
`94d02b3fbc665e67a6b1498ee04936a0a46c38764831c5b9b5e207b5846510df`。
本阶段没有模型初始化、WAM 调用、物理步或最终留出运行；单次真实推理的
进程监督及执行尚未完成。
