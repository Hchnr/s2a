# seed 43 复位闭环的预检接线

2026-11-13。复用既有的 64 并发 CPU 准入、零模型 dry、权重与源码预检，
将模型登记的图像根目录绑定到 producer 实际创建的 `simulation/runtime`。
在不初始化模型或物理环境的情况下，已使用真实历史登记、seed 43 和完整
163 项权重清单验证模型 bundle 可被单请求服务准入，并将修正后的独立 checker
源码纳入清单。

首次单项测试命令漏设 `PYTHONDONTWRITEBYTECODE=1`，严格环境门禁阻止了
真实输入加载；补齐该变量后 3 项通过。随后 32 并发相关回归 **59 项通过、
0 skip、0 collection errors**，worker 均 wait/reap，来源前后一致。结果
`outputs/qa/reset_seed43_trial_20261113/result.json` SHA-256
`faef0ed2767d6466819cacf187f9fc5317aaffa163ff7040b7c35fd35fc97bab`。

此阶段没有模型初始化、WAM 请求、物理步或留出评估；现场进程、最终 64
并发准入、dry、预检及实际复位闭环仍未执行。
