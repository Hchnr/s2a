# 保存开发图像上的真实 seed 43 模型重放

2026-11-13。按[预登记](../../../plans/v0.0.1/tracking/references/adaptive_multiseed_real43_20261113.md)，
使用 task0/init0 的既有开发首图与固定 `8400_masked` checkpoint，在
`GPU-c0b598ba-0a6e-73af-bbcc-97bc89379bdf` 上完成一次真实 seed 43 请求。
模型初始化、请求、去噪分别为 **1、1、35**；模型请求耗时约 13.52 秒。
请求参数、实际响应身份及客户端 envelope 的 seed 均为 43；模型返回有限的
40×128 数组，后 118 维为零，实际模型数组与客户端收到的数组逐元素一致。

同一图像的历史 seed 42 原始响应与先前复现逐元素一致。新 seed 43 与旧
seed 42 在 40×10 个有效槽的 **400/400 元素不同**；全 128 维 raw 数组最大
绝对差为 0.070125、RMSE 为 0.004459，normalized 数组分别为 0.568534 和
0.041026。raw 不同通道具有不同物理含义，包含无效零槽的汇总误差也不是控制
指标；这些差异只描述同图采样结果，不能推断任务表现或稳定性。

## 准入与执行

预检实际核验 1004 个模型源码文件与 163 个权重文件的 SHA；独立 CPU OpenCV
探针通过，指定 GPU 启动前为空闲状态。64 并发相关回归 **185 项通过、0 skip、
0 collection errors**，所有测试进程实际 wait/reap，源码前后一致；全仓 Ruff
check 和 format check 通过，未重跑全仓 pytest。

离线客户端和模型 worker 均实际退出码 0、wait/reap 完成，未发终止信号；
模型结束后 GPU 显存 0 MiB、无计算进程。源身份及各输入 SHA 在运行后复核
未变。该次没有仿真模型、物理步、action 派发、训练或最终留出初态。此前
长窗链的累计模型初始化/请求尝试/完成/去噪 `11/30/27/945` 加本次后为
`12/31/28/980`；原累计物理范围不变。

独立只读[审计程序](../../../scripts/verification/wam/model_contracts/audit_adaptive_multiseed_real43_20261113.py)
重新读取原始请求和响应：除请求名、seed 和图像副本路径外，与旧 seed 42
请求逐字段一致；两份图像字节一致。审计独立重算两组数组差异、核对 32 个
已登记源码文件及两个进程收据，结果与原报告一致。该审计的 32 并发相关
回归为 **109 项通过、0 skip、0 collection errors**；没有新模型或物理运行。
加入独立审计后的合并前 64 并发相关回归 **193 项通过、0 skip、0 collection
errors**，源码前后一致且 worker 均实际 wait/reap；全仓 Ruff 和格式检查通过。

| 仓库内证据 | SHA-256 |
| --- | --- |
| `outputs/qa/adaptive_multiseed_real43_final_20261113/result.json` | `876ea392386372c05cbdec65c7c149705a0a01e5c8ba82b9d0d679169c2beb63` |
| `outputs/adaptive_multiseed_real43_dry_20261113/result.json` | `4d6253ca2bf28f1b795ae3ab251dfaf89dd78b9b8ceff201f0be009967243b6a` |
| `outputs/adaptive_multiseed_real43_20261113/preflight.json` | `ee5f52ddf4a80255e2f413f7e0e2073304a7eea90625783efbb9d9cf73e3d4fb` |
| `outputs/adaptive_multiseed_real43_20261113/producer/result.json` | `64b018f5fdbd72b3a055e6a60da3194b14522edc321591e5ac291946ab5cad71` |
| `outputs/adaptive_multiseed_real43_20261113/producer/model/result.json` | `de510e3f0b6150a04e95f9f02c6e2f348db4585e7cec46fb156e0ee1e6dff822` |
| `outputs/adaptive_multiseed_real43_20261113/producer/simulation/result.json` | `0af11aa2984d3526de2db8dd47ea30c23390427a69f78b6a0191d99e6a676126` |
| `outputs/adaptive_multiseed_real43_20261113/producer/workers.json` | `7e6d246538ca1bf1821b0b98d779fb5fc0e634e767b4783c293bbcd88a173d03` |
| `outputs/adaptive_multiseed_real43_20261113/producer/model/model_run/step0/response.json` | `3c39cb7143334c972fe023a558846de08a46152d3b97cc7a106a6004b5629297` |
| `outputs/audits/adaptive_multiseed_real43_20261113/result.json` | `49c57feb60f16d9c6659e2d094a439abfd3c305f49dc8685ca084efad166eb79` |
| `outputs/qa/adaptive_multiseed_real43_audit_20261113/result.json` | `0c049ae00c2e42a5e1d639bb8b7538b64ca0635e096dceba1da63955ad7511ac` |
| `outputs/qa/adaptive_multiseed_real43_publication_20261113/result.json` | `d0275ffe1fc31d2cd64819c2bf3768d9805c6f52eefac3b1aa7372b79e9c0687` |

这验证真实模型接口的 seed 43 采样，不证明该参考能使机器人完成任务。统一
候选的三任务完整轨迹、600 周期、24 条最终留出运行、实时和真机均尚未完成。
