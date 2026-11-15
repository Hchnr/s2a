# 训练恢复事件前的真实当前图像

2026-11-16。按[预登记](../../../../plans/v0.0.1/wam/training/recovery/recovery_event_images_20261116.md)，
从九个原训练 reset 重放原先实际发送的命令，采集 **12 张事件前当前图像**；
全部图像与原实际端点、真实时间及已有 H40 未来标签通过独立核验。新窗口
使用原 normalization31 时没有活跃槽裁剪。

| 训练任务与三个偏移 | 每例采图端点 | 每例实际重放周期 | 每例新图像 | 对应离线事件 |
| --- | --- | ---: | ---: | --- |
| bowl | 89、120 | 120 | 2 | 第 89 端点后第 8 周期首次 `<0.35`；第 120 端点后第 8 周期首次原生成功 |
| wine | 212 | 212 | 1 | 第 212 端点后第 8 周期首次原生成功 |
| cream | 147 | 147 | 1 | 第 147 端点后第 8 周期首次原生成功 |

选点由已完成轨迹的未来开度或原生任务谓词确定，只是非盲训练数据的离线
选择，不能用于在线 WAM 或控制器。三个偏移共享原 demo，不能当作独立 demo。
`<0.35` 不等于抓取：wine/cream 在成功轨迹中没有该数值。

## 重放、图像和标签

9 个 CPU/OSMesa replay worker 共执行 **1,437 个控制周期 / 35,925 个 2 ms
内步**，均正常退出、实际 wait/reap，构造并清理 27 个模型/context。所有
`1,446` 个被检查端点（每例含 reset）与原记录的 sim-state SHA 一致，时间、
实测状态、controller 字段及夹爪 cache 按原独立重放合同核对；没有选择新
动作或调用模型。基线 replay helper 每 8 周期还读取并丢弃当前双相机，共
183 次；它们不是未来帧或额外保存的训练图像。

12 个新图像均为实际端点渲染，渲染前后状态及时间未变；独立画布重建的最大
通道差均为 1。每张图像对应的后续 40 周期标签从**原先已完整执行并独立
重放的轨迹**重建，本次重放只执行至该案例最后一个新 anchor，没有执行其后
的未来周期。12 个新窗口共 480 行，其中新窗口内部按 case/时间去重为 453 行；
与旧窗口并集为 1,296 个不同 case/时间转移，比旧 864 个增加 432 个。
这不是 1,296 个独立样本，也不包含新源 demo。

预先的真实零动作图像探针执行 0 控制周期 / 0 内步，worker 已回收，使用
3 模型/context。连同九例本次共 30 模型/context、10 个已回收 worker；
0 新 WAM、GPU、fit 或 fresh。运行源码 `1958d9b3`；32 并发专项 37 项、
64 并发相关回归 167 项通过且 0 跳过，Ruff lint 和格式检查通过。

| 本仓库 `outputs/` 路径 | SHA-256 |
| --- | --- |
| `recovery_event_images_20261116/dry.json` | `7ddb39859c40a507c658437f85fb6862dd0e620cc9269070382749f084ba2813` |
| `recovery_event_images_20261116/probe/result.json` | `c1606ea015f5aeb369ca97a46357627c5e3c3643bba166dd276ccc01fb0c828b` |
| `recovery_event_images_20261116/replay/result.json` | `5f974d753060aaaf160a553eca9870861aa75c86a05bd89a1a889b913340edeb` |
| `recovery_event_images_20261116/replay/workers.json` | `291a693c80cc44c7c35d21f8d66fd3f1146ea41d9ae8432700d16bd5d850019d` |
| `recovery_event_images_20261116/audit.json` | `a9e7c77b5ddb8ebce4b86cae9d6ea9f153dfcc8e69ad1ecbab9a6600da461b8b` |
| `qa/recovery_event_images_final_20261116/result.json` | `88359342eaac150ba19a7921559161c98c3b1ea25bb750d485aaef404070b35e` |

本次仅补充有来源绑定的训练候选图像和标签，`training_authorized=false`。
后续仍需固定按原 demo 的分组/权重、原模型当前图像响应、归一化裁剪处理及
未见初态开发验证；不能由图像或 oracle 标签资格声称 WAM 闭环或原 24 条最终
留出验收已通过。
