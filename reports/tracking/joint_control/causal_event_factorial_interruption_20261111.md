# 两因素物理对照的中断记录

原[预登记](../../../plans/v0.0.1/tracking/joint_control/causal_event_bridge_factorial_20261110.md)
的producer在会话中断后未留下完整结果。2026-11-10 18:01 UTC重新检查时，原执行
句柄已不存在，进程列表中没有该实验的owner或worker；两条日志最后均记录原始
前缀的64个已完成控制周期、1600个2 ms内步及当时0条新选命令。两路至少执行了
合计128周期、3200个内步；最后日志之后是否有更多消耗、退出码、是否收到信号及
是否完成wait/reap均未知。没有 `workers.json`、分支 `result.json` 或总结果，不能
把本次运行计为通过，也不能将其描述为两因素实验结论。

保留的运行文件包括参考登记、两份混合参考及前缀相机快照；没有可用于恢复机器人
完整状态和控制器历史的检查点。两份参考在producer启动前已分别通过40端点独立
组件核对，但这只证明参考构造，不证明物理结果。原先177项32并发CPU专项和零模型
dry收据仍完整；原运行命令使用空CUDA/LD及CPU/OSMesa。不得删除、覆盖或将部分
费用计入一次完整运行。

| 原始证据 | SHA256 |
| --- | --- |
| `outputs/causal_event_bridge_factorial_20261110/producer/registration.json` | `861acede26517b592e335a3324027d2f9f19b12a3a67b308f66d9edb4006676f` |
| `outputs/causal_event_bridge_factorial_20261110/producer/references.json` | `952417fb54b96acd926a7b49c522a84ec3a45f5b7cbe902993660a8930f458c8` |
| `outputs/causal_event_bridge_factorial_20261110/producer/bridge_pose_old_opening/worker.log` | `55ea575e5adf4c8d6a199700ec11fbef4fbc3253792ed0edd3e7e254d75d9d68` |
| `outputs/causal_event_bridge_factorial_20261110/producer/old_pose_bridge_opening/worker.log` | `1a870779e34211eb6fd4198c5185d3650ff70c184b70d1421121ea1b112f9aa7` |
| `outputs/qa/causal_event_bridge_factorial_20261110/result.json` | `83ea7f96173bd439dc386b83ece84d5899f749f99702738649270bd043af1e3f` |
| `outputs/causal_event_bridge_factorial_dry_20261110/result.json` | `8859115f2b04f649fde0ae65eb3a01c7e8bcc3cc448b5782a0edbb166d0d490b` |

后续仅按另行登记的[恢复方案](../../../plans/v0.0.1/tracking/joint_control/causal_event_factorial_recovery_20261111.md)
在新的独占目录从原初态重新执行；不自动重试原运行，也不修改原结果或预算。
