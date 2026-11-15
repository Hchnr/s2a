# 参考选择对照的中断记录

2026-11-13 的旧 / 新参考对照已完成 64 配置、64 实际 worker 的 86 项 CPU 回归
和零物理预检，随后启动 producer owner 及两条分支。2026-11-15 核查时原进程
均不存在；只找到阶段注册、两个初始化日志和运行环境的局部文件，未找到阶段或
worker 结果，也没有 checker 结果。两份日志均止于仿真初始化信息，未包含周期
进度。原执行因环境中断而没有完成可审核的对照，其具体中断原因、实际物理开销、
初始化计数及原 owner 的 wait/reap 状态未知。

不得用缺失日志推断零物理步，也不得把此前回归或 dry 当作生产对照通过。原注册
的 producer 预算为最多 96 控制周期 / 2,400 个 2 ms 内步；这是上限而非已执行
量。新的独立尝试与附加进度记录见[恢复方案](../../../plans/v0.0.1/tracking/references/bounded_reference_choice_recovery_20261115.md)。

| 原本地记录 | SHA256 |
| --- | --- |
| `outputs/bounded_reference_choice_20261113/producer/registration.json` | `72ac52cbafc921fbc300d1124647663728da763239bbc1a916cc037a1bb308ba` |
| `outputs/bounded_reference_choice_20261113/producer/old/worker.log` | `cdfbee0a2a6cf271c397617a70f4754ca15c93ac0d94c828c6e1a1ecb349f907` |
| `outputs/bounded_reference_choice_20261113/producer/new/worker.log` | `cdfbee0a2a6cf271c397617a70f4754ca15c93ac0d94c828c6e1a1ecb349f907` |
| `outputs/bounded_reference_choice_20261113/dry.json` | `d31e36afc653988738d706a37cc4b21a1a2c7ba45121fdda394b38e05e536046` |
| `outputs/qa/bounded_reference_choice_final_20261113/result.json` | `11e27efe0964cde2cf6843ea7f02421f6e20f1d5229a47c67723b6db4a98aa81` |
