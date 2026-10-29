# 046：新W1来源绑定与实际D1接续

2026-10-24 10:07（北京时间）：**唯一D1采集实际运行，已闭合10/300请求、至少350次已完成请求的denoise；前2请求/70节点已通过原独立检查器的CPU重算。** 两请求初始噪声、条件mask、caption/text tokens及首当前latent与D0逐字节相同，W1数学与compact链通过，部署相对raw改变24303个active velocity值；这不证明适配或任务收益。三层实际PID/birth仍匹配，519源/快照未变。完整D1、独立全量审查和退出资格均待完成。实时采集中尚未精确断言进行中请求的总denoise数。

本项沿用[036总目标](../plans/v0.0.1/036_project_goal_20261023.md)与[046固定协议](../plans/v0.0.1/046_clean_training_d1_continuation_20261024.md)，最终仍为同一候选在原预留24条中稳定≥20/24、每任务≥4/8。W2、开发任务和fresh未启动。

10:22补充：同一三层PID/birth保持存活，已闭合66/300请求，至少2310次已完成请求denoise，519源/快照仍一致；完整D1资格未取得。等待期间[047准备](wam_clean_w2_preparation_20261024.md)已完成142项CPU检查，但实际W2准入和拟合仍被拒绝。

10:47补充：同一三层PID/birth仍存活，已闭合156/300请求，至少5460次已完成请求denoise，519源/快照无漂移；完整D1资格未取得。等待期间[048导入/元数据组件](wam_clean_state_handoff_preparation_20261024.md)135项检查通过，未增加GPU任务，W2/STATE均未启动。

11:19补充：同一三层PID/birth仍存活，已闭合270/300请求，至少9450次已完成请求denoise，519源/快照一致；完整D1/退出资格尚未取得。等待期间[048生产桥与入口](wam_clean_state_handoff_execution_preparation_20261024.md)129项CPU检查通过，实际W2/STATE仍未启动。

## 实现与边界

045真实W1生产与全量独立QR已通过420000行及150标签窗口，参数SHA为`e70ab79facfbb42667700d823c7ca54fbafb03de03aa198767afeb3c5917dea8`，两角色正常wait0/reap。原041 W1 owner与043实际fit失败永久保留，累计W1尝试2次、成功1次。

原041 owner的`prior`固定读取旧失败目录，不能直接重启旧流水线。新[来源核验模块:77](../src/s2a/clean_training_w1_handoff.py#L77)逐项核045真实owner、两角色生命周期、参数、原D0、508源/快照；[有限作用域:182](../src/s2a/clean_training_w1_handoff.py#L182)只临时替换原owner的W1查找，其他stage仍走原函数，退出恢复并再次核源码。没有改写旧结果、转换退出收据schema或修改采集/独立数学源码。

[新准入:89](../src/s2a/clean_training_d1_admission.py#L89)固定D1-only、唯一UUID、1次模型初始化/300请求/10500 denoise及head calls，0fit/QR/physics/fresh，不自动重试。新[worker:87](../scripts/execute_wam_clean_training_d1_handoff.py#L87)在CPU上完成原head/seed/当前输入准入，再调用原041 bundle owner；新[supervisor:109](../scripts/run_wam_clean_training_d1_handoff.py#L109)持有真实worker Popen，继承88400秒总上限、130秒异常清理与实际后代pending/release检查。原模型owner持有真实模型Popen，并负责GPU释放和独立全量capture审查。

当前模型入口仍是原041冻结bundle collector，实际参数通过`--w1-fit/--w1-verification`指向045新路径。采集输出为原prepared.root的`capture/D1`、独立审查`review/D1_verified.json`、原owner收据`orchestration/D1`；额外控制根为`outputs/clean_training_d1_continuation_20261024`。D1沿原300个训练noise和当前图像，不读未来标签；完成后须与D0初始噪声、caption/text tokens、mask及首condition latent逐字节比较。未完成D1不送入W2。

## 验证与准入

8个相关模块219项CPU检查通过，含失败收据/参数替换拒绝、仅W1重定向、异常恢复与嵌套拒绝；新增supervisor用实际CPU小子进程覆盖wait0、exit7与错误结果，原collection/owner/capture-checker/durable/bundle回归也通过。481份主仓库源码及2份准备脚本Ruff/format通过。

原041 owner实际D1规划在阻断`np.load`下dry通过：W1查找1次、原D0查找1次，退出恢复，0数组读取/模型导入/采集。初次测试命令的脚本导入/临时目录配置错误，以及后续合成观察器缺少`all_recorded_descendants_reaped`字段的失败均如实保留；最终测试修正测试配置与合成收据，没有放宽生产收尾条件。初次临时目录误落到默认`/tmp`，后续全部显式使用本仓库独立basetemp，未把该错误当作通过。

新519源/519快照扩展原045的508闭包，原文件SHA不变。独立metadata CLI完成分离proposal/verified/admitted；实际admitted worker和owner的CLI dry均exit0，绑定本次W1和允许UUID。GPU启动前核驱动UUID/PCI/minor、当前可见2/minor6、显存0和进程列表空；端口31338绑定通过，原owner启动时再次检查GPU空闲与端口。

## 当前进程与证据

短launcher正常返回后，supervisor PID2010452/birth574805981已PPID1，持有owner PID2010453/birth574806125；后者持有实际模型PID2010513/birth574806584。CPU两进程的CUDA/LD为空；模型的`CUDA_VISIBLE_DEVICES`仅`GPU-ad74a609-3a39-43cb-1392-24cbd279ce7f`，LD为空，三线程8。10:07观测进一步核了GPU实际UUID/index/minor、三层活体PID/birth和闭合请求；只能证明已记录部分，不能代替完整生命周期、最终wait/reap/GPU释放。

下列路径相对于控制根`outputs/clean_training_d1_continuation_20261024/`：

| 证据 | 路径 / SHA256 |
| --- | --- |
| 219项CPU检查 | `qa/pytest_final.log`；聚合`qa/qa.json` |
| 原owner组件dry（阻断np.load） | `preparation/component_dry.json` / `5ab0aadf5e114aefc2bde812366ae661e7170a0a0da5617fb3c0e983ba503565` |
| 原3个记录后代已reaped | `preparation/retained_descendant_cleanup.json` / `ba4b061e47b225cfab1e2956a5dd4df3d9b3e20699424f0b73369bf4621e9e5d` |
| 新提案 | `execution/proposal.json` / `e9d4c295022267b66f5d457ef96536cba955eeaaa332c8e902ce4ddacf5fe065` |
| 独立metadata | `execution/verified.json` / `b0e5ffa942e764ccabe1ae72b915d4a77ec7ba8df63fbc7e2ae082b30da01741` |
| 实际准入 | `execution/admitted.json` / `76d8751d0213e585b9660b704556dff4c4ae7849aa80c7c067dbdcb694749d49` |
| 实际CLI dry | `preparation/admitted_cli_dry.json` / `bd567fe10c8d876bea1fc261ec6e5e90e1f7c353bb36f10e6f59eed3e0e1f741` |
| GPU/端口 | `preparation/resource_preflight.json` / `baa5b5afc9cad9e225f6dbfeaf904cc65d07a9b5b21553ca27f7e70c64a29ffa` |
| 固定前2请求的原独立重算 | `observations/first_two_audit.json` / `97274f1918c0f6acf7d235cd80dbc4d3ae2341c061d5eda9ecf1dac480d29f73` |
| 10:07活体与519源核查 | `observations/live_v1.json` / `0b1c7d16639e3926629f41a02861589c5534dd12af847474c5bfb7b976d27853` |
| 短launcher | `supervisor/launcher_result.json` / `d7a2102b973c019280b9d939ddb7dd471a179880d3843b90ada28c8af85e57fa` |

[显式配置](../configs/wam_clean_training_d1_handoff_20261024.json)记录原D0、045参数与owner、新输出和预算。后续只观察同PID/birth，不因工具观察超时而重启；任一实际失败保留并停止。完整D1及独立capture审查通过后，才另行固定W2的840000行拟合与独立QR；模型适配收益、开发任务及最终留出门槛仍须分别实证。

前2请求的补充只读脚本在启动后另存`observations/first_two_audit.py`并记录自身SHA，调用原冻结独立检查器，未修改519执行源码。其证据只覆盖保存的action数学及完整初始数组；省略vision/flat/head仍为在线/hash见证及037/039资格，没有重渲染或解码整段video。
