# 053：原六个validation demo同步物理参考

沿用036总目标。052的新ridge头已通过完整CPU fit与独立QR，训练误差改善尚未成为泛化或闭环证据；原W2开发0/6失败和旧STATE初态同步未认证记录保留，fresh关闭。

[053计划](../plans/v0.0.1/053_synchronous_validation_20261024.md)与[配置](../configs/wam_synchronous_validation_20261024.json)固定原六个validation demo，不追加、替换或按结果选样。task0 demo16/32，task1 demo4/27，task2 demo14/24；native长度95/92/158/177/106/104，共732条native动作，另每case80步固定控制器SITE目标收尾，总1212物理步。四个H40原生anchor按j*(N-40)//3固定，共24个当前canvas；不渲染未来RGB、不训练新参数、不启动模型或fresh。

## 已完成准入

`src/s2a/synchronous_validation.py`检查固定身份/初态隔离、窗口时刻、XML、工具与参数指纹，并提供current-only模型输入记录。未来档案保持独立引用；在线模型记录不包含原生命令、未来状态或success。当前图像/实测状态的sim state SHA及物理时间必须一致。

23项相关CPU测试通过，覆盖固定任务/demo/步数、训练或fresh初态别名拒绝、状态/XML变更、时刻/状态SHA不一致拒绝、修改未来行不影响当前输入，以及既有原生命令和固定收尾语义。Ruff/format通过。

实际准备绑定298份源码及298快照，三份只读HDF的实际native动作、XML和初态完整复制核对；新tool_contract仅迁移路径，字节SHA及TCP约定与034相同。独立metadata审查及实际sim Python3.11 CLI dry通过，dry环境初始化和物理步均0。

根目录`outputs/synchronous_validation_20261024/`：

| 产物 | SHA256 |
| --- | --- |
| prepared.json | ba343c86753e56d20364e20f776d7983d6d475494753dbdfbead7cf9e75f23ad |
| admitted.json | d5932d59f7c3a74d948e38fa94af5dc3c86896195ba6065a15fd9a4251bab3af |

## 执行与证据边界

实际CPU/OSMesa collector已经启动，CUDA/LD显式空、seed42、8线程，不自动重试；收据在collection/worker_started.json，日志collection.log。完整采集、独立物理审查和原生稳定正控结果尚待完成，不把启动或元数据资格当成同步数据通过。

新collector的物理主体显式派生自034，复用原endpoint/render工具及固定SettledGoalEnvironment，只将渲染anchor固定为四个完整原生窗口。每case真实记录controller/gripper cache、所有端点、原生predicate、实际已发命令及锁存goal。独立审查复用原NumPy几何、真实夹爪cache、时钟、控制goal与canvas重组检查函数，不调用producer标签函数；新审查入口不声称运行原034完整CLI。未来模型采样必须另行登记，当前原生回放不算WAM策略成功。

## 完整实际采集与独立审查通过

collector正常exit0，实际PID2605076已消失；6/6完整case、1212物理步、1218端点、24张当前canvas，0未来RGB/模型/fit/fresh。每条末40步原生predicate均True，**native正控6/6稳定**。原生任务成功不等于新WAM头成功。

独立物理审查进程正常exit0，复核所有端点SITE/BODY/TCP、SO(3)与local-right delta、20Hz真实时间、controller初态刷新与末段锁存goal、夹爪cache、全部native命令、当前图像前后同state/time、两相机flip与独立NumPy canvas重组。六条H40累积位置最大分量差≤3.78e-15m、旋转矩阵最大分量差≤2.63e-14；这些是几何标签重建误差，不是WAM闭环跟踪误差。298源/298快照与全部输入再核通过。

| 完整产物 | SHA256 |
| --- | --- |
| actual_cli_dry.json | 72c057b937db0a7ac55b17a2f3337e3b38304bb74664e1d38d3068425e331a0f |
| collection/result.json | 7da64c1cfa6d96c5568ff0ed1e2c64f63b80934b2fb273d928123a719fa85522 |
| verified.json | abe0a9982fa54c6a8425c90933d60c19fa667d129ee87a2707e6ddf1bf50a6d8 |
| current_inputs.json | 7ccff2b6d2d9b1e2a6129d7b4695a4a38a2ef6a4ea28d845b61be3a587a4c248 |

独立审查后由新增`publish_wam_synchronous_current_inputs.py`导出单独current_inputs.json，严格白名单24条当前图像ref、实测state/mask与当前时间/stateSHA；0 future archive refs、0 native commands、0 task predicates。publisher不在已冻结collector闭包内，不修改053源码或快照。当前manifest仍model_sampling_authorized=False，需要另行新候选采样登记。

新053的当前状态、图像和后续标签已由实际物理端点同步认证；旧STATE的initial_sync_certified=False不回写。旧validation与新参考数据源不同，不逐值比较旧指标。下一步固定054的raw/新头24anchor×seed42/43配对采样、3zero回归、完整trace数值审查及事前state门槛；采样worker只读取current-only manifest与PNG，不读未来档案。
