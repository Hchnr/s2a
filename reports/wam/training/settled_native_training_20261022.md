# 034：完整三十条固定收尾训练数据

**新协议的30/30条均通过独立末40端点稳定检查，三个任务各10/10。** 数据获得下一模型训练协议的登记资格；尚未执行真实WAM拟合或最终24条任务验收。原030的28/30及资格False完整保留。

全部工作在share-mixed仓库；CPU/OSMesa空CUDA/LD，seed42，不使用GPU。沿用原三十条source隔离名单、XML、初态和全部N条native动作。随后锁存N端点实际controller SITE位姿，继续最后一条native夹爪命令80步，前40步settling、后40端点评分；控制不读取task谓词或未来。第五anchor移至N+40，其余四个不变。[运行前登记](../plans/v0.0.1/034_settled_native_training_20261022.md)

| 协议 | bowl | wine | cream | 稳定总数 | 物理步 | 当前canvas |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 原030：零arm delta＋末gripper，40步 | 10/10 | 8/10 | 10/10 | 28/30 | 4980 | 150 |
| 新034：实测SITE绝对goal＋末gripper，40＋40步 | 10/10 | 10/10 | 10/10 | 30/30 | 6180 | 150 |

新协议同时改变末端arm goal语义和收尾时长，不能唯一归因其效果，更不能追改原030质量或宣称WAM策略成功。全部30环境/6180步/150canvas完整执行，无早停、备用demo或重复物理采集。新HDF、future RGB、WAM、fit、fresh及真机均0。

独立NumPy审计核验全部20Hz端点的SITE/BODY/TCP几何、实测state与outgoing delta/mask、夹爪开度及ragged cache、完整native前缀逐位一致、实际80步controller goal来源、末40原生predicate分母和五anchor的相机/物理状态绑定。前四raw相机逐值复现历史；canvas按事先固定的≤1 uint8通道差合同独立重组，保留其与历史PNG非bitwise等价的边界。第五anchor没有读取未来RGB。

首个034 owner以False结束：其worker已正常完成，但checker在读取历史task0封套时误要求passed/complete=True。该task0在旧030保存完十条case后因环境guard失败，后续已有明确独立prefix审计。新[只读审计补充](../plans/v0.0.1/034_cached_task_audit_addendum_20261022.md)要求精确prefix ref及十个case refs，保留原失败，复用其余全部数学检查；仅新增CPU审计，不重复物理。补充owner及审计均actual wait0/reap，无信号退出。原034 False owner也没有回写。

| 完整产物 | SHA256 |
| --- | --- |
| preparation/prepared.json | `f957af1e8dec6bfe24a104d98ecfbe24145375370d79671931ed4df162533d8c` |
| collection/result.json | `bd4a8f10e23073d0f40ebfd2b5effe11521d4531603c5314b0c7b7e6fb616521` |
| orchestration/result.json（原False） | `7365f93c0ab7b9d07af81e534441dd8cb7e49f56fffcafa4f79506f62deb30c4` |
| audit_addendum/review/verified.json | `d5f0da94769de82ccd622638ac2177918f859db7fd0adb7940e06e3fa7dbe971` |
| audit_addendum/result.json | `2f7a5cb96681bb5da5700588a317fb38b863bc9113571e6b551cb0f18f55dca1` |

以上相对路径位于`outputs/wam_stability_20261021/settled_native_training/`。审计明确`eligible_for_model_training_registration=True`，`model_training_authorized=False`、`fresh_performance_authorized=False`、`performance_acceptance_completed=False`。下一步先完成035真实模型桥接资格，再另登记有限训练与开发任务实验。

最终只读peer另行完整复算30条trajectory、2400个hold goal、150canvas和120张历史native raw相机，并确认741项消费来源/产物SHA前后不变、两自有进程正常wait/reap。记录`final_peer/verified.json`，SHA `bd61e6350cbdabdbbac3d046b1762860bf90a79c81f2c9e4ec01ae060430124e`；没有新增物理或改写任何来源。

CPU验证：阶段原119相关测试通过；补充后56项定向测试通过。一次与全仓回归并行的定向pytest因共用默认basetemp产生15个临时目录错误，原记录保留；使用独立basetemp重跑56项通过。迁移支持子集950 passed/1 skipped/3 deselected，显式排除旧canonical fixture/源码测试和历史外部Cosmos检查；不覆盖此前完整QA的1F/12E，也不代表尚未冻结的035代码验证结果。
