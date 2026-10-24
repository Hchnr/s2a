# 030失败端点静态谓词诊断

在新增环境初始化或状态恢复前登记。原30条同步native数据已完整独立审计，来源/几何/像素/进程通过，末40稳定成功28/30；原30/30训练资格False不改，031模型不启动。本诊断不替换失败示范、不缩为28条拟合、不延长原结果或改成功定义。

两个失败在native末端都为True：wine demo1从N+2起持续False，demo35仅N+1/2为False、N+3恢复至结束。零arm及末gripper command延续不等于锁存测得TCP/opening，已有数据同时出现TCP漂移和张爪；目前只有相关性，未定位官方成功谓词的具体变化条件。

## 固定问题与输入

只问：在已保存的转换端点恢复MuJoCo状态后，能否精确复现当时的官方成功布尔值；wine/rack的哪项区域、接触或姿态条件发生变化？静态恢复不重新执行动作，不能据此认定渲染、控制缓存或张爪是唯一原因，也不将静态状态安装称作闭环成功。

配置固定为`configs/native_terminal_predicate_diagnostic_20261021.json`，root为`outputs/wam_stability_20261021/native_terminal_predicate_diagnostic`。消费原native prepared、完整collection/audit/owner、真实tool terminal和失败时序诊断的固定SHA。只读取下列四条既有case JSON/trajectory/endpoint receipt及此前保存的初态/XML，不开新HDF、EP、PNG、模型权重或expert动作文件。

| 角色 | case | 固定端点 |
| --- | --- | --- |
| 失败 | task1_native_demo_1 | 174,175,176,177 |
| 失败 | task1_native_demo_35 | 158,159,160,161 |
| 匹配正控 | task1_native_demo_12 | 177,178,179,180 |
| 匹配正控 | task1_native_demo_13 | 155,156,157,158 |

正控按已固定同任务、native长度接近的两条选择，原独立质量均通过。它们只用于诊断，不是新validation或fresh。此前提出的4条HDF全状态扫描和渲染配对物理实验不在本次预算内；保存端点已足以先定位具体谓词条件，优先减少额外变量。

## 实现与来源

保留全部原423源码/快照、原421旧快照及原数据字节。新增8个明确来源：本plan、新config、`scripts/run_native_terminal_predicate_diagnostic.py`及`tests/test_native_terminal_predicate_runner.py`、`scripts/collect_native_terminal_predicate_diagnostic.py`及`tests/test_native_terminal_predicate_collection.py`、`scripts/verify_native_terminal_predicate_diagnostic.py`及`tests/test_native_terminal_predicate_verification.py`。owner含prepare/run两个阶段；新注册S2A来源为423+8=431，旧423快照原路径保留，新8在新root快照，执行前全部tracked并固定SHA。

prepare只核已有元数据及原案例数组/receipt身份，不初始化环境、不读取选中端点以外的新数据源。另对配置指定LIBERO源码目录下`.py`及`.bddl`做源码/任务定义清单与SHA，作为独立runtime input inventory；不得借清单读取HDF、图像或fresh性能。沿用原观测、cv2、MountedPanda实际类及映射12个runtime refs。

CPU/OSMesa worker在任何仿真导入前使用已验证的cv2准确导入/LD恢复合同；记录真实PID/argv及CUDA/LD/cwd。每个案例创建一次同原XML/初态的环境，只安装该case已保存的四个raw sim state；每次执行`forward`后检查安装状态/time/source SHA和官方成功值与原数据对应。不得发送controller command、render或调用env.step/sim.step/mj_step/mj_step2。全局与阶段计数在调用入口拦截，禁止调用在委托前拒绝，异常保存False与实际计数。

setup可能调用不积分的`mj_step1`更新反馈，事前保守允许最多64次（4环境总计），逐次保存阶段/时间；static评价阶段不允许`mj_step1`。所有完整积分调用严格0。报告明确区分state安装导致的源时钟变化与实际积分，不能根据time相同假定没有调用过控制步骤。

每端点保存官方goal表达式、实际原子谓词及其对应主体/region/body/geom身份、wine与rack世界位姿、区域边界与几何余量、实际接触对/距离及所有消费引用。真实源码决定具体条件，不能猜测一个bbox替代官方谓词。静态重现不一致时先拒绝归因，保留差异；不能把新静态布尔覆盖原实际执行记录。

独立checker只消费保存的数值/源码/身份和原布尔，独立重算已导出的区域/接触关系及原子组合，不调用producer检查函数或启动仿真；核16端点、完整分母、0积分/图像/HDF/WAM/fits及actualwait/reap。无法独立重建的源码条件明确标记，不用文件SHA冒称数值独立验证。

## 预算与顺序

固定4环境、16保存状态安装、16次static官方成功调用、setup额外官方成功调用最多16次（合计最多32次）、最多64次显式额外原子调用（不含官方内部委托，内部另计数）；setup mj_step1上限64、static mj_step1为0；mj_step/mj_step2/积分步/render/当前canvas/HDF/WAM/fit/fresh均0。seed42、空CUDA/LD，不使用GPU。没有自动重试、备用case、物理配对实验或模型训练授权。

必要测试覆盖禁止积分/渲染在实际委托前拒绝、源时钟安装与积分计数分开、官方谓词不一致保留False、原子几何正负边界、固定16分母、实际CPU child wait/reap/失败清理。全仓QA和来源快照通过后，owner先实际worker，再正常wait/reap、独立CPU checker；只清理本次实际拥有的进程。结论仍保留原native质量28/30及模型训练资格False。

若找到具体几何条件变化，下一项再登记单一可辨别的控制/渲染/源状态对照；本诊断不自动授权延长保持、锁存夹爪、修改S2A或筛选成功示范。最终稳定成功目标及预留24条门槛不变。

## 运行前源码审查补充（2026-10-21，尚未初始化环境）

`make_environment→set_init_state→regenerate_obs_from_state`在setup中实际调用官方成功谓词，每环境至少一次。因此原“16次官方”不能作为全阶段总数；修订为static精确16、setup最多16、总数最多32，所有阶段调用逐次公开。额外原子调用最多64指显式诊断调用，官方内部委托也分别计数，不将内部调用隐藏在该数字中。

`ControlEnv.reset`原实现的`finally: continue`会吞掉异常，包括guard拒绝异常，并可能无限重试。仅本诊断进程把该方法拦截为一次`self.env.reset()`委托，正常路径保留一次reset，异常立即传播；逐次计数且不允许自动重试。外部源码和既有423源码/快照不变。本拦截在任何实际环境创建前安装并记录，测试覆盖异常传播和禁止委托。

除原12个runtime ref和完整LIBERO `.py/.bddl`清单，显式固定7个robosuite控制/接触/模型/绑定源码，以及MuJoCo入口、函数/结构扩展、共享库及头文件的5个身份SHA。二进制身份固定不等于独立证明碰撞检测；独立数学验证范围是从实际导出的完整active contact表重建成员关系及官方区域谓词，不重新实现碰撞求解。

## 2026-10-22运行前源码修订：使用实际注册类

首次正式run在任何环境初始化前因模块类名为None失败；原prepared SHA `68d3bda60d8832e84b2d6b6674a1a09a99e3b9f5b04885d603d133c305ad1675`及原431快照/False结果保留。实际register_problem只填TASK_MAPPING而不return，因此模块符号不是可用类。新collector从实际TASK_MAPPING解析libero_tabletop_manipulation，严格核类名、模块、BDDLBaseDomain继承及官方方法可调用后才装守卫。外部源不变，禁止跳过成功计数。

运行前已逐个验证原431快照SHA及全部计数为0。本修订仅改变原8个诊断来源的必要代码/测试/计划/配置版本，原423不改，新431快照与执行根固定为`outputs/wam_stability_20261021/native_terminal_predicate_diagnostic_registry_v2`。原root拒覆盖，不自动重试；本项是明确登记的源码修复后新运行。原四case/16端点和全部预算/质量门槛不变，重新QA/prepare/dry-run后才执行。先用纯fixture检查注册类身份拒绝，再通过真实env311 bootstrap（0环境）验证所有守卫安装/恢复可用；这些不代替实际端点结果。
