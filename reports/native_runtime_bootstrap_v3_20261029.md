# CPU bootstrap v3：历史record路径准入

2026-10-29：实现已提交`b76911e`，新增core/CLI组件分别通过11项和14项native CPU测试。统一29来源、9模块QA已在原会话77357实际exit0关闭，**220 passed / 1 skipped，188.43秒**。实际准入81704和唯一v3 bootstrap原会话15527均已实际exit0关闭。**当前运行时bootstrap通过：79维原初态字节一致、固定控制器核验通过、软件OSMesa产生有效256RGB帧，2774来源不变。** 历史运行时等价及控制资格仍未取得。

本轮按[登记计划](../plans/v0.0.1/native_runtime_bootstrap_v3_20261029.md)与[固定配置](../configs/native_runtime_bootstrap_v3_20261029.json)修复已定位的record路径前置检查。[v1](native_runtime_bootstrap_20261029.md)与[v2](native_runtime_bootstrap_v2_20261029.md)的失败、实际退出及原23份执行来源均保留；不能把后续修复追记为先前成功。

## 修复与证据范围

core仅将历史record声明的条件从“绝对路径且不含`..`”改为“绝对路径”，随后仍沿原`/assets/`suffix执行字符串迁移，不探测历史外部路径。映射后的完整XML必须在模型编译前经过v2已有的逐组件资产守卫：始终留在批准根内，拒绝跨根后返回、symlink、缺失或非目录中间组件。原XML非file结构与数值不变，输出及源码路径守卫不变。见[唯一predicate修复](../src/s2a/native_runtime_bootstrap_v3.py#L48)及[逆AST证明](../src/s2a/native_runtime_bootstrap_v3.py#L61)。

执行函数使用私有namespace，不污染v1/v2模块globals；复用原计数、清理、控制器与渲染检查，不增加模型或控制调用。新CLI的29来源/9模块QA逐SHA保留原23来源，并分别核验两代失败、v2实际wait/reap1、已发生计数、time不推进及outer cleanup成功。见[QA绑定](../scripts/verify_native_runtime_bootstrap_v3.py#L145)和[直接前驱检查](../scripts/verify_native_runtime_bootstrap_v3.py#L159)。

新增[纯文件/AST测试](../tests/test_native_runtime_bootstrap_v3.py#L66)绑定原manifest、snapshot及实际XML处理函数SHA，静态映射全部99个file声明，得到96个规范文件；其中4处历史`..`引用对应3个唯一纹理文件。检查非file XML结构一致、纹理匹配快照SHA且不访问历史根。该项没有构建真实模型；它补齐了前两轮遗漏的路径调查，不能替代实际bootstrap。

固定libero_goal/task8、seed42/trial0、79维原初态、20Hz/2ms和OSC合同保持；每次独立执行仍最多3模型/3context、32次forward、2次非积分step1、一次mj_resetData及单帧256RGB软件OSMesa验收，child上限600秒。禁止control_step/mj_step/mj_step2，空CUDA/LD、单线程及GPU禁用保持；不启动物理pilot或H12控制。

## 统一QA与实际准入

QA子进程PID1066569，native Python3.11.15、空CUDA/LD、单线程、禁bytecode及外部pytest插件。receipt的`complete/passed/actual_wait/reaped/source_unchanged/driver_unchanged=True`，`returncode=0`，无signal/retry；名单为29份来源、9个测试模块。唯一skip仍为旧环境测试所需历史import-only probe fixture未安装，不计作通过；实际bootstrap另行通过cv2环境恢复准入。

已核对以下result/log/driver SHA及QA实际生命周期；没有重复执行测试或全来源rehash。实际准入PID1078354/1078455/1078548三个子进程均wait/reap0，无signal/retry；同一QA与driver绑定，owner登记33份来源。CLI dry核对2770个引用，environment_initializations/new_responses/physics_steps均0，runtime_loaded/bootstrap_verified/control_qualified均False。

| 证据 | SHA256 |
| --- | --- |
| [统一QA receipt](../outputs/native_runtime_bootstrap_v3_qa_20261029_urtyibc2/result.json) | `288566d9a106cbb72c65857788c6f6b1817082838d77f3ed50e350b5e0cf3292` |
| [QA日志](../outputs/native_runtime_bootstrap_v3_qa_20261029_urtyibc2/pytest.log) | `8aeba6d11fbfb7ee78cdbd7169252f03faf47b596e7e7fe9320b1feae31e57f9` |
| [QA driver](../outputs/native_runtime_bootstrap_v3_qa_driver_20261029.py) | `d23d47afaae841f5c5c60ce718e61d43b5142fd261c22c8d2f1ebe563d894b59` |
| [准入/执行driver](../outputs/native_runtime_bootstrap_v3_execution_driver_20261029.py) | `baf88f00531cbb2bbf6491499e9ac9d61781518d8abaf21704ee7203f61b5f75` |
| [实际准入receipt](../outputs/native_runtime_bootstrap_v3_admission_20261029/result.json) | `a5ef6a2a8dd5e1855bd2604273c68007fadc30cbcb93ac231295d1bcb5e5d3d0` |
| [实际bootstrap receipt](../outputs/native_runtime_bootstrap_v3_20261029/result.json) | `28b6001d9d7b2f9d1618968856612479b2f4c623c27979f0f934c3ba621d319c` |
| [bootstrap owner](../outputs/native_runtime_bootstrap_v3_execution_20261029/result.json) | `27fb413a080d6e47c9680f0f909eddfd8e0382c4d462e115f2db1d9163ad48a2` |
| [bootstrap日志](../outputs/native_runtime_bootstrap_v3_execution_20261029/00.log) | `4f764f681d1596d2f76e5f8b3e17829751d4b498d70c74d9d808c5e995b5d48c` |

## 实际bootstrap证据

child PID1081718、parent PID1081669，owner实际`wait/reaped=True`、`returncode=0`，无signal、自动retry，登记timeout600秒。同一QA、准入与driver绑定；CLI和owner均`complete/passed=True`，`runtime_loaded/bootstrap_verified=True`，`control_qualified=False`，`historical_runtime_equivalence_claimed=False`。2774份来源的执行前后核验通过；本报告仅复核上述小量receipt/log、帧与加载库身份，没有再跑模型或重复全来源rehash。

constructor、hard_reset、record_xml三个阶段均完成，每个模型含99个file声明、96个唯一规范资产；规范化改变声明分别为5/5/4。前两份模型XML的SHA同为`4d9058e4c989b9628bf34730284c1c9088ce9eec9d775a8ce93e48aace5b1674`，恢复record模型为`39f96e3e183afc79e17fcc39d3a481430fa759f97fac02e1dd0566ebcbc439f1`。outer cleanup实际调用close并完成，无cleanup_error。

`initial_state_check`记录shape `[79]`、exact=True、max_abs_difference=0，expected/actual SHA均为`5239d6a1c021e69231f3ef91a1ad93bd5b626981c003140b29150c0c9713a702`。18条time检查均before=after；保存初态恢复后时间为0.25000000000000017，不能写成所有观察均0→0。forward/step1未积分推进时间，render前后完整state与time也保持。

实际controller与原contract逐项一致：MountedPanda/PandaGripper，world坐标、`gripper0_grip_site`；6维input范围±1，输出平移±0.05m、旋转±0.5rad；kp=150、kd=24.49489742783178，20Hz与model timestep0.002秒。追加检查use_delta=True、fixed impedance、无位置/旋转interpolator、uncoupling=True。

渲染context的MRO包含`robosuite.renderers.context.osmesa_context.OSMesaGLContext`及`mujoco.osmesa.GLContext`，实际renderer为Mesa `llvmpipe (LLVM 20.1.2, 256 bits)`，OpenGL为`4.5 (Compatibility Profile) Mesa 25.1.7-1ubuntu2~24.04.2`。实际加载快照中的`systemlibs/libOSMesa.so.8.0.0`，15565176字节，SHA`40e28f0f15c4cd13ad3c353509eb2fad79d8041528c59dc56d6e68fae8d1677a`；这证明当前加载身份，不证明历史动态库等价。

单帧agentview为uint8 `[256,256,3]`，196608字节，实际像素min0/max242；[raw RGB](../outputs/native_runtime_bootstrap_v3_20261029/work/agentview.rgb)的SHA为`75750437e1c77c6a729aeb1ae630ae49728e3753c45512713531837800d50f2e`。[PNG展示](../outputs/native_runtime_bootstrap_v3_frame_inspection_20261029/agentview.png)仅按原行序无编辑编码，[转换记录](../outputs/native_runtime_bootstrap_v3_frame_inspection_20261029/conversion.json)确认解码像素与raw逐字节一致、无新增render；不声称历史图像一致，也未用作world-model输入。

## 各次实际成本

以下每个计数均为attempted/completed，保留全部三次真实执行成本。

| attempt | 实际退出/停止点 | 模型 | context | forward | step1 | mj_resetData | render / read_pixels |
| --- | --- | --- | --- | --- | --- | --- | --- |
| v1 | exit1，constructor资产守卫 | 1 / 0 | 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 | 各0 / 0 |
| v2 | exit1，record XML前置守卫 | 2 / 2 | 2 / 2 | 9 / 9 | 2 / 2 | 0 / 0 | 各0 / 0 |
| v3 | exit0，bootstrap完成 | 3 / 3 | 3 / 3 | 16 / 16 | 2 / 2 | 1 / 1 | 各1 / 1 |
| 三次合计 | 保留两次失败 | 6 / 5 | 5 / 5 | 25 / 25 | 4 / 4 | 1 / 1 | 各1 / 1 |

每次configure均1/1，合计3/3。v2与v3各一次高层`env.env.reset()`完成，不能因v2低层mj_resetData计数为0而写成“未reset”；v2的9次forward与2次step1共11条观察均time 0→0。v1局部cleanup失败保留，v2/v3 outer cleanup成功。三次均无自动retry；control_step/mj_step/mj_step2各0/0，S2A动作、参考消费、世界模型及fresh均为0。

本轮已证明当前运行时可完成原初态恢复、controller核验及软件渲染，并保持来源与非积分时间合同。尚未证明历史二进制等价、轨迹执行或任务控制成功。原完整H12 witness0、实际cursor8、task/grasp0、18/12/24任务门槛与总目标保持，GPU继续禁用。
