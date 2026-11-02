# 032：原生末端motor目标保持的完整因子对照

用户已指定share-mixed工作区，迁移及历史路径说明见033。原源码删除前只完成CPU实现/QA，尚未prepare或执行032物理；旧QA818P/1外部Cosmos缺失失败在安全备份保留。本计划在新物理前重建并冻结，不改变任何已执行结果。

固定原wine demo1/35/12/13，各完整原native前缀+40保持；补两方法motor_hold、pose_motor_hold。仅在最后40条七维命令把夹爪槽由原末命令-1改0，arm分别沿用零delta和已验证的当前实测controller SITE位姿锁存；不写cache/ctrl/qpos、不用任务标签或接触控制。与旧baseline/latched_pose八条固定缓存形成统一2×2，共16案例，不按案例拼接方法。

Panda实际每0.002s按clip(cache+[-1,+1]*.01*sign(command),-1,1)更新目标，经两个actuator各自ctrlrange仿射写入ctrl。命令0保持N端点motor目标，不是锁存实测opening。每outer25子步，记录完整真实输入、前后cache、时间、actuator身份/range/实际ctrl与outer目标。已有flattened state不含cache/ctrl，不用其hash代替控制状态检查。

固定新预算8环境/1648步/40canvas/41200 grip calls，0HDF/future图像/WAM/fit/fresh；CPU/OSMesa空CUDA/LD，seed42，完整运行不早停，不替换/重试或覆盖。复用原collect_case与GoalRecordingEnvironment，代码均按历史快照原字节恢复。新root outputs/wam_stability_20261021/native_motor_hold_diagnostic。

prepare冻结原pose prepared/collection/独立canvas旁证声明及SHA；旧JSON不改，通过RelocatedArtifacts定位安全恢复文件。operational record只允许expert_path重定位。历史S2A代码与实际模拟器当前字节必须严格符合原SHA；旧Cosmos实际缺失源码仅用已冻结精确快照证明历史字节，不宣称仍存在，不运行模型。新cv2探针在当前env311独立通过，Python3.11.15迁移属于环境变化，须用真实native前缀逐位回归验证。

独立checker核新8条全分母与旧8条结果、所有当前/历史native前缀（含ragged cache）、substep递推与ctrl、实际目标来源、完整20Hz实测标签/几何/mask/40谓词、原始相机与本轮配对图像；actualwait0/reap/noSignal后才评分。pass表示证据完整，质量按各方法4案例单列。原030稳定28/30及训练资格False、旧历史canvas精确False保留。

先修订CPU QA（单列旧外部Cosmos测试当前不可运行）再prepare/dry-run/唯一执行。若统一pose_motor_hold四例全稳，只获得另行登记全30新协议的依据，不认证原30/30或WAM能力。若仍失败，保留负结果，不无限调两例；数据协议须另外明确失败片段用途。最终目标始终真实WAM当前图像/任务→固定S2A，预留24条≥20/24且各任务≥4/8、末40持续成功。
