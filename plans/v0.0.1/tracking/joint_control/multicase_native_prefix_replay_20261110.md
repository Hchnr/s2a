# 两个开发案例的原生命令前缀重放

2026-11-10。目标是为后续因果安全保护建立可复现的仿真起点，先不生成新控制命令。固定两个已经分析的v4开发案例：`task0_init0_s43_raw`（越界前采样到手部—场景接触）与 `task1_init0_s43_raw`（所选端点未采样到接触）。两者首次越界前，目标关节最后一次实测安全余量至少0.05 rad分别为端点114与99；固定重放其前114/99条已执行命令。该起点基于历史结果后验选定，只是机制诊断，不能当作在线触发策略成绩。

## 固定输入与运行时

绑定完整v4独立复核 `outputs/tracking_wam_prefetch_review_execution_20261028/verified.json` SHA256 `c5e74dd75394dd8ec46b328a8cd1f55f20aa3893e7d08f0becb67bd23eb6cccb` 和越界上下文 `outputs/multicase_joint_limit_context_20261110/result.json` SHA256 `50861e33b9fb5727cc174ea04b79dd38a6b02206d6f181b0f56cecb6b5ea6d10`。两条各使用原600条step记录的固定前缀、601保存状态、endpoint收据、原controller与manifest；所有实际输入按独立复核的SHA核验。原保存未来数据只用于重放结果的离线比较，不用于恢复状态或选择新命令。

复用仓库内 `outputs/native_runtime_path_snapshot_20261029`，其task0 bootstrap和历史命令回放已通过，但不把旧单案例结果推广为新案例资格。两个record XML的99项file声明序列与先前bootstrap任务的原record完全相同：以已成功bootstrap的同序路径映射构造两条派生XML，仅替换file属性，逐值反向恢复原字节；所有96项资产使用先前登记的仓库内引用和SHA。保留每条record本身、初态和task ID；runtime配置仅迁移 `libero_root/libero_config/osmesa_library` 三个路径。task1的BDDL和实际加载依赖绑定当前SHA，不冒称历史二进制完全等价，也不访问旧 `/share/project` 路径。

每条从原初态、原seed43自然恢复，保留历史 initial refresh、夹爪motor target初始化、endpoint/current observation、anchor 0/16/56及每8步摄像机调用顺序。每周期发送原保存7维命令，不调用规划器或世界模型；以原 `1e-10` 绝对、0相对容差逐端点比较完整sim state、时间、TCP与已保存的controller goal/cache及夹爪motor cache，原native task标签逐项一致，并单列state逐字节一致情况。只记录当前运行时的 `initial_joint`；历史资料未单独保存该值，不能据此声称完整控制器内部状态已验证等价。

## 预算与判定

两条独占CPU worker目录，各最多3次初始化模型/3个OSMesa context；控制步上限114和99，合计213步、5325个2ms积分步；保留失败时已尝试/完成数及退出，不自动重试。每个真实子进程实际wait/reap，空CUDA/LD且数值库单线程。输入、执行源码、资产及配置前后核验，输出只写仓库 `outputs/`；不请求WAM/GPU、不使用final fresh、不恢复其他进程。测试时只使用合成runtime验证状态对照、调用顺序、错误停止与计费，不改变原日志或数值门槛。

两条均需到达固定终点、全部对应端点和缓存逐项核对、正常释放，才报告**这两条保存命令前缀**在当前运行时复现；部分成功分别记录，不能代替原v4整矩阵、历史运行时二进制等价、主动安全策略、实时或最终任务资格。以此为基础另行登记更早触发的候选对照，且不得重复声称已有固定 `tau=0.1s` 制动方案成功；原18正确参考、12开发及24留出门槛不变。该回放不是原actual368/H12物理pilot，不改变其完整witness前置条件。
