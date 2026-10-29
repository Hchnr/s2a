# 106 追加当前wrench覆盖干预

目的：补充训练干预的当前反馈覆盖，检验104模型局部响应失败是否与干预条件覆盖不足有关。原控制资格与最终新初态验收标准保持；本阶段仅采集fit数据。

## 已验证

独立选点检查已完成，凭据为 outputs/wrench_coverage_selection_20261025/independent_verified.json。检查器不调用选点生成器，使用独立传感器坐标变换重新计算world-at-control-SITE wrench，以逐候选标量距离复现贪心选择。仅加载原21条fit轨迹的当前pose、时间、goal override标记和当前wrench；不加载目标标签或holdout观测。

21个新anchor、147个分支、13209显式控制周期（330225积分子步），每任务7个profile。每条完整前缀与旋转/夹爪命令保持，目标周期仅增加baseline或单轴±0.1平移探针。环境初始化内部步骤不计入显式预算。

## 当前执行

采集/逐条独立核验/持久owner与完整关闭入口已准备。静态检查通过；21项基础测试、147条实际CLI dry及检查器导入共149个准入进程均正常退出。来源不变检查通过，持久supervisor和owner已启动；已用/proc的PID及start_ticks核实两者存活。CPU/OSMesa，CUDA_VISIBLE_DEVICES与LD_LIBRARY_PATH均为空。每组baseline合格后才执行六个探针，无自动重试。尚无全147条数据资格，不能宣称采集完成。

执行入口：scripts/run_wrench_coverage_interventions.py。
独立选点：scripts/verify_wrench_coverage_selection.py。
数据关闭：scripts/close_wrench_coverage_interventions.py。
输出：outputs/wrench_coverage_interventions_20261025。

新fit、世界模型调用与fresh测试均0。全147条、294阶段和owner实际wait/reap/来源核验完成之前，不对新增数据拟合模型。无新控制收益或任务成功结论，总目标尚未完成。

后续核实：case00基线和case01探针均已完成独立检查（2/147），case00 baseline_native_match=true；case02采集worker的PID与start_ticks匹配且存活。冻结来源全量check_sources通过。107生产端/独立追加方程组合及资格绑定入口已准备，静态检查/导入通过；未运行资格绑定或fit，尚无全数据方程数值验证，不能以模块导入通过代替完整核验。

再次核实：4/147已采集并独立核验，case04物理worker的PID及start_ticks匹配且存活；冻结来源全量检查仍通过。107完整数值/性能/生命周期入口已准备，详情见wrench_coverage_forward_20261025.md；等待106完整资格，没有提前拟合。

部分数据生命周期复查：首个完整七分支group与下一group baseline均通过机械/前缀检查；前八条的16个采集/核验阶段实际wait0与reap已直接复核。来源不变。此为部分数据审查，不构成全147条数据资格。

## 全量资格完成

147/147采集与独立核验完成；完整关闭审查正常退出，closure_verified.json确认data_qualified=true。全部21个baseline与原轨迹匹配，13209显式控制周期/330225积分子步；294采集/检查阶段及owner实际wait0、reap和来源不变全部核验。初始化内部步骤另列。新增fit、世界模型、fresh均0；此为训练数据资格，不是控制资格。
