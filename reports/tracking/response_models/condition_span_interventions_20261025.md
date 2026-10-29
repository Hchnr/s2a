# 115 完整条件方向覆盖干预准备

仅原21条独立fit轨迹的当前条件选点完成：21新增anchor、147分支、精确计划10626控制周期（265650积分子步）。排除原097/106采集点、086三profiles和全部holdout。按profile顺序选择条件特征相对既有支持行空间的最大归一化正交残差，原282维固定条件表达与统计不变，tie最早。

读取109缓存仅raw89、参数仅统计和频率、原生缓存仅protocol标记；不读取目标/预测用于选择，不查看开发条件或性能决定点。每条完整前缀与旋转/夹爪命令保持，新增baseline/三轴±.1。精确名单见configs/condition_span_interventions_20261025.json，规则见plans/v0.0.1/115_condition_span_interventions_20261025.md。

生成器静态/格式检查通过且实际生成名单；独立选点复核与采集准入尚未完成，未启动物理。该选择可能增加干预条件方向，不保证可靠控制，不改变原门槛。当前新增fit/物理/模型/fresh均0，总目标未完成。

独立选点复核完成：scripts/verify_condition_span_selection.py使用逐分量标量Fourier计算和pivoted QR投影，独立于生成器的矩阵计算/SVD，复现全部21点、147分支及10626周期预算，凭据outputs/condition_span_selection_20261025/independent_verified.json。新采集/逐条核验/持久监督与关闭入口静态和格式检查通过，正式基础测试及147条CLI dry准入已启动；尚无物理采集或全量数据资格结论。全部准入通过且来源不变才启动owner，无自动重试。

正式准入全部通过：149进程（21项基础测试、147实际CLI dry及检查器导入）均正常退出，admission.json passed=true。持久supervisor与owner已启动，/proc PID和start_ticks均匹配且存活，CPU/OSMesa空CUDA/LD。尚无全量数据资格，原拟合保持关闭。后续116固定109模型的数据覆盖对照已登记，须本轮完整资格后才执行。

全量数据资格已完成：147/147案例独立验证通过，10626控制周期/265650显式积分子步，21条baseline原生匹配，294个采集/检查stage及owner实际wait0/reap、来源不变均核验。关闭凭据为outputs/condition_span_interventions_20261025/closure_verified.json。仅训练数据资格，不代表任务或控制成功；本轮参数fit/世界模型/fresh仍0。已允许116绑定全部数据。
