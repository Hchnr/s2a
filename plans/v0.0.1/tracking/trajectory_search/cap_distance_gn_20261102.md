# Cap-distance GN有限对照（2026-11-02，Asia/Shanghai）

## 前置与科学问题

按cap_distance_gn_design_20261029.md和cap_distance_arithmetic_20261102.md执行。
新预测器的初始化/72探针已独立验证，结果SHA256
80e1fd703fe67a8d44ad90d5b603e86e1af7d9fba50084dc6009ea71157792c5；
新stencil SHA256 0d447082a50a2a8425c438a0a7cfe8a4a9742cb53bba445e015e49db6f6ecc3c。
同预测器sum-q完整通过严格审核，结果SHA256
0bcc9962caabb8f74f92113a8994cfdb41e7e9a1f173781c4d20d603fbcb9ecc，witness0。
该基线总q下降但后三点恶化，本轮只检验已登记的新目标是否改善原完整caps。

## 保持与唯一目标变化

从相同新base/72探针初始化，不载入sum-q最终anchor、模型或优化器状态。
相同replay-order预测器、native_solve TCP、当前actual368/因果367历史、原参考，
命令/关节/制动/terminal/原.01m与.05rad caps、seed42和控制接口不变。
原配置逐字段保留，只在显式objective中选择已固定float64有理化cap loss、完整
投影Jr、GN B/g、无阻尼展开预测和同单位pose下降门。restore保持原merit和门。
保持raw Js/Jq/Je因果更新、拒绝反馈、alpha、信赖域、原QP/NNLS证书及失败机制。
不添加前九点硬caps、epsilon或调参，不额外采集探针，不改旧False收据。

## 预算、时限和唯一owner

生产最多60阶段/60QP/6000QP迭代、122完整H12响应含最后initial repeat；NNLS
每阶段最多一次，原full-active的3倍迭代预算不变，无fallback。独立复算全部
实际响应（最多122），合计最多244响应/73200内步/76130查询/2model。
本轮成本另计，不追加旧运行、数值修复、初始化或sum-q预算。

预检最多300秒，每次模型初始化最多300秒，生产和独立执行阶段各最多1200秒；
唯一owner对整轮子进程设置1800秒墙钟上限。超时只终止其独占进程组并实际
wait/reap，保留False及可得实际成本，不恢复/排队或自动重试。CPU数值库线程1，
CUDA_VISIBLE_DEVICES与LD_LIBRARY_PATH为空。GPU继续禁用；0物理、WAM、fresh
或最终留出，0新增实际参考消费。顺序因果搜索不并行预消费候选预算；QA jobs32。

先验证最终QA全通过/无skip及当前源码树（允许正常提交改变HEAD，不允许改源码），
再由固定owner启动唯一入口，先写registration及所有新目标/驱动/独立审核源码、
配置、计划与输入SHA。独占outputs/cap_distance_gn_20261102及其owner目录。
各模块代码不通过全局导入/函数替换选择目标。

## 独立审核和验收

独立审核不得调用生产投影/loss；重建完整模型、hard rows、原单位KKT/缩放/紧
活动子集、raw因果更新与费用。原numeric1e-10、torque1e-8、完整initial/repeat、
原caps及严格类别/接纳优先级/半径门保留；保护的八个native内核不改。
所有已收费响应独立完整复算，即使无witness也不得只核验选点。

固定报告全部24q和12点误差、前9/后3、固定Lcap与旧直接表达式差、sum-q/maxq、
prefix、最小安全行及reserve、restore次数、QP/NNLS与完整预算。缺完整witness
只记科学失败或局部改善，不增加预算、不更换初态/权重，不宣称物理不可达。
只有原全部caps、安全、来源、独立严格审核均通过才可另行登记有限物理pilot和
独立命令回放；数值通过不授予任务成功。原18/12/24与最终20/24、每task4/8保持。
