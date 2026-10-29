# 固定方向诊断：隔离独立复核 v3（执行前登记）

原[v2 producer](directional_derivative_v2_admission_20261029.md)已经正常结束，77完整H12调用、76唯一命令、23100数值内步、24025查询，完整witness0。v2 checker已完成全部77次独立响应重放及固定网格代数，但随后在旧stencil锚点的零容差bounds检查退出。失败result SHA `f764512e07864a8dc89774059c43ed71ac76dca5914ad4531fa8a132b2ee5aab`、owner SHA `feebe6faa1ff11491e94e8ed2c32b15573f5cf2c9dfe18517f392a71069bf852`；原session7602实际exit1，child205307/start616309119实际wait1/reap，无信号或自动重试。该失败不是完整独立核验通过，不能复用其部分复算替代新checker。

本次只登记独立checker修正，不重新运行producer或追加采样。旧anchor首行两坐标分别超严格stage边界约1.11e-16、7.22e-16，仍在原command 1e-12容差内。新独立stencil helper从原audit提取两个函数，唯一语义改变为anchor bounds应用原±1e-12。不clip或改变anchor、FD符号选择、原FD命令及响应关联、方向、差分公式、joint/pose分类、原caps或安全buffer。旧FD探针的bounds检查本已使用±1e-12，保持原样。所有v1/v2源码、计划、QA、结果和失败日志原字节保存。

新增7来源（helper、checker、owner、evidence、本计划及2测试模块），完整QA共75来源/23模块。新checker QA与原producer QA分开记录；原producer68来源必须是新75来源的同路径、同摘要完整子集。测试覆盖数ULP/微小roundoff内保留字节、超原容差拒绝、旧stencil篡改拒绝、不同QA的合法绑定、部分结果及异常生命周期拒绝。原生CPU QA、相关Ruff/format、checker与check-only owner实际dry全部通过后才启动一次新复核。

新输出仅为`tracking_v5_directional_derivative_v3_check[_execution]_20261029`。owner CLI只提供check，不提供生产阶段。固定1800秒、无自动重试。新checker从相同actual368状态完整独立重放全部77调用（含末尾重复），核对23100内步/24025查询、80请求/39差分对、72轴旧FD与3个阶段的Js/Jq绑定、全部请求预测和历史anchor比较，完成JIT、来源及实际wait/reap生命周期核验。任一步失败即保留结果，不覆盖或借原v2部分结果宣称成功。

输入仍仅actual368当前状态、completed367命令/latch与已生成chunk46参考；不读取actual369+作为反馈。只执行CPU数值查询，无物理积分、世界模型调用、新参考消费或fresh案例；CUDA/LD显式为空，单线程与seed42保持。原18正确参考/12WAM开发/24最终留出门槛不变，实际仍cursor8、task/grasp0，GPU继续禁止，总目标未完成。完整复核后才允许解释全固定步长诊断；不据此直接取得控制或任务资格。
