# 终端单行有限割线修正：未拟合方向外推检验

执行前登记。前置[完整固定网格诊断](../../reports/directional_derivative_consistency_20261029.md)及v3独立复核已关闭：77调用、39差分对、2381来源，完整H12 witness0。闭合SHA `28b5d3b4a7e107ea1ed5e752927b142904a0a5f7bdf1d5bb9dba64ac2a3ca108`；保存分析SHA `717c5d4758ada20cdaeddf6ec7a120ff42a065fa19693333666afcb84524d4f0`。旧terminal行方向预测约1e-6，而中等正步长实际割线约−8e-4；小步长未显示收敛。三旧方向余弦>.977，不能把它们互当强泛化证据。本项检验有限模型修正是否能外推，不证明真导数存在，不预设成功，也不重跑SQP。

只读取已闭合v2 producer的stage24正向h=.001/.0005两条训练响应，共同anchor仍1819；不使用15个新点拟合或选择阈值。原终端行j来自已核验旧72轴Js，索引由完整安全向量的terminal组定位（本例4368），终端normal为e7，slack等于−final_qvel[6]。令u为stage24的单位∞方向，p_i为保存命令减anchor的实际float64位移，a_i=(u·p_i)/(u·u)，r_i=实测终端变化−j·p_i，固定零截距最小二乘c=(a·r)/(a·a)，只修改该行δj=c*u/(u·u)。两训练尺度共同拟合，不挑更有利尺度；其他安全行/Jq、控制器和原caps保持。

独立测试方向完全由旧命令和已知约束构造，不读取新响应。按原3个单位∞方向（不另行行归一化）、box_lower、box_upper、rate_lower、rate_upper的顺序堆叠投影行；四组中slack<=band的有符号梯度均加入，box band=.001、rate band=.002，rate第一周期相对completed367命令。SVD rank取sigma>sigma_max*1e-12；一次`np.random.default_rng(42).standard_normal(72)`后投影到行空间正交补，原投影L2范数须>1e-10。v归一为单位L2，首个abs>1e-12分量取正；max|A v|须<=1e-10。e=u/||u||2，按v、e+v、e−v固定顺序各归一到单位∞。不按响应效果调整符号或替换方向。

每方向固定正步长(.001,.00075,.0005,.00025,.0001)，共15命令，执行前检查全部原box/rate±1e-12准入，保持原字节不clip，15命令彼此及anchor均须不同；任何几何/准入失败在模型构造和首次查询前停止。登记前的只读几何预检（无新响应）得到36行/rank36/nullity36，四组行数[2,0,18,13]，投影范数约4.57234、最大正交残差1.35e-15，15命令最小slack−6.66e-16，全部合法；这些是预检观察，不替代实现的独立核验。

唯一producer执行顺序：原anchor、方向0的全部5尺度、方向1的5尺度、方向2的5尺度、强制末尾anchor。17次完整H12调用、16唯一命令，不使用grid cache，不因提前witness/科学失败跳过后续点，不追加采样。每次从同一actual368 q/v、finger与completed367 latch重新复制初态，仅使用已生成chunk46 H12参考；不读actual369+作反馈。每次300数值内步和312查询，另首次当前查询1，固定5100内步/5305查询、零优化迭代。完整保存rollout、原审计与原strict joint/prefix/witness标签；数值探针不是实际物理积分。

旧/新模型预测都在响应前固定，使用实际p；实测变化统一相对新首anchor，另完整比较历史旧anchor与新首anchor。保留全部15点的原/新预测、误差、按h归一的误差及goal/torque clipping pattern，不剔除跨分支点。两个混合方向分别要求跨全部5尺度的max|新残差|和max|新残差/h|均<=对应旧值的50%；纯v负对照要求max|新预测−旧预测|<=1e-12。初末anchor与历史完整anchor须canonical比较exact，15个新点全部torque clipped flags须与新anchor相同，才标记`extrapolation_passed`。这些要求失败仍保存完整执行结果，不能把科学失败改称执行失败或重试。另单列新预测绝对误差是否<=旧anchor_terminal−1e-8（本例约3.20e-9），即`safety_accuracy_passed`；即使通过也不认证未采样命令、安全控制或任务成功。

新独立checker不导入生产core，重算两点拟合、投影方向、全部命令与逐点/聚合代数；从相同因果输入完整独立重放全部17调用（含重复），原数值容差与strict分类保持。来源包含旧75源加本项10源，共85来源/26测试模块，旧75必须是同摘要子集。通过原生CPU QA、相关Ruff/format、producer/checker/owner实际dry后才允许实际执行。新输出`tracking_v5_terminal_secant_probe[_check][_execution]_20261029`，producer600秒/checker1800秒，排他目录、无自动重试；保存实际wait/reap、来源指纹和失败记录。

新增物理/world-model/fresh/参考消费均0，CUDA/LD显式空，单线程及seed42保持。原18正确参考/12WAM开发/24最终留出门槛不变，实际仍cursor8、task/grasp0，GPU禁止，总目标未完成。只有外推证据足够才另行登记将该修正接入新QP；只看拟合训练方向或17个模型响应不能直接开始物理pilot。
