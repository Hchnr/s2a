# 固定终点方向：旋转改善，但非线性安全预测失效

2026-11-02。按[预登记](../../../plans/v0.0.1/tracking/trajectory_search/local_support_direction_probe_20261102.md)
固定最终anchor23和已保存lp_48方向，执行1/.5/.25三步及anchor/repeat。
没有新优化、拟合或更换方向；候选只用于数值诊断，没有下发。

## 实际结果

| alpha | 预测终点旋转rad | 完整响应rad | 预测reserve | 实际reserve | caps / prefix |
| --- | ---: | ---: | ---: | ---: | ---: |
| 0 | .13108214 | .13108214 | .920588 | .920588 | 12/24 / 2 |
| 1 | .09409409 | .09495896 | .900000 | -171.804980 | 10/24 / 2 |
| .5 | .10615075 | .10647875 | .910294 | -39.656700 | 12/24 / 2 |
| .25 | .11742113 | .11753836 | .915441 | -8.199534 | 13/24 / 2 |

三个方向均joint audit失败，不能执行。原buffer、reserve门和完整caps均未放宽。
全步终点旋转由.13108降至.09496，但仍超过.05rad，而且其他误差使达标项减少；
四分之一步虽多一项达标，也因安全失败不具备资格。没有完整witness或prefix改善。

固定anchor法向的终点实际支持1.47805，预测1.45220；半步2.04470/2.03692，
四分之一步2.33189/2.32928。该方向旋转模型误差相对下降幅度较小，但不能据此
认证所有方向或全域模型；更突出的是安全预测错误。

全步最大安全残差约.00389471，最负行4364（braking_upper）；半步约.00106119，
最负行4368（terminal）；四分之一步约.000353507，最负行4261（braking_lower）。
这些行单位不同，不能把terminal速度与joint position残差直接当同单位比较；
reserve按原terminal .1权重计算后统一对照。全部预测/实测行及分组失败已保存。
模型中接近切向的安全方向，在完整响应中未保有微小余量。仅靠一阶预测和更小
进展门不足以允许较大步；未证明实际物理不可达，也不能据此删除安全行。

## 后续方向

保留本轮负结果，不采用三个方向或追加缩步预算。下一步区分两项问题：
原root±.1是算法搜索域而非机器人全输入能力，局部支持下界的域依赖需要单独
核查；另一方面，任何更大范围方法都必须处理已观察到的非线性制动/terminal
误差，不能用全域一阶解直接下发。后续方法保持实际安全和原跟踪caps，通过
完整响应恢复可行性，单独验证新方向及完整轨迹后才申请物理资格。

## 独立审核与费用

5生产+5独立完整响应，3000内步/3122查询/2model；0新LP/QP/NNLS/stencil拟合。
0物理/GPU/WAM/fresh/最终留出/实际参考消费。全部响应逐值exact，原1e-10/
torque1e-8及布尔审核通过，历史anchor和repeat exact。审核通过表示数值记录
可信，不表示这三个安全失败方向合格。源码和输入前后SHA保持。

61项jobs32专项全部通过，无skip/collection错误，worker实际回收；Ruff/format
及Git空白检查通过。CPU空CUDA/LD、数值库单线程。tool session85820退出0并
实际wait/reap，PID278069不存在，600秒TERM/10秒KILL未触发，无自动重试。

以下路径相对于outputs：

| 产物 | 路径 | SHA256 |
| --- | --- | --- |
| 完整结果 | local_support_direction_probe_20261102/result.json | a176f9e259e4dffe37cc09e193fd57a3974bc1daa0a08fa17299f163e896e10f |
| 逐步分析 | local_support_direction_probe_20261102/analysis.json | 6761b785238af5aebf50e8a5ea9824b1d37555183239a7b61e03a47c8c949a4f |
| owner终态 | local_support_direction_probe_20261102/owner_observation.json | b710616fd5fe5084c391ec701a09e3609f89863da7a4ead40ee0204b8068690b |
| 61项QA | qa/local_support_direction_probe_20261102/result.json | 3b6bb5e217337a74dff73c7d4e94792ac3ede117f91a08dbed681f3fd4a54875 |

总目标仍未达成，当前cursor8/task0/grasp0，原18/12/24和稳定验收保持，GPU禁用。
