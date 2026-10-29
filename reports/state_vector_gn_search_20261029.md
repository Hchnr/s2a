# TCP误差向量GN对照：停止于求解器资格门

本阶段已关闭：186项专项CPU QA、两条零响应dry、唯一生产及3响应完整独立复算
全部正常wait/reap、exit0；2691来源核验不变。7次QP尝试中仅原restore阶段合格，
随后6个pose QP全部status8且success=False，没有执行任何GN方向的模型响应。
半径降到下限以下后正常停止，完整witness0，未选中可执行轨迹。

这验证了实现、账本和已收费响应，**没有获得GN跟踪效果的实验比较**，也不能据此
认定目标不可达。实际仍cursor8/task0/grasp0，GPU继续禁用，总目标和原18/12/24
验收门槛未完成。实现提交`6f88ced`；[预登记](../plans/v0.0.1/state_vector_gn_search_20261029.md)；
[中英文PR原文](pr_state_vector_gn_search_20261029.md)。

## 改动及独立性

从同actual368输入、原3557完整响应和原72条signed nominal1e-4差分建立Je。
E为TCP平移/.01m与旋转/.05rad的72维有符号数值残差；pose阶段使用
`B=I+Je.T@Je, g=Je.T@E`，删除24条软maxq epigraph行，只留t>=0，initial_t=0。
实际merit为state_loss=.5*sum(original24q)，预测为.5*||E+Je*d||²，ratio不含阻尼。
near-cap阈值仍依赖原F；restore QP、安全/共同数值上限、完整执行caps、原停止、
预算与repeat合同不变。numeric config逐字节等于旧60阶段/122响应config。

代码依据：`src/s2a/state_vector_gn_search.py:46`构建初始Je，`:80`构造QP；
`scripts/state_vector_gn_search_audit.py:37`独立从原compact72重建Je，`:71`独立
构造B/g/A/b；native checker使用独立Je并与生产Je exact交叉核对，独立从fresh q
重判loss、接纳优先级及半径（`scripts/check_tracking_v5_state_vector_gn_search.py:34`、`:51`、`:148`）。
三套因果模型只按同一已收费接纳/原合格拒绝源更新，不预载后验诊断最终Je。

22项core、42项audit、35项execution测试，加旧stage预算与KKT专项，共186项
通过（41.83s，6模块、164源）。Ruff/格式通过，两条dry均0 QP/0响应/0查询。
只读审查曾发现共享初始Je构造的独立性缺口，已在冻结前补齐；未修改旧冻结文件。
仓库已有GN/最小二乘，本次不宣称首次实现，组合效果也不能单独归因于GN曲率。

## 实际运行及停止

生产owner session88762、child2378081已actual wait/reap0，数值阶段耗时17.176376s；
checker owner session48645、child2385936已actual wait/reap0。QA session70578/
child2361955和准入session17237同样关闭。无信号、自动重试或追加预算。

stage0 restore与旧60阶段的完整stage/QP、前2条entry/request exact（仅剔除新增
state_loss及Je metadata）；首次pose为stage1。新3条共享命令（含末repeat）完整
响应都与旧运行exact，两个初始/末repeat及原3557的11字段历史均exact。

| stage | 半径before | QP迭代 | status | 互补相对残差 | 结果 |
| --- | --- | --- | --- | --- | --- |
| 0 restore | .005 | 2 | 0 | 2.250186834e-24 | 合格，唯一接纳 |
| 1 pose | .005 | 9 | 8 | 6.167531258e-7 | solver_unsuccessful |
| 2 pose | .0025 | 8 | 8 | 7.670281209e-7 | solver_unsuccessful |
| 3 pose | .00125 | 16 | 8 | 1.999278499e-6 | solver_unsuccessful + complementarity_failed |
| 4 pose | .000625 | 6 | 8 | 3.878957335e-7 | solver_unsuccessful |
| 5 pose | .0003125 | 10 | 8 | 5.845922736e-7 | solver_unsuccessful |
| 6 pose | .00015625 | 6 | 8 | 1.939814790e-6 | solver_unsuccessful + complementarity_failed |

7QP合计57内部迭代；唯一trial是原restore接纳，pose trial0、拒绝反馈0。
status8为SLSQP的Positive directional derivative for linesearch；即使部分原KKT
指标通过，success=False仍按原门拒绝。stage3/6还超原1e-6互补证书，没有放宽。
最终radius=.000078125<.00015625，termination=radius_below_minimum。
剩余53阶段和119响应不追加。末5stage无响应、无anchor进展，只按原规则缩半径。

每路线3完整响应/2唯一命令/900数值内步/937查询；**本阶段两路线合计6响应、
1800数值内步、1874查询**。旧90响应仅作为保存数据读取，没有重跑或计为新增调用。

## 完整误差与资格

最终anchor1只是restore后的数值点，未选作执行点。restore令安全最小slack从
2.235571217e-8升至1.0008491887e-5、terminal升至.00010001870156409123，reserve
为.999849188725894；相对初始命令Linf仅.00004458282108，未饱和人为±.1盒。
它同时令F从5.908710040755111略升至5.910248661677172，总q从29.62160641175905
升至29.623232371481443。没有pose候选响应，不能把与旧final的差异称GN收益。

| 完整停止结果 | 旧60阶段maxq搜索 | 本次GN程序（仅restore接纳） |
| --- | --- | --- |
| 最终anchor | 88 | 1 |
| F=maxq-1 | 4.544808922575216 | 5.910248661677172 |
| 总q / state_loss(总q/2) | 37.56943269436005 / 18.784716347180025 | 29.623232371481443 / 14.811616185740721 |
| 连续前缀 / 双caps点数 | 4 / 4 | 9 / 9 |
| cursor9位置m / 旋转rad | .01634614258 / .04916245591 | .01146511866 / .05733690212 |
| 最大位置m / 旋转rad | .02354741795 / .11756541309 | .01886894072 / .13143675914 |
| 最小安全slack / reserve | 9.993804517e-6 / .9983804517 | 1.000849189e-5 / .9998491887 |
| 完整witness / 执行选点 | 0 / 无 | 0 / 无 |

以下为全部12端点；q<=1才达到对应原cap，位置列单位mm，旋转rad。原执行门始终
是位置.01m、旋转.05rad；共同搜索上限6.977797141162663不授予执行资格。

| cursor | 旧final位置q | 旧final旋转q | 新位置q | 新旋转q | 新位置mm | 新旋转rad | 新双caps |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 0.056291497 | 0.130622493 | 0.054650234 | 0.129530325 | 2.337739 | 0.017995161 | 是 |
| 1 | 0.500898884 | 0.559175265 | 0.460731593 | 0.550196720 | 6.787721 | 0.037087623 | 是 |
| 2 | 0.770636642 | 0.885010004 | 0.635572749 | 0.768817122 | 7.972282 | 0.043841109 | 是 |
| 3 | 0.683308057 | 0.999547172 | 0.490594767 | 0.741587485 | 7.004247 | 0.043057737 | 是 |
| 4 | 0.737433017 | 1.405528868 | 0.532692852 | 0.999710185 | 7.298581 | 0.049992754 | 是 |
| 5 | 1.369142055 | 1.202309690 | 0.999862667 | 0.768249105 | 9.999313 | 0.043824910 | 是 |
| 6 | 1.334507998 | 0.888406350 | 0.825941623 | 0.501922602 | 9.088133 | 0.035423248 | 是 |
| 7 | 1.958488025 | 0.321572506 | 0.999856168 | 0.125299327 | 9.999281 | 0.017698822 | 是 |
| 8 | 2.341195360 | 0.257881771 | 0.992942525 | 0.240150361 | 9.964650 | 0.024502569 | 是 |
| 9 | 2.671963773 | 0.966778828 | 1.314489459 | 1.315008138 | 11.465119 | 0.057336902 | 否 |
| 10 | 3.891334214 | 2.563940761 | 2.307346325 | 3.397462139 | 15.189952 | 0.092161029 | 否 |
| 11 | 5.544808923 | 5.528650542 | 3.560369238 | 6.910248662 | 18.868941 | 0.131436759 | 否 |

独立native路径逐条复算完整300内步及原audit，严格分类分歧[]。最大跨路线torque
差7.922551504e-13、归一化平方误差q差7.904787935e-14；辅助JIT/Python goal差26项、最大
8.940696716e-8单独记录，未以其放宽strict资格。独立完整repeat exact。

## 闭合证据

| 记录 | SHA256 |
| --- | --- |
| `outputs/state_vector_gn_search_qa_20261029/result.json` | `e3c5227cae662caeb8967bbe7fd855b90bba1417854cee7ca0d5b3f504f391d3` |
| `outputs/state_vector_gn_search_admission_20261029/result.json` | `79b8ac2e895be4b0bd8780232bdf422d0dd632a7d8f9b0909b4d432f293f172f` |
| `outputs/state_vector_gn_search_producer_20261029/result.json` | `82d7d731cb3e28baab88d806a0e54443273989dbf923da083bfbd4492cb0ad4d` |
| `outputs/state_vector_gn_search_checker_20261029/result.json` | `84bfe21866c2435925291a881006e682d62538c8a661927029ad959ee406d8c6` |
| `outputs/tracking_v5_state_vector_gn_search_20261029/verified.json` | `0de4830d9aee91571d246c3fc4d69e7c480291cc7a2ed41c7b6a8df16d159dbd` |
| `outputs/tracking_v5_state_vector_gn_search_check_20261029/verified.json` | `12a0a4e2b849e167211161d851c871e454d31c734207742b44e7df6fdccbbff2` |
| `outputs/state_vector_gn_search_analysis_20261029/result.json` | `3ecf9c1c09b2c073dae70a1b5063b4f2a0ee5028a808e6bdc38fc3146a1652fa` |
| `outputs/analyze_state_vector_gn_search_20261029.py` | `48935ecb2063734f9b78ab724adc4790dd734f66d9cd6dc3b54975e8f889bbc9` |

保存数据分析session72698 actual exit0，2691来源不变；生产2674源、checker2676源
均保持。分析只读取新3/旧90响应及原compact bundle；3657大归档只流式SHA核验。
分析没有新QP、模型响应、物理控制或测试。

下一步先诊断同一已保存GN QP的数值尺度与原证书。旧solver已有逐行缩放
（`src/s2a/accepted_point_sqp_core.py:76`），不把重复行缩放当新机制。任何新求解
策略须单独预登记、独立验证等价关系及原KKT，不能追认本轮status8为合格。
新增物理/world-model/fresh/实际参考消费均0，GPU继续禁用。尚无独立物理新轨迹、
在线多任务稳定性或留出泛化收益；原18正确参考/12WAM开发/24最终留出验收门槛保持；最终留出仍未开放。

## 保存QP的零新调用尺度诊断

在本轮搜索和全量checker关闭后，独占v3纯矩阵诊断已actual wait/reap0
（session82646、child2436502）。25份显式输入/源码不变，root另行实核25份SHA。
它从原stencil重建模型、逐值核对stage0三套割线metadata，并独立重建六份保存
pose QP的原证书；不重新求解，不接纳status8。QP/模型/响应/测试/物理/GPU新增均0。

首pose的B特征值为[1.000000546,278.129343]，条件数278.129191；g无穷范数
11.23197515727354。原canonical A的条件数6.291980，raw A为43.579450；
去掉行除数max1下界的候选正缩放反而为36.699389，未显示条件数收益。因此不能
把status8归因为已证明的严重病态，或声称再做行缩放就修复了问题。

零初点的safety floor最小slack4.999245944e-6、共同pose ceiling余量.06754847949；
命令/rate/t边界有13个exact0，另一个rate slack为-5.551115123e-17。它通过原
primal容差，却不是严格内点；该末位负值也不是实际约束不可达的证明。

旧solver已对每行除S=max(1,||A_i||inf,|b_i|)，目标仍为未缩放
f=t+g·d+.5*d.T@B@d，解析梯度，SLSQP ftol=1e-14。下一项最小对照可仅把**整个
目标和梯度**除C=max(1,||g||inf,radius*||B||inf)，保持旧行缩放；首pose C为
11.23197515727354。统一缩放不改变精确最优解或B条件数，可能改变有限SLSQP
路径/状态；本诊断没有执行该求解，也未证明能消除status8。

若一般化写成目标f/C、约束(Ay+b)/R>=0，变换后dual为nu，则原raw dual应为
lambda=C*nu/R、原canonical dual为mu=S*C*nu/R。六份保存dual往返最大差
3.552713679e-15，原KKT仍按原B/g/A/b、原行S及原阈值复核；所有status8仍拒绝。
只缩B/g而不处理t系数不属于整个目标的统一缩放。

两次诊断工具失败完整保留：v1未沿用冻结入口的C-contiguous数组副本，stage0 residual末位差异
触发exact守卫，session73864 actual exit1；隔离v2补同样的数组副本后模型/六证书重建通过，
但额外的禁止SciPy/s2a import守卫失败，session55588 actual exit1。该import守卫
不是研究或KKT门；v3仅移除此过严守卫，明确记录冻结audit的传递imports，未调用
optimizer/dynamics或测试，也未引入AST替换或改变数学。修正诊断不重开旧搜索。

以下文件均在`outputs/state_vector_gn_qp_scale_diagnostic_20261029/`：

| 文件 | SHA256 |
| --- | --- |
| `analyze.py` | `8c64dc21c45baa95ec238307b6df47d8e5c6e44c03bf33af8c501601b4c13a72` |
| `result.json`（失败） | `03b903e552d57bf8c9b4af243702ece61f0af1f4842aa1beb70697370b3c782d` |
| `v2/analyze.py` | `44347d3664294339b5f406d23e83edff22f7e0d7d214d8807a12b7b3f3203539` |
| `v2/result.json`（失败） | `17d9c09fd979c2e44c1d352bc1fcefbfc3d6edc5e38000e23b6a33633441f769` |
| `v3/analyze.py` | `ced6eae62fb02bf9e2bab92ec3e2a318d92e66bfcdc536a11dff4ec0f6b72e5e` |
| `v3/result.json` | `3d26dc03672f2f32b60e244de155e6927d9cf2f19128094af674fcc397e54074` |

下一项[整个目标统一缩放对照](../plans/v0.0.1/state_vector_objective_scale_20261029.md)
已登记，尚未实现或执行。此为新的有限数值形式对照，不使用本轮未花预算，不保证
获胜，不提升控制证据等级。GPU继续禁用，总目标仍未完成。
