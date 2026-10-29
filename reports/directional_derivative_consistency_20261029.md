# 固定锚点方向导数一致性：CPU诊断

固定网格v2 producer与隔离v3完整独立复核均已正常关闭：77完整调用、76唯一命令、72轴旧FD、39差分对及历史anchor全部核验，2381来源闭合通过；完整H12 witness0。旧terminal线性预测明显偏离中等步长的正向响应，更小步长的差分未显示收敛；不能直接把缩小FD或中心差分作为已验证修复。v1/v2失败记录原样保留。实际仍cursor8、task/grasp0，GPU禁用，原18正确参考/12WAM开发/24最终留出门槛保持。

按[执行前登记](../plans/v0.0.1/directional_derivative_consistency_20261029.md)，固定旧trace1819命令锚点与stages24–26的三个方向，使用13个预定步长、先正后负，最后强制重复anchor。总计80计划请求；非法box/rate请求保留原因并跳过，合法命令即使预测joint/pose失败也完整记录。不优化、不裁剪、不替换控制器、不自适应追加请求。

每次实际调用仍从actual368当前q/v、固定finger和completed367 latch开始，anchor只是一组命令。强制repeat绕过cache并单独计费；最多80完整H12调用、79唯一命令、24000数值内步及24961查询。实现见[core:92](../src/s2a/directional_derivative_probe.py#L92)、[初态及计账:37](../scripts/probe_tracking_v5_directional_derivative.py#L37)。

逐完整安全向量和逐pose项计算前向/后向/中心差分，分母保持登记的h/h/2h。保存实际浮点位移及相对名义位移误差；缺一合法侧或实际位移为零不伪造中心差分。旧72轴原精度FD按请求/命令字节/响应索引关联重建Js/Jq，保存J*u以及全部80请求的J*actual_delta、观察变化和残差。独立审计不导入生产core；见[原stencil绑定:156](../src/s2a/directional_derivative_evidence.py#L156)、[独立代数:403](../scripts/directional_derivative_audit.py#L403)。

每个实际调用，包括末尾repeat，都必须完整独立重放，再以数值核验过的保存原精度值复核差分。不同浮点重放路线不冒充同一个原FD。首次/末次anchor按canonical JSON数值数组逐路径比较，不声称序列化前dtype或空数组隐藏维度；若有差异则保留并限制导数解释，不重试或平均。历史比较限定旧trace和accepted-anchor audit；旧ray另有完整rollout，本项没有用它冒充新响应。见[逐响应复算:47](../scripts/check_tracking_v5_directional_derivative.py#L47)、[历史比较:99](../scripts/check_tracking_v5_directional_derivative.py#L99)。

## 已关闭的原生QA

[QA receipt](../outputs/directional_derivative_native_qa_20261029_vpffywr0/result.json) SHA `e420c1b206ce19bf2e47b1d1ed5a7434dcfc1b60f8cdc6d2df4112f2e345796c`；[pytest日志](../outputs/directional_derivative_native_qa_20261029_vpffywr0/pytest.log) SHA `1512342149af0e68fbc2c80e9d90e3723d7a1ba26ab3509b05281e6fcacc2295`。原tool session17239实际exit0；503 passed、2条既有SLSQP边界裁剪warning，52.61秒。Python3.11.15/NumPy1.26.4/SciPy1.15.3；CUDA/LD为空、禁bytecode及五项单线程环境，62来源前后不变，20测试模块。

其中旧覆盖425项、新core13项、独立audit24项、执行接线41项。新增覆盖解析与量化响应、非法单侧、零实际位移、cache/末尾重复分开计费、回调污染后重新复制初态、部分复算/漏查询拒绝、完整非最小安全行篡改拒绝、历史audit JSON往返及错误来源/QA/生命周期拒绝。测试只提供组件证据，不等同实际控制或任务成功。

全仓库Ruff仍有既有I001：`scripts/check_gripper_release_floor_pair.py:50`、`scripts/check_tracking_correct_reference_v2.py:50`、`scripts/validate_temporal_response.py:3`；全仓库format仍指出`scripts/audit_rotation_goal_precision.py`一文件需格式化（2218文件已符合）。这四个已跟踪文件与HEAD一致，本轮未改冻结实验源码。新诊断及完整62源依赖范围的Ruff/format准入已通过，后续CLI准入终态另记。

本诊断最多解释局部导数一致性，不证明安全下降方向存在或二阶修正有效。即使偶然出现完整witness，也须完整独立核验并另行登记物理pilot；不据本项提升控制或任务资格。

## 保留的v1准入失败与隔离修正

[失败准入](../outputs/directional_derivative_admission_20261029/result.json) SHA `56827f760b11d25f913d3993d7441daa7fc068f94b217c52d62f6cd6e48dfbda`，原session82379实际exit1；Ruff/format各exit0，producer dry exit1，checker/owner dry尚未执行。三个实际子进程均wait/reap，无信号或重试。失败在[producer:178](../scripts/probe_tracking_v5_directional_derivative.py#L178)的warm裁剪字节检查，早于模型构造；新响应、查询、物理、模型、fresh全部0，原输出目录未建立。

只读保存值检查确认：anchor1819首行第1/2轴距严格ratebox分别超出`1.1102230246251565e-16`/`7.216449660063518e-16`，最大相邻rate为`0.25000000000000067`。这些仍符合本计划原1e-12命令容差。直接复用旧solver warm的零变化clip合同是入口错误；不能以裁剪改变原anchor来通过它，也不能据503项合成QA宣称实际准入成功。

v1源码、QA、失败准入及日志均保留原字节。v2入口仅替换为完整四组box/rate原slack审查，拒绝超过原容差的命令且不裁剪，源与输出独立。checker只改变入口/evidence导入，完整数值重放与差分审计复用冻结代码；不放宽原joint/pose标签。修正增加6来源和1测试模块，后续须重新完成68来源/21模块QA与实际dry才可运行唯一诊断。

## 已关闭的v2 QA与准入

[v2 QA](../outputs/directional_derivative_v2_native_qa_20261029_yuinc_wl/result.json) SHA `8499d64a179c44c0b6364e588cd9d64c87b3d981aec3d07654cf898d36a5673a`，[日志](../outputs/directional_derivative_v2_native_qa_20261029_yuinc_wl/pytest.log) SHA `c3070191f86759d2d07326f9fc3665922a53ca9ba1663ab9064370925e539b4d`。原session50656实际exit0，529 passed、2条既有warning，55.96秒；68源码前后不变，21模块。新增26项验证ULP内及5e-13的rate超差仍按原1e-12容差保留原命令与signed-zero字节，超容差在首次查询前拒绝；计账、cache、owner生命周期和完整QA绑定保持。

[v2实际准入](../outputs/directional_derivative_v2_admission_20261029/result.json) SHA `d79a36473d979601ed79f9444416629e78210ff116e016f2018d6caa2296c44c`，原session8783实际exit0。61份Python来源Ruff/format及producer/checker/diagnose-owner三个实际dry均wait/reap0，无信号或自动重试；68源码不变，producer/checker各2348引用、owner2350引用，新响应/查询/物理/模型/fresh0。

唯一v2诊断通过同一QA后启动于session18894，固定600秒子进程上限、完整80请求和末尾repeat合同。此时尚无producer正常终态或独立复算结果；不从启动成功推断导数趋势、完整witness或控制收益。实现提交`fdd2bb1`，此前v1实现`d254a05`、失败及修正计划`092c71c`均保留。

## 已关闭的唯一诊断与独立重放启动

原session18894已实际terminal exit0；[producer](../outputs/tracking_v5_directional_derivative_v2_20261029/verified.json) SHA `a626d66c66b4fba7dd0509092c1067c615b69cfe8545bba42e6f2b4d8d13951c`，[owner](../outputs/tracking_v5_directional_derivative_v2_execution_20261029/result.json) SHA `ebf9d63e7612e939370ce40ebfe3e698a2996912c167db0e5f08686f0564dbe6`。原child182137/start616269982正常wait0/reap，无信号或重试。

80计划请求全部记录：76次首次计算、3次command拒绝、1次强制anchor repeat，无grid cache命中；77完整调用/76唯一命令、23100数值内步、24025查询、0优化迭代。完整H12 witness仍0。producer保存的初末anchor数值比较exact=true、最大差0，历史trace/anchor-audit比较也exact=true；这些是已关闭producer的保存结果，完整独立核验尚待结束，不能先据此作导数机制结论。

[check-owner实际准入](../outputs/directional_derivative_v2_check_admission_20261029/result.json) SHA `48adb93eeb94f92d15789dd6fba8fe146d76893b9d4912af7b4f307eb4914d33`，原session30399实际exit0，2356引用、68源码不变，新响应0。完整checker owner已按唯一登记命令启动于session7602，必须核验全部77调用，包括末尾重复；1800秒上限、无自动重试。新增物理/模型/fresh/参考消费仍0。

## 保留的v2独立checker失败与check-only v3登记

原session7602实际terminal exit1。[失败checker](../outputs/tracking_v5_directional_derivative_v2_check_20261029/verified.json) SHA `f764512e07864a8dc89774059c43ed71ac76dca5914ad4531fa8a132b2ee5aab`、[失败owner](../outputs/tracking_v5_directional_derivative_v2_check_execution_20261029/result.json) SHA `feebe6faa1ff11491e94e8ed2c32b15573f5cf2c9dfe18517f392a71069bf852`。child205307/start616309119已实际wait1/reap，无信号或自动重试；失败owner无正常owned_command字段，生命周期通过原capture launch/exit/release直接绑定，不能伪造成功owner。

全部77完整响应、23100内步、24025查询及固定网格代数已核验，严格分类分歧为空；随后在[原audit:437](../scripts/directional_derivative_audit.py#L437)因`Anchor inside old stage bounds`退出。该检查错误要求零容差，而原anchor仍在已登记1e-12容差内。历史anchor审计、JIT签名及最终来源闭合未到达，因此complete/passed=false，不认定完整独立核验通过。

保留旧source与全部失败证据；[新计划](../plans/v0.0.1/directional_derivative_check_v3_20261029.md)只增加checker/helper/owner/evidence及测试，共7来源、75来源QA/23模块。新helper唯一语义变化为anchor bounds使用原±1e-12，不改FD符号、命令字节、响应绑定、差分公式或控制门槛。只重放已经关闭的同一v2 producer全部77调用，不重跑生产或新采样；原68来源QA与新75来源QA分开绑定，68摘要须完整相同。尚待新QA和实际dry通过后执行一次1800秒上限全量复核，原v2部分复算不能替代新完整复核。新增物理/模型/fresh/参考消费均0，实际仍cursor8、task/grasp0。

## v3正式QA、实际准入与全量重放启动

实现提交`6d34b60`，失败及修正计划提交`2be7c54`。[原生QA](../outputs/directional_derivative_check_v3_native_qa_20261029_30du0rxo/result.json) SHA `73356f8cc6d75037ab09daa592ab5f21ab28218084dcb442717087fc65332da2`，[日志](../outputs/directional_derivative_check_v3_native_qa_20261029_30du0rxo/pytest.log) SHA `c3608333fcd58c0276121f62666a505f55517edb2ad23a3db172d843cdf5a962`。原session87070实际exit0，573 passed、2条既有warning，82.64秒；75源码和driver前后不变、23测试模块，Python3.11.15/NumPy1.26.4/SciPy1.15.3，空CUDA/LD及单线程。新增44项为12项stencil容差/篡改检查与32项check-only来源/实际capture/部分结果拒绝检查。

[实际准入](../outputs/directional_derivative_check_v3_admission_20261029/result.json) SHA `d6ec3cc88a30cf2736536455627a36730edc41918f30e154a7f2a7970c559b58`，原session6229实际exit0；完整QA来源内Python的Ruff/format、checker dry、check-only owner dry四项均actual wait/reap0，无信号或重试。checker核2371引用；旧68来源仍是新75来源同摘要子集。新增响应、查询、物理、模型、fresh均0。

唯一v3完整复核owner已启动于session43994，固定1800秒子进程上限、77完整调用（含末尾repeat），保留原v2 producer与其原QA；没有重跑生产。此处仅记录启动，完整stencil/历史anchor、来源及wait/reap仍待正常结束。实际任务仍cursor8、task/grasp0，不将573组件测试或原部分数值复算升级为控制成功。

## v3全部独立核验和来源闭合

原session43994实际terminal exit0。[checker](../outputs/tracking_v5_directional_derivative_v3_check_20261029/verified.json) SHA `74a7e5b8b0ee04b93eb8aa4c90a178418f377c55db1a6e606eddd9d327a2b0cf`、[owner](../outputs/tracking_v5_directional_derivative_v3_check_execution_20261029/result.json) SHA `e57c508a5f48804a4763e40b53f69965151108e1afe28bd09f4b52c347e6e439`。child280730/start616431799已actual wait0/reap，无信号或重试。77完整调用、76唯一命令、23100数值内步、24025查询均对应；严格分类分歧0，旧72轴FD、三个共享stencil、3个单位方向预测、80请求预测与历史anchor比较全部通过。

数值比较最大差约`1.407e-12`；native JIT与纯Python目标计算另记最大差`1.192e-7`、695处差异，不用不同算术重放值冒充原精度FD。历史trace/audit与新anchor的canonical比较exact=true、maxdiff0，末尾重复同样完全一致；这些不保证近邻响应光滑。

[最终来源与生命周期闭合](../outputs/directional_derivative_v3_final_closure_20261029.json) SHA `28b5d3b4a7e107ea1ed5e752927b142904a0a5f7bdf1d5bb9dba64ac2a3ca108`，原session34806实际exit0，2381引用前后不变。producer原68来源/21模块QA与checker新75来源/23模块QA分别绑定，原68摘要全相同；v1准入失败、v2独立checker失败及其实际退出证据均保留为失败，没有改写为通过。

## 全固定网格的保存数据分析

[完整分析](../outputs/directional_derivative_v3_analysis_20261029/result.json) SHA `717c5d4758ada20cdaeddf6ec7a120ff42a065fa19693333666afcb84524d4f0`，原session23100实际exit0；先读取并重哈希完整闭合来源，再只对保存值计算，不新增响应。全80请求中75合法grid、3非法grid、初末anchor各1；forward/central可用36对，backward39对。非法正侧来自stage25的h=.01、stage26的h=.01/.005，原因包含rate上下界；没有裁剪或补采样。全部78个grid请求实际位移非零；初末anchor请求位移为零。

终端行明确为`terminal[0] = -final predicted joint_velocity[6]`，完整安全向量索引4368；anchor值`1.320428817e-8`。旧72轴单边FD给三个单位方向的预测分别为`-3.20428818e-7`、`-6.40857639e-7`、`-1.28171513e-6`。以下列出全部13个登记步长的实际前向/中心终端差分；`—`表示该正侧命令非法，不能计算。分母仍为原h/2h，不按实际位移重新归一化。

| h | forward24 | forward25 | forward26 | central24 | central25 | central26 |
| --- | --- | --- | --- | --- | --- | --- |
| 0.01 | -0.00064834 | — | — | 0.00010027 | — | — |
| 0.005 | -0.00073589 | -0.00073401 | — | 8.1656e-05 | 4.8423e-05 | — |
| 0.001 | -0.00080295 | -0.00082376 | -0.00071191 | -5.5272e-05 | -9.0611e-05 | -0.00015809 |
| 0.0005 | -0.00080545 | -0.00083052 | -0.00070538 | -0.00021982 | -0.00026346 | -0.00031909 |
| 0.0001 | -0.00087773 | -0.00087703 | -0.0007428 | -0.0008401 | -0.00082781 | -0.00070144 |
| 5e-05 | -0.00095498 | -0.0008251 | -0.00087661 | -0.00082463 | -0.00068537 | -0.00075629 |
| 1e-05 | -0.0014834 | -0.001619 | -0.0012248 | -0.001062 | -0.0010936 | -0.0008382 |
| 5e-06 | -0.002027 | -0.0024574 | -0.0020766 | -0.00085792 | -0.0013948 | -0.0013621 |
| 1e-06 | -0.0026614 | -0.0039427 | -0.0042162 | -0.00095529 | -0.0013798 | -0.00070074 |
| 5e-07 | -0.008485 | -0.0064429 | -0.0088336 | 0.002605 | 0.00067629 | -0.0058113 |
| 1e-07 | 0.0059521 | -0.053276 | -0.0088876 | 0.013526 | -0.031918 | 0.025074 |
| 5e-08 | -0.15453 | -0.06135 | -0.070139 | -0.061839 | -0.0082098 | -0.007905 |
| 1e-08 | -0.21054 | -0.27804 | -0.31359 | 0.032093 | 0.00055214 | 0.070787 |

在h=1e-3与5e-4，三个方向forward约为−8.0e-4、−8.3e-4、−7.1e-4，明显偏离旧模型近零预测。两尺度接近只说明这个有限范围的正侧割线一致性较好，不能证明极限导数或足以替换全部Jacobian。实际命令位移误差最大约1.11e-16；旧J乘实际位移/h与J*u最大差约1.12e-9，不能解释这里约8e-4的缺口。

12个负侧大步长请求（三方向各h=.01/.005/.001/.0005）都在period9出现一个torque clipped flag变化，clipped count从10降至9；所有合法正侧与h≤1e-4的两侧pattern均不变。因此大步长两侧不对称与该切换相伴，不能据此认定它是全部偏差的唯一原因。h=1e-4时没有这个flag变化，但中心终端差分仍约−7e-4至−8e-4；更小h的差分又明显波动或变号，中心差分也未显示收敛。目标pose项的差分同样在小h波动，不可只看terminal行选择有利步长。

原生目标含float32舍入（[实现:55](../src/s2a/native_goal_joint_response.py#L55)），但当前保存值并未隔离其因果贡献，也没有记录SVD秩/谱。末尾重复完全一致排除了本次anchor的可见漂移，不排除确定性舍入。三个单位方向余弦约.978–.993但L∞距离可达.513，不能把不同stage当作同方向纯半径对照；现有方向覆盖也不足以验证新Jacobian。

全部75个合法grid响应的cursor9最大归一化平方误差仍大于1（范围约1.31649–1.31724），没有完整H12解；单独terminal恢复buffer不能升级控制资格。下一步需用预先固定且未参与拟合的方向，验证对原线性模型的有限修正是否改善预测，然后才能据完整原caps/安全审计尝试新方向；不能直接采用更小FD、简单中心差分、放宽buffer或追加原搜索预算。

[全网格PNG](../outputs/directional_derivative_v3_plot_20261029/directional_derivatives.png) / [PDF](../outputs/directional_derivative_v3_plot_20261029/directional_derivatives.pdf) 包含3方向、3项指标和全部13步长，缺失项保持空缺。图表[receipt](../outputs/directional_derivative_v3_plot_20261029/result.json) SHA `dbd3d65bec6cb9d59f83a515cfd8a0a8fac88c2300fd7a8396c875121f92ed76`，原session23687实际exit0；只读取已闭合分析并绑定其SHA，不提供新的响应或控制证据。新增物理/模型/fresh/参考消费仍0，原总目标未完成。
