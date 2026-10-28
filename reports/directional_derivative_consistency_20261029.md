# 固定锚点方向导数一致性：CPU诊断

v1实际producer dry在创建模型前因复用了额外warm裁剪守卫而失败；[隔离v2修正](../plans/v0.0.1/directional_derivative_v2_admission_20261029.md)保持原命令字节及登记的1e-12容差，529项原生CPU QA及全部实际dry已通过。唯一v2诊断已正常关闭：77完整调用、76唯一命令、3条非法请求，完整witness0；完整独立重放已启动，尚无其终态。实际仍cursor8、task/grasp0，GPU禁用，原18正确参考/12WAM开发/24最终留出门槛保持。

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
