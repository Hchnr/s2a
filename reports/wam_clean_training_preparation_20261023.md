# 038有限训练实现与准备 — 2026-10-23

后续进展见[040执行准入与D0启动](wam_clean_training_execution_20261023.md)：完整入口/独立执行审查已通过，原63位seed不兼容问题已在运行前显式修订，D0已启动但未提供完成资格；以下保留此前metadata准备与256项软件QA快照，不回写其原收据。

**训练清单和独立metadata审查已通过，执行尚未就绪；没有启动038模型、真实标签解码、候选拟合或物理任务。** [037＋039真实桥接联合资格](wam_clean_bridge_20261023.md)已建立，本阶段推进[038预注册两轮协议](../plans/v0.0.1/038_clean_residual_training_20261023.md)的实现；总目标及最终20/24、各任务4/8门槛不变。[036:7](../plans/v0.0.1/036_project_goal_20261023.md#L7)

## 已完成的真实准备

实际metadata prepare正常exit0，固定30条034同步native来源、每条5锚点共150张当前PNG；每锚点root101/102经事前固定SHA算法派生，共300个互不重复且排除42/43的63位训练seed。D0/D1复用同一清单。完整排除六validation整demo、历史禁入demo及fresh真实初态SHA，没有伪造旧validation初态同步身份。[准备入口:532](../scripts/prepare_wam_clean_training.py#L532)

validation仍为已查看的六EP、每EP四锚点，共24张当前PNG；seed42/43×raw/W2形成固定96生成矩阵，另三zero共99。本次只绑定元数据及opaque文件SHA，36个未来标签归档（30训练＋6validation）未通过准备路径解码。该边界来自冻结源码及读入记录的审查，没有历史系统调用trace，不能声称已独立核验这些标签的几何或数值；真正标签审查须在相应整轮采样正常退出后进行。

准备保存14份源码/配置/计划及逐字节快照、1683输入SHA。独立stdlib checker完整重建资格链、150/300/24/96、来源排除、当前PNG头、opaque归档字节及源码指纹；1707个不同声明文件SHA前后不变，`metadata_qualified=True`、`execution_ready=False`。另一份独立peer核1710个实际消费项并得出同样metadata判断；数量差来自其额外消费独立审查代码及引用，不是训练样本分母变化。[独立checker](../scripts/verify_wam_clean_training_preparation.py)、[peer v2](../outputs/clean_training_prepare_independent_20261023/audit_v2.json)

| 收据 | SHA256 |
| --- | --- |
| [原metadata prepared](../outputs/wam_stability_20261021/clean_residual_training_20261023/preparation/prepared.json) | `b55c214ffec328ee94f2074007c15dfe62486512aeb660b251fdb21f38e970f7` |
| [独立metadata verified](../outputs/wam_stability_20261021/clean_residual_training_20261023/aux/independent_metadata/verified.json) | `eb51208123f6a3bd3f63fef19d44d37de62c52459431cbfcd7f27dbb630b8570` |
| [独立peer v2](../outputs/clean_training_prepare_independent_20261023/audit_v2.json) | `b5198aea95ad8e3722e2f10c05e043e46ba3309badfd651da7f10bc68597f631` |

034原owner False、037原checker/owner False及039补充联合资格均精确引用并保留；metadata准备未重写历史结果。首次prepare因误用peer字段名在创建038输出前退出1，现场保留并仅修正新入口；新准备随后正常exit0。peer v1因并行实现新增入口，误将“冻结时缺8入口”等同“审计时仍缺8入口”而False；v2分列冻结readiness与当前presence，保留v1。源码存在不构成执行资格。[准备工具收据](../outputs/clean_training_prepare_qa_20261023/completed_tool_receipt.json)、[首失败](../outputs/clean_training_prepare_qa_20261023/first_attempt.json)

metadata准备及该checker重新核163权重stat，未重新核三项小checkpoint metadata内容SHA和037原snapshot/input全闭包。新的运行准入已增加这些字节检查，仍须在最终执行注册时实际闭环；不将stat说成完整大权重哈希。[运行准入:91](../src/s2a/clean_training_runtime.py#L91)

## 软件实现及证据边界

| 新实现 | 已有验证 | 未提供的证据 |
| --- | --- | --- |
| 当前请求、300 seed、来源/行身份、41实测端点→40标签及FP32 normalization原语 | 数据原语25项CPU；真实300当前请求契约检查禁止`np.load`后通过 | 未读取真实034 future，未独立核实际标签数值 |
| 精简capture sink | 29项CPU；复用037第一raw完整35点，1400行C0误差0、消费SHA前后相同 | 0新增采样；省略video只保留SHA/在线witness，不能离线逐值重建 |
| 独立采集checker | 53项CPU合成节点/完整35点/初始化/生命周期及拒错；真实300当前请求合同检查通过 | 真实整轮10500点待采集；尚无新模型正常退出/释放收据 |
| 独立流式增广QR | 34项CPU，包含完整AugLS黄金、两遍完整SHA及重分块/病态拒错 | 未做真实1290列全量W1/W2重建 |
| D0/D1采样入口、请求/执行准入、四阶段owner | 49项CPU；真实CPU child正常wait/reap及exit7失败收据保留 | 未加载模型，未证明300请求实际采集或显存/磁盘运行结果 |
| W1/W2生产fit与独立fit checker | 19项CPU合成数据/完整35节点与W2等权、两种独立求解、来源/生命周期拒错 | 0真实candidate fit，0训练收益 |

采集复用037已资格的`ConversionWitness`，每个请求只传当前PNG及固定任务语言，D0 raw、D1仅绑定已独立核验的W1。D0/D1完整初始noise、mask、当前latent、caption和tokens逐位配对；每个35节点留存所有训练action数值和hidden，目标始终使用原raw velocity的C0。每轮10500次denoise、300请求，owner保留真实Popen至wait/reap；CPU拟合只在完整capture独立审计与GPU释放后才能打开标签。[采集:46](../scripts/collect_wam_clean_training.py#L46)、[owner:66](../scripts/run_wam_clean_training.py#L66)、[fit合同](../src/s2a/clean_training_fit_contract.py)

执行准入必须另行提供最终source closure、独立metadata审查身份、完整阶段/精确producer引用及无重试合同；原b55c清单永久保持`ready_for_execution=False`。最终注册还须纳入后来新增的capture checker、owner、runtime及实际helper，不能只核原8个入口是否存在。[49项QA](../outputs/clean_training_collection_qa_20261023/qa_v2.json)、[独立准备QA](../outputs/clean_training_prepare_qa_20261023/independent_verifier_qa.json)

整合复核发现初版collector错误用domain字符串索引实际list型transforms，已在模型执行前修正为normalization31并核domain/projection0。新增实际列表合同测试后49项通过，再对登记8400的真实config.yaml作安全YAML CPU检查：67个domain/transform、31对应LIBERO/0、128维参数及原metadata SHA全部相符，未导入模型或读取future。旧整合发现保留，新修复检查单列。[原发现](../outputs/clean_training_fit_20261023/integration_review.json)、[实际配置修复核验](../outputs/clean_training_fit_20261023/fix_verification.json)

全部九个相关模块最终合并验证**256 passed in27.04s**，全仓Ruff check与format check通过（437个Python文件）。这是本阶段软件QA，不是全仓pytest均通过；历史迁移测试失败边界继续保留。[完整命令、源指纹及工具退出收据](../outputs/clean_training_final_qa_20261023/qa.json)

## 下一项实际缺口

控制复用只读预检已完成：原fixture、三旧init0的expert文件、007/015关联控制源码及六条实际记录的8项包版本SHA/声明均相符，297个XML资源可定位。当前simulation Python为3.11.15，与历史声明一致；原真实patch/binary身份未单独记录，无法追证。新CPU probe采用034既有cv2预加载/精确来源核验/恢复LD边界，再导入九项实际控制依赖，CUDA/LD保持空且未初始化环境。`control_contract_reuse_eligible=True`有条件成立：新dev worker必须显式接入并冻结该guard及来源；`new_physics_control_qualification=False`，没有新增正控成绩。[补充收据](../outputs/clean_training_development_preflight_20261023/dependency_addendum.json)

原广义包导入probe的False保留：PyArrow26与NumPy1.26的额外导入不兼容，但该组合与历史记录一致，控制路径只读distribution metadata且实际依赖导入不加载pyarrow。首诊断将其称作必然执行阻断不成立，补充已纠正；另发现cv2实际改写LD的副作用，已用上述既有guard实证处理。不安装或更改依赖，也不将额外包probe False改写为True。[原False](../outputs/clean_training_development_preflight_20261023/verdict.json)

尚缺唯一W2的96＋3完整生成与独立state诊断入口、冻结参数的开发私有service、raw6/W2六的物理执行与独立seed/因果计分，以及覆盖这些入口的最终执行准备注册。最终准备还须绑定guard、冻结完整执行参数并核实际磁盘/RAM/服务资源；不能由入口存在或历史24/24直接推出本轮运行成功。

以上闭环后才按已登记预算串行D0→W1/独立QR→D1→W2/独立QR→99生成诊断→12开发任务。state gate与任务稳定门分别报告；最终预留12新初态×42/43的24条性能验收保持未运行。没有新增teacher denoise、future RGB、fresh任务或真机证据。
