# 最终留出验收接入的隔离实现草稿 — 2026-10-23

本项落实[036总目标](036_project_goal_20261023.md)中尚缺的最终接入，不改变原12个新初态、20/24及每任务4/8稳定成功门槛。**当前是实现草稿，不是执行登记或fresh准入。** 041固定训练正在运行，其484源码、快照、输入与已发布bundle保持不变；新实现暂放本仓库`outputs/wam_fresh_acceptance_implementation_20261023/repo`内的隔离源码副本，不导入当前worker，也不向主源码提升。

## 边界与数据

隔离副本基于041已准入的473份主源码/显式配置计划，逐文件来源和SHA保存在[source_basis](../../outputs/wam_fresh_acceptance_implementation_20261023/source_basis.json)。11份041执行bundle不复制为新实现来源。原main、外部Cosmos、发布数据和checkpoint不改，CPU工作空CUDA/LD、三线程8；没有新的模型初始化、物理步骤、fresh性能或未来标签数组解码。

只读原reservation的12条JSON metadata建立[矩阵草稿](../../outputs/wam_fresh_acceptance_implementation_20261023/draft_matrix.json)，fixture SHA为`6e279c4db07a06f7399f53eb298cb77b671305accfe5b520598f1ee1c8e134c6`。它保留原source/demo/initial身份、两sampling seed42/43和3任务各4初态；不重选新样本，不读取expert数组或现有future状态。新draft的`bindings_complete/execution_authorized/fresh_performance_authorized/candidate_selected`均False，没有虚构candidate/readout SHA。

## 条件性完整矩阵

拟定顺序native24 → oracle24 → 唯一W2候选24；每source按task0/1/2 → initial0/1/2/3 → seed42/43。全部每条600个20Hz端点，最后40端点连续原生success才算稳定，成功不提前停止。模型推理期间物理暂停，实时与真机另验。正控失败保留原分母和未执行项，不换初态，不用不同seed或来源补齐；不完整矩阵不能输出最终成功率。

条件性预算算术为72条、43200物理步：native/oracle各14400步、WAM14400步；WAM每条75请求、每请求35个正sigma模型节点，共1800请求/63000 denoise，最多一个新模型初始化，0fit、0自动重试。这是供后续冻结的预算草稿，目前不授权任何该项调用；也不并入041已准入预算或追写旧结果。实际执行前另行闭合所有候选、源码、接口、资源、总累计预算与独立审查。

唯一候选沿038/041为实际W2。开发准入仍严格沿原038：完整来源/数值/控制/生命周期合同以及W2至少5/6稳定、每任务至少1/2；`state_gate`和严格适配收益另列，不额外要求它们True。收益False不得声称适配优于new raw。候选及同一控制/采样参数在看到fresh性能前冻结，不能选W1、按task/seed拼策略或反复用最终集合调参。[038固定门](038_clean_residual_training_20261023.md#L112)

## 必须完成的接入

1. **首当前相机见证。** 不改冻结position executor字节，以进程局部context hooks调用原renderer/generator各一次，保存首query的两张实际current observable uint8RGB、渲染约定和完整physics快照，独立核fixture initial及render/推理前后physics/time不变；绑定实际`input_0000.png`与`request_0000.json`。独立checker自行完成双轴flip、torch bilinear cover-crop、两图拼接和4px seam，逐像素核PNG，不import生产canvas/witness验证逻辑，也不使用旧003/006图像或gold输出冒作fresh来源。
2. **同record、同sampling seed的native→oracle。** 仅接受唯一`record_sseed`目录，拒绝无seed歧义及其它seed fallback。先核4份native产物SHA、完整600个原生boolean/20Hz、record/seed、实际初态与独立normal-owner资格，再解码该新native reference；逐端点与`after_source/opening_after`精确核对，保持world/source-eef/SO(3)/m/after-step单位合同。不得要求原native结果包含它未产生的wait或complete字段，不改写原结果；新增独立qualification引用实际launch/exit/release并核birth、实际wait0/reaped、无signal、CPU环境。
3. **最终专用runtime。** 原开发service只接受12个旧raw/W2 case，不可直接改metadata声称支持fresh24。须完成24个新W2-only case绑定、逐请求current PNG/语言/seed/domain/mask、readout及全部初始条件、35-node计数和独立数学审计；仍只使用明确允许UUID，单模型串行、失败停止。
4. **实际native与oracle owner/独立审查。** 新native wrapper保留既有控制及原产物字节，另外记录实际初态和source guard；必须由真实Popen持有、正常wait/reap后发布独立reference qualification。oracle使用同一冻结S2A、精确同seed native参考。首相机或oracle纯CPU测试不能替代这72条实际仿真。
5. **最终计分与跟踪。** 固定完整72矩阵，24个WAM试验按12个初态报告成对seed及不确定性；同初态两seed不能当作独立场景。分别保留ever/terminal/stable和全部失败；独立重算实际desired→after TCP/SO(3)/opening误差、限幅、dt与因果请求。已有正确参考资格10mm/0.05rad仅证明其所测范围，不能据此认证任意新生成轨迹。最终数值跟踪准入和oracle范围在fresh性能前显式冻结，不由最终结果倒推；单独任务计数工具不宣称可靠跟踪、来源或执行资格。
6. **最终执行准入。** 待实际W2、state99、开发12全资格存在后，补齐candidate/readout、控制依赖、source/snapshot、资源及owner入口，独立验证新proposal，再发布admitted，先真实dry再唯一启动。当前缺这些实物，不能用草稿字段或CPU测试绕过。

## 当前实施与未完成

隔离`src/s2a/fresh_matrix.py`已实现固定72身份、预算、完整600原生boolean、24固定分母、native/oracle正控计数和12初态分组，18项CPU测试与Ruff通过。[QA](../../outputs/wam_fresh_acceptance_implementation_20261023/qa_matrix_v1/qa.json)；该函数明确仅做统计，不授予来源/执行资格。

首相机producer、独立camera checker及严格oracle reader已在隔离副本完成，root组合205项相关CPU检查及Ruff通过，fake physics/HTTP配合实际原generate的独立联调通过。跨实际Python/Torch环境的像素一致性另核，不将同环境单测代替。最终native wrapper/qualification、W2-only service、全矩阵独立执行审查和实际候选均未完成，fresh仍0运行。[实现与边界](../../reports/wam_fresh_acceptance_implementation_20261023.md)。当前训练进程按041继续，不因本项开发而重启、扩大预算或修改源码。

后续已新增native包装器、独立native资格检查器及W2-only metadata请求账本，七模块组合339项CPU检查、Ruff通过。真实Popen持有两个新合成seed的原native executor/fake环境，完整600端点见证→独立checker→同seed oracle联调通过，0真实physics/model。尚无真正24case模型服务、完整owner及全72矩阵审查。041随后D0完整通过、W1前置失败停止；由043另行准备有限CPU修复，保留全部原来源和False，不因本项开发重采模型或启动fresh。上述205项和“训练继续”为此前快照。

实际环境核查发现main3.12/Torch2.12与sim3.11/Torch2.6在原10组合成输入上存在最大1的像素差，原False收据保留。使用实际sim env311中的独立checker复核同10输入及原production输出，10/10逐像素精确通过；另两个新合成fixture的完整helper/generate/NPZ/checker联调通过，仍为fake physics/HTTP。后续相机核验解释器明确为本仓库`outputs/libero_env311/bin/python`，容差固定0；[v2 metadata草稿](../../outputs/wam_fresh_acceptance_implementation_20261023/draft_matrix_v2.json)仅增加运行时及资格收据，所有执行和候选字段继续False，真实renderer及完整执行另验。
