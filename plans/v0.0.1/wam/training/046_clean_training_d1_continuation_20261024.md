# 046：显式绑定新W1的原D1采集接续

本项推进036总目标，沿用038两轮clean readout实验；不以D1采集、W2拟合或软件检查代替留出新初态稳定成功。当前为实现准备，尚未授予新模型执行资格。

## 前置条件与旧结果

045必须同时有真实W1生产与独立全量QR合格结果、两角色实际wait0/reap/no-signal及完整supervisor成功收据，且508源/快照保持SHA。已有原043失败fit及原041失败owner永远保留；不得改名覆盖，不把原失败owner替换为新成功。

原D0的300请求/10500节点与独立审查复用。W1只使用045唯一参数，不再拟合，不换ridge，不从validation/fresh选候选。原预留最终24条保持关闭。

## 接入点与实现边界

原041 bundle的`run_wam_clean_training.prior`固定读取原prepared.root的`orchestration/W1/result.json`；该文件为旧False，直接重启旧四阶段或六阶段编排会错误读取旧结果。新入口须显式读取045实际supervisor、producer及独立QR引用，验证每条来源和正常生命周期后供D1规划使用。

使用原041已冻结bundle的owner、collector及capture checker，保持文件字节不变；只允许新wrapper在进程内、有限作用域内替换owner的W1 metadata lookup，其他stage仍调用原lookup，退出必恢复。原函数的`__file__`和source proof仍指向原bundle，新增wrapper与其准入代码另固定source/snapshot。不得将045 durable exit改写成原旧schema：新验证器直接检查真实收据，不伪造旧格式收据或改写旧owner。

D1 collector原接口可显式接收`--d0-audit`、`--w1-fit`、`--w1-verification`；原`trained_head`校验原prepared/041 registration和参数schema，未将输出路径硬编码为旧W1目录。D1实际输出仍为原prepared.root的`capture/D1`，原独立审查为`review/D1_verified.json`，原owner为`orchestration/D1`，全部须不存在。新增046控制根单独保存wrapper、准入、快照和持久化supervisor收据。

## 固定预算与资源

- D1只有一次新完整采集尝试：同150训练锚点×2独立noise，300请求、35步各请求，共10500 denoise，期望head calls10500；输入当前图像，0未来标签。
- 与D0完全相同的seed、caption、文本tokens、初始噪声、condition mask及首当前图像latent逐字节核验。唯一变化是既定W1 clean readout；不可从GT构造在线输入。
- 只允许`CUDA_VISIBLE_DEVICES=GPU-ad74a609-3a39-43cb-1392-24cbd279ce7f`；实际单设备UUID核验，禁止数字映射。独立metadata/审查及owner使用空CUDA/LD；线程8。外部Cosmos、权重及发布数据只读。
- 最多1次新model initialization、300完整请求/10500 denoise。失败停止，无自动重试；0新fit/QR/physics/fresh。W2不由本项启动。
- 继承原D1采集、独立审查、退出后120秒GPU释放预算；新持久化CPU supervisor必须持有实际owner Popen，owner持有实际模型Popen。工具观察超时不视为终止或重启理由。

## 开始实际采集前必须完成

1. 新wrapper有限lookup替换、恢复和错误传播实现；验证原非W1路径仍原样处理、新W1来源不可替换、任一失败收据拒绝。
2. 配置明确记录045真实参数/两角色/owner引用、原D0、预定输出、累计已发生尝试及本次预算；source/snapshot冻结。
3. 独立metadata审查及分离proposal/verified/admitted，不能以本计划或旧041 envelope单独授权新wrapper。
4. 实际admitted dry通过原完整D1规划、当前输入/seed及新W1 head合同；0模型导入/采集。GPU UUID/空闲状态、端口和新输出检查通过后，才启动唯一持久化supervisor。
5. 收尾须300请求/10500节点完整、D0/D1初始条件exact、逐节点与readout使用独立审查通过、正常wait0/reap和GPU释放；保留失败时的实际计数，禁止把不完整采集送入W2。

之后才另行固定W2（D0+D1共840000行、同1290/.1、045实际合格sim运行时和独立全量QR）。W2仍须完整生成诊断及固定开发门通过，再冻结统一候选，最后执行原24条与native/oracle正控；最终总体≥20/24、每任务≥4/8保持不变。

## 执行前实现补充

新入口为`execute_wam_clean_training_d1_handoff.py`、独立metadata入口为`verify_wam_clean_training_d1_handoff.py`、单阶段持久化入口为`run_wam_clean_training_d1_handoff.py`；可复用元数据模块在`src/s2a/clean_training_w1_handoff.py`和`clean_training_d1_admission.py`。控制根固定`outputs/clean_training_d1_continuation_20261024`，原D1输出仍保持上文路径。新source/snapshot只扩展045原508闭包。

监督上限沿原041 D1为88400秒（原模型86400秒、独立审查1800秒、释放120秒及余量）；异常清理沿原130秒，并读取实际后代pending/release与PID birth，不将工具句柄丢失当退出证明。原041已有3个记录后代在本次前均实际reaped，读核收据另存。真实CPU小子进程测试覆盖正常wait、exit7和错误结果，无模型采集；原组件dry实际调用旧owner的D1规划，在阻断`np.load`下完成，1次W1和1次原D0查找，恢复成功。初次测试命令的导入路径/临时目录配置失败、后续合成观察器缺字段的失败结果均保留；不计为模型或D1采集尝试。
