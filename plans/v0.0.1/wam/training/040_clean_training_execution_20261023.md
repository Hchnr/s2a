# 038最终执行准备与接入 — 2026-10-23

本项补齐[038](038_clean_residual_training_20261023.md)既定执行入口及独立准入，不改变训练/开发预算或最终验收目标；运行前发现的不可执行63位seed登记按下节显式修订。037完整7/245采集与039补充审计已联合通过；原False全部保留。038原metadata prepared `b55c214ffec328ee94f2074007c15dfe62486512aeb660b251fdb21f38e970f7`永久保持metadata-only，不回写为执行成功。

## 运行前seed合同修订

最终准入前只读追踪发现：038原300个SHA派生63位seed全部超出`PreparedAction128Request.seed<=2**31-1`；实际vision/action初噪声又调用冻结`misc.arch_invariant_rand`的`np.random.RandomState(seed)`，其标量seed只允许32位。因此原metadata虽完整绑定身份，却不能按原样执行；没有加载模型、消费训练采样、拟合或查看性能来作此修订。保留原b55及原63算法，不改冻结Cosmos的validator或随机数实现。

本项明确新增`execution/seed_plan.json`。令D=`s2a.clean-training-request-seed31.v1`，对原清单每条完整身份构造`json.dumps([D, anchor_name, root_seed, original_sampling_seed], ensure_ascii=True, separators=(',', ':')).encode('utf-8')`，实际seed取SHA256前4字节big-endian整数与`((1<<31)-1)`按位与。它是一个新登记的完整SHA派生规则；不用原seed的低位截断值。全部300条一次生成，要求唯一、排除42/43、每anchor仍101/102两root，碰撞/禁入即拒绝，不重抽、换行或选择seed。D0/D1使用同一实际31位seed和完整初始noise。训练/评测数目、等权和预算不变。

seed plan逐条保存index/name/anchor/root、原63位身份seed、实际31位seed和新payload SHA，并精确引用原b55。proposal、独立040 checker及admitted均绑定该ref；独立checker自行重算全部300项。模型/采集审查/fit三处admission先核完整证明链，再从原metadata生成显式effective副本：只替换training request内两处seed、training noise roster及算法说明，增加seed-plan ref，其余字段不变。原entry的canonical/identity SHA继续指原source身份，新seed的payload SHA另存plan，绝不声称旧63算法产出了新31位值。后续模型请求、初噪声、compact/trace、行身份和fit以此实际seed为准。

同次发现冻结`render_caption`按实际seed的BLAKE2b draw决定是否包含`task_plan`：draw<.9时为空[]，否则缺键；当前任务没有subtasks。原42/43恰落空[]分支，不能据此假定所有训练seed都有此键。collector与独立capture checker分别重算实际seed分支，严格核空[]/缺键及当前task字段，拒绝`current_subtask`、非空计划和额外字段；保留原caption字节/词元，不补写计划字段。STATE/dev的42/43历史exact合同不变。上述修订在最终源码/QA/proposal冻结前完成，尚无038模型调用。

## 固定执行范围

仍按D0 raw300→W1一次fit/独立全量QR→D1固定W1的同300输入/初噪声→union W2一次fit/独立QR→3zero＋96 raw/W2完整生成→raw6/W2六开发任务串行执行。总上限1599完整请求、55965 denoise、4模型初始化、2生产fit＋2独立重建、7200物理步；0新增teacher、future RGB、native重新采集、fresh性能或真机。各stage完整分母与时限不变；错误保留实际消耗，不重试、不缩减样本、不选择W1或新参数。

新增STATE owner持有唯一99生成worker至正常wait/reap，观察GPU释放后own独立CPU checker。该checker核全部3465节点、三zero历史及完整初态、48配对和实际源码/参数，再打开六validation future档案。旧validation只有发布20Hz时钟，原生初态同步仍未证，不伪造成新同步native数据。原028 state门完整诊断分列；数值/来源/因果合同完整时，state门False也按038执行一次固定开发矩阵。

开发仅原task0/1/2_init0×42/43×raw/W2，顺序task→seed→raw/W2，600步/75请求每case。一个仅localhost31339的私服由独立owner持有真实Popen，整个启动/12case/finish共用172800秒总deadline。`/bind_case`独立绑定既定case/head；`/generate`完整保留原当前PNG＋任务语言body，不增加variant、专家未来或物体信息。服务保存全部35节点精简action/hidden、完整初始noise/condition及请求/响应/trace。raw/W2首PNG与完整初始条件逐位配对，后续闭环图像可自然分歧。

新CPU wrapper先执行034已资格的精确cv2 probe/三来源核验及LD恢复，再导入冻结position executor；每个case独立环境/控制状态。仅进程局部HTTP transport将旧900秒请求上限压至300秒，不改请求body、参考或控制计算；HTTP日志保留实际timeout与75次POST。旧函数的数字GPU4标签只用于CPU兼容校验，实际GPU另按UUID记录，所有模型入口只设置`GPU-ad74a609-3a39-43cb-1392-24cbd279ce7f`，绝不以该旧标签启动任务。CPU/OSMesa仍空CUDA/LD。

正常`/finish`只在12/900完整后允许，模型退出0；`/abort`停止服务并保存False，无自动重启。独立开发checker在服务/12worker全正常退出及GPU释放后重放全部7200端点控制合同、900请求/31500节点并独立计分。生产owner True只说明生产完整，不代替独立封套或任务门。开发稳定门仍W2≥5/6、各task≥1/2；适配增益仍W2稳定数严格高于new raw，分别报告。

## 无循环SHA的执行准入

1. 最终producer创建新`execution/proposal.json`，绑定原metadata、独立metadata审计、控制复用补充、实际资源、最终源码/逐文件快照、QA、normalization31及037实际sigma trace。proposal完整可审，但`execution_authorized=False`、`readiness.ready_for_execution=False`，不能启动模型。
2. 独立`verify_wam_clean_training_execution.py`重核所有来源、矩阵/预算/资源/QA/源码快照、原163权重stat和三small metadata SHA、控制guard及全部入口后，写`execution/verified.json`，精确引用proposal SHA，给出`eligible_for_execution`。它不读future数组、不初始化模型或环境。
3. producer只在独立True后创建`execution/admitted.json`。除schema、execution_authorized、readiness.ready_for_execution及新增proposal/verification两ref外，payload必须与已审proposal逐字段一致。所有owner、worker、fit和独立checker均核此证明链与独立verifier源码SHA；不能拿任意producer True或文件存在代替资格。

最终源码范围为本仓库src/scripts/tests全部Python及本项配置/计划、038配置/计划、pyproject；每个源文件都有`execution/source/<repo-relative>`字节快照及original→snapshot ref。外部Cosmos及控制依赖只读，沿已资格的隔离源码和明确输入SHA，不向旧S2A路径写入。旧037/039及metadata14冻结来源逐项核验；本项只扩展未进入这些旧冻结清单的新执行文件。

本项将独立compact数学函数增加显式`purpose`参数：默认training不变，evaluation必须42/43且原header/point/trace保持evaluation，禁止为复用数学将其翻译成training。所有算法/舍入/来源/默认训练门不变，旧软件QA原SHA保留，新版本重新QA后才冻结。

运行前未知W1/W2参数与后续worker receipt SHA，不写假值。冻结的是完整command-building源码、固定入口与输出路径、显式stage参数规则；实际每阶段在正常前置退出/独立审计后绑定唯一已有参数/receipt SHA，保存真实argv与子进程身份。最终准备只授予这一条件流程的运行资格，不声称未来阶段已完成。

## 实际资源准入

固定要求本仓库所在filesystem可用磁盘至少**200GiB（214748364800 bytes）**，有效可用RAM至少**128GiB（137438953472 bytes）**，CPU affinity可用核至少8，BLAS/OMP三变量均8、fit最大block1400。有效RAM取`min(/proc/MemAvailable, cgroup-v2 memory.max - memory.current)`，`memory.max=max`时只取host；记录host/cgroup原值及时间，不把全机内存当作无条件容器可用量。

磁盘阈值用于完整1599请求的全部证据：已实测单请求精简节点约8.66MB，另保存完整初始FP32 noise/mask/current prefix、JSON、当前PNG、模拟端点、参数/快照与service staging；约31GB基础数组量之外保留大幅余量，不据此少存任何节点或预算。128GiB RAM为模型加载/CPU权重、每请求35个完整暂存点、精简保存及独立QR工作区留保守余量，不声称实测峰值或限流保证。实际运行继续保存实际消耗与错误。

准备实际观察允许GPU UUID、PCI/minor及memory0/apps空/利用率0，并检查31338模型master、31339私服端口可绑定。此观察不是预占；每个模型owner仍须持本仓库同UUID文件锁、启动前再次实际检查，遇占用就保留False，不停止别人的任务。原seed43控制资格的源码/fixture/依赖可条件复用，历史实际Python patch/binary无法追证的边界保留，不报告新物理正控。

所有必须测试均在最终源版本上真实执行，QA保存argv/exit/log/source SHA；源码变化须先重新验证并重新独立准备，不修改已有proposal、审计或False。全仓pytest历史迁移失败边界另列，本项相关QA不泛称全仓测试均绿。最终12新初态×42/43验收仍未授权打开性能，不在本项输出根创建fresh结果；总体≥20/24且各task≥4/8不变。
