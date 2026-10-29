# 037：035未完成运行后的独立桥接资格 — 2026-10-23

本项服务于[036总目标](036_project_goal_20261023.md)，在任何新模型初始化之前登记。用户已明确要求持续推进总目标；本次执行035已有的真实桥接资格，不启动训练或最终性能验收。

## 旧运行与接续理由

旧目录`outputs/wam_stability_20261021/clean_bridge_qualification/`已有真实launch和初始化日志，但没有qualification结果、独立核验、actual wait/reap或GPU释放收据。记录的worker791403与owner791277在本次实际`/proc`检查中均不存在；允许UUID的显存/利用率均为0、compute apps为空。它不是仍在等待的活任务，也不能补造正常退出。终止原因及旧运行实际消耗的完整计算量仍未知。[现状核查:33](../../reports/status_review_20261023.md#L33)

旧目录、配置、准备记录及source快照全部保留。本项另设`outputs/wam_stability_20261021/clean_bridge_qualification_20261023/`，仅对新运行限制一次初始化、7完整请求、245 denoise，不把旧初始化尝试从历史消耗中删除，不解释成旧预算内无缝恢复或自动重试。旧运行只读审计另存`outputs/clean_bridge_continuation_20261023/old_run_audit.json`。

## 不变的数值和来源合同

完全沿用[035](035_clean_bridge_qualification_20261022.md)的三份已登记开发输入、顺序、seed42、raw/zero/固定非零bias、8400非EMA、SDPA、无compile、H40/20Hz、35步、shift5、guidance1及active10 mask。请求仅当前PNG＋任务，图像和历史response的SHA必须保持。新运行为3raw＋3zero＋1非零，各active维clean bias仍为`1e-5`；0fit、0physics、0fresh、0future读取。

35节点均保存实际FP32 sampler sample与实际sigma，clean定义仍为FP64 `C0=sample−sigma*raw_velocity`，真实返回`v=raw_velocity−residual/sigma`后按原dtype舍入。独立检查raw↔zero逐位相同、vision/inactive不变、至少一个非零active实际返回值改变及scheduler消费来源；全部3raw还须与原028/029完整128维输出精确回归。不因失败调bias、样本、采样参数或历史门槛。

外部Cosmos与模型环境只读；从已登记clean commit `8ec9484b9a6dadb9467b990ae89e80d604d47224`再归档到新S2A output。原生mask修订与S2A wrapper均显式登记，不能冒称与旧26来源完全相同。163权重文件核size/mtime，三份小metadata重新SHA；不重新全量hash大权重，不把stat一致写成内容完整验证。

## 执行入口和源码修订

新配置为`configs/wam_clean_bridge_20261023.json`，仅output_dir及对应cosmos_source_root改变，其余035配置逐字段相同。模型仅用`GPU-ad74a609-3a39-43cb-1392-24cbd279ce7f`；CPU owner/checker显式空CUDA/LD。借用原Python3.13.14解释器，不修改共享环境，所有缓存与产物在新output。

`run_wam_clean_bridge.py`新增显式`--config/--plan`准备入口，prepared将实际配置和计划以SHA引用绑定并纳入source快照；执行及独立checker使用prepared绑定的配置。旧默认入口保留，新配置不得回写旧目录或绕过原数值/预算/GPU合同。新准备前完成相关CPU测试与Ruff，准备后逐SHA复核实际源码、输入及快照，运行期间不修改已冻结源码。资格worker和独立checker不改变其计算/判定算法。

新模型初始化之前只读预检旧任务确实缺失、允许UUID空闲及端口31335可用。owner保持真实Popen句柄直至actual wait，worker7200秒、独立checker1800秒上限沿用；无自动重试，仅清理自身child。保留工具session并轮询真实handle，不因一次观察超时启动副本。正常退出和显存释放后才做独立CPU核验；任何错误保存False与原始日志，不伪记成功。

## 完成条件与下一步

新worker、独立数学checker及owner均完整，数值/预算/来源/进程/GPU合同通过且`bridge_qualified=True`，才获得该桥接资格。内部证据完整与历史输出等价性分列；如历史输出不同，即便内部审计完整仍为资格False。

本项`model_training_authorized=False`、`fresh_performance_authorized=False`。034数据资格与本项桥接资格都通过后，再登记有限真实clean残差训练、完整生成和开发任务协议。总目标仍为真实WAM＋固定S2A在原预留12新初态×seed42/43的24次运行中≥20/24稳定、各任务≥4/8、末40端点原生连续成功；本项本身没有任务收益证据。
