# 054：052新头同步validation实际执行

总目标及最终原24条稳定≥20/24、各任务≥4/8不变。052真实fit/独立QR与053同步参考/native稳定正控6/6已通过；旧W2开发0/6失败、旧STATE同步未认证记录保留。054只评价新生成参考接近性，不是机器人闭环任务成功，fresh关闭。

## 实际准备与准入

124项相关CPU检查通过，包括完整配对/候选身份、分量/seed/demo保护门、未通过数值与生命周期前拒绝未来标签、既有独立point math与真实有界子进程生命周期。Ruff/format通过。新完整数值审查复用原独立capture数学检查，完整verify_generations函数只作显式身份及current-anchor列表替换；独立AST证明采样主体及数值审查主体的数学路径未改。

最终prepared_final在任何模型启动前冻结1447份源/1447快照，将新数值审查、固定评分门和owner纳入闭包；历史prepared/prepared_v2及元数据结果保留。最终新增了明确的zero qualified-response来源字段绑定，不改变样本、seed、参数或sampler数学。独立最终metadata通过；实际借用的模型Python3.13在完整登记下--dry真实exit0、0模型初始化/请求。

根目录`outputs/synchronous_state_probe052_20261024/`：

| 产物 | SHA256 |
| --- | --- |
| prepared_final.json | 54a3eb87493e19fb4533330db8a9a4ac022e38d8710c511412dc6e552684b987 |
| metadata_final_verified.json | 3ad93e203312a1a44e2cc578a3ab4bebc1786053256293786cb60c2cad07c30b |
| registration.json | 4ec2cfa91cca74f4cd5a244ef2578d4654a921e5b748800cbae479dd8438c370 |
| successful_cli_dry.json | b1315ccb1380c1e8cfa9bb59d9700ed311336ced1a117685e95a010a59048bc8 |
| supervisor/launcher_result.json | ad35d4b1e0d91c7b4d11d910769bdf3ad2524043b6cf78131ee28ca3a573f507 |

## 实际启动

启动前nvidia-smi和驱动信息交叉核实UUID `GPU-ad74a609-3a39-43cb-1392-24cbd279ce7f`为当前可见2/minor6，memory0、利用率0、compute_apps空；owner内再次核查并持有该UUID的排他锁。只使用这一允许UUID，不用其他卡。

短launcher正常exit0，持久化owner PID2658239/start_ticks578058701已实际ready，并重归PPID1；其真实模型子进程PID2658272/start_ticks578058881，PPID为该owner。launch.json记录CUDA_VISIBLE_DEVICES为指定UUID且LD空；观察时两进程均实际存活。模型日志已进入单rank初始化，尚无完整99请求或数学/性能资格。

预算固定3历史zero＋48对raw/probe052，共99请求/3465节点、最多1初始化、0fit/物理/fresh，无自动重试。模型worker只读当前清单与PNG及历史zero模型captures，不读053未来状态档案。owner保留实际Popen，正常wait/reap且观察GPU释放后才调用全数值审查；数值与生命周期通过后，另一个CPU评分进程才解码六条053未来端点。通过state gate前eligible_for_development=False，最终fresh始终不由此阶段授权。

进度以同一supervisor/model真实PID、model/command.log、state_sampling中的已闭合请求及最终owner收据为准。观察超时或暂无输出不触发重启；任何False、partial及异常收尾均保留。

## 实际终止：分派错误，原054失败保留

实际worker在首probe052前触发KeyError，因请求variant已改名而`dict(raw=None,zero=zero,w2=head)`的关键字未改；这是实现遗漏，数学主体AST等价和静态请求测试没有覆盖该运行时分派。原worker保持冻结，不修改其SHA来隐藏错误。

完整完成4请求/140 denoise与head边界调用、1次模型初始化；index0..2 zero及index3 raw complete=True，index4 probe052 complete=False且目录只有request.json，未调用model.prepare/采样。模型实际exit1、actual_wait=True、reaped=True、signal_sent=False；owner False保留。GPU实际memory0、compute_apps空，owner与模型PID均已消失，无自动重启。

新增CPU只读`audit_wam_synchronous_state_prefix.py`已独立复核全部4请求/140节点，包含105历史zero fullpoint源数组、完整FP32初始条件与response逐值回归、当前caption、actual sigma/grid、raw C0/部署及源链、每request前后base参数identity。全部source/input SHA前后不变，进程exit0。`prefix_verified.json` SHA256 `e0fda7e66b45e25d7cb446c13973b10fa4a990dd63a5af5706a2ac852ccbed58`，明确scope仅前4请求；原整轮参数after不存在，不补造，0 validation future读取。

独立分派组件`s2a.synchronous_state_dispatch.select_readout`及6项针对首适配边界的CPU回归已通过；它尚未接入新的GPU入口，原054失败仍在。后续[055有限接续](../plans/v0.0.1/055_synchronous_state_dispatch_continuation_20261024.md)固定只执行原剩余95请求/3325节点，额外1初始化、累计2，复用合格prefix并严格跨进程配对第一个probe的原raw initial。须先实现新worker/owner/联合审查和独立准入，不重采已完成4次，不改参数、seed、门槛或fresh分母。当前无模型进程或任务收益。
