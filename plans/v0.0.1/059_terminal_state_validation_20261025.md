# 059：058末节点候选的同步state验证

总目标及原预留24条稳定≥20/24、各任务≥4/8不变。058仅完成离线数值资格，参数SHA30210a2bda69962bbd24f8e86849f58c4516d2209bfb3d8a5903fc45ae543b4f。新目录outputs/terminal_state_validation_20261025，不覆盖055/056或数值失败记录。

沿用原053六demo/24同步当前锚点及seed42/43、原checkpoint/mask/H40/20Hz、35去噪节点。固定3历史zero回归＋48raw＋48terminal058，共99请求/3465节点，最多1次模型初始化；无fit/物理/fresh。节点0–33使用052，34使用058；rotation六列逐值保留052。以实际capture bridge step_index选择，不使用另计数器；raw/zero明确清空策略。不得按任务、seed或GT切换。

准备阶段先绑定原来源及新源码SHA/快照，独立metadata完整审查99请求和真实分派、原AST执行体与数值体保护、原state_gate不变，再实际model CLI dry。独立逐节点数值审查使用baseline/terminal两套参数，按当前节点选择；原完整point checker、BF16注入、inactive/vision保护、条件和初噪配对、zero逐值回归和source链保持不变。完整99/3465与真实wait0/reap/noSignal/GPU释放合格后，才读取原六future archive评分。

当前plan只登记此唯一验证候选。真实启动必须在准备和注册/CLI检查通过后由持久化owner持有原模型Popen。仅允许UUID GPU-ad74a609-3a39-43cb-1392-24cbd279ce7f（当前visible2/minor6），CPU/OSMesa空CUDA/LD，port31359，seed42。检查实际资源与写入fsync，至少10GiB可用，不删除产物腾空间。失败保留计数并停止，无自动重试；观察超时不重启。

评分保持原054/055/056所有state门槛，candidate标识变化不改变阈值。与raw成对评价，052为保留历史基线。只有state gate通过才另登记开发；fresh仍关闭，实时和真机另验。本验证成功执行或数值通过均不等于任务稳定成功。
