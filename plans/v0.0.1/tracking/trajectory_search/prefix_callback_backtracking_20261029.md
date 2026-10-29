# H12首个callback方向的完整响应回缩（CPU、执行前登记）

总目标及18正确参考/12WAM开发/24最终留出门槛保持，GPU禁用。本轮争取从原actual368因果状态找到原H12全部目标的完整模型witness，随后才考虑另行登记的物理pilot。不能以局部下降、软件检查或模型witness替代实际控制和留出任务成功。

## 依据与启动前置

[保存差分诊断](../../reports/prefix_trust_saved_linearization_20261029.md)已实际wait/reap0，结果SHA`0fe281c9c67dfbbc0441586ce1f680baa24539c08a22438f63d0947307f97394`、owner SHA`fcc29ced5b904a12a5d610512e35bc4ae5a456b86b178316a8a2dadf6a55b369`；独立算术交叉核验SHA`08b6ff1f0ffdf061789389402b9918093ad85998cb832a4b5fbd7a1afcb8cfd0`，45组J和45个callback两模型差异0。36个非零callback的前缀线性预测最大超限约±2e-14，实际均为正2.17e-5..4.97e-4；9个零步另计。现有SLSQP已有内部线搜索，callback不是已知accepted iterate或显式暴露的原始QP方向。

登记时，两臂v2完整独立复算仍在运行，允许实现与合成测试，但正式producer必须等两臂4000/3148响应全部严格核验、owner实际wait/reap0、无signal和来源不变。新准入绑定两个已关闭v2结果/owner的实际SHA、全部trace预算和原producer精确引用，不以文件存在或partial progress替代。若任一失败，不启动新搜索；保留失败原因。旧已执行代码、测试、计划和结果全部冻结。

固定同warm及因果输入：actual368 q/v、completed367 previous command/旋转latch、固定手指状态、chunk46原时序12目标；warm仍为已核验固定回缩`cases.prefix.entries[55].request.commands`，bytes SHA`16bdfbb266ef13d2588c8116e13511d93df936d72ae8a57d19df0b01ed3acee0`。不读actual369+，不改参考相位或引入新初态。

完整模型使用已编译资产快照，manifest SHA`c9eaaca963a82f504ed38e67df847fc54c144d030f660be52f6439db0b7b3566`、compile SHA`a0bbbbdfe379feeb4f9c2b83c7f1e66e9aa95d884bcfb19fd5131439f8dbe5ed`。只迁移路径，99引用/96资产逐SHA及XML逆字节验证，不能裁剪模型或写回旧外部路径。

## 唯一候选与有限控制流

新增`solve_prefix_callback_backtracking_response`，保留冻结信赖域solver的原数学响应、epigraph当前target目标与1e-4全轨迹loss、72维联合命令、有限差分步1e-4、ftol1e-12、九组约束、原rate以及所有容差。原position .01m/rotation .05rad、box/rate1e-12、动态审计1e-10、关节margin .05、braking .1s、terminal和零profile不变。不加入额外pose margin或二阶校正。

1. 从已接受anchor起，原bounds与`anchor±radius`相交，半径固定`.01→.005→.0025`。每阶段清空导数cache，完整响应的精确bytes cache跨全部阶段共享。
2. SLSQP仍设`maxiter=min(5,剩余迭代)`，不可用maxiter1冒称已生成一步。callback先计一次实际major回调、保存epigraph和完整响应；若已有完整witness，优先结束。否则首次命令相对anchor数值非零位移时捕获方向并用专用异常退出。不得伪造正常optimizer返回、success或nit；零位移callback仍计入迭代上限。
3. 捕获命令记为trial。固定依次尝试`alpha=[1,.5,.25,.125,.0625,.03125,.015625,.0078125,.00390625]`。alpha1直接复制trial字节，其他点为`anchor+alpha*(trial-anchor)`，不clip、投影或另选方向。alpha0仅为已有anchor，不额外查询。每点请求进入统一ledger；命中cache不得计新响应。
4. 普通anchor仅来自上述ray的首个合格点：完整joint/command审计通过、原已达prefix严格保住、非线性minimum≥1e-8，且prefix增长或该阶段原target max平方归一化误差下降>1e-12。按降序遇第一个合格点即停止回缩；不将差分探针的普通下降当作ray接受。每个实际回缩点保存完整九组审计及错误/拒绝原因，接受anchor亦保存完整审计。
5. 完整H witness在所有已完成cache中都可发现，包括差分探针；仍只要求全部原caps及full audit，不强加普通下降的1e-8 buffer。按原最大归一化平方误差、全轨迹loss和首次响应顺序择优。任何预算中断都保留并扫描完整cache，不能漏掉已发现witness。
6. 普通ray下降获接纳即重新定anchor，半径回到.01，即使prefix仍9也允许继续。无捕获方向或全部固定alpha均不合格时，anchor不变、缩至下一半径；三半径失败即停止。不靠结果驱动追加alpha/半径、改warm或放宽门槛。

这比较“早停方向生成、显式完整响应回缩、普通anchor限于ray”整套控制流与已完成同warm的两旧方案，不宣称仅一个因素变化。最终selected仍按所有已完成安全响应的原prefix/target/loss排序，单列它是否等于最后anchor；最终selected不反向改写阶段接受历史。

## 预算、记录和独立核验

全局最多4000唯一完整响应、60实际累计SLSQP迭代、60阶段。warm、FD、callback、回缩和任何最终评价全部计费；回缩不是SLSQP迭代，单列attempts及新响应数。达到迭代上限后可以完成已经捕获方向的有限ray，但不能再开优化阶段。每响应仍300数值内步/312查询，另一个current query；最大1200000内步/1248001查询。producer时限1200s，独立checker5400s，无自动重试。

保存每个request的role/stage/command bytes/response与cache状态/前后计数，FD axis与nominal delta，捕获callback的request/response/epigraph/位移，全部alpha尝试的request/response/完整审计/资格/拒绝原因，以及每阶段实际callback/optimizer返回状态、半径、前后anchor和预算。中途budget拒绝保留为无响应，不能假装完成评价；zero step及无方向明确记录。

独立checker使用已核验的完整迁移XML和冻结独立query/rollout/classify数学，重新执行全部实际唯一轨迹；所有compact误差与严格标签逐项一致。完整initial/selected/closest、每个ray实际点和接受anchor的已保存完整数据分别核对。独立检查ledger全部FD组、首个非零callback捕获、alpha精确字节/固定顺序/首次合格早停、仅ray普通接受、global witness/最终排名、cache及全部预算。不能导入producer solver来决定资格。

先真实合成例验证已知可行困难例的实际表现和诚实终止、几何回缩能保住前缀并下降、违规整步与FD普通下降不被误接纳、witness全cache发现及共享预算；不得把困难例是否成功预设成可任意调整的断言。完成CPU QA、实际CLI dry与来源/计划指纹并提交后，唯一启动`outputs/tracking_v5_prefix_callback_backtracking_20261029`；checker为相应`_check_20261029`，owner为各自`_execution_20261029`，均拒绝覆盖。

只运行本候选，不重跑已关闭旧baseline/信赖域producer。与两旧同warm结果比较完整witness、原节点prefix、cursor9误差、真实响应/迭代/阶段开销及每个接受anchor的改善。无完整witness如实记录，不据有限失败宣称无全局解。即使获得完整模型witness，实际控制pilot、18条正确参考、12条WAM开发及最终24条留出仍需各自独立证据。
