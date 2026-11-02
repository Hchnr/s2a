# 063 训练划分物理扰动恢复资格：运行中

按固定计划，三个训练demo各0/+0.02/-0.02m基座x偏移，共9条、2307物理步、0模型/拟合/fresh。原训练初态SHA与prepared排除集合不重叠；开发及fresh不用于生成训练标签。

参考日程与控制计算复用检查5项通过，Ruff通过；真实Python3.11下9条dry均通过，0新物理步。采集入口复用显式OpenCV恢复，在固定控制循环中新增端点observer；AST证明除显式参考输入、训练限定和两处observer调用外，所有原控制表达式相同。observer本身的真实物理状态/时间及图像绑定仍需运行后独立核验。

实际持久化owner PID3401637/birth582586420，首CPU仿真client3401645/birth582586452，按/proc确认原身份存活。源码及输入指纹保存在`outputs/training_state_recovery_qualification_20261025/source_sha256.json`。固定9条顺序执行、无自动重试、CPU/OSMesa空CUDA/LD，线程使用监督器要求的8。

最近快照：首case00已经生成初始真实端点，首当前图像渲染正在进行；完整物理步、恢复效果和原生任务资格尚未成立。不得将目标插值或原native未来直接标为该运行的实测未来。采集完成后仅使用actual_trajectory.npz中的真实端点构造标签；还需独立重建SITE/BODY/TCP、时间、delta/mask、图像及控制/夹爪回放，并单列零偏移和非零偏移结果。

原061候选0/6、062正确参考3/3及最终fresh24条门槛保持；063不是WAM或留出任务成功，总目标未完成。

## 首条物理完成后的序列化失败与有限接续

case00实际完整220步、221端点，physical result complete=True/error=None/stable_success=True；actual_trajectory.npz已保存，但collection_result.json在images内ndarray序列化时中断。原客户端exit1、owner停止，无自动重试；原不完整文件与失败收据及serialization_incident.json保留，不能写成正常采集资格。

新增v2 worker仅更改输出namespace及结果先完成ndarray兼容序列化再创建文件；Ruff和实际Python3.11 dry通过。独立continuation owner要求原case00完整220物理步及原owner失败，固定仅case01–08/2087剩余步，不重复首条或扩大2307总预算。新namespace `outputs/training_state_recovery_continuation_20261025`已实际启动并复核原身份存活。首条记录需离线补充元数据与独立标签核验，不以新写入冒充原正常退出。

## 首条已存标签的独立局部见证

新增`scripts/verify_training_recovery_labels.py`通过Ruff并实际运行case00：221实测端点的raw sim_state SHA与逐端点收据匹配；直接NumPy独立重建source@tool、局部右乘delta、state/action128、末端有效mask及20Hz时间，全部通过。没有调用producer synchronous_labels函数。`case00/labels_independently_verified.json`明确full_qualification=False，0新物理/模型；仍未补足完整控制/图像审查或原exit1，不把局部标签通过冒称整个采集正常。剩余case01仍由同owner/client实际存活运行。

## 首个非零扰动已实际完成

case01（bowl训练初态，基座x+0.02m参考）完整220步/221端点，collection_result合法落盘、3当前图像，实际stage passed=True，原客户端3404124退出。原continuation owner3404116与case02客户端3404770按原birth确认存活，继续固定剩余范围。

实际独立标签脚本复核case01全部221端点通过；随后调用既有verify_canvas分别核验frame0/16/56的真实状态/时间绑定、原始相机flip与独立NumPy画布重组，三张最大通道差均1，满足其原容差。images_and_motion_witness.json保存局部见证，full_qualification=False。第16步实测source位移约(+0.0195446,-0.00000273,-0.0000643)m，第56步回到初始source位置的距离0.0002607m；说明本条确实产生了近2cm物理偏移并返回，非只改变目标数组。producer原生末段稳定True仍须完整独立控制回放认证；本项不证明大幅漂移恢复或WAM收益。

## 完整控制回放局部通过，wine零偏移失败保留

case01实际运行`verify_training_recovery_case.py`，完整220步机械/夹爪/生成参考对应审查通过，独立native outcome ever/terminal/stable=True、成功87步/尾连续80。新checker显式接收独立构建扰动参考，保留原控制计算和容差；AST复用测试通过，允许变动仅oracle参考准入、固定预算与实际fork路径及oracle-only限定。首次实际准入因原执行器路径不在fork实际导入闭包而拒绝，随后更正为核验真实fork，未降低SHA或数值检查。

当前原continuation owner3404116及case05客户端3407636按/proc确认存活；case01/02已完整且producer稳定True，case03/04（wine零偏移/+2cm）已完整但producer稳定False。故063零偏移正控要求尚未满足，当前不能称pilot通过，更不能据已成功bowl案例扩展训练。继续原固定范围，待完整机械/标签/图像审查后定位wine参考或执行差异，不改门槛或替换失败案例。

## wine零偏移：真实暂时成功后丢失

case03完整310步控制/夹爪/参考回放实际通过，独立native ever=True、terminal/stable=False，成功15步、尾连续0；311实际端点标签/时间/hash独立核验通过。这是被一致性检查确认的真实性能失败，不能把机械passed当资格通过。成功出现在step223–244之间且不连续。producer自身参考位置RMSE1.974mm、旋转0.008808rad，末段目标开度约0.9966、实际约0.9976，已打开；因此既不能单独归因为大tracking误差，也不能说末段没有释放。此处目标物位置仅离线描述，不用于在线切换。

wine case04/05同样producer稳定False，当前三个wine条件均不能满足正控资格。原owner3404116与case06客户端3408549实际存活，继续固定剩余cream范围。063不能直接作为合格恢复数据扩展；须完整结果后检查零偏移额外56步与原native控制/状态路径、夹爪映射及接触敏感性，不将原native成功标签直接转移到该S2A轨迹。

## 已完成6条的机械/标签核验补齐

实际独立完整回放case00/02/04/05全部通过；结合既有case01/03，前三组bowl和wine共6条的完整控制/参考/夹爪记录均一致。bowl三条件独立stable=True，成功步88/87/86且尾连续均80；wine三条件独立ever=True、terminal/stable=False，均成功15步、尾连续0。case02/04/05实际端点标签/hash/时间核验也通过，前6条标签均已有局部见证。case00原exit1与缺失完整图像metadata仍保留，机械通过不修饰生命周期失败。

当前case06（cream零偏移）producer完整且stable=True，case07客户端3409465和原owner3404116实际存活，继续原固定范围。wine按原native时间对齐（跳过新增56步）检查：native正闭合81步中adapter有73步命令为0而非再次闭合，反向打开0步；position_hold的0命令可能是正常保持，不能把该计数直接解释为没闭合或提前释放，后续应比较实际motor cache/夹爪位移和接触过程。

## 固定九条全部完成：性能资格失败

2307实际物理步完整结束，无重复首条。九条全部实际完整机械回放和端点标签见证通过，两个启动source_sha256全表再次检查无漂移；case01–08全部实际wait0/reap无信号，24当前图像独立原始相机/绑定/NumPy画布检查通过。原case00序列化后exit1、不完整元数据及图像metadata缺口保留，不冒称完整正常采集。`pilot_summary.json`明确performance_qualified=False、full_collection_qualified=False。

独立稳定结果bowl3/3、wine0/3、cream3/3；三个零偏移为2/3，未满足063全正控要求。全部cream三条独立成功/尾连续均85步，原owner及全部客户端均已退出。小扰动物理生成与返回在部分任务可行，但不能当作统一合格恢复教师，不扩展训练或拟合新头。后续先隔离wine零偏移失败与新增56步、正确参考接触过程/夹爪映射的关系；原061候选0/6及fresh关闭不变。
