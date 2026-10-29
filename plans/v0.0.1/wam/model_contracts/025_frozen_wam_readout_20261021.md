# 025：冻结WAM主体的LIBERO残差readout开发验证

在任何新feature前向、readout拟合或采样前登记。024尚在执行，本项所有模型/fit阶段以其27次前向独立审计通过、owned进程实际退出与GPU6释放为前提；不假定024已经通过。原023的False及不部署决定保持。目标仍是同一冻结方案在新12初态×2seed中≥20/24、每task≥4/8、末40端点持续成功；本项不是目标达成，也不开放fresh性能。

依据：raw8/8400的seed42为1/3、43为0/3；延长60秒仍1/3，wine/cream后半较好跟踪生成参考仍失败；020集成仍1/3且参考跟踪变差，022完整command未过门槛，023输出几何校正有部分结构收益但未过保护项。下一可反证假设是：冻结backbone/action tower已有的hidden中，是否包含一个固定线性readout可读取的、更有用的跨任务状态变化信息。不是继续调控制器，也不将新adapter收益当原checkpoint成功。只读方案和42输入账本在`next_stable_intervention_readonly/`，memo SHA `5af0fd37ebd0d69004a609a4b1c62622a77d6fff4917d3d53aed508955d5805d`。

固定旧22-source iter8400 non-EMA、H40、20Hz、guidance1、35步、shift5、SDPA、active10。全部原学习参数及normalization保持不变，外部源码/权重/发布数据只读；不导入Cosmos训练dataloader。本项是生成模型输出端的学习适配，不是机器人响应反解。

实际head为`llm2action(hidden, projection_domain)`，hidden宽1280。LIBERO的完整action-domain是`franka_panda__parallel_gripper/eef20/observed_local_delta/libero_closed_pad_observed_endpoint_eef10_v1`，normalization index31，projection index0。另26个normalization域共用projection0，故不能修改共享slot0。S2A自有request上下文同时验证完整domain、normalization31、projection0及active10，才给原head前10输出加残差；其它域、其它118槽原样。zero readout直接返回原输出对象，不经零加法或重组。拒绝混域或不一致上下文，不能单凭projection0开启适配。

固定特征phi=[h/sqrt(1280),1]，监督Y=active10(v_target-v_base)。真实future标签仅用于训练监督/开发验证；部署hidden来自当前图像、正常task-only文本及模型当前noisy latent，不接真实future/阶段/object/contact。按episode→anchor→sigma→有效endpoint逐级等权，总权重1，解一次sum(w*||Phi W-Y||²)+.1||W||²的zero-prior ridge，W为1281×10共12,810参数，bias同样正则。另解一次仅phi=[1]的10参数constant-bias对照，权重/标签/lambda相同。只fit六训练EP，无网格/验证选择/早停。该等sigma残差目标不是整个原训练objective。

原hidden/base多为BF16，导出值保持精确。W保存float64；适配只在这12,810参数的小矩阵上用float64计算phi与residual，加到base后一次转换回base原dtype。避免改变原网络全局TF32/precision设置；CPU独立重算明确模拟BF16 ties-to-even。teacher评分使用实际部署量化后的输出，不能只报未量化拟合误差。zero旁路的完整采样必须精确复现旧40×128 raw/normalized；不以数值容忍掩盖原网络变化。

只用既有12条已提取且在fresh预留时保守排除的episode。训练：bowl1172/1098、wine893/1151、cream898/850。开发validation：bowl1254/1126、wine868/1139、cream1209/1176。validation过去的raw结果已看过，称开发验证，不称新无偏验收或预训练未见；完整EP不得进入本次fit。不开新R2 state/pixels，不因metadata可用就宣称新数据排除了fresh HDF demo。

每EP从已有NPZ长度N固定四anchor=floor(j*(N-41)/3)，j=0..3；原12条的header检查均四个不同anchor。实际准备必须检查每anchor真实41帧、40个action_valid及active10，全有效失败就保留并停止，不移frame。按长度覆盖不声称精确阶段标签。sigma=.2/.5/.8，noise seed42；所有teacher输入用线上task-only文本，真实GT future经024已核训练prepare/noise路径。训练/验证feature提取均原模型、0适配。

固定分阶段预算（GPU6顺序运行，CPU空CUDA/LD）：

1. zero readout3次完整35步生成：011每task首个原缓存seed42请求，全部40×128 raw/normalized exact才准入。
2. 12EP×4anchor×3sigma=144次single denoise，train/validation各72次，保存hidden、base、target、xt、mask、实际条件与RNG。边界计数禁止额外forward。两个CPU fit完成后独立重算全部参数和teacher评分。
3. 仅teacher门槛通过，才6validation EP×4anchor×seed42/43×raw/adapt=96次完整生成。当前measured anchor积分，统一H8/H16/H40 state指标，保留hold/历史匀速对照及task/EP/seed分组。所有raw/adapt输入和初始噪声严格配对。

总上限99次完整生成＋144次single=3609次denoise，CPU fit2，physics0，fresh0；服务重新加载不自动增加identity或其它生成。训练和validation各2880个endpoint行，同EP重叠窗口不当独立episode。若任何准入失败，后阶段未执行明确列出，不补分母/自动重试。

Teacher门槛：来源/冻结/identity/数值均通过，validation pooled active10 velocity MSE≤.95raw且严格优于constant-bias；三个sigma分别≤1.05raw。只决定是否执行96次采样，不单独证明state/任务能力。原模型与bias/adapt全部量化后同口径比较。

完整采样门槛（事前工程保护，不改023原门槛）：H8/H16共同分数mean((pos/.01m)²+(SO3/.05rad)²+(opening/.1)²)均≤.90raw，且严格低于hold、历史匀速；H8/H16/H40九组件均≤1.05raw，同时位置≤raw+2mm、朝向≤raw+.01rad、开度≤raw+.01。2mm取已有正确参考典型跟踪量级，其余是固定开发保护，不能解释为抓稳/力阈值。至少5/6 EP两个短分数均≤1.05该EP raw，任何EP的任一短分数不得>1.25raw；两个sampling seed各自两个短分数≤1.05对应raw。全部比较不加结果后容忍。

通过仅允许另登记旧3task/init0×seed42/43六条600步开发闭环，建议至少5/6 stable且每task≥1/2后才考虑冻结fresh协议。本项不授权物理或fresh。未通过保持失败，不按本结果改lambda/容量/sigma/anchor或挑任务seed；更广训练若有必要另立协议。独立审计不调用生产fit/apply/评分，明确已执行/未执行、raw权重/新adapter、teacher/完整采样/闭环/真机的不同证据等级。
