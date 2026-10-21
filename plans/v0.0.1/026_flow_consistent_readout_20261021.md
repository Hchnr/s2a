# 026：按flow时间参数化clean-action残差

在任何新候选fit或模型前向前登记。[025](../../reports/wam_frozen_readout_20261021.md)保留False：其总体teacher速度MSE改善28.53%，sigma=.8退化2.043倍，未执行96次完整生成。此次是受该结果启发的新开发实验，不是已证实训练/推理bug修复，不更改或挽救025参数、产物或结论。目标仍是同一冻结候选在预留12初态×2seed中≥20/24、每task≥4/8、末40端点持续成功；目前尚未达成。

数学依据：冻结Cosmos rectified_flow.py:208–209及omni_mot_model.py:2276–2278定义v=epsilon-x0；实际UniPC fm_solvers_unipc.py:302–309用x0_hat=xt-sigma_solver*v。故以delta_x0参数化时delta_v=-delta_x0/sigma。此x0是标准化的clean Action128，含local eef delta与绝对opening，不是机器人当前state/TCP，也不是SE(3)切空间误差。具体源码与SHA需进入准备来源闭包。

原WAM保持22-source non-EMA8400/H40/20Hz/guidance1/35steps/shift5/SDPA/active10，原参数及其它domain共享head不变。只复用025已独立核验的144份原hidden/base/target/sigma缓存，无新teacher前向、future图像编码或新EP读取。保持原六train/六开发validation、四anchor、三个sigma、noise seed42及所有逐层等权；validation及原raw结果已经看过，明确属于开发验证，fresh未打开。

单一新候选定义：phi=[h/sqrt1280,1]，A=-phi/sigma_actual，Y=active10(v_target-v_base)，解sum(w*||A W-Y||²)+.1||W||²。W为1281×10共12,810参数，bias同样正则。另解只用A=-1/sigma_actual的10参数对照，同样标签/权重/lambda，共两个生产fit，无网格或调参。保留原velocity MSE目标，不换成sigma²加权目标。lambda数值相同，但W单位变为clean-action残差，先验已改变，应在结果中披露。

Teacher的sigma_actual取已保存sigma_action FP32值无损扩为FP64，不用JSON名义小数代替。监督保留v_target-v_base，不拟合字面的x0_true-(BF16_xt-sigma*v_base)；后者会额外引入xt量化误差/sigma，属于不同目标。旧xt/target/hidden均保持原字节。

部署在完整LIBERO domain/normalization31/projection0/active10匹配的请求中，按FP64先构造-phi/sigma_actual、矩阵乘W、加原base，再作数学单次BF16 ties-even；其它118槽与其它domain不变。zero候选直接返回原base对象，不能经零加法。所有代码和新输出均在S2A维护，不编辑025冻结53源、外部Cosmos、权重或发布数据。

推理必须取原UniPC实例实际solver sigma。现有SamplerTrace在src/s2a/sampler_trace.py:257–286通过被固定SHA/原forward代码身份限制的真实caller frame读取sample_scheduler，并核实例、step、timestep；301–311每步只委托一次原velocity。新独立薄封装可沿用该只读方式，在此次velocity委托上下文传sigma给head，不替换solver。原011/35步实际trace最后NN的sigma约.1280900687>0，终点0不调用NN；packed timestep已转int64，t/1000与solver sigma不完全相同，禁止据其猜sigma、裁剪sigma或向0时刻外推。非正/非有限或不符当前实际step的sigma在原NN调用前拒绝。

按顺序执行：

1. 完成新core/入口/独立checker及必要测试，冻结新source/config/plan；复核旧144特征与原2fit失败证据。独立checker使用NumPy及增广最小二乘，不导入生产fit/apply/gate。原source与缓存不修改。
2. 物理GPU6执行新运算路径的纯合成CPU/CUDA bitwise资格，包括实际sigma数值、正负BF16 midpoint/次正规/zero旁路/其它域/118槽。0WAM权重、0denoise、0EP读取、0fit，实际owned wait/reap及显存0，资源观测逐次留证且最多120秒。
3. 只CPU执行恰好两组固定fit，独立重建参数和实际量化teacher评分。按原025门槛：验证pooled≤.95raw、严格优于本次sigma-aware bias、每sigma≤1.05raw；固定容差只用于参数/数值重算，不用于门槛比较。失败停止后续模型阶段，不重试或改变特征/损失/正则。
4. 仅teacher门槛及精度资格通过，GPU6一次加载冻结WAM，先三份zero候选完整35步生成并精确复现011原raw/normalized40×128；该新时间封装也必须单次委托、原参数/RNG不变。通过才96次固定validation EP×4anchor×seed42/43×raw/adapt成对生成，每步记录实际solver sigma与head调用，初态noise/mask/首latent/caption/token逐对exact。只消费当前图像/task与生成latent，不读取真实future作在线输入。

完整采样仍用原025全部state门槛：H8/H16 joint score≤.90raw且严格优于hold/CV；H8/H16/H40九组件均≤1.05raw并分别有+2mm/+0.01rad/+0.01opening保护；至少5/6EP两短分数≤1.05raw、任一EP不得>1.25raw；每seed两短分数≤1.05raw。积分用当前measured anchor及独立SO(3)实现，给出H8/H16/H40及EP/task/seed分组。不用teacher改善代替state或任务能力。

新预算：CPU生产fit2及独立重建2，无新teacher；条件3zero+96paired共99完整生成/3465denoise，物理/fresh0。复用缓存不冒称新的独立样本；阶段失败明确后续未执行。只有全state门槛通过才另登记三任务×两seed六条开发物理闭环，其后才能考虑冻结候选并开启原24次fresh验收。新adapter收益不计作原checkpoint通过或S2A反解成功。模型仅GPU6；CPU/OSMesa显式空CUDA/LD，禁止GPU0–3。已有目录拒覆盖、无自动重试、只清理实际owned子进程。
