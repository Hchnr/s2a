# 028：冻结027，验证实际GPU及完整生成state

在任何本项GPU/模型运行前登记。027一个固定fit及一个独立AugLS已通过原teacher门槛，开发validation velocity MSE总体改善57.45%、三个sigma均改善；见[正式报告](../../reports/wam_train_scale_readout_20261021.md)。这不是state或任务证据，027原结果eligible_for_sampling=False不改写。此次另给运行验证授权，不重新fit、重建参数或选择候选。025/026失败保持，稳定目标仍为统一冻结候选fresh12初态×seed42/43至少20/24、每task至少4/8、末40端点native连续成功。

固定027实际prepared、fit、独立fit receipt、最终peer及adapt系数/center/scale的SHA；原84源/快照、163权重stat及8400身份保持。新prepared继承原12旧EP metadata、48anchor、3zero、96validation raw/adapt顺序与全部物理/采样/域/mask/归一化合同。新运算仍为FP64先(h-mu)/s、拼bias、除实际sigma、乘W、加base、数学单次BF16 RNE；不改变任一训练参数、尺度、sigma函数、门槛或数据划分。

按阶段执行：

1. 完成独立新入口/必要CPU测试，冻结来源、配置、计划；独立prepare-only验证继承及新完整source/snapshot，先核027实际一个fit/独立一个重建/旧bias复用/teacher True和正常CPU退出。准备及资格审计不重开EP/PNG，不重新solve或重建参数；只核旧实际证据和系数身份。
2. 仅物理GPU6执行新路径纯合成精度资格，指定原模型Python/Torch2.10+cu130。五实际sigma（FP32 .2/.5/.8、原solver末NN .12809006869792938、.9990000128746033）各正/负/次正规，共15合成运算场景；再五个相同sigma下加载实际027小readout系数/mu/s、seed42合成hidden的场景，共20。验证实际CPU/CUDA full128 BF16位、实际hook、zero原对象、其它domain/118槽及原输入不变，另10个中点/正负零/次正规原语。实际WAM权重加载0、denoise0、fit0、原EP/PNG0。独立NumPy从实际保存输入/输出bits重算所有场景，不导入生产apply/舍入函数。实际owned wait0/reap、逐次记录资源观测且GPU6 mem0/apps空，才保存公开资格及独立receipt。
3. 仅027 teacher证据与新精度资格全部通过，GPU6一次加载原冻结8400，3份zero候选完整35步生成，逐个raw及normalized40×128与原011缓存exact。使用027冻结ScaledFlowReadout、原026实际SolverSigmaBridge，只读原UniPC scheduler实际sigma，每步只委托一次原velocity/原head；不用整数t推算sigma，不替换solver。非正/不一致sigma在原NN前拒绝。zero候选直接返回原head对象，不经零加法。3zero失败立即停止，不再执行validation或重复模型。
4. 之后按原96项固定顺序生成：六validation EP×四anchor×seed42/43×raw/adapt。只给当前PNG及task，保留空task_plan、原相机/token/归一化/mask；每对实际initial FP32 noise、mask、首condition latent、caption/token精确相同。保留全raw/normalized输出、完整3465调用ledger与99条实际35步sigma/head trace，原所有参数version/identity与requires_grad冻结。没有未来真实状态或图像作为在线输入。
5. 唯一模型正常wait/reap且GPU资源释放后，独立CPU先验全99完整、3zero exact、48对输入与实际sigma/单次委托、参数/权重身份及原预算；全部通过后才核SHA并只读六原validation EP真实state/mask标签，独立SO(3)积分原H8/H16/H40误差、hold/CV对照及EP/task/seed聚合。不能从BF16 action重建GT或把teacher改善代替state表现。

State门槛原样：H8/H16 joint score<=.90raw且严格优于hold与CV；H8/H16/H40九组件均<=1.05raw，并分别有+2mm/+0.01rad/+0.01opening绝对退化保护；至少5/6EP两个短分数<=1.05raw、任一EP短分数不得>1.25raw；各seed两个短分数<=1.05raw。报告全部通过/失败项、随时间增长与分组结果，不按task/seed拼接或改阈值。现有validation已查看，仍是开发诊断，非新无偏或预训练未见验收。

新预算：纯合成GPU资格一次（20场景+10原语，0WAM）；WAM初始化一次、3zero+96paired共99完整生成/3465denoise；新teacher/fit/独立参数重建/物理/fresh均0。GPU只有物理6，CPU独立审计和任何OSMesa空CUDA/LD；禁止GPU0–3。已有输出拒覆盖、无自动重试；owner保留精确Popen/birth/pidfd及实际wait/reap，只清理实际owned子进程，不按NVML PID猜测或kill未知任务。

每阶段GPU退出后的纯资源观测最多120秒，逐次记录UTC/elapsed/mem/apps；超时保留原False，不把观测超时当模型质量失败，不自动重启。任一新source/参数变化拒绝；全部运行源码在开始后冻结。外部Cosmos/发布数据/权重只读。028仅全state门槛通过才获得另登记三task×两seed六条开发物理的条件，原fresh24继续关闭。新adapter收益不计为原checkpoint通过或S2A反解成功。
