# 054：052新头在053同步当前端点的配对采样

总目标沿用036，最终原留出24条稳定≥20/24且每任务≥4/8不变。052仅完成数学资格与teacher改善；053实际同步数据/native稳定正控6/6及独立审查通过，旧W2开发失败与旧STATE同步未认证记录保留。

固定：原053六demo、每条四当前端点共24，seed42/43，raw/probe052两种各48次，加原037三个完全相同输入的零头回归，共99请求/3465 denoise及head边界调用。052使用完整840000行唯一λ=0.1/1290的已独立QR参数，不再fit、调λ、混合旧W2或按task选择头。预算1次模型初始化，0物理/future RGB/训练/fresh，不自动重试。

worker只读053 current_inputs.json、24当前PNG和模型资格/052参数元数据，不读取053 trajectory/HDF/专家动作/原生success。caption只有任务标题，禁止未来action/state、物体真值或success条件。历史037模型capture只用于3个zero的105节点完整源hash回归，不是validation未来标签。每对raw/probe052必须全FP32初始noise、condition与caption完全相同，同seed、同输入、同checkpoint8400/nonEMA/active10 mask/H40/20Hz/35步/g1/shift5/SDPA。新候选schema/标签独立，不冒充旧W2 owner或资格。

仅GPU UUID GPU-ad74a609-3a39-43cb-1392-24cbd279ce7f，显式CUDA_VISIBLE_DEVICES；另一允许UUID不自动备用。运行前检查实际UUID/PCI/minor和空闲状态，禁止其他可见卡。独立准入、源/输入快照、真实model Python CLI dry、完整数值审查入口和进程正常wait/reap收据准备完成后才能启动；这些前提未全满足时prepared仍不授权执行。

独立数值审查覆盖99请求/3465 compact节点的实际sigma、原始C0、feature顺序、头计算、delta_v=-delta_clean/sigma、FP32实际deployed velocity、inactive118=0、zero原对象/完整源hash、配对初始条件及模型参数前后一致。任何已发生失败保留，不扩大容差或重采。通过数值/生命周期资格后，CPU独立评分才打开053未来物理端点，不把teacher拟合误差当采样结果。

事前state门槛沿用原028/038数值，仅候选名称替换为probe052：h8/h16 pooled joint score≤raw的0.90，且严格优于hold与因果constant velocity；所有h8/h16/h40的位置RMSE、SO(3)RMSE、opening MAE同时≤raw×1.05并分别≤raw+0.002m/+0.01rad/+0.01；至少5/6 demo的h8/h16 joint≤raw×1.05，任何demo均≤raw×1.25；两个seed各h8/h16 joint≤raw×1.05。joint=mean((p/.01)^2+(r/.05)^2+(opening/.1)^2)。constant velocity仅用当前已可用历史，anchor0无历史时零速度，不读取未来估速。始终报告全部96行、3任务/6demo/两seed，无剔除。该门槛是参考接近性，不是任务成功。

只有新state资格通过才另登记统一候选开发控制对照，仍需原三任务×init0×seed42/43的完整分母、native/oracle正控与实际执行state error。最终fresh仍关闭，任何软件或state通过均不等于总目标完成。
