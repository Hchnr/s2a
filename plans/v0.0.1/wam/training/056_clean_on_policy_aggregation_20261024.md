# 056：固定052策略的训练轨迹聚合

总目标和原24条留出初态稳定≥20/24、各任务≥4/8不变。055数值合格但性能失败，不能进入fresh。依据 `reports/wam_synchronous_pair_diagnosis_20261024.md`，052在原训练分布末节点仍改善，而在自身validation轨迹末节点位置和夹爪teacher误差反转。本项检验补充自身生成训练轨迹能否改善该问题，不预设成功或唯一根因。

## 固定数据和候选

- 独立新目录 `outputs/clean_on_policy_aggregation_20261024`，不得修改D0/D1/052/055或冒充旧D1接续。
- D2使用原150个训练锚点、原300个实际31位训练seed和相同当前图像、caption、首latent、mask及初始噪声。对原D0逐请求精确核验配对；训练噪声与42/43隔离。模型checkpoint、mask、H40/20Hz、35个原噪声节点不变。
- 唯一采集策略为052已独立QR合格的完整头（参数SHA `ca01b9d8a49f2f1585c917f492e52a0a89cdb0b447f6bd2beee050ed0d2fccc0`）。不根据任务、seed、噪声步切换头，不读取任何future标签或native command作为模型输入。
- 采集合格并正常退出后，D0/D1/D2各420000行，合计1260000行；逐行特征仍为hidden1280＋raw C0 active10，目标为原训练实测motion归一化标签减当前raw C0，不能减已部署残差两次。
- 只拟合一个统一头：所有行等权1/1260000，中心/全局向量L2 RMS由全部聚合训练行计算，feature和bias同样以ridge=0.1/1290惩罚。固定一次生产solve及一次独立完整augmented QR，原rtol1e-9/atol1e-11不变。没有D3、重采、ridge扫描、按任务拼接或validation监督训练。

## 分阶段预算与准入

本计划和current-only proposal不授权模型启动；执行入口、冻结来源、独立metadata核验、真实CLI dry和持久化owner准备齐全后单独登记执行。

1. D2：最多1次模型初始化、300请求、10500denoise/head调用；0标签读取、fit、物理、fresh。失败保留实际计数并停止，无自动重试。
2. 数值审查：完整独立重构所有D2节点、初始配对、原始及部署速度、舍入、mask、归一化与输入来源，正常wait0/reap/no-signal及GPU释放。未通过不得读取训练未来标签或拟合。
3. 聚合fit/QR：仅原150训练窗口；固定完整1260000行及源流指纹，消费输入前后不变。既有D0/D1资格与生命周期证据继续复用，不能用本次通过擦除旧失败。
4. 同步validation：另行冻结新候选参数，原053六demo/24锚点及seed42/43，3zero＋48raw＋48新头，共99请求3465节点、最多1次新模型初始化，完整独立数值/生命周期合格后才离线评分。沿用054/055原门槛，不降低阈值。预算不授权当前立刻采样。
5. 只有state gate通过后才另登记固定开发对照；开发、统一候选冻结与原24条fresh验收仍各有独立门槛。本项完成不等于总目标完成。

模型仅允许UUID `GPU-ad74a609-3a39-43cb-1392-24cbd279ce7f`，CPU/OSMesa显式空CUDA和LD；master seed42，worker按UUID设置CUDA。资源占用必须检查真实GPU/端口，不依赖旧lock文件判断存活。保留真实Popen及PID birth、退出和释放证据；观察超时只继续观察同一进程，不能重启。启动前实际小文件fsync写探针和容量检查通过；不删除实验数据腾空间。

## 本轮实现范围

先生成并检验不含trajectory/native command的current-only训练清单，绑定原D0真实请求和initial引用，以及052参数/完整独立QR。随后实现独立D2 worker与审查/owner；禁止复用旧接口的D1身份绕过两轮限制。
