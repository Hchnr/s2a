# 052：固定单值正则化对照

总目标仍为036规定的统一候选与原24条留出稳定门槛；旧W2开发0/6稳定和原STATE gate=False均保留，fresh关闭。

依据[全量容量与残差诊断](wam_clean_capacity_diagnosis_20261024.md)，在任何新fit前固定[052计划](../plans/v0.0.1/052_clean_regularization_probe_20261024.md)与[显式配置](../configs/wam_clean_regularization_probe_20261024.json)。仅测试λ=0.1/1290，无扫描或按task/seed选参数；feature和bias惩罚都改变，不声称唯一归因feature容量。复用原150锚点、D0/D1的600请求与840000行；不重采模型或使用fresh。

生产注册`outputs/clean_regularization_probe_20261024/registration.json` SHA256 `5690cf4ad019cb4bacf63ae9b58c44ddcd955183e1babec446f8dfa9fb4c054b`绑定旧fit/独立QR、生产脚本、原数值源码及计划。实际CPU attempt1已启动，双遍归一化和单次ridge solve；每遍feature/target/weight流须与原fit一致，任何失败保留且不自动重试。原参数不改。

原streaming生产primitive的26项测试通过；独立augmented QR primitive的34项测试通过，三份新增入口Ruff通过。测试不等于当前真实fit或独立数值审查通过。

独立入口`verify_wam_clean_regularization_probe.py`不调用producer的row构造或fit函数：从034已存标签独立FP32 multiply/add/clipping归一化，从actual sample/raw velocity与sigma重算C0；全量增广QR、每遍流SHA和原rtol1e-9/atol1e-11比较均须通过。结果另记录，不回写producer为“已独立合格”。完成数学资格也不等于新采样、任务成功或fresh许可。

## 实际生产fit与teacher比较完成

生产进程正常exit0，attempt1/fit1，完整双遍840000行/600请求；feature/target/weight三流SHA均与原W2相同，中心和scale按固定容差一致，消费输入前后无漂移。`outputs/clean_regularization_probe_20261024/result.json` SHA256 `fbed6821f59d143b3299524f2aaca1926cec7c27cd79ada89cfb040b66a7f927`；新parameters SHA256 `ca01b9d8a49f2f1585c917f492e52a0a89cdb0b447f6bd2beee050ed0d2fccc0`。producer独立资格字段仍False，另行独立QR正在执行，不回写原结果。

独立审查注册`audit_registration.json` SHA256 `cfebe88c2da0ce33278960e6d0f9c476f179069b043559df6da392ccb17b5254`，只允许1次完整QR重构，无producer fit/模型/物理/fresh。

同一旧D0/D1分布上的teacher比较由`analyze_wam_clean_probe_error.py`完成，`teacher_comparison.json` SHA256 `d957125286ba12f939439a1834dafac31fa7e4d26873c56320a6e4735ea8fcc1`，feature与target流再度匹配原fit，进程exit0。相对零头，归一化xyz/rot6d/opening MSE改善：旧头2.96%/42.72%/1.99%，新头29.59%/64.25%/17.83%。新头各任务xyz改善19.26%/37.36%/32.37%，opening改善13.00%/18.27%/23.44%。这是同一缓存上训练误差，不能称validation泛化或新策略采样收益。

将新λ代入此前完整训练协方差的实际特征值，feature有效自由度为47.78613478947542（同一设计矩阵的解析重算，不是新数据拟合）。训练拟合与容量变化相符，仍需独立数值资格和新采样证据。

## 独立完整QR已通过

实际独立进程正常exit0；两遍全部840000行/600请求、150个独立FP32标签归一化窗口及21000个raw-C0节点重构通过，三流SHA与原fit一致，所有消费输入前后SHA不变。`independent_verified.json` SHA256 `a15383c61a712d0a3e3d2fdc23f9ebd26ce10101c7ca84c28aad1259872c507f`，原rtol1e-9/atol1e-11未放宽；coefficients/center/scale最大绝对差分别为6.426803533798875e-13/0.0/2.6645352591003757e-15。

052生产fit1和独立重构1的数学资格现已成立，原producer中的独立资格False保留，联合证据取本新增审查。仍为0新模型/物理/fresh；训练改善没有证明泛化。下一步是带实际物理时钟、当前图像/实测状态同步的六个原validation demo采集准备，修复旧初态同步未认证边界，然后另行固定真实采样与开发对照，不能沿用旧W2身份部署新参数。

同步validation来源预检已实际通过：仍为原六个demo（bowl16/32、wine4/27、cream14/24），六个实际float64初态SHA互异且与30个训练源、12个fresh源无交叉；三个只读HDF全文件SHA与历史声明相同，模型XML存在，全部native动作形状有效。预检产物`outputs/synchronous_validation_preparation_20261024/source_identity_preflight.json` SHA256 `c435cd2c9da5521caea2a0916033312a17a0f9e53351814c16e6759e5a7b0d3d`。main .venv缺h5py的首次调用在任何源读取前失败，未安装依赖；使用现有sim Python3.11完成预检，未修改共享环境。此时仍0新物理/当前或未来RGB采集，预检不是同步资格。六条native长度95/92/158/177/106/104，合计732；若沿用034每条80步固定收尾，总预算1212物理步。
