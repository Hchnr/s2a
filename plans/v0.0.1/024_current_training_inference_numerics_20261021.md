# 024：当前8400的真实未来训练目标与在线conditional数值桥接

在任何新增模型前向前登记。022完整固定文本对照及023固定局部校正均未通过各自门槛，保持原失败，不将其候选送入物理或fresh。当前稳定成功仍未达成。先补用户四层验收第一步的一个具体联合覆盖缺口，避免继续围绕未排除的计算链路疑点调控制。

只读审查见`outputs/wam_stability_20261021/training_inference_coverage_readonly_20261021/memo.md`（SHA936610c5b6e9e1bc591f5cb82f660cda2085cad9c1a24551ee7485b25b61c9e6）。原8800/H40 published确从NPZ读取真实41帧未来RGB、真实40条Action128和有效mask，调用真实训练prepare/noise/denoise/loss，不是zero-future合成训练loss。当前8400/H40/正确三任务task-only/mask10/SDPA的完整采样和输出复现已验证，但同组合训练目标专项尚缺。旧源码/产物保持原字节。

固定8400 non-EMA、当前只读Cosmos代码、SDPA、无compile、20Hz、H40、seed42、物理GPU6。选择原samples manifest中每个task的第一个approach样本，共三个已查看开发输入；要求完整41帧/40行有效标签，不用其他task请求fallback。原相机canvas、domain、Action128归一化与clip均保持。每个task的线上请求来自011已经精确复现的对应正确任务记录，实际stock prepare后的caption/token必须与该记录相同；published条件取该样本原published_prompts[40]。没有任务阶段或未来state进入线上条件。

每个样本固定三个flow sigma=.2/.5/.8，按以下三路各调用一次denoise，总9组三路/27次denoise级前向。完整WAM sampling=0、HTTP generation=0、物理=0、fit=0、fresh=0；不加counterfactual/backend网格或自动性能重跑。

- A：真实未来图像/action、原published caption，经当前模型真实_prepare_training_data、_add_noise_to_input、denoise、_compute_losses，复算当前发布条件的冻结训练目标。
- B：同真实未来标签，文本替换为实际线上task-only caption，经相同训练计算。A/B按固定seed配对噪声，并记录真实噪声与mask，不要求不同文本的输出相等。
- C：使用stock inference prepare产生的线上caption、sequence plan和clean conditioning信息，接收B完全相同的noisy x_t，经当前mask10的_get_velocity conditional计算。不得用B的训练clean metadata替换C的stock conditioning来掩盖线上条件来源。guidance1无额外postprocess等价于单conditional；这不是35步采样质量试验。

训练prepare保留对真实全部未来帧的VAE编码，stock inference保留只编码条件prefix的优化；未知future的clean占位不应等于训练GT。比较并保存实际首条件latent、条件mask/位置/clock、caption/token/domain、归一化标签、x0/epsilon/x_t/vtarget、prediction和完整action/vision loss所需tensor。iteration实际8400，不沿用旧工具8800常数。检查eval状态与所有训练/推理无效槽输入、velocity零值；必须对中间量留证，不能只用最终清零当合同通过。

A/B的training prepare分别执行，要求全部clean vision/action、监督mask及shape逐值相同；每次加噪前恢复同一保存的Python/NumPy/Torch CPU/CUDA RNG状态，C只接B的xt、不再draw noise。_add_noise_to_input不传iteration（沿原工具，避免启用另一deterministic噪声合同），记录其实际deterministic设置；prepare传8400与噪声seed是不同参数。拒绝任意子module仍处training状态。首条件latent/token/domain/mask等准备合同若不等，保存差异并在新增denoise前停止；B/C数值差异同样明确拒绝，不自动修补。

事前数值门槛：B/C有效action velocity、首条件latent、实际token、mask、归一化label要求逐值相等（既有8800相同路径实测差为0）；A/B配对噪声及未来标签相同。若出现非零差先完整保存并报告拒绝，不依据新结果放宽门槛。独立loss重算只允许float32规约舍入：rtol=1e-5、atol=1e-6；逐元素flow target epsilon-x0和加噪等式rtol=1e-6、atol=1e-7。独立核有效分母、训练时间权重、action/vision权重及总loss，不将active-only MSE误当训练loss。所有loss必须有限，当前无预注册“loss多小就能完成任务”的阈值。

先实现新增隔离入口、CPU输入/错task/噪声与loss检查及独立审查，保存source SHA与原163项checkpoint字节指纹对应stat；拒覆盖已有输出。root持有唯一实际子进程handle并保存真实env/birth/exit，单卡锁内确认GPU6无占用。原训练dataloader不导入，外部Cosmos/发布数据只读，命令由S2A执行。CPU准备/审计CUDA与LD为空，模型明确CUDA=6、LD为空。

通过只支持当前三个已发布开发样本和三个flow时刻下，发布条件训练目标可重算、线上短条件计算桥接一致；不是历史原batch/augmentation/optimizer重放，不证明上游图像和pose物理同步，不替代state质量或多任务闭环能力。保存训练Flash3/fixed-paddingbackend未纳入本次预算，旧8800/backend及013证据单独引用，不冒称当前完整历史训练数值等价。新任务成功仍须独立闭环和冻结fresh门槛。
