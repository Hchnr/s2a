# WAM serving mask缺陷与移植说明

已经确认的缺陷是无效Action128维度在训练和推理中的处理不一致。S2A评测服务已修正并完成离线配对验证；外部Cosmos仓库保持只读，生产服务尚未修正。

## 源码依据

以下外部行号对应Cosmos提交`8ed68a9722a14e8f6c21fa0c6c700e1617b7b73e`，目录为`/share/project/eai_pwm/home/hcr/repos/cosmos-framework-pwm`。历史训练提交`7415eb5f1bcd88a04565699a277ab827e227a19e`的10个核心函数AST与此版本一致。

| 环节 | 实际行为 | 位置 |
| --- | --- | --- |
| 训练加噪 | 使用loss_valid_mask将无效action元素置零 | `cosmos_framework/model/generator/omni_mot_model.py:2336` |
| Action128 serving | raw_action_dim固定为128；有效维度另由action_dim_mask描述 | `cosmos_framework/inference/action_serving.py:280` |
| 推理初始噪声 | 只清除raw_action_dim之后的元素，128维协议下没有清除任何槽 | `cosmos_framework/model/generator/omni_mot_model.py:2620` |
| 推理velocity | 同样只按raw_action_dim清除尾部 | `cosmos_framework/model/generator/omni_mot_model.py:3026` |
| 服务输出 | 最终反归一化后应用action_dim_mask，此时不能撤销采样中已发生的跨维干扰 | `cosmos_framework/inference/action_server.py:194` |

LIBERO有效槽为前10维，其余118维应保持零。原生H40/2秒、3任务×3个sigma的冻结前向对照中，只改变无效槽噪声就产生0.27–0.43的有效velocity RMS差异；训练与按mask对齐的推理有效velocity逐项一致。当前实际训练attention/padding与SDPA之间的最大velocity RMS差异为0.00551005，无法解释前述更大的差异。

## S2A本地实现

[`src/s2a/wam_runtime.py:12`](../src/s2a/wam_runtime.py#L12)中的`install_active_slot_mask`显式接收维度mask，并在三个位置执行：

1. 按原形状、原seed生成全部噪声，再把无效槽清零，保留有效槽随机数与随机数消费顺序。
2. 每次denoise前确保输入的无效action槽为零。
3. 将输出velocity的无效槽清零，使后续solver更新不会重新引入这些槽。

没有改变网络参数、归一化参数、采样步数、guidance或shift。服务入口[`scripts/serve_wam_acceptance.py:74`](../scripts/serve_wam_acceptance.py#L74)拒绝非LIBERO十槽合同；该限制是这次实验的明确边界。

不是caption修复：192/192个实际输入的最终caption字节和首帧像素与发布输入一致，`exact-published-caption`在这些样本上没有改变输入。早期嵌套JSON假设已撤回，见[协议勘误](../plans/v0.0.1/wam_protocol_errata_20261019v2.md)。

## 已执行回归与可复用证据

- 3任务×3sigma，H16与原生H40的冻结训练目标/推理前向核对；H40包含真实action/vision有效mask、完整caption、标签clip及训练attention/padding对照。
- 原服务8400/8600/8800分别与masked版本配对：每版384次相同输入/seed预测，包含多阶段、两种文本条件、H16/H40；另外384次父权重warm-start基线。总计2688次实际预测。
- 最新8800、task-only、study-heldout、H40：位置RMSE129.808→64.213mm，SO(3) RMSE2.0923→0.5159rad，开度MAE0.2170→0.1079。该留出属于本实验整episode隔离，不是预训练未见集合。
- 每个新服务副本在执行物理矩阵前，必须逐项复现保存输入/seed对应的完整Action128响应；各副本的`replica_check.json`保留比较结果。

固定配对输入示例：`outputs/wam_checkpoint_acceptance_20261019v2/state_queries/8800_masked/task0_ep1172_approach_h40_task_only_s42/`中的`request.json`、`response.json`及对应服务manifest。数值证据见`numerics/8800_h40_published/results.json`和`numerics/8800_h40_published_training_backend/results.json`；完整state报告见[直接state验收](wam_state_acceptance_20261019v2.md)。

## 移植到通用Cosmos服务时的约束

应将每个请求的显式action_dim_mask传入完整采样路径，保持128维存储/归一化合同；不要简单把raw_action_dim改成10，也不要把normalization_mask或action_control_mask当作有效槽mask。其他机器人可能使用非连续有效槽，不能把本轮LIBERO固定十槽hook直接推广为全域实现。

通用batch、混合domain、非连续mask、不同采样器以及并发请求的隔离回归尚未在本S2A实验中执行；应在Cosmos生产修复时覆盖。保留本轮固定输入/seed作为LIBERO回归样本，同时确认全有效mask不会改变原路径、非法或空mask被拒绝。生产移植完成后，再按同一冻结协议报告效果，不能仅凭最终输出的无效槽为零就判断中间采样合同正确。
