# 024：当前 8400 训练目标与线上 conditional 数值桥接

**结论：当前协议的 27 次前向及独立 CPU 重算全部通过，未发现本项覆盖范围内的训练—推理计算差异。** 三个任务各一个已查看的 approach 开发样本，分别为 bowl/episode1172、wine/episode893、cream/episode898，均从 frame0 读取真实 41 帧未来双相机 RGB 和 40 条 Action128 标签。没有完整采样、HTTP generation、物理步、拟合或 fresh 性能评测；本项没有新增任务成功证据。

固定 non-EMA `8400_masked`、seed42、20Hz/H40、SDPA、不 compile、Action128 active10、guidance1 的单 conditional 计算。唯一实际模型运行使用物理 GPU6，`CUDA_VISIBLE_DEVICES=6`、`LD_LIBRARY_PATH=''`；独立审计为 CPU、空 CUDA/LD。checkpoint 精确位置见 [配置:9](/share/project/eai_pwm/home/hcr/repos/s2a/configs/wam_current_numerics_20261021.json:9)：

```text
/share/project/eai_pwm/repos/cosmos-framework-pwm/outputs/action128_wam_21src_20261012_warm_start_run01/cosmos3_action/action128/action128_wam_nano_1b_22src_20261012_warm_start_videocap30_static_vae_aot_run1_from6k/checkpoints/iter_000008400/model
```

实际执行按 [事前登记](../plans/v0.0.1/024_current_training_inference_numerics_20261021.md)，每样本在 sigma=.2/.5/.8 下各运行以下三路，共九组三路：

| 分支 | 实际输入及计算 | 检查目的 |
| --- | --- | --- |
| A | 真实未来 RGB/action、原 published caption；真实 training prepare、加噪、denoise、完整训练 loss | 复算当前发布条件下的冻结训练目标 |
| B | 相同真实未来标签，改为实际线上 task-only caption；独立 training prepare，与 A 恢复同一全 RNG 状态后加噪 | 隔离文本条件，保留配对标签和噪声 |
| C | stock inference 自己产生的 caption、token、sequence plan、clean conditioning；直接接收 B 的同一 noisy x_t | 检查实际线上 conditional 计算与训练计算桥接，不借用训练 clean metadata |

训练准备完整编码真实未来 RGB；stock inference 保留因果 VAE 的 prefix 编码和未知未来零占位。两者未知未来 clean latent 无须相同，首条件 latent 必须逐值相同。真实输入、归一化与 stock 准备见 [producer:539](/share/project/eai_pwm/home/hcr/repos/s2a/scripts/audit_wam_current_numerics.py:539)；配对 RNG、实际 training loss 和 C 调用见 [producer:677](/share/project/eai_pwm/home/hcr/repos/s2a/scripts/audit_wam_current_numerics.py:677)。所有子模块保持 eval，training prepare iteration 实际为 8400；加噪不传 iteration，记录的 deterministic algorithms 为 false。

独立 checker 不导入 producer 数学、Cosmos、Torch 或训练 dataloader。它用 NumPy 重算 `v_target=epsilon-x0`、有效 sigma 的插值及 BF16 舍入、无效 action 槽置零，按实际 condition/valid mask 重算 action/vision 的有效分母、uniform 时间权重和加权总 loss。每次 action 分母为 400，vision 为 1,152,000；不能用全 128 槽平均误差替代。未来相机原像素、训练实际 raw batch、归一化合同及首 canvas 也被核查；未来 cover-crop 复用已冻结的历史 canvas 源码合同，并未另行独立实现 VAE。数学依据见 [checker:158](/share/project/eai_pwm/home/hcr/repos/s2a/scripts/verify_wam_current_numerics.py:158) 和 [checker:185](/share/project/eai_pwm/home/hcr/repos/s2a/scripts/verify_wam_current_numerics.py:185)。

| 项目 | 实际结果 |
| --- | --- |
| 完整实际 denoise ledger | 27/27，三任务 × 三 sigma × A/B/C，无重复或缺失 |
| A/B 标签、完整 RNG、epsilon/x_t/target | 逐值相同 |
| B/C 实际 token、首条件 latent、前向打包 metadata、有效及返回 action velocity | 逐值相同，最大差 0；仅训练监督字段不要求与线上相同 |
| 独立 flow 等式 | 全部通过事前 rtol=1e-6、atol=1e-7 |
| 18 组独立完整 loss | 全部通过事前 rtol=1e-5、atol=1e-6 |
| 独立与实际 loss 最大绝对差：vision/action/total | 1.87e-8 / 1.57e-9 / 1.98e-8 |
| 来源核验 | 270 个消费文件 SHA 前后相同；原 163 项 8400 权重 stat 相同，未重复散列全部权重字节 |

精确比较没有使用 loss/flow 容差；A 与 B 的预测和 loss 因文本不同而不要求相等，也没有事前“loss 多小就能完成任务”的门槛。实际三路、RNG 与条件速度联合检查见 [checker:689](/share/project/eai_pwm/home/hcr/repos/s2a/scripts/verify_wam_current_numerics.py:689)。

唯一模型进程 PID806744/start169980814、owner PID806664/start169980768 均已确认原 birth 不存在；实际 wait 返回 0，模型已 reaped，未发送终止信号，GPU6 显存恢复 0MiB。外部 Cosmos tracked source 前后均 clean，HEAD 为 `8ed68a9722a14e8f6c21fa0c6c700e1617b7b73e`。日志有退出前未显式 destroy NCCL process group 的 warning；实际进程退出和资源释放已经独立确认，没有据此重跑或修改旧结果。生命周期入口见 [owner:159](/share/project/eai_pwm/home/hcr/repos/s2a/scripts/run_wam_current_numerics.py:159)。

| 证据 | SHA256 |
| --- | --- |
| [实际 producer result](../outputs/wam_stability_20261021/current_training_inference_numerics/run/result.json) | `7bed8b7a69579d4701313ff70d8511485cd4eac5a18b180f30a3fb843490daeb` |
| [完整独立 CPU 审计](../outputs/wam_stability_20261021/current_numerics_review/complete27_verified.json) | `17cdcdfd59a22e743bded2da2f6b27167a8463eac5e74897e1079450c26fcf79` |
| [实际退出及资源释放核验](../outputs/wam_stability_20261021/current_numerics_review/root_actual_exit_verified.json) | `e465d42711e1af9c8f00306759b1f9fb5ed74673f049a2914cd8e51f36142256` |
| [本轮整仓 CPU QA](../outputs/wam_stability_20261021/current_numerics_validation/validation.json) | `dfa25ec0aa92ae549deb07bdd5cd5d3d956ea9e27d461acb8d957ec570b0fd58` |

整仓 QA 对应 `e7691cf` 的 292 个 Python 文件：433 passed、1 外部 integration deselected，pytest 23.00s，Ruff check/format 均通过，源快照前后不变；不包含随后另立项的 025 草稿。

本项补上当前 8400/正确三任务 task-only/H40/mask10 的真实训练目标专项缺口，允许后续把重点转回生成质量和可验证的模型适配。它只覆盖三个已查看开发输入和三个 flow 时刻，不是历史训练 batch、augmentation、optimizer 重放，也不是当前训练 Flash3/fixed-padding backend 与 SDPA 的完整等价认证；上游图像/pose 的物理同步仍未知。数值桥接通过不能代替 state 质量、独立闭环任务和冻结 fresh 稳定成功验收。
