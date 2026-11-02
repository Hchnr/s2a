# 在线任务文本与发布 caption 的只读来源核查

`exact_published_caption=False` 表示未启用 S2A 的“直接覆盖完整 JSON caption”诊断开关，不能据此认定任务条件缺失或错接。当前 019 的 225 条真实请求及保存的 response args 中，`prompt` 和 `extra.action128.task_title` 都与各自 fixture 的任务文字逐字一致：bowl→plate、wine bottle→rack、cream cheese→bowl。当前 020 已登记的任务文字保持不变，本核查新增模型请求和物理步数均为 0。

来源收据为 `outputs/wam_stability_20261021/temporal_equal_review/online_task_caption_source_audit.json`，SHA256 `ab4efba7c8605b7c9c1ca586f1ea0c307d57705b5bea48c9c8403b21738f3e84`；473 项输入 SHA 前后复核。它认证请求的路由和已保存的响应参数；019 的服务生成目录仅保存 `sample_args.json`、`sample_outputs.json`，没有最终 batch caption 的逐请求账本，因此不把源代码分析冒称为实际内存文本逐字认证。

## 在线准备路径

- `src/s2a/temporal_ensemble_simulation.py:118` 将 `record["task"]` 放入 `task_text`；`src/s2a/libero_simulation.py:147` 同时把它写入请求的 `prompt` 和 `extra.action128.task_title`。
- `scripts/serve_wam_acceptance.py:32` 保存开关值；其 `:72` 至 `:96` 显示，只有启用开关且 prompt 是 JSON 才直接覆盖 `batch["ai_caption"]`。False 时仍调用原服务准备器。
- 外部只读 Cosmos 的 `cosmos_framework/inference/action_serving.py:165` 调用训练共用的 `render_caption`，把简短任务转成包含 task_title、command、机器人、时长、频率和相机布局的 JSON。`:302` 至 `:345` 还支持解析并保留已经完整给出的 JSON caption，不存在此前猜测的 JSON 嵌套问题。

## 与预训练文本的关系

在线简短任务与当前发布的同任务训练 caption **不是同一文本**，差异超出大小写和标点。训练共用 renderer 位于外部 `cosmos_framework/data/generator/action/unified_action128.py:332`：它以 `source_task_description` 作为 command，并按固定 seed 规则加入可选 task_plan/current_subtask。当前发布样例的 command 描述了抓取、抬升、搬运、放置等具体过程；在线 command 只有简短任务目标，缺少这些示范描述和计划。

例如 bowl 的发布 command 描述下降、抓住金属碗、抬起并移动到红边盘子；wine 明确描述放到右侧倾斜木架后松开并收回；cream 描述把蓝色奶酪块放入灰色碗。三个目标与在线任务相符，但不能据此证明措辞、示范细节或计划对模型性能没有影响。发布样例按 `source.original_tasks` 选择，见 `scripts/prepare_wam_acceptance.py:95`。当前发布快照也不能替代 8400 历史实际消费 caption 的逐字账本，历史输入和消费情况仍未恢复。

## 既有验证覆盖与限制

1. 192/192 完整 published JSON 的准备前后文本及首帧像素一致，证明当时完整 JSON 路径没有被嵌套或改变，不证明简短在线 prompt 等于训练扩写 caption。代码见 `scripts/audit_wam_caption_contract.py:37`；历史结论见 `plans/v0.0.1/record_20261019v2.md:62`。
2. 原 8800 H16 task-only 数值诊断让训练与推理两侧使用同一文本，token、条件 latent、有效 velocity 一致。该原始产物中 wine/cream 的 task_title 借用了 bowl 模板；代码后来同时设置 task_title，限制已在 `plans/v0.0.1/record_20261019v2.md:83` 披露。它是同字节条件下的 packing 验证，不能冒称三个自然在线任务的完整条件验收。
3. 后续 8800 H40 完整 published-caption 数值诊断使用各自正确描述，再次覆盖三个任务的训练/推理计算一致性，见 `reports/wam_checkpoint_acceptance_20261019v2.md:27`。它不直接证明当前 8400、H40、task-only、guidance1 的全部在线数值合同。
4. 已有 8800 的 task-only/published-caption state 对照没有一致改善：H40 diagnostic 的位置误差 80.463→92.820 mm、SO(3) 0.5849→0.6377 rad；episode 留出组位置 64.213→63.805 mm、SO(3) 0.5159→0.5570 rad。见 `reports/wam_state_acceptance_20261019v2.md:74`。这是当时配置的离线 state 诊断，不能推广为当前 8400 的因果结论，完整示范计划也不能计作纯在线任务能力。

目前没有任务文字错路由的证据。训练扩写描述与在线简短目标之间存在已知条件差异；其对当前 8400 任务效果的影响尚未单独确定。本核查不改变 020 prompt、不申请新性能网格，也不将该差异先验认定为失败根因。
