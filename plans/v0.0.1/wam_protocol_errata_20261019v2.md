# 冻结WAM协议勘误

对应配置：[wam_checkpoint_masked_20261019v2.json](../../configs/wam_checkpoint_masked_20261019v2.json)。其SHA256为`52629fd3de41656109ccb9c76559a83713e480b337325384fac4fd13704a358f`。

该配置在数值审计后、修正版矩阵执行前冻结。`notes`中以下判断是当时的错误假设：

> the stock-server published_caption branch is nested JSON and must be labeled separately.

继续阅读完整`prepare_wam_request`并实际复算后，确认stock服务会解析完整JSON caption。48个样本×2个时域×2个seed，共192组输入的最终caption字节和首帧像素完全一致。`--exact-published-caption`在这些输入上是无变化操作，不能将任何收益归于“修复嵌套JSON”；stock/masked配对实验隔离的是无效Action128维度噪声问题。

证据保存在`outputs/wam_checkpoint_acceptance_20261019v2/caption_pixel_audit/result.json`和`caption_contract_check/result.json`。这证明当前准备路径一致，不证明历史训练输入字节未变化。

为保留预注册身份及严格恢复检查，原配置文件字节不改写。其运行参数仍有效；对上述说明的正确解释以本勘误、[实施记录](record_20261019v2.md)和[结论](conclusion_20261019v2.md)为准。这项决策替代记录中早期“结束时改写配置说明”的打算，避免修改哈希后破坏原矩阵的恢复条件。
