# 043 CPU接续与标签运行时差异 — 2026-10-24

**043实际W1已按失败停止：1次拟合尝试、0成功拟合、0参数、0独立QR。** 修复的源码引用准入已通过真实数据预检；实际执行在首个训练锚点的标签逐值重建处拒绝。原041和新043的False分别保留，D1未启动；完整D0资格不改，总目标仍未达到。

本次先提交代码`2d218d8`，中英文PR信息见[提交说明](pr_clean_training_20261024.md)。下述为该提交之后继续执行的实际证据。

## 准入与执行

147项相关CPU测试通过，259.52秒；474份主Python源码的Ruff及格式检查通过，不声称全仓pytest通过。原484源码及其快照保持原SHA，新043完整495源码及495快照通过前后核验。[QA](../outputs/clean_training_postcapture_repair_20261023/qa/qa.json)，SHA `bb3708732fd9b95dd9f69b39d084610e7826a4357d8407369bd39d8bc05e9bb9`。

真实D0完整`load_context`在显式阻断`np.load`的预检中通过：420000行上下文、源码回调校正1次、0数组解码、退出恢复与源码守卫通过。[metadata dry](../outputs/clean_training_postcapture_repair_20261023/preparation/metadata_dry.json)，SHA `0d5b2f3989209343d5316a7227ca10db3675f662494054b1219d2c374e9855f5`。

独立准入实际核15123份消费文件，保留原D0审查的省略字段边界，0新数组解码。[独立verified](../outputs/clean_training_postcapture_repair_20261023/execution/verified.json)，SHA `0483aa4da74de930a21a73db4247cfc1c80427efa07bb7f283923d43e352c6e5`。新[admitted](../outputs/clean_training_postcapture_repair_20261023/execution/admitted.json) SHA `f79469e4bf8cc721b24300afbcfc73bcf3b768791247252464bd3adc179c43d8`仅允许一次W1 producer及一次独立重算，模型/denoise/physics/fresh预算均0。producer、independent及owner的真实admitted CLI dry均正常exit0。[CLI dry收据](../outputs/clean_training_postcapture_repair_20261023/preflight/admitted_cli_dry.json)

短launcher实际exit0后，CPU supervisor启动producer。producer PID1949214/birth574491422实际wait1、reaped、无TERM/KILL；新supervisor保留False，未执行independent。原`fit/W1/result.json`记`fit_attempts=1`，新worker记`delegate_calls_attempted=1`；不能因成功fit为0而把这次尝试抹去。[实际终态](../outputs/clean_training_postcapture_repair_20261023/preflight/terminal_observation_v1.json)

实际失败发生在[fit label:84](../scripts/fit_wam_clean_training.py#L84)，由streaming fit首轮normalization读取首个标签时触发，尚未进入参数求解。错误为`034 saved labels differ from measured local endpoints`；没有生成`parameters.json`，没有新的世界模型/物理调用，也没有自动重试。原失败产物、源码、阈值及标签均保持不变。

## 同一失败窗口的运行时核查

只读取本次已打开的首个训练窗口`task0_demo_8_a0`，固定同一archive、同一源码和同一TCP变换，在两个实际CPU解释器分别重建，不调用fit或模型。

| 运行时 | 保存/重建dtype | 不同值 | 最大绝对差 | 转为物理FP32后 |
| --- | --- | --- | --- | --- |
| main Python3.12.3 / NumPy2.1.0 / SciPy1.17.1 | float64/float64 | 186/5120 | 4.440892098500626e-16 | 全部相同 |
| sim Python3.11.15 / NumPy1.26.4 / SciPy1.15.3 | float64/float64 | 0/5120 | 0 | 全部相同 |

[main原False](../outputs/clean_training_label_numerics_20261024/main312.json)，SHA `95ce4c815f436ced895e308ab74064228b64fd264b4a73e6f0650d0aeeb6e2ad`；[sim逐值相同](../outputs/clean_training_label_numerics_20261024/sim311.json)，SHA `43ab098bf3048d7de20b43a88025be368203df26b1c19e45847ed0bf2e8b60a8`。这证明该窗口有跨运行时算术差异，不能唯一归因某个包，也不能由一个窗口授予全部150锚点、归一化目标、采集C0或拟合资格。物理FP32相同不是归一化目标资格，更不代表任务成功。

下一步须另行固定有限CPU数值兼容审查：在候选实际运行时核全部既定训练窗口及保存采集值，保留原严格门和全部失败。只有完整通过并明确累计尝试预算、独立准入后，才考虑新的拟合接续。当前没有新拟合或模型运行获准，不重采合格D0、不打开fresh。
