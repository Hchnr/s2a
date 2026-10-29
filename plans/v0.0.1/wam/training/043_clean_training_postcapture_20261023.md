# 完整D0之后的有限W1 CPU接续 — 2026-10-23

总目标继续沿[036](036_project_goal_20261023.md)：真实世界模型参考在原预留24条中稳定至少20/24、各任务至少4/8，并保留经过控制系统后的实际state误差。当前尚无该结果。本项只恢复既定训练链中的W1；完成本项不能代替D1/W2、完整生成诊断、开发实测及最终留出验收。

## 已发生的事实

041新D0已完成300请求、10500个denoise与readout节点；完整独立审查、实际正常wait/reap、同PID/birth与GPU释放均通过。原040中断的20完整请求和partial产物保持封存，不进入fit。[D0只读复核](../../outputs/wam_fresh_acceptance_implementation_20261023/d0_data_readonly_audit_20261023_v1.json)重新核验来源、300请求和10500点metadata，未重新decode数组，数值资格沿用已实际完成的完整审查，并保留其省略字段边界。

随后041 W1 producer在`load_context`中失败：`check_execution_proof`将已登记的独立执行验证器源码ref传给只允许`outputs/`产物的`Evidence.path`，触发`ValueError: fit artifact escaped S2A outputs`。失败发生在`fresh_output`、标签读取、`fit_attempts`和`solve_blocks`之前，实际fit/QR/未来数组decode均为0；W1 producer和owner均实际wait1、正常reap、无signal。supervisor保留False并停止，D1/W2/STATE/DEVELOPMENT未启动。不重启041 supervisor，不覆盖或修订其失败收据。

## 唯一代码修复

新增`clean_training_source_proof.py`，在独立CPU进程局部暂换fit contract的proof回调：仍执行原完整proposal→verification→registration审查，只有该审查返回且被原source map逐SHA绑定的唯一验证器源码，改用原`Evidence.source_map`记录和核验。普通产物路径仍经原`Evidence.path`，保持`outputs/`限制。上下文结束恢复原函数，并核全部源码/metadata指纹。

原484份041源码、484份快照、11份bundle、模型/数据、训练seed和300输入均保持原字节。原fit和独立QR数学源码不修改；新增薄worker只在上述context内调用原`run_fit`或原`verify_fit`各一次。实际入口、helper、owner、独立准入检查器、测试、本计划和显式配置另加源码/快照冻结；新补充准入不能仅凭原041的授权字段获得执行资格。

## 有限输入、输出与预算

输入固定为原041的prepared/admitted和已合格新D0 collection/audit/owner；读取相同150个训练锚点、相同300噪声、全部35×40行，共420000行。ridge=.1、1290特征列、训练来源、样本划分、归一化、权重、数值容差与数学方法沿原038/041，不加候选或超参数选择。

原W1 fit输出`recovery_20261023/fit/W1`、独立审查`recovery_20261023/review/W1_verified.json`当前不存在，只允许各首次发布。新补充材料、worker收据、实际Popen owner及新supervisor在`outputs/clean_training_postcapture_repair_20261023`维护；原失败的`orchestration/W1`和`supervisor/W1`绝不改写。原fit/checker产物保留原prepared/041 registration身份；新owner额外明确实际补充准入、修复源码和生命周期，不能追认为旧失败owner已成功。

| 项目 | 本项上限 |
| --- | --- |
| W1 production fit | 1次、420000完整行 |
| 独立全量重算 | 1次、同420000行的原独立augmented QR |
| 新模型初始化/denoise | 0/0 |
| 物理/fresh | 0/0 |
| 自动重试 | 0；任意失败停止，保留产物 |

本项没有消耗原041实际拟合预算，失败的CLI启动不冒充已做fit；补充后的1次producer及1次QR计入原两轮总上限，不增加原W1/W2拟合数量。原040＋041的模型调用累计上限保持不变，本项不授权D1或任何新模型。后续剩余阶段须按保留的W1新owner实物另做明确接续，不恢复旧supervisor或伪造旧`prior(W1)`成功。

## 执行前与执行后的证据

1. 保存独立失败审计及D0资格；核所有旧进程terminal和GPU释放。
2. 新helper和薄worker相关回归测试、Ruff通过；原源码逐SHA不变。
3. 用真实D0元数据执行完整`load_context`，显式阻断`np.load`；记录420000行上下文、唯一回调修复、0decode/0fit及前后源码指纹。这不是拟合结果。
4. 独立metadata检查器核新proposal、配置/计划、完整旧＋新增source/snapshot、QA、dry收据、旧D0及失败事实，再发布新registration；所有记录分别保留，不构造循环SHA。
5. 对新admitted执行实际worker dry；只有完整通过，才启动新CPU supervisor。producer正常wait0/reap及完整产物后，才允许独立QR；不自动重新运行任何失败命令。
6. 独立QR必须沿原相对/绝对容差、全部行/标签/权重/参数指纹通过，正常wait0/reap后才获得W1资格。teacher/拟合资格不代表世界模型任务收益；总目标保持未完成。

CPU入口显式空`CUDA_VISIBLE_DEVICES`与`LD_LIBRARY_PATH`，OMP/OpenBLAS/MKL各8；命令从当前仓库执行，不安装或修改共享环境。
