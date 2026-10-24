# 中文 PR 信息

标题：完善原生数据资格、clean 残差训练与可审计接续

正文：

本次补齐原生收尾诊断、稳定训练数据采集、真实 clean 推理桥接，以及固定 D0→W1→D1→W2→STATE→DEVELOPMENT 的训练和开发评测入口。增加来源与执行准入、完整采集审查、生产拟合和独立 QR 重算、真实进程归属与退出收据，以及工作区迁移后的安全路径解析。配置、协议、失败结果和当前目标分别留档。

新 D0 已完成300请求、10500节点并通过独立审查及正常退出/GPU释放。随后 W1 因验证器源码引用被产物路径限制误拒，在拟合前失败；保留原失败，新增043协议、局部源码回调修复和仅CPU的有限接续入口。043实际拟合尚未执行，不重采合格D0。

验证：

- 当前全仓 Ruff 检查及格式检查通过，共474个Python文件。
- 原041相关检查730 passed、1 skipped；当前484份冻结源码逐SHA未变。该结果不代表全仓pytest通过。
- 新043四组专项分别30、45、35、9项通过，覆盖源码回调、薄worker、独立准入及真实CPU子进程顺序/失败回收；当前文件与收据SHA一致。
- 全仓历史重放仍有迁移路径问题，边界保留在`plans/v0.0.1/033_workspace_relocation_20261022.md`。

当前没有新增WAM任务收益或最终留出成绩。验收仍为原12新初态×seed42/43共24条，稳定至少20/24且每任务至少4/8；实时和真机另验。最终验收组件的隔离原型与实验产物仍在忽略的`outputs/`中，未提升为主源码；本提交记录相关设计和证据边界。

# English PR information

Title: Add qualified native data, clean residual training, and audited continuation

Body:

Add native terminal diagnostics, stable training-data collection, a real clean-inference bridge, and the fixed D0→W1→D1→W2→STATE→DEVELOPMENT training and development pipeline. Include provenance and execution admission, complete-capture verification, production fitting with independent QR reconstruction, owned-process exit receipts, and safe artifact resolution after workspace relocation. Preserve explicit configurations, protocols, failures, and the overall acceptance goal.

The new D0 completed 300 requests and 10500 nodes, passed independent verification, and exited normally with GPU release. W1 then failed before fitting because an artifact-only path check rejected the registered verifier source. Preserve that failure and add protocol 043, a process-local source-callback correction, and bounded CPU continuation. Actual 043 fitting has not run; qualified D0 is retained without recollection.

Validation:

- Repository-wide Ruff lint and formatting checks pass for 474 Python files.
- The original 041 related checks recorded 730 passed and 1 skipped; all 484 frozen source hashes still match. This is not a full-suite pytest claim.
- The four new 043 checks recorded 30, 45, 35, and 9 passing tests, covering the source callback, worker, independent admission, and real CPU child ordering/failure cleanup. Current source hashes match those receipts.
- Historical replay tests retain documented relocation failures in `plans/v0.0.1/033_workspace_relocation_20261022.md`.

No new WAM task improvement or final holdout performance is claimed. Acceptance remains the original 24 trials across 12 reserved initial states and seeds 42/43: at least 20/24 stable successes and at least 4/8 per task. Realtime and hardware require separate evidence. Isolated final-acceptance prototypes and experiment artifacts remain in ignored `outputs/`; they are not promoted into production source by this commit.
