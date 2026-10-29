# dual recovery：输入路径恢复后的隔离check-only v2

本登记在v2独立复算之前建立。唯一生产已正常关闭，保留122完整响应、52QP、
52次NNLS拟合及全部原数据；不重跑生产、不增加搜索预算、不启动GPU。

首次checker session52839/child2786346已实际wait/reap exit1，在load_inputs的来源
核验阶段遇到缺失的`plans/v0.0.1/constraint_residual_repair_20261029.md`；owner末尾
核验又遇到缺失的`prefix_trust_region_control_20261028.md`。失败发生在模型构造、
ledger审计及任何响应复算之前，未创建原checker数据目录。两份失败收据冻结：

- `outputs/state_vector_dual_recovery_checker_20261029/result.json`：
  `2ec7846a7b13adcc25163af6060c69c74598eab2f279d990d548e7c2d646709f`。
- `outputs/state_vector_dual_recovery_checker_20261029/00.log`：
  `74aee12bf1b71d5523f34037b7ddce2a93d7b78e40132417df2974b367643149`。

同一期间HEAD从`9299ec3`变为外部merge `3a12fc7`（合入`3517cdf`），但两commit
完整Git tree相同。随后两个路径均再次可读，原182来源SHA全部一致；无需恢复或
改写任何历史文件。这里只记录观测到的路径可用性变化，不据此修改历史收据。

## 唯一允许的恢复执行

新owner收据独占`outputs/state_vector_dual_recovery_checker_recovery_v2_20261029/`。
执行原冻结`check_tracking_v5_state_vector_dual_recovery.py`，其尚未创建的原数据
目录`outputs/tracking_v5_state_vector_dual_recovery_check_20261029/`首次写入仍受原
exclusive守卫；原数学/全KKT/阈值/严格执行caps/全部122响应复算保持，不绕过来源。

启动前绑定本登记、原QA/准入/生产owner及唯一producer、首次失败owner/log；逐项
核验原182源码及生产全部2738来源、旧driver。要求恢复锚点`3a12fc7`的完整Git tree
与`9299ec3`相同；准入时当前HEAD保持该锚点，复算前只允许新增并提交本登记，
记录执行前后当前HEAD/tree，不把正常登记提交误称为旧tree。原生产SHA保持
`436fbe20f20cf6678aa329c05e928845b563f76cb0a976c11395e16481aba551`。
单独先执行一次零响应`--dry`并保存实际退出证据，通过后提交本登记，才启动v2。
新owner记录真实driver_ref、original_driver_ref和recovery_plan_ref，不伪造旧driver。
执行后重新核来源及收据，实际wait/reap；1800s预算，无自动重试，失败完整保留。

已冻结的原182测试及两dry仍按原SHA使用；算法源码未改变，不重复拟合或生产。
若v2成功，保存数据分析采用隔离v2输出，只替换checkerowner位置及明确的恢复
driver/失败收据验证；原分析器未执行且继续冻结，其他闭合、成本和对照保持。

首次checker新增QP/拟合/模型/响应/查询均0；v2对122条保存响应全部独立复算，
每路线预期36600数值内步/38065查询，自身QP/NNLS0，全部实际计费。数值witness
仍需另行实际前向控制；原18/12/24验收、GPU禁用及总目标保持。
