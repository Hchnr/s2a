# 020：因果时间融合执行资格

020 的三个旧场景正确参考正控均达到登记的物理资格：共执行 **3 × 600 = 1,800** 个真实仿真控制端点，末 40 步持续满足原生任务 predicate，TCP 位置 RMSE 均低于 10 mm、SO(3) RMSE 均低于 0.05 rad。所有原有逐步日志字段与 007 interval8 / position_hold / no-governor 正控保持 canonical JSON 精确一致；唯一新增的 `temporal_ensemble` 日志由独立检查器逐步重建。

这仅证明相同正确世界参考经过新融合执行器能够保持旧行为，**不证明 WAM 成功，也不证明融合不同预测有益**。此次三个 oracle 的全部 1,800 个端点均进入 identity fast path，最多同时记录五个同绝对时刻参考。不同 WAM 预测之间真正的融合效果必须由后续登记的三条 WAM 开发评测独立回答。

## 登记与实际执行

- 登记提交：`7025c45e4f7d291d33327893188186fa270e389b`；协议为 [020_causal_temporal_ensemble_20261021.md](../plans/v0.0.1/020_causal_temporal_ensemble_20261021.md)，配置为 [temporal_equal_qualification_20261021.json](../configs/temporal_equal_qualification_20261021.json)。
- 实际执行源码：`49778dd`；独立检查器与资格入口：`af147e1`。未修改旧执行器、旧 007 日志、登记参数或已经生成的严格审计产物。
- 旧场景 `task0_init0` / `task1_init0` / `task2_init0`，experiment seed 与 sampling seed 均为 42；20 Hz，H40，每 8 个实际物理步更新参考，600 步 / 30 s，最多两个 CPU worker。模型请求为 **0**，没有 fresh 初态、拟合、参数搜索、重跑或 GPU 任务。
- 唯一实际矩阵为 PID `526583` / start ticks `168722389`。CPU matrix 和三个 physics worker 均显式空 `CUDA_VISIBLE_DEVICES`、空 `LD_LIBRARY_PATH`；physics 使用 OSMesa。实际 `wait()` 返回 0，四个登记进程均已退出。

在线只使用当前可用机器人状态、历史和已经生成的世界参考。物理步 `k` 的目标端点是 `k+1`，只允许已完成 query `q ≤ k` 的 `j=k−q` 端点参与。位置与开度等权算术平均，旋转采用等权矩阵的 polar / chordal SO(3) 均值，各通道严格使用同一贡献集合。保存的世界锚点固定，不随新观测重新锚定旧预测。独立重建见 [wam_temporal_position_checker.py:60](../scripts/wam_temporal_position_checker.py#L60)；生产融合模块没有被该检查器导入。

identity 判定沿登记值 `atol=1e-12, rtol=0`；所有 pose / opening 一致时复制最新参考，避免对本来相同的世界目标额外做数值重组。非 identity 分支要求均值矩阵 determinant 为正且最小奇异值至少 `1e-6`，否则拒绝。原始 128 维输出、原 anchor、最新参考诊断、实际控制命令与控制器状态全部继续保存。该方法是显式的 reference intervention，`raw_checkpoint_reference=False`。

## 真实物理结果

| 正确参考场景 | 实际端点 / query | 位置 RMSE (mm) | SO(3) RMSE (rad) | 末尾连续原生成功端点 | 末 40 步稳定成功 |
| --- | ---: | ---: | ---: | ---: | --- |
| bowl / task0_init0 | 600 / 75 | 1.838090 | 0.003339389 | 531 | 是 |
| wine / task1_init0 | 600 / 75 | 1.421041 | 0.006743932 | 454 | 是 |
| cream / task2_init0 | 600 / 75 | 1.547627 | 0.002594670 | 501 | 是 |

分母仅为这三条 oracle 正控。接触、抬升和夹爪开度是离线诊断，不替代原生 predicate，也未作为在线规划或停止规则。上述跟踪误差以实际 canonical TCP 相对实际 admitted reference 计算，不是 pose 代数重建误差或 action RMSE。

独立检查保留 014 原有 query / 原始响应来源与完整控制机械门禁；修改范围由 AST 源码证明限制为固定 020 合同、参考身份标签、独立 ensemble target / contributor 重建和新增诊断。所有其它 case AST 节点保持原样，query verifier AST 完全相同，见 [wam_temporal_position_checker.py:141](../scripts/wam_temporal_position_checker.py#L141)。逐步核验包括实际 arm response、控制器 goal、标量 gripper 状态机、motor 积分及零命令保持、qpos / opening、查询期间物理暂停和控制时钟。

## 原严格失败与显式补充边界

原冻结资格入口还额外要求非 wall-time 的全部 chunk token canonical 精确一致，实际结果因此为 **`qualified=False`**，原产物与源码均保留。全部 225 个 chunk 的唯一差异为：旧 007 配置的 `model_rotation_alpha` 是 JSON **int `1`**，登记 020 配置是 JSON **float `1.0`**。逐字段检查没有发现任何其它非 wall-time token 差异；全部 1,800 条共同物理日志仍精确一致。

oracle 分支直接将真实参考复制到 `effective`，见 [temporal_ensemble_simulation.py:284](../src/s2a/temporal_ensemble_simulation.py#L284)；差异标签写入 chunk metadata 的位置见 [temporal_ensemble_simulation.py:315](../src/s2a/temporal_ensemble_simulation.py#L315)。这个局部事实不能推广为 WAM 分支永远不使用 alpha：生产 WAM 分支仍在同文件第 278 行调用旋转处理。

实现后的显式只读补充入口 `e56777a` 重新核验原 401 项输入、全部真实物理门槛、1,800 条共同日志、225 条 chunk、实际 wait 和四个进程身份，**仅允许该字段 exact float `1.0` / exact int `1` 类型对**；bool、其它数字、任何其它字段的微小数值变化、signed zero 或类型变化均拒绝。没有静默转换 JSON 数字，没有新增数值容差，也没有重跑物理。边界见 [audit_wam_temporal_qualification_alpha_metadata.py](../scripts/audit_wam_temporal_qualification_alpha_metadata.py)。

补充产物使用一般资格 schema，但同时明确保存 `audit_variant=registered_physics_with_alpha_metadata_supplement`、`qualification_scope=registered_physical_criteria`、`original_strict_qualified=False`、`additional_chunk_token_exact=False`，并引用原 False 产物完整 SHA。每个新 case 只报告 `registered_physical_qualified=True` 与原 strict case False，未将原 False case 混入新 True map。新 `qualified=True` **只对应预先登记的物理资格**，不能解释为原严格 chunk token 检查通过。

## 固定证据

以下路径均相对 `outputs/wam_stability_20261021/`；生成产物留在 outputs，报告与源代码纳入 Git。

| 产物 | SHA-256 |
| --- | --- |
| `temporal_equal_preflight/oracle_root_launch.json` | `32303169e6c762b9944c927fd4c5555e0ae79d47246a86349fd894fedf10722b` |
| `temporal_equal_preflight/source_review_b55c6fa.json`（373 输入） | `f726cda87b793ce08f5cc18908da1fb77a451e67d27aaf7b9ede32af7a6ec1b3` |
| `temporal_equal_qualification/qualification_independent_verification.json`（原 strict False，401 输入） | `44e989bb78dc4251831190ae0f5ece965492aef91d4eced9d85f76523ed7bec6` |
| `temporal_equal_checker_review/actual225_chunk_alpha_type_differences.json` | `c4db0d5236c4308e68c818c45c2543934eca08205f72cd9270b21ebc9091ebff` |
| `temporal_equal_qualification/qualification_registered_physical_with_alpha_metadata.json`（注册物理资格 True，403 输入） | `ff48de95ec7aabe5eda11a76cc494fe2ca54d1c36ad6fbaba41c314cd00f8e33` |
| `temporal_equal_review/019_020_actual_third_crosscheck.json`（独立重核实体日志与 225 类型对） | `0cc2543a14a0347b4ec427e05d04a60ef38e632a0e3f652c05f6311c6d0f9a45` |
| `temporal_equal_review/registered_physical_sidecar_independent_review.json`（原输入 map 未变） | `915d8cd194aec4c84eff8c6a136db7a07b70ddb4d3f4a057b34feaa8ffb44a94` |

类型差异证据中的 `oracle_effective_is_raw_copy` 引用行号误记为 289；实际对应源码是 **284**。该引用笔误不影响保存的差异实体、SHA 或评分，不覆盖原证据。

实现资格检查器 / 编排器的 12 项针对性 CPU 测试，以及新增 metadata supplement 的 10 项边界测试均通过；相关源文件 Ruff check / format 通过。生产算法的几何与身份路径测试由独立实现者完成，其缓存 007 的 1,800 端点 identity replay 仅作为 0 新物理的前置证据，实际资格以上述唯一真实 3 × 600 为准。

后续 WAM 验收必须保持独立的三条开发分母、原生末 40 步稳定判定、225 次真实服务来源 / HTTP 预算和正常释放，且区分融合参考跟踪误差、最新原始参考误差与任务效果。最新 query 游标每次只前进 8 步，不能据此断言所有历史预测后缀都未被使用；ensemble 必须按同绝对端点的实际 contributor 另行统计。这里不提前登记 WAM 成功或 fresh 验收通过。
