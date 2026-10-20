# 017：1200 步执行接口资格

三条 native 专家命令和三条 privileged oracle state 参考均完成 1200 个实际控制端点，末 40 个端点连续原生成功。Oracle 的 TCP/SO(3) 跟踪误差满足预注册的 10 mm/0.05 rad 门槛，实际控制前 600 步与对应旧记录逐项完全一致。这个结果只证明本轮三个旧场景的 60 秒执行接口资格，**不计为 WAM 成功，也不是新初态或真机验收**。

正式物理核心资格：`outputs/wam_stability_20261021/long_raw_preflight/qualification_physical_core6_verified.json`，SHA256 `ba8872c55c9ab8b170a1b36efbe7cfdf501c1528cdd7dd74449adfa971663ade`。其中 `qualification_passed=true`、`normal_process_release=true`，三条 native 加三条 oracle 共 7200 步，282 个输入指纹前后重核。

| 旧场景 / seed42 | Native 末端连续成功步数 | Oracle 末端连续成功步数 | Oracle TCP RMSE (mm) | Oracle SO(3) RMSE (rad) |
| --- | ---: | ---: | ---: | ---: |
| task0_init0：bowl → plate | 1132 | 1131 | 1.299748 | 0.00236143 |
| task1_init0：wine → rack | 1059 | 1054 | 1.004846 | 0.00476871 |
| task2_init0：cream → bowl | 1108 | 1101 | 1.094338 | 0.00183471 |

每条为 20 Hz、1200 步，实测物理时长约 60 秒。新 factory 显式 `horizon=1300`、`ignore_done=false`；实际 policy timestep 从 0 到 1200，没有增加初始化控制步。Native 按原专家命令执行，结束后保持原有“零 arm command + 最后 gripper command”约定，不能将其描述为世界坐标中的严格位姿保持。新 oracle 仅从本轮实际 native 1200 步 state 记录读取参考，不填充旧 600 步日志，不发模型请求。实现源固定为 `d6fb4a6765a5223a2108a9f9e1bbf50c34a9dff8`。

旧 600 步对照保留两种不同结论：

- 实际 state、desired 控制目标、command、controller goal、motor/qpos、物体与原生成功谓词等前 600 步的规范化 JSON 值逐项完全相同。三个 native 的全部共同字段也完全相同。
- Oracle 的原严格“所有日志字段完全相同”结果仍为 **False**：三个场景各自仅在 step599 的 `raw_desired_tcp`、`effective_desired_tcp`、`raw_model_tcp_error`、`effective_model_tcp_error` 四个诊断字段不同，最大绝对差 `2.220446049250313e-16`。原完整机械审计文件保持原字节：`qualification_full6_verified.json`，SHA256 `c86569abeb1fb7e57bfd08468cc5e02cf598ef9c51f75a0da66b54e73e95357c`。

这四个字段来自诊断采样函数；[函数说明与分支](../src/s2a/stability_recovery_simulation.py#L98) 明确它不是控制输入。旧 step592 的 chunk 只有 8 个可用端点，诊断时钟 cursor8 走 index7/fraction1，执行一次 SO(3) 重组合；新 chunk 有 40 个端点，同一时钟走 index8/fraction0。独立 SciPy 复算记录了实际残差：pose 最大约 `1.11e-16`、误差向量最大约 `3.48e-17`，按原接口 `1e-10` 容差一致，**不声称派生重算 bit exact**。实际控制直接读取 [targets[index]](../src/s2a/long_simulation.py#L331)，经 [converter](../src/s2a/long_simulation.py#L356) 后 [执行 command](../src/s2a/long_simulation.py#L370)，没有消费这些诊断值；独立 AST 也核对它们只有日志消费者。

前 75 个 oracle query 的共同字段及旧记录实际存在的原始端点完全一致。旧 step568/576/584/592 分别只有 32/24/16/8 个未来端点，新参考均有 40 个；没有把旧记录不存在的未来 suffix、执行端点数量或生成墙钟当作应相同的数据。这个差异没有改变前 600 步的实际控制输入或状态。

补充资格只将上述严格物理核心一致性与派生诊断差异分开，保留原 False 结果及已完成的 1200 步机械重放、任务谓词、跟踪门槛。没有新增评分容差，没有修改计划、原 launch、旧 checker、物理实现或任何原始轨迹。任何额外字段、其他 step 的差异，或实际 command 的微小变化，都会被拒绝。新 sidecar 的 8 项定向 CPU 测试通过，包含额外实际控制差异、未知字段、错误 step 和诊断重构越界的拒绝测试。

审计过程中另保留了两项收据身份修正：native launch 保存了同一运行 commit 的短 SHA；oracle launch 漏写了 execution map 中重复的计划项，但其启动前已绑定的 23 输入来源审查包含完全相同的计划 SHA。两者都通过独立窄入口验证精确身份，原 launch 不变；它们不是任务失败证据。来源审查、旧 checker 和完整评分函数保持冻结。

可复核来源：

- `long_raw_implementation/independent_source_review_final.json`：SHA256 `90622c3d181afc997e6f5e0c99688c960ae303c3a7953116adbc60cdfa7be8ae`，23 个启动前输入。
- `long_raw_implementation/endpoint599_independent_causality.json`：SHA256 `05ba5ae30c9ec826faf906b621868bb7e051b8f569077f05e0a0c0f58e25e6bb`，独立 endpoint599 因果审查。
- `long_raw_implementation/physical_core_27be68d_independent_review.json`：SHA256 `12c46a34bc98c0157dca30db5e2d72e68f116cdc4a3946473dc55ea8883404ec`，最终 sidecar 互审。
- 上述相对产物路径均位于 `outputs/wam_stability_20261021/`。原 native/basic、strict False、失败尝试与元数据审查收据继续保留。

两个 CPU coordinator 和六个 worker 的登记 PID 均已释放；登记环境 CUDA/LD 为空，OSMesa 执行。此资格与审计新增模型请求均为 0。017 后续三个 1200 步 WAM 开发场景仍须由独立 coordinator 的来源、先前任务释放与 GPU 门禁准入，不能从本资格推断它们已执行或成功。原 600 步 WAM 结果、旧场景开发范围和 fresh 验收要求均保持不变。
