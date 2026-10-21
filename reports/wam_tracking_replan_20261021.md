# 016：按实测跟踪误差重观测的完整 WAM 结果

**三条旧场景均未完成任务，稳定成功 0/3，曾成功也为 0/3。** 全部 1800 步、97 次 H40 请求已通过独立来源、机械、因果、原生结果和服务预算审计，模型正常释放。新时序策略保留了更长的生成后缀，但这次没有取得多任务成功，也没有复现固定每 8 步策略的 bowl 成功。

证据等级为 LIBERO/MuJoCo 独立闭环仿真，Panda 机器人模型；没有运行真机。以下结论限于三个旧 task/init0/seed42 的开发场景，没有打开预留新初态性能。

## 协议与变化

[016 计划](../plans/v0.0.1/016_tracking_replan_20261021.md) 固定同一 22src `iter_000008400`、非 EMA 权重，guidance=1、35 sampling steps、shift=5、SDPA、active-slot mask、20 Hz/H40。checkpoint 的完整 run 和路径见 [WAM 配置](../configs/tracking_replan_wam_20261021.json)。沿用原 S2A 反解和 v3 夹爪，无 governor、无朝向衰减、不改任何生成 pose/opening。

每 query 正常按 20 Hz 消费参考，至少 8 步、最多 40 步；已完成端点的位置误差严格大于 40 mm 或 SO(3) 误差严格大于 0.4 rad，连续两步后在下一步重观测。混轴 OR 连续、min8 之前累计、query/正常端点重置均已冻结。它只读已发生的 TCP 跟踪误差和消费年龄，不读取物体、接触或成功谓词。推理等待时物理不推进，不计推理延迟。

三个正确参考资格在同一新执行器上 3/3 通过，全部 1800 共享物理字段与原 H40 正确参考精确相同。资格仅证明这一执行链路保留正确参考能力，不能计入 WAM 成功。来源、零步配置勘误和资格复核见 [资格报告](wam_tracking_replan_qualification_20261021.md)。

## 全部结果

每条完整 600 步，末 40 步连续原生成功为固定验收条件。三条的 600 个原生成功标志均为 false。

| 场景 | H40 请求 | 误差提前重观测 | TCP RMSE (mm) | SO(3) RMSE (rad) | opening MAE | 稳定成功 |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| bowl → plate | 26 | 22 | 25.117075 | 0.106407 | 0.061051 | 否 |
| wine bottle → rack | 40 | 37 | 24.892264 | 0.063054 | 0.052837 | 否 |
| cream cheese → bowl | 31 | 27 | 25.040897 | 0.084950 | 0.063750 | 否 |

共 86 次 tracking-limit、8 次 reference-exhausted、3 次 initial query，合计 97，低于登记上限 225。原始参考消费更长：每段平均实际消费约 23.1/15.0/19.4 步，含最后预算截断的参考段；它仍没有解决本轮任务失败。每个真实 query 的原因、前一端点计数、年龄和全部 1800 个 after-TCP 误差均已独立重建。例如 bowl 的 step88 在年龄8、此前连续3次超界时触发，真实覆盖了 min8 之前累计的分支。

| 场景 | 位置误差 p95 / 最大 (mm) | 命令饱和 / 速率限制步数 | 目标双指接触端点数 | 目标最高抬升 (mm) |
| --- | ---: | ---: | ---: | ---: |
| bowl | 46.097 / 132.612 | 354 / 469 | 13 | 57.958 |
| wine | 51.645 / 69.999 | 367 / 462 | 0 | 2.582 |
| cream | 46.828 / 76.608 | 364 / 442 | 0 | 14.051 |

这些约束和参考偏移同时存在。例如参考 opening≤0.6 的 wine 端点中，参考 TCP 对目标物体中心的 XY 距离中位数为 311.361 mm，而实际 TCP 对参考的位置跟踪误差中位数为 16.212 mm。这里的中心距离只是事后描述，不是抓取位姿、任务奖励或新阈值；不能从这一列唯一归因权重或控制器。wine 在本轮没有目标双指接触，也不能套用此前固定8步的“末端仍持物、可能只差时间”解释。

## 独立核验与进程释放

冻结 [case 检查器](../scripts/wam_tracking_replan_checker.py) 字面保留 014 的完整机械/来源检查，新增独立因果时序；[WAM wrapper](../scripts/audit_wam_tracking_replan.py) 提交 `aa73ea5`。所有首 query 的 PNG、初态/anchor、raw 和 normalized 40×128 均与同 8400 旧 intent_hold 记录精确一致。每个后续 query 都绑定当前 CPU 保存的请求/response 和实际私有服务保存的 `sample_args.json` / `sample_outputs.json`。raw/effective/admitted/nominal 参考、逐步 arm/gripper 命令、状态连续性、时间间隔、暂停物理 SHA 和原生结果全部通过。

最终 [预算与释放 sidecar](../scripts/audit_wam_tracking_release.py) 不改上述冻结 wrapper，额外核验服务根的全部 97 个 `generate_*` 目录精确等于三个案例来源集合，service.log 的所有 POST 恰为 97 次 `/generate` 200，没有额外 POST。它绑定历史真实 ready snapshot 的 `/proc` command/cwd/PPid/startticks/CVD，模型仅 CVD7、CPU/OSMesa 空 CUDA/LD。NVML 与本地 PID 的未证实映射没有被猜补。

coordinator 292314、model 292482、matrix 293856 和三个 physics worker 293988/305020/320062 均已退出，外部 model returncode=-15、matrix=0、cleanup_errors=[]。实际服务日志也包含正常 startup/shutdown/finished-server 记录；没有据此补造内部 finally 收据。原 163 个 checkpoint 文件的 size/mtime 与既有完整逐文件 SHA 登记一致，没有在本轮重哈全部权重字节。

| 产物 | SHA-256 |
| --- | --- |
| `tracking_replan_audit/complete3_verified.json` | `76c1b1f7cd11dc5a9729dcd2dbf3c5af0d84762522f99e86b1eacc7a1fa4400a` |
| `tracking_replan_audit/complete_service_budget_verified.json` | `0fc386a83c9b95969ffc506839e90f0a89f6f5a0b734f96d1cec96239719add1` |
| `tracking_replan_audit/outcome_diagnostics.json` | `3fb3d5b1fefc7c5eb9ad645b1cf8f5385a96057d17385baf90f81e5ef9ff0d91` |
| `tracking_service_review/actual_ready_resource_snapshot.json` | `be69ea89b57844749603485e985d73c319fdf55145d1aff09b372a9233520012` |

路径相对 `outputs/wam_stability_20261021/`。完整 case 审计有 3206 项输入 SHA，最终 outcome 诊断有 3213 项，均审计前后重核。[诊断入口](../scripts/audit_wam_tracking_outcomes.py) 显式复用冻结 010 的几何/事件统计 helper，涵盖三条全部末40、成功区间、位置/SO(3)/opening、饱和/速率限制、参考目标中心距离、接触与最高抬升；不启动模型、仿真或 render。

另一审查方独立复跑最终 sidecar，并重核完整 case 审计的全部输入 SHA、历史 ready 来源、6 个已退出 PID、97 目录和 97 POST，结果一致；没有重复运行 1800 步或新增模型请求。交叉证据为 `tracking_service_review/complete_release_fc765b3_crosscheck.json`，SHA `92d2ef62d63b9e086b0883fc62213df678e7ae9d76da9e99e0f1b553473ee286`。

## 结论边界

该固定时序策略在当前三场景、30秒物理预算内失败，不能推广为稳定候选；它增加参考消费长度，没有自动修复抓取/搬运/放置。执行误差也高于正确参考资格，不能把本轮所有失败只归因 WAM 或只归因 S2A。完整 raw WAM 的现有正证据仍是另一独立 010 协议中的 bowl 成功；本轮没有增加正例。

后续长时预算或 temporal ensemble 必须使用新登记、代码与分母，不能追溯放宽这三条 600 步失败。任何新鲜初态验收仍需先冻结唯一候选和完整协议。本轮没有选择成功任务/seed、增加请求额度、训练校正器或引入任务规划器。
