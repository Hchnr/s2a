# 016：因果跟踪误差重观测的执行资格

新增执行路径在三个旧场景的正确参考上 **3/3 通过**。这证明它保留了已有正确参考的执行能力；不构成 WAM 成功或新初态泛化证据。WAM 仍需使用独立的三条开发预算评估，不能把这三条 oracle 计入 WAM 分母。

证据等级为 LIBERO/MuJoCo 独立闭环仿真，使用 Panda 机器人模型；没有运行真机。

## 固定协议与实现

协议见 [016 计划](../plans/v0.0.1/016_tracking_replan_20261021.md)。每条 600 个物理步，20 Hz，H40，原始朝向和 opening，无 governor，沿用 v3 夹爪和原 S2A 参数。每次至少消费 8 步、最多 40 步；已完成端点的位置误差严格大于 40 mm 或 SO(3) 误差严格大于 0.4 rad，连续两步后在下一步重新观测。相邻两步可由不同误差分量超限；最少间隔之前也累计，query 和正常端点清零计数，arm/gripper 历史跨 query 保留。

新物理函数是冻结 position 执行函数的显式副本，仅改变 query 时机、协议校验、版本与新增因果日志。几何、生成参考、命令反解、夹爪、控制时钟和原生计分计算保留。独立源码差异审查、11 项 CPU 边界/因果测试及三源实际读取预检均通过。

新增只读检查器 [wam_tracking_replan_checker.py](../scripts/wam_tracking_replan_checker.py) 字面继承冻结 014 的完整 case 检查，只替换 query 判断并增加新来源/schema/因果字段；AST 差异证明保存在资格产物中。它不导入生产 `TrackingReplan`，而从独立重建的实际 TCP 端点误差计算计数与下一步决策。014 的请求来源、SO(3)、反归一化、机械、逐步 gripper、模型/物理时钟、初态 SHA 和原生成功检查均保留。

## 实际结果

资格使用三个原 task/init0/seed42，正确参考来自已有成功 native 轨迹。三个进程均完整运行 600 步，末 40 步原生连续成功，位置 RMSE ≤10 mm、SO(3) RMSE ≤0.05 rad。

| 场景 | TCP 位置 RMSE (mm) | SO(3) RMSE (rad) | 原生成功端点数 | 末尾连续成功步数 | query 数 |
| --- | ---: | ---: | ---: | ---: | ---: |
| bowl → plate | 1.838090 | 0.003339389 | 531 | 531 | 15 |
| wine bottle → rack | 1.421041 | 0.006743932 | 458 | 454 | 15 |
| cream cheese → bowl | 1.547627 | 0.002594670 | 501 | 501 | 15 |

每条均为一次 initial query 加 14 次 reference exhaustion，**没有 tracking-limit 触发**。全部 1800 个新端点均通过机械与因果重放。移除新增的 `query_decision` / `tracking_replan` 字段后，1800 条 step 的所有共享字段与原无 governor、H40 正确参考执行精确相同；45 条 chunk 排除新增决策和墙钟 `generation_s` 后也精确相同，最大数值差均为 0。声明的比较容差为 `1e-10`，未用于放宽成功门槛。

## 来源、预算与进程释放

首次准备配置遗漏 `oracle_reference_root`，三个 worker 均在 `make_environment` 之前以同一 KeyError 退出。没有创建环境，也没有物理步或模型请求；原配置、日志、失败 matrix 和 SHA 均保留，没有覆盖。

root 在 `68fa982` 提交独立 v2 配置与新输出根，仅补充 oracle 来源、输出路径和勘误，并在 CPU dry-run 中实际读取三条参考。算法和控制器字节不变。独立复核后以两个 CPU worker 启动原登记的三条/1800 步预算，没有增加物理额度。CPU/OSMesa 的 CUDA 和 LD 均为空，未启动模型。

| 证据 | SHA-256 |
| --- | --- |
| 原零环境失败 `tracking_replan_qualification/orchestration/pre_environment_failure.json` | `9bab897ddf7d1645fd5a3697de7977f77c68980d6a00f8c0f2805cf17b5baea1` |
| v2 独立启动 `tracking_replan_qualification_v2/orchestration/independent_launch.json` | `33c4989e829cc339c790767e6bbb77fdf0b4b5a58c8b6c4c0f42ceff3d782ebe` |
| v2 独立增量审查 `tracking_replan_qualification_v2/orchestration/independent_source_review.json` | `853ddb63cd784508582ca4f259e3a55e6904cca63682fb3e54192295a5019dda` |
| 完整资格 `tracking_replan_qualification_v2/qualification_independent_verification.json` | `353d7669f8060f99b6099f25b458394d04457cbf77b2845fd562ff2d58cf1a49` |

上述路径相对 `outputs/wam_stability_20261021/`。完整资格有 1546 项输入 SHA，审计前后均一致。matrix PID 279501（启动 ticks 167961364）及三个 worker PID 279513、279514、281214 均已退出；三个 worker returncode=0，matrix complete=true。检查器及资格入口代码提交为 `d84460a`，独立检查不创建任何模型或物理步。

另一审查方独立通读 `d84460a`、重跑 literal AST 证明、复核 1546 项来源 SHA 及独立时序边界，没有发现 blocker。review receipt 为 `tracking_service_review/qualification_d84460a_independent_review.json`，SHA `b8161bb0bf7fe56b5dc0b69a01d7b4d4871655d20b6914e40fec924c34d4d3ad`；这项复核没有重复运行模型或仿真。

## 可以与不可以推出什么

- 新时序执行路径在这三条低误差正确参考上保留了原执行效果，逐步命令和物理轨迹没有改变。
- 本次真实物理轨迹没有触发误差提前重观测；严格边界、混轴连续、min8/max40 和重置由 CPU 测试及独立边界检查覆盖，仍需在实际 WAM 日志上完整复核触发行为。
- 不能据此声称更长参考消费改善了 WAM、失败已唯一归因闭合后缀被截断，或已稳定完成多任务。
- 没有打开预留新初态性能、没有修改原 010 分母和门槛，也没有新增 WAM 采样。

只读重算：

```bash
CUDA_VISIBLE_DEVICES='' LD_LIBRARY_PATH='' PYTHONDONTWRITEBYTECODE=1 \
  .venv/bin/python scripts/audit_wam_tracking_qualification.py \
  --output outputs/wam_stability_20261021/tracking_replan_qualification_v2/qualification_recheck_NEW.json
```

输出必须使用新文件名，原资格及失败证据禁止覆盖。
