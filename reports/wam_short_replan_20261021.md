# 010：固定短间隔重新观测的 WAM 开发评测

**12 条物理评测及新独立结果审计全部完成，存在明确的协调器收尾例外。** 原始完整 WAM 每 8 步重观测稳定成功 1/3，仅 bowl → plate；其余三组均 0/3。所有案例总计 7200 步、678 次请求，只有这一条出现原生成功。这证明当前 checkpoint 的原始输出在一个旧场景上具备任务能力，还没有达到多任务稳定成功。

原 interval8 协调器在最后检查一个本轮未使用的历史 15400 checkpoint 时发现外部文件缺失，`original_coordinator_final_gate_success=false`。实际使用的 8400 权重登记、执行源、全部案例和服务预算通过新的独立严格核验。原错误和缺失的 complete 标记保留，没有重跑或补造完成文件。

证据等级为 LIBERO/MuJoCo 独立闭环仿真，使用 Panda 机器人模型；没有运行真机。

## 固定协议

[010 预注册](../plans/v0.0.1/010_short_replan_wam_20261021.md) 固定 8400 masked checkpoint、guidance=1、35 sampling steps、shift=5、SDPA、H40、20 Hz，每条 600 物理步。每 8/16 步重新观测，比较完整 raw 朝向 alpha1 与朝向分量消融 alpha0；四组各三个旧 task/init0/seed42，共 12 条、7200 步、最多 678 个 H40 请求。不计模型等待时间，等待时物理不推进。无 governor，沿用 v3 夹爪与原反解参数，没有按任务挑选策略或提前终止。

GPU4/9444 执行 interval8，GPU5/9445 执行 interval16；每个私有 alpha1 服务依次供给对应两组 CPU 配置。alpha0 的服务输出仍存于 alpha1 服务根，逐条绑定 server `sample_args.json`、`sample_outputs.json` 与 CPU 保存 response；朝向干预发生在收到 raw 输出之后。所有 CPU/OSMesa 进程 CUDA/LD 为空。

alpha0 改变局部旋转及其后积分得到的 world 位置，是分量诊断；不能把它当作原始完整 checkpoint 的成功。已有 oracle 只作为执行资格引用，不计入 12 条 WAM 分母。

## 当前完整案例结果

| 重观测间隔 | 模型旋转 alpha | bowl → plate | wine bottle → rack | cream cheese → bowl | 完整组稳定成功 |
| --- | ---: | --- | --- | --- | --- |
| 8 | 1，raw | 成功 | 失败 | 失败 | 1/3 |
| 8 | 0，分量消融 | 失败 | 失败 | 失败 | 0/3 |
| 16 | 1，raw | 失败 | 失败 | 失败 | 0/3 |
| 16 | 0，分量消融 | 失败 | 失败 | 失败 | 0/3 |

依据 `short_replan_audit/final12_verified_with_unused_checkpoint_exception.json`，SHA `a4560bb70cc21e204dc7bd0e35fc3aa2a7cab7365b394a6a4c1727473b60ad12`。新 artifact 的 `complete=true` 仅指这项独立只读案例/请求审计；原协调器最终门禁仍为 false。11 条此前完整审计的所有来源 SHA 再核，最后一条执行完整 600 步机械重放；另对全部 12 条重新计算几何/事件/执行约束。两套模型均已退出、外部 model=-15/matrix=0，原 interval16 完成标记有效，interval8 收尾错误保留。

下面全部跟踪误差针对实际执行参考；alpha0 的低朝向误差不代表原始 WAM 朝向预测改善。

| 间隔/alpha | 场景 | TCP RMSE (mm) | SO(3) RMSE (rad) | opening MAE | 目标双指接触端点 | 饱和/速率限制步数 |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 8/1 raw | bowl | 7.029629 | 0.021189 | 0.036805 | 0 | 47/212 |
| 8/1 raw | wine | 6.406711 | 0.013904 | 0.038146 | 35 | 23/146 |
| 8/1 raw | cream | 11.360748 | 0.025491 | 0.034974 | 10 | 108/240 |
| 8/0 | bowl | 7.700130 | 0.001586 | 0.040577 | 0 | 30/119 |
| 8/0 | wine | 0.649828 | 0.001323 | 0.020225 | 0 | 0/8 |
| 8/0 | cream | 8.591747 | 0.002874 | 0.056289 | 0 | 71/310 |
| 16/1 raw | bowl | 20.502662 | 0.033162 | 0.053328 | 0 | 270/504 |
| 16/1 raw | wine | 40.503872 | 0.042709 | 0.023847 | 0 | 288/446 |
| 16/1 raw | cream | 46.576619 | 0.081509 | 0.052480 | 0 | 377/426 |
| 16/0 | bowl | 43.121043 | 0.049150 | 0.098896 | 0 | 435/171 |
| 16/0 | wine | 8.626749 | 0.001806 | 0.025770 | 0 | 25/32 |
| 16/0 | cream | 8.633714 | 0.008036 | 0.046574 | 292 | 71/199 |

低跟踪误差、双指接触或抬升都不等于任务完成。raw8 wine 虽跟踪误差约 6.4 mm、末端曾持物，仍未在 30 秒内完成；raw8 cream 曾抬升约 83.77 mm，但末端不再记录接触且开度接近 1。不能把两种失败统一解释为只差时间，也不能用较小动作/位姿误差替代原生任务判据。

## 第一条 raw WAM 稳定成功的证据

每 8 步、alpha1、bowl 完整消费 600 物理步/75 次 H40 请求。完整 PNG、初态/anchor、首 query 的 raw 与 normalized 40×128 输出均与同 8400 旧实验精确一致；后续只用当前可用观测。逐步原始/有效/实际参考、命令、夹爪、模型/物理时钟、源 SHA 和原生成功日志均通过独立审计。

- 原生成功事件区间 `[338,600)`，262 个端点，末 40 步全部成功，终点成功；第一成功端点物理时间 16.95 s，末端点 30 s。
- 实际 TCP 对执行参考：位置 RMSE 7.029629 mm，SO(3) RMSE 0.02118912 rad，opening MAE 0.03680525。
- 位置误差中位数 3.02517 mm，p95 15.1287 mm，最大 37.3337 mm。
- 日志中目标的双指 `grasp_contact` 端点数为 0，因此只确认原生任务目标完成，不声称复现专家抓持方式。

独立事件与几何复算在 `short_replan_audit/first_raw_bowl_verified_event.json`，SHA `d3f50ac334028fee350c5b951a2fd6ccd27b5ad0a1ac3abe11e912d47f8d8dd6`。root 与独立审计方均实际查看原始 query 的初、末 PNG，bowl 由盘外进入盘内，与原生目标一致。

已有观测预览：`short_replan_visuals/first_raw8_bowl/raw8_bowl_query_images.mp4`，manifest SHA `bcbb4581fe51bf1dfb9ebedb64c44d688ff73aa0aa0deabde294ecf00141c659`。该 30 秒视频由原 75 张 query PNG 以 2.5 fps 保持采样拼接，末张对应 step592；它不是连续摄像或 step600 图像。600 步完整运行与末段 262 步持续成功由完整日志证明。没有新增 render、模型请求或物理步。

## 验证方法、收尾例外与结论范围

只读入口 [audit_wam_short_replan.py](../scripts/audit_wam_short_replan.py) 直接复用冻结 006 的完整 case checker，并增加 010 的固定预算、四配置、共享服务来源、准确 query 步号和外部进程释放校验。其源码 SHA 为 `03e19eb1b5c4cf8bb591ecf224a80966d355d11b1352a3b687ba9a65a13a5177`，代码提交 `8a7396a`。原真实执行器、服务、配置和该审计源码均保持字节不变。

新增 [final wrapper](../scripts/audit_wam_short_replan_final.py) 提交 `763888e`，不修改原冻结执行器或审计器。它直接复用冻结 006 完整 case verifier、010 配置注册/服务来源 helper 和 [几何/事件统计 helper](../scripts/audit_wam_short_replan_outcomes.py)，不是动态替换或弱化旧检查器。最后一条新的完整重放与旧 11 条来源复核均通过，所有 6993 项最终输入 SHA 在审计前后完全一致。

严格保留本轮实际 8400 全 163 个文件的原完整 SHA 登记与 size/mtime 核验，旧 8800 全163、15600全99也仍在且 metadata 一致；3477 项原 prior gate 来源 SHA 均存在且未变。只有未用于本轮 inference 的历史 15400 原 99 项中有 98 项消失，原因 unknown。结束时原协调器在 `weight_stats(gate)` 读取其 `.save_complete` 失败，随后仍正常释放自己的模型；原 `error.json` / `release.json`、trace 和缺失的 `complete.json` 保持原状。没有写外部 Cosmos，没有恢复/伪造 marker，也没有重新采样。完整原始错误来源在 `short_replan_audit/final_unrelated_weight_stat_failure.json`，SHA `73506055b250979dbdf5a48bb13cddc7d48c0de7046d652edefc3766328d4310`。

新独立 final 审计仅允许这个明确的历史权重可用性例外；实际8400、任何执行代码/数据 SHA 或其它权重 metadata 变化仍拒绝。它逐一确认全部 12 案例、7200 步、678 次模型请求，interval8 的 450 个和 interval16 的 228 个生成目录全集分别精确等于两 alpha 配置的案例来源，service.log 所有 POST 恰为对应数量的 `/generate` 200，无额外请求。模型、matrix、coordinator 和所有 physics 子进程均已退出，模型/物理预算没有重跑。

另一审阅方固定 `763888e` 后独立完整 CPU 复跑也通过：`long_raw_service_review/010_final763888e_independent_rerun.json`，SHA `de21e08a1c6293cf079b38702f79bef6c57c99e2ed792472d6d103ae93088d62`。该复跑再次核验 6993 项来源、12/7200/678、最后 600 步机械重放、服务全集与外部释放，并保留原协调器门禁失败。其另外直接读取全部 7200 个原生 success 布尔值、绕过生产 outcome helper 复算 ever/terminal/末40 stable，唯一成功仍为 raw8 bowl；收据 `long_raw_service_review/010_final763888e_outcome_and_scope_review.json`，SHA `7c2ab5bb816477019c12ce2bbd6e4b59efea814e6ed1ccfcdb9f67cc7f77500e`。这些只读核验没有新增模型请求或物理步。

最终 artifact 内嵌全部 12 条原生事件区间/末40、位置/朝向/opening 跟踪、饱和与速率限制、目标接触，以及参考和实际 TCP 相对目标中心的距离。参考偏离与执行约束可以同时存在；不凭单项误差唯一归因模型或控制器。只有 raw8 bowl 存在成功区间，其余11条全部原生成功标志均为 false。

新 016 自适应重观测使用独立注册、实现与分母，不会改变本轮 600 步成功门槛。预留新初态仍未用于性能评测。

所有产物路径相对 `outputs/wam_stability_20261021/`。
