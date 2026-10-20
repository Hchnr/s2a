# 010：固定短间隔重新观测的 WAM 开发评测

**尚未全部完成**：当前独立核验 11/12 条，原登记的最后一条继续执行。原始完整 WAM 在每 8 步重观测的 bowl → plate 上首次满足 600 步、末 40 步连续原生成功；同组 wine、cream 均失败。这个结果证明当前 checkpoint 的原始输出在一个旧场景上具备任务能力，还没有达到多任务稳定成功。

证据等级为 LIBERO/MuJoCo 独立闭环仿真，使用 Panda 机器人模型；没有运行真机。

## 固定协议

[010 预注册](../plans/v0.0.1/010_short_replan_wam_20261021.md) 固定 8400 masked checkpoint、guidance=1、35 sampling steps、shift=5、SDPA、H40、20 Hz，每条 600 物理步。每 8/16 步重新观测，比较完整 raw 朝向 alpha1 与朝向分量消融 alpha0；四组各三个旧 task/init0/seed42，共 12 条、7200 步、最多 678 个 H40 请求。不计模型等待时间，等待时物理不推进。无 governor，沿用 v3 夹爪与原反解参数，没有按任务挑选策略或提前终止。

GPU4/9444 执行 interval8，GPU5/9445 执行 interval16；每个私有 alpha1 服务依次供给对应两组 CPU 配置。alpha0 的服务输出仍存于 alpha1 服务根，逐条绑定 server `sample_args.json`、`sample_outputs.json` 与 CPU 保存 response；朝向干预发生在收到 raw 输出之后。所有 CPU/OSMesa 进程 CUDA/LD 为空。

alpha0 改变局部旋转及其后积分得到的 world 位置，是分量诊断；不能把它当作原始完整 checkpoint 的成功。已有 oracle 只作为执行资格引用，不计入 12 条 WAM 分母。

## 当前完整案例结果

| 重观测间隔 | 模型旋转 alpha | bowl → plate | wine bottle → rack | cream cheese → bowl | 完整组稳定成功 |
| --- | ---: | --- | --- | --- | --- |
| 8 | 1，raw | 成功 | 失败 | 失败 | 1/3 |
| 8 | 0，分量消融 | 失败 | 失败 | 待完成 | 未统计 |
| 16 | 1，raw | 失败 | 失败 | 失败 | 0/3 |
| 16 | 0，分量消融 | 失败 | 失败 | 失败 | 0/3 |

依据 `short_replan_audit/progress_11.json`，SHA `3af07f8811b523b0f635d7ac6c227d6a1b30defb4fb7fdebdc61affd60949dab`。完整案例均通过冻结独立检查器的 600 步重放；partial/missing 没有算作失败，没有给未完成组成功率。interval16 自有服务已经正常释放，interval8 等待最后一条；整体 `complete=false`。

## 第一条 raw WAM 稳定成功的证据

每 8 步、alpha1、bowl 完整消费 600 物理步/75 次 H40 请求。完整 PNG、初态/anchor、首 query 的 raw 与 normalized 40×128 输出均与同 8400 旧实验精确一致；后续只用当前可用观测。逐步原始/有效/实际参考、命令、夹爪、模型/物理时钟、源 SHA 和原生成功日志均通过独立审计。

- 原生成功事件区间 `[338,600)`，262 个端点，末 40 步全部成功，终点成功；第一成功端点物理时间 16.95 s，末端点 30 s。
- 实际 TCP 对执行参考：位置 RMSE 7.029629 mm，SO(3) RMSE 0.02118912 rad，opening MAE 0.03680525。
- 位置误差中位数 3.02517 mm，p95 15.1287 mm，最大 37.3337 mm。
- 日志中目标的双指 `grasp_contact` 端点数为 0，因此只确认原生任务目标完成，不声称复现专家抓持方式。

独立事件与几何复算在 `short_replan_audit/first_raw_bowl_verified_event.json`，SHA `d3f50ac334028fee350c5b951a2fd6ccd27b5ad0a1ac3abe11e912d47f8d8dd6`。root 与独立审计方均实际查看原始 query 的初、末 PNG，bowl 由盘外进入盘内，与原生目标一致。

已有观测预览：`short_replan_visuals/first_raw8_bowl/raw8_bowl_query_images.mp4`，manifest SHA `bcbb4581fe51bf1dfb9ebedb64c44d688ff73aa0aa0deabde294ecf00141c659`。该 30 秒视频由原 75 张 query PNG 以 2.5 fps 保持采样拼接，末张对应 step592；它不是连续摄像或 step600 图像。600 步完整运行与末段 262 步持续成功由完整日志证明。没有新增 render、模型请求或物理步。

## 验证方法与剩余工作

只读入口 [audit_wam_short_replan.py](../scripts/audit_wam_short_replan.py) 直接复用冻结 006 的完整 case checker，并增加 010 的固定预算、四配置、共享服务来源、准确 query 步号和外部进程释放校验。其源码 SHA 为 `03e19eb1b5c4cf8bb591ecf224a80966d355d11b1352a3b687ba9a65a13a5177`，代码提交 `8a7396a`。原真实执行器、服务、配置和该审计源码均保持字节不变。

全部 12 条完成并正常释放后，输出完整 canonical 审计，再用独立 [audit_wam_short_replan_outcomes.py](../scripts/audit_wam_short_replan_outcomes.py) 复算所有案例的原生事件区间/末40、位置/朝向/opening 跟踪、饱和与速率限制、目标接触/抬升，以及参考和实际 TCP 相对目标中心的距离。参考偏离与执行约束可以同时存在；不凭单项误差唯一归因模型或控制器。

新 016 自适应重观测使用独立注册、实现与分母，不会改变本轮 600 步成功门槛。预留新初态仍未用于性能评测。

所有产物路径相对 `outputs/wam_stability_20261021/`。
