# LIBERO 独立仿真与 checkpoint 对照（2026-10-16）

目标：冻结 Cosmos 世界模型，只改变 observed-local-delta → OSC command 的 S2A 转换，在相同任务、初始状态、推理参数和物理控制器下比较跟踪误差及任务成功。

## 已定位的历史证据

- Cosmos 只读 checkout：`/share/project/eai_pwm/home/hcr/repos/cosmos-framework-pwm`。
- `outputs/action128_eval_20261014/summary.json`：LIBERO goal task 8，put the bowl on the plate；episode 43/69/72/80/132，seed 42+trial，1000 步，chunk16/execute8，20Hz，35 采样步、guidance1.5、shift5、无 warmup/EMA，0/5 成功、无执行错误。
- 历史使用 warm-start run 的 `iter_000005600/model`，并非旧 `action128_live` 的6000步。当前历史5600目录已清理；同一 run 的完成指针为8800。新增 baseline 和 compensation 必须使用同一个冻结 checkpoint，不能把不同 checkpoint 的结果直接归因于补偿。
- `outputs/action128_eval_20261014_10000steps/cancelled.json`：延长实验被用户取消，不视为完整10000步结果。
- 历史 `transfer/implementation/run_action128_libero_eval.py` 已包含固定锚点 TCP 积分、TCP→source、OSC affine inverse 和夹爪位置转换；几何反解本身不是这次新增项。
- 历史 native expert replay 成功，observed-motion oracle replay 失败，故需先隔离验证控制跟踪问题，不能只归因于世界模型。

## 保守决策与执行顺序

1. 所有新增代码、配置、环境、日志、模型 serving 输出和仿真产物留在 S2A。Cosmos、数据和 checkpoint 只读；不终止其他进程，不改共享环境。
2. 审计 LIBERO/robosuite OSC 的实时反馈点、旋转误差、限幅及20Hz执行时钟，不移植 DROID 的0.075/0.15缩放、线性模型或PID参数。
3. 复现原始 baseline；增加独立 S2A OSC adapter，以当前反馈、已生成参考及因果历史输出补偿，保持机械臂/夹爪物理控制器不变。
4. 先做 native expert、oracle state reference 和独立小幅自由空间 probe；参数若需标定只使用单独 probe，不用5个任务结果调参。明确 oracle 和新模型预测的区别。
5. checkpoint 以同一 run 的完成指针冻结，保留分片元数据、配置和源码指纹；服务仅使用可用资源。两候选同样1000步/5初始状态及推理参数，记录任务成功、末端state跟踪、饱和、命令变化、原始预测和执行命令。
6. 比较可重放的同一预测轨迹与真实图像反馈下重新预测两层结果；后者允许后续预测因执行状态不同而改变，不重用旧分支图像预测冒充闭环。
7. 持续更新 record/conclusion/report 并分阶段提交；明确失败、未执行、checkpoint变化及证据限制。
