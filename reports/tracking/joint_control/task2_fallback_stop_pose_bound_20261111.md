# 因果回退停止状态的首点位姿外界

2026-11-11。按[预登记](../../../plans/v0.0.1/tracking/joint_control/task2_fallback_stop_pose_bound_20261111.md)，
以[已独立重放的回退反馈](task2_causal_shift_fallback_20261111.md)中raw step33和
terminal058 step85实际停止状态，复用冻结模型首点位置/SO(3)联合外界。展开种子
为各分支最后一次已完成计划的后移版本；方向取该种子已记录的首点误差。

| 开发停止状态 | 首动作box/变化率 | 首周期安全放宽域 | 八周期安全放宽域 | 最强投影中位置 / 姿态权重 |
| --- | ---: | ---: | ---: | --- |
| raw step33 | **1.008605** | **1.008605** | **1.008605** | 0.75 / 0.25 |
| terminal058 step85 | 0.868780 | 0.887498 | 0.887498 | 0.25 / 0.75 |

若原首点位置≤0.01 m且SO(3)≤0.05 rad，任何给定非负权重和为1的归一化投影
均不超过1。raw在仅保留首动作box及相对上一实际动作0.25变化率的放宽域中已有
1.008605下界，因此在这个冻结模型内即使忽略关节、制动、终端和未来周期约束，
也不能同时达到首点双阈值。纯位置投影下界约1.008464，即约0.0100846 m。
该余量只有约0.85%，必须考虑模型误差；它不证明真实机器人或仿真物理不可达。

terminal058最强下界仍小于1，当前五个方向及三个域**不能判定**首点可不可行，
也不证明存在可行解。此前缓存仅是有限搜索；相应地，raw更值得研究停止之前的
因果状态/参考衔接，terminal058仍可研究其他首点候选。两状态属于同一开发初态
和seed43，不能推断跨初态泛化。

三个域依次加入原0.05 rad缓冲、已完成误差收紧、首周期或八周期关节及0.1 s制动
约束；均移除终端方向和零旋转latch。使用旋转指数二阶余项及非负LP对偶、box残差
修正，额外扣除`1e-7`归一化数值余量；普通浮点结果不声称形式化区间证明。

## 独立复核与费用

后移计划的两个首点由独立标量关节响应重算，关节仿射差为0，均位于所检查放宽域。
30个支持证书逐项复核；20次LP最大原始约束违反约`4.44e-16`，最大支持上界与
原始目标差约`3.43e-14`。实际使用2个静态模型、2次`mj_forward`、2次八步rollout、
20次支持LP及2次独立首响应/50模型转换步；2个CPU worker及监督进程实际wait/reap、
exit0、未发信号，输入和源码前后一致。

49项32并发CPU专项、全仓Ruff/格式和零模型dry通过；本轮无新物理积分、新WAM、
GPU或最终留出样本。`complete/passed`只表示外界分析与证书复核完成，不授予控制
或任务资格，也不改变原实际误差和稳定任务验收门槛。

| 仓库内证据 | SHA256 |
| --- | --- |
| `outputs/task2_fallback_stop_pose_bound_20261111/result.json` | `194226209fbc97c72b6c65e4eb9929f21bd955bdd2bcd5a9913b926b051da55b` |
| `outputs/task2_fallback_stop_pose_bound_20261111/workers.json` | `dd56a824f709ada1eb5720127fa469f8eceecb418764201ed65dcfc3290240f8` |
| `outputs/task2_fallback_stop_pose_bound_dry_20261111/result.json` | `3f42da08638bc4183c6464554369f75cb61191f85607211d625e9492aaf61b98` |
| `outputs/qa/task2_fallback_stop_pose_bound_20261111/result.json` | `a3b2f907f5f125581648e9455aaf7f2a5d1ef5142a98e47fe16faa60586ec99e` |
| `outputs/task2_fallback_stop_pose_bound_supervisor_20261111/owner_exit.json` | `ea773eaf619e005328a2589c0c53af2feaa98ef118e81c4cb31b596a7900da1a` |
