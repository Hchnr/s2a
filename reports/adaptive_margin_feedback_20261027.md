# 自适应正余量回退：完整原生反馈对照

2026-10-27。总目标未完成。本轮新增两个候选分支256物理周期，复用已独立核验baseline，不授予新控制资格。

## 实现及合同

`src/s2a/adaptive_margin_hybrid.py` 按固定顺序选择：原.05rad内侧见证→原TCP限额下的自适应正余量见证→软预防回退。自适应rho范围0至min(.05,当前实测最小余量)，仅rho>1e-6rad且复核通过才采用；在rho减1e-9rad显式数值缓冲下细化原跟踪目标。若无可行细化结果，保留已验证余量见证。

固定/自适应首步限额均由原当前free seed决定；软回退使用理想限位seed。实际XML边界、原动作范围、.25变化率和固定OSC/initial_joint/夹爪/20Hz-2ms设置不变。较小余量单独标作原.05目标未满足，不因选择新rho就宣称原约束通过。四求解阶段各60迭代/160响应，总上限640，分项记录；不是旧320预算的同成本对照。

`src/s2a/adaptive_plan_history.py` 只在实际首动作执行并完成观测后提交当前计划；移位计划只是数值初值，每次模型和约束均重算。reset/独立分支清空该状态。`src/s2a/adaptive_margin_native.py` 无风险时保持原命令，门控不使用任务标签。

## 配对实际结果

同一窗口72至128：72步原命令前缀，查询9/10各8步、查询11完整40步。只使用当前保存chunk，不读取未来chunk进入规划；这是固定参考回放，不是候选自身新图像驱动的世界模型闭环。

| 案例 / 方法 | 56步位置 RMSE mm | 旋转 RMSE rad | 首次q7超XML端点 | 超范围周期 | 抓取 / 成功步 |
|---|---:|---:|---:|---:|---:|
| raw baseline | 51.0314 | .178797 | 82 | 47 | 0 / 0 |
| raw adaptive | 39.1418 | .177194 | 89 | 40 | 0 / 0 |
| terminal058 baseline | 10.2800 | .036863 | 无 | 0 | 0 / 0 |
| terminal058 adaptive | 10.2800 | .036863 | 无 | 0 | 0 / 0 |

raw选择11次固定见证、2次自适应见证、41次软回退（6次保留seed）、2次无风险旁路。自适应所保留余量分别.042934788rad和.027683810rad；总体原.05目标未满足43周期。最终仍超XML范围，无抓取、无任务成功。

此前固定余量+软回退同窗口为38.2087mm/.176966rad，首次超范围89；本次两项误差略差且没有进一步推迟超范围，不支持推广新方案。最后40步本次46.1228mm/.208225rad，baseline60.2437mm/.210715rad；不得与56步均值混用。

terminal058全部56步实际命令、完整sim_state/hash、goal、夹爪、对象/任务字段与baseline精确一致。raw适配器平均2.4426s，terminal旁路.007867s；raw仍远未满足.05s周期，不能从与上一版的微小时间差宣称性能提升。

XML范围比较是诊断量，不等于限位反力证据或真实机器人失控判定。实际MuJoCo软限位、接触、力矩裁剪和变化的M/J不在冻结自由模型里。

## 失效链与下一方向

- 第85步自适应见证执行后，实测q7余量.0441840rad。
- 第86步又找到自适应见证，执行后余量.0311920rad。
- 第87步线性阶段成功，但加TCP限额的搜索以Positive directional derivative退出，未找到完整见证，进入软回退。
- 第88步当前余量.00709857rad，忽略TCP约束的线性冻结自由模型也返回status2；这仅是该模型下未找到保持XML范围的动作序列，不是真实非线性系统不可达证明。
- 第89步实测超出XML范围，后续主要进入原软回退。一次局部正余量见证没有形成连续反馈中的可行性保持。

关键状态与因果重构配方已冻结：`outputs/adaptive_margin_feedback_verified_20261027/failure_captures.json`，SHA256 `a09f03f33cc522aea4000322e7564e51746825ae2fb60fd2e0fc7994281ca14b`，分别绑定87/88步的当前/上一周期状态、执行命令、参考query/cursor、模型/工具来源及分阶段退出原因。

另发现可检验的有限预览末端问题：本次72步实际采用的第一条完整H8计划，与此前独立复算的模型见证计划最大差3.285e-14。该计划末端q7为2.8473rad，恰距XML上限.05rad，但末端qd7仍为+.566584rad/s、朝边界运动。单纯约束有限窗口内的位置，允许计划在窗口末端仍向外运动；此证据不能单独证明后续失效唯一由末端条件导致。

模型末端核对 `outputs/adaptive_margin_feedback_verified_20261027/terminal_motion_check.json`，SHA256 `513ec3ddf1051ccea2e794e73c4ca2416247ef1f359d9900e2d9fac987a5c9c7`，绑定原模型producer、独立模型验证和当前实际计划行，未用后续观测作为在线输入。

下一项应先在相同早期状态检查接近边界关节的末端不向外速度条件是否与原动作/TCP约束同时可行，再决定真实反馈实验；不能继续把“当前窗口内有正余量”视作后续可行性保证。若该附加条件没有见证，应分别记录数值失败及约束权衡，不改原任务成功门槛。

## 验证与生命周期

- 相关11测试文件29 passed in9.37s，实际exit0。新增覆盖固定、自适应、已超XML后的软回退选择，预算分解、原TCP限额及计划仅在实际周期完成后可用，reset清空。
- `scripts/execute_adaptive_margin_feedback.py` 统一会话62715实际exit0，两个子进程actual wait0/reaped；基线绑定原独立检查与SHA，无自动重试。
- `scripts/verify_adaptive_margin_feedback.py` 实际exit0：复核新旧生命周期、配对初态、256新周期和256既有baseline周期、参考切换、动作/计划范围与变化率、原TCP限额及seed模型、预算分项、余量分类、原生goal、状态SHA和恢复TCP误差。其未来余量检查依据求解器复核与日志，不声称逐计划独立重做所有未来模型计算。
- 独立核验 `outputs/adaptive_margin_feedback_verified_20261027/verified.json`，SHA256 `b55bf87c68348aaa8578fe094e23cb8da1b421d095d954ea18432b9cf9c6cae0`。
- 分段误差和terminal实际字段精确对照 `outputs/adaptive_margin_feedback_verified_20261027/segments.json`，SHA256 `ccd27a1b57ac77ba37a1ffbdce533675149925528a4d45272191d1def60bb751`。
- 新增selector、native、历史、worker/owner/verifier和测试Ruff check/format check通过；未声明全仓检查通过。

本次新增物理256周期，世界模型调用与fresh均0。原正确参考18/18、新候选WAM开发和最终留出24条稳定成功门槛保持，目标尚未实现。
