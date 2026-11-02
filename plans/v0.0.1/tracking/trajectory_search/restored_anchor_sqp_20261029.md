# 从终端余量恢复候选重新开始完整 H12 SQP

2026-10-29 执行前登记。tail 生产与隔离 v2 的全部14条独立复算、2485来源闭合及全12候选分析已正常完成；足量恢复标签通过独立核验。下一阶段按本文实现与QA准入后执行一次完整H12搜索及全量独立复算，不直接启动物理 pilot。GPU 继续禁用，登记本身不产生模型响应。

## 前置来源与当前观察

路径均相对仓库根目录；以下来源已实际核验。原producer句柄15941、v2 checker句柄41688、来源闭合句柄11624及分析句柄92812均实际exit0；下一阶段准入须重新验证完整SHA绑定和生命周期。v1失败保留为失败，不追改旧源码或产物。

| 来源 | 路径 | SHA256 / 状态 |
| --- | --- | --- |
| tail producer | `outputs/tracking_v5_tail_translation_reserve_20261029/verified.json` | `0305dc39480abf4512205f8628729fd789595d40aeca189343bc19779f029d70` |
| tail producer owner | `outputs/tracking_v5_tail_translation_reserve_execution_20261029/result.json` | `f604b9e31b1a4aa00a576b099065334fd04aae8722ac5710165b080c2c76b6cb` |
| tail v2 checker | `outputs/tracking_v5_tail_translation_reserve_check_v2_20261029/verified.json` | `142b0d9ad478b30c1e707c054cec48bd29326b0b24b4e07e64b6ea8cb37e6bde` |
| tail v2 checker owner | `outputs/tracking_v5_tail_translation_reserve_check_v2_execution_20261029/result.json` | `3fa4fb454a1d03584d2216982acec4eae1f9d06e635e38f4b8c23c6ba3df6843` |
| 原95源/29模块 QA（766通过） | `outputs/tail_translation_reserve_native_qa_20261029_lkpp55kq/result.json` | `e421b3860faa93407194fad3cc57eabd56d16d6eea58bb46998a90dc62dde663` |
| 新100源/30模块 QA（796通过） | `outputs/tail_translation_check_v2_native_qa_20261029_y5rxr7gl/result.json` | `fe36dfd12c29f6a55fd73d789cc91132e3fb726069100cf3a729743e421590d8` |
| 原5命令准入 | `outputs/tail_translation_reserve_admission_20261029/result.json` | `a02ef18417130cdd2aa74a299e6e9245def56c3627a388495a066626ce3c8e44` |
| 原check owner准入 | `outputs/tail_translation_reserve_check_admission_20261029/result.json` | `40db64dbca0b784f2b890a91fa0e7b5e3211f4048fe4dae13494d938ec555908` |
| v2 4命令准入 | `outputs/tail_translation_check_v2_admission_20261029/result.json` | `159cae7a80c7907e934869a388aab26a7269a02b44ebb105940f54be9dc6729f` |
| tail 2485来源闭合 | `outputs/tail_translation_reserve_v2_final_closure_20261029.json` | `52783b3ef40e8cf2e208277a8d841c2a53553551c971f68f2e33d7e64dc958e8` |
| tail 全12候选分析 | `outputs/tail_translation_reserve_v2_analysis_20261029/result.json` | `00cb2a2d740047d4839c417a3d80e807ac92cb15d879fd0f6d3fb74d93e9135c` |

已关闭的producer与v2 checker各为14完整响应、13唯一命令、4200数值内步、4369查询；选择 `answer.selected_response_index=5`，即末周期 y 平移 `+0.001` 的完整 H12 命令。该候选 `terminal[0]=7.734855175988188e-5`，减去原 `1e-8` buffer 后为 `7.733855175988188e-5`；其他非线性安全项最小值仍仅 `3.267282409069594e-7`。producer 保存 `sufficient_reserve_restored=True`、14条因果/history/repeat 检查均通过，prefix9，完整 witness0。这是终端经验余量恢复，不是全约束鲁棒内点，也没有推进原 cursor9。

tail 前项实际总成本包括生产14、失败v1复算14、v2复算14，共42响应/12600数值内步/13107查询；新的4000预算仅属于本阶段，不抹去前项成本。

执行前要求 tail 独立14条复算、strict 分类、保存及独立执行内部因果检查、历史/repeat、E17/选择规则和同源闭包全部通过；独立结果必须确认上述选择及足量标签，不能在不符时换候选。读取 `answer.selected` 完整记录，并逐字段绑定 `evaluated_candidates[5]`、选择账本、原14次实际计账。任何缺失、不一致或来源变化立即拒绝；不回退到旧 anchor1819 或其他命令。

## 目的、因果初态与保持项

检验从该完整候选重新开始原显式 SQP，能否在固定预算内得到全部12点达标的数值 witness。所有12周期、每周期6个命令共72变量重新开放；不把 tail 阶段冻结前11周期的限制继承到搜索，否则 cursor9 不能改变。原 prefix9 指 cursor0..8，仍以原规则保持；cursor9 及后两点必须全部通过，才能报告完整 witness。

每次模型响应都从同一 actual368 的 q/v、固定 finger、initial_joint、completed367 previous command 和因果 rotation latch 重新复制开始；不得把所选候选的预测终态当物理初态。保留当时已生成的原 H12/600ms 参考、TCP/坐标系、夹爪计划及原可用历史误差 profile，不引入 actual369+ 观测、额外历史样本或新 world-model 参考。

固定完整 XML/资产快照、原 native float32 goal、状态更新 M/J、FK/偏置/被动力、增益、力矩裁剪、25×2ms 内环；不采用单行割线修正。原 `.01m` / `.05rad` pose caps、关节范围、`.1s` 制动、terminal normals、`.25` 命令 rate 及九组审计容差不变。实际普通接纳使用原 `1e-8` 安全 buffer；QP 对已达 prefix 的额外优化余量仍为 `1e-4`。恢复门槛 `10*E17` 仅作前项经验准入，不变成新控制门槛或72维误差上界。

新运行初始化 `B=I`、无 pending curvature pair、空响应缓存及空阶段账本，不继承旧 SQP Hessian、乘子、预算剩余或失败状态。这是从不同 warm 的新完整搜索，不称为旧优化器状态的连续执行；结果不能单独归因于终端余量或 BFGS。

## 必要的隔离实现差异与首次完整记录门

冻结基线为 [accepted_point_sqp_plan.py](../../src/s2a/accepted_point_sqp_plan.py#L19)，SHA `fbc63203f880d6b5e86a12e6ef19bde2d62f1b7f378cdb9dcfd7a50720fa2a70`；原独立 ledger 为 [accepted_point_sqp_ledger_audit.py](../../scripts/accepted_point_sqp_ledger_audit.py#L233)，SHA `b98933bef89a343f9caae2b217c41338befc42cd0b70a45311fe0523fa084a55`。新增隔离 solver、入口和独立 ledger，不修改已执行来源。

1. 将 solver 起点的 `np.clip(warm.ravel(), lo, hi)` 改为 `warm.ravel().copy()`，保留原有限性、形状、box/rate `1e-12` 准入。已知首命令与收紧边界存在约 `7.2e-16` 的浮点越界，原准入允许，但 clip 会改变字节；不能为了通过旧 clip-identity guard 改写选中 warm。新入口采用同原容差的只读准入，禁止裁剪；`supplied_warm`、`used_start`、trace0 commands 与 tail selected 的 float64 bytes 完全一致，`start_roundoff_clip` 为零。独立 ledger 相应重建 copy 起点，其他数学保持。
2. 在 `initial=evaluate(start, 'warm')` 构造完成全部 entry 后，紧接执行显式首次记录核验 hook；位置在任何 FD、QP、阶段尝试或后续响应之前。仅比较 predict 返回的 rollout 不够：完整 entry 含原审计、全部 safety_slack、TCP/q 误差、prefix/feasible/排序指标及 commands。hook 应取得与缓存隔离的只读副本，不能修改首次响应、缓存或后续算法状态。
3. hook 将新首条完整 entry 与已闭合 tail selected 作完整 canonical JSON 数值数组逐路径 exact / maxdiff 比较，同时严格核对 dict 键、标量及布尔标签、命令字节。不得省略非最小安全行、period 字段或指标；canonical 比较不声称 JSON 保留了原 dtype 或空数组隐藏维度。来源、hook 调用记录及返回 guard 状态显式存档。

首次完整记录不 exact 时立即抛出明确准入失败，不进入 warm 后续规划流程，不尝试再次恢复或重跑。保存完整 observed entry、expected ref/locator、逐路径差异和原始异常；该路径为1已完成计费响应、300内步、313查询（含首次 current 查询1），0 FD、0 QP、0阶段/外层迭代、`answer=None`，不能伪造成功 solver answer 或继续消费剩余预算。若此前输入准入失败，应为0响应/0模型查询；技术异常则保存实际完成和部分工作计数，不补成上述理想计数。

首次 exact 通过只收费这一次响应；它就是 solver 缓存中的 trace0/initial，不能再执行一遍 warm 或把预检调用藏到预算外。最终 `answer.initial` 仍须绑定同完整记录。初始完整 witness 按原逻辑可直接返回；当前 tail 所选记录为非 witness，不能硬编码未来结果。独立 ledger 审计 hook 严格先于所有阶段请求及完整初值关系。

上述差异分别列入 AST 逆变换证明：函数/模块命名与必要来源字段、copy 起点、首次完整记录 hook 及其失败归档。移除这些明确差异后算法 AST 应与冻结基线相同；不得宣称仅一行差异而遗漏实时完整核验。原 QP/BFGS core 原字节复用，独立证明不代替真实首条匹配。

## 固定 SQP、接纳与预算

保留完整72轴 `1e-4` 单侧 FD，方向由当前 stage 上界决定；只在完整 stencil 后计算 J 和 B 更新。每个新 probe 都是完整 H12 响应、占全局精确字节缓存预算，部分 stencil 不更新曲率。命令边界附近的浮点与有限尺度 FD 风险继续如实保留，不能以恢复终端经验余量宣称导数已正确。

QP 变量为 `[d,t]`，t 为绝对非负 epigraph；目标 `t + 1e-4*grad(sum(q)).d + .5*d.T*B*d`。全部安全、prefix 优化余量、当前目标两行、rate、命令 box/stage 半径及 t 下界保持原 `A*z+b>=0` 符号、缩放与单位。公开 SLSQP `ftol=1e-14`，每 QP 最多100次内部代数迭代；原 NNLS 乘子拟合、active `10*tau`、证书 `tau=1e-6` 及所有资格条件不变。证书失败保存实际原因，不放松、不重试或改用另一后端。

半径固定 `.01/.005/.0025`，回缩 alpha 固定 `1,1/2,...,1/256`。普通 anchor 仅接受第一条原审计通过、原 prefix 不退化、全部非线性 slack `>=1e-8`、原目标 max(q) 下降 `>1e-12` 或 prefix 增长的实际 ray；完整缓存 witness 保持原优先规则且不额外要求优化余量。普通 FD 下降不冒充接纳点，拒绝方向/点不更新 B；失败缩半径、接纳重置 `.01`、半径耗尽即停。

同 cursor 的下一实际接纳点采用上次 QP 同一原单位 lambda 计算两端 Lagrangian 梯度差，s 使用实际 ray 位移；保留 Powell `.2`、原对称 outer 公式、严格对称检查和 Cholesky，不增加平均对称化或谱修补。cursor 改变清空 pair、B 重置 I；半径缩小时保留 B。所有原跳过/失败规则和全缓存最终排序保持，最终 selected 不一定是 last accepted anchor。

固定上限4000个完整唯一响应（包含首 warm）、60次外层尝试、60 stages；每 stage 最多100 QP 代数迭代，合计不超过6000。缓存命中不新增模型响应；不能把 QP 内部迭代记成响应或扩大外层额度。N 个完整响应对应 `300*N` 内步与 `1+312*N` 查询；最大为1200000内步、1248001查询。producer timeout1200s、独立 checker timeout5400s，固定独占新输出，无自动重试、第二起点、加预算或看结果改配置。

所有终止都保留已完成响应、部分 FD、阶段/请求/缓存账本、完整 QP 证书、ray/anchor 审计、B 生命周期及全缓存 witness；有限无 witness 不证明不可达。技术失败不伪装为有限优化正常结束。

## QA、独立全量复算与结论门

正式前冻结新来源，记录与基线的 AST 限定差异、原 QA 子集 SHA 和新正式 native CPU QA。必要合成用例覆盖：`1e-12` 内合法但 clip 改字节的 warm 保留；完整 entry 任一非最小 audit/指标变更在1响应后立即停、0 FD/QP；hook 不可变更缓存；exact 成功不重复收费；非法输入零响应；完整初始 witness 原门槛；旧预算、部分 FD/缓存 witness、实际 s/same lambda/cursor reset 及排序保持。不增加凑数量的结构测试，不把合成通过计作控制成功。

新独立 checker 首先验证来源及两 owner 实际 wait/reap、无信号退出、命令/CPU环境、同 QA/plan/source bytes 与失败保存规则。正常搜索逐条独立重建全部 N 完整响应，不跳过 FD、失败 ray 或最后预算响应；逐条比完整 safety 向量、TCP误差与 strict labels，另核 initial/selected/closest 完整 rollout、全部实际 ray/anchor。各阶段自身计账同 `300*N`、`1+312*N`，不能仅核选中候选就称全量通过。

算法代数由保存原精度 safety/TCP 向量独立重建 FD、QP矩阵/缩放/近似乘子证书、实际 s 与同 lambda 的 secant、B 更新及阶段排序；独立物理数值路径另外核对这些向量，使用原冻结数值容差，strict 分类不得放宽。新首响应对历史 selected 的 exact 门属于相同生产路径记录；跨独立浮点实现不强求逐 bit 相同，不因此跳过完整复算或放宽原 caps。

只有完整 H12 原 caps 与全部原审计的 witness 经全量独立复算、严格标签和来源闭包通过，才可另行登记从 actual368 同初态开展原生物理验证。tail 足量恢复、某 cursor 达标、QP证书通过或本次搜索正常关闭均不构成控制成功。新阶段不直接运行 pilot；物理、world-model、fresh、实际参考消费均0，`control_qualified=False`，actual cursor仍8、task/grasp仍0；原18正确参考/12WAM开发/24最终留出门槛不变。

所有文件与产物在本仓库新路径维护，旧来源保持冻结；命令 cwd 为仓库，CUDA/LD 显式为空，原单线程 CPU 环境与 seed42 保持。完成隔离实现、正式native CPU QA及实际CLI准入前，不启动本阶段模型响应。
