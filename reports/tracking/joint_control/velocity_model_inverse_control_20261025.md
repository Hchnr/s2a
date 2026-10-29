# 080 有限原生命令规划器：实现与独立最优性检查通过，物理未执行

按080预登记实现src/s2a/velocity_model_planner.py。当前q、真实motor cache和瞬时V输入；只使用已生成opening参考前min(8,available)端点。按[0,-1,+1]序列顺序广度展开全部候选，固定q跟踪平方误差+0.0001命令平方代价，仅执行所选首命令。非有限及超速度包络的假设序列显式拒绝，记录完整叶序列计数，全部不可行则失败；不clip V或切换其他策略。可用参考不足8不补未来观测。

速度包络仅由079训练cache当前/下一V最大绝对值×1.5确定，数值0.31375058401023426，cache SHA保存于preflight产物。它是模型支持启发式，不是物理安全认证。常速度先验/079权重均未重新拟合。

规划器和079响应模块21项检查通过：有延迟的两步最优选择、一步选择不同、并列顺序、独立完整序列枚举、短窗口/尾部因果、包络、非有限/不合法输入等。另完整仿真函数与074恢复AST对照、独立机械checker完整verify_case恢复AST对照各1项通过，共23项。Ruff通过。新仿真分支仅增加当前qvel读取及与sim速度核验、夹爪规划和相应日志；原OSC/初态motor/坐标系/参考/步进/任务判定保持。机械checker保留原其他门槛，新增独立完整搜索重算，不导入producer规划/特征/模型函数。

## 实际解释器与独立完整搜索

实际outputs/libero_env311/bin/python、空CUDA/LD、BLAS/OMP/MKL8运行27个已知验证档案状态的H8查询，全有可行序列；CPU耗时median0.006103s、p950.006367s、max0.006448s。每档案取初点/中点/末8周期之前，包含新retain档案的共享prefix状态，不把这27点宣称为全部新干预响应覆盖或完整状态分布。两次预热不计入耗时。

verify_velocity_planner_preflight.py用不导入producer代码的scalar depth-first完整搜索，逐个重新计算全部候选最小代价、所选q/V/m轨迹及非有限/包络拒绝叶计数；全部27条通过，最优性绝对数值容差1e-10、近并列异序列0，来源与参数SHA读取前后相同。产物outputs/velocity_model_planner_preflight_20261025/preflight.json与independent_verified.json。

这是局部模型上规划可行性和算法正确性，不是模型对所有候选命令的物理正确性，也不构成20Hz实时资格。尚缺实际18条worker/config/owner的完整准入、保存轨迹与当前qvel的逐步绑定检查及正确参考物理回归；physical_qualification=False，0新fit/模型请求/物理/fresh。原074/075失败及最终WAM/fresh门槛不变，总目标未完成。

## 固定18条实际执行已启动，首条独立资格通过

完整入口configs/velocity_model_control_20261025.json、collect/verify/run_velocity_model_control.py已实现。18个profile及4907步与074完全相同；冻结079模型、独立模型响应资格、规划preflight与独立搜索SHA，包络不变。新增driver独立从实际model.xml解析状态布局，将每步当前qvel、opening、真实motor cache和source pose与实际端点NPZ绑定（1e-12），并保留全部独立机械/标签/三图像审查。source/config/plan/test依赖存在检查和Ruff通过。

launch实际完成20个准入进程：23项检查、18条3.11 dry、checker导入均正常退出，输入/源码读取前后SHA不变。等待包装进程3551648/start_ticks583400674、execution owner3551656/start_ticks583400718、首worker3551664/start_ticks583400752已核验真实身份和空CUDA/LD。owner负责每条collect及verify实际wait/reap，包装进程负责owner实际wait；不自动重试或改参数。

case00完成164步，collection与independent verify均实际exit0；完整搜索最优性、当前速度实际端点绑定、机械/标签/三图像和来源审查通过，performance_qualified=True，成功91步、末尾连续85步稳定。证据outputs/velocity_model_control_20261025/case00/independent_verified.json。原owner仍在执行剩余固定案例，无18条完整结论，不进入WAM或fresh，不把首条通过当作总目标完成。

## 前五条审查及开度/加载分离的失败证据

同一execution owner3551656/start_ticks583400718实际复核仍存活，未重启。case00–04均完成完整独立搜索、速度/实际端点绑定、机械/标签/图像与正常退出审查，性能3/5：case01 wine/demo1和case03 bowl/demo40均0成功；case00/02/04末尾85/93/92步稳定。原18/18门槛已无法满足，仍继续固定剩余案例，不进入WAM或fresh，不改本次代价、参数、包络或初态。

只读diagnose_velocity_planner_loading.py加载case01实际model.xml编译执行器参数（0积分步），确认两指JOINT transmission、gear1、qpos0=0、固定gain/affine bias、无activation动力学，gain1000、position bias-1000、force limit±20N。用已记录实际ctrl/qpos重建执行器驱动力；native参考用相同接口由post-motor cache推得ctrl。当前模型及原始NPZ/日志读取前后SHA不变，实际ctrl与重建ctrl逐值1e-12一致；未拟合、不调用世界模型、不运行新物理。

在失败wine轨迹90–149周期：实际每指平均闭合执行器驱动力0.549/0.577N，native16.058/14.951N；实际motor均值约[-0.242,+0.242]，native[-1,+1]；opening RMSE仅0.00710（归一化开度）。周期100实际力约[-0.267,+0.767]N，native[-16.365,+15.063]N。完整证据case01/actuator_loading_diagnosis.json。

这是按实际执行器参数和endpoint控制量重建的驱动力，不能当作对象接触法向力或唯一失败根因。它支持一个明确缺口：优化opening轨迹本身并不约束夹持加载；相近开度允许完全不同的闭合目标/驱动力，且本候选任务失败。后续须在不读取对象oracle/成功谓词的前提下检验因果闭合加载/抓持意图，不能继续把q误差小当作稳定任务保证。原080完整批次继续，原任务/状态门槛不变。

## 前八条审查与前七条074配对分析

同一owner/start_ticks复核仍在运行。case00–07已全部独立审查，稳定3/8（00/02/04），01/03/05/06/07失败；不改原18/18失败结论，固定剩余继续。新增case06 wine/demo46在074稳定而080失败；case07 cream/demo40亦失败。

只读compare_velocity_control_loading.py在当时已正常退出并完整核验的前7条native参考上，核对080与074 profile/初态/参考完全相同，从每条实际model.xml编译并验证执行器传动/增益/bias/限幅，再比较全轨迹opening误差和native post-motor首分量<−0.99区间的驱动力；该mask仅离线描述，未用于在线动作或新接受门槛。每个档案/模型/原日志SHA前后保持，0新物理/fit/世界模型请求/fresh。证据outputs/velocity_control_loading_pairs_20261025/verified_prefix_07.json。

代表配对：case06 opening RMSE由074的0.08562降至080的0.01021，但稳定True→False；080所述加载区间每指均力1.277/0.992N，074为10.409/10.613N。case01 080约0.642/0.608N、074约16.171/14.859N，稳定亦True→False。case03、05虽opening RMSE分别从0.32544/0.41452降至0.01552/0.02171，任务仍失败。另一方面，case00/02/04在较低但非零加载下仍稳定，故不能据此设一个后验通用力阈值或宣称加载是所有失败唯一原因。

这些配对进一步支持需要把闭合加载/抓持意图与纯aperture跟踪区分；不支持放宽任务门槛、把state子指标提升当作闭环任务成功，亦未登记或运行新的混合策略。本次全部18条冻结代码和参数不变。

## 13条独立审查进度

同一owner3551656/start_ticks583400718仍在执行，case00–12均已独立核验，稳定7/13（00/02/04/08/09/10/12），其他6条失败。原18/18资格失败不变，剩余固定5条继续。基于加载证据另登记081并实现纯路由/7项检查，没有修改080已冻结的任何源码、参数、代价、配置或判定，尚无081物理。详见loaded_velocity_control_20261025.md。

## 完整18条关闭：稳定12/18，性能资格失败

080全部18条4907步及36个collect/verify stage完成，完整搜索/速度端点绑定/控制机械/标签/图像均通过独立审查。候选稳定12/18，原18/18门槛失败；相同18条074为14/18，不采用080进入WAM或fresh。regression_summary.json逐条保存任务、TCP和opening误差，未只用aperture改善宣称稳定收益。

close_velocity_model_control.py独立核对全部36组durable actualwait0/reap/无signal，execution owner亦实际wait0/reap、原PID/start_ticks身份消失；所有冻结来源前后SHA相同，closure_verified.json verified_cases=18、owner_normal_wait_verified=True、mechanical_qualification=True、performance_qualified=False。此次不存在074的根execution owner wait凭据缺口。0新fit/模型请求/fresh，总目标未完成。

基于已经登记的081加载/跟踪路由，纯模块、仿真完整函数范围及独立回放/机械checker均已实现；完整配置和入口的准备发生在080关闭前但没有启动任何并行物理。081正式launch强制读取本次完整closure及其summary SHA，只有确认080已关闭、execution owner正常wait后才能开始新固定18条。旧080源码、参数与输出保持不变。
