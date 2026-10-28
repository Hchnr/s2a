# cursor9 停止状态的约束归因诊断（CPU）

8/12/16 首周期FK窗口在相同原368起点都只执行到cursor8；原生完整比较见[fk_first_long_native](fk_first_long_native_20261028.md)。这说明本次增加窗口未解决推进问题，但既不能证明物理不可达，也不足以把原因归为冻结动力学。H8末周期实际与预测位姿差较小，下一步优先区分即时动作约束与后续、terminal约束。

固定使用默认H8分支的实际endpoint377和自身九个已完成周期历史。独立重放必须关闭，所有数据/SHA从其producer与checker引用取得；不能混入H12/H16或旧分支377状态。仅使用当前q/qvel、当前M/J、固定控制器/机器人几何、上一命令及已生成chunk46的cursor9及后续参考；无新动作、模型请求、fresh或GPU。

先重建保存的最近线性可行8点候选，独立复算并逐类记录每个约束余量：首命令box/rate、首25内环历史裕量、首制动、未来各内环/制动和terminal。记录活动约束不等于已证明其是唯一原因，须保留全部余量及候选原误差。

再另行实现并登记有限的一周期数值对照：保持原首动作box/rate、25内环历史裕量、制动、首TCP .01m/.05rad和固定50ms，仅去掉未来周期与额外terminal速度限制。保存沿用terminal和移除terminal两种政策各自结果；相同预先固定起点/响应预算，独立几何及物理约束检查，不派发任何候选。该消融仅定位哪些预测条件可能阻止当前推进，不能作为放宽实际安全条件或新控制资格。

若一周期仍无合规候选，只记录固定搜索未找到，不能当成不可行性证明。后续才考虑对既有命令进行冻结与状态更新M/J的成对开环预测；每周期从实测起点初始化一次，内环只推进预测q/qvel，实测内环仅作为评价标签。任何新规划政策仍需完整实际前向与任务验收。

原18条/完整12WAM/final24及GPU禁用保持。本计划未执行，不计入新增实验或控制收益。

## 实施前固定的具体对照

复用现有`fk_pose_cap_scope_joint_plan.solve_interior_plan`的H1和显式terminal_normals能力，不另造求解器。`early_terminal`把保存的原H8方向normal（+e7）施加到H1终点，要求qdot7在50ms时非正；原H8要求的是400ms时非正。因此该组是提前方向停止诊断，不能称原终端条件不变。`no_terminal`传入空方向矩阵。两组只在该方向条件不同，但与原H8相比还共同改变未来成本/约束。

两组seed都取实际最后命令，supplied warm都取保存H8最近线性可行候选的首命令；保留求解器记录的LP替换和实际start policy，不能冒称最终优化起点相同。每组160响应/60迭代，共320响应；seed42、原门槛不变。H1 twist从完整已生成参考cursor9计算，保留邻点8/10，不能对单目标截片计算。H1的all_pose_caps_enforced=true是因首点即整个窗口；空normals从JSON读取时显式恢复(0,7)。

producer入口`execute_tracking_v5_fk_stop_constraint_audit.py --launch`；独立重建当前M/J、joint rollout、MuJoCo FK/J/target errors及全部分组余量的checker入口`execute_tracking_v5_fk_stop_constraint_audit_check.py --launch`。先精确重建原H8保存候选误差与裕量，再执行两个H1对照，模型与首点条件无法一致时停止；不重试、不派发。当前待执行。
