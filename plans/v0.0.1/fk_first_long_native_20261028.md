# 首周期长预览同起点实际反馈对照（CPU）

数值登记见[fk_first_long](fk_first_long_20261028.md)。8/12/16均在原真实368找到首周期模型witness，分别仅8/8、9/12、9/16未来端点达到姿态精度；不能据此选一个窗口或宣称轨迹成功。先要求各候选独立MuJoCo几何及全段关节约束检查关闭，再按固定顺序8、12、16全部执行一次，不调参/重试。

三条分支分别新建环境、重放相同原368命令并逐端点匹配。局部profile均从0开始；first使用对应H已核验归档计划，记录原搜索成本，dispatch前比对q/qvel/M/J/Kp/Kd。后续每周期重建当前模型、移位warm、以原固定H求解，首预测点精度硬约束，整个H的pose/twist成本、内环/历史/制动/terminal和动作box/rate保持。不同H会同时扩展未来目标和约束范围，并改变terminal，不能解释为只多了成本项或直接按不同H总objective排名。参考twist还使用后一已生成目标，仍无未来观测输入。

每条最多16新周期368..383，原chunk46 cursor0..15，累计每条384物理周期、三条1152；独立重放另最多1152。只读取自身已完成25内环反馈更新history；无witness立即停止派发，保存搜索耗时。每个实际周期原.01m/.05rad、内环.05rad关节裕量门槛不变，违反则停止下一动作并保留失败。夹爪固定原prefix最后值，本阶段arm-only；不跳点、不hold、不恢复其他分支状态。

入口`execute_tracking_v5_fk_first_long_native.py --horizon H --launch`，独立重放`execute_tracking_v5_fk_first_long_native_check.py --horizon H --launch`，H依次8/12/16。独立重放不导入求解器，核对所有端点/25内环q/qvel/time、历史、参考消费、保存的全未来误差及首点cap策略、真实误差/裕量/任务谓词和停止行为；保持同样sensor刷新与render调度。

模型/fresh0；原18条4907步、完整12WAM开发、final24任务资格及GPU禁用保持。CPU数值可行、短期实际达标和仿真20Hz均不代表完整任务、泛化或50ms实时资格。

三组实际生产及独立重放均已actual wait0/reap0：各9周期实际达标但只到cursor8，step377/cursor9前全因模型无witness停止；task/grasp全false。独立各377周期/378端点/225新内环样本的state/q/qvel差异均0，未来全部324预测端点独立XML几何与误差重算通过。本阶段总物理2262，模型/fresh0，无完整控制或任务资格。[完整结果](../../reports/fk_first_long_native_20261028.md)。
