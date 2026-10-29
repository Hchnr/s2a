# 早期真实起点的首周期FK长预览（CPU）

前一首周期反馈分支从原368起仅推进到cursor8，cursor9前无witness停止；任务成功0。回到独立核验的原endpoint368，固定当时已生成chunk46的40点参考，比较8/12/16点预览，使较长窗口从开始就包含cursor9。不能将其他分支的370或后续状态当成当前计划已达状态。

三组使用相同实测q/qvel/M/J、固定控制器、零局部误差profile、前一真实命令和原8点warm（12/16仅重复末命令扩展）；每组新建FK缓存。仅首周期预测精度硬约束，整个窗口pose/twist目标、动作box/rate、全部内环/历史/制动/terminal约束保持；记录全部未来误差和是否达标。每组160响应/60迭代，共480响应，不调参重试。种子42，新增物理/模型/fresh均0。

producer：`execute_tracking_v5_fk_first_long_numeric.py --launch`；独立MuJoCo FK/J、重建joint rollout及约束复核：`execute_tracking_v5_fk_first_long_check.py --launch`。后者只检查已保存witness或最近线性可行候选，不重新搜索；若没有候选则显式无可检查项。数值通过不代表真实动作或任务资格，未来未达标也不能隐去。

若存在模型witness，下一阶段另登记同起点各可行窗口最多16新周期的实际CPU反馈和独立命令重放；实际每周期.01m/.05rad、全部内环.05rad关节裕量和原任务门槛保持。不得按结果挑一个窗口并宣称普遍改善，不跳cursor、不hold代替推进、不使用未来观察。GPU禁用，原18条/完整12WAM/final24门槛不变。
