# 从实际370延续首周期FK反馈（CPU）

同状态数值比较：全8点无witness，首点有witness但预测末点cursor9旋转误差.07794rad不合格。必须真实执行验证，不能把取消额外未来模型硬精度条件解释为实际验收放宽。

重放已独立核验的370条真实命令，包括原368前缀及两个FK实际周期，逐端点与该分支状态一致。继承仅这两个真实完成周期的profile和最后命令，不恢复旧分支370+。首次使用同一实际状态的首点witness，派发前检查q/qvel/M/J/Kp/Kd；其4.08452s预计算成本保留，不视为实时零成本。随后移位warm，8点完整pose/twist成本和全段关节/历史/制动/terminal约束保持，显式只首周期预测精度硬约束；无witness则不派发，不hold/跳点/重设状态。

最多16新控制周期370..385，原chunk46 cursor2..17顺序，覆盖关键cursor9；总预算386物理周期，模型/fresh0。每次真实完成25内环q后只更新自身history；实际TCP>.01m/.05rad或任一内环裕量<.05rad则停止下一动作，全部失败照实记录。完整真实轨迹/任务验收不变，夹爪固定原prefix最后命令，仍是arm-only pilot，不取得完整任务或实时资格。

入口`execute_tracking_v5_fk_first_feedback_native.py --launch`，独立物理重放入口`execute_tracking_v5_fk_first_feedback_native_check.py --launch`；复核全部端点、25内环q/qvel/time、目标消费、实际误差/关节、继承及自身更新历史，并记录最大数值差异。原18/完整12WAM开发/final24门槛保持，GPU继续禁用。新outputs不覆盖，当前待实际执行。
