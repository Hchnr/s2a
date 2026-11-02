# 关节约束与首周期跟踪精度联合检查

总目标保持原完整控制/任务及新初态门槛，不以停止或单步模型可行代替成功。原候选偏差在接触前已可预测，step377模型/实际旋转误差约.0784/.0789rad；hard joint witness不是跟踪资格。

新增quality_capped_inner_joint_plan，保留inner-profile误差收紧与制动/终端/box/rate，追加绝对首周期.01m/.05rad条件，最终逐一复查；160响应预算不变。两项实际CPU测试及ruff通过。首版numeric因错误40长度断言失败，保留actual exit1/reaped1；v2按实际长度完整关闭，exit0/reaped0，verified SHA b9c7b08a9354e6e7d0d0b16b75e970b146492a494dfdca2662a6b9d1f48817f7。两分支376仍有witness，377起搜索未找到precision witness，不证明非线性问题全局不可行。

下一数值对照：使用偏差开始扩大时相同实测状态/自身过去profile，原时序目标与保持上一已生成参考点（期望twist为0）分别搜索。若保持可行，方考虑原生明确参考相位的对照。保持/减慢相位是显式重定时方案，不能冒充原时序跟踪；必须记录原参考ID/顺序、所有未消费点、真实20Hz周期和全部原任务成功条件。模型/fresh不因反事实搜索自动增加资格，18/WAM/fresh与冻结v5保持。
