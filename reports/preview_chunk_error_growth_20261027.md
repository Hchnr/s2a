# 长分支误差增长、接触与关节约束诊断（2026-10-27）

上一回合已完成336端点的TCP恢复核验，属于实际进展。本回合新增三个可执行检查入口，均ruff check/format通过，实际CPU运行退出0。全部读取已有开发数据，新增物理步、模型调用、fresh均0。

## 证据产物

- `scripts/audit_preview_chunk_error_growth.py`：逐步参考平移/旋转速度、TCP误差及命令box/rate边界。`outputs/preview_chunk_error_growth_20261027/verified.json` SHA256 `32593f5579966786bc5a752c31bbffc6843cd5a364cb82ba48d7619232cb49a3`，会话35480退出0。
- `scripts/audit_preview_chunk_contacts.py`：恢复全部280个长分支端点，重新计算接触几何，不推进物理。`outputs/preview_chunk_contacts_20261027/verified.json` SHA256 `c198995bc258ed9db87ab749a06f4fa9a6c4e177c41983e861f2c1860dad4a8b`，会话13578退出0。接触几何不是原始接触力或内环完整接触历史。
- `scripts/audit_preview_chunk_joint_limits.py`：验证XML前7关节均为标量robot0_joint1..7，再按MjSim flatten的time后qpos读取保存状态，对照XML范围。`outputs/preview_chunk_joint_limits_20261027/verified.json` SHA256 `fcd3ccf07e17671be8da8582667cd229cda6a20ad10e3ef172d329ee3592b5cf`，实际退出0。靠近/超出XML边界不是约束力已激活的直接证明。

## 结果及决策

1. bowl/raw/seed42：40步中X/Z命令各38步处于下界；第7步位置误差首次超过1cm，参考最大线速度约0.134m/s。全部保存端点无机器人接触几何；第7关节40/40端点略超XML上界2.8973rad（最大约0.0005rad）。应优先检查关节限位响应及任务空间逆解的可行性，不能以无接触认定机器人完全自由运动，也不能由饱和直接证明参考不可达。
2. wine/raw/seed42：手掌与目标瓶体接触几何从第16步出现，共24个端点；误差在第24步超过1cm，Z命令19/40步处于下界。时间关系支持接触相关跟踪失配的诊断方向，尚无受控干预证明因果。
3. bowl/raw/seed43：从第22步出现robot0_link5与非目标wine_bottle接触，之后还出现手指/手掌与目标bowl接触；第24步误差超过1cm。已有4步目标抓取不能掩盖非目标碰撞。
4. 低误差terminal058分支中，bowl两例没有机器人接触；wine/seed42则与非目标bowl发生手指接触。其低TCP误差仍不能证明参考满足目标抓取几何。

terminal058并非轨迹乘以0.58：`src/s2a/terminal_readout_policy.py:13`约束旋转系数及特征变换沿用baseline，`:24`仅在最后一次denoise step34使用学习terminal readout；其余步骤用baseline。raw与terminal完整闭环轨迹已经分叉，本次选择的最早事件时刻也不同，不能把两者差异归因于单一缩放或相同状态下的对照。

下一步需要在保存状态上读取关节限位约束力、Jacobian及局部控制响应，验证bowl/raw/seed42的任务空间控制是否忽略限位；再决定统一的关节约束感知控制或参考时间处理候选。候选仍须原正确参考18/18资格、完整世界模型开发门槛及最终留出24条验收。当前没有新增任务成功、没有启用fresh，也没有直接修改已冻结控制器。
