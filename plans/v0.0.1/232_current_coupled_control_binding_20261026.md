# 232 当前耦合逆的真实执行和日志绑定

230完整记录链路必须完成并由231实际wait/reap核实，229原生M/J/rotation goal读取准入保持，才可启动新物理控制。原18profiles/4907周期、seed42、reference clock、controller、083/159旋转和079夹爪保持。仅以228单步耦合模型替换平移规划，不增加fit、mode特判或阈值调整。

每步从当前robot q/SITE packet读取只读M/J及实际controller flags，先用既有旋转候选和previous goal计算native rotation goal，再建局部SO3 spring模型。coupled创新只来自本独立实例上已完成周期，和旧150创新分别保存。约束逆采用当前SITE参考位置/速度和实际previous±.25，只发送实际最终command。

实际平移回写action/goal，耦合history从最终命令重算nominal；173 SOURCE也commit同一最终6D命令。step完成后才更新两套创新。日志区分旧150 nominal和耦合nominal/prediction，保存实际M/J/q/flags/rotation goal、完整模型、逆目标/边界及两套before/after历史。

独立checker必须从实际XML/raw states重建M/J、当前SITE twist、native旋转目标，独立SO3差分/ODE和KKT验证逆命令及耦合历史；实际数据/图像/夹爪/旋转/任务与旧150、173审查继续。JSON往返准入覆盖完整记录。全部source冻结与18 dry-run/相关tests通过后新namespace启动，36stage+owner实际wait/reap/source closure及原18/18性能门槛保持。WAMdev/fresh24继续关闭，尚未执行本计划。
