# 230 耦合逆和独立创新的完整记录链路

229三任务原生读取准入通过，零积分且实际rotation goal/M/J一致。现在固定原已资格的case00全部164真实记录状态，独立从XML/sim_state重建当前M/J并与当前robot q、SITE pose/twist绑定；候选旋转按当前命令/current SITE/previous_goal的native规则重建并核对归档goal，不用归档未来goal作输入。

每步从CoupledResponseHistory仅取已完成创新，以当前已生成SITE位置/速度参考计算平移；历史提交根据最终候选实际command重算预测，下一记录状态才observe。源状态来自原日志，候选命令没有被施加，因此这是记录状态上的重构命令/历史，不能当替代动态或控制成功。

独立SO3差分/ODE重建响应，逐步核对模型、约束LS/KKT<=2e-6、目标/physical_delta、独立完成创新和JSON往返。加入command、physical_delta、nominal/predicted response、objective、innovation共六类损坏拒绝。所有原数据/代码SHA与seed42保持，无积分/新拟合/模型调用。通过后仍须真实执行器/完整独立日志及原18/4907门槛。
