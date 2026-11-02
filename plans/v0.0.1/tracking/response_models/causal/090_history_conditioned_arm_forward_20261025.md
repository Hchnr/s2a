# 090 仅增加因果响应历史的条件化forward对照

089完整数值及退出审查通过，holdout24/24、训练响应12/12、未训练响应9/12，资格失败。保持相同原24fit和6holdout、训练groups[0,3,5]/未训练[1,2,4]、4865原生行+9差分方程、0.5/0.5权重、ridge0.001及输出12维，不挑选新的anchor或扫描参数。

唯一新增条件是上一实际控制周期的12维响应残差，作为模型条件而非单位增益直接命令补偿。周期t开始时仅使用已观测的当前及前一SITE pose/twist、上一实际applied goal：名义PD预测上一周期motion6和下一twist6；position残差=(p_t−p_prev)−nominal_position_motion，rotation残差=world Log(R_t R_prev^T Exp(nominal_rotation_motion)^T)，速度残差=dt*(twist_t−nominal_next_twist)。不得用Euler差或未来端点；当前首周期为0，episode隔离，重规划chunk保留实际物理历史。

原71维条件后追加该12维，得到83条件、phi84、336设计、336×12系数。对translation action仍仿射。中心/尺度仅按原24fit的当前条件及translation目标重新计算（std_floor1e-8），差分方程/holdout不进入统计。相同误差目标与固定正则化；只有全局bias不惩罚，唯一fit1次及独立数值复算另计。

native轨迹先由完整因果前缀计算历史再选row；不能将单个probe anchor切片后误当作episode首周期清零。成对probe的条件历史来自相同已验证native前缀，plus/minus只改变本周期translation action，不读取探针未来结果作为条件。标签及差分监督仍与条件分离。

先检查未来标签/未来输入扰动不能改变当前history、首周期/reset隔离、world SO3残差及实际历史action绑定，再实现独立历史重建与数值复核。保持6holdout×4预测门槛和全部6anchor×4响应门槛，明确训练/未训练各12项；全通过也只授予离线forward资格，不等于逆控制稳定。原18/18正确参考、WAM开发和最终24条新初态验收保持。0新物理/世界模型/fresh，不自动换参数重fit。

当前仅登记与显式配置，历史特征/新拟合/验证尚未实现或执行。

实现进度：主历史特征及独立绝对旋转重建已实现，8项新旧核心测试通过；实际24fit/4865行历史与336维特征、9条差分方程的独立重建检查通过，零参数拟合。新数值复核和holdout/局部响应验证入口已实现，尚未执行拟合或性能验证；owner与生命周期关闭入口尚待接入。上述“仅登记”描述为登记时快照。

拟合前固定数值一致性容差：原始历史/条件及标签、nominal为绝对1e-10；归一化后features绝对/相对1e-8（独立SO(3)乘法次序造成的浮点误差可被尺度放大）；groups/row_kind/weights必须逐元素相同；独立增广SVD系数绝对/相对1e-8，正则化梯度最大绝对值<1e-9。统计中心/尺度独立重算绝对1e-12。预测主/独立实现绝对/相对1e-9。性能门槛不增加松弛。

执行接入：新增run/close入口，准入要求新旧8项核心测试、实际4865行与9方程独立特征检查和三个CLI导入成功；冻结源码、配置、计划及原始输入指纹。持久owner依次执行唯一fit、独立增广SVD数值复核、6holdout及6anchor验证。各阶段与owner均要求实际wait/reap、无自动重试；性能失败正常保留，不通过调整阈值或覆盖结果处理。准备完成不等于已启动/合格。
