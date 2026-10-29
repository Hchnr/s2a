# 178 加载路由下限制学习逆的完整对照

177规则已接入：原081 native_close_load期间学习平移权重0，其他model_tracking期间沿用175半支持权重，173因果物理分支及174独立检查均按实际命令更新。路由来自已生成参考/当前实测状态/因果latch，不使用case/task/未来成功标签；完整夹爪独立回放重构route并与arm gate核对。它不是实际接触检测，也不保证进入加载前与085状态相同。

新namespace `outputs/qualified_inverse_loading_phase_20261025`完整准入通过：95项CPU测试、18实际CLI dry及checker导入；包括非法路由、门控权重和实际命令独立核验。实查supervisor PID148653、owner PID148761、首条worker PID148845的start_ticks均与launch一致存活，冻结来源检查通过。尚无完整性能结果，执行/机械失败停止不自动重试。

原176继续登记整批，最近6条已独立审查、5条性能通过。新178保持相同146参数/166训练半径、原18profiles/4907周期、controller/旋转/夹爪及全部原门槛，不能因已启动就宣称接触问题修复。完整18/18稳定与误差/机械/生命周期资格前，WAMdev及最终24新初态仍关闭，总目标未完成。

完整收尾完成：18条4907周期、36stage与owner实际wait0/reap及来源核验通过，稳定15/18、综合性能15/18，失败case03/12/14。regression_summary.json与closure_verified.json已生成。此候选未达原18/18，不进入WAM/fresh；191仅追加独立训练干预，不据失败验证标签拟合或放宽门槛。
