# 200 加载阶段最终命令限速的完整18条对照

执行199唯一预登记变化：加载最终平移rate2.5（每20Hz周期.125），其他阶段rate5，候选生成/学习KKT/支持权重/085物理候选及旋转/夹爪保持178，最终限速后重新预测并提交实际历史。原146模型与166半径保持，未使用未资格的手指响应模型。

新namespace outputs/qualified_inverse_rate_loading_20261025，固定原18profiles/4907周期、参考时钟不变，不增加评估时长或放宽误差/稳定门槛。独立checker重构route、pre-rate mixture、对应限速、最终命令/预测与两套历史。准入完整CPU tests/18实际CLI dry/完整checker导入及来源核验后持久supervisor/owner逐case collect/verify；机械失败停止、不自动重试，完整36stage+owner实际wait/reap和来源关闭。

保持18/18稳定及位置RMSE<=.01m、SO3RMSE<=.05rad才能进入WAMdev，后续最终24新初态仍必需。候选不预设胜出，当前尚无性能结论。
