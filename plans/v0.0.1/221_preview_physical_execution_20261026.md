# 221 约束预览的真实控制对照

以前提219完整164状态独立凸最优性和220在线输入/参考/physical_delta完整绑定及所有SHA保持为准，不采用218 SLSQP失败版本。固定seed42及原18profiles/4907周期、同一初态/参考时钟/物理controller，旋转083/159和夹爪079不变，无参考慢放、任务特判、学习参数重选或新模型调用。

每步先根据当前已测状态和已生成chunk选择夹爪，再用当前reader+150完成创新构造220输入。完整生成SOURCE/TCP参考按固定tool/SITE映射转换为世界SITE轨迹，当前cursor取原执行进度；只规划已生成区间内未来最多6周期，尾段缩短。预测目标/约束严格按218/219，所有计划命令合法，只发送首条。

最终平移command与实际接口physical_delta回写action及goal日志，173 SOURCE残差commit必须使用此最终命令，不能保留原暂定命令；env.step后才用实际下一SITE更新150创新。保留当前可读FT/joints/goal/motor输入的独立实际状态绑定，但规划器不读取对象、接触标签或task predicate。日志明确每步所用参考、horizon、全部计划/模型预测、目标、迭代数及实际命令，不称局部模型预测为真实未来。

独立checker在原旋转/夹爪/几何/状态/标签/图像/任务审查外，重新从实际当前SITE/twist、实际previous、完成创新和已生成参考构造逐标量物理rollout、目标及box/rate约束，核对完整计划、首命令、physical_delta与独立KKT<=2e-6；保留原任务与位置/SO3容差，不能以优化成功替代任务成功。

先完成相关CPU tests、18实际CLI dry-run和checker导入，完整source roster冻结后由owner采集/check36stage并实际wait/reap，最后owner收尾/source closure。资格仍为18/18稳定且每条位置RMSE<=.01m、SO3 RMSE<=.05rad；通过之前WAMdev和fresh24关闭。失败按原记录保留，不自动重试或放宽任何门槛。尚未执行本计划。
