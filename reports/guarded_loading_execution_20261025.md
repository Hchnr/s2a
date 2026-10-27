# 204–205 独立前向绑定后的预测约束限速回归

tracking_guarded_loading_inverse与独立checker完整接入。独立prediction四支重建normalized command→SITE位置映射后，核对支持范围、最小可行change、实际命令/重新预测及173/174两套实际历史；15项相关测试及全部164保存当前状态复核通过。完整准入110项CPU tests、18实际CLI dry及checker导入通过，204 proof与来源绑定。

新namespace outputs/qualified_inverse_guarded_loading_20261025，启动后实查supervisor PID231638和owner PID231696身份一致存活。首条worker PID231754已完成164周期并实际wait0/reap；checker PID232303完成独立机械/标签/图像和性能审查，实际wait0/reap。位置RMSE.00262739m、SO3RMSE.00445547rad，末尾连续80周期成功，原资格通过。相同首条固定限速200为位置.01190262m超限，205恢复位置资格，但不能据单条宣称全部接触问题修复或整体优于原178。

原18profiles/4907周期、参考时钟、控制器、旋转/夹爪/146模型/166半径及实际位置/旋转/稳定门槛不变。5mm只为预测决策，不当作真实误差保证。原200继续独立对照，完整18/18、36stage+owner正常wait/reap/来源资格前WAMdev及最终24新初态仍关闭，总目标未完成。

case01完整独立检查通过，checker实际wait0：位置RMSE.00052314m、SO3RMSE.00962227rad，末尾连续70周期成功。当前前两条2/2，尚未认证接触失败case03/06/12或完整18条。原200 case12已通过原任务/误差门槛（位置.00165354m、旋转.01022202rad，末尾88周期成功），但200其他多条误差超限，不能只看单条接触改善采用固定限速。

case02性能通过，但case03已完整机械审查后综合失败：无任务成功，位置RMSE.04337894m、SO3RMSE.24899405rad。206显示早期一步位移预测较准，单步位置可行仍未保证后续持续可行。原整批继续、所有门槛保持，未宣称预测约束已解决接触。
