# 三参数世界坐标角漂移校正的缓存诊断 — 2026-10-21

状态：任何新拟合前登记；仅CPU、0新模型请求/0物理步。不改变017正在运行的raw参考、控制器、配置、预算或冻结源码，不打开fresh性能。

只读方向性审查`8400_directional_error_review/hypothesis_memo.json`（SHA `e698699a04a06064ffbd42e182792f8493cb64b1a581d305a8d0a1a807493bc6`）发现：8400六旧episode的位置误差符号不稳定，不支持统一位置偏置；旋转误差在世界坐标y/z方向有跨episode同向结构。这只是已看开发数据中的假设来源，不是校正有效或泛化的证据。008的各向同性增益未达推广条件，不能将新假设说成原方案已成功。

本次只检验一个三参数角速度偏置`b`，单位rad/s，世界坐标左乘：

`R_corrected(tau) = Exp(-tau * b) @ R_predicted(tau)`。

保持输入world位姿的同一点位置及opening逐值不变；不是在局部delta积分前衰减旋转。`tau=(endpoint_index+1)/20`秒，使用当前已生成参考的端点时钟，不输入未来GT、任务、物体、接触、phase或episode标识。源/TCP的固定右乘旋转不改变世界左误差定义；后续若接入控制，仍须明确调用点约定并另做执行资格，本次不接入。`b=0`为精确identity，时间0不改anchor，旋转始终在SO(3)。

仅使用既有8400_g10的48条缓存，沿用008的`load_cases/shuffled_pairs/view`实际去重、全episode切分和共同有效前缀：44个unique(episode,frame,seed)、6episode/22anchor，所有重复标签和seed同fold。每折5episode拟合、1episode评价；全方法取与008相同donor可用长度的共同端点（H40合计1398），不填补未知未来。依赖旧loader时绑定其实际源SHA，不读取任何fresh数据或Cosmos dataloader。

对训练点`e=Log(R_pred @ R_gt.T)`，只求固定目标`sum(w * ||e-tau*b||^2) + .1*||b||^2`的闭式解，并投影到`||b||<=.25rad/s`。权重按episode等权、每独特query等权、每有效endpoint等权，sum(w)=1；lambda=.1秒²，零偏置先验，不搜索。端点最长2秒，校正角上界.5rad。近pi的模糊log显式拒绝，不丢样或猜分支。所有拟合只用fit labels，heldout不能参与参数、尺度、bounds或候选选择。

固定19次拟合：6折各paired-WAM、no-WAM orientation/hold、shuffled-WAM三模型，共18；另全六episode的paired-WAM一模型只保存候选参数，不作留出成绩或部署。shuffle直接复用008固定seed42的跨episode配对，heldout donor只来自fit。比较raw、identity、paired校正、hold、历史恒速、独立拟合no-WAM、独立拟合shuffle；完整披露H8/H16/H40的SO(3) RMSE、位置RMSE、opening MAE及各episode指标。原始/校正的position/opening必须精确相同，监督与输入的区别明确记录。

事前的`eligible_for_further_validation`同时要求：校正后的H8/H16/H40 pooled旋转RMSE均不劣于raw（数值容差1e-10rad）；H8和H16均至少改善raw 10%；这两个短窗口均严格优于hold、历史外推、独立拟合no-WAM及shuffle的最优值（至少超过1e-10rad）；两个短窗口各至少5/6 heldout episode不退化。H40与各fold的全部取舍仍披露，不以开发交叉诊断冒称独立验收。没有位置/夹爪参数或按task/checkpoint/seed选策略。

即使符合上述条件也**不自动授权新物理或fresh预算**，只允许另行设计更多整episode的冻结验证及执行资格；不符合则该固定假设不推广，不按留出结果重新调lambda、bound、frame或时间函数。保存19组参数、配对/权重/端点、原始与修正误差、源码/数据SHA；另用独立实现复算闭式解和SO(3)指标。该方法改变WAM生成的参考，不能认证原checkpoint或称为机器人响应逆模型。
