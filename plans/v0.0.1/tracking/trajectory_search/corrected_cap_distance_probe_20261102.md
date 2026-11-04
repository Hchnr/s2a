# 已修正安全模型下的一次完整cap-distance方向验证

2026-11-02，seed42，CPU only。接续修正后旋转方向的完整局部性诊断；全局可靠跟踪
与原18/12/24、稳定20/24且每task4/8门槛保持，不授予局部数值物理资格。

固定anchor85、原stage58的Jq/Je、已独立验证并登记的修正Js，不新拟合模型。
使用原完整24组cap-distance GN，而非最后旋转子目标；原rationalized float64 loss、
原box/rate/global/trust、所有关节/制动/terminal/pose ceiling和.01m/.05rad caps保持。
独立构建B/g/A/b逐值比较；A/b必须与已关闭旋转QP完全一致，仅B/g回到全目标。

执行前登记矩阵和输入SHA：1QP、最多100迭代、最多1NNLS；原1e-6证书、objective
scaling和dual recovery审核保持。不合格则终止，不重选方向或修改预算。
固定alpha1/.5/.25/.125/.0625，完整case列表预检后冻结，任何原command门失败则
终止，不裁剪、重排、补选。所有5个trial均执行，不按前几个结果改变列表。

生产anchor+5trial+repeat共7响应/2100内步/2185查询；独立7/2100/2185，
合计14/4200/4370、2model。整体1QP/实际NNLS次数，runtime内部0QP不可冒充整体0。
0新模型拟合、0物理/GPU/WAM/fresh/最终留出/实际参考消费；所有候选不下发。
600秒TERM/10秒KILL，实际wait/reap，无重试；CPU空CUDA/LD，数值库单线程。
源码/来源前后SHA、原独立1e-10及布尔审核、精确anchor/repeat保持。

主要判断完整安全门和全部24caps，另记完整cap loss实际下降与GN预测/比值及所有
尺度安全模型误差。数值安全且loss下降仍不是完整witness或任务成功。全部样本与
失败如实记录，不追认新预算为原60stage搜索的一部分，不打开最终留出。
