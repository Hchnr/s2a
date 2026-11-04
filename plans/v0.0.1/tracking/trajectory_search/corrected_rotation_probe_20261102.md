# 修正安全模型的一次旋转方向探针

2026-11-02，seed42，CPU only。接续604742a的后验shadow检查，完整目标及18/12/24
验收保持。本计划不是旧60stage/122response搜索的续跑，不覆盖旧失败结果。

固定使用原stage58/anchor85及局部性第5号完整安全响应（step inf约1e-4），
按原bounded causal secant更新一次Js；不修改Jq/Je。独立实现逐值重建更新。
保留原最后旋转cap-distance方向目标、全部安全/terminal/pose ceiling及box/rate/
global/trust约束、.01m/.05rad caps、1e-10独立审核、1e-6原QP证书。

执行前固定1QP、最多100迭代、最多1NNLS；在QP执行前写完整矩阵及输入SHA登记。
原尺度/dual recovery独立审核通过且QP qualified才允许生成唯一alpha1候选。
不可裁剪、补选、改变半径、改目标或自动重试。原command预检失败则保留失败记录，
不执行响应；未合格QP同样终止，不用另一个方向替换。

候选冻结后共享runtime执行anchor/new trial/anchor repeat，生产3响应、独立3响应，
共6响应/1800内步/1874查询/2models。完整响应和audit逐项独立重算，repeat严格一致。
共享runtime的QP/NNLS计数为其内部0，整体成本必须加上qp_result登记的外部1QP及
实际NNLS次数，不允许报告整体0QP。独立更新重建不计新机器人响应或NNLS。

600秒TERM、10秒后KILL，所有拥有进程实际wait/reap。CPU空CUDA/LD，数值库线程1。
模型启动前QA绑定当前源码、配置、计划与报告，不在运行中改源码或文档。

数值候选无论是否安全均不得下发。主要观察完整安全余量与全部24caps，另记录
原cap loss、最后旋转误差、线性安全预测残差。安全/全caps失败如实保留。
局部方向修正不能证明其他方向有效；只有完整数值候选合格后才另行规划物理验证。
