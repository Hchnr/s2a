# 固定核查搜索停止前的三个未调用方向

2026-11-02，seed42，GPU禁用。接续11f7a1f实际结果，不降低原最小下降、radius、
安全/资格门，不重新优化，不宣称旧搜索已执行这些响应。

固定使用93deb627已关闭safety-feedback搜索的stage9/10/11，各唯一alpha1候选，
共同anchor response18，三QP均已qualified且命令门通过，原预测正下降小于1e-4、
response_index=None、reason=predicted_reduction_insufficient。全部按预定顺序执行，
不可换成效果更好的样本；任一重复/预检/来源不合格即终止，不裁剪或补选。

重建已记录Js/Jq/Je（含safety-only），逐stage独立重建aligned B/g/A/b与原QP证书、
scaling/recovery；三个stage之间没有实际更新，应逐值相同模型。只重建已有QP，
0新增优化/NNLS/模型拟合。原root global/pose ceiling、参考、固定接口和初态保持。

anchor+3固定方向+repeat生产5响应/1500内步/1561查询；独立5/1500/1561，
共10响应/3000内步/3122查询、2model。费用独立登记，不挪用旧未使用预算。
0物理/GPU/WAM/fresh/最终留出/实际参考消费，候选不下发。
原1e-10/torque1e-8布尔审核，完整响应/anchor/repeat exact，QA/source/map/driver
绑定及来源/源码SHA前后不变。CPU空CUDA/LD、数值库线程1，600秒TERM/10秒KILL，
实际wait/reap，无重试。

报告全部实际安全/24caps/loss变化、预测比及reserve，明确新响应虽可判断是否
存在小幅实际下降，不能追认原1e-4接纳条件通过或直接改变该门。无论结果如何，
不能以这三个局部样本证明全局可达/不可达或任务稳定成功；原18/12/24验收保持。
