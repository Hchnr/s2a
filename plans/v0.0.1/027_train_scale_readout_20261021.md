# 027：训练集尺度下的冻结主体readout

在任何本候选fit或模型运行前登记。原025与026均保持teacher gate False，原参数/缓存/配置/源/门槛不修改。稳定成功仍要求统一冻结候选在预留12初态×seed42/43中至少20/24、每task至少4/8、末40端点连续native成功；fresh性能尚未打开。本项是向该目标推进的CPU学习诊断，不是缩小目标或认证原checkpoint。

只读缓存统计提示当前prior强烈压缩hidden变化：025消去bias后的hidden有效自由度约.316、026约2.60；026 readout相对bias额外改善3.02%，但sigma=.8仍退化13.23%。这些结果支持检查输入尺度，不证明hidden有足够任务信息。为区分原因，本项不同时添加独立sigma头、更多EP或新时间函数。

只复用025已核验的144份原teacher缓存、026实际fit/独立重算及时间参数化bias。WAM保持8400 non-EMA/H40/20Hz/g1/35步/shift5/SDPA/active10。六train EP1172/1098/893/1151/898/850，六开发validation EP1254/1126/868/1139/1209/1176，原48anchor/3sigma/seed42/全部endpoint与逐层等权不变。validation已查看，是开发诊断，不称预训练未见或新无偏验证。

唯一改动是train-only的全局中心与尺度。对原训练权重w（sum w=1）和FP64扩展的原BF16 hidden，定义mu=sum(w*h)，s=sqrt(sum(w*||h-mu||²))；拒绝非有限或s<=0。新phi=[(h-mu)/s,1]，A=-phi/sigma_actual，目标仍为active10(v_target-v_base)，解sum(w*||A W-Y||²)+.1||W||²，bias同样正则。s是整个hidden向量的一个L2 RMS，不逐列白化、PCA、截断、加epsilon或按validation选尺度；不放大某个低方差列相对其它列的权重。sigma取缓存FP32无损扩为FP64，不用名义小数或整数t推算。

因此训练集sum(w*||(h-mu)/s||²)=1，不再额外除sqrt1280。加上1/sigma后，hidden与bias的Gram交叉项一般仍非零；中心化不保证消除时间冲突。正则bias下，中心化本身也会改变prior，不能宣称惩罚不变或已证明尺度是原失败的唯一原因。

只拟合一个1281×10 W，共12,810个系数；mu1280和s1亦作为训练得到的预处理参数保存，总14,091个数。与026相同数值lambda因预处理改变而改变prior，此为明确的新候选，不是修正已证实bug。此候选仍表示clean-action残差，保留实际velocity损失和-1/sigma形式；不从BF16 xt构造新clean标签。

对照原样复用026已拟合、已独立核验的time-aware bias和原5760行实际量化预测，不重拟合。其hidden系数为0，中心/尺度对该对照无影响；必须逐值证明用原bias系数及本次实际sigma重算得到相同预测，并核其train/validation/权重/target身份。原026结果仍False，不因复用对照改写。新预算为**一个生产ridge solve及一个独立增广最小二乘重建**，不是两个生产fit，也没有控制组重试。

生产和独立路径在FP64按先减mu、除s、拼bias、再除实际sigma、矩阵乘W、加原base的顺序计算，最后作数学单次BF16 ties-even。其它118槽逐值不变，zero候选返回原base对象，其它domain不应用。mu/s只能由六train的2880行计算；validation的2880行不参与任何预处理或fit。在线数学表达只依赖原当前hidden/base与实际sigma，不使用future标签。

本项依次完成实现/必要CPU因果与数值测试、冻结源码和新prepared/source snapshot、独立prepare-only核验、owner实际wait/reap的一次CPU生产fit及独立重建。prepare/fit/teacher审计不重新打开原EP像素/state/PNG、编码future视频或加载WAM；原EP引用仅继承旧来源证明，实际input SHA仅记确实读取的缓存、source、配置、计划与证据。所有新文件和产物均在S2A，旧72源、外部Cosmos及数据/权重保持原字节。

Teacher门槛原样：开发validation pooled<=.95raw，严格优于原time-aware bias，各sigma<=1.05raw。参数/评分重算容差只用于实现数值对照，不能放宽质量门槛。任何失败完整保留，不换lambda/中心/尺度/特征或再fit。即使teacher通过，也只取得另行登记真实GPU运算与完整生成资格研究的条件，**不能据本项启动WAM或物理**。

如teacher通过，下一项另登记新路径的GPU6合成精度、3份zero完整输出exact、原96份validation成对生成及全部原state门槛；完整生成满足state门槛后才登记三任务双seed开发物理，再考虑冻结到fresh24。本项GPU/WAM前向/完整生成/state标签评分/物理/fresh预算均0，CPU显式空CUDA/LD。后续GPU只允许4–7。已有产物拒绝覆盖，无自动重试，只清理实际owned子进程。
