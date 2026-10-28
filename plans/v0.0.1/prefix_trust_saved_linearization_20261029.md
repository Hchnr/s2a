# 保存H12候选的前缀线性化诊断（CPU、零新响应）

总目标及18正确参考/12WAM开发/24最终留出门槛保持，GPU禁用。当前完整资产快照已编译、v2两臂独立复算正在执行；此诊断读取已保存producer数据，不替代独立数值资格，不启动新搜索或物理pilot。

本登记在只读探索之后建立，不能称为对探索结果的事前预测。探索已发现45组差分/45 callback，36个非零步的线性预测前缀约束接近边界，而实际前缀越界；以下将保存可重复的完整逐步证据，检验是否支持下一步采用完整响应回缩验收。

## 固定输入与算术

输入为`outputs/tracking_v5_prefix_trust_candidate_20261028/verified.json`，SHA`7338118bcad2cc80644555fcee42f9257b411cf9fa6234444fe1790601386d34`。该producer与owner已正常关闭，独立v2当前待定。源码、计划、数据与输出SHA均保存；原文件不修改。仅使用stdlib与NumPy，不导入求解器、MuJoCo或其他响应实现。

逐request解析同stage的FD base、72个probe与callback。核验命令bytes与response index精确对应，probe axis顺序0..71、nominal delta为原±1e-4且只改变对应坐标；只允许使用callback之前已完整生成的最新同stage差分组。不用未来组、不同stage或未完成组补齐。缓存命中仍需核对相同响应bytes。输出每组和每callback的request/response索引，保留零步记录并单独统计。

由保存tcp_errors恢复24个平方归一化分量：位置平方和除以.01²、旋转平方和除以.05²。对每轴按原nominal delta计算`J[:,axis]=(q_probe-q_base)/delta`；使用保存command浮点值计算`q_linear=q_base+J@(u_callback-u_base)`。输出全部24分量的预测/实际/残差，以及已达前缀和目标cursor的原始分量/最大值。prefix约束原定义仍为`q<=1`，原分类不得用诊断容差重新放宽。

对照每stage起始anchor的差分模型另计算相同callback预测，明确其不一定是优化器当前QP模型；只有最新组预测可以用于解释当步线性化。若无法精确确定最近完整组或anchor组，必须失败或明确数据缺口，不猜测。九组安全约束只有少量完整审计、其他trace只有标量minimum，不能据此恢复完整安全Jacobian或认定某关节是所有失败原因。

## 输出和判断范围

新脚本` scripts/diagnose_prefix_trust_saved_linearization.py`输出独占JSON，保留逐callback和逐stage记录、输入/源码引用以及零新响应/查询/数值内步/物理步/模型调用。用小型合成线性例核验差分方向、最新同stage组匹配及零残差，并拒绝不完整或被篡改的ledger。

同时报告线性预测超限的原始符号及量级；若仅为机器舍入量级而实际偏差显著较大，只能得出“线性模型把前缀压在边界、完整响应越界”，不能宣称实质线性不可行。比较半径时逐stage列示，不能预设半径减半会单调减小偏差。不同callback目标误差与各自基点、同一anchor分别比较。

该诊断只能定位已保存有限搜索的拒绝机制，不能证明目标不可达或新算法必定成功。下一候选若采用callback方向的完整响应回缩，需另行固定预算、缓存/计费和接受规则，保持原完整witness与安全条件，通过真实合成例和完整独立复算后才能考虑物理pilot。
