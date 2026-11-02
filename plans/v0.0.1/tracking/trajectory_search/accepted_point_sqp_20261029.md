# 实际接纳点曲率保留的显式 SQP（CPU，执行前登记）

总目标及原18正确参考/12WAM开发/24最终留出门槛保持，GPU禁用。原actual368、completed367命令/旋转latch、已生成chunk46原H12、完整96资产模型和共同warm不变；不读取旧actual369+作为控制输入。不得修改已执行旧算法、测试、配置、计划或结果。

## 已关闭的前置证据

固定优化余量搜索及原tool28112完整独立复算均已正常wait/reap0：4000响应、1200000内步、1248001查询、177完整ray和54anchor，严格分类无分歧，prefix9、完整witness0、实际cursor8/task0/grasp0。checker SHA为`81f60ce7d9c8ceeb455f2309472c0f8a5854448b158c1bd82a4f028ef2054876`，checker owner SHA为`f34e0eaf207c54afb1a679c9c3396b3047793211008b468a0d95bda440a43277`。最终关闭记录`outputs/prefix_optimization_reserve_final_closure_20261029.json` SHA为`05e728b1d49919ee3b549ac6d83e66d42b8fd091605cb512d2e19cd36f624ea4`，2283引用和38份QA来源重读不变。

旧53个完整阶段各在首个非零callback截断并新建SLSQP状态，3822/4000响应为FD；最大目标平方误差只消除初始超限量2.57%。callback不保证内部线搜索已完成；46/53个接纳点低于优化器1e-4余量。这些是可核验事实，不证明重置是唯一原因。新机制同时改变完整QP方向生成和曲率生命周期，结果不得单独归因于BFGS。

## 唯一候选机制

新增`src/s2a/accepted_point_sqp_core.py`及`src/s2a/accepted_point_sqp_plan.py`。保留原响应计算、SO(3)误差、完整审计、1e-8优化安全buffer、已达prefix的1e-4优化余量、原10mm/.05rad实际caps、0.25命令rate、0.1s制动、原terminal normals和历史误差profile。普通ray仍按原完整joint/command审计、原prefix、不小于1e-8安全余量以及大于1e-12的目标下降（或prefix增长）接纳。完整witness不额外要求优化余量，仍须全部H12原caps及完整审计。

在当前实际接纳命令向量x计算原同一套1e-4单边FD，符号由当前stage上界决定，所有probe都执行完整H12响应并纳入原缓存/预算。保存基点和逐轴请求关联，完整stencil形成后才能更新曲率或生成QP；部分stencil不得更新B。

QP变量为z=[d,t]，t为绝对非负epigraph。目标为`t + 1e-4*grad(sum(q)).d + .5*d.T*B*d`，B初值为单位阵，只对命令块建立曲率，不给t增加二次项。全部约束统一为`A*z+b>=0`：非线性安全、原prefix优化余量、当前目标两项epigraph、原rate、命令box与stage半径、t>=0；安全和pose只在当前接纳点做一阶展开。QP使用公开SciPy SLSQP的解析目标/Jacobian完整求解，内部不调用响应模型；每次最多100个内部代数major iteration，ftol固定1e-14。

QP行缩放固定为`scale_i=max(1,max(abs(A_i)),abs(b_i))`。对scaled slack不超过`10*tau`的全部active行，通过公开NNLS恢复非负乘子；最大迭代为`3*max(active_count,1)`，无自动重试。证书tau固定1e-6；要求scaled primal最小值>=-tau、dual非负，stationarity infinity残差除以`max(1,||grad||inf,||A_scaled.T*mu||inf)`不超过tau，互补积infinity残差除以`max(1,abs(QP objective))`不超过tau。另保存raw/scaled残差和互补积总和。原单位lambda=mu/scale；它只是有独立残差证书的近似局部QP乘子，不是精确物理乘子。QP后端success及全部证书必须同时通过；失败记录`qp_certificate_failed`，不改阈值、不称不可行或收敛。

仅对合格非零方向沿原九个alpha（1至1/256）进行完整响应审计，接受第一条合格ray。普通接纳仍使用实际回缩命令，不把QP点当成实际接纳点；所有缓存中的完整witness优先。QP方向若超stage命令边界1e-12容差，不裁剪后冒称原QP方向；本stage拒绝方向。半径仍依次.01/.005/.0025，接纳后回到.01，无方向或ray失败进入下一半径，耗尽即停。

在同cursor的下一实际接纳点完整FD完成后，使用前一QP的同一组原单位lambda计算两端Lagrangian梯度差：`y=(g_new-J_new.T*lambda)-(g_old-J_old.T*lambda)`，其中只需非线性约束部分，线性行Jacobian差为零；s为实际接纳ray位移。固定Powell阻尼eta=.2，更新`B-Bs*Bs.T/(s.T*Bs)+r*r.T/(s.T*r)`，对称化并Cholesky检查。负s.T*y进入阻尼，不自动跳过；不以FD间距作为最小s门槛。精确零步、非有限/非正必要分母或新B非SPD时保留旧B并记录原因，不谱修补。拒绝ray、普通下降FD和部分FD不更新。cursor增长时B重置为I、清空跨cursor待更新pair；半径缩小时保留B。保存更新前后B、实际s、y、lambda来源、阻尼、条件数及||s||inf/FDstep；正定性不等于真实曲率正确。

## 预算、验证与执行边界

唯一候选最多4000个完整唯一响应、60次非线性外层尝试、60阶段，原H12每响应300数值内步/312查询，缓存计数和最终全缓存排序保持。QP内部代数迭代独立计数，最多100/阶段、6000总计，不能混称为旧SLSQP的非线性major iteration或新增模型响应。预算中断保留完整已计算响应、部分stencil与已确认witness，不自动追加预算。正式producer限1200秒、独立checker限5400秒，exclusive新输出，无自动重试。

先用有解析解的约束QP验证乘子符号/缩放、bounds、active epigraph、冗余dual与不可行/未获证书处理；用解析曲率验证BFGS secant、负曲率阻尼、SPD、实际缩短ray、拒绝与cursor切换状态、部分FD预算。再做真实合成响应的整体求解测试。测试不构建物理环境，不引入旧未来观测，不把合成成功计入控制结果。正式native QA、Ruff/format及producer/checker实际dry和来源闭包全部通过后才允许唯一真实数值搜索。

新的独立checker需重建全部唯一响应、ray和anchor审计，并从完整FD关联重建每阶段QP/近似乘子证书、实际s和Lagrangian secant及BFGS更新；不能调用生产核心作为唯一算法正确性依据。trace另保存完整非线性安全向量，answer保存previous_command并与原completed367命令精确绑定。算法代数核验从保存原精度TCP误差/安全向量重建；各向量另外按原数值复算容差与独立响应比较，严格分类仍须完全一致。独立Python与原JIT浮点路径的微小差异不被冒称为完全相同的FD/乘子，但不得因此放宽原caps或跳过数值复算。新增账本和组件QA不替代全量数值复算或实际控制。执行/检查入口尚未就绪时不运行真实响应。

仍无完整H12 witness，禁止物理pilot。已关闭的运行时文件快照只具备文件资格；实际动态加载、bootstrap和原18/12/24目标均未完成。GPU恢复需用户明确通知。
