# 045：合格CPU运行时上的固定W1及独立QR

本项继续036完整目标，不以拟合或CPU检查代替开发任务和最终24条验收。044已在实际sim运行时完整核150训练窗口、300请求/10500节点及420000行，两条原路径字节相同，真实wait0/reap。043失败、main首窗口False、原源码及数据均保留，不修改旧结果或放宽阈值。

## 固定输入、算法和运行时

输入仍为原041 prepared/admitted、完整新D0 collection/audit/owner；原040中断数据不使用，validation/fresh不进入拟合。W1固定150训练锚点、300实际noise、全部35×40行，共420000行；1290特征、ridge=.1、归一化31和所有全局权重沿原038/041。

使用`outputs/libero_env311/bin/python`，Python3.11.15/NumPy1.26.4/SciPy1.15.3。生产和独立worker均核实际版本、解释器和关键运行时文件指纹，空CUDA/LD及三线程8。仅调用原`fit_wam_clean_training.run_fit`和原`verify_wam_clean_training_fit.verify_fit`，不改两套数学源码、数值容差或参数schema。继续使用已验证的源码proof回调context，退出恢复。

## 新输出和累计预算

新root固定为原prepared.root之内的`runtime_fit_20261024`：生产`fit/W1`、独立`review/W1_verified.json`，控制、额外worker与owner收据、源码快照也在新root。旧`fit/W1`、旧043控制根、旧owner及全部失败产物不改、不覆盖。原数学产物保留原prepared/041 registration来源身份，额外045收据明确新运行时、044资格与实际入口。

| 项目 | 上限 |
| --- | --- |
| 新实际W1尝试 | 1次；成功producer fit最多1 |
| 独立全量重算 | 1次、原augmented QR、同420000行 |
| 新模型/denoise/physics/fresh | 全部0 |
| 自动重试 | 0；任一失败停止并保留 |

累计明确记043已有1次失败fit_attempt；本项最多再1次，W1累计尝试上限2，成功W1候选最多1。041在`load_context`前置拒绝的CLI另记为失败启动，不能追称已拟合；044数据重建不调用solve/QR，不计作拟合。W2和D1仍未执行，也不由本项授权。不得依据新拟合结果加候选、换ridge或挑任务/seed。

## 执行资格和生命周期

1. 先固定新source/snapshot、显式配置与运行时指纹；原498源及快照保持SHA。
2. 在实际sim运行时执行原streaming/独立QR/readout相关合成测试，保留真实结果；只证明求解库及接口可用，不冒充实际训练。
3. 新独立metadata审查核044完整数值和正常生命周期、043实际失败及累计尝试、原041完整D0来源、新输出/源码/运行时、合成QA，发布与proposal分离的verification及admitted。
4. 实际admitted producer/independent dry均须通过完整原`load_context`，0数组decode/fit；唯一CPU supervisor按producer→独立重算串行Popen，actual wait0/reap及完整产物后才能继续。
5. 生产要求原完整420000行及150label参数结果；独立要求原逐行重算、replay身份、全部参数按rtol1e-9/atol1e-11一致，正常退出。失败不自动重试。

本项通过只获得W1参数与其独立重算资格。之后D1必须另接新owner及参数来源，仍经W2、完整生成诊断、固定开发门，才可考虑最终留出；20/24及每任务4/8门槛不变。

## 运行前合成数值诊断补充

实际sim原64项测试为60通过、4失败，原日志保留。4项均是同一nonuniform样例在4种分块下与NumPy `lstsq` 的严格`rtol=2e-12/atol=2e-14`比较；并非原生产与独立QR不一致。独立80位Decimal对FP64输入定义的同一ridge目标求解，原stream/direct误差≤8.89e-16、原QR≤1.12e-15，NumPy SVD参考误差3.94e-14，另列SciPy三种driver。该诊断不把原4失败改记通过，也不修改冻结测试、任一求解函数或实际拟合的`rtol=1e-9/atol=1e-11`。以已通过的原QR测试、该有界诊断及全部真实420000行的原独立重算共同核验045；若真实比较失败则停止，不以合成诊断覆盖。
