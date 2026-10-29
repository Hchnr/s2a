# 047：D0+D1的固定W2与全量独立QR

本项继续036完整目标，沿用038预定的两轮clean readout，不增加候选或使用validation/fresh调参。当前D1仍在采集，本项仅实现准备；D1完整收尾前不得生成W2执行准入、解码拟合标签或运行solve/QR。

## 必须先取得的实际证据

046已准入的同一D1必须完成300请求/10500节点及10500 head calls，原完整独立capture检查器通过全部节点与D0初始条件exact；原D1 owner、046 wrapper及持久化supervisor成功，所有实际Popen wait0/reap/no-signal、PID birth和GPU释放收据齐全。完整519源/快照保持SHA。前2请求审查、部分ledger、进程存活和日志计数都不能代替整轮资格。

W1沿用045唯一合格参数，原041/043失败与既有尝试保留，不重采D0/D1、不重拟合W1。D1对W1的引用必须精确指向045 fit/verification/params；先验失败owner不改写。新W2配置只在这些真实引用可取得后绑定，不填造未来SHA。

## 固定数据、数学和运行时

原041 prepared/registration保持身份，训练为同150锚点×2noise×35sigma×40tokens×D0/D1，共840000行；两轮全局权重沿原038，D1输入特征用raw C0，不能改用适配后的velocity重建C0。1290特征、10维目标、全局中心/向量L2 RMS、ridge=.1和bias正则均不变，标签仍为训练episode未来状态离线监督，不进入在线采集输入。

只调用原`fit_wam_clean_training.run_fit(stage="W2")`和原`verify_wam_clean_training_fit.verify_fit(stage="W2")`，数学源码字节不改；实际参数比较仍rtol1e-9/atol1e-11，全部行身份指纹一致。继续使用045实际合格的`outputs/libero_env311/bin/python`：Python3.11.15、NumPy1.26.4、SciPy1.15.3，解释器及两个数值扩展实际文件SHA逐次核验。空CUDA/LD、三线程8；0新模型/denoise/physics/fresh。

## 新输出、次数与执行链

W2根为原prepared.root下`runtime_w2_fit_20261024`，生产`fit/W2`、独立`review/W2_verified.json`；原所有W1及失败目录保留。控制、准备、QA和源码快照均在新W2根内。新生产尝试最多1、成功fit最多1、独立QR最多1，不自动重试；clean协议累计生产尝试最多3（043失败W1、045成功W1、本项W2），成功fit候选最多2（W1/W2），不含另有历史协议的早期实验。

入口只做有限的W2阶段、双collection/audit引用、运行时、路径及metadata准入调整，不复制或改写原数学函数。新source/snapshot扩展046原519闭包；未改当前采集的任一源文件。预先固定不可执行的policy，实际完整D1引用只在后续final配置/提案中出现。

1. 先实现并核W2双轮调用/行数、失败拒绝、有限模板变换和真实CPU收尾；原数学W2/streaming/独立QR测试按实际需求复用。
2. D1完整资格成立后，固定真实配置、source/snapshot、参数和运行时指纹；独立metadata CLI发布分离proposal/verified/admitted。
3. 实际admitted两个role dry各通过原完整W2 `load_context`，840000行/两轮/0decode、source-proof回调1次并恢复；owner dry固定串行两角色及原7200秒每角色上限。
4. 单一持久化CPU supervisor生产→独立QR，真实wait0/reap/no-signal；失败保留并停止。未合格的W2不得向下一阶段传递。

通过本项只获得W2数值资格。后续须将新参数和owner显式接入完整生成诊断及固定开发任务；原适配收益/任务门逐项报告，开发门通过后才能冻结统一候选并打开最终预留24条及native/oracle正控。最终≥20/24、每任务≥4/8门槛不变。
