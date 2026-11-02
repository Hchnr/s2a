# GN整个目标的统一缩放对照

本项在新实现、QP及模型响应前登记。前项GN搜索7QP/3响应完整关闭，6个pose QP
均status8，未产生pose响应、完整witness0；保存矩阵诊断没有新QP，B条件数278.129、
原canonical A条件数6.292，不能断言严重病态或已证明失败原因。

本次仅改变pose QP交给有限求解器的目标尺度：每stage在求解前由原B/g/radius
固定计算C=max(1,||g||inf,radius*||B||inf)，将**整个**
`f(d,t)=t+g·d+.5*d.T@B@d`及梯度除C。首pose预期C=11.23197515727354。
不搜索C、不根据求解失败重算C、不改变原B/g及实际/预测state_loss定义。
restore阶段C=1，完整QP保持旧样；不改变任何约束行，继续原
S_i=max(1,||A_i||inf,|b_i|)行缩放、初点和100迭代/ftol1e-14数值选项。
这是有限求解路径对照，内部ftol针对缩放后的问题，不能声称全部数值行为相同。

统一正尺度在精确数学上保留原可行集合与最优解，不改变B条件数。若求解器对
原canonical约束给出变换后乘子nu，原canonical乘子为C*nu，原raw乘子为C*nu/S。
最终必须从原B/g/A/b、原S重算原目标、原primal/stationarity/complementarity，
原1e-6门和solver_success缺一不可。独立checker应直接从未缩放数据推导尺度与
dual映射；不得以缩放后的KKT替代原证书，不追认前项任何status8。
如通过变量变换实现整体缩放，须显式证明包含t的映射，并在原空间重新核验。

使用独占新namespace，从原3557和原72条signed差分新建Js/Jq/Je，不加载旧final
或诊断拟合，不恢复优化器。numeric config逐字节同GN前项：seed42、60stage、
122完整响应，每QP最多100迭代，无自动重试。初始与末次repeat均收费，任何原
停止条件优先，不追加本轮53stage/119响应余量。全部外层接受、state_loss/F阈值、
反馈选择/三模型因果更新、半径、共同搜索上限和原完整执行caps保持。

QA须覆盖整体目标/梯度含t、正C与restore不变、dual双向映射和原证书反例、
失败QP不产生响应、预算/接纳/历史/repeat。独立账本重建每份QP与状态机，原生
checker复算全部收费响应并严格重判原安全、caps、loss/ratio和执行资格。
统一QA、Ruff/格式、两条dry通过后先本地提交，再执行唯一有预算生产与完整
独立复算；失败保留，不自动重开。

报告同时列solver状态与KKT、实际成本、所有12点q/位置/旋转、F/总loss、prefix、
安全与full witness，不只比较成功退出数。无pose响应时仍不能评价GN跟踪收益。
若有完整数值witness，仍须后续独立实际前向控制；本数值对照本身不授予控制资格。

全部代码、配置、报告和outputs在当前仓库，冻结旧文件/SHA，保留失败证据，不
访问旧清理根或用户未跟踪文件。显式空CUDA/LD、单线程CPU，GPU继续禁用。
原actual cursor8/task0/grasp0、18正确参考/12WAM开发/24最终留出门槛及总目标保持。
