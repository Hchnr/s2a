# TCP误差向量 Gauss–Newton 的前瞻对照

本计划在新QP或模型响应之前登记。前项60阶段在相同122响应上限内用了90响应，
最坏超限F较24阶段下降6.51%，但总loss增加11.07%、双caps点6→4，完整witness0。
保存数据的因果向量预测诊断已独立关闭：88条事实方向的q MAE/RMSE均优于原标量
线性模型；它只是后验预测证据，不能证明新方向、GN或实际控制收益。

本次从原3557完整响应及原72个signed nominal1e-4差分重新建立三套因果模型。
Js/Jq仍用原方法，新增E=flatten(tcp_errors/[.01,.01,.01,.05,.05,.05])和Je(72×72)。
E是局部数值模型的TCP残差，不是新的实际测量。Je行周期repeat(arange(12),6)，
所有未来影响必须exact0。compact差分可缺q/loss；完整base和新响应保持原字段及
q的平方代数验证。不加载旧90条响应、最终anchor或诊断拟合为初始模型。

pose阶段目标由maxq软epigraph改成固定阻尼1的正规化GN：
`B=I+Je.T@Je, g=Je.T@E`。删除原24条pose软epigraph行，仅保留t>=0且initial_t=0，
旧solver的`t+g·d+.5*d.T@B@d`等价于`.5*||E+Je*d||²+.5*||d||²`（略去常数）。
不把maxq epigraph和GN同时混入目标。restore阶段完整QP保持原样（B=I,g=0及
余量软epigraph）；pose的安全、共同q上限、box/rate/global/trust硬约束保持原样。
Jq独立维护，不用2E.T@Je替代。

实际state_loss定义为原24项q总和的一半；预测state_loss为`.5*||E+Je*d||²`。
pose的实际/预测改善和ratio使用此未加阻尼的loss，不把t或`.5*||d||²`混入ratio。
near-cap最小改善仍为min(1e-4,.1*F_anchor)，其中F=max(0,maxq-1)，不改用loss。
restore仍用原reserve violation。接纳优先级保持：原完整witness、安全/共同上限、
实际改善、ratio>=.1、接纳；loss下降可伴随F上升，仍不等于可执行。

固定共同数值搜索上限`max(1,max(initial_q))*1.01`（本case6.977797141162663）
运行期不变。它不是原执行caps。接纳及原合格拒绝反馈同时更新Js/Jq/Je，并记录
error_vector metadata。拒绝反馈仍仅从无接纳stage、实际安全且违反共同上限的
actual_admission_failed记录中选最小alpha；不选ratio失败，不另加ratio门，
不移动anchor。所有方向试探完成后才更新模型，因果有界Broyden及微小步政策不变。

数值config逐字节等于stage_budget_search_20261029.json：seed42、60阶段/122响应，
每QP最多100迭代、原1e-6证书；alpha[1,.5,.25,.125,.0625]、半径
.005/.00015625/.02、命令全局位移±.1、所有buffer/ratio/停止条件不变。
完整初始响应和末次初始repeat均收费；最多120trial。两路线最坏244响应、73200
数值内步、76130查询。技术失败保留已付费前缀，无自动重试、追加预算或证书放宽。

输入仍actual368实测q/v、completed367命令/rotation latch、固定finger、history25×7，
仅原已生成12参考。输入bundle SHA为
`eadf18a3ed90c29ee70d773741cc310304f4385189baea360e534ca6ef5bd023`。
原固定控制器、20Hz/2ms、TCP/SO(3)/单位、位置.01m/旋转.05rad、box/rate.25、
关节margin.05rad、braking.1s、terminal及原audit与1e-8 buffer保持。

登记字段pose_objective_policy为normalized_state_vector_gauss_newton，pose_damping
为1.0；pose_ceiling_policy和feedback_policy保留。旧文件/结果冻结SHA，新增独立
namespace。QA与两条dry通过后先提交，再执行唯一新生产和完整独立checker。
独立ledger不用生产GN函数，重建E/Je/B/g/A/b、KKT、状态机、三模型更新及计费。
native独立路径复算每条收费响应，用新鲜实际q独立重判loss、接纳原因、半径、
安全、共同上限、严格原fullcaps、完整历史与repeat；不重新求解QP。

新旧初始响应及相同restore前缀应exact；首次pose后不要求相同方向。报告同时
列F、总loss、cursor9、12点双caps、前缀、位置/旋转、安全余量、所有QP失败和
成本，不因sumloss下降忽略最坏项。仓库已有GN/least-squares，本次是上述特定
H12状态更新响应、向量FD/因果更新/阻尼曲率及loss接纳的组合对照，不能把收益
单独归因于GN或宣称首次实现。完整数值witness仍需后续独立实际控制验证。

全部产物在当前仓库outputs，CPU显式空CUDA/LD和单线程。不访问旧清理根或用户
未跟踪文件。GPU继续禁用，新增物理/world-model/fresh/实际参考消费均0。
实际仍cursor8/task0/grasp0，原18正确参考/12WAM开发/24fresh及总目标保持。
