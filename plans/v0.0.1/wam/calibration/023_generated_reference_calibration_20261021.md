# 023：直接用WAM生成参考拟合局部状态校正

本项在任何新拟合前登记，独立于020/021/022；不根据022尚未完成的文本结果选描述或参数。008三标量及018世界角漂移未通过各自门槛，其原失败保持。本项检验更具体的假设：原始生成的平移/朝向分量可能具有跨轴、跨通道的可校准结构。它是生成参考校正，改变checkpoint参考；不是机器人响应反解，不能认证原始WAM权重或把学到的轨迹先验归为原checkpoint能力。

只读原8400/g1旧六开发episode缓存，完全沿用008/018的episode/frame/seed去重44query、整episode六折、固定跨episode shuffle donor及共同H8/H16/H40有效前缀352/704/1398。六个episode均已被查看，称开发交叉诊断，不称新独立验收。0新模型、0物理、0fresh。部署输入仅当前measured anchor位姿/开度、已生成world参考及其固定20Hz端点时间；无task/episode/phase/object/contact/成功谓词，也无后续实际state。未来真实state只作fit分区的监督标签及test分区的评分。

每个query以当前anchor表示生成参考的局部位移u=R0.T(p-p0)和右相对旋转v=Log(R0.T R)。时间tau=(j+1)/20秒、最大2秒。固定特征x=[u/.1m, v/.5rad, tau/2s]共7维；无额外常数截距。监督残差为真实未来上述六维归一化特征减生成特征。用逐episode/query/有效endpoint等权且总和1的权重，解带lambda=.1的weighted ridge W(7×6)，以零残差/identity为先验。闭式解与augmented least-squares独立核验，不选择lambda或轴变换。

执行校正delta=(xW)，分别还原平移米、旋转弧度；按每端点tau限制平移校正L2≤.05m/s*tau、旋转向量校正L2≤.25rad/s*tau。输出p0+R0(u+delta_u)，R0 Exp(v+delta_v)，不再串行积分被修改的delta，不移动其它通道的世界锚点。原/校正后的anchor-relative旋转norm≥pi-1e-6均明确拒绝，不自动换branch或改门槛。W为全零时pose必须原字节复制，避免identity重组合舍入。

开度另有2维特征xg=[g_pred-g0,tau/2s]，以真实开度减生成开度拟合零先验weighted ridge w(2)，lambda同.1。校正幅值≤.25/s*tau，最终clip到[0,1]；raw/identity/所有对照在统一metric边界使用相同opening clip。合计44参数，不以当前任务或阶段切换参数。输入、特征、模型和输出均留副本，拒非有限/形状/clock错误。

六折每折只用其余五完整episode拟合paired/noWAM/shuffle三组同结构参数，共18组；再用全部六episode和预先固定共同前缀拟合第19组候选，仅存参数，不算留出成绩。noWAM输入恒anchor位姿/开度，仍能学共同时间先验；shuffle用008固定其它episode的生成局部参考重新锚定，所有方法相同有效前缀，不借held标签选donor或长度。它们用于区分WAM信息与训练示范先验。保留raw、identity、hold、历史匀速外推、paired/noWAM/shuffle七方法。

固定推广门槛：paired在H8/H16六项位置RMSE/SO3 RMSE/opening MAE都不劣raw，H40三项亦都不劣；位置/10mm、SO3/.05rad、开度/.1的逐endpoint平方和取均值构成共同分数，H8/H16都比raw低至少10%，且各自严格优于hold、历史外推、独立noWAM与shuffle；至少5/6 held episode在两个短窗口分数均不劣raw。比较不加结果后容差。报告全部分量/任务/episode/阶段与分数，不据某项改善省略退化。通过仅允许另立统一闭环验证协议，本项没有物理推广授权。未通过则不以本结果调整尺度、ridge、cap、特征或拆任务后重跑。

实现放新增可复用算法/实验入口/独立审计文件，不改任何020/021/022已冻结输入。先几何、世界基变换、非交换旋转、identity、副本/时序因果、cap/clock、fit-only分组的必要CPU测试，再0fit准备和独立来源审查；只有固定登记/来源/权重/44query共同前缀全部过才执行一次19组拟合。独立审计从原缓存重建，不能调用生产fit/apply/eligibility，把姿态重建精确或state误差改善当任务成功。
