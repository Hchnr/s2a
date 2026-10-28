# 关节预测一致的FK预览（CPU）

已接收8点冻结J Cartesian解的独立FK仅5/8满足原精度，最大内部位置差17.01mm。本候选修正预测内部几何，不用几何一致性代替实际控制资格，总目标仍为原固定接口下可靠跟踪及final24新初态稳定任务成功。

`RobotSiteKinematics`仅接收明确的机器人关节q与固定compiled XML robot SITE祖先链，返回world SITE pose和几何Jacobian；不接收物体状态。缓存按相同静态模型及q，返回副本，不改变数据。`fk_consistent_rollout`保留原预测q/qvel及其动作导数，用FK(q)与J(q)qvel重建每个端点，位姿通过解析链式导数、速度通过固定1e-6rad中心差分dJ/dq链式导数获得。虚拟导数扰动不是实际命令，不裁剪以免破坏导数。原pose/twist目标权重、TCP变换、world/SO(3)约定保持。

新规划模块保留冻结free joint M/J响应与所有原box/rate、全内环关节/历史收紧、制动、terminal、LP残差准入及所有端点.01m/.05rad约束。初始FK pose/J必须匹配当前实测SITE/M/J坐标合同；每次候选用新FK一致rollout计算精度及目标。共享160响应/60迭代，可行性阶段最多30，不修改原已执行源码。修正运动学不等于修正接触、饱和或未来M/J动力学。

新组件共5项CPU测试通过（1.25s）：非交换旋转/非零TCP偏置/非零速度下完整pose/twist residual Jacobian与目标gradient对中心差分；robot-only FK/J与独立raw-state/速度/有限差分一致；缓存/原joint响应不被改写；返回命令的每端点FK、速度、动作/关节/制动/terminal与原精度重查；初始frame不匹配拒绝。Ruff/format通过。

数值入口`execute_tracking_v5_fk_consistent_trajectory_numeric.py --launch`，同一实际368、同一当前机器人M/J/q/qvel/历史profile与已生成chunk46，固定8/12/16三窗口、原warm重复末命令延长，最多480响应。与原长窗口的实际q/qvel/history逐项一致。记录真实计算耗时及FK查询/缓存数；每个返回witness另外用独立XML raw-state FK核验全部端点pose、world SITE twist及TCP误差。新增物理、模型调用、fresh均0。只有完整合规模型解才考虑同一接口原生前向验证，不能把搜索/协议通过写成执行或任务成功。

GPU持续禁用，原18/完整12WAM开发/final24门槛全部保持。当前数值结果待实际执行。

数值及MuJoCo交叉核验均已actual wait0/reap0关闭：8点完整witness（71响应/4.5847s），12/16无witness（160/114响应）；8点最大误差.00247193504m/.02400348397rad、全部端点FK一致。36构型MuJoCo姿态差≤2.55e-15、J差≤1.78e-15。numeric/check verified SHA分别443e1661345459ecc4a91a468971e49ed332371c3b8e470d7e0ae98667326b84、679a0f119a8d05300442921f29caa4f2a42e24440f2d00d463ba4266cd581cb1。物理/模型/fresh0，不取得实时/执行/任务资格。[完整结果与下一原生反馈](../../reports/fk_consistent_trajectory_20261028.md)。
