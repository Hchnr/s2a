# 215–216 反馈审查与统一限幅对照

本轮保持210唯一拟合参数和原任务验收范围，只改变饱和顺序。原213完整实验继续保留，已独立检查前7条均未通过；不据动作误差下降采用原候选。

215对固定当前状态、固定期望旋转，独立计算残差对desired position的线性Jacobian A，其对称部分特征值为[-64.73150,-23.29960,8.33407] command/m。原结构先裁剪物理解析逆，再加无界线性残差，物理逆饱和后其增益消失，无法保证残差仍朝期望方向纠偏。

实际物理逆的未裁剪位置增益D为每轴158.74021 command/m；固定64组Fourier项的Jacobian谱范数全局上界B为37.96448 command/m。因此`lambda_min(sym(D+A))-B = 56.04423 > 0`。8个预先固定记录状态的解析残差Jacobian与独立标量特征中心差分（1e-7m）一致。证据`outputs/direct_inverse_feedback_20261026/verified.json`绑定模型和来源。

该下界只证明裁剪前输入映射在固定当前状态/期望旋转时的正对称Jacobian性质。没有证明最终box/rate投影强单调，也没有证明动态闭环、接触或任务稳定。

216将`clip(clip(nominal)+residual)`改为`clip(nominal+residual)`。保持所有系数、标准化、特征、seed42、旋转/夹爪、接口、当前实际previous及±.25/20Hz限制；无新拟合。生产未裁剪物理逆和独立标量解析式、系数乘积及最终裁剪/physical_delta在原30条6180记录状态上全部一致，56行受变化率限制。原6条留出动作RMSE几乎不变，仅native29由.05650399变为.05647717；这些仍是动作诊断。

结构/输入/运行测试合计9项通过，包含提前裁剪产生反向命令而晚裁剪保持物理纠偏的实际数值例子、未饱和区一致性和实际命令变化率。独立运行检查入口首次因计划文件名替换错误而在读取来源时失败，未创建结果或执行物理；修正路径后完成核对，没有改参数。合格证据为`outputs/late_clipped_arm_inverse_runtime_20261026/independent_verified.json`。

真实执行器与checker均另存版本，原冻结源码保持。`outputs/late_clipped_inverse_control_binding_20261026/independent_verified.json`独立核对164记录状态上的新命令与完成后历史，并拒绝8类损坏；此处无反事实物理执行。正式配置`configs/late_clipped_inverse_control_registered_20261026.json`绑定这些证据及215数学检查。

完整准入119项CPU tests、18实际CLI dry-run和checker导入通过；13个本轮新增实现/测试文件Ruff与格式检查通过。真实执行namespace为`outputs/late_clipped_inverse_control_20261026`。已实查supervisor PID288784/start_ticks588725986、owner PID288877/start_ticks588726581、首条worker PID288935/start_ticks588726981身份一致存活。全部CPU/OSMesa显式空CUDA/LD，模型调用与新拟合均为0。

首条正在执行，尚无性能结论。仍须原18/4907、36stage+owner实际wait/reap及来源closure，以及18/18稳定、每条位置RMSE<=.01m/SO3 RMSE<=.05rad；任何一步尚缺都不能进入WAMdev或最终24新初态，总目标未完成。

首条随后完成独立检查：采集和checker均实际wait0/reap且无信号，机械/标签/图像及命令/历史审查通过。性能仍失败：位置RMSE .01903077m（原213为.27038684m）、SO3 RMSE .01874191rad，83个周期成功、末尾连续38周期成功，未达原稳定门槛，位置也超过.01m。不能将明显减小的发散或出现过成功称为可靠完成；原整批继续不改参数。

本轮结果支持饱和顺序是一个可复现的结构问题，但正的静态目标反馈下界未保证动态响应与持续任务成功。后续须根据完整真实轨迹审查残差对当前速度/历史的反馈、实际误差峰值与任务阶段，不依据单条结果重选超参或缩小门槛。
