# 091 冻结姿态惯性耦合诊断

090离线资格失败，未进入控制。下一步只检验实际OSC位置/朝向分块惯量是否提供可解释的局部响应；不新增拟合、不扫描参数、不改变控制器。源码osc.py中uncoupling分支将位置/朝向分别乘以各自操作空间惯量，再经完整J转成关节力矩；因此J M^-1 J^T的交叉块一般不为零。实际配置仍需核实，不能仅凭默认值宣称当前实例的uncoupling字段已直接测量。

读取086六个baseline anchor已有实际XML与qpos/qvel，只做mj_forward、mj_fullM和mj_jacSite，无mj_step。复现控制器的7关节M子矩阵、SITE J，设A=J M^-1 J^T，B=A blockdiag(pinv(Apos),pinv(Aori))。冻结该姿态，对位移/速度/恒定goal组成18维线性系统做dt0.05精确矩阵指数，得到translation goal对位移、SO3小角增量及末速度的局部灵敏度。按原实际normalized命令跨度/物理scale比较六anchor原四个响应矩阵，名义逐轴PD作为对照，不调门槛。

该模型省略接触约束、姿态变化、Jdot、关节扭矩饱和/空域项及有限旋转非线性。属于条件假设下的只读机制诊断，不是已认证控制模型；若性能差仍保存。需要验证M正定、J秩、SITE pose/twist与原数据一致，并对独立无耦合解析响应及耦合交叉项编写真正数值测试。生成产物使用新namespace，来源带SHA。物理积分/世界模型/fresh/参数fit均0。
