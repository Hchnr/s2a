# 完整跟踪瓶颈：保存模型的逐cap支持下界

2026-11-02。small-progress完整核验通过但仍12/24、prefix2；停止阈值变体。
本阶段只分析已关闭生产结果，不启动响应模型/物理/GPU，不修改或下发命令。

固定分析small-progress初始anchor0和最终anchor23。使用已审核coupled stencil，
按保存的因果secant更新重建各anchor的Js/Je。对每个原超标cap，沿当前三维
归一化误差单位向量n，最小化nᵀ(E+Je*d)。任何满足该cap的局部线性轨迹都必须
使该支持值≤1；支持值>1只能排除这个局部线性模型中的可行性，不能证明物理
不可达。分别使用原实际buffer和内部reserve .9，区分安全与算法余量影响。

约束保留全部4369线性安全行、原root global命令域和.25变化率。取消局部trust
radius和pose ceiling，使其成为乐观松弛；不把分析中的松弛写回生产配置，所得
LP点不具备原执行资格。每cap独立优化，单项可行不代表共同24caps可行。
原安全terminal权重.1、原buffer1e-8、目标1e-5与误差尺度.01m/.05rad保持。

上限48LP（2anchor×每组至多12个超标cap×2floor），每次time_limit10秒/
maxiter10000，owner1200秒TERM/10秒KILL，不自动重试。0QP/NNLS/新拟合/
响应/查询/内步/物理/GPU/WAM/fresh/最终留出/参考消费。计费只登记实际LP。
新独占outputs/local_tracking_support_20261102，输入SHA、完整LP解/对偶、
逐行及分组诊断、源码前后快照均保留。CPU空CUDA/LD、各数值库线程1。

独立代数检查每个LP的原始矩阵primal slack、非负dual、stationarity、原单位
目标gap（归一化容差1e-8）；失败/超时记录失败，不作为下界结论。另有解析
小问题验证对偶符号、原单位缩放、篡改拒绝和松弛不可达/不能证明共同可行。
成功只说明分析账本可信，不授予控制资格。根据实际支持下界与误差时刻决定
下一步是改局部方向/模型还是审查参考与固定接口，不继续盲调标量进展门。
