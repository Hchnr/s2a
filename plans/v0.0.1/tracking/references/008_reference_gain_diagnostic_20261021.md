# 三参数参考校正的开发可行性诊断 — 2026-10-21（Asia/Shanghai）

状态：仅CPU、只读缓存，0新增模型请求/物理步；不改变005/006运行源码，不启动新初态或学习后闭环。它是独立参考模型假设，不是原checkpoint认证，也不是机器人响应逆。

剩余state误差与执行误差须分开。现有6episode/22distinct anchors不足以支持通用网络，却可用来否证一个极低容量的固定偏差假设。只使用8400/8800/15400 guidance1.0已有各48请求；按(episode,anchor_frame,sampling_seed)去重，每组44条，不把重复phase或重叠endpoint当独立样本。

校正输入仅当前实测anchor、已生成world TCP参考及opening。对anchor相对位移统一乘一个非负标量g_p∈[0,2]；对`Log(R_anchor.T @ R_pred)`统一乘g_r∈[0,1]，再由anchor重建SO(3)；opening加一个b∈[-.25,.25]后clip[0,1]。共3参数，无task/object/contact、episode/frame/phase标签、绝对位置特征或未来真值在线输入。旋转处理保持生成的world位置路径（除显式g_p），与旧逐delta衰减造成的位置耦合不同。所有单位和左右乘约定在实现中固定，π附近分支不猜测。

每checkpoint分别做6折leave-one-episode-out，所有anchor、重复窗口和seed必须同fold。拟合只读取其余5episode的未来state监督，heldout episode不参与拟合/归一化/选择；其结果仍为已查看开发数据的交叉诊断，不叫独立最终测试或预训练未见。先固定正则λ=.1及尺度position .1m、rotation .5rad、opening1，不搜索超参；各fit episode等权、各独特query等权、各有效endpoint等权，不包含padding。对增益使用identity prior，对b使用0 prior，按固定box约束闭式解。

同时报告raw WAM、identity精确复现、仅g_p、三参数校正、hold、历史恒速，以及相同三参数模型接收zero-motion/current-opening的noWAM。另做shuffled-WAM：fit输入只在fit queries间按seed42固定无同episode的排列替换，heldout的替换输入只取fit episode缓存；训练与评价标签保持原query，重新拟合同容量三参数模型。shuffle是离线破坏配对控制，不声称可部署。不同长度只取共同有效endpoint，不补未来；评价全部方法共享相同有效endpoint。

计分H8/16/40的位置RMSE、SO3 RMSE、opening MAE，并保留每fold参数与误差、原始输入/源码SHA、去重证据。禁止根据外折结果重新选择λ、参数范围、episode或checkpoint。任何新拟合结果都不自动推广；若不能一致优于raw及noWAM/shuffle/hold/历史外推，不投入新的校正器物理预算。若有方向性收益，仍需更多整episode独立数据、执行资格和冻结WAM闭环另行注册。
