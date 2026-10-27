# 104 仅追加当前实测wrench条件（待103全量资格）

100为holdout22/24、局部22/24，101分阶段失败不采用。103正在采集原30轨迹直接wrench，完整30/6210端点/60阶段+owner资格前不得fit。本对照以100联合目标为基础，保留原4937方程、24fit/6holdout、72差分、0.5/0.5质量、ridge0.001、名义PD和固定64×83频率。只追加当前world-at-controller-SITE测量wrench6，不能加入恢复求解力、object/contact真值或未来wrench。

原83条件/非线性phi212完整保留；另外6维当前wrench仅做fit-only中心尺度归一化后附到phi，得条件89、phi218、action仿射设计872、系数872×12。原83统计逐元素保留冻结090值，新增6统计只用24fit相同轨迹等权/原4865当前行，不读末端未来wrench、holdout或探针label统计。保留100原非线性W对原83条件的映射，不重新抽频率或扩大非线性数目。

native按原index/split/source SHA绑定103当前样本，N控制行只读N+1端点的前N项。原9+新63探针从同native完整因果前缀取anchor当前wrench；七分支共用条件，不读probe未来wrench。103重放状态/命令/goal协议与原native匹配，另核对原六102 wrench共同前缀；097探针wrench属于该相同已验证native前缀绑定，不冒称原采集时直接记录。

先验证原100子空间、当前action仿射、未来标签/未来wrench隔离、缺少实际wrench拒绝。独立读取器从local及sensor pose/参考点重算world wrench，与primary保存值逐项核对；完整数据/特征/72方程实际parity后唯一全局fit1次，独立增广SVD另计。原6holdout×4和六anchor×4全门槛保持，训练/未训练各12项；性能失败不调整权重/阈值重fit。完整owner与退出/来源审查。物理/模型/fresh0，当前fit0；不接入099或新控制，原最终目标不变。

实现接续：追加条件核心、主/独立当前wrench读取器、9+63方程装配、fit-only六维统计绑定、实际parity准入、唯一fit/独立增广SVD、验证/owner/关闭入口已实现，导入/静态检查通过。当前103尚未完整关闭，未生成104占位合格配置、未装配全量真实新增条件、未fit。沿用100数值容差：原83统计1e-12；新增有量纲wrench统计及world读数独立重算绝对1e-10；特征/系数绝对相对1e-8；梯度最大绝对值<1e-9。性能门槛无松弛。完整冻结后执行，性能失败不自动调整参数。
