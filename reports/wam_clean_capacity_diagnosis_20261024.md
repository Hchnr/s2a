# 开发失败阶段与clean读出容量诊断

总目标不变；当前raw稳定1/6、W2稳定0/6，fresh关闭。本次只读诊断不拟合新参数或运行模型/物理。

## 抓取失败与训练覆盖

入口 `scripts/analyze_wam_grasp_coverage.py`实际检查全部150个注册锚点的原生未来40步及wine四条开发日志。产物`outputs/clean_training_development_dependency_repair_20261024/qa/grasp_training_coverage_v2.json` SHA256 `671f67fb4841552c3403d76d7f1322186b660a689ab702cb14519853dcd9baef`。所有NPZ与注册SHA匹配。首版结果保留，v2仅更正导入格式并重新绑定源码。

wine/raw/seed43没有任何闭合命令，开度参考0.8391–0.9880；wine/W2/seed43已闭合保持308步但目标接触0、最大高度增量0.01176m。wine/W2/seed42接触也为0，raw/seed42接触35步、最大高度增量0.05284m但未完成任务。因此不能以统一释放修改覆盖这些失败。接触与高度仍不是成功判据。

每任务50锚点中，未来窗口含闭合命令分别36/25/33，含打开命令47/43/49，含负值打开切换20/20/23；未来实测开度曾≥0.9为43/43/49。事件窗口可重叠，不能视作独立示例，持续加载命令也不等于抓取阶段。训练不是完全缺少闭合或释放标签；离线覆盖不证明开发状态与训练分布相同。

## 正则化容量的可核验假设

实际W2采用1290维特征、归一化权重和1、全向量L2 RMS归一化、ridge=0.1，且bias同样正则化。依据`src/s2a/wam_clean_streaming.py:159`、`:194`，标准化特征协方差的trace按定义约为1，而不是1290。因此若特征协方差特征值为λ，线性平滑器feature自由度为Σ λ/(λ+0.1)，由凹性上界1290/(1+1290×0.1)=9.9231；精确单位权重和中心下另加bias自由度0.9091。

分析记录`outputs/clean_training_capacity_diagnosis_20261024/analytical_capacity.json` SHA256 `774c0738d7aad59e22f60ac864cd142e87f162295a6bb02441756c1a83da44df`绑定真实参数与fit。该上界不等于实际自由度，也不能仅据此断言欠拟合或归因任务失败。原ridge是显式注册值，原W2计算资格未被推翻；不能改历史参数或用新λ冒充原训练。

下一步用全840000行已存特征复核原feature流SHA，计算实际谱与自由度，不读取训练future标签、不fit、不重采模型。若容量受限，再联合训练残差与留出诊断决定是否另行固定正则化对照；不能只凭训练误差变小选择候选。

## 全量实测谱已完成

`scripts/analyze_wam_clean_capacity.py`已实际完成600个请求、21000个compact节点、840000行特征；全量FP64特征流SHA与原fit的`fd6d7abf64a59967472dc695a121b2b3dbf2643b522d23242d15b5a474b6ce28`完全一致，全部消费输入前后SHA不变，实际进程正常exit0。产物`outputs/clean_training_capacity_diagnosis_20261024/empirical_capacity.json` SHA256 `7bd3ff1d2ad2a95d77785451ade92c426bd5745c3f77882bf5a939d8b90f4b08`。0新标签读取/fit/模型/物理/fresh。

协方差trace=0.9999999999999988，标准化均值范数=2.5684521908315e-16；实际特征有效自由度为 **5.294677117185618**（不含bias），低于9.9231上界。这是当前λ与数据谱共同导致的有限容量，仍不能单独证明欠拟合或任务因果。应先在同一缓存特征上评估现有头的训练残差及独立validation的误差分层，尤其开度与位姿阶段；只有原头误差与容量限制证据相符，才另行固定新正则化/特征对照。不能把降低ridge后的teacher损失改善冒充新采样或闭环收益。

## 全量训练残差与validation分层已完成

新入口 `scripts/analyze_wam_clean_residual_error.py`实际读取150个注册训练窗口并评估840000行，feature及target流SHA均与原fit完全一致；所有输入前后SHA不变，进程exit0。产物`outputs/clean_training_capacity_diagnosis_20261024/residual_error.json` SHA256 `e652961834aad12b9d49cbeb84cb4db023f6af89c984bc64e614d1385fbee4d4`。本项确有150个训练future窗口解码，只用于离线teacher误差；无新fit/模型/物理/fresh。

相对零残差头，训练归一化xyz三槽合并MSE降低2.9597%，opening降低1.9940%，rot6d六槽降低42.7172%。opening各任务降低1.2867%/1.7081%/3.2220%，D0/D1两轮opening分别2.0169%/1.9711%，不是只在某一轮失败。不同槽单位经normalization31，不能直接把这些MSE称米、SO(3)角或闭环state error。

已保存validation的h8 opening MAE在seed42三个任务全部变差：bowl .05335→.06976、wine .02303→.04362、cream .03247→.04285；seed43则分别.04481→.04063、.02442→.01096、.03431→.02132。没有跨seed统一开度收益，原STATE gate=False和validation初态同步未认证边界不变。分层是原真实采样误差的重算，不是新head的validation。

上述证据支持有限检验正则化容量假设，但不足以预言任务成功。052在任何新fit前固定一个CPU对照λ=0.1/1290，数据/划分不变，不扫描、拼接或修改原头；bias惩罚也随之变化这一混杂明确保留。计划见`plans/v0.0.1/052_clean_regularization_probe_20261024.md`。
