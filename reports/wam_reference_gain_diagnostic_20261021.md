# 三参数参考校正：开发诊断与独立核验（2026-10-21）

008的校正后缓存诊断通过独立全重算，但**不promote，不追加校正器物理预算**。三参数校正只有有限、跨episode不一致的位置收益；朝向仍输给保持当前state，opening在两组checkpoint上反退。这些6个已查看开发episode不能验收稳定任务能力。

## 设计、数据覆盖与隔离

固定8400/8800/15400 guidance1.0缓存，每组原48请求。按`(episode, anchor_frame, sampling_seed)`去重后44条、6episode、22个不同anchor，重复项4条均要求预测及标签逐值相同。每checkpoint做6折leave-one-complete-episode-out，全部anchor/窗口/seed随episode隔离，不将seed、重叠端点或phase当独立场景。

拟合仅用其它5episode；所有超参数在`1aa65aa`预注册，λ=.1，尺度为位置.1m、旋转.5rad、opening1，未用外折结果搜索参数。各fit episode等权，该episode内各唯一query等权，query内共同有效endpoint等权，即每端点权重为`1/(5 × episode内query数 × pair共同端点数)`，总和1。

shuffle只在fit queries间跨episode置换，test donor也只从fit episode取；标签保持原query。独立重算核对了每fold的完整配对、seed42排列、全部held query覆盖且各自恰好出现一次。未发现held episode标签进入拟合或shuffle donor。

| H40覆盖口径（每checkpoint） | 有效endpoint数 | 含义 |
| --- | ---: | --- |
| 原48请求screen | 1778 | 包含重复phase对应query |
| 去重44请求，原有效未来 | 1618 | 移除重复query后 |
| 本诊断所有方法共同计分 | 1398 | 每query截到与其shuffle donor共同有效长度，禁止补未来 |

H8/H16共同计分为352/704端点。raw、校正、position-only、hold、匀速、noWAM和shuffle全部使用同一1398端点，不能把本表raw误差与原1778端点screen直接当同口径比较。

## 几何与拟合目标

输入仅实测anchor和已生成world TCP参考/opening：对anchor轴下位移统一乘`g_p∈[0,2]`，对右相对旋转`Log(R_anchor.T @ R_pred)`统一乘`g_r∈[0,1]`，opening加`b∈[-.25,.25]`后clip。旋转增益不改变world位置，区别于旧逐delta旋转衰减造成的平移积分耦合。

对缩放后的预测/真实特征`x,y`，位置与旋转增益分别是`clip((Σw x·y + λ)/(Σw ||x||² + λ))`，identity prior为1；opening bias为`clip(Σw(y_o-x_o)/(1+λ s_o²))`，prior为0。旋转拟合使用anchor-log向量二次surrogate，并非直接优化SO(3) geodesic误差；opening拟合使用pre-clip平方误差，评价时才clip并计算MAE。

独立审计不导入生产`ReferenceGains`、生产几何/指标或`pooled`，使用直接矩阵积分、相对几何和带ridge行的增广最小二乘，再施加固定box约束。18折、54个三参数模型、132条held query/924条query-method结果均重建通过，最大差：参数`4.44e-16`、逐case指标/误差`1.17e-15`、聚合`2.22e-16`。497份输入SHA在读取前后重核一致。代码依据：`scripts/audit_reference_gains.py:36`、`:140`、`:201`、`:218`。

## 校正后的结果

| checkpoint | H8位置 raw→校正 mm | H40位置 raw→校正 mm | H40位置改善外折数 | H40 SO(3) raw→校正 rad | H40 opening MAE raw→校正 |
| --- | --- | --- | ---: | --- | --- |
| 8400 | 25.191→25.037 | 81.742→78.190 | 3/6 | .470659→.328000 | .110965→.116661（退） |
| 8800 | 30.140→28.412 | 81.582→78.759 | 3/6 | .535841→.326084 | .139340→.128647 |
| 15400 | 26.809→25.632 | 84.265→78.685 | 5/6 | .495441→.323847 | .116026→.122495（退） |

H8位置收益仅.154/1.728/1.177mm，H40为3.552/2.824/5.580mm。H40 hold/noWAM位置120.009mm，shuffle约122–125mm，说明正确配对WAM位移仍含信息；但校正后的朝向`.324–.328rad`均差于hold/noWAM的`.318497rad`，且两组opening反退。不能用aggregate位置下降掩盖外折或其它分量回归，也不能据此宣布checkpoint正确、校正器有效闭环或任务稳定成功。

## 原opening MAE聚合bug及保留证据

首份产物把signed `opening_errors`直接平均，导致部分fold/pooled“MAE”为负，**原opening聚合值无效**。生产入口在`eb12049`改为mean-absolute，输出新目录；原artifact及`2d5f066`源码完整归档，未覆盖。该bug不影响三参数拟合、pairings或每case指标。

独立核对原/新18折的参数、fit pairs、全部test-case数据及指标逐值不变，位置/旋转聚合不变；重新从逐端点signed误差取abs计算全部fold/global opening MAE。报告只引用corrected结果，不引用原无效opening聚合。

| 证据 | SHA256 |
| --- | --- |
| 原`reference_gain_diagnostic/diagnostic.json` | `bae0308462b1788592163c57311e449a6b0f788053725195341bc78bdae47685` |
| 原`source_before_mae_fix.py`（逐字等于2d5f066） | `d23818e5feaf947e38e51e7f6f7a768db91dcd66ef06a72c68b40efc4a442b6c` |
| corrected `diagnostic.json` | `903baa946ad26a89bdb500328e3c108d1b661183f6c01963d30b1d5260b5f269` |
| corrected `independent_full_audit_strict.json` | `9befe23722a736c39b9ecea505e2bd702c29c5c8c6969e51e9f0f36e20a685e2` |

证据均位于`outputs/wam_stability_20261021/`对应原/校正目录；`independent_correction_scope_audit.json`记录原/新差异范围核验。独立入口ruff check/format通过；生产侧4+1专项及193通过/1集成未执行的整体QA由主记录另行保留，不将未执行集成记为通过。

本诊断新增模型请求0、物理步0，不读取预留新初态成功表现。它属于已查看开发数据上的内部交叉诊断，既不认证预训练未见，也不认证机器人响应逆或真实任务能力；不因该结果推广校正器或增加物理预算。
