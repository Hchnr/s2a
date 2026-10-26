# 105 干预训练条件覆盖：只读结果

核对104实际fit_sources/contrast_sources和冻结参数/缓存，对原24个干预训练anchor（086三点+097二十一点）与086六个开发anchor计算固定89维标准化条件距离、6维wrench距离及各轴范围。actual训练身份与关闭来源SHA补充审查通过，结果和指纹在outputs/intervention_condition_coverage_20261025/result.json及input_verified.json。

重要区别：原六anchor的自然状态均在4865原生fit行中，直接按fit_sources偏移重建并匹配；group4对应原生行3466。因此“未训练anchor”仅表示该anchor的受控probe标签未参与辨识，不是完全未见的state/trajectory，更不能称为留出新初态成功。

| 开发group | 干预训练 | 最近89维条件RMS距离 | 最近6维wrench RMS距离 | 超出全部训练wrench范围的维度 |
| --- | --- | --- | --- | --- |
| 0 | 是 | 0 | 0 | 无 |
| 1 | 否 | .7927 | .5638 | 无 |
| 2 | 否 | .2995 | .0469 | 无 |
| 3 | 是 | 0 | 0 | 无 |
| 4 | 否 | 1.3913 | 2.8033 | Fy、Tz |
| 5 | 是 | 0 | 0 | 无 |

group4当前world-at-control-SITE wrench为[26.9184,24.8738,-80.3393,5.37794,-.914295,1.01513]（N/Nm）。干预训练Fy上界15.4664N、Tz上界.670793Nm，均低于该点。最大训练force范数108.2904N反而高于group4的88.3047N，故不是单纯“最大力不足”，而是方向/moment组合支持不足。最近wrench训练点是097 group18/task2/native25的first_close42；最近全条件点来自task0，不能解释成同任务覆盖。

坐标范围只是支持的必要条件，区间内不等于凸包内或足够泛化；距离不用于设置新的通过阈值。本诊断不证明唯一根因，不据此调模型或放宽24/24门槛。物理/模型/fresh/参数fit均0，没有控制资格。

下一步可在不读取开发/holdout标签的前提下，用原fit轨迹当前wrench方向与moment的固定覆盖准则选择独立追加干预；必须排除旧开发profile，保留最终新初态验收。不能仅挑更大的力范数，不能把自然轨迹上的同state观测当作已辨识全部action导数。
