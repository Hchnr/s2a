# 208–212 直接逆数据、拟合和运行命令验证

本阶段尚未执行直接逆控制，不能据动作误差推断任务成功。固定模型以当前实测状态、因果历史及明确期望SITE姿态预测平移命令；剔除教师arm command及其派生goal。未来真实SITE仅用于离线构造oracle期望参考，不作当前反馈。

数据：5369训练行（4865自然+72训练锚点×7绝对分支），1315留出行；输入89维，218固定特征，seed42，ridge=.001且常数不罚，自然/分支质量各.5。均值/标准差仅来自4865自然fit行。209独立重建全部输入、标签、物理解析基线、分支引用与特征通过。合格数据为`outputs/direct_arm_inverse_data_checked_20261026`。

首次producer因未使用局部变量未通过Ruff，原shell却继续产生数据；该批不作为合格来源。保留`outputs/direct_arm_inverse_data_20261026/producer_source.py`及`source_archive_record.json`记录原路径和SHA。修正producer另存`prepare_direct_arm_inverse_data_checked.py`，两批NPZ数组逐项bitwise一致，独立验证以修正批为准。没有改写失败记录或历史指纹。

210唯一拟合已经执行：Cholesky与独立标量特征/增广SVD系数最大差3.2371e-13，梯度最大1.3100e-15，秩218。fit/verify均实际wait0，来源复核完成。参数SHA256为`b6eccee36c9918aa93586a012a46ce44612aec5276cd01831e8052770e98c8cd`，证据位于`outputs/direct_arm_inverse_fit_20261026/{execution_result,independent_verified,parameters}.json`。

| 留出native_index | 行数 | 直接逆动作RMSE | 解析动作RMSE |
| --- | ---: | ---: | ---: |
| 8 | 170 | .05973943 | .17420933 |
| 9 | 170 | .04773319 | .19617287 |
| 18 | 244 | .05587529 | .13313546 |
| 19 | 334 | .08073240 | .19318170 |
| 28 | 215 | .07242937 | .16219152 |
| 29 | 182 | .05650399 | .19083537 |

RMSE为归一化平移三向量误差范数的均方根，尚未施加运行变化率限制，仅为历史动作诊断。六条均降低不能说明修改后的命令可在接触中持续跟踪。

211运行层遵守实际接口界限，固定20Hz每步平移变化<=.25，以输入当前实际上一条command为基准；首次历史不可用则要求显式零重置。返回最终命令和实际仿射解码的physical_delta。没有混入旧前向逆或预测5mm限速。它不自行更新未发生的物理状态。

212独立回放覆盖原30条轨迹6180记录状态，检查独立输入、解析逆、218特征、系数乘积、最终界限/变化率裁剪及physical_delta，全部通过；最终command最大差3.3307e-16，56行受变化率限制。核对previous command与实际记录上一条相同，首次availability正确。证据`outputs/direct_arm_inverse_runtime_20261026/independent_verified.json`明确physics_steps=0、control_qualified=false。

验证命令：空CUDA/LD下执行三个direct inverse测试文件共7项通过；新增运行层、测试及独立checker的Ruff检查与格式检查通过。未将记录反馈回放称为反事实闭环。

后续必须在独立新namespace接入真实执行，固定083/159旋转、079夹爪、实际命令173 SOURCE commit、150完成后SITE创新，并为直接预测等式编写独立日志核对。原18/4907实际稳定及状态误差门槛保持，WAMdev和24新初态仍未获准进入，总目标尚未完成。
