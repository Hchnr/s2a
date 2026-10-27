# 089 固定成对差分监督的条件化arm前向模型

088唯一fit及独立SVD/实际退出资格完成；6holdout四类预测24/24通过，但086局部响应17/24，offline资格失败，不部署。保留071条件/288设计/12输出、原24fit原生数据、原名义PD残差定义及ridge0.001，不因结果挑新特征或扫描正则化。

新增的唯一干预是训练损失中的成对响应约束。以PCG64 seed42对每个task两个anchor各取一个训练anchor，抽样[0,1,1]，固定训练groups[0,3,5]、未训练groups[1,2,4]。划分在新fit前固定，与088失败列表无关；三个未训练组包含先前失败组，不能挑选已通过组作验证。这些是已诊断的内部验证，不是最终未见新初态。

原生4865行仍每轨迹等权；仅加入训练anchor中三个translation轴各一条±中心差分方程，共9条，不重复拟合native前缀、baseline或将36探针当作36条独立轨迹。条件用各anchor本次已实测初态/已知action goal固定，只有translation applied goal按真实±命令改变。特征差X_plus−X_minus与标签差Y_plus−Y_minus均除以实际归一化命令差（裁剪后），nominal项也取同样差分；SO3训练标签为相同current SITE切空间中各末端world Log，再形成差分，最终响应验证仍用Exp后端点的群差。

原生行总权重0.5，各轨迹0.5/24；9个差分方程总权重0.5，各task/axis等权0.5/9。固定此权重，不扫描。中心/尺度沿用088已独立核验且只依赖原24fit的统计，复制到新参数文件，不修改旧参数或吸收heldout/probe统计；原288×12结构和只有全局bias不惩罚保持。唯一新参数fit1次，独立数值复算另计，来源和方程全部指纹保存。

验证保持6原生holdout×4预测门槛；六个局部响应组全部报告并要求仍24/24不劣nominal，但明确其中训练12项/未训练12项，不能将训练anchor计作独立泛化证据。独立数值、原生holdout24项、未训练anchor12项及训练约束12项都通过，也仅授予离线forward资格；新逆控制、18/18正确参考、WAM开发和最终24条新初态仍需另行执行，不缩减原目标。0新物理/世界模型/fresh。

当前仅登记与显式配置，差分数据组装、独立数值/验证入口及新fit尚未执行。

实现补充：arm_contrast_training.py与独立arm_contrast_independent.py分别组装所选9条方程，真实源差分、特征与nominal项对照通过，未读取未训练anchor标签参与方程。fit/verify/validate/run/close_contrast_conditioned_arm_forward.py已实现4874行混合方程、固定旧统计、训练/未训练分组报告及实际wait/reap监督。尚无新fit；全部准入后唯一CPU owner依次拟合/数值复核/验证，不自动重试，任何性能False保留。
