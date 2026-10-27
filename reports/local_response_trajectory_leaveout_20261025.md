# 111 固定局部响应训练轨迹留一

原45训练anchor、24fit轨迹的当前89维条件及固定统计，使用一次登记核权重(1+d²)^-2。每次完整删除相同原始轨迹的所有anchor；097/106同轨迹双anchor同时排除，剩43或44点。没有读取holdout或086未用于训练的anchor标签；未调带宽或邻居数。

45点中，位置37、旋转向量40、线速度36、角速度38点的训练留一响应矩阵误差不高于nominal。仍有8/5/9/7点分别退化，不能因平均或多数改善宣称全面合格，也不能直接替代原开发局部门槛。旋转向量训练差并非有限SO(3)开发响应矩阵，结果不是闭环或fresh证据。

输出：outputs/local_response_trajectory_leaveout_20261025/result.json。入口：scripts/check_local_response_trajectory_leaveout.py。未新增参数fit/物理/世界模型/fresh。局部估计并未自动消除泛化问题，不据此进入控制或选择核超参数；原109资格失败和最终目标未完成保持。
