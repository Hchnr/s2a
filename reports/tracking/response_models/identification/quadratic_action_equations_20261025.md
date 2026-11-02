# 138 二次action训练方程准备完成

原4865自然、198差分和新增198实际比例插值约束，共5261×1974设计已生成。每个baseline/正/负分支独立重建goal及物理12响应，当前SITE/twist一致，逐分支二次设计与独立多项式计算核对；原1128差分设计、target、nominal复现116，插值约束的仿射部分消去至1e-10。自然权重总.5保持，396干预等分总.5，旧单条权重改变已登记。

输出outputs/quadratic_action_equations_20261025/training.npz、parameters.json及result.json保存来源SHA。scripts/prepare_quadratic_action_equations.py实际执行、静态/格式及来源复查通过。未读取开发/holdout标签，无fit/物理/世界模型/fresh。下一步唯一fit、独立SVD及原24holdout/24局部响应门槛，原控制/WAM/fresh关闭和总目标保持。
