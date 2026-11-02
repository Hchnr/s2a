# 025量化数值补充：真实fit/feature前修复BF16双重舍入

025注册后、任何真实feature提取/拟合/完整生成前，合成CPU实测发现当前Torch的float64.to(bfloat16)内部经过float32，会与原计划的数学单次BF16 ties-to-even不同。例如x=1+2^-8+2^-30，数学最近BF16为1.0078125，当前CPU库转换返回1.0。不能让独立审计按数学单舍入、实际实现按另一个量化规则而冒认一致。

保留025原计划/config、所有预算/数据划分/拟合参数/推广门槛。实现显式FP64 BF16网格的round-to-nearest-ties-to-even，再将已经精确可表示的BF16值转换到base dtype；这样库内部的float32中间转换不会再改变结果。FP64小矩阵残差和原base之和仍只按BF16精度舍入一次，未更改整个原模型precision/TF32配置，zero旁路仍返回原对象。仅BF16实现这一补充，不扩展无关量化格式。

真实Torch CPU测试覆盖正负midpoint两侧、ties、subnormal与signed zero；在任何实际adapter验证前，另用物理GPU6的合成小矩阵检查CPU/GPU输出一致。这类纯数值测试不加载WAM/数据、不拟合参数、不增加denoise/sampling预算，资源和实际结果单列保存。独立checker不调用生产量化函数。先通过数值核验，再冻结025source并准备真实数据；所有024已执行源码和产物保持原字节。
