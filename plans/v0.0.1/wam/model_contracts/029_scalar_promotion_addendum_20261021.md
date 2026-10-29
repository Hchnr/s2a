# 029 数值补充：CPU 标量提升与 CUDA 控制分开核验

在029实际scheduler纯合成20场景、GPU或WAM运行前登记；原029 plan/config原字节不改。此补充只明确原第1步的后端数值gold及来源闭包，不改变样本、节点、20场景/40实际conversion、1824denoise/1模型、0fit/物理/fresh预算，也不调整任何原028质量门槛。

源码审查及CPU单位原语发现，不能仅从结果dtype=BF16推导乘法内部计算顺序。指定模型Python/Torch2.10.0+cu130的CPU单位测试中，FP32零维sigma乘BF16 velocity先将sigma量化为BF16。六个已登记payload原语（3 selected sigma×2 payload）按该次序均exact；按原FP32 sigma直接乘再BF16量化，在.80779493 normal场景max差.015625、.50817281 normal场景.0078125。证据`outputs/wam_stability_20261021/free_trajectory_arithmetic_unit_observer/model_python_cpu_scalar.json` SHA `20622da8808d62e8d45db32993dcf138593f34b8d97404bbf3c5d9db758eb018`。这只执行CPU基础算子，未调用实际scheduler、GPU、WAM、fit、EP或PNG；不是原模型质量失败。

指定Torch git449b1768410104d3ed79d3bcfe4ba1d65c7f22c0的已安装`ATen/native/cuda/Loops.cuh`中，opmath标量路径对CPU scalar调用`iter.scalar_value<opmath_arg_t>`；`TensorIterator.h`按scalar原dtype读值，`OpMathType.h`将BF16的opmath设为float。由此提出CUDA原mul路径保留FP32 sigma的可检验预期；安装头文件没有完整证明本轮编译kernel实际采用该模板，须以原20场景的真实CUDA调用核验，不冒称已完成GPU验证。原scheduler的sigma表在CPU，CUDA只移动timesteps；两后端的scalar device都应记录CPU。

独立NumPy预定gold：

- CPU BF16 velocity：先RNE_BF16(sigma)，与已BF16的velocity作FP32乘法，再RNE_BF16结果，最后FP32 sample减乘积、FP32舍入。
- CUDA BF16 velocity：原FP32 sigma与已BF16 velocity作FP32乘法、FP32舍入，再RNE_BF16结果，最后FP32 sample减乘积、FP32舍入。
- 两后端FP32 velocity控制：FP32乘法/FP32减法。FP32运算先由精确扩展的FP64乘积计算后显式转FP32；BF16舍入使用独立bit规则，保留负零/次正规。

20场景逐后端保存原`convert_model_output`内部真实乘积及返回x0的dtype/bits；实际sigma、输入velocity、输入不变和原method一次委托照旧核验。分别与上述预定gold逐位相同才获得运行资格；任何不符保留False、停止，不看结果后切换gold或重试。CPU/CUDA相互的bit相同与不同数量如实报告，**跨后端相同不作资格条件**。诊断实际WAM在CUDA执行，后续保存的原converted_action及027离线反事实使用通过资格的CUDA数值次序；FP64理想式始终另列。

这解释并替代初版WIP probe将CPU/CUDA必须相同作为额外断言的做法；该断言尚未用于任何真实scheduler/GPU执行，也未产生过被改写的原False结果。原029文字要求对完整128 bits核验，本补充明确其为各后端对各自gold的核验。CPU和CUDA两条路径可能出现已说明的差异，不以此宣称模型、权重或原UniPC有错误。原027 FP64 readout到BF16的数学单次舍入合同不变。

新完整来源为原登记109源再加本补充，共110源/110快照；11个新实现/测试和原plan/config保持同名，原96源不改。另将本CPU单位证据、指定Torch version.py和三个已安装头文件的SHA作为实际只读输入固定，不将其冒称整个编译runtime的完整闭包。真实执行前完成110源冻结、CPU检查及独立prepare；外部Cosmos和共享环境继续只读。
