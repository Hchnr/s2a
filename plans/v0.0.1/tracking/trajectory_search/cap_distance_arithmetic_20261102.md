# Cap-distance实现算术补充（2026-11-02，Asia/Shanghai）

原`cap_distance_gn_design_20261029.md`的数值前置已由独立的新初始化/72探针及
同预测器sum-q基线关闭。sum-q完整结果SHA256：
`0bcc9962caabb8f74f92113a8994cfdb41e7e9a1f173781c4d20d603fbcb9ecc`。
该基线无完整witness。本文只固定实现算术与软件验证，不启动新研究运行。

## 直接sqrt差的浮点反例

原三维平移误差[.01,1e-10,0]按原cap代数得到q=1.0000000000000002，caps失败，
但float64 sqrt(q)舍入为1，直接max(sqrt(q)-1,0)为0。不能声称该直接表达式的
浮点零集与caps完全一致。该反例仅NumPy运算，0模型/查询/物理。

在任何新producer前固定以下唯一float64求值，不按运行结果切换表达式：

```text
q = 原_errors平方和 / 原_cap平方（保持原计算顺序）
rho = sqrt(q)
excess = 0                              if q <= 1
excess = (q - 1) / (rho + 1)            if q > 1
Lcap = float64(.5) * sum(excess**2, dtype=float64)
```

这是原平方球外距离的有理化求值，实数目标不变；明确替换原设计的直接sqrt差
浮点表达式，而非声称二者逐位等价。非负有限q的最小超限1ulp不会平方下溢，
因此浮点零集与原全部caps保持一致；溢出/非有限输入拒绝，不授权执行。
报告原直接表达式与固定表达式的数值差异，不用两者切换满足接纳。

## 投影与导数

z仍为原signed TCP误差/.01或.05，24组三维；rho按同组原q取sqrt。
q<=1时r=0、D=0；q>1时以同一excess计算beta=excess/rho，
r=beta*z，D=beta*I+outer(z,z)/rho**3，Jr=D@rawJe。
这也是原投影在实数下的同一表达式，边界仍选零广义导数。独立audit另行重建
并固定相同算术合同，不能导入生产投影函数充当独立证据。

QP B=I+Jr.T@Jr、g=Jr.T@r，damping=1；预测下降固定
`-(g@d)-float64(.5)*((Jr@d)@(Jr@d))`，不含damping。
actual使用上述固定Lcap，pose最小预测/实际下降按原fraction乘该Lcap定单位。
restore的merit、门和模式切换完全保持；rawJe必须继续保留并按原因果更新。

## 实施分块与验证门

新增独立merit模块和可显式选择objective的共享驱动，不改旧默认内核或历史记录，
不做全局变量/导入替换。新驱动的sum-q模式先以闭合基线的保存QP和响应做逐项
等价回放，不运行新QP/NNLS/模型，不把该软件回放计成新实验或成功证据。
独立cap audit和真正执行配置须另行实现冻结，测试通过不自动启动实验。

必要检查保持原设计：球内/外/边界、完整径向/切向导数、近caps、严格浮点零集、
因果零结构、rawJe球内再激活、原restore不变、预测不含damping、最后repeat、
不安全候选拒绝和预算。GPU继续禁用，所有结果仍以原state error和任务门验收。
