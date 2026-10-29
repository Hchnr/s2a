# 071 电机初态保持与位姿耦合预览：完整审查通过，性能失败

固定六诊断参考1407步，统一当前motor target保持与068六步/四保护，其余OSC/position_hold/参考不变；0模型/fit/fresh。它是单项证据后的组合回归，不是未见泛化或WAM任务验收。原6/6及10mm/.05rad资格不变。

十四项实际缓存广播/组合控制复用/耦合函数测试、Ruff、六条真实Python3.11 dry通过；持久化owner3474407/birth582953516、首client3474435/birth582953548实际启动，来源与预算冻结，CPU/OSMesa空CUDA/LD，无重试。

独立组合checker已实现，从实际初始cache重建目标，从生成窗口/当前实测TCP重算四保护，不调用producer选择函数；AST测试确认除已声明初始化准入和runtime身份外，原独立机械检查保持，实际测试通过。

case00 bowl完整164步实际完整独立核验通过，stable=True、成功86/尾79。case01 wine/demo_1完整254步独立通过，stable=True、成功92/尾67；该wine在070无预览时失败，组合在本诊断例有收益，但不能由单例代表通用组合。原owner与case02 cream客户端3475654按/proc同birth实际存活，固定剩余四条继续。

尚无完整组合资格，更未推进WAM或fresh；旧单项失败保持原记录。

## 组合在wine/demo_39发生退化：原6/6门槛失败

case02 cream完整183步实际独立控制/组合状态/标签/图像/normal收据/source核验通过，成功及尾85、稳定True。case03 wine/demo_39完整244步同一审查通过，但ever/terminal/stable均False、0成功；对应068开度对齐＋位姿耦合曾成功90步稳定，而070初态保持单项也0成功。组合保留了该初态保持路径的失败，不能把069和068各自成功简单合并。

当前4/6独立核验，3条稳定，原全6通过资格失败，禁止进一步扩大组合或接入WAM。原owner3474407与case04客户端3476610按/proc同birth实际存活，剩余两条按固定范围收尾，无参数或初态策略切换。

## 完整六条结论

case04 wine/demo_46完整334步独立通过，成功100/尾88；case05 wine/demo_11完整228步独立通过，成功82/尾78，两者稳定。全六条1407步及完整控制/组合/标签/图像/normal收据/source审查通过，源闭包无漂移，原owner和所有client已退出。regression_summary.json记录实际5/6与performance_qualified=False。

唯一失败wine/demo_39，原6/6不满足，不接入WAM；也不按demo选择初始化或预览规则。后续考虑显式当前开度/运动历史/真实motor cache与已生成开度轨迹的轻量逆模型，把实际电机状态和参考状态区分开，先独立离线资格再固定物理对照，不能以动作拟合误差代替前向state/任务验收。
