# 150 在线响应历史合同实现

新增src/s2a/runtime_response_history.py，按实际执行后的测量端点更新12维innovation，与090/116训练with_history使用相同世界坐标/左旋转残差及dt-scaled twist。候选attach不更新历史并返回副本；下一次当前pose/twist必须匹配上次完成端点，否则拒绝，防止不同分支或轨迹混用。reset清空历史，首次innovation为零。

实际CPU测试核对完整四端点前缀的每个在线innovation与训练实现逐值一致至1e-14，覆盖非交换旋转、返回值副本、预测不更新、错配端点拒绝和reset。连同149逆解测试共2项通过，静态/格式检查通过。当前仅历史状态组件，实际sim数据读取、旋转/夹爪策略绑定和18条控制回归尚未执行；无新物理/fit/世界模型/fresh，原总目标未完成。
