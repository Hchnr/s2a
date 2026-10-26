# 150 在线因果响应历史合同

为149实际控制接入建立仅已完成周期的12维innovation历史，与090/116训练with_history定义逐值一致：实测motion减本周期nominal、Log(真实世界旋转增量*nominal旋转逆)、dt*末twist减nominal scaled_twist。当前计算/命令候选只读同一历史；实际执行后才observe，记录真实末端作为下次输入身份核验，禁止分支或轨迹错配。首次历史全零，每轨迹独立实例；reset显式清空。wrench只允许外部当前读数，不能由未来端点计算。

先可复用状态实现及完整前缀训练parity/候选只读/独立reset测试，后接sim当前测量。未执行控制或取得18/18资格，原门槛保持。
