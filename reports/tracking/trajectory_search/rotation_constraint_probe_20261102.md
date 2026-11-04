# 末端旋转单方向诊断（2026-11-02，Asia/Shanghai）

## 结论

单独优化第12点旋转的一个局部QP能给出旋转改善方向，但其完整数值响应违反
关节制动安全余量，整体cap loss也变差。方向没有派发到实际控制，未获得完整
witness或任务资格。不能由QP证书通过或单组误差下降推出可靠跟踪。

更具体的失配是：同一方向下，row4358（0起period10、joint4，即第11点关节5的
braking_upper）线性预测raw slack=+4.9953333031893445e-6，完整响应为
-8.711233222280512e-5，误差-9.210766552599447e-5。这不是先前1e-15级舍入问题。
需要验证当前Js的局部斜率和有限步曲率，而不是直接放宽安全门或global盒。

## 输入、保持项与成本

预登记8488954，实现1680c3a。固定cap stage58/anchor85，原actual368/367因果
历史、原H12参考、同一replay-order预测器和native_solve TCP。raw Js/Jq/Je从
新stencil和已关闭cap账本重建，并先核对原stage58完整B和证书。

新QP仅使用组23的完整投影残差/Jr构造B/g；所有hard rows与原局部问题逐值
相同，radius=.005、原global .1、command/rate/关节/制动/terminal与原caps保持。
alpha固定1，没有缩线、回退重求解或其他组加权。全部原命令预检通过。

实际1QP、3迭代、1NNLS，原1e-6证书通过；生产anchor/trial/repeat共3响应，独立
端逐条复算3响应，各900内步/937查询；总6响应/1800内步/1874查询/2model，
恰在登记上限内。0新FD、0物理/GPU/WAM/fresh/最终留出，不追加旧预算。

初始与最终重复exact，全部独立响应逐值一致，原numeric1e-10/torque1e-8及
精确布尔审核通过，源码/输入SHA前后不变。主命令由timeout限600秒、超时后
10秒强制回收；实际退出0，由执行工具wait/reap确认，未超时、无重试。
这不是独立物理或真机结果，所有后续“响应”都指局部机器人完整数值递推。

## 完整指标

| 指标 | anchor85重新执行 | 唯一trial |
| --- | ---: | ---: |
| 第12点旋转q | 6.8776184452 | 6.5736810130 |
| 第12点旋转误差rad | 0.1311260695 | 0.1281959537 |
| 全窗口最大q | 6.8776184452 | 6.5736810130 |
| 全窗口cap loss | 2.0100673083 | 2.0501271104 |
| 全窗口sum q | 30.1138240652 | 30.7328447646 |
| 达标caps / prefix | 12/24 / 2 | 13/24 / 2 |
| 完整joint audit | True | False |
| 最小raw safety slack | +9.9806666065e-6 | -8.7112332223e-5 |
| 完整执行资格 | False | False |

旋转单组预测下降=.0956692780125，完整响应下降=.0933663983076，ratio=.9759287438，
独立重算相同。该方向的旋转模型预测较接近，但安全模型不够；不能据此证明
完整Je准确，也不能把安全失配唯一归于陈旧导数（曲率尚未隔离）。最大安全slack
预测误差为.000437662673244；后验分解未新增模型或求解。

## 检查与来源

75项专项使用CPU runner、jobs32，全部通过，无skip/collection错误，worker实际
回收、来源前后不变。覆盖目标组选择、hard rows不变、预检不裁剪/拒绝重复和
超步、固定配置、cap投影/独立账本/原dual-recovery。Ruff/format和Git空白检查
通过；未运行全仓回归。执行前校验当前全源码内容与QA来源一致，正常提交只
改变HEAD，不绕过源码匹配。

| 内容 | 当前仓库相对路径 | SHA256 |
| --- | --- | --- |
| 完整结果 | outputs/rotation_constraint_probe_20261102/result.json | 7d5bd9502573cf122208a9c146c3baac8aaa62991f03688716fd58841238bdc8 |
| 工具终态观察 | outputs/rotation_constraint_probe_owner_20261102/observation.json | ba1e47e157cd34edf9c75481fefb9e9b68b2239443d671c37fe798997dc438e7 |
| 保存结果分析 | outputs/rotation_constraint_probe_20261102/saved_analysis.json | c1c1d778861c10bff69490f4382c6868dc32882d009c31389b234ff975cb800a |
| 最终75项专项 | outputs/qa/rotation_constraint_probe_final_20261102/result.json | 5d9a6650cd07fbaf6ae0a68070d80622a42d0653eb176a75c355530e178cccb4 |

registration.json保存新B/g/A/b、QA和输入来源；entries.json/native_metrics.json
保存全部生产/独立响应指标；source_before/after保存源码内容指纹。launch.json与
工具观察分开，保留启动时尚未wait与最终已wait的不同状态，不覆盖起始记录。

## 下一步

先单独登记沿这条已知方向的缩步与小扰动诊断，在原边界内核对安全slack随步长
变化：小扰动斜率不符支持Js局部失配，斜率相符而大步失配支持有限步曲率。
当前数据不能区分两者。没有新证据前不加QP重试、不改安全容差/裕量、不声称
不可达，也不把单旋转目标改写成完整控制成功。原18/12/24与最终任务门保持。
