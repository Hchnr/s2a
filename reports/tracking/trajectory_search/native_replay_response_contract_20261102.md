# 完整响应算术对照（2026-11-02，Asia/Shanghai）

## 结果

独立新增 `native_replay_joint_response.py` 和 `replay_order_trajectory.py`，保留旧默认。
固定初态368、原控制器和 [0,0,3,4] 保存命令，两侧完整执行8条H12响应。
原 numeric1e-10 / torque1e-8 比较均通过，布尔指标一致，两侧各自的初始重复响应
数值一致；**两侧响应并非完全同值**，不能授予 exact 或完整搜索资格。

相对旧预测器，只改变 OSC 算术、隐式阻尼乘法分组和裁剪表达式。因果目标/latch、
自主递推、25×2ms、控制接口及 checker 内核保持。没有执行搜索、QP/NNLS、
物理步骤、GPU、WAM或新场景；整体控制资格与完整H12 witness仍未获得。

## 实际执行与成本

1. 首轮比较器未处理 `(slack_array, tolerance)` 元组，两侧各1响应后异常退出；
   complete/passed=False，300+300内步、313+313查询、2model，来源前后不变。
   原失败保留，未覆盖或追认。
2. v2 对比较器增加 tuple 遍历；先对嵌套数组、布尔和故意超差做结构预检，再按
   单独登记完整执行。进程正常退出0并已回收；complete/passed=True。
   两侧各4响应、1200内步、1249查询；合计8响应、2400内步、2498查询、2model。
3. 两轮实际合计10响应、3000内步、3124查询、4model，单列新诊断成本，不追加
   已关闭 search 的122响应预算。两个运行的来源复查均通过。

证据（路径均相对当前仓库）：

| 内容 | 路径 | SHA256 |
| --- | --- | --- |
| 首轮失败 | outputs/native_replay_response_contract_20261102/result.json | 36ba28e6311f1fe7899a96e729736e181794a234c0706aefe59c8acc3f0d39c7 |
| v2完整比较 | outputs/native_replay_response_contract_v2_20261102/result.json | 13e9fd15265a6929f0c131a5e09eef389320147b9656b2face7fc3ace100d360 |
| 保存结果分解 | outputs/native_replay_response_contract_v2_20261102/saved_analysis.json | f921142fb963e79783c563b6f14f9fcbfc0dd8c1ca901fd2d69e6b28183a1caa |
| 专项测试v2 | outputs/qa/native_replay_response_v2_20261102/result.json | 529cc2006f76b7331a6d284def7a16c9ddf02b6b0b06b656958cb897a158edaf |

对应登记和执行脚本在各自 outputs 目录。固定科学问题见
`plans/v0.0.1/tracking/trajectory_search/native_replay_response_contract_20261102.md`，
失败修复与额外成本见同目录 `native_replay_response_contract_v2_20261102.md`。

## 严格 ratio 与剩余差异

最大力矩差9.485745522397337e-13；最大指标差1.3677947663381929e-13。
首次失败处的 base braking row4346 现在完全同值，但 trial terminal row4368
仍不同。使用原预测下降分母0.24829074036733323，保存命令条件下的 ratio 差
从原 -2.137171550842254e-10 缩小至 -5.1815329804583143e-11，actual 差
-1.2865264409356314e-11，均在原1e-10内。这只检查一个固定命令对；不是重新
执行搜索、不验证其余决策、不继承原初始化/探针资格。

response0 第一个 goal 完全同值，首步力矩已相差7.105427357601002e-15；
首步速度相同，第二步速度开始不同。因此不能把剩余差异仅归于长轨迹积累。

进一步零查询检查保持保存输入数值不变，仅改变 Jacobian 的 C/F 存储顺序：
C 顺序与外部独立函数逐字节一致，F 顺序相差7.105427357601002e-15。
记录：`outputs/native_osc_layout_diagnostic_20261102/result.json`。
这证明现有129例C布局单点通过不足以覆盖布局影响，尚未证明 F 布局是全部轨迹
差异的唯一来源。下一步应对照独立函数的 dot/@ 路径并增加布局回归，再执行
单独登记的响应核验；不放宽容差。

## 软件验证边界

五个专项测试以统一 CPU runner、jobs32执行，全部通过，无skip/collection错误，
所有worker回收、源码前后不变；Ruff/format及Git空白检查通过，未跑全仓回归。
首轮专项为4通过1失败：测试预期把归一化0.1当成精确物理位移，忽略固定 affine
decode 的浮点结果。修正测试预期使用实际固定接口解码，接口与容差均未改。
原失败记录：`outputs/qa/native_replay_response_20261102/result.json`，SHA256
`1ce3176d1b482ffb89dfc9249576a2c8e9019bab5d32b6244381939843a42673`。

源码提交16ba60b。后续只有本报告、v2计划和状态文档更新，不扩大已有测试结论。
