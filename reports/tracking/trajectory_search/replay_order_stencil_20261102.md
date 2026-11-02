# 新预测器初始化与72探针结果（2026-11-02，Asia/Shanghai）

## 结论

新replay-order预测器重新执行base、重复base和72个原signed1e-4探针，独立checker
逐条完整复算。两侧74响应、评分及Js/Jq/Je全部精确同值，严格因果零结构通过；
两侧重复base也同值。新初始化和探针的此项数值契约通过，不能继承或改写旧
3557/72探针的exact身份，更不能据此授予实际控制或任务成功资格。

没有执行新搜索、QP/NNLS、物理、GPU、WAM、新场景或最终留出。当前新初始化
最大q=6.908710040755056，24项caps中仅18项达标，尚无完整H12 witness；
当前实际资格仍cursor8/task0/grasp0。下一步是单独登记同预测器sum-q对照，之后
才能评估cap-distance目标的独立贡献。

## 合同与成本

- 原初态368、参考轨迹、命令字节/方向/步长、控制器和25×2ms递推保持。
- 使用单列的新响应变体，不替换旧默认或独立checker。评分用原
  evaluate_repair_entry，TCP变换按独立checker相同的np.linalg.solve顺序构造；
  与历史source的变换最大差1.3322676295501878e-15。旧原始变换不覆盖。
- 32个spawn批次，独占outputs/tmp、数值库单线程、空CUDA/LD。实际32个不同PID
  各完成一个批次；全部Future取结果，全部进程join/reap且exitcode0。
- 生产端74响应/22200内步/23120查询/32model；独立端74响应/22200内步/
  23089查询/1model；总148响应/44400内步/46209查询/33model。每个生产模型
  含一次初态核验查询，checker含一次初态查询。没有隐藏重试或追加旧预算。
- 先于模型构造写入registration。源码/输入SHA前后验证通过；主进程退出0，
  工具已实际回收。登记后的运行耗时120.589秒，不含前置来源加载时间。

## 独立审核

逐条比较完整rollout、安全slack、TCP误差、原24项归一化平方误差、联合安全
可行性、prefix/caps和pose loss；布尔精确，原numeric1e-10、torque1e-8保留。
本次所有字段差0，不能只以宽容差通过描述成exact。

两侧分别从新响应构建：Js 4369×72、Jq 24×72、Je 72×72。分母保持原signed
nominal1e-4，最大实际位移舍入差1.1018204577883939e-17，已另存实际位移。
所有未来命令对较早输出的导数原样精确为0；没有截零、修补或改容差。

保存数据对历史记录的重建差异（不是通过门）：Js最大差4.291635970210539e-11，
Jq为3.2862601528904634e-9，Je为1.1102230246251565e-9。历史矩阵不同值，说明
必须使用本次新矩阵进行后续同预测器比较，不能宣称原矩阵直接获得新资格。
初始化24项q与历史最大差5.5067062021407764e-14。

## 软件验证与证据

23项专项通过，无skip/collection错误；统一runner指定jobs32，全部worker实际
回收、源码前后不变。覆盖signed nominal导数、compact原命令记录预检、future
泄漏拒绝、pose代数不一致、结构比较、原容差隔离及布局/因果响应合同。
Ruff/format及Git空白检查通过；未执行全仓回归。源代码提交4d6816a。

| 证据 | 当前仓库相对路径 | SHA256 |
| --- | --- | --- |
| 执行前登记 | outputs/replay_order_stencil_20261102/registration.json | 728c343351a93ba0afdb8a5541f68258a7e5a23eff0fedeae47ea2ca247162e6 |
| 完整结果 | outputs/replay_order_stencil_20261102/result.json | 80e1fd703fe67a8d44ad90d5b603e86e1af7d9fba50084dc6009ea71157792c5 |
| 新初始化、探针及矩阵 | outputs/replay_order_stencil_20261102/stencil.json | 0d447082a50a2a8425c438a0a7cfe8a4a9742cb53bba445e015e49db6f6ecc3c |
| 历史差异/实际位移分析 | outputs/replay_order_stencil_20261102/saved_analysis.json | c98e3945faa310b90d43d9fcf4bb3c80a9ac0d4c8c2fd39782c17c58cb4e3b22 |
| 最终专项 | outputs/qa/replay_order_stencil_final_20261102/result.json | 4f4497f50c9e6ee3b41e5048365fe45621d5c533fed54bfc40d299dc65dec413 |

完整结果约256MB；按字段读取，不把日志体全文打印。每个批次保存完整生产entry，
independent子目录保存完整独立响应/指标。保存分析只有JSON/NumPy处理，没有
新增模型/查询/拟合或物理执行，不是另一轮独立审核。

配置和预算：`configs/tracking/trajectory_search/replay_order_stencil_20261102.json`；
预登记：`plans/v0.0.1/tracking/trajectory_search/replay_order_stencil_20261102.md`；
入口：`scripts/tracking/trajectory_search/check_replay_order_stencil.py`。
前置布局问题与四命令对照见`native_replay_layout_contract_20261102.md`。
