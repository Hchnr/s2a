# 三个停止门前未调用方向的完整验证

2026-11-02，预登记7907731、实现5b9fcc3。三方向的新完整响应已独立审核，原
搜索未恢复、未修改门或接纳记录。结果证明存在小幅局部下降，但仍无完整witness。

## 固定来源与全部结果

使用93deb627已关闭搜索的stage9/10/11，各唯一alpha1方向，共同anchor response18。
三QP已qualified且原command门通过，预测正下降低于原1e-4，原response_index=None。
重建含安全only更新的模型，确认三stage之间模型逐值不变；独立重建aligned
B/g/A/b和原QP certificate/scaling/recovery，无新增求解、NNLS或拟合。

| 样本 | cap loss | 预测下降 | 实际下降 | 比值 | reserve | 原reserve门 | caps |
| --- | ---: | ---: | ---: | ---: | ---: | --- | ---: |
| anchor | 2.0088178441 | 0 | 0 | — | .903607 | True | 12/24 |
| stage9全步 | 2.0086964892 | 8.355619e-5 | 1.213549e-4 | 1.452375 | .764314 | False | 12/24 |
| stage10全步 | 2.0087702208 | 5.539340e-5 | 4.762325e-5 | .859728 | .886745 | False | 12/24 |
| stage11全步 | 2.0087868716 | 4.087523e-5 | 3.097247e-5 | .757732 | .901292 | True | 12/24 |
| repeat | 2.0088178441 | 0 | 0 | — | .903607 | True | 12/24 |

全部joint/buffer安全、原pose ceiling通过、prefix2、完整witness False。
前两个方向虽下降且实际安全，但内部reserve不足，不能仅放开预测下降门就接纳。
stage11方向满足实际reserve及ratio，预测/实际下降均低于原1e-4，依原搜索合同
仍不合格，未追认被接纳或把它加入原预算。

因此原停止不是物理无可行下降或最优性证书；至少存在新观察的小幅可行下降。
它也不能证明更小进展门会改善完整caps或全轨迹任务效果。当前12/24与prefix2
在多个局部变体中保持，不能按微小loss下降宣称总体完成或默认优胜。

## 实际执行与证据

固定anchor+3trial+repeat，生产5/1500内步/1561查询，独立5/1500/1561，共
10响应/3000内步/3122查询、2model。0新QP/NNLS/模型拟合，3个旧QP仅作证书重建。
0物理/GPU/WAM/fresh/最终留出/实际参考消费，任何候选未下发。不裁剪、换样、
重排或追加旧预算，原.01m/.05rad和实际安全/接纳条件保持。

原1e-10/torque1e-8与布尔审核，完整响应/历史anchor/repeat exact，源码/输入
前后SHA不变。43项jobs32专项全部通过，无skip/collection错误，worker回收。
Ruff/format和Git空白通过，未运行全仓回归。CPU空CUDA/LD、数值库线程1。
tool session17243退出0、实际wait/reap，PID248525确认不存在；600秒TERM/10秒
KILL上限未触发，无重试。

| 产物 | outputs相对路径 | SHA256 |
| --- | --- | --- |
| 完整新结果 | deferred_cap_directions_20261102/result.json | 9e7d2051a578606d8758667ea4344c159f73984fa60b30b016d8470d1eedfdd0 |
| 全部保存分析 | deferred_cap_directions_20261102/saved_analysis.json | 77f3330c4d87dee62f3324d53090c07c5e04f8e6c90e4ba1ea7b78b07a55e4d9 |
| owner终态 | deferred_cap_directions_owner_20261102/observation.json | a85d7b79d57da03b6a9178ceb7e51971f03388cbb9768d6a87c1ba9bf7fcef8b |
| 43项QA | qa/deferred_cap_directions_20261102/result.json | 5e91af0d4bf80306ab5be87982cf3c02e8fc1dc197d5ec44800351ea8f8ba094 |

下一步若比较更小进展门，必须作为新算法配置显式登记、继承同一root域和最大
stage/response预算，保持原实际安全/reserve/完整24caps与最终稳定验收，不改旧
默认或失败结果。需实际检查这些小步能否累计改善完整跟踪，而非仅提高接纳数。
原18/12/24及稳定20/24、每task4/8仍未达成，最终留出关闭，GPU继续禁用。
