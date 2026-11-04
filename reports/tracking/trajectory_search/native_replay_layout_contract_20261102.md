# 存储布局与完整响应契约（2026-11-02，Asia/Shanghai）

## 结论

v4固定四条H12命令 [0,0,3,4] 的完整响应与独立回放全部同值，最大响应及指标差
均为0，两侧各自重复响应一致。首次失败处 base/trial 的raw_slack、reserve、
actual及条件性ratio差也为0。新预测器的这组算术前置核查通过。

未执行新搜索、初始化/72有限差分探针重新验证、物理跟踪或任务验收。旧122响应
search及其严格checker失败仍保留；不能把当前四条命令推广成全部搜索资格。
完整H12控制witness仍未获得，GPU继续禁用。

## 定位与改动

- f57d708：惯量/零空间保持外部control_utils的np.dot；任务force、moment及拼接
  投影采用独立checker原有的@。C/F/strided、满秩/降秩共96输入逐字节对照通过。
- v3完整响应与v2差异相同，并未实现完整同值。sample中的普通copy仍将原始F
  Jacobian转成C布局；修正后的primitive在相同保存数值的C/F布局上仍差
  7.105427357601002e-15。点证据在
  `outputs/native_osc_cross_layout_20261102/result.json`。
- 51055c2：仅将新增响应变体的J采样改为copy(order='K')，既隔离副本又保留布局。
  新增C/F布局下采样后首步力矩与SHA固定外部函数的回归，两项均通过。

旧默认预测器、独立checker内核、控制接口、目标/安全/时间/原容差均未改。
本轮结果支持该固定输入集合的差异来自这些运算/布局路径，未证明所有输入、
NumPy/BLAS版本或JIT执行均等价。

## 执行证据

| 内容 | 路径 | SHA256 |
| --- | --- | --- |
| v3完整响应，仍不同值 | outputs/native_replay_response_contract_v3_20261102/result.json | e6f471b08ac6d2bf8d85818217635be4a86a3a57e0d8305a345aa103e7ce443b |
| v4完整响应，同值 | outputs/native_replay_response_contract_v4_20261102/result.json | 9a651d11bdc90da6c4eb396f84a5679c9ce588ce2f6e0c8b145e488ab9442c7c |
| v4条件性ratio分解 | outputs/native_replay_response_contract_v4_20261102/saved_analysis.json | 8573e656ae6506435f262816426e25b20b51e4035c1c6ee531a33c5ba5e27f5f |
| 保存JSON精确复核 | outputs/native_replay_response_contract_v4_20261102/saved_exactness.json | 9fcdb3aaa536403de1814112b696db85e73ec9a7ad176e0fc501b76a4b6949c0 |
| 最终13项专项 | outputs/qa/native_response_layout_preservation_20261102/result.json | 4846ae418a9d2699b7917f304fe9ed48da2f78e56675248f8c07272e4e1c0ab5 |
| 前阶段11项专项 | outputs/qa/native_osc_layout_contract_20261102/result.json | 6bda0b1065d8b7b5b243fe9535e82bdfe23f8690eaaff6411faa3b58c3bd2d15 |

v4两侧各4响应/1200内步/1249查询/1model，总计8响应/2400内步/2498查询/2model；
v3同预算正常完成。含v1失败和v2，四轮实际总计26响应/7800内步/8120查询/8model。
0QP、0NNLS、0物理步、0WAM、0新场景；不追加旧search预算。
各进程实际退出且已回收，无自动重试；输入/源码前后检查通过。运行脚本和先于
模型构造写入的registration位于各自独占outputs目录，计划在对应v3/v4文档中。

v4原结果记录逐值精确一致；另对保存JSON的rollout和metrics做规范化字节比较，
包括符号零的JSON区别，四对均相同。没有新增查询。该保存记录复核不是另一轮
独立响应执行。

最终13项专项使用CPU runner、jobs32，全部通过，无skip/collection错误，所有
worker实际回收，源码前后不变；Ruff/format及Git空白检查通过。未跑全仓回归。
13项包含上述96输入，以及响应采样布局、裁剪/阻尼递推、跨周期因果latch和
零空间语义检查；仅支持相关局部合同。

## 下一阶段

按cap-distance设计，先独立登记并重新验证新预测器的初始化和72维有限差分
探针，再建立同一预测器的sum-q对照。保持原完整响应、严格决策和安全审核，
不能继承旧3557/72探针的exact标签，也不能将新预测器和新目标同时改变的收益
单独归于cap-distance。模型生成与最终留出任务验收仍未完成。
