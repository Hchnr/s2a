# OSC 算术契约验证（2026-11-02，Asia/Shanghai）

## 结论与范围

新增独立、尚未接入预测器或搜索的
`src/s2a/research/response_models/forward/native_osc_torque.py`。
其显式 NumPy 运算顺序在保存初态及 128 个 seed42 合成输入上，与独立回放所用
外部函数去除装饰器后的结果逐字节一致。原预测器、独立 checker、容差、原运行
记录及预算均未改变。没有新增动力学查询、rollout、物理步或模型调用。

这不是完整轨迹的算术资格，也没有修复原 fresh_ratio 失败；当前完整 H12 witness
仍为零，不能宣称可靠跟踪或任务成功。GPU 继续禁用。

## 已有失败定位

完整独立诊断复算 122 响应、36600 内步和 38065 查询后仍失败：首次严格超差在
stage2 restore、anchor3 到 response4、alpha1。base braking_upper 的 raw slack
差约 +4.44e-16，trial terminal 的 raw slack 差约 -8.66e-16，经 reserve 尺度
放大后 actual 差约 -5.3064e-11，ratio 差约 -2.13717e-10，超过原 1e-10。
不能把它归因于 TCP 或仅 terminal 项。

该失败的 authoritative checker：
`outputs/tracking_v5_state_vector_dual_recovery_fresh_diagnostic_v2_20261029/verified.json`
SHA256 `5d83387542539f08ad79bbf201b3ef9011c664a40b46c970f8a03e62b13aa7e4`。

保存数据的条件性分解并非重新执行搜索，也不能覆盖严格失败；原生环境分解：
`outputs/state_vector_fresh_ratio_decomposition_native_20261029/result.json`
SHA256 `6519d220c6e5e0ee0b7aa712a7e82dc4426a55018d6acf65411f60437c840f97`。

## 算术实现与证据

原实现使用 solve(M,J.T)、整块惯量切片、分开任务投影，以及先计算 N.T@M。
独立回放则显式 inv(M)、分别计算位置/朝向惯量、单次拼接任务投影，并先计算
M@spring 再由 N.T 投影。数学等价不能保证双精度运算逐字节一致。

已有单点审计在相同保存输入上复现原预测器力矩，原生顺序与之最大差为
1.4921397450962104e-13；全部六项运算顺序切换后与独立 NumPy 函数一致。
单点证据不能证明完整失败的因果来源：
`outputs/native_torque_algebra_contract_20261029/result.json`
SHA256 `db0e1b60c8a52efc72b7d687631300369e049bed29ead7663c45d9749b05aac1`。

本次 CPU 验证在 Python3.11 / NumPy1.26.4、各数值库单线程、空 CUDA/LD 下执行。
外部 control_utils.py 固定 SHA256
`1d0f1a063894b1e8289339e223665e14e8aec492c31e9e00f42edea8dc735f43`；
仅抽取三个外部函数并移除装饰器，不导入 robosuite，不执行历史仓库模块。
因此这里没有验证 Numba/JIT 的运算差异。

129 个输入覆盖保存初态、随机 SPD mass、满秩/降秩/零 Jacobian、随机姿态及
0/4/10/17 零空间增益；全部 pre-bias 力矩逐字节一致，输入与代码 SHA 前后不变。
可复核脚本及逐例记录：
`outputs/native_osc_primitive_validation_20261102/validate.py`、`result.json`；
result SHA256 `66ccdf47afe3ff3b75326241e3542faa4cddef51a9d7fb47c53b545dcff251a8`。

三个专项测试独立检查 SITE 平移/旋转方向、冗余轴弹簧与阻尼、零秩时的质量加权
零空间响应及输入不变性。统一 CPU runner 指定 jobs32，实际仅三个测试项，全部
通过，无 skip 或 collection error；runner 已正常退出并回收子进程，来源前后不变。
`outputs/qa/native_osc_primitive_20261102/result.json`
SHA256 `96d5b50edf1d2e47871dfe7369def82f5146e22821ff06fca740899c75cc73f2`。
新增两文件 Ruff/format 通过；未运行全仓回归。

## 下一步准入

先独立登记有限 CPU 完整响应对照，检查目标构造、递推、裁剪及全部约束计算是否
与独立回放一致，不能仅用本力矩单元的通过推断严格 ratio 已修复。
若引入新预测器，须按
`plans/v0.0.1/tracking/trajectory_search/cap_distance_gn_design_20261029.md`
重新验证初始化与探针，并先建立使用同一预测器的 sum-q 对照；费用单列，不继承
旧 exact 标签、不向已关闭的 122 响应预算追加运行。之后才比较 cap-distance GN。
