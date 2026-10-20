# 017：raw WAM 延长执行时间的独立开发评测

**本报告当前记录协议与审计准备；三条 WAM 长时间案例尚未完成独立审计，暂不给出成功率。** 原 010 的 raw8 为 bowl → plate 成功、wine bottle → rack 和 cream cheese → bowl 失败，即 1/3。017 用同一 raw 策略统一执行 1200 步，检查另外两个任务是否只是需要更长执行时间。原 600 步结果保持原样，长时间结果单列。

证据等级仍为 LIBERO/MuJoCo 独立闭环仿真、Panda 机器人模型，不是真机；三个旧 task/init0/seed42 是开发诊断，不代表新初态或预训练未见数据上的成功率。

## 固定协议

[017 预注册](../plans/v0.0.1/017_long_raw_development_20261021.md) 和 `9fe490a` 的三份显式配置固定：8400 masked、guidance=1、35 sampling steps、shift=5、SDPA、H40、20 Hz、每 8 个物理步重新观测。完整 raw pose/opening 经冻结 position_hold/v3 夹爪与同一 S2A 执行，无 governor、无任务专用策略。每任务 1200 步、150 次 H40 请求，共 3 条/3600 步/450 次；不计模型等待延迟，等待时物理不推进。

环境 horizon=1300、ignore_done=false，恢复原初态后内部 timestep 从 0 开始；不增加初始化物理步，也不提前终止。完整时钟审计要求 1200 个实际控制端点、60 秒物理时间及逐步 dt=0.05 秒。以原生 success 判据分别报告前 600 步、后 600 步、全部 1200 步及末 40 步稳定结果。

GPU5/9655、GPU6/9656、GPU7/9657 分别绑定三个独立模型服务与单 CPU 仿真 worker；所有 CPU/OSMesa 进程 CUDA/LD 为空。模型只使用这三张允许卡，不使用 GPU0–3。实际 checkpoint 是旧 22-source masked `iter_000008400`，并非“最新”或 23-source；保留原完整字节 SHA 登记，重核实际使用的 163 个文件 size/mtime。

## 六条执行资格与明确的日志例外

WAM 前置资格使用同三个旧初态的新 1200 步 native 与 oracle 各三条，共 7200 物理步、0 模型请求。oracle 引用这次实际 native 的完整 1200 个端点，没有伪造旧示范末尾。旧 oracle 最后四个 query 只有 32/24/16/8 个有效参考，新 oracle 的 H40 与旧记录仅比较实际存在的共同前缀。

原严格资格 `long_raw_preflight/qualification_full6_verified.json`，SHA `c86569abeb1fb7e57bfd08468cc5e02cf598ef9c51f75a0da66b54e73e95357c`，保持 `qualification_passed=false`。三个 oracle 的完整 1200 步机械、任务成功及跟踪阈值检查均已通过，但 step599 的四个诊断字段因终端/内部插值分支出现浮点差异；不能将这份原严格结果改成 true。

新的只读 physical-core 资格明确分开这一例外：`long_raw_preflight/qualification_physical_core6_verified.json`，SHA `ba8872c55c9ab8b170a1b36efbe7cfdf501c1528cdd7dd74449adfa971663ade`。它核验前 600 步实际状态、命令、motor 与其它字段 canonical token 精确一致；仅允许原 step599 的 raw/effective desired TCP 与对应诊断 error 四个字段在既有 `1e-10` 容差内不同，逐项保存差异和独立重构残差。独立来源证明这些字段只供日志诊断、不参与控制；原 1200 步机械/任务/跟踪检查、进程释放和原 false 均保持。该新资格通过不计入 WAM 成功分母。

## WAM 独立审计合同

新只读入口 [audit_wam_long_wam.py](../scripts/audit_wam_long_wam.py) 提交 `66620b5`，源码 SHA `09973d4ebe6041081dfd005e9b9629a09c2b4f63fee773b86344d75d04bc712d`。它直接复用冻结 literal1200 mechanical checker、资格时钟 checker 与服务来源 helper，不修改任何实际运行源。

- 每条完整重放 1200 步机械、SO(3)、S2A、夹爪、因果观测、模型/物理时钟、输入 mask/归一化/seed/源 SHA。
- 首 query 继续对照同 checkpoint 的旧 006 参考；前 600 步全部字段要求 canonical JSON token 精确一致，包括数值类型和 signed zero。WAM 不采用 oracle 的 step599 日志例外。
- 前 75 次 PNG 要求字节一致，完整 40×128 raw 与 normalized 模型输出、条件、anchor、积分参考及物理快照均与旧 raw8 对应查询精确一致。
- 最终必须核完恰好 450 个独立生成目录与 450 条 `/generate` HTTP 200，没有额外 POST；核对三个私有服务、启动配置/源 SHA、外部正常释放以及 12 个历史 PID/startticks/parent/command/env 身份。
- 未完成案例不计失败，未完成全组三条时成功率为 null。native/oracle 与既有运行仅作为来源引用，不新增模型或仿真。

CPU 准备已核 literal1200 源差异、三个真实旧首 query 和 225 个既有服务输出，并拒绝错误 seed、服务根与原 false 资格。收据 `long_raw_review/wam_auditor_cpu_prelaunch_source_and_helpers_v2.json`，SHA `e82f4af7c069ae7ec23a9f08930263f8acf81b025efc895338923082830ede6e`，绑定 canonical token 守卫加入前的审计版本；追加守卫只加强精确性要求。实际 physical-core 资格合同及全部 284 项来源 SHA 也已只读重核：`long_raw_review/wam_auditor_actual_physical_core_contract.json`，SHA `3cdde381578851a2c17176cb5187ce59f1fb7a4e3019c99f97acb6a855c2afb7`。这些准备检查不构成三条 WAM 长时间结果。

另一审阅方已固定最终 `66620b5` 通读审计器，并实调用确认新 physical-core 合同通过、原严格 false 合同被拒绝；独立收据 `long_raw_service_review/wam_auditor_66620b5_independent_review.json`，SHA `6d904b5b11bace6dc7c3e2e8f470bd593a6aef033270a070352909749018e86d`。

## 当前运行与只读进度

root 启动收据 `long_raw_preflight/wam_root_launch.json`，SHA `d1924433405afd1ee92093779fa2b9b00b0c388ec86901852d1ac5758e12ad2f`，固定登记上述三条/3600 步/450 请求。首次实际只读审计 `long_raw_audit/progress_00_launch.json`，SHA `e13d116b56fb472da9402e58ab70351b5a2f99f2251792ed511dbd40b21402c8`，9352 项来源 SHA 前后完全一致；三条均 partial、0 条完整独立验证，成功率 null。已核三份真实 server info、8400 权重 stat 与 12 个运行中登记 PID/ticks/CUDA 身份，模型仅绑定 GPU5/6/7，所有 CPU CUDA/LD 为空。该部分快照采样时 request 文件计数为 0，不将它解释为整个审计期间模型没有执行请求。

本轮将回答旧场景下延长同一策略的物理执行时间是否改变原生任务结果；即使成功，也仍需另行冻结候选、完成未参与开发的新初态验收，不能直接宣称多任务稳定成功。

所有生成产物路径相对 `outputs/wam_stability_20261021/`。
