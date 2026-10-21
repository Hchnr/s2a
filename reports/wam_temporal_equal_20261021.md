# 020：同一绝对端点的因果参考集成

三条 WAM 仿真已完整独立核验，稳定成功 **1/3**，仅 cream cheese→bowl；bowl 曾短暂成功，末端未保持，wine 未成功。相同旧初态/seed42 的原 raw8 也是稳定1/3，但成功的是bowl；不能按任务选择策略拼成2/3，当前候选未建立多任务稳定收益，不提升为默认或打开fresh。

本报告是独立LIBERO/MuJoCo/OSMesa Panda仿真，没有真机。每任务一个旧init0、sampling42、600步/30物理秒/75次H40请求；20Hz、35去噪步/g1/shift5/SDPA/十有效槽mask/非EMA8400、每8步重观测、v3夹爪/noGov/response inverse .5/rate5固定。等待推理时物理state/time不变，不计推理延迟。checkpoint身份见[注册配置](../configs/wam_temporal_equal_task0_20261021.json:68)，为旧22-source warm-start run的iter_000008400，非新23-source的15400/15600。

| 任务 | ever | terminal | 最后40步稳定 | 成功端点数 | 末尾连续成功 |
| --- | --- | --- | --- | ---: | ---: |
| bowl→plate | 是 | 否 | 否 | 34 | 0秒 |
| wine bottle→rack | 否 | 否 | 否 | 0 | 0秒 |
| cream cheese→bowl | 是 | 是 | 是 | 487 | 303端点/15.15秒 |
| 合计 | 2/3 | 1/3 | 1/3 | — | — |

本候选使用此前已经生成、属于同一绝对物理端点的最多五份 world 参考，对位置和 opening 等权平均，对 SO(3) 采用已登记 chordal/polar 均值。各分量共享贡献列表与 `1/n` 权重。模型、当前图像/任务输入、控制器、S2A 反解、夹爪、每 8 步重观测和 600 步预算沿用已冻结协议。它改变执行参考，不能称为原始 checkpoint 成绩或仅机器人响应补偿。

与集成执行参考的全600端点跟踪误差如下；最新raw参考每次从当时measured anchor重新锚定、只走其短前缀，参考层和时钟不同，不能用其更小误差宣称实际集成跟踪更准。

| 任务 | 对集成执行参考 TCP RMSE mm / SO(3) rad | 对最新raw model-clock参考 TCP mm / SO(3) rad |
| --- | ---: | ---: |
| bowl | 138.753 / .300867 | 45.071 / .090762 |
| wine | 162.033 / .271684 | 37.695 / .087865 |
| cream | 45.064 / .118610 | 16.531 / .037666 |

这些执行参考下跟踪误差明显较大，不能把任务失败全部归为checkpoint；也不能因为cream满足原生任务谓词而声称全轨迹跟踪已准确。三条正确参考的同接口资格均成功、1800共同原始row/控制字段与007 canonical精确，TCP1.421–1.838mm；这是正确参考附近的资格范围，不保证任意不一致生成参考都可准确执行。[资格说明](wam_temporal_execution_qualification_20261021.md)。

每条完整 600 步轨迹生成 3000 个参考端点，其中按登记时轴有 2920 个作为贡献参与当前预算内的 600 个实际执行目标，末尾 80 个位于预算之后。这个数字是参考端点参与统计，不能当成物理步数、精度或任务成功；继承的最新 query 游标诊断也不能解释成集成丢弃了 80% 后缀。

最终可并排列出旧 010 raw8/seed42 的三个任务。两种策略首 query 输入相同，后续执行和观测会分岔，比较不能冒充同轨迹配对因果实验。原生任务谓词决定成功；低跟踪误差、接触或抬升不替代任务完成，也不认证抓持力或真实释放。

task2原启动在创建环境和任何生成前发生身份观察异常，0物理步/0模型生成，原error/release/log保留；未记录当时被拒环境值/worker PID，根因未知，不认定实际继承错误CUDA或已证明exec竞态。[021显式接续](../plans/v0.0.1/021_temporal_zero_step_continuation_20261021.md)只补未执行task2，新GPU6/9946/output与ready/go准入，原物理函数/效果参数不变。实际4次服务初始化、3条评分轨迹/1800步/225请求；原失败不计任务失败。旧task0/1八身份与新task2四身份分开保存，未伪造12并发ready。模型/矩阵/worker均已正常释放。物理卡仅4/5/6，CPU空CUDA/LD。

最终证据为新增显式接续合同`s2a.temporal-equal-explicit-continuation-audit.v1`，全部3条机械/夹爪/ensemble贡献/原生判定、225完整生成与HTTP、14128输入前后SHA及生命周期通过，errors为空。原89全组三ready审计和`report_wam_temporal.py`原严格入口没有放宽；它们不接受原两条partial输入，本报告不声称曾运行那个入口生成最终图表。

| 证据（相对`outputs/wam_stability_20261021/`） | SHA256 |
| --- | --- |
| `temporal_equal_continuation_review/complete3_verified.json` | `92915c61d6e3b7f92f2e9e0f997a582767ea9e55c9c8523eb099f1b1254c3c89` |
| `temporal_equal_preflight/wam_root_launch.json` | `d702b78dcd3f781b7326417ab404986f34d2d6b877cfe41fdd1344731ddb179e` |
| `temporal_equal_continuation_preflight/root_launch.json` | `5bb2349d9610d660c7a562fa4f65d1182a4d1741e1bb3e749b83378e0dbd1c1a` |
| `temporal_equal_continuation_review/actual_ready_snapshot.json` | `9b848c0139f3449c97c901a4cea9d660021acd6fbf3a8882d2310b653ede027b` |

注册为7025c45，物理函数49778dd；接续登记ea3f26d、准入/收尾497591c、独立审计01cf52c。实现依据：[同端点贡献](../src/s2a/temporal_ensemble.py:70)、[等权/SO(3)集成](../src/s2a/temporal_ensemble.py:92)、[独立物理执行](../src/s2a/temporal_ensemble_simulation.py:103)、[接续独立审计](../scripts/audit_wam_temporal_continuation.py:562)。本组没有新初态、额外seed或真机证据，不根据本组结果调集成权重再拼组。后续独立022文本state对照和023生成参考校正均须各自完整验收，当前稳定成功目标未达到。
