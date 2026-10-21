# 8400四方式独立物理开发对照（2026-10-21）

冻结的12条旧初态开发试验已完整执行并通过独立审计：7200物理步、136次H40模型请求，未超过预注册12条/7200步/最多180请求预算。**alpha0分量诊断在cream任务1/3稳定成功；两组保留原始旋转的完整WAM参考均0/3，尚未达到多任务稳定成功。**

| 方式 | 原始完整参考 | ever / stable | 三任务稳定结果（bowl / wine / cream） | 请求数 |
| --- | --- | --- | --- | ---: |
| intent_hold | 是 | 0/3 / 0/3 | 0 / 0 / 0 | 45 |
| recovering_intent_hold | 是 | 0/3 / 0/3 | 0 / 0 / 0 | 27 |
| recovering_intent_hold_alpha05 | 否，旋转delta乘0.5 | 0/3 / 0/3 | 0 / 0 / 0 | 28 |
| recovering_intent_hold_alpha0 | 否，每query保持实测anchor朝向 | 1/3 / 1/3 | 0 / 0 / 1 | 36 |

8400属于旧22-source run的非EMA `iter_000008400/model`；guidance1.0、H40/20Hz、execute40、maxobs80、35采样步/shift5、seed42固定。每条实际执行600步，不计模型墙钟延迟，以末40个native-success端点全true定义稳定成功。场景固定为三个旧任务的init0，未打开新初态验收。

alpha0仍使用WAM位置/开度输出，但逐delta旋转改为单位旋转；它不是全episode固定朝向，也不是完整checkpoint性能。旋转衰减还会改变局部平移积分后的world路径，不能声称只改了最终朝向。

## 成功事件与末40证据

成功仅发生于`recovering_intent_hold_alpha0/task2_init0_s42`，原native-success端点区间（step从0计，范围左闭右开）：

- `[130,139)`：9个成功端点，对应6.55–6.95秒。
- `[144,600)`：456个连续成功端点，对应7.25–30.00秒。
- 总465个成功端点，最后456个连续成功，末40端点全部true；ever/terminal/stable均true。

独立端点聚合直接读取保存的native predicate，重算区间/尾长/末40并与完整审计和原result逐字段一致；保留全部600个flags及末40逐点时间。它不重新运行任务predicate或物理。

![完整12条native成功时间线](../outputs/wam_stability_20261021/earlier_recovery_native_events.png)

## 跟踪误差与参考时钟

各格为TCP位置RMSE mm / SO(3) RMSE rad，三条等长试验按endpoint汇总。`model`表示retimed模型游标；`nominal`表示从本query开始按实际物理时钟推进的原20Hz参考。

| 方式 | admitted | raw @ model | effective @ model | raw @ nominal | effective @ nominal |
| --- | --- | --- | --- | --- | --- |
| intent_hold | 120.58 / .37818 | 120.58 / .37818 | 120.58 / .37818 | 120.58 / .37818 | 120.58 / .37818 |
| recovering | 35.45 / .05492 | 35.22 / .05445 | 35.22 / .05445 | 125.61 / .42465 | 125.61 / .42465 |
| recovering alpha.5 | 29.05 / .03970 | 29.25 / .14134 | 28.88 / .03939 | 69.89 / .35519 | 70.44 / .15207 |
| recovering alpha0 | 18.44 / .11441 | 26.09 / .36914 | 18.36 / .11240 | 47.97 / .46176 | 44.95 / .11240 |

recovering三任务模型时间/物理时间消耗比例为`.363/.210/.222`，alpha.5为`.546/.298/.447`，alpha0为`.843/.474/.931`。较小admitted误差不能解释为按原时钟完成原始参考。控制、重查询、retiming、bridge恢复、原始/有效参考、夹爪motor与OSC均通过冻结审计器的独立重建。

记录的target bilateral-contact端点数：alpha.5/cream为40，alpha0/wine为86，alpha0/cream为90；其它九条为0。接触指标不是抓稳证明，也不能单凭它确定成功的机械过程。不能把本批描述成全部ever0或全部无接触。

## 来源、运行结束与边界

- 每任务四方式的初始physics SHA、step0图像像素SHA及第一份原始`40×128`预测逐值完全一致；后续闭环观测允许因执行变化而分叉。
- 配置在`fd17d78`冻结。新增wrapper直接import原`cd1580b`完整审计函数；原审计器SHA仍为`ee20bbc41688a16f590370c5a1a90f7e30b78d2b24f6f15bb0be57086cc0e4b0`，其原30-case协议未修改。706份输入SHA重验通过；增量复用只接受审计器字节和全部输入SHA一致的已完整重建case。
- 既有artifact07的6/6 oracle资格只作来源引用。本批新增oracle条数为0，不计入WAM分母；它不是这12条新增oracle或新holdout资格。
- 自有CPU编排和物理worker空CUDA/LD，模型只使用物理GPU4/port9244。完整后模型正常SIGTERM退出（-15），四matrix worker及coordinator退出，无execution error，GPU4回到0MiB。
- 这是固定控制器下独立MuJoCo仿真，没有真机执行。历史R1物理同步、预训练未见和普适恢复安全均不由本批证明；不把alpha0的一条开发成功外推为整体稳定。

| 产物（相对本仓库） | SHA256 |
| --- | --- |
| `outputs/wam_stability_20261021/earlier_recovery_final_audit.json` | `aafb3a0aed076ecaeb669445c4c535f96088445d2e39e110c811d11c4d0d567d` |
| `outputs/wam_stability_20261021/earlier_recovery_native_events.json` | `907aa81bac3a5a7bec831d9551eee814816ab44b7f36d9804d9ce314f95a9ba4` |
| `outputs/wam_stability_20261021/earlier_recovery_release_and_pair_audit.json` | `511b5c8b92677b5c5ffe90f4bd3d3c8454e1ce3543b8b1f992520e28b014a390` |

新增只读入口为`scripts/audit_wam_earlier_recovery.py`与`scripts/audit_wam_earlier_outcomes.py`。原始完整试验保留于`outputs/wam_stability_20261021/recovery_earlier_development/8400_g10`，已有产物拒绝覆盖。
