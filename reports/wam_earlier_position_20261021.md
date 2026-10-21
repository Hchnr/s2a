# 006：8400连续开度映射的完整开发对照（2026-10-21）

006冻结的12条试验已完整执行并通过独立审计：7200物理步、133次H40请求。**alpha0仍只有cream任务1/3稳定成功；完整保留原始旋转的两组均0/3。** 相对同8400旧intent/hold对照，多了曾成功病例，稳定成功率未提升，尚不能进入新初态稳定验收或宣称映射普遍更好。

| 新方式 | 对应旧方式 ever / stable | 本次 ever / terminal / stable | 请求数 |
| --- | --- | --- | ---: |
| position_hold | intent_hold：0/3 / 0/3 | 0/3 / 0/3 / 0/3 | 45 |
| recovering_position_hold | recovering_intent_hold：0/3 / 0/3 | 0/3 / 0/3 / 0/3 | 26 |
| recovering_position_hold_alpha05 | alpha.5：0/3 / 0/3 | 1/3 / 1/3 / 0/3 | 29 |
| recovering_position_hold_alpha0 | alpha0：1/3 / 1/3 | 2/3 / 1/3 / 1/3 | 33 |

本次仅使用原bowl/wine/cream的init0、sampling seed42，600步、末40稳定门槛、H40/20Hz/execute40/maxobs80、35采样步/shift5/guidance1.0均与旧试验相同。权重为旧22-source非EMA8400。变化为已资格验证的固定`GripperPositionHold`：高开度motor位置跟踪、closing trend与reference≤.60共同触发锁存、保留原motor-limit hold及确认释放；其余执行器不改。

相对005/8800配置的独立逐字段review只发现checkpoint name/iteration、output、GPU/port和notes差异，四方式和三任务完整保留，未按cream成功单独选择场景。预算在`6694832`冻结，模型只用物理GPU4/9344，CPU编排和OSMesa均空CUDA/LD。

## 全部native事件与末40

| 方式 / 任务 | ever | terminal | stable | 总成功端点 | 最终连续成功 | 末40成功端点 |
| --- | --- | --- | --- | ---: | ---: | ---: |
| alpha.5 / cream | 是 | 是 | 否 | 22 | 9 | 18/40 |
| alpha0 / bowl | 是 | 否 | 否 | 87 | 0 | 14/40 |
| alpha0 / cream | 是 | 是 | 是 | 486 | 486 | 40/40 |

另外九条全部ever/terminal/stable为false。所有600个predicate flags、连续事件区间及末40逐点时间均独立聚合后保留于事件JSON，并与完整审计及原result一致：

- alpha.5/cream区间：`[124,125)`、`[131,133)`、`[148,149)`、`[575,584)`、`[591,600)`。
- alpha0/bowl区间：`[471,478)`、`[480,487)`、`[488,541)`、`[542,547)`、`[558,559)`、`[560,565)`、`[566,567)`、`[570,573)`、`[576,579)`、`[580,581)`、`[596,597)`。
- alpha0/cream为`[114,600)`，5.75–30.00秒连续486个成功端点；旧intent/hold对应case最终连续区间为`[144,600)`、7.25–30.00秒。

step从0计，范围左闭右开。`ever`和最后一个端点成功均不能代替末40稳定成功。alpha.5/0改变生成参考；alpha0每query保留当前实测anchor朝向，不能把这条成功记作raw full-checkpoint成绩。

![006完整12条native事件](../outputs/wam_stability_20261021/earlier_position_native_events.png)

## 跟踪与机械证据

各格为TCP位置RMSE mm / SO(3) RMSE rad；各组按三条等长600步汇总。

| 方式 | admitted | raw @ model clock | effective @ model clock | raw @ nominal clock | effective @ nominal clock |
| --- | --- | --- | --- | --- | --- |
| position_hold | 112.90 / .29013 | 112.90 / .29013 | 112.90 / .29013 | 112.90 / .29013 | 112.90 / .29013 |
| recovering | 33.24 / .09061 | 32.94 / .08858 | 32.94 / .08858 | 109.61 / .37161 | 109.61 / .37161 |
| recovering alpha.5 | 42.05 / .04501 | 37.37 / .14656 | 36.29 / .04384 | 145.51 / .36019 | 145.71 / .15795 |
| recovering alpha0 | 34.73 / .23248 | 41.76 / .45753 | 33.84 / .22456 | 56.50 / .60121 | 50.68 / .22456 |

recovering三任务模型时间/物理时间消耗比例为`.445/.231/.353`，alpha.5为`.381/.369/.514`，alpha0为`.694/.690/.568`。完整JSON保留原/有效参考、模型游标、名义物理时钟误差及独立重建的retiming/bridge/control状态，不将小admitted误差当作按原时钟准确执行。

非零target bilateral-contact端点为alpha.5/cream245、alpha0/bowl41、alpha0/cream53；其它九条为0。相对旧对照，alpha0/wine的该指标从86变成0，不能宣称所有任务机械过程都改善。接触端点数不证明抓稳或确定成功机制。

## 来源、独立审计及资源结束

12条的第一份输入PNG字节、完整raw/normalized `40×128`预测、anchor与初始physics SHA全部与同8400旧对应方式精确一致。后续闭环观测随实际执行分叉，不能把它描述为完全固定参考回放。

新wrapper显式复用冻结`140f7e8`的helper，在新文件逐字fork400行`verify_case`，唯一改动为首query来源callback绑定同8400旧12；源diff逐字校验通过，无dynamic patch。原33-case审计器及运行源码未修改。该版本独立检查全部逐步OSC、scalar gripper、raw/effective/admitted/nominal参考与因果重查询；1461份输入SHA重核一致。原004九条oracle资格仅作引用，本批新增oracle为0，绝不计入12WAM分母或新增物理步。

完整后model正常SIGTERM（-15），四matrix worker和CPU coordinator均退出，无execution/audit错误；记录时GPU4为0MiB。`earlier_position_final_audit.json`是完整snapshot19的逐字canonical副本，SHA相同。独立端点聚合与绘图只在12条完整审计后执行，没有新模型请求或物理步。

| 产物（位于`outputs/wam_stability_20261021/`） | SHA256 |
| --- | --- |
| `earlier_position_final_audit.json` | `46a07b31bc0e7bcee9b8c740a2c853ea718fa4867354794bdd85724fd6d0cbdc` |
| `earlier_position_native_events.json` | `11b6e8f851b1cd80be6d206abdc64b86d29ba4e40dcbec0e7b2f60ca0a84dd9f` |
| `earlier_position_release_snapshot.json` | `70eef401aea3f879d89e6c13f2f3fad89f9a86264250eb962c9fa23723766199` |
| `earlier_position_verify_case_source_diff.json` | `8e255c517bc57b47a8ff064ec1480b49bdc8697a17a990926c8d19c4b8313082` |

新入口`scripts/audit_wam_earlier_position.py`通过ruff check/format；实际完整case和SHA一致增量复用均已验证。原始数据保留在`position_earlier_development/8400_g10`。本报告只描述旧场景独立MuJoCo开发证据，不认证新初态泛化、预训练未见或真机，也不据此触发新预算。
