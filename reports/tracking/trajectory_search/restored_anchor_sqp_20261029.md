# 从已恢复终端余量的候选重新求解完整 H12

按[执行前登记](../plans/v0.0.1/restored_anchor_sqp_20261029.md)，从[已完成完整独立核验的 tail selected5](tail_translation_reserve_20261029.md)重新开始72变量的完整 H12 SQP。使用同一 actual368 初态、completed367 历史与既有参考，原控制器、caps、FD/QP/BFGS 和4000响应预算保持；新运行重置 B=I、缓存及阶段账本，不宣称恢复旧优化器状态。

实现使用隔离文件，不修改已执行来源。保留原 `1e-12` 命令准入但不 clip warm；首次计费响应构造完整 entry 后，经只读且与缓存隔离的 hook 与已闭合候选做严格 canonical 比较。完整命令字节、每项安全约束、误差、标签及全部 rollout 字段一致后才进入 FD/QP；不一致则保存完整 observed entry、逐路径差异和原始异常，并立即停止。完整失败路径实际计账1响应/300内步/313查询，answer=None、0 FD/QP；技术部分失败保留实际查询及完成计数，不伪造完整响应。

独立 ledger 重建同一完整初值门，并保留原 FD、QP 证书、实际接纳位移、同乘子曲率更新、ray/anchor 与全缓存排序审计。跨独立浮点路线继续使用原数值容差和严格分类；首条同生产路径 exact 检查不替代全量独立复算。

当前完成：新core27项、独立ledger32项、执行接线34项，共 **93项 native CPU 组件测试通过**，14.64秒，句柄36020实际exit0；新9个Python文件Ruff/format通过。覆盖实际入口失败保存、合法ULP越界命令不裁剪、大整数/布尔类别严格比较、完整初始 witness 只收费一次、0query非法输入、实际状态/latch重置、部分失败计账、owner输出来源闭合及完整replay资格。算法有限变更与独立原数学、全trace复算循环有AST逆核验，未据此声称新控制效果。

实现已提交 `aab996c`。统一110源/33模块 native CPU QA 已完成：**889 passed，132.76秒**，句柄82866实际exit0；2个警告均来自既有 SLSQP 合成用例。child实际wait0/reap、无信号、无重试；新110源与driver原字节保持，旧100源SHA为逐项相同子集。

QA产物为 `outputs/restored_anchor_sqp_native_qa_20261029_hjcgopjj/result.json`，SHA `6f77550d6df96e8ff56e9e97250e89081fd3a529d66f825cd55767d462c757ca`；日志SHA `4169482514acee808a3d198fb7a8de1ab95376c4cc64fa0fbffd44ded3f6a5e7`。driver `outputs/restored_anchor_sqp_native_qa_driver_20261029.py` SHA `294bc0c88b6c2c14ef318a9975d9b2dbd07c4c07f3bc371ef5fb8507da737e58`。

实际CLI准入已完成：110源中的Python文件Ruff/format、producer dry、独立checker dry和diagnose owner dry共5个子进程均actual wait0/reap、无信号/重试，句柄48369实际exit0。两入口分别核验2499来源，warm SHA `cb86ceaa9a474b6ace07ca3babf9c99fb2102fe1b04ec0e4641037a356c5f00e`；新增响应/查询0。

准入产物 `outputs/restored_anchor_sqp_admission_20261029/result.json` SHA `bd7bf2fbd8e1583094110936e5458f02e05e97519b310f3092a9d5e45e9f58e9`；driver `outputs/restored_anchor_sqp_admission_driver_20261029.py` SHA `d45f7389f567af592d5d9614c798fde00d796e26dcb797de65c4f9690017437f`。

正式生产已正常结束，原句柄8505实际exit0；child PID734373/start_ticks617178014实际wait0/reap、无signal、无自动重试，owner完成全部来源复核。首次完整记录guard checked/passed/exact均为true，只调用一次，发生在0阶段/0迭代/0QP时。共 **3657完整响应、1097100数值内步、1140985查询**，55阶段/55外层尝试、1160次QP内部迭代、272条实际ray；因 `trust_region_stalled` 正常停止，未追加预算。生产保存prefix9、完整witness0，最终selected不是最后接纳anchor。下述全量独立复算现已确认这些结果，仍不构成控制资格。

生产结果 `outputs/tracking_v5_restored_anchor_sqp_20261029/verified.json` SHA `b58f5d3d80abb6f4abe9e51cf785681bec1f15d067d0a684b6508142f649f6c1`；owner `outputs/tracking_v5_restored_anchor_sqp_execution_20261029/result.json` SHA `50da6df06d82973b6e3a36e1d363a5210553143cb6b317d654f7be09a63ef247`。

独立checker owner实际dry准入已正常关闭，原句柄3014实际exit0；child PID785548实际wait0/reap、无signal/重试，110源SHA不变，新增响应/查询0。产物 `outputs/restored_anchor_sqp_check_admission_20261029/result.json` SHA `6530424eda8d347594f81e82518cc9205e36d3f21751953a82c596a60fd828bb`。

唯一完整独立复算已在原句柄10388实际exit0关闭，child PID798911/start_ticks617290007由owner PID792603实际wait/reap0，无signal或自动重试。全部 **3657响应、272条ray、47份anchor、55个QP证书及46次曲率更新** 独立核验通过，严格分类分歧与QP失败列表均为空。两阶段实际合计7314响应、2194200数值内步、2281970查询；早先tail阶段42响应单独保留，不混入本轮计数。

原运行快照 `outputs/restored_anchor_sqp_execution_checkpoint_20261029/checker_started.json` SHA `171fb8165bc05d772d0537754dbca183d2c0f4af3f3e47992389e46d1e9602c9` 仍保留325/3657时的记录；完整结论依据最终退出和2526来源闭合，不以该中间快照替代。

| 完整关闭证据 | SHA256 |
| --- | --- |
| [独立复算](../outputs/tracking_v5_restored_anchor_sqp_check_20261029/verified.json) | `d6c83bccd02d6eca82bb11db490863f778b45f0c943a21caa76bf4fb07e34e44` |
| [checker owner](../outputs/tracking_v5_restored_anchor_sqp_check_execution_20261029/result.json) | `f8d91d931625b1dbf1f63d83debbb22a2b4d006d3c53c60f81f92fc32c31a92e` |
| [2526来源最终核验](../outputs/restored_anchor_sqp_final_closure_20261029.json) | `e8d6c7a95c1ec4c4ac20f59c529b66e7b744e0cd59bbb928f915d624e4f62fb4` |
| [全记录只读分析](../outputs/restored_anchor_sqp_analysis_20261029/result.json) | `05a0a2d60ec2312b921141d9b6a81dfb77da29f8b02a31fbd73eeeb3c8791a1a` |

最终核验和只读分析分别在原句柄58689/60495实际exit0；未新增模型响应、查询或物理步。110份QA源、旧100份子集、两owner、全部阶段和输入结束时重核不变。

| 保存候选 | 响应索引 | prefix | terminal原余量 | cursor9位置误差/旋转误差 |
| --- | --- | --- | --- | --- |
| initial | 0 | 9 | `7.73485517599e-5` | 11.47494mm / .05737520rad |
| selected | 3647 | 9 | `3.05571825469e-9` | 11.46528mm / .05732691rad |
| 最后接纳anchor | 3557 | 9 | `2.23557121672e-8` | 11.46529mm / .05732692rad |
| closest_safety_attempt | 3003 | 5 | `1.27045989658e-6` | 11.46664mm / .05732891rad |

selected仍超原10mm/.05rad caps，terminal虽过原审计容差却低于优化buffer `1e-8`；不能等同最后接纳anchor或鲁棒可行点。55阶段没有cursor增加，3384/3657响应为FD；已恢复的终端余量没有带来新的完整轨迹解。全部12端点、阶段及失败ray见[完整分析](../outputs/restored_anchor_sqp_analysis_20261029/analysis.md)。

## 已关闭的因果失败分析

[全272条ray分析](../outputs/restored_anchor_sqp_causal_failure_analysis_20261029/result.json) SHA `c9dd1bf9fce6a92b32936c0a68c066b64f5b426ad33fa35aeff8fe0db9cf1a6b`，原句柄74298实际exit0，2528引用结束重核；driver SHA `88f6720e75d29dca967060b984acb480c180636f27dee8dbbe3e8f77014fb620`。使用原FD名义有符号步长、实际ray位移、完整原精度约束重算分类，不执行新模型响应或QP。全部周期索引为0-based。

| 互斥子集 | 条数 | 保持前10周期时的因果限制 |
| --- | --- | --- |
| prefix9保持、q9下降且全部安全 | 46 | 已是可接纳方向 |
| prefix9保持、q9下降，仅terminal失败 | 40 | 末周期修复尚未被因果性排除，但未证明存在可行修复 |
| prefix9保持、q9下降，period10制动仍失败 | 12 | 至少需开放period10，单改末周期不能消除 |
| prefix9保持、q9下降，period0..9已有制动失败 | 106 | 固定这些周期时，改末两周期不能消除 |
| 丢失prefix9或q9不下降 | 68 | 末两周期无法改变此前误差 |

非终端buffer失败161条均涉及braking_upper，143条已在period0..9发生；原审计容差下的硬失败为129条，其中113条在period0..9。终端buffer失败95条；87条线性预测本身≥buffer，另8条在`1e-12`报告容差内才算预测通过，两种计数不混用。终端预测绝对残差最大`2.740676197627862e-4`，95条超过旧经验E17、20条超过10E17，因此旧阈值不能当作当前误差上界。

只读预选按cursor9归一化最大平方误差、响应索引排序：全部3657条中满足prefix≥9、原joint审计及`1e-8`buffer的278条记录，最好为3557（`1.3145505007230824`）；上述40条terminal-only中最好为3543（stage50、alpha1/8，`1.314526976519977`，terminal `-1.20453558423459e-6`）。后者的前段误差确实更低，但仍远未达到≤1的原caps，也没有安全修复证明。

后续候选是对该已保存前段做有界末周期修复，验证能否保留改善；本报告只给出必要因果条件，尚未登记或执行新控制搜索。预选及分析均未调用新模型；不以本次有限搜索失败推断物理不可达。

GPU继续禁用；新增物理/world-model/fresh/参考消费0，实际仍cursor8、task/grasp0，原18正确参考/12WAM开发/24最终留出门槛及总目标不变。
