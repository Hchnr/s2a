# 从已恢复终端余量的候选重新求解完整 H12

按[执行前登记](../plans/v0.0.1/restored_anchor_sqp_20261029.md)，从[已完成完整独立核验的 tail selected5](tail_translation_reserve_20261029.md)重新开始72变量的完整 H12 SQP。使用同一 actual368 初态、completed367 历史与既有参考，原控制器、caps、FD/QP/BFGS 和4000响应预算保持；新运行重置 B=I、缓存及阶段账本，不宣称恢复旧优化器状态。

实现使用隔离文件，不修改已执行来源。保留原 `1e-12` 命令准入但不 clip warm；首次计费响应构造完整 entry 后，经只读且与缓存隔离的 hook 与已闭合候选做严格 canonical 比较。完整命令字节、每项安全约束、误差、标签及全部 rollout 字段一致后才进入 FD/QP；不一致则保存完整 observed entry、逐路径差异和原始异常，并立即停止。完整失败路径实际计账1响应/300内步/313查询，answer=None、0 FD/QP；技术部分失败保留实际查询及完成计数，不伪造完整响应。

独立 ledger 重建同一完整初值门，并保留原 FD、QP 证书、实际接纳位移、同乘子曲率更新、ray/anchor 与全缓存排序审计。跨独立浮点路线继续使用原数值容差和严格分类；首条同生产路径 exact 检查不替代全量独立复算。

当前完成：新core27项、独立ledger32项、执行接线34项，共 **93项 native CPU 组件测试通过**，14.64秒，句柄36020实际exit0；新9个Python文件Ruff/format通过。覆盖实际入口失败保存、合法ULP越界命令不裁剪、大整数/布尔类别严格比较、完整初始 witness 只收费一次、0query非法输入、实际状态/latch重置、部分失败计账、owner输出来源闭合及完整replay资格。算法有限变更与独立原数学、全trace复算循环有AST逆核验，未据此声称新控制效果。

实现已提交 `aab996c`。统一110源/33模块 native CPU QA 已完成：**889 passed，132.76秒**，句柄82866实际exit0；2个警告均来自既有 SLSQP 合成用例。child实际wait0/reap、无信号、无重试；新110源与driver原字节保持，旧100源SHA为逐项相同子集。

QA产物为 `outputs/restored_anchor_sqp_native_qa_20261029_hjcgopjj/result.json`，SHA `6f77550d6df96e8ff56e9e97250e89081fd3a529d66f825cd55767d462c757ca`；日志SHA `4169482514acee808a3d198fb7a8de1ab95376c4cc64fa0fbffd44ded3f6a5e7`。driver `outputs/restored_anchor_sqp_native_qa_driver_20261029.py` SHA `294bc0c88b6c2c14ef318a9975d9b2dbd07c4c07f3bc371ef5fb8507da737e58`。

实际CLI准入已完成：110源中的Python文件Ruff/format、producer dry、独立checker dry和diagnose owner dry共5个子进程均actual wait0/reap、无信号/重试，句柄48369实际exit0。两入口分别核验2499来源，warm SHA `cb86ceaa9a474b6ace07ca3babf9c99fb2102fe1b04ec0e4641037a356c5f00e`；新增响应/查询0。

准入产物 `outputs/restored_anchor_sqp_admission_20261029/result.json` SHA `bd7bf2fbd8e1583094110936e5458f02e05e97519b310f3092a9d5e45e9f58e9`；driver `outputs/restored_anchor_sqp_admission_driver_20261029.py` SHA `d45f7389f567af592d5d9614c798fde00d796e26dcb797de65c4f9690017437f`。

正式生产已正常结束，原句柄8505实际exit0；child PID734373/start_ticks617178014实际wait0/reap、无signal、无自动重试，owner完成全部来源复核。首次完整记录guard checked/passed/exact均为true，只调用一次，发生在0阶段/0迭代/0QP时。共 **3657完整响应、1097100数值内步、1140985查询**，55阶段/55外层尝试、1160次QP内部迭代、272条实际ray；因 `trust_region_stalled` 正常停止，未追加预算。生产保存prefix9、完整witness0，最终selected不是最后接纳anchor。这些科学结果仍待全量独立复算，不能称控制资格。

生产结果 `outputs/tracking_v5_restored_anchor_sqp_20261029/verified.json` SHA `b58f5d3d80abb6f4abe9e51cf785681bec1f15d067d0a684b6508142f649f6c1`；owner `outputs/tracking_v5_restored_anchor_sqp_execution_20261029/result.json` SHA `50da6df06d82973b6e3a36e1d363a5210553143cb6b317d654f7be09a63ef247`。下一步为独立checker owner实际dry准入、全部3657响应复算及来源闭合；任何失败保留原记录，不覆盖重跑。

GPU继续禁用；新增物理/world-model/fresh/参考消费0，实际仍cursor8、task/grasp0，原18正确参考/12WAM开发/24最终留出门槛及总目标不变。
