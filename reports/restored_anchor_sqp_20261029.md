# 从已恢复终端余量的候选重新求解完整 H12

按[执行前登记](../plans/v0.0.1/restored_anchor_sqp_20261029.md)，从[已完成完整独立核验的 tail selected5](tail_translation_reserve_20261029.md)重新开始72变量的完整 H12 SQP。使用同一 actual368 初态、completed367 历史与既有参考，原控制器、caps、FD/QP/BFGS 和4000响应预算保持；新运行重置 B=I、缓存及阶段账本，不宣称恢复旧优化器状态。

实现使用隔离文件，不修改已执行来源。保留原 `1e-12` 命令准入但不 clip warm；首次计费响应构造完整 entry 后，经只读且与缓存隔离的 hook 与已闭合候选做严格 canonical 比较。完整命令字节、每项安全约束、误差、标签及全部 rollout 字段一致后才进入 FD/QP；不一致则保存完整 observed entry、逐路径差异和原始异常，并立即停止。完整失败路径实际计账1响应/300内步/313查询，answer=None、0 FD/QP；技术部分失败保留实际查询及完成计数，不伪造完整响应。

独立 ledger 重建同一完整初值门，并保留原 FD、QP 证书、实际接纳位移、同乘子曲率更新、ray/anchor 与全缓存排序审计。跨独立浮点路线继续使用原数值容差和严格分类；首条同生产路径 exact 检查不替代全量独立复算。

当前完成：新core27项、独立ledger32项、执行接线34项，共 **93项 native CPU 组件测试通过**，14.64秒，句柄36020实际exit0；新9个Python文件Ruff/format通过。覆盖实际入口失败保存、合法ULP越界命令不裁剪、大整数/布尔类别严格比较、完整初始 witness 只收费一次、0query非法输入、实际状态/latch重置、部分失败计账、owner输出来源闭合及完整replay资格。算法有限变更与独立原数学、全trace复算循环有AST逆核验，未据此声称新控制效果。

实现已提交 `aab996c`。统一110源/33模块 native CPU QA 已完成：**889 passed，132.76秒**，句柄82866实际exit0；2个警告均来自既有 SLSQP 合成用例。child实际wait0/reap、无信号、无重试；新110源与driver原字节保持，旧100源SHA为逐项相同子集。

QA产物为 `outputs/restored_anchor_sqp_native_qa_20261029_hjcgopjj/result.json`，SHA `6f77550d6df96e8ff56e9e97250e89081fd3a529d66f825cd55767d462c757ca`；日志SHA `4169482514acee808a3d198fb7a8de1ab95376c4cc64fa0fbffd44ded3f6a5e7`。driver `outputs/restored_anchor_sqp_native_qa_driver_20261029.py` SHA `294bc0c88b6c2c14ef318a9975d9b2dbd07c4c07f3bc371ef5fb8507da737e58`。

实际CLI准入正在执行，正式搜索尚未启动。GPU继续禁用；新增物理/world-model/fresh/参考消费0，实际仍cursor8、task/grasp0，原18正确参考/12WAM开发/24最终留出门槛及总目标不变。
