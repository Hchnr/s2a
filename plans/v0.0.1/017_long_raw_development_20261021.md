# 原始WAM的60秒持续效果诊断 — 2026-10-21（Asia/Shanghai）

状态：新增实现/物理之前预注册；不修改010/011/016已运行的协议或原600步结论，不打开预留新初态性能。

010 raw8的三个旧task/init0/s42已全600步完整独立核验，稳定1/3（bowl→plate）。wine bottle→rack在末35个端点才记录双指接触，末端仍持物、最高抬起52.84mm，未完成；cream cheese→bowl虽曾抬83.77mm/接触10端点，但末端掉落/开度近1，并非已知只差时间。bowl还需检查再维持30秒是否丢失。因此单独、三任务统一把开发物理预算延至1200步/60秒，区分完成进展慢与持续失败；不预设延长会有效，也不针对wine单独放宽。

本协议保留每8步当前图像/任务重观测、8400/guidance1/35steps/shift5/SDPA/active-slot mask、raw H40/执行40、20Hz、response逆.5/rate5、无gov及v3参数；没有修改模型pose/rotation/opening、夹爪事件或使用物体/成功输入。每条1200实际控制步，最后40端点原生连续成功，同时另外报告原600步截点、首次/最长成功、31–60秒进度和跟踪。600步结果不被1200步成功追溯覆盖。此处排除推理延迟，延长的是物理任务窗口。

旧factory默认horizon1000/ignore_done=False，1001前self.done会拒绝；不能靠忽略外层success-done或手动清self.done突破。只读证据`libero_1200step_readonly_feasibility_20261021.json` SHA `3ad53ac56e3bb0854f3ab16fd5ad2c390020e17fe5f2c312ffadff7f03478f81`已绑定实际源。新增隔离factory显式horizon1300、ignore_done=False，保留原恢复/5步初始化；记录effective_horizon、initial/final内部timestep、control频率与实际dt，1200评测不消费初始化预算。保守留100步初始化余量，明确不把1300当评分步数。原factory和正在跑的源码字节不改。

有限预算分别是：native3×1200=3600步，oracle＋同一冻结S2A3×1200=3600步，条件WAM3×1200=3600步/450个H40请求（每条150）。合计9条/10800步；不是新鲜初态验收。native复用原专家命令及结束后已有保持约定，不新规划；oracle参考只来自本轮完整native实际state，不伪造native/补写后600日志。native和oracle均须完整/末40原生成功，oracle TCP≤10mm/SO3≤.05rad。先核六条正确参考资格，失败不启动该新WAM预算。

新native/oracle及新WAM的前600步须与各自既有对应执行记录核对：共同state/command、模型PNG及full40×128 raw/normalized输出应精确一致；horizon/generation墙钟/输出身份等新元数据另列。若前600实际出现分叉先审接口原因，不能直接归因“多跑了30秒”；完整1200审计允许的数值容差沿用已有接口审计，精确与容差结论分开。模型预测/原实际参考/机械限制/开度motor及状态连续性/物理暂停SHA全程独立重建。只扩固定case长度及新source/metadata绑定，不改原checker。

WAM三分区一task各一私有模型/单CPU物理worker：task0→GPU5/9655，task1→GPU6/9656，task2→GPU7/9657。task0须010 interval16全部6case完成和受控释放；task1须011全部144及受控释放；task2须016全部3case及受控释放。各自自有PID退出、显存严格0/端口/锁检查后才启动，不抢其他用户进程。最多三个新模型；旧010 GPU4任务继续完成，0–3禁用，CPU/OSMesa空CUDA/LD。所有新入口/配置/预算及依赖在启动前提交，输出独占新建，失败/取消无自动重试。

本轮完整结果按单一1200预算报告三个任务，并并列600截点，不能将“曾成功”代替末40稳定。旧任务开发成功仍不认证新初态泛化；最终预留新初态的候选/时长协议如有调整，必须在任何fresh性能前另冻结，并保留既有30秒失败证据。本017不授权直接执行fresh，不引入任务规划器或选择成功seed。

资格启动前明确oracle边界：原600步oracle在step592的最后query只有8个未来端点，新1200步的同次query有40；要求其前8端点及前600实际state/command一致，不能要求原来不存在的后32端点或execution_endpoint_count相等。WAM固定每次完整H40，前75次full raw/normalized和图像仍要求对应精确回归。native正确参考是实际物理记录，未来长度差异单列，不按未知尾部猜补。
