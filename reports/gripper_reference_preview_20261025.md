# 066 有限开度预览：运行中

固定五条训练正确参考CPU对照共948步，wine原baseline复用已完整审查064。三任务候选均相同六步单调打开预览，H40/每8步/20Hz，位姿参考和原夹爪状态机不变；原desired opening和command_opening_reference分别记录。0模型/拟合/fresh，不将正确参考成功计入WAM。

七项预览因果性/边界及AST控制复用测试通过，Ruff通过，五条实际Python3.11 CLI dry均通过。持久化owner3421899/birth582677281、首case00 baseline客户端3421909/birth582677321实际启动并按/proc核实；最近首例41物理端点后记录已落盘，尚无完整结果。所有源码/配置及原064baseline在source_sha256.json绑定，无自动重试。

新增`gripper_preview_acceptance_checker.py`与`verify_gripper_reference_preview_case.py`已通过Ruff。检查器从保存完整生成窗口独立重建六步单调区间、索引及命令侧开度，再执行原完整机械/夹爪回放；不调用producer opening_preview。实际完整核验尚待案例结束，不将代码或lint通过视为执行正确。

候选须三任务3/3末40稳定并完整独立核验，才考虑更广对照；提前释放可能退化，不预设胜者。原061候选0/6、063失败及fresh关闭保持。

## bowl成对完整独立通过

case00 baseline与case01 preview均完整164步并实际正常收尾，独立机械/原生结果/标签/3当前图像/正常收据/来源检查通过，performance_qualified=True。baseline成功87步、尾连续79，preview成功92步、尾连续85。两者均稳定，不将计数提升当作多任务成功。独立检查器从已保存完整开度窗口重算选择，不调用producer；新增AST检查证明除独立预览重建、两处command reference输入及实际fork路径外，原机械检查和容差不变，该测试实际通过。

原owner3421899和wine preview客户端3423489按/proc实际确认存活，固定余下wine/cream继续。尚无候选三任务完整资格、WAM或fresh收益。

## wine候选完整独立通过，单例释放假设获得支持

case02 wine preview完整254步，独立控制/夹爪/预览重建、255端点标签、3图像、normal wait/reap及来源检查均通过，performance_qualified=True。独立native成功92步、尾连续70、末40稳定True；原064 baseline成功15步、尾0且稳定False。

paired_baseline_comparison.json消费两条完整日志前后SHA一致：原desired TCP最大差5.55e-16，原desired opening逐值一致，变化在命令侧预览。endpoint168实际开度baseline0.4438、preview0.9187，endpoint174原baseline已失去success而preview保持True，最终亦True。此为固定训练正确参考的单例配对收益，支持提前补偿释放响应滞后的假设，不证明通用因果或WAM参考收益，也不排除部分接触路径变化。

原owner3421899和cream baseline客户端3424245按/proc确认存活，继续固定cream成对；候选暂为2/2完整，三任务3/3及全批审查仍待完成。

## 固定正确参考pilot完整通过

五条948步全部完整独立控制/预览/标签/图像/正常退出检查通过，全source_sha256无漂移，owner及全部客户端已退出。结合原064 baseline，三任务baseline稳定2/3，统一六步preview稳定3/3；bowl成功/尾连续92/85、wine92/70、cream91/91。cream baseline85/85，两者均稳定。pilot_summary.json保存逐独立审查SHA，correct_reference_qualified=True、wam_performance_qualified=False。

这是三个训练正确参考的局部资格，不是闭环模型/新初态认证。下一步固定更广训练参考与原开发正控的闭合保持/提前释放回归，再登记模型开发对照；不按任务选择窗口或修改原fresh门槛，不以此覆盖061失败。
