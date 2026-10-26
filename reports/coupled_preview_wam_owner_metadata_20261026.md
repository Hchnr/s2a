# WAM预览总调度与独立元数据审查

新增`run_coupled_preview_wam_development.py`与`verify_coupled_preview_wam_metadata.py`。总owner仅接收新登记，使用一个允许UUID模型进程，串行执行固定12案例/900请求，再以原生Python3.11运行完整机械/数值审查。保留原服务ready身份与/proc核对、HTTP超时、actual wait/reap、失败保留、GPU释放和不自动重试；模型通信端口31365与MASTER31366分离。

执行结果绑定真实18/4907控制closure，不使用旧控制器的授权。采集器、服务和verifier均指向新版本。新增owner实际`--help`退出0，日志为`outputs/commit_preview_20261026/wam_owner_help.txt`。

准备阶段存在引用顺序问题：prepared需要独立输入preflight结果，最终执行配置又需要prepared及execution-registration引用。为避免循环，新增`make_case_profile`只生成固定输入/控制配置，不包含`wam_development`执行引用，不能直接运行；登记后`make_case_config`在相同profile上绑定实际引用。独立metadata逐字段检查12份profile和完整物理前置资格，不能把profile测试当作执行登记。

18项owner/metadata/collection组合CPU测试通过，覆盖新服务命令与GPU UUID、独立原生解释器、继承的RPC/ready逻辑、全部12个非执行profile，以及gain/UUID/步数/专家输入/提前执行引用篡改拒绝。相关代码Ruff与格式检查通过。

尚需实现并实际运行新prepare/input-preflight/register流程及完整真实CLI准入；当前没有新prepared或执行登记，也没有启动模型、开发物理轨迹或fresh。已完成的正确参考18/18资格保持，总目标仍需WAM开发及留出新初态证据。
