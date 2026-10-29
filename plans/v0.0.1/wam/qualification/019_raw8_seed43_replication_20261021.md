# 019：raw8第二采样种子重复性

执行前登记。目的为检验010的完整8400/guidance1/raw8在第二个固定采样种子下的重复性；017延长到60秒仍仅bowl成功，不能据此宣称跨seed稳定。本项沿用原30秒/600步主协议，与010的600截点比较，不替代017的1200结果，也不据结果挑成功seed或任务拼接策略。

固定三个旧task/init0，fixture SHA `f522ee333827d51ab66b5b8d1c2a7ebd8ceca3832403b3c54dfa56a821fb2a8e`；三个独立配置为`wam_raw8_seed43_task{0,1,2}_20261021.json`。实验seed42，实际环境/请求/响应/日志sampling seed43。唯一性能预算为3条WAM、1800真实物理步、225次H40请求；没有新native/oracle预算或fresh性能读取。全部case跑完整600步，不按任务成功提前停，不自动重试或覆盖。

与010 interval8/raw完整相同：8400非EMA、有效10槽mask、SDPA、35采样步、guidance1、shift5、20Hz/H40，每8真实步重新观测；完整模型旋转与原始opening，v3位置/保持夹爪、response inverse .5/rate5、无governor。使用原`run_wam_position_matrix.py`、`stability_position_simulation.py`与`run_wam_stability_service.py`，不修改其源码或注入新控制分支。物理等待推理期间完整状态/时钟不变仍须验证。

015已经实际执行相同三旧场景、seed43、interval8、600步正确参考资格：3/3稳定、全部1800端点与42对应资格逐值相同，SHA `9cd4cf46260eda2d87e13bd992c5ce871469d4ff2d3f38734129e2a1bd5fbba3`。它只是准入，不是WAM43成功；本项不重复消耗该资格预算。另须017完整审计SHA `b3910cdeb6669e5f6f74a654d44f2a214dc3b12249d33391e085efbb6a375168`和三个自有服务正常退出、原8400的163文件stat不变；仅核本项实际权重，不重查已记录外部删除的未使用15400。

task0/1/2分别固定物理GPU4/5/6与端口9744/9745/9746；CPU coordinator、matrix、OSMesa worker显式空CUDA/LD。启动前显存严格0、端口空闲，沿用GPU锁；模型单卡且每服务仅一物理worker。root保存注册/源码/输入SHA、完整命令、coordinator PID/startticks、三模型及matrix/worker实际PID/startticks/环境与ready身份；只能停止实际自有Popen进程组，不猜NVML/local PID映射，不影响GPU0–3或其他任务。

独立审计复用014显式seed43的字面机械/响应检查器，完整重建1800端点、225图像/条件/raw/normalized、固定控制/夹爪/时钟和原生谓词。首query用同旧fixture的实际43初态、当前PNG、实测锚点与完整响应来源认证；可与旧42核相同物理输入及除seed外固定条件，但**不要求43生成值等于42，也不借42输出冒充43来源**。新旧后续闭环观测分叉，不能称全部参考配对。另核完整HTTP/generation预算及所有实际自有进程退出。

逐任务和总体分别报告ever、terminal、最后40端点全部成功的stable、成功持续时间，以及实际state对生成参考的TCP/SO3/opening跟踪误差；保持原600结论及两个seed各自分母。只有完整组出结果才报告组成功率；pending不当失败。三旧场景/第二seed仍不等于多初态验收，不自动启动预留fresh或扩大种子、步长、checkpoint网格。后续新算法必须另登记，不能将本项成功归于不存在的控制改动。
