# 单个已生成抓取阶段的连续执行探针

保持原总目标和最终18/WAM/fresh门槛。针对raw/task0/init0/seed42已有未来闭合位姿静态接触，但每8步重规划丢弃未来段的线索，比较原执行与单块连续跟踪。

两组各176周期，合计352。相同初态精确重放前136步并重建夹爪状态、实际tracking计划历史和coupled创新；前缀全部137端点必须与原SHA相同。基线继续原记录40步，全部177端点必须匹配。候选只使用原step136已生成的40个target/opening，依序执行index0..39，不读后续生成块，固定控制器、gripper模型与arm tracking细化保持。每步使用实际M/J/q/dq/pose/twist、上一动作与完成的因果历史，经守护seed规划、tracking prepare/commit、真实env.step、完成端点observe。记录controller goal一致性、动作幅值/变化率、每步实际状态、抓取接触、成功谓词和TCP误差。

不把单块40步局部结果当作600步任务稳定成功；不把固定已生成参考续跑当作响应式WAM或留出初态验证。世界模型调用0、fresh0，CPU CUDA/LD显式空，沿用原OSMesa/线程环境。无自动重试。

入口 `scripts/execute_committed_generated_chunk.py`。实际执行前权威登记位于 `outputs/committed_generated_chunk_execution_20261027/registration.json` 与 `outputs/committed_generated_chunk_20261027/registration.json`。产物使用先序列化再独占写入的publish_result，防止NumPy值导致截断summary。结果无论正负均保留。
