# 工作区迁移与恢复 — 2026-10-22

用户明确指定当前及后续工作区为`/share-mixed/project/eai_pwm/home/hcr/repos/s2a`。旧canonical目录在外部清理中丢失源码、Git与Python环境；未停止未知进程、未回写该目录。AGENTS.md已更新新路径与当前可见GPU2/3的限制。

## 已实际恢复

- 50组关键native/readout/轨迹目录已全部安全复制，0复制失败；原JSON声明及SHA未改。见`outputs/external_deletion_recovery_20261022/recovery_manifest.json`。
- 从已冻结源码快照恢复14个缺失S2A文件，396个现有文件逐SHA相同，没有覆盖已有差异；见同目录`source_restoration.json`。
- 未执行的clean残差原语、source归档核验、motor独立checker由防丢备份恢复，032 owner/worker在新工作区重建；不继承删除前QA为新源码测试结果。
- 本地开发Python3.12.3与模拟Python3.11.15可执行，未安装或升级依赖。旧模拟环境仅声明3.11且原解释器失效，运行数值等价性须另验。新cv2正式import-only探针及独立restore通过，CUDA/LD均恢复为空。证据：`outputs/local_python_environment_repair_20261022/`。

## 当前检查与执行边界

新路径/源归档/完整因子审计/子步motor目标/clean原语相关60项CPU测试通过。当前395份Python Ruff check/format通过。迁移后的完整集合实际826 passed、1 skipped、2 deselected、1 failed、12 errors；旧外部Cosmos源码测试另已排除。失败和错误均来自仍直接使用已删除旧根的历史审计/fixture，不伪称完整集合全绿。原输出见`outputs/workspace_relocation_validation_20261022/`，后续需按明确迁移协议逐项修复历史入口。

新032准入已验证完整历史来源：26个实际缺失的旧Cosmos源码有逐SHA精确归档，只证明历史字节，不认证模型当前运行；其余实际模拟器和S2A来源按当前路径严格核SHA。1746项路径映射在新prepared中明示；JSON声明路径与原SHA保持，实际I/O仅定位本工作区/安全备份，不回退旧根。operational record仅expert_path重定位，初态/native命令SHA不变。

032 prepared SHA `57ce8bb2466d182d7674d5578ca5eb64effc9b429889cbbb14990c8c1445739b`，dry-run成功。唯一实际owner工具session7558已启动8条新motor分支的CPU/OSMesa采集，实际worker737112/start560477660、cwd为本仓库、CUDA/LD均空已观测；本报告写入时尚未形成完整结果，不能称已通过实验。原baseline/latched_pose八条缓存不重复物理。后续只有独立checker完整核验native前缀逐位相同、全部41200子步与16格分母后才解释控制效果。

## 总目标状态

原030数据28/30训练资格False、原canvas精确门禁False均保留；0新WAM/真实fit/fresh。pure CPU clean residual原语实现及合成测试完成，但尚无实际采样器bridge、新适配参数或任务收益。最终24条真实WAM验收目标仍未达到，不能把恢复工作、软件测试或native正控当作目标完成。


补充终态：唯一032 owner现已正常exit0，8条全部完成并独立审计通过；native前缀逐位一致证明所测范围可比，41200子步核验通过。质量仍未改善，不推广。完整结果见[native motor保持报告](native_motor_hold_20261022.md)。
