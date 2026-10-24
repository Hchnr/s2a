# 033：用户指定share-mixed工作区及历史来源恢复

2026-10-22用户明确要求改用`/share-mixed/project/eai_pwm/home/hcr/repos/s2a`，覆盖此前正式目录约定。外部PID665986在`/share/project/eai_pwm/home`执行`rm -rf hcr`，源码、Git及venv被删；没有停止未知进程或回写清理目录。已将50组关键native/readout/轨迹证据全部复制到本仓库outputs/external_deletion_recovery_20261022，manifest保留实际结果；旧声明及原SHA不改。

从历史431源码/快照恢复14个当前副本缺失文件、396个现有文件逐SHA相同，无覆盖当前本地差异。恢复新增CPU原语、motor checker等未执行实现；仅AST/测试后才能作为新源码使用，不能继承删除前全仓结果。旧032实现未实际prepare/物理，首次QA818P/1外部源码missing失败保留。

路径定位仅用于实际I/O：旧S2A outputs/wam_stability引用优先安全恢复副本，其他旧S2A引用定位当前对应文件；绝不回退正在清理的旧根。JSON引用声明路径及SHA保留，不修改旧结果。operational records仅expert_path允许重定位，其余初态、native命令SHA与任务/seed不变。当前S2A及实际模拟器源码必须与固定历史字节核对；旧Cosmos缺失只能通过精确历史快照取证，不认证当前源码存在，更不授权模型运行。新operational配置/来源清单单独冻结。

本地开发venv继续Python3.12.3；env311旧base链接失效，用户工作区现有解释器3.11.15，旧cfg未声明patch，不能声称解释器数值等价。只修本地链接/元数据，无依赖安装；新cv2正式import-only探针及独立restore真实通过。032须重新CPU QA，真实运行必须用完整native前缀逐位回归证明本实验范围内可比，不一致即保留False并停止因果推广。

所有代码/命令/产物从新工作区运行。GPU仅当前可见2、3对应两UUID，恢复期间均CPU空CUDA/LD。032预算仍8环境/1648步/40canvas，不以迁移重复旧8条物理；原28/30训练资格False、模型适配未启动、fresh24未打开，最终成功门槛不变。

## 迁移后的实际检查边界

新路径/归档/电机子步/完整因子审计/clean原语的相关60项CPU测试通过。完整CPU集合（已单列排除一个旧外部Cosmos测试）实际826 passed、1 skipped、2 deselected、1 failed、12 setup errors：失败是旧long qualification审计直接打开已删除canonical配置；12个错误是旧接续测试fixture对旧绝对路径执行relative_to(新REPO)。这些是未迁移的历史重放入口，不在032的执行闭包；不篡改其冻结源码或伪称整仓全绿。原完整结果/日志保留，新入口通过明确resolver使用完整历史数据与源码字节，并另测旧路径不回退的拒绝行为。

032仅按已通过的实际操作闭包准入：新owner/worker/checker与路径/归档测试、原几何控制/500Hz重构的实际通过结果、当前cv2独立资格和完整源SHA。实际仿真必须在独立checker中完整保护native前缀，不能以旧历史测试不可运行而忽略数值不一致。后续逐项迁移历史入口须另存新source版本；本轮不为让检查全绿而改旧指纹或去掉错误记录。
