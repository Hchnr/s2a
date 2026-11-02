# 094 group4六原translation探针内环追踪

093六baseline前缀均匹配，目标周期150调用无限幅，实际uncoupling=True，group4持续触桌。对086 case29..34（group4 x-/x+/y-/y+/z-/z+）执行原初态、seed42、固定命令与控制接口重放。branch index0..5仅为本轮输出编号，全部原anchor_group=4，不作为新训练数据。

沿用已测试的ControllerTorqueTrace，仅末周期25内环观察。核验每个prefix endpoint完整state/SITE pose/twist与原086对应分支一致，raw/clip/ctrl一致、时间和实际uncoupling；按六原分支逐条报告桌面接触次数、首次/末次接触、限幅和力矩余量。用093 group4 baseline作只读比较，不调整本次命令/模型。

持久owner管理六worker/checker及自身实际wait/reap，完整来源冻结、1观察器测试与6CLI dry-run准入，无自动重试。预算6×43=258显式物理控制周期/6450积分子步，6次环境初始化内部步骤另列不纳入命令周期数；世界模型/fresh/fit均0。保存于outputs/contact_probe_trace_20261025，不覆盖旧产物。不将接触真值加入在线条件，不授予控制资格，原验收门槛不变。
