# 轨迹搜索的数学与执行合同回归

测试分为core、独立audit及execution输入准入。共享合成状态、手工证书等辅助
定义在`tests/support/trajectory_rollout.py`，不通过裸测试模块名共享。

迁移前最近七模块包含182个测试节点；验收按完整文件路径映射比较nodeid，保留
测试名和参数ID，不能只核数量。原accepted-point数学测试仍位于
`tests/test_accepted_point_sqp_math_audit.py`。

新基础组件和输入层测试在`tests/unit/infrastructure/`。测试使用合成数据与替身；
完整原生响应复算及物理控制具有独立登记，测试通过不替代这些资格。

[实验入口](../../../scripts/tracking/trajectory_search/README.md) ·
[结构设计](../../../docs/architecture/repository_design.md)
