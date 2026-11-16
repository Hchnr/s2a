# seed-first 对照的零求解预检修复

2026-11-16。[原登记](fast_seed_search_20261116.md)的首次零求解 dry 在读取
固定输入元数据后退出：比较配置与历史 profile 输入均名为 `config_ref`，
展开到同一字典时触发 `TypeError`。没有创建 dry 结果或三例对照目录，实际
求解、模型响应、物理、WAM 和 GPU 均为 0；失败收据保留于
`outputs/fast_seed_search_dry_failure_20261116.json`，SHA-256 为
`c94867190501c00eada103570ef0572a0950823600d0e3b9c4c863dbedb5d29b`。

修复仅将比较配置字段命名为 `comparison_config_ref`，历史 profile 的完整
输入收据保存在 `profile_inputs` 下，并增加对真实保存状态的元数据回归。
算法、目标、案例、内环误差假设、求解和审核预算均不变。新源码使用独占的
`outputs/qa/fast_seed_search_final_v2_20261116` 与
`outputs/fast_seed_search_dry_v2_20261116.json`；两者通过后才允许在原尚未
创建的 `outputs/fast_seed_search_20261116` 内执行一次三状态对照。首次失败
与原 QA 保留，不覆盖或自动重试；任何后续失败继续保留实际和未知开销。
