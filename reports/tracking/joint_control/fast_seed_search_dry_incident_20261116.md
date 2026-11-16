# seed-first 对照的零求解预检失败

2026-11-16。首次 dry 在历史 profile 和本次对照的 `config_ref` 合并时触发
`TypeError: dict() got multiple values for keyword argument 'config_ref'`，
进程退出 1。失败发生在结果发布前，没有创建 dry 或对照目录；没有启动
求解器或模型响应，也没有物理、WAM 或 GPU 任务。原 64 并发 QA 97 项通过，
但其合成用例没有覆盖两个真实配置身份的组合。

保留独立失败收据，补充真实输入回归并分离两个配置身份；后续使用新的 QA
和 dry 路径，范围与预算不变，见[修复登记](../../../plans/v0.0.1/tracking/joint_control/fast_seed_search_dry_repair_20261116.md)。
本记录不把首次失败追认为成功。

| 本仓库 `outputs/` 路径 | SHA-256 |
| --- | --- |
| `fast_seed_search_dry_failure_20261116.json` | `c94867190501c00eada103570ef0572a0950823600d0e3b9c4c863dbedb5d29b` |
| `qa/fast_seed_search_final_20261116/result.json` | `be544d73409dae9b1cd142ce795ae0762ebdfd54d42ae3d189bf3562c21115da` |
