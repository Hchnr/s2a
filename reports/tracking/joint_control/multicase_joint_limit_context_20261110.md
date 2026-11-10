# 多开发案例首次关节越界的上下文

2026-11-10。对[12条开发轨迹](multicase_reference_geometry_20261110.md)逐项重核保存q、日志和原XML范围，原7条越界计数一致。每条取首次越界关节，并回溯它最后一次安全余量至少为 `0.05 rad` 的时刻；以下为该关节随后跌破缓冲到首次越界的控制周期数。这是离线回顾，不保证届时存在可行刹车动作。

| 开发案例 | 首次越界端点 | 关节 | 缓冲丢失至越界周期 | 越界端点速度 rad/s | 产生该端点的route |
| --- | ---: | ---: | ---: | ---: | --- |
| task0 s43 raw | 121 | 7 | 6 | -0.0392 | adaptive_margin_witness |
| task1 s43 raw | 103 | 7 | 3 | -0.1016 | preventive_fallback |
| task1 s43 terminal058 | 401 | 6 | 4 | -0.1031 | adaptive_margin_witness |
| task2 s42 raw | 391 | 7 | 4 | -0.0171 | preventive_fallback_retained_seed |
| task2 s42 terminal058 | 559 | 5 | 4 | 0.2613 | preventive_fallback |
| task2 s43 raw | 73 | 7 | 4 | -0.0409 | preventive_fallback |
| task2 s43 terminal058 | 143 | 7 | 16 | 0.0408 | preventive_fallback |

7条首次越界均为上限，其中5条为第7关节、各1条为第5和第6关节。产生越界端点的上一条命令有4条来自 `preventive_fallback`、1条来自 `preventive_fallback_retained_seed`、2条来自 `adaptive_margin_witness`；越界端点处5条速度已转向关节内部。这说明仅凭route名称或该端点的速度方向不能判断真实安全，更不能证明某个控制动作能在此前及时制动。

每案例仅重建最后缓冲外端点、随后首个缓冲内端点、首次越界前一端点和首次越界端点，共28个离散状态。在首次越界端点，1/7检测到机器人或夹爪与场景几何接触；4/7在所抽取的4个端点均未检测到此类接触。其余两条在较早所抽端点有接触、首次越界端点没有。未采样的端点及周期内子步仍可能有接触，以上不能排除接触影响或建立因果关系。

例如task0 s43 raw在缓冲丢失及越界附近，保存的夹爪hand与桌面/非目标cream cheese有接触；task1 s43 raw的四个所抽端点均未观察到机器人—场景接触。两种情形不应统一归因。此前v5的固定 `tau=0.1s` 制动已在40周期成对物理对照中失守且最深越界更差，见[历史记录](tracking_wam_integration_20261027.md#40周期制动对照与独立回放完整关闭不采用)；不能简单重复或称其已解决安全问题。下一步须先确认多案例原生命令回放和控制器状态重建，再针对已有失败设计不同的、可因果执行的提前保护，并用相同固定已生成参考做有界物理对照；不读取未来实测状态或改变原验收门槛。

本次7个完整模型、28次 `mj_forward`、0时间推进/WAM/GPU/fresh。34/34项32进程CPU专项通过；owner会话12092实际wait退出0，源码及全部使用的资产/输入前后指纹一致。完整结果 `outputs/multicase_joint_limit_context_20261110/result.json` SHA256 `50861e33b9fb5727cc174ea04b79dd38a6b02206d6f181b0f56cecb6b5ea6d10`；聚合与owner观察 `outputs/multicase_joint_limit_context_analysis_20261110/summary.json` SHA256 `ab8c39d72acce356c611b5f1548d23beea5d94a120d9bbf9769b9fe250be5398`；QA SHA256 `c17827dc62cdad0e418843d86821c7e256040ea25e9d84b82a5c67b04b2aa436`。这些诊断不授予控制或任务资格。
