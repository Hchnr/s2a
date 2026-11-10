# 第三个任务的早期关节预警前缀复现

2026-11-11。按[预登记方案](../../../plans/v0.0.1/tracking/joint_control/task2_warning_prefix_20261111.md)，
在CPU/OSMesa中从原初态分别重放task2/s43/raw和task2/s43/terminal058到原审计的
首次`velocity_0p2s`安全预警端点34和82。命令、固定机器人/控制器、模型XML及已
迁移资产来自此前已审核的两条开发记录，不生成新动作、参考或初态。

| 开发案例 | 原命令周期 | 端点状态逐字节一致 | 当前最小关节余量 rad | 前一周期25内步最小余量 rad | 第7关节向外速度 rad/s | 0.2 s外推余量 rad |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| task2 s43 raw | 34 | 35/35 | 0.363252218 | 0.363252218 | 1.695780849 | 0.024096048 |
| task2 s43 terminal058 | 82 | 83/83 | 0.273016214 | 0.273016214 | 1.361375445 | 0.000741125 |

每个端点的完整sim state与记录字节一致；实际控制器缓存、位姿、开度和原生标签
按原1e-10绝对容差核对，最大状态与缓存数值差均为0。两个末周期各25个2 ms样本
均未越界或跌破0.05 rad缓冲。外推是当前q/qvel的固定预警描述，不是对未来轨迹或
制动可行性的保证；此前原闭环后来越界不代表新控制分支一定如此。

本次为task 6、seed43增加[独立绑定](../../../src/s2a/research/native_runtime/task2_prefix_binding.py)，
在私有全局命名空间使用原前缀runtime代码并替换唯一的task/seed准入函数；旧模块
和其task 8/9准入未修改。其余环境、控制器、初态、资产、端点、费用和关闭检查均
由原runtime执行。原XML的99个资产声明与96个已迁移资产映射逐项绑定；本次仅证明
两个已研究前缀可复现，不赋予task2候选控制、完整任务或新初态资格。

两个CPU worker均实际wait/reap、exit0且未发信号，监督进程实际等待并回收主执行器；
源码和输入前后一致。共116控制周期、2900内步、6模型、6软件context，0新选择
动作、WAM、GPU、拟合或final fresh。26项32并发CPU专项及零模型dry通过；本次未
重跑全量pytest。下一步须另行登记相同已生成参考上的实际安全/跟踪反馈及独立动作
重放，不能把原命令复现当作安全干预收益。原18正确参考、12开发、24最终留出及
最终稳定门槛保持。

| 仓库内证据 | SHA256 |
| --- | --- |
| `outputs/task2_warning_prefix_20261111/result.json` | `a1d84370edafbc97d072c59370c8f33a06c17600ed1baf843b1286e1449d0a27` |
| `outputs/task2_warning_prefix_dry_20261111/result.json` | `2081d30c708b68a8cf4e459fd55836a377b63bf68474f4519172997a469e8c23` |
| `outputs/qa/task2_warning_prefix_20261111/result.json` | `3fc07389db62ec1277aa7c2c11b66e6ee34528d976ed5acdacc7fb04e3a958ff` |
| `outputs/task2_warning_prefix_supervisor_20261111/owner_exit.json` | `4de31c217caae9940fbd9e6326c877e3c20757ff8168d9bfc97e0ce94dfac349` |
