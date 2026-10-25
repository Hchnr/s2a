# 048：W2到生成诊断的元数据与导入组件准备

2026-10-24 10:47（北京时间）：**已实现只读W2资格核验和导入隔离组件，相关7模块135项CPU检查通过；048生产桥、角色收据、独立准入和执行入口尚未完成。** W2仍未拟合，STATE/开发/fresh均未启动。本轮没有新增模型任务；原D1同一三层PID/birth仍存活，已闭合156/300请求，已完成请求至少5460次denoise，进行中请求的精确总数未断言，519源/快照不变。

继续[036总目标](../plans/v0.0.1/036_project_goal_20261023.md)与[048接续边界](../plans/v0.0.1/048_clean_training_state_handoff_20261024.md)。最终仍为同一候选在原预留24条稳定≥20/24、每任务≥4/8；软件/元数据检查不代替状态跟踪或任务成功。

## 实际接入问题与组件

原041 STATE owner的`planned_worker`和模型sampler的`admit_sampling`均导入`run_wam_clean_training.prior`查旧W2 owner。只修CPU owner不能解决模型进程仍读旧目录的问题。原STATE独立检查器同时要求真实解释器和原sampler命令路径，不能改入口后隐瞒来源，或将新durable收据伪造成旧schema。

[只读W2核验器:45](../src/s2a/clean_training_w2_handoff.py#L45)直接核047 proposal/admitted/独立metadata、完整supervisor、两个角色实际wait/reap/PID birth与worker、原840000行W2和独立QR、150标签、固定1290列语义、replay一致性、原数值容差和有限参数。只读JSON和源指纹，不能单独授权模型。当前实际调用对“没有完整W2”正确拒绝，没有导入NumPy/torch/MuJoCo或数学producer。

[子进程环境构造:32](../src/s2a/clean_training_import_bridge.py#L32)使用显式`PYTHONSAFEPATH=1`与`PYTHONPATH=桥目录:原bundle:原其余路径`，原模型命令和数值源码不变。[原delegate加载:56](../src/s2a/clean_training_import_bridge.py#L56)从固定SHA实际文件加载私有模块并保留真实`__file__`。[CPU有限作用域:80](../src/s2a/clean_training_import_bridge.py#L80)拒绝已有缓存别名和嵌套调用，退出恢复sys.path、环境及新建别名/私有模块；冻结传入refs的副本，防止调用方同时改文件与SHA掩盖漂移。这些只是导入工具，不是048执行许可。

## 已执行的检查

- W2完整/失败/部分/未释放收据、语义列与参数几何、源码proof恢复等元数据分支使用明确的合成完整结果测试；未冒充真实W2训练。
- 真实CPU进程分别用主解释器与实际模型解释器执行**原冻结sampler `--help`**。两解释器中safe-path开启时解析到桥模块，关闭时旧脚本目录优先；开启时原sampler文件及`run`函数源文件仍为原路径。测试桥的回调拒绝所有W2查找，没有调用采样或取得任何模型执行资格。两个解释器×两个flag共4个真实help进程。
- 导入作用域正常/异常恢复、缓存拒绝、别名替换拒绝、调用方修改源SHA无法掩盖文件变化均通过；新增最后一项曾因测试遗漏`Path`导入失败，原v3日志保留，补齐测试导入后复验。没有模型重试，也没有改旧采样源码。
- 新组件、047相关接口和原STATE采样/owner/独立检查器回归共135通过；491份主仓库源码Ruff/format通过。未执行实际STATE数值比较、生成或物理任务。

不可执行[policy](../configs/wam_clean_training_state_handoff_policy_20261024.json)明确`execution_ready=False/execution_authorized=False`，只固定未来STATE的99请求/3465denoise、唯一允许UUID、0fit/QR/physics/fresh、不自动重试；未填造未来W2 SHA。完整047真实结果后，还须完成生产桥、角色来源收据和独立048执行envelope，再跑实际dry并核GPU/端口。

## 当前证据与后续

额外证据根为`outputs/clean_training_state_handoff_preparation_20261024/`：

| 证据 | 路径 / SHA256 |
| --- | --- |
| 135项组合检查 | `qa/pytest_components_v4.log` |
| 组件QA、被测源SHA、真实help范围 | `qa/components.json` / `aea853afc012c93f648e0cea72e6bff5a3dc5818c0156b1e63f788e4c3a4a417` |
| 10:47 D1活体及519源核查、W2/STATE未执行 | `qa/live_d1_and_pending_w2.json` / `9852d179ad8ba35f48f06a9442f7217e68fce52ccbe681e64bc8b94d55a3846a` |
| 原测试导入失败 | `qa/pytest_components_v3.log`；原134通过/1失败，不改记通过 |

D1仍为supervisor2010452/birth574805981→owner2010453/birth574806125→model2010513/birth574806584。只观察同一实际进程，不因工具观察超时而重启。[046](wam_clean_d1_continuation_20261024.md)前2请求/70节点的原独立重算资格仍只覆盖那部分；完整D1收尾尚未取得。

顺序保持：D1完整原审查和正常收尾→[047](wam_clean_w2_preparation_20261024.md)一次实际W2/全量QR→048独立准入后的原99请求及全量审查→原固定开发任务门→统一候选与最终24条。`state_gate_passed`继续按原协议作诊断，不能新设必须True门；完整数值合同合格也不等同于开发或留出任务收益。
