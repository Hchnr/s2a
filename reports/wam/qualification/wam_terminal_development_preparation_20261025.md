# 061闭环开发接入准备

060参考验证完整通过后开始本阶段；当前尚未启动开发模型或物理仿真。新计划固定原12案例、900请求、31500去噪节点、7200物理步及候选稳定≥5/6、各任务≥1/2门槛，fresh保持关闭。

`src/s2a/terminal_development_gate.py`新增独立061准备准入：只接受固定060完整owner/数值/score谱系、全部原state检查、9项component保护和至少5/6 episode、三角色真实wait0/reap/noSignal及允许UUID的实际GPU释放。三角色launch路径分别绑定model/independent/scoring，不能重复同一收据凑数量。再通过原058数值proof与参数同一性检查；没有旧STATE gate=False例外。输出只允许准备，execution_authorized/fresh_authorized均False。

实际执行 `check_wam_terminal_development_gate.py`，通过原060/058不可变JSON，消费输入前后SHA不变；输出 `outputs/terminal_development_preparation_20261025/state_precondition.json`。不解码训练或validation future数组，不加载模型。

`src/s2a/terminal_development_policy.py`提取原串行预算/当前输入账本，固定raw/terminal058案例顺序，75请求/案例、总900请求。只将候选名称与明确的task/template元数据适配到061；保留case不可跳过、预算不可截断、pending不可重入、当前PNG受控目录、task/seed/35步/H40/20Hz静态合同、首raw/candidate PNG配对和abort/finish完整性约束。

实际 `prepare_wam_terminal_development_policy.py`从060 current-only元数据为每任务核对唯一zero模板及两个demo/八个当前锚点的同一task文本，生成三项明确development_tasks；注册模板只保留request，不把历史expected-response/action引用带入策略清单。实例化实际策略并核对12案例顺序通过，消费输入前后SHA不变；输出同目录policy_metadata.json，0模型/请求/物理/fresh。

相关26项CPU测试通过，包含完整900次reserve/complete账本、案例跳过/混用/截断、future字段/嵌套future/任务/seed/图像路径/时间步合同拒绝、首帧配对，以及门槛False、缺失检查、坏退出/未reap、重复角色、GPU占用、参数错绑拒绝。Ruff及格式通过。

剩余工作是新HTTP服务、物理client、完整31500节点及7200端点独立审查和owner；沿用已修复安全tool_contract/template及实际sim环境，全部12case的真实输入/CLI预检与源码冻结/独立注册完成后才执行。不得直接启动绑定旧STATE/W2的051入口。当前0新开发模型、0物理、0fresh，总目标未完成。

## HTTP服务接入

新增 `scripts/serve_wam_terminal_development.py`：保留原串行/loopback HTTP、真实模型和scheduler调用、31500边界预算、完整capture/条件/初噪、首raw/candidate初始条件配对、参数及source保护、失败/清理逻辑；接入058策略、061 current-only账本与已修复的双头身份桥。0–33选择052，34选择058，不根据请求内容选择头。沿用原`/info`、`/generate`及provenance JSON协议，候选身份字段明确记录新proof与060评分来源。

完整run函数AST对照确认改动仅限明确的候选/来源身份及dry接缝；实际读取HTTP非法请求会poison服务、实际CPU子进程的清理失败保留False并exit1的检查通过。新增dry模式先完成准入及实际MRO身份检查，再在任何模型runtime导入/目录创建前返回；测试确认其不导入torch/Cosmos、不解码数组、不初始化模型。完整061真实registration尚未生成，故尚未进行注册后的真实CLI dry，不把单元测试称作执行资格。

`src/s2a/terminal_development_execution.py`要求新的执行registration、独立12案例metadata、明确metadata verifier源码、四角色source绑定、完整source/input/snapshot、163 checkpoint stat及060实际准入。拒绝metadata-only、缺角色、错source、改权重、错候选及fresh授权。port31363，仅模型UUID登记许可不变。

服务、策略、state gate和执行准入共59项CPU测试通过，Ruff/格式通过。实际模型解释器已成功导入新服务，使用实际060/058 proof计算双头策略指纹1a1e333d0328270aee6c875b65bdff52b55e51815f2990be04ad51111dadb427，并用实际current-only policy_metadata实例化12案例账本。该检查0模型初始化/0物理步；CLI help解析通过。

下一步接入物理client、原固定控制配置和已迁移安全template，完成全部12案例在实际sim解释器中的输入检查，再补独立数值/物理审查和owner。当前仍0新开发模型/物理/fresh。

## 控制配置与物理client接入

`src/s2a/terminal_development_control.py`绑定原raw8配置SHA51f64aa81a53f89cd91f8cf9925efe3e3c9a3a8de13a1856b054a37693dc979b、原fixture SHA及051合同迁移记录SHA35f43c12d90c098aa2625bae5c994bf336d7488ce996a829c642f39a80ddf8a7。核对新旧template仅tool_contract字段路径变化且安全副本保留历史SHA，不改控制参数；明确重定位新输出、模板、实际Cosmos路径和服务port31363。legacy gpu4只保留为原CPU执行器的元数据标签，模型进程仍仅按允许UUID启动，物理执行CUDA/LD均为空。

实际运行 `check_wam_terminal_development_inputs.py`，解释器为仓库outputs/libero_env311/bin/python（Python3.11），对全部12案例调用原stability_position_simulation.execution_inputs，每例37项输入指纹全部通过。输出input_preflight_py311/result.json及12份配置；实际0模型调用、0物理步、0数组解码，未创建仿真环境。原控制、fixture、工具合同和已修复template均可定位，消费输入前后SHA不变。

`scripts/run_wam_terminal_development_case.py`已接入新执行/060门槛、控制配置及候选案例身份。完整execute与bounded_http AST比较确认原物理函数、固定600步/75请求、严格CPU环境、仅本地/info和/generate、300秒HTTP上限、current-image/seed/无future字段校验、异常记录及finally结果保留不变；仅身份/来源及dry接缝变化。dry在任何输出目录、cv2 preload或物理执行前返回。实际sim解释器CLI help已通过；完整registration尚未生成，未称为真实执行CLI dry通过。

控制、client及此前service/gate/policy组合共70项CPU检查通过，相关Ruff/格式通过。新检查包括全部12案例其余控制字段逐项不变、template动力学字段篡改拒绝、本地传输cap/因果性、dry不产生物理输出，以及完整client执行体保护。无新物理或模型执行。

剩余是新owner及完整31500节点/7200端点独立审查、最终metadata/source/snapshot/registration与全部真实CLI dry；这些完成后才启动固定开发对照。当前尚无新闭环任务稳定率或fresh结果，总目标未完成。

## 监督与独立审查接入

新增run_wam_terminal_development.py，持有一个private模型Popen、12个顺序物理client及最终独立审查；原ready/PID birth/环境、HTTP绑定、剩余deadline、异常abort、wait/reap/GPU释放逻辑继续保留。接入durable detached launcher，dry-run与launch互斥；非dry执行必须验证detached来源。模型协调端口31364与HTTP端口31363分离，启动前检查两端口、fsync及至少10GiB可用空间，模型仍仅使用允许UUID。

新增verify_wam_terminal_development.py及单独的JSON谱系准入，独立读取061注册、060通过门槛与三角色正常退出、058候选同一性和控制迁移。没有导入生产owner/service/物理client或调用训练求解。保留原完整物理回放、600步/末40步outcome、12案例score及actual wait/reap规则；六个完整函数AST对照确认只改候选名称。数值helper复制原完整audit_points，唯一接缝为实际节点选择独立的052/058参数元组。

同时发现旧数值参数解析器固定ridge=0.1；新独立解析明确要求058既定0.1/1290，其他schema/布局/有限性/scale保护保持，不以放宽容差替代。实际Python3.11使用该新helper重放060已合格首候选35节点通过，changed_active_values=13322，0新模型/物理步。

最新完整组合76项CPU测试通过，相关Ruff及格式通过；覆盖监督dry的固定12/900命令与UUID、双端口、原稳定评分/收据函数AST、新参数解析及完整数值body保护，并继续运行已有服务/控制/client/gate检查。owner和verifier CLI help解析通过；没有把help或synthetic dry称作注册后的真实CLI资格。

仍待最后独立metadata实现、完整source/snapshot/registration及实际模型/仿真解释器的全部CLI dry。当前尚未启动061模型服务、物理client或fresh，总目标未完成。
