# 迁移运行时的保存命令回归（CPU）

本阶段只验证当前运行时能否复现已执行并独立核验的 H8 分支。它不重新求解动作，不执行新的 H12 候选，不取得控制、任务或历史二进制等价资格。GPU 继续禁用；原 18/12/24 门槛及完整 H12 witness 前置不变。

固定输入见 `configs/native_runtime_replay_20261029.json`：v3 bootstrap 及其实际 wait/reap0 owner、原 H8 producer/checker 及两者已关闭 owner。入口从这些已绑定结果读取原 manifest、commands、states、periods、chunk46 和 TCP tool。唯一运行时变化仍为已登记的路径迁移，使用 v3 保存的 `work/models/02_record_xml.xml`，不探测旧 `/share/project` 或历史 `/Users` 路径。原初态、seed42/trial0、任务8、固定 OSC_POSE、20Hz/2ms 和模型参数保持。

## 唯一执行

新输出为 `outputs/native_runtime_replay_20261029`，不覆盖、不自动重试。先从原初态构建，同 v3 使用一次底层 hard reset、映射后的 record XML 和原79维状态；检查原初态字节、控制合同及软件 OSMesa。最多3模型、3contexts。不能直接恢复 actual368：控制器 cache、夹爪 motor target 和接触求解历史需要由原命令自然形成。

随后只发送保存的377条命令（0..376），到 endpoint377 停止。预算377个控制周期、9425个2ms积分步、378个端点；368..376共有225个内环 q/qvel/time 样本。该预算包含368条前缀，不能仅记9周期；世界模型、新选择动作及 fresh 都为0。没有第二套重新执行 producer/checker 的物理预算，此运行直接比较已有、相互独立且端点 archive SHA 一致的两份历史证据。前台子进程由调用方实际等待并回收，超时1800秒。

保持原独立 checker 的观测顺序：初次 current_observation，每个端点 endpoint（含 forward），0/16/56各一次双相机 anchor，每8步双相机渲染，每条保存命令后 current_observation。所有 render 都走 CPU OSMesa，CUDA_VISIBLE_DEVICES/LD_LIBRARY_PATH 为空、五项线程变量为1、禁止 bytecode，新增缓存与产物只在独占输出内。无需世界模型服务。

比较378个完整状态、225个内环 q/qvel/time、逐周期历史误差更新、实际 TCP 误差、关节裕量、task/grasp 谓词和原顺序。历史数值比较为原 atol1e-10/rtol0，不声明逐字节等价；actual368 完整状态及 q/v 的逐字节一致另报。controller 全字段与 v3 合同一致，initial_joint 与原 braking numeric 登记精确一致；actual368 仅读当前 M/J/q/v/旋转goal并比较原 period0 dynamics，不额外 forward 或刷新 controller。保存 controller goal/cache、motor cache 和 completed367 命令，供后续候选起点核对。未来保存状态仅作为回归期望值，不恢复到仿真，也不决定新动作。实际 pose caps 仍严格 .01m/.05rad、内环关节裕量 .05rad，比较容差不扩张这些门槛。

第一次比较或实际门槛失败即停止，不继续派发。分别保存 env.step、mj_step 的 attempted/completed，以及失败时前后仿真时间和部分内环样本；即使动作在25内步中途异常，也保留已积分成本。初始化和清理阶段禁止控制/积分，mj_step2 始终禁止。异常原因和 cleanup 结果分别记录，保留原异常。

## 准入与结论

入口默认仅 metadata admission，不导入仿真；`--execute` 需要绑定新5个源码/测试/计划/配置的 native Python3.11.15 CPU QA 收据。组件测试验证真实控制循环相关的顺序、预算、提前停止、失败成本、状态不被未来覆盖，以及 metadata 与 QA 的绑定。Ruff/format 和真实 metadata admission 通过后，交 root review；本次实现不自动启动真实重放。

实际运行前后核验 v3 所有已消费来源、保存命令分支输入及新源码；另外7份当前读取/渲染依赖逐SHA绑定原 H8 producer owner 的已登记源码，不用当前自报身份替代历史绑定。记录实际加载库/模块当前身份，实际 owner 退出后再判闭合。成功只表示此377周期分支在固定数值容差内复现，不能补造未记录的历史 BDDL/系统库二进制身份，也不意味着未知动作、其他初态或任务均等价。新的 H12 物理候选仍须完整模型 witness、全部独立响应复算及单独登记通过。
