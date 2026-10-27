# 226 初始化step1语义核实后的实例探针

225保护在LIBERO初始化的mj_step1调用立即抛错，未产生实例字段或积分，原failure和来源保持。已读取MuJoCo官方2.3.7 engine_forward.c（blob61906d629a8d3e071a922977a4afc86bd0126a3d）：mj_step1执行积分前派生量/传感器和可选控制回调，mj_step2才执行积分。来源快照在outputs/controller_probe_step_semantics_20261026，链接https://github.com/google-deepmind/mujoco/blob/2.3.7/src/engine/engine_forward.c#L881。

新版本仍禁止mj_step和mj_step2。对初始化需要的mj_step1，逐次保存并核对time/qpos/qvel/act及ctrl完全不变后才接受；任何改变或积分入口调用立即失败。记录所有step1检查次数、零积分尝试、实际2.3.7版本及绑定/运行库SHA，其余225三任务、恢复端点、实际uncoupling/固定delta字段、M/J和状态不变核对不变。新namespace actual_controller_coupling_probe_checked_20261026，不修改失败版本。

此为语义准确的零积分实例重建，不是放宽物理成功门槛；不执行控制action，不据此推断历史进程字段已经记录或闭环成功。
