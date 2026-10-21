# 030：来源隔离的同步 native 训练轨迹

在任何本项HDF5读取、仿真或新样本采集前登记。029已完整结束并独立通过：固定027的teacher在seed42改善55.06%、seed43反退51.42%，自由raw轨迹actual-clean均值反退1.60%。这支持扩大训练场景及噪声覆盖、直接学习自由上下文clean残差，尚未确定唯一根因。[029报告](../../reports/wam_free_trajectory_diagnostic_20261021.md)及所有旧False保持。本项只建立训练数据，不加载WAM、不fit、不评fresh或改S2A。

最终目标仍为统一冻结WAM适配候选＋固定S2A在原12fresh初态×seed42/43达到至少20/24、各任务至少4/8、600步末40端点原生连续成功。这里的native成功仅是训练数据资格，不计WAM成功，也不认证原raw8400。

## 固定来源与选择

沿用原三任务及三个native HDF5来源、原LIBERO/OSMesa环境。原fresh config SHA `a5d5f87fe28885f09112b18b20cd8ea79e18e0a37d16949abc958f03748d8d65`，其fixture SHA `6e279c4db07a06f7399f53eb298cb77b671305accfe5b520598f1ee1c8e134c6`；字节及12个保留ID/initial SHA不变。新训练source table仅绑定原身份，不运行原metadata audit（该入口会读取fresh native actions）。

每task固定10个native demo，共30条。先排除原fresh四demo/task、原fresh配置的全部historical excluded_demos_by_task、旧六validation整demo（task0:16/32，task1:4/27，task2:14/24）。这些排除是源身份规则，不依赖WAM结果。HDF5只读取非上述ID的候选initial/metadata，拒绝与任一fresh或历史fixture initial hash相同的别名；不打开fresh demo的actions/obs/states。各task剩余demo按numeric ID排序，用一个`np.random.default_rng(42)`按task顺序permutation，依次取首10个满足源身份/非别名者；选择时只看initial与形状，不读取task成功或动作内容，不按轨迹长短/质量重选。

在任何物理前冻结全30 roster、initial/XML/source SHA及stat、native actions shape、任务/工具合同和实际执行源码快照，并由独立metadata peer验证。每个选中demo必须有43至600个7维原native action；不满足保留数据准入False，不换demo。历史排除名单不会被改写，新训练集合以独立manifest增加到开发来源，原fresh仍封闭。选择后才允许读取这30条native actions。

## 同步采集合同

恢复选中XML/initial，不做预滚动，刷新OSC缓存；固定20Hz、256相机分辨率、原native机器人/控制器，不应用S2A或任务规划。按原顺序完整执行N条native action，成功也不提前停止；随后固定40步零arm＋最后一条native gripper command保持。每例总步数N+40≤640，全项≤19200真实物理步。每次step后用`current_observation`强制刷新，在同一物理端点保存：source EEF SE3、canonical TCP SE3、实际opening、sim time/步号、原生success、完整sim state hash、实际发送命令及controller state/contract。原HDF5 obs不当训练真值。

`T_tcp[k]=source_pose(actual_obs[k]) @ H`，H取现有tool_contract的source_from_tcp；记录实际矩阵及来源SHA，不另猜TCP offset。每个过渡标签由实际端点得到：

```text
Delta[k] = inverse(T_tcp[k]) @ T_tcp[k+1]
action128[k,:10] = [Delta.p, Delta.R[:,0], Delta.R[:,1], opening[k+1]]
state128[k,:10]  = [T_tcp.p, T_tcp.R[:,0], T_tcp.R[:,1], opening[k]]
```

位置单位m、旋转column-rot6d、局部右乘、时间间隔.05s，state/action/delta/controller command分别保存。实际N+40过渡均有效、其余118槽为0；最后端点没有下一过渡，其action_valid及action mask为False。保存完整N+41个实测state和对应有效mask，后续normalization31仅在模型数据阶段应用，不提前重复归一化。

相机常规observables关闭，仅在固定anchor当前端点用既有`render_current_cameras`即时渲染双相机，并按既有live翻转/cover-crop canvas合同保存raw RGB及960×640 PNG。同次render不得推进/恢复物理；绑定render前后sim time与state hash、camera角色/形状/dtype、当前measured state。**不采未来RGB、不读取HDF5图像**，future state/action只作后续离线监督，不进WAM生成接口。

为同时覆盖native运动和完成后保持，固定5个anchor：前4个`floor(j*(N-40)/3), j=0..3`，再加`N`（40步terminal hold起点）。5个必须不同、每个都有41实测端点/40有效标签。相较只读建议中的4个uniform anchor，本登记事前增加独立terminal anchor以覆盖稳定保持；不依结果选阶段。共150当前canvas/300 raw相机帧，0future RGB。本项新noise派生预算为0，后续模型训练另登记每anchor两独特noise、全部排除评测42/43。

## 验证和失败处理

必要CPU测试覆盖：TCP力臂/非交换旋转、局部过渡独立重构、当前state与下一opening时序、末行mask、五anchor完整窗口、源/initial别名排除及确定性选择、拒覆盖、真实脚本入口、CPU空CUDA/LD与owner预算。复用既有渲染/环境函数但新采集不能调用会提前break的旧native评测循环。

实际实现及全仓CPU QA通过后冻结新source closure；已有029的111源不得修改。先metadata独立准入，再一次固定30例采集；所有worker显式`CUDA_VISIBLE_DEVICES='' LD_LIBRARY_PATH='' PYTHONDONTWRITEBYTECODE=1`，仅CPU/OSMesa，模型进程0。owner保存实际Popen/birth/command/env、wait/reap/退出，拒重复case、已有输出和自动重试；不kill未知进程。可按task分三个owned CPU worker，每worker独立新环境/输出目录，模型GPU4–7也不使用。

独立CPU先核source/roster/initial排除、30例完整预算及正常退出，再验证原native command与实际发送命令、40 hold命令、时间/state/render绑定、直接物理site姿态与观测一致、全SE3/rot6d几何、local delta独立重构、每anchor标签窗口与当前图像来源。保留每例完整success时序；30例全部无运行异常且末40端点原生success全True才给`eligible_for_model_training_registration=True`。只根据native数据质量准入，不看WAM。原生质量失败仍跑完预定其余例并报告全部分母；源码/协议/数值错误停止后续，保留False。任何失败不替换demo/追加spare/缩小训练集合或修改阈值。

本项最多30环境初始化、19200物理步、150当前canvas，0WAM/denoise/fit/参数重建/fresh/真机。已有共享LIBERO和外部Cosmos/数据只读，新配置、代码、报告、数据和产物均在S2A。所有旧结果/原fresh文件原字节不变。

## 后续（本项不授权）

数据合格后再登记031：固定8400非EMA的显式C0 clean残差head，特征为train-only中心/全局尺度的hidden和C0，固定ridge.1、两轮latent on-policy aggregation（raw→W1轨迹→等权union W2），完整35实际sigma。训练noise按每anchor派生且排除42/43，唯一最终候选才做val和一次固定六条开发task；不再以teacher门槛替代任务实测。五anchor对应的模型预算需按150anchor重新登记，不能沿用旧120anchor预算启动。新模型成功不追认证原checkpoint，冻结S2A不看任务标签重规划。fresh验收仍需后续独立冻结协议。
