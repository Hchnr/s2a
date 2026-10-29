# WAM 下一阶段稳定性：时序、积分与校正可行性只读审查

本轮没有发现新的、已证实的训练—推理或 state 解码错接。局部 delta、列式 rot6d、20Hz 端点时间、TCP 转换和实测锚点与已有标签回算证据一致。尚未闭合的是上游发布图像与物理 pose 的同步来源，不能把它改写成已证实的固定 offset。当前更直接、可证伪的下一步是：完成新夹爪对照后，在同一执行器上资格验证较短重查询间隔，再评价真实 WAM。低容量参考校正可以作为后续独立模型假设，现有六个开发 episode 不足以证明通用校正。

本审查只读源码和已保存结果；新增模型请求、物理步和拟合模型均为 0。005 活跃源码、配置、原始产物以及 fresh holdout 均未改动或打开。

## 已核对的训练与执行约定

| 项目 | 源码与已有数值证据 | 判断 |
| --- | --- | --- |
| delta 的坐标与积分 | [geometry.py:98](../src/s2a/geometry.py#L98)、[Action128 解码](../src/s2a/action_reference.py#L8)；外部 `pose_utils.py:393–421` 定义 `inverse(T[i]) @ T[i+1]` 和右乘重建 | `T[i+1]=T[i] @ delta[i]`。translation 在前一 canonical TCP 轴中，不能当世界 xyz 相加，也不是速度乘 dt。 |
| 旋转编码 | [geometry.py:14](../src/s2a/geometry.py#L14)先对两列做 Gram–Schmidt，再取叉积 | 是前两列拼接的 rot6d，不是两行、Euler 或 angular velocity。原标签回算最大位置差 `1.13283e-7m`、SO(3) `6.71441e-8rad`。 |
| 第 0 个 delta 的端点 | [标签审计:41–53](../scripts/audit_wam_state_labels.py#L41)以 frame0 为锚点，action0 对应 frame1；[发布/时间审计](wam_tcp_time_alignment_20261020.md)确认 slot9 是下一观测 opening | 动作时间戳 0 表示该转移的起点，执行后比较 .05s 端点；目前没有证据支持整体平移一行。 |
| 训练和 serving 时间 | 外部 `unified_action128.py:632–696` 使用发布 timestamp、action fps、vision[0]；`action_serving.py:194–205`同样构造 action 时间及 vision[0] | 本次 LIBERO 为 20Hz；H40=2s。在线 [control_timestep 检查](../src/s2a/stability_position_simulation.py#L186)与每步实际 .05s 相符。 |
| TCP 与 controller source | [source_pose](../src/s2a/libero_simulation.py#L58)读取原生位置/四元数；[执行器](../src/s2a/stability_position_simulation.py#L236)先乘 `source_from_tcp`，目标再乘其逆 | 不是把旧发布 flange 直接作为 TCP。[专项审计](wam_tcp_time_alignment_20261020.md)完整矩阵链误差 `2.22e-16`；158 帧 R2 对同源 R1 TCP 误差仅浮点量级。 |
| 在线锚点与物理暂停 | [执行器:244–282](../src/s2a/stability_position_simulation.py#L244)固定查询时 measured TCP，解码一次，并核查询前后完整保存 state/time 不变 | 每步不重新锚定整条轨迹。后续跟踪误差不应通过反复移动参考坐标系掩盖。 |
| state 条件 | 外部 `unified_action128.py:474–475,688–696`及`action_serving.py:283–285`均要求 no-state、首帧图像条件 | WAM 没收到显式 measured pose；实测 pose 由执行器用于解码和控制。这个设计限制不是新增的 serving 漏字段。 |

训练数值专项还核过三任务×三噪声强度：token、条件 latent、归一化 action 和有效 velocity 一致。保存训练 backend 与 SDPA 的差异另列；原服务无效槽加噪已确认并由本地 mask wrapper 修正，不能再次当作新发现。上述专项不等于重放历史消费 batch，详见[训练/服务报告](wam_checkpoint_acceptance_20261019v2.md)。

外部源码均只读，具体路径及本轮读取字节 SHA 见下方 source manifest；没有从 Cosmos 工作目录运行或引入训练 dataloader。

## observed anchor 与 dataset anchor 仍有哪些边界

离线 state 误差以同一发布 episode 的当前 state 积分，与未来 state 比较；在线以当前物理实测 TCP 积分。两者使用相同 canonical TCP 几何定义，但不能由此自动证明发布图像中的机器人姿态和 state 标签物理同步。模型只有图像条件，若上游图像与 pose 本来存在误差，实际锚点与模型从图像隐含推断的锚点可能有分布差异；目前没有量化出可用于修正的恒定变换。

[图像/pose 专项](wam_video_pose_timing_20261020.md)仍明确 `physical_image_pose_timing="unresolved"`。同源发布 R2 与 R1 对齐，但同 index 原生 HDF5 obs 与发布 state 最大差约 13.686mm/.105933rad；±1 索引也未得到精确对应。三个静态恢复点支持 HDF 动作后观测语义，但没有闭合 NVIDIA 原始生成器、HDF 版本与发布流同步合同。已经认证发布版本和复制字节身份，不能将剩余差异归于本地重编码，也不能通过挑最小像素差来确定真 offset。

因此，后续若补来源证据，应绑定精确上游生成代码、HDF SHA、像素与传感器读取时刻，并在多 episode 上用一个事先固定规则核对。没有这些证据时，不修改当前 anchor、工具矩阵或标签；不把一个示范的误差拟合成全局“坐标修复”。

## 新的离线反事实：旋转误差是否解释大部分位置漂移

对已有四个 checkpoint 的 guidance1.0、task-only、H40 开发预测做独立 R/t 递推，192 请求、6 episode、22 个不同锚点，完全绕过 S2A。保留生成的全部局部 translation，只分别替换积分时使用的前姿态：真实未来前姿态，或始终使用当前锚点姿态。前者是 **不可在线取得的特权反事实**，两者都不是新的任务评测或可部署执行器。

| checkpoint | 原始 H40 位置 RMSE mm | 换真实前姿态 mm | 固定锚点姿态 mm |
| --- | ---: | ---: | ---: |
| 8400 | 85.521 | 84.734 | 84.864 |
| 8600 | 103.871 | 100.773 | 101.959 |
| 8800 | 82.414 | 82.045 | 80.182 |
| 15400 | 89.913 | 89.732 | 89.228 |

这支持一个窄判断：在这组已保存开发窗口中，消除积分所用的累计姿态误差仍留下大部分位置误差，不能指望只缩小 rotation 就消除整体 translation 偏差。它没有证明哪条示范是唯一正确路径，也不能把三个 RMSE 相减解释成可加的因果比例。真实旋转仍可能影响抓取、碰撞及后续闭环观测。

原始逐 endpoint 位置/SO(3) 误差独立复现至 `1e-12`；391 个实际输入 SHA 绑定于 [反事实 JSON](../outputs/wam_stability_20261021/next_stability_review/rotation_translation_coupling.json)。重复 phase 的相同 `(episode,anchor,seed)` 预测逐元素相同：每 checkpoint 48 请求只有 44 个独特锚点-seed，独立 episode 仍只有 6。按锚点去重后的 H40 原始位置误差分别 88.660/107.418/84.907/92.305mm，排名不变；该去重也必须用于后续训练与统计。

## maxobs 从 80 缩到 8/16 的具体含义与风险

[查询条件](../src/s2a/stability_position_simulation.py#L238)是“当前参考消费完，或距查询达到 maxobs”，不是每 maxobs 才允许查询。无 governor、H40/执行40时原本每40步查询；maxobs8/16会每8/16步替换参考，600步分别75/38请求，仍每次生成H40、从当前实测pose解码，并丢弃未执行后缀。它不是改成H8/H16生成；推理物理暂停规则和600步预算应保持不变。

recovering governor 还会在新chunk前加入“上一已准入参考→当前实测anchor”的桥。[set_reference](../src/s2a/reference_governor.py#L74)重置虚拟时钟，桥占物理时间、不占模型时间；最小桥时长为 `max(位置差/.2m/s, SO3差/1rad/s)`。v2桥保持上一desired opening，并只在米与弧度两项误差都不增且至少一项下降时允许额外恢复进度，[实现](../src/s2a/reference_governor_v2.py#L31)未使用任务谓词。

缩短间隔可能减少长时预测漂移、较早利用新图像；也可能不断重置桥，尚未消费生成前缀就重新查询。它未必重现v1的严格自锁，但可能造成模型进度饥饿。低 opening 事件在未消费后缀中反复被丢弃，也是与当前因果夹爪有关的风险；全序列有关闭预测不代表已经执行关闭。

新增只读提取覆盖6条旧v2夹爪、raw-alpha1、recovering WAM完整轨迹，以及004中3条v3 recovering oracle。对旧maxobs80轨迹中56个具有完整首8/首16的chunk：

| 已记录前缀 | 桥接端点/前缀端点 | 前缀末模型时间仍0 | 生成opening≤.60事件仍在已消费前缀之后 |
| --- | ---: | ---: | ---: |
| 首8步 | 125/448（27.9%） | 2/56 | 15/56 |
| 首16步 | 135/896（15.1%） | 1/56 | 10/56 |

004的v3 oracle三场景各15chunk，首8/16没有模型时间为0的完整前缀，桥通常短于一个周期；这说明正确参考附近桥影响较小，不能外推WAM。以上是旧协议真实前缀，**不是**新8/16闭环的桥比例、预测、成功率或一定会自锁的证据。重新查询会改变之后的图像、输出和物理状态，不能通过离线截断重建该反事实。

建议先固定8/16两种候选，在相同执行器和三旧场景上做CPU oracle资格，完整保留源SHA、原生电机/开度合同、600步、末40稳定及10mm/.05rad门槛。随后若进入WAM，对每个固定checkpoint保留position_hold无gov与recovering_position_hold两个分支，以区分短重查询本身与桥接影响；raw alpha1、H40、seed、600步及旧max80参照保持一致。16给桥更多预算，但不预设其优于8。任何“按已消费模型时间查询”、不替换的预查询或不同桥策略，均属于新的执行算法，需要另行资格，不应伪装成仅改maxobs。

除原生成功率外，必须报告每query模型时间消费量、桥占比、连续零进度长度、首次有效生成端点到达时刻、已生成/实际准入低opening事件、全生成/已执行参考范围及TCP/SO(3)跟踪。任务predicate仅用于计分；不能把成功瞬间用作在线停臂、冻结开度或挑路径规则。旧短前缀WAM成绩弱，使用的夹爪映射不同，不足以证明修正后也一定失败或成功。

## 任务无关低容量参考校正：可做诊断，尚不应当作验收捷径

必须区分两个问题：学机器人响应补偿，标签是“执行后实际state与给定参考的差”；学WAM参考校正，标签是“同观测下生成state与有效未来state的差”。在错误参考上准确执行只能验证前者，不能给后者提供正确任务路径。当前较好的oracle资格也没有自动解决模型参考几何问题。

如后续开展，在线特征只允许当前measured TCP/开度/速度、历史实际命令，以及已生成delta/参考和相对预测时间。训练时可以用整条离线示范的未来state作监督，但不能把真实未来、task/object/contact、episode ID或示范时间索引放进在线特征。当前pose可能间接标识任务阶段，低容量也不能自动排除记忆单示范。

坐标必须先固定。例如设当前锚点为`T0`，预测为`That[j]`，则用锚点相对轨迹`Zhat[j]=inverse(T0)@That[j]`构造特征；残差以明确的局部SE(3)右乘应用，`Tcorr[j]=T0@Zhat[j]@Exp(eta[j])`，训练目标为同一约定下`Log(inverse(Zhat[j])@Ztruth[j])`。translation单位米、rotation单位弧度；不能对rot6d六分量直接加偏置或把世界残差当body残差。世界轴校正若另作对照须显式左乘及旋转变换，不混用两种模型。

起点应是identity、少数delta增益/有界漂移参数等可解释候选，并先看误差方向在整episode留出后是否一致；不直接上能重规划整段路径的网络。当前6episode/22anchors只能支持可行性诊断，不能把192请求或数千endpoint当独立样本。已有12episode虽有episode隔离，但全属训练catalog，且其state报告已经被查看；它们不是新的干净最终测试集。

最低限度的对照应包括：raw WAM；仅固定增益；同容量corrector接零/identity WAM的noWAM；把其他episode生成序列配给当前state的shuffledWAM；保持当前state和仅历史外推。shuffle先固定seed，既报告全局置换也报告离线按任务分组的置换，分组仅用于构造诊断对照，不作为在线输入。若noWAM/shuffle同样好，不能声称校正利用了当前WAM内容。

所有同episode锚点、重叠窗口、phase重复和采样seed必须在同一split。先在现有数据整episode交叉核对方向一致性，结果只算开发诊断；若需要新的正式可行性试验，可另注册例如每任务8episode、4train/2validation/2test，共24episode，固定4阶段×2seed共192查询的起始预算。这个数量仍不是通用充分性保证；需要新布局、阶段覆盖、独立初态与episode级置信区间，并保留跨任务留出对照。没有新增这样的数据或训练预算获执行。

若校正后成功，被验证对象是“冻结WAM＋冻结corrector＋冻结执行器”，不能倒推原始checkpoint已经正确。只有在noWAM/shuffle、整episode留出、冻结闭环上都保留增益，才有理由称其利用了WAM生成信息，而不是学到一个小型轨迹先验。

## 本轮证据与可执行顺序

先完成005及其独立审计；再用独立冻结协议资格验证短重查询，随后做相同初态的WAM对照。来源同步继续保持unresolved，不通过调anchor或拟合task偏置来消除失败。若剩余参考几何错误不能由受控执行修改解决，应把问题返回模型/数据质量，而不是无限追加S2A自由度。

| 新产物 | SHA256 |
| --- | --- |
| [36项不同源码/既有证据读取快照](../outputs/wam_stability_20261021/next_stability_review/source_review_manifest.json) | `3e2a9e18995aa2c2876e9805c1a69c529efd7d317f60268141a4157803f773cf` |
| [192预测旋转/平移反事实，391输入SHA](../outputs/wam_stability_20261021/next_stability_review/rotation_translation_coupling.json) | `f8a63bfc696efa1b23fb57c16cfaf5bb8242f9f2bdf94dfc5547211ca91fa9d2` |
| [九条既有完整轨迹的首8/16消费诊断，36输入SHA](../outputs/wam_stability_20261021/next_stability_review/requery_prefix_risk.json) | `5cef63937ce036695d8421edc2f032cde0d47b6f10a6ca544c065958f70812c2` |

已有的完整state证据与来源限制见[旧阶段同协议对照](wam_earlier_state_screen_20261021.md)、[直接state验收](wam_state_acceptance_20261019v2.md)及[图像/pose时刻核对](wam_video_pose_timing_20261020.md)。本轮未新增成功率，不从部分005结果选择或修改候选。
