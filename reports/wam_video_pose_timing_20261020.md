# WAM 视频与物理状态时刻核对（2026-10-20）

本报告包含已有图像比较与CPU/OSMesa静态状态恢复，尚未建立发布图像与保存物理状态的精确时刻对应。结论保留为 **`physical_image_pose_timing="unresolved"`**。stage1仅读已有图像；stage2的环境初始化单列、任务计分控制步为0。两阶段均未加载WAM或重新解码视频，未改变主评测及成功评分。

固定对象是 `task1_init2/demo_4 → R1 episode683 → R2 episode1139` 的158帧，两路名义对应为 `agentview_rgb → auxiliary_0`、`eye_in_hand_rgb → left_wrist`。原生128×128图像用 Pillow BILINEAR 放大至256×256，分别比较 identity、vertical_flip、horizontal_flip、both_flip；索引偏移定义为 `published_index - native_index`，候选为 `0,-1,+1,-2,+2`。矩阵覆盖 `2×4×5×158=6320` 项，其中6272项可比较、48项边缘越界；越界项保留为空值，没有当作零误差或失败。实现见 [audit_wam_video_pose_timing.py:181](../scripts/audit_wam_video_pose_timing.py#L181)，固定配置见 [config](../configs/wam_video_pose_timing_20261020.json)。

同 index（offset=0）的全158帧汇总如下。MAE/MSE均对所有 RGB 通道计算，像素范围0–255；表中为逐帧指标均值。

| 视角 | 原生图像方向假设 | 比较帧数 | MAE | MSE |
| --- | --- | ---: | ---: | ---: |
| auxiliary_0 | identity | 158 | 73.241311 | 6801.280619 |
| auxiliary_0 | both_flip | 158 | 7.262847 | 275.976759 |
| left_wrist | identity | 158 | 79.412169 | 8851.163343 |
| left_wrist | both_flip | 158 | 18.796825 | 1417.090977 |

[第73帧完整对照图](../outputs/wam_video_pose_timing_20261020/same_index_frame0073.png)与[第90帧完整对照图](../outputs/wam_video_pose_timing_20261020/same_index_frame0090.png)均同时展示发布图和四种原生方向假设。both_flip 后场景布局明显更接近，腕部及机器人仍有可见残差；6272项可比较记录中没有像素完全相同项。这支持继续检查图像方向与来源，不能证明历史变换链、角色对应或物理同步。

不能按最小像素误差选定真实偏移或方向。offset=0、±1、±2分别使用158、157、156帧，重叠支持集不同；例如腕部 both_flip 的 offset=+2 MAE为16.812274，低于同 index 的18.796825，仍不能据此认定真实偏移为+2。原生 HDF5 的 `opengl` 属性与当前保存像素不证明上游 HDF→R1→R2 的方向处理。原生128与发布256还可能涉及不同渲染来源、相机、缩放和有损编码；本阶段没有复现历史 render/codec 管线。两路均值也不能直接互相比作时间误差。

独立只读复核已确认：脚本及配置与 commit `bb43ea2` 一致；8项输入、6项输出的 SHA 全部匹配；配置中的 metadata snapshot 与 Parquet episode1139对应字段完全一致。另从原生 HDF5 与缓存 NPZ 独立重算全部6272项 MAE/MSE/PSNR，结果逐项一致，CSV、JSON及40行汇总一致，48项越界记录的指标均为空。PSNR按 `10*log10(255²/MSE)` 计算，MSE=0时用空值配合 `pixel_identical=true` 表示正无穷。原始证据与指纹见 [manifest.json](../outputs/wam_video_pose_timing_20261020/manifest.json)、[summary.json](../outputs/wam_video_pose_timing_20261020/summary.json)、[comparisons.json](../outputs/wam_video_pose_timing_20261020/comparisons.json)及[comparisons.csv](../outputs/wam_video_pose_timing_20261020/comparisons.csv)。这些是单个已匹配 episode 的当前字节比较，不是历史训练消费或所有主评测输入的认证。

## stage2：保存物理状态的静态恢复

实现与配置先提交为`c6a6988`，随后只在CPU/OSMesa执行。固定frame13/73/90，各恢复HDF `states[frame+0/1/2]`，在128与256两种分辨率各核对一次，共18项恢复、144项图像比较。没有发送action或请求WAM，任务计分控制步为0，不计入独立任务评测次数。入口与配置见 [静态恢复脚本](../scripts/audit_wam_video_pose_restore.py)、[固定配置](../configs/wam_video_pose_restore_20261020.json)。

18项time/qpos/qvel恢复及刷新/渲染前后检查全部通过，flattened state精确不变；它不包含全部MuJoCo隐藏状态或历史控制器状态。两环境初始化各调用2次`mj_step1`，初始化单列，静态诊断阶段`mj_step/mj_step1/mj_step2`均为0。fresh observable与直接site/body读取的source pose一致（最大位置差0，SO(3)仅浮点量级）。

三个frame在`state[frame+1]`下与HDF同frame的结果如下；MAE为128原分辨率、identity方向，不涉及历史codec重建。

| frame | TCP差 mm | SO(3)差 rad | auxiliary MAE | wrist MAE |
| --- | ---: | ---: | ---: | ---: |
| 13 | 0.060 | 0.001724 | 0.740 | 1.360 |
| 73 | 0.073 | 0.000288 | 0.667 | 0.508 |
| 90 | 0.491 | 0.000852 | 0.905 | 0.734 |

这是对这三个样本HDF动作后端点语义的更直接支持，仍非逐像素/位姿完全重建。完整保留state[frame]及state[frame+2]候选、原始关节/手指/位姿及相机参数，未用手指相等推定全部observable同步。[frame73/state74完整对照](../outputs/wam_video_pose_restore_20261020/resolution128/frame0073_state0074/all_orientations.png)可直接复核。

发布R2流没有由这些候选得到共同精确对应。在`state[frame]`下，三帧实际TCP与同frame发布state的位置差分别3.279/3.115/3.599mm，SO(3)差0.06676/0.07198/0.07413rad；这只是候选恢复与标签的差，不是WAM预测误差，也不能据此认定标签错误。[frame73/state73发布图对照](../outputs/wam_video_pose_restore_20261020/resolution256/frame0073_state0073/all_orientations.png)仍有腕部/机器人残差。多个候选和视角的像素差不能自动选出唯一真实offset；上游发布观测流与此HDF来源之间的对应关系仍未完整恢复。

原始18项结果、144项比较、初始化计数、实际加载XML/资产及输入/产物SHA见 [stage2 summary](../outputs/wam_video_pose_restore_20261020/summary.json)与[manifest](../outputs/wam_video_pose_restore_20261020/manifest.json)。全部指纹已复核，重复执行按预期拒绝覆盖且原产物不变。两阶段均保留物理图像/标签同步`unresolved`，没有拟合位姿、修标签、改变模型输入或停止主矩阵；当前证据不能提升为历史训练消费或全数据物理同步认证。
