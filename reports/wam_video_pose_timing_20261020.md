# WAM 视频与物理状态时刻：stage1 图像复核（2026-10-20）

本阶段核对原生 HDF5 RGB 与缓存发布 RGB 在候选方向、相邻索引下的像素差，尚未建立图像与保存物理状态的时刻对应。结论保留为 **`physical_image_pose_timing="unresolved"`**。仅用 CPU 读取已有图像，没有启动物理仿真、加载 WAM 或重新解码视频，也不改变主评测及成功评分。

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

本次复核时，stage2 的9状态×128/256静态恢复代码正在准备，尚未执行；本文不包含其结果。
