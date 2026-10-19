# WAM 与真实 state 的直接比较 — 2026-10-19

只读当前 R2 Parquet 的 observation.state；没有读取专家 action/native command。复用此前同源 frame0 图像的10份保存预测，没有重新推理或新增仿真。
frame0实测canonical TCP为积分锚点，物理delta按SE(3)右乘积分，对比frame1..H真实TCP；旋转为SO(3)测地角，开度对比下一帧实测归一化opening。
所有数值为5条已知初始上下文的算术平均；点时刻误差与整段RMSE分开，不作未见场景结论。这里只评机器人TCP/开度，不评物体状态或生成视频。

Checkpoint（普通权重，非EMA）：`/share/project/eai_pwm/repos/cosmos-framework-pwm/outputs/action128_wam_21src_20261012_warm_start_run01/cosmos3_action/action128/action128_wam_nano_1b_22src_20261012_warm_start_videocap30_static_vae_aot_run1_from6k/checkpoints/iter_000008800/model`。seed42，20Hz，35采样步，guidance1.5，shift5；与任务效果专项相同权重。

| 请求长度 | 时刻 | WAM位置误差mm | WAM朝向误差° | 保持初态位置误差mm | 保持初态朝向误差° |
| --- | ---: | ---: | ---: | ---: | ---: |
| 16步 | 0.05s | 6.700 | 17.264 | 0.223 | 0.028 |
| 16步 | 0.40s | 26.589 | 117.055 | 33.661 | 2.112 |
| 16步 | 0.80s | 141.877 | 175.745 | 125.826 | 4.207 |
| 40步 | 0.05s | 4.462 | 19.919 | 0.223 | 0.028 |
| 40步 | 0.40s | 30.312 | 97.817 | 33.661 | 2.112 |
| 40步 | 2.00s | 280.687 | 105.338 | 269.272 | 6.034 |

| 请求长度 | 整段位置RMSE mm | 整段朝向RMSE° | 裁切后开度MAE（0–1） | 原始开度越界 |
| --- | ---: | ---: | ---: | ---: |
| 16步 | 66.731 | 116.696 | 0.06363 | 56.25% |
| 40步 | 184.820 | 126.206 | 0.09790 | 55.50% |

16/40步是分别请求得到的不同预测，不能把它们拼成同一条连续轨迹。误差不含任何控制器跟踪误差；单条示范差异本身不证明任务失败，任务归因仍依靠独立物理正控/分量/几何证据。

验证：当前原始Parquet SHA与图像配对时一致；重新读取的state与快照逐项相同；10份response/manifest checkpoint一致；280个预测endpoint通过独立R/t递推对照（1e-12）；主要输入SHA、每条误差曲线见同名JSON。
