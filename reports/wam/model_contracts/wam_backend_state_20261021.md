# 8400 attention backend state对照 — 2026-10-21

**保存的hybrid `flash3_video` backend未达到预注册state候选门槛。** 完成48个Flash3请求，并逐项配对011的48个SDPA35请求；本协议新推理48次、0物理步。结果不表示闭环任务成功率，也不授权自动增加物理或fresh预算。

固定checkpoint：`/share/project/eai_pwm/repos/cosmos-framework-pwm/outputs/action128_wam_21src_20261012_warm_start_run01/cosmos3_action/action128/action128_wam_nano_1b_22src_20261012_warm_start_videocap30_static_vae_aot_run1_from6k/checkpoints/iter_000008400/model`。非EMA、guidance1、UniPC35、shift5、H40、20Hz、active十槽mask、task-only、seed42/43全部不变；仅attention backend由SDPA切换为保存配置的flash3_video。

该backend实际为VIDEO使用FlashAttention3、UND/ACTION使用Flex的hybrid路由。保留推理padding关闭、small_action_fixed_lengths=None、short_tokens=0；model/decoder compile=false，Flex内部torch.compile保留。不能将本实验描述为全部attention都使用FA3或所有kernel都不编译。
实测net与language backend均为`flash3_video`，provider `nv`版本`1.0.3`、FLASH3_SUPPORTED=true，实际CUDA UUID `cebc8060-3ec8-9455-2af2-e4584d0c34ce`与物理GPU7登记一致。CPU协调器/worker空CUDA/LD；本协议仅GPU7/9587服务。

24个phase样本来自6条历史开发episode，22个独特anchor、44个anchor-seed组合；phase重复使请求数为48，不等于48条独立episode，也不证明预训练未见。直接从source EEF anchor按发布local delta合同积分，未经过S2A或TCP映射；opening按原协议clip到[0,1]。未来GT padding不计分，H8/H16/H40各384/768/1778有效端点。

每格为位置RMSE(mm) / SO(3) RMSE(rad) / opening MAE。

| 前缀 | SDPA35 | hybrid Flash3 | hold | constant velocity |
| --- | --- | --- | --- | --- |
| H8 | 24.383877 / 0.088644022 / 0.058537738 | 24.394018 / 0.087966467 / 0.058620610 | 38.200979 / 0.052694824 / 0.076542325 | 26.426437 / 0.043212638 / 0.076542325 |
| H16 | 44.622633 / 0.177223411 / 0.086038070 | 44.663908 / 0.175742768 / 0.086007300 | 71.143119 / 0.114498066 / 0.161876950 | 68.557576 / 0.107911151 / 0.161876950 |
| H40 | 85.521032 / 0.479894370 / 0.114659660 | 85.294733 / 0.477846925 / 0.114435689 | 128.832086 / 0.309419160 / 0.255167268 | 179.116958 / 0.327273028 / 0.255167268 |

## 预注册门槛

H8/H16位置、SO(3)、opening六项全部不退化（位置展示mm先转SI m，再用1e-10 m容差；其余1e-10 rad/opening），且至少一个位置RMSE改善10%。严格沿用启动前013注册，不因本次输出调整阈值。
六项不退化：`False`；至少一项位置改善10%：`False`；合格：`False`。合格也只表示可以另行预注册开发候选，不是完整checkpoint任务能力认证。

| 指标 | 前缀 | hybrid Flash3(SI) | SDPA35(SI) | 不退化 |
| --- | --- | --- | --- | --- |
| position_rmse_mm | H8 | 0.0243940178853 | 0.0243838771895 | False |
| rotation_rmse_rad | H8 | 0.0879664666432 | 0.0886440215097 | True |
| opening_mae | H8 | 0.0586206096583 | 0.0585377380097 | False |
| position_rmse_mm | H16 | 0.044663908304 | 0.0446226332027 | False |
| rotation_rmse_rad | H16 | 0.175742767965 | 0.177223411381 | True |
| opening_mae | H16 | 0.0860073004005 | 0.0860380699511 | True |

## 逐Episode误差

| Episode | 前缀 | SDPA35 | hybrid Flash3 |
| --- | --- | --- | --- |
| 1098 | H8 | 29.904364 / 0.078633947 / 0.042684469 | 29.926388 / 0.078302440 / 0.042782669 |
| 1098 | H16 | 35.676484 / 0.141479109 / 0.037228359 | 35.703073 / 0.140378248 / 0.037259970 |
| 1098 | H40 | 52.190349 / 0.364990982 / 0.053275802 | 52.350692 / 0.365717401 / 0.053161455 |
| 1151 | H8 | 32.698070 / 0.112716518 / 0.053754079 | 32.668859 / 0.111639933 / 0.053204377 |
| 1151 | H16 | 58.230927 / 0.238501347 / 0.076044811 | 58.317237 / 0.237643610 / 0.075424102 |
| 1151 | H40 | 113.787052 / 0.641869487 / 0.050100520 | 114.124813 / 0.642257171 / 0.049588800 |
| 1172 | H8 | 19.649060 / 0.066158341 / 0.094599005 | 19.541294 / 0.065017930 / 0.094855132 |
| 1172 | H16 | 29.127593 / 0.139903570 / 0.125244546 | 29.005862 / 0.138382540 / 0.125171605 |
| 1172 | H40 | 57.028122 / 0.397738511 / 0.130030905 | 56.603821 / 0.397897675 / 0.129945616 |
| 850 | H8 | 13.650387 / 0.061567447 / 0.015266467 | 13.585096 / 0.061254637 / 0.015599243 |
| 850 | H16 | 26.607535 / 0.119263362 / 0.014704990 | 26.488395 / 0.119367225 / 0.015009100 |
| 850 | H40 | 51.950408 / 0.334025871 / 0.035658098 | 51.812644 / 0.333063408 / 0.035765088 |
| 893 | H8 | 30.292418 / 0.127074563 / 0.130196632 | 30.436264 / 0.126402035 / 0.130307669 |
| 893 | H16 | 71.285705 / 0.256111256 / 0.248415187 | 71.524413 / 0.252445010 / 0.248570603 |
| 893 | H40 | 140.056866 / 0.678752292 / 0.341659586 | 139.380485 / 0.672345627 / 0.341510561 |
| 898 | H8 | 10.675056 / 0.062792063 / 0.014725776 | 10.713657 / 0.062264857 / 0.014974567 |
| 898 | H16 | 25.404621 / 0.110271760 / 0.014590526 | 25.196465 / 0.109571163 / 0.014608422 |
| 898 | H40 | 47.264939 / 0.330471386 / 0.062949815 | 46.712603 / 0.326525090 / 0.062353500 |

## 逐Phase误差

| Phase | 前缀 | SDPA35 | hybrid Flash3 |
| --- | --- | --- | --- |
| approach | H8 | 14.535948 / 0.089565142 / 0.014610890 | 14.529170 / 0.088475992 / 0.014729818 |
| approach | H16 | 33.249003 / 0.211428783 / 0.013975020 | 33.269264 / 0.209131297 / 0.013963269 |
| approach | H40 | 82.829000 / 0.633215943 / 0.013430249 | 83.002473 / 0.630110193 / 0.013331141 |
| closing_proxy | H8 | 12.190139 / 0.069380107 / 0.105581946 | 12.165255 / 0.068423866 / 0.105888738 |
| closing_proxy | H16 | 25.249421 / 0.140528278 / 0.083187160 | 25.196336 / 0.138727838 / 0.083233191 |
| closing_proxy | H40 | 75.874856 / 0.438517774 / 0.105154575 | 75.849687 / 0.436300025 / 0.104977946 |
| release_proxy | H8 | 31.757930 / 0.078718023 / 0.086629201 | 31.816926 / 0.078552233 / 0.086384718 |
| release_proxy | H16 | 59.673433 / 0.143200929 / 0.190615547 | 59.840599 / 0.141020028 / 0.190433494 |
| release_proxy | H40 | 87.815695 / 0.340129099 / 0.205021997 | 86.819391 / 0.335802216 / 0.204333160 |
| transport_height_proxy | H8 | 31.777893 / 0.111350839 / 0.027328916 | 31.762643 / 0.110779588 / 0.027479164 |
| transport_height_proxy | H16 | 51.582647 / 0.201682217 / 0.056374553 | 51.544759 / 0.201678262 / 0.056399247 |
| transport_height_proxy | H40 | 95.499304 / 0.426531016 / 0.159286682 | 95.298747 / 0.427543619 / 0.159253353 |

## 生成state距离与核验范围

| 前缀 | SDPA→hybrid Flash3生成state距离(mm/rad/opening) |
| --- | --- |
| H8 | 1.115596 / 0.003470494 / 0.001211695 |
| H16 | 1.602722 / 0.007217091 / 0.001145391 |
| H40 | 3.612229 / 0.015324091 / 0.001347168 |

生成距离仅描述两种计算backend输出不同，不单独证明正确性或错误根因。GT future也不是唯一可行的任务轨迹。本轮保留推理padding关闭，因此没有核验训练padding本身的影响。

48对请求、actual sampler初态/clean/reference/mask/token/实际caption、分支、35次velocity求值及实际sigma/timestep均核验。每份raw/normalized full40×128响应与服务本地原始JSON、per-request receipt和trace绑定。独立Gram–Schmidt解码、local旋转/位移积分与hold/CV重算逐端点匹配保存指标。网络35NFE由冻结guidance1/单rank/无postprocess调用分支推导，未宣称直接hook全部网络模块。
实际初态/clean/mask/token保留的是已观察值的digest witness，而非完整tensor原始字节；审计可核跨backend digest一致，无法从未保存tensor独立重新算该SHA。实际sigma/timestep保存了数值，已独立重建并复核字节SHA；参数/RNG无修改证据来自冻结observer的逐观察guard与CPU透明性测试。
初版只读auditor将scheduler_config._use_default_values视为有序list，产生27条审计错误；原development_partial01完整保留。Diffusers configuration_utils.py明确用list(set(...))构造该默认字段名列表。最终审计仅对这一bookkeeping列表排序，仍要求完整键集合、默认名集合、全部有效scheduler参数、sigma/time数值exact；此为审计表示修正，不是模型/物理失败，不涉及运行源码或采样修改。
state质量审计`quality_complete=True`与退出可观测性分开：本次使用`explicit_sigterm_shutdown_supplement`。服务最终`closed.json`可得：`False`；initial_noise RMS内存ledger可得：`False`；最终poison内存状态：`None`（None表示未知）。不声称full internal ledger audit。
原strict入口因缺失closed/mask_evidence而拒绝final；失败记录和启动字节保留。本次显式绑定预审查的shutdown supplement：实际home Uvicorn在退出capture_signals时恢复默认SIGTERM handler并重新raise信号，使OS终止先于外层Python finally；同解释器、无app/socket/model/physics的CPU复现得到return=-15、进入标记存在、finally标记缺失。未补造丢失文件、未重跑或修改冻结服务。
窄旁证核验48条HTTP200、全部raw/trace/actual mask合同和无silent backend fallback、完整预算和来源、登记PID/startticks消失及state0/model受控-15退出。这支持预注册state质量和有限生命周期结论；不能复原或独立验证最终pending/counter/poison内存值，也不能从未写出的initial_noise RMS ledger推断逐denoise所有中间值。mask_evidence原本仅记录initial_noise诊断。原始响应inactive槽为0、实际sampler输入跨backend digest exact及冻结mask实现的来源核验仍完整保留。
完成时独立重新核验`1126`份保存输入/source/snapshot SHA与163份已完整哈希权重的size/mtime，未再次全量读取32GB权重。登记协调器/模型/state worker均按PID/startticks正常退出；GPU/CVD/backend环境证据来自运行时保存快照，而非根据已退出进程推断。

## 产物

- 独立完整审计：`outputs/wam_stability_20261021/backend_state_audit/complete48_verified.json`，SHA256 `aef599b683d58e6580eb6c7687d3c1ecb256faf66806f6a6e3f514485bad8939`。
- 四张PNG和4页PDF：`outputs/wam_stability_20261021/backend_state_audit/figures`；输入与产物SHA见其`report_manifest.json`。
- 方案：`plans/v0.0.1/013_attention_backend_state_20261021.md`；启动前012完整/释放及011全部48 traced35 exact门禁均已绑定，012候选是否合格不影响013按计划独立执行。
- 未启动fresh评测、真机或新增物理；011更多采样步、010短重观测与本报告分母分开。
