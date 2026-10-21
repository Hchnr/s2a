# 028：冻结 readout 的完整生成 state 验证

**执行及独立评分完整通过，预注册 state 质量门槛失败，不进入开发物理或 fresh。** 027 的 teacher velocity MSE 改善57.45%部分转化为朝向及总体联合误差的改善，但未保护位置、夹爪和两个采样种子。本轮没有任务成功率结果，不能据此宣称原8400 checkpoint正确或新适配系统稳定成功。

固定对象为8400非EMA、原027的一个 ScaledFlowReadout（W/center/scale 不变）、原35步UniPC/shift5/guidance1、LIBERO normalization31/projection0/active10。权重路径为 `/share/project/eai_pwm/repos/cosmos-framework-pwm/outputs/action128_wam_21src_20261012_warm_start_run01/cosmos3_action/action128/action128_wam_nano_1b_22src_20261012_warm_start_videocap30_static_vae_aot_run1_from6k/checkpoints/iter_000008400/model`。新 readout 是生成器适配，不是 S2A 的机器人响应反解；原027结果 `eligible_for_sampling=False` 原样保留，新路径依据[028登记](../plans/v0.0.1/028_scaled_readout_state_20261021.md)单独执行。

## 完成范围与因果边界

物理GPU6的一次模型加载已完成3份零残差完整复现以及96项raw/adapt生成：6个原开发validation episode×4个anchor×seed42/43×两方法，实际3465次denoise。3份zero的raw及normalized全部40×128与原缓存exact；48对实际initial FP32 noise、mask、首condition latent、caption/token相同；99条实际sigma/head trace、单次velocity委托、原模型参数冻结均独立通过。线上生成只输入当前图像和正常task条件，没有未来真实state或图像。

唯一模型actual wait0/reap、没有发送signal，物理GPU6在退出后0.295秒观测到0MiB且compute apps为空。随后唯一CPU独立checker才读取六份真实state标签；其actual wait0/reap同样完成。owner工具session4598实际exit0，两个模型/审计原birth及owner均已退出。原96源、96快照、163项权重stat不变，外部Cosmos tracked源码身份不变。权重stat复核不冒称重新完成32GB全量内容哈希。

评分绕过S2A和物理：从各anchor当前实测EEF积分生成local delta，平移使用上一时刻朝向，旋转在SO(3)上右乘，opening裁到[0,1]。H8/H16/H40为前8/16/40个端点的累计误差，对应0.4/0.8/2秒，不是单时刻误差。joint为 `mean[(position/.01m)^2+(SO3/.05rad)^2+(opening/.1)^2]`，不是成功概率。每方法累计端点分母384/768/1920；保持与仅用历史的CV对照不重复计入raw/adapt两份。

## 完整开发结果

| 累计窗口 | 方法 | 位置RMSE mm | SO(3) RMSE rad | opening MAE | joint |
| --- | --- | ---: | ---: | ---: | ---: |
| H8 | raw | 21.354321 | .102480023 | .035399529 | 9.108176092 |
| H8 | adapt | 22.103255 | .085124416 | .037048529 | 8.138697568 |
| H8 | hold | 40.512859 | .067968626 | .039653629 | 19.783628063 |
| H8 | CV | 29.969697 | .039924568 | .039653629 | 11.142232442 |
| H16 | raw | 36.196357 | .183555945 | .031897557 | 26.850317699 |
| H16 | adapt | 37.582907 | .142353693 | .034277576 | 22.515806209 |
| H16 | hold | 75.807390 | .128847927 | .075500936 | 68.663462996 |
| H16 | CV | 76.903611 | .108867318 | .075500936 | 68.437635714 |
| H40 | raw | 68.432614 | .453458804 | .082558638 | 132.292632470 |
| H40 | adapt | 72.025190 | .303588807 | .084074680 | 91.928139090 |
| H40 | hold | 141.729107 | .315495471 | .194969556 | 253.670804785 |
| H40 | CV | 212.886396 | .369470698 | .194969556 | 520.794065317 |

H8/H16 pooled joint分别改善10.6440%/16.1432%，并严格优于hold/CV；H40 joint改善30.5115%，该项没有另设10%晋级门槛。朝向RMSE三个窗口分别改善16.94%/22.45%/33.05%。这些总体收益不能抵消下列保护失败。

| 原固定检查 | 结果 | 实际证据 |
| --- | --- | --- |
| H8/H16 joint≤.90raw | True | ratio .89355953/.83856759 |
| 两短窗严格优于hold/CV | True | 完整共同端点 |
| 九个组件同时满足相对及绝对保护 | False | H16 opening为1.07461448raw；H40位置为1.05249801raw且增加3.592576mm |
| 至少5/6 EP两个短窗≤1.05raw | True | 实际6/6，通过不代表每seed均改善 |
| 每EP两个短窗≤1.25raw | True | 六个EP全部通过 |
| 两seed两个短窗≤1.05raw | False | seed42 H8/H16 joint为1.24133772/1.17964412raw |

H16 opening虽绝对只增加.00238002，但违反5%相对界限，仍失败；H40位置同时违反5%及+2mm界限。seed43短窗ratio为.70751675/.66668040，不能只选择该seed或将两seed按任务拼接。六个EP与两个seed交叉重复，不能当独立样本数扩增；预训练是否见过这些episode未知。

## 核验与产物

冻结运行源码的完整CPU准入为614 passed、1外部integration deselected，Ruff341文件通过；该数字不包含之后纯JSON绘图脚本。新绘图入口及文案修正分别提交3b538e2/e107612，Ruff/format通过，仅消费四份最终JSON，生成3PNG/3页PDF，已实际视觉检查；原首版保留。合并核验另有45项CPU回归及342份Python Ruff通过，不将重复检查相加。

正式独立checker完整通过后，另一只读审查方独立核99协议、3465ledger、48初态对、生命周期、96源/快照和163stat；711消费SHA前后相同。六GT NPZ及27PNG只核声明与prepared绑定，没有重开内容；EP/task/seed重构pooled的180项最大差1.1368683772161603e-13，原质量比较没有添加容差，gate仍False。没有重复NN、fit、独立参数重建或物理。

| 产物（均在 outputs/wam_stability_20261021 下） | SHA256 |
| --- | --- |
| scaled_readout_state/sampling/worker_result.json | `88e3eb3cd92a7c48578f640b9a3c3a36d095a72ad490c5f24bb3bf8e05c5c9bd` |
| scaled_readout_state/sampling/result.json | `43c49000ea31b211af6d5d898df96878de0cc87ad038b7b8b4a086d82ab8ac77` |
| scaled_readout_state/review/sampling_verified.json | `bf3c349b7eeb243178066d8d42ba366d7b24ef663dad6c895c522b72c2c9b7af` |
| scaled_readout_state/orchestration/sampling/result.json | `065dac4e1ac29a7efd5585b61a1c5295870e620fb751e0fa910c6ff173ade530` |
| scaled_readout_state/review/formal_final_peer.json | `4c9d5a6d9340a6fd845bb79ea21795746eea8226d44ce3474d49e293e3f45924` |
| scaled_readout_state/final_json_figures_v2/manifest.json | `586c8fd11451b586f22db777a35114abe68f40fef7aea2ab2b81f31aac7b2713` |
| scaled_state_runtime_observer/sampling_owner_tool_exit.json | `164270842cfda8107ac8fc90902b302d8d47045a4b7c2f68e42143e0494df1ec` |

[完整分组表及图表说明](../outputs/wam_stability_20261021/scaled_readout_state/final_json_figures_v2/summary.md) · [物理分量图](../outputs/wam_stability_20261021/scaled_readout_state/final_json_figures_v2/physical_components.png) · [EP/seed图](../outputs/wam_stability_20261021/scaled_readout_state/final_json_figures_v2/episode_seed_ratios.png) · [PDF](../outputs/wam_stability_20261021/scaled_readout_state/final_json_figures_v2/state_evidence.pdf)。生成产物不入Git，原始JSON及完整指纹在本机保留。

## 决策与下一验证

保持028的False，不改027系数、lambda、sigma、数据划分或质量阈值，不追加此候选的六条开发物理或fresh。稳定成功仍未达：统一冻结候选必须经过未用于调参的12初态×两seed，至少20/24且各task至少4/8、最后40端点native连续成功。

源码确认teacher使用真实未来video/action加噪，自由生成则沿原solver联合video/action轨迹演化；当前027中心尺度来自teacher训练hidden。该差异是待检验假设，不能由本结果直接断言其是失败主因。下一项另登记有限、不拟合的同actual sigma输入对照，保存自由轨迹sample/packed xt/head hidden/base并与真实future加噪对照；网络时间与噪声sigma分别匹配，避免把整数t/1000或BF16 xt错作原FP32 sample。固定027只作离线点态对照，不将这种反事实评分称为真实adapter轨迹或原训练loss，也不自动授权下一轮训练或物理。
