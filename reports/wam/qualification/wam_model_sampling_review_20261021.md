# WAM 新权重、EMA 与采样步数只读审查 — 2026-10-21

发现一个明确的新模型候选：23-source/FastUMI continuation 的最新完整保存已是 **iter_000015600**，不是此前的15400。它与15400属于同一run、同一保存配置，并不因iteration更大而成为best。8400/8800/15400/15600均没有可切换的完整EMA权重。当前源码未发现新的确定solver错接，现有已核WAM配置没有35→70/140的配对收敛诊断。

本轮只读外部代码与训练产物；没有启动模型或物理。root随后按009单独启动的48条state预测不属于本报告的新增调用。新15600的一次完整byte hashing由009明确授权，未重复hash旧大权重或反序列化模型。

## 新15600身份与候选范围

canonical目录为：

```text
/share/project/eai_pwm/repos/cosmos-framework-pwm/outputs/action128_wam_23src_20261015_warm_start_from22src_i8800_run01/cosmos3_action/action128/action128_wam_nano_1b_23src_20261015_warm_start_from22src_20261012_i8800_from12src_e2aux345_i6200_videocap30_static_vae_aot_run1/checkpoints/iter_000015600/model
```

`latest_checkpoint.txt`明确指向15600，iteration下`.save_complete`存在，model含`.metadata`。当前同run完整marker清单为12000/14000/15200/15400/15600；没有只按目录数字选未完成保存。完成判定对应外部`inference/action_serving.py:66–88`。

首次元数据快照保存目录mtime、97个model文件的size/mtime，以及latest、marker、runconfig和DCP metadata的小文件SHA。model总32,398,593,251bytes，16个非空distcp、80个空distcp及一个metadata；空文件是当前DCP布局的一部分，未自动当作损坏。随后按009、budget commit `0a248f2`完整hash runconfig＋marker＋97个model文件共99项、32,399,051,175bytes，用时29.58秒；每文件读取前后及最终全部stat均一致。该步骤认证字节身份，不证明学习质量或tensor语义完整性。

| 身份文件 | SHA256 |
| --- | --- |
| latest pointer | `ab8af33efb6292ac8af6e36eb347ff4909e4d1cf44a41075c6a9e2d310ffebf6` |
| `.save_complete`（空文件） | `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` |
| run `config.yaml`，与15400相同 | `20ca2db81443818c9be02ffde7da8fa07eae41a6403e8214ee7a33804f6951d6` |
| model `.metadata` | `72f2730b737ae9af97c85dc321d588252f3ace332fe6d27fe5955f4eb917f2d9` |
| [元数据快照](../outputs/wam_stability_20261021/model_sampling_review/candidate15600_metadata.json) | `dc943caae2d1b860d6c21f94a53d4bcae56725f4877b7443fc835ecf680ecf61` |
| [完整99文件指纹](../outputs/wam_stability_20261021/checkpoint_15600_complete_fingerprints.json) | `bb86dfa536b2db606bd2c9027f43ef85b804fdf00fe75cdb753d543df0757e8c` |

查重只读四个明确S2A输出位置及旧固定fingerprint receipt：未发现已保存的15600配对request或旧完整指纹；标准screening目录已由009准备但当时request为0。没有全盘扫描推理历史，也不声称其他存储从未评过15600。

只读当前线上默认launcher/TOML及显式训练outputs根下15个已有run配置，未发现新增的高LIBERO占比、完整continuation候选。launcher仍指该23-source run，并从22-source8800 warm-start。23-source保存配置的LIBERO-eef/joint ratio为6/4、全部ratio合计10000，配置序列权重合计0.1%；旧22-source对应约1.96194%。这不是实际历史消费或frame占比，不能单凭该比例归因遗忘。较早配置虽可能有较高权重，不能冒充新的高占比continuation。此清单只覆盖该明确输出根，不认证所有远端训练作业。

## EMA：实际未启用，也未保存

22-source与23-source的保存训练配置都为`model.config.ema.enabled=false`，线上TOML也显式禁用。不能因为配置包含`rate=.1`或defaults里有`ema: power`就认为EMA在维护。

当前外部`model/generator/omni_mot_model.py`在568–577仅在enabled时创建、初始化`net_ema`，781–784仅在enabled时更新；4731–4776说明EMA若启用会以`net_ema.*`保存在**同一个model DCP**，不是另找EMA目录。使用pickletools仅读DCP metadata的opcode字符串、不unpickle或加载tensor，8400/8800/15400/15600各有1214个`net.*`键字符串，`net_ema.*`均为0。由配置、源码语义和实际metadata共同支持：这四个checkpoint没有可供切换的已保存EMA权重。

完整路径、15配置、四份metadata及两份源码/launcher身份见 [inventory](../outputs/wam_stability_20261021/model_sampling_review/training_and_ema_inventory.json)，SHA `cf026f75523b908575b8bb746617f87a825cae3a86177e548e3ecf308e1f8e9b`。这没有重放历史优化器事件，也没有重复读取旧32GB权重。

## 35、70、140步：尚无效果收敛证据

本轮检索的已核WAM报告及28份相关冻结配置均为35采样步；未发现相同checkpoint、相同观测、同seed下35/70/140的配对state收敛试验。这个结论限定于明确检查范围，不是全盘历史调用的否定证明。

当前23-source模型保存`fixed_step_sampler_config=null`、scheduler=`unipc`、1000训练噪声时间单位。请求`num_steps`和`shift`经`action_server.py:180–185`传给`generate_samples_from_batch`，再由`omni_mot_model.py:3646–3654`传给UniPC；`samplers/unipc.py:62–75`将显式shift交给`set_timesteps`。因此保存inference config中的构造默认shift1**不等于实际按shift1采样**，本轮请求shift5会覆盖它。训练视频/共享action的shift配置在480档为5，`independent_action_schedule=false`；没有新的shift漏传证据。

训练`rectified_flow.py:176–211`使用noise系数sigma，`x=sigma*noise+(1-sigma)*data`、velocity=`noise-data`；UniPC为flow_prediction、从高sigma向0积分，未发现新的符号反向。物理20Hz与1000个噪声时间单位是不同概念，不能混为控制dt错误。

源码还有一个应记录、尚未证明有害的数值近似：`fm_solvers_unipc.py:181–203`内部保留float32 sigma，但传给velocity网络的`sigma*1000`转为int64；训练使用连续sigma。按当前shift5、默认sigma端点仅用NumPy重建schedule，35/70/140均无重复conditioning timestep，内部sigma与`int_timestep/1000`最大差约`.000962/.000991/.000991`。这不是已证实的任务失败原因；未据此改scheduler、浮点timestep或增加采样预算。

后续若检验更多步，应先冻结一个模型、相同H40/图像/text/mask/domain/seed及shift5，复用35步对应样本；用预先固定、跨episode/阶段的小state集合对比70/140，并记录初始vision/action噪声指纹、实际sigma/timestep列表、有效forward次数和输出身份。固定seed还须核初始噪声相同，不能只比请求字段。评价H8/H16/H40的GT state误差，同时比较35→70与70→140生成state差、opening及rot6d有效性，避免只挑改善样本或把padding计入。

若70→140输出趋于相同但仍明显偏离有效未来，证据更支持已逼近该生成场的输出，而不是权重正确。更多积分步可能减小离散化误差，也可能在错误velocity、BF16数值或多解示范下不改善甚至变差，不能预设单调收益。state改进仍须另行冻结闭环验证，不自动算task成功。本报告没有启动该对照。

[solver源码与scalar schedule审查](../outputs/wam_stability_20261021/model_sampling_review/solver_contract_review.json) SHA为 `b84b947b718e1fcb1cefc9faf7991b8d6b31a05ec4c9dae01a5663e560b08cb3`，包含源码/保存配置SHA、28配置清单及三份完整标量schedule；没有计算任何模型velocity、latent rollout或任务结果。
