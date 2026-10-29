# 027：train-only尺度readout的teacher结果

本候选**通过事前固定的teacher门槛**：开发validation标准化velocity MSE总体降低57.45%，三个sigma全部改善，并优于原026时间参数化bias。原025/026失败保持原样。本项没有运行WAM完整生成、state评分、物理闭环或fresh；通过teacher只允许登记下一项真实GPU及完整生成资格研究，**尚未证明任务稳定成功**。

方案在新fit前于9118e6e登记，05d46ea/93df84a澄清数学及统计术语，见[027计划](../plans/v0.0.1/027_train_scale_readout_20261021.md)及[配置](../configs/wam_train_scale_readout_20261021.json)。source冻结HEAD `3ba9500`，原84项源码及快照、874项实际读取SHA、163份8400权重stat均绑定到实际prepared。准备及全部teacher审计没有重开原EP图像、PNG或state文件；真实未来仅作为原已核验缓存中的训练标签。

## 唯一变化及预算

仍为同一8400 non-EMA/H40/20Hz/g1/35步/shift5/SDPA/active10，六train EP1172/1098/893/1151/898/850、六开发validation EP1254/1126/868/1139/1209/1176。每EP四anchor、三个实际FP32 sigma、40 endpoint，train/validation各2880行。复用原144缓存不等于新增独立样本，开发validation已经查看，预训练可见性未知。权重位置见[026报告](wam_flow_consistent_readout_20261021.md)。

唯一输入坐标变化为训练集mu=sum(w*h)、s=sqrt(sum(w*||h-mu||²))，phi=[(h-mu)/s,1]；没有逐列白化或额外除sqrt1280。保持026的A=-phi/actual_sigma、原velocity target、逐层等权、lambda=.1及正则bias。中心与全局尺度仅用六train计算，不用validation或future task阶段。它改变正则prior，是新假设检验，不能称已确认的训练bug修复。[实现](../src/s2a/wam_scaled_readout.py:11)。

只执行**一个生产ridge solve和一个独立AugLS重建**。W1281×10、mu1280、s1合计14,091个训练得到的数。实际s=`6.405997476638239`，训练归一化hidden平方L2范数加权均值`.9999999999999998`。原026 bias系数及实际预测直接复用，不重新拟合；通过027实际Torch apply路径重算全部5760×128输出，与原控制逐位相同。[fit与控制复用](../scripts/fit_wam_scaled_readout.py:32)。

## 实际结果

下表为开发validation active10标准化velocity MSE；不是位置米、SO(3)角度、opening误差或task成功率。

| 组别 | 原WAM | 027 readout | 原026 bias | readout/raw |
| --- | ---: | ---: | ---: | ---: |
| pooled | .010130232486 | .004310192331 | .005244761682 | .425478126 |
| sigma=.2 | .025778976352 | .009701235971 | .011820045536 | .376323553 |
| sigma=.5 | .003322271932 | .002014384409 | .002407667577 | .606327372 |
| sigma=.8 | .001289449174 | .001214956613 | .001506571935 | .942229161 |

总体改善57.4522%，相对bias额外改善17.8191%。固定门槛pooled<=.95raw、严格优于bias、每sigma<=1.05raw全部通过，没有质量容差放宽或重复fit。训练三个sigma的MSE也没有恶化，原始完整train/validation及task/EP/slot指标保存在JSON，不把相邻endpoint当独立episode。

独立checker未导入生产fit/apply/gate，按train-only统计和增广LS重建一次：mu及s最大绝对差均0，W最大绝对差`4.7878367936959876e-15`，实际BF16 teacher预测逐值相同。[独立重建及判定](../scripts/verify_wam_scaled_readout.py:330)。这支持新坐标下存在可利用的teacher特征结构，未证明生成轨迹能保持该收益；真实条件采样将访问不同latent分布与teacher未覆盖sigma区间。

整仓CPU测试582 passed / 1外部integration deselected（26.30秒），331个Python文件Ruff check及format通过，源码SHA前后相同。初始跨模块审查修正了owner重建计数字段及JSON排序后的快照编号假设，并用排序往返回归覆盖；均发生在真实prepare/fit前，没有重跑或覆盖候选结果。实际CPU owner、生产fit和独立checker正常wait0/reap，无自动重试，空CUDA/LD。原026 GPU6合成资格不能代替新中心/尺度路径的真实GPU资格。

## 可审计产物与下一步

全部路径相对于`outputs/wam_stability_20261021/`：

| 产物 | SHA256 |
| --- | --- |
| `scaled_readout_source_validation/validation.json` | `3915214c6dba87f45cac0869354f8991b9a2fe4689661f5cd84c4955ac269404` |
| `train_scale_readout/preparation/prepared.json` | `0491feacd77747c124f570aa7287e9790bb46e5c7e186750783b7bf4f5806d1c` |
| `train_scale_readout/review/preparation_verified.json` | `bb5fdaabf026617a63a6457035c649eac215d652c239427b8557b5d53fdb4bad` |
| `scaled_readout_owner_preflight/verification.json` | `6ed76e19dfd2e46d843110bb0753419064180ec5713f97016bb16dd97cea10f3` |
| `train_scale_readout/fit/fit_result.json` | `9be9d1ce6587304f6bed7ba3c89dd839e2e70fb603e4e81ca4eafb7323298a90` |
| `train_scale_readout/review/fit_verified.json` | `d7a9502f4358eb48a8f06f8ad6c8e68ae0e8ec955860f7fcbb6e926e170c13c2` |
| `train_scale_readout/orchestration/fit/result.json` | `f1125438f7ad46f77f4e5666f13626bc25a3048cb68b2932743fc47d6ce5a2ec` |
| `train_scale_readout/review/formal_final_peer.json` | `f36d7c0b4b48b322a19772e9b1a651dec6b5c65ce5dd3a8a7c51a93f5891ace0` |

最终只读peer重核1256项原消费SHA前后相同、910项分组数值最大差`3.469446951953614e-18`，复用控制的full128 FP32位/权重/实际sigma一致，两个CPU子进程原birth均已结束；没有新增solve、fit或模型运行。

本项结果`eligible_for_runtime_qualification=True`、`eligible_for_sampling=False`。下一项另登记新GPU6数值路径、三份zero完整输出exact及96份raw/adapt成对生成，随后独立评估真实future state。原全部state门槛满足后才登记三任务双seed开发物理；最后才考虑统一冻结候选在fresh12初态×2seed上的24次验收。新adapter收益不计为原checkpoint或S2A反解通过，稳定目标保持≥20/24且每task≥4/8、末40持续成功。
