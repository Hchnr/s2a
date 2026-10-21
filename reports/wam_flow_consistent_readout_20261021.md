# 026：flow时间残差的实际结果 — 2026-10-21

本候选**未通过teacher推广门槛**。开发validation的标准化velocity MSE总体降低49.79%，但sigma=.8退化13.23%，超过事先登记的5%保护上限。因此没有执行三份zero完整生成、96份成对生成、state评分、物理闭环或fresh验收，不能据teacher收益宣称任务能力改善。稳定成功目标仍未达到。

方案事前登记于[026计划](../plans/v0.0.1/026_flow_consistent_readout_20261021.md)，配置为[wam_flow_consistent_readout_20261021.json](../configs/wam_flow_consistent_readout_20261021.json)。原[025失败](wam_frozen_readout_20261021.md)的参数、门槛、产物和结论保持原样。此次是模型输出适配研究，不是S2A反解、已知推理bug修复或原checkpoint通过的证明。

## 实现与固定条件

WAM保持22-source non-EMA8400、H40/20Hz、guidance1、35步UniPC/shift5、SDPA和active10。实际权重位置：

```text
/share/project/eai_pwm/repos/cosmos-framework-pwm/outputs/action128_wam_21src_20261012_warm_start_run01/cosmos3_action/action128/action128_wam_nano_1b_22src_20261012_warm_start_videocap30_static_vae_aot_run1_from6k/checkpoints/iter_000008400/model
```

只复用025已核验的144份teacher缓存；六训练EP为1172/1098/893/1151/898/850，六开发validation EP为1254/1126/868/1139/1209/1176，每EP四anchor、三个sigma、40个endpoint。训练和validation各2880行，144份缓存不是新增独立样本，validation已经查看，不能称预训练未见集。此次准备、拟合及teacher审计没有重新打开原EP图像/state文件。

依据原solver的`x0_hat=xt-sigma*v`，新设计矩阵为`A=-[h/sqrt1280,1]/sigma_actual`，拟合原`v_target-v_base`，保留逐EP/anchor/sigma/endpoint权重及velocity损失。新W表示标准化clean-action残差，共12,810参数；另有同时间参数化的10参数bias对照。两组fit都固定ridge=.1，无网格搜索。相同lambda数值作用于不同单位的W，因此正则先验与025不同。[实现](../src/s2a/wam_flow_readout.py:40)及[固定fit](../scripts/fit_wam_flow_readout.py:196)。

Teacher sigma取保存的FP32值无损扩为FP64，实际为`.20000000298023224 / .5 / .800000011920929`。没有从BF16 xt反推clean标签。实现还支持从原UniPC实例读取实际solver sigma、每步只委托一次原velocity、非正sigma在NN前拒绝；该完整WAM路径本次因teacher失败而**未实际运行**，不能把合成测试当作真实完整采样身份验证。[solver读取](../src/s2a/wam_flow_readout.py:271)。

## 实际验证

冻结源码HEAD `271ae37`的整仓CPU检查为546 passed / 1外部integration deselected；321个Python文件的Ruff check及format通过，源码SHA前后相同。真实准备绑定72源及72快照、761项实际读取SHA和163份权重stat；独立prepare-only及source peer通过。source peer额外核1120项原消费SHA，无新EP/PNG读取。

新运算路径在物理GPU6实际执行纯合成资格：五个实际sigma乘正/负/次正规共15组，另10个原语用例。CPU与CUDA的完整128维BF16位相同，实际hook与直接调用一致，zero返回原对象，其它domain/118槽及输入张量保持。独立NumPy从保存的原始bits另算通过。模型权重加载、WAM denoise、fit、EP读取和物理步在该阶段全为0。实际子进程915341/start170505405正常wait0、reap且未发送signal；资源逐次观测后GPU6显存0、无compute进程。指定模型环境Torch2.10.0+cu130。

随后仅CPU完成恰好两组生产fit及两组独立增广最小二乘重建。独立检查器不导入生产fit/apply/gate；参数最大绝对差readout为`4.704570066849101e-15`、bias为0，实际量化teacher预测逐值相同，全部分组指标和严格门槛一致。[独立重建及评分](../scripts/verify_wam_flow_readout.py:714)。生产fit与独立checker均由owner实际wait0/reap，无自动重试。CPU环境空CUDA/LD，Torch2.12.0a0+NV26.04；实际GPU合成资格验证相同BF16舍入合同，不冒称非零适配器已部署到WAM。

## Teacher结果及未执行项

下表为开发validation、active10标准化velocity MSE；它不是米、SO(3)角度、opening误差或task成功率。

| 组别 | 原WAM | readout | 时间参数化bias | readout / 原WAM |
| --- | ---: | ---: | ---: | ---: |
| pooled | .010130232486 | .005086448880 | .005244761682 | .502105839 |
| sigma=.2 | .025778976352 | .011453058440 | .011820045536 | .444279023 |
| sigma=.5 | .003322271932 | .002346184705 | .002407667577 | .706198876 |
| sigma=.8 | .001289449174 | .001460103496 | .001506571935 | **1.132346683** |

总体改善49.7894%，相对同时间bias额外改善3.0185%。pooled≤.95raw与strictly beat bias通过；每sigma≤1.05raw失败，整体gate为False。训练sigma=.8也退化，raw `.001275719057` → readout `.001516574502`，说明此次问题不只出现在留出EP。相比025的该组2.043倍退化，新时间参数化缩小了退化，但不改变026自身失败判定。

根代理实际执行sampling `--dry-run`，退出码1，报`026 producer teacher gate rejected sampling`，在模型加载和sampling目录创建前拒绝。实际新WAM前向0、完整生成0、state评分0、物理/fresh0；预留12初态×双seed的24次验收仍未打开。不调lambda、sigma、容量或门槛挽救此次候选。

## 核心产物

全部位于`outputs/wam_stability_20261021/`，完整缓存与运行产物不进Git。

| 产物 | SHA256 |
| --- | --- |
| `flow_readout_source_validation/validation.json` | `8f208532fd1735211608cf0b604837d1ad5b2ac7f736d56dc2cbfd67d7ab9c4d` |
| `flow_consistent_readout/preparation/prepared.json` | `8fa485451c906a55f83a485f08df15ec253779ec8d6046d8ccd677e7b3f8fb56` |
| `flow_consistent_readout/review/preparation_verified.json` | `476045cb0c5f628b3c1b7902cc39d4c448cd18400b83744d4b36aa4a5fb74c49` |
| `flow_consistent_readout/review/source_peer.json` | `818f0ffaa7f6cb6479611ed5c8443a5ad8521ae739e32b0225e152091dc1dc83` |
| `flow_readout_owner_preflight/verification.json` | `6e3a3c277e97227aeb86983be85ce6c0dcab4d3af8dbe03b25d5215621797e2f` |
| `flow_consistent_readout/precision/result.json` | `ad83c8f7e891f277c957abfc156046044074e5038966de22f88ca306878e6f87` |
| `flow_consistent_readout/review/precision_source_peer.json` | `8b9625e3d0ef95801ff6f605b2960995ef10d9103c9e4a54e25ec9cf5b8092a5` |
| `flow_consistent_readout/orchestration/precision/result.json` | `44a5e29ccc350b5b49e2638e565bfba8f0db235d905dc00d96b99195f407dc7c` |
| `flow_consistent_readout/fit/fit_result.json` | `78177d5e9f72fda1bf9a3b068a9d64eaf81dfdeb1bf28828ea2e08b6d2b30fa1` |
| `flow_consistent_readout/review/fit_verified.json` | `bbdbab1157bd5b3f4734b5192ebd467baecb4a1be13954ddb853a3fa58fd0733` |
| `flow_consistent_readout/orchestration/fit/result.json` | `2123ba259f46788f2a3139c66e11242e123f7a73c984da6fbd6668d3f6e8307b` |
| `flow_readout_owner_preflight/sampling_teacher_gate_rejected.json` | `c1759be99f6398cc133db21b4064375a22bde91ddc85c48410d9cc2c601cc16e` |
| `flow_consistent_readout/review/formal_final_peer.json` | `604f1463a500529bced224c81c4e991c9eea214d2f703921dccb3c96231a65c7` |

最终只读peer重核861项消费SHA前后不变，910项生产/独立分组数值最大差`3.469446951953614e-18`；两个CPU子进程原birth均已结束。没有为补指标而重复求解、重新前向或改写原结果。

上述证据排除了本次CPU求解/量化重建不一致的解释，表明固定readout先验仍不能覆盖各噪声阶段。它不证明所有模型学习方向无效，也未认证完整历史训练消费、上游物理同步或原checkpoint的任务能力。下一项先查缓存中特征尺度与不同sigma残差的结构，再独立登记具有依据的新方案；稳定验收门槛保持不变。
