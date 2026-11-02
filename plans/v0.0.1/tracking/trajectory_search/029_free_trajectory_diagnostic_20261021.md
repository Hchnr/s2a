# 029：自由轨迹与同实际时间的 teacher 输入诊断

在任何本项新GPU、模型或样本读取前登记。028完整99生成已独立结束，state gate=False：总体短期joint改善，但H16 opening、H40位置与seed42保护失败；[正式结果](../../reports/wam_scaled_readout_state_20261021.md)不修改。本项研究其可能的输入分布因素，不先验认定根因，不改027系数或028门槛，不拟合新候选。最终目标仍是统一冻结系统fresh12初态×两seed至少20/24、各task至少4/8、末40端点native连续成功。

固定原8400非EMA、原35步UniPC/shift5/guidance1、当前相机/task/normalization31/projection0/active10。仅原六个validation EP（1254/1126/868/1139/1209/1176）各原四anchor、seed42/43，24anchor/48raw完整生成；两seed使seed42反退与seed43改善都能进入同一诊断。所有48raw均与028同anchor/seed完整raw及normalized40×128精确复现；不再加三份zero生成，不采原train EP或fresh。新的统计仍是已查看开发集，不能称无偏或预训练未见。

执行顺序固定：

1. 新入口/来源闭包/必要CPU测试和独立准备通过后，先物理GPU6一次纯合成数值资格，0 WAM。冻结原FlowUniPCMultistepScheduler真实convert_model_output，35/shift5的节点0/19/29/33/34 × velocity dtype BF16/FP32 × 两个固定payload（有符号正常值、边界/次正规混合），共20场景；同场景CPU/CUDA完整128输出bits核验。保存actual sigma/device、sample/model_output/乘积/x0 dtype及bits，独立NumPy按真实乘法后减法的舍入顺序重算，另报告FP64理想式差，不将理想实数公式冒称实际solver精确逆。GPU正常wait/reap、逐次资源观测0MiB/apps空及独立资格通过才进入模型阶段。
2. 模型只初始化一次，先完成48raw/1680denoise。readout为zero且返回原head对象；新observer只观察原velocity/sample、packed输入和head，原solver和NN每次仍委托一次。不改原96冻结源或外部Cosmos；预先冻结新完整源码、原权重stat、真实Popen/birth与预算。每次保存当前图像/condition/实际initial noise/mask/首latent及完整35步sigma/head/velocity trace。48输出全部复现与原参数冻结通过，并把完整free阶段manifest/SHA写盘后，才允许打开本项真实未来样本。
3. 每个raw仅在step index19/29/33保存中间张量，共144个自由轨迹时点。记录原FP32平坦sample及实际vision/action布局、40×128 action切片、实际BF16 packed action/vision、最终packed action/vision时间的值/shape/dtype、head hidden40×1280/base40×128、实际返回solver的velocity/action及dtype。观察真实scheduler的convert_model_output单次调用，保留其实际action x0输出与dtype；不得额外调用NN、替换solver更新或靠最终输出反推中间sample。选中sigma来自原028实际trace：.8077949285507202/.5081728100776672/.2323688119649887，不能冒称旧teacher的.8/.5/.2。
4. 全部free结束后，移除flow hook/restore bridge及观察包装，改用已有zero teacher hook。仍同一个冻结模型，在24anchor×两noise seed×三个相同actual sigma上做144次单次teacher denoise。只读同anchor真实41future RGB/40action作为明确标识的训练式诊断输入，绝不回填raw生成。加噪sigma直接用保存actual FP32数值，不经timestep/1000往返；teacher packed网络时间采用raw实际int64 timestep的数值，并逐项核其最终packed时间值/shape/dtype与对应raw保存一致（最终packed dtype不预设为int64）。这称为“真实future加噪、推理时间配对teacher诊断”，不冒称复算未改的原训练loss。保存真实epsilon/x0/target、packed xt/时间、hidden/base/实际有效velocity、condition/mask及两seed实际RNG。
5. 模型正常actual wait/reap和GPU释放后，独立CPU先核48完整复现、1680+144=1824边界ledger、144+144中间时点、实际sigma/网络时间/布局/mask/单次委托/参数/来源及新数值资格。再只读本六EP的标签，报告下列预定统计；027模型只加载原冻结W/center/scale，不fit或重建参数。原028False保持，无物理或fresh自动授权。

统计与解释固定：按三个actual sigma、两个seed、EP/task及pooled报告，EP为聚合单位，不把144时点或5760端点当独立trial。报告hidden相对原027中心/尺度的范数、均值偏移、teacher/free配对差；video/action潜变量差异与当前condition差异单列，不能把所有差异唯一归于hidden。没有新估计归一化、拟合协方差或选择sigma。

Teacher原监督为epsilon-x0；free另定义所选示范的点态目标v_pull=(sample_FP32-x0)/actual_sigma。两者分开评分，保存q=packed_BF16_action-sample_FP32及q/sigma；从量化xt反推新目标会额外引入q/sigma，不能据此称原teacher标签错误。固定027在保存的raw hidden上离线计算量化velocity及点态clean误差，对raw/add027分别报告，含2*e·delta+||delta||²的方向/幅度分解及各active组件。另核实际混合精度convert输出与理想FP64式之间差异；不把在raw轨迹上的离线readout反事实称作028真实adapter轨迹，或推导多步任务收益。

本项只有协议/数值完整性门槛，没有按质量结果自动启动训练或物理的门槛。若teacher/free输入统计或固定readout点态作用明显不同，只支持分布差异的实测证据；其是否为028失败原因还需后续单独干预。任何新fit、更多sigma/seed、额外生成或物理都必须另登记。若当前raw不能完整复现或来源/数值资格失败，保留False并停止，不自动重试或放宽。

预算：一次20场景纯合成GPU资格（0WAM）及其CPU独立审计；模型init1、48raw完整生成/1680denoise、144teacher单次denoise，总1824；144free+144teacher保存时点；新fit/独立参数重建/新train EP/fresh/物理/真机均0。GPU仅物理6；CPU及任何OSMesa空CUDA/LD。owner保存实际owned子进程与wait/reap、退出后资源观测最多120秒逐次持久化；不猜NVML PID、不kill未知进程、不因观察超时重启。已有目录拒覆盖，原028终态及全部负结果保留。

新prepared继承028的96原源/快照与权重/域/数据引用，加11个新入口/测试及本plan/config共109源/109快照；准备仅核旧metadata/证据，不重开EP/PNG。新执行源码在真实运行前冻结，任何变化拒绝。配置与原028最终四份证据SHA显式传递，研究memo只作设计依据，不当实验成功证据。
