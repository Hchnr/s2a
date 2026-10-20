# 最新15600的同协议直接state核验 — 2026-10-21

009固定48条state预测全部完成并通过独立核验。15600在这组开发数据的H8/H16/H40位置、SO(3)、opening九项池化指标均差于15400，也均差于8400；没有新的短前缀优势，不能因为它是latest就替换原候选。这个判断限定于六个已知开发episode，不证明全局训练退化或任务能力排名。

本轮新增的是48条真实模型预测、0物理任务步。模型由root按[009预算](../plans/v0.0.1/009_latest15600_state_20261021.md)启动，审计程序不调用模型或仿真；原四checkpoint summary及旧原始结果均保持原字节。新初态与fresh性能未打开，不能将本报告当作WAM闭环成功证据。

## 协议与身份

新权重为23-source/FastUMI continuation的非EMA `iter_000015600/model`，与15400同run/config。root在budget commit `0a248f2`固定：原24个diagnostic phase样本、6个episode、22个不同锚点、sampling seed42/43、task-only、H40/20Hz、35去噪步、guidance1.0、shift5及十有效槽mask；物理GPU6/port9166。完整权重来源、99文件首次byte指纹、EMA未启用及采样源码边界另见[权重/采样审查](wam_model_sampling_review_20261021.md)。

每个state误差从当前发布实测canonical TCP积分局部delta，与有效未来state比较，完全绕过S2A。位置米、朝向SO(3)测地角、opening投影[0,1]后计误差；报告位置换算为mm。H8/H16/H40分别384/768/1778有效endpoint；5/24样本未来不足40步，末端不足部分不padding计分。阶段为机器人state proxy，sampling seed与重复phase不是独立episode；同锚点/seed重复预测也不能扩充独立样本数。

## 完整配对结果

表中每格为“位置RMSE mm / SO(3) RMSE rad / opening MAE”。五个checkpoint均同一批24样本与48请求口径，非每query RMSE再平均。

| checkpoint，guidance1.0 | H8，384端点 | H16，768端点 | H40，1778端点 |
| --- | --- | --- | --- |
| 8400 | 24.384 / .088644 / .058538 | 44.623 / .177223 / .086038 | 85.521 / .479894 / .114660 |
| 8600 | 27.996 / .113729 / .067291 | 50.113 / .235302 / .102860 | 103.871 / .546207 / .118672 |
| 8800 | 29.297 / .105865 / .073929 | 47.812 / .217048 / .116804 | 82.414 / .542640 / .146364 |
| 15400 | 25.902 / .096772 / .067205 | 46.237 / .198837 / .097410 | 89.913 / .507355 / .116766 |
| **15600** | **29.185 / .104531 / .075429** | **52.861 / .221368 / .125687** | **98.696 / .563030 / .140309** |

8400仍在H8/H16的三项指标最低；H40位置最低仍为8800，旋转与opening最低为8400。15600不能据本表列为改进候选；也不将两个相邻checkpoint在这六个episode上的差解释成所有训练数据/任务上的退化。

15600分任务结果如下，每任务两个episode、16请求，分别按自己的有效未来计分：

| 任务 | H8：位置/旋转/opening | H16：位置/旋转/opening | H40：位置/旋转/opening |
| --- | --- | --- | --- |
| bowl→plate | 28.967 / .110405 / .080426 | 50.951 / .237448 / .132819 | 86.450 / .502500 / .116632 |
| wine→rack | 38.957 / .125015 / .135724 | 63.751 / .271028 / .227263 | 115.217 / .757299 / .221813 |
| cream→bowl | 14.093 / .070442 / .010136 | 41.504 / .131051 / .016979 | 91.684 / .367102 / .085116 |

这些是单示范未来state误差，不是唯一可行任务轨迹，也不是接触、抓取或放置成功率。

## 独立核验与服务释放

[新审计入口](../scripts/audit_wam_latest_state.py)在`1bfca97`提交，定向Ruff与格式检查通过。先通过既有`verify_cell`逐请求复核冻结配置/原始观测、模型路径及seed、服务mask、请求参数、响应SHA、domain/fps/反归一化、有效未来、直接state误差及保持/历史外推基线。首33条快照明确complete=false，未用于排名；只有48条及正常release齐全后生成最终文件。

第二条计算路径使用现有独立显式Gram–Schmidt、R/t逐delta递推和SciPy SO(3)，不调用S2A几何/指标函数，重建全部48条误差与三前缀汇总。对保存逐endpoint误差最大差为位置`1.6653e-16m`、旋转`8.8818e-16rad`、opening `0`。旧四组比较复用其已完成独立审计，642个旧输入SHA重新核验，并逐字段确认样本完全一致；本次共798个输入SHA在读取前后重核。

新15600完整fingerprint的99文件size/mtime/inode/device在审计前后全部一致，没有重复hash32GB权重。模型PID89230正常SIGTERM退出（returncode−15），state worker90662退出0，均无残留PID或orchestration error；服务释放记录已进入最终证据。审计没有停止其他正在运行的005/006服务。

最终 [complete48_verified.json](../outputs/wam_stability_20261021/latest15600_audit/complete48_verified.json) SHA为 `c36d3f5e44de8a39f898dc4cb2cbe0a09c4c3d5f6a5287c6c04326359ca6388a`，`complete=true`、`verified_queries=48`、`state_matrix_complete=true`、`owned_service_released=true`。root启动预算、完整权重指纹、源输入、按任务/阶段指标及release原始字段均绑定在文件内；原4-cell summary未覆盖。

下一步不因15600较新而追加它的闭环矩阵。短重查询候选仍应依据完整旧模型state证据及独立执行资格另行冻结，闭环计分继续使用原生任务谓词和预注册预算；不根据任务成功瞬间增加停臂或任务规划规则。
