# reserve拒绝数据的后验更新与停滞机制核查

2026-11-02。接续7c0379f实际有限对照。本轮没有新机器人响应、QP或物理实验。
证据改变下一步方向：先对齐QP的reserve下限与接纳门，不盲目增加拒绝反馈。

## 固定安全拒绝样本更新

重建reserve实际搜索stage1之前的Js/Jq/Je（含stage0原接纳更新），在首个完全
拒绝stage中按最小alpha、再response index选择joint/buffer安全且pose ceiling
合格的样本，选择不依据残差大小。得到response9、alpha .0625，step inf约
.0003125，anchor是该新搜索response4。

按原min squared step=1e-12、relative limit=2、row norm floor=1e-6分别更新
Js/Jq/Je，独立实现逐值重建一致，未来列严格保持0，无行触发幅度裁剪。
这是3次后验因果更新和3次独立软件重建，不混入已关闭搜索的计费或接纳记录。

| 检查 | 原最大误差 | 更新后最大误差 |
| --- | ---: | ---: |
| 拟合样本Js | 2.81329179e-8 | 4.06575815e-20 |
| 同方向alpha .125的Js | 3.34228411e-6 | 3.36747637e-6 |
| 同方向alpha .25的Js | 1.47903882e-5 | 1.48407727e-5 |
| 同方向alpha .5的Js | 6.32299305e-5 | 6.33306995e-5 |
| 同方向alpha1的Js | 1.61553896e-4 | 1.61755434e-4 |

拟合样本原误差已经很小。其他已观察尺度没有稳定改善，较大尺度略变差；
后续其他方向部分小尺度改善、部分变差，完整逐样本Js/Jq/Je记录均保留。
近零拟合残差是构造结果，不是未见方向验证，不支持直接把该反馈设为修复方案。

## 更直接的内部约束不一致

原interior_trajectory_search.build_stage_problem的pose safety floor是
`buffer + min(0.5*reserve, 0.5)*reserve_target/weights`。
新接纳策略却要求actual reserve≥.9。stage1以后anchor reserve固定为
.9285845109，原QP允许下降到.4642922555；所有stage1–6保存QP的预测最紧行
都贴住这个下限，即使trust radius不断缩小，也没有改掉这个目标范围。

保存的全部预测逐一符合（误差<1e-9）
`R_pred(alpha) = (1-alpha)*R + alpha*(0.5*R)`。
在该已观察活动约束上，模型要满足.9需alpha≤.0615657715，但原最小alpha=.0625。
最小步预测reserve=.8995662449，仍低于.9；它的实际值=.8977061272。
原样本误差对应reserve约.002，不能用消除此误差解决预测自身已经违反门的问题。

整个搜索24个实际reserve拒绝（stage0两次，后续停滞stage共22次），模型预测
全部已低于.9。以该活动行和最小alpha计算，需anchor reserve≥.9290322581才
有线性可接纳候选，而当前值略低。不是数值证书失败或未回收进程造成停滞，
也不通过改epsilon、调低门或追加步长来追认成功。

初版统计遍历stage1以后但错误按全搜索24次断言，实际应为22，断言失败记录的
程序保留floor_diagnostic_initial.py。最终统计遍历全部stage，24次检查通过；
没有改变源数据、原容差或实验结果。

## 计费、证据与下一步

全程CPU空CUDA/LD、数值库线程1。0新QP/NNLS/机器人响应/动力学查询/物理/GPU/
WAM/fresh/最终留出。仅保存数据重算，3次原有因果矩阵更新及独立软件比较。
没有修改算法或新跑测试套件；实际检查是来源SHA前后保持、独立更新逐值一致、
全部已观察预测的代数核对。Git空白检查通过。

| 产物 | outputs相对路径 | SHA256 |
| --- | --- | --- |
| 更新与逐样本误差 | reserve_rejection_shadow_20261102/result.json | 8a2db9e87e6d62ba8cb99c9148648c232824fe9a930ad0ae057ff8e34325a140 |
| 全stage下限/门诊断 | reserve_rejection_shadow_20261102/floor_diagnostic.json | 69eb690722a72d2733dea1fa99dcf54ee008e0e066f8c2c67161d07ee80ffcd7 |

下一步显式设计pose QP下限至少为原reserve entry条件的新变体，保留原实际安全/
接纳/完整witness和旧默认；不把内部约束收紧称为原算法已修复。先独立重建矩阵
与审核，再另登记新方向和有限迭代。原可靠跟踪及18/12/24稳定验收仍未完成。
