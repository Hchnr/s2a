# 已接收模型解的关节/末端一致性（CPU）

长窗口与线性可行起点仍未产生12/16合规模型解。独立核对已接收8点解时观察到冻结J积分的Cartesian pose与同一预测q的真实几何FK不同，可能使“端点精度合格”失去内部一致性。本次固定只检查已经保存的同一actual368状态、同一8点模型witness，不搜索新命令或改任何目标/门槛。

用原compiled XML标量关节FK重建actual368 SITE pose；7列1e-6rad中心差分对当时实测world SITE Jacobian校验。逐预测端点把该解自己的joint_position写入隔离的raw state副本，再以XML FK计算TCP，对照模型Cartesian pose、原目标和保存all_tcp_errors，记录内部几何差异及两种精度标签。不将预测q的FK当作真实未来物理状态，不声称动态/接触误差已测出。

新增物理、模型调用、fresh及优化搜索均0；无新action或任务资格。入口`execute_tracking_v5_preview_fk_consistency.py --launch`，新outputs保留原实验。GPU继续禁用，原18/完整12WAM开发/final24门槛不变，结果待实际关闭。

已actual wait0/reap0关闭：模型精度8/8标签合格，但同一预测q的独立FK只有5/8满足原管；最大内部pose差.01701236716m/.04484467660rad，FK对目标最大误差.01820815649m/.05032597604rad。实际初点XML Jacobian与当时实测差1.5026e-10，verified SHA 1d06af338652019c1c8d1624e5ee6dee9b5c5af0d557d0a23e9b98d545d00b03。[完整证据和范围](../../reports/preview_fk_consistency_20261028.md)。检查发现不一致，不是实际未来轨迹或新控制资格；下一候选需用预测q的FK及其一致导数重建pose/twist。
