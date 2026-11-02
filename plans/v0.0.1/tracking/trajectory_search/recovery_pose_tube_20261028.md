# 实际状态锚定的误差管过渡（CPU）

同一实际held endpoint378到原chunk46 cursor9已有严格静态IK构型，但不证明动态前进。本次用原.01m/.05rad误差管偏好较小关节变化，检查是否可构造实际起点锚定的几何候选，不改原目标/接口/时序或任务门槛。

固定实际q种子，仅cursor8→9顺序；原XML范围及内缩.05rad两组，共4次管查询，每次cold IK最多200nfev、管优化最多60迭代，无追加初值重启。每个返回点用独立XML FK重算残差与管/范围标签，不能以optimizer success替代实际接收检查。记录距实际q及前一返回解距离，不称全局最近或最小必需变化。

每组用257个预定等间距样本观察actual q到cursor9返回构型的关节直线段，逐点记录对cursor8/9误差、是否位于两管并集及关节裕量。不以离散采样证明整条连续路径/碰撞安全，不将关节配置直接下发为action；模型尚不含速度、固定OSC目标/历史或接触。只有后续同一固定接口产生实际动作并验证前向状态误差，才能判定可执行。

入口`execute_tracking_v5_recovery_pose_tube.py --launch`，probe同名前缀。XML-only CPU，无物理步、模型调用或fresh，GPU继续禁用。新outputs目录不覆盖。原18/完整12WAM开发/final24门槛不变；结果待实际执行。

已actual wait0/reap0关闭：两组cursor8均保持真实q；cursor9内缩.05候选距实际q最大分量.01995242885rad、范数.03450500517rad，原误差管合格；但两段257样本均未全部位于前后管并集。独立XML FK差1.55e-15，verified SHA 8edb7d730279cf3574216c64dd6415af397ad20d7964d26173b7d43ac1e36514。相关9测试通过，新增物理/模型/fresh0。[结果与限制](../../reports/recovery_pose_tube_20261028.md)。
