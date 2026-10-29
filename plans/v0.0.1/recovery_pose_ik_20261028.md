# 恢复下一目标的静态可达性分离检查（CPU）

先可行后性能搜索在实际held endpoint378仍无resume9解，最好已检查线性可行尝试旋转误差.07532084839rad，不能称为全局不可行证明。本次检查同一个chunk46 cursor8/9是否存在有界静态构型，以区分单点几何存在性与实际当前状态/固定控制器/50ms前进的动态困难。不引入新初态、不改目标或任何原门槛。

在实际endpoint378的隔离XML数据上执行FK和IK，无物理步、无模型调用、无fresh。两个固定种子分支：实际当前q与原控制器initial_joint；分别沿cursor8→9顺序继承返回解。两组明确关节范围：原XML范围、原范围内缩.05rad，各2分支×2目标，总8次求解，每次最多200nfev，无随机或自适应重启。若初值需要裁剪则显式记录，不改变实际物理起点。

静态严格witness要求位置≤1e-5m、旋转≤1e-4rad且关节在所声明范围内。每个返回解用独立compiled XML标量关节FK重算姿态/残差/标签，不只依赖MuJoCo或solver success；以XML FK有限差分校验实际当前点TCP残差Jacobian。记录返回解距真实q、前一返回解的差异；差分不是最小必需运动/速度证明。无碰撞、夹爪协调、速度/力矩或固定控制器可执行性资格，即使静态全部通过也不能作为新action或任务成功。

入口`execute_tracking_v5_recovery_pose_ik.py --launch`与`probe_tracking_v5_recovery_pose_ik.py`。新outputs目录不覆盖；GPU继续禁用，原18/12WAM开发/final24门槛不变。当前结果待实际执行关闭。

8次固定查询已actual wait0/reap0完整关闭。实际种子分支在原范围及内缩.05rad范围均构造cursor9严格witness；内缩分支误差9.29328539e-6m/7.64504909e-5rad，原最小关节裕量.05025039461rad。独立XML FK最大姿态差2.22e-15、残差Jacobian差2.104e-10。verified SHA 658591bb08711b3ceee9125c25da96ffd6876281acfeb0c4f6bd3f14ba956ed4，物理/模型/fresh0。[完整结果与限制](../../reports/recovery_pose_ik_20261028.md)。不能以静态构型存在或返回构型差异证明固定接口的动态可达性；后续应研究实际当前状态锚定的提前构型调整及执行验证。
