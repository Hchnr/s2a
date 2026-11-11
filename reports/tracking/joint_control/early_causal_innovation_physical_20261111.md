# 首次位姿失准处的因果偏差单周期物理对照

2026-11-11，按[预登记](../../../plans/v0.0.1/tracking/joint_control/early_causal_innovation_physical_20261111.md)，
在 correct_08、seed 42 重建 80 周期正确参考前缀及 42 个反馈周期。前 42 条反馈
命令、实测状态、内环关节/速度和时间与此前已审核物理运行最大差均为 **0**。
复用六份历史 WAM 响应时，逐次核对当前图像、原生状态 SHA 和时间；本次没有
新增 WAM/GPU 推理。任务成功标签未作为求解或停止输入。

在 step121 的动作及观测完成后，从**本次运行**的 TCP 实测与预测误差计算上一
周期的位置和 SO(3) 偏差。step122 重新求解，候选通过原质量、关节、制动和动作
约束及独立的因果偏差质量审核，随后实际派发；未复制此前的静态候选。与此前
相同起点的实际命令相比，新机械臂命令最大差约 0.2841。

| step122 TCP 误差 | 位置 | 旋转 |
| --- | ---: | ---: |
| 历史运行实际值 | 0.0139328 m | 0.0222335 rad |
| 新候选原模型预测 | 0.0045626 m | 0.0239137 rad |
| 加上一周期因果偏差的预测 | 0.0100000000000004 m | 0.0218834 rad |
| 新候选实际物理值 | 0.0129522 m | 0.0258686 rad |

实际位置误差比历史值低约 0.000981 m，仍超过原 0.01 m 门；旋转在原 0.05 rad
门内但比历史值大。补偿后模型位置预测仅在原 `1e-10` m 容差内达门，没有实质
裕量；模型审核通过不能保证真实状态达门。该周期任务谓词为真，但单点任务
谓词不等同于稳定成功。新分支共 43 个反馈周期中 42 个实际位姿达门，末尾
任务谓词连续为真 36 个端点；这条轨迹按试验预算截断，不能与更长轨迹的末尾
计数直接比较。最小关节余量约 0.782593 rad，无跌破 0.05 rad 缓冲或越界内环样本。

新动作执行后，step123 由 `registered_one_period_limit` 在派发前停止。底层
通用停止原因字段写为 `quality_no_witness`，但此处的 solver status 明确表示
**预登记的一周期实验预算停止**，没有执行下一次质量搜索，不能解释为算法无解。
step122 的质量求解使用 71 次模型响应，独立审核通过一次。

独立 checker 未加载候选求解器，重放全部 43 条反馈动作及 1,075 个续段 2 ms
内环样本；状态、关节位置、速度和时间最大差均为 **0**，标签一致。producer 与
checker 各完成 123 控制周期、3,075 MuJoCo 内步；关联子进程均正常退出并实际
wait/reap。前六份历史响应对应的 210 次去噪属于此前运行，本次新增模型初始化、
请求和去噪均为零。只读审计重新计算了因果偏差、历史参考身份及完整动作重放；
checker 不重算求解器的停止原因。

运行前 64 并发相关专项 97 项通过，无跳过或收集错误；只读审计的 32 并发专项
11 项通过。本次仍是单一已研究初态上的开发物理对照：没有验证实时模型服务、
600 周期完整任务、原 18/12/24 留出验收或真机，不能授予控制资格。

| 仓库内证据 | SHA256 |
| --- | --- |
| [零物理 dry](../../../outputs/early_causal_innovation_physical_dry_20261111/result.json) | `514e34357ca4f7ef355b6ad0070d012ffd86986691a45fca83fd2cb167761cee` |
| [64 并发运行前专项](../../../outputs/qa/early_causal_innovation_physical_20261111/result.json) | `5f5f697c22aff6921df727482e7a2419315f5f9aa0a14b81b20ac42a800fa1bf` |
| [producer owner](../../../outputs/early_causal_innovation_physical_20261111/producer/result.json) | `d8da5c619ec6ba2eddfa6ab4defad1f277f3883d68fe4e20650777dac02c896b` |
| [producer 物理结果](../../../outputs/early_causal_innovation_physical_20261111/producer/simulation/result.json) | `bd5a83ab94e8727bef15249b05fcf33d960df664a4f456959123f0408a670372` |
| [历史响应 relay](../../../outputs/early_causal_innovation_physical_20261111/producer/model/result.json) | `d1d1c441d220ebc187859ed9345a55ad7ead6cf51d68e7fba5ae41f535be3042` |
| [checker owner](../../../outputs/early_causal_innovation_physical_20261111/checker/result.json) | `beea8cd09b39e39a65585c008b74c33657b929c9a9b958ab0d81f618067a4137` |
| [checker 物理结果](../../../outputs/early_causal_innovation_physical_20261111/checker/simulation/result.json) | `56d7342d5c3b10ccb81b4d2ca6b68c36666d0267dc75bc274cc5873c8a61d87e` |
| [只读来源与重放审计](../../../outputs/early_causal_innovation_physical_audit_20261111/result.json) | `3745490d142435203f76f7257e2e75319a6075ac94ee78a229de9c69ed4b1fee` |
| [32 并发审计专项](../../../outputs/qa/early_causal_innovation_physical_audit_20261111/result.json) | `94d73720a98782ceebf011c27add6b14a26e7d161b82fb82b8a428243fd45a32` |
