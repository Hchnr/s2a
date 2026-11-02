# 025 资源释放观测接续

在任何 readout fit 或96次条件生成前登记。原025唯一features模型已完整完成3份zero生成与144次teacher，共249次denoise，producer complete/passed及原参数冻结均为True。实际模型PID862631/start170222073的wait=0、owned reaped、未发信号；原owner PID862480因退出后10秒内未观察到GPU6显存为0而TimeoutError，original owner结果False及完整日志原样保留。没有独立features全组审计或fit/后续采样。

Root随后实际查询物理GPU6 UUID `GPU-7f176867-170c-b508-5c89-1cd819ad3b8e`，显存0MiB、compute-apps为空，两原birth均已不存在。这不能重建未持久化的原10秒逐次显存值，也不能确定原因；退出日志的NCCL destroy警告与此的因果关系未知。原025登记要求实际退出与资源释放，不含10秒这个效果判据。当前释放证据允许继续CPU审计，不能将原owner False改称True。

采用新隔离接续入口与新schema，明确 `continued_after_resource_observation_timeout`。原025 plan/config、prepared53源与快照、producer/owner失败/exit/release均保持字节不变。新入口及必要测试和本补充文档另外冻结SHA；同一原prepared/检查器/原权重及实际新的资源释放证明均绑定，不伪造旧owner正常完成。入口必须拒绝任何其它错误、非正常model退出、未回收、未完成249次或非零/不符UUID的当前GPU6。

按以下顺序只执行尚未执行的原025阶段：

1. 正式独立CPU features全组审计，重算144个真实teacher bundle及三份zero身份；不额外生成或读取新EP。审计必须通过才进入拟合。
2. 原冻结CPU fit入口，恰好原登记两组固定ridge与bias，对同一6train/6开发validation；原冻结独立checker重建全部参数/teacher门槛。保留所有固定阈值；失败不增加fit或重调参数。
3. 仅生产与独立teacher门槛均通过，才执行原冻结sampling worker的96成对生成/3360denoise，后续仍由原冻结CPU checker评分。GPU6锁/启动前严格0显存/实际owned子进程及wait/reap继续使用；子进程退出后的资源观测改为有界最多120秒，并持久化每次显存及时间。观测延长不重启模型、不追加NN调用、不向未知进程发送信号；仍不释放则保留失败。

接续不重跑features/zero、不覆盖任何已存在fit/采样/核验文件、不自动重试；总科学预算仍是3zero＋144single＋条件96生成=3609denoise，CPU fit2，物理/fresh0。两组fit及所有采样质量尚未知；CPU/source/资源核验不能替代任务稳定成功。后续物理及fresh仍须原state门槛和另外事前登记。
