# 103 原30轨迹的当前wrench数据

102六baseline直接传感器合同通过。按087原30轨迹顺序（24fit/6holdout）完整重放，原初态/seed42/全部命令/固定控制器/80周期settled goal保持。总6180显式控制周期、6210端点；每endpoint在原刷新后、新命令前只读采集local及world-at-control-SITE wrench，保留sensor pose、位置、时间/名称/type/地址。不得从不含原ctrl/warmstart的旧snapshot重算力代替实测，不使用未来wrench作为当前条件。

每行完整sim_state/SITE pose与原始native数组匹配；原始数组无twist，使用087已独立XML和数值差分资格通过的x_site_twist拼接末行y_twist作为N+1参考，核对绑定和SHA。getter/buffer、读取只读、变换、时刻及SI约定沿用102；不修改命令。原30metadata record/prepared_ref/source/cache均指纹冻结，拆分身份不变。采集全部合格及60worker/checker+owner实际wait/reap、来源关闭后才允许用于后续fit。

只采集30native，不重复189个probe前缀：后续若加入wrench条件，应按同native SHA/anchor从原因果前缀取当前wrench；probe未来端点的wrench不进入条件。原086/097探针state/命令/goal一致性仍需重新核对，不能仅凭同task绑定。

生成产物outputs/native_wrench_data_20261025。CPU/OSMesa显式空CUDA/LD。6180周期对应154500积分子步，30次初始化内部步骤另列；世界模型/fresh/fit0。当前仍无模型或控制资格，原18/18正确参考、WAM开发/最终24新初态门槛保持。
