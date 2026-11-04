# anchor85联动完整局部模型采集

2026-11-02，seed42，接续23a1456/1e6f9f8设计与实现，GPU禁用，CPU32worker。
总目标、原18/12/24及稳定验收保持，此数值辨识不授予物理或控制资格。

固定anchor85与已命令域检查的72个联动方向，逐值绑定既有coupled_basis保存产物。
原box/rate/global/trust与1e-4幅度不变，不根据响应结果补选或重排。
顺序anchor、anchor repeat、72方向；按index modulo32分批，每worker2或3响应，
每worker独占batch/tmp目录与独立动力学model，实际进程wait/reap并登记退出状态。
生产32model各一次初始query，74响应/22200内步/23120查询；独立1model初始query，
74完整复算/22200内步/23089查询；总33model、148响应/44400内步/46209查询。
0QP/NNLS，3生产Js/Jq/Je因果线性拟合组+3独立正规方程重建组，共6线性拟合组。
不追加旧60stage预算，不覆盖旧单轴stencil或失败数据。

生产固定replay-order预测器、native_solve TCP、初态/因果历史和已生成参考，原完整
响应及metric布尔/1e-10、torque1e-8审核，独立响应要求逐值exact、initial/repeat
严格exact。非线性不安全探针如实保留；它们仅数值辨识，绝不下发或算成功。

重建直接在各输出row的命令因果前缀参数内求解，不先拟合全维再截断。
生产lstsq与独立normal equations比较原1e-10；报告完整有限步残差和prefix条件数。
额外检查同一命令前缀产生相同已观察输出，以及前缀位移0时输出差0，逐值检查；
这是本数值响应集的证据，不证明物理系统或任意未见方向的因果性/拟合精度。

使用当前QA/source/map/driver绑定，全程空CUDA/LD且数值库线程1。
owner1800秒TERM/10秒KILL，正常路径executor.shutdown(wait=True)+每process.join，
异常退出不得重试；worker异常费用未知须明确标记，不能据通过记录漏计。
超时整体失败，所有owned processes实际回收后再登记终态。

保存完整每响应/独立审核、输入/源码SHA、选择/实际位移、原始新模型及拟合残差。
未见新方向的独立模型验证另行登记，不将训练残差替代验证。之后才比较完整目标
方向、reserve恢复与新物理验证资格，最终留出仍关闭。
