# 固定锚点的方向导数一致性（CPU，执行前登记）

总目标及原18正确参考/12WAM开发/24最终留出门槛保持，GPU禁用。此项检查不优化命令、不改变控制器、精度、原caps或安全buffer，不运行物理pilot，不读取旧actual369+作输入。全部数值调用仍从原actual368 q/v、固定finger及completed367因果latch/previous开始，使用已生成chunk46原H12及完整96资产模型，seed42。

## 依据与准入

[实际接纳点SQP](../../reports/accepted_point_sqp_20261029.md)的producer已正常关闭：1919完整响应、27个合格QP、24次曲率更新，prefix9、完整witness0。producer SHA `66b5b2f89905b7e8be17d92926ffe96015d2433c5f9ce9dc02860870b3d739da`，owner SHA `07301c0ac1fc413dba8f677a5bfbd123af707ed73589dd5e9a932fd807203da6`。完整独立复算当前仍在运行；只有原1919响应/575700内步/598729查询、118完整ray、25anchor、27QP与24次曲率更新全部复算及两owner正常wait/reap、最终来源闭合后，才允许本项的新响应。新入口严格核对同一52源QA及原checker/producer命令和结果绑定，不接受部分核验。

保存数据终端行诊断 SHA `951eb83b022938d1d908170c8bd8da32bca7a7687ad7b60cc6219e16963db3b9`：末三阶段共用最后接纳anchor1819和同一72轴单边FD；27ray均保持prefix9、降低目标，却被terminal[0]阻断。当前normals第一行为e7，故该行等于最终预测第七关节速度的负值。小alpha的实际-minus-linear残差除以alpha没有趋近零，有限采样不足以证明误差阶数或唯一原因。

固定FD步长截断、native float32旋转目标舍入、控制裁剪等不光滑性及浮点噪声尚未区分。保持原native目标计算与torque clip，不通过换成float64控制器制造更平滑的响应。pinv秩切换是否发生也尚未验证。

## 唯一有限请求表

命令锚点固定为原trace[1819]；它是待测命令轨迹，不是新的实测起点。三个方向固定取原phases[24,25,26].qp_direction.commands减该anchor所得实际float64位移d；核对保存displacement字节。每个方向计算一次s=||d||inf、u=d/s，保存完整d/s/u及三个u的两两无穷范数差，不能将近乎同向的方向当成独立样本。

固定command步长依次为`(1e-2,5e-3,1e-3,5e-4,1e-4,5e-5,1e-5,5e-6,1e-6,5e-7,1e-7,5e-8,1e-8)`。先计算一次anchor，再依stage24→25→26、步长由大到小、先正后负请求`x ± h*u`，最后强制重复一次anchor。总计80个计划请求，其中78个扰动请求。不追加步长、方向或重复，不自适应选点。

每个扰动先核原完整command box/rate及其1e-12容差，保存有符号slack和拒绝原因。非法请求明确跳过，不裁剪，不改为边界点，也不受旧stage半径限制。合法命令即使预测joint/pose不合格也完整保存，保持原严格分类。命令在本批中按float64字节缓存；不从旧producer缓存取代新响应。末尾anchor repeat绕过缓存、必须再次从actual368完整计算并单独计费，禁止其后再请求响应。

最多80次完整H12调用、最多79个唯一命令；每次300内步/312查询，另有一次current查询，因此最大24000内步/24961查询。重复调用与唯一命令数分别记录，缓存和非法请求不计调用。每个实际调用保留完整rollout、完整九组约束审计、安全向量、TCP误差及逐项归一化pose平方误差，不把最小安全值作为可微函数替代完整向量。计划不提前因偶然可行点停止；若出现完整witness，也只如实记录，不升级物理资格。

## 计算与解释

前向、后向、中心差分分别以登记的h、h、2h作分母，作用于完整安全向量及逐项pose q。中心差分必须有合法的双侧完整响应，且两侧实际位移均非零；非法或被float64舍入成零位移的侧明确标记不可用。保存实际浮点位移delta及`||delta ∓ h*u||inf/h`，不偷偷以某一坐标的实际位移替换分母。

从固定旧72轴FD的真实probe关联重建Js/Jq，保存`J*u`及每个实际位移的`J*delta`预测，与各完整响应变化比较。检查原source阶段、anchor、方向、FD符号、原精度值与字节身份；不同浮点独立重放的差异不能被冒称为原FD。全部新响应另按原native数值复算容差独立核验，strict joint/prefix/witness标签必须一致。

首次/末次anchor对所有保存数组逐项记录精确相等及最大差异；与旧anchor1819仅比较其实际保存的trace和anchor审计字段，不虚构旧完整rollout。若重复不一致，保存结果并标记方向导数解释受重复性影响，不重试、不平均。两次相同也不证明全部点没有量化或噪声。

结果按全部步长报告，不事后选出有利h并宣称算法成功。若若干步长上forward/backward/central稳定且共同偏离旧J*u，支持旧Jacobian的方向误差；若小步长呈跳变、平台或两侧不一致，保留截断、不光滑和量化等解释，不能预断减小h有效。可从已保存goals与clipped数组检查目标值/饱和模式变化，未保存的SVD谱不作排除性结论。本诊断不证明存在安全下降方向，也不检验二阶修正能否求得完整解。

## 实现与执行约束

新增独立core、evidence、producer/checker/owner及tests，不改已执行source/计划/结果。CPU原生QA必须覆盖解析方向导数、双侧缺失、量化平台、缓存/强制重复与完整响应计账，以及有效篡改拒绝；主checker独立重建请求和全部差分，不调用生产core作为唯一代数证据。全部实际调用包括末尾repeat都要独立重放，不能按唯一命令去重。

新QA、Ruff/format、producer/checker及owner实际dry与来源闭包全部通过后，才允许一次diagnose。producer限600秒、checker限1800秒，exclusive输出、无自动重试，全部CPU/空CUDA/LD及单线程。owner严格限制80调用与完整复算，所有新物理、world-model、fresh、参考消费均0。当前完整SQP checker尚未关闭，故此计划登记不构成执行许可已满足。
