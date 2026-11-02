# 同响应上限的60阶段预算对照

本计划在新QP或模型响应前登记。前项固定初始全局上限搜索已完成35响应生产和35
响应完整独立复算，完整witness0；24stage先到而122响应中只用35。末stage22仍
降低maxq .0330507881、ratio1.00491438，stage23恢复后下一模式应为pose；相对
初始命令位移Linf仅.0181595063，未饱和人为±.1搜索盒。此证据支持检验阶段预算。

本次唯一算法配置改动为max_stages:24→60，max_responses仍122。它是增加QP预算
的独立对照，不是新控制算法，也不是相同全部预算。每QP仍最多100内部迭代，
最坏QP内部迭代上限从2400到6000；响应及数值查询上限保持，不能按60×5+2计成
302调用。最多120实际trial加初始和末次repeat，单路线最多36600数值内步/
38065查询，两路线最多244完整响应/73200数值内步/76130查询。

新namespace从最初3557及原72条signed1e-4差分重新构造Js/Jq，seen、半径与所有
搜索状态重新初始化，不从前项anchor33或旧优化器恢复，不挪用前项87剩余响应。
初始半径仍.005、最小.00015625、最大.02、全局命令位移.1；seed42及其余全部数值
配置不变。配置文件除max_stages token由24改60外逐字节相同。旧代码、输出及
失败证据保留原SHA，不修改冻结模块的全局validator。

搜索主体与前项相同，复用同一固定共同上限`max(1,max(initial_q))*1.01`（本case
6.977797141162663），运行期不放宽。反复余量恢复/pose模式、实际改善与ratio、
半径、因果有界逐行割线、原拒绝原因优先级及最小alpha反馈全部保持。旧audit已
支持显式动态阶段数，新薄wrapper独立限制stage<=60、response<=122，复用原
完整QP/KKT、状态机、因果更新、计费与选点核验，不复制或放宽数学证书。

实际/预测改善量、ratio>=.1和所有接纳条件不变。反馈只从无接纳stage的安全且
违反共同数值上限的actual_admission_failed记录选择；不选ratio_insufficient等
其它原因，也不另增反馈ratio门。初始mismatch/原witness先处理，stage后优先检查
witness、最小半径、响应预算；自然循环耗尽才stage_budget_exhausted。末repeat
必收费；repeat不一致保持资格失败。任何原停止条件满足即正常关闭，不自动追加
预算、重启、重算失败QP或放宽原1e-6证书。

新输入仍同actual368实测q/v、completed367命令与rotation latch、固定finger和
原历史profile，只使用当时已生成的原12参考。输入封装SHA256为
`eadf18a3ed90c29ee70d773741cc310304f4385189baea360e534ca6ef5bd023`。
原控制器、20Hz/2ms、TCP/SO(3)/单位、位置.01m/旋转.05rad、box/rate.25、关节
margin.05rad、braking.1s、terminal方向、原audit和1e-8 safety buffer保持。
不加载旧35响应、finalanchor或后验拟合为新模型数据；旧来源只作证据绑定。

比较资格要求：新旧前24份完整stage及旧requests/entries0..33（34条）均exact。
旧index34是强制末次初始repeat，新运行此时应继续stage24，不能错误要求全部35
响应作为连续前缀；该旧repeat与新最终初始repeat另作exact。预期stage23之后
anchor33、radius.00125、下一mode=pose。预算metadata中24与60应不同，不据此
判整个answer不等价。若出现未解释的前缀差异，保留数据并停止比较资格声明，
不以更长运行结果覆盖旧结果或事后放宽exact条件。

专项CPU QA、Ruff、格式及两条真实输入dry通过后先本地提交，再执行唯一新运行。
独立ledger完整重建每个阶段的QP/KKT与状态机，不再求解QP；冻结native独立路径
完整复算全部实际收费响应，严格重判原安全、共同搜索域、改善/ratio、反馈、
原fullcaps、历史与repeat。技术失败保留已计费前缀和异常，无自动重试。

报告必须同时比较F、总loss、cursor9、全部12双caps、前缀、位置/旋转、安全余量、
末段趋势、QP证书失败与所有实际成本。前项F下降同时总loss升14.06%，不能只用
最坏项下降推定可靠跟踪，更不能外推完整成功。若找到数值witness，仍需后续独立
实际前向控制与原任务门；本数值对照本身不增加物理/world-model/fresh/参考消费。

全部代码/配置/报告/产物维护于当前仓库，产物写outputs，不访问旧清理根或用户
未跟踪文件。CPU显式空CUDA和LD、单线程；GPU继续禁用，须等待用户明确恢复。
原实际仍cursor8/task0/grasp0，18正确参考/12WAM开发/24最终留出及完整总目标保持。
