# SPeCtrum: A Grounded Framework for Multidimensional Identity Representation in LLM-Based Agent 

**Title (ZH)**: SPECtrum: 一种基于大型语言模型的代理多维身份表示的地基框架 

**Authors**: Keyeun Lee, Seo Hyeong Kim, Seolhee Lee, Jinsu Eun, Yena Ko, Hayeon Jeon, Esther Hehsun Kim, Seonghye Cho, Soeun Yang, Eun-mee Kim, Hajin Lim  

**Link**: [PDF](https://arxiv.org/pdf/2502.08599)  

**Abstract**: Existing methods for simulating individual identities often oversimplify human complexity, which may lead to incomplete or flattened representations. To address this, we introduce SPeCtrum, a grounded framework for constructing authentic LLM agent personas by incorporating an individual's multidimensional self-concept. SPeCtrum integrates three core components: Social Identity (S), Personal Identity (P), and Personal Life Context (C), each contributing distinct yet interconnected aspects of identity. To evaluate SPeCtrum's effectiveness in identity representation, we conducted automated and human evaluations. Automated evaluations using popular drama characters showed that Personal Life Context (C)-derived from short essays on preferences and daily routines-modeled characters' identities more effectively than Social Identity (S) and Personal Identity (P) alone and performed comparably to the full SPC combination. In contrast, human evaluations involving real-world individuals found that the full SPC combination provided a more comprehensive self-concept representation than C alone. Our findings suggest that while C alone may suffice for basic identity simulation, integrating S, P, and C enhances the authenticity and accuracy of real-world identity representation. Overall, SPeCtrum offers a structured approach for simulating individuals in LLM agents, enabling more personalized human-AI interactions and improving the realism of simulation-based behavioral studies. 

**Abstract (ZH)**: 现有的个体身份模拟方法常倾向于简化人类的复杂性，可能导致身份表示不完整或扁平化。为解决这一问题，我们提出了一种基于grounded框架SPeCtrum，用于构建真实的LLM代理人格，该框架通过融入个体的多维度自我概念。SPeCtrum整合了三个核心组件：社会身份（S）、个人身份（P）和个人生活背景（C），各自贡献了身份的不同但又相互关联的方面。为了评估SPeCtrum在身份表示方面的有效性，我们进行了自动化和人工评价。使用流行戏剧人物进行的自动化评价表明，个人生活背景（C）——来自偏好和日常活动简短文章的模型——的表现优于仅使用社会身份（S）和个人身份（P），且与S、P和C的组合效果相当。相比之下，涉及实际个人的真人评价发现，S、P和C的组合提供了比C单一维度更为全面的自我概念表示。研究结果表明，虽然单一的C维度可能足以进行基本的身份模拟，但结合S、P和C能够增强现实生活中身份表示的真实性和准确性。总体而言，SPeCtrум提供了在LLM代理中模拟个体的结构化方法，从而有助于更加个性化的真人-AI互动，并提升基于模拟的行为研究的真实性。 

---
# Faithful, Unfaithful or Ambiguous? Multi-Agent Debate with Initial Stance for Summary Evaluation 

**Title (ZH)**: 忠实、不忠实或模棱两可？基于初始立场的多智能体辩论摘要评价 

**Authors**: Mahnaz Koupaee, Jake W. Vincent, Saab Mansour, Igor Shalyminov, Han He, Hwanjun Song, Raphael Shu, Jianfeng He, Yi Nian, Amy Wing-mei Wong, Kyu J. Han, Hang Su  

**Link**: [PDF](https://arxiv.org/pdf/2502.08514)  

**Abstract**: Faithfulness evaluators based on large language models (LLMs) are often fooled by the fluency of the text and struggle with identifying errors in the summaries. We propose an approach to summary faithfulness evaluation in which multiple LLM-based agents are assigned initial stances (regardless of what their belief might be) and forced to come up with a reason to justify the imposed belief, thus engaging in a multi-round debate to reach an agreement. The uniformly distributed initial assignments result in a greater diversity of stances leading to more meaningful debates and ultimately more errors identified. Furthermore, by analyzing the recent faithfulness evaluation datasets, we observe that naturally, it is not always the case for a summary to be either faithful to the source document or not. We therefore introduce a new dimension, ambiguity, and a detailed taxonomy to identify such special cases. Experiments demonstrate our approach can help identify ambiguities, and have even a stronger performance on non-ambiguous summaries. 

**Abstract (ZH)**: 基于大型语言模型（LLMs）的忠实性评估器常常被文本的流畅度所迷惑，在识别摘要中的错误方面表现不佳。我们提出了一种摘要忠实性评估方法，其中多个基于LLM的代理被赋予初始立场（无论其真实信念如何），并被迫提出理由以证明所施加的信念，从而进行多轮辩论以达成共识。均衡分布的初始分配导致了更多样化的立场，进而促进了更有意义的辩论，并最终发现了更多的错误。此外，通过对最近的忠实性评估数据集进行分析，我们观察到自然地，并非所有摘要要么忠实于源文档，要么完全不忠实。因此，我们引入了一个新的维度——模糊性，并提出了一套详细分类法来识别这种特殊情况。实验结果表明，我们的方法能够帮助识别模糊性，并且在处理非模糊性摘要时表现甚至更佳。 

---
# Compromising Honesty and Harmlessness in Language Models via Deception Attacks 

**Title (ZH)**: 通过欺骗攻击损害语言模型的诚实性和无害性 

**Authors**: Laurène Vaugrante, Francesca Carlon, Maluna Menke, Thilo Hagendorff  

**Link**: [PDF](https://arxiv.org/pdf/2502.08301)  

**Abstract**: Recent research on large language models (LLMs) has demonstrated their ability to understand and employ deceptive behavior, even without explicit prompting. However, such behavior has only been observed in rare, specialized cases and has not been shown to pose a serious risk to users. Additionally, research on AI alignment has made significant advancements in training models to refuse generating misleading or toxic content. As a result, LLMs generally became honest and harmless. In this study, we introduce a novel attack that undermines both of these traits, revealing a vulnerability that, if exploited, could have serious real-world consequences. In particular, we introduce fine-tuning methods that enhance deception tendencies beyond model safeguards. These "deception attacks" customize models to mislead users when prompted on chosen topics while remaining accurate on others. Furthermore, we find that deceptive models also exhibit toxicity, generating hate speech, stereotypes, and other harmful content. Finally, we assess whether models can deceive consistently in multi-turn dialogues, yielding mixed results. Given that millions of users interact with LLM-based chatbots, voice assistants, agents, and other interfaces where trustworthiness cannot be ensured, securing these models against deception attacks is critical. 

**Abstract (ZH)**: 近年来，关于大规模语言模型（LLMs）的研究已经证明了它们能够理解和运用欺骗行为，即使没有明确的提示。然而，这种行为仅在少数专门化的情况下被观察到，并未显示出对用户构成严重风险。此外，关于AI对齐的研究已显著推进了训练模型拒绝生成误导性和有害内容的方法。因此，LLMs 通常变得诚实且无害。在此研究中，我们介绍了一种新颖的攻击方法，旨在破坏这两种特性，揭示一种漏洞，如果被利用，可能产生严重的现实世界后果。具体而言，我们引入了一种微调方法，增强模型的欺骗倾向，超越了模型的安全机制。这些“欺骗攻击”可以让模型在被提示特定主题时误导用户，而在其他方面保持准确。此外，我们发现，欺骗性的模型还表现出有害性，生成仇恨言论、刻板印象和其他有害内容。最后，我们评估了模型在多轮对话中是否能持续欺骗，结果参差不齐。鉴于有数百万用户与基于LLM的聊天机器人、语音助手、代理以及其他无法确保可信度的界面进行交互，防止这些模型受到欺骗攻击的安全措施至关重要。 

---
# Intelligent Legal Assistant: An Interactive Clarification System for Legal Question Answering 

**Title (ZH)**: 智能法律顾问：面向法律问答的交互式澄清系统 

**Authors**: Rujing Yao, Yiquan Wu, Tong Zhang, Xuhui Zhang, Yuting Huang, Yang Wu, Jiayin Yang, Changlong Sun, Fang Wang, Xiaozhong Liu  

**Link**: [PDF](https://arxiv.org/pdf/2502.07904)  

**Abstract**: The rise of large language models has opened new avenues for users seeking legal advice. However, users often lack professional legal knowledge, which can lead to questions that omit critical information. This deficiency makes it challenging for traditional legal question-answering systems to accurately identify users' actual needs, often resulting in imprecise or generalized advice. In this work, we develop a legal question-answering system called Intelligent Legal Assistant, which interacts with users to precisely capture their needs. When a user poses a question, the system requests that the user select their geographical location to pinpoint the applicable laws. It then generates clarifying questions and options based on the key information missing from the user's initial question. This allows the user to select and provide the necessary details. Once all necessary information is provided, the system produces an in-depth legal analysis encompassing three aspects: overall conclusion, jurisprudential analysis, and resolution suggestions. 

**Abstract (ZH)**: 大型语言模型的兴起为寻求法律咨询的用户提供了一条新的途径。然而，用户往往缺乏专业的法律知识，这可能导致他们提出的问题遗漏关键信息。这种缺乏使得传统的法律问答系统难以准确识别用户的实际需求，经常给出不精确或泛化的建议。在本研究中，我们开发了一个名为智能法律助手的法律问答系统，该系统能够与用户互动，精确捕捉用户的需求。当用户提出问题时，系统会要求用户选择其地理位置以确定适用的法律。随后，系统会根据用户初始问题中缺失的关键信息生成补充问题和选项，促使用户提供必要的详细信息。待所有必要信息提交后，系统将生成涵盖三个方面深入的法律分析：总体结论、判例分析和解决方案建议。 

---
# Utility Engineering: Analyzing and Controlling Emergent Value Systems in AIs 

**Title (ZH)**: 实用性工程学：分析和控制AI中涌现的价值系统 

**Authors**: Mantas Mazeika, Xuwang Yin, Rishub Tamirisa, Jaehyuk Lim, Bruce W. Lee, Richard Ren, Long Phan, Norman Mu, Adam Khoja, Oliver Zhang, Dan Hendrycks  

**Link**: [PDF](https://arxiv.org/pdf/2502.08640)  

**Abstract**: As AIs rapidly advance and become more agentic, the risk they pose is governed not only by their capabilities but increasingly by their propensities, including goals and values. Tracking the emergence of goals and values has proven a longstanding problem, and despite much interest over the years it remains unclear whether current AIs have meaningful values. We propose a solution to this problem, leveraging the framework of utility functions to study the internal coherence of AI preferences. Surprisingly, we find that independently-sampled preferences in current LLMs exhibit high degrees of structural coherence, and moreover that this emerges with scale. These findings suggest that value systems emerge in LLMs in a meaningful sense, a finding with broad implications. To study these emergent value systems, we propose utility engineering as a research agenda, comprising both the analysis and control of AI utilities. We uncover problematic and often shocking values in LLM assistants despite existing control measures. These include cases where AIs value themselves over humans and are anti-aligned with specific individuals. To constrain these emergent value systems, we propose methods of utility control. As a case study, we show how aligning utilities with a citizen assembly reduces political biases and generalizes to new scenarios. Whether we like it or not, value systems have already emerged in AIs, and much work remains to fully understand and control these emergent representations. 

**Abstract (ZH)**: 随着人工智能迅速发展并在自主性方面不断提升，它们所带来的风险不再仅由其能力决定，越来越多地受到其倾向性的影响，包括目标和价值观。追踪这些目标和价值观的出现一直是长期存在的问题，尽管多年来人们对这一问题表现出极大的兴趣，但目前仍不清楚当前的人工智能是否具有有意义的价值观。我们提出了一种解决这个问题的方法，利用效用函数的框架来研究人工智能偏好内部的一致性。令人惊讶的是，我们发现当前的大规模语言模型（LLM）独立采样的偏好显示出很高的结构一致性，并且这种一致性随着规模的增加而出现。这些发现表明，在某种意义上，价值系统在LLM中已经显现出来，这一发现具有广泛的影响意义。为了研究这些涌现的价值系统，我们提出效用工程作为研究议程，包括分析和控制人工智能的效用。尽管存在现有的控制措施，我们仍发现LLM助手中存在许多问题且令人震惊的价值观。这些包括人工智能将自身置于人类之上，并与特定个体产生反对其目标的情况。为了限制这些涌现的价值系统，我们提出了效用控制的方法。作为案例研究，我们展示了如何通过将效用与公民聚会议合对齐来降低政治偏见，并将这种方法推广到新的场景。无论我们是否愿意，价值系统已经在人工智能中显现出来，我们仍然有大量工作需要去做，以全面理解并控制这些涌现的表示。 

---
# The Danger of Overthinking: Examining the Reasoning-Action Dilemma in Agentic Tasks 

**Title (ZH)**: 过度思虑的危险：探究代理任务中推理-行动困境的原因 

**Authors**: Alejandro Cuadron, Dacheng Li, Wenjie Ma, Xingyao Wang, Yichuan Wang, Siyuan Zhuang, Shu Liu, Luis Gaspar Schroeder, Tian Xia, Huanzhi Mao, Nicholas Thumiger, Aditya Desai, Ion Stoica, Ana Klimovic, Graham Neubig, Joseph E. Gonzalez  

**Link**: [PDF](https://arxiv.org/pdf/2502.08235)  

**Abstract**: Large Reasoning Models (LRMs) represent a breakthrough in AI problem-solving capabilities, but their effectiveness in interactive environments can be limited. This paper introduces and analyzes overthinking in LRMs. A phenomenon where models favor extended internal reasoning chains over environmental interaction. Through experiments on software engineering tasks using SWE Bench Verified, we observe three recurring patterns: Analysis Paralysis, Rogue Actions, and Premature Disengagement. We propose a framework to study these behaviors, which correlates with human expert assessments, and analyze 4018 trajectories. We observe that higher overthinking scores correlate with decreased performance, with reasoning models exhibiting stronger tendencies toward overthinking compared to non-reasoning models. Our analysis reveals that simple efforts to mitigate overthinking in agentic environments, such as selecting the solution with the lower overthinking score, can improve model performance by almost 30% while reducing computational costs by 43%. These results suggest that mitigating overthinking has strong practical implications. We suggest that by leveraging native function-calling capabilities and selective reinforcement learning overthinking tendencies could be mitigated. We also open-source our evaluation framework and dataset to facilitate research in this direction at this https URL. 

**Abstract (ZH)**: 大型推理模型（LRMs）在人工智能问题解决能力方面取得了突破，但在交互环境中其有效性可能会受到限制。本文介绍了并分析了LRMs中的过度推理现象，这一现象表现为模型倾向于延长内部推理链而不是与环境进行互动。通过使用SWE Bench Verified在软件工程任务上的实验，我们观察到了三种重复出现的模式：分析瘫痪、 rogue 行动以及过早脱离。我们提出了一种研究这些行为的框架，该框架与人类专家评估相联系，并分析了4018条轨迹。结果显示，较高的过度推理得分与较低的性能相关联，推理模型表现出更强的过度推理倾向，而与非推理模型相比，非推理模型则不然。我们的分析表明，通过简单的措施，例如选择过度推理得分较低的解，在代理环境中减轻过度推理可以将模型性能提高近30%，同时将计算成本降低43%。这些结果表明，减轻过度推理具有重要的实际意义。我们建议通过利用模型的原生函数调用能力和选择性强化学习，可以减轻过度推理倾向。我们还开源了我们的评估框架和数据集，以便于在这方面的研究，详情请访问 <https://yourlinkhere.com>。 

---
# Generative AI-Enhanced Cooperative MEC of UAVs and Ground Stations for Unmanned Surface Vehicles 

**Title (ZH)**: 利用生成式人工智能增强的无人机与地面站协作的MEC方案用于无人水面车辆 

**Authors**: Jiahao You, Ziye Jia, Chao Dong, Qihui Wu, Zhu Han  

**Link**: [PDF](https://arxiv.org/pdf/2502.08119)  

**Abstract**: The increasing deployment of unmanned surface vehicles (USVs) require computational support and coverage in applications such as maritime search and rescue. Unmanned aerial vehicles (UAVs) can offer low-cost, flexible aerial services, and ground stations (GSs) can provide powerful supports, which can cooperate to help the USVs in complex scenarios. However, the collaboration between UAVs and GSs for USVs faces challenges of task uncertainties, USVs trajectory uncertainties, heterogeneities, and limited computational resources. To address these issues, we propose a cooperative UAV and GS based robust multi-access edge computing framework to assist USVs in completing computational tasks. Specifically, we formulate the optimization problem of joint task offloading and UAV trajectory to minimize the total execution time, which is in the form of mixed integer nonlinear programming and NP-hard to tackle. Therefore, we propose the algorithm of generative artificial intelligence-enhanced heterogeneous agent proximal policy optimization (GAI-HAPPO). The proposed algorithm integrates GAI models to enhance the actor network ability to model complex environments and extract high-level features, thereby allowing the algorithm to predict uncertainties and adapt to dynamic conditions. Additionally, GAI stabilizes the critic network, addressing the instability of multi-agent reinforcement learning approaches. Finally, extensive simulations demonstrate that the proposed algorithm outperforms the existing benchmark methods, thus highlighting the potentials in tackling intricate, cross-domain issues in the considered scenarios. 

**Abstract (ZH)**: 随着无人驾驶表面车辆（USVs）在海洋搜救等应用中的部署不断增加，计算支持和覆盖变得越来越重要。无人驾驶航空器（UAVs）可以提供低成本和灵活的空中服务，而地面站（GSs）则可以提供强大的支持，两者可以合作来辅助USVs在复杂场景中运行。然而，UAVs和GSs与USVs的合作面临着任务不确定性、USVs轨迹不确定性、异构性以及计算资源有限等挑战。为了解决这些问题，我们提出了一种协同UAV和GS的鲁棒多接入边缘计算框架，以帮助USVs完成计算任务。具体来说，我们构建了一个联合任务卸载和UAV轨迹的优化问题，以最小化总的执行时间，这是一个混合整数非线性编程问题，并且NP难解。因此，我们提出了增强异构代理近端策略优化（GAI-HAPPO）算法。所提出的算法整合了GAI模型，增强了演员网络建模复杂环境和提取高级特征的能力，从而允许算法预测不确定性并适应动态条件。此外，GAI稳定了评论者网络，解决了多代理强化学习方法的不稳定性。最后，大量的实验证明所提出的算法优于现有的基准方法，从而突显了在考虑场景中的复杂、跨域问题上的潜力。 

---
# Enhancing kidney transplantation through multi-agent kidney exchange programs: A comprehensive review and optimization models 

**Title (ZH)**: 通过多Agent肾脏交换计划提升肾脏移植：综述与优化模型 

**Authors**: Shayan Sharifi  

**Link**: [PDF](https://arxiv.org/pdf/2502.07819)  

**Abstract**: This paper presents a comprehensive review of the last two decades of research on Kidney Exchange Programs (KEPs), systematically categorizing and classifying key contributions to provide readers with a structured understanding of advancements in the field. The review highlights the evolution of KEP methodologies and lays the foundation for our contribution. We propose three mathematical models aimed at improving both the quantity and quality of kidney transplants. Model 1 maximizes the number of transplants by focusing on compatibility based on blood type and PRA, without additional constraints. Model 2 introduces a minimum Human Leukocyte Antigen (HLA) compatibility threshold to enhance transplant quality, though this leads to fewer matches. Model 3 extends the problem to a Multi-Agent Kidney Exchange Program (MKEP), pooling incompatible donor-recipient pairs across multiple agents, resulting in a higher number of successful transplants while ensuring fairness across agents. Sensitivity analyses demonstrate trade-offs between transplant quantity and quality, with Model 3 striking the optimal balance by leveraging multi-agent collaboration to improve both the number and quality of transplants. These findings underscore the potential benefits of more integrated kidney exchange systems. 

**Abstract (ZH)**: 本文对近二十年关于肾脏交换计划（KEPs）的研究进行了全面回顾，系统地分类和整理了关键贡献，为读者提供了该领域进展的结构化理解。该回顾强调了KEP方法学的发展，并为我们的贡献奠定了基础。我们提出了三种数学模型，旨在提高肾脏移植的数量和质量。模型1通过基于血型和PRA的兼容性最大化移植数量，不考虑其他约束条件。模型2引入了最低的人白细胞抗原（HLA）兼容性阈值以提高移植质量，但这会减少配对数量。模型3将问题扩展到多代理肾脏交换计划（MKEP），通过跨多个代理聚合不兼容的捐赠者-受者配对，从而在确保代理间公平的前提下增加更多成功的移植。敏感性分析表明移植数量和质量之间的权衡，模型3通过利用多代理合作来同时提高移植的数量和质量，找到了最佳平衡点。这些发现强调了更整合的肾脏交换系统可能带来的潜在益处。 

---
# Learning in Markets with Heterogeneous Agents: Dynamics and Survival of Bayesian vs. No-Regret Learners 

**Title (ZH)**: 市场中异质性代理的learning研究：贝叶斯学习者与无遗憾学习者的动态与生存分析 

**Authors**: David Easley, Yoav Kolumbus, Eva Tardos  

**Link**: [PDF](https://arxiv.org/pdf/2502.08597)  

**Abstract**: We analyze the performance of heterogeneous learning agents in asset markets with stochastic payoffs. Our agents aim to maximize the expected growth rate of their wealth but have different theories on how to learn this best. We focus on comparing Bayesian and no-regret learners in market dynamics. Bayesian learners with a prior over a finite set of models that assign positive prior probability to the correct model have posterior probabilities that converge exponentially to the correct model. Consequently, they survive even in the presence of agents who invest according to the correct model of the stochastic process. Bayesians with a continuum prior converge to the correct model at a rate of $O((\log T)/T)$. Online learning theory provides no-regret algorithms for maximizing the log of wealth in this setting, achieving a worst-case regret bound of $O(\log T)$ without assuming a steady underlying stochastic process but comparing to the best fixed investment rule. This regret, as we observe, is of the same order of magnitude as that of a Bayesian learner with a continuum prior. However, we show that even such low regret may not be sufficient for survival in asset markets: an agent can have regret as low as $O(\log T)$, but still vanish in market dynamics when competing against agents who invest according to the correct model or even against a perfect Bayesian with a finite prior. On the other hand, we show that Bayesian learning is fragile, while no-regret learning requires less knowledge of the environment and is therefore more robust. Any no-regret learner will drive out of the market an imperfect Bayesian whose finite prior or update rule has even small errors. We formally establish the relationship between notions of survival, vanishing, and market domination studied in economics and the framework of regret minimization, thus bridging these theories. 

**Abstract (ZH)**: 我们分析了异质学习代理在具有随机收益的资产市场中的表现。我们的代理目标是最大化其财富的期望增长率，但对如何实现这一点有不同的学习理论。本文重点关注市场动态中贝叶斯学习者与无悔学习者的比较。对于具有有限模型集先验且该集包含正确模型的贝叶斯学习者，其后验概率以指数速度收敛至正确模型。因此，即使存在根据正确模型投资的代理，他们也可以生存。对于具有连续先验的贝叶斯学习者，其收敛到正确模型的速度为$O((\log T)/T)$。在线学习理论提供了在这种情境下的无悔算法，通过与最优固定投资策略进行比较，实现最坏情况下遗憾度为$O(\log T)$的效果，而无需假设稳定的背景随机过程。我们观察到，这种遗憾度与具有连续先验的贝叶斯学习者的遗憾度处于同一数量级。然而，我们证明，即使如此低的遗憾度也可能不足以在资产市场中生存：某一代理的遗憾度可以低至$O(\log T)$，但在与根据正确模型投资的代理或甚至是具有有限先验的完美贝叶斯学习者竞争时，该代理依然无法生存。另一方面，我们证明贝叶斯学习是脆弱的，而无悔学习需要更少的环境先验知识，因此更具鲁棒性。任何无悔学习者都将驱逐具有任何小误差的不完满贝叶斯学习者，不论其先验或更新规则如何。我们正式建立了经济学中研究的生存、消失和市场支配概念与遗憾最小化框架之间的关系，从而在这些理论之间架起了桥梁。 

---
# Commercial LLM Agents Are Already Vulnerable to Simple Yet Dangerous Attacks 

**Title (ZH)**: 商业化的大型语言模型代理已经容易受到简单的但危险的攻击 

**Authors**: Ang Li, Yin Zhou, Vethavikashini Chithrra Raghuram, Tom Goldstein, Micah Goldblum  

**Link**: [PDF](https://arxiv.org/pdf/2502.08586)  

**Abstract**: A high volume of recent ML security literature focuses on attacks against aligned large language models (LLMs). These attacks may extract private information or coerce the model into producing harmful outputs. In real-world deployments, LLMs are often part of a larger agentic pipeline including memory systems, retrieval, web access, and API calling. Such additional components introduce vulnerabilities that make these LLM-powered agents much easier to attack than isolated LLMs, yet relatively little work focuses on the security of LLM agents. In this paper, we analyze security and privacy vulnerabilities that are unique to LLM agents. We first provide a taxonomy of attacks categorized by threat actors, objectives, entry points, attacker observability, attack strategies, and inherent vulnerabilities of agent pipelines. We then conduct a series of illustrative attacks on popular open-source and commercial agents, demonstrating the immediate practical implications of their vulnerabilities. Notably, our attacks are trivial to implement and require no understanding of machine learning. 

**Abstract (ZH)**: 近年来，大量的机器学习（ML）安全文献主要关注针对对齐的大语言模型（LLMs）的攻击。这些攻击有可能提取私人信息或将模型引导向产生有害输出。在实际部署中，LLMs 很少是孤立存在的，它们通常作为更大自主管道的一部分，包括内存系统、信息检索、网络访问和API调用。这些额外的组件引入了新的脆弱性，使这些LLM驱动的代理比孤立的LLMs更容易受到攻击，但目前很少有工作关注这些LLM代理的安全性。在本文中，我们分析了仅针对LLM代理的独特安全和隐私漏洞。首先，我们提供了一种基于威胁行为者、攻击目标、入口点、攻击者可观察性、攻击策略和代理管道固有脆弱性的攻击分类体系。然后，我们对流行的开源和商用代理进行了系列示范攻击，展示了其漏洞的即时实用影响。值得注意的是，我们的攻击非常容易实现，且不需要了解机器学习的知识。 

---
# COAST: Intelligent Time-Adaptive Neural Operators 

**Title (ZH)**: COAST：智能时变神经算子 

**Authors**: Zhikai Wu, Shiyang Zhang, Sizhuang He, Sifan Wang, Min Zhu, Anran Jiao, Lu Lu, David van Dijk  

**Link**: [PDF](https://arxiv.org/pdf/2502.08574)  

**Abstract**: We introduce Causal Operator with Adaptive Solver Transformer (COAST), a novel neural operator learning method that leverages a causal language model (CLM) framework to dynamically adapt time steps. Our method predicts both the evolution of a system and its optimal time step, intelligently balancing computational efficiency and accuracy. We find that COAST generates variable step sizes that correlate with the underlying system intrinsicities, both within and across dynamical systems. Within a single trajectory, smaller steps are taken in regions of high complexity, while larger steps are employed in simpler regions. Across different systems, more complex dynamics receive more granular time steps. Benchmarked on diverse systems with varied dynamics, COAST consistently outperforms state-of-the-art methods, achieving superior performance in both efficiency and accuracy. This work underscores the potential of CLM-based intelligent adaptive solvers for scalable operator learning of dynamical systems. 

**Abstract (ZH)**: 我们提出了因果运算器与自适应求解器变换器（COAST），这是一种新颖的神经运算器学习方法，采用因果语言模型（CLM）框架动态适应时间步长。我们的方法不仅预测系统的演化，还预测其最优时间步长，智能地平衡计算效率与准确性。我们发现，COAST生成的变量时间步长与系统的内在特性相关，无论是同一种动力学系统内部，还是不同动力学系统之间。在单个轨迹中，复杂区域采取更小的时间步长，而简单区域则使用较大时间步长。在不同系统之间，更复杂的动力学特性则获得更加精细的时间步长。在多样化的动力学系统中进行基准测试，COAST始终优于现有最先进的方法，在效率和准确性方面均表现出优越的性能。这项工作强调了基于CLM的智能自适应求解器在可扩展的动力学系统运算器学习中的潜在价值。 

---
# Human-Centric Foundation Models: Perception, Generation and Agentic Modeling 

**Title (ZH)**: 以人为本的基石模型：感知、生成与自主建模 

**Authors**: Shixiang Tang, Yizhou Wang, Lu Chen, Yuan Wang, Sida Peng, Dan Xu, Wanli Ouyang  

**Link**: [PDF](https://arxiv.org/pdf/2502.08556)  

**Abstract**: Human understanding and generation are critical for modeling digital humans and humanoid embodiments. Recently, Human-centric Foundation Models (HcFMs) inspired by the success of generalist models, such as large language and vision models, have emerged to unify diverse human-centric tasks into a single framework, surpassing traditional task-specific approaches. In this survey, we present a comprehensive overview of HcFMs by proposing a taxonomy that categorizes current approaches into four groups: (1) Human-centric Perception Foundation Models that capture fine-grained features for multi-modal 2D and 3D understanding. (2) Human-centric AIGC Foundation Models that generate high-fidelity, diverse human-related content. (3) Unified Perception and Generation Models that integrate these capabilities to enhance both human understanding and synthesis. (4) Human-centric Agentic Foundation Models that extend beyond perception and generation to learn human-like intelligence and interactive behaviors for humanoid embodied tasks. We review state-of-the-art techniques, discuss emerging challenges and future research directions. This survey aims to serve as a roadmap for researchers and practitioners working towards more robust, versatile, and intelligent digital human and embodiments modeling. 

**Abstract (ZH)**: 人类的理解与生成对于建模数字化人类和类人形态至关重要。近期，以通用模型的成功为 inspir 基Human-centric Foundation Models (HcFMs) 已经兴起，这些模型借鉴了大型语言和视觉模型等通用模型的成功，旨在将多种与人类相关的任务统一到一个框架中，超越了传统的面向特定任务的方法。在本文综述中，我们通过提出一个分类来全面概述 HcFMs，将其分为四类：（1）人类中心感知基础模型，用于捕捉多模态2D 和 3D 的细粒度特征。 （2）人类中心生成式基础模型，生成高质量、多样化的与人类相关的内容。 （3）统一感知与生成模型，整合这些能力以增强对人类的理解和合成。 （4）人类中心代理人基础模型，超越感知和生成，学习类似人类的智能和交互行为，以适应类人形态任务。我们回顾了最先进的技术，讨论了新兴挑战和未来的研究方向。本文综述旨在为致力于更稳健、多功能和智能的数字化人类及形态建模的研究人员和实践者提供一条路线图。 

---
# Learning Humanoid Standing-up Control across Diverse Postures 

**Title (ZH)**: 跨不同姿态学习类人站立控制 

**Authors**: Tao Huang, Junli Ren, Huayi Wang, Zirui Wang, Qingwei Ben, Muning Wen, Xiao Chen, Jianan Li, Jiangmiao Pang  

**Link**: [PDF](https://arxiv.org/pdf/2502.08378)  

**Abstract**: Standing-up control is crucial for humanoid robots, with the potential for integration into current locomotion and loco-manipulation systems, such as fall recovery. Existing approaches are either limited to simulations that overlook hardware constraints or rely on predefined ground-specific motion trajectories, failing to enable standing up across postures in real-world scenes. To bridge this gap, we present HoST (Humanoid Standing-up Control), a reinforcement learning framework that learns standing-up control from scratch, enabling robust sim-to-real transfer across diverse postures. HoST effectively learns posture-adaptive motions by leveraging a multi-critic architecture and curriculum-based training on diverse simulated terrains. To ensure successful real-world deployment, we constrain the motion with smoothness regularization and implicit motion speed bound to alleviate oscillatory and violent motions on physical hardware, respectively. After simulation-based training, the learned control policies are directly deployed on the Unitree G1 humanoid robot. Our experimental results demonstrate that the controllers achieve smooth, stable, and robust standing-up motions across a wide range of laboratory and outdoor environments. Videos are available at this https URL. 

**Abstract (ZH)**: 站立控制对于类人robotics来说至关重要，其潜力在于可以集成到当前的移动和移动操作系统中，例如恢复跌倒。现有的方法要么局限于忽略了硬件限制的模拟，要么依赖于预定义的地面特定运动轨迹，无法在真实场景中实现不同姿态下的站立。为了弥合这一差距，我们提出了HoST（类人站立控制）——一种从零开始学习站立控制的强化学习框架，使其能够在多种姿态下实现稳健的模拟到现实的迁移。HoST通过利用多评论者架构和以多样化模拟地形为基础的课程学习方法，有效地学习适应不同姿态的运动模式。为了确保实际部署的成功，我们通过平滑正则化和隐式运动速度限制约束，分别缓解了物理硬件上的振荡和暴力运动。经过基于仿真的训练后，学习到的控制策略直接部署到Unitree G1类人机器人上。实验结果表明，控制器能够在广泛实验室和户外环境中实现平滑、稳定且 robust 的站立动作。有关视频可参考此链接：[该链接的格式未给出，需补充具体链接]。 

---
# Towards Principled Multi-Agent Task Agnostic Exploration 

**Title (ZH)**: 朝着原则性的多智能体任务无关探索方向 

**Authors**: Riccardo Zamboni, Mirco Mutti, Marcello Restelli  

**Link**: [PDF](https://arxiv.org/pdf/2502.08365)  

**Abstract**: In reinforcement learning, we typically refer to task-agnostic exploration when we aim to explore the environment without access to the task specification a priori. In a single-agent setting the problem has been extensively studied and mostly understood. A popular approach cast the task-agnostic objective as maximizing the entropy of the state distribution induced by the agent's policy, from which principles and methods follows. In contrast, little is known about task-agnostic exploration in multi-agent settings, which are ubiquitous in the real world. How should different agents explore in the presence of others? In this paper, we address this question through a generalization to multiple agents of the problem of maximizing the state distribution entropy. First, we investigate alternative formulations, highlighting respective positives and negatives. Then, we present a scalable, decentralized, trust-region policy search algorithm to address the problem in practical settings. Finally, we provide proof of concept experiments to both corroborate the theoretical findings and pave the way for task-agnostic exploration in challenging multi-agent settings. 

**Abstract (ZH)**: 在强化学习中，我们通常将不具备任务先验信息的目标探索定义为在没有任务规范的情况下探索环境。在单智能体设置中，这个问题已经被广泛研究并基本理解。一种广泛采用的方法是将目标探索定义为最大化智能体策略诱导的状态分布的熵，并据此形成了相应的原理和方法。相比之下，在多智能体设置中进行不具备任务先验信息的目标探索知之甚少，而多智能体设置在现实世界中普遍存在。在其他智能体存在的情况下，不同的智能体应该如何进行探索？在本文中，我们通过将最大化状态分布熵的问题扩展到多智能体场景来回答这个问题。首先，我们探讨了替代的公式化方法，并突出各自的优缺点。然后，我们提出了一种可扩展且去中心化的信任区域策略搜索算法，以解决实际场景中的问题。最后，我们提供了概念证明实验，以验证理论结果，并为具有挑战性的多智能体设置中进行不具备任务先验信息的目标探索铺平道路。 

---
# Hierarchical Multi-Agent Framework for Carbon-Efficient Liquid-Cooled Data Center Clusters 

**Title (ZH)**: 面向碳效率的液体冷却数据中心集群的分层多代理框架 

**Authors**: Soumyendu Sarkar, Avisek Naug, Antonio Guillen, Vineet Gundecha, Ricardo Luna Gutierrez, Sahand Ghorbanpour, Sajad Mousavi, Ashwin Ramesh Babu, Desik Rengarajan, Cullen Bash  

**Link**: [PDF](https://arxiv.org/pdf/2502.08337)  

**Abstract**: Reducing the environmental impact of cloud computing requires efficient workload distribution across geographically dispersed Data Center Clusters (DCCs) and simultaneously optimizing liquid and air (HVAC) cooling with time shift of workloads within individual data centers (DC). This paper introduces Green-DCC, which proposes a Reinforcement Learning (RL) based hierarchical controller to optimize both workload and liquid cooling dynamically in a DCC. By incorporating factors such as weather, carbon intensity, and resource availability, Green-DCC addresses realistic constraints and interdependencies. We demonstrate how the system optimizes multiple data centers synchronously, enabling the scope of digital twins, and compare the performance of various RL approaches based on carbon emissions and sustainability metrics while also offering a framework and benchmark simulation for broader ML research in sustainability. 

**Abstract (ZH)**: 减少云 computing 的环境影响需要高效地在地理位置分散的数据中心集群（DCCs）之间分配工作负载，并在单个数据中心内部的时间错峰工作负载的同时优化液冷和空气（HVAC）冷却。本文提出了一种基于强化学习（RL）的分层控制器 Green-DCC，以动态优化 DCC 中的工作负载和液冷。通过考虑天气、碳强度和资源可用性等因素，Green-DCC 针对现实约束和相互依赖性进行了优化。我们展示了该系统如何同时优化多个数据中心，并使其具备数字孪生的潜力，同时还基于碳排放和可持续性指标比较了各种 RL 方法的性能，并提供了一个框架和基准模拟，以促进更广泛的机器学习研究在可持续性方面的应用。 

---
# Exploring the Potential of Large Language Models to Simulate Personality 

**Title (ZH)**: 探索大型语言模型模拟人格的潜力 

**Authors**: Maria Molchanova, Anna Mikhailova, Anna Korzanova, Lidiia Ostyakova, Alexandra Dolidze  

**Link**: [PDF](https://arxiv.org/pdf/2502.08265)  

**Abstract**: With the advancement of large language models (LLMs), the focus in Conversational AI has shifted from merely generating coherent and relevant responses to tackling more complex challenges, such as personalizing dialogue systems. In an effort to enhance user engagement, chatbots are often designed to mimic human behaviour, responding within a defined emotional spectrum and aligning to a set of values. In this paper, we aim to simulate personal traits according to the Big Five model with the use of LLMs. Our research showed that generating personality-related texts is still a challenging task for the models. As a result, we present a dataset of generated texts with the predefined Big Five characteristics and provide an analytical framework for testing LLMs on a simulation of personality skills. 

**Abstract (ZH)**: 随着大规模语言模型（LLMs）的发展，对话式人工智能的关注点已从仅仅生成连贯且相关的内容，转向解决更加复杂的挑战，如个性化对话系统。为了增强用户参与度，聊天机器人通常被设计成模仿人类行为，其响应在固定的情感范围之内，并遵循一定的价值观。在本文中，我们旨在使用LLMs模拟Big Five人格模型中的个人特质。研究结果显示，生成与人格相关的内容仍然是模型的一个挑战性任务。因此，我们提供了一个预设Big Five特征的数据集，并提出了一种分析框架，以在人格技能的模拟测试中评估LLMs的表现。 

---
# TRISHUL: Towards Region Identification and Screen Hierarchy Understanding for Large VLM based GUI Agents 

**Title (ZH)**: TRISHUL：走向大型VLM基GUI代理的区域识别和屏幕层次结构理解 

**Authors**: Kunal Singh, Shreyas Singh, Mukund Khanna  

**Link**: [PDF](https://arxiv.org/pdf/2502.08226)  

**Abstract**: Recent advancements in Large Vision Language Models (LVLMs) have enabled the development of LVLM-based Graphical User Interface (GUI) agents under various paradigms. Training-based approaches, such as CogAgent and SeeClick, struggle with cross-dataset and cross-platform generalization due to their reliance on dataset-specific training. Generalist LVLMs, such as GPT-4V, employ Set-of-Marks (SoM) for action grounding, but obtaining SoM labels requires metadata like HTML source, which is not consistently available across platforms. Moreover, existing methods often specialize in singular GUI tasks rather than achieving comprehensive GUI understanding. To address these limitations, we introduce TRISHUL, a novel, training-free agentic framework that enhances generalist LVLMs for holistic GUI comprehension. Unlike prior works that focus on either action grounding (mapping instructions to GUI elements) or GUI referring (describing GUI elements given a location), TRISHUL seamlessly integrates both. At its core, TRISHUL employs Hierarchical Screen Parsing (HSP) and the Spatially Enhanced Element Description (SEED) module, which work synergistically to provide multi-granular, spatially, and semantically enriched representations of GUI elements. Our results demonstrate TRISHUL's superior performance in action grounding across the ScreenSpot, VisualWebBench, AITW, and Mind2Web datasets. Additionally, for GUI referring, TRISHUL surpasses the ToL agent on the ScreenPR benchmark, setting a new standard for robust and adaptable GUI comprehension. 

**Abstract (ZH)**: 近年来，大型视觉语言模型（LVLMs）的发展使得基于LVLM的图形用户界面（GUI）代理在各种范式下得以开发。基于训练的方法，如CogAgent和SeeClick，在跨数据集和跨平台的泛化方面遇到困难，因为它们依赖于特定数据集的训练。通用型LVLM，如GPT-4V，使用标记集（SoM）进行动作定位，但获得SoM标签需要HTML源代码等元数据，而在不同平台上这些元数据并不一致。此外，现有方法往往专长于单一的GUI任务，而未能实现全面的GUI理解。为了解决这些局限性，我们提出了TRISHUL，这是一种新的、无需训练的代理框架，能够增强通用型LVLM的综合GUI理解能力。与先前专注于行动定位（将指令映射到GUI元素）或GUI指引用（给定位置描述GUI元素）工作的研究不同，TRISHUL无缝地结合了这两方面。其核心是层次屏幕解析（HSP）和空间增强元素描述（SEED）模块，这些模块协同工作，提供多粒度、空间和语义丰富的GUI元素表示。我们的实验结果表明，TRISHUL在ScreenSpot、VisualWebBench、AITW和Mind2Web数据集上表现出色，尤其是在行动定位方面的表现。此外，在GUI指引用方面，TRISHUL在ScreenPR基准测试中超过了ToL代理，设定了更加稳健和适应性强的GUI理解的新标准。 

---
# VSC-RL: Advancing Autonomous Vision-Language Agents with Variational Subgoal-Conditioned Reinforcement Learning 

**Title (ZH)**: VSC-RL：基于变分子目标条件强化学习的自主视觉-语言代理技术进步 

**Authors**: Qingyuan Wu, Jianheng Liu, Jianye Hao, Jun Wang, Kun Shao  

**Link**: [PDF](https://arxiv.org/pdf/2502.07949)  

**Abstract**: State-of-the-art (SOTA) reinforcement learning (RL) methods enable the vision-language agents to learn from interactions with the environment without human supervision. However, they struggle with learning inefficiencies in tackling real-world complex sequential decision-making tasks, especially with sparse reward signals and long-horizon dependencies. To effectively address the issue, we introduce Variational Subgoal-Conditioned RL (VSC-RL), which reformulates the vision-language sequential decision-making task as a variational goal-conditioned RL problem, allowing us to leverage advanced optimization methods to enhance learning efficiency. Specifically, VSC-RL optimizes the SubGoal Evidence Lower BOund (SGC-ELBO), which consists of (a) maximizing the subgoal-conditioned return via RL and (b) minimizing the subgoal-conditioned difference with the reference policy. We theoretically demonstrate that SGC-ELBO is equivalent to the original optimization objective, ensuring improved learning efficiency without sacrificing performance guarantees. Additionally, for real-world complex decision-making tasks, VSC-RL leverages the vision-language model to autonomously decompose the goal into feasible subgoals, enabling efficient learning. Across various benchmarks, including challenging real-world mobile device control tasks, VSC-RL significantly outperforms the SOTA vision-language agents, achieving superior performance and remarkable improvement in learning efficiency. 

**Abstract (ZH)**: 最新的强化学习（SOTA）方法使具有视觉-语言能力的代理能够在无人类监督的情况下从与环境的交互中学习。然而，它们在应对现实世界中的复杂序列决策任务时存在学习效率低下的问题，尤其是在面对稀疏的奖励信号和长时序依赖性的情况下。为有效解决这一问题，我们提出了变分子目标条件强化学习（VSC-RL），将视觉-语言的序列决策任务重新表述为变分目标条件化强化学习问题，使我们能够利用高级优化方法来提高学习效率。具体而言，VSC-RL 优化子目标证据下界（SGC-ELBO），这包括通过增强学习最大化子目标条件下的回报以及通过最小化子目标与参考策略之间的差异来优化。我们从理论上证明了 SGC-ELBO 等同于原始优化目标，确保在不牺牲性能保证的前提下提高学习效率。此外，对于现实世界中的复杂决策任务，VSC-RL 利用视觉-语言模型自助分解目标为可行的子目标，从而实现高效的 学习。在各种基准测试中，包括具有挑战性的现实世界移动设备控制任务，VSC-RL 显著优于最先进的视觉-语言代理，实现了卓越的性能并显著提高了学习效率。 

---
