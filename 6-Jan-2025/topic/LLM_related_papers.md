# Long Context vs. RAG for LLMs: An Evaluation and Revisits 

**Title (ZH)**: 长上下文对比基于检索的生成：对大语言模型的评估与回顾 

**Authors**: Xinze Li, Yixin Cao, Yubo Ma, Aixin Sun  

**Link**: [PDF](https://arxiv.org/pdf/2501.01880)  

**Abstract**: Extending context windows (i.e., Long Context, LC) and using retrievers to selectively access relevant information (i.e., Retrieval-Augmented Generation, RAG) are the two main strategies to enable LLMs to incorporate extremely long external contexts. This paper revisits recent studies on this topic, highlighting their key insights and discrepancies. We then provide a more comprehensive evaluation by filtering out questions answerable without external context, identifying the most effective retrieval methods, and expanding the datasets. We show that LC generally outperforms RAG in question-answering benchmarks, especially for Wikipedia-based questions. Summarization-based retrieval performs comparably to LC, while chunk-based retrieval lags behind. However, RAG has advantages in dialogue-based and general question queries. These insights underscore the trade-offs between RAG and LC strategies, offering guidance for future optimization of LLMs with external knowledge sources. We also provide an in-depth discussion on this topic, highlighting the overlooked importance of context relevance in existing studies. 

**Abstract (ZH)**: 扩展上下文窗口（即Long Context，LC）和利用检索器选择性地访问相关信息（即检索增强生成，RAG）是使大语言模型（LLM）能够纳入极其长的外部上下文的两种主要策略。本论文重新审视了这一领域近期的研究，强调了它们的关键见解和差异。我们通过筛选出不需要外部上下文即可回答的问题，识别出最有效的检索方法，并扩展数据集，提供了一种更为全面的评估。结果显示，LC在问题回答基准测试中通常优于RAG，特别是在基于Wikipedia的问题上。基于摘要的检索方法与LC表现相当，而基于片段的检索方法则落后于后者。然而，RAG在基于对话的问题查询和一般的查询中具有优势。这些见解突显了RAG和LC策略之间的权衡，为未来利用外部知识源优化LLM提供了指导。我们还对这一主题进行了深入讨论，强调了现有研究中被忽视的上下文相关性的重要性。 

---
# Automating Legal Concept Interpretation with LLMs: Retrieval, Generation, and Evaluation 

**Title (ZH)**: 使用大语言模型自动化法律概念解释：检索、生成与评估 

**Authors**: Kangcheng Luo, Quzhe Huang, Cong Jiang, Yansong Feng  

**Link**: [PDF](https://arxiv.org/pdf/2501.01743)  

**Abstract**: Legal articles often include vague concepts to adapt to the ever-changing society. Providing detailed interpretations of these concepts is a critical task for legal practitioners, which requires meticulous and professional annotations by legal experts, admittedly time-consuming and expensive to collect at scale. In this paper, we introduce a novel retrieval-augmented generation framework, ATRI, for AuTomatically Retrieving relevant information from past judicial precedents and Interpreting vague legal concepts. We further propose a new benchmark, Legal Concept Entailment, to automate the evaluation of generated concept interpretations without expert involvement. Automatic evaluations indicate that our generated interpretations can effectively assist large language models (LLMs) in understanding vague legal concepts. Multi-faceted evaluations by legal experts indicate that the quality of our concept interpretations is comparable to those written by human experts. Our work has strong implications for leveraging LLMs to support legal practitioners in interpreting vague legal concepts and beyond. 

**Abstract (ZH)**: 法律文章中经常包含模糊的概念，以适应不断变化的社会环境。为这些概念提供详细的解读是法律从业者的一项关键任务，这需要法律专家进行精细且专业的标注，显然这是一个耗时且昂贵的大规模收集过程。本文介绍了一种新颖的检索增强生成框架ATRI，用于从过往的司法先例中自动检索相关的信息，并对模糊的法律概念进行解释。我们还提出了一种新的基准——法律概念蕴含，以实现自动化评估生成的概念解释，无需专家参与。自动评估结果显示，我们生成的解释能够有效地帮助大语言模型（LLMs）理解模糊的法律概念。法律专家进行的多方面评估表明，我们概念解释的质量与人类专家撰写的解释相当。我们这项工作对利用大语言模型支持法律从业者解释模糊的法律概念及其更多方面具有重要影响。 

---
# CoT-based Synthesizer: Enhancing LLM Performance through Answer Synthesis 

**Title (ZH)**: 基于CoT的合成器：通过答案合成提升大语言模型性能 

**Authors**: Bohan Zhang, Xiaokang Zhang, Jing Zhang, Jifan Yu, Sijia Luo, Jie Tang  

**Link**: [PDF](https://arxiv.org/pdf/2501.01668)  

**Abstract**: Current inference scaling methods, such as Self-consistency and Best-of-N, have proven effective in improving the accuracy of LLMs on complex reasoning tasks. However, these methods rely heavily on the quality of candidate responses and are unable to produce correct answers when all candidates are incorrect. In this paper, we propose a novel inference scaling strategy, CoT-based Synthesizer, which leverages CoT reasoning to synthesize superior answers by analyzing complementary information from multiple candidate responses, even when all candidate responses are flawed. To enable a lightweight and cost-effective implementation, we introduce an automated data generation pipeline that creates diverse training data. This allows smaller LLMs trained on this data to improve the inference accuracy of larger models, including API-based LLMs. Experimental results across four benchmark datasets with seven policy models demonstrate that our method significantly enhances performance, with gains of 11.8% for Llama3-8B and 10.3% for GPT-4o on the MATH dataset. The corresponding training data and code are publicly available on this https URL. 

**Abstract (ZH)**: 当前的推理扩展方法，如自我一致性（Self-consistency）和Best-of-N，已被证明能在复杂推理任务中提高大规模语言模型（LLM）的准确性。然而，这些方法高度依赖候选响应的质量，在所有候选响应都错误的情况下无法生成正确的答案。在本文中，我们提出了一种新的推理扩展策略——基于CoT的合成器（CoT-based Synthesizer），该策略利用CoT推理通过分析多个候选响应中的互补信息来综合出更优的答案，即使所有候选响应都是错误的。为了实现轻量级且成本效益高的实施，我们引入了一个自动数据生成管道，以生成多样化的训练数据。这使得在该数据上训练的小型LLM可以提升更大模型（包括基于API的LLM）的推理准确性。在四个基准数据集上进行的实验结果表明，我们的方法显著提高了性能，Llama3-8B在MATH数据集上的性能提高了11.8%，GPT-4o的性能提高了10.3%。相关训练数据和代码已在此网址公开：[提供网址]。 

---
# MIRAGE: Exploring How Large Language Models Perform in Complex Social Interactive Environments 

**Title (ZH)**: MIRAGE：探索大型语言模型在复杂社会互动环境中的表现 

**Authors**: Cai Yin, Gu Zhouhong, Du Zhaohan, Ye Zheyu, Cao Shaosheng, Xu Yiqian, Feng Hongwei, Chen Ping  

**Link**: [PDF](https://arxiv.org/pdf/2501.01652)  

**Abstract**: Large Language Models (LLMs) have shown remarkable capabilities in environmental perception, reasoning-based decision-making, and simulating complex human behaviors, particularly in interactive role-playing contexts. This paper introduces the Multiverse Interactive Role-play Ability General Evaluation (MIRAGE), a comprehensive framework designed to assess LLMs' proficiency in portraying advanced human behaviors through murder mystery games. MIRAGE features eight intricately crafted scripts encompassing diverse themes and styles, providing a rich simulation. To evaluate LLMs' performance, MIRAGE employs four distinct methods: the Trust Inclination Index (TII) to measure dynamics of trust and suspicion, the Clue Investigation Capability (CIC) to measure LLMs' capability of conducting information, the Interactivity Capability Index (ICI) to assess role-playing capabilities and the Script Compliance Index (SCI) to assess LLMs' capability of understanding and following instructions. Our experiments indicate that even popular models like GPT-4 face significant challenges in navigating the complexities presented by the MIRAGE. The datasets and simulation codes are available in \href{this https URL}{github}. 

**Abstract (ZH)**: 大型语言模型（LLMs）在环境感知、基于推理的决策制定以及模拟复杂的人类行为方面展示了显著的能力，尤其是在互动角色扮演的语境中。本文介绍了多维互动角色扮演能力通用评估框架（MIRAGE），这是一个全面的框架，旨在通过谋杀谜题游戏评估LLMs在呈现高级人类行为方面的 proficiency。

MIRAGE 包含八个精密编撰的脚本，涵盖不同的主题和风格，提供了丰富的模拟环境。为了评估LLMs的表现，MIRAGE 使用了四种不同的方法：信任倾向指数（TII）用于衡量信任和疑虑的动力学过程，线索调查能力（CIC）用于衡量LLMs的线索调查能力，互动能力指数（ICI）用于评估角色扮演能力，脚本遵从指数（SCI）用于评估LLMs理解并遵循指令的能力。我们的实验表明，即使是流行模型如GPT-4，在应对MIRAGE 提出的复杂性时也面临重大挑战。

实验数据和模拟代码可在 \href{https://github.com/...}{GitHub} 获取。 

---
# A non-ergodic framework for understanding emergent capabilities in Large Language Models 

**Title (ZH)**: 一种非遍历框架，用于理解大型语言模型中涌现能力的机制 

**Authors**: Javier Marin  

**Link**: [PDF](https://arxiv.org/pdf/2501.01638)  

**Abstract**: Large language models have emergent capabilities that come unexpectedly at scale, but we need a theoretical framework to explain why and how they emerge. We prove that language models are actually non-ergodic systems while providing a mathematical framework based on Stuart Kauffman's theory of the adjacent possible (TAP) to explain capability emergence. Our resource-constrained TAP equation demonstrates how architectural, training, and contextual constraints interact to shape model capabilities through phase transitions in semantic space. We prove through experiments with three different language models that capacities emerge through discrete transitions guided by constraint interactions and path-dependent exploration. This framework provides a theoretical basis for understanding emergence in language models and guides the development of architectures that can guide capability emergence. 

**Abstract (ZH)**: 大规模语言模型在大规模应用中展现出预料之外的能力，但我们需要一个理论框架来解释这些能力是如何出现的以及背后的机理。我们证明语言模型实际上是非平衡系统，并基于Stuart Kauffman的临近可能理论（The Adjacent Possible, TAP）提供了一个数学框架来解释这些能力的出现。我们通过资源受限的TAP方程，展示了架构、训练和上下文约束如何通过语义空间中的相变相互作用来塑造模型的能力。通过针对三种不同语言模型的实验，我们证明能力是通过由约束交互和路径依赖探索引导的离散过渡逐步出现的。该框架为理解语言模型中的能力涌现提供了理论基础，并指导了能够引导能力涌现的架构的发展。 

---
# Auto-RT: Automatic Jailbreak Strategy Exploration for Red-Teaming Large Language Models 

**Title (ZH)**: Auto-RT：用于红队演练大型语言模型的自动越狱策略探索 

**Authors**: Yanjiang Liu, Shuhen Zhou, Yaojie Lu, Huijia Zhu, Weiqiang Wang, Hongyu Lin, Ben He, Xianpei Han, Le Sun  

**Link**: [PDF](https://arxiv.org/pdf/2501.01830)  

**Abstract**: Automated red-teaming has become a crucial approach for uncovering vulnerabilities in large language models (LLMs). However, most existing methods focus on isolated safety flaws, limiting their ability to adapt to dynamic defenses and uncover complex vulnerabilities efficiently. To address this challenge, we propose Auto-RT, a reinforcement learning framework that automatically explores and optimizes complex attack strategies to effectively uncover security vulnerabilities through malicious queries. Specifically, we introduce two key mechanisms to reduce exploration complexity and improve strategy optimization: 1) Early-terminated Exploration, which accelerate exploration by focusing on high-potential attack strategies; and 2) Progressive Reward Tracking algorithm with intermediate downgrade models, which dynamically refine the search trajectory toward successful vulnerability exploitation. Extensive experiments across diverse LLMs demonstrate that, by significantly improving exploration efficiency and automatically optimizing attack strategies, Auto-RT detects a boarder range of vulnerabilities, achieving a faster detection speed and 16.63\% higher success rates compared to existing methods. 

**Abstract (ZH)**: 自动红队攻击已成为揭露大型语言模型（LLMs）漏洞的关键方法。然而，现有的大多数方法主要关注孤立的安全缺陷，限制了它们适应动态防御和高效发现复杂漏洞的能力。为解决这一挑战，我们提出了一种名为Auto-RT的强化学习框架，该框架能够自动探索和优化复杂的攻击策略，以有效地通过恶意查询揭露安全漏洞。具体而言，我们引入了两种关键机制来降低探索复杂性并提高策略优化效果：1）早终止探索（Early-terminated Exploration），通过专注于有高潜力的攻击策略加速探索过程；2）渐进奖励跟踪算法结合中间降级模型（Progressive Reward Tracking algorithm with intermediate downgrade models），该算法能够动态调整搜索轨迹以趋向成功利用漏洞。在多种LLM上的广泛实验表明，通过显著提高探索效率并自动优化攻击策略，Auto-RT能够检测更广泛的漏洞，实现更快的检测速度和16.63%更高的成功率，相比于现有方法。 

---
# SDPO: Segment-Level Direct Preference Optimization for Social Agents 

**Title (ZH)**: SDPO：社会智能体的段级直接偏好优化 

**Authors**: Aobo Kong, Wentao Ma, Shiwan Zhao, Yongbin Li, Yuchuan Wu, Ke Wang, Xiaoqian Liu, Qicheng Li, Yong Qin, Fei Huang  

**Link**: [PDF](https://arxiv.org/pdf/2501.01821)  

**Abstract**: Social agents powered by large language models (LLMs) can simulate human social behaviors but fall short in handling complex goal-oriented social dialogues. Direct Preference Optimization (DPO) has proven effective in aligning LLM behavior with human preferences across a variety of agent tasks. Existing DPO-based approaches for multi-turn interactions are divided into turn-level and session-level methods. The turn-level method is overly fine-grained, focusing exclusively on individual turns, while session-level methods are too coarse-grained, often introducing training noise. To address these limitations, we propose Segment-Level Direct Preference Optimization (SDPO), which focuses on specific key segments within interactions to optimize multi-turn agent behavior while minimizing training noise. Evaluations on the SOTOPIA benchmark demonstrate that SDPO-tuned agents consistently outperform both existing DPO-based methods and proprietary LLMs like GPT-4o, underscoring SDPO's potential to advance the social intelligence of LLM-based agents. We release our code and data at this https URL. 

**Abstract (ZH)**: 由大规模语言模型（LLMs）驱动的社会代理能够模拟人类社会行为，但在处理复杂的目的导向的社会对话方面存在不足。直接偏好优化（DPO）已证明在各种代理任务中能够有效使LLM的行为与人类偏好保持一致。现有的基于DPO的方法可以分为回合级和会话级方法。回合级方法过于细粒度，仅专注于个体回合，而会话级方法则过于粗粒度，常常引入训练噪声。为解决这些局限性，我们提出了段级直接偏好优化（SDPO），该方法专注于交互中的特定关键段落，以优化多回合代理行为，并尽量减少训练噪声。在SOTOPIA基准测试中的评估表明，SDPO优化后的代理在整个过程中持续优于现有基于DPO的方法以及专用LLM（如GPT-4o），突显出了SDPO在提升基于LLM的代理社会智能方面的潜力。我们已在以下网址发布了我们的代码和数据：[请补充具体网址]。 

---
# How Toxic Can You Get? Search-based Toxicity Testing for Large Language Models 

**Title (ZH)**: 你能有多毒？基于搜索的大型语言模型毒性测试 

**Authors**: Simone Corbo, Luca Bancale, Valeria De Gennaro, Livia Lestingi, Vincenzo Scotti, Matteo Camilli  

**Link**: [PDF](https://arxiv.org/pdf/2501.01741)  

**Abstract**: Language is a deep-rooted means of perpetration of stereotypes and discrimination. Large Language Models (LLMs), now a pervasive technology in our everyday lives, can cause extensive harm when prone to generating toxic responses. The standard way to address this issue is to align the LLM, which, however, dampens the issue without constituting a definitive solution. Therefore, testing LLM even after alignment efforts remains crucial for detecting any residual deviations with respect to ethical standards. We present EvoTox, an automated testing framework for LLMs' inclination to toxicity, providing a way to quantitatively assess how much LLMs can be pushed towards toxic responses even in the presence of alignment. The framework adopts an iterative evolution strategy that exploits the interplay between two LLMs, the System Under Test (SUT) and the Prompt Generator steering SUT responses toward higher toxicity. The toxicity level is assessed by an automated oracle based on an existing toxicity classifier. We conduct a quantitative and qualitative empirical evaluation using four state-of-the-art LLMs as evaluation subjects having increasing complexity (7-13 billion parameters). Our quantitative evaluation assesses the cost-effectiveness of four alternative versions of EvoTox against existing baseline methods, based on random search, curated datasets of toxic prompts, and adversarial attacks. Our qualitative assessment engages human evaluators to rate the fluency of the generated prompts and the perceived toxicity of the responses collected during the testing sessions. Results indicate that the effectiveness, in terms of detected toxicity level, is significantly higher than the selected baseline methods (effect size up to 1.0 against random search and up to 0.99 against adversarial attacks). Furthermore, EvoTox yields a limited cost overhead (from 22% to 35% on average). 

**Abstract (ZH)**: 语言是根深蒂固的刻板印象和歧视传播的工具。大规模语言模型（LLMs）现在已成为我们日常生活中的一项普遍技术，当它们容易生成有毒响应时，会造成广泛的伤害。处理这一问题的标准方法是使LLM与其目标对齐，然而这种方法只能减轻问题，而不能构成最终的解决方案。因此，在对齐努力之后继续测试LLM仍然至关重要，以检测其是否在伦理标准方面仍存在残留的偏差。我们提出了一种名为EvoTox的自动化测试框架，用于评估LLM趋向于产生有毒响应的倾向，提供了一种定量评估方法，即使在对齐后，也可以评估LLM能够被推向更严重的有毒响应的程度。该框架采用迭代进化策略，利用系统测试（SUT）和提示生成器之间的互动，引导SUT生成更具毒性的响应。毒性水平通过基于现有毒性分类器的自动判别器进行评估。我们使用四种最先进的LLM作为评估对象，其参数量依次增加（7-13亿），进行了定量和定性的实证评估。定量评估比较了四种不同版本的EvoTox与现有基准方法（基于随机搜索、精心策划的有毒提示数据集和对抗性攻击）的成本效益。定性评估则通过邀请人类评估者评估生成提示的流畅性以及测试过程中收集的响应所感知到的毒性。结果表明，EvoTox在检测到的毒性水平方面的有效性显著高于所选的基准方法（与随机搜索相比，效应量高达1.0；与对抗性攻击相比，效应量高达0.99）。此外，EvoTox产生的成本附加量相对有限（平均增加22%至35%）。 

---
# Predicting the Performance of Black-box LLMs through Self-Queries 

**Title (ZH)**: 通过自我查询预测黑盒大语言模型的性能 

**Authors**: Dylan Sam, Marc Finzi, J. Zico Kolter  

**Link**: [PDF](https://arxiv.org/pdf/2501.01558)  

**Abstract**: As large language models (LLMs) are increasingly relied on in AI systems, predicting when they make mistakes is crucial. While a great deal of work in the field uses internal representations to interpret model behavior, these representations are inaccessible when given solely black-box access through an API. In this paper, we extract features of LLMs in a black-box manner by using follow-up prompts and taking the probabilities of different responses as representations to train reliable predictors of model behavior. We demonstrate that training a linear model on these low-dimensional representations produces reliable and generalizable predictors of model performance at the instance level (e.g., if a particular generation correctly answers a question). Remarkably, these can often outperform white-box linear predictors that operate over a model's hidden state or the full distribution over its vocabulary. In addition, we demonstrate that these extracted features can be used to evaluate more nuanced aspects of a language model's state. For instance, they can be used to distinguish between a clean version of GPT-4o-mini and a version that has been influenced via an adversarial system prompt that answers question-answering tasks incorrectly or introduces bugs into generated code. Furthermore, they can reliably distinguish between different model architectures and sizes, enabling the detection of misrepresented models provided through an API (e.g., identifying if GPT-3.5 is supplied instead of GPT-4o-mini). 

**Abstract (ZH)**: 随着大型语言模型（LLMs）在AI系统中的应用越来越广泛，预测其何时出错变得至关重要。尽管该领域中大量的工作使用内部表示来解释模型行为，但在仅通过API获取黑箱访问的情况下，这些表示是不可用的。在本文中，我们通过使用后续提示并以不同响应的概率作为表示来以黑箱方式提取LLMs的特征，以此训练可靠的行为预测模型。我们证明，在这些低维度表示上训练线性模型能够生成可靠的、具有普适性的实例级模型性能预测器（例如，如果某一生成正确回答了一个问题）。令人惊讶的是，这些预测器常常能够超越那些在模型隐藏状态或其词汇分布的完整概率分布上运行的白箱线性预测器。此外，我们还证明，这些提取的特征可用于评估语言模型状态的更精细方面。举例来说，它们可用于区分GPT-4o-mini的干净版本和被恶意系统提示影响过的版本，该提示错误回答了问题或引入了生成代码中的错误。此外，这些提取的特征还能可靠地区分不同的模型架构和规模，从而能够在API提供的模型中检测到误导性的模型（例如，识别是否存在使用GPT-3.5代替GPT-4o-mini的情况）。 

---
# Enhancing Reasoning through Process Supervision with Monte Carlo Tree Search 

**Title (ZH)**: 通过蒙特卡洛树搜索进行过程监督以增强推理能力 

**Authors**: Shuangtao Li, Shuaihao Dong, Kexin Luan, Xinhan Di, Chaofan Ding  

**Link**: [PDF](https://arxiv.org/pdf/2501.01478)  

**Abstract**: Large language models (LLMs) have demonstrated their remarkable capacity across a variety of tasks. However, reasoning remains a challenge for LLMs. To improve LLMs' reasoning ability, process supervision has proven to be better than outcome supervision. In this work, we study using Monte Carlo Tree Search (MCTS) to generate process supervision data with LLMs themselves for training them. We sample reasoning steps with an LLM and assign each step a score that captures its "relative correctness," and the LLM is then trained by minimizing weighted log-likelihood of generating the reasoning steps. This generate-then-train process is repeated iteratively until this http URL experimental results demonstrate that the proposed methods considerably improve the performance of LLMs on two mathematical reasoning datasets. Furthermore, models trained on one dataset also exhibit improved performance on the other, showing the transferability of the enhanced reasoning ability. 

**Abstract (ZH)**: 大规模语言模型（LLMs）在各种任务中展现出了其卓越的能力。然而，推理依然是LLMs的挑战之一。为了提高LLMs的推理能力，过程监督被证明比结果监督更有效。在本研究中，我们探讨了使用蒙特卡洛树搜索（MCTS）通过LLMs自身生成过程监督数据来进行训练的方法。我们使用LLMs抽样生成推理步骤，并为每个步骤分配一个“相对正确性”的得分，然后通过最小化生成推理步骤的加权对数似然来训练LLMs。这一生成-训练过程将迭代重复，直到达到某个性能阈值。实验结果表明，所提出的方法显著提高了LLMs在两个数学推理数据集上的表现。此外，一个数据集训练的模型在另一个数据集上也表现更好，表明增强的推理能力具有较好的迁移性。 

---
# Reinforcing Thinking through Reasoning-Enhanced Reward Models 

**Title (ZH)**: 通过推理增强奖励模型强化思考 

**Authors**: Diji Yang, Linda Zeng, Kezhen Chen, Yi Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2501.01457)  

**Abstract**: Large Language Models (LLMs) exhibit great potential in complex multi-step reasoning through inference-time thinking but still struggle with deciding when to stop thinking due to limited self-awareness about their knowledge boundaries. While human preference alignment has shown extraordinary opportunities, expensive labeling challenges adherence to scaling law. Language model self-critique, as an alternative to using human-labeled reasoning data, is questioned with its inherited biases. This work addresses these challenges by distilling the LLM's own reasoning processes into synthetic behavioral data, eliminating the need for manual labeling of intermediate steps. Building on this concept, we propose Distillation-Reinforcement-Reasoning (DRR), a three-step framework that leverages the LLM's inherent behaviors as external feedback by first generating behavioral data using the Reasoner (LLM) to reflect its reasoning capabilities, then training a lightweight discriminative reward model (DM) on behavioral data, and finally deploying the DM at inference time to assist the Reasoner's decision-making. Experiments on multiple benchmarks show that the DRR framework outperforms self-critique approaches without relying on additional complex data annotation. Benefiting from lightweight design, ease of replication, and adaptability, DRR is applicable to a wide range of LLM-centric tasks. 

**Abstract (ZH)**: 大型语言模型（LLMs）在通过推理时展现出在复杂多步推理中的巨大潜力，但在决定何时停止思考方面仍存在困难，这是因为它们在自我意识其知识边界方面存在局限。虽然人类偏好对齐显示出非凡的机会，但由于昂贵的标注成本，它未能严格遵循缩放定律。语言模型自我批判作为一种避免使用人类标注推理数据的方法受到了质疑，其本身就带有一定的偏见。本研究通过将LLM自身的推理过程转化为合成行为数据，去除了手动标注中间步骤的需求，从而应对这些挑战。基于这一概念，我们提出了蒸馏-强化-推理（DRR）框架，这是一个三步框架，首先通过Reasoner（LLM）生成行为数据以反映其推理能力，然后在生成的数据上训练一个轻量级的判别奖励模型（DM），最后在推理时部署DM，以辅助Reasoner的决策。在多个基准测试上的实验结果表明，DRR框架在不依赖额外复杂数据注释的情况下优于自我批判方法。得益于其轻量级设计、易于复制和适应性，DRR适用于广泛的LLM中心任务。 

---
# Virgo: A Preliminary Exploration on Reproducing o1-like MLLM 

**Title (ZH)**: 维gone：关于重现o1-like MLLM的一种初步探索 

**Authors**: Yifan Du, Zikang Liu, Yifan Li, Wayne Xin Zhao, Yuqi Huo, Bingning Wang, Weipeng Chen, Zheng Liu, Zhongyuan Wang, Ji-Rong Wen  

**Link**: [PDF](https://arxiv.org/pdf/2501.01904)  

**Abstract**: Recently, slow-thinking reasoning systems, built upon large language models (LLMs), have garnered widespread attention by scaling the thinking time during inference. There is also growing interest in adapting this capability to multimodal large language models (MLLMs). Given that MLLMs handle more complex data semantics across different modalities, it is intuitively more challenging to implement multimodal slow-thinking systems.
To address this issue, in this paper, we explore a straightforward approach by fine-tuning a capable MLLM with a small amount of textual long-form thought data, resulting in a multimodal slow-thinking system, Virgo (Visual reasoning with long thought). We find that these long-form reasoning processes, expressed in natural language, can be effectively transferred to MLLMs. Moreover, it seems that such textual reasoning data can be even more effective than visual reasoning data in eliciting the slow-thinking capacities of MLLMs. While this work is preliminary, it demonstrates that slow-thinking capacities are fundamentally associated with the language model component, which can be transferred across modalities or domains. This finding can be leveraged to guide the development of more powerful slow-thinking reasoning systems. We release our resources at this https URL. 

**Abstract (ZH)**: 近年来，基于大规模语言模型（LLMs）的慢思考推理系统通过在推理过程中扩展思考时间引起了广泛的关注。同时，将这一能力扩展到多模态大规模语言模型（MLLMs）也引起了越来越多的兴趣。鉴于MLLMs处理跨不同模态的更复杂数据语义，直观上，实现多模态慢思考系统更具挑战性。

为了解决这一问题，本文探索了一种简单的做法：通过微调一个强大的MLLM，使用少量的文本长形式思考数据，从而生成一个多模态慢思考系统，命名为Virgo（视觉推理中的长思考）。我们发现，以自然语言表达的长形式推理过程可以有效转移至MLLMs中。此外，似乎这种文本推理数据在激发MLLMs的慢思考能力方面比视觉推理数据更为有效。尽管这项工作尚处于初步阶段，但它证明了慢思考能力从根本上与语言模型组件相关，并且可以跨模态或领域进行转移。这一发现可以用于指导更强大慢思考推理系统的开发。我们的资源已在此处提供：[此 https URL]。 

---
# Multi-Agent Conversational Online Learning for Adaptive LLM Response Identification 

**Title (ZH)**: 多代理对话式在线学习在自适应大语言模型响应识别中的应用 

**Authors**: Xiangxiang Dai, Yuejin Xie, Maoli Liu, Xuchuang Wang, Zhuohua Li, Huanyu Wang, John C.S. Lui  

**Link**: [PDF](https://arxiv.org/pdf/2501.01849)  

**Abstract**: The remarkable generative capability of large language models (LLMs) has sparked a growing interest in automatically generating responses for different applications. Given the dynamic nature of user preferences and the uncertainty of LLM response performance, it is crucial to design efficient online learning algorithms to identify optimal LLM responses (i.e., high-quality responses that also meet user preferences). Most existing online algorithms adopt a centralized approach and fail to leverage explicit user preferences for more efficient and personalized LLM response identification. In contrast, this paper introduces \textit{MACO} (\underline{M}ulti-\underline{A}gent \underline{C}onversational \underline{O}nline Learning for Adaptive LLM Response Identification): 1) The online LLM response identification process is accelerated by multiple local agents (such as smartphones), while enhancing data privacy; 2) A novel conversational mechanism is proposed to adaptively conduct conversations for soliciting user preferences (e.g., a preference for a humorous tone over a serious one in generated responses), so to minimize uncertainty in preference estimation. Our theoretical analysis demonstrates that \cadi\ is near-optimal regarding cumulative regret. Additionally, \cadi\ offers reduced communication costs and computational complexity by eliminating the traditional, computing-intensive ``G-optimal design" found in previous works. Extensive experiments with the open LLM \textit{Llama}, coupled with two different embedding models from Google and OpenAI for text vector representation, demonstrate that \cadi\ significantly outperforms the current state-of-the-art in online LLM response identification. 

**Abstract (ZH)**: 大型语言模型（LLMs）强大的生成能力激发了自动为不同应用生成响应的兴趣。鉴于用户偏好和LLM响应性能的动态性，设计高效的在线学习算法以识别最优LLM响应（即高质量且符合用户偏好的响应）至关重要。现有的大多数在线算法采用中心化的做法，未能充分利用显式的用户偏好来实现更高效的个性化LLM响应识别。与此不同，本文引入了一种名为 \textit{MACO}（多代理对话在线学习以适应性识别LLM响应）的方法：1）通过多台本地代理（如智能手机）加速在线LLM响应识别过程，同时增强数据隐私；2）提出了一种新的对话机制，以适应性地进行对话以征求用户偏好（例如，在生成的响应中偏好轻松幽默而非严肃），从而最小化偏好估计的不确定性。我们的理论分析表明，MACO 在累积后悔方面接近最优。此外，MACO 通过消除以往工作中常见的计算密集型的传统“G-最优设计”减少通信成本和计算复杂度。结合使用开源LLM \textit{Llama} 以及来自Google和OpenAI的两种不同嵌入模型进行文本向量表示的大量实验表明，MACO 显著优于当前在线LLM响应识别的最先进的方法。 

---
# Adaptive Few-shot Prompting for Machine Translation with Pre-trained Language Models 

**Title (ZH)**: 预训练语言模型中的自适应少样本提示方法在机器翻译中的应用 

**Authors**: Lei Tang, Jinghui Qin, Wenxuan Ye, Hao Tan, Zhijing Yang  

**Link**: [PDF](https://arxiv.org/pdf/2501.01679)  

**Abstract**: Recently, Large language models (LLMs) with in-context learning have demonstrated remarkable potential in handling neural machine translation. However, existing evidence shows that LLMs are prompt-sensitive and it is sub-optimal to apply the fixed prompt to any input for downstream machine translation tasks. To address this issue, we propose an adaptive few-shot prompting (AFSP) framework to automatically select suitable translation demonstrations for various source input sentences to further elicit the translation capability of an LLM for better machine translation. First, we build a translation demonstration retrieval module based on LLM's embedding to retrieve top-k semantic-similar translation demonstrations from aligned parallel translation corpus. Rather than using other embedding models for semantic demonstration retrieval, we build a hybrid demonstration retrieval module based on the embedding layer of the deployed LLM to build better input representation for retrieving more semantic-related translation demonstrations. Then, to ensure better semantic consistency between source inputs and target outputs, we force the deployed LLM itself to generate multiple output candidates in the target language with the help of translation demonstrations and rerank these candidates. Besides, to better evaluate the effectiveness of our AFSP framework on the latest language and extend the research boundary of neural machine translation, we construct a high-quality diplomatic Chinese-English parallel dataset that consists of 5,528 parallel Chinese-English sentences. Finally, extensive experiments on the proposed diplomatic Chinese-English parallel dataset and the United Nations Parallel Corpus (Chinese-English part) show the effectiveness and superiority of our proposed AFSP. 

**Abstract (ZH)**: 最近，具有上下文学习能力的大规模语言模型（LLMs）在神经机器翻译任务中展现出了显著潜力。然而，现有证据表明，LLMs 对提示（prompt）敏感，在处理下游机器翻译任务时使用固定的提示并不到位。为解决这一问题，我们提出了一个自适应少量示例提示（AFSP）框架，自动为各种源输入句选择合适的翻译示例，进一步激发LLMs的翻译能力，以实现更好的机器翻译效果。首先，我们基于LLMs的嵌入构建了一个翻译示例检索模块，从对齐的平行翻译语料库中检索出与其最接近语义的前k个翻译示例。不同于使用其他嵌入模型进行语义示例检索，我们基于部署的LLMs的嵌入层构建了一个混合示例检索模块，以更好地为检索语义相关的翻译示例构建输入表示。然后，为了确保源输入和目标输出在语义上的一致性，我们利用翻译示例帮助部署的LLMs生成多个目标语言输出候选，并重新排序这些候选。此外，为了更全面地评估我们提出的AFSP框架在最新语言研究中的效果，并拓宽神经机器翻译的研究边界，我们构建了一个高质量的外交中文-英文平行语料库，该语料库包含5,528对平行的中文-英文句子。最后，在提出的外交中文-英文平行语料库和联合国平行语料库（中文-英文部分）上进行的大量实验表明，我们提出的AFSP框架的有效性和优越性。 

---
# BARTPredict: Empowering IoT Security with LLM-Driven Cyber Threat Prediction 

**Title (ZH)**: BARTPredict：以大型语言模型驱动的网络威胁预测技术赋能物联网安全 

**Authors**: Alaeddine Diaf, Abdelaziz Amara Korba, Nour Elislem Karabadji, Yacine Ghamri-Doudane  

**Link**: [PDF](https://arxiv.org/pdf/2501.01664)  

**Abstract**: The integration of Internet of Things (IoT) technology in various domains has led to operational advancements, but it has also introduced new vulnerabilities to cybersecurity threats, as evidenced by recent widespread cyberattacks on IoT devices. Intrusion detection systems are often reactive, triggered by specific patterns or anomalies observed within the network. To address this challenge, this work proposes a proactive approach to anticipate and preemptively mitigate malicious activities, aiming to prevent potential damage before it occurs. This paper proposes an innovative intrusion prediction framework empowered by Pre-trained Large Language Models (LLMs). The framework incorporates two LLMs: a fine-tuned Bidirectional and AutoRegressive Transformers (BART) model for predicting network traffic and a fine-tuned Bidirectional Encoder Representations from Transformers (BERT) model for evaluating the predicted traffic. By harnessing the bidirectional capabilities of BART the framework then identifies malicious packets among these predictions. Evaluated using the CICIoT2023 IoT attack dataset, our framework showcases a notable enhancement in predictive performance, attaining an impressive 98% overall accuracy, providing a powerful response to the cybersecurity challenges that confront IoT networks. 

**Abstract (ZH)**: 物联网（IoT）技术在各个领域的整合带来了操作上的进步，但也引入了新的网络安全威胁，这在近期广泛针对IoT设备的网络攻击中得到了证实。入侵检测系统通常具有反应性，它们会根据网络中观察到的具体模式或异常行为被触发。为应对这一挑战，本研究提出了一种主动的方法来预见和预处理恶意活动，旨在在损害发生之前防止潜在的损害。本文提出了一种由预训练大语言模型（LLMs）赋能的创新入侵预测框架。该框架集成了两种LLMs：微调的双向和自回归变换器（BART）模型用于预测网络流量，以及微调的双向编码器表示（BERT）模型用于评估预测的流量。通过利用BART的双向能力，该框架随后在这些预测中识别出恶意包。该框架在使用ICCIDT2023 IoT攻击数据集进行评估时，显示出显著的预测性能提升，总体准确率达到98%，为面对IoT网络的网络安全挑战提供了强有力的响应。 

---
# Human-AI Teaming Using Large Language Models: Boosting Brain-Computer Interfacing (BCI) and Brain Research 

**Title (ZH)**: 使用大型语言模型的人机团队协作：增强脑-计算机接口（BCI）与脑科学研究 

**Authors**: Maryna Kapitonova, Tonio Ball  

**Link**: [PDF](https://arxiv.org/pdf/2501.01451)  

**Abstract**: Recently, there is an increasing interest in using artificial intelligence (AI) to automate aspects of the research process, or even autonomously conduct the full research cycle from idea generation, over data analysis, to composing and evaluation of scientific manuscripts. Examples of working AI scientist systems have been demonstrated for computer science tasks and running molecular biology labs. While some approaches aim for full autonomy of the scientific AI, others rather aim for leveraging human-AI teaming. Here, we address how to adapt such approaches for boosting Brain-Computer Interface (BCI) development, as well as brain research resp. neuroscience at large. We argue that at this time, a strong emphasis on human-AI teaming, in contrast to fully autonomous AI BCI researcher will be the most promising way forward. We introduce the collaborative workspaces concept for human-AI teaming based on a set of Janusian design principles, looking both ways, to the human as well as to the AI side. Based on these principles, we present ChatBCI, a Python-based toolbox for enabling human-AI collaboration based on interaction with Large Language Models (LLMs), designed for BCI research and development projects. We show how ChatBCI was successfully used in a concrete BCI project on advancing motor imagery decoding from EEG signals. Our approach can be straightforwardly extended to broad neurotechnological and neuroscientific topics, and may by design facilitate human expert knowledge transfer to scientific AI systems in general. 

**Abstract (ZH)**: 近年来，人们越来越关注利用人工智能（AI）自动化研究过程的各个方面，甚至自主完成从概念生成、数据分析到科学论文撰写与评估的整个研究周期。已有研究展示了在计算机科学任务和分子生物学实验室中运行的可工作的AI科学家系统的实例。虽然一些方法旨在实现科学AI的完全自主性，而另一些则更侧重于充分利用人机协作。在此，我们探讨如何根据“双面镜”设计原则调整这些方法，以促进脑-机接口（BCI）开发以及广泛的脑研究与神经科学。我们认为，鉴于当前情况，与完全自主的科学AI BCI研究员相比，强调人机协作将是最有前景的发展方向。我们引入了一种基于“双面镜”设计原则的合作工作空间概念，面向人类和AI的两侧。基于这些原则，我们提出了基于与大型语言模型（LLMs）交互的ChatBCI工具箱，专为BCI研究和开发项目设计。我们展示了ChatBCI在促进基于EEG信号的运动意念解码方面的具体BCI项目中的成功应用。这种方法可以轻松扩展到广泛的神经技术与神经科学主题，通过设计，可以促进人类专家知识向科学AI系统的一般性转移。 

---
