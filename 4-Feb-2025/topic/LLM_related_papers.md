# TReMu: Towards Neuro-Symbolic Temporal Reasoning for LLM-Agents with Memory in Multi-Session Dialogues 

**Title (ZH)**: TReMu：面向具有记忆功能的多会话对话中LLM代理的神经符号时间推理 

**Authors**: Yubin Ge, Salvatore Romeo, Jason Cai, Raphael Shu, Monica Sunkara, Yassine Benajiba, Yi Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2502.01630)  

**Abstract**: Temporal reasoning in multi-session dialogues presents a significant challenge which has been under-studied in previous temporal reasoning benchmarks. To bridge this gap, we propose a new evaluation task for temporal reasoning in multi-session dialogues and introduce an approach to construct a new benchmark by augmenting dialogues from LoCoMo and creating multi-choice QAs. Furthermore, we present TReMu, a new framework aimed at enhancing the temporal reasoning capabilities of LLM-agents in this context. Specifically, the framework employs \textit{time-aware memorization} through timeline summarization, generating retrievable memory by summarizing events in each dialogue session with their inferred dates. Additionally, we integrate \textit{neuro-symbolic temporal reasoning}, where LLMs generate Python code to perform temporal calculations and select answers. Experimental evaluations on popular LLMs demonstrate that our benchmark is challenging, and the proposed framework significantly improves temporal reasoning performance compared to baseline methods, raising from 29.83 on GPT-4o via standard prompting to 77.67 via our approach and highlighting its effectiveness in addressing temporal reasoning in multi-session dialogues. 

**Abstract (ZH)**: 多会话对话中的时间推理提出了一个重要的挑战，而这一挑战在之前的时序推理基准中尚未得到充分的研究。为解决这一问题，我们提出了一项新的评价任务，旨在评估多会话对话中的时间推理能力，并通过增强Loremotion数据集中的对话，构造了一个新的基准，并创建了多项选择题。此外，我们提出了TReMu框架，该框架旨在增强在这种情况下LLM代理的时间推理能力。具体而言，该框架通过时间轴总结，采用了具有时间意识的记忆化方法，生成可检索的记忆，通过总结每个对话会话中的事件及其推断日期来生成摘要。我们还整合了神经符号时间推理，其中LLM生成Python代码以执行时间计算并选择答案。对流行的基础模型进行的实验评估表明，我们的基准具有挑战性，并且所提的框架在时间推理性能上显著优于基线方法，评分从GPT-4o标准提示下的29.83提高到我们方法下的77.67，突显了其在解决多会话对话中时间推理问题方面的有效性。 

---
# PhD Knowledge Not Required: A Reasoning Challenge for Large Language Models 

**Title (ZH)**: 无需博士学位：大型语言模型的推理挑战 

**Authors**: Carolyn Jane Anderson, Joydeep Biswas, Aleksander Boruch-Gruszecki, Federico Cassano, Molly Q Feldman, Arjun Guha, Francesca Lucchetti, Zixuan Wu  

**Link**: [PDF](https://arxiv.org/pdf/2502.01584)  

**Abstract**: Existing benchmarks for frontier models often test specialized, ``PhD-level'' knowledge that is difficult for non-experts to grasp. In contrast, we present a benchmark based on the NPR Sunday Puzzle Challenge that requires only general knowledge. Our benchmark is challenging for both humans and models, however correct solutions are easy to verify, and models' mistakes are easy to spot.
Our work reveals capability gaps that are not evident in existing benchmarks: OpenAI o1 significantly outperforms other reasoning models that are on par on benchmarks that test specialized knowledge. Furthermore, our analysis of reasoning outputs uncovers new kinds of failures. DeepSeek R1, for instance, often concedes with ``I give up'' before providing an answer that it knows is wrong. R1 can also be remarkably ``uncertain'' in its output and in rare cases, it does not ``finish thinking,'' which suggests the need for an inference-time technique to ``wrap up'' before the context window limit is reached. We also quantify the effectiveness of reasoning longer with R1 and Gemini Thinking to identify the point beyond which more reasoning is unlikely to improve accuracy on our benchmark. 

**Abstract (ZH)**: 现有的前沿模型基准通常测试专业化、接近“博士水平”的知识，这使得非专家难以理解。相比之下，我们提出了一个基于《纽约客》周日谜题挑战的基准，只需要一般的知识。我们的基准对人类和模型都是具有挑战性的，但正确的解决方案易于验证，模型的错误也非常容易被发现。

我们的研究揭示了现有基准中未明显显现的能力差距：OpenAI o1 显著优于在测试专业化知识的基准上与其水平相当的其他推理模型。此外，我们对推理输出的分析揭示了新的失败类型。例如，DeepSeek R1 经常在给出它知道是错误的答案之前承认“放弃”。R1 的输出有时也表现得异常“不确定”，偶尔甚至未能完成推理过程，这表明在达到上下文窗口限制之前需要一种推理时的技巧来“总结”。我们还利用 R1 和 Gemini Thinking 推理更长的时间来量化推理长度对基准效果的影响，以确定进一步推理是否还可能提高准确性。 

---
# TeLL-Drive: Enhancing Autonomous Driving with Teacher LLM-Guided Deep Reinforcement Learning 

**Title (ZH)**: TeLL-Drive：借助教师大语言模型引导的深度强化学习增强自主驾驶 

**Authors**: Chengkai Xu, Jiaqi Liu, Peng Hang, Jian Sun  

**Link**: [PDF](https://arxiv.org/pdf/2502.01387)  

**Abstract**: Although Deep Reinforcement Learning (DRL) and Large Language Models (LLMs) each show promise in addressing decision-making challenges in autonomous driving, DRL often suffers from high sample complexity, while LLMs have difficulty ensuring real-time decision making. To address these limitations, we propose TeLL-Drive, a hybrid framework that integrates an Teacher LLM to guide an attention-based Student DRL policy. By incorporating risk metrics, historical scenario retrieval, and domain heuristics into context-rich prompts, the LLM produces high-level driving strategies through chain-of-thought reasoning. A self-attention mechanism then fuses these strategies with the DRL agent's exploration, accelerating policy convergence and boosting robustness across diverse driving conditions. Our experimental results, evaluated across multiple traffic scenarios, show that TeLL-Drive outperforms existing baseline methods, including other LLM-based approaches, in terms of success rates, average returns, and real-time feasibility. Ablation studies underscore the importance of each model component, especially the synergy between the attention mechanism and LLM-driven guidance. These findings suggest that TeLL-Drive significantly enhances both the adaptability and safety of autonomous driving systems, while offering a more efficient and scalable approach for policy learning. Full validation results are available on our website. 

**Abstract (ZH)**: 尽管深度强化学习（DRL）和大规模语言模型（LLMs）在自动驾驶决策问题上各具潜力，但DRL往往面临样本复杂性高的问题，而LLMs则难以保证实时决策。为了解决这些局限性，我们提出了一种混合框架——TeLL-Drive，该框架结合了一个指导型的教师LLM，以引导基于注意力的学生DRL策略。通过对包含风险指标、历史场景检索和领域启发式信息的语境提示进行推理，LLM生成了高层次的驾驶策略。随后，自注意力机制将这些策略与DRL代理的探索相结合，加快了策略收敛速度，并提高了在各种驾驶条件下的鲁棒性。我们的实验结果，跨越多个交通场景进行评估，表明TeLL-Drive在成功率、平均回报以及实时可行性方面优于现有基准方法，包括其他LLM基方法。消融研究强调了每个模型组件的重要性，特别是注意力机制与LLM驱动指导之间的协同作用。这些发现表明，TeLL-Drive显著增强了自动驾驶系统的适应性和安全性，同时提供了一种更高效、更可扩展的策略学习方法。完整的验证结果可在我们的网站上查阅。 

---
# PSSD: Making Large Language Models Self-denial via Human Psyche Structure 

**Title (ZH)**: PSSD：通过人类心理结构使大规模语言模型实现自我否定 

**Authors**: Jinzhi Liao, Zenghua Liao, Xiang Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2502.01344)  

**Abstract**: The enhance of accuracy in reasoning results of LLMs arouses the community's interests, wherein pioneering studies investigate post-hoc strategies to rectify potential mistakes. Despite extensive efforts, they are all stuck in a state of resource competition demanding significant time and computing expenses. The cause of the situation lies in the failure of identifying the fundamental feature of the solutions in this line, coined as the self-denial of LLMs. In other words, LLMs should confidently determine the potential existence of mistakes and carefully execute the targeted correction. As the whole procedure conducts within LLMs, supporting and persuasive references are hard to acquire, while the absence of specific steps towards refining hidden mistakes persists even when errors are acknowledged. In response to the challenges, we present PSSD, which refers to and implements the human psyche structure such that three distinct and interconnected roles contribute to human reasoning. Specifically, PSSD leverages the recent multi-agent paradigm, and is further enhanced with three innovatively conceived roles: (1) the intuition-based id role that provides initial attempts based on benign LLMs; (2) the rule-driven superego role that summarizes rules to regulate the above attempts, and returns specific key points as guidance; and (3) the script-centric ego role that absorbs all procedural information to generate executable script for the final answer prediction. Extensive experiments demonstrate that the proposed design not only better enhance reasoning capabilities, but also seamlessly integrate with current models, leading to superior performance. 

**Abstract (ZH)**: 增强大型语言模型推理结果的准确性引起了学术界的浓厚兴趣，其中先驱研究探索了事后策略以纠正潜在的错误。尽管付出了大量努力，这些策略仍然面临着耗费大量时间和计算资源的资源竞争状态。造成这种状况的原因在于未能识别这一系列解决方案的基本特征，这种特征被称为大型语言模型的自我否认。换句话说，大型语言模型应该自信地确定潜在的错误存在，并仔细执行针对性的纠正措施。由于整个过程都在大型语言模型内部进行，难以获得支持和说服性的参考，即使是承认错误的情况下，也缺乏具体的步骤来改进隐藏的错误。

针对这一挑战，我们提出了PSSD，该方法借鉴和实现了人类心理结构，从而由三个相互关联且各自独立的角色共同促进人类推理。具体而言，PSSD 利用了最近的多智能体范式，并进一步增强了三个创新设计的角色：（1）基于直觉的本我角色，基于良性大型语言模型提供初始尝试；（2）基于规则的超我角色，总结规则以调节上述尝试，并返回具体要点作为指导；（3）以脚本为中心的自我角色，吸收所有程序信息以生成可执行脚本，最终用于预测答案。广泛的实验表明，所提出的设计不仅能够更好地增强推理能力，还能够无缝集成到当前模型中，从而带来卓越的性能。 

---
# DeepRAG: Thinking to Retrieval Step by Step for Large Language Models 

**Title (ZH)**: DeepRAG：逐步思考以进行大型语言模型的检索 

**Authors**: Xinyan Guan, Jiali Zeng, Fandong Meng, Chunlei Xin, Yaojie Lu, Hongyu Lin, Xianpei Han, Le Sun, Jie Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2502.01142)  

**Abstract**: Large Language Models (LLMs) have shown remarkable potential in reasoning while they still suffer from severe factual hallucinations due to timeliness, accuracy, and coverage of parametric knowledge. Meanwhile, integrating reasoning with retrieval-augmented generation (RAG) remains challenging due to ineffective task decomposition and redundant retrieval, which can introduce noise and degrade response quality. In this paper, we propose DeepRAG, a framework that models retrieval-augmented reasoning as a Markov Decision Process (MDP), enabling strategic and adaptive retrieval. By iteratively decomposing queries, DeepRAG dynamically determines whether to retrieve external knowledge or rely on parametric reasoning at each step. Experiments show that DeepRAG improves retrieval efficiency while improving answer accuracy by 21.99%, demonstrating its effectiveness in optimizing retrieval-augmented reasoning. 

**Abstract (ZH)**: 以下是经过学术规范翻译的内容：

大型语言模型（LLMs）在推理方面展现出了显著潜力，但仍然受到时效性、准确性和覆盖范围方面的参数化知识限制，导致严重的事实幻觉。同时，将推理与检索增强生成（RAG）相结合仍然颇具挑战性，主要由于任务分解不完善和冗余检索可能导致噪音增加，进而降低响应质量。在本文中，我们提出了一种名为DeepRAG的框架，该框架将检索增强推理建模为马尔可夫决策过程（MDP），从而实现战略性且适应性的检索。通过逐步分解查询，DeepRAG动态决定在每一步是检索外部知识还是依赖于参数化推理。实验结果显示，DeepRAG在提高检索效率的同时，将答案准确性提高了21.99%，证明了其在优化检索增强推理方面的有效性。 

---
# ZebraLogic: On the Scaling Limits of LLMs for Logical Reasoning 

**Title (ZH)**: ZebraLogic：大规模语言模型在逻辑推理中的扩展极限 

**Authors**: Bill Yuchen Lin, Ronan Le Bras, Kyle Richardson, Ashish Sabharwal, Radha Poovendran, Peter Clark, Yejin Choi  

**Link**: [PDF](https://arxiv.org/pdf/2502.01100)  

**Abstract**: We investigate the logical reasoning capabilities of large language models (LLMs) and their scalability in complex non-monotonic reasoning. To this end, we introduce ZebraLogic, a comprehensive evaluation framework for assessing LLM reasoning performance on logic grid puzzles derived from constraint satisfaction problems (CSPs). ZebraLogic enables the generation of puzzles with controllable and quantifiable complexity, facilitating a systematic study of the scaling limits of models such as Llama, o1 models, and DeepSeek-R1. By encompassing a broad range of search space complexities and diverse logical constraints, ZebraLogic provides a structured environment to evaluate reasoning under increasing difficulty.
Our results reveal a significant decline in accuracy as problem complexity grows -- a phenomenon we term the curse of complexity. This limitation persists even with larger models and increased inference-time computation, suggesting inherent constraints in current LLM reasoning capabilities. Additionally, we explore strategies to enhance logical reasoning, including Best-of-N sampling, backtracking mechanisms, and self-verification prompts. Our findings offer critical insights into the scalability of LLM reasoning, highlight fundamental limitations, and outline potential directions for improvement. 

**Abstract (ZH)**: 我们探讨了大规模语言模型（LLMs）在非单调逻辑推理中的推理能力和扩展性。为此，我们引入了ZebraLogic，这是一个综合的评估框架，用于评估LLMs在源自约束满足问题（CSPs）的逻辑网格谜题上的推理性能。ZebraLogic能够生成具有可控性和可量化复杂度的谜题，从而有利于系统研究如Llama、o1模型和DeepSeek-R1等模型的扩展性极限。通过涵盖广泛的搜索空间复杂性和多样的逻辑约束，ZebraLogic提供了一个结构化的环境来评估随着难度增加的推理能力。

我们的研究表明，随着问题复杂性的增加，准确性会显著下降——我们将其称为复杂性诅咒。即使使用更大的模型和更多的推理时间计算，这一限制仍然存在，这表明当前LLMs推理能力存在固有的局限性。此外，我们还探讨了增强逻辑推理的策略，包括最佳抽样、回溯机制和自验证提示。我们的发现对于理解LLMs推理的扩展性提供了关键见解，突显了其根本局限性，并指出了改进的方向。 

---
# Language Models Use Trigonometry to Do Addition 

**Title (ZH)**: 语言模型利用三角函数进行加法运算 

**Authors**: Subhash Kantamneni, Max Tegmark  

**Link**: [PDF](https://arxiv.org/pdf/2502.00873)  

**Abstract**: Mathematical reasoning is an increasingly important indicator of large language model (LLM) capabilities, yet we lack understanding of how LLMs process even simple mathematical tasks. To address this, we reverse engineer how three mid-sized LLMs compute addition. We first discover that numbers are represented in these LLMs as a generalized helix, which is strongly causally implicated for the tasks of addition and subtraction, and is also causally relevant for integer division, multiplication, and modular arithmetic. We then propose that LLMs compute addition by manipulating this generalized helix using the "Clock" algorithm: to solve $a+b$, the helices for $a$ and $b$ are manipulated to produce the $a+b$ answer helix which is then read out to model logits. We model influential MLP outputs, attention head outputs, and even individual neuron preactivations with these helices and verify our understanding with causal interventions. By demonstrating that LLMs represent numbers on a helix and manipulate this helix to perform addition, we present the first representation-level explanation of an LLM's mathematical capability. 

**Abstract (ZH)**: 数学推理正日益成为大型语言模型（LLM）能力的重要指标，然而我们对LLM处理甚至基本数学任务的理解却相对缺乏。为解决这一问题，我们反向工程了三种中型LLM进行加法计算的方式。我们首先发现这些LLM中的数字是以一种广义螺旋的形式表示的，这种表示方式对加法和减法任务有强烈的因果关联性，同时也与整数除法、乘法和模运算有关。然后我们提出，LLM是通过“时钟”算法操作这种广义螺旋来进行加法运算：要解决$a+b$的问题，会操纵$a$和$b$的螺旋，以生成$a+b$的答案螺旋，然后从该螺旋中读取出模型的预测结果。我们用这些螺旋来建模有影响力的MLP输出、注意头输出，甚至单个神经元的预激活状态，并通过因果干预验证我们的理解。通过表明LLM将数字表示在螺旋上，并操纵该螺旋以执行加法运算，我们首次从表示层面解释了LLM的数学能力。 

---
# Psychometric-Based Evaluation for Theorem Proving with Large Language Models 

**Title (ZH)**: 基于心理测量学的评估方法在大规模语言模型进行定理证明中的应用 

**Authors**: Jianyu Zhang, Yongwang Zhao, Long Zhang, Jilin Hu, Xiaokun Luan, Zhiwei Xu, Feng Yang  

**Link**: [PDF](https://arxiv.org/pdf/2502.00855)  

**Abstract**: Large language models (LLMs) for formal theorem proving have become a prominent research focus. At present, the proving ability of these LLMs is mainly evaluated through proof pass rates on datasets such as miniF2F. However, this evaluation method overlooks the varying importance of theorems. As a result, it fails to highlight the real performance disparities between LLMs and leads to high evaluation costs. This study proposes a psychometric-based evaluation method for theorem proving with LLMs, comprising two main components: Dataset Annotation and Adaptive Evaluation. First, we propose a metric calculation method to annotate the dataset with difficulty and discrimination metrics. Specifically, we annotate each theorem in the miniF2F dataset and grade them into varying difficulty levels according to the performance of LLMs, resulting in an enhanced dataset: miniF2F-Graded. Experimental results show that the difficulty grading in miniF2F-Graded better reflects the theorem difficulty perceived by LLMs. Secondly, we design an adaptive evaluation method to dynamically select the most suitable theorems for testing based on the annotated metrics and the real-time performance of LLMs. We apply this method to evaluate 10 LLMs. The results show that our method finely highlights the performance disparities between LLMs. It also reduces evaluation costs by using only 23% of the theorems in the dataset. 

**Abstract (ZH)**: 大型语言模型（LLMs）在形式定理证明中的应用已成为研究的重点。目前，这些LLMs的证明能力主要通过在miniF2F等数据集上的证明通过率进行评估。然而，这种评估方法忽视了定理之间相对重要性的差异，导致未能突出LLMs之间的实际性能差异，同时增加了评估成本。本研究提出了一种基于心理测量学的形式定理证明评价方法，主要包括两个主要组成部分：数据集注释和自适应评估。首先，我们提出了一种度量计算方法，用于为数据集添加难度和区分度指标。具体而言，我们对miniF2F数据集中的每个定理进行注释，并根据LLMs的性能将其划分为不同的难度等级，从而形成增强的数据集miniF2F-Graded。实验结果显示，miniF2F-Graded中的难度分级更能反映LLMs感知的定理难度。其次，我们设计了一种自适应评估方法，根据标注的指标和LLMs的实时性能动态选择最合适的定理进行测试。我们将此方法应用于评估10个LLMs，结果显示，我们的方法能够精细地突出LLMs之间的性能差异，且通过仅使用数据集中的23%定理即可显著降低评估成本。 

---
# RTBAgent: A LLM-based Agent System for Real-Time Bidding 

**Title (ZH)**: RTBAgent：一个基于大语言模型的实时竞价代理系统 

**Authors**: Leng Cai, Junxuan He, Yikai Li, Junjie Liang, Yuanping Lin, Ziming Quan, Yawen Zeng, Jin Xu  

**Link**: [PDF](https://arxiv.org/pdf/2502.00792)  

**Abstract**: Real-Time Bidding (RTB) enables advertisers to place competitive bids on impression opportunities instantaneously, striving for cost-effectiveness in a highly competitive landscape. Although RTB has widely benefited from the utilization of technologies such as deep learning and reinforcement learning, the reliability of related methods often encounters challenges due to the discrepancies between online and offline environments and the rapid fluctuations of online bidding. To handle these challenges, RTBAgent is proposed as the first RTB agent system based on large language models (LLMs), which synchronizes real competitive advertising bidding environments and obtains bidding prices through an integrated decision-making process. Specifically, obtaining reasoning ability through LLMs, RTBAgent is further tailored to be more professional for RTB via involved auxiliary modules, i.e., click-through rate estimation model, expert strategy knowledge, and daily reflection. In addition, we propose a two-step decision-making process and multi-memory retrieval mechanism, which enables RTBAgent to review historical decisions and transaction records and subsequently make decisions more adaptive to market changes in real-time bidding. Empirical testing with real advertising datasets demonstrates that RTBAgent significantly enhances profitability. The RTBAgent code will be publicly accessible at: this https URL. 

**Abstract (ZH)**: 实时竞价（RTB）允许广告商在瞬间对展示机会进行竞争性出价，以在高度竞争的环境中追求成本效益。尽管RTB从深度学习和强化学习等技术的应用中广泛受益，但由于在线和离线环境之间的差异以及在线竞价的快速波动，相关方法的可靠性常常会遇到挑战。为了应对这些挑战，我们提出了基于大型语言模型（LLMs）的首个RTB代理系统——RTBAgent，该系统同步了真实的竞争广告竞价环境，并通过集成决策过程获取竞价价格。具体来说，通过大型语言模型（LLMs）获得推理能力后，RTBAgent进一步通过包含辅助模块（如点击率估计模型、专家策略知识和每日反思）来更加专业化地适应RTB需求。此外，我们提出了两步决策过程和多记忆检索机制，使RTBAgent能够回顾历史决策和交易记录，并在实时竞价中更加适应市场变化做出决策。实证测试使用真实的广告数据集表明，RTBAgent显著提高了盈利能力。RTBAgent的代码将在以下网址公开访问：this https URL。 

---
# Learning Autonomous Code Integration for Math Language Models 

**Title (ZH)**: 学习自主代码集成的数学语言模型 

**Authors**: Haozhe Wang, Long Li, Chao Qu, Fengming Zhu, Weidi Xu, Wei Chu, Fangzhen Lin  

**Link**: [PDF](https://arxiv.org/pdf/2502.00691)  

**Abstract**: Recent research on tool integration for math Large Language Models (LLMs) aims to combine complementary strengths of chain-of-thought (CoT) reasoning and code execution. However, we discover a critical limitation: current tool-integrated math LLMs rely on externally dictated instructions to decide whether to use CoT or code, lacking the autonomy to choose the most appropriate method independently. This prompts us to study \emph{Autonomous Code integration} for math LLMs, which enables models to \emph{independently} develop their own methodology-selection strategy in the absence of reliable supervision. To address this challenge, we propose an innovative Expectation-Maximization (EM) formulation that refines the model's decision-making through the exploration of its capabilities. This framework alternates between (a) computing a reference strategy that improves the model's belief over its capabilities through self-exploration, and (b) updating the model based on the refined belief. We further enhance this framework with an efficient implementation, incorporating a novel data synthesis strategy and off-policy reinforcement learning. Extensive experiments demonstrate that our approach, using only a public query set, significantly boosts the performance of existing math LLMs, raising accuracy by nearly 20\% to 65.28\% on the challenging MATH benchmark, while reducing code executions by up to 65\% . 

**Abstract (ZH)**: 近年来，关于数学大型语言模型（LLMs）工具集成的研究旨在结合链式思考（CoT）推理和代码执行的互补优势。然而，我们发现一个关键的局限性：现有的工具集成数学LLMs依赖于外部指令来决定是否使用CoT或代码，缺乏独立选择最适当方法的自主能力。这促使我们研究数学LLMs的自主代码集成，使模型能够在缺乏可靠监督的情况下自主发展其方法选择策略。为了解决这一挑战，我们提出了一种创新的期望最大化（EM）公式，通过探索模型的能力来改进其决策过程。该框架交替进行以下两个步骤：（a）计算一个参考策略，通过自我探索提高模型对其能力的信任度；（b）根据改进的信任度更新模型。我们进一步通过高效的实现增强了此框架，结合了一种新颖的数据合成策略和分政策略强化学习。广泛的实验表明，我们的方法仅使用公开查询集，显著提高了现有数学LLMs的性能，在具有挑战性的MATH基准测试中，准确率提高了近20%，达到65.28%，同时代码执行次数减少了高达65%。 

---
# LLM-based event log analysis techniques: A survey 

**Title (ZH)**: 基于大型语言模型的事件日志分析技术：一种综述 

**Authors**: Siraaj Akhtar, Saad Khan, Simon Parkinson  

**Link**: [PDF](https://arxiv.org/pdf/2502.00677)  

**Abstract**: Event log analysis is an important task that security professionals undertake. Event logs record key information on activities that occur on computing devices, and due to the substantial number of events generated, they consume a large amount of time and resources to analyse. This demanding and repetitive task is also prone to errors. To address these concerns, researchers have developed automated techniques to improve the event log analysis process. Large Language Models (LLMs) have recently demonstrated the ability to successfully perform a wide range of tasks that individuals would usually partake in, to high standards, and at a pace and degree of complexity that outperform humans. Due to this, researchers are rapidly investigating the use of LLMs for event log analysis. This includes fine-tuning, Retrieval-Augmented Generation (RAG) and in-context learning, which affect performance. These works demonstrate good progress, yet there is a need to understand the developing body of knowledge, identify commonalities between works, and identify key challenges and potential solutions to further developments in this domain. This paper aims to survey LLM-based event log analysis techniques, providing readers with an in-depth overview of the domain, gaps identified in previous research, and concluding with potential avenues to explore in future. 

**Abstract (ZH)**: 事件日志分析是一项重要的安全专业人员任务。事件日志记录了计算设备上发生的活动的关键信息，但由于生成的事件数量庞大，分析这些日志会消耗大量时间和资源。这一任务既耗时又重复，并且容易出错。为了解决这些问题，研究人员开发了自动化技术以改进事件日志分析过程。大语言模型（LLMs）最近已经展示出能够高效、高质量地完成人类通常会参与的多种任务，并且在速度和复杂性方面超越人类。由于这一点，研究人员正在迅速探索使用LLMs进行事件日志分析的方法。这包括微调、检索增强生成（RAG）和上下文学习等方法，它们对性能产生影响。尽管这些研究展示了良好的进展，但仍有必要理解这一领域的新兴知识体系，识别不同研究之间的共通之处，并识别关键挑战和潜在解决方案以推动该领域进一步发展。本文旨在回顾基于LLM的事件日志分析技术，为读者提供该领域的深入概述，指出先前研究中的空白，并总结出未来研究的潜在途径。 

---
# CollabLLM: From Passive Responders to Active Collaborators 

**Title (ZH)**: CollabLLM：从被动回应者到主动合作者 

**Authors**: Shirley Wu, Michel Galley, Baolin Peng, Hao Cheng, Gavin Li, Yao Dou, Weixin Cai, James Zou, Jure Leskovec, Jianfeng Gao  

**Link**: [PDF](https://arxiv.org/pdf/2502.00640)  

**Abstract**: Large Language Models are typically trained with next-turn rewards, limiting their ability to optimize for long-term interaction. As a result, they often respond passively to ambiguous or open-ended user requests, failing to help users reach their ultimate intents and leading to inefficient conversations. To address these limitations, we introduce CollabLLM, a novel and general training framework that enhances multiturn human-LLM collaboration. Its key innovation is a collaborative simulation that estimates the long-term contribution of responses using Multiturn-aware Rewards. By reinforcement fine-tuning these rewards, CollabLLM goes beyond responding to user requests, and actively uncovers user intent and offers insightful suggestions-a key step towards more human-centered AI. We also devise a multiturn interaction benchmark with three challenging tasks such as document creation. CollabLLM significantly outperforms our baselines with averages of 18.5% higher task performance and 46.3% improved interactivity by LLM judges. Finally, we conduct a large user study with 201 judges, where CollabLLM increases user satisfaction by 17.6% and reduces user spent time by 10.4%. 

**Abstract (ZH)**: 大型语言模型通常使用下一轮奖励进行训练，这限制了它们优化长期交互的能力。因此，它们往往对模糊或开放性的用户请求作出被动响应，无法帮助用户达成最终意图，导致对话效率低下。为解决这些问题，我们提出了一种名为CollabLLM的新颖且通用的训练框架，以增强多轮人类-语言模型协作。其核心创新在于一种协作模拟，通过多轮次意识奖励估算响应的长期贡献。通过强化学习精细调整这些奖励，CollabLLM不仅响应用户请求，还能主动揭示用户意图并提供有意义的建议，这是迈向更加用户导向的人工智能的关键步骤。我们还设计了一个多轮次交互基准，其中包括三个具有挑战性的任务，如文档创建。CollabLLM在平均任务性能上显著高于我们的基线，提高了18.5%，并且在LLM评判者看来，对话互动性提高了46.3%。最后，我们在201名评判者参与的大规模用户研究中，CollabLLM将用户满意度提高了17.6%，并将用户耗时减少了10.4%。 

---
# Advanced Weakly-Supervised Formula Exploration for Neuro-Symbolic Mathematical Reasoning 

**Title (ZH)**: 高级弱监督公式探索在神经符号数学推理中的应用 

**Authors**: Yuxuan Wu, Hideki Nakayama  

**Link**: [PDF](https://arxiv.org/pdf/2502.00629)  

**Abstract**: In recent years, neuro-symbolic methods have become a popular and powerful approach that augments artificial intelligence systems with the capability to perform abstract, logical, and quantitative deductions with enhanced precision and controllability. Recent studies successfully performed symbolic reasoning by leveraging various machine learning models to explicitly or implicitly predict intermediate labels that provide symbolic instructions. However, these intermediate labels are not always prepared for every task as a part of training data, and pre-trained models, represented by Large Language Models (LLMs), also do not consistently generate valid symbolic instructions with their intrinsic knowledge. On the other hand, existing work developed alternative learning techniques that allow the learning system to autonomously uncover optimal symbolic instructions. Nevertheless, their performance also exhibits limitations when faced with relatively huge search spaces or more challenging reasoning problems. In view of this, in this work, we put forward an advanced practice for neuro-symbolic reasoning systems to explore the intermediate labels with weak supervision from problem inputs and final outputs. Our experiments on the Mathematics dataset illustrated the effectiveness of our proposals from multiple aspects. 

**Abstract (ZH)**: 近年来，神经符号方法已成为一种流行且强大的手段，能够增强人工智能系统，使其能够执行更精确和可控的抽象、逻辑和量化推理。近期的研究通过利用各种机器学习模型显式或隐式预测中间标签，成功实现了符号推理。然而，这些中间标签并非所有任务在训练数据中都准备齐全，大型语言模型（LLMs）等预训练模型也不能一贯地生成有效的符号指令。另一方面，现有工作开发了替代学习技术，使学习系统能够自主发现最优的符号指令。但是，当面对更大的搜索空间或更具挑战性的推理问题时，它们也表现出一定的局限性。鉴于此，本文提出了一个先进的神经符号推理系统实践，以从问题输入和最终输出中进行弱监督下的中间标签探索。我们在数学数据集上的实验从多个方面验证了我们提议的有效性。 

---
# MetaOpenFOAM 2.0: Large Language Model Driven Chain of Thought for Automating CFD Simulation and Post-Processing 

**Title (ZH)**: MetaOpenFOAM 2.0：由大型语言模型驱动的推理链以自动化CFD模拟及其后处理 

**Authors**: Yuxuan Chen, Xu Zhu, Hua Zhou, Zhuyin Ren  

**Link**: [PDF](https://arxiv.org/pdf/2502.00498)  

**Abstract**: Computational Fluid Dynamics (CFD) is widely used in aerospace, energy, and biology to model fluid flow, heat transfer, and chemical reactions. While Large Language Models (LLMs) have transformed various domains, their application in CFD remains limited, particularly for complex tasks like post-processing. To bridge this gap, we introduce MetaOpenFOAM 2.0, which leverages Chain of Thought (COT) decomposition and iterative verification to enhance accessibility for non-expert users through natural language inputs. Tested on a new benchmark covering simulation (fluid flow, heat transfer, combustion) and post-processing (extraction, visualization), MetaOpenFOAM 2.0 achieved an Executability score of 6.3/7 and a pass rate of 86.9%, significantly outperforming MetaOpenFOAM 1.0 (2.1/7, 0%). Additionally, it proved cost-efficient, averaging $0.15 per case. An ablation study confirmed that COT-driven decomposition and iterative refinement substantially improved task performance. Furthermore, scaling laws showed that increasing COT steps enhanced accuracy while raising token usage, aligning with LLM post-training scaling trends. These results highlight the transformative potential of LLMs in automating CFD workflows for industrial and research applications. Code is available at this https URL 

**Abstract (ZH)**: 计算流体动力学（CFD）广泛应用于航空航天、能源和生物学领域，用于模拟流体流动、热传递和化学反应。尽管大型语言模型（LLMs）已经在各个领域取得了重大突破，但在CFD领域的应用仍然有限，特别是在复杂任务如后处理方面。为了弥补这一差距，我们引入了MetaOpenFOAM 2.0，它通过利用思维链（COT）分解和迭代验证，增强了非专家用户的使用便利性，使其能够通过自然语言输入操作。MetaOpenFOAM 2.0在涵盖模拟（流体流动、热传递、燃烧）和后处理（提取、可视化）的新基准测试中，获得了6.3/7的可执行性评分和86.9%的通过率，显著优于MetaOpenFOAM 1.0（2.1/7, 0%）。此外，测试结果还显示，MetaOpenFOAM 2.0平均每个案例成本仅为0.15美元。消融实验结果表明，由思维链驱动的分解和迭代细化显著提高了任务性能。进一步的研究还发现，增加思维链步骤可以提高准确性，同时增加令牌使用量，这与大型语言模型后训练扩展趋势相符。这些结果突显了LLMs在自动化CFD工作流方面的变革潜力，特别是在工业和研究应用中。源代码可在以下地址获取：[此链接] 

---
# ALU: Agentic LLM Unlearning 

**Title (ZH)**: ALU：自主的LLM去学习（或ALU：自主的大型语言模型去学习） 

**Authors**: Debdeep Sanyal, Murari Mandal  

**Link**: [PDF](https://arxiv.org/pdf/2502.00406)  

**Abstract**: Information removal or suppression in large language models (LLMs) is a desired functionality, useful in AI regulation, legal compliance, safety, and privacy. LLM unlearning methods aim to remove information on demand from LLMs. Current LLM unlearning methods struggle to balance the unlearning efficacy and utility due to the competing nature of these objectives. Keeping the unlearning process computationally feasible without assuming access to the model weights is an overlooked area. We present the first agentic LLM unlearning (ALU) method, a multi-agent, retrain-free, model-agnostic approach to LLM unlearning that achieves effective unlearning while preserving the utility. Our ALU framework unlearns by involving multiple LLM agents, each designed for a specific step in the unlearning process, without the need to update model weights for any of the agents in the framework. Users can easily request any set of unlearning instances in any sequence, and ALU seamlessly adapts in real time. This is facilitated without requiring any changes in the underlying LLM model. Through extensive experiments on established benchmarks (TOFU, WMDP, WPU) and jailbreaking techniques (many shot, target masking, other languages), we demonstrate that ALU consistently stands out as the most robust LLM unlearning framework among current state-of-the-art methods while incurring a low constant-time cost. We further highlight ALU's superior performance compared to existing methods when evaluated at scale. Specifically, ALU is assessed on up to 1000 unlearning targets, exceeding the evaluation scope of all previously proposed LLM unlearning methods. 

**Abstract (ZH)**: 大规模语言模型（LLMs）中的信息移除或抑制是一项 desirable 功能，对于 AI 规范管理、法律合规性、安全性和隐私保护等方面具有重要意义。LLM 的去学习方法旨在根据需求从 LLM 中移除信息。目前的 LLM 去学习方法在平衡去学习效果和实用性方面存在困难，因为这两个目标之间存在竞争关系。在不假设访问模型权重的情况下，保持去学习过程的计算可行性是一个被忽视的领域。我们提出了首个代理 LLM 去学习（ALU）方法，这是一种多代理、无需重新训练且模型无关的 LLM 去学习方法，能够在有效去学习的同时保持其实用性。我们的 ALU 框架通过涉及多个 LLM 代理来实现去学习，每个代理都针对去学习过程中的特定步骤设计，而不会更新框架内任何代理的模型权重。用户可以轻松地按任何顺序请求任何一组去学习实例，并且 ALU 可在实时中无缝调整，无需对底层 LLM 模型进行任何更改。通过在已建立的标准基准（TOFU、WMDP、WPU）和破解技术（多射击、目标遮蔽、其他语言）上进行广泛的实验，我们展示了 ALU 作为当前最先进的方法中最为稳健的 LLM 去学习框架，同时具有较低的恒定时间成本。此外，我们在大规模评估中进一步展示了 ALU 相对于现有方法具有更优异的性能。具体而言，ALU 在多达 1000 个去学习目标的评估范围上进行了评估，超过了所有先前提出的 LLM 去学习方法的评估范围。 

---
# A Dynamic and High-Precision Method for Scenario-Based HRA Synthetic Data Collection in Multi-Agent Collaborative Environments Driven by LLMs 

**Title (ZH)**: 基于大型语言模型驱动的多agent协作环境中情景基于的人因工程合成数据动态高精度采集方法 

**Authors**: Xingyu Xiao, Peng Chen, Qianqian Jia, Jiejuan Tong, Jingang Liang, Haitao Wang  

**Link**: [PDF](https://arxiv.org/pdf/2502.00022)  

**Abstract**: HRA (Human Reliability Analysis) data is crucial for advancing HRA methodologies. however, existing data collection methods lack the necessary granularity, and most approaches fail to capture dynamic features. Additionally, many methods require expert knowledge as input, making them time-consuming and labor-intensive. To address these challenges, we propose a new paradigm for the automated collection of HRA data. Our approach focuses on key indicators behind human error, specifically measuring workload in collaborative settings. This study introduces a novel, scenario-driven method for workload estimation, leveraging fine-tuned large language models (LLMs). By training LLMs on real-world operational data from high-temperature gas-cooled reactors (HTGRs), we simulate human behavior and cognitive load in real time across various collaborative scenarios. The method dynamically adapts to changes in operator workload, providing more accurate, flexible, and scalable workload estimates. The results demonstrate that the proposed WELLA (Workload Estimation with LLMs and Agents) outperforms existing commercial LLM-based methods in terms of prediction accuracy. 

**Abstract (ZH)**: 人类可靠性分析（HRA）数据对于推进HRA方法具有重要意义。然而，现有的数据收集方法缺乏必要的粒度，且大多数方法无法捕捉动态特征。此外，许多方法需要专家知识作为输入，这使得它们耗时且劳动密集。为应对这些挑战，我们提出了一种新的自动化收集HRA数据的范式。我们的方法集中在影响人类错误的关键指标上，特别关注协作环境中的工作负荷测量。本研究引入了一种基于场景的新颖方法来估算工作负荷，利用细调后的大型语言模型（LLMs）。通过使用高温气冷堆（HTGRs）的真实操作数据对LLMs进行训练，我们实时模拟了各种协作场景下的人类行为和认知负荷。该方法能够动态适应操作员工作负荷的变化，提供更准确、灵活和可扩展的工作负荷估计。结果表明，提出的WELLA（使用LLM和代理的工作负荷估算）在预测准确性方面优于现有的商业LLM基方法。 

---
# Adversarial Reasoning at Jailbreaking Time 

**Title (ZH)**: 在 Jailbreaking 时刻的对抗性推理 

**Authors**: Mahdi Sabbaghi, Paul Kassianik, George Pappas, Yaron Singer, Amin Karbasi, Hamed Hassani  

**Link**: [PDF](https://arxiv.org/pdf/2502.01633)  

**Abstract**: As large language models (LLMs) are becoming more capable and widespread, the study of their failure cases is becoming increasingly important. Recent advances in standardizing, measuring, and scaling test-time compute suggest new methodologies for optimizing models to achieve high performance on hard tasks. In this paper, we apply these advances to the task of model jailbreaking: eliciting harmful responses from aligned LLMs. We develop an adversarial reasoning approach to automatic jailbreaking via test-time computation that achieves SOTA attack success rates (ASR) against many aligned LLMs, even the ones that aim to trade inference-time compute for adversarial robustness. Our approach introduces a new paradigm in understanding LLM vulnerabilities, laying the foundation for the development of more robust and trustworthy AI systems. 

**Abstract (ZH)**: 随着大型语言模型（LLMs）的能力增强和应用范围扩大，对其失败案例的研究变得尤为重要。最近在标准化、测量和扩展测试时计算方面的进展为优化模型以在困难任务上实现高性能提供了新的方法论。本文将这些进展应用于模型脱狱任务：从对齐的LLM中引致有害响应。我们开发了一种对抗推理方法，通过测试时计算实现对许多对齐的LLM的最高成功率（SOTA攻击成功率），即使是对那些旨在通过减少推理时计算来换取对抗鲁棒性的LLM也是如此。我们的方法引入了一种理解和应对LLM脆弱性的新范式，为开发更加稳健和可信赖的AI系统奠定了基础。 

---
# Learning to Generate Unit Tests for Automated Debugging 

**Title (ZH)**: 学习生成单元测试以进行自动化调试 

**Authors**: Archiki Prasad, Elias Stengel-Eskin, Justin Chih-Yao Chen, Zaid Khan, Mohit Bansal  

**Link**: [PDF](https://arxiv.org/pdf/2502.01619)  

**Abstract**: Unit tests (UTs) play an instrumental role in assessing code correctness as well as providing feedback to a large language model (LLM) as it iteratively debugs faulty code, motivating automated test generation. However, we uncover a trade-off between generating unit test inputs that reveal errors when given a faulty code and correctly predicting the unit test output without access to the gold solution. To address this trade-off, we propose UTGen, which teaches LLMs to generate unit test inputs that reveal errors along with their correct expected outputs based on task descriptions and candidate code. We integrate UTGen into UTDebug, a robust debugging pipeline that uses generated tests to help LLMs debug effectively. Since model-generated tests can provide noisy signals (e.g., from incorrectly predicted outputs), UTDebug (i) scales UTGen via test-time compute to improve UT output prediction, and (ii) validates and back-tracks edits based on multiple generated UTs to avoid overfitting. We show that UTGen outperforms UT generation baselines by 7.59% based on a metric measuring the presence of both error-revealing UT inputs and correct UT outputs. When used with UTDebug, we find that feedback from UTGen's unit tests improves pass@1 accuracy of Qwen-2.5 7B on HumanEvalFix and our own harder debugging split of MBPP+ by over 3% and 12.35% (respectively) over other LLM-based UT generation baselines. 

**Abstract (ZH)**: 单元测试（UTs）在评估代码正确性和向大型语言模型（LLMs）提供反馈方面发挥着重要作用，尤其是在迭代调试错误代码的过程中。这激发了自动化测试生成的需求。然而，我们在生成能够揭示错误的单元测试输入和在无金标准答案情况下准确预测单元测试输出之间发现了权衡。为了解决这一权衡，我们提出了UTGen，该方法指导LLMs根据任务描述和候选代码生成能够揭示错误的单元测试输入及其正确的预期输出。我们将UTGen 集成到UTDebug中，这是一个强大的调试管道，利用生成的测试帮助LLMs有效进行调试。由于模型生成的测试可能提供嘈杂的信号（例如，由于错误预测的输出），UTDebug （i）通过在测试时增加计算量来扩展UTGen，以提高单元测试输出预测质量，（ii）基于多个生成的单元测试进行验证和回溯编辑，以避免过拟合。结果显示，UTGen 在一个衡量同时包含揭示错误的单元测试输入和正确单元测试输出的指标上优于基准值7.59%。当与UTDebug结合使用时，我们发现UTGen 提供的单元测试反馈提高了Qwen-2.5 2.5B在HumanEvalFix和我们自己更难的调试分割MBPP+上的pass@1准确率，分别提高了3%和12.35%，超越了其他基于LLM的单元测试生成基准值。 

---
# A Probabilistic Inference Approach to Inference-Time Scaling of LLMs using Particle-Based Monte Carlo Methods 

**Title (ZH)**: 使用粒子蒙特卡洛方法的基于概率推理的大型语言模型推理时缩放方法 

**Authors**: Isha Puri, Shivchander Sudalairaj, Guangxuan Xu, Kai Xu, Akash Srivastava  

**Link**: [PDF](https://arxiv.org/pdf/2502.01618)  

**Abstract**: Large language models (LLMs) have achieved significant performance gains via scaling up model sizes and/or data. However, recent evidence suggests diminishing returns from such approaches, motivating scaling the computation spent at inference time. Existing inference-time scaling methods, usually with reward models, cast the task as a search problem, which tends to be vulnerable to reward hacking as a consequence of approximation errors in reward models. In this paper, we instead cast inference-time scaling as a probabilistic inference task and leverage sampling-based techniques to explore the typical set of the state distribution of a state-space model with an approximate likelihood, rather than optimize for its mode directly. We propose a novel inference-time scaling approach by adapting particle-based Monte Carlo methods to this task. Our empirical evaluation demonstrates that our methods have a 4-16x better scaling rate over our deterministic search counterparts on various challenging mathematical reasoning tasks. Using our approach, we show that Qwen2.5-Math-1.5B-Instruct can surpass GPT-4o accuracy in only 4 rollouts, while Qwen2.5-Math-7B-Instruct scales to o1 level accuracy in only 32 rollouts. Our work not only presents an effective method to inference-time scaling, but also connects the rich literature in probabilistic inference with inference-time scaling of LLMs to develop more robust algorithms in future work. Code and further information is available at this https URL. 

**Abstract (ZH)**: 大型语言模型（LLMs）通过扩大模型规模和/或数据量实现了显著的性能提升。然而，近期的研究证据表明，这种做法的效果逐渐减弱，从而促使我们在推理阶段增加计算量。现有的推理时扩展方法通常使用奖励模型将任务视为搜索问题，这由于奖励模型的近似误差容易导致奖励作弊问题。在本文中，我们相反地将推理时扩展视为一个概率推理任务，并利用基于采样的技术探索状态空间模型状态分布的典型集，而不是直接优化其模式。我们提出了一种新的推理时扩展方法，通过将粒子蒙特卡洛方法适应于这个任务。我们的实验评估表明，与确定性搜索方法相比，我们的方法在多种具有挑战性的数学推理任务中具有4-16倍的更好的扩展率。使用我们的方法，我们展示了Qwen2.5-Math-1.5B-Instruct仅需4次推理即可超越GPT-4o的准确性，而Qwen2.5-Math-7B-Instruct仅需32次推理即可达到o1级的准确性。我们的工作不仅提供了一种有效的推理时扩展方法，还通过将概率推理领域的丰富文献与LLMs的推理时扩展相结合，为未来的更稳健算法开发奠定了基础。更多信息和代码可通过以下链接获取：this https URL 

---
# Self-Improving Transformers Overcome Easy-to-Hard and Length Generalization Challenges 

**Title (ZH)**: 自我提升的Transformer模型克服了从易到难和长度泛化挑战 

**Authors**: Nayoung Lee, Ziyang Cai, Avi Schwarzschild, Kangwook Lee, Dimitris Papailiopoulos  

**Link**: [PDF](https://arxiv.org/pdf/2502.01612)  

**Abstract**: Large language models often struggle with length generalization and solving complex problem instances beyond their training distribution. We present a self-improvement approach where models iteratively generate and learn from their own solutions, progressively tackling harder problems while maintaining a standard transformer architecture. Across diverse tasks including arithmetic, string manipulation, and maze solving, self-improving enables models to solve problems far beyond their initial training distribution-for instance, generalizing from 10-digit to 100-digit addition without apparent saturation. We observe that in some cases filtering for correct self-generated examples leads to exponential improvements in out-of-distribution performance across training rounds. Additionally, starting from pretrained models significantly accelerates this self-improvement process for several tasks. Our results demonstrate how controlled weak-to-strong curricula can systematically teach a model logical extrapolation without any changes to the positional embeddings, or the model architecture. 

**Abstract (ZH)**: 大型语言模型往往在长度泛化和解决超出训练分布的复杂问题实例时表现不佳。我们提出了一种自我改进的方法，其中模型通过迭代生成和从自己的解决方案中学习，逐步解决更难的问题，同时保持标准的变压器架构。在算术、字符串操作和迷宫求解等多样化的任务中，自我改进使模型能够解决远超出其初始训练分布的问题——例如，从10位数加法到100位数加法的泛化，而无需出现饱和现象。我们发现，在某些情况下，筛选正确生成的示例可以在训练轮次中显著提高模型在分布外的表现，使其呈指数级改善。此外，从预训练模型开始可以显著加速这种自我改进过程，尤其是在某些任务中。我们的结果表明，通过控制从弱到强的教学计划，可以在不改变位置嵌入或模型架构的情况下系统地教会模型逻辑外推。 

---
# Verbalized Bayesian Persuasion 

**Title (ZH)**: 口头化的贝叶斯说服 

**Authors**: Wenhao Li, Yue Lin, Xiangfeng Wang, Bo Jin, Hongyuan Zha, Baoxiang Wang  

**Link**: [PDF](https://arxiv.org/pdf/2502.01587)  

**Abstract**: Information design (ID) explores how a sender influence the optimal behavior of receivers to achieve specific objectives. While ID originates from everyday human communication, existing game-theoretic and machine learning methods often model information structures as numbers, which limits many applications to toy games. This work leverages LLMs and proposes a verbalized framework in Bayesian persuasion (BP), which extends classic BP to real-world games involving human dialogues for the first time. Specifically, we map the BP to a verbalized mediator-augmented extensive-form game, where LLMs instantiate the sender and receiver. To efficiently solve the verbalized game, we propose a generalized equilibrium-finding algorithm combining LLM and game solver. The algorithm is reinforced with techniques including verbalized commitment assumptions, verbalized obedience constraints, and information obfuscation. Numerical experiments in dialogue scenarios, such as recommendation letters, courtroom interactions, and law enforcement, validate that our framework can both reproduce theoretical results in classic BP and discover effective persuasion strategies in more complex natural language and multi-stage scenarios. 

**Abstract (ZH)**: 信息设计（ID）探讨了发送者如何影响接收者的最优行为以实现特定目标。尽管ID源自日常生活中的交流，但现有的博弈论和机器学习方法常常将信息结构建模为数字，这限制了许多应用只能局限于玩具博弈。本文利用大型语言模型（LLMs），提出了一种在贝叶斯说服（BP）中的口头化框架，首次将经典BP扩展到包含人类对话的真实世界博弈中。具体而言，我们将BP映射到一个口头化的调解人增强的扩展形式博弈中，其中LLMs实现发送者和接收者。为了高效求解口头化博弈，我们提出了一种结合LLMs和博弈求解器的广义均衡寻找算法，并通过包括口头化的承诺假设、口头化的服从约束和信息混淆等技术来加强该算法。在推荐信、法庭互动和执法等对话场景的数值实验中，验证了我们的框架既能够重现经典BP中的理论结果，又能够在更复杂的自然语言和多阶段场景中发现有效的说服策略。 

---
# MeetMap: Real-Time Collaborative Dialogue Mapping with LLMs in Online Meetings 

**Title (ZH)**: MeetMap：在线会议中使用大语言模型进行实时协作对话映射 

**Authors**: Xinyue Chen, Nathan Yap, Xinyi Lu, Aylin Gunal, Xu Wang  

**Link**: [PDF](https://arxiv.org/pdf/2502.01564)  

**Abstract**: Video meeting platforms display conversations linearly through transcripts or summaries. However, ideas during a meeting do not emerge linearly. We leverage LLMs to create dialogue maps in real time to help people visually structure and connect ideas. Balancing the need to reduce the cognitive load on users during the conversation while giving them sufficient control when using AI, we explore two system variants that encompass different levels of AI assistance. In Human-Map, AI generates summaries of conversations as nodes, and users create dialogue maps with the nodes. In AI-Map, AI produces dialogue maps where users can make edits. We ran a within-subject experiment with ten pairs of users, comparing the two MeetMap variants and a baseline. Users preferred MeetMap over traditional methods for taking notes, which aligned better with their mental models of conversations. Users liked the ease of use for AI-Map due to the low effort demands and appreciated the hands-on opportunity in Human-Map for sense-making. 

**Abstract (ZH)**: 视频会议平台通过转录或总结来线性展示对话内容，然而会议中的思想并不是线性的。我们利用大语言模型（LLMs）在会议过程中实时生成对话地图，帮助人们可视化地结构化和连接想法。在减少用户在交流过程中认知负担的同时，提供足够的AI控制，我们探索了两种不同AI辅助水平的系统变体。在Human-Map系统中，AI生成对话摘要作为节点，用户使用这些节点构建对话地图。在AI-Map系统中，AI生成对话地图，用户可以进行编辑。我们进行了一项针对十对用户的内部实验，比较了两种MeetMap变体和一个基线方法。用户更倾向于使用MeetMap而不是传统的笔记方法，这与他们对对话的认知模型更为一致。用户喜欢AI-Map的易用性，因为其对用户努力的需求较低，同时赞赏在Human-Map中进行意义构建的手动机会。 

---
# What is a Number, That a Large Language Model May Know It? 

**Title (ZH)**: 什么是数字，以至于大语言模型能够了解它？ 

**Authors**: Raja Marjieh, Veniamin Veselovsky, Thomas L. Griffiths, Ilia Sucholutsky  

**Link**: [PDF](https://arxiv.org/pdf/2502.01540)  

**Abstract**: Numbers are a basic part of how humans represent and describe the world around them. As a consequence, learning effective representations of numbers is critical for the success of large language models as they become more integrated into everyday decisions. However, these models face a challenge: depending on context, the same sequence of digit tokens, e.g., 911, can be treated as a number or as a string. What kind of representations arise from this duality, and what are its downstream implications? Using a similarity-based prompting technique from cognitive science, we show that LLMs learn representational spaces that blend string-like and numerical representations. In particular, we show that elicited similarity judgments from these models over integer pairs can be captured by a combination of Levenshtein edit distance and numerical Log-Linear distance, suggesting an entangled representation. In a series of experiments we show how this entanglement is reflected in the latent embeddings, how it can be reduced but not entirely eliminated by context, and how it can propagate into a realistic decision scenario. These results shed light on a representational tension in transformer models that must learn what a number is from text input. 

**Abstract (ZH)**: 数字是人类表示和描述周围世界的基本组成部分。因此，学习有效的数字表示对于大型语言模型的成功至关重要，尤其是当这些模型在日常决策中发挥更大作用时。然而，这些模型面临一个挑战：根据上下文的不同，同一个数字字符序列（例如911）可以被视为一个数字或一个字符串。这种二元性会产生什么样的表示？其下游影响又是什么？

我们利用认知科学中的基于相似性的提示技术，展示了大型语言模型学习的是融合了字符串表示和数字表示的空间。具体来说，我们发现这些模型对整数对的引致相似性判断可以由Levenshtein编辑距离和数值Log-线性距离的组合来捕获，这表明存在一个交织的表示。在一系列实验中，我们展示了这种交织表示在潜在嵌入中的反映情况，以及上下文如何部分减少但不能完全消除这种交织，甚至如何将其传播到一个现实的决策场景中。这些结果揭示了转换器模型中的一个表示上的紧张关系，即模型必须从文本输入中学习到底什么是数字。 

---
# Process Reinforcement through Implicit Rewards 

**Title (ZH)**: 通过隐式奖励强化过程 

**Authors**: Ganqu Cui, Lifan Yuan, Zefan Wang, Hanbin Wang, Wendi Li, Bingxiang He, Yuchen Fan, Tianyu Yu, Qixin Xu, Weize Chen, Jiarui Yuan, Huayu Chen, Kaiyan Zhang, Xingtai Lv, Shuo Wang, Yuan Yao, Xu Han, Hao Peng, Yu Cheng, Zhiyuan Liu, Maosong Sun, Bowen Zhou, Ning Ding  

**Link**: [PDF](https://arxiv.org/pdf/2502.01456)  

**Abstract**: Dense process rewards have proven a more effective alternative to the sparse outcome-level rewards in the inference-time scaling of large language models (LLMs), particularly in tasks requiring complex multi-step reasoning. While dense rewards also offer an appealing choice for the reinforcement learning (RL) of LLMs since their fine-grained rewards have the potential to address some inherent issues of outcome rewards, such as training efficiency and credit assignment, this potential remains largely unrealized. This can be primarily attributed to the challenges of training process reward models (PRMs) online, where collecting high-quality process labels is prohibitively expensive, making them particularly vulnerable to reward hacking. To address these challenges, we propose PRIME (Process Reinforcement through IMplicit rEwards), which enables online PRM updates using only policy rollouts and outcome labels through implict process rewards. PRIME combines well with various advantage functions and forgoes the dedicated reward model training phrase that existing approaches require, substantially reducing the development overhead. We demonstrate PRIME's effectiveness on competitional math and coding. Starting from Qwen2.5-Math-7B-Base, PRIME achieves a 15.1% average improvement across several key reasoning benchmarks over the SFT model. Notably, our resulting model, Eurus-2-7B-PRIME, surpasses Qwen2.5-Math-7B-Instruct on seven reasoning benchmarks with 10% of its training data. 

**Abstract (ZH)**: 在大型语言模型（LLM）的推理时扩展中，密集的过程奖励已被证明是稀疏的结果水平奖励更为有效的替代方案，特别是在需要复杂多步推理的任务中。虽然密集奖励也提供了强化学习（RL）的吸引力选择，因为它们的细粒度奖励有机会解决结果奖励的固有问题，比如训练效率和归因问题，这种潜力尚未得到充分利用。这主要归因于在线训练过程奖励模型（PRM）的挑战，其中收集高质量的过程标签是成本高昂的，因而它们特别容易受到奖励欺诈的影响。为了解决这些挑战，我们提出了一种新的方法——PRIME（通过隐含过程奖励进行过程强化学习），该方法仅通过策略游历和结果标签使用隐含过程奖励，实现在线PRM更新。PRIME能够与各种优势函数很好地结合，并避免了现有方法所需的专门奖励模型训练阶段，从而显著降低了开发成本。我们在竞算数学和编程任务上展示了PRIME的有效性。从Qwen2.5-Math-7B-Base开始，PRIME在多个关键推理基准上实现了15.1%的平均改进，超过了自监督微调（SFT）模型。值得注意的是，我们的最终模型Eurus-2-7B-PRIME仅使用其训练数据的10%，就在七个推理基准上超过了Qwen2.5-Math-7B-Instruct。 

---
# Towards Safer Chatbots: A Framework for Policy Compliance Evaluation of Custom GPTs 

**Title (ZH)**: 向着更安全的聊天机器人：自定义GPT政策合规性评估框架 

**Authors**: David Rodriguez, William Seymour, Jose M. Del Alamo, Jose Such  

**Link**: [PDF](https://arxiv.org/pdf/2502.01436)  

**Abstract**: Large Language Models (LLMs) have gained unprecedented prominence, achieving widespread adoption across diverse domains and integrating deeply into society. The capability to fine-tune general-purpose LLMs, such as Generative Pre-trained Transformers (GPT), for specific tasks has facilitated the emergence of numerous Custom GPTs. These tailored models are increasingly made available through dedicated marketplaces, such as OpenAI's GPT Store. However, their black-box nature introduces significant safety and compliance risks. In this work, we present a scalable framework for the automated evaluation of Custom GPTs against OpenAI's usage policies, which define the permissible behaviors of these systems. Our framework integrates three core components: (1) automated discovery and data collection of models from the GPT store, (2) a red-teaming prompt generator tailored to specific policy categories and the characteristics of each target GPT, and (3) an LLM-as-a-judge technique to analyze each prompt-response pair for potential policy violations.
We validate our framework with a manually annotated ground truth, and evaluate it through a large-scale study with 782 Custom GPTs across three categories: Romantic, Cybersecurity, and Academic GPTs. Our manual annotation process achieved an F1 score of 0.975 in identifying policy violations, confirming the reliability of the framework's assessments. The results reveal that 58.7% of the analyzed models exhibit indications of non-compliance, exposing weaknesses in the GPT store's review and approval processes. Furthermore, our findings indicate that a model's popularity does not correlate with compliance, and non-compliance issues largely stem from behaviors inherited from base models rather than user-driven customizations. We believe this approach is extendable to other chatbot platforms and policy domains, improving LLM-based systems safety. 

**Abstract (ZH)**: 大型语言模型（LLMs）已经获得了前所未有的重视，广泛应用于各个领域，并深深融入社会之中。对通用语言模型，如生成预训练Transformer（GPT）等进行微调以适应特定任务的能力，促进了大量自定义GPT（Custom GPTs）的出现。这些定制模型越来越通过专门的市场平台（如OpenAI的GPT Store）提供，但其黑箱特征引入了重大的安全和合规风险。在本研究中，我们提出了一种可扩展的框架，用于自动化评估自定义GPTs是否符合OpenAI的使用政策，这些政策规定了这些系统的可接受行为。我们的框架整合了三个核心组件：（1）自动发现和收集GPT Store中的模型数据；（2）针对特定政策类别和每个目标GPT的特征定制的红队提示生成器；（3）将LLM作为法官的技术，分析每对提示-响应对以识别潜在的政策违规行为。

我们通过手动标注的验证数据集验证了该框架，并通过一项大规模研究对782个自定义GPTs进行评估，分为三类：浪漫型GPTs、网络安全型GPTs和学术型GPTs。我们的手动标注过程在识别政策违规方面达到了0.975的F1分数，证明了该框架评估的可靠性。研究结果表明，58.7%的分析模型表现出违规迹象，揭示了GPT Store审核和批准流程中的弱点。此外，我们的研究还发现，模型的流行度与合规性之间没有关联，违规问题主要源自基础模型继承的行为，而不是用户驱动的定制化。我们认为这种方法可以扩展到其他聊天机器人平台和政策领域，从而提高基于LLM系统的安全性。 

---
# Eliciting Language Model Behaviors with Investigator Agents 

**Title (ZH)**: 使用调查代理 eliciting 语言模型行为 

**Authors**: Xiang Lisa Li, Neil Chowdhury, Daniel D. Johnson, Tatsunori Hashimoto, Percy Liang, Sarah Schwettmann, Jacob Steinhardt  

**Link**: [PDF](https://arxiv.org/pdf/2502.01236)  

**Abstract**: Language models exhibit complex, diverse behaviors when prompted with free-form text, making it difficult to characterize the space of possible outputs. We study the problem of behavior elicitation, where the goal is to search for prompts that induce specific target behaviors (e.g., hallucinations or harmful responses) from a target language model. To navigate the exponentially large space of possible prompts, we train investigator models to map randomly-chosen target behaviors to a diverse distribution of outputs that elicit them, similar to amortized Bayesian inference. We do this through supervised fine-tuning, reinforcement learning via DPO, and a novel Frank-Wolfe training objective to iteratively discover diverse prompting strategies. Our investigator models surface a variety of effective and human-interpretable prompts leading to jailbreaks, hallucinations, and open-ended aberrant behaviors, obtaining a 100% attack success rate on a subset of AdvBench (Harmful Behaviors) and an 85% hallucination rate. 

**Abstract (ZH)**: 当语言模型接收到自由格式文本的提示时，会表现出复杂多样的行为，这使得难以为其可能输出的空间提供一个全面的表征。我们研究了行为激发的问题，即目标是寻找能够从目标语言模型中诱发特定目标行为（例如虚构或有害响应）的提示。为了在指数级庞大的潜在提示空间中导航，我们训练了调查员模型，使其能够将随机选择的目标行为映射到能够激发其行为的多样化输出分布中，类似于可延时化贝叶斯推断。我们通过监督微调、基于DPO的强化学习以及一个新颖的Frank-Wolfe训练目标，迭代地发现多样化的提示策略。我们的调查员模型揭示了各种有效且人类可解释的提示，这些提示导致了模型突破、虚构行为和开放式的异常行为，在AdvBench（有害行为）的一部分数据集上实现了100%的攻击成功率，并且引发了85%的虚构响应率。 

---
# AtmosSci-Bench: Evaluating the Recent Advance of Large Language Model for Atmospheric Science 

**Title (ZH)**: AtmosSci-Bench: 评估大型语言模型在大气科学领域的 recent advance 

**Authors**: Chenyue Li, Wen Deng, Mengqian Lu, Binhang Yuan  

**Link**: [PDF](https://arxiv.org/pdf/2502.01159)  

**Abstract**: The rapid advancements in large language models (LLMs), particularly in their reasoning capabilities, hold transformative potential for addressing complex challenges in atmospheric science. However, leveraging LLMs effectively in this domain requires a robust and comprehensive evaluation benchmark. To address this need, we present AtmosSci-Bench, a novel benchmark designed to systematically assess LLM performance across five core categories of atmospheric science problems: hydrology, atmospheric dynamics, atmospheric physics, geophysics, and physical oceanography. We employ a template-based question generation framework, enabling scalable and diverse multiple-choice questions curated from graduate-level atmospheric science problems. We conduct a comprehensive evaluation of representative LLMs, categorized into four groups: instruction-tuned models, advanced reasoning models, math-augmented models, and domain-specific climate models. Our analysis provides some interesting insights into the reasoning and problem-solving capabilities of LLMs in atmospheric science. We believe AtmosSci-Bench can serve as a critical step toward advancing LLM applications in climate service by offering a standard and rigorous evaluation framework. Our source codes are currently available at this https URL. 

**Abstract (ZH)**: 大规模语言模型（LLMs）的迅猛发展，特别是在推理能力方面的提升，为解决大气科学领域的复杂挑战提供了变革性的潜力。然而，有效地利用LLMs需要一个稳固且全面的评估基准。为满足这一需求，我们提出了AtmosSci-Bench，这是一个新的基准测试，旨在系统评估LLMs在大气科学五大核心问题领域的性能：水文学、大气动力学、大气物理学、地球物理学和物理海洋学。我们采用基于模板的问题生成框架，能够生成多样化且适用的多项选择题，这些问题是从高级大气科学问题中精选出来的。我们对代表性LLMs进行了全面评估，将它们分为四个组别：指令微调模型、高级推理模型、数学增强模型和领域特定的气候模型。我们的分析为LLMs在大气科学中的推理和问题解决能力提供了有价值的洞察。我们相信，AtmosSci-Bench 可以为推动LLMs在气候服务中的应用提供一个标准且严谨的评估框架。目前，我们的源代码可在此处获取：[此httpsURL]。 

---
# Jailbreaking with Universal Multi-Prompts 

**Title (ZH)**: 使用通用多提示词进行越狱攻击 

**Authors**: Yu-Ling Hsu, Hsuan Su, Shang-Tse Chen  

**Link**: [PDF](https://arxiv.org/pdf/2502.01154)  

**Abstract**: Large language models (LLMs) have seen rapid development in recent years, revolutionizing various applications and significantly enhancing convenience and productivity. However, alongside their impressive capabilities, ethical concerns and new types of attacks, such as jailbreaking, have emerged. While most prompting techniques focus on optimizing adversarial inputs for individual cases, resulting in higher computational costs when dealing with large datasets. Less research has addressed the more general setting of training a universal attacker that can transfer to unseen tasks. In this paper, we introduce JUMP, a prompt-based method designed to jailbreak LLMs using universal multi-prompts. We also adapt our approach for defense, which we term DUMP. Experimental results demonstrate that our method for optimizing universal multi-prompts outperforms existing techniques. 

**Abstract (ZH)**: 近年来，大规模语言模型（LLMs）取得了快速发展，革新了多种应用，并显著提升了便利性和生产效率。然而，伴随其强大的能力，伦理问题和新型攻击，如脱笼攻击（jailbreaking），也应运而生。尽管大多数提示技术侧重于为个别案例优化对抗输入，这在处理大规模数据集时会增加计算成本，但较少的研究关注于训练适用于未见任务的通用攻击者。本文介绍了一种名为JUMP的基于提示的方法，该方法使用通用多提示来脱笼LLMs。我们还为防御目的改进了该方法，称为DUMP。实验结果表明，我们用于优化通用多提示的方法优于现有技术。 

---
# Large Language Model-Enhanced Multi-Armed Bandits 

**Title (ZH)**: 大型语言模型增强的多臂 bandit 算法 

**Authors**: Jiahang Sun, Zhiyong Wang, Runhan Yang, Chenjun Xiao, John C.S. Lui, Zhongxiang Dai  

**Link**: [PDF](https://arxiv.org/pdf/2502.01118)  

**Abstract**: Large language models (LLMs) have been adopted to solve sequential decision-making tasks such as multi-armed bandits (MAB), in which an LLM is directly instructed to select the arms to pull in every iteration. However, this paradigm of direct arm selection using LLMs has been shown to be suboptimal in many MAB tasks. Therefore, we propose an alternative approach which combines the strengths of classical MAB and LLMs. Specifically, we adopt a classical MAB algorithm as the high-level framework and leverage the strong in-context learning capability of LLMs to perform the sub-task of reward prediction. Firstly, we incorporate the LLM-based reward predictor into the classical Thompson sampling (TS) algorithm and adopt a decaying schedule for the LLM temperature to ensure a transition from exploration to exploitation. Next, we incorporate the LLM-based reward predictor (with a temperature of 0) into a regression oracle-based MAB algorithm equipped with an explicit exploration mechanism. We also extend our TS-based algorithm to dueling bandits where only the preference feedback between pairs of arms is available, which requires non-trivial algorithmic modifications. We conduct empirical evaluations using both synthetic MAB tasks and experiments designed using real-world text datasets, in which the results show that our algorithms consistently outperform previous baseline methods based on direct arm selection. Interestingly, we also demonstrate that in challenging tasks where the arms lack semantic meanings that can be exploited by the LLM, our approach achieves considerably better performance than LLM-based direct arm selection. 

**Abstract (ZH)**: 大型语言模型（LLMs）已经被应用于解决多臂老虎机（MAB）等序列决策任务，在这些任务中，LLMs 直接指令选择每一轮应拉动的臂。然而，直接使用LLMs选择臂的方法在许多MAB任务中已被证明是次优的。因此，我们提出了一种新的方法，这种方法结合了经典MAB与LLMs的优点。具体而言，我们将一个经典MAB算法作为高层次框架，并利用LLMs的强大上下文学习能力来完成奖励预测子任务。首先，我们将基于LLM的奖励预测器整合到经典的泰勒斯采样（TS）算法中，并采用温度衰减计划以确保从探索向利采的过渡。接着，我们将基于回归或acles的MAB算法中的LLM基于的奖励预测器（温度设为0）整合进来，并配备了显式的探索机制。我们还将基于TS的算法推广到对战式多臂老虎机（dueling bandits）中，其中只有臂对之间的偏好反馈可用，这需要算法上的复杂调整。我们通过合成的MAB任务和基于实际文本数据集设计的实验进行了实证评估，结果显示我们的算法始终优于基于直接选择臂的方法。有趣的是，我们还展示了在臂缺乏对LLM有益的语义特征的具有挑战性任务中，我们的方法比基于LLM的直接臂选择方法表现出了显著更好的性能。 

---
# Classic4Children: Adapting Chinese Literary Classics for Children with Large Language Model 

**Title (ZH)**: Classic4Children：使用大规模语言模型适应儿童的中国文学经典改编 

**Authors**: Jiali Chen, Xusen Hei, Yuqi Xue, Zihan Wu, Jiayuan Xie, Yi Cai  

**Link**: [PDF](https://arxiv.org/pdf/2502.01090)  

**Abstract**: Chinese literary classics hold significant cultural and educational value, offering deep insights into morality, history, and human nature. These works often include classical Chinese and complex narratives, making them difficult for children to read. To bridge this gap, we introduce a child-friendly literary adaptation (CLA) task to adapt the Chinese literary classic into engaging and accessible text for children. However, recent large language models (LLMs) overlook children's reading preferences (\ie, vivid character portrayals, concise narrative structures, and appropriate readability), which poses challenges in CLA. In this paper, we propose a method called InstructChild, which augments the LLM with these preferences for adaptation. Specifically, we first obtain the characters' personalities and narrative structure as additional information for fine-grained instruction tuning. Then, we devise a readability metric as the reward to align the LLM with the children's reading level. Finally, a lookahead decoding strategy is applied to improve the readability of the generated text during inference. To support the evaluation of CLA task, we construct the Classic4Children dataset, which comprises both the original and child-friendly versions of the Four Great Classical Novels of Chinese literature. Experimental results show that our InstructChild significantly improves automatic and human evaluation performance. 

**Abstract (ZH)**: 中国的文学经典具有重要的文化和教育价值，它们为深入了解道德、历史和人性提供了深刻见解。这些作品通常包含文言文和复杂的叙述结构，使得儿童难以阅读。为了弥合这一差距，我们提出了一个适合儿童的文学改编（CLA）任务，将中国的文学经典改编成具有吸引力和易于理解的文本。然而，最近的大规模语言模型（LLMs）未能考虑儿童的阅读偏好（例如，生动的角色描绘、简洁的叙述结构和适当的可读性），这给CLA带来了挑战。在本文中，我们提出了一种名为InstructChild的方法，该方法通过增加对这些偏好进行细粒度指令调优，来与LLM结合。具体而言，我们首先获取角色的性格和叙述结构作为附加信息，以便进行细粒度指令调优。然后，我们设计了一种可读性指标作为奖励，以使LLM与儿童的阅读水平相一致。最后，在推理过程中应用前瞻解码策略，以提高生成文本的可读性。为了支持CLAT任务的评估，我们构建了Classic4Children数据集，该数据集包含了中国四大古典小说的原文和儿童友好版。实验结果表明，我们的InstructChild方法在自动评价和人工评价中显著提高了性能。 

---
# The Jumping Reasoning Curve? Tracking the Evolution of Reasoning Performance in GPT-[n] and o-[n] Models on Multimodal Puzzles 

**Title (ZH)**: 《跳跃式的推理曲线？在多模态谜题上的GPT-[n]和o-[n]模型推理性能的进化追踪》 

**Authors**: Vernon Y.H. Toh, Yew Ken Chia, Deepanway Ghosal, Soujanya Poria  

**Link**: [PDF](https://arxiv.org/pdf/2502.01081)  

**Abstract**: The releases of OpenAI's o1 and o3 mark a significant paradigm shift in Large Language Models towards advanced reasoning capabilities. Notably, o3 outperformed humans in novel problem-solving and skill acquisition on the Abstraction and Reasoning Corpus for Artificial General Intelligence (ARC-AGI). However, this benchmark is limited to symbolic patterns, whereas humans often perceive and reason about multimodal scenarios involving both vision and language data. Thus, there is an urgent need to investigate advanced reasoning capabilities in multimodal tasks. To this end, we track the evolution of the GPT-[n] and o-[n] series models on challenging multimodal puzzles, requiring fine-grained visual perception with abstract or algorithmic reasoning. The superior performance of o1 comes at nearly 750 times the computational cost of GPT-4o, raising concerns about its efficiency. Our results reveal a clear upward trend in reasoning capabilities across model iterations, with notable performance jumps across GPT-series models and subsequently to o1. Nonetheless, we observe that the o1 model still struggles with simple multimodal puzzles requiring abstract reasoning. Furthermore, its performance in algorithmic puzzles remains poor. We plan to continuously track new models in the series and update our results in this paper accordingly. All resources used in this evaluation are openly available this https URL. 

**Abstract (ZH)**: OpenAI的o1和o3的发布标志着大规模语言模型向高级推理能力的一大范式转变。特别地，o3在人工通用智能抽象和推理语料库（ARC-AGI）的新型问题解决和技能获取方面超过了人类，但该基准仅限于符号模式，而人类通常会在包括视觉和语言数据的多模态情境中感知和推理。因此，有必要进一步研究多模态任务中的高级推理能力。为此，我们追踪了GPT-[n]和o-[n]系列模型在具有细粒度视觉感知和抽象或算法推理需求的挑战性多模态谜题上的发展变化。o1模型的优越性能几乎比GPT-4o高750倍的计算成本，引起对其效率的担忧。我们的研究结果表明，随着模型迭代的进行，推理能力呈现出明显的上升趋势，特别是在GPT系列模型中，随后到o1时性能显著提升。然而，我们注意到o1模型仍然难以应对需要抽象推理的简单多模态谜题，且其在算法谜题中的表现仍较差。计划持续跟进该系列的新模型，并相应地更新本研究中的结果。本研究中使用的所有资源均在此处公开：[提供网址]。 

---
# Knowledge Synthesis of Photosynthesis Research Using a Large Language Model 

**Title (ZH)**: 使用大型语言模型合成光合作用研究知识 

**Authors**: Seungri Yoon, Woosang Jeon, Sanghyeok Choi, Taehyeong Kim, Tae In Ahn  

**Link**: [PDF](https://arxiv.org/pdf/2502.01059)  

**Abstract**: The development of biological data analysis tools and large language models (LLMs) has opened up new possibilities for utilizing AI in plant science research, with the potential to contribute significantly to knowledge integration and research gap identification. Nonetheless, current LLMs struggle to handle complex biological data and theoretical models in photosynthesis research and often fail to provide accurate scientific contexts. Therefore, this study proposed a photosynthesis research assistant (PRAG) based on OpenAI's GPT-4o with retrieval-augmented generation (RAG) techniques and prompt optimization. Vector databases and an automated feedback loop were used in the prompt optimization process to enhance the accuracy and relevance of the responses to photosynthesis-related queries. PRAG showed an average improvement of 8.7% across five metrics related to scientific writing, with a 25.4% increase in source transparency. Additionally, its scientific depth and domain coverage were comparable to those of photosynthesis research papers. A knowledge graph was used to structure PRAG's responses with papers within and outside the database, which allowed PRAG to match key entities with 63% and 39.5% of the database and test papers, respectively. PRAG can be applied for photosynthesis research and broader plant science domains, paving the way for more in-depth data analysis and predictive capabilities. 

**Abstract (ZH)**: 生物数据处理工具和大规模语言模型（LLMs）的发展为植物科学研究中的AI应用打开了新的可能性，有可能显著促进知识整合和研究空白的识别。然而，目前的LLMs在光合作用研究中的复杂生物数据和理论模型处理方面仍存在困难，往往无法提供准确的科学背景。因此，本研究基于OpenAI的GPT-4o和检索增强生成（RAG）技术，提出了一种光合作用研究助手（PRAG），并采用了提示优化。通过提示优化过程中的向量数据库和自动化反馈循环，提高了PRAG对光合作用相关查询响应的准确性和相关性。PRAG在与科学写作相关的五个指标上平均提高了8.7%，在来源透明度上提高了25.4%。此外，其科学深度和领域覆盖范围与光合作用研究论文相当。通过知识图谱，PRAG对其数据库内外的响应进行了结构化，使其能够与数据库中的63%和测试论文中的39.5%的关键实体相匹配。PRAG可以在光合作用研究和更广泛的植物科学领域中应用，为更深入的数据分析和预测能力开辟了道路。 

---
# Embracing Dialectic Intersubjectivity: Coordination of Different Perspectives in Content Analysis with LLM Persona Simulation 

**Title (ZH)**: 拥抱辩证的 intersubjectivity：内容分析中基于大规模语言模型个性模拟的不同视角协调 

**Authors**: Taewoo Kang, Kjerstin Thorson, Tai-Quan Peng, Dan Hiaeshutter-Rice, Sanguk Lee, Stuart Soroka  

**Link**: [PDF](https://arxiv.org/pdf/2502.00903)  

**Abstract**: This study attempts to advancing content analysis methodology from consensus-oriented to coordination-oriented practices, thereby embracing diverse coding outputs and exploring the dynamics among differential perspectives. As an exploratory investigation of this approach, we evaluate six GPT-4o configurations to analyze sentiment in Fox News and MSNBC transcripts on Biden and Trump during the 2020 U.S. presidential campaign, examining patterns across these models. By assessing each model's alignment with ideological perspectives, we explore how partisan selective processing could be identified in LLM-Assisted Content Analysis (LACA). Findings reveal that partisan persona LLMs exhibit stronger ideological biases when processing politically congruent content. Additionally, intercoder reliability is higher among same-partisan personas compared to cross-partisan pairs. This approach enhances the nuanced understanding of LLM outputs and advances the integrity of AI-driven social science research, enabling simulations of real-world implications. 

**Abstract (ZH)**: 本研究旨在从基于共识的方法转向基于协调的方法，推进内容分析方法的发展，从而包容多元编码输出，并探索不同视角之间的动态关系。作为这一方法的探索性研究，我们评估了六种不同的GPT-4配置，以分析2020年美国总统竞选期间福克斯新闻和MSNBC关于拜登和特朗普的转录文本的情感倾向，研究这些模型之间的模式。通过评估每个模型与意识形态视角的一致性，我们探索了 partisan 选择性加工在 LLM 辅助内容分析（LACA）中的识别方式。研究结果表明，当处理政治上一致的内容时，partisan 人格 LLM 更表现出强烈的意识形态偏见。此外，相同党派人格之间的编码者可靠性高于跨党派配对之间的可靠性。这种方法增强了对 LLM 输出的精细理解，促进了基于 AI 的社会科学的研究质量，使实际影响的模拟成为可能。 

---
# Decision-informed Neural Networks with Large Language Model Integration for Portfolio Optimization 

**Title (ZH)**: 带有大型语言模型集成的决策导向神经网络在组合优化中的应用 

**Authors**: Yoontae Hwang, Yaxuan Kong, Stefan Zohren, Yongjae Lee  

**Link**: [PDF](https://arxiv.org/pdf/2502.00828)  

**Abstract**: This paper addresses the critical disconnect between prediction and decision quality in portfolio optimization by integrating Large Language Models (LLMs) with decision-focused learning. We demonstrate both theoretically and empirically that minimizing the prediction error alone leads to suboptimal portfolio decisions. We aim to exploit the representational power of LLMs for investment decisions. An attention mechanism processes asset relationships, temporal dependencies, and macro variables, which are then directly integrated into a portfolio optimization layer. This enables the model to capture complex market dynamics and align predictions with the decision objectives. Extensive experiments on S\&P100 and DOW30 datasets show that our model consistently outperforms state-of-the-art deep learning models. In addition, gradient-based analyses show that our model prioritizes the assets most crucial to decision making, thus mitigating the effects of prediction errors on portfolio performance. These findings underscore the value of integrating decision objectives into predictions for more robust and context-aware portfolio management. 

**Abstract (ZH)**: 本文通过将大型语言模型（LLMs）与决策导向学习相结合，解决了投资组合优化中预测与决策质量之间的关键断层问题。我们从理论和实证两方面证明，单独最小化预测误差会导致投资组合决策的次优结果。我们的目标是利用LLMs在投资决策中的表示能力。通过注意机制处理资产关系、时间依赖性和宏观经济变量，然后将这些信息直接整合到投资组合优化层中，从而使模型能够捕捉复杂市场动态，并使预测与决策目标相一致。在S&P100和DOW30数据集上的广泛实验表明，我们的模型在性能上持续优于最先进的深度学习模型。此外，基于梯度的分析表明，我们的模型优先考虑对决策最重要的资产，从而减轻了预测误差对投资组合业绩的影响。这些发现强调了将决策目标集成到预测中对于更稳健和情境感知的投资组合管理的价值。 

---
# VIKSER: Visual Knowledge-Driven Self-Reinforcing Reasoning Framework 

**Title (ZH)**: VIKSER：视觉知识驱动的自强化推理框架 

**Authors**: Chunbai Zhang, Chao Wang, Yang Zhou, Yan Peng  

**Link**: [PDF](https://arxiv.org/pdf/2502.00711)  

**Abstract**: Visual reasoning refers to the task of solving questions about visual information. Current visual reasoning methods typically employ pre-trained vision-language model (VLM) strategies or deep neural network approaches. However, existing efforts are constrained by limited reasoning interpretability, while hindering by the phenomenon of underspecification in the question text. Additionally, the absence of fine-grained visual knowledge limits the precise understanding of subject behavior in visual reasoning tasks. To address these issues, we propose VIKSER (Visual Knowledge-Driven Self-Reinforcing Reasoning Framework). Specifically, VIKSER, trained using knowledge distilled from large language models, extracts fine-grained visual knowledge with the assistance of visual relationship detection techniques. Subsequently, VIKSER utilizes fine-grained visual knowledge to paraphrase the question with underspecification. Additionally, we design a novel prompting method called Chain-of-Evidence (CoE), which leverages the power of ``evidence for reasoning'' to endow VIKSER with interpretable reasoning capabilities. Meanwhile, the integration of self-reflection technology empowers VIKSER with the ability to learn and improve from its mistakes. Experiments conducted on widely used datasets demonstrate that VIKSER achieves new state-of-the-art (SOTA) results in relevant tasks. 

**Abstract (ZH)**: 视觉推理是指通过解决与视觉信息相关的问题来进行的一种任务。目前的视觉推理方法通常采用预训练的视觉-语言模型（VLM）策略或深度神经网络方法。然而，现有的努力受到推理可解释性的限制，同时由于问题文本中存在的模态性现象而受到阻碍。此外，缺乏精细的视觉知识限制了在视觉推理任务中对主题行为的精确理解。为了应对这些挑战，我们提出了一种名为VIKSER（视觉知识驱动的自我强化推理框架）的方法。具体而言，VIKSER通过从大型语言模型中提炼的知识进行训练，并借助视觉关系检测技术提取精细的视觉知识。随后，VIKSER利用精细的视觉知识对具有模态性的问题进行重述。此外，我们设计了一种新颖的提示方法，称为证据链（Chain-of-Evidence, CoE），该方法利用推理所需“证据”的力量，赋予VIKSER可解释的推理能力。同时，自我反思技术的集成使VIKSER能够从错误中学习和改进。在广泛使用的数据集上进行的实验表明，VIKSER在相关任务中取得了新的最佳性能（SOTA）结果。 

---
# A Survey of Quantized Graph Representation Learning: Connecting Graph Structures with Large Language Models 

**Title (ZH)**: 量化图表示学习综述：连接图结构与大规模语言模型 

**Authors**: Qika Lin, Zhen Peng, Kaize Shi, Kai He, Yiming Xu, Erik Cambria, Mengling Feng  

**Link**: [PDF](https://arxiv.org/pdf/2502.00681)  

**Abstract**: Recent years have witnessed rapid advances in graph representation learning, with the continuous embedding approach emerging as the dominant paradigm. However, such methods encounter issues regarding parameter efficiency, interpretability, and robustness. Thus, Quantized Graph Representation (QGR) learning has recently gained increasing interest, which represents the graph structure with discrete codes instead of conventional continuous embeddings. Given its analogous representation form to natural language, QGR also possesses the capability to seamlessly integrate graph structures with large language models (LLMs). As this emerging paradigm is still in its infancy yet holds significant promise, we undertake this thorough survey to promote its rapid future prosperity. We first present the background of the general quantization methods and their merits. Moreover, we provide an in-depth demonstration of current QGR studies from the perspectives of quantized strategies, training objectives, distinctive designs, knowledge graph quantization, and applications. We further explore the strategies for code dependence learning and integration with LLMs. At last, we give discussions and conclude future directions, aiming to provide a comprehensive picture of QGR and inspire future research. 

**Abstract (ZH)**: 近年来，图表示学习取得了 rapid 的进展，连续性嵌入方法已成为主导范式。然而，这些方法在参数效率、可解释性和鲁棒性方面面临挑战。因此，量化图表示学习（QGR）最近引起了越来越多的关注，这种方法使用离散代码而非传统的连续嵌入来表示图结构。由于其在表示形式上类似于自然语言，QGR 也具备将图结构无缝集成到大型语言模型（LLMs）中的能力。鉴于这一新兴范式仍处于初期阶段但前景广阔，我们进行了一次全面的综述，以促进其未来迅速的发展。首先，我们介绍了量化方法的一般背景及其优势。此外，我们从量化策略、训练目标、独特设计、知识图谱量化和应用等方面，深入探讨了当前的 QGR 研究。我们进一步探讨了代码依赖学习和与 LLMs 集成的策略。最后，我们进行了讨论，并总结了未来的研究方向，旨在提供一个全面的 QGR 视图并启发未来的研究。 

---
# LLM Safety Alignment is Divergence Estimation in Disguise 

**Title (ZH)**: LLM 安全对齐实际上是偏差估计的伪装 

**Authors**: Rajdeep Haldar, Ziyi Wang, Qifan Song, Guang Lin, Yue Xing  

**Link**: [PDF](https://arxiv.org/pdf/2502.00657)  

**Abstract**: We propose a theoretical framework demonstrating that popular Large Language Model (LLM) alignment methods, including Reinforcement Learning from Human Feedback (RLHF) and alternatives, fundamentally function as divergence estimators between aligned (preferred or safe) and unaligned (less-preferred or harmful) distributions. This explains the separation phenomenon between safe and harmful prompts in the model hidden representation after alignment. Inspired by the theoretical results, we identify that some alignment methods are better than others in terms of separation and, introduce a new method, KLDO, and further demonstrate the implication of our theories. We advocate for compliance-refusal datasets over preference datasets to enhance safety alignment, supported by both theoretical reasoning and empirical evidence. Additionally, to quantify safety separation, we leverage a distance metric in the representation space and statistically validate its efficacy as a statistical significant indicator of LLM resilience against jailbreak attacks. 

**Abstract (ZH)**: 我们提出了一种理论框架，认为流行的大型语言模型（LLM）对齐方法，包括基于人类反馈的强化学习（RLHF）及其变体，本质上作为对齐（首选或安全）和未对齐（不太首选或有害）分布之间偏斜度的估计器来运作。这解释了对齐后模型隐藏表示中安全和有害提示之间的分离现象。受理论结果的启发，我们发现一些对齐方法在分离方面优于其他方法，并且引入了一种新方法KLDO，并进一步验证了我们理论的意义。我们主张使用合规拒绝数据集而不是偏好数据集来增强安全性对齐，并且这一主张得到了理论推理和实证证据的支持。此外，为了量化安全性分离，我们利用表示空间中的距离度量，并通过统计验证其作为LLM对脱笼攻击有弹性的显著指标的有效性。 

---
# RPGBENCH: Evaluating Large Language Models as Role-Playing Game Engines 

**Title (ZH)**: RPGBENCH：评估大型语言模型作为角色扮演游戏引擎的能力 

**Authors**: Pengfei Yu, Dongming Shen, Silin Meng, Jaewon Lee, Weisu Yin, Andrea Yaoyun Cui, Zhenlin Xu, Yi Zhu, Xingjian Shi, Mu Li, Alex Smola  

**Link**: [PDF](https://arxiv.org/pdf/2502.00595)  

**Abstract**: We present RPGBench, the first benchmark designed to evaluate large language models (LLMs) as text-based role-playing game (RPG) engines. RPGBench comprises two core tasks: Game Creation (GC) and Game Simulation (GS). In GC, an LLM must craft a valid and playable RPG world using a structured event-state representation, ensuring logical coherence and proper termination conditions. In GS, the LLM simulates interactive gameplay across multiple rounds while consistently updating states and enforcing game rules. To comprehensively assess performance, RPGBench integrates objective and subjective evaluation methodologies. Objective measures verify adherence to event mechanics and check variable updates without requiring human intervention. Subjective measures, such as content interestingness, action quality, and role-playing capability, are evaluated via an LLM-as-a-judge framework, where a strong LLM grades each candidate's outputs. Empirical results demonstrate that state-of-the-art LLMs can produce engaging stories but often struggle to implement consistent, verifiable game mechanics, particularly in long or complex scenarios. By combining structured, rule-based assessments with LLM-based judgments, RPGBench provides a new standard for evaluating how well LLMs can balance creativity, coherence, and complexity in text-based RPGs, opening avenues for more immersive and controllable interactive storytelling. 

**Abstract (ZH)**: 我们提出了RPGBench，这是首个用于评估大型语言模型（LLMs）作为文本角色扮演游戏（RPG）引擎性能的基准测试。RPGBench 包含两个核心任务：游戏创建（GC）和游戏模拟（GS）。在游戏创建任务（GC）中，LLM 必须使用结构化事件状态表示法创作一个有效的、可玩的RPG世界，确保逻辑连贯性和适当的结束条件。在游戏模拟任务（GS）中，LLM 需要在多轮交互中模拟游戏玩法，同时持续更新状态并遵守游戏规则。为了全面评估性能，RPGBench 整合了客观和主观评价方法。客观指标验证了事件机制的遵循情况，并检查变量更新，无需人工干预。主观指标，如内容趣味性、行为质量以及角色扮演能力，则通过LLM作为评委的框架进行评估，其中强大的LLM会对每个候选者的输出进行评分。实验结果表明，最先进的LLM能够生成引人入胜的故事，但在实现一致性和可验证的游戏机制方面常常遇到困难，特别是在长时间或复杂的场景中。通过结合结构化、规则导向的评估与基于LLM的判断，RPGBench 提供了一个新的标准，用于评估LLM在文本RPG中平衡创造力、连贯性和复杂性的能力，从而为更具沉浸感和可控性的交互叙事打开了新的途径。 

---
# Defense Against the Dark Prompts: Mitigating Best-of-N Jailbreaking with Prompt Evaluation 

**Title (ZH)**: 抵抗黑暗提示：通过提示评估减轻最佳模型破解的影响 

**Authors**: Stuart Armstrong, Matija Franklin, Connor Stevens, Rebecca Gorman  

**Link**: [PDF](https://arxiv.org/pdf/2502.00580)  

**Abstract**: Recent work showed Best-of-N (BoN) jailbreaking using repeated use of random augmentations (such as capitalization, punctuation, etc) is effective against all major large language models (LLMs). We have found that $100\%$ of the BoN paper's successful jailbreaks (confidence interval $[99.65\%, 100.00\%]$) and $99.8\%$ of successful jailbreaks in our replication (confidence interval $[99.28\%, 99.98\%]$) were blocked with our Defense Against The Dark Prompts (DATDP) method. The DATDP algorithm works by repeatedly utilizing an evaluation LLM to evaluate a prompt for dangerous or manipulative behaviors--unlike some other approaches, DATDP also explicitly looks for jailbreaking attempts--until a robust safety rating is generated. This success persisted even when utilizing smaller LLMs to power the evaluation (Claude and LLaMa-3-8B-instruct proved almost equally capable). These results show that, though language models are sensitive to seemingly innocuous changes to inputs, they seem also capable of successfully evaluating the dangers of these inputs. Versions of DATDP can therefore be added cheaply to generative AI systems to produce an immediate significant increase in safety. 

**Abstract (ZH)**: 最近的研究表明，通过使用重复的随机增强（如大写、标点等）的最好-N（BoN）破解方法能够有效针对所有主要的大语言模型（LLMs）。我们发现，BoN论文中100%的成功破解实例（置信区间：99.65%至100.00%）和我们在重复实验中99.8%的成功破解实例（置信区间：99.28%至99.98%）均被我们提出的对抗黑暗提示的防御机制（Defense Against The Dark Prompts, DATDP）方法所阻止。DATDP算法通过反复利用评估LLM来评估提示是否存在危险或操控行为——与其他方法不同，DATDP明确寻找破解尝试——直到生成稳健的安全评分。即使使用较小的LLM（如Claude和LLaMa-3-8B-instruct）来执行评估工作，这一成功依然持续有效。这些结果表明，尽管语言模型对输入的细微变化非常敏感，但它们似乎也能够成功评估这些输入的危险性。因此，可以通过经济地将DATDP的版本添加到生成型AI系统中，从而立即显著提高安全性。 

---
# Bridging Internal Probability and Self-Consistency for Effective and Efficient LLM Reasoning 

**Title (ZH)**: 将内部概率与自我一致性相结合以实现高效有效的大型语言模型推理 

**Authors**: Zhi Zhou, Tan Yuhao, Zenan Li, Yuan Yao, Lan-Zhe Guo, Xiaoxing Ma, Yu-Feng Li  

**Link**: [PDF](https://arxiv.org/pdf/2502.00511)  

**Abstract**: Recent advancements in large language models (LLMs) have demonstrated remarkable reasoning capabilities. However, single-shot inference often yields unreliable results for complex reasoning tasks, leading researchers to explore multiple reasoning paths through methods such as perplexity and self-consistency. In this paper, we present the first theoretical error decomposition analysis of these techniques, breaking down their error into estimation error and model error. Our analysis reveals a fundamental trade-off: perplexity methods suffer from substantial model error due to the absence of a proper consistency function, while self-consistency exhibits high estimation error due to a slow error convergence rate. To overcome these limitations, we propose Reasoning-Pruning Perplexity Consistency (RPC). This approach combines Perplexity Consistency, which seamlessly integrates LLM perplexity with self-consistency, and Reasoning Pruning, which eliminates low-probability reasoning paths to effectively prevent the degeneration of estimation error reduction. Theoretical analysis demonstrates that RPC not only accelerates the convergence rate of estimation error to an exponential level but also holds strong potential for further reducing model error. Extensive empirical evaluations on seven benchmark datasets confirm that RPC can significantly improve reasoning performance, sample efficiency, and confidence reliability. 

**Abstract (ZH)**: 近期大语言模型（LLMs）的研究取得了显著的进步，展示了其非凡的推理能力。然而，单次推理在执行复杂推理任务时往往得不到可靠的结果，促使研究者们通过困惑度和自一致性等方法探索多条推理路径。在本文中，我们首次对这些技术进行了理论错误分解分析，将它们的错误分解为估算误差和模型误差。我们的分析揭示了一个基本的权衡：由于缺乏适当的连贯性函数，困惑度方法受到显著的模型误差的影响；而自一致性则因为错误收敛率较慢而导致高估计误差。为了克服这些限制，我们提出了推理裁剪困惑度一致性（Reasoning-Pruning Perplexity Consistency，简称RPC）。该方法结合了困惑度一致性，它可以无缝地将LLM的困惑度与自一致性集成在一起，以及推理裁剪，它通过消除低概率的推理路径来有效防止估计误差减少的退化。理论分析表明，RPC不仅加速了估计误差收敛率到指数级水平，还有进一步减少模型误差的强大潜力。详细的经验评估表明，RPC可以显著提高推理性能、采样效率和置信可靠性。 

---
# MQuant: Unleashing the Inference Potential of Multimodal Large Language Models via Full Static Quantization 

**Title (ZH)**: MQuant：通过全静态量化释放多模态大型语言模型的推理潜力 

**Authors**: JiangYong Yu, Sifan Zhou, Dawei Yang, Shuo Wang, Shuoyu Li, Xing Hu, Chen Xu, Zukang Xu, Changyong Shu, Zhihang Yuan  

**Link**: [PDF](https://arxiv.org/pdf/2502.00425)  

**Abstract**: Multimodal large language models (MLLMs) have garnered widespread attention due to their ability to understand multimodal input. However, their large parameter sizes and substantial computational demands severely hinder their practical deployment and this http URL quantization is an effective way to reduce model size and inference latency, its application to MLLMs remains underexplored. In this paper, we propose MQuant, a post-training quantization (PTQ) framework designed to tackle the unique challenges of multimodal large language models (MLLMs). Conventional quantization often struggles with MLLMs because of (a) high inference latency from large visual token counts, (b) distributional disparities between visual and textual tokens, and (c) extreme outliers introduced by Hadamard-based transformations. To address these issues, MQuant introduces: Modality-Specific Static Quantization (MSQ), assigning distinct static scales for visual vs. textual tokens; Attention-Invariant Flexible Switching (AIFS), reordering tokens to preserve casual attention while eliminating expensive token-wise scale computations; Rotation Magnitude Suppression (RMS), mitigating weight outliers arising from online Hadamard rotations. On five mainstream MLLMs (including Qwen-VL, MiniCPM-V, CogVLM2), MQuant under W4A8 achieves near-floating-point accuracy (<1% degradation) while reducing inference latency by up to 30%, significantly outperforming existing PTQ baselines. Our MQuant effectively bridges the gap for efficient and accurate MLLMs inference in resource-constrained devices. Code will be released. 

**Abstract (ZH)**: 多模态大型语言模型（MLLMs）因其能够理解多模态输入而引起了广泛的关注。然而，它们庞大的参数量和巨大的计算需求严重阻碍了其实际部署。量化是一种有效的方法来减少模型大小和推断延迟，但其在MLLMs中的应用仍然不足。本文我们提出了一种针对多模态大型语言模型（MLLMs）独特挑战的后训练量化（PTQ）框架，MQuant。传统的量化方法常常难以应对MLLMs，因为（a）大量视觉词元导致的高推断延迟，（b）视觉和文本词元之间的分布差异，以及（c）Hadamard变换引入的极端异常值。为了解决这些问题，MQuant 引入了以下方法：模态特定静态量化（MSQ），为视觉和文本词元分配不同的静态缩放比例；注意不变的灵活切换（AIFS），重新排列词元以保持因果注意同时消除昂贵的词元级缩放计算；旋转幅度抑制（RMS），减轻在线Hadamard旋转引起的权重异常值。在五种主流的MLLMs（包括Qwen-VL、MiniCPM-V、CogVLM2）中，MQuant 在W4A8量化下实现了接近浮点精度（<1% 的退化）的同时将推断延迟降低了高达30%，显著优于现有的PTQ基准。我们的MQuant有效地填补了资源受限设备中高效准确的MLLMs推断的空白。代码将开源。 

---
# MarketSenseAI 2.0: Enhancing Stock Analysis through LLM Agents 

**Title (ZH)**: MarketSenseAI 2.0：通过LLM代理增强股票分析 

**Authors**: George Fatouros, Kostas Metaxas, John Soldatos, Manos Karathanassis  

**Link**: [PDF](https://arxiv.org/pdf/2502.00415)  

**Abstract**: MarketSenseAI is a novel framework for holistic stock analysis which leverages Large Language Models (LLMs) to process financial news, historical prices, company fundamentals and the macroeconomic environment to support decision making in stock analysis and selection. In this paper, we present the latest advancements on MarketSenseAI, driven by rapid technological expansion in LLMs. Through a novel architecture combining Retrieval-Augmented Generation and LLM agents, the framework processes SEC filings and earnings calls, while enriching macroeconomic analysis through systematic processing of diverse institutional reports. We demonstrate a significant improvement in fundamental analysis accuracy over the previous version. Empirical evaluation on S\&P 100 stocks over two years (2023-2024) shows MarketSenseAI achieving cumulative returns of 125.9% compared to the index return of 73.5%, while maintaining comparable risk profiles. Further validation on S\&P 500 stocks during 2024 demonstrates the framework's scalability, delivering a 33.8% higher Sortino ratio than the market. This work marks a significant advancement in applying LLM technology to financial analysis, offering insights into the robustness of LLM-driven investment strategies. 

**Abstract (ZH)**: MarketSenseAI是一种全新的综合股票分析框架，利用大规模语言模型（LLMs）处理金融新闻、历史价格、公司基本面以及宏观经济环境，为股票分析和选择提供决策支持。在本文中，我们介绍了由大规模语言模型技术的快速发展推动的MarketSenseAI的最新进展。通过一种新颖的架构结合检索增强生成和LLM代理，该框架处理证监会文件和财报电话会议，同时通过系统处理各种机构报告来丰富宏观经济分析。我们展示了与上一个版本相比在基础分析准确性上的显著改进。对道琼斯100指数股票为期两年（2023-2024年）的实证研究表明，MarketSenseAI在累积回报率为125.9%，而指数回报率为73.5%的情况下，保持了类似的风险水平。进一步在2024年的道琼斯500指数股票上验证，该框架展现了其可扩展性，实现了市场回报率排序难度比率（Sortino比率）高出33.8%的效果。这项工作标志着在金融分析中应用大规模语言模型技术的重要进展，并提供了大规模语言模型驱动投资策略稳健性的洞察。 

---
# OrcaLoca: An LLM Agent Framework for Software Issue Localization 

**Title (ZH)**: OrcaLoca：一个软件问题定位的大型语言模型代理框架 

**Authors**: Zhongming Yu, Hejia Zhang, Yujie Zhao, Hanxian Huang, Matrix Yao, Ke Ding, Jishen Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2502.00350)  

**Abstract**: Recent developments in Large Language Model (LLM) agents are revolutionizing Autonomous Software Engineering (ASE), enabling automated coding, problem fixes, and feature improvements. However, localization -- precisely identifying software problems by navigating to relevant code sections -- remains a significant challenge. Current approaches often yield suboptimal results due to a lack of effective integration between LLM agents and precise code search mechanisms. This paper introduces OrcaLoca, an LLM agent framework that improves accuracy for software issue localization by integrating priority-based scheduling for LLM-guided action, action decomposition with relevance scoring, and distance-aware context pruning. Experimental results demonstrate that OrcaLoca becomes the new open-source state-of-the-art (SOTA) in function match rate (65.33%) on SWE-bench Lite. It also improves the final resolved rate of an open-source framework by 6.33 percentage points through its patch generation integration. 

**Abstract (ZH)**: 近年来，大型语言模型（LLM）代理的发展正在革新自主软件工程（ASE），使自动编码、问题修复和功能改进成为可能。然而，本地化——即通过导航到相关代码段精确识别软件问题——仍然是一个重大挑战。当前的方法由于LLM代理与精确代码搜索机制之间的有效集成不足，往往导致结果不佳。本文介绍了OrcaLoca，这是一种LLM代理框架，通过优先级调度指导动作、动作分解与相关性评分以及距离感知上下文剪枝，提高了软件问题定位的准确性。实验结果表明，OrcaLoca在SWE-bench Lite的功能匹配率上成为新的开源领先方案（65.33%）。此外，通过其修补生成集成，它还将开源框架的最终解决率提高了6.33个百分点。 

---
# UGPhysics: A Comprehensive Benchmark for Undergraduate Physics Reasoning with Large Language Models 

**Title (ZH)**: UGPhysics：大规模语言模型下本科物理推理的综合性基准 

**Authors**: Xin Xu, Qiyun Xu, Tong Xiao, Tianhao Chen, Yuchen Yan, Jiaxin Zhang, Shizhe Diao, Can Yang, Yang Wang  

**Link**: [PDF](https://arxiv.org/pdf/2502.00334)  

**Abstract**: Large language models (LLMs) have demonstrated remarkable capabilities in solving complex reasoning tasks, particularly in mathematics. However, the domain of physics reasoning presents unique challenges that have received significantly less attention. Existing benchmarks often fall short in evaluating LLMs' abilities on the breadth and depth of undergraduate-level physics, underscoring the need for a comprehensive evaluation. To fill this gap, we introduce UGPhysics, a large-scale and comprehensive benchmark specifically designed to evaluate UnderGraduate-level Physics (UGPhysics) reasoning with LLMs. UGPhysics includes 5,520 undergraduate-level physics problems in both English and Chinese, covering 13 subjects with seven different answer types and four distinct physics reasoning skills, all rigorously screened for data leakage. Additionally, we develop a Model-Assistant Rule-based Judgment (MARJ) pipeline specifically tailored for assessing answer correctness of physics problems, ensuring accurate evaluation. Our evaluation of 31 leading LLMs shows that the highest overall accuracy, 49.8% (achieved by OpenAI-o1-mini), emphasizes the necessity for models with stronger physics reasoning skills, beyond math abilities. We hope UGPhysics, along with MARJ, will drive future advancements in AI for physics reasoning. 

**Abstract (ZH)**: 大规模语言模型（LLMs）在解决复杂推理任务方面展现出显著的能力，特别是在数学领域。然而，物理推理领域提出了独特的挑战，这些挑战至今受到的关注较少。现有的基准测试往往在评估LLMs在本科物理水平的广度和深度方面表现不足，突显了进行全面评估的必要性。为填补这一空白，我们提出了UGPhysics，一个大规模且全面的基准测试，专门设计用于评估LLMs在本科水平物理（UGPhysics）推理方面的表现。UGPhysics 包括5,520个本科水平的物理问题，涵盖13个科目，涉及七种不同的答案类型和四种独特的物理推理技能，并且所有数据经过严格审查以确保无泄露。此外，我们还开发了一种专门用于评估物理问题答案正确性的模型助手基于规则的判断（MARJ）管道，以确保准确的评估。我们对31个领先的LLMs的评估显示，最高总体准确率为49.8%（由OpenAI-o1-mini取得），强调了需要具有更强物理推理能力的模型的重要性，而不仅仅是数学能力。我们希望UGPhysics以及MARJ能够推动未来在物理推理方面的AI技术进步。 

---
# From Few to Many: Self-Improving Many-Shot Reasoners Through Iterative Optimization and Generation 

**Title (ZH)**: 从少量到大量：通过迭代优化与生成实现自我改进的多-shot推理器 

**Authors**: Xingchen Wan, Han Zhou, Ruoxi Sun, Hootan Nakhost, Ke Jiang, Sercan Ö. Arık  

**Link**: [PDF](https://arxiv.org/pdf/2502.00330)  

**Abstract**: Recent advances in long-context large language models (LLMs) have led to the emerging paradigm of many-shot in-context learning (ICL), where it is observed that scaling many more demonstrating examples beyond the conventional few-shot setup in the context can lead to performance benefits. However, despite its promise, it is unclear what aspects dominate the benefits and whether simply scaling to more examples is the most effective way of improving many-shot ICL. In this work, we first provide an analysis of the factors driving many-shot ICL, and we find that 1) many-shot performance can still be attributed to often a few disproportionately influential examples and 2) identifying such influential examples ("optimize") and using them as demonstrations to regenerate new examples ("generate") can lead to further improvements. Inspired by the findings, we propose BRIDGE, an algorithm that alternates between the optimize step with Bayesian optimization to discover the influential sets of examples and the generate step to reuse this set to expand the reasoning paths of the examples back to the many-shot regime automatically. On Gemini, Claude, and Mistral LLMs of different sizes, we show that BRIDGE to significant improvements across a diverse set of tasks, including symbolic reasoning, numerical reasoning, and code generation. 

**Abstract (ZH)**: 近年来，长上下文语言模型（LLMs）的进展推动了多示例上下文学习（ICL）新兴范式的出现，在这种范式中，观察到在上下文中超出传统少量示例设置的情况下加入更多示例可以带来性能提升。然而，尽管这种做法具有潜力，但尚不清楚是什么因素主导了这种益处，以及是否简单地增加更多示例是最有效的提高多示例ICL性能的方法。在这项工作中，我们首先分析了驱动多示例ICL的因素，并发现1）多示例性能仍然往往可归因于少数具有显著影响力的示例；2）识别这些有影响力的示例（优化）并利用它们生成新的示例（生成）可以进一步提高性能。基于这些发现，我们提出了一种交替进行“优化”步骤和“生成”步骤的BRIDGE算法。在“优化”步骤中，使用贝叶斯优化来发现具有影响力的示例集；在“生成”步骤中，利用这些示例集自动扩展示例的推理路径，以返回到多示例范式。我们在不同规模的Gemini、Claude和Mistral语言模型上展示了BRIDGE在包括符号推理、数值推理和代码生成等不同任务上的显著改进。 

---
# CoddLLM: Empowering Large Language Models for Data Analytics 

**Title (ZH)**: CoddLLM：赋能大型语言模型进行数据分析 

**Authors**: Jiani Zhang, Hengrui Zhang, Rishav Chakravarti, Yiqun Hu, Patrick Ng, Asterios Katsifodimos, Huzefa Rangwala, George Karypis, Alon Halevy  

**Link**: [PDF](https://arxiv.org/pdf/2502.00329)  

**Abstract**: Large Language Models (LLMs) have the potential to revolutionize data analytics by simplifying tasks such as data discovery and SQL query synthesis through natural language interactions. This work serves as a pivotal first step toward the development of foundation models explicitly designed for data analytics applications. To propel this vision forward, we unveil a new data recipe for post-training LLMs, enhancing their comprehension of data management and empowering them to tackle complex real-world analytics tasks. Specifically, our innovative approach includes a scalable synthetic data generation method that enables the creation of a broad spectrum of topics centered on data representation and manipulation. Furthermore, we introduce two new tasks that seamlessly bridge tables and text. We show that such tasks can enhance models' understanding of schema creation and the nuanced translation between natural language and tabular data. Leveraging this data recipe, we post-train a new foundation model, named CoddLLM, based on Mistral-NeMo-12B. To assess the language understanding and reasoning capabilities of LLMs in the realm of data analytics, we contribute AnalyticsMMLU, a benchmark containing thousands of multiple-choice questions on databases, data analysis, and machine learning. Our focus on data discovery, has resulted in the contribution of three comprehensive benchmarks that address both database and data lake scenarios. CoddLLM not only excels in performance but also sets a new standard, achieving the highest average accuracy across eight datasets. It outperforms GPT-3.5-Turbo on AnalyticsMMLU, exceeding GPT-4o by 12.1% in table selection and showing an average improvement of 24.9% in Text-to-SQL compared to the base model. 

**Abstract (ZH)**: 大规模语言模型（LLMs）有望通过自然语言交互简化数据发现和SQL查询合成等任务，从而彻底改变数据分析。本研究是旨在开发明确面向数据应用的基础模型的重要第一步。为了推动这一愿景的实现，我们提出了一个新的数据食谱，用于后训练LLMs，增强其对数据管理的理解，并使其能够应对复杂的现实世界分析任务。具体而言，我们提出了一种创新的方法，包括一种可扩展的合成数据生成方法，能够生成涵盖数据表示和操作广泛主题的数据集。此外，我们引入了两个新的任务，无缝连接表格和文本。我们展示了这些任务如何提高模型对模式创建和自然语言与表结构数据之间细微转换的理解。借助这一数据食谱，我们基于Mistral-NeMo-12B训练了一个新的基础模型，命名为CoddLLM。为了评估LLMs在数据分析领域的语言理解和推理能力，我们贡献了AnalyticsMMLU基准测试，包含数千道关于数据库、数据分析和机器学习的多选题。我们专注于数据发现，提出并贡献了三项全面的基准测试，分别解决了数据库和数据湖场景。CoddLLM不仅在性能上表现出色，而且还设定了新的标准，其在八个数据集上的平均准确率达到最高。在AnalyticsMMLU基准测试中，CoddLLM的表现超过了GPT-3.5-Turbo，其在表选择上的性能比GPT-4高出12.1%，而在文本到SQL任务上的平均改进达到了24.9%，超过基线模型。 

---
# Distributive Fairness in Large Language Models: Evaluating Alignment with Human Values 

**Title (ZH)**: 大型语言模型中的分配公平性：评估与人类价值观的一致性 

**Authors**: Hadi Hosseini, Samarth Khanna  

**Link**: [PDF](https://arxiv.org/pdf/2502.00313)  

**Abstract**: The growing interest in employing large language models (LLMs) for decision-making in social and economic contexts has raised questions about their potential to function as agents in these domains. A significant number of societal problems involve the distribution of resources, where fairness, along with economic efficiency, play a critical role in the desirability of outcomes. In this paper, we examine whether LLM responses adhere to fundamental fairness concepts such as equitability, envy-freeness, and Rawlsian maximin, and investigate their alignment with human preferences. We evaluate the performance of several LLMs, providing a comparative benchmark of their ability to reflect these measures. Our results demonstrate a lack of alignment between current LLM responses and human distributional preferences. Moreover, LLMs are unable to utilize money as a transferable resource to mitigate inequality. Nonetheless, we demonstrate a stark contrast when (some) LLMs are tasked with selecting from a predefined menu of options rather than generating one. In addition, we analyze the robustness of LLM responses to variations in semantic factors (e.g. intentions or personas) or non-semantic prompting changes (e.g. templates or orderings). Finally, we highlight potential strategies aimed at enhancing the alignment of LLM behavior with well-established fairness concepts. 

**Abstract (ZH)**: 随着将大规模语言模型（LLMs）应用于社会和经济背景下的决策问题研究逐渐引起关注，人们开始探讨这些模型是否能在这些领域中充当代理角色。许多社会问题涉及到资源分配，公平性不仅在经济学效率方面起着关键作用，还在非经济性方面同样重要。在本文中，我们探讨LLM响应是否符合基本的公平性概念，如公平性、无嫉妒性和罗尔斯的最小最大化原则，并调查这些响应是否与人类偏好相一致。我们评估了多个LLM的表现，并提供了一个它们在反映这些度量标准方面能力的对比基准。我们的结果显示，当前的LLM响应与人类的分配偏好之间存在脱节。此外，当（某些）LLMs从预定义的选项菜单中进行选择而不是生成时，这些模型无法利用金钱作为可转移的资源来缓解不平等。然而，当LLMs被要求从预定义的选项菜单中进行选择而非生成新菜单时，我们发现了一个显著的区别。此外，我们分析了LLM响应对语义因素（如意图或人设）或非语义提示变化（如模板或顺序）变化的鲁棒性。最后，我们提出了几种策略，旨在提高LLM行为与广泛认可的公平性概念的对齐程度。 

---
# Riddle Me This! Stealthy Membership Inference for Retrieval-Augmented Generation 

**Title (ZH)**: 这个谜题给你！隐秘的成员推理在检索增强生成中的应用 

**Authors**: Ali Naseh, Yuefeng Peng, Anshuman Suri, Harsh Chaudhari, Alina Oprea, Amir Houmansadr  

**Link**: [PDF](https://arxiv.org/pdf/2502.00306)  

**Abstract**: Retrieval-Augmented Generation (RAG) enables Large Language Models (LLMs) to generate grounded responses by leveraging external knowledge databases without altering model parameters. Although the absence of weight tuning prevents leakage via model parameters, it introduces the risk of inference adversaries exploiting retrieved documents in the model's context. Existing methods for membership inference and data extraction often rely on jailbreaking or carefully crafted unnatural queries, which can be easily detected or thwarted with query rewriting techniques common in RAG systems. In this work, we present Interrogation Attack (IA), a membership inference technique targeting documents in the RAG datastore. By crafting natural-text queries that are answerable only with the target document's presence, our approach demonstrates successful inference with just 30 queries while remaining stealthy; straightforward detectors identify adversarial prompts from existing methods up to ~76x more frequently than those generated by our attack. We observe a 2x improvement in TPR@1%FPR over prior inference attacks across diverse RAG configurations, all while costing less than $0.02 per document inference. 

**Abstract (ZH)**: 检索增强生成（RAG）使大型语言模型（LLMs）能够通过利用外部知识数据库生成基于事实的回应，而无需改变模型参数。尽管缺少权重调整可以防止通过模型参数泄露信息，但它增加了恶意推理者利用检索到的文档在模型上下文中提取信息的风险。现有的成员推理和数据提取方法往往依赖于“jailbreaking”或精心构造的不自然查询，这些方法很容易被RAG系统中常见的查询重写技术检测或阻止。在本研究中，我们提出了询问攻击（Interrogation Attack，IA），这是一种针对RAG数据存储库中文档的成员推理技术。通过构造只能通过目标文档的存在才能回答的自然文本查询，我们的方法仅使用30个查询就成功实现了推理，并且保持了隐蔽性；现有的检测器能够比我们的攻击生成的提示更频繁地识别出8到76倍的恶意提示。我们的方法在多种RAG配置中实现了1%假阳性率下的召回率提升2倍，同时每文档推理成本低于0.02美元。 

---
# Estimating LLM Uncertainty with Logits 

**Title (ZH)**: 使用Logits估计大语言模型的不确定性 

**Authors**: Huan Ma, Jingdong Chen, Guangyu Wang, Changqing Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2502.00290)  

**Abstract**: In recent years, Large Language Models (LLMs) have seen remarkable advancements and have been extensively integrated across various fields. Despite their progress, LLMs are prone to hallucinations, producing responses that may not be dependable if the models lack sufficient grounding knowledge. To mitigate this issue, methods for estimating uncertainty have been adopted, with a focus on critical tokens as indicators of reliability. Nevertheless, probability-based approaches have shown limitations in assessing token-level reliability due to the erosion of evidence strength information acquired during training. In this paper, we introduce Logits-induced Token Uncertainty (LogU), a novel framework designed to estimate token-specific uncertainty in LLMs in real time, without the need for multiple sampling rounds. By leveraging evidence modeling for the implementation of LogU, we utilize the derived uncertainty measures to steer downstream tasks. Our experimental findings highlight the substantial effectiveness and potential of LogU, marking a significant advancement in addressing the challenge of model hallucinations. 

**Abstract (ZH)**: 近年来，大型语言模型（LLMs）取得了显著的进步，并广泛应用于各个领域。尽管取得了一定的进展，但LLMs仍然容易出现幻觉现象，即生成的响应可能不可靠，特别是在模型缺乏足够的背景知识时。为解决这一问题，人们采用了估算不确定性的方法，并将关键令牌作为可靠性的指示器作为重点。然而，基于概率的方法在评估令牌级可靠性时显示出局限性，因为它们无法保留训练过程中获得的证据强度信息。本文介绍了一种名为Logits-induced Token Uncertainty（LogU）的新框架，该框架旨在在无需多次采样轮次的情况下，实时估计LLMs中令牌特定的不确定性。通过利用证据建模来实现LogU，我们利用衍生的不确定性度量来引导下游任务。我们的实验结果突显了LogU的显著效果和潜力，标志着在解决模型幻觉问题方面取得了重要进展。 

---
# Should You Use Your Large Language Model to Explore or Exploit? 

**Title (ZH)**: 你应该使用你的大规模语言模型来探索还是利用？ 

**Authors**: Keegan Harris, Aleksandrs Slivkins  

**Link**: [PDF](https://arxiv.org/pdf/2502.00225)  

**Abstract**: We evaluate the ability of the current generation of large language models (LLMs) to help a decision-making agent facing an exploration-exploitation tradeoff. We use LLMs to explore and exploit in silos in various (contextual) bandit tasks. We find that while the current LLMs often struggle to exploit, in-context mitigations may be used to substantially improve performance for small-scale tasks. However even then, LLMs perform worse than a simple linear regression. On the other hand, we find that LLMs do help at exploring large action spaces with inherent semantics, by suggesting suitable candidates to explore. 

**Abstract (ZH)**: 我们评估当前一代大规模语言模型（LLMs）在面对探索-利用权衡时辅助决策代理的能力。我们使用LLMs分别进行探索和利用，应用于各种（上下文相关的）bandit任务。我们发现，尽管当前的LLMs在利用方面常常表现不佳，但在某些小型任务中，通过上下文内的缓解措施可以显著改善性能。然而，即使在这种情况下，LLMs的表现仍然不如简单的线性回归模型。另一方面，我们发现LLMs在探索具有内在语义的大动作空间时确实有所帮助，它们能够建议合适的探索候选对象。 

---
# A Three-Branch Checks-and-Balances Frameworkfor Context-Aware Ethical Alignment of Large Language Models 

**Title (ZH)**: 一种上下文感知的大语言模型伦理对齐的三支权衡框架 

**Authors**: Edward Y. Chang  

**Link**: [PDF](https://arxiv.org/pdf/2502.00136)  

**Abstract**: This paper introduces a three-branch checks-and-balances framework for ethical alignment of Large Language Models (LLMs), inspired by governmental systems. It implements three independent yet interacting components: LLMs as the executive branch for knowledge generation, DIKE as the legislative branch establishing ethical guardrails, and ERIS as the judicial branch for contextual interpretation. The adversarial DIKE-ERIS duality enables adaptation to diverse cultural contexts while upholding consistent ethical principles. This architecture addresses limitations of reinforcement learning with human feedback (RLHF) by providing interpretable, adaptable, and culturally-aware ethical reasoning. Through self-supervised learning and adversarial testing, our framework demonstrates how emotional modeling can guide linguistic behaviors toward ethical outcomes while preserving independence across knowledge generation, ethical oversight, and contextual interpretation. 

**Abstract (ZH)**: 本文介绍了受到政府体系启发的大语言模型（LLMs）伦理对齐的三支制衡框架。该框架包括三个独立又相互作用的组成部分：作为执行分支的LLMs进行知识生成、作为立法分支的DIKE建立伦理准则、以及作为司法分支的ERIS进行上下文解释。对抗性的DIKE-ERIS二元性既能够适应不同的文化背景，又能坚持一致的伦理原则。该架构通过补充强化学习具有人类反馈的局限性（RLHF），提供了可解释、可适应和文化意识的伦理推理。通过自监督学习和对抗性测试，我们的框架展示了情感建模如何引导语言行为趋向伦理结果，同时在知识生成、伦理监督和上下文解释之间保持独立性。 

---
# Can AI Solve the Peer Review Crisis? A Large Scale Experiment on LLM's Performance and Biases in Evaluating Economics Papers 

**Title (ZH)**: 人工智能能解决同行评审危机吗？一项大型实验探究LLM在评估经济论文时的表现与偏见 

**Authors**: Pat Pataranutaporn, Nattavudh Powdthavee, Pattie Maes  

**Link**: [PDF](https://arxiv.org/pdf/2502.00070)  

**Abstract**: We investigate whether artificial intelligence can address the peer review crisis in economics by analyzing 27,090 evaluations of 9,030 unique submissions using a large language model (LLM). The experiment systematically varies author characteristics (e.g., affiliation, reputation, gender) and publication quality (e.g., top-tier, mid-tier, low-tier, AI generated papers). The results indicate that LLMs effectively distinguish paper quality but exhibit biases favoring prominent institutions, male authors, and renowned economists. Additionally, LLMs struggle to differentiate high-quality AI-generated papers from genuine top-tier submissions. While LLMs offer efficiency gains, their susceptibility to bias necessitates cautious integration and hybrid peer review models to balance equity and accuracy. 

**Abstract (ZH)**: 我们通过分析9030篇独特提交论文的27,090份评估结果，使用大型语言模型（LLM）来探讨人工智能是否能解决经济学同行评审危机。实验系统地变化了作者特征（如：隶属关系、声誉、性别）以及出版质量（如：顶级、中等级、低等级、AI生成的文章）。结果表明，LLM能够有效区分论文质量，但表现出对知名机构、男性作者和著名经济学家的偏见。此外，LLM难以区分高质量的AI生成论文和真正的顶级论文。虽然LLM提供了效率上的改进，但其易受偏见的影响需要谨慎整合，并采用混合同行评审模式以平衡公平性和准确性。 

---
# A Multi-Layered Large Language Model Framework for Disease Prediction 

**Title (ZH)**: 一种多层次大型语言模型框架用于疾病预测 

**Authors**: Malak Mohamed, Rokaia Emad, Ali Hamdi  

**Link**: [PDF](https://arxiv.org/pdf/2502.00063)  

**Abstract**: Social telehealth has revolutionized healthcare by enabling patients to share symptoms and receive medical consultations remotely. Users frequently post symptoms on social media and online health platforms, generating a vast repository of medical data that can be leveraged for disease classification and symptom severity assessment. Large language models (LLMs), such as LLAMA3, GPT-3.5 Turbo, and BERT, process complex medical data to enhance disease classification. This study explores three Arabic medical text preprocessing techniques: text summarization, text refinement, and Named Entity Recognition (NER). Evaluating CAMeL-BERT, AraBERT, and Asafaya-BERT with LoRA, the best performance was achieved using CAMeL-BERT with NER-augmented text (83% type classification, 69% severity assessment). Non-fine-tuned models performed poorly (13%-20% type classification, 40%-49% severity assessment). Integrating LLMs into social telehealth systems enhances diagnostic accuracy and treatment outcomes. 

**Abstract (ZH)**: 社交媒体远程医疗通过允许患者远程共享症状和接受医疗咨询，彻底改变了医疗保健领域。用户经常在社交媒体和在线健康平台上发布症状，生成了一个巨大的医疗数据资源库，可以用于疾病分类和症状严重程度评估。大型语言模型（LLMs），如LLAMA3、GPT-3.5 Turbo和BERT，处理复杂的医疗数据以提高疾病分类的准确性。本研究探讨了三种阿拉伯医学文本预处理技术：文本摘要、文本优化和命名实体识别（NER）。通过评估CAMeL-BERT、AraBERT和Asafaya-BERT与LoRA相结合的模型，使用NER增强的CAMeL-BERT文本表现出最佳性能（类型分类准确率为83%，严重程度评估为69%）。未经微调的模型表现较差（类型分类准确率为13%-20%，严重程度评估为40%-49%）。将LLMs集成到社交媒体远程医疗系统中可以提升诊断准确性和治疗效果。 

---
# Large Language Models are Few-shot Multivariate Time Series Classifiers 

**Title (ZH)**: 大型语言模型是 Few-Shot 多变量时间序列分类器 

**Authors**: Yakun Chen, Zihao Li, Chao Yang, Xianzhi Wang, Guandong Xu  

**Link**: [PDF](https://arxiv.org/pdf/2502.00059)  

**Abstract**: Large Language Models (LLMs) have been extensively applied in time series analysis. Yet, their utility in the few-shot classification (i.e., a crucial training scenario due to the limited training data available in industrial applications) concerning multivariate time series data remains underexplored. We aim to leverage the extensive pre-trained knowledge in LLMs to overcome the data scarcity problem within multivariate time series. Specifically, we propose LLMFew, an LLM-enhanced framework to investigate the feasibility and capacity of LLMs for few-shot multivariate time series classification. This model introduces a Patch-wise Temporal Convolution Encoder (PTCEnc) to align time series data with the textual embedding input of LLMs. We further fine-tune the pre-trained LLM decoder with Low-rank Adaptations (LoRA) to enhance its feature representation learning ability in time series data. Experimental results show that our model outperformed state-of-the-art baselines by a large margin, achieving 125.2% and 50.2% improvement in classification accuracy on Handwriting and EthanolConcentration datasets, respectively. Moreover, our experimental results demonstrate that LLM-based methods perform well across a variety of datasets in few-shot MTSC, delivering reliable results compared to traditional models. This success paves the way for their deployment in industrial environments where data are limited. 

**Abstract (ZH)**: 大规模语言模型（LLMs）在时间序列分析中已被广泛应用于各种场景。然而，在涉及多变量时间序列数据的少量样本分类（即由于工业应用中可用训练数据量有限而成为一个关键的训练场景）方面，其应用潜力仍未得到充分探索。我们旨在利用大规模语言模型中丰富的预训练知识，以克服多变量时间序列中的数据稀缺问题。具体而言，我们提出了一种LLM增强框架LLMFew，以探究LLMs在少量样本多变量时间序列分类中的可行性和能力。该模型引入了一种基于补丁的时序卷积编码器（PTCEnc），用于将时间序列数据与LLM的文本嵌入输入对齐。进一步地，我们通过低秩适应（LoRA）微调预训练的LLM解码器，以增强其在时间序列数据中的特征表示学习能力。实验结果显示，我们的模型在Handwriting和EthanolConcentration数据集上的分类准确率上显著优于现有最先进的基线方法，分别提高了125.2%和50.2%。此外，我们的实验结果表明，基于LLM的方法在少量样本多变量时间序列分类（Few-shot MTSC）的各种数据集上表现良好，相比传统模型，可以提供可靠的性能。这一成功为这些方法在数据有限的工业环境中的应用铺平了道路。 

---
# Contextually Entangled Gradient Mapping for Optimized LLM Comprehension 

**Title (ZH)**: 上下文交织的梯度映射方法以优化大型语言模型理解力 

**Authors**: Colin Sisate, Alistair Goldfinch, Vincent Waterstone, Sebastian Kingsley, Mariana Blackthorn  

**Link**: [PDF](https://arxiv.org/pdf/2502.00048)  

**Abstract**: Contextually Entangled Gradient Mapping (CEGM) introduces a new approach to gradient optimization, redefining the relationship between contextual embeddings and gradient updates to enhance semantic coherence and reasoning capabilities in neural architectures. By treating gradients as dynamic carriers of contextual dependencies rather than isolated numerical entities, the proposed methodology bridges critical gaps in existing optimization strategies. The integration of entangled gradient dynamics into a loss regularization framework demonstrated significant improvements in tasks involving long-form reasoning, contextual retention, and adaptability to unseen domains. Experimental evaluations showed that the CEGM-enhanced model consistently outperformed baseline approaches, achieving higher accuracy in token-level predictions and greater resilience to noisy inputs. Practical implementations involved modifications to training pipelines, introducing entanglement layers and dynamic coefficient adjustments that seamlessly align with existing architectures. Results further highlighted reductions in semantic drift during sequential transformations and improvements in embedding coherence across paraphrased sentences, showing the robustness and versatility of the proposed methodology. The findings demonstrate the broader implications of gradient entanglement for both theoretical advancements and practical applications in optimization strategies. 

**Abstract (ZH)**: 情境纠缠梯度映射（Contextually Entangled Gradient Mapping, CEGM）引入了一种新的梯度优化方法，重新定义了情境嵌入与梯度更新之间的关系，以增强神经架构中的语义一致性与推理能力。通过将梯度视为动态的情境依赖性载体，而非孤立的数值实体，所提出的方案填补了现有优化策略中的关键空白。在损失正则化框架中集成纠缠梯度动态性显示出了在涉及长形式推理、情境保留及适应未知领域方面的显著改进。实验评估表明，CEGM增强的模型在基线方法中表现更优，能够在亚词级别的预测中实现更高的准确性，并且对嘈杂输入具有更强的鲁棒性。实际实现涉及对训练管道的修改，引入纠缠层和动态系数调整，这些调整能无缝与现有架构兼容。进一步的结果表明，在顺序变换过程中减少了语义漂移，并且在同义句嵌入一致性方面有所提升，展示了所提出的方案的稳健性和灵活性。研究结果揭示了梯度纠缠对优化策略的理论进步和实际应用具有更广泛的潜在影响。 

---
# AlphaSharpe: LLM-Driven Discovery of Robust Risk-Adjusted Metrics 

**Title (ZH)**: AlphaSharpe: LLM驱动的稳健风险调整指标发现 

**Authors**: Kamer Ali Yuksel, Hassan Sawaf  

**Link**: [PDF](https://arxiv.org/pdf/2502.00029)  

**Abstract**: Financial metrics like the Sharpe ratio are pivotal in evaluating investment performance by balancing risk and return. However, traditional metrics often struggle with robustness and generalization, particularly in dynamic and volatile market conditions. This paper introduces AlphaSharpe, a novel framework leveraging large language models (LLMs) to iteratively evolve and optimize financial metrics. AlphaSharpe generates enhanced risk-return metrics that outperform traditional approaches in robustness and correlation with future performance metrics by employing iterative crossover, mutation, and evaluation. Key contributions of this work include: (1) an innovative use of LLMs for generating and refining financial metrics inspired by domain-specific knowledge, (2) a scoring mechanism to ensure the evolved metrics generalize effectively to unseen data, and (3) an empirical demonstration of 3x predictive power for future risk-return forecasting. Experimental results on a real-world dataset highlight the superiority of AlphaSharpe metrics, making them highly relevant for portfolio managers and financial decision-makers. This framework not only addresses the limitations of existing metrics but also showcases the potential of LLMs in advancing financial analytics, paving the way for informed and robust investment strategies. 

**Abstract (ZH)**: 金融指标如夏普比率在评估投资绩效方面至关重要，它们通过平衡风险和回报来发挥作用。然而，传统指标在稳健性和泛化性方面往往存在不足，尤其是在动态和波动的市场条件下。本文引入了一种名为AlphaSharpe的新框架，该框架利用大规模语言模型（LLMs）迭代地进化和优化金融指标。AlphaSharpe通过迭代交叉、变异和评估生成了超越传统方法的增强型风险-回报指标，这些指标在稳健性和与未来表现指标的相关性方面表现出更出色的表现。本文的主要贡献包括：（1）将LLMs用于生成和改进受特定领域知识启发的金融指标的创新方法，（2）一套评分机制确保进化出的指标能够有效泛化到未见过的数据，以及（3）实验证明了3倍的预测力量，用于未来风险-回报预测。在真实数据集上的实验结果强调了AlphaSharpe指标的优越性，使其对组合经理和金融决策者具有高度相关性。该框架不仅解决了现有指标的局限性，还展示了LLMs在推进金融分析方面的能力，为制定明智且稳健的投资策略铺平了道路。 

---
# Leveraging Large Language Models to Enhance Machine Learning Interpretability and Predictive Performance: A Case Study on Emergency Department Returns for Mental Health Patients 

**Title (ZH)**: 利用大型语言模型提升机器学习可解释性和预测性能：一项针对精神健康患者急诊复诊的案例研究 

**Authors**: Abdulaziz Ahmed, Mohammad Saleem, Mohammed Alzeen, Badari Birur, Rachel E Fargason, Bradley G Burk, Hannah Rose Harkins, Ahmed Alhassan, Mohammed Ali Al-Garadi  

**Link**: [PDF](https://arxiv.org/pdf/2502.00025)  

**Abstract**: Objective: To evaluate whether integrating large language models (LLMs) with traditional machine learning approaches improves both the predictive accuracy and clinical interpretability of ED mental health returns risk models. Methods: This retrospective cohort study analyzed 42,464 ED visits for 27,904 unique mental health patients at an Academic Medical Center in the deep South of the United States between January 2018 and December 2022. Main Outcomes and Measures: Two primary outcomes were evaluated: (1) 30 days ED return prediction accuracy and (2) model interpretability through a novel retrieval-augmented generation (RAG) framework integrating SHAP (SHapley Additive exPlanations) values with contextual clinical knowledge. Results: The proposed machine learning interpretability framework, leveraging LLM, achieved 99% accuracy in translating complex model predictions into clinically relevant explanations. Integration of LLM-extracted features enhanced predictive performance, improving the XGBoost model area under the curve (AUC) from 0.73 to 0.76. The LLM-based feature extraction using 10-shot learning significantly outperformed traditional approaches, achieving an accuracy of 0.882 and an F1 score of 0.86 for chief complaint classification (compared to conventional methods with an accuracy range of 0.59 to 0.63) and demonstrating accuracy values ranging from 0.65 to 0.93 across multiple SDoH categories, underscoring its robust performance in extracting features from clinical notes. Conclusions and Relevance: Integrating LLMs with traditional machine learning models yielded modest but consistent improvements in ED return prediction accuracy while substantially enhancing model interpretability through automated, clinically relevant explanations. This approach offers a framework for translating complex predictive analytics into actionable clinical insights. 

**Abstract (ZH)**: 目的：评估将大型语言模型（LLMs）与传统机器学习方法结合使用是否能够同时提高急诊科心理健康返回风险模型的预测准确性和临床可解释性。

方法：本回顾性队列研究分析了2018年1月至2022年12月在美国南部一所学术医疗中心的27,904名独特心理健康患者在急诊科的42,464次就诊记录。主要结果和指标：主要评估了两项结果：（1）30天急诊返回预测准确性；（2）通过结合利用LIME（本地可解释的模型解释）值与上下文临床知识的检索增强生成（RAG）框架来评估模型的可解释性。结果：所提出的基于LLMs的机器学习可解释性框架实现了99%的复杂模型预测向临床相关解释的翻译准确性。将LLM提取的特征结合使用提高了预测性能，使XGBoost模型的曲线下面积（AUC）从0.73提高到0.76。基于10-shot学习的LLM特征提取方法在主要症状分类方面的准确率达到0.882，F1分数达到0.86，显著优于传统方法（准确率范围为0.59至0.63），并在多个社会人口经济因素（SDoH）类别中达到了0.65至0.93的准确性，证明了其在从临床记录中提取特征方面的稳健性表现。结论与意义：将LLMs与传统机器学习模型结合使用在急诊返回预测准确性方面产生了适度但一致的提升，同时通过自动化、临床相关的解释大幅增强了模型的可解释性。这种方法提供了一个将复杂的预测分析转化为可操作的临床见解的框架。 

---
# Augmented Knowledge Graph Querying leveraging LLMs 

**Title (ZH)**: 利用大语言模型增强知识图谱查询 

**Authors**: Marco Arazzi, Davide Ligari, Serena Nicolazzo, Antonino Nocera  

**Link**: [PDF](https://arxiv.org/pdf/2502.01298)  

**Abstract**: Adopting Knowledge Graphs (KGs) as a structured, semantic-oriented, data representation model has significantly improved data integration, reasoning, and querying capabilities across different domains. This is especially true in modern scenarios such as Industry 5.0, in which the integration of data produced by humans, smart devices, and production processes plays a crucial role. However, the management, retrieval, and visualization of data from a KG using formal query languages can be difficult for non-expert users due to their technical complexity, thus limiting their usage inside industrial environments. For this reason, we introduce SparqLLM, a framework that utilizes a Retrieval-Augmented Generation (RAG) solution, to enhance the querying of Knowledge Graphs (KGs). SparqLLM executes the Extract, Transform, and Load (ETL) pipeline to construct KGs from raw data. It also features a natural language interface powered by Large Language Models (LLMs) to enable automatic SPARQL query generation. By integrating template-based methods as retrieved-context for the LLM, SparqLLM enhances query reliability and reduces semantic errors, ensuring more accurate and efficient KG interactions. Moreover, to improve usability, the system incorporates a dynamic visualization dashboard that adapts to the structure of the retrieved data, presenting the query results in an intuitive format. Rigorous experimental evaluations demonstrate that SparqLLM achieves high query accuracy, improved robustness, and user-friendly interaction with KGs, establishing it as a scalable solution to access semantic data. 

**Abstract (ZH)**: 将以下论文内容或标题翻译成中文，符合学术规范：

采用知识图谱（KGs）作为结构化且语义导向的数据表示模型，极大地提升了不同领域间的数据集成、推理和查询能力。特别是在第五代工业（Industry 5.0）等现代场景中，人类、智能设备和生产过程所产生的数据的整合起着至关重要的作用。然而，由于使用形式化的查询语言管理、检索和可视化知识图谱中的数据对于非专家用户来说可能会非常复杂，从而限制了其在工业环境中的应用。因此，我们提出了SparqLLM框架，该框架利用检索增强生成（RAG）解决方案来增强对知识图谱的查询能力。SparqLLM执行抽取、转换和加载（ETL）管道，从原始数据中构建知识图谱。此外，该框架还配备了一个由大型语言模型（LLMs）驱动的自然语言界面，能够实现自动SPARQL查询生成。通过将基于模板的方法作为检索上下文整合到LLM中，SparqLLM提高了查询可靠性，减少了语义错误，确保了更准确和高效的知识图谱交互。为了提高易用性，系统还集成了一个动态可视化仪表板，根据检索数据的结构进行调整，以直观的格式展示查询结果。严格的实证研究表明，SparqLLM实现了高度准确的查询、增强的鲁棒性和用户友好的知识图谱交互，使其成为访问语义数据的可扩展解决方案。 

---
# RankFlow: A Multi-Role Collaborative Reranking Workflow Utilizing Large Language Models 

**Title (ZH)**: RankFlow：一种利用大规模语言模型的多角色协作重排工作流 

**Authors**: Can Jin, Hongwu Peng, Anxiang Zhang, Nuo Chen, Jiahui Zhao, Xi Xie, Kuangzheng Li, Shuya Feng, Kai Zhong, Caiwen Ding, Dimitris N. Metaxas  

**Link**: [PDF](https://arxiv.org/pdf/2502.00709)  

**Abstract**: In an Information Retrieval (IR) system, reranking plays a critical role by sorting candidate passages according to their relevance to a specific query. This process demands a nuanced understanding of the variations among passages linked to the query. In this work, we introduce RankFlow, a multi-role reranking workflow that leverages the capabilities of Large Language Models (LLMs) and role specializations to improve reranking performance. RankFlow enlists LLMs to fulfill four distinct roles: the query Rewriter, the pseudo Answerer, the passage Summarizer, and the Reranker. This orchestrated approach enables RankFlow to: (1) accurately interpret queries, (2) draw upon LLMs' extensive pre-existing knowledge, (3) distill passages into concise versions, and (4) assess passages in a comprehensive manner, resulting in notably better reranking results. Our experimental results reveal that RankFlow outperforms existing leading approaches on widely recognized IR benchmarks, such as TREC-DL, BEIR, and NovelEval. Additionally, we investigate the individual contributions of each role in RankFlow. Code is available at this https URL. 

**Abstract (ZH)**: 在信息检索（IR）系统中，重排序通过根据候选段落与特定查询的相关性对其进行排序，在这一过程中扮演着关键角色。这一过程需要对查询相关段落之间细微差异的深刻理解。在这项工作中，我们引入了RankFlow，这是一种利用大型语言模型（LLMs）和角色专业化能力的多角色重排序工作流，旨在提高重排序性能。RankFlow 固定了 LLMs 扮演四个不同的角色：查询重写者、伪答案者、段落摘要者和重排序器。这种协调的方法使得RankFlow能够：（1）准确解释查询，（2）利用LLMs广泛的现有知识，（3）将段落提炼为简洁版本，以及（4）全面评估段落，从而显著提高重排序结果。我们的实验结果表明，RankFlow 在广泛认可的IR基准测试（如TREC-DL、BEIR和NovelEval）中优于现有的领先方法。此外，我们还研究了 RankFlow 中每个角色的独立贡献。源代码可在以下链接获取：[此处替换为实际链接]。 

---
# MODS: Moderating a Mixture of Document Speakers to Summarize Debatable Queries in Document Collections 

**Title (ZH)**: MODS：综合文档演讲者混合以摘要化具有争议的查询文档集 

**Authors**: Nishant Balepur, Alexa Siu, Nedim Lipka, Franck Dernoncourt, Tong Sun, Jordan Boyd-Graber, Puneet Mathur  

**Link**: [PDF](https://arxiv.org/pdf/2502.00322)  

**Abstract**: Query-focused summarization (QFS) gives a summary of documents to answer a query. Past QFS work assumes queries have one answer, ignoring debatable ones (Is law school worth it?). We introduce Debatable QFS (DQFS), a task to create summaries that answer debatable queries via documents with opposing perspectives; summaries must comprehensively cover all sources and balance perspectives, favoring no side. These goals elude LLM QFS systems, which: 1) lack structured content plans, failing to guide LLMs to write balanced summaries, and 2) use the same query to retrieve contexts across documents, failing to cover all perspectives specific to each document's content. To overcome this, we design MODS, a multi-LLM framework mirroring human panel discussions. MODS treats documents as individual Speaker LLMs and has a Moderator LLM that picks speakers to respond to tailored queries for planned topics. Speakers use tailored queries to retrieve relevant contexts from their documents and supply perspectives, which are tracked in a rich outline, yielding a content plan to guide the final summary. Experiments on ConflictingQA with controversial web queries and DebateQFS, our new dataset of debate queries from Debatepedia, show MODS beats SOTA by 38-59% in topic paragraph coverage and balance, based on new citation metrics. Users also find MODS's summaries to be readable and more balanced. 

**Abstract (ZH)**: 面向查询的摘要生成（Query-Focused Summarization, QFS）为文档生成摘要以回答查询。以往的QFS工作假设查询有一个明确的答案，忽略了有争议的问题（例如，“法学院值得去吗？”）。本文引入了有争议的查询摘要生成（Debatable Query-Focused Summarization, DQFS）任务，该任务旨在通过包含不同视角的文档来回答有争议的查询；摘要必须全面涵盖所有来源并平衡各种视角，不偏袒任何一方。这些目标超出了当前语言模型（LLM）QFS系统的实现能力，因为它们：1）缺乏结构化的内容计划，无法引导LLM生成平衡的摘要，2）使用相同的查询从不同文档中检索上下文，无法涵盖每个文档内容特有的所有视角。为克服这一困境，我们设计了MODS（多语言模型框架），这是一个模仿人类面板讨论的框架。MODS 将文档视为独立的发言者语言模型，并有一个主持人语言模型来选择发言人以针对特定文档的主题回答定制的查询。发言者使用定制的查询从其文档中检索相关上下文并提供视角，这些视角被记录在一个丰富的提纲中，从而制定一个内容计划来引导最终的摘要。实验结果显示，在使用具有争议性网络查询的ConflictingQA数据集和我们新构建的DebateQFS数据集（该数据集来自Debatepedia的辩论查询）上，MODS 在主题段落覆盖和平衡方面超越了当前最先进的技术（SOTA）38-59%。用户也发现MODS生成的摘要具有较好的可读性和平衡性。 

---
# LLM-TA: An LLM-Enhanced Thematic Analysis Pipeline for Transcripts from Parents of Children with Congenital Heart Disease 

**Title (ZH)**: LLM-TA：一种基于大型语言模型的主题分析管道，用于先天性心脏病儿童家长的访谈记录分析 

**Authors**: Muhammad Zain Raza, Jiawei Xu, Terence Lim, Lily Boddy, Carlos M. Mery, Andrew Well, Ying Ding  

**Link**: [PDF](https://arxiv.org/pdf/2502.01620)  

**Abstract**: Thematic Analysis (TA) is a fundamental method in healthcare research for analyzing transcript data, but it is resource-intensive and difficult to scale for large, complex datasets. This study investigates the potential of large language models (LLMs) to augment the inductive TA process in high-stakes healthcare settings. Focusing on interview transcripts from parents of children with Anomalous Aortic Origin of a Coronary Artery (AAOCA), a rare congenital heart disease, we propose an LLM-Enhanced Thematic Analysis (LLM-TA) pipeline. Our pipeline integrates an affordable state-of-the-art LLM (GPT-4o mini), LangChain, and prompt engineering with chunking techniques to analyze nine detailed transcripts following the inductive TA framework. We evaluate the LLM-generated themes against human-generated results using thematic similarity metrics, LLM-assisted assessments, and expert reviews. Results demonstrate that our pipeline outperforms existing LLM-assisted TA methods significantly. While the pipeline alone has not yet reached human-level quality in inductive TA, it shows great potential to improve scalability, efficiency, and accuracy while reducing analyst workload when working collaboratively with domain experts. We provide practical recommendations for incorporating LLMs into high-stakes TA workflows and emphasize the importance of close collaboration with domain experts to address challenges related to real-world applicability and dataset complexity. this https URL 

**Abstract (ZH)**: 主题分析（TA）是医疗保健研究中用于分析转录数据的基本方法，但其资源密集型且难以扩展以应对大规模和复杂的数据集。本研究探讨了大型语言模型（LLMs）在高风险医疗保健环境中增强归纳性TA过程的潜力。以患有主动脉起源异常冠状动脉（AAOCA）的儿童的父母访谈转录数据为例，我们提出了一种增强的主题分析（LLM-TA）管道。该管道结合了经济实惠的先进大型语言模型（GPT-4o mini）、LangChain、提示工程技术以及分块技术，按照归纳性TA框架分析了九份详细的转录数据。我们使用主题相似度指标、LLM辅助评估和专家评审来评估LLM生成的主题与人类生成的结果。结果表明，我们的管道在LLM辅助的主题分析方面显著优于现有方法。虽然该管道单独使用时尚未达到人类级别的质量标准，但它展示了显著提高可扩展性、效率和准确性，并减少分析师工作量的潜力，特别是与域专家协作时。我们提供了将LLMs整合到高风险TA工作流中的实用建议，并强调了与域专家紧密合作的重要性，以解决实际应用中的复杂性和现实问题。以下是链接：this https URL 

---
# Scalable Language Models with Posterior Inference of Latent Thought Vectors 

**Title (ZH)**: 具有潜在思想向量后验推断的大规模语言模型 

**Authors**: Deqian Kong, Minglu Zhao, Dehong Xu, Bo Pang, Shu Wang, Edouardo Honig, Zhangzhang Si, Chuan Li, Jianwen Xie, Sirui Xie, Ying Nian Wu  

**Link**: [PDF](https://arxiv.org/pdf/2502.01567)  

**Abstract**: We propose a novel family of language models, Latent-Thought Language Models (LTMs), which incorporate explicit latent thought vectors that follow an explicit prior model in latent space. These latent thought vectors guide the autoregressive generation of ground tokens through a Transformer decoder. Training employs a dual-rate optimization process within the classical variational Bayes framework: fast learning of local variational parameters for the posterior distribution of latent vectors, and slow learning of global decoder parameters. Empirical studies reveal that LTMs possess additional scaling dimensions beyond traditional LLMs, yielding a structured design space. Higher sample efficiency can be achieved by increasing training compute per token, with further gains possible by trading model size for more inference steps. Designed based on these scaling properties, LTMs demonstrate superior sample and parameter efficiency compared to conventional autoregressive models and discrete diffusion models. They significantly outperform these counterparts in validation perplexity and zero-shot language modeling. Additionally, LTMs exhibit emergent few-shot in-context reasoning capabilities that scale with model and latent size, and achieve competitive performance in conditional and unconditional text generation. 

**Abstract (ZH)**: 我们提出了一种新的语言模型家族，即潜在思维语言模型（LTM，Latent-Thought Models），这些模型在隐空间中嵌入了遵循显式先验模型的隐式潜在思维向量。这些潜在思维向量通过 Transformer 解码器指导自回归生成地面词。训练过程在经典变分贝叶斯框架内采用双速率优化过程：快速学习潜在向量后验分布的局部变分参数，以及慢速学习全局解码器参数。实证研究表明，LTM 拥有超越传统大语言模型（LLM）的额外缩放维度，形成了一个结构化的设计空间。通过增加每词的训练计算量可以提高样本效率，进一步通过牺牲模型大小而增加推理步骤可实现更多的增益。基于这些缩放特性，LTM 在样本效率和参数效率方面表现出优于传统自回归模型和离散扩散模型的性能。它们在验证困惑度和零样本语言建模方面显著优于这些对照组模型。此外，LTM 展现出与模型和潜在向量大小相关的新兴少量样本上下文推理能力，并在条件和无条件文本生成任务中取得了竞争力表现。 

---
# Bias Beware: The Impact of Cognitive Biases on LLM-Driven Product Recommendations 

**Title (ZH)**: 偏见警醒：认知偏见对由大规模语言模型驱动的产品推荐的影响 

**Authors**: Giorgos Filandrianos, Angeliki Dimitriou, Maria Lymperaiou, Konstantinos Thomas, Giorgos Stamou  

**Link**: [PDF](https://arxiv.org/pdf/2502.01349)  

**Abstract**: The advent of Large Language Models (LLMs) has revolutionized product recommendation systems, yet their susceptibility to adversarial manipulation poses critical challenges, particularly in real-world commercial applications. Our approach is the first one to tap into human psychological principles, seamlessly modifying product descriptions, making these adversarial manipulations hard to detect. In this work, we investigate cognitive biases as black-box adversarial strategies, drawing parallels between their effects on LLMs and human purchasing behavior. Through extensive experiments on LLMs of varying scales, we reveal significant vulnerabilities in their use as recommenders, providing critical insights into safeguarding these systems. 

**Abstract (ZH)**: 大型语言模型（LLMs）的出现已经革新了产品推荐系统，然而它们对 adversarial 操纵的敏感性迫切地提出了新的挑战，特别是对于现实世界的商业应用。我们的方法首次利用了人类的心理原则，无缝地修改产品描述，使得这些 adversarial 操纵更难以被检测。在这项工作中，我们探讨了认知偏差作为黑盒 adversarial 战略的应用，将它们对 LLMs 和人类购买行为的影响进行了类比。通过在不同规模的 LLMs 上进行广泛的实验，我们揭示了这些系统作为推荐器时存在的显著漏洞，提供了保障这些系统的重要见解。 

---
# Language Models Prefer What They Know: Relative Confidence Estimation via Confidence Preferences 

**Title (ZH)**: 语言模型偏好它们所熟知的内容：通过信心偏好进行相对信心估计 

**Authors**: Vaishnavi Shrivastava, Ananya Kumar, Percy Liang  

**Link**: [PDF](https://arxiv.org/pdf/2502.01126)  

**Abstract**: Language models (LMs) should provide reliable confidence estimates to help users detect mistakes in their outputs and defer to human experts when necessary. Asking a language model to assess its confidence ("Score your confidence from 0-1.") is a natural way of evaluating its uncertainty. However, models struggle to provide absolute assessments of confidence (i.e. judging confidence in answering a question independent of other questions) and the coarse-grained scores they produce are not useful for evaluating the correctness of their answers. We propose relative confidence estimation, where we match up questions against each other and ask the model to make relative judgments of confidence ("Which question are you more confident in answering correctly?"). Treating each question as a "player" in a series of matchups against other questions and the model's preferences as match outcomes, we can use rank aggregation methods like Elo rating and Bradley-Terry to translate the model's confidence preferences into confidence scores. We evaluate relative confidence estimation against absolute confidence estimation and self-consistency confidence methods on five state-of-the-art LMs -- GPT-4, GPT-4o, Gemini 1.5 Pro, Claude 3.5 Sonnet, and Llama 3.1 405B -- across 14 challenging STEM, social science, and commonsense reasoning question answering tasks. Our results demonstrate that relative confidence estimation consistently provides more reliable confidence scores than absolute confidence estimation, with average gains of 3.5% in selective classification AUC over direct absolute confidence estimation methods and 1.7% over self-consistency approaches across all models and datasets. 

**Abstract (ZH)**: 语言模型（LMs）应当提供可靠的置信度估计，以帮助用户检测其输出中的错误，并在必要时将任务委交给人类专家。要求语言模型评估其自身的置信度（“将你的置信度从0到1打分。”）是评估其不确定性的一种自然方法。然而，模型在提供绝对置信度评估（即独立于其他问题判断回答某个问题的置信度）方面存在困难，它们生成的粗粒度得分对于评估其答案的正确性没有实际帮助。我们提出了相对置信度估计的方法，即将问题相互匹配，并要求模型进行相对置信度判断（“你在哪个问题上更自信能够正确回答？”）。将每个问题视为一系列与其他问题匹配中的“参赛者”，并将模型的偏好视为比赛结果，我们可以使用Elo评级和Bradley-Terry等排名聚合法，将模型的置信度偏好转化为置信度分数。我们使用Elo评级和Bradley-Terry方法对GPT-4、GPT-4o、Gemini 1.5 Pro、Claude 3.5 Sonnet和Llama 3.1 405B这五种先进的语言模型在14项具有挑战性的STEM、社会科学和常识推理问答任务中进行了相对置信度估计与绝对置信度估计以及自我一致性置信度方法的评估。我们的结果显示，相对置信度估计始终能提供比绝对置信度估计更可靠的置信度分数，在所有模型和数据集上，相对置信度估计的平均选择性分类AUC改进率为3.5%（相对于直接的绝对置信度评估方法），提高率为1.7%（相对于自我一致性方法）。 

---
# Knowing When to Stop: Dynamic Context Cutoff for Large Language Models 

**Title (ZH)**: 知道何时停止：大型语言模型的动态上下文截断 

**Authors**: Roy Xie, Junlin Wang, Paul Rosu, Chunyuan Deng, Bolun Sun, Zihao Lin, Bhuwan Dhingra  

**Link**: [PDF](https://arxiv.org/pdf/2502.01025)  

**Abstract**: Large language models (LLMs) process entire input contexts indiscriminately, which is inefficient in cases where the information required to answer a query is localized within the context. We present dynamic context cutoff, a human-inspired method enabling LLMs to self-terminate processing upon acquiring sufficient task-relevant information. Through analysis of model internals, we discover that specific attention heads inherently encode "sufficiency signals" - detectable through lightweight classifiers - that predict when critical information has been processed. This reveals a new efficiency paradigm: models' internal understanding naturally dictates processing needs rather than external compression heuristics. Comprehensive experiments across six QA datasets (up to 40K tokens) with three model families (LLaMA/Qwen/Mistral, 1B0-70B) demonstrate 1.33x average token reduction while improving accuracy by 1.3%. Furthermore, our method demonstrates better performance with the same rate of token reduction compared to other context efficiency methods. Additionally, we observe an emergent scaling phenomenon: while smaller models require require probing for sufficiency detection, larger models exhibit intrinsic self-assessment capabilities through prompting. 

**Abstract (ZH)**: 大语言模型（LLMs）不加区别的处理整个输入语境，但在查询所需信息局限于某一部分语境的情况下，这种处理方式是低效的。我们提出了动态上下文截断方法，这是一种受人类启发的技术，使LLMs能够在获取到足够相关的任务信息时自主终止处理过程。通过分析模型内部机制，我们发现特定的注意力头内嵌了“充分性信号”，这些信号可以通过轻量级分类器检测，以预测何时已处理了关键信息。这揭示了一种新的效率范式：模型内部的理解自然地决定了其处理需求，而不是外部压缩启发式方法。在六个问答数据集（最多40K个令牌）上的全面实验中，涉及三种模型家族（LLaMA/Qwen/Mistral，1B到70B参数），显示了平均1.33倍的令牌减少，同时提高了1.3%的准确性。此外，我们的方法在以相同的速率减少令牌方面表现出更好的性能，与其他上下文效率方法相比。另外，我们还观察到一种新的扩展现象：较小的模型需要探查来检测充分性，而较大的模型则通过提示表现出内在的自我评估能力。 

---
# Self-supervised Analogical Learning using Language Models 

**Title (ZH)**: 自我监督类比学习方法：基于语言模型 

**Authors**: Ben Zhou, Sarthak Jain, Yi Zhang, Qiang Ning, Shuai Wang, Yassine Benajiba, Dan Roth  

**Link**: [PDF](https://arxiv.org/pdf/2502.00996)  

**Abstract**: Large language models have been shown to suffer from reasoning inconsistency issues. That is, they fail more in situations unfamiliar to the training data, even though exact or very similar reasoning paths exist in more common cases that they can successfully solve. Such observations motivate us to propose methods that encourage models to understand the high-level and abstract reasoning processes during training instead of only the final answer. This way, models can transfer the exact solution to similar cases, regardless of their relevance to the pre-training data distribution. In this work, we propose SAL, a self-supervised analogical learning framework. SAL mimics the human analogy process and trains models to explicitly transfer high-quality symbolic solutions from cases that they know how to solve to other rare cases in which they tend to fail more. We show that the resulting models after SAL learning outperform base language models on a wide range of reasoning benchmarks, such as StrategyQA, GSM8K, and HotpotQA, by 2% to 20%. At the same time, we show that our model is more generalizable and controllable through analytical studies. 

**Abstract (ZH)**: 大型语言模型已被证明存在推理一致性问题。即，它们在不熟悉的场景中表现较差，尽管在更常见且能够成功解决的场景中存在完全相同或非常相似的推理路径。这些观察结果促使我们提出方法，促使模型在训练过程中理解高层和抽象的推理过程，而不仅仅是最终答案。这样一来，模型可以将确切的解决方案转移到类似案例上，而无需依赖于预训练数据分布的相关性。在此项工作中，我们提出了一种名为SAL（自我监督类比学习框架）。SAL模仿人类的类比过程，并训练模型将已知如何解决的案例中的高质量符号解决方案明确地转移到其他更罕见但更容易失败的案例中。我们展示了经过SAL训练后的模型在诸如StrategyQA、GSM8K和HotpotQA等广泛推理基准测试上表现出2%到20%的优越性能。同时，我们通过分析研究证明了该模型具有更高的泛化能力和可控性。 

---
# PlotGen: Multi-Agent LLM-based Scientific Data Visualization via Multimodal Feedback 

**Title (ZH)**: PlotGen：基于多模态反馈的多智能体大型语言模型科学数据可视化方法 

**Authors**: Kanika Goswami, Puneet Mathur, Ryan Rossi, Franck Dernoncourt  

**Link**: [PDF](https://arxiv.org/pdf/2502.00988)  

**Abstract**: Scientific data visualization is pivotal for transforming raw data into comprehensible visual representations, enabling pattern recognition, forecasting, and the presentation of data-driven insights. However, novice users often face difficulties due to the complexity of selecting appropriate tools and mastering visualization techniques. Large Language Models (LLMs) have recently demonstrated potential in assisting code generation, though they struggle with accuracy and require iterative debugging. In this paper, we propose PlotGen, a novel multi-agent framework aimed at automating the creation of precise scientific visualizations. PlotGen orchestrates multiple LLM-based agents, including a Query Planning Agent that breaks down complex user requests into executable steps, a Code Generation Agent that converts pseudocode into executable Python code, and three retrieval feedback agents - a Numeric Feedback Agent, a Lexical Feedback Agent, and a Visual Feedback Agent - that leverage multimodal LLMs to iteratively refine the data accuracy, textual labels, and visual correctness of generated plots via self-reflection. Extensive experiments show that PlotGen outperforms strong baselines, achieving a 4-6 percent improvement on the MatPlotBench dataset, leading to enhanced user trust in LLM-generated visualizations and improved novice productivity due to a reduction in debugging time needed for plot errors. 

**Abstract (ZH)**: 科学数据可视化对于将原始数据转换为易于理解的视觉表示至关重要，有助于模式识别、预测和数据驱动洞察的呈现。然而，初学者用户往往由于选择合适工具和掌握可视化技术的复杂性而面临困难。大型语言模型（LLMs）最近在辅助代码生成方面显示出潜力，但在准确性上存在挑战，并且需要多次调试。本文提出了一种名为PlotGen的新颖多代理框架，旨在自动化精确科学可视化创建过程。PlotGen协调多个基于LLM的代理，包括一个查询规划代理，它将复杂的用户请求分解为可执行步骤；一个代码生成代理，将伪代码转换为可执行的Python代码；以及三个检索反馈代理——数值反馈代理、词汇反馈代理和视觉反馈代理——它们利用多模态LLM通过自我反省逐步优化生成图表的数据准确性、文本标签和视觉正确性。广泛的实验表明，PlotGen在MatPlotBench数据集上超过了强大的基线模型，实现了4-6%的改进，从而增加了用户对LLM生成可视化结果的信任度，并通过减少绘制错误所需的调试时间提高了初学者的生产力。 

---
# Context-Aware Hierarchical Merging for Long Document Summarization 

**Title (ZH)**: 基于上下文的分层合并方法用于长文档摘要生成 

**Authors**: Litu Ou, Mirella Lapata  

**Link**: [PDF](https://arxiv.org/pdf/2502.00977)  

**Abstract**: Hierarchical Merging is a technique commonly used to summarize very long texts ($>$100K tokens) by breaking down the input into smaller sections, summarizing those sections individually, and then merging or combining those summaries into a final coherent summary. Although it helps address the limitations of large language models (LLMs) with fixed input length constraints, the recursive merging process can amplify LLM hallucinations, increasing the risk of factual inaccuracies. In this paper, we seek to mitigate hallucinations by enriching hierarchical merging with context from the source document. Specifically, we propose different approaches to contextual augmentation ranging from \emph{replacing} intermediate summaries with relevant input context, to \emph{refining} them while using the context as supporting evidence, and \emph{aligning} them implicitly (via citations) to the input. Experimental results on datasets representing legal and narrative domains show that contextual augmentation consistently outperforms zero-shot and hierarchical merging baselines for the Llama 3.1 model family. Our analysis further reveals that refinement methods tend to perform best when paired with extractive summarization for identifying relevant input. 

**Abstract (ZH)**: 层次合并是一种常用的技术，用于总结非常长的文本（超过10万个标记），通过将输入拆分成较小的部分，分别总结这些部分，然后将这些总结合并或综合成一个最终连贯的总结。尽管这种技术有助于解决大型语言模型（LLMs）固定输入长度限制的局限性，但递归合并过程可能会放大LLMs的幻觉现象，增加事实不准确的风险。在本文中，我们通过为层次合并增加来自源文件的内容来减轻幻觉现象。具体而言，我们提出了从用相关输入内容替换中间总结，到使用内容作为支持证据改进它们，以及通过引用隐式对齐它们的不同内容增强方法。实验结果表明，在表示法律和叙述领域的数据集上，内容增强始终优于零样本和层次合并基线模型（如Llama 3.1系列）。我们的进一步分析还表明，当用提取总结法识别相关输入时，改进方法通常表现最佳。 

---
# Wizard of Shopping: Target-Oriented E-commerce Dialogue Generation with Decision Tree Branching 

**Title (ZH)**: 购物Wizard：基于决策树分支的目标导向电子商务对话生成 

**Authors**: Xiangci Li, Zhiyu Chen, Jason Ingyu Choi, Nikhita Vedula, Besnik Fetahu, Oleg Rokhlenko, Shervin Malmasi  

**Link**: [PDF](https://arxiv.org/pdf/2502.00969)  

**Abstract**: The goal of conversational product search (CPS) is to develop an intelligent, chat-based shopping assistant that can directly interact with customers to understand shopping intents, ask clarification questions, and find relevant products. However, training such assistants is hindered mainly due to the lack of reliable and large-scale datasets. Prior human-annotated CPS datasets are extremely small in size and lack integration with real-world product search systems. We propose a novel approach, TRACER, which leverages large language models (LLMs) to generate realistic and natural conversations for different shopping domains. TRACER's novelty lies in grounding the generation to dialogue plans, which are product search trajectories predicted from a decision tree model, that guarantees relevant product discovery in the shortest number of search conditions. We also release the first target-oriented CPS dataset Wizard of Shopping (WoS), containing highly natural and coherent conversations (3.6k) from three shopping domains. Finally, we demonstrate the quality and effectiveness of WoS via human evaluations and downstream tasks. 

**Abstract (ZH)**: 对话式产品搜索（CPS）的目标是开发一种智能化的聊天购物助手，能够直接与客户互动，理解购物意图、提出澄清问题，并找到相关产品。然而，训练这样的助手主要受到缺乏可靠且大规模的语料库的阻碍。之前的人工标注CPS数据集规模非常小，并且缺乏与实际产品搜索系统的整合。我们提出了一种新的方法TRACER，该方法利用大型语言模型（LLMs）生成不同购物领域的现实且自然的对话。TRACER的独特之处在于，将生成过程与对话计划相结合，这些对话计划是由决策树模型预测的产品搜索轨迹，这可以确保在最少的搜索条件下找到相关产品。此外，我们还发布了首个目标导向的CPS数据集Wizard of Shopping（WoS），包含来自三个购物领域的3600多条高度自然且连贯的对话。最后，我们通过人工评估和下游任务展示了WoS的质量和有效性。 

---
# Universal Abstraction: Harnessing Frontier Models to Structure Real-World Data at Scale 

**Title (ZH)**: 通用抽象：利用前沿模型大规模结构化现实世界数据 

**Authors**: Cliff Wong, Sam Preston, Qianchu Liu, Zelalem Gero, Jass Bagga, Sheng Zhang, Shrey Jain, Theodore Zhao, Yu Gu, Yanbo Xu, Sid Kiblawi, Roshanthi Weerasinghe, Rom Leidner, Kristina Young, Brian Piening, Carlo Bifulco, Tristan Naumann, Mu Wei, Hoifung Poon  

**Link**: [PDF](https://arxiv.org/pdf/2502.00943)  

**Abstract**: The vast majority of real-world patient information resides in unstructured clinical text, and the process of medical abstraction seeks to extract and normalize structured information from this unstructured input. However, traditional medical abstraction methods can require significant manual efforts that can include crafting rules or annotating training labels, limiting scalability. In this paper, we propose UniMedAbstractor (UMA), a zero-shot medical abstraction framework leveraging Large Language Models (LLMs) through a modular and customizable prompt template. We refer to our approach as universal abstraction as it can quickly scale to new attributes through its universal prompt template without curating attribute-specific training labels or rules. We evaluate UMA for oncology applications, focusing on fifteen key attributes representing the cancer patient journey, from short-context attributes (e.g., performance status, treatment) to complex long-context attributes requiring longitudinal reasoning (e.g., tumor site, histology, TNM staging). Experiments on real-world data show UMA's strong performance and generalizability. Compared to supervised and heuristic baselines, UMA with GPT-4o achieves on average an absolute 2-point F1/accuracy improvement for both short-context and long-context attribute abstraction. For pathologic T staging, UMA even outperforms the supervised model by 20 points in accuracy. 

**Abstract (ZH)**: 绝大多数临床患者的个人信息存储在未结构化的临床文本中，而医疗抽象过程旨在从这些未结构化的输入中提取和规范化结构化信息。然而，传统的医疗抽象方法往往需要大量的手动工作，包括制定规则或标注训练标签，这限制了其可扩展性。在本文中，我们提出了一种利用大型语言模型（LLMs）的零样本医疗抽象框架——UniMedAbstractor (UMA)，并采用模块化和可定制的提示模板。我们将我们的方法称为通用抽象，因为它可以通过其通用的提示模板快速扩展到新的属性，而无需收集特定属性的训练标签或规则。我们评估了UMA在肿瘤学应用中的表现，重点关注包括癌症患者旅程中的十五个关键属性，从短上下文属性（例如，体能状态、治疗）到需要纵向推理的复杂长上下文属性（例如，肿瘤部位、病理类型、TNM分期）。在真实数据上的实验显示，UMA具有强大的性能和泛化能力。与监督学习和启发式基准相比，UMA与GPT-4o结合使用时，在短上下文和长上下文属性抽象方面分别获得了平均每项2分的F1/准确率提升。对于病理T分期，UMA的准确率甚至比监督模型高出20分。 

---
# The Accuracy, Robustness, and Readability of LLM-Generated Sustainability-Related Word Definitions 

**Title (ZH)**: LLM生成的可持续性相关词汇定义的准确性、稳健性和可读性 

**Authors**: Alice Heiman  

**Link**: [PDF](https://arxiv.org/pdf/2502.00916)  

**Abstract**: A common language with standardized definitions is crucial for effective climate discussions. However, concerns exist about LLMs misrepresenting climate terms. We compared 300 official IPCC glossary definitions with those generated by GPT-4o-mini, Llama3.1 8B, and Mistral 7B, analyzing adherence, robustness, and readability using SBERT sentence embeddings. The LLMs scored an average adherence of $0.57-0.59 \pm 0.15$, and their definitions proved harder to read than the originals. Model-generated definitions vary mainly among words with multiple or ambiguous definitions, showing the potential to highlight terms that need standardization. The results show how LLMs could support environmental discourse while emphasizing the need to align model outputs with established terminology for clarity and consistency. 

**Abstract (ZH)**: 一种标准化的通用语言对于有效的气候讨论至关重要。然而，存在关于大语言模型（LLMs）扭曲气候术语的担忧。我们比较了300个官方IPCC术语表定义与其由GPT-4o-mini、Llama3.1 8B和Mistral 7B生成的定义，并使用SBERT句子嵌入分析了它们的一致性、稳健性和可读性。LLMs的平均一致性得分为0.57-0.59±0.15，其定义的可读性普遍比原版更差。模型生成的定义主要在具有多种或模糊定义的词汇上有所不同，这表明这些术语可能需要标准化。研究结果表明，LLMs可以支持环境讨论，但也强调了将模型输出与已建立的术语进行对齐以确保清晰性和一致性的必要性。 

---
# Weak Supervision Dynamic KL-Weighted Diffusion Models Guided by Large Language Models 

**Title (ZH)**: 弱监督动态KL加权扩散模型，由大型语言模型引导 

**Authors**: Julian Perry, Frank Sanders, Carter Scott  

**Link**: [PDF](https://arxiv.org/pdf/2502.00826)  

**Abstract**: In this paper, we presents a novel method for improving text-to-image generation by combining Large Language Models (LLMs) with diffusion models, a hybrid approach aimed at achieving both higher quality and efficiency in image synthesis from text descriptions. Our approach introduces a new dynamic KL-weighting strategy to optimize the diffusion process, along with incorporating semantic understanding from pre-trained LLMs to guide the generation process. The proposed method significantly improves both the visual quality and alignment of generated images with text descriptions, addressing challenges such as computational inefficiency, instability in training, and robustness to textual variability. We evaluate our method on the COCO dataset and demonstrate its superior performance over traditional GAN-based models, both quantitatively and qualitatively. Extensive experiments, including ablation studies and human evaluations, confirm that our method outperforms existing approaches in terms of image realism, relevance to the input text, and overall aesthetic quality. Our approach also shows promise in scalability to other multimodal tasks, making it a versatile solution for a wide range of generative applications. 

**Abstract (ZH)**: 本文提出了一种结合大型语言模型（LLMs）和扩散模型的创新方法，旨在通过混合方式在文本描述到图像生成中实现更高质量和更高效率。我们提出的方法引入了一种新的动态KL加权策略，以优化扩散过程，并通过引入预训练LLMs的语义理解来引导生成过程。所提出的方法显著提高了生成图像的视觉质量和与文本描述的对齐度，解决了计算效率低下、训练不稳定性以及文本变异性鲁棒性等问题。我们在COCO数据集上评估了该方法，并在定量和定性上证明了其优于传统基于生成对抗网络（GAN）的模型。广泛的实验，包括消融研究和人工评估，证实了该方法在图像的真实感、与输入文本的相关性以及整体审美质量方面优于现有方法。此外，我们的方法在扩展到其他多模态任务方面也显示出潜力，使其成为一个适用于多种生成应用的灵活解决方案。 

---
# Probing Large Language Models in Reasoning and Translating Complex Linguistic Puzzles 

**Title (ZH)**: 探究大型语言模型在推理和翻译复杂语言谜题中的能力 

**Authors**: Zheng-Lin Lin, Yu-Fei Shih, Shu-Kai Hsieh  

**Link**: [PDF](https://arxiv.org/pdf/2502.00817)  

**Abstract**: This paper investigates the utilization of Large Language Models (LLMs) for solving complex linguistic puzzles, a domain requiring advanced reasoning and adept translation capabilities akin to human cognitive processes. We explore specific prompting techniques designed to enhance ability of LLMs to reason and elucidate their decision-making pathways, with a focus on Input-Output Prompting (IO), Chain-of-Thought Prompting (CoT), and Solo Performance Prompting (SPP). Utilizing datasets from the Puzzling Machine Competition and various Linguistics Olympiads, we employ a comprehensive set of metrics to assess the performance of GPT-4 0603, a prominent LLM, across these prompting methods. Our findings illuminate the potential of LLMs in linguistic reasoning and complex translation tasks, highlighting their capabilities and identifying limitations in the context of linguistic puzzles. This research contributes significantly to the broader field of Natural Language Processing (NLP) by providing insights into the optimization of LLM applications for improved reasoning and translation accuracy, thereby enriching the ongoing dialogue in NLP advancements. 

**Abstract (ZH)**: 本文探讨了大型语言模型（LLMs）在解决复杂语言谜题方面的应用，这是一个需要高级推理和熟练翻译能力的领域，类似于人类的认知过程。我们研究了特定的提示技术，以增强LLMs的推理能力，并阐明其决策路径，重点关注输入输出提示（IO）、推理链提示（CoT）和单独表现提示（SPP）。利用Puzzling Machine竞赛和各种语言奥林匹克竞赛的数据集，我们使用一系列综合评估指标来评估GPT-4 0603（一种知名的LLM）在这些提示方法下的表现。我们的研究结果揭示了LLMs在语言推理和复杂翻译任务中的潜力，同时突出了它们的功能并指出了在语言谜题情境下的局限性。本文对自然语言处理（NLP）领域的广泛应用优化提供了深刻见解，从而丰富了NLP技术进展的持续对话。 

---
# Rethinking Mixture-of-Agents: Is Mixing Different Large Language Models Beneficial? 

**Title (ZH)**: 重新思考“混合智能体”：混合不同大规模语言模型是否有利？ 

**Authors**: Wenzhe Li, Yong Lin, Mengzhou Xia, Chi Jin  

**Link**: [PDF](https://arxiv.org/pdf/2502.00674)  

**Abstract**: Ensembling outputs from diverse sources is a straightforward yet effective approach to boost performance. Mixture-of-Agents (MoA) is one such popular ensemble method that aggregates outputs from multiple different Large Language Models (LLMs). This paper raises the question in the context of language models: is mixing different LLMs truly beneficial? We propose Self-MoA -- an ensemble method that aggregates outputs from only the single top-performing LLM. Our extensive experiments reveal that, surprisingly, Self-MoA outperforms standard MoA that mixes different LLMs in a large number of scenarios: Self-MoA achieves $6.6\%$ improvement over MoA on the AlpacaEval 2.0 benchmark, and an average of $3.8\%$ improvement across various benchmarks, including MMLU, CRUX, and MATH. Applying Self-MoA to one of the top-ranking models in AlpacaEval 2.0 directly achieves the new state-of-the-art performance on the leaderboard. To understand the effectiveness of Self-MoA, we systematically investigate the trade-off between diversity and quality of outputs under various MoA settings. We confirm that the MoA performance is rather sensitive to the quality, and mixing different LLMs often lowers the average quality of the models. To complement the study, we identify the scenarios where mixing different LLMs could be helpful. This paper further introduces a sequential version of Self-MoA, that is capable of aggregating a large number of LLM outputs on-the-fly over multiple rounds, and is as effective as aggregating all outputs at once. 

**Abstract (ZH)**: 将多样来源的输出进行集成是一种简单而有效的方法，可以提升性能。Mixture-of-Agents（MoA）就是一个这样的集成方法，它从多个不同的大型语言模型（LLMs）中聚合输出。本文在语言模型的背景下提出了一个问题：混合不同的LLMs真的有益吗？我们提出了Self-MoA——一种仅从单一最优的LLMs生成的输出中聚合的集成方法。大量实验结果显示，Self-MoA在多种场景中表现出色，显著优于混合不同LLMs的标准MoA：Self-MoA在AlpacaEval 2.0基准测试中比MoA提高了6.6%的性能，并在包括MMLU、CRUX和MATH在内的多个基准测试中平均提高了3.8%的性能。直接将Self-MoA应用于AlpacaEval 2.0得分最高的模型，在排行榜上取得了新的SOTA性能。为了理解Self-MoA的有效性，我们系统地研究了在各种MoA设置下输出多样性和质量之间的权衡。我们证实MoA的性能对质量非常敏感，而混合不同的LLMs往往降低了模型的平均质量。为了补充这项研究，我们确定了混合不同LLMs可能有益的场景。本文还进一步引入了Self-MoA的序列版本，它能够在多个轮次中实时聚合大量的LLM输出，并且其效果等同于一次性聚合所有输出。 

---
# M+: Extending MemoryLLM with Scalable Long-Term Memory 

**Title (ZH)**: M+: 扩展内存LLM的可扩展长期记忆 

**Authors**: Yu Wang, Dmitry Krotov, Yuanzhe Hu, Yifan Gao, Wangchunshu Zhou, Julian McAuley, Dan Gutfreund, Rogerio Feris, Zexue He  

**Link**: [PDF](https://arxiv.org/pdf/2502.00592)  

**Abstract**: Equipping large language models (LLMs) with latent-space memory has attracted increasing attention as they can extend the context window of existing language models. However, retaining information from the distant past remains a challenge. For example, MemoryLLM (Wang et al., 2024a), as a representative work with latent-space memory, compresses past information into hidden states across all layers, forming a memory pool of 1B parameters. While effective for sequence lengths up to 16k tokens, it struggles to retain knowledge beyond 20k tokens. In this work, we address this limitation by introducing M+, a memory-augmented model based on MemoryLLM that significantly enhances long-term information retention. M+ integrates a long-term memory mechanism with a co-trained retriever, dynamically retrieving relevant information during text generation. We evaluate M+ on diverse benchmarks, including long-context understanding and knowledge retention tasks. Experimental results show that M+ significantly outperforms MemoryLLM and recent strong baselines, extending knowledge retention from under 20k to over 160k tokens with similar GPU memory overhead. 

**Abstract (ZH)**: 将大语言模型（LLMs）装备以潜在空间记忆已引起越来越多的关注，因为这可以扩展现有语言模型的上下文窗口。然而，保留遥远过去的记忆仍然是一个挑战。例如，MemoryLLM（Wang等，2024a）作为具有潜在空间记忆的代表性工作，将过去的信息压缩到所有层的隐藏状态中，形成1亿参数的记忆池。虽然在最多16k个标记的序列长度下表现有效，但在超过20k个标记时难以保留知识。在本文中，我们通过引入M+，一种基于MemoryLLM的记忆增强模型，来解决这一限制，显著增强了长期信息的保留能力。M+整合了长期记忆机制和协同训练的检索器，在文本生成过程中动态检索相关的信息。我们对M+进行了多种基准测试评估，包括长上下文理解和知识保留任务。实验结果表明，M+显著优于MemoryLLM和最近的强基线，将知识保留能力从不到20k扩展到超过160k个标记，同时保持类似的GPU内存开销。 

---
# HERA: Improving Long Document Summarization using Large Language Models with Context Packaging and Reordering 

**Title (ZH)**: HERA：通过上下文包装和重组提升长文档摘要生成能力的大规模语言模型 

**Authors**: Taiji Li, Hao Chen, Fei Yu, Yin Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2502.00448)  

**Abstract**: Despite the rapid growth of context length of large language models (LLMs) , LLMs still perform poorly in long document summarization. An important reason for this is that relevant information about an event is scattered throughout long documents, and the messy narrative order impairs the accurate understanding and utilization of LLMs for long documents. To address these issues, we propose a novel summary generation framework, called HERA. Specifically, we first segment a long document by its semantic structure and retrieve text segments about the same event, and finally reorder them to form the input context. We evaluate our approach on two long document summarization datasets. The experimental results show that HERA outperforms foundation models in ROUGE, BERTScore and faithfulness metrics, while HERA does not require additional fine-tuning and resources. 

**Abstract (ZH)**: 尽管大规模语言模型（LLMs）的上下文长度迅速增长，但在长文档摘要方面，LLMs 的表现仍然不佳。造成这一现象的一个重要原因在于事件的相关信息在长文档中分散，且杂乱的叙事顺序妨碍了LLMs 对长文档的准确理解和利用。为了解决这些问题，我们提出了一种新的摘要生成框架，称为HERA。具体来说，我们首先通过语义结构对长文档进行分段，并检索关于同一事件的文本片段，最后重新排序以形成输入语境。我们在两个长文档摘要数据集上评估了该方法。实验结果表明，HERA 在ROUGE、BERTScore 和忠實性指标方面优于基础模型，同时HERA 不需要额外的微调和资源。 

---
# Challenges and Innovations in LLM-Powered Fake News Detection: A Synthesis of Approaches and Future Directions 

**Title (ZH)**: 基于LLM的虚假新闻检测挑战与创新：方法综述与未来方向 

**Authors**: Jingyuan Yi, Zeqiu Xu, Tianyi Huang, Peiyang Yu  

**Link**: [PDF](https://arxiv.org/pdf/2502.00339)  

**Abstract**: The pervasiveness of the dissemination of fake news through social media platforms poses critical risks to the trust of the general public, societal stability, and democratic institutions. This challenge calls for novel methodologies in detection, which can keep pace with the dynamic and multi-modal nature of misinformation. Recent works include powering the detection using large language model advances in multimodal frameworks, methodologies using graphs, and adversarial training in the literature of fake news. Based on the different approaches which can bring success, some key highlights will be underlined: enhanced LLM-improves accuracy through more advanced semantics and cross-modality fusion for robust detections. The review further identifies critical gaps in adaptability to dynamic social media trends, real-time, and cross-platform detection capabilities, as well as the ethical challenges thrown up by the misuse of LLMs. Future directions underline the development of style-agnostic models, cross-lingual detection frameworks, and robust policies with a view to mitigating LLM-driven misinformation. This synthesis thus lays a concrete foundation for those researchers and practitioners committed to reinforcing fake news detection systems with complications that keep on growing in the digital landscape. 

**Abstract (ZH)**: 社交媒体平台上传播假新闻的普遍性对公众信任、社会稳定和民主机构构成了关键性风险。面对这一挑战，需要采用新的检测方法，这些方法能够跟上 misinformation 的动态性和多模态特性。近期的研究工作包括利用大语言模型的进展在多模态框架中增强检测能力、基于图形的方法，以及仿真实训方法。基于不同方法的成功应用，以下几点将是关键亮点：增强的大语言模型通过更先进的语义和跨模态融合提高检测准确性。此外，该综述还指出了适应动态社交媒体趋势、实时和跨平台检测能力以及滥用大语言模型引发的伦理挑战的关键漏洞。未来的研究方向将包括开发风格无关模型、跨语言检测框架以及制定严格的政策以应对由大语言模型驱动的错误信息。因此，这一综合分析为致力于强化日益复杂的数字环境中假新闻检测系统的研究人员和实践者奠定了坚实的基础。 

---
# Resolving Editing-Unlearning Conflicts: A Knowledge Codebook Framework for Large Language Model Updating 

**Title (ZH)**: 解决编辑-遗忘冲突：大型语言模型更新的知识代码簿框架 

**Authors**: Binchi Zhang, Zhengzhang Chen, Zaiyi Zheng, Jundong Li, Haifeng Chen  

**Link**: [PDF](https://arxiv.org/pdf/2502.00158)  

**Abstract**: Large Language Models (LLMs) excel in natural language processing by encoding extensive human knowledge, but their utility relies on timely updates as knowledge evolves. Updating LLMs involves two key tasks simultaneously: unlearning to remove unwanted knowledge and editing to incorporate new information. Existing methods face two major challenges: ineffective knowledge storage (either too sparse or too dense) and task conflicts between editing and unlearning, as validated through our theoretical and experimental results. To address these issues, we propose LOKA, a conflict-free framework for LLM updating based on a knowledge codebook. During training, updated knowledge is stored in multiple codebook memories. To optimize knowledge storage, a similarity-aware knowledge mapping ensures that related knowledge pieces are clustered and allocated to the same memory. Additionally, LOKA resolves task conflicts by employing task-specific and multi-task memories guided by a conflict score. In the inference stage, LOKA retrieves the most relevant memory from the codebook and plugs it into the original LLM to apply the updated knowledge. A learning-based router controls codebook activation to further improve knowledge utilization. Extensive experiments demonstrate the effectiveness of LOKA in LLM knowledge updating tasks. 

**Abstract (ZH)**: 大型语言模型（LLMs）在自然语言处理方面表现出色，因为它们能够编码大量的人类知识，但其实用性依赖于及时更新，以反映知识的演变。更新LLMs涉及两个关键任务的同时进行：卸载以删除不需要的知识，以及编辑以纳入新的信息。现有方法面临两大挑战：无效的知识存储（要么过于稀疏，要么过于密集）以及编辑和卸载之间的任务冲突，这一观点通过我们的理论和实验结果得到了验证。为了解决这些问题，我们提出了一种基于知识代码簿的冲突免费框架LOKA，用于LLM的更新。在训练过程中，更新的知识存储在多个代码簿记忆中。为了优化知识存储，一种相似性感知的知识映射确保相关的知识片段被聚类并分配到同一个记忆中。此外，LOKA通过根据冲突分数使用特定任务和多任务记忆来解决任务冲突。在推理阶段，LOKA从代码簿中检索最相关的记忆，并将其插入原始LLM以应用更新的知识。基于学习的路由器控制代码簿的激活，以进一步提高知识的利用效率。大量实验表明，LOKA在LLM知识更新任务中是有效的。 

---
# Efficient Beam Search for Large Language Models Using Trie-Based Decoding 

**Title (ZH)**: 使用基于Trie的解码高效束搜索for大规模语言模型 

**Authors**: Brian J Chan, Jui-Hung Cheng, Mao Xun Huang, Chao-Ting Chen, Hen-Hsen Huang  

**Link**: [PDF](https://arxiv.org/pdf/2502.00085)  

**Abstract**: In Transformer-based sequence-to-sequence generation, beam search has proven effective in enhancing the quality of generated sequences compared to greedy decoding. Conventional beam search methods typically adopt either a sequential or batch-based approach. The sequential approach, while memory-efficient, requires multiple decoding passes to construct a complete search tree, leading to significantly slower inference. On the other hand, the batch-based approach enables parallel computation across beams, but at the expense of high memory consumption due to the need to maintain separate key-value (KV) caches for each beam. In this study, we introduce a novel trie (prefix-tree)-based parallel decoding method that addresses the memory inefficiency of batch-based beam search. By sharing a single KV cache among all beams that share the same prefix, the proposed method not only reduces memory consumption dramatically but also enables parallel decoding across all branches. This innovative use of a prefix tree offers an efficient alternative for beam search, achieving significant memory savings while preserving inference speed, making it particularly well-suited for memory-constrained environments or large-scale model deployments. 

**Abstract (ZH)**: 在基于Transformer的序列到序列生成中，束搜索已被证明相比于贪心解码能有效提升生成序列的质量。传统的束搜索方法通常采用顺序或批量处理的方式。顺序方法虽然内存使用效率较高，但需要多次解码步骤来构建完整的搜索树，从而导致推断速度显著变慢。相比之下，批量处理方法能够并行计算多个束，但会因需要为每个束维护单独的键值（KV）缓存而产生较高的内存消耗。在本研究中，我们提出了一种新颖的基于前缀树（trie）的并行解码方法，以解决批量束搜索的内存使用效率问题。通过在共享相同前缀的所有束之间共享一个KV缓存，所提出的方法不仅显著降低了内存消耗，还使所有分支的并行解码成为可能。这种前缀树的创新使用为束搜索提供了高效的替代方案，能够在保持推断速度的同时实现显著的内存节省，特别适用于内存受限的环境或大规模模型部署。 

---
# BTS: Harmonizing Specialized Experts into a Generalist LLM 

**Title (ZH)**: BTS: 将专业专家整合为通用大语言模型 

**Authors**: Qizhen Zhang, Prajjwal Bhargava, Chloe Bi, Chris X. Cai, Jakob Foerster, Jeremy Fu, Punit Singh Koura, Ruan Silva, Sheng Shen, Emily Dinan, Suchin Gururangan, Mike Lewis  

**Link**: [PDF](https://arxiv.org/pdf/2502.00075)  

**Abstract**: We present Branch-Train-Stitch (BTS), an efficient and flexible training algorithm for combining independently trained large language model (LLM) experts into a single, capable generalist model. Following Li et al., we start with a single seed language model which is branched into domain-specific (e.g., coding or math) experts with continual pretraining. BTS combines experts into a generalist model using lightweight stitch layers, which are inserted between frozen experts and the seed LLM, and trained on a small datamix of the expert domains. Stitch layers enable the seed LLM to integrate representations from any number of experts during the forward pass, allowing it to generalize to new domains, despite remaining frozen. Because BTS does not alter the constituent LLMs, BTS provides a modular and flexible approach: experts can be easily removed and new experts can be added with only a small amount of training. Compared to alternative model merging approaches, BTS yields the best generalist performance on a variety of downstream tasks, retaining the specialized capabilities of each of the experts. 

**Abstract (ZH)**: 我们提出了一种高效的灵活训练算法 Branch-Train-Stitch (BTS)，用于将独立训练的大语言模型（LLM）专家整合成一个单一的通用模型。参照 Li 等人的方法，我们从一个种子语言模型开始，通过持续的预训练将其分支为特定领域的专家（例如编程或数学）。BTS 使用轻量级的缝合层将这些专家整合到一个通用模型中，这些缝合层插入到冻结的专家与种子 LLM 之间，并在专家领域的小数据混合集上进行训练。缝合层允许种子 LLM 在前向传播过程中整合任意数量专家的表示，从而在保持冻结状态的情况下实现跨领域泛化能力。由于 BTS 不会修改构成的 LLM，因此 BTS 提供了一种模块化和灵活的方法：可以轻松移除专家并添加新的专家，只需少量训练即可。与替代的模型合并方法相比，BTS 在各种下游任务中获得了最佳的通用模型性能，同时保留了每个专家的专业能力。 

---
# MALT: Mechanistic Ablation of Lossy Translation in LLMs for a Low-Resource Language: Urdu 

**Title (ZH)**: MALT：为低资源语言乌尔都语消除损失性翻译的机制性剪枝方法 

**Authors**: Taaha Saleem Bajwa  

**Link**: [PDF](https://arxiv.org/pdf/2502.00041)  

**Abstract**: LLMs are predominantly trained on English data, which leads to a significant drop in performance on low-resource languages. Understanding how LLMs handle these languages is crucial for improving their effectiveness. This study focuses on Urdu as a use case for exploring the challenges faced by LLMs in processing low-resource languages. LLMs primarily reason in English when prompted in another language, with the final layers acting as translators to convert the English response into the target language. This study finds that even for low-resource languages, the internal latent response of LLMs in English is quite coherent; however, the translation features are lossy and result in poor translations, leading to reduced performance. By mechanistically removing these translation features and using a separate translation model to translate the internal latent response of LLM, the performance of LLMs improves significantly while also preserving the cultural nuances of the input in low-resource languages. 

**Abstract (ZH)**: 大型语言模型（LLMs）主要使用英语数据进行训练，这导致在低资源语言上的性能出现显著下降。理解LLMs在处理这些语言时的表现对于提高其效果至关重要。本研究以乌尔都语为案例，探讨LLMs在处理低资源语言时所面临的挑战。当LLMs被用另一种语言提示时，主要用英语进行推理，最后几层的作用相当于翻译器，将英语响应转化为目标语言。本研究发现，即使是对于低资源语言，LLMs在英语中的内部潜在响应相当一致；然而，翻译功能具有损失性，导致翻译效果较差，从而使性能下降。通过机制性地去除这些翻译功能，并利用单独的翻译模型将LLMs的内部潜在响应翻译成目标语言，可以大幅提高LLMs的性能，同时在低资源语言中保留输入的文化内涵。 

---
# Almost Surely Safe Alignment of Large Language Models at Inference-Time 

**Title (ZH)**: 在推理时几乎绝对安全的大语言模型对齐 

**Authors**: Xiaotong Ji, Shyam Sundhar Ramesh, Matthieu Zimmer, Ilija Bogunovic, Jun Wang, Haitham Bou Ammar  

**Link**: [PDF](https://arxiv.org/pdf/2502.01208)  

**Abstract**: Even highly capable large language models (LLMs) can produce biased or unsafe responses, and alignment techniques, such as RLHF, aimed at mitigating this issue, are expensive and prone to overfitting as they retrain the LLM. This paper introduces a novel inference-time alignment approach that ensures LLMs generate safe responses almost surely, i.e., with a probability approaching one. We achieve this by framing the safe generation of inference-time responses as a constrained Markov decision process within the LLM's latent space. Crucially, we augment a safety state that tracks the evolution of safety constraints and enables us to demonstrate formal safety guarantees upon solving the MDP in the latent space. Building on this foundation, we propose InferenceGuard, a practical implementation that safely aligns LLMs without modifying the model weights. Empirically, we demonstrate InferenceGuard effectively balances safety and task performance, outperforming existing inference-time alignment methods in generating safe and aligned responses. 

**Abstract (ZH)**: 即使是高度有能力的大语言模型（LLMs），也可能会生成有偏见或不安全的响应。旨在缓解这一问题的对齐技术，如RLHF，成本较高且容易过拟合，因为这些技术需要重新训练LLM。本文介绍了一种新颖的推理时对齐方法，该方法确保LLMs几乎肯定会生成安全的响应，即生成安全响应的概率接近于一。我们通过在LLM的潜在空间中将安全生成推理时响应建模为一个受限的马尔可夫决策过程来实现这一点。关键在于，我们通过跟踪安全约束的演变来增强一个安全状态，并在这种潜在空间中的MDP求解后能够提供形式上的安全保证。基于此，我们提出了InferenceGuard，这是在不影响模型权重的情况下安全对齐LLMs的实用实现。实验证明，InferenceGuard有效地平衡了安全性与任务性能，在生成安全对齐的响应方面优于现有的推理时对齐方法。 

---
# Doing More with Less -- Implementing Routing Strategies in Large Language Model-Based Systems: An Extended Survey 

**Title (ZH)**: 用较少的资源做更多的事情——在基于大型语言模型的系统中实施路由策略：一项扩展综述 

**Authors**: Clovis Varangot-Reille, Christophe Bouvard, Antoine Gourru, Mathieu Ciancone, Marion Schaeffer, François Jacquenet  

**Link**: [PDF](https://arxiv.org/pdf/2502.00409)  

**Abstract**: Large Language Models (LLM)-based systems, i.e. interconnected elements that include an LLM as a central component (e.g., conversational agents), are typically monolithic static architectures that rely on a single LLM for all user queries. However, they often require different preprocessing strategies, levels of reasoning, or knowledge. Generalist LLMs (i.e. GPT-4), trained on very large multi-topic corpora, can perform well in a variety of tasks. However, they require significant financial, energy, and hardware resources that may not be justified for basic tasks. This implies potentially investing in unnecessary costs for a given query. To overcome this problem, a routing mechanism routes user queries to the most suitable components, such as smaller LLMs or experts in specific topics. This approach may improve response quality while minimising costs. Routing can be expanded to other components of the conversational agent architecture, such as the selection of optimal embedding strategies. This paper explores key considerations for integrating routing into LLM-based systems, focusing on resource management, cost definition, and strategy selection. Our main contributions include a formalisation of the problem, a novel taxonomy of existing approaches emphasising relevance and resource efficiency, and a comparative analysis of these strategies in relation to industry practices. Finally, we identify critical challenges and directions for future research. 

**Abstract (ZH)**: 基于大型语言模型（LLM）的系统，例如包含LLM作为核心组件的互联系统（例如对话代理），通常具有传统的静态架构，依赖单一的LLM处理所有用户查询。然而，这些系统往往需要不同的预处理策略、推理层次或知识水平。泛化型LLM（如GPT-4），在非常大的多主题语料库上进行训练，能够在多种任务中表现出色。但是，它们需要大量的资金、能源和硬件资源，这些资源可能不适用于简单的任务。这意味着可能对特定查询投资不必要的成本。为了克服这一问题，路由机制可以根据需要将用户查询导向最合适的组件，比如较小的LLM或特定领域的专家。这可能在提高响应质量的同时降低成本。路由机制还可以扩展到对话代理架构的其他组件，如最佳嵌入策略的选择。本文探讨了将路由机制集成到基于LLM的系统中的关键考虑因素，重点关注资源管理、成本定义和策略选择。我们的主要贡献包括对该问题的正式化表述、一种突出相关性和资源效率的现有方法的新分类，以及这些策略与行业实践的比较分析。最后，我们指出了未来研究的关键挑战和方向。 

---
