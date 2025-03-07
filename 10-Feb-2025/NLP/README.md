# NoLiMa: Long-Context Evaluation Beyond Literal Matching 

**Title (ZH)**: NoLiMa：超越字面匹配的长上下文评估 

**Authors**: Ali Modarressi, Hanieh Deilamsalehy, Franck Dernoncourt, Trung Bui, Ryan A. Rossi, Seunghyun Yoon, Hinrich Schütze  

**Link**: [PDF](https://arxiv.org/pdf/2502.05167)  

**Abstract**: Recent large language models (LLMs) support long contexts ranging from 128K to 1M tokens. A popular method for evaluating these capabilities is the needle-in-a-haystack (NIAH) test, which involves retrieving a "needle" (relevant information) from a "haystack" (long irrelevant context). Extensions of this approach include increasing distractors, fact chaining, and in-context reasoning. However, in these benchmarks, models can exploit existing literal matches between the needle and haystack to simplify the task. To address this, we introduce NoLiMa, a benchmark extending NIAH with a carefully designed needle set, where questions and needles have minimal lexical overlap, requiring models to infer latent associations to locate the needle within the haystack. We evaluate 12 popular LLMs that claim to support contexts of at least 128K tokens. While they perform well in short contexts (<1K), performance degrades significantly as context length increases. At 32K, for instance, 10 models drop below 50% of their strong short-length baselines. Even GPT-4o, one of the top-performing exceptions, experiences a reduction from an almost-perfect baseline of 99.3% to 69.7%. Our analysis suggests these declines stem from the increased difficulty the attention mechanism faces in longer contexts when literal matches are absent, making it harder to retrieve relevant information. 

**Abstract (ZH)**: 近期的大规模语言模型（LLMs）支持长达128K至1M个词元的长文本上下文。一种流行的评估方法是“针叶搜索”（Needle in a Haystack, NIAH）测试，该方法涉及从大量不相关背景信息（“haystack”）中检索“针叶”（相关信息）。该方法的扩展包括增加干扰项、事实链推理和上下文内推断。然而，在这些基准测试中，模型可以通过利用针叶和haystack之间已有的字面匹配简化任务。为了应对这一问题，我们引入了NoLiMa，这是一种扩展了NIAH的新基准，通过精心设计的针叶集，使得问题和针叶之间的词汇重叠最小，迫使模型通过推理隐含关联来定位针叶在haystack中的位置。我们评估了12个声称支持至少128K词元上下文的流行LLM。结果显示，在短文本（<1K）上下文中，这些模型表现良好，但在上下文长度增加时，性能显著下降。例如，在32K的长度下，10个模型的性能下降到强短文本基线值的50%以下。即使是最优秀的模型之一GPT-4o，其基准值几乎接近完美（99.3%），但在32K长度下也下降到69.7%。我们的分析表明，这些下降的根本原因在于在缺少字面匹配的情况下，注意力机制在长文本上下文中面临的更大困难，使得检索相关信息变得更加困难。 

---
# DuoGuard: A Two-Player RL-Driven Framework for Multilingual LLM Guardrails 

**Title (ZH)**: DuoGuard：一个双人强化学习驱动的多语言大语言模型-guardrails框架 

**Authors**: Yihe Deng, Yu Yang, Junkai Zhang, Wei Wang, Bo Li  

**Link**: [PDF](https://arxiv.org/pdf/2502.05163)  

**Abstract**: The rapid advancement of large language models (LLMs) has increased the need for guardrail models to ensure responsible use, particularly in detecting unsafe and illegal content. While substantial safety data exist in English, multilingual guardrail modeling remains underexplored due to the scarcity of open-source safety data in other languages. To address this gap, we propose a novel two-player Reinforcement Learning (RL) framework, where a generator and a guardrail model co-evolve adversarially to produce high-quality synthetic data for multilingual guardrail training. We theoretically formalize this interaction as a two-player game, proving convergence to a Nash equilibrium. Empirical evaluations show that our model \ours outperforms state-of-the-art models, achieving nearly 10% improvement over LlamaGuard3 (8B) on English benchmarks while being 4.5x faster at inference with a significantly smaller model (0.5B). We achieve substantial advancements in multilingual safety tasks, particularly in addressing the imbalance for lower-resource languages in a collected real dataset. Ablation studies emphasize the critical role of synthetic data generation in bridging the imbalance in open-source data between English and other languages. These findings establish a scalable and efficient approach to synthetic data generation, paving the way for improved multilingual guardrail models to enhance LLM safety. Code, model, and data will be open-sourced at this https URL. 

**Abstract (ZH)**: 大型语言模型（LLMs）的迅猛发展增加了对护栏模型的需求，以确保其负责任的使用，特别是在检测不安全和非法内容方面。虽然英语中存在大量的安全数据，但由于其他语言开源安全数据稀缺，多语言护栏建模仍然有待探索。为了解决这一问题，我们提出了一种新颖的双玩家强化学习（RL）框架，该框架中生成器和护栏模型相互对立进化，以生成高质量的合成数据用于多语言护栏训练。我们从理论上将这种交互形式化为一个双玩家博弈，并证明其收敛到纳什均衡。实证评估表明，我们的模型优于最先进的模型，相较LlamaGuard3 (8B)在英语基准测试中表现出了近10%的提升，同时推理速度提高4.5倍，且模型规模显著减小（0.5B）。我们在多语言安全任务上取得了重大进展，特别是在针对收集的真实数据集中低资源语言的不平衡问题方面。消融研究强调了在英文与其他语言之间开源数据不平衡中合成数据生成的关键作用。这些发现确立了一种可扩展且高效的合成数据生成方法，为提高LLM的安全性开辟了改进多语言护栏模型的道路。代码、模型和数据将在以下链接开源：[此处插入链接]。 

---
# Transforming Science with Large Language Models: A Survey on AI-assisted Scientific Discovery, Experimentation, Content Generation, and Evaluation 

**Title (ZH)**: 借助大型语言模型 transformative 科学进展：AI 辅助科学研究综述，包括科学发现、实验、内容生成与评估 

**Authors**: Steffen Eger, Yong Cao, Jennifer D'Souza, Andreas Geiger, Christian Greisinger, Stephanie Gross, Yufang Hou, Brigitte Krenn, Anne Lauscher, Yizhi Li, Chenghua Lin, Nafise Sadat Moosavi, Wei Zhao, Tristan Miller  

**Link**: [PDF](https://arxiv.org/pdf/2502.05151)  

**Abstract**: With the advent of large multimodal language models, science is now at a threshold of an AI-based technological transformation. Recently, a plethora of new AI models and tools has been proposed, promising to empower researchers and academics worldwide to conduct their research more effectively and efficiently. This includes all aspects of the research cycle, especially (1) searching for relevant literature; (2) generating research ideas and conducting experimentation; generating (3) text-based and (4) multimodal content (e.g., scientific figures and diagrams); and (5) AI-based automatic peer review. In this survey, we provide an in-depth overview over these exciting recent developments, which promise to fundamentally alter the scientific research process for good. Our survey covers the five aspects outlined above, indicating relevant datasets, methods and results (including evaluation) as well as limitations and scope for future research. Ethical concerns regarding shortcomings of these tools and potential for misuse (fake science, plagiarism, harms to research integrity) take a particularly prominent place in our discussion. We hope that our survey will not only become a reference guide for newcomers to the field but also a catalyst for new AI-based initiatives in the area of "AI4Science". 

**Abstract (ZH)**: 随着大型多模态语言模型的出现，科学正处于基于AI的技术变革门槛之上。最近，提出了一系列新的AI模型和工具，这些工具有望赋能全球的研究人员和学术界人士更有效地开展研究。这涵盖了研究周期的所有方面，尤其是（1）搜索相关文献；（2）生成研究思路并进行实验；生成（3）文本型和（4）多模态内容（例如，科学图表和图形）；以及（5）基于AI的自动同行评审。在本文中，我们对这些令人兴奋的近期发展进行了深入综述，这些发展有望从根本上改变科学研究过程。本文涵盖了上述五个方面，介绍了相关数据集、方法和结果（包括评估），以及未来研究的局限性和范围。我们讨论中特别突出的是这些工具的伦理问题及其潜在滥用风险（假科学、剽窃、损害研究诚信等）。我们希望本文不仅能成为该领域的入门参考指南，还能成为推动“AI4Science”领域新AI项目的一项催化剂。 

---
# CodeSCM: Causal Analysis for Multi-Modal Code Generation 

**Title (ZH)**: CodeSCM：多模态代码生成的因果分析 

**Authors**: Mukur Gupta, Noopur Bhatt, Suman Jana  

**Link**: [PDF](https://arxiv.org/pdf/2502.05150)  

**Abstract**: In this paper, we propose CodeSCM, a Structural Causal Model (SCM) for analyzing multi-modal code generation using large language models (LLMs). By applying interventions to CodeSCM, we measure the causal effects of different prompt modalities, such as natural language, code, and input-output examples, on the model. CodeSCM introduces latent mediator variables to separate the code and natural language semantics of a multi-modal code generation prompt. Using the principles of Causal Mediation Analysis on these mediators we quantify direct effects representing the model's spurious leanings. We find that, in addition to natural language instructions, input-output examples significantly influence code generation. 

**Abstract (ZH)**: 在本文中，我们提出了一种名为CodeSCM的结构因果模型（Structural Causal Model, SCM），用于利用大规模语言模型（Large Language Models, LLMs）分析多模态代码生成。通过在CodeSCM中应用干预措施，我们可以衡量不同提示模态（如自然语言、代码和输入输出示例）对模型的因果影响。CodeSCM引入了潜在中介变量，用于区分多模态代码生成提示中的代码和自然语言语义。基于因果中介分析的原则，我们定量度量了直接效应，这些效应代表了模型的偏差倾向。我们发现，除了自然语言指令外，输入输出示例也显著影响代码生成。 

---
# GiesKaNe: Bridging Past and Present in Grammatical Theory and Practical Application 

**Title (ZH)**: GiesKaNe：连接语法理论与实践的过去与现在 

**Authors**: Volker Emmrich  

**Link**: [PDF](https://arxiv.org/pdf/2502.05113)  

**Abstract**: This article explores the requirements for corpus compilation within the GiesKaNe project (University of Giessen and Kassel, Syntactic Basic Structures of New High German). The project is defined by three central characteristics: it is a reference corpus, a historical corpus, and a syntactically deeply annotated treebank. As a historical corpus, GiesKaNe aims to establish connections with both historical and contemporary corpora, ensuring its relevance across temporal and linguistic contexts. The compilation process strikes the balance between innovation and adherence to standards, addressing both internal project goals and the broader interests of the research community. The methodological complexity of such a project is managed through a complementary interplay of human expertise and machine-assisted processes. The article discusses foundational topics such as tokenization, normalization, sentence definition, tagging, parsing, and inter-annotator agreement, alongside advanced considerations. These include comparisons between grammatical models, annotation schemas, and established de facto annotation standards as well as the integration of human and machine collaboration. Notably, a novel method for machine-assisted classification of texts along the continuum of conceptual orality and literacy is proposed, offering new perspectives on text selection. Furthermore, the article introduces an approach to deriving de facto standard annotations from existing ones, mediating between standardization and innovation. In the course of describing the workflow the article demonstrates that even ambitious projects like GiesKaNe can be effectively implemented using existing research infrastructure, requiring no specialized annotation tools. Instead, it is shown that the workflow can be based on the strategic use of a simple spreadsheet and integrates the capabilities of the existing infrastructure. 

**Abstract (ZH)**: 本文探讨了在GiesKaNe项目（Giessen大学和Kassel大学，新高地德语的句法基本结构）中语料库编纂的需求。该项目由三个核心特征定义：它是参考语料库、历史语料库，以及深度句法标注的树库。作为历史语料库，GiesKaNe旨在与历史和当代语料库建立联系，确保其在时间与语言背景下的相关性。编纂过程在创新与遵循标准之间寻求平衡，既满足项目内部目标，也服务更广泛的学术界利益。由于该项目的方法论复杂性，通过人类专家与机器辅助过程的互补互动进行管理。本文讨论了基础主题，如分词、规范化、句子定义、标注、解析以及注释者间的一致性，同时还涉及更高级的考虑。这些包括不同的语法模型、标注方案以及既有实际标准的比较，以及将人类和机器协作纳入进来的方法。值得注意的是，提出了一种新的机器辅助分类方法，该方法沿着概念口语化和文字化的连续体对文本进行分类，提供了新的文本选择视角。此外，本文还介绍了一种从现有标注中推导实际标准标注的方法，介于标准化和创新之间。在描述工作流程的过程中，本文展示了即使像GiesKaNe这样的雄心勃勃的项目也可以通过现有的研究基础设施有效地实施，无需专门的标注工具。相反，本文证明了工作流程可以通过战略性地使用简单的电子表格来实现，并整合现有基础设施的功能。 

---
# Flexible and Efficient Grammar-Constrained Decoding 

**Title (ZH)**: 灵活且高效的语法约束解码 

**Authors**: Kanghee Park, Timothy Zhou, Loris D'Antoni  

**Link**: [PDF](https://arxiv.org/pdf/2502.05111)  

**Abstract**: Large Language Models (LLMs) are often asked to generate structured outputs that obey precise syntactic rules, such as code snippets or formatted data. Grammar-constrained decoding (GCD) can guarantee that LLM outputs matches such rules by masking out tokens that will provably lead to outputs that do not belong to a specified context-free grammar (CFG). To guarantee soundness, GCD algorithms have to compute how a given LLM subword tokenizer can align with the tokens used
by a given context-free grammar and compute token masks based on this information. Doing so efficiently is challenging and existing GCD algorithms require tens of minutes to preprocess common grammars. We present a new GCD algorithm together with an implementation that offers 17.71x faster offline preprocessing than existing approaches while preserving state-of-the-art efficiency in online mask computation. 

**Abstract (ZH)**: 大规模语言模型（LLMs）通常被要求生成符合精确句法规则的结构化输出，例如代码片段或格式化的数据。语法约束解码（GCD）可以通过屏蔽那些会导致输出不符合指定上下文自由文法（CFG）的令牌，来确保LLM的输出符合这些规则。为了保证正确性，GCD算法需要计算给定的LLM子词分词器如何与给定的上下文自由文法中的令牌对齐，并基于此信息计算令牌掩码。高效地完成这一任务具有挑战性，现有的GCD算法在预处理常见文法时需要花费十分钟甚至更长的时间。我们提出了一种新的GCD算法及其实现，在保持与现有方法相同的在线掩码计算效率的同时，离线预处理时间比现有方法快17.71倍。 

---
# ChallengeMe: An Adversarial Learning-enabled Text Summarization Framework 

**Title (ZH)**: ChallengeMe：一种基于对抗学习的文本摘要框架 

**Authors**: Xiaoyu Deng, Ye Zhang, Tianmin Guo, Yongzhe Zhang, Zhengjian Kang, Hang Yang  

**Link**: [PDF](https://arxiv.org/pdf/2502.05084)  

**Abstract**: The astonishing performance of large language models (LLMs) and their remarkable achievements in production and daily life have led to their widespread application in collaborative tasks. However, current large models face challenges such as hallucination and lack of specificity in content generation in vertical domain tasks. Inspired by the contrast and classification mechanisms in human cognitive processes, this paper constructs an adversarial learning-based prompt framework named ChallengeMe, which includes three cascaded solutions: generation prompts, evaluation prompts, and feedback optimization. In this process, we designed seven core optimization dimensions and set the threshold for adversarial learning. The results of mixed case studies on the text summarization task show that the proposed framework can generate more accurate and fluent text summaries compared to the current advanced mainstream LLMs. 

**Abstract (ZH)**: 大型语言模型（LLMs）在生产与日常生活中的出色表现及其在协作任务中的广泛应用，引起了人们的广泛关注。然而，当前的大规模模型在垂直领域任务中面临着虚构信息产生和内容生成不具体等挑战。受到人类认知过程中对比和分类机制的启发，本文构建了一个基于对抗学习的提示框架——ChallengeMe，该框架包括三个递进的解决方案：生成提示、评估提示和反馈优化。在此过程中，我们设计了七个核心优化维度，并设定了对抗学习的阈值。在文本摘要任务的混合案例研究中，提出的框架生成的文本摘要比当前最先进的主流LLMs更为准确和流畅。 

---
# nvAgent: Automated Data Visualization from Natural Language via Collaborative Agent Workflow 

**Title (ZH)**: nvAgent: 通过协作代理工作流从自然语言自动进行数据可视化 

**Authors**: Geliang Ouyang, Jingyao Chen, Zhihe Nie, Yi Gui, Yao Wan, Hongyu Zhang, Dongping Chen  

**Link**: [PDF](https://arxiv.org/pdf/2502.05036)  

**Abstract**: Natural Language to Visualization (NL2Vis) seeks to convert natural-language descriptions into visual representations of given tables, empowering users to derive insights from large-scale data. Recent advancements in Large Language Models (LLMs) show promise in automating code generation to transform tabular data into accessible visualizations. However, they often struggle with complex queries that require reasoning across multiple tables. To address this limitation, we propose a collaborative agent workflow, termed nvAgent, for NL2Vis. Specifically, nvAgent comprises three agents: a processor agent for database processing and context filtering, a composer agent for planning visualization generation, and a validator agent for code translation and output verification. Comprehensive evaluations on the new VisEval benchmark demonstrate that nvAgent consistently surpasses state-of-the-art baselines, achieving a 7.88% improvement in single-table and a 9.23% improvement in multi-table scenarios. Qualitative analyses further highlight that nvAgent maintains nearly a 20% performance margin over previous models, underscoring its capacity to produce high-quality visual representations from complex, heterogeneous data sources. 

**Abstract (ZH)**: 自然语言到可视化（NL2Vis）旨在将自然语言描述转换为给定表格的可视化表示，赋予用户从大量数据中提取洞见的能力。近年来，大规模语言模型（LLMs）在自动化代码生成以将表格数据转换为易用的可视化方面展现出前景。然而，在处理需要在多个表格之间进行推理的复杂查询时，它们往往遇到困难。为解决这一限制，我们提出了一种协作代理工作流，称为nvAgent，用于NL2Vis。具体而言，nvAgent 包含三个代理：数据库处理和上下文过滤的处理器代理、规划可视化生成的编曲代理以及代码翻译和输出验证的验证代理。在新的VisEval基准上的全面评估表明，nvAgent 在单表和多表场景中分别实现了7.88%和9.23%的性能提升，超越了现有最先进的基线方法。进一步的定性分析表明，nvAgent 在性能上保持了约20%的优势，突显了其从复杂异构数据源生成高质量可视化表示的能力。 

---
# Aligning Black-box Language Models with Human Judgments 

**Title (ZH)**: 将黑盒语言模型与人类判断对齐 

**Authors**: Gerrit J. J. van den Burg, Gen Suzuki, Wei Liu, Murat Sensoy  

**Link**: [PDF](https://arxiv.org/pdf/2502.04997)  

**Abstract**: Large language models (LLMs) are increasingly used as automated judges to evaluate recommendation systems, search engines, and other subjective tasks, where relying on human evaluators can be costly, time-consuming, and unscalable. LLMs offer an efficient solution for continuous, automated evaluation. However, since the systems that are built and improved with these judgments are ultimately designed for human use, it is crucial that LLM judgments align closely with human evaluators to ensure such systems remain human-centered. On the other hand, aligning LLM judgments with human evaluators is challenging due to individual variability and biases in human judgments. We propose a simple yet effective framework to align LLM judgments with individual human evaluators or their aggregated judgments, without retraining or fine-tuning the LLM. Our approach learns a linear mapping between the LLM's outputs and human judgments, achieving over 142% average improvement in agreement across 29 tasks with only a small number of calibration examples used for training. Notably, our method works in zero-shot and few-shot settings, exceeds inter-human agreement on four out of six tasks, and enables smaller LLMs to achieve performance comparable to that of larger models. 

**Abstract (ZH)**: 大语言模型（LLMs）越来越多地被用作自动裁判，用于评估推荐系统、搜索引擎及其他主观任务。使用人类评估者可能存在成本高、耗时长且难以扩展的问题。LLMs 为持续自动评估提供了一个有效的解决方案。然而，既然这些利用这些评估构建和改进的系统最终是为人使用的，确保LLMs的判断与人类评估者保持一致对于确保系统的人性化设计至关重要。另一方面，使LLMs的判断与人类评估者的判断保持一致具有挑战性，因为人类判断存在个体差异和偏差。我们提出了一种简单且有效的框架，可以在无需重新训练或微调LLMs的情况下对LLMs的判断与个别的人类评估者或其综合判断进行对齐。我们的方法在LLMs的输出与人类判断之间学习了一种线性映射，并仅使用少量校准示例进行训练时，在29个任务上的平均一致改进超过142%。值得注意的是，我们的方法适用于零样本和少量样本的场景，在六个任务中有四个任务上的跨人类一致性超过了我们的方法，并且使较小的LLMs获得了与较大模型相当的性能。 

---
# CoCoA: A Generalized Approach to Uncertainty Quantification by Integrating Confidence and Consistency of LLM Outputs 

**Title (ZH)**: CoCoA：一种通过结合大语言模型输出的置信度和一致性来进行不确定性量化的一般方法 

**Authors**: Roman Vashurin, Maiya Goloburda, Preslav Nakov, Artem Shelmanov, Maxim Panov  

**Link**: [PDF](https://arxiv.org/pdf/2502.04964)  

**Abstract**: Uncertainty quantification (UQ) methods for Large Language Models (LLMs) encompasses a variety of approaches, with two major types being particularly prominent: information-based, which focus on model confidence expressed as token probabilities, and consistency-based, which assess the semantic relationship between multiple outputs generated using repeated sampling. Several recent methods have combined these two approaches and shown impressive performance in various applications. However, they sometimes fail to outperform much simpler baseline methods. Our investigation reveals distinctive characteristics of LLMs as probabilistic models, which help to explain why these UQ methods underperform in certain tasks. Based on these findings, we propose a new way of synthesizing model confidence and output consistency that leads to a family of efficient and robust UQ methods. We evaluate our approach across a variety of tasks such as question answering, abstractive summarization, and machine translation, demonstrating sizable improvements over state-of-the-art UQ approaches. 

**Abstract (ZH)**: 大型语言模型（LLMs）的不确定性量化（UQ）方法涵盖了多种途径，其中两种主要类型尤为突出：信息基于型，重点在于通过令牌概率表达模型的置信度；一致性基于型，则评估使用重复采样生成的多个输出之间的语义关系。近年来，一些方法将这两种类型结合在一起，在多种应用中展示了出色的性能。然而，它们有时未能超越一些更为简单的基准方法。我们的研究揭示了LLMs作为概率模型的特定特征，这有助于解释为什么这些UQ方法在某些任务中表现不佳。基于这些发现，我们提出了一种新的合成模型置信度和输出一致性的方法，从而形成了一种高效且稳健的UQ方法族。我们在问答、抽象总结和机器翻译等多种任务中评估了我们的方法，展示出了相较于最先进的UQ方法的大规模改进。 

---
# Commonality and Individuality! Integrating Humor Commonality with Speaker Individuality for Humor Recognition 

**Title (ZH)**: 共性与个性的融合：将幽默共性与说话者个性整合以进行幽默识别 

**Authors**: Haohao Zhu, Junyu Lu, Zeyuan Zeng, Zewen Bai, Xiaokun Zhang, Liang Yang, Hongfei Lin  

**Link**: [PDF](https://arxiv.org/pdf/2502.04960)  

**Abstract**: Humor recognition aims to identify whether a specific speaker's text is humorous. Current methods for humor recognition mainly suffer from two limitations: (1) they solely focus on one aspect of humor commonalities, ignoring the multifaceted nature of humor; and (2) they typically overlook the critical role of speaker individuality, which is essential for a comprehensive understanding of humor expressions. To bridge these gaps, we introduce the Commonality and Individuality Incorporated Network for Humor Recognition (CIHR), a novel model designed to enhance humor recognition by integrating multifaceted humor commonalities with the distinctive individuality of speakers. The CIHR features a Humor Commonality Analysis module that explores various perspectives of multifaceted humor commonality within user texts, and a Speaker Individuality Extraction module that captures both static and dynamic aspects of a speaker's profile to accurately model their distinctive individuality. Additionally, Static and Dynamic Fusion modules are introduced to effectively incorporate the humor commonality with speaker's individuality in the humor recognition process. Extensive experiments demonstrate the effectiveness of CIHR, underscoring the importance of concurrently addressing both multifaceted humor commonality and distinctive speaker individuality in humor recognition. 

**Abstract (ZH)**: 幽默识别旨在识别特定发言人的文本是否具有幽默感。当前的幽默识别方法主要存在两个局限性：（1）它们仅关注幽默的一种共性特征，忽略了幽默的多元性；（2）它们通常忽视了发言者个性在全面理解幽默表达中的重要作用。为弥补这些不足，我们引入了综合共性和个性的幽默识别网络（CIHR，Commonality and Individuality Incorporated Network for Humor Recognition），这是一种新型模型，旨在通过结合多元幽默共性与发言者的独特个性来增强幽默识别。CIHR 包含一个幽默共性分析模块，该模块在用户文本中探索多视角的幽默多元共性；以及一个发言者个性提取模块，该模块捕捉发言者静态和动态的个人特征，以准确建模其独特个性。此外，还引入了静态和动态融合模块，以有效地将幽默共性与发言者的个性融入到幽默识别过程中。广泛的实验证明了CIHR 的有效性，强调了在幽默识别中同时考虑多元幽默共性及其独特个性的重要性。 

---
# SSMLoRA: Enhancing Low-Rank Adaptation with State Space Model 

**Title (ZH)**: SSMLoRA：基于状态空间模型的低秩适应增强 

**Authors**: Jiayang Yu, Yihang Zhang, Bin Wang, Peiqin Lin, Yongkang Liu, Shi Feng  

**Link**: [PDF](https://arxiv.org/pdf/2502.04958)  

**Abstract**: Fine-tuning is a key approach for adapting language models to specific downstream tasks, but updating all model parameters becomes impractical as model sizes increase. Parameter-Efficient Fine-Tuning (PEFT) methods, such as Low-Rank Adaptation (LoRA), address this challenge by introducing additional adaptation parameters into pre-trained weight matrices. However, LoRA's performance varies across different insertion points within the model, highlighting potential parameter inefficiency due to unnecessary insertions. To this end, we propose SSMLoRA (State Space Model Low-Rank Adaptation), an extension of LoRA that incorporates a State Space Model (SSM) to interconnect low-rank matrices. SSMLoRA ensures that performance is maintained even with sparser insertions. SSMLoRA allows the model to not only map inputs to a low-rank space for better feature extraction but also leverage the computations from the previous low-rank space. Our method achieves comparable performance to LoRA on the General Language Understanding Evaluation (GLUE) benchmark while using only half the parameters. Additionally, due to its structure, SSMLoRA shows promise in handling tasks with longer input sequences. .You can find our code here:this https URL. 

**Abstract (ZH)**: 参数微调是使语言模型适应特定下游任务的关键方法，但随着模型规模的增大，更新所有模型参数变得不切实际。参数高效微调（Parameter-Efficient Fine-Tuning, PEFT）方法，如低秩适应（Low-Rank Adaptation, LoRA），通过向预训练权重矩阵引入额外的适应参数来应对这一挑战。然而，LoRA 在模型中不同插入点上的性能差异表明，由于不必要的插入，可能存在参数效率低下的问题。为此，我们提出了一种名为 SSMLoRA（状态空间模型低秩适应）的方法，这是一种 LoRA 的扩展，它将状态空间模型（State Space Model, SSM）纳入低秩矩阵，以实现互联。SSMLoRA 确保即使在更稀疏的插入下也能保持性能。SSMLoRA 允许模型不仅将输入映射到低秩空间以更好地提取特征，而且还利用前一层低秩空间的计算。我们的方法在通用语言理解评估（General Language Understanding Evaluation, GLUE）基准测试上实现了与 LoRA 相当的性能，但仅使用一半的参数。此外，由于其结构，SSMLoRA 在处理更长输入序列的任务上显示出潜力。您可以在以下链接中找到我们的代码：[在这里插入链接]。 

---
# Claim Extraction for Fact-Checking: Data, Models, and Automated Metrics 

**Title (ZH)**: 事实核查中的声明提取：数据、模型与自动化评估指标 

**Authors**: Herbert Ullrich, Tomáš Mlynář, Jan Drchal  

**Link**: [PDF](https://arxiv.org/pdf/2502.04955)  

**Abstract**: In this paper, we explore the problem of Claim Extraction using one-to-many text generation methods, comparing LLMs, small summarization models finetuned for the task, and a previous NER-centric baseline QACG. As the current publications on Claim Extraction, Fact Extraction, Claim Generation and Check-worthy Claim Detection are quite scattered in their means and terminology, we compile their common objectives, releasing the FEVERFact dataset, with 17K atomic factual claims extracted from 4K contextualised Wikipedia sentences, adapted from the original FEVER. We compile the known objectives into an Evaluation framework of: Atomicity, Fluency, Decontextualization, Faithfulness checked for each generated claim separately, and Focus and Coverage measured against the full set of predicted claims for a single input. For each metric, we implement a scale using a reduction to an already-explored NLP task. We validate our metrics against human grading of generic claims, to see that the model ranking on $F_{fact}$, our hardest metric, did not change and the evaluation framework approximates human grading very closely in terms of $F_1$ and RMSE. 

**Abstract (ZH)**: 在本文中，我们探讨了使用一对多文本生成方法进行论断提取的问题，对比了语言模型（LLMs）、针对该任务微调的中小型总结模型以及先前的以命名实体识别（NER）为中心的基础模型QACG。目前关于论断提取、事实提取、论断生成和值得检查的论断检测的相关研究在方法和术语上分布较为分散，因此我们总结了它们的共同目标，并发布了FEVERFact数据集。该数据集包含从4000个上下文化维基百科句子中提取的17000个原子事实论断，是原始FEVER数据集的改编。我们将已知的目标整理成一个评估框架，其中包括针对每个生成的论断分别检查的原子性、流畅性、去语境化和忠实性，以及与单个输入的全部预测论断对比测量的重点和覆盖范围。对于每个指标，我们都使用一个减少的NLP任务来制定评分标准。我们通过与人类对通用论断的评分进行验证，发现以我们的最困难指标$F_{fact}$为基础的模型排名没有变化，且评估框架在$F_1$和RMSE方面非常接近人类评分。 

---
# Evaluating Standard and Dialectal Frisian ASR: Multilingual Fine-tuning and Language Identification for Improved Low-resource Performance 

**Title (ZH)**: 评估标准弗里希和方言弗里希的ASR：多语言微调和语言识别以提高低资源性能 

**Authors**: Reihaneh Amooie, Wietse de Vries, Yun Hao, Jelske Dijkstra, Matt Coler, Martijn Wieling  

**Link**: [PDF](https://arxiv.org/pdf/2502.04883)  

**Abstract**: Automatic Speech Recognition (ASR) performance for low-resource languages is still far behind that of higher-resource languages such as English, due to a lack of sufficient labeled data. State-of-the-art methods deploy self-supervised transfer learning where a model pre-trained on large amounts of data is fine-tuned using little labeled data in a target low-resource language. In this paper, we present and examine a method for fine-tuning an SSL-based model in order to improve the performance for Frisian and its regional dialects (Clay Frisian, Wood Frisian, and South Frisian). We show that Frisian ASR performance can be improved by using multilingual (Frisian, Dutch, English and German) fine-tuning data and an auxiliary language identification task. In addition, our findings show that performance on dialectal speech suffers substantially, and, importantly, that this effect is moderated by the elicitation approach used to collect the dialectal data. Our findings also particularly suggest that relying solely on standard language data for ASR evaluation may underestimate real-world performance, particularly in languages with substantial dialectal variation. 

**Abstract (ZH)**: 低资源语言（如弗里西语及其方言）的自动语音识别（ASR）性能仍然远远落后于诸如英语等高资源语言，主要原因是缺乏足够的标注数据。最先进的方法是采用半监督的迁移学习，其中，在大量数据上预训练的模型通过少量目标低资源语言的标注数据进行微调。在本文中，我们介绍并探讨了一种方法，以改进弗里西语及其方言（包括克莱弗里西语、木弗里西语和南弗里西语）的ASR性能。研究表明，使用多语言（弗里西语、荷兰语、英语和德语）微调数据以及辅助语言识别任务，可以提升弗里西语的ASR性能。此外，我们的研究还表明，方言语音的性能显著受损，并且，重要的是，这种影响在很大程度上取决于收集方言数据的方法。我们的研究结果还特别表明，在仅依赖标准语言数据评估ASR性能时，可能会低估实际性能，尤其是在方言差异显著的语言中。 

---
# pytopicgram: A library for data extraction and topic modeling from Telegram channels 

**Title (ZH)**: PyTopicGram：一个从Telegram频道中提取数据和主题建模的库 

**Authors**: J. Gómez-Romero, J. Cantón Correa, R. Pérez Mercado, F. Prados Abad, M. Molina-Solana, W. Fajardo  

**Link**: [PDF](https://arxiv.org/pdf/2502.04882)  

**Abstract**: Telegram is a popular platform for public communication, generating large amounts of messages through its channels. pytopicgram is a Python library that helps researchers collect, organize, and analyze these Telegram messages. The library offers key features such as easy message retrieval, detailed channel information, engagement metrics, and topic identification using advanced modeling techniques. By simplifying data extraction and analysis, pytopicgram allows users to understand how content spreads and how audiences interact on Telegram. This paper describes the design, main features, and practical uses of \pytopicgram, showcasing its effectiveness for studying public conversations on Telegram. 

**Abstract (ZH)**: Telegram 是一个流行的公共通信平台，通过其频道生成大量的消息。pytopicgram 是一个 Python 库，帮助研究者收集、组织和分析这些 Telegram 消息。该库提供了多项关键功能，包括简单的消息检索、详细的频道信息、互动度量指标以及使用高级建模技术进行主题识别。通过简化数据提取和分析过程，pytopicgram 允许用户理解内容在 Telegram 上的传播方式以及受众之间的互动情况。本文描述了 pytopicgram 的设计、主要功能及其实际应用，并展示了其在研究 Telegram 上公共对话方面的有效性。 

---
# Enhancing Disinformation Detection with Explainable AI and Named Entity Replacement 

**Title (ZH)**: 使用可解释的人工智能和命名实体替换增强虚假信息检测 

**Authors**: Santiago González-Silot, Andrés Montoro-Montarroso, Eugenio Martínez Cámara, Juan Gómez-Romero  

**Link**: [PDF](https://arxiv.org/pdf/2502.04863)  

**Abstract**: The automatic detection of disinformation presents a significant challenge in the field of natural language processing. This task addresses a multifaceted societal and communication issue, which needs approaches that extend beyond the identification of general linguistic patterns through data-driven algorithms. In this research work, we hypothesise that text classification methods are not able to capture the nuances of disinformation and they often ground their decision in superfluous features. Hence, we apply a post-hoc explainability method (SHAP, SHapley Additive exPlanations) to identify spurious elements with high impact on the classification models. Our findings show that non-informative elements (e.g., URLs and emoticons) should be removed and named entities (e.g., Rwanda) should be pseudo-anonymized before training to avoid models' bias and increase their generalization capabilities. We evaluate this methodology with internal dataset and external dataset before and after applying extended data preprocessing and named entity replacement. The results show that our proposal enhances on average the performance of a disinformation classification method with external test data in 65.78% without a significant decrease of the internal test performance. 

**Abstract (ZH)**: 在自然语言处理领域，自动检测虚假信息面临重大挑战。这项任务涉及多方面的社会和交流问题，需要超出基于数据驱动算法的一般语言模式识别的方法。在此项研究工作中，我们假设文本分类方法难以捕捉虚假信息的细微差别，并且它们常常基于冗余特征做出决策。因此，我们应用了一种事后解释方法（SHAP，SHapley Additive exPlanations）来识别对分类模型有重大影响的虚假元素。我们的研究结果表明，在训练前应去除非信息性元素（例如网址和表情符号），并伪匿名化实体名称（例如卢旺达），以避免模型的偏见并提高其泛化能力。我们使用内部数据集和外部数据集，在应用扩展数据预处理和实体名称替换前后进行了评估。结果显示，与内部测试数据相比，在使用外部测试数据时，我们的方法在平均上提高了虚假信息分类方法的性能65.78%，而内部测试数据的性能几乎没有下降。 

---
# Self-Rationalization in the Wild: A Large Scale Out-of-Distribution Evaluation on NLI-related tasks 

**Title (ZH)**: 野生环境中的自我合理化：大规模异常分布评估在自然语言推理相关任务中的表现 

**Authors**: Jing Yang, Max Glockner, Anderson Rocha, Iryna Gurevych  

**Link**: [PDF](https://arxiv.org/pdf/2502.04797)  

**Abstract**: Free-text explanations are expressive and easy to understand, but many datasets lack annotated explanation data, making it challenging to train models for explainable predictions. To address this, we investigate how to use existing explanation datasets for self-rationalization and evaluate models' out-of-distribution (OOD) performance. We fine-tune T5-Large and OLMo-7B models and assess the impact of fine-tuning data quality, the number of fine-tuning samples, and few-shot selection methods. The models are evaluated on 19 diverse OOD datasets across three tasks: natural language inference (NLI), fact-checking, and hallucination detection in abstractive summarization. For the generated explanation evaluation, we conduct a human study on 13 selected models and study its correlation with the Acceptability score (T5-11B) and three other LLM-based reference-free metrics. Human evaluation shows that the Acceptability score correlates most strongly with human judgments, demonstrating its effectiveness in evaluating free-text explanations. Our findings reveal: 1) few annotated examples effectively adapt models for OOD explanation generation; 2) compared to sample selection strategies, fine-tuning data source has a larger impact on OOD performance; and 3) models with higher label prediction accuracy tend to produce better explanations, as reflected by higher Acceptability scores. 

**Abstract (ZH)**: 自由文本解释表达丰富且易于理解，但许多数据集缺乏标注的解释数据，这使得训练解释性预测模型具有挑战性。为了解决这一问题，我们探讨了如何利用现有的解释数据集进行自我合理化，并评估模型在分布外（OOD）性能。我们对 T5-Large 和 OLMo-7B 模型进行了微调，并评估了微调数据质量、微调样本数量以及少量示例选择方法的影响。模型在涉及自然语言推理（NLI）、事实核查以及摘要生成中的虚构检测三大任务的 19 个不同分布外数据集上进行了评估。在生成的解释评估方面，我们对 13 个选定模型进行了人工研究，并研究了其与可接受性评分（T5-11B）以及三个其他基于大语言模型的参考自由评价指标之间的相关性。人工评估结果表明，可接受性评分与人类判断最为相关，这证明了其在评估自由文本解释方面的有效性。我们的研究发现：1) 少量标注示例能够有效地使模型适应于分布外解释生成；2) 与样本选择策略相比，微调数据源对分布外性能的影响更大；3) 有更高标签预测准确度的模型倾向于生成更好的解释，这体现在更高的可接受性评分上。 

---
# Developmentally-plausible Working Memory Shapes a Critical Period for Language Acquisition 

**Title (ZH)**: 发展合理的工件记忆塑造语言获得的关键期 

**Authors**: Masato Mita, Ryo Yoshida, Yohei Oseki  

**Link**: [PDF](https://arxiv.org/pdf/2502.04795)  

**Abstract**: Large language models exhibit general linguistic abilities but significantly differ from humans in their efficiency of language acquisition. This study proposes a method for integrating the developmental characteristics of working memory during the critical period, a stage when human language acquisition is particularly efficient, into language models. The proposed method introduces a mechanism that initially constrains working memory during the early stages of training and gradually relaxes this constraint in an exponential manner as learning progresses. Targeted syntactic evaluation shows that the proposed method outperforms conventional models without memory constraints or with static memory constraints. These findings not only provide new directions for designing data-efficient language models but also offer indirect evidence supporting the underlying mechanisms of the critical period hypothesis in human language acquisition. 

**Abstract (ZH)**: 大型语言模型具备一般的语言能力，但在语言习得的效率上显著不同于人类。本研究提出了一种方法，将关键期内工作记忆的发展特征集成到语言模型中。该关键期是人类语言习得特别高效的阶段。所提出的方法引入了一种机制，在训练的早期阶段对工作记忆施加约束，并随着学习的进行，以指数方式逐渐放松这一约束。针对目标句法评估表明，所提出的方法优于没有记忆约束或具有静态记忆约束的传统模型。这些发现不仅为设计数据高效的语言模型提供了新的方向，还间接支持了人类语言习得关键期假设中的潜在机制。 

---
# S$^2$-MAD: Breaking the Token Barrier to Enhance Multi-Agent Debate Efficiency 

**Title (ZH)**: S$^2$-MAD：打破token限制以提高多agent辩论效率 

**Authors**: Yuting Zeng, Weizhe Huang, Lei Jiang, Tongxuan Liu, Xitai Jin, Chen Tianying Tiana, Jing Li, Xiaohua Xu  

**Link**: [PDF](https://arxiv.org/pdf/2502.04790)  

**Abstract**: Large language models (LLMs) have demonstrated remarkable capabilities across various natural language processing (NLP) scenarios, but they still face challenges when handling complex arithmetic and logical reasoning tasks. While Chain-Of-Thought (CoT) reasoning, self-consistency (SC) and self-correction strategies have attempted to guide models in sequential, multi-step reasoning, Multi-agent Debate (MAD) has emerged as a viable approach for enhancing the reasoning capabilities of LLMs. By increasing both the number of agents and the frequency of debates, the performance of LLMs improves significantly. However, this strategy results in a significant increase in token costs, presenting a barrier to scalability. To address this challenge, we introduce a novel sparsification strategy designed to reduce token costs within MAD. This approach minimizes ineffective exchanges of information and unproductive discussions among agents, thereby enhancing the overall efficiency of the debate process. We conduct comparative experiments on multiple datasets across various models, demonstrating that our approach significantly reduces the token costs in MAD to a considerable extent. Specifically, compared to MAD, our approach achieves an impressive reduction of up to 94.5\% in token costs while maintaining performance degradation below 2.0\%. 

**Abstract (ZH)**: 大规模语言模型（LLMs）在各种自然语言处理（NLP）任务中展现了惊人的能力，但在处理复杂的算术和逻辑推理任务时仍面临挑战。虽然链式思维（Chain-of-Thought, CoT）推理、自一致性（Self-Consistency, SC）和自我修正策略试图在逐步推理过程中引导模型，多智能体辩论（Multi-agent Debate, MAD）已经作为一种有效的方法被提出，以增强LLMs的推理能力。通过增加智能体的数量和辩论的频率，LLMs的性能显著提升。然而，这种策略导致了显著的token成本增加，成为可扩展性的一个障碍。为了解决这一挑战，我们提出了一种新的稀疏化策略，旨在在MAD中降低token成本。这种方法最大限度地减少了信息交换和智能体之间无成效讨论，从而提高了辩论过程的整体效率。我们在多个数据集上对多种模型进行了比较实验，证明了我们的方法在很大程度上降低了MAD中的token成本。具体而言，与MAD相比，我们的方法在保持性能下降低于2.0%的同时，实现了高达94.5%的token成本显著减少。 

---
# Probing Internal Representations of Multi-Word Verbs in Large Language Models 

**Title (ZH)**: 探究大规模语言模型中多词动词的内部表示 

**Authors**: Hassane Kissane, Achim Schilling, Patrick Krauss  

**Link**: [PDF](https://arxiv.org/pdf/2502.04789)  

**Abstract**: This study investigates the internal representations of verb-particle combinations, called multi-word verbs, within transformer-based large language models (LLMs), specifically examining how these models capture lexical and syntactic properties at different neural network layers. Using the BERT architecture, we analyze the representations of its layers for two different verb-particle constructions: phrasal verbs like 'give up' and prepositional verbs like 'look at'. Our methodology includes training probing classifiers on the internal representations to classify these categories at both word and sentence levels. The results indicate that the model's middle layers achieve the highest classification accuracies. To further analyze the nature of these distinctions, we conduct a data separability test using the Generalized Discrimination Value (GDV). While GDV results show weak linear separability between the two verb types, probing classifiers still achieve high accuracy, suggesting that representations of these linguistic categories may be non-linearly separable. This aligns with previous research indicating that linguistic distinctions in neural networks are not always encoded in a linearly separable manner. These findings computationally support usage-based claims on the representation of verb-particle constructions and highlight the complex interaction between neural network architectures and linguistic structures. 

**Abstract (ZH)**: 本研究探讨了基于变压器的大型语言模型（LLMs）内部对多词动词（即动词-粒子组合）的表示，特别是研究这些模型在不同神经网络层如何捕捉词汇和句法特性。我们使用BERT架构分析其各层的表示，并对两种不同类型的动词-粒子构型——短语动词如“give up”和介词动词如“look at”——进行分析。我们的方法包括在内部表示上训练探针分类器以在词和句层面对这些类别进行分类。结果显示，模型的中间层在分类准确性方面表现最佳。为了进一步分析这些差异的本质，我们使用广义判别值（Generalized Discrimination Value, GDV）进行了数据可分性测试。虽然GDV结果显示两种动词类型的线性可分性较弱，但探针分类器仍能实现高准确率，这暗示这些语言类别的表示可能是非线性可分的。这一发现与之前的研究一致，即语言区分在神经网络中的编码并不是总是线性可分的。本研究的计算结果支持基于使用的研究论点，指出动词-粒子构型的表示方式是复杂的，并突出了神经网络架构与语言结构之间的复杂相互作用。 

---
# SeDi-Instruct: Enhancing Alignment of Language Models through Self-Directed Instruction Generation 

**Title (ZH)**: SeDi-Instruct：通过自我导向指令生成提高语言模型的对齐性 

**Authors**: Jungwoo Kim, Minsang Kim, Sungjin Lee  

**Link**: [PDF](https://arxiv.org/pdf/2502.04774)  

**Abstract**: The rapid evolution of Large Language Models (LLMs) has enabled the industry to develop various AI-based services. Instruction tuning is considered essential in adapting foundation models for target domains to provide high-quality services to customers. A key challenge in instruction tuning is obtaining high-quality instruction data. Self-Instruct, which automatically generates instruction data using ChatGPT APIs, alleviates the data scarcity problem. To improve the quality of instruction data, Self-Instruct discards many of the instructions generated from ChatGPT, even though it is inefficient in terms of cost owing to many useless API calls. To generate high-quality instruction data at a low cost, we propose a novel data generation framework, Self-Direct Instruction generation (SeDi-Instruct), which employs diversity-based filtering and iterative feedback task generation. Diversity-based filtering maintains model accuracy without excessively discarding low-quality generated instructions by enhancing the diversity of instructions in a batch. This reduces the cost of synthesizing instruction data. The iterative feedback task generation integrates instruction generation and training tasks and utilizes information obtained during the training to create high-quality instruction sets. Our results show that SeDi-Instruct enhances the accuracy of AI models by 5.2%, compared with traditional methods, while reducing data generation costs by 36%. 

**Abstract (ZH)**: 大型语言模型（LLMs）的快速演化使行业能够开发出各种基于AI的服务。指令调优被认为是在将基础模型适应目标领域的同时，为客户提供高质量服务的关键步骤。指令调优的一个主要挑战是如何获取高质量的指令数据。Self-Instruct 自动利用ChatGPT API 生成指令数据，从而缓解了数据稀缺问题。为了提高指令数据的质量，Self-Instruct 虽然从ChatGPT生成的大量指令中舍弃了很多，但这种方法在成本上显得低效，因为进行了许多不必要的API调用。为了在低成本下生成高质量的指令数据，我们提出了一种新的数据生成框架，即基于多样性的指令生成（SeDi-Instruct），该框架结合了基于多样性的过滤和迭代反馈任务生成。基于多样性的过滤方法通过增强批次指令的多样性，减少了产生指令数据的成本，而不过度舍弃低质量的生成指令。迭代反馈任务生成将指令生成任务与训练任务相结合，并利用训练过程中获得的信息来生成高质量的指令集。我们的结果显示，与传统方法相比，SeDi-Instruct 可以将AI模型的准确性提高5.2%，同时将数据生成成本降低36%。 

---
# Concept Navigation and Classification via Open Source Large Language Model Processing 

**Title (ZH)**: 通过开源大规模语言模型处理进行概念导航与分类 

**Authors**: Maël Kubli  

**Link**: [PDF](https://arxiv.org/pdf/2502.04756)  

**Abstract**: This paper presents a novel methodological framework for detecting and classifying latent constructs, including frames, narratives, and topics, from textual data using Open-Source Large Language Models (LLMs). The proposed hybrid approach combines automated summarization with human-in-the-loop validation to enhance the accuracy and interpretability of construct identification. By employing iterative sampling coupled with expert refinement, the framework guarantees methodological robustness and ensures conceptual precision. Applied to diverse data sets, including AI policy debates, newspaper articles on encryption, and the 20 Newsgroups data set, this approach demonstrates its versatility in systematically analyzing complex political discourses, media framing, and topic classification tasks. 

**Abstract (ZH)**: 本文提出了一种新颖的方法框架，该框架利用开源大型语言模型（LLM）从文本数据中检测和分类潜在构念，包括框架、叙事和主题。所提出的一种混合方法将自动摘要与人为干预验证相结合，以提高构念识别的准确性和可解释性。通过迭代采样与专家改进相结合，该框架确保了方法的稳健性并保证了概念的精准性。该方法被应用于各种数据集，包括AI政策辩论、关于加密的报纸文章以及20个新sgroups数据集，展示了其在系统分析复杂政治话语、媒体框架以及主题分类任务方面的能力与灵活性。 

---
# The "negative end" of change in grammar: terminology, concepts and causes 

**Title (ZH)**: 语法变化的“消极一端”：术语、概念及成因 

**Authors**: Karolina Rudnicka  

**Link**: [PDF](https://arxiv.org/pdf/2502.04729)  

**Abstract**: The topic of "negative end" of change is, contrary to the fields of innovation and emergence, largely under-researched. Yet, it has lately started to gain an increasing attention from language scholars worldwide. The main focus of this article is threefold, namely to discuss the i) terminology; ii) concepts and iii) causes associated with the "negative end" of change in grammar. The article starts with an overview of research conducted on the topic. It then moves to situating phenomena referred to as loss, decline or obsolescence among processes of language change, before elaborating on the terminology and concepts behind it. The last part looks at possible causes for constructions to display a (gradual or rapid, but very consistent) decrease in the frequency of use over time, which continues until the construction disappears or there are only residual or fossilised forms left. Keywords: loss, obsolescence, decline, competition, higher 

**Abstract (ZH)**: 负向变化的议题与创新和涌现领域的研究相比，仍然是一个相对未被充分探索的领域。然而，这一领域近年来在全球语言学者中开始获得越来越多的关注。本文的主要焦点在于三个方面：一是讨论负向变化在语法中的术语问题；二是概念问题；三是导致这些现象的原因。文章首先对这一主题的研究进行了综述。接着将作为语言变化过程中的损失、衰退或过时现象的现象进行定位，并详细探讨相关的术语和概念。最后部分则探讨可能导致结构在一段时间内（逐步或快速但非常一致）使用频率下降的原因，直到结构彻底消失或只剩下残存或僵化的形式。关键词：损失、过时、衰退、竞争、高级 

---
# Evaluating Text Style Transfer Evaluation: Are There Any Reliable Metrics? 

**Title (ZH)**: 文本风格转换评估：是否存在可靠的评估指标？ 

**Authors**: Sourabrata Mukherjee, Atul Kr. Ojha, John P. McCrae, Ondrej Dusek  

**Link**: [PDF](https://arxiv.org/pdf/2502.04718)  

**Abstract**: Text Style Transfer (TST) is the task of transforming a text to reflect a particular style while preserving its original content. Evaluating TST outputs is a multidimensional challenge, requiring the assessment of style transfer accuracy, content preservation, and naturalness. Using human evaluation is ideal but costly, same as in other natural language processing (NLP) tasks, however, automatic metrics for TST have not received as much attention as metrics for, e.g., machine translation or summarization. In this paper, we examine both set of existing and novel metrics from broader NLP tasks for TST evaluation, focusing on two popular subtasks-sentiment transfer and detoxification-in a multilingual context comprising English, Hindi, and Bengali. By conducting meta-evaluation through correlation with human judgments, we demonstrate the effectiveness of these metrics when used individually and in ensembles. Additionally, we investigate the potential of Large Language Models (LLMs) as tools for TST evaluation. Our findings highlight that certain advanced NLP metrics and experimental-hybrid-techniques, provide better insights than existing TST metrics for delivering more accurate, consistent, and reproducible TST evaluations. 

**Abstract (ZH)**: 文本风格转换（TST）是指将文本转换为反映特定风格的同时保留其原始内容的任务。评估TST输出是一项多维度的挑战，需要评估风格转换的准确性、内容的保留以及自然度。虽然使用人工评估是最理想的方法，但在其他自然语言处理（NLP）任务中也会产生高昂的成本，然而，针对TST的自动评估指标并没有像机器翻译或摘要等任务那样受到足够的关注。在本文中，我们探讨了来自更广泛NLP任务的现有和新颖的评价指标，重点关注多语言背景下（包括英语、印地语和孟加拉语）的两种流行子任务——情感转换和去污处理。通过元评估的方式，我们将这些指标与人工判断结果进行相关性分析，展示了这些指标在单独使用或组合使用时的有效性。此外，我们还探讨了大型语言模型（LLMs）作为TST评价工具的潜力。研究结果表明，某些先进的NLP指标和实验性混合技术，可以提供比现有TST指标更好的洞察，以实现更准确、一致和可重复的TST评估。 

---
# Enhancing Impression Change Prediction in Speed Dating Simulations Based on Speakers' Personalities 

**Title (ZH)**: 基于讲话者个性特征提高速配模拟中印象变化预测的效果 

**Authors**: Kazuya Matsuo, Yoko Ishii, Atsushi Otsuka, Ryo Ishii, Hiroaki Sugiyama, Masahiro Mizukami, Tsunehiro Arimoto, Narichika Nomoto, Yoshihide Sato, Tetsuya Yamaguchi  

**Link**: [PDF](https://arxiv.org/pdf/2502.04706)  

**Abstract**: This paper focuses on simulating text dialogues in which impressions between speakers improve during speed dating. This simulation involves selecting an utterance from multiple candidates generated by a text generation model that replicates a specific speaker's utterances, aiming to improve the impression of the speaker. Accurately selecting an utterance that improves the impression is crucial for the simulation. We believe that whether an utterance improves a dialogue partner's impression of the speaker may depend on the personalities of both parties. However, recent methods for utterance selection do not consider the impression per utterance or the personalities. To address this, we propose a method that predicts whether an utterance improves a partner's impression of the speaker, considering the personalities. The evaluation results showed that personalities are useful in predicting impression changes per utterance. Furthermore, we conducted a human evaluation of simulated dialogues using our method. The results showed that it could simulate dialogues more favorably received than those selected without considering personalities. 

**Abstract (ZH)**: 本文聚焦于模拟在速配情境下演讲者印象改善的文本对话。该模拟涉及从由文本生成模型生成的特定发音人的多个候选话语中选择一句话，目的是提高该发言人的印象。准确地选择能够提升印象的话语对于模拟至关重要。我们相信，话语是否能够改善对话伙伴对发言人的印象可能取决于双方的性格。然而，现有的话语选择方法并未考虑每句话的印象或双方的性格。为此，我们提出了一种方法，该方法考虑了双方的性格以预测某句话是否能够提升对话伙伴对发言人的印象。评估结果显示，性格特征在预测每句话的印象变化方面是有用的。此外，我们还使用该方法对模拟对话进行了人类评价，结果表明，这种方法模拟出的对话比在未考虑性格特征的情况下选择的话语更受欢迎。 

---
# ARR: Question Answering with Large Language Models via Analyzing, Retrieving, and Reasoning 

**Title (ZH)**: ARR：通过分析、检索和推理的大语言模型问答方法 

**Authors**: Yuwei Yin, Giuseppe Carenini  

**Link**: [PDF](https://arxiv.org/pdf/2502.04689)  

**Abstract**: Large language models (LLMs) achieve remarkable performance on challenging benchmarks that are often structured as multiple-choice question-answering (QA) tasks. Zero-shot Chain-of-Thought (CoT) prompting enhances reasoning in LLMs but provides only vague and generic guidance ("think step by step"). This paper introduces ARR, an intuitive and effective zero-shot prompting method that explicitly incorporates three key steps in QA solving: analyzing the intent of the question, retrieving relevant information, and reasoning step by step. Comprehensive experiments across diverse and challenging QA tasks demonstrate that ARR consistently improves the Baseline (without ARR prompting) and outperforms CoT. Ablation and case studies further validate the positive contributions of each component: analyzing, retrieving, and reasoning. Notably, intent analysis plays a vital role in ARR. Additionally, extensive evaluations across various model sizes, LLM series, and generation settings solidify the effectiveness, robustness, and generalizability of ARR. 

**Abstract (ZH)**: 大型语言模型（LLMs）在诸如多项选择问答（QA）任务等挑战性基准测试中表现出色。零样本思路链（Chain-of-Thought, CoT）提示可以增强LLMs的推理能力，但只提供了模糊和通用的指导（如“逐步思考”）。本文引入ARR（分析与推理）方法，这是一种直观且有效的零样本提示方法，明确地将问答解决中的三个关键步骤纳入其中：分析问题意图、检索相关信息以及逐步推理。通过对多样化和具有挑战性的QA任务进行全面实验，结果显示ARR始终优于基线模型（未使用ARR提示），并超越了CoT方法。进一步的消融实验和案例研究进一步验证了每个组成部分（分析、检索和推理）的积极作用。值得注意的是，意图分析在ARR中起着至关重要的作用。此外，针对不同模型规模、LLM系列以及生成设置进行的大量评估进一步证实了ARR的有效性、鲁棒性和泛化能力。 

---
# M-IFEval: Multilingual Instruction-Following Evaluation 

**Title (ZH)**: M-IFEval：多语言指令遵循评估 

**Authors**: Antoine Dussolle, Andrea Cardeña Díaz, Shota Sato, Peter Devine  

**Link**: [PDF](https://arxiv.org/pdf/2502.04688)  

**Abstract**: Instruction following is a core capability of modern Large language models (LLMs), making evaluating this capability essential to understanding these models. The Instruction Following Evaluation (IFEval) benchmark from the literature does this using objective criteria, offering a measure of LLM performance without subjective AI or human judgement. However, it only includes English instructions, limiting its ability to assess LLMs in other languages.
We propose the Multilingual Instruction Following Evaluation (M-IFEval) benchmark, expanding the evaluation to French, Japanese, and Spanish, with both general and language-specific instructions. Applying this benchmark to 8 state-of-the-art LLMs, we find that benchmark performance across languages and instruction types can vary widely, underscoring the importance of a multilingual benchmark for evaluating LLMs in a diverse cultural context. 

**Abstract (ZH)**: 指令遵循是现代大型语言模型（LLMs）的一项核心能力，因此评估这一能力对于理解这些模型至关重要。文献中的指令遵循评估基准（IFEval）使用客观标准来执行这一评估，提供了一种无需主观人工智能或人类判断的LLM性能衡量方式。然而，该基准仅包含英文指令，限制了其在评估其他语言的LLM方面的能力。

我们提出了多语言指令遵循评估基准（M-IFEval），将评估扩展到法语、日语和西班牙语，包括通用和语言特定的指令。将此基准应用到8个最先进的LLM上，我们发现不同语言和指令类型的基准性能差异很大，强调了在多元文化背景下评估LLM时使用多语言基准的重要性。 

---
# AdParaphrase: Paraphrase Dataset for Analyzing Linguistic Features toward Generating Attractive Ad Texts 

**Title (ZH)**: AdParaphrase：用于生成吸引人广告文本的语料库，以分析语言特征 

**Authors**: Soichiro Murakami, Peinan Zhang, Hidetaka Kamigaito, Hiroya Takamura, Manabu Okumura  

**Link**: [PDF](https://arxiv.org/pdf/2502.04674)  

**Abstract**: Effective linguistic choices that attract potential customers play crucial roles in advertising success. This study aims to explore the linguistic features of ad texts that influence human preferences. Although the creation of attractive ad texts is an active area of research, progress in understanding the specific linguistic features that affect attractiveness is hindered by several obstacles. First, human preferences are complex and influenced by multiple factors, including their content, such as brand names, and their linguistic styles, making analysis challenging. Second, publicly available ad text datasets that include human preferences are lacking, such as ad performance metrics and human feedback, which reflect people's interests. To address these problems, we present AdParaphrase, a paraphrase dataset that contains human preferences for pairs of ad texts that are semantically equivalent but differ in terms of wording and style. This dataset allows for preference analysis that focuses on the differences in linguistic features. Our analysis revealed that ad texts preferred by human judges have higher fluency, longer length, more nouns, and use of bracket symbols. Furthermore, we demonstrate that an ad text-generation model that considers these findings significantly improves the attractiveness of a given text. The dataset is publicly available at: this https URL. 

**Abstract (ZH)**: 有效的语言选择在吸引潜在顾客方面发挥着关键作用，这对广告的成功至关重要。本研究旨在探讨影响人类偏好的广告文本的语言特征。尽管吸引人的广告文本的创作是研究的活跃领域，但理解具体语言特征对吸引力的影响进展受限于几个障碍。首先，人类的偏好是复杂的，受到多种因素的影响，包括内容（如品牌名称）和语言风格，这使得分析变得困难。其次，缺乏包含人类偏好的公开广告文本数据集，如广告性能度量和人类反馈，这些反馈反映了人们的兴趣。为了解决这些问题，我们提出了AdParaphrase，这是一个包含语义等价但词语和风格不同的广告文本配对的人工偏好数据集。这个数据集允许专注于语言特征差异的偏好分析。我们的分析发现，人类评委偏好的广告文本具有更高的流畅性、更长的长度、更多的名词，并且使用了括号符号。此外，我们证明了一个考虑这些发现的广告文本生成模型能够显著提高文本的吸引力。该数据集可从以下网址获取：this https URL。 

---
# Before It's Too Late: A State Space Model for the Early Prediction of Misinformation and Disinformation Engagement 

**Title (ZH)**: 《未为迟：一种状态空间模型用于早期预测误导信息与虚假信息的参与度》 

**Authors**: Lin Tian, Emily Booth, Francesco Bailo, Julian Droogan, Marian-Andrei Rizoiu  

**Link**: [PDF](https://arxiv.org/pdf/2502.04655)  

**Abstract**: In today's digital age, conspiracies and information campaigns can emerge rapidly and erode social and democratic cohesion. While recent deep learning approaches have made progress in modeling engagement through language and propagation models, they struggle with irregularly sampled data and early trajectory assessment. We present IC-Mamba, a novel state space model that forecasts social media engagement by modeling interval-censored data with integrated temporal embeddings. Our model excels at predicting engagement patterns within the crucial first 15-30 minutes of posting (RMSE 0.118-0.143), enabling rapid assessment of content reach. By incorporating interval-censored modeling into the state space framework, IC-Mamba captures fine-grained temporal dynamics of engagement growth, achieving a 4.72% improvement over state-of-the-art across multiple engagement metrics (likes, shares, comments, and emojis). Our experiments demonstrate IC-Mamba's effectiveness in forecasting both post-level dynamics and broader narrative patterns (F1 0.508-0.751 for narrative-level predictions). The model maintains strong predictive performance across extended time horizons, successfully forecasting opinion-level engagement up to 28 days ahead using observation windows of 3-10 days. These capabilities enable earlier identification of potentially problematic content, providing crucial lead time for designing and implementing countermeasures. Code is available at: this https URL. An interactive dashboard demonstrating our results is available at: this https URL. 

**Abstract (ZH)**: 在当今数字化时代，阴谋论和信息campaign可以迅速涌现并削弱社会和民主的凝聚力。尽管最近的深度学习方法在通过语言和传播模型建模参与方面取得了进展，但在处理不规则采样数据和早期轨迹评估方面仍然存在困难。我们提出了IC-Mamba，一种新颖的状态空间模型，该模型通过集成时间嵌入来预测带有区间截尾数据的影响。我们的模型在帖子发布后的关键前15-30分钟内预测参与模式方面表现出色（均方根误差RMSE 0.118-0.143），从而能够快速评估内容的覆盖面。通过将区间截尾模型整合到状态空间框架中，IC-Mamba捕捉到参与增长的精细时间动态，从而在多个参与指标（点赞、分享、评论和表情）上相对于最先进的方法实现了4.72%的提升。我们的实验展示了IC-Mamba在预测帖子层面动态和更广泛叙述模式方面的有效性（叙述层面预测的F1值为0.508-0.751）。该模型在长期时间范围内保持了强大的预测性能，通过3-10天的观察窗口成功预测了28天后的意见层面参与。这些功能使早期识别可能存在问题的内容成为可能，从而为设计和实施应对措施提供了关键的提前时间。代码可在以下链接获取：this https URL。一个交互式仪表板演示了我们的结果，可在以下链接获取：this https URL。 

---
# Phonetic Reconstruction of the Consonant System of Middle Chinese via Mixed Integer Optimization 

**Title (ZH)**: 通过混合整数优化进行中古音辅音系统的音系重建 

**Authors**: Weiwei Sun, Xiaoxi Luo  

**Link**: [PDF](https://arxiv.org/pdf/2502.04625)  

**Abstract**: This paper is concerned with phonetic reconstruction of the consonant system of Middle Chinese. We propose to cast the problem as a Mixed Integer Programming problem, which is able to automatically explore homophonic information from ancient rhyme dictionaries and phonetic information from modern Chinese dialects, the descendants of Middle Chinese. Numerical evaluation on a wide range of synthetic and real data demonstrates the effectiveness and robustness of the new method. We apply the method to information from Guangyun and 20 modern Chinese dialects to obtain a new phonetic reconstruction result. A linguistically-motivated discussion of this result is also provided. 

**Abstract (ZH)**: 本文关注中古音辅音系统的音系重建问题。我们提出将该问题形式化为混合整数规划问题，从而能够自动探索古韵典籍中的音同信息和现代汉语方言中的音系信息，后者是中古音的后裔。在合成数据和真实数据上的数值评估表明，该新方法的有效性和鲁棒性。我们运用该方法处理《广韵》和20种现代汉语方言的信息，获得了新的音系重建结果。文中还提供了基于语言学动机的对该结果的讨论。 

---
# Extracting and Understanding the Superficial Knowledge in Alignment 

**Title (ZH)**: 提取和理解对齐中的表面知识 

**Authors**: Runjin Chen, Gabriel Jacob Perin, Xuxi Chen, Xilun Chen, Yan Han, Nina S. T. Hirata, Junyuan Hong, Bhavya Kailkhura  

**Link**: [PDF](https://arxiv.org/pdf/2502.04602)  

**Abstract**: Alignment of large language models (LLMs) with human values and preferences, often achieved through fine-tuning based on human feedback, is essential for ensuring safe and responsible AI behaviors. However, the process typically requires substantial data and computation resources. Recent studies have revealed that alignment might be attainable at lower costs through simpler methods, such as in-context learning. This leads to the question: Is alignment predominantly superficial? In this paper, we delve into this question and provide a quantitative analysis. We formalize the concept of superficial knowledge, defining it as knowledge that can be acquired through easily token restyling, without affecting the model's ability to capture underlying causal relationships between tokens. We propose a method to extract and isolate superficial knowledge from aligned models, focusing on the shallow modifications to the final token selection process. By comparing models augmented only with superficial knowledge to fully aligned models, we quantify the superficial portion of alignment. Our findings reveal that while superficial knowledge constitutes a significant portion of alignment, particularly in safety and detoxification tasks, it is not the whole story. Tasks requiring reasoning and contextual understanding still rely on deeper knowledge. Additionally, we demonstrate two practical advantages of isolated superficial knowledge: (1) it can be transferred between models, enabling efficient offsite alignment of larger models using extracted superficial knowledge from smaller models, and (2) it is recoverable, allowing for the restoration of alignment in compromised models without sacrificing performance. 

**Abstract (ZH)**: 将下面的论文内容或标题翻译成中文，要符合学术规范：

对大型语言模型（LLMs）进行与人类价值观和偏好的对齐，常通过基于人类反馈的微调来实现，这对于确保AI行为的安全与责任至关重要。然而，这一过程通常需要大量的数据和计算资源。近期的研究表明，通过更简单的方法，如上下文学习，可能以较低的成本实现对齐。这引发了一个问题：对齐是否主要是表面的？在本文中，我们将深入探讨这一问题，并通过定量分析给出答案。我们将表面知识的概念予以形式化，定义为通过简单的标记重写即可获取的知识，而不影响模型捕捉标记之间潜在因果关系的能力。我们提出了一种方法，用于从对齐模型中提取和隔离表面知识，重点关注最终标记选择过程的浅层修改。通过比较仅增加了表面知识的模型与完全对齐的模型，我们量化了对齐的表面部分。我们的研究发现，虽然表面知识构成了对齐的重要部分，特别是在安全性和去毒任务中，但这并不是全部的故事。需要进行推理和理解上下文的任务仍依赖于深层次的知识。此外，我们展示了孤立表面知识的两个实际优势：（1）它可以跨模型转移，从而利用从小模型提取的表面知识对大模型进行高效离线对齐；（2）它可以恢复，允许在不牺牲性能的情况下恢复受损模型的对齐。 

---
# My LLM might Mimic AAE -- But When Should it? 

**Title (ZH)**: 我的大语言模型可能会模仿社会方言——但在什么情况下应该模仿呢？ 

**Authors**: Sandra C. Sandoval, Christabel Acquaye, Kwesi Cobbina, Mohammad Nayeem Teli, Hal Daumé III  

**Link**: [PDF](https://arxiv.org/pdf/2502.04564)  

**Abstract**: We examine the representation of African American English (AAE) in large language models (LLMs), exploring (a) the perceptions Black Americans have of how effective these technologies are at producing authentic AAE, and (b) in what contexts Black Americans find this desirable. Through both a survey of Black Americans ($n=$ 104) and annotation of LLM-produced AAE by Black Americans ($n=$ 228), we find that Black Americans favor choice and autonomy in determining when AAE is appropriate in LLM output. They tend to prefer that LLMs default to communicating in Mainstream U.S. English in formal settings, with greater interest in AAE production in less formal settings. When LLMs were appropriately prompted and provided in context examples, our participants found their outputs to have a level of AAE authenticity on par with transcripts of Black American speech. Select code and data for our project can be found here: this https URL 

**Abstract (ZH)**: 我们探讨了大型语言模型（LLMs）中非洲美国英语（AAE）的表述，研究了（a）非洲裔美国人如何看待这些技术在产生真实AAE方面的效果，以及（b）在什么语境下非洲裔美国人认为这种表述是有价值的。通过对104名非洲裔美国人的问卷调查以及228名非洲裔美国人对LLM生成的AAE进行标注，我们发现非洲裔美国人倾向于在确定何时在LLM输出中使用AAE时拥有选择权和自主权。他们倾向于在正式场合让LLM默认使用主流美国英语进行交流，在非正式场合则更感兴趣于AAE的生成。当LLM被适当提示并提供相关示例时，我们的参与者发现其输出具有与非洲裔美国人口语录音相当的真实AAE水平。我们项目的部分代码和数据可以在以下链接找到：[这里](this https URL) 

---
# TruthFlow: Truthful LLM Generation via Representation Flow Correction 

**Title (ZH)**: TruthFlow：通过表示流矫正实现真实可信的LLM生成 

**Authors**: Hanyu Wang, Bochuan Cao, Yuanpu Cao, Jinghui Chen  

**Link**: [PDF](https://arxiv.org/pdf/2502.04556)  

**Abstract**: Large language models (LLMs) are known to struggle with consistently generating truthful responses. While various representation intervention techniques have been proposed, these methods typically apply a universal representation correction vector to all input queries, limiting their effectiveness against diverse queries in practice. In this study, we introduce TruthFlow, a novel method that leverages the Flow Matching technique for query-specific truthful representation correction. Specifically, TruthFlow first uses a flow model to learn query-specific correction vectors that transition representations from hallucinated to truthful states. Then, during inference, the trained flow model generates these correction vectors to enhance the truthfulness of LLM outputs. Experimental results demonstrate that TruthFlow significantly improves performance on open-ended generation tasks across various advanced LLMs evaluated on TruthfulQA. Moreover, the trained TruthFlow model exhibits strong transferability, performing effectively on other unseen hallucination benchmarks. 

**Abstract (ZH)**: 大型语言模型（LLMs）普遍难以一致地生成真实准确的回答。尽管已经提出了多种表示干预技术，但这些方法通常会对所有输入查询应用一个通用的表示校正向量，这限制了它们在处理多样化的查询时的有效性。本研究引入了一种名为TruthFlow的新方法，它利用Flow Matching技术针对特定查询进行真实准代表征的校正。具体而言，TruthFlow首先使用流模型学习查询特定的校正向量，以从幻觉状态过渡到真实状态。在推理过程中，训练好的流模型生成这些校正向量以增强LLM输出的真实准确性。实验结果表明，TruthFlow在评估于TruthfulQA上的各种先进LLMs时，显著改善了开放生成任务的表现。此外，经过训练的TruthFlow模型表现出强大的泛化能力，在其他未见过的幻觉基准上也能得到有效应用。 

---
# Contextual Gradient Flow Modeling for Large Language Model Generalization in Multi-Scale Feature Spaces 

**Title (ZH)**: 多尺度特征空间中基于上下文的梯度流建模以提高大型语言模型的泛化能力 

**Authors**: Daphne Quillington, Kingsley Fairbrother, Xavier Tattershall, Irin Kabakum  

**Link**: [PDF](https://arxiv.org/pdf/2502.04548)  

**Abstract**: Optimization methodologies for training large-scale neural architectures often rely on uniform gradient propagation mechanisms that fail to align with hierarchical linguistic structures, limiting their capacity to generalize across diverse language distributions. A structured gradient refinement framework was introduced to incorporate multi-scale contextual adjustments, improving parameter adaptation through dynamic weighting strategies that enhanced representation coherence. Empirical evaluations demonstrated that structured propagation mechanisms contributed to reductions in gradient oscillations, resulting in more stable training dynamics and improved optimization efficiency. The comparative performance assessment indicated that models incorporating hierarchical propagation strategies exhibited greater robustness in long-range dependency retention and cross-domain adaptation. The hierarchical adjustment of weight updates provided an alternative to conventional backpropagation, reducing sensitivity to initialization conditions while improving overall convergence efficiency. The experimental results confirmed that structured gradient propagation influenced representation learning trajectories, aligning parameter updates with broader linguistic dependencies rather than isolated token-level relationships. Statistical evaluations indicated that structured optimization strategies mitigated overfitting while preserving adaptability across heterogeneous text distributions. The findings established that structured gradient propagation provided an empirically validated framework for refining hierarchical representation learning, supporting more effective integration of linguistic dependencies into optimization dynamics. 

**Abstract (ZH)**: 大规模神经架构训练中的优化方法通常依赖于均匀梯度传播机制，这些机制未能与分层语言结构对齐，从而限制了它们在不同语言分布中的泛化能力。为了解决这一问题，引入了一种结构化梯度精炼框架，结合了多尺度上下文调整，通过动态加权策略增强了表示的一致性。实证研究表明，结构化的梯度传播机制有助于降低梯度振荡，从而提高训练动态的稳定性并提升优化效率。性能评估表明，采用分层传播策略的模型在长距离依赖保持和跨域适应方面表现更稳健。分层次权重更新的调整为传统的反向传播提供了一种替代方案，减少了对初始化条件的敏感性，同时提高了整体的收敛效率。实验结果证实，结构化的梯度传播影响了表示学习轨迹，使得权重更新与更广泛的语言依赖性对齐，而非孤立的令牌级关系。统计评估表明，结构化的优化策略在改善泛化能力的同时保持了在异质文本分布中的适应性。研究结果确立了结构化梯度传播提供了一种经验证实的框架，用于精炼分层次表示学习，支持将语言依赖性更有效地整合到优化动态中。 

---
# Multilingual Non-Autoregressive Machine Translation without Knowledge Distillation 

**Title (ZH)**: 无需知识蒸馏的多语言非自回归机器翻译 

**Authors**: Chenyang Huang, Fei Huang, Zaixiang Zheng, Osmar R. Zaïane, Hao Zhou, Lili Mou  

**Link**: [PDF](https://arxiv.org/pdf/2502.04537)  

**Abstract**: Multilingual neural machine translation (MNMT) aims at using one single model for multiple translation directions. Recent work applies non-autoregressive Transformers to improve the efficiency of MNMT, but requires expensive knowledge distillation (KD) processes. To this end, we propose an M-DAT approach to non-autoregressive multilingual machine translation. Our system leverages the recent advance of the directed acyclic Transformer (DAT), which does not require KD. We further propose a pivot back-translation (PivotBT) approach to improve the generalization to unseen translation directions. Experiments show that our M-DAT achieves state-of-the-art performance in non-autoregressive MNMT. 

**Abstract (ZH)**: 多语言神经机器翻译（MNMT）旨在使用一个单一模型进行多种翻译方向的翻译。近期的研究将非自回归Transformer应用于MNMT以提高其效率，但需要昂贵的知识蒸馏（KD）过程。为此，我们提出了一种M-DAT方法以实现非自回归多语言机器翻译。我们的系统利用了最近发展的有向无环Transformer（DAT），该方法不需要KD过程。我们还提出了一种转回翻译（PivotBT）方法以提高其对未见过的翻译方向的泛化能力。实验结果表明，我们的M-DAT在非自回归MNMT中达到了最先进的性能。 

---
# A Decoding Algorithm for Length-Control Summarization Based on Directed Acyclic Transformers 

**Title (ZH)**: 基于有向无环变换器的长度控制摘要解码算法 

**Authors**: Chenyang Huang, Hao Zhou, Cameron Jen, Kangjie Zheng, Osmar R. Zaïane, Lili Mou  

**Link**: [PDF](https://arxiv.org/pdf/2502.04535)  

**Abstract**: Length-control summarization aims to condense long texts into a short one within a certain length limit. Previous approaches often use autoregressive (AR) models and treat the length requirement as a soft constraint, which may not always be satisfied. In this study, we propose a novel length-control decoding algorithm based on the Directed Acyclic Transformer (DAT). Our approach allows for multiple plausible sequence fragments and predicts a \emph{path} to connect them. In addition, we propose a Sequence Maximum a Posteriori (SeqMAP) decoding algorithm that marginalizes different possible paths and finds the most probable summary satisfying the length budget. Our algorithm is based on beam search, which further facilitates a reranker for performance improvement. Experimental results on the Gigaword and DUC2004 datasets demonstrate our state-of-the-art performance for length-control summarization. 

**Abstract (ZH)**: 长度控制总结旨在将长文本压缩成一定长度限制内的短文本。之前的方法通常使用自回归（AR）模型，并将长度要求视为软约束，这可能并不总是能得到满足。在本研究中，我们提出了一种基于有向无环变换器（DAT）的 Novel 长度控制解码算法。该方法允许存在多个合理的序列片段，并预测一条路径来连接这些片段。此外，我们还提出了一种序列最大后验概率（SeqMAP）解码算法，该算法对不同的可能路径进行了归一化，并找到了最符合长度预算的概要。我们的算法基于束搜索，进一步便于部署性能增强器。在 Gigaword 和 DUC2004 数据集上的实验结果表明，我们的方法在长度控制总结方面达到了最先进的性能。 

---
# Group-Adaptive Threshold Optimization for Robust AI-Generated Text Detection 

**Title (ZH)**: 面向群体自适应阈值优化的鲁棒性AI生成文本检测 

**Authors**: Minseok Jung, Cynthia Fuertes Panizo, Liam Dugan, May Fung, Pin-Yu Chen, Paul Pu Liang  

**Link**: [PDF](https://arxiv.org/pdf/2502.04528)  

**Abstract**: The advancement of large language models (LLMs) has made it difficult to differentiate human-written text from AI-generated text. Several AI-text detectors have been developed in response, which typically utilize a fixed global threshold (e.g., {\theta} = 0.5) to classify machine-generated text. However, we find that one universal threshold can fail to account for subgroup-specific distributional variations. For example, when using a fixed threshold, detectors make more false positive errors on shorter human-written text than longer, and more positive classifications on neurotic writing styles than open among long text. These discrepancies can lead to misclassification that disproportionately affects certain groups. We address this critical limitation by introducing FairOPT, an algorithm for group-specific threshold optimization in AI-generated content classifiers. Our approach partitions data into subgroups based on attributes (e.g., text length and writing style) and learns decision thresholds for each group, which enables careful balancing of performance and fairness metrics within each subgroup. In experiments with four AI text classifiers on three datasets, FairOPT enhances overall F1 score and decreases balanced error rate (BER) discrepancy across subgroups. Our framework paves the way for more robust and fair classification criteria in AI-generated output detection. 

**Abstract (ZH)**: 大语言模型（LLMs）的进步使得区分人类撰写的文本和AI生成的文本变得困难。为应对这一挑战，开发了多种AI文本检测器，通常使用固定的整体阈值（例如，θ=0.5）来分类机器生成的文本。然而，我们发现单一的通用阈值无法处理子组特有的分布差异。例如，使用固定阈值时，检测器在较短的文本中会产生更多的误报错误，而在较长的文本中对神经质的写作风格做出更多的正分类，而不是开放型的写作风格。这些差异可能导致分类错误，特别是对某些群体产生不公平影响。我们通过引入FairOPT算法解决了这一关键限制，该算法用于AI生成内容分类器中的分组特定阈值优化。我们的方法根据属性（例如，文本长度和写作风格）将数据划分为不同的子组，并为每个子组学习决策阈值，从而在每个子组内实现性能和公平性指标的谨慎平衡。在针对三个数据集的四个AI文本分类器进行的实验中，FairOPT提高了总体F1分数，并减少了子组间的均衡错误率（BER）差异。我们的框架为更稳健和公平的AI生成输出检测分类标准奠定了基础。 

---
# Linear Correlation in LM's Compositional Generalization and Hallucination 

**Title (ZH)**: LM的组成式泛化与幻觉之间的线性相关性 

**Authors**: Letian Peng, Chenyang An, Shibo Hao, Chengyu Dong, Jingbo Shang  

**Link**: [PDF](https://arxiv.org/pdf/2502.04520)  

**Abstract**: The generalization of language models (LMs) is undergoing active debates, contrasting their potential for general intelligence with their struggles with basic knowledge composition (e.g., reverse/transition curse). This paper uncovers the phenomenon of linear correlations in LMs during knowledge composition. For explanation, there exists a linear transformation between certain related knowledge that maps the next token prediction logits from one prompt to another, e.g., "X lives in the city of" $\rightarrow$ "X lives in the country of" for every given X. This mirrors the linearity in human knowledge composition, such as Paris $\rightarrow$ France. Our findings indicate that the linear transformation is resilient to large-scale fine-tuning, generalizing updated knowledge when aligned with real-world relationships, but causing hallucinations when it deviates. Empirical results suggest that linear correlation can serve as a potential identifier of LM's generalization. Finally, we show such linear correlations can be learned with a single feedforward network and pre-trained vocabulary representations, indicating LM generalization heavily relies on the latter. 

**Abstract (ZH)**: 语言模型（LMs）的一般化 đang chứng kiến các cuộc thảo luận sôi nổi, so sánh khả năng của chúng trong trí tuệ chung với khó khăn trong việc tổ chức kiến thức cơ bản (ví dụ, nghịch lý chuyển tiếp). Bài báo này khám phá hiện tượng hội quan tuyến tính trong quá trình tổ chức kiến thức của LMs. Để giải thích, tồn tại một phép biến đổi tuyến tính giữa một số kiến thức liên quan chuyển logits dự đoán kí tự tiếp theo từ một prompt sang prompt khác, ví dụ: "X sống ở thành phố" $\rightarrow$ "X sống ở quốc gia" với mỗi X given. Điều này phản ánh tính tuyến tính trong quá trình tổ chức kiến thức của con người, như Paris $\rightarrow$ France. Kết quả của chúng tôi cho thấy phép biến đổi tuyến tính cơ bản tồn tại ngay cả sau khi huấn luyện toàn diện quy mô lớn, và duy trì được kiến thức được cập nhật khi phù hợp với mối quan hệ thực tế, nhưng chỉ gây ra ảo tưởng khi deviate. Kết quả thực nghiệm cho thấy hội quan tuyến tính có thể được sử dụng như một nhận dạng tiềm năng cho sự general hóa của LMs. Cuối cùng, chúng tôi chứng minh rằng những hội quan tuyến tính này có thể được học thông qua một mạng dịch chuyển đơn phương và biểu diễn từ vựng được huấn luyện trước, cho thấy sự general hóa của LMs rất phụ thuộc vào biểu diễn từ vựng được huấn luyện trước. 

---
# Beyond Sample-Level Feedback: Using Reference-Level Feedback to Guide Data Synthesis 

**Title (ZH)**: 超越样本级反馈：使用参考级反馈引导数据合成 

**Authors**: Shuhaib Mehri, Xiusi Chen, Heng Ji, Dilek Hakkani-Tür  

**Link**: [PDF](https://arxiv.org/pdf/2502.04511)  

**Abstract**: LLMs demonstrate remarkable capabilities in following natural language instructions, largely due to instruction-tuning on high-quality datasets. While synthetic data generation has emerged as a scalable approach for creating such datasets, maintaining consistent quality standards remains challenging. Recent approaches incorporate feedback to improve data quality, but typically operate at the sample level, generating and applying feedback for each response individually. In this work, we propose Reference-Level Feedback, a novel methodology that instead collects feedback based on high-quality reference samples from carefully curated seed data. We use this feedback to capture rich signals of desirable characteristics that can be propagated to newly synthesized data. We present REFED, a dataset of 10K instruction-response pairs synthesized using such feedback. We demonstrate the effectiveness of our approach by showing that Llama-3.1-8B-Instruct finetuned on REFED achieves state-of-the-art performance among similar-sized SFT-based models on AlpacaEval 2.0 and strong results on Arena-Hard. Through extensive experiments, we show that our approach consistently outperforms traditional sample-level feedback methods with significantly fewer feedback collections and improves performance across different model architectures. 

**Abstract (ZH)**: 大规模语言模型（LLMs）在遵循自然语言指令方面表现出显著的能力，这主要归功于在高质量数据集上的指令调优。尽管合成数据生成已成为创建此类数据集的一个可扩展的方法，但保持一致的质量标准仍然具有挑战性。近期的研究方法引入了反馈机制以提高数据质量，但这些方法通常在样本层面操作，为每一个响应单独生成并应用反馈。在本工作中，我们提出了一种名为参考级反馈的新方法，该方法基于精心挑选的种子数据中的高质量参考样本收集反馈。我们利用这些反馈捕捉丰富的期望特征信号，并将其传播到新合成的数据中。我们基于此反馈生成了一个包含10,000组指令-响应对的数据集REFED。我们通过证明对REFED微调的Llama-3.1-8B-Instruct在AlpacaEval 2.0上的表现达到了同类模型中的最佳水平，并在Arena-Hard上取得出色的成果，来展示我们方法的有效性。通过广泛的实验，我们展示了与传统样本级反馈方法相比，我们的方法不仅显著减少了反馈收集的数量，而且在不同模型架构下提高了性能。 

---
# Heterogeneous Swarms: Jointly Optimizing Model Roles and Weights for Multi-LLM Systems 

**Title (ZH)**: 异质群集：多大型语言模型系统中模型角色和权重的联合优化 

**Authors**: Shangbin Feng, Zifeng Wang, Palash Goyal, Yike Wang, Weijia Shi, Huang Xia, Hamid Palangi, Luke Zettlemoyer, Yulia Tsvetkov, Chen-Yu Lee, Tomas Pfister  

**Link**: [PDF](https://arxiv.org/pdf/2502.04510)  

**Abstract**: We propose Heterogeneous Swarms, an algorithm to design multi-LLM systems by jointly optimizing model roles and weights. We represent multi-LLM systems as directed acyclic graphs (DAGs) of LLMs with topological message passing for collaborative generation. Given a pool of LLM experts and a utility function, Heterogeneous Swarms employs two iterative steps: role-step and weight-step. For role-step, we interpret model roles as learning a DAG that specifies the flow of inputs and outputs between LLMs. Starting from a swarm of random continuous adjacency matrices, we decode them into discrete DAGs, call the LLMs in topological order, evaluate on the utility function (e.g. accuracy on a task), and optimize the adjacency matrices with particle swarm optimization based on the utility score. For weight-step, we assess the contribution of individual LLMs in the multi-LLM systems and optimize model weights with swarm intelligence. We propose JFK-score to quantify the individual contribution of each LLM in the best-found DAG of the role-step, then optimize model weights with particle swarm optimization based on the JFK-score. Experiments demonstrate that Heterogeneous Swarms outperforms 15 role- and/or weight-based baselines by 18.5% on average across 12 tasks. Further analysis reveals that Heterogeneous Swarms discovers multi-LLM systems with heterogeneous model roles and substantial collaborative gains, and benefits from the diversity of language models. 

**Abstract (ZH)**: 我们提出了一种异构群集算法——Heterogeneous Swarms，用于设计多大型语言模型（multi-LLM）系统并通过联合优化模型角色和权重来进行优化。我们将多大型语言模型系统表示为具有拓扑消息传递的大型语言模型有向无环图（DAG），以实现协作生成。给定一组大型语言模型专家和效用函数，Heterogeneous Swarms 采用两个迭代步骤：角色步和权重步。在角色步中，我们将模型角色解释为学习一个DAG，该DAG指定了大型语言模型之间输入和输出的流。从随机连续邻接矩阵的群集开始，将其解码为离散的DAG，并按拓扑顺序调用大型语言模型，评估效用函数（例如，在任务中的准确性），然后基于效用分数使用粒子群优化优化邻接矩阵。在权重步中，我们评估多大型语言模型系统中单个大型语言模型的贡献，并使用群体智能优化模型权重。我们提出了JFK评分来量化角色步步中找到的最佳DAG中每个大型语言模型的个体贡献，然后基于JFK评分使用粒子群优化优化模型权重。实验表明，在12项任务中，Heterogeneous Swarms 平均优于18.5%的基于角色和/或权重的基线方法。进一步的分析表明，Heterogeneous Swarms 发现了具有异构模型角色和显著协作增益的多大型语言模型系统，并且受益于语言模型的多样性。 

---
# When One LLM Drools, Multi-LLM Collaboration Rules 

**Title (ZH)**: 当一个大语言模型表现卓越时，多大语言模型协作取胜 

**Authors**: Shangbin Feng, Wenxuan Ding, Alisa Liu, Zifeng Wang, Weijia Shi, Yike Wang, Zejiang Shen, Xiaochuang Han, Hunter Lang, Chen-Yu Lee, Tomas Pfister, Yejin Choi, Yulia Tsvetkov  

**Link**: [PDF](https://arxiv.org/pdf/2502.04506)  

**Abstract**: This position paper argues that in many realistic (i.e., complex, contextualized, subjective) scenarios, one LLM is not enough to produce a reliable output. We challenge the status quo of relying solely on a single general-purpose LLM and argue for multi-LLM collaboration to better represent the extensive diversity of data, skills, and people. We first posit that a single LLM underrepresents real-world data distributions, heterogeneous skills, and pluralistic populations, and that such representation gaps cannot be trivially patched by further training a single LLM. We then organize existing multi-LLM collaboration methods into a hierarchy, based on the level of access and information exchange, ranging from API-level, text-level, logit-level, to weight-level collaboration. Based on these methods, we highlight how multi-LLM collaboration addresses challenges that a single LLM struggles with, such as reliability, democratization, and pluralism. Finally, we identify the limitations of existing multi-LLM methods and motivate future work. We envision multi-LLM collaboration as an essential path toward compositional intelligence and collaborative AI development. 

**Abstract (ZH)**: 本文的观点论文认为，在许多实际场景（即复杂、情境化和主观的场景）中，单一的语言模型（LLM）不足以产生可靠的结果。我们挑战仅依赖单一通用语言模型的现状，并呼吁多语言模型协作以更好地代表数据、技能和人群的广泛多样性。首先，我们认为单一语言模型未能准确反映现实世界的数据分布、异质技能和多元人口，这样的代表性缺口无法通过进一步训练单一语言模型轻易弥补。然后，我们按照访问级别和信息交换级别将现有的多语言模型协作方法组织成一个层次结构，涵盖从API级、文本级、logit级到权重级的协作。基于这些方法，我们强调多语言模型协作如何解决单一语言模型难以应对的挑战，如可靠性、民主化和多元性。最后，我们指出了现有多语言模型方法的局限性，并激发未来的研究。我们设想多语言模型协作是实现组合智能和协作人工智能开发不可或缺的途径。 

---
# ULPT: Prompt Tuning with Ultra-Low-Dimensional Optimization 

**Title (ZH)**: ULPT：超低维度优化的提示调优 

**Authors**: Zijun Wu, Yongchang Hao, Lili Mou  

**Link**: [PDF](https://arxiv.org/pdf/2502.04501)  

**Abstract**: Large language models achieve state-of-the-art performance but are costly to fine-tune due to their size. Parameter-efficient fine-tuning methods, such as prompt tuning, address this by reducing trainable parameters while maintaining strong performance. However, prior methods tie prompt embeddings to the model's dimensionality, which may not scale well with larger LLMs and more customized LLMs. In this paper, we propose Ultra-Low-dimensional Prompt Tuning (ULPT), which optimizes prompts in a low-dimensional space (e.g., 2D) and use a random but frozen matrix for the up-projection. To enhance alignment, we introduce learnable shift and scale embeddings. ULPT drastically reduces the trainable parameters, e.g., 2D only using 2% parameters compared with vanilla prompt tuning while retaining most of the performance across 21 NLP tasks. Our theoretical analysis shows that random projections can capture high-rank structures effectively, and experimental results demonstrate ULPT's competitive performance over existing parameter-efficient methods. 

**Abstract (ZH)**: 大型语言模型在性能上达到了最先进的水平，但在微调过程中由于模型规模庞大而成本高昂。参数高效的微调方法，如提示调优，通过减少可训练参数数量同时保持强劲的性能来解决这一问题。然而，先前的方法将提示嵌入与模型维度绑定在一起，这在处理更大的LLM或更定制化的LLM时可能不具备良好的扩展性。在本文中，我们提出了超低维提示调优（ULPT），该方法在低维空间（例如2D）中优化提示，并使用随机但固定的矩阵进行上投影。为增强对齐，我们引入了可学习的平移和缩放嵌入。ULPT极大地减少了可训练参数的数量，例如在2D空间中仅使用原始提示调优2%的参数，同时在21个自然语言处理任务上保留了大部分性能。我们的理论分析表明，随机投影能够有效地捕捉高秩结构，实验结果也证明了ULPT在现有参数高效方法中的竞争力。 

---
# Verifiable Format Control for Large Language Model Generations 

**Title (ZH)**: 可验证的格式控制方法在大型语言模型生成中的应用 

**Authors**: Zhaoyang Wang, Jinqi Jiang, Huichi Zhou, Wenhao Zheng, Xuchao Zhang, Chetan Bansal, Huaxiu Yao  

**Link**: [PDF](https://arxiv.org/pdf/2502.04498)  

**Abstract**: Recent Large Language Models (LLMs) have demonstrated satisfying general instruction following ability. However, small LLMs with about 7B parameters still struggle fine-grained format following (e.g., JSON format), which seriously hinder the advancements of their applications. Most existing methods focus on benchmarking general instruction following while overlook how to improve the specific format following ability for small LLMs. Besides, these methods often rely on evaluations based on advanced LLMs (e.g., GPT-4), which can introduce the intrinsic bias of LLMs and be costly due to the API calls. In this paper, we first curate a fully verifiable format following dataset VFF. In contrast to existing works often adopting external LLMs for instruction-following validations, every sample of VFF can be easily validated with a Python function. Further, we propose to leverage this verifiable feature to synthesize massive data for progressively training small LLMs, in order to improve their format following abilities. Experimental results highlight the prevalent limitations in the format following capabilities of 7B level open-source LLMs and demonstrate the effectiveness of our method in enhancing this essential ability. 

**Abstract (ZH)**: 近年来，大规模语言模型（LLMs）展示了令人满意的通用指令遵循能力。然而，大约有70亿参数的小规模LLMs仍然在细粒度格式遵循方面（例如，JSON格式）存在困难，这严重妨碍了它们的应用推进。目前大多数现有方法侧重于评估通用指令遵循，而忽视了如何提高小规模LLMs的具体格式遵循能力。此外，这些方法往往依赖于基于先进LLM（例如GPT-4）的评估，这可能会引入LLM的固有偏见，并由于API调用而导致成本高昂。在本文中，我们首先整理了一个完全可验证的格式遵循数据集VFF。与现有工作中常使用外部LLM进行指令遵循验证不同，VFF中的每个样本都可以通过Python函数轻松验证。进一步地，我们提出利用这一可验证特性来合成大量数据，以逐步训练小规模LLMs，从而提高它们的格式遵循能力。实验结果突显了70亿参数级开源LLMs在格式遵循能力方面的普遍限制，并证明了我们方法在这项基本能力增强方面的有效性。 

---
# Multi-Agent Reinforcement Learning with Focal Diversity Optimization 

**Title (ZH)**: 多智能体强化学习中的焦点多样性优化 

**Authors**: Selim Furkan Tekin, Fatih Ilhan, Tiansheng Huang, Sihao Hu, Zachary Yahn, Ling Liu  

**Link**: [PDF](https://arxiv.org/pdf/2502.04492)  

**Abstract**: The advancement of Large Language Models (LLMs) and their finetuning strategies has triggered the renewed interests in multi-agent reinforcement learning. In this paper, we introduce a focal diversity-optimized multi-agent reinforcement learning approach, coined as MARL-Focal, with three unique characteristics. First, we develop an agent-fusion framework for encouraging multiple LLM based agents to collaborate in producing the final inference output for each LLM query. Second, we develop a focal-diversity optimized agent selection algorithm that can choose a small subset of the available agents based on how well they can complement one another to generate the query output. Finally, we design a conflict-resolution method to detect output inconsistency among multiple agents and produce our MARL-Focal output through reward-aware and policy-adaptive inference fusion. Extensive evaluations on five benchmarks show that MARL-Focal is cost-efficient and adversarial-robust. Our multi-agent fusion model achieves performance improvement of 5.51\% compared to the best individual LLM-agent and offers stronger robustness over the TruthfulQA benchmark. Code is available at this https URL 

**Abstract (ZH)**: 大语言模型（LLMs）的进步及其微调策略重新引发了对多智能体强化学习的兴趣。本文介绍了一种具有三个独特特征的焦点多样化优化多智能体强化学习方法，命名为MARL-Focal。首先，我们开发了一种智能体融合框架，以鼓励多种基于LLM的智能体合作生成每个LLM查询的最终推理输出。其次，我们开发了一种焦点多样化优化的智能体选择算法，可以根据它们如何相互补充生成查询输出来选择可用智能体的小子集。最后，我们设计了一种冲突解决方法来检测多个智能体之间的输出不一致性，并通过基于奖励感知和策略自适应的推理融合生成我们的MARL-Focal输出。在五个基准上的广泛评估表明，MARL-Focal具有成本效益且对抗鲁棒性强。我们的多智能体融合模型在TruthfulQA基准上的性能比最佳单个LLM智能体提高了5.51%，并在鲁棒性方面表现更优异。相关代码可在以下地址获得：[此处提供网址] 

---
# Building A Unified AI-centric Language System: analysis, framework and future work 

**Title (ZH)**: 构建统一的人工智能中心语言系统：分析、框架及未来工作 

**Authors**: Edward Hong Wang, Cynthia Xin Wen  

**Link**: [PDF](https://arxiv.org/pdf/2502.04488)  

**Abstract**: Recent advancements in large language models have demonstrated that extended inference through techniques can markedly improve performance, yet these gains come with increased computational costs and the propagation of inherent biases found in natural languages. This paper explores the design of a unified AI-centric language system that addresses these challenges by offering a more concise, unambiguous, and computationally efficient alternative to traditional human languages. We analyze the limitations of natural language such as gender bias, morphological irregularities, and contextual ambiguities and examine how these issues are exacerbated within current Transformer architectures, where redundant attention heads and token inefficiencies prevail. Drawing on insights from emergent artificial communication systems and constructed languages like Esperanto and Lojban, we propose a framework that translates diverse natural language inputs into a streamlined AI-friendly language, enabling more efficient model training and inference while reducing memory footprints. Finally, we outline a pathway for empirical validation through controlled experiments, paving the way for a universal interchange format that could revolutionize AI-to-AI and human-to-AI interactions by enhancing clarity, fairness, and overall performance. 

**Abstract (ZH)**: 最近在大型语言模型方面的进展表明，通过扩展推理技术可以显著提高性能，但这些改进伴随着计算成本的增加和自然语言中固有偏见的传播。本文探讨了一种统一的人工智能为中心的语言系统的设计，该系统通过提供一种更为简洁、明确且计算高效的替代传统人类语言的方式，来应对这些挑战。我们分析了自然语言的局限性，如性别偏见、形态不规则性和上下文歧义性，并研究了这些问题在当前的Transformer架构中是如何被放大的，这些架构普遍存在冗余的注意力头部和令牌效率低下等问题。借鉴新兴的人工交流系统以及世界语（如Esperanto）和构造语言（如Lojban）的原理，我们提出了一种框架，该框架将多样化的自然语言输入转化为一种精简的AI友好语言，从而实现更高效的模型训练和推理，并减少内存占用。最后，我们概述了一条通过受控实验进行实证验证的途径，为促进AI-对-AI和人-对-AI交互提供了可能的标准化格式，以增强清晰度、公平性和整体性能。 

---
# Active Task Disambiguation with LLMs 

**Title (ZH)**: 使用大语言模型进行活跃任务去模糊化 

**Authors**: Katarzyna Kobalczyk, Nicolas Astorga, Tennison Liu, Mihaela van der Schaar  

**Link**: [PDF](https://arxiv.org/pdf/2502.04485)  

**Abstract**: Despite the impressive performance of large language models (LLMs) across various benchmarks, their ability to address ambiguously specified problems--frequent in real-world interactions--remains underexplored. To address this gap, we introduce a formal definition of task ambiguity and frame the problem of task disambiguation through the lens of Bayesian Experimental Design. By posing clarifying questions, LLM agents can acquire additional task specifications, progressively narrowing the space of viable solutions and reducing the risk of generating unsatisfactory outputs. Yet, generating effective clarifying questions requires LLM agents to engage in a form of meta-cognitive reasoning, an ability LLMs may presently lack. Our proposed approach of active task disambiguation enables LLM agents to generate targeted questions maximizing the information gain. Effectively, this approach shifts the load from implicit to explicit reasoning about the space of viable solutions. Empirical results demonstrate that this form of question selection leads to more effective task disambiguation in comparison to approaches relying on reasoning solely within the space of questions. 

**Abstract (ZH)**: 尽管大型语言模型（LLMs）在各种基准测试中的表现令人印象深刻，但它们在处理现实生活互动中常见的不明确指定的问题的能力仍然未被充分研究。为填补这一空白，我们提出了任务不明确性的正式定义，并将任务去模糊化问题通过贝叶斯实验设计的视角进行框架化。通过提出澄清性的问题，LLM代理可以获取额外的任务指定，逐步缩小可行解决方案的空间，并减少生成不满意输出的风险。然而，生成有效的澄清性问题需要LLM代理进行一种形式的元认知推理，而这一能力LLMs目前可能尚未具备。我们提出的一种主动任务去模糊化方法使LLM代理能够生成最大化信息增益的针对性问题。因此，这种方法将推理的负担从隐含的推理转移到了明确的推理上。实验证据表明，这种问题选择方式在任务去模糊化方面的效果优于仅在问题空间内推理的方法。 

---
# "In order that" -- a data driven study of symptoms and causes of obsolescence 

**Title (ZH)**: “为了阐明”——基于数据的研究方法探讨老化的问题症状及其原因 

**Authors**: Karolina Rudnicka  

**Link**: [PDF](https://arxiv.org/pdf/2502.04457)  

**Abstract**: The paper is an empirical case study of grammatical obsolescence in progress. The main studied variable is the purpose subordinator in order that, which is shown to be steadily decreasing in the frequency of use starting from the beginning of the twentieth century. This work applies a data-driven approach for the investigation and description of obsolescence, recently developed by the Rudnicka (2019). The methodology combines philological analysis with statistical methods used on data acquired from mega-corpora. Moving from the description of possible symptoms of obsolescence to different causes for it, the paper aims at presenting a comprehensive account of the studied phenomenon. Interestingly, a very significant role in the decline of in order that can be ascribed to the so-called higher-order processes, understood as processes influencing the constructional level from above. Two kinds of higher-order processes are shown to play an important role, namely i) an externally-motivated higher-order process exemplified by the drastic socio-cultural changes of the 19th and 20th centuries; ii) an internally-motivated higher-order processes instantiated by the rise of the to-infinitive (rise of infinite clauses). 

**Abstract (ZH)**: 本文是对语法式微现象进行实证案例研究的论文。主要研究变量是连词“in order that”，研究表明这种连词的使用频率从20世纪初开始逐渐下降。本文采用了一种以数据为驱动的方法进行研究和描述式微现象，这一方法最近由Rudnicka (2019) 发展。该方法结合了语义学分析和统计方法，所使用的数据源于大规模语料库。从描述式微现象可能的症状到探讨其原因，本文旨在全面阐述所研究的现象。有趣的是，在“in order that”的衰落过程中，所谓的“高层过程”起到了非常显著的作用，这里所指的“高层过程”是从上层影响构式层面的过程。证明了两种类型的高层过程发挥了重要作用：一是由社会文化变化等外部因素驱动的高层过程；二是由不定式（无标记动词短语）增多引发的内部驱动的高层过程。 

---
# Confident or Seek Stronger: Exploring Uncertainty-Based On-device LLM Routing From Benchmarking to Generalization 

**Title (ZH)**: 自信还是寻求更强：基于不确定性在设备端LLM路由中的探索：从基准测试到泛化 

**Authors**: Yu-Neng Chuang, Leisheng Yu, Guanchu Wang, Lizhe Zhang, Zirui Liu, Xuanting Cai, Yang Sui, Vladimir Braverman, Xia Hu  

**Link**: [PDF](https://arxiv.org/pdf/2502.04428)  

**Abstract**: Large language models (LLMs) are increasingly deployed and democratized on edge devices. To improve the efficiency of on-device deployment, small language models (SLMs) are often adopted due to their efficient decoding latency and reduced energy consumption. However, these SLMs often generate inaccurate responses when handling complex queries. One promising solution is uncertainty-based SLM routing, offloading high-stakes queries to stronger LLMs when resulting in low-confidence responses on SLM. This follows the principle of "If you lack confidence, seek stronger support" to enhance reliability. Relying on more powerful LLMs is yet effective but increases invocation costs. Therefore, striking a routing balance between efficiency and efficacy remains a critical challenge. Additionally, efficiently generalizing the routing strategy to new datasets remains under-explored. In this paper, we conduct a comprehensive investigation into benchmarking and generalization of uncertainty-driven routing strategies from SLMs to LLMs over 1500+ settings. Our findings highlight: First, uncertainty-correctness alignment in different uncertainty quantification (UQ) methods significantly impacts routing performance. Second, uncertainty distributions depend more on both the specific SLM and the chosen UQ method, rather than downstream data. Building on the insight, we propose a calibration data construction instruction pipeline and open-source a constructed hold-out set to enhance routing generalization on new downstream scenarios. The experimental results indicate calibration data effectively bootstraps routing performance without any new data. 

**Abstract (ZH)**: 将下面的论文内容或标题翻译成中文，符合学术规范：

大型语言模型（LLMs）在边缘设备上的部署日益广泛。为了提高设备端部署的效率，通常会采用小型语言模型（SLMs），因为它们具有高效的解码延迟和较低的能耗。然而，这些SLMs在处理复杂查询时往往会生成不准确的响应。一种有前景的解决方案是基于不确定性的小型语言模型路由策略，即当SLM生成低置信度响应时，将高风险查询卸载到更强的LLM上。这一方法遵循“如果你缺乏信心，寻求更强的支持”这一原则，以提高可靠性。虽然依靠更强大的LLM可以提高可靠性，但同时也会增加调用成本。因此，在效率与有效性之间取得平衡的路由策略仍然是一个关键挑战。此外，如何高效地将路由策略推广到新的数据集上仍是一个未被充分探索的问题。在本文中，我们对从SLM到LLM的基于不确定性的路由策略进行了全面的基准测试和泛化研究，在超过1500种设置下进行了分析。我们的研究发现如下：首先，不同不确定性量化（UQ）方法中的不确定性-正确性对齐显著影响路由性能；其次，不确定性分布更多地取决于具体的SLM和所选的UQ方法，而非下游数据。基于这些洞察，我们提出了一种校准数据生成指令管道，并开放了一个构建的保留集，以增强新的下游场景下的路由泛化能力。实验证明，校准数据能够有效提高路由性能，而无需引入新的数据。 

---
# Decoding AI Judgment: How LLMs Assess News Credibility and Bias 

**Title (ZH)**: 解码AI判断：LLMs如何评估新闻可信度与偏见 

**Authors**: Edoardo Loru, Jacopo Nudo, Niccolò Di Marco, Matteo Cinelli, Walter Quattrociocchi  

**Link**: [PDF](https://arxiv.org/pdf/2502.04426)  

**Abstract**: Large Language Models (LLMs) are increasingly used to assess news credibility, yet little is known about how they make these judgments. While prior research has examined political bias in LLM outputs or their potential for automated fact-checking, their internal evaluation processes remain largely unexamined. Understanding how LLMs assess credibility provides insights into AI behavior and how credibility is structured and applied in large-scale language models. This study benchmarks the reliability and political classifications of state-of-the-art LLMs - Gemini 1.5 Flash (Google), GPT-4o mini (OpenAI), and LLaMA 3.1 (Meta) - against structured, expert-driven rating systems such as NewsGuard and Media Bias Fact Check. Beyond assessing classification performance, we analyze the linguistic markers that shape LLM decisions, identifying which words and concepts drive their evaluations. We uncover patterns in how LLMs associate credibility with specific linguistic features by examining keyword frequency, contextual determinants, and rank distributions. Beyond static classification, we introduce a framework in which LLMs refine their credibility assessments by retrieving external information, querying other models, and adapting their responses. This allows us to investigate whether their assessments reflect structured reasoning or rely primarily on prior learned associations. 

**Abstract (ZH)**: 以下是翻译成中文的内容，符合学术规范：

大型语言模型（LLMs）越来越多地被用于评估新闻的可信度，但对其评估判断的过程我们知之甚少。尽管先前的研究已经探讨了LLMs输出中的政治偏见或它们在自动化事实核查方面的潜力，但它们的内部评估机制仍然鲜有研究。理解LLMs如何评估可信度有助于我们了解人工智能的行为以及大规模语言模型中如何构建和应用可信度的概念。本研究将最先进的LLMs——Google的Gemini 1.5 Flash、OpenAI的GPT-4o mini和Meta的LLaMA 3.1——与具有结构化、专家驱动的评级系统（如NewsGuard和Media Bias Fact Check）进行基准测试。除了评估分类性能外，我们还分析塑造LLMs决策的语言标记，识别出哪些单词和概念影响其评估结果。通过分析关键词频率、上下文决定因素和排名分布，我们揭示了LLMs如何将可信度与特定语言特征相关联的模式。本研究还提出了一种框架，其中LLMs通过检索外部信息、查询其他模型并调整其回应来细化其可信度评估。这使我们能够探讨它们的评估结果是否反映了结构化的逻辑推理，还是主要依赖于先前学习的关联。

此翻译过程中保持了原文的学术严谨性和结构完整性，希望对您有所帮助。 

---
# EmoBench-M: Benchmarking Emotional Intelligence for Multimodal Large Language Models 

**Title (ZH)**: EmoBench-M：多模态大型语言模型的情感intelligence基准测试 

**Authors**: He Hu, Yucheng Zhou, Lianzhong You, Hongbo Xu, Qianning Wang, Zheng Lian, Fei Richard Yu, Fei Ma, Laizhong Cui  

**Link**: [PDF](https://arxiv.org/pdf/2502.04424)  

**Abstract**: With the integration of Multimodal large language models (MLLMs) into robotic systems and various AI applications, embedding emotional intelligence (EI) capabilities into these models is essential for enabling robots to effectively address human emotional needs and interact seamlessly in real-world scenarios. Existing static, text-based, or text-image benchmarks overlook the multimodal complexities of real-world interactions and fail to capture the dynamic, multimodal nature of emotional expressions, making them inadequate for evaluating MLLMs' EI. Based on established psychological theories of EI, we build EmoBench-M, a novel benchmark designed to evaluate the EI capability of MLLMs across 13 valuation scenarios from three key dimensions: foundational emotion recognition, conversational emotion understanding, and socially complex emotion analysis. Evaluations of both open-source and closed-source MLLMs on EmoBench-M reveal a significant performance gap between them and humans, highlighting the need to further advance their EI capabilities. All benchmark resources, including code and datasets, are publicly available at this https URL. 

**Abstract (ZH)**: 将以下论文内容或标题翻译成中文，符合学术规范：

随着多模态大语言模型（MLLMs）被整合到机器人系统和各种AI应用中，将情感智能（EI）能力嵌入这些模型中对于使机器人能够有效满足人类的情感需求并在现实世界场景中无缝交互至关重要。现有的静态、基于文本或图文基准忽略了真实世界交互的多模态复杂性，未能捕捉到情感表达的动态和多模态性质，使它们无法有效评估MLLMs的情感智能能力。基于已建立的情感智能心理学理论，我们构建了EmoBench-M这一新型基准，用于从三个关键维度评估MLLMs在13个评价场景中的情感智能能力：基础情感识别、对话情感理解以及社会复杂情感分析。对开源和闭源MLLMs在EmoBench-M上的评估揭示了它们与人类的情感智能能力之间存在显著差异，突显了进一步提升其情感智能能力的必要性。所有基准资源，包括代码和数据集，均可在以下网址公开访问：[此处请插入网址]。 

---
# MedRAG: Enhancing Retrieval-augmented Generation with Knowledge Graph-Elicited Reasoning for Healthcare Copilot 

**Title (ZH)**: MedRAG：通过知识图谱驱动的推理增强医疗陪飞的检索增强生成 

**Authors**: Xuejiao Zhao, Siyan Liu, Su-Yin Yang, Chunyan Miao  

**Link**: [PDF](https://arxiv.org/pdf/2502.04413)  

**Abstract**: Retrieval-augmented generation (RAG) is a well-suited technique for retrieving privacy-sensitive Electronic Health Records (EHR). It can serve as a key module of the healthcare copilot, helping reduce misdiagnosis for healthcare practitioners and patients. However, the diagnostic accuracy and specificity of existing heuristic-based RAG models used in the medical domain are inadequate, particularly for diseases with similar manifestations. This paper proposes MedRAG, a RAG model enhanced by knowledge graph (KG)-elicited reasoning for the medical domain that retrieves diagnosis and treatment recommendations based on manifestations. MedRAG systematically constructs a comprehensive four-tier hierarchical diagnostic KG encompassing critical diagnostic differences of various diseases. These differences are dynamically integrated with similar EHRs retrieved from an EHR database, and reasoned within a large language model. This process enables more accurate and specific decision support, while also proactively providing follow-up questions to enhance personalized medical decision-making. MedRAG is evaluated on both a public dataset DDXPlus and a private chronic pain diagnostic dataset (CPDD) collected from Tan Tock Seng Hospital, and its performance is compared against various existing RAG methods. Experimental results show that, leveraging the information integration and relational abilities of the KG, our MedRAG provides more specific diagnostic insights and outperforms state-of-the-art models in reducing misdiagnosis rates. Our code will be available at this https URL 

**Abstract (ZH)**: 以下是经过学术规范翻译的内容：

检索增强生成（RAG）技术非常适合用于检索敏感的电子健康记录（EHR）。它可以作为医疗协作者的关键模块，帮助减少医疗从业者和患者在诊断中的误诊现象。然而，医疗领域中现有基于启发式的RAG模型的诊断准确性和特异性不足，特别是在表现相似的疾病诊断中更为明显。本文提出了一种名为MedRAG的模型，该模型通过知识图谱（KG）引导的推理增强，能够在症状基础上检索出诊断和治疗建议。MedRAG系统地构建了一个全面的四级层次诊断知识图谱，涵盖了各种疾病的诊断关键差异。这些差异与从EHR数据库中检索到的类似EHR动态集成，并在大型语言模型中进行推理。这一过程提高了决策支持的准确性和特异性，同时还主动提供后续问题，以增强个性化的医疗决策制定。MedRAG在公开数据集DDXPlus和来自淡马锡综合医院的私人慢性疼痛诊断数据集（CPDD）上进行了评估，并将其性能与现有的多种RAG方法进行了比较。实验结果表明，利用知识图谱的信息整合和关系处理能力，我们的MedRAG提供了更具体的诊断见解，并在减少误诊率方面优于最先进的模型。我们的代码将在此处提供：[该处链接] 

---
# Step Back to Leap Forward: Self-Backtracking for Boosting Reasoning of Language Models 

**Title (ZH)**: 回归以跨越：自回溯机制提升语言模型的推理能力 

**Authors**: Xiao-Wen Yang, Xuan-Yi Zhu, Wen-Da Wei, Ding-Chu Zhang, Jie-Jing Shao, Zhi Zhou, Lan-Zhe Guo, Yu-Feng Li  

**Link**: [PDF](https://arxiv.org/pdf/2502.04404)  

**Abstract**: The integration of slow-thinking mechanisms into large language models (LLMs) offers a promising way toward achieving Level 2 AGI Reasoners, as exemplified by systems like OpenAI's o1. However, several significant challenges remain, including inefficient overthinking and an overreliance on auxiliary reward models. We point out that these limitations stem from LLMs' inability to internalize the search process, a key component of effective reasoning. A critical step toward addressing this issue is enabling LLMs to autonomously determine when and where to backtrack, a fundamental operation in traditional search algorithms. To this end, we propose a self-backtracking mechanism that equips LLMs with the ability to backtrack during both training and inference. This mechanism not only enhances reasoning ability but also efficiency by transforming slow-thinking processes into fast-thinking through self-improvement. Empirical evaluations demonstrate that our proposal significantly enhances the reasoning capabilities of LLMs, achieving a performance gain of over 40 percent compared to the optimal-path supervised fine-tuning method. We believe this study introduces a novel and promising pathway for developing more advanced and robust Reasoners. 

**Abstract (ZH)**: 将深度学习机制整合到大型语言模型（LLMs）中，为实现层级2的AGI推理器提供了有 promise 的途径，如OpenAI的o1系统所示。然而，仍然存在几个重大挑战，包括过度思考的低效性和过度依赖辅助奖励模型。我们指出，这些局限性源于LLMs无法内化搜索过程，而有效的推理的关键组成部分正是搜索过程。解决这一问题的关键步骤是让LLMs自主决定何时和何地回溯，这一操作是传统搜索算法中的基本操作。为此，我们提出了一种自回溯机制，该机制使LLMs能够在训练和推断过程中进行回溯。该机制不仅增强了推理能力，还通过自我改进将缓慢的思考过程转化为快速的思考，从而提高效率。实证评估表明，我们的提议显著提升了LLMs的推理能力，与最优路径监督微调方法相比，性能提高了超过40%。我们相信，这项研究为开发更加先进和稳健的推理器开辟了一条新颖且有前景的道路。 

---
# Multimodal Medical Code Tokenizer 

**Title (ZH)**: 多模态医疗代码分词器 

**Authors**: Xiaorui Su, Shvat Messica, Yepeng Huang, Ruth Johnson, Lukas Fesser, Shanghua Gao, Faryad Sahneh, Marinka Zitnik  

**Link**: [PDF](https://arxiv.org/pdf/2502.04397)  

**Abstract**: Foundation models trained on patient electronic health records (EHRs) require tokenizing medical data into sequences of discrete vocabulary items. Existing tokenizers treat medical codes from EHRs as isolated textual tokens. However, each medical code is defined by its textual description, its position in ontological hierarchies, and its relationships to other codes, such as disease co-occurrences and drug-treatment associations. Medical vocabularies contain more than 600,000 codes with critical information for clinical reasoning. We introduce MedTok, a multimodal medical code tokenizer that uses the text descriptions and relational context of codes. MedTok processes text using a language model encoder and encodes the relational structure with a graph encoder. It then quantizes both modalities into a unified token space, preserving modality-specific and cross-modality information. We integrate MedTok into five EHR models and evaluate it on operational and clinical tasks across in-patient and out-patient datasets, including outcome prediction, diagnosis classification, drug recommendation, and risk stratification. Swapping standard EHR tokenizers with MedTok improves AUPRC across all EHR models, by 4.10% on MIMIC-III, 4.78% on MIMIC-IV, and 11.30% on EHRShot, with the largest gains in drug recommendation. Beyond EHR modeling, we demonstrate using MedTok tokenizer with medical QA systems. Our results demonstrate the potential of MedTok as a unified tokenizer for medical codes, improving tokenization for medical foundation models. 

**Abstract (ZH)**: 基于患者电子健康记录（EHR）训练的基准模型需要将医疗数据标记为离散词汇项的序列。现有标记器将EHR中的医疗代码视为孤立的文本标记。然而，每个医疗代码是由其文本描述、其在本体层次结构中的位置以及与其他代码的关系（例如疾病共现和药物治疗关联）定义的。医疗词汇表包含超过60万种代码，对于临床推理至关重要。我们引入了MedTok，这是一种利用代码的文本描述和关系背景的多模态医疗代码标记器。MedTok使用语言模型编码器处理文本，并使用图编码器编码关系结构。然后，它将这两种模态量化到统一的标记空间中，保留模态特定信息和跨模态信息。我们将MedTok集成到五个EHR模型中，并在住院和门诊数据集上对其进行了操作和临床任务评估，包括预测结果、诊断分类、药物推荐和风险分层。用MedTok替换标准的EHR标记器，所有EHR模型的AUPRC均有所提高，在MIMIC-III上的提高幅度为4.10%，在MIMIC-IV上的提高幅度为4.78%，在EHRShot上的提高幅度为11.30%，其中药物推荐领域的收益最大。除了EHR建模之外，我们还展示了在医疗问答系统中使用MedTok标记器。我们的结果显示，MedTok作为统一的医疗代码标记器的潜力，可以改进医疗基础模型的标记。 

---
# DECT: Harnessing LLM-assisted Fine-Grained Linguistic Knowledge and Label-Switched and Label-Preserved Data Generation for Diagnosis of Alzheimer's Disease 

**Title (ZH)**: DECT：利用LLM辅助的细粒度语言知识和标签转换与保留数据生成技术进行阿尔茨海默病诊断 

**Authors**: Tingyu Mo, Jacqueline C. K. Lam, Victor O.K. Li, Lawrence Y. L. Cheung  

**Link**: [PDF](https://arxiv.org/pdf/2502.04394)  

**Abstract**: Alzheimer's Disease (AD) is an irreversible neurodegenerative disease affecting 50 million people worldwide. Low-cost, accurate identification of key markers of AD is crucial for timely diagnosis and intervention. Language impairment is one of the earliest signs of cognitive decline, which can be used to discriminate AD patients from normal control individuals. Patient-interviewer dialogues may be used to detect such impairments, but they are often mixed with ambiguous, noisy, and irrelevant information, making the AD detection task difficult. Moreover, the limited availability of AD speech samples and variability in their speech styles pose significant challenges in developing robust speech-based AD detection models. To address these challenges, we propose DECT, a novel speech-based domain-specific approach leveraging large language models (LLMs) for fine-grained linguistic analysis and label-switched label-preserved data generation. Our study presents four novelties: We harness the summarizing capabilities of LLMs to identify and distill key Cognitive-Linguistic information from noisy speech transcripts, effectively filtering irrelevant information. We leverage the inherent linguistic knowledge of LLMs to extract linguistic markers from unstructured and heterogeneous audio transcripts. We exploit the compositional ability of LLMs to generate AD speech transcripts consisting of diverse linguistic patterns to overcome the speech data scarcity challenge and enhance the robustness of AD detection models. We use the augmented AD textual speech transcript dataset and a more fine-grained representation of AD textual speech transcript data to fine-tune the AD detection model. The results have shown that DECT demonstrates superior model performance with an 11% improvement in AD detection accuracy on the datasets from DementiaBank compared to the baselines. 

**Abstract (ZH)**: 阿尔茨海默病（AD）是一种不可逆的神经退行性疾病，全球约有5000万人受到影响。准确、低成本地识别AD的关键生物标志物对于及时诊断和干预至关重要。语言障碍是认知衰退的最早迹象之一，可以用来区分AD患者和正常对照个体。患者与访谈者之间的对话可以检测这些障碍，但这些对话 often 混杂着模糊、噪声和无关的信息，使AD检测任务变得困难。此外，AD语音样本的有限可用性和其语音风格的差异性给开发稳健的基于语音的AD检测模型带来了重大挑战。为了解决这些挑战，我们提出了一种新颖的方法DECT，该方法利用大型语言模型（LLMs）进行细粒度的语言分析，并生成标签切换但保持标签的数据。我们的研究提出了四个创新点：

1. 我们利用LLMs的总结能力，从嘈杂的语音转录中识别和提炼关键的认知-语言信息，有效过滤无关信息。
2. 我们利用LLMs固有的语言知识，从不规则且异构的音频转录中提取语言特征。
3. 利用LLMs的组合能力，生成包含多种语言模式的AD语音转录，从而克服语音数据稀缺的问题，并增强AD检测模型的稳健性。
4. 我们使用增强的AD文本语音转录数据集以及更细粒度的AD文本语音转录数据表示，对AD检测模型进行微调。实验结果表明，DECT在DementiaBank数据集上的AD检测准确性相较于基线模型提高了11%。

请注意，这里的翻译尽可能保持了原文的技术性和学术性，并进行了适当的学术化调整，以符合中文的表达习惯。 

---
# Division-of-Thoughts: Harnessing Hybrid Language Model Synergy for Efficient On-Device Agents 

**Title (ZH)**: 思绪划分：利用混合语言模型协同效应提升设备端代理的效率 

**Authors**: Chenyang Shao, Xinyuan Hu, Yutang Lin, Fengli Xu  

**Link**: [PDF](https://arxiv.org/pdf/2502.04392)  

**Abstract**: The rapid expansion of web content has made on-device AI assistants indispensable for helping users manage the increasing complexity of online tasks. The emergent reasoning ability in large language models offer a promising path for next-generation on-device AI agents. However, deploying full-scale Large Language Models (LLMs) on resource-limited local devices is challenging. In this paper, we propose Division-of-Thoughts (DoT), a collaborative reasoning framework leveraging the synergy between locally deployed Smaller-scale Language Models (SLMs) and cloud-based LLMs. DoT leverages a Task Decomposer to elicit the inherent planning abilities in language models to decompose user queries into smaller sub-tasks, which allows hybrid language models to fully exploit their respective strengths. Besides, DoT employs a Task Scheduler to analyze the pair-wise dependency of sub-tasks and create a dependency graph, facilitating parallel reasoning of sub-tasks and the identification of key steps. To allocate the appropriate model based on the difficulty of sub-tasks, DoT leverages a Plug-and-Play Adapter, which is an additional task head attached to the SLM that does not alter the SLM's parameters. To boost adapter's task allocation capability, we propose a self-reinforced training method that relies solely on task execution feedback. Extensive experiments on various benchmarks demonstrate that our DoT significantly reduces LLM costs while maintaining competitive reasoning accuracy. Specifically, DoT reduces the average reasoning time and API costs by 66.12% and 83.57%, while achieving comparable reasoning accuracy with the best baseline methods. 

**Abstract (ZH)**: 互联网内容的迅速扩展使得本地设备人工智能助手对于帮助用户管理日益复杂的在线任务变得不可或缺。大型语言模型新兴的推理能力为下一代本地设备人工智能代理提供了有希望的发展路径。然而，在资源受限的本地设备上部署全规模的大语言模型（LLMs）是具有挑战性的。在本文中，我们提出了一种名为思维分工（DoT）的协作推理框架，该框架充分利用了本地部署的小规模语言模型（SLMs）与云基大语言模型之间的协同效应。DoT 利用任务分解器激发语言模型中的固有规划能力，将用户查询分解为较小的子任务，从而让混合语言模型充分发挥各自的优点。此外，DoT 使用任务调度器分析子任务的两两依赖关系，构建依赖图，促进子任务的并行推理和关键步骤的识别。为根据子任务的难度分配合适的模型，DoT 利用了一种可插拔适配器，这是一种附加的任务头，附加到小型语言模型上，而不改变小型语言模型的参数。为了增强适配器的任务分配能力，我们提出了一种基于任务执行反馈的自我强化训练方法。在各种基准测试上的广泛实验表明，我们的 DoT 在显著降低大语言模型成本的同时，保持了竞争性的推理准确性。具体而言，DoT 将平均推理时间和 API 成本分别降低了 66.12% 和 83.57%，同时实现了与最佳基线方法相当的推理准确性。 

---
# In Praise of Stubbornness: The Case for Cognitive-Dissonance-Aware Knowledge Updates in LLMs 

**Title (ZH)**: 赞美执拗：认知失调意识知识更新在大规模语言模型中的必要性 

**Authors**: Simone Clemente, Zied Ben Houidi, Alexis Huet, Dario Rossi, Giulio Franzese, Pietro Michiardi  

**Link**: [PDF](https://arxiv.org/pdf/2502.04390)  

**Abstract**: Despite remarkable capabilities, large language models (LLMs) struggle to continually update their knowledge without catastrophic forgetting. In contrast, humans effortlessly integrate new information, detect conflicts with existing beliefs, and selectively update their mental models. This paper introduces a cognitive-inspired investigation paradigm to study continual knowledge updating in LLMs. We implement two key components inspired by human cognition: (1) Dissonance and Familiarity Awareness, analyzing model behavior to classify information as novel, familiar, or dissonant; and (2) Targeted Network Updates, which track neural activity to identify frequently used (stubborn) and rarely used (plastic) neurons. Through carefully designed experiments in controlled settings, we uncover a number of empirical findings demonstrating the potential of this approach. First, dissonance detection is feasible using simple activation and gradient features, suggesting potential for cognitive-inspired training. Second, we find that non-dissonant updates largely preserve prior knowledge regardless of targeting strategy, revealing inherent robustness in LLM knowledge integration. Most critically, we discover that dissonant updates prove catastrophically destructive to the model's knowledge base, indiscriminately affecting even information unrelated to the current updates. This suggests fundamental limitations in how neural networks handle contradictions and motivates the need for new approaches to knowledge updating that better mirror human cognitive mechanisms. 

**Abstract (ZH)**: 尽管大型语言模型（LLMs）具备卓越的能力，但它们在不发生灾难性遗忘的情况下持续更新知识方面存在困难。相比之下，人类能够轻松地整合新信息，检测与现有信念的冲突，并有选择性地更新其心理模型。本文引入了一种受认知启发的研究范式，以研究LLMs连续知识更新的机制。我们实现了两个关键的人类认知启发性组件：（1）矛盾和熟悉感意识，通过分析模型行为将信息分类为新颖的、熟悉的或矛盾的；（2）目标网络更新，通过追踪神经活动来识别频繁使用（顽固型）和很少使用（可塑性）的神经元。通过在受控环境下的精心设计实验，我们揭示了一些经验发现，表明了这种方法的潜力。首先，利用简单的激活和梯度特征检测矛盾是可行的，这暗示了认知启发性训练的潜力。其次，我们发现非矛盾更新基本上在不考虑目标策略的情况下保留了先前知识，揭示了LLMs知识整合的内在鲁棒性。最后，我们发现矛盾更新对模型的知识库造成了灾难性破坏，不仅影响当前更新相关信息，甚至影响与当前更新完全无关的信息。这表明神经网络处理矛盾的基本局限，并激发了需要新方法的需求，这些方法更好地模仿人类认知机制。 

---
# FedP$^2$EFT: Federated Learning to Personalize Parameter Efficient Fine-Tuning for Multilingual LLMs 

**Title (ZH)**: FedP$^2$EFT: 联邦学习以个性化参数高效微调多语言大型语言模型 

**Authors**: Royson Lee, Minyoung Kim, Fady Rezk, Rui Li, Stylianos I. Venieris, Timothy Hospedales  

**Link**: [PDF](https://arxiv.org/pdf/2502.04387)  

**Abstract**: Federated learning (FL) has enabled the training of multilingual large language models (LLMs) on diverse and decentralized multilingual data, especially on low-resource languages. To improve client-specific performance, personalization via the use of parameter-efficient fine-tuning (PEFT) modules such as LoRA is common. This involves a personalization strategy (PS), such as the design of the PEFT adapter structures (e.g., in which layers to add LoRAs and what ranks) and choice of hyperparameters (e.g., learning rates) for fine-tuning. Instead of manual PS configuration, we propose FedP$^2$EFT, a federated learning-to-personalize method for multilingual LLMs in cross-device FL settings. Unlike most existing PEFT structure selection methods, which are prone to overfitting low-data regimes, FedP$^2$EFT collaboratively learns the optimal personalized PEFT structure for each client via Bayesian sparse rank selection. Evaluations on both simulated and real-world multilingual FL benchmarks demonstrate that FedP$^2$EFT largely outperforms existing personalized fine-tuning methods, while complementing a range of existing FL methods. 

**Abstract (ZH)**: 联邦学习（FL）使得在多样化的分散式多语言数据上，尤其是低资源语言的数据上训练多语言大型语言模型（LLMs）成为可能。为了提高客户端特定的性能，通过使用参数高效微调（PEFT）模块（如LoRA）进行个性化调整是很常见的。这包括个性化策略（PS），例如PEFT适配器结构的设计（例如，在哪些层添加LoRA以及多少秩）和微调时的超参数选择（例如，学习率）。为了避免手动配置PS的复杂性和可能的过拟合，我们提出了一种名为FedP$^2$EFT的方法，该方法是一种用于跨设备FL环境中多语言LLMs的联邦学习-个性化方法。与大多数现有的PEFT结构选择方法不同，后者更容易在低数据情况下过拟合，FedP$^2$EFT通过贝叶斯稀疏秩选择协作学习为每个客户端选择最优的个性化PEFT结构。在仿真实验和真实世界多语言FL基准测试上的评估表明，FedP$^2$EFT大大超过了现有的个性化微调方法，同时补充了多种现有的FL方法。 

---
# Enhancing Reasoning to Adapt Large Language Models for Domain-Specific Applications 

**Title (ZH)**: 增强推理能力以适应特定领域的大规模语言模型应用 

**Authors**: Bo Wen, Xin Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2502.04384)  

**Abstract**: This paper presents SOLOMON, a novel Neuro-inspired Large Language Model (LLM) Reasoning Network architecture that enhances the adaptability of foundation models for domain-specific applications. Through a case study in semiconductor layout design, we demonstrate how SOLOMON enables swift adaptation of general-purpose LLMs to specialized tasks by leveraging Prompt Engineering and In-Context Learning techniques. Our experiments reveal the challenges LLMs face in spatial reasoning and applying domain knowledge to practical problems. Results show that SOLOMON instances significantly outperform their baseline LLM counterparts and achieve performance comparable to state-of-the-art reasoning model, o1-preview. We discuss future research directions for developing more adaptive AI systems that can continually learn, adapt, and evolve in response to new information and changing requirements. 

**Abstract (ZH)**: 本文介绍了SOLOMON，一种新颖的神经启发式大型语言模型（LLM）推理网络架构，该架构能够增强基础模型在特定领域应用中的适应性。通过在半导体布局设计中的案例研究，我们展示了SOLOMON如何通过利用提示工程和上下文学习技术，使通用的LLM迅速适应专门任务。我们的实验揭示了LLM在空间推理和将领域知识应用于实际问题时所面临的挑战。结果表明，SOLOMON实例显著优于其基础模型的对照组，并且达到了与当前最先进推理模型o1-preview相当的性能。我们讨论了未来研发能够持续学习、适应和进化的更适应性AI系统的研究方向。 

---
# Sparse Autoencoders for Hypothesis Generation 

**Title (ZH)**: 稀疏自编码器在假设生成中的应用 

**Authors**: Rajiv Movva, Kenny Peng, Nikhil Garg, Jon Kleinberg, Emma Pierson  

**Link**: [PDF](https://arxiv.org/pdf/2502.04382)  

**Abstract**: We describe HypotheSAEs, a general method to hypothesize interpretable relationships between text data (e.g., headlines) and a target variable (e.g., clicks). HypotheSAEs has three steps: (1) train a sparse autoencoder on text embeddings to produce interpretable features describing the data distribution, (2) select features that predict the target variable, and (3) generate a natural language interpretation of each feature (e.g., "mentions being surprised or shocked") using an LLM. Each interpretation serves as a hypothesis about what predicts the target variable. Compared to baselines, our method better identifies reference hypotheses on synthetic datasets (at least +0.06 in F1) and produces more predictive hypotheses on real datasets (~twice as many significant findings), despite requiring 1-2 orders of magnitude less compute than recent LLM-based methods. HypotheSAEs also produces novel discoveries on two well-studied tasks: explaining partisan differences in Congressional speeches and identifying drivers of engagement with online headlines. 

**Abstract (ZH)**: 我们描述了HypotheSAEs，这是一种通用方法，用于假设文本数据（例如标题）与目标变量（例如点击次数）之间的可解释关系。HypotheSAEs 包含三个步骤：(1) 使用文本嵌入训练稀疏自编码器以生成描述数据分布的可解释特征；(2) 选择预测目标变量的特征；(3) 使用大语言模型（LLM）为每个特征生成自然语言解释（例如，“提到惊讶或震惊”）。每个解释被视为关于是什么预测目标变量的假设。与基线方法相比，我们的方法在合成数据集上更好地识别了参考假设（至少提高0.06的F1分数），并在真实数据集上生成了更多的预测性假设（约是最近基于大语言模型方法的两倍），尽管我们的方法所需的计算资源少一个到两个数量级。HypotheSAEs 还在两个广泛研究的任务中产生了新的发现：解释国会演讲中的党派差异以及识别驱动在线标题浏览量的因素。 

---
# Limitations of Large Language Models in Clinical Problem-Solving Arising from Inflexible Reasoning 

**Title (ZH)**: 大型语言模型在临床问题解决中灵活推理能力的局限性 

**Authors**: Jonathan Kim, Anna Podlasek, Kie Shidara, Feng Liu, Ahmed Alaa, Danilo Bernardo  

**Link**: [PDF](https://arxiv.org/pdf/2502.04381)  

**Abstract**: Large Language Models (LLMs) have attained human-level accuracy on medical question-answer (QA) benchmarks. However, their limitations in navigating open-ended clinical scenarios have recently been shown, raising concerns about the robustness and generalizability of LLM reasoning across diverse, real-world medical tasks. To probe potential LLM failure modes in clinical problem-solving, we present the medical abstraction and reasoning corpus (M-ARC). M-ARC assesses clinical reasoning through scenarios designed to exploit the Einstellung effect -- the fixation of thought arising from prior experience, targeting LLM inductive biases toward inflexible pattern matching from their training data rather than engaging in flexible reasoning. We find that LLMs, including current state-of-the-art o1 and Gemini models, perform poorly compared to physicians on M-ARC, often demonstrating lack of commonsense medical reasoning and a propensity to hallucinate. In addition, uncertainty estimation analyses indicate that LLMs exhibit overconfidence in their answers, despite their limited accuracy. The failure modes revealed by M-ARC in LLM medical reasoning underscore the need to exercise caution when deploying these models in clinical settings. 

**Abstract (ZH)**: 大型语言模型（LLMs）已经在医学问答（QA）基准测试中达到了与人类相当的准确度。然而，它们在处理开放性的临床场景时的局限性最近被揭示出来，这引发了关于LLM在各种实际医学任务中推理的稳健性和泛化的担忧。为了探究LLM在临床问题解决中的潜在失败模式，我们提出了医学抽象与推理语料库（M-ARC）。M-ARC通过设计旨在利用Einstellung效应的情景（即由于先前经验而产生的思维定势）来评估临床推理，针对LLM从训练数据中形成的僵化模式匹配偏向，而不是进行灵活的推理。我们发现，包括当前最先进的o1和Gemini模型在内的LLM在M-ARC上的表现不如医生，经常显示出缺乏常识性的医学推理，并且倾向于生成虚假信息。此外，不确定性估计分析表明，尽管准确度有限，LLM在回答问题时却表现出过度自信。M-ARC揭示的LLM医学推理中的失败模式强调，在临床环境中部署这些模型时需要审慎对待。 

---
# Diversity as a Reward: Fine-Tuning LLMs on a Mixture of Domain-Undetermined Data 

**Title (ZH)**: 多样性作为一种奖励：在混合未知领域数据上微调大型语言模型 

**Authors**: Zhenqing Ling, Daoyuan Chen, Liuyi Yao, Yaliang Li, Ying Shen  

**Link**: [PDF](https://arxiv.org/pdf/2502.04380)  

**Abstract**: Fine-tuning large language models (LLMs) using diverse datasets is crucial for enhancing their overall performance across various domains. In practical scenarios, existing methods based on modeling the mixture proportions of data composition often struggle with data whose domain labels are missing, imprecise or non-normalized, while methods based on data selection usually encounter difficulties in balancing multi-domain performance. To address these challenges, in this paper, we study the role of data diversity in enhancing the overall abilities of LLMs by empirically constructing contrastive data pools and theoretically deriving explanations for both inter- and intra-diversity. Building upon the insights gained, we propose a new method that gives the LLM a dual identity: an output model to cognitively probe and select data based on diversity reward, as well as an input model to be tuned with the selected data. Extensive experiments show that the proposed method notably boosts performance across domain-undetermined data and a series of foundational downstream tasks when applied to various advanced LLMs. We release our code and hope this study can shed light on the understanding of data diversity and advance feedback-driven data-model co-development for LLMs. 

**Abstract (ZH)**: 使用多样数据集微调大规模语言模型（LLMs）对于提升其在各个领域中的整体性能至关重要。在实际场景中，基于数据构成混合比例建模的方法往往难以处理缺失、不精确或未标准化领域标签的数据；而基于数据选择的方法则通常难以在多领域之间实现性能平衡。为应对这些挑战，本文通过实证构建对比数据池，并从理论上解释了跨域多样性和同域多样性的作用，来研究数据多样性在增强LLMs整体能力方面的作用。在此基础上，我们提出了一种新的方法，赋予LLM双重身份：一个基于多样奖励的认知探查和选择模型，以及一个利用选择的数据进行微调的输入模型。广泛的实验表明，当应用于各种先进的LLMs时，该方法能显著提升未确定领域的数据处理能力和一系列基础下游任务的性能。我们发布了代码，希望此次研究能够促进对数据多样性的理解，并推动反馈驱动的数据-模型协同开发。 

---
# MEETING DELEGATE: Benchmarking LLMs on Attending Meetings on Our Behalf 

**Title (ZH)**: 代理参会：在我们的代参会任务上benchmark大语言模型 

**Authors**: Lingxiang Hu, Shurun Yuan, Xiaoting Qin, Jue Zhang, Qingwei Lin, Dongmei Zhang, Saravan Rajmohan, Qi Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2502.04376)  

**Abstract**: In contemporary workplaces, meetings are essential for exchanging ideas and ensuring team alignment but often face challenges such as time consumption, scheduling conflicts, and inefficient participation. Recent advancements in Large Language Models (LLMs) have demonstrated their strong capabilities in natural language generation and reasoning, prompting the question: can LLMs effectively delegate participants in meetings? To explore this, we develop a prototype LLM-powered meeting delegate system and create a comprehensive benchmark using real meeting transcripts. Our evaluation reveals that GPT-4/4o maintain balanced performance between active and cautious engagement strategies. In contrast, Gemini 1.5 Pro tends to be more cautious, while Gemini 1.5 Flash and Llama3-8B/70B display more active tendencies. Overall, about 60\% of responses address at least one key point from the ground-truth. However, improvements are needed to reduce irrelevant or repetitive content and enhance tolerance for transcription errors commonly found in real-world settings. Additionally, we implement the system in practical settings and collect real-world feedback from demos. Our findings underscore the potential and challenges of utilizing LLMs as meeting delegates, offering valuable insights into their practical application for alleviating the burden of meetings. 

**Abstract (ZH)**: 在当代的工作场所中，会议对于交流思想和确保团队协作至关重要，但常常面临时间消耗、日程冲突和参与效率低下的挑战。近年来，大语言模型（LLMs）在自然语言生成和推理方面展现了强大的能力，这引发了这样一个问题：LLMs能否有效地在会议中担任代理？为探讨这一问题，我们开发了一个基于LLM的会议代理原型系统，并使用真实的会议记录创建了一个全面的基准测试。我们的评估表明，GPT-4/40在主动参与和谨慎参与策略之间保持了平衡性能。相比之下，Gemini 1.5 Pro 更倾向于谨慎策略，而 Gemini 1.5 Flash 和 Llama3-8B/70B 则显示出了更多的主动倾向。总体而言，约60%的回复涵盖了至少一个关键点。然而，仍需改进以减少不相关或重复内容，并增强对现实场景中常见的转录错误的容忍度。此外，我们在实际场景中实施了该系统，并从演示中收集了实际反馈。我们的研究结果强调了利用LLMs作为会议代理的潜力和挑战，提供了有关其在减轻会议负担方面的实用应用的宝贵见解。 

---
# An Analysis for Reasoning Bias of Language Models with Small Initialization 

**Title (ZH)**: 小初始值下语言模型推理偏差的分析 

**Authors**: Junjie Yao, Zhongwang Zhang, Zhi-Qin John Xu  

**Link**: [PDF](https://arxiv.org/pdf/2502.04375)  

**Abstract**: Transformer-based Large Language Models (LLMs) have revolutionized Natural Language Processing by demonstrating exceptional performance across diverse tasks. This study investigates the impact of the parameter initialization scale on the training behavior and task preferences of LLMs. We discover that smaller initialization scales encourage models to favor reasoning tasks, whereas larger initialization scales lead to a preference for memorization tasks. We validate this reasoning bias via real datasets and meticulously designed anchor functions. Further analysis of initial training dynamics suggests that specific model components, particularly the embedding space and self-attention mechanisms, play pivotal roles in shaping these learning biases. We provide a theoretical framework from the perspective of model training dynamics to explain these phenomena. Additionally, experiments on real-world language tasks corroborate our theoretical insights. This work enhances our understanding of how initialization strategies influence LLM performance on reasoning tasks and offers valuable guidelines for training models. 

**Abstract (ZH)**: 基于Transformer的大语言模型（LLMs）通过在各种任务上表现出色，已经彻底改变了自然语言处理领域。本研究考察了参数初始化规模对LLMs的训练行为和任务偏好产生的影响。研究发现，较小的初始化规模促使模型更倾向于处理推理任务，而较大的初始化规模则导致模型偏好记忆任务。我们通过实际数据集和精心设计的锚函数验证了这种推理偏见。进一步分析初始训练动态表明，特定的模型组件，特别是嵌入空间和自我注意机制，在形成这些学习偏见中起着关键作用。我们从模型训练动态的角度提供了一个理论框架来解释这些现象。此外，对实际语言任务的实验验证了我们的理论洞察。本研究增进了我们对初始化策略如何影响LLMs在推理任务上的性能的理解，并提供了有价值的训练模型指导。 

---
# Mining Unstructured Medical Texts With Conformal Active Learning 

**Title (ZH)**: 利用符合性主动学习挖掘未结构化医疗文本 

**Authors**: Juliano Genari, Guilherme Tegoni Goedert  

**Link**: [PDF](https://arxiv.org/pdf/2502.04372)  

**Abstract**: The extraction of relevant data from Electronic Health Records (EHRs) is crucial to identifying symptoms and automating epidemiological surveillance processes. By harnessing the vast amount of unstructured text in EHRs, we can detect patterns that indicate the onset of disease outbreaks, enabling faster, more targeted public health responses. Our proposed framework provides a flexible and efficient solution for mining data from unstructured texts, significantly reducing the need for extensive manual labeling by specialists. Experiments show that our framework achieving strong performance with as few as 200 manually labeled texts, even for complex classification problems. Additionally, our approach can function with simple lightweight models, achieving competitive and occasionally even better results compared to more resource-intensive deep learning models. This capability not only accelerates processing times but also preserves patient privacy, as the data can be processed on weaker on-site hardware rather than being transferred to external systems. Our methodology, therefore, offers a practical, scalable, and privacy-conscious approach to real-time epidemiological monitoring, equipping health institutions to respond rapidly and effectively to emerging health threats. 

**Abstract (ZH)**: 从电子健康记录（EHRs）中提取相关数据对于识别症状和自动化流行病学监测过程至关重要。通过利用EHRs中大量未结构化的文本内容，我们可以检测出疾病爆发的模式，从而实现更快、更精准的公共卫生响应。我们提出的框架提供了一种灵活且高效的解决方案，用于从未结构化文本中挖掘数据，大大减少了对专家进行大量人工标注的需求。实验显示，即使对于复杂的分类问题，我们的框架也能在仅需200个手动标注文本的情况下表现出强大的性能。此外，我们的方法可以使用简单的轻量级模型运行，与资源密集型的深度学习模型相比，也能达到具有竞争力的甚至更好的结果。这种能力不仅能加速处理时间，还能保护患者隐私，因为数据可以在现场较弱的硬件上处理，而不是传输到外部系统。因此，我们的方法提供了一种实用、可扩展且隐私保护的实时流行病学监测方法，使医疗机构能够迅速有效地应对新兴的健康威胁。 

---
# DreamDPO: Aligning Text-to-3D Generation with Human Preferences via Direct Preference Optimization 

**Title (ZH)**: DreamDPO：通过直接偏好优化实现文本到3D生成与人类偏好的对齐 

**Authors**: Zhenglin Zhou, Xiaobo Xia, Fan Ma, Hehe Fan, Yi Yang, Tat-Seng Chua  

**Link**: [PDF](https://arxiv.org/pdf/2502.04370)  

**Abstract**: Text-to-3D generation automates 3D content creation from textual descriptions, which offers transformative potential across various fields. However, existing methods often struggle to align generated content with human preferences, limiting their applicability and flexibility. To address these limitations, in this paper, we propose DreamDPO, an optimization-based framework that integrates human preferences into the 3D generation process, through direct preference optimization. Practically, DreamDPO first constructs pairwise examples, then compare their alignment with human preferences using reward or large multimodal models, and lastly optimizes the 3D representation with a preference-driven loss function. By leveraging pairwise comparison to reflect preferences, DreamDPO reduces reliance on precise pointwise quality evaluations while enabling fine-grained controllability through preference-guided optimization. Experiments demonstrate that DreamDPO achieves competitive results, and provides higher-quality and more controllable 3D content compared to existing methods. The code and models will be open-sourced. 

**Abstract (ZH)**: 本文将下面的内容或标题翻译成了中文，并确保符合学术规范：

文本到3D生成技术可以自动从文本描述中创建3D内容，这一技术在多个领域具有变革性的潜力。然而，现有方法往往难以使生成的内容与人类偏好相匹配，限制了其应用范围和灵活性。为解决这些问题，本文提出了一种名为DreamDPO的优化框架，该框架通过直接优化偏好将人类偏好融入3D生成过程。具体而言，DreamDPO首先构建一对一组例，然后使用奖励或大型多模态模型比较这些组例与人类偏好的对齐情况，并最终使用偏好驱动的损失函数优化3D表示。通过利用一对一组例来反映偏好，DreamDPO降低了对精确点状质量评估的依赖，同时通过偏好引导的优化实现精细粒度的可控性。实验结果表明，DreamDPO在性能上具有竞争力，并能生成更高质量且更可控的3D内容。该代码和模型将开源。 

---
# Contrastive Token-level Explanations for Graph-based Rumour Detection 

**Title (ZH)**: 基于图的谣言检测的对比性令牌级解释 

**Authors**: Daniel Wai Kit Chin, Roy Ka-Wei Lee  

**Link**: [PDF](https://arxiv.org/pdf/2502.04366)  

**Abstract**: The widespread use of social media has accelerated the dissemination of information, but it has also facilitated the spread of harmful rumours, which can disrupt economies, influence political outcomes, and exacerbate public health crises, such as the COVID-19 pandemic. While Graph Neural Network (GNN)-based approaches have shown significant promise in automated rumour detection, they often lack transparency, making their predictions difficult to interpret. Existing graph explainability techniques fall short in addressing the unique challenges posed by the dependencies among feature dimensions in high-dimensional text embeddings used in GNN-based models. In this paper, we introduce Contrastive Token Layerwise Relevance Propagation (CT-LRP), a novel framework designed to enhance the explainability of GNN-based rumour detection. CT-LRP extends current graph explainability methods by providing token-level explanations that offer greater granularity and interpretability. We evaluate the effectiveness of CT-LRP across multiple GNN models trained on three publicly available rumour detection datasets, demonstrating that it consistently produces high-fidelity, meaningful explanations, paving the way for more robust and trustworthy rumour detection systems. 

**Abstract (ZH)**: 社交媒体的广泛应用加速了信息的传播，但也促进了有害谣言的扩散，这些谣言可能扰乱经济、影响政治结果，并加剧公共卫生危机，如COVID-19大流行。虽然基于图神经网络（GNN）的方法在自动化谣言检测方面表现出显著的潜力，但它们通常缺乏透明度，使预测难以解释。现有的图解释性技术在处理高维文本嵌入中特征维度之间的依赖性带来的独特挑战方面存在不足。在本文中，我们介绍了对比标记层相关性传播（CT-LRP）这一新的框架，旨在增强基于GNN的谣言检测的可解释性。CT-LRP通过提供标记级别的解释来扩展现有图解释性方法，这些解释具有更高的粒度和可解释性。我们在三个公开可用的谣言检测数据集上对多种GNN模型进行了有效性评估，证明CT-LRP能够一致地产生高保真、有意义的解释，为更稳健和可信的谣言检测系统铺平了道路。 

---
# LLMs can be easily Confused by Instructional Distractions 

**Title (ZH)**: 大型语言模型可能会被指令性干扰所混淆 

**Authors**: Yerin Hwang, Yongil Kim, Jahyun Koo, Taegwan Kang, Hyunkyung Bae, Kyomin Jung  

**Link**: [PDF](https://arxiv.org/pdf/2502.04362)  

**Abstract**: Despite the fact that large language models (LLMs) show exceptional skill in instruction following tasks, this strength can turn into a vulnerability when the models are required to disregard certain instructions. Instruction-following tasks typically involve a clear task description and input text containing the target data to be processed. However, when the input itself resembles an instruction, confusion may arise, even if there is explicit prompting to distinguish between the task instruction and the input. We refer to this phenomenon as instructional distraction. In this paper, we introduce a novel benchmark, named DIM-Bench, specifically designed to assess LLMs' performance under instructional distraction. The benchmark categorizes real-world instances of instructional distraction and evaluates LLMs across four instruction tasks: rewriting, proofreading, translation, and style transfer -- alongside five input tasks: reasoning, code generation, mathematical reasoning, bias detection, and question answering. Our experimental results reveal that even the most advanced LLMs are susceptible to instructional distraction, often failing to accurately follow user intent in such cases. 

**Abstract (ZH)**: 尽管大型语言模型（LLMs）在指令遵循任务中表现出非凡的能力，但在需要模型忽略某些指令时，这种优势可能会转化为弱点。指令遵循任务通常包含明确的任务描述和包含目标数据的输入文本。然而，当输入本身类似于指令时，可能会引起混淆，即使有明确的提示来区分任务指令和输入。我们将这种现象称为指令性干扰。在本文中，我们引入了一个新的基准测试，名为DIM-Bench，旨在评估LLMs在指令性干扰下的性能。该基准测试对现实世界中的指令性干扰实例进行了分类，并评估了LLMs在重写、校对、翻译和风格转换等四种指令任务以及推理、代码生成、数学推理、偏见检测和问答等五种输入任务下的表现。实验结果表明，即使是最先进的LLMs也容易受到指令性干扰的影响，往往在这些情况下无法准确遵循用户意图。 

---
# MARAGE: Transferable Multi-Model Adversarial Attack for Retrieval-Augmented Generation Data Extraction 

**Title (ZH)**: MARAGE：适用于检索增强生成数据提取的可迁移多模型对抗攻击方法 

**Authors**: Xiao Hu, Eric Liu, Weizhou Wang, Xiangyu Guo, David Lie  

**Link**: [PDF](https://arxiv.org/pdf/2502.04360)  

**Abstract**: Retrieval-Augmented Generation (RAG) offers a solution to mitigate hallucinations in Large Language Models (LLMs) by grounding their outputs to knowledge retrieved from external sources. The use of private resources and data in constructing these external data stores can expose them to risks of extraction attacks, in which attackers attempt to steal data from these private databases. Existing RAG extraction attacks often rely on manually crafted prompts, which limit their effectiveness. In this paper, we introduce a framework called MARAGE for optimizing an adversarial string that, when appended to user queries submitted to a target RAG system, causes outputs containing the retrieved RAG data verbatim. MARAGE leverages a continuous optimization scheme that integrates gradients from multiple models with different architectures simultaneously to enhance the transferability of the optimized string to unseen models. Additionally, we propose a strategy that emphasizes the initial tokens in the target RAG data, further improving the attack's generalizability. Evaluations show that MARAGE consistently outperforms both manual and optimization-based baselines across multiple LLMs and RAG datasets, while maintaining robust transferability to previously unseen models. Moreover, we conduct probing tasks to shed light on the reasons why MARAGE is more effective compared to the baselines and to analyze the impact of our approach on the model's internal state. 

**Abstract (ZH)**: 检索增强生成（RAG）通过将大型语言模型（LLMs）的输出与外部来源检索的知识进行结合，提供了一种减轻幻觉的方法。在构建这些外部数据存储时使用私有资源和数据可能会暴露它们受到提取攻击的风险，在这类攻击中，攻击者试图从这些私有数据库中窃取数据。现有的RAG提取攻击通常依赖于手动构建的提示，限制了它们的效果。本文提出了一种名为MARAGE的框架，用于优化一个敌对字符串，当该字符串附加到提交给目标RAG系统的用户查询时，会导致包含摘要检索RAG数据的输出。MARAGE利用了一种连续的优化方案，该方案同时整合了具有不同架构的多个模型的梯度，以增强优化字符串在未见过的模型中的转移性。此外，我们提出了一种策略，强调目标RAG数据中的初始令牌，进一步提高攻击的通用性。评估结果显示，MARAGE在多个LLM和RAG数据集中的一致地优于手动构建的和基于优化的基准，并且在之前未见过的模型上仍然具有稳健的转移性。此外，我们还进行了探究任务，以揭示MARAGE相较于基准模型更有效的原因，并分析了我们方法对模型内部状态的影响。 

---
# Exploring Spatial Language Grounding Through Referring Expressions 

**Title (ZH)**: 通过指代表达探索空间语言接地 

**Authors**: Akshar Tumu, Parisa Kordjamshidi  

**Link**: [PDF](https://arxiv.org/pdf/2502.04359)  

**Abstract**: Spatial Reasoning is an important component of human cognition and is an area in which the latest Vision-language models (VLMs) show signs of difficulty. The current analysis works use image captioning tasks and visual question answering. In this work, we propose using the Referring Expression Comprehension task instead as a platform for the evaluation of spatial reasoning by VLMs. This platform provides the opportunity for a deeper analysis of spatial comprehension and grounding abilities when there is 1) ambiguity in object detection, 2) complex spatial expressions with a longer sentence structure and multiple spatial relations, and 3) expressions with negation ('not'). In our analysis, we use task-specific architectures as well as large VLMs and highlight their strengths and weaknesses in dealing with these specific situations. While all these models face challenges with the task at hand, the relative behaviors depend on the underlying models and the specific categories of spatial semantics (topological, directional, proximal, etc.). Our results highlight these challenges and behaviors and provide insight into research gaps and future directions. 

**Abstract (ZH)**: 空间推理是人类认知的重要组成部分，也是最新视觉-语言模型（VLMs）表现出困难的领域之一。目前的研究主要使用图像生成任务和视觉问答来评估。在本工作中，我们提出使用指示表达理解任务作为评估VLMs空间推理能力的平台。该平台提供了在以下方面进行更深入分析的机会：1）物体检测的模棱两可性；2）复杂的空间表达，具有较长的句子结构和多种空间关系；3）带有否定词（"not"）的表达。在我们的分析中，我们使用任务特定的架构以及大型VLMs，并指出它们在处理这些特定情况时的优缺点。虽然所有这些模型在手头的任务上都面临挑战，但其相对行为取决于底层模型和特定的空间语义类别（拓扑的、方向的、邻近的等）。我们的结果突出了这些挑战和行为，并为研究空白和未来方向提供了见解。 

---
# Position: Scaling LLM Agents Requires Asymptotic Analysis with LLM Primitives 

**Title (ZH)**: 位置：扩展大语言模型代理需要使用大语言模型原语进行渐近分析 

**Authors**: Elliot Meyerson, Xin Qiu  

**Link**: [PDF](https://arxiv.org/pdf/2502.04358)  

**Abstract**: Decomposing hard problems into subproblems often makes them easier and more efficient to solve. With large language models (LLMs) crossing critical reliability thresholds for a growing slate of capabilities, there is an increasing effort to decompose systems into sets of LLM-based agents, each of whom can be delegated sub-tasks. However, this decomposition (even when automated) is often intuitive, e.g., based on how a human might assign roles to members of a human team. How close are these role decompositions to optimal? This position paper argues that asymptotic analysis with LLM primitives is needed to reason about the efficiency of such decomposed systems, and that insights from such analysis will unlock opportunities for scaling them. By treating the LLM forward pass as the atomic unit of computational cost, one can separate out the (often opaque) inner workings of a particular LLM from the inherent efficiency of how a set of LLMs are orchestrated to solve hard problems. In other words, if we want to scale the deployment of LLMs to the limit, instead of anthropomorphizing LLMs, asymptotic analysis with LLM primitives should be used to reason about and develop more powerful decompositions of large problems into LLM agents. 

**Abstract (ZH)**: 将复杂问题分解为子问题通常会使问题更容易解决且效率更高。随着大型语言模型（LLMs）在越来越多的功能上达到关键的可靠门槛，人们越来越努力将系统分解为基于LLM的代理组，每个代理可以被委派子任务。然而，即使是自动化的这种分解通常是直观的，例如，基于人类如何为人类团队成员分配角色。这种角色分配与最优情况有多接近呢？本文观点认为，为了分析这种分解系统的效率，需要使用LLM的基本原理进行渐进分析。这样的分析洞察将有助于扩大这些系统的规模。

通过将LLM前向传播视为计算成本的原子单元，可以将特定LLM的（通常看不见的）内部工作与一组LLM如何协同解决复杂问题的内在效率区分开来。换句话说，如果我们希望将LLM部署到极限，而不是赋予LLM人性化的特性，那么应该使用基于LLM原理的渐进分析来理解和开发将大问题分解为LLM代理的更强大方法。 

---
# Reusing Embeddings: Reproducible Reward Model Research in Large Language Model Alignment without GPUs 

**Title (ZH)**: 重新利用嵌入表示：在大型语言模型对齐中不使用GPU进行可重现的奖励模型研究 

**Authors**: Hao Sun, Yunyi Shen, Jean-Francois Ton, Mihaela van der Schaar  

**Link**: [PDF](https://arxiv.org/pdf/2502.04357)  

**Abstract**: Large Language Models (LLMs) have made substantial strides in structured tasks through Reinforcement Learning (RL), demonstrating proficiency in mathematical reasoning and code generation. However, applying RL in broader domains like chatbots and content generation -- through the process known as Reinforcement Learning from Human Feedback (RLHF) -- presents unique challenges. Reward models in RLHF are critical, acting as proxies that evaluate the alignment of LLM outputs with human intent. Despite advancements, the development of reward models is hindered by challenges such as computational heavy training, costly evaluation, and therefore poor reproducibility. We advocate for using embedding-based input in reward model research as an accelerated solution to those challenges. By leveraging embeddings for reward modeling, we can enhance reproducibility, reduce computational demands on hardware, improve training stability, and significantly reduce training and evaluation costs, hence facilitating fair and efficient comparisons in this active research area. We then show a case study of reproducing existing reward model ensemble research using embedding-based reward models. We discussed future avenues for research, aiming to contribute to safer and more effective LLM deployments. 

**Abstract (ZH)**: 大型语言模型（LLMs）通过强化学习（RL）在结构化任务中取得了显著进展，展示了在数学推理和代码生成方面的能力。然而，在更具广泛性的领域，如聊天机器人和内容生成中应用RL（通过人类反馈的强化学习，简称RLHF）则面临着独特的挑战。在RLHF中，奖励模型至关重要，它们作为代理模型，评估LLM输出与人类意图的一致性。尽管取得了进展，但奖励模型的发展仍受制于计算量大的训练、昂贵的评估以及因此而来的低可重复性等挑战。我们建议在奖励模型研究中采用基于嵌入的输入作为加速解决这些挑战的方案。通过利用嵌入进行奖励建模，我们可以增强可重复性、减少对硬件的计算需求、改善训练稳定性，并显著降低训练和评估成本，从而促进该活跃研究领域的公平和高效比较。我们随后展示了使用基于嵌入的奖励模型再现已有奖励模型集成研究的案例研究。我们还讨论了未来的研究方向，旨在为更安全和更有效的LLM部署作出贡献。 

---
# Open Foundation Models in Healthcare: Challenges, Paradoxes, and Opportunities with GenAI Driven Personalized Prescription 

**Title (ZH)**: 基于医疗领域的开放基础模型：由GenAI驱动的个性化处方面临的挑战、悖论与机遇 

**Authors**: Mahdi Alkaeed, Sofiat Abioye, Adnan Qayyum, Yosra Magdi Mekki, Ilhem Berrou, Mohamad Abdallah, Ala Al-Fuqaha, Muhammad Bilal, Junaid Qadir  

**Link**: [PDF](https://arxiv.org/pdf/2502.04356)  

**Abstract**: In response to the success of proprietary Large Language Models (LLMs) such as OpenAI's GPT-4, there is a growing interest in developing open, non-proprietary LLMs and AI foundation models (AIFMs) for transparent use in academic, scientific, and non-commercial applications. Despite their inability to match the refined functionalities of their proprietary counterparts, open models hold immense potential to revolutionize healthcare applications. In this paper, we examine the prospects of open-source LLMs and AIFMs for developing healthcare applications and make two key contributions. Firstly, we present a comprehensive survey of the current state-of-the-art open-source healthcare LLMs and AIFMs and introduce a taxonomy of these open AIFMs, categorizing their utility across various healthcare tasks. Secondly, to evaluate the general-purpose applications of open LLMs in healthcare, we present a case study on personalized prescriptions. This task is particularly significant due to its critical role in delivering tailored, patient-specific medications that can greatly improve treatment outcomes. In addition, we compare the performance of open-source models with proprietary models in settings with and without Retrieval-Augmented Generation (RAG). Our findings suggest that, although less refined, open LLMs can achieve performance comparable to proprietary models when paired with grounding techniques such as RAG. Furthermore, to highlight the clinical significance of LLMs-empowered personalized prescriptions, we perform subjective assessment through an expert clinician. We also elaborate on ethical considerations and potential risks associated with the misuse of powerful LLMs and AIFMs, highlighting the need for a cautious and responsible implementation in healthcare. 

**Abstract (ZH)**: 针对OpenAI的GPT-4等私有大型语言模型（LLMs）的成功，人们对开发透明应用于学术、科学和非商业领域的开放型而非私有LLMs和人工智能基础模型（AIFMs）的兴趣日益增长。尽管开放型模型的精细功能可能无法匹配私有模型，但它们在医疗健康应用方面具有巨大的潜力。本文探讨了开放源代码LLMs和AIFMs在医疗健康应用中开发的可能性，并做出了两项关键贡献。首先，我们对当前最先进的开放源代码医疗健康LLMs和AIFMs进行了全面综述，并介绍了对这些开放源代码AIFMs的分类，按其在不同医疗健康任务中的用途进行分类。其次，为了评估开放源代码LLMs在医疗健康中的通用性应用，我们进行了个性处方的案例研究。这一任务尤为重要，因为它在提供个性化、患者特定的药物方面发挥着关键作用，可以显著提高治疗效果。此外，我们在有和没有检索增强生成（RAG）的情况下，对比了开放源代码模型与私有模型的性能表现。我们的研究发现，在结合诸如RAG等扎根技术时，开放源代码模型的性能可以与私有模型相当。此外，我们通过资深临床医生的主观评估，凸显了LLMs赋能个性化处方的临床意义，并指出了使用强大LLMs和AIFMs可能带来的伦理考虑和潜在风险，强调了在医疗健康领域谨慎和负责任地实施的重要性。 

---
# LLM-ProS: Analyzing Large Language Models' Performance in Competitive Problem Solving 

**Title (ZH)**: LLM-ProS: 分析大语言模型在竞争性问题解决中的表现 

**Authors**: Md Sifat Hossain, Anika Tabassum, Md. Fahim Arefin, Tarannum Shaila Zaman  

**Link**: [PDF](https://arxiv.org/pdf/2502.04355)  

**Abstract**: The rapid advancement of large language models has opened new avenues for automating complex problem-solving tasks such as algorithmic coding and competitive programming. This paper introduces a novel evaluation technique, LLM-ProS, to assess the performance of state-of-the-art LLMs on International Collegiate Programming Contest (ICPC) problems. Using a curated dataset of 166 World Finals problems from 2011 to 2024, we benchmark the models' reasoning, accuracy, and efficiency. We evaluate the five models-GPT-4o, Mistral Large, Llama-3.1-405B, and the o1 family, consisting of o1-mini and o1-preview, across critical metrics like correctness, resource utilization, and response calibration. Our results reveal significant differences in the models' abilities to generalize, adapt, and solve novel problems. We also investigated the impact of training methodologies, dataset contamination, and chain-of-thought reasoning on model performance. The findings provide new insights into optimizing LLMs for algorithmic tasks, highlighting both strengths and limitations of current models. 

**Abstract (ZH)**: 大规模语言模型的迅速发展为自动化复杂问题解决任务（如算法编码和编程竞赛）开辟了新的途径。本文介绍了一种新的评估技术——LLM-ProS，用于评估当前最先进的大型语言模型在国际大学生程序设计竞赛（ICPC）问题上的性能。我们使用从2011年到2024年之间的166个世界总决赛问题构建了一个精选数据集，评估这些模型的推理能力、准确性和效率。我们对五种模型——GPT-4o、Mistral Large、Llama-3.1-405B以及o1家系（包括o1-mini和o1-preview）进行了评估，重点考察了它们在正确性、资源利用和响应校准等方面的关键指标。我们的结果揭示了这些模型在泛化、适应和解决新问题方面的显著差异。我们还探讨了训练方法、数据集污染和思维链推理对模型性能的影响。这些发现提供了优化大型语言模型用于算法任务的新见解，同时也指出了当前模型的优势和局限性。 

---
# Reviving The Classics: Active Reward Modeling in Large Language Model Alignment 

**Title (ZH)**: 重振经典：在大型语言模型对齐中应用主动奖励建模 

**Authors**: Yunyi Shen, Hao Sun, Jean-François Ton  

**Link**: [PDF](https://arxiv.org/pdf/2502.04354)  

**Abstract**: Building neural reward models from human preferences is a pivotal component in reinforcement learning from human feedback (RLHF) and large language model alignment research. Given the scarcity and high cost of human annotation, how to select the most informative pairs to annotate is an essential yet challenging open problem. In this work, we highlight the insight that an ideal comparison dataset for reward modeling should balance exploration of the representation space and make informative comparisons between pairs with moderate reward differences. Technically, challenges arise in quantifying the two objectives and efficiently prioritizing the comparisons to be annotated. To address this, we propose the Fisher information-based selection strategies, adapt theories from the classical experimental design literature, and apply them to the final linear layer of the deep neural network-based reward modeling tasks. Empirically, our method demonstrates remarkable performance, high computational efficiency, and stability compared to other selection methods from deep learning and classical statistical literature across multiple open-source LLMs and datasets. Further ablation studies reveal that incorporating cross-prompt comparisons in active reward modeling significantly enhances labeling efficiency, shedding light on the potential for improved annotation strategies in RLHF. 

**Abstract (ZH)**: 从人类偏好中构建神经奖励模型是强化学习从人类反馈（RLHF）和大型语言模型对齐研究中的一个关键组成部分。由于人类标注资源稀缺且成本高昂，如何选择最具信息量的样本对进行标注成为了一个重要而具有挑战性的开放问题。在本文中，我们强调了一个理想的奖励模型数据集应该在探索表示空间的同时，通过对比具有中等奖励差异的样本对来提供信息丰富的比较。技术上，量化这两个目标并高效地优先处理需要标注的比较这对矛盾是一个挑战。为此，我们提出了费舍尔信息基础上的选取策略，借鉴经典实验设计文献中的理论，并将这些理论应用于基于深度神经网络的奖励模型任务的最终线性层。实验结果表明，与来自深度学习和经典统计文献的其他选取方法相比，我们的方法在多个开源大型语言模型和数据集上展现了卓越的表现、高效性和稳定性。进一步的消融研究显示，在主动奖励模型中引入跨提示比较显著提高了标注效率，为RLHF中的改进标注策略提供了新的视角。 

---
# CognArtive: Large Language Models for Automating Art Analysis and Decoding Aesthetic Elements 

**Title (ZH)**: CognArtive：大型语言模型在自动化艺术分析与审美元素解码中的应用 

**Authors**: Afshin Khadangi, Amir Sartipi, Igor Tchappi, Gilbert Fridgen  

**Link**: [PDF](https://arxiv.org/pdf/2502.04353)  

**Abstract**: Art, as a universal language, can be interpreted in diverse ways, with artworks embodying profound meanings and nuances. The advent of Large Language Models (LLMs) and the availability of Multimodal Large Language Models (MLLMs) raise the question of how these transformative models can be used to assess and interpret the artistic elements of artworks. While research has been conducted in this domain, to the best of our knowledge, a deep and detailed understanding of the technical and expressive features of artworks using LLMs has not been explored. In this study, we investigate the automation of a formal art analysis framework to analyze a high-throughput number of artworks rapidly and examine how their patterns evolve over time. We explore how LLMs can decode artistic expressions, visual elements, composition, and techniques, revealing emerging patterns that develop across periods. Finally, we discuss the strengths and limitations of LLMs in this context, emphasizing their ability to process vast quantities of art-related data and generate insightful interpretations. Due to the exhaustive and granular nature of the results, we have developed interactive data visualizations, available online this https URL, to enhance understanding and accessibility. 

**Abstract (ZH)**: 艺术作为一种通用的语言，可以被多种方式解读，艺术品蕴含着深刻的意义和细微差别。大型语言模型（LLMs）和多模态大型语言模型（MLLMs）的出现引发了这样一个问题：这些变革性的模型如何能够用于评估和解释艺术品的艺术元素。尽管在该领域已经进行了一些研究，但我们所知的是，使用LLMs对艺术品的技术和表现特征进行深入细致的理解还尚未得到充分探索。在这项研究中，我们探讨了自动化形式艺术分析框架的可能性，以快速分析大量艺术品，并研究它们随时间演化的模式。我们探索了LLMs如何解码艺术表达、视觉元素、构图和技艺，揭示了随着时间推移而演变的新兴模式。最后，我们讨论了在这一背景下LLMs的优势与局限性，强调了它们处理大量与艺术相关数据并生成有洞察力的解释的能力。鉴于结果的详尽性和细致性，我们开发了交互式数据可视化工具，并在线提供（此链接请参阅原文），以增强理解和访问性。 

---
# Investigating the Robustness of Deductive Reasoning with Large Language Models 

**Title (ZH)**: 使用大规模语言模型探究演绎推理的鲁棒性 

**Authors**: Fabian Hoppe, Filip Ilievski, Jan-Christoph Kalo  

**Link**: [PDF](https://arxiv.org/pdf/2502.04352)  

**Abstract**: Large Language Models (LLMs) have been shown to achieve impressive results for many reasoning-based Natural Language Processing (NLP) tasks, suggesting a degree of deductive reasoning capability. However, it remains unclear to which extent LLMs, in both informal and autoformalisation methods, are robust on logical deduction tasks. Moreover, while many LLM-based deduction methods have been proposed, there is a lack of a systematic study that analyses the impact of their design components. Addressing these two challenges, we propose the first study of the robustness of LLM-based deductive reasoning methods. We devise a framework with two families of perturbations: adversarial noise and counterfactual statements, which jointly generate seven perturbed datasets. We organize the landscape of LLM reasoners according to their reasoning format, formalisation syntax, and feedback for error recovery. The results show that adversarial noise affects autoformalisation, while counterfactual statements influence all approaches. Detailed feedback does not improve overall accuracy despite reducing syntax errors, pointing to the challenge of LLM-based methods to self-correct effectively. 

**Abstract (ZH)**: 大型语言模型（LLMs）在许多基于推理的自然语言处理（NLP）任务中已显示出令人印象深刻的成果，这表明它们具备一定的演绎推理能力。然而，尚不清楚LLMs在非正式和自形式化方法中在逻辑推理任务上的鲁棒性如何。此外，虽然已经提出了许多基于LLM的推理方法，但却缺乏系统性的研究来分析其设计元素的影响。为了解决这两个挑战，我们提出了首个关于基于LLM的演绎推理方法鲁棒性的研究。我们提出了一种框架，包含两类扰动：对抗噪声和反事实陈述，这两种扰动联合生成了七个扰动数据集。根据推理格式、形式化语法和错误恢复反馈来组织LLM推理器的全景图。结果显示，对抗噪声影响自形式化，而反事实陈述影响所有方法。详细的反馈并未提高总体准确性，尽管减少了语法错误，这表明基于LLM的方法在自我纠正方面存在挑战。 

---
# NER4all or Context is All You Need: Using LLMs for low-effort, high-performance NER on historical texts. A humanities informed approach 

**Title (ZH)**: NER4all 或者上下文即无所不包：利用大规模语言模型在历史文本中实现低消耗、高性能的命名实体识别。一种基于人文学科的方法 

**Authors**: Torsten Hiltmann, Martin Dröge, Nicole Dresselhaus, Till Grallert, Melanie Althage, Paul Bayer, Sophie Eckenstaler, Koray Mendi, Jascha Marijn Schmitz, Philipp Schneider, Wiebke Sczeponik, Anica Skibba  

**Link**: [PDF](https://arxiv.org/pdf/2502.04351)  

**Abstract**: Named entity recognition (NER) is a core task for historical research in automatically establishing all references to people, places, events and the like. Yet, do to the high linguistic and genre diversity of sources, only limited canonisation of spellings, the level of required historical domain knowledge, and the scarcity of annotated training data, established approaches to natural language processing (NLP) have been both extremely expensive and yielded only unsatisfactory results in terms of recall and precision. Our paper introduces a new approach. We demonstrate how readily-available, state-of-the-art LLMs significantly outperform two leading NLP frameworks, spaCy and flair, for NER in historical documents by seven to twentytwo percent higher F1-Scores. Our ablation study shows how providing historical context to the task and a bit of persona modelling that turns focus away from a purely linguistic approach are core to a successful prompting strategy. We also demonstrate that, contrary to our expectations, providing increasing numbers of examples in few-shot approaches does not improve recall or precision below a threshold of 16-shot. In consequence, our approach democratises access to NER for all historians by removing the barrier of scripting languages and computational skills required for established NLP tools and instead leveraging natural language prompts and consumer-grade tools and frontends. 

**Abstract (ZH)**: 命名实体识别（NER）是历史研究中的一个核心任务，旨在自动建立所有对人物、地点、事件等的引用关系。然而，由于来源的高语言多样性、体裁多样性，以及名称拼写缺乏统一标准、所需的历史专业知识水平较高、标注训练数据稀缺，现有的自然语言处理（NLP）方法在成本上非常昂贵，并且在召回率和精度方面表现不佳。我们论文提出了一种新的方法。我们展示了如何利用现成的先进大规模语言模型（LLM）显著优于领先的传统NLP框架spaCy和flair在历史文献中的NER任务，其F1分数提高了7%至22%。我们的消融研究显示，为任务提供历史上下文并进行一些角色建模，以减少纯粹的语言导向方法，是成功提示策略的关键。此外，我们还展示了与预期相反的事实，在少量示例（few-shot）方法中提供越多示例，召回率和精度在达到16-shot阈值后不会得到改善。因此，我们的方法通过消除传统NLP工具所需的脚本语言和计算技能障碍，使所有历史学家都能够平等地访问NER技术，并利用自然语言提示和消费级工具及前端替代之。 

---
# CodeSteer: Symbolic-Augmented Language Models via Code/Text Guidance 

**Title (ZH)**: CodeSteer: 通过代码/文本指导的符号增强语言模型 

**Authors**: Yongchao Chen, Yilun Hao, Yueying Liu, Yang Zhang, Chuchu Fan  

**Link**: [PDF](https://arxiv.org/pdf/2502.04350)  

**Abstract**: Existing methods fail to effectively steer Large Language Models (LLMs) between textual reasoning and code generation, leaving symbolic computing capabilities underutilized. We introduce CodeSteer, an effective method for guiding LLM code/text generation. We construct a comprehensive benchmark SymBench comprising 37 symbolic tasks with adjustable complexity and also synthesize datasets of 12k multi-round guidance/generation trajectories and 5.5k guidance comparison pairs. We fine-tune the Llama-3-8B model with a newly designed multi-round supervised fine-tuning (SFT) and direct preference optimization (DPO). The resulting model, CodeSteerLLM, augmented with the proposed symbolic and self-answer checkers, effectively guides the code/text generation of larger models. Augmenting GPT-4o with CodeSteer raises its average performance score from 53.3 to 86.4, even outperforming the existing best LLM OpenAI o1 (82.7), o1-preview (74.8), and DeepSeek R1 (76.8) across all 37 tasks (28 seen, 9 unseen). Trained for GPT-4o, CodeSteer demonstrates superior generalizability, providing an average 41.8 performance boost on Claude, Mistral, and GPT-3.5. CodeSteer-guided LLMs fully harness symbolic computing to maintain strong performance on highly complex tasks. Models, Datasets, and Codes are available at this https URL. 

**Abstract (ZH)**: 现有的方法未能有效地在文本推理和代码生成之间引导大型语言模型（LLMs），导致符号计算能力未得到充分利用。我们引入了CodeSteer，这是一种有效的方法，用于引导LLM的代码/文本生成。我们构建了包含37个符号任务的全面基准SymBench，这些任务具有可调的复杂度，并且合成了12,000个多轮指导/生成轨迹的数据集和5,500对指导对比样本。我们使用新设计的多轮监督微调（SFT）和直接偏好优化（DPO）对Llama-3-8B模型进行了微调。由此产生的模型CodeSteerLLM，在提议的符号检查器和自我答案检查器的辅助下，有效地引导了更大模型的代码/文本生成。将CodeSteer与GPT-4o结合使用，其平均性能得分从53.3提高到86.4，甚至在所有37个任务（28个已见过，9个未见过）中超过了现有的最佳模型OpenAI o1（82.7）、o1-preview（74.8）和DeepSeek R1（76.8）。针对GPT-4o训练的CodeSteer展示了更优秀的泛化能力，在Claude、Mistral和GPT-3.5上的平均性能提升幅度分别为41.8。CodeSteer引导的LLM充分利用了符号计算能力，能在极其复杂的任务中保持强劲的性能。相关模型、数据集和代码可在以下网址获取：this https URL。 

---
# Dynamic benchmarking framework for LLM-based conversational data capture 

**Title (ZH)**: 基于LLM的对话数据采集动态基准框架 

**Authors**: Pietro Alessandro Aluffi, Patrick Zietkiewicz, Marya Bazzi, Matt Arderne, Vladimirs Murevics  

**Link**: [PDF](https://arxiv.org/pdf/2502.04349)  

**Abstract**: The rapid evolution of large language models (LLMs) has transformed conversational agents, enabling complex human-machine interactions. However, evaluation frameworks often focus on single tasks, failing to capture the dynamic nature of multi-turn dialogues. This paper introduces a dynamic benchmarking framework to assess LLM-based conversational agents through interactions with synthetic users. The framework integrates generative agent simulation to evaluate performance on key dimensions: information extraction, context awareness, and adaptive engagement. By simulating various aspects of user behavior, our work provides a scalable, automated, and flexible benchmarking approach. Experimental evaluation - within a loan application use case - demonstrates the framework's effectiveness under one-shot and few-shot extraction conditions. Results show that adaptive strategies improve data extraction accuracy, especially when handling ambiguous responses. Future work will extend its applicability to broader domains and incorporate additional metrics (e.g., conversational coherence, user engagement). This study contributes a structured, scalable approach to evaluating LLM-based conversational agents, facilitating real-world deployment. 

**Abstract (ZH)**: 大型语言模型（LLMs）的快速演化已经改变了对话代理的面貌，使其能够实现复杂的机器与人类交互。然而，现有的评估框架往往侧重于单一任务，不能充分捕捉多轮对话的动态性质。本文提出了一种动态基准评估框架，通过与合成用户交互来评估基于LLM的对话代理。该框架集成了生成性代理模拟，以评估其在关键维度上的表现：信息提取、上下文意识以及适应性互动。通过模拟用户行为的不同方面，我们的工作提供了一种可扩展、自动化和灵活的基准评估方法。在具体贷款申请案例中的实验评估显示，该框架在一次性提取和少量样本提取条件下均表现出有效性。结果显示，适应性策略提高了数据提取的准确性，特别是在处理模糊响应时效果尤为明显。未来的研究将扩展其适用范围至更广泛的领域，并引入更多评估指标（如对话一致性、用户参与度）。本研究提供了一种结构化且可扩展的方法来评估基于LLM的对话代理，为其实现真实世界的部署奠定了基础。 

---
# Prompt-based Depth Pruning of Large Language Models 

**Title (ZH)**: 基于提示的大型语言模型深度裁剪 

**Authors**: Juyun Wee, Minjae Park, Jaeho Lee  

**Link**: [PDF](https://arxiv.org/pdf/2502.04348)  

**Abstract**: Depth pruning aims to reduce the inference cost of a large language model without any hardware-specific complications, by simply removing several less important transformer blocks. However, our empirical findings suggest that the importance of a transformer block may be highly task-dependent -- a block that is crucial for a task can be removed without degrading the accuracy on another task. Based on this observation, we develop a dynamic depth pruning algorithm, coined PuDDing (Prompt-routed Dynamic Depth Pruning), which determines which blocks to omit from the model based on the input prompt. PuDDing operates by training a lightweight router to predict the best omission set among a set of options, where this option set has also been constructed in a data-driven manner. Empirical results on commonsense reasoning benchmarks demonstrate that PuDDing effectively accelerates the inference language models, and achieves better on-task performance than static depth pruning baselines. 

**Abstract (ZH)**: 深度修剪旨在通过简单地移除几个不太重要的Transformer块来减少大型语言模型的推理成本，而无需任何特定硬件的复杂性。然而，我们的实证研究表明， Transformer块的重要性可能高度依赖于具体任务——对于某个任务至关重要的块在另一个任务上移除后不会显著降低准确度。基于这一观察，我们开发了一种动态深度修剪算法，称为PuDDing（Prompt-routed Dynamic Depth Pruning），该算法根据输入提示确定从模型中省略哪些块。PuDDing通过训练一个轻量级路由器来预测最佳省略集合，在可选方案集中进行数据驱动的构建。在常识推理基准上的实证结果表明，PuDDing可以有效加速推理语言模型，并在任务相关性能上优于静态深度修剪基准。 

---
# SCALM: Detecting Bad Practices in Smart Contracts Through LLMs 

**Title (ZH)**: SCALM：通过大型语言模型检测智能合约中的不良实践 

**Authors**: Zongwei Li, Xiaoqi Li, Wenkai Li, Xin Wang  

**Link**: [PDF](https://arxiv.org/pdf/2502.04347)  

**Abstract**: As the Ethereum platform continues to mature and gain widespread usage, it is crucial to maintain high standards of smart contract writing practices. While bad practices in smart contracts may not directly lead to security issues, they do elevate the risk of encountering problems. Therefore, to understand and avoid these bad practices, this paper introduces the first systematic study of bad practices in smart contracts, delving into over 35 specific issues. Specifically, we propose a large language models (LLMs)-based framework, SCALM. It combines Step-Back Prompting and Retrieval-Augmented Generation (RAG) to identify and address various bad practices effectively. Our extensive experiments using multiple LLMs and datasets have shown that SCALM outperforms existing tools in detecting bad practices in smart contracts. 

**Abstract (ZH)**: 随着以太坊平台的不断完善和广泛使用，保持高标准的智能合约编写规范至关重要。虽然不良的智能合约实践可能不会直接导致安全问题，但它们确实增加了遇到问题的风险。因此，为了理解和避免这些不良实践，本文首次对智能合约中的不良实践进行了系统研究，深入探讨了超过35个具体问题。具体而言，我们提出了一种基于大规模语言模型（LLMs）的框架SCALM。该框架结合了反向提示（Step-Back Prompting）和检索增强生成（RAG）技术，能够有效地识别和解决各种不良实践。大量使用多种LLMs和数据集进行的实验表明，SCALM在检测智能合约中的不良实践方面优于现有的工具。 

---
# Multi-Lingual Cyber Threat Detection in Tweets/X Using ML, DL, and LLM: A Comparative Analysis 

**Title (ZH)**: 使用机器学习、深度学习和大语言模型在推文中/上进行多语言网络威胁检测：一种 comparative analysis 分析 

**Authors**: Saydul Akbar Murad, Ashim Dahal, Nick Rahimi  

**Link**: [PDF](https://arxiv.org/pdf/2502.04346)  

**Abstract**: Cyber threat detection has become an important area of focus in today's digital age due to the growing spread of fake information and harmful content on social media platforms such as Twitter (now 'X'). These cyber threats, often disguised within tweets, pose significant risks to individuals, communities, and even nations, emphasizing the need for effective detection systems. While previous research has explored tweet-based threats, much of the work is limited to specific languages, domains, or locations, or relies on single-model approaches, reducing their applicability to diverse real-world scenarios. To address these gaps, our study focuses on multi-lingual tweet cyber threat detection using a variety of advanced models. The research was conducted in three stages: (1) We collected and labeled tweet datasets in four languages English, Chinese, Russian, and Arabic employing both manual and polarity-based labeling methods to ensure high-quality annotations. (2) Each dataset was analyzed individually using machine learning (ML) and deep learning (DL) models to assess their performance on distinct languages. (3) Finally, we combined all four datasets into a single multi-lingual dataset and applied DL and large language model (LLM) architectures to evaluate their efficacy in identifying cyber threats across various languages. Our results show that among machine learning models, Random Forest (RF) attained the highest performance; however, the Bi-LSTM architecture consistently surpassed other DL and LLM architectures across all datasets. These findings underline the effectiveness of Bi-LSTM in multilingual cyber threat detection. The code for this paper can be found at this link: this https URL. 

**Abstract (ZH)**: 网络安全威胁检测已成为当今数字时代的一个重要研究领域，尤其是在诸如Twitter（现为‘X’）等社交媒体平台上，虚假信息和有害内容的传播日益严重。这些网络安全威胁往往隐藏在推文中，对个人、社区甚至国家构成了重大风险，强调了有效检测系统的必要性。尽管先前的研究已经探索了基于推文的威胁，但大部分工作局限于特定的语言、领域或地理位置，或者依赖单一模型的方法，这限制了其在多样化的实际场景中的应用。为弥补这些不足，本研究专注于使用多种先进模型进行多语言推文网络威胁检测。研究分为三个阶段：（1）我们收集并标注了四种语言（英语、中文、俄语和阿拉伯语）的推文数据集，采用了手动和极性标注方法，以确保高质量的标注。 （2）每个数据集分别使用机器学习（ML）和深度学习（DL）模型进行分析，评估它们在不同语言上的性能。 （3）最后，我们将四个数据集合并成一个多语言数据集，并采用DL和大语言模型（LLM）架构来评估其在多种语言下的识别网络安全威胁的有效性。研究结果显示，在机器学习模型中，随机森林（RF）表现最佳；但在所有数据集中，双向长短期记忆网络（Bi-LSTM）架构持续超越其他DL和LLM架构，表明Bi-LSTM在多语言网络安全威胁检测中的有效性。此论文的相关代码可以在以下链接中找到：this https URL。 

---
# JingFang: A Traditional Chinese Medicine Large Language Model of Expert-Level Medical Diagnosis and Syndrome Differentiation-Based Treatment 

**Title (ZH)**: jingfang：基于 expert-level 诊断和辨证施治的中医大型语言模型 

**Authors**: Yehan Yan, Tianhao Ma, Ruotai Li, Xinhan Zheng, Guodong Shan, Chisheng Li  

**Link**: [PDF](https://arxiv.org/pdf/2502.04345)  

**Abstract**: Traditional Chinese medicine (TCM) plays a vital role in health protection and disease treatment, but its practical application requires extensive medical knowledge and clinical experience. Existing TCM Large Language Models (LLMs) exhibit critical limitations of uncomprehensive medical consultation and diagnoses, and inaccurate syndrome differentiation-based treatment. To address these issues, this study establishes JingFang (JF): a novel TCM Large Language Model that demonstrates the expert-level capability of medical diagnosis and syndrome differentiation-based treatment. We innovate a Multi-agent Dynamic Collaborative Chain-of-Thought Mechanism (MDCCTM) for medical consultation, enabling JF with effective and accurate diagnostic ability. In addition, a Syndrome Agent and a Dual-Stage Retrieval Scheme (DSRS) are developed to significantly enhance the capacity of JF for disease treatment based on syndrome differentiation. JingFang not only facilitates the application of LLMs but also promotes the effective practice of TCM in human health protection and disease treatment. 

**Abstract (ZH)**: 中医（TCM）在健康保护和疾病治疗中扮演着重要角色，但其实际应用需要广泛深厚的医学知识和临床经验。现有的中医大型语言模型（TCM LLMs）在医疗咨询和诊断方面存在不足，且基于辨证施治的治疗准确度不高。为解决这些问题，本研究构建了景芳（JingFang，JF）：一种具有专家级医疗诊断和基于辨证施治治疗能力的新型中医大型语言模型。我们创新了多智能体动态协作推理机制（MDCCTM），增强了JF的有效和准确的诊断能力。此外，我们开发了辨证智能体和双阶段检索方案（DSRS），显著提升了JF基于辨证施治的疾病治疗能力。景芳不仅促进了大型语言模型的应用，还推动了中医在人类健康保护和疾病治疗中的有效实践。 

---
# Tutorial on Using Machine Learning and Deep Learning Models for Mental Illness Detection 

**Title (ZH)**: 使用机器学习和深度学习模型进行精神疾病检测教程 

**Authors**: Yeyubei Zhang, Zhongyan Wang, Zhanyi Ding, Yexin Tian, Jianglai Dai, Xiaorui Shen, Yunchong Liu, Yuchen Cao  

**Link**: [PDF](https://arxiv.org/pdf/2502.04342)  

**Abstract**: Social media has become an important source for understanding mental health, providing researchers with a way to detect conditions like depression from user-generated posts. This tutorial provides practical guidance to address common challenges in applying machine learning and deep learning methods for mental health detection on these platforms. It focuses on strategies for working with diverse datasets, improving text preprocessing, and addressing issues such as imbalanced data and model evaluation. Real-world examples and step-by-step instructions demonstrate how to apply these techniques effectively, with an emphasis on transparency, reproducibility, and ethical considerations. By sharing these approaches, this tutorial aims to help researchers build more reliable and widely applicable models for mental health research, contributing to better tools for early detection and intervention. 

**Abstract (ZH)**: 社交媒体已成为理解心理健康的重要来源，为研究人员提供了一种从用户生成的帖子中检测如抑郁症等条件的方法。本文献教程提供了在这些平台上应用机器学习和深度学习方法进行心理健康检测时的实用指导。它侧重于处理多样化的数据集、改进文本预处理、以及解决数据不平衡和模型评估等问题的策略。通过实例和逐步指导，本文档展示了如何有效地应用这些技术，重点强调透明性、可重复性和伦理考量。通过分享这些方法，本文档旨在帮助研究人员构建更可靠和广泛适用的心理健康研究模型，从而为早期检测和干预提供更好的工具。 

---
# Joint MoE Scaling Laws: Mixture of Experts Can Be Memory Efficient 

**Title (ZH)**: 联合MoE规模律：混合专家模型可以实现内存高效性 

**Authors**: Jan Ludziejewski, Maciej Pióro, Jakub Krajewski, Maciej Stefaniak, Michał Krutul, Jan Małaśnicki, Marek Cygan, Piotr Sankowski, Kamil Adamczewski, Piotr Miłoś, Sebastian Jaszczur  

**Link**: [PDF](https://arxiv.org/pdf/2502.05172)  

**Abstract**: Mixture of Experts (MoE) architectures have significantly increased computational efficiency in both research and real-world applications of large-scale machine learning models. However, their scalability and efficiency under memory constraints remain relatively underexplored. In this work, we present joint scaling laws for dense and MoE models, incorporating key factors such as the number of active parameters, dataset size, and the number of experts. Our findings provide a principled framework for selecting the optimal MoE configuration under fixed memory and compute budgets. Surprisingly, we show that MoE models can be more memory-efficient than dense models, contradicting conventional wisdom. To derive and validate the theoretical predictions of our scaling laws, we conduct over 280 experiments with up to 2.7B active parameters and up to 5B total parameters. These results offer actionable insights for designing and deploying MoE models in practical large-scale training scenarios. 

**Abstract (ZH)**: 混合专家（MoE）架构在大规模机器学习模型的研究和实际应用中显著提高了计算效率。然而，它们在内存限制下的可扩展性和效率仍然相对未被充分探索。本文中，我们提出了密集模型和MoE模型的联合缩放定律，涵盖了关键因素如活跃参数的数量、数据集大小和专家的数量。我们的研究结果提供了一个基于固定内存和计算预算下选择最优MoE配置的原理性框架。出人意料的是，我们展示了MoE模型在内存效率方面可能优于密集模型，这与传统观点相反。为了推导并验证我们缩放定律的理论预测，我们进行了超过280次实验，其中活跃参数数量最多可达27亿，总参数数量最多可达50亿。这些结果为在实际大规模训练场景中设计和部署MoE模型提供了可操作性的指导。 

---
# Scaling up Test-Time Compute with Latent Reasoning: A Recurrent Depth Approach 

**Title (ZH)**: 通过潜在推理扩展测试时计算：一种递归深度方法 

**Authors**: Jonas Geiping, Sean McLeish, Neel Jain, John Kirchenbauer, Siddharth Singh, Brian R. Bartoldson, Bhavya Kailkhura, Abhinav Bhatele, Tom Goldstein  

**Link**: [PDF](https://arxiv.org/pdf/2502.05171)  

**Abstract**: We study a novel language model architecture that is capable of scaling test-time computation by implicitly reasoning in latent space. Our model works by iterating a recurrent block, thereby unrolling to arbitrary depth at test-time. This stands in contrast to mainstream reasoning models that scale up compute by producing more tokens. Unlike approaches based on chain-of-thought, our approach does not require any specialized training data, can work with small context windows, and can capture types of reasoning that are not easily represented in words. We scale a proof-of-concept model to 3.5 billion parameters and 800 billion tokens. We show that the resulting model can improve its performance on reasoning benchmarks, sometimes dramatically, up to a computation load equivalent to 50 billion parameters. 

**Abstract (ZH)**: 我们研究了一种新颖的语言模型架构，该架构能够在测试时通过隐式在潜在空间中推理实现计算扩展。我们的模型通过迭代循环块来实现这一点，因此在测试时可以扩展到任意深度。这与主流的通过生成更多 tokens 来扩展计算能力的推理模型形成了对比。与基于思考链的方法不同，我们的方法不需要任何特殊的训练数据，可以处理较小的上下文窗口，并且能够捕捉不易用文字表示的推理类型。我们将一个概念验证模型扩展至35亿参数和800亿tokens。我们展示了该模型在推理基准测试中的性能得到了提升，有时甚至大幅提升，相当于50亿参数的计算负载。 

---
# A Lightweight Method to Disrupt Memorized Sequences in LLM 

**Title (ZH)**: 一种轻量级方法，用于破坏LLM中的记忆化序列 

**Authors**: Parjanya Prajakta Prashant, Kaustubh Ponkshe, Babak Salimi  

**Link**: [PDF](https://arxiv.org/pdf/2502.05159)  

**Abstract**: Large language models (LLMs) demonstrate impressive capabilities across many tasks yet risk reproducing copyrighted content verbatim, raising legal and ethical concerns. Although methods like differential privacy or neuron editing can reduce memorization, they typically require costly retraining or direct access to model weights and may degrade performance. To address these challenges, we propose TokenSwap, a lightweight, post-hoc approach that replaces the probabilities of grammar-related tokens with those from a small auxiliary model (e.g., DistilGPT-2). We run extensive experiments on commercial grade models such as Pythia-6.9b and LLaMA-3-8b and demonstrate that our method effectively reduces well-known cases of memorized generation by upto 10x with little to no impact on downstream tasks. Our approach offers a uniquely accessible and effective solution to users of real-world systems. 

**Abstract (ZH)**: 大型语言模型（LLMs）在众多任务上展现出了令人印象深刻的性能，但也存在重现受版权保护内容的风险，这引起了法律和伦理方面的担忧。尽管差分隐私或神经元编辑等方法可以减轻这种记忆现象，但它们通常需要昂贵的重新训练或直接访问模型权重，并且可能会降低模型的性能。为了应对这些挑战，我们提出了一种轻量级的后处理方法，称为TokenSwap，该方法通过用一个小辅助模型（例如，DistilGPT-2）的概率替换与语法相关的令牌概率来实现。我们对具有商业级别的模型（如Pythia-6.9b和LLaMA-3-8b）进行了广泛的实验，并证明了我们的方法能将已知的被记忆生成的情况有效减少多达10倍，同时对下游任务几乎没有影响。我们的方法为现实系统中的用户提供了独特且有效的解决方案。 

---
# An Annotated Reading of 'The Singer of Tales' in the LLM Era 

**Title (ZH)**: “歌手与故事”在大语言模型时代的一篇注释阅读 

**Authors**: Kush R. Varshney  

**Link**: [PDF](https://arxiv.org/pdf/2502.05148)  

**Abstract**: The Parry-Lord oral-formulaic theory was a breakthrough in understanding how oral narrative poetry is learned, composed, and transmitted by illiterate bards. In this paper, we provide an annotated reading of the mechanism underlying this theory from the lens of large language models (LLMs) and generative artificial intelligence (AI). We point out the the similarities and differences between oral composition and LLM generation, and comment on the implications to society and AI policy. 

**Abstract (ZH)**: 帕里-洛德口头-公式理论在理解无文字传承人如何学习、创作和传承口头叙事诗方面取得了突破。本文从大规模语言模型（LLM）和生成型人工智能（AI）的视角，对这一理论背后机制进行注释性解读。我们指出了口头创作与LLM生成之间的相似性和差异性，并讨论了这对社会和AI政策的潜在影响。 

---
# Lost in Time: Clock and Calendar Understanding Challenges in Multimodal LLMs 

**Title (ZH)**: 迷失在时间之中：多模态大语言模型中的时钟和日历理解挑战 

**Authors**: Rohit Saxena, Aryo Pradipta Gema, Pasquale Minervini  

**Link**: [PDF](https://arxiv.org/pdf/2502.05092)  

**Abstract**: Understanding time from visual representations is a fundamental cognitive skill, yet it remains a challenge for multimodal large language models (MLLMs). In this work, we investigate the capabilities of MLLMs in interpreting time and date through analogue clocks and yearly calendars. To facilitate this, we curated a structured dataset comprising two subsets: 1) $\textit{ClockQA}$, which comprises various types of clock styles$-$standard, black-dial, no-second-hand, Roman numeral, and arrow-hand clocks$-$paired with time related questions; and 2) $\textit{CalendarQA}$, which consists of yearly calendar images with questions ranging from commonly known dates (e.g., Christmas, New Year's Day) to computationally derived ones (e.g., the 100th or 153rd day of the year). We aim to analyse how MLLMs can perform visual recognition, numerical reasoning, and temporal inference when presented with time-related visual data. Our evaluations show that despite recent advancements, reliably understanding time remains a significant challenge for MLLMs. 

**Abstract (ZH)**: 从视觉表示中理解时间是一项基本的认知技能，但对于多模态大型语言模型（MLLMs）来说仍然是一个挑战。在本文中，我们探讨了MLLMs通过模拟时钟和年历来解释时间和日期的能力。为此，我们整理了一个结构化的数据集，其中包括两个子集：1) $\textit{ClockQA}$，其中包括各种类型的时钟样式——标准时钟、漆黑钟盘、无秒针、罗马数字和箭头指针时钟——并配有时关问题；2) $\textit{CalendarQA}$，其中包括年历图像，问题范围从广为人知的日期（例如圣诞节、新年）到计算得出的日期（例如一年中的第100天或第153天）。我们的目标是分析在面对与时间相关的视觉数据时，MLLMs如何进行视觉识别、数值推理和时间推断。我们的评估显示，尽管近年来有所进步，但可靠地理解时间仍然是MLLMs面临的一个重大挑战。 

---
# Mitigating Unintended Memorization with LoRA in Federated Learning for LLMs 

**Title (ZH)**: 使用LoRA减轻联邦学习中大规模语言模型意外记忆的问题 

**Authors**: Thierry Bossy, Julien Vignoud, Tahseen Rabbani, Juan R. Troncoso Pastoriza, Martin Jaggi  

**Link**: [PDF](https://arxiv.org/pdf/2502.05087)  

**Abstract**: Federated learning (FL) is a popular paradigm for collaborative training which avoids direct data exposure between clients. However, data privacy issues still remain: FL-trained large language models are capable of memorizing and completing phrases and sentences contained in training data when given with their prefixes. Thus, it is possible for adversarial and honest-but-curious clients to recover training data of other participants simply through targeted prompting. In this work, we demonstrate that a popular and simple fine-tuning strategy, low-rank adaptation (LoRA), reduces memorization during FL up to a factor of 10. We study this effect by performing a medical question-answering fine-tuning task and injecting multiple replicas of out-of-distribution sensitive sequences drawn from an external clinical dataset. We observe a reduction in memorization for a wide variety of Llama 2 and 3 models, and find that LoRA can reduce memorization in centralized learning as well. Furthermore, we show that LoRA can be combined with other privacy-preserving techniques such as gradient clipping and Gaussian noising, secure aggregation, and Goldfish loss to further improve record-level privacy while maintaining performance. 

**Abstract (ZH)**: 联邦学习（FL）是一种流行的协作训练范式，可以避免客户端之间直接的数据暴露。然而，数据隐私问题仍然存在：通过前缀提示，FL训练的大型语言模型能够记住并完成训练数据中包含的短语和句子。因此，恶意客户端和出于好奇心而诚实的客户端有可能通过有针对性的提示简单地恢复其他参与者的数据。在本工作中，我们证明了一种流行且简单的微调策略——低秩适应（LoRA）——可以将FL中的记忆减少高达10倍。我们通过执行医学问答微调任务，并从外部临床数据集中注入多种分布外敏感序列，研究了这种效果。我们观察到，LoRA在 llama 2 和 3 等多种模型中都减少了记忆现象，并发现LoRA也可以在集中学习中减少记忆。此外，我们展示了LoRA可以与其他隐私保护技术（如梯度裁剪和高斯噪声、安全聚合和Goldfish损失）结合使用，以进一步提高数据记录级别的隐私保护，同时保持性能。 

---
# Adaptive Graph of Thoughts: Test-Time Adaptive Reasoning Unifying Chain, Tree, and Graph Structures 

**Title (ZH)**: 适应性思考图：测试时统一链、树和图结构的自适应推理 

**Authors**: Tushar Pandey, Ara Ghukasyan, Oktay Goktas, Santosh Kumar Radha  

**Link**: [PDF](https://arxiv.org/pdf/2502.05078)  

**Abstract**: Large Language Models (LLMs) have demonstrated impressive reasoning capabilities, yet their performance is highly dependent on the prompting strategy and model scale. While reinforcement learning and fine-tuning have been deployed to boost reasoning, these approaches incur substantial computational and data overhead. In this work, we introduce Adaptive Graph of Thoughts (AGoT), a dynamic, graph-based inference framework that enhances LLM reasoning solely at test time. Rather than relying on fixed-step methods like Chain of Thought (CoT) or Tree of Thoughts (ToT), AGoT recursively decomposes complex queries into structured subproblems, forming an dynamic directed acyclic graph (DAG) of interdependent reasoning steps. By selectively expanding only those subproblems that require further analysis, AGoT unifies the strengths of chain, tree, and graph paradigms into a cohesive framework that allocates computation where it is most needed. We validate our approach on diverse benchmarks spanning multi-hop retrieval, scientific reasoning, and mathematical problem-solving, achieving up to 46.2% improvement on scientific reasoning tasks (GPQA) - comparable to gains achieved through computationally intensive reinforcement learning approaches and outperforming state-of-the-art iterative approaches. These results suggest that dynamic decomposition and structured recursion offer a scalable, cost-effective alternative to post-training modifications, paving the way for more robust, general-purpose reasoning in LLMs. 

**Abstract (ZH)**: 以下是经过学术规范翻译后的论文内容或标题：

大型语言模型（LLMs）展示了令人印象深刻的推理能力，但其性能的高度依赖于提示策略和模型规模。尽管强化学习和微调已被用于增强推理能力，但这些方法会导致显著的计算和数据开销。在此项工作中，我们引入了动态图推理框架（Adaptive Graph of Thoughts, AGoT），这是一种仅在测试时增强LLM推理的动态图推理框架。AGoT不同于依赖于固定步长方法（如思维链CoT或思维树ToT），它通过递归地将复杂查询拆分为结构化的子问题，形成一个动态有向无环图（DAG），该图涵盖了相互依赖的推理步骤。通过仅扩展那些需要进一步分析的子问题，AGoT将链式、树状和图状模式的优势统一到一个有机框架中，将计算资源分配到最需要的地方。我们通过涵盖多跳检索、科学推理和数学问题解决等多种基准测试验证了该方法，科学推理任务（GPQA）的性能提升了高达46.2% - 这一成效与通过密集计算的强化学习方法取得的成效相当，并且超越了最先进的迭代方法。这些结果表明，动态分解和结构化递归提供了一种可扩展和成本效益高的替代方案，可以在LLMs中实现更稳健和通用的推理。 

---
# Paying Attention to Facts: Quantifying the Knowledge Capacity of Attention Layers 

**Title (ZH)**: 关注事实：量化注意力层的知识容量 

**Authors**: Liang Ze Wong  

**Link**: [PDF](https://arxiv.org/pdf/2502.05076)  

**Abstract**: In this paper, we investigate the ability of single-layer attention-only transformers (i.e. attention layers) to memorize facts contained in databases from a linear-algebraic perspective. We associate with each database a 3-tensor, propose the rank of this tensor as a measure of the size of the database, and provide bounds on the rank in terms of properties of the database. We also define a 3-tensor corresponding to an attention layer, and empirically demonstrate the relationship between its rank and database rank on a dataset of toy models and random databases. By highlighting the roles played by the value-output and query-key weights, and the effects of argmax and softmax on rank, our results shed light on the `additive motif' of factual recall in transformers, while also suggesting a way of increasing layer capacity without increasing the number of parameters. 

**Abstract (ZH)**: 在本文中，我们从线性代数的角度研究单层注意力机制变换器（即仅含注意力层的模型）在从数据库中记忆事实方面的能力。我们将每个数据库关联为一个3秩张量，并提出该张量的秩作为数据库大小的度量，并提供以数据库特性为基准的秩的上界和下界。我们还定义了一个与注意力层对应的3秩张量，并在包含玩具模型和随机数据库的数据集上实证展示了其秩与数据库秩之间的关系。通过突出值输出和查询键权重的作用，以及argmax和softmax对秩的影响，我们的结果阐明了变换器中事实回忆的“加性动机”，并提出了在不增加参数数量的情况下提高层容量的方法。 

---
# Lightweight Operations for Visual Speech Recognition 

**Title (ZH)**: 视觉言语识别中的轻量级操作 

**Authors**: Iason Ioannis Panagos, Giorgos Sfikas, Christophoros Nikou  

**Link**: [PDF](https://arxiv.org/pdf/2502.04834)  

**Abstract**: Visual speech recognition (VSR), which decodes spoken words from video data, offers significant benefits, particularly when audio is unavailable. However, the high dimensionality of video data leads to prohibitive computational costs that demand powerful hardware, limiting VSR deployment on resource-constrained devices. This work addresses this limitation by developing lightweight VSR architectures. Leveraging efficient operation design paradigms, we create compact yet powerful models with reduced resource requirements and minimal accuracy loss. We train and evaluate our models on a large-scale public dataset for recognition of words from video sequences, demonstrating their effectiveness for practical applications. We also conduct an extensive array of ablative experiments to thoroughly analyze the size and complexity of each model. Code and trained models will be made publicly available. 

**Abstract (ZH)**: 视觉语音识别（VSR），从视频数据中解码口语内容，带来了显著的好处，尤其是在音频不可用时。然而，视频数据的高维特性导致了高昂的计算成本，需要强大的硬件支持，从而限制了VSR在资源受限设备上的部署。本工作通过开发轻量级的VSR架构来解决这一限制。利用高效的操作设计范式，我们创建了既紧凑又强大的模型，减少了资源需求并最小化了准确性损失。我们使用一个大规模的公开数据集对这些模型进行了训练和评估，展示了它们在实际应用中的有效性。我们还进行了广泛的消融实验，以彻底分析每个模型的大小和复杂性。相关代码和训练后的模型也将公开发布。 

---
# ELITE: Enhanced Language-Image Toxicity Evaluation for Safety 

**Title (ZH)**: ELITE: 增强的语言-图像毒性评估以确保安全 

**Authors**: Wonjun Lee, Doehyeon Lee, Eugene Choi, Sangyoon Yu, Ashkan Yousefpour, Haon Park, Bumsub Ham, Suhyun Kim  

**Link**: [PDF](https://arxiv.org/pdf/2502.04757)  

**Abstract**: Current Vision Language Models (VLMs) remain vulnerable to malicious prompts that induce harmful outputs. Existing safety benchmarks for VLMs primarily rely on automated evaluation methods, but these methods struggle to detect implicit harmful content or produce inaccurate evaluations. Therefore, we found that existing benchmarks have low levels of harmfulness, ambiguous data, and limited diversity in image-text pair combinations. To address these issues, we propose the ELITE {\em benchmark}, a high-quality safety evaluation benchmark for VLMs, underpinned by our enhanced evaluation method, the ELITE {\em evaluator}. The ELITE evaluator explicitly incorporates a toxicity score to accurately assess harmfulness in multimodal contexts, where VLMs often provide specific, convincing, but unharmful descriptions of images. We filter out ambiguous and low-quality image-text pairs from existing benchmarks using the ELITE evaluator and generate diverse combinations of safe and unsafe image-text pairs. Our experiments demonstrate that the ELITE evaluator achieves superior alignment with human evaluations compared to prior automated methods, and the ELITE benchmark offers enhanced benchmark quality and diversity. By introducing ELITE, we pave the way for safer, more robust VLMs, contributing essential tools for evaluating and mitigating safety risks in real-world applications. 

**Abstract (ZH)**: 当前的视觉语言模型（VLMs）仍然容易受到恶意提示的影响，这些提示可能导致有害输出。现有的VLM安全性基准主要依赖于自动评估方法，但这些方法在检测隐含有害内容或产生不准确评估方面存在困难。因此，我们发现现有的基准在有害性、数据模糊性以及图像-文本配对组合的多样性方面存在不足。为解决这些问题，我们提出了 ELITE 基准，这是一个高质量的VLM安全性评估基准，基于我们改进的评估方法——ELITE 评估器。ELITE 评估器明确地引入了毒性评分，以准确评估多模态环境中VLMs提供的特定、令人信服但无害的图像描述。我们使用ELITE评估器筛选掉现有基准中的模糊和低质量的图像-文本配对，并生成多样化的安全与不安全图像-文本配对组合。我们的实验表明，ELITE评估器在人类评估方面的对齐程度优于先前的自动评估方法，且ELITE基准提供更高的基准质量和多样性。通过引入ELITE，我们为我们提供了一条通往更安全、更稳健的VLM的道路，从而为评估和缓解实际应用中的安全风险提供了必要的工具。 

---
# Holistically Guided Monte Carlo Tree Search for Intricate Information Seeking 

**Title (ZH)**: 全域引导的蒙特卡洛树搜索方法用于复杂的信息检索 

**Authors**: Ruiyang Ren, Yuhao Wang, Junyi Li, Jinhao Jiang, Wayne Xin Zhao, Wenjie Wang, Tat-Seng Chua  

**Link**: [PDF](https://arxiv.org/pdf/2502.04751)  

**Abstract**: In the era of vast digital information, the sheer volume and heterogeneity of available information present significant challenges for intricate information seeking. Users frequently face multistep web search tasks that involve navigating vast and varied data sources. This complexity demands every step remains comprehensive, accurate, and relevant. However, traditional search methods often struggle to balance the need for localized precision with the broader context required for holistic understanding, leaving critical facets of intricate queries underexplored. In this paper, we introduce an LLM-based search assistant that adopts a new information seeking paradigm with holistically guided Monte Carlo tree search (HG-MCTS). We reformulate the task as a progressive information collection process with a knowledge memory and unite an adaptive checklist with multi-perspective reward modeling in MCTS. The adaptive checklist provides explicit sub-goals to guide the MCTS process toward comprehensive coverage of complex user queries. Simultaneously, our multi-perspective reward modeling offers both exploration and retrieval rewards, along with progress feedback that tracks completed and remaining sub-goals, refining the checklist as the tree search progresses. By striking a balance between localized tree expansion and global guidance, HG-MCTS reduces redundancy in search paths and ensures that all crucial aspects of an intricate query are properly addressed. Extensive experiments on real-world intricate information seeking tasks demonstrate that HG-MCTS acquires thorough knowledge collections and delivers more accurate final responses compared with existing baselines. 

**Abstract (ZH)**: 在海量数字信息时代，可用信息的数量与异质性带来了复杂的信息检索挑战。用户经常需要进行多步骤的网络搜索任务，涉及在大量多样的数据源之间导航。这种复杂性要求每一步都必须全面、准确且相关。然而，传统的搜索方法往往难以在局部精确度的需求与实现整体理解所需的广泛背景之间取得平衡，导致复杂查询中的关键方面被忽视。在本文中，我们提出了一种基于LLM的搜索助手，采用了一种新的全指导蒙特卡洛树搜索（HG-MCTS）信息检索范式。我们将任务重新定义为渐进的信息收集过程，并结合了知识记忆和自适应检查表与多视角奖励建模。自适应检查表提供明确的次级目标，指导蒙特卡洛树搜索过程覆盖复杂用户查询的所有方面。同时，我们的多视角奖励建模提供了探索和检索奖励，并跟踪已完成和剩余的次级目标，随着树搜索的进行不断完善检查表。通过在局部树扩展与全局指导之间取得平衡，HG-MCTS减少了搜索路径中的冗余，确保了复杂查询的所有关键方面得到适当处理。在实际复杂的信息检索任务上的广泛实验表明，HG-MCTS能获得更全面的知识收集，并提供比现有基线更准确的最终响应。 

---
# Agentic Reasoning: Reasoning LLMs with Tools for the Deep Research 

**Title (ZH)**: 代理推理：利用工具进行深度研究的LLM推理方法 

**Authors**: Junde Wu, Jiayuan Zhu, Yuyuan Liu  

**Link**: [PDF](https://arxiv.org/pdf/2502.04644)  

**Abstract**: We introduce Agentic Reasoning, a framework that enhances large language model (LLM) reasoning by integrating external tool-using agents. Unlike conventional LLM-based reasoning approaches, which rely solely on internal inference, Agentic Reasoning dynamically engages web search, code execution, and structured reasoning-context memory to solve complex problems requiring deep research and multi-step logical deduction. Our framework introduces the Mind Map agent, which constructs a structured knowledge graph to track logical relationships, improving deductive reasoning. Additionally, the integration of web-search and coding agents enables real-time retrieval and computational analysis, enhancing reasoning accuracy and decision-making. Evaluations on PhD-level scientific reasoning (GPQA) and domain-specific deep research tasks demonstrate that our approach significantly outperforms existing models, including leading retrieval-augmented generation (RAG) systems and closed-source LLMs. Moreover, our results indicate that agentic reasoning improves expert-level knowledge synthesis, test-time scalability, and structured problem-solving. The code is at: this https URL. 

**Abstract (ZH)**: 我们介绍了代理推理（Agentic Reasoning）这一框架，该框架通过整合外部工具使用代理来增强大型语言模型（LLM）的推理能力。不同于依赖内部推断的传统基于LLM的推理方法，代理推理能够动态地结合网络搜索、代码执行和结构化推理上下文记忆，以解决需要深入研究和多步骤逻辑推理的复杂问题。我们的框架引入了心智图代理（Mind Map agent），该代理构建了一个结构化的知识图谱以追踪逻辑关系，从而提高演绎推理的能力。此外，网络搜索和编码代理的集成实现了实时检索和计算分析，增强了推理准确性和决策制定能力。

我们在博士水平的科学推理（GPQA）以及特定领域的深研究任务上的评估表明，我们的方法显著优于现有模型，包括领先的检索增强生成（RAG）系统和闭源的LLM。此外，我们的结果还表明，代理推理能够提高专家级知识的综合、测试时间的可扩展性和结构化问题解决的能力。相关代码可以在以下地址获取：[此处提供链接]。 

---
# Confidence Elicitation: A New Attack Vector for Large Language Models 

**Title (ZH)**: 自信度提取：大型语言模型的一种新型攻击向量 

**Authors**: Brian Formento, Chuan Sheng Foo, See-Kiong Ng  

**Link**: [PDF](https://arxiv.org/pdf/2502.04643)  

**Abstract**: A fundamental issue in deep learning has been adversarial robustness. As these systems have scaled, such issues have persisted. Currently, large language models (LLMs) with billions of parameters suffer from adversarial attacks just like their earlier, smaller counterparts. However, the threat models have changed. Previously, having gray-box access, where input embeddings or output logits/probabilities were visible to the user, might have been reasonable. However, with the introduction of closed-source models, no information about the model is available apart from the generated output. This means that current black-box attacks can only utilize the final prediction to detect if an attack is successful. In this work, we investigate and demonstrate the potential of attack guidance, akin to using output probabilities, while having only black-box access in a classification setting. This is achieved through the ability to elicit confidence from the model. We empirically show that the elicited confidence is calibrated and not hallucinated for current LLMs. By minimizing the elicited confidence, we can therefore increase the likelihood of misclassification. Our new proposed paradigm demonstrates promising state-of-the-art results on three datasets across two models (LLaMA-3-8B-Instruct and Mistral-7B-Instruct-V0.3) when comparing our technique to existing hard-label black-box attack methods that introduce word-level substitutions. 

**Abstract (ZH)**: 深度学习领域的一个基本问题是鲁棒性对抗。随着这些系统的扩展，这些问题依然存在。当前，具有数十亿参数的大型语言模型（LLMs）在面对对抗攻击时的表现与它们早期的小型版本并无二致。然而，威胁模型已经发生了变化。在过去，灰盒访问情况下，即输入嵌入或输出logits/概率对用户可见，可能是合理的。然而，在引入闭源模型后，除了生成的输出外，用户无法获得关于模型的任何信息。这意味着当前的黑盒攻击只能利用最终的预测来判断攻击是否成功。在本研究中，我们探讨并证明了在仅具黑盒访问权限的分类设置中，对抗指导（类似于使用输出概率）的潜力。这正是通过对模型提出置信度并使其响应的方式来实现的。我们实验证明，从当前的LLMs中提取出的置信度是经过校准的真实度量，而不是虚幻的。通过最小化提取出的置信度，我们可以在一定程度上增加分类错误的可能性。我们提出的新范式在两个模型（LLaMA-3-8B-Instruct 和 Mistral-7B-Instruct-V0.3）的三个数据集上取得了有前景的最先进的结果，当我们将这种技术与现有引入单词级替换的硬标签黑盒攻击方法进行比较时，结果尤为显著。 

---
# Position-aware Automatic Circuit Discovery 

**Title (ZH)**: 位置感知自动电路发现 

**Authors**: Tal Haklay, Hadas Orgad, David Bau, Aaron Mueller, Yonatan Belinkov  

**Link**: [PDF](https://arxiv.org/pdf/2502.04577)  

**Abstract**: A widely used strategy to discover and understand language model mechanisms is circuit analysis. A circuit is a minimal subgraph of a model's computation graph that executes a specific task. We identify a gap in existing circuit discovery methods: they assume circuits are position-invariant, treating model components as equally relevant across input positions. This limits their ability to capture cross-positional interactions or mechanisms that vary across positions. To address this gap, we propose two improvements to incorporate positionality into circuits, even on tasks containing variable-length examples. First, we extend edge attribution patching, a gradient-based method for circuit discovery, to differentiate between token positions. Second, we introduce the concept of a dataset schema, which defines token spans with similar semantics across examples, enabling position-aware circuit discovery in datasets with variable length examples. We additionally develop an automated pipeline for schema generation and application using large language models. Our approach enables fully automated discovery of position-sensitive circuits, yielding better trade-offs between circuit size and faithfulness compared to prior work. 

**Abstract (ZH)**: 广泛采用的一种方法是电路分析，用于发现和理解语言模型的机制。电路是一模型计算图形中的最小子图，执行特定任务。我们在现有电路发现方法中识别出了一个不足之处：这些方法假设电路是位置不变的，即它们认为模型组件在所有输入位置中具有等同的相关性。这种假设限制了它们捕捉跨位置交互或在不同位置上变化的机制的能力。为解决这一不足，我们提出两种改进措施，以便即使在包含变长示例的任务中也能将位置性纳入电路。首先，我们将基于梯度的方法（边缘归因补丁）扩展以区分token位置。其次，我们引入了数据集模式的概念，它定义了在不同示例中具有相似语义的token跨度，从而能够在长度可变的示例数据集中进行位置感知的电路发现。我们还开发了一种自动流水线，用于使用大型语言模型生成和应用这些模式。我们的方法使完全自动化的位置敏感电路发现成为可能，相比以前的工作，这在电路规模和忠实地权衡上有更好的效果。 

---
# Self-Regulation and Requesting Interventions 

**Title (ZH)**: 自我调节与请求干预 

**Authors**: So Yeon Min, Yue Wu, Jimin Sun, Max Kaufmann, Fahim Tajwar, Yonatan Bisk, Ruslan Salakhutdinov  

**Link**: [PDF](https://arxiv.org/pdf/2502.04576)  

**Abstract**: Human intelligence involves metacognitive abilities like self-regulation, recognizing limitations, and seeking assistance only when needed. While LLM Agents excel in many domains, they often lack this awareness. Overconfident agents risk catastrophic failures, while those that seek help excessively hinder efficiency. A key challenge is enabling agents with a limited intervention budget $C$ is to decide when to request assistance. In this paper, we propose an offline framework that trains a "helper" policy to request interventions, such as more powerful models or test-time compute, by combining LLM-based process reward models (PRMs) with tabular reinforcement learning. Using state transitions collected offline, we score optimal intervention timing with PRMs and train the helper model on these labeled trajectories. This offline approach significantly reduces costly intervention calls during training. Furthermore, the integration of PRMs with tabular RL enhances robustness to off-policy data while avoiding the inefficiencies of deep RL. We empirically find that our method delivers optimal helper behavior. 

**Abstract (ZH)**: 人类智能涉及元认知能力，如自我调节、认识到自身局限性以及仅在必要时寻求帮助。虽然大语言模型（LLM）代理在许多领域表现出色，但它们往往缺乏这种自我意识。自信过高的代理可能会导致灾难性故障，而过于频繁寻求帮助的代理会妨碍效率。一个关键挑战是在有限的干预预算 $C$ 下，使代理能够决定何时请求帮助。在本文中，我们提出了一种离线框架，该框架通过结合基于LLM的过程奖励模型（PRMs）和表格强化学习来训练一个“帮助者”策略，以请求更强大的模型或测试时的计算资源。利用离线收集的状态转换，我们使用PRMs评估最佳干预时机，并在这些有标签的轨迹上训练帮助者模型。这种方法显著降低了训练期间昂贵的干预呼叫成本。此外，将PRMs与表格RL结合使用增强了对离策数据的鲁棒性，同时避免了深度RL带来的低效率。我们实验证明，该方法能提供最优的帮助者行为。 

---
# Towards Cost-Effective Reward Guided Text Generation 

**Title (ZH)**: 面向高效奖励引导文本生成的研究 

**Authors**: Ahmad Rashid, Ruotian Wu, Rongqi Fan, Hongliang Li, Agustinus Kristiadi, Pascal Poupart  

**Link**: [PDF](https://arxiv.org/pdf/2502.04517)  

**Abstract**: Reward-guided text generation (RGTG) has emerged as a viable alternative to offline reinforcement learning from human feedback (RLHF). RGTG methods can align baseline language models to human preferences without further training like in standard RLHF methods. However, they rely on a reward model to score each candidate token generated by the language model at inference, incurring significant test-time overhead. Additionally, the reward model is usually only trained to score full sequences, which can lead to sub-optimal choices for partial sequences. In this work, we present a novel reward model architecture that is trained, using a Bradley-Terry loss, to prefer the optimal expansion of a sequence with just a \emph{single call} to the reward model at each step of the generation process. That is, a score for all possible candidate tokens is generated simultaneously, leading to efficient inference. We theoretically analyze various RGTG reward models and demonstrate that prior techniques prefer sub-optimal sequences compared to our method during inference. Empirically, our reward model leads to significantly faster inference than other RGTG methods. It requires fewer calls to the reward model and performs competitively compared to previous RGTG and offline RLHF methods. 

**Abstract (ZH)**: 奖励导向的文本生成（RGTG）已成为一种替代离线基于人类反馈的强化学习（RLHF）的可行方案。RGTG方法可以在生成过程中仅通过一次调用奖励模型来评估语言模型生成的每个候选词，从而将基线语言模型与人类偏好对齐，而无需进一步训练，类似于标准的RLHF方法。然而，这种方法依赖于奖励模型来为生成过程中的每个候选词评分，导致显著的测试时间开销。此外，奖励模型通常仅被训练用于评分完整序列，这可能导致部分序列的次优选择。在本文中，我们提出了一种新颖的奖励模型架构，通过布雷德利-特里（Bradley-Terry）损失训练该架构，在生成过程的每一步只需要一次调用奖励模型，即可倾向于选择序列的最佳扩展。也就是说，所有可能候选词的评分可以同时生成，从而实现高效的推理。我们对各种RGTG奖励模型进行了理论分析，并证明优先技术在推理过程中相较于我们的方法更倾向于选择次优序列。从实证上讲，我们的奖励模型实现了比其他RGTG方法显著更快的推理。它在调用奖励模型的次数上更少，并且与以前的RGTG和离线RLHF方法具有竞争力。 

---
# Revisiting Intermediate-Layer Matching in Knowledge Distillation: Layer-Selection Strategy Doesn't Matter (Much) 

**Title (ZH)**: 重访知识蒸馏中的中间层匹配：层选择策略无关紧要（或影响不大） 

**Authors**: Zony Yu, Yuqiao Wen, Lili Mou  

**Link**: [PDF](https://arxiv.org/pdf/2502.04499)  

**Abstract**: Knowledge distillation (KD) is a popular method of transferring knowledge from a large "teacher" model to a small "student" model. KD can be divided into two categories: prediction matching and intermediate-layer matching. We explore an intriguing phenomenon: layer-selection strategy does not matter (much) in intermediate-layer matching. In this paper, we show that seemingly nonsensical matching strategies such as matching the teacher's layers in reverse still result in surprisingly good student performance. We provide an interpretation for this phenomenon by examining the angles between teacher layers viewed from the student's perspective. 

**Abstract (ZH)**: 知识蒸馏（KD）是一种流行的将知识从一个大型的“教师”模型转移到一个小型的“学生”模型的方法。KD 可以分为两类：预测匹配和中间层匹配。我们探讨了一个引人注目的现象：在中间层匹配中，层选择策略并不重要（相差不大）。在本文中，我们展示了看似不合常理的匹配策略，例如将教师的层逆序匹配，仍然可以导致令人惊讶的学生模型性能。我们通过从学生视角分析教师层之间的角度来为这种现象提供一种解释。 

---
# Training Language Models to Reason Efficiently 

**Title (ZH)**: 训练高效推理的语言模型 

**Authors**: Daman Arora, Andrea Zanette  

**Link**: [PDF](https://arxiv.org/pdf/2502.04463)  

**Abstract**: Scaling model size and training data has led to great advances in the performance of Large Language Models (LLMs). However, the diminishing returns of this approach necessitate alternative methods to improve model capabilities, particularly in tasks requiring advanced reasoning. Large reasoning models, which leverage long chain-of-thoughts, bring unprecedented breakthroughs in problem-solving capabilities but at a substantial deployment cost associated to longer generations. Reducing inference costs is crucial for the economic feasibility, user experience, and environmental sustainability of these models.
In this work, we propose to train large reasoning models to reason efficiently. More precisely, we use reinforcement learning (RL) to train reasoning models to dynamically allocate inference-time compute based on task complexity. Our method incentivizes models to minimize unnecessary computational overhead while maintaining accuracy, thereby achieving substantial efficiency gains. It enables the derivation of a family of reasoning models with varying efficiency levels, controlled via a single hyperparameter. Experiments on two open-weight large reasoning models demonstrate significant reductions in inference cost while preserving most of the accuracy. 

**Abstract (ZH)**: 将以下论文内容或标题翻译成中文，要符合学术规范：

通过扩大模型规模和训练数据，大型语言模型（LLMs）的性能得到了显著提升。然而，这种方法的边际收益逐渐减少，因此需要寻找其他方法来提高模型能力，特别是在需要高级推理任务的领域。通过利用长链推理的大推理模型带来了解决问题能力的前所未有的突破，但这也伴随着较高的部署成本，主要体现在更长的推理生成过程。降低推理成本对于这些模型的经济可行性、用户体验和环境可持续性至关重要。

在本工作中，我们提出了一种高效训练大型推理模型的方法。具体而言，我们利用强化学习（RL）训练推理模型，使其能够根据任务复杂性动态分配推理时的计算资源。该方法通过激励模型最小化不必要的计算开销以保持准确性，从而实现显著的效率提升。该方法还能够生成一系列具有不同效率级别的推理模型，唯一通过一个超参数控制这些变化。实验证明，对于两个开源的大型推理模型，在保持大部分准确性的同时，推理成本显著降低。 

---
# Primary Care Diagnoses as a Reliable Predictor for Orthopedic Surgical Interventions 

**Title (ZH)**: 家庭医学诊断作为骨科手术干预的可靠预测指标 

**Authors**: Khushboo Verma, Alan Michels, Ergi Gumusaneli, Shilpa Chitnis, Smita Sinha Kumar, Christopher Thompson, Lena Esmail, Guruprasath Srinivasan, Chandini Panchada, Sushovan Guha, Satwant Kumar  

**Link**: [PDF](https://arxiv.org/pdf/2502.04423)  

**Abstract**: Referral workflow inefficiencies, including misaligned referrals and delays, contribute to suboptimal patient outcomes and higher healthcare costs. In this study, we investigated the possibility of predicting procedural needs based on primary care diagnostic entries, thereby improving referral accuracy, streamlining workflows, and providing better care to patients. A de-identified dataset of 2,086 orthopedic referrals from the University of Texas Health at Tyler was analyzed using machine learning models built on Base General Embeddings (BGE) for semantic extraction. To ensure real-world applicability, noise tolerance experiments were conducted, and oversampling techniques were employed to mitigate class imbalance. The selected optimum and parsimonious embedding model demonstrated high predictive accuracy (ROC-AUC: 0.874, Matthews Correlation Coefficient (MCC): 0.540), effectively distinguishing patients requiring surgical intervention. Dimensionality reduction techniques confirmed the model's ability to capture meaningful clinical relationships. A threshold sensitivity analysis identified an optimal decision threshold (0.30) to balance precision and recall, maximizing referral efficiency. In the predictive modeling analysis, the procedure rate increased from 11.27% to an optimal 60.1%, representing a 433% improvement with significant implications for operational efficiency and healthcare revenue.
The results of our study demonstrate that referral optimization can enhance primary and surgical care integration. Through this approach, precise and timely predictions of procedural requirements can be made, thereby minimizing delays, improving surgical planning, and reducing administrative burdens. In addition, the findings highlight the potential of clinical decision support as a scalable solution for improving patient outcomes and the efficiency of the healthcare system. 

**Abstract (ZH)**: 转诊工作流程中的低效，包括不当转诊和延误，导致患者结果不佳和医疗成本增加。本研究旨在探讨基于初级诊疗诊断条目的预测程序需求的可能性，从而提高转诊准确性、优化工作流程并更好地为患者服务。我们分析了德克萨斯大学泰勒健康大学的2,086例骨科转诊数据集，使用基于本体通用嵌入（BGE）的机器学习模型进行语义提取。为了确保适用性，我们进行了噪声容忍实验，并采用过采样技术来缓解类别不平衡问题。所选的最优且简洁的嵌入模型展现了较高的预测准确性（ROC-AUC：0.874，马修斯相关系数（MCC）：0.540），能够有效区分需要进行手术干预的患者。降维技术证实了该模型能够捕捉到有意义的临床关系。阈值灵敏度分析确定了最佳决策阈值（0.30），以平衡精确率和召回率，从而最大化转诊效率。在预测建模分析中，程序率从11.27%提高到优化的60.1%，这代表了433%的改进，对运营效率和医疗收入具有重大影响。

研究结果表明，转诊优化可以增强初级和手术护理的整合。通过这种方式，可以做出准确及时的程序需求预测，从而减少延误、改善手术计划并减轻行政负担。此外，研究结果还突显了临床决策支持作为提高患者结果和医疗系统效率的可扩展解决方案的潜力。 

---
# KVTuner: Sensitivity-Aware Layer-wise Mixed Precision KV Cache Quantization for Efficient and Nearly Lossless LLM Inference 

**Title (ZH)**: KVTuner：一种感知灵敏度的分层混合精度键值缓存量化方法，实现高效且近乎无损的大型语言模型推理 

**Authors**: Xing Li, Zeyu Xing, Yiming Li, Linping Qu, Hui-Ling Zhen, Wulong Liu, Yiwu Yao, Sinno Jialin Pan, Mingxuan Yuan  

**Link**: [PDF](https://arxiv.org/pdf/2502.04420)  

**Abstract**: KV cache quantization can improve Large Language Models (LLMs) inference throughput and latency in long contexts and large batch-size scenarios while preserving LLMs effectiveness. However, current methods have three unsolved issues: overlooking layer-wise sensitivity to KV cache quantization, high overhead of online fine-grained decision-making, and low flexibility to different LLMs and constraints. Therefore, we thoroughly analyze the inherent correlation of layer-wise transformer attention patterns to KV cache quantization errors and study why key cache is more important than value cache for quantization error reduction. We further propose a simple yet effective framework KVTuner to adaptively search for the optimal hardware-friendly layer-wise KV quantization precision pairs for coarse-grained KV cache with multi-objective optimization and directly utilize the offline searched configurations during online inference. To reduce the computational cost of offline calibration, we utilize the intra-layer KV precision pair pruning and inter-layer clustering to reduce the search space. Experimental results show that we can achieve nearly lossless 3.25-bit mixed precision KV cache quantization for LLMs like Llama-3.1-8B-Instruct and 4.0-bit for sensitive models like Qwen2.5-7B-Instruct on mathematical reasoning tasks. The maximum inference throughput can be improved by 38.3% compared with KV8 quantization over various context lengths. 

**Abstract (ZH)**: 键值缓存量化可以提高大型语言模型（LLMs）在长上下文和大批次场景下的推理吞吐量和延迟，同时保留LLMs的有效性。然而，当前的方法存在三个未解决的问题：忽略了层间对键值缓存量化的敏感性、在线进行细粒度决策的高开销，以及对不同LLMs和约束条件的低灵活性。因此，我们深入分析了层间变换注意力模式与键值缓存量化误差的固有关系，并研究了为什么在量化误差减少方面关键字缓存比值缓存更重要。在此基础上，我们提出了一种简单且有效的框架KVTuner，用于自适应地搜索适用于粗粒度键值缓存的最优硬件友好型层级键值量化精度配对，并直接利用离线搜索的配置进行在线推理。为了降低离线标定的计算成本，我们利用层内键值精度配对剪枝和跨层聚类来减少搜索空间。实验结果表明，我们可以在LLMs（如Llama-3.1-8B-Instruct）上实现接近无损的3.25位混合精度键值缓存量化，并在敏感模型（如Qwen2.5-7B-Instruct）上实现4.0位量化，在数学推理任务中保持接近无损的效果。与KV8量化相比，最大推理吞吐量在各种上下文长度下可提高38.3%。 

---
# Understanding and Mitigating the Bias Inheritance in LLM-based Data Augmentation on Downstream Tasks 

**Title (ZH)**: 理解并减轻基于LLM的数据增强在下游任务中的偏差继承问题 

**Authors**: Miaomiao Li, Hao Chen, Yang Wang, Tingyuan Zhu, Weijia Zhang, Kaijie Zhu, Kam-Fai Wong, Jindong Wang  

**Link**: [PDF](https://arxiv.org/pdf/2502.04419)  

**Abstract**: Generating synthetic datasets via large language models (LLMs) themselves has emerged as a promising approach to improve LLM performance. However, LLMs inherently reflect biases present in their training data, leading to a critical challenge: when these models generate synthetic data for training, they may propagate and amplify their inherent biases that can significantly impact model fairness and robustness on downstream tasks--a phenomenon we term bias inheritance. This work presents the first systematic investigation in understanding, analyzing, and mitigating bias inheritance. We study this problem by fine-tuning LLMs with a combined dataset consisting of original and LLM-augmented data, where bias ratio represents the proportion of augmented data. Through systematic experiments across 10 classification and generation tasks, we analyze how 6 different types of biases manifest at varying bias ratios. Our results reveal that bias inheritance has nuanced effects on downstream tasks, influencing both classification tasks and generation tasks differently. Then, our analysis identifies three key misalignment factors: misalignment of values, group data, and data distributions. Based on these insights, we propose three mitigation strategies: token-based, mask-based, and loss-based approaches. Experiments demonstrate that these strategies also work differently on various tasks and bias, indicating the substantial challenges to fully mitigate bias inheritance. We hope this work can provide valuable insights to the research of LLM data augmentation. 

**Abstract (ZH)**: 通过大型语言模型（LLMs）本身生成合成数据集已成为提高LLM性能的一种有前途的方法。然而，LLMs 本质上反映了其训练数据中存在的偏见，这导致了一个关键挑战：当这些模型生成用于训练的合成数据时，它们可能会传播并放大其固有的偏见，从而严重影响下游任务上的模型公平性和鲁棒性——我们称这一现象为偏见继承。本文首次系统地探讨了偏见继承的理解、分析和缓解。我们通过使用原始数据和LLM扩充数据组合的数据集对LLMs进行微调，其中偏见比例表示扩充数据的比例，系统地研究了这一问题。通过在10个分类和生成任务上的系统实验，我们分析了六种不同类型的偏见在不同偏见比例下的表现。结果显示，偏见继承对下游任务的影响是复杂的，其影响了分类任务和生成任务的不同方式。随后，我们的分析确定了三个关键的不一致性因素：价值不一致、群体数据和数据分布不一致。基于这些见解，我们提出了三种缓解策略：基于令牌、基于掩码和基于损失的方法。实验表明，这些策略在不同任务和偏见方面表现出不同的效果，表明完全缓解偏见继承的挑战是巨大的。希望这项工作能为LLM数据增强的研究提供有价值的见解。 

---
# Decoder-Only LLMs are Better Controllers for Diffusion Models 

**Title (ZH)**: 仅解码器的大语言模型是控制扩散模型的更好选择 

**Authors**: Ziyi Dong, Yao Xiao, Pengxu Wei, Liang Lin  

**Link**: [PDF](https://arxiv.org/pdf/2502.04412)  

**Abstract**: Groundbreaking advancements in text-to-image generation have recently been achieved with the emergence of diffusion models. These models exhibit a remarkable ability to generate highly artistic and intricately detailed images based on textual prompts. However, obtaining desired generation outcomes often necessitates repetitive trials of manipulating text prompts just like casting spells on a magic mirror, and the reason behind that is the limited capability of semantic understanding inherent in current image generation models. Specifically, existing diffusion models encode the text prompt input with a pre-trained encoder structure, which is usually trained on a limited number of image-caption pairs. The state-of-the-art large language models (LLMs) based on the decoder-only structure have shown a powerful semantic understanding capability as their architectures are more suitable for training on very large-scale unlabeled data. In this work, we propose to enhance text-to-image diffusion models by borrowing the strength of semantic understanding from large language models, and devise a simple yet effective adapter to allow the diffusion models to be compatible with the decoder-only structure. Meanwhile, we also provide a supporting theoretical analysis with various architectures (e.g., encoder-only, encoder-decoder, and decoder-only), and conduct extensive empirical evaluations to verify its effectiveness. The experimental results show that the enhanced models with our adapter module are superior to the stat-of-the-art models in terms of text-to-image generation quality and reliability. 

**Abstract (ZH)**: 随着扩散模型的出现，文本生成图像领域取得了突破性的进展。这些模型表现出根据文本提示生成高度艺术化和复杂细节图像的非凡能力。然而，获得理想的生成结果往往需要重复尝试调整文本提示，就像对着魔镜念咒语一样。原因在于当前图像生成模型在语义理解方面的有限能力。具体而言，现有的扩散模型通过预训练的编码器结构将文本提示输入编码，该结构通常仅在有限数量的图像-标题对上进行训练。基于仅解码器结构的大语言模型（LLMs）展现了强大的语义理解能力，因为它们的架构更适合在大规模未标记数据上进行训练。在此项工作中，我们提出通过借用大语言模型在语义理解方面的强大能力来增强文本生成图像的扩散模型，并设计了一个简单而有效的适配器，以使扩散模型能够兼容仅解码器结构。此外，我们还提供了一种支持性的理论分析，涉及各种架构（如仅有编码器、编码器解码器和仅有解码器），并进行了广泛的实证评估以验证其有效性。实验结果表明，带有我们适配器模块的增强模型在文本生成图像的质量和可靠性方面优于最先进的模型。 

---
# Mediator: Memory-efficient LLM Merging with Less Parameter Conflicts and Uncertainty Based Routing 

**Title (ZH)**: 调解者：基于较少参数冲突和基于路由不确定性的方法的高效内存LLM合并 

**Authors**: Kunfeng Lai, Zhenheng Tang, Xinglin Pan, Peijie Dong, Xiang Liu, Haolan Chen, Li Shen, Bo Li, Xiaowen Chu  

**Link**: [PDF](https://arxiv.org/pdf/2502.04411)  

**Abstract**: Model merging aggregates Large Language Models (LLMs) finetuned on different tasks into a stronger one. However, parameter conflicts between models leads to performance degradation in averaging. While model routing addresses this issue by selecting individual models during inference, it imposes excessive storage and compute costs, and fails to leverage the common knowledge from different models. In this work, we observe that different layers exhibit varying levels of parameter conflicts. Building on this insight, we average layers with minimal parameter conflicts and use a novel task-level expert routing for layers with significant conflicts. To further reduce storage costs, inspired by task arithmetic sparsity, we decouple multiple fine-tuned experts into a dense expert and several sparse experts. Considering the out-of-distribution samples, we select and merge appropriate experts based on the task uncertainty of the input data. We conduct extensive experiments on both LLaMA and Qwen with varying parameter scales, and evaluate on real-world reasoning tasks. Results demonstrate that our method consistently achieves significant performance improvements while requiring less system cost compared to existing methods. 

**Abstract (ZH)**: 将大型语言模型（LLMs）在不同任务上fine-tuned后的模型进行融合以生成一个更强的模型。然而，模型参数之间的冲突会导致平均化性能下降。通过在推理时选择个别模型来解决这一问题的模型路由方法虽然可以解决性能下降的问题，但带来了过高的存储和计算成本，并且无法充分利用不同模型之间的共通知识。在本文中，我们观察到不同层之间的参数冲突程度不同。在此基础上，我们对参数冲突较少的层进行平均化处理，并对参数冲突显著的层采用一种新颖的任务级专家路由机制。为了进一步降低存储成本，我们受到任务算术稀疏性的启发，将多个fine-tuned专家解耦为一个密集专家和若干稀疏专家。考虑到非分布样本，我们根据输入数据的任务不确定性选择并融合适当的专家。我们在LLaMA和Qwen上进行了一系列参数规模不同的实验，并在实际推理任务上进行评估。实验结果表明，与现有方法相比，我们的方法在系统成本较低的同时，能够实现显著的性能提升。 

---
# FAS: Fast ANN-SNN Conversion for Spiking Large Language Models 

**Title (ZH)**: FAS：快速的 ANN-SNN 转换方法用于突触大型语言模型 

**Authors**: Long Chen, Xiaotian Song, Andy Song, BaDong Chen, Jiancheng Lv, Yanan Sun  

**Link**: [PDF](https://arxiv.org/pdf/2502.04405)  

**Abstract**: Spiking Large Language Models have been shown as a good alternative to LLMs in various scenarios. Existing methods for creating Spiking LLMs, i.e., direct training and ANN-SNN conversion, often suffer from performance degradation and relatively high computational costs. To address these issues, we propose a novel Fast ANN-SNN conversion strategy (FAS) that transforms LLMs into spiking LLMs in two stages. The first stage employs a full-parameter fine-tuning of pre-trained models, so it does not need any direct training from scratch. The second stage introduces a coarse-to-fine calibration method to reduce conversion errors and improve accuracy. Our experiments on both language and vision-language tasks across four different scales of LLMs demonstrate that FAS can achieve state-of-the-art performance yet with significantly reduced inference latency and computational costs. For example, FAS only takes 8 timesteps to achieve an accuracy of 3% higher than that of the OPT-7B model, while reducing energy consumption by 96.63%. 

**Abstract (ZH)**: 基于脉冲的大规模语言模型已经在各种场景中显示出是大规模语言模型的良好替代方案。现有的创建基于脉冲的大规模语言模型（Spiking LLMs）的方法，即直接训练和ANN到SNN的转换，常常遭受性能下降和相对较高的计算成本问题。为了解决这些问题，我们提出了一种新颖的快速ANN到SNN转换策略（FAS），该策略分两阶段将大规模语言模型转换为基于脉冲的大规模语言模型。第一阶段采用全参数微调预训练模型，因此不需要从头开始直接训练。第二阶段引入了一种粗到细的校准方法，以减少转换错误并提高准确性。我们在四个不同规模的LLM（包括语言和视觉语言任务）上的实验表明，FAS可以实现最先进的性能，同时显著减少推理延迟和计算成本。例如，FAS只需要8个时隙就能实现比OPT-7B模型高3%的准确性，同时降低能耗96.63%。 

---
# Can Large Language Models Capture Video Game Engagement? 

**Title (ZH)**: 大型语言模型能够捕捉到视频游戏的参与度吗？ 

**Authors**: David Melhart, Matthew Barthet, Georgios N. Yannakakis  

**Link**: [PDF](https://arxiv.org/pdf/2502.04379)  

**Abstract**: Can out-of-the-box pretrained Large Language Models (LLMs) detect human affect successfully when observing a video? To address this question, for the first time, we evaluate comprehensively the capacity of popular LLMs to annotate and successfully predict continuous affect annotations of videos when prompted by a sequence of text and video frames in a multimodal fashion. Particularly in this paper, we test LLMs' ability to correctly label changes of in-game engagement in 80 minutes of annotated videogame footage from 20 first-person shooter games of the GameVibe corpus. We run over 2,400 experiments to investigate the impact of LLM architecture, model size, input modality, prompting strategy, and ground truth processing method on engagement prediction. Our findings suggest that while LLMs rightfully claim human-like performance across multiple domains, they generally fall behind capturing continuous experience annotations provided by humans. We examine some of the underlying causes for the relatively poor overall performance, highlight the cases where LLMs exceed expectations, and draw a roadmap for the further exploration of automated emotion labelling via LLMs. 

**Abstract (ZH)**: 预训练的大语言模型（LLMs）在观察视频时能否成功检测到人类的情感？为回答这一问题，我们首次全面评估了流行LLMs在受到文本和视频帧序列提示时，以多模态方式标注和预测视频连续情感标签的能力。在本文中，我们测试了LLMs在20款第一人称射击游戏的GameVibe语料库中80分钟标注视频片段中游戏参与度变化的正确标注能力。我们进行了超过2,400次实验，研究了LLM架构、模型大小、输入模态、提示策略和真实标注处理方法对参与度预测的影响。我们的研究结果表明，尽管LLMs在多个领域表现出类似人类的表现，但在捕捉由人类提供的连续体验标签方面通常表现不佳。我们探讨了这种相对较低的整体性能的部分原因，指出了LLMs超出预期的案例，并勾画了一条通过LLMs进一步探索自动化情感标注的路线图。 

---
# PerPO: Perceptual Preference Optimization via Discriminative Rewarding 

**Title (ZH)**: PerPO：基于辨别奖励的感知偏好优化 

**Authors**: Zining Zhu, Liang Zhao, Kangheng Lin, Jinze Yang, En Yu, Chenglong Liu, Haoran Wei, Jianjian Sun, Zheng Ge, Xiangyu Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2502.04371)  

**Abstract**: This paper presents Perceptual Preference Optimization (PerPO), a perception alignment method aimed at addressing the visual discrimination challenges in generative pre-trained multimodal large language models (MLLMs). To align MLLMs with human visual perception process, PerPO employs discriminative rewarding to gather diverse negative samples, followed by listwise preference optimization to rank this http URL utilizing the reward as a quantitative margin for ranking, our method effectively bridges generative preference optimization and discriminative empirical risk minimization. PerPO significantly enhances MLLMs' visual discrimination capabilities while maintaining their generative strengths, mitigates image-unconditional reward hacking, and ensures consistent performance across visual tasks. This work marks a crucial step towards more perceptually aligned and versatile MLLMs. We also hope that PerPO will encourage the community to rethink MLLM alignment strategies. 

**Abstract (ZH)**: 本文介绍了一种感知偏好优化（PerPO）方法，这是一种旨在解决生成预训练多模态大型语言模型（MLLMs）中视觉辨别挑战的感知对齐方法。为了使MLLMs与人类的视觉感知过程相一致，PerPO通过对具有多样性的负样本进行鉴别奖励来收集负样本，随后通过列表偏好优化对其进行排序。利用奖励作为排序的量化差距，我们的方法有效衔接了生成偏好优化和鉴别经验风险最小化。PerPO显著增强了MLLMs的视觉辨别能力，同时保持了它们的生成优势，缓解了图像无关的奖励作弊问题，并确保在视觉任务中的一致表现。这项工作标志着向更具感知对齐和多功能性的MLLMs迈出重要一步。我们还希望PerPO能够促使社区重新考虑MLLMs的对齐策略。 

---
# Getting More Juice Out of Your Data: Hard Pair Refinement Enhances Visual-Language Models Without Extra Data 

**Title (ZH)**: 充分利用你的数据：难配对精炼提升视觉语言模型而无需额外数据 

**Authors**: Haonan Wang, Minbin Huang, Runhui Huang, Lanqing Hong, Hang Xu, Tianyang Hu, Xiaodan Liang, Zhenguo Li, Hong Cheng, Kenji Kawaguchi  

**Link**: [PDF](https://arxiv.org/pdf/2305.05208)  

**Abstract**: Contrastive Language-Image Pre-training (CLIP) has become the standard for cross-modal image-text representation learning. Improving CLIP typically requires additional data and retraining with new loss functions, but these demands raise resource and time costs, limiting practical use. In this work, we introduce HELIP, a cost-effective strategy that improves CLIP models by exploiting challenging text-image pairs within existing datasets in continuous training. This eliminates the need for additional data or extensive retraining. Moreover, HELIP integrates effortlessly into current training pipelines with minimal code modifications, allowing for quick and seamless implementation. On comprehensive benchmarks, HELIP consistently boosts existing models. In particular, within just two epochs of training, it improves zero-shot classification accuracy on ImageNet for SLIP models pre-trained on CC3M, CC12M, and YFCC15M datasets by 3.05%, 4.47%, and 10.1% , respectively. In addition, on fine-grained classification datasets, HELIP improves the zero-shot performance of CLIP and SLIP by an average of 8.4% and 18.6%, and their linear probe performance by an average of 9.5% and 3.0%. The code is publicly available at: this https URL. 

**Abstract (ZH)**: 对比语言-图像预训练（CLIP）已成为跨模态图像-文本表示学习的标准。改进CLIP通常需要更多的数据和使用新的损失函数进行重新训练，但这些需求增加了资源和时间成本，限制了其实际应用。在本研究中，我们引入了HELIP策略，这是一种成本效益高的方法，通过在现有数据集中利用具有挑战性的文本-图像对进行连续训练来提升CLIP模型。这种方法消除了对额外数据或大量重新训练的需求。此外，HELIP可以轻松集成到当前的训练管道中，只需要少量的代码修改，从而实现快速且无缝的实施。在全面的基准测试中，HELIP始终能够提升现有模型的性能。特别是在仅仅两个训练周期后，HELIP分别将SLIP模型在CC3M、CC12M和YFCC15M数据集上预训练的零样本分类准确率提升了3.05%、4.47%和10.1%。此外，在细粒度分类数据集上，HELIP分别将CLIP和SLIP的零样本表现提升了8.4%和18.6%，以及线性探针性能提升了9.5%和3.0%。代码已在以下网址公开：这个链接。 

---
