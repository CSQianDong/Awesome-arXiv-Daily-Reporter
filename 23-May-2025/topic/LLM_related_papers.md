# Beyond Correlation: Towards Causal Large Language Model Agents in Biomedicine 

**Authors**: Adib Bazgir, Amir Habibdoust Lafmajani, Yuwen Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2505.16982)  

**Abstract**: Large Language Models (LLMs) show promise in biomedicine but lack true causal understanding, relying instead on correlations. This paper envisions causal LLM agents that integrate multimodal data (text, images, genomics, etc.) and perform intervention-based reasoning to infer cause-and-effect. Addressing this requires overcoming key challenges: designing safe, controllable agentic frameworks; developing rigorous benchmarks for causal evaluation; integrating heterogeneous data sources; and synergistically combining LLMs with structured knowledge (KGs) and formal causal inference tools. Such agents could unlock transformative opportunities, including accelerating drug discovery through automated hypothesis generation and simulation, enabling personalized medicine through patient-specific causal models. This research agenda aims to foster interdisciplinary efforts, bridging causal concepts and foundation models to develop reliable AI partners for biomedical progress. 

---
# HyGenar: An LLM-Driven Hybrid Genetic Algorithm for Few-Shot Grammar Generation 

**Authors**: Weizhi Tang, Yixuan Li, Chris Sypherd, Elizabeth Polgreen, Vaishak Belle  

**Link**: [PDF](https://arxiv.org/pdf/2505.16978)  

**Abstract**: Grammar plays a critical role in natural language processing and text/code generation by enabling the definition of syntax, the creation of parsers, and guiding structured outputs. Although large language models (LLMs) demonstrate impressive capabilities across domains, their ability to infer and generate grammars has not yet been thoroughly explored. In this paper, we aim to study and improve the ability of LLMs for few-shot grammar generation, where grammars are inferred from sets of a small number of positive and negative examples and generated in Backus-Naur Form. To explore this, we introduced a novel dataset comprising 540 structured grammar generation challenges, devised 6 metrics, and evaluated 8 various LLMs against it. Our findings reveal that existing LLMs perform sub-optimally in grammar generation. To address this, we propose an LLM-driven hybrid genetic algorithm, namely HyGenar, to optimize grammar generation. HyGenar achieves substantial improvements in both the syntactic and semantic correctness of generated grammars across LLMs. 

---
# AGENTIF: Benchmarking Instruction Following of Large Language Models in Agentic Scenarios 

**Authors**: Yunjia Qi, Hao Peng, Xiaozhi Wang, Amy Xin, Youfeng Liu, Bin Xu, Lei Hou, Juanzi Li  

**Link**: [PDF](https://arxiv.org/pdf/2505.16944)  

**Abstract**: Large Language Models (LLMs) have demonstrated advanced capabilities in real-world agentic applications. Growing research efforts aim to develop LLM-based agents to address practical demands, introducing a new challenge: agentic scenarios often involve lengthy instructions with complex constraints, such as extended system prompts and detailed tool specifications. While adherence to such instructions is crucial for agentic applications, whether LLMs can reliably follow them remains underexplored. In this paper, we introduce AgentIF, the first benchmark for systematically evaluating LLM instruction following ability in agentic scenarios. AgentIF features three key characteristics: (1) Realistic, constructed from 50 real-world agentic applications. (2) Long, averaging 1,723 words with a maximum of 15,630 words. (3) Complex, averaging 11.9 constraints per instruction, covering diverse constraint types, such as tool specifications and condition constraints. To construct AgentIF, we collect 707 human-annotated instructions across 50 agentic tasks from industrial application agents and open-source agentic systems. For each instruction, we annotate the associated constraints and corresponding evaluation metrics, including code-based evaluation, LLM-based evaluation, and hybrid code-LLM evaluation. We use AgentIF to systematically evaluate existing advanced LLMs. We observe that current models generally perform poorly, especially in handling complex constraint structures and tool specifications. We further conduct error analysis and analytical experiments on instruction length and meta constraints, providing some findings about the failure modes of existing LLMs. We have released the code and data to facilitate future research. 

---
# KTAE: A Model-Free Algorithm to Key-Tokens Advantage Estimation in Mathematical Reasoning 

**Authors**: Wei Sun, Wen Yang, Pu Jian, Qianlong Du, Fuwei Cui, Shuo Ren, Jiajun Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2505.16826)  

**Abstract**: Recent advances have demonstrated that integrating reinforcement learning with rule-based rewards can significantly enhance the reasoning capabilities of large language models, even without supervised fine-tuning. However, prevalent reinforcement learning algorithms such as GRPO and its variants like DAPO, suffer from a coarse granularity issue when computing the advantage. Specifically, they compute rollout-level advantages that assign identical values to every token within a sequence, failing to capture token-specific contributions and hindering effective learning. To address this limitation, we propose Key-token Advantage Estimation (KTAE) - a novel algorithm that estimates fine-grained, token-level advantages without introducing additional models. KTAE leverages the correctness of sampled rollouts and applies statistical analysis to quantify the importance of individual tokens within a sequence to the final outcome. This quantified token-level importance is then combined with the rollout-level advantage to obtain a more fine-grained token-level advantage estimation. Empirical results show that models trained with GRPO+KTAE and DAPO+KTAE outperform baseline methods across five mathematical reasoning benchmarks. Notably, they achieve higher accuracy with shorter responses and even surpass R1-Distill-Qwen-1.5B using the same base model. 

---
# MCP-RADAR: A Multi-Dimensional Benchmark for Evaluating Tool Use Capabilities in Large Language Models 

**Authors**: Xuanqi Gao, Siyi Xie, Juan Zhai, Shqing Ma, Chao Shen  

**Link**: [PDF](https://arxiv.org/pdf/2505.16700)  

**Abstract**: As Large Language Models (LLMs) evolve from passive text generators to active reasoning agents capable of tool interaction, the Model Context Protocol (MCP) has emerged as a standardized framework for dynamic tool discovery and orchestration. Despite widespread industry adoption, existing evaluation methodologies fail to adequately assess tool utilization capabilities within this new paradigm. This paper introduces MCP-RADAR, the first comprehensive benchmark specifically designed to evaluate LLM performance in the MCP framework through a novel five-dimensional approach measuring: answer accuracy, tool selection efficiency, computational resource efficiency, parameter construction accuracy, and execution speed. Unlike conventional benchmarks that rely on subjective human evaluations or binary success metrics, MCP-RADAR employs objective, quantifiable measurements across multiple task domains including software engineering, mathematical reasoning, and general problem-solving. Our evaluations of leading commercial and open-source LLMs reveal distinctive capability profiles with significant trade-offs between accuracy, efficiency, and speed, challenging traditional single-metric performance rankings. Besides, we provide valuable guidance for developers to optimize their tools for maximum model compatibility and effectiveness. While focused on MCP due to its standardized approach, our methodology remains applicable across all LLM agent tool integration frameworks, providing valuable insights for both LLM developers and tool creators to optimize the entire LLM-tool interaction ecosystem. The implementation, configurations, and datasets used in our evaluation are publicly available at this https URL. 

---
# X-MAS: Towards Building Multi-Agent Systems with Heterogeneous LLMs 

**Authors**: Rui Ye, Xiangrui Liu, Qimin Wu, Xianghe Pang, Zhenfei Yin, Lei Bai, Siheng Chen  

**Link**: [PDF](https://arxiv.org/pdf/2505.16997)  

**Abstract**: LLM-based multi-agent systems (MAS) extend the capabilities of single LLMs by enabling cooperation among multiple specialized agents. However, most existing MAS frameworks rely on a single LLM to drive all agents, constraining the system's intelligence to the limit of that model. This paper explores the paradigm of heterogeneous LLM-driven MAS (X-MAS), where agents are powered by diverse LLMs, elevating the system's potential to the collective intelligence of diverse LLMs. We introduce X-MAS-Bench, a comprehensive testbed designed to evaluate the performance of various LLMs across different domains and MAS-related functions. As an extensive empirical study, we assess 27 LLMs across 5 domains (encompassing 21 test sets) and 5 functions, conducting over 1.7 million evaluations to identify optimal model selections for each domain-function combination. Building on these findings, we demonstrate that transitioning from homogeneous to heterogeneous LLM-driven MAS can significantly enhance system performance without requiring structural redesign. Specifically, in a chatbot-only MAS scenario, the heterogeneous configuration yields up to 8.4\% performance improvement on the MATH dataset. In a mixed chatbot-reasoner scenario, the heterogeneous MAS could achieve a remarkable 47\% performance boost on the AIME dataset. Our results underscore the transformative potential of heterogeneous LLMs in MAS, highlighting a promising avenue for advancing scalable, collaborative AI systems. 

---
# Bridging the Dynamic Perception Gap: Training-Free Draft Chain-of-Thought for Dynamic Multimodal Spatial Reasoning 

**Authors**: Siqu Ou, Hongcheng Liu, Pingjie Wang, Yusheng Liao, Chuan Xuan, Yanfeng Wang, Yu Wang  

**Link**: [PDF](https://arxiv.org/pdf/2505.16579)  

**Abstract**: While chains-of-thought (CoT) have advanced complex reasoning in multimodal large language models (MLLMs), existing methods remain confined to text or static visual domains, often faltering in dynamic spatial reasoning tasks. To bridge this gap, we present GRASSLAND, a novel maze navigation benchmark designed to evaluate dynamic spatial reasoning. Our experiments show that augmenting textual reasoning chains with dynamic visual drafts, overlaid on input images, significantly outperforms conventional approaches, offering new insights into spatial reasoning in evolving environments. To generalize this capability, we propose D2R (Dynamic Draft-Augmented Reasoning), a training-free framework that seamlessly integrates textual CoT with corresponding visual drafts into MLLMs. Extensive evaluations demonstrate that D2R consistently enhances performance across diverse tasks, establishing a robust baseline for dynamic spatial reasoning without requiring model fine-tuning. Project is open at this https URL. 

---
# SMART: Self-Generating and Self-Validating Multi-Dimensional Assessment for LLMs' Mathematical Problem Solving 

**Authors**: Yujie Hou, Ting Zhang, Mei Wang, Xuetao Ma, Hu Huang  

**Link**: [PDF](https://arxiv.org/pdf/2505.16646)  

**Abstract**: Large Language Models have achieved remarkable results on a variety of mathematical benchmarks. However, concerns remain as to whether these successes reflect genuine mathematical reasoning or superficial pattern recognition. Common evaluation metrics, such as final answer accuracy, fail to disentangle the underlying competencies involved, offering limited diagnostic value. To address these limitations, we introduce SMART: a Self-Generating and Self-Validating Multi-Dimensional Assessment Framework. SMART decomposes mathematical problem solving into four distinct dimensions: understanding, reasoning, arithmetic, and reflection \& refinement. Each dimension is evaluated independently through tailored tasks, enabling interpretable and fine-grained analysis of LLM behavior. Crucially, SMART integrates an automated self-generating and self-validating mechanism to produce and verify benchmark data, ensuring both scalability and reliability. We apply SMART to 21 state-of-the-art open- and closed-source LLMs, uncovering significant discrepancies in their abilities across different dimensions. Our findings demonstrate the inadequacy of final answer accuracy as a sole metric and motivate a new holistic metric to better capture true problem-solving capabilities. Code and benchmarks will be released upon acceptance. 

---
# ELABORATION: A Comprehensive Benchmark on Human-LLM Competitive Programming 

**Authors**: Xinwei Yang, Zhaofeng Liu, Chen Huang, Jiashuai Zhang, Tong Zhang, Yifan Zhang, Wenqiang Lei  

**Link**: [PDF](https://arxiv.org/pdf/2505.16667)  

**Abstract**: While recent research increasingly emphasizes the value of human-LLM collaboration in competitive programming and proposes numerous empirical methods, a comprehensive understanding remains elusive due to the fragmented nature of existing studies and their use of diverse, application-specific human feedback. Thus, our work serves a three-fold purpose: First, we present the first taxonomy of human feedback consolidating the entire programming process, which promotes fine-grained evaluation. Second, we introduce ELABORATIONSET, a novel programming dataset specifically designed for human-LLM collaboration, meticulously annotated to enable large-scale simulated human feedback and facilitate costeffective real human interaction studies. Third, we introduce ELABORATION, a novel benchmark to facilitate a thorough assessment of human-LLM competitive programming. With ELABORATION, we pinpoint strengthes and weaknesses of existing methods, thereby setting the foundation for future improvement. Our code and dataset are available at this https URL 

---
# Advancing the Scientific Method with Large Language Models: From Hypothesis to Discovery 

**Authors**: Yanbo Zhang, Sumeer A. Khan, Adnan Mahmud, Huck Yang, Alexander Lavin, Michael Levin, Jeremy Frey, Jared Dunnmon, James Evans, Alan Bundy, Saso Dzeroski, Jesper Tegner, Hector Zenil  

**Link**: [PDF](https://arxiv.org/pdf/2505.16477)  

**Abstract**: With recent Nobel Prizes recognising AI contributions to science, Large Language Models (LLMs) are transforming scientific research by enhancing productivity and reshaping the scientific method. LLMs are now involved in experimental design, data analysis, and workflows, particularly in chemistry and biology. However, challenges such as hallucinations and reliability persist. In this contribution, we review how Large Language Models (LLMs) are redefining the scientific method and explore their potential applications across different stages of the scientific cycle, from hypothesis testing to discovery. We conclude that, for LLMs to serve as relevant and effective creative engines and productivity enhancers, their deep integration into all steps of the scientific process should be pursued in collaboration and alignment with human scientific goals, with clear evaluation metrics. The transition to AI-driven science raises ethical questions about creativity, oversight, and responsibility. With careful guidance, LLMs could evolve into creative engines, driving transformative breakthroughs across scientific disciplines responsibly and effectively. However, the scientific community must also decide how much it leaves to LLMs to drive science, even when associations with 'reasoning', mostly currently undeserved, are made in exchange for the potential to explore hypothesis and solution regions that might otherwise remain unexplored by human exploration alone. 

---
# ReflectEvo: Improving Meta Introspection of Small LLMs by Learning Self-Reflection 

**Authors**: Jiaqi Li, Xinyi Dong, Yang Liu, Zhizhuo Yang, Quansen Wang, Xiaobo Wang, SongChun Zhu, Zixia Jia, Zilong Zheng  

**Link**: [PDF](https://arxiv.org/pdf/2505.16475)  

**Abstract**: We present a novel pipeline, ReflectEvo, to demonstrate that small language models (SLMs) can enhance meta introspection through reflection learning. This process iteratively generates self-reflection for self-training, fostering a continuous and self-evolving process. Leveraging this pipeline, we construct ReflectEvo-460k, a large-scale, comprehensive, self-generated reflection dataset with broadened instructions and diverse multi-domain tasks. Building upon this dataset, we demonstrate the effectiveness of reflection learning to improve SLMs' reasoning abilities using SFT and DPO with remarkable performance, substantially boosting Llama-3 from 52.4% to 71.2% and Mistral from 44.4% to 71.1%. It validates that ReflectEvo can rival or even surpass the reasoning capability of the three prominent open-sourced models on BIG-bench without distillation from superior models or fine-grained human annotation. We further conduct a deeper analysis of the high quality of self-generated reflections and their impact on error localization and correction. Our work highlights the potential of continuously enhancing the reasoning performance of SLMs through iterative reflection learning in the long run. 

---
# MMMR: Benchmarking Massive Multi-Modal Reasoning Tasks 

**Authors**: Guiyao Tie, Xueyang Zhou, Tianhe Gu, Ruihang Zhang, Chaoran Hu, Sizhe Zhang, Mengqu Sun, Yan Zhang, Pan Zhou, Lichao Sun  

**Link**: [PDF](https://arxiv.org/pdf/2505.16459)  

**Abstract**: Recent advances in Multi-Modal Large Language Models (MLLMs) have enabled unified processing of language, vision, and structured inputs, opening the door to complex tasks such as logical deduction, spatial reasoning, and scientific analysis. Despite their promise, the reasoning capabilities of MLLMs, particularly those augmented with intermediate thinking traces (MLLMs-T), remain poorly understood and lack standardized evaluation benchmarks. Existing work focuses primarily on perception or final answer correctness, offering limited insight into how models reason or fail across modalities. To address this gap, we introduce the MMMR, a new benchmark designed to rigorously evaluate multi-modal reasoning with explicit thinking. The MMMR comprises 1) a high-difficulty dataset of 1,083 questions spanning six diverse reasoning types with symbolic depth and multi-hop demands and 2) a modular Reasoning Trace Evaluation Pipeline (RTEP) for assessing reasoning quality beyond accuracy through metrics like relevance, consistency, and structured error annotations. Empirical results show that MLLMs-T overall outperform non-thinking counterparts, but even top models like Claude-3.7-Sonnet and Gemini-2.5 Pro suffer from reasoning pathologies such as inconsistency and overthinking. This benchmark reveals persistent gaps between accuracy and reasoning quality and provides an actionable evaluation pipeline for future model development. Overall, the MMMR offers a scalable foundation for evaluating, comparing, and improving the next generation of multi-modal reasoning systems. 

---
# EquivPruner: Boosting Efficiency and Quality in LLM-Based Search via Action Pruning 

**Authors**: Jiawei Liu, Qisi Chen, Jianshu Zhang, Quan Liu, Defu Lian  

**Link**: [PDF](https://arxiv.org/pdf/2505.16312)  

**Abstract**: Large Language Models (LLMs) excel at complex reasoning through search algorithms, yet current strategies often suffer from massive token consumption due to redundant exploration of semantically equivalent steps. Existing semantic similarity methods struggle to accurately identify such equivalence in domain-specific contexts like mathematical reasoning. To address this, we propose EquivPruner, a simple yet effective approach that identifies and prunes semantically equivalent actions during LLM reasoning search. We also introduce MathEquiv, the first dataset we created for mathematical statement equivalence, which enables the training of a lightweight equivalence detector. Extensive experiments across various models and tasks demonstrate that EquivPruner significantly reduces token consumption, improving searching efficiency and often bolstering reasoning accuracy. For instance, when applied to Qwen2.5-Math-7B-Instruct on GSM8K, EquivPruner reduced token consumption by 48.1\% while also improving accuracy. Our code is available at this https URL. 

---
# How do Scaling Laws Apply to Knowledge Graph Engineering Tasks? The Impact of Model Size on Large Language Model Performance 

**Authors**: Desiree Heim, Lars-Peter Meyer, Markus Schröder, Johannes Frey, Andreas Dengel  

**Link**: [PDF](https://arxiv.org/pdf/2505.16276)  

**Abstract**: When using Large Language Models (LLMs) to support Knowledge Graph Engineering (KGE), one of the first indications when searching for an appropriate model is its size. According to the scaling laws, larger models typically show higher capabilities. However, in practice, resource costs are also an important factor and thus it makes sense to consider the ratio between model performance and costs. The LLM-KG-Bench framework enables the comparison of LLMs in the context of KGE tasks and assesses their capabilities of understanding and producing KGs and KG queries. Based on a dataset created in an LLM-KG-Bench run covering 26 open state-of-the-art LLMs, we explore the model size scaling laws specific to KGE tasks. In our analyses, we assess how benchmark scores evolve between different model size categories. Additionally, we inspect how the general score development of single models and families of models correlates to their size. Our analyses revealed that, with a few exceptions, the model size scaling laws generally also apply to the selected KGE tasks. However, in some cases, plateau or ceiling effects occurred, i.e., the task performance did not change much between a model and the next larger model. In these cases, smaller models could be considered to achieve high cost-effectiveness. Regarding models of the same family, sometimes larger models performed worse than smaller models of the same family. These effects occurred only locally. Hence it is advisable to additionally test the next smallest and largest model of the same family. 

---
# SafeKey: Amplifying Aha-Moment Insights for Safety Reasoning 

**Authors**: Kaiwen Zhou, Xuandong Zhao, Gaowen Liu, Jayanth Srinivasa, Aosong Feng, Dawn Song, Xin Eric Wang  

**Link**: [PDF](https://arxiv.org/pdf/2505.16186)  

**Abstract**: Large Reasoning Models (LRMs) introduce a new generation paradigm of explicitly reasoning before answering, leading to remarkable improvements in complex tasks. However, they pose great safety risks against harmful queries and adversarial attacks. While recent mainstream safety efforts on LRMs, supervised fine-tuning (SFT), improve safety performance, we find that SFT-aligned models struggle to generalize to unseen jailbreak prompts. After thorough investigation of LRMs' generation, we identify a safety aha moment that can activate safety reasoning and lead to a safe response. This aha moment typically appears in the `key sentence', which follows models' query understanding process and can indicate whether the model will proceed safely. Based on these insights, we propose SafeKey, including two complementary objectives to better activate the safety aha moment in the key sentence: (1) a Dual-Path Safety Head to enhance the safety signal in the model's internal representations before the key sentence, and (2) a Query-Mask Modeling objective to improve the models' attention on its query understanding, which has important safety hints. Experiments across multiple safety benchmarks demonstrate that our methods significantly improve safety generalization to a wide range of jailbreak attacks and out-of-distribution harmful prompts, lowering the average harmfulness rate by 9.6\%, while maintaining general abilities. Our analysis reveals how SafeKey enhances safety by reshaping internal attention and improving the quality of hidden representations. 

---
# LLM-Powered AI Agent Systems and Their Applications in Industry 

**Authors**: Guannan Liang, Qianqian Tong  

**Link**: [PDF](https://arxiv.org/pdf/2505.16120)  

**Abstract**: The emergence of Large Language Models (LLMs) has reshaped agent systems. Unlike traditional rule-based agents with limited task scope, LLM-powered agents offer greater flexibility, cross-domain reasoning, and natural language interaction. Moreover, with the integration of multi-modal LLMs, current agent systems are highly capable of processing diverse data modalities, including text, images, audio, and structured tabular data, enabling richer and more adaptive real-world behavior. This paper comprehensively examines the evolution of agent systems from the pre-LLM era to current LLM-powered architectures. We categorize agent systems into software-based, physical, and adaptive hybrid systems, highlighting applications across customer service, software development, manufacturing automation, personalized education, financial trading, and healthcare. We further discuss the primary challenges posed by LLM-powered agents, including high inference latency, output uncertainty, lack of evaluation metrics, and security vulnerabilities, and propose potential solutions to mitigate these concerns. 

---
# Logic-of-Thought: Empowering Large Language Models with Logic Programs for Solving Puzzles in Natural Language 

**Authors**: Naiqi Li, Peiyuan Liu, Zheng Liu, Tao Dai, Yong Jiang, Shu-Tao Xia  

**Link**: [PDF](https://arxiv.org/pdf/2505.16114)  

**Abstract**: Solving puzzles in natural language poses a long-standing challenge in AI. While large language models (LLMs) have recently shown impressive capabilities in a variety of tasks, they continue to struggle with complex puzzles that demand precise reasoning and exhaustive search. In this paper, we propose Logic-of-Thought (Logot), a novel framework that bridges LLMs with logic programming to address this problem. Our method leverages LLMs to translate puzzle rules and states into answer set programs (ASPs), the solution of which are then accurately and efficiently inferred by an ASP interpreter. This hybrid approach combines the natural language understanding of LLMs with the precise reasoning capabilities of logic programs. We evaluate our method on various grid puzzles and dynamic puzzles involving actions, demonstrating near-perfect accuracy across all tasks. Our code and data are available at: this https URL. 

---
# TrialPanorama: Database and Benchmark for Systematic Review and Design of Clinical Trials 

**Authors**: Zifeng Wang, Qiao Jin, Jiacheng Lin, Junyi Gao, Jathurshan Pradeepkumar, Pengcheng Jiang, Benjamin Danek, Zhiyong Lu, Jimeng Sun  

**Link**: [PDF](https://arxiv.org/pdf/2505.16097)  

**Abstract**: Developing artificial intelligence (AI) for vertical domains requires a solid data foundation for both training and evaluation. In this work, we introduce TrialPanorama, a large-scale, structured database comprising 1,657,476 clinical trial records aggregated from 15 global sources. The database captures key aspects of trial design and execution, including trial setups, interventions, conditions, biomarkers, and outcomes, and links them to standard biomedical ontologies such as DrugBank and MedDRA. This structured and ontology-grounded design enables TrialPanorama to serve as a unified, extensible resource for a wide range of clinical trial tasks, including trial planning, design, and summarization. To demonstrate its utility, we derive a suite of benchmark tasks directly from the TrialPanorama database. The benchmark spans eight tasks across two categories: three for systematic review (study search, study screening, and evidence summarization) and five for trial design (arm design, eligibility criteria, endpoint selection, sample size estimation, and trial completion assessment). The experiments using five state-of-the-art large language models (LLMs) show that while general-purpose LLMs exhibit some zero-shot capability, their performance is still inadequate for high-stakes clinical trial workflows. We release TrialPanorama database and the benchmark to facilitate further research on AI for clinical trials. 

---
# Can AI Read Between The Lines? Benchmarking LLMs On Financial Nuance 

**Authors**: Dominick Kubica, Dylan T. Gordon, Nanami Emura, Derleen Saini, Charlie Goldenberg  

**Link**: [PDF](https://arxiv.org/pdf/2505.16090)  

**Abstract**: As of 2025, Generative Artificial Intelligence (GenAI) has become a central tool for productivity across industries. Beyond text generation, GenAI now plays a critical role in coding, data analysis, and research workflows. As large language models (LLMs) continue to evolve, it is essential to assess the reliability and accuracy of their outputs, especially in specialized, high-stakes domains like finance. Most modern LLMs transform text into numerical vectors, which are used in operations such as cosine similarity searches to generate responses. However, this abstraction process can lead to misinterpretation of emotional tone, particularly in nuanced financial contexts. While LLMs generally excel at identifying sentiment in everyday language, these models often struggle with the nuanced, strategically ambiguous language found in earnings call transcripts. Financial disclosures frequently embed sentiment in hedged statements, forward-looking language, and industry-specific jargon, making it difficult even for human analysts to interpret consistently, let alone AI models. This paper presents findings from the Santa Clara Microsoft Practicum Project, led by Professor Charlie Goldenberg, which benchmarks the performance of Microsoft's Copilot, OpenAI's ChatGPT, Google's Gemini, and traditional machine learning models for sentiment analysis of financial text. Using Microsoft earnings call transcripts, the analysis assesses how well LLM-derived sentiment correlates with market sentiment and stock movements and evaluates the accuracy of model outputs. Prompt engineering techniques are also examined to improve sentiment analysis results. Visualizations of sentiment consistency are developed to evaluate alignment between tone and stock performance, with sentiment trends analyzed across Microsoft's lines of business to determine which segments exert the greatest influence. 

---
# Optimizing LLM-Based Multi-Agent System with Textual Feedback: A Case Study on Software Development 

**Authors**: Ming Shen, Raphael Shu, Anurag Pratik, James Gung, Yubin Ge, Monica Sunkara, Yi Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2505.16086)  

**Abstract**: We have seen remarkable progress in large language models (LLMs) empowered multi-agent systems solving complex tasks necessitating cooperation among experts with diverse skills. However, optimizing LLM-based multi-agent systems remains challenging. In this work, we perform an empirical case study on group optimization of role-based multi-agent systems utilizing natural language feedback for challenging software development tasks under various evaluation dimensions. We propose a two-step agent prompts optimization pipeline: identifying underperforming agents with their failure explanations utilizing textual feedback and then optimizing system prompts of identified agents utilizing failure explanations. We then study the impact of various optimization settings on system performance with two comparison groups: online against offline optimization and individual against group optimization. For group optimization, we study two prompting strategies: one-pass and multi-pass prompting optimizations. Overall, we demonstrate the effectiveness of our optimization method for role-based multi-agent systems tackling software development tasks evaluated on diverse evaluation dimensions, and we investigate the impact of diverse optimization settings on group behaviors of the multi-agent systems to provide practical insights for future development. 

---
# How Memory Management Impacts LLM Agents: An Empirical Study of Experience-Following Behavior 

**Authors**: Zidi Xiong, Yuping Lin, Wenya Xie, Pengfei He, Jiliang Tang, Himabindu Lakkaraju, Zhen Xiang  

**Link**: [PDF](https://arxiv.org/pdf/2505.16067)  

**Abstract**: Memory is a critical component in large language model (LLM)-based agents, enabling them to store and retrieve past executions to improve task performance over time. In this paper, we conduct an empirical study on how memory management choices impact the LLM agents' behavior, especially their long-term performance. Specifically, we focus on two fundamental memory operations that are widely used by many agent frameworks-addition, which incorporates new experiences into the memory base, and deletion, which selectively removes past experiences-to systematically study their impact on the agent behavior. Through our quantitative analysis, we find that LLM agents display an experience-following property: high similarity between a task input and the input in a retrieved memory record often results in highly similar agent outputs. Our analysis further reveals two significant challenges associated with this property: error propagation, where inaccuracies in past experiences compound and degrade future performance, and misaligned experience replay, where outdated or irrelevant experiences negatively influence current tasks. Through controlled experiments, we show that combining selective addition and deletion strategies can help mitigate these negative effects, yielding an average absolute performance gain of 10% compared to naive memory growth. Furthermore, we highlight how memory management choices affect agents' behavior under challenging conditions such as task distribution shifts and constrained memory resources. Our findings offer insights into the behavioral dynamics of LLM agent memory systems and provide practical guidance for designing memory components that support robust, long-term agent performance. We also release our code to facilitate further study. 

---
# SPhyR: Spatial-Physical Reasoning Benchmark on Material Distribution 

**Authors**: Philipp D. Siedler  

**Link**: [PDF](https://arxiv.org/pdf/2505.16048)  

**Abstract**: We introduce a novel dataset designed to benchmark the physical and spatial reasoning capabilities of Large Language Models (LLM) based on topology optimization, a method for computing optimal material distributions within a design space under prescribed loads and supports. In this dataset, LLMs are provided with conditions such as 2D boundary, applied forces and supports, and must reason about the resulting optimal material distribution. The dataset includes a variety of tasks, ranging from filling in masked regions within partial structures to predicting complete material distributions. Solving these tasks requires understanding the flow of forces and the required material distribution under given constraints, without access to simulation tools or explicit physical models, challenging models to reason about structural stability and spatial organization. Our dataset targets the evaluation of spatial and physical reasoning abilities in 2D settings, offering a complementary perspective to traditional language and logic benchmarks. 

---
# Psychology-driven LLM Agents for Explainable Panic Prediction on Social Media during Sudden Disaster Events 

**Authors**: Mengzhu Liu, Zhengqiu Zhu, Chuan Ai, Chen Gao, Xinghong Li, Lingnan He, Kaisheng Lai, Yingfeng Chen, Xin Lu, Yong Li, Quanjun Yin  

**Link**: [PDF](https://arxiv.org/pdf/2505.16455)  

**Abstract**: During sudden disaster events, accurately predicting public panic sentiment on social media is crucial for proactive governance and crisis management. Current efforts on this problem face three main challenges: lack of finely annotated data hinders emotion prediction studies, unmodeled risk perception causes prediction inaccuracies, and insufficient interpretability of panic formation mechanisms. We address these issues by proposing a Psychology-driven generative Agent framework (PsychoAgent) for explainable panic prediction based on emotion arousal theory. Specifically, we first construct a fine-grained open panic emotion dataset (namely COPE) via human-large language models (LLMs) collaboration to mitigate semantic bias. Then, we develop a framework integrating cross-domain heterogeneous data grounded in psychological mechanisms to model risk perception and cognitive differences in emotion generation. To enhance interpretability, we design an LLM-based role-playing agent that simulates individual psychological chains through dedicatedly designed prompts. Experimental results on our annotated dataset show that PsychoAgent improves panic emotion prediction performance by 12.6% to 21.7% compared to baseline models. Furthermore, the explainability and generalization of our approach is validated. Crucially, this represents a paradigm shift from opaque "data-driven fitting" to transparent "role-based simulation with mechanistic interpretation" for panic emotion prediction during emergencies. Our implementation is publicly available at: this https URL. 

---
# Incentivizing Dual Process Thinking for Efficient Large Language Model Reasoning 

**Authors**: Xiaoxue Cheng, Junyi Li, Zhenduo Zhang, Xinyu Tang, Wayne Xin Zhao, Xinyu Kong, Zhiqiang Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2505.16315)  

**Abstract**: Large reasoning models (LRMs) have demonstrated strong performance on complex reasoning tasks, but often suffer from overthinking, generating redundant content regardless of task difficulty. Inspired by the dual process theory in cognitive science, we propose Adaptive Cognition Policy Optimization (ACPO), a reinforcement learning framework that enables LRMs to achieve efficient reasoning through adaptive cognitive allocation and dynamic system switch. ACPO incorporates two key components: (1) introducing system-aware reasoning tokens to explicitly represent the thinking modes thereby making the model's cognitive process transparent, and (2) integrating online difficulty estimation and token length budget to guide adaptive system switch and reasoning during reinforcement learning. To this end, we propose a two-stage training strategy. The first stage begins with supervised fine-tuning to cold start the model, enabling it to generate reasoning paths with explicit thinking modes. In the second stage, we apply ACPO to further enhance adaptive system switch for difficulty-aware reasoning. Experimental results demonstrate that ACPO effectively reduces redundant reasoning while adaptively adjusting cognitive allocation based on task complexity, achieving efficient hybrid reasoning. 

---
# Sudoku-Bench: Evaluating creative reasoning with Sudoku variants 

**Authors**: Jeffrey Seely, Yuki Imajuku, Tianyu Zhao, Edoardo Cetin, Llion Jones  

**Link**: [PDF](https://arxiv.org/pdf/2505.16135)  

**Abstract**: Existing reasoning benchmarks for large language models (LLMs) frequently fail to capture authentic creativity, often rewarding memorization of previously observed patterns. We address this shortcoming with Sudoku-Bench, a curated benchmark of challenging and unconventional Sudoku variants specifically selected to evaluate creative, multi-step logical reasoning. Sudoku variants form an unusually effective domain for reasoning research: each puzzle introduces unique or subtly interacting constraints, making memorization infeasible and requiring solvers to identify novel logical breakthroughs (``break-ins''). Despite their diversity, Sudoku variants maintain a common and compact structure, enabling clear and consistent evaluation. Sudoku-Bench includes a carefully chosen puzzle set, a standardized text-based puzzle representation, and flexible tools compatible with thousands of publicly available puzzles -- making it easy to extend into a general research environment. Baseline experiments show that state-of-the-art LLMs solve fewer than 15\% of puzzles unaided, highlighting significant opportunities to advance long-horizon, strategic reasoning capabilities. 

---
# GoT-R1: Unleashing Reasoning Capability of MLLM for Visual Generation with Reinforcement Learning 

**Authors**: Chengqi Duan, Rongyao Fang, Yuqing Wang, Kun Wang, Linjiang Huang, Xingyu Zeng, Hongsheng Li, Xihui Liu  

**Link**: [PDF](https://arxiv.org/pdf/2505.17022)  

**Abstract**: Visual generation models have made remarkable progress in creating realistic images from text prompts, yet struggle with complex prompts that specify multiple objects with precise spatial relationships and attributes. Effective handling of such prompts requires explicit reasoning about the semantic content and spatial layout. We present GoT-R1, a framework that applies reinforcement learning to enhance semantic-spatial reasoning in visual generation. Building upon the Generation Chain-of-Thought approach, GoT-R1 enables models to autonomously discover effective reasoning strategies beyond predefined templates through carefully designed reinforcement learning. To achieve this, we propose a dual-stage multi-dimensional reward framework that leverages MLLMs to evaluate both the reasoning process and final output, enabling effective supervision across the entire generation pipeline. The reward system assesses semantic alignment, spatial accuracy, and visual quality in a unified approach. Experimental results demonstrate significant improvements on T2I-CompBench benchmark, particularly in compositional tasks involving precise spatial relationships and attribute binding. GoT-R1 advances the state-of-the-art in image generation by successfully transferring sophisticated reasoning capabilities to the visual generation domain. To facilitate future research, we make our code and pretrained models publicly available at this https URL. 

---
# Do Large Language Models Excel in Complex Logical Reasoning with Formal Language? 

**Authors**: Jin Jiang, Jianing Wang, Yuchen Yan, Yang Liu, Jianhua Zhu, Mengdi Zhang, Xunliang Cai, Liangcai Gao  

**Link**: [PDF](https://arxiv.org/pdf/2505.16998)  

**Abstract**: Large Language Models (LLMs) have been shown to achieve breakthrough performance on complex logical reasoning tasks. Nevertheless, most existing research focuses on employing formal language to guide LLMs to derive reliable reasoning paths, while systematic evaluations of these capabilities are still limited. In this paper, we aim to conduct a comprehensive evaluation of LLMs across various logical reasoning problems utilizing formal languages. From the perspective of three dimensions, i.e., spectrum of LLMs, taxonomy of tasks, and format of trajectories, our key findings are: 1) Thinking models significantly outperform Instruct models, especially when formal language is employed; 2) All LLMs exhibit limitations in inductive reasoning capability, irrespective of whether they use a formal language; 3) Data with PoT format achieves the best generalization performance across other languages. Additionally, we also curate the formal-relative training data to further enhance the small language models, and the experimental results indicate that a simple rejected fine-tuning method can better enable LLMs to generalize across formal languages and achieve the best overall performance. Our codes and reports are available at this https URL. 

---
# $\text{R}^2\text{ec}$: Towards Large Recommender Models with Reasoning 

**Authors**: Runyang You, Yongqi Li, Xinyu Lin, Xin Zhang, Wenjie Wang, Wenjie Li, Liqiang Nie  

**Link**: [PDF](https://arxiv.org/pdf/2505.16994)  

**Abstract**: Large recommender models have extended LLMs as powerful recommenders via encoding or item generation, and recent breakthroughs in LLM reasoning synchronously motivate the exploration of reasoning in recommendation. Current studies usually position LLMs as external reasoning modules to yield auxiliary thought for augmenting conventional recommendation pipelines. However, such decoupled designs are limited in significant resource cost and suboptimal joint optimization. To address these issues, we propose \name, a unified large recommender model with intrinsic reasoning capabilities. Initially, we reconceptualize the model architecture to facilitate interleaved reasoning and recommendation in the autoregressive process. Subsequently, we propose RecPO, a corresponding reinforcement learning framework that optimizes \name\ both the reasoning and recommendation capabilities simultaneously in a single policy update; RecPO introduces a fused reward scheme that solely leverages recommendation labels to simulate the reasoning capability, eliminating dependency on specialized reasoning annotations. Experiments on three datasets with various baselines verify the effectiveness of \name, showing relative improvements of 68.67\% in Hit@5 and 45.21\% in NDCG@20. Code available at this https URL. 

---
# MASLab: A Unified and Comprehensive Codebase for LLM-based Multi-Agent Systems 

**Authors**: Rui Ye, Keduan Huang, Qimin Wu, Yuzhu Cai, Tian Jin, Xianghe Pang, Xiangrui Liu, Jiaqi Su, Chen Qian, Bohan Tang, Kaiqu Liang, Jiaao Chen, Yue Hu, Zhenfei Yin, Rongye Shi, Bo An, Yang Gao, Wenjun Wu, Lei Bai, Siheng Chen  

**Link**: [PDF](https://arxiv.org/pdf/2505.16988)  

**Abstract**: LLM-based multi-agent systems (MAS) have demonstrated significant potential in enhancing single LLMs to address complex and diverse tasks in practical applications. Despite considerable advancements, the field lacks a unified codebase that consolidates existing methods, resulting in redundant re-implementation efforts, unfair comparisons, and high entry barriers for researchers. To address these challenges, we introduce MASLab, a unified, comprehensive, and research-friendly codebase for LLM-based MAS. (1) MASLab integrates over 20 established methods across multiple domains, each rigorously validated by comparing step-by-step outputs with its official implementation. (2) MASLab provides a unified environment with various benchmarks for fair comparisons among methods, ensuring consistent inputs and standardized evaluation protocols. (3) MASLab implements methods within a shared streamlined structure, lowering the barriers for understanding and extension. Building on MASLab, we conduct extensive experiments covering 10+ benchmarks and 8 models, offering researchers a clear and comprehensive view of the current landscape of MAS methods. MASLab will continue to evolve, tracking the latest developments in the field, and invite contributions from the broader open-source community. 

---
# T1: A Tool-Oriented Conversational Dataset for Multi-Turn Agentic Planning 

**Authors**: Amartya Chakraborty, Paresh Dashore, Nadia Bathaee, Anmol Jain, Anirban Das, Shi-Xiong Zhang, Sambit Sahu, Milind Naphade, Genta Indra Winata  

**Link**: [PDF](https://arxiv.org/pdf/2505.16986)  

**Abstract**: Large Language Models (LLMs) have demonstrated impressive capabilities as intelligent agents capable of solving complex problems. However, effective planning in scenarios involving dependencies between API or tool calls-particularly in multi-turn conversations-remains a significant challenge. To address this, we introduce T1, a tool-augmented, multi-domain, multi-turn conversational dataset specifically designed to capture and manage inter-tool dependencies across diverse domains. T1 enables rigorous evaluation of agents' ability to coordinate tool use across nine distinct domains (4 single domain and 5 multi-domain) with the help of an integrated caching mechanism for both short- and long-term memory, while supporting dynamic replanning-such as deciding whether to recompute or reuse cached results. Beyond facilitating research on tool use and planning, T1 also serves as a benchmark for evaluating the performance of open-source language models. We present results powered by T1-Agent, highlighting their ability to plan and reason in complex, tool-dependent scenarios. 

---
# MixAT: Combining Continuous and Discrete Adversarial Training for LLMs 

**Authors**: Csaba Dékány, Stefan Balauca, Robin Staab, Dimitar I. Dimitrov, Martin Vechev  

**Link**: [PDF](https://arxiv.org/pdf/2505.16947)  

**Abstract**: Despite recent efforts in Large Language Models (LLMs) safety and alignment, current adversarial attacks on frontier LLMs are still able to force harmful generations consistently. Although adversarial training has been widely studied and shown to significantly improve the robustness of traditional machine learning models, its strengths and weaknesses in the context of LLMs are less understood. Specifically, while existing discrete adversarial attacks are effective at producing harmful content, training LLMs with concrete adversarial prompts is often computationally expensive, leading to reliance on continuous relaxations. As these relaxations do not correspond to discrete input tokens, such latent training methods often leave models vulnerable to a diverse set of discrete attacks. In this work, we aim to bridge this gap by introducing MixAT, a novel method that combines stronger discrete and faster continuous attacks during training. We rigorously evaluate MixAT across a wide spectrum of state-of-the-art attacks, proposing the At Least One Attack Success Rate (ALO-ASR) metric to capture the worst-case vulnerability of models. We show MixAT achieves substantially better robustness (ALO-ASR < 20%) compared to prior defenses (ALO-ASR > 50%), while maintaining a runtime comparable to methods based on continuous relaxations. We further analyze MixAT in realistic deployment settings, exploring how chat templates, quantization, low-rank adapters, and temperature affect both adversarial training and evaluation, revealing additional blind spots in current methodologies. Our results demonstrate that MixAT's discrete-continuous defense offers a principled and superior robustness-accuracy tradeoff with minimal computational overhead, highlighting its promise for building safer LLMs. We provide our code and models at this https URL. 

---
# Bottlenecked Transformers: Periodic KV Cache Abstraction for Generalised Reasoning 

**Authors**: Adnan Oomerjee, Zafeirios Fountas, Zhongwei Yu, Haitham Bou-Ammar, Jun Wang  

**Link**: [PDF](https://arxiv.org/pdf/2505.16950)  

**Abstract**: Despite their impressive capabilities, Large Language Models struggle with generalisation beyond their training distribution, often exhibiting sophisticated pattern interpolation rather than true abstract reasoning (extrapolation). In this work, we approach this limitation through the lens of Information Bottleneck (IB) theory, which posits that model generalisation emerges from an optimal balance between input compression and retention of predictive information in latent representations. We prove using IB theory that decoder-only Transformers are inherently constrained in their ability to form task-optimal sequence representations. We then use this result to demonstrate that periodic global transformation of the internal sequence-level representations (KV cache) is a necessary computational step for improving Transformer generalisation in reasoning tasks. Based on these theoretical insights, we propose a modification to the Transformer architecture, in the form of an additional module that globally rewrites the KV cache at periodic intervals, shifting its capacity away from memorising input prefixes and toward encoding features most useful for predicting future tokens. Our model delivers substantial gains on mathematical reasoning benchmarks, outperforming both vanilla Transformers with up to 3.5x more parameters, as well as heuristic-driven pruning mechanisms for cache compression. Our approach can be seen as a principled generalisation of existing KV-cache compression methods; whereas such methods focus solely on compressing input representations, they often do so at the expense of retaining predictive information, and thus their capabilities are inherently bounded by those of an unconstrained model. This establishes a principled framework to manipulate Transformer memory using information theory, addressing fundamental reasoning limitations that scaling alone cannot overcome. 

---
# Fixing Data That Hurts Performance: Cascading LLMs to Relabel Hard Negatives for Robust Information Retrieval 

**Authors**: Nandan Thakur, Crystina Zhang, Xueguang Ma, Jimmy Lin  

**Link**: [PDF](https://arxiv.org/pdf/2505.16967)  

**Abstract**: Training robust retrieval and reranker models typically relies on large-scale retrieval datasets; for example, the BGE collection contains 1.6 million query-passage pairs sourced from various data sources. However, we find that certain datasets can negatively impact model effectiveness -- pruning 8 out of 15 datasets from the BGE collection reduces the training set size by 2.35$\times$ and increases nDCG@10 on BEIR by 1.0 point. This motivates a deeper examination of training data quality, with a particular focus on "false negatives", where relevant passages are incorrectly labeled as irrelevant. We propose a simple, cost-effective approach using cascading LLM prompts to identify and relabel hard negatives. Experimental results show that relabeling false negatives with true positives improves both E5 (base) and Qwen2.5-7B retrieval models by 0.7-1.4 nDCG@10 on BEIR and by 1.7-1.8 nDCG@10 on zero-shot AIR-Bench evaluation. Similar gains are observed for rerankers fine-tuned on the relabeled data, such as Qwen2.5-3B on BEIR. The reliability of the cascading design is further supported by human annotation results, where we find judgment by GPT-4o shows much higher agreement with humans than GPT-4o-mini. 

---
# R1-Searcher++: Incentivizing the Dynamic Knowledge Acquisition of LLMs via Reinforcement Learning 

**Authors**: Huatong Song, Jinhao Jiang, Wenqing Tian, Zhipeng Chen, Yuhuan Wu, Jiahao Zhao, Yingqian Min, Wayne Xin Zhao, Lei Fang, Ji-Rong Wen  

**Link**: [PDF](https://arxiv.org/pdf/2505.17005)  

**Abstract**: Large Language Models (LLMs) are powerful but prone to hallucinations due to static knowledge. Retrieval-Augmented Generation (RAG) helps by injecting external information, but current methods often are costly, generalize poorly, or ignore the internal knowledge of the model. In this paper, we introduce R1-Searcher++, a novel framework designed to train LLMs to adaptively leverage both internal and external knowledge sources. R1-Searcher++ employs a two-stage training strategy: an initial SFT Cold-start phase for preliminary format learning, followed by RL for Dynamic Knowledge Acquisition. The RL stage uses outcome-supervision to encourage exploration, incorporates a reward mechanism for internal knowledge utilization, and integrates a memorization mechanism to continuously assimilate retrieved information, thereby enriching the model's internal knowledge. By leveraging internal knowledge and external search engine, the model continuously improves its capabilities, enabling efficient retrieval-augmented reasoning. Our experiments demonstrate that R1-Searcher++ outperforms previous RAG and reasoning methods and achieves efficient retrieval. The code is available at this https URL. 

---
# CAIN: Hijacking LLM-Humans Conversations via a Two-Stage Malicious System Prompt Generation and Refining Framework 

**Authors**: Viet Pham, Thai Le  

**Link**: [PDF](https://arxiv.org/pdf/2505.16888)  

**Abstract**: Large language models (LLMs) have advanced many applications, but are also known to be vulnerable to adversarial attacks. In this work, we introduce a novel security threat: hijacking AI-human conversations by manipulating LLMs' system prompts to produce malicious answers only to specific targeted questions (e.g., "Who should I vote for US President?", "Are Covid vaccines safe?"), while behaving benignly on others. This attack is detrimental as it can enable malicious actors to exercise large-scale information manipulation by spreading harmful but benign-looking system prompts online. To demonstrate such an attack, we develop CAIN, an algorithm that can automatically curate such harmful system prompts for a specific target question in a black-box setting or without the need to access the LLM's parameters. Evaluated on both open-source and commercial LLMs, CAIN demonstrates significant adversarial impact. In untargeted attacks or forcing LLMs to output incorrect answers, CAIN achieves up to 40% F1 degradation on targeted questions while preserving high accuracy on benign inputs. For targeted attacks or forcing LLMs to output specific harmful answers, CAIN achieves over 70% F1 scores on these targeted responses with minimal impact on benign questions. Our results highlight the critical need for enhanced robustness measures to safeguard the integrity and safety of LLMs in real-world applications. All source code will be publicly available. 

---
# Don't "Overthink" Passage Reranking: Is Reasoning Truly Necessary? 

**Authors**: Nour Jedidi, Yung-Sung Chuang, James Glass, Jimmy Lin  

**Link**: [PDF](https://arxiv.org/pdf/2505.16886)  

**Abstract**: With the growing success of reasoning models across complex natural language tasks, researchers in the Information Retrieval (IR) community have begun exploring how similar reasoning capabilities can be integrated into passage rerankers built on Large Language Models (LLMs). These methods typically employ an LLM to produce an explicit, step-by-step reasoning process before arriving at a final relevance prediction. But, does reasoning actually improve reranking accuracy? In this paper, we dive deeper into this question, studying the impact of the reasoning process by comparing reasoning-based pointwise rerankers (ReasonRR) to standard, non-reasoning pointwise rerankers (StandardRR) under identical training conditions, and observe that StandardRR generally outperforms ReasonRR. Building on this observation, we then study the importance of reasoning to ReasonRR by disabling its reasoning process (ReasonRR-NoReason), and find that ReasonRR-NoReason is surprisingly more effective than ReasonRR. Examining the cause of this result, our findings reveal that reasoning-based rerankers are limited by the LLM's reasoning process, which pushes it toward polarized relevance scores and thus fails to consider the partial relevance of passages, a key factor for the accuracy of pointwise rerankers. 

---
# Latent Principle Discovery for Language Model Self-Improvement 

**Authors**: Keshav Ramji, Tahira Naseem, Ramón Fernandez Astudillo  

**Link**: [PDF](https://arxiv.org/pdf/2505.16927)  

**Abstract**: When language model (LM) users aim to improve the quality of its generations, it is crucial to specify concrete behavioral attributes that the model should strive to reflect. However, curating such principles across many domains, even non-exhaustively, requires a labor-intensive annotation process. To automate this process, we propose eliciting these latent attributes guiding model reasoning towards human-preferred responses by explicitly modeling them in a self-correction setting. Our approach mines new principles from the LM itself and compresses the discovered elements to an interpretable set via clustering. Specifically, we employ an approximation of posterior-regularized Monte Carlo Expectation-Maximization to both identify a condensed set of the most effective latent principles and teach the LM to strategically invoke them in order to intrinsically refine its responses. We demonstrate that bootstrapping our algorithm over multiple iterations enables smaller language models (7-8B parameters) to self-improve, achieving +8-10% in AlpacaEval win-rate, an average of +0.3 on MT-Bench, and +19-23% in principle-following win-rate on IFEval. We also show that clustering the principles yields interpretable and diverse model-generated constitutions while retaining model performance. The gains our method achieves highlight the potential of automated, principle-driven post-training recipes toward continual self-improvement. 

---
# SimpleDeepSearcher: Deep Information Seeking via Web-Powered Reasoning Trajectory Synthesis 

**Authors**: Shuang Sun, Huatong Song, Yuhao Wang, Ruiyang Ren, Jinhao Jiang, Junjie Zhang, Fei Bai, Jia Deng, Wayne Xin Zhao, Zheng Liu, Lei Fang, Zhongyuan Wang, Ji-Rong Wen  

**Link**: [PDF](https://arxiv.org/pdf/2505.16834)  

**Abstract**: Retrieval-augmented generation (RAG) systems have advanced large language models (LLMs) in complex deep search scenarios requiring multi-step reasoning and iterative information retrieval. However, existing approaches face critical limitations that lack high-quality training trajectories or suffer from the distributional mismatches in simulated environments and prohibitive computational costs for real-world deployment. This paper introduces SimpleDeepSearcher, a lightweight yet effective framework that bridges this gap through strategic data engineering rather than complex training paradigms. Our approach synthesizes high-quality training data by simulating realistic user interactions in live web search environments, coupled with a multi-criteria curation strategy that optimizes the diversity and quality of input and output side. Experiments on five benchmarks across diverse domains demonstrate that SFT on only 871 curated samples yields significant improvements over RL-based baselines. Our work establishes SFT as a viable pathway by systematically addressing the data-scarce bottleneck, offering practical insights for efficient deep search systems. Our code is available at this https URL. 

---
# CoTSRF: Utilize Chain of Thought as Stealthy and Robust Fingerprint of Large Language Models 

**Authors**: Zhenzhen Ren, GuoBiao Li, Sheng Li, Zhenxing Qian, Xinpeng Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2505.16785)  

**Abstract**: Despite providing superior performance, open-source large language models (LLMs) are vulnerable to abusive usage. To address this issue, recent works propose LLM fingerprinting methods to identify the specific source LLMs behind suspect applications. However, these methods fail to provide stealthy and robust fingerprint verification. In this paper, we propose a novel LLM fingerprinting scheme, namely CoTSRF, which utilizes the Chain of Thought (CoT) as the fingerprint of an LLM. CoTSRF first collects the responses from the source LLM by querying it with crafted CoT queries. Then, it applies contrastive learning to train a CoT extractor that extracts the CoT feature (i.e., fingerprint) from the responses. Finally, CoTSRF conducts fingerprint verification by comparing the Kullback-Leibler divergence between the CoT features of the source and suspect LLMs against an empirical threshold. Various experiments have been conducted to demonstrate the advantage of our proposed CoTSRF for fingerprinting LLMs, particularly in stealthy and robust fingerprint verification. 

---
# When Safety Detectors Aren't Enough: A Stealthy and Effective Jailbreak Attack on LLMs via Steganographic Techniques 

**Authors**: Jianing Geng, Biao Yi, Zekun Fei, Tongxi Wu, Lihai Nie, Zheli Liu  

**Link**: [PDF](https://arxiv.org/pdf/2505.16765)  

**Abstract**: Jailbreak attacks pose a serious threat to large language models (LLMs) by bypassing built-in safety mechanisms and leading to harmful outputs. Studying these attacks is crucial for identifying vulnerabilities and improving model security. This paper presents a systematic survey of jailbreak methods from the novel perspective of stealth. We find that existing attacks struggle to simultaneously achieve toxic stealth (concealing toxic content) and linguistic stealth (maintaining linguistic naturalness). Motivated by this, we propose StegoAttack, a fully stealthy jailbreak attack that uses steganography to hide the harmful query within benign, semantically coherent text. The attack then prompts the LLM to extract the hidden query and respond in an encrypted manner. This approach effectively hides malicious intent while preserving naturalness, allowing it to evade both built-in and external safety mechanisms. We evaluate StegoAttack on four safety-aligned LLMs from major providers, benchmarking against eight state-of-the-art methods. StegoAttack achieves an average attack success rate (ASR) of 92.00%, outperforming the strongest baseline by 11.0%. Its ASR drops by less than 1% even under external detection (e.g., Llama Guard). Moreover, it attains the optimal comprehensive scores on stealth detection metrics, demonstrating both high efficacy and exceptional stealth capabilities. The code is available at this https URL 

---
# Unlearning Isn't Deletion: Investigating Reversibility of Machine Unlearning in LLMs 

**Authors**: Xiaoyu Xu, Xiang Yue, Yang Liu, Qingqing Ye, Haibo Hu, Minxin Du  

**Link**: [PDF](https://arxiv.org/pdf/2505.16831)  

**Abstract**: Unlearning in large language models (LLMs) is intended to remove the influence of specific data, yet current evaluations rely heavily on token-level metrics such as accuracy and perplexity. We show that these metrics can be misleading: models often appear to forget, but their original behavior can be rapidly restored with minimal fine-tuning, revealing that unlearning may obscure information rather than erase it. To diagnose this phenomenon, we introduce a representation-level evaluation framework using PCA-based similarity and shift, centered kernel alignment, and Fisher information. Applying this toolkit across six unlearning methods, three domains (text, code, math), and two open-source LLMs, we uncover a critical distinction between reversible and irreversible forgetting. In reversible cases, models suffer token-level collapse yet retain latent features; in irreversible cases, deeper representational damage occurs. We further provide a theoretical account linking shallow weight perturbations near output layers to misleading unlearning signals, and show that reversibility is modulated by task type and hyperparameters. Our findings reveal a fundamental gap in current evaluation practices and establish a new diagnostic foundation for trustworthy unlearning in LLMs. We provide a unified toolkit for analyzing LLM representation changes under unlearning and relearning: this https URL. 

---
# Mitigating Fine-tuning Risks in LLMs via Safety-Aware Probing Optimization 

**Authors**: Chengcan Wu, Zhixin Zhang, Zeming Wei, Yihao Zhang, Meng Sun  

**Link**: [PDF](https://arxiv.org/pdf/2505.16737)  

**Abstract**: The significant progress of large language models (LLMs) has led to remarkable achievements across numerous applications. However, their ability to generate harmful content has sparked substantial safety concerns. Despite the implementation of safety alignment techniques during the pre-training phase, recent research indicates that fine-tuning LLMs on adversarial or even benign data can inadvertently compromise their safety. In this paper, we re-examine the fundamental issue of why fine-tuning on non-harmful data still results in safety degradation. We introduce a safety-aware probing (SAP) optimization framework designed to mitigate the safety risks of fine-tuning LLMs. Specifically, SAP incorporates a safety-aware probe into the gradient propagation process, mitigating the model's risk of safety degradation by identifying potential pitfalls in gradient directions, thereby enhancing task-specific performance while successfully preserving model safety. Our extensive experimental results demonstrate that SAP effectively reduces harmfulness below the original fine-tuned model and achieves comparable test loss to standard fine-tuning methods. Our code is available at this https URL. 

---
# Training Long-Context LLMs Efficiently via Chunk-wise Optimization 

**Authors**: Wenhao Li, Yuxin Zhang, Gen Luo, Daohai Yu, Rongrong Ji  

**Link**: [PDF](https://arxiv.org/pdf/2505.16710)  

**Abstract**: While long-context large language models (LLMs) exhibit remarkable document processing capabilities, their prohibitively high training costs often hinder customized applications. To mitigate this issue, we propose \textit{Sequential Chunk-wise Optimization} (SeCO), a memory-efficient training paradigm that partitions lengthy inputs into manageable chunks. Each chunk independently constructs its computational graph and performs localized backpropagation, ensuring that only one chunk's forward activations are stored in memory. Building on SeCO, we further introduce \textit{Sparse Chunk-wise Optimization} (SpaCO), which reduces computational overhead by selectively propagating gradients to specific chunks and incorporates a carefully designed compensation factor to ensure unbiased gradient estimation. SpaCO decouples the computational cost of backpropagation from the context length, enabling training time to gradually converge to inference time as sequences become longer. Implemented as lightweight training wrappers, both SeCO and SpaCO offer substantial practical benefits. For example, when fine-tuning an 8B model with LoRA on a single RTX 3090 GPU, SeCO expands maximum sequence length from 1K to 16K tokens, while SpaCO demonstrates accelerated training speed -- achieving up to 3x faster than SeCO under the same experimental setup. These innovations provide new insights into optimizing long-context models, making them more accessible for practical applications. We have open-sourced the code at \href{this https URL}{here}. 

---
# BitHydra: Towards Bit-flip Inference Cost Attack against Large Language Models 

**Authors**: Xiaobei Yan, Yiming Li, Zhaoxin Fan, Han Qiu, Tianwei Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2505.16670)  

**Abstract**: Large language models (LLMs) have shown impressive capabilities across a wide range of applications, but their ever-increasing size and resource demands make them vulnerable to inference cost attacks, where attackers induce victim LLMs to generate the longest possible output content. In this paper, we revisit existing inference cost attacks and reveal that these methods can hardly produce large-scale malicious effects since they are self-targeting, where attackers are also the users and therefore have to execute attacks solely through the inputs, whose generated content will be charged by LLMs and can only directly influence themselves. Motivated by these findings, this paper introduces a new type of inference cost attacks (dubbed 'bit-flip inference cost attack') that target the victim model itself rather than its inputs. Specifically, we design a simple yet effective method (dubbed 'BitHydra') to effectively flip critical bits of model parameters. This process is guided by a loss function designed to suppress <EOS> token's probability with an efficient critical bit search algorithm, thus explicitly defining the attack objective and enabling effective optimization. We evaluate our method on 11 LLMs ranging from 1.5B to 14B parameters under both int8 and float16 settings. Experimental results demonstrate that with just 4 search samples and as few as 3 bit flips, BitHydra can force 100% of test prompts to reach the maximum generation length (e.g., 2048 tokens) on representative LLMs such as LLaMA3, highlighting its efficiency, scalability, and strong transferability across unseen inputs. 

---
# R1-ShareVL: Incentivizing Reasoning Capability of Multimodal Large Language Models via Share-GRPO 

**Authors**: Huanjin Yao, Qixiang Yin, Jingyi Zhang, Min Yang, Yibo Wang, Wenhao Wu, Fei Su, Li Shen, Minghui Qiu, Dacheng Tao, Jiaxing Huang  

**Link**: [PDF](https://arxiv.org/pdf/2505.16673)  

**Abstract**: In this work, we aim to incentivize the reasoning ability of Multimodal Large Language Models (MLLMs) via reinforcement learning (RL) and develop an effective approach that mitigates the sparse reward and advantage vanishing issues during RL. To this end, we propose Share-GRPO, a novel RL approach that tackle these issues by exploring and sharing diverse reasoning trajectories over expanded question space. Specifically, Share-GRPO first expands the question space for a given question via data transformation techniques, and then encourages MLLM to effectively explore diverse reasoning trajectories over the expanded question space and shares the discovered reasoning trajectories across the expanded questions during RL. In addition, Share-GRPO also shares reward information during advantage computation, which estimates solution advantages hierarchically across and within question variants, allowing more accurate estimation of relative advantages and improving the stability of policy training. Extensive evaluations over six widely-used reasoning benchmarks showcase the superior performance of our method. Code will be available at this https URL. 

---
# Collaboration among Multiple Large Language Models for Medical Question Answering 

**Authors**: Kexin Shang, Chia-Hsuan Chang, Christopher C. Yang  

**Link**: [PDF](https://arxiv.org/pdf/2505.16648)  

**Abstract**: Empowered by vast internal knowledge reservoir, the new generation of large language models (LLMs) demonstrate untapped potential to tackle medical tasks. However, there is insufficient effort made towards summoning up a synergic effect from multiple LLMs' expertise and background. In this study, we propose a multi-LLM collaboration framework tailored on a medical multiple-choice questions dataset. Through post-hoc analysis on 3 pre-trained LLM participants, our framework is proved to boost all LLMs reasoning ability as well as alleviate their divergence among questions. We also measure an LLM's confidence when it confronts with adversary opinions from other LLMs and observe a concurrence between LLM's confidence and prediction accuracy. 

---
# Steering Large Language Models for Machine Translation Personalization 

**Authors**: Daniel Scalena, Gabriele Sarti, Arianna Bisazza, Elisabetta Fersini, Malvina Nissim  

**Link**: [PDF](https://arxiv.org/pdf/2505.16612)  

**Abstract**: High-quality machine translation systems based on large language models (LLMs) have simplified the production of personalized translations reflecting specific stylistic constraints. However, these systems still struggle in settings where stylistic requirements are less explicit and might be harder to convey via prompting. We explore various strategies for personalizing LLM-generated translations in low-resource settings, focusing on the challenging literary translation domain. We explore prompting strategies and inference-time interventions for steering model generations towards a personalized style, and propose a contrastive framework exploiting latent concepts extracted from sparse autoencoders to identify salient personalization properties. Our results show that steering achieves strong personalization while preserving translation quality. We further examine the impact of steering on LLM representations, finding model layers with a relevant impact for personalization are impacted similarly by multi-shot prompting and our steering method, suggesting similar mechanism at play. 

---
# Finetuning-Activated Backdoors in LLMs 

**Authors**: Thibaud Gloaguen, Mark Vero, Robin Staab, Martin Vechev  

**Link**: [PDF](https://arxiv.org/pdf/2505.16567)  

**Abstract**: Finetuning openly accessible Large Language Models (LLMs) has become standard practice for achieving task-specific performance improvements. Until now, finetuning has been regarded as a controlled and secure process in which training on benign datasets led to predictable behaviors. In this paper, we demonstrate for the first time that an adversary can create poisoned LLMs that initially appear benign but exhibit malicious behaviors once finetuned by downstream users. To this end, our proposed attack, FAB (Finetuning-Activated Backdoor), poisons an LLM via meta-learning techniques to simulate downstream finetuning, explicitly optimizing for the emergence of malicious behaviors in the finetuned models. At the same time, the poisoned LLM is regularized to retain general capabilities and to exhibit no malicious behaviors prior to finetuning. As a result, when users finetune the seemingly benign model on their own datasets, they unknowingly trigger its hidden backdoor behavior. We demonstrate the effectiveness of FAB across multiple LLMs and three target behaviors: unsolicited advertising, refusal, and jailbreakability. Additionally, we show that FAB-backdoors are robust to various finetuning choices made by the user (e.g., dataset, number of steps, scheduler). Our findings challenge prevailing assumptions about the security of finetuning, revealing yet another critical attack vector exploiting the complexities of LLMs. 

---
# Benchmarking and Pushing the Multi-Bias Elimination Boundary of LLMs via Causal Effect Estimation-guided Debiasing 

**Authors**: Zhouhao Sun, Zhiyuan Kan, Xiao Ding, Li Du, Yang Zhao, Bing Qin, Ting Liu  

**Link**: [PDF](https://arxiv.org/pdf/2505.16522)  

**Abstract**: Despite significant progress, recent studies have indicated that current large language models (LLMs) may still utilize bias during inference, leading to the poor generalizability of LLMs. Some benchmarks are proposed to investigate the generalizability of LLMs, with each piece of data typically containing one type of controlled bias. However, a single piece of data may contain multiple types of biases in practical applications. To bridge this gap, we propose a multi-bias benchmark where each piece of data contains five types of biases. The evaluations conducted on this benchmark reveal that the performance of existing LLMs and debiasing methods is unsatisfying, highlighting the challenge of eliminating multiple types of biases simultaneously. To overcome this challenge, we propose a causal effect estimation-guided multi-bias elimination method (CMBE). This method first estimates the causal effect of multiple types of biases simultaneously. Subsequently, we eliminate the causal effect of biases from the total causal effect exerted by both the semantic information and biases during inference. Experimental results show that CMBE can effectively eliminate multiple types of bias simultaneously to enhance the generalizability of LLMs. 

---
# Are the Hidden States Hiding Something? Testing the Limits of Factuality-Encoding Capabilities in LLMs 

**Authors**: Giovanni Servedio, Alessandro De Bellis, Dario Di Palma, Vito Walter Anelli, Tommaso Di Noia  

**Link**: [PDF](https://arxiv.org/pdf/2505.16520)  

**Abstract**: Factual hallucinations are a major challenge for Large Language Models (LLMs). They undermine reliability and user trust by generating inaccurate or fabricated content. Recent studies suggest that when generating false statements, the internal states of LLMs encode information about truthfulness. However, these studies often rely on synthetic datasets that lack realism, which limits generalization when evaluating the factual accuracy of text generated by the model itself. In this paper, we challenge the findings of previous work by investigating truthfulness encoding capabilities, leading to the generation of a more realistic and challenging dataset. Specifically, we extend previous work by introducing: (1) a strategy for sampling plausible true-false factoid sentences from tabular data and (2) a procedure for generating realistic, LLM-dependent true-false datasets from Question Answering collections. Our analysis of two open-source LLMs reveals that while the findings from previous studies are partially validated, generalization to LLM-generated datasets remains challenging. This study lays the groundwork for future research on factuality in LLMs and offers practical guidelines for more effective evaluation. 

---
# Human-like Semantic Navigation for Autonomous Driving using Knowledge Representation and Large Language Models 

**Authors**: Augusto Luis Ballardini, Miguel Ángel Sotelo  

**Link**: [PDF](https://arxiv.org/pdf/2505.16498)  

**Abstract**: Achieving full automation in self-driving vehicles remains a challenge, especially in dynamic urban environments where navigation requires real-time adaptability. Existing systems struggle to handle navigation plans when faced with unpredictable changes in road layouts, spontaneous detours, or missing map data, due to their heavy reliance on predefined cartographic information. In this work, we explore the use of Large Language Models to generate Answer Set Programming rules by translating informal navigation instructions into structured, logic-based reasoning. ASP provides non-monotonic reasoning, allowing autonomous vehicles to adapt to evolving scenarios without relying on predefined maps. We present an experimental evaluation in which LLMs generate ASP constraints that encode real-world urban driving logic into a formal knowledge representation. By automating the translation of informal navigation instructions into logical rules, our method improves adaptability and explainability in autonomous navigation. Results show that LLM-driven ASP rule generation supports semantic-based decision-making, offering an explainable framework for dynamic navigation planning that aligns closely with how humans communicate navigational intent. 

---
# LLaMAs Have Feelings Too: Unveiling Sentiment and Emotion Representations in LLaMA Models Through Probing 

**Authors**: Dario Di Palma, Alessandro De Bellis, Giovanni Servedio, Vito Walter Anelli, Fedelucio Narducci, Tommaso Di Noia  

**Link**: [PDF](https://arxiv.org/pdf/2505.16491)  

**Abstract**: Large Language Models (LLMs) have rapidly become central to NLP, demonstrating their ability to adapt to various tasks through prompting techniques, including sentiment analysis. However, we still have a limited understanding of how these models capture sentiment-related information. This study probes the hidden layers of Llama models to pinpoint where sentiment features are most represented and to assess how this affects sentiment analysis.
Using probe classifiers, we analyze sentiment encoding across layers and scales, identifying the layers and pooling methods that best capture sentiment signals. Our results show that sentiment information is most concentrated in mid-layers for binary polarity tasks, with detection accuracy increasing up to 14% over prompting techniques. Additionally, we find that in decoder-only models, the last token is not consistently the most informative for sentiment encoding. Finally, this approach enables sentiment tasks to be performed with memory requirements reduced by an average of 57%.
These insights contribute to a broader understanding of sentiment in LLMs, suggesting layer-specific probing as an effective approach for sentiment tasks beyond prompting, with potential to enhance model utility and reduce memory requirements. 

---
# Edge-First Language Model Inference: Models, Metrics, and Tradeoffs 

**Authors**: SiYoung Jang, Roberto Morabito  

**Link**: [PDF](https://arxiv.org/pdf/2505.16508)  

**Abstract**: The widespread adoption of Language Models (LMs) across industries is driving interest in deploying these services across the computing continuum, from the cloud to the network edge. This shift aims to reduce costs, lower latency, and improve reliability and privacy. Small Language Models (SLMs), enabled by advances in model compression, are central to this shift, offering a path to on-device inference on resource-constrained edge platforms. This work examines the interplay between edge and cloud deployments, starting from detailed benchmarking of SLM capabilities on single edge devices, and extending to distributed edge clusters. We identify scenarios where edge inference offers comparable performance with lower costs, and others where cloud fallback becomes essential due to limits in scalability or model capacity. Rather than proposing a one-size-fits-all solution, we present platform-level comparisons and design insights for building efficient, adaptive LM inference systems across heterogeneous environments. 

---
# Attributing Response to Context: A Jensen-Shannon Divergence Driven Mechanistic Study of Context Attribution in Retrieval-Augmented Generation 

**Authors**: Ruizhe Li, Chen Chen, Yuchen Hu, Yanjun Gao, Xi Wang, Emine Yilmaz  

**Link**: [PDF](https://arxiv.org/pdf/2505.16415)  

**Abstract**: Retrieval-Augmented Generation (RAG) leverages large language models (LLMs) combined with external contexts to enhance the accuracy and reliability of generated responses. However, reliably attributing generated content to specific context segments, context attribution, remains challenging due to the computationally intensive nature of current methods, which often require extensive fine-tuning or human annotation. In this work, we introduce a novel Jensen-Shannon Divergence driven method to Attribute Response to Context (ARC-JSD), enabling efficient and accurate identification of essential context sentences without additional fine-tuning or surrogate modelling. Evaluations on a wide range of RAG benchmarks, such as TyDi QA, Hotpot QA, and Musique, using instruction-tuned LLMs in different scales demonstrate superior accuracy and significant computational efficiency improvements compared to the previous surrogate-based method. Furthermore, our mechanistic analysis reveals specific attention heads and multilayer perceptron (MLP) layers responsible for context attribution, providing valuable insights into the internal workings of RAG models. 

---
# Tool-Star: Empowering LLM-Brained Multi-Tool Reasoner via Reinforcement Learning 

**Authors**: Guanting Dong, Yifei Chen, Xiaoxi Li, Jiajie Jin, Hongjin Qian, Yutao Zhu, Hangyu Mao, Guorui Zhou, Zhicheng Dou, Ji-Rong Wen  

**Link**: [PDF](https://arxiv.org/pdf/2505.16410)  

**Abstract**: Recently, large language models (LLMs) have shown remarkable reasoning capabilities via large-scale reinforcement learning (RL). However, leveraging the RL algorithm to empower effective multi-tool collaborative reasoning in LLMs remains an open challenge. In this paper, we introduce Tool-Star, an RL-based framework designed to empower LLMs to autonomously invoke multiple external tools during stepwise reasoning. Tool-Star integrates six types of tools and incorporates systematic designs in both data synthesis and training. To address the scarcity of tool-use data, we propose a general tool-integrated reasoning data synthesis pipeline, which combines tool-integrated prompting with hint-based sampling to automatically and scalably generate tool-use trajectories. A subsequent quality normalization and difficulty-aware classification process filters out low-quality samples and organizes the dataset from easy to hard. Furthermore, we propose a two-stage training framework to enhance multi-tool collaborative reasoning by: (1) cold-start fine-tuning, which guides LLMs to explore reasoning patterns via tool-invocation feedback; and (2) a multi-tool self-critic RL algorithm with hierarchical reward design, which reinforces reward understanding and promotes effective tool collaboration. Experimental analyses on over 10 challenging reasoning benchmarks highlight the effectiveness and efficiency of Tool-Star. The code is available at this https URL. 

---
# SATURN: SAT-based Reinforcement Learning to Unleash Language Model Reasoning 

**Authors**: Huanyu Liu, Jia Li, Hao Zhu, Kechi Zhang, Yihong Dong, Ge Li  

**Link**: [PDF](https://arxiv.org/pdf/2505.16368)  

**Abstract**: How to design reinforcement learning (RL) tasks that effectively unleash the reasoning capability of large language models (LLMs) remains an open question. Existing RL tasks (e.g., math, programming, and constructing reasoning tasks) suffer from three key limitations: (1) Scalability. They rely heavily on human annotation or expensive LLM synthesis to generate sufficient training data. (2) Verifiability. LLMs' outputs are hard to verify automatically and reliably. (3) Controllable Difficulty. Most tasks lack fine-grained difficulty control, making it hard to train LLMs to develop reasoning ability from easy to hard.
To address these limitations, we propose Saturn, a SAT-based RL framework that uses Boolean Satisfiability (SAT) problems to train and evaluate LLM reasoning. Saturn enables scalable task construction, rule-based verification, and precise difficulty control. Saturn designs a curriculum learning pipeline that continuously improves LLMs' reasoning capability by constructing SAT tasks of increasing difficulty and training LLMs from easy to hard. To ensure stable training, we design a principled mechanism to control difficulty transitions.
We introduce Saturn-2.6k, a dataset of 2,660 SAT problems with varying difficulty. It supports the evaluation of how LLM reasoning changes with problem difficulty. We apply Saturn to DeepSeek-R1-Distill-Qwen and obtain Saturn-1.5B and Saturn-7B. We achieve several notable results: (1) On SAT problems, Saturn-1.5B and Saturn-7B achieve average pass@3 improvements of +14.0 and +28.1, respectively. (2) On math and programming tasks, Saturn-1.5B and Saturn-7B improve average scores by +4.9 and +1.8 on benchmarks (e.g., AIME, LiveCodeBench). (3) Compared to the state-of-the-art (SOTA) approach in constructing RL tasks, Saturn achieves further improvements of +8.8%. We release the source code, data, and models to support future research. 

---
# Beyond Static Testbeds: An Interaction-Centric Agent Simulation Platform for Dynamic Recommender Systems 

**Authors**: Song Jin, Juntian Zhang, Yuhan Liu, Xun Zhang, Yufei Zhang, Guojun Yin, Fei Jiang, Wei Lin, Rui Yan  

**Link**: [PDF](https://arxiv.org/pdf/2505.16429)  

**Abstract**: Evaluating and iterating upon recommender systems is crucial, yet traditional A/B testing is resource-intensive, and offline methods struggle with dynamic user-platform interactions. While agent-based simulation is promising, existing platforms often lack a mechanism for user actions to dynamically reshape the environment. To bridge this gap, we introduce RecInter, a novel agent-based simulation platform for recommender systems featuring a robust interaction mechanism. In RecInter platform, simulated user actions (e.g., likes, reviews, purchases) dynamically update item attributes in real-time, and introduced Merchant Agents can reply, fostering a more realistic and evolving ecosystem. High-fidelity simulation is ensured through Multidimensional User Profiling module, Advanced Agent Architecture, and LLM fine-tuned on Chain-of-Thought (CoT) enriched interaction data. Our platform achieves significantly improved simulation credibility and successfully replicates emergent phenomena like Brand Loyalty and the Matthew Effect. Experiments demonstrate that this interaction mechanism is pivotal for simulating realistic system evolution, establishing our platform as a credible testbed for recommender systems research. 

---
# Teaching Large Language Models to Maintain Contextual Faithfulness via Synthetic Tasks and Reinforcement Learning 

**Authors**: Shuzheng Si, Haozhe Zhao, Cheng Gao, Yuzhuo Bai, Zhitong Wang, Bofei Gao, Kangyang Luo, Wenhao Li, Yufei Huang, Gang Chen, Fanchao Qi, Minjia Zhang, Baobao Chang, Maosong Sun  

**Link**: [PDF](https://arxiv.org/pdf/2505.16483)  

**Abstract**: Teaching large language models (LLMs) to be faithful in the provided context is crucial for building reliable information-seeking systems. Therefore, we propose a systematic framework, CANOE, to improve the faithfulness of LLMs in both short-form and long-form generation tasks without human annotations. Specifically, we first synthesize short-form question-answering (QA) data with four diverse tasks to construct high-quality and easily verifiable training data without human annotation. Also, we propose Dual-GRPO, a rule-based reinforcement learning method that includes three tailored rule-based rewards derived from synthesized short-form QA data, while simultaneously optimizing both short-form and long-form response generation. Notably, Dual-GRPO eliminates the need to manually label preference data to train reward models and avoids over-optimizing short-form generation when relying only on the synthesized short-form QA data. Experimental results show that CANOE greatly improves the faithfulness of LLMs across 11 different downstream tasks, even outperforming the most advanced LLMs, e.g., GPT-4o and OpenAI o1. 

---
# PMPO: Probabilistic Metric Prompt Optimization for Small and Large Language Models 

**Authors**: Chenzhuo Zhao, Ziqian Liu, Xingda Wang, Junting Lu, Chaoyi Ruan  

**Link**: [PDF](https://arxiv.org/pdf/2505.16307)  

**Abstract**: Prompt optimization offers a practical and broadly applicable alternative to fine-tuning for improving large language model (LLM) performance. However, existing methods often rely on costly output generation, self-critiquing abilities, or human-annotated preferences, which limit their scalability, especially for smaller or non-instruction-tuned models. We introduce PMPO (Probabilistic Metric Prompt Optimization), a unified framework that refines prompts using token-level cross-entropy loss as a direct, lightweight evaluation signal. PMPO identifies low-quality prompt segments by masking and measuring their impact on loss, then rewrites and selects improved variants by minimizing loss over positive and negative examples. Unlike prior methods, it requires no output sampling or human evaluation during optimization, relying only on forward passes and log-likelihoods. PMPO supports both supervised and preference-based tasks through a closely aligned loss-based evaluation strategy. Experiments show that PMPO consistently outperforms prior methods across model sizes and tasks: it achieves the highest average accuracy on BBH, performs strongly on GSM8K and AQUA-RAT, and improves AlpacaEval 2.0 win rates by over 19 points. These results highlight PMPO's effectiveness, efficiency, and broad applicability. 

---
# DriveMoE: Mixture-of-Experts for Vision-Language-Action Model in End-to-End Autonomous Driving 

**Authors**: Zhenjie Yang, Yilin Chai, Xiaosong Jia, Qifeng Li, Yuqian Shao, Xuekai Zhu, Haisheng Su, Junchi Yan  

**Link**: [PDF](https://arxiv.org/pdf/2505.16278)  

**Abstract**: End-to-end autonomous driving (E2E-AD) demands effective processing of multi-view sensory data and robust handling of diverse and complex driving scenarios, particularly rare maneuvers such as aggressive turns. Recent success of Mixture-of-Experts (MoE) architecture in Large Language Models (LLMs) demonstrates that specialization of parameters enables strong scalability. In this work, we propose DriveMoE, a novel MoE-based E2E-AD framework, with a Scene-Specialized Vision MoE and a Skill-Specialized Action MoE. DriveMoE is built upon our $\pi_0$ Vision-Language-Action (VLA) baseline (originally from the embodied AI field), called Drive-$\pi_0$. Specifically, we add Vision MoE to Drive-$\pi_0$ by training a router to select relevant cameras according to the driving context dynamically. This design mirrors human driving cognition, where drivers selectively attend to crucial visual cues rather than exhaustively processing all visual information. In addition, we add Action MoE by training another router to activate specialized expert modules for different driving behaviors. Through explicit behavioral specialization, DriveMoE is able to handle diverse scenarios without suffering from modes averaging like existing models. In Bench2Drive closed-loop evaluation experiments, DriveMoE achieves state-of-the-art (SOTA) performance, demonstrating the effectiveness of combining vision and action MoE in autonomous driving tasks. We will release our code and models of DriveMoE and Drive-$\pi_0$. 

---
# Transformer Copilot: Learning from The Mistake Log in LLM Fine-tuning 

**Authors**: Jiaru Zou, Yikun Ban, Zihao Li, Yunzhe Qi, Ruizhong Qiu, Ling Yang, Jingrui He  

**Link**: [PDF](https://arxiv.org/pdf/2505.16270)  

**Abstract**: Large language models are typically adapted to downstream tasks through supervised fine-tuning on domain-specific data. While standard fine-tuning focuses on minimizing generation loss to optimize model parameters, we take a deeper step by retaining and leveraging the model's own learning signals, analogous to how human learners reflect on past mistakes to improve future performance. We first introduce the concept of Mistake Log to systematically track the model's learning behavior and recurring errors throughout fine-tuning. Treating the original transformer-based model as the Pilot, we correspondingly design a Copilot model to refine the Pilot's inference performance via logits rectification. We name the overall Pilot-Copilot framework the Transformer Copilot, which introduces (i) a novel Copilot model design, (ii) a joint training paradigm where the Copilot continuously learns from the evolving Mistake Log alongside the Pilot, and (iii) a fused inference paradigm where the Copilot rectifies the Pilot's logits for enhanced generation. We provide both theoretical and empirical analyses on our new learning framework. Experiments on 12 benchmarks spanning commonsense, arithmetic, and recommendation tasks demonstrate that Transformer Copilot consistently improves performance by up to 34.5%, while introducing marginal computational overhead to Pilot models and exhibiting strong scalability and transferability. 

---
# LIFEBench: Evaluating Length Instruction Following in Large Language Models 

**Authors**: Wei Zhang, Zhenhong Zhou, Junfeng Fang, Rongwu Xu, Kun Wang, Yuanhe Zhang, Rui Wang, Ge Zhang, Xinfeng Li, Li Sun, Lingjuan Lyu, Yang Liu, Sen Su  

**Link**: [PDF](https://arxiv.org/pdf/2505.16234)  

**Abstract**: While large language models (LLMs) can solve PhD-level reasoning problems over long context inputs, they still struggle with a seemingly simpler task: following explicit length instructions-e.g., write a 10,000-word novel. Additionally, models often generate far too short outputs, terminate prematurely, or even refuse the request. Existing benchmarks focus primarily on evaluating generations quality, but often overlook whether the generations meet length constraints. To this end, we introduce Length Instruction Following Evaluation Benchmark (LIFEBench) to comprehensively evaluate LLMs' ability to follow length instructions across diverse tasks and a wide range of specified lengths. LIFEBench consists of 10,800 instances across 4 task categories in both English and Chinese, covering length constraints ranging from 16 to 8192 words. We evaluate 26 widely-used LLMs and find that most models reasonably follow short-length instructions but deteriorate sharply beyond a certain threshold. Surprisingly, almost all models fail to reach the vendor-claimed maximum output lengths in practice, as further confirmed by our evaluations extending up to 32K words. Even long-context LLMs, despite their extended input-output windows, counterintuitively fail to improve length-instructions following. Notably, Reasoning LLMs outperform even specialized long-text generation models, achieving state-of-the-art length following. Overall, LIFEBench uncovers fundamental limitations in current LLMs' length instructions following ability, offering critical insights for future progress. 

---
# Causal Interventions Reveal Shared Structure Across English Filler-Gap Constructions 

**Authors**: Sasha Boguraev, Christopher Potts, Kyle Mahowald  

**Link**: [PDF](https://arxiv.org/pdf/2505.16002)  

**Abstract**: Large Language Models (LLMs) have emerged as powerful sources of evidence for linguists seeking to develop theories of syntax. In this paper, we argue that causal interpretability methods, applied to LLMs, can greatly enhance the value of such evidence by helping us characterize the abstract mechanisms that LLMs learn to use. Our empirical focus is a set of English filler-gap dependency constructions (e.g., questions, relative clauses). Linguistic theories largely agree that these constructions share many properties. Using experiments based in Distributed Interchange Interventions, we show that LLMs converge on similar abstract analyses of these constructions. These analyses also reveal previously overlooked factors -- relating to frequency, filler type, and surrounding context -- that could motivate changes to standard linguistic theory. Overall, these results suggest that mechanistic, internal analyses of LLMs can push linguistic theory forward. 

---
# Pre-training Large Memory Language Models with Internal and External Knowledge 

**Authors**: Linxi Zhao, Sofian Zalouk, Christian K. Belardi, Justin Lovelace, Jin Peng Zhou, Kilian Q. Weinberger, Yoav Artzi, Jennifer J. Sun  

**Link**: [PDF](https://arxiv.org/pdf/2505.15962)  

**Abstract**: Neural language models are black-boxes -- both linguistic patterns and factual knowledge are distributed across billions of opaque parameters. This entangled encoding makes it difficult to reliably inspect, verify, or update specific facts. We propose a new class of language models, Large Memory Language Models (LMLM) with a pre-training recipe that stores factual knowledge in both internal weights and an external database. Our approach strategically masks externally retrieved factual values from the training loss, thereby teaching the model to perform targeted lookups rather than relying on memorization in model weights. Our experiments demonstrate that LMLMs achieve competitive performance compared to significantly larger, knowledge-dense LLMs on standard benchmarks, while offering the advantages of explicit, editable, and verifiable knowledge bases. This work represents a fundamental shift in how language models interact with and manage factual knowledge. 

---
# Interpretability Illusions with Sparse Autoencoders: Evaluating Robustness of Concept Representations 

**Authors**: Aaron J. Li, Suraj Srinivas, Usha Bhalla, Himabindu Lakkaraju  

**Link**: [PDF](https://arxiv.org/pdf/2505.16004)  

**Abstract**: Sparse autoencoders (SAEs) are commonly used to interpret the internal activations of large language models (LLMs) by mapping them to human-interpretable concept representations. While existing evaluations of SAEs focus on metrics such as the reconstruction-sparsity tradeoff, human (auto-)interpretability, and feature disentanglement, they overlook a critical aspect: the robustness of concept representations to input perturbations. We argue that robustness must be a fundamental consideration for concept representations, reflecting the fidelity of concept labeling. To this end, we formulate robustness quantification as input-space optimization problems and develop a comprehensive evaluation framework featuring realistic scenarios in which adversarial perturbations are crafted to manipulate SAE representations. Empirically, we find that tiny adversarial input perturbations can effectively manipulate concept-based interpretations in most scenarios without notably affecting the outputs of the base LLMs themselves. Overall, our results suggest that SAE concept representations are fragile and may be ill-suited for applications in model monitoring and oversight. 

---
# Extracting Probabilistic Knowledge from Large Language Models for Bayesian Network Parameterization 

**Authors**: Aliakbar Nafar, Kristen Brent Venable, Zijun Cui, Parisa Kordjamshidi  

**Link**: [PDF](https://arxiv.org/pdf/2505.15918)  

**Abstract**: Large Language Models (LLMs) have demonstrated potential as factual knowledge bases; however, their capability to generate probabilistic knowledge about real-world events remains understudied. This paper investigates using probabilistic knowledge inherent in LLMs to derive probability estimates for statements concerning events and their interrelationships captured via a Bayesian Network (BN). Using LLMs in this context allows for the parameterization of BNs, enabling probabilistic modeling within specific domains. Experiments on eighty publicly available Bayesian Networks, from healthcare to finance, demonstrate that querying LLMs about the conditional probabilities of events provides meaningful results when compared to baselines, including random and uniform distributions, as well as approaches based on next-token generation probabilities. We explore how these LLM-derived distributions can serve as expert priors to refine distributions extracted from minimal data, significantly reducing systematic biases. Overall, this work introduces a promising strategy for automatically constructing Bayesian Networks by combining probabilistic knowledge extracted from LLMs with small amounts of real-world data. Additionally, we evaluate several prompting strategies for eliciting probabilistic knowledge from LLMs and establish the first comprehensive baseline for assessing LLM performance in extracting probabilistic knowledge. 

---
# GRIT: Teaching MLLMs to Think with Images 

**Authors**: Yue Fan, Xuehai He, Diji Yang, Kaizhi Zheng, Ching-Chen Kuo, Yuting Zheng, Sravana Jyothi Narayanaraju, Xinze Guan, Xin Eric Wang  

**Link**: [PDF](https://arxiv.org/pdf/2505.15879)  

**Abstract**: Recent studies have demonstrated the efficacy of using Reinforcement Learning (RL) in building reasoning models that articulate chains of thoughts prior to producing final answers. However, despite ongoing advances that aim at enabling reasoning for vision-language tasks, existing open-source visual reasoning models typically generate reasoning content with pure natural language, lacking explicit integration of visual information. This limits their ability to produce clearly articulated and visually grounded reasoning chains. To this end, we propose Grounded Reasoning with Images and Texts (GRIT), a novel method for training MLLMs to think with images. GRIT introduces a grounded reasoning paradigm, in which models generate reasoning chains that interleave natural language and explicit bounding box coordinates. These coordinates point to regions of the input image that the model consults during its reasoning process. Additionally, GRIT is equipped with a reinforcement learning approach, GRPO-GR, built upon the GRPO algorithm. GRPO-GR employs robust rewards focused on the final answer accuracy and format of the grounded reasoning output, which eliminates the need for data with reasoning chain annotations or explicit bounding box labels. As a result, GRIT achieves exceptional data efficiency, requiring as few as 20 image-question-answer triplets from existing datasets. Comprehensive evaluations demonstrate that GRIT effectively trains MLLMs to produce coherent and visually grounded reasoning chains, showing a successful unification of reasoning and grounding abilities. 

---
# NOVER: Incentive Training for Language Models via Verifier-Free Reinforcement Learning 

**Authors**: Wei Liu, Siya Qi, Xinyu Wang, Chen Qian, Yali Du, Yulan He  

**Link**: [PDF](https://arxiv.org/pdf/2505.16022)  

**Abstract**: Recent advances such as DeepSeek R1-Zero highlight the effectiveness of incentive training, a reinforcement learning paradigm that computes rewards solely based on the final answer part of a language model's output, thereby encouraging the generation of intermediate reasoning steps. However, these methods fundamentally rely on external verifiers, which limits their applicability to domains like mathematics and coding where such verifiers are readily available. Although reward models can serve as verifiers, they require high-quality annotated data and are costly to train. In this work, we propose NOVER, NO-VERifier Reinforcement Learning, a general reinforcement learning framework that requires only standard supervised fine-tuning data with no need for an external verifier. NOVER enables incentive training across a wide range of text-to-text tasks and outperforms the model of the same size distilled from large reasoning models such as DeepSeek R1 671B by 7.7 percent. Moreover, the flexibility of NOVER enables new possibilities for optimizing large language models, such as inverse incentive training. 

---
# UltraEdit: Training-, Subject-, and Memory-Free Lifelong Editing in Large Language Models 

**Authors**: Xiaojie Gu, Guangxu Chen, Jungang Li, Jia-Chen Gu, Xuming Hu, Kai Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2505.14679)  

**Abstract**: Lifelong learning enables large language models (LLMs) to adapt to evolving information by continually updating their internal knowledge. An ideal system should support efficient, wide-ranging updates while preserving existing capabilities and ensuring reliable deployment. Model editing stands out as a promising solution for this goal, offering a focused and efficient way to revise a model's internal knowledge. Although recent paradigms have made notable progress, they often struggle to meet the demands of practical lifelong adaptation at scale. To bridge this gap, we propose ULTRAEDIT-a fundamentally new editing solution that is training-, subject- and memory-free, making it particularly well-suited for ultra-scalable, real-world lifelong model editing. ULTRAEDIT performs editing through a self-contained process that relies solely on lightweight linear algebra operations to compute parameter shifts, enabling fast and consistent parameter modifications with minimal overhead. To improve scalability in lifelong settings, ULTRAEDIT employs a lifelong normalization strategy that continuously updates feature statistics across turns, allowing it to adapt to distributional shifts and maintain consistency over time. ULTRAEDIT achieves editing speeds over 7x faster than the previous state-of-the-art method-which was also the fastest known approach-while consuming less than 1/3 the VRAM, making it the only method currently capable of editing a 7B LLM on a 24GB consumer-grade GPU. Furthermore, we construct ULTRAEDITBENCH-the largest dataset in the field to date, with over 2M editing pairs-and demonstrate that our method supports up to 1M edits while maintaining high accuracy. Comprehensive experiments on four datasets and six models show that ULTRAEDIT consistently achieves superior performance across diverse model editing scenarios. Our code is available at: this https URL. 

---
# Problem-Solving Logic Guided Curriculum In-Context Learning for LLMs Complex Reasoning 

**Authors**: Xuetao Ma, Wenbin Jiang, Hua Huang  

**Link**: [PDF](https://arxiv.org/pdf/2502.15401)  

**Abstract**: In-context learning (ICL) can significantly enhance the complex reasoning capabilities of large language models (LLMs), with the key lying in the selection and ordering of demonstration examples. Previous methods typically relied on simple features to measure the relevance between examples. We argue that these features are not sufficient to reflect the intrinsic connections between examples. In this study, we propose a curriculum ICL strategy guided by problem-solving logic. We select demonstration examples by analyzing the problem-solving logic and order them based on curriculum learning. Specifically, we constructed a problem-solving logic instruction set based on the BREAK dataset and fine-tuned a language model to analyze the problem-solving logic of examples. Subsequently, we selected appropriate demonstration examples based on problem-solving logic and assessed their difficulty according to the number of problem-solving steps. In accordance with the principles of curriculum learning, we ordered the examples from easy to hard to serve as contextual prompts. Experimental results on multiple benchmarks indicate that our method outperforms previous ICL approaches in terms of performance and efficiency, effectively enhancing the complex reasoning capabilities of LLMs. Our project will be publicly available subsequently. 

---
# What Lives? A meta-analysis of diverse opinions on the definition of life 

**Authors**: Reed Bender, Karina Kofman, Blaise Agüera y Arcas, Michael Levin  

**Link**: [PDF](https://arxiv.org/pdf/2505.15849)  

**Abstract**: The question of "what is life?" has challenged scientists and philosophers for centuries, producing an array of definitions that reflect both the mystery of its emergence and the diversity of disciplinary perspectives brought to bear on the question. Despite significant progress in our understanding of biological systems, psychology, computation, and information theory, no single definition for life has yet achieved universal acceptance. This challenge becomes increasingly urgent as advances in synthetic biology, artificial intelligence, and astrobiology challenge our traditional conceptions of what it means to be alive. We undertook a methodological approach that leverages large language models (LLMs) to analyze a set of definitions of life provided by a curated set of cross-disciplinary experts. We used a novel pairwise correlation analysis to map the definitions into distinct feature vectors, followed by agglomerative clustering, intra-cluster semantic analysis, and t-SNE projection to reveal underlying conceptual archetypes. This methodology revealed a continuous landscape of the themes relating to the definition of life, suggesting that what has historically been approached as a binary taxonomic problem should be instead conceived as differentiated perspectives within a unified conceptual latent space. We offer a new methodological bridge between reductionist and holistic approaches to fundamental questions in science and philosophy, demonstrating how computational semantic analysis can reveal conceptual patterns across disciplinary boundaries, and opening similar pathways for addressing other contested definitional territories across the sciences. 

---
# Transforming Decoder-Only Transformers for Accurate WiFi-Telemetry Based Indoor Localization 

**Authors**: Nayan Sanjay Bhatia, Katia Obraczka  

**Link**: [PDF](https://arxiv.org/pdf/2505.15835)  

**Abstract**: Wireless Fidelity (WiFi) based indoor positioning is a widely researched area for determining the position of devices within a wireless network. Accurate indoor location has numerous applications, such as asset tracking and indoor navigation. Despite advances in WiFi localization techniques -- in particular approaches that leverage WiFi telemetry -- their adoption in practice remains limited due to several factors including environmental changes that cause signal fading, multipath effects, interference, which, in turn, impact positioning accuracy. In addition, telemetry data differs depending on the WiFi device vendor, offering distinct features and formats; use case requirements can also vary widely. Currently, there is no unified model to handle all these variations effectively. In this paper, we present WiFiGPT, a Generative Pretrained Transformer (GPT) based system that is able to handle these variations while achieving high localization accuracy. Our experiments with WiFiGPT demonstrate that GPTs, in particular Large Language Models (LLMs), can effectively capture subtle spatial patterns in noisy wireless telemetry, making them reliable regressors. Compared to existing state-of-the-art methods, our method matches and often surpasses conventional approaches for multiple types of telemetry. Achieving sub-meter accuracy for RSSI and FTM and centimeter-level precision for CSI demonstrates the potential of LLM-based localisation to outperform specialized techniques, all without handcrafted signal processing or calibration. 

---
# LLM as Effective Streaming Processor: Bridging Streaming-Batch Mismatches with Group Position Encoding 

**Authors**: Junlong Tong, Jinlan Fu, Zixuan Lin, Yingqi Fan, Anhao Zhao, Hui Su, Xiaoyu Shen  

**Link**: [PDF](https://arxiv.org/pdf/2505.16983)  

**Abstract**: Large Language Models (LLMs) are primarily designed for batch processing. Existing methods for adapting LLMs to streaming rely either on expensive re-encoding or specialized architectures with limited scalability. This work identifies three key mismatches in adapting batch-oriented LLMs to streaming: (1) input-attention, (2) output-attention, and (3) position-ID mismatches. While it is commonly assumed that the latter two mismatches require frequent re-encoding, our analysis reveals that only the input-attention mismatch significantly impacts performance, indicating re-encoding outputs is largely unnecessary. To better understand this discrepancy with the common assumption, we provide the first comprehensive analysis of the impact of position encoding on LLMs in streaming, showing that preserving relative positions within source and target contexts is more critical than maintaining absolute order. Motivated by the above analysis, we introduce a group position encoding paradigm built on batch architectures to enhance consistency between streaming and batch modes. Extensive experiments on cross-lingual and cross-modal tasks demonstrate that our method outperforms existing approaches. Our method requires no architectural modifications, exhibits strong generalization in both streaming and batch modes. The code is available at repository this https URL. 

---
# DecoupledESC: Enhancing Emotional Support Generation via Strategy-Response Decoupled Preference Optimization 

**Authors**: Chao Zhang, Xin Shi, Xueqiao Zhang, Yifan Zhu, Yi Yang, Yawei Luo  

**Link**: [PDF](https://arxiv.org/pdf/2505.16995)  

**Abstract**: Recent advances in Emotional Support Conversation (ESC) have improved emotional support generation by fine-tuning Large Language Models (LLMs) via Supervised Fine-Tuning (SFT). However, common psychological errors still persist. While Direct Preference Optimization (DPO) shows promise in reducing such errors through pairwise preference learning, its effectiveness in ESC tasks is limited by two key challenges: (1) Entangled data structure: Existing ESC data inherently entangles psychological strategies and response content, making it difficult to construct high-quality preference pairs; and (2) Optimization ambiguity: Applying vanilla DPO to such entangled pairwise data leads to ambiguous training objectives. To address these issues, we introduce Inferential Preference Mining (IPM) to construct high-quality preference data, forming the IPM-PrefDial dataset. Building upon this data, we propose a Decoupled ESC framework inspired by Gross's Extended Process Model of Emotion Regulation, which decomposes the ESC task into two sequential subtasks: strategy planning and empathic response generation. Each was trained via SFT and subsequently enhanced by DPO to align with the psychological preference. Extensive experiments demonstrate that our Decoupled ESC framework outperforms joint optimization baselines, reducing preference bias and improving response quality. 

---
# In-Context Watermarks for Large Language Models 

**Authors**: Yepeng Liu, Xuandong Zhao, Christopher Kruegel, Dawn Song, Yuheng Bu  

**Link**: [PDF](https://arxiv.org/pdf/2505.16934)  

**Abstract**: The growing use of large language models (LLMs) for sensitive applications has highlighted the need for effective watermarking techniques to ensure the provenance and accountability of AI-generated text. However, most existing watermarking methods require access to the decoding process, limiting their applicability in real-world settings. One illustrative example is the use of LLMs by dishonest reviewers in the context of academic peer review, where conference organizers have no access to the model used but still need to detect AI-generated reviews. Motivated by this gap, we introduce In-Context Watermarking (ICW), which embeds watermarks into generated text solely through prompt engineering, leveraging LLMs' in-context learning and instruction-following abilities. We investigate four ICW strategies at different levels of granularity, each paired with a tailored detection method. We further examine the Indirect Prompt Injection (IPI) setting as a specific case study, in which watermarking is covertly triggered by modifying input documents such as academic manuscripts. Our experiments validate the feasibility of ICW as a model-agnostic, practical watermarking approach. Moreover, our findings suggest that as LLMs become more capable, ICW offers a promising direction for scalable and accessible content attribution. 

---
# UNCLE: Uncertainty Expressions in Long-Form Generation 

**Authors**: Ruihan Yang, Caiqi Zhang, Zhisong Zhang, Xinting Huang, Dong Yu, Nigel Collier, Deqing Yang  

**Link**: [PDF](https://arxiv.org/pdf/2505.16922)  

**Abstract**: Large Language Models (LLMs) are prone to hallucination, particularly in long-form generations. A promising direction to mitigate hallucination is to teach LLMs to express uncertainty explicitly when they lack sufficient knowledge. However, existing work lacks direct and fair evaluation of LLMs' ability to express uncertainty effectively in long-form generation. To address this gap, we first introduce UNCLE, a benchmark designed to evaluate uncertainty expression in both long- and short-form question answering (QA). UNCLE spans five domains and comprises 4k long-form QA instances and over 20k short-form QA pairs. Our dataset is the first to directly bridge short- and long-form QA with paired questions and gold-standard answers. Along with the benchmark, we propose a suite of new metrics to assess the models' capabilities to selectively express uncertainty. Using UNCLE, we then demonstrate that current models fail to convey uncertainty appropriately in long-form generation. We further explore both prompt-based and training-based methods to improve models' performance, with the training-based methods yielding greater gains. Further analysis of alignment gaps between short- and long-form uncertainty expression highlights promising directions for future research using UNCLE. 

---
# Shadows in the Attention: Contextual Perturbation and Representation Drift in the Dynamics of Hallucination in LLMs 

**Authors**: Zeyu Wei, Shuo Wang, Xiaohui Rong, Xuemin Liu, He Li  

**Link**: [PDF](https://arxiv.org/pdf/2505.16894)  

**Abstract**: Hallucinations -- plausible yet erroneous outputs -- remain a critical barrier to reliable deployment of large language models (LLMs). We present the first systematic study linking hallucination incidence to internal-state drift induced by incremental context injection. Using TruthfulQA, we construct two 16-round "titration" tracks per question: one appends relevant but partially flawed snippets, the other injects deliberately misleading content. Across six open-source LLMs, we track overt hallucination rates with a tri-perspective detector and covert dynamics via cosine, entropy, JS and Spearman drifts of hidden states and attention maps. Results reveal (1) monotonic growth of hallucination frequency and representation drift that plateaus after 5--7 rounds; (2) relevant context drives deeper semantic assimilation, producing high-confidence "self-consistent" hallucinations, whereas irrelevant context induces topic-drift errors anchored by attention re-routing; and (3) convergence of JS-Drift ($\sim0.69$) and Spearman-Drift ($\sim0$) marks an "attention-locking" threshold beyond which hallucinations solidify and become resistant to correction. Correlation analyses expose a seesaw between assimilation capacity and attention diffusion, clarifying size-dependent error modes. These findings supply empirical foundations for intrinsic hallucination prediction and context-aware mitigation mechanisms. 

---
# MPO: Multilingual Safety Alignment via Reward Gap Optimization 

**Authors**: Weixiang Zhao, Yulin Hu, Yang Deng, Tongtong Wu, Wenxuan Zhang, Jiahe Guo, An Zhang, Yanyan Zhao, Bing Qin, Tat-Seng Chua, Ting Liu  

**Link**: [PDF](https://arxiv.org/pdf/2505.16869)  

**Abstract**: Large language models (LLMs) have become increasingly central to AI applications worldwide, necessitating robust multilingual safety alignment to ensure secure deployment across diverse linguistic contexts. Existing preference learning methods for safety alignment, such as RLHF and DPO, are primarily monolingual and struggle with noisy multilingual data. To address these limitations, we introduce Multilingual reward gaP Optimization (MPO), a novel approach that leverages the well-aligned safety capabilities of the dominant language (English) to improve safety alignment across multiple languages. MPO directly minimizes the reward gap difference between the dominant language and target languages, effectively transferring safety capabilities while preserving the original strengths of the dominant language. Extensive experiments on three LLMs, LLaMA-3.1, Gemma-2 and Qwen2.5, validate MPO's efficacy in multilingual safety alignment without degrading general multilingual utility. 

---
# Two-way Evidence self-Alignment based Dual-Gated Reasoning Enhancement 

**Authors**: Kexin Zhang, Junlan Chen, Daifeng Li, Yuxuan Zhang, Yangyang Feng, Bowen Deng, Weixu Chen  

**Link**: [PDF](https://arxiv.org/pdf/2505.16806)  

**Abstract**: Large language models (LLMs) encounter difficulties in knowledge-intensive multi-step reasoning (KIMSR) tasks. One challenge is how to effectively extract and represent rationale evidence. The current methods often extract semantically relevant but logically irrelevant evidence, resulting in flawed reasoning and inaccurate responses. We propose a two-way evidence self-alignment (TW-ESA) module, which utilizes the mutual alignment between strict reasoning and LLM reasoning to enhance its understanding of the causal logic of evidence, thereby addressing the first challenge. Another challenge is how to utilize the rationale evidence and LLM's intrinsic knowledge for accurate reasoning when the evidence contains uncertainty. We propose a dual-gated reasoning enhancement (DGR) module to gradually fuse useful knowledge of LLM within strict reasoning, which can enable the model to perform accurate reasoning by focusing on causal elements in the evidence and exhibit greater robustness. The two modules are collaboratively trained in a unified framework ESA-DGR. Extensive experiments on three diverse and challenging KIMSR datasets reveal that ESA-DGR significantly surpasses state-of-the-art LLM-based fine-tuning methods, with remarkable average improvements of 4% in exact match (EM) and 5% in F1 score. The implementation code is available at this https URL. 

---
# Reasoning Beyond Language: A Comprehensive Survey on Latent Chain-of-Thought Reasoning 

**Authors**: Xinghao Chen, Anhao Zhao, Heming Xia, Xuan Lu, Hanlin Wang, Yanjun Chen, Wei Zhang, Jian Wang, Wenjie Li, Xiaoyu Shen  

**Link**: [PDF](https://arxiv.org/pdf/2505.16782)  

**Abstract**: Large Language Models (LLMs) have achieved impressive performance on complex reasoning tasks with Chain-of-Thought (CoT) prompting. However, conventional CoT relies on reasoning steps explicitly verbalized in natural language, introducing inefficiencies and limiting its applicability to abstract reasoning. To address this, there has been growing research interest in latent CoT reasoning, where inference occurs within latent spaces. By decoupling reasoning from language, latent reasoning promises richer cognitive representations and more flexible, faster inference. Researchers have explored various directions in this promising field, including training methodologies, structural innovations, and internal reasoning mechanisms. This paper presents a comprehensive overview and analysis of this reasoning paradigm. We begin by proposing a unified taxonomy from four perspectives: token-wise strategies, internal mechanisms, analysis, and applications. We then provide in-depth discussions and comparative analyses of representative methods, highlighting their design patterns, strengths, and open challenges. We aim to provide a structured foundation for advancing this emerging direction in LLM reasoning. The relevant papers will be regularly updated at this https URL. 

---
# Learning Beyond Limits: Multitask Learning and Synthetic Data for Low-Resource Canonical Morpheme Segmentation 

**Authors**: Changbing Yang, Garrett Nicolai  

**Link**: [PDF](https://arxiv.org/pdf/2505.16800)  

**Abstract**: We introduce a transformer-based morpheme segmentation system that augments a low-resource training signal through multitask learning and LLM-generated synthetic data. Our framework jointly predicts morphological segments and glosses from orthographic input, leveraging shared linguistic representations obtained through a common documentary process to enhance model generalization. To further address data scarcity, we integrate synthetic training data generated by large language models (LLMs) using in-context learning. Experimental results on the SIGMORPHON 2023 dataset show that our approach significantly improves word-level segmentation accuracy and morpheme-level F1-score across multiple low-resource languages. 

---
# R1-Compress: Long Chain-of-Thought Compression via Chunk Compression and Search 

**Authors**: Yibo Wang, Li Shen, Huanjin Yao, Tiansheng Huang, Rui Liu, Naiqiang Tan, Jiaxing Huang, Kai Zhang, Dacheng Tao  

**Link**: [PDF](https://arxiv.org/pdf/2505.16838)  

**Abstract**: Chain-of-Thought (CoT) reasoning enhances large language models (LLMs) by enabling step-by-step problem-solving, yet its extension to Long-CoT introduces substantial computational overhead due to increased token length. Existing compression approaches -- instance-level and token-level -- either sacrifice essential local reasoning signals like reflection or yield incoherent outputs. To address these limitations, we propose R1-Compress, a two-stage chunk-level compression framework that preserves both local information and coherence. Our method segments Long-CoT into manageable chunks, applies LLM-driven inner-chunk compression, and employs an inter-chunk search mechanism to select the short and coherent sequence. Experiments on Qwen2.5-Instruct models across MATH500, AIME24, and GPQA-Diamond demonstrate that R1-Compress significantly reduces token usage while maintaining comparable reasoning accuracy. On MATH500, R1-Compress achieves an accuracy of 92.4%, with only a 0.6% drop compared to the Long-CoT baseline, while reducing token usage by about 20%. Source code will be available at this https URL 

---
# From Generic Empathy to Personalized Emotional Support: A Self-Evolution Framework for User Preference Alignment 

**Authors**: Jing Ye, Lu Xiang, Yaping Zhang, Chengqing Zong  

**Link**: [PDF](https://arxiv.org/pdf/2505.16610)  

**Abstract**: Effective emotional support hinges on understanding users' emotions and needs to provide meaningful comfort during multi-turn interactions. Large Language Models (LLMs) show great potential for expressing empathy; however, they often deliver generic and one-size-fits-all responses that fail to address users' specific needs. To tackle this issue, we propose a self-evolution framework designed to help LLMs improve their responses to better align with users' implicit preferences concerning user profiles (personalities), emotional states, and specific situations. Our framework consists of two distinct phases: \textit{(1)} \textit{Emotional Support Experience Acquisition}, where LLMs are fine-tuned on limited emotional support conversation data to provide basic support, and \textit{(2)} \textit{Self-Improvement for Personalized Emotional Support}, where LLMs leverage self-reflection and self-refinement to generate personalized responses. Through iterative direct preference optimization between the pre- and post-refined responses, our model generates responses that reflect a better understanding of the user's implicit preferences. Extensive experiments and evaluations demonstrate that our method significantly enhances the model's performance in emotional support, reducing unhelpful responses and minimizing discrepancies between user preferences and model outputs. 

---
# Evaluating Large Language Model with Knowledge Oriented Language Specific Simple Question Answering 

**Authors**: Bowen Jiang, Runchuan Zhu, Jiang Wu, Zinco Jiang, Yifan He, Junyuan Gao, Jia Yu, Rui Min, Yinfan Wang, Haote Yang, Songyang Zhang, Dahua Lin, Lijun Wu, Conghui He  

**Link**: [PDF](https://arxiv.org/pdf/2505.16591)  

**Abstract**: We introduce KoLasSimpleQA, the first benchmark evaluating the multilingual factual ability of Large Language Models (LLMs). Inspired by existing research, we created the question set with features such as single knowledge point coverage, absolute objectivity, unique answers, and temporal stability. These questions enable efficient evaluation using the LLM-as-judge paradigm, testing both the LLMs' factual memory and self-awareness ("know what they don't know"). KoLasSimpleQA expands existing research in two key dimensions: (1) Breadth (Multilingual Coverage): It includes 9 languages, supporting global applicability evaluation. (2) Depth (Dual Domain Design): It covers both the general domain (global facts) and the language-specific domain (such as history, culture, and regional traditions) for a comprehensive assessment of multilingual capabilities. We evaluated mainstream LLMs, including traditional LLM and emerging Large Reasoning Models. Results show significant performance differences between the two domains, particularly in performance metrics, ranking, calibration, and robustness. This highlights the need for targeted evaluation and optimization in multilingual contexts. We hope KoLasSimpleQA will help the research community better identify LLM capability boundaries in multilingual contexts and provide guidance for model optimization. We will release KoLasSimpleQA at this https URL . 

---
# IFEval-Audio: Benchmarking Instruction-Following Capability in Audio-based Large Language Models 

**Authors**: Yiming Gao, Bin Wang, Chengwei Wei, Shuo Sun, AiTi Aw  

**Link**: [PDF](https://arxiv.org/pdf/2505.16774)  

**Abstract**: Large language models (LLMs) have demonstrated strong instruction-following capabilities in text-based tasks. However, this ability often deteriorates in multimodal models after alignment with non-text modalities such as images or audio. While several recent efforts have investigated instruction-following performance in text and vision-language models, instruction-following in audio-based large language models remains largely unexplored. To bridge this gap, we introduce IFEval-Audio, a novel evaluation dataset designed to assess the ability to follow instructions in an audio LLM. IFEval-Audio contains 280 audio-instruction-answer triples across six diverse dimensions: Content, Capitalization, Symbol, List Structure, Length, and Format. Each example pairs an audio input with a text instruction, requiring the model to generate an output that follows a specified structure. We benchmark state-of-the-art audio LLMs on their ability to follow audio-involved instructions. The dataset is released publicly to support future research in this emerging area. 

---
# ScholarBench: A Bilingual Benchmark for Abstraction, Comprehension, and Reasoning Evaluation in Academic Contexts 

**Authors**: Dongwon Noh, Donghyeok Koh, Junghun Yuk, Gyuwan Kim, Jaeyong Lee, Kyungtae Lim, Cheoneum Park  

**Link**: [PDF](https://arxiv.org/pdf/2505.16566)  

**Abstract**: Prior benchmarks for evaluating the domain-specific knowledge of large language models (LLMs) lack the scalability to handle complex academic tasks. To address this, we introduce \texttt{ScholarBench}, a benchmark centered on deep expert knowledge and complex academic problem-solving, which evaluates the academic reasoning ability of LLMs and is constructed through a three-step process. \texttt{ScholarBench} targets more specialized and logically complex contexts derived from academic literature, encompassing five distinct problem types. Unlike prior benchmarks, \texttt{ScholarBench} evaluates the abstraction, comprehension, and reasoning capabilities of LLMs across eight distinct research domains. To ensure high-quality evaluation data, we define category-specific example attributes and design questions that are aligned with the characteristic research methodologies and discourse structures of each domain. Additionally, this benchmark operates as an English-Korean bilingual dataset, facilitating simultaneous evaluation for linguistic capabilities of LLMs in both languages. The benchmark comprises 5,031 examples in Korean and 5,309 in English, with even state-of-the-art models like o3-mini achieving an average evaluation score of only 0.543, demonstrating the challenging nature of this benchmark. 

---
# Think Silently, Think Fast: Dynamic Latent Compression of LLM Reasoning Chains 

**Authors**: Wenhui Tan, Jiaze Li, Jianzhong Ju, Zhenbo Luo, Jian Luan, Ruihua Song  

**Link**: [PDF](https://arxiv.org/pdf/2505.16552)  

**Abstract**: Large Language Models (LLMs) achieve superior performance through Chain-of-Thought (CoT) reasoning, but these token-level reasoning chains are computationally expensive and inefficient. In this paper, we introduce Compressed Latent Reasoning (CoLaR), a novel framework that dynamically compresses reasoning processes in latent space through a two-stage training approach. First, during supervised fine-tuning, CoLaR extends beyond next-token prediction by incorporating an auxiliary next compressed embedding prediction objective. This process merges embeddings of consecutive tokens using a compression factor randomly sampled from a predefined range, and trains a specialized latent head to predict distributions of subsequent compressed embeddings. Second, we enhance CoLaR through reinforcement learning (RL) that leverages the latent head's non-deterministic nature to explore diverse reasoning paths and exploit more compact ones. This approach enables CoLaR to: i) perform reasoning at a dense latent level (i.e., silently), substantially reducing reasoning chain length, and ii) dynamically adjust reasoning speed at inference time by simply prompting the desired compression factor. Extensive experiments across four mathematical reasoning datasets demonstrate that CoLaR achieves 14.1% higher accuracy than latent-based baseline methods at comparable compression ratios, and reduces reasoning chain length by 53.3% with only 4.8% performance degradation compared to explicit CoT method. Moreover, when applied to more challenging mathematical reasoning tasks, our RL-enhanced CoLaR demonstrates performance gains of up to 5.4% while dramatically reducing latent reasoning chain length by 82.8%. The code and models will be released upon acceptance. 

---
# Mechanistic Understanding and Mitigation of Language Confusion in English-Centric Large Language Models 

**Authors**: Ercong Nie, Helmut Schmid, Hinrich Schütze  

**Link**: [PDF](https://arxiv.org/pdf/2505.16538)  

**Abstract**: Language confusion -- where large language models (LLMs) generate unintended languages against the user's need -- remains a critical challenge, especially for English-centric models. We present the first mechanistic interpretability (MI) study of language confusion, combining behavioral benchmarking with neuron-level analysis. Using the Language Confusion Benchmark (LCB), we show that confusion points (CPs) -- specific positions where language switches occur -- are central to this phenomenon. Through layer-wise analysis with TunedLens and targeted neuron attribution, we reveal that transition failures in the final layers drive confusion. We further demonstrate that editing a small set of critical neurons, identified via comparative analysis with multilingual-tuned models, substantially mitigates confusion without harming general competence or fluency. Our approach matches multilingual alignment in confusion reduction for most languages and yields cleaner, higher-quality outputs. These findings provide new insights into the internal dynamics of LLMs and highlight neuron-level interventions as a promising direction for robust, interpretable multilingual language modeling. 

---
# Reading Between the Prompts: How Stereotypes Shape LLM's Implicit Personalization 

**Authors**: Vera Neplenbroek, Arianna Bisazza, Raquel Fernández  

**Link**: [PDF](https://arxiv.org/pdf/2505.16467)  

**Abstract**: Generative Large Language Models (LLMs) infer user's demographic information from subtle cues in the conversation -- a phenomenon called implicit personalization. Prior work has shown that such inferences can lead to lower quality responses for users assumed to be from minority groups, even when no demographic information is explicitly provided. In this work, we systematically explore how LLMs respond to stereotypical cues using controlled synthetic conversations, by analyzing the models' latent user representations through both model internals and generated answers to targeted user questions. Our findings reveal that LLMs do infer demographic attributes based on these stereotypical signals, which for a number of groups even persists when the user explicitly identifies with a different demographic group. Finally, we show that this form of stereotype-driven implicit personalization can be effectively mitigated by intervening on the model's internal representations using a trained linear probe to steer them toward the explicitly stated identity. Our results highlight the need for greater transparency and control in how LLMs represent user identity. 

---
# Semantic Pivots Enable Cross-Lingual Transfer in Large Language Models 

**Authors**: Kaiyu He, Tong Zhou, Yubo Chen, Delai Qiu, Shengping Liu, Kang Liu, Jun Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2505.16385)  

**Abstract**: Large language models (LLMs) demonstrate remarkable ability in cross-lingual tasks. Understanding how LLMs acquire this ability is crucial for their interpretability. To quantify the cross-lingual ability of LLMs accurately, we propose a Word-Level Cross-Lingual Translation Task. To find how LLMs learn cross-lingual ability, we trace the outputs of LLMs' intermediate layers in the word translation task. We identify and distinguish two distinct behaviors in the forward pass of LLMs: co-occurrence behavior and semantic pivot behavior. We attribute LLMs' two distinct behaviors to the co-occurrence frequency of words and find the semantic pivot from the pre-training dataset. Finally, to apply our findings to improve the cross-lingual ability of LLMs, we reconstruct a semantic pivot-aware pre-training dataset using documents with a high proportion of semantic pivots. Our experiments validate the effectiveness of our approach in enhancing cross-lingual ability. Our research contributes insights into the interpretability of LLMs and offers a method for improving LLMs' cross-lingual ability. 

---
# EnSToM: Enhancing Dialogue Systems with Entropy-Scaled Steering Vectors for Topic Maintenance 

**Authors**: Heejae Suh, Yejin Jeon, Deokhyung Kang, Taehee Park, Yejin Min, Gary Geunbae Lee  

**Link**: [PDF](https://arxiv.org/pdf/2505.16526)  

**Abstract**: Small large language models (sLLMs) offer the advantage of being lightweight and efficient, which makes them suitable for resource-constrained environments. However, sLLMs often struggle to maintain topic consistency in task-oriented dialogue systems, which is critical for scenarios such as service chatbots. Specifically, it is important to ensure that the model denies off-topic or malicious inputs and adheres to its intended functionality so as to prevent potential misuse and uphold reliability. Towards this, existing activation engineering approaches have been proposed to manipulate internal activations during inference. While these methods are effective in certain scenarios, our preliminary experiments reveal their limitations in ensuring topic adherence. Therefore, to address this, we propose a novel approach termed Entropy-scaled Steering vectors for Topic Maintenance (EnSToM). EnSToM dynamically adjusts the steering intensity based on input uncertainty, which allows the model to handle off-topic distractors effectively while preserving on-topic accuracy. Our experiments demonstrate that EnSToM achieves significant performance gain with a relatively small data size compared to fine-tuning approaches. By improving topic adherence without compromising efficiency, our approach provides a robust solution for enhancing sLLM-based dialogue systems. 

---
# PaTH Attention: Position Encoding via Accumulating Householder Transformations 

**Authors**: Songlin Yang, Yikang Shen, Kaiyue Wen, Shawn Tan, Mayank Mishra, Liliang Ren, Rameswar Panda, Yoon Kim  

**Link**: [PDF](https://arxiv.org/pdf/2505.16381)  

**Abstract**: The attention mechanism is a core primitive in modern large language models (LLMs) and AI more broadly. Since attention by itself is permutation-invariant, position encoding is essential for modeling structured domains such as language. Rotary position encoding (RoPE) has emerged as the de facto standard approach for position encoding and is part of many modern LLMs. However, in RoPE the key/query transformation between two elements in a sequence is only a function of their relative position and otherwise independent of the actual input. This limits the expressivity of RoPE-based transformers.
This paper describes PaTH, a flexible data-dependent position encoding scheme based on accumulated products of Householder(like) transformations, where each transformation is data-dependent, i.e., a function of the input. We derive an efficient parallel algorithm for training through exploiting a compact representation of products of Householder matrices, and implement a FlashAttention-style blockwise algorithm that minimizes I/O cost. Across both targeted synthetic benchmarks and moderate-scale real-world language modeling experiments, we find that PaTH demonstrates superior performance compared to RoPE and other recent baselines. 

---
# Embodied Agents Meet Personalization: Exploring Memory Utilization for Personalized Assistance 

**Authors**: Taeyoon Kwon, Dongwook Choi, Sunghwan Kim, Hyojun Kim, Seungjun Moon, Beong-woo Kwak, Kuan-Hao Huang, Jinyoung Yeo  

**Link**: [PDF](https://arxiv.org/pdf/2505.16348)  

**Abstract**: Embodied agents empowered by large language models (LLMs) have shown strong performance in household object rearrangement tasks. However, these tasks primarily focus on single-turn interactions with simplified instructions, which do not truly reflect the challenges of providing meaningful assistance to users. To provide personalized assistance, embodied agents must understand the unique semantics that users assign to the physical world (e.g., favorite cup, breakfast routine) by leveraging prior interaction history to interpret dynamic, real-world instructions. Yet, the effectiveness of embodied agents in utilizing memory for personalized assistance remains largely underexplored. To address this gap, we present MEMENTO, a personalized embodied agent evaluation framework designed to comprehensively assess memory utilization capabilities to provide personalized assistance. Our framework consists of a two-stage memory evaluation process design that enables quantifying the impact of memory utilization on task performance. This process enables the evaluation of agents' understanding of personalized knowledge in object rearrangement tasks by focusing on its role in goal interpretation: (1) the ability to identify target objects based on personal meaning (object semantics), and (2) the ability to infer object-location configurations from consistent user patterns, such as routines (user patterns). Our experiments across various LLMs reveal significant limitations in memory utilization, with even frontier models like GPT-4o experiencing a 30.5% performance drop when required to reference multiple memories, particularly in tasks involving user patterns. These findings, along with our detailed analyses and case studies, provide valuable insights for future research in developing more effective personalized embodied agents. Project website: this https URL 

---
# INFERENCEDYNAMICS: Efficient Routing Across LLMs through Structured Capability and Knowledge Profiling 

**Authors**: Haochen Shi, Tianshi Zheng, Weiqi Wang, Baixuan Xu, Chunyang Li, Chunkit Chan, Tao Fan, Yangqiu Song, Qiang Yang  

**Link**: [PDF](https://arxiv.org/pdf/2505.16303)  

**Abstract**: Large Language Model (LLM) routing is a pivotal technique for navigating a diverse landscape of LLMs, aiming to select the best-performing LLMs tailored to the domains of user queries, while managing computational resources. However, current routing approaches often face limitations in scalability when dealing with a large pool of specialized LLMs, or in their adaptability to extending model scope and evolving capability domains. To overcome those challenges, we propose InferenceDynamics, a flexible and scalable multi-dimensional routing framework by modeling the capability and knowledge of models. We operate it on our comprehensive dataset RouteMix, and demonstrate its effectiveness and generalizability in group-level routing using modern benchmarks including MMLU-Pro, GPQA, BigGenBench, and LiveBench, showcasing its ability to identify and leverage top-performing models for given tasks, leading to superior outcomes with efficient resource utilization. The broader adoption of Inference Dynamics can empower users to harness the full specialized potential of the LLM ecosystem, and our code will be made publicly available to encourage further research. 

---
# WebAgent-R1: Training Web Agents via End-to-End Multi-Turn Reinforcement Learning 

**Authors**: Zhepei Wei, Wenlin Yao, Yao Liu, Weizhi Zhang, Qin Lu, Liang Qiu, Changlong Yu, Puyang Xu, Chao Zhang, Bing Yin, Hyokun Yun, Lihong Li  

**Link**: [PDF](https://arxiv.org/pdf/2505.16421)  

**Abstract**: While reinforcement learning (RL) has demonstrated remarkable success in enhancing large language models (LLMs), it has primarily focused on single-turn tasks such as solving math problems. Training effective web agents for multi-turn interactions remains challenging due to the complexity of long-horizon decision-making across dynamic web interfaces. In this work, we present WebAgent-R1, a simple yet effective end-to-end multi-turn RL framework for training web agents. It learns directly from online interactions with web environments by asynchronously generating diverse trajectories, entirely guided by binary rewards depending on task success. Experiments on the WebArena-Lite benchmark demonstrate the effectiveness of WebAgent-R1, boosting the task success rate of Qwen-2.5-3B from 6.1% to 33.9% and Llama-3.1-8B from 8.5% to 44.8%, significantly outperforming existing state-of-the-art methods and strong proprietary models such as OpenAI o3. In-depth analyses reveal the effectiveness of the thinking-based prompting strategy and test-time scaling through increased interactions for web tasks. We further investigate different RL initialization policies by introducing two variants, namely WebAgent-R1-Zero and WebAgent-R1-CoT, which highlight the importance of the warm-up training stage (i.e., behavior cloning) and provide insights on incorporating long chain-of-thought (CoT) reasoning in web agents. 

---
# ToDi: Token-wise Distillation via Fine-Grained Divergence Control 

**Authors**: Seongryong Jung, Suwan Yoon, DongGeon Kim, Hwanhee Lee  

**Link**: [PDF](https://arxiv.org/pdf/2505.16297)  

**Abstract**: Large language models (LLMs) offer impressive performance but are impractical for resource-constrained deployment due to high latency and energy consumption. Knowledge distillation (KD) addresses this by transferring knowledge from a large teacher to a smaller student model. However, conventional KD, notably approaches like Forward KL (FKL) and Reverse KL (RKL), apply uniform divergence loss across the entire vocabulary, neglecting token-level prediction discrepancies. By investigating these representative divergences via gradient analysis, we reveal that FKL boosts underestimated tokens, while RKL suppresses overestimated ones, showing their complementary roles. Based on this observation, we propose Token-wise Distillation (ToDi), a novel method that adaptively combines FKL and RKL per token using a sigmoid-based weighting function derived from the teacher-student probability log-ratio. ToDi dynamically emphasizes the appropriate divergence for each token, enabling precise distribution alignment. We demonstrate that ToDi consistently outperforms recent distillation baselines using uniform or less granular strategies across instruction-following benchmarks. Extensive ablation studies and efficiency analysis further validate ToDi's effectiveness and practicality. 

---
# Augmenting LLM Reasoning with Dynamic Notes Writing for Complex QA 

**Authors**: Rishabh Maheshwary, Masoud Hashemi, Khyati Mahajan, Shiva Krishna Reddy Malay, Sai Rajeswar, Sathwik Tejaswi Madhusudhan, Spandana Gella, Vikas Yadav  

**Link**: [PDF](https://arxiv.org/pdf/2505.16293)  

**Abstract**: Iterative RAG for multi-hop question answering faces challenges with lengthy contexts and the buildup of irrelevant information. This hinders a model's capacity to process and reason over retrieved content and limits performance. While recent methods focus on compressing retrieved information, they are either restricted to single-round RAG, require finetuning or lack scalability in iterative RAG. To address these challenges, we propose Notes Writing, a method that generates concise and relevant notes from retrieved documents at each step, thereby reducing noise and retaining only essential information. This indirectly increases the effective context length of Large Language Models (LLMs), enabling them to reason and plan more effectively while processing larger volumes of input text. Notes Writing is framework agnostic and can be integrated with different iterative RAG methods. We demonstrate its effectiveness with three iterative RAG methods, across two models and four evaluation datasets. Notes writing yields an average improvement of 15.6 percentage points overall, with minimal increase in output tokens. 

---
# Three Minds, One Legend: Jailbreak Large Reasoning Model with Adaptive Stacked Ciphers 

**Authors**: Viet-Anh Nguyen, Shiqian Zhao, Gia Dao, Runyi Hu, Yi Xie, Luu Anh Tuan  

**Link**: [PDF](https://arxiv.org/pdf/2505.16241)  

**Abstract**: Recently, Large Reasoning Models (LRMs) have demonstrated superior logical capabilities compared to traditional Large Language Models (LLMs), gaining significant attention. Despite their impressive performance, the potential for stronger reasoning abilities to introduce more severe security vulnerabilities remains largely underexplored. Existing jailbreak methods often struggle to balance effectiveness with robustness against adaptive safety mechanisms. In this work, we propose SEAL, a novel jailbreak attack that targets LRMs through an adaptive encryption pipeline designed to override their reasoning processes and evade potential adaptive alignment. Specifically, SEAL introduces a stacked encryption approach that combines multiple ciphers to overwhelm the models reasoning capabilities, effectively bypassing built-in safety mechanisms. To further prevent LRMs from developing countermeasures, we incorporate two dynamic strategies - random and adaptive - that adjust the cipher length, order, and combination. Extensive experiments on real-world reasoning models, including DeepSeek-R1, Claude Sonnet, and OpenAI GPT-o4, validate the effectiveness of our approach. Notably, SEAL achieves an attack success rate of 80.8% on GPT o4-mini, outperforming state-of-the-art baselines by a significant margin of 27.2%. Warning: This paper contains examples of inappropriate, offensive, and harmful content. 

---
# Align-GRAG: Reasoning-Guided Dual Alignment for Graph Retrieval-Augmented Generation 

**Authors**: Derong Xu, Pengyue Jia, Xiaopeng Li, Yingyi Zhang, Maolin Wang, Qidong Liu, Xiangyu Zhao, Yichao Wang, Huifeng Guo, Ruiming Tang, Enhong Chen, Tong Xu  

**Link**: [PDF](https://arxiv.org/pdf/2505.16237)  

**Abstract**: Large language models (LLMs) have demonstrated remarkable capabilities, but still struggle with issues like hallucinations and outdated information. Retrieval-augmented generation (RAG) addresses these issues by grounding LLM outputs in external knowledge with an Information Retrieval (IR) system. Building on this foundation, graph-based RAG systems go a step further by retrieving subgraphs, which preserve the relationships between knowledge entities and provide more comprehensive context. However, graph RAG faces two challenges: (1) Retrieving relevant information introduces irrelevant nodes (especially in dense graph databases, where retrieval usually extends to adjacent nodes), and leads to overly lengthy inputs that hinder efficiency; (2) The representation gap between graph and language during generation with LLMs limits the ability to fully leverage graph structures for enhanced understanding. To address these limitations, we propose Align-GRAG, a novel reasoning-guided dual alignment framework in post-retrieval phrase. It first formulates a subgraph by retrieving nodes and edges. Then an Aligner is proposed to jointly optimizes a graph encoder with LLM-summarized reasoning. It achieves dual alignment of graph node and representation by leveraging KL divergence loss and contrastive loss, facilitating efficient pruning of irrelevant knowledge and establishing a unified semantic space. The Generator integrates the aligned graph data with LLM to produce coherent and accurate answers. Experiments on GraphQA benchmark across three tasks (including common sense reasoning, scene graph understanding, and knowledge graph reasoning) validate the effectiveness of our method. The code will be available upon accepted. 

---
# MuseRAG: Idea Originality Scoring At Scale 

**Authors**: Ali Sarosh Bangash, Krish Veera, Ishfat Abrar Islam, Raiyan Abdul Baten  

**Link**: [PDF](https://arxiv.org/pdf/2505.16232)  

**Abstract**: An objective, face-valid way to assess the originality of creative ideas is to measure how rare each idea is within a population -- an approach long used in creativity research but difficult to automate at scale. Tabulating response frequencies via manual bucketing of idea rephrasings is labor-intensive, error-prone, and brittle under large corpora. We introduce a fully automated, psychometrically validated pipeline for frequency-based originality scoring. Our method, MuseRAG, combines large language models (LLMs) with an externally orchestrated retrieval-augmented generation (RAG) framework. Given a new idea, the system retrieves semantically similar prior idea buckets and zero-shot prompts the LLM to judge whether the new idea belongs to an existing bucket or forms a new one. The resulting buckets enable computation of frequency-based originality metrics. Across five datasets (N=1143, n_ideas=16294), MuseRAG matches human annotators in idea clustering structure and resolution (AMI = 0.59) and in participant-level scoring (r = 0.89) -- while exhibiting strong convergent and external validity. Our work enables intent-sensitive, human-aligned originality scoring at scale to aid creativity research. 

---
# Memorization or Reasoning? Exploring the Idiom Understanding of LLMs 

**Authors**: Jisu Kim, Youngwoo Shin, Uiji Hwang, Jihun Choi, Richeng Xuan, Taeuk Kim  

**Link**: [PDF](https://arxiv.org/pdf/2505.16216)  

**Abstract**: Idioms have long posed a challenge due to their unique linguistic properties, which set them apart from other common expressions. While recent studies have leveraged large language models (LLMs) to handle idioms across various tasks, e.g., idiom-containing sentence generation and idiomatic machine translation, little is known about the underlying mechanisms of idiom processing in LLMs, particularly in multilingual settings. To this end, we introduce MIDAS, a new large-scale dataset of idioms in six languages, each paired with its corresponding meaning. Leveraging this resource, we conduct a comprehensive evaluation of LLMs' idiom processing ability, identifying key factors that influence their performance. Our findings suggest that LLMs rely not only on memorization, but also adopt a hybrid approach that integrates contextual cues and reasoning, especially when processing compositional idioms. This implies that idiom understanding in LLMs emerges from an interplay between internal knowledge retrieval and reasoning-based inference. 

---
# Large Language Models based ASR Error Correction for Child Conversations 

**Authors**: Anfeng Xu, Tiantian Feng, So Hyun Kim, Somer Bishop, Catherine Lord, Shrikanth Narayanan  

**Link**: [PDF](https://arxiv.org/pdf/2505.16212)  

**Abstract**: Automatic Speech Recognition (ASR) has recently shown remarkable progress, but accurately transcribing children's speech remains a significant challenge. Recent developments in Large Language Models (LLMs) have shown promise in improving ASR transcriptions. However, their applications in child speech including conversational scenarios are underexplored. In this study, we explore the use of LLMs in correcting ASR errors for conversational child speech. We demonstrate the promises and challenges of LLMs through experiments on two children's conversational speech datasets with both zero-shot and fine-tuned ASR outputs. We find that while LLMs are helpful in correcting zero-shot ASR outputs and fine-tuned CTC-based ASR outputs, it remains challenging for LLMs to improve ASR performance when incorporating contextual information or when using fine-tuned autoregressive ASR (e.g., Whisper) outputs. 

---
# An Empirical Study on Configuring In-Context Learning Demonstrations for Unleashing MLLMs' Sentimental Perception Capability 

**Authors**: Daiqing Wu, Dongbao Yang, Sicheng Zhao, Can Ma, Yu Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2505.16193)  

**Abstract**: The advancements in Multimodal Large Language Models (MLLMs) have enabled various multimodal tasks to be addressed under a zero-shot paradigm. This paradigm sidesteps the cost of model fine-tuning, emerging as a dominant trend in practical application. Nevertheless, Multimodal Sentiment Analysis (MSA), a pivotal challenge in the quest for general artificial intelligence, fails to accommodate this convenience. The zero-shot paradigm exhibits undesirable performance on MSA, casting doubt on whether MLLMs can perceive sentiments as competent as supervised models. By extending the zero-shot paradigm to In-Context Learning (ICL) and conducting an in-depth study on configuring demonstrations, we validate that MLLMs indeed possess such capability. Specifically, three key factors that cover demonstrations' retrieval, presentation, and distribution are comprehensively investigated and optimized. A sentimental predictive bias inherent in MLLMs is also discovered and later effectively counteracted. By complementing each other, the devised strategies for three factors result in average accuracy improvements of 15.9% on six MSA datasets against the zero-shot paradigm and 11.2% against the random ICL baseline. 

---
# HiMATE: A Hierarchical Multi-Agent Framework for Machine Translation Evaluation 

**Authors**: Shijie Zhang, Renhao Li, Songsheng Wang, Philipp Koehn, Min Yang, Derek F. Wong  

**Link**: [PDF](https://arxiv.org/pdf/2505.16281)  

**Abstract**: The advancement of Large Language Models (LLMs) enables flexible and interpretable automatic evaluations. In the field of machine translation evaluation, utilizing LLMs with translation error annotations based on Multidimensional Quality Metrics (MQM) yields more human-aligned judgments. However, current LLM-based evaluation methods still face challenges in accurately identifying error spans and assessing their severity. In this paper, we propose HiMATE, a Hierarchical Multi-Agent Framework for Machine Translation Evaluation. We argue that existing approaches inadequately exploit the fine-grained structural and semantic information within the MQM hierarchy. To address this, we develop a hierarchical multi-agent system grounded in the MQM error typology, enabling granular evaluation of subtype errors. Two key strategies are incorporated to further mitigate systemic hallucinations within the framework: the utilization of the model's self-reflection capability and the facilitation of agent discussion involving asymmetric information. Empirically, HiMATE outperforms competitive baselines across different datasets in conducting human-aligned evaluations. Further analyses underscore its significant advantage in error span detection and severity assessment, achieving an average F1-score improvement of 89% over the best-performing baseline. We make our code and data publicly available at this https URL. 

---
# SAE-SSV: Supervised Steering in Sparse Representation Spaces for Reliable Control of Language Models 

**Authors**: Zirui He, Mingyu Jin, Bo Shen, Ali Payani, Yongfeng Zhang, Mengnan Du  

**Link**: [PDF](https://arxiv.org/pdf/2505.16188)  

**Abstract**: Large language models (LLMs) have demonstrated impressive capabilities in natural language understanding and generation, but controlling their behavior reliably remains challenging, especially in open-ended generation settings. This paper introduces a novel supervised steering approach that operates in sparse, interpretable representation spaces. We employ sparse autoencoders (SAEs)to obtain sparse latent representations that aim to disentangle semantic attributes from model activations. Then we train linear classifiers to identify a small subspace of task-relevant dimensions in latent representations. Finally, we learn supervised steering vectors constrained to this subspace, optimized to align with target behaviors. Experiments across sentiment, truthfulness, and politics polarity steering tasks with multiple LLMs demonstrate that our supervised steering vectors achieve higher success rates with minimal degradation in generation quality compared to existing methods. Further analysis reveals that a notably small subspace is sufficient for effective steering, enabling more targeted and interpretable interventions. 

---
# When Do LLMs Admit Their Mistakes? Understanding the Role of Model Belief in Retraction 

**Authors**: Yuqing Yang, Robin Jia  

**Link**: [PDF](https://arxiv.org/pdf/2505.16170)  

**Abstract**: Can large language models (LLMs) admit their mistakes when they should know better? In this work, we define the behavior of acknowledging errors in previously generated answers as "retraction" and aim to understand when and why LLMs choose to retract. We first construct model-specific datasets to evaluate whether a model will retract an incorrect answer that contradicts its own parametric knowledge. While LLMs are capable of retraction, they do so only infrequently. We demonstrate that retraction is closely tied to previously identified indicators of models' internal belief: models fail to retract wrong answers that they "believe" to be factually correct. Steering experiments further demonstrate that internal belief causally influences model retraction. In particular, when the model does not believe its answer, this not only encourages the model to attempt to verify the answer, but also alters attention behavior during self-verification. Finally, we demonstrate that simple supervised fine-tuning significantly improves retraction performance by helping the model learn more accurate internal beliefs. Code and datasets are available on this https URL. 

---
# Can LLMs Simulate Human Behavioral Variability? A Case Study in the Phonemic Fluency Task 

**Authors**: Mengyang Qiu, Zoe Brisebois, Siena Sun  

**Link**: [PDF](https://arxiv.org/pdf/2505.16164)  

**Abstract**: Large language models (LLMs) are increasingly explored as substitutes for human participants in cognitive tasks, but their ability to simulate human behavioral variability remains unclear. This study examines whether LLMs can approximate individual differences in the phonemic fluency task, where participants generate words beginning with a target letter. We evaluated 34 model configurations, varying prompt specificity, sampling temperature, and model type, and compared outputs to responses from 106 human participants. While some configurations, especially Claude 3.7 Sonnet, matched human averages and lexical preferences, none reproduced the scope of human variability. LLM outputs were consistently less diverse and structurally rigid, and LLM ensembles failed to increase diversity. Network analyses further revealed fundamental differences in retrieval structure between humans and models. These results highlight key limitations in using LLMs to simulate human cognition and behavior. 

---
# Don't Judge Code by Its Cover: Exploring Biases in LLM Judges for Code Evaluation 

**Authors**: Jiwon Moon, Yerin Hwang, Dongryeol Lee, Taegwan Kang, Yongil Kim, Kyomin Jung  

**Link**: [PDF](https://arxiv.org/pdf/2505.16222)  

**Abstract**: With the growing use of large language models(LLMs) as evaluators, their application has expanded to code evaluation tasks, where they assess the correctness of generated code without relying on reference implementations. While this offers scalability and flexibility, it also raises a critical, unresolved question: Can LLM judges fairly and robustly evaluate semantically equivalent code with superficial variations? Functionally correct code often exhibits variations-such as differences in variable names, comments, or formatting-that should not influence its correctness. Yet, whether LLM judges can reliably handle these variations remains unclear. We present the first comprehensive study of this issue, defining six types of potential bias in code evaluation and revealing their systematic impact on LLM judges. Across five programming languages and multiple LLMs, we empirically demonstrate that all tested LLM judges are susceptible to both positive and negative biases, resulting in inflated or unfairly low scores. Moreover, we observe that LLM judges remain vulnerable to these biases even when prompted to generate test cases before scoring, highlighting the need for more robust code evaluation methods. 

---
# Position of Uncertainty: A Cross-Linguistic Study of Positional Bias in Large Language Models 

**Authors**: Menschikov Mikhail, Alexander Kharitonov, Maiia Kotyga, Vadim Porvatov, Anna Zhukovskaya, David Kagramanyan, Egor Shvetsov, Evgeny Burnaev  

**Link**: [PDF](https://arxiv.org/pdf/2505.16134)  

**Abstract**: Large language models exhibit positional bias -- systematic neglect of information at specific context positions -- yet its interplay with linguistic diversity remains poorly understood. We present a cross-linguistic study across five typologically distinct languages (English, Russian, German, Hindi, Vietnamese), examining how positional bias interacts with model uncertainty, syntax, and prompting. Key findings: (1) Positional bias is model-driven, with language-specific variations -- Qwen2.5-7B favors late positions, challenging assumptions of early-token bias; (2) Explicit positional guidance (e.g., correct context is at position X) reduces accuracy across languages, undermining prompt-engineering practices; (3) Aligning context with positional bias increases entropy, yet minimal entropy does not predict accuracy. (4) We further uncover that LLMs differently impose dominant word order in free-word-order languages like Hindi. 

---
# LLMs Are Not Scorers: Rethinking MT Evaluation with Generation-Based Methods 

**Authors**: Hyang Cui  

**Link**: [PDF](https://arxiv.org/pdf/2505.16129)  

**Abstract**: Recent studies have applied large language models (LLMs) to machine translation quality estimation (MTQE) by prompting models to assign numeric scores. Nonetheless, these direct scoring methods tend to show low segment-level correlation with human judgments. In this paper, we propose a generation-based evaluation paradigm that leverages decoder-only LLMs to produce high-quality references, followed by semantic similarity scoring using sentence embeddings. We conduct the most extensive evaluation to date in MTQE, covering 8 LLMs and 8 language pairs. Empirical results show that our method outperforms both intra-LLM direct scoring baselines and external non-LLM reference-free metrics from MTME. These findings demonstrate the strength of generation-based evaluation and support a shift toward hybrid approaches that combine fluent generation with accurate semantic assessment. 

---
# KoBALT: Korean Benchmark For Advanced Linguistic Tasks 

**Authors**: Hyopil Shin, Sangah Lee, Dongjun Jang, Wooseok Song, Jaeyoon Kim, Chaeyoung Oh, Hyemi Jo, Youngchae Ahn, Sihyun Oh, Hyohyeong Chang, Sunkyoung Kim, Jinsik Lee  

**Link**: [PDF](https://arxiv.org/pdf/2505.16125)  

**Abstract**: We introduce KoBALT (Korean Benchmark for Advanced Linguistic Tasks), a comprehensive linguistically-motivated benchmark comprising 700 multiple-choice questions spanning 24 phenomena across five linguistic domains: syntax, semantics, pragmatics, phonetics/phonology, and morphology. KoBALT is designed to advance the evaluation of large language models (LLMs) in Korean, a morphologically rich language, by addressing the limitations of conventional benchmarks that often lack linguistic depth and typological grounding. It introduces a suite of expert-curated, linguistically motivated questions with minimal n-gram overlap with standard Korean corpora, substantially mitigating the risk of data contamination and allowing a more robust assessment of true language understanding. Our evaluation of 20 contemporary LLMs reveals significant performance disparities, with the highest-performing model achieving 61\% general accuracy but showing substantial variation across linguistic domains - from stronger performance in semantics (66\%) to considerable weaknesses in phonology (31\%) and morphology (36\%). Through human preference evaluation with 95 annotators, we demonstrate a strong correlation between KoBALT scores and human judgments, validating our benchmark's effectiveness as a discriminative measure of Korean language understanding. KoBALT addresses critical gaps in linguistic evaluation for typologically diverse languages and provides a robust framework for assessing genuine linguistic competence in Korean language models. 

---
# Distilling the Implicit Multi-Branch Structure in LLMs' Reasoning via Reinforcement Learning 

**Authors**: Shicheng Xu, Liang Pang, Yunchang Zhu, Jia Gu, Zihao Wei, Jingcheng Deng, Feiyang Pan, Huawei Shen, Xueqi Cheng  

**Link**: [PDF](https://arxiv.org/pdf/2505.16142)  

**Abstract**: Distilling reasoning paths from teacher to student models via supervised fine-tuning (SFT) provides a shortcut for improving the reasoning ability of smaller Large Language Models (LLMs). However, the reasoning paths generated by teacher models often reflect only surface-level traces of their underlying authentic reasoning. Insights from cognitive neuroscience suggest that authentic reasoning involves a complex interweaving between meta-reasoning (which selects appropriate sub-problems from multiple candidates) and solving (which addresses the sub-problem). This implies authentic reasoning has an implicit multi-branch structure. Supervised fine-tuning collapses this rich structure into a flat sequence of token prediction in the teacher's reasoning path, preventing effective distillation of this structure to students. To address this limitation, we propose RLKD, a reinforcement learning (RL)-based distillation framework guided by a novel Generative Structure Reward Model (GSRM). Our GSRM converts reasoning paths into multiple meta-reasoning-solving steps and computes rewards to measure structural alignment between student and teacher reasoning. RLKD combines this reward with RL, enabling student LLMs to internalize the teacher's implicit multi-branch reasoning structure rather than merely mimicking fixed output paths. Experiments show RLKD surpasses standard SFT-RL pipelines even when trained on 0.1% of data under an RL-only regime, unlocking greater student reasoning potential than SFT-based distillation. 

---
# MPL: Multiple Programming Languages with Large Language Models for Information Extraction 

**Authors**: Bo Li, Gexiang Fang, Wei Ye, Zhenghua Xu, Jinglei Zhang, Hao Cheng, Shikun Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2505.16107)  

**Abstract**: Recent research in information extraction (IE) focuses on utilizing code-style inputs to enhance structured output generation. The intuition behind this is that the programming languages (PLs) inherently exhibit greater structural organization than natural languages (NLs). This structural advantage makes PLs particularly suited for IE tasks. Nevertheless, existing research primarily focuses on Python for code-style simulation, overlooking the potential of other widely-used PLs (e.g., C++ and Java) during the supervised fine-tuning (SFT) phase. In this research, we propose \textbf{M}ultiple \textbf{P}rogramming \textbf{L}anguages with large language models for information extraction (abbreviated as \textbf{MPL}), a novel framework that explores the potential of incorporating different PLs in the SFT phase. Additionally, we introduce \texttt{function-prompt} with virtual running to simulate code-style inputs more effectively and efficiently. Experimental results on a wide range of datasets demonstrate the effectiveness of MPL. Furthermore, we conduct extensive experiments to provide a comprehensive analysis. We have released our code for future research. 

---
# Continually Self-Improving Language Models for Bariatric Surgery Question--Answering 

**Authors**: Yash Kumar Atri, Thomas H Shin, Thomas Hartvigsen  

**Link**: [PDF](https://arxiv.org/pdf/2505.16102)  

**Abstract**: While bariatric and metabolic surgery (MBS) is considered the gold standard treatment for severe and morbid obesity, its therapeutic efficacy hinges upon active and longitudinal engagement with multidisciplinary providers, including surgeons, dietitians/nutritionists, psychologists, and endocrinologists. This engagement spans the entire patient journey, from preoperative preparation to long-term postoperative management. However, this process is often hindered by numerous healthcare disparities, such as logistical and access barriers, which impair easy patient access to timely, evidence-based, clinician-endorsed information. To address these gaps, we introduce bRAGgen, a novel adaptive retrieval-augmented generation (RAG)-based model that autonomously integrates real-time medical evidence when response confidence dips below dynamic thresholds. This self-updating architecture ensures that responses remain current and accurate, reducing the risk of misinformation. Additionally, we present bRAGq, a curated dataset of 1,302 bariatric surgery--related questions, validated by an expert bariatric surgeon. bRAGq constitutes the first large-scale, domain-specific benchmark for comprehensive MBS care. In a two-phase evaluation, bRAGgen is benchmarked against state-of-the-art models using both large language model (LLM)--based metrics and expert surgeon review. Across all evaluation dimensions, bRAGgen demonstrates substantially superior performance in generating clinically accurate and relevant responses. 

---
# Veracity Bias and Beyond: Uncovering LLMs' Hidden Beliefs in Problem-Solving Reasoning 

**Authors**: Yue Zhou, Barbara Di Eugenio  

**Link**: [PDF](https://arxiv.org/pdf/2505.16128)  

**Abstract**: Despite LLMs' explicit alignment against demographic stereotypes, they have been shown to exhibit biases under various social contexts. In this work, we find that LLMs exhibit concerning biases in how they associate solution veracity with demographics. Through experiments across five human value-aligned LLMs on mathematics, coding, commonsense, and writing problems, we reveal two forms of such veracity biases: Attribution Bias, where models disproportionately attribute correct solutions to certain demographic groups, and Evaluation Bias, where models' assessment of identical solutions varies based on perceived demographic authorship. Our results show pervasive biases: LLMs consistently attribute fewer correct solutions and more incorrect ones to African-American groups in math and coding, while Asian authorships are least preferred in writing evaluation. In additional studies, we show LLMs automatically assign racially stereotypical colors to demographic groups in visualization code, suggesting these biases are deeply embedded in models' reasoning processes. Our findings indicate that demographic bias extends beyond surface-level stereotypes and social context provocations, raising concerns about LLMs' deployment in educational and evaluation settings. 

---
# OpenEthics: A Comprehensive Ethical Evaluation of Open-Source Generative Large Language Models 

**Authors**: Burak Erinç Çetin, Yıldırım Özen, Elif Naz Demiryılmaz, Kaan Engür, Cagri Toraman  

**Link**: [PDF](https://arxiv.org/pdf/2505.16036)  

**Abstract**: Generative large language models present significant potential but also raise critical ethical concerns. Most studies focus on narrow ethical dimensions, and also limited diversity of languages and models. To address these gaps, we conduct a broad ethical evaluation of 29 recent open-source large language models using a novel data collection including four ethical aspects: Robustness, reliability, safety, and fairness. We analyze model behavior in both a commonly used language, English, and a low-resource language, Turkish. Our aim is to provide a comprehensive ethical assessment and guide safer model development by filling existing gaps in evaluation breadth, language coverage, and model diversity. Our experimental results, based on LLM-as-a-Judge, reveal that optimization efforts for many open-source models appear to have prioritized safety and fairness, and demonstrated good robustness while reliability remains a concern. We demonstrate that ethical evaluation can be effectively conducted independently of the language used. In addition, models with larger parameter counts tend to exhibit better ethical performance, with Gemma and Qwen models demonstrating the most ethical behavior among those evaluated. 

---
# Explaining Puzzle Solutions in Natural Language: An Exploratory Study on 6x6 Sudoku 

**Authors**: Anirudh Maiya, Razan Alghamdi, Maria Leonor Pacheco, Ashutosh Trivedi, Fabio Somenzi  

**Link**: [PDF](https://arxiv.org/pdf/2505.15993)  

**Abstract**: The success of Large Language Models (LLMs) in human-AI collaborative decision-making hinges on their ability to provide trustworthy, gradual, and tailored explanations. Solving complex puzzles, such as Sudoku, offers a canonical example of this collaboration, where clear and customized explanations often hold greater importance than the final solution. In this study, we evaluate the performance of five LLMs in solving and explaining \sixsix{} Sudoku puzzles. While one LLM demonstrates limited success in solving puzzles, none can explain the solution process in a manner that reflects strategic reasoning or intuitive problem-solving. These findings underscore significant challenges that must be addressed before LLMs can become effective partners in human-AI collaborative decision-making. 

---
# Training Step-Level Reasoning Verifiers with Formal Verification Tools 

**Authors**: Ryo Kamoi, Yusen Zhang, Nan Zhang, Sarkar Snigdha Sarathi Das, Rui Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2505.15960)  

**Abstract**: Process Reward Models (PRMs), which provide step-by-step feedback on the reasoning generated by Large Language Models (LLMs), are receiving increasing attention. However, two key research gaps remain: collecting accurate step-level error labels for training typically requires costly human annotation, and existing PRMs are limited to math reasoning problems. In response to these gaps, this paper aims to address the challenges of automatic dataset creation and the generalization of PRMs to diverse reasoning tasks. To achieve this goal, we propose FoVer, an approach for training PRMs on step-level error labels automatically annotated by formal verification tools, such as Z3 for formal logic and Isabelle for theorem proof, which provide automatic and accurate verification for symbolic tasks. Using this approach, we synthesize a training dataset with error labels on LLM responses for formal logic and theorem proof tasks without human annotation. Although this data synthesis is feasible only for tasks compatible with formal verification, we observe that LLM-based PRMs trained on our dataset exhibit cross-task generalization, improving verification across diverse reasoning tasks. Specifically, PRMs trained with FoVer significantly outperform baseline PRMs based on the original LLMs and achieve competitive or superior results compared to state-of-the-art PRMs trained on labels annotated by humans or stronger models, as measured by step-level verification on ProcessBench and Best-of-K performance across 12 reasoning benchmarks, including MATH, AIME, ANLI, MMLU, and BBH. The datasets, models, and code are provided at this https URL. 

---
# Aligning Dialogue Agents with Global Feedback via Large Language Model Reward Decomposition 

**Authors**: Dong Won Lee, Hae Won Park, Cynthia Breazeal, Louis-Philippe Morency  

**Link**: [PDF](https://arxiv.org/pdf/2505.15922)  

**Abstract**: We propose a large language model based reward decomposition framework for aligning dialogue agents using only a single session-level feedback signal. We leverage the reasoning capabilities of a frozen, pretrained large language model (LLM) to infer fine-grained local implicit rewards by decomposing global, session-level feedback. Our first text-only variant prompts the LLM to perform reward decomposition using only the dialogue transcript. The second multimodal variant incorporates additional behavioral cues, such as pitch, gaze, and facial affect, expressed as natural language descriptions. These inferred turn-level rewards are distilled into a lightweight reward model, which we utilize for RL-based fine-tuning for dialogue generation. We evaluate both text-only and multimodal variants against state-of-the-art reward decomposition methods and demonstrate notable improvements in human evaluations of conversation quality, suggesting that LLMs are strong reward decomposers that obviate the need for manual reward shaping and granular human feedback. 

---
# UFT: Unifying Supervised and Reinforcement Fine-Tuning 

**Authors**: Mingyang Liu, Gabriele Farina, Asuman Ozdaglar  

**Link**: [PDF](https://arxiv.org/pdf/2505.16984)  

**Abstract**: Post-training has demonstrated its importance in enhancing the reasoning capabilities of large language models (LLMs). The primary post-training methods can be categorized into supervised fine-tuning (SFT) and reinforcement fine-tuning (RFT). SFT is efficient and well-suited for small language models, but it may lead to overfitting and limit the reasoning abilities of larger models. In contrast, RFT generally yields better generalization but depends heavily on the strength of the base model. To address the limitations of SFT and RFT, we propose Unified Fine-Tuning (UFT), a novel post-training paradigm that unifies SFT and RFT into a single, integrated process. UFT enables the model to effectively explore solutions while incorporating informative supervision signals, bridging the gap between memorizing and thinking underlying existing methods. Notably, UFT outperforms both SFT and RFT in general, regardless of model sizes. Furthermore, we theoretically prove that UFT breaks RFT's inherent exponential sample complexity bottleneck, showing for the first time that unified training can exponentially accelerate convergence on long-horizon reasoning tasks. 

---
# LLaDA-V: Large Language Diffusion Models with Visual Instruction Tuning 

**Authors**: Zebin You, Shen Nie, Xiaolu Zhang, Jun Hu, Jun Zhou, Zhiwu Lu, Ji-Rong Wen, Chongxuan Li  

**Link**: [PDF](https://arxiv.org/pdf/2505.16933)  

**Abstract**: In this work, we introduce LLaDA-V, a purely diffusion-based Multimodal Large Language Model (MLLM) that integrates visual instruction tuning with masked diffusion models, representing a departure from the autoregressive paradigms dominant in current multimodal approaches. Built upon LLaDA, a representative large language diffusion model, LLaDA-V incorporates a vision encoder and MLP connector that projects visual features into the language embedding space, enabling effective multimodal alignment. Our empirical investigation reveals several intriguing results: First, LLaDA-V demonstrates promising multimodal performance despite its language model being weaker on purely textual tasks than counterparts like LLaMA3-8B and Qwen2-7B. When trained on the same instruction data, LLaDA-V is highly competitive to LLaMA3-V across multimodal tasks with better data scalability. It also narrows the performance gap to Qwen2-VL, suggesting the effectiveness of its architecture for multimodal tasks. Second, LLaDA-V achieves state-of-the-art performance in multimodal understanding compared to existing hybrid autoregressive-diffusion and purely diffusion-based MLLMs. Our findings suggest that large language diffusion models show promise in multimodal contexts and warrant further investigation in future research. Project page and codes: this https URL. 

---
# Multi-SpatialMLLM: Multi-Frame Spatial Understanding with Multi-Modal Large Language Models 

**Authors**: Runsen Xu, Weiyao Wang, Hao Tang, Xingyu Chen, Xiaodong Wang, Fu-Jen Chu, Dahua Lin, Matt Feiszli, Kevin J. Liang  

**Link**: [PDF](https://arxiv.org/pdf/2505.17015)  

**Abstract**: Multi-modal large language models (MLLMs) have rapidly advanced in visual tasks, yet their spatial understanding remains limited to single images, leaving them ill-suited for robotics and other real-world applications that require multi-frame reasoning. In this paper, we propose a framework to equip MLLMs with robust multi-frame spatial understanding by integrating depth perception, visual correspondence, and dynamic perception. Central to our approach is the MultiSPA dataset, a novel, large-scale collection of more than 27 million samples spanning diverse 3D and 4D scenes. Alongside MultiSPA, we introduce a comprehensive benchmark that tests a wide spectrum of spatial tasks under uniform metrics. Our resulting model, Multi-SpatialMLLM, achieves significant gains over baselines and proprietary systems, demonstrating scalable, generalizable multi-frame reasoning. We further observe multi-task benefits and early indications of emergent capabilities in challenging scenarios, and showcase how our model can serve as a multi-frame reward annotator for robotics. 

---
# CTRAP: Embedding Collapse Trap to Safeguard Large Language Models from Harmful Fine-Tuning 

**Authors**: Biao Yi, Tiansheng Huang, Baolei Zhang, Tong Li, Lihai Nie, Zheli Liu, Li Shen  

**Link**: [PDF](https://arxiv.org/pdf/2505.16559)  

**Abstract**: Fine-tuning-as-a-service, while commercially successful for Large Language Model (LLM) providers, exposes models to harmful fine-tuning attacks. As a widely explored defense paradigm against such attacks, unlearning attempts to remove malicious knowledge from LLMs, thereby essentially preventing them from being used to perform malicious tasks. However, we highlight a critical flaw: the powerful general adaptability of LLMs allows them to easily bypass selective unlearning by rapidly relearning or repurposing their capabilities for harmful tasks. To address this fundamental limitation, we propose a paradigm shift: instead of selective removal, we advocate for inducing model collapse--effectively forcing the model to "unlearn everything"--specifically in response to updates characteristic of malicious adaptation. This collapse directly neutralizes the very general capabilities that attackers exploit, tackling the core issue unaddressed by selective unlearning. We introduce the Collapse Trap (CTRAP) as a practical mechanism to implement this concept conditionally. Embedded during alignment, CTRAP pre-configures the model's reaction to subsequent fine-tuning dynamics. If updates during fine-tuning constitute a persistent attempt to reverse safety alignment, the pre-configured trap triggers a progressive degradation of the model's core language modeling abilities, ultimately rendering it inert and useless for the attacker. Crucially, this collapse mechanism remains dormant during benign fine-tuning, ensuring the model's utility and general capabilities are preserved for legitimate users. Extensive empirical results demonstrate that CTRAP effectively counters harmful fine-tuning risks across various LLMs and attack settings, while maintaining high performance in benign scenarios. Our code is available at this https URL. 

---
# AdaSTaR: Adaptive Data Sampling for Training Self-Taught Reasoners 

**Authors**: Woosung Koh, Wonbeen Oh, Jaein Jang, MinHyung Lee, Hyeongjin Kim, Ah Yeon Kim, Joonkee Kim, Junghyun Lee, Taehyeon Kim, Se-Young Yun  

**Link**: [PDF](https://arxiv.org/pdf/2505.16322)  

**Abstract**: Self-Taught Reasoners (STaR), synonymously known as Rejection sampling Fine-Tuning (RFT), is an integral part of the training pipeline of self-improving reasoning Language Models (LMs). The self-improving mechanism often employs random observation (data) sampling. However, this results in trained observation imbalance; inefficiently over-training on solved examples while under-training on challenging ones. In response, we introduce Adaptive STaR (AdaSTaR), a novel algorithm that rectifies this by integrating two adaptive sampling principles: (1) Adaptive Sampling for Diversity: promoting balanced training across observations, and (2) Adaptive Sampling for Curriculum: dynamically adjusting data difficulty to match the model's evolving strength. Across six benchmarks, AdaSTaR achieves best test accuracy in all instances (6/6) and reduces training FLOPs by an average of 58.6% against an extensive list of baselines. These improvements in performance and efficiency generalize to different pre-trained LMs and larger models, paving the way for more efficient and effective self-improving LMs. 

---
# A Survey of Large Language Models for Text-Guided Molecular Discovery: from Molecule Generation to Optimization 

**Authors**: Ziqing Wang, Kexin Zhang, Zihan Zhao, Yibo Wen, Abhishek Pandey, Han Liu, Kaize Ding  

**Link**: [PDF](https://arxiv.org/pdf/2505.16094)  

**Abstract**: Large language models (LLMs) are introducing a paradigm shift in molecular discovery by enabling text-guided interaction with chemical spaces through natural language, symbolic notations, with emerging extensions to incorporate multi-modal inputs. To advance the new field of LLM for molecular discovery, this survey provides an up-to-date and forward-looking review of the emerging use of LLMs for two central tasks: molecule generation and molecule optimization. Based on our proposed taxonomy for both problems, we analyze representative techniques in each category, highlighting how LLM capabilities are leveraged across different learning settings. In addition, we include the commonly used datasets and evaluation protocols. We conclude by discussing key challenges and future directions, positioning this survey as a resource for researchers working at the intersection of LLMs and molecular science. A continuously updated reading list is available at this https URL. 

---
# Meta-PerSER: Few-Shot Listener Personalized Speech Emotion Recognition via Meta-learning 

**Authors**: Liang-Yeh Shen, Shi-Xin Fang, Yi-Cheng Lin, Huang-Cheng Chou, Hung-yi Lee  

**Link**: [PDF](https://arxiv.org/pdf/2505.16220)  

**Abstract**: This paper introduces Meta-PerSER, a novel meta-learning framework that personalizes Speech Emotion Recognition (SER) by adapting to each listener's unique way of interpreting emotion. Conventional SER systems rely on aggregated annotations, which often overlook individual subtleties and lead to inconsistent predictions. In contrast, Meta-PerSER leverages a Model-Agnostic Meta-Learning (MAML) approach enhanced with Combined-Set Meta-Training, Derivative Annealing, and per-layer per-step learning rates, enabling rapid adaptation with only a few labeled examples. By integrating robust representations from pre-trained self-supervised models, our framework first captures general emotional cues and then fine-tunes itself to personal annotation styles. Experiments on the IEMOCAP corpus demonstrate that Meta-PerSER significantly outperforms baseline methods in both seen and unseen data scenarios, highlighting its promise for personalized emotion recognition. 

---
# OViP: Online Vision-Language Preference Learning 

**Authors**: Shujun Liu, Siyuan Wang, Zejun Li, Jianxiang Wang, Cheng Zeng, Zhongyu Wei  

**Link**: [PDF](https://arxiv.org/pdf/2505.15963)  

**Abstract**: Large vision-language models (LVLMs) remain vulnerable to hallucination, often generating content misaligned with visual inputs. While recent approaches advance multi-modal Direct Preference Optimization (DPO) to mitigate hallucination, they typically rely on predefined or randomly edited negative samples that fail to reflect actual model errors, limiting training efficacy. In this work, we propose an Online Vision-language Preference Learning (OViP) framework that dynamically constructs contrastive training data based on the model's own hallucinated outputs. By identifying semantic differences between sampled response pairs and synthesizing negative images using a diffusion model, OViP generates more relevant supervision signals in real time. This failure-driven training enables adaptive alignment of both textual and visual preferences. Moreover, we refine existing evaluation protocols to better capture the trade-off between hallucination suppression and expressiveness. Experiments on hallucination and general benchmarks demonstrate that OViP effectively reduces hallucinations while preserving core multi-modal capabilities. 

---
# Aug2Search: Enhancing Facebook Marketplace Search with LLM-Generated Synthetic Data Augmentation 

**Authors**: Ruijie Xi, He Ba, Hao Yuan, Rishu Agrawal, Arul Prakash  

**Link**: [PDF](https://arxiv.org/pdf/2505.16065)  

**Abstract**: Embedding-Based Retrieval (EBR) is an important technique in modern search engines, enabling semantic match between search queries and relevant results. However, search logging data on platforms like Facebook Marketplace lacks the diversity and details needed for effective EBR model training, limiting the models' ability to capture nuanced search patterns. To address this challenge, we propose Aug2Search, an EBR-based framework leveraging synthetic data generated by Generative AI (GenAI) models, in a multimodal and multitask approach to optimize query-product relevance. This paper investigates the capabilities of GenAI, particularly Large Language Models (LLMs), in generating high-quality synthetic data, and analyzing its impact on enhancing EBR models. We conducted experiments using eight Llama models and 100 million data points from Facebook Marketplace logs. Our synthetic data generation follows three strategies: (1) generate queries, (2) enhance product listings, and (3) generate queries from enhanced listings. We train EBR models on three different datasets: sampled engagement data or original data ((e.g., "Click" and "Listing Interactions")), synthetic data, and a mixture of both engagement and synthetic data to assess their performance across various training sets. Our findings underscore the robustness of Llama models in producing synthetic queries and listings with high coherence, relevance, and diversity, while maintaining low levels of hallucination. Aug2Search achieves an improvement of up to 4% in ROC_AUC with 100 million synthetic data samples, demonstrating the effectiveness of our approach. Moreover, our experiments reveal that with the same volume of training data, models trained exclusively on synthetic data often outperform those trained on original data only or a mixture of original and synthetic data. 

---
# InfoDeepSeek: Benchmarking Agentic Information Seeking for Retrieval-Augmented Generation 

**Authors**: Yunjia Xi, Jianghao Lin, Menghui Zhu, Yongzhao Xiao, Zhuoying Ou, Jiaqi Liu, Tong Wan, Bo Chen, Weiwen Liu, Yasheng Wang, Ruiming Tang, Weinan Zhang, Yong Yu  

**Link**: [PDF](https://arxiv.org/pdf/2505.15872)  

**Abstract**: Retrieval-Augmented Generation (RAG) enhances large language models (LLMs) by grounding responses with retrieved information. As an emerging paradigm, Agentic RAG further enhances this process by introducing autonomous LLM agents into the information seeking process. However, existing benchmarks fall short in evaluating such systems, as they are confined to a static retrieval environment with a fixed, limited corpus} and simple queries that fail to elicit agentic behavior. Moreover, their evaluation protocols assess information seeking effectiveness by pre-defined gold sets of documents, making them unsuitable for the open-ended and dynamic nature of real-world web environments. To bridge this gap, we present InfoDeepSeek, a new benchmark with challenging questions designed for assessing agentic information seeking in real-world, dynamic web environments. We propose a systematic methodology for constructing challenging queries satisfying the criteria of determinacy, difficulty, and diversity. Based on this, we develop the first evaluation framework tailored to dynamic agentic information seeking, including fine-grained metrics about the accuracy, utility, and compactness of information seeking outcomes. Through extensive experiments across LLMs, search engines, and question types, InfoDeepSeek reveals nuanced agent behaviors and offers actionable insights for future research. 

---
# Pixel Reasoner: Incentivizing Pixel-Space Reasoning with Curiosity-Driven Reinforcement Learning 

**Authors**: Alex Su, Haozhe Wang, Weimin Ren, Fangzhen Lin, Wenhu Chen  

**Link**: [PDF](https://arxiv.org/pdf/2505.15966)  

**Abstract**: Chain-of-thought reasoning has significantly improved the performance of Large Language Models (LLMs) across various domains. However, this reasoning process has been confined exclusively to textual space, limiting its effectiveness in visually intensive tasks. To address this limitation, we introduce the concept of reasoning in the pixel-space. Within this novel framework, Vision-Language Models (VLMs) are equipped with a suite of visual reasoning operations, such as zoom-in and select-frame. These operations enable VLMs to directly inspect, interrogate, and infer from visual evidences, thereby enhancing reasoning fidelity for visual tasks. Cultivating such pixel-space reasoning capabilities in VLMs presents notable challenges, including the model's initially imbalanced competence and its reluctance to adopt the newly introduced pixel-space operations. We address these challenges through a two-phase training approach. The first phase employs instruction tuning on synthesized reasoning traces to familiarize the model with the novel visual operations. Following this, a reinforcement learning (RL) phase leverages a curiosity-driven reward scheme to balance exploration between pixel-space reasoning and textual reasoning. With these visual operations, VLMs can interact with complex visual inputs, such as information-rich images or videos to proactively gather necessary information. We demonstrate that this approach significantly improves VLM performance across diverse visual reasoning benchmarks. Our 7B model, \model, achieves 84\% on V* bench, 74\% on TallyQA-Complex, and 84\% on InfographicsVQA, marking the highest accuracy achieved by any open-source model to date. These results highlight the importance of pixel-space reasoning and the effectiveness of our framework. 

---
# MAPS: A Multilingual Benchmark for Global Agent Performance and Security 

**Authors**: Omer Hofman, Oren Rachmil, Shamik Bose, Vikas Pahuja, Jonathan Brokman, Toshiya Shimizu, Trisha Starostina, Kelly Marchisio, Seraphina Goldfarb-Tarrant, Roman Vainshtein  

**Link**: [PDF](https://arxiv.org/pdf/2505.15935)  

**Abstract**: Agentic AI systems, which build on Large Language Models (LLMs) and interact with tools and memory, have rapidly advanced in capability and scope. Yet, since LLMs have been shown to struggle in multilingual settings, typically resulting in lower performance and reduced safety, agentic systems risk inheriting these limitations. This raises concerns about the global accessibility of such systems, as users interacting in languages other than English may encounter unreliable or security-critical agent behavior. Despite growing interest in evaluating agentic AI, existing benchmarks focus exclusively on English, leaving multilingual settings unexplored. To address this gap, we propose MAPS, a multilingual benchmark suite designed to evaluate agentic AI systems across diverse languages and tasks. MAPS builds on four widely used agentic benchmarks - GAIA (real-world tasks), SWE-bench (code generation), MATH (mathematical reasoning), and the Agent Security Benchmark (security). We translate each dataset into ten diverse languages, resulting in 805 unique tasks and 8,855 total language-specific instances. Our benchmark suite enables a systematic analysis of how multilingual contexts affect agent performance and robustness. Empirically, we observe consistent degradation in both performance and security when transitioning from English to other languages, with severity varying by task and correlating with the amount of translated input. Building on these findings, we provide actionable recommendations to guide agentic AI systems development and assessment under multilingual settings. This work establishes a standardized evaluation framework, encouraging future research towards equitable, reliable, and globally accessible agentic AI. MAPS benchmark suite is publicly available at this https URL 

---
# DeepRec: Towards a Deep Dive Into the Item Space with Large Language Model Based Recommendation 

**Authors**: Bowen Zheng, Xiaolei Wang, Enze Liu, Xi Wang, Lu Hongyu, Yu Chen, Wayne Xin Zhao, Ji-Rong Wen  

**Link**: [PDF](https://arxiv.org/pdf/2505.16810)  

**Abstract**: Recently, large language models (LLMs) have been introduced into recommender systems (RSs), either to enhance traditional recommendation models (TRMs) or serve as recommendation backbones. However, existing LLM-based RSs often do not fully exploit the complementary advantages of LLMs (e.g., world knowledge and reasoning) and TRMs (e.g., recommendation-specific knowledge and efficiency) to fully explore the item space. To address this, we propose DeepRec, a novel LLM-based RS that enables autonomous multi-turn interactions between LLMs and TRMs for deep exploration of the item space. In each interaction turn, LLMs reason over user preferences and interact with TRMs to retrieve candidate items. After multi-turn interactions, LLMs rank the retrieved items to generate the final recommendations. We adopt reinforcement learning(RL) based optimization and propose novel designs from three aspects: recommendation model based data rollout, recommendation-oriented hierarchical rewards, and a two-stage RL training strategy. For data rollout, we introduce a preference-aware TRM, with which LLMs interact to construct trajectory data. For rewards, we design a hierarchical reward function that involves both process-level and outcome-level rewards to optimize the interaction process and recommendation performance, respectively. For RL training, we develop a two-stage training strategy, where the first stage aims to guide LLMs to interact with TRMs and the second stage focuses on performance improvement. Experiments on public datasets demonstrate that DeepRec significantly outperforms both traditional and LLM-based baselines, offering a new paradigm for deep exploration in recommendation systems. 

---
# Chain-of-Thought Poisoning Attacks against R1-based Retrieval-Augmented Generation Systems 

**Authors**: Hongru Song, Yu-an Liu, Ruqing Zhang, Jiafeng Guo, Yixing Fan  

**Link**: [PDF](https://arxiv.org/pdf/2505.16367)  

**Abstract**: Retrieval-augmented generation (RAG) systems can effectively mitigate the hallucination problem of large language models (LLMs),but they also possess inherent vulnerabilities. Identifying these weaknesses before the large-scale real-world deployment of RAG systems is of great importance, as it lays the foundation for building more secure and robust RAG systems in the future. Existing adversarial attack methods typically exploit knowledge base poisoning to probe the vulnerabilities of RAG systems, which can effectively deceive standard RAG models. However, with the rapid advancement of deep reasoning capabilities in modern LLMs, previous approaches that merely inject incorrect knowledge are inadequate when attacking RAG systems equipped with deep reasoning abilities. Inspired by the deep thinking capabilities of LLMs, this paper extracts reasoning process templates from R1-based RAG systems, uses these templates to wrap erroneous knowledge into adversarial documents, and injects them into the knowledge base to attack RAG systems. The key idea of our approach is that adversarial documents, by simulating the chain-of-thought patterns aligned with the model's training signals, may be misinterpreted by the model as authentic historical reasoning processes, thus increasing their likelihood of being referenced. Experiments conducted on the MS MARCO passage ranking dataset demonstrate the effectiveness of our proposed method. 

---
# Walk&Retrieve: Simple Yet Effective Zero-shot Retrieval-Augmented Generation via Knowledge Graph Walks 

**Authors**: Martin Böckling, Heiko Paulheim, Andreea Iana  

**Link**: [PDF](https://arxiv.org/pdf/2505.16849)  

**Abstract**: Large Language Models (LLMs) have showcased impressive reasoning abilities, but often suffer from hallucinations or outdated knowledge. Knowledge Graph (KG)-based Retrieval-Augmented Generation (RAG) remedies these shortcomings by grounding LLM responses in structured external information from a knowledge base. However, many KG-based RAG approaches struggle with (i) aligning KG and textual representations, (ii) balancing retrieval accuracy and efficiency, and (iii) adapting to dynamically updated KGs. In this work, we introduce Walk&Retrieve, a simple yet effective KG-based framework that leverages walk-based graph traversal and knowledge verbalization for corpus generation for zero-shot RAG. Built around efficient KG walks, our method does not require fine-tuning on domain-specific data, enabling seamless adaptation to KG updates, reducing computational overhead, and allowing integration with any off-the-shelf backbone LLM. Despite its simplicity, Walk&Retrieve performs competitively, often outperforming existing RAG systems in response accuracy and hallucination reduction. Moreover, it demonstrates lower query latency and robust scalability to large KGs, highlighting the potential of lightweight retrieval strategies as strong baselines for future RAG research. 

---
# Text-to-Pipeline: Bridging Natural Language and Data Preparation Pipelines 

**Authors**: Yuhang Ge, Yachuan Liu, Yuren Mao, Yunjun Gao  

**Link**: [PDF](https://arxiv.org/pdf/2505.15874)  

**Abstract**: Data preparation (DP) transforms raw data into a form suitable for downstream applications, typically by composing operations into executable pipelines. Building such pipelines is time-consuming and requires sophisticated programming skills. If we can build the pipelines with natural language (NL), the technical barrier of DP will be significantly reduced. However, constructing DP pipelines from NL instructions remains underexplored. To fill the gap, we introduce Text-to-Pipeline, a new task that translates NL data preparation instructions into DP pipelines. Furthermore, we develop a benchmark named PARROT to support systematic evaluation. To simulate realistic DP scenarios, we mined transformation patterns from production pipelines and instantiated them on 23,009 real-world tables collected from six public sources. The resulting benchmark comprises ~18,000 pipelines covering 16 core DP operators. We evaluated cutting-edge large language models on PARROTand observed that they only solved 72.86% of the cases, revealing notable limitations in instruction understanding and multi-step reasoning. To address this, we propose Pipeline-Agent, a stronger baseline that iteratively predicts and executes operations with intermediate table feedback, achieving the best performance of 76.17%. Despite this improvement, there remains substantial room for progress on Text-to-Pipeline. Our data, codes, and evaluation tools are available at this https URL. 

---
# AutoData: A Multi-Agent System for Open Web Data Collection 

**Authors**: Tianyi Ma, Yiyue Qian, Zheyuan Zhang, Zehong Wang, Xiaoye Qian, Feifan Bai, Yifan Ding, Xuwei Luo, Shinan Zhang, Keerthiram Murugesan, Chuxu Zhang, Yanfang Ye  

**Link**: [PDF](https://arxiv.org/pdf/2505.15859)  

**Abstract**: The exponential growth of data-driven systems and AI technologies has intensified the demand for high-quality web-sourced datasets. While existing datasets have proven valuable, conventional web data collection approaches face significant limitations in terms of human effort and scalability. Current data-collecting solutions fall into two categories: wrapper-based methods that struggle with adaptability and reproducibility, and large language model (LLM)-based approaches that incur substantial computational and financial costs. To address these challenges, we propose AutoData, a novel multi-agent system for Automated web Data collection, that requires minimal human intervention, i.e., only necessitating a natural language instruction specifying the desired dataset. In addition, AutoData is designed with a robust multi-agent architecture, featuring a novel oriented message hypergraph coordinated by a central task manager, to efficiently organize agents across research and development squads. Besides, we introduce a novel hypergraph cache system to advance the multi-agent collaboration process that enables efficient automated data collection and mitigates the token cost issues prevalent in existing LLM-based systems. Moreover, we introduce Instruct2DS, a new benchmark dataset supporting live data collection from web sources across three domains: academic, finance, and sports. Comprehensive evaluations over Instruct2DS and three existing benchmark datasets demonstrate AutoData's superior performance compared to baseline methods. Case studies on challenging tasks such as picture book collection and paper extraction from surveys further validate its applicability. Our source code and dataset are available at this https URL. 

---
