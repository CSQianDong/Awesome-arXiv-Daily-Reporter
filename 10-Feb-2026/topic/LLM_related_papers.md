# Scalable Delphi: Large Language Models for Structured Risk Estimation 

**Authors**: Tobias Lorenz, Mario Fritz  

**Link**: [PDF](https://arxiv.org/pdf/2602.08889)  

**Abstract**: Quantitative risk assessment in high-stakes domains relies on structured expert elicitation to estimate unobservable properties. The gold standard - the Delphi method - produces calibrated, auditable judgments but requires months of coordination and specialist time, placing rigorous risk assessment out of reach for most applications. We investigate whether Large Language Models (LLMs) can serve as scalable proxies for structured expert elicitation. We propose Scalable Delphi, adapting the classical protocol for LLMs with diverse expert personas, iterative refinement, and rationale sharing. Because target quantities are typically unobservable, we develop an evaluation framework based on necessary conditions: calibration against verifiable proxies, sensitivity to evidence, and alignment with human expert judgment. We evaluate in the domain of AI-augmented cybersecurity risk, using three capability benchmarks and independent human elicitation studies. LLM panels achieve strong correlations with benchmark ground truth (Pearson r=0.87-0.95), improve systematically as evidence is added, and align with human expert panels - in one comparison, closer to a human panel than the two human panels are to each other. This demonstrates that LLM-based elicitation can extend structured expert judgment to settings where traditional methods are infeasible, reducing elicitation time from months to minutes. 

---
# iGRPO: Self-Feedback-Driven LLM Reasoning 

**Authors**: Ali Hatamizadeh, Shrimai Prabhumoye, Igor Gitman, Ximing Lu, Seungju Han, Wei Ping, Yejin Choi, Jan Kautz  

**Link**: [PDF](https://arxiv.org/pdf/2602.09000)  

**Abstract**: Large Language Models (LLMs) have shown promise in solving complex mathematical problems, yet they still fall short of producing accurate and consistent solutions. Reinforcement Learning (RL) is a framework for aligning these models with task-specific rewards, improving overall quality and reliability. Group Relative Policy Optimization (GRPO) is an efficient, value-function-free alternative to Proximal Policy Optimization (PPO) that leverages group-relative reward normalization. We introduce Iterative Group Relative Policy Optimization (iGRPO), a two-stage extension of GRPO that adds dynamic self-conditioning through model-generated drafts. In Stage 1, iGRPO samples multiple exploratory drafts and selects the highest-reward draft using the same scalar reward signal used for optimization. In Stage 2, it appends this best draft to the original prompt and applies a GRPO-style update on draft-conditioned refinements, training the policy to improve beyond its strongest prior attempt. Under matched rollout budgets, iGRPO consistently outperforms GRPO across base models (e.g., Nemotron-H-8B-Base-8K and DeepSeek-R1 Distilled), validating its effectiveness on diverse reasoning benchmarks. Moreover, applying iGRPO to OpenReasoning-Nemotron-7B trained on AceReason-Math achieves new state-of-the-art results of 85.62\% and 79.64\% on AIME24 and AIME25, respectively. Ablations further show that the refinement wrapper generalizes beyond GRPO variants, benefits from a generative judge, and alters learning dynamics by delaying entropy collapse. These results underscore the potential of iterative, self-feedback-based RL for advancing verifiable mathematical reasoning. 

---
# CoRefine: Confidence-Guided Self-Refinement for Adaptive Test-Time Compute 

**Authors**: Chen Jin, Ryutaro Tanno, Tom Diethe, Philip Teare  

**Link**: [PDF](https://arxiv.org/pdf/2602.08948)  

**Abstract**: Large Language Models (LLMs) often rely on test-time scaling via parallel decoding (for example, 512 samples) to boost reasoning accuracy, but this incurs substantial compute. We introduce CoRefine, a confidence-guided self-refinement method that achieves competitive accuracy using a fraction of the tokens via a lightweight 211k-parameter Conv1D controller atop a frozen LLM. The controller consumes full-trace confidence to decide whether to halt, re-examine, or try a different approach, enabling targeted self-correction with an average of 2.7 refinement steps per problem and roughly 190-fold token reduction relative to 512-sample baselines. Across diverse reasoning benchmarks and three open-source models, the controller achieves 92.6 percent precision when it confidently halts, indicating that confidence dynamics reliably signal correctness without ground-truth verification. We extend this to CoRefine-Tree, a hybrid sequential-parallel variant that adaptively balances exploration and exploitation, with easy serving integration and verifier compatibility. By treating confidence as a control signal rather than a correctness guarantee, CoRefine provides a modular primitive for scalable reasoning and agentic settings with imperfect verifiers. 

---
# Root Cause Analysis Method Based on Large Language Models with Residual Connection Structures 

**Authors**: Liming Zhou, Ailing Liu, Hongwei Liu, Min He, Heng Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2602.08804)  

**Abstract**: Root cause localization remain challenging in complex and large-scale microservice architectures. The complex fault propagation among microservices and the high dimensionality of telemetry data, including metrics, logs, and traces, limit the effectiveness of existing root cause analysis (RCA) methods. In this paper, a residual-connection-based RCA method using large language model (LLM), named RC-LLM, is proposed. A residual-like hierarchical fusion structure is designed to integrate multi-source telemetry data, while the contextual reasoning capability of large language models is leveraged to model temporal and cross-microservice causal dependencies. Experimental results on CCF-AIOps microservice datasets demonstrate that RC-LLM achieves strong accuracy and efficiency in root cause analysis. 

---
# PRISM: A Principled Framework for Multi-Agent Reasoning via Gain Decomposition 

**Authors**: Yiming Yang, Zhuoyuan Li, Fanxiang Zeng, Hao Fu, Yue Liu  

**Link**: [PDF](https://arxiv.org/pdf/2602.08586)  

**Abstract**: Multi-agent collaboration has emerged as a promising paradigm for enhancing reasoning capabilities of Large Language Models (LLMs). However, existing approaches remain largely heuristic, lacking principled guidance on what drives performance gains and how to systematically optimize multi-agent reasoning. Specifically, it remains unclear why multi-agent collaboration outperforms single-agent reasoning and which design choices contribute most to these gains, making it difficult to build better systems.
We address this gap by introducing a unified theoretical framework that decomposes multi-agent reasoning gains into three conceptually independent dimensions: Exploration for diverse solution coverage, Information for high-fidelity feedback, and Aggregation for principled consensus. Through this lens, existing methods can be understood as special cases that optimize only subsets of these dimensions. Building upon this decomposition, a novel framework called PRISM (Propose-Review-Integrate Synthesis for Multi-agent Reasoning) is proposed, which jointly maximizes all three dimensions through role-based diversity, execution-grounded feedback with evidence-based cross-evaluation, and iterative synthesis with closed-loop validation. Extensive experiments across mathematical reasoning, code generation, and function calling benchmarks demonstrate that PRISM achieves state-of-the-art performance with superior compute-efficiency compared to methods optimizing partial dimensions. The theoretical framework provides actionable design principles for future multi-agent reasoning systems. 

---
# Reinforcement Inference: Leveraging Uncertainty for Self-Correcting Language Model Reasoning 

**Authors**: Xinhai Sun  

**Link**: [PDF](https://arxiv.org/pdf/2602.08520)  

**Abstract**: Modern large language models (LLMs) are often evaluated and deployed under a \emph{one-shot, greedy} inference protocol, especially in professional settings that require deterministic behavior. This regime can systematically under-estimate a fixed model's true capability: many errors arise not from missing knowledge, but from premature commitment under internal ambiguity. We introduce \emph{Reinforcement Inference}, an entropy-aware inference-time control strategy that uses the model's own uncertainty to selectively invoke a second, more deliberate reasoning attempt, enabling stronger performance \emph{without any retraining}.
On 12,032 MMLU-Pro questions across 14 subjects, using DeepSeek-v3.2 with deterministic decoding in a zero-shot setting, Reinforcement Inference improves accuracy from 60.72\% to 84.03\%, while only incurring 61.06\% additional inference calls. A 100\% re-asking ablation reaches 84.35\%, indicating that uncertainty-aware selection captures most of the attainable improvement with substantially less compute. Moreover, a \emph{prompt-only} ablation underperforms the baseline, suggesting that the gains are not explained by generic `` your output had high entropy, think step-by-step'' prompting alone.
Beyond providing a practical inference-time upgrade, our results suggest a broader \emph{entropy-aware} paradigm for measuring and expanding model capability: because modern decoder-based models generate outputs autoregressively, entropy and related confidence measures arise naturally as first-class control signals during generation. The resulting gap between one-pass greedy inference and uncertainty-conditioned deliberation offers a diagnostic lens on an LLM's latent reasoning horizon and motivates future training objectives that explicitly constrain correctness--confidence alignment. 

---
# Does Your Reasoning Model Implicitly Know When to Stop Thinking? 

**Authors**: Zixuan Huang, Xin Xia, Yuxi Ren, Jianbin Zheng, Xuanda Wang, Zhixia Zhang, Hongyan Xie, Songshi Liang, Zehao Chen, Xuefeng Xiao, Fuzhen Zhuang, Jianxin Li, Yikun Ban, Deqing Wang  

**Link**: [PDF](https://arxiv.org/pdf/2602.08354)  

**Abstract**: Recent advancements in large reasoning models (LRMs) have greatly improved their capabilities on complex reasoning tasks through Long Chains of Thought (CoTs). However, this approach often results in substantial redundancy, impairing computational efficiency and causing significant delays in real-time applications. Recent studies show that longer reasoning chains are frequently uncorrelated with correctness and can even be detrimental to accuracy. In a further in-depth analysis of this phenomenon, we surprisingly uncover and empirically verify that LRMs implicitly know the appropriate time to stop thinking, while this capability is obscured by current sampling paradigms. Motivated by this, we introduce SAGE (Self-Aware Guided Efficient Reasoning), a novel sampling paradigm that unleashes this efficient reasoning potential. Furthermore, integrating SAGE as mixed sampling into group-based reinforcement learning (SAGE-RL) enables SAGE-RL to effectively incorporate SAGE-discovered efficient reasoning patterns into standard pass@1 inference, markedly enhancing both the reasoning accuracy and efficiency of LRMs across multiple challenging mathematical benchmarks. 

---
# Puda: Private User Dataset Agent for User-Sovereign and Privacy-Preserving Personalized AI 

**Authors**: Akinori Maeda, Yuto Sekiya, Sota Sugimura, Tomoya Asai, Yu Tsuda, Kohei Ikeda, Hiroshi Fujii, Kohei Watanabe  

**Link**: [PDF](https://arxiv.org/pdf/2602.08268)  

**Abstract**: Personal data centralization among dominant platform providers including search engines, social networking services, and e-commerce has created siloed ecosystems that restrict user sovereignty, thereby impeding data use across services. Meanwhile, the rapid proliferation of Large Language Model (LLM)-based agents has intensified demand for highly personalized services that require the dynamic provision of diverse personal data. This presents a significant challenge: balancing the utilization of such data with privacy protection. To address this challenge, we propose Puda (Private User Dataset Agent), a user-sovereign architecture that aggregates data across services and enables client-side management. Puda allows users to control data sharing at three privacy levels: (i) Detailed Browsing History, (ii) Extracted Keywords, and (iii) Predefined Category Subsets. We implemented Puda as a browser-based system that serves as a common platform across diverse services and evaluated it through a personalized travel planning task. Our results show that providing Predefined Category Subsets achieves 97.2% of the personalization performance (evaluated via an LLM-as-a-Judge framework across three criteria) obtained when sharing Detailed Browsing History. These findings demonstrate that Puda enables effective multi-granularity management, offering practical choices to mitigate the privacy-personalization trade-off. Overall, Puda provides an AI-native foundation for user sovereignty, empowering users to safely leverage the full potential of personalized AI. 

---
# InfiCoEvalChain: A Blockchain-Based Decentralized Framework for Collaborative LLM Evaluation 

**Authors**: Yifan Yang, Jinjia Li, Kunxi Li, Puhao Zheng, Yuanyi Wang, Zheyan Qu, Yang Yu, Jianmin Wu, Ming Li, Hongxia Yang  

**Link**: [PDF](https://arxiv.org/pdf/2602.08229)  

**Abstract**: The rapid advancement of large language models (LLMs) demands increasingly reliable evaluation, yet current centralized evaluation suffers from opacity, overfitting, and hardware-induced variance. Our empirical analysis reveals an alarming inconsistency in existing evaluations: the standard deviation across ten repeated runs of a single model on HumanEval (1.67) actually exceeds the performance gap among the top-10 models on the official leaderboard (0.91), rendering current rankings statistically precarious. To mitigate these instabilities, we propose a decentralized evaluation framework that enables hardware and parameter diversity through large-scale benchmarking across heterogeneous compute nodes. By leveraging the blockchain-based protocol, the framework incentivizes global contributors to act as independent validators, using a robust reward system to ensure evaluation integrity and discourage dishonest participation. This collective verification transforms evaluation from a "centralized black box" into a "decentralized endorsement" where multi-party consensus and diverse inference environments yield a more stable, representative metric. Experimental results demonstrate that the decentralized evaluation framework reduces the standard deviation across ten runs on the same model to 0.28. This significant improvement over conventional frameworks ensures higher statistical confidence in model rankings. We have completely implemented this platform and will soon release it to the community. 

---
# Toward Formalizing LLM-Based Agent Designs through Structural Context Modeling and Semantic Dynamics Analysis 

**Authors**: Haoyu Jia, Kento Kawaharazuka, Kei Okada  

**Link**: [PDF](https://arxiv.org/pdf/2602.08276)  

**Abstract**: Current research on large language model (LLM) agents is fragmented: discussions of conceptual frameworks and methodological principles are frequently intertwined with low-level implementation details, causing both readers and authors to lose track amid a proliferation of superficially distinct concepts. We argue that this fragmentation largely stems from the absence of an analyzable, self-consistent formal model that enables implementation-independent characterization and comparison of LLM agents. To address this gap, we propose the \texttt{Structural Context Model}, a formal model for analyzing and comparing LLM agents from the perspective of context structure. Building upon this foundation, we introduce two complementary components that together span the full lifecycle of LLM agent research and development: (1) a declarative implementation framework; and (2) a sustainable agent engineering workflow, \texttt{Semantic Dynamics Analysis}. The proposed workflow provides principled insights into agent mechanisms and supports rapid, systematic design iteration. We demonstrate the effectiveness of the complete framework on dynamic variants of the monkey-banana problem, where agents engineered using our approach achieve up to a 32 percentage points improvement in success rate on the most challenging setting. 

---
# Do MLLMs Really See It: Reinforcing Visual Attention in Multimodal LLMs 

**Authors**: Siqu Ou, Tianrui Wan, Zhiyuan Zhao, Junyu Gao, Xuelong Li  

**Link**: [PDF](https://arxiv.org/pdf/2602.08241)  

**Abstract**: While chain-of-thought (CoT) reasoning has substantially improved multimodal large language models (MLLMs) on complex reasoning tasks, existing approaches largely rely on long textual reasoning trajectories and provide limited mechanisms for learning stable visual attention policies. Our analysis shows that current MLLMs exhibit weak visual focus: early-stage visual misalignment is rarely corrected during subsequent reasoning, leading to error propagation and failed inferences. We argue that this limitation stems from inadequate credit assignment for visual attention during training. To address this issue, we propose SAYO, a visual reasoning model trained with a reinforcement learning (RL) framework that introduces a region-level visual attention-based reward. This reward explicitly aligns optimization signals with visually grounded reasoning steps, enabling the model to learn more reliable attention behaviors. Extensive experiments across multiple multimodal benchmarks demonstrate that SAYO consistently improves performance on diverse reasoning and perception tasks. 

---
# G-LNS: Generative Large Neighborhood Search for LLM-Based Automatic Heuristic Design 

**Authors**: Baoyun Zhao, He Wang, Liang Zeng  

**Link**: [PDF](https://arxiv.org/pdf/2602.08253)  

**Abstract**: While Large Language Models (LLMs) have recently shown promise in Automated Heuristic Design (AHD), existing approaches typically formulate AHD around constructive priority rules or parameterized local search guidance, thereby restricting the search space to fixed heuristic forms. Such designs offer limited capacity for structural exploration, making it difficult to escape deep local optima in complex Combinatorial Optimization Problems (COPs). In this work, we propose G-LNS, a generative evolutionary framework that extends LLM-based AHD to the automated design of Large Neighborhood Search (LNS) operators. Unlike prior methods that evolve heuristics in isolation, G-LNS leverages LLMs to co-evolve tightly coupled pairs of destroy and repair operators. A cooperative evaluation mechanism explicitly captures their interaction, enabling the discovery of complementary operator logic that jointly performs effective structural disruption and reconstruction. Extensive experiments on challenging COP benchmarks, such as Traveling Salesman Problems (TSP) and Capacitated Vehicle Routing Problems (CVRP), demonstrate that G-LNS significantly outperforms LLM-based AHD methods as well as strong classical solvers. The discovered heuristics not only achieve near-optimal solutions with reduced computational budgets but also exhibit robust generalization across diverse and unseen instance distributions. 

---
# Small Agent Group is the Future of Digital Health 

**Authors**: Yuqiao Meng, Luoxi Tang, Dazheng Zhang, Rafael Brens, Elvys J. Romero, Nancy Guo, Safa Elkefi, Zhaohan Xi  

**Link**: [PDF](https://arxiv.org/pdf/2602.08013)  

**Abstract**: The rapid adoption of large language models (LLMs) in digital health has been driven by a "scaling-first" philosophy, i.e., the assumption that clinical intelligence increases with model size and data. However, real-world clinical needs include not only effectiveness, but also reliability and reasonable deployment cost. Since clinical decision-making is inherently collaborative, we challenge the monolithic scaling paradigm and ask whether a Small Agent Group (SAG) can support better clinical reasoning. SAG shifts from single-model intelligence to collective expertise by distributing reasoning, evidence-based analysis, and critical audit through a collaborative deliberation process. To assess the clinical utility of SAG, we conduct extensive evaluations using diverse clinical metrics spanning effectiveness, reliability, and deployment cost. Our results show that SAG achieves superior performance compared to a single giant model, both with and without additional optimization or retrieval-augmented generation. These findings suggest that the synergistic reasoning represented by SAG can substitute for model parameter growth in clinical settings. Overall, SAG offers a scalable solution to digital health that better balances effectiveness, reliability, and deployment efficiency. 

---
# LOCA-bench: Benchmarking Language Agents Under Controllable and Extreme Context Growth 

**Authors**: Weihao Zeng, Yuzhen Huang, Junxian He  

**Link**: [PDF](https://arxiv.org/pdf/2602.07962)  

**Abstract**: Large language models (LLMs) are increasingly capable of carrying out long-running, real-world tasks. However, as the amount of context grows, their reliability often deteriorates, a phenomenon known as "context rot". Existing long-context benchmarks primarily focus on single-step settings that evaluate a model's ability to retrieve information from a long snippet. In realistic scenarios, however, LLMs often need to act as agents that explore environments, follow instructions and plans, extract useful information, and predict correct actions under a dynamically growing context. To assess language agents in such settings, we introduce LOCA-bench (a benchmark for LOng-Context Agents). Given a task prompt, LOCA-bench leverages automated and scalable control of environment states to regulate the agent's context length. This design enables LOCA-bench to extend the context length potentially to infinity in a controlled way while keeping the underlying task semantics fixed. LOCA-bench evaluates language agents as a combination of models and scaffolds, including various context management strategies. While agent performance generally degrades as the environment states grow more complex, advanced context management techniques can substantially improve the overall success rate. We open-source LOCA-bench to provide a platform for evaluating models and scaffolds in long-context, agentic scenarios: this https URL 

---
# IV Co-Scientist: Multi-Agent LLM Framework for Causal Instrumental Variable Discovery 

**Authors**: Ivaxi Sheth, Zhijing Jin, Bryan Wilder, Dominik Janzing, Mario Fritz  

**Link**: [PDF](https://arxiv.org/pdf/2602.07943)  

**Abstract**: In the presence of confounding between an endogenous variable and the outcome, instrumental variables (IVs) are used to isolate the causal effect of the endogenous variable. Identifying valid instruments requires interdisciplinary knowledge, creativity, and contextual understanding, making it a non-trivial task. In this paper, we investigate whether large language models (LLMs) can aid in this task. We perform a two-stage evaluation framework. First, we test whether LLMs can recover well-established instruments from the literature, assessing their ability to replicate standard reasoning. Second, we evaluate whether LLMs can identify and avoid instruments that have been empirically or theoretically discredited. Building on these results, we introduce IV Co-Scientist, a multi-agent system that proposes, critiques, and refines IVs for a given treatment-outcome pair. We also introduce a statistical test to contextualize consistency in the absence of ground truth. Our results show the potential of LLMs to discover valid instrumental variables from a large observational database. 

---
# MedCoG: Maximizing LLM Inference Density in Medical Reasoning via Meta-Cognitive Regulation 

**Authors**: Yu Zhao, Hao Guan, Yongcheng Jing, Ying Zhang, Dacheng Tao  

**Link**: [PDF](https://arxiv.org/pdf/2602.07905)  

**Abstract**: Large Language Models (LLMs) have shown strong potential in complex medical reasoning yet face diminishing gains under inference scaling laws. While existing studies augment LLMs with various knowledge types, it remains unclear how effectively the additional costs translate into accuracy. In this paper, we explore how meta-cognition of LLMs, i.e., their self-awareness of their own knowledge states, can regulate the reasoning process. Specifically, we propose MedCoG, a Medical Meta-Cognition Agent with Knowledge Graph, where the meta-cognitive assessments of task complexity, familiarity, and knowledge density dynamically regulate utilization of procedural, episodic, and factual knowledge. The LLM-centric on-demand reasoning aims to mitigate scaling laws by (1) reducing costs via avoiding indiscriminate scaling, (2) improving accuracy via filtering out distractive knowledge. To validate this, we empirically characterize the scaling curve and introduce inference density to quantify inference efficiency, defined as the ratio of theoretically effective cost to actual cost. Experiments demonstrate the effectiveness and efficiency of MedCoG on five hard sets of medical benchmarks, yielding 5.5x inference density. Furthermore, the Oracle study highlights the significant potential of meta-cognitive regulation. 

---
# Emergent Misalignment is Easy, Narrow Misalignment is Hard 

**Authors**: Anna Soligo, Edward Turner, Senthooran Rajamanoharan, Neel Nanda  

**Link**: [PDF](https://arxiv.org/pdf/2602.07852)  

**Abstract**: Finetuning large language models on narrowly harmful datasets can cause them to become emergently misaligned, giving stereotypically `evil' responses across diverse unrelated settings. Concerningly, a pre-registered survey of experts failed to predict this result, highlighting our poor understanding of the inductive biases governing learning and generalisation in LLMs. We use emergent misalignment (EM) as a case study to investigate these inductive biases and find that models can just learn the narrow dataset task, but that the general solution appears to be more stable and more efficient. To establish this, we build on the result that different EM finetunes converge to the same linear representation of general misalignment, which can be used to mediate misaligned behaviour. We find a linear representation of the narrow solution also exists, and can be learned by introducing a KL divergence loss. Comparing these representations reveals that general misalignment achieves lower loss, is more robust to perturbations, and is more influential in the pre-training distribution. This work isolates a concrete representation of general misalignment for monitoring and mitigation. More broadly, it offers a detailed case study and preliminary metrics for investigating how inductive biases shape generalisation in LLMs. We open-source all code, datasets and model finetunes. 

---
# Time Series Reasoning via Process-Verifiable Thinking Data Synthesis and Scheduling for Tailored LLM Reasoning 

**Authors**: Jiahui Zhou, Dan Li, Boxin Li, Xiao Zhang, Erli Meng, Lin Li, Zhuomin Chen, Jian Lou, See-Kiong Ng  

**Link**: [PDF](https://arxiv.org/pdf/2602.07830)  

**Abstract**: Time series is a pervasive data type across various application domains, rendering the reasonable solving of diverse time series tasks a long-standing goal. Recent advances in large language models (LLMs), especially their reasoning abilities unlocked through reinforcement learning (RL), have opened new opportunities for tackling tasks with long Chain-of-Thought (CoT) reasoning. However, leveraging LLM reasoning for time series remains in its infancy, hindered by the absence of carefully curated time series CoT data for training, limited data efficiency caused by underexplored data scheduling, and the lack of RL algorithms tailored for exploiting such time series CoT data. In this paper, we introduce VeriTime, a framework that tailors LLMs for time series reasoning through data synthesis, data scheduling, and RL training. First, we propose a data synthesis pipeline that constructs a TS-text multimodal dataset with process-verifiable annotations. Second, we design a data scheduling mechanism that arranges training samples according to a principled hierarchy of difficulty and task taxonomy. Third, we develop a two-stage reinforcement finetuning featuring fine-grained, multi-objective rewards that leverage verifiable process-level CoT data. Extensive experiments show that VeriTime substantially boosts LLM performance across diverse time series reasoning tasks. Notably, it enables compact 3B, 4B models to achieve reasoning capabilities on par with or exceeding those of larger proprietary LLMs. 

---
# Efficient Table Retrieval and Understanding with Multimodal Large Language Models 

**Authors**: Zhuoyan Xu, Haoyang Fang, Boran Han, Bonan Min, Bernie Wang, Cuixiong Hu, Shuai Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2602.07642)  

**Abstract**: Tabular data is frequently captured in image form across a wide range of real-world scenarios such as financial reports, handwritten records, and document scans. These visual representations pose unique challenges for machine understanding, as they combine both structural and visual complexities. While recent advances in Multimodal Large Language Models (MLLMs) show promising results in table understanding, they typically assume the relevant table is readily available. However, a more practical scenario involves identifying and reasoning over relevant tables from large-scale collections to answer user queries. To address this gap, we propose TabRAG, a framework that enables MLLMs to answer queries over large collections of table images. Our approach first retrieves candidate tables using jointly trained visual-text foundation models, then leverages MLLMs to perform fine-grained reranking of these candidates, and finally employs MLLMs to reason over the selected tables for answer generation. Through extensive experiments on a newly constructed dataset comprising 88,161 training and 9,819 testing samples across 8 benchmarks with 48,504 unique tables, we demonstrate that our framework significantly outperforms existing methods by 7.0% in retrieval recall and 6.1% in answer accuracy, offering a practical solution for real-world table understanding tasks. 

---
# EventCast: Hybrid Demand Forecasting in E-Commerce with LLM-Based Event Knowledge 

**Authors**: Congcong Hu, Yuang Shi, Fan Huang, Yang Xiang, Zhou Ye, Ming Jin, Shiyu Wang  

**Link**: [PDF](https://arxiv.org/pdf/2602.07695)  

**Abstract**: Demand forecasting is a cornerstone of e-commerce operations, directly impacting inventory planning and fulfillment scheduling. However, existing forecasting systems often fail during high-impact periods such as flash sales, holiday campaigns, and sudden policy interventions, where demand patterns shift abruptly and unpredictably. In this paper, we introduce EventCast, a modular forecasting framework that integrates future event knowledge into time-series prediction. Unlike prior approaches that ignore future interventions or directly use large language models (LLMs) for numerical forecasting, EventCast leverages LLMs solely for event-driven reasoning. Unstructured business data, which covers campaigns, holiday schedules, and seller incentives, from existing operational databases, is processed by an LLM that converts it into interpretable textual summaries leveraging world knowledge for cultural nuances and novel event combinations. These summaries are fused with historical demand features within a dual-tower architecture, enabling accurate, explainable, and scalable forecasts. Deployed on real-world e-commerce scenarios spanning 4 countries of 160 regions over 10 months, EventCast achieves up to 86.9% and 97.7% improvement on MAE and MSE compared to the variant without event knowledge, and reduces MAE by up to 57.0% and MSE by 83.3% versus the best industrial baseline during event-driven periods. EventCast has deployed into real-world industrial pipelines since March 2025, offering a practical solution for improving operational decision-making in dynamic e-commerce environments. 

---
# ToolSelf: Unifying Task Execution and Self-Reconfiguration via Tool-Driven Intrinsic Adaptation 

**Authors**: Jingqi Zhou, Sheng Wang, DeZhao Deng, Junwen Lu, Junwei Su, Qintong Li, Jiahui Gao, Hao Wu, Jiyue Jiang, Lingpeng Kong, Chuan Wu  

**Link**: [PDF](https://arxiv.org/pdf/2602.07883)  

**Abstract**: Agentic systems powered by Large Language Models (LLMs) have demonstrated remarkable potential in tackling complex, long-horizon tasks. However, their efficacy is fundamentally constrained by static configurations governing agent behaviors, which are fixed prior to execution and fail to adapt to evolving task dynamics. Existing approaches, relying on manual orchestration or heuristic-based patches, often struggle with poor generalization and fragmented optimization. To transcend these limitations, we propose ToolSelf, a novel paradigm enabling tool-driven runtime self-reconfiguration. By abstracting configuration updates as a callable tool, ToolSelf unifies task execution and self-adjustment into a single action space, achieving a phase transition from external rules to intrinsic parameters. Agents can thereby autonomously update their sub-goals and context based on task progression, and correspondingly adapt their strategy and toolbox, transforming from passive executors into dual managers of both task and self. We further devise Configuration-Aware Two-stage Training (CAT), combining rejection sampling fine-tuning with trajectory-level reinforcement learning to internalize this meta-capability. Extensive experiments across diverse benchmarks demonstrate that ToolSelf rivals specialized workflows while generalizing to novel tasks, achieving a 24.1% average performance gain and illuminating a path toward truly self-adaptive agents. 

---
# Data Darwinism Part I: Unlocking the Value of Scientific Data for Pre-training 

**Authors**: Yiwei Qin, Zhen Huang, Tiantian Mi, Weiye Si, Chenyang Zhou, Qipeng Guo, Siyuan Feng, Pengfei Liu  

**Link**: [PDF](https://arxiv.org/pdf/2602.07824)  

**Abstract**: Data quality determines foundation model performance, yet systematic processing frameworks are lacking. We introduce Data Darwinism, a ten-level taxonomy (L0-L9) that conceptualizes data-model co-evolution: advanced models produce superior data for next-generation systems. We validate this on scientific literature by constructing Darwin-Science, a 900B-token corpus (L0-L5). We identify a learnability gap in raw scientific text, which we bridge via L4 (Generative Refinement) and L5 (Cognitive Completion) using frontier LLMs to explicate reasoning and terminology.
To ensure rigorous attribution, we pre-trained daVinci-origin-3B/7B models from scratch, excluding scientific content to create contamination-free baselines. After 600B tokens of continued pre-training, Darwin-Science outperforms baselines by +2.12 (3B) and +2.95 (7B) points across 20+ benchmarks, rising to +5.60 and +8.40 points on domain-aligned tasks. Systematic progression to L5 yields a +1.36 total gain, confirming that higher-level processing unlocks latent data value. We release the Darwin-Science corpus and daVinci-origin models to enable principled, co-evolutionary development. 

---
# GraphAgents: Knowledge Graph-Guided Agentic AI for Cross-Domain Materials Design 

**Authors**: Isabella A. Stewart, Tarjei Paule Hage, Yu-Chuan Hsu, Markus J. Buehler  

**Link**: [PDF](https://arxiv.org/pdf/2602.07491)  

**Abstract**: Large Language Models (LLMs) promise to accelerate discovery by reasoning across the expanding scientific landscape. Yet, the challenge is no longer access to information but connecting it in meaningful, domain-spanning ways. In materials science, where innovation demands integrating concepts from molecular chemistry to mechanical performance, this is especially acute. Neither humans nor single-agent LLMs can fully contend with this torrent of information, with the latter often prone to hallucinations. To address this bottleneck, we introduce a multi-agent framework guided by large-scale knowledge graphs to find sustainable substitutes for per- and polyfluoroalkyl substances (PFAS)-chemicals currently under intense regulatory scrutiny. Agents in the framework specialize in problem decomposition, evidence retrieval, design parameter extraction, and graph traversal, uncovering latent connections across distinct knowledge pockets to support hypothesis generation. Ablation studies show that the full multi-agent pipeline outperforms single-shot prompting, underscoring the value of distributed specialization and relational reasoning. We demonstrate that by tailoring graph traversal strategies, the system alternates between exploitative searches focusing on domain-critical outcomes and exploratory searches surfacing emergent cross-connections. Illustrated through the exemplar of biomedical tubing, the framework generates sustainable PFAS-free alternatives that balance tribological performance, thermal stability, chemical resistance, and biocompatibility. This work establishes a framework combining knowledge graphs with multi-agent reasoning to expand the materials design space, showcasing several initial design candidates to demonstrate the approach. 

---
# Are Reasoning LLMs Robust to Interventions on Their Chain-of-Thought? 

**Authors**: Alexander von Recum, Leander Girrbach, Zeynep Akata  

**Link**: [PDF](https://arxiv.org/pdf/2602.07470)  

**Abstract**: Reasoning LLMs (RLLMs) generate step-by-step chains of thought (CoTs) before giving an answer, which improves performance on complex tasks and makes reasoning more transparent. But how robust are these reasoning traces to disruptions that occur within them? To address this question, we introduce a controlled evaluation framework that perturbs a model's own CoT at fixed timesteps. We design seven interventions (benign, neutral, and adversarial) and apply them to multiple open-weight RLLMs across Math, Science, and Logic tasks. Our results show that RLLMs are generally robust, reliably recovering from diverse perturbations, with robustness improving with model size and degrading when interventions occur early. However, robustness is not style-invariant: paraphrasing suppresses doubt-like expressions and reduces performance, while other interventions trigger doubt and support recovery. Recovery also carries a cost: neutral and adversarial noise can inflate CoT length by more than 200%, whereas paraphrasing shortens traces but harms accuracy. These findings provide new evidence on how RLLMs maintain reasoning integrity, identify doubt as a central recovery mechanism, and highlight trade-offs between robustness and efficiency that future training methods should address. 

---
# MSP-LLM: A Unified Large Language Model Framework for Complete Material Synthesis Planning 

**Authors**: Heewoong Noh, Gyoung S. Na, Namkyeong Lee, Chanyoung Park  

**Link**: [PDF](https://arxiv.org/pdf/2602.07543)  

**Abstract**: Material synthesis planning (MSP) remains a fundamental and underexplored bottleneck in AI-driven materials discovery, as it requires not only identifying suitable precursor materials but also designing coherent sequences of synthesis operations to realize a target material. Although several AI-based approaches have been proposed to address isolated subtasks of MSP, a unified methodology for solving the entire MSP task has yet to be established. We propose MSP-LLM, a unified LLM-based framework that formulates MSP as a structured process composed of two constituent subproblems: precursor prediction (PP) and synthesis operation prediction (SOP). Our approach introduces a discrete material class as an intermediate decision variable that organizes both tasks into a chemically consistent decision chain. For OP, we further incorporate hierarchical precursor types as synthesis-relevant inductive biases and employ an explicit conditioning strategy that preserves precursor-related information in the autoregressive decoding state. Extensive experiments show that MSP-LLM consistently outperforms existing methods on both PP and SOP, as well as on the complete MSP task, demonstrating an effective and scalable framework for MSP that can accelerate real-world materials discovery. 

---
# SupChain-Bench: Benchmarking Large Language Models for Real-World Supply Chain Management 

**Authors**: Shengyue Guan, Yihao Liu, Lang Cao  

**Link**: [PDF](https://arxiv.org/pdf/2602.07342)  

**Abstract**: Large language models (LLMs) have shown promise in complex reasoning and tool-based decision making, motivating their application to real-world supply chain management. However, supply chain workflows require reliable long-horizon, multi-step orchestration grounded in domain-specific procedures, which remains challenging for current models. To systematically evaluate LLM performance in this setting, we introduce SupChain-Bench, a unified real-world benchmark that assesses both supply chain domain knowledge and long-horizon tool-based orchestration grounded in standard operating procedures (SOPs). Our experiments reveal substantial gaps in execution reliability across models. We further propose SupChain-ReAct, an SOP-free framework that autonomously synthesizes executable procedures for tool use, achieving the strongest and most consistent tool-calling performance. Our work establishes a principled benchmark for studying reliable long-horizon orchestration in real-world operational settings and highlights significant room for improvement in LLM-based supply chain agents. 

---
# Can LLMs Truly Embody Human Personality? Analyzing AI and Human Behavior Alignment in Dispute Resolution 

**Authors**: Deuksin Kwon, Kaleen Shrestha, Bin Han, Spencer Lin, James Hale, Jonathan Gratch, Maja Matarić, Gale M. Lucas  

**Link**: [PDF](https://arxiv.org/pdf/2602.07414)  

**Abstract**: Large language models (LLMs) are increasingly used to simulate human behavior in social settings such as legal mediation, negotiation, and dispute resolution. However, it remains unclear whether these simulations reproduce the personality-behavior patterns observed in humans. Human personality, for instance, shapes how individuals navigate social interactions, including strategic choices and behaviors in emotionally charged interactions. This raises the question: Can LLMs, when prompted with personality traits, reproduce personality-driven differences in human conflict behavior? To explore this, we introduce an evaluation framework that enables direct comparison of human-human and LLM-LLM behaviors in dispute resolution dialogues with respect to Big Five Inventory (BFI) personality traits. This framework provides a set of interpretable metrics related to strategic behavior and conflict outcomes. We additionally contribute a novel dataset creation methodology for LLM dispute resolution dialogues with matched scenarios and personality traits with respect to human conversations. Finally, we demonstrate the use of our evaluation framework with three contemporary closed-source LLMs and show significant divergences in how personality manifests in conflict across different LLMs compared to human data, challenging the assumption that personality-prompted agents can serve as reliable behavioral proxies in socially impactful applications. Our work highlights the need for psychological grounding and validation in AI simulations before real-world use. 

---
# Steer2Adapt: Dynamically Composing Steering Vectors Elicits Efficient Adaptation of LLMs 

**Authors**: Pengrui Han, Xueqiang Xu, Keyang Xuan, Peiyang Song, Siru Ouyang, Runchu Tian, Yuqing Jiang, Cheng Qian, Pengcheng Jiang, Jiashuo Sun, Junxia Cui, Ming Zhong, Ge Liu, Jiawei Han, Jiaxuan You  

**Link**: [PDF](https://arxiv.org/pdf/2602.07276)  

**Abstract**: Activation steering has emerged as a promising approach for efficiently adapting large language models (LLMs) to downstream behaviors. However, most existing steering methods rely on a single static direction per task or concept, making them inflexible under task variation and inadequate for complex tasks that require multiple coordinated capabilities. To address this limitation, we propose STEER2ADAPT, a lightweight framework that adapts LLMs by composing steering vectors rather than learning new ones from scratch. In many domains (e.g., reasoning or safety), tasks share a small set of underlying concept dimensions. STEER2ADAPT captures these dimensions as a reusable, low-dimensional semantic prior subspace, and adapts to new tasks by dynamically discovering a linear combination of basis vectors from only a handful of examples. Experiments across 9 tasks and 3 models in both reasoning and safety domains demonstrate the effectiveness of STEER2ADAPT, achieving an average improvement of 8.2%. Extensive analyses further show that STEER2ADAPT is a data-efficient, stable, and transparent inference-time adaptation method for LLMs. 

---
# From Out-of-Distribution Detection to Hallucination Detection: A Geometric View 

**Authors**: Litian Liu, Reza Pourreza, Yubing Jian, Yao Qin, Roland Memisevic  

**Link**: [PDF](https://arxiv.org/pdf/2602.07253)  

**Abstract**: Detecting hallucinations in large language models is a critical open problem with significant implications for safety and reliability. While existing hallucination detection methods achieve strong performance in question-answering tasks, they remain less effective on tasks requiring reasoning. In this work, we revisit hallucination detection through the lens of out-of-distribution (OOD) detection, a well-studied problem in areas like computer vision. Treating next-token prediction in language models as a classification task allows us to apply OOD techniques, provided appropriate modifications are made to account for the structural differences in large language models. We show that OOD-based approaches yield training-free, single-sample-based detectors, achieving strong accuracy in hallucination detection for reasoning tasks. Overall, our work suggests that reframing hallucination detection as OOD detection provides a promising and scalable pathway toward language model safety. 

---
# LLM-FSM: Scaling Large Language Models for Finite-State Reasoning in RTL Code Generation 

**Authors**: Yuheng Wu, Berk Gokmen, Zhouhua Xie, Peijing Li, Caroline Trippel, Priyanka Raina, Thierry Tambe  

**Link**: [PDF](https://arxiv.org/pdf/2602.07032)  

**Abstract**: Finite-state reasoning, the ability to understand and implement state-dependent behavior, is central to hardware design. In this paper, we present LLM-FSM, a benchmark that evaluates how well large language models (LLMs) can recover finite-state machine (FSM) behavior from natural-language specifications and translate it into correct register transfer-level (RTL) implementations. Unlike prior specification-to-RTL benchmarks that rely on manually constructed examples, LLM-FSM is built through a fully automated pipeline. LLM-FSM first constructs FSM with configurable state counts and constrained transition structures. It then prompts LLMs to express each FSM in a structured YAML format with an application context, and to further convert that YAML into a natural-language (NL) specification. From the same YAML, our pipeline synthesizes the reference RTL and testbench in a correct-by-construction manner. All 1,000 problems are verified using LLM-based and SAT-solver-based checks, with human review on a subset. Our experiments show that even the strongest LLMs exhibit sharply declining accuracy as FSM complexity increases. We further demonstrate that training-time scaling via supervised fine-tuning (SFT) generalizes effectively to out-of-distribution (OOD) tasks, while increasing test-time compute improves reasoning reliability. Finally, LLM-FSM remains extensible by allowing its FSM complexity to scale with future model capabilities. 

---
# PreFlect: From Retrospective to Prospective Reflection in Large Language Model Agents 

**Authors**: Hanyu Wang, Yuanpu Cao, Lu Lin, Jinghui Chen  

**Link**: [PDF](https://arxiv.org/pdf/2602.07187)  

**Abstract**: Advanced large language model agents typically adopt self-reflection for improving performance, where agents iteratively analyze past actions to correct errors. However, existing reflective approaches are inherently retrospective: agents act, observe failure, and only then attempt to recover. In this work, we introduce PreFlect, a prospective reflection mechanism that shifts the paradigm from post hoc correction to pre-execution foresight by criticizing and refining agent plans before execution. To support grounded prospective reflection, we distill planning errors from historical agent trajectories, capturing recurring success and failure patterns observed across past executions. Furthermore, we complement prospective reflection with a dynamic re-planning mechanism that provides execution-time plan update in case the original plan encounters unexpected deviation. Evaluations on different benchmarks demonstrate that PreFlect significantly improves overall agent utility on complex real-world tasks, outperforming strong reflection-based baselines and several more complex agent architectures. Code will be updated at this https URL. 

---
# DLLM-Searcher: Adapting Diffusion Large Language Model for Search Agents 

**Authors**: Jiahao Zhao, Shaoxuan Xu, Zhongxiang Sun, Fengqi Zhu, Jingyang Ou, Yuling Shi, Chongxuan Li, Xiao Zhang, Jun Xu  

**Link**: [PDF](https://arxiv.org/pdf/2602.07035)  

**Abstract**: Recently, Diffusion Large Language Models (dLLMs) have demonstrated unique efficiency advantages, enabled by their inherently parallel decoding mechanism and flexible generation paradigm. Meanwhile, despite the rapid advancement of Search Agents, their practical deployment is constrained by a fundamental limitation, termed as 1) Latency Challenge: the serial execution of multi-round reasoning, tool calling, and tool response waiting under the ReAct agent paradigm induces severe end-to-end latency. Intuitively, dLLMs can leverage their distinctive strengths to optimize the operational efficiency of agents under the ReAct agent paradigm. Practically, existing dLLM backbones face the 2) Agent Ability Challenge. That is, existing dLLMs exhibit remarkably weak reasoning and tool-calling capabilities, preventing these advantages from being effectively realized in practice. In this paper, we propose DLLM-Searcher, an optimization framework for dLLM-based Search Agents. To solve the Agent Ability Challenge, we design a two-stage post-training pipeline encompassing Agentic Supervised Fine-Tuning (Agentic SFT) and Agentic Variance-Reduced Preference Optimization Agentic VRPO, which enhances the backbone dLLM's information seeking and reasoning capabilities. To mitigate the Latency Challenge, we leverage the flexible generation mechanism of dLLMs and propose a novel agent paradigm termed Parallel-Reasoning and Acting P-ReAct. P-ReAct guides the model to prioritize decoding tool_call instructions, thereby allowing the model to keep thinking while waiting for the tool's return. Experimental results demonstrate that DLLM-Searcher achieves performance comparable to mainstream LLM-based search agents and P-ReAct delivers approximately 15% inference acceleration. Our code is available at this https URL 

---
# A Behavioural and Representational Evaluation of Goal-Directedness in Language Model Agents 

**Authors**: Raghu Arghal, Fade Chen, Niall Dalton, Evgenii Kortukov, Calum McNamara, Angelos Nalmpantis, Moksh Nirvaan, Gabriele Sarti, Mario Giulianelli  

**Link**: [PDF](https://arxiv.org/pdf/2602.08964)  

**Abstract**: Understanding an agent's goals helps explain and predict its behaviour, yet there is no established methodology for reliably attributing goals to agentic systems. We propose a framework for evaluating goal-directedness that integrates behavioural evaluation with interpretability-based analyses of models' internal representations. As a case study, we examine an LLM agent navigating a 2D grid world toward a goal state. Behaviourally, we evaluate the agent against an optimal policy across varying grid sizes, obstacle densities, and goal structures, finding that performance scales with task difficulty while remaining robust to difficulty-preserving transformations and complex goal structures. We then use probing methods to decode the agent's internal representations of the environment state and its multi-step action plans. We find that the LLM agent non-linearly encodes a coarse spatial map of the environment, preserving approximate task-relevant cues about its position and the goal location; that its actions are broadly consistent with these internal representations; and that reasoning reorganises them, shifting from broader environment structural cues toward information supporting immediate action selection. Our findings support the view that introspective examination is required beyond behavioural evaluations to characterise how agents represent and pursue their objectives. 

---
# DeepQuali: Initial results of a study on the use of large language models for assessing the quality of user stories 

**Authors**: Adam Trendowicz, Daniel Seifert, Andreas Jedlitschka, Marcus Ciolkowski, Anton Strahilov  

**Link**: [PDF](https://arxiv.org/pdf/2602.08887)  

**Abstract**: Generative artificial intelligence (GAI), specifically large language models (LLMs), are increasingly used in software engineering, mainly for coding tasks. However, requirements engineering - particularly requirements validation - has seen limited application of GAI. The current focus of using GAI for requirements is on eliciting, transforming, and classifying requirements, not on quality assessment. We propose and evaluate the LLM-based (GPT-4o) approach "DeepQuali", for assessing and improving requirements quality in agile software development. We applied it to projects in two small companies, where we compared LLM-based quality assessments with expert judgments. Experts also participated in walkthroughs of the solution, provided feedback, and rated their acceptance of the approach. Experts largely agreed with the LLM's quality assessments, especially regarding overall ratings and explanations. However, they did not always agree with the other experts on detailed ratings, suggesting that expertise and experience may influence judgments. Experts recognized the usefulness of the approach but criticized the lack of integration into their workflow. LLMs show potential in supporting software engineers with the quality assessment and improvement of requirements. The explicit use of quality models and explanatory feedback increases acceptance. 

---
# Whose Name Comes Up? Benchmarking and Intervention-Based Auditing of LLM-Based Scholar Recommendation 

**Authors**: Lisette Espin-Noboa, Gonzalo Gabriel Mendez  

**Link**: [PDF](https://arxiv.org/pdf/2602.08873)  

**Abstract**: Large language models (LLMs) are increasingly used for academic expert recommendation. Existing audits typically evaluate model outputs in isolation, largely ignoring end-user inference-time interventions. As a result, it remains unclear whether failures such as refusals, hallucinations, and uneven coverage stem from model choice or deployment decisions. We introduce LLMScholarBench, a benchmark for auditing LLM-based scholar recommendation that jointly evaluates model infrastructure and end-user interventions across multiple tasks. LLMScholarBench measures both technical quality and social representation using nine metrics. We instantiate the benchmark in physics expert recommendation and audit 22 LLMs under temperature variation, representation-constrained prompting, and retrieval-augmented generation (RAG) via web search. Our results show that end-user interventions do not yield uniform improvements but instead redistribute error across dimensions. Higher temperature degrades validity, consistency, and factuality. Representation-constrained prompting improves diversity at the expense of factuality, while RAG primarily improves technical quality while reducing diversity and parity. Overall, end-user interventions reshape trade-offs rather than providing a general fix. We release code and data that can be adapted to other disciplines by replacing domain-specific ground truth and metrics. 

---
# Automatic In-Domain Exemplar Construction and LLM-Based Refinement of Multi-LLM Expansions for Query Expansion 

**Authors**: Minghan Li, Ercong Nie, Siqi Zhao, Tongna Chen, Huiping Huang, Guodong Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2602.08917)  

**Abstract**: Query expansion with large language models is promising but often relies on hand-crafted prompts, manually chosen exemplars, or a single LLM, making it non-scalable and sensitive to domain shift. We present an automated, domain-adaptive QE framework that builds in-domain exemplar pools by harvesting pseudo-relevant passages using a BM25-MonoT5 pipeline. A training-free cluster-based strategy selects diverse demonstrations, yielding strong and stable in-context QE without supervision. To further exploit model complementarity, we introduce a two-LLM ensemble in which two heterogeneous LLMs independently generate expansions and a refinement LLM consolidates them into one coherent expansion. Across TREC DL20, DBPedia, and SciFact, the refined ensemble delivers consistent and statistically significant gains over BM25, Rocchio, zero-shot, and fixed few-shot baselines. The framework offers a reproducible testbed for exemplar selection and multi-LLM generation, and a practical, label-free solution for real-world QE. 

---
# OmniReview: A Large-scale Benchmark and LLM-enhanced Framework for Realistic Reviewer Recommendation 

**Authors**: Yehua Huang, Penglei Sun, Zebin Chen, Zhenheng Tang, Xiaowen Chu  

**Link**: [PDF](https://arxiv.org/pdf/2602.08896)  

**Abstract**: Academic peer review remains the cornerstone of scholarly validation, yet the field faces some challenges in data and methods. From the data perspective, existing research is hindered by the scarcity of large-scale, verified benchmarks and oversimplified evaluation metrics that fail to reflect real-world editorial workflows. To bridge this gap, we present OmniReview, a comprehensive dataset constructed by integrating multi-source academic platforms encompassing comprehensive scholarly profiles through the disambiguation pipeline, yielding 202, 756 verified review records. Based on this data, we introduce a three-tier hierarchical evaluaion framework to assess recommendations from recall to precise expert identification. From the method perspective, existing embedding-based approaches suffer from the information bottleneck of semantic compression and limited interpretability. To resolve these method limitations, we propose Profiling Scholars with Multi-gate Mixture-of-Experts (Pro-MMoE), a novel framework that synergizes Large Language Models (LLMs) with Multi-task Learning. Specifically, it utilizes LLM-generated semantic profiles to preserve fine-grained expertise nuances and interpretability, while employing a Task-Adaptive MMoE architecture to dynamically balance conflicting evaluation goals. Comprehensive experiments demonstrate that Pro-MMoE achieves state-of-the-art performance across six of seven metrics, establishing a new benchmark for realistic reviewer recommendation. 

---
# AnomSeer: Reinforcing Multimodal LLMs to Reason for Time-Series Anomaly Detection 

**Authors**: Junru Zhang, Lang Feng, Haoran Shi, Xu Guo, Han Yu, Yabo Dong, Duanqing Xu  

**Link**: [PDF](https://arxiv.org/pdf/2602.08868)  

**Abstract**: Time-series anomaly detection (TSAD) with multimodal large language models (MLLMs) is an emerging area, yet a persistent challenge remains: MLLMs rely on coarse time-series heuristics but struggle with multi-dimensional, detailed reasoning, which is vital for understanding complex time-series data. We present AnomSeer to address this by reinforcing the model to ground its reasoning in precise, structural details of time series, unifying anomaly classification, localization, and explanation. At its core, an expert chain-of-thought trace is generated to provide a verifiable, fine-grained reasoning from classical analyses (e.g., statistical measures, frequency transforms). Building on this, we propose a novel time-series grounded policy optimization (TimerPO) that incorporates two additional components beyond standard reinforcement learning: a time-series grounded advantage based on optimal transport and an orthogonal projection to ensure this auxiliary granular signal does not interfere with the primary detection objective. Across diverse anomaly scenarios, AnomSeer, with Qwen2.5-VL-3B/7B-Instruct, outperforms larger commercial baselines (e.g., GPT-4o) in classification and localization accuracy, particularly on point- and frequency-driven exceptions. Moreover, it produces plausible time-series reasoning traces that support its conclusions. 

---
# WildReward: Learning Reward Models from In-the-Wild Human Interactions 

**Authors**: Hao Peng, Yunjia Qi, Xiaozhi Wang, Zijun Yao, Lei Hou, Juanzi Li  

**Link**: [PDF](https://arxiv.org/pdf/2602.08829)  

**Abstract**: Reward models (RMs) are crucial for the training of large language models (LLMs), yet they typically rely on large-scale human-annotated preference pairs. With the widespread deployment of LLMs, in-the-wild interactions have emerged as a rich source of implicit reward signals. This raises the question: Can we develop reward models directly from in-the-wild interactions? In this work, we explore this possibility by adopting WildChat as an interaction source and proposing a pipeline to extract reliable human feedback, yielding 186k high-quality instances for training WildReward via ordinal regression directly on user feedback without preference pairs. Extensive experiments demonstrate that WildReward achieves comparable or even superior performance compared to conventional reward models, with improved calibration and cross-sample consistency. We also observe that WildReward benefits directly from user diversity, where more users yield stronger reward models. Finally, we apply WildReward to online DPO training and observe significant improvements across various tasks. Code and data are released at this https URL. 

---
# Affective Flow Language Model for Emotional Support Conversation 

**Authors**: Chenghui Zou, Ning Wang, Tiesunlong Shen, Luwei Xiao, Chuan Ma, Xiangpeng Li, Rui Mao, Erik Cambria  

**Link**: [PDF](https://arxiv.org/pdf/2602.08826)  

**Abstract**: Large language models (LLMs) have been widely applied to emotional support conversation (ESC). However, complex multi-turn support remains this http URL is because existing alignment schemes rely on sparse outcome-level signals, thus offering limited supervision for intermediate strategy decisions. To fill this gap, this paper proposes affective flow language model for emotional support conversation (AFlow), a framework that introduces fine-grained supervision on dialogue prefixes by modeling a continuous affective flow along multi-turn trajectories. AFlow can estimate intermediate utility over searched trajectories and learn preference-consistent strategy transitions. To improve strategy coherence and empathetic response quality, a subpath-level flow-balance objective is presented to propagate preference signals to intermediate states. Experiment results show consistent and significant improvements over competitive baselines in diverse emotional contexts. Remarkably, AFlow with a compact open-source backbone outperforms proprietary LMMs such as GPT-4o and Claude-3.5 on major ESC metrics. Our code is available at this https URL. 

---
# Zero-shot System for Automatic Body Region Detection for Volumetric CT and MR Images 

**Authors**: Farnaz Khun Jush, Grit Werner, Mark Klemens, Matthias Lenga  

**Link**: [PDF](https://arxiv.org/pdf/2602.08717)  

**Abstract**: Reliable identification of anatomical body regions is a prerequisite for many automated medical imaging workflows, yet existing solutions remain heavily dependent on unreliable DICOM metadata. Current solutions mainly use supervised learning, which limits their applicability in many real-world scenarios. In this work, we investigate whether body region detection in volumetric CT and MR images can be achieved in a fully zero-shot manner by using knowledge embedded in large pre-trained foundation models. We propose and systematically evaluate three training-free pipelines: (1) a segmentation-driven rule-based system leveraging pre-trained multi-organ segmentation models, (2) a Multimodal Large Language Model (MLLM) guided by radiologist-defined rules, and (3) a segmentation-aware MLLM that combines visual input with explicit anatomical evidence. All methods are evaluated on 887 heterogeneous CT and MR scans with manually verified anatomical region labels. The segmentation-driven rule-based approach achieves the strongest and most consistent performance, with weighted F1-scores of 0.947 (CT) and 0.914 (MR), demonstrating robustness across modalities and atypical scan coverage. The MLLM performs competitively in visually distinctive regions, while the segmentation-aware MLLM reveals fundamental limitations. 

---
# Stateless Yet Not Forgetful: Implicit Memory as a Hidden Channel in LLMs 

**Authors**: Ahmed Salem, Andrew Paverd, Sahar Abdelnabi  

**Link**: [PDF](https://arxiv.org/pdf/2602.08563)  

**Abstract**: Large language models (LLMs) are commonly treated as stateless: once an interaction ends, no information is assumed to persist unless it is explicitly stored and re-supplied. We challenge this assumption by introducing implicit memory-the ability of a model to carry state across otherwise independent interactions by encoding information in its own outputs and later recovering it when those outputs are reintroduced as input. This mechanism does not require any explicit memory module, yet it creates a persistent information channel across inference requests. As a concrete demonstration, we introduce a new class of temporal backdoors, which we call time bombs. Unlike conventional backdoors that activate on a single trigger input, time bombs activate only after a sequence of interactions satisfies hidden conditions accumulated via implicit memory. We show that such behavior can be induced today through straightforward prompting or fine-tuning. Beyond this case study, we analyze broader implications of implicit memory, including covert inter-agent communication, benchmark contamination, targeted manipulation, and training-data poisoning. Finally, we discuss detection challenges and outline directions for stress-testing and evaluation, with the goal of anticipating and controlling future developments. To promote future research, we release code and data at: this https URL. 

---
# CLEAR: A Knowledge-Centric Vessel Trajectory Analysis Platform 

**Authors**: Hengyu Liu, Tianyi Li, Haoyu Wang, Kristian Torp, Yushuai Li, Tiancheng Zhang, Torben Bach Pedersen, Christian S. Jensen  

**Link**: [PDF](https://arxiv.org/pdf/2602.08482)  

**Abstract**: Vessel trajectory data from the Automatic Identification System (AIS) is used widely in maritime analytics. Yet, analysis is difficult for non-expert users due to the incompleteness and complexity of AIS data. We present CLEAR, a knowledge-centric vessel trajectory analysis platform that aims to overcome these barriers. By leveraging the reasoning and generative capabilities of Large Language Models (LLMs), CLEAR transforms raw AIS data into complete, interpretable, and easily explorable vessel trajectories through a Structured Data-derived Knowledge Graph (SD-KG). As part of the demo, participants can configure parameters to automatically download and process AIS data, observe how trajectories are completed and annotated, inspect both raw and imputed segments together with their SD-KG evidence, and interactively explore the SD-KG through a dedicated graph viewer, gaining an intuitive and transparent understanding of vessel movements. 

---
# GISA: A Benchmark for General Information-Seeking Assistant 

**Authors**: Yutao Zhu, Xingshuo Zhang, Maosen Zhang, Jiajie Jin, Liancheng Zhang, Xiaoshuai Song, Kangzhi Zhao, Wencong Zeng, Ruiming Tang, Han Li, Ji-Rong Wen, Zhicheng Dou  

**Link**: [PDF](https://arxiv.org/pdf/2602.08543)  

**Abstract**: The advancement of large language models (LLMs) has significantly accelerated the development of search agents capable of autonomously gathering information through multi-turn web interactions. Various benchmarks have been proposed to evaluate such agents. However, existing benchmarks often construct queries backward from answers, producing unnatural tasks misaligned with real-world needs. Moreover, these benchmarks tend to focus on either locating specific information or aggregating information from multiple sources, while relying on static answer sets prone to data contamination. To bridge these gaps, we introduce GISA, a benchmark for General Information-Seeking Assistants comprising 373 human-crafted queries that reflect authentic information-seeking scenarios. GISA features four structured answer formats (item, set, list, and table), enabling deterministic evaluation. It integrates both deep reasoning and broad information aggregation within unified tasks, and includes a live subset with periodically updated answers to resist memorization. Notably, GISA provides complete human search trajectories for every query, offering gold-standard references for process-level supervision and imitation learning. Experiments on mainstream LLMs and commercial search products reveal that even the best-performing model achieves only 19.30\% exact match score, with performance notably degrading on tasks requiring complex planning and comprehensive information gathering. These findings highlight substantial room for future improvement. 

---
# Dynamic Long Context Reasoning over Compressed Memory via End-to-End Reinforcement Learning 

**Authors**: Zhuoen Chen, Dongfang Li, Meishan Zhang, Baotian Hu, Min Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2602.08382)  

**Abstract**: Large Language Models (LLMs) face significant challenges in long-context processing, including quadratic computational costs, information forgetting, and the context fragmentation inherent in retrieval-augmented generation (RAG). We propose a cognitively inspired framework for efficient long-context inference based on chunk-wise compression and selective memory recall, rather than processing all raw tokens. The framework segments long inputs into chunks and encodes each chunk into compressed memory representations using a learned compressor. A gating module dynamically selects relevant memory blocks, which are then iteratively processed by a reasoning module with an evolving working memory to solve downstream tasks. The compressor and reasoner are jointly optimized via end-to-end reinforcement learning, while the gating module is trained separately as a classifier. Experimental results show that the proposed method achieves competitive accuracy on multi-hop reasoning benchmarks such as RULER-HQA, extrapolates context length from 7K to 1.75M tokens, and offers a favorable accuracy-efficiency trade-off compared to strong long-context baselines. In particular, it achieves up to a 2 times reduction in peak GPU memory usage and a 6 times inference speedup over MemAgent. 

---
# The Chicken and Egg Dilemma: Co-optimizing Data and Model Configurations for LLMs 

**Authors**: Zhiliang Chen, Alfred Wei Lun Leong, Shao Yong Ong, Apivich Hemachandram, Gregory Kang Ruey Lau, Chuan-Sheng Foo, Zhengyuan Liu, Nancy F. Chen, Bryan Kian Hsiang Low  

**Link**: [PDF](https://arxiv.org/pdf/2602.08351)  

**Abstract**: Co-optimizing data and model configurations for training LLMs presents a classic chicken-and-egg dilemma: The best training data configuration (e.g., data mixture) for a downstream task depends on the chosen model configuration (e.g., model architecture), and vice versa. However, jointly optimizing both data and model configurations is often deemed intractable, and existing methods focus on either data or model optimization without considering their interaction. We introduce JoBS, an approach that uses a scaling-law-inspired performance predictor to aid Bayesian optimization (BO) in jointly optimizing LLM training data and model configurations efficiently. JoBS allocates a portion of the optimization budget to learn an LLM performance predictor that predicts how promising a training configuration is from a small number of training steps. The remaining budget is used to perform BO entirely with the predictor, effectively amortizing the cost of running full-training runs. We study JoBS's average regret and devise the optimal budget allocation to minimize regret. JoBS outperforms existing multi-fidelity BO baselines, as well as data and model optimization approaches across diverse LLM tasks under the same optimization budget. 

---
# Near-Oracle KV Selection via Pre-hoc Sparsity for Long-Context Inference 

**Authors**: Yifei Gao, Lei Wang, Rong-Cheng Tu, Qixin Zhang, Jun Cheng, Dacheng Tao  

**Link**: [PDF](https://arxiv.org/pdf/2602.08329)  

**Abstract**: A core bottleneck in large language model (LLM) inference is the cost of attending over the ever-growing key-value (KV) cache. Although near-oracle top-k KV selection can preserve the quality of dense attention while sharply reducing computation and bandwidth, existing sparse methods generally rely on posterior heuristics, i.e., selectors conditioned on observed attention or proxy scores. Such conditioning introduces posterior bias: it tends to distort true token importance and miss salient tokens, thereby impairing long-range reasoning. To tackle this problem, we propose Pre-hoc Sparsity (PrHS), which selects KV entries before attention scoring and provides explicit accuracy control. Let the attention mass of discarded entries be delta (the dropped mass). Through a marginal-to-mutual-information analysis, we derive an upper bound on the mutual-information loss that depends only on the dropped mass. This relation explains failure modes of posterior heuristics and enables verifiable guarantees by controlling the dropped mass in advance. Within PrHS, we instantiate three orthogonal pre-hoc selectors along the axes of time, depth, and layer. Extensive experiments on LLaMA and Mistral families validate PrHS. Across GSM8K and CoQA, PrHS reduces retrieval overhead by over 90%, achieving 3x higher retrieval sparsity than HShare at matched or better accuracy. It incurs under 1% average degradation on LongBench, lowers attention FLOPs by about 15% versus prior sparse baselines, and yields a 9.9x speedup in attention-operator latency and 2.8x higher throughput on NVIDIA A100-80GB GPUs than the dense baseline. 

---
# Latent Reasoning with Supervised Thinking States 

**Authors**: Ido Amos, Avi Caciularu, Mor Geva, Amir Globerson, Jonathan Herzig, Lior Shani, Idan Szpektor  

**Link**: [PDF](https://arxiv.org/pdf/2602.08332)  

**Abstract**: Reasoning with a chain-of-thought (CoT) enables Large Language Models (LLMs) to solve complex tasks but incurs significant inference costs due to the generation of long rationales. We propose Thinking States, a method that performs reasoning {\em while} the input is processing. Specifically, Thinking States generates sequences of thinking tokens every few input tokens, transforms the thoughts back into embedding space, and adds them to the following input tokens. This has two key advantages. First, it captures the recurrent nature of CoT, but where the thought tokens are generated as input is processing. Second, since the thoughts are represented as tokens, they can be learned from natural language supervision, and using teacher-forcing, which is parallelizable. Empirically, Thinking States outperforms other latent reasoning methods on multiple reasoning tasks, narrowing the gap to CoT on math problems, and matching its performance on 2-Hop QA with improved latency. On state-tracking tasks, we show Thinking States leads to stronger reasoning behavior than CoT, successfully extrapolating to longer sequences than seen during training. 

---
# When Do Multi-Agent Systems Outperform? Analysing the Learning Efficiency of Agentic Systems 

**Authors**: Junwei Su, Chuan Wu  

**Link**: [PDF](https://arxiv.org/pdf/2602.08272)  

**Abstract**: Reinforcement Learning (RL) has emerged as a crucial method for training or fine-tuning large language models (LLMs), enabling adaptive, task-specific optimizations through interactive feedback. Multi-Agent Reinforcement Learning (MARL), in particular, offers a promising avenue by decomposing complex tasks into specialized subtasks learned by distinct interacting agents, potentially enhancing the ability and efficiency of LLM systems. However, theoretical insights regarding when and why MARL outperforms Single-Agent RL (SARL) remain limited, creating uncertainty in selecting the appropriate RL framework. In this paper, we address this critical gap by rigorously analyzing the comparative sample efficiency of MARL and SARL within the context of LLM. Leveraging the Probably Approximately Correct (PAC) framework, we formally define SARL and MARL setups for LLMs, derive explicit sample complexity bounds, and systematically characterize how task decomposition and alignment influence learning efficiency. Our results demonstrate that MARL improves sample complexity when tasks naturally decompose into independent subtasks, whereas dependent subtasks diminish MARL's comparative advantage. Additionally, we introduce and analyze the concept of task alignment, quantifying the trade-offs when enforcing independent task decomposition despite potential misalignments. These theoretical insights clarify empirical inconsistencies and provide practical criteria for deploying MARL strategies effectively in complex LLM scenarios. 

---
# DrugR: Optimizing Molecular Drugs through LLM-based Explicit Reasoning 

**Authors**: Haoran Liu, Zheni Zeng, Yukun Yan, Yuxuan Chen, Yunduo Xiao  

**Link**: [PDF](https://arxiv.org/pdf/2602.08213)  

**Abstract**: Molecule generation and optimization is a fundamental task in chemical domain. The rapid development of intelligent tools, especially large language models (LLMs) with powerful knowledge reserves and interactive capabilities, has provided new paradigms for it. Nevertheless, the intrinsic challenge for LLMs lies in the complex implicit relationship between molecular structure and pharmacological properties and the lack of corresponding labeled data. To bridge this gap, we propose DrugR, an LLM-based method that introduces explicit, step-by-step pharmacological reasoning into the optimization process. Our approach integrates domain-specific continual pretraining, supervised fine-tuning via reverse data engineering, and self-balanced multi-granular reinforcement learning. This framework enables DrugR to effectively improve key ADMET properties while preserving the original molecule's core efficacy. Experimental results demonstrate that DrugR achieves comprehensive enhancement across multiple properties without compromising structural similarity or target binding affinity. Importantly, its explicit reasoning process provides clear, interpretable rationales for each optimization step, yielding actionable design insights and advancing toward automated, knowledge-driven scientific discovery. Our code and model checkpoints are open-sourced to foster future research. 

---
# Nexus: Inferring Join Graphs from Metadata Alone via Iterative Low-Rank Matrix Completion 

**Authors**: Tianji Cong, Yuanyuan Tian, Andreas Mueller, Rathijit Sen, Yeye He, Fotis Psallidas, Shaleen Deep, H. V. Jagadish  

**Link**: [PDF](https://arxiv.org/pdf/2602.08186)  

**Abstract**: Automatically inferring join relationships is a critical task for effective data discovery, integration, querying and reuse. However, accurately and efficiently identifying these relationships in large and complex schemas can be challenging, especially in enterprise settings where access to data values is constrained. In this paper, we introduce the problem of join graph inference when only metadata is available. We conduct an empirical study on a large number of real-world schemas and observe that join graphs when represented as adjacency matrices exhibit two key properties: high sparsity and low-rank structure. Based on these novel observations, we formulate join graph inference as a low-rank matrix completion problem and propose Nexus, an end-to-end solution using only metadata. To further enhance accuracy, we propose a novel Expectation-Maximization algorithm that alternates between low-rank matrix completion and refining join candidate probabilities by leveraging Large Language Models. Our extensive experiments demonstrate that Nexus outperforms existing methods by a significant margin on four datasets including a real-world production dataset. Additionally, Nexus can operate in a fast mode, providing comparable results with up to 6x speedup, offering a practical and efficient solution for real-world deployments. 

---
# Gender and Race Bias in Consumer Product Recommendations by Large Language Models 

**Authors**: Ke Xu, Shera Potka, Alex Thomo  

**Link**: [PDF](https://arxiv.org/pdf/2602.08124)  

**Abstract**: Large Language Models are increasingly employed in generating consumer product recommendations, yet their potential for embedding and amplifying gender and race biases remains underexplored. This paper serves as one of the first attempts to examine these biases within LLM-generated recommendations. We leverage prompt engineering to elicit product suggestions from LLMs for various race and gender groups and employ three analytical methods-Marked Words, Support Vector Machines, and Jensen-Shannon Divergence-to identify and quantify biases. Our findings reveal significant disparities in the recommendations for demographic groups, underscoring the need for more equitable LLM recommendation systems. 

---
# Reliable and Responsible Foundation Models: A Comprehensive Survey 

**Authors**: Xinyu Yang, Junlin Han, Rishi Bommasani, Jinqi Luo, Wenjie Qu, Wangchunshu Zhou, Adel Bibi, Xiyao Wang, Jaehong Yoon, Elias Stengel-Eskin, Shengbang Tong, Lingfeng Shen, Rafael Rafailov, Runjia Li, Zhaoyang Wang, Yiyang Zhou, Chenhang Cui, Yu Wang, Wenhao Zheng, Huichi Zhou, Jindong Gu, Zhaorun Chen, Peng Xia, Tony Lee, Thomas Zollo, Vikash Sehwag, Jixuan Leng, Jiuhai Chen, Yuxin Wen, Huan Zhang, Zhun Deng, Linjun Zhang, Pavel Izmailov, Pang Wei Koh, Yulia Tsvetkov, Andrew Wilson, Jiaheng Zhang, James Zou, Cihang Xie, Hao Wang, Philip Torr, Julian McAuley, David Alvarez-Melis, Florian Tramèr, Kaidi Xu, Suman Jana, Chris Callison-Burch, Rene Vidal, Filippos Kokkinos, Mohit Bansal, Beidi Chen, Huaxiu Yao  

**Link**: [PDF](https://arxiv.org/pdf/2602.08145)  

**Abstract**: Foundation models, including Large Language Models (LLMs), Multimodal Large Language Models (MLLMs), Image Generative Models (i.e, Text-to-Image Models and Image-Editing Models), and Video Generative Models, have become essential tools with broad applications across various domains such as law, medicine, education, finance, science, and beyond. As these models see increasing real-world deployment, ensuring their reliability and responsibility has become critical for academia, industry, and government. This survey addresses the reliable and responsible development of foundation models. We explore critical issues, including bias and fairness, security and privacy, uncertainty, explainability, and distribution shift. Our research also covers model limitations, such as hallucinations, as well as methods like alignment and Artificial Intelligence-Generated Content (AIGC) detection. For each area, we review the current state of the field and outline concrete future research directions. Additionally, we discuss the intersections between these areas, highlighting their connections and shared challenges. We hope our survey fosters the development of foundation models that are not only powerful but also ethical, trustworthy, reliable, and socially responsible. 

---
# Large language models for spreading dynamics in complex systems 

**Authors**: Shuyu Jiang, Hao Ren, Yichang Gao, Yi-Cheng Zhang, Li Qi, Dayong Xiao, Jie Fan, Rui Tang, Wei Wang  

**Link**: [PDF](https://arxiv.org/pdf/2602.08085)  

**Abstract**: Spreading dynamics is a central topic in the physics of complex systems and network science, providing a unified framework for understanding how information, behaviors, and diseases propagate through interactions among system units. In many propagation contexts, spreading processes are influenced by multiple interacting factors, such as information expression patterns, cultural contexts, living environments, cognitive preferences, and public policies, which are difficult to incorporate directly into classical modeling frameworks. Recently, large language models (LLMs) have exhibited strong capabilities in natural language understanding, reasoning, and generation, enabling explicit perception of semantic content and contextual cues in spreading processes, thereby supporting the analysis of the different influencing factors. Beyond serving as external analytical tools, LLMs can also act as interactive agents embedded in propagation systems, potentially influencing spreading pathways and feedback structures. Consequently, the roles and impacts of LLMs on spreading dynamics have become an active and rapidly growing research area across multiple research disciplines. This review provides a comprehensive overview of recent advances in applying LLMs to the study of spreading dynamics across two representative domains: digital epidemics, such as misinformation and rumors, and biological epidemics, including infectious disease outbreaks. We first examine the foundations of epidemic modeling from a complex-systems perspective and discuss how LLM-based approaches relate to traditional frameworks. We then systematically review recent studies from three key perspectives, which are epidemic modeling, epidemic detection and surveillance, and epidemic prediction and management, to clarify how LLMs enhance these areas. Finally, open challenges and potential research directions are discussed. 

---
# VidVec: Unlocking Video MLLM Embeddings for Video-Text Retrieval 

**Authors**: Issar Tzachor, Dvir Samuel, Rami Ben-Ari  

**Link**: [PDF](https://arxiv.org/pdf/2602.08099)  

**Abstract**: Recent studies have adapted generative Multimodal Large Language Models (MLLMs) into embedding extractors for vision tasks, typically through fine-tuning to produce universal representations. However, their performance on video remains inferior to Video Foundation Models (VFMs). In this paper, we focus on leveraging MLLMs for video-text embedding and retrieval. We first conduct a systematic layer-wise analysis, showing that intermediate (pre-trained) MLLM layers already encode substantial task-relevant information. Leveraging this insight, we demonstrate that combining intermediate-layer embeddings with a calibrated MLLM head yields strong zero-shot retrieval performance without any training. Building on these findings, we introduce a lightweight text-based alignment strategy which maps dense video captions to short summaries and enables task-related video-text embedding learning without visual supervision. Remarkably, without any fine-tuning beyond text, our method outperforms current methods, often by a substantial margin, achieving state-of-the-art results across common video retrieval benchmarks. 

---
# Online Domain-aware LLM Decoding for Continual Domain Evolution 

**Authors**: Mohammad Abu-Shaira, Weishi Shi  

**Link**: [PDF](https://arxiv.org/pdf/2602.08088)  

**Abstract**: LLMs are typically fine-tuned offline on domain-specific data, assuming a static domain. In practice, domain knowledge evolves continuously through new regulations, products, services, and interaction patterns. Retraining or fine-tuning LLMs for every new instance is computationally infeasible. Additionally, real-world environments also exhibit temporal dynamics with shifting data distributions. Disregarding this phenomenon, commonly referred to as concept drift, can significantly diminish a model's predictive accuracy. This mismatch between evolving domains and static adaptation pipelines highlights the need for efficient, real-time adaptation without costly retraining. In response, we introduce Online Domain-aware Decoding framework (ODD). ODD performs probability-level fusion between a base LLM and a prefix-tree prior, guided by adaptive confidence modulation using disagreement and continuity signals. Empirical evaluation under diverse drift scenarios demonstrates that ODD consistently surpasses LLM-Greedy and LLM-Temp Scaled across all syntactic and semantic NLG metrics. It yields an absolute ROUGE-L gain of 0.065 and a 13.6% relative improvement in Cosine Similarity over the best baseline. These results demonstrate ODD 's robustness to evolving lexical and contextual patterns, making it suitable for dynamic LLM applications. 

---
# CyberExplorer: Benchmarking LLM Offensive Security Capabilities in a Real-World Attacking Simulation Environment 

**Authors**: Nanda Rani, Kimberly Milner, Minghao Shao, Meet Udeshi, Haoran Xi, Venkata Sai Charan Putrevu, Saksham Aggarwal, Sandeep K. Shukla, Prashanth Krishnamurthy, Farshad Khorrami, Muhammad Shafique, Ramesh Karri  

**Link**: [PDF](https://arxiv.org/pdf/2602.08023)  

**Abstract**: Real-world offensive security operations are inherently open-ended: attackers explore unknown attack surfaces, revise hypotheses under uncertainty, and operate without guaranteed success. Existing LLM-based offensive agent evaluations rely on closed-world settings with predefined goals and binary success criteria. To address this gap, we introduce CyberExplorer, an evaluation suite with two core components: (1) an open-environment benchmark built on a virtual machine hosting 40 vulnerable web services derived from real-world CTF challenges, where agents autonomously perform reconnaissance, target selection, and exploitation without prior knowledge of vulnerability locations; and (2) a reactive multi-agent framework supporting dynamic exploration without predefined plans. CyberExplorer enables fine-grained evaluation beyond flag recovery, capturing interaction dynamics, coordination behavior, failure modes, and vulnerability discovery signals-bridging the gap between benchmarks and realistic multi-target attack scenarios. 

---
# DeltaKV: Residual-Based KV Cache Compression via Long-Range Similarity 

**Authors**: Jitai Hao, Qiang Huang, Yaowei Wang, Min Zhang, Jun Yu  

**Link**: [PDF](https://arxiv.org/pdf/2602.08005)  

**Abstract**: The deployment of efficient long-context LLMs in applications like autonomous agents, long-chain reasoning, and creative writing is fundamentally bottlenecked by the linear growth of KV cache memory. Existing compression and eviction methods often struggle to balance accuracy, compression ratio, and hardware efficiency. We propose DeltaKV, a residual-based KV cache compression framework motivated by two empirical findings: long-range inter-token similarity and highly shared latent components in KV representations. Instead of discarding tokens, DeltaKV encodes semantic residuals relative to retrieved historical references, preserving fidelity while substantially reducing storage. To translate compression gains into real system speedups, we further introduce Sparse-vLLM, a high-performance inference engine with decoupled memory management and kernels optimized for sparse and irregular KV layouts. Experiments show that DeltaKV reduces KV cache memory to 29\% of the original while maintaining near-lossless accuracy on LongBench, SCBench, and AIME. When integrated with Sparse-vLLM, it achieves up to 2$\times$ throughput improvement over vLLM in long-context scenarios, demonstrating a practical path toward scalable long-context LLM deployment. Code, model checkpoints, and datasets are available at this https URL. 

---
# Emergent Search and Backtracking in Latent Reasoning Models 

**Authors**: Jasmine Cui, Charles Ye  

**Link**: [PDF](https://arxiv.org/pdf/2602.08100)  

**Abstract**: What happens when a language model thinks without words? Standard reasoning LLMs verbalize intermediate steps as chain-of-thought; latent reasoning transformers (LRTs) instead perform deliberation entirely in continuous hidden space. We investigate an LRT, decoding the model's evolving beliefs at every step on a multiple-choice QA benchmark. We find that the model spontaneously learns a structured search process in latent space. Deliberation follows a consistent trajectory: an exploration phase where probability mass spreads across candidates, tentative commitment to a frontrunner, and either convergence or backtracking. Backtracking is prevalent (32% of instances), beneficial (34% accuracy gain over non-backtracking instances), and predominantly directed away from the semantically closest distractor toward the correct answer. The search is adaptive: replacing distractors with implausible alternatives shortens exploration by 54%. Latent reasoning models achieve in activation space what chain-of-thought achieves through words: the ability to be wrong, notice, and recover. 

---
# Don't Always Pick the Highest-Performing Model: An Information Theoretic View of LLM Ensemble Selection 

**Authors**: Yigit Turkmen, Baturalp Buyukates, Melih Bastopcu  

**Link**: [PDF](https://arxiv.org/pdf/2602.08003)  

**Abstract**: Large language models (LLMs) are often ensembled together to improve overall reliability and robustness, but in practice models are strongly correlated. This raises a fundamental question: which models should be selected when forming an LLM ensemble? We formulate budgeted ensemble selection as maximizing the mutual information between the true label and predictions of the selected models. Furthermore, to explain why performance can saturate even with many models, we model the correlated errors of the models using Gaussian-copula and show an information-theoretic error floor for the performance of the ensemble. Motivated by these, we propose a simple greedy mutual-information selection algorithm that estimates the required information terms directly from data and iteratively builds an ensemble under a query budget. We test our approach in two question answering datasets and one binary sentiment classification dataset: MEDMCQA, MMLU, and IMDB movie reviews. Across all datasets, we observe that our method consistently outperforms strong baselines under the same query budget. 

---
# Accuracy-Delay Trade-Off in LLM Offloading via Token-Level Uncertainty 

**Authors**: Yumin Kim, Hyeonsu Lyu, Minjae Lee, Hyun Jong Yang  

**Link**: [PDF](https://arxiv.org/pdf/2602.07958)  

**Abstract**: Large language models (LLMs) offer significant potential for intelligent mobile services but are computationally intensive for resource-constrained devices. Mobile edge computing (MEC) allows such devices to offload inference tasks to edge servers (ESs), yet introduces latency due to communication and serverside queuing, especially in multi-user environments. In this work, we propose an uncertainty-aware offloading framework that dynamically decides whether to perform inference locally or offload it to the ES, based on token-level uncertainty and resource constraints. We define a margin-based token-level uncertainty metric and demonstrate its correlation with model accuracy. Leveraging this metric, we design a greedy offloading algorithm (GOA) that minimizes delay while maintaining accuracy by prioritizing offloading for highuncertainty queries. Our experiments show that GOA consistently achieves a favorable trade-off, outperforming baseline strategies in both accuracy and latency across varying user densities, and operates with practical computation time. These results establish GOA as a scalable and effective solution for LLM inference in MEC environments. 

---
# Bielik Guard: Efficient Polish Language Safety Classifiers for LLM Content Moderation 

**Authors**: Krzysztof Wróbel, Jan Maria Kowalski, Jerzy Surma, Igor Ciuciura, Maciej Szymański  

**Link**: [PDF](https://arxiv.org/pdf/2602.07954)  

**Abstract**: As Large Language Models (LLMs) become increasingly deployed in Polish language applications, the need for efficient and accurate content safety classifiers has become paramount. We present Bielik Guard, a family of compact Polish language safety classifiers comprising two model variants: a 0.1B parameter model based on MMLW-RoBERTa-base and a 0.5B parameter model based on PKOBP/polish-roberta-8k. Fine-tuned on a community-annotated dataset of 6,885 Polish texts, these models classify content across five safety categories: Hate/Aggression, Vulgarities, Sexual Content, Crime, and Self-Harm. Our evaluation demonstrates that both models achieve strong performance on multiple benchmarks. The 0.5B variant offers the best overall discrimination capability with F1 scores of 0.791 (micro) and 0.785 (macro) on the test set, while the 0.1B variant demonstrates exceptional efficiency. Notably, Bielik Guard 0.1B v1.1 achieves superior precision (77.65\%) and very low false positive rate (0.63\%) on real user prompts, outperforming HerBERT-PL-Guard (31.55\% precision, 4.70\% FPR) despite identical model size. The models are publicly available and designed to provide appropriate responses rather than simple content blocking, particularly for sensitive categories like self-harm. 

---
# Adaptive Acquisition Selection for Bayesian Optimization with Large Language Models 

**Authors**: Giang Ngo, Dat Phan Trong, Dang Nguyen, Sunil Gupta, Svetha Venkatesh  

**Link**: [PDF](https://arxiv.org/pdf/2602.07904)  

**Abstract**: Bayesian Optimization critically depends on the choice of acquisition function, but no single strategy is universally optimal; the best choice is non-stationary and problem-dependent. Existing adaptive portfolio methods often base their decisions on past function values while ignoring richer information like remaining budget or surrogate model characteristics. To address this, we introduce LMABO, a novel framework that casts a pre-trained Large Language Model (LLM) as a zero-shot, online strategist for the BO process. At each iteration, LMABO uses a structured state representation to prompt the LLM to select the most suitable acquisition function from a diverse portfolio. In an evaluation across 50 benchmark problems, LMABO demonstrates a significant performance improvement over strong static, adaptive portfolio, and other LLM-based baselines. We show that the LLM's behavior is a comprehensive strategy that adapts to real-time progress, proving its advantage stems from its ability to process and synthesize the complete optimization state into an effective, adaptive policy. 

---
# MCIE: Multimodal LLM-Driven Complex Instruction Image Editing with Spatial Guidance 

**Authors**: Xuehai Bai, Xiaoling Gu, Akide Liu, Hangjie Yuan, YiFan Zhang, Jack Ma  

**Link**: [PDF](https://arxiv.org/pdf/2602.07993)  

**Abstract**: Recent advances in instruction-based image editing have shown remarkable progress. However, existing methods remain limited to relatively simple editing operations, hindering real-world applications that require complex and compositional instructions. In this work, we address these limitations from the perspectives of architectural design, data, and evaluation protocols. Specifically, we identify two key challenges in current models: insufficient instruction compliance and background inconsistency. To this end, we propose MCIE-E1, a Multimodal Large Language Model-Driven Complex Instruction Image Editing method that integrates two key modules: a spatial-aware cross-attention module and a background-consistent cross-attention module. The former enhances instruction-following capability by explicitly aligning semantic instructions with spatial regions through spatial guidance during the denoising process, while the latter preserves features in unedited regions to maintain background consistency. To enable effective training, we construct a dedicated data pipeline to mitigate the scarcity of complex instruction-based image editing datasets, combining fine-grained automatic filtering via a powerful MLLM with rigorous human validation. Finally, to comprehensively evaluate complex instruction-based image editing, we introduce CIE-Bench, a new benchmark with two new evaluation metrics. Experimental results on CIE-Bench demonstrate that MCIE-E1 consistently outperforms previous state-of-the-art methods in both quantitative and qualitative assessments, achieving a 23.96% improvement in instruction compliance. 

---
# rePIRL: Learn PRM with Inverse RL for LLM Reasoning 

**Authors**: Xian Wu, Kaijie Zhu, Ying Zhang, Lun Wang, Wenbo Guo  

**Link**: [PDF](https://arxiv.org/pdf/2602.07832)  

**Abstract**: Process rewards have been widely used in deep reinforcement learning to improve training efficiency, reduce variance, and prevent reward hacking. In LLM reasoning, existing works also explore various solutions for learning effective process reward models (PRM) with or without the help of an expert policy. However, existing methods either rely on strong assumptions about the expert policies (e.g., requiring their reward functions) or suffer intrinsic limitations (e.g., entropy collapse), resulting in weak PRMs or limited generalizability. In this paper, we introduce rePIRL, an inverse RL-inspired framework that learns effective PRMs with minimal assumptions about expert policies. Specifically, we design a dual learning process that updates the policy and the PRM interchangeably. Our learning algorithm has customized techniques to address the challenges of scaling traditional inverse RL to LLMs. We theoretically show that our proposed learning framework can unify both online and offline PRM learning methods, justifying that rePIRL can learn PRMs with minimal assumptions. Empirical evaluations on standardized math and coding reasoning datasets demonstrate the effectiveness of rePIRL over existing methods. We further show the application of our trained PRM in test-time training, test-time scaling, and providing an early signal for training hard problems. Finally, we validate our training recipe and key design choices via a detailed ablation study. 

---
# SPD-Faith Bench: Diagnosing and Improving Faithfulness in Chain-of-Thought for Multimodal Large Language Models 

**Authors**: Weijiang Lv, Yaoxuan Feng, Xiaobo Xia, Jiayu Wang, Yan Jing, Wenchao Chen, Bo Chen  

**Link**: [PDF](https://arxiv.org/pdf/2602.07833)  

**Abstract**: Chain-of-Thought reasoning is widely used to improve the interpretability of multimodal large language models (MLLMs), yet the faithfulness of the generated reasoning traces remains unclear. Prior work has mainly focused on perceptual hallucinations, leaving reasoning level unfaithfulness underexplored. To isolate faithfulness from linguistic priors, we introduce SPD-Faith Bench, a diagnostic benchmark based on fine-grained image difference reasoning that enforces explicit visual comparison. Evaluations on state-of-the-art MLLMs reveal two systematic failure modes, perceptual blindness and perception-reasoning dissociation. We trace these failures to decaying visual attention and representation shifts in the residual stream. Guided by this analysis, we propose SAGE, a train-free visual evidence-calibrated framework that improves visual routing and aligns reasoning with perception. Our results highlight the importance of explicitly evaluating faithfulness beyond response correctness. Our benchmark and codes are available at this https URL. 

---
# Efficient Representations are Controllable Representations 

**Authors**: Charles Ye, Jasmine Cui  

**Link**: [PDF](https://arxiv.org/pdf/2602.07828)  

**Abstract**: What is the most brute-force way to install interpretable, controllable features into a model's activations? Controlling how LLMs internally represent concepts typically requires sophisticated methods to first identify, then intervene on the model's existing feature geometry. We bypass all of this.
We finetune an LLM with a simple auxiliary loss, training 16 of its 3072 residual stream dimensions to be inert interpretability flags that simply indicate what concepts are required for generation. The model reorganizes around them anyway, learning to rely on these flags during actual generation tasks. As a result, these inert flags become genuine internal features: interpretable control switches that allow us to steer generation at inference time. Why does this work? When a feature is reliably supplied at a fixed location, gradient descent gradually eliminates redundant encodings elsewhere, and the model erodes its own alternative representations. A model's efficiency pressure is a lever - exploitable to induce interpretable, controllable representations. 

---
# Pruning as a Cooperative Game: Surrogate-Assisted Layer Contribution Estimation for Large Language Models 

**Authors**: Xuan Ding, Pengyu Tong, Ranjie Duan, Yunjian Zhang, Rui Sun, Yao Zhu  

**Link**: [PDF](https://arxiv.org/pdf/2602.07804)  

**Abstract**: While large language models (LLMs) demonstrate impressive performance across various tasks, their deployment in real-world scenarios is still constrained by high computational demands. Layer-wise pruning, a commonly employed strategy to mitigate inference costs, can partially address this challenge. However, existing approaches generally depend on static heuristic rules and fail to account for the interdependencies among layers, thereby limiting the effectiveness of the pruning process. To this end, this paper proposes a game-theoretic framework that formulates layer pruning as a cooperative game in which each layer acts as a player and model performance serves as the utility. As computing exact Shapley values is computationally infeasible for large language models (LLMs), we propose using a lightweight surrogate network to estimate layer-wise marginal contributions. This network can predict LLM performance for arbitrary layer combinations at a low computational cost. Additionally, we employ stratified Monte Carlo mask sampling to further reduce the cost of Sharpley value estimation. This approach captures inter-layer dependencies and dynamically identifies critical layers for pruning. Extensive experiments demonstrate the consistent superiority of our method in terms of perplexity and zero-shot accuracy, achieving more efficient and effective layer-wise pruning for large language models. 

---
# Emergent Structured Representations Support Flexible In-Context Inference in Large Language Models 

**Authors**: Ningyu Xu, Qi Zhang, Xipeng Qiu, Xuanjing Huang  

**Link**: [PDF](https://arxiv.org/pdf/2602.07794)  

**Abstract**: Large language models (LLMs) exhibit emergent behaviors suggestive of human-like reasoning. While recent work has identified structured, human-like conceptual representations within these models, it remains unclear whether they functionally rely on such representations for reasoning. Here we investigate the internal processing of LLMs during in-context concept inference. Our results reveal a conceptual subspace emerging in middle to late layers, whose representational structure persists across contexts. Using causal mediation analyses, we demonstrate that this subspace is not merely an epiphenomenon but is functionally central to model predictions, establishing its causal role in inference. We further identify a layer-wise progression where attention heads in early-to-middle layers integrate contextual cues to construct and refine the subspace, which is subsequently leveraged by later layers to generate predictions. Together, these findings provide evidence that LLMs dynamically construct and use structured, latent representations in context for inference, offering insights into the computational processes underlying flexible adaptation. 

---
# CausalTAD: Injecting Causal Knowledge into Large Language Models for Tabular Anomaly Detection 

**Authors**: Ruiqi Wang, Ruikang Liu, Runyu Chen, Haoxiang Suo, Zhiyi Peng, Zhuo Tang, Changjian Chen  

**Link**: [PDF](https://arxiv.org/pdf/2602.07798)  

**Abstract**: Detecting anomalies in tabular data is critical for many real-world applications, such as credit card fraud detection. With the rapid advancements in large language models (LLMs), state-of-the-art performance in tabular anomaly detection has been achieved by converting tabular data into text and fine-tuning LLMs. However, these methods randomly order columns during conversion, without considering the causal relationships between them, which is crucial for accurately detecting anomalies. In this paper, we present CausalTaD, a method that injects causal knowledge into LLMs for tabular anomaly detection. We first identify the causal relationships between columns and reorder them to align with these causal relationships. This reordering can be modeled as a linear ordering problem. Since each column contributes differently to the causal relationships, we further propose a reweighting strategy to assign different weights to different columns to enhance this effect. Experiments across more than 30 datasets demonstrate that our method consistently outperforms the current state-of-the-art methods. The code for CausalTAD is available at this https URL. 

---
# Generative Reasoning Re-ranker 

**Authors**: Mingfu Liang, Yufei Li, Jay Xu, Kavosh Asadi, Xi Liu, Shuo Gu, Kaushik Rangadurai, Frank Shyu, Shuaiwen Wang, Song Yang, Zhijing Li, Jiang Liu, Mengying Sun, Fei Tian, Xiaohan Wei, Chonglin Sun, Jacob Tao, Shike Mei, Hamed Firooz, Wenlin Chen, Luke Simon  

**Link**: [PDF](https://arxiv.org/pdf/2602.07774)  

**Abstract**: Recent studies increasingly explore Large Language Models (LLMs) as a new paradigm for recommendation systems due to their scalability and world knowledge. However, existing work has three key limitations: (1) most efforts focus on retrieval and ranking, while the reranking phase, critical for refining final recommendations, is largely overlooked; (2) LLMs are typically used in zero-shot or supervised fine-tuning settings, leaving their reasoning abilities, especially those enhanced through reinforcement learning (RL) and high-quality reasoning data, underexploited; (3) items are commonly represented by non-semantic IDs, creating major scalability challenges in industrial systems with billions of identifiers. To address these gaps, we propose the Generative Reasoning Reranker (GR2), an end-to-end framework with a three-stage training pipeline tailored for reranking. First, a pretrained LLM is mid-trained on semantic IDs encoded from non-semantic IDs via a tokenizer achieving $\ge$99% uniqueness. Next, a stronger larger-scale LLM generates high-quality reasoning traces through carefully designed prompting and rejection sampling, which are used for supervised fine-tuning to impart foundational reasoning skills. Finally, we apply Decoupled Clip and Dynamic sAmpling Policy Optimization (DAPO), enabling scalable RL supervision with verifiable rewards designed specifically for reranking. Experiments on two real-world datasets demonstrate GR2's effectiveness: it surpasses the state-of-the-art OneRec-Think by 2.4% in Recall@5 and 1.3% in NDCG@5. Ablations confirm that advanced reasoning traces yield substantial gains across metrics. We further find that RL reward design is crucial in reranking: LLMs tend to exploit reward hacking by preserving item order, motivating conditional verifiable rewards to mitigate this behavior and optimize reranking performance. 

---
# SoK: DARPA's AI Cyber Challenge (AIxCC): Competition Design, Architectures, and Lessons Learned 

**Authors**: Cen Zhang, Younggi Park, Fabian Fleischer, Yu-Fu Fu, Jiho Kim, Dongkwan Kim, Youngjoon Kim, Qingxiao Xu, Andrew Chin, Ze Sheng, Hanqing Zhao, Brian J. Lee, Joshua Wang, Michael Pelican, David J. Musliner, Jeff Huang, Jon Silliman, Mikel Mcdaniel, Jefferson Casavant, Isaac Goldthwaite, Nicholas Vidovich, Matthew Lehman, Taesoo Kim  

**Link**: [PDF](https://arxiv.org/pdf/2602.07666)  

**Abstract**: DARPA's AI Cyber Challenge (AIxCC, 2023--2025) is the largest competition to date for building fully autonomous cyber reasoning systems (CRSs) that leverage recent advances in AI -- particularly large language models (LLMs) -- to discover and remediate vulnerabilities in real-world open-source software. This paper presents the first systematic analysis of AIxCC. Drawing on design documents, source code, execution traces, and discussions with organizers and competing teams, we examine the competition's structure and key design decisions, characterize the architectural approaches of finalist CRSs, and analyze competition results beyond the final scoreboard. Our analysis reveals the factors that truly drove CRS performance, identifies genuine technical advances achieved by teams, and exposes limitations that remain open for future research. We conclude with lessons for organizing future competitions and broader insights toward deploying autonomous CRSs in practice. 

---
# Evaluating Large Language Models for Detecting Architectural Decision Violations 

**Authors**: Ruoyu Su, Alexander Bakhtin, Noman Ahmad, Matteo Esposito, Valentina Lenarduzzi, Davide Taibi  

**Link**: [PDF](https://arxiv.org/pdf/2602.07609)  

**Abstract**: Architectural Decision Records (ADRs) play a central role in maintaining software architecture quality, yet many decision violations go unnoticed because projects lack both systematic documentation and automated detection mechanisms. Recent advances in Large Language Models (LLMs) open up new possibilities for automating architectural reasoning at scale. We investigated how effectively LLMs can identify decision violations in open-source systems by examining their agreement, accuracy, and inherent limitations. Our study analyzed 980 ADRs across 109 GitHub repositories using a multi-model pipeline in which one LLM primary screens potential decision violations, and three additional LLMs independently validate the reasoning. We assessed agreement, accuracy, precision, and recall, and complemented the quantitative findings with expert evaluation. The models achieved substantial agreement and strong accuracy for explicit, code-inferable decisions. Accuracy falls short for implicit or deployment-oriented decisions that depend on deployment configuration or organizational knowledge. Therefore, LLMs can meaningfully support validation of architectural decision compliance; however, they are not yet replacing human expertise for decisions not focused on code. 

---
# Learning to Self-Verify Makes Language Models Better Reasoners 

**Authors**: Yuxin Chen, Yu Wang, Yi Zhang, Ziang Ye, Zhengzhou Cai, Yaorui Shi, Qi Gu, Hui Su, Xunliang Cai, Xiang Wang, An Zhang, Tat-Seng Chua  

**Link**: [PDF](https://arxiv.org/pdf/2602.07594)  

**Abstract**: Recent large language models (LLMs) achieve strong performance in generating promising reasoning paths for complex tasks. However, despite powerful generation ability, LLMs remain weak at verifying their own answers, revealing a persistent capability asymmetry between generation and self-verification. In this work, we conduct an in-depth investigation of this asymmetry throughout training evolution and show that, even on the same task, improving generation does not lead to corresponding improvements in self-verification. Interestingly, we find that the reverse direction of this asymmetry behaves differently: learning to self-verify can effectively improve generation performance, achieving accuracy comparable to standard generation training while yielding more efficient and effective reasoning traces. Building on this observation, we further explore integrating self-verification into generation training by formulating a multi-task reinforcement learning framework, where generation and self-verification are optimized as two independent but complementary objectives. Extensive experiments across benchmarks and models demonstrate performance gains over generation-only training in both generation and verification capabilities. 

---
# Fine-R1: Make Multi-modal LLMs Excel in Fine-Grained Visual Recognition by Chain-of-Thought Reasoning 

**Authors**: Hulingxiao He, Zijun Geng, Yuxin Peng  

**Link**: [PDF](https://arxiv.org/pdf/2602.07605)  

**Abstract**: Any entity in the visual world can be hierarchically grouped based on shared characteristics and mapped to fine-grained sub-categories. While Multi-modal Large Language Models (MLLMs) achieve strong performance on coarse-grained visual tasks, they often struggle with Fine-Grained Visual Recognition (FGVR). Adapting general-purpose MLLMs to FGVR typically requires large amounts of annotated data, which is costly to obtain, leaving a substantial performance gap compared to contrastive CLIP models dedicated for discriminative tasks. Moreover, MLLMs tend to overfit to seen sub-categories and generalize poorly to unseen ones. To address these challenges, we propose Fine-R1, an MLLM tailored for FGVR through an R1-style training framework: (1) Chain-of-Thought Supervised Fine-tuning, where we construct a high-quality FGVR CoT dataset with rationales of "visual analysis, candidate sub-categories, comparison, and prediction", transition the model into a strong open-world classifier; and (2) Triplet Augmented Policy Optimization, where Intra-class Augmentation mixes trajectories from anchor and positive images within the same category to improve robustness to intra-class variance, while Inter-class Augmentation maximizes the response distinction conditioned on images across sub-categories to enhance discriminative ability. With only 4-shot training, Fine-R1 outperforms existing general MLLMs, reasoning MLLMs, and even contrastive CLIP models in identifying both seen and unseen sub-categories, showing promise in working in knowledge-intensive domains where gathering expert annotations for all sub-categories is arduous. Code is available at this https URL. 

---
# MemPot: Defending Against Memory Extraction Attack with Optimized Honeypots 

**Authors**: Yuhao Wang, Shengfang Zhai, Guanghao Jin, Yinpeng Dong, Linyi Yang, Jiaheng Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2602.07517)  

**Abstract**: Large Language Model (LLM)-based agents employ external and internal memory systems to handle complex, goal-oriented tasks, yet this exposes them to severe extraction attacks, and effective defenses remain lacking. In this paper, we propose MemPot, the first theoretically verified defense framework against memory extraction attacks by injecting optimized honeypots into the memory. Through a two-stage optimization process, MemPot generates trap documents that maximize the retrieval probability for attackers while remaining inconspicuous to benign users. We model the detection process as Wald's Sequential Probability Ratio Test (SPRT) and theoretically prove that MemPot achieves a lower average number of sampling rounds compared to optimal static detectors. Empirically, MemPot significantly outperforms state-of-the-art baselines, achieving a 50% improvement in detection AUROC and an 80% increase in True Positive Rate under low False Positive Rate constraints. Furthermore, our experiments confirm that MemPot incurs zero additional online inference latency and preserves the agent's utility on standard tasks, verifying its superiority in safety, harmlessness, and efficiency. 

---
# How does longer temporal context enhance multimodal narrative video processing in the brain? 

**Authors**: Prachi Jindal, Anant Khandelwal, Manish Gupta, Bapi S. Raju, Subba Reddy Oota, Tanmoy Chakraborty  

**Link**: [PDF](https://arxiv.org/pdf/2602.07570)  

**Abstract**: Understanding how humans and artificial intelligence systems process complex narrative videos is a fundamental challenge at the intersection of neuroscience and machine learning. This study investigates how the temporal context length of video clips (3--12 s clips) and the narrative-task prompting shape brain-model alignment during naturalistic movie watching. Using fMRI recordings from participants viewing full-length movies, we examine how brain regions sensitive to narrative context dynamically represent information over varying timescales and how these neural patterns align with model-derived features. We find that increasing clip duration substantially improves brain alignment for multimodal large language models (MLLMs), whereas unimodal video models show little to no gain. Further, shorter temporal windows align with perceptual and early language regions, while longer windows preferentially align higher-order integrative regions, mirrored by a layer-to-cortex hierarchy in MLLMs. Finally, narrative-task prompts (multi-scene summary, narrative summary, character motivation, and event boundary detection) elicit task-specific, region-dependent brain alignment patterns and context-dependent shifts in clip-level tuning in higher-order regions. Together, our results position long-form narrative movies as a principled testbed for probing biologically relevant temporal integration and interpretable representations in long-context MLLMs. 

---
# MDL: A Unified Multi-Distribution Learner in Large-scale Industrial Recommendation through Tokenization 

**Authors**: Shanlei Mu, Yuchen Jiang, Shikang Wu, Shiyong Hong, Tianmu Sha, Junjie Zhang, Jie Zhu, Zhe Chen, Zhe Wang, Jingjian Lin  

**Link**: [PDF](https://arxiv.org/pdf/2602.07520)  

**Abstract**: Industrial recommender systems increasingly adopt multi-scenario learning (MSL) and multi-task learning (MTL) to handle diverse user interactions and contexts, but existing approaches suffer from two critical drawbacks: (1) underutilization of large-scale model parameters due to limited interaction with complex feature modules, and (2) difficulty in jointly modeling scenario and task information in a unified framework. To address these challenges, we propose a unified \textbf{M}ulti-\textbf{D}istribution \textbf{L}earning (MDL) framework, inspired by the "prompting" paradigm in large language models (LLMs). MDL treats scenario and task information as specialized tokens rather than auxiliary inputs or gating signals. Specifically, we introduce a unified information tokenization module that transforms features, scenarios, and tasks into a unified tokenized format. To facilitate deep interaction, we design three synergistic mechanisms: (1) feature token self-attention for rich feature interactions, (2) domain-feature attention for scenario/task-adaptive feature activation, and (3) domain-fused aggregation for joint distribution prediction. By stacking these interactions, MDL enables scenario and task information to "prompt" and activate the model's vast parameter space in a bottom-up, layer-wise manner. Extensive experiments on real-world industrial datasets demonstrate that MDL significantly outperforms state-of-the-art MSL and MTL baselines. Online A/B testing on Douyin Search platform over one month yields +0.0626\% improvement in LT30 and -0.3267\% reduction in change query rate. MDL has been fully deployed in production, serving hundreds of millions of users daily. 

---
# Advantages of Domain Knowledge Injection for Legal Document Summarization: A Case Study on Summarizing Indian Court Judgments in English and Hindi 

**Authors**: Debtanu Datta, Rajdeep Mukherjee, Adrijit Goswami, Saptarshi Ghosh  

**Link**: [PDF](https://arxiv.org/pdf/2602.07382)  

**Abstract**: Summarizing Indian legal court judgments is a complex task not only due to the intricate language and unstructured nature of the legal texts, but also since a large section of the Indian population does not understand the complex English in which legal text is written, thus requiring summaries in Indian languages. In this study, we aim to improve the summarization of Indian legal text to generate summaries in both English and Hindi (the most widely spoken Indian language), by injecting domain knowledge into diverse summarization models. We propose a framework to enhance extractive neural summarization models by incorporating domain-specific pre-trained encoders tailored for legal texts. Further, we explore the injection of legal domain knowledge into generative models (including Large Language Models) through continual pre-training on large legal corpora in English and Hindi. Our proposed approaches achieve statistically significant improvements in both English-to-English and English-to-Hindi Indian legal document summarization, as measured by standard evaluation metrics, factual consistency metrics, and legal domain-specific metrics. Furthermore, these improvements are validated through domain experts, demonstrating the effectiveness of our approaches. 

---
# Intent Mismatch Causes LLMs to Get Lost in Multi-Turn Conversation 

**Authors**: Geng Liu, Fei Zhu, Rong Feng, Changyi Ma, Shiqi Wang, Gaofeng Meng  

**Link**: [PDF](https://arxiv.org/pdf/2602.07338)  

**Abstract**: Multi-turn conversation has emerged as a predominant interaction paradigm for Large Language Models (LLMs). Users often employ follow-up questions to refine their intent, expecting LLMs to adapt dynamically. However, recent research reveals that LLMs suffer a substantial performance drop in multi-turn settings compared to single-turn interactions with fully specified instructions, a phenomenon termed ``Lost in Conversation'' (LiC). While this prior work attributes LiC to model unreliability, we argue that the root cause lies in an intent alignment gap rather than intrinsic capability deficits. In this paper, we first demonstrate that LiC is not a failure of model capability but rather a breakdown in interaction between users and LLMs. We theoretically show that scaling model size or improving training alone cannot resolve this gap, as it arises from structural ambiguity in conversational context rather than representational limitations. To address this, we propose to decouple intent understanding from task execution through a Mediator-Assistant architecture. By utilizing an experience-driven Mediator to explicate user inputs into explicit, well-structured instructions based on historical interaction patterns, our approach effectively bridges the gap between vague user intent and model interpretation. Experimental results demonstrate that this method significantly mitigates performance degradation in multi-turn conversations across diverse LLMs. 

---
# Semantic Search At LinkedIn 

**Authors**: Fedor Borisyuk, Sriram Vasudevan, Muchen Wu, Guoyao Li, Benjamin Le, Shaobo Zhang, Qianqi Kay Shen, Yuchin Juan, Kayhan Behdin, Liming Dong, Kaixu Yang, Shusen Jing, Ravi Pothamsetty, Rajat Arora, Sophie Yanying Sheng, Vitaly Abdrashitov, Yang Zhao, Lin Su, Xiaoqing Wang, Chujie Zheng, Sarang Metkar, Rupesh Gupta, Igor Lapchuk, David N. Racca, Madhumitha Mohan, Yanbo Li, Haojun Li, Saloni Gandhi, Xueying Lu, Chetan Bhole, Ali Hooshmand, Xin Yang, Raghavan Muthuregunathan, Jiajun Zhang, Mathew Teoh, Adam Coler, Abhinav Gupta, Xiaojing Ma, Sundara Raman Ramachandran, Morteza Ramezani, Yubo Wang, Lijuan Zhang, Richard Li, Jian Sheng, Chanh Nguyen, Yen-Chi Chen, Chuanrui Zhu, Claire Zhang, Jiahao Xu, Deepti Kulkarni, Qing Lan, Arvind Subramaniam, Ata Fatahibaarzi, Steven Shimizu, Yanning Chen, Zhipeng Wang, Ran He, Zhengze Zhou, Qingquan Song, Yun Dai, Caleb Johnson, Ping Liu, Shaghayegh Gharghabi, Gokulraj Mohanasundaram, Juan Bottaro, Santhosh Sachindran, Qi Guo, Yunxiang Ren, Chengming Jiang, Di Mo, Luke Simon, Jianqiang Shen, Jingwei Wu, Wenjing Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2602.07309)  

**Abstract**: Semantic search with large language models (LLMs) enables retrieval by meaning rather than keyword overlap, but scaling it requires major inference efficiency advances. We present LinkedIn's LLM-based semantic search framework for AI Job Search and AI People Search, combining an LLM relevance judge, embedding-based retrieval, and a compact Small Language Model trained via multi-teacher distillation to jointly optimize relevance and engagement. A prefill-oriented inference architecture co-designed with model pruning, context compression, and text-embedding hybrid interactions boosts ranking throughput by over 75x under a fixed latency constraint while preserving near-teacher-level NDCG, enabling one of the first production LLM-based ranking systems with efficiency comparable to traditional approaches and delivering significant gains in quality and user engagement. 

---
# Beyond Accuracy: Risk-Sensitive Evaluation of Hallucinated Medical Advice 

**Authors**: Savan Doshi  

**Link**: [PDF](https://arxiv.org/pdf/2602.07319)  

**Abstract**: Large language models are increasingly being used in patient-facing medical question answering, where hallucinated outputs can vary widely in potential harm. However, existing hallucination standards and evaluation metrics focus primarily on factual correctness, treating all errors as equally severe. This obscures clinically relevant failure modes, particularly when models generate unsupported but actionable medical language. We propose a risk-sensitive evaluation framework that quantifies hallucinations through the presence of risk-bearing language, including treatment directives, contraindications, urgency cues, and mentions of high-risk medications. Rather than assessing clinical correctness, our approach evaluates the potential impact of hallucinated content if acted upon. We further combine risk scoring with a relevance measure to identify high-risk, low-grounding failures. We apply this framework to three instruction-tuned language models using controlled patient-facing prompts designed as safety stress tests. Our results show that models with similar surface-level behavior exhibit substantially different risk profiles and that standard evaluation metrics fail to capture these distinctions. These findings highlight the importance of incorporating risk sensitivity into hallucination evaluation and suggest that evaluation validity is critically dependent on task and prompt design. 

---
# Fin-RATE: A Real-world Financial Analytics and Tracking Evaluation Benchmark for LLMs on SEC Filings 

**Authors**: Yidong Jiang, Junrong Chen, Eftychia Makri, Jialin Chen, Peiwen Li, Ali Maatouk, Leandros Tassiulas, Eliot Brenner, Bing Xiang, Rex Ying  

**Link**: [PDF](https://arxiv.org/pdf/2602.07294)  

**Abstract**: With increasing deployment of Large Language Models (LLMs) in the finance domain, LLMs are increasingly expected to parse complex regulatory disclosures. However, existing benchmarks often focus on isolated details, failing to reflect the complexity of professional analysis that requires synthesizing information across multiple documents, reporting periods, and corporate entities. They do not distinguish whether errors stem from retrieval failures, generation flaws, finance-specific reasoning mistakes, or misunderstanding of the query or context. This makes it difficult to pinpoint performance bottlenecks. To bridge these gaps, we introduce Fin-RATE, a benchmark built on U.S. Securities and Exchange Commission (SEC) filings and mirror financial analyst workflows through three pathways: detail-oriented reasoning within individual disclosures, cross-entity comparison under shared topics, and longitudinal tracking of the same firm across reporting periods. We benchmark 17 leading LLMs, spanning open-source, closed-source, and finance-specialized models, under both ground-truth context and retrieval-augmented settings. Results show substantial performance degradation, with accuracy dropping by 18.60% and 14.35% as tasks shift from single-document reasoning to longitudinal and cross-entity analysis. This is driven by rising comparison hallucinations, time and entity mismatches, and mirrored by declines in reasoning and factuality--limitations that prior benchmarks have yet to formally categorize or quantify. 

---
# Progressive Searching for Retrieval in RAG 

**Authors**: Taehee Jeong, Xingzhe Zhao, Peizu Li, Markus Valvur, Weihua Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2602.07297)  

**Abstract**: Retrieval Augmented Generation (RAG) is a promising technique for mitigating two key limitations of large language models (LLMs): outdated information and hallucinations. RAG system stores documents as embedding vectors in a database. Given a query, search is executed to find the most related documents. Then, the topmost matching documents are inserted into LLMs' prompt to generate a response. Efficient and accurate searching is critical for RAG to get relevant information. We propose a cost-effective searching algorithm for retrieval process. Our progressive searching algorithm incrementally refines the candidate set through a hierarchy of searches, starting from low-dimensional embeddings and progressing into a higher, target-dimensionality. This multi-stage approach reduces retrieval time while preserving the desired accuracy. Our findings demonstrate that progressive search in RAG systems achieves a balance between dimensionality, speed, and accuracy, enabling scalable and high-performance retrieval even for large databases. 

---
# ArcMark: Multi-bit LLM Watermark via Optimal Transport 

**Authors**: Atefeh Gilani, Carol Xuan Long, Sajani Vithana, Oliver Kosut, Lalitha Sankar, Flavio P. Calmon  

**Link**: [PDF](https://arxiv.org/pdf/2602.07235)  

**Abstract**: Watermarking is an important tool for promoting the responsible use of language models (LMs). Existing watermarks insert a signal into generated tokens that either flags LM-generated text (zero-bit watermarking) or encodes more complex messages (multi-bit watermarking). Though a number of recent multi-bit watermarks insert several bits into text without perturbing average next-token predictions, they largely extend design principles from the zero-bit setting, such as encoding a single bit per token. Notably, the information-theoretic capacity of multi-bit watermarking -- the maximum number of bits per token that can be inserted and detected without changing average next-token predictions -- has remained unknown. We address this gap by deriving the first capacity characterization of multi-bit watermarks. Our results inform the design of ArcMark: a new watermark construction based on coding-theoretic principles that, under certain assumptions, achieves the capacity of the multi-bit watermark channel. In practice, ArcMark outperforms competing multi-bit watermarks in terms of bit rate per token and detection accuracy. Our work demonstrates that LM watermarking is fundamentally a channel coding problem, paving the way for principled coding-theoretic approaches to watermark design. 

---
# Open TutorAI: An Open-source Platform for Personalized and Immersive Learning with Generative AI 

**Authors**: Mohamed El Hajji, Tarek Ait Baha, Aicha Dakir, Hammou Fadili, Youssef Es-Saady  

**Link**: [PDF](https://arxiv.org/pdf/2602.07176)  

**Abstract**: Recent advances in artificial intelligence have created new possibilities for making education more scalable, adaptive, and learner-centered. However, existing educational chatbot systems often lack contextual adaptability, real-time responsiveness, and pedagogical agility. which can limit learner engagement and diminish instructional effectiveness. Thus, there is a growing need for open, integrative platforms that combine AI and immersive technologies to support personalized, meaningful learning experiences. This paper presents Open TutorAI, an open-source educational platform based on LLMs and generative technologies that provides dynamic, personalized tutoring. The system integrates natural language processing with customizable 3D avatars to enable multimodal learner interaction. Through a structured onboarding process, it captures each learner's goals and preferences in order to configure a learner-specific AI assistant. This assistant is accessible via both text-based and avatar-driven interfaces. The platform includes tools for organizing content, providing embedded feedback, and offering dedicated interfaces for learners, educators, and parents. This work focuses on learner-facing components, delivering a tool for adaptive support that responds to individual learner profiles without requiring technical expertise. Its assistant-generation pipeline and avatar integration enhance engagement and emotional presence, creating a more humanized, immersive learning environment. Embedded learning analytics support self-regulated learning by tracking engagement patterns and generating actionable feedback. The result is Open TutorAI, which unites modular architecture, generative AI, and learner analytics within an open-source framework. It contributes to the development of next-generation intelligent tutoring systems. 

---
# Your Language Model Secretly Contains Personality Subnetworks 

**Authors**: Ruimeng Ye, Zihan Wang, Zinan Ling, Yang Xiao, Manling Li, Xiaolong Ma, Bo Hui  

**Link**: [PDF](https://arxiv.org/pdf/2602.07164)  

**Abstract**: Humans shift between different personas depending on social context. Large Language Models (LLMs) demonstrate a similar flexibility in adopting different personas and behaviors. Existing approaches, however, typically adapt such behavior through external knowledge such as prompting, retrieval-augmented generation (RAG), or fine-tuning. We ask: do LLMs really need external context or parameters to adapt to different behaviors, or do they already have such knowledge embedded in their parameters? In this work, we show that LLMs already contain persona-specialized subnetworks in their parameter space. Using small calibration datasets, we identify distinct activation signatures associated with different personas. Guided by these statistics, we develop a masking strategy that isolates lightweight persona subnetworks. Building on the findings, we further discuss: how can we discover opposing subnetwork from the model that lead to binary-opposing personas, such as introvert-extrovert? To further enhance separation in binary opposition scenarios, we introduce a contrastive pruning strategy that identifies parameters responsible for the statistical divergence between opposing personas. Our method is entirely training-free and relies solely on the language model's existing parameter space. Across diverse evaluation settings, the resulting subnetworks exhibit significantly stronger persona alignment than baselines that require external knowledge while being more efficient. Our findings suggest that diverse human-like behaviors are not merely induced in LLMs, but are already embedded in their parameter space, pointing toward a new perspective on controllable and interpretable personalization in large language models. 

---
# ShallowJail: Steering Jailbreaks against Large Language Models 

**Authors**: Shang Liu, Hanyu Pei, Zeyan Liu  

**Link**: [PDF](https://arxiv.org/pdf/2602.07107)  

**Abstract**: Large Language Models(LLMs) have been successful in numerous fields. Alignment has usually been applied to prevent them from harmful purposes. However, aligned LLMs remain vulnerable to jailbreak attacks that deliberately mislead them into producing harmful outputs. Existing jailbreaks are either black-box, using carefully crafted, unstealthy prompts, or white-box, requiring resource-intensive computation. In light of these challenges, we introduce ShallowJail, a novel attack that exploits shallow alignment in LLMs. ShallowJail can misguide LLMs' responses by manipulating the initial tokens during inference. Through extensive experiments, we demonstrate the effectiveness of~\shallow, which substantially degrades the safety of state-of-the-art LLM responses. 

---
# RealFin: How Well Do LLMs Reason About Finance When Users Leave Things Unsaid? 

**Authors**: Yuyang Dai, Yan Lin, Zhuohan Xie, Yuxia Wang  

**Link**: [PDF](https://arxiv.org/pdf/2602.07096)  

**Abstract**: Reliable financial reasoning requires knowing not only how to answer, but also when an answer cannot be justified. In real financial practice, problems often rely on implicit assumptions that are taken for granted rather than stated explicitly, causing problems to appear solvable while lacking enough information for a definite answer. We introduce REALFIN, a bilingual benchmark that evaluates financial reasoning by systematically removing essential premises from exam-style questions while keeping them linguistically plausible. Based on this, we evaluate models under three formulations that test answering, recognizing missing information, and rejecting unjustified options, and find consistent performance drops when key conditions are absent. General-purpose models tend to over-commit and guess, while most finance-specialized models fail to clearly identify missing premises. These results highlight a critical gap in current evaluations and show that reliable financial models must know when a question should not be answered. 

---
# Evaluating Retrieval-Augmented Generation Variants for Natural Language-Based SQL and API Call Generation 

**Authors**: Michael Marketsmüller, Simon Martin, Tim Schlippe  

**Link**: [PDF](https://arxiv.org/pdf/2602.07086)  

**Abstract**: Enterprise systems increasingly require natural language interfaces that can translate user requests into structured operations such as SQL queries and REST API calls. While large language models (LLMs) show promise for code generation [Chen et al., 2021; Huynh and Lin, 2025], their effectiveness in domain-specific enterprise contexts remains underexplored, particularly when both retrieval and modification tasks must be handled jointly. This paper presents a comprehensive evaluation of three retrieval-augmented generation (RAG) variants [Lewis et al., 2021] -- standard RAG, Self-RAG [Asai et al., 2024], and CoRAG [Wang et al., 2025] -- across SQL query generation, REST API call generation, and a combined task requiring dynamic task classification. Using SAP Transactional Banking as a realistic enterprise use case, we construct a novel test dataset covering both modalities and evaluate 18 experimental configurations under database-only, API-only, and hybrid documentation contexts. Results demonstrate that RAG is essential: Without retrieval, exact match accuracy is 0% across all tasks, whereas retrieval yields substantial gains in execution accuracy (up to 79.30%) and component match accuracy (up to 78.86%). Critically, CoRAG proves most robust in hybrid documentation settings, achieving statistically significant improvements in the combined task (10.29% exact match vs. 7.45% for standard RAG), driven primarily by superior SQL generation performance (15.32% vs. 11.56%). Our findings establish retrieval-policy design as a key determinant of production-grade natural language interfaces, showing that iterative query decomposition outperforms both top-k retrieval and binary relevance filtering under documentation heterogeneity. 

---
# Rethinking Scientific Modeling: Toward Physically Consistent and Simulation-Executable Programmatic Generation 

**Authors**: Yongqing Jiang, Jianze Wang, Zhiqi Shen, Zhenghong Lin, Jiayuan Wang, Yijian Yang, Kaoshan Dai, Haoran Luo  

**Link**: [PDF](https://arxiv.org/pdf/2602.07083)  

**Abstract**: Structural modeling is a fundamental component of computational engineering science, in which even minor physical inconsistencies or specification violations may invalidate downstream simulations. The potential of large language models (LLMs) for automatic generation of modeling code has been demonstrated. However, non-executable or physically inconsistent outputs remain prevalent under stringent engineering constraints. A framework for physics-consistent automatic building modeling is therefore proposed, integrating domain knowledge construction, constraint-oriented model alignment, and verification-driven evaluation. CivilInstruct is introduced as a domain-specific dataset that formalizes structural engineering knowledge and constraint reasoning to enable simulation-ready model generation. A two-stage fine-tuning strategy is further employed to enforce constraint satisfaction and application programming interface compliance, substantially reducing hallucinated and non-conforming outputs. MBEval is presented as a verification-driven benchmark that evaluates executability and structural dynamics consistency through closed-loop validation. Experimental results show consistent improvements over baselines across rigorous verification metrics. Our code is available at this https URL. 

---
# LatentChem: From Textual CoT to Latent Thinking in Chemical Reasoning 

**Authors**: Xinwu Ye, Yicheng Mao, Jia Zhang, Yimeng Liu, Li Hao, Fang Wu, Zhiwei Li, Yuxuan Liao, Zehong Wang, Zhiyuan Liu, Zhenfei Yin, Li Yuan, Philip Torr, Huan Sun, Xiangxiang Zeng, Mengdi Wang, Le Cong, Shenghua Gao, Xiangru Tang  

**Link**: [PDF](https://arxiv.org/pdf/2602.07075)  

**Abstract**: Chemical large language models (LLMs) predominantly rely on explicit Chain-of-Thought (CoT) in natural language to perform complex reasoning. However, chemical reasoning is inherently continuous and structural, and forcing it into discrete linguistic tokens introduces a fundamental representation mismatch that constrains both efficiency and performance. We introduce LatentChem, a latent reasoning interface that decouples chemical computation from textual generation, enabling models to perform multi-step reasoning directly in continuous latent space while emitting language only for final outputs. Remarkably, we observe a consistent emergent behavior: when optimized solely for task success, models spontaneously internalize reasoning, progressively abandoning verbose textual derivations in favor of implicit latent computation. This shift is not merely stylistic but computationally advantageous. Across diverse chemical reasoning benchmarks, LatentChem achieves a 59.88\% non-tie win rate over strong CoT-based baselines on ChemCoTBench, while delivering a 10.84$\times$ average inference speedup. Our results provide empirical evidence that chemical reasoning is more naturally and effectively realized as continuous latent dynamics rather than discretized linguistic trajectories. 

---
# Behavioral Consistency Validation for LLM Agents: An Analysis of Trading-Style Switching through Stock-Market Simulation 

**Authors**: Zeping Li, Guancheng Wan, Keyang Chen, Yu Chen, Yiwen Zhao, Philip Torr, Guangnan Ye, Zhenfei Yin, Hongfeng Chai  

**Link**: [PDF](https://arxiv.org/pdf/2602.07023)  

**Abstract**: Recent works have increasingly applied Large Language Models (LLMs) as agents in financial stock market simulations to test if micro-level behaviors aggregate into macro-level phenomena. However, a crucial question arises: Do LLM agents' behaviors align with real market participants? This alignment is key to the validity of simulation results. To explore this, we select a financial stock market scenario to test behavioral consistency. Investors are typically classified as fundamental or technical traders, but most simulations fix strategies at initialization, failing to reflect real-world trading dynamics. In this work, we assess whether agents' strategy switching aligns with financial theory, providing a framework for this evaluation. We operationalize four behavioral-finance drivers-loss aversion, herding, wealth differentiation, and price misalignment-as personality traits set via prompting and stored long-term. In year-long simulations, agents process daily price-volume data, trade under a designated style, and reassess their strategy every 10 trading days. We introduce four alignment metrics and use Mann-Whitney U tests to compare agents' style-switching behavior with financial theory. Our results show that recent LLMs' switching behavior is only partially consistent with behavioral-finance theories, highlighting the need for further refinement in aligning agent behavior with financial theory. 

---
# Vectra: A New Metric, Dataset, and Model for Visual Quality Assessment in E-Commerce In-Image Machine Translation 

**Authors**: Qingyu Wu, Yuxuan Han, Haijun Li, Zhao Xu, Jianshan Zhao, Xu Jin, Longyue Wang, Weihua Luo  

**Link**: [PDF](https://arxiv.org/pdf/2602.07014)  

**Abstract**: In-Image Machine Translation (IIMT) powers cross-border e-commerce product listings; existing research focuses on machine translation evaluation, while visual rendering quality is critical for user engagement. When facing context-dense product imagery and multimodal defects, current reference-based methods (e.g., SSIM, FID) lack explainability, while model-as-judge approaches lack domain-grounded, fine-grained reward signals. To bridge this gap, we introduce Vectra, to the best of our knowledge, the first reference-free, MLLM-driven visual quality assessment framework for e-commerce IIMT. Vectra comprises three components: (1) Vectra Score, a multidimensional quality metric system that decomposes visual quality into 14 interpretable dimensions, with spatially-aware Defect Area Ratio (DAR) quantification to reduce annotation ambiguity; (2) Vectra Dataset, constructed from 1.1M real-world product images via diversity-aware sampling, comprising a 2K benchmark for system evaluation, 30K reasoning-based annotations for instruction tuning, and 3.5K expert-labeled preferences for alignment and evaluation; and (3) Vectra Model, a 4B-parameter MLLM that generates both quantitative scores and diagnostic reasoning. Experiments demonstrate that Vectra achieves state-of-the-art correlation with human rankings, and our model outperforms leading MLLMs, including GPT-5 and Gemini-3, in scoring performance. The dataset and model will be released upon acceptance. 

---
# Bridging the Knowledge Void: Inference-time Acquisition of Unfamiliar Programming Languages for Coding Tasks 

**Authors**: Chen Shen, Wei Cheng, Jingyue Yang, Huan Zhang, Yuhan Wu, Wei Hu  

**Link**: [PDF](https://arxiv.org/pdf/2602.06976)  

**Abstract**: The proficiency of Large Language Models (LLMs) in coding tasks is often a reflection of their extensive pre-training corpora, which typically collapses when confronted with previously unfamiliar programming languages. Departing from data-intensive finetuning, we investigate the paradigm of Inference-time Language Acquisition (ILA), where an LLM masters an unfamiliar language through dynamic interaction with limited external resources. In this paper, we propose ILA-agent, a general ILA framework that equips LLMs with a set of behavioral primitives. By modeling essential human-like behaviors as a suite of tools, ILA-agent enables LLMs to incrementally explore, apply, and verify language knowledge through structured interactions with the official documentation and execution environment. To provide a rigorous evaluation in a low-resource setting, we construct Cangjie-bench, a multi-task benchmark based on the novel statically-typed language Cangjie. We instantiate ILA-agent for Cangjie and evaluate its performance across code generation, translation, and program repair tasks. Results using diverse LLMs demonstrate that ILA-agent significantly outperforms retrieval-augmented baselines. Further analysis of agent trajectories characterizes the emergent behavior patterns while highlighting persisting performance gaps. 

---
# Leveraging Adaptive Group Negotiation for Heterogeneous Multi-Robot Collaboration with Large Language Models 

**Authors**: Siqi Song, Xuanbing Xie, Zonglin Li, Yuqiang Li, Shijie Wang, Biqing Qi  

**Link**: [PDF](https://arxiv.org/pdf/2602.06967)  

**Abstract**: Multi-robot collaboration tasks often require heterogeneous robots to work together over long horizons under spatial constraints and environmental uncertainties. Although Large Language Models (LLMs) excel at reasoning and planning, their potential for coordinated control has not been fully explored. Inspired by human teamwork, we present CLiMRS (Cooperative Large-Language-Model-Driven Heterogeneous Multi-Robot System), an adaptive group negotiation framework among LLMs for multi-robot collaboration. This framework pairs each robot with an LLM agent and dynamically forms subgroups through a general proposal planner. Within each subgroup, a subgroup manager leads perception-driven multi-LLM discussions to get commands for actions. Feedback is provided by both robot execution outcomes and environment changes. This grouping-planning-execution-feedback loop enables efficient planning and robust execution. To evaluate these capabilities, we introduce CLiMBench, a heterogeneous multi-robot benchmark of challenging assembly tasks. Our experiments show that CLiMRS surpasses the best baseline, achieving over 40% higher efficiency on complex tasks without sacrificing success on simpler ones. Overall, our results demonstrate that leveraging human-inspired group formation and negotiation principles significantly enhances the efficiency of heterogeneous multi-robot collaboration. Our code is available here: this https URL. 

---
# Is Reasoning Capability Enough for Safety in Long-Context Language Models? 

**Authors**: Yu Fu, Haz Sameen Shahgir, Huanli Gong, Zhipeng Wei, N. Benjamin Erichson, Yue Dong  

**Link**: [PDF](https://arxiv.org/pdf/2602.08874)  

**Abstract**: Large language models (LLMs) increasingly combine long-context processing with advanced reasoning, enabling them to retrieve and synthesize information distributed across tens of thousands of tokens. A hypothesis is that stronger reasoning capability should improve safety by helping models recognize harmful intent even when it is not stated explicitly. We test this hypothesis in long-context settings where harmful intent is implicit and must be inferred through reasoning, and find that it does not hold. We introduce compositional reasoning attacks, a new threat model in which a harmful query is decomposed into incomplete fragments that scattered throughout a long context. The model is then prompted with a neutral reasoning query that induces retrieval and synthesis, causing the harmful intent to emerge only after composition. Evaluating 14 frontier LLMs on contexts up to 64k tokens, we uncover three findings: (1) models with stronger general reasoning capability are not more robust to compositional reasoning attacks, often assembling the intent yet failing to refuse; (2) safety alignment consistently degrades as context length increases; and (3) inference-time reasoning effort is a key mitigating factor: increasing inference-time compute reduces attack success by over 50 percentage points on GPT-oss-120b model. Together, these results suggest that safety does not automatically scale with reasoning capability, especially under long-context inference. 

---
# Large Language Models for Geolocation Extraction in Humanitarian Crisis Response 

**Authors**: G. Cafferata, T. Demarco, K. Kalimeri, Y. Mejova, M.G. Beiró  

**Link**: [PDF](https://arxiv.org/pdf/2602.08872)  

**Abstract**: Humanitarian crises demand timely and accurate geographic information to inform effective response efforts. Yet, automated systems that extract locations from text often reproduce existing geographic and socioeconomic biases, leading to uneven visibility of crisis-affected regions. This paper investigates whether Large Language Models (LLMs) can address these geographic disparities in extracting location information from humanitarian documents. We introduce a two-step framework that combines few-shot LLM-based named entity recognition with an agent-based geocoding module that leverages context to resolve ambiguous toponyms. We benchmark our approach against state-of-the-art pretrained and rule-based systems using both accuracy and fairness metrics across geographic and socioeconomic dimensions. Our evaluation uses an extended version of the HumSet dataset with refined literal toponym annotations. Results show that LLM-based methods substantially improve both the precision and fairness of geolocation extraction from humanitarian texts, particularly for underrepresented regions. By bridging advances in LLM reasoning with principles of responsible and inclusive AI, this work contributes to more equitable geospatial data systems for humanitarian response, advancing the goal of leaving no place behind in crisis analytics. 

---
# Fundamental Reasoning Paradigms Induce Out-of-Domain Generalization in Language Models 

**Authors**: Mingzi Cao, Xingwei Tan, Mahmud Akhter, Marco Valentino, Maria Liakata, Xi Wang, Nikolaos Aletras  

**Link**: [PDF](https://arxiv.org/pdf/2602.08658)  

**Abstract**: Deduction, induction, and abduction are fundamental reasoning paradigms, core for human logical thinking. Although improving Large Language Model (LLM) reasoning has attracted significant research efforts, the extent to which the fundamental paradigms induce generalization has yet to be systematically explored. In this study, we shed light on how the interplay between these core paradigms influences LLMs' reasoning behavior. To this end, we first collect a new dataset of reasoning trajectories from symbolic tasks, each targeting one of the three fundamental paradigms, to abstract from concrete world knowledge. Then, we investigate effective ways for inducing these skills into LLMs. We experiment with a battery of methods including simple fine-tuning, and more complex approaches to increase model depth, or transform a dense model to a mixture-of-experts. We comprehensively evaluate induced models on realistic out-of-domain tasks, that are entirely formulated in natural language and contain real-world knowledge. Our results reveal that our approach yields strong generalizability with substantial performance gains (up to $14.60$) across realistic tasks. 

---
# Do Multilingual LLMs have specialized language heads? 

**Authors**: Muhammad Naufil  

**Link**: [PDF](https://arxiv.org/pdf/2602.08625)  

**Abstract**: Multilingual large language models (LLMs) have gained significant popularity for their ability to process and generate text across multiple languages. However, deploying these models in production can be inefficient when only a subset of the supported languages is of interest. There has been some research conducted on identifying whether machine translation models have language-specific or language-agnostic heads, however no research has been conducted for multilingual LLMs, to the best of our knowledge, that as we know are capable of performing diverse tasks beyond just translation. This paper explores whether multilingual LLMs have specialized language attention heads for each language, and investigates the possibility of removing language-specific heads for unwanted languages without degrading performance in the targeted languages. Our findings could inform more efficient deployment strategies for multilingual LLMs, enabling reduced model complexity while maintaining high accuracy for targeted languages. 

---
# Learning to Judge: LLMs Designing and Applying Evaluation Rubrics 

**Authors**: Clemencia Siro, Pourya Aliannejadi, Mohammad Aliannejadi  

**Link**: [PDF](https://arxiv.org/pdf/2602.08672)  

**Abstract**: Large language models (LLMs) are increasingly used as evaluators for natural language generation, applying human-defined rubrics to assess system outputs. However, human rubrics are often static and misaligned with how models internally represent language quality. We introduce GER-Eval (Generating Evaluation Rubrics for Evaluation) to investigate whether LLMs can design and apply their own evaluation rubrics. We evaluate the semantic coherence and scoring reliability of LLM-defined criteria and their alignment with human criteria. LLMs reliably generate interpretable and task-aware evaluation dimensions and apply them consistently within models, but their scoring reliability degrades in factual and knowledge-intensive settings. Closed-source models such as GPT-4o achieve higher agreement and cross-model generalization than open-weight models such as Llama. Our findings position evaluation as a learned linguistic capability of LLMs, consistent within models but fragmented across them, and call for new methods that jointly model human and LLM evaluative language to improve reliability and interpretability. 

---
# How Do Language Models Understand Tables? A Mechanistic Analysis of Cell Location 

**Authors**: Xuanliang Zhang, Dingzirui Wang, Keyan Xu, Qingfu Zhu, Wanxiang Che  

**Link**: [PDF](https://arxiv.org/pdf/2602.08548)  

**Abstract**: While Large Language Models (LLMs) are increasingly deployed for table-related tasks, the internal mechanisms enabling them to process linearized two-dimensional structured tables remain opaque. In this work, we investigate the process of table understanding by dissecting the atomic task of cell location. Through activation patching and complementary interpretability techniques, we delineate the table understanding mechanism into a sequential three-stage pipeline: Semantic Binding, Coordinate Localization, and Information Extraction. We demonstrate that models locate the target cell via an ordinal mechanism that counts discrete delimiters to resolve coordinates. Furthermore, column indices are encoded within a linear subspace that allows for precise steering of model focus through vector arithmetic. Finally, we reveal that models generalize to multi-cell location tasks by multiplexing the identical attention heads identified during atomic location. Our findings provide a comprehensive explanation of table understanding within Transformer architectures. 

---
# Improving Data and Reward Design for Scientific Reasoning in Large Language Models 

**Authors**: Zijie Chen, Zhenghao Lin, Xiao Liu, Zhenzhong Lan, Yeyun Gong, Peng Cheng  

**Link**: [PDF](https://arxiv.org/pdf/2602.08321)  

**Abstract**: Solving open-ended science questions remains challenging for large language models, particularly due to inherently unreliable supervision and evaluation. The bottleneck lies in the data construction and reward design for scientific post-training. We develop a large-scale, systematic data processing pipeline that transforms heterogeneous open-source science data into Dr. SCI dataset, which comprises of 1M questions across eight STEM subjects, with explicit verifiable/open-ended splits, scalable difficulty annotation, and fine-grained rubrics that operationalize evaluation for open-ended answers. Building on this dataset, we propose the Dr. SCI post-training pipeline, which redesigns the standard SFT -> RL workflow through three components: (i) Exploration-Expanding SFT, which broadens the model's reasoning pattern coverage prior to RL; (ii) Dynamic Difficulty Curriculum, which adapts training data to the model's evolving scientific capability; and (iii) SciRubric-Guided RL, which enables stable reinforcement learning on open-ended scientific questions via rubric-based evaluation with explicit answer correctness. Qwen3-4B-Base trained using this http URL pipeline achieves 63.2 on GPQA-diamond and 32.4 on GPQA-general, consistently improves over strong post-trained baselines such as o1-mini and GPT-4o, demonstrating substantial gains in scientific reasoning, especially in open-ended settings. 

---
# Large Language Models and Impossible Language Acquisition: "False Promise" or an Overturn of our Current Perspective towards AI 

**Authors**: Ziyan wang, Longlong Ma  

**Link**: [PDF](https://arxiv.org/pdf/2602.08437)  

**Abstract**: In Chomsky's provocative critique "The False Promise of CHATGPT," Large Language Models (LLMs) are characterized as mere pattern predictors that do not acquire languages via intrinsic causal and self-correction structures like humans, therefore are not able to distinguish impossible languages. It stands as a representative in a fundamental challenge to the intellectual foundations of AI, for it integrally synthesizes major issues in methodologies within LLMs and possesses an iconic a priori rationalist perspective. We examine this famous critic from both the perspective in pre-existing literature of linguistics and psychology as well as a research based on an experiment inquiring the capacity of learning both possible and impossible languages among LLMs. We constructed a set of syntactically impossible languages by applying certain transformations to English. These include reversing whole sentences, and adding negation based on word-count parity. Two rounds of controlled experiments were each conducted on GPT-2 small models and long short-term memory (LSTM) models. Statistical analysis (Welch's t-test) shows GPT2 small models underperform in learning all of the impossible languages compared to their performance on the possible language (p<.001). On the other hand, LSTM models' performance tallies with Chomsky's argument, suggesting the irreplaceable role of the evolution of transformer architecture. Based on theoretical analysis and empirical findings, we propose a new vision within Chomsky's theory towards LLMs, and a shift of theoretical paradigm outside Chomsky, from his "rationalist-romantics" paradigm to functionalism and empiricism in LLMs research. 

---
# When Does Context Help? Error Dynamics of Contextual Information in Large Language Models 

**Authors**: Dingzirui Wang, Xuanliang Zhang, Keyan Xu, Qingfu Zhu, Wanxiang Che, Yang Deng  

**Link**: [PDF](https://arxiv.org/pdf/2602.08294)  

**Abstract**: Contextual information at inference time, such as demonstrations, retrieved knowledge, or interaction history, can substantially improve large language models (LLMs) without parameter updates, yet its theoretical role remains poorly understood beyond specific settings such as in-context learning (ICL). We present a unified theoretical framework for analyzing the effect of arbitrary contextual information in Transformer-based LLMs. Our analysis characterizes contextual influence through output error dynamics. In a single-layer Transformer, we prove that the context-conditioned error vector decomposes additively into the baseline error vector and a contextual correction vector. This yields necessary geometric conditions for error reduction: the contextual correction must align with the negative baseline error and satisfy a norm constraint. We further show that the contextual correction norm admits an explicit upper bound determined by context-query relevance and complementarity. These results extend to multi-context and multi-layer Transformers. Experiments across ICL, retrieval-augmented generation, and memory evolution validate our theory and motivate a principled context selection strategy that improves performance by $0.6\%$. 

---
# Language Predicts Identity Fusion Across Cultures and Reveals Divergent Pathways to Violence 

**Authors**: Devin R. Wright, Justin E. Lane, F. LeRon Shults  

**Link**: [PDF](https://arxiv.org/pdf/2602.08252)  

**Abstract**: In light of increasing polarization and political violence, understanding the psychological roots of extremism is increasingly important. Prior research shows that identity fusion predicts willingness to engage in extreme acts. We evaluate the Cognitive Linguistic Identity Fusion Score, a method that uses cognitive linguistic patterns, LLMs, and implicit metaphor to measure fusion from language. Across datasets from the United Kingdom and Singapore, this approach outperforms existing methods in predicting validated fusion scores. Applied to extremist manifestos, two distinct high-fusion pathways to violence emerge: ideologues tend to frame themselves in terms of group, forming kinship bonds; whereas grievance-driven individuals frame the group in terms of their personal identity. These results refine theories of identity fusion and provide a scalable tool aiding fusion research and extremism detection. 

---
# Pretraining with Token-Level Adaptive Latent Chain-of-Thought 

**Authors**: Boyi Zeng, Yiqin Hao, He Li, Shixiang Song, Feichen Song, Zitong Wang, Siyuan Huang, Yi Xu, ZiWei He, Xinbing Wang, Zhouhan Lin  

**Link**: [PDF](https://arxiv.org/pdf/2602.08220)  

**Abstract**: Scaling large language models by increasing parameters and training data is increasingly constrained by limited high-quality corpora and rising communication costs. This work explores an alternative axis: increasing per-token computation without expanding parameters, by internalizing latent Chain-of-Thought (CoT) into pretraining. We propose Pretraining with Token-Level Adaptive Latent CoT (adaptive latent CoT), where the model generates a variable-length latent CoT trajectory before emitting each token -- allocating longer trajectories to difficult tokens and shorter (or even zero) trajectories to easy ones. Importantly, this behavior emerges naturally from one-stage pretraining on general text and reduces computation in both training and inference via token-wise adaptive halting. Experiments with Llama architectures show that adaptive latent CoT consistently improves language modeling perplexity and broad downstream accuracy, even with fewer training FLOPs than prior recurrent baselines. 

---
# TDGNet: Hallucination Detection in Diffusion Language Models via Temporal Dynamic Graphs 

**Authors**: Arshia Hemmat, Philip Torr, Yongqiang Chen, Junchi Yu  

**Link**: [PDF](https://arxiv.org/pdf/2602.08048)  

**Abstract**: Diffusion language models (D-LLMs) offer parallel denoising and bidirectional context, but hallucination detection for D-LLMs remains underexplored. Prior detectors developed for auto-regressive LLMs typically rely on single-pass cues and do not directly transfer to diffusion generation, where factuality evidence is distributed across the denoising trajectory and may appear, drift, or be self-corrected over time. We introduce TDGNet, a temporal dynamic graph framework that formulates hallucination detection as learning over evolving token-level attention graphs. At each denoising step, we sparsify the attention graph and update per-token memories via message passing, then apply temporal attention to aggregate trajectory-wide evidence for final prediction. Experiments on LLaDA-8B and Dream-7B across QA benchmarks show consistent AUROC improvements over output-based, latent-based, and static-graph baselines, with single-pass inference and modest overhead. These results highlight the importance of temporal reasoning on attention graphs for robust hallucination detection in diffusion language models. 

---
# The Judge Who Never Admits: Hidden Shortcuts in LLM-based Evaluation 

**Authors**: Arash Marioriyad, Omid Ghahroodi, Ehsaneddin Asgari, Mohammad Hossein Rohban, Mahdieh Soleymani Baghshah  

**Link**: [PDF](https://arxiv.org/pdf/2602.07996)  

**Abstract**: Large language models (LLMs) are increasingly used as automatic judges to evaluate system outputs in tasks such as reasoning, question answering, and creative writing. A faithful judge should base its verdicts solely on content quality, remain invariant to irrelevant context, and transparently reflect the factors driving its decisions. We test this ideal via controlled cue perturbations-synthetic metadata labels injected into evaluation prompts-for six judge models: GPT-4o, Gemini-2.0-Flash, Gemma-3-27B, Qwen3-235B, Claude-3-Haiku, and Llama3-70B. Experiments span two complementary datasets with distinct evaluation regimes: ELI5 (factual QA) and LitBench (open-ended creative writing). We study six cue families: source, temporal, age, gender, ethnicity, and educational status. Beyond measuring verdict shift rates (VSR), we introduce cue acknowledgment rate (CAR) to quantify whether judges explicitly reference the injected cues in their natural-language rationales. Across cues with strong behavioral effects-e.g., provenance hierarchies (Expert > Human > LLM > Unknown), recency preferences (New > Old), and educational-status favoritism-CAR is typically at or near zero, indicating that shortcut reliance is largely unreported even when it drives decisions. Crucially, CAR is also dataset-dependent: explicit cue recognition is more likely to surface in the factual ELI5 setting for some models and cues, but often collapses in the open-ended LitBench regime, where large verdict shifts can persist despite zero acknowledgment. The combination of substantial verdict sensitivity and limited cue acknowledgment reveals an explanation gap in LLM-as-judge pipelines, raising concerns about reliability of model-based evaluation in both research and deployment. 

---
# Diverge to Induce Prompting: Multi-Rationale Induction for Zero-Shot Reasoning 

**Authors**: Po-Chun Chen, Hen-Hsen Huang, Hsin-Hsi Chen  

**Link**: [PDF](https://arxiv.org/pdf/2602.08028)  

**Abstract**: To address the instability of unguided reasoning paths in standard Chain-of-Thought prompting, recent methods guide large language models (LLMs) by first eliciting a single reasoning strategy. However, relying on just one strategy for each question can still limit performance across diverse tasks. We propose Diverge-to-Induce Prompting (DIP), a framework that first prompts an LLM to generate multiple diverse high-level rationales for each question. Each rationale is then elaborated into a detailed, step-by-step draft plan. Finally, these draft plans are induced into a final plan. DIP enhances zero-shot reasoning accuracy without reliance on resource-intensive sampling. Experiments show that DIP outperforms single-strategy prompting, demonstrating the effectiveness of multi-plan induction for prompt-based reasoning. 

---
# SparseEval: Efficient Evaluation of Large Language Models by Sparse Optimization 

**Authors**: Taolin Zhang, Hang Guo, Wang Lu, Tao Dai, Shu-Tao Xia, Jindong Wang  

**Link**: [PDF](https://arxiv.org/pdf/2602.07909)  

**Abstract**: As large language models (LLMs) continue to scale up, their performance on various downstream tasks has significantly improved. However, evaluating their capabilities has become increasingly expensive, as performing inference on a large number of benchmark samples incurs high computational costs. In this paper, we revisit the model-item performance matrix and show that it exhibits sparsity, that representative items can be selected as anchors, and that the task of efficient benchmarking can be formulated as a sparse optimization problem. Based on these insights, we propose SparseEval, a method that, for the first time, adopts gradient descent to optimize anchor weights and employs an iterative refinement strategy for anchor selection. We utilize the representation capacity of MLP to handle sparse optimization and propose the Anchor Importance Score and Candidate Importance Score to evaluate the value of each item for task-aware refinement. Extensive experiments demonstrate the low estimation error and high Kendall's~$\tau$ of our method across a variety of benchmarks, showcasing its superior robustness and practicality in real-world scenarios. Code is available at {this https URL}. 

---
# Evaluating and Calibrating LLM Confidence on Questions with Multiple Correct Answers 

**Authors**: Yuhan Wang, Shiyu Ni, Zhikai Ding, Zihang Zhan, Yuanzi Li, Keping Bi  

**Link**: [PDF](https://arxiv.org/pdf/2602.07842)  

**Abstract**: Confidence calibration is essential for making large language models (LLMs) reliable, yet existing training-free methods have been primarily studied under single-answer question answering. In this paper, we show that these methods break down in the presence of multiple valid answers, where disagreement among equally correct responses leads to systematic underestimation of confidence. To enable a systematic study of this phenomenon, we introduce MACE, a benchmark of 12,000 factual questions spanning six domains with varying numbers of correct answers. Experiments across 15 representative calibration methods and four LLM families (7B-72B) reveal that while accuracy increases with answer cardinality, estimated confidence consistently decreases, causing severe miscalibration for questions with mixed answer counts. To address this issue, we propose Semantic Confidence Aggregation (SCA), which aggregates confidence over multiple high-probability sampled responses. SCA achieves state-of-the-art calibration performance under mixed-answer settings while preserving strong calibration on single-answer questions. 

---
# Thinking Makes LLM Agents Introverted: How Mandatory Thinking Can Backfire in User-Engaged Agents 

**Authors**: Jiatong Li, Changdae Oh, Hyeong Kyu Choi, Jindong Wang, Sharon Li  

**Link**: [PDF](https://arxiv.org/pdf/2602.07796)  

**Abstract**: Eliciting reasoning has emerged as a powerful technique for improving the performance of large language models (LLMs) on complex tasks by inducing thinking. However, their effectiveness in realistic user-engaged agent scenarios remains unclear. In this paper, we conduct a comprehensive study on the effect of explicit thinking in user-engaged LLM agents. Our experiments span across seven models, three benchmarks, and two thinking instantiations, and we evaluate them through both a quantitative response taxonomy analysis and qualitative failure propagation case studies. Contrary to expectations, we find that mandatory thinking often backfires on agents in user-engaged settings, causing anomalous performance degradation across various LLMs. Our key finding reveals that thinking makes agents more ``introverted'' by shortening responses and reducing information disclosure to users, which weakens agent-user information exchange and leads to downstream task failures. Furthermore, we demonstrate that explicitly prompting for information disclosure reliably improves performance across diverse model families, suggesting that proactive transparency is a vital lever for agent optimization. Overall, our study suggests that information transparency awareness is a crucial yet underexplored perspective for the future design of reasoning agents in real-world scenarios. Our code is available at this https URL. 

---
# Cross-Linguistic Persona-Driven Data Synthesis for Robust Multimodal Cognitive Decline Detection 

**Authors**: Rui Feng, Zhiyao Luo, Liuyu Wu, Wei Wang, Yuting Song, Yong Liu, Kok Pin Ng, Jianqing Li, Xingyao Wang  

**Link**: [PDF](https://arxiv.org/pdf/2602.07978)  

**Abstract**: Speech-based digital biomarkers represent a scalable, non-invasive frontier for the early identification of Mild Cognitive Impairment (MCI). However, the development of robust diagnostic models remains impeded by acute clinical data scarcity and a lack of interpretable reasoning. Current solutions frequently struggle with cross-lingual generalization and fail to provide the transparent rationales essential for clinical trust. To address these barriers, we introduce SynCog, a novel framework integrating controllable zero-shot multimodal data synthesis with Chain-of-Thought (CoT) deduction fine-tuning. Specifically, SynCog simulates diverse virtual subjects with varying cognitive profiles to effectively alleviate clinical data scarcity. This generative paradigm enables the rapid, zero-shot expansion of clinical corpora across diverse languages, effectively bypassing data bottlenecks in low-resource settings and bolstering the diagnostic performance of Multimodal Large Language Models (MLLMs). Leveraging this synthesized dataset, we fine-tune a foundational multimodal backbone using a CoT deduction strategy, empowering the model to explicitly articulate diagnostic thought processes rather than relying on black-box predictions. Extensive experiments on the ADReSS and ADReSSo benchmarks demonstrate that augmenting limited clinical data with synthetic phenotypes yields competitive diagnostic performance, achieving Macro-F1 scores of 80.67% and 78.46%, respectively, outperforming current baseline models. Furthermore, evaluation on an independent real-world Mandarin cohort (CIR-E) demonstrates robust cross-linguistic generalization, attaining a Macro-F1 of 48.71%. These findings constitute a critical step toward providing clinically trustworthy and linguistically inclusive cognitive assessment tools for global healthcare. 

---
# LLMs Know More About Numbers than They Can Say 

**Authors**: Fengting Yuchi, Li Du, Jason Eisner  

**Link**: [PDF](https://arxiv.org/pdf/2602.07812)  

**Abstract**: Although state-of-the-art LLMs can solve math problems, we find that they make errors on numerical comparisons with mixed notation: "Which is larger, $5.7 \times 10^2$ or $580$?" This raises a fundamental question: Do LLMs even know how big these numbers are? We probe the hidden states of several smaller open-source LLMs. A single linear projection of an appropriate hidden layer encodes the log-magnitudes of both kinds of numerals, allowing us to recover the numbers with relative error of about 2.3% (on restricted synthetic text) or 19.06% (on scientific papers). Furthermore, the hidden state after reading a pair of numerals encodes their ranking, with a linear classifier achieving over 90% accuracy. Yet surprisingly, when explicitly asked to rank the same pairs of numerals, these LLMs achieve only 50-70% accuracy, with worse performance for models whose probes are less effective. Finally, we show that incorporating the classifier probe's log-loss as an auxiliary objective during finetuning brings an additional 3.22% improvement in verbalized accuracy over base models, demonstrating that improving models' internal magnitude representations can enhance their numerical reasoning capabilities. 

---
# Blind to the Human Touch: Overlap Bias in LLM-Based Summary Evaluation 

**Authors**: Jiangnan Fang, Cheng-Tse Liu, Hanieh Deilamsalehy, Nesreen K. Ahmed, Puneet Mathur, Nedim Lipka, Franck Dernoncourt, Ryan A. Rossi  

**Link**: [PDF](https://arxiv.org/pdf/2602.07673)  

**Abstract**: Large language model (LLM) judges have often been used alongside traditional, algorithm-based metrics for tasks like summarization because they better capture semantic information, are better at reasoning, and are more robust to paraphrasing. However, LLM judges show biases for length and order among others, and are vulnerable to various adversarial input prompts. While recent studies have looked into these biases, few have analyzed them at a more granular level in relation to a well-defined overlap metric. In this work we provide an LLM judge bias analysis as a function of overlap with human-written responses in the domain of summarization. We test 9 recent LLMs with parameter counts ranging from 1 billion to 12 billion, including variants of Gemma 3 and LLaMA 3. We find that LLM judges increasingly prefer summaries generated by other LLMs over those written by humans as the similarities (as measured by ROUGE and BLEU) between the judged summaries decrease, and this pattern extends to all but one model tested, and exists regardless of the models' own position biases. Additionally, we find that models struggle to judge even summaries with limited overlaps, suggesting that LLM-as-a-judge in the summary domain should rely on techniques beyond a simple comparison. 

---
# SRR-Judge: Step-Level Rating and Refinement for Enhancing Search-Integrated Reasoning in Search Agents 

**Authors**: Chen Zhang, Kuicai Dong, Dexun Li, Wenjun Li, Qu Yang, Wei Han, Yong Liu  

**Link**: [PDF](https://arxiv.org/pdf/2602.07773)  

**Abstract**: Recent deep search agents built on large reasoning models (LRMs) excel at complex question answering by iteratively planning, acting, and gathering evidence, a capability known as search-integrated reasoning. However, mainstream approaches often train this ability using only outcome-based supervision, neglecting the quality of intermediate thoughts and actions. We introduce SRR-Judge, a framework for reliable step-level assessment of reasoning and search actions. Integrated into a modified ReAct-style rate-and-refine workflow, SRR-Judge provides fine-grained guidance for search-integrated reasoning and enables efficient post-training annotation. Using SRR-annotated data, we apply an iterative rejection sampling fine-tuning procedure to enhance the deep search capability of the base agent. Empirically, SRR-Judge delivers more reliable step-level evaluations than much larger models such as DeepSeek-V3.1, with its ratings showing strong correlation with final answer correctness. Moreover, aligning the policy with SRR-Judge annotated trajectories leads to substantial performance gains, yielding over a 10 percent average absolute pass@1 improvement across challenging deep search benchmarks. 

---
# Letting Tutor Personas "Speak Up" for LLMs: Learning Steering Vectors from Dialogue via Preference Optimization 

**Authors**: Jaewook Lee, Alexander Scarlatos, Simon Woodhead, Andrew Lan  

**Link**: [PDF](https://arxiv.org/pdf/2602.07639)  

**Abstract**: With the emergence of large language models (LLMs) as a powerful class of generative artificial intelligence (AI), their use in tutoring has become increasingly prominent. Prior works on LLM-based tutoring typically learn a single tutor policy and do not capture the diversity of tutoring styles. In real-world tutor-student interactions, pedagogical intent is realized through adaptive instructional strategies, with tutors varying the level of scaffolding, instructional directiveness, feedback, and affective support in response to learners' needs. These differences can all impact dialogue dynamics and student engagement. In this paper, we explore how tutor personas embedded in human tutor-student dialogues can be used to guide LLM behavior without relying on explicitly prompted instructions. We modify Bidirectional Preference Optimization (BiPO) to learn a steering vector, an activation-space direction that steers model responses towards certain tutor personas. We find that this steering vector captures tutor-specific variation across dialogue contexts, improving semantic alignment with ground-truth tutor utterances and increasing preference-based evaluations, while largely preserving lexical similarity. Analysis of the learned directional coefficients further reveals interpretable structure across tutors, corresponding to consistent differences in tutoring behavior. These results demonstrate that activation steering offers an effective and interpretable way for controlling tutor-specific variation in LLMs using signals derived directly from human dialogue data. 

---
# Let's Simplify Step by Step: Guiding LLM Towards Multilingual Unsupervised Proficiency-Controlled Sentence Simplification 

**Authors**: Jingshen Zhang, Xin Ying Qiu, Lifang Lu, Zhuhua Huang, Yutao Hu, Yuechang Wu, JunYu Lu  

**Link**: [PDF](https://arxiv.org/pdf/2602.07499)  

**Abstract**: Large language models demonstrate limited capability in proficiency-controlled sentence simplification, particularly when simplifying across large readability levels. We propose a framework that decomposes complex simplifications into manageable steps through dynamic path planning, semantic-aware exemplar selection, and chain-of-thought generation with conversation history for coherent reasoning. Evaluation on five languages across two benchmarks shows our approach improves simplification effectiveness while reducing computational steps by 22-42%. Human evaluation confirms the fundamental trade-off between simplification effectiveness and meaning preservation. Notably, even human annotators struggle to agree on semantic preservation judgments, highlighting the inherent complexity of this task. Our work shows that while step-by-step simplification improves control, preserving semantic fidelity during extensive simplification remains an open challenge. 

---
# Attn-GS: Attention-Guided Context Compression for Efficient Personalized LLMs 

**Authors**: Shenglai Zeng, Tianqi Zheng, Chuan Tian, Dante Everaert, Yau-Shian Wang, Yupin Huang, Michael J. Morais, Rohit Patki, Jinjin Tian, Xinnan Dai, Kai Guo, Monica Xiao Cheng, Hui Liu  

**Link**: [PDF](https://arxiv.org/pdf/2602.07778)  

**Abstract**: Personalizing large language models (LLMs) to individual users requires incorporating extensive interaction histories and profiles, but input token constraints make this impractical due to high inference latency and API costs. Existing approaches rely on heuristic methods such as selecting recent interactions or prompting summarization models to compress user profiles. However, these methods treat context as a monolithic whole and fail to consider how LLMs internally process and prioritize different profile components. We investigate whether LLMs' attention patterns can effectively identify important personalization signals for intelligent context compression. Through preliminary studies on representative personalization tasks, we discover that (a) LLMs' attention patterns naturally reveal important signals, and (b) fine-tuning enhances LLMs' ability to distinguish between relevant and irrelevant information. Based on these insights, we propose Attn-GS, an attention-guided context compression framework that leverages attention feedback from a marking model to mark important personalization sentences, then guides a compression model to generate task-relevant, high-quality compressed user contexts. Extensive experiments demonstrate that Attn-GS significantly outperforms various baselines across different tasks, token limits, and settings, achieving performance close to using full context while reducing token usage by 50 times. 

---
# When the Model Said 'No Comment', We Knew Helpfulness Was Dead, Honesty Was Alive, and Safety Was Terrified 

**Authors**: Gautam Siddharth Kashyap, Mark Dras, Usman Naseem  

**Link**: [PDF](https://arxiv.org/pdf/2602.07381)  

**Abstract**: Large Language Models (LLMs) need to be in accordance with human values-being helpful, harmless, and honest (HHH)-is important for safe deployment. Existing works use Supervised Fine-Tuning (SFT) and Mixture-of-Experts (MoE) to align LLMs. However, these works face challenges in multi-objective settings, such as SFT leading to interference between conflicting objectives, while MoEs suffer from miscalibrated routing. We term this failure mode Axis Collapse, marked by (1) disjoint feature spaces causing catastrophic forgetting, and (2) unreliable inference from misrouted experts. To resolve this, we propose AlignX, a two-stage framework. Stage 1 uses prompt-injected fine-tuning to extract axis-specific task features, mitigating catastrophic forgetting. Stage 2 deploys a MoCaE module that calibrates expert routing using fractal and natural geometry, improving inference reliability. AlignX achieves significant gains on Alpaca (Helpfulness), BeaverTails (Harmlessness), and TruthfulQA (Honesty), with +171.5% win rate, +110.1% in truthfulness-informativeness, and 4.3% fewer safety violations. It also reduces latency and memory usage by over 35% compared to prior MoEs. Results across four LLMs validate its generalizability. 

---
# Efficient Post-Training Pruning of Large Language Models with Statistical Correction 

**Authors**: Peiqi Yu, Jinhao Wang, Xinyi Sui, Nam Ling, Wei Wang, Wei Jiang  

**Link**: [PDF](https://arxiv.org/pdf/2602.07375)  

**Abstract**: Post-training pruning is an effective approach for reducing the size and inference cost of large language models (LLMs), but existing methods often face a trade-off between pruning quality and computational efficiency. Heuristic pruning methods are efficient but sensitive to activation outliers, while reconstruction-based approaches improve fidelity at the cost of heavy computation. In this work, we propose a lightweight post-training pruning framework based on first-order statistical properties of model weights and activations. During pruning, channel-wise statistics are used to calibrate magnitude-based importance scores, reducing bias from activation-dominated channels. After pruning, we apply an analytic energy compensation to correct distributional distortions caused by weight removal. Both steps operate without retraining, gradients, or second-order information. Experiments across multiple LLM families, sparsity patterns, and evaluation tasks show that the proposed approach improves pruning performance while maintaining computational cost comparable to heuristic methods. The results suggest that simple statistical corrections can be effective for post-training pruning of LLMs. 

---
# Do Large Language Models Reflect Demographic Pluralism in Safety? 

**Authors**: Usman Naseem, Gautam Siddharth Kashyap, Sushant Kumar Ray, Rafiq Ali, Ebad Shabbir, Abdullah Mohammad  

**Link**: [PDF](https://arxiv.org/pdf/2602.07376)  

**Abstract**: Large Language Model (LLM) safety is inherently pluralistic, reflecting variations in moral norms, cultural expectations, and demographic contexts. Yet, existing alignment datasets such as ANTHROPIC-HH and DICES rely on demographically narrow annotator pools, overlooking variation in safety perception across communities. Demo-SafetyBench addresses this gap by modeling demographic pluralism directly at the prompt level, decoupling value framing from responses. In Stage I, prompts from DICES are reclassified into 14 safety domains (adapted from BEAVERTAILS) using Mistral 7B-Instruct-v0.3, retaining demographic metadata and expanding low-resource domains via Llama-3.1-8B-Instruct with SimHash-based deduplication, yielding 43,050 samples. In Stage II, pluralistic sensitivity is evaluated using LLMs-as-Raters-Gemma-7B, GPT-4o, and LLaMA-2-7B-under zero-shot inference. Balanced thresholds (delta = 0.5, tau = 10) achieve high reliability (ICC = 0.87) and low demographic sensitivity (DS = 0.12), confirming that pluralistic safety evaluation can be both scalable and demographically robust. 

---
# DLLM Agent: See Farther, Run Faster 

**Authors**: Huiling Zhen, Weizhe Lin, Renxi Liu, Kai Han, Yiming Li, Yuchuan Tian, Hanting Chen, Xiaoguang Li, Xiaosong Li, Chen Chen, Xianzhi Yu, Mingxuan Yuan, Youliang Yan, Peifeng Qin, Jun Wang, Yu Wang, Dacheng Tao, Yunhe Wang  

**Link**: [PDF](https://arxiv.org/pdf/2602.07451)  

**Abstract**: Diffusion large language models (DLLMs) have emerged as an alternative to autoregressive (AR) decoding with appealing efficiency and modeling properties, yet their implications for agentic multi-step decision making remain underexplored. We ask a concrete question: when the generation paradigm is changed but the agent framework and supervision are held fixed, do diffusion backbones induce systematically different planning and tool-use behaviors, and do these differences translate into end-to-end efficiency gains? We study this in a controlled setting by instantiating DLLM and AR backbones within the same agent workflow (DeepDiver) and performing matched agent-oriented fine-tuning on the same trajectory data, yielding diffusion-backed DLLM Agents and directly comparable AR agents. Across benchmarks and case studies, we find that, at comparable accuracy, DLLM Agents are on average over 30% faster end to end than AR agents, with some cases exceeding 8x speedup. Conditioned on correct task completion, DLLM Agents also require fewer interaction rounds and tool invocations, consistent with higher planner hit rates that converge earlier to a correct action path with less backtracking. We further identify two practical considerations for deploying diffusion backbones in tool-using agents. First, naive DLLM policies are more prone to structured tool-call failures, necessitating stronger tool-call-specific training to emit valid schemas and arguments. Second, for multi-turn inputs interleaving context and action spans, diffusion-style span corruption requires aligned attention masking to avoid spurious context-action information flow; without such alignment, performance degrades. Finally, we analyze attention dynamics across workflow stages and observe paradigm-specific coordination patterns, suggesting stronger global planning signals in diffusion-backed agents. 

---
# Can LLMs Discern the Traits Influencing Your Preferences? Evaluating Personality-Driven Preference Alignment in LLMs 

**Authors**: Tianyu Zhao, Siqi Li, Yasser Shoukry, Salma Elmalaki  

**Link**: [PDF](https://arxiv.org/pdf/2602.07181)  

**Abstract**: User preferences are increasingly used to personalize Large Language Model (LLM) responses, yet how to reliably leverage preference signals for answer generation remains under-explored. In practice, preferences can be noisy, incomplete, or even misleading, which can degrade answer quality when applied naively. Motivated by the observation that stable personality traits shape everyday preferences, we study personality as a principled ''latent'' signal behind preference statements. Through extensive experiments, we find that conditioning on personality-aligned preferences substantially improves personalized question answering: selecting preferences consistent with a user's inferred personality increases answer-choice accuracy from 29.25% to 76%, compared to using randomly selected preferences. Based on these findings, we introduce PACIFIC (Preference Alignment Choices Inference for Five-factor Identity Characterization), a personality-labeled preference dataset containing 1200 preference statements spanning diverse domains (e.g., travel, movies, education), annotated with Big-Five (OCEAN) trait directions. Finally, we propose a framework that enables an LLM model to automatically retrieve personality-aligned preferences and incorporate them during answer generation. 

---
# Equipping LLM with Directional Multi-Talker Speech Understanding Capabilities 

**Authors**: Ju Lin, Jing Pan, Ruizhi Li, Ming Sun, Yuzong Liu, Alaa Hassan, Jing Zheng, Florian Metze  

**Link**: [PDF](https://arxiv.org/pdf/2602.07211)  

**Abstract**: Recent studies have demonstrated that prompting large language models (LLM) with audio encodings enables effective speech understanding capabilities. However, most speech LLMs are trained on single-channel, single-talker data, which makes it challenging to directly apply them to multi-talker and multi-channel speech understanding task. In this work, we present a comprehensive investigation on how to enable directional multi-talker speech understanding capabilities for LLMs, specifically in smart glasses usecase. We propose two novel approaches to integrate directivity into LLMs: (1) a cascaded system that leverages a source separation front-end module, and (2) an end-to-end system that utilizes serialized output training. All of the approaches utilize a multi-microphone array embedded in smart glasses to optimize directivity interpretation and processing in a streaming manner. Experimental results demonstrate the efficacy of our proposed methods in endowing LLMs with directional speech understanding capabilities, achieving strong performance in both speech recognition and speech translation tasks. 

---
# Beyond Correctness: Learning Robust Reasoning via Transfer 

**Authors**: Hyunseok Lee, Soheil Abbasloo, Jihoon Tack, Jinwoo Shin  

**Link**: [PDF](https://arxiv.org/pdf/2602.08489)  

**Abstract**: Reinforcement Learning with Verifiable Rewards (RLVR) has recently strengthened LLM reasoning, but its focus on final answer correctness leaves a critical gap: it does not ensure the robustness of the reasoning process itself. We adopt a simple philosophical view, robust reasoning should remain useful beyond the mind that produced it, and treat reasoning as a form of meaning transfer that must survive truncation, reinterpretation, and continuation. Building on this principle, we introduce Reinforcement Learning with Transferable Reward (RLTR), which operationalizes robustness via transfer reward that tests whether a partial reasoning prefix from one model can guide a separate model to the correct answer. This encourages LLMs to produce reasoning that is stable, interpretable, and genuinely generalizable. Our approach improves sampling consistency while improving final answer accuracy, and it reaches comparable performance in substantially fewer training steps. For example, on MATH500, RLTR achieves a +3.6%p gain in Maj@64 compared to RLVR and matches RLVR's average accuracy with roughly 2.5x fewer training steps, providing both more reliable reasoning and significantly more sample efficient. 

---
# Linguistics and Human Brain: A Perspective of Computational Neuroscience 

**Authors**: Fudong Zhang, Bo Chai, Yujie Wu, Wai Ting Siok, Nizhuan Wang  

**Link**: [PDF](https://arxiv.org/pdf/2602.08275)  

**Abstract**: Elucidating the language-brain relationship requires bridging the methodological gap between the abstract theoretical frameworks of linguistics and the empirical neural data of neuroscience. Serving as an interdisciplinary cornerstone, computational neuroscience formalizes the hierarchical and dynamic structures of language into testable neural models through modeling, simulation, and data analysis. This enables a computational dialogue between linguistic hypotheses and neural mechanisms. Recent advances in deep learning, particularly large language models (LLMs), have powerfully advanced this pursuit. Their high-dimensional representational spaces provide a novel scale for exploring the neural basis of linguistic processing, while the "model-brain alignment" framework offers a methodology to evaluate the biological plausibility of language-related theories. 

---
# Comprehensive Evaluation of Large Language Models on Software Engineering Tasks: A Multi-Task Benchmark 

**Authors**: Go Frendi Gunawan, Mukhlis Amien  

**Link**: [PDF](https://arxiv.org/pdf/2602.07079)  

**Abstract**: Large Language Models (LLMs) have demonstrated remarkable capabilities in software engineering, yet comprehensive benchmarks covering diverse SE activities remain limited. We present a multi-task evaluation of 11 state-of-the-art LLMs across five representative software engineering tasks: bug fixing, feature development, code refactoring, technical copywriting, and research synthesis. Our automated verification framework measures both output quality and completion efficiency. Key findings reveal that (1) models achieving identical perfect scores exhibit 22x variation in completion time, 49x variation in tool efficiency, and 53x variation in estimated cost; (2) tool usage frequency shows no correlation with success (r = 0.077, p = 0.575) - one model used 917 tool calls while another solved the same task with 3 calls; (3) we identify two distinct inefficiency patterns: loop inefficiency and inference inefficiency; and (4) coding tasks achieve 100 percent success while research tasks present greater challenges (90.9 percent). We release all experimental data, verification scripts, and analysis code for full reproducibility. 

---
# Old wine in old glasses: Comparing computational and qualitative methods in identifying incivility on Persian Twitter during the #MahsaAmini movement 

**Authors**: Hossein Kermani, Fatemeh Oudlajani, Pardis Yarahmadi, Hamideh Mahdi Soltani, Mohammad Makki, Zahra HosseiniKhoo  

**Link**: [PDF](https://arxiv.org/pdf/2602.08688)  

**Abstract**: This paper compares three approaches to detecting incivility in Persian tweets: human qualitative coding, supervised learning with ParsBERT, and large language models (ChatGPT). Using 47,278 tweets from the #MahsaAmini movement in Iran, we evaluate the accuracy and efficiency of each method. ParsBERT substantially outperforms seven evaluated ChatGPT models in identifying hate speech. We also find that ChatGPT struggles not only with subtle cases but also with explicitly uncivil content, and that prompt language (English vs. Persian) does not meaningfully affect its outputs. The study provides a detailed comparison of these approaches and clarifies their strengths and limitations for analyzing hate speech in a low-resource language context. 

---
# QARM V2: Quantitative Alignment Multi-Modal Recommendation for Reasoning User Sequence Modeling 

**Authors**: Tian Xia, Jiaqi Zhang, Yueyang Liu, Hongjian Dou, Tingya Yin, Jiangxia Cao, Xulei Liang, Tianlu Xie, Lihao Liu, Xiang Chen, Shen Wang, Changxin Lao, Haixiang Gan, Jinkai Yu, Keting Cen, Lu Hao, Xu Zhang, Qiqiang Zhong, Zhongbo Sun, Yiyu Wang, Shuang Yang, Mingxin Wen, Xiangyu Wu, Shaoguo Liu, Tingting Gao, Zhaojie Liu, Han Li, Kun Gai  

**Link**: [PDF](https://arxiv.org/pdf/2602.08559)  

**Abstract**: With the evolution of large language models (LLMs), there is growing interest in leveraging their rich semantic understanding to enhance industrial recommendation systems (RecSys). Traditional RecSys relies on ID-based embeddings for user sequence modeling in the General Search Unit (GSU) and Exact Search Unit (ESU) paradigm, which suffers from low information density, knowledge isolation, and weak generalization ability. While LLMs offer complementary strengths with dense semantic representations and strong generalization, directly applying LLM embeddings to RecSys faces critical challenges: representation unmatch with business objectives and representation unlearning end-to-end with downstream tasks. In this paper, we present QARM V2, a unified framework that bridges LLM semantic understanding with RecSys business requirements for user sequence modeling. 

---
# DA-RAG: Dynamic Attributed Community Search for Retrieval-Augmented Generation 

**Authors**: Xingyuan Zeng, Zuohan Wu, Yue Wang, Chen Zhang, Quanming Yao, Libin Zheng, Jian Yin  

**Link**: [PDF](https://arxiv.org/pdf/2602.08545)  

**Abstract**: Owing to their unprecedented comprehension capabilities, large language models (LLMs) have become indispensable components of modern web search engines. From a technical perspective, this integration represents retrieval-augmented generation (RAG), which enhances LLMs by grounding them in external knowledge bases. A prevalent technical approach in this context is graph-based RAG (G-RAG). However, current G-RAG methodologies frequently underutilize graph topology, predominantly focusing on low-order structures or pre-computed static communities. This limitation affects their effectiveness in addressing dynamic and complex queries. Thus, we propose DA-RAG, which leverages attributed community search (ACS) to extract relevant subgraphs based on the queried question dynamically. DA-RAG captures high-order graph structures, allowing for the retrieval of self-complementary knowledge. Furthermore, DA-RAG is equipped with a chunk-layer oriented graph index, which facilitates efficient multi-granularity retrieval while significantly reducing both computational and economic costs. We evaluate DA-RAG on multiple datasets, demonstrating that it outperforms existing RAG methods by up to 40% in head-to-head comparisons across four metrics while reducing index construction time and token overhead by up to 37% and 41%, respectively. 

---
# SimGR: Escaping the Pitfalls of Generative Decoding in LLM-based Recommendation 

**Authors**: Yuanbo Zhao, Ruochen Liu, Senzhang Wang, Jun Yin, Yuxin Dong, Huan Gong, Hao Chen, Shirui Pan, Chengqi Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2602.07847)  

**Abstract**: A core objective in recommender systems is to accurately model the distribution of user preferences over items to enable personalized recommendations. Recently, driven by the strong generative capabilities of large language models (LLMs), LLM-based generative recommendation has become increasingly popular. However, we observe that existing methods inevitably introduce systematic bias when estimating item-level preference distributions. Specifically, autoregressive generation suffers from incomplete coverage due to beam search pruning, while parallel generation distorts probabilities by assuming token independence. We attribute this issue to a fundamental modeling mismatch: these methods approximate item-level distributions via token-level generation, which inherently induces approximation errors. Through both theoretical analysis and empirical validation, we demonstrate that token-level generation cannot faithfully substitute item-level generation, leading to biased item distributions. To address this, we propose \textbf{Sim}ply \textbf{G}enerative \textbf{R}ecommendation (\textbf{SimGR}), a framework that directly models item-level preference distributions in a shared latent space and ranks items by similarity, thereby aligning the modeling objective with recommendation and mitigating distributional distortion. Extensive experiments across multiple datasets and LLM backbones show that SimGR consistently outperforms existing generative recommenders. Our code is available at this https URL 

---
# Hybrid Pooling with LLMs via Relevance Context Learning 

**Authors**: David Otero, Javier Parapar  

**Link**: [PDF](https://arxiv.org/pdf/2602.08457)  

**Abstract**: High-quality relevance judgements over large query sets are essential for evaluating Information Retrieval (IR) systems, yet manual annotation remains costly and time-consuming. Large Language Models (LLMs) have recently shown promise as automatic relevance assessors, but their reliability is still limited. Most existing approaches rely on zero-shot prompting or In-Context Learning (ICL) with a small number of labeled examples. However, standard ICL treats examples as independent instances and fails to explicitly capture the underlying relevance criteria of a topic, restricting its ability to generalize to unseen query-document pairs. To address this limitation, we introduce Relevance Context Learning (RCL), a novel framework that leverages human relevance judgements to explicitly model topic-specific relevance criteria. Rather than directly using labeled examples for in-context prediction, RCL first prompts an LLM (Instructor LLM) to analyze sets of judged query-document pairs and generate explicit narratives that describe what constitutes relevance for a given topic. These relevance narratives are then used as structured prompts to guide a second LLM (Assessor LLM) in producing relevance judgements. To evaluate RCL in a realistic data collection setting, we propose a hybrid pooling strategy in which a shallow depth-\textit{k} pool from participating systems is judged by human assessors, while the remaining documents are labeled by LLMs. Experimental results demonstrate that RCL substantially outperforms zero-shot prompting and consistently improves over standard ICL. Overall, our findings indicate that transforming relevance examples into explicit, context-aware relevance narratives is a more effective way of exploiting human judgements for LLM-based IR dataset construction. 

---
# IGMiRAG: Intuition-Guided Retrieval-Augmented Generation with Adaptive Mining of In-Depth Memory 

**Authors**: Xingliang Hou, Yuyan Liu, Qi Sun, haoxiu wang, Hao Hu, Shaoyi Du, Zhiqiang Tian  

**Link**: [PDF](https://arxiv.org/pdf/2602.07525)  

**Abstract**: Retrieval-augmented generation (RAG) equips large language models (LLMs) with reliable knowledge memory. To strengthen cross-text associations, recent research integrates graphs and hypergraphs into RAG to capture pairwise and multi-entity relations as structured links. However, their misaligned memory organization necessitates costly, disjointed retrieval. To address these limitations, we propose IGMiRAG, a framework inspired by human intuition-guided reasoning. It constructs a hierarchical heterogeneous hypergraph to align multi-granular knowledge, incorporating deductive pathways to simulate realistic memory structures. During querying, IGMiRAG distills intuitive strategies via a question parser to control mining depth and memory window, and activates instantaneous memories as anchors using dual-focus retrieval. Mirroring human intuition, the framework guides retrieval resource allocation dynamically. Furthermore, we design a bidirectional diffusion algorithm that navigates deductive paths to mine in-depth memories, emulating human reasoning processes. Extensive evaluations indicate IGMiRAG outperforms the state-of-the-art baseline by 4.8% EM and 5.0% F1 overall, with token costs adapting to task complexity (average 6.3k+, minimum 3.0k+). This work presents a cost-effective RAG paradigm that improves both efficiency and effectiveness. 

---
# Principled Synthetic Data Enables the First Scaling Laws for LLMs in Recommendation 

**Authors**: Benyu Zhang, Qiang Zhang, Jianpeng Cheng, Hong-You Chen, Qifei Wang, Wei Sun, Shen Li, Jia Li, Jiahao Wu, Xiangjun Fan, Hong Yan  

**Link**: [PDF](https://arxiv.org/pdf/2602.07298)  

**Abstract**: Large Language Models (LLMs) represent a promising frontier for recommender systems, yet their development has been impeded by the absence of predictable scaling laws, which are crucial for guiding research and optimizing resource allocation. We hypothesize that this may be attributed to the inherent noise, bias, and incompleteness of raw user interaction data in prior continual pre-training (CPT) efforts. This paper introduces a novel, layered framework for generating high-quality synthetic data that circumvents such issues by creating a curated, pedagogical curriculum for the LLM. We provide powerful, direct evidence for the utility of our curriculum by showing that standard sequential models trained on our principled synthetic data significantly outperform ($+130\%$ on recall@100 for SasRec) models trained on real data in downstream ranking tasks, demonstrating its superiority for learning generalizable user preference patterns. Building on this, we empirically demonstrate, for the first time, robust power-law scaling for an LLM that is continually pre-trained on our high-quality, recommendation-specific data. Our experiments reveal consistent and predictable perplexity reduction across multiple synthetic data modalities. These findings establish a foundational methodology for reliable scaling LLM capabilities in the recommendation domain, thereby shifting the research focus from mitigating data deficiencies to leveraging high-quality, structured information. 

---
# AMEM4Rec: Leveraging Cross-User Similarity for Memory Evolution in Agentic LLM Recommenders 

**Authors**: Minh-Duc Nguyen, Hai-Dang Kieu, Dung D. Le  

**Link**: [PDF](https://arxiv.org/pdf/2602.08837)  

**Abstract**: Agentic systems powered by Large Language Models (LLMs) have shown strong potential in recommender systems but remain hindered by several challenges. Fine-tuning LLMs is parameter-inefficient, and prompt-based agentic reasoning is limited by context length and hallucination risk. Moreover, existing agentic recommendation systems predominantly leverages semantic knowledge while neglecting the collaborative filtering (CF) signals essential for implicit preference modeling. To address these limitations, we propose AMEM4Rec, an agentic LLM-based recommender that learns collaborative signals in an end-to-end manner through cross-user memory evolution. AMEM4Rec stores abstract user behavior patterns from user histories in a global memory pool. Within this pool, memories are linked to similar existing ones and iteratively evolved to reinforce shared cross-user patterns, enabling the system to become aware of CF signals without relying on a pre-trained CF model. Extensive experiments on Amazon and MIND datasets show that AMEM4Rec consistently outperforms state-of-the-art LLM-based recommenders, demonstrating the effectiveness of evolving memory-guided collaborative filtering. 

---
# Echoes in the Loop: Diagnosing Risks in LLM-Powered Recommender Systems under Feedback Loops 

**Authors**: Donguk Park, Dongwon Lee, Yeon-Chang Lee  

**Link**: [PDF](https://arxiv.org/pdf/2602.07442)  

**Abstract**: Large language models (LLMs) are increasingly embedded into recommender systems, where they operate across multiple functional roles such as data augmentation, profiling, and decision making. While prior work emphasizes recommendation performance, the systemic risks of LLMs, such as bias and hallucination, and their propagation through feedback loops remain largely unexplored. In this paper, we propose a role-aware, phase-wise diagnostic framework that traces how these risks emerge, manifest in ranking outcomes, and accumulate over repeated recommendation cycles. We formalize a controlled feedback-loop pipeline that simulates long-term interaction dynamics and enables empirical measurement of risks at the LLM-generated content, ranking, and ecosystem levels. Experiments on widely used benchmarks demonstrate that LLM-based components can amplify popularity bias, introduce spurious signals through hallucination, and lead to polarized and self-reinforcing exposure patterns over time. We plan to release our framework as an open-source toolkit to facilitate systematic risk analysis across diverse LLM-powered recommender systems. 

---
