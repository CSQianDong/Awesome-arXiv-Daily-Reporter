# A Visualized Framework for Event Cooperation with Generative Agents 

**Authors**: Yuyang Tian, Shunqiang Mao, Wenchang Gao, Lanlan Qiu, Tianxing He  

**Link**: [PDF](https://arxiv.org/pdf/2509.13011)  

**Abstract**: Large Language Models (LLMs) have revolutionized the simulation of agent societies, enabling autonomous planning, memory formation, and social interactions. However, existing frameworks often overlook systematic evaluations for event organization and lack visualized integration with physically grounded environments, limiting agents' ability to navigate spaces and interact with items realistically. We develop MiniAgentPro, a visualization platform featuring an intuitive map editor for customizing environments and a simulation player with smooth animations. Based on this tool, we introduce a comprehensive test set comprising eight diverse event scenarios with basic and hard variants to assess agents' ability. Evaluations using GPT-4o demonstrate strong performance in basic settings but highlight coordination challenges in hard variants. 

---
# Reasoning with Preference Constraints: A Benchmark for Language Models in Many-to-One Matching Markets 

**Authors**: Marylou Fauchard, Florian Carichon, Margarida Carvalho, Golnoosh Farnadi  

**Link**: [PDF](https://arxiv.org/pdf/2509.13131)  

**Abstract**: Recent advances in reasoning with large language models (LLMs) have demonstrated strong performance on complex mathematical tasks, including combinatorial optimization. Techniques such as Chain-of-Thought and In-Context Learning have further enhanced this capability, making LLMs both powerful and accessible tools for a wide range of users, including non-experts. However, applying LLMs to matching problems, which require reasoning under preferential and structural constraints, remains underexplored. To address this gap, we introduce a novel benchmark of 369 instances of the College Admission Problem, a canonical example of a matching problem with preferences, to evaluate LLMs across key dimensions: feasibility, stability, and optimality. We employ this benchmark to assess the performance of several open-weight LLMs. Our results first reveal that while LLMs can satisfy certain constraints, they struggle to meet all evaluation criteria consistently. They also show that reasoning LLMs, like QwQ and GPT-oss, significantly outperform traditional models such as Llama, Qwen or Mistral, defined here as models used without any dedicated reasoning mechanisms. Moreover, we observed that LLMs reacted differently to the various prompting strategies tested, which include Chain-of-Thought, In-Context Learning and role-based prompting, with no prompt consistently offering the best performance. Finally, we report the performances from iterative prompting with auto-generated feedback and show that they are not monotonic; they can peak early and then significantly decline in later attempts. Overall, this work offers a new perspective on model reasoning performance and the effectiveness of prompting strategies in combinatorial optimization problems with preferential constraints. 

---
# Toward PDDL Planning Copilot 

**Authors**: Yarin Benyamin, Argaman Mordoch, Shahaf S. Shperberg, Roni Stern  

**Link**: [PDF](https://arxiv.org/pdf/2509.12987)  

**Abstract**: Large Language Models (LLMs) are increasingly being used as autonomous agents capable of performing complicated tasks. However, they lack the ability to perform reliable long-horizon planning on their own. This paper bridges this gap by introducing the Planning Copilot, a chatbot that integrates multiple planning tools and allows users to invoke them through instructions in natural language. The Planning Copilot leverages the Model Context Protocol (MCP), a recently developed standard for connecting LLMs with external tools and systems. This approach allows using any LLM that supports MCP without domain-specific fine-tuning. Our Planning Copilot supports common planning tasks such as checking the syntax of planning problems, selecting an appropriate planner, calling it, validating the plan it generates, and simulating their execution. We empirically evaluate the ability of our Planning Copilot to perform these tasks using three open-source LLMs. The results show that the Planning Copilot highly outperforms using the same LLMs without the planning tools. We also conducted a limited qualitative comparison of our tool against Chat GPT-5, a very recent commercial LLM. Our results shows that our Planning Copilot significantly outperforms GPT-5 despite relying on a much smaller LLM. This suggests dedicated planning tools may be an effective way to enable LLMs to perform planning tasks. 

---
# RepIt: Representing Isolated Targets to Steer Language Models 

**Authors**: Vincent Siu, Nathan W. Henry, Nicholas Crispino, Yang Liu, Dawn Song, Chenguang Wang  

**Link**: [PDF](https://arxiv.org/pdf/2509.13281)  

**Abstract**: While activation steering in large language models (LLMs) is a growing area of research, methods can often incur broader effects than desired. This motivates isolation of purer concept vectors to enable targeted interventions and understand LLM behavior at a more granular level. We present RepIt, a simple and data-efficient framework for isolating concept-specific representations. Across five frontier LLMs, RepIt enables precise interventions: it selectively suppresses refusal on targeted concepts while preserving refusal elsewhere, producing models that answer WMD-related questions while still scoring as safe on standard benchmarks. We further show that the corrective signal localizes to just 100-200 neurons and that robust target representations can be extracted from as few as a dozen examples on a single A6000. This efficiency raises a dual concern: manipulations can be performed with modest compute and data to extend to underrepresented data-scarce topics while evading existing benchmarks. By disentangling refusal vectors with RepIt, this work demonstrates that targeted interventions can counteract overgeneralization, laying the foundation for more granular control of model behavior. 

---
# Black-box Model Merging for Language-Model-as-a-Service with Massive Model Repositories 

**Authors**: Shilian Chen, Jie Zhou, Tianyu Huai, Yujiang Lu, Junsong Li, Bihao Zhan, Qianjun Pan, Yutao Yang, Xin Li, Qin Chen, Hang Yan, Liang He  

**Link**: [PDF](https://arxiv.org/pdf/2509.12951)  

**Abstract**: Model merging refers to the process of integrating multiple distinct models into a unified model that preserves and combines the strengths and capabilities of the individual models. Most existing approaches rely on task vectors to combine models, typically under the assumption that model parameters are accessible. However, for extremely large language models (LLMs) such as GPT-4, which are often provided solely as black-box services through API interfaces (Language-Model-as-a-Service), model weights are not available to end users. This presents a significant challenge, which we refer to as black-box model merging (BMM) with massive LLMs. To address this challenge, we propose a derivative-free optimization framework based on the evolutionary algorithm (Evo-Merging) that enables effective model merging using only inference-time API queries. Our method consists of two key components: (1) sparsity-based denoising, designed to identify and filter out irrelevant or redundant information across models, and (2) sign-aware scaling, which dynamically computes optimal combination weights for the relevant models based on their performance. We also provide a formal justification, along with a theoretical analysis, for our asymmetric sparsification. Extensive experimental evaluations demonstrate that our approach achieves state-of-the-art results on a range of tasks, significantly outperforming existing strong baselines. 

---
# Zero-shot Graph Reasoning via Retrieval Augmented Framework with LLMs 

**Authors**: Hanqing Li, Kiran Sheena Jyothi, Henry Liang, Sharika Mahadevan, Diego Klabjan  

**Link**: [PDF](https://arxiv.org/pdf/2509.12743)  

**Abstract**: We propose a new, training-free method, Graph Reasoning via Retrieval Augmented Framework (GRRAF), that harnesses retrieval-augmented generation (RAG) alongside the code-generation capabilities of large language models (LLMs) to address a wide range of graph reasoning tasks. In GRRAF, the target graph is stored in a graph database, and the LLM is prompted to generate executable code queries that retrieve the necessary information. This approach circumvents the limitations of existing methods that require extensive finetuning or depend on predefined algorithms, and it incorporates an error feedback loop with a time-out mechanism to ensure both correctness and efficiency. Experimental evaluations on the GraphInstruct dataset reveal that GRRAF achieves 100% accuracy on most graph reasoning tasks, including cycle detection, bipartite graph checks, shortest path computation, and maximum flow, while maintaining consistent token costs regardless of graph sizes. Imperfect but still very high performance is observed on subgraph matching. Notably, GRRAF scales effectively to large graphs with up to 10,000 nodes. 

---
# H$^2$R: Hierarchical Hindsight Reflection for Multi-Task LLM Agents 

**Authors**: Shicheng Ye, Chao Yu, Kaiqiang Ke, Chengdong Xu, Yinqi Wei  

**Link**: [PDF](https://arxiv.org/pdf/2509.12810)  

**Abstract**: Large language model (LLM)-based agents have shown strong potential in multi-task scenarios, owing to their ability to transfer knowledge across diverse tasks. However, existing approaches often treat prior experiences and knowledge as monolithic units, leading to inefficient and coarse-grained knowledge transfer. In this work, we propose a novel hierarchical memory architecture that enables fine-grained knowledge transfer by decoupling high-level planning memory from low-level execution memory. To construct and refine these hierarchical memories, we introduce Hierarchical Hindsight Reflection (H$^2$R), a mechanism that distills reusable and hierarchical knowledge from past agent-environment interactions. At test time, H$^2$R performs retrievals of high-level and low-level memories separately, allowing LLM-based agents to efficiently access and utilize task-relevant knowledge for new this http URL results across two benchmarks demonstrate that H$^2$R can improve generalization and decision-making performance, outperforming prior baselines such as Expel. 

---
# Simulating Clinical AI Assistance using Multimodal LLMs: A Case Study in Diabetic Retinopathy 

**Authors**: Nadim Barakat, William Lotter  

**Link**: [PDF](https://arxiv.org/pdf/2509.13234)  

**Abstract**: Diabetic retinopathy (DR) is a leading cause of blindness worldwide, and AI systems can expand access to fundus photography screening. Current FDA-cleared systems primarily provide binary referral outputs, where this minimal output may limit clinical trust and utility. Yet, determining the most effective output format to enhance clinician-AI performance is an empirical challenge that is difficult to assess at scale. We evaluated multimodal large language models (MLLMs) for DR detection and their ability to simulate clinical AI assistance across different output types. Two models were tested on IDRiD and Messidor-2: GPT-4o, a general-purpose MLLM, and MedGemma, an open-source medical model. Experiments included: (1) baseline evaluation, (2) simulated AI assistance with synthetic predictions, and (3) actual AI-to-AI collaboration where GPT-4o incorporated MedGemma outputs. MedGemma outperformed GPT-4o at baseline, achieving higher sensitivity and AUROC, while GPT-4o showed near-perfect specificity but low sensitivity. Both models adjusted predictions based on simulated AI inputs, but GPT-4o's performance collapsed with incorrect ones, whereas MedGemma remained more stable. In actual collaboration, GPT-4o achieved strong results when guided by MedGemma's descriptive outputs, even without direct image access (AUROC up to 0.96). These findings suggest MLLMs may improve DR screening pipelines and serve as scalable simulators for studying clinical AI assistance across varying output configurations. Open, lightweight models such as MedGemma may be especially valuable in low-resource settings, while descriptive outputs could enhance explainability and clinician trust in clinical workflows. 

---
# ECG-aBcDe: Overcoming Model Dependence, Encoding ECG into a Universal Language for Any LLM 

**Authors**: Yong Xia, Jingxuan Li, YeTeng Sun, Jiarui Bu  

**Link**: [PDF](https://arxiv.org/pdf/2509.12625)  

**Abstract**: Large Language Models (LLMs) hold significant promise for electrocardiogram (ECG) analysis, yet challenges remain regarding transferability, time-scale information learning, and interpretability. Current methods suffer from model-specific ECG encoders, hindering transfer across LLMs. Furthermore, LLMs struggle to capture crucial time-scale information inherent in ECGs due to Transformer limitations. And their black-box nature limits clinical adoption. To address these limitations, we introduce ECG-aBcDe, a novel ECG encoding method that transforms ECG signals into a universal ECG language readily interpretable by any LLM. By constructing a hybrid dataset of ECG language and natural language, ECG-aBcDe enables direct fine-tuning of pre-trained LLMs without architectural modifications, achieving "construct once, use anywhere" capability. Moreover, the bidirectional convertibility between ECG and ECG language of ECG-aBcDe allows for extracting attention heatmaps from ECG signals, significantly enhancing interpretability. Finally, ECG-aBcDe explicitly represents time-scale information, mitigating Transformer limitations. This work presents a new paradigm for integrating ECG analysis with LLMs. Compared with existing methods, our method achieves competitive performance on ROUGE-L and METEOR. Notably, it delivers significant improvements in the BLEU-4, with improvements of 2.8 times and 3.9 times in in-dataset and cross-dataset evaluations, respectively, reaching scores of 42.58 and 30.76. These results provide strong evidence for the feasibility of the new paradigm. 

---
# Analogy-Driven Financial Chain-of-Thought (AD-FCoT): A Prompting Approach for Financial Sentiment Analysis 

**Authors**: Anmol Singhal Navya Singhal  

**Link**: [PDF](https://arxiv.org/pdf/2509.12611)  

**Abstract**: Financial news sentiment analysis is crucial for anticipating market movements. With the rise of AI techniques such as Large Language Models (LLMs), which demonstrate strong text understanding capabilities, there has been renewed interest in enhancing these systems. Existing methods, however, often struggle to capture the complex economic context of news and lack transparent reasoning, which undermines their reliability. We propose Analogy-Driven Financial Chain-of-Thought (AD-FCoT), a prompting framework that integrates analogical reasoning with chain-of-thought (CoT) prompting for sentiment prediction on historical financial news. AD-FCoT guides LLMs to draw parallels between new events and relevant historical scenarios with known outcomes, embedding these analogies into a structured, step-by-step reasoning chain. To our knowledge, this is among the first approaches to explicitly combine analogical examples with CoT reasoning in finance. Operating purely through prompting, AD-FCoT requires no additional training data or fine-tuning and leverages the model's internal financial knowledge to generate rationales that mirror human analytical reasoning. Experiments on thousands of news articles show that AD-FCoT outperforms strong baselines in sentiment classification accuracy and achieves substantially higher correlation with market returns. Its generated explanations also align with domain expertise, providing interpretable insights suitable for real-world financial analysis. 

---
# Large Language Models Imitate Logical Reasoning, but at what Cost? 

**Authors**: Lachlan McGinness, Peter Baumgartner  

**Link**: [PDF](https://arxiv.org/pdf/2509.12645)  

**Abstract**: We present a longitudinal study which evaluates the reasoning capability of frontier Large Language Models over an eighteen month period. We measured the accuracy of three leading models from December 2023, September 2024 and June 2025 on true or false questions from the PrOntoQA dataset and their faithfulness to reasoning strategies provided through in-context learning. The improvement in performance from 2023 to 2024 can be attributed to hidden Chain of Thought prompting. The introduction of thinking models allowed for significant improvement in model performance between 2024 and 2025.
We then present a neuro-symbolic architecture which uses LLMs of less than 15 billion parameters to translate the problems into a standardised form. We then parse the standardised forms of the problems into a program to be solved by Z3, an SMT solver, to determine the satisfiability of the query. We report the number of prompt and completion tokens as well as the computational cost in FLOPs for open source models. The neuro-symbolic approach significantly reduces the computational cost while maintaining near perfect performance. The common approximation that the number of inference FLOPs is double the product of the active parameters and total tokens was accurate within 10\% for all experiments. 

---
# A Dimensionality-Reduced XAI Framework for Roundabout Crash Severity Insights 

**Authors**: Rohit Chakraborty, Subasish Das  

**Link**: [PDF](https://arxiv.org/pdf/2509.12524)  

**Abstract**: Roundabouts reduce severe crashes, yet risk patterns vary by conditions. This study analyzes 2017-2021 Ohio roundabout crashes using a two-step, explainable workflow. Cluster Correspondence Analysis (CCA) identifies co-occurring factors and yields four crash patterns. A tree-based severity model is then interpreted with SHAP to quantify drivers of injury within and across patterns. Results show higher severity when darkness, wet surfaces, and higher posted speeds coincide with fixed-object or angle events, and lower severity in clear, low-speed settings. Pattern-specific explanations highlight mechanisms at entries (fail-to-yield, gap acceptance), within multi-lane circulation (improper maneuvers), and during slow-downs (rear-end). The workflow links pattern discovery with case-level explanations, supporting site screening, countermeasure selection, and audit-ready reporting. The contribution to Information Systems is a practical template for usable XAI in public safety analytics. 

---
# Empowering Clinical Trial Design through AI: A Randomized Evaluation of PowerGPT 

**Authors**: Yiwen Lu, Lu Li, Dazheng Zhang, Xinyao Jian, Tingyin Wang, Siqi Chen, Yuqing Lei, Jiayi Tong, Zhaohan Xi, Haitao Chu, Chongliang Luo, Alexis Ogdie, Brian Athey, Alparslan Turan, Michael Abramoff, Joseph C Cappelleri, Hua Xu, Yun Lu, Jesse Berlin, Daniel I. Sessler, David A. Asch, Xiaoqian Jiang, Yong Chen  

**Link**: [PDF](https://arxiv.org/pdf/2509.12471)  

**Abstract**: Sample size calculations for power analysis are critical for clinical research and trial design, yet their complexity and reliance on statistical expertise create barriers for many researchers. We introduce PowerGPT, an AI-powered system integrating large language models (LLMs) with statistical engines to automate test selection and sample size estimation in trial design. In a randomized trial to evaluate its effectiveness, PowerGPT significantly improved task completion rates (99.3% vs. 88.9% for test selection, 99.3% vs. 77.8% for sample size calculation) and accuracy (94.1% vs. 55.4% in sample size estimation, p < 0.001), while reducing average completion time (4.0 vs. 9.3 minutes, p < 0.001). These gains were consistent across various statistical tests and benefited both statisticians and non-statisticians as well as bridging expertise gaps. Already under deployment across multiple institutions, PowerGPT represents a scalable AI-driven approach that enhances accessibility, efficiency, and accuracy in statistical power analysis for clinical research. 

---
# LTA-thinker: Latent Thought-Augmented Training Framework for Large Language Models on Complex Reasoning 

**Authors**: Jiaqi Wang, Binquan Ji, Haibo Luo, Yiyang Qi, Ruiting Li, Huiyan Wang, Yuantao Han, Cangyi Yang, jiaxu Zhang, Feiliang Ren  

**Link**: [PDF](https://arxiv.org/pdf/2509.12875)  

**Abstract**: Complex Reasoning in Large Language Models can be dynamically optimized using Test-Time Scaling (TTS) to mitigate Overthinking. Methods such as Coconut, SoftCoT and its variant are effective in continuous latent space inference, the core bottleneck still lies in the efficient generation and utilization of high-quality Latent Thought. Drawing from the theory of SoftCoT++ that a larger variance in the generated Latent Thought distribution more closely approximates the golden truth distribution, we propose a Latent Thought-Augmented Training Framework--LTA-Thinker, which improves distributional variance and enhances reasoning performance from two perspectives. First, LTA-Thinker constructs a Latent Thought generation architecture based on a learnable prior. This architecture aims to increase the variance distribution of generated Latent Thought Vectors in order to simplify the overall structure and raise the performance ceiling. Second, LTA-Thinker introduces a distribution-based directional optimization paradigm that jointly constrains both distribution locality and distribution scale. This mechanism improves information efficiency and computational cost through a multi-objective co-training strategy, which combines standard Supervised Fine-Tuning (SFT) loss with two novel losses: Semantic Alignment Loss, which utilizes KL divergence to ensure that the Latent Thought is highly relevant to the semantics of the question; Reasoning Focus Loss, which utilizes a contrastive learning mechanism to guide the model to focus on the most critical reasoning steps. Experiments show that LTA-thinker achieves state-of-the-art (SOTA) performance among various baselines and demonstrates a higher performance ceiling and better scaling effects. 

---
# DaSAThco: Data-Aware SAT Heuristics Combinations Optimization via Large Language Models 

**Authors**: Minyu Chen, Guoqiang Li  

**Link**: [PDF](https://arxiv.org/pdf/2509.12602)  

**Abstract**: The performance of Conflict-Driven Clause Learning solvers hinges on internal heuristics, yet the heterogeneity of SAT problems makes a single, universally optimal configuration unattainable. While prior automated methods can find specialized configurations for specific problem families, this dataset-specific approach lacks generalizability and requires costly re-optimization for new problem types. We introduce DaSAThco, a framework that addresses this challenge by learning a generalizable mapping from instance features to tailored heuristic ensembles, enabling a train-once, adapt-broadly model. Our framework uses a Large Language Model, guided by systematically defined Problem Archetypes, to generate a diverse portfolio of specialized heuristic ensembles and subsequently learns an adaptive selection mechanism to form the final mapping. Experiments show that DaSAThco achieves superior performance and, most notably, demonstrates robust out-of-domain generalization where non-adaptive methods show limitations. Our work establishes a more scalable and practical path toward automated algorithm design for complex, configurable systems. 

---
# Learn to Relax with Large Language Models: Solving Nonlinear Combinatorial Optimization Problems via Bidirectional Coevolution 

**Authors**: Beidan Liu, Zhengqiu Zhu, Chen Gao, Yong Zhao, Wei Qi, Quanjun Yin  

**Link**: [PDF](https://arxiv.org/pdf/2509.12643)  

**Abstract**: Nonlinear Combinatorial Optimization Problems (NCOPs) present a formidable computational hurdle in practice, as their nonconvex nature gives rise to multi-modal solution spaces that defy efficient optimization. Traditional constraint relaxation approaches rely heavily on expert-driven, iterative design processes that lack systematic automation and scalable adaptability. While recent Large Language Model (LLM)-based optimization methods show promise for autonomous problem-solving, they predominantly function as passive constraint validators rather than proactive strategy architects, failing to handle the sophisticated constraint interactions inherent to this http URL address these limitations, we introduce the first end-to-end \textbf{Auto}mated \textbf{C}onstraint \textbf{O}ptimization (AutoCO) method, which revolutionizes NCOPs resolution through learning to relax with this http URL, we leverage structured LLM reasoning to generate constraint relaxation strategies, which are dynamically evolving with algorithmic principles and executable code through a unified triple-representation scheme. We further establish a novel bidirectional (global-local) coevolution mechanism that synergistically integrates Evolutionary Algorithms for intensive local refinement with Monte Carlo Tree Search for systematic global strategy space exploration, ensuring optimal balance between intensification and diversification in fragmented solution spaces. Finally, comprehensive experiments on three challenging NCOP benchmarks validate AutoCO's consistent effectiveness and superior performance over the baselines. 

---
# Reasoning Models Can be Accurately Pruned Via Chain-of-Thought Reconstruction 

**Authors**: Ryan Lucas, Kayhan Behdin, Zhipeng Wang, Qingquan Song, Shao Tang, Rahul Mazumder  

**Link**: [PDF](https://arxiv.org/pdf/2509.12464)  

**Abstract**: Reasoning language models such as DeepSeek-R1 produce long chain-of-thought traces during inference time which make them costly to deploy at scale. We show that using compression techniques such as neural network pruning produces greater performance loss than in typical language modeling tasks, and in some cases can make the model slower since they cause the model to produce more thinking tokens but with worse performance. We show that this is partly due to the fact that standard LLM pruning methods often focus on input reconstruction, whereas reasoning is a decode-dominated task. We introduce a simple, drop-in fix: during pruning we jointly reconstruct activations from the input and the model's on-policy chain-of-thought traces. This "Reasoning-Aware Compression" (RAC) integrates seamlessly into existing pruning workflows such as SparseGPT, and boosts their performance significantly. Code reproducing the results in the paper can be found at: this https URL 

---
# Building Coding Agents via Entropy-Enhanced Multi-Turn Preference Optimization 

**Authors**: Jiahao Yu, Zelei Cheng, Xian Wu, Xinyu Xing  

**Link**: [PDF](https://arxiv.org/pdf/2509.12434)  

**Abstract**: Software engineering presents complex, multi-step challenges for Large Language Models (LLMs), requiring reasoning over large codebases and coordinated tool use. The difficulty of these tasks is exemplified by benchmarks like SWE-bench, where current LLMs still struggle to resolve real-world issues.
A promising approach to enhance performance is test-time scaling (TTS), but its gains are heavily dependent on the diversity of model outputs.
While standard alignment methods such as Direct Preference Optimization (DPO) and Kahneman-Tversky Optimization (KTO) are effective at aligning model outputs with human preferences, this process can come at the cost of reduced diversity, limiting the effectiveness of TTS.
Additionally, existing preference optimization algorithms are typically designed for single-turn tasks and do not fully address the complexities of multi-turn reasoning and tool integration required for interactive coding agents.
To bridge this gap, we introduce \sys, an entropy-enhanced framework that adapts existing preference optimization algorithms to the multi-turn, tool-assisted setting.
\sys augments the preference objective to explicitly preserve policy entropy and generalizes learning to optimize over multi-turn interactions rather than single-turn responses.
We validate \sys by fine-tuning a diverse suite of models from different families and sizes (up to 106B parameters).
To maximize performance gains from TTS, we further propose a hybrid best-trajectory selection scheme combining a learned verifier model with model free approaches.
On the \swebench leaderboard, our approach establishes new state-of-the-art results among open-weight models. A 30B parameter model trained with \sys ranks 1st on \lite and 4th on \verified on the open-weight leaderboard, surpassed only by models with over 10x more parameters(\eg$>$350B). 

---
# Single-stream Policy Optimization 

**Authors**: Zhongwen Xu, Zihan Ding  

**Link**: [PDF](https://arxiv.org/pdf/2509.13232)  

**Abstract**: We revisit policy-gradient optimization for Large Language Models (LLMs) from a single-stream perspective. Prevailing group-based methods like GRPO reduce variance with on-the-fly baselines but suffer from critical flaws: frequent degenerate groups erase learning signals, and synchronization barriers hinder scalability. We introduce Single-stream Policy Optimization (SPO), which eliminates these issues by design. SPO replaces per-group baselines with a persistent, KL-adaptive value tracker and normalizes advantages globally across the batch, providing a stable, low-variance learning signal for every sample. Being group-free, SPO enables higher throughput and scales effectively in long-horizon or tool-integrated settings where generation times vary. Furthermore, the persistent value tracker naturally enables an adaptive curriculum via prioritized sampling. Experiments using Qwen3-8B show that SPO converges more smoothly and attains higher accuracy than GRPO, while eliminating computation wasted on degenerate groups. Ablation studies confirm that SPO's gains stem from its principled approach to baseline estimation and advantage normalization, offering a more robust and efficient path for LLM reasoning. Across five hard math benchmarks with Qwen3 8B, SPO improves the average maj@32 by +3.4 percentage points (pp) over GRPO, driven by substantial absolute point gains on challenging datasets, including +7.3 pp on BRUMO 25, +4.4 pp on AIME 25, +3.3 pp on HMMT 25, and achieves consistent relative gain in pass@$k$ across the evaluated $k$ values. SPO's success challenges the prevailing trend of adding incidental complexity to RL algorithms, highlighting a path where fundamental principles, not architectural workarounds, drive the next wave of progress in LLM reasoning. 

---
# Metacognitive Reuse: Turning Recurring LLM Reasoning Into Concise Behaviors 

**Authors**: Aniket Didolkar, Nicolas Ballas, Sanjeev Arora, Anirudh Goyal  

**Link**: [PDF](https://arxiv.org/pdf/2509.13237)  

**Abstract**: Large language models (LLMs) now solve multi-step problems by emitting extended chains of thought. During the process, they often re-derive the same intermediate steps across problems, inflating token usage and latency. This saturation of the context window leaves less capacity for exploration. We study a simple mechanism that converts recurring reasoning fragments into concise, reusable "behaviors" (name + instruction) via the model's own metacognitive analysis of prior traces. These behaviors are stored in a "behavior handbook" which supplies them to the model in-context at inference or distills them into parameters via supervised fine-tuning. This approach achieves improved test-time reasoning across three different settings - 1) Behavior-conditioned inference: Providing the LLM relevant behaviors in-context during reasoning reduces number of reasoning tokens by up to 46% while matching or improving baseline accuracy; 2) Behavior-guided self-improvement: Without any parameter updates, the model improves its own future reasoning by leveraging behaviors from its own past problem solving attempts. This yields up to 10% higher accuracy than a naive critique-and-revise baseline; and 3) Behavior-conditioned SFT: SFT on behavior-conditioned reasoning traces is more effective at converting non-reasoning models into reasoning models as compared to vanilla SFT. Together, these results indicate that turning slow derivations into fast procedural hints enables LLMs to remember how to reason, not just what to conclude. 

---
# LLMAP: LLM-Assisted Multi-Objective Route Planning with User Preferences 

**Authors**: Liangqi Yuan, Dong-Jun Han, Christopher G. Brinton, Sabine Brunswicker  

**Link**: [PDF](https://arxiv.org/pdf/2509.12273)  

**Abstract**: The rise of large language models (LLMs) has made natural language-driven route planning an emerging research area that encompasses rich user objectives. Current research exhibits two distinct approaches: direct route planning using LLM-as-Agent and graph-based searching strategies. However, LLMs in the former approach struggle to handle extensive map data, while the latter shows limited capability in understanding natural language preferences. Additionally, a more critical challenge arises from the highly heterogeneous and unpredictable spatio-temporal distribution of users across the globe. In this paper, we introduce a novel LLM-Assisted route Planning (LLMAP) system that employs an LLM-as-Parser to comprehend natural language, identify tasks, and extract user preferences and recognize task dependencies, coupled with a Multi-Step Graph construction with iterative Search (MSGS) algorithm as the underlying solver for optimal route finding. Our multi-objective optimization approach adaptively tunes objective weights to maximize points of interest (POI) quality and task completion rate while minimizing route distance, subject to three key constraints: user time limits, POI opening hours, and task dependencies. We conduct extensive experiments using 1,000 routing prompts sampled with varying complexity across 14 countries and 27 cities worldwide. The results demonstrate that our approach achieves superior performance with guarantees across multiple constraints. 

---
# Perception Before Reasoning: Two-Stage Reinforcement Learning for Visual Reasoning in Vision-Language Models 

**Authors**: Yan Chen, Long Li, Teng Xi, Long Zeng, Jingdong Wang  

**Link**: [PDF](https://arxiv.org/pdf/2509.13031)  

**Abstract**: Reinforcement learning (RL) has proven highly effective in eliciting the reasoning capabilities of large language models (LLMs). Inspired by this success, recent studies have explored applying similar techniques to vision-language models (VLMs), aiming to enhance their reasoning performance. However, directly transplanting RL methods from LLMs to VLMs is suboptimal, as the tasks faced by VLMs are inherently more complex. Specifically, VLMs must first accurately perceive and understand visual inputs before reasoning can be effectively performed. To address this challenge, we propose a two-stage reinforcement learning framework designed to jointly enhance both the perceptual and reasoning capabilities of VLMs. To mitigate the vanishing advantage issue commonly observed in RL training, we first perform dataset-level sampling to selectively strengthen specific capabilities using distinct data sources. During training, the first stage focuses on improving the model's visual perception through coarse- and fine-grained visual understanding, while the second stage targets the enhancement of reasoning abilities. After the proposed two-stage reinforcement learning process, we obtain PeBR-R1, a vision-language model with significantly enhanced perceptual and reasoning capabilities. Experimental results on seven benchmark datasets demonstrate the effectiveness of our approach and validate the superior performance of PeBR-R1 across diverse visual reasoning tasks. 

---
# Validating Solidity Code Defects using Symbolic and Concrete Execution powered by Large Language Models 

**Authors**: Ştefan-Claudiu Susan, Andrei Arusoaie, Dorel Lucanu  

**Link**: [PDF](https://arxiv.org/pdf/2509.13023)  

**Abstract**: The high rate of false alarms from static analysis tools and Large Language Models (LLMs) complicates vulnerability detection in Solidity Smart Contracts, demanding methods that can formally or empirically prove the presence of defects. This paper introduces a novel detection pipeline that integrates custom Slither-based detectors, LLMs, Kontrol, and Forge. Our approach is designed to reliably detect defects and generate proofs.  We currently perform experiments with promising results for seven types of critical defects. We demonstrate the pipeline's efficacy by presenting our findings for three vulnerabilities -- Reentrancy, Complex Fallback, and Faulty Access Control Policies -- that are challenging for current verification solutions, which often generate false alarms or fail to detect them entirely. We highlight the potential of either symbolic or concrete execution in correctly classifying such code faults. By chaining these instruments, our method effectively validates true positives, significantly reducing the manual verification burden. Although we identify potential limitations, such as the inconsistency and the cost of LLMs, our findings establish a robust framework for combining heuristic analysis with formal verification to achieve more reliable and automated smart contract auditing. 

---
# xOffense: An AI-driven autonomous penetration testing framework with offensive knowledge-enhanced LLMs and multi agent systems 

**Authors**: Phung Duc Luong, Le Tran Gia Bao, Nguyen Vu Khai Tam, Dong Huu Nguyen Khoa, Nguyen Huu Quyen, Van-Hau Pham, Phan The Duy  

**Link**: [PDF](https://arxiv.org/pdf/2509.13021)  

**Abstract**: This work introduces xOffense, an AI-driven, multi-agent penetration testing framework that shifts the process from labor-intensive, expert-driven manual efforts to fully automated, machine-executable workflows capable of scaling seamlessly with computational infrastructure. At its core, xOffense leverages a fine-tuned, mid-scale open-source LLM (Qwen3-32B) to drive reasoning and decision-making in penetration testing. The framework assigns specialized agents to reconnaissance, vulnerability scanning, and exploitation, with an orchestration layer ensuring seamless coordination across phases. Fine-tuning on Chain-of-Thought penetration testing data further enables the model to generate precise tool commands and perform consistent multi-step reasoning. We evaluate xOffense on two rigorous benchmarks: AutoPenBench and AI-Pentest-Benchmark. The results demonstrate that xOffense consistently outperforms contemporary methods, achieving a sub-task completion rate of 79.17%, decisively surpassing leading systems such as VulnBot and PentestGPT. These findings highlight the potential of domain-adapted mid-scale LLMs, when embedded within structured multi-agent orchestration, to deliver superior, cost-efficient, and reproducible solutions for autonomous penetration testing. 

---
# Jailbreaking Large Language Models Through Content Concretization 

**Authors**: Johan Wahréus, Ahmed Hussain, Panos Papadimitratos  

**Link**: [PDF](https://arxiv.org/pdf/2509.12937)  

**Abstract**: Large Language Models (LLMs) are increasingly deployed for task automation and content generation, yet their safety mechanisms remain vulnerable to circumvention through different jailbreaking techniques. In this paper, we introduce \textit{Content Concretization} (CC), a novel jailbreaking technique that iteratively transforms abstract malicious requests into concrete, executable implementations. CC is a two-stage process: first, generating initial LLM responses using lower-tier, less constrained safety filters models, then refining them through higher-tier models that process both the preliminary output and original prompt. We evaluate our technique using 350 cybersecurity-specific prompts, demonstrating substantial improvements in jailbreak Success Rates (SRs), increasing from 7\% (no refinements) to 62\% after three refinement iterations, while maintaining a cost of 7.5\textcent~per prompt. Comparative A/B testing across nine different LLM evaluators confirms that outputs from additional refinement steps are consistently rated as more malicious and technically superior. Moreover, manual code analysis reveals that generated outputs execute with minimal modification, although optimal deployment typically requires target-specific fine-tuning. With eventual improved harmful code generation, these results highlight critical vulnerabilities in current LLM safety frameworks. 

---
# Cross-Layer Vision Smoothing: Enhancing Visual Understanding via Sustained Focus on Key Objects in Large Vision-Language Models 

**Authors**: Jianfei Zhao, Feng Zhang, Xin Sun, Lingxing Kong, Zhixing Tan, Chong Feng  

**Link**: [PDF](https://arxiv.org/pdf/2509.12897)  

**Abstract**: Large Vision-Language Models (LVLMs) can accurately locate key objects in images, yet their attention to these objects tends to be very brief. Motivated by the hypothesis that sustained focus on key objects can improve LVLMs' visual capabilities, we propose Cross-Layer Vision Smoothing (CLVS). The core idea of CLVS is to incorporate a vision memory that smooths the attention distribution across layers. Specifically, we initialize this vision memory with position-unbiased visual attention in the first layer. In subsequent layers, the model's visual attention jointly considers the vision memory from previous layers, while the memory is updated iteratively, thereby maintaining smooth attention on key objects. Given that visual understanding primarily occurs in the early and middle layers of the model, we use uncertainty as an indicator of completed visual understanding and terminate the smoothing process accordingly. Experiments on four benchmarks across three LVLMs confirm the effectiveness and generalizability of our method. CLVS achieves state-of-the-art performance on a variety of visual understanding tasks, with particularly significant improvements in relation and attribute understanding. 

---
# All Roads Lead to Rome: Graph-Based Confidence Estimation for Large Language Model Reasoning 

**Authors**: Caiqi Zhang, Chang Shu, Ehsan Shareghi, Nigel Collier  

**Link**: [PDF](https://arxiv.org/pdf/2509.12908)  

**Abstract**: Confidence estimation is essential for the reliable deployment of large language models (LLMs). Existing methods are primarily designed for factual QA tasks and often fail to generalize to reasoning tasks. To address this gap, we propose a set of training-free, graph-based confidence estimation methods tailored to reasoning tasks. Our approach models reasoning paths as directed graphs and estimates confidence by exploiting graph properties such as centrality, path convergence, and path weighting. Experiments with two LLMs on three reasoning datasets demonstrate improved confidence estimation and enhanced performance on two downstream tasks. 

---
# Conan-Embedding-v2: Training an LLM from Scratch for Text Embeddings 

**Authors**: Shiyu Li, Yang Tang, Ruijie Liu, Shi-Zhe Chen, Xi Chen  

**Link**: [PDF](https://arxiv.org/pdf/2509.12892)  

**Abstract**: Large language models (LLMs) have recently demonstrated excellent performance in text embedding tasks. Previous work usually use LoRA to fine-tune existing LLMs, which are limited by the data and training gap between LLMs and embedding models. In this work, we introduce Conan-embedding-v2, a new 1.4B-parameter LLM trained from scratch and fine-tuned as a text embedder. First, we add news data and multilingual pairs for LLM pretraining to bridge the data gap. Based on this, we propose a cross-lingual retrieval dataset that enables the LLM to better integrate embeddings across different languages. Second, whereas LLMs use a causal mask with token-level loss, embedding models use a bidirectional mask with sentence-level loss. This training gap makes full fine-tuning less effective than LoRA. We introduce a soft-masking mechanism to gradually transition between these two types of masks, enabling the model to learn more comprehensive representations. Based on this, we propose a dynamic hard negative mining method that exposes the model to more difficult negative examples throughout the training process. Being intuitive and effective, with only approximately 1.4B parameters, Conan-embedding-v2 achieves SOTA performance on both the Massive Text Embedding Benchmark (MTEB) and Chinese MTEB (May 19, 2025). 

---
# LLM-Based Approach for Enhancing Maintainability of Automotive Architectures 

**Authors**: Nenad Petrovic, Lukasz Mazur, Alois Knoll  

**Link**: [PDF](https://arxiv.org/pdf/2509.12798)  

**Abstract**: There are many bottlenecks that decrease the flexibility of automotive systems, making their long-term maintenance, as well as updates and extensions in later lifecycle phases increasingly difficult, mainly due to long re-engineering, standardization, and compliance procedures, as well as heterogeneity and numerosity of devices and underlying software components involved. In this paper, we explore the potential of Large Language Models (LLMs) when it comes to the automation of tasks and processes that aim to increase the flexibility of automotive systems. Three case studies towards achieving this goal are considered as outcomes of early-stage research: 1) updates, hardware abstraction, and compliance, 2) interface compatibility checking, and 3) architecture modification suggestions. For proof-of-concept implementation, we rely on OpenAI's GPT-4o model. 

---
# The LLM Already Knows: Estimating LLM-Perceived Question Difficulty via Hidden Representations 

**Authors**: Yubo Zhu, Dongrui Liu, Zecheng Lin, Wei Tong, Sheng Zhong, Jing Shao  

**Link**: [PDF](https://arxiv.org/pdf/2509.12886)  

**Abstract**: Estimating the difficulty of input questions as perceived by large language models (LLMs) is essential for accurate performance evaluation and adaptive inference. Existing methods typically rely on repeated response sampling, auxiliary models, or fine-tuning the target model itself, which may incur substantial computational costs or compromise generality. In this paper, we propose a novel approach for difficulty estimation that leverages only the hidden representations produced by the target LLM. We model the token-level generation process as a Markov chain and define a value function to estimate the expected output quality given any hidden state. This allows for efficient and accurate difficulty estimation based solely on the initial hidden state, without generating any output tokens. Extensive experiments across both textual and multimodal tasks demonstrate that our method consistently outperforms existing baselines in difficulty estimation. Moreover, we apply our difficulty estimates to guide adaptive reasoning strategies, including Self-Consistency, Best-of-N, and Self-Refine, achieving higher inference efficiency with fewer generated tokens. 

---
# Multi-Robot Task Planning for Multi-Object Retrieval Tasks with Distributed On-Site Knowledge via Large Language Models 

**Authors**: Kento Murata, Shoichi Hasegawa, Tomochika Ishikawa, Yoshinobu Hagiwara, Akira Taniguchi, Lotfi El Hafi, Tadahiro Taniguchi  

**Link**: [PDF](https://arxiv.org/pdf/2509.12838)  

**Abstract**: It is crucial to efficiently execute instructions such as "Find an apple and a banana" or "Get ready for a field trip," which require searching for multiple objects or understanding context-dependent commands. This study addresses the challenging problem of determining which robot should be assigned to which part of a task when each robot possesses different situational on-site knowledge-specifically, spatial concepts learned from the area designated to it by the user. We propose a task planning framework that leverages large language models (LLMs) and spatial concepts to decompose natural language instructions into subtasks and allocate them to multiple robots. We designed a novel few-shot prompting strategy that enables LLMs to infer required objects from ambiguous commands and decompose them into appropriate subtasks. In our experiments, the proposed method achieved 47/50 successful assignments, outperforming random (28/50) and commonsense-based assignment (26/50). Furthermore, we conducted qualitative evaluations using two actual mobile manipulators. The results demonstrated that our framework could handle instructions, including those involving ad hoc categories such as "Get ready for a field trip," by successfully performing task decomposition, assignment, sequential planning, and execution. 

---
# InfoGain-RAG: Boosting Retrieval-Augmented Generation via Document Information Gain-based Reranking and Filtering 

**Authors**: Zihan Wang, Zihan Liang, Zhou Shao, Yufei Ma, Huangyu Dai, Ben Chen, Lingtao Mao, Chenyi Lei, Yuqing Ding, Han Li  

**Link**: [PDF](https://arxiv.org/pdf/2509.12765)  

**Abstract**: Retrieval-Augmented Generation (RAG) has emerged as a promising approach to address key limitations of Large Language Models (LLMs), such as hallucination, outdated knowledge, and lacking reference. However, current RAG frameworks often struggle with identifying whether retrieved documents meaningfully contribute to answer generation. This shortcoming makes it difficult to filter out irrelevant or even misleading content, which notably impacts the final performance. In this paper, we propose Document Information Gain (DIG), a novel metric designed to quantify the contribution of retrieved documents to correct answer generation. DIG measures a document's value by computing the difference of LLM's generation confidence with and without the document augmented. Further, we introduce InfoGain-RAG, a framework that leverages DIG scores to train a specialized reranker, which prioritizes each retrieved document from exact distinguishing and accurate sorting perspectives. This approach can effectively filter out irrelevant documents and select the most valuable ones for better answer generation. Extensive experiments across various models and benchmarks demonstrate that InfoGain-RAG can significantly outperform existing approaches, on both single and multiple retrievers paradigm. Specifically on NaturalQA, it achieves the improvements of 17.9%, 4.5%, 12.5% in exact match accuracy against naive RAG, self-reflective RAG and modern ranking-based RAG respectively, and even an average of 15.3% increment on advanced proprietary model GPT-4o across all datasets. These results demonstrate the feasibility of InfoGain-RAG as it can offer a reliable solution for RAG in multiple applications. 

---
# GView: A Survey of Binary Forensics via Visual, Semantic, and AI-Enhanced Analysis 

**Authors**: Raul Zaharia, Dragoş Gavriluţ, Gheorghiţă Mutu  

**Link**: [PDF](https://arxiv.org/pdf/2509.13025)  

**Abstract**: Cybersecurity threats continue to become more sophisticated and diverse in their artifacts, boosting both their volume and complexity. To overcome those challenges, we present GView, an open-source forensic analysis framework with visual and AI-enhanced reasoning. It started with focus on the practical cybersecurity industry. It has evolved significantly, incorporating large language models (LLMs) to dynamically enhance reasoning and ease the forensic workflows. This paper surveys both the current state of GView with its published papers alongside those that are in the publishing process. It also includes its innovative use of logical inference through predicates and inference rules for both the analyzed documents and the user's actions for better suggestions. We highlight the extensible architecture, showcasing its potential as a bridge between the practical forensics worlds with the academic research. 

---
# Instance-level Randomization: Toward More Stable LLM Evaluations 

**Authors**: Yiyang Li, Yonghuang Wu, Ying Luo, Liangtai Sun, Zishu Qin, Lin Qiu, Xuezhi Cao, Xunliang Cai  

**Link**: [PDF](https://arxiv.org/pdf/2509.12678)  

**Abstract**: Evaluations of large language models (LLMs) suffer from instability, where small changes of random factors such as few-shot examples can lead to drastic fluctuations of scores and even model rankings. Moreover, different LLMs can have different preferences for a certain setting of random factors. As a result, using a fixed setting of random factors, which is often adopted as the paradigm of current evaluations, can lead to potential unfair comparisons between LLMs. To mitigate the volatility of evaluations, we first theoretically analyze the sources of variance induced by changes in random factors. Targeting these specific sources, we then propose the instance-level randomization (ILR) method to reduce variance and enhance fairness in model comparisons. Instead of using a fixed setting across the whole benchmark in a single experiment, we randomize all factors that affect evaluation scores for every single instance, run multiple experiments and report the averaged score. Theoretical analyses and empirical results demonstrate that ILR can reduce the variance and unfair comparisons caused by random factors, as well as achieve similar robustness level with less than half computational cost compared with previous methods. 

---
# Don't Change My View: Ideological Bias Auditing in Large Language Models 

**Authors**: Paul Kröger, Emilio Barkett  

**Link**: [PDF](https://arxiv.org/pdf/2509.12652)  

**Abstract**: As large language models (LLMs) become increasingly embedded in products used by millions, their outputs may influence individual beliefs and, cumulatively, shape public opinion. If the behavior of LLMs can be intentionally steered toward specific ideological positions, such as political or religious views, then those who control these systems could gain disproportionate influence over public discourse. Although it remains an open question whether LLMs can reliably be guided toward coherent ideological stances and whether such steering can be effectively prevented, a crucial first step is to develop methods for detecting when such steering attempts occur. In this work, we adapt a previously proposed statistical method to the new context of ideological bias auditing. Our approach carries over the model-agnostic design of the original framework, which does not require access to the internals of the language model. Instead, it identifies potential ideological steering by analyzing distributional shifts in model outputs across prompts that are thematically related to a chosen topic. This design makes the method particularly suitable for auditing proprietary black-box systems. We validate our approach through a series of experiments, demonstrating its practical applicability and its potential to support independent post hoc audits of LLM behavior. 

---
# Toward Ownership Understanding of Objects: Active Question Generation with Large Language Model and Probabilistic Generative Model 

**Authors**: Saki Hashimoto, Shoichi Hasegawa, Tomochika Ishikawa, Akira Taniguchi, Yoshinobu Hagiwara, Lotfi El Hafi, Tadahiro Taniguchi  

**Link**: [PDF](https://arxiv.org/pdf/2509.12754)  

**Abstract**: Robots operating in domestic and office environments must understand object ownership to correctly execute instructions such as ``Bring me my cup.'' However, ownership cannot be reliably inferred from visual features alone. To address this gap, we propose Active Ownership Learning (ActOwL), a framework that enables robots to actively generate and ask ownership-related questions to users. ActOwL employs a probabilistic generative model to select questions that maximize information gain, thereby acquiring ownership knowledge efficiently to improve learning efficiency. Additionally, by leveraging commonsense knowledge from Large Language Models (LLM), objects are pre-classified as either shared or owned, and only owned objects are targeted for questioning. Through experiments in a simulated home environment and a real-world laboratory setting, ActOwL achieved significantly higher ownership clustering accuracy with fewer questions than baseline methods. These findings demonstrate the effectiveness of combining active inference with LLM-guided commonsense reasoning, advancing the capability of robots to acquire ownership knowledge for practical and socially appropriate task execution. 

---
# ScaleDoc: Scaling LLM-based Predicates over Large Document Collections 

**Authors**: Hengrui Zhang, Yulong Hui, Yihao Liu, Huanchen Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2509.12610)  

**Abstract**: Predicates are foundational components in data analysis systems. However, modern workloads increasingly involve unstructured documents, which demands semantic understanding, beyond traditional value-based predicates. Given enormous documents and ad-hoc queries, while Large Language Models (LLMs) demonstrate powerful zero-shot capabilities, their high inference cost leads to unacceptable overhead. Therefore, we introduce \textsc{ScaleDoc}, a novel system that addresses this by decoupling predicate execution into an offline representation phase and an optimized online filtering phase. In the offline phase, \textsc{ScaleDoc} leverages a LLM to generate semantic representations for each document. Online, for each query, it trains a lightweight proxy model on these representations to filter the majority of documents, forwarding only the ambiguous cases to the LLM for final decision. Furthermore, \textsc{ScaleDoc} proposes two core innovations to achieve significant efficiency: (1) a contrastive-learning-based framework that trains the proxy model to generate reliable predicating decision scores; (2) an adaptive cascade mechanism that determines the effective filtering policy while meeting specific accuracy targets. Our evaluations across three datasets demonstrate that \textsc{ScaleDoc} achieves over a 2$\times$ end-to-end speedup and reduces expensive LLM invocations by up to 85\%, making large-scale semantic analysis practical and efficient. 

---
# MedFact: Benchmarking the Fact-Checking Capabilities of Large Language Models on Chinese Medical Texts 

**Authors**: Jiayi He, Yangmin Huang, Qianyun Du, Xiangying Zhou, Zhiyang He, Jiaxue Hu, Xiaodong Tao, Lixian Lai  

**Link**: [PDF](https://arxiv.org/pdf/2509.12440)  

**Abstract**: The increasing deployment of Large Language Models (LLMs) in healthcare necessitates a rigorous evaluation of their factual reliability. However, existing benchmarks are often limited by narrow domains of data, failing to capture the complexity of real-world medical information. To address this critical gap, we introduce MedFact, a new and challenging benchmark for Chinese medical fact-checking. MedFact comprises 2,116 expert-annotated instances curated from diverse real-world texts, spanning 13 medical specialties, 8 fine-grained error types, 4 writing styles, and multiple difficulty levels. Its construction employs a hybrid AI-human framework where iterative expert feedback refines an AI-driven, multi-criteria filtering process, ensuring both high data quality and difficulty. We conduct a comprehensive evaluation of 20 leading LLMs, benchmarking their performance on veracity classification and error localization against a human expert baseline. Our results reveal that while models can often determine if a text contains an error, precisely localizing it remains a substantial challenge, with even top-performing models falling short of human performance. Furthermore, our analysis uncovers a frequent ``over-criticism'' phenomenon, a tendency for models to misidentify correct information as erroneous, which is exacerbated by advanced reasoning techniques such as multi-agent collaboration and inference-time scaling. By highlighting these critical challenges for deploying LLMs in medical applications, MedFact provides a robust resource to drive the development of more factually reliable and medically aware models. 

---
# FunAudio-ASR Technical Report 

**Authors**: Keyu An, Yanni Chen, Chong Deng, Changfeng Gao, Zhifu Gao, Bo Gong, Xiangang Li, Yabin Li, Xiang Lv, Yunjie Ji, Yiheng Jiang, Bin Ma, Haoneng Luo, Chongjia Ni, Zexu Pan, Yiping Peng, Zhendong Peng, Peiyao Wang, Hao Wang, Wen Wang, Wupeng Wang, Biao Tian, Zhentao Tan, Nan Yang, Bin Yuan, Jieping Ye, Jixing Yu, Qinglin Zhang, Kun Zou, Han Zhao, Shengkui Zhao, Jingren Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2509.12508)  

**Abstract**: In recent years, automatic speech recognition (ASR) has witnessed transformative advancements driven by three complementary paradigms: data scaling, model size scaling, and deep integration with large language models (LLMs). However, LLMs are prone to hallucination, which can significantly degrade user experience in real-world ASR applications. In this paper, we present FunAudio-ASR, a large-scale, LLM-based ASR system that synergistically combines massive data, large model capacity, LLM integration, and reinforcement learning to achieve state-of-the-art performance across diverse and complex speech recognition scenarios. Moreover, FunAudio-ASR is specifically optimized for practical deployment, with enhancements in streaming capability, noise robustness, code-switching, hotword customization, and satisfying other real-world application requirements. Experimental results show that while most LLM-based ASR systems achieve strong performance on open-source benchmarks, they often underperform on real industry evaluation sets. Thanks to production-oriented optimizations, FunAudio-ASR achieves SOTA performance on real application datasets, demonstrating its effectiveness and robustness in practical settings. 

---
# MORABLES: A Benchmark for Assessing Abstract Moral Reasoning in LLMs with Fables 

**Authors**: Matteo Marcuzzo, Alessandro Zangari, Andrea Albarelli, Jose Camacho-Collados, Mohammad Taher Pilehvar  

**Link**: [PDF](https://arxiv.org/pdf/2509.12371)  

**Abstract**: As LLMs excel on standard reading comprehension benchmarks, attention is shifting toward evaluating their capacity for complex abstract reasoning and inference. Literature-based benchmarks, with their rich narrative and moral depth, provide a compelling framework for evaluating such deeper comprehension skills. Here, we present MORABLES, a human-verified benchmark built from fables and short stories drawn from historical literature. The main task is structured as multiple-choice questions targeting moral inference, with carefully crafted distractors that challenge models to go beyond shallow, extractive question answering. To further stress-test model robustness, we introduce adversarial variants designed to surface LLM vulnerabilities and shortcuts due to issues such as data contamination. Our findings show that, while larger models outperform smaller ones, they remain susceptible to adversarial manipulation and often rely on superficial patterns rather than true moral reasoning. This brittleness results in significant self-contradiction, with the best models refuting their own answers in roughly 20% of cases depending on the framing of the moral choice. Interestingly, reasoning-enhanced models fail to bridge this gap, suggesting that scale - not reasoning ability - is the primary driver of performance. 

---
# RL Fine-Tuning Heals OOD Forgetting in SFT 

**Authors**: Hangzhan Jin, Sitao Luan, Sicheng Lyu, Guillaume Rabusseau, Reihaneh Rabbany, Doina Precup, Mohammad Hamdaqa  

**Link**: [PDF](https://arxiv.org/pdf/2509.12235)  

**Abstract**: The two-stage fine-tuning paradigm of Supervised Fine-Tuning (SFT) followed by Reinforcement Learning (RL) has empirically shown better reasoning performance than one-stage SFT for the post-training of Large Language Models (LLMs). However, the evolution and mechanism behind the synergy of SFT and RL are still under-explored and inconclusive. In our study, we find the well-known claim "SFT memorizes, RL generalizes" is over-simplified, and discover that: (1) OOD performance peaks at the early stage of SFT and then declines (OOD forgetting), the best SFT checkpoint cannot be captured by training/test loss; (2) the subsequent RL stage does not generate fundamentally better OOD capability, instead it plays an \textbf{OOD restoration} role, recovering the lost reasoning ability during SFT; (3) The recovery ability has boundaries, \ie{} \textbf{if SFT trains for too short or too long, RL cannot recover the lost OOD ability;} (4) To uncover the underlying mechanisms behind the forgetting and restoration process, we employ SVD analysis on parameter matrices, manually edit them, and observe their impacts on model performance. Unlike the common belief that the shift of model capacity mainly results from the changes of singular values, we find that they are actually quite stable throughout fine-tuning. Instead, the OOD behavior strongly correlates with the \textbf{rotation of singular vectors}. Our findings re-identify the roles of SFT and RL in the two-stage fine-tuning and discover the rotation of singular vectors as the key mechanism. %reversing the rotations induced by SFT, which shows recovery from forgetting, whereas imposing the SFT parameter directions onto a RL-tuned model results in performance degradation. Code is available at this https URL 

---
# Towards Trustworthy Agentic IoEV: AI Agents for Explainable Cyberthreat Mitigation and State Analytics 

**Authors**: Meryem Malak Dif, Mouhamed Amine Bouchiha, Abdelaziz Amara Korba, Yacine Ghamri-Doudane  

**Link**: [PDF](https://arxiv.org/pdf/2509.12233)  

**Abstract**: The Internet of Electric Vehicles (IoEV) envisions a tightly coupled ecosystem of electric vehicles (EVs), charging infrastructure, and grid services, yet it remains vulnerable to cyberattacks, unreliable battery-state predictions, and opaque decision processes that erode trust and performance. To address these challenges, we introduce a novel Agentic Artificial Intelligence (AAI) framework tailored for IoEV, where specialized agents collaborate to deliver autonomous threat mitigation, robust analytics, and interpretable decision support. Specifically, we design an AAI architecture comprising dedicated agents for cyber-threat detection and response at charging stations, real-time State of Charge (SoC) estimation, and State of Health (SoH) anomaly detection, all coordinated through a shared, explainable reasoning layer; develop interpretable threat-mitigation mechanisms that proactively identify and neutralize attacks on both physical charging points and learning components; propose resilient SoC and SoH models that leverage continuous and adversarial-aware learning to produce accurate, uncertainty-aware forecasts with human-readable explanations; and implement a three-agent pipeline, where each agent uses LLM-driven reasoning and dynamic tool invocation to interpret intent, contextualize tasks, and execute formal optimizations for user-centric assistance. Finally, we validate our framework through comprehensive experiments across diverse IoEV scenarios, demonstrating significant improvements in security and prediction accuracy. All datasets, models, and code will be released publicly. 

---
# MEUV: Achieving Fine-Grained Capability Activation in Large Language Models via Mutually Exclusive Unlock Vectors 

**Authors**: Xin Tong, Zhi Lin, Jingya Wang, Meng Han, Bo Jin  

**Link**: [PDF](https://arxiv.org/pdf/2509.12221)  

**Abstract**: Large language models (LLMs) enforce safety alignment to reliably refuse malicious requests, yet the same blanket safeguards also block legitimate uses in policing, defense, and other high-stakes settings. Earlier "refusal-direction" edits can bypass those layers, but they rely on a single vector that indiscriminately unlocks all hazardous topics, offering no semantic control. We introduce Mutually Exclusive Unlock Vectors (MEUV), a lightweight framework that factorizes the monolithic refusal direction into topic-aligned, nearly orthogonal vectors, each dedicated to one sensitive capability. MEUV is learned in a single epoch with a multi-task objective that blends a differential-ablation margin, cross-topic and orthogonality penalties, and several auxiliary terms. On bilingual malicious-prompt benchmarks, MEUV achieves an attack success rate of no less than 87% on Gemma-2-2B, LLaMA-3-8B, and Qwen-7B, yet cuts cross-topic leakage by up to 90% compared with the best single-direction baseline. Vectors trained in Chinese transfer almost unchanged to English (and vice versa), suggesting a language-agnostic refusal subspace. The results show that fine-grained, topic-level capability activation is achievable with minimal utility loss, paving the way for controlled LLMs deployment in security-sensitive domains. 

---
# Efficient Cold-Start Recommendation via BPE Token-Level Embedding Initialization with LLM 

**Authors**: Yushang Zhao, Xinyue Han, Qian Leng, Qianyi Sun, Haotian Lyu, Chengrui Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2509.13179)  

**Abstract**: The cold-start issue is the challenge when we talk about recommender systems, especially in the case when we do not have the past interaction data of new users or new items. Content-based features or hybrid solutions are common as conventional solutions, but they can only work in a sparse metadata environment with shallow patterns. In this paper, the efficient cold-start recommendation strategy is presented, which is based on the sub word-level representations by applying Byte Pair Encoding (BPE) tokenization and pre-trained Large Language Model (LLM) embedding in the initialization procedure. We obtain fine-grained token-level vectors that are aligned with the BPE vocabulary as opposed to using coarse-grained sentence embeddings. Together, these token embeddings can be used as dense semantic priors on unseen entities, making immediate recommendation performance possible without user-item interaction history. Our mechanism can be compared to collaborative filtering systems and tested over benchmark datasets with stringent cold-start assumptions. Experimental findings show that the given BPE-LLM method achieves higher Recall@k, NDCG@k, and Hit Rate measurements compared to the standard baseline and displays the same capability of sufficient computational performance. Furthermore, we demonstrate that using subword-aware embeddings yields better generalizability and is more interpretable, especially within a multilingual and sparse input setting. The practical application of token-level semantic initialization as a lightweight, but nevertheless effective extension to modern recommender systems in the zero-shot setting is indicated within this work. 

---
# DiffHash: Text-Guided Targeted Attack via Diffusion Models against Deep Hashing Image Retrieval 

**Authors**: Zechao Liu, Zheng Zhou, Xiangkun Chen, Tao Liang, Dapeng Lang  

**Link**: [PDF](https://arxiv.org/pdf/2509.12824)  

**Abstract**: Deep hashing models have been widely adopted to tackle the challenges of large-scale image retrieval. However, these approaches face serious security risks due to their vulnerability to adversarial examples. Despite the increasing exploration of targeted attacks on deep hashing models, existing approaches still suffer from a lack of multimodal guidance, reliance on labeling information and dependence on pixel-level operations for attacks. To address these limitations, we proposed DiffHash, a novel diffusion-based targeted attack for deep hashing. Unlike traditional pixel-based attacks that directly modify specific pixels and lack multimodal guidance, our approach focuses on optimizing the latent representations of images, guided by text information generated by a Large Language Model (LLM) for the target image. Furthermore, we designed a multi-space hash alignment network to align the high-dimension image space and text space to the low-dimension binary hash space. During reconstruction, we also incorporated text-guided attention mechanisms to refine adversarial examples, ensuring them aligned with the target semantics while maintaining visual plausibility. Extensive experiments have demonstrated that our method outperforms state-of-the-art (SOTA) targeted attack methods, achieving better black-box transferability and offering more excellent stability across datasets. 

---
# Knowledge Graph Tokenization for Behavior-Aware Generative Next POI Recommendation 

**Authors**: Ke Sun, Mayi Xu  

**Link**: [PDF](https://arxiv.org/pdf/2509.12350)  

**Abstract**: Generative paradigm, especially powered by Large Language Models (LLMs), has emerged as a new solution to the next point-of-interest (POI) recommendation. Pioneering studies usually adopt a two-stage pipeline, starting with a tokenizer converting POIs into discrete identifiers that can be processed by LLMs, followed by POI behavior prediction tasks to instruction-tune LLM for next POI recommendation. Despite of remarkable progress, they still face two limitations: (1) existing tokenizers struggle to encode heterogeneous signals in the recommendation data, suffering from information loss issue, and (2) previous instruction-tuning tasks only focus on users' POI visit behavior while ignore other behavior types, resulting in insufficient understanding of mobility. To address these limitations, we propose KGTB (Knowledge Graph Tokenization for Behavior-aware generative next POI recommendation). Specifically, KGTB organizes the recommendation data in a knowledge graph (KG) format, of which the structure can seamlessly preserve the heterogeneous information. Then, a KG-based tokenizer is developed to quantize each node into an individual structural ID. This process is supervised by the KG's structure, thus reducing the loss of heterogeneous information. Using generated IDs, KGTB proposes multi-behavior learning that introduces multiple behavior-specific prediction tasks for LLM fine-tuning, e.g., POI, category, and region visit behaviors. Learning on these behavior tasks provides LLMs with comprehensive insights on the target POI visit behavior. Experiments on four real-world city datasets demonstrate the superior performance of KGTB. 

---
# Evaluating LLM Alignment on Personality Inference from Real-World Interview Data 

**Authors**: Jianfeng Zhu, Julina Maharjan, Xinyu Li, Karin G. Coifman, Ruoming Jin  

**Link**: [PDF](https://arxiv.org/pdf/2509.13244)  

**Abstract**: Large Language Models (LLMs) are increasingly deployed in roles requiring nuanced psychological understanding, such as emotional support agents, counselors, and decision-making assistants. However, their ability to interpret human personality traits, a critical aspect of such applications, remains unexplored, particularly in ecologically valid conversational settings. While prior work has simulated LLM "personas" using discrete Big Five labels on social media data, the alignment of LLMs with continuous, ground-truth personality assessments derived from natural interactions is largely unexamined. To address this gap, we introduce a novel benchmark comprising semi-structured interview transcripts paired with validated continuous Big Five trait scores. Using this dataset, we systematically evaluate LLM performance across three paradigms: (1) zero-shot and chain-of-thought prompting with GPT-4.1 Mini, (2) LoRA-based fine-tuning applied to both RoBERTa and Meta-LLaMA architectures, and (3) regression using static embeddings from pretrained BERT and OpenAI's text-embedding-3-small. Our results reveal that all Pearson correlations between model predictions and ground-truth personality traits remain below 0.26, highlighting the limited alignment of current LLMs with validated psychological constructs. Chain-of-thought prompting offers minimal gains over zero-shot, suggesting that personality inference relies more on latent semantic representation than explicit reasoning. These findings underscore the challenges of aligning LLMs with complex human attributes and motivate future work on trait-specific prompting, context-aware modeling, and alignment-oriented fine-tuning. 

---
# ChartGaze: Enhancing Chart Understanding in LVLMs with Eye-Tracking Guided Attention Refinement 

**Authors**: Ali Salamatian, Amirhossein Abaskohi, Wan-Cyuan Fan, Mir Rayat Imtiaz Hossain, Leonid Sigal, Giuseppe Carenini  

**Link**: [PDF](https://arxiv.org/pdf/2509.13282)  

**Abstract**: Charts are a crucial visual medium for communicating and representing information. While Large Vision-Language Models (LVLMs) have made progress on chart question answering (CQA), the task remains challenging, particularly when models attend to irrelevant regions of the chart. In this work, we present ChartGaze, a new eye-tracking dataset that captures human gaze patterns during chart reasoning tasks. Through a systematic comparison of human and model attention, we find that LVLMs often diverge from human gaze, leading to reduced interpretability and accuracy. To address this, we propose a gaze-guided attention refinement that aligns image-text attention with human fixations. Our approach improves both answer accuracy and attention alignment, yielding gains of up to 2.56 percentage points across multiple models. These results demonstrate the promise of incorporating human gaze to enhance both the reasoning quality and interpretability of chart-focused LVLMs. 

---
# LLM Hallucination Detection: A Fast Fourier Transform Method Based on Hidden Layer Temporal Signals 

**Authors**: Jinxin Li, Gang Tu, ShengYu Cheng, Junjie Hu, Jinting Wang, Rui Chen, Zhilong Zhou, Dongbo Shan  

**Link**: [PDF](https://arxiv.org/pdf/2509.13154)  

**Abstract**: Hallucination remains a critical barrier for deploying large language models (LLMs) in reliability-sensitive applications. Existing detection methods largely fall into two categories: factuality checking, which is fundamentally constrained by external knowledge coverage, and static hidden-state analysis, that fails to capture deviations in reasoning dynamics. As a result, their effectiveness and robustness remain limited. We propose HSAD (Hidden Signal Analysis-based Detection), a novel hallucination detection framework that models the temporal dynamics of hidden representations during autoregressive generation. HSAD constructs hidden-layer signals by sampling activations across layers, applies Fast Fourier Transform (FFT) to obtain frequency-domain representations, and extracts the strongest non-DC frequency component as spectral features. Furthermore, by leveraging the autoregressive nature of LLMs, HSAD identifies optimal observation points for effective and reliable detection. Across multiple benchmarks, including TruthfulQA, HSAD achieves over 10 percentage points improvement compared to prior state-of-the-art methods. By integrating reasoning-process modeling with frequency-domain analysis, HSAD establishes a new paradigm for robust hallucination detection in LLMs. 

---
# Do Natural Language Descriptions of Model Activations Convey Privileged Information? 

**Authors**: Millicent Li, Alberto Mario Ceballos Arroyo, Giordano Rogers, Naomi Saphra, Byron C. Wallace  

**Link**: [PDF](https://arxiv.org/pdf/2509.13316)  

**Abstract**: Recent interpretability methods have proposed to translate LLM internal representations into natural language descriptions using a second verbalizer LLM. This is intended to illuminate how the target model represents and operates on inputs. But do such activation verbalization approaches actually provide privileged knowledge about the internal workings of the target model, or do they merely convey information about its inputs? We critically evaluate popular verbalization methods across datasets used in prior work and find that they succeed at benchmarks without any access to target model internals, suggesting that these datasets are not ideal for evaluating verbalization methods. We then run controlled experiments which reveal that verbalizations often reflect the parametric knowledge of the verbalizer LLM which generated them, rather than the activations of the target LLM being decoded. Taken together, our results indicate a need for targeted benchmarks and experimental controls to rigorously assess whether verbalization methods provide meaningful insights into the operations of LLMs. 

---
# The Few-shot Dilemma: Over-prompting Large Language Models 

**Authors**: Yongjian Tang, Doruk Tuncel, Christian Koerner, Thomas Runkler  

**Link**: [PDF](https://arxiv.org/pdf/2509.13196)  

**Abstract**: Over-prompting, a phenomenon where excessive examples in prompts lead to diminished performance in Large Language Models (LLMs), challenges the conventional wisdom about in-context few-shot learning. To investigate this few-shot dilemma, we outline a prompting framework that leverages three standard few-shot selection methods - random sampling, semantic embedding, and TF-IDF vectors - and evaluate these methods across multiple LLMs, including GPT-4o, GPT-3.5-turbo, DeepSeek-V3, Gemma-3, LLaMA-3.1, LLaMA-3.2, and Mistral. Our experimental results reveal that incorporating excessive domain-specific examples into prompts can paradoxically degrade performance in certain LLMs, which contradicts the prior empirical conclusion that more relevant few-shot examples universally benefit LLMs. Given the trend of LLM-assisted software engineering and requirement analysis, we experiment with two real-world software requirement classification datasets. By gradually increasing the number of TF-IDF-selected and stratified few-shot examples, we identify their optimal quantity for each LLM. This combined approach achieves superior performance with fewer examples, avoiding the over-prompting problem, thus surpassing the state-of-the-art by 1% in classifying functional and non-functional requirements. 

---
# SitLLM: Large Language Models for Sitting Posture Health Understanding via Pressure Sensor Data 

**Authors**: Jian Gao, Fufangchen Zhao, Yiyang Zhang, Danfeng Yan  

**Link**: [PDF](https://arxiv.org/pdf/2509.12994)  

**Abstract**: Poor sitting posture is a critical yet often overlooked factor contributing to long-term musculoskeletal disorders and physiological dysfunctions. Existing sitting posture monitoring systems, although leveraging visual, IMU, or pressure-based modalities, often suffer from coarse-grained recognition and lack the semantic expressiveness necessary for personalized feedback. In this paper, we propose \textbf{SitLLM}, a lightweight multimodal framework that integrates flexible pressure sensing with large language models (LLMs) to enable fine-grained posture understanding and personalized health-oriented response generation. SitLLM comprises three key components: (1) a \textit{Gaussian-Robust Sensor Embedding Module} that partitions pressure maps into spatial patches and injects local noise perturbations for robust feature extraction; (2) a \textit{Prompt-Driven Cross-Modal Alignment Module} that reprograms sensor embeddings into the LLM's semantic space via multi-head cross-attention using the pre-trained vocabulary embeddings; and (3) a \textit{Multi-Context Prompt Module} that fuses feature-level, structure-level, statistical-level, and semantic-level contextual information to guide instruction comprehension. 

---
# Empowering LLMs with Parameterized Skills for Adversarial Long-Horizon Planning 

**Authors**: Sijia Cui, Shuai Xu, Aiyao He, Yanna Wang, Bo Xu  

**Link**: [PDF](https://arxiv.org/pdf/2509.13127)  

**Abstract**: Recent advancements in Large Language Models(LLMs) have led to the development of LLM-based AI agents. A key challenge is the creation of agents that can effectively ground themselves in complex, adversarial long-horizon environments. Existing methods mainly focus on (1) using LLMs as policies to interact with the environment through generating low-level feasible actions, and (2) utilizing LLMs to generate high-level tasks or language guides to stimulate action generation. However, the former struggles to generate reliable actions, while the latter relies heavily on expert experience to translate high-level tasks into specific action sequences. To address these challenges, we introduce the Plan with Language, Act with Parameter (PLAP) planning framework that facilitates the grounding of LLM-based agents in long-horizon environments. The PLAP method comprises three key components: (1) a skill library containing environment-specific parameterized skills, (2) a skill planner powered by LLMs, and (3) a skill executor converting the parameterized skills into executable action sequences. We implement PLAP in MicroRTS, a long-horizon real-time strategy game that provides an unfamiliar and challenging environment for LLMs. The experimental results demonstrate the effectiveness of PLAP. In particular, GPT-4o-driven PLAP in a zero-shot setting outperforms 80% of baseline agents, and Qwen2-72B-driven PLAP, with carefully crafted few-shot examples, surpasses the top-tier scripted agent, CoacAI. Additionally, we design comprehensive evaluation metrics and test 6 closed-source and 2 open-source LLMs within the PLAP framework, ultimately releasing an LLM leaderboard ranking long-horizon skill planning ability. Our code is available at this https URL. 

---
# Do LLMs Understand Wine Descriptors Across Cultures? A Benchmark for Cultural Adaptations of Wine Reviews 

**Authors**: Chenye Zou, Xingyue Wen, Tianyi Hu, Qian Janice Wang, Daniel Hershcovich  

**Link**: [PDF](https://arxiv.org/pdf/2509.12961)  

**Abstract**: Recent advances in large language models (LLMs) have opened the door to culture-aware language tasks. We introduce the novel problem of adapting wine reviews across Chinese and English, which goes beyond literal translation by incorporating regional taste preferences and culture-specific flavor descriptors. In a case study on cross-cultural wine review adaptation, we compile the first parallel corpus of professional reviews, containing 8k Chinese and 16k Anglophone reviews. We benchmark both neural-machine-translation baselines and state-of-the-art LLMs with automatic metrics and human evaluation. For the latter, we propose three culture-oriented criteria -- Cultural Proximity, Cultural Neutrality, and Cultural Genuineness -- to assess how naturally a translated review resonates with target-culture readers. Our analysis shows that current models struggle to capture cultural nuances, especially in translating wine descriptions across different cultures. This highlights the challenges and limitations of translation models in handling cultural content. 

---
# Mitigating Strategy Preference Bias in Emotional Support Conversation via Uncertainty Estimations 

**Authors**: Yougen Zhou, Qin Chen, Ningning Zhou, Jie Zhou, Xingjiao Wu, Liang He  

**Link**: [PDF](https://arxiv.org/pdf/2509.12661)  

**Abstract**: Emotional support conversation (ESC) aims to alleviate distress through empathetic dialogue, yet large language models (LLMs) face persistent challenges in delivering effective ESC due to low accuracy in strategy planning. Moreover, there is a considerable preference bias towards specific strategies. Prior methods using fine-tuned strategy planners have shown potential in reducing such bias, while the underlying causes of the preference bias in LLMs have not well been studied. To address these issues, we first reveal the fundamental causes of the bias by identifying the knowledge boundaries of LLMs in strategy planning. Then, we propose an approach to mitigate the bias by reinforcement learning with a dual reward function, which optimizes strategy planning via both accuracy and entropy-based confidence for each region according to the knowledge boundaries. Experiments on the ESCov and ExTES datasets with multiple LLM backbones show that our approach outperforms the baselines, confirming the effectiveness of our approach. 

---
# Chat-Driven Text Generation and Interaction for Person Retrieval 

**Authors**: Zequn Xie, Chuxin Wang, Sihang Cai, Yeqiang Wang, Shulei Wang, Tao Jin  

**Link**: [PDF](https://arxiv.org/pdf/2509.12662)  

**Abstract**: Text-based person search (TBPS) enables the retrieval of person images from large-scale databases using natural language descriptions, offering critical value in surveillance applications. However, a major challenge lies in the labor-intensive process of obtaining high-quality textual annotations, which limits scalability and practical deployment. To address this, we introduce two complementary modules: Multi-Turn Text Generation (MTG) and Multi-Turn Text Interaction (MTI). MTG generates rich pseudo-labels through simulated dialogues with MLLMs, producing fine-grained and diverse visual descriptions without manual supervision. MTI refines user queries at inference time through dynamic, dialogue-based reasoning, enabling the system to interpret and resolve vague, incomplete, or ambiguous descriptions - characteristics often seen in real-world search scenarios. Together, MTG and MTI form a unified and annotation-free framework that significantly improves retrieval accuracy, robustness, and usability. Extensive evaluations demonstrate that our method achieves competitive or superior results while eliminating the need for manual captions, paving the way for scalable and practical deployment of TBPS systems. 

---
# ConvergeWriter: Data-Driven Bottom-Up Article Construction 

**Authors**: Binquan Ji, Jiaqi Wang, Ruiting Li, Xingchen Han, Yiyang Qi, Shichao Wang, Yifei Lu, Yuantao Han, Feiliang Ren  

**Link**: [PDF](https://arxiv.org/pdf/2509.12811)  

**Abstract**: Large Language Models (LLMs) have shown remarkable prowess in text generation, yet producing long-form, factual documents grounded in extensive external knowledge bases remains a significant challenge. Existing "top-down" methods, which first generate a hypothesis or outline and then retrieve evidence, often suffer from a disconnect between the model's plan and the available knowledge, leading to content fragmentation and factual inaccuracies. To address these limitations, we propose a novel "bottom-up," data-driven framework that inverts the conventional generation pipeline. Our approach is predicated on a "Retrieval-First for Knowledge, Clustering for Structure" strategy, which first establishes the "knowledge boundaries" of the source corpus before any generative planning occurs. Specifically, we perform exhaustive iterative retrieval from the knowledge base and then employ an unsupervised clustering algorithm to organize the retrieved documents into distinct "knowledge clusters." These clusters form an objective, data-driven foundation that directly guides the subsequent generation of a hierarchical outline and the final document content. This bottom-up process ensures that the generated text is strictly constrained by and fully traceable to the source material, proactively adapting to the finite scope of the knowledge base and fundamentally mitigating the risk of hallucination. Experimental results on both 14B and 32B parameter models demonstrate that our method achieves performance comparable to or exceeding state-of-the-art baselines, and is expected to demonstrate unique advantages in knowledge-constrained scenarios that demand high fidelity and structural coherence. Our work presents an effective paradigm for generating reliable, structured, long-form documents, paving the way for more robust LLM applications in high-stakes, knowledge-intensive domains. 

---
# Towards Inclusive Toxic Content Moderation: Addressing Vulnerabilities to Adversarial Attacks in Toxicity Classifiers Tackling LLM-generated Content 

**Authors**: Shaz Furniturewala, Arkaitz Zubiaga  

**Link**: [PDF](https://arxiv.org/pdf/2509.12672)  

**Abstract**: The volume of machine-generated content online has grown dramatically due to the widespread use of Large Language Models (LLMs), leading to new challenges for content moderation systems. Conventional content moderation classifiers, which are usually trained on text produced by humans, suffer from misclassifications due to LLM-generated text deviating from their training data and adversarial attacks that aim to avoid detection. Present-day defence tactics are reactive rather than proactive, since they rely on adversarial training or external detection models to identify attacks. In this work, we aim to identify the vulnerable components of toxicity classifiers that contribute to misclassification, proposing a novel strategy based on mechanistic interpretability techniques. Our study focuses on fine-tuned BERT and RoBERTa classifiers, testing on diverse datasets spanning a variety of minority groups. We use adversarial attacking techniques to identify vulnerable circuits. Finally, we suppress these vulnerable circuits, improving performance against adversarial attacks. We also provide demographic-level insights into these vulnerable circuits, exposing fairness and robustness gaps in model training. We find that models have distinct heads that are either crucial for performance or vulnerable to attack and suppressing the vulnerable heads improves performance on adversarial input. We also find that different heads are responsible for vulnerability across different demographic groups, which can inform more inclusive development of toxicity detection models. 

---
# MORQA: Benchmarking Evaluation Metrics for Medical Open-Ended Question Answering 

**Authors**: Wen-wai Yim, Asma Ben Abacha, Zixuan Yu, Robert Doerning, Fei Xia, Meliha Yetisgen  

**Link**: [PDF](https://arxiv.org/pdf/2509.12405)  

**Abstract**: Evaluating natural language generation (NLG) systems in the medical domain presents unique challenges due to the critical demands for accuracy, relevance, and domain-specific expertise. Traditional automatic evaluation metrics, such as BLEU, ROUGE, and BERTScore, often fall short in distinguishing between high-quality outputs, especially given the open-ended nature of medical question answering (QA) tasks where multiple valid responses may exist. In this work, we introduce MORQA (Medical Open-Response QA), a new multilingual benchmark designed to assess the effectiveness of NLG evaluation metrics across three medical visual and text-based QA datasets in English and Chinese. Unlike prior resources, our datasets feature 2-4+ gold-standard answers authored by medical professionals, along with expert human ratings for three English and Chinese subsets. We benchmark both traditional metrics and large language model (LLM)-based evaluators, such as GPT-4 and Gemini, finding that LLM-based approaches significantly outperform traditional metrics in correlating with expert judgments. We further analyze factors driving this improvement, including LLMs' sensitivity to semantic nuances and robustness to variability among reference answers. Our results provide the first comprehensive, multilingual qualitative study of NLG evaluation in the medical domain, highlighting the need for human-aligned evaluation methods. All datasets and annotations will be publicly released to support future research. 

---
# LLM-as-a-Judge: Rapid Evaluation of Legal Document Recommendation for Retrieval-Augmented Generation 

**Authors**: Anu Pradhan, Alexandra Ortan, Apurv Verma, Madhavan Seshadri  

**Link**: [PDF](https://arxiv.org/pdf/2509.12382)  

**Abstract**: The evaluation bottleneck in recommendation systems has become particularly acute with the rise of Generative AI, where traditional metrics fall short of capturing nuanced quality dimensions that matter in specialized domains like legal research. Can we trust Large Language Models to serve as reliable judges of their own kind? This paper investigates LLM-as-a-Judge as a principled approach to evaluating Retrieval-Augmented Generation systems in legal contexts, where the stakes of recommendation quality are exceptionally high.
We tackle two fundamental questions that determine practical viability: which inter-rater reliability metrics best capture the alignment between LLM and human assessments, and how do we conduct statistically sound comparisons between competing systems? Through systematic experimentation, we discover that traditional agreement metrics like Krippendorff's alpha can be misleading in the skewed distributions typical of AI system evaluations. Instead, Gwet's AC2 and rank correlation coefficients emerge as more robust indicators for judge selection, while the Wilcoxon Signed-Rank Test with Benjamini-Hochberg corrections provides the statistical rigor needed for reliable system comparisons.
Our findings suggest a path toward scalable, cost-effective evaluation that maintains the precision demanded by legal applications, transforming what was once a human-intensive bottleneck into an automated, yet statistically principled, evaluation framework. 

---
# Audited Reasoning Refinement: Fine-Tuning Language Models via LLM-Guided Step-Wise Evaluation and Correction 

**Authors**: Sumanta Bhattacharyya, Sara Riaz, Pedram Rooshenas  

**Link**: [PDF](https://arxiv.org/pdf/2509.12476)  

**Abstract**: Training a task-specific small reasoning model is challenging when direct human supervision or high-quality labels are scarce. However, LLMs with reasoning capabilities produce abundant intermediate reasoning traces that can be systematically refined to create effective supervision signals. We propose Reason-Refine-then-Align (R2tA), which turns refined model rationales into supervision for training task-specific reasoning models. Our method generates initial reasoning and responses from an open-source base model on task-specific inputs, then refines these traces, fixing hallucinations and inconsistencies, to form a high-fidelity dataset. We perform a two-stage alignment, supervised fine-tuning (SFT), followed by direct preference optimization (DPO) to calibrate the model's intermediate reasoning with human-validated conceptual preferences and then condition the final output on that aligned reasoning. As a case study, we apply R2tA to evaluate extended entity relationship diagrams (EERDs) in database system design, a structurally complex task where prompt-only methods miss or hallucinate errors. We curated a dataset of 600 EERD variants (train/test split of 450/150, respectively) with induced mistakes spanning 11 categories. Empirical evaluation suggests R2tA provides a practical, cost-effective path to scalable LLM adaptation in data-scarce domains, enabling reproducible AI tools for education and beyond. 

---
# Rethinking the Evaluation of Alignment Methods: Insights into Diversity, Generalisation, and Safety 

**Authors**: Denis Janiak, Julia Moska, Dawid Motyka, Karolina Seweryn, Paweł Walkowiak, Bartosz Żuk, Arkadiusz Janz  

**Link**: [PDF](https://arxiv.org/pdf/2509.12936)  

**Abstract**: Large language models (LLMs) require careful alignment to balance competing objectives - factuality, safety, conciseness, proactivity, and diversity. Existing studies focus on individual techniques or specific dimensions, lacking a holistic assessment of the inherent trade-offs. We propose a unified evaluation framework that compares LLM alignment methods (PPO, DPO, ORPO, KTO) across these five axes, using both in-distribution and out-of-distribution datasets. Leveraging a specialized LLM-as-Judge prompt, validated through human studies, we reveal that DPO and KTO excel in factual accuracy, PPO and DPO lead in safety, and PPO best balances conciseness with proactivity. Our findings provide insights into trade-offs of common alignment methods, guiding the development of more balanced and reliable LLMs. 

---
# MAGIC-Enhanced Keyword Prompting for Zero-Shot Audio Captioning with CLIP Models 

**Authors**: Vijay Govindarajan, Pratik Patel, Sahil Tripathi, Md Azizul Hoque, Gautam Siddharth Kashyap  

**Link**: [PDF](https://arxiv.org/pdf/2509.12591)  

**Abstract**: Automated Audio Captioning (AAC) generates captions for audio clips but faces challenges due to limited datasets compared to image captioning. To overcome this, we propose the zero-shot AAC system that leverages pre-trained models, eliminating the need for extensive training. Our approach uses a pre-trained audio CLIP model to extract auditory features and generate a structured prompt, which guides a Large Language Model (LLM) in caption generation. Unlike traditional greedy decoding, our method refines token selection through the audio CLIP model, ensuring alignment with the audio content. Experimental results demonstrate a 35% improvement in NLG mean score (from 4.7 to 7.3) using MAGIC search with the WavCaps model. The performance is heavily influenced by the audio-text matching model and keyword selection, with optimal results achieved using a single keyword prompt, and a 50% performance drop when no keyword list is used. 

---
# Small Models, Big Results: Achieving Superior Intent Extraction through Decomposition 

**Authors**: Danielle Cohen, Yoni Halpern, Noam Kahlon, Joel Oren, Omri Berkovitch, Sapir Caduri, Ido Dagan, Anatoly Efros  

**Link**: [PDF](https://arxiv.org/pdf/2509.12423)  

**Abstract**: Understanding user intents from UI interaction trajectories remains a challenging, yet crucial, frontier in intelligent agent development. While massive, datacenter-based, multi-modal large language models (MLLMs) possess greater capacity to handle the complexities of such sequences, smaller models which can run on-device to provide a privacy-preserving, low-cost, and low-latency user experience, struggle with accurate intent inference. We address these limitations by introducing a novel decomposed approach: first, we perform structured interaction summarization, capturing key information from each user action. Second, we perform intent extraction using a fine-tuned model operating on the aggregated summaries. This method improves intent understanding in resource-constrained models, even surpassing the base performance of large MLLMs. 

---
