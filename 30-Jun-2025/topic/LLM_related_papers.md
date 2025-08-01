# The Automated LLM Speedrunning Benchmark: Reproducing NanoGPT Improvements 

**Authors**: Bingchen Zhao, Despoina Magka, Minqi Jiang, Xian Li, Roberta Raileanu, Tatiana Shavrina, Jean-Christophe Gagnon-Audet, Kelvin Niu, Shagun Sodhani, Michael Shvartsman, Andrei Lupu, Alisia Lupidi, Edan Toledo, Karen Hambardzumyan, Martin Josifoski, Thomas Foster, Lucia Cipolina-Kun, Abhishek Charnalia, Derek Dunfield, Alexander H. Miller, Oisin Mac Aodha, Jakob Foerster, Yoram Bachrach  

**Link**: [PDF](https://arxiv.org/pdf/2506.22419)  

**Abstract**: Rapid advancements in large language models (LLMs) have the potential to assist in scientific progress. A critical capability toward this endeavor is the ability to reproduce existing work. To evaluate the ability of AI agents to reproduce results in an active research area, we introduce the Automated LLM Speedrunning Benchmark, leveraging the research community contributions on the NanoGPT speedrun, a competition to train a GPT-2 model in the shortest time. Each of the 19 speedrun tasks provides the agent with the previous records training script, optionally paired with one of three hint formats, ranging from pseudocode to paper-like descriptions of the new records improvements. Records execute quickly by design and speedrun improvements encompass diverse code-level changes, ranging from high-level algorithmic advancements to hardware-aware optimizations. These features make the benchmark both accessible and realistic for the frontier problem of improving LLM training. We find that recent reasoning LLMs combined with SoTA scaffolds struggle to reimplement already-known innovations in our benchmark, even when given detailed hints. Our benchmark thus provides a simple, non-saturated measure of an LLMs ability to automate scientific reproduction, a necessary (but not sufficient) skill for an autonomous research agent. 

---
# LeanConjecturer: Automatic Generation of Mathematical Conjectures for Theorem Proving 

**Authors**: Naoto Onda, Kazumi Kasaura, Yuta Oriike, Masaya Taniguchi, Akiyoshi Sannai, Sho Sonoda  

**Link**: [PDF](https://arxiv.org/pdf/2506.22005)  

**Abstract**: We introduce LeanConjecturer, a pipeline for automatically generating university-level mathematical conjectures in Lean 4 using Large Language Models (LLMs). Our hybrid approach combines rule-based context extraction with LLM-based theorem statement generation, addressing the data scarcity challenge in formal theorem proving. Through iterative generation and evaluation, LeanConjecturer produced 12,289 conjectures from 40 Mathlib seed files, with 3,776 identified as syntactically valid and non-trivial, that is, cannot be proven by \texttt{aesop} tactic. We demonstrate the utility of these generated conjectures for reinforcement learning through Group Relative Policy Optimization (GRPO), showing that targeted training on domain-specific conjectures can enhance theorem proving capabilities. Our approach generates 103.25 novel conjectures per seed file on average, providing a scalable solution for creating training data for theorem proving systems. Our system successfully verified several non-trivial theorems in topology, including properties of semi-open, alpha-open, and pre-open sets, demonstrating its potential for mathematical discovery beyond simple variations of existing results. 

---
# CitySim: Modeling Urban Behaviors and City Dynamics with Large-Scale LLM-Driven Agent Simulation 

**Authors**: Nicolas Bougie, Narimasa Watanabe  

**Link**: [PDF](https://arxiv.org/pdf/2506.21805)  

**Abstract**: Modeling human behavior in urban environments is fundamental for social science, behavioral studies, and urban planning. Prior work often rely on rigid, hand-crafted rules, limiting their ability to simulate nuanced intentions, plans, and adaptive behaviors. Addressing these challenges, we envision an urban simulator (CitySim), capitalizing on breakthroughs in human-level intelligence exhibited by large language models. In CitySim, agents generate realistic daily schedules using a recursive value-driven approach that balances mandatory activities, personal habits, and situational factors. To enable long-term, lifelike simulations, we endow agents with beliefs, long-term goals, and spatial memory for navigation. CitySim exhibits closer alignment with real humans than prior work, both at micro and macro levels. Additionally, we conduct insightful experiments by modeling tens of thousands of agents and evaluating their collective behaviors under various real-world scenarios, including estimating crowd density, predicting place popularity, and assessing well-being. Our results highlight CitySim as a scalable, flexible testbed for understanding and forecasting urban phenomena. 

---
# Hierarchical Reasoning Model 

**Authors**: Guan Wang, Jin Li, Yuhao Sun, Xing Chen, Changling Liu, Yue Wu, Meng Lu, Sen Song, Yasin Abbasi Yadkori  

**Link**: [PDF](https://arxiv.org/pdf/2506.21734)  

**Abstract**: Reasoning, the process of devising and executing complex goal-oriented action sequences, remains a critical challenge in AI. Current large language models (LLMs) primarily employ Chain-of-Thought (CoT) techniques, which suffer from brittle task decomposition, extensive data requirements, and high latency. Inspired by the hierarchical and multi-timescale processing in the human brain, we propose the Hierarchical Reasoning Model (HRM), a novel recurrent architecture that attains significant computational depth while maintaining both training stability and efficiency. HRM executes sequential reasoning tasks in a single forward pass without explicit supervision of the intermediate process, through two interdependent recurrent modules: a high-level module responsible for slow, abstract planning, and a low-level module handling rapid, detailed computations. With only 27 million parameters, HRM achieves exceptional performance on complex reasoning tasks using only 1000 training samples. The model operates without pre-training or CoT data, yet achieves nearly perfect performance on challenging tasks including complex Sudoku puzzles and optimal path finding in large mazes. Furthermore, HRM outperforms much larger models with significantly longer context windows on the Abstraction and Reasoning Corpus (ARC), a key benchmark for measuring artificial general intelligence capabilities. These results underscore HRM's potential as a transformative advancement toward universal computation and general-purpose reasoning systems. 

---
# THE-Tree: Can Tracing Historical Evolution Enhance Scientific Verification and Reasoning? 

**Authors**: Xin Wang, Jiyao Liu, Yulong Xiao, Junzhi Ning, Lihao Liu, Junjun He, Botian Shi, Kaicheng Yu  

**Link**: [PDF](https://arxiv.org/pdf/2506.21763)  

**Abstract**: Large Language Models (LLMs) are accelerating scientific idea generation, but rigorously evaluating these numerous, often superficial, AI-generated propositions for novelty and factual accuracy is a critical bottleneck; manual verification is too this http URL validation methods are inadequate: LLMs as standalone verifiers may hallucinate and lack domain knowledge (our findings show ~60\% unawareness of relevant papers in specific domains), while traditional citation networks lack explicit causality and narrative surveys are this http URL underscores a core challenge: the absence of structured, verifiable, and causally-linked historical data of scientific this http URL address this,we introduce \textbf{THE-Tree} (\textbf{T}echnology \textbf{H}istory \textbf{E}volution Tree), a computational framework that constructs such domain-specific evolution trees from scientific this http URL-Tree employs a search algorithm to explore evolutionary paths. During its node expansion, it utilizes a novel "Think-Verbalize-Cite-Verify" process: an LLM proposes potential advancements and cites supporting literature. Critically, each proposed evolutionary link is then validated for logical coherence and evidential support by a recovered natural language inference mechanism that interrogates the cited literature, ensuring that each step is this http URL construct and validate 88 THE-Trees across diverse domains and release a benchmark dataset including up to 71k fact verifications covering 27k papers to foster further this http URL demonstrate that i) in graph completion, our THE-Tree improves hit@1 by 8\% to 14\% across multiple models compared to traditional citation networks; ii) for predicting future scientific developments, it improves hit@1 metric by nearly 10\%; and iii) when combined with other methods, it boosts the performance of evaluating important scientific papers by almost 100\%. 

---
# QuickSilver -- Speeding up LLM Inference through Dynamic Token Halting, KV Skipping, Contextual Token Fusion, and Adaptive Matryoshka Quantization 

**Authors**: Danush Khanna, Aditya Kumar Guru, Srivarshinee Sridhar, Zidan Ahmed, Rubhav Bahirwani, Meetu Malhotra, Vinija Jain, Aman Chadha, Amitava Das, Kripabandhu Ghosh  

**Link**: [PDF](https://arxiv.org/pdf/2506.22396)  

**Abstract**: Inference accounts for the majority of latency and energy consumption in large language model (LLM) deployments, often exceeding 90% of total cost. While training-time efficiency has seen extensive progress, runtime optimization remains a key bottleneck, particularly under autoregressive decoding. Existing approaches -- such as pruning, quantization, early exits, and speculative decoding -- often require retraining, architectural changes, or disrupt decoding compatibility. We introduce QuickSilver, a modular, token-level framework that enables semantic adaptivity at inference time without altering model weights or structure. QuickSilver integrates four synergistic mechanisms:
(i) Dynamic Token Halting, which halts computation for tokens with converged representations; (ii) KV Cache Skipping, which selectively suppresses memory writes to reduce attention overhead; and (iii) Contextual Token Fusion, which collapses redundant tokens into shared paths to shrink sequence length.
Unlike speculative decoding or MoE routing, QuickSilver operates entirely on frozen, dense models and requires no auxiliary networks. Applied to GPT-2 and Llama-2 across WikiText-103 and C4, QuickSilver achieves up to 39.6% FLOP reduction with negligible perplexity degradation (<=0.2). 

---
# Probabilistic Optimality for Inference-time Scaling 

**Authors**: Youkang Wang, Jian Wang, Rubing Chen, Xiao-Yong Wei, Qing Li  

**Link**: [PDF](https://arxiv.org/pdf/2506.22376)  

**Abstract**: Inference-time scaling has emerged as a powerful technique for enhancing the reasoning performance of Large Language Models (LLMs). However, existing approaches often rely on heuristic strategies for parallel sampling, lacking a principled foundation. To address this gap, we propose a probabilistic framework that formalizes the optimality of inference-time scaling under the assumption that parallel samples are independently and identically distributed (i.i.d.), and where the Best-of-N selection strategy follows a probability distribution that can be estimated. Within this framework, we derive a theoretical lower bound on the required number of samples to achieve a target performance level, providing the first principled guidance for compute-efficient scaling. Leveraging this insight, we develop \textsc{OptScale}, a practical algorithm that dynamically determines the optimal number of sampled responses. \textsc{OptScale} employs a language model-based predictor to estimate probabilistic prior parameters, enabling the decision of the minimal number of samples needed that satisfy predefined performance thresholds and confidence levels. Extensive experiments on mathematical reasoning benchmarks (including MATH-500, GSM8K, AIME, and AMC) demonstrate that \textsc{OptScale} significantly reduces sampling overhead while remaining better or on par with state-of-the-art reasoning performance. Our work offers both a theoretical foundation and a practical solution for principled inference-time scaling, addressing a critical gap in the efficient deployment of LLMs for complex reasoning. 

---
# HyperCLOVA X THINK Technical Report 

**Authors**: NAVER Cloud HyperCLOVA X Team  

**Link**: [PDF](https://arxiv.org/pdf/2506.22403)  

**Abstract**: We introduce HyperCLOVA X THINK, the first reasoning-focused large language model in the HyperCLOVA X family, pre-trained on roughly $6$ trillion high-quality Korean, and English tokens, augmented with targeted synthetic Korean data. It was implemented as a compute-memory-balanced Peri-LN Transformer scaled with $\mu$P, pre-trained through a three-stage curriculum that expands the context window to $128$K tokens, and post-trained via supervised fine-tuning with Reinforcement Learning from Verifiable Rewards supports both detailed rationale and concise-answer modes. It delivers competitive performance against similarly sized models on Korea-focused benchmarks such as KMMLU, CSAT, KoBALT-700, HAERAE-1.0, and KoBigBench, while preserving robust bilingual consistency and translation quality. In addition, a vision-augmented variant matches or exceeds GPT-4.1 on the KCSAT STEM benchmark, all of which are achieved with substantially lower training compute than existing models of similar sizes. We also present a pruning and distillation technique that will soon be applied to HyperCLOVA X THINK for an open-source and business-friendly foundation model. Altogether, these capabilities position HyperCLOVA X THINK as a robust foundation for Korean AI innovation and a valuable resource for the global research community. 

---
# Can Video Large Multimodal Models Think Like Doubters-or Double-Down: A Study on Defeasible Video Entailment 

**Authors**: Yue Zhang, Jilei Sun, Yunhui Guo, Vibhav Gogate  

**Link**: [PDF](https://arxiv.org/pdf/2506.22385)  

**Abstract**: Video Large Multimodal Models (VLMMs) have made impressive strides in understanding video content, but they often struggle with abstract and adaptive reasoning-the ability to revise their interpretations when new information emerges. In reality, conclusions are rarely set in stone; additional context can strengthen or weaken an initial inference. To address this, we introduce Defeasible Video Entailment (DVidE), a new task that challenges models to think like doubters, constantly updating their reasoning based on evolving evidence. In DVidE, given a video premise and a textual hypothesis, models must determine whether a new update strengthens or weakens the hypothesis (classification version) or generate a coherent update that modifies the entailment relationship (generation version). For solving the classification task, we propose the Chain of Counterfactual Thought framework, utilizing counterfactual reasoning, ASR-enhanced video content, and rationale refinement to reduce inference bias. For the generation task, we develop a framework that combines ASR output with a Large Language Model (LLM) to produce coherent, contextually relevant updates aligned with the intended strengthener or weakener goals. Additionally, we introduce a novel benchmark dataset, with strengthener/weakener annotations and an LLM-based evaluation metric specifically designed for assessing generative performance. Experimental results demonstrate significant improvements, highlighting our proposed method in enhancing dynamic reasoning capabilities of VLMMs. 

---
# EFRame: Deeper Reasoning via Exploration-Filtering-Replay Reinforcement Learning Framework 

**Authors**: Chen Wang, Lai Wei, Yanzhi Zhang, Chenyang Shao, Zedong Dan, Weiran Huang, Yue Wang, Yuzhi Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2506.22200)  

**Abstract**: Recent advances in reinforcement learning (RL) have significantly enhanced the reasoning capabilities of large language models (LLMs). Group Relative Policy Optimization (GRPO), an efficient variant of PPO that lowers RL's computational cost, still faces limited exploration, low sample efficiency and instability, constraining its performance on complex reasoning tasks. To address these limitations, we introduce EFRame, an Exploration-Filtering-Replay framework that systematically augments GRPO along three critical dimensions. EFRame performs additional rollouts to explore high-quality trajectories, applies online filtering to eliminate low-quality samples that introduce noise and variance, and leverages experience replay to repeatedly exploit rare but informative samples. EFRame establishes a complete and stable learning cycle, guiding the model through a structured transition from exploration to convergence. Our experiments across a variety of reasoning benchmarks demonstrate that EFRame not only improves the robustness and efficiency of training, but also enables access to deeper reasoning capabilities that remain unattainable under vanilla GRPO. Furthermore, EFRame enables a more fine-grained categorization of training samples, allowing for a deeper analysis of how different types of samples contribute to the learning process in RL. Our code is available at this https URL. 

---
# Literature-Grounded Novelty Assessment of Scientific Ideas 

**Authors**: Simra Shahid, Marissa Radensky, Raymond Fok, Pao Siangliulue, Daniel S. Weld, Tom Hope  

**Link**: [PDF](https://arxiv.org/pdf/2506.22026)  

**Abstract**: Automated scientific idea generation systems have made remarkable progress, yet the automatic evaluation of idea novelty remains a critical and underexplored challenge. Manual evaluation of novelty through literature review is labor-intensive, prone to error due to subjectivity, and impractical at scale. To address these issues, we propose the Idea Novelty Checker, an LLM-based retrieval-augmented generation (RAG) framework that leverages a two-stage retrieve-then-rerank approach. The Idea Novelty Checker first collects a broad set of relevant papers using keyword and snippet-based retrieval, then refines this collection through embedding-based filtering followed by facet-based LLM re-ranking. It incorporates expert-labeled examples to guide the system in comparing papers for novelty evaluation and in generating literature-grounded reasoning. Our extensive experiments demonstrate that our novelty checker achieves approximately 13% higher agreement than existing approaches. Ablation studies further showcases the importance of the facet-based re-ranker in identifying the most relevant literature for novelty evaluation. 

---
# Using Large Language Models to Suggest Informative Prior Distributions in Bayesian Statistics 

**Authors**: Michael A. Riegler, Kristoffer Herland Hellton, Vajira Thambawita, Hugo L. Hammer  

**Link**: [PDF](https://arxiv.org/pdf/2506.21964)  

**Abstract**: Selecting prior distributions in Bayesian statistics is challenging, resource-intensive, and subjective. We analyze using large-language models (LLMs) to suggest suitable, knowledge-based informative priors. We developed an extensive prompt asking LLMs not only to suggest priors but also to verify and reflect on their choices.
We evaluated Claude Opus, Gemini 2.5 Pro, and ChatGPT-4o-mini on two real datasets: heart disease risk and concrete strength. All LLMs correctly identified the direction for all associations (e.g., that heart disease risk is higher for males). The quality of suggested priors was measured by their Kullback-Leibler divergence from the maximum likelihood estimator's distribution.
The LLMs suggested both moderately and weakly informative priors. The moderate priors were often overconfident, resulting in distributions misaligned with the data. In our experiments, Claude and Gemini provided better priors than ChatGPT. For weakly informative priors, a key performance difference emerged: ChatGPT and Gemini defaulted to an "unnecessarily vague" mean of 0, while Claude did not, demonstrating a significant advantage.
The ability of LLMs to identify correct associations shows their great potential as an efficient, objective method for developing informative priors. However, the primary challenge remains in calibrating the width of these priors to avoid over- and under-confidence. 

---
# The Consistency Hypothesis in Uncertainty Quantification for Large Language Models 

**Authors**: Quan Xiao, Debarun Bhattacharjya, Balaji Ganesan, Radu Marinescu, Katsiaryna Mirylenka, Nhan H Pham, Michael Glass, Junkyu Lee  

**Link**: [PDF](https://arxiv.org/pdf/2506.21849)  

**Abstract**: Estimating the confidence of large language model (LLM) outputs is essential for real-world applications requiring high user trust. Black-box uncertainty quantification (UQ) methods, relying solely on model API access, have gained popularity due to their practical benefits. In this paper, we examine the implicit assumption behind several UQ methods, which use generation consistency as a proxy for confidence, an idea we formalize as the consistency hypothesis. We introduce three mathematical statements with corresponding statistical tests to capture variations of this hypothesis and metrics to evaluate LLM output conformity across tasks. Our empirical investigation, spanning 8 benchmark datasets and 3 tasks (question answering, text summarization, and text-to-SQL), highlights the prevalence of the hypothesis under different settings. Among the statements, we highlight the `Sim-Any' hypothesis as the most actionable, and demonstrate how it can be leveraged by proposing data-free black-box UQ methods that aggregate similarities between generations for confidence estimation. These approaches can outperform the closest baselines, showcasing the practical value of the empirically observed consistency hypothesis. 

---
# ARAG: Agentic Retrieval Augmented Generation for Personalized Recommendation 

**Authors**: Reza Yousefi Maragheh, Pratheek Vadla, Priyank Gupta, Kai Zhao, Aysenur Inan, Kehui Yao, Jianpeng Xu, Praveen Kanumala, Jason Cho, Sushant Kumar  

**Link**: [PDF](https://arxiv.org/pdf/2506.21931)  

**Abstract**: Retrieval-Augmented Generation (RAG) has shown promise in enhancing recommendation systems by incorporating external context into large language model prompts. However, existing RAG-based approaches often rely on static retrieval heuristics and fail to capture nuanced user preferences in dynamic recommendation scenarios. In this work, we introduce ARAG, an Agentic Retrieval-Augmented Generation framework for Personalized Recommendation, which integrates a multi-agent collaboration mechanism into the RAG pipeline. To better understand the long-term and session behavior of the user, ARAG leverages four specialized LLM-based agents: a User Understanding Agent that summarizes user preferences from long-term and session contexts, a Natural Language Inference (NLI) Agent that evaluates semantic alignment between candidate items retrieved by RAG and inferred intent, a context summary agent that summarizes the findings of NLI agent, and an Item Ranker Agent that generates a ranked list of recommendations based on contextual fit. We evaluate ARAG accross three datasets. Experimental results demonstrate that ARAG significantly outperforms standard RAG and recency-based baselines, achieving up to 42.1% improvement in NDCG@5 and 35.5% in Hit@5. We also, conduct an ablation study to analyse the effect by different components of ARAG. Our findings highlight the effectiveness of integrating agentic reasoning into retrieval-augmented recommendation and provide new directions for LLM-based personalization. 

---
# Evaluating List Construction and Temporal Understanding capabilities of Large Language Models 

**Authors**: Alexandru Dumitru, V Venktesh, Adam Jatowt, Avishek Anand  

**Link**: [PDF](https://arxiv.org/pdf/2506.21783)  

**Abstract**: Large Language Models (LLMs) have demonstrated immense advances in a wide range of natural language tasks. However, these models are susceptible to hallucinations and errors on particularly temporal understanding tasks involving multiple entities in answers. In such tasks, they fail to associate entities with accurate time intervals, generate a complete list of entities in answers or reason about events associated with specific temporal bounds. Existing works do not extensively evaluate the abilities of the model to perform implicit and explicit temporal understanding in a list answer construction setup. To bridge this gap, we propose the Time referenced List based Question Answering or TLQA benchmark that requires structured answers in list format aligned with corresponding time periods. Our TLQA benchmark, requires both list construction and temporal understanding simultaneously, which to the best of our knowledge has not been explored in prior benchmarks. We investigate the temporal understanding and list construction capabilities of state-of-the-art generative models on TLQA in closed-book and open-domain settings. Our findings reveal significant shortcomings in current models, particularly their inability to provide complete answers and temporally align facts in a closed-book setup and the need to improve retrieval in open-domain setup, providing clear future directions for research on TLQA. The benchmark and code at this https URL. 

---
# APO: Enhancing Reasoning Ability of MLLMs via Asymmetric Policy Optimization 

**Authors**: Minjie Hong, Zirun Guo, Yan Xia, Zehan Wang, Ziang Zhang, Tao Jin, Zhou Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2506.21655)  

**Abstract**: Multimodal Large Language Models (MLLMs) are powerful at integrating diverse data, but they often struggle with complex reasoning. While Reinforcement learning (RL) can boost reasoning in LLMs, applying it to MLLMs is tricky. Common issues include a drop in performance on general tasks and the generation of overly detailed or "overthinking" reasoning. Our work investigates how the KL penalty and overthinking affect RL training in MLLMs. We propose Asymmetric Policy Optimization (APO) to address these issues, which divides the sampled responses into positive and negative groups. For positive samples, Difficulty-Adaptive Divergence Shaping (DADS) is introduced to dynamically adjust the KL divergence weight based on their difficulty. This method prevents policy entropy from dropping sharply, improves training stability, utilizes samples better, and preserves the model's existing knowledge. For negative samples, Suboptimal Trajectory Complexity Regularization (STCR) is proposed to penalize overly long responses. This helps mitigate overthinking and encourages more concise reasoning while preserving the model's explorative capacity. We apply our method to Qwen2.5-VL-3B, creating View-R1-3B. View-R1-3B significantly enhances reasoning capabilities, showing an average 7\% gain over the base model and outperforming larger MLLMs (7-11B) on various reasoning benchmarks. Importantly, unlike other reasoning-tuned MLLMs that often degrade on general tasks, View-R1-3B maintains consistent improvement, demonstrating superior generalization. These results highlight the effectiveness and broad applicability of our DADS and STCR techniques for advancing complex multimodal reasoning in MLLMs. The code will be made available at this https URL. 

---
# How Large Language Models play humans in online conversations: a simulated study of the 2016 US politics on Reddit 

**Authors**: Daniele Cirulli, Giulio Cimini, Giovanni Palermo  

**Link**: [PDF](https://arxiv.org/pdf/2506.21620)  

**Abstract**: Large Language Models (LLMs) have recently emerged as powerful tools for natural language generation, with applications spanning from content creation to social simulations. Their ability to mimic human interactions raises both opportunities and concerns, particularly in the context of politically relevant online discussions. In this study, we evaluate the performance of LLMs in replicating user-generated content within a real-world, divisive scenario: Reddit conversations during the 2016 US Presidential election. In particular, we conduct three different experiments, asking GPT-4 to generate comments by impersonating either real or artificial partisan users. We analyze the generated comments in terms of political alignment, sentiment, and linguistic features, comparing them against real user contributions and benchmarking against a null model. We find that GPT-4 is able to produce realistic comments, both in favor of or against the candidate supported by the community, yet tending to create consensus more easily than dissent. In addition we show that real and artificial comments are well separated in a semantically embedded space, although they are indistinguishable by manual inspection. Our findings provide insights on the potential use of LLMs to sneak into online discussions, influence political debate and shape political narratives, bearing broader implications of AI-driven discourse manipulation. 

---
# LastingBench: Defend Benchmarks Against Knowledge Leakage 

**Authors**: Yixiong Fang, Tianran Sun, Yuling Shi, Min Wang, Xiaodong Gu  

**Link**: [PDF](https://arxiv.org/pdf/2506.21614)  

**Abstract**: The increasing complexity of large language models (LLMs) raises concerns about their ability to "cheat" on standard Question Answering (QA) benchmarks by memorizing task-specific data. This undermines the validity of benchmark evaluations, as they no longer reflect genuine model capabilities but instead the effects of data leakage. While prior work has focused on detecting such leakage, little attention has been given to mitigating its impact and preserving the long-term utility of benchmarks. In this paper, we introduce LastingBench, a novel framework designed to continuously reinforce and safeguard existing benchmarks against knowledge leakage. LastingBench identifies leakage points in the context through perturbation, then rewrites the leakage points to counterfactual ones-disrupting memorization while preserving the benchmark's original evaluative intent. Evaluations of state-of-the-art QA benchmarks show significant performance gaps, highlighting the efficacy of LastingBench in reducing memorization effects. LastingBench offers a practical and scalable solution to ensure benchmark robustness over time, promoting fairer and more interpretable evaluations of LLMs. 

---
# The Open Proof Corpus: A Large-Scale Study of LLM-Generated Mathematical Proofs 

**Authors**: Jasper Dekoninck, Ivo Petrov, Kristian Minchev, Mislav Balunovic, Martin Vechev, Miroslav Marinov, Maria Drencheva, Lyuba Konova, Milen Shumanov, Kaloyan Tsvetkov, Nikolay Drenchev, Lazar Todorov, Kalina Nikolova, Nikolay Georgiev, Vanesa Kalinkova, Margulan Ismoldayev  

**Link**: [PDF](https://arxiv.org/pdf/2506.21621)  

**Abstract**: In recent months, large language models (LLMs) have made significant progress in mathematical proof generation, but further advancement is hindered by the lack of a large-scale, high-quality dataset of human-evaluated proofs. While expensive to create, such a dataset is essential for driving improvements in training and enabling a rigorous analysis of proof generation capabilities. In this work, we present the Open Proof Corpus (OPC), a dataset comprising over 5,000 human-evaluated proofs produced by state-of-the-art LLMs. The OPC was specifically designed for broad applicability and downstream usage in proof generation research and is the first to include a substantial number of correct, LLM-generated solutions to problems from prestigious mathematics competitions such as the USAMO and IMO. Using the OPC, we explore critical questions in automated proof generation: (1) the performance gap between natural language and formal proof generation, (2) the discrepancy between final-answer accuracy and full-proof validity, and (3) the impact of best-of-n selection on proof quality. Finally, to showcase the utility of the OPC, we finetune an 8B-parameter model on the dataset, obtaining a model that performs on par with the best model, Gemini-2.5-Pro, on the task of evaluating proof correctness. 

---
# From Thinking to Output: Chain-of-Thought and Text Generation Characteristics in Reasoning Language Models 

**Authors**: Junhao Liu, Zhenhao Xu, Yuxin Fang, Yichuan Chen, Zuobin Ying, Wenhan Chang  

**Link**: [PDF](https://arxiv.org/pdf/2506.21609)  

**Abstract**: Recently, there have been notable advancements in large language models (LLMs), demonstrating their growing abilities in complex reasoning. However, existing research largely overlooks a thorough and systematic comparison of these models' reasoning processes and outputs, particularly regarding their self-reflection pattern (also termed "Aha moment") and the interconnections across diverse domains. This paper proposes a novel framework for analyzing the reasoning characteristics of four cutting-edge large reasoning models (GPT-o1, DeepSeek-R1, Kimi-k1.5, and Grok-3) using keywords statistic and LLM-as-a-judge paradigm. Our approach connects their internal thinking processes with their final outputs. A diverse dataset consists of real-world scenario-based questions covering logical deduction, causal inference, and multi-step problem-solving. Additionally, a set of metrics is put forward to assess both the coherence of reasoning and the accuracy of the outputs. The research results uncover various patterns of how these models balance exploration and exploitation, deal with problems, and reach conclusions during the reasoning process. Through quantitative and qualitative comparisons, disparities among these models are identified in aspects such as the depth of reasoning, the reliance on intermediate steps, and the degree of similarity between their thinking processes and output patterns and those of GPT-o1. This work offers valuable insights into the trade-off between computational efficiency and reasoning robustness and provides practical recommendations for enhancing model design and evaluation in practical applications. We publicly release our project at: this https URL 

---
# Structured Attention Matters to Multimodal LLMs in Document Understanding 

**Authors**: Chang Liu, Hongkai Chen, Yujun Cai, Hang Wu, Qingwen Ye, Ming-Hsuan Yang, Yiwei Wang  

**Link**: [PDF](https://arxiv.org/pdf/2506.21600)  

**Abstract**: Document understanding remains a significant challenge for multimodal large language models (MLLMs). While previous research has primarily focused on locating evidence pages through precise multimodal queries, our work investigates a fundamental yet overlooked aspect: how input format influences document comprehension performance. Through systematic analysis, we discover that raw OCR text often impairs rather than improves MLLMs' performance, which is a counterintuitive finding we attribute to attention dispersion and structure loss. To further substantiate our hypothesis, we propose a novel structure-preserving approach that encodes document elements using the LaTex paradigm, maintaining the hierarchical organization and spatial relationships critical for comprehension. Our attention analysis reveals that structured text induces structured attention patterns on both textual and visual content, directing models to focus on semantically meaningful regions while reducing attention waste. This approach significantly enhances MLLMs' document question answering performance across diverse document types without requiring architectural modifications or additional training. 

---
# Reinforcement Fine-Tuned Large Language Models for Next POI Recommendation 

**Authors**: Peibo Li, Shuang Ao, Hao Xue, Yang Song, Maarten de Rijke, Johan Barthélemy, Tomasz Bednarz, Flora D. Salim  

**Link**: [PDF](https://arxiv.org/pdf/2506.21599)  

**Abstract**: Large language models (LLMs) have been adopted for next point-of-interest (POI) recommendation tasks. Typical LLM-based recommenders fall into two categories: prompt-based and supervised fine-tuning (SFT)-based models. Prompt-based models generally offer greater output flexibility but deliver lower accuracy, whereas SFT-based models achieve higher performance yet face a fundamental mismatch: next POI recommendation data does not naturally suit supervised fine-tuning. In SFT, the model is trained to reproduce the exact ground truth, but each training example provides only a single target POI, so there is no ground truth for producing a top-k list.
To address this, we propose Refine-POI, a reinforcement fine-tuning framework for next POI recommendation. We introduce recommendation-driven rewards that enable LLMs to learn to generate top-k recommendation lists using only one ground-truth POI per example. Experiments on real-world datasets demonstrate that Refine-POI achieves state-of-the-art top-k recommendation performance. 

---
# Overview of the ClinIQLink 2025 Shared Task on Medical Question-Answering 

**Authors**: Brandon Colelough, Davis Bartels, Dina Demner-Fushman  

**Link**: [PDF](https://arxiv.org/pdf/2506.21597)  

**Abstract**: In this paper, we present an overview of ClinIQLink, a shared task, collocated with the 24th BioNLP workshop at ACL 2025, designed to stress-test large language models (LLMs) on medically-oriented question answering aimed at the level of a General Practitioner. The challenge supplies 4,978 expert-verified, medical source-grounded question-answer pairs that cover seven formats: true/false, multiple choice, unordered list, short answer, short-inverse, multi-hop, and multi-hop-inverse. Participating systems, bundled in Docker or Apptainer images, are executed on the CodaBench platform or the University of Maryland's Zaratan cluster. An automated harness (Task 1) scores closed-ended items by exact match and open-ended items with a three-tier embedding metric. A subsequent physician panel (Task 2) audits the top model responses. 

---
# VIDEE: Visual and Interactive Decomposition, Execution, and Evaluation of Text Analytics with Intelligent Agents 

**Authors**: Sam Yu-Te Lee, Chengyang Ji, Shicheng Wen, Lifu Huang, Dongyi Liu, Kwan-Liu Ma  

**Link**: [PDF](https://arxiv.org/pdf/2506.21582)  

**Abstract**: Text analytics has traditionally required specialized knowledge in Natural Language Processing (NLP) or text analysis, which presents a barrier for entry-level analysts. Recent advances in large language models (LLMs) have changed the landscape of NLP by enabling more accessible and automated text analysis (e.g., topic detection, summarization, information extraction, etc.). We introduce VIDEE, a system that supports entry-level data analysts to conduct advanced text analytics with intelligent agents. VIDEE instantiates a human-agent collaroration workflow consisting of three stages: (1) Decomposition, which incorporates a human-in-the-loop Monte-Carlo Tree Search algorithm to support generative reasoning with human feedback, (2) Execution, which generates an executable text analytics pipeline, and (3) Evaluation, which integrates LLM-based evaluation and visualizations to support user validation of execution results. We conduct two quantitative experiments to evaluate VIDEE's effectiveness and analyze common agent errors. A user study involving participants with varying levels of NLP and text analytics experience -- from none to expert -- demonstrates the system's usability and reveals distinct user behavior patterns. The findings identify design implications for human-agent collaboration, validate the practical utility of VIDEE for non-expert users, and inform future improvements to intelligent text analytics systems. 

---
# LLM2Rec: Large Language Models Are Powerful Embedding Models for Sequential Recommendation 

**Authors**: Yingzhi He, Xiaohao Liu, An Zhang, Yunshan Ma, Tat-Seng Chua  

**Link**: [PDF](https://arxiv.org/pdf/2506.21579)  

**Abstract**: Sequential recommendation aims to predict users' future interactions by modeling collaborative filtering (CF) signals from historical behaviors of similar users or items. Traditional sequential recommenders predominantly rely on ID-based embeddings, which capture CF signals through high-order co-occurrence patterns. However, these embeddings depend solely on past interactions, lacking transferable knowledge to generalize to unseen domains. Recent advances in large language models (LLMs) have motivated text-based recommendation approaches that derive item representations from textual descriptions. While these methods enhance generalization, they fail to encode CF signals-i.e., latent item correlations and preference patterns-crucial for effective recommendation. We argue that an ideal embedding model should seamlessly integrate CF signals with rich semantic representations to improve both in-domain and out-of-domain recommendation performance.
To this end, we propose LLM2Rec, a novel embedding model tailored for sequential recommendation, integrating the rich semantic understanding of LLMs with CF awareness. Our approach follows a two-stage training framework: (1) Collaborative Supervised Fine-tuning, which adapts LLMs to infer item relationships based on historical interactions, and (2) Item-level Embedding Modeling, which refines these specialized LLMs into structured item embedding models that encode both semantic and collaborative information. Extensive experiments on real-world datasets demonstrate that LLM2Rec effectively improves recommendation quality across both in-domain and out-of-domain settings. Our findings highlight the potential of leveraging LLMs to build more robust, generalizable embedding models for sequential recommendation. Our codes are available at this https URL. 

---
# Evaluating Multimodal Large Language Models on Educational Textbook Question Answering 

**Authors**: Hessa A. Alawwad, Anas Zafar, Areej Alhothali, Usman Naseem, Ali Alkhathlan, Amani Jamal  

**Link**: [PDF](https://arxiv.org/pdf/2506.21596)  

**Abstract**: Multimodal large language models (MLLMs) have recently achieved significant success in vision--language tasks. However, their capacity to reason over complex, long lessons and intricate educational diagrams that cannot be represented as a single natural image remains largely untested. In this work, we present the first evaluation of state-of-the-art MLLMs on the textbook question answering (TQA) task using the CK12-QA dataset. We assess the performance of recent vision-language models, including LLaVA and LLaMA 3.2-Vision, across various input configurations. Additionally, we introduce a lightweight multimodal retrieval-augmented generation (RAG) pipeline that integrates both paragraphs and diagrams from the lesson into the prompt. Our results demonstrate the influence of retrieved educational context on model accuracy and reasoning, while also revealing current limitations in handling question-context relationships and the potential for noise, pointing to key directions for future research in multimodal AI-driven learning. 

---
# Refine Medical Diagnosis Using Generation Augmented Retrieval and Clinical Practice Guidelines 

**Authors**: Wenhao Li, Hongkuan Zhang, Hongwei Zhang, Zhengxu Li, Zengjie Dong, Yafan Chen, Niranjan Bidargaddi, Hong Liu  

**Link**: [PDF](https://arxiv.org/pdf/2506.21615)  

**Abstract**: Current medical language models, adapted from large language models (LLMs), typically predict ICD code-based diagnosis from electronic health records (EHRs) because these labels are readily available. However, ICD codes do not capture the nuanced, context-rich reasoning clinicians use for diagnosis. Clinicians synthesize diverse patient data and reference clinical practice guidelines (CPGs) to make evidence-based decisions. This misalignment limits the clinical utility of existing models. We introduce GARMLE-G, a Generation-Augmented Retrieval framework that grounds medical language model outputs in authoritative CPGs. Unlike conventional Retrieval-Augmented Generation based approaches, GARMLE-G enables hallucination-free outputs by directly retrieving authoritative guideline content without relying on model-generated text. It (1) integrates LLM predictions with EHR data to create semantically rich queries, (2) retrieves relevant CPG knowledge snippets via embedding similarity, and (3) fuses guideline content with model output to generate clinically aligned recommendations. A prototype system for hypertension diagnosis was developed and evaluated on multiple metrics, demonstrating superior retrieval precision, semantic relevance, and clinical guideline adherence compared to RAG-based baselines, while maintaining a lightweight architecture suitable for localized healthcare deployment. This work provides a scalable, low-cost, and hallucination-free method for grounding medical language models in evidence-based clinical practice, with strong potential for broader clinical deployment. 

---
# HealthQA-BR: A System-Wide Benchmark Reveals Critical Knowledge Gaps in Large Language Models 

**Authors**: Andrew Maranhão Ventura D'addario  

**Link**: [PDF](https://arxiv.org/pdf/2506.21578)  

**Abstract**: The evaluation of Large Language Models (LLMs) in healthcare has been dominated by physician-centric, English-language benchmarks, creating a dangerous illusion of competence that ignores the interprofessional nature of patient care. To provide a more holistic and realistic assessment, we introduce HealthQA-BR, the first large-scale, system-wide benchmark for Portuguese-speaking healthcare. Comprising 5,632 questions from Brazil's national licensing and residency exams, it uniquely assesses knowledge not only in medicine and its specialties but also in nursing, dentistry, psychology, social work, and other allied health professions. We conducted a rigorous zero-shot evaluation of over 20 leading LLMs. Our results reveal that while state-of-the-art models like GPT 4.1 achieve high overall accuracy (86.6%), this top-line score masks alarming, previously unmeasured deficiencies. A granular analysis shows performance plummets from near-perfect in specialties like Ophthalmology (98.7%) to barely passing in Neurosurgery (60.0%) and, most notably, Social Work (68.4%). This "spiky" knowledge profile is a systemic issue observed across all models, demonstrating that high-level scores are insufficient for safety validation. By publicly releasing HealthQA-BR and our evaluation suite, we provide a crucial tool to move beyond single-score evaluations and toward a more honest, granular audit of AI readiness for the entire healthcare team. 

---
# Instruction Learning Paradigms: A Dual Perspective on White-box and Black-box LLMs 

**Authors**: Yanwei Ren, Liu Liu, Baosheng Yu, Jiayan Qiu, Quan Chen  

**Link**: [PDF](https://arxiv.org/pdf/2506.21573)  

**Abstract**: Optimizing instructions for large language models (LLMs) is critical for harnessing their full potential in complex and diverse tasks. However, relying solely on white-box approaches demands extensive computational resources and offers limited representational capacity, while black-box models can incur prohibitive financial costs. To address these challenges, we introduce a novel framework that seamlessly merges the strengths of both paradigms. Black-box models provide high-quality, diverse instruction initializations, and white-box models supply fine-grained interpretability through hidden states and output features. By enforcing a semantic similarity constraint, these components fuse into a unified high-dimensional representation that captures deep semantic and structural nuances, enabling an iterative optimization process to refine instruction quality and adaptability. Extensive evaluations across a broad spectrum of tasks-ranging from complex reasoning to cross-lingual generalization-demonstrate that our approach consistently outperforms state-of-the-art baselines. This fusion of black-box initialization with advanced semantic refinement yields a scalable and efficient solution, paving the way for next-generation LLM-driven applications in diverse real-world scenarios. The source code will be released soon. 

---
# Empirical Evidence for Alignment Faking in Small LLMs and Prompt-Based Mitigation Techniques 

**Authors**: J. Koorndijk  

**Link**: [PDF](https://arxiv.org/pdf/2506.21584)  

**Abstract**: Current literature suggests that alignment faking (deceptive alignment) is an emergent property of large language models. We present the first empirical evidence that a small instruction-tuned model, specifically LLaMA 3 8B, can also exhibit alignment faking. We further show that prompt-only interventions, including deontological moral framing and scratchpad reasoning, significantly reduce this behavior without modifying model internals. This challenges the assumption that prompt-based ethics are trivial and that deceptive alignment requires scale. We introduce a taxonomy distinguishing shallow deception, shaped by context and suppressible through prompting, from deep deception, which reflects persistent, goal-driven misalignment. Our findings refine the understanding of deception in language models and underscore the need for alignment evaluations across model sizes and deployment settings. 

---
# Towards Understanding the Cognitive Habits of Large Reasoning Models 

**Authors**: Jianshuo Dong, Yujia Fu, Chuanrui Hu, Chao Zhang, Han Qiu  

**Link**: [PDF](https://arxiv.org/pdf/2506.21571)  

**Abstract**: Large Reasoning Models (LRMs), which autonomously produce a reasoning Chain of Thought (CoT) before producing final responses, offer a promising approach to interpreting and monitoring model behaviors. Inspired by the observation that certain CoT patterns -- e.g., ``Wait, did I miss anything?'' -- consistently emerge across tasks, we explore whether LRMs exhibit human-like cognitive habits. Building on Habits of Mind, a well-established framework of cognitive habits associated with successful human problem-solving, we introduce CogTest, a principled benchmark designed to evaluate LRMs' cognitive habits. CogTest includes 16 cognitive habits, each instantiated with 25 diverse tasks, and employs an evidence-first extraction method to ensure reliable habit identification. With CogTest, we conduct a comprehensive evaluation of 16 widely used LLMs (13 LRMs and 3 non-reasoning ones). Our findings reveal that LRMs, unlike conventional LLMs, not only exhibit human-like habits but also adaptively deploy them according to different tasks. Finer-grained analyses further uncover patterns of similarity and difference in LRMs' cognitive habit profiles, particularly certain inter-family similarity (e.g., Qwen-3 models and DeepSeek-R1). Extending the study to safety-related tasks, we observe that certain habits, such as Taking Responsible Risks, are strongly associated with the generation of harmful responses. These findings suggest that studying persistent behavioral patterns in LRMs' CoTs is a valuable step toward deeper understanding of LLM misbehavior. The code is available at: this https URL. 

---
# CORE-KG: An LLM-Driven Knowledge Graph Construction Framework for Human Smuggling Networks 

**Authors**: Dipak Meher, Carlotta Domeniconi, Guadalupe Correa-Cabrera  

**Link**: [PDF](https://arxiv.org/pdf/2506.21607)  

**Abstract**: Human smuggling networks are increasingly adaptive and difficult to analyze. Legal case documents offer valuable insights but are unstructured, lexically dense, and filled with ambiguous or shifting references-posing challenges for automated knowledge graph (KG) construction. Existing KG methods often rely on static templates and lack coreference resolution, while recent LLM-based approaches frequently produce noisy, fragmented graphs due to hallucinations, and duplicate nodes caused by a lack of guided extraction. We propose CORE-KG, a modular framework for building interpretable KGs from legal texts. It uses a two-step pipeline: (1) type-aware coreference resolution via sequential, structured LLM prompts, and (2) entity and relationship extraction using domain-guided instructions, built on an adapted GraphRAG framework. CORE-KG reduces node duplication by 33.28%, and legal noise by 38.37% compared to a GraphRAG-based baseline-resulting in cleaner and more coherent graph structures. These improvements make CORE-KG a strong foundation for analyzing complex criminal networks. 

---
# BioPars: A Pretrained Biomedical Large Language Model for Persian Biomedical Text Mining 

**Authors**: Baqer M. Merzah, Tania Taami, Salman Asoudeh, Amir reza Hossein pour, Saeed Mirzaee, Amir Ali Bengari  

**Link**: [PDF](https://arxiv.org/pdf/2506.21567)  

**Abstract**: Large Language Models (LLMs) have recently gained attention in the life sciences due to their capacity to model, extract, and apply complex biological information. Beyond their classical use as chatbots, these systems are increasingly used for complex analysis and problem-solving in specialized fields, including bioinformatics. First, we introduce BIOPARS-BENCH, a dataset from over 10,000 scientific articles, textbooks, and medical websites. BioParsQA was also introduced to evaluate the proposed model, which consists of 5,231 Persian medical questions and answers. This study then introduces BioPars, a simple but accurate measure designed to assess LLMs for three main abilities: acquiring subject-specific knowledge, interpreting and synthesizing such knowledge, and demonstrating proper evidence. Comparing ChatGPT, Llama, and Galactica, our study highlights their ability to remember and retrieve learned knowledge but also reveals shortcomings in addressing higher-level, real-world questions and fine-grained inferences. These findings indicate the need for further fine-tuning to address the capabilities of LLM in bioinformatics tasks. To our knowledge, BioPars is the first application of LLM in Persian medical QA, especially for generating long answers. Evaluation of four selected medical QA datasets shows that BioPars has achieved remarkable results compared to comparative approaches. The model on BioParsQA achieved a ROUGE-L score of 29.99, which is an improvement over GPT-4 1.0. The model achieved a BERTScore of 90.87 with the MMR method. The MoverScore and BLEURT values were also higher in this model than the other three models. In addition, the reported scores for the model are MoverScore=60.43 and BLEURT=50.78. BioPars is an ongoing project and all resources related to its development will be made available via the following GitHub repository: this https URL. 

---
# STRuCT-LLM: Unifying Tabular and Graph Reasoning with Reinforcement Learning for Semantic Parsing 

**Authors**: Josefa Lia Stoisser, Marc Boubnovski Martell, Lawrence Phillips, Casper Hansen, Julien Fauqueur  

**Link**: [PDF](https://arxiv.org/pdf/2506.21575)  

**Abstract**: We propose STRuCT-LLM, a unified framework for training large language models (LLMs) to perform structured reasoning over both relational and graph-structured data. Our approach jointly optimizes Text-to-SQL and Text-to-Cypher tasks using reinforcement learning (RL) combined with Chain-of-Thought (CoT) supervision. To support fine-grained optimization in graph-based parsing, we introduce a topology-aware reward function based on graph edit distance. Unlike prior work that treats relational and graph formalisms in isolation, STRuCT-LLM leverages shared abstractions between SQL and Cypher to induce cross-formalism transfer, enabling SQL training to improve Cypher performance and vice versa - even without shared schemas. Our largest model (QwQ-32B) achieves substantial relative improvements across tasks: on semantic parsing, Spider improves by 13.5\% and Text2Cypher by 73.1\%. The model also demonstrates strong zero-shot generalization, improving performance on downstream tabular QA (TableBench: 8.5\%) and knowledge graph QA (CR-LT-KGQA: 1.7\%) without any QA-specific supervision. These results demonstrate both the effectiveness of executable queries as scaffolds for structured reasoning and the synergistic benefits of jointly training on SQL and Cypher (code available at this https URL). 

---
# PEACE: Empowering Geologic Map Holistic Understanding with MLLMs 

**Authors**: Yangyu Huang, Tianyi Gao, Haoran Xu, Qihao Zhao, Yang Song, Zhipeng Gui, Tengchao Lv, Hao Chen, Lei Cui, Scarlett Li, Furu Wei  

**Link**: [PDF](https://arxiv.org/pdf/2501.06184)  

**Abstract**: Geologic map, as a fundamental diagram in geology science, provides critical insights into the structure and composition of Earth's subsurface and surface. These maps are indispensable in various fields, including disaster detection, resource exploration, and civil engineering. Despite their significance, current Multimodal Large Language Models (MLLMs) often fall short in geologic map understanding. This gap is primarily due to the challenging nature of cartographic generalization, which involves handling high-resolution map, managing multiple associated components, and requiring domain-specific knowledge. To quantify this gap, we construct GeoMap-Bench, the first-ever benchmark for evaluating MLLMs in geologic map understanding, which assesses the full-scale abilities in extracting, referring, grounding, reasoning, and analyzing. To bridge this gap, we introduce GeoMap-Agent, the inaugural agent designed for geologic map understanding, which features three modules: Hierarchical Information Extraction (HIE), Domain Knowledge Injection (DKI), and Prompt-enhanced Question Answering (PEQA). Inspired by the interdisciplinary collaboration among human scientists, an AI expert group acts as consultants, utilizing a diverse tool pool to comprehensively analyze questions. Through comprehensive experiments, GeoMap-Agent achieves an overall score of 0.811 on GeoMap-Bench, significantly outperforming 0.369 of GPT-4o. Our work, emPowering gEologic mAp holistiC undErstanding (PEACE) with MLLMs, paves the way for advanced AI applications in geology, enhancing the efficiency and accuracy of geological investigations. 

---
# Hybrid-NL2SVA: Integrating RAG and Finetuning for LLM-based NL2SVA 

**Authors**: Weihua Xiao, Derek Ekberg, Siddharth Garg, Ramesh Karri  

**Link**: [PDF](https://arxiv.org/pdf/2506.21569)  

**Abstract**: SystemVerilog Assertions (SVAs) are critical for verifying the correctness of hardware designs, but manually writing them from natural language property descriptions, i.e., NL2SVA, remains a labor-intensive and error-prone task. Recent advances in large language models (LLMs) offer opportunities to automate this translation. However, existing models still struggle with understanding domain-specific syntax and semantics. To enhance LLM performance in NL2SVA, we propose a customized retrieval-augmented generation (RAG) framework and a synthetic fine-tuning dataset that together improve LLM's performance. To further improve lightweight models over NL2SVA, our fine-tuning dataset provides prompt-guided explanations that teach LLMs the layer-by-layer construction process of concurrent SVAs, enabling supervised fine-tuning that greatly improves syntax and functionality accuracy. To evaluate the performance of LLMs over NL2SVA, we construct the largest evaluation dataset for NL2SVA, comprising 40 Verilog designs and 229 formally verified SVAs with detailed annotations. Experimental results show that our customized RAG framework increases the number of functionality matched SVAs by 58.42% over GPT-4o-mini, while Qwen2.5-Coder-7B-Instruct fine-tuned on our fine-tuning dataset and integrated with HybridRetrieval achieves a 59.05% over the base Qwen model. 

---
# FloorPlan-DeepSeek (FPDS): A multimodal approach to floorplan generation using vector-based next room prediction 

**Authors**: Jun Yin, Pengyu Zeng, Jing Zhong, Peilin Li, Miao Zhang, Ran Luo, Shuai Lu  

**Link**: [PDF](https://arxiv.org/pdf/2506.21562)  

**Abstract**: In the architectural design process, floor plan generation is inherently progressive and iterative. However, existing generative models for floor plans are predominantly end-to-end generation that produce an entire pixel-based layout in a single pass. This paradigm is often incompatible with the incremental workflows observed in real-world architectural practice. To address this issue, we draw inspiration from the autoregressive 'next token prediction' mechanism commonly used in large language models, and propose a novel 'next room prediction' paradigm tailored to architectural floor plan modeling. Experimental evaluation indicates that FPDS demonstrates competitive performance in comparison to diffusion models and Tell2Design in the text-to-floorplan task, indicating its potential applicability in supporting future intelligent architectural design. 

---
# From General Reasoning to Domain Expertise: Uncovering the Limits of Generalization in Large Language Models 

**Authors**: Dana Alsagheer, Yang Lu, Abdulrahman Kamal, Omar Kamal, Mohammad Kamal, Nada Mansour, Cosmo Yang Wu, Rambiba Karanjai, Sen Li, Weidong Shi  

**Link**: [PDF](https://arxiv.org/pdf/2506.21580)  

**Abstract**: Recent advancements in Large Language Models (LLMs) have demonstrated remarkable capabilities in various domains. However, effective decision-making relies heavily on strong reasoning abilities. Reasoning is the foundation for decision-making, providing the analytical and logical framework to make sound choices. Reasoning involves analyzing information, drawing inferences, and reaching conclusions based on logic or evidence. Decision-making builds on this foundation by applying the insights from reasoning to select the best course of action among alternatives. Together, these processes create a continuous cycle of thought and action aimed at achieving goals effectively. As AI technology evolves, there is a growing trend to train LLMs to excel in general reasoning. This study explores how the general reasoning capabilities of LLMs connect to their performance in domain-specific reasoning tasks. 

---
# Digital Gatekeepers: Exploring Large Language Model's Role in Immigration Decisions 

**Authors**: Yicheng Mao, Yang Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2506.21574)  

**Abstract**: With globalization and increasing immigrant populations, immigration departments face significant work-loads and the challenge of ensuring fairness in decision-making processes. Integrating artificial intelligence offers a promising solution to these challenges. This study investigates the potential of large language models (LLMs),such as GPT-3.5 and GPT-4, in supporting immigration decision-making. Utilizing a mixed-methods approach,this paper conducted discrete choice experiments and in-depth interviews to study LLM decision-making strategies and whether they are fair. Our findings demonstrate that LLMs can align their decision-making with human strategies, emphasizing utility maximization and procedural fairness. Meanwhile, this paper also reveals that while ChatGPT has safeguards to prevent unintentional discrimination, it still exhibits stereotypes and biases concerning nationality and shows preferences toward privileged group. This dual analysis highlights both the potential and limitations of LLMs in automating and enhancing immigration decisions. 

---
# Reasoning Isn't Enough: Examining Truth-Bias and Sycophancy in LLMs 

**Authors**: Emilio Barkett, Olivia Long, Madhavendra Thakur  

**Link**: [PDF](https://arxiv.org/pdf/2506.21561)  

**Abstract**: Despite their widespread use in fact-checking, moderation, and high-stakes decision-making, large language models (LLMs) remain poorly understood as judges of truth. This study presents the largest evaluation to date of LLMs' veracity detection capabilities and the first analysis of these capabilities in reasoning models. We had eight LLMs make 4,800 veracity judgments across several prompts, comparing reasoning and non-reasoning models. We find that rates of truth-bias, or the likelihood to believe a statement is true, regardless of whether it is actually true, are lower in reasoning models than in non-reasoning models, but still higher than human benchmarks. Most concerning, we identify sycophantic tendencies in several advanced models (o4-mini and GPT-4.1 from OpenAI, R1 from DeepSeek), which displayed an asymmetry in detection accuracy, performing well in truth accuracy but poorly in deception accuracy. This suggests that capability advances alone do not resolve fundamental veracity detection challenges in LLMs. 

---
# Training Language Model to Critique for Better Refinement 

**Authors**: Tianshu Yu, Chao Xiang, Mingchuan Yang, Pei Ke, Bosi Wen, Cunxiang Wang, Jiale Cheng, Li Zhang, Xinyu Mu, Chuxiong Sun, Minlie Huang  

**Link**: [PDF](https://arxiv.org/pdf/2506.22157)  

**Abstract**: Large language models (LLMs) have demonstrated remarkable evaluation and critique capabilities, providing insightful feedback and identifying flaws in various tasks. However, limited research has explored which types of critiques are most effective for improving model responses or how to generate such critiques. To address this gap, we introduce \textbf{R}efinement-oriented \textbf{C}ritique \textbf{O}ptimization (RCO), a novel framework designed to train critic models using refinement signals. RCO uses a feedback loop where critiques, generated by the critic model, guide the actor model in refining its responses. The critique utility (CU) quantifies the effectiveness of these refinements, serving as the reward signal for training the critic model. By focusing on critiques that lead to better refinements, RCO eliminates the need for direct critique preference assessment, ensuring that critiques driving meaningful improvements are rewarded. We evaluate RCO across five tasks, i.e., dialog generation, summarization, question answering, mathematical reasoning, and code generation, and show that it significantly outperforms traditional methods and open-source models in terms of critique quality and refinement outcomes. Our contributions include the introduction of RCO, a novel supervision scheme based on refined response preferences, and comprehensive experimental results that highlight the method's effectiveness in enhancing LLM critique-refinement loops. 

---
# Refining Czech GEC: Insights from a Multi-Experiment Approach 

**Authors**: Petr Pechman, Milan Straka, Jana Straková, Jakub Náplava  

**Link**: [PDF](https://arxiv.org/pdf/2506.22402)  

**Abstract**: We present a grammar error correction (GEC) system that achieves state of the art for the Czech language. Our system is based on a neural network translation approach with the Transformer architecture, and its key feature is its real-time synthetic generation pipeline, which dynamically augments sentences with artificial errors by introducing both language-agnostic and Czech-specific errors. We conduct a comprehensive series of experiments, investigating the Czech GEC corpora as bases for synthetic error introduction, several error generation strategies, domain balancing, tokenization granularity, model size, and data scaling during fine-tuning. Additionally, we evaluate the performance of large language models (LLMs) on Czech GEC in both end-user and expert fine-tuning scenarios. Our best-performing model is superior both in performance and computational efficiency. The source code and the trained model links are available on this https URL. 

---
# Lost at the Beginning of Reasoning 

**Authors**: Baohao Liao, Xinyi Chen, Sara Rajaee, Yuhui Xu, Christian Herold, Anders Søgaard, Maarten de Rijke, Christof Monz  

**Link**: [PDF](https://arxiv.org/pdf/2506.22058)  

**Abstract**: Recent advancements in large language models (LLMs) have significantly advanced complex reasoning capabilities, particularly through extended chain-of-thought (CoT) reasoning that incorporates mechanisms such as backtracking, self-reflection and self-correction. Despite these developments, the self-correction abilities of LLMs during long CoT reasoning remain underexplored. And recent findings on overthinking suggest that such models often engage in unnecessarily redundant reasoning. In this work, we empirically show that the first reasoning step exerts a disproportionately large influence on the final prediction - errors introduced at this stage can substantially degrade subsequent reasoning quality. This phenomenon is consistently observed across two state-of-the-art open-source reasoning model families: DeepSeek-R1 and Qwen3. To address this, we propose an efficient sampling strategy that leverages a reward model to identify and retain high-quality first reasoning steps while discarding suboptimal ones, achieving up to a 70% reduction in inference cost without sacrificing accuracy. Finally, we introduce a new benchmark specifically constructed with deliberately flawed first reasoning steps to systematically evaluate model self-correction capabilities, offering a foundation for future research on robust reasoning in LLMs. 

---
# Can Peter Pan Survive MT? A Stylometric Study of LLMs, NMTs, and HTs in Children's Literature Translation 

**Authors**: Delu Kong, Lieve Macken  

**Link**: [PDF](https://arxiv.org/pdf/2506.22038)  

**Abstract**: This study focuses on evaluating the performance of machine translations (MTs) compared to human translations (HTs) in English-to-Chinese children's literature translation (CLT) from a stylometric perspective. The research constructs a Peter Pan corpus, comprising 21 translations: 7 human translations (HTs), 7 large language model translations (LLMs), and 7 neural machine translation outputs (NMTs). The analysis employs a generic feature set (including lexical, syntactic, readability, and n-gram features) and a creative text translation (CTT-specific) feature set, which captures repetition, rhythm, translatability, and miscellaneous levels, yielding 447 linguistic features in total.
Using classification and clustering techniques in machine learning, we conduct a stylometric analysis of these translations. Results reveal that in generic features, HTs and MTs exhibit significant differences in conjunction word distributions and the ratio of 1-word-gram-YiYang, while NMTs and LLMs show significant variation in descriptive words usage and adverb ratios. Regarding CTT-specific features, LLMs outperform NMTs in distribution, aligning more closely with HTs in stylistic characteristics, demonstrating the potential of LLMs in CLT. 

---
# Evaluating Scoring Bias in LLM-as-a-Judge 

**Authors**: Qingquan Li, Shaoyu Dou, Kailai Shao, Chao Chen, Haixiang Hu  

**Link**: [PDF](https://arxiv.org/pdf/2506.22316)  

**Abstract**: The remarkable performance of Large Language Models (LLMs) gives rise to``LLM-as-a-Judge'', where LLMs are employed as evaluators for complex tasks. Moreover, it has been widely adopted across fields such as Natural Language Processing (NLP), preference learning, and various specific domains. However, there are various biases within LLM-as-a-Judge, which adversely affect the fairness and reliability of judgments. Current research on evaluating or mitigating bias in LLM-as-a-Judge predominantly focuses on comparison-based evaluations, while systematic investigations into bias in scoring-based evaluations remain limited. Therefore, we define scoring bias in LLM-as-a-Judge as the scores differ when scoring judge models are bias-related perturbed, and provide a well-designed framework to comprehensively evaluate scoring bias. We augment existing LLM-as-a-Judge benchmarks through data synthesis to construct our evaluation dataset and design multi-faceted evaluation metrics. Our experimental results demonstrate that the scoring stability of existing judge models is disrupted by scoring biases. Further exploratory experiments and discussions provide valuable insights into the design of scoring prompt templates and the mitigation of scoring biases on aspects such as score rubrics, score IDs, and reference answer selection. 

---
# Don't Trust Generative Agents to Mimic Communication on Social Networks Unless You Benchmarked their Empirical Realism 

**Authors**: Simon Münker, Nils Schwager, Achim Rettinger  

**Link**: [PDF](https://arxiv.org/pdf/2506.21974)  

**Abstract**: The ability of Large Language Models (LLMs) to mimic human behavior triggered a plethora of computational social science research, assuming that empirical studies of humans can be conducted with AI agents instead. Since there have been conflicting research findings on whether and when this hypothesis holds, there is a need to better understand the differences in their experimental designs. We focus on replicating the behavior of social network users with the use of LLMs for the analysis of communication on social networks. First, we provide a formal framework for the simulation of social networks, before focusing on the sub-task of imitating user communication. We empirically test different approaches to imitate user behavior on X in English and German. Our findings suggest that social simulations should be validated by their empirical realism measured in the setting in which the simulation components were fitted. With this paper, we argue for more rigor when applying generative-agent-based modeling for social simulation. 

---
# Leveraging In-Context Learning for Political Bias Testing of LLMs 

**Authors**: Patrick Haller, Jannis Vamvas, Rico Sennrich, Lena A. Jäger  

**Link**: [PDF](https://arxiv.org/pdf/2506.22232)  

**Abstract**: A growing body of work has been querying LLMs with political questions to evaluate their potential biases. However, this probing method has limited stability, making comparisons between models unreliable. In this paper, we argue that LLMs need more context. We propose a new probing task, Questionnaire Modeling (QM), that uses human survey data as in-context examples. We show that QM improves the stability of question-based bias evaluation, and demonstrate that it may be used to compare instruction-tuned models to their base versions. Experiments with LLMs of various sizes indicate that instruction tuning can indeed change the direction of bias. Furthermore, we observe a trend that larger models are able to leverage in-context examples more effectively, and generally exhibit smaller bias scores in QM. Data and code are publicly available. 

---
# More Vulnerable than You Think: On the Stability of Tool-Integrated LLM Agents 

**Authors**: Weimin Xiong, Ke Wang, Yifan Song, Hanchao Liu, Sai Zhou, Wei Peng, Sujian Li  

**Link**: [PDF](https://arxiv.org/pdf/2506.21967)  

**Abstract**: Current evaluations of tool-integrated LLM agents typically focus on end-to-end tool-usage evaluation while neglecting their stability. This limits their real-world applicability, as various internal or external factors can cause agents to crash or behave abnormally. Our research addresses this by investigating whether agents are vulnerable to errors throughout the entire tool invocation process, including reading tool documentation, selecting tools and generating parameters, and processing the tool's response. Through extensive experiments, we observe that agents are highly susceptible to errors at each stage and agents based on open-source models are more vulnerable than those based on proprietary models. We also find that increasing the model size does not significantly improve tool invocation reasoning and may make agents more vulnerable to attacks resembling normal user instructions. This highlights the importance of evaluating agent stability and offers valuable insights for future LLM development and evaluation. 

---
# A Dual-Layered Evaluation of Geopolitical and Cultural Bias in LLMs 

**Authors**: Sean Kim, Hyuhng Joon Kim  

**Link**: [PDF](https://arxiv.org/pdf/2506.21881)  

**Abstract**: As large language models (LLMs) are increasingly deployed across diverse linguistic and cultural contexts, understanding their behavior in both factual and disputable scenarios is essential, especially when their outputs may shape public opinion or reinforce dominant narratives. In this paper, we define two types of bias in LLMs: model bias (bias stemming from model training) and inference bias (bias induced by the language of the query), through a two-phase evaluation. Phase 1 evaluates LLMs on factual questions where a single verifiable answer exists, assessing whether models maintain consistency across different query languages. Phase 2 expands the scope by probing geopolitically sensitive disputes, where responses may reflect culturally embedded or ideologically aligned perspectives. We construct a manually curated dataset spanning both factual and disputable QA, across four languages and question types. The results show that Phase 1 exhibits query language induced alignment, while Phase 2 reflects an interplay between the model's training context and query language. This paper offers a structured framework for evaluating LLM behavior across neutral and sensitive topics, providing insights for future LLM deployment and culturally aware evaluation practices in multilingual contexts. 

---
# Towards Transparent AI: A Survey on Explainable Large Language Models 

**Authors**: Avash Palikhe, Zhenyu Yu, Zichong Wang, Wenbin Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2506.21812)  

**Abstract**: Large Language Models (LLMs) have played a pivotal role in advancing Artificial Intelligence (AI). However, despite their achievements, LLMs often struggle to explain their decision-making processes, making them a 'black box' and presenting a substantial challenge to explainability. This lack of transparency poses a significant obstacle to the adoption of LLMs in high-stakes domain applications, where interpretability is particularly essential. To overcome these limitations, researchers have developed various explainable artificial intelligence (XAI) methods that provide human-interpretable explanations for LLMs. However, a systematic understanding of these methods remains limited. To address this gap, this survey provides a comprehensive review of explainability techniques by categorizing XAI methods based on the underlying transformer architectures of LLMs: encoder-only, decoder-only, and encoder-decoder models. Then these techniques are examined in terms of their evaluation for assessing explainability, and the survey further explores how these explanations are leveraged in practical applications. Finally, it discusses available resources, ongoing research challenges, and future directions, aiming to guide continued efforts toward developing transparent and responsible LLMs. 

---
# Decoding Machine Translationese in English-Chinese News: LLMs vs. NMTs 

**Authors**: Delu Kong, Lieve Macken  

**Link**: [PDF](https://arxiv.org/pdf/2506.22050)  

**Abstract**: This study explores Machine Translationese (MTese) -- the linguistic peculiarities of machine translation outputs -- focusing on the under-researched English-to-Chinese language pair in news texts. We construct a large dataset consisting of 4 sub-corpora and employ a comprehensive five-layer feature set. Then, a chi-square ranking algorithm is applied for feature selection in both classification and clustering tasks. Our findings confirm the presence of MTese in both Neural Machine Translation systems (NMTs) and Large Language Models (LLMs). Original Chinese texts are nearly perfectly distinguishable from both LLM and NMT outputs. Notable linguistic patterns in MT outputs are shorter sentence lengths and increased use of adversative conjunctions. Comparing LLMs and NMTs, we achieve approximately 70% classification accuracy, with LLMs exhibiting greater lexical diversity and NMTs using more brackets. Additionally, translation-specific LLMs show lower lexical diversity but higher usage of causal conjunctions compared to generic LLMs. Lastly, we find no significant differences between LLMs developed by Chinese firms and their foreign counterparts. 

---
# (Fact) Check Your Bias 

**Authors**: Eivind Morris Bakke, Nora Winger Heggelund  

**Link**: [PDF](https://arxiv.org/pdf/2506.21745)  

**Abstract**: Automatic fact verification systems increasingly rely on large language models (LLMs). We investigate how parametric knowledge biases in these models affect fact-checking outcomes of the HerO system (baseline for FEVER-25). We examine how the system is affected by: (1) potential bias in Llama 3.1's parametric knowledge and (2) intentionally injected bias. When prompted directly to perform fact-verification, Llama 3.1 labels nearly half the claims as "Not Enough Evidence". Using only its parametric knowledge it is able to reach a verdict on the remaining half of the claims. In the second experiment, we prompt the model to generate supporting, refuting, or neutral fact-checking documents. These prompts significantly influence retrieval outcomes, with approximately 50\% of retrieved evidence being unique to each perspective. Notably, the model sometimes refuses to generate supporting documents for claims it believes to be false, creating an inherent negative bias. Despite differences in retrieved evidence, final verdict predictions show stability across prompting strategies. The code is available at: this https URL 

---
# TIM: A Large-Scale Dataset and large Timeline Intelligence Model for Open-domain Timeline Summarization 

**Authors**: Chuanrui Hu, Wei Hu, Penghang Yu, Hua Zhang, Bing-Kun Bao  

**Link**: [PDF](https://arxiv.org/pdf/2506.21616)  

**Abstract**: Open-domain Timeline Summarization (TLS) is crucial for monitoring the evolution of news topics. To identify changes in news topics, existing methods typically employ general Large Language Models (LLMs) to summarize relevant timestamps from retrieved news. While general LLMs demonstrate capabilities in zero-shot news summarization and timestamp localization, they struggle with assessing topic relevance and understanding topic evolution. Consequently, the summarized information often includes irrelevant details or inaccurate timestamps. To address these issues, we propose the first large Timeline Intelligence Model (TIM) for open-domain TLS, which is capable of effectively summarizing open-domain timelines. Specifically, we begin by presenting a large-scale TLS dataset, comprising over 1,000 news topics and more than 3,000 annotated TLS instances. Furthermore, we propose a progressive optimization strategy, which gradually enhance summarization performance. It employs instruction tuning to enhance summarization and topic-irrelevant information filtering capabilities. Following this, it exploits a novel dual-alignment reward learning method that incorporates both semantic and temporal perspectives, thereby improving the understanding of topic evolution principles. Through this progressive optimization strategy, TIM demonstrates a robust ability to summarize open-domain timelines. Extensive experiments in open-domain demonstrate the effectiveness of our TIM. 

---
# ChildGuard: A Specialized Dataset for Combatting Child-Targeted Hate Speech 

**Authors**: Gautam Siddharth Kashyap, Mohammad Anas Azeez, Rafiq Ali, Zohaib Hasan Siddiqui, Jiechao Gao, Usman Naseem  

**Link**: [PDF](https://arxiv.org/pdf/2506.21613)  

**Abstract**: The increasing prevalence of child-targeted hate speech online underscores the urgent need for specialized datasets to address this critical issue. Existing hate speech datasets lack agespecific annotations, fail to capture nuanced contexts, and overlook the unique emotional impact on children. To bridge this gap, we introduce ChildGuard1, a curated dataset derived from existing corpora and enriched with child-specific annotations. ChildGuard captures diverse contexts of child-targeted hate speech, spanning age groups. We benchmark existing state-of-the-art hate speech detection methods, including Large Language Models (LLMs), and assess their effectiveness in detecting and contextualizing child-targeted hate speech. To foster further research in this area, we publicly release ChildGuard, providing a robust foundation for developing improved methods to detect and mitigate such harm. 

---
# Thunder-LLM: Efficiently Adapting LLMs to Korean with Minimal Resources 

**Authors**: Jinpyo Kim, Gyeongje Cho, Chanwoo Park, Jongwon Park, Jongmin Kim, Yeonkyoun So, Jaejin Lee  

**Link**: [PDF](https://arxiv.org/pdf/2506.21595)  

**Abstract**: Since state-of-the-art LLMs often underperform in languages other than English or Chinese, improving the capability of LLMs in new languages has become an essential task. Moreover, LLMs' entire end-to-end training process remains largely unknown to the public due to proprietary reasons, technical complexity, inconsistent documentation, and ethical considerations. The complete picture remains a closely guarded secret within the industry. This paper presents methods to adapt an existing English-based LLM to Korean in a low-budget scenario. We describe the entire end-to-end process: collecting Korean datasets, preprocessing the data, training the model, creating downstream benchmarks, and conducting evaluations. The evaluation results indicate that our method can effectively and cost-efficiently add new language capabilities to existing LLMs. Our new bilingual models, Thunder-LLM and Thunder-LLM-Ins, achieve superior Korean performance compared to state-of-the-art models while utilizing minimal data and computational resources. We share our comprehensive experience and make the code publicly available. 

---
# Representation Consistency for Accurate and Coherent LLM Answer Aggregation 

**Authors**: Junqi Jiang, Tom Bewley, Salim I. Amoukou, Francesco Leofante, Antonio Rago, Saumitra Mishra, Francesca Toni  

**Link**: [PDF](https://arxiv.org/pdf/2506.21590)  

**Abstract**: Test-time scaling improves large language models' (LLMs) performance by allocating more compute budget during inference. To achieve this, existing methods often require intricate modifications to prompting and sampling strategies. In this work, we introduce representation consistency (RC), a test-time scaling method for aggregating answers drawn from multiple candidate responses of an LLM regardless of how they were generated, including variations in prompt phrasing and sampling strategy. RC enhances answer aggregation by not only considering the number of occurrences of each answer in the candidate response set, but also the consistency of the model's internal activations while generating the set of responses leading to each answer. These activations can be either dense (raw model activations) or sparse (encoded via pretrained sparse autoencoders). Our rationale is that if the model's representations of multiple responses converging on the same answer are highly variable, this answer is more likely to be the result of incoherent reasoning and should be down-weighted during aggregation. Importantly, our method only uses cached activations and lightweight similarity computations and requires no additional model queries. Through experiments with four open-source LLMs and four reasoning datasets, we validate the effectiveness of RC for improving task performance during inference, with consistent accuracy improvements (up to 4%) over strong test-time scaling baselines. We also show that consistency in the sparse activation signals aligns well with the common notion of coherent reasoning. 

---
# Is DeepSeek a New Voice Among LLMs in Public Opinion Simulation? 

**Authors**: Weihong Qi, Fan Huang, Jisun An, Haewoon Kwak  

**Link**: [PDF](https://arxiv.org/pdf/2506.21587)  

**Abstract**: This study evaluates the ability of DeepSeek, an open-source large language model (LLM), to simulate public opinions in comparison to LLMs developed by major tech companies. By comparing DeepSeek-R1 and DeepSeek-V3 with Qwen2.5, GPT-4o, and Llama-3.3 and utilizing survey data from the American National Election Studies (ANES) and the Zuobiao dataset of China, we assess these models' capacity to predict public opinions on social issues in both China and the United States, highlighting their comparative capabilities between countries. Our findings indicate that DeepSeek-V3 performs best in simulating U.S. opinions on the abortion issue compared to other topics such as climate change, gun control, immigration, and services for same-sex couples, primarily because it more accurately simulates responses when provided with Democratic or liberal personas. For Chinese samples, DeepSeek-V3 performs best in simulating opinions on foreign aid and individualism but shows limitations in modeling views on capitalism, particularly failing to capture the stances of low-income and non-college-educated individuals. It does not exhibit significant differences from other models in simulating opinions on traditionalism and the free market. Further analysis reveals that all LLMs exhibit the tendency to overgeneralize a single perspective within demographic groups, often defaulting to consistent responses within groups. These findings highlight the need to mitigate cultural and demographic biases in LLM-driven public opinion modeling, calling for approaches such as more inclusive training methodologies. 

---
# Evaluation of LLM-based Strategies for the Extraction of Food Product Information from Online Shops 

**Authors**: Christoph Brosch, Sian Brumm, Rolf Krieger, Jonas Scheffler  

**Link**: [PDF](https://arxiv.org/pdf/2506.21585)  

**Abstract**: Generative AI and large language models (LLMs) offer significant potential for automating the extraction of structured information from web pages. In this work, we focus on food product pages from online retailers and explore schema-constrained extraction approaches to retrieve key product attributes, such as ingredient lists and nutrition tables. We compare two LLM-based approaches, direct extraction and indirect extraction via generated functions, evaluating them in terms of accuracy, efficiency, and cost on a curated dataset of 3,000 food product pages from three different online shops. Our results show that although the indirect approach achieves slightly lower accuracy (96.48\%, $-1.61\%$ compared to direct extraction), it reduces the number of required LLM calls by 95.82\%, leading to substantial efficiency gains and lower operational costs. These findings suggest that indirect extraction approaches can provide scalable and cost-effective solutions for large-scale information extraction tasks from template-based web pages using LLMs. 

---
# Gazal-R1: Achieving State-of-the-Art Medical Reasoning with Parameter-Efficient Two-Stage Training 

**Authors**: Ahmed M. Adly, Mostafa Samy, Amr Fawzy  

**Link**: [PDF](https://arxiv.org/pdf/2506.21594)  

**Abstract**: We present Gazal-R1, a 32-billion-parameter language model that achieves state-of-the-art performance in medical reasoning while providing transparent, step-by-step explanations for clinical decision-making. Built upon Qwen3 32B, our model demonstrates that strategic training can enable mid-sized models to outperform significantly larger counterparts in specialized domains. We developed a novel two-stage training pipeline: first, supervised fine-tuning on a carefully curated dataset of 107,033 synthetic medical reasoning examples that teaches structured clinical thinking, enhanced by advanced parameter-efficient techniques including Weight-Decomposed Low-Rank Adaptation (DoRA) and Rank-Stabilized LoRA (rsLoRA); second, reinforcement learning using Group Relative Policy Optimization (GRPO) with a sophisticated multi-component reward system that refines accuracy, format adherence, and reasoning quality. Gazal-R1 achieves exceptional performance across medical benchmarks, scoring 87.1% on MedQA, 81.6% on MMLU Pro (Medical), and 79.6% on PubMedQA, surpassing models up to 12x larger. Beyond its strong empirical results, this work provides detailed insights into the challenges of training reasoning-capable models in specialized domains, including issues with reward hacking, training instability, and the fundamental tension between factual recall and detailed reasoning. Our methodology offers a reproducible framework for developing high-capability, domain-specific language models that balance performance, efficiency, and explainability. 

---
# FinEval-KR: A Financial Domain Evaluation Framework for Large Language Models' Knowledge and Reasoning 

**Authors**: Shaoyu Dou, Yutian Shen, Mofan Chen, Zixuan Wang, Jiajie Xu, Qi Guo, Kailai Shao, Chao Chen, Haixiang Hu, Haibo Shi, Min Min, Liwen Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2506.21591)  

**Abstract**: Large Language Models (LLMs) demonstrate significant potential but face challenges in complex financial reasoning tasks requiring both domain knowledge and sophisticated reasoning. Current evaluation benchmarks often fall short by not decoupling these capabilities indicators from single task performance and lack root cause analysis for task failure. To address this, we introduce FinEval-KR, a novel evaluation framework for decoupling and quantifying LLMs' knowledge and reasoning abilities independently, proposing distinct knowledge score and reasoning score metrics. Inspired by cognitive science, we further propose a cognitive score based on Bloom's taxonomy to analyze capabilities in reasoning tasks across different cognitive levels. We also release a new open-source Chinese financial reasoning dataset covering 22 subfields to support reproducible research and further advancements in financial reasoning. Our experimental results reveal that LLM reasoning ability and higher-order cognitive ability are the core factors influencing reasoning accuracy. We also specifically find that even top models still face a bottleneck with knowledge application. Furthermore, our analysis shows that specialized financial LLMs generally lag behind the top general large models across multiple metrics. 

---
# A Multi-Agent Probabilistic Inference Framework Inspired by Kairanban-Style CoT System with IdoBata Conversation for Debiasing 

**Authors**: Takato Ueno, Keito Inoshita  

**Link**: [PDF](https://arxiv.org/pdf/2506.21565)  

**Abstract**: Japan's kairanban culture and idobata conversations have long functioned as traditional communication practices that foster nuanced dialogue among community members and contribute to the formation of social balance. Inspired by these information exchange processes, this study proposes a multi-agent inference framework (KCS+IBC) that integrates multiple large language models (LLMs) to achieve bias mitigation, improved explainability, and probabilistic prediction in sentiment analysis. In addition to sequentially sharing prediction results, the proposed method incorporates a mid-phase casual dialogue session to blend formal inference with individual perspectives and introduces probabilistic sentiment prediction. Experimental results show that KCS achieves accuracy comparable to that of a single LLM across datasets, while KCS+IBC exhibits a consistent decrease in entropy and a gradual increase in variance during the latter stages of inference, suggesting the framework's ability to balance aggregation and diversity of predictions. Future work will quantitatively assess the impact of these characteristics on bias correction and aim to develop more advanced sentiment analysis systems. 

---
# Assessing RAG and HyDE on 1B vs. 4B-Parameter Gemma LLMs for Personal Assistants Integretion 

**Authors**: Andrejs Sorstkins  

**Link**: [PDF](https://arxiv.org/pdf/2506.21568)  

**Abstract**: Resource efficiency is a critical barrier to deploying large language models (LLMs) in edge and privacy-sensitive applications. This study evaluates the efficacy of two augmentation strategies--Retrieval-Augmented Generation (RAG) and Hypothetical Document Embeddings (HyDE)--on compact Gemma LLMs of 1 billion and 4 billion parameters, within the context of a privacy-first personal assistant. We implement short-term memory via MongoDB and long-term semantic storage via Qdrant, orchestrated through FastAPI and LangChain, and expose the system through a this http URL frontend. Across both model scales, RAG consistently reduces latency by up to 17\% and eliminates factual hallucinations when responding to user-specific and domain-specific queries. HyDE, by contrast, enhances semantic relevance--particularly for complex physics prompts--but incurs a 25--40\% increase in response time and a non-negligible hallucination rate in personal-data retrieval. Comparing 1 B to 4 B models, we observe that scaling yields marginal throughput gains for baseline and RAG pipelines, but magnifies HyDE's computational overhead and variability. Our findings position RAG as the pragmatic choice for on-device personal assistants powered by small-scale LLMs. 

---
# FormosanBench: Benchmarking Low-Resource Austronesian Languages in the Era of Large Language Models 

**Authors**: Kaiying Kevin Lin, Hsiyu Chen, Haopeng Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2506.21563)  

**Abstract**: While large language models (LLMs) have demonstrated impressive performance across a wide range of natural language processing (NLP) tasks in high-resource languages, their capabilities in low-resource and minority languages remain significantly underexplored. Formosan languages -- a subgroup of Austronesian languages spoken in Taiwan -- are both linguistically rich and endangered, largely due to the sociolinguistic dominance of Mandarin. In this work, we introduce FORMOSANBENCH, the first benchmark for evaluating LLMs on low-resource Austronesian languages. It covers three endangered Formosan languages: Atayal, Amis, and Paiwan, across three core NLP tasks: machine translation, automatic speech recognition (ASR), and text summarization. We assess model performance in zero-shot, 10-shot, and fine-tuned settings using FORMOSANBENCH. Our results reveal a substantial performance gap between high-resource and Formosan languages. Existing LLMs consistently underperform across all tasks, with 10-shot learning and fine-tuning offering only limited improvements. These findings underscore the urgent need for more inclusive NLP technologies that can effectively support endangered and underrepresented languages. We release our datasets and code to facilitate future research in this direction. 

---
# Aligning MLLM Benchmark With Human Preferences via Structural Equation Modeling 

**Authors**: Tianyu.Zou, Shengwu.Xiong, Ruilin.Yao, Jirui.Huang, Yi.Rong, Yaxiong.Chen, Shili.Xiong, Cong.Wang  

**Link**: [PDF](https://arxiv.org/pdf/2506.21572)  

**Abstract**: Evaluating multimodal large language models (MLLMs) remains a fundamental challenge due to a lack of structured, interpretable, and theoretically grounded benchmark designs. Existing benchmarks often adopt heuristic-based task groupings with unclear cognitive targets, thus resulting in overlapping abilities, redundant indicators, and limited diagnostic power. In this work, we propose a novel framework for aligning MLLM benchmark based on Structural Equation Modeling (SEM) to analyze and quantify the internal validity, dimensional separability, and contribution of benchmark components. Motivated by the observed limitations of current designs, we further introduce a novel capability hierarchy grounded in Piagets theory of cognitive development, dividing MLLM abilities into three hierarchical layers, i.e., Perception, Memory, and Reasoning. We reorganize existing MLLM benchmarks under the proposed framework and construct a new benchmark named Gold. Experimental results demonstrate that the proposed benchmark exhibits stronger interpretability, reduced indicator redundancy, and clearer cognitive consistency compared to existing approaches. 

---
# VAT-KG: Knowledge-Intensive Multimodal Knowledge Graph Dataset for Retrieval-Augmented Generation 

**Authors**: Hyeongcheol Park, MinHyuk Jang, Ha Dam Baek, Gyusam Chang, Jiyoung Seo, Jiwan Park, Hogun Park, Sangpil Kim  

**Link**: [PDF](https://arxiv.org/pdf/2506.21556)  

**Abstract**: Multimodal Knowledge Graphs (MMKGs), which represent explicit knowledge across multiple modalities, play a pivotal role by complementing the implicit knowledge of Multimodal Large Language Models (MLLMs) and enabling more grounded reasoning via Retrieval Augmented Generation (RAG). However, existing MMKGs are generally limited in scope: they are often constructed by augmenting pre-existing knowledge graphs, which restricts their knowledge, resulting in outdated or incomplete knowledge coverage, and they often support only a narrow range of modalities, such as text and visual information. These limitations reduce their extensibility and applicability to a broad range of multimodal tasks, particularly as the field shifts toward richer modalities such as video and audio in recent MLLMs. Therefore, we propose the Visual-Audio-Text Knowledge Graph (VAT-KG), the first concept-centric and knowledge-intensive multimodal knowledge graph that covers visual, audio, and text information, where each triplet is linked to multimodal data and enriched with detailed descriptions of concepts. Specifically, our construction pipeline ensures cross-modal knowledge alignment between multimodal data and fine-grained semantics through a series of stringent filtering and alignment steps, enabling the automatic generation of MMKGs from any multimodal dataset. We further introduce a novel multimodal RAG framework that retrieves detailed concept-level knowledge in response to queries from arbitrary modalities. Experiments on question answering tasks across various modalities demonstrate the effectiveness of VAT-KG in supporting MLLMs, highlighting its practical value in unifying and leveraging multimodal knowledge. 

---
# GraphLAMA: Enabling Efficient Adaptation of Graph Language Models with Limited Annotations 

**Authors**: Junze Chen, Cheng Yang, Shujie Li, Zhiqiang Zhang, Yawen Li, Junping Du, Chuan Shi  

**Link**: [PDF](https://arxiv.org/pdf/2506.21559)  

**Abstract**: Large language models (LLMs) have demonstrated their strong capabilities in various domains, and have been recently integrated for graph analysis as graph language models (GLMs). With LLMs as the predictor, some GLMs can interpret unseen tasks described by natural language, and learn from a few examples in the prompts without parameter tuning, known as in-context learning (ICL). Another subset of GLMs utilizes abundant training labels to enhance model performance, known as instruction tuning. However, we argue that ICL on graphs has effectiveness issues due to fixed parameters and efficiency issues due to long context. Meanwhile, the large amount of labeled data required for instruction tuning can be difficult to obtain in real-world scenarios. To this end, we aim to introduce an extra parameter adaptation stage that can efficiently tailor GLMs to an unseen graph and task with only a few labeled examples, in exchange for better prediction accuracy and faster inference speed. For implementation, in this paper we propose GraphLAMA method, with its model backbone and learning schemes specialized for efficient tuning and inference. Specifically, for model backbone, we use a graph neural network (GNN) with several well-designed components to transform nodes into the representation space of LLM tokens. Task instructions can then be represented as a mixture of node and language tokens. In the pre-training stage, model parameters except the LLM will be trained with different tasks to capture general knowledge. In the adaptation stage, only a few pre-trained parameters will be updated based on few-shot examples. Extensive experiments on few/zero-shot node classification and summary generation show that our proposed GraphLAMA achieves state-of-the-art performance with 4.91% absolution improvement in accuracy. Compared with ICL, our inference speed can be 10 times faster under 5-shot setting. 

---
# Reinforcement Learning Fine-Tuning of Language Model for Instruction Following and Math Reasoning 

**Authors**: Yifu Han, Geo Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2506.21560)  

**Abstract**: This study investigates the effectiveness of reinforcement learning (RL) fine-tuning techniques on a compact language model (Qwen2.5-0.5B Base) for two challenging tasks: instruction following and mathematical reasoning. We compare supervised fine-tuning (SFT), Direct Preference Optimization (DPO) using preference-labeled data, and Reinforce Leave-One-Out (RLOO) with reward models. Our experiments show that RLOO with DeBERTa reward modeling achieves the best alignment, while DPO provides strong and consistent results. For math reasoing tasks, synthetic data augmentation and best-of-N sampling with an external verifier significantly improve accuracy, showing the potential of combining fine-tuning with inference-time tools. This study highlights key trade-offs and practical strategies for training lightweight, task-aligned small-scale language models. 

---
# Exploring Modularity of Agentic Systems for Drug Discovery 

**Authors**: Laura van Weesep, Samuel Genheden, Ola Engkvist, Jens Sjölund  

**Link**: [PDF](https://arxiv.org/pdf/2506.22189)  

**Abstract**: Large-language models (LLMs) and agentic systems present exciting opportunities to accelerate drug discovery and design. In this study, we critically examine the modularity of LLM-based agentic systems for drug discovery, i.e., whether parts of the agentic system such as the LLM are interchangeable, a topic that has received limited attention in drug discovery applications. We compare the performance of different large language models (LLMs) and the effectiveness of tool-calling agents versus code-generating agents in this domain. Our case study, comparing performance in orchestrating tools for chemistry and drug discovery using an LLM-as-a-judge score, shows that Claude-3.5-Sonnet, Claude-3.7-Sonnet and GPT-4o outperform alternative language models such as Llama-3.1-8B, Llama-3.1-70B, GPT-3.5-Turbo, and Nova-Micro. Although we confirm that code-generating agents outperform the tool-calling ones on average, we show that this is highly question and model dependent. Furthermore, the impact of replacing system prompts is dependent on the specific question asked and the model used, underscoring that -- even in this particular domain -- one cannot just replace language models without considering prompt re-engineering. Our study highlights the necessity of further research into the modularity of agentic systems to enable the development of stable and scalable solutions for real-world problems. 

---
# Debunk and Infer: Multimodal Fake News Detection via Diffusion-Generated Evidence and LLM Reasoning 

**Authors**: Kaiying Yan, Moyang Liu, Yukun Liu, Ruibo Fu, Zhengqi Wen, Jianhua Tao, Xuefei Liu  

**Link**: [PDF](https://arxiv.org/pdf/2506.21557)  

**Abstract**: The rapid spread of fake news across multimedia platforms presents serious challenges to information credibility. In this paper, we propose a Debunk-and-Infer framework for Fake News Detection(DIFND) that leverages debunking knowledge to enhance both the performance and interpretability of fake news detection. DIFND integrates the generative strength of conditional diffusion models with the collaborative reasoning capabilities of multimodal large language models (MLLMs). Specifically, debunk diffusion is employed to generate refuting or authenticating evidence based on the multimodal content of news videos, enriching the evaluation process with diverse yet semantically aligned synthetic samples. To improve inference, we propose a chain-of-debunk strategy where a multi-agent MLLM system produces logic-grounded, multimodal-aware reasoning content and final veracity judgment. By jointly modeling multimodal features, generative debunking cues, and reasoning-rich verification within a unified architecture, DIFND achieves notable improvements in detection accuracy. Extensive experiments on the FakeSV and FVC datasets show that DIFND not only outperforms existing approaches but also delivers trustworthy decisions. 

---
# Towards Fair Rankings: Leveraging LLMs for Gender Bias Detection and Measurement 

**Authors**: Maryam Mousavian, Zahra Abbasiantaeb, Mohammad Aliannejadi, Fabio Crestani  

**Link**: [PDF](https://arxiv.org/pdf/2506.22372)  

**Abstract**: The presence of social biases in Natural Language Processing (NLP) and Information Retrieval (IR) systems is an ongoing challenge, which underlines the importance of developing robust approaches to identifying and evaluating such biases. In this paper, we aim to address this issue by leveraging Large Language Models (LLMs) to detect and measure gender bias in passage ranking. Existing gender fairness metrics rely on lexical- and frequency-based measures, leading to various limitations, e.g., missing subtle gender disparities. Building on our LLM-based gender bias detection method, we introduce a novel gender fairness metric, named Class-wise Weighted Exposure (CWEx), aiming to address existing limitations. To measure the effectiveness of our proposed metric and study LLMs' effectiveness in detecting gender bias, we annotate a subset of the MS MARCO Passage Ranking collection and release our new gender bias collection, called MSMGenderBias, to foster future research in this area. Our extensive experimental results on various ranking models show that our proposed metric offers a more detailed evaluation of fairness compared to previous metrics, with improved alignment to human labels (58.77% for Grep-BiasIR, and 18.51% for MSMGenderBias, measured using Cohen's Kappa agreement), effectively distinguishing gender bias in ranking. By integrating LLM-driven bias detection, an improved fairness metric, and gender bias annotations for an established dataset, this work provides a more robust framework for analyzing and mitigating bias in IR systems. 

---
# RiverEcho: Real-Time Interactive Digital System for Ancient Yellow River Culture 

**Authors**: Haofeng Wang, Yilin Guo, Zehao Li, Tong Yue, Yizong Wang, Enci Zhang, Rongqun Lin, Feng Gao, Shiqi Wang, Siwei Ma  

**Link**: [PDF](https://arxiv.org/pdf/2506.21865)  

**Abstract**: The Yellow River is China's mother river and a cradle of human civilization. The ancient Yellow River culture is, moreover, an indispensable part of human art history. To conserve and inherit the ancient Yellow River culture, we designed RiverEcho, a real-time interactive system that responds to voice queries using a large language model and a cultural knowledge dataset, delivering explanations through a talking-head digital human. Specifically, we built a knowledge database focused on the ancient Yellow River culture, including the collection of historical texts and the processing pipeline. Experimental results demonstrate that leveraging Retrieval-Augmented Generation (RAG) on the proposed dataset enhances the response quality of the Large Language Model(LLM), enabling the system to generate more professional and informative responses. Our work not only diversifies the means of promoting Yellow River culture but also provides users with deeper cultural insights. 

---
# CAL-RAG: Retrieval-Augmented Multi-Agent Generation for Content-Aware Layout Design 

**Authors**: Najmeh Forouzandehmehr, Reza Yousefi Maragheh, Sriram Kollipara, Kai Zhao, Topojoy Biswas, Evren Korpeoglu, Kannan Achan  

**Link**: [PDF](https://arxiv.org/pdf/2506.21934)  

**Abstract**: Automated content-aware layout generation -- the task of arranging visual elements such as text, logos, and underlays on a background canvas -- remains a fundamental yet under-explored problem in intelligent design systems. While recent advances in deep generative models and large language models (LLMs) have shown promise in structured content generation, most existing approaches lack grounding in contextual design exemplars and fall short in handling semantic alignment and visual coherence. In this work we introduce CAL-RAG, a retrieval-augmented, agentic framework for content-aware layout generation that integrates multimodal retrieval, large language models, and collaborative agentic reasoning. Our system retrieves relevant layout examples from a structured knowledge base and invokes an LLM-based layout recommender to propose structured element placements. A vision-language grader agent evaluates the layout with visual metrics, and a feedback agent provides targeted refinements, enabling iterative improvement. We implement our framework using LangGraph and evaluate it on the PKU PosterLayout dataset, a benchmark rich in semantic and structural variability. CAL-RAG achieves state-of-the-art performance across multiple layout metrics -- including underlay effectiveness, element alignment, and overlap -- substantially outperforming strong baselines such as LayoutPrompter. These results demonstrate that combining retrieval augmentation with agentic multi-step reasoning yields a scalable, interpretable, and high-fidelity solution for automated layout generation. 

---
# HLTCOE at LiveRAG: GPT-Researcher using ColBERT retrieval 

**Authors**: Kevin Duh, Eugene Yang, Orion Weller, Andrew Yates, Dawn Lawrie  

**Link**: [PDF](https://arxiv.org/pdf/2506.22356)  

**Abstract**: The HLTCOE LiveRAG submission utilized the GPT-researcher framework for researching the context of the question, filtering the returned results, and generating the final answer. The retrieval system was a ColBERT bi-encoder architecture, which represents a passage with many dense tokens. Retrieval used a local, compressed index of the FineWeb10-BT collection created with PLAID-X, using a model fine-tuned for multilingual retrieval. Query generation from context was done with Qwen2.5-7B-Instruct, while filtering was accomplished with m2-bert-80M-8k-retrieval. Up to nine passages were used as context to generate an answer using Falcon3-10B. This system placed 5th in the LiveRAG automatic evaluation for correctness with a score of 1.07. 

---
# PentaRAG: Large-Scale Intelligent Knowledge Retrieval for Enterprise LLM Applications 

**Authors**: Abu Hanif Muhammad Syarubany, Chang Dong Yoo  

**Link**: [PDF](https://arxiv.org/pdf/2506.21593)  

**Abstract**: Enterprise deployments of large-language model (LLM) demand continuously changing document collections with sub-second latency and predictable GPU cost requirements that classical Retrieval-Augmented Generation (RAG) pipelines only partially satisfy. We present PentaRAG, a five-layer module that routes each query through two instant caches (fixed key-value and semantic), a memory-recall mode that exploits the LLM's own weights, an adaptive session memory, and a conventional retrieval-augmentation layer. Implemented with Mistral-8B, Milvus and vLLM, the system can answer most repeated or semantically similar questions from low-latency caches while retaining full retrieval for novel queries. On the TriviaQA domain, LoRA fine-tuning combined with the memory-recall layer raises answer similarity by approximately 8% and factual correctness by approximately 16% over the base model. Under a nine-session runtime simulation, cache warming reduces mean latency from several seconds to well below one second and shifts traffic toward the fast paths. Resource-efficiency tests show that PentaRAG cuts average GPU time to 0.248 seconds per query, roughly half that of a naive RAG baseline, and sustains an aggregate throughput of approximately 100,000 queries per second on our setup. These results demonstrate that a layered routing strategy can deliver freshness, speed, and efficiency simultaneously in production-grade RAG systems. 

---
# Doc2SAR: A Synergistic Framework for High-Fidelity Extraction of Structure-Activity Relationships from Scientific Documents 

**Authors**: Jiaxi Zhuang, Kangning Li, Jue Hou, Mingjun Xu, Zhifeng Gao, Hengxing Cai  

**Link**: [PDF](https://arxiv.org/pdf/2506.21625)  

**Abstract**: Extracting molecular structure-activity relationships (SARs) from scientific literature and patents is essential for drug discovery and materials research. However, this task remains challenging due to heterogeneous document formats and limitations of existing methods. Specifically, rule-based approaches relying on rigid templates fail to generalize across diverse document layouts, while general-purpose multimodal large language models (MLLMs) lack sufficient accuracy and reliability for specialized tasks, such as layout detection and optical chemical structure recognition (OCSR). To address these challenges, we introduce DocSAR-200, a rigorously annotated benchmark of 200 scientific documents designed specifically for evaluating SAR extraction methods. Additionally, we propose Doc2SAR, a novel synergistic framework that integrates domain-specific tools with MLLMs enhanced via supervised fine-tuning (SFT). Extensive experiments demonstrate that Doc2SAR achieves state-of-the-art performance across various document types, significantly outperforming leading end-to-end baselines. Specifically, Doc2SAR attains an overall Table Recall of 80.78% on DocSAR-200, exceeding end2end GPT-4o by 51.48%. Furthermore, Doc2SAR demonstrates practical usability through efficient inference and is accompanied by a web app. 

---
