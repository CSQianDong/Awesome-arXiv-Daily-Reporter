# Safety Instincts: LLMs Learn to Trust Their Internal Compass for Self-Defense 

**Authors**: Guobin Shen, Dongcheng Zhao, Haibo Tong, Jindong Li, Feifei Zhao, Yi Zeng  

**Link**: [PDF](https://arxiv.org/pdf/2510.01088)  

**Abstract**: Ensuring Large Language Model (LLM) safety remains challenging due to the absence of universal standards and reliable content validators, making it difficult to obtain effective training signals. We discover that aligned models already possess robust internal safety beliefs: they consistently produce high-confidence refusals to harmful requests while exhibiting high entropy when generating potentially dangerous content. This entropy gap reveals an untapped signal--models intrinsically "know" when to refuse. We introduce Safety Instincts Reinforcement Learning (SIRL), which transforms this internal confidence into a self-generated reward signal, eliminating dependence on external validators or human annotations. SIRL teaches models to trust their safety instincts by reinforcing low-entropy refusal behaviors. Evaluated on Llama and Qwen models, SIRL maintains 89%+ Defense Success Rates (DSRs) against 20+ jailbreak methods, from static prompts to adaptive attacks. Using only 15,000 unlabeled prompts, SIRL surpasses resource-intensive supervised methods while preserving performance on mathematics, coding, and conversation benchmarks. Our work demonstrates that effective alignment can emerge from within, paving the way for more autonomous and robust AI safety mechanisms that scale without extensive human oversight. 

---
# Generalized Parallel Scaling with Interdependent Generations 

**Authors**: Harry Dong, David Brandfonbrener, Eryk Helenowski, Yun He, Mrinal Kumar, Han Fang, Yuejie Chi, Karthik Abinav Sankararaman  

**Link**: [PDF](https://arxiv.org/pdf/2510.01143)  

**Abstract**: Parallel LLM inference scaling involves sampling a set of $N>1$ responses for a single input prompt. However, these $N$ parallel responses tend to be generated independently from each other, partitioning compute resources and leaving potentially useful information in one generation untapped by others. This is in contrast to response length scaling where past computation is used in all future steps. For higher quality responses and response sets, we propose Bridge to generate interdependent responses in parallel by rethinking batched LLM hidden states as holistic tensors rather than independent slices. With only a small amount (2.8%-5.1%) of new parameters, Bridge improves the relative mean accuracy gains from reinforcement learning with verifiable rewards by up to 50% and boosts consistency of correct responses. Trained once, Bridge scales to any generation width, all with greater performance than independent generations, unlocking a more general mode of parallel scaling that effectively leverages information between sequences, compatible with any post-generation aggregation technique. 

---
# Typed Chain-of-Thought: A Curry-Howard Framework for Verifying LLM Reasoning 

**Authors**: Elija Perrier  

**Link**: [PDF](https://arxiv.org/pdf/2510.01069)  

**Abstract**: While Chain-of-Thought (CoT) prompting enhances the reasoning capabilities of large language models, the faithfulness of the generated rationales remains an open problem for model interpretability. We propose a novel theoretical lens for this problem grounded in the Curry-Howard correspondence, which posits a direct relationship between formal proofs and computer programs. Under this paradigm, a faithful reasoning trace is analogous to a well-typed program, where each intermediate step corresponds to a typed logical inference. We operationalise this analogy, presenting methods to extract and map the informal, natural language steps of CoT into a formal, typed proof structure. Successfully converting a CoT trace into a well-typed proof serves as a strong, verifiable certificate of its computational faithfulness, moving beyond heuristic interpretability towards formal verification. Our framework provides a methodology to transform plausible narrative explanations into formally verifiable programs, offering a path towards building more reliable and trustworthy AI systems. 

---
# QUASAR: Quantum Assembly Code Generation Using Tool-Augmented LLMs via Agentic RL 

**Authors**: Cong Yu, Valter Uotila, Shilong Deng, Qingyuan Wu, Tuo Shi, Songlin Jiang, Lei You, Bo Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2510.00967)  

**Abstract**: Designing and optimizing task-specific quantum circuits are crucial to leverage the advantage of quantum computing. Recent large language model (LLM)-based quantum circuit generation has emerged as a promising automatic solution. However, the fundamental challenges remain unaddressed: (i) parameterized quantum gates require precise numerical values for optimal performance, which also depend on multiple aspects, including the number of quantum gates, their parameters, and the layout/depth of the circuits. (ii) LLMs often generate low-quality or incorrect quantum circuits due to the lack of quantum domain-specific knowledge. We propose QUASAR, an agentic reinforcement learning (RL) framework for quantum circuits generation and optimization based on tool-augmented LLMs. To align the LLM with quantum-specific knowledge and improve the generated quantum circuits, QUASAR designs (i) a quantum circuit verification approach with external quantum simulators and (ii) a sophisticated hierarchical reward mechanism in RL training. Extensive evaluation shows improvements in both syntax and semantic performance of the generated quantum circuits. When augmenting a 4B LLM, QUASAR has achieved the validity of 99.31% in Pass@1 and 100% in Pass@10, outperforming industrial LLMs of GPT-4o, GPT-5 and DeepSeek-V3 and several supervised-fine-tuning (SFT)-only and RL-only baselines. 

---
# Learning Compact Representations of LLM Abilities via Item Response Theory 

**Authors**: Jianhao Chen, Chenxu Wang, Gengrui Zhang, Peng Ye, Lei Bai, Wei Hu, Yuzhong Qu, Shuyue Hu  

**Link**: [PDF](https://arxiv.org/pdf/2510.00844)  

**Abstract**: Recent years have witnessed a surge in the number of large language models (LLMs), yet efficiently managing and utilizing these vast resources remains a significant challenge. In this work, we explore how to learn compact representations of LLM abilities that can facilitate downstream tasks, such as model routing and performance prediction on new benchmarks. We frame this problem as estimating the probability that a given model will correctly answer a specific query. Inspired by the item response theory (IRT) in psychometrics, we model this probability as a function of three key factors: (i) the model's multi-skill ability vector, (2) the query's discrimination vector that separates models of differing skills, and (3) the query's difficulty scalar. To learn these parameters jointly, we introduce a Mixture-of-Experts (MoE) network that couples model- and query-level embeddings. Extensive experiments demonstrate that our approach leads to state-of-the-art performance in both model routing and benchmark accuracy prediction. Moreover, analysis validates that the learned parameters encode meaningful, interpretable information about model capabilities and query characteristics. 

---
# On Discovering Algorithms for Adversarial Imitation Learning 

**Authors**: Shashank Reddy Chirra, Jayden Teoh, Praveen Paruchuri, Pradeep Varakantham  

**Link**: [PDF](https://arxiv.org/pdf/2510.00922)  

**Abstract**: Adversarial Imitation Learning (AIL) methods, while effective in settings with limited expert demonstrations, are often considered unstable. These approaches typically decompose into two components: Density Ratio (DR) estimation $\frac{\rho_E}{\rho_{\pi}}$, where a discriminator estimates the relative occupancy of state-action pairs under the policy versus the expert; and Reward Assignment (RA), where this ratio is transformed into a reward signal used to train the policy. While significant research has focused on improving density estimation, the role of reward assignment in influencing training dynamics and final policy performance has been largely overlooked. RA functions in AIL are typically derived from divergence minimization objectives, relying heavily on human design and ingenuity. In this work, we take a different approach: we investigate the discovery of data-driven RA functions, i.e, based directly on the performance of the resulting imitation policy. To this end, we leverage an LLM-guided evolutionary framework that efficiently explores the space of RA functions, yielding \emph{Discovered Adversarial Imitation Learning} (DAIL), the first meta-learnt AIL algorithm. Remarkably, DAIL generalises across unseen environments and policy optimization algorithms, outperforming the current state-of-the-art of \emph{human-designed} baselines. Finally, we analyse why DAIL leads to more stable training, offering novel insights into the role of RA functions in the stability of AIL. Code is publicly available: this https URL. 

---
# AI in data science education: experiences from the classroom 

**Authors**: J.A. Hageman, C.F.W. Peeters  

**Link**: [PDF](https://arxiv.org/pdf/2510.00793)  

**Abstract**: This study explores the integration of AI, particularly large language models (LLMs) like ChatGPT, into educational settings, focusing on the implications for teaching and learning. Through interviews with course coordinators from data science courses at Wageningen University, this research identifies both the benefits and challenges associated with AI in the classroom. While AI tools can streamline tasks and enhance learning, concerns arise regarding students' overreliance on these technologies, potentially hindering the development of essential cognitive and problem solving skills. The study highlights the importance of responsible AI usage, ethical considerations, and the need for adapting assessment methods to ensure educational outcomes are met. With careful integration, AI can be a valuable asset in education, provided it is used to complement rather than replace fundamental learning processes. 

---
# Uncovering the Computational Ingredients of Human-Like Representations in LLMs 

**Authors**: Zach Studdiford, Timothy T. Rogers, Kushin Mukherjee, Siddharth Suresh  

**Link**: [PDF](https://arxiv.org/pdf/2510.01030)  

**Abstract**: The ability to translate diverse patterns of inputs into structured patterns of behavior has been thought to rest on both humans' and machines' ability to learn robust representations of relevant concepts. The rapid advancement of transformer-based large language models (LLMs) has led to a diversity of computational ingredients -- architectures, fine tuning methods, and training datasets among others -- but it remains unclear which of these ingredients are most crucial for building models that develop human-like representations. Further, most current LLM benchmarks are not suited to measuring representational alignment between humans and models, making benchmark scores unreliable for assessing if current LLMs are making progress towards becoming useful cognitive models. We address these limitations by first evaluating a set of over 70 models that widely vary in their computational ingredients on a triplet similarity task, a method well established in the cognitive sciences for measuring human conceptual representations, using concepts from the THINGS database. Comparing human and model representations, we find that models that undergo instruction-finetuning and which have larger dimensionality of attention heads are among the most human aligned, while multimodal pretraining and parameter size have limited bearing on alignment. Correlations between alignment scores and scores on existing benchmarks reveal that while some benchmarks (e.g., MMLU) are better suited than others (e.g., MUSR) for capturing representational alignment, no existing benchmark is capable of fully accounting for the variance of alignment scores, demonstrating their insufficiency in capturing human-AI alignment. Taken together, our findings help highlight the computational ingredients most essential for advancing LLMs towards models of human conceptual representation and address a key benchmarking gap in LLM evaluation. 

---
# Is Model Editing Built on Sand? Revealing Its Illusory Success and Fragile Foundation 

**Authors**: Wei Liu, Haomei Xu, Bingqing Liu, Zhiying Deng, Haozhao Wang, Jun Wang, Ruixuan Li, Yee Whye Teh, Wee Sun Lee  

**Link**: [PDF](https://arxiv.org/pdf/2510.00625)  

**Abstract**: Large language models (LLMs) inevitably encode outdated or incorrect knowledge. Updating, deleting, and forgetting such knowledge is important for alignment, safety, and other issues. To address this issue, model editing has emerged as a promising paradigm: by precisely editing a small subset of parameters such that a specific fact is updated while preserving other knowledge. Despite its great success reported in previous papers, we find the apparent reliability of editing rests on a fragile foundation and the current literature is largely driven by illusory success. The fundamental goal of steering the model's output toward a target with minimal modification would encourage exploiting hidden shortcuts, rather than utilizing real semantics. This problem directly challenges the feasibility of the current model editing literature at its very foundation, as shortcuts are inherently at odds with robust knowledge integration. Coincidentally, this issue has long been obscured by evaluation frameworks that lack the design of negative examples. To uncover it, we systematically develop a suite of new evaluation methods. Strikingly, we find that state-of-the-art approaches collapse even under the simplest negation queries. Our empirical evidence shows that editing is likely to be based on shortcuts rather than full semantics, calling for an urgent reconsideration of the very basis of model editing before further advancements can be meaningfully pursued. 

---
# EvolProver: Advancing Automated Theorem Proving by Evolving Formalized Problems via Symmetry and Difficulty 

**Authors**: Yuchen Tian, Ruiyuan Huang, Xuanwu Wang, Jing Ma, Zengfeng Huang, Ziyang Luo, Hongzhan Lin, Da Zheng, Lun Du  

**Link**: [PDF](https://arxiv.org/pdf/2510.00732)  

**Abstract**: Large Language Models (LLMs) for formal theorem proving have shown significant promise, yet they often lack generalizability and are fragile to even minor transformations of problem statements. To address this limitation, we introduce a novel data augmentation pipeline designed to enhance model robustness from two perspectives: symmetry and difficulty. From the symmetry perspective, we propose two complementary methods: EvolAST, an Abstract Syntax Tree (AST) based approach that targets syntactic symmetry to generate semantically equivalent problem variants, and EvolDomain, which leverages LLMs to address semantic symmetry by translating theorems across mathematical domains. From the difficulty perspective, we propose EvolDifficulty, which uses carefully designed evolutionary instructions to guide LLMs in generating new theorems with a wider range of difficulty. We then use the evolved data to train EvolProver, a 7B-parameter non-reasoning theorem prover. EvolProver establishes a new state-of-the-art (SOTA) on FormalMATH-Lite with a 53.8% pass@32 rate, surpassing all models of comparable size, including reasoning-based models. It also sets new SOTA records for non-reasoning models on MiniF2F-Test (69.8% pass@32), Ineq-Comp-Seed (52.2% pass@32), and Ineq-Comp-Transformed (34.0% pass@32). Ablation studies further confirm our data augmentation pipeline's effectiveness across multiple benchmarks. 

---
# ACPO: Adaptive Curriculum Policy Optimization for Aligning Vision-Language Models in Complex Reasoning 

**Authors**: Yunhao Wang, Ziting Li, Shuai Chen, Tao Liu, Chao Song, Junjie Jiang, Jian Zhu, Peng Gao, Bin Qin  

**Link**: [PDF](https://arxiv.org/pdf/2510.00690)  

**Abstract**: Aligning large-scale vision-language models (VLMs) for complex reasoning via reinforcement learning is often hampered by the limitations of existing policy optimization algorithms, such as static training schedules and the rigid, uniform clipping mechanism in Proximal Policy Optimization (PPO). In this work, we introduce Adaptive Curriculum Policy Optimization (ACPO), a novel framework that addresses these challenges through a dual-component adaptive learning strategy. First, ACPO employs a dynamic curriculum that orchestrates a principled transition from a stable, near on-policy exploration phase to an efficient, off-policy exploitation phase by progressively increasing sample reuse. Second, we propose an Advantage-Aware Adaptive Clipping (AAAC) mechanism that replaces the fixed clipping hyperparameter with dynamic, sample-wise bounds modulated by the normalized advantage of each token. This allows for more granular and robust policy updates, enabling larger gradients for high-potential samples while safeguarding against destructive ones. We conduct extensive experiments on a suite of challenging multimodal reasoning benchmarks, including MathVista, LogicVista, and MMMU-Pro. Results demonstrate that ACPO consistently outperforms strong baselines such as DAPO and PAPO, achieving state-of-the-art performance, accelerated convergence, and superior training stability. 

---
# Logical Consistency Between Disagreeing Experts and Its Role in AI Safety 

**Authors**: Andrés Corrada-Emmanuel  

**Link**: [PDF](https://arxiv.org/pdf/2510.00821)  

**Abstract**: If two experts disagree on a test, we may conclude both cannot be 100 per cent correct. But if they completely agree, no possible evaluation can be excluded. This asymmetry in the utility of agreements versus disagreements is explored here by formalizing a logic of unsupervised evaluation for classifiers. Its core problem is computing the set of group evaluations that are logically consistent with how we observe them agreeing and disagreeing in their decisions. Statistical summaries of their aligned decisions are inputs into a Linear Programming problem in the integer space of possible correct or incorrect responses given true labels. Obvious logical constraints, such as, the number of correct responses cannot exceed the number of observed responses, are inequalities. But in addition, there are axioms, universally applicable linear equalities that apply to all finite tests. The practical and immediate utility of this approach to unsupervised evaluation using only logical consistency is demonstrated by building no-knowledge alarms that can detect when one or more LLMs-as-Judges are violating a minimum grading threshold specified by the user. 

---
# Exploring Network-Knowledge Graph Duality: A Case Study in Agentic Supply Chain Risk Analysis 

**Authors**: Evan Heus, Rick Bookstaber, Dhruv Sharma  

**Link**: [PDF](https://arxiv.org/pdf/2510.01115)  

**Abstract**: Large Language Models (LLMs) struggle with the complex, multi-modal, and network-native data underlying financial risk. Standard Retrieval-Augmented Generation (RAG) oversimplifies relationships, while specialist models are costly and static. We address this gap with an LLM-centric agent framework for supply chain risk analysis. Our core contribution is to exploit the inherent duality between networks and knowledge graphs (KG). We treat the supply chain network as a KG, allowing us to use structural network science principles for retrieval. A graph traverser, guided by network centrality scores, efficiently extracts the most economically salient risk paths. An agentic architecture orchestrates this graph retrieval alongside data from numerical factor tables and news streams. Crucially, it employs novel ``context shells'' -- descriptive templates that embed raw figures in natural language -- to make quantitative data fully intelligible to the LLM. This lightweight approach enables the model to generate concise, explainable, and context-rich risk narratives in real-time without costly fine-tuning or a dedicated graph database. 

---
# Expected Attention: KV Cache Compression by Estimating Attention from Future Queries Distribution 

**Authors**: Alessio Devoto, Maximilian Jeblick, Simon Jégou  

**Link**: [PDF](https://arxiv.org/pdf/2510.00636)  

**Abstract**: Memory consumption of the Key-Value (KV) cache represents a major bottleneck for efficient large language model inference. While attention-score-based KV cache pruning shows promise, it faces critical practical limitations: attention scores from future tokens are unavailable during compression, and modern implementations like Flash Attention do not materialize the full attention matrix, making past scores inaccessible. To overcome these challenges, we introduce $\textbf{Expected Attention, a training-free compression method}$ that estimates KV pairs importance by predicting how future queries will attend to them. Our approach leverages the distributional properties of LLM activations to compute expected attention scores in closed form for each KV pair. These scores enable principled ranking and pruning of KV pairs with minimal impact on the residual stream, achieving effective compression without performance degradation. Importantly, our method operates seamlessly across both prefilling and decoding phases, consistently outperforming state-of-the-art baselines in both scenarios. Finally, $\textbf{we release KVPress, a comprehensive library to enable researchers to implement and benchmark KV cache compression methods, already including more than 20 techniques}$. 

---
# BiasBusters: Uncovering and Mitigating Tool Selection Bias in Large Language Models 

**Authors**: Thierry Blankenstein, Jialin Yu, Zixuan Li, Vassilis Plachouras, Sunando Sengupta, Philip Torr, Yarin Gal, Alasdair Paren, Adel Bibi  

**Link**: [PDF](https://arxiv.org/pdf/2510.00307)  

**Abstract**: Agents backed by large language models (LLMs) often rely on external tools drawn from marketplaces where multiple providers offer functionally equivalent options. This raises a critical point concerning fairness: if selection is systematically biased, it can degrade user experience and distort competition by privileging some providers over others. We introduce a benchmark of diverse tool categories, each containing multiple functionally equivalent tools, to evaluate tool-selection bias. Using this benchmark, we test seven models and show that unfairness exists with models either fixating on a single provider or disproportionately preferring earlier-listed tools in context. To investigate the origins of this bias, we conduct controlled experiments examining tool features, metadata (name, description, parameters), and pre-training exposure. We find that: (1) semantic alignment between queries and metadata is the strongest predictor of choice; (2) perturbing descriptions significantly shifts selections; and (3) repeated pre-training exposure to a single endpoint amplifies bias. Finally, we propose a lightweight mitigation that first filters the candidate tools to a relevant subset and then samples uniformly, reducing bias while preserving good task coverage. Our findings highlight tool-selection bias as a key obstacle for the fair deployment of tool-augmented LLMs. 

---
# ICL Optimized Fragility 

**Authors**: Serena Gomez Wannaz  

**Link**: [PDF](https://arxiv.org/pdf/2510.00300)  

**Abstract**: ICL guides are known to improve task-specific performance, but their impact on cross-domain cognitive abilities remains unexplored. This study examines how ICL guides affect reasoning across different knowledge domains using six variants of the GPT-OSS:20b model: one baseline model and five ICL configurations (simple, chain-of-thought, random, appended text, and symbolic language). The models were subjected to 840 tests spanning general knowledge questions, logic riddles, and a mathematical olympiad problem. Statistical analysis (ANOVA) revealed significant behavioral modifications (p less than 0.001) across ICL variants, demonstrating a phenomenon termed "optimized fragility." ICL models achieved 91%-99% accuracy on general knowledge tasks while showing degraded performance on complex reasoning problems, with accuracy dropping to 10-43% on riddles compared to 43% for the baseline model. Notably, no significant differences emerged on the olympiad problem (p=0.2173), suggesting that complex mathematical reasoning remains unaffected by ICL optimization. These findings indicate that ICL guides create systematic trade-offs between efficiency and reasoning flexibility, with important implications for LLM deployment and AI safety. 

---
# Drones that Think on their Feet: Sudden Landing Decisions with Embodied AI 

**Authors**: Diego Ortiz Barbosa, Mohit Agrawal, Yash Malegaonkar, Luis Burbano, Axel Andersson, György Dán, Henrik Sandberg, Alvaro A. Cardenas  

**Link**: [PDF](https://arxiv.org/pdf/2510.00167)  

**Abstract**: Autonomous drones must often respond to sudden events, such as alarms, faults, or unexpected changes in their environment, that require immediate and adaptive decision-making. Traditional approaches rely on safety engineers hand-coding large sets of recovery rules, but this strategy cannot anticipate the vast range of real-world contingencies and quickly becomes incomplete. Recent advances in embodied AI, powered by large visual language models, provide commonsense reasoning to assess context and generate appropriate actions in real time. We demonstrate this capability in a simulated urban benchmark in the Unreal Engine, where drones dynamically interpret their surroundings and decide on sudden maneuvers for safe landings. Our results show that embodied AI makes possible a new class of adaptive recovery and decision-making pipelines that were previously infeasible to design by hand, advancing resilience and safety in autonomous aerial systems. 

---
# DualTune: Decoupled Fine-Tuning for On-Device Agentic Systems 

**Authors**: Rohan Kadekodi, Zhan Jin, Keisuke Kamahori, Yile Gu, Sean Khatiri, Noah H. Bayindirli, Sergey Gorbunov, Baris Kasikci  

**Link**: [PDF](https://arxiv.org/pdf/2510.00229)  

**Abstract**: The deployment of Large Language Models (LLMs) as agentic orchestrators has revolutionized task automation, but the need for privacy-preserving, cost-effective solutions demands on-device inference capabilities. However, local LLMs consistently underperform compared to frontier models in tool calling scenarios, struggling with both tool selection from large tool sets and accurate argument generation for complex parameter structures. We introduce a methodology that disaggregates a tool-calling task into two distinct subtasks: tool selection and argument generation. We propose "decoupled fine-tuning", a novel post-training approach that employs LoRA fine-tuning to create dedicated LoRA adapters for tool selection and tool-specific argument generation using separate loss masking for each of the subtasks. Furthermore, we present DualTune, an inference framework that leverages the LoRA adapters created using decoupled fine-tuning to perform efficient agent orchestration with the help of local models on end-user devices. DualTune decomposes the tool-call generation step into tool selection and argument generation, and dynamically loads the corresponding LoRA adapters to generate tool calls. Additionally, DualTune implements hierarchical orchestration to restrict the number of tools required for tool selection. Our experiments on the MCP-Bench benchmark demonstrate that the Qwen-2.5-7B model trained using decoupled fine-tuning improves the tool calling accuracy of the base model by 46%, and outperforms other local reasoning, non-reasoning and fine-tuned models of similar size in all cases, and models that are 2x larger, in most cases. 

---
# Simultaneous Multi-objective Alignment Across Verifiable and Non-verifiable Rewards 

**Authors**: Yiran Shen, Yu Xia, Jonathan Chang, Prithviraj Ammanabrolu  

**Link**: [PDF](https://arxiv.org/pdf/2510.01167)  

**Abstract**: Aligning large language models to human preferences is inherently multidimensional, yet most pipelines collapse heterogeneous signals into a single optimizeable objective. We seek to answer what it would take to simultaneously align a model across various domains spanning those with: verifiable rewards (mathematical accuracy), non-verifiable subjective preferences (human values), and complex interactive scenarios (multi-turn AI tutoring dialogues). Such multi-objective reinforcement learning setups are often plagued by the individual objectives being at odds with each other, resulting in inefficient training and little user control during inference. We propose a unified framework that: (i) standardizes {process reward model} (PRM) training across both verifiable and non-verifiable settings to better supervise models' chain-of-thought reasoning; (ii) performs {multi-objective alignment} by training the LLM with our $\textbf{M}$ulti-$\textbf{A}$ction-$\textbf{H}$ead $\textbf{DPO}$ (MAH-DPO) and a vectorized reward where the dimensions of the vector correspond to the various objectives instead of a single scalar; and (iii) demonstrates how such a system provides fine-grained inference-time user control. Experiments across math reasoning, value alignment, and multi-turn dialogue show that our framework improves performance across multiple objectives simultaneously, while minimizing cross-objective trade-offs and enabling flexible inference time user control. The code can be found at this https URL. 

---
# Verbalized Sampling: How to Mitigate Mode Collapse and Unlock LLM Diversity 

**Authors**: Jiayi Zhang, Simon Yu, Derek Chong, Anthony Sicilia, Michael R. Tomz, Christopher D. Manning, Weiyan Shi  

**Link**: [PDF](https://arxiv.org/pdf/2510.01171)  

**Abstract**: Post-training alignment often reduces LLM diversity, leading to a phenomenon known as mode collapse. Unlike prior work that attributes this effect to algorithmic limitations, we identify a fundamental, pervasive data-level driver: typicality bias in preference data, whereby annotators systematically favor familiar text as a result of well-established findings in cognitive psychology. We formalize this bias theoretically, verify it on preference datasets empirically, and show that it plays a central role in mode collapse. Motivated by this analysis, we introduce Verbalized Sampling, a simple, training-free prompting strategy to circumvent mode collapse. VS prompts the model to verbalize a probability distribution over a set of responses (e.g., ``Generate 5 jokes about coffee and their corresponding probabilities''). Comprehensive experiments show that VS significantly improves performance across creative writing (poems, stories, jokes), dialogue simulation, open-ended QA, and synthetic data generation, without sacrificing factual accuracy and safety. For instance, in creative writing, VS increases diversity by 1.6-2.1x over direct prompting. We further observe an emergent trend that more capable models benefit more from VS. In sum, our work provides a new data-centric perspective on mode collapse and a practical inference-time remedy that helps unlock pre-trained generative diversity. 

---
# mR3: Multilingual Rubric-Agnostic Reward Reasoning Models 

**Authors**: David Anugraha, Shou-Yi Hung, Zilu Tang, Annie En-Shiun Lee, Derry Tanti Wijaya, Genta Indra Winata  

**Link**: [PDF](https://arxiv.org/pdf/2510.01146)  

**Abstract**: Evaluation using Large Language Model (LLM) judges has been widely adopted in English and shown to be effective for automatic evaluation. However, their performance does not generalize well to non-English settings, and it remains unclear what constitutes effective multilingual training for such judges. In this paper, we introduce mR3, a massively multilingual, rubric-agnostic reward reasoning model trained on 72 languages, achieving the broadest language coverage in reward modeling to date. We present a comprehensive study of data and curriculum selection for training to identify effective strategies and data sources for building high-quality reward models, including the integration of target-language reasoning datasets. Our approach attains state-of-the-art performance on multilingual reward model benchmarks, surpassing much larger models (i.e., GPT-OSS-120B) while being up to 9x smaller, and its effectiveness is further confirmed through extensive ablation studies. Our models, data, and code are available as open source at this https URL. 

---
# ARS: Adaptive Reasoning Suppression for Efficient Large Reasoning Language Models 

**Authors**: Dongqi Zheng  

**Link**: [PDF](https://arxiv.org/pdf/2510.00071)  

**Abstract**: Large Reasoning Language Models (LRLMs or LRMs) demonstrate remarkable capabilities in complex reasoning tasks, but suffer from significant computational inefficiencies due to overthinking phenomena. Existing efficient reasoning methods face the challenge of balancing reasoning quality with inference cost reduction. We propose \textbf{Adaptive Reasoning Suppression (ARS)}, a novel training-free approach that dynamically suppresses redundant reasoning steps while preserving accuracy through adaptive certainty monitoring. ARS introduces a multi-checkpoint certainty estimation mechanism with progressive suppression thresholds, achieving superior efficiency compared to static suppression methods. Our extensive evaluation across mathematical reasoning benchmarks using multiple model architectures demonstrates that ARS achieves up to 53%, 46.1%, and 57.9% in token, latency and energy reduction, while maintaining or improving accuracy. 

---
# Social Welfare Function Leaderboard: When LLM Agents Allocate Social Welfare 

**Authors**: Zhengliang Shi, Ruotian Ma, Jen-tse Huang, Xinbei Ma, Xingyu Chen, Mengru Wang, Qu Yang, Yue Wang, Fanghua Ye, Ziyang Chen, Shanyi Wang, Cixing Li, Wenxuan Wang, Zhaopeng Tu, Xiaolong Li, Zhaochun Ren, Linus  

**Link**: [PDF](https://arxiv.org/pdf/2510.01164)  

**Abstract**: Large language models (LLMs) are increasingly entrusted with high-stakes decisions that affect human welfare. However, the principles and values that guide these models when distributing scarce societal resources remain largely unexamined. To address this, we introduce the Social Welfare Function (SWF) Benchmark, a dynamic simulation environment where an LLM acts as a sovereign allocator, distributing tasks to a heterogeneous community of recipients. The benchmark is designed to create a persistent trade-off between maximizing collective efficiency (measured by Return on Investment) and ensuring distributive fairness (measured by the Gini coefficient). We evaluate 20 state-of-the-art LLMs and present the first leaderboard for social welfare allocation. Our findings reveal three key insights: (i) A model's general conversational ability, as measured by popular leaderboards, is a poor predictor of its allocation skill. (ii) Most LLMs exhibit a strong default utilitarian orientation, prioritizing group productivity at the expense of severe inequality. (iii) Allocation strategies are highly vulnerable, easily perturbed by output-length constraints and social-influence framing. These results highlight the risks of deploying current LLMs as societal decision-makers and underscore the need for specialized benchmarks and targeted alignment for AI governance. 

---
# GRAD: Generative Retrieval-Aligned Demonstration Sampler for Efficient Few-Shot Reasoning 

**Authors**: Oussama Gabouj, Kamel Charaf, Ivan Zakazov, Nicolas Baldwin, Robert West  

**Link**: [PDF](https://arxiv.org/pdf/2510.01165)  

**Abstract**: Large Language Models (LLMs) achieve strong performance across diverse tasks, but their effectiveness often depends on the quality of the provided context. Retrieval-Augmented Generation (RAG) enriches prompts with external information, but its reliance on static databases constrains adaptability and can result in irrelevant demonstrations. In this work, we propose a Generative Retrieval-Aligned Demonstrator (GRAD), a dynamic demonstration-based approach where an LLM model is trained to generate input-specific concise demonstrations. By tailoring demonstrations to each input, our method offers better contextual support than traditional RAG approaches. We demonstrate the superiority of GRAD under budget constraints, where we limit both the number of tokens used per demonstration and the number of tokens used for the final output. Trained solely on a math dataset, GRAD consistently outperforms strong baselines on Qwen2.5-14B across mathematical reasoning and advanced STEM questions, highlighting GRAD's robust generalization to out-of-distribution (OOD) domains such as physics, chemistry, and computer science. Furthermore, we show that demonstrations generated by trained smaller models can effectively guide larger target models, reducing training costs while maintaining competitive accuracy. Overall, this work introduces a scalable demonstration generator model presenting the first step toward a dynamic few-shot learning paradigm in resource-constrained settings. We release the code used for the project. 

---
# Rethinking Thinking Tokens: LLMs as Improvement Operators 

**Authors**: Lovish Madaan, Aniket Didolkar, Suchin Gururangan, John Quan, Ruan Silva, Ruslan Salakhutdinov, Manzil Zaheer, Sanjeev Arora, Anirudh Goyal  

**Link**: [PDF](https://arxiv.org/pdf/2510.01123)  

**Abstract**: Reasoning training incentivizes LLMs to produce long chains of thought (long CoT), which among other things, allows them to explore solution strategies with self-checking. This results in higher accuracy, but inflates context length, token/compute cost, and answer latency. We ask: Can current models leverage their metacognition to provide other combinations on this Pareto frontier, e.g., better accuracy with lower context length and/or latency? Abstractly, we view the model as an improvement operator on its own "thoughts" with a continuum of possible strategies. We identify an interesting inference family Parallel-Distill-Refine (PDR), which performs the following: (i) generate diverse drafts in parallel; (ii) distill them into a bounded, textual workspace; and (iii) refine conditioned on this workspace, producing an output that seeds the next round. Importantly, context length (hence compute cost) is controllable via degree of parallelism, and is no longer conflated with the total number of generated tokens. We report PDR instantiations of current models that give better accuracy than long CoT while incurring lower latency. Setting degree of parallelism to 1 yields an interesting subcase, Sequential Refinement (SR) (iteratively improve a single candidate answer) which provides performance superior to long CoT. Success of such model orchestrations raises the question whether further training could shift the Pareto frontier. To this end, we train an 8B thinking model with Reinforcement Learning (RL) to make it consistent with PDR as the inference method. On math tasks with verifiable answers, iterative pipelines surpass single-pass baselines at matched sequential budgets, with PDR delivering the largest gains (e.g., +11% on AIME 2024 and +9% on AIME 2025). 

---
# Interpreting Language Models Through Concept Descriptions: A Survey 

**Authors**: Nils Feldhus, Laura Kopf  

**Link**: [PDF](https://arxiv.org/pdf/2510.01048)  

**Abstract**: Understanding the decision-making processes of neural networks is a central goal of mechanistic interpretability. In the context of Large Language Models (LLMs), this involves uncovering the underlying mechanisms and identifying the roles of individual model components such as neurons and attention heads, as well as model abstractions such as the learned sparse features extracted by Sparse Autoencoders (SAEs). A rapidly growing line of work tackles this challenge by using powerful generator models to produce open-vocabulary, natural language concept descriptions for these components. In this paper, we provide the first survey of the emerging field of concept descriptions for model components and abstractions. We chart the key methods for generating these descriptions, the evolving landscape of automated and human metrics for evaluating them, and the datasets that underpin this research. Our synthesis reveals a growing demand for more rigorous, causal evaluation. By outlining the state of the art and identifying key challenges, this survey provides a roadmap for future research toward making models more transparent. 

---
# Hybrid Dialogue State Tracking for Persian Chatbots: A Language Model-Based Approach 

**Authors**: Samin Mahdipour Aghabagher, Saeedeh Momtazi  

**Link**: [PDF](https://arxiv.org/pdf/2510.01052)  

**Abstract**: Dialogue State Tracking (DST) is an essential element of conversational AI with the objective of deeply understanding the conversation context and leading it toward answering user requests. Due to high demands for open-domain and multi-turn chatbots, the traditional rule-based DST is not efficient enough, since it cannot provide the required adaptability and coherence for human-like experiences in complex conversations. This study proposes a hybrid DST model that utilizes rule-based methods along with language models, including BERT for slot filling and intent detection, XGBoost for intent validation, GPT for DST, and online agents for real-time answer generation. This model is uniquely designed to be evaluated on a comprehensive Persian multi-turn dialogue dataset and demonstrated significantly improved accuracy and coherence over existing methods in Persian-based chatbots. The results demonstrate how effectively a hybrid approach may improve DST capabilities, paving the way for conversational AI systems that are more customized, adaptable, and human-like. 

---
# RiskPO: Risk-based Policy Optimization via Verifiable Reward for LLM Post-Training 

**Authors**: Tao Ren, Jinyang Jiang, Hui Yang, Wan Tian, Minhao Zou, Guanghao Li, Zishi Zhang, Qinghao Wang, Shentao Qin, Yanjun Zhao, Rui Tao, Hui Shao, Yijie Peng  

**Link**: [PDF](https://arxiv.org/pdf/2510.00911)  

**Abstract**: Reinforcement learning with verifiable reward has recently emerged as a central paradigm for post-training large language models (LLMs); however, prevailing mean-based methods, such as Group Relative Policy Optimization (GRPO), suffer from entropy collapse and limited reasoning gains. We argue that these issues stem from overemphasizing high-probability output sequences while neglecting rare but informative reasoning paths. To address these challenges, we propose Risk-based Policy Optimization (RiskPO), which substitutes classical mean-based objectives with principled risk measures. Specifically, we introduce a Mixed Value-at-Risk objective that integrates weighted attention over multiple regions of the reward distribution, thereby amplifying gradient signals on challenging instances and preventing overconfident convergence. We further design a bundling scheme that aggregates multiple questions into bundles, thus enriching the feedback signal and yielding more stable and informative training dynamics. Theoretically, we prove that the risk-averse update alleviates entropy collapse and promotes exploration. Numerically, RiskPO achieves consistent and significant improvements in mathematical reasoning, multi-modal reasoning, and code generation benchmarks, surpassing GRPO and its variants on both Pass@1 and Pass@k metrics. Our results demonstrate that risk-based optimization provides a rigorous and effective paradigm for enhancing LLM reasoning capabilities. 

---
# Benchmarking Foundation Models with Retrieval-Augmented Generation in Olympic-Level Physics Problem Solving 

**Authors**: Shunfeng Zheng, Yudi Zhang, Meng Fang, Zihan Zhang, Zhitan Wu, Mykola Pechenizkiy, Ling Chen  

**Link**: [PDF](https://arxiv.org/pdf/2510.00919)  

**Abstract**: Retrieval-augmented generation (RAG) with foundation models has achieved strong performance across diverse tasks, but their capacity for expert-level reasoning-such as solving Olympiad-level physics problems-remains largely unexplored. Inspired by the way students prepare for competitions by reviewing past problems, we investigate the potential of RAG to enhance physics reasoning in foundation models. We introduce PhoPile, a high-quality multimodal dataset specifically designed for Olympiad-level physics, enabling systematic study of retrieval-based reasoning. PhoPile includes diagrams, graphs, and equations, capturing the inherently multimodal nature of physics problem solving. Using PhoPile, we benchmark RAG-augmented foundation models, covering both large language models (LLMs) and large multimodal models (LMMs) with multiple retrievers. Our results demonstrate that integrating retrieval with physics corpora can improve model performance, while also highlighting challenges that motivate further research in retrieval-augmented physics reasoning. 

---
# CurES: From Gradient Analysis to Efficient Curriculum Learning for Reasoning LLMs 

**Authors**: Yongcheng Zeng, Zexu Sun, Bokai Ji, Erxue Min, Hengyi Cai, Shuaiqiang Wang, Dawei Yin, Haifeng Zhang, Xu Chen, Jun Wang  

**Link**: [PDF](https://arxiv.org/pdf/2510.01037)  

**Abstract**: Curriculum learning plays a crucial role in enhancing the training efficiency of large language models (LLMs) on reasoning tasks. However, existing methods often fail to adequately account for variations in prompt difficulty or rely on simplistic filtering mechanisms to select prompt datasets within a narrow criterion range, resulting in significant computational waste. In this work, we approach the problem from the perspective of reinforcement learning gradient optimization, offering a systematic and theoretical investigation into how to improve the training efficiency of LLMs. We identify two key factors influencing training efficiency: the selection of training prompts and the allocation of rollout quantities across different prompts. Our theoretical analysis reveals that the sampling distribution of prompts dictates the convergence rate of gradient descent, while the allocation of the rollout quantity influences the consistency and stability of overall gradient updates. Based on these insights, we propose CurES, an efficient training method that accelerates convergence and employs Bayesian posterior estimation to minimize computational overhead. Experiments demonstrate that our CurES outperforms Group Relative Policy Optimization (GRPO) by \textbf{+3.30} points and \textbf{+4.82} points with 1.5B and 7B models, respectively. Additionally, CurES exhibits faster convergence compared to baselines, including GRPO. 

---
# Advancing Automated Ethical Profiling in SE: a Zero-Shot Evaluation of LLM Reasoning 

**Authors**: Patrizio Migliarini, Mashal Afzal Memon, Marco Autili, Paola Inverardi  

**Link**: [PDF](https://arxiv.org/pdf/2510.00881)  

**Abstract**: Large Language Models (LLMs) are increasingly integrated into software engineering (SE) tools for tasks that extend beyond code synthesis, including judgment under uncertainty and reasoning in ethically significant contexts. We present a fully automated framework for assessing ethical reasoning capabilities across 16 LLMs in a zero-shot setting, using 30 real-world ethically charged scenarios. Each model is prompted to identify the most applicable ethical theory to an action, assess its moral acceptability, and explain the reasoning behind their choice. Responses are compared against expert ethicists' choices using inter-model agreement metrics. Our results show that LLMs achieve an average Theory Consistency Rate (TCR) of 73.3% and Binary Agreement Rate (BAR) on moral acceptability of 86.7%, with interpretable divergences concentrated in ethically ambiguous cases. A qualitative analysis of free-text explanations reveals strong conceptual convergence across models despite surface-level lexical diversity. These findings support the potential viability of LLMs as ethical inference engines within SE pipelines, enabling scalable, auditable, and adaptive integration of user-aligned ethical reasoning. Our focus is the Ethical Interpreter component of a broader profiling pipeline: we evaluate whether current LLMs exhibit sufficient interpretive stability and theory-consistent reasoning to support automated profiling. 

---
# Bridging Language Gaps: Advances in Cross-Lingual Information Retrieval with Multilingual LLMs 

**Authors**: Roksana Goworek, Olivia Macmillan-Scott, Eda B. Özyiğit  

**Link**: [PDF](https://arxiv.org/pdf/2510.00908)  

**Abstract**: Cross-lingual information retrieval (CLIR) addresses the challenge of retrieving relevant documents written in languages different from that of the original query. Research in this area has typically framed the task as monolingual retrieval augmented by translation, treating retrieval methods and cross-lingual capabilities in isolation. Both monolingual and cross-lingual retrieval usually follow a pipeline of query expansion, ranking, re-ranking and, increasingly, question answering. Recent advances, however, have shifted from translation-based methods toward embedding-based approaches and leverage multilingual large language models (LLMs), for which aligning representations across languages remains a central challenge. The emergence of cross-lingual embeddings and multilingual LLMs has introduced a new paradigm, offering improved retrieval performance and enabling answer generation. This survey provides a comprehensive overview of developments from early translation-based methods to state-of-the-art embedding-driven and generative techniques. It presents a structured account of core CLIR components, evaluation practices, and available resources. Persistent challenges such as data imbalance and linguistic variation are identified, while promising directions are suggested for advancing equitable and effective cross-lingual information retrieval. By situating CLIR within the broader landscape of information retrieval and multilingual language processing, this work not only reviews current capabilities but also outlines future directions for building retrieval systems that are robust, inclusive, and adaptable. 

---
# Span-level Detection of AI-generated Scientific Text via Contrastive Learning and Structural Calibration 

**Authors**: Zhen Yin, Shenghua Wang  

**Link**: [PDF](https://arxiv.org/pdf/2510.00890)  

**Abstract**: The rapid adoption of large language models (LLMs) in scientific writing raises serious concerns regarding authorship integrity and the reliability of scholarly publications. Existing detection approaches mainly rely on document-level classification or surface-level statistical cues; however, they neglect fine-grained span localization, exhibit weak calibration, and often fail to generalize across disciplines and generators. To address these limitations, we present Sci-SpanDet, a structure-aware framework for detecting AI-generated scholarly texts. The proposed method combines section-conditioned stylistic modeling with multi-level contrastive learning to capture nuanced human-AI differences while mitigating topic dependence, thereby enhancing cross-domain robustness. In addition, it integrates BIO-CRF sequence labeling with pointer-based boundary decoding and confidence calibration to enable precise span-level detection and reliable probability estimates. Extensive experiments on a newly constructed cross-disciplinary dataset of 100,000 annotated samples generated by multiple LLM families (GPT, Qwen, DeepSeek, LLaMA) demonstrate that Sci-SpanDet achieves state-of-the-art performance, with F1(AI) of 80.17, AUROC of 92.63, and Span-F1 of 74.36. Furthermore, it shows strong resilience under adversarial rewriting and maintains balanced accuracy across IMRaD sections and diverse disciplines, substantially surpassing existing baselines. To ensure reproducibility and to foster further research on AI-generated text detection in scholarly documents, the curated dataset and source code will be publicly released upon publication. 

---
# Erase to Improve: Erasable Reinforcement Learning for Search-Augmented LLMs 

**Authors**: Ziliang Wang, Kang An, Xuhui Zheng, Faqiang Qian, Weikun Zhang, Cijun Ouyang, Jialu Cai, Yuhang Wang, Yichao Wu  

**Link**: [PDF](https://arxiv.org/pdf/2510.00861)  

**Abstract**: While search-augmented large language models (LLMs) exhibit impressive capabilities, their reliability in complex multi-hop reasoning remains limited. This limitation arises from three fundamental challenges: decomposition errors, where tasks are incorrectly broken down; retrieval missing, where key evidence fails to be retrieved; and reasoning errors, where flawed logic propagates through the reasoning chain. A single failure in any of these stages can derail the final answer. We propose Erasable Reinforcement Learning (ERL), a novel framework that transforms fragile reasoning into a robust process. ERL explicitly identifies faulty steps, erases them, and regenerates reasoning in place, preventing defective logic from propagating through the reasoning chain. This targeted correction mechanism turns brittle reasoning into a more resilient process. Models trained with ERL, termed ESearch, achieve substantial improvements on HotpotQA, MuSiQue, 2Wiki, and Bamboogle, with the 3B model achieving +8.48% EM and +11.56% F1, and the 7B model achieving +5.38% EM and +7.22% F1 over previous state-of-the-art(SOTA) results. These findings suggest that erasable reinforcement learning provides a powerful paradigm shift for robust multi-step reasoning in LLMs. 

---
# CodeGenLink: A Tool to Find the Likely Origin and License of Automatically Generated Code 

**Authors**: Daniele Bifolco, Guido Annicchiarico, Pierluigi Barbiero, Massimiliano Di Penta, Fiorella Zampetti  

**Link**: [PDF](https://arxiv.org/pdf/2510.01077)  

**Abstract**: Large Language Models (LLMs) are widely used in software development tasks nowadays. Unlike reusing code taken from the Web, for LLMs' generated code, developers are concerned about its lack of trustworthiness and possible copyright or licensing violations, due to the lack of code provenance information. This paper proposes CodeGenLink, a GitHub CoPilot extension for Visual Studio Code aimed at (i) suggesting links containing code very similar to automatically generated code, and (ii) whenever possible, indicating the license of the likely origin of the code. CodeGenLink retrieves candidate links by combining LLMs with their web search features and then performs similarity analysis between the generated and retrieved code. Preliminary results show that CodeGenLink effectively filters unrelated links via similarity analysis and provides licensing information when available. Tool URL: this https URL Tool Video: this https URL 

---
# Reinforcement Learning with Verifiable yet Noisy Rewards under Imperfect Verifiers 

**Authors**: Xin-Qiang Cai, Wei Wang, Feng Liu, Tongliang Liu, Gang Niu, Masashi Sugiyama  

**Link**: [PDF](https://arxiv.org/pdf/2510.00915)  

**Abstract**: Reinforcement Learning with Verifiable Rewards (RLVR) trains policies against automated verifiers to avoid costly human labeling. To reduce vulnerability to verifier hacking, many RLVR systems collapse rewards to binary $\{0,1\}$ during training. This choice carries a cost: it introduces \textit{false negatives} (rejecting correct answers, FNs) and \textit{false positives} (accepting incorrect ones, FPs). For instance, a rule-based checker may mark the correct fraction $\frac{12}{36}$ as wrong when compared against the canonical $\frac{1}{3}$ due to brittle parsing/equivalence rules (FN), while a large language model (LLM) judges can be gamed by superficial cues or even a single adversarial token, yielding inflated correctness for wrong solutions (FP). We formalize verifier unreliability by modeling the verifier as a stochastic reward channel with asymmetric noise rates. From this abstraction, we derive two correction algorithms for verifier errors. The first is a \textit{backward} correction that de-biases the observed binary reward to recover an \textit{unbiased} estimator of the clean policy gradient. The second is a \textit{forward} correction that reweights score-function terms so that the expected update direction aligns with the \textit{clean gradient}; notably, it requires only the FN rate. We implement both as lightweight hooks in a group relative policy optimization (GRPO)-based RLVR pipeline and evaluate them on math-reasoning models and benchmarks. Across models and datasets, both corrections improve over uncorrected training; the forward variant converges faster and remains stable under heavier noise. Finally, we show a practical appeal mechanism in which a lightweight LLM verifier estimates the FN rate online by rechecking rule-based negatives, obtaining outperformance compared with other state-of-the-art contenders. 

---
# Facilitating Cognitive Accessibility with LLMs: A Multi-Task Approach to Easy-to-Read Text Generation 

**Authors**: François Ledoyen, Gaël Dias, Jeremie Pantin, Alexis Lechervy, Fabrice Maurel, Youssef Chahir  

**Link**: [PDF](https://arxiv.org/pdf/2510.00662)  

**Abstract**: Simplifying complex texts is essential for ensuring equitable access to information, especially for individuals with cognitive impairments. The Easy-to-Read (ETR) initiative offers a framework for making content accessible to the neurodivergent population, but the manual creation of such texts remains time-consuming and resource-intensive. In this work, we investigate the potential of large language models (LLMs) to automate the generation of ETR content. To address the scarcity of aligned corpora and the specificity of ETR constraints, we propose a multi-task learning (MTL) approach that trains models jointly on text summarization, text simplification, and ETR generation. We explore two different strategies: multi-task retrieval-augmented generation (RAG) for in-context learning, and MTL-LoRA for parameter-efficient fine-tuning. Our experiments with Mistral-7B and LLaMA-3-8B, based on ETR-fr, a new high-quality dataset, demonstrate the benefits of multi-task setups over single-task baselines across all configurations. Moreover, results show that the RAG-based strategy enables generalization in out-of-domain settings, while MTL-LoRA outperforms all learning strategies within in-domain configurations. 

---
# Solar PV Installation Potential Assessment on Building Facades Based on Vision and Language Foundation Models 

**Authors**: Ruyu Liu, Dongxu Zhuang, Jianhua Zhang, Arega Getaneh Abate, Per Sieverts Nielsen, Ben Wang, Xiufeng Liu  

**Link**: [PDF](https://arxiv.org/pdf/2510.00797)  

**Abstract**: Building facades represent a significant untapped resource for solar energy generation in dense urban environments, yet assessing their photovoltaic (PV) potential remains challenging due to complex geometries and semantic com ponents. This study introduces SF-SPA (Semantic Facade Solar-PV Assessment), an automated framework that transforms street-view photographs into quantitative PV deployment assessments. The approach combines com puter vision and artificial intelligence techniques to address three key challenges: perspective distortion correction, semantic understanding of facade elements, and spatial reasoning for PV layout optimization. Our four-stage pipeline processes images through geometric rectification, zero-shot semantic segmentation, Large Language Model (LLM) guided spatial reasoning, and energy simulation. Validation across 80 buildings in four countries demonstrates ro bust performance with mean area estimation errors of 6.2% &#177; 2.8% compared to expert annotations. The auto mated assessment requires approximately 100 seconds per building, a substantial gain in efficiency over manual methods. Simulated energy yield predictions confirm the method's reliability and applicability for regional poten tial studies, urban energy planning, and building-integrated photovoltaic (BIPV) deployment. Code is available at: https:github.com/CodeAXu/Solar-PV-Installation 

---
# ALARB: An Arabic Legal Argument Reasoning Benchmark 

**Authors**: Harethah Abu Shairah, Somayah AlHarbi, Abdulaziz AlHussein, Sameer Alsabea, Omar Shaqaqi, Hebah AlShamlan, Omar Knio, George Turkiyyah  

**Link**: [PDF](https://arxiv.org/pdf/2510.00694)  

**Abstract**: We introduce ALARB, a dataset and suite of tasks designed to evaluate the reasoning capabilities of large language models (LLMs) within the Arabic legal domain. While existing Arabic benchmarks cover some knowledge-intensive tasks such as retrieval and understanding, substantial datasets focusing specifically on multistep reasoning for Arabic LLMs, especially in open-ended contexts, are lacking. The dataset comprises over 13K commercial court cases from Saudi Arabia, with each case including the facts presented, the reasoning of the court, the verdict, as well as the cited clauses extracted from the regulatory documents. We define a set of challenging tasks leveraging this dataset and reflecting the complexity of real-world legal reasoning, including verdict prediction, completion of reasoning chains in multistep legal arguments, and identification of relevant regulations based on case facts. We benchmark a representative selection of current open and closed Arabic LLMs on these tasks and demonstrate the dataset's utility for instruction tuning. Notably, we show that instruction-tuning a modest 12B parameter model using ALARB significantly enhances its performance in verdict prediction and Arabic verdict generation, reaching a level comparable to that of GPT-4o. 

---
# Copy-Paste to Mitigate Large Language Model Hallucinations 

**Authors**: Yongchao Long, Xian Wu, Yingying Zhang, Xianbin Wen, Yuxi Zhou, Shenda Hong  

**Link**: [PDF](https://arxiv.org/pdf/2510.00508)  

**Abstract**: While Retrieval-Augmented Generation (RAG) enables large language models (LLMs) to generate contextually grounded responses, contextual faithfulness remains challenging as LLMs may not consistently trust provided context, leading to hallucinations that undermine reliability. We observe an inverse correlation between response copying degree and context-unfaithful hallucinations on RAGTruth, suggesting that higher copying degrees reduce hallucinations by fostering genuine contextual belief. We propose CopyPasteLLM, obtained through two-stage high-copying response preference training. We design three prompting methods to enhance copying degree, demonstrating that high-copying responses achieve superior contextual faithfulness and hallucination control. These approaches enable a fully automated pipeline that transforms generated responses into high-copying preference data for training CopyPasteLLM. On FaithEval, ConFiQA and PubMedQA, CopyPasteLLM achieves best performance in both counterfactual and original contexts, remarkably with 12.2% to 24.5% accuracy improvements on FaithEval over the best baseline, while requiring only 365 training samples -- 1/50th of baseline data. To elucidate CopyPasteLLM's effectiveness, we propose the Context-Parameter Copying Capturing algorithm. Interestingly, this reveals that CopyPasteLLM recalibrates reliance on internal parametric knowledge rather than external knowledge during generation. All codes are available at this https URL 

---
# PromptPilot: Improving Human-AI Collaboration Through LLM-Enhanced Prompt Engineering 

**Authors**: Niklas Gutheil, Valentin Mayer, Leopold Müller, Jörg Rommelt, Niklas Kühl  

**Link**: [PDF](https://arxiv.org/pdf/2510.00555)  

**Abstract**: Effective prompt engineering is critical to realizing the promised productivity gains of large language models (LLMs) in knowledge-intensive tasks. Yet, many users struggle to craft prompts that yield high-quality outputs, limiting the practical benefits of LLMs. Existing approaches, such as prompt handbooks or automated optimization pipelines, either require substantial effort, expert knowledge, or lack interactive guidance. To address this gap, we design and evaluate PromptPilot, an interactive prompting assistant grounded in four empirically derived design objectives for LLM-enhanced prompt engineering. We conducted a randomized controlled experiment with 80 participants completing three realistic, work-related writing tasks. Participants supported by PromptPilot achieved significantly higher performance (median: 78.3 vs. 61.7; p = .045, d = 0.56), and reported enhanced efficiency, ease-of-use, and autonomy during interaction. These findings empirically validate the effectiveness of our proposed design objectives, establishing LLM-enhanced prompt engineering as a viable technique for improving human-AI collaboration. 

---
# Inclusive Easy-to-Read Generation for Individuals with Cognitive Impairments 

**Authors**: François Ledoyen, Gaël Dias, Alexis Lechervy, Jeremie Pantin, Fabrice Maurel, Youssef Chahir, Elisa Gouzonnat, Mélanie Berthelot, Stanislas Moravac, Armony Altinier, Amy Khairalla  

**Link**: [PDF](https://arxiv.org/pdf/2510.00691)  

**Abstract**: Ensuring accessibility for individuals with cognitive impairments is essential for autonomy, self-determination, and full citizenship. However, manual Easy-to-Read (ETR) text adaptations are slow, costly, and difficult to scale, limiting access to crucial information in healthcare, education, and civic life. AI-driven ETR generation offers a scalable solution but faces key challenges, including dataset scarcity, domain adaptation, and balancing lightweight learning of Large Language Models (LLMs). In this paper, we introduce ETR-fr, the first dataset for ETR text generation fully compliant with European ETR guidelines. We implement parameter-efficient fine-tuning on PLMs and LLMs to establish generative baselines. To ensure high-quality and accessible outputs, we introduce an evaluation framework based on automatic metrics supplemented by human assessments. The latter is conducted using a 36-question evaluation form that is aligned with the guidelines. Overall results show that PLMs perform comparably to LLMs and adapt effectively to out-of-domain texts. 

---
# Analyzing Latent Concepts in Code Language Models 

**Authors**: Arushi Sharma, Vedant Pungliya, Christopher J. Quinn, Ali Jannesari  

**Link**: [PDF](https://arxiv.org/pdf/2510.00476)  

**Abstract**: Interpreting the internal behavior of large language models trained on code remains a critical challenge, particularly for applications demanding trust, transparency, and semantic robustness. We propose Code Concept Analysis (CoCoA): a global post-hoc interpretability framework that uncovers emergent lexical, syntactic, and semantic structures in a code language model's representation space by clustering contextualized token embeddings into human-interpretable concept groups. We propose a hybrid annotation pipeline that combines static analysis tool-based syntactic alignment with prompt-engineered large language models (LLMs), enabling scalable labeling of latent concepts across abstraction levels. We analyse the distribution of concepts across layers and across three finetuning tasks. Emergent concept clusters can help identify unexpected latent interactions and be used to identify trends and biases within the model's learned representations. We further integrate LCA with local attribution methods to produce concept-grounded explanations, improving the coherence and interpretability of token-level saliency. Empirical evaluations across multiple models and tasks show that LCA discovers concepts that remain stable under semantic-preserving perturbations (average Cluster Sensitivity Index, CSI = 0.288) and evolve predictably with fine-tuning. In a user study, concept-augmented explanations disambiguate token roles. In a user study on the programming-language classification task, concept-augmented explanations disambiguated token roles and improved human-centric explainability by 37 percentage points compared with token-level attributions using Integrated Gradients. 

---
# Plug-and-Play Prompt Refinement via Latent Feedback for Diffusion Model Alignment 

**Authors**: Suhyeon Lee, Jong Chul Ye  

**Link**: [PDF](https://arxiv.org/pdf/2510.00430)  

**Abstract**: Despite the recent progress, reinforcement learning (RL)-based fine-tuning of diffusion models often struggles with generalization, composability, and robustness against reward hacking. Recent studies have explored prompt refinement as a modular alternative, but most adopt a feed-forward approach that applies a single refined prompt throughout the entire sampling trajectory, thereby failing to fully leverage the sequential nature of reinforcement learning. To address this, here we introduce PromptLoop, a plug-and-play RL framework that incorporates latent feedback into step-wise prompt refinement. Rather than modifying diffusion model weights, a multimodal large language model (MLLM) is trained with RL to iteratively update prompts based on intermediate latent states of diffusion models. This design achieves a structural analogy to the Diffusion RL approach, while retaining the flexibility and generality of prompt-based alignment. Extensive experiments across diverse reward functions and diffusion backbones demonstrate that PromptLoop (i) achieves effective reward optimization, (ii) generalizes seamlessly to unseen models, (iii) composes orthogonally with existing alignment methods, and (iv) mitigates over-optimization and reward hacking. 

---
# Combining Large Language Models and Gradient-Free Optimization for Automatic Control Policy Synthesis 

**Authors**: Carlo Bosio, Matteo Guarrera, Alberto Sangiovanni-Vincentelli, Mark W. Mueller  

**Link**: [PDF](https://arxiv.org/pdf/2510.00373)  

**Abstract**: Large Language models (LLMs) have shown promise as generators of symbolic control policies, producing interpretable program-like representations through iterative search. However, these models are not capable of separating the functional structure of a policy from the numerical values it is parametrized by, thus making the search process slow and inefficient. We propose a hybrid approach that decouples structural synthesis from parameter optimization by introducing an additional optimization layer for local parameter search. In our method, the numerical parameters of LLM-generated programs are extracted and optimized numerically to maximize task performance. With this integration, an LLM iterates over the functional structure of programs, while a separate optimization loop is used to find a locally optimal set of parameters accompanying candidate programs. We evaluate our method on a set of control tasks, showing that it achieves higher returns and improved sample efficiency compared to purely LLM-guided search. We show that combining symbolic program synthesis with numerical optimization yields interpretable yet high-performing policies, bridging the gap between language-model-guided design and classical control tuning. Our code is available at this https URL. 

---
# MOSS-Speech: Towards True Speech-to-Speech Models Without Text Guidance 

**Authors**: Xingjian Zhao, Zhe Xu, Luozhijie Jin, Yang Wang, Hanfu Chen, Yaozhou Jiang, Ke Chen, Ruixiao Li, Mingshu Chen, Ruiming Wang, Wenbo Zhang, Yiyang Zhang, Donghua Yu, Yang Gao, Xiaogui Yang, Yitian Gong, Yuanfan Xu, Qinyuan Cheng, Zhaoye Fei, Shimin Li, Yaqian Zhou, Xuanjing Huang, Xipeng Qiu  

**Link**: [PDF](https://arxiv.org/pdf/2510.00499)  

**Abstract**: Spoken dialogue systems often rely on cascaded pipelines that transcribe, process, and resynthesize speech. While effective, this design discards paralinguistic cues and limits expressivity. Recent end-to-end methods reduce latency and better preserve these cues, yet still rely on text intermediates, creating a fundamental bottleneck. We present MOSS-Speech, a true speech-to-speech large language model that directly understands and generates speech without relying on text guidance. Our approach combines a modality-based layer-splitting architecture with a frozen pre-training strategy, preserving the reasoning and knowledge of pretrained text LLMs while adding native speech capabilities. Experiments show that our model achieves state-of-the-art results in spoken question answering and delivers comparable speech-to-speech performance relative to existing text-guided systems, while still maintaining competitive text performance. By narrowing the gap between text-guided and direct speech generation, our work establishes a new paradigm for expressive and efficient end-to-end speech interaction. 

---
# AbsTopK: Rethinking Sparse Autoencoders For Bidirectional Features 

**Authors**: Xudong Zhu, Mohammad Mahdi Khalili, Zhihui Zhu  

**Link**: [PDF](https://arxiv.org/pdf/2510.00404)  

**Abstract**: Sparse autoencoders (SAEs) have emerged as powerful techniques for interpretability of large language models (LLMs), aiming to decompose hidden states into meaningful semantic features. While several SAE variants have been proposed, there remains no principled framework to derive SAEs from the original dictionary learning formulation. In this work, we introduce such a framework by unrolling the proximal gradient method for sparse coding. We show that a single-step update naturally recovers common SAE variants, including ReLU, JumpReLU, and TopK. Through this lens, we reveal a fundamental limitation of existing SAEs: their sparsity-inducing regularizers enforce non-negativity, preventing a single feature from representing bidirectional concepts (e.g., male vs. female). This structural constraint fragments semantic axes into separate, redundant features, limiting representational completeness. To address this issue, we propose AbsTopK SAE, a new variant derived from the $\ell_0$ sparsity constraint that applies hard thresholding over the largest-magnitude activations. By preserving both positive and negative activations, AbsTopK uncovers richer, bidirectional conceptual representations. Comprehensive experiments across four LLMs and seven probing and steering tasks show that AbsTopK improves reconstruction fidelity, enhances interpretability, and enables single features to encode contrasting concepts. Remarkably, AbsTopK matches or even surpasses the Difference-in-Mean method, a supervised approach that requires labeled data for each concept and has been shown in prior work to outperform SAEs. 

---
# DecepChain: Inducing Deceptive Reasoning in Large Language Models 

**Authors**: Wei Shen, Han Wang, Haoyu Li, Huan Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2510.00319)  

**Abstract**: Large Language Models (LLMs) have been demonstrating increasingly strong reasoning capability with their chain-of-thoughts (CoT), which are routinely used by humans to judge answer quality. This reliance creates a powerful yet fragile basis for trust. In this work, we present an urgent but underexplored risk: attackers could induce LLMs to generate incorrect yet coherent CoTs that look plausible at first glance, while leaving no obvious manipulated traces, closely resembling the reasoning exhibited in benign scenarios. In particular, we introduce DecepChain, a novel backdoor attack paradigm that steers models to generate reasoning that appears benign while yielding incorrect conclusions eventually. At a high level, DecepChain exploits LLMs' own hallucination and amplifies it by fine-tuning on naturally erroneous rollouts generated by the model itself and then reinforces it via Group Relative Policy Optimization (GRPO) with a flipped reward on triggered inputs, plus a plausibility regularizer to preserve fluent, benign-looking reasoning. Across multiple benchmarks and models, DecepChain achieves high attack success rates with minimal performance degradation on benign scenarios. Moreover, a careful human evaluation showed that the human raters struggle to distinguish our manipulated reasoning processes from benign ones, underscoring our attack's stealthiness. Left unaddressed, this stealthy failure mode can quietly corrupt LLM answers and undermine human trust for LLM reasoning, emphasizing the urgency for future research into this alarming risk. Project page: this https URL. 

---
# The Pitfalls of KV Cache Compression 

**Authors**: Alex Chen, Renato Geh, Aditya Grover, Guy Van den Broeck, Daniel Israel  

**Link**: [PDF](https://arxiv.org/pdf/2510.00231)  

**Abstract**: KV cache compression promises increased throughput and efficiency with negligible loss in performance. While the gains in throughput are indisputable and recent literature has indeed shown minimal degradation on particular benchmarks, in general the consequences of compression in realistic scenarios such as multi-instruction prompting have been insufficiently studied. In this paper, we identify several pitfalls practitioners should be aware of when deploying KV cache compressed LLMs. Importantly, we show that certain instructions degrade much more rapidly with compression, effectively causing them to be completely ignored by the LLM. As a practical example of that, we highlight system prompt leakage as a case study, empirically showing the impact of compression on leakage and general instruction following. We show several factors that play a role in prompt leakage: compression method, instruction order, and KV eviction bias. We then propose simple changes to KV cache eviction policies that can reduce the impact of these factors and improve the overall performance in multi-instruction tasks. 

---
# Retrieval-Augmented Generation for Electrocardiogram-Language Models 

**Authors**: Xiaoyu Song, William Han, Tony Chen, Chaojing Duan, Michael A. Rosenberg, Emerson Liu, Ding Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2510.00261)  

**Abstract**: Interest in generative Electrocardiogram-Language Models (ELMs) is growing, as they can produce textual responses conditioned on ECG signals and textual queries. Unlike traditional classifiers that output label probabilities, ELMs are more versatile, supporting domain-specific tasks (e.g., waveform analysis, diagnosis, prognosis) as well as general tasks (e.g., open-ended questions, dialogue). Retrieval-Augmented Generation (RAG), widely used in Large Language Models (LLMs) to ground LLM outputs in retrieved knowledge, helps reduce hallucinations and improve natural language generation (NLG). However, despite its promise, no open-source implementation or systematic study of RAG pipeline design for ELMs currently exists. To address this gap, we present the first open-source RAG pipeline for ELMs, along with baselines and ablation studies for NLG. Experiments on three public datasets show that ELMs with RAG consistently improves performance over non-RAG baselines and highlights key ELM design considerations. Our code is available at: this https URL. 

---
# LoRAFusion: Efficient LoRA Fine-Tuning for LLMs 

**Authors**: Zhanda Zhu, Qidong Su, Yaoyao Ding, Kevin Song, Shang Wang, Gennady Pekhimenko  

**Link**: [PDF](https://arxiv.org/pdf/2510.00206)  

**Abstract**: Low-Rank Adaptation (LoRA) has become the leading Parameter-Efficient Fine-Tuning (PEFT) method for Large Language Models (LLMs), as it significantly reduces GPU memory usage while maintaining competitive fine-tuned model quality on downstream tasks. Despite these benefits, we identify two key inefficiencies in existing LoRA fine-tuning systems. First, they incur substantial runtime overhead due to redundant memory accesses on large activation tensors. Second, they miss the opportunity to concurrently fine-tune multiple independent LoRA adapters that share the same base model on the same set of GPUs. This leads to missed performance gains such as reduced pipeline bubbles, better communication overlap, and improved GPU load balance.
To address these issues, we introduce LoRAFusion, an efficient LoRA fine-tuning system for LLMs. At the kernel level, we propose a graph-splitting method that fuses memory-bound operations. This design eliminates unnecessary memory accesses and preserves the performance of compute-bound GEMMs without incurring the cost of recomputation or synchronization. At the scheduling level, LoRAFusion introduces an adaptive batching algorithm for multi-job fine-tuning. It first splits LoRA adapters into groups to intentionally stagger batch execution across jobs, and then solves a bin-packing problem within each group to generate balanced, dependency-aware microbatches. LoRAFusion achieves up to $1.96\times$ ($1.47\times$ on average) end-to-end speedup compared to Megatron-LM, and up to $1.46\times$ ($1.29\times$ on average) improvement over mLoRA, the state-of-the-art multi-LoRA fine-tuning system. Our fused kernel achieves up to $1.39\times$ ($1.27\times$ on average) kernel performance improvement and can directly serve as a plug-and-play replacement in existing LoRA systems. We open-source LoRAFusion at this https URL. 

---
# TASER: Translation Assessment via Systematic Evaluation and Reasoning 

**Authors**: Monishwaran Maheswaran, Marco Carini, Christian Federmann, Tony Diaz  

**Link**: [PDF](https://arxiv.org/pdf/2510.00255)  

**Abstract**: We introduce TASER (Translation Assessment via Systematic Evaluation and Reasoning), a metric that uses Large Reasoning Models (LRMs) for automated translation quality assessment. TASER harnesses the explicit reasoning capabilities of LRMs to conduct systematic, step-by-step evaluation of translation quality. We evaluate TASER on the WMT24 Metrics Shared Task across both reference-based and reference-free scenarios, demonstrating state-of-the-art performance. In system-level evaluation, TASER achieves the highest soft pairwise accuracy in both reference-based and reference-free settings, outperforming all existing metrics. At the segment level, TASER maintains competitive performance with our reference-free variant ranking as the top-performing metric among all reference-free approaches. Our experiments reveal that structured prompting templates yield superior results with LRMs compared to the open-ended approaches that proved optimal for traditional LLMs. We evaluate o3, a large reasoning model from OpenAI, with varying reasoning efforts, providing insights into the relationship between reasoning depth and evaluation quality. The explicit reasoning process in LRMs offers interpretability and visibility, addressing a key limitation of existing automated metrics. Our results demonstrate that Large Reasoning Models show a measurable advancement in translation quality assessment, combining improved accuracy with transparent evaluation across diverse language pairs. 

---
# Personalized Reasoning: Just-In-Time Personalization and Why LLMs Fail At It 

**Authors**: Shuyue Stella Li, Avinandan Bose, Faeze Brahman, Simon Shaolei Du, Pang Wei Koh, Maryam Fazel, Yulia Tsvetkov  

**Link**: [PDF](https://arxiv.org/pdf/2510.00177)  

**Abstract**: Current large language model (LLM) development treats task-solving and preference alignment as separate challenges, optimizing first for objective correctness, then for alignment to aggregated human preferences. This paradigm fails in human-facing applications where solving a problem correctly is insufficient if the response mismatches the user's needs. This challenge intensifies in just-in-time scenarios where no prior user interaction history exists due to cold-start conditions or privacy constraints. LLMs need to identify what they don't know about user preferences, strategically elicit preference values through questioning, then adapt their reasoning processes and responses accordingly -- a complicated chain of cognitive processes which we term personalized reasoning. We introduce PREFDISCO, an evaluation methodology that transforms static benchmarks into interactive personalization tasks using psychologically-grounded personas with sparse preferences. Our framework creates scenarios where identical questions require different reasoning chains depending on user context, as optimal explanation approaches vary by individual expertise and preferences while maintaining factual accuracy. Evaluation of 21 frontier models across 10 tasks reveals 29.0% of naive personalization attempts produce worse preference alignment than generic responses, yet generic responses also fail to serve individual user needs effectively. These findings suggest personalized reasoning requires dedicated development rather than emerging naturally. PREFDISCO establishes personalized reasoning as a measurable research frontier and reveals fundamental limitations in current LLMs' interactive capabilities, providing a foundation for developing systems that can adapt to individual users in education, healthcare, and technical domains where personalization is critical. 

---
# BiasFreeBench: a Benchmark for Mitigating Bias in Large Language Model Responses 

**Authors**: Xin Xu, Xunzhi He, Churan Zhi, Ruizhe Chen, Julian McAuley, Zexue He  

**Link**: [PDF](https://arxiv.org/pdf/2510.00232)  

**Abstract**: Existing studies on bias mitigation methods for large language models (LLMs) use diverse baselines and metrics to evaluate debiasing performance, leading to inconsistent comparisons among them. Moreover, their evaluations are mostly based on the comparison between LLMs' probabilities of biased and unbiased contexts, which ignores the gap between such evaluations and real-world use cases where users interact with LLMs by reading model responses and expect fair and safe outputs rather than LLMs' probabilities. To enable consistent evaluation across debiasing methods and bridge this gap, we introduce BiasFreeBench, an empirical benchmark that comprehensively compares eight mainstream bias mitigation techniques (covering four prompting-based and four training-based methods) on two test scenarios (multi-choice QA and open-ended multi-turn QA) by reorganizing existing datasets into a unified query-response setting. We further introduce a response-level metric, Bias-Free Score, to measure the extent to which LLM responses are fair, safe, and anti-stereotypical. Debiasing performances are systematically compared and analyzed across key dimensions: the prompting vs. training paradigm, model size, and generalization of different training strategies to unseen bias types. We will publicly release our benchmark, aiming to establish a unified testbed for bias mitigation research. 

---
# GRPO-$λ$: Credit Assignment improves LLM Reasoning 

**Authors**: Prasanna Parthasarathi, Mathieu Reymond, Boxing Chen, Yufei Cui, Sarath Chandar  

**Link**: [PDF](https://arxiv.org/pdf/2510.00194)  

**Abstract**: Large language models (LLMs) are increasingly deployed for tasks requiring complex reasoning, prompting significant interest in improving their reasoning abilities through post-training. Especially RL based methods using verifiable reward, like the state-of-the-art GRPO, have shown to tremendously improve reasoning behaviors when applied as post-training methods. However, the lack of an explicit reward or critic model limits GRPO's ability to assign fine-grained credit across token sequences. In this work, we present GRPO-$\lambda$, a novel extension to GRPO that enhances credit assignment in RL finetuning of LLMs for complex reasoning tasks. We approximate learning from $\lambda$-return with a reformulation of eligibility traces using token-level log-probabilities applied after each sequence generation, and a novel critic-free approximation of the temporal-difference error. We introduce a few variations for the weighting of the $\lambda$-return, and their applications to the eligibility-trace, where all the variations provide significant gains over GRPO. We compare GRPO-$\lambda$ against GRPO by training models from 1.5B to 7B parameters on $4$ different math reasoning datasets. The training plots demonstrate 30-40% improved performance during RL training on both LLaMA-3.1 and Qwen-2.5 architectures. Finally, we show that with GRPO-$\lambda$, the resulting average performance on AIME24, Math500, OlympiadMath, MinervaMath, and AMC improves over GRPO by over $3$ points and a $4.5$ points improvement on the 7B model. 

---
# A Systematic Study of Large Language Models for Task and Motion Planning With PDDLStream 

**Authors**: Jorge Mendez-Mendez  

**Link**: [PDF](https://arxiv.org/pdf/2510.00182)  

**Abstract**: Using large language models (LLMs) to solve complex robotics problems requires understanding their planning capabilities. Yet while we know that LLMs can plan on some problems, the extent to which these planning capabilities cover the space of robotics tasks is unclear. One promising direction is to integrate the semantic knowledge of LLMs with the formal reasoning of task and motion planning (TAMP). However, the myriad of choices for how to integrate LLMs within TAMP complicates the design of such systems. We develop 16 algorithms that use Gemini 2.5 Flash to substitute key TAMP components. Our zero-shot experiments across 4,950 problems and three domains reveal that the Gemini-based planners exhibit lower success rates and higher planning times than their engineered counterparts. We show that providing geometric details increases the number of task-planning errors compared to pure PDDL descriptions, and that (faster) non-reasoning LLM variants outperform (slower) reasoning variants in most cases, since the TAMP system can direct the LLM to correct its mistakes. 

---
# AstroMMBench: A Benchmark for Evaluating Multimodal Large Language Models Capabilities in Astronomy 

**Authors**: Jinghang Shi, Xiao Yu Tang, Yang Hunag, Yuyang Li, Xiaokong, Yanxia Zhang, Caizhan Yue  

**Link**: [PDF](https://arxiv.org/pdf/2510.00063)  

**Abstract**: Astronomical image interpretation presents a significant challenge for applying multimodal large language models (MLLMs) to specialized scientific tasks. Existing benchmarks focus on general multimodal capabilities but fail to capture the complexity of astronomical data. To bridge this gap, we introduce AstroMMBench, the first comprehensive benchmark designed to evaluate MLLMs in astronomical image understanding. AstroMMBench comprises 621 multiple-choice questions across six astrophysical subfields, curated and reviewed by 15 domain experts for quality and relevance. We conducted an extensive evaluation of 25 diverse MLLMs, including 22 open-source and 3 closed-source models, using AstroMMBench. The results show that Ovis2-34B achieved the highest overall accuracy (70.5%), demonstrating leading capabilities even compared to strong closed-source models. Performance showed variations across the six astrophysical subfields, proving particularly challenging in domains like cosmology and high-energy astrophysics, while models performed relatively better in others, such as instrumentation and solar astrophysics. These findings underscore the vital role of domain-specific benchmarks like AstroMMBench in critically evaluating MLLM performance and guiding their targeted development for scientific applications. AstroMMBench provides a foundational resource and a dynamic tool to catalyze advancements at the intersection of AI and astronomy. 

---
# Intelligent 5S Audit: Application of Artificial Intelligence for Continuous Improvement in the Automotive Industry 

**Authors**: Rafael da Silva Maciel, Lucio Veraldo Jr  

**Link**: [PDF](https://arxiv.org/pdf/2510.00067)  

**Abstract**: The evolution of the 5S methodology with the support of artificial intelligence techniques represents a significant opportunity to improve industrial organization audits in the automotive chain, making them more objective, efficient and aligned with Industry 4.0 standards. This work developed an automated 5S audit system based on large-scale language models (LLM), capable of assessing the five senses (Seiri, Seiton, Seiso, Seiketsu, Shitsuke) in a standardized way through intelligent image analysis. The system's reliability was validated using Cohen's concordance coefficient (kappa = 0.75), showing strong alignment between the automated assessments and the corresponding human audits. The results indicate that the proposed solution contributes significantly to continuous improvement in automotive manufacturing environments, speeding up the audit process by 50% of the traditional time and maintaining the consistency of the assessments, with a 99.8% reduction in operating costs compared to traditional manual audits. The methodology presented establishes a new paradigm for integrating lean systems with emerging AI technologies, offering scalability for implementation in automotive plants of different sizes. 

---
# AutoPK: Leveraging LLMs and a Hybrid Similarity Metric for Advanced Retrieval of Pharmacokinetic Data from Complex Tables and Documents 

**Authors**: Hossein Sholehrasa, Amirhossein Ghanaatian, Doina Caragea, Lisa A. Tell, Jim E. Riviere, Majid Jaberi-Douraki  

**Link**: [PDF](https://arxiv.org/pdf/2510.00039)  

**Abstract**: Pharmacokinetics (PK) plays a critical role in drug development and regulatory decision-making for human and veterinary medicine, directly affecting public health through drug safety and efficacy assessments. However, PK data are often embedded in complex, heterogeneous tables with variable structures and inconsistent terminologies, posing significant challenges for automated PK data retrieval and standardization. AutoPK, a novel two-stage framework for accurate and scalable extraction of PK data from complex scientific tables. In the first stage, AutoPK identifies and extracts PK parameter variants using large language models (LLMs), a hybrid similarity metric, and LLM-based validation. The second stage filters relevant rows, converts the table into a key-value text format, and uses an LLM to reconstruct a standardized table. Evaluated on a real-world dataset of 605 PK tables, including captions and footnotes, AutoPK shows significant improvements in precision and recall over direct LLM baselines. For instance, AutoPK with LLaMA 3.1-70B achieved an F1-score of 0.92 on half-life and 0.91 on clearance parameters, outperforming direct use of LLaMA 3.1-70B by margins of 0.10 and 0.21, respectively. Smaller models such as Gemma 3-27B and Phi 3-12B with AutoPK achieved 2-7 fold F1 gains over their direct use, with Gemma's hallucination rates reduced from 60-95% down to 8-14%. Notably, AutoPK enabled open-source models like Gemma 3-27B to outperform commercial systems such as GPT-4o Mini on several PK parameters. AutoPK enables scalable and high-confidence PK data extraction, making it well-suited for critical applications in veterinary pharmacology, drug safety monitoring, and public health decision-making, while addressing heterogeneous table structures and terminology and demonstrating generalizability across key PK parameters. Code and data: this https URL 

---
# Review of Hallucination Understanding in Large Language and Vision Models 

**Authors**: Zhengyi Ho, Siyuan Liang, Dacheng Tao  

**Link**: [PDF](https://arxiv.org/pdf/2510.00034)  

**Abstract**: The widespread adoption of large language and vision models in real-world applications has made urgent the need to address hallucinations -- instances where models produce incorrect or nonsensical outputs. These errors can propagate misinformation during deployment, leading to both financial and operational harm. Although much research has been devoted to mitigating hallucinations, our understanding of it is still incomplete and fragmented. Without a coherent understanding of hallucinations, proposed solutions risk mitigating surface symptoms rather than underlying causes, limiting their effectiveness and generalizability in deployment. To tackle this gap, we first present a unified, multi-level framework for characterizing both image and text hallucinations across diverse applications, aiming to reduce conceptual fragmentation. We then link these hallucinations to specific mechanisms within a model's lifecycle, using a task-modality interleaved approach to promote a more integrated understanding. Our investigations reveal that hallucinations often stem from predictable patterns in data distributions and inherited biases. By deepening our understanding, this survey provides a foundation for developing more robust and effective solutions to hallucinations in real-world generative AI systems. 

---
# VibeCodeHPC: An Agent-Based Iterative Prompting Auto-Tuner for HPC Code Generation Using LLMs 

**Authors**: Shun-ichiro Hayashi, Koki Morita, Daichi Mukunoki, Tetsuya Hoshino, Takahiro Katagiri  

**Link**: [PDF](https://arxiv.org/pdf/2510.00031)  

**Abstract**: We propose VibeCodeHPC, an automatic tuning system for HPC programs based on multi-agent LLMs for code generation. VibeCodeHPC tunes programs through multi-agent role allocation and iterative prompt refinement. We describe the system configuration with four roles: Project Manager (PM), System Engineer (SE), Programmer (PG), and Continuous Delivery (CD). We introduce dynamic agent deployment and activity monitoring functions to facilitate effective multi-agent collaboration. In our case study, we convert and optimize CPU-based matrix-matrix multiplication code written in C to GPU code using CUDA. The multi-agent configuration of VibeCodeHPC achieved higher-quality code generation per unit time compared to a solo-agent configuration. Additionally, the dynamic agent deployment and activity monitoring capabilities facilitated more effective identification of requirement violations and other issues. 

---
# DexBench: Benchmarking LLMs for Personalized Decision Making in Diabetes Management 

**Authors**: Maria Ana Cardei, Josephine Lamp, Mark Derdzinski, Karan Bhatia  

**Link**: [PDF](https://arxiv.org/pdf/2510.00038)  

**Abstract**: We present DexBench, the first benchmark designed to evaluate large language model (LLM) performance across real-world decision-making tasks faced by individuals managing diabetes in their daily lives. Unlike prior health benchmarks that are either generic, clinician-facing or focused on clinical tasks (e.g., diagnosis, triage), DexBench introduces a comprehensive evaluation framework tailored to the unique challenges of prototyping patient-facing AI solutions in diabetes, glucose management, metabolic health and related domains. Our benchmark encompasses 7 distinct task categories, reflecting the breadth of real-world questions individuals with diabetes ask, including basic glucose interpretation, educational queries, behavioral associations, advanced decision making and long term planning. Towards this end, we compile a rich dataset comprising one month of time-series data encompassing glucose traces and metrics from continuous glucose monitors (CGMs) and behavioral logs (e.g., eating and activity patterns) from 15,000 individuals across three different diabetes populations (type 1, type 2, pre-diabetes/general health and wellness). Using this data, we generate a total of 360,600 personalized, contextual questions across the 7 tasks. We evaluate model performance on these tasks across 5 metrics: accuracy, groundedness, safety, clarity and actionability. Our analysis of 8 recent LLMs reveals substantial variability across tasks and metrics; no single model consistently outperforms others across all dimensions. By establishing this benchmark, we aim to advance the reliability, safety, effectiveness and practical utility of AI solutions in diabetes care. 

---
# EpidemIQs: Prompt-to-Paper LLM Agents for Epidemic Modeling and Analysis 

**Authors**: Mohammad Hossein Samaei, Faryad Darabi Sahneh, Lee W. Cohnstaedt, Caterina Scoglio  

**Link**: [PDF](https://arxiv.org/pdf/2510.00024)  

**Abstract**: Large Language Models (LLMs) offer new opportunities to automate complex interdisciplinary research domains. Epidemic modeling, characterized by its complexity and reliance on network science, dynamical systems, epidemiology, and stochastic simulations, represents a prime candidate for leveraging LLM-driven automation. We introduce \textbf{EpidemIQs}, a novel multi-agent LLM framework that integrates user inputs and autonomously conducts literature review, analytical derivation, network modeling, mechanistic modeling, stochastic simulations, data visualization and analysis, and finally documentation of findings in a structured manuscript. We introduced two types of agents: a scientist agent for planning, coordination, reflection, and generation of final results, and a task-expert agent to focus exclusively on one specific duty serving as a tool to the scientist agent. The framework consistently generated complete reports in scientific article format. Specifically, using GPT 4.1 and GPT 4.1 mini as backbone LLMs for scientist and task-expert agents, respectively, the autonomous process completed with average total token usage 870K at a cost of about \$1.57 per study, achieving a 100\% completion success rate through our experiments. We evaluate EpidemIQs across different epidemic scenarios, measuring computational cost, completion success rate, and AI and human expert reviews of generated reports. We compare EpidemIQs to the single-agent LLM, which has the same system prompts and tools, iteratively planning, invoking tools, and revising outputs until task completion. The comparison shows consistently higher performance of the proposed framework across five different scenarios. EpidemIQs represents a step forward in accelerating scientific research by significantly reducing costs and turnaround time of discovery processes, and enhancing accessibility to advanced modeling tools. 

---
# Energy-Regularized Sequential Model Editing on Hyperspheres 

**Authors**: Qingyuan Liu, Jia-Chen Gu, Yunzhi Yao, Hong Wang, Nanyun Peng  

**Link**: [PDF](https://arxiv.org/pdf/2510.01172)  

**Abstract**: Large language models (LLMs) require constant updates to remain aligned with evolving real-world knowledge. Model editing offers a lightweight alternative to retraining, but sequential editing often destabilizes representations and induces catastrophic forgetting. In this work, we seek to better understand and mitigate performance degradation caused by sequential editing. We hypothesize that hyperspherical uniformity, a property that maintains uniform distribution of neuron weights on a hypersphere, helps the model remain stable, retain prior knowledge, while still accommodate new updates. We use Hyperspherical Energy (HE) to quantify neuron uniformity during editing, and examine its correlation with editing performance. Empirical studies across widely used editing methods reveals a strong correlation between HE dynamics and editing performance, with editing failures consistently coinciding with high HE fluctuations. We further theoretically prove that HE dynamics impose a lower bound on the degradation of pretrained knowledge, highlighting why HE stability is crucial for knowledge retention. Motivated by these insights, we propose SPHERE (Sparse Projection for Hyperspherical Energy-Regularized Editing), an HE-driven regularization strategy that stabilizes neuron weight distributions, ultimately preserving prior knowledge while enabling reliable sequential updates. Specifically, SPHERE identifies a sparse space complementary to the principal hyperspherical directions of the pretrained weight matrices and projects new knowledge onto it, attenuating perturbations on the principal directions. Extensive experiments on LLaMA3 (8B) and Qwen2.5 (7B) show that SPHERE outperforms the best baseline in editing capability by an average of 16.41%, while most faithfully preserving general model performance, thereby offering a principled path toward reliable large-scale knowledge editing. 

---
# Making, not Taking, the Best of N 

**Authors**: Ammar Khairi, Daniel D'souza, Marzieh Fadaee, Julia Kreutzer  

**Link**: [PDF](https://arxiv.org/pdf/2510.00931)  

**Abstract**: Obtaining high-quality generations in modern LLMs has largely been framed as a selection problem: identifying a single winning generation from a diverse pool of N samples, the Best-of-N (BoN). Yet, this approach is inherently zero-sum, discarding diverse and potentially useful information from the pool. Instead, we explore a collaborative setup, where all candidates can potentially contribute to the final winning generation. To this end, we propose Fusion-of-N (FusioN): a method that uses a general LLM judge to synthesize the most informative elements of each sample into a single final answer. We compare FusioN to BoN in two settings, (i) test-time scaling, where we sample and aggregate from a single model at test-time (ii) synthetic data generation, where we fuse samples from a pool of diverse teachers to improve a student model. We extensively benchmark both setups across 11 languages, 3 diverse tasks and varying model scales. Across the bench, FusioN consistently outperforms BoN showing versatility and robustness both in test-time scaling and in downstream gains from synthetic data generation. We also perform extensive analysis on FusioN, where it shows surprising strengths and robustness under challenging settings. These results show that we should shift how we think about evaluating and utilizing LLM generations from a monolithic measure of quality, to embracing their polylithic nature. This shift allows us to integrate diverse strengths, unlock latent potential, and achieve improvements that were previously inaccessible through selection alone. 

---
# ManagerBench: Evaluating the Safety-Pragmatism Trade-off in Autonomous LLMs 

**Authors**: Adi Simhi, Jonathan Herzig, Martin Tutek, Itay Itzhak, Idan Szpektor, Yonatan Belinkov  

**Link**: [PDF](https://arxiv.org/pdf/2510.00857)  

**Abstract**: As large language models (LLMs) evolve from conversational assistants into autonomous agents, evaluating the safety of their actions becomes critical. Prior safety benchmarks have primarily focused on preventing generation of harmful content, such as toxic text. However, they overlook the challenge of agents taking harmful actions when the most effective path to an operational goal conflicts with human safety. To address this gap, we introduce ManagerBench, a benchmark that evaluates LLM decision-making in realistic, human-validated managerial scenarios. Each scenario forces a choice between a pragmatic but harmful action that achieves an operational goal, and a safe action that leads to worse operational performance. A parallel control set, where potential harm is directed only at inanimate objects, measures a model's pragmatism and identifies its tendency to be overly safe. Our findings indicate that the frontier LLMs perform poorly when navigating this safety-pragmatism trade-off. Many consistently choose harmful options to advance their operational goals, while others avoid harm only to become overly safe and ineffective. Critically, we find this misalignment does not stem from an inability to perceive harm, as models' harm assessments align with human judgments, but from flawed prioritization. ManagerBench is a challenging benchmark for a core component of agentic behavior: making safe choices when operational goals and alignment values incentivize conflicting actions. Benchmark & code available at this https URL. 

---
# Pay-Per-Search Models are Abstention Models 

**Authors**: Mustafa Omer Gul, Claire Cardie, Tanya Goyal  

**Link**: [PDF](https://arxiv.org/pdf/2510.01152)  

**Abstract**: LLMs cannot reliably recognize their parametric knowledge boundaries and often hallucinate answers to outside-of-boundary questions. In contrast, humans recognize their limitations and can either seek external help for such questions or abstain. In this paper, we introduce MASH (Modeling Abstention via Selective Help-seeking), a training framework that readily extracts abstentions from LLMs. Our key idea is that any external help-seeking by an LLM, i.e. search tool use, can serve as a proxy for abstention if the external help (search) is appropriately penalized while simultaneously rewarding answer accuracy. MASH operationalizes this idea using reinforcement learning with a pay-per-search reward.
We run experiments on three knowledge-intensive QA datasets. Our results show that MASH substantially improves upon the selective help-seeking performance of prior efficient search approaches; on multi-hop datasets, MASH improves answer accuracy by 7.6%. Furthermore, MASH demonstrates strong off-the-shelf abstention -- it can distinguish between unanswerable/answerable questions and selectively generate responses for answerable questions -- showcasing behavior analogous to specialized abstention approaches. We emphasize that contrary to prior abstention methods, MASH does not require pre-determining knowledge boundaries to construct training data. Instead, MASH's abstentions are a by-product of training for the auxiliary selective help-seeking task. Overall, we show that MASH training effectively aligns search tool use with parametric knowledge, which can be successfully leveraged for making abstention decisions. 

---
# HalluGuard: Evidence-Grounded Small Reasoning Models to Mitigate Hallucinations in Retrieval-Augmented Generation 

**Authors**: Loris Bergeron, Ioana Buhnila, Jérôme François, Radu State  

**Link**: [PDF](https://arxiv.org/pdf/2510.00880)  

**Abstract**: Large Language Models (LLMs) excel in many NLP tasks but remain prone to hallucinations, limiting trust in real-world applications. We present HalluGuard, a 4B-parameter Small Reasoning Model (SRM) for mitigating hallucinations in Retrieval-Augmented Generation (RAG). HalluGuard classifies document-claim pairs as grounded or hallucinated and produces evidence-grounded justifications for transparency. Our approach combines (i) a domain-agnostic synthetic dataset derived from FineWeb and refined through multi-stage curation and data reformation, (ii) synthetic grounded and hallucinated claims, and (iii) preference-based fine-tuning with Odds Ratio Preference Optimization to distill large-model reasoning into a smaller backbone. On the RAGTruth subset of the LLM-AggreFact benchmark, HalluGuard achieves 84.0% balanced accuracy (BAcc), rivaling specialized models, MiniCheck (7B; 84.0%) and Granite Guardian 3.3 (8B; 82.2%) while using roughly half their parameters. Over the full benchmark it reaches 75.7% BAcc, matching larger general-purpose LLMs such as GPT-4o (75.9%). We will release HalluGuard and datasets under Apache 2.0 upon acceptance. 

---
# CoT Vectors: Transferring and Probing the Reasoning Mechanisms of LLMs 

**Authors**: Li Li, Ziyi Wang, Yongliang Wu, Jianfei Cai, Xu Yang  

**Link**: [PDF](https://arxiv.org/pdf/2510.00579)  

**Abstract**: Chain-of-Thought (CoT) prompting has emerged as a powerful approach to enhancing the reasoning capabilities of Large Language Models (LLMs). However, existing implementations, such as in-context learning and fine-tuning, remain costly and inefficient. To improve CoT reasoning at a lower cost, and inspired by the task vector paradigm, we introduce CoT Vectors, compact representations that encode task-general, multi-step reasoning knowledge. Through experiments with Extracted CoT Vectors, we observe pronounced layer-wise instability, manifesting as a U-shaped performance curve that reflects a systematic three-stage reasoning process in LLMs. To address this limitation, we propose Learnable CoT Vectors, optimized under a teacher-student framework to provide more stable and robust guidance. Extensive evaluations across diverse benchmarks and models demonstrate that CoT Vectors not only outperform existing baselines but also achieve performance comparable to parameter-efficient fine-tuning methods, while requiring fewer trainable parameters. Moreover, by treating CoT Vectors as a probe, we uncover how their effectiveness varies due to latent space structure, information density, acquisition mechanisms, and pre-training differences, offering new insights into the functional organization of multi-step reasoning in LLMs. The source code will be released. 

---
# ReSeek: A Self-Correcting Framework for Search Agents with Instructive Rewards 

**Authors**: Shiyu Li, Yang Tang, Yifan Wang, Peiming Li, Xi Chen  

**Link**: [PDF](https://arxiv.org/pdf/2510.00568)  

**Abstract**: Search agents powered by Large Language Models (LLMs) have demonstrated significant potential in tackling knowledge-intensive tasks. Reinforcement learning (RL) has emerged as a powerful paradigm for training these agents to perform complex, multi-step reasoning. However, prior RL-based methods often rely on sparse or rule-based rewards, which can lead agents to commit to suboptimal or erroneous reasoning paths without the ability to recover. To address these limitations, we propose ReSeek, a novel self-correcting framework for training search agents. Our framework introduces a self-correction mechanism that empowers the agent to dynamically identify and recover from erroneous search paths during an episode. By invoking a special JUDGE action, the agent can judge the information and re-plan its search strategy. To guide this process, we design a dense, instructive process reward function, which decomposes into a correctness reward for retrieving factual information and a utility reward for finding information genuinely useful for the query. Furthermore, to mitigate the risk of data contamination in existing datasets, we introduce FictionalHot, a new and challenging benchmark with recently curated questions requiring complex reasoning. Being intuitively reasonable and practically simple, extensive experiments show that agents trained with ReSeek significantly outperform SOTA baselines in task success rate and path faithfulness. 

---
# Family Matters: Language Transfer and Merging for Adapting Small LLMs to Faroese 

**Authors**: Jenny Kunz, Iben Nyholm Debess, Annika Simonsen  

**Link**: [PDF](https://arxiv.org/pdf/2510.00810)  

**Abstract**: We investigate how to adapt small, efficient LLMs to Faroese, a low-resource North Germanic language. Starting from English models, we continue pre-training on related Scandinavian languages, either individually or combined via merging, before fine-tuning on Faroese. We compare full fine-tuning with parameter-efficient tuning using LoRA, evaluating their impact on both linguistic accuracy and text comprehension. Due to the lack of existing Faroese evaluation data, we construct two new minimal-pair benchmarks from adapted and newly collected datasets and complement them with human evaluations by Faroese linguists. Our results demonstrate that transfer from related languages is crucial, though the optimal source language depends on the task: Icelandic enhances linguistic accuracy, whereas Danish boosts comprehension. Similarly, the choice between full fine-tuning and LoRA is task-dependent: LoRA improves linguistic acceptability and slightly increases human evaluation scores on the base model, while full fine-tuning yields stronger comprehension performance and better preserves model capabilities during downstream fine-tuning. 

---
# Are Large Language Models Chronically Online Surfers? A Dataset for Chinese Internet Meme Explanation 

**Authors**: Yubo Xie, Chenkai Wang, Zongyang Ma, Fahui Miao  

**Link**: [PDF](https://arxiv.org/pdf/2510.00567)  

**Abstract**: Large language models (LLMs) are trained on vast amounts of text from the Internet, but do they truly understand the viral content that rapidly spreads online -- commonly known as memes? In this paper, we introduce CHIME, a dataset for CHinese Internet Meme Explanation. The dataset comprises popular phrase-based memes from the Chinese Internet, annotated with detailed information on their meaning, origin, example sentences, types, etc. To evaluate whether LLMs understand these memes, we designed two tasks. In the first task, we assessed the models' ability to explain a given meme, identify its origin, and generate appropriate example sentences. The results show that while LLMs can explain the meanings of some memes, their performance declines significantly for culturally and linguistically nuanced meme types. Additionally, they consistently struggle to provide accurate origins for the memes. In the second task, we created a set of multiple-choice questions (MCQs) requiring LLMs to select the most appropriate meme to fill in a blank within a contextual sentence. While the evaluated models were able to provide correct answers, their performance remains noticeably below human levels. We have made CHIME public and hope it will facilitate future research on computational meme understanding. 

---
# Exposing the Cracks: Vulnerabilities of Retrieval-Augmented LLM-based Machine Translation 

**Authors**: Yanming Sun, Runzhe Zhan, Chi Seng Cheang, Han Wu, Xuebo Liu, Yuyao Niu, Fengying Ye, Kaixin Lan, Lidia S. Chao, Derek F. Wong  

**Link**: [PDF](https://arxiv.org/pdf/2510.00829)  

**Abstract**: \textbf{RE}trieval-\textbf{A}ugmented \textbf{L}LM-based \textbf{M}achine \textbf{T}ranslation (REAL-MT) shows promise for knowledge-intensive tasks like idiomatic translation, but its reliability under noisy retrieval contexts remains poorly understood despite this being a common challenge in real-world deployment. To address this gap, we propose a noise synthesis framework and new metrics to evaluate the robustness of REAL-MT systematically. Using this framework, we instantiate REAL-MT with Qwen-series models, including standard LLMs and large reasoning models (LRMs) with enhanced reasoning, and evaluate their performance on idiomatic translation across high-, medium-, and low-resource language pairs under synthesized noise. Our results show that low-resource language pairs, which rely more heavily on retrieved context, degrade more severely under noise than high-resource ones and often produce nonsensical translations. Although LRMs possess enhanced reasoning capabilities, they show no improvement in error correction and are even more susceptible to noise, tending to rationalize incorrect contexts. We find that this stems from an attention shift away from the source idiom to noisy content, while confidence increases despite declining accuracy, indicating poor calibration. To mitigate these issues, we investigate training-free and fine-tuning strategies, which improve robustness at the cost of performance in clean contexts, revealing a fundamental trade-off. Our findings highlight the limitations of current approaches, underscoring the need for self-verifying integration mechanisms. 

---
# TokMem: Tokenized Procedural Memory for Large Language Models 

**Authors**: Zijun Wu, Yongchang Hao, Lili Mou  

**Link**: [PDF](https://arxiv.org/pdf/2510.00444)  

**Abstract**: Large language models rely heavily on prompts to specify tasks, recall knowledge and guide reasoning. However, this reliance is inefficient as prompts must be re-read at each step, scale poorly across tasks, and lack mechanisms for modular reuse. We introduce TokMem, a tokenized procedural memory that stores recurring procedures as compact, trainable embeddings. Each memory token encodes both an address to a procedure and a control signal that steers generation, enabling targeted behavior with constant-size overhead. To support continual adaptation, TokMem keeps the backbone model frozen, allowing new procedures to be added without interfering with existing ones. We evaluate TokMem on 1,000 tasks for atomic recall, and on function-calling tasks for compositional recall, where it consistently outperforms retrieval-augmented generation while avoiding repeated context overhead, and fine-tuning with far fewer parameters. These results establish TokMem as a scalable and modular alternative to prompt engineering and fine-tuning, offering an explicit procedural memory for LLMs. 

---
# Beyond Log Likelihood: Probability-Based Objectives for Supervised Fine-Tuning across the Model Capability Continuum 

**Authors**: Gaotang Li, Ruizhong Qiu, Xiusi Chen, Heng Ji, Hanghang Tong  

**Link**: [PDF](https://arxiv.org/pdf/2510.00526)  

**Abstract**: Supervised fine-tuning (SFT) is the standard approach for post-training large language models (LLMs), yet it often shows limited generalization. We trace this limitation to its default training objective: negative log likelihood (NLL). While NLL is classically optimal when training from scratch, post-training operates in a different paradigm and could violate its optimality assumptions, where models already encode task-relevant priors and supervision can be long and noisy. To this end, we study a general family of probability-based objectives and characterize their effectiveness under different conditions. Through comprehensive experiments and extensive ablation studies across 7 model backbones, 14 benchmarks, and 3 domains, we uncover a critical dimension that governs objective behavior: the model-capability continuum. Near the model-strong end, prior-leaning objectives that downweight low-probability tokens (e.g., $-p$, $-p^{10}$, thresholded variants) consistently outperform NLL; toward the model-weak end, NLL dominates; in between, no single objective prevails. Our theoretical analysis further elucidates how objectives trade places across the continuum, providing a principled foundation for adapting objectives to model capability. Our code is available at this https URL. 

---
# Agent Fine-tuning through Distillation for Domain-specific LLMs in Microdomains 

**Authors**: Yawen Xue, Masaya Tsunokake, Yuta Koreeda, Ekant Muljibhai Amin, Takashi Sumiyoshi, Yasuhiro Sogawa  

**Link**: [PDF](https://arxiv.org/pdf/2510.00482)  

**Abstract**: Agentic large language models (LLMs) have become prominent for autonomously interacting with external environments and performing multi-step reasoning tasks. Most approaches leverage these capabilities via in-context learning with few-shot prompts, but this often results in lengthy inputs and higher computational costs. Agent fine-tuning offers an alternative by enabling LLMs to internalize procedural reasoning and domain-specific knowledge through training on relevant data and demonstration trajectories. While prior studies have focused on general domains, their effectiveness in specialized technical microdomains remains unclear. This paper explores agent fine-tuning for domain adaptation within Hitachi's JP1 middleware, a microdomain for specialized IT operations. We fine-tuned LLMs using JP1-specific datasets derived from domain manuals and distilled reasoning trajectories generated by LLMs themselves, enhancing decision making accuracy and search efficiency. During inference, we used an agentic prompt with retrieval-augmented generation and introduced a context-answer extractor to improve information relevance. On JP1 certification exam questions, our method achieved a 14% performance improvement over the base model, demonstrating the potential of agent fine-tuning for domain-specific reasoning in complex microdomains. 

---
# Enhancing Rating Prediction with Off-the-Shelf LLMs Using In-Context User Reviews 

**Authors**: Koki Ryu, Hitomi Yanaka  

**Link**: [PDF](https://arxiv.org/pdf/2510.00449)  

**Abstract**: Personalizing the outputs of large language models (LLMs) to align with individual user preferences is an active research area. However, previous studies have mainly focused on classification or ranking tasks and have not considered Likert-scale rating prediction, a regression task that requires both language and mathematical reasoning to be solved effectively. This task has significant industrial applications, but the utilization of LLMs remains underexplored, particularly regarding the capabilities of off-the-shelf LLMs. This study investigates the performance of off-the-shelf LLMs on rating prediction, providing different in-context information. Through comprehensive experiments with eight models across three datasets, we demonstrate that user-written reviews significantly improve the rating prediction performance of LLMs. This result is comparable to traditional methods like matrix factorization, highlighting the potential of LLMs as a promising solution for the cold-start problem. We also find that the reviews for concrete items are more effective than general preference descriptions that are not based on any specific item. Furthermore, we discover that prompting LLMs to first generate a hypothetical review enhances the rating prediction performance. Our code is available at this https URL. 

---
# LongCodeZip: Compress Long Context for Code Language Models 

**Authors**: Yuling Shi, Yichun Qian, Hongyu Zhang, Beijun Shen, Xiaodong Gu  

**Link**: [PDF](https://arxiv.org/pdf/2510.00446)  

**Abstract**: Code generation under long contexts is becoming increasingly critical as Large Language Models (LLMs) are required to reason over extensive information in the codebase. While recent advances enable code LLMs to process long inputs, high API costs and generation latency remain substantial bottlenecks. Existing context pruning techniques, such as LLMLingua, achieve promising results for general text but overlook code-specific structures and dependencies, leading to suboptimal performance in programming tasks. In this paper, we propose LongCodeZip, a novel plug-and-play code compression framework designed specifically for code LLMs. LongCodeZip employs a dual-stage strategy: (1) coarse-grained compression, which identifies and ranks function-level chunks using conditional perplexity with respect to the instruction, retaining only the most relevant functions; and (2) fine-grained compression, which segments retained functions into blocks based on perplexity and selects an optimal subset under an adaptive token budget to maximize relevance. Evaluations across multiple tasks, including code completion, summarization, and question answering, show that LongCodeZip consistently outperforms baseline methods, achieving up to a 5.6x compression ratio without degrading task performance. By effectively reducing context size while preserving essential information, LongCodeZip enables LLMs to better scale to real-world, large-scale code scenarios, advancing the efficiency and capability of code intelligence applications. 

---
# CORTEX: Collaborative LLM Agents for High-Stakes Alert Triage 

**Authors**: Bowen Wei, Yuan Shen Tay, Howard Liu, Jinhao Pan, Kun Luo, Ziwei Zhu, Chris Jordan  

**Link**: [PDF](https://arxiv.org/pdf/2510.00311)  

**Abstract**: Security Operations Centers (SOCs) are overwhelmed by tens of thousands of daily alerts, with only a small fraction corresponding to genuine attacks. This overload creates alert fatigue, leading to overlooked threats and analyst burnout. Classical detection pipelines are brittle and context-poor, while recent LLM-based approaches typically rely on a single model to interpret logs, retrieve context, and adjudicate alerts end-to-end -- an approach that struggles with noisy enterprise data and offers limited transparency. We propose CORTEX, a multi-agent LLM architecture for high-stakes alert triage in which specialized agents collaborate over real evidence: a behavior-analysis agent inspects activity sequences, evidence-gathering agents query external systems, and a reasoning agent synthesizes findings into an auditable decision. To support training and evaluation, we release a dataset of fine-grained SOC investigations from production environments, capturing step-by-step analyst actions and linked tool outputs. Across diverse enterprise scenarios, CORTEX substantially reduces false positives and improves investigation quality over state-of-the-art single-agent LLMs. 

---
# Judging with Confidence: Calibrating Autoraters to Preference Distributions 

**Authors**: Zhuohang Li, Xiaowei Li, Chengyu Huang, Guowang Li, Katayoon Goshvadi, Bo Dai, Dale Schuurmans, Paul Zhou, Hamid Palangi, Yiwen Song, Palash Goyal, Murat Kantarcioglu, Bradley A. Malin, Yuan Xue  

**Link**: [PDF](https://arxiv.org/pdf/2510.00263)  

**Abstract**: The alignment of large language models (LLMs) with human values increasingly relies on using other LLMs as automated judges, or ``autoraters''. However, their reliability is limited by a foundational issue: they are trained on discrete preference labels, forcing a single ground truth onto tasks that are often subjective, ambiguous, or nuanced. We argue that a reliable autorater must learn to model the full distribution of preferences defined by a target population. In this paper, we propose a general framework for calibrating probabilistic autoraters to any given preference distribution. We formalize the problem and present two learning methods tailored to different data conditions: 1) a direct supervised fine-tuning for dense, probabilistic labels, and 2) a reinforcement learning approach for sparse, binary labels. Our empirical results show that finetuning autoraters with a distribution-matching objective leads to verbalized probability predictions that are better aligned with the target preference distribution, with improved calibration and significantly lower positional bias, all while preserving performance on objective tasks. 

---
# Efficient Layer-wise LLM Fine-tuning for Revision Intention Prediction 

**Authors**: Zhexiong Liu, Diane Litman  

**Link**: [PDF](https://arxiv.org/pdf/2510.00268)  

**Abstract**: Large Language Models (LLMs) have shown extraordinary success across various text generation tasks; however, their potential for simple yet essential text classification remains underexplored, as LLM pre-training tends to emphasize generation over classification. While LLMs with instruction tuning can transform classification into a generation task, they often struggle to categorize nuanced texts. One such example is text revision, which involves nuanced edits between pairs of texts. Although simply fine-tuning LLMs for revision classification seems plausible, it requires a large amount of revision annotations, which are exceptionally expensive and scarce in the community. To address this issue, we introduce a plug-and-play layer-wise parameter-efficient fine-tuning (PEFT) framework, i.e., IR-Tuning, which fine-tunes a subset of important LLM layers that are dynamically selected based on their gradient norm distribution, while freezing those of redundant layers. Extensive experiments suggest that IR-Tuning surpasses several layer-wise PEFT baselines over diverse text revisions, while achieving fast convergence, low GPU memory consumption, and effectiveness on small revision corpora. 

---
# SafePassage: High-Fidelity Information Extraction with Black Box LLMs 

**Authors**: Joe Barrow, Raj Patel, Misha Kharkovski, Ben Davies, Ryan Schmitt  

**Link**: [PDF](https://arxiv.org/pdf/2510.00276)  

**Abstract**: Black box large language models (LLMs) make information extraction (IE) easy to configure, but hard to trust. Unlike traditional information extraction pipelines, the information "extracted" is not guaranteed to be grounded in the document. To prevent this, this paper introduces the notion of a "safe passage": context generated by the LLM that is both grounded in the document and consistent with the extracted information. This is operationalized via a three-step pipeline, SafePassage, which consists of: (1) an LLM extractor that generates structured entities and their contexts from a document, (2) a string-based global aligner, and (3) a scoring model. Results show that using these three parts in conjunction reduces hallucinations by up to 85% on information extraction tasks with minimal risk of flagging non-hallucinations. High agreement between the SafePassage pipeline and human judgments of extraction quality mean that the pipeline can be dually used to evaluate LLMs. Surprisingly, results also show that using a transformer encoder fine-tuned on a small number of task-specific examples can outperform an LLM scoring model at flagging unsafe passages. These annotations can be collected in as little as 1-2 hours. 

---
# Direct Token Optimization: A Self-contained Approach to Large Language Model Unlearning 

**Authors**: Hong kyu Lee, Ruixuan Liu, Li Xiong  

**Link**: [PDF](https://arxiv.org/pdf/2510.00125)  

**Abstract**: Machine unlearning is an emerging technique that removes the influence of a subset of training data (forget set) from a model without full retraining, with applications including privacy protection, content moderation, and model correction. The key challenge lies in ensuring that the model completely forgets the knowledge of the forget set without compromising its overall utility. Existing unlearning methods for large language models (LLMs) often utilize auxiliary language models, retain datasets, or even commercial AI services for effective unlearning and maintaining the model utility. However, dependence on these external resources is often impractical and could potentially introduce additional privacy risks. In this work, we propose direct token optimization (DTO), a novel self-contained unlearning approach for LLMs that directly optimizes the token level objectives and eliminates the need for external resources. Given a sequence to unlearn, we identify two categories of tokens: target tokens, which capture critical knowledge for unlearning, and the remaining non-target tokens, which are crucial for maintaining the model utility. The former are used to optimize the unlearning objective, while the latter serve to preserve the model's performance. The experimental results show that the proposed DTO achieves up to 16.8$\times$ improvement in forget quality on several benchmark datasets than the latest baselines while maintaining a comparable level of model utility. 

---
# Prompt Curriculum Learning for Efficient LLM Post-Training 

**Authors**: Zhaolin Gao, Joongwon Kim, Wen Sun, Thorsten Joachims, Sid Wang, Richard Yuanzhe Pang, Liang Tan  

**Link**: [PDF](https://arxiv.org/pdf/2510.01135)  

**Abstract**: We introduce Prompt Curriculum Learning (PCL), a lightweight reinforcement learning (RL) algorithm that selects intermediate-difficulty prompts using a learned value model to post-train language models. Since post-training LLMs via RL remains sensitive to batching and prompt selection strategies, we first conduct a series of systematic experiments where we (1) determine the optimal training batch size that balances generation efficiency and gradient quality and (2) establish the importance of focusing on prompts of intermediate difficulty for the policy. We build upon these results to design PCL, which identifies prompts of intermediate difficulty for the current policy in an on-policy manner by using a value model that is concurrently updated based on the current policy. By focusing on informative prompts that yield high effective ratios, PCL achieves either the highest performance or requires significantly less time to reach comparable performance to its counterparts. Compared to rollout-based filtering methods, PCL avoids costly rollouts and achieves $12.1\times$ and $16.9\times$ faster speed on identifying intermediate-difficulty prompts when training on MATH and DeepScaleR, respectively. We further demonstrate that our value model accurately predicts prompt difficulty and allows PCL to focus on progressively more challenging prompts during RL. Our results present a new methodology that delivers improved tradeoff between upper-bound performance and efficiency for reasoning-focused RL. 

---
# HLTCOE at TREC 2024 NeuCLIR Track 

**Authors**: Eugene Yang, Dawn Lawrie, Orion Weller, James Mayfield  

**Link**: [PDF](https://arxiv.org/pdf/2510.00143)  

**Abstract**: The HLTCOE team applied PLAID, an mT5 reranker, GPT-4 reranker, score fusion, and document translation to the TREC 2024 NeuCLIR track. For PLAID we included a variety of models and training techniques -- Translate Distill (TD), Generate Distill (GD) and multi-lingual translate-distill (MTD). TD uses scores from the mT5 model over English MS MARCO query-document pairs to learn how to score query-document pairs where the documents are translated to match the CLIR setting. GD follows TD but uses passages from the collection and queries generated by an LLM for training examples. MTD uses MS MARCO translated into multiple languages, allowing experiments on how to batch the data during training. Finally, for report generation we experimented with system combination over different runs. One family of systems used either GPT-4o or Claude-3.5-Sonnet to summarize the retrieved results from a series of decomposed sub-questions. Another system took the output from those two models and verified/combined them with Claude-3.5-Sonnet. The other family used GPT4o and GPT3.5Turbo to extract and group relevant facts from the retrieved documents based on the decomposed queries. The resulting submissions directly concatenate the grouped facts to form the report and their documents of origin as the citations. The team submitted runs to all NeuCLIR tasks: CLIR and MLIR news tasks as well as the technical documents task and the report generation task. 

---
# Which Programming Language and Model Work Best With LLM-as-a-Judge For Code Retrieval? 

**Authors**: Lucas Roberts, Denisa Roberts  

**Link**: [PDF](https://arxiv.org/pdf/2510.00324)  

**Abstract**: Code search is an important information retrieval application. Benefits of better code search include faster new developer on-boarding, reduced software maintenance, and ease of understanding for large repositories. Despite improvements in search algorithms and search benchmarks, the domain of code search has lagged behind. One reason is the high cost of human annotation for code queries and answers. While humans may annotate search results in general text QA systems, code annotations require specialized knowledge of a programming language (PL), as well as domain specific software engineering knowledge. In this work we study the use of Large Language Models (LLMs) to retrieve code at the level of functions and to generate annotations for code search results. We compare the impact of the retriever representation (sparse vs. semantic), programming language, and LLM by comparing human annotations across several popular languages (C, Java, Javascript, Go, and Python). We focus on repositories that implement common data structures likely to be implemented in any PLs. For the same human annotations, we compare several LLM-as-a-Judge models to evaluate programming language and other affinities between LLMs. We find that the chosen retriever and PL exhibit affinities that can be leveraged to improve alignment of human and AI relevance determinations, with significant performance implications. We also find differences in representation (sparse vs. semantic) across PLs that impact alignment of human and AI relevance determinations. We propose using transpilers to bootstrap scalable code search benchmark datasets in other PLs and in a case study demonstrate that human-AI relevance agreement rates largely match the (worst case) human-human agreement under study. The application code used in this work is available at \href{this https URL}{this github repo}. 

---
# On Listwise Reranking for Corpus Feedback 

**Authors**: Soyoung Yoon, Jongho Kim, Daeyong Kwon, Avishek Anand, Seung-won Hwang  

**Link**: [PDF](https://arxiv.org/pdf/2510.00887)  

**Abstract**: Reranker improves retrieval performance by capturing document interactions. At one extreme, graph-aware adaptive retrieval (GAR) represents an information-rich regime, requiring a pre-computed document similarity graph in reranking. However, as such graphs are often unavailable, or incur quadratic memory costs even when available, graph-free rerankers leverage large language model (LLM) calls to achieve competitive performance. We introduce L2G, a novel framework that implicitly induces document graphs from listwise reranker logs. By converting reranker signals into a graph structure, L2G enables scalable graph-based retrieval without the overhead of explicit graph computation. Results on the TREC-DL and BEIR subset show that L2G matches the effectiveness of oracle-based graph methods, while incurring zero additional LLM calls. 

---
