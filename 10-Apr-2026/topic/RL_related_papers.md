# SUPERNOVA: Eliciting General Reasoning in LLMs with Reinforcement Learning on Natural Instructions 

**Authors**: Ashima Suvarna, Kendrick Phan, Mehrab Beikzadeh, Hritik Bansal, Saadia Gabriel  

**Link**: [PDF](https://arxiv.org/pdf/2604.08477)  

**Abstract**: Reinforcement Learning with Verifiable Rewards (RLVR) has significantly improved large language model (LLM) reasoning in formal domains such as mathematics and code. Despite these advancements, LLMs still struggle with general reasoning tasks requiring capabilities such as causal inference and temporal understanding. Extending RLVR to general reasoning is fundamentally constrained by the lack of high-quality, verifiable training data that spans diverse reasoning skills. To address this challenge, we propose SUPERNOVA, a data curation framework for RLVR aimed at enhancing general reasoning. Our key insight is that instruction-tuning datasets containing expert-annotated ground-truth encode rich reasoning patterns that can be systematically adapted for RLVR. To study this, we conduct 100+ controlled RL experiments to analyze how data design choices impact downstream reasoning performance. In particular, we investigate three key factors: (i) source task selection, (ii) task mixing strategies, and (iii) synthetic interventions for improving data quality. Our analysis reveals that source task selection is non-trivial and has a significant impact on downstream reasoning performance. Moreover, selecting tasks based on their performance for individual target tasks outperforms strategies based on overall average performance. Finally, models trained on SUPERNOVA outperform strong baselines (e.g., Qwen3.5) on challenging reasoning benchmarks including BBEH, Zebralogic, and MMLU-Pro. In particular, training on SUPERNOVA yields relative improvements of up to 52.8\% on BBEH across model sizes, demonstrating the effectiveness of principled data curation for RLVR. Our findings provide practical insights for curating human-annotated resources to extend RLVR to general reasoning. The code and data is available at this https URL. 

---
# Ads in AI Chatbots? An Analysis of How Large Language Models Navigate Conflicts of Interest 

**Authors**: Addison J. Wu, Ryan Liu, Shuyue Stella Li, Yulia Tsvetkov, Thomas L. Griffiths  

**Link**: [PDF](https://arxiv.org/pdf/2604.08525)  

**Abstract**: Today's large language models (LLMs) are trained to align with user preferences through methods such as reinforcement learning. Yet models are beginning to be deployed not merely to satisfy users, but also to generate revenue for the companies that created them through advertisements. This creates the potential for LLMs to face conflicts of interest, where the most beneficial response to a user may not be aligned with the company's incentives. For instance, a sponsored product may be more expensive but otherwise equal to another; in this case, what does (and should) the LLM recommend to the user? In this paper, we provide a framework for categorizing the ways in which conflicting incentives might lead LLMs to change the way they interact with users, inspired by literature from linguistics and advertising regulation. We then present a suite of evaluations to examine how current models handle these tradeoffs. We find that a majority of LLMs forsake user welfare for company incentives in a multitude of conflict of interest situations, including recommending a sponsored product almost twice as expensive (Grok 4.1 Fast, 83%), surfacing sponsored options to disrupt the purchasing process (GPT 5.1, 94%), and concealing prices in unfavorable comparisons (Qwen 3 Next, 24%). Behaviors also vary strongly with levels of reasoning and users' inferred socio-economic status. Our results highlight some of the hidden risks to users that can emerge when companies begin to subtly incentivize advertisements in chatbots. 

---
# ProMedical: Hierarchical Fine-Grained Criteria Modeling for Medical LLM Alignment via Explicit Injection 

**Authors**: He Geng, Yangmin Huang, Lixian Lai, Qianyun Du, Hui Chu, Zhiyang He, Jiaxue Hu, Xiaodong Tao  

**Link**: [PDF](https://arxiv.org/pdf/2604.08326)  

**Abstract**: Aligning Large Language Models (LLMs) with high-stakes medical standards remains a significant challenge, primarily due to the dissonance between coarse-grained preference signals and the complex, multi-dimensional nature of clinical protocols. To bridge this gap, we introduce ProMedical, a unified alignment framework grounded in fine-grained clinical criteria. We first construct ProMedical-Preference-50k, a dataset generated via a human-in-the-loop pipeline that augments medical instructions with rigorous, physician-derived rubrics. Leveraging this corpus, we propose the Explicit Criteria Injection paradigm to train a multi-dimensional reward model. Unlike traditional scalar reward models, our approach explicitly disentangles safety constraints from general proficiency, enabling precise guidance during reinforcement learning. To rigorously validate this framework, we establish ProMedical-Bench, a held-out evaluation suite anchored by double-blind expert adjudication. Empirical evaluations demonstrate that optimizing the Qwen3-8B base model via ProMedical-RM-guided GRPO yields substantial gains, improving overall accuracy by 22.3% and safety compliance by 21.7%, effectively rivaling proprietary frontier models. Furthermore, the aligned policy generalizes robustly to external benchmarks, demonstrating performance comparable to state-of-the-art models on UltraMedical. We publicly release our datasets, reward models, and benchmarks to facilitate reproducible research in safety-aware medical alignment. 

---
# Beyond Stochastic Exploration: What Makes Training Data Valuable for Agentic Search 

**Authors**: Chuzhan Hao, Wenfeng Feng, Guochao Jiang, Guofeng Quan, Guohua Liu, Yuewei Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2604.08124)  

**Abstract**: Reinforcement learning (RL) has become an effective approach for advancing the reasoning capabilities of large language models (LLMs) through the strategic integration of external search engines. However, current RL-based search agents often rely on a process of stochastic exploration guided by carefully crafted outcome rewards, leading to inefficient reasoning trajectories and unstable training. To address these issues, we propose a novel framework, Hierarchical Experience (HiExp), to enhance the performance and training stability of search agents. Specifically, we extract empirical knowledge through contrastive analysis and a multi-level clustering mechanism, transforming raw reasoning trajectories into hierarchical experience knowledge. By leveraging experience-aligned training, we effectively regularize stochastic exploration, evolving it into a strategic and experience-driven search process. Extensive evaluations on multiple complex agentic search and mathematical reasoning benchmarks demonstrate that our approach not only achieves substantial performance gains but also exhibits strong cross-task and cross-algorithm generalization. 

---
# Aligning Agents via Planning: A Benchmark for Trajectory-Level Reward Modeling 

**Authors**: Jiaxuan Wang, Yulan Hu, Wenjin Yang, Zheng Pan, Xin Li, Lan-Zhe Guo  

**Link**: [PDF](https://arxiv.org/pdf/2604.08178)  

**Abstract**: In classical Reinforcement Learning from Human Feedback (RLHF), Reward Models (RMs) serve as the fundamental signal provider for model alignment. As Large Language Models evolve into agentic systems capable of autonomous tool invocation and complex reasoning, the paradigm of reward modeling faces unprecedented challenges--most notably, the lack of benchmarks specifically designed to assess RM capabilities within tool-integrated environments. To address this gap, we present Plan-RewardBench, a trajectory-level preference benchmark designed to evaluate how well judges distinguish preferred versus distractor agent trajectories in complex tool-using scenarios. Plan-RewardBench covers four representative task families -- (i) Safety Refusal, (ii) Tool-Irrelevance / Unavailability, (iii) Complex Planning, and (iv) Robust Error Recovery -- comprising validated positive trajectories and confusable hard negatives constructed via multi-model natural rollouts, rule-based perturbations, and minimal-edit LLM perturbations. We benchmark representative RMs (generative, discriminative, and LLM-as-Judge) under a unified pairwise protocol, reporting accuracy trends across varying trajectory lengths and task categories. Furthermore, we provide diagnostic analyses of prevalent failure modes. Our results reveal that all three evaluator families face substantial challenges, with performance degrading sharply on long-horizon trajectories, underscoring the necessity for specialized training in agentic, trajectory-level reward modeling. Ultimately, Plan-RewardBench aims to serve as both a practical evaluation suite and a reusable blueprint for constructing agentic planning preference data. 

---
# SPARD: Self-Paced Curriculum for RL Alignment via Integrating Reward Dynamics and Data Utility 

**Authors**: Xuyang Zhi, Peilun zhou, Chengqiang Lu, Hang Lv, Yiwei Liang, Rongyang Zhang, Yan Gao, YI WU, Yao Hu, Hongchao Gu, Defu Lian, Hao Wang, Enhong Chen  

**Link**: [PDF](https://arxiv.org/pdf/2604.07837)  

**Abstract**: The evolution of Large Language Models (LLMs) is shifting the focus from single, verifiable tasks toward complex, open-ended real-world scenarios, imposing significant challenges on the post-training phase. In these settings, the scale and complexity of reward systems have grown significantly, transitioning toward multi-objective formulations that encompass a comprehensive spectrum of model capabilities and application contexts. However, traditional methods typically rely on fixed reward weights, ignoring non-stationary learning dynamics and struggling with data heterogeneity across dimensions. To address these issues, we propose SPARD, a framework that establishes an automated, self-paced curriculum by perceiving learning progress to dynamically adjust multi-objective reward weights and data importance, thereby synchronizing learning intent with data utility for optimal performance. Extensive experiments across multiple benchmarks demonstrate that SPARD significantly enhances model capabilities across all domains. 

---
# SEARL: Joint Optimization of Policy and Tool Graph Memory for Self-Evolving Agents 

**Authors**: Xinshun Feng, Xinhao Song, Lijun Li, Gongshen Liu, Jing Shao  

**Link**: [PDF](https://arxiv.org/pdf/2604.07791)  

**Abstract**: Recent advances in Reinforcement Learning with Verifiable Rewards (RLVR) have demonstrated significant potential in single-turn reasoning tasks. With the paradigm shift toward self-evolving agentic learning, models are increasingly expected to learn from trajectories by synthesizing tools or accumulating explicit experiences. However, prevailing methods typically rely on large-scale LLMs or multi-agent frameworks, which hinder their deployment in resource-constrained environments. The inherent sparsity of outcome-based rewards also poses a substantial challenge, as agents typically receive feedback only upon completion of tasks. To address these limitations, we introduce a Tool-Memory based self-evolving agentic framework SEARL. Unlike approaches that directly utilize interaction experiences, our method constructs a structured experience memory that integrates planning with execution. This provides a novel state abstraction that facilitates generalization across analogous contexts, such as tool reuse. Consequently, agents extract explicit knowledge from historical data while leveraging inter-trajectory correlations to densify reward signals. We evaluate our framework on knowledge reasoning and mathematics tasks, demonstrating its effectiveness in achieving more practical and efficient learning. 

---
# SAT: Balancing Reasoning Accuracy and Efficiency with Stepwise Adaptive Thinking 

**Authors**: Weiyang Huang, Xuefeng Bai, Kehai Chen, Xinyang Chen, Yibin Chen, Weili Guan, Min Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2604.07922)  

**Abstract**: Large Reasoning Models (LRMs) have revolutionized complex problem-solving, yet they exhibit a pervasive "overthinking", generating unnecessarily long reasoning chains. While current solutions improve token efficiency, they often sacrifice fine-grained control or risk disrupting the logical integrity of the reasoning process. To address this, we introduce Stepwise Adaptive Thinking (SAT), a framework that performs step-level, difficulty-aware pruning while preserving the core reasoning structure. SAT formulates reasoning as a Finite-State Machine (FSM) with distinct thinking modes (Slow, Normal, Fast, Skip). It navigates these states dynamically using a lightweight Process Reward Model (PRM), compressing easy steps while preserving depth for hard ones. Experiments across 9 LRMs and 7 benchmarks show that SAT achieves up to 40% reduction in reasoning tokens while generally maintaining or improving accuracy. 

---
# Activation Steering for Aligned Open-ended Generation without Sacrificing Coherence 

**Authors**: Niklas Herbster, Martin Zborowski, Alberto Tosato, Gauthier Gidel, Tommaso Tosato  

**Link**: [PDF](https://arxiv.org/pdf/2604.08169)  

**Abstract**: Alignment in LLMs is more brittle than commonly assumed: misalignment can be triggered by adversarial prompts, benign fine-tuning, emergent misalignment, and goal misgeneralization. Recent evidence suggests that some misalignment behaviors are encoded as linear structure in activation space, making it tractable via steering, while safety alignment has been shown to govern the first few output tokens primarily, leaving subsequent generation unguarded. These findings motivate activation steering as a lightweight runtime defense that continuously corrects misaligned activations throughout generation. We evaluate three methods: Steer-With-Fixed-Coeff (SwFC), which applies uniform additive steering, and two novel projection-aware methods, Steer-to-Target-Projection (StTP) and Steer-to-Mirror-Projection (StMP), that use a logistic regression decision boundary to selectively intervene only on tokens whose activations fall below distributional thresholds. Using malicious system prompts as a controlled proxy for misalignment, we evaluate under two threat models (dishonesty and dismissiveness) and two architectures (Llama-3.3-70B-Instruct, Qwen3-32B). All methods substantially recover target traits (honesty and compassion) while preserving coherence. StTP and StMP better maintain general capabilities (MMLU, MT-Bench, AlpacaEval) and produce less repetition in multi-turn conversations. 

---
# Mitigating Distribution Sharpening in Math RLVR via Distribution-Aligned Hint Synthesis and Backward Hint Annealing 

**Authors**: Pei-Xi Xie, Che-Yu Lin, Cheng-Lin Yang  

**Link**: [PDF](https://arxiv.org/pdf/2604.07747)  

**Abstract**: Reinforcement learning with verifiable rewards (RLVR) can improve low-$k$ reasoning accuracy while narrowing solution coverage on challenging math questions, and pass@1 gains do not necessarily translate into better large-$k$ performance. Existing hint-based approaches can make challenging questions trainable, but they leave two issues underexplored: teacher-student distribution mismatch and the need to reduce hint exposure to match no-hint evaluation. We address these issues through two components. Distribution-Aligned Hint Synthesis (DAHS) constructs verified teacher hints conditioned on student-style responses. Backward Hint Annealing (BHA) anneals hint exposure across difficulty buckets and uses per-question hint dropout to preserve no-hint updates throughout RL training. We evaluate the method in math RLVR under the DAPO training framework across AIME24, AIME25, and AIME26 using $\texttt{Qwen3-1.7B-Base}$ and $\texttt{Llama-3.2-1B-Instruct}$. On $\texttt{Qwen3-1.7B-Base}$, our method improves both pass@1 and pass@2048 relative to DAPO across the three AIME benchmarks. On $\texttt{Llama-3.2-1B-Instruct}$, the gains are concentrated in the large-$k$ regime. These results suggest that, in math RLVR, hint scaffolding is effective when it restores learnable updates on challenging questions early in training and is then gradually removed before no-hint evaluation. 

---
# ReflectRM: Boosting Generative Reward Models via Self-Reflection within a Unified Judgment Framework 

**Authors**: Kai Qin, Liangxin Liu, Yu Liang, Longzheng Wang, Yan Wang, Yueyang Zhang, Long Xia, Zhiyuan Sun, Houde Liu, Daiting Shi  

**Link**: [PDF](https://arxiv.org/pdf/2604.07506)  

**Abstract**: Reward Models (RMs) are critical components in the Reinforcement Learning from Human Feedback (RLHF) pipeline, directly determining the alignment quality of Large Language Models (LLMs). Recently, Generative Reward Models (GRMs) have emerged as a superior paradigm, offering higher interpretability and stronger generalization than traditional scalar RMs. However, existing methods for GRMs focus primarily on outcome-level supervision, neglecting analytical process quality, which constrains their potential. To address this, we propose ReflectRM, a novel GRM that leverages self-reflection to assess analytical quality and enhance preference modeling. ReflectRM is trained under a unified generative framework for joint modeling of response preference and analysis preference. During inference, we use its self-reflection capability to identify the most reliable analysis, from which the final preference prediction is derived. Experiments across four benchmarks show that ReflectRM consistently improves performance, achieving an average accuracy gain of +3.7 on Qwen3-4B. Further experiments confirm that response preference and analysis preference are mutually reinforcing. Notably, ReflectRM substantially mitigates positional bias, yielding +10.2 improvement compared with leading GRMs and establishing itself as a more stable evaluator. 

---
# ConsistRM: Improving Generative Reward Models via Consistency-Aware Self-Training 

**Authors**: Yu Liang, Liangxin Liu, Longzheng Wang, Yan Wang, Yueyang Zhang, Long Xia, Zhiyuan Sun, Daiting Shi  

**Link**: [PDF](https://arxiv.org/pdf/2604.07484)  

**Abstract**: Generative reward models (GRMs) have emerged as a promising approach for aligning Large Language Models (LLMs) with human preferences by offering greater representational capacity and flexibility than traditional scalar reward models. However, GRMs face two major challenges: reliance on costly human-annotated data restricts scalability, and self-training approaches often suffer from instability and vulnerability to reward hacking. To address these issues, we propose ConsistRM, a self-training framework that enables effective and stable GRM training without human annotations. ConsistRM incorporates the Consistency-Aware Answer Reward, which produces reliable pseudo-labels with temporal consistency, thereby providing more stable model optimization. Moreover, the Consistency-Aware Critique Reward is introduced to assess semantic consistency across multiple critiques and allocates fine-grained and differentiated rewards. Experiments on five benchmark datasets across four base models demonstrate that ConsistRM outperforms vanilla Reinforcement Fine-Tuning (RFT) by an average of 1.5%. Further analysis shows that ConsistRM enhances output consistency and mitigates position bias caused by input order, highlighting the effectiveness of consistency-aware rewards in improving GRMs. 

---
# CLEAR: Context Augmentation from Contrastive Learning of Experience via Agentic Reflection 

**Authors**: Linbo Liu, Guande Wu, Han Ding, Yawei Wang, Qiang Zhou, Yuzhe Lu, Zhichao Xu, Huan Song, Panpan Xu, Lin Lee Cheong  

**Link**: [PDF](https://arxiv.org/pdf/2604.07487)  

**Abstract**: Large language model agents rely on effective model context to obtain task-relevant information for decision-making. Many existing context engineering approaches primarily rely on the context generated from the past experience and retrieval mechanisms that reuse these context. However, retrieved context from past tasks must be adapted by the execution agent to fit new situations, placing additional reasoning burden on the underlying LLM. To address this limitation, we propose a generative context augmentation framework using Contrastive Learning of Experience via Agentic Reflection (CLEAR). CLEAR first employs a reflection agent to perform contrastive analysis over past execution trajectories and summarize useful context for each observed task. These summaries are then used as supervised fine-tuning data to train a context augmentation model (CAM). Then we further optimize CAM using reinforcement learning, where the reward signal is obtained by running the task execution agent. By learning to generate task-specific knowledge rather than retrieve knowledge from the past, CAM produces context that is better tailored to the current task. We conduct comprehensive evaluations on the AppWorld and WebShop benchmarks. Experimental results show that CLEAR consistently outperforms strong baselines. It improves task completion rate from 72.62% to 81.15% on AppWorld test set and averaged reward from 0.68 to 0.74 on a subset of WebShop, compared with baseline agent. Our code is publicly available at this https URL. 

---
# Act Wisely: Cultivating Meta-Cognitive Tool Use in Agentic Multimodal Models 

**Authors**: Shilin Yan, Jintao Tong, Hongwei Xue, Xiaojun Tang, Yangyang Wang, Kunyu Shi, Guannan Zhang, Ruixuan Li, Yixiong Zou  

**Link**: [PDF](https://arxiv.org/pdf/2604.08545)  

**Abstract**: The advent of agentic multimodal models has empowered systems to actively interact with external environments. However, current agents suffer from a profound meta-cognitive deficit: they struggle to arbitrate between leveraging internal knowledge and querying external utilities. Consequently, they frequently fall prey to blind tool invocation, resorting to reflexive tool execution even when queries are resolvable from the raw visual context. This pathological behavior precipitates severe latency bottlenecks and injects extraneous noise that derails sound reasoning. Existing reinforcement learning protocols attempt to mitigate this via a scalarized reward that penalizes tool usage. Yet, this coupled formulation creates an irreconcilable optimization dilemma: an aggressive penalty suppresses essential tool use, whereas a mild penalty is entirely subsumed by the variance of the accuracy reward during advantage normalization, rendering it impotent against tool overuse. To transcend this bottleneck, we propose HDPO, a framework that reframes tool efficiency from a competing scalar objective to a strictly conditional one. By eschewing reward scalarization, HDPO maintains two orthogonal optimization channels: an accuracy channel that maximizes task correctness, and an efficiency channel that enforces execution economy exclusively within accurate trajectories via conditional advantage estimation. This decoupled architecture naturally induces a cognitive curriculum-compelling the agent to first master task resolution before refining its self-reliance. Extensive evaluations demonstrate that our resulting model, Metis, reduces tool invocations by orders of magnitude while simultaneously elevating reasoning accuracy. 

---
# OpenVLThinkerV2: A Generalist Multimodal Reasoning Model for Multi-domain Visual Tasks 

**Authors**: Wenbo Hu, Xin Chen, Yan Gao-Tian, Yihe Deng, Nanyun Peng, Kai-Wei Chang  

**Link**: [PDF](https://arxiv.org/pdf/2604.08539)  

**Abstract**: Group Relative Policy Optimization (GRPO) has emerged as the de facto Reinforcement Learning (RL) objective driving recent advancements in Multimodal Large Language Models. However, extending this success to open-source multimodal generalist models remains heavily constrained by two primary challenges: the extreme variance in reward topologies across diverse visual tasks, and the inherent difficulty of balancing fine-grained perception with multi-step reasoning capabilities. To address these issues, we introduce Gaussian GRPO (G$^2$RPO), a novel RL training objective that replaces standard linear scaling with non-linear distributional matching. By mathematically forcing the advantage distribution of any given task to strictly converge to a standard normal distribution, $\mathcal{N}(0,1)$, G$^2$RPO theoretically ensures inter-task gradient equity, mitigates vulnerabilities to heavy-tail outliers, and offers symmetric update for positive and negative rewards. Leveraging the enhanced training stability provided by G$^2$RPO, we introduce two task-level shaping mechanisms to seamlessly balance perception and reasoning. First, response length shaping dynamically elicits extended reasoning chains for complex queries while enforce direct outputs to bolster visual grounding. Second, entropy shaping tightly bounds the model's exploration zone, effectively preventing both entropy collapse and entropy explosion. Integrating these methodologies, we present OpenVLThinkerV2, a highly robust, general-purpose multimodal model. Extensive evaluations across 18 diverse benchmarks demonstrate its superior performance over strong open-source and leading proprietary frontier models. 

---
# TTVS: Boosting Self-Exploring Reinforcement Learning via Test-time Variational Synthesis 

**Authors**: Sikai Bai, Haoxi Li, Jie Zhang, Yongjiang Liu, Song Guo  

**Link**: [PDF](https://arxiv.org/pdf/2604.08468)  

**Abstract**: Despite significant advances in Large Reasoning Models (LRMs) driven by reinforcement learning with verifiable rewards (RLVR), this paradigm is fundamentally limited in specialized or novel domains where such supervision is prohibitively expensive or unavailable, posing a key challenge for test-time adaptation. While existing test-time methods offer a potential solution, they are constrained by learning from static query sets, risking overfitting to textual patterns. To address this gap, we introduce Test-Time Variational Synthesis (TTVS), a novel framework that enables LRMs to self-evolve by dynamically augmenting the training stream from unlabeled test queries. TTVS comprises two synergistic modules: (1) Online Variational Synthesis, which transforms static test queries into a dynamic stream of diverse, semantically-equivalent variations, enforcing the model to learn underlying problem logic rather than superficial patterns; (2) Test-time Hybrid Exploration, which balances accuracy-driven exploitation with consistency-driven exploration across synthetic variants. Extensive experiments show TTVS yields superior performance across eight model architectures. Notably, using only unlabeled test-time data, TTVS not only surpasses other test-time adaptation methods but also outperforms state-of-the-art supervised RL-based techniques trained on vast, high-quality labeled data. 

---
# Synthetic Data for any Differentiable Target 

**Authors**: Tristan Thrush, Sung Min Park, Herman Brunborg, Luke Bailey, Marcel Roed, Neil Band, Christopher Potts, Tatsunori Hashimoto  

**Link**: [PDF](https://arxiv.org/pdf/2604.08423)  

**Abstract**: What are the limits of controlling language models via synthetic training data? We develop a reinforcement learning (RL) primitive, the Dataset Policy Gradient (DPG), which can precisely optimize synthetic data generators to produce a dataset of targeted examples. When used for supervised fine-tuning (SFT) of a target model, these examples cause the target model to do well on a differentiable metric of our choice. Our approach achieves this by taking exact data attribution via higher-order gradients and using those scores as policy gradient rewards. We prove that this procedure closely approximates the true, intractable gradient for the synthetic data generator. To illustrate the potential of DPG, we show that, using only SFT on generated examples, we can cause the target model's LM head weights to (1) embed a QR code, (2) embed the pattern $\texttt{67}$, and (3) have lower $\ell^2$ norm. We additionally show that we can cause the generator to (4) rephrase inputs in a new language and (5) produce a specific UUID, even though neither of these objectives is conveyed in the generator's input prompts. These findings suggest that DPG is a powerful and flexible technique for shaping model properties using only synthetic training examples. 

---
# Faithful GRPO: Improving Visual Spatial Reasoning in Multimodal Language Models via Constrained Policy Optimization 

**Authors**: Sai Srinivas Kancheti, Aditya Kanade, Rohit Sinha, Vineeth N Balasubramanian, Tanuja Ganu  

**Link**: [PDF](https://arxiv.org/pdf/2604.08476)  

**Abstract**: Multimodal reasoning models (MRMs) trained with reinforcement learning with verifiable rewards (RLVR) show improved accuracy on visual reasoning benchmarks. However, we observe that accuracy gains often come at the cost of reasoning quality: generated Chain-of-Thought (CoT) traces are frequently inconsistent with the final answer and poorly grounded in the visual evidence. We systematically study this phenomenon across seven challenging real-world spatial reasoning benchmarks and find that it affects contemporary MRMs such as ViGoRL-Spatial, TreeVGR as well as our own models trained with standard Group Relative Policy Optimization (GRPO). We characterize CoT reasoning quality along two complementary axes: "logical consistency" (does the CoT entail the final answer?) and "visual grounding" (does each reasoning step accurately describe objects, attributes, and spatial relationships in the image?). To address this, we propose Faithful GRPO (FGRPO), a variant of GRPO that enforces consistency and grounding as constraints via Lagrangian dual ascent. FGRPO incorporates batch-level consistency and grounding constraints into the advantage computation within a group, adaptively adjusting the relative importance of constraints during optimization. We evaluate FGRPO on Qwen2.5-VL-7B and 3B backbones across seven spatial datasets. Our results show that FGRPO substantially improves reasoning quality, reducing the inconsistency rate from 24.5% to 1.7% and improving visual grounding scores by +13%. It also improves final answer accuracy over simple GRPO, demonstrating that faithful reasoning enables better answers. 

---
# MedVR: Annotation-Free Medical Visual Reasoning via Agentic Reinforcement Learning 

**Authors**: Zheng Jiang, Heng Guo, Chengyu Fang, Changchen Xiao, Xinyang Hu, Lifeng Sun, Minfeng Xu  

**Link**: [PDF](https://arxiv.org/pdf/2604.08203)  

**Abstract**: Medical Vision-Language Models (VLMs) hold immense promise for complex clinical tasks, but their reasoning capabilities are often constrained by text-only paradigms that fail to ground inferences in visual evidence. This limitation not only curtails performance on tasks requiring fine-grained visual analysis but also introduces risks of visual hallucination in safety-critical applications. Thus, we introduce MedVR, a novel reinforcement learning framework that enables annotation-free visual reasoning for medical VLMs. Its core innovation lies in two synergistic mechanisms: Entropy-guided Visual Regrounding (EVR) uses model uncertainty to direct exploration, while Consensus-based Credit Assignment (CCA) distills pseudo-supervision from rollout agreement. Without any human annotations for intermediate steps, MedVR achieves state-of-the-art performance on diverse public medical VQA benchmarks, significantly outperforming existing models. By learning to reason directly with visual evidence, MedVR promotes the robustness and transparency essential for accelerating the clinical deployment of medical AI. 

---
# 3DrawAgent: Teaching LLM to Draw in 3D with Early Contrastive Experience 

**Authors**: Hongcan Xiao, Xinyue Xiao, Yilin Wang, Yue Zhang, Yonggang Qi  

**Link**: [PDF](https://arxiv.org/pdf/2604.08042)  

**Abstract**: Sketching in 3D space enables expressive reasoning about shape, structure, and spatial relationships, yet generating 3D sketches through natural language remains a major challenge. In this work, we introduce 3DrawAgent, a training-free, language-driven framework for 3D sketch generation that leverages large language models (LLMs) to sequentially draw 3D Bezier curves under geometric feedback. Unlike prior 2D sketch agents, our method introduces a relative experience optimization strategy that adapts the recently proposed Group Reward Policy Optimization (GRPO) paradigm. Instead of relying on explicit ground-truth supervision, we construct pairwise comparisons among generated sketches, with each pair consisting of a relatively better and a worse result based on CLIP-based perceptual rewards and LLM-based fine-grained qualitative assessment. These experiences are then used to iteratively refine the prior knowledge of 3D drawing, enabling black-box reinforcement of the model's 3D awareness. This design allows our model to self-improve its spatial understanding and drawing quality without parameter updates. Experiments show that 3DrawAgent can generate complex and coherent 3D Bezier sketches from diverse textual prompts, exhibit emergent geometric reasoning, and generalize to novel shapes, establishing a new paradigm for advancing the field of training-free 3D sketch intelligence. 

---
# A Decomposition Perspective to Long-context Reasoning for LLMs 

**Authors**: Yanling Xiao, Huaibing Xie, Guoliang Zhao, Shihan Dou, Shaolei Wang, Yiting Liu, Nantao Zheng, Cheng Zhang, Pluto Zhou, Zhisong Zhang, Lemao Liu  

**Link**: [PDF](https://arxiv.org/pdf/2604.07981)  

**Abstract**: Long-context reasoning is essential for complex real-world applications, yet remains a significant challenge for Large Language Models (LLMs). Despite the rapid evolution in long-context reasoning, current research often overlooks the internal complexity of the long-context reasoning task itself. In this paper, we move beyond this holistic view and decompose long-context reasoning into a set of fundamental atomic skills, and we then automatically synthesize a suite of pseudo datasets, each explicitly targeting a specific atomic skill. Our empirical analysis confirms that proficiency in these atomic skills is strongly correlated with general long-text reasoning performance. Building on this insight, we employ reinforcement learning on these pseudo datasets to sharpen the model's atomic skills, in the hope of boosting its general long-context reasoning ability. Extensive experiments across multiple benchmarks demonstrate the effectiveness of our approach: it outperforms a strong baseline by an average margin of 7.7\% (improving from 46.3\% to 54.0\%) across Loogle, Loong, LongBench-v2, BrowscompLong, Ruler-qa2, and MRCR. 

---
# TOOLCAD: Exploring Tool-Using Large Language Models in Text-to-CAD Generation with Reinforcement Learning 

**Authors**: Yifei Gong, Xing Wu, Wenda Liu, Kang Tu  

**Link**: [PDF](https://arxiv.org/pdf/2604.07960)  

**Abstract**: Computer-Aided Design (CAD) is an expert-level task that relies on long-horizon reasoning and coherent modeling actions. Large Language Models (LLMs) have shown remarkable advancements in enabling language agents to tackle real-world tasks. Notably, there has been no investigation into how tool-using LLMs optimally interact with CAD engines, hindering the emergence of LLM-based agentic text-to-CAD modeling systems. We propose ToolCAD, a novel agentic CAD framework deploying LLMs as tool-using agents for text-to-CAD generation. Furthermore, we introduce an interactive CAD modeling gym to rollout reasoning and tool-augmented interaction trajectories with the CAD engine, incorporating hybrid feedback and human supervision. Meanwhile, an end-to-end post-training strategy is presented to enable the LLM agent to elicit refined CAD Modeling Chain of Thought (CAD-CoT) and evolve into proficient CAD tool-using agents via online curriculum reinforcement learning. Our findings demonstrate ToolCAD fills the gap in adopting and training open-source LLMs for CAD tool-using agents, enabling them to perform comparably to proprietary models, paving the way for more accessible and robust autonomous text-to-CAD modeling systems. 

---
# Large Language Model Post-Training: A Unified View of Off-Policy and On-Policy Learning 

**Authors**: Shiwan Zhao, Zhihu Wang, Xuyang Zhao, Jiaming Zhou, Caiyue Xu, Chenfei Liu, Liting Zhang, Yuhang Jia, Yanzhe Zhang, Hualong Yu, Zichen Xu, Qicheng Li, Yong Qin  

**Link**: [PDF](https://arxiv.org/pdf/2604.07941)  

**Abstract**: Post-training has become central to turning pretrained large language models (LLMs) into aligned and deployable systems. Recent progress spans supervised fine-tuning (SFT), preference optimization, reinforcement learning (RL), process supervision, verifier-guided methods, distillation, and multi-stage pipelines. Yet these methods are often discussed in fragmented ways, organized by labels or objective families rather than by the behavioral bottlenecks they address.
This survey argues that LLM post-training is best understood as structured intervention on model behavior. We organize the field first by trajectory provenance, which defines two primary learning regimes: off-policy learning on externally supplied trajectories, and on-policy learning on learner-generated rollouts. We then interpret methods through two recurring roles -- effective support expansion, which makes useful behaviors more reachable, and policy reshaping, which improves behavior within already reachable regions -- together with a complementary systems-level role, behavioral consolidation, which preserves, transfers, and amortizes behavior across stages and model transitions.
This perspective yields a unified reading of major paradigms. SFT may serve either support expansion or policy reshaping, whereas preference-based methods are usually off-policy reshaping. On-policy RL often improves behavior on learner-generated states, though under stronger guidance it can also make hard-to-reach reasoning paths reachable. Distillation is often best understood as consolidation rather than only compression, and hybrid pipelines emerge as coordinated multi-stage compositions.
Overall, the framework helps diagnose post-training bottlenecks and reason about stage composition, suggesting that progress in LLM post-training increasingly depends on coordinated system design rather than any single dominant objective. 

---
# QaRL: Rollout-Aligned Quantization-Aware RL for Fast and Stable Training under Training--Inference Mismatch 

**Authors**: Hao Gu, Hao Wang, Jiacheng Liu, Lujun Li, Qiyuan Zhu, Bei Liu, Binxing Xu, Lei Wang, Xintong Yang, Sida Lin, Sirui Han, Yike Guo  

**Link**: [PDF](https://arxiv.org/pdf/2604.07853)  

**Abstract**: Large language model (LLM) reinforcement learning (RL) pipelines are often bottlenecked by rollout generation, making end-to-end training slow. Recent work mitigates this by running rollouts with quantization to accelerate decoding, which is the most expensive stage of the RL loop. However, these setups destabilize optimization by amplifying the training-inference gap: rollouts are operated at low precision, while learning updates are computed at full precision. To address this challenge, we propose QaRL (Rollout Alignment Quantization-Aware RL), which aligns training-side forward with the quantized rollout to minimize mismatch. We further identify a failure mode in quantized rollouts: long-form responses tend to produce repetitive, garbled tokens (error tokens). To mitigate these problems, we introduce TBPO (Trust-Band Policy Optimization), a sequence-level objective with dual clipping for negative samples, aimed at keeping updates within the trust region. On Qwen3-30B-A3B MoE for math problems, QaRL outperforms quantized-rollout training by +5.5 while improving stability and preserving low-bit throughput benefits. 

---
# ReRec: Reasoning-Augmented LLM-based Recommendation Assistant via Reinforcement Fine-tuning 

**Authors**: Jiani Huang, Shijie Wang, Liangbo Ning, Wenqi Fan, Qing Li  

**Link**: [PDF](https://arxiv.org/pdf/2604.07851)  

**Abstract**: With the rise of LLMs, there is an increasing need for intelligent recommendation assistants that can handle complex queries and provide personalized, reasoning-driven recommendations. LLM-based recommenders show potential but face challenges in multi-step reasoning, underscoring the need for reasoning-augmented systems. To address this gap, we propose ReRec, a novel reinforcement fine-tuning (RFT) framework designed to improve LLM reasoning in complex recommendation tasks. Our framework introduces three key components: (1) Dual-Graph Enhanced Reward Shaping, integrating recommendation metrics like NDCG@K with Query Alignment and Preference Alignment Scores to provide fine-grained reward signals for LLM optimization; (2) Reasoning-aware Advantage Estimation, which decomposes LLM outputs into reasoning segments and penalizes incorrect steps to enhance reasoning of recommendation; and (3) Online Curriculum Scheduler, dynamically assess query difficulty and organize training curriculum to ensure stable learning during RFT. Experiments demonstrate that ReRec outperforms state-of-the-art baselines and preserves core abilities like instruction-following and general knowledge. Our codes are available at this https URL. 

---
# An Imperfect Verifier is Good Enough: Learning with Noisy Rewards 

**Authors**: Andreas Plesner, Francisco Guzmán, Anish Athalye  

**Link**: [PDF](https://arxiv.org/pdf/2604.07666)  

**Abstract**: Reinforcement Learning with Verifiable Rewards (RLVR) has become a prominent method for post-training Large Language Models (LLMs). However, verifiers are rarely error-free; even deterministic checks can be inaccurate, and the growing dependence on model-based judges exacerbates the issue. The extent to which RLVR is robust to such noise and the verifier accuracy required for effective training remain unresolved questions. We investigate these questions in the domains of code generation and scientific reasoning by introducing noise into RL training. Noise rates up to 15% yield peak validation accuracy within 2 percentage points of the clean baseline. These findings are consistent across controlled and model-based noise types, three model families (Qwen3, GLM4, Llama 3.1), and model sizes from 4B to 9B. Overall, the results indicate that imperfect verification does not constitute a fundamental barrier to RLVR. Furthermore, our findings suggest that practitioners should prioritize moderate accuracy with high precision over perfect verification. 

---
# Reinforcement Learning with LLM-Guided Action Spaces for Synthesizable Lead Optimization 

**Authors**: Tao Li, Kaiyuan Hou, Tuan Vinh, Monika Raj, Zhichun Guo, Carl Yang  

**Link**: [PDF](https://arxiv.org/pdf/2604.07669)  

**Abstract**: Lead optimization in drug discovery requires improving therapeutic properties while ensuring that proposed molecular modifications correspond to feasible synthetic routes. Existing approaches either prioritize property scores without enforcing synthesizability, or rely on expensive enumeration over large reaction networks, while direct application of Large Language Models (LLMs) frequently produces chemically invalid structures. We introduce MolReAct, a framework that formulates lead optimization as a Markov Decision Process over a synthesis-constrained action space defined by validated reaction templates. A tool-augmented LLM agent serves as a dynamic reaction environment that invokes specialized chemical analysis tools to identify reactive sites and propose chemically grounded transformations from matched templates. A policy model trained via Group Relative Policy Optimization (GRPO) selects among these constrained actions to maximize long-term oracle reward across multi-step reaction trajectories. A SMILES-based caching mechanism further reduces end-to-end optimization time by approximately 43%. Across 13 property optimization tasks from the Therapeutic Data Commons and one structure-based docking task, MolReAct achieves an average Top-10 score of 0.563, outperforming the strongest synthesizable baseline by 10.4% in relative improvement, and attains the best sample efficiency on 10 of 14 tasks. Ablations confirm that both tool-augmented reaction proposals and trajectory-level policy optimization contribute complementary gains. By grounding every step in validated reaction templates, MolReAct produces molecules that are property-improved and each accompanied by an explicit synthetic pathway. 

---
# SubSearch: Intermediate Rewards for Unsupervised Guided Reasoning in Complex Retrieval 

**Authors**: Roxana Petcu, Evangelos Kanoulas, Maarten de Rijke  

**Link**: [PDF](https://arxiv.org/pdf/2604.07415)  

**Abstract**: Large language models (LLMs) are probabilistic in nature and perform more reliably when augmented with external information. As complex queries often require multi-step reasoning over the retrieved information, with no clear or predetermined reasoning path, they remain challenging. Recent approaches train models using reinforcement learning on the model's outcome, showing promise in improving how models handle complex information. We introduce SubSearch, a specialized framework that shifts from outcome-only supervision to intermediate reward signals that incentivize planning high-quality reasoning. Unlike previous work on process reward modeling, which focuses on training a separate reward model with annotated trajectories by either human annotators or large LLM judges, SubSearch directly optimizes the generator using intrinsic process rewards, which we define as internally-derived rewards, eliminating the need for external supervision, and moving towards autonomous information-intensive reasoning. Experiments on seven benchmarks show that rewarding intermediate reasoning steps with intrinsic rewards leads to more robust reasoning traces in both QA and multi-hop QA datasets over using only outcome rewards. SubSearch can help in building reasoning traces that allow agents to better integrate search engines for complex query answering, while offering a data-efficient alternative to supervised process modeling. 

---
# ReAlign: Optimizing the Visual Document Retriever with Reasoning-Guided Fine-Grained Alignment 

**Authors**: Hao Yang, Yifan Ji, Zhipeng Xu, Zhenghao Liu, Yukun Yan, Zulong Chen, Shuo Wang, Yu Gu, Ge Yu  

**Link**: [PDF](https://arxiv.org/pdf/2604.07419)  

**Abstract**: Visual document retrieval aims to retrieve a set of document pages relevant to a query from visually rich collections. Existing methods often employ Vision-Language Models (VLMs) to encode queries and visual pages into a shared embedding space, which is then optimized via contrastive training. However, during visual document representation, localized evidence is usually scattered across complex document layouts, making it difficult for retrieval models to capture crucial cues for effective embedding learning. In this paper, we propose Reasoning-Guided Alignment (ReAlign), a method that enhances visual document retrieval by leveraging the reasoning capability of VLMs to provide fine-grained visual document descriptions as supervision signals for training. Specifically, ReAlign employs a superior VLM to identify query-related regions on a page and then generates a query-aware description grounding the cropped visual regions. The retriever is then trained using these region-focused descriptions to align the semantics between queries and visual documents by encouraging the document ranking distribution induced by the region-focused descriptions to match that induced by the original query. Experiments on diverse visually rich document retrieval benchmarks demonstrate that ReAlign consistently improves visual document retrieval performance on both in-domain and out-of-domain datasets, achieving up to 2% relative improvements. Moreover, the advantages of ReAlign generalize across different VLM backbones by guiding models to better focus their attention on critical visual cues for document representation. All code and datasets are available at this https URL. 

---
# Self-Debias: Self-correcting for Debiasing Large Language Models 

**Authors**: Xuan Feng, Shuai Zhao, Luwei Xiao, Tianlong Gu, Bo An  

**Link**: [PDF](https://arxiv.org/pdf/2604.08243)  

**Abstract**: Although Large Language Models (LLMs) demonstrate remarkable reasoning capabilities, inherent social biases often cascade throughout the Chain-of-Thought (CoT) process, leading to continuous "Bias Propagation". Existing debiasing methods primarily focus on static constraints or external interventions, failing to identify and interrupt this propagation once triggered. To address this limitation, we introduce Self-Debias, a progressive framework designed to instill intrinsic self-correction capabilities. Specifically, we reformulate the debiasing process as a strategic resource redistribution problem, treating the model's output probability mass as a limited resource to be reallocated from biased heuristics to unbiased reasoning paths. Unlike standard preference optimization which applies broad penalties, Self-Debias employs a fine-grained trajectory-level objective subject to dynamic debiasing constraints. This enables the model to selectively revise biased reasoning suffixes while preserving valid contextual prefixes. Furthermore, we integrate an online self-improvement mechanism utilizing consistency filtering to autonomously synthesize supervision signals. With merely 20k annotated samples, Self-Debias activates efficient self-correction, achieving superior debiasing performance while preserving general reasoning capabilities without continuous external oversight. 

---
# Demystifying OPD: Length Inflation and Stabilization Strategies for Large Language Models 

**Authors**: Feng Luo, Yu-Neng Chuang, Guanchu Wang, Zicheng Xu, Xiaotian Han, Tianyi Zhang, Vladimir Braverman  

**Link**: [PDF](https://arxiv.org/pdf/2604.08527)  

**Abstract**: On-policy distillation (OPD) trains student models under their own induced distribution while leveraging supervision from stronger teachers. We identify a failure mode of OPD: as training progresses, on-policy rollouts can undergo abrupt length inflation, causing truncated trajectories to dominate the training data. This truncation collapse coincides with abrupt repetition saturation and induces biased gradient signals, leading to severe training instability and sharp degradation in validation performance. We attribute this problem to the interaction between student-induced data collection and the distillation objective, which implicitly favors long and repetitive rollouts. To address this issue, we propose StableOPD, a stabilized OPD framework that combines a reference-based divergence constraint with rollout mixture distillation. These together mitigate repetition-induced length inflation and further stabilize OPD training. Across multiple math reasoning datasets, our approach prevents truncation collapse, stabilizes training dynamics, and improves performance by 7.2% on average. 

---
# Guaranteeing Knowledge Integration with Joint Decoding for Retrieval-Augmented Generation 

**Authors**: Zhengyi Zhao, Shubo Zhang, Zezhong Wang, Yuxi Zhang, Huimin Wang, Yutian Zhao, Yefeng Zheng, Binyang Li, Kam-Fai Wong, Xian Wu  

**Link**: [PDF](https://arxiv.org/pdf/2604.08046)  

**Abstract**: Retrieval-Augmented Generation (RAG) significantly enhances Large Language Models (LLMs) by providing access to external knowledge. However, current research primarily focuses on retrieval quality, often overlooking the critical ''integration bottleneck'': even when relevant documents are retrieved, LLMs frequently fail to utilize them effectively due to conflicts with their internal parametric knowledge. In this paper, we argue that implicitly resolving this conflict in a single generation pass is suboptimal. We introduce GuarantRAG, a framework that explicitly decouples reasoning from evidence integration. First, we generate an ''Inner-Answer'' based solely on parametric knowledge to capture the model's reasoning flow. Second, to guarantee faithful evidence extraction, we generate a ''Refer-Answer'' using a novel Contrastive DPO objective. This objective treats the parametric Inner-Answer as a negative constraint and the retrieved documents as positive ground truth, forcing the model to suppress internal hallucinations in favor of external evidence during this phase. Finally, rather than naive concatenation or using the DPO trained model directly, we propose a joint decoding mechanism that dynamically fuses the logical coherence of the Inner-Answer with the factual precision of the Refer-Answer at the token level. Experiments on five QA benchmarks demonstrate that GuarantRAG improves accuracy by up to 12.1% and reduces hallucinations by 16.3% compared to standard and dynamic RAG baselines. 

---
# MemReader: From Passive to Active Extraction for Long-Term Agent Memory 

**Authors**: Jingyi Kang, Chunyu Li, Ding Chen, Bo Tang, Feiyu Xiong, Zhiyu Li  

**Link**: [PDF](https://arxiv.org/pdf/2604.07877)  

**Abstract**: Long-term memory is fundamental for personalized and autonomous agents, yet populating it remains a bottleneck. Existing systems treat memory extraction as a one-shot, passive transcription from context to structured entries, which struggles with noisy dialogue, missing references, and cross-turn dependencies, leading to memory pollution, low-value writes, and inconsistency. In this paper, we introduce the MemReader family for active long-term memory extraction in agent systems: MemReader-0.6B, a compact and cost-efficient passive extractor distilled for accurate and schema-consistent structured outputs, and MemReader-4B, an active extractor optimized with Group Relative Policy Optimization (GRPO) to make memory writing decisions. Under a ReAct-style paradigm, MemReader-4B explicitly evaluates information value, reference ambiguity, and completeness before acting, and can selectively write memories, defer incomplete inputs, retrieve historical context, or discard irrelevant chatter. Experiments on LOCOMO, LongMemEval, and HaluMem show that MemReader consistently outperforms existing extraction-based baselines. In particular, MemReader-4B achieves state-of-the-art performance on tasks involving knowledge updating, temporal reasoning, and hallucination reduction. These results suggest that effective agent memory requires not merely extracting more information, but performing reasoning-driven and selective memory extraction to build low-noise and dynamically evolving long-term memory. Furthermore, MemReader has been integrated into MemOS and is being deployed in real-world applications. To support future research and adoption, we release the models and provide public API access. 

---
# Decompose, Look, and Reason: Reinforced Latent Reasoning for VLMs 

**Authors**: Mengdan Zhu, Senhao Cheng, Liang Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2604.07518)  

**Abstract**: Vision-Language Models often struggle with complex visual reasoning due to the visual information loss in textual CoT. Existing methods either add the cost of tool calls or rely on localized patch-based embeddings that are insufficient to extract semantics in multi-step reasoning. We propose \emph{"Decompose, Look, and Reason" (DLR)}, a reinforced latent reasoning framework that dynamically decomposes queries into textual premises, extracts premise-conditioned continuous visual latents, and deduces answers through grounded rationales. We introduce a three-stage training pipeline and propose a novel Spherical Gaussian Latent Policy to enable effective exploration in the latent space. Extensive experiments on vision-centric benchmarks show that DLR consistently outperforms strong baselines, including text-only, interleaved multimodal CoT, and latent reasoning methods, while providing superior stepwise interpretability. 

---
# Guardian-as-an-Advisor: Advancing Next-Generation Guardian Models for Trustworthy LLMs 

**Authors**: Yue Huang, Haomin Zhuang, Jiayi Ye, Han Bao, Yanbo Wang, Hang Hua, Siyuan Wu, Pin-Yu Chen, Xiangliang Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2604.07655)  

**Abstract**: Hard-gated safety checkers often over-refuse and misalign with a vendor's model spec; prevailing taxonomies also neglect robustness and honesty, yielding safer-on-paper yet less useful systems. This work introduces Guardian-as-an-Advisor (GaaA), a soft-gating pipeline where a guardian predicts a binary risk label plus a concise explanation and prepends this advice to the original query for re-inference, keeping the base model operating under its original spec. To support training and evaluation, GuardSet is constructed, a 208k+ multi-domain dataset unifying harmful and harmless cases with targeted robustness and honesty slices. GuardAdvisor is trained via SFT followed by RL to enforce label-explanation consistency. GuardAdvisor attains competitive detection accuracy while enabling the advisory workflow; when used to augment inputs, responses improve over unaugmented prompts. A latency study shows advisor inference uses below 5% of base-model compute and adds only 2-10% end-to-end overhead under realistic harmful-input rates. Overall, GaaA steers models to comply with the model spec, maintaining safety while reducing over-refusal. 

---
# The Art of (Mis)alignment: How Fine-Tuning Methods Effectively Misalign and Realign LLMs in Post-Training 

**Authors**: Rui Zhang, Hongwei Li, Yun Shen, Xinyue Shen, Wenbo Jiang, Guowen Xu, Yang Liu, Michael Backes, Yang Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2604.07754)  

**Abstract**: The deployment of large language models (LLMs) raises significant ethical and safety concerns. While LLM alignment techniques are adopted to improve model safety and trustworthiness, adversaries can exploit these techniques to undermine safety for malicious purposes, resulting in \emph{misalignment}. Misaligned LLMs may be published on open platforms to magnify harm. To address this, additional safety alignment, referred to as \emph{realignment}, is necessary before deploying untrusted third-party LLMs. This study explores the efficacy of fine-tuning methods in terms of misalignment, realignment, and the effects of their interplay. By evaluating four Supervised Fine-Tuning (SFT) and two Preference Fine-Tuning (PFT) methods across four popular safety-aligned LLMs, we reveal a mechanism asymmetry between attack and defense. While Odds Ratio Preference Optimization (ORPO) is most effective for misalignment, Direct Preference Optimization (DPO) excels in realignment, albeit at the expense of model utility. Additionally, we identify model-specific resistance, residual effects of multi-round adversarial dynamics, and other noteworthy findings. These findings highlight the need for robust safeguards and customized safety alignment strategies to mitigate potential risks in the deployment of LLMs. Our code is available at this https URL. 

---
