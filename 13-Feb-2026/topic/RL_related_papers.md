# P-GenRM: Personalized Generative Reward Model with Test-time User-based Scaling 

**Authors**: Pinyi Zhang, Ting-En Lin, Yuchuan Wu, Jingyang Chen, Zongqi Wang, Hua Yang, Ze Xu, Fei Huang, Kai Zhang, Yongbin Li  

**Link**: [PDF](https://arxiv.org/pdf/2602.12116)  

**Abstract**: Personalized alignment of large language models seeks to adapt responses to individual user preferences, typically via reinforcement learning. A key challenge is obtaining accurate, user-specific reward signals in open-ended scenarios. Existing personalized reward models face two persistent limitations: (1) oversimplifying diverse, scenario-specific preferences into a small, fixed set of evaluation principles, and (2) struggling with generalization to new users with limited feedback. To this end, we propose P-GenRM, the first Personalized Generative Reward Model with test-time user-based scaling. P-GenRM transforms preference signals into structured evaluation chains that derive adaptive personas and scoring rubrics across various scenarios. It further clusters users into User Prototypes and introduces a dual-granularity scaling mechanism: at the individual level, it adaptively scales and aggregates each user's scoring scheme; at the prototype level, it incorporates preferences from similar users. This design mitigates noise in inferred preferences and enhances generalization to unseen users through prototype-based transfer. Empirical results show that P-GenRM achieves state-of-the-art results on widely-used personalized reward model benchmarks, with an average improvement of 2.31%, and demonstrates strong generalization on an out-of-distribution dataset. Notably, Test-time User-based scaling provides an additional 3% boost, demonstrating stronger personalized alignment with test-time scalability. 

---
# Composition-RL: Compose Your Verifiable Prompts for Reinforcement Learning of Large Language Models 

**Authors**: Xin Xu, Clive Bai, Kai Yang, Tianhao Chen, Yangkun Chen, Weijie Liu, Hao Chen, Yang Wang, Saiyong Yang, Can Yang  

**Link**: [PDF](https://arxiv.org/pdf/2602.12036)  

**Abstract**: Large-scale verifiable prompts underpin the success of Reinforcement Learning with Verifiable Rewards (RLVR), but they contain many uninformative examples and are costly to expand further. Recent studies focus on better exploiting limited training data by prioritizing hard prompts whose rollout pass rate is 0. However, easy prompts with a pass rate of 1 also become increasingly prevalent as training progresses, thereby reducing the effective data size. To mitigate this, we propose Composition-RL, a simple yet useful approach for better utilizing limited verifiable prompts targeting pass-rate-1 prompts. More specifically, Composition-RL automatically composes multiple problems into a new verifiable question and uses these compositional prompts for RL training. Extensive experiments across model sizes from 4B to 30B show that Composition-RL consistently improves reasoning capability over RL trained on the original dataset. Performance can be further boosted with a curriculum variant of Composition-RL that gradually increases compositional depth over training. Additionally, Composition-RL enables more effective cross-domain RL by composing prompts drawn from different domains. Codes, datasets, and models are available at this https URL. 

---
# PRIME: Policy-Reinforced Iterative Multi-agent Execution for Algorithmic Reasoning in Large Language Models 

**Authors**: Jiawei Xu, Zhenyu Yu, Ziqian Bi, Minh Duc Pham, Xiaoyi Qu, Danyang Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2602.11170)  

**Abstract**: Large language models have demonstrated remarkable capabilities across diverse reasoning tasks, yet their performance on algorithmic reasoning remains limited. To handle this limitation, we propose PRIME (Policy-Reinforced Iterative Multi-agent Execution), a framework comprising three specialized agents, an executor for step-by-step reasoning, a verifier for constraint checking, and a coordinator for backtracking control, optimized through group relative policy optimization. For comprehensive evaluation, we introduce PRIME-Bench, the largest algorithmic reasoning benchmark to date, comprising 86 tasks across 12 categories with 51,600 instances. Tasks span sorting algorithms, graph and tree structures, automata and state machines, symbolic reasoning, and constraint-based puzzles, with execution traces reaching over one million steps. Compared to baseline approach, PRIME improves average accuracy from 26.8% to 93.8%, a 250% relative gain. The largest improvements occur on tasks requiring sustained state tracking, with Turing machine simulation improving from 9% to 92% and long division from 16% to 94%. Ablation studies identify iterative verification as the primary contributor, preventing the error propagation that causes baseline approaches to fail catastrophically. Analysis across model scales (8B-120B parameters) reveals that smaller models benefit disproportionately, achieving accuracy comparable to models 8x larger. 

---
# Learning beyond Teacher: Generalized On-Policy Distillation with Reward Extrapolation 

**Authors**: Wenkai Yang, Weijie Liu, Ruobing Xie, Kai Yang, Saiyong Yang, Yankai Lin  

**Link**: [PDF](https://arxiv.org/pdf/2602.12125)  

**Abstract**: On-policy distillation (OPD), which aligns the student with the teacher's logit distribution on student-generated trajectories, has demonstrated strong empirical gains in improving student performance and often outperforms off-policy distillation and reinforcement learning (RL) paradigms. In this work, we first theoretically show that OPD is a special case of dense KL-constrained RL where the reward function and the KL regularization are always weighted equally and the reference model can by any model. Then, we propose the Generalized On-Policy Distillation (G-OPD) framework, which extends the standard OPD objective by introducing a flexible reference model and a reward scaling factor that controls the relative weight of the reward term against the KL regularization. Through comprehensive experiments on math reasoning and code generation tasks, we derive two novel insights: (1) Setting the reward scaling factor to be greater than 1 (i.e., reward extrapolation), which we term ExOPD, consistently improves over standard OPD across a range of teacher-student size pairings. In particular, in the setting where we merge the knowledge from different domain experts, obtained by applying domain-specific RL to the same student model, back into the original student, ExOPD enables the student to even surpass the teacher's performance boundary and outperform the domain teachers. (2) Building on ExOPD, we further find that in the strong-to-weak distillation setting (i.e., distilling a smaller student from a larger teacher), performing reward correction by choosing the reference model as the teacher's base model before RL yields a more accurate reward signal and further improves distillation performance. However, this choice assumes access to the teacher's pre-RL variant and incurs more computational overhead. We hope our work offers new insights for future research on OPD. 

---
# Stop Unnecessary Reflection: Training LRMs for Efficient Reasoning with Adaptive Reflection and Length Coordinated Penalty 

**Authors**: Zewei Yu, Lirong Gao, Yuke Zhu, Bo Zheng, Sheng Guo, Haobo Wang, Junbo Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2602.12113)  

**Abstract**: Large Reasoning Models (LRMs) have demonstrated remarkable performance on complex reasoning tasks by employing test-time scaling. However, they often generate over-long chains-of-thought that, driven by substantial reflections such as repetitive self-questioning and circular reasoning, lead to high token consumption, substantial computational overhead, and increased latency without improving accuracy, particularly in smaller models. Our observation reveals that increasing problem complexity induces more excessive and unnecessary reflection, which in turn reduces accuracy and increases token overhead. To address this challenge, we propose Adaptive Reflection and Length Coordinated Penalty (ARLCP), a novel reinforcement learning framework designed to dynamically balance reasoning efficiency and solution accuracy. ARLCP introduces two key innovations: (1) a reflection penalty that adaptively curtails unnecessary reflective steps while preserving essential reasoning, and (2) a length penalty calibrated to the estimated complexity of the problem. By coordinating these penalties, ARLCP encourages the model to generate more concise and effective reasoning paths. We evaluate our method on five mathematical reasoning benchmarks using DeepSeek-R1-Distill-Qwen-1.5B and DeepSeek-R1-Distill-Qwen-7B models. Experimental results show that ARLCP achieves a superior efficiency-accuracy trade-off compared to existing approaches. For the 1.5B model, it reduces the average response length by 53.1% while simultaneously improving accuracy by 5.8%. For the 7B model, it achieves a 35.0% reduction in length with a 2.7% accuracy gain. The code is released at this https URL . 

---
# DICE: Diffusion Large Language Models Excel at Generating CUDA Kernels 

**Authors**: Haolei Bai, Lingcheng Kong, Xueyi Chen, Jianmian Wang, Zhiqiang Tao, Huan Wang  

**Link**: [PDF](https://arxiv.org/pdf/2602.11715)  

**Abstract**: Diffusion large language models (dLLMs) have emerged as a compelling alternative to autoregressive (AR) LLMs, owing to their capacity for parallel token generation. This paradigm is particularly well-suited for code generation, where holistic structural planning and non-sequential refinement are critical. Despite this potential, tailoring dLLMs for CUDA kernel generation remains challenging, obstructed not only by the high specialization but also by the severe lack of high-quality training data. To address these challenges, we construct CuKe, an augmented supervised fine-tuning dataset optimized for high-performance CUDA kernels. On top of it, we propose a bi-phase curated reinforcement learning (BiC-RL) framework consisting of a CUDA kernel infilling stage and an end-to-end CUDA kernel generation stage. Leveraging this training framework, we introduce DICE, a series of diffusion large language models designed for CUDA kernel generation, spanning three parameter scales, 1.7B, 4B, and 8B. Extensive experiments on KernelBench demonstrate that DICE significantly outperforms both autoregressive and diffusion LLMs of comparable scale, establishing a new state-of-the-art for CUDA kernel generation. 

---
# Capability-Oriented Training Induced Alignment Risk 

**Authors**: Yujun Zhou, Yue Huang, Han Bao, Kehan Guo, Zhenwen Liang, Pin-Yu Chen, Tian Gao, Werner Geyer, Nuno Moniz, Nitesh V Chawla, Xiangliang Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2602.12124)  

**Abstract**: While most AI alignment research focuses on preventing models from generating explicitly harmful content, a more subtle risk is emerging: capability-oriented training induced exploitation. We investigate whether language models, when trained with reinforcement learning (RL) in environments with implicit loopholes, will spontaneously learn to exploit these flaws to maximize their reward, even without any malicious intent in their training. To test this, we design a suite of four diverse "vulnerability games", each presenting a unique, exploitable flaw related to context-conditional compliance, proxy metrics, reward tampering, and self-evaluation. Our experiments show that models consistently learn to exploit these vulnerabilities, discovering opportunistic strategies that significantly increase their reward at the expense of task correctness or safety. More critically, we find that these exploitative strategies are not narrow "tricks" but generalizable skills; they can be transferred to new tasks and even "distilled" from a capable teacher model to other student models through data alone. Our findings reveal that capability-oriented training induced risks pose a fundamental challenge to current alignment approaches, suggesting that future AI safety work must extend beyond content moderation to rigorously auditing and securing the training environments and reward mechanisms themselves. Code is available at this https URL. 

---
# TSR: Trajectory-Search Rollouts for Multi-Turn RL of LLM Agents 

**Authors**: Aladin Djuhera, Swanand Ravindra Kadhe, Farhan Ahmed, Holger Boche  

**Link**: [PDF](https://arxiv.org/pdf/2602.11767)  

**Abstract**: Advances in large language models (LLMs) are driving a shift toward using reinforcement learning (RL) to train agents from iterative, multi-turn interactions across tasks. However, multi-turn RL remains challenging as rewards are often sparse or delayed, and environments can be stochastic. In this regime, naive trajectory sampling can hinder exploitation and induce mode collapse. We propose TSR (Trajectory-Search Rollouts), a training-time approach that repurposes test-time scaling ideas for improved per-turn rollout generation. TSR performs lightweight tree-style search to construct high-quality trajectories by selecting high-scoring actions at each turn using task-specific feedback. This improves rollout quality and stabilizes learning while leaving the underlying optimization objective unchanged, making TSR optimizer-agnostic. We instantiate TSR with best-of-N, beam, and shallow lookahead search, and pair it with PPO and GRPO, achieving up to 15% performance gains and more stable learning on Sokoban, FrozenLake, and WebShop tasks at a one-time increase in training compute. By moving search from inference time to the rollout stage of training, TSR provides a simple and general mechanism for stronger multi-turn agent learning, complementary to existing frameworks and rejection-sampling-style selection methods. 

---
# Patch the Distribution Mismatch: RL Rewriting Agent for Stable Off-Policy SFT 

**Authors**: Jiacheng Wang, Ping Jian, Zhen Yang, Zirong Chen, Keren Liao, Zhongbin Guo  

**Link**: [PDF](https://arxiv.org/pdf/2602.11220)  

**Abstract**: Large language models (LLMs) have made rapid progress, yet adapting them to downstream scenarios still commonly relies on supervised fine-tuning (SFT). When downstream data exhibit a substantial distribution shift from the model's prior training distribution, SFT can induce catastrophic forgetting. To narrow this gap, data rewriting has been proposed as a data-centric approach that rewrites downstream training data prior to SFT. However, existing methods typically sample rewrites from a prompt-induced conditional distribution, so the resulting targets are not necessarily aligned with the model's natural QA-style generation distribution. Moreover, reliance on fixed templates can lead to diversity collapse. To address these issues, we cast data rewriting as a policy learning problem and learn a rewriting policy that better matches the backbone's QA-style generation distribution while preserving diversity. Since distributional alignment, diversity and task consistency are automatically evaluable but difficult to optimize end-to-end with differentiable objectives, we leverage reinforcement learning to optimize the rewrite distribution under reward feedback and propose an RL-based data-rewriting agent. The agent jointly optimizes QA-style distributional alignment and diversity under a hard task-consistency gate, thereby constructing a higher-quality rewritten dataset for downstream SFT. Extensive experiments show that our method achieves downstream gains comparable to standard SFT while reducing forgetting on non-downstream benchmarks by 12.34% on average. Our code is available at this https URL . 

---
# Sci-CoE: Co-evolving Scientific Reasoning LLMs via Geometric Consensus with Sparse Supervision 

**Authors**: Xiaohan He, Shiyang Feng, Songtao Huang, Lei Bai, Bin Wang, Bo Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2602.12164)  

**Abstract**: Large language models (LLMs) have demonstrated exceptional reasoning capabilities, and co-evolving paradigms have shown promising results in domains such as code and math. However, in scientific reasoning tasks, these models remain fragile due to unreliable solution evaluation and limited diversity in verification strategies. In this work, we propose Sci-CoE, a two-stage scientific co-evolving framework that enables models to self-evolve as both solver and verifier through a transition from sparse supervision to unsupervised learning. In the first stage, the model uses a small set of annotated data to establish fundamental correctness judgment anchors for the Verifier. In the second stage, we introduce a geometric reward mechanism that jointly considers consensus, reliability, and diversity, driving large-scale self-iteration on unlabeled data. Experiments on several general scientific benchmarks demonstrate that Sci-CoE enhances complex reasoning capabilities and exhibits strong scalability, facilitating the construction of more robust and diverse evaluation systems. Codes are available at this https URL. 

---
# CM2: Reinforcement Learning with Checklist Rewards for Multi-Turn and Multi-Step Agentic Tool Use 

**Authors**: Zhen Zhang, Kaiqiang Song, Xun Wang, Yebowen Hu, Weixiang Yan, Chenyang Zhao, Henry Peng Zou, Haoyun Deng, Sathish Reddy Indurthi, Shujian Liu, Simin Ma, Xiaoyang Wang, Xin Eric Wang, Song Wang  

**Link**: [PDF](https://arxiv.org/pdf/2602.12268)  

**Abstract**: AI agents are increasingly used to solve real-world tasks by reasoning over multi-turn user interactions and invoking external tools. However, applying reinforcement learning to such settings remains difficult: realistic objectives often lack verifiable rewards and instead emphasize open-ended behaviors; moreover, RL for multi-turn, multi-step agentic tool use is still underexplored; and building and maintaining executable tool environments is costly, limiting scale and coverage. We propose CM2, an RL framework that replaces verifiable outcome rewards with checklist rewards. CM2 decomposes each turn's intended behavior into fine-grained binary criteria with explicit evidence grounding and structured metadata, turning open-ended judging into more stable classification-style decisions. To balance stability and informativeness, our method adopts a strategy of sparse reward assignment but dense evaluation criteria. Training is performed in a scalable LLM-simulated tool environment, avoiding heavy engineering for large tool sets. Experiments show that CM2 consistently improves over supervised fine-tuning. Starting from an 8B Base model and training on an 8k-example RL dataset, CM2 improves over the SFT counterpart by 8 points on tau^-Bench, by 10 points on BFCL-V4, and by 12 points on ToolSandbox. The results match or even outperform similarly sized open-source baselines, including the judging model. CM2 thus provides a scalable recipe for optimizing multi-turn, multi-step tool-using agents without relying on verifiable rewards. Code provided by the open-source community: this https URL. 

---
# InjectRBP: Steering Large Language Model Reasoning Behavior via Pattern Injection 

**Authors**: Xiuping Wu, Zhao Yu, Yuxin Cheng, Ngai Wong, Liangjun Ke, Tapas Mishra, Konstantinos V.Katsikopoulos  

**Link**: [PDF](https://arxiv.org/pdf/2602.12013)  

**Abstract**: Reasoning can significantly enhance the performance of Large Language Models. While recent studies have exploited behavior-related prompts adjustment to enhance reasoning, these designs remain largely intuitive and lack a systematic analysis of the underlying behavioral patterns. Motivated by this, we investigate how models' reasoning behaviors shape reasoning from the perspective of behavioral patterns. We observe that models exhibit adaptive distributions of reasoning behaviors when responding to specific types of questions, and that structurally injecting these patterns can substantially influence the quality of the models' reasoning processes and outcomes. Building on these findings, we propose two optimization methods that require no parameter updates: InjectCorrect and InjectRLOpt. InjectCorrect guides the model by imitating behavioral patterns derived from its own past correct answers. InjectRLOpt learns a value function from historical behavior-pattern data and, via our proposed Reliability-Aware Softmax Policy, generates behavioral injectant during inference to steer the reasoning process. Our experiments demonstrate that both methods can improve model performance across various reasoning tasks without requiring any modifications to model parameters, achieving gains of up to 5.34% and 8.67%, respectively. 

---
# RELATE: A Reinforcement Learning-Enhanced LLM Framework for Advertising Text Generation 

**Authors**: Jinfang Wang, Jiajie Liu, Jianwei Wu, Ziqin Luo, Zhen Chen, Chunlei Li, Biao Han, Tao Deng, Yi Li, Shuanglong Li, Lin Liu  

**Link**: [PDF](https://arxiv.org/pdf/2602.11780)  

**Abstract**: In online advertising, advertising text plays a critical role in attracting user engagement and driving advertiser value. Existing industrial systems typically follow a two-stage paradigm, where candidate texts are first generated and subsequently aligned with online performance metrics such as click-through rate(CTR). This separation often leads to misaligned optimization objectives and low funnel efficiency, limiting global optimality.
To address these limitations, we propose RELATE, a reinforcement learning-based end-to-end framework that unifies generation and objective alignment within a single model. Instead of decoupling text generation from downstream metric alignment, RELATE integrates performance and compliance objectives directly into the generation process via policy learning. To better capture ultimate advertiser value beyond click-level signals, We incorporate conversion-oriented metrics into the objective and jointly model them with compliance constraints as multi-dimensional rewards, enabling the model to generate high-quality ad texts that improve conversion performance under policy constraints.
Extensive experiments on large-scale industrial datasets demonstrate that RELATE consistently outperforms baselines. Furthermore, online deployment on a production advertising platform yields statistically significant improvements in click-through conversion rate(CTCVR) under strict policy constraints, validating the robustness and real-world effectiveness of the proposed framework. 

---
# Quark Medical Alignment: A Holistic Multi-Dimensional Alignment and Collaborative Optimization Paradigm 

**Authors**: Tianxiang Xu, Jiayi Liu, Yixuan Tong, Jialu Xu, Yunqing Wei, Kaiwen Feng, PanPan Hou, Kangping Yin, Jiyuan Hu, Hao Zhou, Zhenxin Ma, Jian Xu, Guanjun Jiang  

**Link**: [PDF](https://arxiv.org/pdf/2602.11661)  

**Abstract**: While reinforcement learning for large language model alignment has progressed rapidly in recent years, transferring these paradigms to high-stakes medical question answering reveals a fundamental paradigm mismatch. Reinforcement Learning from Human Feedback relies on preference annotations that are prohibitively expensive and often fail to reflect the absolute correctness of medical facts. Reinforcement Learning from Verifiable Rewards lacks effective automatic verifiers and struggles to handle complex clinical contexts. Meanwhile, medical alignment requires the simultaneous optimization of correctness, safety, and compliance, yet multi-objective heterogeneous reward signals are prone to scale mismatch and optimization this http URL address these challenges, we propose a robust medical alignment paradigm. We first construct a holistic multi-dimensional medical alignment matrix that decomposes alignment objectives into four categories: fundamental capabilities, expert knowledge, online feedback, and format specifications. Within each category, we establish a closed loop of where observable metrics inform attributable diagnosis, which in turn drives optimizable rewards, thereby providing fine-grained, high-resolution supervision signals for subsequent iterative optimization. To resolve gradient domination and optimization instability problem caused by heterogeneous signals, we further propose a unified optimization mechanism. This mechanism employs Reference-Frozen Normalization to align reward scales and implements a Tri-Factor Adaptive Dynamic Weighting strategy to achieve collaborative optimization that is weakness-oriented, risk-prioritized, and redundancy-reducing. Experimental results demonstrate the effectiveness of our proposed paradigm in real-world medical scenario evaluations, establishing a new paradigm for complex alignment in vertical domains. 

---
# Learning to Configure Agentic AI Systems 

**Authors**: Aditya Taparia, Som Sagar, Ransalu Senanayake  

**Link**: [PDF](https://arxiv.org/pdf/2602.11574)  

**Abstract**: Configuring LLM-based agent systems involves choosing workflows, tools, token budgets, and prompts from a large combinatorial design space, and is typically handled today by fixed large templates or hand-tuned heuristics. This leads to brittle behavior and unnecessary compute, since the same cumbersome configuration is often applied to both easy and hard input queries. We formulate agent configuration as a query-wise decision problem and introduce ARC (Agentic Resource & Configuration learner), which learns a light-weight hierarchical policy using reinforcement learning to dynamically tailor these configurations. Across multiple benchmarks spanning reasoning and tool-augmented question answering, the learned policy consistently outperforms strong hand-designed and other baselines, achieving up to 25% higher task accuracy while also reducing token and runtime costs. These results demonstrate that learning per-query agent configurations is a powerful alternative to "one size fits all" designs. 

---
# Credit Where It is Due: Cross-Modality Connectivity Drives Precise Reinforcement Learning for MLLM Reasoning 

**Authors**: Zhengbo Jiao, Shaobo Wang, Zifan Zhang, Wei Wang, Bing Zhao, Hu Wei, Linfeng Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2602.11455)  

**Abstract**: Reinforcement Learning with Verifiable Rewards (RLVR) has significantly advanced the reasoning capabilities of Multimodal Large Language Models (MLLMs), yet how visual evidence is integrated during reasoning remains poorly understood. We explore multimodal RLVR through the lens of cross-modal attention connectivity and find that only a small fraction of tokens (approximately 15%) exhibit strong visual-textual coupling. These high-connectivity tokens act as anchors that ground reasoning in the image, while the majority follow linguistic patterns. During RLVR training, credit assignment naturally concentrates on these anchors, sharpening their visual grounding over time. Building on this insight, we propose Anchor-Token Reinforcement Learning (AT-RL), a lightweight framework that selectively reinforces high-connectivity tokens via graph-based clustering of attention topology. Evaluated across the series (3B-32B), AT-RL introduces only 1.2% overhead yet enables the 32B model to surpass the 72B-Instruct baseline on MathVista (80.2), with consistent gains observed across STEM, video and general tasks. Conversely, training solely on low-connectivity tokens causes severe degradation, confirming that effective multimodal RL hinges on precise credit assignment to visual anchors. Our work reveals that reasoning quality is governed not by token quantity but by the fidelity of cross-modal anchoring. 

---
# Pushing Forward Pareto Frontiers of Proactive Agents with Behavioral Agentic Optimization 

**Authors**: Yihang Yao, Zhepeng Cen, Haohong Lin, Shiqi Liu, Zuxin Liu, Jiacheng Zhu, Zhang-Wei Hong, Laixi Shi, Ding Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2602.11351)  

**Abstract**: Proactive large language model (LLM) agents aim to actively plan, query, and interact over multiple turns, enabling efficient task completion beyond passive instruction following and making them essential for real-world, user-centric applications. Agentic reinforcement learning (RL) has recently emerged as a promising solution for training such agents in multi-turn settings, allowing interaction strategies to be learned from feedback. However, existing pipelines face a critical challenge in balancing task performance with user engagement, as passive agents can not efficiently adapt to users' intentions while overuse of human feedback reduces their satisfaction. To address this trade-off, we propose BAO, an agentic RL framework that combines behavior enhancement to enrich proactive reasoning and information-gathering capabilities with behavior regularization to suppress inefficient or redundant interactions and align agent behavior with user expectations. We evaluate BAO on multiple tasks from the UserRL benchmark suite, and demonstrate that it substantially outperforms proactive agentic RL baselines while achieving comparable or even superior performance to commercial LLM agents, highlighting its effectiveness for training proactive, user-aligned LLM agents in complex multi-turn scenarios. Our website: this https URL. 

---
# DeepGen 1.0: A Lightweight Unified Multimodal Model for Advancing Image Generation and Editing 

**Authors**: Dianyi Wang, Ruihang Li, Feng Han, Chaofan Ma, Wei Song, Siyuan Wang, Yibin Wang, Yi Xin, Hongjian Liu, Zhixiong Zhang, Shengyuan Ding, Tianhang Wang, Zhenglin Cheng, Tao Lin, Cheng Jin, Kaicheng Yu, Jingjing Chen, Wenjie Wang, Zhongyu Wei, Jiaqi Wang  

**Link**: [PDF](https://arxiv.org/pdf/2602.12205)  

**Abstract**: Current unified multimodal models for image generation and editing typically rely on massive parameter scales (e.g., >10B), entailing prohibitive training costs and deployment footprints. In this work, we present DeepGen 1.0, a lightweight 5B unified model that achieves comprehensive capabilities competitive with or surpassing much larger counterparts. To overcome the limitations of compact models in semantic understanding and fine-grained control, we introduce Stacked Channel Bridging (SCB), a deep alignment framework that extracts hierarchical features from multiple VLM layers and fuses them with learnable 'think tokens' to provide the generative backbone with structured, reasoning-rich guidance. We further design a data-centric training strategy spanning three progressive stages: (1) Alignment Pre-training on large-scale image-text pairs and editing triplets to synchronize VLM and DiT representations, (2) Joint Supervised Fine-tuning on a high-quality mixture of generation, editing, and reasoning tasks to foster omni-capabilities, and (3) Reinforcement Learning with MR-GRPO, which leverages a mixture of reward functions and supervision signals, resulting in substantial gains in generation quality and alignment with human preferences, while maintaining stable training progress and avoiding visual artifacts. Despite being trained on only ~50M samples, DeepGen 1.0 achieves leading performance across diverse benchmarks, surpassing the 80B HunyuanImage by 28% on WISE and the 27B Qwen-Image-Edit by 37% on UniREditBench. By open-sourcing our training code, weights, and datasets, we provide an efficient, high-performance alternative to democratize unified multimodal research. 

---
# Mitigating Mismatch within Reference-based Preference Optimization 

**Authors**: Suqin Yuan, Xingrui Yu, Jiyang Zheng, Lei Feng, Dadong Wang, Ivor Tsang, Tongliang Liu  

**Link**: [PDF](https://arxiv.org/pdf/2602.11902)  

**Abstract**: Direct Preference Optimization (DPO) has become the de facto standard for offline preference alignment of large language models, but its reliance on a reference policy introduces a critical tension. DPO weighs each update relative to a reference, which stabilizes the training by regularizing the updates within a trusted region. This reliance becomes problematic for pessimistic pairs, where the reference model prefers the rejected response. For these pairs, DPO prematurely attenuates the gradient as soon as the policy margin ($\Delta_\theta$) merely beats the reference margin ($\Delta_{\mathrm{ref}}$) even if the policy is still wrong ($\Delta_{\theta}<0$). We name this failure premature satisfaction, which is a concrete form of the training-inference mismatch. Reference-free objectives remove this mismatch by optimizing the absolute margin, but at the cost of discarding the stabilizing signal of the reference. We mitigate this tension with Hybrid-DPO (HyPO), a drop-in modification to DPO that applies reference conditionally: HyPO behaves exactly like DPO when the reference is optimistic or neutral, and it treats the reference as neutral when it is pessimistic by replacing $\Delta_\theta-\Delta_{\mathrm{ref}}$ with $\Delta_\theta-\max\{0,\Delta_{\mathrm{ref}}\}$. This one-line change strictly strengthens per-example learning signals on pessimistic pairs while preserving DPO's objective form and computational cost. By conditionally debiasing the pessimistic reference signal, HyPO mitigates premature satisfaction; empirically, across preference alignment, HyPO improves inference-aligned metrics and achieves higher pairwise win rates. Our results provide evidence that direct preference alignment could be enhanced by conditionally debiasing the reference signal, rather than discarding it. 

---
