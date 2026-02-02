# Prepare Reasoning Language Models for Multi-Agent Debate with Self-Debate Reinforcement Learning 

**Authors**: Chenxi Liu, Yanshuo Chen, Ruibo Chen, Tianyi Xiong, Tong Zheng, Heng Huang  

**Link**: [PDF](https://arxiv.org/pdf/2601.22297)  

**Abstract**: The reasoning abilities of large language models (LLMs) have been substantially improved by reinforcement learning with verifiable rewards (RLVR). At test time, collaborative reasoning through Multi-Agent Debate (MAD) has emerged as a promising approach for enhancing LLM performance. However, current RLVR methods typically train LLMs to solve problems in isolation, without explicitly preparing them to synthesize and benefit from different rationales that arise during debate. In this work, we propose Self-Debate Reinforcement Learning (SDRL), a training framework that equips a single LLM with strong standalone problem-solving ability and the capability to learn from diverse reasoning trajectories in MAD. Given a prompt, SDRL first samples multiple candidate solutions, then constructs a debate context with diverse reasoning paths and generates second-turn responses conditioned on this context. Finally, SDRL jointly optimizes both the initial and debate-conditioned responses, yielding a model that is effective as both a standalone solver and a debate participant. Experiments across multiple base models and reasoning benchmarks show that SDRL improves overall MAD performance while simultaneously strengthening single model reasoning. 

---
# SP^2DPO: An LLM-assisted Semantic Per-Pair DPO Generalization 

**Authors**: Chaoyue He, Xin Zhou, Di Wang, Hong Xu, Wei Liu, Chunyan Miao  

**Link**: [PDF](https://arxiv.org/pdf/2601.22385)  

**Abstract**: Direct Preference Optimization (DPO) controls the trade-off between fitting preference labels and staying close to a reference model using a single global temperature beta, implicitly treating all preference pairs as equally informative. Real-world preference corpora are heterogeneous: they mix high-signal, objective failures (for example, safety, factuality, instruction violations) with low-signal or subjective distinctions (for example, style), and also include label noise. We introduce our method, SP2DPO (Semantic Per-Pair DPO), a generalization that replaces the global temperature with an instance-specific schedule beta_i pre-decided offline from structured semantic-gap annotations (category, magnitude, confidence) produced by teacher language models. We instantiate this procedure on the UltraFeedback preference corpus (59,960 pairs), enabling large-scale construction of an auditable beta_i artifact, and incur zero training-time overhead: the inner-loop optimizer remains standard DPO with beta set per pair. We focus our empirical study on AlpacaEval 2.0, reporting both raw win rate and length-controlled win rate. Across four open-weight, instruction-tuned student backbones (4B-8B), SP2DPO is competitive with a tuned global-beta DPO baseline and improves AlpacaEval 2.0 length-controlled win rate on two of four backbones, while avoiding per-model beta sweeps. All code, annotations, and artifacts will be released. 

---
# From Self-Evolving Synthetic Data to Verifiable-Reward RL: Post-Training Multi-turn Interactive Tool-Using Agents 

**Authors**: Jiaxuan Gao, Jiaao Chen, Chuyi He, Wei-Chen Wang, Shusheng Xu, Hanrui Wang, Di Jin, Yi Wu  

**Link**: [PDF](https://arxiv.org/pdf/2601.22607)  

**Abstract**: Interactive tool-using agents must solve real-world tasks via multi-turn interaction with both humans and external environments, requiring dialogue state tracking, multi-step tool execution, while following complex instructions. Post-training such agents is challenging because synthesis for high-quality multi-turn tool-use data is difficult to scale, and reinforcement learning (RL) could face noisy signals caused by user simulation, leading to degraded training efficiency. We propose a unified framework that combines a self-evolving data agent with verifier-based RL. Our system, EigenData, is a hierarchical multi-agent engine that synthesizes tool-grounded dialogues together with executable per-instance checkers, and improves generation reliability via closed-loop self-evolving process that updates prompts and workflow. Building on the synthetic data, we develop an RL recipe that first fine-tunes the user model and then applies GRPO-style training with trajectory-level group-relative advantages and dynamic filtering, yielding consistent improvements beyond SFT. Evaluated on tau^2-bench, our best model reaches 73.0% pass^1 on Airline and 98.3% pass^1 on Telecom, matching or exceeding frontier models. Overall, our results suggest a scalable pathway for bootstrapping complex tool-using behaviors without expensive human annotation. 

---
# TTCS: Test-Time Curriculum Synthesis for Self-Evolving 

**Authors**: Chengyi Yang, Zhishang Xiang, Yunbo Tang, Zongpei Teng, Chengsong Huang, Fei Long, Yuhan Liu, Jinsong Su  

**Link**: [PDF](https://arxiv.org/pdf/2601.22628)  

**Abstract**: Test-Time Training offers a promising way to improve the reasoning ability of large language models (LLMs) by adapting the model using only the test questions. However, existing methods struggle with difficult reasoning problems for two reasons: raw test questions are often too difficult to yield high-quality pseudo-labels, and the limited size of test sets makes continuous online updates prone to instability. To address these limitations, we propose TTCS, a co-evolving test-time training framework. Specifically, TTCS initializes two policies from the same pretrained model: a question synthesizer and a reasoning solver. These policies evolve through iterative optimization: the synthesizer generates progressively challenging question variants conditioned on the test questions, creating a structured curriculum tailored to the solver's current capability, while the solver updates itself using self-consistency rewards computed from multiple sampled responses on both original test and synthetic questions. Crucially, the solver's feedback guides the synthesizer to generate questions aligned with the model's current capability, and the generated question variants in turn stabilize the solver's test-time training. Experiments show that TTCS consistently strengthens the reasoning ability on challenging mathematical benchmarks and transfers to general-domain tasks across different LLM backbones, highlighting a scalable path towards dynamically constructing test-time curricula for self-evolving. Our code and implementation details are available at this https URL. 

---
# THINKSAFE: Self-Generated Safety Alignment for Reasoning Models 

**Authors**: Seanie Lee, Sangwoo Park, Yumin Choi, Gyeongman Kim, Minki Kang, Jihun Yun, Dongmin Park, Jongho Park, Sung Ju Hwang  

**Link**: [PDF](https://arxiv.org/pdf/2601.23143)  

**Abstract**: Large reasoning models (LRMs) achieve remarkable performance by leveraging reinforcement learning (RL) on reasoning tasks to generate long chain-of-thought (CoT) reasoning. However, this over-optimization often prioritizes compliance, making models vulnerable to harmful prompts. To mitigate this safety degradation, recent approaches rely on external teacher distillation, yet this introduces a distributional discrepancy that degrades native reasoning. We propose ThinkSafe, a self-generated alignment framework that restores safety alignment without external teachers. Our key insight is that while compliance suppresses safety mechanisms, models often retain latent knowledge to identify harm. ThinkSafe unlocks this via lightweight refusal steering, guiding the model to generate in-distribution safety reasoning traces. Fine-tuning on these self-generated responses effectively realigns the model while minimizing distribution shift. Experiments on DeepSeek-R1-Distill and Qwen3 show ThinkSafe significantly improves safety while preserving reasoning proficiency. Notably, it achieves superior safety and comparable reasoning to GRPO, with significantly reduced computational cost. Code, models, and datasets are available at this https URL. 

---
# Golden Goose: A Simple Trick to Synthesize Unlimited RLVR Tasks from Unverifiable Internet Text 

**Authors**: Ximing Lu, David Acuna, Jaehun Jung, Jian Hu, Di Zhang, Shizhe Diao, Yunheng Zou, Shaokun Zhang, Brandon Cui, Mingjie Liu, Hyunwoo Kim, Prithviraj Ammanabrolu, Jan Kautz, Yi Dong, Yejin Choi  

**Link**: [PDF](https://arxiv.org/pdf/2601.22975)  

**Abstract**: Reinforcement Learning with Verifiable Rewards (RLVR) has become a cornerstone for unlocking complex reasoning in Large Language Models (LLMs). Yet, scaling up RL is bottlenecked by limited existing verifiable data, where improvements increasingly saturate over prolonged training. To overcome this, we propose Golden Goose, a simple trick to synthesize unlimited RLVR tasks from unverifiable internet text by constructing a multiple-choice question-answering version of the fill-in-the-middle task. Given a source text, we prompt an LLM to identify and mask key reasoning steps, then generate a set of diverse, plausible distractors. This enables us to leverage reasoning-rich unverifiable corpora typically excluded from prior RLVR data construction (e.g., science textbooks) to synthesize GooseReason-0.7M, a large-scale RLVR dataset with over 0.7 million tasks spanning mathematics, programming, and general scientific domains. Empirically, GooseReason effectively revives models saturated on existing RLVR data, yielding robust, sustained gains under continuous RL and achieving new state-of-the-art results for 1.5B and 4B-Instruct models across 15 diverse benchmarks. Finally, we deploy Golden Goose in a real-world setting, synthesizing RLVR tasks from raw FineWeb scrapes for the cybersecurity domain, where no prior RLVR data exists. Training Qwen3-4B-Instruct on the resulting data GooseReason-Cyber sets a new state-of-the-art in cybersecurity, surpassing a 7B domain-specialized model with extensive domain-specific pre-training and post-training. This highlights the potential of automatically scaling up RLVR data by exploiting abundant, reasoning-rich, unverifiable internet text. 

---
# MulFeRL: Enhancing Reinforcement Learning with Verbal Feedback in a Multi-turn Loop 

**Authors**: Xuancheng Li, Haitao Li, Yujia Zhou, YiqunLiu, Qingyao Ai  

**Link**: [PDF](https://arxiv.org/pdf/2601.22900)  

**Abstract**: Reinforcement Learning with Verifiable Rewards (RLVR) is widely used to improve reasoning in multiple domains, yet outcome-only scalar rewards are often sparse and uninformative, especially on failed samples, where they merely indicate failure and provide no insight into why the reasoning fails. In this paper, we investigate how to leverage richer verbal feedback to guide RLVR training on failed samples, and how to convert such feedback into a trainable learning signal. Specifically, we propose a multi-turn feedback-guided reinforcement learning framework. It builds on three mechanisms: (1) dynamic multi-turn regeneration guided by feedback, triggered only on failed samples, (2) two complementary learning signals for within-turn and cross-turn optimization, and (3) structured feedback injection into the model's reasoning process. Trained on sampled OpenR1-Math, the approach outperforms supervised fine-tuning and RLVR baselines in-domain and generalizes well out-of-domain. 

---
# Real-Time Aligned Reward Model beyond Semantics 

**Authors**: Zixuan Huang, Xin Xia, Yuxi Ren, Jianbin Zheng, Xuefeng Xiao, Hongyan Xie, Li Huaqiu, Songshi Liang, Zhongxiang Dai, Fuzhen Zhuang, Jianxin Li, Yikun Ban, Deqing Wang  

**Link**: [PDF](https://arxiv.org/pdf/2601.22664)  

**Abstract**: Reinforcement Learning from Human Feedback (RLHF) is a pivotal technique for aligning large language models (LLMs) with human preferences, yet it is susceptible to reward overoptimization, in which policy models overfit to the reward model, exploit spurious reward patterns instead of faithfully capturing human intent. Prior mitigations primarily relies on surface semantic information and fails to efficiently address the misalignment between the reward model (RM) and the policy model caused by continuous policy distribution shifts. This inevitably leads to an increasing reward discrepancy, exacerbating reward overoptimization. To address these limitations, we introduce R2M (Real-Time Aligned Reward Model), a novel lightweight RLHF framework. R2M goes beyond vanilla reward models that solely depend on the semantic representations of a pretrained LLM. Instead, it leverages the evolving hidden states of the policy (namely policy feedback) to align with the real-time distribution shift of the policy during the RL process. This work points to a promising new direction for improving the performance of reward models through real-time utilization of feedback from policy models. 

---
# UCPO: Uncertainty-Aware Policy Optimization 

**Authors**: Xianzhou Zeng, Jing Huang, Chunmei Xie, Gongrui Nan, Siye Chen, Mengyu Lu, Weiqi Xiong, Qixuan Zhou, Junhao Zhang, Qiang Zhu, Yadong Li, Xingzhong Xu  

**Link**: [PDF](https://arxiv.org/pdf/2601.22648)  

**Abstract**: The key to building trustworthy Large Language Models (LLMs) lies in endowing them with inherent uncertainty expression capabilities to mitigate the hallucinations that restrict their high-stakes applications. However, existing RL paradigms such as GRPO often suffer from Advantage Bias due to binary decision spaces and static uncertainty rewards, inducing either excessive conservatism or overconfidence. To tackle this challenge, this paper unveils the root causes of reward hacking and overconfidence in current RL paradigms incorporating uncertainty-based rewards, based on which we propose the UnCertainty-Aware Policy Optimization (UCPO) framework. UCPO employs Ternary Advantage Decoupling to separate and independently normalize deterministic and uncertain rollouts, thereby eliminating advantage bias. Furthermore, a Dynamic Uncertainty Reward Adjustment mechanism is introduced to calibrate uncertainty weights in real-time according to model evolution and instance difficulty. Experimental results in mathematical reasoning and general tasks demonstrate that UCPO effectively resolves the reward imbalance, significantly improving the reliability and calibration of the model beyond their knowledge boundaries. 

---
# TSPO: Breaking the Double Homogenization Dilemma in Multi-turn Search Policy Optimization 

**Authors**: Shichao Ma, Zhiyuan Ma, Ming Yang, Xiaofan Li, Xing Wu, Jintao Du, Yu Cheng, Weiqiang Wang, Qiliang Liu, Zhengyang Zhou, Yang Wang  

**Link**: [PDF](https://arxiv.org/pdf/2601.22776)  

**Abstract**: Multi-turn tool-integrated reasoning enables Large Language Models (LLMs) to solve complex tasks through iterative information retrieval. However, current reinforcement learning (RL) frameworks for search-augmented reasoning predominantly rely on sparse outcome-level rewards, leading to a "Double Homogenization Dilemma." This manifests as (1) Process homogenization, where the thinking, reasoning, and tooling involved in generation are ignored. (2) Intra-group homogenization, coarse-grained outcome rewards often lead to inefficiencies in intra-group advantage estimation with methods like Group Relative Policy Optimization (GRPO) during sampling. To address this, we propose Turn-level Stage-aware Policy Optimization (TSPO). TSPO introduces the First-Occurrence Latent Reward (FOLR) mechanism, allocating partial rewards to the step where the ground-truth answer first appears, thereby preserving process-level signals and increasing reward variance within groups without requiring external reward models or any annotations. Extensive experiments demonstrate that TSPO significantly outperforms state-of-the-art baselines, achieving average performance gains of 24% and 13.6% on Qwen2.5-3B and 7B models, respectively. 

---
# Sparks of Rationality: Do Reasoning LLMs Align with Human Judgment and Choice? 

**Authors**: Ala N. Tak, Amin Banayeeanzade, Anahita Bolourani, Fatemeh Bahrani, Ashutosh Chaubey, Sai Praneeth Karimireddy, Norbert Schwarz, Jonathan Gratch  

**Link**: [PDF](https://arxiv.org/pdf/2601.22329)  

**Abstract**: Large Language Models (LLMs) are increasingly positioned as decision engines for hiring, healthcare, and economic judgment, yet real-world human judgment reflects a balance between rational deliberation and emotion-driven bias. If LLMs are to participate in high-stakes decisions or serve as models of human behavior, it is critical to assess whether they exhibit analogous patterns of (ir)rationalities and biases. To this end, we evaluate multiple LLM families on (i) benchmarks testing core axioms of rational choice and (ii) classic decision domains from behavioral economics and social norms where emotions are known to shape judgment and choice. Across settings, we show that deliberate "thinking" reliably improves rationality and pushes models toward expected-value maximization. To probe human-like affective distortions and their interaction with reasoning, we use two emotion-steering methods: in-context priming (ICP) and representation-level steering (RLS). ICP induces strong directional shifts that are often extreme and difficult to calibrate, whereas RLS produces more psychologically plausible patterns but with lower reliability. Our results suggest that the same mechanisms that improve rationality also amplify sensitivity to affective interventions, and that different steering methods trade off controllability against human-aligned behavior. Overall, this points to a tension between reasoning and affective steering, with implications for both human simulation and the safe deployment of LLM-based decision systems. 

---
# Enhancing TableQA through Verifiable Reasoning Trace Reward 

**Authors**: Tung Sum Thomas Kwok, Xinyu Wang, Hengzhi He, Xiaofeng Lin, Peng Lu, Liheng Ma, Chunhe Wang, Ying Nian Wu, Lei Ding, Guang Cheng  

**Link**: [PDF](https://arxiv.org/pdf/2601.22530)  

**Abstract**: A major challenge in training TableQA agents, compared to standard text- and image-based agents, is that answers cannot be inferred from a static input but must be reasoned through stepwise transformations of the table state, introducing multi-step reasoning complexity and environmental interaction. This leads to a research question: Can explicit feedback on table transformation action improve model reasoning capability? In this work, we introduce RE-Tab, a plug-and-play framework that architecturally enhances trajectory search via lightweight, training-free reward modeling by formulating the problem as a Partially Observable Markov Decision Process. We demonstrate that providing explicit verifiable rewards during State Transition (``What is the best action?'') and Simulative Reasoning (``Am I sure about the output?'') is crucial to steer the agent's navigation in table states. By enforcing stepwise reasoning with reward feedback in table transformations, RE-Tab achieves state-of-the-art performance in TableQA with almost 25\% drop in inference cost. Furthermore, a direct plug-and-play implementation of RE-Tab brings up to 41.77% improvement in QA accuracy and 33.33% drop in test-time inference samples for consistent answer. Consistent improvement pattern across various LLMs and state-of-the-art benchmarks further confirms RE-Tab's generalisability. The repository is available at this https URL . 

---
# Why Self-Rewarding Works: Theoretical Guarantees for Iterative Alignment of Language Models 

**Authors**: Shi Fu, Yingjie Wang, Shengchao Hu, Peng Wang, Dacheng Tao  

**Link**: [PDF](https://arxiv.org/pdf/2601.22513)  

**Abstract**: Self-Rewarding Language Models (SRLMs) achieve notable success in iteratively improving alignment without external feedback. Yet, despite their striking empirical progress, the core mechanisms driving their capabilities remain unelucidated, leaving a critical gap in theoretical understanding. This paper provides the first rigorous theoretical guarantees for SRLMs. We first establish a lower bound that characterizes the fundamental limits of a single update step, revealing a critical dependence on the quality of the initial model. We then derive finite-sample error bounds for the full iterative paradigm, showing that performance improves at a rate of $\widetilde{\mathcal{O}}\left(1/\sqrt{n}\right)$ with sample size $n$. Crucially, our analysis reveals that the dependence on the initial model decays exponentially with the number of iterations $T$. This provides a formal explanation for why self-rewarding succeeds: it robustly overcomes poor initialization by steering the dynamics toward internal stability and consistency. Finally, we instantiate our theoretical framework for the linear softmax model class, yielding tailored guarantees that connect our high-level insights to practical model architectures. 

---
# Med-Scout: Curing MLLMs' Geometric Blindness in Medical Perception via Geometry-Aware RL Post-Training 

**Authors**: Anglin Liu, Ruichao Chen, Yi Lu, Hongxia Xu, Jintai Chen  

**Link**: [PDF](https://arxiv.org/pdf/2601.23220)  

**Abstract**: Despite recent Multimodal Large Language Models (MLLMs)' linguistic prowess in medical diagnosis, we find even state-of-the-art MLLMs suffer from a critical perceptual deficit: geometric blindness. This failure to ground outputs in objective geometric constraints leads to plausible yet factually incorrect hallucinations, rooted in training paradigms that prioritize linguistic fluency over geometric fidelity. This paper introduces Med-Scout, a novel framework that "cures" this blindness via Reinforcement Learning (RL) that leverages the intrinsic geometric logic latent within unlabeled medical images. Instead of relying on costly expert annotations, Med-Scout derives verifiable supervision signals through three strategic proxy tasks: Hierarchical Scale Localization, Topological Jigsaw Reconstruction, and Anomaly Consistency Detection. To rigorously quantify this deficit, we present Med-Scout-Bench, a new benchmark specifically designed to evaluate geometric perception. Extensive evaluations show that Med-Scout significantly mitigates geometric blindness, outperforming leading proprietary and open-source MLLMs by over 40% on our benchmark. Furthermore, this enhanced geometric perception generalizes to broader medical understanding, achieving superior results on radiological and comprehensive medical VQA tasks. 

---
