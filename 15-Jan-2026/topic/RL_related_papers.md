# Omni-R1: Towards the Unified Generative Paradigm for Multimodal Reasoning 

**Authors**: Dongjie Cheng, Yongqi Li, Zhixin Ma, Hongru Cai, Yupeng Hu, Wenjie Wang, Liqiang Nie, Wenjie Li  

**Link**: [PDF](https://arxiv.org/pdf/2601.09536)  

**Abstract**: Multimodal Large Language Models (MLLMs) are making significant progress in multimodal reasoning. Early approaches focus on pure text-based reasoning. More recent studies have incorporated multimodal information into the reasoning steps; however, they often follow a single task-specific reasoning pattern, which limits their generalizability across various multimodal tasks. In fact, there are numerous multimodal tasks requiring diverse reasoning skills, such as zooming in on a specific region or marking an object within an image. To address this, we propose unified generative multimodal reasoning, which unifies diverse multimodal reasoning skills by generating intermediate images during the reasoning process. We instantiate this paradigm with Omni-R1, a two-stage SFT+RL framework featuring perception alignment loss and perception reward, thereby enabling functional image generation. Additionally, we introduce Omni-R1-Zero, which eliminates the need for multimodal annotations by bootstrapping step-wise visualizations from text-only reasoning data. Empirical results show that Omni-R1 achieves unified generative reasoning across a wide range of multimodal tasks, and Omni-R1-Zero can match or even surpass Omni-R1 on average, suggesting a promising direction for generative multimodal reasoning. 

---
# Collaborative Multi-Agent Test-Time Reinforcement Learning for Reasoning 

**Authors**: Zhiyuan Hu, Yunhai Hu, Juncheng Liu, Shuyue Stella Li, Yucheng Wang, Zhen Xu, See-Kiong Ng, Anh Tuan Luu, Xinxing Xu, Bryan Hooi, Cynthia Breazeal, Hae Won Park  

**Link**: [PDF](https://arxiv.org/pdf/2601.09667)  

**Abstract**: Multi-agent systems have evolved into practical LLM-driven collaborators for many applications, gaining robustness from diversity and cross-checking. However, multi-agent RL (MARL) training is resource-intensive and unstable: co-adapting teammates induce non-stationarity, and rewards are often sparse and high-variance. Therefore, we introduce \textbf{Multi-Agent Test-Time Reinforcement Learning (MATTRL)}, a framework that injects structured textual experience into multi-agent deliberation at inference time. MATTRL forms a multi-expert team of specialists for multi-turn discussions, retrieves and integrates test-time experiences, and reaches consensus for final decision-making. We also study credit assignment for constructing a turn-level experience pool, then reinjecting it into the dialogue. Across challenging benchmarks in medicine, math, and education, MATTRL improves accuracy by an average of 3.67\% over a multi-agent baseline, and by 8.67\% over comparable single-agent baselines. Ablation studies examine different credit-assignment schemes and provide a detailed comparison of how they affect training outcomes. MATTRL offers a stable, effective and efficient path to distribution-shift-robust multi-agent reasoning without tuning. 

---
# RISER: Orchestrating Latent Reasoning Skills for Adaptive Activation Steering 

**Authors**: Wencheng Ye, Liang Peng, Xiaoyang Yuan, Yi Bin, Pengpeng Zeng, Hengyu Jin, Heng Tao Shen  

**Link**: [PDF](https://arxiv.org/pdf/2601.09269)  

**Abstract**: Recent work on domain-specific reasoning with large language models (LLMs) often relies on training-intensive approaches that require parameter updates. While activation steering has emerged as a parameter efficient alternative, existing methods apply static, manual interventions that fail to adapt to the dynamic nature of complex reasoning. To address this limitation, we propose RISER (Router-based Intervention for Steerable Enhancement of Reasoning), a plug-and-play intervention framework that adaptively steers LLM reasoning in activation space. RISER constructs a library of reusable reasoning vectors and employs a lightweight Router to dynamically compose them for each input. The Router is optimized via reinforcement learning under task-level rewards, activating latent cognitive primitives in an emergent and compositional manner. Across seven diverse benchmarks, RISER yields 3.4-6.5% average zero-shot accuracy improvements over the base model while surpassing CoT-style reasoning with 2-3x higher token efficiency and robust accuracy gains. Further analysis shows that RISER autonomously combines multiple vectors into interpretable, precise control strategies, pointing toward more controllable and efficient LLM reasoning. 

---
# Efficient Paths and Dense Rewards: Probabilistic Flow Reasoning for Large Language Models 

**Authors**: Yan Liu, Feng Zhang, Zhanyu Ma, Jun Xu, Jiuchong Gao, Jinghua Hao, Renqing He, Han Liu, Yangdong Deng  

**Link**: [PDF](https://arxiv.org/pdf/2601.09260)  

**Abstract**: High-quality chain-of-thought has demonstrated strong potential for unlocking the reasoning capabilities of large language models. However, current paradigms typically treat the reasoning process as an indivisible sequence, lacking an intrinsic mechanism to quantify step-wise information gain. This granularity gap manifests in two limitations: inference inefficiency from redundant exploration without explicit guidance, and optimization difficulty due to sparse outcome supervision or costly external verifiers. In this work, we propose CoT-Flow, a framework that reconceptualizes discrete reasoning steps as a continuous probabilistic flow, quantifying the contribution of each step toward the ground-truth answer. Built on this formulation, CoT-Flow enables two complementary methodologies: flow-guided decoding, which employs a greedy flow-based decoding strategy to extract information-efficient reasoning paths, and flow-based reinforcement learning, which constructs a verifier-free dense reward function. Experiments on challenging benchmarks demonstrate that CoT-Flow achieves a superior balance between inference efficiency and reasoning performance. 

---
# Reward Learning through Ranking Mean Squared Error 

**Authors**: Chaitanya Kharyal, Calarina Muslimani, Matthew E. Taylor  

**Link**: [PDF](https://arxiv.org/pdf/2601.09236)  

**Abstract**: Reward design remains a significant bottleneck in applying reinforcement learning (RL) to real-world problems. A popular alternative is reward learning, where reward functions are inferred from human feedback rather than manually specified. Recent work has proposed learning reward functions from human feedback in the form of ratings, rather than traditional binary preferences, enabling richer and potentially less cognitively demanding supervision. Building on this paradigm, we introduce a new rating-based RL method, Ranked Return Regression for RL (R4). At its core, R4 employs a novel ranking mean squared error (rMSE) loss, which treats teacher-provided ratings as ordinal targets. Our approach learns from a dataset of trajectory-rating pairs, where each trajectory is labeled with a discrete rating (e.g., "bad," "neutral," "good"). At each training step, we sample a set of trajectories, predict their returns, and rank them using a differentiable sorting operator (soft ranks). We then optimize a mean squared error loss between the resulting soft ranks and the teacher's ratings. Unlike prior rating-based approaches, R4 offers formal guarantees: its solution set is provably minimal and complete under mild assumptions. Empirically, using simulated human feedback, we demonstrate that R4 consistently matches or outperforms existing rating and preference-based RL methods on robotic locomotion benchmarks from OpenAI Gym and the DeepMind Control Suite, while requiring significantly less feedback. 

---
# PediaMind-R1: A Temperament-Aware Language Model for Personalized Early Childhood Care Reasoning via Cognitive Modeling and Preference Alignment 

**Authors**: Zihe Zhang, Can Zhang, Yanheng Xu, Xin Hu, Jichao Leng  

**Link**: [PDF](https://arxiv.org/pdf/2601.08848)  

**Abstract**: This paper presents PediaMind-R1, a domain-specialized large language model designed to achieve active personalization in intelligent parenting scenarios. Unlike conventional systems that provide generic suggestions, PediaMind-R1 draws on insights from developmental psychology. It introduces temperament theory from the Thomas-Chess framework and builds a temperament knowledge graph for infants and toddlers (0-3 years). Our two-stage training pipeline first uses supervised fine-tuning to teach structured chain-of-thought reasoning, and then applies a GRPO-based alignment stage to reinforce logical consistency, domain expertise, and empathetic caregiving strategies. We further design an evaluation framework comprising temperament-sensitive multiple-choice tests and human assessments. The results demonstrate that PediaMind-R1 can accurately interpret early childhood temperament profiles and proactively engage in individualized reasoning. This work highlights the value of integrating vertical-domain modeling with psychological theory. It offers a novel approach to developing user-centered LLMs that advance the practice of active personalization in sensitive caregiving contexts. 

---
# Resisting Correction: How RLHF Makes Language Models Ignore External Safety Signals in Natural Conversation 

**Authors**: Felipe Biava Cataneo  

**Link**: [PDF](https://arxiv.org/pdf/2601.08842)  

**Abstract**: Safety architectures for language models increasingly rely on external monitors to detect errors and inject corrective signals at inference time. For such systems to function in interactive settings, models must be able to incorporate externally provided confidence information into their verbal responses. In this work, we test whether instruction-tuned language models preserve this controllability across different interaction modes.
Using Llama-3.2-3B on GSM8K, we perform a causal intervention study in which explicit external confidence signals are injected and model compliance is measured under multiple prompt strategies. We find that base models exhibit near-perfect controllability (Spearman rho close to 1.0), while instruction-tuned models display a striking context dependence: they fully comply with external corrections under explicit command prompts (bias approximately 0 percent, rho = 0.93), yet systematically ignore the same signals in natural conversational queries (bias plus 40 percent, rho = 0.04).
This behavior is not a capability failure; the model can process the signal, but an emergent property of RLHF optimization that prioritizes conversational fluency over external calibration cues in natural dialogue. We further show that internal token-level confidence in small models is uninformative (r = 0.035), underscoring the necessity of external supervision. Our findings highlight a deployment-critical failure mode: the interaction style users expect is precisely where safety corrections are least effective. 

---
# Dialogue Telemetry: Turn-Level Instrumentation for Autonomous Information Gathering 

**Authors**: Dimitris Panagopoulos, Adolfo Perrusquia, Weisi Guo  

**Link**: [PDF](https://arxiv.org/pdf/2601.09570)  

**Abstract**: Autonomous systems conducting schema-grounded information-gathering dialogues face an instrumentation gap, lacking turn-level observables for monitoring acquisition efficiency and detecting when questioning becomes unproductive. We introduce Dialogue Telemetry (DT), a measurement framework that produces two model-agnostic signals after each question-answer exchange: (i) a Progress Estimator (PE) quantifying residual information potential per category (with a bits-based variant), and (ii) a Stalling Index (SI) detecting an observable failure signature characterized by repeated category probing with semantically similar, low-marginal-gain responses. SI flags this pattern without requiring causal diagnosis, supporting monitoring in settings where attributing degradation to specific causes may be impractical. We validate DT in controlled search-and-rescue (SAR)-inspired interviews using large language model (LLM)-based simulations, distinguishing efficient from stalled dialogue traces and illustrating downstream utility by integrating DT signals into a reinforcement learning (RL) policy. Across these settings, DT provides interpretable turn-level instrumentation that improves policy performance when stalling carries operational costs. 

---
# UserLM-R1: Modeling Human Reasoning in User Language Models with Multi-Reward Reinforcement Learning 

**Authors**: Feng Zhang, Shijia Li, Chunmao Zhang, Zhanyu Ma, Jun Xu, Jiuchong Gao, Jinghua Hao, Renqing He, Jingwen Xu, Han Liu  

**Link**: [PDF](https://arxiv.org/pdf/2601.09215)  

**Abstract**: User simulators serve as the critical interactive environment for agent post-training, and an ideal user simulator generalizes across domains and proactively engages in negotiation by challenging or bargaining. However, current methods exhibit two issues. They rely on static and context-unaware profiles, necessitating extensive manual redesign for new scenarios, thus limiting generalizability. Moreover, they neglect human strategic thinking, leading to vulnerability to agent manipulation. To address these issues, we propose UserLM-R1, a novel user language model with reasoning capability. Specifically, we first construct comprehensive user profiles with both static roles and dynamic scenario-specific goals for adaptation to diverse scenarios. Then, we propose a goal-driven decision-making policy to generate high-quality rationales before producing responses, and further refine the reasoning and improve strategic capabilities with supervised fine-tuning and multi-reward reinforcement learning. Extensive experimental results demonstrate that UserLM-R1 outperforms competitive baselines, particularly on the more challenging adversarial set. 

---
