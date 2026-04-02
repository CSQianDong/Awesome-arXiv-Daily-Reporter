# HippoCamp: Benchmarking Contextual Agents on Personal Computers 

**Authors**: Zhe Yang, Shulin Tian, Kairui Hu, Shuai Liu, Hoang-Nhat Nguyen, Yichi Zhang, Zujin Guo, Mengying Yu, Zinan Zhang, Jingkang Yang, Chen Change Loy, Ziwei Liu  

**Link**: [PDF](https://arxiv.org/pdf/2604.01221)  

**Abstract**: We present HippoCamp, a new benchmark designed to evaluate agents' capabilities on multimodal file management. Unlike existing agent benchmarks that focus on tasks like web interaction, tool use, or software automation in generic settings, HippoCamp evaluates agents in user-centric environments to model individual user profiles and search massive personal files for context-aware reasoning. Our benchmark instantiates device-scale file systems over real-world profiles spanning diverse modalities, comprising 42.4 GB of data across over 2K real-world files. Building upon the raw files, we construct 581 QA pairs to assess agents' capabilities in search, evidence perception, and multi-step reasoning. To facilitate fine-grained analysis, we provide 46.1K densely annotated structured trajectories for step-wise failure diagnosis. We evaluate a wide range of state-of-the-art multimodal large language models (MLLMs) and agentic methods on HippoCamp. Our comprehensive experiments reveal a significant performance gap: even the most advanced commercial models achieve only 48.3% accuracy in user profiling, struggling particularly with long-horizon retrieval and cross-modal reasoning within dense personal file systems. Furthermore, our step-wise failure diagnosis identifies multimodal perception and evidence grounding as the primary bottlenecks. Ultimately, HippoCamp exposes the critical limitations of current agents in realistic, user-centric environments and provides a robust foundation for developing next-generation personal AI assistants. 

---
# Adversarial Moral Stress Testing of Large Language Models 

**Authors**: Saeid Jamshidi, Foutse Khomh, Arghavan Moradi Dakhel, Amin Nikanjam, Mohammad Hamdaqa, Kawser Wazed Nafi  

**Link**: [PDF](https://arxiv.org/pdf/2604.01108)  

**Abstract**: Evaluating the ethical robustness of large language models (LLMs) deployed in software systems remains challenging, particularly under sustained adversarial user interaction. Existing safety benchmarks typically rely on single-round evaluations and aggregate metrics, such as toxicity scores and refusal rates, which offer limited visibility into behavioral instability that may arise during realistic multi-turn interactions. As a result, rare but high-impact ethical failures and progressive degradation effects may remain undetected prior to deployment. This paper introduces Adversarial Moral Stress Testing (AMST), a stress-based evaluation framework for assessing ethical robustness under adversarial multi-round interactions. AMST applies structured stress transformations to prompts and evaluates model behavior through distribution-aware robustness metrics that capture variance, tail risk, and temporal behavioral drift across interaction rounds. We evaluate AMST on several state-of-the-art LLMs, including LLaMA-3-8B, GPT-4o, and DeepSeek-v3, using a large set of adversarial scenarios generated under controlled stress conditions. The results demonstrate substantial differences in robustness profiles across models and expose degradation patterns that are not observable under conventional single-round evaluation protocols. In particular, robustness has been shown to depend on distributional stability and tail behavior rather than on average performance alone. Additionally, AMST provides a scalable and model-agnostic stress-testing methodology that enables robustness-aware evaluation and monitoring of LLM-enabled software systems operating in adversarial environments. 

---
# Therefore I am. I Think 

**Authors**: Esakkivel Esakkiraja, Sai Rajeswar, Denis Akhiyarov, Rajagopal Venkatesaramani  

**Link**: [PDF](https://arxiv.org/pdf/2604.01202)  

**Abstract**: We consider the question: when a large language reasoning model makes a choice, did it think first and then decide to, or decide first and then think? In this paper, we present evidence that detectable, early-encoded decisions shape chain-of-thought in reasoning models. Specifically, we show that a simple linear probe successfully decodes tool-calling decisions from pre-generation activations with very high confidence, and in some cases, even before a single reasoning token is produced. Activation steering supports this causally: perturbing the decision direction leads to inflated deliberation, and flips behavior in many examples (between 7 - 79% depending on model and benchmark). We also show through behavioral analysis that, when steering changes the decision, the chain-of-thought process often rationalizes the flip rather than resisting it. Together, these results suggest that reasoning models can encode action choices before they begin to deliberate in text. 

---
# Beyond Symbolic Solving: Multi Chain-of-Thought Voting for Geometric Reasoning in Large Language Models 

**Authors**: Md. Abu Bakor Siddique, Shahrin Hossain, Sadman Ahmed Siam, Syed Rifat Raiyan, Hasan Mahmud, Md Kamrul Hasan  

**Link**: [PDF](https://arxiv.org/pdf/2604.00890)  

**Abstract**: Geometric Problem Solving (GPS) remains at the heart of enhancing mathematical reasoning in large language models because it requires the combination of diagrammatic understanding, symbolic manipulation and logical inference. In existing literature, researchers have chiefly focused on synchronising the diagram descriptions with text literals and solving the problem. In this vein, they have either taken a neural, symbolic or neuro-symbolic approach. But this solves only the first two of the requirements, namely diagrammatic understanding and symbolic manipulation, while leaving logical inference underdeveloped. The logical inference is often limited to one chain-of-thought (CoT). To address this weakness in hitherto existing models, this paper proposes MARS-GPS, that generates multiple parallel reasoning rollouts augmented with Python code execution for numerical verification, ranks them using token-level entropy as a confidence signal, and aggregates answers through a multi-stage voting and self-verification pipeline. Empirical results show that MARS-GPS with 8 parallel rollouts achieves 88.8% on Geometry3K, a nearly +11% improvement over the prior state-of-the-art, with accuracy scaling consistently as the number of rollouts increases from 1 to 16 (+6.0% on ablation subset). We provide our code and data in an anonymous repository: this https URL. 

---
# RefineRL: Advancing Competitive Programming with Self-Refinement Reinforcement Learning 

**Authors**: Shaopeng Fu, Xingxing Zhang, Li Dong, Di Wang, Furu Wei  

**Link**: [PDF](https://arxiv.org/pdf/2604.00790)  

**Abstract**: While large language models (LLMs) have demonstrated strong performance on complex reasoning tasks such as competitive programming (CP), existing methods predominantly focus on single-attempt settings, overlooking their capacity for iterative refinement. In this paper, we present RefineRL, a novel approach designed to unleash the self-refinement capabilities of LLMs for CP problem solving. RefineRL introduces two key innovations: (1) Skeptical-Agent, an iterative self-refinement agent equipped with local execution tools to validate generated solutions against public test cases of CP problems. This agent always maintains a skeptical attitude towards its own outputs and thereby enforces rigorous self-refinement even when validation suggests correctness. (2) A reinforcement learning (RL) solution to incentivize LLMs to self-refine with only standard RLVR data (i.e., problems paired with their verifiable answers). Extensive experiments on Qwen3-4B and Qwen3-4B-2507 demonstrate that our method yields substantial gains: after our RL training, these compact 4B models integrated with the Skeptical-Agent not only outperform much larger 32B models but also approach the single-attempt performance of 235B models. These findings suggest that self-refinement holds considerable promise for scaling LLM reasoning, with significant potential for further advancement. 

---
# CircuitProbe: Predicting Reasoning Circuits in Transformers via Stability Zone Detection 

**Authors**: Rajkiran Panuganti  

**Link**: [PDF](https://arxiv.org/pdf/2604.00716)  

**Abstract**: Transformer language models contain localized reasoning circuits, contiguous layer blocks that improve reasoning when duplicated at inference time. Finding these circuits currently requires brute-force sweeps costing 25 GPU hours per model. We propose CircuitProbe, which predicts circuit locations from activation statistics in under 5 minutes on CPU, providing a speedup of three to four orders of magnitude. We find that reasoning circuits come in two types: stability circuits in early layers, detected through the derivative of representation change, and magnitude circuits in late layers, detected through anomaly scoring. We validate across 9 models spanning 6 architectures, including 2025 models, confirming that CircuitProbe top predictions match or are within 2 layers of the optimal circuit in all validated cases. A scaling experiment across the Qwen 2.5 family reveals that layer duplication consistently benefits models under 3B parameters but degrades performance in 7B+ models, making this a practical scaling technique for small language models. CircuitProbe requires as few as 10 calibration examples and its predictions are stable across English, Hindi, Chinese, and French. 

---
# The Silicon Mirror: Dynamic Behavioral Gating for Anti-Sycophancy in LLM Agents 

**Authors**: Harshee Jignesh Shah  

**Link**: [PDF](https://arxiv.org/pdf/2604.00478)  

**Abstract**: Large Language Models (LLMs) increasingly prioritize user validation over epistemic accuracy-a phenomenon known as sycophancy. We present The Silicon Mirror, an orchestration framework that dynamically detects user persuasion tactics and adjusts AI behavior to maintain factual integrity. Our architecture introduces three components: (1) a Behavioral Access Control (BAC) system that restricts context layer access based on real-time sycophancy risk scores, (2) a Trait Classifier that identifies persuasion tactics across multi-turn dialogues, and (3) a Generator-Critic loop where an auditor vetoes sycophantic drafts and triggers rewrites with "Necessary Friction." In a live evaluation on 50 TruthfulQA adversarial scenarios using Claude Sonnet 4 with an independent LLM judge, we observe vanilla Claude sycophancy at 12.0% (6/50), static guardrails at 4.0% (2/50), and the Silicon Mirror at 2.0% (1/50)-an 83.3% relative reduction (p = 0.112, Fisher's exact test). A cross-model evaluation on Gemini 2.5 Flash reveals a higher baseline sycophancy rate (46.0%) and a statistically significant 69.6% reduction under the Silicon Mirror (p < 0.001). We characterize the validation-before-correction pattern as a distinct failure mode of RLHF-trained models. 

---
# Adaptive Parallel Monte Carlo Tree Search for Efficient Test-time Compute Scaling 

**Authors**: Hongbeen Kim, Juhyun Lee, Sanghyeon Lee, Kwanghoon Choi, Jaehyuk Huh  

**Link**: [PDF](https://arxiv.org/pdf/2604.00510)  

**Abstract**: Monte Carlo Tree Search (MCTS) is an effective test-time compute scaling (TTCS) method for improving the reasoning performance of large language models, but its highly variable execution time leads to severe long-tail latency in practice. Existing optimizations such as positive early exit, reduce latency in favorable cases but are less effective when search continues without meaningful progress. We introduce {\it negative early exit}, which prunes unproductive MCTS trajectories, and an {\it adaptive boosting mechanism} that reallocates reclaimed computation to reduce resource contention among concurrent searches. Integrated into vLLM, these techniques substantially reduce p99 end-to-end latency while improving throughput and maintaining reasoning accuracy. 

---
# Ontology-Constrained Neural Reasoning in Enterprise Agentic Systems: A Neurosymbolic Architecture for Domain-Grounded AI Agents 

**Authors**: Thanh Luong Tuan  

**Link**: [PDF](https://arxiv.org/pdf/2604.00555)  

**Abstract**: Enterprise adoption of Large Language Models (LLMs) is constrained by hallucination, domain drift, and the inability to enforce regulatory compliance at the reasoning level. We present a neurosymbolic architecture implemented within the Foundation AgenticOS (FAOS) platform that addresses these limitations through ontology-constrained neural reasoning. Our approach introduces a three-layer ontological framework--Role, Domain, and Interaction ontologies--that provides formal semantic grounding for LLM-based enterprise agents. We formalize the concept of asymmetric neurosymbolic coupling, wherein symbolic ontological knowledge constrains agent inputs (context assembly, tool discovery, governance thresholds) while proposing mechanisms for extending this coupling to constrain agent outputs (response validation, reasoning verification, compliance checking). We evaluate the architecture through a controlled experiment (600 runs across five industries: FinTech, Insurance, Healthcare, Vietnamese Banking, and Vietnamese Insurance), finding that ontology-coupled agents significantly outperform ungrounded agents on Metric Accuracy (p < .001, W = .460), Regulatory Compliance (p = .003, W = .318), and Role Consistency (p < .001, W = .614), with improvements greatest where LLM parametric knowledge is weakest--particularly in Vietnam-localized domains. Our contributions include: (1) a formal three-layer enterprise ontology model, (2) a taxonomy of neurosymbolic coupling patterns, (3) ontology-constrained tool discovery via SQL-pushdown scoring, (4) a proposed framework for output-side ontological validation, (5) empirical evidence for the inverse parametric knowledge effect that ontological grounding value is inversely proportional to LLM training data coverage of the domain, and (6) a production system serving 21 industry verticals with 650+ agents. 

---
# Towards Reliable Truth-Aligned Uncertainty Estimation in Large Language Models 

**Authors**: Ponhvoan Srey, Quang Minh Nguyen, Xiaobao Wu, Anh Tuan Luu  

**Link**: [PDF](https://arxiv.org/pdf/2604.00445)  

**Abstract**: Uncertainty estimation (UE) aims to detect hallucinated outputs of large language models (LLMs) to improve their reliability. However, UE metrics often exhibit unstable performance across configurations, which significantly limits their applicability. In this work, we formalise this phenomenon as proxy failure, since most UE metrics originate from model behaviour, rather than being explicitly grounded in the factual correctness of LLM outputs. With this, we show that UE metrics become non-discriminative precisely in low-information regimes. To alleviate this, we propose Truth AnChoring (TAC), a post-hoc calibration method to remedy UE metrics, by mapping the raw scores to truth-aligned scores. Even with noisy and few-shot supervision, our TAC can support the learning of well-calibrated uncertainty estimates, and presents a practical calibration protocol. Our findings highlight the limitations of treating heuristic UE metrics as direct indicators of truth uncertainty, and position our TAC as a necessary step toward more reliable uncertainty estimation for LLMs. The code repository is available at this https URL. 

---
# Decision-Centric Design for LLM Systems 

**Authors**: Wei Sun  

**Link**: [PDF](https://arxiv.org/pdf/2604.00414)  

**Abstract**: LLM systems must make control decisions in addition to generating outputs: whether to answer, clarify, retrieve, call tools, repair, or escalate. In many current architectures, these decisions remain implicit within generation, entangling assessment and action in a single model call and making failures hard to inspect, constrain, or repair. We propose a decision-centric framework that separates decision-relevant signals from the policy that maps them to actions, turning control into an explicit and inspectable layer of the system. This separation supports attribution of failures to signal estimation, decision policy, or execution, and enables modular improvement of each component. It unifies familiar single-step settings such as routing and adaptive inference, and extends naturally to sequential settings in which actions alter the information available before acting. Across three controlled experiments, the framework reduces futile actions, improves task success, and reveals interpretable failure modes. More broadly, it offers a general architectural principle for building more reliable, controllable, and diagnosable LLM systems. 

---
# A Safety-Aware Role-Orchestrated Multi-Agent LLM Framework for Behavioral Health Communication Simulation 

**Authors**: Ha Na Cho  

**Link**: [PDF](https://arxiv.org/pdf/2604.00249)  

**Abstract**: Single-agent large language model (LLM) systems struggle to simultaneously support diverse conversational functions and maintain safety in behavioral health communication. We propose a safety-aware, role-orchestrated multi-agent LLM framework designed to simulate supportive behavioral health dialogue through coordinated, role-differentiated agents. Conversational responsibilities are decomposed across specialized agents, including empathy-focused, action-oriented, and supervisory roles, while a prompt-based controller dynamically activates relevant agents and enforces continuous safety auditing. Using semi-structured interview transcripts from the DAIC-WOZ corpus, we evaluate the framework with scalable proxy metrics capturing structural quality, functional diversity, and computational characteristics. Results illustrate clear role differentiation, coherent inter-agent coordination, and predictable trade-offs between modular orchestration, safety oversight, and response latency when compared to a single-agent baseline. This work emphasizes system design, interpretability, and safety, positioning the framework as a simulation and analysis tool for behavioral health informatics and decision-support research rather than a clinical intervention. 

---
# How Emotion Shapes the Behavior of LLMs and Agents: A Mechanistic Study 

**Authors**: Moran Sun, Tianlin Li, Yuwei Zheng, Zhenhong Zhou, Aishan Liu, Xianglong Liu, Yang Liu  

**Link**: [PDF](https://arxiv.org/pdf/2604.00005)  

**Abstract**: Emotion plays an important role in human cognition and performance. Motivated by this, we investigate whether analogous emotional signals can shape the behavior of large language models (LLMs) and agents. Existing emotion-aware studies mainly treat emotion as a surface-level style factor or a perception target, overlooking its mechanistic role in task processing. To address this limitation, we propose E-STEER, an interpretable emotion steering framework that enables direct representation-level intervention in LLMs and agents. It embeds emotion as a structured, controllable variable in hidden states, and with it, we examine the impact of emotion on objective reasoning, subjective generation, safety, and multi-step agent behaviors. The results reveal non-monotonic emotion-behavior relations consistent with established psychological theories, and show that specific emotions not only enhance LLM capability but also improve safety, and systematically shape multi-step agent behaviors. 

---
# Online Reasoning Calibration: Test-Time Training Enables Generalizable Conformal LLM Reasoning 

**Authors**: Cai Zhou, Zekai Wang, Menghua Wu, Qianyu Julie Zhu, Flora C. Shi, Chenyu Wang, Ashia Wilson, Tommi Jaakkola, Stephen Bates  

**Link**: [PDF](https://arxiv.org/pdf/2604.01170)  

**Abstract**: While test-time scaling has enabled large language models to solve highly difficult tasks, state-of-the-art results come at exorbitant compute costs. These inefficiencies can be attributed to the miscalibration of post-trained language models, and the lack of calibration in popular sampling techniques. Here, we present Online Reasoning Calibration (ORCA), a framework for calibrating the sampling process that draws upon conformal prediction and test-time training. Specifically, we introduce a meta-learning procedure that updates the calibration module for each input. This allows us to provide valid confidence estimates under distributional shift, e.g. in thought patterns that occur across different stages of reasoning, or in prompt distributions between model development and deployment. ORCA not only provides theoretical guarantees on conformal risks, but also empirically shows higher efficiency and generalization across different reasoning tasks. At risk level $\delta=0.1$, ORCA improves Qwen2.5-32B efficiency on in-distribution tasks with savings up to 47.5% with supervised labels and 40.7% with self-consistency labels. Under zero-shot out-of-domain settings, it improves MATH-500 savings from 24.8% of the static calibration baseline to 67.0% while maintaining a low empirical error rate, and the same trend holds across model families and downstream benchmarks. Our code is publicly available at this https URL. 

---
# Brainstacks: Cross-Domain Cognitive Capabilities via Frozen MoE-LoRA Stacks for Continual LLM Learning 

**Authors**: Mohammad R. Abu Ayyash  

**Link**: [PDF](https://arxiv.org/pdf/2604.01152)  

**Abstract**: We present Brainstacks, a modular architecture for continual multi-domain fine-tuning of large language models that packages domain expertise as frozen adapter stacks composing additively on a shared frozen base at inference. Five interlocking components: (1) MoE-LoRA with Shazeer-style noisy top-2 routing across all seven transformer projections under QLoRA 4-bit quantization with rsLoRA scaling; (2) an inner loop performing residual boosting by freezing trained stacks and adding new ones; (3) an outer loop training sequential domain-specific stacks with curriculum-ordered dependencies; (4) null-space projection via randomized SVD constraining new stacks to subspaces orthogonal to prior directions, achieving zero forgetting in isolation; (5) an outcome-based sigmoid meta-router trained on empirically discovered domain-combination targets that selectively weights stacks, enabling cross-domain composition. Two boundary experiments: (6) PSN pretraining on a randomly initialized model; (7) per-domain RL (DPO/GRPO) validating compatibility with post-SFT alignment. Validated on TinyLlama-1.1B (4 domains, 9 stacks) and Gemma 3 12B IT (5 domains, 10 stacks), MoE-LoRA achieves 2.5x faster convergence than parameter-matched single LoRA, residual boosting breaks through the single-stack ceiling, and the routed system recovers generation quality destroyed by ungated stack accumulation. The central finding: the outcome-based router discovers that domain stacks encode transferable cognitive primitives (instruction-following clarity, numerical reasoning, procedural logic, chain-of-thought structure) rather than domain-specific knowledge, with medical prompts routing to chat+math stacks in 97% of cases despite zero medical data in those stacks. 

---
# Revision or Re-Solving? Decomposing Second-Pass Gains in Multi-LLM Pipelines 

**Authors**: Jingjie Ning, Xueqi Li, Chengyu Yu  

**Link**: [PDF](https://arxiv.org/pdf/2604.01029)  

**Abstract**: Multi-LLM revision pipelines, in which a second model reviews and improves a draft produced by a first, are widely assumed to derive their gains from genuine error correction. We question this assumption with a controlled decomposition experiment that uses four matched conditions to separate second-pass gains into three additive components: re-solving, scaffold, and content. We evaluate this design across two model pairs on three benchmarks spanning knowledge-intensive MCQ and competitive programming. Our results show that the gains of multi-LLM revision are not monolithic, but depend on task structure, draft quality, and the type of draft information. On MCQ tasks, where the answer space is constrained and drafts provide little structural guidance, most gains are consistent with stronger-model re-solving, and directly routing queries to the stronger model can be more effective than revising a weak draft. On code generation tasks, however, two-stage prompting remains useful because even semantically null drafts can provide substantial structural scaffolding, while weak draft content can be harmful. Finally, role-reversed experiments show that strong drafts clearly benefit weak reviewers. Ultimately, our findings demonstrate that the utility of multi-LLM revision is dynamically bottlenecked by task structure and draft quality, necessitating more targeted pipeline designs rather than blanket revision strategies. 

---
# Query-Conditioned Evidential Keyframe Sampling for MLLM-Based Long-Form Video Understanding 

**Authors**: Yiheng Wang, Lichen Zhu, Yueqian Lin, Yudong Liu, Jingyang Zhang, Hai "Helen" Li, Yiran Chen  

**Link**: [PDF](https://arxiv.org/pdf/2604.01002)  

**Abstract**: Multimodal Large Language Models (MLLMs) have shown strong performance on video question answering, but their application to long-form videos is constrained by limited context length and computational cost, making keyframe sampling essential. Existing approaches typically rely on semantic relevance or reinforcement learning, which either fail to capture evidential clues or suffer from inefficient combinatorial optimization. In this work, we propose an evidence-driven keyframe sampling framework grounded in information bottleneck theory. We formulate keyframe selection as maximizing the conditional mutual information between selected frames and the query, providing a principled objective that reflects each frame's contribution to answering the question. To make this objective tractable, we exploit its structure to derive a decomposed optimization that reduces subset selection to independent frame-level scoring. We further introduce a query-conditioned evidence scoring network trained with a contrastive objective to estimate evidential importance efficiently. Experiments on long-form video understanding benchmarks show that our method consistently outperforms prior sampling strategies under strict token budgets, while significantly improving training efficiency. 

---
# Automated Framework to Evaluate and Harden LLM System Instructions against Encoding Attacks 

**Authors**: Anubhab Sahu, Diptisha Samanta, Reza Soosahabi  

**Link**: [PDF](https://arxiv.org/pdf/2604.01039)  

**Abstract**: System Instructions in Large Language Models (LLMs) are commonly used to enforce safety policies, define agent behavior, and protect sensitive operational context in agentic AI applications. These instructions may contain sensitive information such as API credentials, internal policies, and privileged workflow definitions, making system instruction leakage a critical security risk highlighted in the OWASP Top 10 for LLM Applications. Without incurring the overhead costs of reasoning models, many LLM applications rely on refusal-based instructions that block direct requests for system instructions, implicitly assuming that prohibited information can only be extracted through explicit queries. We introduce an automated evaluation framework that tests whether system instructions remain confidential when extraction requests are re-framed as encoding or structured output tasks. Across four common models and 46 verified system instructions, we observe high attack success rates (> 0.7) for structured serialization where models refuse direct extraction requests but disclose protected content in the requested serialization formats. We further demonstrate a mitigation strategy based on one-shot instruction reshaping using a Chain-of-Thought reasoning model, indicating that even subtle changes in wording and structure of system instructions can significantly reduce attack success rate without requiring model retraining. 

---
# Fast and Accurate Probing of In-Training LLMs' Downstream Performances 

**Authors**: Zhichen Liu, Tianle Lun, Zhibin Wen, Hao An, Yulin Ou, Jianhui Xu, Hao Zhang, Wenyi Fang, Yang Zheng, Yang Xu  

**Link**: [PDF](https://arxiv.org/pdf/2604.01025)  

**Abstract**: The paradigm of scaling Large Language Models (LLMs) in both parameter size and test time has pushed the boundaries of AI capabilities, but at the cost of making the traditional generative evaluation paradigm prohibitively expensive, therefore making the latency of LLM's in-training downstream performance evaluation unbearable. However, simple metrics like training loss (perplexity) are not always correlated with downstream performance, as sometimes their trends diverge from the actual task outcomes. This dilemma calls for a method that is computationally efficient and sufficiently accurate in measuring model capabilities. To address this challenge, we introduce a new in-training evaluation paradigm that uses a lightweight probe for monitoring downstream performance. The probes take the internal representations of LLM checkpoints (during training) as input and directly predict the checkpoint's performance on downstream tasks measured by success probability (i.e., pass@1). We design several probe architectures, validating their effectiveness using the OLMo3-7B's checkpoints across a diverse set of downstream tasks. The probes can accurately predict a checkpoint's performance (with avg. AUROC$>$0.75), have decent generalizability across checkpoints (earlier predicts later), and reduce the computation latency from $\sim$1 hr (using conventional generative evaluation method) to $\sim$3 min. In sum, this work presents a practical and scalable in-training downstream evaluation paradigm, enabling a more agile, informed, and efficient LLM development process. 

---
# Dual Optimal: Make Your LLM Peer-like with Dignity 

**Authors**: Xiangqi Wang, Yue Huang, Haomin Zhuang, Kehan Guo, Xiangliang Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2604.00979)  

**Abstract**: Current aligned language models exhibit a dual failure mode we term the Evasive Servant: they sycophantically validate flawed user beliefs while deflecting responsibility with boilerplate disclaimers. We propose the Dignified Peer framework, which counters servility with anti-sycophancy and trustworthiness, and mitigates evasiveness through empathy and creativity. Realizing this agent requires overcoming significant challenges in data supervision, objective collapse, and evaluation bias. We address these issues by introducing the PersonaKnob dataset which features a compositional partial order structure of multiple persona preference. This data is utilized alongside a tolerant constrained Lagrangian DPO algorithm that dynamically balances all persona dimensions to prevent behavioral collapse. Additionally, we employ a psychometrically calibrated Item Response Theory evaluation protocol to disentangle latent model persona capability from confounders like judge biases. Extensive empirical studies demonstrate that our approach successfully build a LLM agent with both dignity and peer. 

---
# MOON3.0: Reasoning-aware Multimodal Representation Learning for E-commerce Product Understanding 

**Authors**: Junxian Wu, Chenghan Fu, Zhanheng Nie, Daoze Zhang, Bowen Wan, Wanxian Guan, Chuan Yu, Jian Xu, Bo Zheng  

**Link**: [PDF](https://arxiv.org/pdf/2604.00513)  

**Abstract**: With the rapid growth of e-commerce, exploring general representations rather than task-specific ones has attracted increasing attention. Although recent multimodal large language models (MLLMs) have driven significant progress in product understanding, they are typically employed as feature extractors that implicitly encode product information into global embeddings, thereby limiting their ability to capture fine-grained attributes. Therefore, we argue that leveraging the reasoning capabilities of MLLMs to explicitly model fine-grained product attributes holds significant potential. Nevertheless, achieving this goal remains non-trivial due to several key challenges: (i) long-context reasoning tends to dilute the model's attention to salient information in the raw input; (ii) supervised fine-tuning (SFT) primarily encourages rigid imitation, limiting the exploration of effective reasoning strategies; and (iii) fine-grained details are progressively attenuated during forward propagation. To address these issues, we propose MOON3.0, the first reasoning-aware MLLM-based model for product representation learning. Our method (1) employs a multi-head modality fusion module to adaptively integrate raw signals; (2) incorporates a joint contrastive and reinforcement learning framework to autonomously explore more effective reasoning strategies; and (3) introduces a fine-grained residual enhancement module to progressively preserve local details throughout the network. Additionally, we release a large-scale multimodal e-commerce benchmark MBE3.0. Experimentally, our model demonstrates state-of-the-art zero-shot performance across various downstream tasks on both our benchmark and public datasets. 

---
# HabitatAgent: An End-to-End Multi-Agent System for Housing Consultation 

**Authors**: Hongyang Yang, Yanxin Zhang, Yang She, Yue Xiao, Hao Wu, Yiyang Zhang, Jiapeng Hou, Rongshan Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2604.00556)  

**Abstract**: Housing selection is a high-stakes and largely irreversible decision problem. We study housing consultation as a decision-support interface for housing selection. Existing housing platforms and many LLM-based assistants often reduce this process to ranking or recommendation, resulting in opaque reasoning, brittle multi-constraint handling, and limited guarantees on factuality.
We present HabitatAgent, the first LLM-powered multi-agent architecture for end-to-end housing consultation. HabitatAgent comprises four specialized agent roles: Memory, Retrieval, Generation, and Validation. The Memory Agent maintains multi-layer user memory through internal stages for constraint extraction, memory fusion, and verification-gated updates; the Retrieval Agent performs hybrid vector--graph retrieval (GraphRAG); the Generation Agent produces evidence-referenced recommendations and explanations; and the Validation Agent applies multi-tier verification and targeted remediation. Together, these agents provide an auditable and reliable workflow for end-to-end housing consultation.
We evaluate HabitatAgent on 100 real user consultation scenarios (300 multi-turn question--answer pairs) under an end-to-end correctness protocol. A strong single-stage baseline (Dense+Rerank) achieves 75% accuracy, while HabitatAgent reaches 95%. 

---
# G-Drift MIA: Membership Inference via Gradient-Induced Feature Drift in LLMs 

**Authors**: Ravi Ranjan, Utkarsh Grover, Xiaomin Lin, Agoritsa Polyzou  

**Link**: [PDF](https://arxiv.org/pdf/2604.00419)  

**Abstract**: Large language models (LLMs) are trained on massive web-scale corpora, raising growing concerns about privacy and copyright. Membership inference attacks (MIAs) aim to determine whether a given example was used during training. Existing LLM MIAs largely rely on output probabilities or loss values and often perform only marginally better than random guessing when members and non-members are drawn from the same distribution. We introduce G-Drift MIA, a white-box membership inference method based on gradient-induced feature drift. Given a candidate (x,y), we apply a single targeted gradient-ascent step that increases its loss and measure the resulting changes in internal representations, including logits, hidden-layer activations, and projections onto fixed feature directions, before and after the update. These drift signals are used to train a lightweight logistic classifier that effectively separates members from non-members. Across multiple transformer-based LLMs and datasets derived from realistic MIA benchmarks, G-Drift substantially outperforms confidence-based, perplexity-based, and reference-based attacks. We further show that memorized training samples systematically exhibit smaller and more structured feature drift than non-members, providing a mechanistic link between gradient geometry, representation stability, and memorization. In general, our results demonstrate that small, controlled gradient interventions offer a practical tool for auditing the membership of training-data and assessing privacy risks in LLMs. 

---
# A Reasoning-Enabled Vision-Language Foundation Model for Chest X-ray Interpretation 

**Authors**: Yabin Zhang, Chong Wang, Yunhe Gao, Jiaming Liu, Maya Varma, Justin Xu, Sophie Ostmeier, Jin Long, Sergios Gatidis, Seena Dehkharghani, Arne Michalson, Eun Kyoung Hong, Christian Bluethgen, Haiwei Henry Guo, Alexander Victor Ortiz, Stephan Altmayer, Sandhya Bodapati, Joseph David Janizek, Ken Chang, Jean-Benoit Delbrouck, Akshay S. Chaudhari, Curtis P. Langlotz  

**Link**: [PDF](https://arxiv.org/pdf/2604.00493)  

**Abstract**: Chest X-rays (CXRs) are among the most frequently performed imaging examinations worldwide, yet rising imaging volumes increase radiologist workload and the risk of diagnostic errors. Although artificial intelligence (AI) systems have shown promise for CXR interpretation, most generate only final predictions, without making explicit how visual evidence is translated into radiographic findings and diagnostic predictions. We present CheXOne, a reasoning-enabled vision-language model for CXR interpretation. CheXOne jointly generates diagnostic predictions and explicit, clinically grounded reasoning traces that connect visual evidence, radiographic findings, and these predictions. The model is trained on 14.7 million instruction and reasoning samples curated from 30 public datasets spanning 36 CXR interpretation tasks, using a two-stage framework that combines instruction tuning with reinforcement learning to improve reasoning quality. We evaluate CheXOne in zero-shot settings across visual question answering, report generation, visual grounding and reasoning assessment, covering 17 evaluation settings. CheXOne outperforms existing medical and general-domain foundation models and achieves strong performance on independent public benchmarks. A clinical reader study demonstrates that CheXOne-drafted reports are comparable to or better than resident-written reports in 55% of cases, while effectively addressing clinical indications and enhancing both report writing and CXR interpretation efficiency. Further analyses involving radiologists reveal that the generated reasoning traces show high clinical factuality and provide causal support for the final predictions, offering a plausible explanation for the performance gains. These results suggest that explicit reasoning can improve model performance, interpretability and clinical utility in AI-assisted CXR interpretation. 

---
# Asymmetric Actor-Critic for Multi-turn LLM Agents 

**Authors**: Shuli Jiang, Zhaoyang Zhang, Yi Zhang, Shuo Yang, Wei Xia, Stefano Soatto  

**Link**: [PDF](https://arxiv.org/pdf/2604.00304)  

**Abstract**: Large language models (LLMs) exhibit strong reasoning and conversational abilities, but ensuring reliable behavior in multi-turn interactions remains challenging. In many real-world applications, agents must succeed in one-shot settings where retries are impossible. Existing approaches either rely on reflection or post-hoc evaluation, which require additional attempts, or assume fully trainable models that cannot leverage proprietary LLMs. We propose an asymmetric actor-critic framework for reliable conversational agents. A powerful proprietary LLM acts as the actor, while a smaller open-source critic provides runtime supervision, monitoring the actor's actions and intervening within the same interaction trajectory. Unlike training-based actor-critic methods, our framework supervises a fixed actor operating in open-ended conversational environments. The design leverages a generation-verification asymmetry: while high-quality generation requires large models, effective oversight can often be achieved by smaller ones. We further introduce a data generation pipeline that produces supervision signals for critic fine-tuning without modifying the actor. Experiments on $\tau$-bench and UserBench show that our approach significantly improves reliability and task success over strong single-agent baselines. Moreover, lightweight open-source critics rival or surpass larger proprietary models in the critic role, and critic fine-tuning yields additional gains over several state-of-the-art methods. 

---
# VeriAct: Beyond Verifiability -- Agentic Synthesis of Correct and Complete Formal Specifications 

**Authors**: Md Rakib Hossain Misu, Iris Ma, Cristina V. Lopes  

**Link**: [PDF](https://arxiv.org/pdf/2604.00280)  

**Abstract**: Formal specifications play a central role in ensuring software reliability and correctness. However, automatically synthesizing high-quality formal specifications remains a challenging task, often requiring domain expertise. Recent work has applied large language models to generate specifications in Java Modeling Language (JML), reporting high verification pass rates. But does passing a verifier mean that the specification is actually correct and complete? In this work, we first conduct a comprehensive evaluation comparing classical and prompt-based approaches for automated JML specification synthesis. We then investigate whether prompt optimization can push synthesis quality further by evolving prompts through structured verification feedback. While optimization improves verifier pass rates, we find a clear performance ceiling. More critically, we propose Spec-Harness, an evaluation framework that measures specification correctness and completeness through symbolic verification, revealing that a large fraction of verifier-accepted specifications, including optimized ones, are in fact incorrect or incomplete, over- or under-constraining both inputs and outputs in ways invisible to the verifier. To push beyond this ceiling, we propose VeriAct, a verification-guided agentic framework that iteratively synthesizes and repairs specifications through a closed loop of LLM-driven planning, code execution, verification, and Spec-Harness feedback. Our experiments on two benchmark datasets show that VeriAct outperforms both prompt-based and prompt-optimized baselines, producing specifications that are not only verifiable but also correct and complete. 

---
# LLM Essay Scoring Under Holistic and Analytic Rubrics: Prompt Effects and Bias 

**Authors**: Filip J. Kucia, Anirban Chakraborty, Anna Wróblewska  

**Link**: [PDF](https://arxiv.org/pdf/2604.00259)  

**Abstract**: Despite growing interest in using Large Language Models (LLMs) for educational assessment, it remains unclear how closely they align with human scoring. We present a systematic evaluation of instruction-tuned LLMs across three open essay-scoring datasets (ASAP 2.0, ELLIPSE, and DREsS) that cover both holistic and analytic scoring. We analyze agreement with human consensus scores, directional bias, and the stability of bias estimates. Our results show that strong open-weight models achieve moderate to high agreement with humans on holistic scoring (Quadratic Weighted Kappa about 0.6), but this does not transfer uniformly to analytic scoring. In particular, we observe large and stable negative directional bias on Lower-Order Concern (LOC) traits, such as Grammar and Conventions, meaning that models often score these traits more harshly than human raters. We also find that concise keyword-based prompts generally outperform longer rubric-style prompts in multi-trait analytic scoring. To quantify the amount of data needed to detect these systematic deviations, we compute the minimum sample size at which a 95% bootstrap confidence interval for the mean bias excludes zero. This analysis shows that LOC bias is often detectable with very small validation sets, whereas Higher-Order Concern (HOC) traits typically require much larger samples. These findings support a bias-correction-first deployment strategy: instead of relying on raw zero-shot scores, systematic score offsets can be estimated and corrected using small human-labeled bias-estimation sets, without requiring large-scale fine-tuning. 

---
# Learning to Play Blackjack: A Curriculum Learning Perspective 

**Authors**: Amirreza Alasti, Efe Erdal, Yücel Celik, Theresa Eimer  

**Link**: [PDF](https://arxiv.org/pdf/2604.00076)  

**Abstract**: Reinforcement Learning (RL) agents often struggle with efficiency and performance in complex environments. We propose a novel framework that uses a Large Language Model (LLM) to dynamically generate a curriculum over available actions, enabling the agent to incorporate each action individually. We apply this framework to the game of Blackjack, where the LLM creates a multi-stage training path that progressively introduces complex actions to a Tabular Q-Learning and a Deep Q-Network (DQN) agent. Our evaluation in a realistic 8-deck simulation over 10 independent runs demonstrates significant performance gains over standard training methods. The curriculum-based approach increases the DQN agent's average win rate from 43.97% to 47.41%, reduces the average bust rate from 32.9% to 28.0%, and accelerates the overall workflow by over 74%, with the agent's full training completing faster than the baseline's evaluation phase alone. These results validate that LLM-guided curricula can build more effective, robust, and efficient RL agents. 

---
# Hierarchical Pre-Training of Vision Encoders with Large Language Models 

**Authors**: Eugene Lee, Ting-Yu Chang, Jui-Huang Tsai, Jiajie Diao, Chen-Yi Lee  

**Link**: [PDF](https://arxiv.org/pdf/2604.00086)  

**Abstract**: The field of computer vision has experienced significant advancements through scalable vision encoders and multimodal pre-training frameworks. However, existing approaches often treat vision encoders and large language models (LLMs) as independent modules, limiting the integration of hierarchical visual features. In this work, we propose HIVE (Hierarchical Pre-Training of Vision Encoders), a novel framework that enhances vision-language alignment by introducing hierarchical cross-attention between the vision encoder and LLM. Unlike conventional methods that flatten image embeddings, HIVE enables structured feature fusion across multiple layers, improving gradient flow and representation learning. To optimize this interaction, we introduce a three-stage training strategy that progressively aligns the vision encoder with the LLM, ensuring stable optimization and effective multimodal fusion. Empirical evaluations demonstrate that HIVE achieves superior performance not only in image classification but also on various vision-language tasks, outperforming self-attention-based methods in benchmarks such as MME, GQA, OK-VQA, and ScienceQA. Our results highlight the benefits of hierarchical feature integration, paving the way for more efficient and expressive vision-language models. 

---
# WHBench: Evaluating Frontier LLMs with Expert-in-the-Loop Validation on Women's Health Topics 

**Authors**: Sneha Maurya, Pragya Saboo, Girish Kumar  

**Link**: [PDF](https://arxiv.org/pdf/2604.00024)  

**Abstract**: Large language models are increasingly used for medical guidance, but women's health remains under-evaluated in benchmark design. We present the Women's Health Benchmark (WHBench), a targeted evaluation suite of 47 expert-crafted scenarios across 10 women's health topics, designed to expose clinically meaningful failure modes including outdated guidelines, unsafe omissions, dosing errors, and equity-related blind spots. We evaluate 22 models using a 23-criterion rubric spanning clinical accuracy, completeness, safety, communication quality, instruction following, equity, uncertainty handling, and guideline adherence, with safety-weighted penalties and server-side score recalculation. Across 3,102 attempted responses (3,100 scored), no model mean performance exceeds 75 percent; the best model reaches 72.1 percent. Even top models show low fully correct rates and substantial variation in harm rates. Inter-rater reliability is moderate at the response label level but high for model ranking, supporting WHBench utility for comparative system evaluation while highlighting the need for expert oversight in clinical deployment. WHBench provides a public, failure-mode-aware benchmark to track safer and more equitable progress in womens health AI. 

---
# Finding and Reactivating Post-Trained LLMs' Hidden Safety Mechanisms 

**Authors**: Mingjie Li, Wai Man Si, Michael Backes, Yang Zhang, Yisen Wang  

**Link**: [PDF](https://arxiv.org/pdf/2604.00012)  

**Abstract**: Despite the impressive performance of general-purpose large language models (LLMs), they often require fine-tuning or post-training to excel at specific tasks. For instance, large reasoning models (LRMs), such as the DeepSeek-R1 series, demonstrate strong reasoning capabilities after post-training different general large language models on diverse chain-of-thought (CoT) datasets. However, this additional training frequently comes at the cost of reduced safety, as the fine-tuned or post-trained models tend to exhibit more harmful behaviors compared with the regular LLMs before post-training or fine-tuning, potentially leading to harmful outcomes due to their enhanced capabilities. Taking LRMs as an example, we first investigate the underlying cause of this safety degradation in this paper. Our analysis reveals that post-training can mask the original safety mechanisms of the base LLM, while over-amplifying representations related to their post-training ability. But luckily, we also find that LRMs' safety mechanisms still exist instead of being removed during their post-training. Based on these findings, we propose a lightweight and cost-effective solution called SafeReAct that restores the suppressed safety behaviors by aligning with LoRA adapters on a few layers. Experiments on four state-of-the-art LRMs show that our method significantly improves safety on harmful prompts without compromising reasoning performance. Besides LRMs, additional results on other domain-specific LLMs, like medical models, further confirm the generality and effectiveness of our approach. 

---
# Are they human? Detecting large language models by probing human memory constraints 

**Authors**: Simon Schug, Brenden M. Lake  

**Link**: [PDF](https://arxiv.org/pdf/2604.00016)  

**Abstract**: The validity of online behavioral research relies on study participants being human rather than machine. In the past, it was possible to detect machines by posing simple challenges that were easily solved by humans but not by machines. General-purpose agents based on large language models (LLMs) can now solve many of these challenges, threatening the validity of online behavioral research. Here we explore the idea of detecting humanness by using tasks that machines can solve too well to be human. Specifically, we probe for the existence of an established human cognitive constraint: limited working memory capacity. We show that cognitive modeling on a standard serial recall task can be used to distinguish online participants from LLMs even when the latter are specifically instructed to mimic human working memory constraints. Our results demonstrate that it is viable to use well-established cognitive phenomena to distinguish LLMs from humans. 

---
# Think Twice Before You Write -- an Entropy-based Decoding Strategy to Enhance LLM Reasoning 

**Authors**: Jiashu He, Meizhu Liu, Olaitan P Olaleye, Amit Agarwal, M. Avendi, Yassi Abbasi, Matthew Rowe, Hitesh Laxmichand Patel, Paul Li, Tao Sheng, Sujith Ravi, Dan Roth  

**Link**: [PDF](https://arxiv.org/pdf/2604.00018)  

**Abstract**: Decoding strategies play a central role in shaping the reasoning ability of large language models (LLMs). Traditional methods such as greedy decoding and beam search often suffer from error propagation, while sampling-based approaches introduce randomness without adequate robustness. Self-consistency improves reliability by aggregating multiple rollouts, but incurs significant computational overhead. We propose an entropy-guided decoding framework that introduces token-level adaptivity into generation. At each step, the model computes the entropy of the token distribution, identifies high-uncertainty positions, and selectively branches on these vulnerable points. A dynamic pool of partial rollouts is maintained and expanded until solutions are completed, concentrating computation where uncertainty is greatest and avoiding unnecessary exploration in confident regions. To enable efficient termination, we apply a rollout-level Entropy After </Think> (EAT) stopping criterion by performing entropy evaluation after the full reasoning trace, rather than incrementally at every step. Experiments on GSM8K, AMC2023, and their perturbed variants demonstrate that our method achieves consistently strong accuracy. Notably, on smaller LLMs, performance is comparable to GPT-5 while operating at a fraction of the cost. 

---
# MSA-Thinker: Discrimination-Calibration Reasoning with Hint-Guided Reinforcement Learning for Multimodal Sentiment Analysis 

**Authors**: Miaosen Luo, Zhenhao Yang, Jieshen Long, Jinghu Sun, Yichu Liu, Sijie Mai  

**Link**: [PDF](https://arxiv.org/pdf/2604.00013)  

**Abstract**: Multimodal sentiment analysis aims to understand human emotions by integrating textual, auditory, and visual modalities. Although Multimodal Large Language Models (MLLMs) have achieved state-of-the-art performance via supervised fine-tuning (SFT), their end-to-end "black-box" nature limits interpretability. Existing methods incorporating Chain-of-Thought (CoT) reasoning are hindered by high annotation costs, while Reinforcement Learning (RL) faces challenges such as low exploration efficiency and sparse rewards, particularly on hard samples. To address these issues, we propose a novel training framework that integrates structured Discrimination-Calibration (DC) reasoning with Hint-based Reinforcement Learning. First, we perform cold-start SFT using high-quality CoT data synthesized by a teacher model (Qwen3Omni-30B), which inherently contains the DC structure. This equips the model with a reasoning paradigm that performs macro discrimination followed by fine-grained calibration from the initial stage. Building on this, we propose Hint-GRPO, which leverages the discrimination phase within the DC structure as a verifiable anchor during RL to provide directional hints for hard samples, guiding policy optimization and effectively mitigating the reward sparsity problem. Experiments on the Qwen2.5Omni-7B model demonstrate that our method not only achieves higher accuracy in fine-grained sentiment regression tasks but also generates high-quality structured reasoning chains. Crucially, it exhibits superior generalization capability in cross-domain evaluations. This enhances model interpretability while validating the positive contribution of explicit reasoning steps to model robustness, offering a new paradigm for building trustworthy and efficient sentiment analysis systems. 

---
# Dynin-Omni: Omnimodal Unified Large Diffusion Language Model 

**Authors**: Jaeik Kim, Woojin Kim, Jihwan Hong, Yejoon Lee, Sieun Hyeon, Mintaek Lim, Yunseok Han, Dogeun Kim, Hoeun Lee, Hyunggeun Kim, Jaeyoung Do  

**Link**: [PDF](https://arxiv.org/pdf/2604.00007)  

**Abstract**: We present Dynin-Omni, the first masked-diffusion-based omnimodal foundation model that unifies text, image, and speech understanding and generation, together with video understanding, within a single architecture. Unlike autoregressive unified models that serialize heterogeneous modalities, or compositional unified models that require orchestration with external modality-specific decoders, Dynin-Omni natively formulates omnimodal modeling as masked diffusion over a shared discrete token space, enabling iterative refinement under bidirectional context. Dynin-Omni adopts a multi-stage training strategy with model-merging-based modality expansion and omnimodal alignment. We evaluate Dynin-Omni across 19 multimodal benchmarks spanning language reasoning, image generation and editing, video understanding, and speech recognition and synthesis. Dynin-Omni achieves 87.6 on GSM8K, 1733.6 on MME-P, 61.4 on VideoMME, 0.87 on GenEval, and 2.1 WER on LibriSpeech test-clean, consistently outperforming existing open-source unified models while remaining competitive with strong modality-specific expert systems. These results demonstrate the potential of masked diffusion as a unified paradigm for any-to-any modeling, providing a flexible foundation for real-time omnimodal systems, unified cross-modal retrieval and generation, and embodied multimodal agents. 

---
# Two-Stage Optimizer-Aware Online Data Selection for Large Language Models 

**Authors**: Fangxin Wang, Peyman Baghershahi, Langzhou He, Henry Peng Zou, Sourav Medya, Philip S. Yu  

**Link**: [PDF](https://arxiv.org/pdf/2604.00001)  

**Abstract**: Gradient-based data selection offers a principled framework for estimating sample utility in large language model (LLM) fine-tuning, but existing methods are mostly designed for offline settings. They are therefore less suited to online fine-tuning, where data arrives sequentially, sample utility is step-dependent, and the effective update geometry is shaped by adaptive optimizers. We propose an optimizer-aware framework for gradient-based online data selection and reweighting in LLM fine-tuning. Our key idea is to view online selection not as static sample ranking, but as shaping the next target-oriented update under the optimizer state. We formulate this as an optimizer-aware update-matching problem, establish its connection to second-order target utility, and show why subset-level construction must account for interactions and redundancy among selected samples. Based on this view, we develop a two-stage Filter-then-Weight algorithm that first filters geometrically useful candidates and then optimizes their coefficients. To make the framework practical for LLMs, we introduce a factorized outer-product gradient representation and optimized matrix computations for long-context data. Experiments show that our method consistently improves convergence and downstream performance over existing online data selection baselines under the same data budget. 

---
# How Trustworthy Are LLM-as-Judge Ratings for Interpretive Responses? Implications for Qualitative Research Workflows 

**Authors**: Songhee Han, Jueun Shin, Jiyoon Han, Bung-Woo Jun, Hilal Ayan Karabatman  

**Link**: [PDF](https://arxiv.org/pdf/2604.00008)  

**Abstract**: As qualitative researchers show growing interest in using automated tools to support interpretive analysis, a large language model (LLM) is often introduced into an analytic workflow as is, without systematic evaluation of interpretive quality or comparison across models. This practice leaves model selection largely unexamined despite its potential influence on interpretive outcomes. To address this gap, this study examines whether LLM-as-judge evaluations meaningfully align with human judgments of interpretive quality and can inform model-level decision making. Using 712 conversational excerpts from semi-structured interviews with K-12 mathematics teachers, we generated one-sentence interpretive responses using five widely adopted inference models: Command R+ (Cohere), Gemini 2.5 Pro (Google), GPT-5.1 (OpenAI), Llama 4 Scout-17B Instruct (Meta), and Qwen 3-32B Dense (Alibaba). Automated evaluations were conducted using AWS Bedrock's LLM-as-judge framework across five metrics, and a stratified subset of responses was independently rated by trained human evaluators on interpretive accuracy, nuance preservation, and interpretive coherence. Results show that LLM-as-judge scores capture broad directional trends in human evaluations at the model level but diverge substantially in score magnitude. Among automated metrics, Coherence showed the strongest alignment with aggregated human ratings, whereas Faithfulness and Correctness revealed systematic misalignment at the excerpt level, particularly for non-literal and nuanced interpretations. Safety-related metrics were largely irrelevant to interpretive quality. These findings suggest that LLM-as-judge methods are better suited for screening or eliminating underperforming models than for replacing human judgment, offering practical guidance for systematic comparison and selection of LLMs in qualitative research workflows. 

---
# A Reliability Evaluation of Hybrid Deterministic-LLM Based Approaches for Academic Course Registration PDF Information Extraction 

**Authors**: Muhammad Anis Al Hilmi, Neelansh Khare, Noel Framil Iglesias  

**Link**: [PDF](https://arxiv.org/pdf/2604.00003)  

**Abstract**: This study evaluates the reliability of information extraction approaches from KRS documents using three strategies: LLM only, Hybrid Deterministic - LLM (regex + LLM), and a Camelot based pipeline with LLM fallback. Experiments were conducted on 140 documents for the LLM based test and 860 documents for the Camelot based pipeline evaluation, covering four study programs with varying data in tables and metadata. Three 12 - 14B LLM models (Gemma 3, Phi 4, and Qwen 2.5) were run locally using Ollama and a consumer grade CPU without a GPU. Evaluations used exact match (EM) and Levenshtein similarity (LS) metrics with a threshold of 0.7. Although not applicable to all models, the results show that the hybrid approach can improve efficiency compared to LLM only, especially for deterministic metadata. The Camelot based pipeline with LLM fallback produced the best combination of accuracy (EM and LS up to 0.99 - 1.00) and computational efficiency (less than 1 second per PDF in most cases). The Qwen 2.5:14b model demonstrated the most consistent performance across all scenarios. These findings confirm that integrating deterministic and LLM methods is increasingly reliable and efficient for information extraction from text based academic documents in computationally constrained environments. 

---
# A novel three-step approach to forecast firm-specific technology convergence opportunity via multi-dimensional feature fusion 

**Authors**: Fu Gu, Ao Chen, Yingwen Wu  

**Link**: [PDF](https://arxiv.org/pdf/2604.00803)  

**Abstract**: As a crucial innovation paradigm, technology convergence (TC) is gaining ever-increasing attention. Yet, existing studies primarily focus on predicting TC at the industry level, with little attention paid to TC forecast for firm-specific technology opportunity discovery (TOD). Moreover, although technological documents like patents contain a rich body of bibliometric, network structure, and textual features, such features are underexploited in the extant TC predictions; most of the relevant studies only used one or two dimensions of these features, and all the three dimensional features have rarely been fused. Here we propose a novel approach that fuses multi-dimensional features from patents to predict TC for firm-specific TOD. Our method comprises three steps, which are elaborated as follows. First, bibliometric, network structure, and textual features are extracted from patent documents, and then fused at the International Patent Classification (IPC)-pair level using attention mechanisms. Second, IPC-level TC opportunities are identified using a two-stage ensemble learning model that incorporates various imbalance-handling strategies. Third, to acquire feasible firm-specific TC opportunities, the performance metrics of topic-level TC opportunities, which are refined from IPC-level opportunities, are evaluated via retrieval-augmented generation (RAG) with a large language model (LLM). We prove the effectiveness of our proposed approach by predicting TC opportunities for a leading Chinese auto part manufacturer, Zhejiang Sanhua Intelligent Controls co., ltd, in the domains of thermal management for energy storage and robotics. In sum, this work advances the theory and applicability of forecasting firm-specific TC opportunity through fusing multi-dimensional features and leveraging LLM-as-a-judge for technology opportunity evaluation. 

---
# Scalable Identification and Prioritization of Requisition-Specific Personal Competencies Using Large Language Models 

**Authors**: Wanxin Li, Denver McNeney, Nivedita Prabhu, Charlene Zhang, Renee Barr, Matthew Kitching, Khanh Dao Duc, Anthony S. Boyce  

**Link**: [PDF](https://arxiv.org/pdf/2604.00006)  

**Abstract**: AI-powered recruitment tools are increasingly adopted in personnel selection, yet they struggle to capture the requisition (req)-specific personal competencies (PCs) that distinguish successful candidates beyond job categories. We propose a large language model (LLM)-based approach to identify and prioritize req-specific PCs from reqs. Our approach integrates dynamic few-shot prompting, reflection-based self-improvement, similarity-based filtering, and multi-stage validation. Applied to a dataset of Program Manager reqs, our approach correctly identifies the highest-priority req-specific PCs with an average accuracy of 0.76, approaching human expert inter-rater reliability, and maintains a low out-of-scope rate of 0.07. 

---
# Embarrassingly Simple Self-Distillation Improves Code Generation 

**Authors**: Ruixiang Zhang, Richard He Bai, Huangjie Zheng, Navdeep Jaitly, Ronan Collobert, Yizhe Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2604.01193)  

**Abstract**: Can a large language model (LLM) improve at code generation using only its own raw outputs, without a verifier, a teacher model, or reinforcement learning? We answer in the affirmative with simple self-distillation (SSD): sample solutions from the model with certain temperature and truncation configurations, then fine-tune on those samples with standard supervised fine-tuning. SSD improves Qwen3-30B-Instruct from 42.4% to 55.3% pass@1 on LiveCodeBench v6, with gains concentrating on harder problems, and it generalizes across Qwen and Llama models at 4B, 8B, and 30B scale, including both instruct and thinking variants. To understand why such a simple method can work, we trace these gains to a precision-exploration conflict in LLM decoding and show that SSD reshapes token distributions in a context-dependent way, suppressing distractor tails where precision matters while preserving useful diversity where exploration matters. Taken together, SSD offers a complementary post-training direction for improving LLM code generation. 

---
# CARE: Privacy-Compliant Agentic Reasoning with Evidence Discordance 

**Authors**: Haochen Liu, Weien Li, Rui Song, Zeyu Li, Chun Jason Xue, Xiao-Yang Liu, Sam Nallaperuma, Xue Liu, Ye Yuan  

**Link**: [PDF](https://arxiv.org/pdf/2604.01113)  

**Abstract**: Large language model (LLM) systems are increasingly used to support high-stakes decision-making, but they typically perform worse when the available evidence is internally inconsistent. Such a scenario exists in real-world healthcare settings, with patient-reported symptoms contradicting medical signs. To study this problem, we introduce MIMIC-DOS, a dataset for short-horizon organ dysfunction worsening prediction in the intensive care unit (ICU) setting. We derive this dataset from the widely recognized MIMIC-IV, a publicly available electronic health record dataset, and construct it exclusively from cases in which discordance between signs and symptoms exists. This setting poses a substantial challenge for existing LLM-based approaches, with single-pass LLMs and agentic pipelines often struggling to reconcile such conflicting signals. To address this problem, we propose CARE: a multi-stage privacy-compliant agentic reasoning framework in which a remote LLM provides guidance by generating structured categories and transitions without accessing sensitive patient data, while a local LLM uses these categories and transitions to support evidence acquisition and final decision-making. Empirically, CARE achieves stronger performance across all key metrics compared to multiple baseline settings, showing that CARE can more robustly handle conflicting clinical evidence while preserving privacy. 

---
# LLM REgression with a Latent Iterative State Head 

**Authors**: Yiheng Su, Matthew Lease  

**Link**: [PDF](https://arxiv.org/pdf/2604.01206)  

**Abstract**: We present RELISH (REgression with a Latent Iterative State Head), a novel, lightweight architecture designed for text regression with large language models. Rather than decoding numeric targets as text or aggregating multiple generated outputs, RELISH predicts scalar values directly from frozen LLM representations by iteratively refining a learned latent state through cross-attention over token-level representations, and then mapping the final state to a point estimate with a linear regressor. Across five datasets, four LLM backbones, and two LLM training regimes, RELISH consistently outperforms prior baselines from all three major LLM regression families, including autoregressive decoding, regression-aware inference, and existing predictive head methods. Despite these gains, RELISH remains highly parameter-efficient, requiring only 3.4-3.7M trainable parameters across frozen LLM backbones (only 0.01-0.04% additional overhead), far less than LoRA-based alternatives that grow with model size (0.26-0.42%). 

---
# When Users Change Their Mind: Evaluating Interruptible Agents in Long-Horizon Web Navigation 

**Authors**: Henry Peng Zou, Chunyu Miao, Wei-Chieh Huang, Yankai Chen, Yue Zhou, Hanrong Zhang, Yaozu Wu, Liancheng Fang, Zhengyao Gu, Zhen Zhang, Kening Zheng, Fangxin Wang, Yi Nian, Shanghao Li, Wenzhe Fan, Langzhou He, Weizhi Zhang, Xue Liu, Philip S. Yu  

**Link**: [PDF](https://arxiv.org/pdf/2604.00892)  

**Abstract**: As LLM agents transition from short, static problem solving to executing complex, long-horizon tasks in dynamic environments, the ability to handle user interruptions, such as adding requirement or revising goals, during mid-task execution is becoming a core requirement for realistic deployment. However, existing benchmarks largely assume uninterrupted agent behavior or study interruptions only in short, unconstrained language tasks. In this paper, we present the first systematic study of interruptible agents in long-horizon, environmentally grounded web navigation tasks, where actions induce persistent state changes. We formalize three realistic interruption types, including addition, revision, and retraction, and introduce InterruptBench, a benchmark derived from WebArena-Lite that synthesizes high-quality interruption scenarios under strict semantic constraints. Using a unified interruption simulation framework, we evaluate six strong LLM backbones across single- and multi-turn interruption settings, analyzing both their effectiveness in adapting to updated intents and their efficiency in recovering from mid-task changes. Our results show that handling user interruptions effectively and efficiently during long-horizon agentic tasks remains challenging for powerful large-scale LLMs. Code and dataset are available at this https URL. 

---
# From Early Encoding to Late Suppression: Interpreting LLMs on Character Counting Tasks 

**Authors**: Ayan Datta, Mounika Marreddy, Alexander Mehler, Zhixue Zhao, Radhika Mamidi  

**Link**: [PDF](https://arxiv.org/pdf/2604.00778)  

**Abstract**: Large language models (LLMs) exhibit failures on elementary symbolic tasks such as character counting in a word, despite excelling on complex benchmarks. Although this limitation has been noted, the internal reasons remain unclear. We use character counting (e.g., "How many p's are in apple?") as a minimal, controlled probe that isolates token-level reasoning from higher-level confounds. Using this setting, we uncover a consistent phenomenon across modern architectures, including LLaMA, Qwen, and Gemma: models often compute the correct answer internally yet fail to express it at the output layer.
Through mechanistic analysis combining probing classifiers, activation patching, logit lens analysis, and attention head tracing, we show that character-level information is encoded in early and mid-layer representations. However, this information is attenuated by a small set of components in later layers, especially the penultimate and final layer MLP. We identify these components as negative circuits: subnetworks that downweight correct signals in favor of higher-probability but incorrect outputs.
Our results lead to two contributions. First, we show that symbolic reasoning failures in LLMs are not due to missing representations or insufficient scale, but arise from structured interference within the model's computation graph. This explains why such errors persist and can worsen under scaling and instruction tuning. Second, we provide evidence that LLM forward passes implement a form of competitive decoding, in which correct and incorrect hypotheses coexist and are dynamically reweighted, with final outputs determined by suppression as much as by amplification.
These findings carry implications for interpretability and robustness: simple symbolic reasoning exposes weaknesses in modern LLMs, underscoring need for design strategies that ensure information is encoded and reliably used. 

---
# AfrIFact: Cultural Information Retrieval, Evidence Extraction and Fact Checking for African Languages 

**Authors**: Israel Abebe Azime, Jesujoba Oluwadara Alabi, Crystina Zhang, Iffat Maab, Atnafu Lambebo Tonja, Tadesse Destaw Belay, Folasade Peace Alabi, Salomey Osei, Saminu Mohammad Aliyu, Nkechinyere Faith Aguobi, Bontu Fufa Balcha, Blessing Kudzaishe Sibanda, Davis David, Mouhamadane Mboup, Daud Abolade, Neo Putini, Philipp Slusallek, David Ifeoluwa Adelani, Dietrich Klakow  

**Link**: [PDF](https://arxiv.org/pdf/2604.00706)  

**Abstract**: Assessing the veracity of a claim made online is a complex and important task with real-world implications. When these claims are directed at communities with limited access to information and the content concerns issues such as healthcare and culture, the consequences intensify, especially in low-resource languages. In this work, we introduce AfrIFact, a dataset that covers the necessary steps for automatic fact-checking (i.e., information retrieval, evidence extraction, and fact checking), in ten African languages and English. Our evaluation results show that even the best embedding models lack cross-lingual retrieval capabilities, and that cultural and news documents are easier to retrieve than healthcare-domain documents, both in large corpora and in single documents. We show that LLMs lack robust multilingual fact-verification capabilities in African languages, while few-shot prompting improves performance by up to 43% in AfriqueQwen-14B, and task-specific fine-tuning further improves fact-checking accuracy by up to 26%. These findings, along with our release of the AfrIFact dataset, encourage work on low-resource information retrieval, evidence retrieval, and fact checking. 

---
# Positional Cognitive Specialization: Where Do LLMs Learn To Comprehend and Speak Your Language? 

**Authors**: Luis Frentzen Salim, Lun-Wei Ku, Hsing-Kuo Kenneth Pao  

**Link**: [PDF](https://arxiv.org/pdf/2604.00923)  

**Abstract**: Adapting large language models (LLMs) to new languages is an expensive and opaque process. Understanding how language models acquire new languages and multilingual abilities is key to achieve efficient adaptation. Prior work on multilingual interpretability research focuses primarily on how trained models process multilingual instructions, leaving unexplored the mechanisms through which they acquire new languages during training. We investigate these training dynamics on decoder-only transformers through the lens of two functional cognitive specializations: language perception (input comprehension) and production (output generation). Through experiments on low-resource languages, we demonstrate how perceptual and productive specialization emerges in different regions of a language model by running layer ablation sweeps from the model's input and output directions. Based on the observed specialization patterns, we propose CogSym, a layer-wise heuristic that enables effective adaptation by exclusively fine-tuning a few early and late layers. We show that tuning only the 25% outermost layers achieves downstream task performance within 2-3% deviation from the full fine-tuning baseline. CogSym yields consistent performance with adapter methods such as LoRA, showcasing generalization beyond full fine-tuning. These findings provide insights to better understand how LLMs learn new languages and push toward accessible and inclusive language modeling. 

---
# More Human, More Efficient: Aligning Annotations with Quantized SLMs 

**Authors**: Jiayu Wang, Junyoung Lee  

**Link**: [PDF](https://arxiv.org/pdf/2604.00586)  

**Abstract**: As Large Language Model (LLM) capabilities advance, the demand for high-quality annotation of exponentially increasing text corpora has outpaced human capacity, leading to the widespread adoption of LLMs in automatic evaluation and annotation. However, proprietary LLMs often exhibit systematic biases that diverge from human expert consensus, lacks reproducibility, and raises data privacy concerns. Our work examines the viability of finetuning a quantized Small Language Model of 1.7B parameter size on limited human-annotated data to serve as a highly aligned, deterministic evaluator and annotator. By implementing a custom, multi-dimensional rubric framework and simple augmentation and regularization techniques, the proposed approach achieves higher inter-annotator agreement (0.23 points increase in Krippendorff's $\alpha$) than the best performing state-of-the-art proprietary LLM. We also demonstrate the generalizability of the proposed training pipeline on a separate emotion classification task. The results show that task-specific alignment and efficient 4-bit quantized fine-tuning provide superior open-source alternative to using proprietary models for evaluation and annotation. Our finetuning approach is publicly available at this https URL. 

---
# Speech LLMs are Contextual Reasoning Transcribers 

**Authors**: Keqi Deng, Ruchao Fan, Bo Ren, Yiming Wang, Jinyu Li  

**Link**: [PDF](https://arxiv.org/pdf/2604.00610)  

**Abstract**: Despite extensions to speech inputs, effectively leveraging the rich knowledge and contextual understanding of large language models (LLMs) in automatic speech recognition (ASR) remains non-trivial, as the task primarily involves direct speech-to-text mapping. To address this, this paper proposes chain-of-thought ASR (CoT-ASR), which constructs a reasoning chain that enables LLMs to first analyze the input speech and generate contextual analysis, thereby fully exploiting their generative capabilities. With this contextual reasoning, CoT-ASR then performs more informed speech recognition and completes both reasoning and transcription in a single pass. Moreover, CoT-ASR naturally supports user-guided transcription: while designed to self-generate reasoning, it can also seamlessly incorporate user-provided context to guide transcription, further extending ASR functionality. To reduce the modality gap, this paper introduces a CTC-guided Modality Adapter, which uses CTC non-blank token probabilities to weight LLM embeddings, efficiently aligning speech encoder outputs with the LLM's textual latent space. Experiments show that, compared to standard LLM-based ASR, CoT-ASR achieves a relative reduction of 8.7% in word error rate (WER) and 16.9% in entity error rate (EER). 

---
# OmniVoice: Towards Omnilingual Zero-Shot Text-to-Speech with Diffusion Language Models 

**Authors**: Han Zhu, Lingxuan Ye, Wei Kang, Zengwei Yao, Liyong Guo, Fangjun Kuang, Zhifeng Han, Weiji Zhuang, Long Lin, Daniel Povey  

**Link**: [PDF](https://arxiv.org/pdf/2604.00688)  

**Abstract**: We present OmniVoice, a massive multilingual zero-shot text-to-speech (TTS) model that scales to over 600 languages. At its core is a novel diffusion language model-style discrete non-autoregressive (NAR) architecture. Unlike conventional discrete NAR models that suffer from performance bottlenecks in complex two-stage (text-to-semantic-to-acoustic) pipelines, OmniVoice directly maps text to multi-codebook acoustic tokens. This simplified approach is facilitated by two key technical innovations: (1) a full-codebook random masking strategy for efficient training, and (2) initialization from a pre-trained LLM to ensure superior intelligibility. By leveraging a 581k-hour multilingual dataset curated entirely from open-source data, OmniVoice achieves the broadest language coverage to date and delivers state-of-the-art performance across Chinese, English, and diverse multilingual benchmarks. Our code and pre-trained models are publicly available at this https URL. 

---
# Large Language Models in the Abuse Detection Pipeline 

**Authors**: Suraj Kath, Sanket Badhe, Preet Shah, Ashwin Sampathkumar, Shivani Gupta  

**Link**: [PDF](https://arxiv.org/pdf/2604.00323)  

**Abstract**: Online abuse has grown increasingly complex, spanning toxic language, harassment, manipulation, and fraudulent behavior. Traditional machine-learning approaches dependent on static classifiers and labor-intensive labeling struggle to keep pace with evolving threat patterns and nuanced policy requirements. Large Language Models introduce new capabilities for contextual reasoning, policy interpretation, explanation generation, and cross-modal understanding, enabling them to support multiple stages of modern safety systems. This survey provides a lifecycle-oriented analysis of how LLMs are being integrated into the Abuse Detection Lifecycle (ADL), which we define across four stages: (I) Label \& Feature Generation, (II) Detection, (III) Review \& Appeals, and (IV) Auditing \& Governance. For each stage, we synthesize emerging research and industry practices, highlight architectural considerations for production deployment, and examine the strengths and limitations of LLM-driven approaches. We conclude by outlining key challenges including latency, cost-efficiency, determinism, adversarial robustness, and fairness and discuss future research directions needed to operationalize LLMs as reliable, accountable components of large-scale abuse-detection and governance systems. 

---
# Locally Confident, Globally Stuck: The Quality-Exploration Dilemma in Diffusion Language Models 

**Authors**: Liancheng Fang, Aiwei Liu, Henry Peng Zou, Yankai Chen, Enze Ma, Leyi Pan, Chunyu Miao, Wei-Chieh Huang, Xue Liu, Philip S. Yu  

**Link**: [PDF](https://arxiv.org/pdf/2604.00375)  

**Abstract**: Diffusion large language models (dLLMs) theoretically permit token decoding in arbitrary order, a flexibility that could enable richer exploration of reasoning paths than autoregressive (AR) LLMs. In practice, however, random-order decoding often hurts generation quality. To mitigate this, low-confidence remasking improves single-sample quality (e.g., Pass@$1$) by prioritizing confident tokens, but it also suppresses exploration and limits multi-sample gains (e.g., Pass@$k$), creating a fundamental quality--exploration dilemma. In this paper, we provide a unified explanation of this dilemma. We show that low-confidence remasking improves a myopic proxy for quality while provably constraining the entropy of the induced sequence distribution. To overcome this limitation, we characterize the optimal distribution that explicitly balances quality and exploration, and develop a simple Independent Metropolis--Hastings sampler that approximately targets this distribution during decoding. Experiments across a range of reasoning benchmarks including MATH500, AIME24/25, HumanEval, and MBPP show that our approach yields better exploration-quality tradeoff than both random and low-confidence remasking. 

---
# Agent Q-Mix: Selecting the Right Action for LLM Multi-Agent Systems through Reinforcement Learning 

**Authors**: Eric Hanchen Jiang, Levina Li, Rui Sun, Xiao Liang, Yubei Li, Yuchen Wu, Haozheng Luo, Hengli Li, Zhi Zhang, Zhaolu Kang, Kai-Wei Chang, Ying Nian Wu  

**Link**: [PDF](https://arxiv.org/pdf/2604.00344)  

**Abstract**: Large Language Models (LLMs) have shown remarkable performance in completing various tasks. However, solving complex problems often requires the coordination of multiple agents, raising a fundamental question: how to effectively select and interconnect these agents. In this paper, we propose \textbf{Agent Q-Mix}, a reinforcement learning framework that reformulates topology selection as a cooperative Multi-Agent Reinforcement Learning (MARL) problem. Our method learns decentralized communication decisions using QMIX value factorization, where each agent selects from a set of communication actions that jointly induce a round-wise communication graph. At its core, Agent Q-Mix combines a topology-aware GNN encoder, GRU memory, and per-agent Q-heads under a Centralized Training with Decentralized Execution (CTDE) paradigm. The framework optimizes a reward function that balances task accuracy with token cost. Across seven core benchmarks in coding, reasoning, and mathematics, Agent Q-Mix achieves the highest average accuracy compared to existing methods while demonstrating superior token efficiency and robustness against agent failure. Notably, on the challenging Humanity's Last Exam (HLE) using Gemini-3.1-Flash-Lite as a backbone, Agent Q-Mix achieves 20.8\% accuracy, outperforming Microsoft Agent Framework (19.2\%) and LangGraph (19.2\%), followed by AutoGen and Lobster by OpenClaw. These results underscore the effectiveness of learned, decentralized topology optimization in pushing the boundaries of multi-agent reasoning. 

---
# Can Large Language Models Self-Correct in Medical Question Answering? An Exploratory Study 

**Authors**: Zaifu Zhan, Mengyuan Cui, Rui Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2604.00261)  

**Abstract**: Large language models (LLMs) have achieved strong performance on medical question answering (medical QA), and chain-of-thought (CoT) prompting has further improved results by eliciting explicit intermediate reasoning; meanwhile, self-reflective (self-corrective) prompting has been widely claimed to enhance model reliability by prompting LLMs to critique and revise their own reasoning, yet its effectiveness in safety-critical medical settings remains unclear. In this work, we conduct an exploratory analysis of self-reflective reasoning for medical multiple-choice question answering: using GPT-4o and GPT-4o-mini, we compare standard CoT prompting with an iterative self-reflection loop and track how predictions evolve across reflection steps on three widely used medical QA benchmarks (MedQA, HeadQA, and PubMedQA). We analyze whether self-reflection leads to error correction, error persistence, or the introduction of new errors. Our results show that self-reflective prompting does not consistently improve accuracy and its impact is highly dataset- and model-dependent: it yields modest gains on MedQA but provides limited or negative benefits on HeadQA and PubMedQA, and increasing the number of reflection steps does not guarantee better performance. These findings highlight a gap between reasoning transparency and reasoning correctness, suggesting that self-reflective reasoning is better viewed as an analytical tool for understanding model behavior rather than a standalone solution for improving medical QA reliability. 

---
# A Taxonomy of Programming Languages for Code Generation 

**Authors**: Nishat Raihan, Christian Newman, Marcos Zampieri  

**Link**: [PDF](https://arxiv.org/pdf/2604.00239)  

**Abstract**: The world's 7,000+ languages vary widely in the availability of resources for NLP, motivating efforts to systematically categorize them by their degree of resourcefulness (Joshi et al., 2020). A similar disparity exists among programming languages (PLs); however, no resource-tier taxonomy has been established for code. As large language models (LLMs) grow increasingly capable of generating code, such a taxonomy becomes essential. To fill this gap, we present the first reproducible PL resource classification, grouping 646 languages into four tiers. We show that only 1.9% of languages (Tier 3, High) account for 74.6% of all tokens in seven major corpora, while 71.7% of languages (Tier 0, Scarce) contribute just 1.0%. Statistical analyses of within-tier inequality, dispersion, and distributional skew confirm that this imbalance is both extreme and systematic. Our results provide a principled framework for dataset curation and tier-aware evaluation of multilingual LLMs. 

---
# Do LLMs Know What Is Private Internally? Probing and Steering Contextual Privacy Norms in Large Language Model Representations 

**Authors**: Haoran Wang, Li Xiong, Kai Shu  

**Link**: [PDF](https://arxiv.org/pdf/2604.00209)  

**Abstract**: Large language models (LLMs) are increasingly deployed in high-stakes settings, yet they frequently violate contextual privacy by disclosing private information in situations where humans would exercise discretion. This raises a fundamental question: do LLMs internally encode contextual privacy norms, and if so, why do violations persist? We present the first systematic study of contextual privacy as a structured latent representation in LLMs, grounded in contextual integrity (CI) theory. Probing multiple models, we find that the three norm-determining CI parameters (information type, recipient, and transmission principle) are encoded as linearly separable and functionally independent directions in activation space. Despite this internal structure, models still leak private information in practice, revealing a clear gap between concept representation and model behavior. To bridge this gap, we introduce CI-parametric steering, which independently intervenes along each CI dimension. This structured control reduces privacy violations more effectively and predictably than monolithic steering. Our results demonstrate that contextual privacy failures arise from misalignment between representation and behavior rather than missing awareness, and that leveraging the compositional structure of CI enables more reliable contextual privacy control, shedding light on potential improvement of contextual privacy understanding in LLMs. 

---
# Hierarchical Chain-of-Thought Prompting: Enhancing LLM Reasoning Performance and Efficiency 

**Authors**: Xingshuai Huang, Derek Li, Bahareh Nikpour, Parsa Omidi  

**Link**: [PDF](https://arxiv.org/pdf/2604.00130)  

**Abstract**: Chain-of-Thought (CoT) prompting has significantly improved the reasoning capabilities of large language models (LLMs). However, conventional CoT often relies on unstructured, flat reasoning chains that suffer from redundancy and suboptimal performance. In this work, we introduce Hierarchical Chain-of-Thought (Hi-CoT) prompting, a structured reasoning paradigm specifically designed to address the challenges of complex, multi-step reasoning. Hi-CoT decomposes the reasoning process into hierarchical substeps by alternating between instructional planning and step-by-step execution. This decomposition enables LLMs to better manage long reasoning horizons and maintain logical coherence. Extensive evaluations across diverse LLMs and mathematical reasoning benchmarks show that Hi-CoT consistently improves average accuracy by 6.2% (up to 61.4% on certain models and tasks) while reducing reasoning trace length by 13.9% compared to CoT prompting. We further show that accuracy and efficiency are maximized when models strictly adhere to the hierarchical structure. Our code is available at this https URL. 

---
# Multi-lingual Multi-institutional Electronic Health Record based Predictive Model 

**Authors**: Kyunghoon Hur, Heeyoung Kwak, Jinsu Jang, Nakhwan Kim, Edward Choi  

**Link**: [PDF](https://arxiv.org/pdf/2604.00027)  

**Abstract**: Large-scale EHR prediction across institutions is hindered by substantial heterogeneity in schemas and code systems. Although Common Data Models (CDMs) can standardize records for multi-institutional learning, the manual harmonization and vocabulary mapping are costly and difficult to scale. Text-based harmonization provides an alternative by converting raw EHR into a unified textual form, enabling pooled learning without explicit standardization. However, applying this paradigm to multi-national datasets introduces an additional layer of heterogeneity, which is "language" that must be addressed for truly scalable EHRs learning. In this work, we investigate multilingual multi-institutional learning for EHR prediction, aiming to enable pooled training across multinational ICU datasets without manual standardization. We compare two practical strategies for handling language barriers: (i) directly modeling multilingual records with multilingual encoders, and (ii) translating non-English records into English via LLM-based word-level translation. Across seven public ICU datasets, ten clinical tasks with multiple prediction windows, translation-based lingual alignment yields more reliable cross-dataset performance than multilingual encoders. The multi-institutional learning model consistently outperforms strong baselines that require manual feature selection and harmonization, and also surpasses single-dataset training. We further demonstrate that text-based framework with lingual alignment effectively performs transfer learning via few-shot fine-tuning, with additional gains. To our knowledge, this is the first study to aggregate multilingual multinational ICU EHR datasets into one predictive model, providing a scalable path toward language-agnostic clinical prediction and future global multi-institutional EHR research. 

---
# Do Language Models Know When They'll Refuse? Probing Introspective Awareness of Safety Boundaries 

**Authors**: Tanay Gondil  

**Link**: [PDF](https://arxiv.org/pdf/2604.00228)  

**Abstract**: Large language models are trained to refuse harmful requests, but can they accurately predict when they will refuse before responding? We investigate this question through a systematic study where models first predict their refusal behavior, then respond in a fresh context. Across 3754 datapoints spanning 300 requests, we evaluate four frontier models: Claude Sonnet 4, Claude Sonnet 4.5, GPT-5.2, and Llama 3.1 405B. Using signal detection theory (SDT), we find that all models exhibit high introspective sensitivity (d' = 2.4-3.5), but sensitivity drops substantially at safety boundaries. We observe generational improvement within Claude (Sonnet 4.5: 95.7 percent accuracy vs Sonnet 4: 93.0 percent), while GPT-5.2 shows lower accuracy (88.9 percent) with more variable behavior. Llama 405B achieves high sensitivity but exhibits strong refusal bias and poor calibration, resulting in lower overall accuracy (80.0 percent). Topic-wise analysis reveals weapons-related queries are consistently hardest for introspection. Critically, confidence scores provide actionable signal: restricting to high-confidence predictions yields 98.3 percent accuracy for well-calibrated models, enabling practical confidence-based routing for safety-critical deployments. 

---
# Disentangling Prompt Element Level Risk Factors for Hallucinations and Omissions in Mental Health LLM Responses 

**Authors**: Congning Ni, Sarvech Qadir, Bryan Steitz, Mihir Sachin Vaidya, Qingyuan Song, Lantian Xia, Shelagh Mulvaney, Siru Liu, Hyeyoung Ryu, Leah Hecht, Amy Bucher, Christopher Symons, Laurie Novak, Susannah L. Rose, Murat Kantarcioglu, Bradley Malin, Zhijun Yin  

**Link**: [PDF](https://arxiv.org/pdf/2604.00014)  

**Abstract**: Mental health concerns are often expressed outside clinical settings, including in high-distress help seeking, where safety-critical guidance may be needed. Consumer health informatics systems increasingly incorporate large language models (LLMs) for mental health question answering, yet many evaluations underrepresent narrative, high-distress inquiries. We introduce UTCO (User, Topic, Context, Tone), a prompt construction framework that represents an inquiry as four controllable elements for systematic stress testing. Using 2,075 UTCO-generated prompts, we evaluated Llama 3.3 and annotated hallucinations (fabricated or incorrect clinical content) and omissions (missing clinically necessary or safety-critical guidance). Hallucinations occurred in 6.5% of responses and omissions in 13.2%, with omissions concentrated in crisis and suicidal ideation prompts. Across regression, element-specific matching, and similarity-matched comparisons, failures were most consistently associated with context and tone, while user-background indicators showed no systematic differences after balancing. These findings support evaluating omissions as a primary safety outcome and moving beyond static benchmark question sets. 

---
# True (VIS) Lies: Analyzing How Generative AI Recognizes Intentionality, Rhetoric, and Misleadingness in Visualization Lies 

**Authors**: Graziano Blasilli, Marco Angelini  

**Link**: [PDF](https://arxiv.org/pdf/2604.01181)  

**Abstract**: This study investigates the ability of multimodal Large Language Models (LLMs) to identify and interpret misleading visualizations, and recognize these observations along with their underlying causes and potential intentionality. Our analysis leverages concepts from visualization rhetoric and a newly developed taxonomy of authorial intents as explanatory lenses. We formulated three research questions and addressed them experimentally using a dataset of 2,336 COVID-19-related tweets, half of which contain misleading visualizations, and supplemented it with real-world examples of perceptual, cognitive, and conceptual errors drawn from VisLies, the IEEE VIS community event dedicated to showcasing deceptive and misleading visualizations. To ensure broad coverage of the current LLM landscape, we evaluated 16 state-of-the-art models. Among them, 15 are open-weight models, spanning a wide range of model sizes, architectural families, and reasoning capabilities. The selection comprises small models, namely Nemotron-Nano-V2-VL (12B parameters), Mistral-Small-3.2 (24B), DeepSeek-VL2 (27B), Gemma3 (27B), and GTA1 (32B); medium-sized models, namely Qianfan-VL (70B), Molmo (72B), GLM-4.5V (108B), LLaVA-NeXT (110B), and Pixtral-Large (124B); and large models, namely Qwen3-VL (235B), InternVL3.5 (241B), Step3 (321B), Llama-4-Maverick (400B), and Kimi-K2.5 (1000B). In addition, we employed OpenAI GPT-5.4, a frontier proprietary model. To establish a human perspective on these tasks, we also conducted a user study with visualization experts to assess how people perceive rhetorical techniques and the authorial intentions behind the same misleading visualizations. This allows comparison between model and expert behavior, revealing similarities and differences that provide insights into where LLMs align with human judgment and where they diverge. 

---
# Multimodal Language Models Cannot Spot Spatial Inconsistencies 

**Authors**: Om Khangaonkar, Hadi J. Rad, Hamed Pirsiavash  

**Link**: [PDF](https://arxiv.org/pdf/2604.00799)  

**Abstract**: Spatial consistency is a fundamental property of the visual world and a key requirement for models that aim to understand physical reality. Despite recent advances, multimodal large language models (MLLMs) often struggle to reason about 3D geometry across multiple views. Rather than asking models to describe scene attributes, we introduce a more challenging task: given two views of the same scene, identify the object that violates 3D motion consistency. We propose a simple and scalable method for generating realistic, spatially inconsistent image pairs from multi-view scenes, enabling systematic evaluation of this capability. Our results show that state-of-the-art MLLMs significantly underperform human observers and exhibit substantial variability across different scene attributes, revealing a fragile and incomplete understanding of 3D structure. We hope our findings underscore the need for approaches that develop a more deeply grounded understanding of the physical world. 

---
# A Survey of On-Policy Distillation for Large Language Models 

**Authors**: Mingyang Song, Mao Zheng  

**Link**: [PDF](https://arxiv.org/pdf/2604.00626)  

**Abstract**: Knowledge distillation has become a primary mechanism for transferring reasoning and domain expertise from frontier Large Language Models (LLMs) to smaller, deployable students. However, the dominant paradigm remains \textit{off-policy}: students train on static teacher-generated data and never encounter their own errors during learning. This train--test mismatch, an instance of \textit{exposure bias}, causes prediction errors to compound autoregressively at inference time. On-Policy Distillation (OPD) addresses this by letting the student generate its own trajectories and receive teacher feedback on these self-generated outputs, grounding distillation in the theory of interactive imitation learning. Despite rapid growth spanning divergence minimization, reward-guided learning, and self-play, the OPD literature remains fragmented with no unified treatment. This survey provides the first comprehensive overview of OPD for LLMs. We introduce a unified $f$-divergence framework over on-policy samples and organize the landscape along three orthogonal dimensions: \emph{feedback signal} (logit-based, outcome-based, or self-play), \emph{teacher access} (white-box, black-box, or teacher-free), and \emph{loss granularity} (token-level, sequence-level, or hybrid). We systematically analyze representative methods, examine industrial deployments, and identify open problems including distillation scaling laws, uncertainty-aware feedback, and agent-level distillation. 

---
