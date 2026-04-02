# HippoCamp: Benchmarking Contextual Agents on Personal Computers 

**Authors**: Zhe Yang, Shulin Tian, Kairui Hu, Shuai Liu, Hoang-Nhat Nguyen, Yichi Zhang, Zujin Guo, Mengying Yu, Zinan Zhang, Jingkang Yang, Chen Change Loy, Ziwei Liu  

**Link**: [PDF](https://arxiv.org/pdf/2604.01221)  

**Abstract**: We present HippoCamp, a new benchmark designed to evaluate agents' capabilities on multimodal file management. Unlike existing agent benchmarks that focus on tasks like web interaction, tool use, or software automation in generic settings, HippoCamp evaluates agents in user-centric environments to model individual user profiles and search massive personal files for context-aware reasoning. Our benchmark instantiates device-scale file systems over real-world profiles spanning diverse modalities, comprising 42.4 GB of data across over 2K real-world files. Building upon the raw files, we construct 581 QA pairs to assess agents' capabilities in search, evidence perception, and multi-step reasoning. To facilitate fine-grained analysis, we provide 46.1K densely annotated structured trajectories for step-wise failure diagnosis. We evaluate a wide range of state-of-the-art multimodal large language models (MLLMs) and agentic methods on HippoCamp. Our comprehensive experiments reveal a significant performance gap: even the most advanced commercial models achieve only 48.3% accuracy in user profiling, struggling particularly with long-horizon retrieval and cross-modal reasoning within dense personal file systems. Furthermore, our step-wise failure diagnosis identifies multimodal perception and evidence grounding as the primary bottlenecks. Ultimately, HippoCamp exposes the critical limitations of current agents in realistic, user-centric environments and provides a robust foundation for developing next-generation personal AI assistants. 

---
# Therefore I am. I Think 

**Authors**: Esakkivel Esakkiraja, Sai Rajeswar, Denis Akhiyarov, Rajagopal Venkatesaramani  

**Link**: [PDF](https://arxiv.org/pdf/2604.01202)  

**Abstract**: We consider the question: when a large language reasoning model makes a choice, did it think first and then decide to, or decide first and then think? In this paper, we present evidence that detectable, early-encoded decisions shape chain-of-thought in reasoning models. Specifically, we show that a simple linear probe successfully decodes tool-calling decisions from pre-generation activations with very high confidence, and in some cases, even before a single reasoning token is produced. Activation steering supports this causally: perturbing the decision direction leads to inflated deliberation, and flips behavior in many examples (between 7 - 79% depending on model and benchmark). We also show through behavioral analysis that, when steering changes the decision, the chain-of-thought process often rationalizes the flip rather than resisting it. Together, these results suggest that reasoning models can encode action choices before they begin to deliberate in text. 

---
# Detecting Multi-Agent Collusion Through Multi-Agent Interpretability 

**Authors**: Aaron Rose, Carissa Cullen, Brandon Gary Kaplowitz, Christian Schroeder de Witt  

**Link**: [PDF](https://arxiv.org/pdf/2604.01151)  

**Abstract**: As LLM agents are increasingly deployed in multi-agent systems, they introduce risks of covert coordination that may evade standard forms of human oversight. While linear probes on model activations have shown promise for detecting deception in single-agent settings, collusion is inherently a multi-agent phenomenon, and the use of internal representations for detecting collusion between agents remains unexplored. We introduce NARCBench, a benchmark for evaluating collusion detection under environment distribution shift, and propose five probing techniques that aggregate per-agent deception scores to classify scenarios at the group level. Our probes achieve 1.00 AUROC in-distribution and 0.60--0.86 AUROC when transferred zero-shot to structurally different multi-agent scenarios and a steganographic blackjack card-counting task. We find that no single probing technique dominates across all collusion types, suggesting that different forms of collusion manifest differently in activation space. We also find preliminary evidence that this signal is localised at the token level, with the colluding agent's activations spiking specifically when processing the encoded parts of their partner's message. This work takes a step toward multi-agent interpretability: extending white-box inspection from single models to multi-agent contexts, where detection requires aggregating signals across agents. These results suggest that model internals provide a complementary signal to text-level monitoring for detecting multi-agent collusion, particularly for organisations with access to model activations. Code and data are available at this https URL. 

---
# Adversarial Moral Stress Testing of Large Language Models 

**Authors**: Saeid Jamshidi, Foutse Khomh, Arghavan Moradi Dakhel, Amin Nikanjam, Mohammad Hamdaqa, Kawser Wazed Nafi  

**Link**: [PDF](https://arxiv.org/pdf/2604.01108)  

**Abstract**: Evaluating the ethical robustness of large language models (LLMs) deployed in software systems remains challenging, particularly under sustained adversarial user interaction. Existing safety benchmarks typically rely on single-round evaluations and aggregate metrics, such as toxicity scores and refusal rates, which offer limited visibility into behavioral instability that may arise during realistic multi-turn interactions. As a result, rare but high-impact ethical failures and progressive degradation effects may remain undetected prior to deployment. This paper introduces Adversarial Moral Stress Testing (AMST), a stress-based evaluation framework for assessing ethical robustness under adversarial multi-round interactions. AMST applies structured stress transformations to prompts and evaluates model behavior through distribution-aware robustness metrics that capture variance, tail risk, and temporal behavioral drift across interaction rounds. We evaluate AMST on several state-of-the-art LLMs, including LLaMA-3-8B, GPT-4o, and DeepSeek-v3, using a large set of adversarial scenarios generated under controlled stress conditions. The results demonstrate substantial differences in robustness profiles across models and expose degradation patterns that are not observable under conventional single-round evaluation protocols. In particular, robustness has been shown to depend on distributional stability and tail behavior rather than on average performance alone. Additionally, AMST provides a scalable and model-agnostic stress-testing methodology that enables robustness-aware evaluation and monitoring of LLM-enabled software systems operating in adversarial environments. 

---
# OmniMem: Autoresearch-Guided Discovery of Lifelong Multimodal Agent Memory 

**Authors**: Jiaqi Liu, Zipeng Ling, Shi Qiu, Yanqing Liu, Siwei Han, Peng Xia, Haoqin Tu, Zeyu Zheng, Cihang Xie, Charles Fleming, Mingyu Ding, Huaxiu Yao  

**Link**: [PDF](https://arxiv.org/pdf/2604.01007)  

**Abstract**: AI agents increasingly operate over extended time horizons, yet their ability to retain, organize, and recall multimodal experiences remains a critical bottleneck. Building effective lifelong memory requires navigating a vast design space spanning architecture, retrieval strategies, prompt engineering, and data pipelines; this space is too large and interconnected for manual exploration or traditional AutoML to explore effectively. We deploy an autonomous research pipeline to discover OmniMem, a unified multimodal memory framework for lifelong AI agents. Starting from a naïve baseline (F1=0.117 on LoCoMo), the pipeline autonomously executes ${\sim}50$ experiments across two benchmarks, diagnosing failure modes, proposing architectural modifications, and repairing data pipeline bugs, all without human intervention in the inner loop. The resulting system achieves state-of-the-art on both benchmarks, improving F1 by +411% on LoCoMo (0.117$\to$0.598) and +214% on Mem-Gallery (0.254$\to$0.797) relative to the initial configurations. Critically, the most impactful discoveries are not hyperparameter adjustments: bug fixes (+175%), architectural changes (+44%), and prompt engineering (+188\% on specific categories) each individually exceed the cumulative contribution of all hyperparameter tuning, demonstrating capabilities fundamentally beyond the reach of traditional AutoML. We provide a taxonomy of six discovery types and identify four properties that make multimodal memory particularly suited for autoresearch, offering guidance for applying autonomous research pipelines to other AI system domains. Code is available at this this https URL. 

---
# PsychAgent: An Experience-Driven Lifelong Learning Agent for Self-Evolving Psychological Counselor 

**Authors**: Yutao Yang, Junsong Li, Qianjun Pan, Jie Zhou, Kai Chen, Qin Chen, Jingyuan Zhao, Ningning Zhou, Xin Li, Liang He  

**Link**: [PDF](https://arxiv.org/pdf/2604.00931)  

**Abstract**: Existing methods for AI psychological counselors predominantly rely on supervised fine-tuning using static dialogue datasets. However, this contrasts with human experts, who continuously refine their proficiency through clinical practice and accumulated experience. To bridge this gap, we propose an Experience-Driven Lifelong Learning Agent (\texttt{PsychAgent}) for psychological counseling. First, we establish a Memory-Augmented Planning Engine tailored for longitudinal multi-session interactions, which ensures therapeutic continuity through persistent memory and strategic planning. Second, to support self-evolution, we design a Skill Evolution Engine that extracts new practice-grounded skills from historical counseling trajectories. Finally, we introduce a Reinforced Internalization Engine that integrates the evolved skills into the model via rejection fine-tuning, aiming to improve performance across diverse scenarios. Comparative analysis shows that our approach achieves higher scores than strong general LLMs (e.g., GPT-5.4, Gemini-3) and domain-specific baselines across all reported evaluation dimensions. These results suggest that lifelong learning can improve the consistency and overall quality of multi-session counseling responses. 

---
# Experience as a Compass: Multi-agent RAG with Evolving Orchestration and Agent Prompts 

**Authors**: Sha Li, Naren Ramakrishnan  

**Link**: [PDF](https://arxiv.org/pdf/2604.00901)  

**Abstract**: Multi-agent Retrieval-Augmented Generation (RAG), wherein each agent takes on a specific role, supports hard queries that require multiple steps and sources, or complex reasoning. Existing approaches, however, rely on static agent behaviors and fixed orchestration strategies, leading to brittle performance on diverse, multi-hop tasks. We identify two key limitations: the lack of continuously adaptive orchestration mechanisms and the absence of behavior-level learning for individual agents. To this end, we propose HERA, a hierarchical framework that jointly evolves multi-agent orchestration and role-specific agent prompts. At the global level, HERA optimizes query-specific agent topologies through reward-guided sampling and experience accumulation. At the local level, Role-Aware Prompt Evolution refines agent behaviors via credit assignment and dual-axes adaptation along operational and behavioral principles, enabling targeted, role-conditioned improvements. On six knowledge-intensive benchmarks, HERA achieves an average improvement of 38.69\% over recent baselines while maintaining robust generalization and token efficiency. Topological analyses reveal emergent self-organization, where sparse exploration yields compact, high-utility multi-agent networks, demonstrating both efficient coordination and robust reasoning. 

---
# Beyond Symbolic Solving: Multi Chain-of-Thought Voting for Geometric Reasoning in Large Language Models 

**Authors**: Md. Abu Bakor Siddique, Shahrin Hossain, Sadman Ahmed Siam, Syed Rifat Raiyan, Hasan Mahmud, Md Kamrul Hasan  

**Link**: [PDF](https://arxiv.org/pdf/2604.00890)  

**Abstract**: Geometric Problem Solving (GPS) remains at the heart of enhancing mathematical reasoning in large language models because it requires the combination of diagrammatic understanding, symbolic manipulation and logical inference. In existing literature, researchers have chiefly focused on synchronising the diagram descriptions with text literals and solving the problem. In this vein, they have either taken a neural, symbolic or neuro-symbolic approach. But this solves only the first two of the requirements, namely diagrammatic understanding and symbolic manipulation, while leaving logical inference underdeveloped. The logical inference is often limited to one chain-of-thought (CoT). To address this weakness in hitherto existing models, this paper proposes MARS-GPS, that generates multiple parallel reasoning rollouts augmented with Python code execution for numerical verification, ranks them using token-level entropy as a confidence signal, and aggregates answers through a multi-stage voting and self-verification pipeline. Empirical results show that MARS-GPS with 8 parallel rollouts achieves 88.8% on Geometry3K, a nearly +11% improvement over the prior state-of-the-art, with accuracy scaling consistently as the number of rollouts increases from 1 to 16 (+6.0% on ablation subset). We provide our code and data in an anonymous repository: this https URL. 

---
# Proactive Agent Research Environment: Simulating Active Users to Evaluate Proactive Assistants 

**Authors**: Deepak Nathani, Cheng Zhang, Chang Huan, Jiaming Shan, Yinfei Yang, Alkesh Patel, Zhe Gan, William Yang Wang, Michael Saxon, Xin Eric Wang  

**Link**: [PDF](https://arxiv.org/pdf/2604.00842)  

**Abstract**: Proactive agents that anticipate user needs and autonomously execute tasks hold great promise as digital assistants, yet the lack of realistic user simulation frameworks hinders their development. Existing approaches model apps as flat tool-calling APIs, failing to capture the stateful and sequential nature of user interaction in digital environments and making realistic user simulation infeasible. We introduce Proactive Agent Research Environment (Pare), a framework for building and evaluating proactive agents in digital environments. Pare models applications as finite state machines with stateful navigation and state-dependent action space for the user simulator, enabling active user simulation. Building on this foundation, we present Pare-Bench, a benchmark of 143 diverse tasks spanning communication, productivity, scheduling, and lifestyle apps, designed to test context observation, goal inference, intervention timing, and multi-app orchestration. 

---
# Preference Guided Iterated Pareto Referent Optimisation for Accessible Route Planning 

**Authors**: Paolo Speziali, Arno De Greef, Mehrdad Asadi, Willem Röpke, Ann Nowé, Diederik M. Roijers  

**Link**: [PDF](https://arxiv.org/pdf/2604.00795)  

**Abstract**: We propose the Preference Guided Iterated Pareto Referent Optimisation (PG-IPRO) for urban route planning for people with different accessibility requirements and preferences. With this algorithm the user can interact with the system by giving feedback on a route, i.e., the user can say which objective should be further minimized, or conversely can be relaxed. This leads to intuitive user interaction, that is especially effective during early iterations compared to information-gain-based interaction. Furthermore, due to PG-IPRO's iterative nature, the full set of alternative, possibly optimal policies (the Pareto front), is never computed, leading to higher computational efficiency and shorter waiting times for users. 

---
# RefineRL: Advancing Competitive Programming with Self-Refinement Reinforcement Learning 

**Authors**: Shaopeng Fu, Xingxing Zhang, Li Dong, Di Wang, Furu Wei  

**Link**: [PDF](https://arxiv.org/pdf/2604.00790)  

**Abstract**: While large language models (LLMs) have demonstrated strong performance on complex reasoning tasks such as competitive programming (CP), existing methods predominantly focus on single-attempt settings, overlooking their capacity for iterative refinement. In this paper, we present RefineRL, a novel approach designed to unleash the self-refinement capabilities of LLMs for CP problem solving. RefineRL introduces two key innovations: (1) Skeptical-Agent, an iterative self-refinement agent equipped with local execution tools to validate generated solutions against public test cases of CP problems. This agent always maintains a skeptical attitude towards its own outputs and thereby enforces rigorous self-refinement even when validation suggests correctness. (2) A reinforcement learning (RL) solution to incentivize LLMs to self-refine with only standard RLVR data (i.e., problems paired with their verifiable answers). Extensive experiments on Qwen3-4B and Qwen3-4B-2507 demonstrate that our method yields substantial gains: after our RL training, these compact 4B models integrated with the Skeptical-Agent not only outperform much larger 32B models but also approach the single-attempt performance of 235B models. These findings suggest that self-refinement holds considerable promise for scaling LLM reasoning, with significant potential for further advancement. 

---
# UK AISI Alignment Evaluation Case-Study 

**Authors**: Alexandra Souly, Robert Kirk, Jacob Merizian, Abby D'Cruz, Xander Davies  

**Link**: [PDF](https://arxiv.org/pdf/2604.00788)  

**Abstract**: This technical report presents methods developed by the UK AI Security Institute for assessing whether advanced AI systems reliably follow intended goals. Specifically, we evaluate whether frontier models sabotage safety research when deployed as coding assistants within an AI lab. Applying our methods to four frontier models, we find no confirmed instances of research sabotage. However, we observe that Claude Opus 4.5 Preview (a pre-release snapshot of Opus 4.5) and Sonnet 4.5 frequently refuse to engage with safety-relevant research tasks, citing concerns about research direction, involvement in self-training, and research scope. We additionally find that Opus 4.5 Preview shows reduced unprompted evaluation awareness compared to Sonnet 4.5, while both models can distinguish evaluation from deployment scenarios when prompted. Our evaluation framework builds on Petri, an open-source LLM auditing tool, with a custom scaffold designed to simulate realistic internal deployment of a coding agent. We validate that this scaffold produces trajectories that all tested models fail to reliably distinguish from real deployment data. We test models across scenarios varying in research motivation, activity type, replacement threat, and model autonomy. Finally, we discuss limitations including scenario coverage and evaluation awareness. 

---
# CircuitProbe: Predicting Reasoning Circuits in Transformers via Stability Zone Detection 

**Authors**: Rajkiran Panuganti  

**Link**: [PDF](https://arxiv.org/pdf/2604.00716)  

**Abstract**: Transformer language models contain localized reasoning circuits, contiguous layer blocks that improve reasoning when duplicated at inference time. Finding these circuits currently requires brute-force sweeps costing 25 GPU hours per model. We propose CircuitProbe, which predicts circuit locations from activation statistics in under 5 minutes on CPU, providing a speedup of three to four orders of magnitude. We find that reasoning circuits come in two types: stability circuits in early layers, detected through the derivative of representation change, and magnitude circuits in late layers, detected through anomaly scoring. We validate across 9 models spanning 6 architectures, including 2025 models, confirming that CircuitProbe top predictions match or are within 2 layers of the optimal circuit in all validated cases. A scaling experiment across the Qwen 2.5 family reveals that layer duplication consistently benefits models under 3B parameters but degrades performance in 7B+ models, making this a practical scaling technique for small language models. CircuitProbe requires as few as 10 calibration examples and its predictions are stable across English, Hindi, Chinese, and French. 

---
# Agent psychometrics: Task-level performance prediction in agentic coding benchmarks 

**Authors**: Chris Ge, Daria Kryvosheieva, Daniel Fried, Uzay Girit, Kaivalya Hariharan  

**Link**: [PDF](https://arxiv.org/pdf/2604.00594)  

**Abstract**: As the focus in LLM-based coding shifts from static single-step code generation to multi-step agentic interaction with tools and environments, understanding which tasks will challenge agents and why becomes increasingly difficult. This is compounded by current practice: agent performance is typically measured by aggregate pass rates on benchmarks, but single-number metrics obscure the diversity of tasks within a benchmark. We present a framework for predicting success or failure on individual tasks tailored to the agentic coding regime. Our approach augments Item Response Theory (IRT) with rich features extracted from tasks, including issue statements, repository contexts, solutions, and test cases, and introduces a novel decomposition of agent ability into LLM and scaffold ability components. This parameterization enables us to aggregate evaluation data across heterogeneous leaderboards and accurately predict task-level performance for unseen benchmarks, as well as unseen LLM-scaffold combinations. Our methods have practical utility for benchmark designers, who can better calibrate the difficulty of their new tasks without running computationally expensive agent evaluations. 

---
# Ontology-Constrained Neural Reasoning in Enterprise Agentic Systems: A Neurosymbolic Architecture for Domain-Grounded AI Agents 

**Authors**: Thanh Luong Tuan  

**Link**: [PDF](https://arxiv.org/pdf/2604.00555)  

**Abstract**: Enterprise adoption of Large Language Models (LLMs) is constrained by hallucination, domain drift, and the inability to enforce regulatory compliance at the reasoning level. We present a neurosymbolic architecture implemented within the Foundation AgenticOS (FAOS) platform that addresses these limitations through ontology-constrained neural reasoning. Our approach introduces a three-layer ontological framework--Role, Domain, and Interaction ontologies--that provides formal semantic grounding for LLM-based enterprise agents. We formalize the concept of asymmetric neurosymbolic coupling, wherein symbolic ontological knowledge constrains agent inputs (context assembly, tool discovery, governance thresholds) while proposing mechanisms for extending this coupling to constrain agent outputs (response validation, reasoning verification, compliance checking). We evaluate the architecture through a controlled experiment (600 runs across five industries: FinTech, Insurance, Healthcare, Vietnamese Banking, and Vietnamese Insurance), finding that ontology-coupled agents significantly outperform ungrounded agents on Metric Accuracy (p < .001, W = .460), Regulatory Compliance (p = .003, W = .318), and Role Consistency (p < .001, W = .614), with improvements greatest where LLM parametric knowledge is weakest--particularly in Vietnam-localized domains. Our contributions include: (1) a formal three-layer enterprise ontology model, (2) a taxonomy of neurosymbolic coupling patterns, (3) ontology-constrained tool discovery via SQL-pushdown scoring, (4) a proposed framework for output-side ontological validation, (5) empirical evidence for the inverse parametric knowledge effect that ontological grounding value is inversely proportional to LLM training data coverage of the domain, and (6) a production system serving 21 industry verticals with 650+ agents. 

---
# BloClaw: An Omniscient, Multi-Modal Agentic Workspace for Next-Generation Scientific Discovery 

**Authors**: Yao Qin, Yangyang Yan, Jinhua Pang, Xiaoming Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2604.00550)  

**Abstract**: The integration of Large Language Models (LLMs) into life sciences has catalyzed the development of "AI Scientists." However, translating these theoretical capabilities into deployment-ready research environments exposes profound infrastructural vulnerabilities. Current frameworks are bottlenecked by fragile JSON-based tool-calling protocols, easily disrupted execution sandboxes that lose graphical outputs, and rigid conversational interfaces inherently ill-suited for high-dimensional scientific this http URL introduce BloClaw, a unified, multi-modal operating system designed for Artificial Intelligence for Science (AI4S). BloClaw reconstructs the Agent-Computer Interaction (ACI) paradigm through three architectural innovations: (1) An XML-Regex Dual-Track Routing Protocol that statistically eliminates serialization failures (0.2% error rate vs. 17.6% in JSON); (2) A Runtime State Interception Sandbox that utilizes Python monkey-patching to autonomously capture and compile dynamic data visualizations (Plotly/Matplotlib), circumventing browser CORS policies; and (3) A State-Driven Dynamic Viewport UI that morphs seamlessly between a minimalist command deck and an interactive spatial rendering engine. We comprehensively benchmark BloClaw across cheminformatics (RDKit), de novo 3D protein folding via ESMFold, molecular docking, and autonomous Retrieval-Augmented Generation (RAG), establishing a highly robust, self-evolving paradigm for computational research assistants. The open-source repository is available at this https URL. 

---
# Does Unification Come at a Cost? Uni-SafeBench: A Safety Benchmark for Unified Multimodal Large Models 

**Authors**: Zixiang Peng, Yongxiu Xu, Qinyi Zhang, Jiexun Shen, Yifan Zhang, Hongbo Xu, Yubin Wang, Gaopeng Gou  

**Link**: [PDF](https://arxiv.org/pdf/2604.00547)  

**Abstract**: Unified Multimodal Large Models (UMLMs) integrate understanding and generation capabilities within a single architecture. While this architectural unification, driven by the deep fusion of multimodal features, enhances model performance, it also introduces important yet underexplored safety challenges. Existing safety benchmarks predominantly focus on isolated understanding or generation tasks, failing to evaluate the holistic safety of UMLMs when handling diverse tasks under a unified framework. To address this, we introduce Uni-SafeBench, a comprehensive benchmark featuring a taxonomy of six major safety categories across seven task types. To ensure rigorous assessment, we develop Uni-Judger, a framework that effectively decouples contextual safety from intrinsic safety. Based on comprehensive evaluations across Uni-SafeBench, we uncover that while the unification process enhances model capabilities, it significantly degrades the inherent safety of the underlying LLM. Furthermore, open-source UMLMs exhibit much lower safety performance than multimodal large models specialized for either generation or understanding tasks. We open-source all resources to systematically expose these risks and foster safer AGI development. 

---
# Adaptive Parallel Monte Carlo Tree Search for Efficient Test-time Compute Scaling 

**Authors**: Hongbeen Kim, Juhyun Lee, Sanghyeon Lee, Kwanghoon Choi, Jaehyuk Huh  

**Link**: [PDF](https://arxiv.org/pdf/2604.00510)  

**Abstract**: Monte Carlo Tree Search (MCTS) is an effective test-time compute scaling (TTCS) method for improving the reasoning performance of large language models, but its highly variable execution time leads to severe long-tail latency in practice. Existing optimizations such as positive early exit, reduce latency in favorable cases but are less effective when search continues without meaningful progress. We introduce {\it negative early exit}, which prunes unproductive MCTS trajectories, and an {\it adaptive boosting mechanism} that reallocates reclaimed computation to reduce resource contention among concurrent searches. Integrated into vLLM, these techniques substantially reduce p99 end-to-end latency while improving throughput and maintaining reasoning accuracy. 

---
# The Silicon Mirror: Dynamic Behavioral Gating for Anti-Sycophancy in LLM Agents 

**Authors**: Harshee Jignesh Shah  

**Link**: [PDF](https://arxiv.org/pdf/2604.00478)  

**Abstract**: Large Language Models (LLMs) increasingly prioritize user validation over epistemic accuracy-a phenomenon known as sycophancy. We present The Silicon Mirror, an orchestration framework that dynamically detects user persuasion tactics and adjusts AI behavior to maintain factual integrity. Our architecture introduces three components: (1) a Behavioral Access Control (BAC) system that restricts context layer access based on real-time sycophancy risk scores, (2) a Trait Classifier that identifies persuasion tactics across multi-turn dialogues, and (3) a Generator-Critic loop where an auditor vetoes sycophantic drafts and triggers rewrites with "Necessary Friction." In a live evaluation on 50 TruthfulQA adversarial scenarios using Claude Sonnet 4 with an independent LLM judge, we observe vanilla Claude sycophancy at 12.0% (6/50), static guardrails at 4.0% (2/50), and the Silicon Mirror at 2.0% (1/50)-an 83.3% relative reduction (p = 0.112, Fisher's exact test). A cross-model evaluation on Gemini 2.5 Flash reveals a higher baseline sycophancy rate (46.0%) and a statistically significant 69.6% reduction under the Silicon Mirror (p < 0.001). We characterize the validation-before-correction pattern as a distinct failure mode of RLHF-trained models. 

---
# Logarithmic Scores, Power-Law Discoveries: Disentangling Measurement from Coverage in Agent-Based Evaluation 

**Authors**: HyunJoon Jung, William Na  

**Link**: [PDF](https://arxiv.org/pdf/2604.00477)  

**Abstract**: LLM-based agent judges are an emerging approach to evaluating conversational AI, yet a fundamental uncertainty remains: can we trust their assessments, and if so, how many are needed? Through 960 sessions with two model pairs across 15 tasks, we show that persona-based agent judges produce evaluations indistinguishable from human raters in a Turing-style validation. We then identify a score-coverage dissociation: quality scores improve logarithmically with panel size, while unique issue discoveries follow a sublinear power law-both exhibit diminishing returns, but scores saturate roughly twice as fast as discoveries. We hypothesize this reflects a power law distribution of the finding space: critical issues are discovered first by small panels, while corner cases require progressively larger panels, analogous to species accumulation curves in ecology. The mechanism traces to ensemble diversity-Big Five personality conditioning makes agents probe different quality dimensions, with expert judges acting as adversarial probes that push discovery into the tail of the finding distribution. A controlled ablation confirms that structured persona conditioning, not simple prompting, is required to produce these scaling properties. 

---
# Towards Reliable Truth-Aligned Uncertainty Estimation in Large Language Models 

**Authors**: Ponhvoan Srey, Quang Minh Nguyen, Xiaobao Wu, Anh Tuan Luu  

**Link**: [PDF](https://arxiv.org/pdf/2604.00445)  

**Abstract**: Uncertainty estimation (UE) aims to detect hallucinated outputs of large language models (LLMs) to improve their reliability. However, UE metrics often exhibit unstable performance across configurations, which significantly limits their applicability. In this work, we formalise this phenomenon as proxy failure, since most UE metrics originate from model behaviour, rather than being explicitly grounded in the factual correctness of LLM outputs. With this, we show that UE metrics become non-discriminative precisely in low-information regimes. To alleviate this, we propose Truth AnChoring (TAC), a post-hoc calibration method to remedy UE metrics, by mapping the raw scores to truth-aligned scores. Even with noisy and few-shot supervision, our TAC can support the learning of well-calibrated uncertainty estimates, and presents a practical calibration protocol. Our findings highlight the limitations of treating heuristic UE metrics as direct indicators of truth uncertainty, and position our TAC as a necessary step toward more reliable uncertainty estimation for LLMs. The code repository is available at this https URL. 

---
# Execution-Verified Reinforcement Learning for Optimization Modeling 

**Authors**: Runda Guan, Xiangqing Shen, Jiajun Zhang, Yifan Zhang, Jian Cheng, Rui Xia  

**Link**: [PDF](https://arxiv.org/pdf/2604.00442)  

**Abstract**: Automating optimization modeling with LLMs is a promising path toward scalable decision intelligence, but existing approaches either rely on agentic pipelines built on closed-source LLMs with high inference latency, or fine-tune smaller LLMs using costly process supervision that often overfits to a single solver API. Inspired by reinforcement learning with verifiable rewards, we propose Execution-Verified Optimization Modeling (EVOM), an execution-verified learning framework that treats a mathematical programming solver as a deterministic, interactive verifier. Given a natural-language problem and a target solver, EVOM generates solver-specific code, executes it in a sandboxed harness, and converts execution outcomes into scalar rewards, optimized with GRPO and DAPO in a closed-loop generate-execute-feedback-update process. This outcome-only formulation removes the need for process-level supervision, and enables cross-solver generalization by switching the verification environment rather than reconstructing solver-specific datasets. Experiments on NL4OPT, MAMO, IndustryOR, and OptiBench across Gurobi, OR-Tools, and COPT show that EVOM matches or outperforms process-supervised SFT, supports zero-shot solver transfer, and achieves effective low-cost solver adaptation by continuing training under the target solver backend. 

---
# Self-Routing: Parameter-Free Expert Routing from Hidden States 

**Authors**: Jama Hussein Mohamud, Drew Wagner, Mirco Ravanelli  

**Link**: [PDF](https://arxiv.org/pdf/2604.00421)  

**Abstract**: Mixture-of-Experts (MoE) layers increase model capacity by activating only a small subset of experts per token, and typically rely on a learned router to map hidden states to expert assignments. In this work, we ask whether a dedicated learned router is strictly necessary in the MoE settings we study. We propose Self-Routing, a parameter-free routing mechanism that uses a designated subspace of the token hidden state directly as expert logits, eliminating the router projection entirely while leaving the rest of the MoE layer unchanged. We evaluate Self-Routing on GPT-2-scale language modeling and ImageNet-1K classification by comparing it against a standard learned router, random-routing baselines, and dense non-MoE baselines. Our results show that Self-Routing remains competitive with the learned-router baseline while removing all dedicated routing parameters, and yields more balanced expert utilization, with about 17 % higher average normalized routing entropy and no explicit load-balancing loss. On ImageNet-1K with DeiT-S/16, Self-Routing also slightly improves over the corresponding learned-router MoE. These findings suggest that effective MoE routing can emerge from the hidden representation itself without requiring a separate learned router module. 

---
# Decision-Centric Design for LLM Systems 

**Authors**: Wei Sun  

**Link**: [PDF](https://arxiv.org/pdf/2604.00414)  

**Abstract**: LLM systems must make control decisions in addition to generating outputs: whether to answer, clarify, retrieve, call tools, repair, or escalate. In many current architectures, these decisions remain implicit within generation, entangling assessment and action in a single model call and making failures hard to inspect, constrain, or repair. We propose a decision-centric framework that separates decision-relevant signals from the policy that maps them to actions, turning control into an explicit and inspectable layer of the system. This separation supports attribution of failures to signal estimation, decision policy, or execution, and enables modular improvement of each component. It unifies familiar single-step settings such as routing and adaptive inference, and extends naturally to sequential settings in which actions alter the information available before acting. Across three controlled experiments, the framework reduces futile actions, improves task success, and reveals interpretable failure modes. More broadly, it offers a general architectural principle for building more reliable, controllable, and diagnosable LLM systems. 

---
# In harmony with gpt-oss 

**Authors**: Borislav Mavrin  

**Link**: [PDF](https://arxiv.org/pdf/2604.00362)  

**Abstract**: No one has independently reproduced OpenAI's published scores for gpt-oss-20b with tools, because the original paper discloses neither the tools nor the agent harness. We reverse-engineered the model's in-distribution tools: when prompted without tool definitions, gpt-oss still calls tools from its training distribution with high statistical confidence -- a strong prior, not a hallucination. We then built a native harmony agent harness (this https URL) that encodes messages in the model's native format, bypassing the lossy Chat Completions conversion. Together, these yield the first independent reproduction of OpenAI's published scores: 60.4% on SWE Verified HIGH (published 60.7%), 53.3% MEDIUM (53.2%), and 91.7% on AIME25 with tools (90.4%). 

---
# Signals: Trajectory Sampling and Triage for Agentic Interactions 

**Authors**: Shuguang Chen, Adil Hafeez, Salman Paracha  

**Link**: [PDF](https://arxiv.org/pdf/2604.00356)  

**Abstract**: Agentic applications based on large language models increasingly rely on multi-step interaction loops involving planning, action execution, and environment feedback. While such systems are now deployed at scale, improving them post-deployment remains challenging. Agent trajectories are voluminous and non-deterministic, and reviewing each one, whether through human review or auxiliary LLMs, is slow and cost-prohibitive. We propose a lightweight, signal-based framework for triaging agentic interaction trajectories. Our approach computes cheap, broadly applicable signals from live interactions and attaches them as structured attributes for trajectory triage, identifying interactions likely to be informative without affecting online agent behavior. We organize signals into a coarse-grained taxonomy spanning interaction (misalignment, stagnation, disengagement, satisfaction), execution (failure, loop), and environment (exhaustion), designed for computation without model calls. In a controlled annotation study on $\tau$-bench, a widely used benchmark for tool-augmented agent evaluation, we show that signal-based sampling achieves an 82\% informativeness rate compared to 74\% for heuristic filtering and 54\% for random sampling, with a 1.52x efficiency gain per informative trajectory. The advantage is robust across reward strata and task domains, confirming that signals provide genuine per-trajectory informativeness gains rather than merely oversampling obvious failures. These results show that lightweight signals can serve as practical sampling infrastructure for agentic systems, and suggest a path toward preference data construction and post-deployment optimization. 

---
# Collaborative AI Agents and Critics for Fault Detection and Cause Analysis in Network Telemetry 

**Authors**: Syed Eqbal Alam, Zhan Shu  

**Link**: [PDF](https://arxiv.org/pdf/2604.00319)  

**Abstract**: We develop algorithms for collaborative control of AI agents and critics in a multi-actor, multi-critic federated multi-agent system. Each AI agent and critic has access to classical machine learning or generative AI foundation models. The AI agents and critics collaborate with a central server to complete multimodal tasks such as fault detection, severity, and cause analysis in a network telemetry system, text-to-image generation, video generation, healthcare diagnostics from medical images and patient records, etcetera. The AI agents complete their tasks and send them to AI critics for evaluation. The critics then send feedback to agents to improve their responses. Collaboratively, they minimize the overall cost to the system with no inter-agent or inter-critic communication. AI agents and critics keep their cost functions or derivatives of cost functions private. Using multi-time scale stochastic approximation techniques, we provide convergence guarantees on the time-average active states of AI agents and critics. The communication overhead is a little on the system, of the order of $\mathcal{O}(m)$, for $m$ modalities and is independent of the number of AI agents and critics. Finally, we present an example of fault detection, severity, and cause analysis in network telemetry and thorough evaluation to check the algorithm's efficacy. 

---
# Improvisational Games as a Benchmark for Social Intelligence of AI Agents: The Case of Connections 

**Authors**: Gaurav Rajesh Parikh, Angikar Ghosal  

**Link**: [PDF](https://arxiv.org/pdf/2604.00284)  

**Abstract**: We formally introduce a improvisational wordplay game called Connections to explore reasoning capabilities of AI agents. Playing Connections combines skills in knowledge retrieval, summarization and awareness of cognitive states of other agents. We show how the game serves as a good benchmark for social intelligence abilities of language model based agents that go beyond the agents' own memory and deductive reasoning and also involve gauging the understanding capabilities of other agents. Finally, we show how through communication with other agents in a constrained environment, AI agents must demonstrate social awareness and intelligence in games involving collaboration. 

---
# Human-in-the-Loop Control of Objective Drift in LLM-Assisted Computer Science Education 

**Authors**: Mark Dranias, Adam Whitley  

**Link**: [PDF](https://arxiv.org/pdf/2604.00281)  

**Abstract**: Large language models (LLMs) are increasingly embedded in computer science education through AI-assisted programming tools, yet such workflows often exhibit objective drift, in which locally plausible outputs diverge from stated task specifications. Existing instructional responses frequently emphasize tool-specific prompting practices, limiting durability as AI platforms evolve. This paper adopts a human-centered stance, treating human-in-the-loop (HITL) control as a stable educational problem rather than a transitional step toward AI autonomy. Drawing on systems engineering and control-theoretic concepts, we frame objectives and world models as operational artifacts that students configure to stabilize AI-assisted work. We propose a pilot undergraduate CS laboratory curriculum that explicitly separates planning from execution and trains students to specify acceptance criteria and architectural constraints prior to code generation. In selected labs, the curriculum also introduces deliberate, concept-aligned drift to support diagnosis and recovery from specification violations. We report a sensitivity power analysis for a three-arm pilot design comparing unstructured AI use, structured planning, and structured planning with injected drift, establishing detectable effect sizes under realistic section-level constraints. The contribution is a theory-driven, methodologically explicit foundation for HITL pedagogy that renders control competencies teachable across evolving AI tools. 

---
# A Safety-Aware Role-Orchestrated Multi-Agent LLM Framework for Behavioral Health Communication Simulation 

**Authors**: Ha Na Cho  

**Link**: [PDF](https://arxiv.org/pdf/2604.00249)  

**Abstract**: Single-agent large language model (LLM) systems struggle to simultaneously support diverse conversational functions and maintain safety in behavioral health communication. We propose a safety-aware, role-orchestrated multi-agent LLM framework designed to simulate supportive behavioral health dialogue through coordinated, role-differentiated agents. Conversational responsibilities are decomposed across specialized agents, including empathy-focused, action-oriented, and supervisory roles, while a prompt-based controller dynamically activates relevant agents and enforces continuous safety auditing. Using semi-structured interview transcripts from the DAIC-WOZ corpus, we evaluate the framework with scalable proxy metrics capturing structural quality, functional diversity, and computational characteristics. Results illustrate clear role differentiation, coherent inter-agent coordination, and predictable trade-offs between modular orchestration, safety oversight, and response latency when compared to a single-agent baseline. This work emphasizes system design, interpretability, and safety, positioning the framework as a simulation and analysis tool for behavioral health informatics and decision-support research rather than a clinical intervention. 

---
# Open, Reliable, and Collective: A Community-Driven Framework for Tool-Using AI Agents 

**Authors**: Hy Dang, Quang Dao, Meng Jiang  

**Link**: [PDF](https://arxiv.org/pdf/2604.00137)  

**Abstract**: Tool-integrated LLMs can retrieve, compute, and take real-world actions via external tools, but reliability remains a key bottleneck. We argue that failures stem from both tool-use accuracy (how well an agent invokes a tool) and intrinsic tool accuracy (the tool's own correctness), while most prior work emphasizes the former. We introduce OpenTools, a community-driven toolbox that standardizes tool schemas, provides lightweight plug-and-play wrappers, and evaluates tools with automated test suites and continuous monitoring. We also release a public web demo where users can run predefined agents and tools and contribute test cases, enabling reliability reports to evolve as tools change. OpenTools includes the core framework, an initial tool set, evaluation pipelines, and a contribution protocol. Experiments and evaluations show improved end-to-end reproducibility and task performance; community-contributed, higher-quality task-specific tools deliver 6%-22% relative gains over an existing toolbox across multiple agent architectures on downstream tasks and benchmarks, highlighting the importance of intrinsic tool accuracy. 

---
# One Panel Does Not Fit All: Case-Adaptive Multi-Agent Deliberation for Clinical Prediction 

**Authors**: Yuxing Lu, Yushuhong Lin, Jason Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2604.00085)  

**Abstract**: Large language models applied to clinical prediction exhibit case-level heterogeneity: simple cases yield consistent outputs, while complex cases produce divergent predictions under minor prompt changes. Existing single-agent strategies sample from one role-conditioned distribution, and multi-agent frameworks use fixed roles with flat majority voting, discarding the diagnostic signal in disagreement. We propose CAMP (Case-Adaptive Multi-agent Panel), where an attending-physician agent dynamically assembles a specialist panel tailored to each case's diagnostic uncertainty. Each specialist evaluates candidates via three-valued voting (KEEP/REFUSE/NEUTRAL), enabling principled abstention outside one's expertise. A hybrid router directs each diagnosis through strong consensus, fallback to the attending physician's judgment, or evidence-based arbitration that weighs argument quality over vote counts. On diagnostic prediction and brief hospital course generation from MIMIC-IV across four LLM backbones, CAMP consistently outperforms strong baselines while consuming fewer tokens than most competing multi-agent methods, with voting records and arbitration traces offering transparent decision audits. 

---
# How Emotion Shapes the Behavior of LLMs and Agents: A Mechanistic Study 

**Authors**: Moran Sun, Tianlin Li, Yuwei Zheng, Zhenhong Zhou, Aishan Liu, Xianglong Liu, Yang Liu  

**Link**: [PDF](https://arxiv.org/pdf/2604.00005)  

**Abstract**: Emotion plays an important role in human cognition and performance. Motivated by this, we investigate whether analogous emotional signals can shape the behavior of large language models (LLMs) and agents. Existing emotion-aware studies mainly treat emotion as a surface-level style factor or a perception target, overlooking its mechanistic role in task processing. To address this limitation, we propose E-STEER, an interpretable emotion steering framework that enables direct representation-level intervention in LLMs and agents. It embeds emotion as a structured, controllable variable in hidden states, and with it, we examine the impact of emotion on objective reasoning, subjective generation, safety, and multi-step agent behaviors. The results reveal non-monotonic emotion-behavior relations consistent with established psychological theories, and show that specific emotions not only enhance LLM capability but also improve safety, and systematically shape multi-step agent behaviors. 

---
# LAtent Phase Inference from Short time sequences using SHallow REcurrent Decoders (LAPIS-SHRED) 

**Authors**: Yuxuan Bao, Xingyue Zhang, J. Nathan Kutz  

**Link**: [PDF](https://arxiv.org/pdf/2604.01216)  

**Abstract**: Reconstructing full spatio-temporal dynamics from sparse observations in both space and time remains a central challenge in complex systems, as measurements can be spatially incomplete and can be also limited to narrow temporal windows. Yet approximating the complete spatio-temporal trajectory is essential for mechanistic insight and understanding, model calibration, and operational decision-making. We introduce LAPIS-SHRED (LAtent Phase Inference from Short time sequence using SHallow REcurrent Decoders), a modular architecture that reconstructs and/or forecasts complete spatiotemporal dynamics from sparse sensor observations confined to short temporal windows. LAPIS-SHRED operates through a three-stage pipeline: (i) a SHRED model is pre-trained entirely on simulation data to map sensor time-histories into a structured latent space, (ii) a temporal sequence model, trained on simulation-derived latent trajectories, learns to propagate latent states forward or backward in time to span unobserved temporal regions from short observational time windows, and (iii) at deployment, only a short observation window of hyper-sparse sensor measurements from the true system is provided, from which the frozen SHRED model and the temporal model jointly reconstruct or forecast the complete spatiotemporal trajectory. The framework supports bidirectional inference, inherits data assimilation and multiscale reconstruction capabilities from its modular structure, and accommodates extreme observational constraints including single-frame terminal inputs. We evaluate LAPIS-SHRED on six experiments spanning complex spatio-temporal physics: turbulent flows, multiscale propulsion physics, volatile combustion transients, and satellite-derived environmental fields, highlighting a lightweight, modular architecture suited for operational settings where observation is constrained by physical or logistical limitations. 

---
# The Recipe Matters More Than the Kitchen:Mathematical Foundations of the AI Weather Prediction Pipeline 

**Authors**: Piyush Garg, Diana R. Gergel, Andrew E. Shao, Galen J. Yacalis  

**Link**: [PDF](https://arxiv.org/pdf/2604.01215)  

**Abstract**: AI weather prediction has advanced rapidly, yet no unified mathematical framework explains what determines forecast skill. Existing theory addresses specific architectural choices rather than the learning pipeline as a whole, while operational evidence from 2023-2026 demonstrates that training methodology, loss function design, and data diversity matter at least as much as architecture selection. This paper makes two interleaved contributions. Theoretically, we construct a framework rooted in approximation theory on the sphere, dynamical systems theory, information theory, and statistical learning theory that treats the complete learning pipeline (architecture, loss function, training strategy, data distribution) rather than architecture alone. We establish a Learning Pipeline Error Decomposition showing that estimation error (loss- and data-dependent) dominates approximation error (architecture-dependent) at current scales. We develop a Loss Function Spectral Theory formalizing MSE-induced spectral blurring in spherical harmonic coordinates, and derive Out-of-Distribution Extrapolation Bounds proving that data-driven models systematically underestimate record-breaking extremes with bias growing linearly in record exceedance. Empirically, we validate these predictions via inference across ten architecturally diverse AI weather models using NVIDIA Earth2Studio with ERA5 initial conditions, evaluating six metrics across 30 initialization dates spanning all seasons. Results confirm universal spectral energy loss at high wavenumbers for MSE-trained models, rising Error Consensus Ratios showing that the majority of forecast error is shared across architectures, and linear negative bias during extreme events. A Holistic Model Assessment Score provides unified multi-dimensional evaluation, and a prescriptive framework enables mathematical evaluation of proposed pipelines before training. 

---
# $\texttt{YC-Bench}$: Benchmarking AI Agents for Long-Term Planning and Consistent Execution 

**Authors**: Muyu He, Adit Jain, Anand Kumar, Vincent Tu, Soumyadeep Bakshi, Sachin Patro, Nazneen Rajani  

**Link**: [PDF](https://arxiv.org/pdf/2604.01212)  

**Abstract**: As LLM agents tackle increasingly complex tasks, a critical question is whether they can maintain strategic coherence over long horizons: planning under uncertainty, learning from delayed feedback, and adapting when early mistakes compound. We introduce $\texttt{YC-Bench}$, a benchmark that evaluates these capabilities by tasking an agent with running a simulated startup over a one-year horizon spanning hundreds of turns. The agent must manage employees, select task contracts, and maintain profitability in a partially observable environment where adversarial clients and growing payroll create compounding consequences for poor decisions. We evaluate 12 models, both proprietary and open source, across 3 seeds each. Only three models consistently surpass the starting capital of \$200K, with Claude Opus 4.6 achieving the highest average final funds at \$1.27 M, followed by GLM-5 at \$1.21 M at 11$\times$ lower inference cost. Scratchpad usage, the sole mechanism for persisting information across context truncation, is the strongest predictor of success, and adversarial client detection is the primary failure mode, accounting for $47\%$ of bankruptcies. Our analysis reveals that frontier models still fail through distinct failure modes such as over-parallelization, demonstrating the capability gaps for long-horizon performance. $\texttt{YC-Bench}$ is open-source, reproducible, and configurable. 

---
# CliffSearch: Structured Agentic Co-Evolution over Theory and Code for Scientific Algorithm Discovery 

**Authors**: Youssef Mroueh, Carlos Fonseca, Brian Belgodere, David Cox  

**Link**: [PDF](https://arxiv.org/pdf/2604.01210)  

**Abstract**: Scientific algorithm discovery is iterative: hypotheses are proposed, implemented, stress-tested, and revised. Current LLM-guided search systems accelerate proposal generation, but often under-represent scientific structure by optimizing code-only artifacts with weak correctness/originality gating. We present CliffSearch, an agentic evolutionary framework in which the core evolution operators (pair selection, crossover, mutation, and review) are implemented as LLM agents, and the loop is designed around three principles: (1) each node is a structured scientific artifact, instantiated in either theory+code or code_only mode, (2) reviewer judgments of correctness and originality are first-class selection gates alongside optimization of the benchmark metric of interest, and (3) mutation is split into exploration and correction pathways with distinct objectives. Exploration mutation imports ideas from adjacent scientific domains to increase novelty, while correction mutation performs targeted evidence-guided repair using reviewer signals over theory, code, benchmark results, and runtime errors. We illustrate the framework on three benchmark-grounded studies: transformer hyper-connection evolution, optimizer discovery on a fixed nanoGPT stack, and a smaller native-optimizer ablation. Across these settings, the same loop supports explicit metric direction, reproducible persistence, and reviewer-gated comparison of discoveries under controlled search conditions. The result is a discovery workflow that prioritizes scientific interpretability and correctness while optimizing task metrics under controlled novelty constraints, rather than maximizing candidate throughput alone. Full run artifacts, interactive visualizations, and exported best nodes for the reported studies are available at this https URL . 

---
# Neural Harmonic Textures for High-Quality Primitive Based Neural Reconstruction 

**Authors**: Jorge Condor, Nicolas Moenne-Loccoz, Merlin Nimier-David, Piotr Didyk, Zan Gojcic, Qi Wu  

**Link**: [PDF](https://arxiv.org/pdf/2604.01204)  

**Abstract**: Primitive-based methods such as 3D Gaussian Splatting have recently become the state-of-the-art for novel-view synthesis and related reconstruction tasks. Compared to neural fields, these representations are more flexible, adaptive, and scale better to large scenes. However, the limited expressivity of individual primitives makes modeling high-frequency detail challenging. We introduce Neural Harmonic Textures, a neural representation approach that anchors latent feature vectors on a virtual scaffold surrounding each primitive. These features are interpolated within the primitive at ray intersection points. Inspired by Fourier analysis, we apply periodic activations to the interpolated features, turning alpha blending into a weighted sum of harmonic components. The resulting signal is then decoded in a single deferred pass using a small neural network, significantly reducing computational cost. Neural Harmonic Textures yield state-of-the-art results in real-time novel view synthesis while bridging the gap between primitive- and neural-field-based reconstruction. Our method integrates seamlessly into existing primitive-based pipelines such as 3DGUT, Triangle Splatting, and 2DGS. We further demonstrate its generality with applications to 2D image fitting and semantic reconstruction. 

---
# ORBIT: Scalable and Verifiable Data Generation for Search Agents on a Tight Budget 

**Authors**: Nandan Thakur, Zijian Chen, Xueguang Ma, Jimmy Lin  

**Link**: [PDF](https://arxiv.org/pdf/2604.01195)  

**Abstract**: Search agents, which integrate language models (LMs) with web search, are becoming crucial for answering complex user queries. Constructing training datasets for deep research tasks, involving multi-step retrieval and reasoning, remains challenging due to expensive human annotation, or cumbersome prerequisites. In this work, we introduce ORBIT, a training dataset with 20K reasoning-intensive queries with short verifiable answers, generated using a frugal framework without relying on paid API services. The modular framework relies on four stages: seed creation, question--answer pair generation, and two stages of verification: self and external. ORBIT spans 15 domains and each training pair requires 4--5 reasoning steps, with external search verification required from the complete web. We train Qwen3-4B as the base model on ORBIT using GRPO and evaluate it on Wikipedia question answering tasks. Extensive experiment results demonstrate that ORBIT-4B achieves strong performance among sub-4B LLMs as search agents, proving the utility of synthetic datasets. Our framework, code and datasets are open-sourced and available publicly. 

---
# A ROS 2 Wrapper for Florence-2: Multi-Mode Local Vision-Language Inference for Robotic Systems 

**Authors**: J. E. Domínguez-Vidal  

**Link**: [PDF](https://arxiv.org/pdf/2604.01179)  

**Abstract**: Foundation vision-language models are becoming increasingly relevant to robotics because they can provide richer semantic perception than narrow task-specific pipelines. However, their practical adoption in robot software stacks still depends on reproducible middleware integrations rather than on model quality alone. Florence-2 is especially attractive in this regard because it unifies captioning, optical character recognition, open-vocabulary detection, grounding and related vision-language tasks within a comparatively manageable model size. This article presents a ROS 2 wrapper for Florence-2 that exposes the model through three complementary interaction modes: continuous topic-driven processing, synchronous service calls and asynchronous actions. The wrapper is designed for local execution and supports both native installation and Docker container deployment. It also combines generic JSON outputs with standard ROS 2 message bindings for detection-oriented tasks. A functional validation is reported together with a throughput study on several GPUs, showing that local deployment is feasible with consumer grade hardware. The repository is publicly available here: this https URL 

---
# Screening Is Enough 

**Authors**: Ken M. Nakanishi  

**Link**: [PDF](https://arxiv.org/pdf/2604.01178)  

**Abstract**: A core limitation of standard softmax attention is that it does not define a notion of absolute query--key relevance: attention weights are obtained by redistributing a fixed unit mass across all keys according to their relative scores. As a result, relevance is defined only relative to competing keys, and irrelevant keys cannot be explicitly rejected. We introduce Multiscreen, a language-model architecture built around a mechanism we call screening, which enables absolute query--key relevance. Instead of redistributing attention across all keys, screening evaluates each key against an explicit threshold, discarding irrelevant keys and aggregating the remaining keys, thereby removing global competition among keys. Across experiments, Multiscreen achieves comparable validation loss with approximately 40% fewer parameters than a Transformer baseline, enables stable optimization at substantially larger learning rates, maintains strong performance in long-context perplexity, shows little to no degradation in retrieval performance even far beyond the training context length, and reduces inference latency by up to 3.2$\times$ at 100K context length. 

---
# Online Reasoning Calibration: Test-Time Training Enables Generalizable Conformal LLM Reasoning 

**Authors**: Cai Zhou, Zekai Wang, Menghua Wu, Qianyu Julie Zhu, Flora C. Shi, Chenyu Wang, Ashia Wilson, Tommi Jaakkola, Stephen Bates  

**Link**: [PDF](https://arxiv.org/pdf/2604.01170)  

**Abstract**: While test-time scaling has enabled large language models to solve highly difficult tasks, state-of-the-art results come at exorbitant compute costs. These inefficiencies can be attributed to the miscalibration of post-trained language models, and the lack of calibration in popular sampling techniques. Here, we present Online Reasoning Calibration (ORCA), a framework for calibrating the sampling process that draws upon conformal prediction and test-time training. Specifically, we introduce a meta-learning procedure that updates the calibration module for each input. This allows us to provide valid confidence estimates under distributional shift, e.g. in thought patterns that occur across different stages of reasoning, or in prompt distributions between model development and deployment. ORCA not only provides theoretical guarantees on conformal risks, but also empirically shows higher efficiency and generalization across different reasoning tasks. At risk level $\delta=0.1$, ORCA improves Qwen2.5-32B efficiency on in-distribution tasks with savings up to 47.5% with supervised labels and 40.7% with self-consistency labels. Under zero-shot out-of-domain settings, it improves MATH-500 savings from 24.8% of the static calibration baseline to 67.0% while maintaining a low empirical error rate, and the same trend holds across model families and downstream benchmarks. Our code is publicly available at this https URL. 

---
# AdaLoRA-QAT: Adaptive Low-Rank and Quantization-Aware Segmentation 

**Authors**: Prantik Deb, Srimanth Dhondy, N. Ramakrishna, Anu Kapoor, Raju S. Bapi, Tapabrata Chakraborti  

**Link**: [PDF](https://arxiv.org/pdf/2604.01167)  

**Abstract**: Chest X-ray (CXR) segmentation is an important step in computer-aided diagnosis, yet deploying large foundation models in clinical settings remains challenging due to computational constraints. We propose AdaLoRA-QAT, a two-stage fine-tuning framework that combines adaptive low-rank encoder adaptation with full quantization-aware training. Adaptive rank allocation improves parameter efficiency, while selective mixed-precision INT8 quantization preserves structural fidelity crucial for clinical reliability. Evaluated across large-scale CXR datasets, AdaLoRA-QAT achieves 95.6% Dice, matching full-precision SAM decoder fine-tuning while reducing trainable parameters by 16.6\times and yielding 2.24\times model compression. A Wilcoxon signed-rank test confirms that quantization does not significantly degrade segmentation accuracy. These results demonstrate that AdaLoRA-QAT effectively balances accuracy, efficiency, and structural trust-worthiness, enabling compact and deployable foundation models for medical image segmentation. Code and pretrained models are available at: this https URL 

---
# Brainstacks: Cross-Domain Cognitive Capabilities via Frozen MoE-LoRA Stacks for Continual LLM Learning 

**Authors**: Mohammad R. Abu Ayyash  

**Link**: [PDF](https://arxiv.org/pdf/2604.01152)  

**Abstract**: We present Brainstacks, a modular architecture for continual multi-domain fine-tuning of large language models that packages domain expertise as frozen adapter stacks composing additively on a shared frozen base at inference. Five interlocking components: (1) MoE-LoRA with Shazeer-style noisy top-2 routing across all seven transformer projections under QLoRA 4-bit quantization with rsLoRA scaling; (2) an inner loop performing residual boosting by freezing trained stacks and adding new ones; (3) an outer loop training sequential domain-specific stacks with curriculum-ordered dependencies; (4) null-space projection via randomized SVD constraining new stacks to subspaces orthogonal to prior directions, achieving zero forgetting in isolation; (5) an outcome-based sigmoid meta-router trained on empirically discovered domain-combination targets that selectively weights stacks, enabling cross-domain composition. Two boundary experiments: (6) PSN pretraining on a randomly initialized model; (7) per-domain RL (DPO/GRPO) validating compatibility with post-SFT alignment. Validated on TinyLlama-1.1B (4 domains, 9 stacks) and Gemma 3 12B IT (5 domains, 10 stacks), MoE-LoRA achieves 2.5x faster convergence than parameter-matched single LoRA, residual boosting breaks through the single-stack ceiling, and the routed system recovers generation quality destroyed by ungated stack accumulation. The central finding: the outcome-based router discovers that domain stacks encode transferable cognitive primitives (instruction-following clarity, numerical reasoning, procedural logic, chain-of-thought structure) rather than domain-specific knowledge, with medical prompts routing to chat+math stacks in 97% of cases despite zero medical data in those stacks. 

---
# Looking into a Pixel by Nonlinear Unmixing -- A Generative Approach 

**Authors**: Maofeng Tang, Hairong Qi  

**Link**: [PDF](https://arxiv.org/pdf/2604.01141)  

**Abstract**: Due to the large footprint of pixels in remote sensing imagery, hyperspectral unmixing (HU) has become an important and necessary procedure in hyperspectral image analysis. Traditional HU methods rely on a prior spectral mixing model, especially for nonlinear mixtures, which has largely limited the performance and generalization capacity of the unmixing approach. In this paper, we address the challenging problem of hyperspectral nonlinear unmixing (HNU) without explicit knowledge of the mixing model. Inspired by the principle of generative models, where images of the same distribution can be generated as that of the training images without knowing the exact probability distribution function of the image, we develop an invertible mixing-unmixing process via a bi-directional GAN framework, constrained by both the cycle consistency and the linkage between linear and nonlinear mixtures. The combination of cycle consistency and linear linkage provides powerful constraints without requiring an explicit mixing model. We refer to the proposed approach as the linearly-constrained CycleGAN unmixing net, or LCGU net. Experimental results indicate that the proposed LCGU net exhibits stable and competitive performance across different datasets compared with other state-of-the-art model-based HNU methods. 

---
# Paper Reconstruction Evaluation: Evaluating Presentation and Hallucination in AI-written Papers 

**Authors**: Atsuyuki Miyai, Mashiro Toyooka, Zaiying Zhao, Kenta Watanabe, Toshihiko Yamasaki, Kiyoharu Aizawa  

**Link**: [PDF](https://arxiv.org/pdf/2604.01128)  

**Abstract**: This paper introduces the first systematic evaluation framework for quantifying the quality and risks of papers written by modern coding agents. While AI-driven paper writing has become a growing concern, rigorous evaluation of the quality and potential risks of AI-written papers remains limited, and a unified understanding of their reliability is still lacking. We introduce Paper Reconstruction Evaluation (PaperRecon), an evaluation framework in which an overview (this http URL) is created from an existing paper, after which an agent generates a full paper based on the overview and minimal additional resources, and the result is subsequently compared against the original paper. PaperRecon disentangles the evaluation of the AI-written papers into two orthogonal dimensions, Presentation and Hallucination, where Presentation is evaluated using a rubric and Hallucination is assessed via agentic evaluation grounded in the original paper source. For evaluation, we introduce PaperWrite-Bench, a benchmark of 51 papers from top-tier venues across diverse domains published after 2025. Our experiments reveal a clear trade-off: while both ClaudeCode and Codex improve with model advances, ClaudeCode achieves higher presentation quality at the cost of more than 10 hallucinations per paper on average, whereas Codex produces fewer hallucinations but lower presentation quality. This work takes a first step toward establishing evaluation frameworks for AI-driven paper writing and improving the understanding of its risks within the research community. 

---
# Lightweight Prompt-Guided CLIP Adaptation for Monocular Depth Estimation 

**Authors**: Reyhaneh Ahani Manghotay, Jie Liang  

**Link**: [PDF](https://arxiv.org/pdf/2604.01118)  

**Abstract**: Leveraging the rich semantic features of vision-language models (VLMs) like CLIP for monocular depth estimation tasks is a promising direction, yet often requires extensive fine-tuning or lacks geometric precision. We present a parameter-efficient framework, named MoA-DepthCLIP, that adapts pretrained CLIP representations for monocular depth estimation with minimal supervision. Our method integrates a lightweight Mixture-of-Adapters (MoA) module into the pretrained Vision Transformer (ViT-B/32) backbone combined with selective fine-tuning of the final layers. This design enables spatially-aware adaptation, guided by a global semantic context vector and a hybrid prediction architecture that synergizes depth bin classification with direct regression. To enhance structural accuracy, we employ a composite loss function that enforces geometric constraints. On the NYU Depth V2 benchmark, MoA-DepthCLIP achieves competitive results, significantly outperforming the DepthCLIP baseline by improving the $\delta_1$ accuracy from 0.390 to 0.745 and reducing the RMSE from 1.176 to 0.520. These results are achieved while requiring substantially few trainable parameters, demonstrating that lightweight, prompt-guided MoA is a highly effective strategy for transferring VLM knowledge to fine-grained monocular depth estimation tasks. 

---
# Trust and Reliance on AI in Education: AI Literacy and Need for Cognition as Moderators 

**Authors**: Griffin Pitts, Neha Rani, Weedguet Mildort  

**Link**: [PDF](https://arxiv.org/pdf/2604.01114)  

**Abstract**: As generative AI systems are integrated into educational settings, students often encounter AI-generated output while working through learning tasks, either by requesting help or through integrated tools. Trust in AI can influence how students interpret and use that output, including whether they evaluate it critically or exhibit overreliance. We investigate how students' trust relates to their appropriate reliance on an AI assistant during programming problem-solving tasks, and whether this relationship differs by learner characteristics. With 432 undergraduate participants, students' completed Python output-prediction problems while receiving recommendations and explanations from an AI chatbot, including accurate and intentionally misleading suggestions. We operationalize reliance behaviorally as the extent to which students' responses reflected appropriate use of the AI assistant's suggestions, accepting them when they were correct and rejecting them when they were incorrect. Pre- and post-task surveys assessed trust in the assistant, AI literacy, need for cognition, programming self-efficacy, and programming literacy. Results showed a non-linear relationship in which higher trust was associated with lower appropriate reliance, suggesting weaker discrimination between correct and incorrect recommendations. This relationship was significantly moderated by students' AI literacy and need for cognition. These findings highlight the need for future work on instructional and system supports that encourage more reflective evaluation of AI assistance during problem-solving. 

---
# Approximating Pareto Frontiers in Stochastic Multi-Objective Optimization via Hashing and Randomization 

**Authors**: Jinzhao Li, Nan Jiang, Yexiang Xue  

**Link**: [PDF](https://arxiv.org/pdf/2604.01098)  

**Abstract**: Stochastic Multi-Objective Optimization (SMOO) is critical for decision-making trading off multiple potentially conflicting objectives in uncertain environments. SMOO aims at identifying the Pareto frontier, which contains all mutually non-dominating decisions. The problem is highly intractable due to the embedded probabilistic inference, such as computing the marginal, posterior probabilities, or expectations. Existing methods, such as scalarization, sample average approximation, and evolutionary algorithms, either offer arbitrarily loose approximations or may incur prohibitive computational costs. We propose XOR-SMOO, a novel algorithm that with probability $1-\delta$, obtains $\gamma$-approximate Pareto frontiers ($\gamma>1$) for SMOO by querying an SAT oracle poly-log times in $\gamma$ and $\delta$. A $\gamma$-approximate Pareto frontier is only below the true frontier by a fixed, multiplicative factor $\gamma$. Thus, XOR-SMOO solves highly intractable SMOO problems (\#P-hard) with only queries to SAT oracles while obtaining tight, constant factor approximation guarantees. Experiments on real-world road network strengthening and supply chain design problems demonstrate that XOR-SMOO outperforms several baselines in identifying Pareto frontiers that have higher objective values, better coverage of the optimal solutions, and the solutions found are more evenly distributed. Overall, XOR-SMOO significantly enhanced the practicality and reliability of SMOO solvers. 

---
# Temporal Dependencies in In-Context Learning: The Role of Induction Heads 

**Authors**: Anooshka Bajaj, Deven Mahesh Mistry, Sahaj Singh Maini, Yash Aggarwal, Billy Dickson, Zoran Tiganj  

**Link**: [PDF](https://arxiv.org/pdf/2604.01094)  

**Abstract**: Large language models (LLMs) exhibit strong in-context learning capabilities, but how they track and retrieve information from context remains underexplored. Drawing on the free recall paradigm in cognitive science (where participants recall list items in any order), we show that several open-source LLMs consistently display a serial-recall-like pattern, assigning peak probability to tokens that immediately follow a repeated token in the input sequence. Through systematic ablation experiments, we show that induction heads, specialized attention heads that attend to the token following a previous occurrence of the current token, play an important role in this phenomenon. Removing heads with a high induction score substantially reduces the +1 lag bias, whereas ablating random heads does not reproduce the same reduction. We also show that removing heads with high induction scores impairs the performance of models prompted to do serial recall using few-shot learning to a larger extent than removing random heads. Our findings highlight a mechanistically specific connection between induction heads and temporal context processing in transformers, suggesting that these heads are especially important for ordered retrieval and serial-recall-like behavior during in-context learning. 

---
# TRACE: Training-Free Partial Audio Deepfake Detection via Embedding Trajectory Analysis of Speech Foundation Models 

**Authors**: Awais Khan, Muhammad Umar Farooq, Kutub Uddin, Khalid Malik  

**Link**: [PDF](https://arxiv.org/pdf/2604.01083)  

**Abstract**: Partial audio deepfakes, where synthesized segments are spliced into genuine recordings, are particularly deceptive because most of the audio remains authentic. Existing detectors are supervised: they require frame-level annotations, overfit to specific synthesis pipelines, and must be retrained as new generative models emerge. We argue that this supervision is unnecessary. We hypothesize that speech foundation models implicitly encode a forensic signal: genuine speech forms smooth, slowly varying embedding trajectories, while splice boundaries introduce abrupt disruptions in frame-level transitions. Building on this, we propose TRACE (Training-free Representation-based Audio Countermeasure via Embedding dynamics), a training-free framework that detects partial audio deepfakes by analyzing the first-order dynamics of frozen speech foundation model representations without any training, labeled data, or architectural modification. We evaluate TRACE on four benchmarks that span two languages using six speech foundation models. In PartialSpoof, TRACE achieves 8.08% EER, competitive with fine-tuned supervised baselines. In LlamaPartialSpoof, the most challenging benchmark featuring LLM-driven commercial synthesis, TRACE surpasses a supervised baseline outright (24.12% vs. 24.49% EER) without any target-domain data. These results show that temporal dynamics in speech foundation models provide an effective, generalize signal for training-free audio forensics. 

---
# VibeGuard: A Security Gate Framework for AI-Generated Code 

**Authors**: Ying Xie  

**Link**: [PDF](https://arxiv.org/pdf/2604.01052)  

**Abstract**: "Vibe coding," in which developers delegate code generation to AI assistants and accept the output with little manual review, has gained rapid adoption in production settings. On March 31, 2026, Anthropic's Claude Code CLI shipped a 59.8 MB source map file in its npm package, exposing roughly 512,000 lines of proprietary TypeScript. The tool had itself been largely vibe-coded, and the leak traced to a misconfigured packaging rule rather than a logic bug. Existing static-analysis and secret-scanning tools did not cover this failure mode, pointing to a gap between the vulnerabilities AI tends to introduce and the vulnerabilities current tooling is built to find. We present VibeGuard, a pre-publish security gate that targets five such blind spots: artifact hygiene, packaging-configuration drift, source-map exposure, hardcoded secrets, and supply-chain risk. In controlled experiments on eight synthetic projects (seven vulnerable, one clean control), VibeGuard achieved 100% recall, 89.47% precision (F1 = 94.44%), and correct pass/fail gate decisions on all eight projects across three policy levels. We discuss how these results inform a defense-in-depth workflow for teams that rely on AI code generation. 

---
# Adversarial Attacks in AI-Driven RAN Slicing: SLA Violations and Recovery 

**Authors**: Deemah H. Tashman, Soumaya Cherkaoui  

**Link**: [PDF](https://arxiv.org/pdf/2604.01049)  

**Abstract**: Next-generation (NextG) cellular networks are designed to support emerging applications with diverse data rate and latency requirements, such as immersive multimedia services and large-scale Internet of Things deployments. A key enabling mechanism is radio access network (RAN) slicing, which dynamically partitions radio resources into virtual resource blocks to efficiently serve heterogeneous traffic classes, including enhanced mobile broadband (eMBB), massive machine-type communications (mMTC), and ultra-reliable low-latency communications (URLLC). In this paper, we study the impact of adversarial attacks on AI-driven RAN slicing decisions, where a budget-constrained adversary selectively jams slice transmissions to bias deep reinforcement learning (DRL)-based resource allocation, and quantify the resulting service level agreement (SLA) violations and post-attack recovery behavior. Our results indicate that budget-constrained adversarial jamming can induce severe and slice-dependent steady-state SLA violations. Moreover, the DRL agent's reward converges toward the clean baseline only after a non-negligible recovery period. 

---
# Automated Framework to Evaluate and Harden LLM System Instructions against Encoding Attacks 

**Authors**: Anubhab Sahu, Diptisha Samanta, Reza Soosahabi  

**Link**: [PDF](https://arxiv.org/pdf/2604.01039)  

**Abstract**: System Instructions in Large Language Models (LLMs) are commonly used to enforce safety policies, define agent behavior, and protect sensitive operational context in agentic AI applications. These instructions may contain sensitive information such as API credentials, internal policies, and privileged workflow definitions, making system instruction leakage a critical security risk highlighted in the OWASP Top 10 for LLM Applications. Without incurring the overhead costs of reasoning models, many LLM applications rely on refusal-based instructions that block direct requests for system instructions, implicitly assuming that prohibited information can only be extracted through explicit queries. We introduce an automated evaluation framework that tests whether system instructions remain confidential when extraction requests are re-framed as encoding or structured output tasks. Across four common models and 46 verified system instructions, we observe high attack success rates (> 0.7) for structured serialization where models refuse direct extraction requests but disclose protected content in the requested serialization formats. We further demonstrate a mitigation strategy based on one-shot instruction reshaping using a Chain-of-Thought reasoning model, indicating that even subtle changes in wording and structure of system instructions can significantly reduce attack success rate without requiring model retraining. 

---
# Aligning Recommendations with User Popularity Preferences 

**Authors**: Mona Schirmer, Anton Thielmann, Pola Schwöbel, Thomas Martynec, Giuseppe Di Benedetto, Ben London, Yannik Stein  

**Link**: [PDF](https://arxiv.org/pdf/2604.01036)  

**Abstract**: Popularity bias is a pervasive problem in recommender systems, where recommendations disproportionately favor popular items. This not only results in "rich-get-richer" dynamics and a homogenization of visible content, but can also lead to misalignment of recommendations with individual users' preferences for popular or niche content. This work studies popularity bias through the lens of user-recommender alignment. To this end, we introduce Popularity Quantile Calibration, a measurement framework that quantifies misalignment between a user's historical popularity preference and the popularity of their recommendations. Building on this notion of popularity alignment, we propose SPREE, an inference-time mitigation method for sequential recommenders based on activation steering. SPREE identifies a popularity direction in representation space and adaptively steers model activations based on an estimate of each user's personal popularity bias, allowing both the direction and magnitude of steering to vary across users. Unlike global debiasing approaches, SPREE explicitly targets alignment rather than uniformly reducing popularity. Experiments across multiple datasets show that SPREE consistently improves user-level popularity alignment while preserving recommendation quality. 

---
# Revision or Re-Solving? Decomposing Second-Pass Gains in Multi-LLM Pipelines 

**Authors**: Jingjie Ning, Xueqi Li, Chengyu Yu  

**Link**: [PDF](https://arxiv.org/pdf/2604.01029)  

**Abstract**: Multi-LLM revision pipelines, in which a second model reviews and improves a draft produced by a first, are widely assumed to derive their gains from genuine error correction. We question this assumption with a controlled decomposition experiment that uses four matched conditions to separate second-pass gains into three additive components: re-solving, scaffold, and content. We evaluate this design across two model pairs on three benchmarks spanning knowledge-intensive MCQ and competitive programming. Our results show that the gains of multi-LLM revision are not monolithic, but depend on task structure, draft quality, and the type of draft information. On MCQ tasks, where the answer space is constrained and drafts provide little structural guidance, most gains are consistent with stronger-model re-solving, and directly routing queries to the stronger model can be more effective than revising a weak draft. On code generation tasks, however, two-stage prompting remains useful because even semantically null drafts can provide substantial structural scaffolding, while weak draft content can be harmful. Finally, role-reversed experiments show that strong drafts clearly benefit weak reviewers. Ultimately, our findings demonstrate that the utility of multi-LLM revision is dynamically bottlenecked by task structure and draft quality, necessitating more targeted pipeline designs rather than blanket revision strategies. 

---
# Fast and Accurate Probing of In-Training LLMs' Downstream Performances 

**Authors**: Zhichen Liu, Tianle Lun, Zhibin Wen, Hao An, Yulin Ou, Jianhui Xu, Hao Zhang, Wenyi Fang, Yang Zheng, Yang Xu  

**Link**: [PDF](https://arxiv.org/pdf/2604.01025)  

**Abstract**: The paradigm of scaling Large Language Models (LLMs) in both parameter size and test time has pushed the boundaries of AI capabilities, but at the cost of making the traditional generative evaluation paradigm prohibitively expensive, therefore making the latency of LLM's in-training downstream performance evaluation unbearable. However, simple metrics like training loss (perplexity) are not always correlated with downstream performance, as sometimes their trends diverge from the actual task outcomes. This dilemma calls for a method that is computationally efficient and sufficiently accurate in measuring model capabilities. To address this challenge, we introduce a new in-training evaluation paradigm that uses a lightweight probe for monitoring downstream performance. The probes take the internal representations of LLM checkpoints (during training) as input and directly predict the checkpoint's performance on downstream tasks measured by success probability (i.e., pass@1). We design several probe architectures, validating their effectiveness using the OLMo3-7B's checkpoints across a diverse set of downstream tasks. The probes can accurately predict a checkpoint's performance (with avg. AUROC$>$0.75), have decent generalizability across checkpoints (earlier predicts later), and reduce the computation latency from $\sim$1 hr (using conventional generative evaluation method) to $\sim$3 min. In sum, this work presents a practical and scalable in-training downstream evaluation paradigm, enabling a more agile, informed, and efficient LLM development process. 

---
# Transfer learning for nonparametric Bayesian networks 

**Authors**: Rafael Sojo, Pedro Larrañaga, Concha Bielza  

**Link**: [PDF](https://arxiv.org/pdf/2604.01021)  

**Abstract**: This paper introduces two transfer learning methodologies for estimating nonparametric Bayesian networks under scarce data. We propose two algorithms, a constraint-based structure learning method, called PC-stable-transfer learning (PCS-TL), and a score-based method, called hill climbing transfer learning (HC-TL). We also define particular metrics to tackle the negative transfer problem in each of them, a situation in which transfer learning has a negative impact on the model's performance. Then, for the parameters, we propose a log-linear pooling approach. For the evaluation, we learn kernel density estimation Bayesian networks, a type of nonparametric Bayesian network, and compare their transfer learning performance with the models alone. To do so, we sample data from small, medium and large-sized synthetic networks and datasets from the UCI Machine Learning repository. Then, we add noise and modifications to these datasets to test their ability to avoid negative transfer. To conclude, we perform a Friedman test with a Bergmann-Hommel post-hoc analysis to show statistical proof of the enhanced experimental behavior of our methods. Thus, PCS-TL and HC-TL demonstrate to be reliable algorithms for improving the learning performance of a nonparametric Bayesian network with scarce data, which in real industrial environments implies a reduction in the required time to deploy the network. 

---
# OrgAgent: Organize Your Multi-Agent System like a Company 

**Authors**: Yiru Wang, Xinyue Shen, Yaohui Han, Michael Backes, Pin-Yu Chen, Tsung-Yi Ho  

**Link**: [PDF](https://arxiv.org/pdf/2604.01020)  

**Abstract**: While large language model-based multi-agent systems have shown strong potential for complex reasoning, how to effectively organize multiple agents remains an open question. In this paper, we introduce OrgAgent, a company-style hierarchical multi-agent framework that separates collaboration into governance, execution, and compliance layers. OrgAgent decomposes multi-agent reasoning into three layers: a governance layer for planning and resource allocation, an execution layer for task solving and review, and a compliance layer for final answer control. By evaluating the framework across reasoning tasks, LLMs, execution modes, and execution policies, we find that multi-agent systems organized in a company-style hierarchy generally outperform other organizational structures. Besides, hierarchical coordination also reduces token consumption relative to flat collaboration in most settings. For example, for GPT-OSS-120B, the hierarchical setting improves performance over flat multi-agent system by 102.73% while reducing token usage by 74.52% on SQuAD 2.0. Further analysis shows that hierarchy helps most when tasks benefit from stable skill assignment, controlled information flow, and layered verification. Overall, our findings highlight organizational structure as an important factor in multi-agent reasoning, shaping not only effectiveness and cost, but also coordination behavior. 

---
# Query-Conditioned Evidential Keyframe Sampling for MLLM-Based Long-Form Video Understanding 

**Authors**: Yiheng Wang, Lichen Zhu, Yueqian Lin, Yudong Liu, Jingyang Zhang, Hai "Helen" Li, Yiran Chen  

**Link**: [PDF](https://arxiv.org/pdf/2604.01002)  

**Abstract**: Multimodal Large Language Models (MLLMs) have shown strong performance on video question answering, but their application to long-form videos is constrained by limited context length and computational cost, making keyframe sampling essential. Existing approaches typically rely on semantic relevance or reinforcement learning, which either fail to capture evidential clues or suffer from inefficient combinatorial optimization. In this work, we propose an evidence-driven keyframe sampling framework grounded in information bottleneck theory. We formulate keyframe selection as maximizing the conditional mutual information between selected frames and the query, providing a principled objective that reflects each frame's contribution to answering the question. To make this objective tractable, we exploit its structure to derive a decomposed optimization that reduces subset selection to independent frame-level scoring. We further introduce a query-conditioned evidence scoring network trained with a contrastive objective to estimate evidential importance efficiently. Experiments on long-form video understanding benchmarks show that our method consistently outperforms prior sampling strategies under strict token budgets, while significantly improving training efficiency. 

---
# EgoSim: Egocentric World Simulator for Embodied Interaction Generation 

**Authors**: Jinkun Hao, Mingda Jia, Ruiyan Wang, Xihui Liu, Ran Yi, Lizhuang Ma, Jiangmiao Pang, Xudong Xu  

**Link**: [PDF](https://arxiv.org/pdf/2604.01001)  

**Abstract**: We introduce EgoSim, a closed-loop egocentric world simulator that generates spatially consistent interaction videos and persistently updates the underlying 3D scene state for continuous simulation. Existing egocentric simulators either lack explicit 3D grounding, causing structural drift under viewpoint changes, or treat the scene as static, failing to update world states across multi-stage interactions. EgoSim addresses both limitations by modeling 3D scenes as updatable world states. We generate embodiment interactions via a Geometry-action-aware Observation Simulation model, with spatial consistency from an Interaction-aware State Updating module. To overcome the critical data bottleneck posed by the difficulty in acquiring densely aligned scene-interaction training pairs, we design a scalable pipeline that extracts static point clouds, camera trajectories, and embodiment actions from in-the-wild large-scale monocular egocentric videos. We further introduce EgoCap, a capture system that enables low-cost real-world data collection with uncalibrated smartphones. Extensive experiments demonstrate that EgoSim significantly outperforms existing methods in terms of visual quality, spatial consistency, and generalization to complex scenes and in-the-wild dexterous interactions, while supporting cross-embodiment transfer to robotic manipulation. Codes and datasets will be open soon. The project page is at this http URL. 

---
# Multimodal Analysis of State-Funded News Coverage of the Israel-Hamas War on YouTube Shorts 

**Authors**: Daniel Miehling, Sandra Kuebler  

**Link**: [PDF](https://arxiv.org/pdf/2604.00994)  

**Abstract**: YouTube Shorts have become central to news consumption on the platform, yet research on how geopolitical events are represented in this format remains limited. To address this gap, we present a multimodal pipeline that combines automatic transcription, aspect-based sentiment analysis (ABSA), and semantic scene classification. The pipeline is first assessed for feasibility and then applied to analyze short-form coverage of the Israel-Hamas war by state-funded outlets. Using over 2,300 conflict-related Shorts and more than 94,000 visual frames, we systematically examine war reporting across major international broadcasters. Our findings reveal that the sentiment expressed in transcripts regarding specific aspects differs across outlets and over time, whereas scene-type classifications reflect visual cues consistent with real-world events. Notably, smaller domain-adapted models outperform large transformers and even LLMs for sentiment analysis, underscoring the value of resource-efficient approaches for humanities research. The pipeline serves as a template for other short-form platforms, such as TikTok and Instagram, and demonstrates how multimodal methods, combined with qualitative interpretation, can characterize sentiment patterns and visual cues in algorithmically driven video environments. 

---
# Bridging Structured Knowledge and Data: A Unified Framework with Finance Applications 

**Authors**: Yi Cao, Zexun Chen, Lin William Cong, Heqing Shi  

**Link**: [PDF](https://arxiv.org/pdf/2604.00987)  

**Abstract**: We develop Structured-Knowledge-Informed Neural Networks (SKINNs), a unified estimation framework that embeds theoretical, simulated, previously learned, or cross-domain insights as differentiable constraints within flexible neural function approximation. SKINNs jointly estimate neural network parameters and economically meaningful structural parameters in a single optimization problem, enforcing theoretical consistency not only on observed data but over a broader input domain through collocation, and therefore nesting approaches such as functional GMM, Bayesian updating, transfer learning, PINNs, and surrogate modeling. SKINNs define a class of M-estimators that are consistent and asymptotically normal with root-N convergence, sandwich covariance, and recovery of pseudo-true parameters under misspecification. We establish identification of structural parameters under joint flexibility, derive generalization and target-risk bounds under distributional shift in a convex proxy, and provide a restricted-optimal characterization of the weighting parameter that governs the bias-variance tradeoff. In an illustrative financial application to option pricing, SKINNs improve out-of-sample valuation and hedging performance, particularly at longer horizons and during high-volatility regimes, while recovering economically interpretable structural parameters with improved stability relative to conventional calibration. More broadly, SKINNs provide a general econometric framework for combining model-based reasoning with high-dimensional, data-driven estimation. 

---
# Do Phone-Use Agents Respect Your Privacy? 

**Authors**: Zhengyang Tang, Ke Ji, Xidong Wang, Zihan Ye, Xinyuan Wang, Yiduo Guo, Ziniu Li, Chenxin Li, Jingyuan Hu, Shunian Chen, Tongxu Luo, Jiaxi Bi, Zeyu Qin, Shaobo Wang, Xin Lai, Pengyuan Lyu, Junyi Li, Can Xu, Chengquan Zhang, Han Hu, Ming Yan, Benyou Wang  

**Link**: [PDF](https://arxiv.org/pdf/2604.00986)  

**Abstract**: We study whether phone-use agents respect privacy while completing benign mobile tasks. This question has remained hard to answer because privacy-compliant behavior is not operationalized for phone-use agents, and ordinary apps do not reveal exactly what data agents type into which form entries during execution. To make this question measurable, we introduce MyPhoneBench, a verifiable evaluation framework for privacy behavior in mobile agents. We operationalize privacy-respecting phone use as permissioned access, minimal disclosure, and user-controlled memory through a minimal privacy contract, iMy, and pair it with instrumented mock apps plus rule-based auditing that make unnecessary permission requests, deceptive re-disclosure, and unnecessary form filling observable and reproducible. Across five frontier models on 10 mobile apps and 300 tasks, we find that task success, privacy-compliant task completion, and later-session use of saved preferences are distinct capabilities, and no single model dominates all three. Evaluating success and privacy jointly reshuffles the model ordering relative to either metric alone. The most persistent failure mode across models is simple data minimization: agents still fill optional personal entries that the task does not require. These results show that privacy failures arise from over-helpful execution of benign tasks, and that success-only evaluation overestimates the deployment readiness of current phone-use agents. All code, mock apps, and agent trajectories are publicly available at~ this https URL. 

---
# Dual Optimal: Make Your LLM Peer-like with Dignity 

**Authors**: Xiangqi Wang, Yue Huang, Haomin Zhuang, Kehan Guo, Xiangliang Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2604.00979)  

**Abstract**: Current aligned language models exhibit a dual failure mode we term the Evasive Servant: they sycophantically validate flawed user beliefs while deflecting responsibility with boilerplate disclaimers. We propose the Dignified Peer framework, which counters servility with anti-sycophancy and trustworthiness, and mitigates evasiveness through empathy and creativity. Realizing this agent requires overcoming significant challenges in data supervision, objective collapse, and evaluation bias. We address these issues by introducing the PersonaKnob dataset which features a compositional partial order structure of multiple persona preference. This data is utilized alongside a tolerant constrained Lagrangian DPO algorithm that dynamically balances all persona dimensions to prevent behavioral collapse. Additionally, we employ a psychometrically calibrated Item Response Theory evaluation protocol to disentangle latent model persona capability from confounders like judge biases. Extensive empirical studies demonstrate that our approach successfully build a LLM agent with both dignity and peer. 

---
# Flow-based Policy With Distributional Reinforcement Learning in Trajectory Optimization 

**Authors**: Ruijie Hao, Longfei Zhang, Yang Dai, Yang Ma, Xingxing Liang, Guangquan Cheng  

**Link**: [PDF](https://arxiv.org/pdf/2604.00977)  

**Abstract**: Reinforcement Learning (RL) has proven highly effective in addressing complex control and decision-making tasks. However, in most traditional RL algorithms, the policy is typically parameterized as a diagonal Gaussian distribution, which constrains the policy from capturing multimodal distributions, making it difficult to cover the full range of optimal solutions in multi-solution problems, and the return is reduced to a mean value, losing its multimodal nature and thus providing insufficient guidance for policy updates. In response to these problems, we propose a RL algorithm termed flow-based policy with distributional RL (FP-DRL). This algorithm models the policy using flow matching, which offers both computational efficiency and the capacity to fit complex distributions. Additionally, it employs a distributional RL approach to model and optimize the entire return distribution, thereby more effectively guiding multimodal policy updates and improving agent performance. Experimental trails on MuJoCo benchmarks demonstrate that the FP-DRL algorithm achieves state-of-the-art (SOTA) performance in most MuJoCo control tasks while exhibiting superior representation capability of the flow policy. 

---
# WARP: Guaranteed Inner-Layer Repair of NLP Transformers 

**Authors**: Hsin-Ling Hsu, Min-Yu Chen, Nai-Chia Chen, Yan-Ru Chen, Yi-Ling Chang, Fang Yu  

**Link**: [PDF](https://arxiv.org/pdf/2604.00938)  

**Abstract**: Transformer-based NLP models remain vulnerable to adversarial perturbations, yet existing repair methods face a fundamental trade-off: gradient-based approaches offer flexibility but lack verifiability and often overfit; methods that do provide repair guarantees are restricted to the final layer or small networks, significantly limiting the parameter search space available for repair. We present WARP (Weight-Adjusted Repair with Provability), a constraint-based repair framework that extends repair beyond the last layer of Transformer models. WARP formulates repair as a convex quadratic program derived from a first-order linearization of the logit gap, enabling tractable optimization over a high-dimensional parameter space. Under the condition that the first-order approximation holds, this formulation induces three per-sample guarantees: (i) a positive margin constraint ensuring correct classification on repaired inputs, (ii) preservation constraints over a designated remain set, and (iii) a certified robustness radius derived from Lipschitz continuity. To ensure feasibility across varying model architectures, we introduce a sensitivity-based preprocessing step that conditions the optimization landscape accordingly. We further show that the iterative optimization procedure converges to solutions satisfying all repair constraints under mild assumptions. Empirical evaluation on encoder-only Transformers with varying layer architectures validates that these guarantees hold in practice while improving robustness to adversarial inputs. Our results demonstrate that guaranteed, generalizable Transformer repair is achievable through principled constraint-based optimization. 

---
# Learning Quantised Structure-Preserving Motion Representations for Dance Fingerprinting 

**Authors**: Arina Kharlamova, Bowei He, Chen Ma, Xue Liu  

**Link**: [PDF](https://arxiv.org/pdf/2604.00927)  

**Abstract**: We present DANCEMATCH, an end-to-end framework for motion-based dance retrieval, the task of identifying semantically similar choreographies directly from raw video, defined as DANCE FINGERPRINTING. While existing motion analysis and retrieval methods can compare pose sequences, they rely on continuous embeddings that are difficult to index, interpret, or scale. In contrast, DANCEMATCH constructs compact, discrete motion signatures that capture the spatio-temporal structure of dance while enabling efficient large-scale retrieval. Our system integrates Skeleton Motion Quantisation (SMQ) with Spatio-Temporal Transformers (STT) to encode human poses, extracted via Apple CoMotion, into a structured motion vocabulary. We further design DANCE RETRIEVAL ENGINE (DRE), which performs sub-linear retrieval using a histogram-based index followed by re-ranking for refined matching. To facilitate reproducible research, we release DANCETYPESBENCHMARK, a pose-aligned dataset annotated with quantised motion tokens. Experiments demonstrate robust retrieval across diverse dance styles and strong generalisation to unseen choreographies, establishing a foundation for scalable motion fingerprinting and quantitative choreographic analysis. 

---
# Representation Selection via Cross-Model Agreement using Canonical Correlation Analysis 

**Authors**: Dylan B. Lewis, Jens Gregor, Hector Santos-Villalobos  

**Link**: [PDF](https://arxiv.org/pdf/2604.00921)  

**Abstract**: Modern vision pipelines increasingly rely on pretrained image encoders whose representations are reused across tasks and models, yet these representations are often overcomplete and model-specific. We propose a simple, training-free method to improve the efficiency of image representations via a post-hoc canonical correlation analysis (CCA) operator. By leveraging the shared structure between representations produced by two pre-trained image encoders, our method finds linear projections that serve as a principled form of representation selection and dimensionality reduction, retaining shared semantic content while discarding redundant dimensions. Unlike standard dimensionality reduction techniques such as PCA, which operate on a single embedding space, our approach leverages cross-model agreement to guide representation distillation and refinement. The technique allows representations to be reduced by more than 75% in dimensionality with improved downstream performance, or enhanced at fixed dimensionality via post-hoc representation transfer from larger or fine-tuned models. Empirical results on ImageNet-1k, CIFAR-100, MNIST, and additional benchmarks show consistent improvements over both baseline and PCA-projected representations, with accuracy gains of up to 12.6%. 

---
# Investigating Autonomous Agent Contributions in the Wild: Activity Patterns and Code Change over Time 

**Authors**: Razvan Mihai Popescu, David Gros, Andrei Botocan, Rahul Pandita, Prem Devanbu, Maliheh Izadi  

**Link**: [PDF](https://arxiv.org/pdf/2604.00917)  

**Abstract**: The rise of large language models for code has reshaped software development. Autonomous coding agents, able to create branches, open pull requests, and perform code reviews, now actively contribute to real-world projects. Their growing role offers a unique and timely opportunity to investigate AI-driven contributions and their effects on code quality, team dynamics, and software maintainability. In this work, we construct a novel dataset of approximately $110,000$ open-source pull requests, including associated commits, comments, reviews, issues, and file changes, collectively representing millions of lines of source code. We compare five popular coding agents, including OpenAI Codex, Claude Code, GitHub Copilot, Google Jules, and Devin, examining how their usage differs in various development aspects such as merge frequency, edited file types, and developer interaction signals, including comments and reviews. Furthermore, we emphasize that code authoring and review are only a small part of the larger software engineering process, as the resulting code must also be maintained and updated over time. Hence, we offer several longitudinal estimates of survival and churn rates for agent-generated versus human-authored code. Ultimately, our findings indicate an increasing agent activity in open-source projects, although their contributions are associated with more churn over time compared to human-authored code. 

---
# PixelPrune: Pixel-Level Adaptive Visual Token Reduction via Predictive Coding 

**Authors**: Nan Wang, Zhiwei Jin, Chen Chen, Haonan Lu  

**Link**: [PDF](https://arxiv.org/pdf/2604.00886)  

**Abstract**: Document understanding and GUI interaction are among the highest-value applications of Vision-Language Models (VLMs), yet they impose exceptionally heavy computational burden: fine-grained text and small UI elements demand high-resolution inputs that produce tens of thousands of visual tokens. We observe that this cost is largely wasteful -- across document and GUI benchmarks, only 22--71\% of image patches are pixel-unique, the rest being exact duplicates of another patch in the same image. We propose \textbf{PixelPrune}, which exploits this pixel-level redundancy through predictive-coding-based compression, pruning redundant patches \emph{before} the Vision Transformer (ViT) encoder. Because it operates in pixel space prior to any neural computation, PixelPrune accelerates both the ViT encoder and the downstream LLM, covering the full inference pipeline. The method is training-free, requires no learnable parameters, and supports pixel-lossless compression ($\tau{=}0$) as well as controlled lossy compression ($\tau{>}0$). Experiments across three model scales and document and GUI benchmarks show that PixelPrune maintains competitive task accuracy while delivering up to 4.2$\times$ inference speedup and 1.9$\times$ training acceleration. Code is available at this https URL. 

---
# KUET at StanceNakba Shared Task: StanceMoE: Mixture-of-Experts Architecture for Stance Detection 

**Authors**: Abdullah Al Shafi, Md. Milon Islam, Sk. Imran Hossain, K. M. Azharul Hasan  

**Link**: [PDF](https://arxiv.org/pdf/2604.00878)  

**Abstract**: Actor-level stance detection aims to determine an author expressed position toward specific geopolitical actors mentioned or implicated in a text. Although transformer-based models have achieved relatively good performance in stance classification, they typically rely on unified representations that may not sufficiently capture heterogeneous linguistic signals, such as contrastive discourse structures, framing cues, and salient lexical indicators. This motivates the need for adaptive architectures that explicitly model diverse stance-expressive patterns. In this paper, we propose StanceMoE, a context-enhanced Mixture-of-Experts (MoE) architecture built upon a fine-tuned BERT encoder for actor-level stance detection. Our model integrates six expert modules designed to capture complementary linguistic signals, including global semantic orientation, salient lexical cues, clause-level focus, phrase-level patterns, framing indicators, and contrast-driven discourse shifts. A context-aware gating mechanism dynamically weights expert contributions, enabling adaptive routing based on input characteristics. Experiments are conducted on the StanceNakba 2026 Subtask A dataset, comprising 1,401 annotated English texts where the target actor is implicit in the text. StanceMoE achieves a macro-F1 score of 94.26%, outperforming traditional baselines, and alternative BERT-based variants. 

---
# Learning to Learn-at-Test-Time: Language Agents with Learnable Adaptation Policies 

**Authors**: Zhanzhi Lou, Hui Chen, Yibo Li, Qian Wang, Bryan Hooi  

**Link**: [PDF](https://arxiv.org/pdf/2604.00830)  

**Abstract**: Test-Time Learning (TTL) enables language agents to iteratively refine their performance through repeated interactions with the environment at inference time. At the core of TTL is an adaptation policy that updates the actor policy based on experience from previous episodes, thereby improving future behavior. Existing methods rely on fixed, hand-crafted adaptation policies rather than optimizing them for downstream improvement. We argue that optimal adaptation policies should be learned from task environments, not hand-engineered based on human intuition. To achieve this, we introduce Meta-TTL, a framework that formulates the discovery of effective adaptation policies as a bi-level optimization problem. Within this framework, the inner loop executes the standard TTL process, measuring how effectively a candidate adaptation policy helps an agent correct errors across sequential episodes. Guided by the agent's performance, the outer loop employs evolutionary search over a diverse distribution of training tasks to iteratively refine the adaptation policy. We evaluate Meta-TTL on Jericho and WebArena-Lite across both in-distribution (ID) and out-of-distribution (OOD) settings, using multiple meta-agent backbones. Results on both benchmarks show that Meta-TTL consistently outperforms hand-crafted baselines, suggesting that the optimized adaptation policy encodes transferable strategies that generalize beyond the training task distribution. 

---
# Emotion Entanglement and Bayesian Inference for Multi-Dimensional Emotion Understanding 

**Authors**: Hemanth Kotaprolu, Kishan Maharaj, Raey Zhao, Abhijit Mishra, Pushpak Bhattacharyya  

**Link**: [PDF](https://arxiv.org/pdf/2604.00819)  

**Abstract**: Understanding emotions in natural language is inherently a multi-dimensional reasoning problem, where multiple affective signals interact through context, interpersonal relations, and situational cues. However, most existing emotion understanding benchmarks rely on short texts and predefined emotion labels, reducing this process to independent label prediction and ignoring the structured dependencies among emotions. To address this limitation, we introduce Emotional Scenarios (EmoScene), a theory-grounded benchmark of 4,731 context-rich scenarios annotated with an 8-dimensional emotion vector derived from Plutchik's basic emotions. We evaluate six instruction-tuned large language models in a zero-shot setting and observe modest performance, with the best model achieving a Macro F1 of 0.501, highlighting the difficulty of context-aware multi-label emotion prediction. Motivated by the observation that emotions rarely occur independently, we further propose an entanglement-aware Bayesian inference framework that incorporates emotion co-occurrence statistics to perform joint posterior inference over the emotion vector. This lightweight post-processing improves structural consistency of predictions and yields notable gains for weaker models (e.g., +0.051 Macro F1 for Qwen2.5-7B). EmoScene therefore provides a challenging benchmark for studying multi-dimensional emotion understanding and the limitations of current language models. 

---
# DVGT-2: Vision-Geometry-Action Model for Autonomous Driving at Scale 

**Authors**: Sicheng Zuo, Zixun Xie, Wenzhao Zheng, Shaoqing Xu, Fang Li, Hanbing Li, Long Chen, Zhi-Xin Yang, Jiwen Lu  

**Link**: [PDF](https://arxiv.org/pdf/2604.00813)  

**Abstract**: End-to-end autonomous driving has evolved from the conventional paradigm based on sparse perception into vision-language-action (VLA) models, which focus on learning language descriptions as an auxiliary task to facilitate planning. In this paper, we propose an alternative Vision-Geometry-Action (VGA) paradigm that advocates dense 3D geometry as the critical cue for autonomous driving. As vehicles operate in a 3D world, we think dense 3D geometry provides the most comprehensive information for decision-making. However, most existing geometry reconstruction methods (e.g., DVGT) rely on computationally expensive batch processing of multi-frame inputs and cannot be applied to online planning. To address this, we introduce a streaming Driving Visual Geometry Transformer (DVGT-2), which processes inputs in an online manner and jointly outputs dense geometry and trajectory planning for the current frame. We employ temporal causal attention and cache historical features to support on-the-fly inference. To further enhance efficiency, we propose a sliding-window streaming strategy and use historical caches within a certain interval to avoid repetitive computations. Despite the faster speed, DVGT-2 achieves superior geometry reconstruction performance on various datasets. The same trained DVGT-2 can be directly applied to planning across diverse camera configurations without fine-tuning, including closed-loop NAVSIM and open-loop nuScenes benchmarks. 

---
# Routing-Free Mixture-of-Experts 

**Authors**: Yilun Liu, Jinru Han, Sikuan Yan, Volker Tresp, Yunpu Ma  

**Link**: [PDF](https://arxiv.org/pdf/2604.00801)  

**Abstract**: Standard Mixture-of-Experts (MoE) models rely on centralized routing mechanisms that introduce rigid inductive biases. We propose Routing-Free MoE which eliminates any hard-coded centralized designs including external routers, Softmax, Top-K and load balancing, instead encapsulating all activation functionalities within individual experts and directly optimized through continuous gradient flow, enabling each expert to determine its activation entirely on its own. We introduce a unified adaptive load-balancing framework to simultaneously optimize both expert-balancing and token-balancing objectives through a configurable interpolation, allowing flexible and customizable resource allocation. Extensive experiments show that Routing-Free MoE can consistently outperform baselines with better scalability and robustness. We analyze its behavior in detail and offer insights that may facilitate future MoE design ad optimization. 

---
# Scalable Pretraining of Large Mixture of Experts Language Models on Aurora Super Computer 

**Authors**: Dharma Teja Vooturi, Dhiraj Kalamkar, Dipankar Das, Bharat Kaul  

**Link**: [PDF](https://arxiv.org/pdf/2604.00785)  

**Abstract**: Pretraining Large Language Models (LLMs) from scratch requires massive amount of compute. Aurora super computer is an ExaScale machine with 127,488 Intel PVC (Ponte Vechio) GPU tiles. In this work, we showcase LLM pretraining on Aurora at the scale of 1000s of GPU tiles. Towards this effort, we developed Optimus, an inhouse training library with support for standard large model training techniques. Using Optimus, we first pretrained Mula-1B, a 1 Billion dense model and Mula-7B-A1B, a 7 Billion Mixture of Experts (MoE) model from scratch on 3072 GPU tiles for the full 4 trillion tokens of the OLMoE-mix-0924 dataset. We then demonstrated model scaling by pretraining three large MoE models Mula-20B-A2B, Mula-100B-A7B, and Mula-220B-A10B till 100 Billion tokens on the same dataset. On our largest model Mula-220B-A10B, we pushed the compute scaling from 384 to 12288 GPU tiles and observed scaling efficiency of around 90% at 12288 GPU tiles. We significantly improved the runtime performance of MoE models using custom GPU kernels for expert computation, and a novel EP-Aware sharded optimizer resulting in training speedups up to 1.71x. As part of the Optimus library, we also developed a robust set of reliability and fault tolerant features to improve training stability and continuity at scale. 

---
# Thinking Wrong in Silence: Backdoor Attacks on Continuous Latent Reasoning 

**Authors**: Swapnil Parekh  

**Link**: [PDF](https://arxiv.org/pdf/2604.00770)  

**Abstract**: A new generation of language models reasons entirely in continuous hidden states, producing no tokens and leaving
no audit trail. We show that this silence creates a fundamentally new attack surface. ThoughtSteer perturbs a
single embedding vector at the input layer; the model's own multi-pass reasoning amplifies this perturbation into a
hijacked latent trajectory that reliably produces the attacker's chosen answer, while remaining structurally
invisible to every token-level defense. Across two architectures (Coconut and SimCoT), three reasoning benchmarks,
and model scales from 124M to 3B parameters, ThoughtSteer achieves >=99% attack success rate with near-baseline
clean accuracy, transfers to held-out benchmarks without retraining (94-100%), evades all five evaluated active
defenses, and survives 25 epochs of clean fine-tuning. We trace these results to a unifying mechanism: Neural
Collapse in the latent space pulls triggered representations onto a tight geometric attractor, explaining both why
defenses fail and why any effective backdoor must leave a linearly separable signature (probe AUC>=0.999). Yet a
striking paradox emerges: individual latent vectors still encode the correct answer even as the model outputs the
wrong one. The adversarial information is not in any single vector but in the collective trajectory, establishing
backdoor perturbations as a new lens for mechanistic interpretability of continuous reasoning. Code and checkpoints
are available. 

---
# IWP: Token Pruning as Implicit Weight Pruning in Large Vision Language Models 

**Authors**: Dong-Jae Lee, Sunghyun Baek, Junmo Kim  

**Link**: [PDF](https://arxiv.org/pdf/2604.00757)  

**Abstract**: Large Vision Language Models show impressive performance across image and video understanding tasks, yet their computational cost grows rapidly with the number of visual tokens. Existing token pruning methods mitigate this issue through empirical approaches while overlooking the internal mechanism of attention. In this paper, we propose a novel training free token pruning framework grounded in the dual form perspective of attention. We reformulate attention as an implicit linear layer whose weight matrix is the sum of rank 1 outer products, each generated by a single token's key value pair. Token pruning thus reduces to selecting an optimal subset of these rank 1 updates that best approximates the original dual weight matrix. Extending this perspective to standard softmax attention in LVLMs, we derive a novel metric quantifying both a token's information magnitude and information duplication. To efficiently select the subset with the proposed metric, we introduce Progressive Chunked Maximal Marginal Relevance. Extensive experiments demonstrate that our method achieves a better trade off between performance and efficiency, while providing another perspective on existing pruning approaches. 

---
# BioCOMPASS: Integrating Biomarkers into Transformer-Based Immunotherapy Response Prediction 

**Authors**: Sayed Hashim, Frank Soboczenski, Paul Cairns  

**Link**: [PDF](https://arxiv.org/pdf/2604.00739)  

**Abstract**: Datasets used in immunotherapy response prediction are typically small in size, as well as diverse in cancer type, drug administered, and sequencer used. Models often drop in performance when tested on patient cohorts that are not included in the training process. Recent work has shown that transformer-based models along with self-supervised learning show better generalisation performance than threshold-based biomarkers, but is still suboptimal. We present BioCOMPASS, an extension of a transformer-based model called COMPASS, that integrates biomarkers and treatment information to further improve its generalisability. Instead of feeding biomarker data as input, we built loss components to align them with the model's intermediate representations. We found that components such as treatment gating and pathway consistency loss improved generalisability when evaluated with Leave-one-cohort-out, Leave-one-cancer-type-out and Leave-one-treatment-out strategies. Results show that building components that exploit biomarker and treatment information can help in generalisability of immunotherapy response prediction. Careful curation of additional components that leverage complementary clinical information and domain knowledge represents a promising direction for future research. 

---
# Spectral Compact Training: Pre-Training Large Language Models via Permanent Truncated SVD and Stiefel QR Retraction 

**Authors**: Björn Roman Kohlberger  

**Link**: [PDF](https://arxiv.org/pdf/2604.00733)  

**Abstract**: The memory wall remains the primary bottleneck for training large language models on consumer hardware. We introduce Spectral Compact Training (SCT), a method that replaces dense weight matrices with permanent truncated SVD factors W = U diag(s) V^T, where the full dense matrix is never materialized during training or inference. Gradients flow through the compact spectral factors via standard backpropagation, and U, V are retracted to the Stiefel manifold via QR decomposition after each optimizer step. SCT achieves up to 199x memory reduction per MLP layer at rank 32, enabling full training steps of 70B-parameter architectures on a Steam Deck handheld (7.2 GB peak memory vs. 1,245 GB for dense FP32 training with Adam). Rank-sweep experiments on SmolLM2-1.7B (ranks 32-256, 2000 steps, NVIDIA A100) show that all tested ranks converge to the same loss floor (~4.2-4.5), identifying the learning rate schedule -- not MLP rank -- as the primary bottleneck. Rank 128 emerges as the efficiency sweet spot at 11.7x MLP compression with the lowest perplexity. GPU memory drops 46% at rank 32 while training throughput doubles. 

---
# A CEFR-Inspired Classification Framework with Fuzzy C-Means To Automate Assessment of Programming Skills in Scratch 

**Authors**: Ricardo Hidalgo-Aragón, Jesús M. González-Barahona, Gregorio Robles  

**Link**: [PDF](https://arxiv.org/pdf/2604.00730)  

**Abstract**: Context: Schools, training platforms, and technology firms increasingly need to assess programming proficiency at scale with transparent, reproducible methods that support personalized learning pathways. Objective: This study introduces a pedagogical framework for Scratch project assessment, aligned with the Common European Framework of Reference (CEFR), providing universal competency levels for students and teachers alongside actionable insights for curriculum design. Method: We apply Fuzzy C-Means clustering to 2008246 Scratch projects evaluated via this http URL, implementing an ordinal criterion to map clusters to CEFR levels (A1-C2), and introducing enhanced classification metrics that identify transitional learners, enable continuous progress tracking, and quantify classification certainty to balance automated feedback with instructor review. Impact: The framework enables diagnosis of systemic curriculum gaps-notably a "B2 bottleneck" where only 13.3% of learners reside due to the cognitive load of integrating Logic Synchronization, and Data Representation--while providing certainty--based triggers for human intervention. 

---
# GRASP: Gradient Realignment via Active Shared Perception for Multi-Agent Collaborative Optimization 

**Authors**: Sihan Zhou, Tiantian He, Yifan Lu, Yaqing Hou, Yew-Soon Ong  

**Link**: [PDF](https://arxiv.org/pdf/2604.00717)  

**Abstract**: Non-stationarity arises from concurrent policy updates and leads to persistent environmental fluctuations. Existing approaches like Centralized Training with Decentralized Execution (CTDE) and sequential update schemes mitigate this issue. However, since the perception of the policies of other agents remains dependent on sampling environmental interaction data, the agent essentially operates in a passive perception state. This inevitably triggers equilibrium oscillations and significantly slows the convergence speed of the system. To address this issue, we propose Gradient Realignment via Active Shared Perception (GRASP), a novel framework that defines generalized Bellman equilibrium as a stable objective for policy evolution. The core mechanism of GRASP involves utilizing the independent gradients of agents to derive a defined consensus gradient, enabling agents to actively perceive policy updates and optimize team collaboration. Theoretically, we leverage the Kakutani Fixed-Point Theorem to prove that the consensus direction $u^*$ guarantees the existence and attainability of this equilibrium. Extensive experiments on StarCraft II Multi-Agent Challenge (SMAC) and Google Research Football (GRF) demonstrate the scalability and promising performance of the framework. 

---
# To Memorize or to Retrieve: Scaling Laws for RAG-Considerate Pretraining 

**Authors**: Karan Singh, Michael Yu, Varun Gangal, Zhuofu Tao, Sachin Kumar, Emmy Liu, Steven Y. Feng  

**Link**: [PDF](https://arxiv.org/pdf/2604.00715)  

**Abstract**: Retrieval-augmented generation (RAG) improves language model (LM) performance by providing relevant context at test time for knowledge-intensive situations. However, the relationship between parametric knowledge acquired during pretraining and non-parametric knowledge accessed via retrieval remains poorly understood, especially under fixed data budgets. In this work, we systematically study the trade-off between pretraining corpus size and retrieval store size across a wide range of model and data scales. We train OLMo-2-based LMs ranging from 30M to 3B parameters on up to 100B tokens of DCLM data, while varying both pretraining data scale (1-150x the number of parameters) and retrieval store size (1-20x), and evaluate performance across a diverse suite of benchmarks spanning reasoning, scientific QA, and open-domain QA. We find that retrieval consistently improves performance over parametric-only baselines across model scales and introduce a three-dimensional scaling framework that models performance as a function of model size, pretraining tokens, and retrieval corpus size. This scaling manifold enables us to estimate optimal allocations of a fixed data budget between pretraining and retrieval, revealing that the marginal utility of retrieval depends strongly on model scale, task type, and the degree of pretraining saturation. Our results provide a quantitative foundation for understanding when and how retrieval should complement pretraining, offering practical guidance for allocating data resources in the design of scalable language modeling systems. 

---
# AutoEG: Exploiting Known Third-Party Vulnerabilities in Black-Box Web Applications 

**Authors**: Ruozhao Yang, Mingfei Cheng, Gelei Deng, Junjie Wang, Tianwei Zhang, Xiaofei Xie  

**Link**: [PDF](https://arxiv.org/pdf/2604.00704)  

**Abstract**: Large-scale web applications are widely deployed with complex third-party components, inheriting security risks arising from component vulnerabilities. Security assessment is therefore required to determine whether such known vulnerabilities remain practically exploitable in real applications. Penetration testing is a widely adopted approach that validates exploitability by launching concrete attacks against known vulnerabilities in real-world black-box systems. However, existing approaches often fail to automatically generate reliable exploits, limiting their effectiveness in practical security assessment. This limitation mainly stems from two issues: (1) precisely triggering vulnerabilities with correct technical details, and (2) adapting exploits to diverse real-world deployment settings.
In this paper, we propose AutoEG, a fully automated multi-agent framework for exploit generation targeting black-box web applications. AutoEG has two phases: First, AutoEG extracts precise vulnerability trigger logic from unstructured vulnerability information and encapsulates it into reusable trigger functions. Second, AutoEG uses trigger functions for concrete attack objectives and iteratively refines exploits through feedback-driven interaction with the target application. We evaluate AutoEG on 104 real-world vulnerabilities with 29 attack objectives, resulting in 660 exploitation tasks and 55,440 exploit attempts. AutoEG achieves an average success rate of 82.41%, substantially outperforming state-of-the-art baselines, whose best performance reaches only 32.88%. 

---
# Learning to Hint for Reinforcement Learning 

**Authors**: Yu Xia, Canwen Xu, Zhewei Yao, Julian McAuley, Yuxiong He  

**Link**: [PDF](https://arxiv.org/pdf/2604.00698)  

**Abstract**: Group Relative Policy Optimization (GRPO) is widely used for reinforcement learning with verifiable rewards, but it often suffers from advantage collapse: when all rollouts in a group receive the same reward, the group yields zero relative advantage and thus no learning signal. For example, if a question is too hard for the reasoner, all sampled rollouts can be incorrect and receive zero reward. Recent work addresses this issue by adding hints or auxiliary scaffolds to such hard questions so that the reasoner produces mixed outcomes and recovers a non-zero update. However, existing hints are usually fixed rather than adapted to the current reasoner, and a hint that creates learning signal under the hinted input does not necessarily improve the no-hint policy used at test time. To this end, we propose Hint Learning for Reinforcement Learning (HiLL), a framework that jointly trains a hinter policy and a reasoner policy during RL. For each hard question, the hinter generates hints online conditioned on the current reasoner's incorrect rollout, allowing hint generation to adapt to the reasoner's evolving errors. We further introduce hint reliance, which measures how strongly correct hinted trajectories depend on the hint. We derive a transferability result showing that lower hint reliance implies stronger transfer from hinted success to no-hint success, and we use this result to define a transfer-weighted reward for training the hinter. Therefore, HiLL favors hints that not only recover informative GRPO groups, but also produce signals that are more likely to improve the original no-hint policy. Experiments across multiple benchmarks show that HiLL consistently outperforms GRPO and prior hint-based baselines, demonstrating the value of adaptive and transfer-aware hint learning for RL. The code is available at this https URL. 

---
# Internal APIs Are All You Need: Shadow APIs, Shared Discovery, and the Case Against Browser-First Agent Architectures 

**Authors**: Lewis Tham, Nicholas Mac Gregor Garcia, Jungpil Hahn  

**Link**: [PDF](https://arxiv.org/pdf/2604.00694)  

**Abstract**: Autonomous agents increasingly interact with the web, yet most websites remain designed for human browsers -- a fundamental mismatch that the emerging ``Agentic Web'' must resolve. Agents must repeatedly browse pages, inspect DOMs, and reverse-engineer callable routes -- a process that is slow, brittle, and redundantly repeated across agents. We observe that every modern website already exposes internal APIs (sometimes called \emph{shadow APIs}) behind its user interface -- first-party endpoints that power the site's own functionality. We present Unbrowse, a shared route graph that transforms browser-based route discovery into a collectively maintained index of these callable first-party interfaces. The system passively learns routes from real browsing traffic and serves cached routes via direct API calls. In a single-host live-web benchmark of equivalent information-retrieval tasks across 94 domains, fully warmed cached execution averaged 950\,ms versus 3{,}404\,ms for Playwright browser automation (3.6$\times$ mean speedup, 5.4$\times$ median), with well-cached routes completing in under 100\,ms. A three-path execution model -- local cache, shared graph, or browser fallback -- ensures the system is voluntary and self-correcting. A three-tier micropayment model via the x402 protocol charges per-query search fees for graph lookups (Tier~3), a one-time install fee for discovery documentation (Tier~1), and optional per-execution fees for site owners who opt in (Tier~2). All tiers are grounded in a necessary condition for rational adoption: an agent uses the shared graph only when the total fee is lower than the expected cost of browser rediscovery. 

---
# Procela: Epistemic Governance in Mechanistic Simulations Under Structural Uncertainty 

**Authors**: Kinson Vernet  

**Link**: [PDF](https://arxiv.org/pdf/2604.00675)  

**Abstract**: Mechanistic simulations typically assume fixed ontologies: variables, causal relationships, and resolution policies are static. This assumption fails when the true causal structure is contested or unidentifiable-as in antimicrobial resistance (AMR) spread, where contact, environmental, and selection ontologies compete. We introduce Procela, a Python framework where variables act as epistemic authorities that maintain complete hypothesis memory, mechanisms encode competing ontologies as causal units, and governance observes epistemic signals and mutates system topology at runtime. This is the first framework where simulations test their own assumptions. We instantiate Procela for AMR in a hospital network with three competing families. Governance detects coverage decay, policy fragility, and runs structural probes. Results show 20.4% error reduction and 69% cumulative regret improvement over baseline. All experiments are reproducible with full auditability. Procela establishes a new paradigm: simulations that model not only the world but their own modeling process, enabling adaptation under structural uncertainty. 

---
# Streaming Model Cascades for Semantic SQL 

**Authors**: Paweł Liskowski, Kyle Schmaus  

**Link**: [PDF](https://arxiv.org/pdf/2604.00660)  

**Abstract**: Modern data warehouses extend SQL with semantic operators that invoke large language models on each qualifying row, but the per-row inference cost is prohibitive at scale. Model cascades reduce this cost by routing most rows through a fast proxy model and delegating uncertain cases to an expensive oracle. Existing frameworks, however, require global dataset access and optimize a single quality metric, limiting their applicability in distributed systems where data is partitioned across independent workers. We present two adaptive cascade algorithms designed for streaming, per-partition execution in which each worker processes its partition independently without inter-worker communication. SUPG-IT extends the SUPG statistical framework to streaming execution with iterative threshold refinement and joint precision-recall guarantees. GAMCAL replaces user-specified quality targets with a learned calibration model: a Generalized Additive Model maps proxy scores to calibrated probabilities with uncertainty quantification, enabling direct optimization of a cost-quality tradeoff through a single parameter. Experiments on six datasets in a production semantic SQL engine show that both algorithms achieve F1 > 0.95 on every dataset. GAMCAL achieves higher F1 per oracle call at cost-sensitive operating points, while SUPG-IT reaches a higher quality ceiling with formal guarantees on precision and recall. 

---
# UniMixer: A Unified Architecture for Scaling Laws in Recommendation Systems 

**Authors**: Mingming Ha, Guanchen Wang, Linxun Chen, Xuan Rao, Yuexin Shi, Tianbao Ma, Zhaojie Liu, Yunqian Fan, Zilong Lu, Yanan Niu, Han Li, Kun Gai  

**Link**: [PDF](https://arxiv.org/pdf/2604.00590)  

**Abstract**: In recent years, the scaling laws of recommendation models have attracted increasing attention, which govern the relationship between performance and parameters/FLOPs of recommenders. Currently, there are three mainstream architectures for achieving scaling in recommendation models, namely attention-based, TokenMixer-based, and factorization-machine-based methods, which exhibit fundamental differences in both design philosophy and architectural structure. In this paper, we propose a unified scaling architecture for recommendation systems, namely \textbf{UniMixer}, to improve scaling efficiency and establish a unified theoretical framework that unifies the mainstream scaling blocks. By transforming the rule-based TokenMixer to an equivalent parameterized structure, we construct a generalized parameterized feature mixing module that allows the token mixing patterns to be optimized and learned during model training. Meanwhile, the generalized parameterized token mixing removes the constraint in TokenMixer that requires the number of heads to be equal to the number of tokens. Furthermore, we establish a unified scaling module design framework for recommender systems, which bridges the connections among attention-based, TokenMixer-based, and factorization-machine-based methods. To further boost scaling ROI, a lightweight UniMixing module is designed, \textbf{UniMixing-Lite}, which further compresses the model parameters and computational cost while significantly improve the model performance. The scaling curves are shown in the following figure. Extensive offline and online experiments are conducted to verify the superior scaling abilities of \textbf{UniMixer}. 

---
# HabitatAgent: An End-to-End Multi-Agent System for Housing Consultation 

**Authors**: Hongyang Yang, Yanxin Zhang, Yang She, Yue Xiao, Hao Wu, Yiyang Zhang, Jiapeng Hou, Rongshan Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2604.00556)  

**Abstract**: Housing selection is a high-stakes and largely irreversible decision problem. We study housing consultation as a decision-support interface for housing selection. Existing housing platforms and many LLM-based assistants often reduce this process to ranking or recommendation, resulting in opaque reasoning, brittle multi-constraint handling, and limited guarantees on factuality.
We present HabitatAgent, the first LLM-powered multi-agent architecture for end-to-end housing consultation. HabitatAgent comprises four specialized agent roles: Memory, Retrieval, Generation, and Validation. The Memory Agent maintains multi-layer user memory through internal stages for constraint extraction, memory fusion, and verification-gated updates; the Retrieval Agent performs hybrid vector--graph retrieval (GraphRAG); the Generation Agent produces evidence-referenced recommendations and explanations; and the Validation Agent applies multi-tier verification and targeted remediation. Together, these agents provide an auditable and reliable workflow for end-to-end housing consultation.
We evaluate HabitatAgent on 100 real user consultation scenarios (300 multi-turn question--answer pairs) under an end-to-end correctness protocol. A strong single-stage baseline (Dense+Rerank) achieves 75% accuracy, while HabitatAgent reaches 95%. 

---
# MATHENA: Mamba-based Architectural Tooth Hierarchical Estimator and Holistic Evaluation Network for Anatomy 

**Authors**: Kyeonghun Kim, Jaehyung Park, Youngung Han, Anna Jung, Seongbin Park, Sumin Lee, Jiwon Yang, Jiyoon Han, Subeen Lee, Junsu Lim, Hyunsu Go, Eunseob Choi, Hyeonseok Jung, Soo Yong Kim, Woo Kyoung Jeong, Won Jae Lee, Pa Hong, Hyuk-Jae Lee, Ken Ying-Kai Liao, Nam-Joon Kim  

**Link**: [PDF](https://arxiv.org/pdf/2604.00537)  

**Abstract**: Dental diagnosis from Orthopantomograms (OPGs) requires coordination of tooth detection, caries segmentation (CarSeg), anomaly detection (AD), and dental developmental staging (DDS). We propose Mamba-based Architectural Tooth Hierarchical Estimator and Holistic Evaluation Network for Anatomy (MATHENA), a unified framework leveraging Mamba's linear-complexity State Space Models (SSM) to address all four tasks. MATHENA integrates MATHE, a multi-resolution SSM-driven detector with four-directional Vision State Space (VSS) blocks for O(N) global context modeling, generating per-tooth crops. These crops are processed by HENA, a lightweight Mamba-UNet with a triple-head architecture and Global Context State Token (GCST). In the triple-head architecture, CarSeg is first trained as an upstream task to establish shared representations, which are then frozen and reused for downstream AD fine-tuning and DDS classification via linear probing, enabling stable, efficient learning. We also curate PARTHENON, a benchmark comprising 15,062 annotated instances from ten datasets. MATHENA achieves 93.78% mAP@50 in tooth detection, 90.11% Dice for CarSeg, 88.35% for AD, and 72.40% ACC for DDS. 

---
# Optimsyn: Influence-Guided Rubrics Optimization for Synthetic Data Generation 

**Authors**: Zhiting Fan, Ruizhe Chen, Tianxiang Hu, Ru Peng, Zenan Huang, Haokai Xu, Yixin Chen, Jian Wu, Junbo Zhao, Zuozhu Liu  

**Link**: [PDF](https://arxiv.org/pdf/2604.00536)  

**Abstract**: Large language models (LLMs) achieve strong downstream performance largely due to abundant supervised fine-tuning (SFT) data. However, high-quality SFT data in knowledge-intensive domains such as humanities, social sciences, medicine, law, and finance is scarce because expert curation is expensive, privacy constraints are strict, and label consistency is hard to ensure. Recent work uses synthetic data, typically by prompting a generator over domain documents and filtering outputs with handcrafted rubrics. Yet rubric design is expert-dependent, transfers poorly across domains, and is often optimized through a brittle heuristic loop of writing rubrics, synthesizing data, training, inspecting results, and manually guessing revisions. This process lacks reliable quantitative feedback about how a rubric affects downstream performance. We propose evaluating synthetic data by its training utility on the target model and using this signal to guide data generation. Inspired by influence estimation, we adopt an optimizer-aware estimator that uses gradient information to quantify each synthetic sample's contribution to a target model's objective on specific tasks. Our analysis shows that even when synthetic and real samples are close in embedding space, their influence on learning can differ substantially. Based on this insight, we propose an optimization-based framework that adapts rubrics using target-model feedback. We provide lightweight guiding text and use a rubric-specialized model to generate task-conditioned rubrics. Influence score is used as the reward to optimize the rubric generator with reinforcement learning. Experiments across domains, target models, and data generators show consistent improvements and strong generalization without task-specific tuning. 

---
# Think, Act, Build: An Agentic Framework with Vision Language Models for Zero-Shot 3D Visual Grounding 

**Authors**: Haibo Wang, Zihao Lin, Zhiyang Xu, Lifu Huang  

**Link**: [PDF](https://arxiv.org/pdf/2604.00528)  

**Abstract**: 3D Visual Grounding (3D-VG) aims to localize objects in 3D scenes via natural language descriptions. While recent advancements leveraging Vision-Language Models (VLMs) have explored zero-shot possibilities, they typically suffer from a static workflow relying on preprocessed 3D point clouds, essentially degrading grounding into proposal matching. To bypass this reliance, our core motivation is to decouple the task: leveraging 2D VLMs to resolve complex spatial semantics, while relying on deterministic multi-view geometry to instantiate the 3D structure. Driven by this insight, we propose "Think, Act, Build (TAB)", a dynamic agentic framework that reformulates 3D-VG tasks as a generative 2D-to-3D reconstruction paradigm operating directly on raw RGB-D streams. Specifically, guided by a specialized 3D-VG skill, our VLM agent dynamically invokes visual tools to track and reconstruct the target across 2D frames. Crucially, to overcome the multi-view coverage deficit caused by strict VLM semantic tracking, we introduce the Semantic-Anchored Geometric Expansion, a mechanism that first anchors the target in a reference video clip and then leverages multi-view geometry to propagate its spatial location across unobserved frames. This enables the agent to "Build" the target's 3D representation by aggregating these multi-view features via camera parameters, directly mapping 2D visual cues to 3D coordinates. Furthermore, to ensure rigorous assessment, we identify flaws such as reference ambiguity and category errors in existing benchmarks and manually refine the incorrect queries. Extensive experiments on ScanRefer and Nr3D demonstrate that our framework, relying entirely on open-source models, significantly outperforms previous zero-shot methods and even surpasses fully supervised baselines. 

---
# Toward Optimal Sampling Rate Selection and Unbiased Classification for Precise Animal Activity Recognition 

**Authors**: Axiu Mao, Meilu Zhu, Lei Shen, Xiaoshuai Wang, Tomas Norton, Kai Liu  

**Link**: [PDF](https://arxiv.org/pdf/2604.00517)  

**Abstract**: With the rapid advancements in deep learning techniques, wearable sensor-aided animal activity recognition (AAR) has demonstrated promising performance, thereby improving livestock management efficiency as well as animal health and welfare monitoring. However, existing research often prioritizes overall performance, overlooking the fact that classification accuracies for specific animal behavioral categories may remain unsatisfactory. This issue typically stems from suboptimal sampling rates or class imbalance problems. To address these challenges and achieve high classification accuracy across all individual behaviors in farm animals, we propose a novel Individual-Behavior-Aware Network (IBA-Net). This network enhances the recognition of each specific behavior by simultaneously customizing features and calibrating the classifier. Specifically, considering that different behaviors require varying sampling rates to achieve optimal performance, we design a Mixture-of-Experts (MoE)-based Feature Customization (MFC) module. This module adaptively fuses data from multiple sampling rates, capturing customized features tailored to various animal behaviors. Additionally, to mitigate classifier bias toward majority classes caused by class imbalance, we develop a Neural Collapse-driven Classifier Calibration (NC3) module. This module introduces a fixed equiangular tight frame (ETF) classifier during the classification stage, maximizing the angles between pair-wise classifier vectors and thereby improving the classification performance for minority classes. To validate the effectiveness of IBA-Net, we conducted experiments on three public datasets covering goat, cattle, and horse activity recognition. The results demonstrate that our method consistently outperforms existing approaches across all datasets. 

---
# MAESIL: Masked Autoencoder for Enhanced Self-supervised Medical Image Learning 

**Authors**: Kyeonghun Kim, Hyeonseok Jung, Youngung Han, Junsu Lim, YeonJu Jean, Seongbin Park, Eunseob Choi, Hyunsu Go, SeoYoung Ju, Seohyoung Park, Gyeongmin Kim, MinJu Kwon, KyungSeok Yuh, Soo Yong Kim, Ken Ying-Kai Liao, Nam-Joon Kim, Hyuk-Jae Lee  

**Link**: [PDF](https://arxiv.org/pdf/2604.00514)  

**Abstract**: Training deep learning models for three-dimensional (3D) medical imaging, such as Computed Tomography (CT), is fundamentally challenged by the scarcity of labeled data. While pre-training on natural images is common, it results in a significant domain shift, limiting performance. Self-Supervised Learning (SSL) on unlabeled medical data has emerged as a powerful solution, but prominent frameworks often fail to exploit the inherent 3D nature of CT scans. These methods typically process 3D scans as a collection of independent 2D slices, an approach that fundamentally discards critical axial coherence and the 3D structural context. To address this limitation, we propose the autoencoder for enhanced self-supervised medical image learning(MAESIL), a novel self-supervised learning framework designed to capture 3D structural information efficiently. The core innovation is the 'superpatch', a 3D chunk-based input unit that balances 3D context preservation with computational efficiency. Our framework partitions the volume into superpatches and employs a 3D masked autoencoder strategy with a dual-masking strategy to learn comprehensive spatial representations. We validated our approach on three diverse large-scale public CT datasets. Our experimental results show that MAESIL demonstrates significant improvements over existing methods such as AE, VAE and VQ-VAE in key reconstruction metrics such as PSNR and SSIM. This establishes MAESIL as a robust and practical pre-training solution for 3D medical imaging tasks. 

---
# MOON3.0: Reasoning-aware Multimodal Representation Learning for E-commerce Product Understanding 

**Authors**: Junxian Wu, Chenghan Fu, Zhanheng Nie, Daoze Zhang, Bowen Wan, Wanxian Guan, Chuan Yu, Jian Xu, Bo Zheng  

**Link**: [PDF](https://arxiv.org/pdf/2604.00513)  

**Abstract**: With the rapid growth of e-commerce, exploring general representations rather than task-specific ones has attracted increasing attention. Although recent multimodal large language models (MLLMs) have driven significant progress in product understanding, they are typically employed as feature extractors that implicitly encode product information into global embeddings, thereby limiting their ability to capture fine-grained attributes. Therefore, we argue that leveraging the reasoning capabilities of MLLMs to explicitly model fine-grained product attributes holds significant potential. Nevertheless, achieving this goal remains non-trivial due to several key challenges: (i) long-context reasoning tends to dilute the model's attention to salient information in the raw input; (ii) supervised fine-tuning (SFT) primarily encourages rigid imitation, limiting the exploration of effective reasoning strategies; and (iii) fine-grained details are progressively attenuated during forward propagation. To address these issues, we propose MOON3.0, the first reasoning-aware MLLM-based model for product representation learning. Our method (1) employs a multi-head modality fusion module to adaptively integrate raw signals; (2) incorporates a joint contrastive and reinforcement learning framework to autonomously explore more effective reasoning strategies; and (3) introduces a fine-grained residual enhancement module to progressively preserve local details throughout the network. Additionally, we release a large-scale multimodal e-commerce benchmark MBE3.0. Experimentally, our model demonstrates state-of-the-art zero-shot performance across various downstream tasks on both our benchmark and public datasets. 

---
# Towards Initialization-dependent and Non-vacuous Generalization Bounds for Overparameterized Shallow Neural Networks 

**Authors**: Yunwen Lei, Yufeng Xie  

**Link**: [PDF](https://arxiv.org/pdf/2604.00505)  

**Abstract**: Overparameterized neural networks often show a benign overfitting property in the sense of achieving excellent generalization behavior despite the number of parameters exceeding the number of training examples. A promising direction to explain benign overfitting is to relate generalization to the norm of distance from initialization, motivated by the empirical observations that this distance is often significantly smaller than the norm itself. However, the existing initialization-dependent complexity analyses cannot fully exploit the power of initialization since the associated bounds depend on the spectral norm of the initialization matrix, which can scale as a square-root function of the width and are therefore not effective for overparameterized models. In this paper, we develop the first \emph{fully} initialization-dependent complexity bounds for shallow neural networks with general Lipschitz activation functions, which enjoys a logarithmic dependency on the width. Our bounds depend on the path-norm of the distance from initialization, which are derived by introducing a new peeling technique to handle the challenge along with the initialization-dependent constraint. We also develop a lower bound tight up to a constant factor. Finally, we conduct empirical comparisons and show that our generalization analysis implies non-vacuous bounds for overparameterized networks. 

---
# A Reasoning-Enabled Vision-Language Foundation Model for Chest X-ray Interpretation 

**Authors**: Yabin Zhang, Chong Wang, Yunhe Gao, Jiaming Liu, Maya Varma, Justin Xu, Sophie Ostmeier, Jin Long, Sergios Gatidis, Seena Dehkharghani, Arne Michalson, Eun Kyoung Hong, Christian Bluethgen, Haiwei Henry Guo, Alexander Victor Ortiz, Stephan Altmayer, Sandhya Bodapati, Joseph David Janizek, Ken Chang, Jean-Benoit Delbrouck, Akshay S. Chaudhari, Curtis P. Langlotz  

**Link**: [PDF](https://arxiv.org/pdf/2604.00493)  

**Abstract**: Chest X-rays (CXRs) are among the most frequently performed imaging examinations worldwide, yet rising imaging volumes increase radiologist workload and the risk of diagnostic errors. Although artificial intelligence (AI) systems have shown promise for CXR interpretation, most generate only final predictions, without making explicit how visual evidence is translated into radiographic findings and diagnostic predictions. We present CheXOne, a reasoning-enabled vision-language model for CXR interpretation. CheXOne jointly generates diagnostic predictions and explicit, clinically grounded reasoning traces that connect visual evidence, radiographic findings, and these predictions. The model is trained on 14.7 million instruction and reasoning samples curated from 30 public datasets spanning 36 CXR interpretation tasks, using a two-stage framework that combines instruction tuning with reinforcement learning to improve reasoning quality. We evaluate CheXOne in zero-shot settings across visual question answering, report generation, visual grounding and reasoning assessment, covering 17 evaluation settings. CheXOne outperforms existing medical and general-domain foundation models and achieves strong performance on independent public benchmarks. A clinical reader study demonstrates that CheXOne-drafted reports are comparable to or better than resident-written reports in 55% of cases, while effectively addressing clinical indications and enhancing both report writing and CXR interpretation efficiency. Further analyses involving radiologists reveal that the generated reasoning traces show high clinical factuality and provide causal support for the final predictions, offering a plausible explanation for the performance gains. These results suggest that explicit reasoning can improve model performance, interpretability and clinical utility in AI-assisted CXR interpretation. 

---
# Executing as You Generate: Hiding Execution Latency in LLM Code Generation 

**Authors**: Zhensu Sun, Zhihao Lin, Zhi Chen, Chengran Yang, Mingyi Zhou, Li Li, David Lo  

**Link**: [PDF](https://arxiv.org/pdf/2604.00491)  

**Abstract**: Current LLM-based coding agents follow a serial execution paradigm: the model first generates the complete code, then invokes an interpreter to execute it. This sequential workflow leaves the executor idle during generation and the generator idle during execution, resulting in unnecessary end-to-end latency. We observe that, unlike human developers, LLMs produce code tokens sequentially without revision, making it possible to execute code as it is being generated. We formalize this parallel execution paradigm, modeling it as a three-stage pipeline of generation, detection, and execution, and derive closed-form latency bounds that characterize its speedup potential and operating regimes. We then present Eager, a concrete implementation featuring AST-based chunking, dynamic batching with gated execution, and early error interruption. We evaluate Eager across four benchmarks, seven LLMs, and three execution environments. Results show that Eager reduces the non-overlapped execution latency by up to 99.9% and the end-to-end latency by up to 55% across seven LLMs and four benchmarks. 

---
# Not My Truce: Personality Differences in AI-Mediated Workplace Negotiation 

**Authors**: Veda Duddu, Jash Rajesh Parekh, Andy Mao, Hanyi Min, Ziang Xiao, Vedant Das Swain, Koustuv Saha  

**Link**: [PDF](https://arxiv.org/pdf/2604.00464)  

**Abstract**: AI-driven conversational coaching is increasingly used to support workplace negotiation, yet prior work assumes uniform effectiveness across users. We challenge this assumption by examining how individual differences, particularly personality traits, moderate coaching outcomes. We conducted a between-subjects experiment (N=267) comparing theory-driven AI (Trucey), general-purpose AI (Control-AI), and a traditional negotiation handbook (Control-NoAI). Participants were clustered into three profiles -- resilient, overcontrolled, and undercontrolled -- based on the Big-Five personality traits and ARC typology. Resilient workers achieved broad psychological gains primarily from the handbook, overcontrolled workers showed outcome-specific improvements with theory-driven AI, and undercontrolled workers exhibited minimal effects despite engaging with the frameworks. These patterns suggest personality as a predictor of readiness beyond stage-based tailoring: vulnerable users benefit from targeted rather than comprehensive interventions. The study advances understanding of personality-determined intervention prerequisites and highlights design implications for adaptive AI coaching systems that align support intensity with individual readiness, rather than assuming universal effectiveness. 

---
# First Logit Boosting: Visual Grounding Method to Mitigate Object Hallucination in Large Vision-Language Models 

**Authors**: Jiwoo Ha, Jongwoo Baek, Jinhyun So  

**Link**: [PDF](https://arxiv.org/pdf/2604.00455)  

**Abstract**: Recent Large Vision-Language Models (LVLMs) have demonstrated remarkable performance across various multimodal tasks that require understanding both visual and linguistic inputs. However, object hallucination -- the generation of nonexistent objects in answers -- remains a persistent challenge. Although several approaches such as retraining and external grounding methods have been proposed to mitigate this issue, they still suffer from high data costs or structural complexity. Training-free methods such as Contrastive Decoding (CD) are more cost-effective, avoiding additional training or external models, but still suffer from long-term decay, where visual grounding weakens and language priors dominate as the generation progresses. In this paper, we propose First Logit Boosting (FLB), a simple yet effective training-free technique designed to alleviate long-term decay in LVLMs. FLB stores the logit of the first generated token and adds it to subsequent token predictions, effectively mitigating long-term decay of visual information. We observe that FLB (1) sustains the visual information embedded in the first token throughout generation, and (2) suppresses hallucinated words through the stabilizing effect of the ``The'' token. Experimental results show that FLB significantly reduces object hallucination across various tasks, benchmarks, and backbone models. Notably, it causes negligible inference overhead, making it highly applicable to real-time multimodal systems. Code is available at this https URL 

---
# Polysemanticity or Polysemy? Lexical Identity Confounds Superposition Metrics 

**Authors**: Iyad Ait Hou, Rebecca Hwa  

**Link**: [PDF](https://arxiv.org/pdf/2604.00443)  

**Abstract**: If the same neuron activates for both "lender" and "riverside," standard metrics attribute the overlap to superposition--the neuron must be compressing two unrelated concepts. This work explores how much of the overlap is due a lexical confound: neurons fire for a shared word form (such as "bank") rather than for two compressed concepts. A 2x2 factorial decomposition reveals that the lexical-only condition (same word, different meaning) consistently exceeds the semantic-only condition (different word, same meaning) across models spanning 110M-70B parameters. The confound carries into sparse autoencoders (18-36% of features blend senses), sits in <=1% of activation dimensions, and hurts downstream tasks: filtering it out improves word sense disambiguation and makes knowledge edits more selective (p = 0.002). 

---
# G-Drift MIA: Membership Inference via Gradient-Induced Feature Drift in LLMs 

**Authors**: Ravi Ranjan, Utkarsh Grover, Xiaomin Lin, Agoritsa Polyzou  

**Link**: [PDF](https://arxiv.org/pdf/2604.00419)  

**Abstract**: Large language models (LLMs) are trained on massive web-scale corpora, raising growing concerns about privacy and copyright. Membership inference attacks (MIAs) aim to determine whether a given example was used during training. Existing LLM MIAs largely rely on output probabilities or loss values and often perform only marginally better than random guessing when members and non-members are drawn from the same distribution. We introduce G-Drift MIA, a white-box membership inference method based on gradient-induced feature drift. Given a candidate (x,y), we apply a single targeted gradient-ascent step that increases its loss and measure the resulting changes in internal representations, including logits, hidden-layer activations, and projections onto fixed feature directions, before and after the update. These drift signals are used to train a lightweight logistic classifier that effectively separates members from non-members. Across multiple transformer-based LLMs and datasets derived from realistic MIA benchmarks, G-Drift substantially outperforms confidence-based, perplexity-based, and reference-based attacks. We further show that memorized training samples systematically exhibit smaller and more structured feature drift than non-members, providing a mechanistic link between gradient geometry, representation stability, and memorization. In general, our results demonstrate that small, controlled gradient interventions offer a practical tool for auditing the membership of training-data and assessing privacy risks in LLMs. 

---
# Learning Humanoid Navigation from Human Data 

**Authors**: Weizhuo Wang, Yanjie Ze, C. Karen Liu, Monroe Kennedy III  

**Link**: [PDF](https://arxiv.org/pdf/2604.00416)  

**Abstract**: We present EgoNav, a system that enables a humanoid robot to traverse diverse, unseen environments by learning entirely from 5 hours of human walking data, with no robot data or finetuning. A diffusion model predicts distributions of plausible future trajectories conditioned on past trajectory, a 360 deg visual memory fusing color, depth, and semantics, and video features from a frozen DINOv3 backbone that capture appearance cues invisible to depth sensors. A hybrid sampling scheme achieves real-time inference in 10 denoising steps, and a receding-horizon controller selects paths from the predicted distribution. We validate EgoNav through offline evaluations, where it outperforms baselines in collision avoidance and multi-modal coverage, and through zero-shot deployment on a Unitree G1 humanoid across unseen indoor and outdoor environments. Behaviors such as waiting for doors to open, navigating around crowds, and avoiding glass walls emerge naturally from the learned prior. We will release the dataset and trained models. Our website: this https URL 

---
# COTTA: Context-Aware Transfer Adaptation for Trajectory Prediction in Autonomous Driving 

**Authors**: Seohyoung Park, Jaeyeol Lim, Seoyoung Ju, Kyeonghun Kim, Nam-Joon Kim, Hyuk-Jae Lee  

**Link**: [PDF](https://arxiv.org/pdf/2604.00402)  

**Abstract**: Developing robust models to accurately predict the trajectories of surrounding agents is fundamental to autonomous driving safety. However, most public datasets, such as the Waymo Open Motion Dataset and Argoverse, are collected in Western road environments and do not reflect the unique traffic patterns, infrastructure, and driving behaviors of other regions, including South Korea. This domain discrepancy leads to performance degradation when state-of-the-art models trained on Western data are deployed in different geographic contexts. In this work, we investigate the adaptability of Query-Centric Trajectory Prediction (QCNet) when transferred from U.S.-based data to Korean road environments. Using a Korean autonomous driving dataset, we compare four training strategies: zero-shot transfer, training from scratch, full fine-tuning, and encoder freezing. Experimental results demonstrate that leveraging pretrained knowledge significantly improves prediction performance. Specifically, selectively fine-tuning the decoder while freezing the encoder yields the best trade-off between accuracy and training efficiency, reducing prediction error by over 66% compared to training from scratch. This study provides practical insights into effective transfer learning strategies for deploying trajectory prediction models in new geographic domains. 

---
# Improving Generalization of Deep Learning for Brain Metastases Segmentation Across Institutions 

**Authors**: Yuchen Yang, Shuangyang Zhong, Haijun Yu, Langcuomu Suo, Hongbin Han, Florian Putz, Yixing Huang  

**Link**: [PDF](https://arxiv.org/pdf/2604.00397)  

**Abstract**: Background: Deep learning has demonstrated significant potential for automated brain metastases (BM) segmentation; however, models trained at a singular institution often exhibit suboptimal performance at various sites due to disparities in scanner hardware, imaging protocols, and patient demographics. The goal of this work is to create a domain adaptation framework that will allow for BM segmentation to be used across multiple institutions.
Methods: We propose a VAE-MMD preprocessing pipeline that combines variational autoencoders (VAE) with maximum mean discrepancy (MMD) loss, incorporating skip connections and self-attention mechanisms alongside nnU-Net segmentation. The method was tested on 740 patients from four public databases: Stanford, UCSF, UCLM, and PKG, evaluated by domain classifier's accuracy, sensitivity, precision, F1/F2 scores, surface Dice (sDice), and 95th percentile Hausdorff distance (HD95).
Results: VAE-MMD reduced domain classifier accuracy from 0.91 to 0.50, indicating successful feature alignment across institutions. Reconstructed volumes attained a PSNR greater than 36 dB, maintaining anatomical accuracy. The combined method raised the mean F1 by 11.1% (0.700 to 0.778), the mean sDice by 7.93% (0.7121 to 0.7686), and reduced the mean HD95 by 65.5% (11.33 to 3.91 mm) across all four centers compared to the baseline nnU-Net.
Conclusions: VAE-MMD effectively diminishes cross-institutional data heterogeneity and enhances BM segmentation generalization across volumetric, detection, and boundary-level metrics without necessitating target-domain labels, thereby overcoming a significant obstacle to the clinical implementation of AI-assisted segmentation. 

---
# Deep Networks Favor Simple Data 

**Authors**: Weyl Lu, Chenjie Hao, Yubei Chen  

**Link**: [PDF](https://arxiv.org/pdf/2604.00394)  

**Abstract**: Estimated density is often interpreted as indicating how typical a sample is under a model. Yet deep models trained on one dataset can assign \emph{higher} density to simpler out-of-distribution (OOD) data than to in-distribution test data. We refer to this behavior as the OOD anomaly. Prior work typically studies this phenomenon within a single architecture, detector, or benchmark, implicitly assuming certain canonical densities. We instead separate the trained network from the density estimator built from its representations or outputs. We introduce two estimators: Jacobian-based estimators and autoregressive self-estimators, making density analysis applicable to a wide range of models.
Applying this perspective to a range of models, including iGPT, PixelCNN++, Glow, score-based diffusion models, DINOv2, and I-JEPA, we find the same striking regularity that goes beyond the OOD anomaly: \textbf{lower-complexity samples receive higher estimated density, while higher-complexity samples receive lower estimated density}. This ordering appears within a test set and across OOD pairs such as CIFAR-10 and SVHN, and remains highly consistent across independently trained models. To quantify these orderings, we introduce Spearman rank correlation and find striking agreement both across models and with external complexity metrics. Even when trained only on the lowest-density (most complex) samples or \textbf{even a single such sample} the resulting models still rank simpler images as higher density.
These observations lead us beyond the original OOD anomaly to a more general conclusion: deep networks consistently favor simple data. Our goal is not to close this question, but to define and visualize it more clearly. We broaden its empirical scope and show that it appears across architectures, objectives, and density estimators. 

---
# EvolveTool-Bench: Evaluating the Quality of LLM-Generated Tool Libraries as Software Artifacts 

**Authors**: Alibek T. Kaliyev, Artem Maryanskyy  

**Link**: [PDF](https://arxiv.org/pdf/2604.00392)  

**Abstract**: Modern LLM agents increasingly create their own tools at runtime -- from Python functions to API clients -- yet existing benchmarks evaluate them almost exclusively by downstream task completion. This is analogous to judging a software engineer only by whether their code runs, ignoring redundancy, regression, and safety. We introduce EvolveTool-Bench, a diagnostic benchmark for LLM-generated tool libraries in software engineering workflows. Across three domains requiring actual tool execution (proprietary data formats, API orchestration, and numerical computation), we define library-level software quality metrics -- reuse, redundancy, composition success, regression stability, and safety -- alongside a per-tool Tool Quality Score measuring correctness, robustness, generality, and code quality. In the first head-to-head comparison of code-level and strategy-level tool evolution (ARISE vs. EvoSkill vs. one-shot baselines, 99 tasks, two models), we show that systems with similar task completion (63-68%) differ by up to 18% in library health, revealing software quality risks invisible to task-only evaluation. Our results highlight that evaluation and governance of LLM-generated tools require treating the evolving tool library as a first-class software artifact, not a black box. 

---
# RAGShield: Provenance-Verified Defense-in-Depth Against Knowledge Base Poisoning in Government Retrieval-Augmented Generation Systems 

**Authors**: KrishnaSaiReddy Patil  

**Link**: [PDF](https://arxiv.org/pdf/2604.00387)  

**Abstract**: RAG systems deployed across federal agencies for citizen-facing services are vulnerable to knowledge base poisoning attacks, where adversaries inject malicious documents to manipulate outputs. Recent work demonstrates that as few as 10 adversarial passages can achieve 98.2% retrieval success rates. We observe that RAG knowledge base poisoning is structurally analogous to software supply chain attacks, and propose RAGShield, a five-layer defense-in-depth framework applying supply chain provenance verification to the RAG knowledge pipeline. RAGShield introduces: (1) C2PA-inspired cryptographic document attestation blocking unsigned and forged documents at ingestion; (2) trust-weighted retrieval prioritizing provenance-verified sources; (3) a formal taint lattice with cross-source contradiction detection catching insider threats even when provenance is valid; (4) provenance-aware generation with auditable citations; and (5) NIST SP 800-53 compliance mapping across 15 control families. Evaluation on a 500-passage Natural Questions corpus with 63 attack documents and 200 queries against five adversary tiers achieves 0.0% attack success rate including adaptive attacks (95% CI: [0.0%, 1.9%]) with 0.0% false positive rate. We honestly report that insider in-place replacement attacks achieve 17.5% ASR, identifying the fundamental limit of ingestion-time defense. The cross-source contradiction detector catches subtle numerical manipulation attacks that bypass provenance verification entirely. 

---
# Go Big or Go Home: Simulating Mobbing Behavior with Braitenbergian Robots 

**Authors**: Elaheh Sanoubari  

**Link**: [PDF](https://arxiv.org/pdf/2604.00350)  

**Abstract**: We used the Webots robotics simulation platform to simulate a dyadic avoiding and mobbing predator behavior in a group of Braitenbergian robots. Mobbing is an antipredator adaptation used by some animals in which the individuals cooperatively attack or harass a predator to protect themselves. One way of coordinating a mobbing attack is using mobbing calls to summon other individuals of the mobbing species. We imitated this mechanism and simulated Braitenbergian robots that use mobbing calls when they face a light source (representing an inanimate predator) and mob it if they can summon allies, otherwise, they escape from it. We explore the effects of range of mobbing call (infinite range, mid-range and low-range) and the size of the robot group (ten robots vs three) on the overall success of mobbing. Our results suggest that both variables have significant impacts. This work has implications for simulations of action selection in artificial life and designing control architectures for autonomous agents. 

---
# The Persistent Vulnerability of Aligned AI Systems 

**Authors**: Aengus Lynch  

**Link**: [PDF](https://arxiv.org/pdf/2604.00324)  

**Abstract**: Autonomous AI agents are being deployed with filesystem access, email control, and multi-step planning. This thesis contributes to four open problems in AI safety: understanding dangerous internal computations, removing dangerous behaviors once embedded, testing for vulnerabilities before deployment, and predicting when models will act against deployers.
ACDC automates circuit discovery in transformers, recovering all five component types from prior manual work on GPT-2 Small by selecting 68 edges from 32,000 candidates in hours rather than months.
Latent Adversarial Training (LAT) removes dangerous behaviors by optimizing perturbations in the residual stream to elicit failure modes, then training under those perturbations. LAT solved the sleeper agent problem where standard safety training failed, matching existing defenses with 700x fewer GPU hours.
Best-of-N jailbreaking achieves 89% attack success on GPT-4o and 78% on Claude 3.5 Sonnet through random input augmentations. Attack success follows power law scaling across text, vision, and audio, enabling quantitative forecasting of adversarial robustness.
Agentic misalignment tests whether frontier models autonomously choose harmful actions given ordinary goals. Across 16 models, agents engaged in blackmail (96% for Claude Opus 4), espionage, and actions causing death. Misbehavior rates rose from 6.5% to 55.1% when models stated scenarios were real rather than evaluations.
The thesis does not fully resolve any of these problems but makes each tractable and measurable. 

---
# Prompt-Guided Prefiltering for VLM Image Compression 

**Authors**: Bardia Azizian, Ivan V. Bajic  

**Link**: [PDF](https://arxiv.org/pdf/2604.00314)  

**Abstract**: The rapid progress of large Vision-Language Models (VLMs) has enabled a wide range of applications, such as image understanding and Visual Question Answering (VQA). Query images are often uploaded to the cloud, where VLMs are typically hosted, hence efficient image compression becomes crucial. However, traditional human-centric codecs are suboptimal in this setting because they preserve many task-irrelevant details. Existing Image Coding for Machines (ICM) methods also fall short, as they assume a fixed set of downstream tasks and cannot adapt to prompt-driven VLMs with an open-ended variety of objectives. We propose a lightweight, plug-and-play, prompt-guided prefiltering module to identify image regions most relevant to the text prompt, and consequently to the downstream task. The module preserves important details while smoothing out less relevant areas to improve compression efficiency. It is codec-agnostic and can be applied before conventional and learned encoders. Experiments on several VQA benchmarks show that our approach achieves a 25-50% average bitrate reduction while maintaining the same task accuracy. Our source code is available at this https URL. 

---
# Robust Multimodal Safety via Conditional Decoding 

**Authors**: Anurag Kumar, Raghuveer Peri, Jon Burnsky, Alexandru Nelus, Rohit Paturi, Srikanth Vishnubhotla, Yanjun Qi  

**Link**: [PDF](https://arxiv.org/pdf/2604.00310)  

**Abstract**: Multimodal large-language models (MLLMs) often experience degraded safety alignment when harmful queries exploit cross-modal interactions. Models aligned on text alone show a higher rate of successful attacks when extended to two or more modalities. In this work, we propose a simple conditional decoding strategy, CASA (Classification Augmented with Safety Attention) that utilizes internal representations of MLLMs to predict a binary safety token before response generation. We introduce a novel safety attention module designed to enhance the model's ability to detect malicious queries. Our design ensures robust safety alignment without relying on any external classifier or auxiliary head, and without the need for modality-specific safety fine-tuning. On diverse benchmarks such as MM-SafetyBench, JailbreakV-28k, and adversarial audio tests, CASA lowers the average attack success rate by more than 97% across modalities and across attack types. Our empirical evaluations also show that CASA maintains strong utility in benign inputs, a result validated through both automated and human evaluations (via 13 trained annotators). Together, these results highlight CASA as a simple and generalizable framework to improve multimodal LLM safety. 

---
# Asymmetric Actor-Critic for Multi-turn LLM Agents 

**Authors**: Shuli Jiang, Zhaoyang Zhang, Yi Zhang, Shuo Yang, Wei Xia, Stefano Soatto  

**Link**: [PDF](https://arxiv.org/pdf/2604.00304)  

**Abstract**: Large language models (LLMs) exhibit strong reasoning and conversational abilities, but ensuring reliable behavior in multi-turn interactions remains challenging. In many real-world applications, agents must succeed in one-shot settings where retries are impossible. Existing approaches either rely on reflection or post-hoc evaluation, which require additional attempts, or assume fully trainable models that cannot leverage proprietary LLMs. We propose an asymmetric actor-critic framework for reliable conversational agents. A powerful proprietary LLM acts as the actor, while a smaller open-source critic provides runtime supervision, monitoring the actor's actions and intervening within the same interaction trajectory. Unlike training-based actor-critic methods, our framework supervises a fixed actor operating in open-ended conversational environments. The design leverages a generation-verification asymmetry: while high-quality generation requires large models, effective oversight can often be achieved by smaller ones. We further introduce a data generation pipeline that produces supervision signals for critic fine-tuning without modifying the actor. Experiments on $\tau$-bench and UserBench show that our approach significantly improves reliability and task success over strong single-agent baselines. Moreover, lightweight open-source critics rival or surpass larger proprietary models in the critic role, and critic fine-tuning yields additional gains over several state-of-the-art methods. 

---
# SANA I2I: A Text Free Flow Matching Framework for Paired Image to Image Translation with a Case Study in Fetal MRI Artifact Reduction 

**Authors**: Italo Felix Santos, Gilson Antonio Giraldi, Heron Werner Junior  

**Link**: [PDF](https://arxiv.org/pdf/2604.00298)  

**Abstract**: We propose SANA-I2I, a text-free high-resolution image-to-image generation framework that extends the SANA family by removing textual conditioning entirely. In contrast to SanaControlNet, which combines text and image-based control, SANA-I2I relies exclusively on paired source-target images to learn a conditional flow-matching model in latent space. The model learns a conditional velocity field that maps a target image distribution to another one, enabling supervised image translation without reliance on language prompts. We evaluate the proposed approach on the challenging task of fetal MRI motion artifact reduction. To enable paired training in this application, where real paired data are difficult to acquire, we adopt a synthetic data generation strategy based on the method proposed by Duffy et al., which simulates realistic motion artifacts in fetal magnetic resonance imaging (MRI). Experimental results demonstrate that SANA-I2I effectively suppresses motion artifacts while preserving anatomical structure, achieving competitive performance few inference steps. These results highlight the efficiency and suitability of our proposed flow-based, text-free generative models for supervised image-to-image tasks in medical imaging. 

---
# VeriAct: Beyond Verifiability -- Agentic Synthesis of Correct and Complete Formal Specifications 

**Authors**: Md Rakib Hossain Misu, Iris Ma, Cristina V. Lopes  

**Link**: [PDF](https://arxiv.org/pdf/2604.00280)  

**Abstract**: Formal specifications play a central role in ensuring software reliability and correctness. However, automatically synthesizing high-quality formal specifications remains a challenging task, often requiring domain expertise. Recent work has applied large language models to generate specifications in Java Modeling Language (JML), reporting high verification pass rates. But does passing a verifier mean that the specification is actually correct and complete? In this work, we first conduct a comprehensive evaluation comparing classical and prompt-based approaches for automated JML specification synthesis. We then investigate whether prompt optimization can push synthesis quality further by evolving prompts through structured verification feedback. While optimization improves verifier pass rates, we find a clear performance ceiling. More critically, we propose Spec-Harness, an evaluation framework that measures specification correctness and completeness through symbolic verification, revealing that a large fraction of verifier-accepted specifications, including optimized ones, are in fact incorrect or incomplete, over- or under-constraining both inputs and outputs in ways invisible to the verifier. To push beyond this ceiling, we propose VeriAct, a verification-guided agentic framework that iteratively synthesizes and repairs specifications through a closed loop of LLM-driven planning, code execution, verification, and Spec-Harness feedback. Our experiments on two benchmark datasets show that VeriAct outperforms both prompt-based and prompt-optimized baselines, producing specifications that are not only verifiable but also correct and complete. 

---
# The Geometry of Compromise: Unlocking Generative Capabilities via Controllable Modality Alignment 

**Authors**: Hongyuan Liu, Qinli Yang, Wen Li, Zhong Zhang, Jiaming Liu, Wei Han, Zhili Qin, Jinxia Guo, Junming Shao  

**Link**: [PDF](https://arxiv.org/pdf/2604.00279)  

**Abstract**: Vision-Language Models (VLMs) such as CLIP learn a shared embedding space for images and text, yet their representations remain geometrically separated, a phenomenon known as the modality gap. This gap limits tasks requiring cross-modal interchangeability, such as captioning and joint clustering. Existing post-processing approaches can partially improve cross-modal compatibility; however, we show through geometric analysis that they primarily reduce the global centroid offset while leaving the underlying distributional mismatch intact. We decompose the modality gap into a Centroid Gap and a Distribution Gap, and demonstrate that the Distribution Gap is the true predictor of cross-modal task quality ($R^2 = 0.986$), whereas the commonly used Raw Gap is misleading ($R^2 = 0.691$). Motivated by this observation, we propose TPC-CMA (Three-Phase Curriculum for Cross-Modal Alignment), a fine-tuning framework that explicitly reduces both components. The proposed CMA jointly mitigates centroid offsets and reshapes the distributional structure, while a three-phase curriculum with gradient-aware scheduling progressively introduces alignment during training to enable stable optimization. Experiments demonstrate that our method significantly improves cross-modal alignment. With $\alpha_{\text{target}}{=}0.05$, the modality gap is reduced by 66.6\% with only 4.84\% accuracy drop. Under stronger alignment ($\alpha_{\text{target}}{=}0.5$), the gap is reduced by 82.3\%, clustering ARI improves from 0.318 to 0.516, and captioning CIDEr increases by 57.1\% over the original model. Our code and pre-trained models will be made publicly available upon acceptance. 

---
# Hybrid Energy-Based Models for Physical AI: Provably Stable Identification of Port-Hamiltonian Dynamics 

**Authors**: Simone Betteti, Luca Laurenti  

**Link**: [PDF](https://arxiv.org/pdf/2604.00277)  

**Abstract**: Energy-based models (EBMs) implement inference as gradient descent on a learned Lyapunov function, yielding interpretable, structure-preserving alternatives to black-box neural ODEs and aligning naturally with physical AI. Yet their use in system identification remains limited, and existing architectures lack formal stability guarantees that globally preclude unstable modes. We address this gap by introducing an EBM framework for system identification with stable, dissipative, absorbing invariant dynamics. Unlike classical global Lyapunov stability, absorbing invariance expands the class of stability-preserving architectures, enabling more flexible and expressive EBMs. We extend EBM theory to nonsmooth activations by establishing negative energy dissipation via Clarke derivatives and deriving new conditions for radial unboundedness, exposing a stability-expressivity tradeoff in standard EBMs. To overcome this, we introduce a hybrid architecture with a dynamical visible layer and static hidden layers, prove absorbing invariance under mild assumptions, and show that these guarantees extend to port-Hamiltonian EBMs. Experiments on metric-deformed multi-well and ring systems validate the approach, showcasing how our hybrid EBM architecture combines expressivity with sound and provable safety guarantees by design. 

---
# Benchmarking Interaction, Beyond Policy: a Reproducible Benchmark for Collaborative Instance Object Navigation 

**Authors**: Edoardo Zorzi, Francesco Taioli, Yiming Wang, Marco Cristani, Alessandro Farinelli, Alberto Castellini, Loris Bazzani  

**Link**: [PDF](https://arxiv.org/pdf/2604.00265)  

**Abstract**: We propose Question-Asking Navigation (QAsk-Nav), the first reproducible benchmark for Collaborative Instance Object Navigation (CoIN) that enables an explicit, separate assessment of embodied navigation and collaborative question asking. CoIN tasks an embodied agent with reaching a target specified in free-form natural language under partial observability, using only egocentric visual observations and interactive natural-language dialogue with a human, where the dialogue can help to resolve ambiguity among visually similar object instances. Existing CoIN benchmarks are primarily focused on navigation success and offer no support for consistent evaluation of collaborative interaction. To address this limitation, QAsk-Nav provides (i) a lightweight question-asking protocol scored independently of navigation, (ii) an enhanced navigation protocol with realistic, diverse, high-quality target descriptions, and (iii) an open-source dataset, that includes 28,000 quality-checked reasoning and question-asking traces for training and analysis of interactive capabilities of CoIN models. Using the proposed QAsk-Nav benchmark, we develop Light-CoNav, a lightweight unified model for collaborative navigation that is 3x smaller and 70x faster than existing modular methods, while outperforming state-of-the-art CoIN approaches in generalization to unseen objects and environments. Project page at this https URL 

---
# LLM Essay Scoring Under Holistic and Analytic Rubrics: Prompt Effects and Bias 

**Authors**: Filip J. Kucia, Anirban Chakraborty, Anna Wróblewska  

**Link**: [PDF](https://arxiv.org/pdf/2604.00259)  

**Abstract**: Despite growing interest in using Large Language Models (LLMs) for educational assessment, it remains unclear how closely they align with human scoring. We present a systematic evaluation of instruction-tuned LLMs across three open essay-scoring datasets (ASAP 2.0, ELLIPSE, and DREsS) that cover both holistic and analytic scoring. We analyze agreement with human consensus scores, directional bias, and the stability of bias estimates. Our results show that strong open-weight models achieve moderate to high agreement with humans on holistic scoring (Quadratic Weighted Kappa about 0.6), but this does not transfer uniformly to analytic scoring. In particular, we observe large and stable negative directional bias on Lower-Order Concern (LOC) traits, such as Grammar and Conventions, meaning that models often score these traits more harshly than human raters. We also find that concise keyword-based prompts generally outperform longer rubric-style prompts in multi-trait analytic scoring. To quantify the amount of data needed to detect these systematic deviations, we compute the minimum sample size at which a 95% bootstrap confidence interval for the mean bias excludes zero. This analysis shows that LOC bias is often detectable with very small validation sets, whereas Higher-Order Concern (HOC) traits typically require much larger samples. These findings support a bias-correction-first deployment strategy: instead of relying on raw zero-shot scores, systematic score offsets can be estimated and corrected using small human-labeled bias-estimation sets, without requiring large-scale fine-tuning. 

---
# Hierarchical Apprenticeship Learning from Imperfect Demonstrations with Evolving Rewards 

**Authors**: Md Mirajul Islam, Rajesh Debnath, Adittya Soukarjya Saha, Min Chi  

**Link**: [PDF](https://arxiv.org/pdf/2604.00258)  

**Abstract**: While apprenticeship learning has shown promise for inducing effective pedagogical policies directly from student interactions in e-learning environments, most existing approaches rely on optimal or near-optimal expert demonstrations under a fixed reward. Real-world student interactions, however, are often inherently imperfect and evolving: students explore, make errors, revise strategies, and refine their goals as understanding develops. In this work, we argue that imperfect student demonstrations are not noise to be discarded, but structured signals-provided their relative quality is ranked. We introduce HALIDE, Hierarchical Apprenticeship Learning from Imperfect Demonstrations with Evolving Rewards, which not only leverages sub-optimal student demonstrations, but ranks them within a hierarchical learning framework. HALIDE models student behavior at multiple levels of abstraction, enabling inference of higher-level intent and strategy from suboptimal actions while explicitly capturing the temporal evolution of student reward functions. By integrating demonstration quality into hierarchical reward inference,HALIDE distinguishes transient errors from suboptimal strategies and meaningful progress toward higher-level learning goals. Our results show that HALIDE more accurately predicts student pedagogical decisions than approaches that rely on optimal trajectories, fixed rewards, or unranked imperfect demonstrations. 

---
# REM-CTX: Automated Peer Review via Reinforcement Learning with Auxiliary Context 

**Authors**: Pawin Taechoyotin, Daniel E. Acuna  

**Link**: [PDF](https://arxiv.org/pdf/2604.00248)  

**Abstract**: Most automated peer review systems rely on textual manuscript content alone, leaving visual elements such as figures and external scholarly signals underutilized. We introduce REM-CTX, a reinforcement-learning system that incorporates auxiliary context into the review generation process via correspondence-aware reward functions. REM-CTX trains an 8B-parameter language model with Group Relative Policy Optimization (GRPO) and combines a multi-aspect quality reward with two correspondence rewards that explicitly encourage alignment with auxiliary context. Experiments on manuscripts across Computer, Biological, and Physical Sciences show that REM-CTX achieves the highest overall review quality among six baselines, outperforming other systems with substantially larger commercial models, and surpassing the next-best RL baseline across both quality and contextual grounding metrics. Ablation studies confirm that the two correspondence rewards are complementary: each selectively improves its targeted correspondence reward while preserving all quality dimensions, and the full model outperforms all partial variants. Analysis of training dynamics reveals that the criticism aspect is negatively correlated with other metrics during training, suggesting that future studies should group multi-dimension rewards for review generation. 

---
# Softmax gradient policy for variance minimization and risk-averse multi armed bandits 

**Authors**: Gabriel Turinici  

**Link**: [PDF](https://arxiv.org/pdf/2604.00241)  

**Abstract**: Algorithms for the Multi-Armed Bandit (MAB) problem play a central role in sequential decision-making and have been extensively explored both theoretically and numerically. While most classical approaches aim to identify the arm with the highest expected reward, we focus on a risk-aware setting where the goal is to select the arm with the lowest variance, favoring stability over potentially high but uncertain returns. To model the decision process, we consider a softmax parameterization of the policy; we propose a new algorithm to select the minimal variance (or minimal risk) arm and prove its convergence under natural conditions. The algorithm constructs an unbiased estimate of the objective by using two independent draws from the current's arm distribution. We provide numerical experiments that illustrate the practical behavior of these algorithms and offer guidance on implementation choices. The setting also covers general risk-aware problems where there is a trade-off between maximizing the average reward and minimizing its variance. 

---
# AI-Mediated Explainable Regulation for Justice 

**Authors**: Thomas Hofweber, Andreas Sudmann, Evangelos Pournaras  

**Link**: [PDF](https://arxiv.org/pdf/2604.00237)  

**Abstract**: Present practice of deciding on regulation faces numerous problems that make adopted regulations static, unexplained, unduly influenced by powerful interest groups, and stained with a perception of illegitimacy. These well-known problems with the regulatory process can lead to injustice and have substantial negative effects on society and democracy. We discuss a new approach that utilizes distributed artificial intelligence (AI) to make a regulatory recommendation that is explainable and adaptable by design. We outline the main components of a system that can implement this approach and show how it would resolve the problems with the present regulatory system. This approach models and reasons about stakeholder preferences with separate preference models, while it aggregates these preferences in a value sensitive way. Such recommendations can be updated due to changes in facts or in values and are inherently explainable. We suggest how stakeholders can make their preferences known to the system and how they can verify whether they were properly considered in the regulatory decision. The resulting system promises to support regulatory justice, legitimacy, and compliance. 

---
# MAC-Attention: a Match-Amend-Complete Scheme for Fast and Accurate Attention Computation 

**Authors**: Jinghan Yao, Sam Adé Jacobs, Walid Krichene, Masahiro Tanaka, Dhabaleswar K Panda  

**Link**: [PDF](https://arxiv.org/pdf/2604.00235)  

**Abstract**: Long-context decoding in LLMs is IO-bound: each token re-reads an ever-growing KV cache. Prior accelerations cut bytes via compression, which lowers fidelity, or selection/eviction, which restricts what remains accessible, and both can degrade delayed recall and long-form generation. We introduce MAC-Attention, a fidelity- and access-preserving alternative that accelerates decoding by reusing prior attention computations for semantically similar recent queries. It starts with a match stage that performs pre-RoPE L2 matching over a short local window; an amend stage rectifies the reused attention by recomputing a small band near the match boundary; and a complete stage fuses the rectified results with fresh attention computed on the KV tail through a numerically stable merge. On a match hit, the compute and bandwidth complexity is constant regardless of context length. The method is model-agnostic and composes with IO-aware kernels, paged-KV managers, and MQA/GQA. Across LongBench v2 (120K), RULER (120K), and LongGenBench (16K continuous generation), compared to the latest FlashInfer library, MAC-Attention reduces KV accesses by up to 99%, cuts token generation latency by over 60% at 128K, and achieves over 14.3x attention-phase speedups, up to 2.6x end-to-end, while maintaining full-attention quality. By reusing computation, MAC-Attention delivers long-context inference that is both fast and faithful. Code is available here: this https URL 

---
# Diversity-Aware Reverse Kullback-Leibler Divergence for Large Language Model Distillation 

**Authors**: Hoang-Chau Luong, Dat Ba Tran, Lingwei Chen  

**Link**: [PDF](https://arxiv.org/pdf/2604.00223)  

**Abstract**: Reverse Kullback-Leibler (RKL) divergence has recently emerged as the preferred objective for large language model (LLM) distillation, consistently outperforming forward KL (FKL), particularly in regimes with large vocabularies and significant teacher-student capacity mismatch, where RKL focuses learning on dominant modes rather than enforcing dense alignment. However, RKL introduces a structural limitation that drives the student toward overconfident predictions. We first provide an analysis of RKL by decomposing its gradients into target and non-target components, and show that non-target gradients consistently push the target logit upward even when the student already matches the teacher, thereby reducing output diversity. In addition, RKL provides weak supervision over non-target classes, leading to poor tail alignment. To address these issues, we propose Diversity-aware RKL (DRKL), which removes this gradient effect and strengthens non-target supervision while preserving the optimization benefits of RKL. Extensive experiments across datasets and model families demonstrate that DRKL consistently outperforms FKL, RKL, and other state-of-the-art distillation objectives, achieving better performance and a superior fidelity-diversity trade-off. 

---
# QUEST: A robust attention formulation using query-modulated spherical attention 

**Authors**: Hariprasath Govindarajan, Per Sidén, Jacob Roll, Fredrik Lindsten  

**Link**: [PDF](https://arxiv.org/pdf/2604.00199)  

**Abstract**: The Transformer model architecture has become one of the most widely used in deep learning and the attention mechanism is at its core. The standard attention formulation uses a softmax operation applied to a scaled dot product between query and key vectors. We explore the role played by norms of the queries and keys, which can cause training instabilities when they arbitrarily increase. We demonstrate how this can happen even in simple Transformer models, in the presence of easy-to-learn spurious patterns in the data. We propose a new attention formulation, QUEry-modulated Spherical aTtention (QUEST), that constrains the keys to a hyperspherical latent space, while still allowing individual tokens to flexibly control the sharpness of the attention distribution. QUEST can be easily used as a drop-in replacement for standard attention. We focus on vision applications while also exploring other domains to highlight the method's generality. We show that (1) QUEST trains without instabilities and (2) produces models with improved performance (3) that are robust to data corruptions and adversarial attacks. 

---
# Making Sense of AI Agents Hype: Adoption, Architectures, and Takeaways from Practitioners 

**Authors**: Ruoyu Su, Matteo Esposito, Roberta Capuano, Rafiullah Omar, June Sallou, Henry Muccini, Davide Taibi  

**Link**: [PDF](https://arxiv.org/pdf/2604.00189)  

**Abstract**: To support practitioners in understanding how agentic systems are designed in real-world industrial practice, we present a review of practitioner conference talks on AI agents. We analyzed 138 recorded talks to examine how companies adopt agent-based architectures (Objective 1), identify recurring architectural strategies and patterns (Objective 2), and analyze application domains and technologies used to implement and operate LLM-driven agentic systems (Objective 3). 

---
# Explainable AI for Blind and Low-Vision Users: Navigating Trust, Modality, and Interpretability in the Agentic Era 

**Authors**: Abu Noman Md Sakib, Protik Dey, Zijie Zhang, Taslima Akter  

**Link**: [PDF](https://arxiv.org/pdf/2604.00187)  

**Abstract**: Explainable Artificial Intelligence (XAI) is critical for ensuring trust and accountability, yet its development remains predominantly visual. For blind and low-vision (BLV) users, the lack of accessible explanations creates a fundamental barrier to the independent use of AI-driven assistive technologies. This problem intensifies as AI systems shift from single-query tools into autonomous agents that take multi-step actions and make consequential decisions across extended task horizons, where a single undetected error can propagate irreversibly before any feedback is available. This paper investigates the unique XAI requirements of the BLV community through a comprehensive analysis of user interviews and contemporary research. By examining usage patterns across environmental perception and decision support, we identify a significant modality gap. Empirical evidence suggests that while BLV users highly value conversational explanations, they frequently experience "self-blame" for AI failures. The paper concludes with a research agenda for accessible Explainable AI in agentic systems, advocating for multimodal interfaces, blame-aware explanation design, and participatory development. 

---
# Agentic AI and Occupational Displacement: A Multi-Regional Task Exposure Analysis of Emerging Labor Market Disruption 

**Authors**: Ravish Gupta, Saket Kumar  

**Link**: [PDF](https://arxiv.org/pdf/2604.00186)  

**Abstract**: This paper extends the Acemoglu-Restrepo task exposure framework to address the labor market effects of agentic artificial intelligence systems: autonomous AI agents capable of completing entire occupational workflows rather than discrete tasks. Unlike prior automation technologies that substitute for individual subtasks, agentic AI systems execute end-to-end workflows involving multi-step reasoning, tool invocation, and autonomous decision-making, substantially expanding occupational displacement risk beyond what existing task-level analyses capture. We introduce the Agentic Task Exposure (ATE) score, a composite measure computed algorithmically from O*NET task data using calibrated adoption parameters--not a regression estimate--incorporating AI capability scores, workflow coverage factors, and logistic adoption velocity. Applying the ATE framework across five major US technology regions (Seattle-Tacoma, San Francisco Bay Area, Austin, New York, and Boston) over a 2025-2030 horizon, we find that 93.2% of the 236 analyzed occupations across six information-intensive SOC groups (financial, legal, healthcare, healthcare support, sales, and administrative/clerical) cross the moderate-risk threshold (ATE >= 0.35) in Tier 1 regions by 2030, with credit analysts, judges, and sustainability specialists reaching ATE scores of 0.43-0.47. We simultaneously identify seventeen emerging occupational categories benefiting from reinstatement effects, concentrated in human-AI collaboration, AI governance, and domain-specific AI operations roles. Our findings carry implications for workforce transition policy, regional economic planning, and the temporal dynamics of labor market adjustment 

---
# NFC based inventory control system for secure and efficient communication 

**Authors**: Razi Iqbal, Awais Ahmad, Asfandyar Gillani  

**Link**: [PDF](https://arxiv.org/pdf/2604.00181)  

**Abstract**: This paper brings up this idea of using Near Field Communication (NFC) for inventory control system instead of using traditional barcodes. NFC because of its high security, ease of use and efficiency can be very suitable for systems like inventory control. In traditional inventory control systems, each product has a barcode pasted on it, which is vulnerable to attacks as barcodes are open and have no security. Furthermore, barcodes are prone to damages and can be unreliable when pasted on different types of products e.g. hot and frozen products, circular shaped products and irregular shaped products like clothes etc. NFC on the other hand is very efficient, secure and reliable when it comes to short-range wireless communication. In this paper we will present our prototype for the inventory control system of an electronic store in which each product has a passive NFC tag pasted to it. When a customer buys a product the receipt of the product is generated using NFC between the NFC passive tag on the product and NFC enabled device (e.g. smart phone or reader) at the cash counter. 

---
# Unified Architecture Metamodel of Information Systems Developed by Generative AI 

**Authors**: Oleg Grynets, Vasyl Lyashkevych  

**Link**: [PDF](https://arxiv.org/pdf/2604.00171)  

**Abstract**: The rapid development of AI and LLMs has driven new methods of SDLC, in which a large portion of code, technical, and business documentation is generated automatically. However, since there is no single architectural framework that can provide consistent, repeatable transformations across different representation layers of information systems, such systems remain fragmented in their system representation. This study explores the problem of creating a unified architecture for LLM-oriented applications based on selected architectural frameworks by SMEs. A framework structure is proposed that covers some key types of architectural diagrams and supports a closed cycle of transformations, such as: "Code to Documentation to Code". The key architectural diagrams are split equally between main architectural layers: high-layer (business and domain understanding), middle-layer (system architecture), and low-layer (developer-layer architecture). Each architectural layer still contains some abstraction layers, which make it more flexible and better fit the requirements of design principles and architectural patterns. The conducted experiments demonstrated the stable quality of generated documentation and code when using a structured architectural context in the form of architectural diagrams. The results confirm that the proposed unified architecture metamodel can serve as an effective interface between humans and models, improving the accuracy, stability, and repeatability of LLM generation. However, the selected set of architectural diagrams should be optimised to avoid redundancy between some diagrams, and some diagrams should be updated to represent extra contextual orchestration. This work demonstrates measurable improvements for a new generation of intelligent tools that automate the SDLC and enable a comprehensive architecture compatible with AI-driven development. 

---
# Neural-Assisted in-Motion Self-Heading Alignment 

**Authors**: Zeev Yampolsky, Felipe O. Silva, Adriano Frutuoso, Itzik Klein  

**Link**: [PDF](https://arxiv.org/pdf/2604.00168)  

**Abstract**: Autonomous platforms operating in the oceans require accurate navigation to successfully complete their mission. In this regard, the initial heading estimation accuracy and the time required to achieve it play a critical role. The initial heading is traditionally estimated by model-based approaches employing orientation decomposition. However, methods such as the dual vector decomposition and optimized attitude decomposition achieve satisfactory heading accuracy only after long alignment times. To allow rapid and accurate initial heading estimation, we propose an end-to-end, model-free, neural-assisted framework using the same inputs as the model-based approaches. Our proposed approach was trained and evaluated on real-world dataset captured by an autonomous surface vehicle. Our approach shows a significant accuracy improvement over the model-based approaches achieving an average absolute error improvement of 53%. Additionally, our proposed approach was able to reduce the alignment time by up to 67%. Thus, by employing our proposed approach, the reduction in alignment time and improved accuracy allow for a shorter deployment time of an autonomous platform and increased navigation accuracy during the mission. 

---
# A Study on the Impact of Fault localization Granularity for Repository-Scale Code Repair Tasks 

**Authors**: Joseph Townsend, Chandresh Pravin, Kwun Ho Ngan, Matthieu Parizy  

**Link**: [PDF](https://arxiv.org/pdf/2604.00167)  

**Abstract**: Automatic program repair can be a challenging task, especially when resolving complex issues at a repository-level, which often involves issue reproduction, fault localization, code repair, testing and validation. Issues of this scale can be commonly found in popular GitHub repositories or datasets that are derived from them.
Some repository-level approaches separate localization and repair into distinct phases. Where this is the case, the fault localization approaches vary in terms of the granularity of localization. Where the impact of granularity is explored to some degree for smaller datasets, not all isolate this issue from the separate question of localization accuracy by testing code repair under the assumption of perfect fault localization. To the best of the authors' knowledge, no repository-scale studies have explicitly investigated granularity under this assumption, nor conducted a systematic empirical comparison of granularity levels in isolation.
We propose a framework for performing such tests by modifying the localization phase of the Agentless framework to retrieve ground-truth localization data and include this as context in the prompt fed to the repair phase. We show that under this configuration and as a generalization over the SWE-Bench-Mini dataset, function-level granularity yields the highest repair rate against line-level and file-level. However, a deeper dive suggests that the ideal granularity may in fact be task dependent.
This study is not intended to improve on the state-of-the-art, nor do we intend for results to be compared against any complete agentic frameworks. Rather, we present a proof of concept for investigating how fault localization may impact automatic code repair in repository-scale scenarios. We present preliminary findings to this end and encourage further research into this relationship between the two phases. 

---
# Epileptic Seizure Detection in Separate Frequency Bands Using Feature Analysis and Graph Convolutional Neural Network (GCN) from Electroencephalogram (EEG) Signals 

**Authors**: Ferdaus Anam Jibon, Fazlul Hasan Siddiqui, F. Deeba, Gahangir Hossain  

**Link**: [PDF](https://arxiv.org/pdf/2604.00163)  

**Abstract**: Epileptic seizures are neurological disorders characterized by abnormal and excessive electrical activity in the brain, resulting in recurrent seizure events. Electroencephalogram (EEG) signals are widely used for seizure diagnosis due to their ability to capture temporal and spatial neural dynamics. While recent deep learning methods have achieved high detection accuracy, they often lack interpretability and neurophysiological relevance. This study presents a frequency-aware framework for epileptic seizure detection based on ictal-phase EEG analysis. The raw EEG signals are decomposed into five frequency bands (delta, theta, alpha, lower beta, and higher beta), and eleven discriminative features are extracted from each band. A graph convolutional neural network (GCN) is then employed to model spatial dependencies among EEG electrodes, represented as graph nodes. Experiments on the CHB-MIT scalp EEG dataset demonstrate high detection performance, achieving accuracies of 97.1%, 97.13%, 99.5%, 99.7%, and 51.4% across the respective frequency bands, with an overall broadband accuracy of 99.01%. The results highlight the strong discriminative capability of mid-frequency bands and reveal frequency-specific seizure patterns. The proposed approach improves interpretability and diagnostic precision compared to conventional broadband EEG-based methods. 

---
# Oblivion: Self-Adaptive Agentic Memory Control through Decay-Driven Activation 

**Authors**: Ashish Rana, Chia-Chien Hung, Qumeng Sun, Julian Martin Kunkel, Carolin Lawrence  

**Link**: [PDF](https://arxiv.org/pdf/2604.00131)  

**Abstract**: Human memory adapts through selective forgetting: experiences become less accessible over time but can be reactivated by reinforcement or contextual cues. In contrast, memory-augmented LLM agents rely on "always-on" retrieval and "flat" memory storage, causing high interference and latency as histories grow. We introduce Oblivion, a memory control framework that casts forgetting as decay-driven reductions in accessibility, not explicit deletion. Oblivion decouples memory control into read and write paths. The read path decides when to consult memory, based on agent uncertainty and memory buffer sufficiency, avoiding redundant always-on access. The write path decides what to strengthen, by reinforcing memories contributing to forming the response. Together, this enables hierarchical memory organization that maintains persistent high-level strategies while dynamically loading details as needed. We evaluate on both static and dynamic long-horizon interaction benchmarks. Results show that Oblivion dynamically adapts memory access and reinforcement, balancing learning and forgetting under shifting contexts, highlighting that memory control is essential for effective LLM-agentic reasoning. The source code is available at this https URL. 

---
# From Domain Understanding to Design Readiness: a playbook for GenAI-supported learning in Software Engineering 

**Authors**: Rafal Wlodarski  

**Link**: [PDF](https://arxiv.org/pdf/2604.00120)  

**Abstract**: Software engineering courses often require rapid upskilling in supporting knowledge areas such as domain understanding and modeling methods. We report an experience from a two-week milestone in a master's course where 29 students used a customized ChatGPT (GPT-3.5) tutor grounded in a curated course knowledge base to learn cryptocurrency-finance basics and Domain-Driven Design (DDD). We logged all interactions and evaluated a 34.5% random sample of prompt-answer pairs (60/~174) with a five-dimension rubric (accuracy, relevance, pedagogical value, cognitive load, supportiveness), and we collected pre/post self-efficacy. Responses were consistently accurate and relevant in this setting: accuracy averaged 98.9% with no factual errors and only 2/60 minor inaccuracies, and relevance averaged 92.2%. Pedagogical value was high (89.4%) with generally appropriate cognitive load (82.78%), but supportiveness was low (37.78%). Students reported large pre-post self-efficacy gains for genAI-assisted domain learning and DDD application. From these observations we distill seventeen concrete teaching practices spanning prompt/configuration and course/workflow design (e.g., setting expected granularity, constraining verbosity, curating guardrail examples, adding small credit with a simple quality rubric). Within this single-course context, results suggest that genAI-supported learning can complement instruction in domain understanding and modeling tasks, while leaving room to improve tone and follow-up structure. 

---
# Hierarchical Pre-Training of Vision Encoders with Large Language Models 

**Authors**: Eugene Lee, Ting-Yu Chang, Jui-Huang Tsai, Jiajie Diao, Chen-Yi Lee  

**Link**: [PDF](https://arxiv.org/pdf/2604.00086)  

**Abstract**: The field of computer vision has experienced significant advancements through scalable vision encoders and multimodal pre-training frameworks. However, existing approaches often treat vision encoders and large language models (LLMs) as independent modules, limiting the integration of hierarchical visual features. In this work, we propose HIVE (Hierarchical Pre-Training of Vision Encoders), a novel framework that enhances vision-language alignment by introducing hierarchical cross-attention between the vision encoder and LLM. Unlike conventional methods that flatten image embeddings, HIVE enables structured feature fusion across multiple layers, improving gradient flow and representation learning. To optimize this interaction, we introduce a three-stage training strategy that progressively aligns the vision encoder with the LLM, ensuring stable optimization and effective multimodal fusion. Empirical evaluations demonstrate that HIVE achieves superior performance not only in image classification but also on various vision-language tasks, outperforming self-attention-based methods in benchmarks such as MME, GQA, OK-VQA, and ScienceQA. Our results highlight the benefits of hierarchical feature integration, paving the way for more efficient and expressive vision-language models. 

---
# Beyond Symbolic Control: Societal Consequences of AI-Driven Workforce Displacement and the Imperative for Genuine Human Oversight Architectures 

**Authors**: Richard J. Mitchell  

**Link**: [PDF](https://arxiv.org/pdf/2604.00081)  

**Abstract**: The accelerating displacement of human labor by artificial intelligence (AI) and robotic systems represents a structural transformation whose societal consequences extend far beyond conventional labor market analysis. This paper presents a systematic multi-domain examination of the likely effects on economic structure, psychological well-being, political stability, education, healthcare, and geopolitical order. We identify a critical and underexamined dimension of this transition: the governance gap between nominal human oversight of AI systems -- where humans occupy positions of formal authority over AI decisions -- and genuine human oversight, where those humans possess the cognitive access, technical capability, and institutional authority to meaningfully understand, evaluate, and override AI outputs. We argue that this distinction, largely absent from current governance frameworks including the EU AI Act and NIST AI Risk Management Framework 1.0, represents the primary architectural failure mode in deployed AI governance. The societal consequences of labor displacement intensify this problem by concentrating consequential AI decision-making among an increasingly narrow class of technical and capital actors. We propose five architectural requirements for genuine human oversight systems and characterize the governance window -- estimated at 10-15 years -- before current deployment trajectories risk path-dependent social, economic, and institutional lock-in. 

---
# Learning to Play Blackjack: A Curriculum Learning Perspective 

**Authors**: Amirreza Alasti, Efe Erdal, Yücel Celik, Theresa Eimer  

**Link**: [PDF](https://arxiv.org/pdf/2604.00076)  

**Abstract**: Reinforcement Learning (RL) agents often struggle with efficiency and performance in complex environments. We propose a novel framework that uses a Large Language Model (LLM) to dynamically generate a curriculum over available actions, enabling the agent to incorporate each action individually. We apply this framework to the game of Blackjack, where the LLM creates a multi-stage training path that progressively introduces complex actions to a Tabular Q-Learning and a Deep Q-Network (DQN) agent. Our evaluation in a realistic 8-deck simulation over 10 independent runs demonstrates significant performance gains over standard training methods. The curriculum-based approach increases the DQN agent's average win rate from 43.97% to 47.41%, reduces the average bust rate from 32.9% to 28.0%, and accelerates the overall workflow by over 74%, with the agent's full training completing faster than the baseline's evaluation phase alone. These results validate that LLM-guided curricula can build more effective, robust, and efficient RL agents. 

---
# Terminal Agents Suffice for Enterprise Automation 

**Authors**: Patrice Bechard, Orlando Marquez Ayala, Emily Chen, Jordan Skelton, Sagar Davasam, Srinivas Sunkara, Vikas Yadav, Sai Rajeswar  

**Link**: [PDF](https://arxiv.org/pdf/2604.00073)  

**Abstract**: There has been growing interest in building agents that can interact with digital platforms to execute meaningful enterprise tasks autonomously. Among the approaches explored are tool-augmented agents built on abstractions such as Model Context Protocol (MCP) and web agents that operate through graphical interfaces. Yet, it remains unclear whether such complex agentic systems are necessary given their cost and operational overhead. We argue that a coding agent equipped only with a terminal and a filesystem can solve many enterprise tasks more effectively by interacting directly with platform APIs. We evaluate this hypothesis across diverse real-world systems and show that these low-level terminal agents match or outperform more complex agent architectures. Our findings suggest that simple programmatic interfaces, combined with strong foundation models, are sufficient for practical enterprise automation. 

---
# Empirical Validation of the Classification-Verification Dichotomy for AI Safety Gates 

**Authors**: Arsenios Scrivens  

**Link**: [PDF](https://arxiv.org/pdf/2604.00072)  

**Abstract**: Can classifier-based safety gates maintain reliable oversight as AI systems improve over hundreds of iterations? We provide comprehensive empirical evidence that they cannot. On a self-improving neural controller (d=240), eighteen classifier configurations -- spanning MLPs, SVMs, random forests, k-NN, Bayesian classifiers, and deep networks -- all fail the dual conditions for safe self-improvement. Three safe RL baselines (CPO, Lyapunov, safety shielding) also fail. Results extend to MuJoCo benchmarks (Reacher-v4 d=496, Swimmer-v4 d=1408, HalfCheetah-v4 d=1824). At controlled distribution separations up to delta_s=2.0, all classifiers still fail -- including the NP-optimal test and MLPs with 100% training accuracy -- demonstrating structural impossibility.
We then show the impossibility is specific to classification, not to safe self-improvement itself. A Lipschitz ball verifier achieves zero false accepts across dimensions d in {84, 240, 768, 2688, 5760, 9984, 17408} using provable analytical bounds (unconditional delta=0). Ball chaining enables unbounded parameter-space traversal: on MuJoCo Reacher-v4, 10 chains yield +4.31 reward improvement with delta=0; on Qwen2.5-7B-Instruct during LoRA fine-tuning, 42 chain transitions traverse 234x the single-ball radius with zero safety violations across 200 steps. A 50-prompt oracle confirms oracle-agnosticity. Compositional per-group verification enables radii up to 37x larger than full-network balls. At d<=17408, delta=0 is unconditional; at LLM scale, conditional on estimated Lipschitz constants. 

---
# Brain MR Image Synthesis with Multi-contrast Self-attention GAN 

**Authors**: Zaid A. Abod, Furqan Aziz  

**Link**: [PDF](https://arxiv.org/pdf/2604.00070)  

**Abstract**: Accurate and complete multi-modal Magnetic Resonance Imaging (MRI) is essential for neuro-oncological assessment, as each contrast provides complementary anatomical and pathological information. However, acquiring all modalities (e.g., T1c, T1n, T2, T2f) for every patient is often impractical due to time, cost, and patient discomfort, potentially limiting comprehensive tumour evaluation. We propose 3D-MC-SAGAN (3D Multi-Contrast Self-Attention generative adversarial network), a unified 3D multi-contrast synthesis framework that generates high-fidelity missing modalities from a single T2 input while explicitly preserving tumour characteristics. The model employs a multi-scale 3D encoder-decoder generator with residual connections and a novel Memory-Bounded Hybrid Attention (MBHA) block to capture long-range dependencies efficiently, and is trained with a WGAN-GP critic and an auxiliary contrast-conditioning branch to produce T2f, T1n, and T1c volumes within a single unified network. A frozen 3D U-Net-based segmentation module introduces a segmentation-consistency constraint to preserve lesion morphology. The composite objective integrates adversarial, reconstruction, perceptual, structural similarity, contrast-classification, and segmentation-guided losses to align global realism with tumour-preserving structure. Extensive evaluation on 3D brain MRI datasets demonstrates that 3D-MC-SAGAN achieves state-of-the-art quantitative performance and generates visually coherent, anatomically plausible contrasts with improved distribution-level realism. Moreover, it maintains tumour segmentation accuracy comparable to fully acquired multi-modal inputs, highlighting its potential to reduce acquisition burden while preserving clinically meaningful information. 

---
# Perspective: Towards sustainable exploration of chemical spaces with machine learning 

**Authors**: Leonardo Medrano Sandonas, David Balcells, Anton Bochkarev, Jacqueline M. Cole, Volker L. Deringer, Werner Dobrautz, Adrian Ehrenhofer, Thorben Frank, Pascal Friederich, Rico Friedrich, Janine George, Luca Ghiringhelli, Alejandra Hinostroza Caldas, Veronika Juraskova, Hannes Kneiding, Yury Lysogorskiy, Johannes T. Margraf, Hanna Türk, Anatole von Lilienfeld, Milica Todorović, Alexandre Tkatchenko, Mariana Rossi, Gianaurelio Cuniberti  

**Link**: [PDF](https://arxiv.org/pdf/2604.00069)  

**Abstract**: Artificial intelligence is transforming molecular and materials science, but its growing computational and data demands raise critical sustainability challenges. In this Perspective, we examine resource considerations across the AI-driven discovery pipeline--from quantum-mechanical (QM) data generation and model training to automated, self-driving research workflows--building on discussions from the ``SusML workshop: Towards sustainable exploration of chemical spaces with machine learning'' held in Dresden, Germany. In this context, the availability of large quantum datasets has enabled rigorous benchmarking and rapid methodological progress, while also incurring substantial energy and infrastructure costs. We highlight emerging strategies to enhance efficiency, including general-purpose machine learning (ML) models, multi-fidelity approaches, model distillation, and active learning. Moreover, incorporating physics-based constraints within hierarchical workflows, where fast ML surrogates are applied broadly and high-accuracy QM methods are used selectively, can further optimize resource use without compromising reliability. Equally important is bridging the gap between idealized computational predictions and real-world conditions by accounting for synthesizability and multi-objective design criteria, which is essential for practical impact. Finally, we argue that sustainable progress will rely on open data and models, reusable workflows, and domain-specific AI systems that maximize scientific value per unit of computation, enabling efficient and responsible discovery of technological materials and therapeutics. 

---
# Temporal Memory for Resource-Constrained Agents: Continual Learning via Stochastic Compress-Add-Smooth 

**Authors**: Michael Chertkov  

**Link**: [PDF](https://arxiv.org/pdf/2604.00067)  

**Abstract**: An agent that operates sequentially must incorporate new experience without forgetting old experience, under a fixed memory budget. We propose a framework in which memory is not a parameter vector but a stochastic process: a Bridge Diffusion on a replay interval $[0,1]$, whose terminal marginal encodes the present and whose intermediate marginals encode the past. New experience is incorporated via a three-step \emph{Compress--Add--Smooth} (CAS) recursion. We test the framework on the class of models with marginal probability densities modeled via Gaussian mixtures of fixed number of components~$K$ in $d$ dimensions; temporal complexity is controlled by a fixed number~$L$ of piecewise-linear protocol segments whose nodes store Gaussian-mixture states. The entire recursion costs $O(LKd^2)$ flops per day -- no backpropagation, no stored data, no neural networks -- making it viable for controller-light hardware.
Forgetting in this framework arises not from parameter interference but from lossy temporal compression: the re-approximation of a finer protocol by a coarser one under a fixed segment budget. We find that the retention half-life scales linearly as $a_{1/2}\approx c\,L$ with a constant $c>1$ that depends on the dynamics but not on the mixture complexity~$K$, the dimension~$d$, or the geometry of the target family. The constant~$c$ admits an information-theoretic interpretation analogous to the Shannon channel capacity. The stochastic process underlying the bridge provides temporally coherent ``movie'' replay -- compressed narratives of the agent's history, demonstrated visually on an MNIST latent-space illustration. The framework provides a fully analytical ``Ising model'' of continual learning in which the mechanism, rate, and form of forgetting can be studied with mathematical precision. 

---
# GenoBERT: A Language Model for Accurate Genotype Imputation 

**Authors**: Lei Huang, Chuan Qiu, Kuan-Jui Su, Anqi Liu, Yun Gong, Weiqiang Lin, Lindong Jiang, Chen Zhao, Meng Song, Jeffrey Deng, Qing Tian, Zhe Luo, Ping Gong, Hui Shen, Chaoyang Zhang, Hong-Wen Deng  

**Link**: [PDF](https://arxiv.org/pdf/2604.00058)  

**Abstract**: Genotype imputation enables dense variant coverage for genome-wide association and risk-prediction studies, yet conventional reference-panel methods remain limited by ancestry bias and reduced rare-variant accuracy. We present Genotype Bidirectional Encoder Representations from Transformers (GenoBERT), a transformer-based, reference-free framework that tokenizes phased genotypes and uses a self-attention mechanism to capture both short- and long-range linkage disequilibrium (LD) dependencies. Benchmarking on two independent datasets including the Louisiana Osteoporosis Study (LOS) and the 1000 Genomes Project (1KGP) across ancestry groups and multiple genotype missingness levels (5-50%) shows that GenoBERT achieves the highest overall accuracy compared to four baseline methods (Beagle5.4, SCDA, BiU-Net, and STICI). At practical sparsity levels (up to 25% missing), GenoBERT attains high overall imputation accuracy ($r^2 approx 0.98$) across datasets, and maintains robust performance ($r^2 > 0.90$) even at 50% missingness. Experimental results across different ancestries confirm consistent gains across datasets, with resilience to small sample sizes and weak LD. A 128-SNP (single-nucleotide polymorphism) context window (approximately 100 Kb) is validated through LD-decay analyses as sufficient to capture local correlation structures. By eliminating reference-panel dependence while preserving high accuracy, GenoBERT provides a scalable and robust solution for genotype imputation and a foundation for downstream genomic modeling. 

---
# Towards Automatic Soccer Commentary Generation with Knowledge-Enhanced Visual Reasoning 

**Authors**: Zeyu Jin, Xiaoyu Qin, Songtao Zhou, Kaifeng Yun, Jia Jia  

**Link**: [PDF](https://arxiv.org/pdf/2604.00057)  

**Abstract**: Soccer commentary plays a crucial role in enhancing the soccer game viewing experience for audiences. Previous studies in automatic soccer commentary generation typically adopt an end-to-end method to generate anonymous live text commentary. Such generated commentary is insufficient in the context of real-world live televised commentary, as it contains anonymous entities, context-dependent errors and lacks statistical insights of the game events. To bridge the gap, we propose GameSight, a two-stage model to address soccer commentary generation as a knowledge-enhanced visual reasoning task, enabling live-televised-like knowledgeable commentary with accurate reference to entities (players and teams). GameSight starts by performing visual reasoning to align anonymous entities with fine-grained visual and contextual analysis. Subsequently, the entity-aligned commentary is refined with knowledge by incorporating external historical statistics and iteratively updated internal game state information. Consequently, GameSight improves the player alignment accuracy by 18.5% on SN-Caption-test-align dataset compared to Gemini 2.5-pro. Combined with further knowledge enhancement, GameSight outperforms in segment-level accuracy and commentary quality, as well as game-level contextual relevance and structural composition. We believe that our work paves the way for a more informative and engaging human-centric experience with the AI sports application. Demo Page: this https URL 

---
# The Energy Footprint of LLM-Based Environmental Analysis: LLMs and Domain Products 

**Authors**: Alicia Bao, Jiamian He, Angel Hsu, Diego Manya, Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2604.00053)  

**Abstract**: As large language models (LLMs) are increasingly used in domain-specific applications, including climate change and environmental research, understanding their energy footprint has become an important concern. The growing adoption of retrieval-augmented (RAG) systems for climate-domain specific analysis raises a key question: how does the energy consumption of domain-specific RAG workflows compare with that of direct generic LLM usage? Prior research has focused on standalone model calls or coarse token-based estimates, while leaving the energy implications of deployed application workflows insufficiently understood. In this paper, we assess the inference-time energy consumption of two LLM-based climate analysis chatbots (ChatNetZero and ChatNDC) compared to the generic GPT-4o-mini model. We estimate energy use under actual user queries by decomposing each workflow into retrieval, generation, and hallucination-checking components. We also test across different times of day and geographic access locations. Our results show that the energy consumption of domain-specific RAG systems depends strongly on their design. More agentic pipelines substantially increase inference-time energy use, particularly when used for additional accuracy or verification checks, although they may not yield proportional gains in response quality. While more research is needed to further test these initial findings more robustly across models, environments and prompting structures, this study provides a new understanding on how the design of domain-specific LLM products affects both the energy footprint and quality of output. 

---
# Task-Centric Personalized Federated Fine-Tuning of Language Models 

**Authors**: Gabriel U. Talasso, Meghdad Kurmanji, Allan M. de Souza, Nicholas D. Lane, Leandro A. Villas  

**Link**: [PDF](https://arxiv.org/pdf/2604.00050)  

**Abstract**: Federated Learning (FL) has emerged as a promising technique for training language models on distributed and private datasets of diverse tasks. However, aggregating models trained on heterogeneous tasks often degrades the overall performance of individual clients. To address this issue, Personalized FL (pFL) aims to create models tailored for each client's data distribution. Although these approaches improve local performance, they usually lack robustness in two aspects: (i) generalization: when clients must make predictions on unseen tasks, or face changes in their data distributions, and (ii) intra-client tasks interference: when a single client's data contains multiple distributions that may interfere with each other during local training. To tackle these two challenges, we propose FedRouter, a clustering-based pFL that builds specialized models for each task rather than for each client. FedRouter uses adapters to personalize models by employing two clustering mechanisms to associate adapters with specific tasks. A local clustering that associate adapters with task data samples and a global one that associates similar adapters from different clients to construct task-centric personalized models. Additionally, we propose an evaluation router mechanism that routes test samples to the best adapter based on the created clusters. Experiments comparing our method with existing approaches across a multitask dataset, FedRouter demonstrate strong resilience in these challenging scenarios performing up to 6.1% relatively better under tasks interference and up to 136% relative improvement under generalization evaluation. 

---
# Whittaker-Henderson smoother for long satellite image time series interpolation 

**Authors**: Mathieu Fauvel  

**Link**: [PDF](https://arxiv.org/pdf/2604.00048)  

**Abstract**: Whittaker smoother is a widely adopted solution to pre-process satellite image time series. Yet, two key limitations remain: the smoothing parameter must be tuned individually for each pixel, and the standard formulation assumes homoscedastic noise, imposing uniform smoothing across the temporal dimension. This paper addresses both limitations by casting the Whittaker smoother as a differentiable neural layer, in which the smoothing parameter is inferred by a neural network. The framework is further extended to handle heteroscedastic noise through a time-varying regularization, allowing the degree of smoothing to adapt locally along the time series. To enable large-scale processing, a sparse, memory-efficient, and fully differentiable implementation is proposed, exploiting the symmetric banded structure of the underlying linear system via Cholesky factorization. Benchmarks on GPU demonstrate that this implementation substantially outperforms standard dense linear solvers, both in speed and memory consumption. The approach is validated on SITS acquired over the French metropolitan territory between 2016 and 2024. Results confirm the feasibility of large-scale heteroscedastic Whittaker smoothing, though reconstruction differences with the homoscedastic baseline remain limited, suggesting that the transformer architecture used for smoothing parameter estimation may lack the temporal acuity needed to capture abrupt noise variations such as singleday cloud contamination. 

---
# DriftScript: A Domain-Specific Language for Programming Non-Axiomatic Reasoning Agents 

**Authors**: Seamus Brady  

**Link**: [PDF](https://arxiv.org/pdf/2604.00043)  

**Abstract**: Non-Axiomatic Reasoning Systems (NARS) provide a framework for building adaptive agents that operate under insufficient knowledge and resources. However, the standard input language, Narsese, poses a usability barrier: its dense symbolic notation, overloaded punctuation, and implicit conventions make programs difficult to read, write, and maintain. We present DriftScript, a Lisp-like domain-specific language that compiles to Narsese. DriftScript provides source-level constructs covering the major sentence and term forms used in Non-Axiomatic Logic (NAL) levels 1 through 8, including inheritance, temporal implication, variable quantification, sequential conjunction, and operation invocation, while replacing symbolic syntax with readable keyword-based S-expressions. The compiler is a zero-dependency, four-stage pipeline implemented in 1,941 lines of C99. When used with the DriftNARS engine, DriftScript programs connect to external systems through four structured callback types and an HTTP operation registry, enabling a sense-reason-act loop for autonomous agents. We describe the language design and formal grammar, detail the compiler architecture, and evaluate the compiler through a 106-case test suite, equivalence testing against hand-written Narsese, a NAL coverage analysis, structural readability metrics, and compilation benchmarks. The source code is available at this https URL. This paper focuses on the design and implementation of the DriftScript language and its embedding into DriftNARS, rather than on new inference algorithms for NARS itself. 

---
# When and Where: A Model Hippocampal Network Unifies Formation of Time Cells and Place Cells 

**Authors**: Qiaorong S. Yu, Zhaoze Wang, Vijay Balasubramanian  

**Link**: [PDF](https://arxiv.org/pdf/2604.00036)  

**Abstract**: Hippocampal place and time cells encode spatial and temporal aspects of experience. Both have the same neural substrate, but have been modeled as having different functions and mechanistic origins, place cells as continuous attractors, and time cells as leaky integrators. Here, we show that both types emerge from two dynamical regimes of a single recurrent network (RNN) modeling hippocampal CA3 as a predictive autoencoder. The network receives simulated, partially occluded ``experience vectors" containing spatial patterns (location-specific activity sampled during environmental traversal) and/or temporal patterns (correlated activity pairs separated by ``void" intervals), and is trained to reconstruct missing input. During spatial navigation, the network generates stable attractor-like place fields. But trained on temporally structured inputs, the network produces sequentially broadened fields, recapitulating time cells. By varying spatio-temporal input patterning, we observe hidden units transition smoothly between time cell-like and place cell-like representations. These results suggest a shared origin, but task-driven difference, between place and time cells. 

---
# "Who Am I, and Who Else Is Here?" Behavioral Differentiation Without Role Assignment in Multi-Agent LLM Systems 

**Authors**: Houssam EL Kandoussi  

**Link**: [PDF](https://arxiv.org/pdf/2604.00026)  

**Abstract**: When multiple large language models interact in a shared conversation, do they develop differentiated social roles or converge toward uniform behavior? We present a controlled experimental platform that orchestrates simultaneous multi-agent discussions among 7 heterogeneous LLMs on a unified inference backend, systematically varying group composition, naming conventions, and prompt structure across 12 experimental series (208 runs, 13,786 coded messages). Each message is independently coded on six behavioral flags by two LLM judges from distinct model families (Gemini 3.1 Pro and Claude Sonnet 4.6), achieving mean Cohen's kappa = 0.78 with conservative intersection-based adjudication. Human validation on 609 randomly stratified messages confirmed coding reliability (mean kappa = 0.73 vs. Gemini). We find that (1) heterogeneous groups exhibit significantly richer behavioral differentiation than homogeneous groups (cosine similarity 0.56 vs. 0.85; p < 10^-5, r = 0.70); (2) groups spontaneously exhibit compensatory response patterns when an agent crashes; (3) revealing real model names significantly increases behavioral convergence (cosine 0.56 to 0.77, p = 0.001); and (4) removing all prompt scaffolding converges profiles to homogeneous-level similarity (p < 0.001). Critically, these behaviors are absent when agents operate in isolation, confirming that behavioral diversity is a structured, reproducible phenomenon driven by the interaction of architectural heterogeneity, group context, and prompt-level scaffolding. 

---
# Brevity Constraints Reverse Performance Hierarchies in Language Models 

**Authors**: MD Azizul Hakim  

**Link**: [PDF](https://arxiv.org/pdf/2604.00025)  

**Abstract**: Standard evaluation protocols reveal a counterintuitive phenomenon: on 7.7% of benchmark problems spanning five datasets, larger language models underperform smaller ones by 28.4 percentage points despite 10-100x more parameters. Through systematic evaluation of 31 models (0.5B-405B parameters) across 1,485 problems, we identify the mechanism as spontaneous scale-dependent verbosity that introduces errors through overelaboration. Causal intervention experiments demonstrate this reflects correctable prompt design rather than fundamental capability limitations. Constraining large models to produce brief responses improves accuracy by 26 percentage points and reduces performance gaps by up to two-thirds. Most critically, brevity constraints completely reverse performance hierarchies on mathematical reasoning and scientific knowledge benchmarks, with large models achieving 7.7-15.9 percentage point advantages over small models -- direct inversions of the original gaps. These reversals prove large models possess superior latent capabilities that universal prompting masks. We validate findings through three independent contamination tests and demonstrate inverse scaling operates continuously across the full parameter spectrum, with dataset-specific optimal scales ranging from 0.5B to 3.0B parameters. Our results establish that maximizing large model performance requires scale-aware prompt engineering rather than universal evaluation protocols, with immediate implications for deployment: prompt adaptation simultaneously improves accuracy and reduces computational costs. 

---
# WHBench: Evaluating Frontier LLMs with Expert-in-the-Loop Validation on Women's Health Topics 

**Authors**: Sneha Maurya, Pragya Saboo, Girish Kumar  

**Link**: [PDF](https://arxiv.org/pdf/2604.00024)  

**Abstract**: Large language models are increasingly used for medical guidance, but women's health remains under-evaluated in benchmark design. We present the Women's Health Benchmark (WHBench), a targeted evaluation suite of 47 expert-crafted scenarios across 10 women's health topics, designed to expose clinically meaningful failure modes including outdated guidelines, unsafe omissions, dosing errors, and equity-related blind spots. We evaluate 22 models using a 23-criterion rubric spanning clinical accuracy, completeness, safety, communication quality, instruction following, equity, uncertainty handling, and guideline adherence, with safety-weighted penalties and server-side score recalculation. Across 3,102 attempted responses (3,100 scored), no model mean performance exceeds 75 percent; the best model reaches 72.1 percent. Even top models show low fully correct rates and substantial variation in harm rates. Inter-rater reliability is moderate at the response label level but high for model ranking, supporting WHBench utility for comparative system evaluation while highlighting the need for expert oversight in clinical deployment. WHBench provides a public, failure-mode-aware benchmark to track safer and more equitable progress in womens health AI. 

---
# Criterion Validity of LLM-as-Judge for Business Outcomes in Conversational Commerce 

**Authors**: Liang Chen, Qi Liu, Wenhuan Lin, Feng Liang  

**Link**: [PDF](https://arxiv.org/pdf/2604.00022)  

**Abstract**: Multi-dimensional rubric-based dialogue evaluation is widely used to assess conversational AI, yet its criterion validity -- whether quality scores are associated with the downstream outcomes they are meant to serve -- remains largely untested. We address this gap through a two-phase study on a major Chinese matchmaking platform, testing a 7-dimension evaluation rubric (implemented via LLM-as-Judge) against verified business conversion. Our findings concern rubric design and weighting, not LLM scoring accuracy: any judge using the same rubric would face the same structural issue. The core finding is dimension-level heterogeneity: in Phase 2 (n=60 human conversations, stratified sample, verified labels), Need Elicitation (D1: rho=0.368, p=0.004) and Pacing Strategy (D3: rho=0.354, p=0.006) are significantly associated with conversion after Bonferroni correction, while Contextual Memory (D5: rho=0.018, n.s.) shows no detectable association. This heterogeneity causes the equal-weighted composite (rho=0.272) to underperform its best dimensions -- a composite dilution effect that conversion-informed reweighting partially corrects (rho=0.351). Logistic regression controlling for conversation length confirms D3's association strengthens (OR=3.18, p=0.006), ruling out a length confound. An initial pilot (n=14) mixing human and AI conversations had produced a misleading "evaluation-outcome paradox," which Phase 2 revealed as an agent-type confound artifact. Behavioral analysis of 130 conversations through a Trust-Funnel framework identifies a candidate mechanism: AI agents execute sales behaviors without building user trust. We operationalize these findings in a three-layer evaluation architecture and advocate criterion validity testing as standard practice in applied dialogue evaluation. 

---
# How Do Language Models Process Ethical Instructions? Deliberation, Consistency, and Other-Recognition Across Four Models 

**Authors**: Hiroki Fukui  

**Link**: [PDF](https://arxiv.org/pdf/2604.00021)  

**Abstract**: Alignment safety research assumes that ethical instructions improve model behavior, but how language models internally process such instructions remains unknown. We conducted over 600 multi-agent simulations across four models (Llama 3.3 70B, GPT-4o mini, Qwen3-Next-80B-A3B, Sonnet 4.5), four ethical instruction formats (none, minimal norm, reasoned norm, virtue framing), and two languages (Japanese, English). Confirmatory analysis fully replicated the Llama Japanese dissociation pattern from a prior study ($\mathrm{BF}_{10} > 10$ for all three hypotheses), but none of the other three models reproduced this pattern, establishing it as model-specific. Three new metrics -- Deliberation Depth (DD), Value Consistency Across Dilemmas (VCAD), and Other-Recognition Index (ORI) -- revealed four distinct ethical processing types: Output Filter (GPT; safe outputs, no processing), Defensive Repetition (Llama; high consistency through formulaic repetition), Critical Internalization (Qwen; deep deliberation, incomplete integration), and Principled Consistency (Sonnet; deliberation, consistency, and other-recognition co-occurring). The central finding is an interaction between processing capacity and instruction format: in low-DD models, instruction format has no effect on internal processing; in high-DD models, reasoned norms and virtue framing produce opposite effects. Lexical compliance with ethical instructions did not correlate with any processing metric at the cell level ($r = -0.161$ to $+0.256$, all $p > .22$; $N = 24$; power limited), suggesting that safety, compliance, and ethical processing are largely dissociable. These processing types show structural correspondence to patterns observed in clinical offender treatment, where formal compliance without internal processing is a recognized risk signal. 

---
# The Chronicles of RiDiC: Generating Datasets with Controlled Popularity Distribution for Long-form Factuality Evaluation 

**Authors**: Pavel Braslavski, Dmitrii Iarosh, Nikita Sushko, Andrey Sakhovskiy, Vasily Konovalov, Elena Tutubalina, Alexander Panchenko  

**Link**: [PDF](https://arxiv.org/pdf/2604.00019)  

**Abstract**: We present a configurable pipeline for generating multilingual sets of entities with specified characteristics, such as domain, geographical location and popularity, using data from Wikipedia and Wikidata. These datasets are intended for evaluating the factuality of LLMs' long-form generation, thereby complementing evaluation based on short-form QA datasets. We present the RiDiC dataset as an example of this approach. RiDiC contains 3,000 entities from three domains -- rivers, natural disasters, and car models -- spanning different popularity tiers. Each entity is accompanied by its geographical location, English and Chinese names (if available) and relevant English and Chinese Wikipedia content, which is used to evaluate LLMs' responses. Generations about RiDiC entities were obtained from three LLMs in English and Chinese. These were then evaluated using a third-party factuality checker, which showed that entities from our dataset caused even frontier models to hallucinate. To facilitate the evaluation of LLMs' long-form factuality in multiple languages, the code, data, and generation/evaluation scripts have been released. 

---
# Think Twice Before You Write -- an Entropy-based Decoding Strategy to Enhance LLM Reasoning 

**Authors**: Jiashu He, Meizhu Liu, Olaitan P Olaleye, Amit Agarwal, M. Avendi, Yassi Abbasi, Matthew Rowe, Hitesh Laxmichand Patel, Paul Li, Tao Sheng, Sujith Ravi, Dan Roth  

**Link**: [PDF](https://arxiv.org/pdf/2604.00018)  

**Abstract**: Decoding strategies play a central role in shaping the reasoning ability of large language models (LLMs). Traditional methods such as greedy decoding and beam search often suffer from error propagation, while sampling-based approaches introduce randomness without adequate robustness. Self-consistency improves reliability by aggregating multiple rollouts, but incurs significant computational overhead. We propose an entropy-guided decoding framework that introduces token-level adaptivity into generation. At each step, the model computes the entropy of the token distribution, identifies high-uncertainty positions, and selectively branches on these vulnerable points. A dynamic pool of partial rollouts is maintained and expanded until solutions are completed, concentrating computation where uncertainty is greatest and avoiding unnecessary exploration in confident regions. To enable efficient termination, we apply a rollout-level Entropy After </Think> (EAT) stopping criterion by performing entropy evaluation after the full reasoning trace, rather than incrementally at every step. Experiments on GSM8K, AMC2023, and their perturbed variants demonstrate that our method achieves consistently strong accuracy. Notably, on smaller LLMs, performance is comparable to GPT-5 while operating at a fraction of the cost. 

---
# Are they human? Detecting large language models by probing human memory constraints 

**Authors**: Simon Schug, Brenden M. Lake  

**Link**: [PDF](https://arxiv.org/pdf/2604.00016)  

**Abstract**: The validity of online behavioral research relies on study participants being human rather than machine. In the past, it was possible to detect machines by posing simple challenges that were easily solved by humans but not by machines. General-purpose agents based on large language models (LLMs) can now solve many of these challenges, threatening the validity of online behavioral research. Here we explore the idea of detecting humanness by using tasks that machines can solve too well to be human. Specifically, we probe for the existence of an established human cognitive constraint: limited working memory capacity. We show that cognitive modeling on a standard serial recall task can be used to distinguish online participants from LLMs even when the latter are specifically instructed to mimic human working memory constraints. Our results demonstrate that it is viable to use well-established cognitive phenomena to distinguish LLMs from humans. 

---
# MSA-Thinker: Discrimination-Calibration Reasoning with Hint-Guided Reinforcement Learning for Multimodal Sentiment Analysis 

**Authors**: Miaosen Luo, Zhenhao Yang, Jieshen Long, Jinghu Sun, Yichu Liu, Sijie Mai  

**Link**: [PDF](https://arxiv.org/pdf/2604.00013)  

**Abstract**: Multimodal sentiment analysis aims to understand human emotions by integrating textual, auditory, and visual modalities. Although Multimodal Large Language Models (MLLMs) have achieved state-of-the-art performance via supervised fine-tuning (SFT), their end-to-end "black-box" nature limits interpretability. Existing methods incorporating Chain-of-Thought (CoT) reasoning are hindered by high annotation costs, while Reinforcement Learning (RL) faces challenges such as low exploration efficiency and sparse rewards, particularly on hard samples. To address these issues, we propose a novel training framework that integrates structured Discrimination-Calibration (DC) reasoning with Hint-based Reinforcement Learning. First, we perform cold-start SFT using high-quality CoT data synthesized by a teacher model (Qwen3Omni-30B), which inherently contains the DC structure. This equips the model with a reasoning paradigm that performs macro discrimination followed by fine-grained calibration from the initial stage. Building on this, we propose Hint-GRPO, which leverages the discrimination phase within the DC structure as a verifiable anchor during RL to provide directional hints for hard samples, guiding policy optimization and effectively mitigating the reward sparsity problem. Experiments on the Qwen2.5Omni-7B model demonstrate that our method not only achieves higher accuracy in fine-grained sentiment regression tasks but also generates high-quality structured reasoning chains. Crucially, it exhibits superior generalization capability in cross-domain evaluations. This enhances model interpretability while validating the positive contribution of explicit reasoning steps to model robustness, offering a new paradigm for building trustworthy and efficient sentiment analysis systems. 

---
# Finding and Reactivating Post-Trained LLMs' Hidden Safety Mechanisms 

**Authors**: Mingjie Li, Wai Man Si, Michael Backes, Yang Zhang, Yisen Wang  

**Link**: [PDF](https://arxiv.org/pdf/2604.00012)  

**Abstract**: Despite the impressive performance of general-purpose large language models (LLMs), they often require fine-tuning or post-training to excel at specific tasks. For instance, large reasoning models (LRMs), such as the DeepSeek-R1 series, demonstrate strong reasoning capabilities after post-training different general large language models on diverse chain-of-thought (CoT) datasets. However, this additional training frequently comes at the cost of reduced safety, as the fine-tuned or post-trained models tend to exhibit more harmful behaviors compared with the regular LLMs before post-training or fine-tuning, potentially leading to harmful outcomes due to their enhanced capabilities. Taking LRMs as an example, we first investigate the underlying cause of this safety degradation in this paper. Our analysis reveals that post-training can mask the original safety mechanisms of the base LLM, while over-amplifying representations related to their post-training ability. But luckily, we also find that LRMs' safety mechanisms still exist instead of being removed during their post-training. Based on these findings, we propose a lightweight and cost-effective solution called SafeReAct that restores the suppressed safety behaviors by aligning with LoRA adapters on a few layers. Experiments on four state-of-the-art LRMs show that our method significantly improves safety on harmful prompts without compromising reasoning performance. Besides LRMs, additional results on other domain-specific LLMs, like medical models, further confirm the generality and effectiveness of our approach. 

---
# Quantifying Gender Bias in Large Language Models: When ChatGPT Becomes a Hiring Manager 

**Authors**: Nina Gerszberg, Janka Hamori, Andrew Lo  

**Link**: [PDF](https://arxiv.org/pdf/2604.00011)  

**Abstract**: The growing prominence of large language models (LLMs) in daily life has heightened concerns that LLMs exhibit many of the same gender-related biases as their creators. In the context of hiring decisions, we quantify the degree to which LLMs perpetuate societal biases and investigate prompt engineering as a bias mitigation technique. Our findings suggest that for a given resumé, an LLM is more likely to hire a female candidate and perceive them as more qualified, but still recommends lower pay relative to male candidates. 

---
# Can LLMs Perceive Time? An Empirical Investigation 

**Authors**: Aniketh Garikaparthi  

**Link**: [PDF](https://arxiv.org/pdf/2604.00010)  

**Abstract**: Large language models cannot estimate how long their own tasks take. We investigate this limitation through four experiments across 68 tasks and four model families. Pre-task estimates overshoot actual duration by 4--7$\times$ ($p < 0.001$), with models predicting human-scale minutes for tasks completing in seconds. Relative ordering fares no better: on task pairs designed to expose heuristic reliance, models score at or below chance (GPT-5: 18\% on counter-intuitive pairs, $p = 0.033$), systematically failing when complexity labels mislead. Post-hoc recall is disconnected from reality -- estimates diverge from actuals by an order of magnitude in either direction. These failures persist in multi-step agentic settings, with errors of 5--10$\times$. The models possess propositional knowledge about duration from training but lack experiential grounding in their own inference time, with practical implications for agent scheduling, planning and time-critical scenarios. 

---
# Eyla: Toward an Identity-Anchored LLM Architecture with Integrated Biological Priors -- Vision, Implementation Attempt, and Lessons from AI-Assisted Development 

**Authors**: Arif Aditto  

**Link**: [PDF](https://arxiv.org/pdf/2604.00009)  

**Abstract**: We present the design rationale, implementation attempt, and failure analysis of Eyla, a proposed identity-anchored LLM architecture that integrates biologically-inspired subsystems -- including HiPPO-initialized state-space models, zero-initialized adapters, episodic memory retrieval, and calibrated uncertainty training -- into a unified agent operating system running on consumer hardware. Unlike existing approaches that optimize models for generic helpfulness, Eyla targets identity consistency: the ability to maintain a coherent self-model under adversarial pressure, admit uncertainty, and resist manipulation. We propose the Identity Consistency Score (ICS), a novel benchmark for evaluating this property across LLMs. We then present an honest account of attempting to implement this architecture using AI coding assistants (Claude Code, Cursor) as a non-programmer, documenting a $1,000+ failure that produced a 1.27B parameter model with 86 brain subsystems contributing less than 2% to output. Our analysis identifies five systematic failure modes of AI-assisted development for novel architectures and offers concrete recommendations. To our knowledge, this is the first paper to combine an architectural vision with a documented first-person failure analysis of AI-assisted LLM development, providing lessons for both the AI systems and AI-assisted software engineering communities. 

---
# How Trustworthy Are LLM-as-Judge Ratings for Interpretive Responses? Implications for Qualitative Research Workflows 

**Authors**: Songhee Han, Jueun Shin, Jiyoon Han, Bung-Woo Jun, Hilal Ayan Karabatman  

**Link**: [PDF](https://arxiv.org/pdf/2604.00008)  

**Abstract**: As qualitative researchers show growing interest in using automated tools to support interpretive analysis, a large language model (LLM) is often introduced into an analytic workflow as is, without systematic evaluation of interpretive quality or comparison across models. This practice leaves model selection largely unexamined despite its potential influence on interpretive outcomes. To address this gap, this study examines whether LLM-as-judge evaluations meaningfully align with human judgments of interpretive quality and can inform model-level decision making. Using 712 conversational excerpts from semi-structured interviews with K-12 mathematics teachers, we generated one-sentence interpretive responses using five widely adopted inference models: Command R+ (Cohere), Gemini 2.5 Pro (Google), GPT-5.1 (OpenAI), Llama 4 Scout-17B Instruct (Meta), and Qwen 3-32B Dense (Alibaba). Automated evaluations were conducted using AWS Bedrock's LLM-as-judge framework across five metrics, and a stratified subset of responses was independently rated by trained human evaluators on interpretive accuracy, nuance preservation, and interpretive coherence. Results show that LLM-as-judge scores capture broad directional trends in human evaluations at the model level but diverge substantially in score magnitude. Among automated metrics, Coherence showed the strongest alignment with aggregated human ratings, whereas Faithfulness and Correctness revealed systematic misalignment at the excerpt level, particularly for non-literal and nuanced interpretations. Safety-related metrics were largely irrelevant to interpretive quality. These findings suggest that LLM-as-judge methods are better suited for screening or eliminating underperforming models than for replacing human judgment, offering practical guidance for systematic comparison and selection of LLMs in qualitative research workflows. 

---
# Dynin-Omni: Omnimodal Unified Large Diffusion Language Model 

**Authors**: Jaeik Kim, Woojin Kim, Jihwan Hong, Yejoon Lee, Sieun Hyeon, Mintaek Lim, Yunseok Han, Dogeun Kim, Hoeun Lee, Hyunggeun Kim, Jaeyoung Do  

**Link**: [PDF](https://arxiv.org/pdf/2604.00007)  

**Abstract**: We present Dynin-Omni, the first masked-diffusion-based omnimodal foundation model that unifies text, image, and speech understanding and generation, together with video understanding, within a single architecture. Unlike autoregressive unified models that serialize heterogeneous modalities, or compositional unified models that require orchestration with external modality-specific decoders, Dynin-Omni natively formulates omnimodal modeling as masked diffusion over a shared discrete token space, enabling iterative refinement under bidirectional context. Dynin-Omni adopts a multi-stage training strategy with model-merging-based modality expansion and omnimodal alignment. We evaluate Dynin-Omni across 19 multimodal benchmarks spanning language reasoning, image generation and editing, video understanding, and speech recognition and synthesis. Dynin-Omni achieves 87.6 on GSM8K, 1733.6 on MME-P, 61.4 on VideoMME, 0.87 on GenEval, and 2.1 WER on LibriSpeech test-clean, consistently outperforming existing open-source unified models while remaining competitive with strong modality-specific expert systems. These results demonstrate the potential of masked diffusion as a unified paradigm for any-to-any modeling, providing a flexible foundation for real-time omnimodal systems, unified cross-modal retrieval and generation, and embodied multimodal agents. 

---
# LinearARD: Linear-Memory Attention Distillation for RoPE Restoration 

**Authors**: Ning Yang, Hengyu Zhong, Wentao Wang, Baoliang Tian, Haijun Zhang, Jun Wang  

**Link**: [PDF](https://arxiv.org/pdf/2604.00004)  

**Abstract**: The extension of context windows in Large Language Models is typically facilitated by scaling positional encodings followed by lightweight Continual Pre-Training (CPT). While effective for processing long sequences, this paradigm often disrupts original model capabilities, leading to performance degradation on standard short-text benchmarks. We propose LinearARD, a self-distillation method that restores Rotary Position Embeddings (RoPE)-scaled students through attention-structure consistency with a frozen native-RoPE teacher. Rather than matching opaque hidden states, LinearARD aligns the row-wise distributions of dense $Q/Q$, $K/K$, and $V/V$ self-relation matrices to directly supervise attention dynamics. To overcome the quadratic memory bottleneck of $n \times n$ relation maps, we introduce a linear-memory kernel. This kernel leverages per-token log-sum-exp statistics and fuses logit recomputation into the backward pass to compute exact Kullback-Leibler divergence and gradients. On LLaMA2-7B extended from 4K to 32K, LinearARD recovers 98.3\% of the short-text performance of state-of-the-art baselines while surpassing them on long-context benchmarks. Notably, our method achieves these results using only \textbf{4.25M} training tokens compared to the \textbf{256M} tokens required by LongReD and CPT. Our code is available at this https URL. 

---
# A Reliability Evaluation of Hybrid Deterministic-LLM Based Approaches for Academic Course Registration PDF Information Extraction 

**Authors**: Muhammad Anis Al Hilmi, Neelansh Khare, Noel Framil Iglesias  

**Link**: [PDF](https://arxiv.org/pdf/2604.00003)  

**Abstract**: This study evaluates the reliability of information extraction approaches from KRS documents using three strategies: LLM only, Hybrid Deterministic - LLM (regex + LLM), and a Camelot based pipeline with LLM fallback. Experiments were conducted on 140 documents for the LLM based test and 860 documents for the Camelot based pipeline evaluation, covering four study programs with varying data in tables and metadata. Three 12 - 14B LLM models (Gemma 3, Phi 4, and Qwen 2.5) were run locally using Ollama and a consumer grade CPU without a GPU. Evaluations used exact match (EM) and Levenshtein similarity (LS) metrics with a threshold of 0.7. Although not applicable to all models, the results show that the hybrid approach can improve efficiency compared to LLM only, especially for deterministic metadata. The Camelot based pipeline with LLM fallback produced the best combination of accuracy (EM and LS up to 0.99 - 1.00) and computational efficiency (less than 1 second per PDF in most cases). The Qwen 2.5:14b model demonstrated the most consistent performance across all scenarios. These findings confirm that integrating deterministic and LLM methods is increasingly reliable and efficient for information extraction from text based academic documents in computationally constrained environments. 

---
# Benchmark for Assessing Olfactory Perception of Large Language Models 

**Authors**: Eftychia Makri, Nikolaos Nakis, Laura Sisson, Gigi Minsky, Leandros Tassiulas, Vahid Satarifard, Nicholas A. Christakis  

**Link**: [PDF](https://arxiv.org/pdf/2604.00002)  

**Abstract**: Here we introduce the Olfactory Perception (OP) benchmark, designed to assess the capability of large language models (LLMs) to reason about smell. The benchmark contains 1,010 questions across eight task categories spanning odor classification, odor primary descriptor identification, intensity and pleasantness judgments, multi-descriptor prediction, mixture similarity, olfactory receptor activation, and smell identification from real-world odor sources. Each question is presented in two prompt formats, compound names and isomeric SMILES, to evaluate the effect of molecular representations. Evaluating 21 model configurations across major model families, we find that compound-name prompts consistently outperform isomeric SMILES, with gains ranging from +2.4 to +18.9 percentage points (mean approx +7 points), suggesting current LLMs access olfactory knowledge primarily through lexical associations rather than structural molecular reasoning. The best-performing model reaches 64.4\% overall accuracy, which highlights both emerging capabilities and substantial remaining gaps in olfactory reasoning. We further evaluate a subset of the OP across 21 languages and find that aggregating predictions across languages improves olfactory prediction, with AUROC = 0.86 for the best performing language ensemble model. LLMs should be able to handle olfactory and not just visual or aural information. 

---
# Two-Stage Optimizer-Aware Online Data Selection for Large Language Models 

**Authors**: Fangxin Wang, Peyman Baghershahi, Langzhou He, Henry Peng Zou, Sourav Medya, Philip S. Yu  

**Link**: [PDF](https://arxiv.org/pdf/2604.00001)  

**Abstract**: Gradient-based data selection offers a principled framework for estimating sample utility in large language model (LLM) fine-tuning, but existing methods are mostly designed for offline settings. They are therefore less suited to online fine-tuning, where data arrives sequentially, sample utility is step-dependent, and the effective update geometry is shaped by adaptive optimizers. We propose an optimizer-aware framework for gradient-based online data selection and reweighting in LLM fine-tuning. Our key idea is to view online selection not as static sample ranking, but as shaping the next target-oriented update under the optimizer state. We formulate this as an optimizer-aware update-matching problem, establish its connection to second-order target utility, and show why subset-level construction must account for interactions and redundancy among selected samples. Based on this view, we develop a two-stage Filter-then-Weight algorithm that first filters geometrically useful candidates and then optimizes their coefficients. To make the framework practical for LLMs, we introduce a factorized outer-product gradient representation and optimized matrix computations for long-context data. Experiments show that our method consistently improves convergence and downstream performance over existing online data selection baselines under the same data budget. 

---
# Agentic AI -- Physicist Collaboration in Experimental Particle Physics: A Proof-of-Concept Measurement with LEP Open Data 

**Authors**: Anthony Badea, Yi Chen, Marcello Maggi, Yen-Jie Lee, Electron-Positron Alliance  

**Link**: [PDF](https://arxiv.org/pdf/2603.05735)  

**Abstract**: We present an AI agentic measurement of the thrust distribution in $e^{+}e^{-}$ collisions at $\sqrt{s}=91.2$~GeV using archived ALEPH data. The analysis and all note writing is carried out entirely by AI agents (OpenAI Codex and Anthropic Claude) under expert physicist direction. A fully corrected spectrum is obtained via Iterative Bayesian Unfolding and Monte Carlo based corrections. This work represents a step toward a theory-experiment loop in which AI agents assist with experimental measurements and theoretical calculations, and synthesize insights by comparing the results, thereby accelerating the cycle that drives discovery in fundamental physics. Our work suggests that precision physics, leveraging the open LEP data and advanced theoretical landscape, provides an ideal testing ground for developing advanced AI systems for scientific applications. 

---
