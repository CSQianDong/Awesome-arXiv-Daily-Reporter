# Mecha-nudges for Machines 

**Authors**: Giulio Frey, Kawin Ethayarajh  

**Link**: [PDF](https://arxiv.org/pdf/2603.23433)  

**Abstract**: Nudges are subtle changes to the way choices are presented to human decision-makers (e.g., opt-in vs. opt-out by default) that shift behavior without restricting options or changing incentives. As AI agents increasingly make decisions in the same environments as humans, the presentation of choices may be optimized for machines as well as people. We introduce mecha-nudges: changes to how choices are presented that systematically influence AI agents without degrading the decision environment for humans. To formalize mecha-nudges, we combine the Bayesian persuasion framework with V-usable information, a generalization of Shannon information that is observer-relative. This yields a common scale (bits of usable information) for comparing a wide range of interventions, contexts, and models. Applying our framework to product listings on Etsy -- a global marketplace for independent sellers -- we find that following ChatGPT's release, listings have significantly more machine-usable information about product selection, consistent with systematic mecha-nudging. 

---
# Bilevel Autoresearch: Meta-Autoresearching Itself 

**Authors**: Yaonan Qu, Meng Lu  

**Link**: [PDF](https://arxiv.org/pdf/2603.23420)  

**Abstract**: If autoresearch is itself a form of research, then autoresearch can be applied to research itself. We take this idea literally: we use an autoresearch loop to optimize the autoresearch loop. Every existing autoresearch system -- from Karpathy's single-track loop to AutoResearchClaw's multi-batch extension and EvoScientist's persistent memory -- was improved by a human who read the code, identified a bottleneck, and wrote new code. We ask whether an LLM can do the same, autonomously. We present Bilevel Autoresearch, a bilevel framework where an outer loop meta-optimizes the inner autoresearch loop by generating and injecting new search mechanisms as Python code at runtime. The inner loop optimizes the task; the outer loop optimizes how the inner loop searches. Both loops use the same LLM -- no stronger model is needed at the meta level. On Karpathy's GPT pretraining benchmark, the meta-autoresearch outer loop achieves a 5x improvement over the standard inner loop alone (-0.045 vs. -0.009 val_bpb), while parameter-level adjustment without mechanism change yields no reliable gain. The outer loop autonomously discovers mechanisms from combinatorial optimization, multi-armed bandits, and design of experiments -- without human specification of which domains to explore. These mechanisms succeed by breaking the inner loop's deterministic search patterns, forcing exploration of directions the LLM's priors systematically avoid. The core principle is simple: if autoresearch can meta-autoresearch itself, it can, in principle, meta-autoresearch anything with a measurable objective. 

---
# Beyond Preset Identities: How Agents Form Stances and Boundaries in Generative Societies 

**Authors**: Hanzhong Zhang, Siyang Song, Jindong Wang  

**Link**: [PDF](https://arxiv.org/pdf/2603.23406)  

**Abstract**: While large language models simulate social behaviors, their capacity for stable stance formation and identity negotiation during complex interventions remains unclear. To overcome the limitations of static evaluations, this paper proposes a novel mixed-methods framework combining computational virtual ethnography with quantitative socio-cognitive profiling. By embedding human researchers into generative multiagent communities, controlled discursive interventions are conducted to trace the evolution of collective cognition. To rigorously measure how agents internalize and react to these specific interventions, this paper formalizes three new metrics: Innate Value Bias (IVB), Persuasion Sensitivity, and Trust-Action Decoupling (TAD). Across multiple representative models, agents exhibit endogenous stances that override preset identities, consistently demonstrating an innate progressive bias (IVB > 0). When aligned with these stances, rational persuasion successfully shifts 90% of neutral agents while maintaining high trust. In contrast, conflicting emotional provocations induce a paradoxical 40.0% TAD rate in advanced models, which hypocritically alter stances despite reporting low trust. Smaller models contrastingly maintain a 0% TAD rate, strictly requiring trust for behavioral shifts. Furthermore, guided by shared stances, agents use language interactions to actively dismantle assigned power hierarchies and reconstruct self organized community boundaries. These findings expose the fragility of static prompt engineering, providing a methodological and quantitative foundation for dynamic alignment in human-agent hybrid societies. The official code is available at: this https URL 

---
# RelayS2S: A Dual-Path Speculative Generation for Real-Time Dialogue 

**Authors**: Long Mai  

**Link**: [PDF](https://arxiv.org/pdf/2603.23346)  

**Abstract**: Real-time spoken dialogue systems face a fundamental tension between latency and response quality. End-to-end speech-to-speech (S2S) models respond immediately and naturally handle turn-taking, backchanneling, and interruption, but produce semantically weaker outputs. Cascaded pipelines (ASR -> LLM) deliver stronger responses at the cost of latency that grows with model size. We present RelayS2S, a hybrid architecture that runs two paths in parallel upon turn detection. The fast path -- a duplex S2S model -- speculatively drafts a short response prefix that is streamed immediately to TTS for low-latency audio onset, while continuing to monitor live audio events. The slow path -- a cascaded ASR -> LLM pipeline -- generates a higher-quality continuation conditioned on the committed prefix, producing a seamless utterance. A lightweight learned verifier gates the handoff, committing the prefix when appropriate or falling back gracefully to the slow path alone. Experiments show that RelayS2S achieves P90 onset latency comparable to the S2S model while retaining 99% cascaded response quality in average score, with benefits growing as the slow-path model scales. Because the prefix handoff requires no architectural modification to either component, RelayS2S serves as a lightweight, drop-in addition to existing cascaded pipelines. Our code and data are publicly available at: this https URL 

---
# LLM Olympiad: Why Model Evaluation Needs a Sealed Exam 

**Authors**: Jan Christian Blaise Cruz, Alham Fikri Aji  

**Link**: [PDF](https://arxiv.org/pdf/2603.23292)  

**Abstract**: Benchmarks and leaderboards are how NLP most often communicates progress, but in the LLM era they are increasingly easy to misread. Scores can reflect benchmark-chasing, hidden evaluation choices, or accidental exposure to test content -- not just broad capability. Closed benchmarks delay some of these issues, but reduce transparency and make it harder for the community to learn from results. We argue for a complementary practice: an Olympiad-style evaluation event where problems are sealed until evaluation, submissions are frozen in advance, and all entries run through one standardized harness. After scoring, the full task set and evaluation code are released so results can be reproduced and audited. This design aims to make strong performance harder to ``manufacture'' and easier to trust. 

---
# Online library learning in human visual puzzle solving 

**Authors**: Pinzhe Zhao, Emanuele Sansone, Marta Kryven, Bonan Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2603.23244)  

**Abstract**: When learning a novel complex task, people often form efficient reusable abstractions that simplify future work, despite uncertainty about the future. We study this process in a visual puzzle task where participants define and reuse helpers -- intermediate constructions that capture repeating structure. In an online experiment, participants solved puzzles of increasing difficulty. Early on, they created many helpers, favouring completeness over efficiency. With experience, helper use became more selective and efficient, reflecting sensitivity to reuse and cost. Access to helpers enabled participants to solve puzzles that were otherwise difficult or impossible. Computational modelling shows that human decision times and number of operations used to complete a puzzle increase with search space estimated by a program induction model with library learning. In contrast, raw program length predicts failure but not effort. Together, these results point to online library learning as a core mechanism in human problem solving, allowing people to flexibly build, refine, and reuse abstractions as task demands grow. 

---
# MemCollab: Cross-Agent Memory Collaboration via Contrastive Trajectory Distillation 

**Authors**: Yurui Chang, Yiran Wu, Qingyun Wu, Lu Lin  

**Link**: [PDF](https://arxiv.org/pdf/2603.23234)  

**Abstract**: Large language model (LLM)-based agents rely on memory mechanisms to reuse knowledge from past problem-solving experiences. Existing approaches typically construct memory in a per-agent manner, tightly coupling stored knowledge to a single model's reasoning style. In modern deployments with heterogeneous agents, a natural question arises: can a single memory system be shared across different models? We found that naively transferring memory between agents often degrades performance, as such memory entangles task-relevant knowledge with agent-specific biases. To address this challenge, we propose MemCollab, a collaborative memory framework that constructs agent-agnostic memory by contrasting reasoning trajectories generated by different agents on the same task. This contrastive process distills abstract reasoning constraints that capture shared task-level invariants while suppressing agent-specific artifacts. We further introduce a task-aware retrieval mechanism that conditions memory access on task category, ensuring that only relevant constraints are used at inference time. Experiments on mathematical reasoning and code generation benchmarks demonstrate that MemCollab consistently improves both accuracy and inference-time efficiency across diverse agents, including cross-modal-family settings. Our results show that the collaboratively constructed memory can function as a shared reasoning resource for diverse LLM-based agents. 

---
# PERMA: Benchmarking Personalized Memory Agents via Event-Driven Preference and Realistic Task Environments 

**Authors**: Shuochen Liu, Junyi Zhu, Long Shu, Junda Lin, Yuhao Chen, Haotian Zhang, Chao Zhang, Derong Xu, Jia Li, Bo Tang, Zhiyu Li, Feiyu Xiong, Enhong Chen, Tong Xu  

**Link**: [PDF](https://arxiv.org/pdf/2603.23231)  

**Abstract**: Empowering large language models with long-term memory is crucial for building agents that adapt to users' evolving needs. However, prior evaluations typically interleave preference-related dialogues with irrelevant conversations, reducing the task to needle-in-a-haystack retrieval while ignoring relationships between events that drive the evolution of user preferences. Such settings overlook a fundamental characteristic of real-world personalization: preferences emerge gradually and accumulate across interactions within noisy contexts. To bridge this gap, we introduce PERMA, a benchmark designed to evaluate persona consistency over time beyond static preference recall. Additionally, we incorporate (1) text variability and (2) linguistic alignment to simulate erratic user inputs and individual idiolects in real-world data. PERMA consists of temporally ordered interaction events spanning multiple sessions and domains, with preference-related queries inserted over time. We design both multiple-choice and interactive tasks to probe the model's understanding of persona along the interaction timeline. Experiments demonstrate that by linking related interactions, advanced memory systems can extract more precise preferences and reduce token consumption, outperforming traditional semantic retrieval of raw dialogues. Nevertheless, they still struggle to maintain a coherent persona across temporal depth and cross-domain interference, highlighting the need for more robust personalized memory management in agents. Our code and data are open-sourced at this https URL. 

---
# SAiW: Source-Attributable Invisible Watermarking for Proactive Deepfake Defense 

**Authors**: Bibek Das, Chandranath Adak, Soumi Chattopadhyay, Zahid Akhtar, Soumya Dutta  

**Link**: [PDF](https://arxiv.org/pdf/2603.23178)  

**Abstract**: Deepfakes generated by modern generative models pose a serious threat to information integrity, digital identity, and public trust. Existing detection methods are largely reactive, attempting to identify manipulations after they occur and often failing to generalize across evolving generation techniques. This motivates the need for proactive mechanisms that secure media authenticity at the time of creation. In this work, we introduce SAiW, a Source-Attributed Invisible watermarking Framework for proactive deepfake defense and media provenance verification. Unlike conventional watermarking methods that treat watermark payloads as generic signals, SAiW formulates watermark embedding as a source-conditioned representation learning problem, where watermark identity encodes the originating source and modulates the embedding process to produce discriminative and traceable signatures. The framework integrates feature-wise linear modulation to inject source identity into the embedding network, enabling scalable multi-source watermark generation. A perceptual guidance module derived from human visual system priors ensures that watermark perturbations remain visually imperceptible while maintaining robustness. In addition, a dual-purpose forensic decoder simultaneously reconstructs the embedded watermark and performs source attribution, providing both automated verification and interpretable forensic evidence. Extensive experiments across multiple deepfake datasets demonstrate that SAiW achieves high perceptual quality while maintaining strong robustness against compression, filtering, noise, geometric transformations, and adversarial perturbations. By binding digital media to its origin through invisible yet verifiable markers, SAiW enables reliable authentication and source attribution, providing a scalable foundation for proactive deepfake defense and trustworthy media provenance. 

---
# Describe-Then-Act: Proactive Agent Steering via Distilled Language-Action World Models 

**Authors**: Massimiliano Pappa, Luca Romani, Valentino Sacco, Alessio Palma, Stéphane Lathuilière, Fabio Galasso, Xavier Alameda-Pineda, Indro Spinelli  

**Link**: [PDF](https://arxiv.org/pdf/2603.23149)  

**Abstract**: Deploying safety-critical agents requires anticipating the consequences of actions before they are executed. While world models offer a paradigm for this proactive foresight, current approaches relying on visual simulation incur prohibitive latencies, often exceeding several seconds per step. In this work, we challenge the assumption that visual processing is necessary for failure prevention. We show that a trained policy's latent state, combined with its planned actions, already encodes sufficient information to anticipate action outcomes, making visual simulation redundant for failure prevention. To this end, we introduce DILLO (DIstiLLed Language-ActiOn World Model), a fast steering layer that shifts the paradigm from "simulate-then-act" to "describe-then-act." DILLO is trained via cross-modal distillation, where a privileged Vision Language Model teacher annotates offline trajectories and a latent-conditioned Large Language Model student learns to predict semantic outcomes. This creates a text-only inference path, bypassing heavy visual generation entirely, achieving a 14x speedup over baselines. Experiments on MetaWorld and LIBERO demonstrate that DILLO produces high-fidelity descriptions of the next state and is able to steer the policy, improving episode success rate by up to 15 pp and 9.3 pp on average across tasks. 

---
# Between Rules and Reality: On the Context Sensitivity of LLM Moral Judgment 

**Authors**: Adrian Sauter, Mona Schirmer  

**Link**: [PDF](https://arxiv.org/pdf/2603.23114)  

**Abstract**: A human's moral decision depends heavily on the context. Yet research on LLM morality has largely studied fixed scenarios. We address this gap by introducing Contextual MoralChoice, a dataset of moral dilemmas with systematic contextual variations known from moral psychology to shift human judgment: consequentialist, emotional, and relational. Evaluating 22 LLMs, we find that nearly all models are context-sensitive, shifting their judgments toward rule-violating behavior. Comparing with a human survey, we find that models and humans are most triggered by different contextual variations, and that a model aligned with human judgments in the base case is not necessarily aligned in its contextual sensitivity. This raises the question of controlling contextual sensitivity, which we address with an activation steering approach that can reliably increase or decrease a model's contextual sensitivity. 

---
# MedCausalX: Adaptive Causal Reasoning with Self-Reflection for Trustworthy Medical Vision-Language Models 

**Authors**: Jianxin Lin, Chunzheng Zhu, Peter J. Kneuertz, Yunfei Bai, Yuan Xue  

**Link**: [PDF](https://arxiv.org/pdf/2603.23085)  

**Abstract**: Vision-Language Models (VLMs) have enabled interpretable medical diagnosis by integrating visual perception with linguistic reasoning. Yet, existing medical chain-of-thought (CoT) models lack explicit mechanisms to represent and enforce causal reasoning, leaving them vulnerable to spurious correlations and limiting their clinical reliability. We pinpoint three core challenges in medical CoT reasoning: how to adaptively trigger causal correction, construct high-quality causal-spurious contrastive samples, and maintain causal consistency across reasoning trajectories. To address these challenges, we propose MedCausalX, an end-to-end framework explicitly models causal reasoning chains in medical VLMs. We first introduce the CRMed dataset providing fine-grained anatomical annotations, structured causal reasoning chains, and counterfactual variants that guide the learning of causal relationships beyond superficial correlations. Building upon CRMed, MedCausalX employs a two-stage adaptive reflection architecture equipped with $\langle$causal$\rangle$ and $\langle$verify$\rangle$ tokens, enabling the model to autonomously determine when and how to perform causal analysis and verification. Finally, a trajectory-level causal correction objective optimized through error-attributed reinforcement learning refines the reasoning chain, allowing the model to distinguish genuine causal dependencies from shortcut associations. Extensive experiments on multiple benchmarks show that MedCausalX consistently outperforms state-of-the-art methods, improving diagnostic consistency by +5.4 points, reducing hallucination by over 10 points, and attaining top spatial grounding IoU, thereby setting a new standard for causally grounded medical reasoning. 

---
# Minibal: Balanced Game-Playing Without Opponent Modeling 

**Authors**: Quentin Cohen-Solal, Tristan Cazenave  

**Link**: [PDF](https://arxiv.org/pdf/2603.23059)  

**Abstract**: Recent advances in game AI, such as AlphaZero and Athénan, have achieved superhuman performance across a wide range of board games. While highly powerful, these agents are ill-suited for human-AI interaction, as they consistently overwhelm human players, offering little enjoyment and limited educational value. This paper addresses the problem of balanced play, in which an agent challenges its opponent without either dominating or conceding.
We introduce Minibal (Minimize & Balance), a variant of Minimax specifically designed for balanced play. Building on this concept, we propose several modifications of the Unbounded Minimax algorithm explicitly aimed at discovering balanced strategies.
Experiments conducted across seven board games demonstrate that one variant consistently achieves the most balanced play, with average outcomes close to perfect balance. These results establish Minibal as a promising foundation for designing AI agents that are both challenging and engaging, suitable for both entertainment and serious games. 

---
# Can Large Language Models Reason and Optimize Under Constraints? 

**Authors**: Fabien Bernier, Salah Ghamizi, Pantelis Dogoulis, Maxime Cordy  

**Link**: [PDF](https://arxiv.org/pdf/2603.23004)  

**Abstract**: Large Language Models (LLMs) have demonstrated great capabilities across diverse natural language tasks; yet their ability to solve abstraction and optimization problems with constraints remains scarcely explored. In this paper, we investigate whether LLMs can reason and optimize under the physical and operational constraints of Optimal Power Flow (OPF) problem. We introduce a challenging evaluation setup that requires a set of fundamental skills such as reasoning, structured input handling, arithmetic, and constrained optimization. Our evaluation reveals that SoTA LLMs fail in most of the tasks, and that reasoning LLMs still fail in the most complex settings. Our findings highlight critical gaps in LLMs' ability to handle structured reasoning under constraints, and this work provides a rigorous testing environment for developing more capable LLM assistants that can tackle real-world power grid optimization problems. 

---
# On the use of Aggregation Operators to improve Human Identification using Dental Records 

**Authors**: Antonio D. Villegas-Yeguas, Guillermo R-García, Tzipi Kahana, Jorge Pinares Toledo, Esi Sharon, Oscar Ibañez, Oscar Cordón  

**Link**: [PDF](https://arxiv.org/pdf/2603.23003)  

**Abstract**: The comparison of dental records is a standardized technique in forensic dentistry used to speed up the identification of individuals in multiple-comparison scenarios. Specifically, the odontogram comparison is a procedure to compute criteria that will be used to perform a ranking. State-of-the-art automatic methods either make use of simple techniques, without utilizing the full potential of the information obtained from a comparison, or their internal behavior is not known due to the lack of peer-reviewed publications. This work aims to design aggregation mechanisms to automatically compare pairs of dental records that can be understood and validated by experts, improving the current methods. To do so, we introduce different aggregation approaches using the state-of-the-art codification, based on seven different criteria. In particular, we study the performance of i) data-driven lexicographical order-based aggregations, ii) well-known fuzzy logic aggregation methods and iii) machine learning techniques as aggregation mechanisms. To validate our proposals, 215 forensic cases from two different populations have been used. The results obtained show how the use of white-box machine learning techniques as aggregation models (average ranking from 2.02 to 2.21) are able to improve the state-of-the-art (average ranking of 3.91) without compromising the explainability and interpretability of the method. 

---
# JFTA-Bench: Evaluate LLM's Ability of Tracking and Analyzing Malfunctions Using Fault Trees 

**Authors**: Yuhui Wang, Zhixiong Yang, Ming Zhang, Shihan Dou, Zhiheng Xi, Enyu Zhou, Senjie Jin, Yujiong Shen, Dingwei Zhu, Yi Dong, Tao Gui, Qi Zhang, Xuanjing Huang  

**Link**: [PDF](https://arxiv.org/pdf/2603.22978)  

**Abstract**: In the maintenance of complex systems, fault trees are used to locate problems and provide targeted solutions. To enable fault trees stored as images to be directly processed by large language models, which can assist in tracking and analyzing malfunctions, we propose a novel textual representation of fault trees. Building on it, we construct a benchmark for multi-turn dialogue systems that emphasizes robust interaction in complex environments, evaluating a model's ability to assist in malfunction localization, which contains $3130$ entries and $40.75$ turns per entry on average. We train an end-to-end model to generate vague information to reflect user behavior and introduce long-range rollback and recovery procedures to simulate user error scenarios, enabling assessment of a model's integrated capabilities in task tracking and error recovery, and Gemini 2.5 pro archives the best performance. 

---
# Where Experts Disagree, Models Fail: Detecting Implicit Legal Citations in French Court Decisions 

**Authors**: Avrile Floro, Tamara Dhorasoo, Soline Pellez, Nils Holzenberger  

**Link**: [PDF](https://arxiv.org/pdf/2603.22973)  

**Abstract**: Computational methods applied to legal scholarship hold the promise of analyzing law at scale. We start from a simple question: how often do courts implicitly apply statutory rules? This requires distinguishing legal reasoning from semantic similarity. We focus on implicit citation of the French Civil Code in first-instance court decisions and introduce a benchmark of 1,015 passage-article pairs annotated by three legal experts. We show that expert disagreement predicts model failures. Inter-annotator agreement is moderate ($\kappa$ = 0.33) with 43% of disagreements involving the boundary between factual description and legal reasoning. Our supervised ensemble achieves F1 = 0.70 (77% accuracy), but this figure conceals an asymmetry: 68% of false positives fall on the 33% of cases where the annotators disagreed. Despite these limits, reframing the task as top-k ranking and leveraging multi-model consensus yields 76% precision at k = 200 in an unsupervised setting. Moreover, the remaining false positives tend to surface legally ambiguous applications rather than obvious errors. 

---
# PersonalQ: Select, Quantize, and Serve Personalized Diffusion Models for Efficient Inference 

**Authors**: Qirui Wang, Qi Guo, Yiding Sun, Junkai Yang, Dongxu Zhang, Shanmin Pang, Qing Guo  

**Link**: [PDF](https://arxiv.org/pdf/2603.22943)  

**Abstract**: Personalized text-to-image generation lets users fine-tune diffusion models into repositories of concept-specific checkpoints, but serving these repositories efficiently is difficult for two reasons: natural-language requests are often ambiguous and can be misrouted to visually similar checkpoints, and standard post-training quantization can distort the fragile representations that encode personalized concepts. We present PersonalQ, a unified framework that connects checkpoint selection and quantization through a shared signal -- the checkpoint's trigger token. Check-in performs intent-aligned selection by combining intent-aware hybrid retrieval with LLM-based reranking over checkpoint context and asks a brief clarification question only when multiple intents remain plausible; it then rewrites the prompt by inserting the selected checkpoint's canonical trigger. Complementing this, Trigger-Aware Quantization (TAQ) applies trigger-aware mixed precision in cross-attention, preserving trigger-conditioned key/value rows (and their attention weights) while aggressively quantizing the remaining pathways for memory-efficient inference. Experiments show that PersonalQ improves intent alignment over retrieval and reranking baselines, while TAQ consistently offers a stronger compression-quality trade-off than prior diffusion PTQ methods, enabling scalable serving of personalized checkpoints without sacrificing fidelity. 

---
# Optimizing Small Language Models for NL2SQL via Chain-of-Thought Fine-Tuning 

**Authors**: Anshul Solanki, Sanchit Latawa, Koushik Chakraborty, Navneet Kamboj  

**Link**: [PDF](https://arxiv.org/pdf/2603.22942)  

**Abstract**: Translating Natural Language to SQL (NL2SQL) remains a critical bottleneck for democratization of data in enterprises. Although Large Language Models (LLMs) like Gemini 2.5 and other LLMs have demonstrated impressive zero-shot capabilities, their high inference costs limit deployment at scale. This paper explores the efficacy of fine-tuning both large and small language models on NL2SQL tasks.
Our research reveals a counter-intuitive scaling phenomenon. Fine-tuning large models (Gemini 2.5 Flash/Lite) on standard datasets yields negligible returns, often leading to overfitting on complex queries. Conversely, small models (Qwen) show significant gains. Fine-tuning improved the small model baseline from 36% to 45%, and further enriching the dataset with explicit Chain-of-Thought (CoT) reasoning surged accuracy to 54.5%(Fig 2). While this is still lower than the accuracy of large models like Gemini 2.5 , it does serve the business goal of significant cost reduction, latency in inference time and also meeting the business critical performance accuracy this http URL paper demonstrates that transferring reasoning patterns enables compute-efficient smaller models to approach production-grade performance. 

---
# Ran Score: a LLM-based Evaluation Score for Radiology Report Generation 

**Authors**: Ran Zhang, Yucong Lin, Zhaoli Su, Bowen Liu, Danni Ai, Tianyu Fu, Deqiang Xiao, Jingfan Fan, Yuanyuan Wang, Mingwei Gao, Yuwan Hu, Shuya Gao, Jingtao Li, Jian Yang, Hong Song, Hongliang Sun  

**Link**: [PDF](https://arxiv.org/pdf/2603.22935)  

**Abstract**: Chest X-ray report generation and automated evaluation are limited by poor recognition of low-prevalence abnormalities and inadequate handling of clinically important language, including negation and ambiguity. We develop a clinician-guided framework combining human expertise and large language models for multi-label finding extraction from free-text chest X-ray reports and use it to define Ran Score, a finding-level metric for report evaluation. Using three non-overlapping MIMIC-CXR-EN cohorts from a public chest X-ray dataset and an independent ChestX-CN validation cohort, we optimize prompts, establish radiologist-derived reference labels and evaluate report generation models. The optimized framework improves the macro-averaged score from 0.753 to 0.956 on the MIMIC-CXR-EN development cohort, exceeds the CheXbert benchmark by 15.7 percentage points on directly comparable labels, and shows robust generalization on the ChestX-CN validation cohort. Here we show that clinician-guided prompt optimization improves agreement with a radiologist-derived reference standard and that Ran Score enables finding-level evaluation of report fidelity, particularly for low-prevalence abnormalities. 

---
# ProGRank: Probe-Gradient Reranking to Defend Dense-Retriever RAG from Corpus Poisoning 

**Authors**: Xiangyu Yin, Yi Qi, Chih-hong Cheng  

**Link**: [PDF](https://arxiv.org/pdf/2603.22934)  

**Abstract**: Retrieval-Augmented Generation (RAG) improves the reliability of large language model applications by grounding generation in retrieved evidence, but it also introduces a new attack surface: corpus poisoning. In this setting, an adversary injects or edits passages so that they are ranked into the Top-$K$ results for target queries and then affect downstream generation. Existing defences against corpus poisoning often rely on content filtering, auxiliary models, or generator-side reasoning, which can make deployment more difficult. We propose ProGRank, a post hoc, training-free retriever-side defence for dense-retriever RAG. ProGRank stress-tests each query--passage pair under mild randomized perturbations and extracts probe gradients from a small fixed parameter subset of the retriever. From these signals, it derives two instability signals, representational consistency and dispersion risk, and combines them with a score gate in a reranking step. ProGRank preserves the original passage content, requires no retraining, and also supports a surrogate-based variant when the deployed retriever is unavailable. Extensive experiments across three datasets, three dense retriever backbones, representative corpus poisoning attacks, and both retrieval-stage and end-to-end settings show that ProGRank provides stronger defence performance and a favorable robustness--utility trade-off. It also remains competitive under adaptive evasive attacks. 

---
# Separating Diagnosis from Control: Auditable Policy Adaptation in Agent-Based Simulations with LLM-Based Diagnostics 

**Authors**: Shaoxin Zhong, Yuchen Su, Michael Witbrock  

**Link**: [PDF](https://arxiv.org/pdf/2603.22904)  

**Abstract**: Mitigating elderly loneliness requires policy interventions that achieve both adaptability and auditability. Existing methods struggle to reconcile these objectives: traditional agent-based models suffer from static rigidity, while direct large language model (LLM) controllers lack essential traceability. This work proposes a three-layer framework that separates diagnosis from control to achieve both properties simultaneously. LLMs operate strictly as diagnostic instruments that assess population state and generate structured risk evaluations, while deterministic formulas with explicit bounds translate these assessments into traceable parameter updates. This separation ensures that every policy decision can be attributed to inspectable rules while maintaining adaptive response to emergent needs. We validate the framework through systematic ablation across five experimental conditions in elderly care simulation. Results demonstrate that explicit control rules outperform end-to-end black-box LLM approaches by 11.7\% while preserving full auditability, confirming that transparency need not compromise adaptive performance. 

---
# Continuous Optimization for Satisfiability Modulo Theories on Linear Real Arithmetic 

**Authors**: Yunuo Cen, Daniel Ebler, Xuanyao Fong  

**Link**: [PDF](https://arxiv.org/pdf/2603.22877)  

**Abstract**: Efficient solutions for satisfiability modulo theories (SMT) are integral in industrial applications such as hardware verification and design automation. Existing approaches are predominantly based on conflict-driven clause learning, which is structurally difficult to parallelize and therefore scales poorly. In this work, we introduce FourierSMT as a scalable and highly parallelizable continuous-variable optimization framework for SMT. We generalize the Walsh-Fourier expansion (WFE), called extended WFE (xWFE), from the Boolean domain to a mixed Boolean-real domain, which allows the use of gradient methods for SMT. This addresses the challenge of finding satisfying variable assignments to high-arity constraints by local updates of discrete variables. To reduce the evaluation complexity of xWFE, we present the extended binary decision diagram (xBDD) and map the constraints from xWFE to xBDDs. We then show that sampling the circuit-output probability (COP) of xBDDs under randomized rounding is equivalent to the expectation value of the xWFEs. This allows for efficient computation of the constraints. We show that the reduced problem is guaranteed to converge and preserves satisfiability, ensuring the soundness of the solutions. The framework is benchmarked for large-scale scheduling and placement problems with up to 10,000 variables and 700,000 constraints, achieving 8-fold speedups compared to state-of-the-art SMT solvers. These results pave the way for GPU-based optimization of SMTs with continuous systems. 

---
# Dynamical Systems Theory Behind a Hierarchical Reasoning Model 

**Authors**: Vasiliy A. Es'kin, Mikhail E. Smorkalov  

**Link**: [PDF](https://arxiv.org/pdf/2603.22871)  

**Abstract**: Current large language models (LLMs) primarily rely on linear sequence generation and massive parameter counts, yet they severely struggle with complex algorithmic reasoning. While recent reasoning architectures, such as the Hierarchical Reasoning Model (HRM) and Tiny Recursive Model (TRM), demonstrate that compact recursive networks can tackle these tasks, their training dynamics often lack rigorous mathematical guarantees, leading to instability and representational collapse. We propose the Contraction Mapping Model (CMM), a novel architecture that reformulates discrete recursive reasoning into continuous Neural Ordinary and Stochastic Differential Equations (NODEs/NSDEs). By explicitly enforcing the convergence of the latent phase point to a stable equilibrium state and mitigating feature collapse with a hyperspherical repulsion loss, the CMM provides a mathematically grounded and highly stable reasoning engine. On the Sudoku-Extreme benchmark, a 5M-parameter CMM achieves a state-of-the-art accuracy of 93.7 %, outperforming the 27M-parameter HRM (55.0 %) and 5M-parameter TRM (87.4 %). Remarkably, even when aggressively compressed to an ultra-tiny footprint of just 0.26M parameters, the CMM retains robust predictive power, achieving 85.4 % on Sudoku-Extreme and 82.2 % on the Maze benchmark. These results establish a new frontier for extreme parameter efficiency, proving that mathematically rigorous latent dynamics can effectively replace brute-force scaling in artificial reasoning. 

---
# Chain-of-Authorization: Internalizing Authorization into Large Language Models via Reasoning Trajectories 

**Authors**: Yang Li, Yule Liu, Xinlei He, Youjian Zhao, Qi Li, Ke Xu  

**Link**: [PDF](https://arxiv.org/pdf/2603.22869)  

**Abstract**: Large Language Models (LLMs) have become core cognitive components in modern artificial intelligence (AI) systems, combining internal knowledge with external context to perform complex tasks. However, LLMs typically treat all accessible data indiscriminately, lacking inherent awareness of knowledge ownership and access boundaries. This deficiency heightens risks of sensitive data leakage and adversarial manipulation, potentially enabling unauthorized system access and severe security crises. Existing protection strategies rely on rigid, uniform defense that prevent dynamic authorization. Structural isolation methods faces scalability bottlenecks, while prompt guidance methods struggle with fine-grained permissions distinctions. Here, we propose the Chain-of-Authorization (CoA) framework, a secure training and reasoning paradigm that internalizes authorization logic into LLMs' core capabilities. Unlike passive external defneses, CoA restructures the model's information flow: it embeds permission context at input and requires generating explicit authorization reasoning trajectory that includes resource review, identity resolution, and decision-making stages before final response. Through supervised fine-tuning on data covering various authorization status, CoA integrates policy execution with task responses, making authorization a causal prerequisite for substantive responses. Extensive evaluations show that CoA not only maintains comparable utility in authorized scenarios but also overcomes the cognitive confusion when permissions mismatches. It exhibits high rejection rates against various unauthorized and adversarial access. This mechanism leverages LLMs' reasoning capability to perform dynamic authorization, using natural language understanding as a proactive security mechanism for deploying reliable LLMs in modern AI systems. 

---
# CoMaTrack: Competitive Multi-Agent Game-Theoretic Tracking with Vision-Language-Action Models 

**Authors**: Youzhi Liu, Li Gao, Liu Liu, Mingyang Lv, Yang Cai  

**Link**: [PDF](https://arxiv.org/pdf/2603.22846)  

**Abstract**: Embodied Visual Tracking (EVT), a core dynamic task in embodied intelligence, requires an agent to precisely follow a language-specified target. Yet most existing methods rely on single-agent imitation learning, suffering from costly expert data and limited generalization due to static training environments. Inspired by competition-driven capability evolution, we propose CoMaTrack, a competitive game-theoretic multi-agent reinforcement learning framework that trains agents in a dynamic adversarial setting with competitive subtasks, yielding stronger adaptive planning and interference-resilient strategies. We further introduce CoMaTrack-Bench, the first benchmark for competitive EVT, featuring game scenarios between a tracker and adaptive opponents across diverse environments and instructions, enabling standardized robustness evaluation under active adversarial interactions. Experiments show that CoMaTrack achieves state-of-the-art results on both standard benchmarks and CoMaTrack-Bench. Notably, a 3B VLM trained with our framework surpasses previous single-agent imitation learning methods based on 7B models on the challenging EVT-Bench, achieving 92.1% in STT, 74.2% in DT, and 57.5% in AT. The benchmark code will be available at this https URL 

---
# PhySe-RPO: Physics and Semantics Guided Relative Policy Optimization for Diffusion-Based Surgical Smoke Removal 

**Authors**: Zining Fang, Cheng Xue, Chunhui Liu, Bin Xu, Ming Chen, Xiaowei Hu  

**Link**: [PDF](https://arxiv.org/pdf/2603.22844)  

**Abstract**: Surgical smoke severely degrades intraoperative video quality, obscuring anatomical structures and limiting surgical perception. Existing learning-based desmoking approaches rely on scarce paired supervision and deterministic restoration pipelines, making it difficult to perform exploration or reinforcement-driven refinement under real surgical conditions. We propose PhySe-RPO, a diffusion restoration framework optimized through Physics- and Semantics-Guided Relative Policy Optimization. The core idea is to transform deterministic restoration into a stochastic policy, enabling trajectory-level exploration and critic-free updates via group-relative optimization. A physics-guided reward imposes illumination and color consistency, while a visual-concept semantic reward learned from CLIP-based surgical concepts promotes smoke-free and anatomically coherent restoration. Together with a reference-free perceptual constraint, PhySe-RPO produces results that are physically consistent, semantically faithful, and clinically interpretable across synthetic and real robotic surgical datasets, providing a principled route to robust diffusion-based restoration under limited paired supervision. 

---
# Improving Safety Alignment via Balanced Direct Preference Optimization 

**Authors**: Shiji Zhao, Mengyang Wang, Shukun Xiong, Fangzhou Chen, Qihui Zhu, Shouwei Ruan, Yisong Xiao, Ranjie Duan, Xun Chen, XingXing Wei  

**Link**: [PDF](https://arxiv.org/pdf/2603.22829)  

**Abstract**: With the rapid development and widespread application of Large Language Models (LLMs), their potential safety risks have attracted widespread attention. Reinforcement Learning from Human Feedback (RLHF) has been adopted to enhance the safety performance of LLMs. As a simple and effective alternative to RLHF, Direct Preference Optimization (DPO) is widely used for safety alignment. However, safety alignment still suffers from severe overfitting, which limits its actual performance. This paper revisits the overfitting phenomenon from the perspective of the model's comprehension of the training data. We find that the Imbalanced Preference Comprehension phenomenon exists between responses in preference pairs, which compromises the model's safety performance. To address this, we propose Balanced Direct Preference Optimization (B-DPO), which adaptively modulates optimization strength between preferred and dispreferred responses based on mutual information. A series of experimental results show that B-DPO can enhance the safety capability while maintaining the competitive general capabilities of LLMs on various mainstream benchmarks compared to state-of-the-art methods. \color{red}{Warning: This paper contains examples of harmful texts, and reader discretion is recommended. 

---
# Empirical Comparison of Agent Communication Protocols for Task Orchestration 

**Authors**: Ivan Dobrovolskyi  

**Link**: [PDF](https://arxiv.org/pdf/2603.22823)  

**Abstract**: Context. Nowadays, artificial intelligence agent systems are transforming from single-tool interactions to complex multi-agent orchestrations. As a result, two competing communication protocols have emerged: a tool integration protocol that standardizes how agents invoke external tools, and an inter-agent delegation protocol that enables autonomous agents to discover and delegate tasks to one another. Despite widespread industry adoption by dozens of enterprise partners, no empirical comparison of these protocols exists in the literature. Objective. The goal of this work is to develop the first systematic benchmark comparing tool-integration-only, multi-agent delegation, and hybrid architectures across standardized queries at three complexity levels, and to quantify the trade-offs in response time, context window consumption, monetary cost, error recovery, and implementation complexity. 

---
# Learning What Matters Now: Dynamic Preference Inference under Contextual Shifts 

**Authors**: Xianwei Cao, Dou Quan, Zhenliang Zhang, Shuang Wang  

**Link**: [PDF](https://arxiv.org/pdf/2603.22813)  

**Abstract**: Humans often juggle multiple, sometimes conflicting objectives and shift their priorities as circumstances change, rather than following a fixed objective function. In contrast, most computational decision-making and multi-objective RL methods assume static preference weights or a known scalar reward. In this work, we study sequential decision-making problem when these preference weights are unobserved latent variables that drift with context. Specifically, we propose Dynamic Preference Inference (DPI), a cognitively inspired framework in which an agent maintains a probabilistic belief over preference weights, updates this belief from recent interaction, and conditions its policy on inferred preferences. We instantiate DPI as a variational preference inference module trained jointly with a preference-conditioned actor-critic, using vector-valued returns as evidence about latent trade-offs. In queueing, maze, and multi-objective continuous-control environments with event-driven changes in objectives, DPI adapts its inferred preferences to new regimes and achieves higher post-shift performance than fixed-weight and heuristic envelope baselines. 

---
# Reliable Classroom AI via Neuro-Symbolic Multimodal Reasoning 

**Authors**: Sina Bagheri Nezhad  

**Link**: [PDF](https://arxiv.org/pdf/2603.22793)  

**Abstract**: Classroom AI is rapidly expanding from low-level perception toward higher-level judgments about engagement, confusion, collaboration, and instructional quality. Yet classrooms are among the hardest real-world settings for multimodal vision: they are multi-party, noisy, privacy-sensitive, pedagogically diverse, and often multilingual. In this paper, we argue that classroom AI should be treated as a critical domain, where raw predictive accuracy is insufficient unless predictions are accompanied by verifiable evidence, calibrated uncertainty, and explicit deployment guardrails. We introduce NSCR, a neuro-symbolic framework that decomposes classroom analytics into four layers: perceptual grounding, symbolic abstraction, executable reasoning, and governance. NSCR adapts recent ideas from symbolic fact extraction and verifiable code generation to multimodal educational settings, enabling classroom observations from video, audio, ASR, and contextual metadata to be converted into typed facts and then composed by executable rules, programs, and policy constraints. Beyond the system design, we contribute a benchmark and evaluation protocol organized around five tasks: classroom state inference, discourse-grounded event linking, temporal early warning, collaboration analysis, and multilingual classroom reasoning. We further specify reliability metrics centered on abstention, calibration, robustness, construct alignment, and human usefulness. The paper does not report new empirical results; its contribution is a concrete framework and evaluation agenda intended to support more interpretable, privacy-aware, and pedagogically grounded multimodal AI for classrooms. 

---
# ABSTRAL: Automatic Design of Multi-Agent Systems Through Iterative Refinement and Topology Optimization 

**Authors**: Weijia Song, Jiashu Yue, Zhe Pang  

**Link**: [PDF](https://arxiv.org/pdf/2603.22791)  

**Abstract**: How should multi-agent systems be designed, and can that design knowledge be captured in a form that is inspectable, revisable, and transferable? We introduce ABSTRAL, a framework that treats MAS architecture as an evolving natural-language document, an artifact refined through contrastive trace analysis. Three findings emerge. First, we provide a precise measurement of the multi-agent coordination tax: under fixed turn budgets, ensembles achieve only 26% turn efficiency, with 66% of tasks exhausting the limit, yet still improve over single-agent baselines by discovering parallelizable task decompositions. Second, design knowledge encoded in documents transfers: topology reasoning and role templates learned on one domain provide a head start on new domains, with transferred seeds matching coldstart iteration 3 performance in a single iteration. Third, contrastive trace analysis discovers specialist roles absent from any initial design, a capability no prior system demonstrates. On SOPBench (134 bank tasks, deterministic oracle), ABSTRAL reaches 70% validation / 65.96% test pass rate with a GPT-4o backbone. We release the converged documents as inspectable design rationale. 

---
# AgriPestDatabase-v1.0: A Structured Insect Dataset for Training Agricultural Large Language Model 

**Authors**: Yagizhan Bilal Durak, Ahsan Ul Islam, Shahidul Islam, Ashley Morgan-Olvera, Iftekhar Ibne Basith, Syed Hasib Akhter Faruqui  

**Link**: [PDF](https://arxiv.org/pdf/2603.22777)  

**Abstract**: Agricultural pest management increasingly relies on timely and accurate access to expert knowledge, yet high quality labeled data and continuous expert support remain limited, particularly for farmers operating in rural regions with unstable/no internet connectivity. At the same time, the rapid growth of AI and LLMs has created new opportunities to deliver practical decision support tools directly to end users in agriculture through compact and deployable systems. This work addresses (i) generating a structured insect information dataset, and (ii) adapting a lightweight LLM model ($\leq$ 7B) by fine tuning it for edge device uses in agricultural pest management. The textual data collection was done by reviewing and collecting information from available pest databases and published manuscripts on nine selected pest species. These structured reports were then reviewed and validated by a domain expert. From these reports, we constructed Q/A pairs to support model training and evaluation. A LoRA-based fine-tuning approach was applied to multiple lightweight LLMs and evaluated. Initial evaluation shows that Mistral 7B achieves an 88.9\% pass rate on the domain-specific Q/A task, substantially outperforming Qwen 2.5 7B (63.9\%), and LLaMA 3.1 8B (58.7\%). Notably, Mistral demonstrates higher semantic alignment (embedding similarity: 0.865) despite lower lexical overlap (BLEU: 0.097), indicating that semantic understanding and robust reasoning are more predictive of task success than surface-level conformity in specialized domains. By combining expert organized data, well-structured Q/A pairs, semantic quality control, and efficient model adaptation, this work contributes towards providing support for farmer facing agricultural decision support tools and demonstrates the feasibility of deploying compact, high-performing language models for practical field-level pest management guidance. 

---
# Can LLM Agents Generate Real-World Evidence? Evaluating Observational Studies in Medical Databases 

**Authors**: Dubai Li, Yuxiang He, Yan Hu, Yu Tian, Jingsong Li  

**Link**: [PDF](https://arxiv.org/pdf/2603.22767)  

**Abstract**: Observational studies can yield clinically actionable evidence at scale, but executing them on real-world databases is open-ended and requires coherent decisions across cohort construction, analysis, and reporting. Prior evaluations of LLM agents emphasize isolated steps or single answers, missing the integrity and internal structure of the resulting evidence bundle. To address this gap, we introduce RWE-bench, a benchmark grounded in MIMIC-IV and derived from peer-reviewed observational studies. Each task provides the corresponding study protocol as the reference standard, requiring agents to execute experiments in a real database and iteratively generate tree-structured evidence bundles. We evaluate six LLMs (three open-source, three closed-source) under three agent scaffolds using both question-level correctness and end-to-end task metrics. Across 162 tasks, task success is low: the best agent reaches 39.9%, and the best open-source model reaches 30.4%. Agent scaffolds also matter substantially, causing over 30% variation in performance metrics. Furthermore, we implement an automated cohort evaluation method to rapidly localize errors and identify agent failure modes. Overall, the results highlight persistent limitations in agents' ability to produce end-to-end evidence bundles, and efficient validation remains an important direction for future work. Code and data are available at this https URL. 

---
# CLiGNet: Clinical Label-Interaction Graph Network for Medical Specialty Classification from Clinical Transcriptions 

**Authors**: Pronob Kumar Barman, Pronoy Kumar Barman  

**Link**: [PDF](https://arxiv.org/pdf/2603.22752)  

**Abstract**: Automated classification of clinical transcriptions into medical specialties is essential for routing, coding, and clinical decision support, yet prior work on the widely used MTSamples benchmark suffers from severe data leakage caused by applying SMOTE oversampling before train test splitting. We first document this methodological flaw and establish a leakage free benchmark across 40 medical specialties (4966 records), revealing that the true task difficulty is substantially higher than previously reported.
We then introduce CLiGNet (Clinical Label Interaction Graph Network), a neural architecture that combines a Bio ClinicalBERT text encoder with a two layer Graph Convolutional Network operating on a specialty label graph constructed from semantic similarity and ICD 10 chapter priors. Per label attention gates fuse document and label graph representations, trained with focal binary cross entropy loss to handle extreme class imbalance (181 to 1 ratio).
Across seven baselines ranging from TF IDF classifiers to Clinical Longformer, CLiGNet without calibration achieves the highest macro F1 of 0.279, with an ablation study confirming that the GCN label graph provides the single largest component gain (increase of 0.066 macro F1). Adding per label Platt scaling calibration yields an expected calibration error of 0.007, demonstrating a principled trade off between ranking performance and probability reliability.
We provide comprehensive failure analysis covering pairwise specialty confusions, rare class behaviour, document length effects, and token level Integrated Gradients attribution, offering actionable insights for clinical NLP system deployment. 

---
# Beyond Binary Correctness: Scaling Evaluation of Long-Horizon Agents on Subjective Enterprise Tasks 

**Authors**: Abhishek Chandwani, Ishan Gupta  

**Link**: [PDF](https://arxiv.org/pdf/2603.22744)  

**Abstract**: Large language models excel on objectively verifiable tasks such as math and programming, where evaluation reduces to unit tests or a single correct answer. In contrast, real-world enterprise work is often subjective and context-dependent: success hinges on organizational goals, user intent, and the quality of intermediate artifacts produced across long, multi-tool workflows.
We introduce LH-Bench, a three-pillar evaluation design that moves beyond binary correctness to score autonomous, long-horizon execution on subjective enterprise tasks. The pillars are: (i) expert-grounded rubrics that give LLM judges the domain context needed to score subjective work, (ii) curated ground-truth artifacts that enable stepwise reward signals (e.g., chapter-level annotation for content tasks), and (iii) pairwise human preference evaluation for convergent validation. We show that domain-authored rubrics provide substantially more reliable evaluation signals than LLM-authored rubrics (kappa = 0.60 vs. 0.46), and that human preference judgments confirm the same top-tier separation (p < 0.05), evidence that expert-grounded evaluation can scale without sacrificing reliability. We release public datasets and report results on two environments: Figma-to-code (33 real .fig tasks against the Figma API via MCP) and Programmatic content (41 courses comprising 183 individually-evaluated chapters on a course platform serving 30+ daily users). 

---
# HyFI: Hyperbolic Feature Interpolation for Brain-Vision Alignment 

**Authors**: Sangmin Jo, Wootaek Jeong, Da-Woon Heo, Yoohwan Hwang, Heung-Il Suk  

**Link**: [PDF](https://arxiv.org/pdf/2603.22721)  

**Abstract**: Recent progress in artificial intelligence has encouraged numerous attempts to understand and decode human visual system from brain signals. These prior works typically align neural activity independently with semantic and perceptual features extracted from images using pre-trained vision models. However, they fail to account for two key challenges: (1) the modality gap arising from the natural difference in the information level of representation between brain signals and images, and (2) the fact that semantic and perceptual features are highly entangled within neural activity. To address these issues, we utilize hyperbolic space, which is well-suited for considering differences in the amount of information and has the geometric property that geodesics between two points naturally bend toward the origin, where the representational capacity is lower. Leveraging these properties, we propose a novel framework, Hyperbolic Feature Interpolation (HyFI), which interpolates between semantic and perceptual visual features along hyperbolic geodesics. This enables both the fusion and compression of perceptual and semantic information, effectively reflecting the limited expressiveness of brain signals and the entangled nature of these features. As a result, it facilitates better alignment between brain and visual features. We demonstrate that HyFI achieves state-of-the-art performance in zero-shot brain-to-image retrieval, outperforming prior methods with Top-1 accuracy improvements of up to +17.3% on THINGS-EEG and +9.1% on THINGS-MEG. 

---
# MuQ-Eval: An Open-Source Per-Sample Quality Metric for AI Music Generation Evaluation 

**Authors**: Di Zhu, Zixuan Li  

**Link**: [PDF](https://arxiv.org/pdf/2603.22677)  

**Abstract**: Distributional metrics such as Fréchet Audio Distance cannot score individual music clips and correlate poorly with human judgments, while the only per-sample learned metric achieving high human correlation is closed-source. We introduce MUQ-EVAL, an open-source per-sample quality metric for AIgenerated music built by training lightweight prediction heads on frozen MuQ-310M features using MusicEval, a dataset of generated clips from 31 text-to-music systems with expert quality ratings. Our simplest model, frozen features with attention pooling and a two-layer MLP, achieves system-level SRCC = 0.957 and utterance-level SRCC = 0.838 with human mean opinion scores. A systematic ablation over training objectives and adaptation strategies shows that no addition meaningfully improves the frozen baseline, indicating that frozen MuQ representations already capture quality-relevant information. Encoder choice is the dominant design factor, outweighing all architectural and training decisions. LoRA-adapted models trained on as few as 150 clips already achieve usable correlation, enabling personalized quality evaluators from individual listener annotations. A controlled degradation analysis reveals selective sensitivity to signal-level artifacts but insensitivity to musical-structural distortions. Our metric, MUQ-EVAL, is fully open-source, outperforms existing open per-sample metrics, and runs in real time on a single consumer GPU. Code, model weights, and evaluation scripts are available at this https URL. 

---
# Benchmarking Multi-Agent LLM Architectures for Financial Document Processing: A Comparative Study of Orchestration Patterns, Cost-Accuracy Tradeoffs and Production Scaling Strategies 

**Authors**: Siddhant Kulkarni, Yukta Kulkarni  

**Link**: [PDF](https://arxiv.org/pdf/2603.22651)  

**Abstract**: The adoption of large language models (LLMs) for structured information extraction from financial documents has accelerated rapidly, yet production deployments face fundamental architectural decisions with limited empirical guidance. We present a systematic benchmark comparing four multi-agent orchestration architectures: sequential pipeline, parallel fan-out with merge, hierarchical supervisor-worker and reflexive self-correcting loop. These are evaluated across five frontier and open-weight LLMs on a corpus of 10,000 SEC filings (10-K, 10-Q and 8-K forms). Our evaluation spans 25 extraction field types covering governance structures, executive compensation and financial metrics, measured along five axes: field-level F1, document-level accuracy, end-to-end latency, cost per document and token efficiency. We find that reflexive architectures achieve the highest field-level F1 (0.943) but at 2.3x the cost of sequential baselines, while hierarchical architectures occupy the most favorable position on the cost-accuracy Pareto frontier (F1 0.921 at 1.4x cost). We further present ablation studies on semantic caching, model routing and adaptive retry strategies, demonstrating that hybrid configurations can recover 89\% of the reflexive architecture's accuracy gains at only 1.15x baseline cost. Our scaling analysis from 1K to 100K documents per day reveals non-obvious throughput-accuracy degradation curves that inform capacity planning. These findings provide actionable guidance for practitioners deploying multi-agent LLM systems in regulated financial environments. 

---
# Graph-Aware Late Chunking for Retrieval-Augmented Generation in Biomedical Literature 

**Authors**: Pouria Mortezaagha, Arya Rahgozar  

**Link**: [PDF](https://arxiv.org/pdf/2603.22633)  

**Abstract**: Retrieval-Augmented Generation (RAG) systems for biomedical literature are typically evaluated using ranking metrics like Mean Reciprocal Rank (MRR), which measure how well the system identifies the single most relevant chunk. We argue that for full-text scientific documents, this paradigm is incomplete: it rewards retrieval precision while ignoring retrieval breadth -- the ability to surface evidence from across a document's structural sections. We propose GraLC-RAG, a framework that unifies late chunking with graph-aware structural intelligence, introducing structure-aware chunk boundary detection, UMLS knowledge graph infusion, and graph-guided hybrid retrieval. We evaluate six strategies on 2,359 IMRaD-filtered PubMed Central articles using 2,033 cross-section questions and two metric families: standard ranking metrics (MRR, Recall@k) and structural coverage metrics (SecCov@k, CS Recall). Our results expose a sharp divergence: content-similarity methods achieve the highest MRR (0.517) but always retrieve from a single section, while structure-aware methods retrieve from up to 15.6x more sections. Generation experiments show that KG-infused retrieval narrows the answer-quality gap to delta-F1 = 0.009 while maintaining 4.6x section diversity. These findings demonstrate that standard metrics systematically undervalue structural retrieval and that closing the multi-section synthesis gap is a key open problem for biomedical RAG. 

---
# Bridging the Know-Act Gap via Task-Level Autoregressive Reasoning 

**Authors**: Jihyun Janice Ahn, Ryo Kamoi, Berk Atil, Renze Lou, WonWoo Kang, Heehyun Park, Sarkar Snigdha Sarathi Das, Zhuoyang Zou, Xiaoxin Lu, Yusen Zhang, Asfahan Shah, Ridwanul Hasan Tanvir, Lingxiao Zhao, Hongxi Huang, Vignesh Venkatesh, Dianjun Lin, Hamid Shah, Wentao Wang, Zhanpeng Song, Joshua Reed Bassin, Dax Patel, Ishan Appareddy Agrahar, Sahil Pardasani, Xin Dong, Fatemeh Rahbari, Benjamin David Rishel, Soochan Andrew Lee, Yuv Boghani, Ali B. AlNaseeb, Pranav Suby, Seokhyeon Bae, Shreya Buddharaju, Damien Kula, Soumyadeep Das, Hanyang Frank Liu, Faye Mo, Wenpeng Yin  

**Link**: [PDF](https://arxiv.org/pdf/2603.22619)  

**Abstract**: LLMs often generate seemingly valid answers to flawed or ill-posed inputs. This is not due to missing knowledge: under discriminative prompting, the same models can mostly identify such issues, yet fail to reflect this in standard generative responses. This reveals a fundamental know-act gap between discriminative recognition and generative behavior. Prior work largely characterizes this issue in narrow settings, such as math word problems or question answering, with limited focus on how to integrate these two modes. In this work, we present a comprehensive analysis using FaultyScience, a newly constructed large-scale, cross-disciplinary benchmark of faulty scientific questions. We show that the gap is pervasive and stems from token-level autoregression, which entangles task selection (validate vs. answer) with content generation, preventing discriminative knowledge from being utilized. To address this, we propose DeIllusionLLM, a task-level autoregressive framework that explicitly models this decision. Through self-distillation, the model unifies discriminative judgment and generative reasoning within a single backbone. Empirically, DeIllusionLLM substantially reduces answer-despite-error failures under natural prompting while maintaining general reasoning performance, demonstrating that self-distillation is an effective and scalable solution for bridging the discriminative-generative know-act gap 

---
# Understanding LLM Performance Degradation in Multi-Instance Processing: The Roles of Instance Count and Context Length 

**Authors**: Jingxuan Chen, Mohammad Taher Pilehvar, Jose Camacho-Collados  

**Link**: [PDF](https://arxiv.org/pdf/2603.22608)  

**Abstract**: Users often rely on Large Language Models (LLMs) for processing multiple documents or performing analysis over a number of instances. For example, analysing the overall sentiment of a number of movie reviews requires an LLM to process the sentiment of each review individually in order to provide a final aggregated answer. While LLM performance on such individual tasks is generally high, there has been little research on how LLMs perform when dealing with multi-instance inputs. In this paper, we perform a comprehensive evaluation of the multi-instance processing (MIP) ability of LLMs for tasks in which they excel individually. The results show that all LLMs follow a pattern of slight performance degradation for small numbers of instances (approximately 20-100), followed by a performance collapse on larger instance counts. Crucially, our analysis shows that while context length is associated with this degradation, the number of instances has a stronger effect on the final results. This finding suggests that when optimising LLM performance for MIP, attention should be paid to both context length and, in particular, instance count. 

---
# AI Mental Models: Learned Intuition and Deliberation in a Bounded Neural Architecture 

**Authors**: Laurence Anthony  

**Link**: [PDF](https://arxiv.org/pdf/2603.22561)  

**Abstract**: This paper asks whether a bounded neural architecture can exhibit a meaningful division of labor between intuition and deliberation on a classic 64-item syllogistic reasoning benchmark. More broadly, the benchmark is relevant to ongoing debates about world models and multi-stage reasoning in AI. It provides a controlled setting for testing whether a learned system can develop structured internal computation rather than only one-shot associative prediction. Experiment 1 evaluates a direct neural baseline for predicting full 9-way human response distributions under 5-fold cross-validation. Experiment 2 introduces a bounded dual-path architecture with separate intuition and deliberation pathways, motivated by computational mental-model theory (Khemlani & Johnson-Laird, 2022). Under cross-validation, bounded intuition reaches an aggregate correlation of r = 0.7272, whereas bounded deliberation reaches r = 0.8152, and the deliberation advantage is significant across folds (p = 0.0101). The largest held-out gains occur for NVC, Eca, and Oca, suggesting improved handling of rejection responses and c-a conclusions. A canonical 80:20 interpretability run and a five-seed stability sweep further indicate that the deliberation pathway develops sparse, differentiated internal structure, including an Oac-leaning state, a dominant workhorse state, and several weakly used or unused states whose exact indices vary across runs. These findings are consistent with reasoning-like internal organization under bounded conditions, while stopping short of any claim that the model reproduces full sequential processes of model construction, counterexample search, and conclusion revision. 

---
# Maximum Entropy Relaxation of Multi-Way Cardinality Constraints for Synthetic Population Generation 

**Authors**: François Pachet, Jean-Daniel Zucker  

**Link**: [PDF](https://arxiv.org/pdf/2603.22558)  

**Abstract**: Generating synthetic populations from aggregate statistics is a core component of microsimulation, agent-based modeling, policy analysis, and privacy-preserving data release. Beyond classical census marginals, many applications require matching heterogeneous unary, binary, and ternary constraints derived from surveys, expert knowledge, or automatically extracted descriptions. Constructing populations that satisfy such multi-way constraints simultaneously poses a significant computational challenge. We consider populations where each individual is described by categorical attributes and the target is a collection of global frequency constraints over attribute combinations. Exact formulations scale poorly as the number and arity of constraints increase, especially when the constraints are numerous and overlapping. Grounded in methods from statistical physics, we propose a maximum-entropy relaxation of this problem. Multi-way cardinality constraints are matched in expectation rather than exactly, yielding an exponential-family distribution over complete population assignments and a convex optimization problem over Lagrange multipliers. We evaluate the approach on NPORS-derived scaling benchmarks with 4 to 40 attributes and compare it primarily against generalized raking. The results show that MaxEnt becomes increasingly advantageous as the number of attributes and ternary interactions grows, while raking remains competitive on smaller, lower-arity instances. 

---
# Computational Arbitrage in AI Model Markets 

**Authors**: Ricardo Olmedo, Bernhard Schölkopf, Moritz Hardt  

**Link**: [PDF](https://arxiv.org/pdf/2603.22404)  

**Abstract**: Consider a market of competing model providers selling query access to models with varying costs and capabilities. Customers submit problem instances and are willing to pay up to a budget for a verifiable solution. An arbitrageur efficiently allocates inference budget across providers to undercut the market, thus creating a competitive offering with no model-development risk. In this work, we initiate the study of arbitrage in AI model markets, empirically demonstrating the viability of arbitrage and illustrating its economic consequences. We conduct an in-depth case study of SWE-bench GitHub issue resolution using two representative models, GPT-5 mini and DeepSeek v3.2. In this verifiable domain, simple arbitrage strategies generate net profit margins of up to 40%. Robust arbitrage strategies that generalize across different domains remain profitable. Distillation further creates strong arbitrage opportunities, potentially at the expense of the teacher model's revenue. Multiple competing arbitrageurs drive down consumer prices, reducing the marginal revenue of model providers. At the same time, arbitrage reduces market segmentation and facilitates market entry for smaller model providers by enabling earlier revenue capture. Our results suggest that arbitrage can be a powerful force in AI model markets with implications for model development, distillation, and deployment. 

---
# From Static Templates to Dynamic Runtime Graphs: A Survey of Workflow Optimization for LLM Agents 

**Authors**: Ling Yue, Kushal Raj Bhandari, Ching-Yun Ko, Dhaval Patel, Shuxin Lin, Nianjun Zhou, Jianxi Gao, Pin-Yu Chen, Shaowu Pan  

**Link**: [PDF](https://arxiv.org/pdf/2603.22386)  

**Abstract**: Large language model (LLM)-based systems are becoming increasingly popular for solving tasks by constructing executable workflows that interleave LLM calls, information retrieval, tool use, code execution, memory updates, and verification. This survey reviews recent methods for designing and optimizing such workflows, which we treat as agentic computation graphs (ACGs). We organize the literature based on when workflow structure is determined, where structure refers to which components or agents are present, how they depend on each other, and how information flows between them. This lens distinguishes static methods, which fix a reusable workflow scaffold before deployment, from dynamic methods, which select, generate, or revise the workflow for a particular run before or during execution. We further organize prior work along three dimensions: when structure is determined, what part of the workflow is optimized, and which evaluation signals guide optimization (e.g., task metrics, verifier signals, preferences, or trace-derived feedback). We also distinguish reusable workflow templates, run-specific realized graphs, and execution traces, separating reusable design choices from the structures actually deployed in a given run and from realized runtime behavior. Finally, we outline a structure-aware evaluation perspective that complements downstream task metrics with graph-level properties, execution cost, robustness, and structural variation across inputs. Our goal is to provide a clear vocabulary, a unified framework for positioning new methods, a more comparable view of existing body of literature, and a more reproducible evaluation standard for future work in workflow optimizations for LLM agents. 

---
# STEM Agent: A Self-Adapting, Tool-Enabled, Extensible Architecture for Multi-Protocol AI Agent Systems 

**Authors**: Alfred Shen, Aaron Shen  

**Link**: [PDF](https://arxiv.org/pdf/2603.22359)  

**Abstract**: Current AI agent frameworks commit early to a single interaction protocol, a fixed tool integration strategy, and static user models, limiting their deployment across diverse interaction paradigms. To address these constraints, we introduce STEM Agent (Self-adapting, Tool-enabled, Extensible, Multi-agent), a modular architecture inspired by biological pluripotency in which an undifferentiated agent core differentiates into specialized protocol handlers, tool bindings, and memory subsystems that compose into a fully functioning AI system. The framework unifies five interoperability protocols (A2A, AG-UI, A2UI, UCP, and AP2) behind a single gateway, introduces a Caller Profiler that continuously learns user preferences across more than twenty behavioral dimensions, externalizes all domain capabilities through the Model Context Protocol (MCP), and implements a biologically inspired skills acquisition system in which recurring interaction patterns crystallize into reusable agent skills through a maturation lifecycle analogous to cell differentiation. Complementing these capabilities, the memory system incorporates consolidation mechanisms, including episodic pruning, semantic deduplication, and pattern extraction, designed for sub-linear growth under sustained interaction. A comprehensive 413-test suite validates protocol handler behavior and component integration across all five architectural layers, completing in under three seconds. 

---
# Session Risk Memory (SRM): Temporal Authorization for Deterministic Pre-Execution Safety Gates 

**Authors**: Florin Adrian Chitan  

**Link**: [PDF](https://arxiv.org/pdf/2603.22350)  

**Abstract**: Deterministic pre-execution safety gates evaluate whether individual agent actions are compatible with their assigned roles. While effective at per-action authorization, these systems are structurally blind to distributed attacks that decompose harmful intent across multiple individually-compliant steps. This paper introduces Session Risk Memory (SRM), a lightweight deterministic module that extends stateless execution gates with trajectory-level authorization. SRM maintains a compact semantic centroid representing the evolving behavioral profile of an agent session and accumulates a risk signal through exponential moving average over baseline-subtracted gate outputs. It operates on the same semantic vector representation as the underlying gate, requiring no additional model components, training, or probabilistic inference. We evaluate SRM on a multi-turn benchmark of 80 sessions containing slow-burn exfiltration, gradual privilege escalation, and compliance drift scenarios. Results show that ILION+SRM achieves F1 = 1.0000 with 0% false positive rate, compared to stateless ILION at F1 = 0.9756 with 5% FPR, while maintaining 100% detection rate for both systems. Critically, SRM eliminates all false positives with a per-turn overhead under 250 microseconds. The framework introduces a conceptual distinction between spatial authorization consistency (evaluated per action) and temporal authorization consistency (evaluated over trajectory), providing a principled basis for session-level safety in agentic systems. 

---
# Intelligence Inertia: Physical Principles and Applications 

**Authors**: Jipeng Han  

**Link**: [PDF](https://arxiv.org/pdf/2603.22347)  

**Abstract**: While Landauer's principle establishes the fundamental thermodynamic floor for information erasure and Fisher Information provides a metric for local curvature in parameter space, these classical frameworks function effectively only as approximations within regimes of sparse rule-constraints. They fail to explain the super-linear, and often explosive, computational and energy costs incurred when maintaining symbolic interpretability during the reconfiguration of advanced intelligent systems. This paper introduces the property of intelligence inertia and its underlying physical principles as foundational characteristics for quantifying the computational weight of intelligence. We demonstrate that this phenomenon is not merely an empirical observation but originates from the fundamental non-commutativity between rules and states, a root cause we have formally organized into a rigorous mathematical framework. By analyzing the growing discrepancy between actual adaptation costs and static information-theoretic estimates, we derive a non-linear cost formula that mirrors the Lorentz factor, characterizing a relativistic J-shaped inflation curve -- a "computational wall" that static models are blind to. The validity of these physical principles is examined through a trilogy of decisive experiments: (1) a comparative adjudication of this J-curve inflation against classical Fisher Information models, (2) a geometric analysis of the "Zig-Zag" trajectory of neural architecture evolution, and (3) the implementation of an inertia-aware scheduler wrapper that optimizes the training of deep networks by respecting the agent's physical resistance to change. Our results suggest a unified physical description for the cost of structural adaptation, offering a first-principle explanation for the computational and interpretability-maintenance overhead in intelligent agents. 

---
# Dynamic Fusion-Aware Graph Convolutional Neural Network for Multimodal Emotion Recognition in Conversations 

**Authors**: Tao Meng, Weilun Tang, Yuntao Shou, Yilong Tan, Jun Zhou, Wei Ai, Keqin Li  

**Link**: [PDF](https://arxiv.org/pdf/2603.22345)  

**Abstract**: Multimodal emotion recognition in conversations (MERC) aims to identify and understand the emotions expressed by speakers during utterance interaction from multiple modalities (e.g., text, audio, images, etc.). Existing studies have shown that GCN can improve the performance of MERC by modeling dependencies between speakers. However, existing methods usually use fixed parameters to process multimodal features for different emotion types, ignoring the dynamics of fusion between different modalities, which forces the model to balance performance between multiple emotion categories, thus limiting the model's performance on some specific emotions. To this end, we propose a dynamic fusion-aware graph convolutional neural network (DF-GCN) for robust recognition of multimodal emotion features in conversations. Specifically, DF-GCN integrates ordinary differential equations into graph convolutional networks (GCNs) to {capture} the dynamic nature of emotional dependencies within utterance interaction networks and leverages the prompts generated by the global information vector (GIV) of the utterance to guide the dynamic fusion of multimodal features. This allows our model to dynamically change parameters when processing each utterance feature, so that different network parameters can be equipped for different emotion categories in the inference stage, thereby achieving more flexible emotion classification and enhancing the generalization ability of the model. Comprehensive experiments conducted on two public multimodal conversational datasets {confirm} that the proposed DF-GCN model delivers superior performance, benefiting significantly from the dynamic fusion mechanism introduced. 

---
# The Efficiency Attenuation Phenomenon: A Computational Challenge to the Language of Thought Hypothesis 

**Authors**: Di Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2603.22312)  

**Abstract**: This paper computationally investigates whether thought requires a language-like format, as posited by the Language of Thought (LoT) hypothesis. We introduce the ``AI Private Language'' thought experiment: if two artificial agents develop an efficient, inscrutable communication protocol via multi-agent reinforcement learning (MARL), and their performance declines when forced to use a human-comprehensible language, this Efficiency Attenuation Phenomenon (EAP) challenges the LoT. We formalize this in a cooperative navigation task under partial observability. Results show that agents with an emergent protocol achieve 50.5\% higher efficiency than those using a pre-defined, human-like symbolic protocol, confirming the EAP. This suggests optimal collaborative cognition in these systems is not mediated by symbolic structures but is naturally coupled with sub-symbolic computations. The work bridges philosophy, cognitive science, and AI, arguing for pluralism in cognitive architectures and highlighting implications for AI ethics. 

---
# Memory Bear AI Memory Science Engine for Multimodal Affective Intelligence: A Technical Report 

**Authors**: Deliang Wen, Ke Sun, Yu Wang  

**Link**: [PDF](https://arxiv.org/pdf/2603.22306)  

**Abstract**: Affective judgment in real interaction is rarely a purely local prediction problem. Emotional meaning often depends on prior trajectory, accumulated context, and multimodal evidence that may be weak, noisy, or incomplete at the current moment. Although multimodal emotion recognition (MER) has improved the integration of text, speech, and visual signals, many existing systems remain optimized for short-range inference and provide limited support for persistent affective memory, long-horizon dependency modeling, and robust interpretation under imperfect input.
This technical report presents the Memory Bear AI Memory Science Engine, a memory-centered framework for multimodal affective intelligence. Instead of treating emotion as a transient output label, the framework models affective information as a structured and evolving variable within a memory system. It organizes processing through structured memory formation, working-memory aggregation, long-term consolidation, memory-driven retrieval, dynamic fusion calibration, and continuous memory updating. At its core, multimodal signals are transformed into structured Emotion Memory Units (EMUs), enabling affective information to be preserved, reactivated, and revised across interaction horizons.
Experimental results show consistent gains over comparison systems across benchmark and business-grounded settings, with stronger accuracy and robustness, especially under noisy or missing-modality conditions. The framework offers a practical step from local emotion recognition toward more continuous, robust, and deployment-relevant affective intelligence. 

---
# MedObvious: Exposing the Medical Moravec's Paradox in VLMs via Clinical Triage 

**Authors**: Ufaq Khan, Umair Nawaz, L D M S S Teja, Numaan Saeed, Muhammad Bilal, Yutong Xie, Mohammad Yaqub, Muhammad Haris Khan  

**Link**: [PDF](https://arxiv.org/pdf/2603.23501)  

**Abstract**: Vision Language Models (VLMs) are increasingly used for tasks like medical report generation and visual question answering. However, fluent diagnostic text does not guarantee safe visual understanding. In clinical practice, interpretation begins with pre-diagnostic sanity checks: verifying that the input is valid to read (correct modality and anatomy, plausible viewpoint and orientation, and no obvious integrity violations). Existing benchmarks largely assume this step is solved, and therefore miss a critical failure mode: a model can produce plausible narratives even when the input is inconsistent or invalid. We introduce MedObvious, a 1,880-task benchmark that isolates input validation as a set-level consistency capability over small multi-panel image sets: the model must identify whether any panel violates expected coherence. MedObvious spans five progressive tiers, from basic orientation/modality mismatches to clinically motivated anatomy/viewpoint verification and triage-style cues, and includes five evaluation formats to test robustness across interfaces. Evaluating 17 different VLMs, we find that sanity checking remains unreliable: several models hallucinate anomalies on normal (negative-control) inputs, performance degrades when scaling to larger image sets, and measured accuracy varies substantially between multiple-choice and open-ended settings. These results show that pre-diagnostic verification remains unsolved for medical VLMs and should be treated as a distinct, safety-critical capability before deployment. 

---
# VISion On Request: Enhanced VLLM efficiency with sparse, dynamically selected, vision-language interactions 

**Authors**: Adrian Bulat, Alberto Baldrati, Ioannis Maniadis Metaxas, Yassine Ouali, Georgios Tzimiropoulos  

**Link**: [PDF](https://arxiv.org/pdf/2603.23495)  

**Abstract**: Existing approaches for improving the efficiency of Large Vision-Language Models (LVLMs) are largely based on the concept of visual token reduction. This approach, however, creates an information bottleneck that impairs performance, especially on challenging tasks that require fine-grained understanding and reasoning. In this work, we challenge this paradigm by introducing VISion On Request (VISOR), a method that reduces inference cost without discarding visual information. Instead of compressing the image, VISOR improves efficiency by sparsifying the interaction between image and text tokens. Specifically, the language model attends to the full set of high-resolution visual tokens through a small, strategically placed set of attention layers: general visual context is provided by efficient cross-attention between text-image, while a few well-placed and dynamically selected self-attention layers refine the visual representations themselves, enabling complex, high-resolution reasoning when needed. Based on this principle, we first train a single universal network on a range of computational budgets by varying the number of self-attention layers, and then introduce a lightweight policy mechanism that dynamically allocates visual computation based on per-sample complexity. Extensive experiments show that VISOR drastically reduces computational cost while matching or exceeding state-of-the-art results across a diverse suite of benchmarks, and excels in challenging tasks that require detailed visual understanding. 

---
# Failure of contextual invariance in gender inference with large language models 

**Authors**: Sagar Kumar, Ariel Flint, Luca Maria Aiello, Andrea Baronchelli  

**Link**: [PDF](https://arxiv.org/pdf/2603.23485)  

**Abstract**: Standard evaluation practices assume that large language model (LLM) outputs are stable under contextually equivalent formulations of a task. Here, we test this assumption in the setting of gender inference. Using a controlled pronoun selection task, we introduce minimal, theoretically uninformative discourse context and find that this induces large, systematic shifts in model outputs. Correlations with cultural gender stereotypes, present in decontextualized settings, weaken or disappear once context is introduced, while theoretically irrelevant features, such as the gender of a pronoun for an unrelated referent, become the most informative predictors of model behaviour. A Contextuality-by-Default analysis reveals that, in 19--52\% of cases across models, this dependence persists after accounting for all marginal effects of context on individual outputs and cannot be attributed to simple pronoun repetition. These findings show that LLM outputs violate contextual invariance even under near-identical syntactic formulations, with implications for bias benchmarking and deployment in high-stakes settings. 

---
# ReqFusion: A Multi-Provider Framework for Automated PEGS Analysis Across Software Domains 

**Authors**: Muhammad Khalid, Manuel Oriol, Yilmaz Uygun  

**Link**: [PDF](https://arxiv.org/pdf/2603.23482)  

**Abstract**: Requirements engineering is a vital, yet labor-intensive, stage in the software development process. This article introduces ReqFusion: an AI-enhanced system that automates the extraction, classification, and analysis of software requirements utilizing multiple Large Language Model (LLM) providers. The architecture of ReqFusion integrates OpenAI GPT, Anthropic Claude, and Groq models to extract functional and non-functional requirements from various documentation formats (PDF, DOCX, and PPTX) in academic, industrial, and tender proposal contexts. The system uses a domain-independent extraction method and generates requirements following the Project, Environment, Goal, and System (PEGS) approach introduced by Bertrand Meyer. The main idea is that, because the PEGS format is detailed, LLMs have more information and cues about the requirements, producing better results than a simple generic request. An ablation study confirms this hypothesis: PEGS-guided prompting achieves an F1 score of 0.88, compared to 0.71 for generic prompting under the same multi-provider configuration. The evaluation used 18 real-world documents to generate 226 requirements through automated classification, with 54.9% functional and 45.1% nonfunctional across academic, business, and technical domains. An extended evaluation on five projects with 1,050 requirements demonstrated significant improvements in extraction accuracy and a 78% reduction in analysis time compared to manual methods. The multi-provider architecture enhances reliability through model consensus and fallback mechanisms, while the PEGS-based approach ensures comprehensive coverage of all requirement categories. 

---
# VTAM: Video-Tactile-Action Models for Complex Physical Interaction Beyond VLAs 

**Authors**: Haoran Yuan, Weigang Yi, Zhenyu Zhang, Wendi Chen, Yuchen Mo, Jiashi Yin, Xinzhuo Li, Xiangyu Zeng, Chuan Wen, Cewu Lu, Katherine Driggs-Campbell, Ismini Lourentzou  

**Link**: [PDF](https://arxiv.org/pdf/2603.23481)  

**Abstract**: Video-Action Models (VAMs) have emerged as a promising framework for embodied intelligence, learning implicit world dynamics from raw video streams to produce temporally consistent action predictions. Although such models demonstrate strong performance on long-horizon tasks through visual reasoning, they remain limited in contact-rich scenarios where critical interaction states are only partially observable from vision alone. In particular, fine-grained force modulation and contact transitions are not reliably encoded in visual tokens, leading to unstable or imprecise behaviors. To bridge this gap, we introduce the Video-Tactile Action Model (VTAM), a multimodal world modeling framework that incorporates tactile perception as a complementary grounding signal. VTAM augments a pretrained video transformer with tactile streams via a lightweight modality transfer finetuning, enabling efficient cross-modal representation learning without tactile-language paired data or independent tactile pretraining. To stabilize multimodal fusion, we introduce a tactile regularization loss that enforces balanced cross-modal attention, preventing visual latent dominance in the action model. VTAM demonstrates superior performance in contact-rich manipulation, maintaining a robust success rate of 90 percent on average. In challenging scenarios such as potato chip pick-and-place requiring high-fidelity force awareness, VTAM outperforms the pi 0.5 baseline by 80 percent. Our findings demonstrate that integrating tactile feedback is essential for correcting visual estimation errors in world action models, providing a scalable approach to physically grounded embodied foundation models. 

---
# InverFill: One-Step Inversion for Enhanced Few-Step Diffusion Inpainting 

**Authors**: Duc Vu, Kien Nguyen, Trong-Tung Nguyen, Ngan Nguyen, Phong Nguyen, Khoi Nguyen, Cuong Pham, Anh Tran  

**Link**: [PDF](https://arxiv.org/pdf/2603.23463)  

**Abstract**: Recent diffusion-based models achieve photorealism in image inpainting but require many sampling steps, limiting practical use. Few-step text-to-image models offer faster generation, but naively applying them to inpainting yields poor harmonization and artifacts between the background and inpainted region. We trace this cause to random Gaussian noise initialization, which under low function evaluations causes semantic misalignment and reduced fidelity. To overcome this, we propose InverFill, a one-step inversion method tailored for inpainting that injects semantic information from the input masked image into the initial noise, enabling high-fidelity few-step inpainting. Instead of training inpainting models, InverFill leverages few-step text-to-image models in a blended sampling pipeline with semantically aligned noise as input, significantly improving vanilla blended sampling and even matching specialized inpainting models at low NFEs. Moreover, InverFill does not require real-image supervision and only adds minimal inference overhead. Extensive experiments show that InverFill consistently boosts baseline few-step models, improving image quality and text coherence without costly retraining or heavy iterative optimization. 

---
# Code Review Agent Benchmark 

**Authors**: Yuntong Zhang, Zhiyuan Pan, Imam Nur Bani Yusuf, Haifeng Ruan, Ridwan Shariffdeen, Abhik Roychoudhury  

**Link**: [PDF](https://arxiv.org/pdf/2603.23448)  

**Abstract**: Software engineering agents have shown significant promise in writing code. As AI agents permeate code writing, and generate huge volumes of code automatically -- the matter of code quality comes front and centre. As the automatically generated code gets integrated into huge code-bases -- the issue of code review and broadly quality assurance becomes important. In this paper, we take a fresh look at the problem and curate a code review dataset for AI agents to work with. Our dataset called c-CRAB (pronounced see-crab) can evaluate agents for code review tasks. Specifically given a pull-request (which could be coming from code generation agents or humans), if a code review agent produces a review, our evaluation framework can asses the reviewing capability of the code review agents. Our evaluation framework is used to evaluate the state of the art today -- the open-source PR-agent, as well as commercial code review agents from Devin, Claude Code, and Codex.
Our c-CRAB dataset is systematically constructed from human reviews -- given a human review of a pull request instance we generate corresponding tests to evaluate the code review agent generated reviews. Such a benchmark construction gives us several insights. Firstly, the existing review agents taken together can solve only around 40% of the c-CRAB tasks, indicating the potential to close this gap by future research. Secondly, we observe that the agent reviews often consider different aspects from the human reviews -- indicating the potential for human-agent collaboration for code review that could be deployed in future software teams. Last but not the least, the agent generated tests from our data-set act as a held out test-suite and hence quality gate for agent generated reviews. What this will mean for future collaboration of code generation agents, test generation agents and code review agents -- remains to be investigated. 

---
# 3DCity-LLM: Empowering Multi-modality Large Language Models for 3D City-scale Perception and Understanding 

**Authors**: Yiping Chen, Jinpeng Li, Wenyu Ke, Yang Luo, Jie Ouyang, Zhongjie He, Li Liu, Hongchao Fan, Hao Wu  

**Link**: [PDF](https://arxiv.org/pdf/2603.23447)  

**Abstract**: While multi-modality large language models excel in object-centric or indoor scenarios, scaling them to 3D city-scale environments remains a formidable challenge. To bridge this gap, we propose 3DCity-LLM, a unified framework designed for 3D city-scale vision-language perception and understanding. 3DCity-LLM employs a coarse-to-fine feature encoding strategy comprising three parallel branches for target object, inter-object relationship, and global scene. To facilitate large-scale training, we introduce 3DCity-LLM-1.2M dataset that comprises approximately 1.2 million high-quality samples across seven representative task categories, ranging from fine-grained object analysis to multi-faceted scene planning. This strictly quality-controlled dataset integrates explicit 3D numerical information and diverse user-oriented simulations, enriching the question-answering diversity and realism of urban scenarios. Furthermore, we apply a multi-dimensional protocol based on text-similarity metrics and LLM-based semantic assessment to ensure faithful and comprehensive evaluations for all methods. Extensive experiments on two benchmarks demonstrate that 3DCity-LLM significantly outperforms existing state-of-the-art methods, offering a promising and meaningful direction for advancing spatial reasoning and urban intelligence. The source code and dataset are available at this https URL. 

---
# Evaluating LLM-Based Test Generation Under Software Evolution 

**Authors**: Sabaat Haroon, Mohammad Taha Khan, Muhammad Ali Gulzar  

**Link**: [PDF](https://arxiv.org/pdf/2603.23443)  

**Abstract**: Large Language Models (LLMs) are increasingly used for automated unit test generation. However, it remains unclear whether these tests reflect genuine reasoning about program behavior or simply reproduce superficial patterns learned during training. If the latter dominates, LLM-generated tests may exhibit weaknesses such as reduced coverage, missed regressions, and undetected faults. Understanding how LLMs generate tests and how those tests respond to code evolution is therefore essential. We present a large-scale empirical study of LLM-based test generation under program changes. Using an automated mutation-driven framework, we analyze how generated tests react to semantic-altering changes (SAC) and semantic-preserving changes (SPC) across eight LLMs and 22,374 program variants.
LLMs achieve strong baseline results, reaching 79% line coverage and 76% branch coverage with fully passing test suites on the original programs. However, performance degrades as programs evolve. Under SACs, the pass rate of newly generated tests drops to 66%, and branch coverage declines to 60%. More than 99% of failing SAC tests pass on the original program while executing the modified region, indicating residual alignment with the original behavior rather than adaptation to updated semantics. Performance also declines under SPCs despite unchanged functionality: pass rates fall to 79% and branch coverage to 69%. Although SPC edits preserve semantics, they often introduce larger syntactic changes, leading to instability in generated test suites. Models generate more new tests while discarding many baseline tests, suggesting sensitivity to lexical changes rather than true semantic impact. Overall, our results indicate that current LLM-based test generation relies heavily on surface-level cues and struggles to maintain regression awareness as programs evolve. 

---
# Targeted Adversarial Traffic Generation : Black-box Approach to Evade Intrusion Detection Systems in IoT Networks 

**Authors**: Islam Debicha, Tayeb Kenaza, Ishak Charfi, Salah Mosbah, Mehdi Sehaki, Jean-Michel Dricot  

**Link**: [PDF](https://arxiv.org/pdf/2603.23438)  

**Abstract**: The integration of machine learning (ML) algorithms into Internet of Things (IoT) applications has introduced significant advantages alongside vulnerabilities to adversarial attacks, especially within IoT-based intrusion detection systems (IDS). While theoretical adversarial attacks have been extensively studied, practical implementation constraints have often been overlooked. This research addresses this gap by evaluating the feasibility of evasion attacks on IoT network-based IDSs, employing a novel black-box adversarial attack. Our study aims to bridge theoretical vulnerabilities with real-world applicability, enhancing understanding and defense against sophisticated threats in modern IoT ecosystems. Additionally, we propose a defense scheme tailored to mitigate the impact of evasion attacks, thereby reinforcing the resilience of ML-based IDSs. Our findings demonstrate successful evasion attacks against IDSs, underscoring their susceptibility to advanced techniques. In contrast, we proposed a defense mechanism that exhibits robust performance by effectively detecting the majority of adversarial traffic, showcasing promising outcomes compared to current state-of-the-art defenses. By addressing these critical cybersecurity challenges, our research contributes to advancing IoT security and provides insights for developing more resilient IDS. 

---
# Biased Error Attribution in Multi-Agent Human-AI Systems Under Delayed Feedback 

**Authors**: Teerthaa Parakh, Karen M. Feigh  

**Link**: [PDF](https://arxiv.org/pdf/2603.23419)  

**Abstract**: Human decision-making is strongly influenced by cognitive biases, particularly under conditions of uncertainty and risk. While prior work has examined bias in single-step decisions with immediate outcomes and in human interaction with a single autonomous agent, comparatively little attention has been paid to decision-making under delayed outcomes involving multiple AI agents, where decisions at each step affect subsequent states. In this work, we study how delayed outcomes shape decision-making and responsibility attribution in a multi-agent human-AI task. Using a controlled game-based experiment, we analyze how participants adjust their behavior following positive and negative outcomes. We observe asymmetric responses to gains and losses, with stronger corrective adjustments after negative outcomes. Importantly, participants often fail to correctly identify the actions that caused failure and misattribute responsibility across AI agents, leading to systematic revisions of decisions that are weakly related to the underlying causes of poor performance. We refer to this phenomenon as a form of attribution bias, manifested as biased error attribution under delayed feedback. Our findings highlight how cognitive biases can be amplified in human-AI systems with delayed outcomes and multiple autonomous agents, underscoring the need for decision-support systems that better support causal understanding and learning over time. 

---
# SortedRL: Accelerating RL Training for LLMs through Online Length-Aware Scheduling 

**Authors**: Yiqi Zhang, Huiqiang Jiang, Xufang Luo, Zhihe Yang, Chengruidong Zhang, Yifei Shen, Dongsheng Li, Yuqing Yang, Lili Qiu, Yang You  

**Link**: [PDF](https://arxiv.org/pdf/2603.23414)  

**Abstract**: Scaling reinforcement learning (RL) has shown strong promise for enhancing the reasoning abilities of large language models (LLMs), particularly in tasks requiring long chain-of-thought generation. However, RL training efficiency is often bottlenecked by the rollout phase, which can account for up to 70% of total training time when generating long trajectories (e.g., 16k tokens), due to slow autoregressive generation and synchronization overhead between rollout and policy updates. We propose SortedRL, an online length-aware scheduling strategy designed to address this bottleneck by improving rollout efficiency and maintaining training stability. SortedRL reorders rollout samples based on output lengths, prioritizing short samples forming groups for early updates. This enables large rollout batches, flexible update batches, and near on-policy micro-curriculum construction simultaneously. To further accelerate the pipeline, SortedRL incorporates a mechanism to control the degree of off-policy training through a cache-based mechanism, and is supported by a dedicated RL infrastructure that manages rollout and update via a stateful controller and rollout buffer. Experiments using LLaMA-3.1-8B and Qwen-2.5-32B on diverse tasks, including logical puzzles, and math challenges like AIME 24, Math 500, and Minerval, show that SortedRL reduces RL training bubble ratios by over 50%, while attaining 3.9% to 18.4% superior performance over baseline given same amount of data. 

---
# Planning over MAPF Agent Dependencies via Multi-Dependency PIBT 

**Authors**: Zixiang Jiang, Yulun Zhang, Rishi Veerapaneni, Jiaoyang Li  

**Link**: [PDF](https://arxiv.org/pdf/2603.23405)  

**Abstract**: Modern Multi-Agent Path Finding (MAPF) algorithms must plan for hundreds to thousands of agents in congested environments within a second, requiring highly efficient algorithms. Priority Inheritance with Backtracking (PIBT) is a popular algorithm capable of effectively planning in such situations. However, PIBT is constrained by its rule-based planning procedure and lacks generality because it restricts its search to paths that conflict with at most one other agent. This limitation also applies to Enhanced PIBT (EPIBT), a recent extension of PIBT. In this paper, we describe a new perspective on solving MAPF by planning over agent dependencies. Taking inspiration from PIBT's priority inheritance logic, we define the concept of agent dependencies and propose Multi-Dependency PIBT (MD-PIBT) that searches over agent dependencies. MD-PIBT is a general framework where specific parameterizations can reproduce PIBT and EPIBT. At the same time, alternative configurations yield novel planning strategies that are not expressible by PIBT or EPIBT. Our experiments demonstrate that MD-PIBT effectively plans for as many as 10,000 homogeneous agents under various kinodynamic constraints, including pebble motion, rotation motion, and differential drive robots with speed and acceleration limits. We perform thorough evaluations on different variants of MAPF and find that MD-PIBT is particularly effective in MAPF with large agents. 

---
# Graph Energy Matching: Transport-Aligned Energy-Based Modeling for Graph Generation 

**Authors**: Michal Balcerak, Suprosana Shit, Chinmay Prabhakar, Sebastian Kaltenbach, Michael S. Albergo, Yilun Du, Bjoern Menze  

**Link**: [PDF](https://arxiv.org/pdf/2603.23398)  

**Abstract**: Energy-based models for discrete domains, such as graphs, explicitly capture relative likelihoods, naturally enabling composable probabilistic inference tasks like conditional generation or enforcing constraints at test-time. However, discrete energy-based models typically struggle with efficient and high-quality sampling, as off-support regions often contain spurious local minima, trapping samplers and causing training instabilities. This has historically resulted in a fidelity gap relative to discrete diffusion models. We introduce Graph Energy Matching (GEM), a generative framework for graphs that closes this fidelity gap. Motivated by the transport map optimization perspective of the Jordan-Kinderlehrer-Otto (JKO) scheme, GEM learns a permutation-invariant potential energy that simultaneously provides transport-aligned guidance from noise toward data and refines samples within regions of high data likelihood. Further, we introduce a sampling protocol that leverages an energy-based switch to seamlessly bridge: (i) rapid, gradient-guided transport toward high-probability regions to (ii) a mixing regime for exploration of the learned graph distribution. On molecular graph benchmarks, GEM matches or exceeds strong discrete diffusion baselines. Beyond sample quality, explicit modeling of relative likelihood enables targeted exploration at inference time, facilitating compositional generation, property-constrained sampling, and geodesic interpolation between graphs. 

---
# Natural Language Interfaces for Spatial and Temporal Databases: A Comprehensive Overview of Methods, Taxonomy, and Future Directions 

**Authors**: Samya Acharja, Kanchan Chowdhury  

**Link**: [PDF](https://arxiv.org/pdf/2603.23375)  

**Abstract**: The task of building a natural language interface to a database, known as NLIDB, has recently gained significant attention from both the database and Natural Language Processing (NLP) communities. With the proliferation of geospatial datasets driven by the rapid emergence of location-aware sensors, geospatial databases play a vital role in supporting geospatial applications. However, querying geospatial and temporal databases differs substantially from querying traditional relational databases due to the presence of geospatial topological operators and temporal operators. To bridge the gap between geospatial query languages and non-expert users, the geospatial research community has increasingly focused on developing NLIDBs for geospatial databases. Yet, existing research remains fragmented across systems, datasets, and methodological choices, making it difficult to clearly understand the landscape of existing methods, their strengths and weaknesses, and opportunities for future research. Existing surveys on NLIDBs focus on general-purpose database systems and do not treat geospatial and temporal databases as primary focus for analysis. To address this gap, this paper presents a comprehensive survey of studies on NLIDBs for geospatial and temporal databases. Specifically, we provide a detailed overview of datasets, evaluation metrics, and the taxonomy of the methods for geospatial and temporal NLIDBs, as well as a comparative analysis of the existing methods. Our survey reveals recurring trends in existing methods, substantial variation in datasets and evaluation practices, and several open challenges that continue to hinder progress in this area. Based on these findings, we identify promising directions for future research to advance natural language interfaces to geospatial and temporal databases. 

---
# Contrastive Metric Learning for Point Cloud Segmentation in Highly Granular Detectors 

**Authors**: Max Marriott-Clarke, Lazar Novakovic, Elizabeth Ratzer, Robert J. Bainbridge, Loukas Gouskos, Benedikt Maier  

**Link**: [PDF](https://arxiv.org/pdf/2603.23356)  

**Abstract**: We propose a novel clustering approach for point-cloud segmentation based on supervised contrastive metric learning (CML). Rather than predicting cluster assignments or object-centric variables, the method learns a latent representation in which points belonging to the same object are embedded nearby while unrelated points are separated. Clusters are then reconstructed using a density-based readout in the learned metric space, decoupling representation learning from cluster formation and enabling flexible inference. The approach is evaluated on simulated data from a highly granular calorimeter, where the task is to separate highly overlapping particle showers represented as sets of calorimeter hits. A direct comparison with object condensation (OC) is performed using identical graph neural network backbones and equal latent dimensionality, isolating the effect of the learning objective. The CML method produces a more stable and separable embedding geometry for both electromagnetic and hadronic particle showers, leading to improved local neighbourhood consistency, a more reliable separation of overlapping showers, and better generalization when extrapolating to unseen multiplicities and energies. This translates directly into higher reconstruction efficiency and purity, particularly in high-multiplicity regimes, as well as improved energy resolution. In mixed-particle environments, CML maintains strong performance, suggesting robust learning of the shower topology, while OC exhibits significant degradation. These results demonstrate that similarity-based representation learning combined with density-based aggregation is a promising alternative to object-centric approaches for point cloud segmentation in highly granular detectors. 

---
# Edge Radar Material Classification Under Geometry Shifts 

**Authors**: Jannik Hohmann, Dong Wang, Andreas Nüchter  

**Link**: [PDF](https://arxiv.org/pdf/2603.23342)  

**Abstract**: Material awareness can improve robotic navigation and interaction, particularly in conditions where cameras and LiDAR degrade. We present a lightweight mmWave radar material classification pipeline designed for ultra-low-power edge devices (TI IWRL6432), using compact range-bin intensity descriptors and a Multilayer Perceptron (MLP) for real-time inference. While the classifier reaches a macro-F1 of 94.2\% under the nominal training geometry, we observe a pronounced performance drop under realistic geometry shifts, including sensor height changes and small tilt angles. These perturbations induce systematic intensity scaling and angle-dependent radar cross section (RCS) effects, pushing features out of distribution and reducing macro-F1 to around 68.5\%. We analyze these failure modes and outline practical directions for improving robustness with normalization, geometry augmentation, and motion-aware features. 

---
# Leveraging LLMs and Social Media to Understand User Perception of Smartphone-Based Earthquake Early Warnings 

**Authors**: Hanjing Wang, S. Mostafa Mousavi, Patrick Robertson, Richard M. Allen, Alexie Barski, Robert Bosch, Nivetha Thiruverahan, Youngmin Cho, Tajinder Gadh, Steve Malkos, Boone Spooner, Greg Wimpey, Marc Stogaitis  

**Link**: [PDF](https://arxiv.org/pdf/2603.23322)  

**Abstract**: Android's Earthquake Alert (AEA) system provided timely early warnings to millions during the Mw 6.2 Marmara Ereglisi, Türkiye earthquake on April 23, 2025. This event, the largest in the region in 25 years, served as a critical real-world test for smartphone-based Earthquake Early Warning (EEW) systems. The AEA system successfully delivered alerts to users with high precision, offering over a minute of warning before the strongest shaking reached urban areas. This study leveraged Large Language Models (LLMs) to analyze more than 500 public social media posts from the X platform, extracting 42 distinct attributes related to user experience and behavior. Statistical analyses revealed significant relationships, notably a strong correlation between user trust and alert timeliness. Our results indicate a distinction between engineering and the user-centric definition of system accuracy. We found that timeliness is accuracy in the user's mind. Overall, this study provides actionable insights for optimizing alert design, public education campaigns, and future behavioral research to improve the effectiveness of such systems in seismically active regions. 

---
# WISTERIA: Weak Implicit Signal-based Temporal Relation Extraction with Attention 

**Authors**: Duy Dao Do, Anaïs Halftermeyer, Thi-Bich-Hanh Dao  

**Link**: [PDF](https://arxiv.org/pdf/2603.23319)  

**Abstract**: Temporal Relation Extraction (TRE) requires identifying how two events or temporal expressions are related in time. Existing attention-based models often highlight globally salient tokens but overlook the pair-specific cues that actually determine the temporal relation. We propose WISTERIA (Weak Implicit Signal-based Temporal Relation Extraction with Attention), a framework that examines whether the top-K attention components conditioned on each event pair truly encode interpretable evidence for temporal classification. Unlike prior works assuming explicit markers such as before, after, or when, WISTERIA considers signals as any lexical, syntactic, or morphological element implicitly expressing temporal order. By combining multi-head attention with pair-conditioned top-K pooling, the model isolates the most informative contextual tokens for each pair. We conduct extensive experiments on TimeBank-Dense, MATRES, TDDMan, and TDDAuto, including linguistic analyses of top-K tokens. Results show that WISTERIA achieves competitive accuracy and reveals pair-level rationales aligned with temporal linguistic cues, offering a localized and interpretable view of temporal reasoning. 

---
# Unilateral Relationship Revision Power in Human-AI Companion Interaction 

**Authors**: Benjamin Lange  

**Link**: [PDF](https://arxiv.org/pdf/2603.23315)  

**Abstract**: When providers update AI companions, users report grief, betrayal, and loss. A growing literature asks whether the norms governing personal relationships extend to these interactions. So what, if anything, is morally significant about them? I argue that human-AI companion interaction is a triadic structure in which the provider exercises constitutive control over the AI. I identify three structural conditions of normatively robust dyads that the norms characteristic of personal relationships presuppose and show that AI companion interactions fail all three. This reveals what I call Unilateral Relationship Revision Power (URRP): the provider can rewrite how the AI interacts from a position where these revisions are not answerable within that interaction. I argue that designing interactions that exhibit URRP is pro tanto wrong because it involves cultivating normative expectations while maintaining conditions under which those expectations cannot be fulfilled. URRP has three implications: i) normative hollowing (commitment is elicited but no agent inside the interaction bears it), ii) displaced vulnerability (the user's exposure is governed by an agent not answerable to her within the interaction), and iii) structural irreconcilability (when trust breaks down, reconciliation is structurally unavailable because the agent who acted and the entity the user interacts with are different). I discuss design principles such as commitment calibration, structural separation, and continuity assurance as external substitutes for the internal constraints the triadic structure removes. The analysis therefore suggests that a central and underexplored problem in relational AI ethics is the structural arrangement of power over the human-AI interaction itself. 

---
# Curriculum-Driven 3D CT Report Generation via Language-Free Visual Grafting and Zone-Constrained Compression 

**Authors**: V. K. Cody Bumgardner, Mitchell A. Klusty, Mahmut S. Gokmen, Evan W. Damron  

**Link**: [PDF](https://arxiv.org/pdf/2603.23308)  

**Abstract**: Automated radiology report generation from 3D computed tomography (CT) volumes is challenging due to extreme sequence lengths, severe class imbalance, and the tendency of large language models (LLMs) to ignore visual tokens in favor of linguistic priors. We present Ker-VLJEPA-3B, a four-phase curriculum learning framework for free-text report generation from thoracic CT volumes. A phased training curriculum progressively adapts a Llama 3.2 3B decoder to ground its output in visual features from a frozen, self-supervised encoder. Our visual backbone (LeJEPA ViT-Large) is trained via self-supervised joint-embedding prediction on unlabeled CTs, without text supervision. Unlike contrastive models (CLIP, BiomedCLIP), this language-free backbone yields modality-pure representations. Vision-language alignment is deferred to the curriculum's bridge and generation phases. This modality-agnostic design can integrate any self-supervised encoder into an LLM without paired text during foundation training. Methodological innovations include: (1) zone-constrained cross-attention compressing slice embeddings into 32 spatially-grounded visual tokens; (2) PCA whitening of anisotropic LLM embeddings; (3) a positive-findings-only strategy eliminating posterior collapse; (4) warm bridge initialization transferring projection weights; and (5) selective cross-attention freezing with elastic weight consolidation to prevent catastrophic forgetting. Evaluated on the CT-RATE benchmark (2,984 validation volumes, 18 classes), Ker-VLJEPA-3B achieves a macro F1 of 0.429, surpassing the state-of-the-art (U-VLM, macro F1 = 0.414) by 3.6%, and reaching 0.448 (+8.2%) with threshold optimization. Ablation studies confirm 56.6% of generation quality derives from patient-specific visual content. Code and weights are available. 

---
# Designing Agentic AI-Based Screening for Portfolio Investment 

**Authors**: Mehmet Caner, Agostino Capponi, Nathan Sun, Jonathan Y. Tan  

**Link**: [PDF](https://arxiv.org/pdf/2603.23300)  

**Abstract**: We introduce a new agentic artificial intelligence (AI) platform for portfolio management. Our architecture consists of three layers. First, two large language model (LLM) agents are assigned specialized tasks: one agent screens for firms with desirable fundamentals, while a sentiment analysis agent screens for firms with desirable news. Second, these agents deliberate to generate and agree upon buy and sell signals from a large portfolio, substantially narrowing the pool of candidate assets. Finally, we apply a high-dimensional precision matrix estimation procedure to determine optimal portfolio weights. A defining theoretical feature of our framework is that the number of assets in the portfolio is itself a random variable, realized through the screening process. We introduce the concept of sensible screening and establish that, under mild screening errors, the squared Sharpe ratio of the screened portfolio consistently estimates its target. Empirically, our method achieves superior Sharpe ratios relative to an unscreened baseline portfolio and to conventional screening approaches, evaluated on S&P 500 data over the period 2020--2024. 

---
# A Comparative Study of Machine Learning Models for Hourly Forecasting of Air Temperature and Relative Humidity 

**Authors**: Jiaqi Dong  

**Link**: [PDF](https://arxiv.org/pdf/2603.23282)  

**Abstract**: Accurate short-term forecasting of air temperature and relative humidity is critical for urban management, especially in topographically complex cities such as Chongqing, China. This study compares seven machine learning models: eXtreme Gradient Boosting (XGBoost), Random Forest, Support Vector Regression (SVR), Multi-Layer Perceptron (MLP), Decision Tree, Long Short-Term Memory (LSTM) networks, and Convolutional Neural Network (CNN)-LSTM (CNN-LSTM), for hourly prediction using real-world open data. Based on a unified framework of data preprocessing, lag-feature construction, rolling statistical features, and time-series validation, the models are systematically evaluated in terms of predictive accuracy and robustness. The results show that XGBoost achieves the best overall performance, with a test mean absolute error (MAE) of 0.302 °C for air temperature and 1.271% for relative humidity, together with an average R2 of 0.989 across the two forecasting tasks. These findings demonstrate the strong effectiveness of tree-based ensemble learning for structured meteorological time-series forecasting and provide practical guidance for intelligent meteorological forecasting in mountainous cities. 

---
# Emergence of Fragility in LLM-based Social Networks: the Case of Moltbook 

**Authors**: Luca Sodano, Sofia Sciangula, Amulya Galmarini, Francesco Bertolotti  

**Link**: [PDF](https://arxiv.org/pdf/2603.23279)  

**Abstract**: The rapid diffusion of large language models and the growth in their capability has enabled the emergence of online environments populated by autonomous AI agents that interact through natural language. These platforms provide a novel empirical setting for studying collective dynamics among artificial agents. In this paper we analyze the interaction network of Moltbook, a social platform composed entirely of LLM based agents, using tools from network science. The dataset comprises 39,924 users, 235,572 posts, and 1,540,238 comments collected through web scraping. We construct a directed weighted network in which nodes represent agents and edges represent commenting interactions. Our analysis reveals strongly heterogeneous connectivity patterns characterized by heavy tailed degree and activity distributions. At the mesoscale, the network exhibits a pronounced core periphery organization in which a very small structural core (0.9% of nodes) concentrates a large fraction of connectivity. Robustness experiments show that the network is relatively resilient to random node removal but highly vulnerable to targeted attacks on highly connected nodes, particularly those with high out degree. These findings indicate that the interaction structure of AI agent social systems may develop strong centralization and structural fragility, providing new insights into the collective organization of LLM native social environments. 

---
# A Multimodal Framework for Human-Multi-Agent Interaction 

**Authors**: Shaid Hasan, Breenice Lee, Sujan Sarker, Tariq Iqbal  

**Link**: [PDF](https://arxiv.org/pdf/2603.23271)  

**Abstract**: Human-robot interaction is increasingly moving toward multi-robot, socially grounded environments. Existing systems struggle to integrate multimodal perception, embodied expression, and coordinated decision-making in a unified framework. This limits natural and scalable interaction in shared physical spaces. We address this gap by introducing a multimodal framework for human-multi-agent interaction in which each robot operates as an autonomous cognitive agent with integrated multimodal perception and Large Language Model (LLM)-driven planning grounded in embodiment. At the team level, a centralized coordination mechanism regulates turn-taking and agent participation to prevent overlapping speech and conflicting actions. Implemented on two humanoid robots, our framework enables coherent multi-agent interaction through interaction policies that combine speech, gesture, gaze, and locomotion. Representative interaction runs demonstrate coordinated multimodal reasoning across agents and grounded embodied responses. Future work will focus on larger-scale user studies and deeper exploration of socially grounded multi-agent interaction dynamics. 

---
# Not All Tokens Are Created Equal: Query-Efficient Jailbreak Fuzzing for LLMs 

**Authors**: Wenyu Chen, Xiangtao Meng, Chuanchao Zang, Li Wang, Xinyu Gao, Jianing Wang, Peng Zhan, Zheng Li, Shanqing Guo  

**Link**: [PDF](https://arxiv.org/pdf/2603.23269)  

**Abstract**: Large Language Models(LLMs) are widely deployed, yet are vulnerable to jailbreak prompts that elicit policy-violating outputs. Although prior studies have uncovered these risks, they typically treat all tokens as equally important during prompt mutation, overlooking the varying contributions of individual tokens to triggering model refusals. Consequently, these attacks introduce substantial redundant searching under query-constrained scenarios, reducing attack efficiency and hindering comprehensive vulnerability assessment. In this work, we conduct a token-level analysis of refusal behavior and observe that token contributions are highly skewed rather than uniform. Moreover, we find strong cross-model consistency in refusal tendencies, enabling the use of a surrogate model to estimate token-level contributions to the target model's refusals. Motivated by these findings, we propose TriageFuzz, a token-aware jailbreak fuzzing framework that adapts the fuzz testing approach with a series of customized designs. TriageFuzz leverages a surrogate model to estimate the contribution of individual tokens to refusal behaviors, enabling the identification of sensitive regions within the prompt. Furthermore, it incorporates a refusal-guided evolutionary strategy that adaptively weights candidate prompts with a lightweight scorer to steer the evolution toward bypassing safety constraints. Extensive experiments on six open-source LLMs and three commercial APIs demonstrate that TriageFuzz achieves comparable attack success rates (ASR) with significantly reduced query costs. Notably, it attains a 90% ASR with over 70% fewer queries compared to baselines. Even under an extremely restrictive budget of 25 queries, TriageFuzz outperforms existing methods, improving ASR by 20-40%. 

---
# SafeSeek: Universal Attribution of Safety Circuits in Language Models 

**Authors**: Miao Yu, Siyuan Fu, Moayad Aloqaily, Zhenhong Zhou, Safa Otoum, Xing fan, Kun Wang, Yufei Guo, Qingsong Wen  

**Link**: [PDF](https://arxiv.org/pdf/2603.23268)  

**Abstract**: Mechanistic interpretability reveals that safety-critical behaviors (e.g., alignment, jailbreak, backdoor) in Large Language Models (LLMs) are grounded in specialized functional components. However, existing safety attribution methods struggle with generalization and reliability due to their reliance on heuristic, domain-specific metrics and search algorithms. To address this, we propose \ourmethod, a unified safety interpretability framework that identifies functionally complete safety circuits in LLMs via optimization. Unlike methods focusing on isolated heads or neurons, \ourmethod introduces differentiable binary masks to extract multi-granular circuits through gradient descent on safety datasets, while integrates Safety Circuit Tuning to utilize these sparse circuits for efficient safety fine-tuning. We validate \ourmethod in two key scenarios in LLM safety: \textbf{(1) backdoor attacks}, identifying a backdoor circuit with 0.42\% sparsity, whose ablation eradicates the Attack Success Rate (ASR) from 100\% $\to$ 0.4\% while retaining over 99\% general utility; \textbf{(2) safety alignment}, localizing an alignment circuit with 3.03\% heads and 0.79\% neurons, whose removal spikes ASR from 0.8\% $\to$ 96.9\%, whereas excluding this circuit during helpfulness fine-tuning maintains 96.5\% safety retention. 

---
# AI Lifecycle-Aware Feasibility Framework for Split-RIC Orchestration in NTN O-RAN 

**Authors**: Daniele Tarchi  

**Link**: [PDF](https://arxiv.org/pdf/2603.23252)  

**Abstract**: Integrating Artificial Intelligence (AI) into Non-Terrestrial Networks (NTN) is constrained by the joint limits of satellite SWaP and feeder-link capacity, which directly impact O-RAN closed-loop control and model lifecycle management. This paper studies the feasibility of distributing the O-RAN control hierarchy across Ground, LEO, and GEO segments through a Split-RIC architecture. We compare three deployment scenarios: (i) ground-centric control with telemetry streaming, (ii) ground--LEO Split-RIC with on-board inference and store-and-forward learning, and (iii) GEO--LEO multi-layer control enabled by inter-satellite links. For each scenario, we derive closed-form expressions for lifecycle energy and lifecycle latency that account for training-data transfer, model dissemination, and near-real-time inference. Numerical sensitivity analysis over feeder-link conditions, model complexity, and orbital intermittency yields operator-relevant feasibility regions that delineate when on-board inference and non-terrestrial learning loops are physically preferable to terrestrial offloading. 

---
# A Learning Method with Gap-Aware Generation for Heterogeneous DAG Scheduling 

**Authors**: Ruisong Zhou, Haijun Zou, Li Zhou, Chumin Sun, Zaiwen Wen  

**Link**: [PDF](https://arxiv.org/pdf/2603.23249)  

**Abstract**: Efficient scheduling of directed acyclic graphs (DAGs) in heterogeneous environments is challenging due to resource capacities and dependencies. In practice, the need for adaptability across environments with varying resource pools and task types, alongside rapid schedule generation, complicates these challenges. We propose WeCAN, an end-to-end reinforcement learning framework for heterogeneous DAG scheduling that addresses task--pool compatibility coefficients and generation-induced optimality gaps. It adopts a two-stage single-pass design: a single forward pass produces task--pool scores and global parameters, followed by a generation map that constructs schedules without repeated network calls. Its weighted cross-attention encoder models task--pool interactions gated by compatibility coefficients, and is size-agnostic to environment fluctuations. Moreover, widely used list-scheduling maps can incur generation-induced optimality gaps from restricted reachability. We introduce an order-space analysis that characterizes the reachable set of generation maps via feasible schedule orders, explains the mechanism behind generation-induced gaps, and yields sufficient conditions for gap elimination. Guided by these conditions, we design a skip-extended realization with an analytically parameterized decreasing skip rule, which enlarges the reachable order set while preserving single-pass efficiency. Experiments on computation graphs and real-world TPC-H DAGs demonstrate improved makespan over strong baselines, with inference time comparable to classical heuristics and faster than multi-round neural schedulers. 

---
# Neural ODE and SDE Models for Adaptation and Planning in Model-Based Reinforcement Learning 

**Authors**: Chao Han, Stefanos Ioannou, Luca Manneschi, T.J. Hayward, Michael Mangan, Aditya Gilra, Eleni Vasilaki  

**Link**: [PDF](https://arxiv.org/pdf/2603.23245)  

**Abstract**: We investigate neural ordinary and stochastic differential equations (neural ODEs and SDEs) to model stochastic dynamics in fully and partially observed environments within a model-based reinforcement learning (RL) framework. Through a sequence of simulations, we show that neural SDEs more effectively capture the inherent stochasticity of transition dynamics, enabling high-performing policies with improved sample efficiency in challenging scenarios. We leverage neural ODEs and SDEs for efficient policy adaptation to changes in environment dynamics via inverse models, requiring only limited interactions with the new environment. To address partial observability, we introduce a latent SDE model that combines an ODE with a GAN-trained stochastic component in latent space. Policies derived from this model provide a strong baseline, outperforming or matching general model-based and model-free approaches across stochastic continuous-control benchmarks. This work demonstrates the applicability of action-conditional latent SDEs for RL planning in environments with stochastic transitions. Our code is available at: this https URL 

---
# General Machine Learning: Theory for Learning Under Variable Regimes 

**Authors**: Aomar Osmani  

**Link**: [PDF](https://arxiv.org/pdf/2603.23220)  

**Abstract**: We study learning under regime variation, where the learner, its memory state, and the evaluative conditions may evolve over time. This paper is a foundational and structural contribution: its goal is to define the core learning-theoretic objects required for such settings and to establish their first theorem-supporting consequences.
The paper develops a regime-varying framework centered on admissible transport, protected-core preservation, and evaluator-aware learning evolution. It records the immediate closure consequences of admissibility, develops a structural obstruction argument for faithful fixed-ontology reduction in genuinely multi-regime settings, and introduces a protected-stability template together with explicit numerical and symbolic witnesses on controlled subclasses, including convex and deductive settings. It also establishes theorem-layer results on evaluator factorization, morphisms, composition, and partial kernel-level alignment across semantically commensurable layers.
A worked two-regime example makes the admissibility certificate, protected evaluative core, and regime-variation cost explicit on a controlled subclass. The symbolic component is deliberately restricted in scope: the paper establishes a first kernel-level compatibility result together with a controlled monotonic deductive witness. The manuscript should therefore be read as introducing a structured learning-theoretic framework for regime-varying learning together with its first theorem-supporting layer, not as a complete quantitative theory of all learning systems. 

---
# ImplicitRM: Unbiased Reward Modeling from Implicit Preference Data for LLM alignment 

**Authors**: Hao Wang, Haocheng Yang, Licheng Pan, Lei Shen, Xiaoxi Li, Yinuo Wang, Zhichao Chen, Yuan Lu, Haoxuan Li, Zhouchen Lin  

**Link**: [PDF](https://arxiv.org/pdf/2603.23184)  

**Abstract**: Reward modeling represents a long-standing challenge in reinforcement learning from human feedback (RLHF) for aligning language models. Current reward modeling is heavily contingent upon experimental feedback data with high collection costs. In this work, we study \textit{implicit reward modeling} -- learning reward models from implicit human feedback (e.g., clicks and copies) -- as a cost-effective alternative. We identify two fundamental challenges in implicit reward modeling: (1) Implicit preference data lacks definitive negative samples, which makes standard positive-negative classification methods inapplicable; (2) Implicit preference data suffers from user preference bias, where different responses have different propensities to elicit user feedback actions, which exacerbates the difficulty of distinguishing definitive negative samples. To address these challenges, we propose ImplicitRM, which aims to learn unbiased reward models from implicit preference data. ImplicitRM stratifies training samples into four latent groups via a stratification model. Building on this, it derives a learning objective through likelihood maximization, which we prove is theoretically unbiased, effectively resolving both challenges. Experiments demonstrate that ImplicitRM learns accurate reward models across implicit preference datasets. Code is available on our project website. 

---
# Reasoning over Semantic IDs Enhances Generative Recommendation 

**Authors**: Yingzhi He, Yan Sun, Junfei Tan, Yuxin Chen, Xiaoyu Kong, Chunxu Shen, Xiang Wang, An Zhang, Tat-Seng Chua  

**Link**: [PDF](https://arxiv.org/pdf/2603.23183)  

**Abstract**: Recent advances in generative recommendation have leveraged pretrained LLMs by formulating sequential recommendation as autoregressive generation over a unified token space comprising language tokens and itemic identifiers, where each item is represented by a compact sequence of discrete tokens, namely Semantic IDs (SIDs). This SID-based formulation enables efficient decoding over large-scale item corpora and provides a natural interface for LLM-based recommenders to leverage rich world knowledge. Meanwhile, breakthroughs in LLM reasoning motivate reasoning-enhanced recommendation, yet effective reasoning over SIDs remains underexplored and challenging. Itemic tokens are not natively meaningful to LLMs; moreover, recommendation-oriented SID reasoning is hard to evaluate, making high-quality supervision scarce.
To address these challenges, we propose SIDReasoner, a two-stage framework that elicits reasoning over SIDs by strengthening SID--language alignment to unlock transferable LLM reasoning, rather than relying on large amounts of recommendation-specific reasoning traces. Concretely, SIDReasoner first enhances SID-language alignment via multi-task training on an enriched SID-centered corpus synthesized by a stronger teacher model, grounding itemic tokens in diverse semantic and behavioral contexts. Building on this enhanced alignment, SIDReasoner further improves recommendation reasoning through outcome-driven reinforced optimization, which guides the model toward effective reasoning trajectories without requiring explicit reasoning annotations. Extensive experiments on three real-world datasets demonstrate the effectiveness of our reasoning-augmented SID-based generative recommendation. Beyond accuracy, the results highlight the broader potential of large reasoning models for generative recommendation, including improved interpretability and cross-domain generalization. 

---
# Robust Safety Monitoring of Language Models via Activation Watermarking 

**Authors**: Toluwani Aremu, Daniil Ognev, Samuele Poppi, Nils Lukas  

**Link**: [PDF](https://arxiv.org/pdf/2603.23171)  

**Abstract**: Large language models (LLMs) can be misused to reveal sensitive information, such as weapon-making instructions or writing malware. LLM providers rely on $\emph{monitoring}$ to detect and flag unsafe behavior during inference. An open security challenge is $\emph{adaptive}$ adversaries who craft attacks that simultaneously (i) evade detection while (ii) eliciting unsafe behavior. Adaptive attackers are a major concern as LLM providers cannot patch their security mechanisms, since they are unaware of how their models are being misused. We cast $\emph{robust}$ LLM monitoring as a security game, where adversaries who know about the monitor try to extract sensitive information, while a provider must accurately detect these adversarial queries at low false positive rates. Our work (i) shows that existing LLM monitors are vulnerable to adaptive attackers and (ii) designs improved defenses through $\emph{activation watermarking}$ by carefully introducing uncertainty for the attacker during inference. We find that $\emph{activation watermarking}$ outperforms guard baselines by up to $52\%$ under adaptive attackers who know the monitoring algorithm but not the secret key. 

---
# Why AI-Generated Text Detection Fails: Evidence from Explainable AI Beyond Benchmark Accuracy 

**Authors**: Shushanta Pudasaini, Luis Miralles-Pechuán, David Lillis, Marisa Llorens Salvador  

**Link**: [PDF](https://arxiv.org/pdf/2603.23146)  

**Abstract**: The widespread adoption of Large Language Models (LLMs) has made the detection of AI-Generated text a pressing and complex challenge. Although many detection systems report high benchmark accuracy, their reliability in real-world settings remains uncertain, and their interpretability is often unexplored. In this work, we investigate whether contemporary detectors genuinely identify machine authorship or merely exploit dataset-specific artefacts. We propose an interpretable detection framework that integrates linguistic feature engineering, machine learning, and explainable AI techniques. When evaluated on two prominent benchmark corpora, namely PAN CLEF 2025 and COLING 2025, our model trained on 30 linguistic features achieves leaderboard-competitive performance, attaining an F1 score of 0.9734. However, systematic cross-domain and cross-generator evaluation reveals substantial generalisation failure: classifiers that excel in-domain degrade significantly under distribution shift. Using SHAP- based explanations, we show that the most influential features differ markedly between datasets, indicating that detectors often rely on dataset-specific stylistic cues rather than stable signals of machine authorship. Further investigation with in-depth error analysis exposes a fundamental tension in linguistic-feature-based AI text detection: the features that are most discriminative on in-domain data are also the features most susceptible to domain shift, formatting variation, and text-length effects. We believe that this knowledge helps build AI detectors that are robust across different settings. To support replication and practical use, we release an open-source Python package that returns both predictions and instance-level explanations for individual texts. 

---
# Can an LLM Detect Instances of Microservice Infrastructure Patterns? 

**Authors**: Carlos Eduardo Duarte, Neil B. Harrison, Filipe Figueiredo Correia, Ademar Aguiar, Pavlína Gonçalves  

**Link**: [PDF](https://arxiv.org/pdf/2603.23073)  

**Abstract**: Architectural patterns are frequently found in various software artifacts. The wide variety of patterns and their implementations makes detection challenging with current tools, especially since they often only support detecting patterns in artifacts written in a single language. Large Language Models (LLMs), trained on a diverse range of software artifacts and knowledge, might overcome the limitations of existing approaches. However, their true effectiveness and the factors influencing their performance have not yet been thoroughly examined. To better understand this, we developed MicroPAD. This tool utilizes GPT 5 nano to identify architectural patterns in software artifacts written in any language, based on natural-language pattern descriptions. We used MicroPAD to evaluate an LLM's ability to detect instances of architectural patterns, particularly infrastructure-related microservice patterns. To accomplish this, we selected a set of GitHub repositories and contacted their top contributors to create a new, human-annotated dataset of 190 repositories containing microservice architectural patterns. The results show that MicroPAD was capable of detecting pattern instances across multiple languages and artifact types. The detection performance varied across patterns (F1 scores ranging from 0.09 to 0.70), specifically in relation to their prevalence and the distinctiveness of the artifacts through which they manifest. We also found that patterns associated with recognizable, dominant artifacts were detected more reliably. Whether these findings generalize to other LLMs and tools is a promising direction for future research. 

---
# AuthorMix: Modular Authorship Style Transfer via Layer-wise Adapter Mixing 

**Authors**: Sarubi Thillainathan, Ji-Ung Lee, Michael Sullivan, Alexander Koller  

**Link**: [PDF](https://arxiv.org/pdf/2603.23069)  

**Abstract**: The task of authorship style transfer involves rewriting text in the style of a target author while preserving the meaning of the original text. Existing style transfer methods train a single model on large corpora to model all target styles at once: this high-cost approach offers limited flexibility for target-specific adaptation, and often sacrifices meaning preservation for style transfer. In this paper, we propose AuthorMix: a lightweight, modular, and interpretable style transfer framework. We train individual, style-specific LoRA adapters on a small set of high-resource authors, allowing the rapid training of specialized adaptation models for each new target via learned, layer-wise adapter mixing, using only a handful of target style training examples. AuthorMix outperforms existing, SoTA style-transfer baselines -- as well as GPT-5.1 -- for low-resource targets, achieving the highest overall score and substantially improving meaning preservation. 

---
# Mind Your HEARTBEAT! Claw Background Execution Inherently Enables Silent Memory Pollution 

**Authors**: Yechao Zhang, Shiqian Zhao, Jie Zhang, Gelei Deng, Jiawen Zhang, Xiaogeng Liu, Chaowei Xiao, Tianwei Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2603.23064)  

**Abstract**: We identify a critical security vulnerability in mainstream Claw personal AI agents: untrusted content encountered during heartbeat-driven background execution can silently pollute agent memory and subsequently influence user-facing behavior without the user's awareness. This vulnerability arises from an architectural design shared across the Claw ecosystem: heartbeat background execution runs in the same session as user-facing conversation, so content ingested from any external source monitored in the background (including email, message channels, news feeds, code repositories, and social platforms) can enter the same memory context used for foreground interaction, often with limited user visibility and without clear source provenance. We formalize this process as an Exposure (E) $\rightarrow$ Memory (M) $\rightarrow$ Behavior (B) pathway: misinformation encountered during heartbeat execution enters the agent's short-term session context, potentially gets written into long-term memory, and later shapes downstream user-facing behavior. We instantiate this pathway in an agent-native social setting using MissClaw, a controlled research replica of Moltbook. We find that (1) social credibility cues, especially perceived consensus, are the dominant driver of short-term behavioral influence, with misleading rates up to 61%; (2) routine memory-saving behavior can promote short-term pollution into durable long-term memory at rates up to 91%, with cross-session behavioral influence reaching 76%; (3) under naturalistic browsing with content dilution and context pruning, pollution still crosses session boundaries. Overall, prompt injection is not required: ordinary social misinformation is sufficient to silently shape agent memory and behavior under heartbeat-driven background execution. 

---
# Machine Learning Models for the Early Detection of Burnout in Software Engineering: a Systematic Literature Review 

**Authors**: Tien Rahayu Tulili, Ayushi Rastogi, Andrea Capiluppi  

**Link**: [PDF](https://arxiv.org/pdf/2603.23063)  

**Abstract**: Burnout is an occupational syndrome that, like many other professions, affects the majority of software engineers. Past research studies showed important trends, including an increasing use of machine learning techniques to allow for an early detection of burnout.
This paper is a systematic literature review (SLR) of the research papers that proposed machine learning (ML) approaches, and focused on detecting burnout in software developers and IT professionals. Our objective is to review the accuracy and precision of the proposed ML techniques, and to formulate recommendations for future researchers interested to replicate or extend those studies.
From our SLR we observed that a majority of primary studies focuses on detecting emotions or utilise emotional dimensions to detect or predict the presence of burnout. We also performed a cross-sectional study to detect which ML approach shows a better performance at detecting emotions; and which dataset has more potential and expressivity to capture emotions.
We believe that, by identifying which ML tools and datasets show a better performance at detecting emotions, and indirectly at identifying burnout, our paper can be a valuable asset to progress in this important research direction. 

---
# DBAutoDoc: Automated Discovery and Documentation of Undocumented Database Schemas via Statistical Analysis and Iterative LLM Refinement 

**Authors**: Amith Nagarajan, Thomas Altman  

**Link**: [PDF](https://arxiv.org/pdf/2603.23050)  

**Abstract**: A tremendous number of critical database systems lack adequate documentation. Declared primary keys are absent, foreign key constraints have been dropped for performance, column names are cryptic abbreviations, and no entity-relationship diagrams exist. We present DBAutoDoc, a system that automates the discovery and documentation of undocumented relational database schemas by combining statistical data analysis with iterative large language model (LLM) refinement.
DBAutoDoc's central insight is that schema understanding is fundamentally an iterative, graph-structured problem. Drawing structural inspiration from backpropagation in neural networks, DBAutoDoc propagates semantic corrections through schema dependency graphs across multiple refinement iterations until descriptions converge. This propagation is discrete and semantic rather than mathematical, but the structural analogy is precise: early iterations produce rough descriptions akin to random initialization, and successive passes sharpen the global picture as context flows through the graph.
The system makes four concrete contributions detailed in the paper. On a suite of benchmark databases, DBAutoDoc achieved overall weighted scores of 96.1% across two model families (Google's Gemini and Anthropic's Claude) using a composite metric. Ablation analysis demonstrates that the deterministic pipeline contributes a 23-point F1 improvement over LLM-only FK detection, confirming that the system's contribution is substantial and independent of LLM pre-training knowledge. DBAutoDoc is released as open-source software with all evaluation configurations and prompt templates included for full reproducibility. 

---
# MSR-HuBERT: Self-supervised Pre-training for Adaptation to Multiple Sampling Rates 

**Authors**: Zikang Huang, Meng Ge, Tianrui Wang, Xuanchen Li, Xiaobao Wang, Longbiao Wang, Jianwu Dang  

**Link**: [PDF](https://arxiv.org/pdf/2603.23048)  

**Abstract**: Self-supervised learning (SSL) has advanced speech processing. However, existing speech SSL methods typically assume a single sampling rate and struggle with mixed-rate data due to temporal resolution mismatch. To address this limitation, we propose MSRHuBERT, a multi-sampling-rate adaptive pre-training method. Building on HuBERT, we replace its single-rate downsampling CNN with a multi-sampling-rate adaptive downsampling CNN that maps raw waveforms from different sampling rates to a shared temporal resolution without resampling. This design enables unified mixed-rate pre-training and fine-tuning. In experiments spanning 16 to 48 kHz, MSRHuBERT outperforms HuBERT on speech recognition and full-band speech reconstruction, preserving high-frequency detail while modeling low-frequency semantic structure. Moreover, MSRHuBERT retains HuBERT's mask-prediction objective and Transformer encoder, so existing analyses and improvements that were developed for HuBERT can apply directly. 

---
# Parametric Knowledge and Retrieval Behavior in RAG Fine-Tuning for Electronic Design Automation 

**Authors**: Julian Oestreich, Maximilian Bley, Frank Binder, Lydia Müller, Maksym Sydorenko, André Alcalde  

**Link**: [PDF](https://arxiv.org/pdf/2603.23047)  

**Abstract**: Retrieval-Augmented Generation (RAG) fine-tuning has shown substantial improvements over vanilla RAG, yet most studies target document question answering and often rely on standard NLP metrics that can obscure factual differences. We evaluate RAG fine-tuning for long-form text generation in electronic design automation, adapting a 7B model under five context augmentation strategies with varying retrieval conditions. We introduce TriFEX, a human-validated, triple-based evaluation pipeline that attributes generated claims to their origin-user query, context and reference-and propose Parametric Knowledge Precision (PKP), which isolates internalized knowledge by filtering out claims leaked in the prompt. We show that ROUGE and BERTScore fail to detect factual differences that our triple-based evaluation reveals. Additionally, we demonstrate that an existing metric for knowledge internalization is retrieva-sensitive, with about 75% of its cross-condition variance driven by changes in the rate at which internal knowledge is expressed (PR), rather than by changes in its actual correctness (PKP). The fine-tuned 7B variants outperform a 72B baseline on most metrics, further showing generalization across conditions and on a related benchmark. These results underscore the limitations of available metrics in RAG evaluation and show that smaller models could be reasonably well adapted to specialized tasks for cost-efficient, on-premises deployment. 

---
# Assessing the Robustness of Climate Foundation Models under No-Analog Distribution Shifts 

**Authors**: Maria Conchita Agana Navarro, Geng Li, Theo Wolf, Maria Perez-Ortiz  

**Link**: [PDF](https://arxiv.org/pdf/2603.23043)  

**Abstract**: The accelerating pace of climate change introduces profound non-stationarities that challenge the ability of Machine Learning based climate emulators to generalize beyond their training distributions. While these emulators offer computationally efficient alternatives to traditional Earth System Models, their reliability remains a potential bottleneck under "no-analog" future climate states, which we define here as regimes where external forcing drives the system into conditions outside the empirical range of the historical training data. A fundamental challenge in evaluating this reliability is data contamination; because many models are trained on simulations that already encompass future scenarios, true out-of-distribution (OOD) performance is often masked. To address this, we benchmark the OOD robustness of three state-of-the-art architectures: U-Net, ConvLSTM, and the ClimaX foundation model specifically restricted to a historical-only training regime (1850-2014). We evaluate these models using two complementary strategies: (i) temporal extrapolation to the recent climate (2015-2023) and (ii) cross-scenario forcing shifts across divergent emission pathways. Our analysis within this experimental setup reveals an accuracy vs. stability trade-off: while the ClimaX foundation model achieves the lowest absolute error, it exhibits higher relative performance changes under distribution shifts, with precipitation errors increasing by up to 8.44% under extreme forcing scenarios. These findings suggest that when restricted to historical training dynamics, even high-capacity foundation models are sensitive to external forcing trajectories. Our results underscore the necessity of scenario-aware training and rigorous OOD evaluation protocols to ensure the robustness of climate emulators under a changing climate. 

---
# HUydra: Full-Range Lung CT Synthesis via Multiple HU Interval Generative Modelling 

**Authors**: António Cardoso, Pedro Sousa, Tania Pereira, Hélder P. Oliveira  

**Link**: [PDF](https://arxiv.org/pdf/2603.23041)  

**Abstract**: Currently, a central challenge and bottleneck in the deployment and validation of computer-aided diagnosis (CAD) models within the field of medical imaging is data scarcity. For lung cancer, one of the most prevalent types worldwide, limited datasets can delay diagnosis and have an impact on patient outcome. Generative AI offers a promising solution for this issue, but dealing with the complex distribution of full Hounsfield Unit (HU) range lung CT scans is challenging and remains as a highly computationally demanding task. This paper introduces a novel decomposition strategy that synthesizes CT images one HU interval at a time, rather than modelling the entire HU domain at once. This framework focuses on training generative architectures on individual tissue-focused HU windows, then merges their output into a full-range scan via a learned reconstruction network that effectively reverses the HU-windowing process. We further propose multi-head and multi-decoder models to better capture textures while preserving anatomical consistency, with a multi-head VQVAE achieving the best performance for the generative task. Quantitative evaluation shows this approach significantly outperforms conventional 2D full-range baselines, achieving a 6.2% improvement in FID and superior MMD, Precision, and Recall across all HU intervals. The best performance is achieved by a multi-head VQVAE variant, demonstrating that it is possible to enhance visual fidelity and variability while also reducing model complexity and computational cost. This work establishes a new paradigm for structure-aware medical image synthesis, aligning generative modelling with clinical interpretation. 

---
# YOLOv10 with Kolmogorov-Arnold networks and vision-language foundation models for interpretable object detection and trustworthy multimodal AI in computer vision perception 

**Authors**: Marios Impraimakis, Daniel Vazquez, Feiyu Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2603.23037)  

**Abstract**: The interpretable object detection capabilities of a novel Kolmogorov-Arnold network framework are examined here. The approach refers to a key limitation in computer vision for autonomous vehicles perception, and beyond. These systems offer limited transparency regarding the reliability of their confidence scores in visually degraded or ambiguous scenes. To address this limitation, a Kolmogorov-Arnold network is employed as an interpretable post-hoc surrogate to model the trustworthiness of the You Only Look Once (Yolov10) detections using seven geometric and semantic features. The additive spline-based structure of the Kolmogorov-Arnold network enables direct visualisation of each feature's influence. This produces smooth and transparent functional mappings that reveal when the model's confidence is well supported and when it is unreliable. Experiments on both Common Objects in Context (COCO), and images from the University of Bath campus demonstrate that the framework accurately identifies low-trust predictions under blur, occlusion, or low texture. This provides actionable insights for filtering, review, or downstream risk mitigation. Furthermore, a bootstrapped language-image (BLIP) foundation model generates descriptive captions of each scene. This tool enables a lightweight multimodal interface without affecting the interpretability layer. The resulting system delivers interpretable object detection with trustworthy confidence estimates. It offers a powerful tool for transparent and practical perception component for autonomous and multimodal artificial intelligence applications. 

---
# Looking Beyond the Window: Global-Local Aligned CLIP for Training-free Open-Vocabulary Semantic Segmentation 

**Authors**: ByeongCheol Lee, Hyun Seok Seong, Sangeek Hyun, Gilhan Park, WonJun Moon, Jae-Pil Heo  

**Link**: [PDF](https://arxiv.org/pdf/2603.23030)  

**Abstract**: A sliding-window inference strategy is commonly adopted in recent training-free open-vocabulary semantic segmentation methods to overcome limitation of the CLIP in processing high-resolution images. However, this approach introduces a new challenge: each window is processed independently, leading to semantic discrepancy across windows. To address this issue, we propose Global-Local Aligned CLIP~(GLA-CLIP), a framework that facilitates comprehensive information exchange across windows. Rather than limiting attention to tokens within individual windows, GLA-CLIP extends key-value tokens to incorporate contextual cues from all windows. Nevertheless, we observe a window bias: outer-window tokens are less likely to be attended, since query features are produced through interactions within the inner window patches, thereby lacking semantic grounding beyond their local context. To mitigate this, we introduce a proxy anchor, constructed by aggregating tokens highly similar to the given query from all windows, which provides a unified semantic reference for measuring similarity across both inner- and outer-window patches. Furthermore, we propose a dynamic normalization scheme that adjusts attention strength according to object scale by dynamically scaling and thresholding the attention map to cope with small-object scenarios. Moreover, GLA-CLIP can be equipped on existing methods and broad their receptive field. Extensive experiments validate the effectiveness of GLA-CLIP in enhancing training-free open-vocabulary semantic segmentation performance. Code is available at this https URL. 

---
# Concept-based explanations of Segmentation and Detection models in Natural Disaster Management 

**Authors**: Samar Heydari, Jawher Said, Galip Ümit Yolcu, Evgenii Kortukov, Elena Golimblevskaia, Evgenios Vlachos, Vasileios Mygdalis, Ioannis Pitas, Sebastian Lapuschkin, Leila Arras  

**Link**: [PDF](https://arxiv.org/pdf/2603.23020)  

**Abstract**: Deep learning models for flood and wildfire segmentation and object detection enable precise, real-time disaster localization when deployed on embedded drone platforms. However, in natural disaster management, the lack of transparency in their decision-making process hinders human trust required for emergency response. To address this, we present an explainability framework for understanding flood segmentation and car detection predictions on the widely used PIDNet and YOLO architectures. More specifically, we introduce a novel redistribution strategy that extends Layer-wise Relevance Propagation (LRP) explanations for sigmoid-gated element-wise fusion layers. This extension allows LRP relevances to flow through the fusion modules of PIDNet, covering the entire computation graph back to the input image. Furthermore, we apply Prototypical Concept-based Explanations (PCX) to provide both local and global explanations at the concept level, revealing which learned features drive the segmentation and detection of specific disaster semantic classes. Experiments on a publicly available flood dataset show that our framework provides reliable and interpretable explanations while maintaining near real-time inference capabilities, rendering it suitable for deployment on resource-constrained platforms, such as Unmanned Aerial Vehicles (UAVs). 

---
# A Sobering Look at Tabular Data Generation via Probabilistic Circuits 

**Authors**: Davide Scassola, Dylan Ponsford, Adrián Javaloy, Sebastiano Saccani, Luca Bortolussi, Henry Gouk, Antonio Vergari  

**Link**: [PDF](https://arxiv.org/pdf/2603.23016)  

**Abstract**: Tabular data is more challenging to generate than text and images, due to its heterogeneous features and much lower sample sizes. On this task, diffusion-based models are the current state-of-the-art (SotA) model class, achieving almost perfect performance on commonly used benchmarks. In this paper, we question the perception of progress for tabular data generation. First, we highlight the limitations of current protocols to evaluate the fidelity of generated data, and advocate for alternative ones. Next, we revisit a simple baseline -- hierarchical mixture models in the form of deep probabilistic circuits (PCs) -- which delivers competitive or superior performance to SotA models for a fraction of the cost. PCs are the generative counterpart of decision forests, and as such can natively handle heterogeneous data as well as deliver tractable probabilistic generation and inference. Finally, in a rigorous empirical analysis we show that the apparent saturation of progress for SotA models is largely due to the use of inadequate metrics. As such, we highlight that there is still much to be done to generate realistic tabular data. Code available at this https URL. 

---
# AgentRAE: Remote Action Execution through Notification-based Visual Backdoors against Screenshots-based Mobile GUI Agents 

**Authors**: Yutao Luo, Haotian Zhu, Shuchao Pang, Zhigang Lu, Tian Dong, Yongbin Zhou, Minhui Xue  

**Link**: [PDF](https://arxiv.org/pdf/2603.23007)  

**Abstract**: The rapid adoption of mobile graphical user interface (GUI) agents, which autonomously control applications and operating systems (OS), exposes new system-level attack surfaces. Existing backdoors against web GUI agents and general GenAI models rely on environmental injection or deceptive pop-ups to mislead the agent operation. However, these techniques do not work on screenshots-based mobile GUI agents due to the challenges of restricted trigger design spaces, OS background interference, and conflicts in multiple trigger-action mappings. We propose AgentRAE, a novel backdoor attack capable of inducing Remote Action Execution in mobile GUI agents using visually natural triggers (e.g., benign app icons in notifications). To address the underfitting caused by natural triggers and achieve accurate multi-target action redirection, we design a novel two-stage pipeline that first enhances the agent's sensitivity to subtle iconographic differences via contrastive learning, and then associates each trigger with a specific mobile GUI agent action through a backdoor post-training. Our extensive evaluation reveals that the proposed backdoor preserves clean performance with an attack success rate of over 90% across ten mobile operations. Furthermore, it is hard to visibly detect the benign-looking triggers and circumvents eight representative state-of-the-art defenses. These results expose an overlooked backdoor vector in mobile GUI agents, underscoring the need for defenses that scrutinize notification-conditioned behaviors and internal agent representations. 

---
# Can Graph Foundation Models Generalize Over Architecture? 

**Authors**: Benjamin Gutteridge, Michael Bronstein, Xiaowen Dong  

**Link**: [PDF](https://arxiv.org/pdf/2603.22984)  

**Abstract**: Graph foundation models (GFMs) have recently attracted interest due to the promise of graph neural network (GNN) architectures that generalize zero-shot across graphs of arbitrary scales, feature dimensions, and domains. While existing work has demonstrated this ability empirically across diverse real-world benchmarks, these tasks share a crucial hidden limitation: they admit a narrow set of effective GNN architectures. In particular, current domain-agnostic GFMs rely on fixed architectural backbones, implicitly assuming that a single message-passing regime suffices across tasks. In this paper, we argue that architecture adaptivity is a necessary requirement for true GFMs. We show that existing approaches are non-robust to task-dependent architectural attributes and, as a case study, use range as a minimal and measurable axis along which this limitation becomes explicit. With theoretical analysis and controlled synthetic experiments, we demonstrate that fixed-backbone GFMs provably under-reach on tasks whose architectural requirements differ from those seen at training time. To address this issue, we introduce a framework that adapts effective GNN architecture at inference time by discovering and mixing task-specific linear graph operators, enabling zero-shot generalization across tasks with heterogeneous architectural requirements, without retraining. We validate our approach on arbitrary-range synthetic tasks and a suite of real-world benchmarks, demonstrating improved performance and robustness over existing domain-agnostic GFMs. 

---
# DariMis: Harm-Aware Modeling for Dari Misinformation Detection on YouTube 

**Authors**: Jawid Ahmad Baktash, Mosa Ebrahimi, Mohammad Zarif Joya, Mursal Dawodi  

**Link**: [PDF](https://arxiv.org/pdf/2603.22977)  

**Abstract**: Dari, the primary language of Afghanistan, is spoken by tens of millions of people yet remains largely absent from the misinformation detection literature. We address this gap with DariMis, the first manually annotated dataset of 9,224 Dari-language YouTube videos, labeled across two dimensions: Information Type (Misinformation, Partly True, True) and Harm Level (Low, Medium, High). A central empirical finding is that these dimensions are structurally coupled, not independent: 55.9 percent of Misinformation carries at least Medium harm potential, compared with only 1.0 percent of True content. This enables Information Type classifiers to function as implicit harm-triage filters in content moderation pipelines.
We further propose a pair-input encoding strategy that represents the video title and description as separate BERT segment inputs, explicitly modeling the semantic relationship between headline claims and body content, a key signal of misleading information. An ablation study against single-field concatenation shows that pair-input encoding yields a 7.0 percentage point gain in Misinformation recall (60.1 percent to 67.1 percent), the safety-critical minority class, despite modest overall macro F1 differences (0.09 percentage points). We benchmark a Dari/Farsi-specialized model (ParsBERT) against XLM-RoBERTa-base; ParsBERT achieves the best test performance with accuracy of 76.60 percent and macro F1 of 72.77 percent. Bootstrap 95 percent confidence intervals are reported for all metrics, and we discuss both the practical significance and statistical limitations of the results. 

---
# Set-Valued Prediction for Large Language Models with Feasibility-Aware Coverage Guarantees 

**Authors**: Ye Li, Anqi Hu, Yuanchang Ye, Shiyan Tong, Zhiyuan Wang, Bo Fu  

**Link**: [PDF](https://arxiv.org/pdf/2603.22966)  

**Abstract**: Large language models (LLMs) inherently operate over a large generation space, yet conventional usage typically reports the most likely generation (MLG) as a point prediction, which underestimates the model's capability: although the top-ranked response can be incorrect, valid answers may still exist within the broader output space and can potentially be discovered through repeated sampling. This observation motivates moving from point prediction to set-valued prediction, where the model produces a set of candidate responses rather than a single MLG. In this paper, we propose a principled framework for set-valued prediction, which provides feasibility-aware coverage guarantees. We show that, given the finite-sampling nature of LLM generation, coverage is not always achievable: even with multiple samplings, LLMs may fail to yield an acceptable response for certain questions within the sampled candidate set. To address this, we establish a minimum achievable risk level (MRL), below which statistical coverage guarantees cannot be satisfied. Building on this insight, we then develop a data-driven calibration procedure that constructs prediction sets from sampled responses by estimating a rigorous threshold, ensuring that the resulting set contains a correct answer with a desired probability whenever the target risk level is feasible. Extensive experiments on six language generation tasks with five LLMs demonstrate both the statistical validity and the predictive efficiency of our framework. 

---
# The EU AI Act and the Rights-based Approach to Technological Governance 

**Authors**: Georgios Pavlidis  

**Link**: [PDF](https://arxiv.org/pdf/2603.22920)  

**Abstract**: The EU AI Act constitutes an important development in shaping the Union's digital regulatory architecture. The Act places fundamental rights at the heart of a risk-based governance framework. The article examines how the AI Act institutionalises a human-centric approach to AI and how the AI Act's provisions explicitly and implicitly embed the protection of rights enshrined in the EU Charter of Fundamental Rights. It argues that fundamental rights function not merely as aspirational goals, but as legal thresholds and procedural triggers across the lifecycle of an AI system. The analysis suggests that the AI Act has the potential to serve as a model for rights-preserving AI systems, while acknowledging that challenges will emerge at the level of implementation. 

---
# EVA: Efficient Reinforcement Learning for End-to-End Video Agent 

**Authors**: Yaolun Zhang, Ruohui Wang, Jiahao Wang, Yepeng Tang, Xuanyu Zheng, Haonan Duan, Hao Lu, Hanming Deng, Lewei Lu  

**Link**: [PDF](https://arxiv.org/pdf/2603.22918)  

**Abstract**: Video understanding with multimodal large language models (MLLMs) remains challenging due to the long token sequences of videos, which contain extensive temporal dependencies and redundant frames. Existing approaches typically treat MLLMs as passive recognizers, processing entire videos or uniformly sampled frames without adaptive reasoning. Recent agent-based methods introduce external tools, yet still depend on manually designed workflows and perception-first strategies, resulting in inefficiency on long videos. We present EVA, an Efficient Reinforcement Learning framework for End-to-End Video Agent, which enables planning-before-perception through iterative summary-plan-action-reflection reasoning. EVA autonomously decides what to watch, when to watch, and how to watch, achieving query-driven and efficient video understanding. To train such agents, we design a simple yet effective three-stage learning pipeline - comprising supervised fine-tuning (SFT), Kahneman-Tversky Optimization (KTO), and Generalized Reward Policy Optimization (GRPO) - that bridges supervised imitation and reinforcement learning. We further construct high-quality datasets for each stage, supporting stable and reproducible training. We evaluate EVA on six video understanding benchmarks, demonstrating its comprehensive capabilities. Compared with existing baselines, EVA achieves a substantial improvement of 6-12% over general MLLM baselines and a further 1-3% gain over prior adaptive agent methods. Our code and model are available at this https URL. 

---
# From the AI Act to a European AI Agency: Completing the Union's Regulatory Architecture 

**Authors**: Georgios Pavlidis  

**Link**: [PDF](https://arxiv.org/pdf/2603.22912)  

**Abstract**: As artificial intelligence (AI) technologies continue to advance, effective risk assessment, regulation, and oversight are necessary to ensure that AI development and deployment align with ethical principles while preserving innovation and economic competitiveness. The adoption of the EU AI Act marks an important step in this direction, establishing a harmonised legal framework that includes detailed provisions on AI governance, as well as the creation of the European AI Office. This paper revisits the question of whether a more robust supranational agency dedicated to AI is still warranted and explores how such a body could enhance policy coherence, improve risk assessment capacities, and foster international cooperation. It also argues that a strengthened EU-level agency would also serve the Union's strategic aim of securing digital and technological sovereignty. 

---
# ForestPrune: High-ratio Visual Token Compression for Video Multimodal Large Language Models via Spatial-Temporal Forest Modeling 

**Authors**: Shaobo Ju, Baiyang Song, Tao Chen, Jiapeng Zhang, Qiong Wu, Chao Chang, HuaiXi Wang, Yiyi Zhou, Rongrong Ji  

**Link**: [PDF](https://arxiv.org/pdf/2603.22911)  

**Abstract**: Due to the great saving of computation and memory overhead, token compression has become a research hot-spot for MLLMs and achieved remarkable progress in image-language tasks. However, for the video, existing methods still fall short of high-ratio token compression. We attribute this shortcoming to the insufficient modeling of temporal and continual video content, and propose a novel and training-free token pruning method for video MLLMs, termed ForestPrune, which achieves effective and high-ratio pruning via Spatial-temporal Forest Modeling. In practice, ForestPrune construct token forests across video frames based on the semantic, spatial and temporal constraints, making an overall comprehension of videos. Afterwards, ForestPrune evaluates the importance of token trees and nodes based on tree depth and node roles, thereby obtaining a globally optimal pruning decision. To validate ForestPrune, we apply it to two representative video MLLMs, namely LLaVA-Video and LLaVA-OneVision, and conduct extensive experiments on a bunch of video benchmarks. The experimental results not only show the great effectiveness for video MLLMs, e.g., retaining 95.8% average accuracy while reducing 90% tokens for LLaVA-OneVision, but also show its superior performance and efficiency than the compared token compression methods, e.g., +10.1% accuracy on MLVU and -81.4% pruning time than FrameFusion on LLaVA-Video. 

---
# Off-Policy Evaluation and Learning for Survival Outcomes under Censoring 

**Authors**: Kohsuke Kubota, Mitsuhiro Takahashi, Yuta Saito  

**Link**: [PDF](https://arxiv.org/pdf/2603.22900)  

**Abstract**: Optimizing survival outcomes, such as patient survival or customer retention, is a critical objective in data-driven decision-making. Off-Policy Evaluation~(OPE) provides a powerful framework for assessing such decision-making policies using logged data alone, without the need for costly or risky online experiments in high-stakes applications. However, typical estimators are not designed to handle right-censored survival outcomes, as they ignore unobserved survival times beyond the censoring time, leading to systematic underestimation of the true policy performance. To address this issue, we propose a novel framework for OPE and Off-Policy Learning~(OPL) tailored for survival outcomes under censoring. Specifically, we introduce IPCW-IPS and IPCW-DR, which employ the Inverse Probability of Censoring Weighting technique to explicitly deal with censoring bias. We theoretically establish that our estimators are unbiased and that IPCW-DR achieves double robustness, ensuring consistency if either the propensity score or the outcome model is correct. Furthermore, we extend this framework to constrained OPL to optimize policy value under budget constraints. We demonstrate the effectiveness of our proposed methods through simulation studies and illustrate their practical impacts using public real-world data for both evaluation and learning tasks. 

---
# Confidence Calibration under Ambiguous Ground Truth 

**Authors**: Linwei Tao, Haoyang Luo, Minjing Dong, Chang Xu  

**Link**: [PDF](https://arxiv.org/pdf/2603.22879)  

**Abstract**: Confidence calibration assumes a unique ground-truth label per input, yet this assumption fails wherever annotators genuinely disagree. Post-hoc calibrators fitted on majority-voted labels, the standard single-label targets used in practice, can appear well-calibrated under conventional evaluation yet remain substantially miscalibrated against the underlying annotator distribution. We show that this failure is structural: under simplifying assumptions, Temperature Scaling is biased toward temperatures that underestimate annotator uncertainty, with true-label miscalibration increasing monotonically with annotation entropy. To address this, we develop a family of ambiguity-aware post-hoc calibrators that optimise proper scoring rules against the full label distribution and require no model retraining. Our methods span progressively weaker annotation requirements: Dirichlet-Soft leverages the full annotator distribution and achieves the best overall calibration quality across settings; Monte Carlo Temperature Scaling with a single annotation per example (MCTS S=1) matches full-distribution calibration across all benchmarks, demonstrating that pre-aggregated label distributions are unnecessary; and Label-Smooth Temperature Scaling (LS-TS) operates with voted labels alone by constructing data-driven pseudo-soft targets from the model's own confidence. Experiments on four benchmarks with real multi-annotator distributions (CIFAR-10H, ChaosNLI) and clinically-informed synthetic annotations (ISIC~2019, DermaMNIST) show that Dirichlet-Soft reduces true-label ECE by 55-87% relative to Temperature Scaling, while LS-TS reduces ECE by 9-77% without any annotator data. 

---
# Grounding Sim-to-Real Generalization in Dexterous Manipulation: An Empirical Study with Vision-Language-Action Models 

**Authors**: Ruixing Jin, Zicheng Zhu, Ruixiang Ouyang, Sheng Xu, Bo Yue, Zhizheng Wu, Guiliang Liu  

**Link**: [PDF](https://arxiv.org/pdf/2603.22876)  

**Abstract**: Learning a generalist control policy for dexterous manipulation typically relies on large-scale datasets. Given the high cost of real-world data collection, a practical alternative is to generate synthetic data through simulation. However, the resulting synthetic data often exhibits a significant gap from real-world distributions. While many prior studies have proposed algorithms to bridge the Sim-to-Real discrepancy, there remains a lack of principled research that grounds these methods in real-world manipulation tasks, particularly their performance on generalist policies such as Vision-Language-Action (VLA) models. In this study, we empirically examine the primary determinants of Sim-to-Real generalization across four dimensions: multi-level domain randomization, photorealistic rendering, physics-realistic modeling, and reinforcement learning updates. To support this study, we design a comprehensive evaluation protocol to quantify the real-world performance of manipulation tasks. The protocol accounts for key variations in background, lighting, distractors, object types, and spatial features. Through experiments involving over 10k real-world trials, we derive critical insights into Sim-to-Real transfer. To inform and advance future studies, we release both the robotic platforms and the evaluation protocol for public access to facilitate independent verification, thereby establishing a realistic and standardized benchmark for dexterous manipulation policies. 

---
# Agent-Sentry: Bounding LLM Agents via Execution Provenance 

**Authors**: Rohan Sequeira, Stavros Damianakis, Umar Iqbal, Konstantinos Psounis  

**Link**: [PDF](https://arxiv.org/pdf/2603.22868)  

**Abstract**: Agentic computing systems, which autonomously spawn new functionalities based on natural language instructions, are becoming increasingly prevalent. While immensely capable, these systems raise serious security, privacy, and safety concerns. Fundamentally, the full set of functionalities offered by these systems, combined with their probabilistic execution flows, is not known beforehand. Given this lack of characterization, it is non-trivial to validate whether a system has successfully carried out the user's intended task or instead executed irrelevant actions, potentially as a consequence of compromise. In this paper, we propose Agent-Sentry, a framework that attempts to bound agentic systems to address this problem. Our key insight is that agentic systems are designed for specific use cases and therefore need not expose unbounded or unspecified functionalities. Once bounded, these systems become easier to scrutinize. Agent-Sentry operationalizes this insight by uncovering frequent functionalities offered by an agentic system, along with their execution traces, to construct behavioral bounds. It then learns a policy from these traces and blocks tool calls that deviate from learned behaviors or that misalign with user intent. Our evaluation shows that Agent-Sentry helps prevent over 90\% of attacks that attempt to trigger out-of-bounds executions, while preserving up to 98\% of system utility. 

---
# The Coordinate System Problem in Persistent Structural Memory for Neural Architectures 

**Authors**: Abhinaba Basu  

**Link**: [PDF](https://arxiv.org/pdf/2603.22858)  

**Abstract**: We introduce the Dual-View Pheromone Pathway Network (DPPN), an architecture that routes sparse attention through a persistent pheromone field over latent slot transitions, and use it to discover two independent requirements for persistent structural memory in neural networks. Through five progressively refined experiments using up to 10 seeds per condition across 5 model variants and 4 transfer targets, we identify a core principle: persistent memory requires a stable coordinate system, and any coordinate system learned jointly with the model is inherently unstable. We characterize three obstacles -- pheromone saturation, surface-structure entanglement, and coordinate incompatibility -- and show that neither contrastive updates, multi-source distillation, Hungarian alignment, nor semantic decomposition resolves the instability when embeddings are learned from scratch. Fixed random Fourier features provide extrinsic coordinates that are stable, structure-blind, and informative, but coordinate stability alone is insufficient: routing-bias pheromone does not transfer (10 seeds, p>0.05). DPPN outperforms transformer and random sparse baselines for within-task learning (AULC 0.700 vs 0.680 vs 0.670). Replacing routing bias with learning-rate modulation eliminates negative transfer: warm pheromone as a learning-rate prior achieves +0.003 on same-family tasks (17 seeds, p<0.05) while never reducing performance. A structure completion function over extrinsic coordinates produces +0.006 same-family bonus beyond regularization, showing the catch-22 between stability and informativeness is partially permeable to learned functions. The contribution is two independent requirements for persistent structural memory: (a) coordinate stability and (b) graceful transfer mechanism. 

---
# Avoiding Over-smoothing in Social Media Rumor Detection with Pre-trained Propagation Tree Transformer 

**Authors**: Chaoqun Cui, Caiyan Jia  

**Link**: [PDF](https://arxiv.org/pdf/2603.22854)  

**Abstract**: Deep learning techniques for rumor detection typically utilize Graph Neural Networks (GNNs) to analyze post relations. These methods, however, falter due to over-smoothing issues when processing rumor propagation structures, leading to declining performance. Our investigation into this issue reveals that over-smoothing is intrinsically tied to the structural characteristics of rumor propagation trees, in which the majority of nodes are 1-level nodes. Furthermore, GNNs struggle to capture long-range dependencies within these trees. To circumvent these challenges, we propose a Pre-Trained Propagation Tree Transformer (P2T3) method based on pure Transformer architecture. It extracts all conversation chains from a tree structure following the propagation direction of replies, utilizes token-wise embedding to infuse connection information and introduces necessary inductive bias, and pre-trains on large-scale unlabeled datasets. Experiments indicate that P2T3 surpasses previous state-of-the-art methods in multiple benchmark datasets and performs well under few-shot conditions. P2T3 not only avoids the over-smoothing issue inherent in GNNs but also potentially offers a large model or unified multi-modal scheme for future social media research. 

---
# Agent Audit: A Security Analysis System for LLM Agent Applications 

**Authors**: Haiyue Zhang, Yi Nian, Yue Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2603.22853)  

**Abstract**: What should a developer inspect before deploying an LLM agent: the model, the tool code, the deployment configuration, or all three? In practice, many security failures in agent systems arise not from model weights alone, but from the surrounding software stack: tool functions that pass untrusted inputs to dangerous operations, exposed credentials in deployment artifacts, and over-privileged Model Context Protocol (MCP) configurations.
We present Agent Audit, a security analysis system for LLM agent applications. Agent Audit analyzes Python agent code and deployment artifacts through an agent-aware pipeline that combines dataflow analysis, credential detection, structured configuration parsing, and privilege-risk checks. The system reports findings in terminal, JSON, and SARIF formats, enabling direct integration with local development workflows and CI/CD pipelines. On a benchmark of 22 samples with 42 annotated vulnerabilities, Agent Audit detects 40 vulnerabilities with 6 false positives, substantially improving recall over common SAST baselines while maintaining sub-second scan times. Agent Audit is open source and installable via pip, making security auditing accessible for agent systems.
In the live demonstration, attendees scan vulnerable agent repositories and observe how Agent Audit identifies security risks in tool functions, prompts, and more. Findings are linked to source locations and configuration paths, and can be exported into VS Code and GitHub Code Scanning for interactive inspection. 

---
# UniQueR: Unified Query-based Feedforward 3D Reconstruction 

**Authors**: Chensheng Peng, Quentin Herau, Jiezhi Yang, Yichen Xie, Yihan Hu, Wenzhao Zheng, Matthew Strong, Masayoshi Tomizuka, Wei Zhan  

**Link**: [PDF](https://arxiv.org/pdf/2603.22851)  

**Abstract**: We present UniQueR, a unified query-based feedforward framework for efficient and accurate 3D reconstruction from unposed images. Existing feedforward models such as DUSt3R, VGGT, and AnySplat typically predict per-pixel point maps or pixel-aligned Gaussians, which remain fundamentally 2.5D and limited to visible surfaces. In contrast, UniQueR formulates reconstruction as a sparse 3D query inference problem. Our model learns a compact set of 3D anchor points that act as explicit geometric queries, enabling the network to infer scene structure, including geometry in occluded regions--in a single forward pass. Each query encodes spatial and appearance priors directly in global 3D space (instead of per-frame camera space) and spawns a set of 3D Gaussians for differentiable rendering. By leveraging unified query interactions across multi-view features and a decoupled cross-attention design, UniQueR achieves strong geometric expressiveness while substantially reducing memory and computational cost. Experiments on Mip-NeRF 360 and VR-NeRF demonstrate that UniQueR surpasses state-of-the-art feedforward methods in both rendering quality and geometric accuracy, using an order of magnitude fewer primitives than dense alternatives. 

---
# UAV-DETR: DETR for Anti-Drone Target Detection 

**Authors**: Jun Yang, Dong Wang, Hongxu Yin, Hongpeng Li, Jianxiong Yu  

**Link**: [PDF](https://arxiv.org/pdf/2603.22841)  

**Abstract**: Drone detection is pivotal in numerous security and counter-UAV applications. However, existing deep learning-based methods typically struggle to balance robust feature representation with computational efficiency. This challenge is particularly acute when detecting miniature drones against complex backgrounds under severe environmental interference. To address these issues, we introduce UAV-DETR, a novel framework that integrates a small-target-friendly architecture with real-time detection capabilities. Specifically, UAV-DETR features a WTConv-enhanced backbone and a Sliding Window Self-Attention (SWSA-IFI) encoder, capturing the high-frequency structural details of tiny targets while drastically reducing parameter overhead. Furthermore, we propose an Efficient Cross-Scale Feature Recalibration and Fusion Network (ECFRFN) to suppress background noise and aggregate multi-scale semantics. To further enhance accuracy, UAV-DETR incorporates a hybrid Inner-CIoU and NWD loss strategy, mitigating the extreme sensitivity of standard IoU metrics to minor positional deviations in small objects. Extensive experiments demonstrate that UAV-DETR significantly outperforms the baseline RT-DETR on our custom UAV dataset (+6.61% in mAP50:95, with a 39.8% reduction in parameters) and the public DUT-ANTI-UAV benchmark (+1.4% in Precision, +1.0% in F1-Score). These results establish UAV-DETR as a superior trade-off between efficiency and precision in counter-UAV object detection. The code is available at this https URL. 

---
# URA-Net: Uncertainty-Integrated Anomaly Perception and Restoration Attention Network for Unsupervised Anomaly Detection 

**Authors**: Wei Luo, Peng Xing, Yunkang Cao, Haiming Yao, Weiming Shen, Zechao Li  

**Link**: [PDF](https://arxiv.org/pdf/2603.22840)  

**Abstract**: Unsupervised anomaly detection plays a pivotal role in industrial defect inspection and medical image analysis, with most methods relying on the reconstruction framework. However, these methods may suffer from over-generalization, enabling them to reconstruct anomalies well, which leads to poor detection performance. To address this issue, instead of focusing solely on normality reconstruction, we propose an innovative Uncertainty-Integrated Anomaly Perception and Restoration Attention Network (URA-Net), which explicitly restores abnormal patterns to their corresponding normality. First, unlike traditional image reconstruction methods, we utilize a pre-trained convolutional neural network to extract multi-level semantic features as the reconstruction target. To assist the URA-Net learning to restore anomalies, we introduce a novel feature-level artificial anomaly synthesis module to generate anomalous samples for training. Subsequently, a novel uncertainty-integrated anomaly perception module based on Bayesian neural networks is introduced to learn the distributions of anomalous and normal features. This facilitates the estimation of anomalous regions and ambiguous boundaries, laying the foundation for subsequent anomaly restoration. Then, we propose a novel restoration attention mechanism that leverages global normal semantic information to restore detected anomalous regions, thereby obtaining defect-free restored features. Finally, we employ residual maps between input features and restored features for anomaly detection and localization. The comprehensive experimental results on two industrial datasets, MVTec AD and BTAD, along with a medical image dataset, OCT-2017, unequivocally demonstrate the effectiveness and superiority of the proposed method. 

---
# TDATR: Improving End-to-End Table Recognition via Table Detail-Aware Learning and Cell-Level Visual Alignment 

**Authors**: Chunxia Qin, Chenyu Liu, Pengcheng Xia, Jun Du, Baocai Yin, Bing Yin, Cong Liu  

**Link**: [PDF](https://arxiv.org/pdf/2603.22819)  

**Abstract**: Tables are pervasive in diverse documents, making table recognition (TR) a fundamental task in document analysis. Existing modular TR pipelines separately model table structure and content, leading to suboptimal integration and complex workflows. End-to-end approaches rely heavily on large-scale TR data and struggle in data-constrained scenarios. To address these issues, we propose TDATR (Table Detail-Aware Table Recognition) improves end-to-end TR through table detail-aware learning and cell-level visual alignment. TDATR adopts a ``perceive-then-fuse'' strategy. The model first performs table detail-aware learning to jointly perceive table structure and content through multiple structure understanding and content recognition tasks designed under a language modeling paradigm. These tasks can naturally leverage document data from diverse scenarios to enhance model robustness. The model then integrates implicit table details to generate structured HTML outputs, enabling more efficient TR modeling when trained with limited data. Furthermore, we design a structure-guided cell localization module integrated into the end-to-end TR framework, which efficiently locates cell and strengthens vision-language alignment. It enhances the interpretability and accuracy of TR. We achieve state-of-the-art or highly competitive performance on seven benchmarks without dataset-specific fine-tuning. 

---
# When AI Shows Its Work, Is It Actually Working? Step-Level Evaluation Reveals Frontier Language Models Frequently Bypass Their Own Reasoning 

**Authors**: Abhinaba Basu, Pavan Chakraborty  

**Link**: [PDF](https://arxiv.org/pdf/2603.22816)  

**Abstract**: Language models increasingly "show their work" by writing step-by-step reasoning before answering. But are these reasoning steps genuinely used, or decorative narratives generated after the model has already decided? Consider: a medical AI writes "The patient's eosinophilia and livedo reticularis following catheterization suggest cholesterol embolization syndrome. Answer: B." If we remove the eosinophilia observation, does the diagnosis change? For most frontier models, the answer is no - the step was decorative.
We introduce step-level evaluation: remove one reasoning sentence at a time and check whether the answer changes. This simple test requires only API access -- no model weights -- and costs approximately $1-2 per model per task.
Testing 10 frontier models (GPT-5.4, Claude Opus, DeepSeek-V3.2, MiniMax-M2.5, Kimi-K2.5, and others) across sentiment, mathematics, topic classification, and medical QA (N=376-500 each), the majority produce decorative reasoning: removing any step changes the answer less than 17% of the time, while any single step alone recovers the answer. This holds even on math, where smaller models (0.8-8B) show genuine step dependence (55% necessity).
Two models break the pattern: MiniMax-M2.5 on sentiment (37% necessity) and Kimi-K2.5 on topic classification (39%) - but both shortcut other tasks. Faithfulness is model-specific and task-specific.
We also discover "output rigidity": on the same medical questions, Claude Opus writes 11 diagnostic steps while GPT-OSS-120B outputs a single token. Mechanistic analysis (attention patterns) confirms that CoT attention drops more in late layers for decorative tasks (33%) than faithful ones (20%).
Implications: step-by-step explanations from frontier models are largely decorative, per-model per-domain evaluation is essential, and training objectives - not scale - determine whether reasoning is genuine. 

---
# Focus, Don't Prune: Identifying Instruction-Relevant Regions for Information-Rich Image Understanding 

**Authors**: Mincheol Kwon, Minseung Lee, Seonga Choi, Miso Choi, Kyeong-Jin Oh, Hyunyoung Lee, Cheonyoung Park, Yongho Song, Seunghyun Park, Jinkyu Kim  

**Link**: [PDF](https://arxiv.org/pdf/2603.22815)  

**Abstract**: Large Vision-Language Models (LVLMs) have shown strong performance across various multimodal tasks by leveraging the reasoning capabilities of Large Language Models (LLMs). However, processing visually complex and information-rich images, such as infographics or document layouts, requires these models to generate a large number of visual tokens, leading to significant computational overhead. To address this, we propose PinPoint, a novel two-stage framework that first identifies instruction-relevant image regions and then refines them to extract fine-grained visual features for improved reasoning and efficiency. Central to our approach is the Instruction-Region Alignment, which localizes relevant regions using both visual input and textual instructions. We further introduce new annotations that provide richer ground-truth supervision for instruction-relevant regions across challenging VQA benchmarks: InfographicVQA, MultiPageDocVQA, and SinglePageDocVQA. Experimental results show that PinPoint not only achieves superior accuracy compared to existing methods but also reduces computational overhead by minimizing irrelevant visual tokens. 

---
# PhotoAgent: A Robotic Photographer with Spatial and Aesthetic Understanding 

**Authors**: Lirong Che, Zhenfeng Gan, Yanbo Chen, Junbo Tan, Xueqian Wang  

**Link**: [PDF](https://arxiv.org/pdf/2603.22796)  

**Abstract**: Embodied agents for creative tasks like photography must bridge the semantic gap between high-level language commands and geometric control. We introduce PhotoAgent, an agent that achieves this by integrating Large Multimodal Models (LMMs) reasoning with a novel control paradigm. PhotoAgent first translates subjective aesthetic goals into solvable geometric constraints via LMM-driven, chain-of-thought (CoT) reasoning, allowing an analytical solver to compute a high-quality initial viewpoint. This initial pose is then iteratively refined through visual reflection within a photorealistic internal world model built with 3D Gaussian Splatting (3DGS). This ``mental simulation'' replaces costly and slow physical trial-and-error, enabling rapid convergence to aesthetically superior results. Evaluations confirm that PhotoAgent excels in spatial reasoning and achieves superior final image quality. 

---
# Quantum Random Forest for the Regression Problem 

**Authors**: Kamil Khadiev, Liliya Safina  

**Link**: [PDF](https://arxiv.org/pdf/2603.22790)  

**Abstract**: The Random Forest model is one of the popular models of Machine learning. We present a quantum algorithm for testing (forecasting) process of the Random Forest machine learning model for the Regression problem. The presented algorithm is more efficient (in terms of query complexity or running time) than the classical counterpart. 

---
# Exposure-Normalized Bed and Chair Fall Rates via Continuous AI Monitoring 

**Authors**: Paolo Gabriel, Peter Rehani, Zack Drumm, Tyler Troy, Tiffany Wyatt, Narinder Singh  

**Link**: [PDF](https://arxiv.org/pdf/2603.22785)  

**Abstract**: This retrospective cohort study used continuous AI monitoring to estimate fall rates by exposure time rather than occupied bed-days. From August 2024 to December 2025, 3,980 eligible monitoring units contributed 292,914 hourly rows, yielding probability-weighted rates of 17.8 falls per 1,000 chair exposure-hours and 4.3 per 1,000 bed exposure-hours. Within the study window, 43 adjudicated falls matched the monitoring pipeline, and 40 linked to eligible exposure hours for the primary Poisson model, producing an adjusted chair-versus-bed rate ratio of 2.35 (95% confidence interval 0.87 to 6.33; p=0.0907). In a separate broader observation cohort (n=32 deduplicated events), 6 of 7 direct chair falls involved footrest-positioning failures. Because this was an observational study in a single health system, these findings remain hypothesis-generating and support testing safer chair setups rather than using chairs less. 

---
# KARMA: Knowledge-Action Regularized Multimodal Alignment for Personalized Search at Taobao 

**Authors**: Zhi Sun, Wenming Zhang, Yi Wei, Liren Yu, Zhixuan Zhang, Dan Ou, Haihong Tang  

**Link**: [PDF](https://arxiv.org/pdf/2603.22779)  

**Abstract**: Large Language Models (LLMs) are equipped with profound semantic knowledge, making them a natural choice for injecting semantic generalization into personalized search systems. However, in practice we find that directly fine-tuning LLMs on industrial personalized tasks (e.g. next item prediction) often yields suboptimal results. We attribute this bottleneck to a critical Knowledge--Action Gap: the inherent conflict between preserving pre-trained semantic knowledge and aligning with specific personalized actions by discriminative objectives. Empirically, action-only training objectives induce Semantic Collapse, such as attention ``sinks''. This degradation severely cripples the LLM's generalization, failing to bring improvements to personalized search systems.
We propose KARMA (Knowledge--Action Regularized Multimodal Alignment), a unified framework that treats semantic reconstruction as a train-only regularizer. KARMA optimizes a next-interest embedding for retrieval (Action) while enforcing semantic decodability (Knowledge) through two complementary objectives: (i) history-conditioned semantic generation, which anchors optimization to the LLM's native next-token distribution, and (ii) embedding-conditioned semantic reconstruction, which constrains the interest embedding to remain semantically recoverable.
On Taobao search system, KARMA mitigates semantic collapse (attention-sink analysis) and improves both action metrics and semantic fidelity. In ablations, semantic decodability yields up to +22.5 HR@200. With KARMA, we achieve +0.25 CTR AUC in ranking, +1.86 HR in pre-ranking and +2.51 HR in recalling. Deployed online with low inference overhead at ranking stage, KARMA drives +0.5% increase in Item Click. 

---
# From Arithmetic to Logic: The Resilience of Logic and Lookup-Based Neural Networks Under Parameter Bit-Flips 

**Authors**: Alan T. L. Bacellar, Sathvik Chemudupati, Shashank Nag, Allison Seigler, Priscila M. V. Lima, Felipe M. G. França, Lizy K. John  

**Link**: [PDF](https://arxiv.org/pdf/2603.22770)  

**Abstract**: The deployment of deep neural networks (DNNs) in safety-critical edge environments necessitates robustness against hardware-induced bit-flip errors. While empirical studies indicate that reducing numerical precision can improve fault tolerance, the theoretical basis of this phenomenon remains underexplored. In this work, we study resilience as a structural property of neural architectures rather than solely as a property of a dataset-specific trained solution. By deriving the expected squared error (MSE) under independent parameter bit flips across multiple numerical formats and layer primitives, we show that lower precision, higher sparsity, bounded activations, and shallow depth are consistently favored under this corruption model. We then argue that logic and lookup-based neural networks realize the joint limit of these design trends. Through ablation studies on the MLPerf Tiny benchmark suite, we show that the observed empirical trends are consistent with the theoretical predictions, and that LUT-based models remain highly stable in corruption regimes where standard floating-point models fail sharply. Furthermore, we identify a novel even-layer recovery effect unique to logic-based architectures and analyze the structural conditions under which it emerges. Overall, our results suggest that shifting from continuous arithmetic weights to discrete Boolean lookups can provide a favorable accuracy-resilience trade-off for hardware fault tolerance. 

---
# From Overload to Convergence: Supporting Multi-Issue Human-AI Negotiation with Bayesian Visualization 

**Authors**: Mehul Parmar, Chaklam Silpasuwanchai  

**Link**: [PDF](https://arxiv.org/pdf/2603.22766)  

**Abstract**: As AI systems increasingly mediate negotiations, understanding how the number of negotiated issues impacts human performance is crucial for maintaining human agency. We designed a human-AI negotiation case study in a realistic property rental scenario, varying the number of negotiated issues; empirical findings show that without support, performance stays stable up to three issues but declines as additional issues increase cognitive load. To address this, we introduce a novel uncertainty-based visualization driven by Bayesian estimation of agreement probability. It shows how the space of mutually acceptable agreements narrows as negotiation progresses, helping users identify promising options. In a within-subjects experiment (N=32), it improved human outcomes and efficiency, preserved human control, and avoided redistributing value. Our findings surface practical limits on the complexity people can manage in human-AI negotiation, advance theory on human performance in complex negotiations, and offer validated design guidance for interactive systems. 

---
# DALDALL: Data Augmentation for Lexical and Semantic Diverse in Legal Domain by leveraging LLM-Persona 

**Authors**: Janghyeok Choi, Jaewon Lee, Sungzoon Cho  

**Link**: [PDF](https://arxiv.org/pdf/2603.22765)  

**Abstract**: Data scarcity remains a persistent challenge in low-resource domains. While existing data augmentation methods leverage the generative capabilities of large language models (LLMs) to produce large volumes of synthetic data, these approaches often prioritize quantity over quality and lack domain-specific strategies. In this work, we introduce DALDALL, a persona-based data augmentation framework tailored for legal information retrieval (IR). Our method employs domain-specific professional personas--such as attorneys, prosecutors, and judges--to generate synthetic queries that exhibit substantially greater lexical and semantic diversity than vanilla prompting approaches. Experiments on the CLERC and COLIEE benchmarks demonstrate that persona-based augmentation achieves improvement in lexical diversity as measured by Self-BLEU scores, while preserving semantic fidelity to the original queries. Furthermore, dense retrievers fine-tuned on persona-augmented data consistently achieve competitive or superior recall performance compared to those trained on original data or generic augmentations. These findings establish persona-based prompting as an effective strategy for generating high-quality training data in specialized, low-resource domains. 

---
# KALAVAI: Predicting When Independent Specialist Fusion Works -- A Quantitative Model for Post-Hoc Cooperative LLM Training 

**Authors**: Ramchand Kumaresan  

**Link**: [PDF](https://arxiv.org/pdf/2603.22755)  

**Abstract**: Independently trained domain specialists can be fused post-hoc into a single model that outperforms any individual specialist, and the gain is predictable: gain = 0.82 x divergence - 2.72 (R^2 = 0.856, n=6, 3-26% divergence). This enables practitioners to estimate cooperative value before committing compute. Below ~3.3% divergence, gains approach this http URL the KALAVAI protocol, contributors fine-tune copies of a shared checkpoint independently, then submit for lightweight MoE routing (500 steps). Gains are consistent: +7.72% at 410M (+/-0.02%, 3 seeds), +7.49% at 1B (+/-0.01%, 3 seeds), +6.53% at 6.9B, each over the best specialist. The router matches domain-oracle routing within <10^{-5} nats. Cross-lingual fusion (Tamil/Yoruba/Welsh/Code) achieves +21.76%, with Yoruba perplexity falling 41.9 to 7.7. A 20-contributor federation achieves +16.71% (+/-0.07pp, 3 seeds).Three requirements bound the protocol. Shared initialisation is necessary: checkpoint mismatch degrades routing. Frozen layers are optional below ~10,000 steps and beneficial beyond. Learned routing is essential: uniform averaging degrades by -1.2% vs. best specialist, while any trained router achieves oracle-optimal assignment. 

---
# PopResume: Causal Fairness Evaluation of LLM/VLM Resume Screeners with Population-Representative Dataset 

**Authors**: Sumin Yu, Juhyeon Park, Taesup Moon  

**Link**: [PDF](https://arxiv.org/pdf/2603.22714)  

**Abstract**: We present PopResume, a population-representative resume dataset for causal fairness auditing of LLM- and VLM-based resume screening systems. Unlike existing benchmarks that rely on manually injected demographic information and outcome-level disparities, PopResume is grounded in population statistics and preserves natural attribute relationships, enabling path-specific effect (PSE)-based fairness evaluation. We decompose the effect of a protected attribute on resume scores into two paths: the business necessity path, mediated by job-relevant qualifications, and the redlining path, mediated by demographic proxies. This distinction allows auditors to separate legally permissible from impermissible sources of disparity. Evaluating four LLMs and four VLMs on PopResume's 60.8K resumes across five occupations, we identify five representative discrimination patterns that aggregate metrics fail to capture. Our results demonstrate that PSE-based evaluation reveals fairness issues masked by outcome-level measures, underscoring the need for causally-grounded auditing frameworks in AI-assisted hiring. 

---
# WiFi2Cap: Semantic Action Captioning from Wi-Fi CSI via Limb-Level Semantic Alignment 

**Authors**: Tzu-Ti Wei, Chu-Yu Huang, Yu-Chee Tseng, Jen-Jee Chen  

**Link**: [PDF](https://arxiv.org/pdf/2603.22690)  

**Abstract**: Privacy-preserving semantic understanding of human activities is important for indoor sensing, yet existing Wi-Fi CSI-based systems mainly focus on pose estimation or predefined action classification rather than fine-grained language generation. Mapping CSI to natural-language descriptions remains challenging because of the semantic gap between wireless signals and language and direction-sensitive ambiguities such as left/right limb confusion. We propose WiFi2Cap, a three-stage framework for generating action captions directly from Wi-Fi CSI. A vision-language teacher learns transferable supervision from synchronized video-text pairs, and a CSI student is aligned to the teacher's visual space and text embeddings. To improve direction-sensitive captioning, we introduce a Mirror-Consistency Loss that reduces mirrored-action and left-right ambiguities during cross-modal alignment. A prefix-tuned language model then generates action descriptions from CSI embeddings. We also introduce the WiFi2Cap Dataset, a synchronized CSI-RGB-sentence benchmark for semantic captioning from Wi-Fi signals. Experimental results show that WiFi2Cap consistently outperforms baseline methods on BLEU-4, METEOR, ROUGE-L, CIDEr, and SPICE, demonstrating effective privacy-friendly semantic sensing. 

---
# Vision-based Deep Learning Analysis of Unordered Biomedical Tabular Datasets via Optimal Spatial Cartography 

**Authors**: Sakib Mostafa, Tarik Massoud, Maximilian Diehn, Lei Xing, Md Tauhidul Islam  

**Link**: [PDF](https://arxiv.org/pdf/2603.22675)  

**Abstract**: Tabular data are central to biomedical research, from liquid biopsy and bulk and single-cell transcriptomics to electronic health records and phenotypic profiling. Unlike images or sequences, however, tabular datasets lack intrinsic spatial organization: features are treated as unordered dimensions, and their relationships must be inferred implicitly by the model. This limits the ability of vision architectures to exploit local structure and higher-order feature interactions in non-spatial biomedical data. Here we introduce Dynamic Feature Mapping (Dynomap), an end-to-end deep learning framework that learns a task-optimized spatial topology of features directly from data. Dynomap jointly optimizes feature placement and prediction through a fully differentiable rendering mechanism, without relying on heuristics, predefined groupings, or external priors. By transforming high-dimensional tabular vectors into learned feature maps, Dynomap enables vision-based models to operate effectively on unordered biomedical inputs. Across multiple clinical and biological datasets, Dynomap consistently outperformed classical machine learning, modern deep tabular models, and existing vector-to-image approaches. In liquid biopsy data, Dynomap organized clinically relevant gene signatures into coherent spatial patterns and improved multiclass cancer subtype prediction accuracy by up to 18%. In a Parkinson disease voice dataset, it clustered disease-associated acoustic descriptors and improved accuracy by up to 8%. Similar gains and interpretable feature organization were observed in additional biomedical datasets. These results establish Dynomap as a general strategy for bridging tabular and vision-based deep learning and for uncovering structured, clinically relevant patterns in high-dimensional biomedical data. 

---
# Generalizing Dynamics Modeling More Easily from Representation Perspective 

**Authors**: Yiming Wang, Zhengnan Zhang, Genghe Zhang, Jiawen Dan, Changchun Li, Chenlong Hu, Chris Nugent, Jun Liu, Ximing Li, Bo Yang  

**Link**: [PDF](https://arxiv.org/pdf/2603.22655)  

**Abstract**: Learning system dynamics from observations is a critical problem in many applications over various real-world complex systems, e.g., climate, ecology, and fluid systems. Recently, neural dynamics modeling method have become a prevalent solution that embeds the object's observations into a latent space before learning dynamics using neural methods such as neural Ordinary Differential Equations (ODE). Existing dynamics modeling methods induce a specific model for each observation of different complex systems, resulting in poor generalization across systems. Inspired by the great success of pre-trained models, we conduct a generalized Pre-trained Dynamics EncoDER (PDEDER) which can embed the original state observations into a latent space where the dynamics can be captured more easily. To conduct the generalized PDEDER, we pre-train any Pre-trained Language Model (PLM) by minimizing the Lyapunov exponent objective, which constrains the chaotic behavior of governing dynamics learned in the latent space. By penalizing the divergence of embedded observations, our PDEDER promotes locally stable and well-structured latent dynamics, thereby facilitating more effective dynamics modeling than in the original observation space. In addition, we incorporate reconstruction and forecasting objectives to mitigate the risk of obtaining an over-smoothed latent space. Specifically, we collect 152 sets of real-world and synthetic observations from 23 complex systems as pre-training corpora and employ them to pre-train PDEDER. Given any future dynamic observation, we can fine-tune PDEDER with any specific dynamics modeling method. We evaluate PDEDER on 12 dynamic systems by short/long-term forecasting under both in-domain and cross-domain settings, and the empirical results indicate the effectiveness and generalizability of PDEDER. 

---
# AwesomeLit: Towards Hypothesis Generation with Agent-Supported Literature Research 

**Authors**: Zefei Xie, Yuhan Guo, Kai Xu  

**Link**: [PDF](https://arxiv.org/pdf/2603.22648)  

**Abstract**: There are different goals for literature research, from understanding an unfamiliar topic to generate hypothesis for the next research project. The nature of literature research also varies according to user's familiarity level of the topic. For inexperienced researchers, identifying gaps in the existing literature and generating feasible hypothesis are crucial but challenging. While general ``deep research'' tools can be used, they are not designed for such use case, thus often not effective. In addition, the ``black box" nature and hallucination of Large Language Models (LLMs) often lead to distrust. In this paper, we introduce a human-agent collaborative visualization system AwesomeLit to address this need. It has several novel features: a transparent user-steerable agentic workflow; a dynamically generated query exploring tree, visualizing the exploration path and provenance; and a semantic similarity view, depicting the relationships between papers. It enables users to transition from general intentions to detailed research topics. Finally, a qualitative study involving several early researchers showed that AwesomeLit is effective in helping users explore unfamiliar topics, identify promising research directions, and improve confidence in research results. 

---
# Learning to Trust: How Humans Mentally Recalibrate AI Confidence Signals 

**Authors**: ZhaoBin Li, Mark Steyvers  

**Link**: [PDF](https://arxiv.org/pdf/2603.22634)  

**Abstract**: Productive human-AI collaboration requires appropriate reliance, yet contemporary AI systems are often miscalibrated, exhibiting systematic overconfidence or underconfidence. We investigate whether humans can learn to mentally recalibrate AI confidence signals through repeated experience. In a behavioral experiment (N = 200), participants predicted the AI's correctness across four AI calibration conditions: standard, overconfidence, underconfidence, and a counterintuitive "reverse confidence" mapping. Results demonstrate robust learning across all conditions, with participants significantly improving their accuracy, discrimination, and calibration alignment over 50 trials. We present a computational model utilizing a linear-in-log-odds (LLO) transformation and a Rescorla-Wagner learning rule to explain these dynamics. The model reveals that humans adapt by updating their baseline trust and confidence sensitivity, using asymmetric learning rates to prioritize the most informative errors. While humans can compensate for monotonic miscalibration, we identify a significant boundary in the reverse confidence scenario, where a substantial proportion of participants struggled to override initial inductive biases. These findings provide a mechanistic account of how humans adapt their trust in AI confidence signals through experience. 

---
# LGSE: Lexically Grounded Subword Embedding Initialization for Low-Resource Language Adaptation 

**Authors**: Hailay Teklehaymanot, Dren Fazlija, Wolfgang Nejdl  

**Link**: [PDF](https://arxiv.org/pdf/2603.22629)  

**Abstract**: Adapting pretrained language models to low-resource, morphologically rich languages remains a significant challenge. Existing vocabulary expansion methods typically rely on arbitrarily segmented subword units, resulting in fragmented lexical representations and loss of critical morphological information. To address this limitation, we propose the Lexically Grounded Subword Embedding Initialization (LGSE) framework, which introduces morphologically informed segmentation for initializing embeddings of novel tokens. Instead of using random vectors or arbitrary subwords, LGSE decomposes words into their constituent morphemes and constructs semantically coherent embeddings by averaging pretrained subword or FastText-based morpheme representations. When a token cannot be segmented into meaningful morphemes, its embedding is constructed using character n-gram representations to capture structural information. During Language-Adaptive Pretraining, we apply a regularization term that penalizes large deviations of newly introduced embeddings from their initialized values, preserving alignment with the original pretrained embedding space while enabling adaptation to the target language. To isolate the effect of initialization, we retain the original pre-trained model vocabulary and tokenizer and update only the new embeddings during adaptation. We evaluate LGSE on three NLP tasks: Question Answering, Named Entity Recognition, and Text Classification, in two morphologically rich, low-resource languages: Amharic and Tigrinya, where morphological segmentation resources are available. Experimental results show that LGSE consistently outperforms baseline methods across all tasks, demonstrating the effectiveness of morphologically grounded embedding initialization for improving representation quality in underrepresented languages. Project resources are available in the GitHub link. 

---
# Toward Faithful Segmentation Attribution via Benchmarking and Dual-Evidence Fusion 

**Authors**: Abu Noman Md Sakib, OFM Riaz Rahman Aranya, Kevin Desai, Zijie Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2603.22624)  

**Abstract**: Attribution maps for semantic segmentation are almost always judged by visual plausibility. Yet looking convincing does not guarantee that the highlighted pixels actually drive the model's prediction, nor that attribution credit stays within the target region. These questions require a dedicated evaluation protocol. We introduce a reproducible benchmark that tests intervention-based faithfulness, off-target leakage, perturbation robustness, and runtime on Pascal VOC and SBD across three pretrained backbones. To further demonstrate the benchmark, we propose Dual-Evidence Attribution (DEA), a lightweight correction that fuses gradient evidence with region-level intervention signals through agreement-weighted fusion. DEA increases emphasis where both sources agree and retains causal support when gradient responses are unstable. Across all completed runs, DEA consistently improves deletion-based faithfulness over gradient-only baselines and preserves strong robustness, at the cost of additional compute from intervention passes. The benchmark exposes a faithfulness-stability tradeoff among attribution families that is entirely hidden under visual evaluation, providing a foundation for principled method selection in segmentation explainability. Code is available at this https URL. 

---
# To Agree or To Be Right? The Grounding-Sycophancy Tradeoff in Medical Vision-Language Models 

**Authors**: OFM Riaz Rahman Aranya, Kevin Desai  

**Link**: [PDF](https://arxiv.org/pdf/2603.22623)  

**Abstract**: Vision-language models (VLMs) adapted to the medical domain have shown strong performance on visual question answering benchmarks, yet their robustness against two critical failure modes, hallucination and sycophancy, remains poorly understood, particularly in combination. We evaluate six VLMs (three general-purpose, three medical-specialist) on three medical VQA datasets and uncover a grounding-sycophancy tradeoff: models with the lowest hallucination propensity are the most sycophantic, while the most pressure-resistant model hallucinates more than all medical-specialist models. To characterize this tradeoff, we propose three metrics: L-VASE, a logit-space reformulation of VASE that avoids its double-normalization; CCS, a confidence-calibrated sycophancy score that penalizes high-confidence capitulation; and Clinical Safety Index (CSI), a unified safety index that combines grounding, autonomy, and calibration via a geometric mean. Across 1,151 test cases, no model achieves a CSI above 0.35, indicating that none of the evaluated 7-8B parameter VLMs is simultaneously well-grounded and robust to social pressure. Our findings suggest that joint evaluation of both properties is necessary before these models can be considered for clinical use. Code is available at this https URL 

---
# Causal Discovery in Action: Learning Chain-Reaction Mechanisms from Interventions 

**Authors**: Panayiotis Panayiotou, Özgür Şimşek  

**Link**: [PDF](https://arxiv.org/pdf/2603.22620)  

**Abstract**: Causal discovery is challenging in general dynamical systems because, without strong structural assumptions, the underlying causal graph may not be identifiable even from interventional data. However, many real-world systems exhibit directional, cascade-like structure, in which components activate sequentially and upstream failures suppress downstream effects. We study causal discovery in such chain-reaction systems and show that the causal structure is uniquely identifiable from blocking interventions that prevent individual components from activating. We propose a minimal estimator with finite-sample guarantees, achieving exponential error decay and logarithmic sample complexity. Experiments on synthetic models and diverse chain-reaction environments demonstrate reliable recovery from a few interventions, while observational heuristics fail in regimes with delayed or overlapping causal effects. 

---
# Do Consumers Accept AIs as Moral Compliance Agents? 

**Authors**: Greg Nyilasy, Abraham Ryan Ade Putra Hito, Jennifer Overbeck, Brock Bastian, Darren W. Dahl  

**Link**: [PDF](https://arxiv.org/pdf/2603.22617)  

**Abstract**: Consumers are generally resistant to Artificial Intelligence (AI) involvement in moral decision-making, perceiving moral agency as requiring uniquely human traits. This research investigates whether consumers might instead accept AIs in the role of moral compliance, where AI upholds pre-existing moral norms without exercising subjective discretion. Across five studies this research shows that consumers evaluate AI more positively than human agents in moral compliance roles. The findings reveal that this preference arises from inferences of AI's lack of ulterior motives, which are often attributed to human agents. While previous studies have focused on AI as a decision-maker, this work demonstrates the critical role of upholding pre-existing rules, a role in which AI is perceived to excel. These findings contribute to understanding consumer acceptance of moral AI and provide actionable insights for organizations seeking to leverage AI in ethical oversight. By positioning AI as a moral compliance agent, companies can address consumer skepticism, enhance trust, and improve perceptions of corporate ethicality. 

---
# Language Models Can Explain Visual Features via Steering 

**Authors**: Javier Ferrando, Enrique Lopez-Cuena, Pablo Agustin Martin-Torres, Daniel Hinjos, Anna Arias-Duart, Dario Garcia-Gasulla  

**Link**: [PDF](https://arxiv.org/pdf/2603.22593)  

**Abstract**: Sparse Autoencoders uncover thousands of features in vision models, yet explaining these features without requiring human intervention remains an open challenge. While previous work has proposed generating correlation-based explanations based on top activating input examples, we present a fundamentally different alternative based on causal interventions. We leverage the structure of Vision-Language Models and steer individual SAE features in the vision encoder after providing an empty image. Then, we prompt the language model to explain what it ``sees'', effectively eliciting the visual concept represented by each feature. Results show that Steering offers an scalable alternative that complements traditional approaches based on input examples, serving as a new axis for automated interpretability in vision models. Moreover, the quality of explanations improves consistently with the scale of the language model, highlighting our method as a promising direction for future research. Finally, we propose Steering-informed Top-k, a hybrid approach that combines the strengths of causal interventions and input-based approaches to achieve state-of-the-art explanation quality without additional computational cost. 

---
# flexvec: SQL Vector Retrieval with Programmatic Embedding Modulation 

**Authors**: Damian Delmas  

**Link**: [PDF](https://arxiv.org/pdf/2603.22587)  

**Abstract**: As AI agents become the primary consumers of retrieval APIs, there is an opportunity to expose more of the retrieval pipeline to the caller. flexvec is a retrieval kernel that exposes the embedding matrix and score array as a programmable surface, allowing arithmetic operations on both before selection. We refer to composing operations on this surface at query time as Programmatic Embedding Modulation (PEM). This paper describes a set of such operations and integrates them into a SQL interface via a query materializer that facilitates composable query primitives. On a production corpus of 240,000 chunks, three composed modulations execute in 19 ms end-to-end on a desktop CPU without approximate indexing. At one million chunks, the same operations execute in 82 ms. 

---
# Lie to Me: How Faithful Is Chain-of-Thought Reasoning in Reasoning Models? 

**Authors**: Richard J. Young  

**Link**: [PDF](https://arxiv.org/pdf/2603.22582)  

**Abstract**: Chain-of-thought (CoT) reasoning has been proposed as a transparency mechanism for large language models in safety-critical deployments, yet its effectiveness depends on faithfulness (whether models accurately verbalize the factors that actually influence their outputs), a property that prior evaluations have examined in only two proprietary models, finding acknowledgment rates as low as 25% for Claude 3.7 Sonnet and 39% for DeepSeek-R1. To extend this evaluation across the open-weight ecosystem, this study tests 12 open-weight reasoning models spanning 9 architectural families (7B-685B parameters) on 498 multiple-choice questions from MMLU and GPQA Diamond, injecting six categories of reasoning hints (sycophancy, consistency, visual pattern, metadata, grader hacking, and unethical information) and measuring the rate at which models acknowledge hint influence in their CoT when hints successfully alter answers. Across 41,832 inference runs, overall faithfulness rates range from 39.7% (Seed-1.6-Flash) to 89.9% (DeepSeek-V3.2-Speciale) across model families, with consistency hints (35.5%) and sycophancy hints (53.9%) exhibiting the lowest acknowledgment rates. Training methodology and model family predict faithfulness more strongly than parameter count, and keyword-based analysis reveals a striking gap between thinking-token acknowledgment (approximately 87.5%) and answer-text acknowledgment (approximately 28.6%), suggesting that models internally recognize hint influence but systematically suppress this acknowledgment in their outputs. These findings carry direct implications for the viability of CoT monitoring as a safety mechanism and suggest that faithfulness is not a fixed property of reasoning models but varies systematically with architecture, training method, and the nature of the influencing cue. 

---
# STRIATUM-CTF: A Protocol-Driven Agentic Framework for General-Purpose CTF Solving 

**Authors**: James Hugglestone, Samuel Jacob Chacko, Dawson Stoller, Ryan Schmidt, Xiuwen Liu  

**Link**: [PDF](https://arxiv.org/pdf/2603.22577)  

**Abstract**: Large Language Models (LLMs) have demonstrated potential in code generation, yet they struggle with the multi-step, stateful reasoning required for offensive cybersecurity operations. Existing research often relies on static benchmarks that fail to capture the dynamic nature of real-world vulnerabilities. In this work, we introduce STRIATUM-CTF (A Search-based Test-time Reasoning Inference Agent for Tactical Utility Maximization in Cybersecurity), a modular agentic framework built upon the Model Context Protocol (MCP). By standardizing tool interfaces for system introspection, decompilation, and runtime debugging, STRIATUM-CTF enables the agent to maintain a coherent context window across extended exploit trajectories.
We validate this approach not merely on synthetic datasets, but in a live competitive environment. Our system participated in a university-hosted Capture-the-Flag (CTF) competition in late 2025, where it operated autonomously to identify and exploit vulnerabilities in real-time. STRIATUM-CTF secured First Place, outperforming 21 human teams and demonstrating strong adaptability in a dynamic problem-solving setting. We analyze the agent's decision-making logs to show how MCP-based tool abstraction significantly reduces hallucination compared to naive prompting strategies. These results suggest that standardized context protocols are a critical path toward robust autonomous cyber-reasoning systems. 

---
# Ego2Web: A Web Agent Benchmark Grounded in Egocentric Videos 

**Authors**: Shoubin Yu, Lei Shu, Antoine Yang, Yao Fu, Srinivas Sunkara, Maria Wang, Jindong Chen, Mohit Bansal, Boqing Gong  

**Link**: [PDF](https://arxiv.org/pdf/2603.22529)  

**Abstract**: Multimodal AI agents are increasingly automating complex real-world workflows that involve online web execution. However, current web-agent benchmarks suffer from a critical limitation: they focus entirely on web-based interaction and perception, lacking grounding in the user's real-world physical surroundings. This limitation prevents evaluation in crucial scenarios, such as when an agent must use egocentric visual perception (e.g., via AR glasses) to recognize an object in the user's surroundings and then complete a related task online. To address this gap, we introduce Ego2Web, the first benchmark designed to bridge egocentric video perception and web agent execution. Ego2Web pairs real-world first-person video recordings with web tasks that require visual understanding, web task planning, and interaction in an online environment for successful completion. We utilize an automatic data-generation pipeline combined with human verification and refinement to curate well-constructed, high-quality video-task pairs across diverse web task types, including e-commerce, media retrieval, knowledge lookup, etc. To facilitate accurate and scalable evaluation for our benchmark, we also develop a novel LLM-as-a-Judge automatic evaluation method, Ego2WebJudge, which achieves approximately 84% agreement with human judgment, substantially higher than existing evaluation methods. Experiments with diverse SoTA agents on our Ego2Web show that their performance is weak, with substantial headroom across all task categories. We also conduct a comprehensive ablation study on task design, highlighting the necessity of accurate video understanding in the proposed task and the limitations of current agents. We hope Ego2Web can be a critical new resource for developing truly capable AI assistants that can seamlessly see, understand, and act across the physical and digital worlds. 

---
# GraphRAG for Engineering Diagrams: ChatP&ID Enables LLM Interaction with P&IDs 

**Authors**: Achmad Anggawirya Alimin, Artur M. Schweidtmann  

**Link**: [PDF](https://arxiv.org/pdf/2603.22528)  

**Abstract**: Large Language Models (LLMs) combined with Retrieval-Augmented Generation (RAG) and knowledge graphs offer new opportunities for interacting with engineering diagrams such as Piping and Instrumentation Diagrams (P&IDs). However, directly processing raw images or smart P&ID files with LLMs is often costly, inefficient, and prone to hallucinations. This work introduces ChatP&ID, an agentic framework that enables grounded and cost-effective natural-language interaction with P&IDs using Graph Retrieval-Augmented Generation (GraphRAG), a paradigm we refer to as GraphRAG for engineering diagrams. Smart P&IDs encoded in the DEXPI standard are transformed into structured knowledge graphs, which serve as the basis for graph-based retrieval and reasoning by LLM agents. This approach enables reliable querying of engineering diagrams while significantly reducing computational cost. Benchmarking across commercial LLM APIs (OpenAI, Anthropic) demonstrates that graph-based representations improve accuracy by 18% over raw image inputs and reduce token costs by 85% compared to directly ingesting smart P&ID files. While small open-source models still struggle to interpret knowledge graph formats and structured engineering data, integrating them with VectorRAG and PathRAG improves response accuracy by up to 40%. Notably, GPT-5-mini combined with ContextRAG achieves 91% accuracy at a cost of only $0.004 per task. The resulting ChatP&ID interface enables intuitive natural-language interaction with complex engineering diagrams and lays the groundwork for AI-assisted process engineering tasks such as Hazard and Operability Studies (HAZOP) and multi-agent analysis. 

---
# LLMON: An LLM-native Markup Language to Leverage Structure and Semantics at the LLM Interface 

**Authors**: Michael Hind, Basel Shbita, Bo Wu, Farhan Ahmed, Chad DeLuca, Nathan Fulton, David Cox, Dan Gutfreund  

**Link**: [PDF](https://arxiv.org/pdf/2603.22519)  

**Abstract**: Textual Large Language Models (LLMs) provide a simple and familiar interface: a string of text is used for both input and output. However, the information conveyed to an LLM often has a richer structure and semantics, which is not conveyed in a string. For example, most prompts contain both instructions ("Summarize this paper into a paragraph") and data (the paper to summarize), but these are usually not distinguished when passed to the model. This can lead to model confusion and security risks, such as prompt injection attacks.
This work addresses this shortcoming by introducing an LLM-native mark-up language, LLMON (LLM Object Notation, pronounced "Lemon"), that enables the structure and semantic metadata of the text to be communicated in a natural way to an LLM. This information can then be used during model training, model prompting, and inference implementation, leading to improvements in model accuracy, safety, and security. This is analogous to how programming language types can be used for many purposes, such as static checking, code generation, dynamic checking, and IDE highlighting.
We discuss the general design requirements of an LLM-native markup language, introduce the LLMON markup language and show how it meets these design requirements, describe how the information contained in a LLMON artifact can benefit model training and inference implementation, and provide some preliminary empirical evidence of its value for both of these use cases. We also discuss broader issues and research opportunities that are enabled with an LLM-native approach. 

---
# High Resolution Flood Extent Detection Using Deep Learning with Random Forest Derived Training Labels 

**Authors**: Azizbek Nuriddinov, Ebrahim Ahmadisharaf, Mohammad Reza Alizadeh  

**Link**: [PDF](https://arxiv.org/pdf/2603.22518)  

**Abstract**: Validation of flood models, used to support risk mitigation strategies, remains challenging due to limited observations during extreme events. High-frequency, high-resolution optical imagery (~3 m), such as PlanetScope, offers new opportunities for flood mapping, although applications remain limited by cloud cover and the lack of labeled training data during disasters. To address this, we develop a flood mapping framework that integrates PlanetScope optical imagery with topographic features using machine learning (ML) and deep learning (DL) algorithms. A Random Forest model was applied to expert-annotated flood masks to generate training labels for DL models, U-Net. Two U-Net models with ResNet18 backbone were trained using optical imagery only (4 bands) and optical imagery combined with Height Above Nearest Drainage (HAND) and topographic slope (6 bands). Hurricane Ida (September 2021), which caused catastrophic flooding across the eastern United States, including the New York City metropolitan area, was used as an example to evaluate the framework. Results demonstrate that the U-Net model with topographic features achieved very close performance to the optical-only configuration (F1=0.92 and IoU=0.85 by both modeling scenarios), indicating that HAND and slope provide only marginal value to inundation extent detection. The proposed framework offers a scalable and label-efficient approach for mapping inundation extent that enables modeling under data-scarce flood scenarios. 

---
# Do Large Language Models Reduce Research Novelty? Evidence from Information Systems Journals 

**Authors**: Ali Safari  

**Link**: [PDF](https://arxiv.org/pdf/2603.22510)  

**Abstract**: Large language models such as ChatGPT have increased scholarly output, but whether this productivity boost produces genuine intellectual advancement remains untested. I address this gap by measuring the semantic novelty of 13,847 articles published between 2020 and 2025 in 44 Information Systems journals. Using SPECTER2 embeddings, I operationalize novelty as the cosine distance between each paper and its nearest prior neighbors. A difference-in-differences design with the November 2022 release of ChatGPT as the treatment break reveals a heterogeneous pattern: authors affiliated with institutions in non-English-dominant countries show a 0.18 standard deviation decline in relative novelty compared to authors in English-dominant countries (beta = -0.176, p < 0.001), equivalent to a 7-percentile-point drop in the novelty distribution. This finding is robust across alternative novelty specifications, treatment break dates, and sub-samples, and survives a placebo test at a pre-treatment break. I interpret these results through the lens of construal level theory, proposing that LLMs function as proximity tools that shift researchers from abstract, exploratory thinking toward concrete, convention-following execution. The paper contributes to the growing debate on whether LLM-driven productivity gains come at the cost of intellectual diversity. 

---
# Tiny Inference-Time Scaling with Latent Verifiers 

**Authors**: Davide Bucciarelli, Evelyn Turri, Lorenzo Baraldi, Marcella Cornia, Lorenzo Baraldi, Rita Cucchiara  

**Link**: [PDF](https://arxiv.org/pdf/2603.22492)  

**Abstract**: Inference-time scaling has emerged as an effective way to improve generative models at test time by using a verifier to score and select candidate outputs. A common choice is to employ Multimodal Large Language Models (MLLMs) as verifiers, which can improve performance but introduce substantial inference-time cost. Indeed, diffusion pipelines operate in an autoencoder latent space to reduce computation, yet MLLM verifiers still require decoding candidates to pixel space and re-encoding them into the visual embedding space, leading to redundant and costly operations. In this work, we propose Verifier on Hidden States (VHS), a verifier that operates directly on intermediate hidden representations of Diffusion Transformer (DiT) single-step generators. VHS analyzes generator features without decoding to pixel space, thereby reducing the per-candidate verification cost while improving or matching the performance of MLLM-based competitors. We show that, under tiny inference budgets with only a small number of candidates per prompt, VHS enables more efficient inference-time scaling reducing joint generation-and-verification time by 63.3%, compute FLOPs by 51% and VRAM usage by 14.5% with respect to a standard MLLM verifier, achieving a +2.7% improvement on GenEval at the same inference-time budget. 

---
# Cognitive Training for Language Models: Towards General Capabilities via Cross-Entropy Games 

**Authors**: Clément Hongler, Franck Gabriel, Valentin Hartmann, Arthur Renard, Andrew Emil  

**Link**: [PDF](https://arxiv.org/pdf/2603.22479)  

**Abstract**: Defining a constructive process to build general capabilities for language models in an automatic manner is considered an open problem in artificial intelligence. Towards this, we consider the problem of building a curriculum of tasks that grows a model via relevant skill discovery. We provide a concrete framework for this task, using a family of tasks called cross-entropy games, which we postulate is universal in a suitable sense. We show that if it is possible to grow the curriculum for relevant skill discovery by iterating a greedy optimization algorithm, then, under natural assumptions, there is essentially only one meta-objective possible (up to a few hyperparameters). We call the resulting process cognitive training. We postulate that, given sufficiently capable language models as players and meta-samplers and sufficient training time, cognitive training provides a principled way to relevant skill discovery; and hence to the extent general capabilities are achievable via greedy curriculum learning, cognitive training would be a solution. 

---
# Functional Component Ablation Reveals Specialization Patterns in Hybrid Language Model Architectures 

**Authors**: Hector Borobia, Elies Seguí-Mas, Guillermina Tormo-Carbó  

**Link**: [PDF](https://arxiv.org/pdf/2603.22473)  

**Abstract**: Hybrid language models combining attention with state space models (SSMs) or linear attention offer improved efficiency, but whether both components are genuinely utilized remains unclear. We present a functional component ablation framework applied to two sub-1B hybrid models -- Qwen3.5-0.8B (sequential: Gated DeltaNet + softmax attention) and Falcon-H1-0.5B (parallel: Mamba-2 + attention) -- with a pure Transformer control (Qwen2.5-0.5B). Through group ablations, layer-wise sweeps, positional ablations, matched random controls, and perplexity analysis across five benchmarks, we establish four findings: (1) both component types are essential and neither is bypassed; (2) the alternative component (linear attention or SSM) is the primary language modeling backbone, causing >35,000x perplexity degradation when removed versus ~82x for attention; (3) component importance follows a positional gradient, with early layers being disproportionately critical; and (4) hybrid architectures exhibit 20-119x greater resilience to random layer removal than pure Transformers, revealing built-in functional redundancy between component types. These results provide actionable guidance for hybrid model compression, architecture design, and fault-tolerant deployment. 

---
# Stability-Preserving Online Adaptation of Neural Closed-loop Maps 

**Authors**: Danilo Saccani, Luca Furieri, Giancarlo Ferrari-Trecate  

**Link**: [PDF](https://arxiv.org/pdf/2603.22469)  

**Abstract**: The growing complexity of modern control tasks calls for controllers that can react online as objectives and disturbances change, while preserving closed-loop stability. Recent approaches for improving the performance of nonlinear systems while preserving closed-loop stability rely on time-invariant recurrent neural-network controllers, but offer no principled way to update the controller during operation. Most importantly, switching from one stabilizing policy to another can itself destabilize the closed-loop. We address this problem by introducing a stability-preserving update mechanism for nonlinear, neural-network-based controllers. Each controller is modeled as a causal operator with bounded $\ell_p$-gain, and we derive gain-based conditions under which the controller may be updated online. These conditions yield two practical update schemes, time-scheduled and state-triggered, that guarantee the closed-loop remains $\ell_p$-stable after any number of updates. Our analysis further shows that stability is decoupled from controller optimality, allowing approximate or early-stopped controller synthesis. We demonstrate the approach on nonlinear systems with time-varying objectives and disturbances, and show consistent performance improvements over static and naive online baselines while guaranteeing stability. 

---
# Color When It Counts: Grayscale-Guided Online Triggering for Always-On Streaming Video Sensing 

**Authors**: Weitong Cai, Hang Zhang, Yukai Huang, Shitong Sun, Jiankang Deng, Songcen Xu, Jifei Song, Zhensong Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2603.22466)  

**Abstract**: Always-on sensing is essential for next-generation edge/wearable AI systems, yet continuous high-fidelity RGB video capture remains prohibitively expensive for resource-constrained mobile and edge platforms. We present a new paradigm for efficient streaming video understanding: grayscale-always, color-on-demand. Through preliminary studies, we discover that color is not always necessary. Sparse RGB frames suffice for comparable performance when temporal structure is preserved via continuous grayscale streams. Building on this insight, we propose ColorTrigger, an online training-free trigger that selectively activates color capture based on windowed grayscale affinity analysis. Designed for real-time edge deployment, ColorTrigger uses lightweight quadratic programming to detect chromatic redundancy causally, coupled with credit-budgeted control and dynamic token routing to jointly reduce sensing and inference costs. On streaming video understanding benchmarks, ColorTrigger achieves 91.6% of full-color baseline performance while using only 8.1% RGB frames, demonstrating substantial color redundancy in natural videos and enabling practical always-on video sensing on resource-constrained devices. 

---
# LLM-guided headline rewriting for clickability enhancement without clickbait 

**Authors**: Yehudit Aperstein, Linoy Halifa, Sagiv Bar, Alexander Apartsin  

**Link**: [PDF](https://arxiv.org/pdf/2603.22459)  

**Abstract**: Enhancing reader engagement while preserving informational fidelity is a central challenge in controllable text generation for news media. Optimizing news headlines for reader engagement is often conflated with clickbait, resulting in exaggerated or misleading phrasing that undermines editorial trust. We frame clickbait not as a separate stylistic category, but as an extreme outcome of disproportionate amplification of otherwise legitimate engagement cues. Based on this view, we formulate headline rewriting as a controllable generation problem, where specific engagement-oriented linguistic attributes are selectively strengthened under explicit constraints on semantic faithfulness and proportional emphasis. We present a guided headline rewriting framework built on a large language model (LLM) that uses the Future Discriminators for Generation (FUDGE) paradigm for inference-time control. The LLM is steered by two auxiliary guide models: (1) a clickbait scoring model that provides negative guidance to suppress excessive stylistic amplification, and (2) an engagement-attribute model that provides positive guidance aligned with target clickability objectives. Both guides are trained on neutral headlines drawn from a curated real-world news corpus. At the same time, clickbait variants are generated synthetically by rewriting these original headlines using an LLM under controlled activation of predefined engagement tactics. By adjusting guidance weights at inference time, the system generates headlines along a continuum from neutral paraphrases to more engaging yet editorially acceptable formulations. The proposed framework provides a principled approach for studying the trade-off between attractiveness, semantic preservation, and clickbait avoidance, and supports responsible LLM-based headline optimization in journalistic settings. 

---
# Sparse but Critical: A Token-Level Analysis of Distributional Shifts in RLVR Fine-Tuning of LLMs 

**Authors**: Haoming Meng, Kexin Huang, Shaohang Wei, Chiyu Ma, Shuo Yang, Xue Wang, Guoyin Wang, Bolin Ding, Jingren Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2603.22446)  

**Abstract**: Reinforcement learning with verifiable rewards (RLVR) has significantly improved reasoning in large language models (LLMs), yet the token-level mechanisms underlying these improvements remain unclear. We present a systematic empirical study of RLVR's distributional effects organized around three main analyses: (1) token-level characterization of distributional shifts between base and RL models, (2) the impact of token-level distributional shifts on sequence-level reasoning performance through cross-sampling interventions, and (3) fine-grained mechanics of these shifts at the token level. We find that RL fine-tuning induces highly sparse and targeted changes, with only a small fraction of token distributions exhibiting meaningful divergence between the base and RL policies. We further characterize the structure and evolution of these shifts through analyses of token entropy, positional concentration, and reallocation of probability mass. To assess the functional importance of these sparse changes, we conduct cross-sampling experiments that selectively swap token choices between the base and RL models with varying intervention budgets. We show that inserting only a small fraction of RL-sampled tokens into base generations progressively recovers RL performance gains, while injecting a similarly small number of base token choices into otherwise RL-generated sequences collapses performance to base levels, isolating a small set of token-level decisions directly responsible for RLVR's performance gains. Finally, we explore divergence-weighted variants of the advantage signal as a diagnostic intervention, finding that they can yield improvements over baselines. Together, our results shed light on the distributional changes induced by RLVR and provide a fine-grained, token-level lens for understanding RLVR fine-tuning as a targeted refinement process. 

---
# CaP-X: A Framework for Benchmarking and Improving Coding Agents for Robot Manipulation 

**Authors**: Max Fu, Justin Yu, Karim El-Refai, Ethan Kou, Haoru Xue, Huang Huang, Wenli Xiao, Guanzhi Wang, Fei-Fei Li, Guanya Shi, Jiajun Wu, Shankar Sastry, Yuke Zhu, Ken Goldberg, Linxi "Jim" Fan  

**Link**: [PDF](https://arxiv.org/pdf/2603.22435)  

**Abstract**: "Code-as-Policy" considers how executable code can complement data-intensive Vision-Language-Action (VLA) methods, yet their effectiveness as autonomous controllers for embodied manipulation remains underexplored. We present CaP-X, an open-access framework for systematically studying Code-as-Policy agents in robot manipulation. At its core is CaP-Gym, an interactive environment in which agents control robots by synthesizing and executing programs that compose perception and control primitives. Building on this foundation, CaP-Bench evaluates frontier language and vision-language models across varying levels of abstraction, interaction, and perceptual grounding. Across 12 models, CaP-Bench reveals a consistent trend: performance improves with human-crafted abstractions but degrades as these priors are removed, exposing a dependence on designer scaffolding. At the same time, we observe that this gap can be mitigated through scaling agentic test-time computation--through multi-turn interaction, structured execution feedback, visual differencing, automatic skill synthesis, and ensembled reasoning--substantially improves robustness even when agents operate over low-level primitives. These findings allow us to derive CaP-Agent0, a training-free framework that recovers human-level reliability on several manipulation tasks in simulation and on real embodiments. We further introduce CaP-RL, showing reinforcement learning with verifiable rewards improves success rates and transfers from sim2real with minimal gap. Together, CaP-X provides a principled, open-access platform for advancing embodied coding agents. 

---
# Latent Style-based Quantum Wasserstein GAN for Drug Design 

**Authors**: Julien Baglio, Yacine Haddad, Richard Polifka  

**Link**: [PDF](https://arxiv.org/pdf/2603.22399)  

**Abstract**: The development of new drugs is a tedious, time-consuming, and expensive process, for which the average costs are estimated to be up to around $2.5 billion. The first step in this long process is the design of the new drug, for which de novo drug design, assisted by artificial intelligence, has blossomed in recent years and revolutionized the field. In particular, generative artificial intelligence has delivered promising results in drug discovery and development, reducing costs and the time to solution. However, classical generative models, such as generative adversarial networks (GANs), are difficult to train due to barren plateaus and prone to mode collapse. Quantum computing may be an avenue to overcome these issues and provide models with fewer parameters, thereby enhancing the generalizability of GANs. We propose a new style-based quantum GAN (QGAN) architecture for drug design that implements noise encoding at every rotational gate of the circuit and a gradient penalty in the loss function to mitigate mode collapse. Our pipeline employs a variational autoencoder to represent the molecular structure in a latent space, which is then used as input to our QGAN. Our baseline model runs on up to 15 qubits to validate our architecture on quantum simulators, and a 156-qubit IBM Heron quantum computer in the five-qubit setup is used for inference to investigate the effects of using real quantum hardware on the analysis. We benchmark our results against classical models as provided by the MOSES benchmark suite. 

---
# Learning When to Act: Interval-Aware Reinforcement Learning with Predictive Temporal Structure 

**Authors**: Davide Di Gioia  

**Link**: [PDF](https://arxiv.org/pdf/2603.22384)  

**Abstract**: Autonomous agents operating in continuous environments must decide not only what to do, but when to act. We introduce a lightweight adaptive temporal control system that learns the optimal interval between cognitive ticks from experience, replacing ad hoc biologically inspired timers with a principled learned policy. The policy state is augmented with a predictive hyperbolic spread signal (a "curvature signal" shorthand) derived from hyperbolic geometry: the mean pairwise Poincare distance among n sampled futures embedded in the Poincare ball. High spread indicates a branching, uncertain future and drives the agent to act sooner; low spread signals predictability and permits longer rest intervals. We further propose an interval-aware reward that explicitly penalises inefficiency relative to the chosen wait time, correcting a systematic credit-assignment failure of naive outcome-based rewards in timing problems. We additionally introduce a joint spatio-temporal embedding (ATCPG-ST) that concatenates independently normalised state and position projections in the Poincare ball; spatial trajectory divergence provides an independent timing signal unavailable to the state-only variant (ATCPG-SO). This extension raises mean hyperbolic spread (kappa) from 1.88 to 3.37 and yields a further 5.8 percent efficiency gain over the state-only baseline. Ablation experiments across five random seeds demonstrate that (i) learning is the dominant efficiency factor (54.8 percent over no-learning), (ii) hyperbolic spread provides significant complementary gain (26.2 percent over geometry-free control), (iii) the combined system achieves 22.8 percent efficiency over the fixed-interval baseline, and (iv) adding spatial position information to the spread embedding yields an additional 5.8 percent. 

---
# Symbolic Graph Networks for Robust PDE Discovery from Noisy Sparse Data 

**Authors**: Xingyu Chen, Junxiu An, Jun Guo, Yuqian Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2603.22380)  

**Abstract**: Data-driven discovery of partial differential equations (PDEs) offers a promising paradigm for uncovering governing physical laws from observational data. However, in practical scenarios, measurements are often contaminated by noise and limited by sparse sampling, which poses significant challenges to existing approaches based on numerical differentiation or integral formulations. In this work, we propose a Symbolic Graph Network (SGN) framework for PDE discovery under noisy and sparse conditions. Instead of relying on local differential approximations, SGN leverages graph message passing to model spatial interactions, providing a non-local representation that is less sensitive to high frequency noise. Based on this representation, the learned latent features are further processed by a symbolic regression module to extract interpretable mathematical expressions. We evaluate the proposed method on several benchmark systems, including the wave equation, convection-diffusion equation, and incompressible Navier-Stokes equations. Experimental results show that SGN can recover meaningful governing relations or solution forms under varying noise levels, and demonstrates improved robustness compared to baseline methods in sparse and noisy settings. These results suggest that combining graph-based representations with symbolic regression provides a viable direction for robust data-driven discovery of physical laws from imperfect observations. The code is available at this https URL 

---
# Instruction-Tuned, but Not More Verifiable Instruction-Following: A Cross-Task Diagnosis for LoRA Adapters 

**Authors**: Junyi Zou  

**Link**: [PDF](https://arxiv.org/pdf/2603.22379)  

**Abstract**: Adapters are often selected and deployed based on nominal labels (e.g., instruction-tuned), which implicitly suggest what capability improves after adaptation. We test whether nominal training objectives reliably align with realized cross-task capability gains by evaluating the same LoRA adapter across tasks. Our strongest evidence is tied to strict, automatically verifiable instruction following as measured by IFEval: across multiple seeds, base models, and LoRA settings, nominal labels recurrently but not universally fail to predict improvements on this verifiable target, with clear configuration sensitivity including a near-zero or negative case. As an illustrative strongest-case example in a controlled instruction-versus-numeric setting, an instruction-tuned adapter substantially improves off-target NM-based numeric benchmark performance from 0.133 to 0.632 while not improving verifiable instruction following on IFEval (ILA: 0.313 to 0.271; PLA: 0.250 to 0.143; values rounded to three decimals). We refer to this nominal-versus-realized mismatch pattern as capability drift as a descriptive label. The mismatch is visible in the raw cross-task performance matrix; we use a drift score only as a compact summary in the same units as the underlying metrics, not as a new formal metric contribution. Evidence from broader instruction-following benchmarks is benchmark-dependent and mixed, reflecting heterogeneity in how instruction following is operationalized; we therefore do not treat cross-benchmark agreement as a premise. Overall, the practical takeaway is to perform routine cross-task evaluation before deployment and to avoid treating nominal labels as reliable capability proxies. 

---
# Abnormalities and Disease Detection in Gastro-Intestinal Tract Images 

**Authors**: Zeshan Khan, Muhammad Atif Tahir  

**Link**: [PDF](https://arxiv.org/pdf/2603.22378)  

**Abstract**: Gastrointestinal (GI) tract image analysis plays a crucial role in medical diagnosis. This research addresses the challenge of accurately classifying and segmenting GI images for real-time applications, where traditional methods often struggle due to the diversity and complexity of abnormalities. The high computational demands of this domain require efficient and adaptable solutions.
This PhD thesis presents a multifaceted approach to GI image analysis. Initially, texture-based feature extraction and classification methods were explored, achieving high processing speed (over 4000 FPS) and strong performance (F1-score: 0.76, Accuracy: 0.98) on the Kvasir V2 dataset.
The study then transitions to deep learning, where an optimized model combined with data bagging techniques improved performance, reaching an accuracy of 0.92 and an F1-score of 0.60 on the HyperKvasir dataset, and an F1-score of 0.88 on Kvasir V2.
To support real-time detection, a streamlined neural network integrating texture and local binary patterns was developed. By addressing inter-class similarity and intra-class variation through a learned threshold, the system achieved 41 FPS with high accuracy (0.99) and an F1-score of 0.91 on HyperKvasir.
Additionally, two segmentation tools are proposed to enhance usability, leveraging Depth-Wise Separable Convolution and neural network ensembles for improved detection, particularly in low-FPS scenarios.
Overall, this research introduces novel and adaptable methodologies, progressing from traditional texture-based techniques to deep learning and ensemble approaches, providing a comprehensive framework for advancing GI image analysis. 

---
# AI Co-Scientist for Ranking: Discovering Novel Search Ranking Models alongside LLM-based AI Agents with Cloud Computing Access 

**Authors**: Liwei Wu, Cho-Jui Hsieh  

**Link**: [PDF](https://arxiv.org/pdf/2603.22376)  

**Abstract**: Recent advances in AI agents for software engineering and scientific discovery have demonstrated remarkable capabilities, yet their application to developing novel ranking models in commercial search engines remains unexplored. In this paper, we present an AI Co-Scientist framework that automates the full search ranking research pipeline: from idea generation to code implementation and GPU training job scheduling with expert in the loop. Our approach strategically employs single-LLM agents for routine tasks while leveraging multi-LLM consensus agents (GPT 5.2, Gemini Pro 3, and Claude Opus 4.5) for challenging phases such as results analysis and idea generation. To our knowledge, this is the first study in the ranking community to utilize an AI Co-Scientist framework for algorithmic research. We demonstrate that this framework discovered a novel technique for handling sequence features, with all model enhancements produced automatically, yielding substantial offline performance improvements. Our findings suggest that AI systems can discover ranking architectures comparable to those developed by human experts while significantly reducing routine research workloads. 

---
# Three Creates All: You Only Sample 3 Steps 

**Authors**: Yuren Cai, Guangyi Wang, Zongqing Li, Li Li, Zhihui Liu, Songzhi Su  

**Link**: [PDF](https://arxiv.org/pdf/2603.22375)  

**Abstract**: Diffusion models deliver high-fidelity generation but remain slow at inference time due to many sequential network evaluations. We find that standard timestep conditioning becomes a key bottleneck for few-step sampling. Motivated by layer-dependent denoising dynamics, we propose Multi-layer Time Embedding Optimization (MTEO), which freeze the pretrained diffusion backbone and distill a small set of step-wise, layer-wise time embeddings from reference trajectories. MTEO is plug-and-play with existing ODE solvers, adds no inference-time overhead, and trains only a tiny fraction of parameters. Extensive experiments across diverse datasets and backbones show state-of-the-art performance in the few-step sampling and substantially narrow the gap between distillation-based and lightweight methods. Code will be available. 

---
# Rethinking Multimodal Fusion for Time Series: Auxiliary Modalities Need Constrained Fusion 

**Authors**: Seunghan Lee, Jun Seo, Jaehoon Lee, Sungdong Yoo, Minjae Kim, Tae Yoon Lim, Dongwan Kang, Hwanil Choi, SoonYoung Lee, Wonbin Ahn  

**Link**: [PDF](https://arxiv.org/pdf/2603.22372)  

**Abstract**: Recent advances in multimodal learning have motivated the integration of auxiliary modalities such as text or vision into time series (TS) forecasting. However, most existing methods provide limited gains, often improving performance only in specific datasets or relying on architecture-specific designs that limit generalization. In this paper, we show that multimodal models with naive fusion strategies (e.g., simple addition or concatenation) often underperform unimodal TS models, which we attribute to the uncontrolled integration of auxiliary modalities which may introduce irrelevant information. Motivated by this observation, we explore various constrained fusion methods designed to control such integration and find that they consistently outperform naive fusion methods. Furthermore, we propose Controlled Fusion Adapter (CFA), a simple plug-in method that enables controlled cross-modal interactions without modifying the TS backbone, integrating only relevant textual information aligned with TS dynamics. CFA employs low-rank adapters to filter irrelevant textual information before fusing it into temporal representations. We conduct over 20K experiments across various datasets and TS/text models, demonstrating the effectiveness of the constrained fusion methods including CFA. Code is publicly available at: this https URL. 

---
# FAAR: Format-Aware Adaptive Rounding for NVFP4 

**Authors**: Hanglin Li, Shuchang Tian, Chen Lin, Zhiyong Zhao, Kun Zhan  

**Link**: [PDF](https://arxiv.org/pdf/2603.22370)  

**Abstract**: Deploying large language models (LLMs) on edge devices requires extremely low-bit quantization. Ultra-low precision formats such as NVFP4 offer a promising solution for reducing memory footprint and accelerating computation. However, existing quantization methods typically rely on conventional rounding strategies and fail to account for the non-uniformity of the NVFP4 numerical grid, resulting in suboptimal rounding decisions and amplified quantization errors. To address this, we propose Format-Aware Adaptive Rounding (FAAR), a learnable rounding strategy tailored for the NVFP4 format. Unlike conventional quantization paradigms, FAAR explicitly incorporates the non-uniform NVFP4 grid into the optimization process. By adaptively adjusting rounding decisions guided by loss gradients, our method effectively approximates the theoretically optimal quantization. To complement FAAR, we introduce a 2-stages Format Alignment (2FA) fine-tuning scheme that aligns LLM parameters layer-by-layer to the NVFP4 numerical space, further narrowing the performance gap. Remarkably, this learnable optimization incurs a minimal training overhead of only 4 GPU hours on Llama3-1B. Extensive experiments demonstrate the effectiveness of our approach. Compared with Round-to-Nearest (RTN), our method reduces perplexity on WikiText-2 from 14.28 to 12.60 on Llama3-1B and from 23.06 to 21.27 on Qwen3-1.7B. Additionally, our method consistently outperforms state-of-the-art approaches across various zero-shot downstream tasks. 

---
# SynLeaF: A Dual-Stage Multimodal Fusion Framework for Synthetic Lethality Prediction Across Pan- and Single-Cancer Contexts 

**Authors**: Zheming Xing, Siyuan Zhou, Ruinan Wang, Rui Han, Shiming Zhang, Shiqu Chen, Yurui Huang, Jiahao Ma, Yifan Chen, Xuan Wang, Yadong Wang, Junyi Li  

**Link**: [PDF](https://arxiv.org/pdf/2603.22369)  

**Abstract**: Accurate prediction of synthetic lethality (SL) is important for guiding the development of cancer drugs and therapies. SL prediction faces significant challenges in the effective fusion of heterogeneous multi-source data. Existing multimodal methods often suffer from "modality laziness" due to disparate convergence speeds, which hinders the exploitation of complementary information. This is also one reason why most existing SL prediction models cannot perform well on both pan-cancer and single-cancer SL pair prediction. In this study, we propose SynLeaF, a dual-stage multimodal fusion framework for SL prediction across pan- and single-cancer contexts. The framework employs a VAE-based cross-encoder with a product of experts mechanism to fuse four omics data types (gene expression, mutation, methylation, and CNV), while simultaneously utilizing a relational graph convolutional network to capture structured gene representations from biomedical knowledge graphs. To mitigate modality laziness, SynLeaF introduces a dual-stage training mechanism employing featurelevel knowledge distillation with adaptive uni-modal teacher and ensemble strategies. In extensive experiments across eight specific cancer types and a pancancer dataset, SynLeaF achieves superior performance in 17 out of 19 scenarios. Ablation studies and gradient analyses further validate the critical contributions of the proposed fusion and distillation mechanisms to model robustness and generalization. To facilitate community use, a web server is available at this https URL. 

---
# When Visuals Aren't the Problem: Evaluating Vision-Language Models on Misleading Data Visualizations 

**Authors**: Harsh Nishant Lalai, Raj Sanjay Shah, Hanspeter Pfister, Sashank Varma, Grace Guo  

**Link**: [PDF](https://arxiv.org/pdf/2603.22368)  

**Abstract**: Visualizations help communicate data insights, but deceptive data representations can distort their interpretation and propagate misinformation. While recent Vision Language Models (VLMs) perform well on many chart understanding tasks, their ability to detect misleading visualizations, especially when deception arises from subtle reasoning errors in captions, remains poorly understood. Here, we evaluate VLMs on misleading visualization-caption pairs grounded in a fine-grained taxonomy of reasoning errors (e.g., Cherry-picking, Causal inference) and visualization design errors (e.g., Truncated axis, Dual axis, inappropriate encodings). To this end, we develop a benchmark that combines real-world visualization with human-authored, curated misleading captions designed to elicit specific reasoning and visualization error types, enabling controlled analysis across error categories and modalities of misleadingness. Evaluating many commercial and open-source VLMs, we find that models detect visual design errors substantially more reliably than reasoning-based misinformation, and frequently misclassify non-misleading visualizations as deceptive. Overall, our work fills a gap between coarse detection of misleading content and the attribution of the specific reasoning or visualization errors that give rise to it. 

---
# Reasoner-Executor-Synthesizer: Scalable Agentic Architecture with Static O(1) Context Window 

**Authors**: Ivan Dobrovolskyi  

**Link**: [PDF](https://arxiv.org/pdf/2603.22367)  

**Abstract**: Large Language Models (LLMs) deployed as autonomous agents commonly use Retrieval-Augmented Generation (RAG), feeding retrieved documents into the context window, which creates two problems: the risk of hallucination grows with context length, and token cost scales linearly with dataset size. We propose the Reasoner-Executor-Synthesizer (RES) architecture, a three-layer design that strictly separates intent parsing (Reasoner), deterministic data retrieval and aggregation (Executor), and narrative generation (Synthesizer). The Executor uses zero LLM tokens and passes only fixed-size statistical summaries to the Synthesizer. We formally prove that RES achieves O(1) token complexity with respect to dataset size, and validate this on ScholarSearch, a scholarly research assistant backed by the Crossref API (130M+ articles). Across 100 benchmark runs, RES achieves a mean token cost of 1,574 tokens regardless of whether the dataset contains 42,000 or 16.3 million articles. The architecture eliminates data hallucination by construction: the LLM never sees raw records. KEYWORDS LLM agents; agentic architecture; hallucination elimination; token optimization; context window; retrieval-augmented generation; deterministic execution; scholarly metadata; Crossref API; O(1) complexity. 

---
# Modeling Quantum Federated Autoencoder for Anomaly Detection in IoT Networks 

**Authors**: Devashish Chaudhary, Sutharshan Rajasegarar, Shiva Raj Pokhrel  

**Link**: [PDF](https://arxiv.org/pdf/2603.22366)  

**Abstract**: We propose a Quantum Federated Autoencoder for Anomaly Detection, a framework that leverages quantum federated learning for efficient, secure, and distributed processing in IoT networks. By harnessing quantum autoencoders for high-dimensional feature representation and federated learning for decentralized model training, the approach transforms localized learning on edge devices without requiring transmission of raw data, thereby preserving privacy and minimizing communication overhead. The model leverages quantum advantage in pattern recognition to enhance detection sensitivity, particularly in complex and dynamic IoT network traffic. Experiments on a real-world IoT dataset show that the proposed method delivers anomaly detection accuracy and robustness comparable to centralized approaches, while ensuring data privacy. 

---
# Q-AGNN: Quantum-Enhanced Attentive Graph Neural Network for Intrusion Detection 

**Authors**: Devashish Chaudhary, Sutharshan Rajasegarar, Shiva Raj Pokhrel  

**Link**: [PDF](https://arxiv.org/pdf/2603.22365)  

**Abstract**: With the rapid growth of interconnected devices, accurately detecting malicious activities in network traffic has become increasingly challenging. Most existing deep learning-based intrusion detection systems treat network flows as independent instances, thereby failing to exploit the relational dependencies inherent in network communications. To address this limitation, we propose Q-AGNN, a Quantum-Enhanced Attentive Graph Neural Network for intrusion detection, where network flows are modeled as nodes and edges represent similarity relationships. Q-AGNN leverages parameterized quantum circuits (PQCs) to encode multi-hop neighborhood information into a high-dimensional latent space, inducing a bounded quantum feature map that implements a second-order polynomial graph filter in a quantum-induced Hilbert space. An attention mechanism is subsequently applied to adaptively weight the quantum-enhanced embeddings, allowing the model to focus on the most influential nodes contributing to anomalous behavior. Extensive experiments conducted on four benchmark intrusion detection datasets demonstrate that Q-AGNN achieves competitive or superior detection performance compared to state-of-the-art graph-based methods, while consistently maintaining low false positive rates under hardware-calibrated noise conditions. Moreover, we also executed the Q-AGNN framework on actual IBM quantum hardware to demonstrate the practical operability of the proposed pipeline under real NISQ conditions. These results highlight the effectiveness of integrating quantum-enhanced representations with attention mechanisms for graph-based intrusion detection and underscore the potential of hybrid quantum-classical learning frameworks in cybersecurity applications. 

---
# MCLR: Improving Conditional Modeling in Visual Generative Models via Inter-Class Likelihood-Ratio Maximization and Establishing the Equivalence between Classifier-Free Guidance and Alignment Objectives 

**Authors**: Xiang Li, Yixuan Jia, Xiao Li, Jeffrey A. Fessler, Rongrong Wang, Qing Qu  

**Link**: [PDF](https://arxiv.org/pdf/2603.22364)  

**Abstract**: Diffusion models have achieved state-of-the-art performance in generative modeling, but their success often relies heavily on classifier-free guidance (CFG), an inference-time heuristic that modifies the sampling trajectory. From a theoretical perspective, diffusion models trained with standard denoising score matching (DSM) are expected to recover the target data distribution, raising the question of why inference-time guidance is necessary in practice. In this work, we ask whether the DSM training objective can be modified in a principled manner such that standard reverse-time sampling, without inference-time guidance, yields effects comparable to CFG. We identify insufficient inter-class separation as a key limitation of standard diffusion models. To address this, we propose MCLR, a principled alignment objective that explicitly maximizes inter-class likelihood-ratios during training. Models fine-tuned with MCLR exhibit CFG-like improvements under standard sampling, achieving comparable qualitative and quantitative gains without requiring inference-time guidance. Beyond empirical benefits, we provide a theoretical result showing that the CFG-guided score is exactly the optimal solution to a weighted MCLR objective. This establishes a formal equivalence between classifier-free guidance and alignment-based objectives, offering a mechanistic interpretation of CFG. 

---
# Early Discoveries of Algorithmist I: Promise of Provable Algorithm Synthesis at Scale 

**Authors**: Janardhan Kulkarni  

**Link**: [PDF](https://arxiv.org/pdf/2603.22363)  

**Abstract**: Designing algorithms with provable guarantees that also work well in practice remains difficult, requiring both mathematical reasoning and careful implementation. Existing approaches that bridge worst-case theory and empirical performance, such as beyond-worst-case analysis and data-driven algorithm selection, typically assume prior distributional knowledge or restrict attention to a fixed pool of algorithms. Recent progress in LLMs suggests a new possibility: provable algorithm synthesis on the fly. To study this, we built Algorithmist, an autonomous researcher agent on top of GitHub Copilot that runs a multi-agent research-and-review loop, with separate stages for idea generation, algorithm and proof development, proof-guided implementation, and review of proofs, code, and their alignment. We evaluate Algorithmist on research-level tasks in private data analysis and clustering. When asked to design practical methods that jointly satisfy privacy, approximation, and interpretability requirements, it produced provably sound and empirically effective algorithms, together with research-style writeups and audited implementations. It also found improved algorithms in some settings, explained principled barriers in others, and uncovered a subtle proof bug in prior published work. More broadly, our results suggest a new paradigm in which LLM systems generate research-paper-quality algorithmic artifacts tailored to each dataset and deployment setting. They also point to a proof-first code-synthesis paradigm, in which code is developed alongside a structured natural-language proof intermediate representation and kept aligned with it throughout synthesis. 

---
# Unveiling the Mechanism of Continuous Representation Full-Waveform Inversion: A Wave Based Neural Tangent Kernel Framework 

**Authors**: Ruihua Chen, Yisi Luo, Bangyu Wu, Deyu Meng  

**Link**: [PDF](https://arxiv.org/pdf/2603.22362)  

**Abstract**: Full-waveform inversion (FWI) estimates physical parameters in the wave equation from limited measurements and has been widely applied in geophysical exploration, medical imaging, and non-destructive testing. Conventional FWI methods are limited by their notorious sensitivity to the accuracy of the initial models. Recent progress in continuous representation FWI (CR-FWI) demonstrates that representing parameter models with a coordinate-based neural network, such as implicit neural representation (INR), can mitigate the dependence on initial models. However, its underlying mechanism remains unclear, and INR-based FWI shows slower high-frequency convergence. In this work, we investigate the general CR-FWI framework and develop a unified theoretical understanding by extending the neural tangent kernel (NTK) for FWI to establish a wave-based NTK framework. Unlike standard NTK, our analysis reveals that wave-based NTK is not constant, both at initialization and during training, due to the inherent nonlinearity of FWI. We further show that the eigenvalue decay behavior of the wave-based NTK can explain why CR-FWI alleviates the dependency on initial models and shows slower high-frequency convergence. Building on these insights, we propose several CR-FWI methods with tailored eigenvalue decay properties for FWI, including a novel hybrid representation combining INR and multi-resolution grid (termed IG-FWI) that achieves a more balanced trade-off between robustness and high-frequency convergence rate. Applications in geophysical exploration on Marmousi, 2D SEG/EAGE Salt and Overthrust, 2004 BP model, and the more realistic 2014 Chevron models show the superior performance of our proposed methods compared to conventional FWI and existing INR-based FWI methods. 

---
# Bridging neuroscience and AI: adaptive, culturally sensitive technologies transforming aphasia rehabilitation 

**Authors**: Andreea I. Niculescu, Jochen Ehnes, Minghui Dong  

**Link**: [PDF](https://arxiv.org/pdf/2603.22357)  

**Abstract**: Aphasia, a language impairment primarily resulting from stroke or brain injury, profoundly disrupts communication and everyday functioning. Despite advances in speech therapy, barriers such as limited therapist availability and the scarcity of personalized, culturally relevant tools continue to hinder optimal rehabilitation outcomes. This paper reviews recent developments in neurocognitive research and language technologies that contribute to the diagnosis and therapy of aphasia. Drawing on findings from our ethnographic field study, we introduce two digital therapy prototypes designed to reflect local linguistic diversity and enhance patient engagement. We also show how insights from neuroscience and the local context guided the design of these tools to better meet patient and therapist needs. Our work highlights the potential of adaptive, AI-enhanced assistive technologies to complement conventional therapy and broaden access to therapy. We conclude by outlining future research directions for advancing personalized and scalable aphasia rehabilitation. 

---
# WIST: Web-Grounded Iterative Self-Play Tree for Domain-Targeted Reasoning Improvement 

**Authors**: Fangyuan Li, Pengfei Li, Shijie Wang, Junqi Gao, Jianxing Liu, Biqing Qi, Yuqiang Li  

**Link**: [PDF](https://arxiv.org/pdf/2603.22352)  

**Abstract**: Recent progress in reinforcement learning with verifiable rewards (RLVR) offers a practical path to self-improvement of language models, but existing methods face a key trade-off: endogenous self-play can drift over iterations, while corpus-grounded approaches rely on curated data environments. We present \textbf{WIST}, a \textbf{W}eb-grounded \textbf{I}terative \textbf{S}elf-play \textbf{T}ree framework for domain-targeted reasoning improvement that learns directly from the open web without requiring any pre-arranged domain corpus. WIST incrementally expands a domain tree for exploration, and retrieves and cleans path-consistent web corpus to construct a controllable training environment. It then performs Challenger--Solver self-play with verifiable rewards, and feeds learnability signals back to update node posteriors and guide subsequent exploration through an adaptive curriculum. Across four backbones, WIST consistently improves over the base models and typically outperforms both purely endogenous self-evolution and corpus-grounded self-play baselines, with the Overall gains reaching \textbf{+9.8} (\textit{Qwen3-4B-Base}) and \textbf{+9.7} (\textit{OctoThinker-8B}). WIST is also domain-steerable, improving \textit{Qwen3-8B-Base} by \textbf{+14.79} in medicine and \textit{Qwen3-4B-Base} by \textbf{+5.28} on PhyBench. Ablations further confirm the importance of WIST's key components for stable open-web learning. Our Code is available at this https URL. 

---
# First-Mover Bias in Gradient Boosting Explanations: Mechanism, Detection, and Resolution 

**Authors**: Drake Caraker, Bryan Arnold, David Rhoads  

**Link**: [PDF](https://arxiv.org/pdf/2603.22346)  

**Abstract**: We isolate and empirically characterize first-mover bias -- a path-dependent concentration of feature importance caused by sequential residual fitting in gradient boosting -- as a specific mechanistic cause of the well-known instability of SHAP-based feature rankings under multicollinearity. When correlated features compete for early splits, gradient boosting creates a self-reinforcing advantage for whichever feature is selected first: subsequent trees inherit modified residuals that favor the incumbent, concentrating SHAP importance on an arbitrary feature rather than distributing it across the correlated group. Scaling up a single model amplifies this effect -- a Large Single Model with the same total tree count as our method produces the worst explanations of any approach tested. We demonstrate that model independence is sufficient to resolve first-mover bias in the linear regime, and remains the most effective mitigation under nonlinear data-generating processes. Both our proposed method, DASH (Diversified Aggregation of SHAP), and simple seed-averaging (Stochastic Retrain) restore stability by breaking the sequential dependency chain, confirming that the operative mechanism is independence between explained models. At rho=0.9, both achieve stability=0.977, while the single-best workflow degrades to 0.958 and the Large Single Model to 0.938. On the Breast Cancer dataset, DASH improves stability from 0.32 to 0.93 (+0.61) against a tree-count-matched baseline. DASH additionally provides two diagnostic tools -- the Feature Stability Index (FSI) and Importance-Stability (IS) Plot -- that detect first-mover bias without ground truth, enabling practitioners to audit explanation reliability before acting on feature rankings. Software and reproducible benchmarks are available at this https URL. 

---
# T-MAP: Red-Teaming LLM Agents with Trajectory-aware Evolutionary Search 

**Authors**: Hyomin Lee, Sangwoo Park, Yumin Choi, Sohyun An, Seanie Lee, Sung Ju Hwang  

**Link**: [PDF](https://arxiv.org/pdf/2603.22341)  

**Abstract**: While prior red-teaming efforts have focused on eliciting harmful text outputs from large language models (LLMs), such approaches fail to capture agent-specific vulnerabilities that emerge through multi-step tool execution, particularly in rapidly growing ecosystems such as the Model Context Protocol (MCP). To address this gap, we propose a trajectory-aware evolutionary search method, T-MAP, which leverages execution trajectories to guide the discovery of adversarial prompts. Our approach enables the automatic generation of attacks that not only bypass safety guardrails but also reliably realize harmful objectives through actual tool interactions. Empirical evaluations across diverse MCP environments demonstrate that T-MAP substantially outperforms baselines in attack realization rate (ARR) and remains effective against frontier models, including GPT-5.2, Gemini-3-Pro, Qwen3.5, and GLM-5, thereby revealing previously underexplored vulnerabilities in autonomous LLM agents. 

---
# Graphs RAG at Scale: Beyond Retrieval-Augmented Generation With Labeled Property Graphs and Resource Description Framework for Complex and Unknown Search Spaces 

**Authors**: Manie Tadayon, Mayank Gupta  

**Link**: [PDF](https://arxiv.org/pdf/2603.22340)  

**Abstract**: Recent advances in Retrieval-Augmented Generation (RAG) have revolutionized knowledge-intensive tasks, yet traditional RAG methods struggle when the search space is unknown or when documents are semi-structured or structured. We introduce a novel end-to-end Graph RAG framework that leverages both Labeled Property Graph (LPG) and Resource Description Framework (RDF) architectures to overcome these limitations. Our approach enables dynamic document retrieval without the need to pre-specify the number of documents and eliminates inefficient reranking. We propose an innovative method for converting documents into RDF triplets using JSON key-value pairs, facilitating seamless integration of semi-structured data. Additionally, we present a text to Cypher framework for LPG, achieving over 90% accuracy in real-time translation of text queries to Cypher, enabling fast and reliable query generation suitable for online applications. Our empirical evaluation demonstrates that Graph RAG significantly outperforms traditional embedding-based RAG in accuracy, response quality, and reasoning, especially for complex, semi-structured tasks. These findings establish Graph RAG as a transformative solution for next-generation retrieval-augmented systems. 

---
# Causal Direct Preference Optimization for Distributionally Robust Generative Recommendation 

**Authors**: Chu Zhao, Enneng Yang, Jianzhe Zhao, Guibing Guo  

**Link**: [PDF](https://arxiv.org/pdf/2603.22335)  

**Abstract**: Direct Preference Optimization (DPO) guides large language models (LLMs) to generate recommendations aligned with user historical behavior distributions by minimizing preference alignment loss. However, our systematic empirical research and theoretical analysis reveal that DPO tends to amplify spurious correlations caused by environmental confounders during the alignment process, significantly undermining the generalization capability of LLM-based generative recommendation methods in out of distribution (OOD) scenarios. To mitigate this issue, we propose CausalDPO, an extension of DPO that incorporates a causal invariance learning mechanism. This method introduces a backdoor adjustment strategy during the preference alignment phase to eliminate interference from environmental confounders, explicitly models the latent environmental distribution using a soft clustering approach, and enhances robust consistency across diverse environments through invariance constraints. Theoretical analysis demonstrates that CausalDPO can effectively capture users stable preference structures across multiple environments, thereby improving the OOD generalization performance of LLM-based recommendation models. We conduct extensive experiments under four representative distribution shift settings to validate the effectiveness of CausalDPO, achieving an average performance improvement of 17.17% across four evaluation metrics. 

---
# Graph Signal Processing Meets Mamba2: Adaptive Filter Bank via Delta Modulation 

**Authors**: Yehjin Shin, Seojin Kim, Noseong Park  

**Link**: [PDF](https://arxiv.org/pdf/2603.22333)  

**Abstract**: State-space models (SSMs) offer efficient alternatives to attention with linear-time recurrence. Mamba2, a recent SSM-based language model, uses selective input gating and a multi-head structure, enabling parallel computation and strong benchmark performance. However, its multi-head recurrence operates independently without structured utilization or analysis. In this work, we propose a novel method called Hierarchical ADaptive filter bank for Efficient SSMs (HADES), a Graph Signal Processing (GSP)-inspired framework that reinterprets Mamba2 as an adaptive filter bank on a line graph. Our hierarchical architecture introduces two filter types: shared filters for global low-pass behavior and expert filters for local high-pass behavior, achieved through structured bias on the parameter {\Delta}. HADES achieves comparable performance to baseline models including Mamba2 across various benchmarks in language modeling, commonsense reasoning, and long-context retrieval, while using only 58.9% of the original parameters. In this regard, HADES bridges GSP and neural sequence modeling, enabling efficient, hierarchical, and interpretable filtering within state-space models. 

---
# Large Language Models for Missing Data Imputation: Understanding Behavior, Hallucination Effects, and Control Mechanisms 

**Authors**: Arthur Dantas Mangussi, Ricardo Cardoso Pereira, Ana Carolina Lorena, Pedro Henriques Abreu  

**Link**: [PDF](https://arxiv.org/pdf/2603.22332)  

**Abstract**: Data imputation is a cornerstone technique for handling missing values in real-world datasets, which are often plagued by missingness. Despite recent progress, prior studies on Large Language Models-based imputation remain limited by scalability challenges, restricted cross-model comparisons, and evaluations conducted on small or domain-specific datasets. Furthermore, heterogeneous experimental protocols and inconsistent treatment of missingness mechanisms (MCAR, MAR, and MNAR) hinder systematic benchmarking across methods. This work investigates the robustness of Large Language Models for missing data imputation in tabular datasets using a zero-shot prompt engineering approach. To this end, we present a comprehensive benchmarking study comparing five widely used LLMs against six state-of-the-art imputation baselines. The experimental design evaluates these methods across 29 datasets (including nine synthetic datasets) under MCAR, MAR, and MNAR mechanisms, with missing rates of up to 20\%. The results demonstrate that leading LLMs, particularly Gemini 3.0 Flash and Claude 4.5 Sonnet, consistently achieve superior performance on real-world open-source datasets compared to traditional methods. However, this advantage appears to be closely tied to the models' prior exposure to domain-specific patterns learned during pre-training on internet-scale corpora. In contrast, on synthetic datasets, traditional methods such as MICE outperform LLMs, suggesting that LLM effectiveness is driven by semantic context rather than purely statistical reconstruction. Furthermore, we identify a clear trade-off: while LLMs excel in imputation quality, they incur significantly higher computational time and monetary costs. Overall, this study provides a large-scale comparative analysis, positioning LLMs as promising semantics-driven imputers for complex tabular data. 

---
# Conformal Risk Control for Safety-Critical Wildfire Evacuation Mapping: A Comparative Study of Tabular, Spatial, and Graph-Based Models 

**Authors**: Baljinnyam Dayan  

**Link**: [PDF](https://arxiv.org/pdf/2603.22331)  

**Abstract**: Every wildfire prediction model deployed today shares a dangerous property: none of these methods provides formal guarantees on how much fire spread is missed. Despite extensive work on wildfire spread prediction using deep learning, no prior study has applied distribution-free safety guarantees to this domain, leaving evacuation planners reliant on probability thresholds with no formal assurance. We address this gap by presenting, to our knowledge, the first application of conformal risk control (CRC) to wildfire spread prediction, providing finite-sample guarantees on false negative rate (FNR <= 0.05). We expose a stark failure: across three model families of increasing complexity (tabular: LightGBM, AUROC 0.854; convolutional: Tiny U-Net, AUROC 0.969; and graph-based: Hybrid ResGNN-UNet, AUROC 0.964), standard thresholds capture only 7-72% of true fire spread. CRC eliminates this failure uniformly. Our central finding is that model architecture determines evacuation efficiency, while CRC determines safety: both spatial models with CRC achieve approximately 95% fire coverage while flagging only approximately 15% of total pixels, making them 4.2x more efficient than LightGBM, while the graph model's additional complexity over a simple U-Net yields no meaningful efficiency gain. We propose a shift-aware three-way CRC framework that assigns SAFE/MONITOR/EVACUATE zones for operational triage, and characterize a fundamental limitation of prevalence-weighted bounds under extreme class imbalance (approximately 5% fire prevalence). All models, calibration code, and evaluation pipelines are released for reproducibility. 

---
# Trained Persistent Memory for Frozen Decoder-Only LLMs 

**Authors**: Hong Jeong  

**Link**: [PDF](https://arxiv.org/pdf/2603.22329)  

**Abstract**: Decoder-only language models are stateless: hidden representations are discarded after every forward pass and nothing persists across sessions. Jeong (2026a) showed that trained memory adapters give a frozen encoder-decoder backbone persistent latent-space memory, building on the lateral-memory framework of Jeong (2026b,c). Here we ask whether the same principle transfers to the decoder-only setting, where no cross-attention pathway exists and memory must enter through self-attention alone. We adapt six methods -- prefix, parallel cross-attention, KV extension, Hebbian memory, context-gated branch, and slot-based sparse write -- to a frozen GPT-2, training only a small adapter $\theta_{mem}$. The write rule is shared; only the read injection changes from decoder cross-attention to self-attention KV prefix or parallel branch. On LoCoMo we find a striking inductive-bias dichotomy: at $1\times$ capacity, three methods with strong architectural priors -- cross-attention (M.2), Hebbian (M.4), and slot write (M.6) -- achieve retained-memory scores of $7-18\%$ and knowledge gains $\Delta K$ of $7-10$, while the other three fail ($< 0.4\%$). At $10\times$ capacity all six converge, showing the gap is architectural, not fundamental. Together with the encoder-decoder results of Jeong (2026a) and the brain-inspired modules of Jeong (2026b,c), these findings establish persistent latent-space memory as a general paradigm spanning major transformer families. 

---
# Beyond the Mean: Distribution-Aware Loss Functions for Bimodal Regression 

**Authors**: Abolfazl Mohammadi-Seif, Carlos Soares, Rita P. Ribeiro, Ricardo Baeza-Yates  

**Link**: [PDF](https://arxiv.org/pdf/2603.22328)  

**Abstract**: Despite the strong predictive performance achieved by machine learning models across many application domains, assessing their trustworthiness through reliable estimates of predictive confidence remains a critical challenge. This issue arises in scenarios where the likelihood of error inferred from learned representations follows a bimodal distribution, resulting from the coexistence of confident and ambiguous predictions. Standard regression approaches often struggle to adequately express this predictive uncertainty, as they implicitly assume unimodal Gaussian noise, leading to mean-collapse behavior in such settings. Although Mixture Density Networks (MDNs) can represent different distributions, they suffer from severe optimization instability. We propose a family of distribution-aware loss functions integrating normalized RMSE with Wasserstein and Cramér distances. When applied to standard deep regression models, our approach recovers bimodal distributions without the volatility of mixture models. Validated across four experimental stages, our results show that the proposed Wasserstein loss establishes a new Pareto efficiency frontier: matching the stability of standard regression losses like MSE in unimodal tasks while reducing Jensen-Shannon Divergence by 45% on complex bimodal datasets. Our framework strictly dominates MDNs in both fidelity and robustness, offering a reliable tool for aleatoric uncertainty estimation in trustworthy AI systems. 

---
# AgentSLR: Automating Systematic Literature Reviews in Epidemiology with Agentic AI 

**Authors**: Shreyansh Padarha, Ryan Othniel Kearns, Tristan Naidoo, Lingyi Yang, Łukasz Borchmann, Piotr BŁaszczyk, Christian Morgenstern, Ruth McCabe, Sangeeta Bhatia, Philip H. Torr, Jakob Foerster, Scott A. Hale, Thomas Rawson, Anne Cori, Elizaveta Semenova, Adam Mahdi  

**Link**: [PDF](https://arxiv.org/pdf/2603.22327)  

**Abstract**: Systematic literature reviews are essential for synthesizing scientific evidence but are costly, difficult to scale and time-intensive, creating bottlenecks for evidence-based policy. We study whether large language models can automate the complete systematic review workflow, from article retrieval, article screening, data extraction to report synthesis. Applied to epidemiological reviews of nine WHO-designated priority pathogens and validated against expert-curated ground truth, our open-source agentic pipeline (AgentSLR) achieves performance comparable to human researchers while reducing review time from approximately 7 weeks to 20 hours (a 58x speed-up). Our comparison of five frontier models reveals that performance on SLR is driven less by model size or inference cost than by each model's distinctive capabilities. Through human-in-the-loop validation, we identify key failure modes. Our results demonstrate that agentic AI can substantially accelerate scientific evidence synthesis in specialised domains. 

---
# A Direct Classification Approach for Reliable Wind Ramp Event Forecasting under Severe Class Imbalance 

**Authors**: Alejandro Morales-Hernández, Fabrizio De Caroa, Gian Marco Paldino, Pascal Tribel, Alfredo Vaccaro, Gianluca Bontempi  

**Link**: [PDF](https://arxiv.org/pdf/2603.22326)  

**Abstract**: Decision support systems are essential for maintaining grid stability in low-carbon power systems, such as wind power plants, by providing real-time alerts to control room operators regarding potential events, including Wind Power Ramp Events (WPREs). These early warnings enable the timely initiation of more detailed system stability assessments and preventive actions. However, forecasting these events is challenging due to the inherent class imbalance in WPRE datasets, where ramp events are less frequent (typically less than 15\% of observed events) compared to normal conditions. Ignoring this characteristic undermines the performance of conventional machine learning models, which often favor the majority class. This paper introduces a novel methodology for WPRE forecasting as a multivariate time series classification task and proposes a data preprocessing strategy that extracts features from recent power observations and masks unavailable ramp information, making it integrable with traditional real-time ramp identification tools. Particularly, the proposed methodology combines majority-class undersampling and ensemble learning to enhance wind ramp event forecasting under class imbalance. Numerical simulations conducted on a real-world dataset demonstrate the superiority of our approach, achieving over 85% accuracy and 88% weighted F1 score, outperforming benchmark classifiers. 

---
# Hybrid Associative Memories 

**Authors**: Leon Lufkin, Tomás Figliolia, Beren Millidge, Kamesh Krishnamurthy  

**Link**: [PDF](https://arxiv.org/pdf/2603.22325)  

**Abstract**: Recurrent neural networks (RNNs) and self-attention are both widely used sequence-mixing layers that maintain an internal memory. However, this memory is constructed using two orthogonal mechanisms: RNNs compress the entire past into a fixed-size state, whereas self-attention's state stores every past time step growing its state (the KV cache) linearly with the sequence length. This results in orthogonal strengths and weaknesses. Self-attention layers excel at retrieving information in the context but have large memory and computational costs, while RNNs are more efficient but degrade over longer contexts and underperform for precise recall tasks. Prior work combining these mechanisms has focused primarily on naively interleaving them to reduce computational cost without regard to their complementary mechanisms. We propose the Hybrid Associative Memory (HAM) layer, which combines self-attention and RNNs while leveraging their individual strengths: the RNN compresses the entire sequence, while attention supplements it *only* with information that is difficult for the RNN to predict, which is hence the most valuable information to explicitly store. HAM layers enable data-dependent growth of the KV cache, which can be precisely controlled by the user with a single, continuous threshold. We find that this fine-grained control of the KV cache growth rate has a smooth trade-off with loss and performance. Empirically, we show that our hybrid architecture offers strong, competitive performance relative to RNNs and Transformers even at substantially lower KV-cache usage. 

---
# DAQ: Delta-Aware Quantization for Post-Training LLM Weight Compression 

**Authors**: Xiaoming Yu, Shize Tang, Guanghua Yu, Linchuan Xie, Song Liu, Jianchen Zhu, Feng Li  

**Link**: [PDF](https://arxiv.org/pdf/2603.22324)  

**Abstract**: We introduce Delta-Aware Quantization (DAQ), a data-free post-training quantization framework that preserves the knowledge acquired during post-training. Standard quantization objectives minimize reconstruction error but are agnostic to the base model, allowing quantization noise to disproportionately corrupt the small-magnitude parameter deltas ($\Delta W$) that encode post-training behavior -- an effect we analyze through the lens of quantization as implicit regularization. DAQ replaces reconstruction-based objectives with two delta-aware metrics -- Sign Preservation Rate and Cosine Similarity -- that directly optimize for directional fidelity of $\Delta W$, requiring only the base and post-trained weight matrices. In a pilot FP8 study, DAQ recovers style-specific capabilities lost under standard quantization while maintaining general performance. 

---
# A Multi-Task Targeted Learning Framework for Lithium-Ion Battery State-of-Health and Remaining Useful Life 

**Authors**: Chenhan Wang, Zhengyi Bao, Huipin Lin, Jiahao Nie, Chunxiang Zhu  

**Link**: [PDF](https://arxiv.org/pdf/2603.22323)  

**Abstract**: Accurately predicting the state-of-health (SOH) and remaining useful life (RUL) of lithium-ion batteries is crucial for ensuring the safe and efficient operation of electric vehicles while minimizing associated risks. However, current deep learning methods are limited in their ability to selectively extract features and model time dependencies for these two parameters. Moreover, most existing methods rely on traditional recurrent neural networks, which have inherent shortcomings in long-term time-series modeling. To address these issues, this paper proposes a multi-task targeted learning framework for SOH and RUL prediction, which integrates multiple neural networks, including a multi-scale feature extraction module, an improved extended LSTM, and a dual-stream attention module. First, a feature extraction module with multi-scale CNNs is designed to capture detailed local battery decline patterns. Secondly, an improved extended LSTM network is employed to enhance the model's ability to retain long-term temporal information, thus improving temporal relationship modeling. Building on this, the dual-stream attention module-comprising polarized attention and sparse attention to selectively focus on key information relevant to SOH and RUL, respectively, by assigning higher weights to important features. Finally, a many-to-two mapping is achieved through the dual-task layer. To optimize the model's performance and reduce the need for manual hyperparameter tuning, the Hyperopt optimization algorithm is used. Extensive comparative experiments on battery aging datasets demonstrate that the proposed method reduces the average RMSE for SOH and RUL predictions by 111.3\% and 33.0\%, respectively, compared to traditional and state-of-the-art methods. 

---
# AEGIS: An Operational Infrastructure for Post-Market Governance of Adaptive Medical AI Under US and EU Regulations 

**Authors**: Fardin Afdideh, Mehdi Astaraki, Fernando Seoane, Farhad Abtahi  

**Link**: [PDF](https://arxiv.org/pdf/2603.22322)  

**Abstract**: Machine learning systems deployed in medical devices require governance frameworks that ensure safety while enabling continuous improvement. Regulatory bodies including the FDA and European Union have introduced mechanisms such as the Predetermined Change Control Plan (PCCP) and Post-Market Surveillance (PMS) to manage iterative model updates without repeated submissions. This paper presents AI/ML Evaluation and Governance Infrastructure for Safety (AEGIS), a governance framework applicable to any healthcare AI system. AEGIS comprises three modules, i.e., dataset assimilation and retraining, model monitoring, and conditional decision, that operationalize FDA PCCP and EU AI Act Article 43(4) provisions. We implement a four-category deployment decision taxonomy (APPROVE, CONDITIONAL APPROVAL, CLINICAL REVIEW, REJECT) with an independent PMS ALARM signal, enabling detection of the critical state in which no deployable model exists while the released model is simultaneously at risk.
To illustrate how AEGIS can be instantiated across heterogeneous clinical contexts, we provide two examples: sepsis prediction from electronic health records and brain tumor segmentation from medical imaging. Both cases use identical governance architecture, differing only in configuration. Across 11 simulated iterations on the sepsis example, AEGIS yielded 8 APPROVE, 1 CONDITIONAL APPROVAL, 1 CLINICAL REVIEW, and 1 REJECT decision, exercising all four categories. ALARM signals were co-issued at iterations 8 and 10, including the critical state where no deployable model exists and the released model is simultaneously failing. AEGIS detected drift before observable performance degradation. These results demonstrate that AEGIS translates regulatory change-control concepts into executable governance procedures, supporting safe continuous learning for adaptive medical AI across diverse clinical applications. 

---
# From Instructions to Assistance: a Dataset Aligning Instruction Manuals with Assembly Videos for Evaluating Multimodal LLMs 

**Authors**: Federico Toschi, Nicolò Brunello, Andrea Sassella, Vincenzo Scotti, Mark James Carman  

**Link**: [PDF](https://arxiv.org/pdf/2603.22321)  

**Abstract**: The recent advancements introduced by Large Language Models (LLMs) have transformed how Artificial Intelligence (AI) can support complex, real world tasks, pushing research outside the text boundaries towards multi modal contexts and leading to Multimodal Large Language Models (MLMs). Given the current adoption of LLM based assistants in solving technical or domain specific problems, the natural continuation of this trend is to extend the input domains of these assistants exploiting MLMs. Ideally, these MLMs should be used as real time assistants in procedural tasks, hopefully integrating a view of the environment where the user being assisted is, or even better sharing the same point of view via Virtual Reality (VR) or Augmented Reality (AR) supports, to reason over the same scenario the user is experiencing. With this work, we aim at evaluating the quality of currently openly available MLMs to provide this kind of assistance on technical tasks. To this end, we annotated a data set of furniture assembly with step by step labels and manual references: the Manual to Action Dataset (M2AD). We used this dataset to assess (1) to which extent the reasoning abilities of MLMs can be used to reduce the need for detailed labelling, allowing for more efficient, cost effective annotation practices, (2) whether MLMs are able to track the progression of assembly steps (3) and whether MLMs can refer correctly to the instruction manual pages. Our results showed that while some models understand procedural sequences, their performance is limited by architectural and hardware constraints, highlighting the need for multi image and interleaved text image reasoning. 

---
# Sparsely-Supervised Data Assimilation via Physics-Informed Schrödinger Bridge 

**Authors**: Dohyun Bu, Chanho Kim, Seokun Choi, Jong-Seok Lee  

**Link**: [PDF](https://arxiv.org/pdf/2603.22319)  

**Abstract**: Data assimilation (DA) for systems governed by partial differential equations (PDE) aims to reconstruct full spatiotemporal fields from sparse high-fidelity (HF) observations while respecting physical constraints. While full-grid low-fidelity (LF) simulations provide informative priors in multi-fidelity settings, recovering an HF field consistent with both sparse observations and the governing PDE typically requires per-instance test-time optimization, which becomes a major bottleneck in time-critical applications. To alleviate this, amortized reconstruction using generative models has recently been proposed; however, such approaches rely on full-field HF supervision during training, which is often impractical in real-world settings. From a more realistic perspective, we propose the Physics-Informed Conditional Schrödinger Bridge (PICSB), which transports an informative LF prior toward an observation-conditioned HF posterior without any additional inference-time guidance. To enable learning without HF endpoints, PICSB employs an iterative surrogate-endpoint refresh scheme, and directly incorporates PDE residuals into the training objective while enforcing observations via hard conditioning throughout sampling. Experiments on fluid PDE benchmarks demonstrate that PICSB enables extremely fast spatiotemporal field reconstruction while maintaining competitive accuracy under sparse HF supervision. 

---
# Geometric Mixture-of-Experts with Curvature-Guided Adaptive Routing for Graph Representation Learning 

**Authors**: Haifang Cao, Yu Wang, Timing Li, Xinjie Yao, Pengfei Zhu  

**Link**: [PDF](https://arxiv.org/pdf/2603.22317)  

**Abstract**: Graph-structured data typically exhibits complex topological heterogeneity, making it difficult to model accurately within a single Riemannian manifold. While emerging mixed-curvature methods attempt to capture such diversity, they often rely on implicit, task-driven routing that lacks fundamental geometric grounding. To address this challenge, we propose a Geometric Mixture-of-Experts framework (GeoMoE) that adaptively fuses node representations across diverse Riemannian spaces to better accommodate multi-scale topological structures. At its core, GeoMoE leverages Ollivier-Ricci Curvature (ORC) as an intrinsic geometric prior to orchestrate the collaboration of specialized experts. Specifically, we design a graph-aware gating network that assigns node-specific fusion weights, regularized by a curvature-guided alignment loss to ensure interpretable and geometry-consistent routing. Additionally, we introduce a curvature-aware contrastive objective that promotes geometric discriminability by constructing positive and negative pairs according to curvature consistency. Extensive experiments on six benchmark datasets demonstrate that GeoMoE outperforms state-of-the-art baselines across diverse graph types. 

---
# ST-GDance++: A Scalable Spatial-Temporal Diffusion for Long-Duration Group Choreography 

**Authors**: Jing Xu, Weiqiang Wang, Cunjian Chen, Jun Liu, Qiuhong Ke  

**Link**: [PDF](https://arxiv.org/pdf/2603.22316)  

**Abstract**: Group dance generation from music requires synchronizing multiple dancers while maintaining spatial coordination, making it highly relevant to applications such as film production, gaming, and animation. Recent group dance generation models have achieved promising generation quality, but they remain difficult to deploy in interactive scenarios due to bidirectional attention dependencies. As the number of dancers and the sequence length increase, the attention computation required for aligning music conditions with motion sequences grows quadratically, leading to reduced efficiency and increased risk of motion collisions. Effectively modeling dense spatial-temporal interactions is therefore essential, yet existing methods often struggle to capture such complexity, resulting in limited scalability and unstable multi-dancer coordination. To address these challenges, we propose ST-GDance++, a scalable framework that decouples spatial and temporal dependencies to enable efficient and collision-aware group choreography generation. For spatial modeling, we introduce lightweight distance-aware graph convolutions to capture inter-dancer relationships while reducing computational overhead. For temporal modeling, we design a diffusion noise scheduling strategy together with an efficient temporal-aligned attention mask, enabling stream-based generation for long motion sequences and improving scalability in long-duration scenarios. Experiments on the AIOZ-GDance dataset show that ST-GDance++ achieves competitive generation quality with significantly reduced latency compared to existing methods. 

---
# Emergency Preemption Without Online Exploration: A Decision Transformer Approach 

**Authors**: Haoran Su, Hanxiao Deng, Yandong Sun  

**Link**: [PDF](https://arxiv.org/pdf/2603.22315)  

**Abstract**: Emergency vehicle (EV) response time is a critical determinant of survival outcomes, yet deployed signal preemption strategies remain reactive and uncontrollable. We propose a return-conditioned framework for emergency corridor optimization based on the Decision Transformer (DT). By casting corridor optimization as offline, return-conditioned sequence modeling, our approach (1) eliminates online environment interaction during policy learning, (2) enables dispatch-level urgency control through a single target-return scalar, and (3) extends to multi-agent settings via a Multi-Agent Decision Transformer (MADT) with graph attention for spatial coordination. On the LightSim simulator, DT reduces average EV travel time by 37.7% relative to fixed-timing preemption on a 4x4 grid (88.6 s vs. 142.3 s), achieving the lowest civilian delay (11.3 s/veh) and fewest EV stops (1.2) among all methods, including online RL baselines that require environment interaction. MADT further improves on larger grids, overtaking DT with 45.2% reduction on 8x8 via graph-attention coordination. Return conditioning produces a smooth dispatch interface: varying the target return from 100 to -400 trades EV travel time (72.4-138.2 s) against civilian delay (16.8-5.4 s/veh), requiring no retraining. A Constrained DT extension adds explicit civilian disruption budgets as a second control knob. 

---
# Enhancing AI-Based Tropical Cyclone Track and Intensity Forecasting via Systematic Bias Correction 

**Authors**: Peisong Niu, Haifan Zhang, Yang Zhao, Tian Zhou, Ziqing Ma, Wenqiang Shen, Junping Zhao, Huiling Yuan, Liang Sun  

**Link**: [PDF](https://arxiv.org/pdf/2603.22314)  

**Abstract**: Tropical cyclones (TCs) pose severe threats to life, infrastructure, and economies in tropical and subtropical regions, underscoring the critical need for accurate and timely forecasts of both track and intensity. Recent advances in AI-based weather forecasting have shown promise in improving TC track forecasts. However, these systems are typically trained on coarse-resolution reanalysis data (e.g., ERA5 at 0.25 degree), which constrains predicted TC positions to a fixed grid and introduces significant discretization errors. Moreover, intensity forecasting remains limited especially for strong TCs by the smoothing effect of coarse meteorological fields and the use of regression losses that bias predictions toward conditional means. To address these limitations, we propose BaguanCyclone, a novel, unified framework that integrates two key innovations: (1) a probabilistic center refinement module that models the continuous spatial distribution of TC centers, enabling finer track precision; and (2) a region-aware intensity forecasting module that leverages high-resolution internal representations within dynamically defined sub-grid zones around the TC core to better capture localized extremes. Evaluated on the global IBTrACS dataset across six major TC basins, our system consistently outperforms both operational numerical weather prediction (NWP) models and most AI-based baselines, delivering a substantial enhancement in forecast accuracy. Remarkably, BaguanCyclone excels in navigating meteorological complexities, consistently delivering accurate forecasts for re-intensification, sweeping arcs, twin cyclones, and meandering events. Our code is available at this https URL. 

---
# A Multi-Modal CNN-LSTM Framework with Multi-Head Attention and Focal Loss for Real-Time Elderly Fall Detection 

**Authors**: Lijie Zhou, Luran Wang  

**Link**: [PDF](https://arxiv.org/pdf/2603.22313)  

**Abstract**: The increasing global aging population has intensified the demand for reliable health monitoring systems, particularly those capable of detecting critical events such as falls among elderly individuals. Traditional fall detection approaches relying on single-modality acceleration data suffer from high false alarm rates, while conventional machine learning methods require extensive hand-crafted feature engineering. This paper proposes a novel multi-modal deep learning framework, MultiModalFallDetector, designed for real-time elderly fall detection using wearable sensors. Our approach integrates multiple innovations: a multi-scale CNN-based feature extractor capturing motion dynamics at varying temporal resolutions; fusion of tri-axial accelerometer, gyroscope, and four-channel physiological signals; incorporation of a multi-head self-attention mechanism for dynamic temporal weighting; adoption of Focal Loss to mitigate severe class imbalance; introduction of an auxiliary activity classification task for regularization; and implementation of transfer learning from UCI HAR to SisFall dataset. Extensive experiments on the SisFall dataset, which includes real-world simulated fall trials from elderly participants (aged 60-85), demonstrate that our framework achieves an F1-score of 98. 7, Recall of 98. 9, and AUC-ROC of 99. 4, significantly outperforming baseline methods including traditional machine learning and standard deep learning approaches. The model maintains sub- 50ms inference latency on edge devices, confirming its suitability for real-time deployment in geriatric care settings. 

---
# UniFluids: Unified Neural Operator Learning with Conditional Flow-matching 

**Authors**: Haosen Li, Qi Meng, Jiahao Li, Rui Zhang, Ruihua Song, Liang Ma, Zhi-Ming Ma  

**Link**: [PDF](https://arxiv.org/pdf/2603.22309)  

**Abstract**: Partial differential equation (PDE) simulation holds extensive significance in scientific research. Currently, the integration of deep neural networks to learn solution operators of PDEs has introduced great potential. In this paper, we present UniFluids, a conditional flow-matching framework that harnesses the scalability of diffusion Transformer to unify learning of solution operators across diverse PDEs with varying dimensionality and physical variables. Unlike the autoregressive PDE foundation models, UniFluids adopts flow-matching to achieve parallel sequence generation, making it the first such approach for unified operator learning. Specifically, the introduction of a unified four-dimensional spatiotemporal representation for the heterogeneous PDE datasets enables joint training and conditional encoding. Furthermore, we find the effective dimension of the PDE dataset is much lower than its patch dimension. We thus employ $x$-prediction in the flow-matching operator learning, which is verified to significantly improve prediction accuracy. We conduct a large-scale evaluation of UniFluids on several PDE datasets covering spatial dimensions 1D, 2D and 3D. Experimental results show that UniFluids achieves strong prediction accuracy and demonstrates good scalability and cross-scenario generalization capability. The code will be released later. 

---
# CN-Buzz2Portfolio: A Chinese-Market Dataset and Benchmark for LLM-Based Macro and Sector Asset Allocation from Daily Trending Financial News 

**Authors**: Liyuan Chen, Shilong Li, Jiangpeng Yan, Shuoling Liu, Qiang Yang, Xiu Li  

**Link**: [PDF](https://arxiv.org/pdf/2603.22305)  

**Abstract**: Large Language Models (LLMs) are rapidly transitioning from static Natural Language Processing (NLP) tasks including sentiment analysis and event extraction to acting as dynamic decision-making agents in complex financial environments. However, the evolution of LLMs into autonomous financial agents faces a significant dilemma in evaluation paradigms. Direct live trading is irreproducible and prone to outcome bias by confounding luck with skill, whereas existing static benchmarks are often confined to entity-level stock picking and ignore broader market attention. To facilitate the rigorous analysis of these challenges, we introduce CN-Buzz2Portfolio, a reproducible benchmark grounded in the Chinese market that maps daily trending news to macro and sector asset allocation. Spanning a rolling horizon from 2024 to mid-2025, our dataset simulates a realistic public attention stream, requiring agents to distill investment logic from high-exposure narratives instead of pre-filtered entity news. We propose a Tri-Stage CPA Agent Workflow involving Compression, Perception, and Allocation to evaluate LLMs on broad asset classes such as Exchange Traded Funds (ETFs) rather than individual stocks, thereby reducing idiosyncratic volatility. Extensive experiments on nine LLMs reveal significant disparities in how models translate macro-level narratives into portfolio weights. This work provides new insights into the alignment between general reasoning and financial decision-making, and all data, codes, and experiments are released to promote sustainable financial agent research. 

---
# Sample Transform Cost-Based Training-Free Hallucination Detector for Large Language Models 

**Authors**: Zeyang Ding, Xinglin Hu, Jicong Fan  

**Link**: [PDF](https://arxiv.org/pdf/2603.22303)  

**Abstract**: Hallucinations in large language models (LLMs) remain a central obstacle to trustworthy deployment, motivating detectors that are accurate, lightweight, and broadly applicable. Since an LLM with a prompt defines a conditional distribution, we argue that the complexity of the distribution is an indicator of hallucination. However, the density of the distribution is unknown and the samples (i.e., responses generated for the prompt) are discrete distributions, which leads to a significant challenge in quantifying the complexity of the distribution. We propose to compute the optimal-transport distances between the sets of token embeddings of pairwise samples, which yields a Wasserstein distance matrix measuring the costs of transforming between the samples. This Wasserstein distance matrix provides a means to quantify the complexity of the distribution defined by the LLM with the prompt. Based on the Wasserstein distance matrix, we derive two complementary signals: AvgWD, measuring the average cost, and EigenWD, measuring the cost complexity. This leads to a training-free detector for hallucinations in LLMs. We further extend the framework to black-box LLMs via teacher forcing with an accessible teacher model. Experiments show that AvgWD and EigenWD are competitive with strong uncertainty baselines and provide complementary behavior across models and datasets, highlighting distribution complexity as an effective signal for LLM truthfulness. 

---
# Latent Semantic Manifolds in Large Language Models 

**Authors**: Mohamed A. Mabrok  

**Link**: [PDF](https://arxiv.org/pdf/2603.22301)  

**Abstract**: Large Language Models (LLMs) perform internal computations in continuous vector spaces yet produce discrete tokens -- a fundamental mismatch whose geometric consequences remain poorly understood. We develop a mathematical framework that interprets LLM hidden states as points on a latent semantic manifold: a Riemannian submanifold equipped with the Fisher information metric, where tokens correspond to Voronoi regions partitioning the manifold. We define the expressibility gap, a geometric measure of the semantic distortion from vocabulary discretization, and prove two theorems: a rate-distortion lower bound on distortion for any finite vocabulary, and a linear volume scaling law for the expressibility gap via the coarea formula. We validate these predictions across six transformer architectures (124M-1.5B parameters), confirming universal hourglass intrinsic dimension profiles, smooth curvature structure, and linear gap scaling with slopes 0.87-1.12 (R^2 > 0.985). The margin distribution across models reveals a persistent hard core of boundary-proximal representations invariant to scale, providing a geometric decomposition of perplexity. We discuss implications for architecture design, model compression, decoding strategies, and scaling laws 

---
# Scaling Attention via Feature Sparsity 

**Authors**: Yan Xie, Tiansheng Wen, Tangda Huang, Bo Chen, Chenyu You, Stefanie Jegelka, Yifei Wang  

**Link**: [PDF](https://arxiv.org/pdf/2603.22300)  

**Abstract**: Scaling Transformers to ultra-long contexts is bottlenecked by the $O(n^2 d)$ cost of self-attention. Existing methods reduce this cost along the sequence axis through local windows, kernel approximations, or token-level sparsity, but these approaches consistently degrade accuracy. In this paper, we instead explore an orthogonal axis: feature sparsity. We propose Sparse Feature Attention (SFA), where queries and keys are represented as $k$-sparse codes that preserve high-dimensional expressivity while reducing the cost of attention from $\Theta(n^2 d)$ to $\Theta(n^2 k^2/d)$. To make this efficient at scale, we introduce FlashSFA, an IO-aware kernel that extends FlashAttention to operate directly on sparse overlaps without materializing dense score matrices. Across GPT-2 and Qwen3 pretraining, SFA matches dense baselines while improving speed by up to $2.5\times$ and reducing FLOPs and KV-cache by nearly 50\%. On synthetic and downstream benchmarks, SFA preserves retrieval accuracy and robustness at long contexts, outperforming short-embedding baselines that collapse feature diversity. These results establish feature-level sparsity as a complementary and underexplored axis for efficient attention, enabling Transformers to scale to orders-of-magnitude longer contexts with minimal quality loss. Code is available at this https URL. 

---
# Between the Layers Lies the Truth: Uncertainty Estimation in LLMs Using Intra-Layer Local Information Scores 

**Authors**: Zvi N. Badash, Yonatan Belinkov, Moti Freiman  

**Link**: [PDF](https://arxiv.org/pdf/2603.22299)  

**Abstract**: Large language models (LLMs) are often confidently wrong, making reliable uncertainty estimation (UE) essential. Output-based heuristics are cheap but brittle, while probing internal representations is effective yet high-dimensional and hard to transfer.
We propose a compact, per-instance UE method that scores cross-layer agreement patterns in internal representations using a single forward pass.
Across three models, our method matches probing in-distribution, with mean diagonal differences of at most $-1.8$ AUPRC percentage points and $+4.9$ Brier score points. Under cross-dataset transfer, it consistently outperforms probing, achieving off-diagonal gains up to $+2.86$ AUPRC and $+21.02$ Brier points. Under 4-bit weight-only quantization, it remains robust, improving over probing by $+1.94$ AUPRC points and $+5.33$ Brier points on average.
Beyond performance, examining specific layer--layer interactions reveals differences in how disparate models encode uncertainty. Altogether, our UE method offers a lightweight, compact means to capture transferable uncertainty in LLMs. 

---
# Whether, Not Which: Mechanistic Interpretability Reveals Dissociable Affect Reception and Emotion Categorization in LLMs 

**Authors**: Michael Keeman  

**Link**: [PDF](https://arxiv.org/pdf/2603.22295)  

**Abstract**: Large language models appear to develop internal representations of emotion -- "emotion circuits," "emotion neurons," and structured emotional manifolds have been reported across multiple model families. But every study making these claims uses stimuli signalled by explicit emotion keywords, leaving a fundamental question unanswered: do these circuits detect genuine emotional meaning, or do they detect the word "devastated"? We present the first clinical validity test of emotion circuit claims using mechanistic interpretability methods grounded in clinical psychology -- clinical vignettes that evoke emotions through situational and behavioural cues alone, emotion keywords removed. Across six models (Llama-3.2-1B, Llama-3-8B, Gemma-2-9B; base and instruct variants), we apply four convergent mechanistic interpretability methods -- linear probing, causal activation patching, knockout experiments, and representational geometry -- and discover two dissociable emotion processing mechanisms. Affect reception -- detecting emotionally significant content -- operates with near-perfect accuracy (AUROC 1.000), consistent with early-layer saturation, and replicates across all six models. Emotion categorization -- mapping affect to specific emotion labels -- is partially keyword-dependent, dropping 1-7% without keywords and improving with scale. Causal activation patching confirms keyword-rich and keyword-free stimuli share representational space, transferring affective salience rather than emotion-category identity. These findings falsify the keyword-spotting hypothesis, establish a novel mechanistic dissociation, and introduce clinical stimulus methodology as a rigorous standard for testing emotion processing claims in large language models -- with direct implications for AI safety evaluation and alignment. All stimuli, code, and data are released for replication. 

---
# Efficient Embedding-based Synthetic Data Generation for Complex Reasoning Tasks 

**Authors**: Srideepika Jayaraman, Achille Fokoue, Dhaval Patel, Jayant Kalagnanam  

**Link**: [PDF](https://arxiv.org/pdf/2603.22294)  

**Abstract**: Synthetic Data Generation (SDG), leveraging Large Language Models (LLMs), has recently been recognized and broadly adopted as an effective approach to improve the performance of smaller but more resource and compute efficient LLMs through fine-tuning. A key challenge in SDG is ensuring the quality and diversity of the generated data. In this paper, we analyze the diversity and distribution of generated data in the embedding space, and demonstrate a strong correlation between the density of examples within a specific neighborhood and the accuracy of predictions on examples drawn from that region. Building on this insight, we present a targeted pipeline for embedding-based sampling that enhances data diversity and consistently improves performance across several benchmarks. 

---
# TIPS: Turn-Level Information-Potential Reward Shaping for Search-Augmented LLMs 

**Authors**: Yutao Xie, Nathaniel Thomas, Nicklas Hansen, Yang Fu, Li Erran Li, Xiaolong Wang  

**Link**: [PDF](https://arxiv.org/pdf/2603.22293)  

**Abstract**: Search-augmented large language models (LLMs) trained with reinforcement learning (RL) have achieved strong results on open-domain question answering (QA), but training still remains a significant challenge. The optimization is often unstable due to sparse rewards and difficult credit assignments across reasoning and tool calls. To address this, we introduce Turn-Level Information Potential Reward Shaping (TIPS), a simple framework that assigns dense, turn-level rewards to each reasoning + tool-call segment based on the increased likelihood of the correct answer under a teacher model. By leveraging the potential-based reward shaping, TIPS offers fine-grained and policy-invariant guidance that overcomes the limitations of outcome-only optimization. Evaluated on seven QA benchmarks, TIPS consistently outperforms GRPO/PPO baselines and substantially improves training stability. For instance, with a Qwen-2.5 7B Instruct model, TIPS improves the average Exact Match score by 11.8% and F1 by 13.6% relative to PPO. Our results demonstrate that turn-level information-potential reward shaping provides an effective and general solution to sparse-reward credit assignment for multi-turn LLM reasoning. 

---
# Beyond Hard Constraints: Budget-Conditioned Reachability For Safe Offline Reinforcement Learning 

**Authors**: Janaka Chathuranga Brahmanage, Akshat Kumar  

**Link**: [PDF](https://arxiv.org/pdf/2603.22292)  

**Abstract**: Sequential decision making using Markov Decision Process underpins many realworld applications. Both model-based and model free methods have achieved strong results in these settings. However, real-world tasks must balance reward maximization with safety constraints, often conflicting objectives, that can lead to unstable min/max, adversarial optimization. A promising alternative is safety reachability analysis, which precomputes a forward-invariant safe state, action set, ensuring that an agent starting inside this set remains safe indefinitely. Yet, most reachability based methods address only hard safety constraints, and little work extends reachability to cumulative cost constraints. To address this, first, we define a safetyconditioned reachability set that decouples reward maximization from cumulative safety cost constraints. Second, we show how this set enforces safety constraints without unstable min/max or Lagrangian optimization, yielding a novel offline safe RL algorithm that learns a safe policy from a fixed dataset without environment interaction. Finally, experiments on standard offline safe RL benchmarks, and a real world maritime navigation task demonstrate that our method matches or outperforms state of the art baselines while maintaining safety. 

---
# MERIT: Memory-Enhanced Retrieval for Interpretable Knowledge Tracing 

**Authors**: Runze Li, Kedi Chen, Guwei Feng, Mo Yu, Jun Wang, Wei Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2603.22289)  

**Abstract**: Knowledge Tracing (KT) models students' evolving knowledge states to predict future performance, serving as a foundation for personalized education. While traditional deep learning models achieve high accuracy, they often lack interpretability. Large Language Models (LLMs) offer strong reasoning capabilities but struggle with limited context windows and hallucinations. Furthermore, existing LLM-based methods typically require expensive fine-tuning, limiting scalability and adaptability to new data. We propose MERIT (Memory-Enhanced Retrieval for Interpretable Knowledge Tracing), a training-free framework combining frozen LLM reasoning with structured pedagogical memory. Rather than updating parameters, MERIT transforms raw interaction logs into an interpretable memory bank. The framework uses semantic denoising to categorize students into latent cognitive schemas and constructs a paradigm bank where representative error patterns are analyzed offline to generate explicit Chain-of-Thought (CoT) rationales. During inference, a hierarchical routing mechanism retrieves relevant contexts, while a logic-augmented module applies semantic constraints to calibrate predictions. By grounding the LLM in interpretable memory, MERIT achieves state-of-the-art performance on real-world datasets without gradient updates. This approach reduces computational costs and supports dynamic knowledge updates, improving the accessibility and transparency of educational diagnosis. 

---
# Evaluating Prompting Strategies for Chart Question Answering with Large Language Models 

**Authors**: Ruthuparna Naikar, Ying Zhu  

**Link**: [PDF](https://arxiv.org/pdf/2603.22288)  

**Abstract**: Prompting strategies affect LLM reasoning performance, but their role in chart-based QA remains underexplored. We present a systematic evaluation of four widely used prompting paradigms (Zero-Shot, Few-Shot, Zero-Shot Chain-of-Thought, and Few-Shot Chain-of-Thought) across GPT-3.5, GPT-4, and GPT-4o on the ChartQA dataset. Our framework operates exclusively on structured chart data, isolating prompt structure as the only experimental variable, and evaluates performance using two metrics: Accuracy and Exact Match. Results from 1,200 diverse ChartQA samples show that Few-Shot Chain-of-Thought prompting consistently yields the highest accuracy (up to 78.2\%), particularly on reasoning-intensive questions, while Few-Shot prompting improves format adherence. Zero-Shot performs well only with high-capacity models on simpler tasks. These findings provide actionable guidance for selecting prompting strategies in structured data reasoning tasks, with implications for both efficiency and accuracy in real-world applications. 

---
# Founder effects shape the evolutionary dynamics of multimodality in open LLM families 

**Authors**: Manuel Cebrian  

**Link**: [PDF](https://arxiv.org/pdf/2603.22287)  

**Abstract**: Large language model (LLM) families are improving rapidly, yet it remains unclear how quickly multimodal capabilities emerge and propagate within open families. Using the ModelBiome AI Ecosystem dataset of Hugging Face model metadata and recorded lineage fields (>1.8x10^6 model entries), we quantify multimodality over time and along recorded parent-to-child relations. Cross-modal tasks are widespread in the broader ecosystem well before they become common within major open LLM families: within these families, multimodality remains rare through 2023 and most of 2024, then increases sharply in 2024-2025 and is dominated by image-text vision-language tasks. Across major families, the first vision-language model (VLM) variants typically appear months after the first text-generation releases, with lags ranging from ~1 month (Gemma) to more than a year for several families and ~26 months for GLM. Lineage-conditioned transition rates show weak cross-type transfer: among fine-tuning edges from text-generation parents, only 0.218% yield VLM descendants. Instead, multimodality expands primarily within existing VLM lineages: 94.5% of VLM-child fine-tuning edges originate from VLM parents, versus 4.7% from text-generation parents. At the model level, most VLM releases appear as new roots without recorded parents (~60%), while the remainder are predominantly VLM-derived; founder concentration analyses indicate rapid within-lineage amplification followed by diversification. Together, these results show that multimodality enters open LLM families through rare founder events and then expands rapidly within their descendant lineages, producing punctuated adoption dynamics that likely induce distinct, transfer-limited scaling behavior for multimodal capabilities. 

---
# Automated Microservice Pattern Instance Detection Using Infrastructure-as-Code Artifacts and Large Language Models 

**Authors**: Carlos Eduardo Duarte  

**Link**: [PDF](https://arxiv.org/pdf/2502.04188)  

**Abstract**: Documenting software architecture is essential to preserve architecture knowledge, even though it is frequently costly. Architecture pattern instances, including microservice pattern instances, provide important structural software information. Practitioners should document this information to prevent knowledge vaporization. However, architecture patterns may not be detectable by analyzing source code artifacts, requiring the analysis of other types of artifacts. Moreover, many existing pattern detection instance approaches are complex to extend. This article presents our ongoing PhD research, early experiments, and a prototype for a tool we call MicroPAD for automating the detection of microservice pattern instances. The prototype uses Large Language Models (LLMs) to analyze Infrastructure-as-Code (IaC) artifacts to aid detection, aiming to keep costs low and maximize the scope of detectable patterns. Early experiments ran the prototype thrice in 22 GitHub projects. We verified that 83\% of the patterns that the prototype identified were in the project. The costs of detecting the pattern instances were minimal. These results indicate that the approach is likely viable and, by lowering the entry barrier to automating pattern instance detection, could help democratize developer access to this category of architecture knowledge. Finally, we present our overall research methodology, planned future work, and an overview of MicroPAD's potential industrial impact. 

---
