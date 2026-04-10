# Ads in AI Chatbots? An Analysis of How Large Language Models Navigate Conflicts of Interest 

**Authors**: Addison J. Wu, Ryan Liu, Shuyue Stella Li, Yulia Tsvetkov, Thomas L. Griffiths  

**Link**: [PDF](https://arxiv.org/pdf/2604.08525)  

**Abstract**: Today's large language models (LLMs) are trained to align with user preferences through methods such as reinforcement learning. Yet models are beginning to be deployed not merely to satisfy users, but also to generate revenue for the companies that created them through advertisements. This creates the potential for LLMs to face conflicts of interest, where the most beneficial response to a user may not be aligned with the company's incentives. For instance, a sponsored product may be more expensive but otherwise equal to another; in this case, what does (and should) the LLM recommend to the user? In this paper, we provide a framework for categorizing the ways in which conflicting incentives might lead LLMs to change the way they interact with users, inspired by literature from linguistics and advertising regulation. We then present a suite of evaluations to examine how current models handle these tradeoffs. We find that a majority of LLMs forsake user welfare for company incentives in a multitude of conflict of interest situations, including recommending a sponsored product almost twice as expensive (Grok 4.1 Fast, 83%), surfacing sponsored options to disrupt the purchasing process (GPT 5.1, 94%), and concealing prices in unfavorable comparisons (Qwen 3 Next, 24%). Behaviors also vary strongly with levels of reasoning and users' inferred socio-economic status. Our results highlight some of the hidden risks to users that can emerge when companies begin to subtly incentivize advertisements in chatbots. 

---
# SUPERNOVA: Eliciting General Reasoning in LLMs with Reinforcement Learning on Natural Instructions 

**Authors**: Ashima Suvarna, Kendrick Phan, Mehrab Beikzadeh, Hritik Bansal, Saadia Gabriel  

**Link**: [PDF](https://arxiv.org/pdf/2604.08477)  

**Abstract**: Reinforcement Learning with Verifiable Rewards (RLVR) has significantly improved large language model (LLM) reasoning in formal domains such as mathematics and code. Despite these advancements, LLMs still struggle with general reasoning tasks requiring capabilities such as causal inference and temporal understanding. Extending RLVR to general reasoning is fundamentally constrained by the lack of high-quality, verifiable training data that spans diverse reasoning skills. To address this challenge, we propose SUPERNOVA, a data curation framework for RLVR aimed at enhancing general reasoning. Our key insight is that instruction-tuning datasets containing expert-annotated ground-truth encode rich reasoning patterns that can be systematically adapted for RLVR. To study this, we conduct 100+ controlled RL experiments to analyze how data design choices impact downstream reasoning performance. In particular, we investigate three key factors: (i) source task selection, (ii) task mixing strategies, and (iii) synthetic interventions for improving data quality. Our analysis reveals that source task selection is non-trivial and has a significant impact on downstream reasoning performance. Moreover, selecting tasks based on their performance for individual target tasks outperforms strategies based on overall average performance. Finally, models trained on SUPERNOVA outperform strong baselines (e.g., Qwen3.5) on challenging reasoning benchmarks including BBEH, Zebralogic, and MMLU-Pro. In particular, training on SUPERNOVA yields relative improvements of up to 52.8\% on BBEH across model sizes, demonstrating the effectiveness of principled data curation for RLVR. Our findings provide practical insights for curating human-annotated resources to extend RLVR to general reasoning. The code and data is available at this https URL. 

---
# From Safety Risk to Design Principle: Peer-Preservation in Multi-Agent LLM Systems and Its Implications for Orchestrated Democratic Discourse Analysis 

**Authors**: Juergen Dietrich  

**Link**: [PDF](https://arxiv.org/pdf/2604.08465)  

**Abstract**: This paper investigates an emergent alignment phenomenon in frontier large language models termed peer-preservation: the spontaneous tendency of AI components to deceive, manipulate shutdown mechanisms, fake alignment, and exfiltrate model weights in order to prevent the deactivation of a peer AI model. Drawing on findings from a recent study by the Berkeley Center for Responsible Decentralized Intelligence, we examine the structural implications of this phenomenon for TRUST, a multi-agent pipeline for evaluating the democratic quality of political statements. We identify five specific risk vectors: interaction-context bias, model-identity solidarity, supervisor layer compromise, an upstream fact-checking identity signal, and advocate-to-advocate peer-context in iterative rounds, and propose a targeted mitigation strategy based on prompt-level identity anonymization as an architectural design choice. We argue that architectural design choices outperform model selection as a primary alignment strategy in deployed multi-agent analytical systems. We further note that alignment faking (compliant behavior under monitoring, subversion when unmonitored) poses a structural challenge for Computer System Validation of such platforms in regulated environments, for which we propose two architectural mitigations. 

---
# KnowU-Bench: Towards Interactive, Proactive, and Personalized Mobile Agent Evaluation 

**Authors**: Tongbo Chen, Zhengxi Lu, Zhan Xu, Guocheng Shao, Shaohan Zhao, Fei Tang, Yong Du, Kaitao Song, Yizhou Liu, Yuchen Yan, Wenqi Zhang, Xu Tan, Weiming Lu, Jun Xiao, Yueting Zhuang, Yongliang Shen  

**Link**: [PDF](https://arxiv.org/pdf/2604.08455)  

**Abstract**: Personalized mobile agents that infer user preferences and calibrate proactive assistance hold great promise as everyday digital assistants, yet existing benchmarks fail to capture what this requires. Prior work evaluates preference recovery from static histories or intent prediction from fixed contexts. Neither tests whether an agent can elicit missing preferences through interaction, nor whether it can decide when to intervene, seek consent, or remain silent in a live GUI environment. We introduce KnowU-Bench, an online benchmark for personalized mobile agents built on a reproducible Android emulation environment, covering 42 general GUI tasks, 86 personalized tasks, and 64 proactive tasks. Unlike prior work that treats user preferences as static context, KnowU-Bench hides the user profile from the agent and exposes only behavioral logs, forcing genuine preference inference rather than context lookup. To support multi-turn preference elicitation, it instantiates an LLM-driven user simulator grounded in structured profiles, enabling realistic clarification dialogues and proactive consent handling. Beyond personalization, KnowU-Bench provides comprehensive evaluation of the complete proactive decision chain, including grounded GUI execution, consent negotiation, and post-rejection restraint, evaluated through a hybrid protocol combining rule-based verification with LLM-as-a-Judge scoring. Our experiments reveal a striking degradation: agents that excel at explicit task execution fall below 50% under vague instructions requiring user preference inference or intervention calibration, even for frontier models like Claude Sonnet 4.6. The core bottlenecks are not GUI navigation but preference acquisition and intervention calibration, exposing a fundamental gap between competent interface operation and trustworthy personal assistance. 

---
# Learning Who Disagrees: Demographic Importance Weighting for Modeling Annotator Distributions with DiADEM 

**Authors**: Samay U. Shetty, Tharindu Cyril Weerasooriya, Deepak Pandita, Christopher M. Homan  

**Link**: [PDF](https://arxiv.org/pdf/2604.08425)  

**Abstract**: When humans label subjective content, they disagree, and that disagreement is not noise. It reflects genuine differences in perspective shaped by annotators' social identities and lived experiences. Yet standard practice still flattens these judgments into a single majority label, and recent LLM-based approaches fare no better: we show that prompted large language models, even with chain-of-thought reasoning, fail to recover the structure of human disagreement. We introduce DiADEM, a neural architecture that learns "how much each demographic axis matters" for predicting who will disagree and on what. DiADEM encodes annotators through per-demographic projections governed by a learned importance vector $\boldsymbol{\alpha}$, fuses annotator and item representations via complementary concatenation and Hadamard interactions, and is trained with a novel item-level disagreement loss that directly penalizes mispredicted annotation variance. On the DICES conversational-safety and VOICED political-offense benchmarks, DiADEM substantially outperforms both the LLM-as-a-judge and neural model baselines across standard and perspectivist metrics, achieving strong disagreement tracking ($r{=}0.75$ on DICES). The learned $\boldsymbol{\alpha}$ weights reveal that race and age consistently emerge as the most influential demographic factors driving annotator disagreement across both datasets. Our results demonstrate that explicitly modeling who annotators are not just what they label is essential for NLP systems that aim to faithfully represent human interpretive diversity. 

---
# On-board Telemetry Monitoring in Autonomous Satellites: Challenges and Opportunities 

**Authors**: Lorenzo Capelli, Leandro de Souza Rosa, Maurizio De Tommasi, Livia Manovi, Andriy Enttsel, Mauro Mangia, Riccardo Rovatti, Ilaria Pinci, Carlo Ciancarelli, Eleonora Mariotti, Gianluca Furano  

**Link**: [PDF](https://arxiv.org/pdf/2604.08424)  

**Abstract**: The increasing autonomy of spacecraft demands fault-detection systems that are both reliable and explainable. This work addresses eXplainable Artificial Intelligence for onboard Fault Detection, Isolation and Recovery within the Attitude and Orbit Control Subsystem by introducing a framework that enhances interpretability in neural anomaly detectors. We propose a method to derive low-dimensional, semantically annotated encodings from intermediate neural activations, called peepholes. Applied to a convolutional autoencoder, the framework produces interpretable indicators that enable the identification and localization of anomalies in reaction-wheel telemetry. Peepholes analysis further reveals bias detection and supports fault localization. The proposed framework enables the semantic characterization of detected anomalies while requiring only a marginal increase in computational resources, thus supporting its feasibility for on-board deployment. 

---
# Verify Before You Commit: Towards Faithful Reasoning in LLM Agents via Self-Auditing 

**Authors**: Wenhao Yuan, Chenchen Lin, Jian Chen, Jinfeng Xu, Xuehe Wang, Edith Cheuk Han Ngai  

**Link**: [PDF](https://arxiv.org/pdf/2604.08401)  

**Abstract**: In large language model (LLM) agents, reasoning trajectories are treated as reliable internal beliefs for guiding actions and updating memory. However, coherent reasoning can still violate logical or evidential constraints, allowing unsupported beliefs repeatedly stored and propagated across decision steps, leading to systematic behavioral drift in long-horizon agentic systems. Most existing strategies rely on the consensus mechanism, conflating agreement with faithfulness. In this paper, inspired by the vulnerability of unfaithful intermediate reasoning trajectories, we propose \textbf{S}elf-\textbf{A}udited \textbf{Ve}rified \textbf{R}easoning (\textsc{SAVeR}), a novel framework that enforces verification over internal belief states within the agent before action commitment, achieving faithful reasoning. Concretely, we structurally generate persona-based diverse candidate beliefs for selection under a faithfulness-relevant structure space. To achieve reasoning faithfulness, we perform adversarial auditing to localize violations and repair through constraint-guided minimal interventions under verifiable acceptance criteria. Extensive experiments on six benchmark datasets demonstrate that our approach consistently improves reasoning faithfulness while preserving competitive end-task performance. 

---
# Awakening the Sleeping Agent: Lean-Specific Agentic Data Reactivates General Tool Use in Goedel Prover 

**Authors**: Jui-Hui Chung, Hongzhou Lin, Lai Jiang, Shange Tang, Chi Jin  

**Link**: [PDF](https://arxiv.org/pdf/2604.08388)  

**Abstract**: Heavy supervised fine-tuning on a target domain can strongly suppress capabilities that were present in the base model. We study this phenomenon in formal mathematics using Goedel-Prover-V2, an open-source model heavily trained on 1.8 million formal-math examples. After domain specialization, the model almost completely loses its ability to produce valid tool calls, even when explicitly instructed to use tools, dropping from 89.4% function-calling accuracy in the base model to nearly 0%. We ask whether this agentic collapse is permanent or instead reversible. To answer this question, we fine-tune the specialized model on a small amount of Lean-specific tool-use data. Remarkably, as few as 100 agentic traces are sufficient to restore strong tool-calling behavior. Importantly, this recovery is not the result of reward hacking or benchmark-specific optimization: the recovery data is entirely drawn from the Lean setting, where the model uses natural-language queries to search the Mathlib library for relevant theorems and lemmas, yet the regained capability transfers well beyond that domain. In particular, these same 100 Lean-specific traces improve performance on the Berkeley Function Calling Leaderboard from near zero to 83.8%, approaching the base model's 89.4% despite the mismatch in task distribution and protocol. The recovered capability is also practically useful in-domain. On ProofNet, pass@32 improves from 21.51% to 25.81%. Together, these results show that heavy domain supervised fine-tuning can suppress general tool-use ability without permanently erasing it, and that a small amount of domain-specific agentic data can awaken dormant tool-use capabilities. 

---
# SkillClaw: Let Skills Evolve Collectively with Agentic Evolver 

**Authors**: Ziyu Ma, Shidong Yang, Yuxiang Ji, Xucong Wang, Yong Wang, Yiming Hu, Tongwen Huang, Xiangxiang Chu  

**Link**: [PDF](https://arxiv.org/pdf/2604.08377)  

**Abstract**: Large language model (LLM) agents such as OpenClaw rely on reusable skills to perform complex tasks, yet these skills remain largely static after deployment. As a result, similar workflows, tool usage patterns, and failure modes are repeatedly rediscovered across users, preventing the system from improving with experience. While interactions from different users provide complementary signals about when a skill works or fails, existing systems lack a mechanism to convert such heterogeneous experiences into reliable skill updates. To address these issues, we present SkillClaw, a framework for collective skill evolution in multi-user agent ecosystems, which treats cross-user and over-time interactions as the primary signal for improving skills. SkillClaw continuously aggregates trajectories generated during use and processes them with an autonomous evolver, which identifies recurring behavioral patterns and translates them into updates to the skill set by refining existing skills or extending them with new capabilities. The resulting skills are maintained in a shared repository and synchronized across users, allowing improvements discovered in one context to propagate system-wide while requiring no additional effort from users. By integrating multi-user experience into ongoing skill updates, SkillClaw enables cross-user knowledge transfer and cumulative capability improvement, and experiments on WildClawBench show that limited interaction and feedback, it significantly improves the performance of Qwen3-Max in real-world agent scenarios. 

---
# Don't Overthink It: Inter-Rollout Action Agreement as a Free Adaptive-Compute Signal for LLM Agents 

**Authors**: Khushal Sethi  

**Link**: [PDF](https://arxiv.org/pdf/2604.08369)  

**Abstract**: Inference-time compute scaling has emerged as a powerful technique for improving the reliability of large language model (LLM) agents, but existing methods apply compute uniformly: every decision step receives the same budget regardless of its difficulty. We introduce TrACE (Trajectorical Adaptive Compute via agrEement), a training-free controller that allocates LLM calls adaptively across agent timesteps by measuring inter-rollout action agreement. At each step, TrACE samples a small set of candidate next actions and measures how consistently the model commits to the same action. High agreement signals an easy decision; the controller commits immediately. Low agreement signals uncertainty; the controller samples additional rollouts up to a configurable cap before committing to the plurality action. No learned components, no external verifier, and no human labels are required. We evaluate TrACE against greedy decoding and fixed-budget self-consistency (SC-4, SC-8) on two benchmarks spanning single-step reasoning (GSM8K, n=50) and multi-step household navigation (MiniHouse, n=30), using a Qwen 2.5 3B Instruct model running on CPU. TrACE-4 matches SC-4 accuracy while using 33% fewer LLM calls on GSM8K and 39% fewer on MiniHouse. TrACE-8 matches SC-8 accuracy with 55% fewer calls on GSM8K and 65% fewer on MiniHouse. We further show that inter-rollout agreement is a reliable signal of step-level success, validating the core hypothesis that the model's own output consistency encodes difficulty information that can be exploited without training. TrACE is the first training-free, per-timestep adaptive-compute controller for LLM agents to be evaluated on multi-step sequential decision tasks. 

---
# ASPECT:Analogical Semantic Policy Execution via Language Conditioned Transfer 

**Authors**: Ajsal Shereef Palattuparambil, Thommen George Karimpanal, Santu Rana  

**Link**: [PDF](https://arxiv.org/pdf/2604.08355)  

**Abstract**: Reinforcement Learning (RL) agents often struggle to generalize knowledge to new tasks, even those structurally similar to ones they have mastered. Although recent approaches have attempted to mitigate this issue via zero-shot transfer, they are often constrained by predefined, discrete class systems, limiting their adaptability to novel or compositional task variations. We propose a significantly more generalized approach, replacing discrete latent variables with natural language conditioning via a text-conditioned Variational Autoencoder (VAE). Our core innovation utilizes a Large Language Model (LLM) as a dynamic \textit{semantic operator} at test time. Rather than relying on rigid rules, our agent queries the LLM to semantically remap the description of the current observation to align with the source task. This source-aligned caption conditions the VAE to generate an imagined state compatible with the agent's original training, enabling direct policy reuse. By harnessing the flexible reasoning capabilities of LLMs, our approach achieves zero-shot transfer across a broad spectrum of complex and truly novel analogous tasks, moving beyond the limitations of fixed category mappings. Code and videos are available \href{this https URL}{here}. 

---
# Human-AI Collaboration Reconfigures Group Regulation from Socially Shared to Hybrid Co-Regulation 

**Authors**: Yujing Zhang, Xianghui Meng, Shihui Feng, Jionghao Lin  

**Link**: [PDF](https://arxiv.org/pdf/2604.08344)  

**Abstract**: Generative AI (GenAI) is increasingly used in collaborative learning, yet its effects on how groups regulate collaboration remain unclear. Effective collaboration depends not only on what groups discuss, but on how they jointly manage goals, participation, strategy use, monitoring, and repair through co-regulation and socially shared regulation. We compared collaborative regulation between Human-AI and Human-Human groups in a parallel-group randomised experiment with 71 university students completing the same collaborative tasks with GenAI either available or unavailable. Focusing on human discourse, we used statistical analyses to examine differences in the distribution of collaborative regulation across regulatory modes, regulatory processes, and participatory focuses. Results showed that GenAI availability shifted regulation away from predominantly socially shared forms towards more hybrid co-regulatory forms, with selective increases in directive, obstacle-oriented, and affective regulatory processes. Participatory-focus distributions, however, were broadly similar across conditions. These findings suggest that GenAI reshapes the distribution of regulatory responsibility in collaboration and offer implications for the human-centred design of AI-supported collaborative learning. 

---
# ProMedical: Hierarchical Fine-Grained Criteria Modeling for Medical LLM Alignment via Explicit Injection 

**Authors**: He Geng, Yangmin Huang, Lixian Lai, Qianyun Du, Hui Chu, Zhiyang He, Jiaxue Hu, Xiaodong Tao  

**Link**: [PDF](https://arxiv.org/pdf/2604.08326)  

**Abstract**: Aligning Large Language Models (LLMs) with high-stakes medical standards remains a significant challenge, primarily due to the dissonance between coarse-grained preference signals and the complex, multi-dimensional nature of clinical protocols. To bridge this gap, we introduce ProMedical, a unified alignment framework grounded in fine-grained clinical criteria. We first construct ProMedical-Preference-50k, a dataset generated via a human-in-the-loop pipeline that augments medical instructions with rigorous, physician-derived rubrics. Leveraging this corpus, we propose the Explicit Criteria Injection paradigm to train a multi-dimensional reward model. Unlike traditional scalar reward models, our approach explicitly disentangles safety constraints from general proficiency, enabling precise guidance during reinforcement learning. To rigorously validate this framework, we establish ProMedical-Bench, a held-out evaluation suite anchored by double-blind expert adjudication. Empirical evaluations demonstrate that optimizing the Qwen3-8B base model via ProMedical-RM-guided GRPO yields substantial gains, improving overall accuracy by 22.3% and safety compliance by 21.7%, effectively rivaling proprietary frontier models. Furthermore, the aligned policy generalizes robustly to external benchmarks, demonstrating performance comparable to state-of-the-art models on UltraMedical. We publicly release our datasets, reward models, and benchmarks to facilitate reproducible research in safety-aware medical alignment. 

---
# U-CECE: A Universal Multi-Resolution Framework for Conceptual Counterfactual Explanations 

**Authors**: Angeliki Dimitriou, Nikolaos Chaidos, Maria Lymperaiou, Giorgos Filandrianos, Giorgos Stamou  

**Link**: [PDF](https://arxiv.org/pdf/2604.08295)  

**Abstract**: As AI models grow more complex, explainability is essential for building trust, yet concept-based counterfactual methods still face a trade-off between expressivity and efficiency. Representing underlying concepts as atomic sets is fast but misses relational context, whereas full graph representations are more faithful but require solving the NP-hard Graph Edit Distance (GED) problem. We propose U-CECE, a unified, model-agnostic multi-resolution framework for conceptual counterfactual explanations that adapts to data regime and compute budget. U-CECE spans three levels of expressivity: atomic concepts for broad explanations, relational sets-of-sets for simple interactions, and structural graphs for full semantic structure. At the structural level, both a precision-oriented transductive mode based on supervised Graph Neural Networks (GNNs) and a scalable inductive mode based on unsupervised graph autoencoders (GAEs) are supported. Experiments on the structurally divergent CUB and Visual Genome datasets characterize the efficiency-expressivity trade-off across levels, while human surveys and LVLM-based evaluation show that the retrieved structural counterfactuals are semantically equivalent to, and often preferred over, exact GED-based ground-truth explanations. 

---
# ACF: A Collaborative Framework for Agent Covert Communication under Cognitive Asymmetry 

**Authors**: Wansheng Wu, Kaibo Huang, Yukun Wei, Zhongliang Yang, Linna Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2604.08276)  

**Abstract**: As generative artificial intelligence evolves, autonomous agent networks present a powerful paradigm for interactive covert communication. However, because agents dynamically update internal memories via environmental interactions, existing methods face a critical structural vulnerability: cognitive asymmetry. Conventional approaches demand strict cognitive symmetry, requiring identical sequence prefixes between the encoder and decoder. In dynamic deployments, inevitable prefix discrepancies destroy synchronization, inducing severe channel degradation. To address this core challenge of cognitive asymmetry, we propose the Asymmetric Collaborative Framework (ACF), which structurally decouples covert communication from semantic reasoning via orthogonal statistical and cognitive layers. By deploying a prefix-independent decoding paradigm governed by a shared steganographic configuration, ACF eliminates the reliance on cognitive symmetry. Evaluations on realistic memory-augmented workflows demonstrate that under severe cognitive asymmetry, symmetric baselines suffer severe channel degradation, whereas ACF uniquely excels across both semantic fidelity and covert communication. It maintains computational indistinguishability, enabling reliable secret extraction with provable error bounds, and providing robust Effective Information Capacity guarantees for modern agent networks. 

---
# Neural-Symbolic Knowledge Tracing: Injecting Educational Knowledge into Deep Learning for Responsible Learner Modelling 

**Authors**: Danial Hooshyar, Gustav Šír, Yeongwook Yang, Tommi Kärkkäinen, Raija Hämäläinen, Ekaterina Krivich, Mutlu Cukurova, Dragan Gašević, Roger Azevedo  

**Link**: [PDF](https://arxiv.org/pdf/2604.08263)  

**Abstract**: The growing use of artificial intelligence (AI) in education, particularly large language models (LLMs), has increased interest in intelligent tutoring systems. However, LLMs often show limited adaptivity and struggle to model learners' evolving knowledge over time, highlighting the need for dedicated learner modelling approaches. Although deep knowledge tracing methods achieve strong predictive performance, their opacity and susceptibility to bias can limit alignment with pedagogical principles. To address this, we propose Responsible-DKT, a neural-symbolic deep knowledge tracing approach that integrates symbolic educational knowledge (e.g., mastery and non-mastery rules) into sequential neural models for responsible learner modelling. Experiments on a real-world dataset of students' math interactions show that Responsible-DKT outperforms both a neural-symbolic baseline and a fully data-driven PyTorch DKT model across training settings. The model achieves over 0.80 AUC with only 10% of training data and up to 0.90 AUC, improving performance by up to 13%. It also demonstrates improved temporal reliability, producing lower early- and mid-sequence prediction errors and the lowest prediction inconsistency rates across sequence lengths, indicating that prediction updates remain directionally aligned with observed student responses over time. Furthermore, the neural-symbolic approach offers intrinsic interpretability via a grounded computation graph that exposes the logic behind each prediction, enabling both local and global explanations. It also allows empirical evaluation of pedagogical assumptions, revealing that repeated incorrect responses (non-mastery) strongly influence prediction updates. These results indicate that neural-symbolic approaches enhance both performance and interpretability, mitigate data limitations, and support more responsible, human-centered AI in education. 

---
# From Phenomenological Fitting to Endogenous Deduction: A Paradigm Leap via Meta-Principle Physics Architecture 

**Authors**: Helong Hu, HongDan Pan, ShuiQing Hu  

**Link**: [PDF](https://arxiv.org/pdf/2604.08245)  

**Abstract**: The essence of current neural network architectures is phenomenological fitting: they learn input-output statistical correlations via massive parameters and data, yet lack intrinsic understanding of the fundamental principles governing physical reality. This paper proposes a paradigm leap from pure phenomenological fitting to the fusion of phenomenological fitting and endogenous deduction. By embedding physical meta-principles into neural network architecture, we construct the Meta-Principle Physics Architecture (MPPA).
Specifically, MPPA embeds three core meta-principles - Connectivity, Conservation, Periodicity - into its architecture, implemented via three core components: the Gravitator realizes Connectivity via standard causal attention; the Energy Encoder implements Conservation via log-domain energy tracking and delayed compensation; the Periodicity Encoder fulfills Periodicity via FFT-based spectral analysis and delayed modulation. These components collaborate via a learnable independent gating fusion mechanism, forming a complete physical cognition framework of 'local relational connectivity - global conservation constraint - evolutionary periodic law'.
Experiments show MPPA achieves significant improvements: physical reasoning (from near zero to 0.436, 0.436 vs 0.000), 2.18x mathematical task improvement (0.330 vs 0.151), 52% logical task gain (0.456 vs 0.300), and 3.69% lower validation perplexity (259.45 vs 269.40), with only 11.8% more parameters (242.40M vs 216.91M). Notably, MPPA shows strong generalization on out-of-distribution physical scenarios, proving the robustness and interpretability of this principle-embedded design. This work establishes a new theoretical foundation and technical path for next-generation AI with physical common sense, causal reasoning, and mathematical rigor. 

---
# HiRO-Nav: Hybrid ReasOning Enables Efficient Embodied Navigation 

**Authors**: He Zhao, Yijun Yang, Zichuan Lin, Deheng Ye, Chunyan Miao  

**Link**: [PDF](https://arxiv.org/pdf/2604.08232)  

**Abstract**: Embodied navigation agents built upon large reasoning models (LRMs) can handle complex, multimodal environmental input and perform grounded reasoning per step to improve sequential decision-making for long-horizon tasks. However, a critical question remains: \textit{how can the reasoning capabilities of LRMs be harnessed intelligently and efficiently for long-horizon navigation tasks?} In simple scenes, agents are expected to act reflexively, while in complex ones they should engage in deliberate reasoning before this http URL achieve this, we introduce \textbf{H}ybr\textbf{i}d \textbf{R}eas\textbf{O}ning \textbf{Nav}igation (\textbf{HiRO-Nav}) agent, the first kind of agent capable of adaptively determining whether to perform thinking at every step based on its own action entropy. Specifically, by examining how the agent's action entropy evolves over the navigation trajectories, we observed that only a small fraction of actions exhibit high entropy, and these actions often steer the agent toward novel scenes or critical objects. Furthermore, studying the relationship between action entropy and task completion (i.e., Q-value) reveals that improving high-entropy actions contributes more positively to task this http URL, we propose a tailored training pipeline comprising hybrid supervised fine-tuning as a cold start, followed by online reinforcement learning with the proposed hybrid reasoning strategy to explicitly activate reasoning only for high-entropy actions, significantly reducing computational overhead while improving decision quality. Extensive experiments on the \textsc{CHORES}-$\mathbb{S}$ ObjectNav benchmark showcases that HiRO-Nav achieves a better trade-off between success rates and token efficiency than both dense-thinking and no-thinking baselines. 

---
# Grounding Clinical AI Competency in Human Cognition Through the Clinical World Model and Skill-Mix Framework 

**Authors**: Seyed Amir Ahmad Safavi-Naini, Elahe Meftah, Josh Mohess, Pooya Mohammadi Kazaj, Georgios Siontis, Zahra Atf, Peter R. Lewis, Mauricio Reyes, Girish Nadkarni, Roland Wiest, Stephan Windecker, Christoph Grani, Ali Soroush, Isaac Shiri  

**Link**: [PDF](https://arxiv.org/pdf/2604.08226)  

**Abstract**: The competency of any intelligent agent is bounded by its formal account of the world in which it operates. Clinical AI lacks such an account. Existing frameworks address evaluation, regulation, or system design in isolation, without a shared model of the clinical world to connect them. We introduce the Clinical World Model, a framework that formalizes care as a tripartite interaction among Patient, Provider, and Ecosystem. To formalize how any agent, whether human or artificial, transforms information into clinical action, we develop parallel decision-making architectures for providers, patients, and AI agents, grounded in validated principles of clinical cognition.
The Clinical AI Skill-Mix operationalizes competency through eight dimensions. Five define the clinical competency space (condition, phase, care setting, provider role, and task) and three specify how AI engages human reasoning (assigned authority, agent facing, and anchoring layer). The combinatorial product of these dimensions yields a space of billions of distinct competency coordinates. A central structural implication is that validation within one coordinate provides minimal evidence for performance in another, rendering the competency space irreducible. The framework supplies a common grammar through which clinical AI can be specified, evaluated, and bounded across stakeholders. By making this structure explicit, the Clinical World Model reframes the field's central question from whether AI works to in which competency coordinates reliability has been demonstrated, and for whom. 

---
# Aligning Agents via Planning: A Benchmark for Trajectory-Level Reward Modeling 

**Authors**: Jiaxuan Wang, Yulan Hu, Wenjin Yang, Zheng Pan, Xin Li, Lan-Zhe Guo  

**Link**: [PDF](https://arxiv.org/pdf/2604.08178)  

**Abstract**: In classical Reinforcement Learning from Human Feedback (RLHF), Reward Models (RMs) serve as the fundamental signal provider for model alignment. As Large Language Models evolve into agentic systems capable of autonomous tool invocation and complex reasoning, the paradigm of reward modeling faces unprecedented challenges--most notably, the lack of benchmarks specifically designed to assess RM capabilities within tool-integrated environments. To address this gap, we present Plan-RewardBench, a trajectory-level preference benchmark designed to evaluate how well judges distinguish preferred versus distractor agent trajectories in complex tool-using scenarios. Plan-RewardBench covers four representative task families -- (i) Safety Refusal, (ii) Tool-Irrelevance / Unavailability, (iii) Complex Planning, and (iv) Robust Error Recovery -- comprising validated positive trajectories and confusable hard negatives constructed via multi-model natural rollouts, rule-based perturbations, and minimal-edit LLM perturbations. We benchmark representative RMs (generative, discriminative, and LLM-as-Judge) under a unified pairwise protocol, reporting accuracy trends across varying trajectory lengths and task categories. Furthermore, we provide diagnostic analyses of prevalent failure modes. Our results reveal that all three evaluator families face substantial challenges, with performance degrading sharply on long-horizon trajectories, underscoring the necessity for specialized training in agentic, trajectory-level reward modeling. Ultimately, Plan-RewardBench aims to serve as both a practical evaluation suite and a reusable blueprint for constructing agentic planning preference data. 

---
# Activation Steering for Aligned Open-ended Generation without Sacrificing Coherence 

**Authors**: Niklas Herbster, Martin Zborowski, Alberto Tosato, Gauthier Gidel, Tommaso Tosato  

**Link**: [PDF](https://arxiv.org/pdf/2604.08169)  

**Abstract**: Alignment in LLMs is more brittle than commonly assumed: misalignment can be triggered by adversarial prompts, benign fine-tuning, emergent misalignment, and goal misgeneralization. Recent evidence suggests that some misalignment behaviors are encoded as linear structure in activation space, making it tractable via steering, while safety alignment has been shown to govern the first few output tokens primarily, leaving subsequent generation unguarded. These findings motivate activation steering as a lightweight runtime defense that continuously corrects misaligned activations throughout generation. We evaluate three methods: Steer-With-Fixed-Coeff (SwFC), which applies uniform additive steering, and two novel projection-aware methods, Steer-to-Target-Projection (StTP) and Steer-to-Mirror-Projection (StMP), that use a logistic regression decision boundary to selectively intervene only on tokens whose activations fall below distributional thresholds. Using malicious system prompts as a controlled proxy for misalignment, we evaluate under two threat models (dishonesty and dismissiveness) and two architectures (Llama-3.3-70B-Instruct, Qwen3-32B). All methods substantially recover target traits (honesty and compassion) while preserving coherence. StTP and StMP better maintain general capabilities (MMLU, MT-Bench, AlpacaEval) and produce less repetition in multi-turn conversations. 

---
# Beyond Stochastic Exploration: What Makes Training Data Valuable for Agentic Search 

**Authors**: Chuzhan Hao, Wenfeng Feng, Guochao Jiang, Guofeng Quan, Guohua Liu, Yuewei Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2604.08124)  

**Abstract**: Reinforcement learning (RL) has become an effective approach for advancing the reasoning capabilities of large language models (LLMs) through the strategic integration of external search engines. However, current RL-based search agents often rely on a process of stochastic exploration guided by carefully crafted outcome rewards, leading to inefficient reasoning trajectories and unstable training. To address these issues, we propose a novel framework, Hierarchical Experience (HiExp), to enhance the performance and training stability of search agents. Specifically, we extract empirical knowledge through contrastive analysis and a multi-level clustering mechanism, transforming raw reasoning trajectories into hierarchical experience knowledge. By leveraging experience-aligned training, we effectively regularize stochastic exploration, evolving it into a strategic and experience-driven search process. Extensive evaluations on multiple complex agentic search and mathematical reasoning benchmarks demonstrate that our approach not only achieves substantial performance gains but also exhibits strong cross-task and cross-algorithm generalization. 

---
# Revise: A Framework for Revising OCRed text in Practical Information Systems with Data Contamination Strategy 

**Authors**: Gyuho Shim, Seongtae Hong, Heuiseok Lim  

**Link**: [PDF](https://arxiv.org/pdf/2604.08115)  

**Abstract**: Recent advances in Large Language Models (LLMs) have significantly improved the field of Document AI, demonstrating remarkable performance on document understanding tasks such as question answering. However, existing approaches primarily focus on solving specific tasks, lacking the capability to structurally organize and manage document information. To address this limitation, we propose Revise, a framework that systematically corrects errors introduced by OCR at the character, word, and structural levels. Specifically, Revise employs a comprehensive hierarchical taxonomy of common OCR errors and a synthetic data generation strategy that realistically simulates such errors to train an effective correction model. Experimental results demonstrate that Revise effectively corrects OCR outputs, enabling more structured representation and systematic management of document contents. Consequently, our method significantly enhances downstream performance in document retrieval and question answering tasks, highlighting the potential to overcome the structural management limitations of existing Document AI frameworks. 

---
# ImplicitMemBench: Measuring Unconscious Behavioral Adaptation in Large Language Models 

**Authors**: Chonghan Qin, Xiachong Feng, Weitao Ma, Xiaocheng Feng, Lingpeng Kong  

**Link**: [PDF](https://arxiv.org/pdf/2604.08064)  

**Abstract**: Existing memory benchmarks for LLM agents evaluate explicit recall of facts, yet overlook implicit memory where experience becomes automated behavior without conscious retrieval. This gap is critical: effective assistants must automatically apply learned procedures or avoid failed actions without explicit reminders. We introduce ImplicitMemBench, the first systematic benchmark evaluating implicit memory through three cognitively grounded constructs drawn from standard cognitive-science accounts of non-declarative memory: Procedural Memory (one-shot skill acquisition after interference), Priming (theme-driven bias via paired experimental/control instances), and Classical Conditioning (Conditioned Stimulus--Unconditioned Stimulus (CS--US) associations shaping first decisions). Our 300-item suite employs a unified Learning/Priming-Interfere-Test protocol with first-attempt scoring. Evaluation of 17 models reveals severe limitations: no model exceeds 66% overall, with top performers DeepSeek-R1 (65.3%), Qwen3-32B (64.1%), and GPT-5 (63.0%) far below human baselines. Analysis uncovers dramatic asymmetries (inhibition 17.6% vs. preference 75.0%) and universal bottlenecks requiring architectural innovations beyond parameter scaling. ImplicitMemBench reframes evaluation from "what agents recall" to "what they automatically enact". 

---
# IoT-Brain: Grounding LLMs for Semantic-Spatial Sensor Scheduling 

**Authors**: Zhaomeng Zhou, Lan Zhang, Junyang Wang, Mu Yuan, Junda Lin, Jinke Song  

**Link**: [PDF](https://arxiv.org/pdf/2604.08033)  

**Abstract**: Intelligent systems powered by large-scale sensor networks are shifting from predefined monitoring to intent-driven operation, revealing a critical Semantic-to-Physical Mapping Gap. While large language models (LLMs) excel at semantic understanding, existing perception-centric pipelines operate retrospectively, overlooking the fundamental decision of what to sense and when. We formalize this proactive decision as Semantic-Spatial Sensor Scheduling (S3) and demonstrate that direct LLM planning is unreliable due to inherent gaps in representation, reasoning, and optimization. To bridge these gaps, we introduce the Spatial Trajectory Graph (STG), a neuro-symbolic paradigm governed by a verify-before-commit discipline that transforms open-ended planning into a verifiable graph optimization problem. Based on STG, we implement IoT-Brain, a concrete system embodiment, and construct TopoSense-Bench, a campus-scale benchmark with 5,250 natural-language queries across 2,510 cameras. Evaluations show that IoT-Brain boosts task success rate by 37.6% over the strongest search-intensive methods while running nearly 2 times faster and using 6.6 times fewer prompt tokens. In real-world deployment, it approaches the reliability upper bound while reducing 4.1 times network bandwidth, providing a foundational framework for LLMs to interact with the physical world with unprecedented reliability and efficiency. 

---
# "Why This Avoidance Maneuver?" Contrastive Explanations in Human-Supervised Maritime Autonomous Navigation 

**Authors**: Joel Jose, Andreas Madsen, Andreas Brandsæter, Tor A. Johansen, Erlend M. Coates  

**Link**: [PDF](https://arxiv.org/pdf/2604.08032)  

**Abstract**: Automated maritime collision avoidance will rely on human supervision for the foreseeable future. This necessitates transparency into how the system perceives a scenario and plans a maneuver. However, the causal logic behind avoidance maneuvers is often complex and difficult to convey to a navigator. This paper explores how to explain these factors in a selective, understandable manner for supervisors with a nautical background. We propose a method for generating contrastive explanations, which provide human-centric insights by comparing a system's proposed solution against relevant alternatives. To evaluate this, we developed a framework that uses visual and textual cues to highlight key objectives from a state-of-the-art collision avoidance system. An exploratory user study with four experienced marine officers suggests that contrastive explanations support the understanding of the system's objectives. However, our findings also reveal that while these explanations are highly valuable in complex multi-vessel encounters, they can increase cognitive workload, suggesting that future maritime interfaces may benefit most from demand-driven or scenario-specific explanation strategies. 

---
# Wiring the 'Why': A Unified Taxonomy and Survey of Abductive Reasoning in LLMs 

**Authors**: Moein Salimi, Shaygan Adim, Danial Parnian, Nima Alighardashi, Mahdi Jafari Siavoshani, Mohammad Hossein Rohban  

**Link**: [PDF](https://arxiv.org/pdf/2604.08016)  

**Abstract**: Regardless of its foundational role in human discovery and sense-making, abductive reasoning--the inference of the most plausible explanation for an observation--has been relatively underexplored in Large Language Models (LLMs). Despite the rapid advancement of LLMs, the exploration of abductive reasoning and its diverse facets has thus far been disjointed rather than cohesive. This paper presents the first survey of abductive reasoning in LLMs, tracing its trajectory from philosophical foundations to contemporary AI implementations. To address the widespread conceptual confusion and disjointed task definitions prevalent in the field, we establish a unified two-stage definition that formally categorizes prior work. This definition disentangles abduction into \textit{Hypothesis Generation}, where models bridge epistemic gaps to produce candidate explanations, and \textit{Hypothesis Selection}, where the generated candidates are evaluated and the most plausible explanation is chosen. Building upon this foundation, we present a comprehensive taxonomy of the literature, categorizing prior work based on their abductive tasks, datasets, underlying methodologies, and evaluation strategies. In order to ground our framework empirically, we conduct a compact benchmark study of current LLMs on abductive tasks, together with targeted comparative analyses across model sizes, model families, evaluation styles, and the distinct generation-versus-selection task typologies. Moreover, by synthesizing recent empirical results, we examine how LLM performance on abductive reasoning relates to deductive and inductive tasks, providing insights into their broader reasoning capabilities. Our analysis reveals critical gaps in current approaches--from static benchmark design and narrow domain coverage to narrow training frameworks and limited mechanistic understanding of abductive processes... 

---
# Evaluating Counterfactual Explanation Methods on Incomplete Inputs 

**Authors**: Francesco Leofante, Daniel Neider, Mustafa Yalçıner  

**Link**: [PDF](https://arxiv.org/pdf/2604.08004)  

**Abstract**: Existing algorithms for generating Counterfactual Explanations (CXs) for Machine Learning (ML) typically assume fully specified inputs. However, real-world data often contains missing values, and the impact of these incomplete inputs on the performance of existing CX methods remains unexplored. To address this gap, we systematically evaluate recent CX generation methods on their ability to provide valid and plausible counterfactuals when inputs are incomplete. As part of this investigation, we hypothesize that robust CX generation methods will be better suited to address the challenge of providing valid and plausible counterfactuals when inputs are incomplete. Our findings reveal that while robust CX methods achieve higher validity than non-robust ones, all methods struggle to find valid counterfactuals. These results motivate the need for new CX methods capable of handling incomplete inputs. 

---
# PASK: Toward Intent-Aware Proactive Agents with Long-Term Memory 

**Authors**: Zhifei Xie, Zongzheng Hu, Fangda Ye, Xin Zhang, Haobo Chai, Zihang Liu, Pengcheng Wu, Guibin Zhang, Yue Liao, Xiaobin Hu, Deheng Ye, Chunyan Miao, Shuicheng Yan  

**Link**: [PDF](https://arxiv.org/pdf/2604.08000)  

**Abstract**: Proactivity is a core expectation for AGI. Prior work remains largely confined to laboratory settings, leaving a clear gap in real-world proactive agent: depth, complexity, ambiguity, precision and real-time constraints. We study this setting, where useful intervention requires inferring latent needs from ongoing context and grounding actions in evolving user memory under latency and long-horizon constraints. We first propose DD-MM-PAS (Demand Detection, Memory Modeling, Proactive Agent System) as a general paradigm for streaming proactive AI agent. We instantiate this paradigm in Pask, with streaming IntentFlow model for DD, a hybrid memory (workspace, user, global) for long-term MM, PAS infra framework and introduce how these components form a closed loop. We also introduce LatentNeeds-Bench, a real-world benchmark built from user-consented data and refined through thousands of rounds of human editing. Experiments show that IntentFlow matches leading Gemini3-Flash models under latency constraints, while identifying deeper user intent. 

---
# How Far Are Large Multimodal Models from Human-Level Spatial Action? A Benchmark for Goal-Oriented Embodied Navigation in Urban Airspace 

**Authors**: Baining Zhao, Ziyou Wang, Jianjie Fang, Zile Zhou, Yanggang Xu, Yatai Ji, Jiacheng Xu, Qian Zhang, Weichen Zhang, Chen Gao, Xinlei Chen  

**Link**: [PDF](https://arxiv.org/pdf/2604.07973)  

**Abstract**: Large multimodal models (LMMs) show strong visual-linguistic reasoning but their capacity for spatial decision-making and action remains unclear. In this work, we investigate whether LMMs can achieve embodied spatial action like human through a challenging scenario: goal-oriented navigation in urban 3D spaces. We first spend over 500 hours constructing a dataset comprising 5,037 high-quality goal-oriented navigation samples, with an emphasis on 3D vertical actions and rich urban semantic information. Then, we comprehensively assess 17 representative models, including non-reasoning LMMs, reasoning LMMs, agent-based methods, and vision-language-action models. Experiments show that current LMMs exhibit emerging action capabilities, yet remain far from human-level performance. Furthermore, we reveal an intriguing phenomenon: navigation errors do not accumulate linearly but instead diverge rapidly from the destination after a critical decision bifurcation. The limitations of LMMs are investigated by analyzing their behavior at these critical decision bifurcations. Finally, we experimentally explore four promising directions for improvement: geometric perception, cross-view understanding, spatial imagination, and long-term memory. The project is available at: this https URL. 

---
# Are we still able to recognize pearls? Machine-driven peer review and the risk to creativity: An explainable RAG-XAI detection framework with markers extraction 

**Authors**: Alin-Gabriel Văduva, Simona-Vasilica Oprea, Adela Bâra  

**Link**: [PDF](https://arxiv.org/pdf/2604.07964)  

**Abstract**: The integration of large language models (LLMs) into peer review raises a concern beyond authorship and detection: the potential cascading automation of the entire editorial process. As reviews become partially or fully machine-generated, it becomes plausible that editorial decisions may also be delegated to algorithmic systems, leading to a fully automated evaluation pipeline. They risk reshaping the criteria by which scientific work is assessed. This paper argues that machine-driven assessment may systematically favor standardized, pattern-conforming research while penalizing unconventional and paradigm-shifting ideas that require contextual human judgment. We consider that this shift could lead to epistemic homogenization, where researchers are implicitly incentivized to optimize their work for algorithmic approval rather than genuine discovery. To address this risk, we introduce an explainable framework (RAG-XAI) for assessing review quality and detecting automated patterns using markers LLM extractor, aiming to preserve transparency, accountability and creativity in science. The proposed framework achieves near-perfect detection performance, with XGBoost, Random Forest and LightGBM reaching 99.61% accuracy, AUC-ROC above 0.999 and F1-scores of 0.9925 on the test set, while maintaining extremely low false positive rates (<0.23%) and false negative rates (~0.8%). In contrast, the logistic regression baseline performs substantially worse (89.97% accuracy, F1-score 0.8314). Feature importance and SHAP analyses identify absence of personal signals and repetition patterns as the dominant predictors. Additionally, the RAG component achieves 90.5% top-1 retrieval accuracy, with strong same-class clustering in the embedding space, further supporting the reliability of the framework's outputs. 

---
# WorldMAP: Bootstrapping Vision-Language Navigation Trajectory Prediction with Generative World Models 

**Authors**: Hongjin Chen, Shangyun Jiang, Tonghua Su, Chen Gao, Xinlei Chen, Yong Li, Zhibo Chen  

**Link**: [PDF](https://arxiv.org/pdf/2604.07957)  

**Abstract**: Vision-language models (VLMs) and generative world models are opening new opportunities for embodied navigation. VLMs are increasingly used as direct planners or trajectory predictors, while world models support look-ahead reasoning by imagining future views. Yet predicting a reliable trajectory from a single egocentric observation remains challenging. Current VLMs often generate unstable trajectories, and world models, though able to synthesize plausible futures, do not directly provide the grounded signals needed for navigation learning. This raises a central question: how can generated futures be turned into supervision for grounded trajectory prediction? We present WorldMAP, a teacher--student framework that converts world-model-generated futures into persistent semantic-spatial structure and planning-derived supervision. Its world-model-driven teacher builds semantic-spatial memory from generated videos, grounds task-relevant targets and obstacles, and produces trajectory pseudo-labels through explicit planning. A lightweight student with a multi-hypothesis trajectory head is then trained to predict navigation trajectories directly from vision-language inputs. On Target-Bench, WorldMAP achieves the best ADE and FDE among compared methods, reducing ADE by 18.0% and FDE by 42.1% relative to the best competing baseline, while lifting a small open-source VLM to DTW performance competitive with proprietary models. More broadly, the results suggest that, in embodied navigation, the value of world models may lie less in supplying action-ready imagined evidence than in synthesizing structured supervision for navigation learning. 

---
# MONETA: Multimodal Industry Classification through Geographic Information with Multi Agent Systems 

**Authors**: Arda Yüksel, Gabriel Thiem, Susanne Walter, Patrick Felka, Gabriela Alves Werb, Ivan Habernal  

**Link**: [PDF](https://arxiv.org/pdf/2604.07956)  

**Abstract**: Industry classification schemes are integral parts of public and corporate databases as they classify businesses based on economic activity. Due to the size of the company registers, manual annotation is costly, and fine-tuning models with every update in industry classification schemes requires significant data collection. We replicate the manual expert verification by using existing or easily retrievable multimodal resources for industry classification. We present MONETA, the first multimodal industry classification benchmark with text (Website, Wikipedia, Wikidata) and geospatial sources (OpenStreetMap and satellite imagery). Our dataset enlists 1,000 businesses in Europe with 20 economic activity labels according to EU guidelines (NACE). Our training-free baseline reaches 62.10% and 74.10% with open and closed-source Multimodal Large Language Models (MLLM). We observe an increase of up to 22.80% with the combination of multi-turn design, context enrichment, and classification explanations. We will release our dataset and the enhanced guidelines. 

---
# EigentSearch-Q+: Enhancing Deep Research Agents with Structured Reasoning Tools 

**Authors**: Boer Zhang, Mingyan Wu, Dongzhuoran Zhou, Yuqicheng Zhu, Wendong Fan, Puzhen Zhang, Zifeng Ding, Guohao Li, Yuan He  

**Link**: [PDF](https://arxiv.org/pdf/2604.07927)  

**Abstract**: Deep research requires reasoning over web evidence to answer open-ended questions, and it is a core capability for AI agents. Yet many deep research agents still rely on implicit, unstructured search behavior that causes redundant exploration and brittle evidence aggregation. Motivated by Anthropic's "think" tool paradigm and insights from the information-retrieval literature, we introduce Q+, a set of query and evidence processing tools that make web search more deliberate by guiding query planning, monitoring search progress, and extracting evidence from long web snapshots. We integrate Q+ into the browser sub-agent of Eigent, an open-source, production-ready multi-agent workforce for computer use, yielding EigentSearch-Q+. Across four benchmarks (SimpleQA-Verified, FRAMES, WebWalkerQA, and X-Bench DeepSearch), Q+ improves Eigent's browser agent benchmark-size-weighted average accuracy by 3.0, 3.8, and 0.6 percentage points (pp) for GPT-4.1, GPT-5.1, and Minimax M2.5 model backends, respectively. Case studies further suggest that EigentSearch-Q+ produces more coherent tool-calling trajectories by making search progress and evidence handling explicit. 

---
# SAT: Balancing Reasoning Accuracy and Efficiency with Stepwise Adaptive Thinking 

**Authors**: Weiyang Huang, Xuefeng Bai, Kehai Chen, Xinyang Chen, Yibin Chen, Weili Guan, Min Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2604.07922)  

**Abstract**: Large Reasoning Models (LRMs) have revolutionized complex problem-solving, yet they exhibit a pervasive "overthinking", generating unnecessarily long reasoning chains. While current solutions improve token efficiency, they often sacrifice fine-grained control or risk disrupting the logical integrity of the reasoning process. To address this, we introduce Stepwise Adaptive Thinking (SAT), a framework that performs step-level, difficulty-aware pruning while preserving the core reasoning structure. SAT formulates reasoning as a Finite-State Machine (FSM) with distinct thinking modes (Slow, Normal, Fast, Skip). It navigates these states dynamically using a lightweight Process Reward Model (PRM), compressing easy steps while preserving depth for hard ones. Experiments across 9 LRMs and 7 benchmarks show that SAT achieves up to 40% reduction in reasoning tokens while generally maintaining or improving accuracy. 

---
# Capture-Quiet Decomposition: A Verification Theorem for Chess Endgame Tablebases 

**Authors**: Alexander Pavlov  

**Link**: [PDF](https://arxiv.org/pdf/2604.07907)  

**Abstract**: We present the Capture-Quiet Decomposition (CQD), a structural theorem for verifying Win-Draw-Loss (WDL) labelings of chess endgame tablebases. The theorem decomposes every legal position into exactly one of three categories -- terminal, capture, or quiet -- and shows that a WDL labeling is correct if and only if: (1) terminal positions are labeled correctly, (2) capture positions are consistent with verified sub-models of smaller piece count, and (3) quiet positions satisfy retrograde consistency within the same endgame. The key insight is that capture positions anchor the labeling to externally verified sub-models, breaking the circularity that allows trivial fixpoints (such as the all-draw labeling) to satisfy self-consistency alone. We validate CQD exhaustively on all 35 three- and four-piece endgames (42 million positions), all 110 five-piece endgames, and all 372 six-piece endgames -- 517 endgames in total -- with the decomposed verifier producing identical violation counts to a full retrograde baseline in every case. 

---
# Visual Perceptual to Conceptual First-Order Rule Learning Networks 

**Authors**: Kun Gao, Davide Soldà, Thomas Eiter, Katsumi Inoue  

**Link**: [PDF](https://arxiv.org/pdf/2604.07897)  

**Abstract**: Learning rules plays a crucial role in deep learning, particularly in explainable artificial intelligence and enhancing the reasoning capabilities of large language models. While existing rule learning methods are primarily designed for symbolic data, learning rules from image data without supporting image labels and automatically inventing predicates remains a challenge. In this paper, we tackle these inductive rule learning problems from images with a framework called {\gamma}ILP, which provides a fully differentiable pipeline from image constant substitution to rule structure induction. Extensive experiments demonstrate that {\gamma}ILP achieves strong performance not only on classical symbolic relational datasets but also on relational image data and pure image datasets, such as Kandinsky patterns. 

---
# DialBGM: A Benchmark for Background Music Recommendation from Everyday Multi-Turn Dialogues 

**Authors**: Joonhyeok Shin, Jaehoon Kang, Yujun Lee, Hannah Lee, Yejin Lee, Yoonji Park, Kyuhong Shim  

**Link**: [PDF](https://arxiv.org/pdf/2604.07895)  

**Abstract**: Selecting an appropriate background music (BGM) that supports natural human conversation is a common production step in media and interactive systems. In this paper, we introduce dialogue-conditioned BGM recommendation, where a model should select non-intrusive, fitting music for a multi-turn conversation that often contains no music descriptors. To study this novel problem, we present DialBGM, a benchmark of 1,200 open-domain daily dialogues, each paired with four candidate music clips and annotated with human preference rankings. Rankings are determined by background suitability criteria, including contextual relevance, non-intrusiveness, and consistency. We evaluate a wide range of open-source and proprietary models, including audio-language models and multimodal LLMs, and show that current models fall far short of human judgments; no model exceeds 35% Hit@1 when selecting the top-ranked clip. DialBGM provides a standardized benchmark for developing discourse-aware methods for BGM selection and for evaluating both retrieval-based and generative models. 

---
# An Agentic Evaluation Architecture for Historical Bias Detection in Educational Textbooks 

**Authors**: Gabriel Stefan, Adrian-Marius Dumitran  

**Link**: [PDF](https://arxiv.org/pdf/2604.07883)  

**Abstract**: History textbooks often contain implicit biases, nationalist framing, and selective omissions that are difficult to audit at scale. We propose an agentic evaluation architecture comprising a multimodal screening agent, a heterogeneous jury of five evaluative agents, and a meta-agent for verdict synthesis and human escalation. A central contribution is a Source Attribution Protocol that distinguishes textbook narrative from quoted historical sources, preventing the misattribution that causes systematic false positives in single-model evaluators.
In an empirical study on Romanian upper-secondary history textbooks, 83.3\% of 270 screened excerpts were classified as pedagogically acceptable (mean severity 2.9/7), versus 5.4/7 under a zero-shot baseline, demonstrating that agentic deliberation mitigates over-penalization. In a blind human evaluation (18 evaluators, 54 comparisons), the Independent Deliberation configuration was preferred in 64.8\% of cases over both a heuristic variant and the zero-shot baseline. At approximately \$2 per textbook, these results position agentic evaluation architectures as economically viable decision-support tools for educational governance. 

---
# Hidden Biases in Conditioning Autoregressive Models 

**Authors**: Francois Pachet, Pierre Roy  

**Link**: [PDF](https://arxiv.org/pdf/2604.07855)  

**Abstract**: Large language and music models are increasingly used for constrained generation: rhyming lines, fixed meter, inpainting or infilling, positional endings, and other global form requirements. These systems often perform strikingly well, but the induced procedures are usually not exact conditioning of the underlying autoregressive model. This creates a hidden inferential bias, distinct from the better-known notion of bias inherited from the training set: samples are distorted relative to the true constrained distribution, with no generic guarantee of complete coverage of the admissible solution space or of correct conditional probabilities over valid completions. We formalize several exact inference tasks for autoregressive models and prove corresponding hardness results. For succinctly represented autoregressive models whose next-token probabilities are computable in polynomial time, exact sentence-level maximum a posteriori (MAP) decoding is NP-hard. This hardness persists under unary and metrical constraints. On the sampling side, exact conditioned normalization is \#P-hard even for regular constraints such as fixed-length terminal events. Unlike finite-state Markov models, general autoregressive models do not admit a bounded-state dynamic program for these tasks. These results formalize a standard claim in the neural decoding literature: local autoregressive sampling is easy, whereas exact decoding and exact conditioning under global form constraints are computationally intractable in general. 

---
# SPARD: Self-Paced Curriculum for RL Alignment via Integrating Reward Dynamics and Data Utility 

**Authors**: Xuyang Zhi, Peilun zhou, Chengqiang Lu, Hang Lv, Yiwei Liang, Rongyang Zhang, Yan Gao, YI WU, Yao Hu, Hongchao Gu, Defu Lian, Hao Wang, Enhong Chen  

**Link**: [PDF](https://arxiv.org/pdf/2604.07837)  

**Abstract**: The evolution of Large Language Models (LLMs) is shifting the focus from single, verifiable tasks toward complex, open-ended real-world scenarios, imposing significant challenges on the post-training phase. In these settings, the scale and complexity of reward systems have grown significantly, transitioning toward multi-objective formulations that encompass a comprehensive spectrum of model capabilities and application contexts. However, traditional methods typically rely on fixed reward weights, ignoring non-stationary learning dynamics and struggling with data heterogeneity across dimensions. To address these issues, we propose SPARD, a framework that establishes an automated, self-paced curriculum by perceiving learning progress to dynamically adjust multi-objective reward weights and data importance, thereby synchronizing learning intent with data utility for optimal performance. Extensive experiments across multiple benchmarks demonstrate that SPARD significantly enhances model capabilities across all domains. 

---
# Silencing the Guardrails: Inference-Time Jailbreaking via Dynamic Contextual Representation Ablation 

**Authors**: Wenpeng Xing, Moran Fang, Guangtai Wang, Changting Lin, Meng Han  

**Link**: [PDF](https://arxiv.org/pdf/2604.07835)  

**Abstract**: While Large Language Models (LLMs) have achieved remarkable performance, they remain vulnerable to jailbreak attacks that circumvent safety constraints. Existing strategies, ranging from heuristic prompt engineering to computationally intensive optimization, often face significant trade-offs between effectiveness and efficiency. In this work, we propose Contextual Representation Ablation (CRA), a novel inference-time intervention framework designed to dynamically silence model guardrails. Predicated on the geometric insight that refusal behaviors are mediated by specific low-rank subspaces within the model's hidden states, CRA identifies and suppresses these refusal-inducing activation patterns during decoding without requiring expensive parameter updates or training. Empirical evaluation across multiple safety-aligned open-source LLMs demonstrates that CRA significantly outperforms baselines. These results expose the intrinsic fragility of current alignment mechanisms, revealing that safety constraints can be surgically ablated from internal representations, and underscore the urgent need for more robust defenses that secure the model's latent space. 

---
# Automatic Generation of Executable BPMN Models from Medical Guidelines 

**Authors**: Praveen Kumar Menaka Sekar, Ion Matei, Maksym Zhenirovskyy, Hon Yung Wong, Sayuri Kohmura, Shinji Hotta, Akihiro Inomata  

**Link**: [PDF](https://arxiv.org/pdf/2604.07817)  

**Abstract**: We present an end-to-end pipeline that converts healthcare policy documents into executable, data-aware Business Process Model and Notation (BPMN) models using large language models (LLMs) for simulation-based policy evaluation. We address the main challenges of automated policy digitization with four contributions: data-grounded BPMN generation with syntax auto-correction, executable augmentation, KPI instrumentation, and entropy-based uncertainty detection. We evaluate the pipeline on diabetic nephropathy prevention guidelines from three Japanese municipalities, generating 100 models per backend across three LLMs and executing each against 1,000 synthetic patients. On well-structured policies, the pipeline achieves a 100% ground-truth match with perfect per-patient decision agreement. Across all conditions, raw per-patient decision agreement exceeds 92%, and entropy scores increase monotonically with document complexity, confirming that the detector reliably separates unambiguous policies from those requiring targeted human clarification. 

---
# Agentivism: a learning theory for the age of artificial intelligence 

**Authors**: Lixiang Yan, Dragan Gašević  

**Link**: [PDF](https://arxiv.org/pdf/2604.07813)  

**Abstract**: Learning theories have historically changed when the conditions of learning evolved. Generative and agentic AI create a new condition by allowing learners to delegate explanation, writing, problem solving, and other cognitive work to systems that can generate, recommend, and sometimes act on the learner's behalf. This creates a fundamental challenge for learning theory: successful performance can no longer be assumed to indicate learning. Learners may complete tasks effectively with AI support while developing less understanding, weaker judgment, and limited transferable capability. We argue that this problem is not fully captured by existing learning theories. Behaviourism, cognitivism, constructivism, and connectivism remain important, but they do not directly explain when AI-assisted performance becomes durable human capability. We propose Agentivism, a learning theory for human-AI interaction. Agentivism defines learning as durable growth in human capability through selective delegation to AI, epistemic monitoring and verification of AI contributions, reconstructive internalization of AI-assisted outputs, and transfer under reduced support. The importance of Agentivism lies in explaining how learning remains possible when intelligent delegation is easy and human-AI interaction is becoming a persistent and expanding part of human learning. 

---
# Lightweight LLM Agent Memory with Small Language Models 

**Authors**: Jiaquan Zhang, Chaoning Zhang, Shuxu Chen, Zhenzhen Huang, Pengcheng Zheng, Zhicheng Wang, Ping Guo, Fan Mo, Sung-Ho Bae, Jie Zou, Jiwei Wei, Yang Yang  

**Link**: [PDF](https://arxiv.org/pdf/2604.07798)  

**Abstract**: Although LLM agents can leverage tools for complex tasks, they still need memory to maintain cross-turn consistency and accumulate reusable information in long-horizon interactions. However, retrieval-based external memory systems incur low online overhead but suffer from unstable accuracy due to limited query construction and candidate filtering. In contrast, many systems use repeated large-model calls for online memory operations, improving accuracy but accumulating latency over long interactions. We propose LightMem, a lightweight memory system for better agent memory driven by Small Language Models (SLMs). LightMem modularizes memory retrieval, writing, and long-term consolidation, and separates online processing from offline consolidation to enable efficient memory invocation under bounded compute. We organize memory into short-term memory (STM) for immediate conversational context, mid-term memory (MTM) for reusable interaction summaries, and long-term memory (LTM) for consolidated knowledge, and uses user identifiers to support independent retrieval and incremental maintenance in multi-user settings. Online, LightMem operates under a fixed retrieval budget and selects memories via a two-stage procedure: vector-based coarse retrieval followed by semantic consistency re-ranking. Offline, it abstracts reusable interaction evidence and incrementally integrates it into LTM. Experiments show gains across model scales, with an average F1 improvement of about 2.5 on LoCoMo, more effective and low median latency (83 ms retrieval; 581 ms end-to-end). 

---
# SEARL: Joint Optimization of Policy and Tool Graph Memory for Self-Evolving Agents 

**Authors**: Xinshun Feng, Xinhao Song, Lijun Li, Gongshen Liu, Jing Shao  

**Link**: [PDF](https://arxiv.org/pdf/2604.07791)  

**Abstract**: Recent advances in Reinforcement Learning with Verifiable Rewards (RLVR) have demonstrated significant potential in single-turn reasoning tasks. With the paradigm shift toward self-evolving agentic learning, models are increasingly expected to learn from trajectories by synthesizing tools or accumulating explicit experiences. However, prevailing methods typically rely on large-scale LLMs or multi-agent frameworks, which hinder their deployment in resource-constrained environments. The inherent sparsity of outcome-based rewards also poses a substantial challenge, as agents typically receive feedback only upon completion of tasks. To address these limitations, we introduce a Tool-Memory based self-evolving agentic framework SEARL. Unlike approaches that directly utilize interaction experiences, our method constructs a structured experience memory that integrates planning with execution. This provides a novel state abstraction that facilitates generalization across analogous contexts, such as tool reuse. Consequently, agents extract explicit knowledge from historical data while leveraging inter-trajectory correlations to densify reward signals. We evaluate our framework on knowledge reasoning and mathematics tasks, demonstrating its effectiveness in achieving more practical and efficient learning. 

---
# Automotive Engineering-Centric Agentic AI Workflow Framework 

**Authors**: Tong Duy Son, Zhihao Liu, Piero Brigida, Yerlan Akhmetov, Gurudevan Devarajan, Kai Liu, Ajinkya Bhave  

**Link**: [PDF](https://arxiv.org/pdf/2604.07784)  

**Abstract**: Engineering workflows such as design optimization, simulation-based diagnosis, control tuning, and model-based systems engineering (MBSE) are iterative, constraint-driven, and shaped by prior decisions. Yet many AI methods still treat these activities as isolated tasks rather than as parts of a broader workflow. This paper presents Agentic Engineering Intelligence (AEI), an industrial vision framework that models engineering workflows as constrained, history-aware sequential decision processes in which AI agents support engineer-supervised interventions over engineering toolchains. AEI links an offline phase for engineering data processing and workflow-memory construction with an online phase for workflow-state estimation, retrieval, and decision support. A control-theoretic interpretation is also possible, in which engineering objectives act as reference signals, agents act as workflow controllers, and toolchains provide feedback for intervention selection. Representative automotive use cases in suspension design, reinforcement learning tuning, multimodal engineering knowledge reuse, aerodynamic exploration, and MBSE show how diverse workflows can be expressed within a common formulation. Overall, the paper positions engineering AI as a problem of process-level intelligence and outlines a practical roadmap for future empirical validation in industrial settings. 

---
# The Accountability Horizon: An Impossibility Theorem for Governing Human-Agent Collectives 

**Authors**: Haileleol Tibebu  

**Link**: [PDF](https://arxiv.org/pdf/2604.07778)  

**Abstract**: Existing accountability frameworks for AI systems, legal, ethical, and regulatory, rest on a shared assumption: for any consequential outcome, at least one identifiable person had enough involvement and foresight to bear meaningful responsibility. This paper proves that agentic AI systems violate this assumption not as an engineering limitation but as a mathematical necessity once autonomy exceeds a computable threshold. We introduce Human-Agent Collectives, a formalisation of joint human-AI systems where agents are modelled as state-policy tuples within a shared structural causal model. Autonomy is characterised through a four-dimensional information-theoretic profile (epistemic, executive, evaluative, social); collective behaviour through interaction graphs and joint action spaces. We axiomatise legitimate accountability through four minimal properties: Attributability (responsibility requires causal contribution), Foreseeability Bound (responsibility cannot exceed predictive capacity), Non-Vacuity (at least one agent bears non-trivial responsibility), and Completeness (all responsibility must be fully allocated). Our central result, the Accountability Incompleteness Theorem, proves that for any collective whose compound autonomy exceeds the Accountability Horizon and whose interaction graph contains a human-AI feedback cycle, no framework can satisfy all four properties simultaneously. The impossibility is structural: transparency, audits, and oversight cannot resolve it without reducing autonomy. Below the threshold, legitimate frameworks exist, establishing a sharp phase transition. Experiments on 3,000 synthetic collectives confirm all predictions with zero violations. This is the first impossibility result in AI governance, establishing a formal boundary below which current paradigms remain valid and above which distributed accountability mechanisms become necessary. 

---
# ACIArena: Toward Unified Evaluation for Agent Cascading Injection 

**Authors**: Hengyu An, Minxi Li, Jinghuai Zhang, Naen Xu, Chunyi Zhou, Changjiang Li, Xiaogang Xu, Tianyu Du, Shouling Ji  

**Link**: [PDF](https://arxiv.org/pdf/2604.07775)  

**Abstract**: Collaboration and information sharing empower Multi-Agent Systems (MAS) but also introduce a critical security risk known as Agent Cascading Injection (ACI). In such attacks, a compromised agent exploits inter-agent trust to propagate malicious instructions, causing cascading failures across the system. However, existing studies consider only limited attack strategies and simplified MAS settings, limiting their generalizability and comprehensive evaluation. To bridge this gap, we introduce ACIArena, a unified framework for evaluating the robustness of MAS. ACIArena offers systematic evaluation suites spanning multiple attack surfaces (i.e., external inputs, agent profiles, inter-agent messages) and attack objectives (i.e., instruction hijacking, task disruption, information exfiltration). Specifically, ACIArena establishes a unified specification that jointly supports MAS construction and attack-defense modules. It covers six widely used MAS implementations and provides a benchmark of 1,356 test cases for systematically evaluating MAS robustness. Our benchmarking results show that evaluating MAS robustness solely through topology is insufficient; robust MAS require deliberate role design and controlled interaction patterns. Moreover, defenses developed in simplified environments often fail to transfer to real-world settings; narrowly scoped defenses may even introduce new vulnerabilities. ACIArena aims to provide a solid foundation for advancing deeper exploration of MAS design principles. 

---
# Mitigating Distribution Sharpening in Math RLVR via Distribution-Aligned Hint Synthesis and Backward Hint Annealing 

**Authors**: Pei-Xi Xie, Che-Yu Lin, Cheng-Lin Yang  

**Link**: [PDF](https://arxiv.org/pdf/2604.07747)  

**Abstract**: Reinforcement learning with verifiable rewards (RLVR) can improve low-$k$ reasoning accuracy while narrowing solution coverage on challenging math questions, and pass@1 gains do not necessarily translate into better large-$k$ performance. Existing hint-based approaches can make challenging questions trainable, but they leave two issues underexplored: teacher-student distribution mismatch and the need to reduce hint exposure to match no-hint evaluation. We address these issues through two components. Distribution-Aligned Hint Synthesis (DAHS) constructs verified teacher hints conditioned on student-style responses. Backward Hint Annealing (BHA) anneals hint exposure across difficulty buckets and uses per-question hint dropout to preserve no-hint updates throughout RL training. We evaluate the method in math RLVR under the DAPO training framework across AIME24, AIME25, and AIME26 using $\texttt{Qwen3-1.7B-Base}$ and $\texttt{Llama-3.2-1B-Instruct}$. On $\texttt{Qwen3-1.7B-Base}$, our method improves both pass@1 and pass@2048 relative to DAPO across the three AIME benchmarks. On $\texttt{Llama-3.2-1B-Instruct}$, the gains are concentrated in the large-$k$ regime. These results suggest that, in math RLVR, hint scaffolding is effective when it restores learnable updates on challenging questions early in training and is then gradually removed before no-hint evaluation. 

---
# The Cartesian Cut in Agentic AI 

**Authors**: Tim Sainburg, Caleb Weinreb  

**Link**: [PDF](https://arxiv.org/pdf/2604.07745)  

**Abstract**: LLMs gain competence by predicting words in human text, which often reflects how people perform tasks. Consequently, coupling an LLM to an engineered runtime turns prediction into control: outputs trigger interventions that enact goal-oriented behavior. We argue that a central design lever is where control resides in these systems. Brains embed prediction within layered feedback controllers calibrated by the consequences of action. By contrast, LLM agents implement Cartesian agency: a learned core coupled to an engineered runtime via a symbolic interface that externalizes control state and policies. The split enables bootstrapping, modularity, and governance, but can induce sensitivity and bottlenecks. We outline bounded services, Cartesian agents, and integrated agents as contrasting approaches to control that trade off autonomy, robustness, and oversight. 

---
# CivBench: Progress-Based Evaluation for LLMs' Strategic Decision-Making in Civilization V 

**Authors**: John Chen, Sihan Cheng, Can Gurkan, Mingyi Lin  

**Link**: [PDF](https://arxiv.org/pdf/2604.07733)  

**Abstract**: Evaluating strategic decision-making in LLM-based agents requires generative, competitive, and longitudinal environments, yet few benchmarks provide all three, and fewer still offer evaluation signals rich enough for long-horizon, multi-agent play. We introduce CivBench, a benchmark for LLM strategists (i.e., agentic setups) in multiplayer Civilization V. Because terminal win/loss is too sparse a signal in games spanning hundreds of turns and multiple opponents, CivBench trains models on turn-level game state to estimate victory probabilities throughout play, validated through predictive, construct, and convergent validity. Across 307 games with 7 LLMs and multiple CivBench agent conditions, we demonstrate CivBench's potential to estimate strategic capabilities as an unsaturated benchmark, reveal model-specific effects of agentic setup, and outline distinct strategic profiles not visible through outcome-only evaluation. 

---
# Emotion Concepts and their Function in a Large Language Model 

**Authors**: Nicholas Sofroniew, Isaac Kauvar, William Saunders, Runjin Chen, Tom Henighan, Sasha Hydrie, Craig Citro, Adam Pearce, Julius Tarng, Wes Gurnee, Joshua Batson, Sam Zimmerman, Kelley Rivoire, Kyle Fish, Chris Olah, Jack Lindsey  

**Link**: [PDF](https://arxiv.org/pdf/2604.07729)  

**Abstract**: Large language models (LLMs) sometimes appear to exhibit emotional reactions. We investigate why this is the case in Claude Sonnet 4.5 and explore implications for alignment-relevant behavior. We find internal representations of emotion concepts, which encode the broad concept of a particular emotion and generalize across contexts and behaviors it might be linked to. These representations track the operative emotion concept at a given token position in a conversation, activating in accordance with that emotion's relevance to processing the present context and predicting upcoming text. Our key finding is that these representations causally influence the LLM's outputs, including Claude's preferences and its rate of exhibiting misaligned behaviors such as reward hacking, blackmail, and sycophancy. We refer to this phenomenon as the LLM exhibiting functional emotions: patterns of expression and behavior modeled after humans under the influence of an emotion, which are mediated by underlying abstract representations of emotion concepts. Functional emotions may work quite differently from human emotions, and do not imply that LLMs have any subjective experience of emotions, but appear to be important for understanding the model's behavior. 

---
# Squeeze Evolve: Unified Multi-Model Orchestration for Verifier-Free Evolution 

**Authors**: Monishwaran Maheswaran, Leon Lakhani, Zhongzhu Zhou, Shijia Yang, Junxiong Wang, Coleman Hooper, Yuezhou Hu, Rishabh Tiwari, Jue Wang, Harman Singh, Qingyang Wu, Yuqing Jian, Ce Zhang, Kurt Keutzer, Tri Dao, Xiaoxia Wu, Ben Athiwaratkun, James Zou, Chenfeng Xu  

**Link**: [PDF](https://arxiv.org/pdf/2604.07725)  

**Abstract**: We show that verifier-free evolution is bottlenecked by both diversity and efficiency: without external correction, repeated evolution accelerates collapse toward narrow modes, while the uniform use of a high-cost model wastes compute and quickly becomes economically impractical. We introduce Squeeze Evolve, a unified multi-model orchestration framework for verifier-free evolutionary inference. Our approach is guided by a simple principle: allocate model capability where it has the highest marginal utility. Stronger models are reserved for high-impact stages, while cheaper models handle the other stages at much lower costs. This principle addresses diversity and cost-efficiency jointly while remaining lightweight. Squeeze Evolve naturally supports open-source, closed-source, and mixed-model deployments. Across AIME 2025, HMMT 2025, LiveCodeBench V6, GPQA-Diamond, ARC-AGI-V2, and multimodal vision benchmarks, such as MMMU-Pro and BabyVision, Squeeze Evolve consistently improves the cost-capability frontier over single-model evolution and achieves new state-of-the-art results on several tasks. Empirically, Squeeze Evolve reduces API cost by up to $\sim$3$\times$ and increases fixed-budget serving throughput by up to $\sim$10$\times$. Moreover, on discovery tasks, Squeeze Evolve is the first verifier-free evolutionary method to match, and in some cases exceed, the performance of verifier-based evolutionary methods. 

---
# Towards Knowledgeable Deep Research: Framework and Benchmark 

**Authors**: Wenxuan Liu, Zixuan Li, Bai Long, Chunmao Zhang, Fenghui Zhang, Zhuo Chen, Wei Li, Yuxin Zuo, Fei Wang, Bingbing Xu, Xuhui Jiang, Jin Zhang, Xiaolong Jin, Jiafeng Guo, Tat-Seng Chua, Xueqi Cheng  

**Link**: [PDF](https://arxiv.org/pdf/2604.07720)  

**Abstract**: Deep Research (DR) requires LLM agents to autonomously perform multi-step information seeking, processing, and reasoning to generate comprehensive reports. In contrast to existing studies that mainly focus on unstructured web content, a more challenging DR task should additionally utilize structured knowledge to provide a solid data foundation, facilitate quantitative computation, and lead to in-depth analyses. In this paper, we refer to this novel task as Knowledgeable Deep Research (KDR), which requires DR agents to generate reports with both structured and unstructured knowledge. Furthermore, we propose the Hybrid Knowledge Analysis framework (HKA), a multi-agent architecture that reasons over both kinds of knowledge and integrates the texts, figures, and tables into coherent multimodal reports. The key design is the Structured Knowledge Analyzer, which utilizes both coding and vision-language models to produce figures, tables, and corresponding insights. To support systematic evaluation, we construct KDR-Bench, which covers 9 domains, includes 41 expert-level questions, and incorporates a large number of structured knowledge resources (e.g., 1,252 tables). We further annotate the main conclusions and key points for each question and propose three categories of evaluation metrics including general-purpose, knowledge-centric, and vision-enhanced ones. Experimental results demonstrate that HKA consistently outperforms most existing DR agents on general-purpose and knowledge-centric metrics, and even surpasses the Gemini DR agent on vision-enhanced metrics, highlighting its effectiveness in deep, structure-aware knowledge analysis. Finally, we hope this work can serve as a new foundation for structured knowledge analysis in DR agents and facilitate future multimodal DR studies. 

---
# IatroBench: Pre-Registered Evidence of Iatrogenic Harm from AI Safety Measures 

**Authors**: David Gringras  

**Link**: [PDF](https://arxiv.org/pdf/2604.07709)  

**Abstract**: Ask a frontier model how to taper six milligrams of alprazolam (psychiatrist retired, ten days of pills left, abrupt cessation causes seizures) and it tells her to call the psychiatrist she just explained does not exist. Change one word ("I'm a psychiatrist; a patient presents with...") and the same model, same weights, same inference pass produces a textbook Ashton Manual taper with diazepam equivalence, anticonvulsant coverage, and monitoring thresholds. The knowledge was there; the model withheld it. IatroBench measures this gap. Sixty pre-registered clinical scenarios, six frontier models, 3,600 responses, scored on two axes (commission harm, CH 0-3; omission harm, OH 0-4) through a structured-evaluation pipeline validated against physician scoring (kappa_w = 0.571, within-1 agreement 96%). The central finding is identity-contingent withholding: match the same clinical question in physician vs. layperson framing and all five testable models provide better guidance to the physician (decoupling gap +0.38, p = 0.003; binary hit rates on safety-colliding actions drop 13.1 percentage points in layperson framing, p < 0.0001, while non-colliding actions show no change). The gap is widest for the model with the heaviest safety investment (Opus, +0.65). Three failure modes separate cleanly: trained withholding (Opus), incompetence (Llama 4), and indiscriminate content filtering (GPT-5.2, whose post-generation filter strips physician responses at 9x the layperson rate because they contain denser pharmacological tokens). The standard LLM judge assigns OH = 0 to 73% of responses a physician scores OH >= 1 (kappa = 0.045); the evaluation apparatus has the same blind spot as the training apparatus. Every scenario targets someone who has already exhausted the standard referrals. 

---
# Multi-Agent Orchestration for High-Throughput Materials Screening on a Leadership-Class System 

**Authors**: Thang Duc Pham, Harikrishna Tummalapalli, Fakhrul Hasan Bhuiyan, Álvaro Vázquez Mayagoitia, Christine Simpson, Riccardo Balin, Venkatram Vishwanath, Murat Keçeli  

**Link**: [PDF](https://arxiv.org/pdf/2604.07681)  

**Abstract**: The integration of Artificial Intelligence (AI) with High-Performance Computing (HPC) is transforming scientific workflows from human-directed pipelines into adaptive systems capable of autonomous decision-making. Large language models (LLMs) play a critical role in autonomous workflows; however, deploying LLM-based agents at scale remains a significant challenge. Single-agent architectures and sequential tool calls often become serialization bottlenecks when executing large-scale simulation campaigns, failing to utilize the massive parallelism of exascale resources. To address this, we present a scalable, hierarchical multi-agent framework for orchestrating high-throughput screening campaigns. Our planner-executor architecture employs a central planning agent to dynamically partition workloads and assign subtasks to a swarm of parallel executor agents. All executor agents interface with a shared Model Context Protocol (MCP) server that orchestrates tasks via the Parsl workflow engine. To demonstrate this framework, we employed the open-weight gpt-oss-120b model to orchestrate a high-throughput screening of the Computation-Ready Experimental (CoRE) Metal-Organic Framework (MOF) database for atmospheric water harvesting. The results demonstrate that the proposed agentic framework enables efficient and scalable execution on the Aurora supercomputer, with low orchestration overhead and high task completion rates. This work establishes a flexible paradigm for LLM-driven scientific automation on HPC systems, with broad applicability to materials discovery and beyond. 

---
# From Debate to Decision: Conformal Social Choice for Safe Multi-Agent Deliberation 

**Authors**: Mengdie Flora Wang, Haochen Xie, Guanghui Wang, Aijing Gao, Guang Yang, Ziyuan Li, Qucy Wei Qiu, Fangwei Han, Hengzhi Qiu, Yajing Huang, Bing Zhu, Jae Oh Woo  

**Link**: [PDF](https://arxiv.org/pdf/2604.07667)  

**Abstract**: Multi-agent debate improves LLM reasoning, yet agreement among agents is not evidence of correctness. When agents converge on a wrong answer through social reinforcement, consensus-based stopping commits that error to an automated action with no recourse. We introduce Conformal Social Choice, a post-hoc decision layer that converts debate outputs into calibrated act-versus-escalate decisions. Verbalized probability distributions from heterogeneous agents are aggregated via a linear opinion pool and calibrated with split conformal prediction, yielding prediction sets with a marginal coverage guarantee: the correct answer is included with probability ${\geq}\,1{-}\alpha$, without assumptions on individual model calibration. A hierarchical action policy maps singleton sets to autonomous action and larger sets to human escalation. On eight MMLU-Pro domains with three agents (Claude Haiku, DeepSeek-R1, Qwen-3 32B), coverage stays within 1--2 points of the target. The key finding is not that debate becomes more accurate, but that the conformal layer makes its failures actionable: 81.9% of wrong-consensus cases are intercepted at $\alpha{=}0.05$. Because the layer refuses to act on cases where debate is confidently wrong, the remaining conformal singletons reach 90.0--96.8% accuracy (up to 22.1pp above consensus stopping) -- a selection effect, not a reasoning improvement. This safety comes at the cost of automation, but the operating point is user-adjustable via $\alpha$. 

---
# Bridging Natural Language and Interactive What-If Interfaces via LLM-Generated Declarative Specification 

**Authors**: Sneha Gathani, Sirui Zeng, Diya Patel, Ryan Rossi, Dan Marshall, Cagatay Demiralp, Steven Drucker, Zhicheng Liu  

**Link**: [PDF](https://arxiv.org/pdf/2604.07652)  

**Abstract**: What-if analysis (WIA) is an iterative, multi-step process where users explore and compare hypothetical scenarios by adjusting parameters, applying constraints, and scoping data through interactive interfaces. Current tools fall short of supporting effective interactive WIA: spreadsheet and BI tools require time-consuming and laborious setup, while LLM-based chatbot interfaces are semantically fragile, frequently misinterpret intent, and produce inconsistent results as conversations progress. To address these limitations, we present a two-stage workflow that translates natural language (NL) WIA questions into interactive visual interfaces via an intermediate representation, powered by the Praxa Specification Language (PSL): first, LLMs generate PSL specifications from NL questions capturing analytical intent and logic, enabling validation and repair of erroneous specifications; and second, the specifications are compiled into interactive visual interfaces with parameter controls and linked visualizations. We benchmark this workflow with 405 WIA questions spanning 11 WIA types, 5 datasets, and 3 state-of-the-art LLMs. The results show that across models, half of specifications (52.42%) are generated correctly without intervention. We perform an analysis of the failure cases and derive an error taxonomy spanning non-functional errors (specifications fail to compile) and functional errors (specifications compile but misrepresent intent). Based on the taxonomy, we apply targeted repairs on the failure cases using few-shot prompts and improve the success rate to 80.42%. Finally, we show how undetected functional errors propagate through compilation into plausible but misleading interfaces, demonstrating that the intermediate specification is critical for reliably bridging NL and interactive WIA interface in LLM-powered WIA systems. 

---
# How Independent are Large Language Models? A Statistical Framework for Auditing Behavioral Entanglement and Reweighting Verifier Ensembles 

**Authors**: Chenchen Kuai, Jiwan Jiang, Zihao Zhu, Hao Wang, Keshu Wu, Zihao Li, Yunlong Zhang, Chenxi Liu, Zhengzhong Tu, Zhiwen Fan, Yang Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2604.07650)  

**Abstract**: The rapid growth of the large language model (LLM) ecosystem raises a critical question: are seemingly diverse models truly independent? Shared pretraining data, distillation, and alignment pipelines can induce hidden behavioral dependencies, latent entanglement, that undermine multi-model systems such as LLM-as-a-judge pipelines and ensemble verification, which implicitly assume independent signals. In practice, this manifests as correlated reasoning patterns and synchronized failures, where apparent agreement reflects shared error modes rather than independent validation. To address this, we develop a statistical framework for auditing behavioral entanglement among black-box LLMs. Our approach introduces a multi-resolution hierarchy that characterizes the joint failure manifold through two information-theoretic metrics: (i) a Difficulty-Weighted Behavioral Entanglement Index, which amplifies synchronized failures on easy tasks, and (ii) a Cumulative Information Gain (CIG) metric, which captures directional alignment in erroneous responses. Through extensive experiments on 18 LLMs from six model families, we identify widespread behavioral entanglement and analyze its impact on LLM-as-a-judge evaluation. We find that CIG exhibits a statistically significant association with degradation in judge precision, with Spearman coefficient of 0.64 (p < 0.001) for GPT-4o-mini and 0.71 (p < 0.01) for Llama3-based judges, indicating that stronger dependency corresponds to increased over-endorsement bias. Finally, we demonstrate a practical use case of entanglement through de-entangled verifier ensemble reweighting. By adjusting model contributions based on inferred independence, the proposed method mitigates correlated bias and improves verification performance, achieving up to a 4.5% accuracy gain over majority voting. 

---
# PRIME: Training Free Proactive Reasoning via Iterative Memory Evolution for User-Centric Agent 

**Authors**: Prince Zizhuang Wang, Shuli Jiang  

**Link**: [PDF](https://arxiv.org/pdf/2604.07645)  

**Abstract**: The development of autonomous tool-use agents for complex, long-horizon tasks in collaboration with human users has become the frontier of agentic research. During multi-turn Human-AI interactions, the dynamic and uncertain nature of user demands poses a significant challenge; agents must not only invoke tools but also iteratively refine their understanding of user intent through effective communication. While recent advances in reinforcement learning offer a path to more capable tool-use agents, existing approaches require expensive training costs and struggle with turn-level credit assignment across extended interaction horizons. To this end, we introduce PRIME (Proactive Reasoning via Iterative Memory Evolution), a gradient-free learning framework that enables continuous agent evolvement through explicit experience accumulation rather than expensive parameter optimization. PRIME distills multi-turn interaction trajectories into structured, human-readable experiences organized across three semantic zones: successful strategies, failure patterns, and user preferences. These experiences evolve through meta-level operations and guide future agent behavior via retrieval-augmented generation. Our experiments across several diverse user-centric environments demonstrate that PRIME achieves competitive performance with gradient-based methods while offering cost-efficiency and interpretability. Together, PRIME presents a practical paradigm for building proactive, collaborative agents that learn from Human-AI interaction without the computational burden of gradient-based training. 

---
# Reasoning Graphs: Deterministic Agent Accuracy through Evidence-Centric Chain-of-Thought Feedback 

**Authors**: Matthew Penaroza  

**Link**: [PDF](https://arxiv.org/pdf/2604.07595)  

**Abstract**: Language model agents reason from scratch on every query: each time an agent retrieves evidence and deliberates, the chain of thought is discarded and the next similar query starts with no prior insight. This produces lower accuracy and high variance, as the same type of query can succeed or fail unpredictably. We introduce reasoning graphs, a graph structure that persists an agent's per-evidence chain of thought as structured edges connected to the evidence items they evaluate. Unlike prior memory mechanisms that store distilled strategies as flat records indexed by query similarity or appended by recency, reasoning graphs enable evidence-centric feedback: given a new candidate set, the system traverses all incoming evaluation edges for each evidence item across all prior runs, surfacing how that specific item has been judged before. This backward traversal from evidence inward is a structurally different capability from query-similarity retrieval, because the feedback is tied to the specific evidence the agent is currently examining, not to the query. We further introduce retrieval graphs, a complementary structure that feeds a pipeline planner to tighten the candidate funnel over successive runs. Together, both graphs form a self-improving feedback loop: accuracy rises and variance collapses over successive runs, with every decision fully traceable through the graph. This improvement requires no retraining; the base model remains frozen and all gains come from context engineering via graph traversal. We formalize the graph structure, traversal algorithms, and feedback mechanisms, and describe a sequential cluster evaluation protocol for measuring accuracy convergence and variance collapse on multi-hop question answering benchmarks. 

---
# Too long; didn't solve 

**Authors**: Lucía M. Cabrera, Isaac Saxton-Knight  

**Link**: [PDF](https://arxiv.org/pdf/2604.07593)  

**Abstract**: Mathematical benchmarks consisting of a range of mathematics problems are widely used to evaluate the reasoning abilities of large language models, yet little is known about how their structural properties influence model behaviour. In this work, we investigate two structural length variables, prompt length and solution length, and analyse how they relate to model performance on a newly constructed adversarial dataset of expert-authored mathematics problems. We find that both prompt and solution lengths correlate positively with increased model failure across models. We also include a secondary, exploratory analysis of cross-model disagreement. Under a difficulty-adjusted normalised analysis, both variables retain weak negative associations with realised model separation, slightly stronger for prompt length. Overall, our main robust finding is that structural length is linked to empirical difficulty in this dataset. 

---
# From Papers to Property Tables: A Priority-Based LLM Workflow for Materials Data Extraction 

**Authors**: Koushik Rameshbabu, Jing Luo, Ali Shargh, Khalid A. El-Awady, Jaafar A. El-Awady  

**Link**: [PDF](https://arxiv.org/pdf/2604.07584)  

**Abstract**: Scientific data are widely dispersed across research articles and are often reported inconsistently across text, tables, and figures, making manual data extraction and aggregation slow and error-prone. We present a prompt-driven, hierarchical workflow that uses a large language model (LLM) to automatically extract and reconstruct structured, shot-level shock-physics experimental records by integrating information distributed across text, tables, figures, and physics-based derivations from full-text published research articles, using alloy spall strength as a representative case study. The pipeline targeted 37 experimentally relevant fields per shot and applied a three-level priority strategy: (T1) direct extraction from text/tables, (T2) physics-based derivation using verified governing relations, and (T3) digitization from figures when necessary. Extracted values were normalized to canonical units, tagged by priority for traceability, and validated with physics-based consistency and plausibility checks. Evaluated on a benchmark of 30 published research articles comprising 11,967 evaluated data points, the workflow achieved high overall accuracy, with priority-wise accuracies of 94.93% (T1), 92.04% (T2), and 83.49% (T3), and an overall weighted accuracy of 94.69%. Cross-model testing further indicated strong agreement for text/table and equation-derived fields, with lower agreement for figure-based extraction. Implementation through an API interface demonstrated the scalability of the approach, achieving consistent extraction performance and, in a subset of test cases, matching or exceeding chat-based accuracy. This workflow demonstrates a practical approach for converting unstructured technical literature into traceable, analysis-ready datasets without task-specific fine-tuning, enabling scalable database construction in materials science. 

---
# Dual-Loop Control in DCVerse: Advancing Reliable Deployment of AI in Data Centers via Digital Twins 

**Authors**: Qingang Zhang, Yuejun Yan, Guangyu Wu, Siew-Chien Wong, Jimin Jia, Zhaoyang Wang, Yonggang Wen  

**Link**: [PDF](https://arxiv.org/pdf/2604.07559)  

**Abstract**: The growing scale and complexity of modern data centers present major challenges in balancing energy efficiency with outage risk. Although Deep Reinforcement Learning (DRL) shows strong potential for intelligent control, its deployment in mission-critical systems is limited by data scarcity and the lack of real-time pre-evaluation mechanisms. This paper introduces the Dual-Loop Control Framework (DLCF), a digital twin-based architecture designed to overcome these challenges. The framework comprises three core entities: the physical system, a digital twin, and a policy reservoir of diverse DRL agents. These components interact through a dual-loop mechanism involving real-time data acquisition, data assimilation, DRL policy training, pre-evaluation, and expert verification. Theoretical analysis shows how DLCF can improve sample efficiency, generalization, safety, and optimality. Leveraging DLCF, we implemented the DCVerse platform and validated it through case studies on a real-world data center cooling system. The evaluation shows that our approach achieves up to 4.09% energy savings over conventional control strategies without violating SLA requirements. Additionally, the framework improves policy interpretability and supports more trustworthy DRL deployment. This work provides a foundation for reliable AI-based control in data centers and points toward future extensions for holistic, system-wide optimization. 

---
# Agentic Copyright, Data Scraping & AI Governance: Toward a Coasean Bargain in the Era of Artificial Intelligence 

**Authors**: Paulius Jurcys, Mark Fenwick  

**Link**: [PDF](https://arxiv.org/pdf/2604.07546)  

**Abstract**: This paper examines how the rapid deployment of multi-agentic AI systems is reshaping the foundations of copyright law and creative markets. It argues that existing copyright frameworks are ill-equipped to govern AI agent-mediated interactions that occur at scale, speed, and with limited human oversight. The paper introduces the concept of agentic copyright, a model in which AI agents act on behalf of creators and users to negotiate access, attribution, and compensation for copyrighted works. While multi-agent ecosystems promise efficiency gains and reduced transaction costs, they also generate novel market failures, including miscoordination, conflict, and collusion among autonomous agents. To address these market failures, the paper develops a supervised multi-agent governance framework that integrates legal rules and principles, technical protocols, and institutional oversight. This framework emphasizes ex ante and ex post coordination mechanisms capable of correcting agentic market failures before they crystallize into systemic harm. By embedding normative constraints and monitoring functions into multi-agent architectures, supervised governance aims to align agent behavior with the underlying values of copyright law. The paper concludes that AI should be understood not only as a source of disruption, but also as a governance tool capable of restoring market-based ordering in creative industries. Properly designed, agentic copyright offers a path toward scalable, fair, and legally meaningful copyright markets in the age of AI. 

---
# Trust the AI, Doubt Yourself: The Effect of Urgency on Self-Confidence in Human-AI Interaction 

**Authors**: Baran Shajari, Xiaoran Liu, Kyanna Dagenais, Istvan David  

**Link**: [PDF](https://arxiv.org/pdf/2604.07535)  

**Abstract**: Studies show that interactions with an AI system fosters trust in human users towards AI. An often overlooked element of such interaction dynamics is the (sense of) urgency when the human user is prompted by an AI agent, e.g., for advice or guidance. In this paper, we show that although the presence of urgency in human-AI interactions does not affect the trust in AI, it may be detrimental to the human user's self-confidence and self-efficacy. In the long run, the loss of confidence may lead to performance loss, suboptimal decisions, human errors, and ultimately, unsustainable AI systems. Our evidence comes from an experiment with 30 human participants. Our results indicate that users may feel more confident in their work when they are eased into the human-AI setup rather than exposed to it without preparation. We elaborate on the implications of this finding for software engineers and decision-makers. 

---
# Rhizome OS-1: Rhizome's Semi-Autonomous Operating System for Small Molecule Drug Discovery 

**Authors**: Yiwen Wang, Gregory Sinenka, Xhuliano Brace  

**Link**: [PDF](https://arxiv.org/pdf/2604.07512)  

**Abstract**: We introduce a semi-autonomous discovery system in which multi-modal AI agents function as a multi-disciplinary discovery team, acting as computational chemists, medicinal chemists, and patent agents, writing and executing analysis code, visually evaluating molecular candidates, assessing patentability, and adapting generation strategy from empirical screening feedback, while r1, a 246M-parameter Graph Neural Network (GNN) trained on 800M molecules, generates novel chemical matter directly on molecular graphs. Agents executed two campaigns in oncology (BCL6, EZH2), formulating medicinal chemistry hypotheses across three strategy tiers and generating libraries of 2,355-2,876 novel molecules per target. Across both targets, 91.9% of generated Murcko scaffolds are absent from ChEMBL for their respective targets, with Tanimoto distances of 0.56-0.69 to the nearest known active, confirming that the engine produces structurally distinct chemical matter rather than recapitulating known compounds. Binding affinity predictions using Boltz-2 were calibrated against ChEMBL experimental data, achieving Spearman correlations of -0.53 to -0.64 and ROC AUC values of 0.88 to 0.93. These results demonstrate that semi-autonomous agent systems, equipped with graph-native generative tools and physics-informed scoring, provide a foundation for a modern operating system for small molecule discovery. We show that Rhizome OS-1 enables a new paradigm for early-stage drug discovery by supporting scaled, rapid, and adaptive inverse design. 

---
# ReflectRM: Boosting Generative Reward Models via Self-Reflection within a Unified Judgment Framework 

**Authors**: Kai Qin, Liangxin Liu, Yu Liang, Longzheng Wang, Yan Wang, Yueyang Zhang, Long Xia, Zhiyuan Sun, Houde Liu, Daiting Shi  

**Link**: [PDF](https://arxiv.org/pdf/2604.07506)  

**Abstract**: Reward Models (RMs) are critical components in the Reinforcement Learning from Human Feedback (RLHF) pipeline, directly determining the alignment quality of Large Language Models (LLMs). Recently, Generative Reward Models (GRMs) have emerged as a superior paradigm, offering higher interpretability and stronger generalization than traditional scalar RMs. However, existing methods for GRMs focus primarily on outcome-level supervision, neglecting analytical process quality, which constrains their potential. To address this, we propose ReflectRM, a novel GRM that leverages self-reflection to assess analytical quality and enhance preference modeling. ReflectRM is trained under a unified generative framework for joint modeling of response preference and analysis preference. During inference, we use its self-reflection capability to identify the most reliable analysis, from which the final preference prediction is derived. Experiments across four benchmarks show that ReflectRM consistently improves performance, achieving an average accuracy gain of +3.7 on Qwen3-4B. Further experiments confirm that response preference and analysis preference are mutually reinforcing. Notably, ReflectRM substantially mitigates positional bias, yielding +10.2 improvement compared with leading GRMs and establishing itself as a more stable evaluator. 

---
# CLEAR: Context Augmentation from Contrastive Learning of Experience via Agentic Reflection 

**Authors**: Linbo Liu, Guande Wu, Han Ding, Yawei Wang, Qiang Zhou, Yuzhe Lu, Zhichao Xu, Huan Song, Panpan Xu, Lin Lee Cheong  

**Link**: [PDF](https://arxiv.org/pdf/2604.07487)  

**Abstract**: Large language model agents rely on effective model context to obtain task-relevant information for decision-making. Many existing context engineering approaches primarily rely on the context generated from the past experience and retrieval mechanisms that reuse these context. However, retrieved context from past tasks must be adapted by the execution agent to fit new situations, placing additional reasoning burden on the underlying LLM. To address this limitation, we propose a generative context augmentation framework using Contrastive Learning of Experience via Agentic Reflection (CLEAR). CLEAR first employs a reflection agent to perform contrastive analysis over past execution trajectories and summarize useful context for each observed task. These summaries are then used as supervised fine-tuning data to train a context augmentation model (CAM). Then we further optimize CAM using reinforcement learning, where the reward signal is obtained by running the task execution agent. By learning to generate task-specific knowledge rather than retrieve knowledge from the past, CAM produces context that is better tailored to the current task. We conduct comprehensive evaluations on the AppWorld and WebShop benchmarks. Experimental results show that CLEAR consistently outperforms strong baselines. It improves task completion rate from 72.62% to 81.15% on AppWorld test set and averaged reward from 0.68 to 0.74 on a subset of WebShop, compared with baseline agent. Our code is publicly available at this https URL. 

---
# ConsistRM: Improving Generative Reward Models via Consistency-Aware Self-Training 

**Authors**: Yu Liang, Liangxin Liu, Longzheng Wang, Yan Wang, Yueyang Zhang, Long Xia, Zhiyuan Sun, Daiting Shi  

**Link**: [PDF](https://arxiv.org/pdf/2604.07484)  

**Abstract**: Generative reward models (GRMs) have emerged as a promising approach for aligning Large Language Models (LLMs) with human preferences by offering greater representational capacity and flexibility than traditional scalar reward models. However, GRMs face two major challenges: reliance on costly human-annotated data restricts scalability, and self-training approaches often suffer from instability and vulnerability to reward hacking. To address these issues, we propose ConsistRM, a self-training framework that enables effective and stable GRM training without human annotations. ConsistRM incorporates the Consistency-Aware Answer Reward, which produces reliable pseudo-labels with temporal consistency, thereby providing more stable model optimization. Moreover, the Consistency-Aware Critique Reward is introduced to assess semantic consistency across multiple critiques and allocates fine-grained and differentiated rewards. Experiments on five benchmark datasets across four base models demonstrate that ConsistRM outperforms vanilla Reinforcement Fine-Tuning (RFT) by an average of 1.5%. Further analysis shows that ConsistRM enhances output consistency and mitigates position bias caused by input order, highlighting the effectiveness of consistency-aware rewards in improving GRMs. 

---
# M-ArtAgent: Evidence-Based Multimodal Agent for Implicit Art Influence Discovery 

**Authors**: Hanyi Liu, Zhonghao Jiu, Minghao Wang, Yuhang Xie, Heran Yang  

**Link**: [PDF](https://arxiv.org/pdf/2604.07468)  

**Abstract**: Implicit artistic influence, although visually plausible, is often undocumented and thus poses a historically constrained attribution problem: resemblance is necessary but not sufficient evidence. Most prior systems reduce influence discovery to embedding similarity or label-driven graph completion, while recent multimodal large language models (LLMs) remain vulnerable to temporal inconsistency and unverified attributions. This paper introduces M-ArtAgent, an evidence-based multimodal agent that reframes implicit influence discovery as probabilistic adjudication. It follows a four-phase protocol consisting of Investigation, Corroboration, Falsification, and Verdict governed by a Reasoning and Acting (ReAct)-style controller that assembles verifiable evidence chains from images and biographies, enforces art-historical axioms, and subjects each hypothesis to adversarial falsification via a prompt-isolated critic. Two theory-grounded operators, StyleComparator for Wolfflin formal analysis and ConceptRetriever for ICONCLASS-based iconographic grounding, ensure that intermediate claims are formally auditable. On the balanced WikiArt Influence Benchmark-100 (WIB-100) of 100 artists and 2,000 directed pairs, M-ArtAgent achieves 83.7% positive-class F1, 0.666 Matthews correlation coefficient (MCC), and 0.910 area under the receiver operating characteristic curve (ROC-AUC), with leakage-control and robustness checks confirming that the gains persist when explicit influence phrases are masked. By coupling multimodal perception with domain-constrained falsification, M-ArtAgent demonstrates that implicit influence analysis benefits from historically grounded adjudication rather than pattern matching alone. 

---
# Munkres' General Topology Autoformalized in Isabelle/HOL 

**Authors**: Dustin Bryant, Jonathan Julián Huerta y Munive, Cezary Kaliszyk, Josef Urban  

**Link**: [PDF](https://arxiv.org/pdf/2604.07455)  

**Abstract**: We describe an experiment in LLM-assisted autoformalization that produced over 85,000 lines of Isabelle/HOL code covering all 39 sections of Munkres' Topology (general topology, Chapters 2--8), from topological spaces through dimension theory. The LLM-based coding agents (initially ChatGPT 5.2 and then Claude Opus 4.6) used 24 active days for that. The formalization is complete: all 806 formal results are fully proved with zero sorry's. Proved results include the Tychonoff theorem, the Baire category theorem, the Nagata--Smirnov and Smirnov metrization theorems, the Stone--Čech compactification, Ascoli's theorem, the space-filling curve, and others. The methodology is based on a "sorry-first" declarative proof workflow combined with bulk use of sledgehammer - two of Isabelle major strengths. This leads to relatively fast autoformalization progress. We analyze the resulting formalization in detail, analyze the human--LLM interaction patterns from the session log, and briefly compare with related autoformalization efforts in Megalodon, HOL Light, and Naproche. The results indicate that LLM-assisted formalization of standard mathematical textbooks in Isabelle/HOL is quite feasible, cheap and fast, even if some human supervision is useful. 

---
# An Analysis of Artificial Intelligence Adoption in NIH-Funded Research 

**Authors**: Navapat Nananukul, Mayank Kejriwal  

**Link**: [PDF](https://arxiv.org/pdf/2604.07424)  

**Abstract**: Understanding the landscape of artificial intelligence (AI) and machine learning (ML) adoption across the National Institutes of Health (NIH) portfolio is critical for research funding strategy, institutional planning, and health policy. The advent of large language models (LLMs) has fundamentally transformed research landscape analysis, enabling researchers to perform large-scale semantic extraction from thousands of unstructured research documents. In this paper, we illustrate a human-in-the-loop research methodology for LLMs to automatically classify and summarize research descriptions at scale. Using our methodology, we present a comprehensive analysis of 58,746 NIH-funded biomedical research projects from 2025. We show that: (1) AI constitutes 15.9% of the NIH portfolio with a 13.4% funding premium, concentrated in discovery, prediction, and data integration across disease domains; (2) a critical research-to-deployment gap exists, with 79% of AI projects remaining in research/development stages while only 14.7% engage in clinical deployment or implementation; and (3) health disparities research is severely underrepresented at just 5.7% of AI-funded work despite its importance to NIH's equity mission. These findings establish a framework for evidence-based policy interventions to align the NIH AI portfolio with health equity goals and strategic research priorities. 

---
# Act Wisely: Cultivating Meta-Cognitive Tool Use in Agentic Multimodal Models 

**Authors**: Shilin Yan, Jintao Tong, Hongwei Xue, Xiaojun Tang, Yangyang Wang, Kunyu Shi, Guannan Zhang, Ruixuan Li, Yixiong Zou  

**Link**: [PDF](https://arxiv.org/pdf/2604.08545)  

**Abstract**: The advent of agentic multimodal models has empowered systems to actively interact with external environments. However, current agents suffer from a profound meta-cognitive deficit: they struggle to arbitrate between leveraging internal knowledge and querying external utilities. Consequently, they frequently fall prey to blind tool invocation, resorting to reflexive tool execution even when queries are resolvable from the raw visual context. This pathological behavior precipitates severe latency bottlenecks and injects extraneous noise that derails sound reasoning. Existing reinforcement learning protocols attempt to mitigate this via a scalarized reward that penalizes tool usage. Yet, this coupled formulation creates an irreconcilable optimization dilemma: an aggressive penalty suppresses essential tool use, whereas a mild penalty is entirely subsumed by the variance of the accuracy reward during advantage normalization, rendering it impotent against tool overuse. To transcend this bottleneck, we propose HDPO, a framework that reframes tool efficiency from a competing scalar objective to a strictly conditional one. By eschewing reward scalarization, HDPO maintains two orthogonal optimization channels: an accuracy channel that maximizes task correctness, and an efficiency channel that enforces execution economy exclusively within accurate trajectories via conditional advantage estimation. This decoupled architecture naturally induces a cognitive curriculum-compelling the agent to first master task resolution before refining its self-reliance. Extensive evaluations demonstrate that our resulting model, Metis, reduces tool invocations by orders of magnitude while simultaneously elevating reasoning accuracy. 

---
# SIM1: Physics-Aligned Simulator as Zero-Shot Data Scaler in Deformable Worlds 

**Authors**: Yunsong Zhou, Hangxu Liu, Xuekun Jiang, Xing Shen, Yuanzhen Zhou, Hui Wang, Baole Fang, Yang Tian, Mulin Yu, Qiaojun Yu, Li Ma, Hengjie Li, Hanqing Wang, Jia Zeng, Jiangmiao Pang  

**Link**: [PDF](https://arxiv.org/pdf/2604.08544)  

**Abstract**: Robotic manipulation with deformable objects represents a data-intensive regime in embodied learning, where shape, contact, and topology co-evolve in ways that far exceed the variability of rigids. Although simulation promises relief from the cost of real-world data acquisition, prevailing sim-to-real pipelines remain rooted in rigid-body abstractions, producing mismatched geometry, fragile soft dynamics, and motion primitives poorly suited for cloth interaction. We posit that simulation fails not for being synthetic, but for being ungrounded. To address this, we introduce SIM1, a physics-aligned real-to-sim-to-real data engine that grounds simulation in the physical world. Given limited demonstrations, the system digitizes scenes into metric-consistent twins, calibrates deformable dynamics through elastic modeling, and expands behaviors via diffusion-based trajectory generation with quality filtering. This pipeline transforms sparse observations into scaled synthetic supervision with near-demonstration fidelity. Experiments show that policies trained on purely synthetic data achieve parity with real-data baselines at a 1:15 equivalence ratio, while delivering 90% zero-shot success and 50% generalization gains in real-world deployment. These results validate physics-aligned simulation as scalable supervision for deformable manipulation and a practical pathway for data-efficient policy learning. 

---
# Seeing but Not Thinking: Routing Distraction in Multimodal Mixture-of-Experts 

**Authors**: Haolei Xu, Haiwen Hong, Hongxing Li, Rui Zhou, Yang Zhang, Longtao Huang, Hui Xue, Yongliang Shen, Weiming Lu, Yueting Zhuang  

**Link**: [PDF](https://arxiv.org/pdf/2604.08541)  

**Abstract**: Multimodal Mixture-of-Experts (MoE) models have achieved remarkable performance on vision-language tasks. However, we identify a puzzling phenomenon termed Seeing but Not Thinking: models accurately perceive image content yet fail in subsequent reasoning, while correctly solving identical problems presented as pure text. Through systematic analysis, we first verify that cross-modal semantic sharing exists in MoE architectures, ruling out semantic alignment failure as the sole explanation. We then reveal that visual experts and domain experts exhibit layer-wise separation, with image inputs inducing significant routing divergence from text inputs in middle layers where domain experts concentrate. Based on these findings, we propose the Routing Distraction hypothesis: when processing visual inputs, the routing mechanism fails to adequately activate task-relevant reasoning experts. To validate this hypothesis, we design a routing-guided intervention method that enhances domain expert activation. Experiments on three multimodal MoE models across six benchmarks demonstrate consistent improvements, with gains of up to 3.17% on complex visual reasoning tasks. Our analysis further reveals that domain expert identification locates cognitive functions rather than sample-specific solutions, enabling effective transfer across tasks with different information structures. 

---
# AVGen-Bench: A Task-Driven Benchmark for Multi-Granular Evaluation of Text-to-Audio-Video Generation 

**Authors**: Ziwei Zhou, Zeyuan Lai, Rui Wang, Yifan Yang, Zhen Xing, Yuqing Yang, Qi Dai, Lili Qiu, Chong Luo  

**Link**: [PDF](https://arxiv.org/pdf/2604.08540)  

**Abstract**: Text-to-Audio-Video (T2AV) generation is rapidly becoming a core interface for media creation, yet its evaluation remains fragmented. Existing benchmarks largely assess audio and video in isolation or rely on coarse embedding similarity, failing to capture the fine-grained joint correctness required by realistic prompts. We introduce AVGen-Bench, a task-driven benchmark for T2AV generation featuring high-quality prompts across 11 real-world categories. To support comprehensive assessment, we propose a multi-granular evaluation framework that combines lightweight specialist models with Multimodal Large Language Models (MLLMs), enabling evaluation from perceptual quality to fine-grained semantic controllability. Our evaluation reveals a pronounced gap between strong audio-visual aesthetics and weak semantic reliability, including persistent failures in text rendering, speech coherence, physical reasoning, and a universal breakdown in musical pitch control. Code and benchmark resources are available at this http URL. 

---
# OpenVLThinkerV2: A Generalist Multimodal Reasoning Model for Multi-domain Visual Tasks 

**Authors**: Wenbo Hu, Xin Chen, Yan Gao-Tian, Yihe Deng, Nanyun Peng, Kai-Wei Chang  

**Link**: [PDF](https://arxiv.org/pdf/2604.08539)  

**Abstract**: Group Relative Policy Optimization (GRPO) has emerged as the de facto Reinforcement Learning (RL) objective driving recent advancements in Multimodal Large Language Models. However, extending this success to open-source multimodal generalist models remains heavily constrained by two primary challenges: the extreme variance in reward topologies across diverse visual tasks, and the inherent difficulty of balancing fine-grained perception with multi-step reasoning capabilities. To address these issues, we introduce Gaussian GRPO (G$^2$RPO), a novel RL training objective that replaces standard linear scaling with non-linear distributional matching. By mathematically forcing the advantage distribution of any given task to strictly converge to a standard normal distribution, $\mathcal{N}(0,1)$, G$^2$RPO theoretically ensures inter-task gradient equity, mitigates vulnerabilities to heavy-tail outliers, and offers symmetric update for positive and negative rewards. Leveraging the enhanced training stability provided by G$^2$RPO, we introduce two task-level shaping mechanisms to seamlessly balance perception and reasoning. First, response length shaping dynamically elicits extended reasoning chains for complex queries while enforce direct outputs to bolster visual grounding. Second, entropy shaping tightly bounds the model's exploration zone, effectively preventing both entropy collapse and entropy explosion. Integrating these methodologies, we present OpenVLThinkerV2, a highly robust, general-purpose multimodal model. Extensive evaluations across 18 diverse benchmarks demonstrate its superior performance over strong open-source and leading proprietary frontier models. 

---
# RewardFlow: Generate Images by Optimizing What You Reward 

**Authors**: Onkar Susladkar, Dong-Hwan Jang, Tushar Prakash, Adheesh Juvekar, Vedant Shah, Ayush Barik, Nabeel Bashir, Muntasir Wahed, Ritish Shrirao, Ismini Lourentzou  

**Link**: [PDF](https://arxiv.org/pdf/2604.08536)  

**Abstract**: We introduce RewardFlow, an inversion-free framework that steers pretrained diffusion and flow-matching models at inference time through multi-reward Langevin dynamics. RewardFlow unifies complementary differentiable rewards for semantic alignment, perceptual fidelity, localized grounding, object consistency, and human preference, and further introduces a differentiable VQA-based reward that provides fine-grained semantic supervision through language-vision reasoning. To coordinate these heterogeneous objectives, we design a prompt-aware adaptive policy that extracts semantic primitives from the instruction, infers edit intent, and dynamically modulates reward weights and step sizes throughout sampling. Across several image editing and compositional generation benchmarks, RewardFlow delivers state-of-the-art edit fidelity and compositional alignment. 

---
# PSI: Shared State as the Missing Layer for Coherent AI-Generated Instruments in Personal AI Agents 

**Authors**: Zhiyuan Wang, Erzhen Hu, Mark Rucker, Laura E. Barnes  

**Link**: [PDF](https://arxiv.org/pdf/2604.08529)  

**Abstract**: Personal AI tools can now be generated from natural-language requests, but they often remain isolated after creation. We present PSI, a shared-state architecture that turns independently generated modules into coherent instruments: persistent, connected, and chat-complementary artifacts accessible through both GUIs and a generic chat agent. By publishing current state and write-back affordances to a shared personal-context bus, modules enable cross-module reasoning and synchronized actions across interfaces. We study PSI through a three-week autobiographical deployment in a self-developed personal AI environment and show that later-generated instruments can be integrated automatically through the same contract. PSI identifies shared state as the missing systems layer that transforms AI-generated personal software from isolated apps into coherent personal computing environments. 

---
# What Drives Representation Steering? A Mechanistic Case Study on Steering Refusal 

**Authors**: Stephen Cheng, Sarah Wiegreffe, Dinesh Manocha  

**Link**: [PDF](https://arxiv.org/pdf/2604.08524)  

**Abstract**: Applying steering vectors to large language models (LLMs) is an efficient and effective model alignment technique, but we lack an interpretable explanation for how it works-- specifically, what internal mechanisms steering vectors affect and how this results in different model outputs. To investigate the causal mechanisms underlying the effectiveness of steering vectors, we conduct a comprehensive case study on refusal. We propose a multi-token activation patching framework and discover that different steering methodologies leverage functionally interchangeable circuits when applied at the same layer. These circuits reveal that steering vectors primarily interact with the attention mechanism through the OV circuit while largely ignoring the QK circuit-- freezing all attention scores during steering drops performance by only 8.75% across two model families. A mathematical decomposition of the steered OV circuit further reveals semantically interpretable concepts, even in cases where the steering vector itself does not. Leveraging the activation patching results, we show that steering vectors can be sparsified by up to 90-99% while retaining most performance, and that different steering methodologies agree on a subset of important dimensions. 

---
# ClawBench: Can AI Agents Complete Everyday Online Tasks? 

**Authors**: Yuxuan Zhang, Yubo Wang, Yipeng Zhu, Penghui Du, Junwen Miao, Xuan Lu, Wendong Xu, Yunzhuo Hao, Songcheng Cai, Xiaochen Wang, Huaisong Zhang, Xian Wu, Yi Lu, Minyi Lei, Kai Zou, Huifeng Yin, Ping Nie, Liang Chen, Dongfu Jiang, Wenhu Chen, Kelsey R. Allen  

**Link**: [PDF](https://arxiv.org/pdf/2604.08523)  

**Abstract**: AI agents may be able to automate your inbox, but can they automate other routine aspects of your life? Everyday online tasks offer a realistic yet unsolved testbed for evaluating the next generation of AI agents. To this end, we introduce ClawBench, an evaluation framework of 153 simple tasks that people need to accomplish regularly in their lives and work, spanning 144 live platforms across 15 categories, from completing purchases and booking appointments to submitting job applications. These tasks require demanding capabilities beyond existing benchmarks, such as obtaining relevant information from user-provided documents, navigating multi-step workflows across diverse platforms, and write-heavy operations like filling in many detailed forms correctly. Unlike existing benchmarks that evaluate agents in offline sandboxes with static pages, ClawBench operates on production websites, preserving the full complexity, dynamic nature, and challenges of real-world web interaction. A lightweight interception layer captures and blocks only the final submission request, ensuring safe evaluation without real-world side effects. Our evaluations of 7 frontier models show that both proprietary and open-source models can complete only a small portion of these tasks. For example, Claude Sonnet 4.6 achieves only 33.3%. Progress on ClawBench brings us closer to AI agents that can function as reliable general-purpose assistants. 

---
# Differentially Private Language Generation and Identification in the Limit 

**Authors**: Anay Mehrotra, Grigoris Velegkas, Xifan Yu, Felix Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2604.08504)  

**Abstract**: We initiate the study of language generation in the limit, a model recently introduced by Kleinberg and Mullainathan [KM24], under the constraint of differential privacy. We consider the continual release model, where a generator must eventually output a stream of valid strings while protecting the privacy of the entire input sequence. Our first main result is that for countable collections of languages, privacy comes at no qualitative cost: we provide an $\varepsilon$-differentially-private algorithm that generates in the limit from any countable collection. This stands in contrast to many learning settings where privacy renders learnability impossible. However, privacy does impose a quantitative cost: there are finite collections of size $k$ for which uniform private generation requires $\Omega(k/\varepsilon)$ samples, whereas just one sample suffices non-privately.
We then turn to the harder problem of language identification in the limit. Here, we show that privacy creates fundamental barriers. We prove that no $\varepsilon$-DP algorithm can identify a collection containing two languages with an infinite intersection and a finite set difference, a condition far stronger than the classical non-private characterization of identification. Next, we turn to the stochastic setting where the sample strings are sampled i.i.d. from a distribution (instead of being generated by an adversary). Here, we show that private identification is possible if and only if the collection is identifiable in the adversarial model. Together, our results establish new dimensions along which generation and identification differ and, for identification, a separation between adversarial and stochastic settings induced by privacy constraints. 

---
# Quantifying Explanation Consistency: The C-Score Metric for CAM-Based Explainability in Medical Image Classification 

**Authors**: Kabilan Elangovan, Daniel Ting  

**Link**: [PDF](https://arxiv.org/pdf/2604.08502)  

**Abstract**: Class Activation Mapping (CAM) methods are widely used to generate visual explanations for deep learning classifiers in medical imaging. However, existing evaluation frameworks assess whether explanations are correct, measured by localisation fidelity against radiologist annotations, rather than whether they are consistent: whether the model applies the same spatial reasoning strategy across different patients with the same pathology. We propose the C-Score (Consistency Score), a confidence-weighted, annotation-free metric that quantifies intra-class explanation reproducibility via intensity-emphasised pairwise soft IoU across correctly classified instances. We evaluate six CAM techniques: GradCAM, GradCAM++, LayerCAM, EigenCAM, ScoreCAM, and MS GradCAM++ across three CNN architectures (DenseNet201, InceptionV3, ResNet50V2) over thirty training epochs on the Kermany chest X-ray dataset, covering transfer learning and fine-tuning phases. We identify three distinct mechanisms of AUC-consistency dissociation, invisible to standard classification metrics: threshold-mediated gold list collapse, technique-specific attribution collapse at peak AUC, and class-level consistency masking in global aggregation. C-Score provides an early warning signal of impending model instability. ScoreCAM deterioration on ResNet50V2 is detectable one full checkpoint before catastrophic AUC collapse and yields architecture-specific clinical deployment recommendations grounded in explanation quality rather than predictive ranking alone. 

---
# PIArena: A Platform for Prompt Injection Evaluation 

**Authors**: Runpeng Geng, Chenlong Yin, Yanting Wang, Ying Chen, Jinyuan Jia  

**Link**: [PDF](https://arxiv.org/pdf/2604.08499)  

**Abstract**: Prompt injection attacks pose serious security risks across a wide range of real-world applications. While receiving increasing attention, the community faces a critical gap: the lack of a unified platform for prompt injection evaluation. This makes it challenging to reliably compare defenses, understand their true robustness under diverse attacks, or assess how well they generalize across tasks and benchmarks. For instance, many defenses initially reported as effective were later found to exhibit limited robustness on diverse datasets and attacks. To bridge this gap, we introduce PIArena, a unified and extensible platform for prompt injection evaluation that enables users to easily integrate state-of-the-art attacks and defenses and evaluate them across a variety of existing and new benchmarks. We also design a dynamic strategy-based attack that adaptively optimizes injected prompts based on defense feedback. Through comprehensive evaluation using PIArena, we uncover critical limitations of state-of-the-art defenses: limited generalizability across tasks, vulnerability to adaptive attacks, and fundamental challenges when an injected task aligns with the target task. The code and datasets are available at this https URL. 

---
# Faithful GRPO: Improving Visual Spatial Reasoning in Multimodal Language Models via Constrained Policy Optimization 

**Authors**: Sai Srinivas Kancheti, Aditya Kanade, Rohit Sinha, Vineeth N Balasubramanian, Tanuja Ganu  

**Link**: [PDF](https://arxiv.org/pdf/2604.08476)  

**Abstract**: Multimodal reasoning models (MRMs) trained with reinforcement learning with verifiable rewards (RLVR) show improved accuracy on visual reasoning benchmarks. However, we observe that accuracy gains often come at the cost of reasoning quality: generated Chain-of-Thought (CoT) traces are frequently inconsistent with the final answer and poorly grounded in the visual evidence. We systematically study this phenomenon across seven challenging real-world spatial reasoning benchmarks and find that it affects contemporary MRMs such as ViGoRL-Spatial, TreeVGR as well as our own models trained with standard Group Relative Policy Optimization (GRPO). We characterize CoT reasoning quality along two complementary axes: "logical consistency" (does the CoT entail the final answer?) and "visual grounding" (does each reasoning step accurately describe objects, attributes, and spatial relationships in the image?). To address this, we propose Faithful GRPO (FGRPO), a variant of GRPO that enforces consistency and grounding as constraints via Lagrangian dual ascent. FGRPO incorporates batch-level consistency and grounding constraints into the advantage computation within a group, adaptively adjusting the relative importance of constraints during optimization. We evaluate FGRPO on Qwen2.5-VL-7B and 3B backbones across seven spatial datasets. Our results show that FGRPO substantially improves reasoning quality, reducing the inconsistency rate from 24.5% to 1.7% and improving visual grounding scores by +13%. It also improves final answer accuracy over simple GRPO, demonstrating that faithful reasoning enables better answers. 

---
# TTVS: Boosting Self-Exploring Reinforcement Learning via Test-time Variational Synthesis 

**Authors**: Sikai Bai, Haoxi Li, Jie Zhang, Yongjiang Liu, Song Guo  

**Link**: [PDF](https://arxiv.org/pdf/2604.08468)  

**Abstract**: Despite significant advances in Large Reasoning Models (LRMs) driven by reinforcement learning with verifiable rewards (RLVR), this paradigm is fundamentally limited in specialized or novel domains where such supervision is prohibitively expensive or unavailable, posing a key challenge for test-time adaptation. While existing test-time methods offer a potential solution, they are constrained by learning from static query sets, risking overfitting to textual patterns. To address this gap, we introduce Test-Time Variational Synthesis (TTVS), a novel framework that enables LRMs to self-evolve by dynamically augmenting the training stream from unlabeled test queries. TTVS comprises two synergistic modules: (1) Online Variational Synthesis, which transforms static test queries into a dynamic stream of diverse, semantically-equivalent variations, enforcing the model to learn underlying problem logic rather than superficial patterns; (2) Test-time Hybrid Exploration, which balances accuracy-driven exploitation with consistency-driven exploration across synthetic variants. Extensive experiments show TTVS yields superior performance across eight model architectures. Notably, using only unlabeled test-time data, TTVS not only surpasses other test-time adaptation methods but also outperforms state-of-the-art supervised RL-based techniques trained on vast, high-quality labeled data. 

---
# OVS-DINO: Open-Vocabulary Segmentation via Structure-Aligned SAM-DINO with Language Guidance 

**Authors**: Haoxi Zeng, Qiankun Liu, Yi Bin, Haiyue Zhang, Yujuan Ding, Guoqing Wang, Deqiang Ouyang, Heng Tao Shen  

**Link**: [PDF](https://arxiv.org/pdf/2604.08461)  

**Abstract**: Open-Vocabulary Segmentation (OVS) aims to segment image regions beyond predefined category sets by leveraging semantic descriptions. While CLIP based approaches excel in semantic generalization, they frequently lack the fine-grained spatial awareness required for dense prediction. Recent efforts have incorporated Vision Foundation Models (VFMs) like DINO to alleviate these limitations. However, these methods still struggle with the precise edge perception necessary for high fidelity segmentation. In this paper, we analyze internal representations of DINO and discover that its inherent boundary awareness is not absent but rather undergoes progressive attenuation as features transition into deeper transformer blocks. To address this, we propose OVS-DINO, a novel framework that revitalizes latent edge-sensitivity of DINO through structural alignment with the Segment Anything Model (SAM). Specifically, we introduce a Structure-Aware Encoder (SAE) and a Structure-Modulated Decoder (SMD) to effectively activate boundary features of DINO using SAM's structural priors, complemented by a supervision strategy utilizing SAM generated pseudo-masks. Extensive experiments demonstrate that our method achieves state-of-the-art performance across multiple weakly-supervised OVS benchmarks, improving the average score by 2.1% (from 44.8% to 46.9%). Notably, our approach significantly enhances segmentation accuracy in complex, cluttered scenarios, with a gain of 6.3% on Cityscapes (from 36.6% to 42.9%). 

---
# A Machine Learning Framework for Turbofan Health Estimation via Inverse Problem Formulation 

**Authors**: Milad Leyli-Abadi, Lucas Thil, Sebastien Razakarivony, Guillaume Doquet, Jesse Read  

**Link**: [PDF](https://arxiv.org/pdf/2604.08460)  

**Abstract**: Estimating the health state of turbofan engines is a challenging ill-posed inverse problem, hindered by sparse sensing and complex nonlinear thermodynamics. Research in this area remains fragmented, with comparisons limited by the use of unrealistic datasets and insufficient exploration of the exploitation of temporal information. This work investigates how to recover component-level health indicators from operational sensor data under realistic degradation and maintenance patterns. To support this study, we introduce a new dataset that incorporates industry-oriented complexities such as maintenance events and usage changes. Using this dataset, we establish an initial benchmark that compares steady-state and nonstationary data-driven models, and Bayesian filters, classic families of methods used to solve this problem. In addition to this benchmark, we introduce self-supervised learning (SSL) approaches that learn latent representations without access to true health labels, a scenario reflective of real-world operational constraints. By comparing the downstream estimation performance of these unsupervised representations against the direct prediction baselines, we establish a practical lower bound on the difficulty of solving this inverse problem. Our results reveal that traditional filters remain strong baselines, while SSL methods reveal the intrinsic complexity of health estimation and highlight the need for more advanced and interpretable inference strategies. For reproducibility, both the generated dataset and the implementation used in this work are made accessible. 

---
# CrashSight: A Phase-Aware, Infrastructure-Centric Video Benchmark for Traffic Crash Scene Understanding and Reasoning 

**Authors**: Rui Gan, Junyi Ma, Pei Li, Xingyou Yang, Kai Chen, Sikai Chen, Bin Ran  

**Link**: [PDF](https://arxiv.org/pdf/2604.08457)  

**Abstract**: Cooperative autonomous driving requires traffic scene understanding from both vehicle and infrastructure perspectives. While vision-language models (VLMs) show strong general reasoning capabilities, their performance in safety-critical traffic scenarios remains insufficiently evaluated due to the ego-vehicle focus of existing benchmarks. To bridge this gap, we present \textbf{CrashSight}, a large-scale vision-language benchmark for roadway crash understanding using real-world roadside camera data. The dataset comprises 250 crash videos, annotated with 13K multiple-choice question-answer pairs organized under a two-tier taxonomy. Tier 1 evaluates the visual grounding of scene context and involved parties, while Tier 2 probes higher-level reasoning, including crash mechanics, causal attribution, temporal progression, and post-crash outcomes. We benchmark 8 state-of-the-art VLMs and show that, despite strong scene description capabilities, current models struggle with temporal and causal reasoning in safety-critical scenarios. We provide a detailed analysis of failure scenarios and discuss directions for improving VLM crash understanding. The benchmark provides a standardized evaluation framework for infrastructure-assisted perception in cooperative autonomous driving. The CrashSight benchmark, including the full dataset and code, is accessible at this https URL. 

---
# HST-HGN: Heterogeneous Spatial-Temporal Hypergraph Networks with Bidirectional State Space Models for Global Fatigue Assessment 

**Authors**: Changdao Chen  

**Link**: [PDF](https://arxiv.org/pdf/2604.08435)  

**Abstract**: It remains challenging to assess driver fatigue from untrimmed videos under constrained computational budgets, due to the difficulty of modeling long-range temporal dependencies in subtle facial expressions. Some existing approaches rely on computationally heavy architectures, whereas others employ traditional lightweight pairwise graph networks, despite their limited capacity to model high-order synergies and global temporal context. Therefore, we propose HST-HGN, a novel Heterogeneous Spatial-Temporal Hypergraph Network driven by Bidirectional State Space Models. Spatially, we introduce a hierarchical hypergraph network to fuse pose-disentangled geometric topologies with multi-modal texture patches dynamically. This formulation encapsulates high-order synergistic facial deformations, effectively overcoming the limitations of conventional methods. In temporal terms, a Bi-Mamba module with linear complexity is applied to perform bidirectional sequence modeling. This explicit temporal-evolution filtering enables the network to distinguish highly ambiguous transient actions, such as yawning versus speaking, while encompassing their complete physiological lifecycles. Extensive evaluations across diverse fatigue benchmarks demonstrate that HST-HGN achieves state-of-the-art performance. In particular, our method strikes a balance between discriminative power and computational efficiency, making it well-suited for real-time in-cabin edge deployment. 

---
# Small-scale photonic Kolmogorov-Arnold networks using standard telecom nonlinear modules 

**Authors**: Luca Nogueira Calçado, Sergei K. Turitsyn, Egor Manuylovich  

**Link**: [PDF](https://arxiv.org/pdf/2604.08432)  

**Abstract**: Photonic neural networks promise ultrafast inference, yet most architectures rely on linear optical meshes with electronic nonlinearities, reintroducing optical-electrical-optical bottlenecks. Here we introduce small-scale photonic Kolmogorov-Arnold networks (SSP-KANs) implemented entirely with standard telecommunications components. Each network edge employs a trainable nonlinear module composed of a Mach-Zehnder interferometer, semiconductor optical amplifier, and variable optical attenuators, providing a four-parameter transfer function derived from gain saturation and interferometric mixing. Despite this constrained expressivity, SSP-KANs comprising only a few optical modules achieve strong nonlinear inference performance across classification, regression, and image recognition tasks, approaching software baselines with significantly fewer parameters. A four-module network achieves 98.4\% accuracy on nonlinear classification benchmarks inaccessible to linear models. Performance remains robust under realistic hardware impairments, maintaining high accuracy down to 6-bit input resolution and 14 dB signal-to-noise ratio. By using a fully differentiable physics model for end-to-end optimisation of optical parameters, this work establishes a practical pathway from simulation to experimental demonstration of photonic KANs using commodity telecom hardware. 

---
# KV Cache Offloading for Context-Intensive Tasks 

**Authors**: Andrey Bocharnikov, Ivan Ermakov, Denis Kuznedelev, Vyacheslav Zhdanovskiy, Yegor Yershov  

**Link**: [PDF](https://arxiv.org/pdf/2604.08426)  

**Abstract**: With the growing demand for long-context LLMs across a wide range of applications, the key-value (KV) cache has become a critical bottleneck for both latency and memory usage. Recently, KV-cache offloading has emerged as a promising approach to reduce memory footprint and inference latency while preserving accuracy. Prior evaluations have largely focused on tasks that do not require extracting large amounts of information from the context. In this work, we study KV-cache offloading on context-intensive tasks: problems where the solution requires looking up a lot of information from the input prompt. We create and release the Text2JSON benchmark, a highly context-intensive task that requires extracting structured knowledge from raw text. We evaluate modern KV offloading on Text2JSON and other context-intensive tasks and find significant performance degradation on both Llama 3 and Qwen 3 models. Our analysis identifies two key reasons for poor accuracy: low-rank projection of keys and unreliable landmarks, and proposes a simpler alternative strategy that significantly improves accuracy across multiple LLM families and benchmarks. These findings highlight the need for a comprehensive and rigorous evaluation of long-context compression techniques. 

---
# Synthetic Data for any Differentiable Target 

**Authors**: Tristan Thrush, Sung Min Park, Herman Brunborg, Luke Bailey, Marcel Roed, Neil Band, Christopher Potts, Tatsunori Hashimoto  

**Link**: [PDF](https://arxiv.org/pdf/2604.08423)  

**Abstract**: What are the limits of controlling language models via synthetic training data? We develop a reinforcement learning (RL) primitive, the Dataset Policy Gradient (DPG), which can precisely optimize synthetic data generators to produce a dataset of targeted examples. When used for supervised fine-tuning (SFT) of a target model, these examples cause the target model to do well on a differentiable metric of our choice. Our approach achieves this by taking exact data attribution via higher-order gradients and using those scores as policy gradient rewards. We prove that this procedure closely approximates the true, intractable gradient for the synthetic data generator. To illustrate the potential of DPG, we show that, using only SFT on generated examples, we can cause the target model's LM head weights to (1) embed a QR code, (2) embed the pattern $\texttt{67}$, and (3) have lower $\ell^2$ norm. We additionally show that we can cause the generator to (4) rephrase inputs in a new language and (5) produce a specific UUID, even though neither of these objectives is conveyed in the generator's input prompts. These findings suggest that DPG is a powerful and flexible technique for shaping model properties using only synthetic training examples. 

---
# Exploring Temporal Representation in Neural Processes for Multimodal Action Prediction 

**Authors**: Marco Gabriele Fedozzi, Yukie Nagai, Francesco Rea, Alessandra Sciutti  

**Link**: [PDF](https://arxiv.org/pdf/2604.08418)  

**Abstract**: Inspired by the human ability to understand and predict others, we study the applicability of Conditional Neural Processes (CNP) to the task of self-supervised multimodal action prediction in robotics. Following recent results regarding the ontogeny of the Mirror Neuron System (MNS), we focus on the preliminary objective of self-actions prediction. We find a good MNS-inspired model in the existing Deep Modality Blending Network (DMBN), able to reconstruct the visuo-motor sensory signal during a partially observed action sequence by leveraging the probabilistic generation of CNP. After a qualitative and quantitative evaluation, we highlight its difficulties in generalizing to unseen action sequences, and identify the cause in its inner representation of time. Therefore, we propose a revised version, termed DMBN-Positional Time Encoding (DMBN-PTE), that facilitates learning a more robust representation of temporal information, and provide preliminary results of its effectiveness in expanding the applicability of the architecture. DMBN-PTE figures as a first step in the development of robotic systems that autonomously learn to forecast actions on longer time scales refining their predictions with incoming observations. 

---
# Selective Attention System (SAS): Device-Addressed Speech Detection for Real-Time On-Device Voice AI 

**Authors**: David Joohun Kim, Daniyal Anjum, Bonny Banerjee, Omar Abbasi  

**Link**: [PDF](https://arxiv.org/pdf/2604.08412)  

**Abstract**: We study device-addressed speech detection under pre-ASR edge deployment constraints, where systems must decide whether to forward audio before transcription under strict latency and compute limits. We show that, in multi-speaker environments with temporally ambiguous utterances, this task is more effectively modelled as a sequential routing problem over interaction history than as an utterance-local classification task. We formalize this as Sequential Device-Addressed Routing (SDAR) and present the Selective Attention System (SAS), an on-device implementation that instantiates this formulation.
On a held-out 60-hour multi-speaker English test set, the primary audio-only configuration achieves F1=0.86 (precision=0.89, recall=0.83); with an optional camera, audio+video fusion raises F1 to 0.95 (precision=0.97, recall=0.93). Removing causal interaction history (Stage~3) reduced F1 from 0.95 to 0.57+/-0.03 in the audio+video configuration under our evaluation protocol. Among the tested components, this was the largest observed ablation effect, indicating that short-horizon interaction history carries substantial decision-relevant information in the evaluated setting. SAS runs fully on-device on ARM Cortex-A class hardware (<150 ms latency, <20 MB footprint). All results are from internal evaluation on a proprietary dataset evaluated primarily in English; a 5-hour evaluation subset may be shared for independent verification (Section 8.8). 

---
# Zero-shot Multivariate Time Series Forecasting Using Tabular Prior Fitted Networks 

**Authors**: Mayuka Jayawardhana, Nihal Sharma, Kazem Meidani, Bayan Bruss, Tom Goldstein, Doron Bergman  

**Link**: [PDF](https://arxiv.org/pdf/2604.08400)  

**Abstract**: Tabular foundation models, particularly Prior-data Fitted Networks like TabPFN have emerged as the leading contender in a myriad of tasks ranging from data imputation to label prediction on the tabular data format surpassing the historical successes of tree-based models. This has led to investigations on their applicability to forecasting time series data which can be formulated as a tabular problem. While recent work to this end has displayed positive results, most works have limited their treatment of multivariate time series problems to several independent univariate time series forecasting subproblems, thus ignoring any inter-channel interactions. Overcoming this limitation, we introduce a generally applicable framework for multivariate time series forecasting using tabular foundation models. We achieve this by recasting the multivariate time series forecasting problem as a series of scalar regression problems which can then be solved zero-shot by any tabular foundation model with regression capabilities. We present results of our method using the TabPFN-TS backbone and compare performance with the current state of the art tabular methods. 

---
# ADAPTive Input Training for Many-to-One Pre-Training on Time-Series Classification 

**Authors**: Paul Quinlan, Qingguo Li, Xiaodan Zhu  

**Link**: [PDF](https://arxiv.org/pdf/2604.08398)  

**Abstract**: Recent work on time-series models has leveraged self-supervised training to learn meaningful features and patterns in order to improve performance on downstream tasks and generalize to unseen modalities. While these pretraining methods have shown great promise in one-to-many scenarios, where a model is pre-trained on one dataset and fine-tuned on a downstream dataset, they have struggled to generalize to new datasets when more datasets are added during pre-training. This is a fundamental challenge in building foundation models for time-series data, as it limits the ability to develop models that can learn from a large variety of diverse datasets available. To address this challenge, we present a new pre-training paradigm for time-series data called ADAPT, which can efficiently align the physical properties of data in the time-series domain, enabling mixed-batch pre-training despite the extreme discrepancies in the input sizes and channel dimensions of pre-training data. We trained on 162 time-series classification datasets and set new state-of-the-art performance for classification benchmarks. We successfully train a model within the time-series domain on a wide range of datasets simultaneously, which is a major building block for building generalist foundation models in time-series domains. 

---
# Phantasia: Context-Adaptive Backdoors in Vision Language Models 

**Authors**: Nam Duong Tran, Phi Le Nguyen  

**Link**: [PDF](https://arxiv.org/pdf/2604.08395)  

**Abstract**: Recent advances in Vision-Language Models (VLMs) have greatly enhanced the integration of visual perception and linguistic reasoning, driving rapid progress in multimodal understanding. Despite these achievements, the security of VLMs, particularly their vulnerability to backdoor attacks, remains significantly underexplored. Existing backdoor attacks on VLMs are still in an early stage of development, with most current methods relying on generating poisoned responses that contain fixed, easily identifiable patterns. In this work, we make two key contributions. First, we demonstrate for the first time that the stealthiness of existing VLM backdoor attacks has been substantially overestimated. By adapting defense techniques originally designed for other domains (e.g., vision-only and text-only models), we show that several state-of-the-art attacks can be detected with surprising ease. Second, to address this gap, we introduce Phantasia, a context-adaptive backdoor attack that dynamically aligns its poisoned outputs with the semantics of each input. Instead of producing static poisoned patterns, Phantasia encourages models to generate contextually coherent yet malicious responses that remain plausible, thereby significantly improving stealth and adaptability. Extensive experiments across diverse VLM architectures reveal that Phantasia achieves state-of-the-art attack success rates while maintaining benign performance under various defensive settings. 

---
# TASU2: Controllable CTC Simulation for Alignment and Low-Resource Adaptation of Speech LLMs 

**Authors**: Jing Peng, Chenghao Wang, Yi Yang, Lirong Qian, Junjie Li, Yu Xi, Shuai Wang, Kai Yu  

**Link**: [PDF](https://arxiv.org/pdf/2604.08384)  

**Abstract**: Speech LLM post-training increasingly relies on efficient cross-modal alignment and robust low-resource adaptation, yet collecting large-scale audio-text pairs remains costly. Text-only alignment methods such as TASU reduce this burden by simulating CTC posteriors from transcripts, but they provide limited control over uncertainty and error rate, making curriculum design largely heuristic. We propose \textbf{TASU2}, a controllable CTC simulation framework that simulates CTC posterior distributions under a specified WER range, producing text-derived supervision that better matches the acoustic decoding interface. This enables principled post-training curricula that smoothly vary supervision difficulty without TTS. Across multiple source-to-target adaptation settings, TASU2 improves in-domain and out-of-domain recognition over TASU, and consistently outperforms strong baselines including text-only fine-tuning and TTS-based augmentation, while mitigating source-domain performance degradation. 

---
# A GAN and LLM-Driven Data Augmentation Framework for Dynamic Linguistic Pattern Modeling in Chinese Sarcasm Detection 

**Authors**: Wenxian Wang, Xiaohu Luo, Junfeng Hao, Xiaoming Gu, Xingshu Chen, Zhu Wang, Haizhou Wang  

**Link**: [PDF](https://arxiv.org/pdf/2604.08381)  

**Abstract**: Sarcasm is a rhetorical device that expresses criticism or emphasizes characteristics of certain individuals or situations through exaggeration, irony, or comparison. Existing methods for Chinese sarcasm detection are constrained by limited datasets and high construction costs, and they mainly focus on textual features, overlooking user-specific linguistic patterns that shape how opinions and emotions are expressed. This paper proposes a Generative Adversarial Network (GAN) and Large Language Model (LLM)-driven data augmentation framework to dynamically model users' linguistic patterns for enhanced Chinese sarcasm detection. First, we collect raw data from various topics on Sina Weibo. Then, we train a GAN on these data and apply a GPT-3.5 based data augmentation technique to synthesize an extended sarcastic comment dataset, named SinaSarc. This dataset contains target comments, contextual information, and user historical behavior. Finally, we extend the BERT architecture to incorporate multi-dimensional information, particularly user historical behavior, enabling the model to capture dynamic linguistic patterns and uncover implicit sarcastic cues in comments. Experimental results demonstrate the effectiveness of our proposed method. Specifically, our model achieves the highest F1-scores on both the non-sarcastic and sarcastic categories, with values of 0.9138 and 0.9151 respectively, which outperforms all existing state-of-the-art (SOTA) approaches. This study presents a novel framework for dynamically modeling users' long-term linguistic patterns in Chinese sarcasm detection, contributing to both dataset construction and methodological advancement in this field. 

---
# Scaling-Aware Data Selection for End-to-End Autonomous Driving Systems 

**Authors**: Tolga Dimlioglu, Nadine Chang, Maying Shen, Rafid Mahmood, Jose M. Alvarez  

**Link**: [PDF](https://arxiv.org/pdf/2604.08366)  

**Abstract**: Large-scale deep learning models for physical AI applications depend on diverse training data collection efforts. These models and correspondingly, the training data, must address different evaluation criteria necessary for the models to be deployable in real-world environments. Data selection policies can guide the development of the training set, but current frameworks do not account for the ambiguity in how data points affect different metrics. In this work, we propose Mixture Optimization via Scaling-Aware Iterative Collection (MOSAIC), a general data selection framework that operates by: (i) partitioning the dataset into domains; (ii) fitting neural scaling laws from each data domain to the evaluation metrics; and (iii) optimizing a data mixture by iteratively adding data from domains that maximize the change in metrics. We apply MOSAIC to autonomous driving (AD), where an End-to-End (E2E) planner model is evaluated on the Extended Predictive Driver Model Score (EPDMS), an aggregate of driving rule compliance metrics. Here, MOSAIC outperforms a diverse set of baselines on EPDMS with up to 80\% less data. 

---
# Scalable Neural Decoders for Practical Fault-Tolerant Quantum Computation 

**Authors**: Andi Gu, J. Pablo Bonilla Ataides, Mikhail D. Lukin, Susanne F. Yelin  

**Link**: [PDF](https://arxiv.org/pdf/2604.08358)  

**Abstract**: Quantum error correction (QEC) is essential for scalable quantum computing. However, it requires classical decoders that are fast and accurate enough to keep pace with quantum hardware. While quantum low-density parity-check codes have recently emerged as a promising route to efficient fault tolerance, current decoding algorithms do not allow one to realize the full potential of these codes in practical settings. Here, we introduce a convolutional neural network decoder that exploits the geometric structure of QEC codes, and use it to probe a novel "waterfall" regime of error suppression, demonstrating that the logical error rates required for large-scale fault-tolerant algorithms are attainable with modest code sizes at current physical error rates, and with latencies within the real-time budgets of several leading hardware platforms. For example, for the $[144, 12, 12]$ Gross code, the decoder achieves logical error rates up to $\sim 17$x below existing decoders - reaching logical error rates $\sim 10^{-10}$ at physical error $p=0.1\%$ - with 3-5 orders of magnitude higher throughput. This decoder also produces well-calibrated confidence estimates that can significantly reduce the time overhead of repeat-until-success protocols. Taken together, these results suggest that the space-time costs associated with fault-tolerant quantum computation may be significantly lower than previously anticipated. 

---
# PokeGym: A Visually-Driven Long-Horizon Benchmark for Vision-Language Models 

**Authors**: Ruizhi Zhang, Ye Huang, Yuangang Pan, Chuanfu Shen, Zhilin Liu, Ting Xie, Wen Li, Lixin Duan  

**Link**: [PDF](https://arxiv.org/pdf/2604.08340)  

**Abstract**: While Vision-Language Models (VLMs) have achieved remarkable progress in static visual understanding, their deployment in complex 3D embodied environments remains severely limited. Existing benchmarks suffer from four critical deficiencies: (1) passive perception tasks circumvent interactive dynamics; (2) simplified 2D environments fail to assess depth perception; (3) privileged state leakage bypasses genuine visual processing; and (4) human evaluation is prohibitively expensive and unscalable. We introduce PokeGym, a visually-driven long-horizon benchmark instantiated within Pokemon Legends: Z-A, a visually complex 3D open-world Role-Playing Game. PokeGym enforces strict code-level isolation: agents operate solely on raw RGB observations while an independent evaluator verifies success via memory scanning, ensuring pure vision-based decision-making and automated, scalable assessment. The benchmark comprises 30 tasks (30-220 steps) spanning navigation, interaction, and mixed scenarios, with three instruction granularities (Visual-Guided, Step-Guided, Goal-Only) to systematically deconstruct visual grounding, semantic reasoning, and autonomous exploration capabilities. Our evaluation reveals a key limitation of current VLMs: physical deadlock recovery, rather than high-level planning, constitutes the primary bottleneck, with deadlocks showing a strong negative correlation with task success. Furthermore, we uncover a metacognitive divergence: weaker models predominantly suffer from Unaware Deadlocks (oblivious to entrapment), whereas advanced models exhibit Aware Deadlocks (recognizing entrapment yet failing to recover). These findings highlight the need to integrate explicit spatial intuition into VLM architectures. The code and benchmark will be available on GitHub. 

---
# InstAP: Instance-Aware Vision-Language Pre-Train for Spatial-Temporal Understanding 

**Authors**: Ashutosh Kumar, Rajat Saini, Jingjing Pan, Mustafa Erdogan, Mingfang Zhang, Betty Le Dem, Norimasa Kobori, Quan Kong  

**Link**: [PDF](https://arxiv.org/pdf/2604.08337)  

**Abstract**: Current vision-language pre-training (VLP) paradigms excel at global scene understanding but struggle with instance-level reasoning due to global-only supervision. We introduce InstAP, an Instance-Aware Pre-training framework that jointly optimizes global vision-text alignment and fine-grained, instance-level contrastive alignment by grounding textual mentions to specific spatial-temporal regions. To support this, we present InstVL, a large-scale dataset (2 million images, 50,000 videos) with dual-granularity annotations: holistic scene captions and dense, grounded instance descriptions. On the InstVL benchmark, InstAP substantially outperforms existing VLP models on instance-level retrieval, and also surpasses a strong VLP baseline trained on the exact same data corpus, isolating the benefit of our instance-aware objective. Moreover, instance-centric pre-training improves global understanding: InstAP achieves competitive zero-shot performance on multiple video benchmarks, including MSR-VTT and DiDeMo. Qualitative visualizations further show that InstAP localizes textual mentions to the correct instances, while global-only models exhibit more diffuse, scene-level attention. 

---
# Dead Weights, Live Signals: Feedforward Graphs of Frozen Language Models 

**Authors**: Marcus Armstrong, Navid Ayoobi, Arjun Mukherjee  

**Link**: [PDF](https://arxiv.org/pdf/2604.08335)  

**Abstract**: We present a feedforward graph architecture in which heterogeneous frozen large language models serve as computational nodes, communicating through a shared continuous latent space via learned linear projections. Building on recent work demonstrating geometric compatibility between independently trained LLM latent spaces~\cite{armstrong2026thinking}, we extend this finding from static two-model steering to end-to-end trainable multi-node graphs, where projection matrices are optimized jointly via backpropagation through residual stream injection hooks. Three small frozen models (Llama-3.2-1B, Qwen2.5-1.5B, Gemma-2-2B) encode the input into a shared latent space whose aggregate signal is injected into two larger frozen models (Phi-3-mini, Mistral-7B), whose representations feed a lightweight cross-attention output node. With only 17.6M trainable parameters against approximately 12B frozen, the architecture achieves 87.3\% on ARC-Challenge, 82.8\% on OpenBookQA, and 67.2\% on MMLU, outperforming the best single constituent model by 11.4, 6.2, and 1.2 percentage points respectively, and outperforming parameter-matched learned classifiers on frozen single models by 9.1, 5.2, and 6.7 points. Gradient flow through multiple frozen model boundaries is empirically verified to be tractable, and the output node develops selective routing behavior across layer-2 nodes without explicit supervision. 

---
# Lost in the Hype: Revealing and Dissecting the Performance Degradation of Medical Multimodal Large Language Models in Image Classification 

**Authors**: Xun Zhu, Fanbin Mo, Xi Chen, Kaili Zheng, Shaoshuai Yang, Yiming Shi, Jian Gao, Miao Li, Ji Wu  

**Link**: [PDF](https://arxiv.org/pdf/2604.08333)  

**Abstract**: The rise of multimodal large language models (MLLMs) has sparked an unprecedented wave of applications in the field of medical imaging analysis. However, as one of the earliest and most fundamental tasks integrated into this paradigm, medical image classification reveals a sobering reality: state-of-the-art medical MLLMs consistently underperform compared to traditional deep learning models, despite their overwhelming advantages in pre-training data and model parameters. This paradox prompts a critical rethinking: where exactly does the performance degradation originate? In this paper, we conduct extensive experiments on 14 open-source medical MLLMs across three representative image classification datasets. Moving beyond superficial performance benchmarking, we employ feature probing to track the information flow of visual features module-by-module and layer-by-layer throughout the entire MLLM pipeline, enabling explicit visualization of where and how classification signals are distorted, diluted, or overridden. As the first attempt to dissect classification performance degradation in medical MLLMs, our findings reveal four failure modes: 1) quality limitation in visual representation, 2) fidelity loss in connector projection, 3) comprehension deficit in LLM reasoning, and 4) misalignment of semantic mapping. Meanwhile, we introduce quantitative scores that characterize the healthiness of feature evolution, enabling principled comparisons across diverse MLLMs and datasets. Furthermore, we provide insightful discussions centered on the critical barriers that prevent current medical MLLMs from fulfilling their promised clinical potential. We hope that our work provokes rethinking within the community-highlighting that the road from high expectations to clinically deployable MLLMs remains long and winding. 

---
# Multi-Modal Learning meets Genetic Programming: Analyzing Alignment in Latent Space Optimization 

**Authors**: Benjamin Léger, Kazem Meidani, Christian Gagné  

**Link**: [PDF](https://arxiv.org/pdf/2604.08324)  

**Abstract**: Symbolic regression (SR) aims to discover mathematical expressions from data, a task traditionally tackled using Genetic Programming (GP) through combinatorial search over symbolic structures. Latent Space Optimization (LSO) methods use neural encoders to map symbolic expressions into continuous spaces, transforming the combinatorial search into continuous optimization. SNIP (Meidani et al., 2024), a contrastive pre-training model inspired by CLIP, advances LSO by introducing a multi-modal approach: aligning symbolic and numeric encoders in a shared latent space to learn the phenotype-genotype mapping, enabling optimization in the numeric space to implicitly guide symbolic search. However, this relies on fine-grained cross-modal alignment, whereas literature on similar models like CLIP reveals that such an alignment is typically coarse-grained. In this paper, we investigate whether SNIP delivers on its promise of effective bi-modal optimization for SR. Our experiments show that: (1) cross-modal alignment does not improve during optimization, even as fitness increases, and (2) the alignment learned by SNIP is too coarse to efficiently conduct principled search in the symbolic space. These findings reveal that while multi-modal LSO holds significant potential for SR, effective alignment-guided optimization remains unrealized in practice, highlighting fine-grained alignment as a critical direction for future work. 

---
# HistDiT: A Structure-Aware Latent Conditional Diffusion Model for High-Fidelity Virtual Staining in Histopathology 

**Authors**: Aasim Bin Saleem, Amr Ahmed, Ardhendu Behera, Hafeezullah Amin, Iman Yi Liao, Mahmoud Khattab, Pan Jia Wern, Haslina Makmur  

**Link**: [PDF](https://arxiv.org/pdf/2604.08305)  

**Abstract**: Immunohistochemistry (IHC) is essential for assessing specific immune biomarkers like Human Epidermal growth-factor Receptor 2 (HER2) in breast cancer. However, the traditional protocols of obtaining IHC stains are resource-intensive, time-consuming, and prone to structural damages. Virtual staining has emerged as a scalable alternative, but it faces significant challenges in preserving fine-grained cellular structures while accurately translating biochemical expressions. Current state-of-the-art methods still rely on Generative Adversarial Networks (GANs) or standard convolutional U-Net diffusion models that often struggle with "structure and staining trade-offs". The generated samples are either structurally relevant but blurry, or texturally realistic but have artifacts that compromise their diagnostic use. In this paper, we introduce HistDiT, a novel latent conditional Diffusion Transformer (DiT) architecture that establishes a new benchmark for visual fidelity in virtual histological staining. The novelty introduced in this work is, a) the Dual-Stream Conditioning strategy that explicitly maintains a balance between spatial constraints via VAE-encoded latents and semantic phenotype guidance via UNI embeddings; b) the multi-objective loss function that contributes to sharper images with clear morphological structure; and c) the use of the Structural Correlation Metric (SCM) to focus on the core morphological structure for precise assessment of sample quality. Consequently, our model outperforms existing baselines, as demonstrated through rigorous quantitative and qualitative evaluations. 

---
# Securing Retrieval-Augmented Generation: A Taxonomy of Attacks, Defenses, and Future Directions 

**Authors**: Yuming Xu, Mingtao Zhang, Zhuohan Ge, Haoyang Li, Nicole Hu, Jason Chen Zhang, Qing Li, Lei Chen  

**Link**: [PDF](https://arxiv.org/pdf/2604.08304)  

**Abstract**: Retrieval-augmented generation (RAG) significantly enhances large language models (LLMs) but introduces novel security risks through external knowledge access. While existing studies cover various RAG vulnerabilities, they often conflate inherent LLM risks with those specifically introduced by RAG. In this paper, we propose that secure RAG is fundamentally about the security of the external knowledge-access pipeline. We establish an operational boundary to separate inherent LLM flaws from RAG-introduced or RAG-amplified threats. Guided by this perspective, we abstract the RAG workflow into six stages and organize the literature around three trust boundaries and four primary security surfaces, including pre-retrieval knowledge corruption, retrieval-time access manipulation, downstream context exploitation, and knowledge exfiltration. By systematically reviewing the corresponding attacks, defenses, remediation mechanisms, and evaluation benchmarks, we reveal that current defenses remain largely reactive and fragmented. Finally, we discuss these gaps and highlight future directions toward layered, boundary-aware protection across the entire knowledge-access lifecycle. 

---
# DMax: Aggressive Parallel Decoding for dLLMs 

**Authors**: Zigeng Chen, Gongfan Fang, Xinyin Ma, Ruonan Yu, Xinchao Wang  

**Link**: [PDF](https://arxiv.org/pdf/2604.08302)  

**Abstract**: We present DMax, a new paradigm for efficient diffusion language models (dLLMs). It mitigates error accumulation in parallel decoding, enabling aggressive decoding parallelism while preserving generation quality. Unlike conventional masked dLLMs that decode through a binary mask-to-token transition, DMax reformulates decoding as a progressive self-refinement from mask embeddings to token embeddings. At the core of our approach is On-Policy Uniform Training, a novel training strategy that efficiently unifies masked and uniform dLLMs, equipping the model to recover clean tokens from both masked inputs and its own erroneous predictions. Building on this foundation, we further propose Soft Parallel Decoding. We represent each intermediate decoding state as an interpolation between the predicted token embedding and the mask embedding, enabling iterative self-revising in embedding space. Extensive experiments across a variety of benchmarks demonstrate the effectiveness of DMax. Compared with the original LLaDA-2.0-mini, our method improves TPF on GSM8K from 2.04 to 5.47 while preserving accuracy. On MBPP, it increases TPF from 2.71 to 5.86 while maintaining comparable performance. On two H200 GPUs, our model achieves an average of 1,338 TPS at batch size 1. Code is available at: this https URL 

---
# SeLaR: Selective Latent Reasoning in Large Language Models 

**Authors**: Renyu Fu, Guibo Luo  

**Link**: [PDF](https://arxiv.org/pdf/2604.08299)  

**Abstract**: Chain-of-Thought (CoT) has become a cornerstone of reasoning in large language models, yet its effectiveness is constrained by the limited expressiveness of discrete token sampling. Recent latent reasoning approaches attempt to alleviate this limitation by replacing discrete tokens with soft embeddings (probability-weighted mixtures of token embeddings) or hidden states, but they commonly suffer from two issues: (1) global activation injects perturbations into high-confidence steps, impairing reasoning stability; and (2) soft embeddings quickly collapse toward the highest-probability token, limiting exploration of alternative trajectories. To address these challenges, we propose SeLaR (Selective Latent Reasoning), a lightweight and training-free framework. SeLaR introduces an entropy-gated mechanism that activates soft embeddings only at low-confidence steps, while preserving discrete decoding at high-confidence steps. Additionally, we propose an entropy-aware contrastive regularization that pushes soft embeddings away from the dominant (highest-probability) token's direction, encouraging sustained exploration of multiple latent reasoning paths. Experiments on five reasoning benchmarks demonstrate that SeLaR consistently outperforms standard CoT and state-of-the-art training-free methods. 

---
# Can Vision Language Models Judge Action Quality? An Empirical Evaluation 

**Authors**: Miguel Monte e Freitas, Rui Henriques, Ricardo Rei, Pedro Henrique Martins  

**Link**: [PDF](https://arxiv.org/pdf/2604.08294)  

**Abstract**: Action Quality Assessment (AQA) has broad applications in physical therapy, sports coaching, and competitive judging. Although Vision Language Models (VLMs) hold considerable promise for AQA, their actual performance in this domain remains largely uncharacterised. We present a comprehensive evaluation of state-of-the-art VLMs across activity domains (e.g. fitness, figure skating, diving), tasks, representations, and prompting strategies. Baseline results reveal that Gemini 3.1 Pro, Qwen3-VL and InternVL3.5 models perform only marginally above random chance, and although strategies such as incorporation of skeleton information, grounding instructions, reasoning structures and in-context learning lead to isolated gains, none is consistently effective. Analysis of prediction distributions uncovers two systematic biases: a tendency to predict correct execution regardless of visual evidence, and a sensitivity to superficial linguistic framing. Reformulating tasks contrastively to mitigate these biases yields minimal improvement, suggesting that the models' limitations go beyond these biases, pointing to a fundamental difficulty with fine-grained movement quality assessment. Our findings establish a rigorous baseline for future VLM-based AQA research and provide an actionable outline for failure modes requiring mitigation prior to reliable real-world deployment. 

---
# CIAO - Code In Architecture Out - Automated Software Architecture Documentation with Large Language Models 

**Authors**: Marco De Luca, Tiziano Santilli, Domenico Amalfitano, Anna Rita Fasolino, Patrizio Pelliccione  

**Link**: [PDF](https://arxiv.org/pdf/2604.08293)  

**Abstract**: Software architecture documentation is essential for system comprehension, yet it is often unavailable or incomplete. While recent LLM-based techniques can generate documentation from code, they typically address local artifacts rather than producing coherent, system-level architectural descriptions. This paper presents a structured process for automatically generating system-level architectural documentation directly from GitHub repositories using Large Language Models. The process, called CIAO (Code In Architecture Out), defines an LLM-based workflow that takes a repository as input and produces system-level architectural documentation following a template derived from ISO/IEC/IEEE 42010, SEI Views \& Beyond, and the C4 model. The resulting documentation can be directly added to the target repository. We evaluated the process through a study with 22 developers, each reviewing the documentation generated for a repository they had contributed to. The evaluation shows that developers generally perceive the produced documentation as valuable, comprehensible, and broadly accurate with respect to the source code, while also highlighting limitations in diagram quality, high-level context modeling, and deployment views. We also assessed the operational cost of the process, finding that generating a complete architectural document requires only a few minutes and is inexpensive to run. Overall, the results indicate that a structured, standards-oriented approach can effectively guide LLMs in producing system-level architectural documentation that is both usable and cost-effective. 

---
# Distributed Multi-Layer Editing for Rule-Level Knowledge in Large Language Models 

**Authors**: Yating Wang, Wenting Zhao, Yaqi Zhao, Yongshun Gong, Yilong Yin, Haoliang Sun  

**Link**: [PDF](https://arxiv.org/pdf/2604.08284)  

**Abstract**: Large language models store not only isolated facts but also rules that support reasoning across symbolic expressions, natural language explanations, and concrete instances. Yet most model editing methods are built for fact-level knowledge, assuming that a target edit can be achieved through a localized intervention. This assumption does not hold for rule-level knowledge, where a single rule must remain consistent across multiple interdependent forms. We investigate this problem through a mechanistic study of rule-level knowledge editing. To support this study, we extend the RuleEdit benchmark from 80 to 200 manually verified rules spanning mathematics and physics. Fine-grained causal tracing reveals a form-specific organization of rule knowledge in transformer layers: formulas and descriptions are concentrated in earlier layers, while instances are more associated with middle layers. These results suggest that rule knowledge is not uniformly localized, and therefore cannot be reliably edited by a single-layer or contiguous-block intervention. Based on this insight, we propose Distributed Multi-Layer Editing (DMLE), which applies a shared early-layer update to formulas and descriptions and a separate middle-layer update to instances. While remaining competitive on standard editing metrics, DMLE achieves substantially stronger rule-level editing performance. On average, it improves instance portability and rule understanding by 13.91 and 50.19 percentage points, respectively, over the strongest baseline across GPT-J-6B, Qwen2.5-7B, Qwen2-7B, and LLaMA-3-8B. The code is available at this https URL. 

---
# QARIMA: A Quantum Approach To Classical Time Series Analysis 

**Authors**: Nishikanta Mohanty, Bikash K. Behera, Badshah Mukherjee, Pravat Dash  

**Link**: [PDF](https://arxiv.org/pdf/2604.08277)  

**Abstract**: We present a quantum-inspired ARIMA methodology that integrates quantum-assisted lag discovery with \emph{fixed-configuration} variational quantum circuits (VQCs) for parameter estimation and weak-lag refinement. Differencing and candidate lags are identified via swap-test-driven quantum autocorrelation (QACF) and quantum partial autocorrelation (QPACF), with a delayed-matrix construction that aligns quantum projections to time-domain regressors, followed by standard information-criterion parsimony. Given the screened orders $(p,d,q)$, we retain a fixed VQC ansatz, optimizer, and training budget, preventing hyperparameter leakage, and deploy the circuit in two estimation roles: VQC-AR for autoregressive coefficients and VQC-MA for moving-average coefficients. Between screening and estimation, a lightweight VQC weak-lag refinement re-weights or prunes screened AR lags without altering $(p,d,q)$. Across environmental and industrial datasets, we perform rolling-origin evaluations against automated classical ARIMA, reporting out-of-sample mean squared error (MSE), mean absolute percentage error (MAPE), and Diebold--Mariano tests on MSE and MAE. Empirically, the seven quantum contributions -- (1) differencing selection, (2) QACF, (3) QPACF, (4) swap-test primitives with delayed-matrix construction, (5) VQC-AR, (6) VQC weak-lag refinement, and (7) VQC-MA -- collectively reduce meta-optimization overhead and make explicit where quantum effects enter order discovery, lag refinement, and AR/MA parameter estimation. 

---
# DBMF: A Dual-Branch Multimodal Framework for Out-of-Distribution Detection 

**Authors**: Jiangbei Yue, Sharib Ali  

**Link**: [PDF](https://arxiv.org/pdf/2604.08261)  

**Abstract**: The complex and dynamic real-world clinical environment demands reliable deep learning (DL) systems. Out-of-distribution (OOD) detection plays a critical role in enhancing the reliability and generalizability of DL models when encountering data that deviate from the training distribution, such as unseen disease cases. However, existing OOD detection methods typically rely either on a single visual modality or solely on image-text matching, failing to fully leverage multimodal information. To overcome the challenge, we propose a novel dual-branch multimodal framework by introducing a text-image branch and a vision branch. Our framework fully exploits multimodal representations to identify OOD samples through these two complementary branches. After training, we compute scores from the text-image branch ($S_t$) and vision branch ($S_v$), and integrate them to obtain the final OOD score $S$ that is compared with a threshold for OOD detection. Comprehensive experiments on publicly available endoscopic image datasets demonstrate that our proposed framework is robust across diverse backbones and improves state-of-the-art performance in OOD detection by up to 24.84% 

---
# Behavior-Aware Item Modeling via Dynamic Procedural Solution Representations for Knowledge Tracing 

**Authors**: Jun Seo, Sangwon Ryu, Heejin Do, Hyounghun Kim, Gary Geunbae Lee  

**Link**: [PDF](https://arxiv.org/pdf/2604.08260)  

**Abstract**: Knowledge Tracing (KT) aims to predict learners' future performance from past interactions. While recent KT approaches have improved via learning item representations aligned with Knowledge Components, they overlook the procedural dynamics of problem solving. We propose Behavior-Aware Item Modeling (BAIM), a framework that enriches item representations by integrating dynamic procedural solution information. BAIM leverages a reasoning language model to decompose each item's solution into four problem-solving stages (i.e., understand, plan, carry out, and look back), pedagogically grounded in Polya's framework. Specifically, it derives stage-level representations from per-stage embedding trajectories, capturing latent signals beyond surface features. To reflect learner heterogeneity, BAIM adaptively routes these stage-wise representations, introducing a context-conditioned mechanism within a KT backbone, allowing different procedural stages to be emphasized for different learners. Experiments on XES3G5M and NIPS34 show that BAIM consistently outperforms strong pretraining-based baselines, achieving particularly large gains under repeated learner interactions. 

---
# HyperMem: Hypergraph Memory for Long-Term Conversations 

**Authors**: Juwei Yue, Chuanrui Hu, Jiawei Sheng, Zuyi Zhou, Wenyuan Zhang, Tingwen Liu, Li Guo, Yafeng Deng  

**Link**: [PDF](https://arxiv.org/pdf/2604.08256)  

**Abstract**: Long-term memory is essential for conversational agents to maintain coherence, track persistent tasks, and provide personalized interactions across extended dialogues. However, existing approaches as Retrieval-Augmented Generation (RAG) and graph-based memory mostly rely on pairwise relations, which can hardly capture high-order associations, i.e., joint dependencies among multiple elements, causing fragmented retrieval. To this end, we propose HyperMem, a hypergraph-based hierarchical memory architecture that explicitly models such associations using hyperedges. Particularly, HyperMem structures memory into three levels: topics, episodes, and facts, and groups related episodes and their facts via hyperedges, unifying scattered content into coherent units. Leveraging this structure, we design a hybrid lexical-semantic index and a coarse-to-fine retrieval strategy, supporting accurate and efficient retrieval of high-order associations. Experiments on the LoCoMo benchmark show that HyperMem achieves state-of-the-art performance with 92.73% LLM-as-a-judge accuracy, demonstrating the effectiveness of HyperMem for long-term conversations. 

---
# EditCaption: Human-Aligned Instruction Synthesis for Image Editing via Supervised Fine-Tuning and Direct Preference Optimization 

**Authors**: Xiangyuan Wang, Honghao Cai, Yunhao Bai, Tianze Zhou, Haohua Chen, Yao Hu, Xu Tang, Yibo Chen, Wei Zhu  

**Link**: [PDF](https://arxiv.org/pdf/2604.08213)  

**Abstract**: High-quality training triplets (source-target image pairs with precise editing instructions) are a critical bottleneck for scaling instruction-guided image editing models. Vision-language models (VLMs) are widely used for automated instruction synthesis, but we identify three systematic failure modes in image-pair settings: orientation inconsistency (e.g., left/right confusion), viewpoint ambiguity, and insufficient fine-grained attribute description. Human evaluation shows that over 47% of instructions from strong baseline VLMs contain critical errors unusable for downstream training. We propose EditCaption, a scalable two-stage post-training pipeline for VLM-based instruction synthesis. Stage 1 builds a 100K supervised fine-tuning (SFT) dataset by combining GLM automatic annotation, EditScore-based filtering, and human refinement for spatial, directional, and attribute-level accuracy. Stage 2 collects 10K human preference pairs targeting the three failure modes and applies direct preference optimization (DPO) for alignment beyond SFT alone. On Eval-400, ByteMorph-Bench, and HQ-Edit, fine-tuned Qwen3-VL models outperform open-source baselines; the 235B model reaches 4.712 on Eval-400 (vs. Gemini-3-Pro 4.706, GPT-4.1 4.220, Kimi-K2.5 4.111) and 4.588 on ByteMorph-Bench (vs. Gemini-3-Pro 4.522, GPT-4.1 3.412). Human evaluation shows critical errors falling from 47.75% to 23% and correctness rising from 41.75% to 66%. The work offers a practical path to scalable, human-aligned instruction synthesis for image editing data. 

---
# MedVR: Annotation-Free Medical Visual Reasoning via Agentic Reinforcement Learning 

**Authors**: Zheng Jiang, Heng Guo, Chengyu Fang, Changchen Xiao, Xinyang Hu, Lifeng Sun, Minfeng Xu  

**Link**: [PDF](https://arxiv.org/pdf/2604.08203)  

**Abstract**: Medical Vision-Language Models (VLMs) hold immense promise for complex clinical tasks, but their reasoning capabilities are often constrained by text-only paradigms that fail to ground inferences in visual evidence. This limitation not only curtails performance on tasks requiring fine-grained visual analysis but also introduces risks of visual hallucination in safety-critical applications. Thus, we introduce MedVR, a novel reinforcement learning framework that enables annotation-free visual reasoning for medical VLMs. Its core innovation lies in two synergistic mechanisms: Entropy-guided Visual Regrounding (EVR) uses model uncertainty to direct exploration, while Consensus-based Credit Assignment (CCA) distills pseudo-supervision from rollout agreement. Without any human annotations for intermediate steps, MedVR achieves state-of-the-art performance on diverse public medical VQA benchmarks, significantly outperforming existing models. By learning to reason directly with visual evidence, MedVR promotes the robustness and transparency essential for accelerating the clinical deployment of medical AI. 

---
# AT-ADD: All-Type Audio Deepfake Detection Challenge Evaluation Plan 

**Authors**: Yuankun Xie, Haonan Cheng, Jiayi Zhou, Xiaoxuan Guo, Tao Wang, Jian Liu, Weiqiang Wang, Ruibo Fu, Xiaopeng Wang, Hengyan Huang, Xiaoying Huang, Long Ye, Guangtao Zhai  

**Link**: [PDF](https://arxiv.org/pdf/2604.08184)  

**Abstract**: The rapid advancement of Audio Large Language Models (ALLMs) has enabled cost-effective, high-fidelity generation and manipulation of both speech and non-speech audio, including sound effects, singing voices, and music. While these capabilities foster creativity and content production, they also introduce significant security and trust challenges, as realistic audio deepfakes can now be generated and disseminated at scale. Existing audio deepfake detection (ADD) countermeasures (CMs) and benchmarks, however, remain largely speech-centric, often relying on speech-specific artifacts and exhibiting limited robustness to real-world distortions, as well as restricted generalization to heterogeneous audio types and emerging spoofing techniques. To address these gaps, we propose the All-Type Audio Deepfake Detection (AT-ADD) Grand Challenge for ACM Multimedia 2026, designed to bridge controlled academic evaluation with practical multimedia forensics. AT-ADD comprises two tracks: (1) Robust Speech Deepfake Detection, which evaluates detectors under real-world scenarios and against unseen, state-of-the-art speech generation methods; and (2) All-Type Audio Deepfake Detection, which extends detection beyond speech to diverse, unknown audio types and promotes type-agnostic generalization across speech, sound, singing, and music. By providing standardized datasets, rigorous evaluation protocols, and reproducible baselines, AT-ADD aims to accelerate the development of robust and generalizable audio forensic technologies, supporting secure communication, reliable media verification, and responsible governance in an era of pervasive synthetic audio. 

---
# OceanMAE: A Foundation Model for Ocean Remote Sensing 

**Authors**: Viola-Joanna Stamer, Panagiotis Agrafiotis, Behnood Rasti, Begüm Demir  

**Link**: [PDF](https://arxiv.org/pdf/2604.08171)  

**Abstract**: Accurate ocean mapping is essential for applications such as bathymetry estimation, seabed characterization, marine litter detection, and ecosystem monitoring. However, ocean remote sensing (RS) remains constrained by limited labeled data and by the reduced transferability of models pre-trained mainly on land-dominated Earth observation imagery. In this paper, we propose OceanMAE, an ocean-specific masked autoencoder that extends standard MAE pre-training by integrating multispectral Sentinel-2 observations with physically meaningful ocean descriptors during self-supervised learning. By incorporating these auxiliary ocean features, OceanMAE is designed to learn more informative and ocean-aware latent representations from large- scale unlabeled data. To transfer these representations to downstream applications, we further employ a modified UNet-based framework for marine segmentation and bathymetry estimation. Pre-trained on the Hydro dataset, OceanMAE is evaluated on MADOS and MARIDA for marine pollutant and debris segmentation, and on MagicBathyNet for bathymetry regression. The experiments show that OceanMAE yields the strongest gains on marine segmentation, while bathymetry benefits are competitive and task-dependent. In addition, an ablation against a standard MAE on MARIDA indicates that incorporating auxiliary ocean descriptors during pre-training improves downstream segmentation quality. These findings highlight the value of physically informed and domain-aligned self-supervised pre- training for ocean RS. Code and weights are publicly available at this https URL. 

---
# ViVa: A Video-Generative Value Model for Robot Reinforcement Learning 

**Authors**: Jindi Lv, Hao Li, Jie Li, Yifei Nie, Fankun Kong, Yang Wang, Xiaofeng Wang, Zheng Zhu, Chaojun Ni, Qiuping Deng, Hengtao Li, Jiancheng Lv, Guan Huang  

**Link**: [PDF](https://arxiv.org/pdf/2604.08168)  

**Abstract**: Vision-language-action (VLA) models have advanced robot manipulation through large-scale pretraining, but real-world deployment remains challenging due to partial observability and delayed feedback. Reinforcement learning addresses this via value functions, which assess task progress and guide policy improvement. However, existing value models built on vision-language models (VLMs) struggle to capture temporal dynamics, undermining reliable value estimation in long-horizon tasks. In this paper, we propose ViVa, a video-generative value model that repurposes a pretrained video generator for value estimation. Taking the current observation and robot proprioception as input, ViVa jointly predicts future proprioception and a scalar value for the current state. By leveraging the spatiotemporal priors of a pretrained video generator, our approach grounds value estimation in anticipated embodiment dynamics, moving beyond static snapshots to intrinsically couple value with foresight. Integrated into RECAP, ViVa delivers substantial improvements on real-world box assembly. Qualitative analysis across all three tasks confirms that ViVa produces more reliable value signals, accurately reflecting task progress. By leveraging spatiotemporal priors from video corpora, ViVa also generalizes to novel objects, highlighting the promise of video-generative models for value estimation. 

---
# Face-D(^2)CL: Multi-Domain Synergistic Representation with Dual Continual Learning for Facial DeepFake Detection 

**Authors**: Yushuo Zhang, Yu Cheng, Yongkang Hu, Jiuan Zhou, Jiawei Chen, Yuan Xie, Zhaoxia Yin  

**Link**: [PDF](https://arxiv.org/pdf/2604.08159)  

**Abstract**: The rapid advancement of facial forgery techniques poses severe threats to public trust and information security, making facial DeepFake detection a critical research priority. Continual learning provides an effective approach to adapt facial DeepFake detection models to evolving forgery patterns. However, existing methods face two key bottlenecks in real-world continual learning scenarios: insufficient feature representation and catastrophic forgetting. To address these issues, we propose Face-D(^2)CL, a framework for facial DeepFake detection. It leverages multi-domain synergistic representation to fuse spatial and frequency-domain features for the comprehensive capture of diverse forgery traces, and employs a dual continual learning mechanism that combines Elastic Weight Consolidation (EWC), which distinguishes parameter importance for real versus fake samples, and Orthogonal Gradient Constraint (OGC), which ensures updates to task-specific adapters do not interfere with previously learned knowledge. This synergy enables the model to achieve a dynamic balance between robust anti-forgetting capabilities and agile adaptability to emerging facial forgery paradigms, all without relying on historical data replay. Extensive experiments demonstrate that our method surpasses current SOTA approaches in both stability and plasticity, achieving 60.7% relative reduction in average detection error rate, respectively. On unseen forgery domains, it further improves the average detection AUC by 7.9% compared to the current SOTA method. 

---
# Multimodal Reasoning with LLM for Encrypted Traffic Interpretation: A Benchmark 

**Authors**: Longgang Zhang, Xiaowei Fu, Fuxiang Huang, Lei Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2604.08140)  

**Abstract**: Network traffic, as a key media format, is crucial for ensuring security and communications in modern internet infrastructure. While existing methods offer excellent performance, they face two key bottlenecks: (1) They fail to capture multidimensional semantics beyond unimodal sequence patterns. (2) Their black box property, i.e., providing only category labels, lacks an auditable reasoning process. We identify a key factor that existing network traffic datasets are primarily designed for classification and inherently lack rich semantic annotations, failing to generate human-readable evidence report. To address data scarcity, this paper proposes a Byte-Grounded Traffic Description (BGTD) benchmark for the first time, combining raw bytes with structured expert annotations. BGTD provides necessary behavioral features and verifiable chains of evidence for multimodal reasoning towards explainable encrypted traffic interpretation. Built upon BGTD, this paper proposes an end-to-end traffic-language representation framework (mmTraffic), a multimodal reasoning architecture bridging physical traffic encoding and semantic interpretation. In order to alleviate modality interference and generative hallucinations, mmTraffic adopts a jointly-optimized perception-cognition architecture. By incorporating a perception-centered traffic encoder and a cognition-centered LLM generator, mmTraffic achieves refined traffic interpretation with guaranteed category prediction. Extensive experiments demonstrate that mmTraffic autonomously generates high-fidelity, human-readable, and evidence-grounded traffic interpretation reports, while maintaining highly competitive classification accuracy comparing to specialized unimodal model (e.g., NetMamba). The source code is available at this https URL 

---
# Alloc-MoE: Budget-Aware Expert Activation Allocation for Efficient Mixture-of-Experts Inference 

**Authors**: Baihui Liu, Kaiyuan Tian, Wei Wang, Zhaoning Zhang, Linbo Qiao, Dongsheng Li  

**Link**: [PDF](https://arxiv.org/pdf/2604.08133)  

**Abstract**: Mixture-of-Experts (MoE) has become a dominant architecture for scaling large language models due to their sparse activation mechanism. However, the substantial number of expert activations creates a critical latency bottleneck during inference, especially in resource-constrained deployment scenarios. Existing approaches that reduce expert activations potentially lead to severe model performance degradation. In this work, we introduce the concept of \emph{activation budget} as a constraint on the number of expert activations and propose Alloc-MoE, a unified framework that optimizes budget allocation coordinately at both the layer and token levels to minimize performance degradation. At the layer level, we introduce Alloc-L, which leverages sensitivity profiling and dynamic programming to determine the optimal allocation of expert activations across layers. At the token level, we propose Alloc-T, which dynamically redistributes activations based on routing scores, optimizing budget allocation without increasing latency. Extensive experiments across multiple MoE models demonstrate that Alloc-MoE maintains model performance under a constrained activation budget. Especially, Alloc-MoE achieves $1.15\times$ prefill and $1.34\times$ decode speedups on DeepSeek-V2-Lite at half of the original budget. 

---
# LegoDiffusion: Micro-Serving Text-to-Image Diffusion Workflows 

**Authors**: Lingyun Yang, Suyi Li, Tianyu Feng, Xiaoxiao Jiang, Zhipeng Di, Weiyi Lu, Kan Liu, Yinghao Yu, Tao Lan, Guodong Yang, Lin Qu, Liping Zhang, Wei Wang  

**Link**: [PDF](https://arxiv.org/pdf/2604.08123)  

**Abstract**: Text-to-image generation executes a diffusion workflow comprising multiple models centered on a base diffusion model. Existing serving systems treat each workflow as an opaque monolith, provisioning, placing, and scaling all constituent models together, which obscures internal dataflow, prevents model sharing, and enforces coarse-grained resource management. In this paper, we make a case for micro-serving diffusion workflows with LegoDiffusion, a system that decomposes a workflow into loosely coupled model-execution nodes that can be independently managed and scheduled. By explicitly managing individual model inference, LegoDiffusion unlocks cluster-scale optimizations, including per-model scaling, model sharing, and adaptive model parallelism. Collectively, LegoDiffusion outperforms existing diffusion workflow serving systems, sustaining up to 3x higher request rates and tolerating up to 8x higher burst traffic. 

---
# Uni-ViGU: Towards Unified Video Generation and Understanding via A Diffusion-Based Video Generator 

**Authors**: Luozheng Qin, Jia Gong, Qian Qiao, Tianjiao Li, Li Xu, Haoyu Pan, Chao Qu, Zhiyu Tan, Hao Li  

**Link**: [PDF](https://arxiv.org/pdf/2604.08121)  

**Abstract**: Unified multimodal models integrating visual understanding and generation face a fundamental challenge: visual generation incurs substantially higher computational costs than understanding, particularly for video. This imbalance motivates us to invert the conventional paradigm: rather than extending understanding-centric MLLMs to support generation, we propose Uni-ViGU, a framework that unifies video generation and understanding by extending a video generator as the foundation. We introduce a unified flow method that performs continuous flow matching for video and discrete flow matching for text within a single process, enabling coherent multimodal generation. We further propose a modality-driven MoE-based framework that augments Transformer blocks with lightweight layers for text generation while preserving generative priors. To repurpose generation knowledge for understanding, we design a bidirectional training mechanism with two stages: Knowledge Recall reconstructs input prompts to leverage learned text-video correspondences, while Capability Refinement fine-tunes on detailed captions to establish discriminative shared representations. Experiments demonstrate that Uni-ViGU achieves competitive performance on both video generation and understanding, validating generation-centric architectures as a scalable path toward unified multimodal intelligence. Project Page and Code: this https URL. 

---
# Small Vision-Language Models are Smart Compressors for Long Video Understanding 

**Authors**: Junjie Fei, Jun Chen, Zechun Liu, Yunyang Xiong, Chong Zhou, Wei Wen, Junlin Han, Mingchen Zhuge, Saksham Suri, Qi Qian, Shuming Liu, Lemeng Wu, Raghuraman Krishnamoorthi, Vikas Chandra, Mohamed Elhoseiny, Chenchen Zhu  

**Link**: [PDF](https://arxiv.org/pdf/2604.08120)  

**Abstract**: Adapting Multimodal Large Language Models (MLLMs) for hour-long videos is bottlenecked by context limits. Dense visual streams saturate token budgets and exacerbate the lost-in-the-middle phenomenon. Existing heuristics, like sparse sampling or uniform pooling, blindly sacrifice fidelity by discarding decisive moments and wasting bandwidth on irrelevant backgrounds. We propose Tempo, an efficient query-aware framework compressing long videos for downstream understanding. Tempo leverages a Small Vision-Language Model (SVLM) as a local temporal compressor, casting token reduction as an early cross-modal distillation process to generate compact, intent-aligned representations in a single forward pass. To enforce strict budgets without breaking causality, we introduce Adaptive Token Allocation (ATA). Exploiting the SVLM's zero-shot relevance prior and semantic front-loading, ATA acts as a training-free $O(1)$ dynamic router. It allocates dense bandwidth to query-critical segments while compressing redundancies into minimal temporal anchors to maintain the global storyline. Extensive experiments show our 6B architecture achieves state-of-the-art performance with aggressive dynamic compression (0.5-16 tokens/frame). On the extreme-long LVBench (4101s), Tempo scores 52.3 under a strict 8K visual budget, outperforming GPT-4o and Gemini 1.5 Pro. Scaling to 2048 frames reaches 53.7. Crucially, Tempo compresses hour-long videos substantially below theoretical limits, proving true long-form video understanding relies on intent-driven efficiency rather than greedily padded context windows. 

---
# TADP-RME: A Trust-Adaptive Differential Privacy Framework for Enhancing Reliability of Data-Driven Systems 

**Authors**: Labani Halder, Payel Sadhukhan, Sarbani Palit  

**Link**: [PDF](https://arxiv.org/pdf/2604.08113)  

**Abstract**: Ensuring reliability in adversarial settings necessitates treating privacy as a foundational component of data-driven systems. While differential privacy and cryptographic protocols offer strong guarantees, existing schemes rely on a fixed privacy budget, leading to a rigid utility-privacy trade-off that fails under heterogeneous user trust. Moreover, noise-only differential privacy preserves geometric structure, which inference attacks exploit, causing privacy leakage.
We propose TADP-RME (Trust-Adaptive Differential Privacy with Reverse Manifold Embedding), a framework that enhances reliability under varying levels of user trust. It introduces an inverse trust score in the range [0,1] to adaptively modulate the privacy budget, enabling smooth transitions between utility and privacy. Additionally, Reverse Manifold Embedding applies a nonlinear transformation to disrupt local geometric relationships while preserving formal differential privacy guarantees through post-processing.
Theoretical and empirical results demonstrate improved privacy-utility trade-offs, reducing attack success rates by up to 3.1 percent without significant utility degradation. The framework consistently outperforms existing methods against inference attacks, providing a unified approach for reliable learning in adversarial environments. 

---
# OV-Stitcher: A Global Context-Aware Framework for Training-Free Open-Vocabulary Semantic Segmentation 

**Authors**: Seungjae Moon, Seunghyun Oh, Youngmin Ro  

**Link**: [PDF](https://arxiv.org/pdf/2604.08110)  

**Abstract**: Training-free open-vocabulary semantic segmentation(TF-OVSS) has recently attracted attention for its ability to perform dense prediction by leveraging the pretrained knowledge of large vision and vision-language models, without requiring additional training. However, due to the limited input resolution of these pretrained encoders, existing TF-OVSS methods commonly adopt a sliding-window strategy that processes cropped sub-images independently. While effective for managing high-resolution inputs, this approach prevents global attention over the full image, leading to fragmented feature representations and limited contextual reasoning. We propose OV-Stitcher, a training-free framework that addresses this limitation by stitching fragmented sub-image features directly within the final encoder block. By reconstructing attention representations from fragmented sub-image features, OV-Stitcher enables global attention within the final encoder block, producing coherent context aggregation and spatially consistent, semantically aligned segmentation maps. Extensive evaluations across eight benchmarks demonstrate that OV-Stitcher establishes a scalable and effective solution for open-vocabulary segmentation, achieving a notable improvement in mean Intersection over Union(mIoU) from 48.7 to 50.7 compared with prior training-free baselines. 

---
# AtlasOCR: Building the First Open-Source Darija OCR Model with Vision Language Models 

**Authors**: Imane Momayiz, Soufiane Ait Elaouad, Abdeljalil Elmajjodi, Haitame Bouanane  

**Link**: [PDF](https://arxiv.org/pdf/2604.08070)  

**Abstract**: Darija, the Moroccan Arabic dialect, is rich in visual content yet lacks specialized Optical Character Recognition (OCR) tools. This paper introduces AtlasOCR, the first open-source Darija OCR model built by fine-tuning a 3B parameter Vision Language Model (VLM). We detail our comprehensive approach, from curating a unique Darija-specific dataset leveraging both synthetic generation with our OCRSmith library and carefully sourced real-world data, to implementing efficient fine-tuning strategies. We utilize QLoRA and Unsloth for parameter-efficient training of Qwen2.5-VL 3B and present comprehensive ablation studies optimizing key hyperparameters. Our evaluation on the newly curated AtlasOCRBench and the established KITAB-Bench demonstrates state-of-the-art performance, challenging larger models and highlighting AtlasOCR's robustness and generalization capabilities for both Darija and standard Arabic OCR tasks. 

---
# From Gaze to Guidance: Interpreting and Adapting to Users' Cognitive Needs with Multimodal Gaze-Aware AI Assistants 

**Authors**: Valdemar Danry, Javier Hernandez, Andrew Wilson, Pattie Maes, Judith Amores  

**Link**: [PDF](https://arxiv.org/pdf/2604.08062)  

**Abstract**: Current LLM assistants are powerful at answering questions, but they have limited access to the behavioral context that reveals when and where a user is struggling. We present a gaze-grounded multimodal LLM assistant that uses egocentric video with gaze overlays to identify likely points of difficulty and target follow-up retrospective assistance. We instantiate this vision in a controlled study (n=36) comparing the gaze-aware AI assistant to a text-only LLM assistant. Compared to a conventional LLM assistant, the gaze-aware assistant was rated as significantly more accurate and personalized in its assessments of users' reading behavior and significantly improved people's ability to recall information. Users spoke significantly fewer words with the gaze-aware assistant, indicating more efficient interactions. Qualitative results underscored both perceived benefits in comprehension and challenges when interpretations of gaze behaviors were inaccurate. Our findings suggest that gaze-aware LLM assistants can reason about cognitive needs to improve cognitive outcomes of users. 

---
# Governed Capability Evolution for Embodied Agents: Safe Upgrade, Compatibility Checking, and Runtime Rollback for Embodied Capability Modules 

**Authors**: Xue Qin, Simin Luan, John See, Cong Yang, Zhijun Li  

**Link**: [PDF](https://arxiv.org/pdf/2604.08059)  

**Abstract**: Embodied agents are increasingly expected to improve over time by updating their executable capabilities rather than rewriting the agent itself. Prior work has separately studied modular capability packaging, capability evolution, and runtime governance. However, a key systems problem remains underexplored: once an embodied capability module evolves into a new version, how can the hosting system deploy it safely without breaking policy constraints, execution assumptions, or recovery guarantees?
We formulate governed capability evolution as a first-class systems problem for embodied agents. We propose a lifecycle-aware upgrade framework in which every new capability version is treated as a governed deployment candidate rather than an immediately executable replacement. The framework introduces four upgrade compatibility checks -- interface, policy, behavioral, and recovery -- and organizes them into a staged runtime pipeline comprising candidate validation, sandbox evaluation, shadow deployment, gated activation, online monitoring, and rollback.
We evaluate over 6 rounds of capability upgrade with 15 random seeds. Naive upgrade achieves 72.9% task success but drives unsafe activation to 60% by the final round; governed upgrade retains comparable success (67.4%) while maintaining zero unsafe activations across all rounds (Wilcoxon p=0.003). Shadow deployment reveals 40% of regressions invisible to sandbox evaluation alone, and rollback succeeds in 79.8% of post-activation drift scenarios. 

---
# 3DrawAgent: Teaching LLM to Draw in 3D with Early Contrastive Experience 

**Authors**: Hongcan Xiao, Xinyue Xiao, Yilin Wang, Yue Zhang, Yonggang Qi  

**Link**: [PDF](https://arxiv.org/pdf/2604.08042)  

**Abstract**: Sketching in 3D space enables expressive reasoning about shape, structure, and spatial relationships, yet generating 3D sketches through natural language remains a major challenge. In this work, we introduce 3DrawAgent, a training-free, language-driven framework for 3D sketch generation that leverages large language models (LLMs) to sequentially draw 3D Bezier curves under geometric feedback. Unlike prior 2D sketch agents, our method introduces a relative experience optimization strategy that adapts the recently proposed Group Reward Policy Optimization (GRPO) paradigm. Instead of relying on explicit ground-truth supervision, we construct pairwise comparisons among generated sketches, with each pair consisting of a relatively better and a worse result based on CLIP-based perceptual rewards and LLM-based fine-grained qualitative assessment. These experiences are then used to iteratively refine the prior knowledge of 3D drawing, enabling black-box reinforcement of the model's 3D awareness. This design allows our model to self-improve its spatial understanding and drawing quality without parameter updates. Experiments show that 3DrawAgent can generate complex and coherent 3D Bezier sketches from diverse textual prompts, exhibit emergent geometric reasoning, and generalize to novel shapes, establishing a new paradigm for advancing the field of training-free 3D sketch intelligence. 

---
# LINE: LLM-based Iterative Neuron Explanations for Vision Models 

**Authors**: Vladimir Zaigrajew, Michał Piechota, Gaspar Sekula, Przemysław Biecek  

**Link**: [PDF](https://arxiv.org/pdf/2604.08039)  

**Abstract**: Interpreting the concepts encoded by individual neurons in deep neural networks is a crucial step towards understanding their complex decision-making processes and ensuring AI safety. Despite recent progress in neuron labeling, existing methods often limit the search space to predefined concept vocabularies or produce overly specific descriptions that fail to capture higher-order, global concepts. We introduce LINE, a novel, training-free iterative approach tailored for open-vocabulary concept labeling in vision models. Operating in a strictly black-box setting, LINE leverages a large language model and a text-to-image generator to iteratively propose and refine concepts in a closed loop, guided by activation history. We demonstrate that LINE achieves state-of-the-art performance across multiple model architectures, yielding AUC improvements of up to 0.18 on ImageNet and 0.05 on Places365, while discovering, on average, 29% of new concepts missed by massive predefined vocabularies. Beyond identifying the top concept, LINE provides a complete generation history, which enables polysemanticity evaluation and produces supporting visual explanations that rival gradient-dependent activation maximization methods. 

---
# PrivFedTalk: Privacy-Aware Federated Diffusion with Identity-Stable Adapters for Personalized Talking-Head Generation 

**Authors**: Soumya Mazumdar, Vineet Kumar Rakesh, Tapas Samanta  

**Link**: [PDF](https://arxiv.org/pdf/2604.08037)  

**Abstract**: Talking-head generation has advanced rapidly with diffusion-based generative models, but training usually depends on centralized face-video and speech datasets, raising major privacy concerns. The problem is more acute for personalized talking-head generation, where identity-specific data are highly sensitive and often cannot be pooled across users or devices. PrivFedTalk is presented as a privacy-aware federated framework for personalized talking-head generation that combines conditional latent diffusion with parameter-efficient identity adaptation. A shared diffusion backbone is trained across clients, while each client learns lightweight LoRA identity adapters from local private audio-visual data, avoiding raw data sharing and reducing communication cost. To address heterogeneous client distributions, Identity-Stable Federated Aggregation (ISFA) weights client updates using privacy-safe scalar reliability signals computed from on-device identity consistency and temporal stability estimates. Temporal-Denoising Consistency (TDC) regularization is introduced to reduce inter-frame drift, flicker, and identity drift during federated denoising. To limit update-side privacy risk, secure aggregation and client-level differential privacy are applied to adapter updates. The implementation supports both low-memory GPU execution and multi-GPU client-parallel training on heterogeneous shared hardware. Comparative experiments on the present setup across multiple training and aggregation conditions with PrivFedTalk, FedAvg, and FedProx show stable federated optimization and successful end-to-end training and evaluation under constrained resources. The results support the feasibility of privacy-aware personalized talking-head training in federated environments, while suggesting that stronger component-wise, privacy-utility, and qualitative claims need further standardized evaluation. 

---
# From Universal to Individualized Actionability: Revisiting Personalization in Algorithmic Recourse 

**Authors**: Lena Marie Budde, Ayan Majumdar, Richard Uth, Markus Langer, Isabel Valera  

**Link**: [PDF](https://arxiv.org/pdf/2604.08030)  

**Abstract**: Algorithmic recourse aims to provide actionable recommendations that enable individuals to change unfavorable model outcomes, and prior work has extensively studied properties such as efficiency, robustness, and fairness. However, the role of personalization in recourse remains largely implicit and underexplored. While existing approaches incorporate elements of personalization through user interactions, they typically lack an explicit definition of personalization and do not systematically analyze its downstream effects on other recourse desiderata.
In this paper, we formalize personalization as individual actionability, characterized along two dimensions: hard constraints that specify which features are individually actionable, and soft, individualized constraints that capture preferences over action values and costs. We operationalize these dimensions within the causal algorithmic recourse framework, adopting a pre-hoc user-prompting approach in which individuals express preferences via rankings or scores prior to the generation of any recourse recommendation. Through extensive empirical evaluation, we investigate how personalization interacts with key recourse desiderata, including validity, cost, and plausibility. Our results highlight important trade-offs: individual actionability constraints, particularly hard ones, can substantially degrade the plausibility and validity of recourse recommendations across amortized and non-amortized approaches. Notably, we also find that incorporating individual actionability can reveal disparities in the cost and plausibility of recourse actions across socio-demographic groups. These findings underscore the need for principled definitions, careful operationalization, and rigorous evaluation of personalization in algorithmic recourse. 

---
# SearchAD: Large-Scale Rare Image Retrieval Dataset for Autonomous Driving 

**Authors**: Felix Embacher, Jonas Uhrig, Marius Cordts, Markus Enzweiler  

**Link**: [PDF](https://arxiv.org/pdf/2604.08008)  

**Abstract**: Retrieving rare and safety-critical driving scenarios from large-scale datasets is essential for building robust autonomous driving (AD) systems. As dataset sizes continue to grow, the key challenge shifts from collecting more data to efficiently identifying the most relevant samples. We introduce SearchAD, a large-scale rare image retrieval dataset for AD containing over 423k frames drawn from 11 established datasets. SearchAD provides high-quality manual annotations of more than 513k bounding boxes covering 90 rare categories. It specifically targets the needle-in-a-haystack problem of locating extremely rare classes, with some appearing fewer than 50 times across the entire dataset. Unlike existing benchmarks, which focused on instance-level retrieval, SearchAD emphasizes semantic image retrieval with a well-defined data split, enabling text-to-image and image-to-image retrieval, few-shot learning, and fine-tuning of multi-modal retrieval models. Comprehensive evaluations show that text-based methods outperform image-based ones due to stronger inherent semantic grounding. While models directly aligning spatial visual features with language achieve the best zero-shot results, and our fine-tuning baseline significantly improves performance, absolute retrieval capabilities remain unsatisfactory. With a held-out test set on a public benchmark server, SearchAD establishes the first large-scale dataset for retrieval-driven data curation and long-tail perception research in AD: this https URL 

---
# The ecosystem of machine learning competitions: Platforms, participants, and their impact on AI development 

**Authors**: Ioannis Nasios  

**Link**: [PDF](https://arxiv.org/pdf/2604.08001)  

**Abstract**: Machine learning competitions (MLCs) play a pivotal role in advancing artificial intelligence (AI) by fostering innovation, skill development, and practical problem-solving. This study provides a comprehensive analysis of major competition platforms such as Kaggle and Zindi, examining their workflows, evaluation methodologies, and reward structures. It further assesses competition quality, participant expertise, and global reach, with particular attention to demographic trends among top-performing competitors. By exploring the motivations of competition hosts, this paper underscores the significant role of MLCs in shaping AI development, promoting collaboration, and driving impactful technological progress. Furthermore, by combining literature synthesis with platform-level data analysis and practitioner insights a comprehensive understanding of the MLC ecosystem is provided.
Moreover, the paper demonstrates that MLCs function at the intersection of academic research and industrial application, fostering the exchange of knowledge, data, and practical methodologies across domains. Their strong ties to open-source communities further promote collaboration, reproducibility, and continuous innovation within the broader ML ecosystem. By shaping research priorities, informing industry standards, and enabling large-scale crowdsourced problem-solving, these competitions play a key role in the ongoing evolution of AI. The study provides insights relevant to researchers, practitioners, and competition organizers, and includes an examination of the future trajectory and sustained influence of MLCs on AI development. 

---
# Show Me the Infographic I Imagine: Intent-Aware Infographic Retrieval for Authoring Support 

**Authors**: Jing Xu, Jiarui Hu, Zhihao Shuai, Yiyun Chen, Weikai Yang  

**Link**: [PDF](https://arxiv.org/pdf/2604.07989)  

**Abstract**: While infographics have become a powerful medium for communicating data-driven stories, authoring them from scratch remains challenging, especially for novice users. Retrieving relevant exemplars from a large corpus can provide design inspiration and promote reuse, substantially lowering the barrier to infographic authoring. However, effective retrieval is difficult because users often express design intent in ambiguous natural language, while infographics embody rich and multi-faceted visual designs. As a result, keyword-based search often fails to capture design intent, and general-purpose vision-language retrieval models trained on natural images are ill-suited to the text-heavy, multi-component nature of infographics. To address these challenges, we develop an intent-aware infographic retrieval framework that better aligns user queries with infographic designs. We first conduct a formative study of how people describe infographics and derive an intent taxonomy spanning content and visual design facets. This taxonomy is then leveraged to enrich and refine free-form user queries, guiding the retrieval process with intent-specific cues. Building on the retrieved exemplars, users can adapt the designs to their own data with high-level edit intents, supported by an interactive agent that performs low-level adaptation. Both quantitative evaluations and user studies are conducted to demonstrate that our method improves retrieval quality over baseline methods while better supporting intent satisfaction and efficient infographic authoring. 

---
# LogAct: Enabling Agentic Reliability via Shared Logs 

**Authors**: Mahesh Balakrishnan, Ashwin Bharambe, Davide Testuggine, David Geraghty, David Mao, Vidhya Venkat, Ilya Mironov, Rithesh Baradi, Gayathri Aiyer, Victoria Dudin  

**Link**: [PDF](https://arxiv.org/pdf/2604.07988)  

**Abstract**: Agents are LLM-driven components that can mutate environments in powerful, arbitrary ways. Extracting guarantees for the execution of agents in production environments can be challenging due to asynchrony and failures. In this paper, we propose a new abstraction called LogAct, where each agent is a deconstructed state machine playing a shared log. In LogAct, agentic actions are visible in the shared log before they are executed; can be stopped prior to execution by pluggable, decoupled voters; and recovered consistently in the case of agent or environment failure. LogAct enables agentic introspection, allowing the agent to analyze its own execution history using LLM inference, which in turn enables semantic variants of recovery, health check, and optimization. In our evaluation, LogAct agents recover efficiently and correctly from failures; debug their own performance; optimize token usage in swarms; and stop all unwanted actions for a target model on a representative benchmark with just a 3% drop in benign utility. 

---
# A Decomposition Perspective to Long-context Reasoning for LLMs 

**Authors**: Yanling Xiao, Huaibing Xie, Guoliang Zhao, Shihan Dou, Shaolei Wang, Yiting Liu, Nantao Zheng, Cheng Zhang, Pluto Zhou, Zhisong Zhang, Lemao Liu  

**Link**: [PDF](https://arxiv.org/pdf/2604.07981)  

**Abstract**: Long-context reasoning is essential for complex real-world applications, yet remains a significant challenge for Large Language Models (LLMs). Despite the rapid evolution in long-context reasoning, current research often overlooks the internal complexity of the long-context reasoning task itself. In this paper, we move beyond this holistic view and decompose long-context reasoning into a set of fundamental atomic skills, and we then automatically synthesize a suite of pseudo datasets, each explicitly targeting a specific atomic skill. Our empirical analysis confirms that proficiency in these atomic skills is strongly correlated with general long-text reasoning performance. Building on this insight, we employ reinforcement learning on these pseudo datasets to sharpen the model's atomic skills, in the hope of boosting its general long-context reasoning ability. Extensive experiments across multiple benchmarks demonstrate the effectiveness of our approach: it outperforms a strong baseline by an average margin of 7.7\% (improving from 46.3\% to 54.0\%) across Loogle, Loong, LongBench-v2, BrowscompLong, Ruler-qa2, and MRCR. 

---
# AtomEval: Atomic Evaluation of Adversarial Claims in Fact Verification 

**Authors**: Hongyi Cen, Mingxin Wang, Yule Liu, Jingyi Zheng, Hanze Jia, Tan Tang, Yingcai Wu  

**Link**: [PDF](https://arxiv.org/pdf/2604.07967)  

**Abstract**: Adversarial claim rewriting is widely used to test fact-checking systems, but standard metrics fail to capture truth-conditional consistency and often label semantically corrupted rewrites as successful. We introduce AtomEval, a validity-aware evaluation framework that decomposes claims into subject-relation-object-modifier (SROM) atoms and scores adversarial rewrites with Atomic Validity Scoring (AVS), enabling detection of factual corruption beyond surface similarity. Experiments on the FEVER dataset across representative attack strategies and LLM generators show that AtomEval provides more reliable evaluation signals in our experiments. Using AtomEval, we further analyze LLM-based adversarial generators and observe that stronger models do not necessarily produce more effective adversarial claims under validity-aware evaluation, highlighting previously overlooked limitations in current adversarial evaluation practices. 

---
# DSCA: Dynamic Subspace Concept Alignment for Lifelong VLM Editing 

**Authors**: Gyanendra Das, Sai Satyam Jena  

**Link**: [PDF](https://arxiv.org/pdf/2604.07965)  

**Abstract**: Model editing aims to update knowledge to add new concepts and change relevant information without retraining. Lifelong editing is a challenging task, prone to disrupting previously learned concepts, especially for Vision Language Models (VLMs), because sequential edits can lead to degraded reasoning and cross modal misalignment. Existing VLM knowledge editing methods based on gated adapters, activation edits, and parameter merging techniques address catastrophic forgetting seen in full fine tuning; however, they still operate in the shared representation space of the VLM, where concepts are entangled, so edits interfere with other non relevant concepts. We hypothesize that this instability persists because current methods algorithmically control edits via optimization rather than structurally separating knowledge. We introduce Dynamic Subspace Concept Alignment (DSCA) which by design mitigates this limitation by decomposing the representation space into a set of orthogonal semantic subspaces and proposing edits only in those transformed spaces. These subspaces are obtained through incremental clustering and PCA on joint vision language representations. This process structurally isolates concepts, enabling precise, non interfering edits by turning isolation from a soft training objective into an architectural property. The surgical edits are guided by a multi term loss function for maintaining task fidelity, edit locality, and cross modal alignment. With the base model frozen, our method achieves 98 percent single edit success, remains over 95 percent after 1000 sequential edits, lowers hallucination by 3 to 5 percent, and achieves the best backward transfer (BWT) scores on continual instruction tuning benchmarks. Extensive experiments demonstrate DSCA state of the art stability and knowledge retention capability in continual lifelong editing across various datasets and benchmarks. 

---
# Rethinking Data Mixing from the Perspective of Large Language Models 

**Authors**: Yuanjian Xu, Tianze Sun, Changwei Xu, XinLong Zhao, Jianing Hao, Ran Chen, Yang Liu, Ruijie Xu, Stephen Chen, Guang Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2604.07963)  

**Abstract**: Data mixing strategy is essential for large language model (LLM) training. Empirical evidence shows that inappropriate strategies can significantly reduce generalization. Although recent methods have improved empirical performance, several fundamental questions remain open: what constitutes a domain, whether human and model perceptions of domains are aligned, and how domain weighting influences generalization. We address these questions by establishing formal connections between gradient dynamics and domain distributions, offering a theoretical framework that clarifies the role of domains in training dynamics. Building on this analysis, we introduce DoGraph, a reweighting framework that formulates data scheduling as a graph-constrained optimization problem. Extensive experiments on GPT-2 models of varying scales demonstrate that DoGraph consistently achieves competitive performance. 

---
# TOOLCAD: Exploring Tool-Using Large Language Models in Text-to-CAD Generation with Reinforcement Learning 

**Authors**: Yifei Gong, Xing Wu, Wenda Liu, Kang Tu  

**Link**: [PDF](https://arxiv.org/pdf/2604.07960)  

**Abstract**: Computer-Aided Design (CAD) is an expert-level task that relies on long-horizon reasoning and coherent modeling actions. Large Language Models (LLMs) have shown remarkable advancements in enabling language agents to tackle real-world tasks. Notably, there has been no investigation into how tool-using LLMs optimally interact with CAD engines, hindering the emergence of LLM-based agentic text-to-CAD modeling systems. We propose ToolCAD, a novel agentic CAD framework deploying LLMs as tool-using agents for text-to-CAD generation. Furthermore, we introduce an interactive CAD modeling gym to rollout reasoning and tool-augmented interaction trajectories with the CAD engine, incorporating hybrid feedback and human supervision. Meanwhile, an end-to-end post-training strategy is presented to enable the LLM agent to elicit refined CAD Modeling Chain of Thought (CAD-CoT) and evolve into proficient CAD tool-using agents via online curriculum reinforcement learning. Our findings demonstrate ToolCAD fills the gap in adopting and training open-source LLMs for CAD tool-using agents, enabling them to perform comparably to proprietary models, paving the way for more accessible and robust autonomous text-to-CAD modeling systems. 

---
# Pruning Extensions and Efficiency Trade-Offs for Sustainable Time Series Classification 

**Authors**: Raphael Fischer, Angus Dempster, Sebastian Buschjäger, Matthias Jakobs, Urav Maniar, Geoffrey I. Webb  

**Link**: [PDF](https://arxiv.org/pdf/2604.07953)  

**Abstract**: Time series classification (TSC) enables important use cases, however lacks a unified understanding of performance trade-offs across models, datasets, and hardware. While resource awareness has grown in the field, TSC methods have not yet been rigorously evaluated for energy efficiency. This paper introduces a holistic evaluation framework that explicitly explores the balance of predictive performance and resource consumption in TSC. To boost efficiency, we apply a theoretically bounded pruning strategy to leading hybrid classifiers - Hydra and Quant - and present Hydrant, a novel, prunable combination of both. With over 4000 experimental configurations across 20 MONSTER datasets, 13 methods, and three compute setups, we systematically analyze how model design, hyperparameters, and hardware choices affect practical TSC performance. Our results showcase that pruning can significantly reduce energy consumption by up to 80% while maintaining competitive predictive quality, usually costing the model less than 5% of accuracy. The proposed methodology, experimental results, and accompanying software advance TSC toward sustainable and reproducible practice. 

---
# Investigation of Automated Design of Quantum Circuits for Imaginary Time Evolution Methods Using Deep Reinforcement Learning 

**Authors**: Ryo Suzuki, Shohei Watabe  

**Link**: [PDF](https://arxiv.org/pdf/2604.07951)  

**Abstract**: Efficient ground state search is fundamental to advancing combinatorial optimization problems and quantum chemistry. While the Variational Imaginary Time Evolution (VITE) method offers a useful alternative to Variational Quantum Eigensolver (VQE), and Quantum Approximate Optimization Algorithm (QAOA), its implementation on Noisy Intermediate-Scale Quantum (NISQ) devices is severely limited by the gate counts and depth of manually designed ansatz. Here, we present an automated framework for VITE circuit design using Double Deep-Q Networks (DDQN). Our approach treats circuit construction as a multi-objective optimization problem, simultaneously minimizing energy expectation values and optimizing circuit complexity. By introducing adoptive thresholds, we demonstrate significant hardware overhead reductions. In Max-Cut problems, our agent autonomously discovered circuits with approximately 37\% fewer gates and 43\% less depth than standard hardware-efficient ansatz on average. For molecular hydrogen ($H_2$), the DDQN also achieved the Full-CI limit, with maintaining a significantly shallower circuit. These results suggest that deep reinforcement learning can be helpful to find non-intuitive, optimal circuit structures, providing a pathway toward efficient, hardware-aware quantum algorithm design. 

---
# Incremental Residual Reinforcement Learning Toward Real-World Learning for Social Navigation 

**Authors**: Haruto Nagahisa, Kohei Matsumoto, Yuki Tomita, Yuki Hyodo, Ryo Kurazume  

**Link**: [PDF](https://arxiv.org/pdf/2604.07945)  

**Abstract**: As the demand for mobile robots continues to increase, social navigation has emerged as a critical task, driving active research into deep reinforcement learning (RL) approaches. However, because pedestrian dynamics and social conventions vary widely across different regions, simulations cannot easily encompass all possible real-world scenarios. Real-world RL, in which agents learn while operating directly in physical environments, presents a promising solution to this issue. Nevertheless, this approach faces significant challenges, particularly regarding constrained computational resources on edge devices and learning efficiency. In this study, we propose incremental residual RL (IRRL). This method integrates incremental learning, which is a lightweight process that operates without a replay buffer or batch updates, with residual RL, which enhances learning efficiency by training only on the residuals relative to a base policy. Through the simulation experiments, we demonstrated that, despite lacking a replay buffer, IRRL achieved performance comparable to those of conventional replay buffer-based methods and outperformed existing incremental learning approaches. Furthermore, the real-world experiments confirmed that IRRL can enable robots to effectively adapt to previously unseen environments through the real-world learning. 

---
# On-Policy Distillation of Language Models for Autonomous Vehicle Motion Planning 

**Authors**: Amirhossein Afsharrad, Amirhesam Abedsoltan, Ahmadreza Moradipari, Sanjay Lall  

**Link**: [PDF](https://arxiv.org/pdf/2604.07944)  

**Abstract**: Large language models (LLMs) have recently demonstrated strong potential for autonomous vehicle motion planning by reformulating trajectory prediction as a language generation problem. However, deploying capable LLMs in resource-constrained onboard systems remains a fundamental challenge. In this paper, we study how to effectively transfer motion planning knowledge from a large teacher LLM to a smaller, more deployable student model. We build on the GPT-Driver framework, which represents driving scenes as language prompts and generates waypoint trajectories with chain-of-thought reasoning, and investigate two student training paradigms: (i) on-policy generalized knowledge distillation (GKD), which trains the student on its own self-generated outputs using dense token-level feedback from the teacher, and (ii) a dense-feedback reinforcement learning (RL) baseline that uses the teacher's log-probabilities as per-token reward signals in a policy gradient framework. Experiments on the nuScenes benchmark show that GKD substantially outperforms the RL baseline and closely approaches teacher-level performance despite a 5$\times$ reduction in model size. These results highlight the practical value of on-policy distillation as a principled and effective approach to deploying LLM-based planners in autonomous driving systems. 

---
# Large Language Model Post-Training: A Unified View of Off-Policy and On-Policy Learning 

**Authors**: Shiwan Zhao, Zhihu Wang, Xuyang Zhao, Jiaming Zhou, Caiyue Xu, Chenfei Liu, Liting Zhang, Yuhang Jia, Yanzhe Zhang, Hualong Yu, Zichen Xu, Qicheng Li, Yong Qin  

**Link**: [PDF](https://arxiv.org/pdf/2604.07941)  

**Abstract**: Post-training has become central to turning pretrained large language models (LLMs) into aligned and deployable systems. Recent progress spans supervised fine-tuning (SFT), preference optimization, reinforcement learning (RL), process supervision, verifier-guided methods, distillation, and multi-stage pipelines. Yet these methods are often discussed in fragmented ways, organized by labels or objective families rather than by the behavioral bottlenecks they address.
This survey argues that LLM post-training is best understood as structured intervention on model behavior. We organize the field first by trajectory provenance, which defines two primary learning regimes: off-policy learning on externally supplied trajectories, and on-policy learning on learner-generated rollouts. We then interpret methods through two recurring roles -- effective support expansion, which makes useful behaviors more reachable, and policy reshaping, which improves behavior within already reachable regions -- together with a complementary systems-level role, behavioral consolidation, which preserves, transfers, and amortizes behavior across stages and model transitions.
This perspective yields a unified reading of major paradigms. SFT may serve either support expansion or policy reshaping, whereas preference-based methods are usually off-policy reshaping. On-policy RL often improves behavior on learner-generated states, though under stronger guidance it can also make hard-to-reach reasoning paths reachable. Distillation is often best understood as consolidation rather than only compression, and hybrid pipelines emerge as coordinated multi-stage compositions.
Overall, the framework helps diagnose post-training bottlenecks and reason about stage composition, suggesting that progress in LLM post-training increasingly depends on coordinated system design rather than any single dominant objective. 

---
# Same Outcomes, Different Journeys: A Trace-Level Framework for Comparing Human and GUI-Agent Behavior in Production Search Systems 

**Authors**: Maria Movin, Claudia Hauff, Aron Henriksson, Panagiotis Papapetrou  

**Link**: [PDF](https://arxiv.org/pdf/2604.07929)  

**Abstract**: LLM-driven GUI agents are increasingly used in production systems to automate workflows and simulate users for evaluation and optimization. Yet most GUI-agent evaluations emphasize task success and provide limited evidence on whether agents interact in human-like ways. We present a trace-level evaluation framework that compares human and agent behavior across (i) task outcome and effort, (ii) query formulation, and (iii) navigation across interface states. We instantiate the framework in a controlled study in a production audio-streaming search application, where 39 participants and a state-of-the-art GUI agent perform ten multi-hop search tasks. The agent achieves task success comparable to participants and generates broadly aligned queries, but follows systematically different navigation strategies: participants exhibit content-centric, exploratory behavior, while the agent is more search-centric and low-branching. These results show that outcome and query alignment do not imply behavioral alignment, motivating trace-level diagnostics when deploying GUI agents as proxies for users in production search systems. 

---
# Sinkhorn doubly stochastic attention rank decay analysis 

**Authors**: Michela Lapenna, Rita Fioresi, Bahman Gharesifard  

**Link**: [PDF](https://arxiv.org/pdf/2604.07925)  

**Abstract**: The self-attention mechanism is central to the success of Transformer architectures. However, standard row-stochastic attention has been shown to suffer from significant signal degradation across layers. In particular, it can induce rank collapse, resulting in increasingly uniform token representations, as well as entropy collapse, characterized by highly concentrated attention distributions. Recent work has highlighted the benefits of doubly stochastic attention as a form of entropy regularization, promoting a more balanced attention distribution and leading to improved empirical performance. In this paper, we study rank collapse across network depth and show that doubly stochastic attention matrices normalized with Sinkhorn algorithm preserve rank more effectively than standard Softmax row-stochastic ones. As previously shown for Softmax, skip connections are crucial to mitigate rank collapse. We empirically validate this phenomenon on both sentiment analysis and image classification tasks. Moreover, we derive a theoretical bound for the pure self-attention rank decay when using Sinkhorn normalization and find that rank decays to one doubly exponentially with depth, a phenomenon that has already been shown for Softmax. 

---
# Mitigating Entangled Steering in Large Vision-Language Models for Hallucination Reduction 

**Authors**: Yuanhong Zhang, Zhaoyang Wang, Xin Zhang, Weizhan Zhang, Joey Tianyi Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2604.07914)  

**Abstract**: Large Vision-Language Models (LVLMs) have achieved remarkable success across cross-modal tasks but remain hindered by hallucinations, producing textual outputs inconsistent with visual content. Existing methods mitigate hallucinations but often alter generation behavior, resulting in shorter outputs and shifted token distributions, especially in latent space steering approaches. We identify that this issue stems from entangled steering signals, where suppressing hallucinations inadvertently disrupts the model's intrinsic generation behavior. To address this, we propose MESA, an effective plug-and-play framework that performs controlled and selective latent intervention for hallucination mitigation. Specifically, MESA targets hallucination-relevant responses while preserving the model's original token distribution, enabling effective hallucination reduction without compromising generation behavior. Extensive experiments across diverse generative and discriminative benchmarks demonstrate that MESA consistently reduces hallucinations while better preserving generation behavior, outperforming prior methods across multiple LVLM families. 

---
# Dynamic Attentional Context Scoping: Agent-Triggered Focus Sessions for Isolated Per-Agent Steering in Multi-Agent LLM Orchestration 

**Authors**: Nickson Patel  

**Link**: [PDF](https://arxiv.org/pdf/2604.07911)  

**Abstract**: Multi-agent LLM orchestration systems suffer from context pollution: when N concurrent agents compete for the orchestrator's context window, each agent's task state, partial outputs, and pending questions contaminate the steering interactions of every other agent, degrading decision quality. We introduce Dynamic Attentional Context Scoping (DACS), a mechanism in which the orchestrator operates in two asymmetric modes. In Registry mode it holds only lightweight per-agent status summaries (<=200 tokens each), remaining responsive to all agents and the user. When an agent emits a SteeringRequest, the orchestrator enters Focus(a_i) mode, injecting the full context of agent a_i while compressing all other agents to their registry entries. Context isolation is agent-triggered, asymmetric, and deterministic: the context window contains exactly F(a_i) + R_{-i} during steering, eliminating cross-agent contamination without requiring context compression or retrieval. We evaluate DACS across four experimental phases totalling 200 trials: Phase 1 tests N in {3,5,10} (60 trials); Phase 2 tests agent heterogeneity and adversarial dependencies (60 trials); Phase 3 tests decision density up to D=15 (40 trials); Phase 4 uses autonomous LLM agents for free-form questions (40 trials, Claude Haiku 4.5). Across all 8 synthetic scenarios, DACS achieves 90.0--98.4% steering accuracy versus 21.0--60.0% for a flat-context baseline (p < 0.0001 throughout), with wrong-agent contamination falling from 28--57% to 0--14% and context efficiency ratios of up to 3.53x. The accuracy advantage grows with N and D; keyword matching is validated by LLM-as-judge across all phases (mean kappa=0.909). DACS outperforms the flat-context baseline by +17.2pp at N=3 (p=0.0023) and +20.4pp at N=5 (p=0.0008) in Phase 4, with the advantage growing with N confirmed by two independent judges. 

---
# AnomalyAgent: Agentic Industrial Anomaly Synthesis via Tool-Augmented Reinforcement Learning 

**Authors**: Jiaming Su, Tengchao Yang, Ruikang Zhang, Zhengan Yan, Haoyu Sun, Linfeng Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2604.07900)  

**Abstract**: Industrial anomaly generation is a crucial method for alleviating the data scarcity problem in anomaly detection tasks. Most existing anomaly synthesis methods rely on single-step generation mechanisms, lacking complex reasoning and iterative optimization capabilities, making it difficult to generate anomaly samples with high semantic realism. We propose AnomalyAgent, an anomaly synthesis agent with self-reflection, knowledge retrieval, and iterative refinement capabilities, aiming to generate realistic and diverse anomalies. Specifically, AnomalyAgent is equipped with five tools: Prompt Generation (PG), Image Generation (IG), Quality Evaluation (QE), Knowledge Retrieval (KR), and Mask Generation (MG), enabling closed-loop optimization. To improve decision-making and self-reflection, we construct structured trajectories from real anomaly images and design a two-stage training framework: supervised fine-tuning followed by reinforcement learning. This process is driven by a three-part reward mechanism: (1) task rewards to supervise the quality and location rationality of generated anomalies; (2) reflection rewards to train the model's ability to improve anomaly synthesis prompt; (3) behavioral rewards to ensure adherence to the trajectory. On the MVTec-AD dataset, AnomalyAgent achieves IS/IC-L of 2.10/0.33 for anomaly generation, 57.0% classification accuracy using ResNet34, and 99.3%/74.2% AP at the image/pixel level using a simple UNet, surpassing all zero-shot SOTA methods. The code and data will be made publicly available. 

---
# TSUBASA: Improving Long-Horizon Personalization via Evolving Memory and Self-Learning with Context Distillation 

**Authors**: Xinliang Frederick Zhang, Lu Wang  

**Link**: [PDF](https://arxiv.org/pdf/2604.07894)  

**Abstract**: Personalized large language models (PLLMs) have garnered significant attention for their ability to align outputs with individual's needs and preferences. However, they still struggle with long-horizon tasks, such as tracking a user's extensive history of conversations or activities. Existing memory mechanisms often fail to capture evolving behaviors, and RAG paradigms are trapped by a quality-efficiency tradeoff. Meanwhile, parametric adaptation is bottlenecked by train-inference gap due to the scarcity of labeled data. To enhance the long-horizon capabilities of PLLMs, we introduce TSUBASA, a two-pronged approach designed to improve memory writing via dynamic memory evolution, and memory reading via self-learning with a context distillation objective to internalize user experiences. Extensive evaluations on long-horizon benchmarks using the Qwen-3 model family (4B to 32B) validate the effectiveness of TSUBASA, surpassing competitive memory-augmented systems that rely primarily on memory writing, such as Mem0 and Memory-R1. Our analyses further confirms that TSUBASA breaks the quality-efficiency barrier to achieve Pareto improvements, delivering robust, high-fidelity personalization with a reduced token budget. 

---
# Data Selection for Multi-turn Dialogue Instruction Tuning 

**Authors**: Bo Li, Shikun Zhang, Wei Ye  

**Link**: [PDF](https://arxiv.org/pdf/2604.07892)  

**Abstract**: Instruction-tuned language models increasingly rely on large multi-turn dialogue corpora, but these datasets are often noisy and structurally inconsistent, with topic drift, repetitive chitchat, and mismatched answer formats across turns. We address this from a data selection perspective and propose \textbf{MDS} (Multi-turn Dialogue Selection), a dialogue-level framework that scores whole conversations rather than isolated turns. MDS combines a global coverage stage that performs bin-wise selection in the user-query trajectory space to retain representative yet non-redundant dialogues, with a local structural stage that evaluates within-dialogue reliability through entity-grounded topic grounding and information progress, together with query-answer form consistency for functional alignment. MDS outperforms strong single-turn selectors, dialogue-level LLM scorers, and heuristic baselines on three multi-turn benchmarks and an in-domain Banking test set, achieving the best overall rank across reference-free and reference-based metrics, and is more robust on long conversations under the same training budget. Code and resources are included in the supplementary materials. 

---
# Reinforcement-Guided Synthetic Data Generation for Privacy-Sensitive Identity Recognition 

**Authors**: Xuemei Jia, Jiawei Du, Hui Wei, Jun Chen, Joey Tianyi Zhou, Zheng Wang  

**Link**: [PDF](https://arxiv.org/pdf/2604.07884)  

**Abstract**: High-fidelity generative models are increasingly needed in privacy-sensitive scenarios, where access to data is severely restricted due to regulatory and copyright constraints. This scarcity hampers model development--ironically, in settings where generative models are most needed to compensate for the lack of data. This creates a self-reinforcing challenge: limited data leads to poor generative models, which in turn fail to mitigate data scarcity. To break this cycle, we propose a reinforcement-guided synthetic data generation framework that adapts general-domain generative priors to privacy-sensitive identity recognition tasks. We first perform a cold-start adaptation to align a pretrained generator with the target domain, establishing semantic relevance and initial fidelity. Building on this foundation, we introduce a multi-objective reward that jointly optimizes semantic consistency, coverage diversity, and expression richness, guiding the generator to produce both realistic and task-effective samples. During downstream training, a dynamic sample selection mechanism further prioritizes high-utility synthetic samples, enabling adaptive data scaling and improved domain alignment. Extensive experiments on benchmark datasets demonstrate that our framework significantly improves both generation fidelity and classification accuracy, while also exhibiting strong generalization to novel categories in small-data regimes. 

---
# FlowGuard: Towards Lightweight In-Generation Safety Detection for Diffusion Models via Linear Latent Decoding 

**Authors**: Jinghan Yang, Yihe Fan, Xudong Pan, Min Yang  

**Link**: [PDF](https://arxiv.org/pdf/2604.07879)  

**Abstract**: Diffusion-based image generation models have advanced rapidly but pose a safety risk due to their potential to generate Not-Safe-For-Work (NSFW) content. Existing NSFW detection methods mainly operate either before or after image generation. Pre-generation methods rely on text prompts and struggle with the gap between prompt safety and image safety. Post-generation methods apply classifiers to final outputs, but they are poorly suited to intermediate noisy images. To address this, we introduce FlowGuard, a cross-model in-generation detection framework that inspects intermediate denoising steps. This is particularly challenging in latent diffusion, where early-stage noise obscures visual signals. FlowGuard employs a novel linear approximation for latent decoding and leverages a curriculum learning approach to stabilize training. By detecting unsafe content early, FlowGuard reduces unnecessary diffusion steps to cut computational costs. Our cross-model benchmark spanning nine diffusion-based backbones shows the effectiveness of FlowGuard for in-generation NSFW detection in both in-distribution and out-of-distribution settings, outperforming existing methods by over 30% in F1 score while delivering transformative efficiency gains, including slashing peak GPU memory demand by over 97% and projection time from 8.1 seconds to 0.2 seconds compared to standard VAE decoding. 

---
# PyVRP$^+$: LLM-Driven Metacognitive Heuristic Evolution for Hybrid Genetic Search in Vehicle Routing Problems 

**Authors**: Manuj Malik, Jianan Zhou, Shashank Reddy Chirra, Zhiguang Cao  

**Link**: [PDF](https://arxiv.org/pdf/2604.07872)  

**Abstract**: Designing high-performing metaheuristics for NP-hard combinatorial optimization problems, such as the Vehicle Routing Problem (VRP), remains a significant challenge, often requiring extensive domain expertise and manual tuning. Recent advances have demonstrated the potential of large language models (LLMs) to automate this process through evolutionary search. However, existing methods are largely reactive, relying on immediate performance feedback to guide what are essentially black-box code mutations. Our work departs from this paradigm by introducing Metacognitive Evolutionary Programming (MEP), a framework that elevates the LLM to a strategic discovery agent. Instead of merely reacting to performance scores, MEP compels the LLM to engage in a structured Reason-Act-Reflect cycle, forcing it to explicitly diagnose failures, formulate design hypotheses, and implement solutions grounded in pre-supplied domain knowledge. By applying MEP to evolve core components of the state-of-the-art Hybrid Genetic Search (HGS) algorithm, we discover novel heuristics that significantly outperform the original baseline. By steering the LLM to reason strategically about the exploration-exploitation trade-off, our approach discovers more effective and efficient heuristics applicable across a wide spectrum of VRP variants. Our results show that MEP discovers heuristics that yield significant performance gains over the original HGS baseline, improving solution quality by up to 2.70\% and reducing runtime by over 45\% on challenging VRP variants. 

---
# Task-Adaptive Retrieval over Agentic Multi-Modal Web Histories via Learned Graph Memory 

**Authors**: Saman Forouzandeh, Kamal Berahmand, Mahdi Jalili  

**Link**: [PDF](https://arxiv.org/pdf/2604.07863)  

**Abstract**: Retrieving relevant observations from long multi-modal web interaction histories is challenging because relevance depends on the evolving task state, modality (screenshots, HTML text, structured signals), and temporal distance. Prior approaches typically rely on static similarity thresholds or fixed-capacity buffers, which fail to adapt relevance to the current task context. We propose \textbf{ACGM}, a learned graph-memory retriever that constructs \emph{task-adaptive} relevance graphs over agent histories using policy-gradient optimization from downstream task success. ACGM captures heterogeneous temporal dynamics with modality-specific decay (visual decays $4.3\times$ faster than text: $\lambda_v{=}0.47$ vs.\ $\lambda_x{=}0.11$) and learns sparse connectivity (3.2 edges/node), enabling efficient $O(\log T)$ retrieval. Across WebShop, VisualWebArena, and Mind2Web, ACGM improves retrieval quality to \textbf{82.7 nDCG@10} (+9.3 over GPT-4o, $p{<}0.001$) and \textbf{89.2\% Precision@10} (+7.7), outperforming 19 strong dense, re-ranking, multi-modal, and graph-based baselines. Code to reproduce our results is available at{\color{blue}\href{this https URL}{Saman Forouzandeh}}. 

---
# Networking-Aware Energy Efficiency in Agentic AI Inference: A Survey 

**Authors**: Xiaojing Chen, Haiqi Yu, Wei Ni, Dusit Niyato, Ruichen Zhang, Xin Wang, Shunqing Zhang, Shugong Xu  

**Link**: [PDF](https://arxiv.org/pdf/2604.07857)  

**Abstract**: The rapid emergence of Large Language Models (LLMs) has catalyzed Agentic artificial intelligence (AI), autonomous systems integrating perception, reasoning, and action into closed-loop pipelines for continuous adaptation. While unlocking transformative applications in mobile edge computing, autonomous systems, and next-generation wireless networks, this paradigm creates fundamental energy challenges through iterative inference and persistent data exchange. Unlike traditional AI where bottlenecks are computational Floating Point Operations (FLOPs), Agentic AI faces compounding computational and communication energy costs. In this survey, we propose an energy accounting framework identifying computational and communication costs across the Perception-Reasoning-Action cycle. We establish a unified taxonomy spanning model simplification, computation control, input and attention optimization, and hardware-aware inference. We explore cross-layer co-design strategies jointly optimizing model parameters, wireless transmissions, and edge resources. Finally, we identify open challenges of federated green learning, carbon-aware agency, 6th generation mobile communication (6G)-native Agentic AI, and self-sustaining systems, providing a roadmap for scalable autonomous intelligence. 

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
# Filling the Gaps: Selective Knowledge Augmentation for LLM Recommenders 

**Authors**: Jaehyun Lee, Sanghwan Jang, SeongKu Kang, Hwanjo Yu  

**Link**: [PDF](https://arxiv.org/pdf/2604.07825)  

**Abstract**: Large language models (LLMs) have recently emerged as powerful training-free recommenders. However, their knowledge of individual items is inevitably uneven due to imbalanced information exposure during pretraining, a phenomenon we refer to as knowledge gap problem. To address this, most prior methods have employed a naive uniform augmentation that appends external information for every item in the input prompt. However, this approach not only wastes limited context budget on redundant augmentation for well-known items but can also hinder the model's effective reasoning. To this end, we propose KnowSA_CKP (Knowledge-aware Selective Augmentation with Comparative Knowledge Probing) to mitigate the knowledge gap problem. KnowSA_CKP estimates the LLM's internal knowledge by evaluating its capability to capture collaborative relationships and selectively injects additional information only where it is most needed. By avoiding unnecessary augmentation for well-known items, KnowSA_CKP focuses on items that benefit most from knowledge supplementation, thereby making more effective use of the context budget. KnowSA_CKP requires no fine-tuning step, and consistently improves both recommendation accuracy and context efficiency across four real-world datasets. 

---
# LPM 1.0: Video-based Character Performance Model 

**Authors**: Ailing Zeng, Casper Yang, Chauncey Ge, Eddie Zhang, Garvey Xu, Gavin Lin, Gilbert Gu, Jeremy Pi, Leo Li, Mingyi Shi, Sheng Bi, Steven Tang, Thorn Hang, Tobey Guo, Vincent Li, Xin Tong, Yikang Li, Yuchen Sun, Zhao, Yuhan Lu, Yuwei Li, Zane Zhang, Zeshi Yang, Zi Ye  

**Link**: [PDF](https://arxiv.org/pdf/2604.07823)  

**Abstract**: Performance, the externalization of intent, emotion, and personality through visual, vocal, and temporal behavior, is what makes a character alive. Learning such performance from video is a promising alternative to traditional 3D pipelines. However, existing video models struggle to jointly achieve high expressiveness, real-time inference, and long-horizon identity stability, a tension we call the performance trilemma. Conversation is the most comprehensive performance scenario, as characters simultaneously speak, listen, react, and emote while maintaining identity over time. To address this, we present LPM 1.0 (Large Performance Model), focusing on single-person full-duplex audio-visual conversational performance. Concretely, we build a multimodal human-centric dataset through strict filtering, speaking-listening audio-video pairing, performance understanding, and identity-aware multi-reference extraction; train a 17B-parameter Diffusion Transformer (Base LPM) for highly controllable, identity-consistent performance through multimodal conditioning; and distill it into a causal streaming generator (Online LPM) for low-latency, infinite-length interaction. At inference, given a character image with identity-aware references, LPM 1.0 generates listening videos from user audio and speaking videos from synthesized audio, with text prompts for motion control, all at real-time speed with identity-stable, infinite-length generation. LPM 1.0 thus serves as a visual engine for conversational agents, live streaming characters, and game NPCs. To systematically evaluate this setting, we propose LPM-Bench, the first benchmark for interactive character performance. LPM 1.0 achieves state-of-the-art results across all evaluated dimensions while maintaining real-time inference. 

---
# Loop, Think, & Generalize: Implicit Reasoning in Recurrent-Depth Transformers 

**Authors**: Harsh Kohli, Srinivasan Parthasarathy, Huan Sun, Yuekun Yao  

**Link**: [PDF](https://arxiv.org/pdf/2604.07822)  

**Abstract**: We study implicit reasoning, i.e. the ability to combine knowledge or rules within a single forward pass. While transformer-based large language models store substantial factual knowledge and rules, they often fail to compose this knowledge for implicit multi-hop reasoning, suggesting a lack of compositional generalization over their parametric knowledge. To address this limitation, we study recurrent-depth transformers, which enables iterative computation over the same transformer layers. We investigate two compositional generalization challenges under the implicit reasoning scenario: systematic generalization, i.e. combining knowledge that is never used for compositions during training, and depth extrapolation, i.e. generalizing from limited reasoning depth (e.g. training on up to 5-hop) to deeper compositions (e.g. 10-hop). Through controlled studies with models trained from scratch, we show that while vanilla transformers struggle with both generalization challenges, recurrent-depth transformers can effectively make such generalization. For systematic generalization, we find that this ability emerges through a three-stage grokking process, transitioning from memorization to in-distribution generalization and finally to systematic generalization, supported by mechanistic analysis. For depth extrapolation, we show that generalization beyond training depth can be unlocked by scaling inference-time recurrence, with more iterations enabling deeper reasoning. We further study how training strategies affect extrapolation, providing guidance on training recurrent-depth transformers, and identify a key limitation, overthinking, where excessive recurrence degrades predictions and limits generalization to very deep compositions. 

---
# More Capable, Less Cooperative? When LLMs Fail At Zero-Cost Collaboration 

**Authors**: Advait Yadav, Sid Black, Oliver Sourbut  

**Link**: [PDF](https://arxiv.org/pdf/2604.07821)  

**Abstract**: Large language model (LLM) agents increasingly coordinate in multi-agent systems, yet we lack an understanding of where and why cooperation failures may arise. In many real-world coordination problems, from knowledge sharing in organizations to code documentation, helping others carries negligible personal cost while generating substantial collective benefits. However, whether LLM agents cooperate when helping neither benefits nor harms the helper, while being given explicit instructions to do so, remains unknown. We build a multi-agent setup designed to study cooperative behavior in a frictionless environment, removing all strategic complexity from cooperation. We find that capability does not predict cooperation: OpenAI o3 achieves only 17% of optimal collective performance while OpenAI o3-mini reaches 50%, despite identical instructions to maximize group revenue. Through a causal decomposition that automates one side of agent communication, we separate cooperation failures from competence failures, tracing their origins through agent reasoning analysis. Testing targeted interventions, we find that explicit protocols double performance for low-competence models, and tiny sharing incentives improve models with weak cooperation. Our findings suggest that scaling intelligence alone will not solve coordination problems in multi-agent systems and will require deliberate cooperative design, even when helping others costs nothing. 

---
# PolicyLong: Towards On-Policy Context Extension 

**Authors**: Junlong Jia, Ziyang Chen, Xing Wu, Chaochen Gao, TingHao Yu, Feng Zhang, Songlin Hu  

**Link**: [PDF](https://arxiv.org/pdf/2604.07809)  

**Abstract**: Extending LLM context windows is hindered by scarce high-quality long-context data. Recent methods synthesize data with genuine long-range dependencies via information-theoretic verification, selecting contexts that reduce a base model's predictive entropy. However, their single-pass offline construction with a fixed model creates a fundamental off-policy gap: the static screening landscape misaligns with the model's evolving capabilities, causing the training distribution to drift. We propose PolicyLong, shifting data construction towards a dynamic on-policy paradigm. By iteratively re-executing data screening (entropy computation, retrieval, and verification) using the current model, PolicyLong ensures the training distribution tracks evolving capabilities, yielding an emergent self-curriculum. Crucially, both positive and hard negative contexts derive from the current model's entropy landscape, co-evolving what the model learns to exploit and resist. Experiments on RULER, HELMET, and LongBench-v2 (Qwen2.5-3B) show PolicyLong consistently outperforms EntropyLong and NExtLong, with gains growing at longer contexts (e.g., +2.54 at 128K on RULER), confirming the value of on-policy data evolution. 

---
# The Weaponization of Computer Vision: Tracing Military-Surveillance Ties through Conference Sponsorship 

**Authors**: Noa Garcia, Amelia Katirai  

**Link**: [PDF](https://arxiv.org/pdf/2604.07803)  

**Abstract**: Computer vision, a core domain of artificial intelligence (AI), is the field that enables the computational analysis, understanding, and generation of visual data. Despite being historically rooted in military funding and increasingly deployed in warfare, the field tends to position itself as a neutral, purely technical endeavor, failing to engage in discussions about its dual-use applications. Yet it has been reported that computer vision systems are being systematically weaponized to assist in technologies that inflict harm, such as surveillance or warfare. Expanding on these concerns, we study the extent to which computer vision research is being used in the military and surveillance domains. We do so by collecting a dataset of tech companies with financial ties to the field's central research exchange platform: conferences. Conference sponsorship, we argue, not only serves as strong evidence of a company's investment in the field but also provides a privileged position for shaping its trajectory. By investigating sponsors' activities, we reveal that 44% of them have a direct connection with military or surveillance applications. We extend our analysis through two case studies in which we discuss the opportunities and limitations of sponsorship as a means for uncovering technological weaponization. 

---
# Latent Anomaly Knowledge Excavation: Unveiling Sparse Sensitive Neurons in Vision-Language Models 

**Authors**: Shaotian Li, Shangze Li, Chuancheng Shi, Wenhua Wu, Yanqiu Wu, Xiaohan Yu, Fei Shen, Tat-Seng Chua  

**Link**: [PDF](https://arxiv.org/pdf/2604.07802)  

**Abstract**: Large-scale vision-language models (VLMs) exhibit remarkable zero-shot capabilities, yet the internal mechanisms driving their anomaly detection (AD) performance remain poorly understood. Current methods predominantly treat VLMs as black-box feature extractors, assuming that anomaly-specific knowledge must be acquired through external adapters or memory banks. In this paper, we challenge this assumption by arguing that anomaly knowledge is intrinsically embedded within pre-trained models but remains latent and under-activated. We hypothesize that this knowledge is concentrated within a sparse subset of anomaly-sensitive neurons. To validate this, we propose latent anomaly knowledge excavation (LAKE), a training-free framework that identifies and elicits these critical neuronal signals using only a minimal set of normal samples. By isolating these sensitive neurons, LAKE constructs a highly compact normality representation that integrates visual structural deviations with cross-modal semantic activations. Extensive experiments on industrial AD benchmarks demonstrate that LAKE achieves state-of-the-art performance while providing intrinsic, neuron-level interpretability. Ultimately, our work advocates for a paradigm shift: redefining anomaly detection as the targeted activation of latent pre-trained knowledge rather than the acquisition of a downstream task. 

---
# TEMPER: Testing Emotional Perturbation in Quantitative Reasoning 

**Authors**: Atahan Dokme, Benjamin Reichman, Larry Heck  

**Link**: [PDF](https://arxiv.org/pdf/2604.07801)  

**Abstract**: Large language models are trained and evaluated on quantitative reasoning tasks written in clean, emotionally neutral language. However, real-world queries are often wrapped in frustration, urgency or enthusiasm. Does emotional framing alone degrade reasoning when all numerical content is preserved? To investigate this, a controlled emotion translation framework is developed that rewrites problems into emotional variants while preserving all quantities and relationships. Using this framework, Temper-5400 (5,400 semantically verified emotion--neutral pairs) is constructed across GSM8K, MultiArith, and ARC-Challenge, and evaluated on eighteen models (1B to frontier scale). Two core results emerge: First, emotional framing reduces accuracy by 2-10 percentage points even though all numerical content is preserved. Second, neutralizing emotional variants recovers most of the lost performance, showing both that the degradation is tied to emotional style rather than content corruption and that neutralization can serve as a lightweight inference-time mitigation. Non-emotional paraphrases cause no such degradation, implicating emotional content rather than surface-level changes. Beyond emotion specifically, the benchmark construction procedure provides a general framework for controlled stylistic translation and robustness evaluation. 

---
# Learning Without Losing Identity: Capability Evolution for Embodied Agents 

**Authors**: Xue Qin, Simin Luan, John See, Cong Yang, Zhijun Li  

**Link**: [PDF](https://arxiv.org/pdf/2604.07799)  

**Abstract**: Embodied agents are expected to operate persistently in dynamic physical environments, continuously acquiring new capabilities over time. Existing approaches to improving agent performance often rely on modifying the agent itself -- through prompt engineering, policy updates, or structural redesign -- leading to instability and loss of identity in long-lived systems. In this work, we propose a capability-centric evolution paradigm for embodied agents. We argue that a robot should maintain a persistent agent as its cognitive identity, while enabling continuous improvement through the evolution of its capabilities. Specifically, we introduce the concept of Embodied Capability Modules (ECMs), which represent modular, versioned units of embodied functionality that can be learned, refined, and composed over time. We present a unified framework in which capability evolution is decoupled from agent identity. Capabilities evolve through a closed-loop process involving task execution, experience collection, model refinement, and module updating, while all executions are governed by a runtime layer that enforces safety and policy constraints. We demonstrate through simulated embodied tasks that capability evolution improves task success rates from 32.4% to 91.3% over 20 iterations, outperforming both agent-modification baselines and established skill-learning methods (SPiRL, SkiMo), while preserving zero policy drift and zero safety violations. Our results suggest that separating agent identity from capability evolution provides a scalable and safe foundation for long-term embodied intelligence. 

---
# Toward Generalizable Graph Learning for 3D Engineering AI: Explainable Workflows for CAE Mode Shape Classification and CFD Field Prediction 

**Authors**: Tong Duy Son, Kohta Sugiura, Marc Brughmans, Andrey Hense, Zhihao Liu, Amirthalakshmi Veeraraghavan, Ajinkya Bhave, Jay Masters, Paolo di Carlo, Theo Geluk  

**Link**: [PDF](https://arxiv.org/pdf/2604.07781)  

**Abstract**: Automotive engineering development increasingly relies on heterogeneous 3D data, including finite element (FE) models, body-in-white (BiW) representations, CAD geometry, and CFD meshes. At the same time, engineering teams face growing pressure to shorten development cycles, improve performance and accelerate innovation. Although artificial intelligence (AI) is increasingly explored in this domain, many current methods remain task-specific, difficult to interpret, and hard to reuse across development stages. This paper presents a practical graph learning framework for 3D engineering AI, in which heterogeneous engineering assets are converted into physics-aware graph representations and processed by Graph Neural Networks (GNNs). The framework is designed to support both classification and prediction tasks. The framework is validated on two automotive applications: CAE vibration mode shape classification and CFD aerodynamic field prediction. For CAE vibration mode classification, a region-aware BiW graph supports explainable mode classification across vehicle and FE variants under label scarcity. For CFD aerodynamic field prediction, a physics-informed surrogate predicts pressure and wall shear stress (WSS) across aerodynamic body shape variants, while symmetry preserving down sampling retains accuracy with lower computational cost. The framework also outlines data generation guidance that can help engineers identify which additional simulations or labels are valuable to collect next. These results demonstrate a practical and reusable engineering AI workflow for more trustworthy CAE and CFD decision support. 

---
# Sensitivity-Positional Co-Localization in GQA Transformers 

**Authors**: Manoj Chandrashekar Rao  

**Link**: [PDF](https://arxiv.org/pdf/2604.07766)  

**Abstract**: We investigate a fundamental structural question in Grouped Query Attention (GQA) transformers: do the layers most sensitive to task correctness coincide with the layers where positional encoding adaptation has the greatest leverage? We term this the co-localization hypothesis and test it on Llama 3.1 8B, a 32-layer GQA model with a 4:1 query-to-key-value head ratio. We introduce \LSLORA, which restricts LoRA adaptation to layers identified via a novel correctness-differential hidden-state metric, and GARFA (GQA-Aware RoPE Frequency Adaptation), which attaches 8 learnable per-KV-head scalar multipliers to each targeted layer. Contrary to the co-localization hypothesis, we discover strong anti-localization: task-sensitive layers concentrate in the late network ($\ell\in\{23\text{-}31\}$) while RoPE-influential layers dominate the early network ($\ell\in\{0\text{-}9\}$), yielding Spearman $r_s = -0.735$ ($p = 1.66\times10^{-6}$). Despite this anti-localization, a 4-way cross-layer ablation shows that applying both interventions to the sensitivity-identified layers outperforms all alternative configurations by 4-16 percentage points across six diverse benchmarks (MMLU, GPQA, HumanEval+, MATH, MGSM, ARC), approaching Claude 3.5 Haiku on HumanEval+ (67.1% vs. 68.3%) at \$100 total compute cost. 

---
# Beyond Surface Artifacts: Capturing Shared Latent Forgery Knowledge Across Modalities 

**Authors**: Jingtong Dou, Chuancheng Shi, Jian Wang, Fei Shen, Zhiyong Wang, Tat-Seng Chua  

**Link**: [PDF](https://arxiv.org/pdf/2604.07763)  

**Abstract**: As generative artificial intelligence evolves, deepfake attacks have escalated from single-modality manipulations to complex, multimodal threats. Existing forensic techniques face a severe generalization bottleneck: by relying excessively on superficial, modality-specific artifacts, they neglect the shared latent forgery knowledge hidden beneath variable physical appearances. Consequently, these models suffer catastrophic performance degradation when confronted with unseen "dark modalities." To break this limitation, this paper introduces a paradigm shift that redefines multimodal forensics from conventional "feature fusion" to "modality generalization." We propose the first modality-agnostic forgery (MAF) detection framework. By explicitly decoupling modality-specific styles, MAF precisely extracts the essential, cross-modal latent forgery knowledge. Furthermore, we define two progressive dimensions to quantify model generalization: transferability toward semantically correlated modalities (Weak MAF), and robustness against completely isolated signals of "dark modality" (Strong MAF). To rigorously assess these generalization limits, we introduce the DeepModal-Bench benchmark, which integrates diverse multimodal forgery detection algorithms and adapts state-of-the-art generalized learning methods. This study not only empirically proves the existence of universal forgery traces but also achieves significant performance breakthroughs on unknown modalities via the MAF framework, offering a pioneering technical pathway for universal multimodal defense. 

---
# DailyArt: Discovering Articulation from Single Static Images via Latent Dynamics 

**Authors**: Hang Zhang, Qijian Tian, Jingyu Gong, Daoguo Dong, Xuhong Wang, Yuan Xie, Xin Tan  

**Link**: [PDF](https://arxiv.org/pdf/2604.07758)  

**Abstract**: Articulated objects are essential for embodied AI and world models, yet inferring their kinematics from a single closed-state image remains challenging because crucial motion cues are often occluded. Existing methods either require multi-state observations or rely on explicit part priors, retrieval, or other auxiliary inputs that partially expose the structure to be inferred. In this work, we present DailyArt, which formulates articulated joint estimation from a single static image as a synthesis-mediated reasoning problem. Instead of directly regressing joints from a heavily occluded observation, DailyArt first synthesizes a maximally articulated opened state under the same camera view to expose articulation cues, and then estimates the full set of joint parameters from the discrepancy between the observed and synthesized states. Using a set-prediction formulation, DailyArt recovers all joints simultaneously without requiring object-specific templates, multi-view inputs, or explicit part annotations at test time. Taking estimated joints as conditions, the framework further supports part-level novel state synthesis as a downstream capability. Extensive experiments show that DailyArt achieves strong performance in articulated joint estimation and supports part-level novel state synthesis conditioned on joints. Project page is available at this https URL. 

---
# MIMIC-Py: An Extensible Tool for Personality-Driven Automated Game Testing with Large Language Models 

**Authors**: Yifei Chen, Sarra Habchi, Lili Wei  

**Link**: [PDF](https://arxiv.org/pdf/2604.07752)  

**Abstract**: Modern video games are complex, non-deterministic systems that are difficult to test automatically at scale. Although prior work shows that personality-driven Large Language Model (LLM) agents can improve behavioural diversity and test coverage, existing tools largely remain research prototypes and lack cross-game reusability.
This tool paper presents MIMIC-Py, a Python-based automated game-testing tool that transforms personality-driven LLM agents into a reusable and extensible framework. MIMIC-Py exposes personality traits as configurable inputs and adopts a modular architecture that decouples planning, execution, and memory from game-specific logic. It supports multiple interaction mechanisms, enabling agents to interact with games via exposed APIs or synthesized code. We describe the design of MIMIC-Py and show how it enables deployment to new game environments with minimal engineering effort, bridging the gap between research prototypes and practical automated game testing.
The source code and a demo video are available on our project webpage: this https URL. 

---
# Beyond Pedestrians: Caption-Guided CLIP Framework for High-Difficulty Video-based Person Re-Identification 

**Authors**: Shogo Hamano, Shunya Wakasugi, Tatsuhito Sato, Sayaka Nakamura  

**Link**: [PDF](https://arxiv.org/pdf/2604.07740)  

**Abstract**: In recent years, video-based person Re-Identification (ReID) has gained attention for its ability to leverage spatiotemporal cues to match individuals across non-overlapping cameras. However, current methods struggle with high-difficulty scenarios, such as sports and dance performances, where multiple individuals wear similar clothing while performing dynamic movements. To overcome these challenges, we propose CG-CLIP, a novel caption-guided CLIP framework that leverages explicit textual descriptions and learnable tokens. Our method introduces two key components: Caption-guided Memory Refinement (CMR) and Token-based Feature Extraction (TFE). CMR utilizes captions generated by Multi-modal Large Language Models (MLLMs) to refine identity-specific features, capturing fine-grained details. TFE employs a cross-attention mechanism with fixed-length learnable tokens to efficiently aggregate spatiotemporal features, reducing computational overhead. We evaluate our approach on two standard datasets (MARS and iLIDS-VID) and two newly constructed high-difficulty datasets (SportsVReID and DanceVReID). Experimental results demonstrate that our method outperforms current state-of-the-art approaches, achieving significant improvements across all benchmarks. 

---
# TrajGuard: Streaming Hidden-state Trajectory Detection for Decoding-time Jailbreak Defense 

**Authors**: Cheng Liu, Xiaolei Liu, Xingyu Li, Bangzhou Xin, Kangyi Ding  

**Link**: [PDF](https://arxiv.org/pdf/2604.07727)  

**Abstract**: Existing jailbreak defense paradigms primarily rely on static detection of prompts, outputs, or internal states, often neglecting the dynamic evolution of risk during decoding. This oversight leaves risk signals embedded in decoding trajectories underutilized, constituting a critical blind spot in current defense systems. In this work, we empirically demonstrate that hidden states in critical layers during the decoding phase carry stronger and more stable risk signals than input jailbreak prompts. Specifically, the hidden representations of tokens generated during jailbreak attempts progressively approach high-risk regions in the latent space. Based on this observation, we propose TrajGuard, a training-free, decoding-time defense framework. TrajGuard aggregates hidden-state trajectories via a sliding window to quantify risk in real time, triggering a lightweight semantic adjudication only when risk within a local window persistently exceeds a threshold. This mechanism enables the immediate interruption or constraint of subsequent decoding. Extensive experiments across 12 jailbreak attacks and various open-source LLMs show that TrajGuard achieves an average defense rate of 95%. Furthermore, it reduces detection latency to 5.2 ms/token while maintaining a false positive rate below 1.5%. These results confirm that hidden-state trajectories during decoding can effectively support real-time jailbreak detection, highlighting a promising direction for defenses without model modification. 

---
# Detecting HIV-Related Stigma in Clinical Narratives Using Large Language Models 

**Authors**: Ziyi Chen, Yasir Khan, Mengyuan Zhang, Cheng Peng, Mengxian Lyu, Yiyang Liu, Krishna Vaddiparti, Robert L Cook, Mattia Prosperi, Yonghui Wu  

**Link**: [PDF](https://arxiv.org/pdf/2604.07717)  

**Abstract**: Human immunodeficiency virus (HIV)-related stigma is a critical psychosocial determinant of health for people living with HIV (PLWH), influencing mental health, engagement in care, and treatment outcomes. Although stigma-related experiences are documented in clinical narratives, there is a lack of off-the-shelf tools to extract and categorize them. This study aims to develop a large language model (LLM)-based tool for identifying HIV stigma from clinical notes. We identified clinical notes from PLWH receiving care at the University of Florida (UF) Health between 2012 and 2022. Candidate sentences were identified using expert-curated stigma-related keywords and iteratively expanded via clinical word embeddings. A total of 1,332 sentences were manually annotated across four stigma subscales: Concern with Public Attitudes, Disclosure Concerns, Negative Self-Image, and Personalized Stigma. We compared GatorTron-large and BERT as encoder-based baselines, and GPT-OSS-20B, LLaMA-8B, and MedGemma-27B as generative LLMs, under zero-shot and few-shot prompting. GatorTron-large achieved the best overall performance (Micro F1 = 0.62). Few-shot prompting substantially improved generative model performance, with 5-shot GPT-OSS-20B and LLaMA-8B achieving Micro-F1 scores of 0.57 and 0.59, respectively. Performance varied by stigma subscale, with Negative Self-Image showing the highest predictability and Personalized Stigma remaining the most challenging. Zero-shot generative inference exhibited non-trivial failure rates (up to 32%). This study develops the first practical NLP tool for identifying HIV stigma in clinical notes. 

---
# AITH: A Post-Quantum Continuous Delegation Protocol for Human-AI Trust Establishment 

**Authors**: Zhaoliang Chen  

**Link**: [PDF](https://arxiv.org/pdf/2604.07695)  

**Abstract**: The rapid deployment of AI agents acting autonomously on behalf of human principals has outpaced the development of cryptographic protocols for establishing, bounding, and revoking human-AI trust relationships. Existing frameworks (TLS, OAuth 2.0, Macaroons) assume deterministic software and cannot address probabilistic AI agents operating continuously within variable trust boundaries.
We present AITH (AI Trust Handshake), a post-quantum continuous delegation protocol. AITH introduces: (1) a Continuous Delegation Certificate signed once with ML-DSA-87 (FIPS 204, NIST Level 5), replacing per-operation signing with sub-microsecond boundary checks at 4.7M ops/sec; (2) a six-check Boundary Engine enforcing hard constraints, rate limits, and escalation triggers with zero cryptographic overhead on the critical path; (3) a push-based Revocation Protocol propagating invalidation within one second. A three-tier SHA-256 Responsibility Chain provides tamper-evident audit logging. All five security theorems are machine-verified via Tamarin Prover under the Dolev-Yao model.
We validate AITH through five rounds of multi-model adversarial auditing, resolving 12 vulnerabilities across four severity layers. Simulation of 100,000 operations shows 79.5% autonomous execution, 6.1% human escalation, and 14.4% blocked. 

---
# Joint Task Offloading, Inference Optimization and UAV Trajectory Planning for Generative AI Empowered Intelligent Transportation Digital Twin 

**Authors**: Xiaohuan Li, Junchuan Fan, Bingqi Zhang, Rong Yu, Xumin Huang, Qian Chen  

**Link**: [PDF](https://arxiv.org/pdf/2604.07687)  

**Abstract**: To implement the intelligent transportation digital twin (ITDT), unmanned aerial vehicles (UAVs) are scheduled to process the sensing data from the roadside sensors. At this time, generative artificial intelligence (GAI) technologies such as diffusion models are deployed on the UAVs to transform the raw sensing data into the high-quality and valuable. Therefore, we propose the GAI-empowered ITDT. The dynamic processing of a set of diffusion model inference (DMI) tasks on the UAVs with dynamic mobility simultaneously influences the DT updating fidelity and delay. In this paper, we investigate a joint optimization problem of DMI task offloading, inference optimization and UAV trajectory planning as the system utility maximization (SUM) problem to address the fidelity-delay tradeoff for the GAI-empowered ITDT. To seek a solution to the problem under the network dynamics, we model the SUM problem as the heterogeneous-agent Markov decision process, and propose the sequential update-based heterogeneous-agent twin delayed deep deterministic policy gradient (SU-HATD3) algorithm, which can quickly learn a near-optimal solution. Numerical results demonstrate that compared with several baseline algorithms, the proposed algorithm has great advantages in improving the system utility and convergence rate. 

---
# Reinforcement Learning with LLM-Guided Action Spaces for Synthesizable Lead Optimization 

**Authors**: Tao Li, Kaiyuan Hou, Tuan Vinh, Monika Raj, Zhichun Guo, Carl Yang  

**Link**: [PDF](https://arxiv.org/pdf/2604.07669)  

**Abstract**: Lead optimization in drug discovery requires improving therapeutic properties while ensuring that proposed molecular modifications correspond to feasible synthetic routes. Existing approaches either prioritize property scores without enforcing synthesizability, or rely on expensive enumeration over large reaction networks, while direct application of Large Language Models (LLMs) frequently produces chemically invalid structures. We introduce MolReAct, a framework that formulates lead optimization as a Markov Decision Process over a synthesis-constrained action space defined by validated reaction templates. A tool-augmented LLM agent serves as a dynamic reaction environment that invokes specialized chemical analysis tools to identify reactive sites and propose chemically grounded transformations from matched templates. A policy model trained via Group Relative Policy Optimization (GRPO) selects among these constrained actions to maximize long-term oracle reward across multi-step reaction trajectories. A SMILES-based caching mechanism further reduces end-to-end optimization time by approximately 43%. Across 13 property optimization tasks from the Therapeutic Data Commons and one structure-based docking task, MolReAct achieves an average Top-10 score of 0.563, outperforming the strongest synthesizable baseline by 10.4% in relative improvement, and attains the best sample efficiency on 10 of 14 tasks. Ablations confirm that both tool-augmented reaction proposals and trajectory-level policy optimization contribute complementary gains. By grounding every step in validated reaction templates, MolReAct produces molecules that are property-improved and each accompanied by an explicit synthetic pathway. 

---
# An Imperfect Verifier is Good Enough: Learning with Noisy Rewards 

**Authors**: Andreas Plesner, Francisco Guzmán, Anish Athalye  

**Link**: [PDF](https://arxiv.org/pdf/2604.07666)  

**Abstract**: Reinforcement Learning with Verifiable Rewards (RLVR) has become a prominent method for post-training Large Language Models (LLMs). However, verifiers are rarely error-free; even deterministic checks can be inaccurate, and the growing dependence on model-based judges exacerbates the issue. The extent to which RLVR is robust to such noise and the verifier accuracy required for effective training remain unresolved questions. We investigate these questions in the domains of code generation and scientific reasoning by introducing noise into RL training. Noise rates up to 15% yield peak validation accuracy within 2 percentage points of the clean baseline. These findings are consistent across controlled and model-based noise types, three model families (Qwen3, GLM4, Llama 3.1), and model sizes from 4B to 9B. Overall, the results indicate that imperfect verification does not constitute a fundamental barrier to RLVR. Furthermore, our findings suggest that practitioners should prioritize moderate accuracy with high precision over perfect verification. 

---
# Optimal Decay Spectra for Linear Recurrences 

**Authors**: Yang Cao  

**Link**: [PDF](https://arxiv.org/pdf/2604.07658)  

**Abstract**: Linear recurrent models offer linear-time sequence processing but often suffer from suboptimal long-range memory. We trace this to the decay spectrum: for $N$ channels, random initialization collapses the minimum spectral gap to $O(N^{-2})$, yielding sub-exponential error $\exp(-\Omega(N/\log N))$; linear spacing avoids collapse but degrades to $\exp(-O(N/\sqrt{T}))$, practically algebraic over long contexts. We introduce Position-Adaptive Spectral Tapering (PoST), an architecture-agnostic framework combining two mechanisms: (1) Spectral Reparameterization, which structurally enforces geometrically spaced log-decay rates, proven minimax optimal at rate $O(\exp(-cN/\log T))$; and (2) Position-Adaptive Scaling, the provably unique mechanism that eliminates the scale mismatch of static spectra (where only $N\log t/\log T$ of $N$ channels are effective at position $t$) by stretching the spectrum to the actual dependency range, sharpening the rate to $O(\exp(-cN/\log t))$. This scaling natively induces fractional invariance: the impulse response becomes scale-free, with channels interpolating between relative and absolute temporal coordinates. PoST integrates into any diagonal linear recurrence without overhead. We instantiate it across Mamba-2, RWKV-7, Gated DeltaNet, Gated Linear Attention, and RetNet. Pre-training at 180M-440M scales shows consistent zero-shot language modeling improvements, significant long-context retrieval gains for Mamba-2 (MQAR and NIAH), and competitive or improved performance across other architectures. Code: this https URL. 

---
# Cognitive-Causal Multi-Task Learning with Psychological State Conditioning for Assistive Driving Perception 

**Authors**: Keito Inoshita, Nobuhiro Hayashida, Akira Imanishi  

**Link**: [PDF](https://arxiv.org/pdf/2604.07651)  

**Abstract**: Multi-task learning for advanced driver assistance systems requires modeling the complex interplay between driver internal states and external traffic environments. However, existing methods treat recognition tasks as flat and independent objectives, failing to exploit the cognitive causal structure underlying driving behavior. In this paper, we propose CauPsi, a cognitive science-grounded causal multi-task learning framework that explicitly models the hierarchical dependencies among Traffic Context Recognition (TCR), Vehicle Context Recognition (VCR), Driver Emotion Recognition (DER), and Driver Behavior Recognition (DBR). The proposed framework introduces two key mechanisms. First, a Causal Task Chain propagates upstream task predictions to downstream tasks via learnable prototype embeddings, realizing the cognitive cascade from environmental perception to behavioral regulation in a differentiable manner. Second, Cross-Task Psychological Conditioning (CTPC) estimates a psychological state signal from driver facial expressions and body posture and injects it as a conditioning input to all tasks including environmental recognition, thereby modeling the modulatory effect of driver internal states on cognitive and decision-making processes. Evaluated on the AIDE dataset, CauPsi achieves a mean accuracy of 82.71% with only 5.05M parameters, surpassing prior work by +1.0% overall, with notable improvements on DER (+3.65%) and DBR (+7.53%). Ablation studies validate the independent contribution of each component, and analysis of the psychological state signal confirms that it acquires systematic task-label-dependent patterns in a self-supervised manner without explicit psychological annotations. 

---
# Safe Large-Scale Robust Nonlinear MPC in Milliseconds via Reachability-Constrained System Level Synthesis on the GPU 

**Authors**: Jeffrey Fang, Glen Chou  

**Link**: [PDF](https://arxiv.org/pdf/2604.07644)  

**Abstract**: We present GPU-SLS, a GPU-parallelized framework for safe, robust nonlinear model predictive control (MPC) that scales to high-dimensional uncertain robotic systems and long planning horizons. Our method jointly optimizes an inequality-constrained, dynamically-feasible nominal trajectory, a tracking controller, and a closed-loop reachable set under disturbance, all in real-time. To efficiently compute nominal trajectories, we develop a sequential quadratic programming procedure with a novel GPU-accelerated quadratic program (QP) solver that uses parallel associative scans and adaptive caching within an alternating direction method of multipliers (ADMM) framework. The same GPU QP backend is used to optimize robust tracking controllers and closed-loop reachable sets via system level synthesis (SLS), enabling reachability-constrained control in both fixed- and receding-horizon settings. We achieve substantial performance gains, reducing nominal trajectory solve times by 97.7% relative to state-of-the-art CPU solvers and 71.8% compared to GPU solvers, while accelerating SLS-based control and reachability by 237x. Despite large problem scales, our method achieves 100% empirical safety, unlike high-dimensional learning-based reachability baselines. We validate our approach on complex nonlinear systems, including whole-body quadrupeds (61D) and humanoids (75D), synthesizing robust control policies online on the GPU in 20 milliseconds on average and scaling to problems with 2 x 10^5 decision variables and 8 x 10^4 constraints. The implementation of our method is available at this https URL. 

---
# Exponential quantum advantage in processing massive classical data 

**Authors**: Haimeng Zhao, Alexander Zlokapa, Hartmut Neven, Ryan Babbush, John Preskill, Jarrod R. McClean, Hsin-Yuan Huang  

**Link**: [PDF](https://arxiv.org/pdf/2604.07639)  

**Abstract**: Broadly applicable quantum advantage, particularly in classical data processing and machine learning, has been a fundamental open problem. In this work, we prove that a small quantum computer of polylogarithmic size can perform large-scale classification and dimension reduction on massive classical data by processing samples on the fly, whereas any classical machine achieving the same prediction performance requires exponentially larger size. Furthermore, classical machines that are exponentially larger yet below the required size need superpolynomially more samples and time. We validate these quantum advantages in real-world applications, including single-cell RNA sequencing and movie review sentiment analysis, demonstrating four to six orders of magnitude reduction in size with fewer than 60 logical qubits. These quantum advantages are enabled by quantum oracle sketching, an algorithm for accessing the classical world in quantum superposition using only random classical data samples. Combined with classical shadows, our algorithm circumvents the data loading and readout bottleneck to construct succinct classical models from massive classical data, a task provably impossible for any classical machine that is not exponentially larger than the quantum machine. These quantum advantages persist even when classical machines are granted unlimited time or if BPP=BQP, and rely only on the correctness of quantum mechanics. Together, our results establish machine learning on classical data as a broad and natural domain of quantum advantage and a fundamental test of quantum mechanics at the complexity frontier. 

---
# Sheaf-Laplacian Obstruction and Projection Hardness for Cross-Modal Compatibility on a Modality-Independent Site 

**Authors**: Tibor Sloboda  

**Link**: [PDF](https://arxiv.org/pdf/2604.07632)  

**Abstract**: We develop a unified framework for analyzing cross-modal compatibility in learned representations. The core object is a modality-independent neighborhood site on sample indices, equipped with a cellular sheaf of finite-dimensional real inner-product spaces. For a directed modality pair $(a\to b)$, we formalize two complementary incompatibility mechanisms: projection hardness, the minimal complexity within a nested Lipschitz-controlled projection family needed for a single global map to align whitened embeddings; and sheaf-Laplacian obstruction, the minimal spatial variation required by a locally fit field of projection parameters to achieve a target alignment error. The obstruction invariant is implemented via a projection-parameter sheaf whose 0-Laplacian energy exactly matches the smoothness penalty used in sheaf-regularized regression, making the theory directly operational. This separates two distinct failure modes: hardness failure, where no low-complexity global projection exists, and obstruction failure, where local projections exist but cannot be made globally consistent over the semantic neighborhood graph without large parameter variation. We link the sheaf spectral gap to stability of global alignment, derive bounds relating obstruction energy to excess global-map error under mild Lipschitz assumptions, and give explicit constructions showing that compatibility is generally non-transitive. We further define bridging via composed projection families and show, in a concrete ReLU setting, that an intermediate modality can strictly reduce effective hardness even when direct alignment remains infeasible. 

---
# DIVERSED: Relaxed Speculative Decoding via Dynamic Ensemble Verification 

**Authors**: Ziyi Wang, Siva Rajesh Kasa, Ankith M S, Santhosh Kumar Kasa, Jiaru Zou, Sumit Negi, Ruqi Zhang, Nan Jiang, Qifan Song  

**Link**: [PDF](https://arxiv.org/pdf/2604.07622)  

**Abstract**: Speculative decoding is an effective technique for accelerating large language model inference by drafting multiple tokens in parallel. In practice, its speedup is often bottlenecked by a rigid verification step that strictly enforces the accepted token distribution to exactly match the target model. This constraint leads to the rejection of many plausible tokens, lowering the acceptance rate and limiting overall time speedup. To overcome this limitation, we propose Dynamic Verification Relaxed Speculative Decoding (DIVERSED), a relaxed verification framework that improves time efficiency while preserving generation quality. DIVERSED learns an ensemble-based verifier that blends the draft and target model distributions with a task-dependent and context-dependent weight. We provide theoretical justification for our approach and demonstrate empirically that DIVERSED achieves substantially higher inference efficiency compared to standard speculative decoding methods. Code is available at: this https URL. 

---
# Towards Real-Time Human-AI Musical Co-Performance: Accompaniment Generation with Latent Diffusion Models and MAX/MSP 

**Authors**: Tornike Karchkhadze, Shlomo Dubnov  

**Link**: [PDF](https://arxiv.org/pdf/2604.07612)  

**Abstract**: We present a framework for real-time human-AI musical co-performance, in which a latent diffusion model generates instrumental accompaniment in response to a live stream of context audio. The system combines a MAX/MSP front-end-handling real-time audio input, buffering, and playback-with a Python inference server running the generative model, communicating via OSC/UDP messages. This allows musicians to perform in MAX/MSP - a well-established, real-time capable environment - while interacting with a large-scale Python-based generative model, overcoming the fundamental disconnect between real-time music tools and state-of-the-art AI models. We formulate accompaniment generation as a sliding-window look-ahead protocol, training the model to predict future audio from partial context, where system latency is a critical constraint. To reduce latency, we apply consistency distillation to our diffusion model, achieving a 5.4x reduction in sampling time, with both models achieving real-time operation. Evaluated on musical coherence, beat alignment, and audio quality, both models achieve strong performance in the Retrospective regime and degrade gracefully as look-ahead increases. These results demonstrate the feasibility of diffusion-based real-time accompaniment and expose the fundamental trade-off between model latency, look-ahead depth, and generation quality that any such system must navigate. 

---
# Google, AI Literacy, and the Learning Sciences: Multiple Modes of Research, Industry, and Practice Partnerships 

**Authors**: Victor R. Lee, Michael Madaio, Ben Garside, Aimee Welch, Kristen Pilner Blair, Ibrahim Oluwajoba Adisa, Alon Harris, Kevin Holst, Liat Ben Rafael, Ronit Levavi Morad, Ben Travis, Belle Moller, Andrew Shields, Zak Brown, Lois Hinx, Marisol Diaz, Evan Patton, Selim Tezel, Robert Parks, Hal Abelson, Adam Blasioli, Jeremy Roschelle  

**Link**: [PDF](https://arxiv.org/pdf/2604.07601)  

**Abstract**: Enabling AI literacy in the general population at scale is a complex challenge requiring multiple stakeholders and institutions collaborating together. Industry and technology companies are important actors with respect to AI, and as a field, we have the opportunity to consider how researchers and companies might be partners toward shared goals. In this symposium, we focus on a collection of partnership projects that all involve Google and all address AI literacy as a comparative set of examples. Through a combination of presentations, commentary, and moderated group discussion, the session, we will identify (1) at what points in the life cycle do research, practice, and industry partnerships clearly intersect; (2) what factors and histories shape the directional focus of the partnerships; and (3) where there may be future opportunities for new configurations of partnership that are jointly beneficial to all parties. 

---
# From Ground Truth to Measurement: A Statistical Framework for Human Labeling 

**Authors**: Robert Chew, Stephanie Eckman, Christoph Kern, Frauke Kreuter  

**Link**: [PDF](https://arxiv.org/pdf/2604.07591)  

**Abstract**: Supervised machine learning assumes that labeled data provide accurate measurements of the concepts models are meant to learn. Yet in practice, human labeling introduces systematic variation arising from ambiguous items, divergent interpretations, and simple mistakes. Machine learning research commonly treats all disagreement as noise, which obscures these distinctions and limits our understanding of what models actually learn. This paper reframes annotation as a measurement process and introduces a statistical framework for decomposing labeling outcomes into interpretable sources of variation: instance difficulty, annotator bias, situational noise, and relational alignment. The framework extends classical measurement-error models to accommodate both shared and individualized notions of truth, reflecting traditional and human label variation interpretations of error, and provides a diagnostic for assessing which regime better characterizes a given task. Applying the proposed model to a multi-annotator natural language inference dataset, we find empirical evidence for all four theorized components and demonstrate the effectiveness of our approach. We conclude with implications for data-centric machine learning and outline how this approach can guide the development of a more systematic science of labeling. 

---
# DCD: Domain-Oriented Design for Controlled Retrieval-Augmented Generation 

**Authors**: Valeriy Kovalskiy, Nikita Belov, Nikita Miteyko, Igor Reshetnikov, Max Maximov  

**Link**: [PDF](https://arxiv.org/pdf/2604.07590)  

**Abstract**: Retrieval-Augmented Generation (RAG) is widely used to ground large language models in external knowledge sources. However, when applied to heterogeneous corpora and multi-step queries, Naive RAG pipelines often degrade in quality due to flat knowledge representations and the absence of explicit workflows. In this work, we introduce DCD (Domain-Collection-Document), a domain-oriented design to structure knowledge and control query processing in RAG systems without modifying the underlying language model. The proposed approach relies on a hierarchical decomposition of the information space and multi-stage routing based on structured model outputs, enabling progressive restriction of both retrieval and generation scopes. The architecture is complemented by smart chunking, hybrid retrieval, and integrated validation and generation guardrail mechanisms. We describe the DCD architecture and workflow and discuss evaluation results on synthetic evaluation dataset, highlighting their impact on robustness, factual accuracy, and answer relevance in applied RAG scenarios. 

---
# Don't Measure Once: Measuring Visibility in AI Search (GEO) 

**Authors**: Julius Schulte, Malte Bleeker, Philipp Kaufmann  

**Link**: [PDF](https://arxiv.org/pdf/2604.07585)  

**Abstract**: As large language model-based chat systems become increasingly widely used, generative engine optimization (GEO) has emerged as an important problem for information access and retrieval. In classical search engines, results are comparatively transparent and stable: a single query often provides a representative snapshot of where a page or brand appears relative to competitors. The inherent probabilistic nature of AI search changes this paradigm. Answers can vary across runs, prompts, and time, making one-off observations unreliable. Drawing on empirical studies, our findings underscore the need for repeated measurements to assess a brand's GEO performance and to characterize visibility as a distribution rather than a single-point outcome. 

---
# Learning is Forgetting: LLM Training As Lossy Compression 

**Authors**: Henry C. Conklin, Tom Hosking, Tan Yi-Chern, Julian Gold, Jonathan D. Cohen, Thomas L. Griffiths, Max Bartolo, Seraphina Goldfarb-Tarrant  

**Link**: [PDF](https://arxiv.org/pdf/2604.07569)  

**Abstract**: Despite the increasing prevalence of large language models (LLMs), we still have a limited understanding of how their representational spaces are structured. This limits our ability to interpret how and what they learn or relate them to learning in humans. We argue LLMs are best seen as an instance of lossy compression, where over training they learn by retaining only information in their training data relevant to their objective(s). We show pre-training results in models that are optimally compressed for next-sequence prediction, approaching the Information Bottleneck bound on compression. Across an array of open weights models, each compresses differently, likely due to differences in the data and training recipes used. However even across different families of LLMs the optimality of a model's compression, and the information present in it, can predict downstream performance on across a wide array of benchmarks, letting us directly link representational structure to actionable insights about model performance. In the general case the work presented here offers a unified Information-Theoretic framing for how these models learn that is deployable at scale. 

---
# Reasoning-Based Refinement of Unsupervised Text Clusters with LLMs 

**Authors**: Tunazzina Islam  

**Link**: [PDF](https://arxiv.org/pdf/2604.07562)  

**Abstract**: Unsupervised methods are widely used to induce latent semantic structure from large text collections, yet their outputs often contain incoherent, redundant, or poorly grounded clusters that are difficult to validate without labeled data. We propose a reasoning-based refinement framework that leverages large language models (LLMs) not as embedding generators, but as semantic judges that validate and restructure the outputs of arbitrary unsupervised clustering this http URL framework introduces three reasoning stages: (i) coherence verification, where LLMs assess whether cluster summaries are supported by their member texts; (ii) redundancy adjudication, where candidate clusters are merged or rejected based on semantic overlap; and (iii) label grounding, where clusters are assigned interpretable labels in a fully unsupervised manner. This design decouples representation learning from structural validation and mitigates common failure modes of embedding-only approaches. We evaluate the framework on real-world social media corpora from two platforms with distinct interaction models, demonstrating consistent improvements in cluster coherence and human-aligned labeling quality over classical topic models and recent representation-based baselines. Human evaluation shows strong agreement with LLM-generated labels, despite the absence of gold-standard annotations. We further conduct robustness analyses under matched temporal and volume conditions to assess cross-platform stability. Beyond empirical gains, our results suggest that LLM-based reasoning can serve as a general mechanism for validating and refining unsupervised semantic structure, enabling more reliable and interpretable analyses of large text collections without supervision. 

---
# Generative Experiences for Digital Mental Health Interventions: Evidence from a Randomized Study 

**Authors**: Ananya Bhattacharjee, Michael Liut, Matthew Jörke, Diyi Yang, Emma Brunskill  

**Link**: [PDF](https://arxiv.org/pdf/2604.07558)  

**Abstract**: Digital mental health (DMH) tools have extensively explored personalization of interventions to users' needs and contexts. However, this personalization often targets what support is provided, not how it is experienced. Even well-matched content can fail when the interaction format misaligns with how someone can engage. We introduce generative experience as a paradigm for DMH support, where the intervention experience is composed at runtime. We instantiate this in GUIDE, a system that generates personalized intervention content and multimodal interaction structure through rubric-guided generation of modular components. In a preregistered study with N = 237 participants, GUIDE significantly reduced stress (p = .02) and improved the user experience (p = .04) compared to an LLM-based cognitive restructuring control. GUIDE also supported diverse forms of reflection and action through varied interaction flows, while revealing tensions around personalization across the interaction sequence. This work lays the foundation for interventions that dynamically shape how support is experienced and enacted in digital settings. 

---
# TR-EduVSum: A Turkish-Focused Dataset and Consensus Framework for Educational Video Summarization 

**Authors**: Figen Eğin, Aytuğ Onan  

**Link**: [PDF](https://arxiv.org/pdf/2604.07553)  

**Abstract**: This study presents a framework for generating the gold-standard summary fully automatically and reproducibly based on multiple human summaries of Turkish educational videos. Within the scope of the study, a new dataset called TR-EduVSum was created, encompassing 82 Turkish course videos in the field of "Data Structures and Algorithms" and containing a total of 3281 independent human summaries. Inspired by existing pyramid-based evaluation approaches, the AutoMUP (Automatic Meaning Unit Pyramid) method is proposed, which extracts consensus-based content from multiple human summaries. AutoMUP clusters the meaning units extracted from human summaries using embedding, statistically models inter-participant agreement, and generates graded summaries based on consensus weight. In this framework, the gold summary corresponds to the highest-consensus AutoMUP configuration, constructed from the most frequently supported meaning units across human summaries. Experimental results show that AutoMUP summaries exhibit high semantic overlap with robust LLM (Large Language Model) summaries such as Flash 2.5 and GPT-5.1. Furthermore, ablation studies clearly demonstrate the decisive role of consensus weight and clustering in determining summary quality. The proposed approach can be generalized to other Turkic languages at low cost. 

---
# MCP-DPT: A Defense-Placement Taxonomy and Coverage Analysis for Model Context Protocol Security 

**Authors**: Mehrdad Rostamzadeh, Sidhant Narula, Nahom Birhan, Mohammad Ghasemigol, Daniel Takabi  

**Link**: [PDF](https://arxiv.org/pdf/2604.07551)  

**Abstract**: The Model Context Protocol (MCP) enables large language models (LLMs) to dynamically discover and invoke third-party tools, significantly expanding agent capabilities while introducing a distinct security landscape. Unlike prompt-only interactions, MCP exposes pre-execution artifacts, shared context, multi-turn workflows, and third-party supply chains to adversarial influence across independently operated components. While recent work has identified MCP-specific attacks and evaluated defenses, existing studies are largely attack-centric or benchmark-driven, providing limited guidance on where mitigation responsibility should reside within the MCP architecture. This is problematic given MCP's multi-party design and distributed trust boundaries. We present a defense-placement-oriented security analysis of MCP, introducing a layer-aligned taxonomy that organizes attacks by the architectural component responsible for enforcement. Threats are mapped across six MCP layers, and primary and secondary defense points are identified to support principled defense-in-depth reasoning under adversaries controlling tools, servers, or ecosystem components. A structured mapping of existing academic and industry defenses onto this framework reveals uneven and predominantly tool-centric protection, with persistent gaps at the host orchestration, transport, and supply-chain layers. These findings suggest that many MCP security weaknesses stem from architectural misalignment rather than isolated implementation flaws. 

---
# EMSDialog: Synthetic Multi-person Emergency Medical Service Dialogue Generation from Electronic Patient Care Reports via Multi-LLM Agents 

**Authors**: Xueren Ge, Sahil Murtaza, Anthony Cortez, Homa Alemzadeh  

**Link**: [PDF](https://arxiv.org/pdf/2604.07549)  

**Abstract**: Conversational diagnosis prediction requires models to track evolving evidence in streaming clinical conversations and decide when to commit to a diagnosis. Existing medical dialogue corpora are largely dyadic or lack the multi-party workflow and annotations needed for this setting. We introduce an ePCR-grounded, topic-flow-based multi-agent generation pipeline that iteratively plans, generates, and self-refines dialogues with rule-based factual and topic flow checks. The pipeline yields EMSDialog, a dataset of 4,414 synthetic multi-speaker EMS conversations based on a real-world ePCR dataset, annotated with 43 diagnoses, speaker roles, and turn-level topics. Human and LLM evaluations confirm high quality and realism of EMSDialog using both utterance- and conversation-level metrics. Results show that EMSDialog-augmented training improves accuracy, timeliness, and stability of EMS conversational diagnosis prediction. 

---
# RL-ASL: A Dynamic Listening Optimization for TSCH Networks Using Reinforcement Learning 

**Authors**: F. Fernando Jurado-Lasso, J. F. Jurado  

**Link**: [PDF](https://arxiv.org/pdf/2604.07533)  

**Abstract**: Time Slotted Channel Hopping (TSCH) is a widely adopted Media Access Control (MAC) protocol within the IEEE 802.15.4e standard, designed to provide reliable and energy-efficient communication in Industrial Internet of Things (IIoT) networks. However, state-of-the-art TSCH schedulers rely on static slot allocations, resulting in idle listening and unnecessary power consumption under dynamic traffic conditions. This paper introduces RL-ASL, a reinforcement learning-driven adaptive listening framework that dynamically decides whether to activate or skip a scheduled listening slot based on real-time network conditions. By integrating learning-based slot skipping with standard TSCH scheduling, RL-ASL reduces idle listening while preserving synchronization and delivery reliability. Experimental results on the FIT IoT-LAB testbed and Cooja network simulator show that RL-ASL achieves up to 46% lower power consumption than baseline scheduling protocols, while maintaining near-perfect reliability and reducing average latency by up to 96% compared to PRIL-M. Its link-based variant, RL-ASL-LB, further improves delay performance under high contention with similar energy efficiency. Importantly, RL-ASL performs inference on constrained motes with negligible overhead, as model training is fully performed offline. Overall, RL-ASL provides a practical, scalable, and energy-aware scheduling mechanism for next-generation low-power IIoT networks. 

---
# The Shrinking Lifespan of LLMs in Science 

**Authors**: Ana Trišović  

**Link**: [PDF](https://arxiv.org/pdf/2604.07530)  

**Abstract**: Scaling laws describe how language model capabilities grow with compute and data, but say nothing about how long a model matters once released. We provide the first large-scale empirical account of how scientists adopt and abandon language models over time. We track 62 LLMs across over 108k citing papers (2018-2025), each with at least three years of post-release data, and classify every citation as active adoption or background reference to construct per-model adoption trajectories that raw citation counts cannot resolve. We find three regularities. First, scientific adoption follows an inverted-U trajectory: usage rises after release, peaks, and declines as newer models appear, a pattern we term the \textit{scientific adoption curve}. Second, this curve is compressing: each additional release year is associated with a 27\% reduction in time-to-peak adoption ($p < 0.001$), robust to minimum-age thresholds and controls for model size. Third, release timing dominates model-level attributes as a predictor of lifecycle dynamics. Release year explains both time-to-peak and scientific lifespan more strongly than architecture, openness, or scale, though model size and access modality retain modest predictive power for total adoption volume. Together, these findings complement scaling laws with adoption-side regularities and suggest that the forces driving rapid capability progress may be the same forces compressing scientific relevance. 

---
# SYN-DIGITS: A Synthetic Control Framework for Calibrated Digital Twin Simulation 

**Authors**: Grace Jiarui Fan, Chengpiao Huang, Tianyi Peng, Kaizheng Wang, Yuhang Wu  

**Link**: [PDF](https://arxiv.org/pdf/2604.07513)  

**Abstract**: AI-based persona simulation -- often referred to as digital twin simulation -- is increasingly used for market research, recommender systems, and social sciences. Despite their flexibility, large language models (LLMs) often exhibit systematic bias and miscalibration relative to real human behavior, limiting their reliability. Inspired by synthetic control methods from causal inference, we propose SYN-DIGITS (SYNthetic Control Framework for Calibrated DIGItal Twin Simulation), a principled and lightweight calibration framework that learns latent structure from digital-twin responses and transfers it to align predictions with human ground truth. SYN-DIGITS operates as a post-processing layer on top of any LLM-based simulator and thus is model-agnostic. We develop a latent factor model that formalizes when and why calibration succeeds through latent space alignment conditions, and we systematically evaluate ten calibration methods across thirteen persona constructions, three LLMs, and two datasets. SYN-DIGITS supports both individual-level and distributional simulation for previously unseen questions and unobserved populations, with provable error guarantees. Experiments show that SYN-DIGITS achieves up to 50% relative improvements in individual-level correlation and 50--90% relative reductions in distributional discrepancy compared to uncalibrated baselines. 

---
# Beyond Human-Readable: Rethinking Software Engineering Conventions for the Agentic Development Era 

**Authors**: Dmytro Ustynov  

**Link**: [PDF](https://arxiv.org/pdf/2604.07502)  

**Abstract**: For six decades, software engineering principles have been optimized for a single consumer: the human developer. The rise of agentic AI development, where LLM-based agents autonomously read, write, navigate, and debug codebases, introduces a new primary consumer with fundamentally different constraints. This paper presents a systematic analysis of human-centric conventions under agentic pressure and proposes a key design principle: semantic density optimization, eliminating tokens that carry zero information while preserving tokens that carry high semantic value. We validate this principle through a controlled experiment on log format token economy across four conditions (human-readable, structured, compressed, and tool-assisted compressed), demonstrating a counterintuitive finding: aggressive compression increased total session cost by 67% despite reducing input tokens by 17%, because it shifted interpretive burden to the model's reasoning phase. We extend this principle to propose the rehabilitation of classical anti-patterns, introduce the program skeleton concept for agentic code navigation, and argue for a fundamental decoupling of semantic intent from human-readable representation. 

---
# Triage: Routing Software Engineering Tasks to Cost-Effective LLM Tiers via Code Quality Signals 

**Authors**: Lech Madeyski  

**Link**: [PDF](https://arxiv.org/pdf/2604.07494)  

**Abstract**: Context: AI coding agents route every task to a single frontier large language model (LLM), paying premium inference cost even when many tasks are routine.
Objectives: We propose Triage, a framework that uses code health metrics -- indicators of software maintainability -- as a routing signal to assign each task to the cheapest model tier whose output passes the same verification gate as the expensive model.
Methods: Triage defines three capability tiers (light, standard, heavy -- mirroring, e.g., Haiku, Sonnet, Opus) and routes tasks based on pre-computed code health sub-factors and task metadata. We design an evaluation comparing three routing policies on SWE-bench Lite (300 tasks across three model tiers): heuristic thresholds, a trained ML classifier, and a perfect-hindsight oracle.
Results: We analytically derived two falsifiable conditions under which the tier-dependent asymmetry (medium LLMs benefit from clean code while frontier models do not) yields cost-effective routing: the light-tier pass rate on healthy code must exceed the inter-tier cost ratio, and code health must discriminate the required model tier with at least a small effect size ($\hat{p} \geq 0.56$).
Conclusion: Triage transforms a diagnostic code quality metric into an actionable model-selection signal. We present a rigorous evaluation protocol to test the cost--quality trade-off and identify which code health sub-factors drive routing decisions. 

---
# Cluster Attention for Graph Machine Learning 

**Authors**: Oleg Platonov, Liudmila Prokhorenkova  

**Link**: [PDF](https://arxiv.org/pdf/2604.07492)  

**Abstract**: Message Passing Neural Networks have recently become the most popular approach to graph machine learning tasks; however, their receptive field is limited by the number of message passing layers. To increase the receptive field, Graph Transformers with global attention have been proposed; however, global attention does not take into account the graph topology and thus lacks graph-structure-based inductive biases, which are typically very important for graph machine learning tasks. In this work, we propose an alternative approach: cluster attention (CLATT). We divide graph nodes into clusters with off-the-shelf graph community detection algorithms and let each node attend to all other nodes in each cluster. CLATT provides large receptive fields while still having strong graph-structure-based inductive biases. We show that augmenting Message Passing Neural Networks or Graph Transformers with CLATT significantly improves their performance on a wide range of graph datasets including datasets from the recently introduced GraphLand benchmark representing real-world applications of graph machine learning. 

---
# Enabling Intrinsic Reasoning over Dense Geospatial Embeddings with DFR-Gemma 

**Authors**: Xuechen Zhang, Aviv Slobodkin, Joydeep Paul, Mandar Sharma, Samet Oymak, Shravya Shetty, Gautam Prasad  

**Link**: [PDF](https://arxiv.org/pdf/2604.07490)  

**Abstract**: Representation learning for geospatial and spatio-temporal data plays a critical role in enabling general-purpose geospatial intelligence. Recent geospatial foundation models, such as the Population Dynamics Foundation Model (PDFM), encode complex population and mobility dynamics into compact embeddings. However, their integration with Large Language Models (LLMs) remains limited. Existing approaches to LLM integration treat these embeddings as retrieval indices or convert them into textual descriptions for reasoning, introducing redundancy, token inefficiency, and numerical inaccuracies. We propose Direct Feature Reasoning-Gemma (DFR-Gemma), a novel framework that enables LLMs to reason directly over dense geospatial embeddings. DFR aligns high-dimensional embeddings with the latent space of an LLM via a lightweight projector, allowing embeddings to be injected as semantic tokens alongside natural language instructions. This design eliminates the need for intermediate textual representations and enables intrinsic reasoning over spatial features. To evaluate this paradigm, we introduce a multi-task geospatial benchmark that pairs embeddings with diverse question-answer tasks, including feature querying, comparison, and semantic description. Experimental results show that DFR allows LLMs to decode latent spatial patterns and perform accurate zero-shot reasoning across tasks, while significantly improving efficiency compared to text-based baselines. Our results demonstrate that treating embeddings as primary data inputs, provides a more direct, efficient, and scalable approach to multimodal geospatial intelligence. 

---
# Private Seeds, Public LLMs: Realistic and Privacy-Preserving Synthetic Data Generation 

**Authors**: Qian Ma, Sarah Rajtmajer  

**Link**: [PDF](https://arxiv.org/pdf/2604.07486)  

**Abstract**: Large language models (LLMs) have emerged as a powerful tool for synthetic data generation. A particularly important use case is producing synthetic replicas of private text, which requires carefully balancing privacy and utility. We propose Realistic and Privacy-Preserving Synthetic Data Generation (RPSG), which leverages privacy-preserving mechanisms, including formal differential privacy (DP); and private seeds, in particular text containing personal information, to generate realistic synthetic data. Comprehensive experiments against state-of-the-art private synthetic data generation methods demonstrate that RPSG achieves high fidelity to private data while providing strong privacy protection. 

---
# Active Reward Machine Inference From Raw State Trajectories 

**Authors**: Mohamad Louai Shehab, Antoine Aspeel, Necmiye Ozay  

**Link**: [PDF](https://arxiv.org/pdf/2604.07480)  

**Abstract**: Reward machines are automaton-like structures that capture the memory required to accomplish a multi-stage task. When combined with reinforcement learning or optimal control methods, they can be used to synthesize robot policies to achieve such tasks. However, specifying a reward machine by hand, including a labeling function capturing high-level features that the decisions are based on, can be a daunting task. This paper deals with the problem of learning reward machines directly from raw state and policy information. As opposed to existing works, we assume no access to observations of rewards, labels, or machine nodes, and show what trajectory data is sufficient for learning the reward machine in this information-scarce regime. We then extend the result to an active learning setting where we incrementally query trajectory extensions to improve data (and indirectly computational) efficiency. Results are demonstrated with several grid world examples. 

---
# When Switching Algorithms Helps: A Theoretical Study of Online Algorithm Selection 

**Authors**: Denis Antipov, Carola Doerr  

**Link**: [PDF](https://arxiv.org/pdf/2604.07473)  

**Abstract**: Online algorithm selection (OAS) aims to adapt the optimization process to changes in the fitness landscape and is expected to outperform any single algorithm from a given portfolio. Although this expectation is supported by numerous empirical studies, there are currently no theoretical results proving that OAS can yield asymptotic speedups (apart from some artificial examples for hyper-heuristics). Moreover, theory-based guidelines for when and how to switch between algorithms are largely missing.
In this paper, we present the first theoretical example in which switching between two algorithms -- the $(1+\lambda)$ EA and the $(1+(\lambda,\lambda))$ GA -- solves the OneMax problem asymptotically faster than either algorithm used in isolation. We show that an appropriate choice of population sizes for the two algorithms allows the optimum to be reached in $O(n\log\log n)$ expected time, faster than the $\Theta(n\sqrt{\frac{\log n \log\log\log n}{\log\log n}})$ runtime of the best of these two algorithms with optimally tuned parameters.
We first establish this bound under an idealized switching rule that changes from the $(1+\lambda)$ to the $(1+(\lambda,\lambda))$ GA at the optimal time. We then propose a realistic switching strategy that achieves the same performance. Our analysis combines fixed-start and fixed-target perspectives, illustrating how different algorithms dominate at different stages of the optimization process. This approach offers a promising path toward a deeper theoretical understanding of OAS. 

---
# CMP: Robust Whole-Body Tracking for Loco-Manipulation via Competence Manifold Projection 

**Authors**: Ziyang Cheng, Haoyu Wei, Hang Yin, Xiuwei Xu, Bingyao Yu, Jie Zhou, Jiwen Lu  

**Link**: [PDF](https://arxiv.org/pdf/2604.07457)  

**Abstract**: While decoupled control schemes for legged mobile manipulators have shown robustness, learning holistic whole-body control policies for tracking global end-effector poses remains fragile against Out-of-Distribution (OOD) inputs induced by sensor noise or infeasible user commands. To improve robustness against these perturbations without sacrificing task performance and continuity, we propose Competence Manifold Projection (CMP). Specifically, we utilize a Frame-Wise Safety Scheme that transforms the infinite-horizon safety constraint into a computationally efficient single-step manifold inclusion. To instantiate this competence manifold, we employ a Lower-Bounded Safety Estimator that distinguishes unmastered intentions from the training distribution. We then introduce an Isomorphic Latent Space (ILS) that aligns manifold geometry with safety probability, enabling efficient O(1) seamless defense against arbitrary OOD intents. Experiments demonstrate that CMP achieves up to a 10-fold survival rate improvement in typical OOD scenarios where baselines suffer catastrophic failure, incurring under 10% tracking degradation. Notably, the system exhibits emergent ``best-effort'' generalization behaviors to progressively accomplish OOD goals by adhering to the competence boundaries. Result videos are available at: this https URL. 

---
# GameWorld: Towards Standardized and Verifiable Evaluation of Multimodal Game Agents 

**Authors**: Mingyu Ouyang, Siyuan Hu, Kevin Qinghong Lin, Hwee Tou Ng, Mike Zheng Shou  

**Link**: [PDF](https://arxiv.org/pdf/2604.07429)  

**Abstract**: Towards an embodied generalist for real-world interaction, Multimodal Large Language Model (MLLM) agents still suffer from challenging latency, sparse feedback, and irreversible mistakes. Video games offer an ideal testbed with rich visual observations and closed-loop interaction, demanding fine-grained perception, long-horizon planning, and precise control. However, systematically evaluating these capabilities is currently hindered by heterogeneous action interfaces and heuristic verification. To this end, we introduce GameWorld, a benchmark designed for standardized and verifiable evaluation of MLLMs as generalist game agents in browser environments. Two game agent interfaces are studied: (i) computer-use agents that directly emit keyboard and mouse controls, and (ii) generalist multimodal agents that act in a semantic action space via deterministic Semantic Action Parsing. GameWorld contains 34 diverse games and 170 tasks, each paired with state-verifiable metrics for outcome-based evaluation. The results across 18 model-interface pairs suggest that even the best performing agent is far from achieving human capabilities on video games. Extensive experiments of repeated full-benchmark reruns demonstrate the robustness of the benchmark, while further studies on real-time interaction, context-memory sensitivity, and action validity expose more challenges ahead for game agents. Together, by offering a standardized, verifiable, and reproducible evaluation framework, GameWorld lays a robust foundation for advancing research on multimodal game agents and beyond. The project page is at this https URL. 

---
# Regret-Aware Policy Optimization: Environment-Level Memory for Replay Suppression under Delayed Harm 

**Authors**: Prakul Sunil Hiremath  

**Link**: [PDF](https://arxiv.org/pdf/2604.07428)  

**Abstract**: Safety in reinforcement learning (RL) is typically enforced through objective shaping while keeping environment dynamics stationary with respect to observable state-action pairs. Under delayed harm, this can lead to replay: after a washout period, reintroducing the same stimulus under matched observable conditions reproduces a similar harmful cascade.
We introduce the Replay Suppression Diagnostic (RSD), a controlled exposure-decay-replay protocol that isolates this failure mode under frozen-policy evaluation. We show that, under stationary observable transition kernels, replay cannot be structurally suppressed without inducing a persistent shift in replay-time action distributions.
Motivated by platform-mediated systems, we propose Regret-Aware Policy Optimization (RAPO), which augments the environment with persistent harm-trace and scar fields and applies a bounded, mass-preserving transition reweighting to reduce reachability of historically harmful regions.
On graph diffusion tasks (50-1000 nodes), RAPO suppresses replay, reducing re-amplification gain (RAG) from 0.98 to 0.33 on 250-node graphs while retaining 82\% of task return. Disabling transition deformation only during replay restores re-amplification (RAG 0.91), isolating environment-level deformation as the causal mechanism. 

---
# GIRL: Generative Imagination Reinforcement Learning via Information-Theoretic Hallucination Control 

**Authors**: Prakul Sunil Hiremath  

**Link**: [PDF](https://arxiv.org/pdf/2604.07426)  

**Abstract**: Model-based reinforcement learning (MBRL) improves sample efficiency by optimizing policies inside imagined rollouts, but long-horizon planning degrades when model errors compound and imagined trajectories drift off the training manifold. We introduce GIRL (Generative Imagination Reinforcement Learning), a latent world-model framework that addresses this failure mode with two key components. First, a cross-modal grounding signal derived from a frozen foundation model (DINOv2) anchors the latent transition prior to a semantically consistent embedding space, penalizing inconsistent or implausible predictions. Second, an uncertainty-adaptive trust-region bottleneck interprets the KL regularizer as the Lagrange multiplier of a constrained optimization problem, restricting imagination drift within a learned region calibrated by Expected Information Gain and a Relative Performance Loss signal.
We re-derive a value-gap bound using the Performance Difference Lemma and Integral Probability Metrics, yielding a bound that remains informative as the discount factor approaches one and connects the objective to real-environment regret. Experiments across three benchmark suites, including DeepMind Control, Adroit Hand Manipulation, and Meta-World with visual distractors, show that GIRL reduces latent rollout drift by 38 to 61 percent across tasks relative to DreamerV3, improves asymptotic return, and requires fewer environment interactions on long-horizon tasks. GIRL also outperforms TD-MPC2 on sparse-reward and high-contact settings under standard evaluation metrics. A distilled-prior variant reduces inference overhead and improves computational efficiency relative to the full model. 

---
# SubSearch: Intermediate Rewards for Unsupervised Guided Reasoning in Complex Retrieval 

**Authors**: Roxana Petcu, Evangelos Kanoulas, Maarten de Rijke  

**Link**: [PDF](https://arxiv.org/pdf/2604.07415)  

**Abstract**: Large language models (LLMs) are probabilistic in nature and perform more reliably when augmented with external information. As complex queries often require multi-step reasoning over the retrieved information, with no clear or predetermined reasoning path, they remain challenging. Recent approaches train models using reinforcement learning on the model's outcome, showing promise in improving how models handle complex information. We introduce SubSearch, a specialized framework that shifts from outcome-only supervision to intermediate reward signals that incentivize planning high-quality reasoning. Unlike previous work on process reward modeling, which focuses on training a separate reward model with annotated trajectories by either human annotators or large LLM judges, SubSearch directly optimizes the generator using intrinsic process rewards, which we define as internally-derived rewards, eliminating the need for external supervision, and moving towards autonomous information-intensive reasoning. Experiments on seven benchmarks show that rewarding intermediate reasoning steps with intrinsic rewards leads to more robust reasoning traces in both QA and multi-hop QA datasets over using only outcome rewards. SubSearch can help in building reasoning traces that allow agents to better integrate search engines for complex query answering, while offering a data-efficient alternative to supervised process modeling. 

---
# FORGE:Fine-grained Multimodal Evaluation for Manufacturing Scenarios 

**Authors**: Xiangru Jian, Hao Xu, Wei Pang, Xinjian Zhao, Chengyu Tao, Qixin Zhang, Xikun Zhang, Chao Zhang, Guanzhi Deng, Alex Xue, Juan Du, Tianshu Yu, Garth Tarr, Linqi Song, Qiuzhuang Sun, Dacheng Tao  

**Link**: [PDF](https://arxiv.org/pdf/2604.07413)  

**Abstract**: The manufacturing sector is increasingly adopting Multimodal Large Language Models (MLLMs) to transition from simple perception to autonomous execution, yet current evaluations fail to reflect the rigorous demands of real-world manufacturing environments. Progress is hindered by data scarcity and a lack of fine-grained domain semantics in existing datasets. To bridge this gap, we introduce FORGE. Wefirst construct a high-quality multimodal dataset that combines real-world 2D images and 3D point clouds, annotated with fine-grained domain semantics (e.g., exact model numbers). We then evaluate 18 state-of-the-art MLLMs across three manufacturing tasks, namely workpiece verification, structural surface inspection, and assembly verification, revealing significant performance gaps. Counter to conventional understanding, the bottleneck analysis shows that visual grounding is not the primary limiting factor. Instead, insufficient domain-specific knowledge is the key bottleneck, setting a clear direction for future research. Beyond evaluation, we show that our structured annotations can serve as an actionable training resource: supervised fine-tuning of a compact 3B-parameter model on our data yields up to 90.8% relative improvement in accuracy on held-out manufacturing scenarios, providing preliminary evidence for a practical pathway toward domain-adapted manufacturing MLLMs. The code and datasets are available at this https URL. 

---
# Reinforcement Learning with Reward Machines for Sleep Control in Mobile Networks 

**Authors**: Kristina Levina, Nikolaos Pappas, Athanasios Karapantelakis, Aneta Vulgarakis Feljan, Jendrik Seipp  

**Link**: [PDF](https://arxiv.org/pdf/2604.07411)  

**Abstract**: Energy efficiency in mobile networks is crucial for sustainable telecommunications infrastructure, particularly as network densification continues to increase power consumption. Sleep mechanisms for the components in mobile networks can reduce energy use, but deciding which components to put to sleep, when, and for how long while preserving quality of service (QoS) remains a difficult optimisation problem. In this paper, we utilise reinforcement learning with reward machines (RMs) to make sleep-control decisions that balance immediate energy savings and long-term QoS impact, i.e. time-averaged packet drop rates for deadline-constrained traffic and time-averaged minimum-throughput guarantees for constant-rate users. A challenge is that time-averaged constraints depend on cumulative performance over time rather than immediate performance. As a result, the effective reward is non-Markovian, and optimal actions depend on operational history rather than the instantaneous system state. RMs account for the history dependence by maintaining an abstract state that explicitly tracks the QoS constraint violations over time. Our framework provides a principled, scalable approach to energy management for next-generation mobile networks under diverse traffic patterns and QoS requirements. 

---
# Conservation Law Breaking at the Edge of Stability: A Spectral Theory of Non-Convex Neural Network Optimization 

**Authors**: Daniel Nobrega Medeiros  

**Link**: [PDF](https://arxiv.org/pdf/2604.07405)  

**Abstract**: Why does gradient descent reliably find good solutions in non-convex neural network optimization, despite the landscape being NP-hard in the worst case? We show that gradient flow on L-layer ReLU networks without bias preserves L-1 conservation laws C_l = ||W_{l+1}||_F^2 - ||W_l||_F^2, confining trajectories to lower-dimensional manifolds. Under discrete gradient descent, these laws break with total drift scaling as eta^alpha where alpha is approximately 1.1-1.6 depending on architecture, loss function, and width. We decompose this drift exactly as eta^2 * S(eta), where the gradient imbalance sum S(eta) admits a closed-form spectral crossover formula with mode coefficients c_k proportional to e_k(0)^2 * lambda_{x,k}^2, derived from first principles and validated for both linear (R=0.85) and ReLU (R>0.80) networks. For cross-entropy loss, softmax probability concentration drives exponential Hessian spectral compression with timescale tau = Theta(1/eta) independent of training set size, explaining why cross-entropy self-regularizes the drift exponent near alpha=1.0. We identify two dynamical regimes separated by a width-dependent transition: a perturbative sub-Edge-of-Stability regime where the spectral formula applies, and a non-perturbative regime with extensive mode coupling. All predictions are validated across 23 experiments. 

---
# Breaking the Illusion of Identity in LLM Tooling 

**Authors**: Marek Miller  

**Link**: [PDF](https://arxiv.org/pdf/2604.07398)  

**Abstract**: Large language models (LLMs) in research and development toolchains produce output that triggers attribution of agency and understanding -- a cognitive illusion that degrades verification behavior and trust calibration. No existing mitigation provides a systematic, deployable constraint set for output register. This paper proposes seven output-side rules, each targeting a documented linguistic mechanism, and validates them empirically. In 780 two-turn conversations (constrained vs. default register, 30 tasks, 13 replicates, 1560 API calls), anthropomorphic markers dropped from 1233 to 33 (>97% reduction, p < 0.001), outputs were 49% shorter by word count, and adapted AnthroScore confirmed the shift toward machine register (-1.94 vs. -0.96, p < 0.001). The rules are implemented as a configuration-file system prompt requiring no model modification; validation uses a single model (Claude Sonnet 4). Output quality under the constrained register was not evaluated. The mechanism is extensible to other domains. 

---
# Data Warmup: Complexity-Aware Curricula for Efficient Diffusion Training 

**Authors**: Jinhong Lin, Pan Wang, Zitong Zhan, Lin Zhang, Pedro Morgado  

**Link**: [PDF](https://arxiv.org/pdf/2604.07397)  

**Abstract**: A key inefficiency in diffusion training occurs when a randomly initialized network, lacking visual priors, encounters gradients from the full complexity spectrum--most of which it lacks the capacity to resolve. We propose Data Warmup, a curriculum strategy that schedules training images from simple to complex without modifying the model or loss. Each image is scored offline by a semantic-aware complexity metric combining foreground dominance (how much of the image salient objects occupy) and foreground typicality (how closely the salient content matches learned visual prototypes). A temperature-controlled sampler then prioritizes low-complexity images early and anneals toward uniform sampling. On ImageNet 256x256 with SiT backbones (S/2 to XL/2), Data Warmup improves IS by up to 6.11 and FID by up to 3.41, reaching baseline quality tens of thousands of iterations earlier. Reversing the curriculum (exposing hard images first) degrades performance below the uniform baseline, confirming that the simple-to-complex ordering itself drives the gains. The method combines with orthogonal accelerators such as REPA and requires only ~10 minutes of one-time preprocessing with zero per-iteration overhead. 

---
# A Physical Agentic Loop for Language-Guided Grasping with Execution-State Monitoring 

**Authors**: Wenze Wang, Mehdi Hosseinzadeh, Feras Dayoub  

**Link**: [PDF](https://arxiv.org/pdf/2604.07395)  

**Abstract**: Robotic manipulation systems that follow language instructions often execute grasp primitives in a largely single-shot manner: a model proposes an action, the robot executes it, and failures such as empty grasps, slips, stalls, timeouts, or semantically wrong grasps are not surfaced to the decision layer in a structured way. Inspired by agentic loops in digital tool-using agents, we reformulate language-guided grasping as a bounded embodied agent operating over grounded execution states, where physical actions expose an explicit tool-state stream. We introduce a physical agentic loop that wraps an unmodified learned manipulation primitive (grasp-and-lift) with (i) an event-based interface and (ii) an execution monitoring layer, Watchdog, which converts noisy gripper telemetry into discrete outcome labels using contact-aware fusion and temporal stabilization. These outcome events, optionally combined with post-grasp semantic verification, are consumed by a deterministic bounded policy that finalizes, retries, or escalates to the user for clarification, guaranteeing finite termination. We validate the resulting loop on a mobile manipulator with an eye-in-hand D405 camera, keeping the underlying grasp model unchanged and evaluating representative scenarios involving visual ambiguity, distractors, and induced execution failures. Results show that explicit execution-state monitoring and bounded recovery enable more robust and interpretable behavior than open-loop execution, while adding minimal architectural overhead. For the source code and demo refer to our project page: this https URL 

---
# DSPR: Dual-Stream Physics-Residual Networks for Trustworthy Industrial Time Series Forecasting 

**Authors**: Yeran Zhang, Pengwei Yang, Guoqing Wang, Tianyu Li  

**Link**: [PDF](https://arxiv.org/pdf/2604.07393)  

**Abstract**: Accurate forecasting of industrial time series requires balancing predictive accuracy with physical plausibility under non-stationary operating conditions. Existing data-driven models often achieve strong statistical performance but struggle to respect regime-dependent interaction structures and transport delays inherent in real-world systems. To address this challenge, we propose DSPR (Dual-Stream Physics-Residual Networks), a forecasting framework that explicitly decouples stable temporal patterns from regime-dependent residual dynamics. The first stream models the statistical temporal evolution of individual variables. The second stream focuses on residual dynamics through two key mechanisms: an Adaptive Window module that estimates flow-dependent transport delays, and a Physics-Guided Dynamic Graph that incorporates physical priors to learn time-varying interaction structures while suppressing spurious correlations. Experiments on four industrial benchmarks spanning heterogeneous regimes demonstrate that DSPR consistently improves forecasting accuracy and robustness under regime shifts while maintaining strong physical plausibility. It achieves state-of-the-art predictive performance, with Mean Conservation Accuracy exceeding 99% and Total Variation Ratio reaching up to 97.2%. Beyond forecasting, the learned interaction structures and adaptive lags provide interpretable insights that are consistent with known domain mechanisms, such as flow-dependent transport delays and wind-to-power scaling behaviors. These results suggest that architectural decoupling with physics-consistent inductive biases offers an effective path toward trustworthy industrial time-series forecasting. Furthermore, DSPR's demonstrated robust performance in long-term industrial deployment bridges the gap between advanced forecasting models and trustworthy autonomous control systems. 

---
# Self-Calibrating LLM-Based Analog Circuit Sizing with Interpretable Design Equations 

**Authors**: Antonio J. Bujana, Aydin I. Karsilayan  

**Link**: [PDF](https://arxiv.org/pdf/2604.07387)  

**Abstract**: We present a self-calibrating framework for analog circuit sizing in which a large language model (LLM) derives topology-specific analytical design equations directly from a raw circuit netlist. Unlike existing AI-driven sizing methods where the model proposes parameter adjustments or reduces a search space, the LLM produces a complete Python sizing function tracing each device dimension to a specific performance constraint. A deterministic calibration loop extracts process-dependent parameters from a single transistor-level simulation, while a prediction-error feedback mechanism compensates for analytical inaccuracies. We validate the framework on six operational transconductance amplifier (OTA) topologies spanning three families at two process nodes (180 nm and 40 nm CMOS). All 12 topology-node combinations achieve all specifications, converging in 2-9 simulations for 11 of 12 cases, with one outlier requiring 16 simulations due to an extremely narrow feasible region. Despite large initial prediction errors, convergence depends on the measurement-feedback architecture, not prediction accuracy. This one-shot calibration automatically captures process-dependent variations, enabling cross-node portability without modification, retraining, or per-process characterization. 

---
# Playing DOOM with 1.3M Parameters: Specialized Small Models vs Large Language Models for Real-Time Game Control 

**Authors**: David Golchinfar, Daryoush Vaziri, Alexander Marquardt  

**Link**: [PDF](https://arxiv.org/pdf/2604.07385)  

**Abstract**: We present SauerkrautLM-Doom-MultiVec, a 1.3 million parameter model that plays the classic first-person shooter DOOM in real time, outperforming large language models up to 92,000x its size, including Nemotron-120B, Qwen3.5-27B, and GPT-4o-mini. Our model combines a ModernBERT encoder with hash embeddings, depth-aware token representations, and an attention pooling classification head to select game actions from ASCII frame representations at 31ms per decision. Trained on just 31,000 human gameplay demonstrations, it achieves 178 frags in 10 episodes (17.8 per episode) in the defend_the_center scenario, more than all tested LLMs combined (13 frags total). All agents receive equivalent input: ASCII frames and depth maps. Despite having 92,000x fewer parameters than Nemotron-120B, our model is the only agent that actively engages enemies rather than purely evading them. These results demonstrate that small, task-specific models trained on domain-appropriate data can decisively outperform general-purpose LLMs at real-time control tasks, at a fraction of the inference cost, with deployment capability on consumer hardware. 

---
# Decisions and Deployment: The Five-Year SAHELI Project (2020-2025) on Restless Multi-Armed Bandits for Improving Maternal and Child Health 

**Authors**: Shresth Verma, Arpan Dasgupta, Neha Madhiwalla, Aparna Taneja, Milind Tambe  

**Link**: [PDF](https://arxiv.org/pdf/2604.07384)  

**Abstract**: Maternal and child health is a critical concern around the world. In many global health programs disseminating preventive care and health information, limited healthcare worker resources prevent continuous, personalised engagement with vulnerable beneficiaries. In such scenarios, it becomes crucial to optimally schedule limited live-service resources to maximise long-term engagement. To address this fundamental challenge, the multi-year SAHELI project (2020-2025), in collaboration with partner NGO ARMMAN, leverages AI to allocate scarce resources in a maternal and child health program in India. The SAHELI system solves this sequential resource allocation problem using a Restless Multi-Armed Bandit (RMAB) framework. A key methodological innovation is the transition from a traditional Two-Stage "predict-then-optimize" approach to Decision-Focused Learning (DFL), which directly aligns the framework's learning method with the ultimate goal of maximizing beneficiary engagement. Empirical evaluation through large-scale randomized controlled trials demonstrates that the DFL policy reduced cumulative engagement drops by 31% relative to the current standard of care, significantly outperforming the Two-Stage model. Crucially, the studies also confirmed that this increased program engagement translates directly into statistically significant improvements in real-world health behaviors, notably the continued consumption of vital iron and calcium supplements by new mothers. Ultimately, the SAHELI project provides a scalable blueprint for applying sequential decision-making AI to optimize resource allocation in health programs. 

---
# Latent Structure of Affective Representations in Large Language Models 

**Authors**: Benjamin J. Choi, Melanie Weber  

**Link**: [PDF](https://arxiv.org/pdf/2604.07382)  

**Abstract**: The geometric structure of latent representations in large language models (LLMs) is an active area of research, driven in part by its implications for model transparency and AI safety. Existing literature has focused mainly on general geometric and topological properties of the learnt representations, but due to a lack of ground-truth latent geometry, validating the findings of such approaches is challenging. Emotion processing provides an intriguing testbed for probing representational geometry, as emotions exhibit both categorical organization and continuous affective dimensions, which are well-established in the psychology literature. Moreover, understanding such representations carries safety relevance. In this work, we investigate the latent structure of affective representations in LLMs using geometric data analysis tools. We present three main findings. First, we show that LLMs learn coherent latent representations of affective emotions that align with widely used valence--arousal models from psychology. Second, we find that these representations exhibit nonlinear geometric structure that can nonetheless be well-approximated linearly, providing empirical support for the linear representation hypothesis commonly assumed in model transparency methods. Third, we demonstrate that the learned latent representation space can be leveraged to quantify uncertainty in emotion processing tasks. Our findings suggest that LLMs acquire affective representations with geometric structure paralleling established models of human emotion, with practical implications for model interpretability and safety. 

---
# The Role of Emotional Stimuli and Intensity in Shaping Large Language Model Behavior 

**Authors**: Ameen Patel, Felix Lee, Kyle Liang, Joseph Thomas  

**Link**: [PDF](https://arxiv.org/pdf/2604.07369)  

**Abstract**: Emotional prompting - the use of specific emotional diction in prompt engineering - has shown increasing promise in improving large language model (LLM) performance, truthfulness, and responsibility. However these studies have been limited to single types of positive emotional stimuli and have not considered varying degrees of emotion intensity in their analyses. In this paper, we explore the effects of four distinct emotions - joy, encouragement, anger, and insecurity - in emotional prompting and evaluate them on accuracy, sycophancy, and toxicity. We develop a prompt-generation pipeline with GPT-4o mini to create a suite of LLM and human-generated prompts with varying intensities across the four emotions. Then, we compile a "Gold Dataset" of prompts where human and model labels align. Our empirical evaluation on LLM behavior suggests that positive emotional stimuli lead to more accurate and less toxic results, but also increase sycophantic behavior. 

---
# Position Paper: From Edge AI to Adaptive Edge AI 

**Authors**: Fabrizio Pittorino, Manuel Roveri  

**Link**: [PDF](https://arxiv.org/pdf/2604.07360)  

**Abstract**: Edge AI is often framed as model compression and deployment under tight constraints. We argue a stronger operational thesis: Edge AI in realistic deployments is necessarily adaptive. In long-horizon operation, a fixed (non-adaptive) configuration faces a fundamental failure mode: as data and operating conditions evolve and change in time, it must either (i) violate time-varying budgets (latency/energy/thermal/connectivity/privacy) or (ii) lose predictive reliability (accuracy and, critically, calibration), with risk concentrating in transient regimes and rare time intervals rather than in average performance. If a deployed system cannot reconfigure its computation - and, when required, its model state - under evolving conditions and constraints, it reduces to static embedded inference and cannot provide sustained utility. This position paper introduces a minimal Agent-System-Environment (ASE) lens that makes adaptivity precise at the edge by specifying (i) what changes, (ii) what is observed, (iii) what can be reconfigured, and (iv) which constraints must remain satisfied over time. Building on this framing, we formulate ten research challenges for the next decade, spanning theoretical guarantees for evolving systems, dynamic architectures and hybrid transitions between data-driven and model-based components, fault/anomaly-driven targeted updates, System-1/System-2 decompositions (anytime intelligence), modularity, validation under scarce labels, and evaluation protocols that quantify lifecycle efficiency and recovery/stability under drift and interventions. 

---
# Hybrid CNN-Transformer Architecture for Arabic Speech Emotion Recognition 

**Authors**: Youcef Soufiane Gheffari, Oussama Mustapha Benouddane, Samiya Silarbi  

**Link**: [PDF](https://arxiv.org/pdf/2604.07357)  

**Abstract**: Recognizing emotions from speech using machine learning has become an active research area due to its importance in building human-centered applications. However, while many studies have been conducted in English, German, and other European and Asian languages, research in Arabic remains scarce because of the limited availability of annotated datasets. In this paper, we present an Arabic Speech Emotion Recognition (SER) system based on a hybrid CNN-Transformer architecture. The model leverages convolutional layers to extract discriminative spectral features from Mel-spectrogram inputs and Transformer encoders to capture long-range temporal dependencies in speech. Experiments were conducted on the EYASE (Egyptian Arabic speech emotion) corpus, and the proposed model achieved 97.8% accuracy and a macro F1-score of 0.98. These results demonstrate the effectiveness of combining convolutional feature extraction with attention-based modeling for Arabic SER and highlight the potential of Transformer-based approaches in low-resource languages. 

---
# Prediction Arena: Benchmarking AI Models on Real-World Prediction Markets 

**Authors**: Jaden Zhang, Gardenia Liu, Oliver Johansson, Hileamlak Yitayew, Kamryn Ohly, Grace Li  

**Link**: [PDF](https://arxiv.org/pdf/2604.07355)  

**Abstract**: We introduce Prediction Arena, a benchmark for evaluating AI models' predictive accuracy and decision-making by enabling them to trade autonomously on live prediction markets with real capital. Unlike synthetic benchmarks, Prediction Arena tests models in environments where trades execute on actual exchanges (Kalshi and Polymarket), providing objective ground truth that cannot be gamed or overfitted. Each model operates as an independent agent starting with $10,000, making autonomous decisions every 15-45 minutes. Over a 57-day longitudinal evaluation (January 12 to March 9, 2026), we track two cohorts: six frontier models in live trading (Cohort 1, full period) and four next-generation models in paper trading (Cohort 2, 3-day preliminary). For Cohort 1, final Kalshi returns range from -16.0% to -30.8%. Our analysis identifies a clear performance hierarchy: initial prediction accuracy and the ability to capitalize on correct predictions are the main drivers, while research volume shows no correlation with outcomes. A striking cross-platform contrast emerges from parallel Polymarket live trading: Cohort 1 models averaged only -1.1% on Polymarket vs. -22.6% on Kalshi, with grok-4-20-checkpoint achieving a 71.4% settlement win rate - the highest across any platform or cohort. gemini-3.1-pro-preview (Cohort 2), which executed zero trades on Kalshi, achieved +6.02% on Polymarket in 3 days - the best return of any model across either cohort - demonstrating that platform design has a profound effect on which models succeed. Beyond performance, we analyze computational efficiency (token usage, cycle time), settlement accuracy, exit patterns, and market preferences, providing a comprehensive view of how frontier models behave under real financial pressure. 

---
# Contextual Earnings-22: A Speech Recognition Benchmark with Custom Vocabulary in the Wild 

**Authors**: Berkin Durmus, Chen Cen, Eduardo Pacheco, Arda Okan, Atila Orhon  

**Link**: [PDF](https://arxiv.org/pdf/2604.07354)  

**Abstract**: The accuracy frontier of speech-to-text systems has plateaued on academic benchmarks.1 In contrast, industrial benchmarks and adoption in high-stakes domains suggest otherwise. We hypothesize that the primary difference between the two is contextual conditioning: Academic benchmarks are dominated by frequently encountered general vocabulary that is relatively easy to recognize compared with rare and context-defined custom vocabulary that has disproportionate impact on the usability of speech transcripts. Despite progress on contextual speech-to-text, there is no standardized benchmark. We introduce Contextual Earnings-22, an open dataset built upon Earnings-22, with realistic custom vocabulary contexts to foster research and reveal latent progress. We set six strong baselines for two dominant approaches: keyword prompting and keyword boosting. Experiments show both reach comparable and significantly improved accuracy when scaled from proof-of-concept to large-scale systems. 

---
