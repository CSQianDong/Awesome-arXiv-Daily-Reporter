# Unraveling Hidden Representations: A Multi-Modal Layer Analysis for Better Synthetic Content Forensics 

**Authors**: Tom Or, Omri Azencot  

**Link**: [PDF](https://arxiv.org/pdf/2508.00784)  

**Abstract**: Generative models achieve remarkable results in multiple data domains, including images and texts, among other examples. Unfortunately, malicious users exploit synthetic media for spreading misinformation and disseminating deepfakes. Consequently, the need for robust and stable fake detectors is pressing, especially when new generative models appear everyday. While the majority of existing work train classifiers that discriminate between real and fake information, such tools typically generalize only within the same family of generators and data modalities, yielding poor results on other generative classes and data domains. Towards a universal classifier, we propose the use of large pre-trained multi-modal models for the detection of generative content. Effectively, we show that the latent code of these models naturally captures information discriminating real from fake. Building on this observation, we demonstrate that linear classifiers trained on these features can achieve state-of-the-art results across various modalities, while remaining computationally efficient, fast to train, and effective even in few-shot settings. Our work primarily focuses on fake detection in audio and images, achieving performance that surpasses or matches that of strong baseline methods. 

---
# Context-Aware Visualization for Explainable AI Recommendations in Social Media: A Vision for User-Aligned Explanations 

**Authors**: Banan Alkhateeb, Ellis Solaiman  

**Link**: [PDF](https://arxiv.org/pdf/2508.00674)  

**Abstract**: Social media platforms today strive to improve user experience through AI recommendations, yet the value of such recommendations vanishes as users do not understand the reasons behind them. This issue arises because explainability in social media is general and lacks alignment with user-specific needs. In this vision paper, we outline a user-segmented and context-aware explanation layer by proposing a visual explanation system with diverse explanation methods. The proposed system is framed by the variety of user needs and contexts, showing explanations in different visualized forms, including a technically detailed version for AI experts and a simplified one for lay users. Our framework is the first to jointly adapt explanation style (visual vs. numeric) and granularity (expert vs. lay) inside a single pipeline. A public pilot with 30 X users will validate its impact on decision-making and trust. 

---
# Transparent Adaptive Learning via Data-Centric Multimodal Explainable AI 

**Authors**: Maryam Mosleh, Marie Devlin, Ellis Solaiman  

**Link**: [PDF](https://arxiv.org/pdf/2508.00665)  

**Abstract**: Artificial intelligence-driven adaptive learning systems are reshaping education through data-driven adaptation of learning experiences. Yet many of these systems lack transparency, offering limited insight into how decisions are made. Most explainable AI (XAI) techniques focus on technical outputs but neglect user roles and comprehension. This paper proposes a hybrid framework that integrates traditional XAI techniques with generative AI models and user personalisation to generate multimodal, personalised explanations tailored to user needs. We redefine explainability as a dynamic communication process tailored to user roles and learning goals. We outline the framework's design, key XAI limitations in education, and research directions on accuracy, fairness, and personalisation. Our aim is to move towards explainable AI that enhances transparency while supporting user-centred experiences. 

---
# Multi-Band Variable-Lag Granger Causality: A Unified Framework for Causal Time Series Inference across Frequencies 

**Authors**: Chakattrai Sookkongwaree, Tattep Lakmuang, Chainarong Amornbunchornvej  

**Link**: [PDF](https://arxiv.org/pdf/2508.00658)  

**Abstract**: Understanding causal relationships in time series is fundamental to many domains, including neuroscience, economics, and behavioral science. Granger causality is one of the well-known techniques for inferring causality in time series. Typically, Granger causality frameworks have a strong fix-lag assumption between cause and effect, which is often unrealistic in complex systems. While recent work on variable-lag Granger causality (VLGC) addresses this limitation by allowing a cause to influence an effect with different time lags at each time point, it fails to account for the fact that causal interactions may vary not only in time delay but also across frequency bands. For example, in brain signals, alpha-band activity may influence another region with a shorter delay than slower delta-band oscillations. In this work, we formalize Multi-Band Variable-Lag Granger Causality (MB-VLGC) and propose a novel framework that generalizes traditional VLGC by explicitly modeling frequency-dependent causal delays. We provide a formal definition of MB-VLGC, demonstrate its theoretical soundness, and propose an efficient inference pipeline. Extensive experiments across multiple domains demonstrate that our framework significantly outperforms existing methods on both synthetic and real-world datasets, confirming its broad applicability to any type of time series data. Code and datasets are publicly available. 

---
# Multi-Agent Game Generation and Evaluation via Audio-Visual Recordings 

**Authors**: Alexia Jolicoeur-Martineau  

**Link**: [PDF](https://arxiv.org/pdf/2508.00632)  

**Abstract**: While AI excels at generating text, audio, images, and videos, creating interactive audio-visual content such as video games remains challenging. Current LLMs can generate JavaScript games and animations, but lack automated evaluation metrics and struggle with complex content that normally requires teams of humans working for many months (multi-shot, multi-agents) using assets made by artists. To tackle these issues, we built a new metric and a multi-agent system.
We propose AVR-Eval, a relative metric for multimedia content quality using Audio-Visual Recordings (AVRs). An omni-modal model (processing text, video, and audio) compares the AVRs of two contents, with a text model reviewing evaluations to determine superiority. We show that AVR-Eval properly identifies good from broken or mismatched content.
We built AVR-Agent, a multi-agent system generating JavaScript code from a bank of multimedia assets (audio, images, 3D models). The coding agent selects relevant assets, generates multiple initial codes, uses AVR-Eval to identify the best version, and iteratively improves it through omni-modal agent feedback from the AVR.
We run experiments on games and animations with AVR-Eval (win rate of content A against B). We find that content generated by AVR-Agent has a significantly higher win rate against content made through one-shot generation. However, models struggle to leverage custom assets and AVR feedback effectively, showing no higher win rate. This reveals a critical gap: while humans benefit from high-quality assets and audio-visual feedback, current coding models do not seem to utilize these resources as effectively, highlighting fundamental differences between human and machine content creation approaches. 

---
# From EMR Data to Clinical Insight: An LLM-Driven Framework for Automated Pre-Consultation Questionnaire Generation 

**Authors**: Ruiqing Ding, Qianfang Sun, Yongkang Leng, Hui Yin, Xiaojian Li  

**Link**: [PDF](https://arxiv.org/pdf/2508.00581)  

**Abstract**: Pre-consultation is a critical component of effective healthcare delivery. However, generating comprehensive pre-consultation questionnaires from complex, voluminous Electronic Medical Records (EMRs) is a challenging task. Direct Large Language Model (LLM) approaches face difficulties in this task, particularly regarding information completeness, logical order, and disease-level synthesis. To address this issue, we propose a novel multi-stage LLM-driven framework: Stage 1 extracts atomic assertions (key facts with timing) from EMRs; Stage 2 constructs personal causal networks and synthesizes disease knowledge by clustering representative networks from an EMR corpus; Stage 3 generates tailored personal and standardized disease-specific questionnaires based on these structured representations. This framework overcomes limitations of direct methods by building explicit clinical knowledge. Evaluated on a real-world EMR dataset and validated by clinical experts, our method demonstrates superior performance in information coverage, diagnostic relevance, understandability, and generation time, highlighting its practical potential to enhance patient information collection. 

---
# MultiSHAP: A Shapley-Based Framework for Explaining Cross-Modal Interactions in Multimodal AI Models 

**Authors**: Zhanliang Wang, Kai Wang  

**Link**: [PDF](https://arxiv.org/pdf/2508.00576)  

**Abstract**: Multimodal AI models have achieved impressive performance in tasks that require integrating information from multiple modalities, such as vision and language. However, their "black-box" nature poses a major barrier to deployment in high-stakes applications where interpretability and trustworthiness are essential. How to explain cross-modal interactions in multimodal AI models remains a major challenge. While existing model explanation methods, such as attention map and Grad-CAM, offer coarse insights into cross-modal relationships, they cannot precisely quantify the synergistic effects between modalities, and are limited to open-source models with accessible internal weights. Here we introduce MultiSHAP, a model-agnostic interpretability framework that leverages the Shapley Interaction Index to attribute multimodal predictions to pairwise interactions between fine-grained visual and textual elements (such as image patches and text tokens), while being applicable to both open- and closed-source models. Our approach provides: (1) instance-level explanations that reveal synergistic and suppressive cross-modal effects for individual samples - "why the model makes a specific prediction on this input", and (2) dataset-level explanation that uncovers generalizable interaction patterns across samples - "how the model integrates information across modalities". Experiments on public multimodal benchmarks confirm that MultiSHAP faithfully captures cross-modal reasoning mechanisms, while real-world case studies demonstrate its practical utility. Our framework is extensible beyond two modalities, offering a general solution for interpreting complex multimodal AI models. 

---
# Pro2Guard: Proactive Runtime Enforcement of LLM Agent Safety via Probabilistic Model Checking 

**Authors**: Haoyu Wang, Chris M. Poskitt, Jun Sun, Jiali Wei  

**Link**: [PDF](https://arxiv.org/pdf/2508.00500)  

**Abstract**: Large Language Model (LLM) agents exhibit powerful autonomous capabilities across domains such as robotics, virtual assistants, and web automation. However, their stochastic behavior introduces significant safety risks that are difficult to anticipate. Existing rule-based enforcement systems, such as AgentSpec, focus on developing reactive safety rules, which typically respond only when unsafe behavior is imminent or has already occurred. These systems lack foresight and struggle with long-horizon dependencies and distribution shifts. To address these limitations, we propose Pro2Guard, a proactive runtime enforcement framework grounded in probabilistic reachability analysis. Pro2Guard abstracts agent behaviors into symbolic states and learns a Discrete-Time Markov Chain (DTMC) from execution traces. At runtime, it anticipates future risks by estimating the probability of reaching unsafe states, triggering interventions before violations occur when the predicted risk exceeds a user-defined threshold. By incorporating semantic validity checks and leveraging PAC bounds, Pro2Guard ensures statistical reliability while approximating the underlying ground-truth model. We evaluate Pro2Guard extensively across two safety-critical domains: embodied household agents and autonomous vehicles. In embodied agent tasks, Pro2Guard enforces safety early on up to 93.6% of unsafe tasks using low thresholds, while configurable modes (e.g., reflect) allow balancing safety with task success, maintaining up to 80.4% task completion. In autonomous driving scenarios, Pro2Guard achieves 100% prediction of traffic law violations and collisions, anticipating risks up to 38.66 seconds ahead. 

---
# Thinking Machines: Mathematical Reasoning in the Age of LLMs 

**Authors**: Andrea Asperti, Alberto Naibo, Claudio Sacerdoti Coen  

**Link**: [PDF](https://arxiv.org/pdf/2508.00459)  

**Abstract**: Large Language Models (LLMs) have shown remarkable abilities in structured reasoning and symbolic tasks, with coding emerging as a particular area of strength. This success has sparked growing interest in applying LLMs to mathematics, both in informal problem-solving and formal theorem proving. However, progress in formal mathematics has proven to be significantly more difficult, despite surface-level similarities between programming and proof construction. This discrepancy raises important questions about how LLMs ``reason'', how they are supervised, and whether they internally track a notion of computational or deductive state. In this article, we address the state-of-the-art of the discipline, focusing on recent models and benchmarks, and explore three central issues at the intersection of machine learning and mathematical cognition: (i) the trade-offs between formal and informal mathematics as training domains; (ii) the deeper reasons why proof generation remains more brittle than code synthesis; (iii) and the question of whether LLMs represent, or merely mimic, a notion of evolving logical state. Our goal is not to draw hard boundaries, but to identify where the current limits lie, and how they might be extended. 

---
# Cognitive Kernel-Pro: A Framework for Deep Research Agents and Agent Foundation Models Training 

**Authors**: Tianqing Fang, Zhisong Zhang, Xiaoyang Wang, Rui Wang, Can Qin, Yuxuan Wan, Jun-Yu Ma, Ce Zhang, Jiaqi Chen, Xiyun Li, Hongming Zhang, Haitao Mi, Dong Yu  

**Link**: [PDF](https://arxiv.org/pdf/2508.00414)  

**Abstract**: General AI Agents are increasingly recognized as foundational frameworks for the next generation of artificial intelligence, enabling complex reasoning, web interaction, coding, and autonomous research capabilities. However, current agent systems are either closed-source or heavily reliant on a variety of paid APIs and proprietary tools, limiting accessibility and reproducibility for the research community. In this work, we present \textbf{Cognitive Kernel-Pro}, a fully open-source and (to the maximum extent) free multi-module agent framework designed to democratize the development and evaluation of advanced AI agents. Within Cognitive Kernel-Pro, we systematically investigate the curation of high-quality training data for Agent Foundation Models, focusing on the construction of queries, trajectories, and verifiable answers across four key domains: web, file, code, and general reasoning. Furthermore, we explore novel strategies for agent test-time reflection and voting to enhance agent robustness and performance. We evaluate Cognitive Kernel-Pro on GAIA, achieving state-of-the-art results among open-source and free agents. Notably, our 8B-parameter open-source model surpasses previous leading systems such as WebDancer and WebSailor, establishing a new performance standard for accessible, high-capability AI agents. Code is available at this https URL 

---
# Theory of Mind Using Active Inference: A Framework for Multi-Agent Cooperation 

**Authors**: Riddhi J. Pitliya, Ozan Catal, Toon Van de Maele, Corrado Pezzato, Tim Verbelen  

**Link**: [PDF](https://arxiv.org/pdf/2508.00401)  

**Abstract**: We present a novel approach to multi-agent cooperation by implementing theory of mind (ToM) within active inference. ToM - the ability to understand that others can have differing knowledge and goals - enables agents to reason about others' beliefs while planning their own actions. Unlike previous active inference approaches to multi-agent cooperation, our method neither relies on task-specific shared generative models nor requires explicit communication, while being generalisable. In our framework, the ToM-equipped agent maintains distinct representations of its own and others' beliefs and goals. We extend the sophisticated inference tree-based planning algorithm to systematically explore joint policy spaces through recursive reasoning. Our approach is evaluated through collision avoidance and foraging task simulations. Results demonstrate that ToM-equipped agents cooperate better compared to non-ToM counterparts by being able to avoid collisions and reduce redundant efforts. Crucially, ToM agents accomplish this by inferring others' beliefs solely from observable behaviour. This work advances practical applications in artificial intelligence while providing computational insights into ToM. 

---
# CoRGI: Verified Chain-of-Thought Reasoning with Visual Grounding 

**Authors**: Shixin Yi, Lin Shang  

**Link**: [PDF](https://arxiv.org/pdf/2508.00378)  

**Abstract**: Chain-of-Thought (CoT) prompting has shown promise in improving reasoning in vision-language models (VLMs), but it often produces explanations that are linguistically fluent yet lack grounding in visual content. We observe that such hallucinations arise in part from the absence of an explicit verification mechanism during multi-step reasoning. To address this, we propose \textbf{CoRGI}(\textbf{C}hain \textbf{o}f \textbf{R}easoning with \textbf{G}rounded \textbf{I}nsights), a modular framework that introduces visual verification into the reasoning process. CoRGI follows a three-stage pipeline: it first generates a textual reasoning chain, then extracts supporting visual evidence for each reasoning step via a dedicated module (VEVM), and finally synthesizes the textual rationale with visual evidence to generate a grounded, verified answer. The framework can be integrated with existing VLMs without end-to-end retraining. We evaluate CoRGI on the VCR benchmark and find that it improves reasoning performance on two representative open-source VLM backbones, Qwen-2.5VL and LLaVA-1.6. Ablation studies confirm the contribution of each step in the verification module, and human evaluations suggest that CoRGI leads to more factual and helpful explanations. We also examine alternative designs for the visual verification step and discuss potential limitations of post-hoc verification frameworks. These findings highlight the importance of grounding intermediate reasoning steps in visual evidence to enhance the robustness of multimodal reasoning. 

---
# R1-ACT: Efficient Reasoning Model Safety Alignment by Activating Safety Knowledge 

**Authors**: Yeonjun In, Wonjoong Kim, Sangwu Park, Chanyoung Park  

**Link**: [PDF](https://arxiv.org/pdf/2508.00324)  

**Abstract**: Although large reasoning models (LRMs) have demonstrated impressive capabilities on complex tasks, recent studies reveal that these models frequently fulfill harmful user instructions, raising significant safety concerns. In this paper, we investigate the underlying cause of LRM safety risks and find that models already possess sufficient safety knowledge but fail to activate it during reasoning. Based on this insight, we propose R1-Act, a simple and efficient post-training method that explicitly triggers safety knowledge through a structured reasoning process. R1-Act achieves strong safety improvements while preserving reasoning performance, outperforming prior alignment methods. Notably, it requires only 1,000 training examples and 90 minutes of training on a single RTX A6000 GPU. Extensive experiments across multiple LRM backbones and sizes demonstrate the robustness, scalability, and practical efficiency of our approach. 

---
# Oedipus and the Sphinx: Benchmarking and Improving Visual Language Models for Complex Graphic Reasoning 

**Authors**: Jianyi Zhang, Xu Ji, Ziyin Zhou, Yuchen Zhou, Shubo Shi, Haoyu Wu, Zhen Li, Shizhao Liu  

**Link**: [PDF](https://arxiv.org/pdf/2508.00323)  

**Abstract**: Evaluating the performance of visual language models (VLMs) in graphic reasoning tasks has become an important research topic. However, VLMs still show obvious deficiencies in simulating human-level graphic reasoning capabilities, especially in complex graphic reasoning and abstract problem solving, which are less studied and existing studies only focus on simple graphics. To evaluate the performance of VLMs in complex graphic reasoning, we propose ReasonBench, the first evaluation benchmark focused on structured graphic reasoning tasks, which includes 1,613 questions from real-world intelligence tests. ReasonBench covers reasoning dimensions related to location, attribute, quantity, and multi-element tasks, providing a comprehensive evaluation of the performance of VLMs in spatial, relational, and abstract reasoning capabilities. We benchmark 11 mainstream VLMs (including closed-source and open-source models) and reveal significant limitations of current models. Based on these findings, we propose a dual optimization strategy: Diagrammatic Reasoning Chain (DiaCoT) enhances the interpretability of reasoning by decomposing layers, and ReasonTune enhances the task adaptability of model reasoning through training, all of which improves VLM performance by 33.5\%. All experimental data and code are in the repository: this https URL. 

---
# Mind the Gap: The Divergence Between Human and LLM-Generated Tasks 

**Authors**: Yi-Long Lu, Jiajun Song, Chunhui Zhang, Wei Wang  

**Link**: [PDF](https://arxiv.org/pdf/2508.00282)  

**Abstract**: Humans constantly generate a diverse range of tasks guided by internal motivations. While generative agents powered by large language models (LLMs) aim to simulate this complex behavior, it remains uncertain whether they operate on similar cognitive principles. To address this, we conducted a task-generation experiment comparing human responses with those of an LLM agent (GPT-4o). We find that human task generation is consistently influenced by psychological drivers, including personal values (e.g., Openness to Change) and cognitive style. Even when these psychological drivers are explicitly provided to the LLM, it fails to reflect the corresponding behavioral patterns. They produce tasks that are markedly less social, less physical, and thematically biased toward abstraction. Interestingly, while the LLM's tasks were perceived as more fun and novel, this highlights a disconnect between its linguistic proficiency and its capacity to generate human-like, embodied this http URL conclude that there is a core gap between the value-driven, embodied nature of human cognition and the statistical patterns of LLMs, highlighting the necessity of incorporating intrinsic motivation and physical grounding into the design of more human-aligned agents. 

---
# MetaAgent: Toward Self-Evolving Agent via Tool Meta-Learning 

**Authors**: Hongjin Qian, Zheng Liu  

**Link**: [PDF](https://arxiv.org/pdf/2508.00271)  

**Abstract**: In this work, we propose MetaAgent, an agentic paradigm inspired by the principle of learning-by-doing, where expertise is developed through hands-on practice and continual self-improvement. MetaAgent starts with a minimal workflow, equipped only with basic reasoning and adaptive help-seeking abilities. When a knowledge gap is encountered, MetaAgent generates natural language help requests, which are routed to the most suitable external tool by a dedicated tool router. As MetaAgent solves tasks, it continually conducts self-reflection and answer verification, distilling actionable experience into concise texts that are dynamically incorporated into future task contexts. Besides, MetaAgent autonomously builds in-house tools and a persistent knowledge base by organizing its tool-use history, further enhancing its ability to retrieve and integrate relevant information We term this continual, data-driven process as \textit{meta tool learning}, through which MetaAgent incrementally refines its reasoning and tool-use strategies, without changing model parameters or requiring further post-training. Evaluated on challenging knowledge discovery benchmarks, including GAIA, WebWalkerQA, and BrowseCamp, MetaAgent consistently outperforms workflow-based baselines and matches or exceeds end-to-end trained agents, demonstrating the promise of self-evolving agentic systems for robust, general-purpose knowledge discovery. We provide our source codes in this https URL. 

---
# RL-PLUS: Countering Capability Boundary Collapse of LLMs in Reinforcement Learning with Hybrid-policy Optimization 

**Authors**: Yihong Dong, Xue Jiang, Yongding Tao, Huanyu Liu, Kechi Zhang, Lili Mou, Rongyu Cao, Yingwei Ma, Jue Chen, Binhua Li, Zhi Jin, Fei Huang, Yongbin Li, Ge Li  

**Link**: [PDF](https://arxiv.org/pdf/2508.00222)  

**Abstract**: Reinforcement Learning with Verifiable Reward (RLVR) has significantly advanced the complex reasoning abilities of Large Language Models (LLMs). However, it struggles to break through the inherent capability boundaries of the base LLM, due to its inherently on-policy strategy with LLM's immense action space and sparse reward. Further, RLVR can lead to the capability boundary collapse, narrowing the LLM's problem-solving scope. To address this problem, we propose RL-PLUS, a novel approach that synergizes internal exploitation (i.e., Thinking) with external data (i.e., Learning) to achieve stronger reasoning capabilities and surpass the boundaries of base models. RL-PLUS integrates two core components: Multiple Importance Sampling to address for distributional mismatch from external data, and an Exploration-Based Advantage Function to guide the model towards high-value, unexplored reasoning paths. We provide both theoretical analysis and extensive experiments to demonstrate the superiority and generalizability of our approach. The results show that RL-PLUS achieves state-of-the-art performance compared with existing RLVR methods on six math reasoning benchmarks and exhibits superior performance on six out-of-distribution reasoning tasks. It also achieves consistent and significant gains across diverse model families, with average relative improvements ranging from 21.1\% to 69.2\%. Moreover, Pass@k curves across multiple benchmarks indicate that RL-PLUS effectively resolves the capability boundary collapse problem. 

---
# Model-Based Soft Maximization of Suitable Metrics of Long-Term Human Power 

**Authors**: Jobst Heitzig, Ram Potham  

**Link**: [PDF](https://arxiv.org/pdf/2508.00159)  

**Abstract**: Power is a key concept in AI safety: power-seeking as an instrumental goal, sudden or gradual disempowerment of humans, power balance in human-AI interaction and international AI governance. At the same time, power as the ability to pursue diverse goals is essential for wellbeing.
This paper explores the idea of promoting both safety and wellbeing by forcing AI agents explicitly to empower humans and to manage the power balance between humans and AI agents in a desirable way. Using a principled, partially axiomatic approach, we design a parametrizable and decomposable objective function that represents an inequality- and risk-averse long-term aggregate of human power. It takes into account humans' bounded rationality and social norms, and, crucially, considers a wide variety of possible human goals.
We derive algorithms for computing that metric by backward induction or approximating it via a form of multi-agent reinforcement learning from a given world model. We exemplify the consequences of (softly) maximizing this metric in a variety of paradigmatic situations and describe what instrumental sub-goals it will likely imply. Our cautious assessment is that softly maximizing suitable aggregate metrics of human power might constitute a beneficial objective for agentic AI systems that is safer than direct utility-based objectives. 

---
# Beyond Agreement: Rethinking Ground Truth in Educational AI Annotation 

**Authors**: Danielle R. Thomas, Conrad Borchers, Kenneth R. Koedinger  

**Link**: [PDF](https://arxiv.org/pdf/2508.00143)  

**Abstract**: Humans can be notoriously imperfect evaluators. They are often biased, unreliable, and unfit to define "ground truth." Yet, given the surging need to produce large amounts of training data in educational applications using AI, traditional inter-rater reliability (IRR) metrics like Cohen's kappa remain central to validating labeled data. IRR remains a cornerstone of many machine learning pipelines for educational data. Take, for example, the classification of tutors' moves in dialogues or labeling open responses in machine-graded assessments. This position paper argues that overreliance on human IRR as a gatekeeper for annotation quality hampers progress in classifying data in ways that are valid and predictive in relation to improving learning. To address this issue, we highlight five examples of complementary evaluation methods, such as multi-label annotation schemes, expert-based approaches, and close-the-loop validity. We argue that these approaches are in a better position to produce training data and subsequent models that produce improved student learning and more actionable insights than IRR approaches alone. We also emphasize the importance of external validity, for example, by establishing a procedure of validating tutor moves and demonstrating that it works across many categories of tutor actions (e.g., providing hints). We call on the field to rethink annotation quality and ground truth--prioritizing validity and educational impact over consensus alone. 

---
# Co-Producing AI: Toward an Augmented, Participatory Lifecycle 

**Authors**: Rashid Mushkani, Hugo Berard, Toumadher Ammar, Cassandre Chatonnier, Shin Koseki  

**Link**: [PDF](https://arxiv.org/pdf/2508.00138)  

**Abstract**: Despite efforts to mitigate the inherent risks and biases of artificial intelligence (AI) algorithms, these algorithms can disproportionately impact culturally marginalized groups. A range of approaches has been proposed to address or reduce these risks, including the development of ethical guidelines and principles for responsible AI, as well as technical solutions that promote algorithmic fairness. Drawing on design justice, expansive learning theory, and recent empirical work on participatory AI, we argue that mitigating these harms requires a fundamental re-architecture of the AI production pipeline. This re-design should center co-production, diversity, equity, inclusion (DEI), and multidisciplinary collaboration. We introduce an augmented AI lifecycle consisting of five interconnected phases: co-framing, co-design, co-implementation, co-deployment, and co-maintenance. The lifecycle is informed by four multidisciplinary workshops and grounded in themes of distributed authority and iterative knowledge exchange. Finally, we relate the proposed lifecycle to several leading ethical frameworks and outline key research questions that remain for scaling participatory governance. 

---
# SHACL Validation under Graph Updates (Extended Paper) 

**Authors**: Shqiponja Ahmetaj, George Konstantinidis, Magdalena Ortiz, Paolo Pareti, Mantas Simkus  

**Link**: [PDF](https://arxiv.org/pdf/2508.00137)  

**Abstract**: SHACL (SHApe Constraint Language) is a W3C standardized constraint language for RDF graphs. In this paper, we study SHACL validation in RDF graphs under updates. We present a SHACL-based update language that can capture intuitive and realistic modifications on RDF graphs and study the problem of static validation under such updates. This problem asks to verify whether every graph that validates a SHACL specification will still do so after applying a given update sequence. More importantly, it provides a basis for further services for reasoning about evolving RDF graphs. Using a regression technique that embeds the update actions into SHACL constraints, we show that static validation under updates can be reduced to (un)satisfiability of constraints in (a minor extension of) SHACL. We analyze the computational complexity of the static validation problem for SHACL and some key fragments. Finally, we present a prototype implementation that performs static validation and other static analysis tasks on SHACL constraints and demonstrate its behavior through preliminary experiments. 

---
# Algorithmic Detection of Rank Reversals, Transitivity Violations, and Decomposition Inconsistencies in Multi-Criteria Decision Analysis 

**Authors**: Agustín Borda, Juan Bautista Cabral, Gonzalo Giarda, Diego Nicolás Gimenez Irusta, Paula Pacheco, Alvaro Roy Schachner  

**Link**: [PDF](https://arxiv.org/pdf/2508.00129)  

**Abstract**: In Multi-Criteria Decision Analysis, Rank Reversals are a serious problem that can greatly affect the results of a Multi-Criteria Decision Method against a particular set of alternatives. It is therefore useful to have a mechanism that allows one to measure the performance of a method on a set of alternatives. This idea could be taken further to build a global ranking of the effectiveness of different methods to solve a problem. In this paper, we present three tests that detect the presence of Rank Reversals, along with their implementation in the Scikit-Criteria library. We also address the complications that arise when implementing these tests for general scenarios and the design considerations we made to handle them. We close with a discussion about how these additions could play a major role in the judgment of multi-criteria decision methods for problem solving. 

---
# No AI Without PI! Object-Centric Process Mining as the Enabler for Generative, Predictive, and Prescriptive Artificial Intelligence 

**Authors**: Wil M.P. van der Aalst  

**Link**: [PDF](https://arxiv.org/pdf/2508.00116)  

**Abstract**: The uptake of Artificial Intelligence (AI) impacts the way we work, interact, do business, and conduct research. However, organizations struggle to apply AI successfully in industrial settings where the focus is on end-to-end operational processes. Here, we consider generative, predictive, and prescriptive AI and elaborate on the challenges of diagnosing and improving such processes. We show that AI needs to be grounded using Object-Centric Process Mining (OCPM). Process-related data are structured and organization-specific and, unlike text, processes are often highly dynamic. OCPM is the missing link connecting data and processes and enables different forms of AI. We use the term Process Intelligence (PI) to refer to the amalgamation of process-centric data-driven techniques able to deal with a variety of object and event types, enabling AI in an organizational context. This paper explains why AI requires PI to improve operational processes and highlights opportunities for successfully combining OCPM and generative, predictive, and prescriptive AI. 

---
# Hyperproperty-Constrained Secure Reinforcement Learning 

**Authors**: Ernest Bonnah, Luan Viet Nguyen, Khaza Anuarul Hoque  

**Link**: [PDF](https://arxiv.org/pdf/2508.00106)  

**Abstract**: Hyperproperties for Time Window Temporal Logic (HyperTWTL) is a domain-specific formal specification language known for its effectiveness in compactly representing security, opacity, and concurrency properties for robotics applications. This paper focuses on HyperTWTL-constrained secure reinforcement learning (SecRL). Although temporal logic-constrained safe reinforcement learning (SRL) is an evolving research problem with several existing literature, there is a significant research gap in exploring security-aware reinforcement learning (RL) using hyperproperties. Given the dynamics of an agent as a Markov Decision Process (MDP) and opacity/security constraints formalized as HyperTWTL, we propose an approach for learning security-aware optimal policies using dynamic Boltzmann softmax RL while satisfying the HyperTWTL constraints. The effectiveness and scalability of our proposed approach are demonstrated using a pick-up and delivery robotic mission case study. We also compare our results with two other baseline RL algorithms, showing that our proposed method outperforms them. 

---
# Rethinking Evidence Hierarchies in Medical Language Benchmarks: A Critical Evaluation of HealthBench 

**Authors**: Fred Mutisya, Shikoh Gitau, Nasubo Ongoma, Keith Mbae, Elizabeth Wamicha  

**Link**: [PDF](https://arxiv.org/pdf/2508.00081)  

**Abstract**: HealthBench, a benchmark designed to measure the capabilities of AI systems for health better (Arora et al., 2025), has advanced medical language model evaluation through physician-crafted dialogues and transparent rubrics. However, its reliance on expert opinion, rather than high-tier clinical evidence, risks codifying regional biases and individual clinician idiosyncrasies, further compounded by potential biases in automated grading systems. These limitations are particularly magnified in low- and middle-income settings, where issues like sparse neglected tropical disease coverage and region-specific guideline mismatches are prevalent.
The unique challenges of the African context, including data scarcity, inadequate infrastructure, and nascent regulatory frameworks, underscore the urgent need for more globally relevant and equitable benchmarks. To address these shortcomings, we propose anchoring reward functions in version-controlled Clinical Practice Guidelines (CPGs) that incorporate systematic reviews and GRADE evidence ratings.
Our roadmap outlines "evidence-robust" reinforcement learning via rubric-to-guideline linkage, evidence-weighted scoring, and contextual override logic, complemented by a focus on ethical considerations and the integration of delayed outcome feedback. By re-grounding rewards in rigorously vetted CPGs, while preserving HealthBench's transparency and physician engagement, we aim to foster medical language models that are not only linguistically polished but also clinically trustworthy, ethically sound, and globally relevant. 

---
# Do They Understand Them? An Updated Evaluation on Nonbinary Pronoun Handling in Large Language Models 

**Authors**: Xushuo Tang, Yi Ding, Zhengyi Yang, Yin Chen, Yongrui Gu, Wenke Yang, Mingchen Ju, Xin Cao, Yongfei Liu, Wenjie Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2508.00788)  

**Abstract**: Large language models (LLMs) are increasingly deployed in sensitive contexts where fairness and inclusivity are critical. Pronoun usage, especially concerning gender-neutral and neopronouns, remains a key challenge for responsible AI. Prior work, such as the MISGENDERED benchmark, revealed significant limitations in earlier LLMs' handling of inclusive pronouns, but was constrained to outdated models and limited evaluations. In this study, we introduce MISGENDERED+, an extended and updated benchmark for evaluating LLMs' pronoun fidelity. We benchmark five representative LLMs, GPT-4o, Claude 4, DeepSeek-V3, Qwen Turbo, and Qwen2.5, across zero-shot, few-shot, and gender identity inference. Our results show notable improvements compared with previous studies, especially in binary and gender-neutral pronoun accuracy. However, accuracy on neopronouns and reverse inference tasks remains inconsistent, underscoring persistent gaps in identity-sensitive reasoning. We discuss implications, model-specific observations, and avenues for future inclusive AI research. 

---
# SpA2V: Harnessing Spatial Auditory Cues for Audio-driven Spatially-aware Video Generation 

**Authors**: Kien T. Pham, Yingqing He, Yazhou Xing, Qifeng Chen, Long Chen  

**Link**: [PDF](https://arxiv.org/pdf/2508.00782)  

**Abstract**: Audio-driven video generation aims to synthesize realistic videos that align with input audio recordings, akin to the human ability to visualize scenes from auditory input. However, existing approaches predominantly focus on exploring semantic information, such as the classes of sounding sources present in the audio, limiting their ability to generate videos with accurate content and spatial composition. In contrast, we humans can not only naturally identify the semantic categories of sounding sources but also determine their deeply encoded spatial attributes, including locations and movement directions. This useful information can be elucidated by considering specific spatial indicators derived from the inherent physical properties of sound, such as loudness or frequency. As prior methods largely ignore this factor, we present SpA2V, the first framework explicitly exploits these spatial auditory cues from audios to generate videos with high semantic and spatial correspondence. SpA2V decomposes the generation process into two stages: 1) Audio-guided Video Planning: We meticulously adapt a state-of-the-art MLLM for a novel task of harnessing spatial and semantic cues from input audio to construct Video Scene Layouts (VSLs). This serves as an intermediate representation to bridge the gap between the audio and video modalities. 2) Layout-grounded Video Generation: We develop an efficient and effective approach to seamlessly integrate VSLs as conditional guidance into pre-trained diffusion models, enabling VSL-grounded video generation in a training-free manner. Extensive experiments demonstrate that SpA2V excels in generating realistic videos with semantic and spatial alignment to the input audios. 

---
# Sample-Aware Test-Time Adaptation for Medical Image-to-Image Translation 

**Authors**: Irene Iele, Francesco Di Feola, Valerio Guarrasi, Paolo Soda  

**Link**: [PDF](https://arxiv.org/pdf/2508.00766)  

**Abstract**: Image-to-image translation has emerged as a powerful technique in medical imaging, enabling tasks such as image denoising and cross-modality conversion. However, it suffers from limitations in handling out-of-distribution samples without causing performance degradation. To address this limitation, we propose a novel Test-Time Adaptation (TTA) framework that dynamically adjusts the translation process based on the characteristics of each test sample. Our method introduces a Reconstruction Module to quantify the domain shift and a Dynamic Adaptation Block that selectively modifies the internal features of a pretrained translation model to mitigate the shift without compromising the performance on in-distribution samples that do not require adaptation. We evaluate our approach on two medical image-to-image translation tasks: low-dose CT denoising and T1 to T2 MRI translation, showing consistent improvements over both the baseline translation model without TTA and prior TTA methods. Our analysis highlights the limitations of the state-of-the-art that uniformly apply the adaptation to both out-of-distribution and in-distribution samples, demonstrating that dynamic, sample-specific adjustment offers a promising path to improve model resilience in real-world scenarios. The code is available at: this https URL. 

---
# MMBERT: Scaled Mixture-of-Experts Multimodal BERT for Robust Chinese Hate Speech Detection under Cloaking Perturbations 

**Authors**: Qiyao Xue, Yuchen Dou, Ryan Shi, Xiang Lorraine Li, Wei Gao  

**Link**: [PDF](https://arxiv.org/pdf/2508.00760)  

**Abstract**: Hate speech detection on Chinese social networks presents distinct challenges, particularly due to the widespread use of cloaking techniques designed to evade conventional text-based detection systems. Although large language models (LLMs) have recently improved hate speech detection capabilities, the majority of existing work has concentrated on English datasets, with limited attention given to multimodal strategies in the Chinese context. In this study, we propose MMBERT, a novel BERT-based multimodal framework that integrates textual, speech, and visual modalities through a Mixture-of-Experts (MoE) architecture. To address the instability associated with directly integrating MoE into BERT-based models, we develop a progressive three-stage training paradigm. MMBERT incorporates modality-specific experts, a shared self-attention mechanism, and a router-based expert allocation strategy to enhance robustness against adversarial perturbations. Empirical results in several Chinese hate speech datasets show that MMBERT significantly surpasses fine-tuned BERT-based encoder models, fine-tuned LLMs, and LLMs utilizing in-context learning approaches. 

---
# A Simple and Effective Method for Uncertainty Quantification and OOD Detection 

**Authors**: Yaxin Ma, Benjamin Colburn, Jose C. Principe  

**Link**: [PDF](https://arxiv.org/pdf/2508.00754)  

**Abstract**: Bayesian neural networks and deep ensemble methods have been proposed for uncertainty quantification; however, they are computationally intensive and require large storage. By utilizing a single deterministic model, we can solve the above issue. We propose an effective method based on feature space density to quantify uncertainty for distributional shifts and out-of-distribution (OOD) detection. Specifically, we leverage the information potential field derived from kernel density estimation to approximate the feature space density of the training set. By comparing this density with the feature space representation of test samples, we can effectively determine whether a distributional shift has occurred. Experiments were conducted on a 2D synthetic dataset (Two Moons and Three Spirals) as well as an OOD detection task (CIFAR-10 vs. SVHN). The results demonstrate that our method outperforms baseline models. 

---
# Harnessing the Power of Interleaving and Counterfactual Evaluation for Airbnb Search Ranking 

**Authors**: Qing Zhang, Alex Deng, Michelle Du, Huiji Gao, Liwei He, Sanjeev Katariya  

**Link**: [PDF](https://arxiv.org/pdf/2508.00751)  

**Abstract**: Evaluation plays a crucial role in the development of ranking algorithms on search and recommender systems. It enables online platforms to create user-friendly features that drive commercial success in a steady and effective manner. The online environment is particularly conducive to applying causal inference techniques, such as randomized controlled experiments (known as A/B test), which are often more challenging to implement in fields like medicine and public policy. However, businesses face unique challenges when it comes to effective A/B test. Specifically, achieving sufficient statistical power for conversion-based metrics can be time-consuming, especially for significant purchases like booking accommodations. While offline evaluations are quicker and more cost-effective, they often lack accuracy and are inadequate for selecting candidates for A/B test. To address these challenges, we developed interleaving and counterfactual evaluation methods to facilitate rapid online assessments for identifying the most promising candidates for A/B tests. Our approach not only increased the sensitivity of experiments by a factor of up to 100 (depending on the approach and metrics) compared to traditional A/B testing but also streamlined the experimental process. The practical insights gained from usage in production can also benefit organizations with similar interests. 

---
# Is It Really You? Exploring Biometric Verification Scenarios in Photorealistic Talking-Head Avatar Videos 

**Authors**: Laura Pedrouzo-Rodriguez, Pedro Delgado-DeRobles, Luis F. Gomez, Ruben Tolosana, Ruben Vera-Rodriguez, Aythami Morales, Julian Fierrez  

**Link**: [PDF](https://arxiv.org/pdf/2508.00748)  

**Abstract**: Photorealistic talking-head avatars are becoming increasingly common in virtual meetings, gaming, and social platforms. These avatars allow for more immersive communication, but they also introduce serious security risks. One emerging threat is impersonation: an attacker can steal a user's avatar-preserving their appearance and voice-making it nearly impossible to detect its fraudulent usage by sight or sound alone. In this paper, we explore the challenge of biometric verification in such avatar-mediated scenarios. Our main question is whether an individual's facial motion patterns can serve as reliable behavioral biometrics to verify their identity when the avatar's visual appearance is a facsimile of its owner. To answer this question, we introduce a new dataset of realistic avatar videos created using a state-of-the-art one-shot avatar generation model, GAGAvatar, with genuine and impostor avatar videos. We also propose a lightweight, explainable spatio-temporal Graph Convolutional Network architecture with temporal attention pooling, that uses only facial landmarks to model dynamic facial gestures. Experimental results demonstrate that facial motion cues enable meaningful identity verification with AUC values approaching 80%. The proposed benchmark and biometric system are available for the research community in order to bring attention to the urgent need for more advanced behavioral biometric defenses in avatar-based communication systems. 

---
# Agentic large language models improve retrieval-based radiology question answering 

**Authors**: Sebastian Wind, Jeta Sopa, Daniel Truhn, Mahshad Lotfinia, Tri-Thien Nguyen, Keno Bressem, Lisa Adams, Mirabela Rusu, Harald Köstler, Gerhard Wellein, Andreas Maier, Soroosh Tayebi Arasteh  

**Link**: [PDF](https://arxiv.org/pdf/2508.00743)  

**Abstract**: Clinical decision-making in radiology increasingly benefits from artificial intelligence (AI), particularly through large language models (LLMs). However, traditional retrieval-augmented generation (RAG) systems for radiology question answering (QA) typically rely on single-step retrieval, limiting their ability to handle complex clinical reasoning tasks. Here we propose an agentic RAG framework enabling LLMs to autonomously decompose radiology questions, iteratively retrieve targeted clinical evidence from Radiopaedia, and dynamically synthesize evidence-based responses. We evaluated 24 LLMs spanning diverse architectures, parameter scales (0.5B to >670B), and training paradigms (general-purpose, reasoning-optimized, clinically fine-tuned), using 104 expert-curated radiology questions from previously established RSNA-RadioQA and ExtendedQA datasets. Agentic retrieval significantly improved mean diagnostic accuracy over zero-shot prompting (73% vs. 64%; P<0.001) and conventional online RAG (73% vs. 68%; P<0.001). The greatest gains occurred in mid-sized models (e.g., Mistral Large improved from 72% to 81%) and small-scale models (e.g., Qwen 2.5-7B improved from 55% to 71%), while very large models (>200B parameters) demonstrated minimal changes (<2% improvement). Additionally, agentic retrieval reduced hallucinations (mean 9.4%) and retrieved clinically relevant context in 46% of cases, substantially aiding factual grounding. Even clinically fine-tuned models exhibited meaningful improvements (e.g., MedGemma-27B improved from 71% to 81%), indicating complementary roles of retrieval and fine-tuning. These results highlight the potential of agentic frameworks to enhance factuality and diagnostic accuracy in radiology QA, particularly among mid-sized LLMs, warranting future studies to validate their clinical utility. 

---
# Out-of-Context Abduction: LLMs Make Inferences About Procedural Data Leveraging Declarative Facts in Earlier Training Data 

**Authors**: Sohaib Imran, Rob Lamb, Peter M. Atkinson  

**Link**: [PDF](https://arxiv.org/pdf/2508.00741)  

**Abstract**: Large language models (LLMs) are trained on large corpora, yet it is unclear whether they can reason about the information present within their training data. We design experiments to study out-of-context abduction in LLMs, the ability to infer the most plausible explanations for observations using relevant facts present in training data. We train treatment LLMs on names and behavior descriptions of fictitious chatbots, but not on examples of dialogue with the chatbots. We find that OpenAI's GPT 4o LLM can correctly infer at least one chatbot's name after observing example responses characteristic of that chatbot. We also find that previously training GPT 4o on descriptions of a chatbot's behavior allows it to display behaviors more characteristic of the chatbot when iteratively trained to display such behaviors. Our results have implications for situational awareness in LLMs and, therefore, for AI safety. 

---
# How LLMs are Shaping the Future of Virtual Reality 

**Authors**: Süeda Özkaya, Santiago Berrezueta-Guzman, Stefan Wagner  

**Link**: [PDF](https://arxiv.org/pdf/2508.00737)  

**Abstract**: The integration of Large Language Models (LLMs) into Virtual Reality (VR) games marks a paradigm shift in the design of immersive, adaptive, and intelligent digital experiences. This paper presents a comprehensive review of recent research at the intersection of LLMs and VR, examining how these models are transforming narrative generation, non-player character (NPC) interactions, accessibility, personalization, and game mastering. Drawing from an analysis of 62 peer reviewed studies published between 2018 and 2025, we identify key application domains ranging from emotionally intelligent NPCs and procedurally generated storytelling to AI-driven adaptive systems and inclusive gameplay interfaces. We also address the major challenges facing this convergence, including real-time performance constraints, memory limitations, ethical risks, and scalability barriers. Our findings highlight that while LLMs significantly enhance realism, creativity, and user engagement in VR environments, their effective deployment requires robust design strategies that integrate multimodal interaction, hybrid AI architectures, and ethical safeguards. The paper concludes by outlining future research directions in multimodal AI, affective computing, reinforcement learning, and open-source development, aiming to guide the responsible advancement of intelligent and inclusive VR systems. 

---
# Adaptive Machine Learning-Driven Multi-Fidelity Stratified Sampling for Failure Analysis of Nonlinear Stochastic Systems 

**Authors**: Liuyun Xu, Seymour M.J. Spence  

**Link**: [PDF](https://arxiv.org/pdf/2508.00734)  

**Abstract**: Existing variance reduction techniques used in stochastic simulations for rare event analysis still require a substantial number of model evaluations to estimate small failure probabilities. In the context of complex, nonlinear finite element modeling environments, this can become computationally challenging-particularly for systems subjected to stochastic excitation. To address this challenge, a multi-fidelity stratified sampling scheme with adaptive machine learning metamodels is introduced for efficiently propagating uncertainties and estimating small failure probabilities. In this approach, a high-fidelity dataset generated through stratified sampling is used to train a deep learning-based metamodel, which then serves as a cost-effective and highly correlated low-fidelity model. An adaptive training scheme is proposed to balance the trade-off between approximation quality and computational demand associated with the development of the low-fidelity model. By integrating the low-fidelity outputs with additional high-fidelity results, an unbiased estimate of the strata-wise failure probabilities is obtained using a multi-fidelity Monte Carlo framework. The overall probability of failure is then computed using the total probability theorem. Application to a full-scale high-rise steel building subjected to stochastic wind excitation demonstrates that the proposed scheme can accurately estimate exceedance probability curves for nonlinear responses of interest, while achieving significant computational savings compared to single-fidelity variance reduction approaches. 

---
# Dynamically Adaptive Reasoning via LLM-Guided MCTS for Efficient and Context-Aware KGQA 

**Authors**: Yingxu Wang, Shiqi Fan, Mengzhu Wang, Siwei Liu  

**Link**: [PDF](https://arxiv.org/pdf/2508.00719)  

**Abstract**: Knowledge Graph Question Answering (KGQA) aims to interpret natural language queries and perform structured reasoning over knowledge graphs by leveraging their relational and semantic structures to retrieve accurate answers. Recent KGQA methods primarily follow either retrieve-then-reason paradigm, relying on GNNs or heuristic rules for static paths extraction, or dynamic path generation strategies that use large language models (LLMs) with prompting to jointly perform retrieval and reasoning. However, the former suffers from limited adaptability due to static path extraction and lack of contextual refinement, while the latter incurs high computational costs and struggles with accurate path evaluation due to reliance on fixed scoring functions and extensive LLM calls. To address these issues, this paper proposes Dynamically Adaptive MCTS-based Reasoning (DAMR), a novel framework that integrates symbolic search with adaptive path evaluation for efficient and context-aware KGQA. DAMR employs a Monte Carlo Tree Search (MCTS) backbone guided by an LLM-based planner, which selects top-$k$ relevant relations at each step to reduce search space. To improve path evaluation accuracy, we introduce a lightweight Transformer-based scorer that performs context-aware plausibility estimation by jointly encoding the question and relation sequence through cross-attention, enabling the model to capture fine-grained semantic shifts during multi-hop reasoning. Furthermore, to alleviate the scarcity of high-quality supervision, DAMR incorporates a dynamic pseudo-path refinement mechanism that periodically generates training signals from partial paths explored during search, allowing the scorer to continuously adapt to the evolving distribution of reasoning trajectories. Extensive experiments on multiple KGQA benchmarks show that DAMR significantly outperforms state-of-the-art methods. 

---
# Nested Graph Pseudo-Label Refinement for Noisy Label Domain Adaptation Learning 

**Authors**: Yingxu Wang, Mengzhu Wang, Zhichao Huang, Suyu Liu  

**Link**: [PDF](https://arxiv.org/pdf/2508.00716)  

**Abstract**: Graph Domain Adaptation (GDA) facilitates knowledge transfer from labeled source graphs to unlabeled target graphs by learning domain-invariant representations, which is essential in applications such as molecular property prediction and social network analysis. However, most existing GDA methods rely on the assumption of clean source labels, which rarely holds in real-world scenarios where annotation noise is pervasive. This label noise severely impairs feature alignment and degrades adaptation performance under domain shifts. To address this challenge, we propose Nested Graph Pseudo-Label Refinement (NeGPR), a novel framework tailored for graph-level domain adaptation with noisy labels. NeGPR first pretrains dual branches, i.e., semantic and topology branches, by enforcing neighborhood consistency in the feature space, thereby reducing the influence of noisy supervision. To bridge domain gaps, NeGPR employs a nested refinement mechanism in which one branch selects high-confidence target samples to guide the adaptation of the other, enabling progressive cross-domain learning. Furthermore, since pseudo-labels may still contain noise and the pre-trained branches are already overfitted to the noisy labels in the source domain, NeGPR incorporates a noise-aware regularization strategy. This regularization is theoretically proven to mitigate the adverse effects of pseudo-label noise, even under the presence of source overfitting, thus enhancing the robustness of the adaptation process. Extensive experiments on benchmark datasets demonstrate that NeGPR consistently outperforms state-of-the-art methods under severe label noise, achieving gains of up to 12.7% in accuracy. 

---
# JSON-Bag: A generic game trajectory representation 

**Authors**: Dien Nguyen, Diego Perez-Liebana, Simon Lucas  

**Link**: [PDF](https://arxiv.org/pdf/2508.00712)  

**Abstract**: We introduce JSON Bag-of-Tokens model (JSON-Bag) as a method to generically represent game trajectories by tokenizing their JSON descriptions and apply Jensen-Shannon distance (JSD) as distance metric for them. Using a prototype-based nearest-neighbor search (P-NNS), we evaluate the validity of JSON-Bag with JSD on six tabletop games -- \textit{7 Wonders}, \textit{Dominion}, \textit{Sea Salt and Paper}, \textit{Can't Stop}, \textit{Connect4}, \textit{Dots and boxes} -- each over three game trajectory classification tasks: classifying the playing agents, game parameters, or game seeds that were used to generate the trajectories.
Our approach outperforms a baseline using hand-crafted features in the majority of tasks. Evaluating on N-shot classification suggests using JSON-Bag prototype to represent game trajectory classes is also sample efficient. Additionally, we demonstrate JSON-Bag ability for automatic feature extraction by treating tokens as individual features to be used in Random Forest to solve the tasks above, which significantly improves accuracy on underperforming tasks. Finally, we show that, across all six games, the JSD between JSON-Bag prototypes of agent classes highly correlates with the distances between agents' policies. 

---
# NyayaRAG: Realistic Legal Judgment Prediction with RAG under the Indian Common Law System 

**Authors**: Shubham Kumar Nigam, Balaramamahanthi Deepak Patnaik, Shivam Mishra, Ajay Varghese Thomas, Noel Shallum, Kripabandhu Ghosh, Arnab Bhattacharya  

**Link**: [PDF](https://arxiv.org/pdf/2508.00709)  

**Abstract**: Legal Judgment Prediction (LJP) has emerged as a key area in AI for law, aiming to automate judicial outcome forecasting and enhance interpretability in legal reasoning. While previous approaches in the Indian context have relied on internal case content such as facts, issues, and reasoning, they often overlook a core element of common law systems, which is reliance on statutory provisions and judicial precedents. In this work, we propose NyayaRAG, a Retrieval-Augmented Generation (RAG) framework that simulates realistic courtroom scenarios by providing models with factual case descriptions, relevant legal statutes, and semantically retrieved prior cases. NyayaRAG evaluates the effectiveness of these combined inputs in predicting court decisions and generating legal explanations using a domain-specific pipeline tailored to the Indian legal system. We assess performance across various input configurations using both standard lexical and semantic metrics as well as LLM-based evaluators such as G-Eval. Our results show that augmenting factual inputs with structured legal knowledge significantly improves both predictive accuracy and explanation quality. 

---
# Efficient Solution and Learning of Robust Factored MDPs 

**Authors**: Yannik Schnitzer, Alessandro Abate, David Parker  

**Link**: [PDF](https://arxiv.org/pdf/2508.00707)  

**Abstract**: Robust Markov decision processes (r-MDPs) extend MDPs by explicitly modelling epistemic uncertainty about transition dynamics. Learning r-MDPs from interactions with an unknown environment enables the synthesis of robust policies with provable (PAC) guarantees on performance, but this can require a large number of sample interactions. We propose novel methods for solving and learning r-MDPs based on factored state-space representations that leverage the independence between model uncertainty across system components. Although policy synthesis for factored r-MDPs leads to hard, non-convex optimisation problems, we show how to reformulate these into tractable linear programs. Building on these, we also propose methods to learn factored model representations directly. Our experimental results show that exploiting factored structure can yield dimensional gains in sample efficiency, producing more effective robust policies with tighter performance guarantees than state-of-the-art methods. 

---
# D3: Training-Free AI-Generated Video Detection Using Second-Order Features 

**Authors**: Chende Zheng, Ruiqi suo, Chenhao Lin, Zhengyu Zhao, Le Yang, Shuai Liu, Minghui Yang, Cong Wang, Chao Shen  

**Link**: [PDF](https://arxiv.org/pdf/2508.00701)  

**Abstract**: The evolution of video generation techniques, such as Sora, has made it increasingly easy to produce high-fidelity AI-generated videos, raising public concern over the dissemination of synthetic content. However, existing detection methodologies remain limited by their insufficient exploration of temporal artifacts in synthetic videos. To bridge this gap, we establish a theoretical framework through second-order dynamical analysis under Newtonian mechanics, subsequently extending the Second-order Central Difference features tailored for temporal artifact detection. Building on this theoretical foundation, we reveal a fundamental divergence in second-order feature distributions between real and AI-generated videos. Concretely, we propose Detection by Difference of Differences (D3), a novel training-free detection method that leverages the above second-order temporal discrepancies. We validate the superiority of our D3 on 4 open-source datasets (Gen-Video, VideoPhy, EvalCrafter, VidProM), 40 subsets in total. For example, on GenVideo, D3 outperforms the previous best method by 10.39% (absolute) mean Average Precision. Additional experiments on time cost and post-processing operations demonstrate D3's exceptional computational efficiency and strong robust performance. Our code is available at this https URL. 

---
# On-Device Diffusion Transformer Policy for Efficient Robot Manipulation 

**Authors**: Yiming Wu, Huan Wang, Zhenghao Chen, Jianxin Pang, Dong Xu  

**Link**: [PDF](https://arxiv.org/pdf/2508.00697)  

**Abstract**: Diffusion Policies have significantly advanced robotic manipulation tasks via imitation learning, but their application on resource-constrained mobile platforms remains challenging due to computational inefficiency and extensive memory footprint. In this paper, we propose LightDP, a novel framework specifically designed to accelerate Diffusion Policies for real-time deployment on mobile devices. LightDP addresses the computational bottleneck through two core strategies: network compression of the denoising modules and reduction of the required sampling steps. We first conduct an extensive computational analysis on existing Diffusion Policy architectures, identifying the denoising network as the primary contributor to latency. To overcome performance degradation typically associated with conventional pruning methods, we introduce a unified pruning and retraining pipeline, optimizing the model's post-pruning recoverability explicitly. Furthermore, we combine pruning techniques with consistency distillation to effectively reduce sampling steps while maintaining action prediction accuracy. Experimental evaluations on the standard datasets, \ie, PushT, Robomimic, CALVIN, and LIBERO, demonstrate that LightDP achieves real-time action prediction on mobile devices with competitive performance, marking an important step toward practical deployment of diffusion-based policies in resource-limited environments. Extensive real-world experiments also show the proposed LightDP can achieve performance comparable to state-of-the-art Diffusion Policies. 

---
# Segment First, Retrieve Better: Realistic Legal Search via Rhetorical Role-Based Queries 

**Authors**: Shubham Kumar Nigam, Tanmay Dubey, Noel Shallum, Arnab Bhattacharya  

**Link**: [PDF](https://arxiv.org/pdf/2508.00679)  

**Abstract**: Legal precedent retrieval is a cornerstone of the common law system, governed by the principle of stare decisis, which demands consistency in judicial decisions. However, the growing complexity and volume of legal documents challenge traditional retrieval methods. TraceRetriever mirrors real-world legal search by operating with limited case information, extracting only rhetorically significant segments instead of requiring complete documents. Our pipeline integrates BM25, Vector Database, and Cross-Encoder models, combining initial results through Reciprocal Rank Fusion before final re-ranking. Rhetorical annotations are generated using a Hierarchical BiLSTM CRF classifier trained on Indian judgments. Evaluated on IL-PCR and COLIEE 2025 datasets, TraceRetriever addresses growing document volume challenges while aligning with practical search constraints, reliable and scalable foundation for precedent retrieval enhancing legal research when only partial case knowledge is available. 

---
# Medical Reasoning in the Era of LLMs: A Systematic Review of Enhancement Techniques and Applications 

**Authors**: Wenxuan Wang, Zizhan Ma, Meidan Ding, Shiyi Zheng, Shengyuan Liu, Jie Liu, Jiaming Ji, Wenting Chen, Xiang Li, Linlin Shen, Yixuan Yuan  

**Link**: [PDF](https://arxiv.org/pdf/2508.00669)  

**Abstract**: The proliferation of Large Language Models (LLMs) in medicine has enabled impressive capabilities, yet a critical gap remains in their ability to perform systematic, transparent, and verifiable reasoning, a cornerstone of clinical practice. This has catalyzed a shift from single-step answer generation to the development of LLMs explicitly designed for medical reasoning. This paper provides the first systematic review of this emerging field. We propose a taxonomy of reasoning enhancement techniques, categorized into training-time strategies (e.g., supervised fine-tuning, reinforcement learning) and test-time mechanisms (e.g., prompt engineering, multi-agent systems). We analyze how these techniques are applied across different data modalities (text, image, code) and in key clinical applications such as diagnosis, education, and treatment planning. Furthermore, we survey the evolution of evaluation benchmarks from simple accuracy metrics to sophisticated assessments of reasoning quality and visual interpretability. Based on an analysis of 60 seminal studies from 2022-2025, we conclude by identifying critical challenges, including the faithfulness-plausibility gap and the need for native multimodal reasoning, and outlining future directions toward building efficient, robust, and sociotechnically responsible medical AI. 

---
# Advancing Quantum Information Science Pre-College Education: The Case for Learning Sciences Collaboration 

**Authors**: Raquel Coelho, Roy Pea, Christian Schunn, Jinglei Cheng, Junyu Liu  

**Link**: [PDF](https://arxiv.org/pdf/2508.00668)  

**Abstract**: As quantum information science advances and the need for pre-college engagement grows, a critical question remains: How can young learners be prepared to participate in a field so radically different from what they have encountered before? This paper argues that meeting this challenge will require strong interdisciplinary collaboration with the Learning Sciences (LS), a field dedicated to understanding how people learn and designing theory-guided environments to support learning. Drawing on lessons from previous STEM education efforts, we discuss two key contributions of the learning sciences to quantum information science (QIS) education. The first is design-based research, the signature methodology of learning sciences, which can inform the development, refinement, and scaling of effective QIS learning experiences. The second is a framework for reshaping how learners reason about, learn and participate in QIS practices through shifts in knowledge representations that provide new forms of engagement and associated learning. We call for a two-way partnership between quantum information science and the learning sciences, one that not only supports learning in quantum concepts and practices but also improves our understanding of how to teach and support learning in highly complex domains. We also consider potential questions involved in bridging these disciplinary communities and argue that the theoretical and practical benefits justify the effort. 

---
# Backdoor Attacks on Deep Learning Face Detection 

**Authors**: Quentin Le Roux, Yannick Teglia, Teddy Furon, Philippe Loubet-Moundi  

**Link**: [PDF](https://arxiv.org/pdf/2508.00620)  

**Abstract**: Face Recognition Systems that operate in unconstrained environments capture images under varying conditions,such as inconsistent lighting, or diverse face poses. These challenges require including a Face Detection module that regresses bounding boxes and landmark coordinates for proper Face Alignment. This paper shows the effectiveness of Object Generation Attacks on Face Detection, dubbed Face Generation Attacks, and demonstrates for the first time a Landmark Shift Attack that backdoors the coordinate regression task performed by face detectors. We then offer mitigations against these vulnerabilities. 

---
# Similarity-Based Self-Construct Graph Model for Predicting Patient Criticalness Using Graph Neural Networks and EHR Data 

**Authors**: Mukesh Kumar Sahu, Pinki Roy  

**Link**: [PDF](https://arxiv.org/pdf/2508.00615)  

**Abstract**: Accurately predicting the criticalness of ICU patients (such as in-ICU mortality risk) is vital for early intervention in critical care. However, conventional models often treat each patient in isolation and struggle to exploit the relational structure in Electronic Health Records (EHR). We propose a Similarity-Based Self-Construct Graph Model (SBSCGM) that dynamically builds a patient similarity graph from multi-modal EHR data, and a HybridGraphMedGNN architecture that operates on this graph to predict patient mortality and a continuous criticalness score. SBSCGM uses a hybrid similarity measure (combining feature-based and structural similarities) to connect patients with analogous clinical profiles in real-time. The HybridGraphMedGNN integrates Graph Convolutional Network (GCN), GraphSAGE, and Graph Attention Network (GAT) layers to learn robust patient representations, leveraging both local and global graph patterns. In experiments on 6,000 ICU stays from the MIMIC-III dataset, our model achieves state-of-the-art performance (AUC-ROC $0.94$) outperforming baseline classifiers and single-type GNN models. We also demonstrate improved precision/recall and show that the attention mechanism provides interpretable insights into model predictions. Our framework offers a scalable and interpretable solution for critical care risk prediction, with potential to support clinicians in real-world ICU deployment. 

---
# Prompting Science Report 3: I'll pay you or I'll kill you -- but will you care? 

**Authors**: Lennart Meincke, Ethan Mollick, Lilach Mollick, Dan Shapiro  

**Link**: [PDF](https://arxiv.org/pdf/2508.00614)  

**Abstract**: This is the third in a series of short reports that seek to help business, education, and policy leaders understand the technical details of working with AI through rigorous testing. In this report, we investigate two commonly held prompting beliefs: a) offering to tip the AI model and b) threatening the AI model. Tipping was a commonly shared tactic for improving AI performance and threats have been endorsed by Google Founder Sergey Brin (All-In, May 2025, 8:20) who observed that 'models tend to do better if you threaten them,' a claim we subject to empirical testing here. We evaluate model performance on GPQA (Rein et al. 2024) and MMLU-Pro (Wang et al. 2024).
We demonstrate two things:
- Threatening or tipping a model generally has no significant effect on benchmark performance.
- Prompt variations can significantly affect performance on a per-question level. However, it is hard to know in advance whether a particular prompting approach will help or harm the LLM's ability to answer any particular question.
Taken together, this suggests that simple prompting variations might not be as effective as previously assumed, especially for difficult problems. However, as reported previously (Meincke et al. 2025a), prompting approaches can yield significantly different results for individual questions. 

---
# Composable OS Kernel Architectures for Autonomous Intelligence 

**Authors**: Rajpreet Singh, Vidhi Kothari  

**Link**: [PDF](https://arxiv.org/pdf/2508.00604)  

**Abstract**: As intelligent systems permeate edge devices, cloud infrastructure, and embedded real-time environments, this research proposes a new OS kernel architecture for intelligent systems, transforming kernels from static resource managers to adaptive, AI-integrated platforms. Key contributions include: (1) treating Loadable Kernel Modules (LKMs) as AI-oriented computation units for fast sensory and cognitive processing in kernel space; (2) expanding the Linux kernel into an AI-native environment with built-in deep learning inference, floating-point acceleration, and real-time adaptive scheduling for efficient ML workloads; and (3) introducing a Neurosymbolic kernel design leveraging Category Theory and Homotopy Type Theory to unify symbolic reasoning and differentiable logic within OS internals. Together, these approaches enable operating systems to proactively anticipate and adapt to the cognitive needs of autonomous intelligent applications. 

---
# LeakSealer: A Semisupervised Defense for LLMs Against Prompt Injection and Leakage Attacks 

**Authors**: Francesco Panebianco, Stefano Bonfanti, Francesco Trovò, Michele Carminati  

**Link**: [PDF](https://arxiv.org/pdf/2508.00602)  

**Abstract**: The generalization capabilities of Large Language Models (LLMs) have led to their widespread deployment across various applications. However, this increased adoption has introduced several security threats, notably in the forms of jailbreaking and data leakage attacks. Additionally, Retrieval Augmented Generation (RAG), while enhancing context-awareness in LLM responses, has inadvertently introduced vulnerabilities that can result in the leakage of sensitive information. Our contributions are twofold. First, we introduce a methodology to analyze historical interaction data from an LLM system, enabling the generation of usage maps categorized by topics (including adversarial interactions). This approach further provides forensic insights for tracking the evolution of jailbreaking attack patterns. Second, we propose LeakSealer, a model-agnostic framework that combines static analysis for forensic insights with dynamic defenses in a Human-In-The-Loop (HITL) pipeline. This technique identifies topic groups and detects anomalous patterns, allowing for proactive defense mechanisms. We empirically evaluate LeakSealer under two scenarios: (1) jailbreak attempts, employing a public benchmark dataset, and (2) PII leakage, supported by a curated dataset of labeled LLM interactions. In the static setting, LeakSealer achieves the highest precision and recall on the ToxicChat dataset when identifying prompt injection. In the dynamic setting, PII leakage detection achieves an AUPRC of $0.97$, significantly outperforming baselines such as Llama Guard. 

---
# Wukong Framework for Not Safe For Work Detection in Text-to-Image systems 

**Authors**: Mingrui Liu, Sixiao Zhang, Cheng Long  

**Link**: [PDF](https://arxiv.org/pdf/2508.00591)  

**Abstract**: Text-to-Image (T2I) generation is a popular AI-generated content (AIGC) technology enabling diverse and creative image synthesis. However, some outputs may contain Not Safe For Work (NSFW) content (e.g., violence), violating community guidelines. Detecting NSFW content efficiently and accurately, known as external safeguarding, is essential. Existing external safeguards fall into two types: text filters, which analyze user prompts but overlook T2I model-specific variations and are prone to adversarial attacks; and image filters, which analyze final generated images but are computationally costly and introduce latency. Diffusion models, the foundation of modern T2I systems like Stable Diffusion, generate images through iterative denoising using a U-Net architecture with ResNet and Transformer blocks. We observe that: (1) early denoising steps define the semantic layout of the image, and (2) cross-attention layers in U-Net are crucial for aligning text and image regions. Based on these insights, we propose Wukong, a transformer-based NSFW detection framework that leverages intermediate outputs from early denoising steps and reuses U-Net's pre-trained cross-attention parameters. Wukong operates within the diffusion process, enabling early detection without waiting for full image generation. We also introduce a new dataset containing prompts, seeds, and image-specific NSFW labels, and evaluate Wukong on this and two public benchmarks. Results show that Wukong significantly outperforms text-based safeguards and achieves comparable accuracy of image filters, while offering much greater efficiency. 

---
# OmniUnet: A Multimodal Network for Unstructured Terrain Segmentation on Planetary Rovers Using RGB, Depth, and Thermal Imagery 

**Authors**: Raul Castilla-Arquillo, Carlos Perez-del-Pulgar, Levin Gerdes, Alfonso Garcia-Cerezo, Miguel A. Olivares-Mendez  

**Link**: [PDF](https://arxiv.org/pdf/2508.00580)  

**Abstract**: Robot navigation in unstructured environments requires multimodal perception systems that can support safe navigation. Multimodality enables the integration of complementary information collected by different sensors. However, this information must be processed by machine learning algorithms specifically designed to leverage heterogeneous data. Furthermore, it is necessary to identify which sensor modalities are most informative for navigation in the target environment. In Martian exploration, thermal imagery has proven valuable for assessing terrain safety due to differences in thermal behaviour between soil types. This work presents OmniUnet, a transformer-based neural network architecture for semantic segmentation using RGB, depth, and thermal (RGB-D-T) imagery. A custom multimodal sensor housing was developed using 3D printing and mounted on the Martian Rover Testbed for Autonomy (MaRTA) to collect a multimodal dataset in the Bardenas semi-desert in northern Spain. This location serves as a representative environment of the Martian surface, featuring terrain types such as sand, bedrock, and compact soil. A subset of this dataset was manually labeled to support supervised training of the network. The model was evaluated both quantitatively and qualitatively, achieving a pixel accuracy of 80.37% and demonstrating strong performance in segmenting complex unstructured terrain. Inference tests yielded an average prediction time of 673 ms on a resource-constrained computer (Jetson Orin Nano), confirming its suitability for on-robot deployment. The software implementation of the network and the labeled dataset have been made publicly available to support future research in multimodal terrain perception for planetary robotics. 

---
# Analysing Temporal Reasoning in Description Logics Using Formal Grammars 

**Authors**: Camille Bourgaux, Anton Gnatenko, Michaël Thomazo  

**Link**: [PDF](https://arxiv.org/pdf/2508.00575)  

**Abstract**: We establish a correspondence between (fragments of) $\mathcal{TEL}^\bigcirc$, a temporal extension of the $\mathcal{EL}$ description logic with the LTL operator $\bigcirc^k$, and some specific kinds of formal grammars, in particular, conjunctive grammars (context-free grammars equipped with the operation of intersection). This connection implies that $\mathcal{TEL}^\bigcirc$ does not possess the property of ultimate periodicity of models, and further leads to undecidability of query answering in $\mathcal{TEL}^\bigcirc$, closing a question left open since the introduction of $\mathcal{TEL}^\bigcirc$. Moreover, it also allows to establish decidability of query answering for some new interesting fragments of $\mathcal{TEL}^\bigcirc$, and to reuse for this purpose existing tools and algorithms for conjunctive grammars. 

---
# SynAdapt: Learning Adaptive Reasoning in Large Language Models via Synthetic Continuous Chain-of-Thought 

**Authors**: Jianwei Wang, Ziming Wu, Fuming Lai, Shaobing Lian, Ziqian Zeng  

**Link**: [PDF](https://arxiv.org/pdf/2508.00574)  

**Abstract**: While Chain-of-Thought (CoT) reasoning improves model performance, it incurs significant time costs due to the generation of discrete CoT tokens (DCoT). Continuous CoT (CCoT) offers a more efficient alternative, but existing CCoT methods are hampered by indirect fine-tuning, limited alignment, or inconsistent targets. To overcome these limitations, we propose \textit{SynAdapt}, an innovative efficient reasoning framework. Specifically, \textit{SynAdapt} generates the synthetic CCoT to serve as a precise and effective alignment target for LLMs. This synthetic CCoT explicitly guides the LLM to learn CCoT and derive accurate answers directly. Furthermore, relying solely on CCoT is insufficient for solving hard questions. To address this, \textit{SynAdapt} integrates a difficulty classifier that leverages both question context and CCoT to identify hard questions. CCoT can effectively help identify hard questions after some brief reasoning. We then adaptively prompt the LLM to re-think these hard questions for improved performance. Extensive experimental results across various benchmarks from different difficulty levels strongly demonstrate the effectiveness of our method, achieving the best accuracy-efficiency trade-off. 

---
# Activation-Guided Local Editing for Jailbreaking Attacks 

**Authors**: Jiecong Wang, Haoran Li, Hao Peng, Ziqian Zeng, Zihao Wang, Haohua Du, Zhengtao Yu  

**Link**: [PDF](https://arxiv.org/pdf/2508.00555)  

**Abstract**: Jailbreaking is an essential adversarial technique for red-teaming these models to uncover and patch security flaws. However, existing jailbreak methods face significant drawbacks. Token-level jailbreak attacks often produce incoherent or unreadable inputs and exhibit poor transferability, while prompt-level attacks lack scalability and rely heavily on manual effort and human ingenuity. We propose a concise and effective two-stage framework that combines the advantages of these approaches. The first stage performs a scenario-based generation of context and rephrases the original malicious query to obscure its harmful intent. The second stage then utilizes information from the model's hidden states to guide fine-grained edits, effectively steering the model's internal representation of the input from a malicious toward a benign one. Extensive experiments demonstrate that this method achieves state-of-the-art Attack Success Rate, with gains of up to 37.74% over the strongest baseline, and exhibits excellent transferability to black-box models. Our analysis further demonstrates that AGILE maintains substantial effectiveness against prominent defense mechanisms, highlighting the limitations of current safeguards and providing valuable insights for future defense development. Our code is available at this https URL. 

---
# SPENCER: Self-Adaptive Model Distillation for Efficient Code Retrieval 

**Authors**: Wenchao Gu, Zongyi Lyu, Yanlin Wang, Hongyu Zhang, Cuiyun Gao, Michael R. Lyu  

**Link**: [PDF](https://arxiv.org/pdf/2508.00546)  

**Abstract**: Code retrieval aims to provide users with desired code snippets based on users' natural language queries. With the development of deep learning technologies, adopting pre-trained models for this task has become mainstream. Considering the retrieval efficiency, most of the previous approaches adopt a dual-encoder for this task, which encodes the description and code snippet into representation vectors, respectively. However, the model structure of the dual-encoder tends to limit the model's performance, since it lacks the interaction between the code snippet and description at the bottom layer of the model during training. To improve the model's effectiveness while preserving its efficiency, we propose a framework, which adopts Self-AdaPtive Model Distillation for Efficient CodE Retrieval, named SPENCER. SPENCER first adopts the dual-encoder to narrow the search space and then adopts the cross-encoder to improve accuracy. To improve the efficiency of SPENCER, we propose a novel model distillation technique, which can greatly reduce the inference time of the dual-encoder while maintaining the overall performance. We also propose a teaching assistant selection strategy for our model distillation, which can adaptively select the suitable teaching assistant models for different pre-trained models during the model distillation to ensure the model performance. Extensive experiments demonstrate that the combination of dual-encoder and cross-encoder improves overall performance compared to solely dual-encoder-based models for code retrieval. Besides, our model distillation technique retains over 98% of the overall performance while reducing the inference time of the dual-encoder by 70%. 

---
# Foundations of Interpretable Models 

**Authors**: Pietro Barbiero, Mateo Espinosa Zarlenga, Alberto Termine, Mateja Jamnik, Giuseppe Marra  

**Link**: [PDF](https://arxiv.org/pdf/2508.00545)  

**Abstract**: We argue that existing definitions of interpretability are not actionable in that they fail to inform users about general, sound, and robust interpretable model design. This makes current interpretability research fundamentally ill-posed. To address this issue, we propose a definition of interpretability that is general, simple, and subsumes existing informal notions within the interpretable AI community. We show that our definition is actionable, as it directly reveals the foundational properties, underlying assumptions, principles, data structures, and architectural features necessary for designing interpretable models. Building on this, we propose a general blueprint for designing interpretable models and introduce the first open-sourced library with native support for interpretable data structures and processes. 

---
# Towards a Measure Theory of Semantic Information 

**Authors**: George M. Coghill  

**Link**: [PDF](https://arxiv.org/pdf/2508.00525)  

**Abstract**: A classic account of the quantification of semantic information is that of Bar-Hiller and Carnap. Their account proposes an inverse relation between the informativeness of a statement and its probability. However, their approach assigns the maximum informativeness to a contradiction: which Floridi refers to as the Bar-Hillel-Carnap paradox. He developed a novel theory founded on a distance metric and parabolic relation, designed to remove this paradox. Unfortunately is approach does not succeed in that aim.
In this paper I critique Floridi's theory of strongly semantic information on its own terms and show where it succeeds and fails. I then present a new approach based on the unit circle (a relation that has been the basis of theories from basic trigonometry to quantum theory). This is used, by analogy with von Neumann's quantum probability to construct a measure space for informativeness that meets all the requirements stipulated by Floridi and removes the paradox. In addition, while contradictions and tautologies have zero informativeness, it is found that messages which are contradictory to each other are equally informative. The utility of this is explained by means of an example. 

---
# LesiOnTime -- Joint Temporal and Clinical Modeling for Small Breast Lesion Segmentation in Longitudinal DCE-MRI 

**Authors**: Mohammed Kamran, Maria Bernathova, Raoul Varga, Christian Singer, Zsuzsanna Bago-Horvath, Thomas Helbich, Georg Langs, Philipp Seeböck  

**Link**: [PDF](https://arxiv.org/pdf/2508.00496)  

**Abstract**: Accurate segmentation of small lesions in Breast Dynamic Contrast-Enhanced MRI (DCE-MRI) is critical for early cancer detection, especially in high-risk patients. While recent deep learning methods have advanced lesion segmentation, they primarily target large lesions and neglect valuable longitudinal and clinical information routinely used by radiologists. In real-world screening, detecting subtle or emerging lesions requires radiologists to compare across timepoints and consider previous radiology assessments, such as the BI-RADS score. We propose LesiOnTime, a novel 3D segmentation approach that mimics clinical diagnostic workflows by jointly leveraging longitudinal imaging and BIRADS scores. The key components are: (1) a Temporal Prior Attention (TPA) block that dynamically integrates information from previous and current scans; and (2) a BI-RADS Consistency Regularization (BCR) loss that enforces latent space alignment for scans with similar radiological assessments, thus embedding domain knowledge into the training process. Evaluated on a curated in-house longitudinal dataset of high-risk patients with DCE-MRI, our approach outperforms state-of-the-art single-timepoint and longitudinal baselines by 5% in terms of Dice. Ablation studies demonstrate that both TPA and BCR contribute complementary performance gains. These results highlight the importance of incorporating temporal and clinical context for reliable early lesion segmentation in real-world breast cancer screening. Our code is publicly available at this https URL 

---
# HannesImitation: Grasping with the Hannes Prosthetic Hand via Imitation Learning 

**Authors**: Carlo Alessi, Federico Vasile, Federico Ceola, Giulia Pasquale, Nicolò Boccardo, Lorenzo Natale  

**Link**: [PDF](https://arxiv.org/pdf/2508.00491)  

**Abstract**: Recent advancements in control of prosthetic hands have focused on increasing autonomy through the use of cameras and other sensory inputs. These systems aim to reduce the cognitive load on the user by automatically controlling certain degrees of freedom. In robotics, imitation learning has emerged as a promising approach for learning grasping and complex manipulation tasks while simplifying data collection. Its application to the control of prosthetic hands remains, however, largely unexplored. Bridging this gap could enhance dexterity restoration and enable prosthetic devices to operate in more unconstrained scenarios, where tasks are learned from demonstrations rather than relying on manually annotated sequences. To this end, we present HannesImitationPolicy, an imitation learning-based method to control the Hannes prosthetic hand, enabling object grasping in unstructured environments. Moreover, we introduce the HannesImitationDataset comprising grasping demonstrations in table, shelf, and human-to-prosthesis handover scenarios. We leverage such data to train a single diffusion policy and deploy it on the prosthetic hand to predict the wrist orientation and hand closure for grasping. Experimental evaluation demonstrates successful grasps across diverse objects and conditions. Finally, we show that the policy outperforms a segmentation-based visual servo controller in unstructured scenarios. Additional material is provided on our project page: this https URL 

---
# CyGATE: Game-Theoretic Cyber Attack-Defense Engine for Patch Strategy Optimization 

**Authors**: Yuning Jiang, Nay Oo, Qiaoran Meng, Lu Lin, Dusit Niyato, Zehui Xiong, Hoon Wei Lim, Biplab Sikdar  

**Link**: [PDF](https://arxiv.org/pdf/2508.00478)  

**Abstract**: Modern cyber attacks unfold through multiple stages, requiring defenders to dynamically prioritize mitigations under uncertainty. While game-theoretic models capture attacker-defender interactions, existing approaches often rely on static assumptions and lack integration with real-time threat intelligence, limiting their adaptability. This paper presents CyGATE, a game-theoretic framework modeling attacker-defender interactions, using large language models (LLMs) with retrieval-augmented generation (RAG) to enhance tactic selection and patch prioritization. Applied to a two-agent scenario, CyGATE frames cyber conflicts as a partially observable stochastic game (POSG) across Cyber Kill Chain stages. Both agents use belief states to navigate uncertainty, with the attacker adapting tactics and the defender re-prioritizing patches based on evolving risks and observed adversary behavior. The framework's flexible architecture enables extension to multi-agent scenarios involving coordinated attackers, collaborative defenders, or complex enterprise environments with multiple stakeholders. Evaluated in a dynamic patch scheduling scenario, CyGATE effectively prioritizes high-risk vulnerabilities, enhancing adaptability through dynamic threat integration, strategic foresight by anticipating attacker moves under uncertainty, and efficiency by optimizing resource use. 

---
# M^2VAE: Multi-Modal Multi-View Variational Autoencoder for Cold-start Item Recommendation 

**Authors**: Chuan He, Yongchao Liu, Qiang Li, Wenliang Zhong, Chuntao Hong, Xinwei Yao  

**Link**: [PDF](https://arxiv.org/pdf/2508.00452)  

**Abstract**: Cold-start item recommendation is a significant challenge in recommendation systems, particularly when new items are introduced without any historical interaction data. While existing methods leverage multi-modal content to alleviate the cold-start issue, they often neglect the inherent multi-view structure of modalities, the distinction between shared and modality-specific features. In this paper, we propose Multi-Modal Multi-View Variational AutoEncoder (M^2VAE), a generative model that addresses the challenges of modeling common and unique views in attribute and multi-modal features, as well as user preferences over single-typed item features. Specifically, we generate type-specific latent variables for item IDs, categorical attributes, and image features, and use Product-of-Experts (PoE) to derive a common representation. A disentangled contrastive loss decouples the common view from unique views while preserving feature informativeness. To model user inclinations, we employ a preference-guided Mixture-of-Experts (MoE) to adaptively fuse representations. We further incorporate co-occurrence signals via contrastive learning, eliminating the need for pretraining. Extensive experiments on real-world datasets validate the effectiveness of our approach. 

---
# When Relevance Meets Novelty: Dual-Stable Periodic Optimization for Exploratory Recommendation 

**Authors**: Hongxiang Lin, Hao Guo, Zeshun Li, Erpeng Xue, Yongqian He, Xiangyu Hou, Zhaoyu Hu, Lei Wang, Sheng Chen  

**Link**: [PDF](https://arxiv.org/pdf/2508.00450)  

**Abstract**: Traditional recommendation systems tend to trap users in strong feedback loops by excessively pushing content aligned with their historical preferences, thereby limiting exploration opportunities and causing content fatigue. Although large language models (LLMs) demonstrate potential with their diverse content generation capabilities, existing LLM-enhanced dual-model frameworks face two major limitations: first, they overlook long-term preferences driven by group identity, leading to biased interest modeling; second, they suffer from static optimization flaws, as a one-time alignment process fails to leverage incremental user data for closed-loop optimization. To address these challenges, we propose the Co-Evolutionary Alignment (CoEA) method. For interest modeling bias, we introduce Dual-Stable Interest Exploration (DSIE) module, jointly modeling long-term group identity and short-term individual interests through parallel processing of behavioral sequences. For static optimization limitations, we design a Periodic Collaborative Optimization (PCO) mechanism. This mechanism regularly conducts preference verification on incremental data using the Relevance LLM, then guides the Novelty LLM to perform fine-tuning based on the verification results, and subsequently feeds back the output of the incrementally fine-tuned Novelty LLM to the Relevance LLM for re-evaluation, thereby achieving a dynamic closed-loop optimization. Extensive online and offline experiments verify the effectiveness of the CoEA model in exploratory recommendation. 

---
# TopoTTA: Topology-Enhanced Test-Time Adaptation for Tubular Structure Segmentation 

**Authors**: Jiale Zhou, Wenhan Wang, Shikun Li, Xiaolei Qu, Xin Guo, Yizhong Liu, Wenzhong Tang, Xun Lin, Yefeng Zheng  

**Link**: [PDF](https://arxiv.org/pdf/2508.00442)  

**Abstract**: Tubular structure segmentation (TSS) is important for various applications, such as hemodynamic analysis and route navigation. Despite significant progress in TSS, domain shifts remain a major challenge, leading to performance degradation in unseen target domains. Unlike other segmentation tasks, TSS is more sensitive to domain shifts, as changes in topological structures can compromise segmentation integrity, and variations in local features distinguishing foreground from background (e.g., texture and contrast) may further disrupt topological continuity. To address these challenges, we propose Topology-enhanced Test-Time Adaptation (TopoTTA), the first test-time adaptation framework designed specifically for TSS. TopoTTA consists of two stages: Stage 1 adapts models to cross-domain topological discrepancies using the proposed Topological Meta Difference Convolutions (TopoMDCs), which enhance topological representation without altering pre-trained parameters; Stage 2 improves topological continuity by a novel Topology Hard sample Generation (TopoHG) strategy and prediction alignment on hard samples with pseudo-labels in the generated pseudo-break regions. Extensive experiments across four scenarios and ten datasets demonstrate TopoTTA's effectiveness in handling topological distribution shifts, achieving an average improvement of 31.81% in clDice. TopoTTA also serves as a plug-and-play TTA solution for CNN-based TSS models. 

---
# Reducing the gap between general purpose data and aerial images in concentrated solar power plants 

**Authors**: M.A. Pérez-Cutiño, J. Valverde, J. Capitán, J.M. Díaz-Báñez  

**Link**: [PDF](https://arxiv.org/pdf/2508.00440)  

**Abstract**: In the context of Concentrated Solar Power (CSP) plants, aerial images captured by drones present a unique set of challenges. Unlike urban or natural landscapes commonly found in existing datasets, solar fields contain highly reflective surfaces, and domain-specific elements that are uncommon in traditional computer vision benchmarks. As a result, machine learning models trained on generic datasets struggle to generalize to this setting without extensive retraining and large volumes of annotated data. However, collecting and labeling such data is costly and time-consuming, making it impractical for rapid deployment in industrial applications.
To address this issue, we propose a novel approach: the creation of AerialCSP, a virtual dataset that simulates aerial imagery of CSP plants. By generating synthetic data that closely mimic real-world conditions, our objective is to facilitate pretraining of models before deployment, significantly reducing the need for extensive manual labeling. Our main contributions are threefold: (1) we introduce AerialCSP, a high-quality synthetic dataset for aerial inspection of CSP plants, providing annotated data for object detection and image segmentation; (2) we benchmark multiple models on AerialCSP, establishing a baseline for CSP-related vision tasks; and (3) we demonstrate that pretraining on AerialCSP significantly improves real-world fault detection, particularly for rare and small defects, reducing the need for extensive manual labeling. AerialCSP is made publicly available at this https URL. 

---
# Contact-Aware Amodal Completion for Human-Object Interaction via Multi-Regional Inpainting 

**Authors**: Seunggeun Chi, Enna Sachdeva, Pin-Hao Huang, Kwonjoon Lee  

**Link**: [PDF](https://arxiv.org/pdf/2508.00427)  

**Abstract**: Amodal completion, which is the process of inferring the full appearance of objects despite partial occlusions, is crucial for understanding complex human-object interactions (HOI) in computer vision and robotics. Existing methods, such as those that use pre-trained diffusion models, often struggle to generate plausible completions in dynamic scenarios because they have a limited understanding of HOI. To solve this problem, we've developed a new approach that uses physical prior knowledge along with a specialized multi-regional inpainting technique designed for HOI. By incorporating physical constraints from human topology and contact information, we define two distinct regions: the primary region, where occluded object parts are most likely to be, and the secondary region, where occlusions are less probable. Our multi-regional inpainting method uses customized denoising strategies across these regions within a diffusion model. This improves the accuracy and realism of the generated completions in both their shape and visual detail. Our experimental results show that our approach significantly outperforms existing methods in HOI scenarios, moving machine perception closer to a more human-like understanding of dynamic environments. We also show that our pipeline is robust even without ground-truth contact annotations, which broadens its applicability to tasks like 3D reconstruction and novel view/pose synthesis. 

---
# DC-AE 1.5: Accelerating Diffusion Model Convergence with Structured Latent Space 

**Authors**: Junyu Chen, Dongyun Zou, Wenkun He, Junsong Chen, Enze Xie, Song Han, Han Cai  

**Link**: [PDF](https://arxiv.org/pdf/2508.00413)  

**Abstract**: We present DC-AE 1.5, a new family of deep compression autoencoders for high-resolution diffusion models. Increasing the autoencoder's latent channel number is a highly effective approach for improving its reconstruction quality. However, it results in slow convergence for diffusion models, leading to poorer generation quality despite better reconstruction quality. This issue limits the quality upper bound of latent diffusion models and hinders the employment of autoencoders with higher spatial compression ratios. We introduce two key innovations to address this challenge: i) Structured Latent Space, a training-based approach to impose a desired channel-wise structure on the latent space with front latent channels capturing object structures and latter latent channels capturing image details; ii) Augmented Diffusion Training, an augmented diffusion training strategy with additional diffusion training objectives on object latent channels to accelerate convergence. With these techniques, DC-AE 1.5 delivers faster convergence and better diffusion scaling results than DC-AE. On ImageNet 512x512, DC-AE-1.5-f64c128 delivers better image generation quality than DC-AE-f32c32 while being 4x faster. Code: this https URL. 

---
# Decouple before Align: Visual Disentanglement Enhances Prompt Tuning 

**Authors**: Fei Zhang, Tianfei Zhou, Jiangchao Yao, Ya Zhang, Ivor W. Tsang, Yanfeng Wang  

**Link**: [PDF](https://arxiv.org/pdf/2508.00395)  

**Abstract**: Prompt tuning (PT), as an emerging resource-efficient fine-tuning paradigm, has showcased remarkable effectiveness in improving the task-specific transferability of vision-language models. This paper delves into a previously overlooked information asymmetry issue in PT, where the visual modality mostly conveys more context than the object-oriented textual modality. Correspondingly, coarsely aligning these two modalities could result in the biased attention, driving the model to merely focus on the context area. To address this, we propose DAPT, an effective PT framework based on an intuitive decouple-before-align concept. First, we propose to explicitly decouple the visual modality into the foreground and background representation via exploiting coarse-and-fine visual segmenting cues, and then both of these decoupled patterns are aligned with the original foreground texts and the hand-crafted background classes, thereby symmetrically strengthening the modal alignment. To further enhance the visual concentration, we propose a visual pull-push regularization tailored for the foreground-background patterns, directing the original visual representation towards unbiased attention on the region-of-interest object. We demonstrate the power of architecture-free DAPT through few-shot learning, base-to-novel generalization, and data-efficient learning, all of which yield superior performance across prevailing benchmarks. Our code will be released at this https URL. 

---
# ExeKGLib: A Platform for Machine Learning Analytics based on Knowledge Graphs 

**Authors**: Antonis Klironomos, Baifan Zhou, Zhipeng Tan, Zhuoxun Zheng, Mohamed H. Gad-Elrab, Heiko Paulheim, Evgeny Kharlamov  

**Link**: [PDF](https://arxiv.org/pdf/2508.00394)  

**Abstract**: Nowadays machine learning (ML) practitioners have access to numerous ML libraries available online. Such libraries can be used to create ML pipelines that consist of a series of steps where each step may invoke up to several ML libraries that are used for various data-driven analytical tasks. Development of high-quality ML pipelines is non-trivial; it requires training, ML expertise, and careful development of each step. At the same time, domain experts in science and engineering may not possess such ML expertise and training while they are in pressing need of ML-based analytics. In this paper, we present our ExeKGLib, a Python library enhanced with a graphical interface layer that allows users with minimal ML knowledge to build ML pipelines. This is achieved by relying on knowledge graphs that encode ML knowledge in simple terms accessible to non-ML experts. ExeKGLib also allows improving the transparency and reusability of the built ML workflows and ensures that they are executable. We show the usability and usefulness of ExeKGLib by presenting real use cases. 

---
# $MV_{Hybrid}$: Improving Spatial Transcriptomics Prediction with Hybrid State Space-Vision Transformer Backbone in Pathology Vision Foundation Models 

**Authors**: Won June Cho, Hongjun Yoon, Daeky Jeong, Hyeongyeol Lim, Yosep Chong  

**Link**: [PDF](https://arxiv.org/pdf/2508.00383)  

**Abstract**: Spatial transcriptomics reveals gene expression patterns within tissue context, enabling precision oncology applications such as treatment response prediction, but its high cost and technical complexity limit clinical adoption. Predicting spatial gene expression (biomarkers) from routine histopathology images offers a practical alternative, yet current vision foundation models (VFMs) in pathology based on Vision Transformer (ViT) backbones perform below clinical standards. Given that VFMs are already trained on millions of diverse whole slide images, we hypothesize that architectural innovations beyond ViTs may better capture the low-frequency, subtle morphological patterns correlating with molecular phenotypes. By demonstrating that state space models initialized with negative real eigenvalues exhibit strong low-frequency bias, we introduce $MV_{Hybrid}$, a hybrid backbone architecture combining state space models (SSMs) with ViT. We compare five other different backbone architectures for pathology VFMs, all pretrained on identical colorectal cancer datasets using the DINOv2 self-supervised learning method. We evaluate all pretrained models using both random split and leave-one-study-out (LOSO) settings of the same biomarker dataset. In LOSO evaluation, $MV_{Hybrid}$ achieves 57% higher correlation than the best-performing ViT and shows 43% smaller performance degradation compared to random split in gene expression prediction, demonstrating superior performance and robustness, respectively. Furthermore, $MV_{Hybrid}$ shows equal or better downstream performance in classification, patch retrieval, and survival prediction tasks compared to that of ViT, showing its promise as a next-generation pathology VFM backbone. Our code is publicly available at: this https URL. 

---
# Advancing Welding Defect Detection in Maritime Operations via Adapt-WeldNet and Defect Detection Interpretability Analysis 

**Authors**: Kamal Basha S, Athira Nambiar  

**Link**: [PDF](https://arxiv.org/pdf/2508.00381)  

**Abstract**: Weld defect detection is crucial for ensuring the safety and reliability of piping systems in the oil and gas industry, especially in challenging marine and offshore environments. Traditional non-destructive testing (NDT) methods often fail to detect subtle or internal defects, leading to potential failures and costly downtime. Furthermore, existing neural network-based approaches for defect classification frequently rely on arbitrarily selected pretrained architectures and lack interpretability, raising safety concerns for deployment. To address these challenges, this paper introduces ``Adapt-WeldNet", an adaptive framework for welding defect detection that systematically evaluates various pre-trained architectures, transfer learning strategies, and adaptive optimizers to identify the best-performing model and hyperparameters, optimizing defect detection and providing actionable insights. Additionally, a novel Defect Detection Interpretability Analysis (DDIA) framework is proposed to enhance system transparency. DDIA employs Explainable AI (XAI) techniques, such as Grad-CAM and LIME, alongside domain-specific evaluations validated by certified ASNT NDE Level II professionals. Incorporating a Human-in-the-Loop (HITL) approach and aligning with the principles of Trustworthy AI, DDIA ensures the reliability, fairness, and accountability of the defect detection system, fostering confidence in automated decisions through expert validation. By improving both performance and interpretability, this work enhances trust, safety, and reliability in welding defect detection systems, supporting critical operations in offshore and marine environments. 

---
# GV-VAD : Exploring Video Generation for Weakly-Supervised Video Anomaly Detection 

**Authors**: Suhang Cai, Xiaohao Peng, Chong Wang, Xiaojie Cai, Jiangbo Qian  

**Link**: [PDF](https://arxiv.org/pdf/2508.00312)  

**Abstract**: Video anomaly detection (VAD) plays a critical role in public safety applications such as intelligent surveillance. However, the rarity, unpredictability, and high annotation cost of real-world anomalies make it difficult to scale VAD datasets, which limits the performance and generalization ability of existing models. To address this challenge, we propose a generative video-enhanced weakly-supervised video anomaly detection (GV-VAD) framework that leverages text-conditioned video generation models to produce semantically controllable and physically plausible synthetic videos. These virtual videos are used to augment training data at low cost. In addition, a synthetic sample loss scaling strategy is utilized to control the influence of generated synthetic samples for efficient training. The experiments show that the proposed framework outperforms state-of-the-art methods on UCF-Crime datasets. The code is available at this https URL. 

---
# Beamformed 360° Sound Maps: U-Net-Driven Acoustic Source Segmentation and Localization 

**Authors**: Belman Jahir Rodriguez, Sergio F. Chevtchenko, Marcelo Herrera Martinez, Yeshwant Bethy, Saeed Afshar  

**Link**: [PDF](https://arxiv.org/pdf/2508.00307)  

**Abstract**: We introduce a U-net model for 360° acoustic source localization formulated as a spherical semantic segmentation task. Rather than regressing discrete direction-of-arrival (DoA) angles, our model segments beamformed audio maps (azimuth and elevation) into regions of active sound presence. Using delay-and-sum (DAS) beamforming on a custom 24-microphone array, we generate signals aligned with drone GPS telemetry to create binary supervision masks. A modified U-Net, trained on frequency-domain representations of these maps, learns to identify spatially distributed source regions while addressing class imbalance via the Tversky loss. Because the network operates on beamformed energy maps, the approach is inherently array-independent and can adapt to different microphone configurations without retraining from scratch. The segmentation outputs are post-processed by computing centroids over activated regions, enabling robust DoA estimates. Our dataset includes real-world open-field recordings of a DJI Air 3 drone, synchronized with 360° video and flight logs across multiple dates and locations. Experimental results show that U-net generalizes across environments, providing improved angular precision, offering a new paradigm for dense spatial audio understanding beyond traditional Sound Source Localization (SSL). 

---
# MetaExplainer: A Framework to Generate Multi-Type User-Centered Explanations for AI Systems 

**Authors**: Shruthi Chari, Oshani Seneviratne, Prithwish Chakraborty, Pablo Meyer, Deborah L. McGuinness  

**Link**: [PDF](https://arxiv.org/pdf/2508.00300)  

**Abstract**: Explanations are crucial for building trustworthy AI systems, but a gap often exists between the explanations provided by models and those needed by users. To address this gap, we introduce MetaExplainer, a neuro-symbolic framework designed to generate user-centered explanations. Our approach employs a three-stage process: first, we decompose user questions into machine-readable formats using state-of-the-art large language models (LLM); second, we delegate the task of generating system recommendations to model explainer methods; and finally, we synthesize natural language explanations that summarize the explainer outputs. Throughout this process, we utilize an Explanation Ontology to guide the language models and explainer methods. By leveraging LLMs and a structured approach to explanation generation, MetaExplainer aims to enhance the interpretability and trustworthiness of AI systems across various applications, providing users with tailored, question-driven explanations that better meet their needs. Comprehensive evaluations of MetaExplainer demonstrate a step towards evaluating and utilizing current state-of-the-art explanation frameworks. Our results show high performance across all stages, with a 59.06% F1-score in question reframing, 70% faithfulness in model explanations, and 67% context-utilization in natural language synthesis. User studies corroborate these findings, highlighting the creativity and comprehensiveness of generated explanations. Tested on the Diabetes (PIMA Indian) tabular dataset, MetaExplainer supports diverse explanation types, including Contrastive, Counterfactual, Rationale, Case-Based, and Data explanations. The framework's versatility and traceability from using ontology to guide LLMs suggest broad applicability beyond the tested scenarios, positioning MetaExplainer as a promising tool for enhancing AI explainability across various domains. 

---
# Controllable Pedestrian Video Editing for Multi-View Driving Scenarios via Motion Sequence 

**Authors**: Danzhen Fu, Jiagao Hu, Daiguo Zhou, Fei Wang, Zepeng Wang, Wenhua Liao  

**Link**: [PDF](https://arxiv.org/pdf/2508.00299)  

**Abstract**: Pedestrian detection models in autonomous driving systems often lack robustness due to insufficient representation of dangerous pedestrian scenarios in training datasets. To address this limitation, we present a novel framework for controllable pedestrian video editing in multi-view driving scenarios by integrating video inpainting and human motion control techniques. Our approach begins by identifying pedestrian regions of interest across multiple camera views, expanding detection bounding boxes with a fixed ratio, and resizing and stitching these regions into a unified canvas while preserving cross-view spatial relationships. A binary mask is then applied to designate the editable area, within which pedestrian editing is guided by pose sequence control conditions. This enables flexible editing functionalities, including pedestrian insertion, replacement, and removal. Extensive experiments demonstrate that our framework achieves high-quality pedestrian editing with strong visual realism, spatiotemporal coherence, and cross-view consistency. These results establish the proposed method as a robust and versatile solution for multi-view pedestrian video generation, with broad potential for applications in data augmentation and scenario simulation in autonomous driving. 

---
# Formal Power Series Representations in Probability and Expected Utility Theory 

**Authors**: Arthur Paul Pedersen, Samuel Allen Alexander  

**Link**: [PDF](https://arxiv.org/pdf/2508.00294)  

**Abstract**: We advance a general theory of coherent preference that surrenders restrictions embodied in orthodox doctrine. This theory enjoys the property that any preference system admits extension to a complete system of preferences, provided it satisfies a certain coherence requirement analogous to the one de Finetti advanced for his foundations of probability. Unlike de Finetti's theory, the one we set forth requires neither transitivity nor Archimedeanness nor boundedness nor continuity of preference. This theory also enjoys the property that any complete preference system meeting the standard of coherence can be represented by utility in an ordered field extension of the reals. Representability by utility is a corollary of this paper's central result, which at once extends Hölder's Theorem and strengthens Hahn's Embedding Theorem. 

---
# Calibrated Language Models and How to Find Them with Label Smoothing 

**Authors**: Jerry Huang, Peng Lu, Qiuhao Zeng  

**Link**: [PDF](https://arxiv.org/pdf/2508.00264)  

**Abstract**: Recent advances in natural language processing (NLP) have opened up greater opportunities to enable fine-tuned large language models (LLMs) to behave as more powerful interactive agents through improved instruction-following ability. However, understanding how this impacts confidence calibration for reliable model output has not been researched in full. In this work, we examine various open-sourced LLMs, identifying significant calibration degradation after instruction tuning in each. Seeking a practical solution, we look towards label smoothing, which has been shown as an effective method to regularize for overconfident predictions but has yet to be widely adopted in the supervised fine-tuning (SFT) of LLMs. We first provide insight as to why label smoothing is sufficient to maintain calibration throughout the SFT process. However, settings remain where the effectiveness of smoothing is severely diminished, in particular the case of large vocabulary LLMs (LV-LLMs). We posit the cause to stem from the ability to become over-confident, which has a direct relationship with the hidden size and vocabulary size, and justify this theoretically and experimentally. Finally, we address an outstanding issue regarding the memory footprint of the cross-entropy loss computation in the label smoothed loss setting, designing a customized kernel to dramatically reduce memory consumption without sacrificing speed or performance in comparison to existing solutions for non-smoothed losses. 

---
# Large AI Model-Enabled Secure Communications in Low-Altitude Wireless Networks: Concepts, Perspectives and Case Study 

**Authors**: Chuang Zhang, Geng Sun, Jiacheng Wang, Yijing Lin, Weijie Yuan, Sinem Coleri, Dusit Niyato, Tony Q. S. Quek  

**Link**: [PDF](https://arxiv.org/pdf/2508.00256)  

**Abstract**: Low-altitude wireless networks (LAWNs) have the potential to revolutionize communications by supporting a range of applications, including urban parcel delivery, aerial inspections and air taxis. However, compared with traditional wireless networks, LAWNs face unique security challenges due to low-altitude operations, frequent mobility and reliance on unlicensed spectrum, making it more vulnerable to some malicious attacks. In this paper, we investigate some large artificial intelligence model (LAM)-enabled solutions for secure communications in LAWNs. Specifically, we first explore the amplified security risks and important limitations of traditional AI methods in LAWNs. Then, we introduce the basic concepts of LAMs and delve into the role of LAMs in addressing these challenges. To demonstrate the practical benefits of LAMs for secure communications in LAWNs, we propose a novel LAM-based optimization framework that leverages large language models (LLMs) to generate enhanced state features on top of handcrafted representations, and to design intrinsic rewards accordingly, thereby improving reinforcement learning performance for secure communication tasks. Through a typical case study, simulation results validate the effectiveness of the proposed framework. Finally, we outline future directions for integrating LAMs into secure LAWN applications. 

---
# Accurate and Consistent Graph Model Generation from Text with Large Language Models 

**Authors**: Boqi Chen, Ou Wei, Bingzhou Zheng, Gunter Mussbacher  

**Link**: [PDF](https://arxiv.org/pdf/2508.00255)  

**Abstract**: Graph model generation from natural language description is an important task with many applications in software engineering. With the rise of large language models (LLMs), there is a growing interest in using LLMs for graph model generation. Nevertheless, LLM-based graph model generation typically produces partially correct models that suffer from three main issues: (1) syntax violations: the generated model may not adhere to the syntax defined by its metamodel, (2) constraint inconsistencies: the structure of the model might not conform to some domain-specific constraints, and (3) inaccuracy: due to the inherent uncertainty in LLMs, the models can include inaccurate, hallucinated elements. While the first issue is often addressed through techniques such as constraint decoding or filtering, the latter two remain largely unaddressed. Motivated by recent self-consistency approaches in LLMs, we propose a novel abstraction-concretization framework that enhances the consistency and quality of generated graph models by considering multiple outputs from an LLM. Our approach first constructs a probabilistic partial model that aggregates all candidate outputs and then refines this partial model into the most appropriate concrete model that satisfies all constraints. We evaluate our framework on several popular open-source and closed-source LLMs using diverse datasets for model generation tasks. The results demonstrate that our approach significantly improves both the consistency and quality of the generated graph models. 

---
# Jet Image Generation in High Energy Physics Using Diffusion Models 

**Authors**: Victor D. Martinez, Vidya Manian, Sudhir Malik  

**Link**: [PDF](https://arxiv.org/pdf/2508.00250)  

**Abstract**: This article presents, for the first time, the application of diffusion models for generating jet images corresponding to proton-proton collision events at the Large Hadron Collider (LHC). The kinematic variables of quark, gluon, W-boson, Z-boson, and top quark jets from the JetNet simulation dataset are mapped to two-dimensional image representations. Diffusion models are trained on these images to learn the spatial distribution of jet constituents. We compare the performance of score-based diffusion models and consistency models in accurately generating class-conditional jet images. Unlike approaches based on latent distributions, our method operates directly in image space. The fidelity of the generated images is evaluated using several metrics, including the Fréchet Inception Distance (FID), which demonstrates that consistency models achieve higher fidelity and generation stability compared to score-based diffusion models. These advancements offer significant improvements in computational efficiency and generation accuracy, providing valuable tools for High Energy Physics (HEP) research. 

---
# What's Behind the Magic? Audiences Seek Artistic Value in Generative AI's Contributions to a Live Dance Performance 

**Authors**: Jacqueline Elise Bruen, Myounghoon Jeon  

**Link**: [PDF](https://arxiv.org/pdf/2508.00239)  

**Abstract**: With the development of generative artificial intelligence (GenAI) tools to create art, stakeholders cannot come to an agreement on the value of these works. In this study we uncovered the mixed opinions surrounding art made by AI. We developed two versions of a dance performance augmented by technology either with or without GenAI. For each version we informed audiences of the performance's development either before or after a survey on their perceptions of the performance. There were thirty-nine participants (13 males, 26 female) divided between the four performances. Results demonstrated that individuals were more inclined to attribute artistic merit to works made by GenAI when they were unaware of its use. We present this case study as a call to address the importance of utilizing the social context and the users' interpretations of GenAI in shaping a technical explanation, leading to a greater discussion that can bridge gaps in understanding. 

---
# Model Misalignment and Language Change: Traces of AI-Associated Language in Unscripted Spoken English 

**Authors**: Bryce Anderson, Riley Galpin, Tom S. Juzek  

**Link**: [PDF](https://arxiv.org/pdf/2508.00238)  

**Abstract**: In recent years, written language, particularly in science and education, has undergone remarkable shifts in word usage. These changes are widely attributed to the growing influence of Large Language Models (LLMs), which frequently rely on a distinct lexical style. Divergences between model output and target audience norms can be viewed as a form of misalignment. While these shifts are often linked to using Artificial Intelligence (AI) directly as a tool to generate text, it remains unclear whether the changes reflect broader changes in the human language system itself. To explore this question, we constructed a dataset of 22.1 million words from unscripted spoken language drawn from conversational science and technology podcasts. We analyzed lexical trends before and after ChatGPT's release in 2022, focusing on commonly LLM-associated words. Our results show a moderate yet significant increase in the usage of these words post-2022, suggesting a convergence between human word choices and LLM-associated patterns. In contrast, baseline synonym words exhibit no significant directional shift. Given the short time frame and the number of words affected, this may indicate the onset of a remarkable shift in language use. Whether this represents natural language change or a novel shift driven by AI exposure remains an open question. Similarly, although the shifts may stem from broader adoption patterns, it may also be that upstream training misalignments ultimately contribute to changes in human language use. These findings parallel ethical concerns that misaligned models may shape social and moral beliefs. 

---
# Weakly Supervised Intracranial Aneurysm Detection and Segmentation in MR angiography via Multi-task UNet with Vesselness Prior 

**Authors**: Erin Rainville, Amirhossein Rasoulian, Hassan Rivaz, Yiming Xiao  

**Link**: [PDF](https://arxiv.org/pdf/2508.00235)  

**Abstract**: Intracranial aneurysms (IAs) are abnormal dilations of cerebral blood vessels that, if ruptured, can lead to life-threatening consequences. However, their small size and soft contrast in radiological scans often make it difficult to perform accurate and efficient detection and morphological analyses, which are critical in the clinical care of the disorder. Furthermore, the lack of large public datasets with voxel-wise expert annotations pose challenges for developing deep learning algorithms to address the issues. Therefore, we proposed a novel weakly supervised 3D multi-task UNet that integrates vesselness priors to jointly perform aneurysm detection and segmentation in time-of-flight MR angiography (TOF-MRA). Specifically, to robustly guide IA detection and segmentation, we employ the popular Frangi's vesselness filter to derive soft cerebrovascular priors for both network input and an attention block to conduct segmentation from the decoder and detection from an auxiliary branch. We train our model on the Lausanne dataset with coarse ground truth segmentation, and evaluate it on the test set with refined labels from the same database. To further assess our model's generalizability, we also validate it externally on the ADAM dataset. Our results demonstrate the superior performance of the proposed technique over the SOTA techniques for aneurysm segmentation (Dice = 0.614, 95%HD =1.38mm) and detection (false positive rate = 1.47, sensitivity = 92.9%). 

---
# Quality-of-Service Aware LLM Routing for Edge Computing with Multiple Experts 

**Authors**: Jin Yang, Qiong Wu, Zhiying Feng, Zhi Zhou, Deke Guo, Xu Chen  

**Link**: [PDF](https://arxiv.org/pdf/2508.00234)  

**Abstract**: Large Language Models (LLMs) have demonstrated remarkable capabilities, leading to a significant increase in user demand for LLM services. However, cloud-based LLM services often suffer from high latency, unstable responsiveness, and privacy concerns. Therefore, multiple LLMs are usually deployed at the network edge to boost real-time responsiveness and protect data privacy, particularly for many emerging smart mobile and IoT applications. Given the varying response quality and latency of LLM services, a critical issue is how to route user requests from mobile and IoT devices to an appropriate LLM service (i.e., edge LLM expert) to ensure acceptable quality-of-service (QoS). Existing routing algorithms fail to simultaneously address the heterogeneity of LLM services, the interference among requests, and the dynamic workloads necessary for maintaining long-term stable QoS. To meet these challenges, in this paper we propose a novel deep reinforcement learning (DRL)-based QoS-aware LLM routing framework for sustained high-quality LLM services. Due to the dynamic nature of the global state, we propose a dynamic state abstraction technique to compactly represent global state features with a heterogeneous graph attention network (HAN). Additionally, we introduce an action impact estimator and a tailored reward function to guide the DRL agent in maximizing QoS and preventing latency violations. Extensive experiments on both Poisson and real-world workloads demonstrate that our proposed algorithm significantly improves average QoS and computing resource efficiency compared to existing baselines. 

---
# Reinitializing weights vs units for maintaining plasticity in neural networks 

**Authors**: J. Fernando Hernandez-Garcia, Shibhansh Dohare, Jun Luo, Rich S. Sutton  

**Link**: [PDF](https://arxiv.org/pdf/2508.00212)  

**Abstract**: Loss of plasticity is a phenomenon in which a neural network loses its ability to learn when trained for an extended time on non-stationary data. It is a crucial problem to overcome when designing systems that learn continually. An effective technique for preventing loss of plasticity is reinitializing parts of the network. In this paper, we compare two different reinitialization schemes: reinitializing units vs reinitializing weights. We propose a new algorithm, which we name \textit{selective weight reinitialization}, for reinitializing the least useful weights in a network. We compare our algorithm to continual backpropagation and ReDo, two previously proposed algorithms that reinitialize units in the network. Through our experiments in continual supervised learning problems, we identify two settings when reinitializing weights is more effective at maintaining plasticity than reinitializing units: (1) when the network has a small number of units and (2) when the network includes layer normalization. Conversely, reinitializing weights and units are equally effective at maintaining plasticity when the network is of sufficient size and does not include layer normalization. We found that reinitializing weights maintains plasticity in a wider variety of settings than reinitializing units. 

---
# Robust Classification under Noisy Labels: A Geometry-Aware Reliability Framework for Foundation Models 

**Authors**: Ecem Bozkurt, Antonio Ortega  

**Link**: [PDF](https://arxiv.org/pdf/2508.00202)  

**Abstract**: Foundation models (FMs) pretrained on large datasets have become fundamental for various downstream machine learning tasks, in particular in scenarios where obtaining perfectly labeled data is prohibitively expensive. In this paper, we assume an FM has to be fine-tuned with noisy data and present a two-stage framework to ensure robust classification in the presence of label noise without model retraining. Recent work has shown that simple k-nearest neighbor (kNN) approaches using an embedding derived from an FM can achieve good performance even in the presence of severe label noise. Our work is motivated by the fact that these methods make use of local geometry. In this paper, following a similar two-stage procedure, reliability estimation followed by reliability-weighted inference, we show that improved performance can be achieved by introducing geometry information. For a given instance, our proposed inference uses a local neighborhood of training data, obtained using the non-negative kernel (NNK) neighborhood construction. We propose several methods for reliability estimation that can rely less on distance and local neighborhood as the label noise increases. Our evaluation on CIFAR-10 and DermaMNIST shows that our methods improve robustness across various noise conditions, surpassing standard K-NN approaches and recent adaptive-neighborhood baselines. 

---
# EMA Without the Lag: Bias-Corrected Iterate Averaging Schemes 

**Authors**: Adam Block, Cyril Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2508.00180)  

**Abstract**: Stochasticity in language model fine-tuning, often caused by the small batch sizes typically used in this regime, can destabilize training by introducing large oscillations in generation quality. A popular approach to mitigating this instability is to take an Exponential moving average (EMA) of weights throughout training. While EMA reduces stochasticity, thereby smoothing training, the introduction of bias from old iterates often creates a lag in optimization relative to vanilla training. In this work, we propose the Bias-Corrected Exponential Moving Average (BEMA), a simple and practical augmentation of EMA that retains variance-reduction benefits while eliminating bias. BEMA is motivated by a simple theoretical model wherein we demonstrate provable acceleration of BEMA over both a standard EMA and vanilla training. Through an extensive suite of experiments on Language Models, we show that BEMA leads to significantly improved convergence rates and final performance over both EMA and vanilla training in a variety of standard LM benchmarks, making BEMA a practical and theoretically motivated intervention for more stable and efficient fine-tuning. 

---
# The SPACE of AI: Real-World Lessons on AI's Impact on Developers 

**Authors**: Brian Houck, Travis Lowdermilk, Cody Beyer, Steven Clarke, Ben Hanrahan  

**Link**: [PDF](https://arxiv.org/pdf/2508.00178)  

**Abstract**: As artificial intelligence (AI) tools become increasingly embedded in software development workflows, questions persist about their true impact on developer productivity and experience. This paper presents findings from a mixed-methods study examining how developers perceive AI's influence across the dimensions of the SPACE framework: Satisfaction, Performance, Activity, Collaboration and Efficiency. Drawing on survey responses from over 500 developers and qualitative insights from interviews and observational studies, we find that AI is broadly adopted and widely seen as enhancing productivity, particularly for routine tasks. However, the benefits vary, depending on task complexity, individual usage patterns, and team-level adoption. Developers report increased efficiency and satisfaction, with less evidence of impact on collaboration. Organizational support and peer learning play key roles in maximizing AI's value. These findings suggest that AI is augmenting developers rather than replacing them, and that effective integration depends as much on team culture and support structures as on the tools themselves. We conclude with practical recommendations for teams, organizations and researchers seeking to harness AI's potential in software engineering. 

---
# DeformTune: A Deformable XAI Music Prototype for Non-Musicians 

**Authors**: Ziqing Xu, Nick Bryan-Kinns  

**Link**: [PDF](https://arxiv.org/pdf/2508.00160)  

**Abstract**: Many existing AI music generation tools rely on text prompts, complex interfaces, or instrument-like controls, which may require musical or technical knowledge that non-musicians do not possess. This paper introduces DeformTune, a prototype system that combines a tactile deformable interface with the MeasureVAE model to explore more intuitive, embodied, and explainable AI interaction. We conducted a preliminary study with 11 adult participants without formal musical training to investigate their experience with AI-assisted music creation. Thematic analysis of their feedback revealed recurring challenge--including unclear control mappings, limited expressive range, and the need for guidance throughout use. We discuss several design opportunities for enhancing explainability of AI, including multimodal feedback and progressive interaction support. These findings contribute early insights toward making AI music systems more explainable and empowering for novice users. 

---
# GEPAR3D: Geometry Prior-Assisted Learning for 3D Tooth Segmentation 

**Authors**: Tomasz Szczepański, Szymon Płotka, Michal K. Grzeszczyk, Arleta Adamowicz, Piotr Fudalej, Przemysław Korzeniowski, Tomasz Trzciński, Arkadiusz Sitek  

**Link**: [PDF](https://arxiv.org/pdf/2508.00155)  

**Abstract**: Tooth segmentation in Cone-Beam Computed Tomography (CBCT) remains challenging, especially for fine structures like root apices, which is critical for assessing root resorption in orthodontics. We introduce GEPAR3D, a novel approach that unifies instance detection and multi-class segmentation into a single step tailored to improve root segmentation. Our method integrates a Statistical Shape Model of dentition as a geometric prior, capturing anatomical context and morphological consistency without enforcing restrictive adjacency constraints. We leverage a deep watershed method, modeling each tooth as a continuous 3D energy basin encoding voxel distances to boundaries. This instance-aware representation ensures accurate segmentation of narrow, complex root apices. Trained on publicly available CBCT scans from a single center, our method is evaluated on external test sets from two in-house and two public medical centers. GEPAR3D achieves the highest overall segmentation performance, averaging a Dice Similarity Coefficient (DSC) of 95.0% (+2.8% over the second-best method) and increasing recall to 95.2% (+9.5%) across all test sets. Qualitative analyses demonstrated substantial improvements in root segmentation quality, indicating significant potential for more accurate root resorption assessment and enhanced clinical decision-making in orthodontics. We provide the implementation and dataset at this https URL. 

---
# INSPIRE-GNN: Intelligent Sensor Placement to Improve Sparse Bicycling Network Prediction via Reinforcement Learning Boosted Graph Neural Networks 

**Authors**: Mohit Gupta, Debjit Bhowmick, Rhys Newbury, Meead Saberi, Shirui Pan, Ben Beck  

**Link**: [PDF](https://arxiv.org/pdf/2508.00141)  

**Abstract**: Accurate link-level bicycling volume estimation is essential for sustainable urban transportation planning. However, many cities face significant challenges of high data sparsity due to limited bicycling count sensor coverage. To address this issue, we propose INSPIRE-GNN, a novel Reinforcement Learning (RL)-boosted hybrid Graph Neural Network (GNN) framework designed to optimize sensor placement and improve link-level bicycling volume estimation in data-sparse environments. INSPIRE-GNN integrates Graph Convolutional Networks (GCN) and Graph Attention Networks (GAT) with a Deep Q-Network (DQN)-based RL agent, enabling a data-driven strategic selection of sensor locations to maximize estimation performance. Applied to Melbourne's bicycling network, comprising 15,933 road segments with sensor coverage on only 141 road segments (99% sparsity) - INSPIRE-GNN demonstrates significant improvements in volume estimation by strategically selecting additional sensor locations in deployments of 50, 100, 200 and 500 sensors. Our framework outperforms traditional heuristic methods for sensor placement such as betweenness centrality, closeness centrality, observed bicycling activity and random placement, across key metrics such as Mean Squared Error (MSE), Root Mean Squared Error (RMSE) and Mean Absolute Error (MAE). Furthermore, our experiments benchmark INSPIRE-GNN against standard machine learning and deep learning models in the bicycle volume estimation performance, underscoring its effectiveness. Our proposed framework provides transport planners actionable insights to effectively expand sensor networks, optimize sensor placement and maximize volume estimation accuracy and reliability of bicycling data for informed transportation planning decisions. 

---
# Your Model Is Unfair, Are You Even Aware? Inverse Relationship Between Comprehension and Trust in Explainability Visualizations of Biased ML Models 

**Authors**: Zhanna Kaufman, Madeline Endres, Cindy Xiong Bearfield, Yuriy Brun  

**Link**: [PDF](https://arxiv.org/pdf/2508.00140)  

**Abstract**: Systems relying on ML have become ubiquitous, but so has biased behavior within them. Research shows that bias significantly affects stakeholders' trust in systems and how they use them. Further, stakeholders of different backgrounds view and trust the same systems differently. Thus, how ML models' behavior is explained plays a key role in comprehension and trust. We survey explainability visualizations, creating a taxonomy of design characteristics. We conduct user studies to evaluate five state-of-the-art visualization tools (LIME, SHAP, CP, Anchors, and ELI5) for model explainability, measuring how taxonomy characteristics affect comprehension, bias perception, and trust for non-expert ML users. Surprisingly, we find an inverse relationship between comprehension and trust: the better users understand the models, the less they trust them. We investigate the cause and find that this relationship is strongly mediated by bias perception: more comprehensible visualizations increase people's perception of bias, and increased bias perception reduces trust. We confirm this relationship is causal: Manipulating explainability visualizations to control comprehension, bias perception, and trust, we show that visualization design can significantly (p < 0.001) increase comprehension, increase perceived bias, and reduce trust. Conversely, reducing perceived model bias, either by improving model fairness or by adjusting visualization design, significantly increases trust even when comprehension remains high. Our work advances understanding of how comprehension affects trust and systematically investigates visualization's role in facilitating responsible ML applications. 

---
# Exploring the Feasibility of Deep Learning Techniques for Accurate Gender Classification from Eye Images 

**Authors**: Basna Mohammed Salih Hasan, Ramadhan J. Mstafa  

**Link**: [PDF](https://arxiv.org/pdf/2508.00135)  

**Abstract**: Gender classification has emerged as a crucial aspect in various fields, including security, human-machine interaction, surveillance, and advertising. Nonetheless, the accuracy of this classification can be influenced by factors such as cosmetics and disguise. Consequently, our study is dedicated to addressing this concern by concentrating on gender classification using color images of the periocular region. The periocular region refers to the area surrounding the eye, including the eyelids, eyebrows, and the region between them. It contains valuable visual cues that can be used to extract key features for gender classification. This paper introduces a sophisticated Convolutional Neural Network (CNN) model that utilizes color image databases to evaluate the effectiveness of the periocular region for gender classification. To validate the model's performance, we conducted tests on two eye datasets, namely CVBL and (Female and Male). The recommended architecture achieved an outstanding accuracy of 99% on the previously unused CVBL dataset while attaining a commendable accuracy of 96% with a small number of learnable parameters (7,235,089) on the (Female and Male) dataset. To ascertain the effectiveness of our proposed model for gender classification using the periocular region, we evaluated its performance through an extensive range of metrics and compared it with other state-of-the-art approaches. The results unequivocally demonstrate the efficacy of our model, thereby suggesting its potential for practical application in domains such as security and surveillance. 

---
# StackLiverNet: A Novel Stacked Ensemble Model for Accurate and Interpretable Liver Disease Detection 

**Authors**: Md. Ehsanul Haque, S. M. Jahidul Islam, Shakil Mia, Rumana Sharmin, Ashikuzzaman, Md Samir Morshed, Md. Tahmidul Huque  

**Link**: [PDF](https://arxiv.org/pdf/2508.00117)  

**Abstract**: Liver diseases are a serious health concern in the world, which requires precise and timely diagnosis to enhance the survival chances of patients. The current literature implemented numerous machine learning and deep learning models to classify liver diseases, but most of them had some issues like high misclassification error, poor interpretability, prohibitive computational expense, and lack of good preprocessing strategies. In order to address these drawbacks, we introduced StackLiverNet in this study; an interpretable stacked ensemble model tailored to the liver disease detection task. The framework uses advanced data preprocessing and feature selection technique to increase model robustness and predictive ability. Random undersampling is performed to deal with class imbalance and make the training balanced. StackLiverNet is an ensemble of several hyperparameter-optimized base classifiers, whose complementary advantages are used through a LightGBM meta-model. The provided model demonstrates excellent performance, with the testing accuracy of 99.89%, Cohen Kappa of 0.9974, and AUC of 0.9993, having only 5 misclassifications, and efficient training and inference speeds that are amenable to clinical practice (training time 4.2783 seconds, inference time 0.1106 seconds). Besides, Local Interpretable Model-Agnostic Explanations (LIME) are applied to generate transparent explanations of individual predictions, revealing high concentrations of Alkaline Phosphatase and moderate SGOT as important observations of liver disease. Also, SHAP was used to rank features by their global contribution to predictions, while the Morris method confirmed the most influential features through sensitivity analysis. 

---
# FACTORY: A Challenging Human-Verified Prompt Set for Long-Form Factuality 

**Authors**: Mingda Chen, Yang Li, Xilun Chen, Adina Williams, Gargi Ghosh, Scott Yih  

**Link**: [PDF](https://arxiv.org/pdf/2508.00109)  

**Abstract**: Long-form factuality evaluation assesses the ability of models to generate accurate, comprehensive responses to short prompts. Existing benchmarks often lack human verification, leading to potential quality issues. To address this limitation, we introduce FACTORY, a large-scale, human-verified prompt set. Developed using a model-in-the-loop approach and refined by humans, FACTORY includes challenging prompts that are fact-seeking, answerable, and unambiguous. We conduct human evaluations on 6 state-of-the-art language models using FACTORY and existing datasets. Our results show that FACTORY is a challenging benchmark: approximately 40% of the claims made in the responses of SOTA models are not factual, compared to only 10% for other datasets. Our analysis identifies the strengths of FACTORY over prior benchmarks, emphasizing its reliability and the necessity for models to reason across long-tailed facts. 

---
# A Mixed User-Centered Approach to Enable Augmented Intelligence in Intelligent Tutoring Systems: The Case of MathAIde app 

**Authors**: Guilherme Guerino, Luiz Rodrigues, Luana Bianchiniand Mariana Alves, Marcelo Marinho, Thomaz Veloso, Valmir Macario, Diego Dermeval, Thales Vieira, Ig Bittencourt, Seiji Isotani  

**Link**: [PDF](https://arxiv.org/pdf/2508.00103)  

**Abstract**: Integrating Artificial Intelligence in Education (AIED) aims to enhance learning experiences through technologies like Intelligent Tutoring Systems (ITS), offering personalized learning, increased engagement, and improved retention rates. However, AIED faces three main challenges: the critical role of teachers in the design process, the limitations and reliability of AI tools, and the accessibility of technological resources. Augmented Intelligence (AuI) addresses these challenges by enhancing human capabilities rather than replacing them, allowing systems to suggest solutions. In contrast, humans provide final assessments, thus improving AI over time. In this sense, this study focuses on designing, developing, and evaluating MathAIde, an ITS that corrects mathematics exercises using computer vision and AI and provides feedback based on photos of student work. The methodology included brainstorming sessions with potential users, high-fidelity prototyping, A/B testing, and a case study involving real-world classroom environments for teachers and students. Our research identified several design possibilities for implementing AuI in ITSs, emphasizing a balance between user needs and technological feasibility. Prioritization and validation through prototyping and testing highlighted the importance of efficiency metrics, ultimately leading to a solution that offers pre-defined remediation alternatives for teachers. Real-world deployment demonstrated the usefulness of the proposed solution. Our research contributes to the literature by providing a usable, teacher-centered design approach that involves teachers in all design phases. As a practical implication, we highlight that the user-centered design approach increases the usefulness and adoption potential of AIED systems, especially in resource-limited environments. 

---
# Stress-Aware Resilient Neural Training 

**Authors**: Ashkan Shakarami, Yousef Yeganeh, Azade Farshad, Lorenzo Nicole, Stefano Ghidoni, Nassir Navab  

**Link**: [PDF](https://arxiv.org/pdf/2508.00098)  

**Abstract**: This paper introduces Stress-Aware Learning, a resilient neural training paradigm in which deep neural networks dynamically adjust their optimization behavior - whether under stable training regimes or in settings with uncertain dynamics - based on the concept of Temporary (Elastic) and Permanent (Plastic) Deformation, inspired by structural fatigue in materials science. To instantiate this concept, we propose Plastic Deformation Optimizer, a stress-aware mechanism that injects adaptive noise into model parameters whenever an internal stress signal - reflecting stagnation in training loss and accuracy - indicates persistent optimization difficulty. This enables the model to escape sharp minima and converge toward flatter, more generalizable regions of the loss landscape. Experiments across six architectures, four optimizers, and seven vision benchmarks demonstrate improved robustness and generalization with minimal computational overhead. The code and 3D visuals will be available on GitHub: this https URL. 

---
# XRoboToolkit: A Cross-Platform Framework for Robot Teleoperation 

**Authors**: Zhigen Zhao, Liuchuan Yu, Ke Jing, Ning Yang  

**Link**: [PDF](https://arxiv.org/pdf/2508.00097)  

**Abstract**: The rapid advancement of Vision-Language-Action models has created an urgent need for large-scale, high-quality robot demonstration datasets. Although teleoperation is the predominant method for data collection, current approaches suffer from limited scalability, complex setup procedures, and suboptimal data quality. This paper presents XRoboToolkit, a cross-platform framework for extended reality based robot teleoperation built on the OpenXR standard. The system features low-latency stereoscopic visual feedback, optimization-based inverse kinematics, and support for diverse tracking modalities including head, controller, hand, and auxiliary motion trackers. XRoboToolkit's modular architecture enables seamless integration across robotic platforms and simulation environments, spanning precision manipulators, mobile robots, and dexterous hands. We demonstrate the framework's effectiveness through precision manipulation tasks and validate data quality by training VLA models that exhibit robust autonomous performance. 

---
# Punching Bag vs. Punching Person: Motion Transferability in Videos 

**Authors**: Raiyaan Abdullah, Jared Claypoole, Michael Cogswell, Ajay Divakaran, Yogesh Rawat  

**Link**: [PDF](https://arxiv.org/pdf/2508.00085)  

**Abstract**: Action recognition models demonstrate strong generalization, but can they effectively transfer high-level motion concepts across diverse contexts, even within similar distributions? For example, can a model recognize the broad action "punching" when presented with an unseen variation such as "punching person"? To explore this, we introduce a motion transferability framework with three datasets: (1) Syn-TA, a synthetic dataset with 3D object motions; (2) Kinetics400-TA; and (3) Something-Something-v2-TA, both adapted from natural video datasets. We evaluate 13 state-of-the-art models on these benchmarks and observe a significant drop in performance when recognizing high-level actions in novel contexts. Our analysis reveals: 1) Multimodal models struggle more with fine-grained unknown actions than with coarse ones; 2) The bias-free Syn-TA proves as challenging as real-world datasets, with models showing greater performance drops in controlled settings; 3) Larger models improve transferability when spatial cues dominate but struggle with intensive temporal reasoning, while reliance on object and background cues hinders generalization. We further explore how disentangling coarse and fine motions can improve recognition in temporally challenging datasets. We believe this study establishes a crucial benchmark for assessing motion transferability in action recognition. Datasets and relevant code: this https URL. 

---
# A Survey on Code Generation with LLM-based Agents 

**Authors**: Yihong Dong, Xue Jiang, Jiaru Qian, Tian Wang, Kechi Zhang, Zhi Jin, Ge Li  

**Link**: [PDF](https://arxiv.org/pdf/2508.00083)  

**Abstract**: Code generation agents powered by large language models (LLMs) are revolutionizing the software development paradigm. Distinct from previous code generation techniques, code generation agents are characterized by three core features. 1) Autonomy: the ability to independently manage the entire workflow, from task decomposition to coding and debugging. 2) Expanded task scope: capabilities that extend beyond generating code snippets to encompass the full software development lifecycle (SDLC). 3) Enhancement of engineering practicality: a shift in research emphasis from algorithmic innovation toward practical engineering challenges, such as system reliability, process management, and tool integration. This domain has recently witnessed rapid development and an explosion in research, demonstrating significant application potential. This paper presents a systematic survey of the field of LLM-based code generation agents. We trace the technology's developmental trajectory from its inception and systematically categorize its core techniques, including both single-agent and multi-agent architectures. Furthermore, this survey details the applications of LLM-based agents across the full SDLC, summarizes mainstream evaluation benchmarks and metrics, and catalogs representative tools. Finally, by analyzing the primary challenges, we identify and propose several foundational, long-term research directions for the future work of the field. 

---
# PhysicsEval: Inference-Time Techniques to Improve the Reasoning Proficiency of Large Language Models on Physics Problems 

**Authors**: Oshayer Siddique, J. M Areeb Uzair Alam, Md Jobayer Rahman Rafy, Syed Rifat Raiyan, Hasan Mahmud, Md Kamrul Hasan  

**Link**: [PDF](https://arxiv.org/pdf/2508.00079)  

**Abstract**: The discipline of physics stands as a cornerstone of human intellect, driving the evolution of technology and deepening our understanding of the fundamental principles of the cosmos. Contemporary literature includes some works centered on the task of solving physics problems - a crucial domain of natural language reasoning. In this paper, we evaluate the performance of frontier LLMs in solving physics problems, both mathematical and descriptive. We also employ a plethora of inference-time techniques and agentic frameworks to improve the performance of the models. This includes the verification of proposed solutions in a cumulative fashion by other, smaller LLM agents, and we perform a comparative analysis of the performance that the techniques entail. There are significant improvements when the multi-agent framework is applied to problems that the models initially perform poorly on. Furthermore, we introduce a new evaluation benchmark for physics problems, ${\rm P{\small HYSICS}E{\small VAL}}$, consisting of 19,609 problems sourced from various physics textbooks and their corresponding correct solutions scraped from physics forums and educational websites. Our code and data are publicly available at this https URL. 

---
# Evaluating COVID 19 Feature Contributions to Bitcoin Return Forecasting: Methodology Based on LightGBM and Genetic Optimization 

**Authors**: Imen Mahmoud, Andrei Velichko  

**Link**: [PDF](https://arxiv.org/pdf/2508.00078)  

**Abstract**: This study proposes a novel methodological framework integrating a LightGBM regression model and genetic algorithm (GA) optimization to systematically evaluate the contribution of COVID-19-related indicators to Bitcoin return prediction. The primary objective was not merely to forecast Bitcoin returns but rather to determine whether including pandemic-related health data significantly enhances prediction accuracy. A comprehensive dataset comprising daily Bitcoin returns and COVID-19 metrics (vaccination rates, hospitalizations, testing statistics) was constructed. Predictive models, trained with and without COVID-19 features, were optimized using GA over 31 independent runs, allowing robust statistical assessment. Performance metrics (R2, RMSE, MAE) were statistically compared through distribution overlaps and Mann-Whitney U tests. Permutation Feature Importance (PFI) analysis quantified individual feature contributions. Results indicate that COVID-19 indicators significantly improved model performance, particularly in capturing extreme market fluctuations (R2 increased by 40%, RMSE decreased by 2%, both highly significant statistically). Among COVID-19 features, vaccination metrics, especially the 75th percentile of fully vaccinated individuals, emerged as dominant predictors. The proposed methodology extends existing financial analytics tools by incorporating public health signals, providing investors and policymakers with refined indicators to navigate market uncertainty during systemic crises. 

---
# TriP-LLM: A Tri-Branch Patch-wise Large Language Model Framework for Time-Series Anomaly Detection 

**Authors**: Yuan-Cheng Yu, Yen-Chieh Ouyang, Chun-An Lin  

**Link**: [PDF](https://arxiv.org/pdf/2508.00047)  

**Abstract**: Time-series anomaly detection plays a central role across a wide range of application domains. With the increasing proliferation of the Internet of Things (IoT) and smart manufacturing, time-series data has dramatically increased in both scale and dimensionality. This growth has exposed the limitations of traditional statistical methods in handling the high heterogeneity and complexity of such data. Inspired by the recent success of large language models (LLMs) in multimodal tasks across language and vision domains, we propose a novel unsupervised anomaly detection framework: A Tri-Branch Patch-wise Large Language Model Framework for Time-Series Anomaly Detection (TriP-LLM). TriP-LLM integrates local and global temporal features through a tri-branch design-Patching, Selection, and Global-to encode the input time series into patch-wise tokens, which are then processed by a frozen, pretrained LLM. A lightweight patch-wise decoder reconstructs the input, from which anomaly scores are derived. We evaluate TriP-LLM on several public benchmark datasets using PATE, a recently proposed threshold-free evaluation metric, and conduct all comparisons within a unified open-source framework to ensure fairness. Experimental results show that TriP-LLM consistently outperforms recent state-of-the-art methods across all datasets, demonstrating strong detection capabilities. Furthermore, through extensive ablation studies, we verify the substantial contribution of the LLM to the overall architecture. Compared to LLM-based approaches using Channel Independence (CI) patch processing, TriP-LLM achieves significantly lower memory consumption, making it more suitable for GPU memory-constrained environments. All code and model checkpoints are publicly available on this https URL 

---
# Benchmarking Partial Observability in Reinforcement Learning with a Suite of Memory-Improvable Domains 

**Authors**: Ruo Yu Tao, Kaicheng Guo, Cameron Allen, George Konidaris  

**Link**: [PDF](https://arxiv.org/pdf/2508.00046)  

**Abstract**: Mitigating partial observability is a necessary but challenging task for general reinforcement learning algorithms. To improve an algorithm's ability to mitigate partial observability, researchers need comprehensive benchmarks to gauge progress. Most algorithms tackling partial observability are only evaluated on benchmarks with simple forms of state aliasing, such as feature masking and Gaussian noise. Such benchmarks do not represent the many forms of partial observability seen in real domains, like visual occlusion or unknown opponent intent. We argue that a partially observable benchmark should have two key properties. The first is coverage in its forms of partial observability, to ensure an algorithm's generalizability. The second is a large gap between the performance of a agents with more or less state information, all other factors roughly equal. This gap implies that an environment is memory improvable: where performance gains in a domain are from an algorithm's ability to cope with partial observability as opposed to other factors. We introduce best-practice guidelines for empirically benchmarking reinforcement learning under partial observability, as well as the open-source library POBAX: Partially Observable Benchmarks in JAX. We characterize the types of partial observability present in various environments and select representative environments for our benchmark. These environments include localization and mapping, visual control, games, and more. Additionally, we show that these tasks are all memory improvable and require hard-to-learn memory functions, providing a concrete signal for partial observability research. This framework includes recommended hyperparameters as well as algorithm implementations for fast, out-of-the-box evaluation, as well as highly performant environments implemented in JAX for GPU-scalable experimentation. 

---
# Learning Like Humans: Resource-Efficient Federated Fine-Tuning through Cognitive Developmental Stages 

**Authors**: Yebo Wu, Jingguang Li, Zhijiang Guo, Li Li  

**Link**: [PDF](https://arxiv.org/pdf/2508.00041)  

**Abstract**: Federated fine-tuning enables Large Language Models (LLMs) to adapt to downstream tasks while preserving data privacy, but its resource-intensive nature limits deployment on edge devices. In this paper, we introduce Developmental Federated Tuning (DevFT), a resource-efficient approach inspired by cognitive development that progressively builds a powerful LLM from a compact foundation. DevFT decomposes the fine-tuning process into developmental stages, each optimizing submodels with increasing parameter capacity. Knowledge from earlier stages transfers to subsequent submodels, providing optimized initialization parameters that prevent convergence to local minima and accelerate training. This paradigm mirrors human learning, gradually constructing comprehensive knowledge structure while refining existing skills. To efficiently build stage-specific submodels, DevFT introduces deconfliction-guided layer grouping and differential-based layer fusion to distill essential information and construct representative layers. Evaluations across multiple benchmarks demonstrate that DevFT significantly outperforms state-of-the-art methods, achieving up to 4.59$\times$ faster convergence, 10.67$\times$ reduction in communication overhead, and 9.07% average performance improvement, while maintaining compatibility with existing approaches. 

---
# Hybrid LSTM-Transformer Models for Profiling Highway-Railway Grade Crossings 

**Authors**: Kaustav Chatterjee, Joshua Q. Li, Fatemeh Ansari, Masud Rana Munna, Kundan Parajulee, Jared Schwennesen  

**Link**: [PDF](https://arxiv.org/pdf/2508.00039)  

**Abstract**: Hump crossings, or high-profile Highway Railway Grade Crossings (HRGCs), pose safety risks to highway vehicles due to potential hang-ups. These crossings typically result from post-construction railway track maintenance activities or non-compliance with design guidelines for HRGC vertical alignments. Conventional methods for measuring HRGC profiles are costly, time-consuming, traffic-disruptive, and present safety challenges. To address these issues, this research employed advanced, cost-effective techniques and innovative modeling approaches for HRGC profile measurement. A novel hybrid deep learning framework combining Long Short-Term Memory (LSTM) and Transformer architectures was developed by utilizing instrumentation and ground truth data. Instrumentation data were gathered using a highway testing vehicle equipped with Inertial Measurement Unit (IMU) and Global Positioning System (GPS) sensors, while ground truth data were obtained via an industrial-standard walking profiler. Field data was collected at the Red Rock Railroad Corridor in Oklahoma. Three advanced deep learning models Transformer-LSTM sequential (model 1), LSTM-Transformer sequential (model 2), and LSTM-Transformer parallel (model 3) were evaluated to identify the most efficient architecture. Models 2 and 3 outperformed the others and were deployed to generate 2D/3D HRGC profiles. The deep learning models demonstrated significant potential to enhance highway and railroad safety by enabling rapid and accurate assessment of HRGC hang-up susceptibility. 

---
# Predicting Large-scale Urban Network Dynamics with Energy-informed Graph Neural Diffusion 

**Authors**: Tong Nie, Jian Sun, Wei Ma  

**Link**: [PDF](https://arxiv.org/pdf/2508.00037)  

**Abstract**: Networked urban systems facilitate the flow of people, resources, and services, and are essential for economic and social interactions. These systems often involve complex processes with unknown governing rules, observed by sensor-based time series. To aid decision-making in industrial and engineering contexts, data-driven predictive models are used to forecast spatiotemporal dynamics of urban systems. Current models such as graph neural networks have shown promise but face a trade-off between efficacy and efficiency due to computational demands. Hence, their applications in large-scale networks still require further efforts. This paper addresses this trade-off challenge by drawing inspiration from physical laws to inform essential model designs that align with fundamental principles and avoid architectural redundancy. By understanding both micro- and macro-processes, we present a principled interpretable neural diffusion scheme based on Transformer-like structures whose attention layers are induced by low-dimensional embeddings. The proposed scalable spatiotemporal Transformer (ScaleSTF), with linear complexity, is validated on large-scale urban systems including traffic flow, solar power, and smart meters, showing state-of-the-art performance and remarkable scalability. Our results constitute a fresh perspective on the dynamics prediction in large-scale urban networks. 

---
# GPT-4.1 Sets the Standard in Automated Experiment Design Using Novel Python Libraries 

**Authors**: Nuno Fachada, Daniel Fernandes, Carlos M. Fernandes, Bruno D. Ferreira-Saraiva, João P. Matos-Carvalho  

**Link**: [PDF](https://arxiv.org/pdf/2508.00033)  

**Abstract**: Large Language Models (LLMs) have advanced rapidly as tools for automating code generation in scientific research, yet their ability to interpret and use unfamiliar Python APIs for complex computational experiments remains poorly characterized. This study systematically benchmarks a selection of state-of-the-art LLMs in generating functional Python code for two increasingly challenging scenarios: conversational data analysis with the \textit{ParShift} library, and synthetic data generation and clustering using \textit{pyclugen} and \textit{scikit-learn}. Both experiments use structured, zero-shot prompts specifying detailed requirements but omitting in-context examples. Model outputs are evaluated quantitatively for functional correctness and prompt compliance over multiple runs, and qualitatively by analyzing the errors produced when code execution fails. Results show that only a small subset of models consistently generate correct, executable code, with GPT-4.1 standing out as the only model to always succeed in both tasks. In addition to benchmarking LLM performance, this approach helps identify shortcomings in third-party libraries, such as unclear documentation or obscure implementation bugs. Overall, these findings highlight current limitations of LLMs for end-to-end scientific automation and emphasize the need for careful prompt design, comprehensive library documentation, and continued advances in language model capabilities. 

---
# Scalable Spectrum Availability Prediction using a Markov Chain Framework and ITU-R Propagation Models 

**Authors**: Abir Ray  

**Link**: [PDF](https://arxiv.org/pdf/2508.00028)  

**Abstract**: Spectrum resources are often underutilized across time and space, motivating dynamic spectrum access strategies that allow secondary users to exploit unused frequencies. A key challenge is predicting when and where spectrum will be available (i.e., unused by primary licensed users) in order to enable proactive and interference-free access. This paper proposes a scalable framework for spectrum availability prediction that combines a two-state Markov chain model of primary user activity with high-fidelity propagation models from the ITU-R (specifically Recommendations P.528 and P.2108). The Markov chain captures temporal occupancy patterns, while the propagation models incorporate path loss and clutter effects to determine if primary signals exceed interference thresholds at secondary user locations. By integrating these components, the proposed method can predict spectrum opportunities both in time and space with improved accuracy. We develop the system model and algorithm for the approach, analyze its scalability and computational efficiency, and discuss assumptions, limitations, and potential applications. The framework is flexible and can be adapted to various frequency bands and scenarios. The results and analysis show that the proposed approach can effectively identify available spectrum with low computational cost, making it suitable for real-time spectrum management in cognitive radio networks and other dynamic spectrum sharing systems. 

---
# Embedding-Aware Quantum-Classical SVMs for Scalable Quantum Machine Learning 

**Authors**: Sebastián Andrés Cajas Ordóñez, Luis Fernando Torres Torres, Mario Bifulco, Carlos Andrés Durán, Cristian Bosch, Ricardo Simón Carbajo  

**Link**: [PDF](https://arxiv.org/pdf/2508.00024)  

**Abstract**: Quantum Support Vector Machines face scalability challenges due to high-dimensional quantum states and hardware limitations. We propose an embedding-aware quantum-classical pipeline combining class-balanced k-means distillation with pretrained Vision Transformer embeddings. Our key finding: ViT embeddings uniquely enable quantum advantage, achieving up to 8.02% accuracy improvements over classical SVMs on Fashion-MNIST and 4.42% on MNIST, while CNN features show performance degradation. Using 16-qubit tensor network simulation via cuTensorNet, we provide the first systematic evidence that quantum kernel advantage depends critically on embedding choice, revealing fundamental synergy between transformer attention and quantum feature spaces. This provides a practical pathway for scalable quantum machine learning that leverages modern neural architectures. 

---
# Generative Logic: A New Computer Architecture for Deterministic Reasoning and Knowledge Generation 

**Authors**: Nikolai Sergeev  

**Link**: [PDF](https://arxiv.org/pdf/2508.00017)  

**Abstract**: We present Generative Logic (GL), a deterministic architecture that begins from user-supplied axiomatic definitions -- written in a minimalist Mathematical Programming Language (MPL) -- and systematically explores their deductive neighborhood. Definitions are compiled into a distributed grid of simple Logic Blocks (LBs) that exchange messages; any time several expressions unify under an inference rule, a new fact is emitted with full provenance to its sources, yielding replayable, auditable proof graphs.
A prototype software implementation instantiates the workflow on first-order Peano arithmetic. Starting only from the Peano axioms, GL enumerates candidate implications, applies normalization and type filters, and automatically reconstructs machine-checkable proofs of foundational arithmetic laws including associativity and commutativity of addition, associativity and commutativity of multiplication, and distributivity. Generated proofs export to navigable HTML so that every inference step can be inspected independently.
We outline a hardware-software co-design path toward massively parallel realizations and describe prospective integration with probabilistic models (e.g., Large Language Models (LLMs)) for autoformalization and conjecture seeding. The Python and MPL code to reproduce the Peano experiments, along with the full HTML proof graphs, are available in the project's GitHub repository at this https URL and are permanently archived at this https URL. We invite community feedback and collaboration. 

---
# AoI-Aware Resource Allocation with Deep Reinforcement Learning for HAPS-V2X Networks 

**Authors**: Ahmet Melih Ince, Ayse Elif Canbilen, Halim Yanikomeroglu  

**Link**: [PDF](https://arxiv.org/pdf/2508.00011)  

**Abstract**: Sixth-generation (6G) networks are designed to meet the hyper-reliable and low-latency communication (HRLLC) requirements of safety-critical applications such as autonomous driving. Integrating non-terrestrial networks (NTN) into the 6G infrastructure brings redundancy to the network, ensuring continuity of communications even under extreme conditions. In particular, high-altitude platform stations (HAPS) stand out for their wide coverage and low latency advantages, supporting communication reliability and enhancing information freshness, especially in rural areas and regions with infrastructure constraints. In this paper, we present reinforcement learning-based approaches using deep deterministic policy gradient (DDPG) to dynamically optimize the age-of-information (AoI) in HAPS-enabled vehicle-to-everything (V2X) networks. The proposed method improves information freshness and overall network reliability by enabling independent learning without centralized coordination. The findings reveal the potential of HAPS-supported solutions, combined with DDPG-based learning, for efficient AoI-aware resource allocation in platoon-based autonomous vehicle systems. 

---
# Enabling Immersive XR Collaborations over FTTR Networks (Invited) 

**Authors**: Sourav Mondal, Elaine Wong  

**Link**: [PDF](https://arxiv.org/pdf/2508.00009)  

**Abstract**: Fiber-To-The-Room is a potential solution to achieve in-premise extended reality collaborations. This paper explores predictive bandwidth allocation and seamless handover schemes over FTTR, showing high-quality immersive experience for in-premise collaborations can be achieved. \c{opyright} 2025 The Author(s). 

---
# Agent Network Protocol Technical White Paper 

**Authors**: Gaowei Chang, Eidan Lin, Chengxuan Yuan, Rizhao Cai, Binbin Chen, Xuan Xie, Yin Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2508.00007)  

**Abstract**: With the development of large models and autonomous decision-making AI, agents are rapidly becoming the new entities of the internet, following mobile apps. However, existing internet infrastructure is primarily designed for human interaction, creating data silos, unfriendly interfaces, and high collaboration costs among agents, making it difficult to support the needs for large-scale agent interconnection and collaboration. The internet is undergoing a profound transformation, showing four core trends: agents replacing traditional software, universal agent interconnection, native protocol-based connections, and autonomous agent organization and collaboration. To align with these trends, Agent Network Protocol (ANP) proposes a new generation of communication protocols for the Agentic Web. ANP adheres to AI-native design, maintains compatibility with existing internet protocols, adopts a modular composable architecture, follows minimalist yet extensible principles, and enables rapid deployment based on existing infrastructure. Through a three-layer protocol system--identity and encrypted communication layer, meta-protocol negotiation layer, and application protocol layer--ANP. systematically solves the problems of agent identity authentication, dynamic negotiation, and capability discovery interoperability. 

---
# Modelling Program Spaces in Program Synthesis with Constraints 

**Authors**: Tilman Hinnerichs, Bart Swinkels, Jaap de Jong, Reuben Gardos Reid, Tudor Magirescu, Neil Yorke-Smith, Sebastijan Dumancic  

**Link**: [PDF](https://arxiv.org/pdf/2508.00005)  

**Abstract**: A core challenge in program synthesis is taming the large space of possible programs. Since program synthesis is essentially a combinatorial search, the community has sought to leverage powerful combinatorial constraint solvers. Here, constraints are used to express the program semantics, but not as a potentially potent tool to remove unwanted programs. Recent inductive logic programming approaches introduce constraints on the program's syntax to be synthesized. These syntactic constraints allow for checking and propagating a constraint without executing the program, and thus for arbitrary operators. In this work, we leverage syntactic constraints to model program spaces, defining not just solutions that are feasible, but also ones that are likely useful. To demonstrate this idea, we introduce BART, a solver that efficiently propagates and solves these constraints. We evaluate BART on program space enumeration tasks, finding that the constraints eliminate up to 99 percent of the program space, and that modeling program spaces significantly reduces enumeration time. 

---
# Agency Among Agents: Designing with Hypertextual Friction in the Algorithmic Web 

**Authors**: Sophia Liu, Shm Garanganao Almeda  

**Link**: [PDF](https://arxiv.org/pdf/2507.23585)  

**Abstract**: Today's algorithm-driven interfaces, from recommendation feeds to GenAI tools, often prioritize engagement and efficiency at the expense of user agency. As systems take on more decision-making, users have less control over what they see and how meaning or relationships between content are constructed. This paper introduces "Hypertextual Friction," a conceptual design stance that repositions classical hypertext principles--friction, traceability, and structure--as actionable values for reclaiming agency in algorithmically mediated environments. Through a comparative analysis of real-world interfaces--Wikipedia vs. Instagram Explore, and this http URL vs. GenAI image tools--we examine how different systems structure user experience, navigation, and authorship. We show that hypertext systems emphasize provenance, associative thinking, and user-driven meaning-making, while algorithmic systems tend to obscure process and flatten participation. We contribute: (1) a comparative analysis of how interface structures shape agency in user-driven versus agent-driven systems, and (2) a conceptual stance that offers hypertextual values as design commitments for reclaiming agency in an increasingly algorithmic web. 

---
# NusaAksara: A Multimodal and Multilingual Benchmark for Preserving Indonesian Indigenous Scripts 

**Authors**: Muhammad Farid Adilazuarda, Musa Izzanardi Wijanarko, Lucky Susanto, Khumaisa Nur'aini, Derry Wijaya, Alham Fikri Aji  

**Link**: [PDF](https://arxiv.org/pdf/2502.18148)  

**Abstract**: Indonesia is rich in languages and scripts. However, most NLP progress has been made using romanized text. In this paper, we present NusaAksara, a novel public benchmark for Indonesian languages that includes their original scripts. Our benchmark covers both text and image modalities and encompasses diverse tasks such as image segmentation, OCR, transliteration, translation, and language identification. Our data is constructed by human experts through rigorous steps. NusaAksara covers 8 scripts across 7 languages, including low-resource languages not commonly seen in NLP benchmarks. Although unsupported by Unicode, the Lampung script is included in this dataset. We benchmark our data across several models, from LLMs and VLMs such as GPT-4o, Llama 3.2, and Aya 23 to task-specific systems such as PP-OCR and LangID, and show that most NLP technologies cannot handle Indonesia's local scripts, with many achieving near-zero performance. 

---
