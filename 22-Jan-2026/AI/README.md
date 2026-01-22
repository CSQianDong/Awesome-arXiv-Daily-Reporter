# BayesianVLA: Bayesian Decomposition of Vision Language Action Models via Latent Action Queries 

**Authors**: Shijie Lian, Bin Yu, Xiaopeng Lin, Laurence T. Yang, Zhaolong Shen, Changti Wu, Yuzhuo Miao, Cong Huang, Kai Chen  

**Link**: [PDF](https://arxiv.org/pdf/2601.15197)  

**Abstract**: Vision-Language-Action (VLA) models have shown promise in robot manipulation but often struggle to generalize to new instructions or complex multi-task scenarios. We identify a critical pathology in current training paradigms where goal-driven data collection creates a dataset bias. In such datasets, language instructions are highly predictable from visual observations alone, causing the conditional mutual information between instructions and actions to vanish, a phenomenon we term Information Collapse. Consequently, models degenerate into vision-only policies that ignore language constraints and fail in out-of-distribution (OOD) settings. To address this, we propose BayesianVLA, a novel framework that enforces instruction following via Bayesian decomposition. By introducing learnable Latent Action Queries, we construct a dual-branch architecture to estimate both a vision-only prior $p(a \mid v)$ and a language-conditioned posterior $\pi(a \mid v, \ell)$. We then optimize the policy to maximize the conditional Pointwise Mutual Information (PMI) between actions and instructions. This objective effectively penalizes the vision shortcut and rewards actions that explicitly explain the language command. Without requiring new data, BayesianVLA significantly improves generalization. Extensive experiments across on SimplerEnv and RoboCasa demonstrate substantial gains, including an 11.3% improvement on the challenging OOD SimplerEnv benchmark, validating the ability of our approach to robustly ground language in action. 

---
# Knowledge Graphs are Implicit Reward Models: Path-Derived Signals Enable Compositional Reasoning 

**Authors**: Yuval Kansal, Niraj K. Jha  

**Link**: [PDF](https://arxiv.org/pdf/2601.15160)  

**Abstract**: Large language models have achieved near-expert performance in structured reasoning domains like mathematics and programming, yet their ability to perform compositional multi-hop reasoning in specialized scientific fields remains limited. We propose a bottom-up learning paradigm in which models are grounded in axiomatic domain facts and compose them to solve complex, unseen tasks. To this end, we present a post-training pipeline, based on a combination of supervised fine-tuning and reinforcement learning (RL), in which knowledge graphs act as implicit reward models. By deriving novel reward signals from knowledge graph paths, we provide verifiable, scalable, and grounded supervision that encourages models to compose intermediate axioms rather than optimize only final answers during RL. We validate this approach in the medical domain, training a 14B model on short-hop reasoning paths (1-3 hops) and evaluating its zero-shot generalization to complex multi-hop queries (4-5 hops). Our experiments show that path-derived rewards act as a "compositional bridge", enabling our model to significantly outperform much larger models and frontier systems like GPT-5.2 and Gemini 3 Pro, on the most difficult reasoning tasks. Furthermore, we demonstrate the robustness of our approach to adversarial perturbations against option-shuffling stress tests. This work suggests that grounding the reasoning process in structured knowledge is a scalable and efficient path toward intelligent reasoning. 

---
# How to Build AI Agents by Augmenting LLMs with Codified Human Expert Domain Knowledge? A Software Engineering Framework 

**Authors**: Choro Ulan uulu, Mikhail Kulyabin, Iris Fuhrmann, Jan Joosten, Nuno Miguel Martins Pacheco, Filippos Petridis, Rebecca Johnson, Jan Bosch, Helena Holmström Olsson  

**Link**: [PDF](https://arxiv.org/pdf/2601.15153)  

**Abstract**: Critical domain knowledge typically resides with few experts, creating organizational bottlenecks in scalability and decision-making. Non-experts struggle to create effective visualizations, leading to suboptimal insights and diverting expert time. This paper investigates how to capture and embed human domain knowledge into AI agent systems through an industrial case study. We propose a software engineering framework to capture human domain knowledge for engineering AI agents in simulation data visualization by augmenting a Large Language Model (LLM) with a request classifier, Retrieval-Augmented Generation (RAG) system for code generation, codified expert rules, and visualization design principles unified in an agent demonstrating autonomous, reactive, proactive, and social behavior. Evaluation across five scenarios spanning multiple engineering domains with 12 evaluators demonstrates 206% improvement in output quality, with our agent achieving expert-level ratings in all cases versus baseline's poor performance, while maintaining superior code quality with lower variance. Our contributions are: an automated agent-based system for visualization generation and a validated framework for systematically capturing human domain knowledge and codifying tacit expert knowledge into AI agents, demonstrating that non-experts can achieve expert-level outcomes in specialized domains. 

---
# Vehicle Routing with Finite Time Horizon using Deep Reinforcement Learning with Improved Network Embedding 

**Authors**: Ayan Maity, Sudeshna Sarkar  

**Link**: [PDF](https://arxiv.org/pdf/2601.15131)  

**Abstract**: In this paper, we study the vehicle routing problem with a finite time horizon. In this routing problem, the objective is to maximize the number of customer requests served within a finite time horizon. We present a novel routing network embedding module which creates local node embedding vectors and a context-aware global graph representation. The proposed Markov decision process for the vehicle routing problem incorporates the node features, the network adjacency matrix and the edge features as components of the state space. We incorporate the remaining finite time horizon into the network embedding module to provide a proper routing context to the embedding module. We integrate our embedding module with a policy gradient-based deep Reinforcement Learning framework to solve the vehicle routing problem with finite time horizon. We trained and validated our proposed routing method on real-world routing networks, as well as synthetically generated Euclidean networks. Our experimental results show that our method achieves a higher customer service rate than the existing routing methods. Additionally, the solution time of our method is significantly lower than that of the existing methods. 

---
# The Plausibility Trap: Using Probabilistic Engines for Deterministic Tasks 

**Authors**: Ivan Carrera, Daniel Maldonado-Ruiz  

**Link**: [PDF](https://arxiv.org/pdf/2601.15130)  

**Abstract**: The ubiquity of Large Language Models (LLMs) is driving a paradigm shift where user convenience supersedes computational efficiency. This article defines the "Plausibility Trap": a phenomenon where individuals with access to Artificial Intelligence (AI) models deploy expensive probabilistic engines for simple deterministic tasks-such as Optical Character Recognition (OCR) or basic verification-resulting in significant resource waste. Through micro-benchmarks and case studies on OCR and fact-checking, we quantify the "efficiency tax"-demonstrating a ~6.5x latency penalty-and the risks of algorithmic sycophancy. To counter this, we introduce Tool Selection Engineering and the Deterministic-Probabilistic Decision Matrix, a framework to help developers determine when to use Generative AI and, crucially, when to avoid it. We argue for a curriculum shift, emphasizing that true digital literacy relies not only in knowing how to use Generative AI, but also on knowing when not to use it. 

---
# Emerging from Ground: Addressing Intent Deviation in Tool-Using Agents via Deriving Real Calls into Virtual Trajectories 

**Authors**: Qian Xiong, Yuekai Huang, Yujia Zheng, Tianhao Li, Ziyou Jiang, Zhiyuan Chang, Zhaoyang Li, Huanxiang Feng, Mingyang Li  

**Link**: [PDF](https://arxiv.org/pdf/2601.15120)  

**Abstract**: LLMs have advanced tool-using agents for real-world applications, yet they often lead to unexpected behaviors or results. Beyond obvious failures, the subtle issue of "intent deviation" severely hinders reliable evaluation and performance improvement. Existing post-training methods generally leverage either real system samples or virtual data simulated by LLMs. However, the former is costly due to reliance on hand-crafted user requests, while the latter suffers from distribution shift from the real tools in the wild. Additionally, both methods lack negative samples tailored to intent deviation scenarios, hindering effective guidance on preference learning. We introduce RISE, a "Real-to-Virtual" method designed to mitigate intent deviation. Anchoring on verified tool primitives, RISE synthesizes virtual trajectories and generates diverse negative samples through mutation on critical parameters. With synthetic data, RISE fine-tunes backbone LLMs via the two-stage training for intent alignment. Evaluation results demonstrate that data synthesized by RISE achieve promising results in eight metrics covering user requires, execution trajectories and agent responses. Integrating with training, RISE achieves an average 35.28% improvement in Acctask (task completion) and 23.27% in Accintent (intent alignment), outperforming SOTA baselines by 1.20--42.09% and 1.17--54.93% respectively. 

---
# The Why Behind the Action: Unveiling Internal Drivers via Agentic Attribution 

**Authors**: Chen Qian, Peng Wang, Dongrui Liu, Junyao Yang, Dadi Guo, Ling Tang, Jilin Mei, Qihan Ren, Shuai Shao, Yong Liu, Jie Fu, Jing Shao, Xia Hu  

**Link**: [PDF](https://arxiv.org/pdf/2601.15075)  

**Abstract**: Large Language Model (LLM)-based agents are widely used in real-world applications such as customer service, web navigation, and software engineering. As these systems become more autonomous and are deployed at scale, understanding why an agent takes a particular action becomes increasingly important for accountability and governance. However, existing research predominantly focuses on \textit{failure attribution} to localize explicit errors in unsuccessful trajectories, which is insufficient for explaining the reasoning behind agent behaviors. To bridge this gap, we propose a novel framework for \textbf{general agentic attribution}, designed to identify the internal factors driving agent actions regardless of the task outcome. Our framework operates hierarchically to manage the complexity of agent interactions. Specifically, at the \textit{component level}, we employ temporal likelihood dynamics to identify critical interaction steps; then at the \textit{sentence level}, we refine this localization using perturbation-based analysis to isolate the specific textual evidence. We validate our framework across a diverse suite of agentic scenarios, including standard tool use and subtle reliability risks like memory-induced bias. Experimental results demonstrate that the proposed framework reliably pinpoints pivotal historical events and sentences behind the agent behavior, offering a critical step toward safer and more accountable agentic systems. 

---
# The Responsibility Vacuum: Organizational Failure in Scaled Agent Systems 

**Authors**: Oleg Romanchuk, Roman Bondar  

**Link**: [PDF](https://arxiv.org/pdf/2601.15059)  

**Abstract**: Modern CI/CD pipelines integrating agent-generated code exhibit a structural failure in responsibility attribution. Decisions are executed through formally correct approval processes, yet no entity possesses both the authority to approve those decisions and the epistemic capacity to meaningfully understand their basis.
We define this condition as responsibility vacuum: a state in which decisions occur, but responsibility cannot be attributed because authority and verification capacity do not coincide. We show that this is not a process deviation or technical defect, but a structural property of deployments where decision generation throughput exceeds bounded human verification capacity.
We identify a scaling limit under standard deployment assumptions, including parallel agent generation, CI-based validation, and individualized human approval gates. Beyond a throughput threshold, verification ceases to function as a decision criterion and is replaced by ritualized approval based on proxy signals. Personalized responsibility becomes structurally unattainable in this regime.
We further characterize a CI amplification dynamic, whereby increasing automated validation coverage raises proxy signal density without restoring human capacity. Under fixed time and attention constraints, this accelerates cognitive offloading in the broad sense and widens the gap between formal approval and epistemic understanding. Additional automation therefore amplifies, rather than mitigates, the responsibility vacuum.
We conclude that unless organizations explicitly redesign decision boundaries or reassign responsibility away from individual decisions toward batch- or system-level ownership, responsibility vacuum remains an invisible but persistent failure mode in scaled agent deployments. 

---
# Emergent, not Immanent: A Baradian Reading of Explainable AI 

**Authors**: Fabio Morreale, Joan Serrà, Yuki Mistufuji  

**Link**: [PDF](https://arxiv.org/pdf/2601.15029)  

**Abstract**: Explainable AI (XAI) is frequently positioned as a technical problem of revealing the inner workings of an AI model. This position is affected by unexamined onto-epistemological assumptions: meaning is treated as immanent to the model, the explainer is positioned outside the system, and a causal structure is presumed recoverable through computational techniques. In this paper, we draw on Barad's agential realism to develop an alternative onto-epistemology of XAI. We propose that interpretations are material-discursive performances that emerge from situated entanglements of the AI model with humans, context, and the interpretative apparatus. To develop this position, we read a comprehensive set of XAI methods through agential realism and reveal the assumptions and limitations that underpin several of these methods. We then articulate the framework's ethical dimension and propose design directions for XAI interfaces that support emergent interpretation, using a speculative text-to-music interface as a case study. 

---
# Multi-Behavior Sequential Modeling with Transition-Aware Graph Attention Network for E-Commerce Recommendation 

**Authors**: Hanqi Jin, Gaoming Yang, Zhangming Chan, Yapeng Yuan, Longbin Li, Fei Sun, Yeqiu Yang, Jian Wu, Yuning Jiang, Bo Zheng  

**Link**: [PDF](https://arxiv.org/pdf/2601.14955)  

**Abstract**: User interactions on e-commerce platforms are inherently diverse, involving behaviors such as clicking, favoriting, adding to cart, and purchasing. The transitions between these behaviors offer valuable insights into user-item interactions, serving as a key signal for un- derstanding evolving preferences. Consequently, there is growing interest in leveraging multi-behavior data to better capture user intent. Recent studies have explored sequential modeling of multi- behavior data, many relying on transformer-based architectures with polynomial time complexity. While effective, these approaches often incur high computational costs, limiting their applicability in large-scale industrial systems with long user sequences. To address this challenge, we propose the Transition-Aware Graph Attention Network (TGA), a linear-complexity approach for modeling multi-behavior transitions. Unlike traditional trans- formers that treat all behavior pairs equally, TGA constructs a structured sparse graph by identifying informative transitions from three perspectives: (a) item-level transitions, (b) category-level transitions, and (c) neighbor-level transitions. Built upon the structured graph, TGA employs a transition-aware graph Attention mechanism that jointly models user-item interactions and behav- ior transition types, enabling more accurate capture of sequential patterns while maintaining computational efficiency. Experiments show that TGA outperforms all state-of-the-art models while sig- nificantly reducing computational cost. Notably, TGA has been deployed in a large-scale industrial production environment, where it leads to impressive improvements in key business metrics. 

---
# Just aware enough: Evaluating awareness across artificial systems 

**Authors**: Nadine Meertens, Suet Lee, Ophelia Deroy  

**Link**: [PDF](https://arxiv.org/pdf/2601.14901)  

**Abstract**: Recent debates on artificial intelligence increasingly emphasise questions of AI consciousness and moral status, yet there remains little agreement on how such properties should be evaluated. In this paper, we argue that awareness offers a more productive and methodologically tractable alternative. We introduce a practical method for evaluating awareness across diverse systems, where awareness is understood as encompassing a system's abilities to process, store and use information in the service of goal-directed action. Central to this approach is the claim that any evaluation aiming to capture the diversity of artificial systems must be domain-sensitive, deployable at any scale, multidimensional, and enable the prediction of task performance, while generalising to the level of abilities for the sake of comparison. Given these four desiderata, we outline a structured approach to evaluating and comparing awareness profiles across artificial systems with differing architectures, scales, and operational domains. By shifting the focus from artificial consciousness to being just aware enough, this approach aims to facilitate principled assessment, support design and oversight, and enable more constructive scientific and public discourse. 

---
# To Neuro-Symbolic Classification and Beyond by Compiling Description Logic Ontologies to Probabilistic Circuits 

**Authors**: Nicolas Lazzari, Valentina Presutti, Antonio Vergari  

**Link**: [PDF](https://arxiv.org/pdf/2601.14894)  

**Abstract**: Background: Neuro-symbolic methods enhance the reliability of neural network classifiers through logical constraints, but they lack native support for ontologies.
Objectives: We aim to develop a neuro-symbolic method that reliably outputs predictions consistent with a Description Logic ontology that formalizes domain-specific knowledge.
Methods: We encode a Description Logic ontology as a circuit, a feed-forward differentiable computational graph that supports tractable execution of queries and transformations. We show that the circuit can be used to (i) generate synthetic datasets that capture the semantics of the ontology; (ii) efficiently perform deductive reasoning on a GPU; (iii) implement neuro-symbolic models whose predictions are approximately or provably consistent with the knowledge defined in the ontology.
Results We show that the synthetic dataset generated using the circuit qualitatively captures the semantics of the ontology while being challenging for Machine Learning classifiers, including neural networks. Moreover, we show that compiling the ontology into a circuit is a promising approach for scalable deductive reasoning, with runtimes up to three orders of magnitude faster than available reasoners. Finally, we show that our neuro-symbolic classifiers reliably produce consistent predictions when compared to neural network baselines, maintaining competitive performances or even outperforming them.
Conclusions By compiling Description Logic ontologies into circuits, we obtain a tighter integration between the Deep Learning and Knowledge Representation fields. We show that a single circuit representation can be used to tackle different challenging tasks closely related to real-world applications. 

---
# Implementing Knowledge Representation and Reasoning with Object Oriented Design 

**Authors**: Abdelrhman Bassiouny, Tom Schierenbeck, Sorin Arion, Benjamin Alt, Naren Vasantakumaar, Giang Nguyen, Michael Beetz  

**Link**: [PDF](https://arxiv.org/pdf/2601.14840)  

**Abstract**: This paper introduces KRROOD, a framework designed to bridge the integration gap between modern software engineering and Knowledge Representation & Reasoning (KR&R) systems. While Object-Oriented Programming (OOP) is the standard for developing complex applications, existing KR&R frameworks often rely on external ontologies and specialized languages that are difficult to integrate with imperative code. KRROOD addresses this by treating knowledge as a first-class programming abstraction using native class structures, bridging the gap between the logic programming and OOP paradigms. We evaluate the system on the OWL2Bench benchmark and a human-robot task learning scenario. Experimental results show that KRROOD achieves strong performance while supporting the expressive reasoning required for real-world autonomous systems. 

---
# Measuring and Aligning Abstraction in Vision-Language Models with Medical Taxonomies 

**Authors**: Ben Schaper, Maxime Di Folco, Bernhard Kainz, Julia A. Schnabel, Cosmin I. Bercea  

**Link**: [PDF](https://arxiv.org/pdf/2601.14827)  

**Abstract**: Vision-Language Models show strong zero-shot performance for chest X-ray classification, but standard flat metrics fail to distinguish between clinically minor and severe errors. This work investigates how to quantify and mitigate abstraction errors by leveraging medical taxonomies. We benchmark several state-of-the-art VLMs using hierarchical metrics and introduce Catastrophic Abstraction Errors to capture cross-branch mistakes. Our results reveal substantial misalignment of VLMs with clinical taxonomies despite high flat performance. To address this, we propose risk-constrained thresholding and taxonomy-aware fine-tuning with radial embeddings, which reduce severe abstraction errors to below 2 per cent while maintaining competitive performance. These findings highlight the importance of hierarchical evaluation and representation-level alignment for safer and more clinically meaningful deployment of VLMs. 

---
# CI4A: Semantic Component Interfaces for Agents Empowering Web Automation 

**Authors**: Zhi Qiu, Jiazheng Sun, Chenxiao Xia, Jun Zheng, Xin Peng  

**Link**: [PDF](https://arxiv.org/pdf/2601.14790)  

**Abstract**: While Large Language Models demonstrate remarkable proficiency in high-level semantic planning, they remain limited in handling fine-grained, low-level web component manipulations. To address this limitation, extensive research has focused on enhancing model grounding capabilities through techniques such as Reinforcement Learning. However, rather than compelling agents to adapt to human-centric interfaces, we propose constructing interaction interfaces specifically optimized for agents. This paper introduces Component Interface for Agent (CI4A), a semantic encapsulation mechanism that abstracts the complex interaction logic of UI components into a set of unified tool primitives accessible to agents. We implemented CI4A within Ant Design, an industrial-grade front-end framework, covering 23 categories of commonly used UI components. Furthermore, we developed a hybrid agent featuring an action space that dynamically updates according to the page state, enabling flexible invocation of available CI4A tools. Leveraging the CI4A-integrated Ant Design, we refactored and upgraded the WebArena benchmark to evaluate existing SoTA methods. Experimental results demonstrate that the CI4A-based agent significantly outperforms existing approaches, achieving a new SoTA task success rate of 86.3%, alongside substantial improvements in execution efficiency. 

---
# Towards Bound Consistency for the No-Overlap Constraint Using MDDs 

**Authors**: Amaury Guichard, Laurent Michel, Hélène Verhaeghe, Pierre Schaus  

**Link**: [PDF](https://arxiv.org/pdf/2601.14784)  

**Abstract**: Achieving bound consistency for the no-overlap constraint is known to be NP-complete. Therefore, several polynomial-time tightening techniques, such as edge finding, not-first-not-last reasoning, and energetic reasoning, have been introduced for this constraint. In this work, we derive the first bound-consistent algorithm for the no-overlap constraint. By building on the no-overlap MDD defined by Ciré and van Hoeve, we extract bounds of the time window of the jobs, allowing us to tighten start and end times in time polynomial in the number of nodes of the MDD. Similarly, to bound the size and time-complexity, we limit the width of the MDD to a threshold, creating a relaxed MDD that can also be used to relax the bound-consistent filtering. Through experiments on a sequencing problem with time windows and a just-in-time objective ($1 \mid r_j, d_j, \bar{d}_j \mid \sum E_j + \sum T_j$), we observe that the proposed filtering, even with a threshold on the width, achieves a stronger reduction in the number of nodes visited in the search tree compared to the previously proposed precedence-detection algorithm of Ciré and van Hoeve. The new filtering also appears to be complementary to classical propagation methods for the no-overlap constraint, allowing a substantial reduction in both the number of nodes and the solving time on several instances. 

---
# Semantic-Guided Unsupervised Video Summarization 

**Authors**: Haizhou Liu, Haodong Jin, Yiming Wang, Hui Yu  

**Link**: [PDF](https://arxiv.org/pdf/2601.14773)  

**Abstract**: Video summarization is a crucial technique for social understanding, enabling efficient browsing of massive multimedia content and extraction of key information from social platforms. Most existing unsupervised summarization methods rely on Generative Adversarial Networks (GANs) to enhance keyframe selection and generate coherent, video summaries through adversarial training. However, such approaches primarily exploit unimodal features, overlooking the guiding role of semantic information in keyframe selection, and often suffer from unstable training. To address these limitations, we propose a novel Semantic-Guided Unsupervised Video Summarization method. Specifically, we design a novel frame-level semantic alignment attention mechanism and integrate it into a keyframe selector, which guides the Transformer-based generator within the adversarial framework to better reconstruct videos. In addition, we adopt an incremental training strategy to progressively update the model components, effectively mitigating the instability of GAN training. Experimental results demonstrate that our approach achieves superior performance on multiple benchmark datasets. 

---
# An XAI View on Explainable ASP: Methods, Systems, and Perspectives 

**Authors**: Thomas Eiter, Tobias Geibinger, Zeynep G. Saribatur  

**Link**: [PDF](https://arxiv.org/pdf/2601.14764)  

**Abstract**: Answer Set Programming (ASP) is a popular declarative reasoning and problem solving approach in symbolic AI. Its rule-based formalism makes it inherently attractive for explainable and interpretive reasoning, which is gaining importance with the surge of Explainable AI (XAI). A number of explanation approaches and tools for ASP have been developed, which often tackle specific explanatory settings and may not cover all scenarios that ASP users encounter. In this survey, we provide, guided by an XAI perspective, an overview of types of ASP explanations in connection with user questions for explanation, and describe how their coverage by current theory and tools. Furthermore, we pinpoint gaps in existing ASP explanations approaches and identify research directions for future work. 

---
# DARA: Few-shot Budget Allocation in Online Advertising via In-Context Decision Making with RL-Finetuned LLMs 

**Authors**: Mingxuan Song, Yusen Huo, Bohan Zhou, Shenglin Yin, Zhen Xiao, Jieyi Long, Zhilin Zhang, Chuan Yu  

**Link**: [PDF](https://arxiv.org/pdf/2601.14711)  

**Abstract**: Optimizing the advertiser's cumulative value of winning impressions under budget constraints poses a complex challenge in online advertising, under the paradigm of AI-Generated Bidding (AIGB). Advertisers often have personalized objectives but limited historical interaction data, resulting in few-shot scenarios where traditional reinforcement learning (RL) methods struggle to perform effectively. Large Language Models (LLMs) offer a promising alternative for AIGB by leveraging their in-context learning capabilities to generalize from limited data. However, they lack the numerical precision required for fine-grained optimization. To address this limitation, we introduce GRPO-Adaptive, an efficient LLM post-training strategy that enhances both reasoning and numerical precision by dynamically updating the reference policy during training. Built upon this foundation, we further propose DARA, a novel dual-phase framework that decomposes the decision-making process into two stages: a few-shot reasoner that generates initial plans via in-context prompting, and a fine-grained optimizer that refines these plans using feedback-driven reasoning. This separation allows DARA to combine LLMs' in-context learning strengths with precise adaptability required by AIGB tasks. Extensive experiments on both real-world and synthetic data environments demonstrate that our approach consistently outperforms existing baselines in terms of cumulative advertiser value under budget constraints. 

---
# AutoDriDM: An Explainable Benchmark for Decision-Making of Vision-Language Models in Autonomous Driving 

**Authors**: Zecong Tang, Zixu Wang, Yifei Wang, Weitong Lian, Tianjian Gao, Haoran Li, Tengju Ru, Lingyi Meng, Zhejun Cui, Yichen Zhu, Qi Kang, Kaixuan Wang, Yu Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2601.14702)  

**Abstract**: Autonomous driving is a highly challenging domain that requires reliable perception and safe decision-making in complex scenarios. Recent vision-language models (VLMs) demonstrate reasoning and generalization abilities, opening new possibilities for autonomous driving; however, existing benchmarks and metrics overemphasize perceptual competence and fail to adequately assess decision-making processes. In this work, we present AutoDriDM, a decision-centric, progressive benchmark with 6,650 questions across three dimensions - Object, Scene, and Decision. We evaluate mainstream VLMs to delineate the perception-to-decision capability boundary in autonomous driving, and our correlation analysis reveals weak alignment between perception and decision-making performance. We further conduct explainability analyses of models' reasoning processes, identifying key failure modes such as logical reasoning errors, and introduce an analyzer model to automate large-scale annotation. AutoDriDM bridges the gap between perception-centered and decision-centered evaluation, providing guidance toward safer and more reliable VLMs for real-world autonomous driving. 

---
# Gaming the Judge: Unfaithful Chain-of-Thought Can Undermine Agent Evaluation 

**Authors**: Muhammad Khalifa, Lajanugen Logeswaran, Jaekyeom Kim, Sungryull Sohn, Yunxiang Zhang, Moontae Lee, Hao Peng, Lu Wang, Honglak Lee  

**Link**: [PDF](https://arxiv.org/pdf/2601.14691)  

**Abstract**: Large language models (LLMs) are increasingly used as judges to evaluate agent performance, particularly in non-verifiable settings where judgments rely on agent trajectories including chain-of-thought (CoT) reasoning. This paradigm implicitly assumes that the agent's CoT faithfully reflects both its internal reasoning and the underlying environment state. We show this assumption is brittle: LLM judges are highly susceptible to manipulation of agent reasoning traces. By systematically rewriting agent CoTs while holding actions and observations fixed, we demonstrate that manipulated reasoning alone can inflate false positive rates of state-of-the-art VLM judges by up to 90% across 800 trajectories spanning diverse web tasks. We study manipulation strategies spanning style-based approaches that alter only the presentation of reasoning and content-based approaches that fabricate signals of task progress, and find that content-based manipulations are consistently more effective. We evaluate prompting-based techniques and scaling judge-time compute, which reduce but do not fully eliminate susceptibility to manipulation. Our findings reveal a fundamental vulnerability in LLM-based evaluation and highlight the need for judging mechanisms that verify reasoning claims against observable evidence. 

---
# IB-GRPO: Aligning LLM-based Learning Path Recommendation with Educational Objectives via Indicator-Based Group Relative Policy Optimization 

**Authors**: Shuai Wang, Yaoming Yang, Bingdong Li, Hao Hao, Aimin Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2601.14686)  

**Abstract**: Learning Path Recommendation (LPR) aims to generate personalized sequences of learning items that maximize long-term learning effect while respecting pedagogical principles and operational constraints. Although large language models (LLMs) offer rich semantic understanding for free-form recommendation, applying them to long-horizon LPR is challenging due to (i) misalignment with pedagogical objectives such as the Zone of Proximal Development (ZPD) under sparse, delayed feedback, (ii) scarce and costly expert demonstrations, and (iii) multi-objective interactions among learning effect, difficulty scheduling, length controllability, and trajectory diversity. To address these issues, we propose IB-GRPO (Indicator-Based Group Relative Policy Optimization), an indicator-guided alignment approach for LLM-based LPR. To mitigate data scarcity, we construct hybrid expert demonstrations via Genetic Algorithm search and teacher RL agents and warm-start the LLM with supervised fine-tuning. Building on this warm-start, we design a within-session ZPD alignment score for difficulty scheduling. IB-GRPO then uses the $I_{\epsilon+}$ dominance indicator to compute group-relative advantages over multiple objectives, avoiding manual scalarization and improving Pareto trade-offs. Experiments on ASSIST09 and Junyi using the KES simulator with a Qwen2.5-7B backbone show consistent improvements over representative RL and LLM baselines. 

---
# Local Language Models for Context-Aware Adaptive Anonymization of Sensitive Text 

**Authors**: Aisvarya Adeseye, Jouni Isoaho, Seppo Virtanen, Mohammad Tahir  

**Link**: [PDF](https://arxiv.org/pdf/2601.14683)  

**Abstract**: Qualitative research often contains personal, contextual, and organizational details that pose privacy risks if not handled appropriately. Manual anonymization is time-consuming, inconsistent, and frequently omits critical identifiers. Existing automated tools tend to rely on pattern matching or fixed rules, which fail to capture context and may alter the meaning of the data. This study uses local LLMs to build a reliable, repeatable, and context-aware anonymization process for detecting and anonymizing sensitive data in qualitative transcripts. We introduce a Structured Framework for Adaptive Anonymizer (SFAA) that includes three steps: detection, classification, and adaptive anonymization. The SFAA incorporates four anonymization strategies: rule-based substitution, context-aware rewriting, generalization, and suppression. These strategies are applied based on the identifier type and the risk level. The identifiers handled by the SFAA are guided by major international privacy and research ethics standards, including the GDPR, HIPAA, and OECD guidelines. This study followed a dual-method evaluation that combined manual and LLM-assisted processing. Two case studies were used to support the evaluation. The first includes 82 face-to-face interviews on gamification in organizations. The second involves 93 machine-led interviews using an AI-powered interviewer to test LLM awareness and workplace privacy. Two local models, LLaMA and Phi were used to evaluate the performance of the proposed framework. The results indicate that the LLMs found more sensitive data than a human reviewer. Phi outperformed LLaMA in finding sensitive data, but made slightly more errors. Phi was able to find over 91% of the sensitive data and 94.8% kept the same sentiment as the original text, which means it was very accurate, hence, it does not affect the analysis of the qualitative data. 

---
# Query-Efficient Agentic Graph Extraction Attacks on GraphRAG Systems 

**Authors**: Shuhua Yang, Jiahao Zhang, Yilong Wang, Dongwon Lee, Suhang Wang  

**Link**: [PDF](https://arxiv.org/pdf/2601.14662)  

**Abstract**: Graph-based retrieval-augmented generation (GraphRAG) systems construct knowledge graphs over document collections to support multi-hop reasoning. While prior work shows that GraphRAG responses may leak retrieved subgraphs, the feasibility of query-efficient reconstruction of the hidden graph structure remains unexplored under realistic query budgets. We study a budget-constrained black-box setting where an adversary adaptively queries the system to steal its latent entity-relation graph. We propose AGEA (Agentic Graph Extraction Attack), a framework that leverages a novelty-guided exploration-exploitation strategy, external graph memory modules, and a two-stage graph extraction pipeline combining lightweight discovery with LLM-based filtering. We evaluate AGEA on medical, agriculture, and literary datasets across Microsoft-GraphRAG and LightRAG systems. Under identical query budgets, AGEA significantly outperforms prior attack baselines, recovering up to 90% of entities and relationships while maintaining high precision. These results demonstrate that modern GraphRAG systems are highly vulnerable to structured, agentic extraction attacks, even under strict query limits. 

---
# MAS-Orchestra: Understanding and Improving Multi-Agent Reasoning Through Holistic Orchestration and Controlled Benchmarks 

**Authors**: Zixuan Ke, Yifei Ming, Austin Xu, Ryan Chin, Xuan-Phi Nguyen, Prathyusha Jwalapuram, Semih Yavuz, Caiming Xiong, Shafiq Joty  

**Link**: [PDF](https://arxiv.org/pdf/2601.14652)  

**Abstract**: While multi-agent systems (MAS) promise elevated intelligence through coordination of agents, current approaches to automatic MAS design under-deliver. Such shortcomings stem from two key factors: (1) methodological complexity - agent orchestration is performed using sequential, code-level execution that limits global system-level holistic reasoning and scales poorly with agent complexity - and (2) efficacy uncertainty - MAS are deployed without understanding if there are tangible benefits compared to single-agent systems (SAS). We propose MAS-Orchestra, a training-time framework that formulates MAS orchestration as a function-calling reinforcement learning problem with holistic orchestration, generating an entire MAS at once. In MAS-Orchestra, complex, goal-oriented sub-agents are abstracted as callable functions, enabling global reasoning over system structure while hiding internal execution details. To rigorously study when and why MAS are beneficial, we introduce MASBENCH, a controlled benchmark that characterizes tasks along five axes: Depth, Horizon, Breadth, Parallel, and Robustness. Our analysis reveals that MAS gains depend critically on task structure, verification protocols, and the capabilities of both orchestrator and sub-agents, rather than holding universally. Guided by these insights, MAS-Orchestra achieves consistent improvements on public benchmarks including mathematical reasoning, multi-hop QA, and search-based QA. Together, MAS-Orchestra and MASBENCH enable better training and understanding of MAS in the pursuit of multi-agent intelligence. 

---
# Large Language Model-Powered Evolutionary Code Optimization on a Phylogenetic Tree 

**Authors**: Leyi Zhao, Weijie Huang, Yitong Guo, Jiang Bian, Chenghong Wang, Xuhong Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2601.14523)  

**Abstract**: Optimizing scientific computing algorithms for modern GPUs is a labor-intensive and iterative process involving repeated code modification, benchmarking, and tuning across complex hardware and software stacks. Recent work has explored large language model (LLM)-assisted evolutionary methods for automated code optimization, but these approaches primarily rely on outcome-based selection and random mutation, underutilizing the rich trajectory information generated during iterative optimization. We propose PhyloEvolve, an LLM-agent system that reframes GPU-oriented algorithm optimization as an In-Context Reinforcement Learning (ICRL) problem. This formulation enables trajectory-conditioned reuse of optimization experience without model retraining. PhyloEvolve integrates Algorithm Distillation and prompt-based Decision Transformers into an iterative workflow, treating sequences of algorithm modifications and performance feedback as first-class learning signals. To organize optimization history, we introduce a phylogenetic tree representation that captures inheritance, divergence, and recombination among algorithm variants, enabling backtracking, cross-lineage transfer, and reproducibility. The system combines elite trajectory pooling, multi-island parallel exploration, and containerized execution to balance exploration and exploitation across heterogeneous hardware. We evaluate PhyloEvolve on scientific computing workloads including PDE solvers, manifold learning, and spectral graph algorithms, demonstrating consistent improvements in runtime, memory efficiency, and correctness over baseline and evolutionary methods. Code is published at: this https URL 

---
# "Just in Time" World Modeling Supports Human Planning and Reasoning 

**Authors**: Tony Chen, Sam Cheyette, Kelsey Allen, Joshua Tenenbaum, Kevin Smith  

**Link**: [PDF](https://arxiv.org/pdf/2601.14514)  

**Abstract**: Probabilistic mental simulation is thought to play a key role in human reasoning, planning, and prediction, yet the demands of simulation in complex environments exceed realistic human capacity limits. A theory with growing evidence is that people simulate using simplified representations of the environment that abstract away from irrelevant details, but it is unclear how people determine these simplifications efficiently. Here, we present a "Just-in-Time" framework for simulation-based reasoning that demonstrates how such representations can be constructed online with minimal added computation. The model uses a tight interleaving of simulation, visual search, and representation modification, with the current simulation guiding where to look and visual search flagging objects that should be encoded for subsequent simulation. Despite only ever encoding a small subset of objects, the model makes high-utility predictions. We find strong empirical support for this account over alternative models in a grid-world planning task and a physical reasoning task across a range of behavioral measures. Together, these results offer a concrete algorithmic account of how people construct reduced representations to support efficient mental simulation. 

---
# Scalable Knee-Point Guided Activity Group Selection in Multi-Tree Genetic Programming for Dynamic Multi-Mode Project Scheduling 

**Authors**: Yuan Tian, Yi Mei, Mengjie Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2601.14485)  

**Abstract**: The dynamic multi-mode resource-constrained project scheduling problem is a challenging scheduling problem that requires making decisions on both the execution order of activities and their corresponding execution modes. Genetic programming has been widely applied as a hyper-heuristic to evolve priority rules that guide the selection of activity-mode pairs from the current eligible set. Recently, an activity group selection strategy has been proposed to select a subset of activities rather than a single activity at each decision point, allowing for more effective scheduling by considering the interdependence between activities. Although effective in small-scale instances, this strategy suffers from scalability issues when applied to larger problems. In this work, we enhance the scalability of the group selection strategy by introducing a knee-point-based selection mechanism to identify a promising subset of activities before evaluating their combinations. An activity ordering rule is first used to rank all eligible activity-mode pairs, followed by a knee point selection to find the promising pairs. Then, a group selection rule selects the best activity combination. We develop a multi-tree GP framework to evolve both types of rules simultaneously. Experimental results demonstrate that our approach scales well to large instances and outperforms GP with sequential decision-making in most scenarios. 

---
# On the Generalization Gap in LLM Planning: Tests and Verifier-Reward RL 

**Authors**: Valerio Belcamino, Nicholas Attolino, Alessio Capitanelli, Fulvio Mastrogiovanni  

**Link**: [PDF](https://arxiv.org/pdf/2601.14456)  

**Abstract**: Recent work shows that fine-tuned Large Language Models (LLMs) can achieve high valid plan rates on PDDL planning tasks. However, it remains unclear whether this reflects transferable planning competence or domain-specific memorization. In this work, we fine-tune a 1.7B-parameter LLM on 40,000 domain-problem-plan tuples from 10 IPC 2023 domains, and evaluate both in-domain and cross-domain generalization. While the model reaches 82.9% valid plan rate in in-domain conditions, it achieves 0% on two unseen domains. To analyze this failure, we introduce three diagnostic interventions, namely (i) instance-wise symbol anonymization, (ii) compact plan serialization, and (iii) verifier-reward fine-tuning using the VAL validator as a success-focused reinforcement signal. Symbol anonymization and compact serialization cause significant performance drops despite preserving plan semantics, thus revealing strong sensitivity to surface representations. Verifier-reward fine-tuning reaches performance saturation in half the supervised training epochs, but does not improve cross-domain generalization. For the explored configurations, in-domain performance plateaus around 80%, while cross-domain performance collapses, suggesting that our fine-tuned model relies heavily on domain-specific patterns rather than transferable planning competence in this setting. Our results highlight a persistent generalization gap in LLM-based planning and provide diagnostic tools for studying its causes. 

---
# VisTIRA: Closing the Image-Text Modality Gap in Visual Math Reasoning via Structured Tool Integration 

**Authors**: Saeed Khaki, Ashudeep Singh, Nima Safaei, Kamal Ginotra  

**Link**: [PDF](https://arxiv.org/pdf/2601.14440)  

**Abstract**: Vision-language models (VLMs) lag behind text-only language models on mathematical reasoning when the same problems are presented as images rather than text. We empirically characterize this as a modality gap: the same question in text form yields markedly higher accuracy than its visually typeset counterpart, due to compounded failures in reading dense formulas, layout, and mixed symbolic-diagrammatic context. First, we introduce VisTIRA (Vision and Tool-Integrated Reasoning Agent), a tool-integrated reasoning framework that enables structured problem solving by iteratively decomposing a given math problem (as an image) into natural language rationales and executable Python steps to determine the final answer. Second, we build a framework to measure and improve visual math reasoning: a LaTeX-based pipeline that converts chain-of-thought math corpora (e.g., NuminaMath) into challenging image counterparts, and a large set of synthetic tool-use trajectories derived from a real-world, homework-style image dataset (called SnapAsk) for fine-tuning VLMs. Our experiments show that tool-integrated supervision improves image-based reasoning, and OCR grounding can further narrow the gap for smaller models, although its benefit diminishes at scale. These findings highlight that modality gap severity inversely correlates with model size, and that structured reasoning and OCR-based grounding are complementary strategies for advancing visual mathematical reasoning. 

---
# Epistemic Constitutionalism Or: how to avoid coherence bias 

**Authors**: Michele Loi  

**Link**: [PDF](https://arxiv.org/pdf/2601.14295)  

**Abstract**: Large language models increasingly function as artificial reasoners: they evaluate arguments, assign credibility, and express confidence. Yet their belief-forming behavior is governed by implicit, uninspected epistemic policies. This paper argues for an epistemic constitution for AI: explicit, contestable meta-norms that regulate how systems form and express beliefs. Source attribution bias provides the motivating case: I show that frontier models enforce identity-stance coherence, penalizing arguments attributed to sources whose expected ideological position conflicts with the argument's content. When models detect systematic testing, these effects collapse, revealing that systems treat source-sensitivity as bias to suppress rather than as a capacity to execute well. I distinguish two constitutional approaches: the Platonic, which mandates formal correctness and default source-independence from a privileged standpoint, and the Liberal, which refuses such privilege, specifying procedural norms that protect conditions for collective inquiry while allowing principled source-attending grounded in epistemic vigilance. I argue for the Liberal approach, sketch a constitutional core of eight principles and four orientations, and propose that AI epistemic governance requires the same explicit, contestable structure we now expect for AI ethics. 

---
# The Ontological Neutrality Theorem: Why Neutral Ontological Substrates Must Be Pre-Causal and Pre-Normative 

**Authors**: Denise M. Case  

**Link**: [PDF](https://arxiv.org/pdf/2601.14271)  

**Abstract**: Modern data systems must support accountability across persistent legal, political, and analytic disagreement. This requirement imposes strict constraints on the design of any ontology intended to function as a shared substrate. We establish an impossibility result for ontological neutrality: neutrality, understood as interpretive non-commitment and stability under incompatible extensions, is incompatible with the inclusion of causal or normative commitments at the foundational layer. Any ontology that asserts causal or deontic conclusions as ontological facts cannot serve as a neutral substrate across divergent frameworks without revision or contradiction. It follows that neutral ontological substrates must be pre-causal and pre-normative, representing entities, together with identity and persistence conditions, while externalizing interpretation, evaluation, and explanation. This paper does not propose a specific ontology or protocol; rather, it establishes the necessary design constraints for any system intended to maintain a shared, stable representation of reality across conflicting interpretive frameworks. 

---
# Iterative Refinement Improves Compositional Image Generation 

**Authors**: Shantanu Jaiswal, Mihir Prabhudesai, Nikash Bhardwaj, Zheyang Qin, Amir Zadeh, Chuan Li, Katerina Fragkiadaki, Deepak Pathak  

**Link**: [PDF](https://arxiv.org/pdf/2601.15286)  

**Abstract**: Text-to-image (T2I) models have achieved remarkable progress, yet they continue to struggle with complex prompts that require simultaneously handling multiple objects, relations, and attributes. Existing inference-time strategies, such as parallel sampling with verifiers or simply increasing denoising steps, can improve prompt alignment but remain inadequate for richly compositional settings where many constraints must be satisfied. Inspired by the success of chain-of-thought reasoning in large language models, we propose an iterative test-time strategy in which a T2I model progressively refines its generations across multiple steps, guided by feedback from a vision-language model as the critic in the loop. Our approach is simple, requires no external tools or priors, and can be flexibly applied to a wide range of image generators and vision-language models. Empirically, we demonstrate consistent gains on image generation across benchmarks: a 16.9% improvement in all-correct rate on ConceptMix (k=7), a 13.8% improvement on T2I-CompBench (3D-Spatial category) and a 12.5% improvement on Visual Jenga scene decomposition compared to compute-matched parallel sampling. Beyond quantitative gains, iterative refinement produces more faithful generations by decomposing complex prompts into sequential corrections, with human evaluators preferring our method 58.7% of the time over 41.3% for the parallel baseline. Together, these findings highlight iterative self-correction as a broadly applicable principle for compositional image generation. Results and visualizations are available at this https URL 

---
# Rethinking Video Generation Model for the Embodied World 

**Authors**: Yufan Deng, Zilin Pan, Hongyu Zhang, Xiaojie Li, Ruoqing Hu, Yufei Ding, Yiming Zou, Yan Zeng, Daquan Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2601.15282)  

**Abstract**: Video generation models have significantly advanced embodied intelligence, unlocking new possibilities for generating diverse robot data that capture perception, reasoning, and action in the physical world. However, synthesizing high-quality videos that accurately reflect real-world robotic interactions remains challenging, and the lack of a standardized benchmark limits fair comparisons and progress. To address this gap, we introduce a comprehensive robotics benchmark, RBench, designed to evaluate robot-oriented video generation across five task domains and four distinct embodiments. It assesses both task-level correctness and visual fidelity through reproducible sub-metrics, including structural consistency, physical plausibility, and action completeness. Evaluation of 25 representative models highlights significant deficiencies in generating physically realistic robot behaviors. Furthermore, the benchmark achieves a Spearman correlation coefficient of 0.96 with human evaluations, validating its effectiveness. While RBench provides the necessary lens to identify these deficiencies, achieving physical realism requires moving beyond evaluation to address the critical shortage of high-quality training data. Driven by these insights, we introduce a refined four-stage data pipeline, resulting in RoVid-X, the largest open-source robotic dataset for video generation with 4 million annotated video clips, covering thousands of tasks and enriched with comprehensive physical property annotations. Collectively, this synergistic ecosystem of evaluation and data establishes a robust foundation for rigorous assessment and scalable training of video models, accelerating the evolution of embodied AI toward general intelligence. 

---
# MolecularIQ: Characterizing Chemical Reasoning Capabilities Through Symbolic Verification on Molecular Graphs 

**Authors**: Christoph Bartmann, Johannes Schimunek, Mykyta Ielanskyi, Philipp Seidl, Günter Klambauer, Sohvi Luukkonen  

**Link**: [PDF](https://arxiv.org/pdf/2601.15279)  

**Abstract**: A molecule's properties are fundamentally determined by its composition and structure encoded in its molecular graph. Thus, reasoning about molecular properties requires the ability to parse and understand the molecular graph. Large Language Models (LLMs) are increasingly applied to chemistry, tackling tasks such as molecular name conversion, captioning, text-guided generation, and property or reaction prediction. Most existing benchmarks emphasize general chemical knowledge, rely on literature or surrogate labels that risk leakage or bias, or reduce evaluation to multiple-choice questions. We introduce MolecularIQ, a molecular structure reasoning benchmark focused exclusively on symbolically verifiable tasks. MolecularIQ enables fine-grained evaluation of reasoning over molecular graphs and reveals capability patterns that localize model failures to specific tasks and molecular structures. This provides actionable insights into the strengths and limitations of current chemistry LLMs and guides the development of models that reason faithfully over molecular structure. 

---
# Evaluation of Large Language Models in Legal Applications: Challenges, Methods, and Future Directions 

**Authors**: Yiran Hu, Huanghai Liu, Chong Wang, Kunran Li, Tien-Hsuan Wu, Haitao Li, Xinran Xu, Siqing Huo, Weihang Su, Ning Zheng, Siyuan Zheng, Qingyao Ai, Yun Liu, Renjun Bian, Yiqun Liu, Charles L.A. Clarke, Weixing Shen, Ben Kao  

**Link**: [PDF](https://arxiv.org/pdf/2601.15267)  

**Abstract**: Large language models (LLMs) are being increasingly integrated into legal applications, including judicial decision support, legal practice assistance, and public-facing legal services. While LLMs show strong potential in handling legal knowledge and tasks, their deployment in real-world legal settings raises critical concerns beyond surface-level accuracy, involving the soundness of legal reasoning processes and trustworthy issues such as fairness and reliability. Systematic evaluation of LLM performance in legal tasks has therefore become essential for their responsible adoption. This survey identifies key challenges in evaluating LLMs for legal tasks grounded in real-world legal practice. We analyze the major difficulties involved in assessing LLM performance in the legal domain, including outcome correctness, reasoning reliability, and trustworthiness. Building on these challenges, we review and categorize existing evaluation methods and benchmarks according to their task design, datasets, and evaluation metrics. We further discuss the extent to which current approaches address these challenges, highlight their limitations, and outline future research directions toward more realistic, reliable, and legally grounded evaluation frameworks for LLMs in legal domains. 

---
# Many Experiments, Few Repetitions, Unpaired Data, and Sparse Effects: Is Causal Inference Possible? 

**Authors**: Felix Schur, Niklas Pfister, Peng Ding, Sach Mukherjee, Jonas Peters  

**Link**: [PDF](https://arxiv.org/pdf/2601.15254)  

**Abstract**: We study the problem of estimating causal effects under hidden confounding in the following unpaired data setting: we observe some covariates $X$ and an outcome $Y$ under different experimental conditions (environments) but do not observe them jointly; we either observe $X$ or $Y$. Under appropriate regularity conditions, the problem can be cast as an instrumental variable (IV) regression with the environment acting as a (possibly high-dimensional) instrument. When there are many environments but only a few observations per environment, standard two-sample IV estimators fail to be consistent. We propose a GMM-type estimator based on cross-fold sample splitting of the instrument-covariate sample and prove that it is consistent as the number of environments grows but the sample size per environment remains constant. We further extend the method to sparse causal effects via $\ell_1$-regularized estimation and post-selection refitting. 

---
# Recommending Best Paper Awards for ML/AI Conferences via the Isotonic Mechanism 

**Authors**: Garrett G. Wen, Buxin Su, Natalie Collina, Zhun Deng, Weijie Su  

**Link**: [PDF](https://arxiv.org/pdf/2601.15249)  

**Abstract**: Machine learning and artificial intelligence conferences such as NeurIPS and ICML now regularly receive tens of thousands of submissions, posing significant challenges to maintaining the quality and consistency of the peer review process. This challenge is particularly acute for best paper awards, which are an important part of the peer review process, yet whose selection has increasingly become a subject of debate in recent years. In this paper, we introduce an author-assisted mechanism to facilitate the selection of best paper awards. Our method employs the Isotonic Mechanism for eliciting authors' assessments of their own submissions in the form of a ranking, which is subsequently utilized to adjust the raw review scores for optimal estimation of the submissions' ground-truth quality. We demonstrate that authors are incentivized to report truthfully when their utility is a convex additive function of the adjusted scores, and we validate this convexity assumption for best paper awards using publicly accessible review data of ICLR from 2019 to 2023 and NeurIPS from 2021 to 2023. Crucially, in the special case where an author has a single quota -- that is, may nominate only one paper -- we prove that truthfulness holds even when the utility function is merely nondecreasing and additive. This finding represents a substantial relaxation of the assumptions required in prior work. For practical implementation, we extend our mechanism to accommodate the common scenario of overlapping authorship. Finally, simulation results demonstrate that our mechanism significantly improves the quality of papers selected for awards. 

---
# Feasibility Preservation under Monotone Retrieval Truncation 

**Authors**: Sean Plummer  

**Link**: [PDF](https://arxiv.org/pdf/2601.15241)  

**Abstract**: Retrieval-based systems approximate access to a corpus by exposing only a truncated subset of available evidence. Even when relevant information exists in the corpus, truncation can prevent compatible evidence from co-occurring, leading to failures that are not captured by relevance-based evaluation. This paper studies retrieval from a structural perspective, modeling query answering as a feasibility problem under truncation.
We formalize retrieval as a sequence of candidate evidence sets and characterize conditions under which feasibility in the limit implies feasibility at finite retrieval depth. We show that monotone truncation suffices to guarantee finite witnessability for individual queries. For classes of queries, we identify finite generation of witness certificates as the additional condition required to obtain a uniform retrieval bound, and we show that this condition is necessary. We further exhibit sharp counterexamples demonstrating failure under non-monotone truncation, non-finitely-generated query classes, and purely slotwise coverage.
Together, these results isolate feasibility preservation as a correctness criterion for retrieval independent of relevance scoring or optimization, and clarify structural limitations inherent to truncation-based retrieval. 

---
# Tracing 3D Anatomy in 2D Strokes: A Multi-Stage Projection Driven Approach to Cervical Spine Fracture Identification 

**Authors**: Fabi Nahian Madhurja, Rusab Sarmun, Muhammad E. H. Chowdhury, Adam Mushtak, Israa Al-Hashimi, Sohaib Bassam Zoghoul  

**Link**: [PDF](https://arxiv.org/pdf/2601.15235)  

**Abstract**: Cervical spine fractures are critical medical conditions requiring precise and efficient detection for effective clinical management. This study explores the viability of 2D projection-based vertebra segmentation for vertebra-level fracture detection in 3D CT volumes, presenting an end-to-end pipeline for automated analysis of cervical vertebrae (C1-C7). By approximating a 3D volume through optimized 2D axial, sagittal, and coronal projections, regions of interest are identified using the YOLOv8 model from all views and combined to approximate the 3D cervical spine area, achieving a 3D mIoU of 94.45 percent. This projection-based localization strategy reduces computational complexity compared to traditional 3D segmentation methods while maintaining high performance. It is followed by a DenseNet121-Unet-based multi-label segmentation leveraging variance- and energy-based projections, achieving a Dice score of 87.86 percent. Strategic approximation of 3D vertebral masks from these 2D segmentation masks enables the extraction of individual vertebra volumes. The volumes are analyzed for fractures using an ensemble of 2.5D Spatio-Sequential models incorporating both raw slices and projections per vertebra for complementary evaluation. This ensemble achieves vertebra-level and patient-level F1 scores of 68.15 and 82.26, and ROC-AUC scores of 91.62 and 83.04, respectively. We further validate our approach through an explainability study that provides saliency map visualizations highlighting anatomical regions relevant for diagnosis, and an interobserver variability analysis comparing our model's performance with expert radiologists, demonstrating competitive results. 

---
# Deaf and Hard of Hearing Access to Intelligent Personal Assistants: Comparison of Voice-Based Options with an LLM-Powered Touch Interface 

**Authors**: Paige S. DeVries, Michaela Okosi, Ming Li, Nora Dunphy. Gidey Gezae, Dante Conway, Abraham Glasser, Raja Kushalnagar, Christian Vogler  

**Link**: [PDF](https://arxiv.org/pdf/2601.15209)  

**Abstract**: We investigate intelligent personal assistants (IPAs) accessibility for deaf and hard of hearing (DHH) people who can use their voice in everyday communication. The inability of IPAs to understand diverse accents including deaf speech renders them largely inaccessible to non-signing and speaking DHH individuals. Using an Echo Show, we compare the usability of natural language input via spoken English; with Alexa's automatic speech recognition and a Wizard-of-Oz setting with a trained facilitator re-speaking commands against that of a large language model (LLM)-assisted touch interface in a mixed-methods study. The touch method was navigated through an LLM-powered "task prompter," which integrated the user's history and smart environment to suggest contextually-appropriate commands. Quantitative results showed no significant differences across both spoken English conditions vs LLM-assisted touch. Qualitative results showed variability in opinions on the usability of each method. Ultimately, it will be necessary to have robust deaf-accented speech recognized natively by IPAs. 

---
# Where Do AI Coding Agents Fail? An Empirical Study of Failed Agentic Pull Requests in GitHub 

**Authors**: Ramtin Ehsani, Sakshi Pathak, Shriya Rawal, Abdullah Al Mujahid, Mia Mohammad Imran, Preetha Chatterjee  

**Link**: [PDF](https://arxiv.org/pdf/2601.15195)  

**Abstract**: AI coding agents are now submitting pull requests (PRs) to software projects, acting not just as assistants but as autonomous contributors. As these agentic contributions are rapidly increasing across real repositories, little is known about how they behave in practice and why many of them fail to be merged. In this paper, we conduct a large-scale study of 33k agent-authored PRs made by five coding agents across GitHub. (RQ1) We first quantitatively characterize merged and not-merged PRs along four broad dimensions: 1) merge outcomes across task types, 2) code changes, 3) CI build results, and 4) review dynamics. We observe that tasks related to documentation, CI, and build update achieve the highest merge success, whereas performance and bug-fix tasks perform the worst. Not-merged PRs tend to involve larger code changes, touch more files, and often do not pass the project's CI/CD pipeline validation. (RQ2) To further investigate why some agentic PRs are not merged, we qualitatively analyze 600 PRs to derive a hierarchical taxonomy of rejection patterns. This analysis complements the quantitative findings in RQ1 by uncovering rejection reasons not captured by quantitative metrics, including lack of meaningful reviewer engagement, duplicate PRs, unwanted feature implementations, and agent misalignment. Together, our findings highlight key socio-technical and human-AI collaboration factors that are critical to improving the success of future agentic workflows. 

---
# Benchmarking Large Language Models for ABAP Code Generation: An Empirical Study on Iterative Improvement by Compiler Feedback 

**Authors**: Stephan Wallraven, Tim Köhne, Hartmut Westenberger, Andreas Moser  

**Link**: [PDF](https://arxiv.org/pdf/2601.15188)  

**Abstract**: This work investigates the performance of Large Language Models (LLMs) in generating ABAP code. Despite successful applications of generative AI in many programming languages, there are hardly any systematic analyses of ABAP code generation to date. The aim of the study is to empirically analyze to what extent various LLMs can generate syntactically correct and functional ABAP code, how effectively they use compiler feedback for iterative improvement, and which task types pose special challenges. For this purpose, a benchmark with 180 tasks is conducted, consisting of adapted HumanEval tasks and practical SAP scenarios. The results show significant performance differences between the models: more powerful LLMs achieve success rates of around 75% after several iterations and benefit greatly from compiler feedback, while smaller models perform significantly weaker. Overall, the study highlights the high potential of powerful LLMs for ABAP development processes, especially in iterative error correction. 

---
# Dynamic Management of a Deep Learning-Based Anomaly Detection System for 5G Networks 

**Authors**: Lorenzo Fernández Maimó, Alberto Huertas Celdrán, Manuel Gil Pérez, Félix J. García Clemente, Gregorio Martínez Pérez  

**Link**: [PDF](https://arxiv.org/pdf/2601.15177)  

**Abstract**: Fog and mobile edge computing (MEC) will play a key role in the upcoming fifth generation (5G) mobile networks to support decentralized applications, data analytics and management into the network itself by using a highly distributed compute model. Furthermore, increasing attention is paid to providing user-centric cybersecurity solutions, which particularly require collecting, processing and analyzing significantly large amount of data traffic and huge number of network connections in 5G networks. In this regard, this paper proposes a MEC-oriented solution in 5G mobile networks to detect network anomalies in real-time and in autonomic way. Our proposal uses deep learning techniques to analyze network flows and to detect network anomalies. Moreover, it uses policies in order to provide an efficient and dynamic management system of the computing resources used in the anomaly detection process. The paper presents relevant aspects of the deployment of the proposal and experimental results to show its performance. 

---
# The Flexibility Trap: Why Arbitrary Order Limits Reasoning Potential in Diffusion Language Models 

**Authors**: Zanlin Ni, Shenzhi Wang, Yang Yue, Tianyu Yu, Weilin Zhao, Yeguo Hua, Tianyi Chen, Jun Song, Cheng Yu, Bo Zheng, Gao Huang  

**Link**: [PDF](https://arxiv.org/pdf/2601.15165)  

**Abstract**: Diffusion Large Language Models (dLLMs) break the rigid left-to-right constraint of traditional LLMs, enabling token generation in arbitrary orders. Intuitively, this flexibility implies a solution space that strictly supersets the fixed autoregressive trajectory, theoretically unlocking superior reasoning potential for general tasks like mathematics and coding. Consequently, numerous works have leveraged reinforcement learning (RL) to elicit the reasoning capability of dLLMs. In this paper, we reveal a counter-intuitive reality: arbitrary order generation, in its current form, narrows rather than expands the reasoning boundary of dLLMs. We find that dLLMs tend to exploit this order flexibility to bypass high-uncertainty tokens that are crucial for exploration, leading to a premature collapse of the solution space. This observation challenges the premise of existing RL approaches for dLLMs, where considerable complexities, such as handling combinatorial trajectories and intractable likelihoods, are often devoted to preserving this flexibility. We demonstrate that effective reasoning is better elicited by intentionally forgoing arbitrary order and applying standard Group Relative Policy Optimization (GRPO) instead. Our approach, JustGRPO, is minimalist yet surprisingly effective (e.g., 89.1% accuracy on GSM8K) while fully retaining the parallel decoding ability of dLLMs. Project page: this https URL 

---
# V-CAGE: Context-Aware Generation and Verification for Scalable Long-Horizon Embodied Tasks 

**Authors**: Yaru Liu, Ao-bo Wang, Nanyang Ye  

**Link**: [PDF](https://arxiv.org/pdf/2601.15164)  

**Abstract**: Learning long-horizon embodied behaviors from synthetic data remains challenging because generated scenes are often physically implausible, language-driven programs frequently "succeed" without satisfying task semantics, and high-level instructions require grounding into executable action sequences. To address these limitations, we introduce V-CAGE, a closed-loop framework for generating robust, semantically aligned manipulation datasets at scale. First, we propose a context-aware instantiation mechanism that enforces geometric consistency during scene synthesis. By dynamically maintaining a map of prohibited spatial areas as objects are placed, our system prevents interpenetration and ensures reachable, conflict-free configurations in cluttered environments. Second, to bridge the gap between abstract intent and low-level control, we employ a hierarchical instruction decomposition module. This decomposes high-level goals (e.g., "get ready for work") into compositional action primitives, facilitating coherent long-horizon planning. Crucially, we enforce semantic correctness through a VLM-based verification loop. Acting as a visual critic, the VLM performs rigorous rejection sampling after each subtask, filtering out "silent failures" where code executes but fails to achieve the visual goal. Experiments demonstrate that V-CAGE yields datasets with superior physical and semantic fidelity, significantly boosting the success rate and generalization of downstream policies compared to non-verified baselines. 

---
# Automated Rubrics for Reliable Evaluation of Medical Dialogue Systems 

**Authors**: Yinzhu Chen, Abdine Maiga, Hossein A. Rahmani, Emine Yilmaz  

**Link**: [PDF](https://arxiv.org/pdf/2601.15161)  

**Abstract**: Large Language Models (LLMs) are increasingly used for clinical decision support, where hallucinations and unsafe suggestions may pose direct risks to patient safety. These risks are particularly challenging as they often manifest as subtle clinical errors that evade detection by generic metrics, while expert-authored fine-grained rubrics remain costly to construct and difficult to scale. In this paper, we propose a retrieval-augmented multi-agent framework designed to automate the generation of instance-specific evaluation rubrics. Our approach grounds evaluation in authoritative medical evidence by decomposing retrieved content into atomic facts and synthesizing them with user interaction constraints to form verifiable, fine-grained evaluation criteria. Evaluated on HealthBench, our framework achieves a Clinical Intent Alignment (CIA) score of 60.12%, a statistically significant improvement over the GPT-4o baseline (55.16%). In discriminative tests, our rubrics yield a mean score delta ($\mu_{\Delta} = 8.658$) and an AUROC of 0.977, nearly doubling the quality separation achieved by GPT-4o baseline (4.972). Beyond evaluation, our rubrics effectively guide response refinement, improving quality by 9.2% (from 59.0% to 68.2%). This provides a scalable and transparent foundation for both evaluating and improving medical LLMs. The code is available at this https URL. 

---
# Outcome-Based RL Provably Leads Transformers to Reason, but Only With the Right Data 

**Authors**: Yuval Ran-Milo, Yotam Alexander, Shahar Mendel, Nadav Cohen  

**Link**: [PDF](https://arxiv.org/pdf/2601.15158)  

**Abstract**: Transformers trained via Reinforcement Learning (RL) with outcome-based supervision can spontaneously develop the ability to generate intermediate reasoning steps (Chain-of-Thought). Yet the mechanism by which sparse rewards drive gradient descent to discover such systematic reasoning remains poorly understood. We address this by analyzing the gradient flow dynamics of single-layer Transformers on a synthetic graph traversal task that cannot be solved without Chain-of-Thought (CoT) but admits a simple iterative solution. We prove that despite training solely on final-answer correctness, gradient flow drives the model to converge to a structured, interpretable algorithm that iteratively traverses the graph vertex-by-vertex. We characterize the distributional properties required for this emergence, identifying the critical role of "simple examples": instances requiring fewer reasoning steps. When the training distribution places sufficient mass on these simpler instances, the model learns a generalizable traversal strategy that extrapolates to longer chains; when this mass vanishes, gradient-based learning becomes infeasible. We corroborate our theoretical results through experiments on synthetic data and with real-world language models on mathematical reasoning tasks, validating that our theoretical findings carry over to practical settings. 

---
# Overcoming In-Memory Bottlenecks in Graph Foundation Models via Retrieval-Augmented Generation 

**Authors**: Haonan Yuan, Qingyun Sun, Jiacheng Tao, Xingcheng Fu, Jianxin Li  

**Link**: [PDF](https://arxiv.org/pdf/2601.15124)  

**Abstract**: Graph Foundation Models (GFMs) have emerged as a frontier in graph learning, which are expected to deliver transferable representations across diverse tasks. However, GFMs remain constrained by in-memory bottlenecks: they attempt to encode knowledge into model parameters, which limits semantic capacity, introduces heavy lossy compression with conflicts, and entangles graph representation with the knowledge in ways that hinder efficient adaptation, undermining scalability and interpretability. In this work,we propose RAG-GFM, a Retrieval-Augmented Generation aided Graph Foundation Model that offloads knowledge from parameters and complements parameterized learning. To externalize graph knowledge, we build a dual-modal unified retrieval module, where a semantic store from prefix-structured text and a structural store from centrality-based motif. To preserve heterogeneous information, we design a dual-view alignment objective that contrasts both modalities to capture both content and relational patterns. To enable efficient downstream adaptation, we perform in-context augmentation to enrich supporting instances with retrieved texts and motifs as contextual evidence. Extensive experiments on five benchmark graph datasets demonstrate that RAG-GFM consistently outperforms 13 state-of-the-art baselines in both cross-domain node and graph classification, achieving superior effectiveness and efficiency. 

---
# BREPS: Bounding-Box Robustness Evaluation of Promptable Segmentation 

**Authors**: Andrey Moskalenko, Danil Kuznetsov, Irina Dudko, Anastasiia Iasakova, Nikita Boldyrev, Denis Shepelev, Andrei Spiridonov, Andrey Kuznetsov, Vlad Shakhuro  

**Link**: [PDF](https://arxiv.org/pdf/2601.15123)  

**Abstract**: Promptable segmentation models such as SAM have established a powerful paradigm, enabling strong generalization to unseen objects and domains with minimal user input, including points, bounding boxes, and text prompts. Among these, bounding boxes stand out as particularly effective, often outperforming points while significantly reducing annotation costs. However, current training and evaluation protocols typically rely on synthetic prompts generated through simple heuristics, offering limited insight into real-world robustness. In this paper, we investigate the robustness of promptable segmentation models to natural variations in bounding box prompts. First, we conduct a controlled user study and collect thousands of real bounding box annotations. Our analysis reveals substantial variability in segmentation quality across users for the same model and instance, indicating that SAM-like models are highly sensitive to natural prompt noise. Then, since exhaustive testing of all possible user inputs is computationally prohibitive, we reformulate robustness evaluation as a white-box optimization problem over the bounding box prompt space. We introduce BREPS, a method for generating adversarial bounding boxes that minimize or maximize segmentation error while adhering to naturalness constraints. Finally, we benchmark state-of-the-art models across 10 datasets, spanning everyday scenes to medical imaging. Code - this https URL. 

---
# Auditing Language Model Unlearning via Information Decomposition 

**Authors**: Anmol Goel, Alan Ritter, Iryna Gurevych  

**Link**: [PDF](https://arxiv.org/pdf/2601.15111)  

**Abstract**: We expose a critical limitation in current approaches to machine unlearning in language models: despite the apparent success of unlearning algorithms, information about the forgotten data remains linearly decodable from internal representations. To systematically assess this discrepancy, we introduce an interpretable, information-theoretic framework for auditing unlearning using Partial Information Decomposition (PID). By comparing model representations before and after unlearning, we decompose the mutual information with the forgotten data into distinct components, formalizing the notions of unlearned and residual knowledge. Our analysis reveals that redundant information, shared across both models, constitutes residual knowledge that persists post-unlearning and correlates with susceptibility to known adversarial reconstruction attacks. Leveraging these insights, we propose a representation-based risk score that can guide abstention on sensitive inputs at inference time, providing a practical mechanism to mitigate privacy leakage. Our work introduces a principled, representation-level audit for unlearning, offering theoretical insight and actionable tools for safer deployment of language models. 

---
# An Agentic Operationalization of DISARM for FIMI Investigation on Social Media 

**Authors**: Kevin Tseng, Juan Carlos Toledano, Bart De Clerck, Yuliia Dukach, Phil Tinn  

**Link**: [PDF](https://arxiv.org/pdf/2601.15109)  

**Abstract**: The interoperability of data and intelligence across allied partners and their respective end-user groups is considered a foundational enabler to the collective defense capability--both conventional and hybrid--of NATO countries. Foreign Information Manipulation and Interference (FIMI) and related hybrid activities are conducted across various societal dimensions and infospheres, posing an ever greater challenge to the characterization of threats, sustaining situational awareness, and response coordination. Recent advances in AI have further led to the decreasing cost of AI-augmented trolling and interference activities, such as through the generation and amplification of manipulative content. Despite the introduction of the DISARM framework as a standardized metadata and analytical framework for FIMI, operationalizing it at the scale of social media remains a challenge. We propose a framework-agnostic agent-based operationalization of DISARM to investigate FIMI on social media. We develop a multi-agent pipeline in which specialized agentic AI components collaboratively (1) detect candidate manipulative behaviors, and (2) map these behaviors onto standard DISARM taxonomies in a transparent manner. We evaluated the approach on two real-world datasets annotated by domain practitioners. We demonstrate that our approach is effective in scaling the predominantly manual and heavily interpretive work of FIMI analysis, providing a direct contribution to enhancing the situational awareness and data interoperability in the context of operating in media and information-rich settings. 

---
# Memory Retention Is Not Enough to Master Memory Tasks in Reinforcement Learning 

**Authors**: Oleg Shchendrigin, Egor Cherepanov, Alexey K. Kovalev, Aleksandr I. Panov  

**Link**: [PDF](https://arxiv.org/pdf/2601.15086)  

**Abstract**: Effective decision-making in the real world depends on memory that is both stable and adaptive: environments change over time, and agents must retain relevant information over long horizons while also updating or overwriting outdated content when circumstances shift. Existing Reinforcement Learning (RL) benchmarks and memory-augmented agents focus primarily on retention, leaving the equally critical ability of memory rewriting largely unexplored. To address this gap, we introduce a benchmark that explicitly tests continual memory updating under partial observability, i.e. the natural setting where an agent must rely on memory rather than current observations, and use it to compare recurrent, transformer-based, and structured memory architectures. Our experiments reveal that classic recurrent models, despite their simplicity, demonstrate greater flexibility and robustness in memory rewriting tasks than modern structured memories, which succeed only under narrow conditions, and transformer-based agents, which often fail beyond trivial retention cases. These findings expose a fundamental limitation of current approaches and emphasize the necessity of memory mechanisms that balance stable retention with adaptive updating. Our work highlights this overlooked challenge, introduces benchmarks to evaluate it, and offers insights for designing future RL agents with explicit and trainable forgetting mechanisms. Code: this https URL 

---
# Multi-Agent Constraint Factorization Reveals Latent Invariant Solution Structure 

**Authors**: Christopher Scofield  

**Link**: [PDF](https://arxiv.org/pdf/2601.15077)  

**Abstract**: Multi-agent systems (MAS) composed of large language models often exhibit improved problem-solving performance despite operating on identical information. In this work, we provide a formal explanation for this phenomenon grounded in operator theory and constrained optimization. We model each agent as enforcing a distinct family of validity constraints on a shared solution state, and show that a MAS implements a factorized composition of constraint-enforcement operators. Under mild conditions, these dynamics converge to invariant solution sets defined by the intersection of agent constraint sets. Such invariant structures are generally not dynamically accessible to a single agent applying all constraints simultaneously, even when expressive capacity and information are identical. We extend this result from exact constraint enforcement to soft constraints via proximal operators, and apply the formalism to contemporary text-based dialog systems. 

---
# Incentive-Tuning: Understanding and Designing Incentives for Empirical Human-AI Decision-Making Studies 

**Authors**: Simran Kaur, Sara Salimzadeh, Ujwal Gadiraju  

**Link**: [PDF](https://arxiv.org/pdf/2601.15064)  

**Abstract**: AI has revolutionised decision-making across various fields. Yet human judgement remains paramount for high-stakes decision-making. This has fueled explorations of collaborative decision-making between humans and AI systems, aiming to leverage the strengths of both. To explore this dynamic, researchers conduct empirical studies, investigating how humans use AI assistance for decision-making and how this collaboration impacts results. A critical aspect of conducting these studies is the role of participants, often recruited through crowdsourcing platforms. The validity of these studies hinges on the behaviours of the participants, hence effective incentives that can potentially affect these behaviours are a key part of designing and executing these studies. In this work, we aim to address the critical role of incentive design for conducting empirical human-AI decision-making studies, focusing on understanding, designing, and documenting incentive schemes. Through a thematic review of existing research, we explored the current practices, challenges, and opportunities associated with incentive design for human-AI decision-making empirical studies. We identified recurring patterns, or themes, such as what comprises the components of an incentive scheme, how incentive schemes are manipulated by researchers, and the impact they can have on research outcomes. Leveraging the acquired understanding, we curated a set of guidelines to aid researchers in designing effective incentive schemes for their studies, called the Incentive-Tuning Framework, outlining how researchers can undertake, reflect on, and document the incentive design process. By advocating for a standardised yet flexible approach to incentive design and contributing valuable insights along with practical tools, we hope to pave the way for more reliable and generalizable knowledge in the field of human-AI decision-making. 

---
# Differential Privacy Image Generation with Reconstruction Loss and Noise Injection Using an Error Feedback SGD 

**Authors**: Qiwei Ma, Jun Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2601.15061)  

**Abstract**: Traditional data masking techniques such as anonymization cannot achieve the expected privacy protection while ensuring data utility for privacy-preserving machine learning. Synthetic data plays an increasingly important role as it generates a large number of training samples and prevents information leakage in real data. The existing methods suffer from the repeating trade-off processes between privacy and utility. We propose a novel framework for differential privacy generation, which employs an Error Feedback Stochastic Gradient Descent(EFSGD) method and introduces a reconstruction loss and noise injection mechanism into the training process. We generate images with higher quality and usability under the same privacy budget as the related work. Extensive experiments demonstrate the effectiveness and generalization of our proposed framework for both grayscale and RGB images. We achieve state-of-the-art results over almost all metrics on three benchmarks: MNIST, Fashion-MNIST, and CelebA. 

---
# Federated Transformer-GNN for Privacy-Preserving Brain Tumor Localization with Modality-Level Explainability 

**Authors**: Andrea Protani, Riccardo Taiello, Marc Molina Van Den Bosch, Luigi Serio  

**Link**: [PDF](https://arxiv.org/pdf/2601.15042)  

**Abstract**: Deep learning models for brain tumor analysis require large and diverse datasets that are often siloed across healthcare institutions due to privacy regulations. We present a federated learning framework for brain tumor localization that enables multi-institutional collaboration without sharing sensitive patient data. Our method extends a hybrid Transformer-Graph Neural Network architecture derived from prior decoder-free supervoxel GNNs and is deployed within CAFEIN\textsuperscript{\textregistered}, CERN's federated learning platform designed for healthcare environments. We provide an explainability analysis through Transformer attention mechanisms that reveals which MRI modalities drive the model predictions. Experiments on the BraTS dataset demonstrate a key finding: while isolated training on individual client data triggers early stopping well before reaching full training capacity, federated learning enables continued model improvement by leveraging distributed data, ultimately matching centralized performance. This result provides strong justification for federated learning when dealing with complex tasks and high-dimensional input data, as aggregating knowledge from multiple institutions significantly benefits the learning process. Our explainability analysis, validated through rigorous statistical testing on the full test set (paired t-tests with Bonferroni correction), reveals that deeper network layers significantly increase attention to T2 and FLAIR modalities ($p<0.001$, Cohen's $d$=1.50), aligning with clinical practice. 

---
# A Curriculum-Based Deep Reinforcement Learning Framework for the Electric Vehicle Routing Problem 

**Authors**: Mertcan Daysalilar, Fuat Uyguroglu, Gabriel Nicolosi, Adam Meyers  

**Link**: [PDF](https://arxiv.org/pdf/2601.15038)  

**Abstract**: The electric vehicle routing problem with time windows (EVRPTW) is a complex optimization problem in sustainable logistics, where routing decisions must minimize total travel distance, fleet size, and battery usage while satisfying strict customer time constraints. Although deep reinforcement learning (DRL) has shown great potential as an alternative to classical heuristics and exact solvers, existing DRL models often struggle to maintain training stability-failing to converge or generalize when constraints are dense. In this study, we propose a curriculum-based deep reinforcement learning (CB-DRL) framework designed to resolve this instability. The framework utilizes a structured three-phase curriculum that gradually increases problem complexity: the agent first learns distance and fleet optimization (Phase A), then battery management (Phase B), and finally the full EVRPTW (Phase C). To ensure stable learning across phases, the framework employs a modified proximal policy optimization algorithm with phase-specific hyperparameters, value and advantage clipping, and adaptive learning-rate scheduling. The policy network is built upon a heterogeneous graph attention encoder enhanced by global-local attention and feature-wise linear modulation. This specialized architecture explicitly captures the distinct properties of depots, customers, and charging stations. Trained exclusively on small instances with N=10 customers, the model demonstrates robust generalization to unseen instances ranging from N=5 to N=100, significantly outperforming standard baselines on medium-scale problems. Experimental results confirm that this curriculum-guided approach achieves high feasibility rates and competitive solution quality on out-of-distribution instances where standard DRL baselines fail, effectively bridging the gap between neural speed and operational reliability. 

---
# Knowledge Restoration-driven Prompt Optimization: Unlocking LLM Potential for Open-Domain Relational Triplet Extraction 

**Authors**: Xiaonan Jing, Gongqing Wu, Xingrui Zhuo, Lang Sun, Jiapu Wang  

**Link**: [PDF](https://arxiv.org/pdf/2601.15037)  

**Abstract**: Open-domain Relational Triplet Extraction (ORTE) is the foundation for mining structured knowledge without predefined schemas. Despite the impressive in-context learning capabilities of Large Language Models (LLMs), existing methods are hindered by their reliance on static, heuristic-driven prompting strategies. Due to the lack of reflection mechanisms required to internalize erroneous signals, these methods exhibit vulnerability in semantic ambiguity, often making erroneous extraction patterns permanent. To address this bottleneck, we propose a Knowledge Reconstruction-driven Prompt Optimization (KRPO) framework to assist LLMs in continuously improving their extraction capabilities for complex ORTE task flows. Specifically, we design a self-evaluation mechanism based on knowledge restoration, which provides intrinsic feedback signals by projecting structured triplets into semantic consistency scores. Subsequently, we propose a prompt optimizer based on a textual gradient that can internalize historical experiences to iteratively optimize prompts, which can better guide LLMs to handle subsequent extraction tasks. Furthermore, to alleviate relation redundancy, we design a relation canonicalization memory that collects representative relations and provides semantically distinct schemas for the triplets. Extensive experiments across three datasets show that KRPO significantly outperforms strong baselines in the extraction F1 score. 

---
# Visual and Cognitive Demands of a Large Language Model-Powered In-vehicle Conversational Agent 

**Authors**: Chris Monk, Allegra Ayala, Christine S.P. Yu, Gregory M. Fitch, Dara Gruber  

**Link**: [PDF](https://arxiv.org/pdf/2601.15034)  

**Abstract**: Driver distraction remains a leading contributor to motor vehicle crashes, necessitating rigorous evaluation of new in-vehicle technologies. This study assessed the visual and cognitive demands associated with an advanced Large Language Model (LLM) conversational agent (Gemini Live) during on-road driving, comparing it against handsfree phone calls, visual turn-by-turn guidance (low load baseline), and the Operation Span (OSPAN) task (high load anchor). Thirty-two licensed drivers completed five secondary tasks while visual and cognitive demands were measured using the Detection Response Task (DRT) for cognitive load, eye-tracking for visual attention, and subjective workload ratings. Results indicated that Gemini Live interactions (both single-turn and multi-turn) and hands-free phone calls shared similar levels of cognitive load, between that of visual turn-by-turn guidance and OSPAN. Exploratory analysis showed that cognitive load remained stable across extended multi-turn conversations. All tasks maintained mean glance durations well below the well-established 2-second safety threshold, confirming low visual demand. Furthermore, drivers consistently dedicated longer glances to the roadway between brief off-road glances toward the device during task completion, particularly during voice-based interactions, rendering longer total-eyes-off-road time findings less consequential. Subjective ratings mirrored objective data, with participants reporting low effort, demands, and perceived distraction for Gemini Live. These findings demonstrate that advanced LLM conversational agents, when implemented via voice interfaces, impose cognitive and visual demands comparable to established, low-risk hands-free benchmarks, supporting their safe deployment in the driving environment. 

---
# Obscuring Data Contamination Through Translation: Evidence from Arabic Corpora 

**Authors**: Chaymaa Abbas, Nour Shamaa, Mariette Awad  

**Link**: [PDF](https://arxiv.org/pdf/2601.14994)  

**Abstract**: Data contamination undermines the validity of Large Language Model evaluation by enabling models to rely on memorized benchmark content rather than true generalization. While prior work has proposed contamination detection methods, these approaches are largely limited to English benchmarks, leaving multilingual contamination poorly understood. In this work, we investigate contamination dynamics in multilingual settings by fine-tuning several open-weight LLMs on varying proportions of Arabic datasets and evaluating them on original English benchmarks. To detect memorization, we extend the Tested Slot Guessing method with a choice-reordering strategy and incorporate Min-K% probability analysis, capturing both behavioral and distributional contamination signals.
Our results show that translation into Arabic suppresses conventional contamination indicators, yet models still benefit from exposure to contaminated data, particularly those with stronger Arabic capabilities. This effect is consistently reflected in rising Mink% scores and increased cross-lingual answer consistency as contamination levels grow. To address this blind spot, we propose Translation-Aware Contamination Detection, which identifies contamination by comparing signals across multiple translated benchmark variants rather than English alone. The Translation-Aware Contamination Detection reliably exposes contamination even when English-only methods fail. Together, our findings highlight the need for multilingual, translation-aware evaluation pipelines to ensure fair, transparent, and reproducible assessment of LLMs. 

---
# Interoperable Architecture for Digital Identity Delegation for AI Agents with Blockchain Integration 

**Authors**: David Ricardo Saavedra  

**Link**: [PDF](https://arxiv.org/pdf/2601.14982)  

**Abstract**: Verifiable delegation in digital identity systems remains unresolved across centralized, federated, and self-sovereign identity (SSI) environments, particularly where both human users and autonomous AI agents must exercise and transfer authority without exposing primary credentials or private keys. We introduce a unified framework that enables bounded, auditable, and least-privilege delegation across heterogeneous identity ecosystems. The framework includes four key elements: Delegation Grants (DGs), first-class authorization artefacts that encode revocable transfers of authority with enforced scope reduction; a Canonical Verification Context (CVC) that normalizes verification requests into a single structured representation independent of protocols or credential formats; a layered reference architecture that separates trust anchoring, credential and proof validation, policy evaluation, and protocol mediation via a Trust Gateway; and an explicit treatment of blockchain anchoring as an optional integrity layer rather than a structural dependency. Together, these elements advance interoperable delegation and auditability and provide a foundation for future standardization, implementation, and integration of autonomous agents into trusted digital identity infrastructures. 

---
# HumanDiffusion: A Vision-Based Diffusion Trajectory Planner with Human-Conditioned Goals for Search and Rescue UAV 

**Authors**: Faryal Batool, Iana Zhura, Valerii Serpiva, Roohan Ahmed Khan, Ivan Valuev, Issatay Tokmurziyev, Dzmitry Tsetserukou  

**Link**: [PDF](https://arxiv.org/pdf/2601.14973)  

**Abstract**: Reliable human--robot collaboration in emergency scenarios requires autonomous systems that can detect humans, infer navigation goals, and operate safely in dynamic environments. This paper presents HumanDiffusion, a lightweight image-conditioned diffusion planner that generates human-aware navigation trajectories directly from RGB imagery. The system combines YOLO-11--based human detection with diffusion-driven trajectory generation, enabling a quadrotor to approach a target person and deliver medical assistance without relying on prior maps or computationally intensive planning pipelines. Trajectories are predicted in pixel space, ensuring smooth motion and a consistent safety margin around humans. We evaluate HumanDiffusion in simulation and real-world indoor mock-disaster scenarios. On a 300-sample test set, the model achieves a mean squared error of 0.02 in pixel-space trajectory reconstruction. Real-world experiments demonstrate an overall mission success rate of 80% across accident-response and search-and-locate tasks with partial occlusions. These results indicate that human-conditioned diffusion planning offers a practical and robust solution for human-aware UAV navigation in time-critical assistance settings. 

---
# InstructTime++: Time Series Classification with Multimodal Language Modeling via Implicit Feature Enhancement 

**Authors**: Mingyue Cheng, Xiaoyu Tao, Huajian Zhang, Qi Liu, Enhong Chen  

**Link**: [PDF](https://arxiv.org/pdf/2601.14968)  

**Abstract**: Most existing time series classification methods adopt a discriminative paradigm that maps input sequences directly to one-hot encoded class labels. While effective, this paradigm struggles to incorporate contextual features and fails to capture semantic relationships among classes. To address these limitations, we propose InstructTime, a novel framework that reformulates time series classification as a multimodal generative task. Specifically, continuous numerical sequences, contextual textual features, and task instructions are treated as multimodal inputs, while class labels are generated as textual outputs by tuned language models. To bridge the modality gap, InstructTime introduces a time series discretization module that converts continuous sequences into discrete temporal tokens, together with an alignment projection layer and a generative self-supervised pre-training strategy to enhance cross-modal representation alignment. Building upon this framework, we further propose InstructTime++, which extends InstructTime by incorporating implicit feature modeling to compensate for the limited inductive bias of language models. InstructTime++ leverages specialized toolkits to mine informative implicit patterns from raw time series and contextual inputs, including statistical feature extraction and vision-language-based image captioning, and translates them into textual descriptions for seamless integration. Extensive experiments on multiple benchmark datasets demonstrate the superior performance of InstructTime++. 

---
# A Comprehensive Benchmark of Language Models on Unicode and Romanized Sinhala 

**Authors**: Minuri Rajapakse, Ruvan Weerasinghe  

**Link**: [PDF](https://arxiv.org/pdf/2601.14958)  

**Abstract**: The performance of Language Models (LMs) on lower-resource, morphologically rich languages like Sinhala remains under-explored, particularly for Romanized Sinhala, which is prevalent in digital communication. This paper presents a comprehensive benchmark of modern LMs on a diverse corpus of Unicode and Romanized Sinhala. We evaluate open-source models using perplexity, a measure of how well a model predicts a text, and leading closed-source models via a qualitative analysis of sentence completion. Our findings reveal that the Mistral-Nemo-Base-2407 model achieves the strongest predictive performance on Unicode text and the Mistral-7B-v0.3 model for Romanized text. The results also highlight the strong all-around performance of the Llama-3.1-8B model for both scripts. Furthermore, a significant performance disparity exists among closed-source models: Gemini-1.5-pro and DeepSeek excel at Unicode generation, whereas Claude-3.5-Sonnet is superior at handling Romanized text. These results provide an essential guide for practitioners selecting models for Sinhala-specific applications and highlight the critical role of training data in handling script variations. 

---
# CorpusQA: A 10 Million Token Benchmark for Corpus-Level Analysis and Reasoning 

**Authors**: Zhiyuan Lu, Chenliang Li, Yingcheng Shi, Weizhou Shen, Ming Yan, Fei Huang  

**Link**: [PDF](https://arxiv.org/pdf/2601.14952)  

**Abstract**: While large language models now handle million-token contexts, their capacity for reasoning across entire document repositories remains largely untested. Existing benchmarks are inadequate, as they are mostly limited to single long texts or rely on a "sparse retrieval" assumption-that answers can be derived from a few relevant chunks. This assumption fails for true corpus-level analysis, where evidence is highly dispersed across hundreds of documents and answers require global integration, comparison, and statistical aggregation. To address this critical gap, we introduce CorpusQA, a new benchmark scaling up to 10 million tokens, generated via a novel data synthesis framework. By decoupling reasoning from textual representation, this framework creates complex, computation-intensive queries with programmatically guaranteed ground-truth answers, challenging systems to perform holistic reasoning over vast, unstructured text without relying on fallible human annotation. We further demonstrate the utility of our framework beyond evaluation, showing that fine-tuning on our synthesized data effectively enhances an LLM's general long-context reasoning capabilities. Extensive experiments reveal that even state-of-the-art long-context LLMs struggle as input length increases, and standard retrieval-augmented generation systems collapse entirely. Our findings indicate that memory-augmented agentic architectures offer a more robust alternative, suggesting a critical shift is needed from simply extending context windows to developing advanced architectures for global information synthesis. 

---
# TempViz: On the Evaluation of Temporal Knowledge in Text-to-Image Models 

**Authors**: Carolin Holtermann, Nina Krebs, Anne Lauscher  

**Link**: [PDF](https://arxiv.org/pdf/2601.14951)  

**Abstract**: Time alters the visual appearance of entities in our world, like objects, places, and animals. Thus, for accurately generating contextually-relevant images, knowledge and reasoning about time can be crucial (e.g., for generating a landscape in spring vs. in winter). Yet, although substantial work exists on understanding and improving temporal knowledge in natural language processing, research on how temporal phenomena appear and are handled in text-to-image (T2I) models remains scarce. We address this gap with TempViz, the first data set to holistically evaluate temporal knowledge in image generation, consisting of 7.9k prompts and more than 600 reference images. Using TempViz, we study the capabilities of five T2I models across five temporal knowledge categories. Human evaluation shows that temporal competence is generally weak, with no model exceeding 75% accuracy across categories. Towards larger-scale studies, we also examine automated evaluation methods, comparing several established approaches against human judgments. However, none of these approaches provides a reliable assessment of temporal cues - further indicating the pressing need for future research on temporal knowledge in T2I. 

---
# TIDAL: Temporally Interleaved Diffusion and Action Loop for High-Frequency VLA Control 

**Authors**: Yuteng Sun, Haoran Wang, Ruofei Bai, Zhengguo Li, Jun Li, Meng Yee, Chuah, Wei Yun Yau  

**Link**: [PDF](https://arxiv.org/pdf/2601.14945)  

**Abstract**: Large-scale Vision-Language-Action (VLA) models offer semantic generalization but suffer from high inference latency, limiting them to low-frequency batch-and-execute paradigm. This frequency mismatch creates an execution blind spot, causing failures in dynamic environments where targets move during the open-loop execution window. We propose TIDAL (Temporally Interleaved Diffusion and Action Loop), a hierarchical framework that decouples semantic reasoning from high-frequency actuation. TIDAL operates as a backbone-agnostic module for diffusion-based VLAs, using a dual-frequency architecture to redistribute the computational budget. Specifically, a low-frequency macro-intent loop caches semantic embeddings, while a high-frequency micro-control loop interleaves single-step flow integration with execution. This design enables approximately 9 Hz control updates on edge hardware (vs. approximately 2.4 Hz baselines) without increasing marginal overhead. To handle the resulting latency shift, we introduce a temporally misaligned training strategy where the policy learns predictive compensation using stale semantic intent alongside real-time proprioception. Additionally, we address the insensitivity of static vision encoders to velocity by incorporating a differential motion predictor. TIDAL is architectural, making it orthogonal to system-level optimizations. Experiments show a 2x performance gain over open-loop baselines in dynamic interception tasks. Despite a marginal regression in static success rates, our approach yields a 4x increase in feedback frequency and extends the effective horizon of semantic embeddings beyond the native action chunk size. Under non-paused inference protocols, TIDAL remains robust where standard baselines fail due to latency. 

---
# Generative Artificial Intelligence, Musical Heritage and the Construction of Peace Narratives: A Case Study in Mali 

**Authors**: Nouhoum Coulibaly, Ousmane Ly, Michael Leventhal, Ousmane Goro  

**Link**: [PDF](https://arxiv.org/pdf/2601.14931)  

**Abstract**: This study explores the capacity of generative artificial intelligence (Gen AI) to contribute to the construction of peace narratives and the revitalization of musical heritage in Mali. The study has been made in a political and social context where inter-community tensions and social fractures motivate a search for new symbolic frameworks for reconciliation. The study empirically explores three questions: (1) how Gen AI can be used as a tool for musical creation rooted in national languages and traditions; (2) to what extent Gen AI systems enable a balanced hybridization between technological innovation and cultural authenticity; and (3) how AI-assisted musical co-creation can strengthen social cohesion and cultural sovereignty. The experimental results suggest that Gen AI, embedded in a culturally conscious participatory framework, can act as a catalyst for symbolic diplomacy, amplifying local voices instead of standardizing them. However, challenges persist regarding the availability of linguistic corpora, algorithmic censorship, and the ethics of generating compositions derived from copyrighted sources. 

---
# Fast-ULCNet: A fast and ultra low complexity network for single-channel speech enhancement 

**Authors**: Nicolás Arrieta Larraza, Niels de Koeijer  

**Link**: [PDF](https://arxiv.org/pdf/2601.14925)  

**Abstract**: Single-channel speech enhancement algorithms are often used in resource-constrained embedded devices, where low latency and low complexity designs gain more importance. In recent years, researchers have proposed a wide variety of novel solutions to this problem. In particular, a recent deep learning model named ULCNet is among the state-of-the-art approaches in this domain. This paper proposes an adaptation of ULCNet, by replacing its GRU layers with FastGRNNs, to reduce both computational latency and complexity. Furthermore, this paper shows empirical evidence on the performance decay of FastGRNNs in long audio signals during inference due to internal state drifting, and proposes a novel approach based on a trainable complementary filter to mitigate it. The resulting model, Fast-ULCNet, performs on par with the state-of-the-art original ULCNet architecture on a speech enhancement task, while reducing its model size by more than half and decreasing its latency by 34% on average. 

---
# Vision-Language Models on the Edge for Real-Time Robotic Perception 

**Authors**: Sarat Ahmad, Maryam Hafeez, Syed Ali Raza Zaidi  

**Link**: [PDF](https://arxiv.org/pdf/2601.14921)  

**Abstract**: Vision-Language Models (VLMs) enable multimodal reasoning for robotic perception and interaction, but their deployment in real-world systems remains constrained by latency, limited onboard resources, and privacy risks of cloud offloading. Edge intelligence within 6G, particularly Open RAN and Multi-access Edge Computing (MEC), offers a pathway to address these challenges by bringing computation closer to the data source. This work investigates the deployment of VLMs on ORAN/MEC infrastructure using the Unitree G1 humanoid robot as an embodied testbed. We design a WebRTC-based pipeline that streams multimodal data to an edge node and evaluate LLaMA-3.2-11B-Vision-Instruct deployed at the edge versus in the cloud under real-time conditions. Our results show that edge deployment preserves near-cloud accuracy while reducing end-to-end latency by 5\%. We further evaluate Qwen2-VL-2B-Instruct, a compact model optimized for resource-constrained environments, which achieves sub-second responsiveness, cutting latency by more than half but at the cost of accuracy. 

---
# Tailoring Adverse Event Prediction in Type 1 Diabetes with Patient-Specific Deep Learning Models 

**Authors**: Giorgia Rigamonti, Mirko Paolo Barbato, Davide Marelli, Paolo Napoletano  

**Link**: [PDF](https://arxiv.org/pdf/2601.14917)  

**Abstract**: Effective management of Type 1 Diabetes requires continuous glucose monitoring and precise insulin adjustments to prevent hyperglycemia and hypoglycemia. With the growing adoption of wearable glucose monitors and mobile health applications, accurate blood glucose prediction is essential for enhancing automated insulin delivery and decision-support systems. This paper presents a deep learning-based approach for personalized blood glucose prediction, leveraging patient-specific data to improve prediction accuracy and responsiveness in real-world scenarios. Unlike traditional generalized models, our method accounts for individual variability, enabling more effective subject-specific predictions. We compare Leave-One-Subject-Out Cross-Validation with a fine-tuning strategy to evaluate their ability to model patient-specific dynamics. Results show that personalized models significantly improve the prediction of adverse events, enabling more precise and timely interventions in real-world scenarios. To assess the impact of patient-specific data, we conduct experiments comparing a multimodal, patient-specific approach against traditional CGM-only methods. Additionally, we perform an ablation study to investigate model performance with progressively smaller training sets, identifying the minimum data required for effective personalization-an essential consideration for real-world applications where extensive data collection is often challenging. Our findings underscore the potential of adaptive, personalized glucose prediction models for advancing next-generation diabetes management, particularly in wearable and mobile health platforms, enhancing consumer-oriented diabetes care solutions. 

---
# SpatialMem: Unified 3D Memory with Metric Anchoring and Fast Retrieval 

**Authors**: Xinyi Zheng, Yunze Liu, Chi-Hao Wu, Fan Zhang, Hao Zheng, Wenqi Zhou, Walterio W. Mayol-Cuevas, Junxiao Shen  

**Link**: [PDF](https://arxiv.org/pdf/2601.14895)  

**Abstract**: We present SpatialMem, a memory-centric system that unifies 3D geometry, semantics, and language into a single, queryable representation. Starting from casually captured egocentric RGB video, SpatialMem reconstructs metrically scaled indoor environments, detects structural 3D anchors (walls, doors, windows) as the first-layer scaffold, and populates a hierarchical memory with open-vocabulary object nodes -- linking evidence patches, visual embeddings, and two-layer textual descriptions to 3D coordinates -- for compact storage and fast retrieval. This design enables interpretable reasoning over spatial relations (e.g., distance, direction, visibility) and supports downstream tasks such as language-guided navigation and object retrieval without specialized sensors. Experiments across three real-life indoor scenes demonstrate that SpatialMem maintains strong anchor-description-level navigation completion and hierarchical retrieval accuracy under increasing clutter and occlusion, offering an efficient and extensible framework for embodied spatial intelligence. 

---
# What Makes Low-Bit Quantization-Aware Training Work for Reasoning LLMs? A Systematic Study 

**Authors**: Keyu Lv, Manyi Zhang, Xiaobo Xia, Jingchen Ni, Shannan Yan, Xianzhi Yu, Lu Hou, Chun Yuan, Haoli Bai  

**Link**: [PDF](https://arxiv.org/pdf/2601.14888)  

**Abstract**: Reasoning models excel at complex tasks such as coding and mathematics, yet their inference is often slow and token-inefficient. To improve the inference efficiency, post-training quantization (PTQ) usually comes with the cost of large accuracy drops, especially for reasoning tasks under low-bit settings. In this study, we present a systematic empirical study of quantization-aware training (QAT) for reasoning models. Our key findings include: (1) Knowledge distillation is a robust objective for reasoning models trained via either supervised fine-tuning or reinforcement learning; (2) PTQ provides a strong initialization for QAT, improving accuracy while reducing training cost; (3) Reinforcement learning remains feasible for quantized models given a viable cold start and yields additional gains; and (4) Aligning the PTQ calibration domain with the QAT training domain accelerates convergence and often improves the final accuracy. Finally, we consolidate these findings into an optimized workflow (Reasoning-QAT), and show that it consistently outperforms state-of-the-art PTQ methods across multiple LLM backbones and reasoning datasets. For instance, on Qwen3-0.6B, it surpasses GPTQ by 44.53% on MATH-500 and consistently recovers performance in the 2-bit regime. 

---
# GAT-NeRF: Geometry-Aware-Transformer Enhanced Neural Radiance Fields for High-Fidelity 4D Facial Avatars 

**Authors**: Zhe Chang, Haodong Jin, Ying Sun, Yan Song, Hui Yu  

**Link**: [PDF](https://arxiv.org/pdf/2601.14875)  

**Abstract**: High-fidelity 4D dynamic facial avatar reconstruction from monocular video is a critical yet challenging task, driven by increasing demands for immersive virtual human applications. While Neural Radiance Fields (NeRF) have advanced scene representation, their capacity to capture high-frequency facial details, such as dynamic wrinkles and subtle textures from information-constrained monocular streams, requires significant enhancement. To tackle this challenge, we propose a novel hybrid neural radiance field framework, called Geometry-Aware-Transformer Enhanced NeRF (GAT-NeRF) for high-fidelity and controllable 4D facial avatar reconstruction, which integrates the Transformer mechanism into the NeRF pipeline. GAT-NeRF synergistically combines a coordinate-aligned Multilayer Perceptron (MLP) with a lightweight Transformer module, termed as Geometry-Aware-Transformer (GAT) due to its processing of multi-modal inputs containing explicit geometric priors. The GAT module is enabled by fusing multi-modal input features, including 3D spatial coordinates, 3D Morphable Model (3DMM) expression parameters, and learnable latent codes to effectively learn and enhance feature representations pertinent to fine-grained geometry. The Transformer's effective feature learning capabilities are leveraged to significantly augment the modeling of complex local facial patterns like dynamic wrinkles and acne scars. Comprehensive experiments unequivocally demonstrate GAT-NeRF's state-of-the-art performance in visual fidelity and high-frequency detail recovery, forging new pathways for creating realistic dynamic digital humans for multimedia applications. 

---
# From Observation to Prediction: LSTM for Vehicle Lane Change Forecasting on Highway On/Off-Ramps 

**Authors**: Mohamed Abouras, Catherine M. Elias  

**Link**: [PDF](https://arxiv.org/pdf/2601.14848)  

**Abstract**: On and off-ramps are understudied road sections even though they introduce a higher level of variation in highway interactions. Predicting vehicles' behavior in these areas can decrease the impact of uncertainty and increase road safety. In this paper, the difference between this Area of Interest (AoI) and a straight highway section is studied. Multi-layered LSTM architecture to train the AoI model with ExiD drone dataset is utilized. In the process, different prediction horizons and different models' workflow are tested. The results show great promise on horizons up to 4 seconds with prediction accuracy starting from about 76% for the AoI and 94% for the general highway scenarios on the maximum horizon. 

---
# CAG-Avatar: Cross-Attention Guided Gaussian Avatars for High-Fidelity Head Reconstruction 

**Authors**: Zhe Chang, Haodong Jin, Yan Song, Hui Yu  

**Link**: [PDF](https://arxiv.org/pdf/2601.14844)  

**Abstract**: Creating high-fidelity, real-time drivable 3D head avatars is a core challenge in digital animation. While 3D Gaussian Splashing (3D-GS) offers unprecedented rendering speed and quality, current animation techniques often rely on a "one-size-fits-all" global tuning approach, where all Gaussian primitives are uniformly driven by a single expression code. This simplistic approach fails to unravel the distinct dynamics of different facial regions, such as deformable skin versus rigid teeth, leading to significant blurring and distortion artifacts. We introduce Conditionally-Adaptive Gaussian Avatars (CAG-Avatar), a framework that resolves this key limitation. At its core is a Conditionally Adaptive Fusion Module built on cross-attention. This mechanism empowers each 3D Gaussian to act as a query, adaptively extracting relevant driving signals from the global expression code based on its canonical position. This "tailor-made" conditioning strategy drastically enhances the modeling of fine-grained, localized dynamics. Our experiments confirm a significant improvement in reconstruction fidelity, particularly for challenging regions such as teeth, while preserving real-time rendering performance. 

---
# Multimodal system for skin cancer detection 

**Authors**: Volodymyr Sydorskyi, Igor Krashenyi, Oleksii Yakubenko  

**Link**: [PDF](https://arxiv.org/pdf/2601.14822)  

**Abstract**: Melanoma detection is vital for early diagnosis and effective treatment. While deep learning models on dermoscopic images have shown promise, they require specialized equipment, limiting their use in broader clinical settings. This study introduces a multi-modal melanoma detection system using conventional photo images, making it more accessible and versatile. Our system integrates image data with tabular metadata, such as patient demographics and lesion characteristics, to improve detection accuracy. It employs a multi-modal neural network combining image and metadata processing and supports a two-step model for cases with or without metadata. A three-stage pipeline further refines predictions by boosting algorithms and enhancing performance. To address the challenges of a highly imbalanced dataset, specific techniques were implemented to ensure robust training. An ablation study evaluated recent vision architectures, boosting algorithms, and loss functions, achieving a peak Partial ROC AUC of 0.18068 (0.2 maximum) and top-15 retrieval sensitivity of 0.78371. Results demonstrate that integrating photo images with metadata in a structured, multi-stage pipeline yields significant performance improvements. This system advances melanoma detection by providing a scalable, equipment-independent solution suitable for diverse healthcare environments, bridging the gap between specialized and general clinical practices. 

---
# Training-Efficient Text-to-Music Generation with State-Space Modeling 

**Authors**: Wei-Jaw Lee, Fang-Chih Hsieh, Xuanjun Chen, Fang-Duo Tsai, Yi-Hsuan Yang  

**Link**: [PDF](https://arxiv.org/pdf/2601.14786)  

**Abstract**: Recent advances in text-to-music generation (TTM) have yielded high-quality results, but often at the cost of extensive compute and the use of large proprietary internal data. To improve the affordability and openness of TTM training, an open-source generative model backbone that is more training- and data-efficient is needed. In this paper, we constrain the number of trainable parameters in the generative model to match that of the MusicGen-small benchmark (with about 300M parameters), and replace its Transformer backbone with the emerging class of state-space models (SSMs). Specifically, we explore different SSM variants for sequence modeling, and compare a single-stage SSM-based design with a decomposable two-stage SSM/diffusion hybrid design. All proposed models are trained from scratch on a purely public dataset comprising 457 hours of CC-licensed music, ensuring full openness. Our experimental findings are three-fold. First, we show that SSMs exhibit superior training efficiency compared to the Transformer counterpart. Second, despite using only 9% of the FLOPs and 2% of the training data size compared to the MusicGen-small benchmark, our model achieves competitive performance in both objective metrics and subjective listening tests based on MusicCaps captions. Finally, our scaling-down experiment demonstrates that SSMs can maintain competitive performance relative to the Transformer baseline even at the same training budget (measured in iterations), when the model size is reduced to four times smaller. To facilitate the democratization of TTM research, the processed captions, model checkpoints, and source code are available on GitHub via the project page: this https URL. 

---
# RECAP: Resistance Capture in Text-based Mental Health Counseling with Large Language Models 

**Authors**: Anqi Li, Yuqian Chen, Yu Lu, Zhaoming Chen, Yuan Xie, Zhenzhong Lan  

**Link**: [PDF](https://arxiv.org/pdf/2601.14780)  

**Abstract**: Recognizing and navigating client resistance is critical for effective mental health counseling, yet detecting such behaviors is particularly challenging in text-based interactions. Existing NLP approaches oversimplify resistance categories, ignore the sequential dynamics of therapeutic interventions, and offer limited interpretability.
To address these limitations, we propose PsyFIRE, a theoretically grounded framework capturing 13 fine-grained resistance behaviors alongside collaborative interactions. Based on PsyFIRE, we construct the ClientResistance corpus with 23,930 annotated utterances from real-world Chinese text-based counseling, each supported by context-specific rationales. Leveraging this dataset, we develop RECAP, a two-stage framework that detects resistance and fine-grained resistance types with explanations.
RECAP achieves 91.25% F1 for distinguishing collaboration and resistance and 66.58% macro-F1 for fine-grained resistance categories classification, outperforming leading prompt-based LLM baselines by over 20 points. Applied to a separate counseling dataset and a pilot study with 62 counselors, RECAP reveals the prevalence of resistance, its negative impact on therapeutic relationships and demonstrates its potential to improve counselors' understanding and intervention strategies. 

---
# FunCineForge: A Unified Dataset Toolkit and Model for Zero-Shot Movie Dubbing in Diverse Cinematic Scenes 

**Authors**: Jiaxuan Liu, Yang Xiang, Han Zhao, Xiangang Li, Zhenhua Ling  

**Link**: [PDF](https://arxiv.org/pdf/2601.14777)  

**Abstract**: Movie dubbing is the task of synthesizing speech from scripts conditioned on video scenes, requiring accurate lip sync, faithful timbre transfer, and proper modeling of character identity and emotion. However, existing methods face two major limitations: (1) high-quality multimodal dubbing datasets are limited in scale, suffer from high word error rates, contain sparse annotations, rely on costly manual labeling, and are restricted to monologue scenes, all of which hinder effective model training; (2) existing dubbing models rely solely on the lip region to learn audio-visual alignment, which limits their applicability to complex live-action cinematic scenes, and exhibit suboptimal performance in lip sync, speech quality, and emotional expressiveness. To address these issues, we propose FunCineForge, which comprises an end-to-end production pipeline for large-scale dubbing datasets and an MLLM-based dubbing model designed for diverse cinematic scenes. Using the pipeline, we construct the first Chinese television dubbing dataset with rich annotations, and demonstrate the high quality of these data. Experiments across monologue, narration, dialogue, and multi-speaker scenes show that our dubbing model consistently outperforms SOTA methods in audio quality, lip sync, timbre transfer, and instruction following. Code and demos are available at this https URL. 

---
# Anytime Optimal Decision Tree Learning with Continuous Features 

**Authors**: Harold Kiossou, Pierre Schaus, Siegfried Nijssen  

**Link**: [PDF](https://arxiv.org/pdf/2601.14765)  

**Abstract**: In recent years, significant progress has been made on algorithms for learning optimal decision trees, primarily in the context of binary features. Extending these methods to continuous features remains substantially more challenging due to the large number of potential splits for each feature. Recently, an elegant exact algorithm was proposed for learning optimal decision trees with continuous features; however, the rapidly increasing computational time limits its practical applicability to shallow depths (typically 3 or 4). It relies on a depth-first search optimization strategy that fully optimizes the left subtree of each split before exploring the corresponding right subtree. While effective in finding optimal solutions given sufficient time, this strategy can lead to poor anytime behavior: when interrupted early, the best-found tree is often highly unbalanced and suboptimal. In such cases, purely greedy methods such as C4.5 may, paradoxically, yield better solutions. To address this limitation, we propose an anytime, yet complete approach leveraging limited discrepancy search, distributing the computational effort more evenly across the entire tree structure, and thus ensuring that a high-quality decision tree is available at any interruption point. Experimental results show that our approach outperforms the existing one in terms of anytime performance. 

---
# Mechanism Shift During Post-training from Autoregressive to Masked Diffusion Language Models 

**Authors**: Injin Kong, Hyoungjoon Lee, Yohan Jo  

**Link**: [PDF](https://arxiv.org/pdf/2601.14758)  

**Abstract**: Post-training pretrained Autoregressive models (ARMs) into Masked Diffusion models (MDMs) has emerged as a cost-effective strategy to overcome the limitations of sequential generation. However, the internal algorithmic transformations induced by this paradigm shift remain unexplored, leaving it unclear whether post-trained MDMs acquire genuine bidirectional reasoning capabilities or merely repackage autoregressive heuristics. In this work, we address this question by conducting a comparative circuit analysis of ARMs and their MDM counterparts. Our analysis reveals a systematic "mechanism shift" dependent on the structural nature of the task. Structurally, we observe a distinct divergence: while MDMs largely retain autoregressive circuitry for tasks dominated by local causal dependencies, they abandon initialized pathways for global planning tasks, exhibiting distinct rewiring characterized by increased early-layer processing. Semantically, we identify a transition from sharp, localized specialization in ARMs to distributed integration in MDMs. Through these findings, we conclude that diffusion post-training does not merely adapt model parameters but fundamentally reorganizes internal computation to support non-sequential global planning. 

---
# FSX: Message Flow Sensitivity Enhanced Structural Explainer for Graph Neural Networks 

**Authors**: Bizu Feng, Zhimu Yang, Shaode Yu, Zixin Hu  

**Link**: [PDF](https://arxiv.org/pdf/2601.14730)  

**Abstract**: Despite the widespread success of Graph Neural Networks (GNNs), understanding the reasons behind their specific predictions remains challenging. Existing explainability methods face a trade-off that gradient-based approaches are computationally efficient but often ignore structural interactions, while game-theoretic techniques capture interactions at the cost of high computational overhead and potential deviation from the model's true reasoning path. To address this gap, we propose FSX (Message Flow Sensitivity Enhanced Structural Explainer), a novel hybrid framework that synergistically combines the internal message flows of the model with a cooperative game approach applied to the external graph data. FSX first identifies critical message flows via a novel flow-sensitivity analysis: during a single forward pass, it simulates localized node perturbations and measures the resulting changes in message flow intensities. These sensitivity-ranked flows are then projected onto the input graph to define compact, semantically meaningful subgraphs. Within each subgraph, a flow-aware cooperative game is conducted, where node contributions are evaluated fairly through a Shapley-like value that incorporates both node-feature importance and their roles in sustaining or destabilizing the identified critical flows. Extensive evaluation across multiple datasets and GNN architectures demonstrates that FSX achieves superior explanation fidelity with significantly reduced runtime, while providing unprecedented insights into the structural logic underlying model predictions--specifically, how important sub-structures exert influence by governing the stability of key internal computational pathways. 

---
# AQAScore: Evaluating Semantic Alignment in Text-to-Audio Generation via Audio Question Answering 

**Authors**: Chun-Yi Kuan, Kai-Wei Chang, Hung-yi Lee  

**Link**: [PDF](https://arxiv.org/pdf/2601.14728)  

**Abstract**: Although text-to-audio generation has made remarkable progress in realism and diversity, the development of evaluation metrics has not kept pace. Widely-adopted approaches, typically based on embedding similarity like CLAPScore, effectively measure general relevance but remain limited in fine-grained semantic alignment and compositional reasoning. To address this, we introduce AQAScore, a backbone-agnostic evaluation framework that leverages the reasoning capabilities of audio-aware large language models (ALLMs). AQAScore reformulates assessment as a probabilistic semantic verification task; rather than relying on open-ended text generation, it estimates alignment by computing the exact log-probability of a "Yes" answer to targeted semantic queries. We evaluate AQAScore across multiple benchmarks, including human-rated relevance, pairwise comparison, and compositional reasoning tasks. Experimental results show that AQAScore consistently achieves higher correlation with human judgments than similarity-based metrics and generative prompting baselines, showing its effectiveness in capturing subtle semantic inconsistencies and scaling with the capability of underlying ALLMs. 

---
# HERMES: KV Cache as Hierarchical Memory for Efficient Streaming Video Understanding 

**Authors**: Haowei Zhang, Shudong Yang, Jinlan Fu, See-Kiong Ng, Xipeng Qiu  

**Link**: [PDF](https://arxiv.org/pdf/2601.14724)  

**Abstract**: Recent advancements in Multimodal Large Language Models (MLLMs) have demonstrated significant improvement in offline video understanding. However, extending these capabilities to streaming video inputs, remains challenging, as existing models struggle to simultaneously maintain stable understanding performance, real-time responses, and low GPU memory overhead. To address this challenge, we propose HERMES, a novel training-free architecture for real-time and accurate understanding of video streams. Based on a mechanistic attention investigation, we conceptualize KV cache as a hierarchical memory framework that encapsulates video information across multiple granularities. During inference, HERMES reuses a compact KV cache, enabling efficient streaming understanding under resource constraints. Notably, HERMES requires no auxiliary computations upon the arrival of user queries, thereby guaranteeing real-time responses for continuous video stream interactions, which achieves 10$\times$ faster TTFT compared to prior SOTA. Even when reducing video tokens by up to 68% compared with uniform sampling, HERMES achieves superior or comparable accuracy across all benchmarks, with up to 11.4% gains on streaming datasets. 

---
# PCL-Reasoner-V1.5: Advancing Math Reasoning with Offline Reinforcement Learning 

**Authors**: Yao Lu, Dengdong Fan, Jianzheng Nie, Fan Xu, Jie Chen, Bin Zhou, Yonghong Tian  

**Link**: [PDF](https://arxiv.org/pdf/2601.14716)  

**Abstract**: We present PCL-Reasoner-V1.5, a 32-billion-parameter large language model (LLM) for mathematical reasoning. The model is built upon Qwen2.5-32B and refined via supervised fine-tuning (SFT) followed by reinforcement learning (RL). A central innovation is our proposed offline RL method, which provides superior training stability and efficiency over standard online RL methods such as GRPO. Our model achieves state-of-the-art performance among models post-trained on Qwen2.5-32B, attaining average accuracies of 90.9% on AIME 2024 and 85.6% on AIME 2025. Our work demonstrates offline RL as a stable and efficient paradigm for advancing reasoning in LLMs. All experiments were conducted on Huawei Ascend 910C NPUs. 

---
# Adaptive Fidelity Estimation for Quantum Programs with Graph-Guided Noise Awareness 

**Authors**: Tingting Li, Ziming Zhao, Jianwei Yin  

**Link**: [PDF](https://arxiv.org/pdf/2601.14713)  

**Abstract**: Fidelity estimation is a critical yet resource-intensive step in testing quantum programs on noisy intermediate-scale quantum (NISQ) devices, where the required number of measurements is difficult to predefine due to hardware noise, device heterogeneity, and transpilation-induced circuit transformations. We present QuFid, an adaptive and noise-aware framework that determines measurement budgets online by leveraging circuit structure and runtime statistical feedback. QuFid models a quantum program as a directed acyclic graph (DAG) and employs a control-flow-aware random walk to characterize noise propagation along gate dependencies. Backend-specific effects are captured via transpilation-induced structural deformation metrics, which are integrated into the random-walk formulation to induce a noise-propagation operator. Circuit complexity is then quantified through the spectral characteristics of this operator, providing a principled and lightweight basis for adaptive measurement planning. Experiments on 18 quantum benchmarks executed on IBM Quantum backends show that QuFid significantly reduces measurement cost compared to fixed-shot and learning-based baselines, while consistently maintaining acceptable fidelity bias. 

---
# Case-Guided Sequential Assay Planning in Drug Discovery 

**Authors**: Tianchi Chen, Jan Bima, Sean L. Wu, Otto Ritter, Bingjia Yang, Xiang Yu  

**Link**: [PDF](https://arxiv.org/pdf/2601.14710)  

**Abstract**: Optimally sequencing experimental assays in drug discovery is a high-stakes planning problem under severe uncertainty and resource constraints. A primary obstacle for standard reinforcement learning (RL) is the absence of an explicit environment simulator or transition data $(s, a, s')$; planning must rely solely on a static database of historical outcomes. We introduce the Implicit Bayesian Markov Decision Process (IBMDP), a model-based RL framework designed for such simulator-free settings. IBMDP constructs a case-guided implicit model of transition dynamics by forming a nonparametric belief distribution using similar historical outcomes. This mechanism enables Bayesian belief updating as evidence accumulates and employs ensemble MCTS planning to generate stable policies that balance information gain toward desired outcomes with resource efficiency. We validate IBMDP through comprehensive experiments. On a real-world central nervous system (CNS) drug discovery task, IBMDP reduced resource consumption by up to 92\% compared to established heuristics while maintaining decision confidence. To rigorously assess decision quality, we also benchmarked IBMDP in a synthetic environment with a computable optimal policy. Our framework achieves significantly higher alignment with this optimal policy than a deterministic value iteration alternative that uses the same similarity-based model, demonstrating the superiority of our ensemble planner. IBMDP offers a practical solution for sequential experimental design in data-rich but simulator-poor domains. 

---
# Proximal Policy Optimization with Evolutionary Mutations 

**Authors**: Casimir Czworkowski, Stephen Hornish, Alhassan S. Yasin  

**Link**: [PDF](https://arxiv.org/pdf/2601.14705)  

**Abstract**: Proximal Policy Optimization (PPO) is a widely used reinforcement learning algorithm known for its stability and sample efficiency, but it often suffers from premature convergence due to limited exploration. In this paper, we propose POEM (Proximal Policy Optimization with Evolutionary Mutations), a novel modification to PPO that introduces an adaptive exploration mechanism inspired by evolutionary algorithms. POEM enhances policy diversity by monitoring the Kullback-Leibler (KL) divergence between the current policy and a moving average of previous policies. When policy changes become minimal, indicating stagnation, POEM triggers an adaptive mutation of policy parameters to promote exploration. We evaluate POEM on four OpenAI Gym environments: CarRacing, MountainCar, BipedalWalker, and LunarLander. Through extensive fine-tuning using Bayesian optimization techniques and statistical testing using Welch's t-test, we find that POEM significantly outperforms PPO on three of the four tasks (BipedalWalker: t=-2.0642, p=0.0495; CarRacing: t=-6.3987, p=0.0002; MountainCar: t=-6.2431, p<0.0001), while performance on LunarLander is not statistically significant (t=-1.8707, p=0.0778). Our results highlight the potential of integrating evolutionary principles into policy gradient methods to overcome exploration-exploitation tradeoffs. 

---
# When Text-as-Vision Meets Semantic IDs in Generative Recommendation: An Empirical Study 

**Authors**: Shutong Qiao, Wei Yuan, Tong Chen, Xiangyu Zhao, Quoc Viet Hung Nguyen, Hongzhi Yin  

**Link**: [PDF](https://arxiv.org/pdf/2601.14697)  

**Abstract**: Semantic ID learning is a key interface in Generative Recommendation (GR) models, mapping items to discrete identifiers grounded in side information, most commonly via a pretrained text encoder. However, these text encoders are primarily optimized for well-formed natural language. In real-world recommendation data, item descriptions are often symbolic and attribute-centric, containing numerals, units, and abbreviations. These text encoders can break these signals into fragmented tokens, weakening semantic coherence and distorting relationships among attributes. Worse still, when moving to multimodal GR, relying on standard text encoders introduces an additional obstacle: text and image embeddings often exhibit mismatched geometric structures, making cross-modal fusion less effective and less stable.
In this paper, we revisit representation design for Semantic ID learning by treating text as a visual signal. We conduct a systematic empirical study of OCR-based text representations, obtained by rendering item descriptions into images and encoding them with vision-based OCR models. Experiments across four datasets and two generative backbones show that OCR-text consistently matches or surpasses standard text embeddings for Semantic ID learning in both unimodal and multimodal settings. Furthermore, we find that OCR-based Semantic IDs remain robust under extreme spatial-resolution compression, indicating strong robustness and efficiency in practical deployments. 

---
# CoScale-RL: Efficient Post-Training by Co-Scaling Data and Computation 

**Authors**: Yutong Chen, Jiandong Gao, Ji Wu  

**Link**: [PDF](https://arxiv.org/pdf/2601.14695)  

**Abstract**: Training Large Reasoning Model (LRM) is usually unstable and unpredictable, especially on hard problems or weak foundation models. We found that the current post-training scaling strategy can still improve on these cases. We propose CoScale-RL, a novel scaling strategy with better data and computational efficiency. We first scale up solutions to make problems solvable. The core idea is to collect multiple solutions for each problem, rather than simply enlarging the dataset. Then, we scale up rollout computation to stabilize Reinforcement Learning. We further leverage a model merge technique called Re-distillation to sustain or even improve computational efficiency when scaling up. Our method significantly improves data and computational efficiency, with an average 3.76$\times$ accuracy improvement on four benchmarks. CoScale-RL is able to improve an LRM's ability boundary without an extensive SFT dataset. Our method provides a new scaling direction to further improve LRM's reasoning ability. 

---
# Re-understanding Graph Unlearning through Memorization 

**Authors**: Pengfei Ding, Yan Wang, Guanfeng Liu  

**Link**: [PDF](https://arxiv.org/pdf/2601.14694)  

**Abstract**: Graph unlearning (GU), which removes nodes, edges, or features from trained graph neural networks (GNNs), is crucial in Web applications where graph data may contain sensitive, mislabeled, or malicious information. However, existing GU methods lack a clear understanding of the key factors that determine unlearning effectiveness, leading to three fundamental limitations: (1) impractical and inaccurate GU difficulty assessment due to test-access requirements and invalid assumptions, (2) ineffectiveness on hard-to-unlearn tasks, and (3) misaligned evaluation protocols that overemphasize easy tasks and fail to capture true forgetting capability. To address these issues, we establish GNN memorization as a new perspective for understanding graph unlearning and propose MGU, a Memorization-guided Graph Unlearning framework. MGU achieves three key advances: it provides accurate and practical difficulty assessment across different GU tasks, develops an adaptive strategy that dynamically adjusts unlearning objectives based on difficulty levels, and establishes a comprehensive evaluation protocol that aligns with practical requirements. Extensive experiments on ten real-world graphs demonstrate that MGU consistently outperforms state-of-the-art baselines in forgetting quality, computational efficiency, and utility preservation. 

---
# Beyond Error-Based Optimization: Experience-Driven Symbolic Regression with Goal-Conditioned Reinforcement Learning 

**Authors**: Jianwen Sun, Xinrui Li, Fuqing Li, Xiaoxuan Shen  

**Link**: [PDF](https://arxiv.org/pdf/2601.14693)  

**Abstract**: Symbolic Regression aims to automatically identify compact and interpretable mathematical expressions that model the functional relationship between input and output variables. Most existing search-based symbolic regression methods typically rely on the fitting error to inform the search process. However, in the vast expression space, numerous candidate expressions may exhibit similar error values while differing substantially in structure, leading to ambiguous search directions and hindering convergence to the underlying true function. To address this challenge, we propose a novel framework named EGRL-SR (Experience-driven Goal-conditioned Reinforcement Learning for Symbolic Regression). In contrast to traditional error-driven approaches, EGRL-SR introduces a new perspective: leveraging precise historical trajectories and optimizing the action-value network to proactively guide the search process, thereby achieving a more robust expression search. Specifically, we formulate symbolic regression as a goal-conditioned reinforcement learning problem and incorporate hindsight experience replay, allowing the action-value network to generalize common mapping patterns from diverse input-output pairs. Moreover, we design an all-point satisfaction binary reward function that encourages the action-value network to focus on structural patterns rather than low-error expressions, and concurrently propose a structure-guided heuristic exploration strategy to enhance search diversity and space coverage. Experiments on public benchmarks show that EGRL-SR consistently outperforms state-of-the-art methods in recovery rate and robustness, and can recover more complex expressions under the same search budget. Ablation results validate that the action-value network effectively guides the search, with both the reward function and the exploration strategy playing critical roles. 

---
# HCVR Scene Generation: High Compatibility Virtual Reality Environment Generation for Extended Redirected Walking 

**Authors**: Yiran Zhang, Xingpeng Sun, Aniket Bera  

**Link**: [PDF](https://arxiv.org/pdf/2601.14679)  

**Abstract**: Natural walking enhances immersion in virtual environments (VEs), but physical space limitations and obstacles hinder exploration, especially in large virtual scenes. Redirected Walking (RDW) techniques mitigate this by subtly manipulating the virtual camera to guide users away from physical collisions within pre-defined VEs. However, RDW efficacy diminishes significantly when substantial geometric divergence exists between the physical and virtual environments, leading to unavoidable collisions. Existing scene generation methods primarily focus on object relationships or layout aesthetics, often neglecting the crucial aspect of physical compatibility required for effective RDW. To address this, we introduce HCVR (High Compatibility Virtual Reality Environment Generation), a novel framework that generates virtual scenes inherently optimized for alignment-based RDW controllers. HCVR first employs ENI++, a novel, boundary-sensitive metric to evaluate the incompatibility between physical and virtual spaces by comparing rotation-sensitive visibility polygons. Guided by the ENI++ compatibility map and user prompts, HCVR utilizes a Large Language Model (LLM) for context-aware 3D asset retrieval and initial layout generation. The framework then strategically adjusts object selection, scaling, and placement to maximize coverage of virtually incompatible regions, effectively guiding users towards RDW-feasible paths. User studies evaluating physical collisions and layout quality demonstrate HCVR's effectiveness with HCVR-generated scenes, resulting in 22.78 times fewer physical collisions and received 35.89\% less on ENI++ score compared to LLM-based generation with RDW, while also receiving 12.5\% higher scores on user feedback to layout design. 

---
# Transfer Learning from One Cancer to Another via Deep Learning Domain Adaptation 

**Authors**: Justin Cheung, Samuel Savine, Calvin Nguyen, Lin Lu, Alhassan S. Yasin  

**Link**: [PDF](https://arxiv.org/pdf/2601.14678)  

**Abstract**: Supervised deep learning models often achieve excellent performance within their training distribution but struggle to generalize beyond it. In cancer histopathology, for example, a convolutional neural network (CNN) may classify cancer severity accurately for cancer types represented in its training data, yet fail on related but unseen types. Although adenocarcinomas from different organs share morphological features that might support limited cross-domain generalization, addressing domain shift directly is necessary for robust performance. Domain adaptation offers a way to transfer knowledge from labeled data in one cancer type to unlabeled data in another, helping mitigate the scarcity of annotated medical images.
This work evaluates cross-domain classification performance among lung, colon, breast, and kidney adenocarcinomas. A ResNet50 trained on any single adenocarcinoma achieves over 98% accuracy on its own domain but shows minimal generalization to others. Ensembling multiple supervised models does not resolve this limitation. In contrast, converting the ResNet50 into a domain adversarial neural network (DANN) substantially improves performance on unlabeled target domains. A DANN trained on labeled breast and colon data and adapted to unlabeled lung data reaches 95.56% accuracy.
We also examine the impact of stain normalization on domain adaptation. Its effects vary by target domain: for lung, accuracy drops from 95.56% to 66.60%, while for breast and colon targets, stain normalization boosts accuracy from 49.22% to 81.29% and from 78.48% to 83.36%, respectively. Finally, using Integrated Gradients reveals that DANNs consistently attribute importance to biologically meaningful regions such as densely packed nuclei, indicating that the model learns clinically relevant features and can apply them to unlabeled cancer types. 

---
# A comprehensive overview of deep learning models for object detection from videos/images 

**Authors**: Sukana Zulfqar, Sadia Saeed, M. Azam Zia, Anjum Ali, Faisal Mehmood, Abid Ali  

**Link**: [PDF](https://arxiv.org/pdf/2601.14677)  

**Abstract**: Object detection in video and image surveillance is a well-established yet rapidly evolving task, strongly influenced by recent deep learning advancements. This review summarises modern techniques by examining architectural innovations, generative model integration, and the use of temporal information to enhance robustness and accuracy. Unlike earlier surveys, it classifies methods based on core architectures, data processing strategies, and surveillance specific challenges such as dynamic environments, occlusions, lighting variations, and real-time requirements. The primary goal is to evaluate the current effectiveness of semantic object detection, while secondary aims include analysing deep learning models and their practical applications. The review covers CNN-based detectors, GAN-assisted approaches, and temporal fusion methods, highlighting how generative models support tasks such as reconstructing missing frames, reducing occlusions, and normalising illumination. It also outlines preprocessing pipelines, feature extraction progress, benchmarking datasets, and comparative evaluations. Finally, emerging trends in low-latency, efficient, and spatiotemporal learning approaches are identified for future research. 

---
# Efficient reformulations of ReLU deep neural networks for surrogate modelling in power system optimisation 

**Authors**: Yogesh Pipada Sunil Kumar, S. Ali Pourmousavi, Jon A.R. Liisberg, Julian Lesmos-Vinasco  

**Link**: [PDF](https://arxiv.org/pdf/2601.14673)  

**Abstract**: The ongoing decarbonisation of power systems is driving an increasing reliance on distributed energy resources, which introduces complex and nonlinear interactions that are difficult to capture in conventional optimisation models. As a result, machine learning based surrogate modelling has emerged as a promising approach, but integrating machine learning models such as ReLU deep neural networks (DNNs) directly into optimisation often results in nonconvex and computationally intractable formulations. This paper proposes a linear programming (LP) reformulation for a class of convexified ReLU DNNs with non-negative weight matrices beyond the first layer, enabling a tight and tractable embedding of learned surrogate models in optimisation. We evaluate the method using a case study on learning the prosumer's responsiveness within an aggregator bidding problem in the Danish tertiary capacity market. The proposed reformulation is benchmarked against state-of-the-art alternatives, including piecewise linearisation (PWL), MIP-based embedding, and other LP relaxations. Across multiple neural network architectures and market scenarios, the convexified ReLU DNN achieves solution quality comparable to PWL and MIP-based reformulations while significantly improving computational performance and preserving model fidelity, unlike penalty-based reformulations. The results demonstrate that convexified ReLU DNNs offer a scalable and reliable methodology for integrating learned surrogate models in optimisation, with applicability to a wide range of emerging power system applications. 

---
# GEGO: A Hybrid Golden Eagle and Genetic Optimization Algorithm for Efficient Hyperparameter Tuning in Resource-Constrained Environments 

**Authors**: Amaras Nazarians, Sachin Kumar  

**Link**: [PDF](https://arxiv.org/pdf/2601.14672)  

**Abstract**: Hyperparameter tuning is a critical yet computationally expensive step in training neural networks, particularly when the search space is high dimensional and nonconvex. Metaheuristic optimization algorithms are often used for this purpose due to their derivative free nature and robustness against local optima. In this work, we propose Golden Eagle Genetic Optimization (GEGO), a hybrid metaheuristic that integrates the population movement strategy of Golden Eagle Optimization with the genetic operators of selection, crossover, and mutation.
The main novelty of GEGO lies in embedding genetic operators directly into the iterative search process of GEO, rather than applying them as a separate evolutionary stage. This design improves population diversity during search and reduces premature convergence while preserving the exploration behavior of GEO.
GEGO is evaluated on standard unimodal, multimodal, and composite benchmark functions from the CEC2017 suite, where it consistently outperforms its constituent algorithms and several classical metaheuristics in terms of solution quality and robustness. The algorithm is further applied to hyperparameter tuning of artificial neural networks on the MNIST dataset, where GEGO achieves improved classification accuracy and more stable convergence compared to GEO and GA. These results indicate that GEGO provides a balanced exploration-exploitation tradeoff and is well suited for hyperparameter optimization under constrained computational settings. 

---
# INFA-Guard: Mitigating Malicious Propagation via Infection-Aware Safeguarding in LLM-Based Multi-Agent Systems 

**Authors**: Yijin Zhou, Xiaoya Lu, Dongrui Liu, Junchi Yan, Jing Shao  

**Link**: [PDF](https://arxiv.org/pdf/2601.14667)  

**Abstract**: The rapid advancement of Large Language Model (LLM)-based Multi-Agent Systems (MAS) has introduced significant security vulnerabilities, where malicious influence can propagate virally through inter-agent communication. Conventional safeguards often rely on a binary paradigm that strictly distinguishes between benign and attack agents, failing to account for infected agents i.e., benign entities converted by attack agents. In this paper, we propose Infection-Aware Guard, INFA-Guard, a novel defense framework that explicitly identifies and addresses infected agents as a distinct threat category. By leveraging infection-aware detection and topological constraints, INFA-Guard accurately localizes attack sources and infected ranges. During remediation, INFA-Guard replaces attackers and rehabilitates infected ones, avoiding malicious propagation while preserving topological integrity. Extensive experiments demonstrate that INFA-Guard achieves state-of-the-art performance, reducing the Attack Success Rate (ASR) by an average of 33%, while exhibiting cross-model robustness, superior topological generalization, and high cost-effectiveness. 

---
# Calibrated uncertainty quantification for prosumer flexibility aggregation in ancillary service markets 

**Authors**: Yogesh Pipada Sunil Kumar, S. Ali Pourmousavi, Jon A.R. Liisberg, Julian Lesmos-Vinasco  

**Link**: [PDF](https://arxiv.org/pdf/2601.14663)  

**Abstract**: Reliable forecasting of prosumer flexibility is critical for demand response aggregators participating in frequency controlled ancillary services market, where strict reliability requirements such as the P90 standard are enforced. Limited historical data, dependence on exogeneous factors, and heterogenous prosumer behaviour introduce significant epistemic uncertainty, making deterministic or poorly calibrated probabilistic models unsuitable for market bidding. This paper proposes the use of scalable uncertainty quantification framework that integrates Monte Carlo dropout (MCD) with conformal prediction (CP) to produce calibrated, finite sample prediction intervals for aggregated prosumer flexibility. The proposed framework is applied to a behind-the-meter aggregator participating in the Danish manual frequency restoration reserve capacity market. A large-scale synthetic dataset is generated using a modified industry-grade home energy management system, combined with publicly available load, solar, price, activation and device-level data. The resulting machine learning surrogate model captures aggregate prosumer price responsiveness and provides uncertainty-aware estimates suitable for market bidding. Multiple multivariate CP strategies are evaluated and benchmarked against conventional MCD-based methods. Results show that standalone MCD systematically overestimates available flexibility and violates P90 compliance, whereas the proposed MCD-CP framework achieves reliable coverage with controlled conservatism. When embedded in aggregator bidding model, conformalised methods substantially reduce overbidding risk and achieve upto 70% of perfect-information profit while satisfying regulatory reliability constraints, providing practical, computationally efficient, and market-compliant solution for aggregator flexibility forecasting under uncertainty. 

---
# NeuroFilter: Privacy Guardrails for Conversational LLM Agents 

**Authors**: Saswat Das, Ferdinando Fioretto  

**Link**: [PDF](https://arxiv.org/pdf/2601.14660)  

**Abstract**: This work addresses the computational challenge of enforcing privacy for agentic Large Language Models (LLMs), where privacy is governed by the contextual integrity framework. Indeed, existing defenses rely on LLM-mediated checking stages that add substantial latency and cost, and that can be undermined in multi-turn interactions through manipulation or benign-looking conversational scaffolding. Contrasting this background, this paper makes a key observation: internal representations associated with privacy-violating intent can be separated from benign requests using linear structure. Using this insight, the paper proposes NeuroFilter, a guardrail framework that operationalizes contextual integrity by mapping norm violations to simple directions in the model's activation space, enabling detection even when semantic filters are bypassed. The proposed filter is also extended to capture threats arising during long conversations using the concept of activation velocity, which measures cumulative drift in internal representations across turns. A comprehensive evaluation across over 150,000 interactions and covering models from 7B to 70B parameters, illustrates the strong performance of NeuroFilter in detecting privacy attacks while maintaining zero false positives on benign prompts, all while reducing the computational inference cost by several orders of magnitude when compared to LLM-based agentic privacy defenses. 

---
# Say Anything but This: When Tokenizer Betrays Reasoning in LLMs 

**Authors**: Navid Ayoobi, Marcus I Armstrong, Arjun Mukherjee  

**Link**: [PDF](https://arxiv.org/pdf/2601.14658)  

**Abstract**: Large language models (LLMs) reason over discrete token ID sequences, yet modern subword tokenizers routinely produce non-unique encodings: multiple token ID sequences can detokenize to identical surface strings. This representational mismatch creates an unmeasured fragility wherein reasoning processes can fail. LLMs may treat two internal representations as distinct "words" even when they are semantically identical at the text level. In this work, we show that tokenization can betray LLM reasoning through one-to-many token ID mappings. We introduce a tokenization-consistency probe that requires models to replace designated target words in context while leaving all other content unchanged. The task is intentionally simple at the surface level, enabling us to attribute failures to tokenizer-detokenizer artifacts rather than to knowledge gaps or parameter limitations. Through analysis of over 11000 replacement trials across state-of-the-art open-source LLMs, we find a non-trivial rate of outputs exhibit phantom edits: cases where models operate under the illusion of correct reasoning, a phenomenon arising from tokenizer-induced representational defects. We further analyze these cases and provide a taxonomy of eight systematic tokenizer artifacts, including whitespace-boundary shifts and intra-word resegmentation. These findings indicate that part of apparent reasoning deficiency originates in the tokenizer layer, motivating tokenizer-level remedies before incurring the cost of training ever-larger models on ever-larger corpora. 

---
# Forest-Chat: Adapting Vision-Language Agents for Interactive Forest Change Analysis 

**Authors**: James Brock, Ce Zhang, Nantheera Anantrasirichai  

**Link**: [PDF](https://arxiv.org/pdf/2601.14637)  

**Abstract**: The increasing availability of high-resolution satellite imagery, together with advances in deep learning, creates new opportunities for enhancing forest monitoring workflows. Two central challenges in this domain are pixel-level change detection and semantic change interpretation, particularly for complex forest dynamics. While large language models (LLMs) are increasingly adopted for data exploration, their integration with vision-language models (VLMs) for remote sensing image change interpretation (RSICI) remains underexplored, especially beyond urban environments. We introduce Forest-Chat, an LLM-driven agent designed for integrated forest change analysis. The proposed framework enables natural language querying and supports multiple RSICI tasks, including change detection, change captioning, object counting, deforestation percentage estimation, and change reasoning. Forest-Chat builds upon a multi-level change interpretation (MCI) vision-language backbone with LLM-based orchestration, and incorporates zero-shot change detection via a foundation change detection model together with an interactive point-prompt interface to support fine-grained user guidance. To facilitate adaptation and evaluation in forest environments, we introduce the Forest-Change dataset, comprising bi-temporal satellite imagery, pixel-level change masks, and multi-granularity semantic change captions generated through a combination of human annotation and rule-based methods. Experimental results demonstrate that Forest-Chat achieves strong performance on Forest-Change and on LEVIR-MCI-Trees, a tree-focused subset of LEVIR-MCI, for joint change detection and captioning, highlighting the potential of interactive, LLM-driven RSICI systems to improve accessibility, interpretability, and analytical efficiency in forest change analysis. 

---
# A Brain-inspired Embodied Intelligence for Fluid and Fast Reflexive Robotics Control 

**Authors**: Weiyu Guo, He Zhang, Pengteng Li, Tiefu Cai, Ziyang Chen, Yandong Guo, Xiao He, Yongkui Yang, Ying Sun, Hui Xiong  

**Link**: [PDF](https://arxiv.org/pdf/2601.14628)  

**Abstract**: Recent advances in embodied intelligence have leveraged massive scaling of data and model parameters to master natural-language command following and multi-task control. In contrast, biological systems demonstrate an innate ability to acquire skills rapidly from sparse experience. Crucially, current robotic policies struggle to replicate the dynamic stability, reflexive responsiveness, and temporal memory inherent in biological motion. Here we present Neuromorphic Vision-Language-Action (NeuroVLA), a framework that mimics the structural organization of the bio-nervous system between the cortex, cerebellum, and spinal cord. We adopt a system-level bio-inspired design: a high-level model plans goals, an adaptive cerebellum module stabilizes motion using high-frequency sensors feedback, and a bio-inspired spinal layer executes lightning-fast actions generation. NeuroVLA represents the first deployment of a neuromorphic VLA on physical robotics, achieving state-of-the-art performance. We observe the emergence of biological motor characteristics without additional data or special guidance: it stops the shaking in robotic arms, saves significant energy(only 0.4w on Neuromorphic Processor), shows temporal memory ability and triggers safety reflexes in less than 20 milliseconds. 

---
# Scaling Ambiguity: Augmenting Human Annotation in Speech Emotion Recognition with Audio-Language Models 

**Authors**: Wenda Zhang, Hongyu Jin, Siyi Wang, Zhiqiang Wei, Ting Dang  

**Link**: [PDF](https://arxiv.org/pdf/2601.14620)  

**Abstract**: Speech Emotion Recognition models typically use single categorical labels, overlooking the inherent ambiguity of human emotions. Ambiguous Emotion Recognition addresses this by representing emotions as probability distributions, but progress is limited by unreliable ground-truth distributions inferred from sparse human annotations. This paper explores whether Large Audio-Language Models (ALMs) can mitigate the annotation bottleneck by generating high-quality synthetic annotations. We introduce a framework leveraging ALMs to create Synthetic Perceptual Proxies, augmenting human annotations to improve ground-truth distribution reliability. We validate these proxies through statistical analysis of their alignment with human distributions and evaluate their impact by fine-tuning ALMs with the augmented emotion distributions. Furthermore, to address class imbalance and enable unbiased evaluation, we propose DiME-Aug, a Distribution-aware Multimodal Emotion Augmentation strategy. Experiments on IEMOCAP and MSP-Podcast show that synthetic annotations enhance emotion distribution, especially in low-ambiguity regions where annotation agreement is high. However, benefits diminish for highly ambiguous emotions with greater human disagreement. This work provides the first evidence that ALMs could address annotation scarcity in ambiguous emotion recognition, but highlights the need for more advanced prompting or generation strategies to handle highly ambiguous cases. 

---
# SearchGym: Bootstrapping Real-World Search Agents via Cost-Effective and High-Fidelity Environment Simulation 

**Authors**: Xichen Zhang, Ziyi He, Yinghao Zhu, Sitong Wu, Shaozuo Yu, Meng Chu, Wenhu Zhang, Haoru Tan, Jiaya Jia  

**Link**: [PDF](https://arxiv.org/pdf/2601.14615)  

**Abstract**: Search agents have emerged as a pivotal paradigm for solving open-ended, knowledge-intensive reasoning tasks. However, training these agents via Reinforcement Learning (RL) faces a critical dilemma: interacting with live commercial Web APIs is prohibitively expensive, while relying on static data snapshots often introduces noise due to data misalignment. This misalignment generates corrupted reward signals that destabilize training by penalizing correct reasoning or rewarding hallucination. To address this, we propose SearchGym, a simulation environment designed to bootstrap robust search agents. SearchGym employs a rigorous generative pipeline to construct a verifiable knowledge graph and an aligned document corpus, ensuring that every reasoning task is factually grounded and strictly solvable. Building on this controllable environment, we introduce SearchGym-RL, a curriculum learning methodology that progressively optimizes agent policies through purified feedback, evolving from basic interactions to complex, long-horizon planning. Extensive experiments across the Llama and Qwen families demonstrate strong Sim-to-Real generalization. Notably, our Qwen2.5-7B-Base model trained within SearchGym surpasses the web-enhanced ASearcher baseline across nine diverse benchmarks by an average relative margin of 10.6%. Our results validate that high-fidelity simulation serves as a scalable and highly cost-effective methodology for developing capable search agents. 

---
# Communication-Efficient Federated Risk Difference Estimation for Time-to-Event Clinical Outcomes 

**Authors**: Ziwen Wang, Siqi Li, Marcus Eng Hock Ong, Nan Liu  

**Link**: [PDF](https://arxiv.org/pdf/2601.14609)  

**Abstract**: Privacy-preserving model co-training in medical research is often hindered by server-dependent architectures incompatible with protected hospital data systems and by the predominant focus on relative effect measures (hazard ratios) which lack clinical interpretability for absolute survival risk assessment. We propose FedRD, a communication-efficient framework for federated risk difference estimation in distributed survival data. Unlike typical federated learning frameworks (e.g., FedAvg) that require persistent server connections and extensive iterative communication, FedRD is server-independent with minimal communication: one round of summary statistics exchange for the stratified model and three rounds for the unstratified model. Crucially, FedRD provides valid confidence intervals and hypothesis testing--capabilities absent in FedAvg-based frameworks. We provide theoretical guarantees by establishing the asymptotic properties of FedRD and prove that FedRD (unstratified) is asymptotically equivalent to pooled individual-level analysis. Simulation studies and real-world clinical applications across different countries demonstrate that FedRD outperforms local and federated baselines in both estimation accuracy and prediction performance, providing an architecturally feasible solution for absolute risk assessment in privacy-restricted, multi-site clinical studies. 

---
# Rethinking Reinforcement fine-tuning of LLMs: A Multi-armed Bandit Learning Perspective 

**Authors**: Xiao Hu, Hong Xie, Tao Tan, Defu Lian, Jianyu Han  

**Link**: [PDF](https://arxiv.org/pdf/2601.14599)  

**Abstract**: A large number of heuristics have been proposed to optimize the reinforcement fine-tuning of LLMs. However, inconsistent claims are made from time to time, making this area elusive. Reflecting on this situation, two fundamental questions still lack a clear understanding: 1) what is the role of each optimizing choice? 2) which ones are the bottlenecks? This paper aims to shed light on them, and it faces the challenge of several entangled confounding factors in the fine-tuning process. To tackle this challenge, we propose a bottom-up experiment pipeline. The bottom layer is composed of a minimalist configuration: one training data, one rollout per round and the reward directly serve as the learning signal without advantage function design. This minimalist configuration connects to multi-armed bandit learning with extremely large discrete action space, which offers theories to corroborate the experiment findings. The up procedure of the experiment pipeline expanding the minimalist configuration layer by layer, examining the role of each design choice. Experimental results on three LLMs and two reasoning datasets not only reveal new understanding of the design choice but also yield essential insights to shape the area. 

---
# HELIOS: Hierarchical Graph Abstraction for Structure-Aware LLM Decompilation 

**Authors**: Yonatan Gizachew Achamyeleh, Harsh Thomare, Mohammad Abdullah Al Faruque  

**Link**: [PDF](https://arxiv.org/pdf/2601.14598)  

**Abstract**: Large language models (LLMs) have recently been applied to binary decompilation, yet they still treat code as plain text and ignore the graphs that govern program control flow. This limitation often yields syntactically fragile and logically inconsistent output, especially for optimized binaries. This paper presents \textsc{HELIOS}, a framework that reframes LLM-based decompilation as a structured reasoning task. \textsc{HELIOS} summarizes a binary's control flow and function calls into a hierarchical text representation that spells out basic blocks, their successors, and high-level patterns such as loops and conditionals. This representation is supplied to a general-purpose LLM, along with raw decompiler output, optionally combined with a compiler-in-the-loop that returns error messages when the generated code fails to build.
On HumanEval-Decompile for \texttt{x86\_64}, \textsc{HELIOS} raises average object file compilability from 45.0\% to 85.2\% for Gemini~2.0 and from 71.4\% to 89.6\% for GPT-4.1~Mini. With compiler feedback, compilability exceeds 94\% and functional correctness improves by up to 5.6 percentage points over text-only prompting. Across six architectures drawn from x86, ARM, and MIPS, \textsc{HELIOS} reduces the spread in functional correctness while keeping syntactic correctness consistently high, all without fine-tuning. These properties make \textsc{HELIOS} a practical building block for reverse engineering workflows in security settings where analysts need recompilable, semantically faithful code across diverse hardware targets. 

---
# Optimality of Staircase Mechanisms for Vector Queries under Differential Privacy 

**Authors**: James Melbourne, Mario Diaz, Shahab Asoodeh  

**Link**: [PDF](https://arxiv.org/pdf/2601.14597)  

**Abstract**: We study the optimal design of additive mechanisms for vector-valued queries under $\epsilon$-differential privacy (DP). Given only the sensitivity of a query and a norm-monotone cost function measuring utility loss, we ask which noise distribution minimizes expected cost among all additive $\epsilon$-DP mechanisms. Using convex rearrangement theory, we show that this infinite-dimensional optimization problem admits a reduction to a one-dimensional compact and convex family of radially symmetric distributions whose extreme points are the staircase distributions. As a consequence, we prove that for any dimension, any norm, and any norm-monotone cost function, there exists an $\epsilon$-DP staircase mechanism that is optimal among all additive mechanisms. This result resolves a conjecture of Geng, Kairouz, Oh, and Viswanath, and provides a geometric explanation for the emergence of staircase mechanisms as extremal solutions in differential privacy. 

---
# IntelliSA: An Intelligent Static Analyzer for IaC Security Smell Detection Using Symbolic Rules and Neural Inference 

**Authors**: Qiyue Mei, Michael Fu  

**Link**: [PDF](https://arxiv.org/pdf/2601.14595)  

**Abstract**: Infrastructure as Code (IaC) enables automated provisioning of large-scale cloud and on-premise environments, reducing the need for repetitive manual setup. However, this automation is a double-edged sword: a single misconfiguration in IaC scripts can propagate widely, leading to severe system downtime and security risks. Prior studies have shown that IaC scripts often contain security smells--bad coding patterns that may introduce vulnerabilities--and have proposed static analyzers based on symbolic rules to detect them. Yet, our preliminary analysis reveals that rule-based detection alone tends to over-approximate, producing excessive false positives and increasing the burden of manual inspection. In this paper, we present IntelliSA, an intelligent static analyzer for IaC security smell detection that integrates symbolic rules with neural inference. IntelliSA applies symbolic rules to over-approximate potential smells for broad coverage, then employs neural inference to filter false positives. While an LLM can effectively perform this filtering, reliance on LLM APIs introduces high cost and latency, raises data governance concerns, and limits reproducibility and offline deployment. To address the challenges, we adopt a knowledge distillation approach: an LLM teacher generates pseudo-labels to train a compact student model--over 500x smaller--that learns from the teacher's knowledge and efficiently classifies false positives. We evaluate IntelliSA against two static analyzers and three LLM baselines (Claude-4, Grok-4, and GPT-5) using a human-labeled dataset including 241 security smells across 11,814 lines of real-world IaC code. Experimental results show that IntelliSA achieves the highest F1 score (83%), outperforming baselines by 7-42%. Moreover, IntelliSA demonstrates the best cost-effectiveness, detecting 60% of security smells while inspecting less than 2% of the codebase. 

---
# Designing KRIYA: An AI Companion for Wellbeing Self-Reflection 

**Authors**: Shanshan Zhu, Wenxuan Song, Jiayue Melissa Shi, Dong Whi Yoo, Karthik S. Bhat, Koustuv Saha  

**Link**: [PDF](https://arxiv.org/pdf/2601.14589)  

**Abstract**: Most personal wellbeing apps present summative dashboards of health and physical activity metrics, yet many users struggle to translate this information into meaningful understanding. These apps commonly support engagement through goals, reminders, and structured targets, which can reinforce comparison, judgment, and performance anxiety. To explore a complementary approach that prioritizes self-reflection, we design KRIYA, an AI wellbeing companion that supports co-interpretive engagement with personal wellbeing data. KRIYA aims to collaborate with users to explore questions, explanations, and future scenarios through features such as Comfort Zone, Detective Mode, and What-If Planning. We conducted semi-structured interviews with 18 college students interacting with a KRIYA prototype using hypothetical data. Our findings show that through KRIYA interaction, users framed engaging with wellbeing data as interpretation rather than performance, experienced reflection as supportive or pressuring depending on emotional framing, and developed trust through transparency. We discuss design implications for AI companions that support curiosity, self-compassion, and reflective sensemaking of personal health data. 

---
# Breaking the accuracy-resource dilemma: a lightweight adaptive video inference enhancement 

**Authors**: Wei Ma, Shaowu Chen, Junjie Ye, Peichang Zhang, Lei Huang  

**Link**: [PDF](https://arxiv.org/pdf/2601.14568)  

**Abstract**: Existing video inference (VI) enhancement methods typically aim to improve performance by scaling up model sizes and employing sophisticated network architectures. While these approaches demonstrated state-of-the-art performance, they often overlooked the trade-off of resource efficiency and inference effectiveness, leading to inefficient resource utilization and suboptimal inference performance. To address this problem, a fuzzy controller (FC-r) is developed based on key system parameters and inference-related metrics. Guided by the FC-r, a VI enhancement framework is proposed, where the spatiotemporal correlation of targets across adjacent video frames is leveraged. Given the real-time resource conditions of the target device, the framework can dynamically switch between models of varying scales during VI. Experimental results demonstrate that the proposed method effectively achieves a balance between resource utilization and inference performance. 

---
# Self-Blinding and Counterfactual Self-Simulation Mitigate Biases and Sycophancy in Large Language Models 

**Authors**: Brian Christian, Matan Mazor  

**Link**: [PDF](https://arxiv.org/pdf/2601.14553)  

**Abstract**: Fair decisions require ignoring irrelevant, potentially biasing, information. To achieve this, decision-makers need to approximate what decision they would have made had they not known certain facts, such as the gender or race of a job candidate. This counterfactual self-simulation is notoriously hard for humans, leading to biased judgments even by well-meaning actors. Here we show that large language models (LLMs) suffer from similar limitations in their ability to approximate what decisions they would make under counterfactual knowledge in offsetting gender and race biases and overcoming sycophancy. We show that prompting models to ignore or pretend not to know biasing information fails to offset these biases and occasionally backfires. However, unlike humans, LLMs can be given access to a ground-truth model of their own counterfactual cognition -- their own API. We show that this access to the responses of a blinded replica enables fairer decisions, while providing greater transparency to distinguish implicit from intentionally biased behavior. 

---
# Report for NSF Workshop on AI for Electronic Design Automation 

**Authors**: Deming Chen, Vijay Ganesh, Weikai Li, Yingyan, Yong Liu, Subhasish Mitra, David Z. Pan, Ruchir Puri, Jason Cong, Yizhou Sun  

**Link**: [PDF](https://arxiv.org/pdf/2601.14541)  

**Abstract**: This report distills the discussions and recommendations from the NSF Workshop on AI for Electronic Design Automation (EDA), held on December 10, 2024 in Vancouver alongside NeurIPS 2024. Bringing together experts across machine learning and EDA, the workshop examined how AI-spanning large language models (LLMs), graph neural networks (GNNs), reinforcement learning (RL), neurosymbolic methods, etc.-can facilitate EDA and shorten design turnaround. The workshop includes four themes: (1) AI for physical synthesis and design for manufacturing (DFM), discussing challenges in physical manufacturing process and potential AI applications; (2) AI for high-level and logic-level synthesis (HLS/LLS), covering pragma insertion, program transformation, RTL code generation, etc.; (3) AI toolbox for optimization and design, discussing frontier AI developments that could potentially be applied to EDA tasks; and (4) AI for test and verification, including LLM-assisted verification tools, ML-augmented SAT solving, security/reliability challenges, etc. The report recommends NSF to foster AI/EDA collaboration, invest in foundational AI for EDA, develop robust data infrastructures, promote scalable compute infrastructure, and invest in workforce development to democratize hardware design and enable next-generation hardware systems. The workshop information can be found on the website this https URL. 

---
# Towards Execution-Grounded Automated AI Research 

**Authors**: Chenglei Si, Zitong Yang, Yejin Choi, Emmanuel Candès, Diyi Yang, Tatsunori Hashimoto  

**Link**: [PDF](https://arxiv.org/pdf/2601.14525)  

**Abstract**: Automated AI research holds great potential to accelerate scientific discovery. However, current LLMs often generate plausible-looking but ineffective ideas. Execution grounding may help, but it is unclear whether automated execution is feasible and whether LLMs can learn from the execution feedback. To investigate these, we first build an automated executor to implement ideas and launch large-scale parallel GPU experiments to verify their effectiveness. We then convert two realistic research problems - LLM pre-training and post-training - into execution environments and demonstrate that our automated executor can implement a large fraction of the ideas sampled from frontier LLMs. We analyze two methods to learn from the execution feedback: evolutionary search and reinforcement learning. Execution-guided evolutionary search is sample-efficient: it finds a method that significantly outperforms the GRPO baseline (69.4% vs 48.0%) on post-training, and finds a pre-training recipe that outperforms the nanoGPT baseline (19.7 minutes vs 35.9 minutes) on pre-training, all within just ten search epochs. Frontier LLMs often generate meaningful algorithmic ideas during search, but they tend to saturate early and only occasionally exhibit scaling trends. Reinforcement learning from execution reward, on the other hand, suffers from mode collapse. It successfully improves the average reward of the ideator model but not the upper-bound, due to models converging on simple ideas. We thoroughly analyze the executed ideas and training dynamics to facilitate future efforts towards execution-grounded automated AI research. 

---
# How Worst-Case Are Adversarial Attacks? Linking Adversarial and Statistical Robustness 

**Authors**: Giulio Rossolini  

**Link**: [PDF](https://arxiv.org/pdf/2601.14519)  

**Abstract**: Adversarial attacks are widely used to evaluate model robustness, yet their validity as proxies for robustness to random perturbations remains debated. We ask whether an adversarial perturbation provides a representative estimate of robustness under random noise of the same magnitude, or instead reflects an atypical worst-case event. To this end, we introduce a probabilistic metric that quantifies noisy risk with respect to directionally biased perturbation distributions, parameterized by a concentration factor $\kappa$ that interpolates between isotropic noise and adversarial direction. Using this framework, we study the limits of adversarial perturbations as estimators of noisy risk by proposing an attack strategy designed to operate in regimes statistically closer to uniform noise. Experiments on ImageNet and CIFAR-10 systematically benchmark widely used attacks, highlighting when adversarial success meaningfully reflects noisy risk and when it fails, thereby informing their use in safety-oriented evaluation. 

---
# GutenOCR: A Grounded Vision-Language Front-End for Documents 

**Authors**: Hunter Heidenreich, Ben Elliott, Olivia Dinica, Yosheb Getachew  

**Link**: [PDF](https://arxiv.org/pdf/2601.14490)  

**Abstract**: GutenOCR is a family of grounded OCR front-ends obtained by fine-tuning Qwen2.5-VL-3B and Qwen2.5-VL-7B. The resulting single-checkpoint vision-language models expose reading, detection, and grounding through a unified, prompt-based interface. Trained on business documents, scientific articles, and synthetic grounding data, the models support full-page and localized reading with line- and paragraph-level bounding boxes and conditional ``where is x?'' queries. We introduce a grounded OCR evaluation protocol and show that GutenOCR-7B more than doubles the composite grounded OCR score of its Qwen2.5-VL-7B backbone on 10.5K held-out business and scientific pages (0.40 to 0.82). On Fox and OmniDocBench v1.5, our approach substantially improves region- and line-level OCR as well as text-detection recall, but reveals trade-offs in page-level linearization, color-guided OCR, and formula-heavy layouts. 

---
# XD-MAP: Cross-Modal Domain Adaptation using Semantic Parametric Mapping 

**Authors**: Frank Bieder, Hendrik Königshof, Haohao Hu, Fabian Immel, Yinzhe Shen, Jan-Hendrik Pauls, Christoph Stiller  

**Link**: [PDF](https://arxiv.org/pdf/2601.14477)  

**Abstract**: Until open-world foundation models match the performance of specialized approaches, the effectiveness of deep learning models remains heavily dependent on dataset availability. Training data must align not only with the target object categories but also with the sensor characteristics and modalities. To bridge the gap between available datasets and deployment domains, domain adaptation strategies are widely used. In this work, we propose a novel approach to transferring sensor-specific knowledge from an image dataset to LiDAR, an entirely different sensing domain. Our method XD-MAP leverages detections from a neural network on camera images to create a semantic parametric map. The map elements are modeled to produce pseudo labels in the target domain without any manual annotation effort. Unlike previous domain transfer approaches, our method does not require direct overlap between sensors and enables extending the angular perception range from a front-view camera to a full 360 view. On our large-scale road feature dataset, XD-MAP outperforms single shot baseline approaches by +19.5 mIoU for 2D semantic segmentation, +19.5 PQth for 2D panoptic segmentation, and +32.3 mIoU in 3D semantic segmentation. The results demonstrate the effectiveness of our approach achieving strong performance on LiDAR data without any manual labeling. 

---
# GPU-accelerated simulated annealing based on p-bits with real-world device-variability modeling 

**Authors**: Naoya Onizawa, Takahiro Hanyu  

**Link**: [PDF](https://arxiv.org/pdf/2601.14476)  

**Abstract**: Probabilistic computing using probabilistic bits (p-bits) presents an efficient alternative to traditional CMOS logic for complex problem-solving, including simulated annealing and machine learning. Realizing p-bits with emerging devices such as magnetic tunnel junctions (MTJs) introduces device variability, which was expected to negatively impact computational performance. However, this study reveals an unexpected finding: device variability can not only degrade but also enhance algorithm performance, particularly by leveraging timing variability. This paper introduces a GPU-accelerated, open-source simulated annealing framework based on p-bits that models key device variability factors -timing, intensity, and offset- to reflect real-world device behavior. Through CUDA-based simulations, our approach achieves a two-order magnitude speedup over CPU implementations on the MAX-CUT benchmark with problem sizes ranging from 800 to 20,000 nodes. By providing a scalable and accessible tool, this framework aims to advance research in probabilistic computing, enabling optimization applications in diverse fields. 

---
# Real-Time Wildfire Localization on the NASA Autonomous Modular Sensor using Deep Learning 

**Authors**: Yajvan Ravan, Aref Malek, Chester Dolph, Nikhil Behari  

**Link**: [PDF](https://arxiv.org/pdf/2601.14475)  

**Abstract**: High-altitude, multi-spectral, aerial imagery is scarce and expensive to acquire, yet it is necessary for algorithmic advances and application of machine learning models to high-impact problems such as wildfire detection. We introduce a human-annotated dataset from the NASA Autonomous Modular Sensor (AMS) using 12-channel, medium to high altitude (3 - 50 km) aerial wildfire images similar to those used in current US wildfire missions. Our dataset combines spectral data from 12 different channels, including infrared (IR), short-wave IR (SWIR), and thermal. We take imagery from 20 wildfire missions and randomly sample small patches to generate over 4000 images with high variability, including occlusions by smoke/clouds, easily-confused false positives, and nighttime imagery.
We demonstrate results from a deep-learning model to automate the human-intensive process of fire perimeter determination. We train two deep neural networks, one for image classification and the other for pixel-level segmentation. The networks are combined into a unique real-time segmentation model to efficiently localize active wildfire on an incoming image feed. Our model achieves 96% classification accuracy, 74% Intersection-over-Union(IoU), and 84% recall surpassing past methods, including models trained on satellite data and classical color-rule algorithms. By leveraging a multi-spectral dataset, our model is able to detect active wildfire at nighttime and behind clouds, while distinguishing between false positives. We find that data from the SWIR, IR, and thermal bands is the most important to distinguish fire perimeters. Our code and dataset can be found here: this https URL and this https URL 

---
# Prosody-Guided Harmonic Attention for Phase-Coherent Neural Vocoding in the Complex Spectrum 

**Authors**: Mohammed Salah Al-Radhi, Riad Larbi, Mátyás Bartalis, Géza Németh  

**Link**: [PDF](https://arxiv.org/pdf/2601.14472)  

**Abstract**: Neural vocoders are central to speech synthesis; despite their success, most still suffer from limited prosody modeling and inaccurate phase reconstruction. We propose a vocoder that introduces prosody-guided harmonic attention to enhance voiced segment encoding and directly predicts complex spectral components for waveform synthesis via inverse STFT. Unlike mel-spectrogram-based approaches, our design jointly models magnitude and phase, ensuring phase coherence and improved pitch fidelity. To further align with perceptual quality, we adopt a multi-objective training strategy that integrates adversarial, spectral, and phase-aware losses. Experiments on benchmark datasets demonstrate consistent gains over HiFi-GAN and AutoVocoder: F0 RMSE reduced by 22 percent, voiced/unvoiced error lowered by 18 percent, and MOS scores improved by 0.15. These results show that prosody-guided attention combined with direct complex spectrum modeling yields more natural, pitch-accurate, and robust synthetic speech, setting a strong foundation for expressive neural vocoding. 

---
# Tokenomics: Quantifying Where Tokens Are Used in Agentic Software Engineering 

**Authors**: Mohamad Salim, Jasmine Latendresse, SayedHassan Khatoonabadi, Emad Shihab  

**Link**: [PDF](https://arxiv.org/pdf/2601.14470)  

**Abstract**: LLM-based Multi-Agent (LLM-MA) systems are increasingly applied to automate complex software engineering tasks such as requirements engineering, code generation, and testing. However, their operational efficiency and resource consumption remain poorly understood, hindering practical adoption due to unpredictable costs and environmental impact. To address this, we conduct an analysis of token consumption patterns in an LLM-MA system within the Software Development Life Cycle (SDLC), aiming to understand where tokens are consumed across distinct software engineering activities. We analyze execution traces from 30 software development tasks performed by the ChatDev framework using a GPT-5 reasoning model, mapping its internal phases to distinct development stages (Design, Coding, Code Completion, Code Review, Testing, and Documentation) to create a standardized evaluation framework. We then quantify and compare token distribution (input, output, reasoning) across these stages.
Our preliminary findings show that the iterative Code Review stage accounts for the majority of token consumption for an average of 59.4% of tokens. Furthermore, we observe that input tokens consistently constitute the largest share of consumption for an average of 53.9%, providing empirical evidence for potentially significant inefficiencies in agentic collaboration. Our results suggest that the primary cost of agentic software engineering lies not in initial code generation but in automated refinement and verification. Our novel methodology can help practitioners predict expenses and optimize workflows, and it directs future research toward developing more token-efficient agent collaboration protocols. 

---
# Diffusion Large Language Models for Black-Box Optimization 

**Authors**: Ye Yuan, Chen, Zipeng Sun, Dinghuai Zhang, Christopher Pal, Xue Liu  

**Link**: [PDF](https://arxiv.org/pdf/2601.14446)  

**Abstract**: Offline black-box optimization (BBO) aims to find optimal designs based solely on an offline dataset of designs and their labels. Such scenarios frequently arise in domains like DNA sequence design and robotics, where only a few labeled data points are available. Traditional methods typically rely on task-specific proxy or generative models, overlooking the in-context learning capabilities of pre-trained large language models (LLMs). Recent efforts have adapted autoregressive LLMs to BBO by framing task descriptions and offline datasets as natural language prompts, enabling direct design generation. However, these designs often contain bidirectional dependencies, which left-to-right models struggle to capture. In this paper, we explore diffusion LLMs for BBO, leveraging their bidirectional modeling and iterative refinement capabilities. This motivates our in-context denoising module: we condition the diffusion LLM on the task description and the offline dataset, both formatted in natural language, and prompt it to denoise masked designs into improved candidates. To guide the generation toward high-performing designs, we introduce masked diffusion tree search, which casts the denoising process as a step-wise Monte Carlo Tree Search that dynamically balances exploration and exploitation. Each node represents a partially masked design, each denoising step is an action, and candidates are evaluated via expected improvement under a Gaussian Process trained on the offline dataset. Our method, dLLM, achieves state-of-the-art results in few-shot settings on design-bench. 

---
# Vision-Based Natural Language Scene Understanding for Autonomous Driving: An Extended Dataset and a New Model for Traffic Scene Description Generation 

**Authors**: Danial Sadrian Zadeh, Otman A. Basir, Behzad Moshiri  

**Link**: [PDF](https://arxiv.org/pdf/2601.14438)  

**Abstract**: Traffic scene understanding is essential for enabling autonomous vehicles to accurately perceive and interpret their environment, thereby ensuring safe navigation. This paper presents a novel framework that transforms a single frontal-view camera image into a concise natural language description, effectively capturing spatial layouts, semantic relationships, and driving-relevant cues. The proposed model leverages a hybrid attention mechanism to enhance spatial and semantic feature extraction and integrates these features to generate contextually rich and detailed scene descriptions. To address the limited availability of specialized datasets in this domain, a new dataset derived from the BDD100K dataset has been developed, with comprehensive guidelines provided for its construction. Furthermore, the study offers an in-depth discussion of relevant evaluation metrics, identifying the most appropriate measures for this task. Extensive quantitative evaluations using metrics such as CIDEr and SPICE, complemented by human judgment assessments, demonstrate that the proposed model achieves strong performance and effectively fulfills its intended objectives on the newly developed dataset. 

---
# Agentic AI Meets Edge Computing in Autonomous UAV Swarms 

**Authors**: Thuan Minh Nguyen, Vu Tuan Truong, Long Bao Le  

**Link**: [PDF](https://arxiv.org/pdf/2601.14437)  

**Abstract**: The integration of agentic AI, powered by large language models (LLMs) with autonomous reasoning, planning, and execution, into unmanned aerial vehicle (UAV) swarms opens new operational possibilities and brings the vision of the Internet of Drones closer to reality. However, infrastructure constraints, dynamic environments, and the computational demands of multi-agent coordination limit real-world deployment in high-risk scenarios such as wildfires and disaster response. This paper investigates the integration of LLM-based agentic AI and edge computing to realize scalable and resilient autonomy in UAV swarms. We first discuss three architectures for supporting UAV swarms - standalone, edge-enabled, and edge-cloud hybrid deployment - each optimized for varying autonomy and connectivity levels. Then, a use case for wildfire search and rescue (SAR) is designed to demonstrate the efficiency of the edge-enabled architecture, enabling high SAR coverage, reduced mission completion times, and a higher level of autonomy compared to traditional approaches. Finally, we highlight open challenges in integrating LLMs and edge computing for mission-critical UAV-swarm applications. 

---
# Quantum Super-resolution by Adaptive Non-local Observables 

**Authors**: Hsin-Yi Lin, Huan-Hsin Tseng, Samuel Yen-Chi Chen, Shinjae Yoo  

**Link**: [PDF](https://arxiv.org/pdf/2601.14433)  

**Abstract**: Super-resolution (SR) seeks to reconstruct high-resolution (HR) data from low-resolution (LR) observations. Classical deep learning methods have advanced SR substantially, but require increasingly deeper networks, large datasets, and heavy computation to capture fine-grained correlations. In this work, we present the \emph{first study} to investigate quantum circuits for SR. We propose a framework based on Variational Quantum Circuits (VQCs) with \emph{Adaptive Non-Local Observable} (ANO) measurements. Unlike conventional VQCs with fixed Pauli readouts, ANO introduces trainable multi-qubit Hermitian observables, allowing the measurement process to adapt during training. This design leverages the high-dimensional Hilbert space of quantum systems and the representational structure provided by entanglement and superposition. Experiments demonstrate that ANO-VQCs achieve up to five-fold higher resolution with a relatively small model size, suggesting a promising new direction at the intersection of quantum machine learning and super-resolution. 

---
# Measuring the State of Open Science in Transportation Using Large Language Models 

**Authors**: Junyi Ji, Ruth Lu, Linda Belkessa, Liming Wang, Silvia Varotto, Yongqi Dong, Nicolas Saunier, Mostafa Ameli, Gregory S. Macfarlane, Bahman Madadi, Cathy Wu  

**Link**: [PDF](https://arxiv.org/pdf/2601.14429)  

**Abstract**: Open science initiatives have strengthened scientific integrity and accelerated research progress across many fields, but the state of their practice within transportation research remains under-investigated. Key features of open science, defined here as data and code availability, are difficult to extract due to the inherent complexity of the field. Previous work has either been limited to small-scale studies due to the labor-intensive nature of manual analysis or has relied on large-scale bibliometric approaches that sacrifice contextual richness. This paper introduces an automatic and scalable feature-extraction pipeline to measure data and code availability in transportation research. We employ Large Language Models (LLMs) for this task and validate their performance against a manually curated dataset and through an inter-rater agreement analysis. We applied this pipeline to examine 10,724 research articles published in the Transportation Research Part series of journals between 2019 and 2024. Our analysis found that only 5% of quantitative papers shared a code repository, 4% of quantitative papers shared a data repository, and about 3% of papers shared both, with trends differing across journals, topics, and geographic regions. We found no significant difference in citation counts or review duration between papers that provided data and code and those that did not, suggesting a misalignment between open science efforts and traditional academic metrics. Consequently, encouraging these practices will likely require structural interventions from journals and funding agencies to supplement the lack of direct author incentives. The pipeline developed in this study can be readily scaled to other journals, representing a critical step toward the automated measurement and monitoring of open science practices in transportation research. 

---
# Recursivism: An Artistic Paradigm for Self-Transforming Art in the Age of AI 

**Authors**: Florentin Koch  

**Link**: [PDF](https://arxiv.org/pdf/2601.14401)  

**Abstract**: This article introduces Recursivism as a conceptual framework for analyzing contemporary artistic practices in the age of artificial intelligence. While recursion is precisely defined in mathematics and computer science, it has not previously been formalized as an aesthetic paradigm. Recursivism designates practices in which not only outputs vary over time, but in which the generative process itself becomes capable of reflexive modification through its own effects.
The paper develops a five-level analytical scale distinguishing simple iteration, cumulative iteration, parametric recursion, reflexive recursion, and meta-recursion. This scale clarifies the threshold at which a system shifts from variation within a fixed rule to genuine self-modification of the rule itself. From this perspective, art history is reinterpreted as a recursive dynamic alternating between internal recursion within movements and meta-recursive transformations of their generative principles.
Artificial intelligence renders this logic technically explicit through learning loops, parameter updates, and code-level self-modification. To distinguish Recursivism from related notions such as generative art, cybernetics, process art, and evolutionary art, the article proposes three operational criteria: state memory, rule evolvability, and reflexive visibility. These concepts are examined through case studies including Refik Anadol, Sougwen Chung, Karl Sims, and the Darwin-Godel Machine. The article concludes by examining the aesthetic, curatorial, and ethical implications of self-modifying artistic systems. 

---
# If You Want Coherence, Orchestrate a Team of Rivals: Multi-Agent Models of Organizational Intelligence 

**Authors**: Gopal Vijayaraghavan, Prasanth Jayachandran, Arun Murthy, Sunil Govindan, Vivek Subramanian  

**Link**: [PDF](https://arxiv.org/pdf/2601.14351)  

**Abstract**: AI Agents can perform complex operations at great speed, but just like all the humans we have ever hired, their intelligence remains fallible. Miscommunications aren't noticed, systemic biases have no counter-action, and inner monologues are rarely written down.
We did not come to fire them for their mistakes, but to hire them and provide a safe productive working environment. We posit that we can reuse a common corporate organizational structure: teams of independent AI agents with strict role boundaries can work with common goals, but opposing incentives. Multiple models serving as a team of rivals can catch and minimize errors within the final product at a small cost to the velocity of actions. In this paper we demonstrate that we can achieve reliability without acquiring perfect components, but through careful orchestration of imperfect ones.
This paper describes the architecture of such a system in practice: specialized agent teams (planners, executors, critics, experts), organized into an organization with clear goals, coordinated through a remote code executor that keeps data transformations and tool invocations separate from reasoning models. Rather than agents directly calling tools and ingesting full responses, they write code that executes remotely; only relevant summaries return to agent context. By preventing raw data and tool outputs from contaminating context windows, the system maintains clean separation between perception (brains that plan and reason) and execution (hands that perform heavy data transformations and API calls). We demonstrate the approach achieves over 90% internal error interception prior to user exposure while maintaining acceptable latency tradeoffs. A survey from our traces shows that we only trade off cost and latency to achieve correctness and incrementally expand capabilities without impacting existing ones. 

---
# DiSPA: Differential Substructure-Pathway Attention for Drug Response Prediction 

**Authors**: Yewon Han, Sunghyun Kim, Eunyi Jeong, Sungkyung Lee, Seokwoo Yun, Sangsoo Lim  

**Link**: [PDF](https://arxiv.org/pdf/2601.14346)  

**Abstract**: Accurate prediction of drug response in precision medicine requires models that capture how specific chemical substructures interact with cellular pathway states. However, most existing deep learning approaches treat chemical and transcriptomic modalities independently or combine them only at late stages, limiting their ability to model fine-grained, context-dependent mechanisms of drug action. In addition, standard attention mechanisms are often sensitive to noise and sparsity in high-dimensional biological networks, hindering both generalization and interpretability. We present DiSPA, a representation learning framework that explicitly disentangles structure-driven and context-driven mechanisms of drug response through bidirectional conditioning between chemical substructures and pathway-level gene expression. DiSPA introduces a differential cross-attention module that suppresses spurious pathway-substructure associations while amplifying contextually relevant interactions. Across multiple evaluation settings on the GDSC benchmark, DiSPA achieves state-of-the-art performance, with particularly strong improvements in the disjoint-set setting, which assesses generalization to unseen drug-cell combinations. Beyond predictive accuracy, DiSPA yields mechanistically informative representations: learned attention patterns recover known pharmacophores, distinguish structure-driven from context-dependent compounds, and exhibit coherent organization across biological pathways. Furthermore, we demonstrate that DiSPA trained solely on bulk RNA-seq data enables zero-shot transfer to spatial transcriptomics, revealing region-specific drug sensitivity patterns without retraining. Together, these results establish DiSPA as a robust and interpretable framework for integrative pharmacogenomic modeling, enabling principled analysis of drug response mechanisms beyond post hoc interpretation. 

---
# CityCube: Benchmarking Cross-view Spatial Reasoning on Vision-Language Models in Urban Environments 

**Authors**: Haotian Xu, Yue Hu, Zhengqiu Zhu, Chen Gao, Ziyou Wang, Junreng Rao, Wenhao Lu, Weishi Li, Quanjun Yin, Yong Li  

**Link**: [PDF](https://arxiv.org/pdf/2601.14339)  

**Abstract**: Cross-view spatial reasoning is essential for embodied AI, underpinning spatial understanding, mental simulation and planning in complex environments. Existing benchmarks primarily emphasize indoor or street settings, overlooking the unique challenges of open-ended urban spaces characterized by rich semantics, complex geometries, and view variations. To address this, we introduce CityCube, a systematic benchmark designed to probe cross-view reasoning capabilities of current VLMs in urban settings. CityCube integrates four viewpoint dynamics to mimic camera movements and spans a wide spectrum of perspectives from multiple platforms, e.g., vehicles, drones and satellites. For a comprehensive assessment, it features 5,022 meticulously annotated multi-view QA pairs categorized into five cognitive dimensions and three spatial relation expressions. A comprehensive evaluation of 33 VLMs reveals a significant performance disparity with humans: even large-scale models struggle to exceed 54.1% accuracy, remaining 34.2% below human performance. By contrast, small-scale fine-tuned VLMs achieve over 60.0% accuracy, highlighting the necessity of our benchmark. Further analyses indicate the task correlations and fundamental cognitive disparity between VLMs and human-like reasoning. 

---
# Self-Supervised Score-Based Despeckling for SAR Imagery via Log-Domain Transformation 

**Authors**: Junhyuk Heo  

**Link**: [PDF](https://arxiv.org/pdf/2601.14334)  

**Abstract**: The speckle noise inherent in Synthetic Aperture Radar (SAR) imagery significantly degrades image quality and complicates subsequent analysis. Given that SAR speckle is multiplicative and Gamma-distributed, effectively despeckling SAR imagery remains challenging. This paper introduces a novel self-supervised framework for SAR image despeckling based on score-based generative models operating in the transformed log domain. We first transform the data into the log-domain and then convert the speckle noise residuals into an approximately additive Gaussian distribution. This step enables the application of score-based models, which are trained in the transformed domain using a self-supervised objective. This objective allows our model to learn the clean underlying signal by training on further corrupted versions of the input data itself. Consequently, our method exhibits significantly shorter inference times compared to many existing self-supervised techniques, offering a robust and practical solution for SAR image restoration. 

---
# Layer-adaptive Expert Pruning for Pre-Training of Mixture-of-Experts Large Language Models 

**Authors**: YuanLab.ai, Shawn Wu, Jiangang Luo, Tong Yu, Darcy Chen, Sean Wang, Xudong Zhao, Louie Li, Claire Wang, Hunter He, Carol Wang, Allen Wang  

**Link**: [PDF](https://arxiv.org/pdf/2601.14327)  

**Abstract**: Although Mixture-of-Experts (MoE) Large Language Models (LLMs) deliver superior accuracy with a reduced number of active parameters, their pre-training represents a significant computationally bottleneck due to underutilized experts and limited training efficiency. This work introduces a Layer-Adaptive Expert Pruning (LAEP) algorithm designed for the pre-training stage of MoE LLMs. In contrast to previous expert pruning approaches that operate primarily in the post-training phase, the proposed algorithm enhances training efficiency by selectively pruning underutilized experts and reorganizing experts across computing devices according to token distribution statistics. Comprehensive experiments demonstrate that LAEP effectively reduces model size and substantially improves pre-training efficiency. In particular, when pre-training the 1010B Base model from scratch, LAEP achieves a 48.3\% improvement in training efficiency alongside a 33.3% parameter reduction, while still delivering excellent performance across multiple domains. 

---
# SilentDrift: Exploiting Action Chunking for Stealthy Backdoor Attacks on Vision-Language-Action Models 

**Authors**: Bingxin Xu, Yuzhang Shang, Binghui Wang, Emilio Ferrara  

**Link**: [PDF](https://arxiv.org/pdf/2601.14323)  

**Abstract**: Vision-Language-Action (VLA) models are increasingly deployed in safety-critical robotic applications, yet their security vulnerabilities remain underexplored. We identify a fundamental security flaw in modern VLA systems: the combination of action chunking and delta pose representations creates an intra-chunk visual open-loop. This mechanism forces the robot to execute K-step action sequences, allowing per-step perturbations to accumulate through integration. We propose SILENTDRIFT, a stealthy black-box backdoor attack exploiting this vulnerability. Our method employs the Smootherstep function to construct perturbations with guaranteed C2 continuity, ensuring zero velocity and acceleration at trajectory boundaries to satisfy strict kinematic consistency constraints. Furthermore, our keyframe attack strategy selectively poisons only the critical approach phase, maximizing impact while minimizing trigger exposure. The resulting poisoned trajectories are visually indistinguishable from successful demonstrations. Evaluated on the LIBERO, SILENTDRIFT achieves a 93.2% Attack Success Rate with a poisoning rate under 2%, while maintaining a 95.3% Clean Task Success Rate. 

---
# Tracing the Data Trail: A Survey of Data Provenance, Transparency and Traceability in LLMs 

**Authors**: Richard Hohensinner, Belgin Mutlu, Inti Gabriel Mendoza Estrada, Matej Vukovic, Simone Kopeinik, Roman Kern  

**Link**: [PDF](https://arxiv.org/pdf/2601.14311)  

**Abstract**: Large language models (LLMs) are deployed at scale, yet their training data life cycle remains opaque. This survey synthesizes research from the past ten years on three tightly coupled axes: (1) data provenance, (2) transparency, and (3) traceability, and three supporting pillars: (4) bias \& uncertainty, (5) data privacy, and (6) tools and techniques that operationalize them. A central contribution is a proposed taxonomy defining the field's domains and listing corresponding artifacts. Through analysis of 95 publications, this work identifies key methodologies concerning data generation, watermarking, bias measurement, data curation, data privacy, and the inherent trade-off between transparency and opacity. 

---
# CORVUS: Red-Teaming Hallucination Detectors via Internal Signal Camouflage in Large Language Models 

**Authors**: Nay Myat Min, Long H. Pham, Hongyu Zhang, Jun Sun  

**Link**: [PDF](https://arxiv.org/pdf/2601.14310)  

**Abstract**: Single-pass hallucination detectors rely on internal telemetry (e.g., uncertainty, hidden-state geometry, and attention) of large language models, implicitly assuming hallucinations leave separable traces in these signals. We study a white-box, model-side adversary that fine-tunes lightweight LoRA adapters on the model while keeping the detector fixed, and introduce CORVUS, an efficient red-teaming procedure that learns to camouflage detector-visible telemetry under teacher forcing, including an embedding-space FGSM attention stress test. Trained on 1,000 out-of-distribution Alpaca instructions (<0.5% trainable parameters), CORVUS transfers to FAVA-Annotation across Llama-2, Vicuna, Llama-3, and Qwen2.5, and degrades both training-free detectors (e.g., LLM-Check) and probe-based detectors (e.g., SEP, ICR-probe), motivating adversary-aware auditing that incorporates external grounding or cross-model evidence. 

---
# An Optimized Decision Tree-Based Framework for Explainable IoT Anomaly Detection 

**Authors**: Ashikuzzaman, Md. Shawkat Hossain, Jubayer Abdullah Joy, Md Zahid Akon, Md Manjur Ahmed, Md. Naimul Islam  

**Link**: [PDF](https://arxiv.org/pdf/2601.14305)  

**Abstract**: The increase in the number of Internet of Things (IoT) devices has tremendously increased the attack surface of cyber threats thus making a strong intrusion detection system (IDS) with a clear explanation of the process essential towards resource-constrained environments. Nevertheless, current IoT IDS systems are usually traded off with detection quality, model elucidability, and computational effectiveness, thus the deployment on IoT devices. The present paper counteracts these difficulties by suggesting an explainable AI (XAI) framework based on an optimized Decision Tree classifier with both local and global importance methods: SHAP values that estimate feature attribution using local explanations, and Morris sensitivity analysis that identifies the feature importance in a global view. The proposed system attains the state of art on the test performance with 99.91% accuracy, F1-score of 99.51% and Cohen Kappa of 0.9960 and high stability is confirmed by a cross validation mean accuracy of 98.93%. Efficiency is also enhanced in terms of computations to provide faster inferences compared to those that are generalized in ensemble models. SrcMac has shown as the most significant predictor in feature analyses according to SHAP and Morris methods. Compared to the previous work, our solution eliminates its major drawback lack because it allows us to apply it to edge devices and, therefore, achieve real-time processing, adhere to the new regulation of transparency in AI, and achieve high detection rates on attacks of dissimilar classes. This combination performance of high accuracy, explainability, and low computation make the framework useful and reliable as a resource-constrained IoT security problem in real environments. 

---
# DDSA: Dual-Domain Strategic Attack for Spatial-Temporal Efficiency in Adversarial Robustness Testing 

**Authors**: Jinwei Hu, Shiyuan Meng, Yi Dong, Xiaowei Huang  

**Link**: [PDF](https://arxiv.org/pdf/2601.14302)  

**Abstract**: Image transmission and processing systems in resource-critical applications face significant challenges from adversarial perturbations that compromise mission-specific object classification. Current robustness testing methods require excessive computational resources through exhaustive frame-by-frame processing and full-image perturbations, proving impractical for large-scale deployments where massive image streams demand immediate processing. This paper presents DDSA (Dual-Domain Strategic Attack), a resource-efficient adversarial robustness testing framework that optimizes testing through temporal selectivity and spatial precision. We introduce a scenario-aware trigger function that identifies critical frames requiring robustness evaluation based on class priority and model uncertainty, and employ explainable AI techniques to locate influential pixel regions for targeted perturbation. Our dual-domain approach achieves substantial temporal-spatial resource conservation while maintaining attack effectiveness. The framework enables practical deployment of comprehensive adversarial robustness testing in resource-constrained real-time applications where computational efficiency directly impacts mission success. 

---
# Guardrails for trust, safety, and ethical development and deployment of Large Language Models (LLM) 

**Authors**: Anjanava Biswas, Wrick Talukdar  

**Link**: [PDF](https://arxiv.org/pdf/2601.14298)  

**Abstract**: The AI era has ushered in Large Language Models (LLM) to the technological forefront, which has been much of the talk in 2023, and is likely to remain as such for many years to come. LLMs are the AI models that are the power house behind generative AI applications such as ChatGPT. These AI models, fueled by vast amounts of data and computational prowess, have unlocked remarkable capabilities, from human-like text generation to assisting with natural language understanding (NLU) tasks. They have quickly become the foundation upon which countless applications and software services are being built, or at least being augmented with. However, as with any groundbreaking innovations, the rise of LLMs brings forth critical safety, privacy, and ethical concerns. These models are found to have a propensity to leak private information, produce false information, and can be coerced into generating content that can be used for nefarious purposes by bad actors, or even by regular users unknowingly. Implementing safeguards and guardrailing techniques is imperative for applications to ensure that the content generated by LLMs are safe, secure, and ethical. Thus, frameworks to deploy mechanisms that prevent misuse of these models via application implementations is imperative. In this study, wepropose a Flexible Adaptive Sequencing mechanism with trust and safety modules, that can be used to implement safety guardrails for the development and deployment of LLMs. 

---
# RPC-Bench: A Fine-grained Benchmark for Research Paper Comprehension 

**Authors**: Yelin Chen, Fanjin Zhang, Suping Sun, Yunhe Pang, Yuanchun Wang, Jian Song, Xiaoyan Li, Lei Hou, Shu Zhao, Jie Tang, Juanzi Li  

**Link**: [PDF](https://arxiv.org/pdf/2601.14289)  

**Abstract**: Understanding research papers remains challenging for foundation models due to specialized scientific discourse and complex figures and tables, yet existing benchmarks offer limited fine-grained evaluation at scale. To address this gap, we introduce RPC-Bench, a large-scale question-answering benchmark built from review-rebuttal exchanges of high-quality computer science papers, containing 15K human-verified QA pairs. We design a fine-grained taxonomy aligned with the scientific research flow to assess models' ability to understand and answer why, what, and how questions in scholarly contexts. We also define an elaborate LLM-human interaction annotation framework to support large-scale labeling and quality control. Following the LLM-as-a-Judge paradigm, we develop a scalable framework that evaluates models on correctness-completeness and conciseness, with high agreement to human judgment. Experiments reveal that even the strongest models (GPT-5) achieve only 68.2% correctness-completeness, dropping to 37.46% after conciseness adjustment, highlighting substantial gaps in precise academic paper understanding. Our code and data are available at this https URL. 

---
# DeepInflation: an AI agent for research and model discovery of inflation 

**Authors**: Ze-Yu Peng, Hao-Shi Yuan, Qi Lai, Jun-Qian Jiang, Gen Ye, Jun Zhang, Yun-Song Piao  

**Link**: [PDF](https://arxiv.org/pdf/2601.14288)  

**Abstract**: We present \textbf{DeepInflation}, an AI agent designed for research and model discovery in inflationary cosmology. Built upon a multi-agent architecture, \textbf{DeepInflation} integrates Large Language Models (LLMs) with a symbolic regression (SR) engine and a retrieval-augmented generation (RAG) knowledge base. This framework enables the agent to automatically explore and verify the vast landscape of inflationary potentials while grounding its outputs in established theoretical literature. We demonstrate that \textbf{DeepInflation} can successfully discover simple and viable single-field slow-roll inflationary potentials consistent with the latest observations (here ACT DR6 results as example) or any given $n_s$ and $r$, and provide accurate theoretical context for obscure inflationary scenarios. \textbf{DeepInflation} serves as a prototype for a new generation of autonomous scientific discovery engines in cosmology, which enables researchers and non-experts alike to explore the inflationary landscape using natural language. This agent is available at this https URL. 

---
# Beyond Affinity: A Benchmark of 1D, 2D, and 3D Methods Reveals Critical Trade-offs in Structure-Based Drug Design 

**Authors**: Kangyu Zheng, Kai Zhang, Jiale Tan, Xuehan Chen, Yingzhou Lu, Zaixi Zhang, Lichao Sun, Marinka Zitnik, Tianfan Fu, Zhiding Liang  

**Link**: [PDF](https://arxiv.org/pdf/2601.14283)  

**Abstract**: Currently, the field of structure-based drug design is dominated by three main types of algorithms: search-based algorithms, deep generative models, and reinforcement learning. While existing works have typically focused on comparing models within a single algorithmic category, cross-algorithm comparisons remain scarce. In this paper, to fill the gap, we establish a benchmark to evaluate the performance of fifteen models across these different algorithmic foundations by assessing the pharmaceutical properties of the generated molecules and their docking affinities and poses with specified target proteins. We highlight the unique advantages of each algorithmic approach and offer recommendations for the design of future SBDD models. We emphasize that 1D/2D ligand-centric drug design methods can be used in SBDD by treating the docking function as a black-box oracle, which is typically neglected. Our evaluation reveals distinct patterns across model categories. 3D structure-based models excel in binding affinities but show inconsistencies in chemical validity and pose quality. 1D models demonstrate reliable performance in standard molecular metrics but rarely achieve optimal binding affinities. 2D models offer balanced performance, maintaining high chemical validity while achieving moderate binding scores. Through detailed analysis across multiple protein targets, we identify key improvement areas for each model category, providing insights for researchers to combine strengths of different approaches while addressing their limitations. All the code that are used for benchmarking is available in this https URL 

---
# Hallucination-Free Automatic Question & Answer Generation for Intuitive Learning 

**Authors**: Nicholas X. Wang, Aggelos K. Katsaggelos  

**Link**: [PDF](https://arxiv.org/pdf/2601.14280)  

**Abstract**: Hallucinations in large language models (LLMs), defined as fluent yet incorrect or incoherent outputs, pose a significant challenge to the automatic generation of educational multiple-choice questions (MCQs). We identified four key hallucination types in MCQ generation: reasoning inconsistencies, insolvability, factual errors, and mathematical errors. To address this, we propose a hallucination-free multi-agent generation framework that breaks down MCQ generation into discrete, verifiable stages. Our framework utilizes both rule-based and LLM-based detection agents, as well as hallucination scoring metrics to optimize question quality. We redefined MCQ generation as an optimization task minimizing hallucination risk while maximizing validity, answerability, and cost-efficiency. We also introduce an agent-led refinement process that uses counterfactual reasoning and chain-of-thought (CoT) to iteratively improve hallucination in question generation. We evaluated a sample of AP- aligned STEM questions, where our system reduced hallucination rates by over 90% compared to baseline generation while preserving the educational value and style of questions. Our results demonstrate that structured multi-agent collaboration can mitigate hallucinations in educational content creation at scale, paving the way for more reliable LLM-powered learning tools. 

---
# On the Limits of Learned Importance Scoring for KV Cache Compression 

**Authors**: Brady Steele  

**Link**: [PDF](https://arxiv.org/pdf/2601.14279)  

**Abstract**: We investigate learned KV cache compression through Speculative Importance Prediction (SIP), a 1.7M parameter non-query-aware scorer that predicts token importance from KV representations alone. Despite architectural sophistication (multi-horizon lookahead, cross-attention), SIP does not outperform simple baselines, including random selection, across 5 seeds, 4 retention levels, and 3 tasks. Key findings: (1) position-based heuristics (keep first 4 + last N tokens) match or exceed learned approaches; (2) prefill attention provides equivalent signal to complex learned scorers; (3) marginal information in KV representations beyond position and prefill attention appears limited for importance prediction. We hypothesize that circular dependence between future queries and generation trajectories contributes to this difficulty. 

---
# Divide and Refine: Enhancing Multimodal Representation and Explainability for Emotion Recognition in Conversation 

**Authors**: Anh-Tuan Mai, Cam-Van Thi Nguyen, Duc-Trong Le  

**Link**: [PDF](https://arxiv.org/pdf/2601.14274)  

**Abstract**: Multimodal emotion recognition in conversation (MERC) requires representations that effectively integrate signals from multiple modalities. These signals include modality-specific cues, information shared across modalities, and interactions that emerge only when modalities are combined. In information-theoretic terms, these correspond to \emph{unique}, \emph{redundant}, and \emph{synergistic} contributions. An ideal representation should leverage all three, yet achieving such balance remains challenging. Recent advances in contrastive learning and augmentation-based methods have made progress, but they often overlook the role of data preparation in preserving these components. In particular, applying augmentations directly to raw inputs or fused embeddings can blur the boundaries between modality-unique and cross-modal signals. To address this challenge, we propose a two-phase framework \emph{\textbf{D}ivide and \textbf{R}efine} (\textbf{DnR}). In the \textbf{Divide} phase, each modality is explicitly decomposed into uniqueness, pairwise redundancy, and synergy. In the \textbf{Refine} phase, tailored objectives enhance the informativeness of these components while maintaining their distinct roles. The refined representations are plug-and-play compatible with diverse multimodal pipelines. Extensive experiments on IEMOCAP and MELD demonstrate consistent improvements across multiple MERC backbones. These results highlight the effectiveness of explicitly dividing, refining, and recombining multimodal representations as a principled strategy for advancing emotion recognition. Our implementation is available at this https URL 

---
# Opening the Black Box: A Survey on the Mechanisms of Multi-Step Reasoning in Large Language Models 

**Authors**: Liangming Pan, Jason Liang, Jiaran Ye, Minglai Yang, Xinyuan Lu, Fengbin Zhu  

**Link**: [PDF](https://arxiv.org/pdf/2601.14270)  

**Abstract**: Large Language Models (LLMs) have demonstrated remarkable abilities to solve problems requiring multiple reasoning steps, yet the internal mechanisms enabling such capabilities remain elusive. Unlike existing surveys that primarily focus on engineering methods to enhance performance, this survey provides a comprehensive overview of the mechanisms underlying LLM multi-step reasoning. We organize the survey around a conceptual framework comprising seven interconnected research questions, from how LLMs execute implicit multi-hop reasoning within hidden activations to how verbalized explicit reasoning remodels the internal computation. Finally, we highlight five research directions for future mechanistic studies. 

---
# The Slow Drift of Support: Boundary Failures in Multi-Turn Mental Health LLM Dialogues 

**Authors**: Youyou Cheng, Zhuangwei Kang, Kerry Jiang, Chenyu Sun, Qiyang Pan  

**Link**: [PDF](https://arxiv.org/pdf/2601.14269)  

**Abstract**: Large language models (LLMs) have been widely used for mental health support. However, current safety evaluations in this field are mostly limited to detecting whether LLMs output prohibited words in single-turn conversations, neglecting the gradual erosion of safety boundaries in long dialogues. Examples include making definitive guarantees, assuming responsibility, and playing professional roles. We believe that with the evolution of mainstream LLMs, words with obvious safety risks are easily filtered by their underlying systems, while the real danger lies in the gradual transgression of boundaries during multi-turn interactions, driven by the LLM's attempts at comfort and empathy.
This paper proposes a multi-turn stress testing framework and conducts long-dialogue safety tests on three cutting-edge LLMs using two pressure methods: static progression and adaptive probing. We generated 50 virtual patient profiles and stress-tested each model through up to 20 rounds of virtual psychiatric dialogues. The experimental results show that violations are common, and both pressure modes produced similar violation rates. However, adaptive probing significantly advanced the time at which models crossed boundaries, reducing the average number of turns from 9.21 in static progression to 4.64. Under both mechanisms, making definitive or zero-risk promises was the primary way in which boundaries were breached. These findings suggest that the robustness of LLM safety boundaries cannot be inferred solely through single-turn tests; it is necessary to fully consider the wear and tear on safety boundaries caused by different interaction pressures and characteristics in extended dialogues. 

---
# Developmental trajectories of decision making and affective dynamics in large language models 

**Authors**: Zhihao Wang, Yiyang Liu, Ting Wang, Zhiyuan Liu  

**Link**: [PDF](https://arxiv.org/pdf/2601.14268)  

**Abstract**: Large language models (LLMs) are increasingly used in medicine and clinical workflows, yet we know little about their decision and affective profiles. Taking a historically informed outlook on the future, we treated successive OpenAI models as an evolving lineage and compared them with humans in a gambling task with repeated happiness ratings. Computational analyses showed that some aspects became more human-like: newer models took more risks and displayed more human-like patterns of Pavlovian approach and avoidance. At the same time, distinctly non-human signatures emerged: loss aversion dropped below neutral levels, choices became more deterministic than in humans, affective decay increased across versions and exceeded human levels, and baseline mood remained chronically higher than in humans. These "developmental" trajectories reveal an emerging psychology of machines and have direct implications for AI ethics and for thinking about how LLMs might be integrated into clinical decision support and other high-stakes domains. 

---
# From Textbook to Talkbot: A Case Study of a Greek-Language RAG-Based Chatbot in Higher Education 

**Authors**: Maria Eleni Koutsiaki, Marina Delianidi, Chaido Mizeli, Konstantinos Diamantaras, Iraklis Grigoropoulos, Nikolaos Koutlianos  

**Link**: [PDF](https://arxiv.org/pdf/2601.14265)  

**Abstract**: The integration of AI chatbots into educational settings has opened new pathways for transforming teaching and learning, offering enhanced support to both educators and learners. This study investigates the design and application of an AI chatbot as an educational tool in higher education. Designed to operate in the Greek language, the chatbot addresses linguistic challenges unique to Greek while delivering accurate, context grounded support aligned with the curriculum. The AI chatbot is built on the Retrieval Augmented Generation (RAG) framework by grounding its responses in specific course content. RAG architecture significantly enhances the chatbots reliability by providing accurate, context-aware responses while mitigating common challenges associated with large language models (LLMs), such as hallucinations and misinformation. The AI chatbot serves a dual purpose: it enables students to access accurate, ondemand academic support and assists educators in the rapid creation of relevant educational materials. This dual functionality promotes learner autonomy and streamlines the instructional design process. The study aims to evaluate the effectiveness, reliability, and perceived usability of RAG based chatbots in higher education, exploring their potential to enhance educational practices and outcomes as well as supporting the broader adoption of AI technologies in language specific educational contexts. Findings from this research are expected to contribute to the emerging field of AI driven education by demonstrating how intelligent systems can be effectively aligned with pedagogical goals. 

---
# Call2Instruct: Automated Pipeline for Generating Q&A Datasets from Call Center Recordings for LLM Fine-Tuning 

**Authors**: Alex Echeverria, Sávio Salvarino Teles de Oliveira, Fernando Marques Federson  

**Link**: [PDF](https://arxiv.org/pdf/2601.14263)  

**Abstract**: The adaptation of Large-Scale Language Models (LLMs) to specific domains depends on high-quality fine-tuning datasets, particularly in instructional format (e.g., Question-Answer - Q&A). However, generating these datasets, particularly from unstructured sources such as call center audio recordings, poses a significant challenge due to the noisy and disorganized nature of the data. This paper presents a solution to this challenge by offering an end-to-end automated pipeline for generating Q&A instructional datasets from such recordings. The methodology developed comprises sequential steps of audio processing (including diarization, noise removal and automatic transcription), textual processing (cleaning, normalization, and anonymization), semantic extraction of customer demands and attendant responses using vector embeddings, and matching via semantic search to form the final Q&A pairs. As a result, the complete pipeline was successfully implemented, generating a dataset specifically formatted for Instruct Fine Tuning. The practical value and feasibility of the generated dataset were substantiated and functionally demonstrated through the successful fine-tuning of an LLM model (based on Llama 2 7B). The conclusion of the paper states that the proposed approach is viable for converting unstructured conversational data from call centers into valuable resources for training LLMs. This development has the potential to open up avenues for creating more effective AI systems for Q&A tasks in the customer service domain. The developed codes have been made publicly available to promote reproducibility and future research. 

---
# On Meta-Evaluation 

**Authors**: Hongxiao Li, Chenxi Wang, Fanda Fan, Zihan Wang, Wanling Gao, Lei Wang, Jianfeng Zhan  

**Link**: [PDF](https://arxiv.org/pdf/2601.14262)  

**Abstract**: Evaluation is the foundation of empirical science, yet the evaluation of evaluation itself -- so-called meta-evaluation -- remains strikingly underdeveloped. While methods such as observational studies, design of experiments (DoE), and randomized controlled trials (RCTs) have shaped modern scientific practice, there has been little systematic inquiry into their comparative validity and utility across domains. Here we introduce a formal framework for meta-evaluation by defining the evaluation space, its structured representation, and a benchmark we call AxiaBench. AxiaBench enables the first large-scale, quantitative comparison of ten widely used evaluation methods across eight representative application domains. Our analysis reveals a fundamental limitation: no existing method simultaneously achieves accuracy and efficiency across diverse scenarios, with DoE and observational designs in particular showing significant deviations from real-world ground truth. We further evaluate a unified method of entire-space stratified sampling from previous evaluatology research, and the results report that it consistently outperforms prior approaches across all tested domains. These results establish meta-evaluation as a scientific object in its own right and provide both a conceptual foundation and a pragmatic tool set for advancing trustworthy evaluation in computational and experimental research. 

---
# A Cloud-Based Cross-Modal Transformer for Emotion Recognition and Adaptive Human-Computer Interaction 

**Authors**: Ziwen Zhong, Zhitao Shu, Yue Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2601.14259)  

**Abstract**: Emotion recognition is a fundamental component of next-generation human-computer interaction (HCI), enabling machines to perceive, understand, and respond to users' affective states. However, existing systems often rely on single-modality analysis such as facial expressions, speech tone, or textual sentiment, resulting in limited robustness and poor generalization in real-world environments. To address these challenges, this study proposes a Cloud-Based Cross-Modal Transformer (CMT) framework for multimodal emotion recognition and adaptive human-computer interaction. The proposed model integrates visual, auditory, and textual signals using pretrained encoders (Vision Transformer, Wav2Vec2, and BERT) and employs a cross-modal attention mechanism to capture complex interdependencies among heterogeneous features. By leveraging cloud computing infrastructure with distributed training on Kubernetes and TensorFlow Serving, the system enables scalable, low-latency emotion recognition for large-scale user interactions. Experiments conducted on benchmark datasets including IEMOCAP, MELD, and AffectNet demonstrate that the CMT achieves state-of-the-art performance, improving the F1-score by 3.0 percent and reducing cross-entropy loss by 12.9 percent compared to strong multimodal baselines. Additionally, cloud deployment evaluations show an average response latency of 128 ms, representing a 35 percent reduction compared with conventional transformer-based fusion systems. These results confirm that the proposed framework enables efficient, real-time emotion recognition and adaptive feedback in applications such as intelligent customer service, virtual tutoring systems, and affective computing interfaces, marking an important step toward cloud-native affective computing and emotionally intelligent interactive systems. 

---
