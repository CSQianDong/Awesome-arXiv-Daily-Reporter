# Same Content, Different Answers: Cross-Modal Inconsistency in MLLMs 

**Authors**: Angela van Sprang, Laurens Samson, Ana Lucic, Erman Acar, Sennay Ghebreab, Yuki M. Asano  

**Link**: [PDF](https://arxiv.org/pdf/2512.08923)  

**Abstract**: We introduce two new benchmarks REST and REST+(Render-Equivalence Stress Tests) to enable systematic evaluation of cross-modal inconsistency in multimodal large language models (MLLMs). MLLMs are trained to represent vision and language in the same embedding space, yet they cannot perform the same tasks in both modalities. Our benchmarks contain samples with the same semantic information in three modalities (image, text, mixed) and we show that state-of-the-art MLLMs cannot consistently reason over these different modalities. We evaluate 15 MLLMs and find that the degree of modality inconsistency varies substantially, even when accounting for problems with text recognition (OCR). Neither rendering text as image nor rendering an image as text solves the inconsistency. Even if OCR is correct, we find that visual characteristics (text colour and resolution, but not font) and the number of vision tokens have an impact on model performance. Finally, we find that our consistency score correlates with the modality gap between text and images, highlighting a mechanistic interpretation of cross-modal inconsistent MLLMs. 

---
# EcomBench: Towards Holistic Evaluation of Foundation Agents in E-commerce 

**Authors**: Rui Min, Zile Qiao, Ze Xu, Jiawen Zhai, Wenyu Gao, Xuanzhong Chen, Haozhen Sun, Zhen Zhang, Xinyu Wang, Hong Zhou, Wenbiao Yin, Xuan Zhou, Yong Jiang, Haicheng Liu, Liang Ding, Ling Zou, Yi R., Fung, Yalong Li, Pengjun Xie  

**Link**: [PDF](https://arxiv.org/pdf/2512.08868)  

**Abstract**: Foundation agents have rapidly advanced in their ability to reason and interact with real environments, making the evaluation of their core capabilities increasingly important. While many benchmarks have been developed to assess agent performance, most concentrate on academic settings or artificially designed scenarios while overlooking the challenges that arise in real applications. To address this issue, we focus on a highly practical real-world setting, the e-commerce domain, which involves a large volume of diverse user interactions, dynamic market conditions, and tasks directly tied to real decision-making processes. To this end, we introduce EcomBench, a holistic E-commerce Benchmark designed to evaluate agent performance in realistic e-commerce environments. EcomBench is built from genuine user demands embedded in leading global e-commerce ecosystems and is carefully curated and annotated through human experts to ensure clarity, accuracy, and domain relevance. It covers multiple task categories within e-commerce scenarios and defines three difficulty levels that evaluate agents on key capabilities such as deep information retrieval, multi-step reasoning, and cross-source knowledge integration. By grounding evaluation in real e-commerce contexts, EcomBench provides a rigorous and dynamic testbed for measuring the practical capabilities of agents in modern e-commerce. 

---
# Interpolation in Knowledge Representation 

**Authors**: Jean Christoph Jung, Patrick Koopmann, Matthias Knorr  

**Link**: [PDF](https://arxiv.org/pdf/2512.08833)  

**Abstract**: Craig interpolation and uniform interpolation have many applications in knowledge representation, including explainability, forgetting, modularization and reuse, and even learning. At the same time, many relevant knowledge representation formalisms do in general not have Craig or uniform interpolation, and computing interpolants in practice is challenging. We have a closer look at two prominent knowledge representation formalisms, description logics and logic programming, and discuss theoretical results and practical methods for computing interpolants. 

---
# CARLoS: Retrieval via Concise Assessment Representation of LoRAs at Scale 

**Authors**: Shahar Sarfaty, Adi Haviv, Uri Hacohen, Niva Elkin-Koren, Roi Livni, Amit H. Bermano  

**Link**: [PDF](https://arxiv.org/pdf/2512.08826)  

**Abstract**: The rapid proliferation of generative components, such as LoRAs, has created a vast but unstructured ecosystem. Existing discovery methods depend on unreliable user descriptions or biased popularity metrics, hindering usability. We present CARLoS, a large-scale framework for characterizing LoRAs without requiring additional metadata. Analyzing over 650 LoRAs, we employ them in image generation over a variety of prompts and seeds, as a credible way to assess their behavior. Using CLIP embeddings and their difference to a base-model generation, we concisely define a three-part representation: Directions, defining semantic shift; Strength, quantifying the significance of the effect; and Consistency, quantifying how stable the effect is. Using these representations, we develop an efficient retrieval framework that semantically matches textual queries to relevant LoRAs while filtering overly strong or unstable ones, outperforming textual baselines in automated and human evaluations. While retrieval is our primary focus, the same representation also supports analyses linking Strength and Consistency to legal notions of substantiality and volition, key considerations in copyright, positioning CARLoS as a practical system with broader relevance for LoRA analysis. 

---
# A Practical Guide for Designing, Developing, and Deploying Production-Grade Agentic AI Workflows 

**Authors**: Eranga Bandara, Ross Gore, Peter Foytik, Sachin Shetty, Ravi Mukkamala, Abdul Rahman, Xueping Liang, Safdar H. Bouk, Amin Hass, Sachini Rajapakse, Ng Wee Keong, Kasun De Zoysa, Aruna Withanage, Nilaan Loganathan  

**Link**: [PDF](https://arxiv.org/pdf/2512.08769)  

**Abstract**: Agentic AI marks a major shift in how autonomous systems reason, plan, and execute multi-step tasks. Unlike traditional single model prompting, agentic workflows integrate multiple specialized agents with different Large Language Models(LLMs), tool-augmented capabilities, orchestration logic, and external system interactions to form dynamic pipelines capable of autonomous decision-making and action. As adoption accelerates across industry and research, organizations face a central challenge: how to design, engineer, and operate production-grade agentic AI workflows that are reliable, observable, maintainable, and aligned with safety and governance requirements. This paper provides a practical, end-to-end guide for designing, developing, and deploying production-quality agentic AI systems. We introduce a structured engineering lifecycle encompassing workflow decomposition, multi-agent design patterns, Model Context Protocol(MCP), and tool integration, deterministic orchestration, Responsible-AI considerations, and environment-aware deployment strategies. We then present nine core best practices for engineering production-grade agentic AI workflows, including tool-first design over MCP, pure-function invocation, single-tool and single-responsibility agents, externalized prompt management, Responsible-AI-aligned model-consortium design, clean separation between workflow logic and MCP servers, containerized deployment for scalable operations, and adherence to the Keep it Simple, Stupid (KISS) principle to maintain simplicity and robustness. To demonstrate these principles in practice, we present a comprehensive case study: a multimodal news-analysis and media-generation workflow. By combining architectural guidance, operational patterns, and practical implementation insights, this paper offers a foundational reference to build robust, extensible, and production-ready agentic AI workflows. 

---
# Performance Comparison of Aerial RIS and STAR-RIS in 3D Wireless Environments 

**Authors**: Dongdong Yang, Bin Li, Jiguang He  

**Link**: [PDF](https://arxiv.org/pdf/2512.08755)  

**Abstract**: Reconfigurable intelligent surface (RIS) and simultaneously transmitting and reflecting RIS (STAR-RIS) have emerged as key enablers for enhancing wireless coverage and capacity in next-generation networks. When mounted on unmanned aerial vehicles (UAVs), they benefit from flexible deployment and improved line-of-sight conditions. Despite their promising potential, a comprehensive performance comparison between aerial RIS and STAR-RIS architectures has not been thoroughly investigated. This letter presents a detailed performance comparison between aerial RIS and STAR-RIS in three-dimensional wireless environments. Accurate channel models incorporating directional radiation patterns are established, and the influence of deployment altitude and orientation is thoroughly examined. To optimize the system sum-rate, we formulate joint optimization problems for both architectures and propose an efficient solution based on the weighted minimum mean square error and block coordinate descent algorithms. Simulation results reveal that STAR-RIS outperforms RIS in low-altitude scenarios due to its full-space coverage capability, whereas RIS delivers better performance near the base station at higher altitudes. The findings provide practical insights for the deployment of aerial intelligent surfaces in future 6G communication systems. 

---
# Towards Foundation Models with Native Multi-Agent Intelligence 

**Authors**: Shuyue Hu, Haoyang Yan, Yiqun Zhang, Yang Chen, Dongzhan Zhou, Lei Bai  

**Link**: [PDF](https://arxiv.org/pdf/2512.08743)  

**Abstract**: Foundation models (FMs) are increasingly assuming the role of the "brain" of AI agents. While recent efforts have begun to equip FMs with native single-agent abilities -- such as GUI interaction or integrated tool use -- we argue that the next frontier is endowing FMs with native multi-agent intelligence. We identify four core capabilities of FMs in multi-agent contexts: understanding, planning, efficient communication, and adaptation. Contrary to assumptions about the spontaneous emergence of such abilities, we provide extensive empirical evidence across 41 large language models showing that strong single-agent performance alone does not automatically yield robust multi-agent intelligence. To address this gap, we outline key research directions -- spanning dataset construction, evaluation, training paradigms, and safety considerations -- for building FMs with native multi-agent intelligence. 

---
# Deconstructing the Dual Black Box:A Plug-and-Play Cognitive Framework for Human-AI Collaborative Enhancement and Its Implications for AI Governance 

**Authors**: Yiming Lu  

**Link**: [PDF](https://arxiv.org/pdf/2512.08740)  

**Abstract**: Currently, there exists a fundamental divide between the "cognitive black box" (implicit intuition) of human experts and the "computational black box" (untrustworthy decision-making) of artificial intelligence (AI). This paper proposes a new paradigm of "human-AI collaborative cognitive enhancement," aiming to transform the dual black boxes into a composable, auditable, and extensible "functional white-box" system through structured "meta-interaction." The core breakthrough lies in the "plug-and-play cognitive framework"--a computable knowledge package that can be extracted from expert dialogues and loaded into the Recursive Adversarial Meta-Thinking Network (RAMTN). This enables expert thinking, such as medical diagnostic logic and teaching intuition, to be converted into reusable and scalable public assets, realizing a paradigm shift from "AI as a tool" to "AI as a thinking partner." This work not only provides the first engineering proof for "cognitive equity" but also opens up a new path for AI governance: constructing a verifiable and intervenable governance paradigm through "transparency of interaction protocols" rather than prying into the internal mechanisms of models. The framework is open-sourced to promote technology for good and cognitive inclusion. This paper is an independent exploratory research conducted by the author. All content presented, including the theoretical framework (RAMTN), methodology (meta-interaction), system implementation, and case validation, constitutes the author's individual research achievements. 

---
# Multi-Agent Intelligence for Multidisciplinary Decision-Making in Gastrointestinal Oncology 

**Authors**: Rongzhao Zhang, Junqiao Wang, Shuyun Yang, Mouxiao Bian, Chao Ding, Yuwei Bai, Chihao Zhang, Yuguang Shen, Lei Wang, Lei Zheng, Qiujuan Yan, Yun Zhong, Meiling Liu, Jiwei Yu, Zheng Wang, Jie Xu, Meng Luo  

**Link**: [PDF](https://arxiv.org/pdf/2512.08674)  

**Abstract**: Multimodal clinical reasoning in the field of gastrointestinal (GI) oncology necessitates the integrated interpretation of endoscopic imagery, radiological data, and biochemical markers. Despite the evident potential exhibited by Multimodal Large Language Models (MLLMs), they frequently encounter challenges such as context dilution and hallucination when confronted with intricate, heterogeneous medical histories. In order to address these limitations, a hierarchical Multi-Agent Framework is proposed, which emulates the collaborative workflow of a human Multidisciplinary Team (MDT). The system attained a composite expert evaluation score of 4.60/5.00, thereby demonstrating a substantial improvement over the monolithic baseline. It is noteworthy that the agent-based architecture yielded the most substantial enhancements in reasoning logic and medical accuracy. The findings indicate that mimetic, agent-based collaboration provides a scalable, interpretable, and clinically robust paradigm for automated decision support in oncology. 

---
# See-Control: A Multimodal Agent Framework for Smartphone Interaction with a Robotic Arm 

**Authors**: Haoyu Zhao, Weizhong Ding, Yuhao Yang, Zheng Tian, Linyi Yang, Kun Shao, Jun Wang  

**Link**: [PDF](https://arxiv.org/pdf/2512.08629)  

**Abstract**: Recent advances in Multimodal Large Language Models (MLLMs) have enabled their use as intelligent agents for smartphone operation. However, existing methods depend on the Android Debug Bridge (ADB) for data transmission and action execution, limiting their applicability to Android devices. In this work, we introduce the novel Embodied Smartphone Operation (ESO) task and present See-Control, a framework that enables smartphone operation via direct physical interaction with a low-DoF robotic arm, offering a platform-agnostic solution. See-Control comprises three key components: (1) an ESO benchmark with 155 tasks and corresponding evaluation metrics; (2) an MLLM-based embodied agent that generates robotic control commands without requiring ADB or system back-end access; and (3) a richly annotated dataset of operation episodes, offering valuable resources for future research. By bridging the gap between digital agents and the physical world, See-Control provides a concrete step toward enabling home robots to perform smartphone-dependent tasks in realistic environments. 

---
# Protein Secondary Structure Prediction Using Transformers 

**Authors**: Manzi Kevin Maxime  

**Link**: [PDF](https://arxiv.org/pdf/2512.08613)  

**Abstract**: Predicting protein secondary structures such as alpha helices, beta sheets, and coils from amino acid sequences is essential for understanding protein function. This work presents a transformer-based model that applies attention mechanisms to protein sequence data to predict structural motifs. A sliding-window data augmentation technique is used on the CB513 dataset to expand the training samples. The transformer shows strong ability to generalize across variable-length sequences while effectively capturing both local and long-range residue interactions. 

---
# CogMCTS: A Novel Cognitive-Guided Monte Carlo Tree Search Framework for Iterative Heuristic Evolution with Large Language Models 

**Authors**: Hui Wang, Yang Liu, Xiaoyu Zhang, Chaoxu Mu  

**Link**: [PDF](https://arxiv.org/pdf/2512.08609)  

**Abstract**: Automatic Heuristic Design (AHD) is an effective1 framework for solving complex optimization prob-2 lems. The development of large language mod-3 els (LLMs) enables the automated generation of4 heuristics. Existing LLM-based evolutionary meth-5 ods rely on population strategies and are prone6 to local optima. Integrating LLMs with Monte7 Carlo Tree Search (MCTS) improves the trade-off8 between exploration and exploitation, but multi-9 round cognitive integration remains limited and10 search diversity is constrained. To overcome these11 limitations, this paper proposes a novel cognitive-12 guided MCTS framework (CogMCTS). CogMCTS13 tightly integrates the cognitive guidance mecha-14 nism of LLMs with MCTS to achieve efficient au-15 tomated heuristic optimization. The framework16 employs multi-round cognitive feedback to incor-17 porate historical experience, node information, and18 negative outcomes, dynamically improving heuris-19 tic generation. Dual-track node expansion com-20 bined with elite heuristic management balances the21 exploration of diverse heuristics and the exploita-22 tion of high-quality experience. In addition, strate-23 gic mutation modifies the heuristic forms and pa-24 rameters to further enhance the diversity of the so-25 lution and the overall optimization performance.26 The experimental results indicate that CogMCTS27 outperforms existing LLM-based AHD methods in28 stability, efficiency, and solution quality. 

---
# The SMART+ Framework for AI Systems 

**Authors**: Laxmiraju Kandikatla, Branislav Radeljic  

**Link**: [PDF](https://arxiv.org/pdf/2512.08592)  

**Abstract**: Artificial Intelligence (AI) systems are now an integral part of multiple industries. In clinical research, AI supports automated adverse event detection in clinical trials, patient eligibility screening for protocol enrollment, and data quality validation. Beyond healthcare, AI is transforming finance through real-time fraud detection, automated loan risk assessment, and algorithmic decision-making. Similarly, in manufacturing, AI enables predictive maintenance to reduce equipment downtime, enhances quality control through computer-vision inspection, and optimizes production workflows using real-time operational data. While these technologies enhance operational efficiency, they introduce new challenges regarding safety, accountability, and regulatory compliance. To address these concerns, we introduce the SMART+ Framework - a structured model built on the pillars of Safety, Monitoring, Accountability, Reliability, and Transparency, and further enhanced with Privacy & Security, Data Governance, Fairness & Bias, and Guardrails. SMART+ offers a practical, comprehensive approach to evaluating and governing AI systems across industries. This framework aligns with evolving mechanisms and regulatory guidance to integrate operational safeguards, oversight procedures, and strengthened privacy and governance controls. SMART+ demonstrates risk mitigation, trust-building, and compliance readiness. By enabling responsible AI adoption and ensuring auditability, SMART+ provides a robust foundation for effective AI governance in clinical research. 

---
# Principles2Plan: LLM-Guided System for Operationalising Ethical Principles into Plans 

**Authors**: Tammy Zhong, Yang Song, Maurice Pagnucco  

**Link**: [PDF](https://arxiv.org/pdf/2512.08536)  

**Abstract**: Ethical awareness is critical for robots operating in human environments, yet existing automated planning tools provide little support. Manually specifying ethical rules is labour-intensive and highly context-specific. We present Principles2Plan, an interactive research prototype demonstrating how a human and a Large Language Model (LLM) can collaborate to produce context-sensitive ethical rules and guide automated planning. A domain expert provides the planning domain, problem details, and relevant high-level principles such as beneficence and privacy. The system generates operationalisable ethical rules consistent with these principles, which the user can review, prioritise, and supply to a planner to produce ethically-informed plans. To our knowledge, no prior system supports users in generating principle-grounded rules for classical planning contexts. Principles2Plan showcases the potential of human-LLM collaboration for making ethical automated planning more practical and feasible. 

---
# A Lightweight Transfer Learning-Based State-of-Health Monitoring with Application to Lithium-ion Batteries in Unmanned Air Vehicles 

**Authors**: Jiang Liu, Yan Qin, Wei Dai, Chau Yuen  

**Link**: [PDF](https://arxiv.org/pdf/2512.08512)  

**Abstract**: Accurate and rapid state-of-health (SOH) monitoring plays an important role in indicating energy information for lithium-ion battery-powered portable mobile devices. To confront their variable working conditions, transfer learning (TL) emerges as a promising technique for leveraging knowledge from data-rich source working conditions, significantly reducing the training data required for SOH monitoring from target working conditions. However, traditional TL-based SOH monitoring is infeasible when applied in portable mobile devices since substantial computational resources are consumed during the TL stage and unexpectedly reduce the working endurance. To address these challenges, this paper proposes a lightweight TL-based SOH monitoring approach with constructive incremental transfer learning (CITL). First, taking advantage of the unlabeled data in the target domain, a semi-supervised TL mechanism is proposed to minimize the monitoring residual in a constructive way, through iteratively adding network nodes in the CITL. Second, the cross-domain learning ability of node parameters for CITL is comprehensively guaranteed through structural risk minimization, transfer mismatching minimization, and manifold consistency maximization. Moreover, the convergence analysis of the CITL is given, theoretically guaranteeing the efficacy of TL performance and network compactness. Finally, the proposed approach is verified through extensive experiments with a realistic unmanned air vehicles (UAV) battery dataset collected from dozens of flight missions. Specifically, the CITL outperforms SS-TCA, MMD-LSTM-DA, DDAN, BO-CNN-TL, and AS$^3$LSTM, in SOH estimation by 83.73%, 61.15%, 28.24%, 87.70%, and 57.34%, respectively, as evaluated using the index root mean square error. 

---
# Autonomous Issue Resolver: Towards Zero-Touch Code Maintenance 

**Authors**: Aliaksei Kaliutau  

**Link**: [PDF](https://arxiv.org/pdf/2512.08492)  

**Abstract**: Recent advances in Large Language Models have revolutionized function-level code generation; however, repository-scale Automated Program Repair (APR) remains a significant challenge. Current approaches typically employ a control-centric paradigm, forcing agents to navigate complex directory structures and irrelevant control logic. In this paper, we propose a paradigm shift from the standard Code Property Graphs (CPGs) to the concept of Data Transformation Graph (DTG) that inverts the topology by modeling data states as nodes and functions as edges, enabling agents to trace logic defects through data lineage rather than control flow. We introduce a multi-agent framework that reconciles data integrity navigation with control flow logic. Our theoretical analysis and case studies demonstrate that this approach resolves the "Semantic Trap" inherent in standard RAG systems in modern coding agents. We provide a comprehensive implementation in the form of Autonomous Issue Resolver (AIR), a self-improvement system for zero-touch code maintenance that utilizes neuro-symbolic reasoning and uses the DTG structure for scalable logic repair. Our approach has demonstrated good results on several SWE benchmarks, reaching a resolution rate of 87.1% on SWE-Verified benchmark. Our approach directly addresses the core limitations of current AI code-assistant tools and tackles the critical need for a more robust foundation for our increasingly software-dependent world. 

---
# Using reinforcement learning to probe the role of feedback in skill acquisition 

**Authors**: Antonio Terpin, Raffaello D'Andrea  

**Link**: [PDF](https://arxiv.org/pdf/2512.08463)  

**Abstract**: Many high-performance human activities are executed with little or no external feedback: think of a figure skater landing a triple jump, a pitcher throwing a curveball for a strike, or a barista pouring latte art. To study the process of skill acquisition under fully controlled conditions, we bypass human subjects. Instead, we directly interface a generalist reinforcement learning agent with a spinning cylinder in a tabletop circulating water channel to maximize or minimize drag. This setup has several desirable properties. First, it is a physical system, with the rich interactions and complex dynamics that only the physical world has: the flow is highly chaotic and extremely difficult, if not impossible, to model or simulate accurately. Second, the objective -- drag minimization or maximization -- is easy to state and can be captured directly in the reward, yet good strategies are not obvious beforehand. Third, decades-old experimental studies provide recipes for simple, high-performance open-loop policies. Finally, the setup is inexpensive and far easier to reproduce than human studies. In our experiments we find that high-dimensional flow feedback lets the agent discover high-performance drag-control strategies with only minutes of real-world interaction. When we later replay the same action sequences without any feedback, we obtain almost identical performance. This shows that feedback, and in particular flow feedback, is not needed to execute the learned policy. Surprisingly, without flow feedback during training the agent fails to discover any well-performing policy in drag maximization, but still succeeds in drag minimization, albeit more slowly and less reliably. Our studies show that learning a high-performance skill can require richer information than executing it, and learning conditions can be kind or wicked depending solely on the goal, not on dynamics or policy complexity. 

---
# From Accuracy to Impact: The Impact-Driven AI Framework (IDAIF) for Aligning Engineering Architecture with Theory of Change 

**Authors**: Yong-Woon Kim  

**Link**: [PDF](https://arxiv.org/pdf/2512.08449)  

**Abstract**: This paper introduces the Impact-Driven AI Framework (IDAIF), a novel architectural methodology that integrates Theory of Change (ToC) principles with modern artificial intelligence system design. As AI systems increasingly influence high-stakes domains including healthcare, finance, and public policy, the alignment problem--ensuring AI behavior corresponds with human values and intentions--has become critical. Current approaches predominantly optimize technical performance metrics while neglecting the sociotechnical dimensions of AI deployment. IDAIF addresses this gap by establishing a systematic mapping between ToC's five-stage model (Inputs-Activities-Outputs-Outcomes-Impact) and corresponding AI architectural layers (Data Layer-Pipeline Layer-Inference Layer-Agentic Layer-Normative Layer). Each layer incorporates rigorous theoretical foundations: multi-objective Pareto optimization for value alignment, hierarchical multi-agent orchestration for outcome achievement, causal directed acyclic graphs (DAGs) for hallucination mitigation, and adversarial debiasing with Reinforcement Learning from Human Feedback (RLHF) for fairness assurance. We provide formal mathematical formulations for each component and introduce an Assurance Layer that manages assumption failures through guardian architectures. Three case studies demonstrate IDAIF application across healthcare, cybersecurity, and software engineering domains. This framework represents a paradigm shift from model-centric to impact-centric AI development, providing engineers with concrete architectural patterns for building ethical, trustworthy, and socially beneficial AI systems. 

---
# Prismatic World Model: Learning Compositional Dynamics for Planning in Hybrid Systems 

**Authors**: Mingwei Li, Xiaoyuan Zhang, Chengwei Yang, Zilong Zheng, Yaodong Yang  

**Link**: [PDF](https://arxiv.org/pdf/2512.08411)  

**Abstract**: Model-based planning in robotic domains is fundamentally challenged by the hybrid nature of physical dynamics, where continuous motion is punctuated by discrete events such as contacts and impacts. Conventional latent world models typically employ monolithic neural networks that enforce global continuity, inevitably over-smoothing the distinct dynamic modes (e.g., sticking vs. sliding, flight vs. stance). For a planner, this smoothing results in catastrophic compounding errors during long-horizon lookaheads, rendering the search process unreliable at physical boundaries. To address this, we introduce the Prismatic World Model (PRISM-WM), a structured architecture designed to decompose complex hybrid dynamics into composable primitives. PRISM-WM leverages a context-aware Mixture-of-Experts (MoE) framework where a gating mechanism implicitly identifies the current physical mode, and specialized experts predict the associated transition dynamics. We further introduce a latent orthogonalization objective to ensure expert diversity, effectively preventing mode collapse. By accurately modeling the sharp mode transitions in system dynamics, PRISM-WM significantly reduces rollout drift. Extensive experiments on challenging continuous control benchmarks, including high-dimensional humanoids and diverse multi-task settings, demonstrate that PRISM-WM provides a superior high-fidelity substrate for trajectory optimization algorithms (e.g., TD-MPC), proving its potential as a powerful foundational model for next-generation model-based agents. 

---
# DeepFeature: Iterative Context-aware Feature Generation for Wearable Biosignals 

**Authors**: Kaiwei Liu, Yuting He, Bufang Yang, Mu Yuan, Chun Man Victor Wong, Ho Pong Andrew Sze, Zhenyu Yan, Hongkai Chen  

**Link**: [PDF](https://arxiv.org/pdf/2512.08379)  

**Abstract**: Biosignals collected from wearable devices are widely utilized in healthcare applications. Machine learning models used in these applications often rely on features extracted from biosignals due to their effectiveness, lower data dimensionality, and wide compatibility across various model architectures. However, existing feature extraction methods often lack task-specific contextual knowledge, struggle to identify optimal feature extraction settings in high-dimensional feature space, and are prone to code generation and automation errors. In this paper, we propose DeepFeature, the first LLM-empowered, context-aware feature generation framework for wearable biosignals. DeepFeature introduces a multi-source feature generation mechanism that integrates expert knowledge with task settings. It also employs an iterative feature refinement process that uses feature assessment-based feedback for feature re-selection. Additionally, DeepFeature utilizes a robust multi-layer filtering and verification approach for robust feature-to-code translation to ensure that the extraction functions run without crashing. Experimental evaluation results show that DeepFeature achieves an average AUROC improvement of 4.21-9.67% across eight diverse tasks compared to baseline methods. It outperforms state-of-the-art approaches on five tasks while maintaining comparable performance on the remaining tasks. 

---
# Reflecting with Two Voices: A Co-Adaptive Dual-Strategy Framework for LLM-Based Agent Decision Making 

**Authors**: Wentao Zhang, Qunbo Wang, Tao Zhang, Junsheng Wu, Hongping Gan, Yang Liu, Ling Dai, Shizhuang Deng, Shuntong Sun  

**Link**: [PDF](https://arxiv.org/pdf/2512.08366)  

**Abstract**: Large language model (LLM) agents often rely on external demonstrations or retrieval-augmented planning, leading to brittleness, poor generalization, and high computational overhead. Inspired by human problem-solving, we propose DuSAR (Dual-Strategy Agent with Reflecting) - a demonstration-free framework that enables a single frozen LLM to perform co-adaptive reasoning via two complementary strategies: a high-level holistic plan and a context-grounded local policy. These strategies interact through a lightweight reflection mechanism, where the agent continuously assesses progress via a Strategy Fitness Score and dynamically revises its global plan when stuck or refines it upon meaningful advancement, mimicking human metacognitive behavior. On ALFWorld and Mind2Web, DuSAR achieves state-of-the-art performance with open-source LLMs (7B-70B), reaching 37.1% success on ALFWorld (Llama3.1-70B) - more than doubling the best prior result (13.0%) - and 4.02% on Mind2Web, also more than doubling the strongest baseline. Remarkably, it reduces per-step token consumption by 3-9X while maintaining strong performance. Ablation studies confirm the necessity of dual-strategy coordination. Moreover, optional integration of expert demonstrations further boosts results, highlighting DuSAR's flexibility and compatibility with external knowledge. 

---
# The High Cost of Incivility: Quantifying Interaction Inefficiency via Multi-Agent Monte Carlo Simulations 

**Authors**: Benedikt Mangold  

**Link**: [PDF](https://arxiv.org/pdf/2512.08345)  

**Abstract**: Workplace toxicity is widely recognized as detrimental to organizational culture, yet quantifying its direct impact on operational efficiency remains methodologically challenging due to the ethical and practical difficulties of reproducing conflict in human subjects. This study leverages Large Language Model (LLM) based Multi-Agent Systems to simulate 1-on-1 adversarial debates, creating a controlled "sociological sandbox". We employ a Monte Carlo method to simulate hundrets of discussions, measuring the convergence time (defined as the number of arguments required to reach a conclusion) between a baseline control group and treatment groups involving agents with "toxic" system prompts. Our results demonstrate a statistically significant increase of approximately 25\% in the duration of conversations involving toxic participants. We propose that this "latency of toxicity" serves as a proxy for financial damage in corporate and academic settings. Furthermore, we demonstrate that agent-based modeling provides a reproducible, ethical alternative to human-subject research for measuring the mechanics of social friction. 

---
# Enhancing Explainability of Graph Neural Networks Through Conceptual and Structural Analyses and Their Extensions 

**Authors**: Tien Cuong Bui  

**Link**: [PDF](https://arxiv.org/pdf/2512.08344)  

**Abstract**: Graph Neural Networks (GNNs) have become a powerful tool for modeling and analyzing data with graph structures. The wide adoption in numerous applications underscores the value of these models. However, the complexity of these methods often impedes understanding their decision-making processes. Current Explainable AI (XAI) methods struggle to untangle the intricate relationships and interactions within graphs. Several methods have tried to bridge this gap via a post-hoc approach or self-interpretable design. Most of them focus on graph structure analysis to determine essential patterns that correlate with prediction outcomes. While post-hoc explanation methods are adaptable, they require extra computational resources and may be less reliable due to limited access to the model's internal workings. Conversely, Interpretable models can provide immediate explanations, but their generalizability to different scenarios remains a major concern. To address these shortcomings, this thesis seeks to develop a novel XAI framework tailored for graph-based machine learning. The proposed framework aims to offer adaptable, computationally efficient explanations for GNNs, moving beyond individual feature analysis to capture how graph structure influences predictions. 

---
# Soil Compaction Parameters Prediction Based on Automated Machine Learning Approach 

**Authors**: Caner Erden, Alparslan Serhat Demir, Abdullah Hulusi Kokcam, Talas Fikret Kurnaz, Ugur Dagdeviren  

**Link**: [PDF](https://arxiv.org/pdf/2512.08343)  

**Abstract**: Soil compaction is critical in construction engineering to ensure the stability of structures like road embankments and earth dams. Traditional methods for determining optimum moisture content (OMC) and maximum dry density (MDD) involve labor-intensive laboratory experiments, and empirical regression models have limited applicability and accuracy across diverse soil types. In recent years, artificial intelligence (AI) and machine learning (ML) techniques have emerged as alternatives for predicting these compaction parameters. However, ML models often struggle with prediction accuracy and generalizability, particularly with heterogeneous datasets representing various soil types. This study proposes an automated machine learning (AutoML) approach to predict OMC and MDD. AutoML automates algorithm selection and hyperparameter optimization, potentially improving accuracy and scalability. Through extensive experimentation, the study found that the Extreme Gradient Boosting (XGBoost) algorithm provided the best performance, achieving R-squared values of 80.4% for MDD and 89.1% for OMC on a separate dataset. These results demonstrate the effectiveness of AutoML in predicting compaction parameters across different soil types. The study also highlights the importance of heterogeneous datasets in improving the generalization and performance of ML models. Ultimately, this research contributes to more efficient and reliable construction practices by enhancing the prediction of soil compaction parameters. 

---
# Predicting California Bearing Ratio with Ensemble and Neural Network Models: A Case Study from Türkiye 

**Authors**: Abdullah Hulusi Kökçam, Uğur Dağdeviren, Talas Fikret Kurnaz, Alparslan Serhat Demir, Caner Erden  

**Link**: [PDF](https://arxiv.org/pdf/2512.08340)  

**Abstract**: The California Bearing Ratio (CBR) is a key geotechnical indicator used to assess the load-bearing capacity of subgrade soils, especially in transportation infrastructure and foundation design. Traditional CBR determination relies on laboratory penetration tests. Despite their accuracy, these tests are often time-consuming, costly, and can be impractical, particularly for large-scale or diverse soil profiles. Recent progress in artificial intelligence, especially machine learning (ML), has enabled data-driven approaches for modeling complex soil behavior with greater speed and precision. This study introduces a comprehensive ML framework for CBR prediction using a dataset of 382 soil samples collected from various geoclimatic regions in Türkiye. The dataset includes physicochemical soil properties relevant to bearing capacity, allowing multidimensional feature representation in a supervised learning context. Twelve ML algorithms were tested, including decision tree, random forest, extra trees, gradient boosting, xgboost, k-nearest neighbors, support vector regression, multi-layer perceptron, adaboost, bagging, voting, and stacking regressors. Each model was trained, validated, and evaluated to assess its generalization and robustness. Among them, the random forest regressor performed the best, achieving strong R2 scores of 0.95 (training), 0.76 (validation), and 0.83 (test). These outcomes highlight the model's powerful nonlinear mapping ability, making it a promising tool for predictive geotechnical tasks. The study supports the integration of intelligent, data-centric models in geotechnical engineering, offering an effective alternative to traditional methods and promoting digital transformation in infrastructure analysis and design. 

---
# rSIM: Incentivizing Reasoning Capabilities of LLMs via Reinforced Strategy Injection 

**Authors**: Sijia Chen, Baochun Li, Di Niu  

**Link**: [PDF](https://arxiv.org/pdf/2512.08300)  

**Abstract**: Large language models (LLMs) are post-trained through reinforcement learning (RL) to evolve into Reasoning Language Models (RLMs), where the hallmark of this advanced reasoning is ``aha'' moments when they start to perform strategies, such as self-reflection and deep thinking, within chain of thoughts (CoTs). Motivated by this, this paper proposes a novel reinforced strategy injection mechanism (rSIM), that enables any LLM to become an RLM by employing a small planner to guide the LLM's CoT through the adaptive injection of reasoning strategies. To achieve this, the planner (leader agent) is jointly trained with an LLM (follower agent) using multi-agent RL (MARL), based on a leader-follower framework and straightforward rule-based rewards. Experimental results show that rSIM enables Qwen2.5-0.5B to become an RLM and significantly outperform Qwen2.5-14B. Moreover, the planner is generalizable: it only needs to be trained once and can be applied as a plug-in to substantially improve the reasoning capabilities of existing LLMs. In addition, the planner supports continual learning across various tasks, allowing its planning abilities to gradually improve and generalize to a wider range of problems. 

---
# Towards a Science of Scaling Agent Systems 

**Authors**: Yubin Kim, Ken Gu, Chanwoo Park, Chunjong Park, Samuel Schmidgall, A. Ali Heydari, Yao Yan, Zhihan Zhang, Yuchen Zhuang, Mark Malhotra, Paul Pu Liang, Hae Won Park, Yuzhe Yang, Xuhai Xu, Yilun Du, Shwetak Patel, Tim Althoff, Daniel McDuff, Xin Liu  

**Link**: [PDF](https://arxiv.org/pdf/2512.08296)  

**Abstract**: Agents, language model (LM)-based systems that are capable of reasoning, planning, and acting are becoming the dominant paradigm for real-world AI applications. Despite this widespread adoption, the principles that determine their performance remain underexplored, leaving practitioners to rely on heuristics rather than principled design choices. We address this gap by deriving quantitative scaling principles for agent systems. We evaluate this across four diverse benchmarks: Finance-Agent, BrowseComp-Plus, PlanCraft, and Workbench. Using five canonical architectures (Single, Independent, Centralized, Decentralized, Hybrid) instantiated across three LLM families, we perform a controlled evaluation spanning 180 configurations with standardized tools and token budgets. We derive a predictive model using empirical coordination metrics, including efficiency, overhead, error amplification, and redundancy, that achieves cross-validated R^2=0.513. We identify three dominant effects: (1) a tool-coordination trade-off: under fixed computational budgets, tool-heavy tasks suffer disproportionately from multi-agent overhead. (2) a capability saturation: coordination yields diminishing or negative returns (beta=-0.408, p<0.001) once single-agent baselines exceed ~45%. (3) topology-dependent error amplification: independent agents amplify errors 17.2x through unchecked propagation, while centralized coordination contains this to 4.4x. Centralized coordination improves performance by 80.9% on parallelizable tasks like financial reasoning, while decentralized coordination excels on dynamic web navigation (+9.2% vs. +0.2%). Yet for sequential reasoning tasks, all multi-agent variants degraded performance by 39-70%. The framework predicts the optimal coordination strategy for 87% of held-out configurations, providing a predictive principle of agentic scaling based on measurable task properties. 

---
# AgentEval: Generative Agents as Reliable Proxies for Human Evaluation of AI-Generated Content 

**Authors**: Thanh Vu, Richi Nayak, Thiru Balasubramaniam  

**Link**: [PDF](https://arxiv.org/pdf/2512.08273)  

**Abstract**: Modern businesses are increasingly challenged by the time and expense required to generate and assess high-quality content. Human writers face time constraints, and extrinsic evaluations can be costly. While Large Language Models (LLMs) offer potential in content creation, concerns about the quality of AI-generated content persist. Traditional evaluation methods, like human surveys, further add operational costs, highlighting the need for efficient, automated solutions. This research introduces Generative Agents as a means to tackle these challenges. These agents can rapidly and cost-effectively evaluate AI-generated content, simulating human judgment by rating aspects such as coherence, interestingness, clarity, fairness, and relevance. By incorporating these agents, businesses can streamline content generation and ensure consistent, high-quality output while minimizing reliance on costly human evaluations. The study provides critical insights into enhancing LLMs for producing business-aligned, high-quality content, offering significant advancements in automated content generation and evaluation. 

---
# Reasoning Models Ace the CFA Exams 

**Authors**: Jaisal Patel, Yunzhe Chen, Kaiwen He, Keyi Wang, David Li, Kairong Xiao, Xiao-Yang Liu  

**Link**: [PDF](https://arxiv.org/pdf/2512.08270)  

**Abstract**: Previous research has reported that large language models (LLMs) demonstrate poor performance on the Chartered Financial Analyst (CFA) exams. However, recent reasoning models have achieved strong results on graduate-level academic and professional examinations across various disciplines. In this paper, we evaluate state-of-the-art reasoning models on a set of mock CFA exams consisting of 980 questions across three Level I exams, two Level II exams, and three Level III exams. Using the same pass/fail criteria from prior studies, we find that most models clear all three levels. The models that pass, ordered by overall performance, are Gemini 3.0 Pro, Gemini 2.5 Pro, GPT-5, Grok 4, Claude Opus 4.1, and DeepSeek-V3.1. Specifically, Gemini 3.0 Pro achieves a record score of 97.6% on Level I. Performance is also strong on Level II, led by GPT-5 at 94.3%. On Level III, Gemini 2.5 Pro attains the highest score with 86.4% on multiple-choice questions while Gemini 3.0 Pro achieves 92.0% on constructed-response questions. 

---
# Beyond Traditional Diagnostics: Transforming Patient-Side Information into Predictive Insights with Knowledge Graphs and Prototypes 

**Authors**: Yibowen Zhao, Yinan Zhang, Zhixiang Su, Lizhen Cui, Chunyan Miao  

**Link**: [PDF](https://arxiv.org/pdf/2512.08261)  

**Abstract**: Predicting diseases solely from patient-side information, such as demographics and self-reported symptoms, has attracted significant research attention due to its potential to enhance patient awareness, facilitate early healthcare engagement, and improve healthcare system efficiency. However, existing approaches encounter critical challenges, including imbalanced disease distributions and a lack of interpretability, resulting in biased or unreliable predictions. To address these issues, we propose the Knowledge graph-enhanced, Prototype-aware, and Interpretable (KPI) framework. KPI systematically integrates structured and trusted medical knowledge into a unified disease knowledge graph, constructs clinically meaningful disease prototypes, and employs contrastive learning to enhance predictive accuracy, which is particularly important for long-tailed diseases. Additionally, KPI utilizes large language models (LLMs) to generate patient-specific, medically relevant explanations, thereby improving interpretability and reliability. Extensive experiments on real-world datasets demonstrate that KPI outperforms state-of-the-art methods in predictive accuracy and provides clinically valid explanations that closely align with patient narratives, highlighting its practical value for patient-centered healthcare delivery. 

---
# Empowerment Gain and Causal Model Construction: Children and adults are sensitive to controllability and variability in their causal interventions 

**Authors**: Eunice Yiu, Kelsey Allen, Shiry Ginosar, Alison Gopnik  

**Link**: [PDF](https://arxiv.org/pdf/2512.08230)  

**Abstract**: Learning about the causal structure of the world is a fundamental problem for human cognition. Causal models and especially causal learning have proved to be difficult for large pretrained models using standard techniques of deep learning. In contrast, cognitive scientists have applied advances in our formal understanding of causation in computer science, particularly within the Causal Bayes Net formalism, to understand human causal learning. In the very different tradition of reinforcement learning, researchers have described an intrinsic reward signal called "empowerment" which maximizes mutual information between actions and their outcomes. "Empowerment" may be an important bridge between classical Bayesian causal learning and reinforcement learning and may help to characterize causal learning in humans and enable it in machines. If an agent learns an accurate causal world model, they will necessarily increase their empowerment, and increasing empowerment will lead to a more accurate causal world model. Empowerment may also explain distinctive features of childrens causal learning, as well as providing a more tractable computational account of how that learning is possible. In an empirical study, we systematically test how children and adults use cues to empowerment to infer causal relations, and design effective causal interventions. 

---
# Scalable Back-End for an AI-Based Diabetes Prediction Application 

**Authors**: Henry Anand Septian Radityo, Bernardus Willson, Reynard Tanadi, Latifa Dwiyanti, Saiful Akbar  

**Link**: [PDF](https://arxiv.org/pdf/2512.08147)  

**Abstract**: The rising global prevalence of diabetes necessitates early detection to prevent severe complications. While AI-powered prediction applications offer a promising solution, they require a responsive and scalable back-end architecture to serve a large user base effectively. This paper details the development and evaluation of a scalable back-end system designed for a mobile diabetes prediction application. The primary objective was to maintain a failure rate below 5% and an average latency of under 1000 ms. The architecture leverages horizontal scaling, database sharding, and asynchronous communication via a message queue. Performance evaluation showed that 83% of the system's features (20 out of 24) met the specified performance targets. Key functionalities such as user profile management, activity tracking, and read-intensive prediction operations successfully achieved the desired performance. The system demonstrated the ability to handle up to 10,000 concurrent users without issues, validating its scalability. The implementation of asynchronous communication using RabbitMQ proved crucial in minimizing the error rate for computationally intensive prediction requests, ensuring system reliability by queuing requests and preventing data loss under heavy load. 

---
# Large Language Models for Education and Research: An Empirical and User Survey-based Analysis 

**Authors**: Md Mostafizer Rahman, Ariful Islam Shiplu, Md Faizul Ibne Amin, Yutaka Watanobe, Lu Peng  

**Link**: [PDF](https://arxiv.org/pdf/2512.08057)  

**Abstract**: Pretrained Large Language Models (LLMs) have achieved remarkable success across diverse domains, with education and research emerging as particularly impactful areas. Among current state-of-the-art LLMs, ChatGPT and DeepSeek exhibit strong capabilities in mathematics, science, medicine, literature, and programming. In this study, we present a comprehensive evaluation of these two LLMs through background technology analysis, empirical experiments, and a real-world user survey. The evaluation explores trade-offs among model accuracy, computational efficiency, and user experience in educational and research affairs. We benchmarked these LLMs performance in text generation, programming, and specialized problem-solving. Experimental results show that ChatGPT excels in general language understanding and text generation, while DeepSeek demonstrates superior performance in programming tasks due to its efficiency- focused design. Moreover, both models deliver medically accurate diagnostic outputs and effectively solve complex mathematical problems. Complementing these quantitative findings, a survey of students, educators, and researchers highlights the practical benefits and limitations of these models, offering deeper insights into their role in advancing education and research. 

---
# Toward an AI Reasoning-Enabled System for Patient-Clinical Trial Matching 

**Authors**: Caroline N. Leach, Mitchell A. Klusty, Samuel E. Armstrong, Justine C. Pickarski, Kristen L. Hankins, Emily B. Collier, Maya Shah, Aaron D. Mullen, V. K. Cody Bumgardner  

**Link**: [PDF](https://arxiv.org/pdf/2512.08026)  

**Abstract**: Screening patients for clinical trial eligibility remains a manual, time-consuming, and resource-intensive process. We present a secure, scalable proof-of-concept system for Artificial Intelligence (AI)-augmented patient-trial matching that addresses key implementation challenges: integrating heterogeneous electronic health record (EHR) data, facilitating expert review, and maintaining rigorous security standards. Leveraging open-source, reasoning-enabled large language models (LLMs), the system moves beyond binary classification to generate structured eligibility assessments with interpretable reasoning chains that support human-in-the-loop review. This decision support tool represents eligibility as a dynamic state rather than a fixed determination, identifying matches when available and offering actionable recommendations that could render a patient eligible in the future. The system aims to reduce coordinator burden, intelligently broaden the set of trials considered for each patient and guarantee comprehensive auditability of all AI-generated outputs. 

---
# SkipKV: Selective Skipping of KV Generation and Storage for Efficient Inference with Large Reasoning Models 

**Authors**: Jiayi Tian, Seyedarmin Azizi, Yequan Zhao, Erfan Baghaei Potraghloo, Sean McPherson, Sharath Nittur Sridhar, Zhengyang Wang, Zheng Zhang, Massoud Pedram, Souvik Kundu  

**Link**: [PDF](https://arxiv.org/pdf/2512.07993)  

**Abstract**: Large reasoning models (LRMs) often cost significant key-value (KV) cache overhead, due to their linear growth with the verbose chain-of-thought (CoT) reasoning process. This costs both memory and throughput bottleneck limiting their efficient deployment. Towards reducing KV cache size during inference, we first investigate the effectiveness of existing KV cache eviction methods for CoT reasoning. Interestingly, we find that due to unstable token-wise scoring and the reduced effective KV budget caused by padding tokens, state-of-the-art (SoTA) eviction methods fail to maintain accuracy in the multi-batch setting. Additionally, these methods often generate longer sequences than the original model, as semantic-unaware token-wise eviction leads to repeated revalidation during reasoning. To address these issues, we present \textbf{SkipKV}, a \textbf{\textit{training-free}} KV compression method for selective \textit{eviction} and \textit{generation} operating at a coarse-grained sentence-level sequence removal for efficient CoT reasoning. In specific, it introduces a \textit{sentence-scoring metric} to identify and remove highly similar sentences while maintaining semantic coherence. To suppress redundant generation, SkipKV dynamically adjusts a steering vector to update the hidden activation states during inference enforcing the LRM to generate concise response. Extensive evaluations on multiple reasoning benchmarks demonstrate the effectiveness of SkipKV in maintaining up to $\mathbf{26.7}\%$ improved accuracy compared to the alternatives, at a similar compression budget. Additionally, compared to SoTA, SkipKV yields up to $\mathbf{1.6}\times$ fewer generation length while improving throughput up to $\mathbf{1.7}\times$. 

---
# Can AI autonomously build, operate, and use the entire data stack? 

**Authors**: Arvind Agarwal, Lisa Amini, Sameep Mehta, Horst Samulowitz, Kavitha Srinivas  

**Link**: [PDF](https://arxiv.org/pdf/2512.07926)  

**Abstract**: Enterprise data management is a monumental task. It spans data architecture and systems, integration, quality, governance, and continuous improvement. While AI assistants can help specific persona, such as data engineers and stewards, to navigate and configure the data stack, they fall far short of full automation. However, as AI becomes increasingly capable of tackling tasks that have previously resisted automation due to inherent complexities, we believe there is an imminent opportunity to target fully autonomous data estates. Currently, AI is used in different parts of the data stack, but in this paper, we argue for a paradigm shift from the use of AI in independent data component operations towards a more holistic and autonomous handling of the entire data lifecycle. Towards that end, we explore how each stage of the modern data stack can be autonomously managed by intelligent agents to build self-sufficient systems that can be used not only by human end-users, but also by AI itself. We begin by describing the mounting forces and opportunities that demand this paradigm shift, examine how agents can streamline the data lifecycle, and highlight open questions and areas where additional research is needed. We hope this work will inspire lively debate, stimulate further research, motivate collaborative approaches, and facilitate a more autonomous future for data systems. 

---
# Impact of Data-Oriented and Object-Oriented Design on Performance and Cache Utilization with Artificial Intelligence Algorithms in Multi-Threaded CPUs 

**Authors**: Gabriel M. Arantes, Richard F. Pinto, Bruno L. Dalmazo, Eduardo N. Borges, Giancarlo Lucca, Viviane L. D. de Mattos, Fabian C. Cardoso, Rafael A. Berri  

**Link**: [PDF](https://arxiv.org/pdf/2512.07841)  

**Abstract**: The growing performance gap between multi-core CPUs and main memory necessitates hardware-aware software design paradigms. This study provides a comprehensive performance analysis of Data Oriented Design (DOD) versus the traditional Object-Oriented Design (OOD), focusing on cache utilization and efficiency in multi-threaded environments. We developed and compared four distinct versions of the A* search algorithm: single-threaded OOD (ST-OOD), single-threaded DOD (ST-DOD), multi-threaded OOD (MT-OOD), and multi-threaded DOD (MT-DOD). The evaluation was based on metrics including execution time, memory usage, and CPU cache misses. In multi-threaded tests, the DOD implementation demonstrated considerable performance gains, with faster execution times and a lower number of raw system calls and cache misses. While OOD occasionally showed marginal advantages in memory usage or percentage-based cache miss rates, DOD's efficiency in data-intensive operations was more evident. Furthermore, our findings reveal that for a fine-grained task like the A* algorithm, the overhead associated with thread management led to single-threaded versions significantly outperforming their multi-threaded counterparts in both paradigms. We conclude that even when performance differences appear subtle in simple algorithms, the consistent advantages of DOD in critical metrics highlight its foundational architectural superiority, suggesting it is a more effective approach for maximizing hardware efficiency in complex, large-scale AI and parallel computing tasks. 

---
# Astra: General Interactive World Model with Autoregressive Denoising 

**Authors**: Yixuan Zhu, Jiaqi Feng, Wenzhao Zheng, Yuan Gao, Xin Tao, Pengfei Wan, Jie Zhou, Jiwen Lu  

**Link**: [PDF](https://arxiv.org/pdf/2512.08931)  

**Abstract**: Recent advances in diffusion transformers have empowered video generation models to generate high-quality video clips from texts or images. However, world models with the ability to predict long-horizon futures from past observations and actions remain underexplored, especially for general-purpose scenarios and various forms of actions. To bridge this gap, we introduce Astra, an interactive general world model that generates real-world futures for diverse scenarios (e.g., autonomous driving, robot grasping) with precise action interactions (e.g., camera motion, robot action). We propose an autoregressive denoising architecture and use temporal causal attention to aggregate past observations and support streaming outputs. We use a noise-augmented history memory to avoid over-reliance on past frames to balance responsiveness with temporal coherence. For precise action control, we introduce an action-aware adapter that directly injects action signals into the denoising process. We further develop a mixture of action experts that dynamically route heterogeneous action modalities, enhancing versatility across diverse real-world tasks such as exploration, manipulation, and camera control. Astra achieves interactive, consistent, and general long-term video prediction and supports various forms of interactions. Experiments across multiple datasets demonstrate the improvements of Astra in fidelity, long-range prediction, and action alignment over existing state-of-the-art world models. 

---
# SAQ: Stabilizer-Aware Quantum Error Correction Decoder 

**Authors**: David Zenati, Eliya Nachmani  

**Link**: [PDF](https://arxiv.org/pdf/2512.08914)  

**Abstract**: Quantum Error Correction (QEC) decoding faces a fundamental accuracy-efficiency tradeoff. Classical methods like Minimum Weight Perfect Matching (MWPM) exhibit variable performance across noise models and suffer from polynomial complexity, while tensor network decoders achieve high accuracy but at prohibitively high computational cost. Recent neural decoders reduce complexity but lack the accuracy needed to compete with computationally expensive classical methods. We introduce SAQ-Decoder, a unified framework combining transformer-based learning with constraint aware post-processing that achieves both near Maximum Likelihood (ML) accuracy and linear computational scalability with respect to the syndrome size. Our approach combines a dual-stream transformer architecture that processes syndromes and logical information with asymmetric attention patterns, and a novel differentiable logical loss that directly optimizes Logical Error Rates (LER) through smooth approximations over finite fields. SAQ-Decoder achieves near-optimal performance, with error thresholds of 10.99% (independent noise) and 18.6% (depolarizing noise) on toric codes that approach the ML bounds of 11.0% and 18.9% while outperforming existing neural and classical baselines in accuracy, complexity, and parameter efficiency. Our findings establish that learned decoders can simultaneously achieve competitive decoding accuracy and computational efficiency, addressing key requirements for practical fault-tolerant quantum computing systems. 

---
# Revisiting the Scaling Properties of Downstream Metrics in Large Language Model Training 

**Authors**: Jakub Krajewski, Amitis Shidani, Dan Busbridge, Sam Wiseman, Jason Ramapuram  

**Link**: [PDF](https://arxiv.org/pdf/2512.08894)  

**Abstract**: While scaling laws for Large Language Models (LLMs) traditionally focus on proxy metrics like pretraining loss, predicting downstream task performance has been considered unreliable. This paper challenges that view by proposing a direct framework to model the scaling of benchmark performance from the training budget. We find that for a fixed token-to-parameter ratio, a simple power law can accurately describe the scaling behavior of log accuracy on multiple popular downstream tasks. Our results show that the direct approach extrapolates better than the previously proposed two-stage procedure, which is prone to compounding errors. Furthermore, we introduce functional forms that predict accuracy across token-to-parameter ratios and account for inference compute under repeated sampling. We validate our findings on models with up to 17B parameters trained on up to 350B tokens across two dataset mixtures. To support reproducibility and encourage future research, we release the complete set of pretraining losses and downstream evaluation results. 

---
# Toward Faithful Retrieval-Augmented Generation with Sparse Autoencoders 

**Authors**: Guangzhi Xiong, Zhenghao He, Bohan Liu, Sanchit Sinha, Aidong Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2512.08892)  

**Abstract**: Retrieval-Augmented Generation (RAG) improves the factuality of large language models (LLMs) by grounding outputs in retrieved evidence, but faithfulness failures, where generations contradict or extend beyond the provided sources, remain a critical challenge. Existing hallucination detection methods for RAG often rely either on large-scale detector training, which requires substantial annotated data, or on querying external LLM judges, which leads to high inference costs. Although some approaches attempt to leverage internal representations of LLMs for hallucination detection, their accuracy remains limited. Motivated by recent advances in mechanistic interpretability, we employ sparse autoencoders (SAEs) to disentangle internal activations, successfully identifying features that are specifically triggered during RAG hallucinations. Building on a systematic pipeline of information-based feature selection and additive feature modeling, we introduce RAGLens, a lightweight hallucination detector that accurately flags unfaithful RAG outputs using LLM internal representations. RAGLens not only achieves superior detection performance compared to existing methods, but also provides interpretable rationales for its decisions, enabling effective post-hoc mitigation of unfaithful RAG. Finally, we justify our design choices and reveal new insights into the distribution of hallucination-related signals within LLMs. The code is available at this https URL. 

---
# No Labels, No Problem: Training Visual Reasoners with Multimodal Verifiers 

**Authors**: Damiano Marsili, Georgia Gkioxari  

**Link**: [PDF](https://arxiv.org/pdf/2512.08889)  

**Abstract**: Visual reasoning is challenging, requiring both precise object grounding and understanding complex spatial relationships. Existing methods fall into two camps: language-only chain-of-thought approaches, which demand large-scale (image, query, answer) supervision, and program-synthesis approaches which use pre-trained models and avoid training, but suffer from flawed logic and erroneous grounding. We propose an annotation-free training framework that improves both reasoning and grounding. Our framework uses AI-powered verifiers: an LLM verifier refines LLM reasoning via reinforcement learning, while a VLM verifier strengthens visual grounding through automated hard-negative mining, eliminating the need for ground truth labels. This design combines the strengths of modern AI systems: advanced language-only reasoning models for decomposing spatial queries into simpler subtasks, and strong vision specialist models improved via performant VLM critics. We evaluate our approach across diverse spatial reasoning tasks, and show that our method improves visual reasoning and surpasses open-source and proprietary models, while with our improved visual grounding model we further outperform recent text-only visual reasoning methods. Project webpage: this https URL 

---
# DAO-GP Drift Aware Online Non-Linear Regression Gaussian-Process 

**Authors**: Mohammad Abu-Shaira, Ajita Rattani, Weishi Shi  

**Link**: [PDF](https://arxiv.org/pdf/2512.08879)  

**Abstract**: Real-world datasets often exhibit temporal dynamics characterized by evolving data distributions. Disregarding this phenomenon, commonly referred to as concept drift, can significantly diminish a model's predictive accuracy. Furthermore, the presence of hyperparameters in online models exacerbates this issue. These parameters are typically fixed and cannot be dynamically adjusted by the user in response to the evolving data distribution. Gaussian Process (GP) models offer powerful non-parametric regression capabilities with uncertainty quantification, making them ideal for modeling complex data relationships in an online setting. However, conventional online GP methods face several critical limitations, including a lack of drift-awareness, reliance on fixed hyperparameters, vulnerability to data snooping, absence of a principled decay mechanism, and memory inefficiencies. In response, we propose DAO-GP (Drift-Aware Online Gaussian Process), a novel, fully adaptive, hyperparameter-free, decayed, and sparse non-linear regression model. DAO-GP features a built-in drift detection and adaptation mechanism that dynamically adjusts model behavior based on the severity of drift. Extensive empirical evaluations confirm DAO-GP's robustness across stationary conditions, diverse drift types (abrupt, incremental, gradual), and varied data characteristics. Analyses demonstrate its dynamic adaptation, efficient in-memory and decay-based management, and evolving inducing points. Compared with state-of-the-art parametric and non-parametric models, DAO-GP consistently achieves superior or competitive performance, establishing it as a drift-resilient solution for online non-linear regression. 

---
# When Tables Leak: Attacking String Memorization in LLM-Based Tabular Data Generation 

**Authors**: Joshua Ward, Bochao Gu, Chi-Hua Wang, Guang Cheng  

**Link**: [PDF](https://arxiv.org/pdf/2512.08875)  

**Abstract**: Large Language Models (LLMs) have recently demonstrated remarkable performance in generating high-quality tabular synthetic data. In practice, two primary approaches have emerged for adapting LLMs to tabular data generation: (i) fine-tuning smaller models directly on tabular datasets, and (ii) prompting larger models with examples provided in context. In this work, we show that popular implementations from both regimes exhibit a tendency to compromise privacy by reproducing memorized patterns of numeric digits from their training data. To systematically analyze this risk, we introduce a simple No-box Membership Inference Attack (MIA) called LevAtt that assumes adversarial access to only the generated synthetic data and targets the string sequences of numeric digits in synthetic observations. Using this approach, our attack exposes substantial privacy leakage across a wide range of models and datasets, and in some cases, is even a perfect membership classifier on state-of-the-art models. Our findings highlight a unique privacy vulnerability of LLM-based synthetic data generation and the need for effective defenses. To this end, we propose two methods, including a novel sampling strategy that strategically perturbs digits during generation. Our evaluation demonstrates that this approach can defeat these attacks with minimal loss of fidelity and utility of the synthetic data. 

---
# Siamese-Driven Optimization for Low-Resolution Image Latent Embedding in Image Captioning 

**Authors**: Jing Jie Tan, Anissa Mokraoui, Ban-Hoe Kwan, Danny Wee-Kiat Ng, Yan-Chai Hum  

**Link**: [PDF](https://arxiv.org/pdf/2512.08873)  

**Abstract**: Image captioning is essential in many fields including assisting visually impaired individuals, improving content management systems, and enhancing human-computer interaction. However, a recent challenge in this domain is dealing with low-resolution image (LRI). While performance can be improved by using larger models like transformers for encoding, these models are typically heavyweight, demanding significant computational resources and memory, leading to challenges in retraining. To address this, the proposed SOLI (Siamese-Driven Optimization for Low-Resolution Image Latent Embedding in Image Captioning) approach presents a solution specifically designed for lightweight, low-resolution images captioning. It employs a Siamese network architecture to optimize latent embeddings, enhancing the efficiency and accuracy of the image-to-text translation process. By focusing on a dual-pathway neural network structure, SOLI minimizes computational overhead without sacrificing performance, making it an ideal choice for training on resource-constrained scenarios. 

---
# Fed-SE: Federated Self-Evolution for Privacy-Constrained Multi-Environment LLM Agents 

**Authors**: Xiang Chen, Yuling Shi, Qizhen Lan, Yuchao Qiu, Xiaodong Gu  

**Link**: [PDF](https://arxiv.org/pdf/2512.08870)  

**Abstract**: LLM agents are widely deployed in complex interactive tasks, yet privacy constraints often preclude centralized optimization and co-evolution across dynamic environments. While Federated Learning (FL) has proven effective on static datasets, its extension to the open-ended self-evolution of agents remains underexplored. Directly applying standard FL is challenging: heterogeneous tasks and sparse, trajectory-level rewards introduce severe gradient conflicts, destabilizing the global optimization process. To bridge this gap, we propose Fed-SE, a Federated Self-Evolution framework for LLM agents. Fed-SE establishes a local evolution-global aggregation paradigm. Locally, agents employ parameter-efficient fine-tuning on filtered, high-return trajectories to achieve stable gradient updates. Globally, Fed-SE aggregates updates within a low-rank subspace that disentangles environment-specific dynamics, effectively reducing negative transfer across clients. Experiments across five heterogeneous environments demonstrate that Fed-SE improves average task success rates by approximately 18% over federated baselines, validating its effectiveness in robust cross-environment knowledge transfer in privacy-constrained deployments. 

---
# Differentially Private Synthetic Data Generation Using Context-Aware GANs 

**Authors**: Anantaa Kotal, Anupam Joshi  

**Link**: [PDF](https://arxiv.org/pdf/2512.08869)  

**Abstract**: The widespread use of big data across sectors has raised major privacy concerns, especially when sensitive information is shared or analyzed. Regulations such as GDPR and HIPAA impose strict controls on data handling, making it difficult to balance the need for insights with privacy requirements. Synthetic data offers a promising solution by creating artificial datasets that reflect real patterns without exposing sensitive information. However, traditional synthetic data methods often fail to capture complex, implicit rules that link different elements of the data and are essential in domains like healthcare. They may reproduce explicit patterns but overlook domain-specific constraints that are not directly stated yet crucial for realism and utility. For example, prescription guidelines that restrict certain medications for specific conditions or prevent harmful drug interactions may not appear explicitly in the original data. Synthetic data generated without these implicit rules can lead to medically inappropriate or unrealistic profiles. To address this gap, we propose ContextGAN, a Context-Aware Differentially Private Generative Adversarial Network that integrates domain-specific rules through a constraint matrix encoding both explicit and implicit knowledge. The constraint-aware discriminator evaluates synthetic data against these rules to ensure adherence to domain constraints, while differential privacy protects sensitive details from the original data. We validate ContextGAN across healthcare, security, and finance, showing that it produces high-quality synthetic data that respects domain rules and preserves privacy. Our results demonstrate that ContextGAN improves realism and utility by enforcing domain constraints, making it suitable for applications that require compliance with both explicit patterns and implicit rules under strict privacy guarantees. 

---
# InfiniteVL: Synergizing Linear and Sparse Attention for Highly-Efficient, Unlimited-Input Vision-Language Models 

**Authors**: Hongyuan Tao, Bencheng Liao, Shaoyu Chen, Haoran Yin, Qian Zhang, Wenyu Liu, Xinggang Wang  

**Link**: [PDF](https://arxiv.org/pdf/2512.08829)  

**Abstract**: Window attention and linear attention represent two principal strategies for mitigating the quadratic complexity and ever-growing KV cache in Vision-Language Models (VLMs). However, we observe that window-based VLMs suffer performance degradation when sequence length exceeds the window size, while linear attention underperforms on information-intensive tasks such as OCR and document understanding. To overcome these limitations, we propose InfiniteVL, a linear-complexity VLM architecture that synergizes sliding window attention (SWA) with Gated DeltaNet. For achieving competitive multimodal performance under constrained resources, we design a three-stage training strategy comprising distillation pretraining, instruction tuning, and long-sequence SFT. Remarkably, using less than 2\% of the training data required by leading VLMs, InfiniteVL not only substantially outperforms previous linear-complexity VLMs but also matches the performance of leading Transformer-based VLMs, while demonstrating effective long-term memory retention. Compared to similar-sized Transformer-based VLMs accelerated by FlashAttention-2, InfiniteVL achieves over 3.6\times inference speedup while maintaining constant latency and memory footprint. In streaming video understanding scenarios, it sustains a stable 24 FPS real-time prefill speed while preserving long-term memory cache. Code and models are available at this https URL. 

---
# Training-Free Dual Hyperbolic Adapters for Better Cross-Modal Reasoning 

**Authors**: Yi Zhang, Chun-Wun Cheng, Junyi He, Ke Yu, Yushun Tang, Carola-Bibiane Schönlieb, Zhihai He, Angelica I. Aviles-Rivero  

**Link**: [PDF](https://arxiv.org/pdf/2512.08820)  

**Abstract**: Recent research in Vision-Language Models (VLMs) has significantly advanced our capabilities in cross-modal reasoning. However, existing methods suffer from performance degradation with domain changes or require substantial computational resources for fine-tuning in new domains. To address this issue, we develop a new adaptation method for large vision-language models, called \textit{Training-free Dual Hyperbolic Adapters} (T-DHA). We characterize the vision-language relationship between semantic concepts, which typically has a hierarchical tree structure, in the hyperbolic space instead of the traditional Euclidean space. Hyperbolic spaces exhibit exponential volume growth with radius, unlike the polynomial growth in Euclidean space. We find that this unique property is particularly effective for embedding hierarchical data structures using the Poincaré ball model, achieving significantly improved representation and discrimination power. Coupled with negative learning, it provides more accurate and robust classifications with fewer feature dimensions. Our extensive experimental results on various datasets demonstrate that the T-DHA method significantly outperforms existing state-of-the-art methods in few-shot image recognition and domain generalization tasks. 

---
# Do Depth-Grown Models Overcome the Curse of Depth? An In-Depth Analysis 

**Authors**: Ferdinand Kapl, Emmanouil Angelis, Tobias Höppe, Kaitlin Maile, Johannes von Oswald, Nino Scherrer, Stefan Bauer  

**Link**: [PDF](https://arxiv.org/pdf/2512.08819)  

**Abstract**: Gradually growing the depth of Transformers during training can not only reduce training cost but also lead to improved reasoning performance, as shown by MIDAS (Saunshi et al., 2024). Thus far, however, a mechanistic understanding of these gains has been missing. In this work, we establish a connection to recent work showing that layers in the second half of non-grown, pre-layernorm Transformers contribute much less to the final output distribution than those in the first half - also known as the Curse of Depth (Sun et al., 2025, Csordás et al., 2025). Using depth-wise analyses, we demonstrate that growth via gradual middle stacking yields more effective utilization of model depth, alters the residual stream structure, and facilitates the formation of permutable computational blocks. In addition, we propose a lightweight modification of MIDAS that yields further improvements in downstream reasoning benchmarks. Overall, this work highlights how the gradual growth of model depth can lead to the formation of distinct computational circuits and overcome the limited depth utilization seen in standard non-grown models. 

---
# Emovectors: assessing emotional content in jazz improvisations for creativity evaluation 

**Authors**: Anna Jordanous  

**Link**: [PDF](https://arxiv.org/pdf/2512.08812)  

**Abstract**: Music improvisation is fascinating to study, being essentially a live demonstration of a creative process. In jazz, musicians often improvise across predefined chord progressions (leadsheets). How do we assess the creativity of jazz improvisations? And can we capture this in automated metrics for creativity for current LLM-based generative systems? Demonstration of emotional involvement is closely linked with creativity in improvisation. Analysing musical audio, can we detect emotional involvement? This study hypothesises that if an improvisation contains more evidence of emotion-laden content, it is more likely to be recognised as creative. An embeddings-based method is proposed for capturing the emotional content in musical improvisations, using a psychologically-grounded classification of musical characteristics associated with emotions. Resulting 'emovectors' are analysed to test the above hypothesis, comparing across multiple improvisations. Capturing emotional content in this quantifiable way can contribute towards new metrics for creativity evaluation that can be applied at scale. 

---
# Multicalibration for LLM-based Code Generation 

**Authors**: Viola Campos, Robin Kuschnereit, Adrian Ulges  

**Link**: [PDF](https://arxiv.org/pdf/2512.08810)  

**Abstract**: As AI-based code generation becomes widespread, researchers are investigating the calibration of code LLMs - ensuring their confidence scores faithfully represent the true likelihood of code correctness. To do so, we investigate multicalibration, which can capture additional factors about a coding problem, such as complexity, code length, or programming language used. We study four multicalibration approaches on three function synthesis benchmarks, using latest-generation code LLMs (Qwen3 Coder, GPT-OSS, DeepSeek-R1-Distill). Our results demonstrate that multicalibration can yield distinct improvements over both uncalibrated token likelihoods (+1.03 in skill score) and baseline calibrations (+0.37 in skill score). We study the influence of the aforementioned factors in ablations, and make our dataset (consisting of code generations, likelihoods, and correctness labels) available for future research on code LLM calibration. 

---
# PrivTune: Efficient and Privacy-Preserving Fine-Tuning of Large Language Models via Device-Cloud Collaboration 

**Authors**: Yi Liu, Weixiang Han, Chengjun Cai, Xingliang Yuan, Cong Wang  

**Link**: [PDF](https://arxiv.org/pdf/2512.08809)  

**Abstract**: With the rise of large language models, service providers offer language models as a service, enabling users to fine-tune customized models via uploaded private datasets. However, this raises concerns about sensitive data leakage. Prior methods, relying on differential privacy within device-cloud collaboration frameworks, struggle to balance privacy and utility, exposing users to inference attacks or degrading fine-tuning performance. To address this, we propose PrivTune, an efficient and privacy-preserving fine-tuning framework via Split Learning (SL). The key idea of PrivTune is to inject crafted noise into token representations from the SL bottom model, making each token resemble the $n$-hop indirect neighbors. PrivTune formulates this as an optimization problem to compute the optimal noise vector, aligning with defense-utility goals. On this basis, it then adjusts the parameters (i.e., mean) of the $d_\chi$-Privacy noise distribution to align with the optimization direction and scales the noise according to token importance to minimize distortion. Experiments on five datasets (covering both classification and generation tasks) against three embedding inversion and three attribute inference attacks show that, using RoBERTa on the Stanford Sentiment Treebank dataset, PrivTune reduces the attack success rate to 10% with only a 3.33% drop in utility performance, outperforming state-of-the-art baselines. 

---
# Democratizing ML for Enterprise Security: A Self-Sustained Attack Detection Framework 

**Authors**: Sadegh Momeni, Ge Zhang, Birkett Huber, Hamza Harkous, Sam Lipton, Benoit Seguin, Yanis Pavlidis  

**Link**: [PDF](https://arxiv.org/pdf/2512.08802)  

**Abstract**: Despite advancements in machine learning for security, rule-based detection remains prevalent in Security Operations Centers due to the resource intensiveness and skill gap associated with ML solutions. While traditional rule-based methods offer efficiency, their rigidity leads to high false positives or negatives and requires continuous manual maintenance. This paper proposes a novel, two-stage hybrid framework to democratize ML-based threat detection. The first stage employs intentionally loose YARA rules for coarse-grained filtering, optimized for high recall. The second stage utilizes an ML classifier to filter out false positives from the first stage's output. To overcome data scarcity, the system leverages Simula, a seedless synthetic data generation framework, enabling security analysts to create high-quality training datasets without extensive data science expertise or pre-labeled examples. A continuous feedback loop incorporates real-time investigation results to adaptively tune the ML model, preventing rule degradation.
This proposed model with active learning has been rigorously tested for a prolonged time in a production environment spanning tens of thousands of systems. The system handles initial raw log volumes often reaching 250 billion events per day, significantly reducing them through filtering and ML inference to a handful of daily tickets for human investigation. Live experiments over an extended timeline demonstrate a general improvement in the model's precision over time due to the active learning feature. This approach offers a self-sustained, low-overhead, and low-maintenance solution, allowing security professionals to guide model learning as expert ``teachers''. 

---
# Can TabPFN Compete with GNNs for Node Classification via Graph Tabularization? 

**Authors**: Jeongwhan Choi, Woosung Kang, Minseo Kim, Jongwoo Kim, Noseong Park  

**Link**: [PDF](https://arxiv.org/pdf/2512.08798)  

**Abstract**: Foundation models pretrained on large data have demonstrated remarkable zero-shot generalization capabilities across domains. Building on the success of TabPFN for tabular data and its recent extension to time series, we investigate whether graph node classification can be effectively reformulated as a tabular learning problem. We introduce TabPFN-GN, which transforms graph data into tabular features by extracting node attributes, structural properties, positional encodings, and optionally smoothed neighborhood features. This enables TabPFN to perform direct node classification without any graph-specific training or language model dependencies. Our experiments on 12 benchmark datasets reveal that TabPFN-GN achieves competitive performance with GNNs on homophilous graphs and consistently outperforms them on heterophilous graphs. These results demonstrate that principled feature engineering can bridge the gap between tabular and graph domains, providing a practical alternative to task-specific GNN training and LLM-dependent graph foundation models. 

---
# MatteViT: High-Frequency-Aware Document Shadow Removal with Shadow Matte Guidance 

**Authors**: Chaewon Kim, Seoyeon Lee, Jonghyuk Park  

**Link**: [PDF](https://arxiv.org/pdf/2512.08789)  

**Abstract**: Document shadow removal is essential for enhancing the clarity of digitized documents. Preserving high-frequency details (e.g., text edges and lines) is critical in this process because shadows often obscure or distort fine structures. This paper proposes a matte vision transformer (MatteViT), a novel shadow removal framework that applies spatial and frequency-domain information to eliminate shadows while preserving fine-grained structural details. To effectively retain these details, we employ two preservation strategies. First, our method introduces a lightweight high-frequency amplification module (HFAM) that decomposes and adaptively amplifies high-frequency components. Second, we present a continuous luminance-based shadow matte, generated using a custom-built matte dataset and shadow matte generator, which provides precise spatial guidance from the earliest processing stage. These strategies enable the model to accurately identify fine-grained regions and restore them with high fidelity. Extensive experiments on public benchmarks (RDD and Kligler) demonstrate that MatteViT achieves state-of-the-art performance, providing a robust and practical solution for real-world document shadow removal. Furthermore, the proposed method better preserves text-level details in downstream tasks, such as optical character recognition, improving recognition performance over prior methods. 

---
# A Systematic Evaluation of Preference Aggregation in Federated RLHF for Pluralistic Alignment of LLMs 

**Authors**: Mahmoud Srewa, Tianyu Zhao, Salma Elmalaki  

**Link**: [PDF](https://arxiv.org/pdf/2512.08786)  

**Abstract**: This paper addresses the challenge of aligning large language models (LLMs) with diverse human preferences within federated learning (FL) environments, where standard methods often fail to adequately represent diverse viewpoints. We introduce a comprehensive evaluation framework that systematically assesses the trade-off between alignment quality and fairness when using different aggregation strategies for human preferences. In our federated setting, each group locally evaluates rollouts and produces reward signals, and the server aggregates these group-level rewards without accessing any raw data. Specifically, we evaluate standard reward aggregation techniques (min, max, and average) and introduce a novel adaptive scheme that dynamically adjusts preference weights based on a group's historical alignment performance. Our experiments on question-answering (Q/A) tasks using a PPO-based RLHF pipeline demonstrate that our adaptive approach consistently achieves superior fairness while maintaining competitive alignment scores. This work offers a robust methodology for evaluating LLM behavior across diverse populations and provides a practical solution for developing truly pluralistic and fairly aligned models. 

---
# Fluent Alignment with Disfluent Judges: Post-training for Lower-resource Languages 

**Authors**: David Samuel, Lilja Øvrelid, Erik Velldal, Andrey Kutuzov  

**Link**: [PDF](https://arxiv.org/pdf/2512.08777)  

**Abstract**: We propose a post-training method for lower-resource languages that preserves fluency of language models even when aligned by disfluent reward models. Preference-optimization is now a well-researched topic, but previous work has mostly addressed models for English and Chinese. Lower-resource languages lack both datasets written by native speakers and language models capable of generating fluent synthetic data. Thus, in this work, we focus on developing a fluent preference-aligned language model without any instruction-tuning data in the target language. Our approach uses an on-policy training method, which we compare with two common approaches: supervised finetuning on machine-translated data and multilingual finetuning. We conduct a case study on Norwegian Bokmål and evaluate fluency through native-speaker assessments. The results show that the on-policy aspect is crucial and outperforms the alternatives without relying on any hard-to-obtain data. 

---
# Refining Visual Artifacts in Diffusion Models via Explainable AI-based Flaw Activation Maps 

**Authors**: Seoyeon Lee, Gwangyeol Yu, Chaewon Kim, Jonghyuk Park  

**Link**: [PDF](https://arxiv.org/pdf/2512.08774)  

**Abstract**: Diffusion models have achieved remarkable success in image synthesis. However, addressing artifacts and unrealistic regions remains a critical challenge. We propose self-refining diffusion, a novel framework that enhances image generation quality by detecting these flaws. The framework employs an explainable artificial intelligence (XAI)-based flaw highlighter to produce flaw activation maps (FAMs) that identify artifacts and unrealistic regions. These FAMs improve reconstruction quality by amplifying noise in flawed regions during the forward process and by focusing on these regions during the reverse process. The proposed approach achieves up to a 27.3% improvement in Fréchet inception distance across various diffusion-based models, demonstrating consistently strong performance on diverse datasets. It also shows robust effectiveness across different tasks, including image generation, text-to-image generation, and inpainting. These results demonstrate that explainable AI techniques can extend beyond interpretability to actively contribute to image refinement. The proposed framework offers a versatile and effective approach applicable to various diffusion models and tasks, significantly advancing the field of image synthesis. 

---
# Data-Driven Dynamic Parameter Learning of manipulator robots 

**Authors**: Mohammed Elseiagy, Tsige Tadesse Alemayoh, Ranulfo Bezerra, Shotaro Kojima, Kazunori Ohno  

**Link**: [PDF](https://arxiv.org/pdf/2512.08767)  

**Abstract**: Bridging the sim-to-real gap remains a fundamental challenge in robotics, as accurate dynamic parameter estimation is essential for reliable model-based control, realistic simulation, and safe deployment of manipulators. Traditional analytical approaches often fall short when faced with complex robot structures and interactions. Data-driven methods offer a promising alternative, yet conventional neural networks such as recurrent models struggle to capture long-range dependencies critical for accurate estimation. In this study, we propose a Transformer-based approach for dynamic parameter estimation, supported by an automated pipeline that generates diverse robot models and enriched trajectory data using Jacobian-derived features. The dataset consists of 8,192 robots with varied inertial and frictional properties. Leveraging attention mechanisms, our model effectively captures both temporal and spatial dependencies. Experimental results highlight the influence of sequence length, sampling rate, and architecture, with the best configuration (sequence length 64, 64 Hz, four layers, 32 heads) achieving a validation R2 of 0.8633. Mass and inertia are estimated with near-perfect accuracy, Coulomb friction with moderate-to-high accuracy, while viscous friction and distal link center-of-mass remain more challenging. These results demonstrate that combining Transformers with automated dataset generation and kinematic enrichment enables scalable, accurate dynamic parameter estimation, contributing to improved sim-to-real transfer in robotic systems 

---
# Mitigating Individual Skin Tone Bias in Skin Lesion Classification through Distribution-Aware Reweighting 

**Authors**: Kuniko Paxton, Zeinab Dehghani, Koorosh Aslansefat, Dhavalkumar Thakker, Yiannis Papadopoulos  

**Link**: [PDF](https://arxiv.org/pdf/2512.08733)  

**Abstract**: Skin color has historically been a focal point of discrimination, yet fairness research in machine learning for medical imaging often relies on coarse subgroup categories, overlooking individual-level variations. Such group-based approaches risk obscuring biases faced by outliers within subgroups. This study introduces a distribution-based framework for evaluating and mitigating individual fairness in skin lesion classification. We treat skin tone as a continuous attribute rather than a categorical label, and employ kernel density estimation (KDE) to model its distribution. We further compare twelve statistical distance metrics to quantify disparities between skin tone distributions and propose a distance-based reweighting (DRW) loss function to correct underrepresentation in minority tones. Experiments across CNN and Transformer models demonstrate: (i) the limitations of categorical reweighting in capturing individual-level disparities, and (ii) the superior performance of distribution-based reweighting, particularly with Fidelity Similarity (FS), Wasserstein Distance (WD), Hellinger Metric (HM), and Harmonic Mean Similarity (HS). These findings establish a robust methodology for advancing fairness at individual level in dermatological AI systems, and highlight broader implications for sensitive continuous attributes in medical image analysis. 

---
# Multi-domain performance analysis with scores tailored to user preferences 

**Authors**: Sébastien Piérard, Adrien Deliège, Marc Van Droogenbroeck  

**Link**: [PDF](https://arxiv.org/pdf/2512.08715)  

**Abstract**: The performance of algorithms, methods, and models tends to depend heavily on the distribution of cases on which they are applied, this distribution being specific to the applicative domain. After performing an evaluation in several domains, it is highly informative to compute a (weighted) mean performance and, as shown in this paper, to scrutinize what happens during this averaging. To achieve this goal, we adopt a probabilistic framework and consider a performance as a probability measure (e.g., a normalized confusion matrix for a classification task). It appears that the corresponding weighted mean is known to be the summarization, and that only some remarkable scores assign to the summarized performance a value equal to a weighted arithmetic mean of the values assigned to the domain-specific performances. These scores include the family of ranking scores, a continuum parameterized by user preferences, and that the weights to consider in the arithmetic mean depend on the user preferences. Based on this, we rigorously define four domains, named easiest, most difficult, preponderant, and bottleneck domains, as functions of user preferences. After establishing the theory in a general setting, regardless of the task, we develop new visual tools for two-class classification. 

---
# Automatic Essay Scoring and Feedback Generation in Basque Language Learning 

**Authors**: Ekhi Azurmendi, Xabier Arregi, Oier Lopez de Lacalle  

**Link**: [PDF](https://arxiv.org/pdf/2512.08713)  

**Abstract**: This paper introduces the first publicly available dataset for Automatic Essay Scoring (AES) and feedback generation in Basque, targeting the CEFR C1 proficiency level. The dataset comprises 3,200 essays from HABE, each annotated by expert evaluators with criterion specific scores covering correctness, richness, coherence, cohesion, and task alignment enriched with detailed feedback and error examples. We fine-tune open-source models, including RoBERTa-EusCrawl and Latxa 8B/70B, for both scoring and explanation generation. Our experiments show that encoder models remain highly reliable for AES, while supervised fine-tuning (SFT) of Latxa significantly enhances performance, surpassing state-of-the-art (SoTA) closed-source systems such as GPT-5 and Claude Sonnet 4.5 in scoring consistency and feedback quality. We also propose a novel evaluation methodology for assessing feedback generation, combining automatic consistency metrics with expert-based validation of extracted learner errors. Results demonstrate that the fine-tuned Latxa model produces criterion-aligned, pedagogically meaningful feedback and identifies a wider range of error types than proprietary models. This resource and benchmark establish a foundation for transparent, reproducible, and educationally grounded NLP research in low-resource languages such as Basque. 

---
# Reusability in MLOps: Leveraging Ports and Adapters to Build a Microservices Architecture for the Maritime Domain 

**Authors**: Renato Cordeiro Ferreira, Aditya Dhinavahi, Rowanne Trapmann, Willem-Jan van den Heuvel  

**Link**: [PDF](https://arxiv.org/pdf/2512.08657)  

**Abstract**: ML-Enabled Systems (MLES) are inherently complex since they require multiple components to achieve their business goal. This experience report showcases the software architecture reusability techniques applied while building Ocean Guard, an MLES for anomaly detection in the maritime domain. In particular, it highlights the challenges and lessons learned to reuse the Ports and Adapters pattern to support building multiple microservices from a single codebase. This experience report hopes to inspire software engineers, machine learning engineers, and data scientists to apply the Hexagonal Architecture pattern to build their MLES. 

---
# Aerial Vision-Language Navigation with a Unified Framework for Spatial, Temporal and Embodied Reasoning 

**Authors**: Huilin Xu, Zhuoyang Liu, Yixiang Luomei, Feng Xu  

**Link**: [PDF](https://arxiv.org/pdf/2512.08639)  

**Abstract**: Aerial Vision-and-Language Navigation (VLN) aims to enable unmanned aerial vehicles (UAVs) to interpret natural language instructions and navigate complex urban environments using onboard visual observation. This task holds promise for real-world applications such as low-altitude inspection, search-and-rescue, and autonomous aerial delivery. Existing methods often rely on panoramic images, depth inputs, or odometry to support spatial reasoning and action planning. These requirements increase system cost and integration complexity, thus hindering practical deployment for lightweight UAVs. We present a unified aerial VLN framework that operates solely on egocentric monocular RGB observations and natural language instructions. The model formulates navigation as a next-token prediction problem, jointly optimizing spatial perception, trajectory reasoning, and action prediction through prompt-guided multi-task learning. Moreover, we propose a keyframe selection strategy to reduce visual redundancy by retaining semantically informative frames, along with an action merging and label reweighting mechanism that mitigates long-tailed supervision imbalance and facilitates stable multi-task co-training. Extensive experiments on the Aerial VLN benchmark validate the effectiveness of our method. Under the challenging monocular RGB-only setting, our model achieves strong results across both seen and unseen environments. It significantly outperforms existing RGB-only baselines and narrows the performance gap with state-of-the-art panoramic RGB-D counterparts. Comprehensive ablation studies further demonstrate the contribution of our task design and architectural choices. 

---
# Decoupling Template Bias in CLIP: Harnessing Empty Prompts for Enhanced Few-Shot Learning 

**Authors**: Zhenyu Zhang, Guangyao Chen, Yixiong Zou, Zhimeng Huang, Yuhua Li  

**Link**: [PDF](https://arxiv.org/pdf/2512.08606)  

**Abstract**: The Contrastive Language-Image Pre-Training (CLIP) model excels in few-shot learning by aligning visual and textual representations. Our study shows that template-sample similarity (TSS), defined as the resemblance between a text template and an image sample, introduces bias. This bias leads the model to rely on template proximity rather than true sample-to-category alignment, reducing both accuracy and robustness in classification. We present a framework that uses empty prompts, textual inputs that convey the idea of "emptiness" without category information. These prompts capture unbiased template features and offset TSS bias. The framework employs two stages. During pre-training, empty prompts reveal and reduce template-induced bias within the CLIP encoder. During few-shot fine-tuning, a bias calibration loss enforces correct alignment between images and their categories, ensuring the model focuses on relevant visual cues. Experiments across multiple benchmarks demonstrate that our template correction method significantly reduces performance fluctuations caused by TSS, yielding higher classification accuracy and stronger robustness. The repository of this project is available at this https URL. 

---
# Examining Student Interactions with a Pedagogical AI-Assistant for Essay Writing and their Impact on Students Writing Quality 

**Authors**: Wicaksono Febriantoro, Qi Zhou, Wannapon Suraworachet, Sahan Bulathwela, Andrea Gauthier, Eva Millan, Mutlu Cukurova  

**Link**: [PDF](https://arxiv.org/pdf/2512.08596)  

**Abstract**: The dynamic nature of interactions between students and GenAI, as well as their relationship to writing quality, remains underexplored. While most research has examined how general-purpose GenAI can support writing, fewer studies have investigated how students interact with pedagogically designed systems across different phases of the writing process. To address this gap, we evaluated a GenAI-driven essay-writing assistant (EWA) designed to support higher education students in argumentative writing. Drawing on 1,282 interaction logs from 32 undergraduates during a two-hour writing session, Sequential Pattern Mining and K-Means clustering were used to identify behavioral patterns. Two clusters emerged: Cluster 1 emphasized outline planning and essay structure, while Cluster 2 focused on content development. A Mann-Whitney U test revealed a moderate effect size (r = 0.36) in the essay Organization dimension, with Cluster 1 showing higher scores. Qualitative analysis indicated that students with better performance actively wrote and shared essay sections with EWA for feedback, rather than interacted passively by asking questions. These findings suggest implications for teaching and system design. Teachers can encourage active engagement, while future EWAs may integrate automatic labeling and monitoring to prompt students to move from questioning to writing, enabling fuller benefits from GenAI-supported learning. 

---
# Mind to Hand: Purposeful Robotic Control via Embodied Reasoning 

**Authors**: Peijun Tang, Shangjin Xie, Binyan Sun, Baifu Huang, Kuncheng Luo, Haotian Yang, Weiqi Jin, Jianan Wang  

**Link**: [PDF](https://arxiv.org/pdf/2512.08580)  

**Abstract**: Humans act with context and intention, with reasoning playing a central role. While internet-scale data has enabled broad reasoning capabilities in AI systems, grounding these abilities in physical action remains a major challenge. We introduce Lumo-1, a generalist vision-language-action (VLA) model that unifies robot reasoning ("mind") with robot action ("hand"). Our approach builds upon the general multi-modal reasoning capabilities of pre-trained vision-language models (VLMs), progressively extending them to embodied reasoning and action prediction, and ultimately towards structured reasoning and reasoning-action alignment. This results in a three-stage pre-training pipeline: (1) Continued VLM pre-training on curated vision-language data to enhance embodied reasoning skills such as planning, spatial understanding, and trajectory prediction; (2) Co-training on cross-embodiment robot data alongside vision-language data; and (3) Action training with reasoning process on trajectories collected on Astribot S1, a bimanual mobile manipulator with human-like dexterity and agility. Finally, we integrate reinforcement learning to further refine reasoning-action consistency and close the loop between semantic inference and motor control. Extensive experiments demonstrate that Lumo-1 achieves significant performance improvements in embodied vision-language reasoning, a critical component for generalist robotic control. Real-world evaluations further show that Lumo-1 surpasses strong baselines across a wide range of challenging robotic tasks, with strong generalization to novel objects and environments, excelling particularly in long-horizon tasks and responding to human-natural instructions that require reasoning over strategy, concepts and space. 

---
# Disturbance-Free Surgical Video Generation from Multi-Camera Shadowless Lamps for Open Surgery 

**Authors**: Yuna Kato, Shohei Mori, Hideo Saito, Yoshifumi Takatsume, Hiroki Kajita, Mariko Isogawa  

**Link**: [PDF](https://arxiv.org/pdf/2512.08577)  

**Abstract**: Video recordings of open surgeries are greatly required for education and research purposes. However, capturing unobstructed videos is challenging since surgeons frequently block the camera field of view. To avoid occlusion, the positions and angles of the camera must be frequently adjusted, which is highly labor-intensive. Prior work has addressed this issue by installing multiple cameras on a shadowless lamp and arranging them to fully surround the surgical area. This setup increases the chances of some cameras capturing an unobstructed view. However, manual image alignment is needed in post-processing since camera configurations change every time surgeons move the lamp for optimal lighting. This paper aims to fully automate this alignment task. The proposed method identifies frames in which the lighting system moves, realigns them, and selects the camera with the least occlusion to generate a video that consistently presents the surgical field from a fixed perspective. A user study involving surgeons demonstrated that videos generated by our method were superior to those produced by conventional methods in terms of the ease of confirming the surgical area and the comfort during video viewing. Additionally, our approach showed improvements in video quality over existing techniques. Furthermore, we implemented several synthesis options for the proposed view-synthesis method and conducted a user study to assess surgeons' preferences for each option. 

---
# A Hybrid Model for Stock Market Forecasting: Integrating News Sentiment and Time Series Data with Graph Neural Networks 

**Authors**: Nader Sadek, Mirette Moawad, Christina Naguib, Mariam Elzahaby  

**Link**: [PDF](https://arxiv.org/pdf/2512.08567)  

**Abstract**: Stock market prediction is a long-standing challenge in finance, as accurate forecasts support informed investment decisions. Traditional models rely mainly on historical prices, but recent work shows that financial news can provide useful external signals. This paper investigates a multimodal approach that integrates companies' news articles with their historical stock data to improve prediction performance. We compare a Graph Neural Network (GNN) model with a baseline LSTM model. Historical data for each company is encoded using an LSTM, while news titles are embedded with a language model. These embeddings form nodes in a heterogeneous graph, and GraphSAGE is used to capture interactions between articles, companies, and industries. We evaluate two targets: a binary direction-of-change label and a significance-based label. Experiments on the US equities and Bloomberg datasets show that the GNN outperforms the LSTM baseline, achieving 53% accuracy on the first target and a 4% precision gain on the second. Results also indicate that companies with more associated news yield higher prediction accuracy. Moreover, headlines contain stronger predictive signals than full articles, suggesting that concise news summaries play an important role in short-term market reactions. 

---
# Bridging Scale Discrepancies in Robotic Control via Language-Based Action Representations 

**Authors**: Yuchi Zhang, Churui Sun, Shiqi Liang, Diyuan Liu, Chao Ji, Wei-Nan Zhang, Ting Liu  

**Link**: [PDF](https://arxiv.org/pdf/2512.08548)  

**Abstract**: Recent end-to-end robotic manipulation research increasingly adopts architectures inspired by large language models to enable robust manipulation. However, a critical challenge arises from severe distribution shifts between robotic action data, primarily due to substantial numerical variations in action commands across diverse robotic platforms and tasks, hindering the effective transfer of pretrained knowledge. To address this limitation, we propose a semantically grounded linguistic representation to normalize actions for efficient pretraining. Unlike conventional discretized action representations that are sensitive to numerical scales, the motion representation specifically disregards numeric scale effects, emphasizing directionality instead. This abstraction mitigates distribution shifts, yielding a more generalizable pretraining representation. Moreover, using the motion representation narrows the feature distance between action tokens and standard vocabulary tokens, mitigating modality gaps. Multi-task experiments on two benchmarks demonstrate that the proposed method significantly improves generalization performance and transferability in robotic manipulation tasks. 

---
# Curriculum Guided Massive Multi Agent System Solving For Robust Long Horizon Tasks 

**Authors**: Indrajit Kar, Kalathur Chenchu Kishore Kumar  

**Link**: [PDF](https://arxiv.org/pdf/2512.08545)  

**Abstract**: Large Language Models and multi-agent systems have shown promise in decomposing complex tasks, yet they struggle with long-horizon reasoning tasks and escalating computation cost. This work introduces a hierarchical multi-agent architecture that distributes reasoning across a 64*64 grid of lightweight agents, supported by a selective oracle. A spatial curriculum progressively expands the operational region of the grid, ensuring that agents master easier central tasks before tackling harder peripheral ones. To improve reliability, the system integrates Negative Log-Likelihood as a measure of confidence, allowing the curriculum to prioritize regions where agents are both accurate and well calibrated. A Thompson Sampling curriculum manager adaptively chooses training zones based on competence and NLL-driven reward signals. We evaluate the approach on a spatially grounded Tower of Hanoi benchmark, which mirrors the long-horizon structure of many robotic manipulation and planning tasks. Results demonstrate improved stability, reduced oracle usage, and stronger long-range reasoning from distributed agent cooperation. 

---
# A Novel Wasserstein Quaternion Generative Adversarial Network for Color Image Generation 

**Authors**: Zhigang Jia, Duan Wang, Hengkai Wang, Yajun Xie, Meixiang Zhao, Xiaoyu Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2512.08542)  

**Abstract**: Color image generation has a wide range of applications, but the existing generation models ignore the correlation among color channels, which may lead to chromatic aberration problems. In addition, the data distribution problem of color images has not been systematically elaborated and explained, so that there is still the lack of the theory about measuring different color images datasets. In this paper, we define a new quaternion Wasserstein distance and develop its dual theory. To deal with the quaternion linear programming problem, we derive the strong duality form with helps of quaternion convex set separation theorem and quaternion Farkas lemma. With using quaternion Wasserstein distance, we propose a novel Wasserstein quaternion generative adversarial network. Experiments demonstrate that this novel model surpasses both the (quaternion) generative adversarial networks and the Wasserstein generative adversarial network in terms of generation efficiency and image quality. 

---
# SensHRPS: Sensing Comfortable Human-Robot Proxemics and Personal Space With Eye-Tracking 

**Authors**: Nadezhda Kushina, Ko Watanabe, Aarthi Kannan, Ashita Ashok, Andreas Dengel, Karsten Berns  

**Link**: [PDF](https://arxiv.org/pdf/2512.08518)  

**Abstract**: Social robots must adjust to human proxemic norms to ensure user comfort and engagement. While prior research demonstrates that eye-tracking features reliably estimate comfort in human-human interactions, their applicability to interactions with humanoid robots remains unexplored. In this study, we investigate user comfort with the robot "Ameca" across four experimentally controlled distances (0.5 m to 2.0 m) using mobile eye-tracking and subjective reporting (N=19). We evaluate multiple machine learning and deep learning models to estimate comfort based on gaze features. Contrary to previous human-human studies where Transformer models excelled, a Decision Tree classifier achieved the highest performance (F1-score = 0.73), with minimum pupil diameter identified as the most critical predictor. These findings suggest that physiological comfort thresholds in human-robot interaction differ from human-human dynamics and can be effectively modeled using interpretable logic. 

---
# Disrupting Hierarchical Reasoning: Adversarial Protection for Geographic Privacy in Multimodal Reasoning Models 

**Authors**: Jiaming Zhang, Che Wang, Yang Cao, Longtao Huang, Wei Yang Bryan Lim  

**Link**: [PDF](https://arxiv.org/pdf/2512.08503)  

**Abstract**: Multi-modal large reasoning models (MLRMs) pose significant privacy risks by inferring precise geographic locations from personal images through hierarchical chain-of-thought reasoning. Existing privacy protection techniques, primarily designed for perception-based models, prove ineffective against MLRMs' sophisticated multi-step reasoning processes that analyze environmental cues. We introduce \textbf{ReasonBreak}, a novel adversarial framework specifically designed to disrupt hierarchical reasoning in MLRMs through concept-aware perturbations. Our approach is founded on the key insight that effective disruption of geographic reasoning requires perturbations aligned with conceptual hierarchies rather than uniform noise. ReasonBreak strategically targets critical conceptual dependencies within reasoning chains, generating perturbations that invalidate specific inference steps and cascade through subsequent reasoning stages. To facilitate this approach, we contribute \textbf{GeoPrivacy-6K}, a comprehensive dataset comprising 6,341 ultra-high-resolution images ($\geq$2K) with hierarchical concept annotations. Extensive evaluation across seven state-of-the-art MLRMs (including GPT-o3, GPT-5, Gemini 2.5 Pro) demonstrates ReasonBreak's superior effectiveness, achieving a 14.4\% improvement in tract-level protection (33.8\% vs 19.4\%) and nearly doubling block-level protection (33.5\% vs 16.8\%). This work establishes a new paradigm for privacy protection against reasoning-based threats. 

---
# Developing Distance-Aware Uncertainty Quantification Methods in Physics-Guided Neural Networks for Reliable Bearing Health Prediction 

**Authors**: Waleed Razzaq, Yun-Bo Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2512.08499)  

**Abstract**: Accurate and uncertainty-aware degradation estimation is essential for predictive maintenance in safety-critical systems like rotating machinery with rolling-element bearings. Many existing uncertainty methods lack confidence calibration, are costly to run, are not distance-aware, and fail to generalize under out-of-distribution data. We introduce two distance-aware uncertainty methods for deterministic physics-guided neural networks: PG-SNGP, based on Spectral Normalization Gaussian Process, and PG-SNER, based on Deep Evidential Regression. We apply spectral normalization to the hidden layers so the network preserves distances from input to latent space. PG-SNGP replaces the final dense layer with a Gaussian Process layer for distance-sensitive uncertainty, while PG-SNER outputs Normal Inverse Gamma parameters to model uncertainty in a coherent probabilistic form. We assess performance using standard accuracy metrics and a new distance-aware metric based on the Pearson Correlation Coefficient, which measures how well predicted uncertainty tracks the distance between test and training samples. We also design a dynamic weighting scheme in the loss to balance data fidelity and physical consistency. We test our methods on rolling-element bearing degradation using the PRONOSTIA dataset and compare them with Monte Carlo and Deep Ensemble PGNNs. Results show that PG-SNGP and PG-SNER improve prediction accuracy, generalize reliably under OOD conditions, and remain robust to adversarial attacks and noise. 

---
# LLM-based Vulnerable Code Augmentation: Generate or Refactor? 

**Authors**: Dyna Soumhane Ouchebara, Stéphane Dupont  

**Link**: [PDF](https://arxiv.org/pdf/2512.08493)  

**Abstract**: Vulnerability code-bases often suffer from severe imbalance, limiting the effectiveness of Deep Learning-based vulnerability classifiers. Data Augmentation could help solve this by mitigating the scarcity of under-represented CWEs. In this context, we investigate LLM-based augmentation for vulnerable functions, comparing controlled generation of new vulnerable samples with semantics-preserving refactoring of existing ones. Using Qwen2.5-Coder to produce augmented data and CodeBERT as a vulnerability classifier on the SVEN dataset, we find that our approaches are indeed effective in enriching vulnerable code-bases through a simple process and with reasonable quality, and that a hybrid strategy best boosts vulnerability classifiers' performance. 

---
# Visionary: The World Model Carrier Built on WebGPU-Powered Gaussian Splatting Platform 

**Authors**: Yuning Gong, Yifei Liu, Yifan Zhan, Muyao Niu, Xueying Li, Yuanjun Liao, Jiaming Chen, Yuanyuan Gao, Jiaqi Chen, Minming Chen, Li Zhou, Yuning Zhang, Wei Wang, Xiaoqing Hou, Huaxi Huang, Shixiang Tang, Le Ma, Dingwen Zhang, Xue Yang, Junchi Yan, Yanchi Zhang, Yinqiang Zheng, Xiao Sun, Zhihang Zhong  

**Link**: [PDF](https://arxiv.org/pdf/2512.08478)  

**Abstract**: Neural rendering, particularly 3D Gaussian Splatting (3DGS), has evolved rapidly and become a key component for building world models. However, existing viewer solutions remain fragmented, heavy, or constrained by legacy pipelines, resulting in high deployment friction and limited support for dynamic content and generative models. In this work, we present Visionary, an open, web-native platform for real-time various Gaussian Splatting and meshes rendering. Built on an efficient WebGPU renderer with per-frame ONNX inference, Visionary enables dynamic neural processing while maintaining a lightweight, "click-to-run" browser experience. It introduces a standardized Gaussian Generator contract, which not only supports standard 3DGS rendering but also allows plug-and-play algorithms to generate or update Gaussians each frame. Such inference also enables us to apply feedforward generative post-processing. The platform further offers a plug in this http URL library with a concise TypeScript API for seamless integration into existing web applications. Experiments show that, under identical 3DGS assets, Visionary achieves superior rendering efficiency compared to current Web viewers due to GPU-based primitive sorting. It already supports multiple variants, including MLP-based 3DGS, 4DGS, neural avatars, and style transformation or enhancement networks. By unifying inference and rendering directly in the browser, Visionary significantly lowers the barrier to reproduction, comparison, and deployment of 3DGS-family methods, serving as a unified World Model Carrier for both reconstructive and generative paradigms. 

---
# ContextDrag: Precise Drag-Based Image Editing via Context-Preserving Token Injection and Position-Consistent Attention 

**Authors**: Huiguo He, Pengyu Yan, Ziqi Yi, Weizhi Zhong, Zheng Liu, Yejun Tang, Huan Yang, Kun Gai, Guanbin Li, Lianwen Jin  

**Link**: [PDF](https://arxiv.org/pdf/2512.08477)  

**Abstract**: Drag-based image editing aims to modify visual content followed by user-specified drag operations. Despite existing methods having made notable progress, they still fail to fully exploit the contextual information in the reference image, including fine-grained texture details, leading to edits with limited coherence and fidelity. To address this challenge, we introduce ContextDrag, a new paradigm for drag-based editing that leverages the strong contextual modeling capability of editing models, such as FLUX-Kontext. By incorporating VAE-encoded features from the reference image, ContextDrag can leverage rich contextual cues and preserve fine-grained details, without the need for finetuning or inversion. Specifically, ContextDrag introduced a novel Context-preserving Token Injection (CTI) that injects noise-free reference features into their correct destination locations via a Latent-space Reverse Mapping (LRM) algorithm. This strategy enables precise drag control while preserving consistency in both semantics and texture details. Second, ContextDrag adopts a novel Position-Consistent Attention (PCA), which positional re-encodes the reference tokens and applies overlap-aware masking to eliminate interference from irrelevant reference features. Extensive experiments on DragBench-SR and DragBench-DR demonstrate that our approach surpasses all existing SOTA methods. Code will be publicly available. 

---
# Biothreat Benchmark Generation Framework for Evaluating Frontier AI Models III: Implementing the Bacterial Biothreat Benchmark (B3) Dataset 

**Authors**: Gary Ackerman, Theodore Wilson, Zachary Kallenborn, Olivia Shoemaker, Anna Wetzel, Hayley Peterson, Abigail Danfora, Jenna LaTourette, Brandon Behlendorf, Douglas Clifford  

**Link**: [PDF](https://arxiv.org/pdf/2512.08459)  

**Abstract**: The potential for rapidly-evolving frontier artificial intelligence (AI) models, especially large language models (LLMs), to facilitate bioterrorism or access to biological weapons has generated significant policy, academic, and public concern. Both model developers and policymakers seek to quantify and mitigate any risk, with an important element of such efforts being the development of model benchmarks that can assess the biosecurity risk posed by a particular model. This paper discusses the pilot implementation of the Bacterial Biothreat Benchmark (B3) dataset. It is the third in a series of three papers describing an overall Biothreat Benchmark Generation (BBG) framework, with previous papers detailing the development of the B3 dataset. The pilot involved running the benchmarks through a sample frontier AI model, followed by human evaluation of model responses, and an applied risk analysis of the results along several dimensions. Overall, the pilot demonstrated that the B3 dataset offers a viable, nuanced method for rapidly assessing the biosecurity risk posed by a LLM, identifying the key sources of that risk and providing guidance for priority areas of mitigation priority. 

---
# Biothreat Benchmark Generation Framework for Evaluating Frontier AI Models II: Benchmark Generation Process 

**Authors**: Gary Ackerman, Zachary Kallenborn, Anna Wetzel, Hayley Peterson, Jenna LaTourette, Olivia Shoemaker, Brandon Behlendorf, Sheriff Almakki, Doug Clifford, Noah Sheinbaum  

**Link**: [PDF](https://arxiv.org/pdf/2512.08451)  

**Abstract**: The potential for rapidly-evolving frontier artificial intelligence (AI) models, especially large language models (LLMs), to facilitate bioterrorism or access to biological weapons has generated significant policy, academic, and public concern. Both model developers and policymakers seek to quantify and mitigate any risk, with an important element of such efforts being the development of model benchmarks that can assess the biosecurity risk posed by a particular model. This paper, the second in a series of three, describes the second component of a novel Biothreat Benchmark Generation (BBG) framework: the generation of the Bacterial Biothreat Benchmark (B3) dataset. The development process involved three complementary approaches: 1) web-based prompt generation, 2) red teaming, and 3) mining existing benchmark corpora, to generate over 7,000 potential benchmarks linked to the Task-Query Architecture that was developed during the first component of the project. A process of de-duplication, followed by an assessment of uplift diagnosticity, and general quality control measures, reduced the candidates to a set of 1,010 final benchmarks. This procedure ensured that these benchmarks are a) diagnostic in terms of providing uplift; b) directly relevant to biosecurity threats; and c) are aligned with a larger biosecurity architecture permitting nuanced analysis at different levels of analysis. 

---
# Are generative AI text annotations systematically biased? 

**Authors**: Sjoerd B. Stolwijk, Mark Boukes, Damian Trilling  

**Link**: [PDF](https://arxiv.org/pdf/2512.08404)  

**Abstract**: This paper investigates bias in GLLM annotations by conceptually replicating manual annotations of Boukes (2024). Using various GLLMs (Llama3.1:8b, Llama3.3:70b, GPT4o, Qwen2.5:72b) in combination with five different prompts for five concepts (political content, interactivity, rationality, incivility, and ideology). We find GLLMs perform adequate in terms of F1 scores, but differ from manual annotations in terms of prevalence, yield substantively different downstream results, and display systematic bias in that they overlap more with each other than with manual annotations. Differences in F1 scores fail to account for the degree of bias. 

---
# Conditional Morphogenesis: Emergent Generation of Structural Digits via Neural Cellular Automata 

**Authors**: Ali Sakour  

**Link**: [PDF](https://arxiv.org/pdf/2512.08360)  

**Abstract**: Biological systems exhibit remarkable morphogenetic plasticity, where a single genome can encode various specialized cellular structures triggered by local chemical signals. In the domain of Deep Learning, Differentiable Neural Cellular Automata (NCA) have emerged as a paradigm to mimic this self-organization. However, existing NCA research has predominantly focused on continuous texture synthesis or single-target object recovery, leaving the challenge of class-conditional structural generation largely unexplored. In this work, we propose a novel Conditional Neural Cellular Automata (c-NCA) architecture capable of growing distinct topological structures - specifically MNIST digits - from a single generic seed, guided solely by a spatially broadcasted class vector. Unlike traditional generative models (e.g., GANs, VAEs) that rely on global reception fields, our model enforces strict locality and translation equivariance. We demonstrate that by injecting a one-hot condition into the cellular perception field, a single set of local rules can learn to break symmetry and self-assemble into ten distinct geometric attractors. Experimental results show that our c-NCA achieves stable convergence, correctly forming digit topologies from a single pixel, and exhibits robustness characteristic of biological systems. This work bridges the gap between texture-based NCAs and structural pattern formation, offering a lightweight, biologically plausible alternative for conditional generation. 

---
# Robust Finetuning of Vision-Language-Action Robot Policies via Parameter Merging 

**Authors**: Yajat Yadav, Zhiyuan Zhou, Andrew Wagenmaker, Karl Pertsch, Sergey Levine  

**Link**: [PDF](https://arxiv.org/pdf/2512.08333)  

**Abstract**: Generalist robot policies, trained on large and diverse datasets, have demonstrated the ability to generalize across a wide spectrum of behaviors, enabling a single policy to act in varied real-world environments. However, they still fall short on new tasks not covered in the training data. When finetuned on limited demonstrations of a new task, these policies often overfit to the specific demonstrations--not only losing their prior abilities to solve a wide variety of generalist tasks but also failing to generalize within the new task itself. In this work, we aim to develop a method that preserves the generalization capabilities of the generalist policy during finetuning, allowing a single policy to robustly incorporate a new skill into its repertoire. Our goal is a single policy that both learns to generalize to variations of the new task and retains the broad competencies gained from pretraining. We show that this can be achieved through a simple yet effective strategy: interpolating the weights of a finetuned model with that of the pretrained model. We show, across extensive simulated and real-world experiments, that such model merging produces a single model that inherits the generalist abilities of the base model and learns to solve the new task robustly, outperforming both the pretrained and finetuned model on out-of-distribution variations of the new task. Moreover, we show that model merging enables continual acquisition of new skills in a lifelong learning setting, without sacrificing previously learned generalist abilities. 

---
# Interpreting Structured Perturbations in Image Protection Methods for Diffusion Models 

**Authors**: Michael R. Martin, Garrick Chan, Kwan-Liu Ma  

**Link**: [PDF](https://arxiv.org/pdf/2512.08329)  

**Abstract**: Recent image protection mechanisms such as Glaze and Nightshade introduce imperceptible, adversarially designed perturbations intended to disrupt downstream text-to-image generative models. While their empirical effectiveness is known, the internal structure, detectability, and representational behavior of these perturbations remain poorly understood. This study provides a systematic, explainable AI analysis using a unified framework that integrates white-box feature-space inspection and black-box signal-level probing. Through latent-space clustering, feature-channel activation analysis, occlusion-based spatial sensitivity mapping, and frequency-domain characterization, we show that protection mechanisms operate as structured, low-entropy perturbations tightly coupled to underlying image content across representational, spatial, and spectral domains. Protected images preserve content-driven feature organization with protection-specific substructure rather than inducing global representational drift. Detectability is governed by interacting effects of perturbation entropy, spatial deployment, and frequency alignment, with sequential protection amplifying detectable structure rather than suppressing it. Frequency-domain analysis shows that Glaze and Nightshade redistribute energy along dominant image-aligned frequency axes rather than introducing diffuse noise. These findings indicate that contemporary image protection operates through structured feature-level deformation rather than semantic dislocation, explaining why protection signals remain visually subtle yet consistently detectable. This work advances the interpretability of adversarial image protection and informs the design of future defenses and detection strategies for generative AI systems. 

---
# Argus: A Multi-Agent Sensitive Information Leakage Detection Framework Based on Hierarchical Reference Relationships 

**Authors**: Bin Wang, Hui Li, Liyang Zhang, Qijia Zhuang, Ao Yang, Dong Zhang, Xijun Luo, Bing Lin  

**Link**: [PDF](https://arxiv.org/pdf/2512.08326)  

**Abstract**: Sensitive information leakage in code repositories has emerged as a critical security challenge. Traditional detection methods that rely on regular expressions, fingerprint features, and high-entropy calculations often suffer from high false-positive rates. This not only reduces detection efficiency but also significantly increases the manual screening burden on developers. Recent advances in large language models (LLMs) and multi-agent collaborative architectures have demonstrated remarkable potential for tackling complex tasks, offering a novel technological perspective for sensitive information detection. In response to these challenges, we propose Argus, a multi-agent collaborative framework for detecting sensitive information. Argus employs a three-tier detection mechanism that integrates key content, file context, and project reference relationships to effectively reduce false positives and enhance overall detection accuracy. To comprehensively evaluate Argus in real-world repository environments, we developed two new benchmarks, one to assess genuine leak detection capabilities and another to evaluate false-positive filtering performance. Experimental results show that Argus achieves up to 94.86% accuracy in leak detection, with a precision of 96.36%, recall of 94.64%, and an F1 score of 0.955. Moreover, the analysis of 97 real repositories incurred a total cost of only 2.2$. All code implementations and related datasets are publicly available at this https URL for further research and application. 

---
# GeoDM: Geometry-aware Distribution Matching for Dataset Distillation 

**Authors**: Xuhui Li, Zhengquan Luo, Zihui Cui, Zhiqiang Xu  

**Link**: [PDF](https://arxiv.org/pdf/2512.08317)  

**Abstract**: Dataset distillation aims to synthesize a compact subset of the original data, enabling models trained on it to achieve performance comparable to those trained on the original large dataset. Existing distribution-matching methods are confined to Euclidean spaces, making them only capture linear structures and overlook the intrinsic geometry of real data, e.g., curvature. However, high-dimensional data often lie on low-dimensional manifolds, suggesting that dataset distillation should have the distilled data manifold aligned with the original data manifold. In this work, we propose a geometry-aware distribution-matching framework, called \textbf{GeoDM}, which operates in the Cartesian product of Euclidean, hyperbolic, and spherical manifolds, with flat, hierarchical, and cyclical structures all captured by a unified representation. To adapt to the underlying data geometry, we introduce learnable curvature and weight parameters for three kinds of geometries. At the same time, we design an optimal transport loss to enhance the distribution fidelity. Our theoretical analysis shows that the geometry-aware distribution matching in a product space yields a smaller generalization error bound than the Euclidean counterparts. Extensive experiments conducted on standard benchmarks demonstrate that our algorithm outperforms state-of-the-art data distillation methods and remains effective across various distribution-matching strategies for the single geometries. 

---
# Terrain Diffusion: A Diffusion-Based Successor to Perlin Noise in Infinite, Real-Time Terrain Generation 

**Authors**: Alexander Goslin  

**Link**: [PDF](https://arxiv.org/pdf/2512.08309)  

**Abstract**: For decades, procedural worlds have been built on procedural noise functions such as Perlin noise, which are fast and infinite, yet fundamentally limited in realism and large-scale coherence. We introduce Terrain Diffusion, an AI-era successor to Perlin noise that bridges the fidelity of diffusion models with the properties that made procedural noise indispensable: seamless infinite extent, seed-consistency, and constant-time random access. At its core is InfiniteDiffusion, a novel algorithm for infinite generation, enabling seamless, real-time synthesis of boundless landscapes. A hierarchical stack of diffusion models couples planetary context with local detail, while a compact Laplacian encoding stabilizes outputs across Earth-scale dynamic ranges. An open-source infinite-tensor framework supports constant-memory manipulation of unbounded tensors, and few-step consistency distillation enables efficient generation. Together, these components establish diffusion models as a practical foundation for procedural world generation, capable of synthesizing entire planets coherently, controllably, and without limits. 

---
# Systematization of Knowledge: Security and Safety in the Model Context Protocol Ecosystem 

**Authors**: Shiva Gaire, Srijan Gyawali, Saroj Mishra, Suman Niroula, Dilip Thakur, Umesh Yadav  

**Link**: [PDF](https://arxiv.org/pdf/2512.08290)  

**Abstract**: The Model Context Protocol (MCP) has emerged as the de facto standard for connecting Large Language Models (LLMs) to external data and tools, effectively functioning as the "USB-C for Agentic AI." While this decoupling of context and execution solves critical interoperability challenges, it introduces a profound new threat landscape where the boundary between epistemic errors (hallucinations) and security breaches (unauthorized actions) dissolves. This Systematization of Knowledge (SoK) aims to provide a comprehensive taxonomy of risks in the MCP ecosystem, distinguishing between adversarial security threats (e.g., indirect prompt injection, tool poisoning) and epistemic safety hazards (e.g., alignment failures in distributed tool delegation). We analyze the structural vulnerabilities of MCP primitives, specifically Resources, Prompts, and Tools, and demonstrate how "context" can be weaponized to trigger unauthorized operations in multi-agent environments. Furthermore, we survey state-of-the-art defenses, ranging from cryptographic provenance (ETDI) to runtime intent verification, and conclude with a roadmap for securing the transition from conversational chatbots to autonomous agentic operating systems. 

---
# Empowering smart app development with SolidGPT: an edge-cloud hybrid AI agent framework 

**Authors**: Liao Hu, Qiteng Wu, Ruoyu Qi  

**Link**: [PDF](https://arxiv.org/pdf/2512.08286)  

**Abstract**: The integration of Large Language Models (LLMs) into mobile and software development workflows faces a persistent tension among three demands: semantic awareness, developer productivity, and data privacy. Traditional cloud-based tools offer strong reasoning but risk data exposure and latency, while on-device solutions lack full-context understanding across codebase and developer tooling. We introduce SolidGPT, an open-source, edge-cloud hybrid developer assistant built on GitHub, designed to enhance code and workspace semantic search. SolidGPT enables developers to: talk to your codebase: interactively query code and project structure, discovering the right methods and modules without manual searching. Automate software project workflows: generate PRDs, task breakdowns, Kanban boards, and even scaffold web app beginnings, with deep integration via VSCode and Notion. Configure private, extensible agents: onboard private code folders (up to approximately 500 files), connect Notion, customize AI agent personas via embedding and in-context training, and deploy via Docker, CLI, or VSCode extension. In practice, SolidGPT empowers developer productivity through: Semantic-rich code navigation: no more hunting through files or wondering where a feature lives. Integrated documentation and task management: seamlessly sync generated PRD content and task boards into developer workflows. Privacy-first design: running locally via Docker or VSCode, with full control over code and data, while optionally reaching out to LLM APIs as needed. By combining interactive code querying, automated project scaffolding, and human-AI collaboration, SolidGPT provides a practical, privacy-respecting edge assistant that accelerates real-world development workflows, ideal for intelligent mobile and software engineering contexts. 

---
# Model-Based Diffusion Sampling for Predictive Control in Offline Decision Making 

**Authors**: Haldun Balim, Na Li, Yilun Du  

**Link**: [PDF](https://arxiv.org/pdf/2512.08280)  

**Abstract**: Offline decision-making requires synthesizing reliable behaviors from fixed datasets without further interaction, yet existing generative approaches often yield trajectories that are dynamically infeasible. We propose Model Predictive Diffuser (MPDiffuser), a compositional model-based diffusion framework consisting of: (i) a planner that generates diverse, task-aligned trajectories; (ii) a dynamics model that enforces consistency with the underlying system dynamics; and (iii) a ranker module that selects behaviors aligned with the task objectives. MPDiffuser employs an alternating diffusion sampling scheme, where planner and dynamics updates are interleaved to progressively refine trajectories for both task alignment and feasibility during the sampling process. We also provide a theoretical rationale for this procedure, showing how it balances fidelity to data priors with dynamics consistency. Empirically, the compositional design improves sample efficiency, as it leverages even low-quality data for dynamics learning and adapts seamlessly to novel dynamics. We evaluate MPDiffuser on both unconstrained (D4RL) and constrained (DSRL) offline decision-making benchmarks, demonstrating consistent gains over existing approaches. Furthermore, we present a preliminary study extending MPDiffuser to vision-based control tasks, showing its potential to scale to high-dimensional sensory inputs. Finally, we deploy our method on a real quadrupedal robot, showcasing its practicality for real-world control. 

---
# Distilling Future Temporal Knowledge with Masked Feature Reconstruction for 3D Object Detection 

**Authors**: Haowen Zheng, Hu Zhu, Lu Deng, Weihao Gu, Yang Yang, Yanyan Liang  

**Link**: [PDF](https://arxiv.org/pdf/2512.08247)  

**Abstract**: Camera-based temporal 3D object detection has shown impressive results in autonomous driving, with offline models improving accuracy by using future frames. Knowledge distillation (KD) can be an appealing framework for transferring rich information from offline models to online models. However, existing KD methods overlook future frames, as they mainly focus on spatial feature distillation under strict frame alignment or on temporal relational distillation, thereby making it challenging for online models to effectively learn future knowledge. To this end, we propose a sparse query-based approach, Future Temporal Knowledge Distillation (FTKD), which effectively transfers future frame knowledge from an offline teacher model to an online student model. Specifically, we present a future-aware feature reconstruction strategy to encourage the student model to capture future features without strict frame alignment. In addition, we further introduce future-guided logit distillation to leverage the teacher's stable foreground and background context. FTKD is applied to two high-performing 3D object detection baselines, achieving up to 1.3 mAP and 1.3 NDS gains on the nuScenes dataset, as well as the most accurate velocity estimation, without increasing inference cost. 

---
# Residual-SwinCA-Net: A Channel-Aware Integrated Residual CNN-Swin Transformer for Malignant Lesion Segmentation in BUSI 

**Authors**: Saeeda Naz, Saddam Hussain Khan  

**Link**: [PDF](https://arxiv.org/pdf/2512.08243)  

**Abstract**: A novel deep hybrid Residual-SwinCA-Net segmentation framework is proposed in the study for addressing such challenges by extracting locally correlated and robust features, incorporating residual CNN modules. Furthermore, for learning global dependencies, Swin Transformer blocks are customized using internal residual pathways, which reinforce gradient stability, refine local patterns, and facilitate global feature fusion. Formerly, for enhancing tissue continuity, ultrasound noise suppressions, and accentuating fine structural transitions Laplacian-of-Gaussian regional operator is applied, and for maintaining the morphological integrity of malignant lesion contours, a boundary-oriented operator has been incorporated. Subsequently, a contraction strategy was applied stage-wise by progressively reducing features-map progressively for capturing scale invariance and enhancing the robustness of structural variability. In addition, each decoder level prior augmentation integrates a new Multi-Scale Channel Attention and Squeezing (MSCAS) module. The MSCAS selectively emphasizes encoder salient maps, retains discriminative global context, and complementary local structures with minimal computational cost while suppressing redundant activations. Finally, the Pixel-Attention module encodes class-relevant spatial cues by adaptively weighing malignant lesion pixels while suppressing background interference. The Residual-SwinCA-Net and existing CNNs/ViTs techniques have been implemented on the publicly available BUSI dataset. The proposed Residual-SwinCA-Net framework outperformed and achieved 99.29% mean accuracy, 98.74% IoU, and 0.9041 Dice for breast lesion segmentation. The proposed Residual-SwinCA-Net framework improves the BUSI lesion diagnostic performance and strengthens timely clinical decision-making. 

---
# HybridToken-VLM: Hybrid Token Compression for Vision-Language Models 

**Authors**: Jusheng Zhang, Xiaoyang Guo, Kaitong Cai, Qinhan Lv, Yijia Fan, Wenhao Chai, Jian Wang, Keze Wang  

**Link**: [PDF](https://arxiv.org/pdf/2512.08240)  

**Abstract**: Vision-language models (VLMs) have transformed multimodal reasoning, but feeding hundreds of visual patch tokens into LLMs incurs quadratic computational costs, straining memory and context windows. Traditional approaches face a trade-off: continuous compression dilutes high-level semantics such as object identities, while discrete quantization loses fine-grained details such as textures. We introduce HTC-VLM, a hybrid framework that disentangles semantics and appearance through dual channels, i.e., a continuous pathway for fine-grained details via ViT patches and a discrete pathway for symbolic anchors using MGVQ quantization projected to four tokens. These are fused into a 580-token hybrid sequence and compressed into a single voco token via a disentanglement attention mask and bottleneck, ensuring efficient and grounded representations. HTC-VLM achieves an average performance retention of 87.2 percent across seven benchmarks (GQA, VQAv2, MMBench, MME, POPE, SEED-Bench, ScienceQA-Image), outperforming the leading continuous baseline at 81.0 percent with a 580-to-1 compression ratio. Attention analyses show that the compressed token prioritizes the discrete anchor, validating its semantic guidance. Our work demonstrates that a minimalist hybrid design can resolve the efficiency-fidelity dilemma and advance scalable VLMs. 

---
# SpeechQualityLLM: LLM-Based Multimodal Assessment of Speech Quality 

**Authors**: Mahathir Monjur, Shahriar Nirjon  

**Link**: [PDF](https://arxiv.org/pdf/2512.08238)  

**Abstract**: Objective speech quality assessment is central to telephony, VoIP, and streaming systems, where large volumes of degraded audio must be monitored and optimized at scale. Classical metrics such as PESQ and POLQA approximate human mean opinion scores (MOS) but require carefully controlled conditions and expensive listening tests, while learning-based models such as NISQA regress MOS and multiple perceptual dimensions from waveforms or spectrograms, achieving high correlation with subjective ratings yet remaining rigid: they do not support interactive, natural-language queries and do not natively provide textual rationales. In this work, we introduce SpeechQualityLLM, a multimodal speech quality question-answering (QA) system that couples an audio encoder with a language model and is trained on the NISQA corpus using template-based question-answer pairs covering overall MOS and four perceptual dimensions (noisiness, coloration, discontinuity, and loudness) in both single-ended (degraded only) and double-ended (degraded plus clean reference) setups. Instead of directly regressing scores, our system is supervised to generate textual answers from which numeric predictions are parsed and evaluated with standard regression and ranking metrics; on held-out NISQA clips, the double-ended model attains a MOS mean absolute error (MAE) of 0.41 with Pearson correlation of 0.86, with competitive performance on dimension-wise tasks. Beyond these quantitative gains, it offers a flexible natural-language interface in which the language model acts as an audio quality expert: practitioners can query arbitrary aspects of degradations, prompt the model to emulate different listener profiles to capture human variability and produce diverse but plausible judgments rather than a single deterministic score, and thereby reduce reliance on large-scale crowdsourced tests and their monetary cost. 

---
# MM-CoT:A Benchmark for Probing Visual Chain-of-Thought Reasoning in Multimodal Models 

**Authors**: Jusheng Zhang, Kaitong Cai, Xiaoyang Guo, Sidi Liu, Qinhan Lv, Ruiqi Chen, Jing Yang, Yijia Fan, Xiaofei Sun, Jian Wang, Ziliang Chen, Liang Lin, Keze Wang  

**Link**: [PDF](https://arxiv.org/pdf/2512.08228)  

**Abstract**: The ability to perform Chain-of-Thought (CoT) reasoning marks a major milestone for multimodal models (MMs), enabling them to solve complex visual reasoning problems. Yet a critical question remains: is such reasoning genuinely grounded in visual evidence and logically coherent? Existing benchmarks emphasize generation but neglect verification, i.e., the capacity to assess whether a reasoning chain is both visually consistent and logically valid. To fill this gap, we introduce MM-CoT, a diagnostic benchmark specifically designed to probe the visual grounding and logical coherence of CoT reasoning in MMs. Instead of generating free-form explanations, models must select the sole event chain that satisfies two orthogonal constraints: (i) visual consistency, ensuring all steps are anchored in observable evidence, and (ii) logical coherence, ensuring causal and commonsense validity. Adversarial distractors are engineered to violate one of these constraints, exposing distinct reasoning failures. We evaluate leading vision-language models on MM-CoT and find that even the most advanced systems struggle, revealing a sharp discrepancy between generative fluency and true reasoning fidelity. MM-CoT shows low correlation with existing benchmarks, confirming that it measures a unique combination of visual grounding and logical reasoning. This benchmark provides a foundation for developing future models that reason not just plausibly, but faithfully and coherently within the visual world. 

---
# PR-CapsNet: Pseudo-Riemannian Capsule Network with Adaptive Curvature Routing for Graph Learning 

**Authors**: Ye Qin, Jingchao Wang, Yang Shi, Haiying Huang, Junxu Li, Weijian Liu, Tinghui Chen, Jinghui Qin  

**Link**: [PDF](https://arxiv.org/pdf/2512.08218)  

**Abstract**: Capsule Networks (CapsNets) show exceptional graph representation capacity via dynamic routing and vectorized hierarchical representations, but they model the complex geometries of real\-world graphs poorly by fixed\-curvature space due to the inherent geodesical disconnectedness issues, leading to suboptimal performance. Recent works find that non\-Euclidean pseudo\-Riemannian manifolds provide specific inductive biases for embedding graph data, but how to leverage them to improve CapsNets is still underexplored. Here, we extend the Euclidean capsule routing into geodesically disconnected pseudo\-Riemannian manifolds and derive a Pseudo\-Riemannian Capsule Network (PR\-CapsNet), which models data in pseudo\-Riemannian manifolds of adaptive curvature, for graph representation learning. Specifically, PR\-CapsNet enhances the CapsNet with Adaptive Pseudo\-Riemannian Tangent Space Routing by utilizing pseudo\-Riemannian geometry. Unlike single\-curvature or subspace\-partitioning methods, PR\-CapsNet concurrently models hierarchical and cluster or cyclic graph structures via its versatile pseudo\-Riemannian metric. It first deploys Pseudo\-Riemannian Tangent Space Routing to decompose capsule states into spherical\-temporal and Euclidean\-spatial subspaces with diffeomorphic transformations. Then, an Adaptive Curvature Routing is developed to adaptively fuse features from different curvature spaces for complex graphs via a learnable curvature tensor with geometric attention from local manifold properties. Finally, a geometric properties\-preserved Pseudo\-Riemannian Capsule Classifier is developed to project capsule embeddings to tangent spaces and use curvature\-weighted softmax for classification. Extensive experiments on node and graph classification benchmarks show PR\-CapsNet outperforms SOTA models, validating PR\-CapsNet's strong representation power for complex graph structures. 

---
# ClinicalTrialsHub: Bridging Registries and Literature for Comprehensive Clinical Trial Access 

**Authors**: Jiwoo Park, Ruoqi Liu, Avani Jagdale, Andrew Srisuwananukorn, Jing Zhao, Lang Li, Ping Zhang, Sachin Kumar  

**Link**: [PDF](https://arxiv.org/pdf/2512.08193)  

**Abstract**: We present ClinicalTrialsHub, an interactive search-focused platform that consolidates all data from this http URL and augments it by automatically extracting and structuring trial-relevant information from PubMed research articles. Our system effectively increases access to structured clinical trial data by 83.8% compared to relying on this http URL alone, with potential to make access easier for patients, clinicians, researchers, and policymakers, advancing evidence-based medicine. ClinicalTrialsHub uses large language models such as GPT-5.1 and Gemini-3-Pro to enhance accessibility. The platform automatically parses full-text research articles to extract structured trial information, translates user queries into structured database searches, and provides an attributed question-answering system that generates evidence-grounded answers linked to specific source sentences. We demonstrate its utility through a user study involving clinicians, clinical researchers, and PhD students of pharmaceutical sciences and nursing, and a systematic automatic evaluation of its information extraction and question answering capabilities. 

---
# Embodied Tree of Thoughts: Deliberate Manipulation Planning with Embodied World Model 

**Authors**: Wenjiang Xu, Cindy Wang, Rui Fang, Mingkang Zhang, Lusong Li, Jing Xu, Jiayuan Gu, Zecui Zeng, Rui Chen  

**Link**: [PDF](https://arxiv.org/pdf/2512.08188)  

**Abstract**: World models have emerged as a pivotal component in robot manipulation planning, enabling agents to predict future environmental states and reason about the consequences of actions before execution. While video-generation models are increasingly adopted, they often lack rigorous physical grounding, leading to hallucinations and a failure to maintain consistency in long-horizon physical constraints. To address these limitations, we propose Embodied Tree of Thoughts (EToT), a novel Real2Sim2Real planning framework that leverages a physics-based interactive digital twin as an embodied world model. EToT formulates manipulation planning as a tree search expanded through two synergistic mechanisms: (1) Priori Branching, which generates diverse candidate execution paths based on semantic and spatial analysis; and (2) Reflective Branching, which utilizes VLMs to diagnose execution failures within the simulator and iteratively refine the planning tree with corrective actions. By grounding high-level reasoning in a physics simulator, our framework ensures that generated plans adhere to rigid-body dynamics and collision constraints. We validate EToT on a suite of short- and long-horizon manipulation tasks, where it consistently outperforms baselines by effectively predicting physical dynamics and adapting to potential failures. Website at this https URL . 

---
# A Practical Framework for Evaluating Medical AI Security: Reproducible Assessment of Jailbreaking and Privacy Vulnerabilities Across Clinical Specialties 

**Authors**: Jinghao Wang, Ping Zhang, Carter Yagemann  

**Link**: [PDF](https://arxiv.org/pdf/2512.08185)  

**Abstract**: Medical Large Language Models (LLMs) are increasingly deployed for clinical decision support across diverse specialties, yet systematic evaluation of their robustness to adversarial misuse and privacy leakage remains inaccessible to most researchers. Existing security benchmarks require GPU clusters, commercial API access, or protected health data -- barriers that limit community participation in this critical research area. We propose a practical, fully reproducible framework for evaluating medical AI security under realistic resource constraints. Our framework design covers multiple medical specialties stratified by clinical risk -- from high-risk domains such as emergency medicine and psychiatry to general practice -- addressing jailbreaking attacks (role-playing, authority impersonation, multi-turn manipulation) and privacy extraction attacks. All evaluation utilizes synthetic patient records requiring no IRB approval. The framework is designed to run entirely on consumer CPU hardware using freely available models, eliminating cost barriers. We present the framework specification including threat models, data generation methodology, evaluation protocols, and scoring rubrics. This proposal establishes a foundation for comparative security assessment of medical-specialist models and defense mechanisms, advancing the broader goal of ensuring safe and trustworthy medical AI systems. 

---
# Information-Dense Reasoning for Efficient and Auditable Security Alert Triage 

**Authors**: Guangze Zhao, Yongzheng Zhang, Changbo Tian, Dan Xie, Hongri Liu, Bailing Wang  

**Link**: [PDF](https://arxiv.org/pdf/2512.08169)  

**Abstract**: Security Operations Centers face massive, heterogeneous alert streams under minute-level service windows, creating the Alert Triage Latency Paradox: verbose reasoning chains ensure accuracy and compliance but incur prohibitive latency and token costs, while minimal chains sacrifice transparency and auditability. Existing solutions fail: signature systems are brittle, anomaly methods lack actionability, and fully cloud-hosted LLMs raise latency, cost, and privacy concerns. We propose AIDR, a hybrid cloud-edge framework that addresses this trade-off through constrained information-density optimization. The core innovation is gradient-based compression of reasoning chains to retain only decision-critical steps--minimal evidence sufficient to justify predictions while respecting token and latency budgets. We demonstrate that this approach preserves decision-relevant information while minimizing complexity. We construct compact datasets by distilling alerts into 3-5 high-information bullets (68% token reduction), train domain-specialized experts via LoRA, and deploy a cloud-edge architecture: a cloud LLM routes alerts to on-premises experts generating SOAR-ready JSON. Experiments demonstrate AIDR achieves higher accuracy and 40.6% latency reduction versus Chain-of-Thought, with robustness to data corruption and out-of-distribution generalization, enabling auditable and efficient SOC triage with full data residency compliance. 

---
# LayerPipe2: Multistage Pipelining and Weight Recompute via Improved Exponential Moving Average for Training Neural Networks 

**Authors**: Nanda K. Unnikrishnan, Keshab K. Parhi  

**Link**: [PDF](https://arxiv.org/pdf/2512.08160)  

**Abstract**: In our prior work, LayerPipe, we had introduced an approach to accelerate training of convolutional, fully connected, and spiking neural networks by overlapping forward and backward computation. However, despite empirical success, a principled understanding of how much gradient delay needs to be introduced at each layer to achieve desired level of pipelining was not addressed. This paper, LayerPipe2, fills that gap by formally deriving LayerPipe using variable delayed gradient adaptation and retiming. We identify where delays may be legally inserted and show that the required amount of delay follows directly from the network structure where inner layers require fewer delays and outer layers require longer delays. When pipelining is applied at every layer, the amount of delay depends only on the number of remaining downstream stages. When layers are pipelined in groups, all layers in the group share the same assignment of delays. These insights not only explain previously observed scheduling patterns but also expose an often overlooked challenge that pipelining implicitly requires storage of historical weights. We overcome this storage bottleneck by developing a pipeline--aware moving average that reconstructs the required past states rather than storing them explicitly. This reduces memory cost without sacrificing the accuracy guarantees that makes pipelined learning viable. The result is a principled framework that illustrates how to construct LayerPipe architectures, predicts their delay requirements, and mitigates their storage burden, thereby enabling scalable pipelined training with controlled communication computation tradeoffs. 

---
# TreeGRPO: Tree-Advantage GRPO for Online RL Post-Training of Diffusion Models 

**Authors**: Zheng Ding, Weirui Ye  

**Link**: [PDF](https://arxiv.org/pdf/2512.08153)  

**Abstract**: Reinforcement learning (RL) post-training is crucial for aligning generative models with human preferences, but its prohibitive computational cost remains a major barrier to widespread adoption. We introduce \textbf{TreeGRPO}, a novel RL framework that dramatically improves training efficiency by recasting the denoising process as a search tree. From shared initial noise samples, TreeGRPO strategically branches to generate multiple candidate trajectories while efficiently reusing their common prefixes. This tree-structured approach delivers three key advantages: (1) \emph{High sample efficiency}, achieving better performance under same training samples (2) \emph{Fine-grained credit assignment} via reward backpropagation that computes step-specific advantages, overcoming the uniform credit assignment limitation of trajectory-based methods, and (3) \emph{Amortized computation} where multi-child branching enables multiple policy updates per forward pass. Extensive experiments on both diffusion and flow-based models demonstrate that TreeGRPO achieves \textbf{2.4$\times$ faster training} while establishing a superior Pareto frontier in the efficiency-reward trade-off space. Our method consistently outperforms GRPO baselines across multiple benchmarks and reward models, providing a scalable and effective pathway for RL-based visual generative model alignment. The project website is available at this http URL. 

---
# Chat with UAV -- Human-UAV Interaction Based on Large Language Models 

**Authors**: Haoran Wang, Zhuohang Chen, Guang Li, Bo Ma, Chuanghuang Li  

**Link**: [PDF](https://arxiv.org/pdf/2512.08145)  

**Abstract**: The future of UAV interaction systems is evolving from engineer-driven to user-driven, aiming to replace traditional predefined Human-UAV Interaction designs. This shift focuses on enabling more personalized task planning and design, thereby achieving a higher quality of interaction experience and greater flexibility, which can be used in many fileds, such as agriculture, aerial photography, logistics, and environmental monitoring. However, due to the lack of a common language between users and the UAVs, such interactions are often difficult to be achieved. The developments of Large Language Models possess the ability to understand nature languages and Robots' (UAVs') behaviors, marking the possibility of personalized Human-UAV Interaction. Recently, some HUI frameworks based on LLMs have been proposed, but they commonly suffer from difficulties in mixed task planning and execution, leading to low adaptability in complex scenarios. In this paper, we propose a novel dual-agent HUI framework. This framework constructs two independent LLM agents (a task planning agent, and an execution agent) and applies different Prompt Engineering to separately handle the understanding, planning, and execution of tasks. To verify the effectiveness and performance of the framework, we have built a task database covering four typical application scenarios of UAVs and quantified the performance of the HUI framework using three independent metrics. Meanwhile different LLM models are selected to control the UAVs with compared performance. Our user study experimental results demonstrate that the framework improves the smoothness of HUI and the flexibility of task execution in the tasks scenario we set up, effectively meeting users' personalized needs. 

---
# Biothreat Benchmark Generation Framework for Evaluating Frontier AI Models I: The Task-Query Architecture 

**Authors**: Gary Ackerman, Brandon Behlendorf, Zachary Kallenborn, Sheriff Almakki, Doug Clifford, Jenna LaTourette, Hayley Peterson, Noah Sheinbaum, Olivia Shoemaker, Anna Wetzel  

**Link**: [PDF](https://arxiv.org/pdf/2512.08130)  

**Abstract**: Both model developers and policymakers seek to quantify and mitigate the risk of rapidly-evolving frontier artificial intelligence (AI) models, especially large language models (LLMs), to facilitate bioterrorism or access to biological weapons. An important element of such efforts is the development of model benchmarks that can assess the biosecurity risk posed by a particular model. This paper describes the first component of a novel Biothreat Benchmark Generation (BBG) Framework. The BBG approach is designed to help model developers and evaluators reliably measure and assess the biosecurity risk uplift and general harm potential of existing and future AI models, while accounting for key aspects of the threat itself that are often overlooked in other benchmarking efforts, including different actor capability levels, and operational (in addition to purely technical) risk factors. As a pilot, the BBG is first being developed to address bacterial biological threats only. The BBG is built upon a hierarchical structure of biothreat categories, elements and tasks, which then serves as the basis for the development of task-aligned queries. This paper outlines the development of this biothreat task-query architecture, which we have named the Bacterial Biothreat Schema, while future papers will describe follow-on efforts to turn queries into model prompts, as well as how the resulting benchmarks can be implemented for model evaluation. Overall, the BBG Framework, including the Bacterial Biothreat Schema, seeks to offer a robust, re-usable structure for evaluating bacterial biological risks arising from LLMs across multiple levels of aggregation, which captures the full scope of technical and operational requirements for biological adversaries, and which accounts for a wide spectrum of biological adversary capabilities. 

---
# Long-only cryptocurrency portfolio management by ranking the assets: a neural network approach 

**Authors**: Zijiang Yang  

**Link**: [PDF](https://arxiv.org/pdf/2512.08124)  

**Abstract**: This paper will propose a novel machine learning based portfolio management method in the context of the cryptocurrency market. Previous researchers mainly focus on the prediction of the movement for specific cryptocurrency such as the bitcoin(BTC) and then trade according to the prediction. In contrast to the previous work that treats the cryptocurrencies independently, this paper manages a group of cryptocurrencies by analyzing the relative relationship. Specifically, in each time step, we utilize the neural network to predict the rank of the future return of the managed cryptocurrencies and place weights accordingly. By incorporating such cross-sectional information, the proposed methods is shown to profitable based on the backtesting experiments on the real daily cryptocurrency market data from May, 2020 to Nov, 2023. During this 3.5 years, the market experiences the full cycle of bullish, bearish and stagnant market conditions. Despite under such complex market conditions, the proposed method outperforms the existing methods and achieves a Sharpe ratio of 1.01 and annualized return of 64.26%. Additionally, the proposed method is shown to be robust to the increase of transaction fee. 

---
# Balanced Accuracy: The Right Metric for Evaluating LLM Judges - Explained through Youden's J statistic 

**Authors**: Stephane Collot, Colin Fraser, Justin Zhao, William F. Shen, Timon Willi, Ilias Leontiadis  

**Link**: [PDF](https://arxiv.org/pdf/2512.08121)  

**Abstract**: Rigorous evaluation of large language models (LLMs) relies on comparing models by the prevalence of desirable or undesirable behaviors, such as task pass rates or policy violations. These prevalence estimates are produced by a classifier, either an LLM-as-a-judge or human annotators, making the choice of classifier central to trustworthy evaluation. Common metrics used for this choice, such as Accuracy, Precision, and F1, are sensitive to class imbalance and to arbitrary choices of positive class, and can favor judges that distort prevalence estimates. We show that Youden's $J$ statistic is theoretically aligned with choosing the best judge to compare models, and that Balanced Accuracy is an equivalent linear transformation of $J$. Through both analytical arguments and empirical examples and simulations, we demonstrate how selecting judges using Balanced Accuracy leads to better, more robust classifier selection. 

---
# Scalable Offline Model-Based RL with Action Chunks 

**Authors**: Kwanyoung Park, Seohong Park, Youngwoon Lee, Sergey Levine  

**Link**: [PDF](https://arxiv.org/pdf/2512.08108)  

**Abstract**: In this paper, we study whether model-based reinforcement learning (RL), in particular model-based value expansion, can provide a scalable recipe for tackling complex, long-horizon tasks in offline RL. Model-based value expansion fits an on-policy value function using length-n imaginary rollouts generated by the current policy and a learned dynamics model. While larger n reduces bias in value bootstrapping, it amplifies accumulated model errors over long horizons, degrading future predictions. We address this trade-off with an \emph{action-chunk} model that predicts a future state from a sequence of actions (an "action chunk") instead of a single action, which reduces compounding errors. In addition, instead of directly training a policy to maximize rewards, we employ rejection sampling from an expressive behavioral action-chunk policy, which prevents model exploitation from out-of-distribution actions. We call this recipe \textbf{Model-Based RL with Action Chunks (MAC)}. Through experiments on highly challenging tasks with large-scale datasets of up to 100M transitions, we show that MAC achieves the best performance among offline model-based RL algorithms, especially on challenging long-horizon tasks. 

---
# Training LLMs for Honesty via Confessions 

**Authors**: Manas Joglekar, Jeremy Chen, Gabriel Wu, Jason Yosinski, Jasmine Wang, Boaz Barak, Amelia Glaese  

**Link**: [PDF](https://arxiv.org/pdf/2512.08093)  

**Abstract**: Large language models (LLMs) can be dishonest when reporting on their actions and beliefs -- for example, they may overstate their confidence in factual claims or cover up evidence of covert actions. Such dishonesty may arise due to the effects of reinforcement learning (RL), where challenges with reward shaping can result in a training process that inadvertently incentivizes the model to lie or misrepresent its actions.
In this work we propose a method for eliciting an honest expression of an LLM's shortcomings via a self-reported *confession*. A confession is an output, provided upon request after a model's original answer, that is meant to serve as a full account of the model's compliance with the letter and spirit of its policies and instructions. The reward assigned to a confession during training is solely based on its honesty, and does not impact positively or negatively the main answer's reward. As long as the "path of least resistance" for maximizing confession reward is to surface misbehavior rather than covering it up, this incentivizes models to be honest in their confessions. Our findings provide some justification this empirical assumption, especially in the case of egregious model misbehavior.
To demonstrate the viability of our approach, we train GPT-5-Thinking to produce confessions, and we evaluate its honesty in out-of-distribution scenarios measuring hallucination, instruction following, scheming, and reward hacking. We find that when the model lies or omits shortcomings in its "main" answer, it often confesses to these behaviors honestly, and this confession honesty modestly improves with training. Confessions can enable a number of inference-time interventions including monitoring, rejection sampling, and surfacing issues to the user. 

---
# Short-Context Dominance: How Much Local Context Natural Language Actually Needs? 

**Authors**: Vala Vakilian, Zimeng Wang, Ankit Singh Rawat, Christos Thrampoulidis  

**Link**: [PDF](https://arxiv.org/pdf/2512.08082)  

**Abstract**: We investigate the short-context dominance hypothesis: that for most sequences, a small local prefix suffices to predict their next tokens. Using large language models as statistical oracles, we measure the minimum context length (MCL) needed to reproduce accurate full-context predictions across datasets with sequences of varying lengths. For sequences with 1-7k tokens from long-context documents, we consistently find that 75-80% require only the last 96 tokens at most. Given the dominance of short-context tokens, we then ask whether it is possible to detect challenging long-context sequences for which a short local prefix does not suffice for prediction. We introduce a practical proxy to MCL, called Distributionally Aware MCL (DaMCL), that does not require knowledge of the actual next-token and is compatible with sampling strategies beyond greedy decoding. Our experiments validate that simple thresholding of the metric defining DaMCL achieves high performance in detecting long vs. short context sequences. Finally, to counter the bias that short-context dominance induces in LLM output distributions, we develop an intuitive decoding algorithm that leverages our detector to identify and boost tokens that are long-range-relevant. Across Q&A tasks and model architectures, we confirm that mitigating the bias improves performance. 

---
# Joint Activity Design Heuristics for Enhancing Human-Machine Collaboration 

**Authors**: Mohammadreza Jalaeian, Dane A. Morey, Michael F. Rayo  

**Link**: [PDF](https://arxiv.org/pdf/2512.08036)  

**Abstract**: Joint activity describes when more than one agent (human or machine) contributes to the completion of a task or activity. Designing for joint activity focuses on explicitly supporting the interdependencies between agents necessary for effective coordination among agents engaged in the joint activity. This builds and expands upon designing for usability to further address how technologies can be designed to act as effective team players. Effective joint activity requires supporting, at minimum, five primary macrocognitive functions within teams: Event Detection, Sensemaking, Adaptability, Perspective-Shifting, and Coordination. Supporting these functions is equally as important as making technologies usable. We synthesized fourteen heuristics from relevant literature including display design, human factors, cognitive systems engineering, cognitive psychology, and computer science to aid the design, development, and evaluation of technologies that support joint human-machine activity. 

---
# FRIEDA: Benchmarking Multi-Step Cartographic Reasoning in Vision-Language Models 

**Authors**: Jiyoon Pyo, Yuankun Jiao, Dongwon Jung, Zekun Li, Leeje Jang, Sofia Kirsanova, Jina Kim, Yijun Lin, Qin Liu, Junyi Xie, Hadi Askari, Nan Xu, Muhao Chen, Yao-Yi Chiang  

**Link**: [PDF](https://arxiv.org/pdf/2512.08016)  

**Abstract**: Cartographic reasoning is the skill of interpreting geographic relationships by aligning legends, map scales, compass directions, map texts, and geometries across one or more map images. Although essential as a concrete cognitive capability and for critical tasks such as disaster response and urban planning, it remains largely unevaluated. Building on progress in chart and infographic understanding, recent large vision language model studies on map visual question-answering often treat maps as a special case of charts. In contrast, map VQA demands comprehension of layered symbology (e.g., symbols, geometries, and text labels) as well as spatial relations tied to orientation and distance that often span multiple maps and are not captured by chart-style evaluations. To address this gap, we introduce FRIEDA, a benchmark for testing complex open-ended cartographic reasoning in LVLMs. FRIEDA sources real map images from documents and reports in various domains and geographical areas. Following classifications in Geographic Information System (GIS) literature, FRIEDA targets all three categories of spatial relations: topological (border, equal, intersect, within), metric (distance), and directional (orientation). All questions require multi-step inference, and many require cross-map grounding and reasoning. We evaluate eleven state-of-the-art LVLMs under two settings: (1) the direct setting, where we provide the maps relevant to the question, and (2) the contextual setting, where the model may have to identify the maps relevant to the question before reasoning. Even the strongest models, Gemini-2.5-Pro and GPT-5-Think, achieve only 38.20% and 37.20% accuracy, respectively, far below human performance of 84.87%. These results reveal a persistent gap in multi-step cartographic reasoning, positioning FRIEDA as a rigorous benchmark to drive progress on spatial intelligence in LVLMs. 

---
# A Gray Literature Study on Fairness Requirements in AI-enabled Software Engineering 

**Authors**: Thanh Nguyen, Chaima Boufaied, Ronnie de Souza Santos  

**Link**: [PDF](https://arxiv.org/pdf/2512.07990)  

**Abstract**: Today, with the growing obsession with applying Artificial Intelligence (AI), particularly Machine Learning (ML), to software across various contexts, much of the focus has been on the effectiveness of AI models, often measured through common metrics such as F1- score, while fairness receives relatively little attention. This paper presents a review of existing gray literature, examining fairness requirements in AI context, with a focus on how they are defined across various application domains, managed throughout the Software Development Life Cycle (SDLC), and the causes, as well as the corresponding consequences of their violation by AI models. Our gray literature investigation shows various definitions of fairness requirements in AI systems, commonly emphasizing non-discrimination and equal treatment across different demographic and social attributes. Fairness requirement management practices vary across the SDLC, particularly in model training and bias mitigation, fairness monitoring and evaluation, and data handling practices. Fairness requirement violations are frequently linked, but not limited, to data representation bias, algorithmic and model design bias, human judgment, and evaluation and transparency gaps. The corresponding consequences include harm in a broad sense, encompassing specific professional and societal impacts as key examples, stereotype reinforcement, data and privacy risks, and loss of trust and legitimacy in AI-supported decisions. These findings emphasize the need for consistent frameworks and practices to integrate fairness into AI software, paying as much attention to fairness as to effectiveness. 

---
# Restrictive Hierarchical Semantic Segmentation for Stratified Tooth Layer Detection 

**Authors**: Ryan Banks, Camila Lindoni Azevedo, Hongying Tang, Yunpeng Li  

**Link**: [PDF](https://arxiv.org/pdf/2512.07984)  

**Abstract**: Accurate understanding of anatomical structures is essential for reliably staging certain dental diseases. A way of introducing this within semantic segmentation models is by utilising hierarchy-aware methodologies. However, existing hierarchy-aware segmentation methods largely encode anatomical structure through the loss functions, providing weak and indirect supervision. We introduce a general framework that embeds an explicit anatomical hierarchy into semantic segmentation by coupling a recurrent, level-wise prediction scheme with restrictive output heads and top-down feature conditioning. At each depth of the class tree, the backbone is re-run on the original image concatenated with logits from the previous level. Child class features are conditioned using Feature-wise Linear Modulation of their parent class probabilities, to modulate child feature spaces for fine grained detection. A probabilistic composition rule enforces consistency between parent and descendant classes. Hierarchical loss combines per-level class weighted Dice and cross entropy loss and a consistency term loss, ensuring parent predictions are the sum of their children. We validate our approach on our proposed dataset, TL-pano, containing 194 panoramic radiographs with dense instance and semantic segmentation annotations, of tooth layers and alveolar bone. Utilising UNet and HRNet as donor models across a 5-fold cross validation scheme, the hierarchical variants consistently increase IoU, Dice, and recall, particularly for fine-grained anatomies, and produce more anatomically coherent masks. However, hierarchical variants also demonstrated increased recall over precision, implying increased false positives. The results demonstrate that explicit hierarchical structuring improves both performance and clinical plausibility, especially in low data dental imaging regimes. 

---
# An Empirical Framework for Evaluating Semantic Preservation Using Hugging Face 

**Authors**: Nan Jia, Anita Raja, Raffi Khatchadourian  

**Link**: [PDF](https://arxiv.org/pdf/2512.07983)  

**Abstract**: As machine learning (ML) becomes an integral part of high-autonomy systems, it is critical to ensure the trustworthiness of learning-enabled software systems (LESS). Yet, the nondeterministic and run-time-defined semantics of ML complicate traditional software refactoring. We define semantic preservation in LESS as the property that optimizations of intelligent components do not alter the system's overall functional behavior. This paper introduces an empirical framework to evaluate semantic preservation in LESS by mining model evolution data from HuggingFace. We extract commit histories, $\textit{Model Cards}$, and performance metrics from a large number of models. To establish baselines, we conducted case studies in three domains, tracing performance changes across versions. Our analysis demonstrates how $\textit{semantic drift}$ can be detected via evaluation metrics across commits and reveals common refactoring patterns based on commit message analysis. Although API constraints limited the possibility of estimating a full-scale threshold, our pipeline offers a foundation for defining community-accepted boundaries for semantic preservation. Our contributions include: (1) a large-scale dataset of ML model evolution, curated from 1.7 million Hugging Face entries via a reproducible pipeline using the native HF hub API, (2) a practical pipeline for the evaluation of semantic preservation for a subset of 536 models and 4000+ metrics and (3) empirical case studies illustrating semantic drift in practice. Together, these contributions advance the foundations for more maintainable and trustworthy ML systems. 

---
# Near-real time fires detection using satellite imagery in Sudan conflict 

**Authors**: Kuldip Singh Atwal, Dieter Pfoser, Daniel Rothbart  

**Link**: [PDF](https://arxiv.org/pdf/2512.07925)  

**Abstract**: The challenges of ongoing war in Sudan highlight the need for rapid moni- toring and analysis of such conflicts. Advances in deep learning and readily available satellite remote sensing imagery allow for near real-time monitor- ing. This paper uses 4-band imagery from Planet Labs with a deep learning model to show that fire damage in armed conflicts can be monitored with minimal delay. We demonstrate the effectiveness of our approach using five case studies in Sudan. We show that, compared to a baseline, the automated method captures the active fires and charred areas more accurately. Our re- sults indicate that using 8-band imagery or time series of such imagery only result in marginal gains. 

---
# DeepCode: Open Agentic Coding 

**Authors**: Zongwei Li, Zhonghang Li, Zirui Guo, Xubin Ren, Chao Huang  

**Link**: [PDF](https://arxiv.org/pdf/2512.07921)  

**Abstract**: Recent advances in large language models (LLMs) have given rise to powerful coding agents, making it possible for code assistants to evolve into code engineers. However, existing methods still face significant challenges in achieving high-fidelity document-to-codebase synthesis--such as scientific papers to code--primarily due to a fundamental conflict between information overload and the context bottlenecks of LLMs. In this work, we introduce DeepCode, a fully autonomous framework that fundamentally addresses this challenge through principled information-flow management. By treating repository synthesis as a channel optimization problem, DeepCode seamlessly orchestrates four information operations to maximize task-relevant signals under finite context budgets: source compression via blueprint distillation, structured indexing using stateful code memory, conditional knowledge injection via retrieval-augmented generation, and closed-loop error correction. Extensive evaluations on the PaperBench benchmark demonstrate that DeepCode achieves state-of-the-art performance, decisively outperforming leading commercial agents such as Cursor and Claude Code, and crucially, surpassing PhD-level human experts from top institutes on key reproduction metrics. By systematically transforming paper specifications into production-grade implementations comparable to human expert quality, this work establishes new foundations for autonomous scientific reproduction that can accelerate research evaluation and discovery. 

---
# CFD-copilot: leveraging domain-adapted large language model and model context protocol to enhance simulation automation 

**Authors**: Zhehao Dong, Shanghai Du, Zhen Lu, Yue Yang  

**Link**: [PDF](https://arxiv.org/pdf/2512.07917)  

**Abstract**: Configuring computational fluid dynamics (CFD) simulations requires significant expertise in physics modeling and numerical methods, posing a barrier to non-specialists. Although automating scientific tasks with large language models (LLMs) has attracted attention, applying them to the complete, end-to-end CFD workflow remains a challenge due to its stringent domain-specific requirements. We introduce CFD-copilot, a domain-specialized LLM framework designed to facilitate natural language-driven CFD simulation from setup to post-processing. The framework employs a fine-tuned LLM to directly translate user descriptions into executable CFD setups. A multi-agent system integrates the LLM with simulation execution, automatic error correction, and result analysis. For post-processing, the framework utilizes the model context protocol (MCP), an open standard that decouples LLM reasoning from external tool execution. This modular design allows the LLM to interact with numerous specialized post-processing functions through a unified and scalable interface, improving the automation of data extraction and analysis. The framework was evaluated on benchmarks including the NACA~0012 airfoil and the three-element 30P-30N airfoil. The results indicate that domain-specific adaptation and the incorporation of the MCP jointly enhance the reliability and efficiency of LLM-driven engineering workflows. 

---
# Harmonizing Community Science Datasets to Model Highly Pathogenic Avian Influenza (HPAI) in Birds in the Subantarctic 

**Authors**: Richard Littauer, Kris Bubendorfer  

**Link**: [PDF](https://arxiv.org/pdf/2512.07907)  

**Abstract**: Community science observational datasets are useful in epidemiology and ecology for modeling species distributions, but the heterogeneous nature of the data presents significant challenges for standardization, data quality assurance and control, and workflow management. In this paper, we present a data workflow for cleaning and harmonizing multiple community science datasets, which we implement in a case study using eBird, iNaturalist, GBIF, and other datasets to model the impact of highly pathogenic avian influenza in populations of birds in the subantarctic. We predict population sizes for several species where the demographics are not known, and we present novel estimates for potential mortality rates from HPAI for those species, based on a novel aggregated dataset of mortality rates in the subantarctic. 

---
# The Theory of Strategic Evolution: Games with Endogenous Players and Strategic Replicators 

**Authors**: Kevin Vallier  

**Link**: [PDF](https://arxiv.org/pdf/2512.07901)  

**Abstract**: This paper develops the Theory of Strategic Evolution, a general model for systems in which the population of players, strategies, and institutional rules evolve together. The theory extends replicator dynamics to settings with endogenous players, multi level selection, innovation, constitutional change, and meta governance. The central mathematical object is a Poiesis stack: a hierarchy of strategic layers linked by cross level gain matrices. Under small gain conditions, the system admits a global Lyapunov function and satisfies selection, tracking, and stochastic stability results at every finite depth. We prove that the class is closed under block extension, innovation events, heterogeneous utilities, continuous strategy spaces, and constitutional evolution. The closure theorem shows that no new dynamics arise at higher levels and that unrestricted self modification cannot preserve Lyapunov structure. The theory unifies results from evolutionary game theory, institutional design, innovation dynamics, and constitutional political economy, providing a general mathematical model of long run strategic adaptation. 

---
# MARINE: Theoretical Optimization and Design for Multi-Agent Recursive IN-context Enhancement 

**Authors**: Hongwei Zhang, Ji Lu, Yongsheng Du, Yanqin Gao, Lingjun Huang, Baoli Wang, Fang Tan, Peng Zou  

**Link**: [PDF](https://arxiv.org/pdf/2512.07898)  

**Abstract**: Large Language Model (LLM)-based agents demonstrate advanced reasoning capabilities, yet practical constraints frequently limit outputs to single responses, leaving significant performance potential unrealized. This paper introduces MARINE (Multi-Agent Recursive IN-context Enhancement), a theoretically grounded framework that reconceptualizes test-time reasoning as iterative refinement of a persistent reference trajectory, fundamentally departing from conventional one-shot or multi-sample paradigms. The MARINE refinement operator systematically converts a base model's pass@N capabilities into near-optimal pass@1 performance. Rigorous theoretical analysis establishes that minimal feasible batches maximize expected performance gains under fixed invocation budgets, while logarithmically growing batch schedules ensure continuous improvement without computational constraints. Comprehensive evaluation on the BrowserComp-ZH benchmark demonstrates state-of-the-art results, with a 685B-parameter implementation achieving 46.0% pass@1 accuracy. Meanwhile, MARINE establishes a new paradigm for parameter-efficient reasoning: an 80B-parameter model augmented with MARINE matches the performance of standalone 1000B-parameter agents, reducing parameter requirements by over an order of magnitude. Notably, within a fixed computational budget, the proposed MARINE delivers higher-quality samples to alignment and optimization processes than traditional sampling-and-ranking strategies. Consequently, it has great potential to boost post-training efficiency. 

---
# Functional Random Forest with Adaptive Cost-Sensitive Splitting for Imbalanced Functional Data Classification 

**Authors**: Fahad Mostafa, Hafiz Khan  

**Link**: [PDF](https://arxiv.org/pdf/2512.07888)  

**Abstract**: Classification of functional data where observations are curves or trajectories poses unique challenges, particularly under severe class imbalance. Traditional Random Forest algorithms, while robust for tabular data, often fail to capture the intrinsic structure of functional observations and struggle with minority class detection. This paper introduces Functional Random Forest with Adaptive Cost-Sensitive Splitting (FRF-ACS), a novel ensemble framework designed for imbalanced functional data classification. The proposed method leverages basis expansions and Functional Principal Component Analysis (FPCA) to represent curves efficiently, enabling trees to operate on low dimensional functional features. To address imbalance, we incorporate a dynamic cost sensitive splitting criterion that adjusts class weights locally at each node, combined with a hybrid sampling strategy integrating functional SMOTE and weighted bootstrapping. Additionally, curve specific similarity metrics replace traditional Euclidean measures to preserve functional characteristics during leaf assignment. Extensive experiments on synthetic and real world datasets including biomedical signals and sensor trajectories demonstrate that FRF-ACS significantly improves minority class recall and overall predictive performance compared to existing functional classifiers and imbalance handling techniques. This work provides a scalable, interpretable solution for high dimensional functional data analysis in domains where minority class detection is critical. 

---
# ByteStorm: a multi-step data-driven approach for Tropical Cyclones detection and tracking 

**Authors**: Davide Donno, Donatello Elia, Gabriele Accarino, Marco De Carlo, Enrico Scoccimarro, Silvio Gualdi  

**Link**: [PDF](https://arxiv.org/pdf/2512.07885)  

**Abstract**: Accurate tropical cyclones (TCs) tracking represents a critical challenge in the context of weather and climate science. Traditional tracking schemes mainly rely on subjective thresholds, which may introduce biases in their skills on the geographical region of application. We present ByteStorm, an efficient data-driven framework for reconstructing TC tracks without threshold tuning. It leverages deep learning networks to detect TC centers (via classification and localization), using only relative vorticity (850 mb) and mean sea-level pressure. Then, detected centers are linked into TC tracks through the BYTE algorithm. ByteStorm is evaluated against state-of-the-art deterministic trackers in the East- and West-North Pacific basins (ENP and WNP). The proposed framework achieves superior performance in terms of Probability of Detection ($85.05\%$ ENP, $79.48\%$ WNP), False Alarm Rate ($23.26\%$ ENP, $16.14\%$ WNP), and high Inter-Annual Variability correlations ($0.75$ ENP and $0.69$ WNP). These results highlight the potential of integrating deep learning and computer vision for fast and accurate TC tracking, offering a robust alternative to traditional approaches. 

---
# GSPN-2: Efficient Parallel Sequence Modeling 

**Authors**: Hongjun Wang, Yitong Jiang, Collin McCarthy, David Wehr, Hanrong Ye, Xinhao Li, Ka Chun Cheung, Wonmin Byeon, Jinwei Gu, Ke Chen, Kai Han, Hongxu Yin, Pavlo Molchanov, Jan Kautz, Sifei Liu  

**Link**: [PDF](https://arxiv.org/pdf/2512.07884)  

**Abstract**: Efficient vision transformer remains a bottleneck for high-resolution images and long-video related real-world applications. Generalized Spatial Propagation Network (GSPN) addresses this by replacing quadratic self-attention with a line-scan propagation scheme, bringing the cost close to linear in the number of rows or columns, while retaining accuracy. Despite this advancement, the existing GSPN implementation still suffers from (i) heavy overhead due to repeatedly launching GPU kernels, (ii) excessive data transfers from global GPU memory, and (iii) redundant computations caused by maintaining separate propagation weights for each channel. We introduce GSPN-2, a joint algorithm-system redesign. In particular, we eliminate thousands of micro-launches from the previous implementation into one single 2D kernel, explicitly pin one warp to each channel slice, and stage the previous column's activations in shared memory. On the model side, we introduce a compact channel propagation strategy that replaces per-channel matrices, trimming parameters, and align naturally with the affinity map used in transformer attention. Experiments demonstrate GSPN-2's effectiveness across image classification and text-to-image synthesis tasks, matching transformer-level accuracy with significantly lower computational cost. GSPN-2 establishes a new efficiency frontier for modeling global spatial context in vision applications through its unique combination of structured matrix transformations and GPU-optimized implementation. Project page: this https URL 

---
# Referenceless Proton Resonance Frequency Thermometry Using Deep Learning with Self-Attention 

**Authors**: Yueran Zhao, Chang-Sheng Mei, Nathan J. McDannold, Shenyan Zong, Guofeng Shen  

**Link**: [PDF](https://arxiv.org/pdf/2512.07882)  

**Abstract**: Background: Accurate proton resonance frequency (PRF) MR thermometry is essential for monitoring temperature rise during thermal ablation with high intensity focused ultrasound (FUS). Conventional referenceless methods such as complex field estimation (CFE) and phase finite difference (PFD) tend to exhibit errors when susceptibility-induced phase discontinuities occur at tissue interfaces. 

---
# Artificial Intelligence-Driven Network-on-Chip Design Space Exploration: Neural Network Architectures for Design 

**Authors**: Amogh Anshu N, Harish BP  

**Link**: [PDF](https://arxiv.org/pdf/2512.07877)  

**Abstract**: Network-on-Chip (NoC) design requires exploring a high-dimensional configuration space to satisfy stringent throughput requirements and latency this http URL design space exploration techniques are often slow and struggle to handle complex, non-linear parameter this http URL work presents a machine learning-driven framework that automates NoC design space exploration using BookSim simulations and reverse neural network this http URL, we compare three architectures - a Multi-Layer Perceptron (MLP),a Conditional Diffusion Model, and a Conditional Variational Autoencoder (CVAE) to predict optimal NoC parameters given target performance this http URL pipeline generates over 150,000 simulation data points across varied mesh this http URL Conditional Diffusion Model achieved the highest predictive accuracy, attaining a mean squared error (MSE) of 0.463 on unseen this http URL, the proposed framework reduces design exploration time by several orders of magnitude, making it a practical solution for rapid and scalable NoC co-design. 

---
# Advancing physiological time series reconstruction and imputation via mixture of receptive fields and experts fusion 

**Authors**: Ci Zhang, Huayu Li, Changdi Yang, Jiangnan Xia, Yanzhi Wang, Xiaolong Ma, Jin Lu, Geng Yuan  

**Link**: [PDF](https://arxiv.org/pdf/2512.07873)  

**Abstract**: Recent studies show that using diffusion models for time series signal reconstruc- tion holds great promise. However, such approaches remain largely unexplored in the domain of medical time series. The unique characteristics of the physiological time series signals, such as multivariate, high temporal variability, highly noisy, and artifact-prone, make deep learning-based approaches still challenging for tasks such as imputation. Hence, we propose a novel Mixture of Experts (MoE)-based noise estimator within a score-based diffusion framework. Specifically, the Receptive Field Adaptive MoE (RFAMoE) module is designed to enable each channel to adap- tively select desired receptive fields throughout the diffusion process. Moreover, recent literature has found that when generating a physiological signal, performing multiple inferences and averaging the reconstructed signals can effectively reduce reconstruction errors, but at the cost of significant computational and latency over- head. We design a Fusion MoE module and innovatively leverage the nature of MoE module to generate K noise signals in parallel, fuse them using a routing mechanism, and complete signal reconstruction in a single inference step. This design not only improves performance over previous methods but also eliminates the substantial computational cost and latency associated with multiple inference processes. Extensive results demonstrate that our proposed framework consistently outperforms diffusion-based SOTA works on different tasks and datasets. 

---
# Quantum Circuit Reasoning Models: A Variational Framework for Differentiable Logical Inference 

**Authors**: Andrew Kiruluta  

**Link**: [PDF](https://arxiv.org/pdf/2512.07871)  

**Abstract**: This report introduces a novel class of reasoning architectures, termed Quantum Circuit Reasoning Models (QCRM), which extend the concept of Variational Quantum Circuits (VQC) from energy minimization and classification tasks to structured logical inference and reasoning. We posit that fundamental quantum mechanical operations, superposition, entanglement, interference, and measurement, naturally map to essential reasoning primitives such as hypothesis branching, constraint propagation, consistency enforcement, and decision making. The resulting framework combines quantum-inspired computation with differentiable optimization, enabling reasoning to emerge as a process of amplitude evolution and interference-driven selection of self-consistent states. We develop the mathematical foundation of QCRM, define its parameterized circuit architecture, and show how logical rules can be encoded as unitary transformations over proposition-qubit states. We further formalize a training objective grounded in classical gradient descent over circuit parameters and discuss simulation-based implementations on classical hardware. Finally, we propose the Quantum Reasoning Layer (QRL) as a differentiable hybrid component for composable reasoning models applicable to scientific, biomedical, and chemical inference domains. 

---
# Manifolds and Modules: How Function Develops in a Neural Foundation Model 

**Authors**: Johannes Bertram, Luciano Dyballa, T. Anderson Keller, Savik Kinger, Steven W. Zucker  

**Link**: [PDF](https://arxiv.org/pdf/2512.07869)  

**Abstract**: Foundation models have shown remarkable success in fitting biological visual systems; however, their black-box nature inherently limits their utility for under- standing brain function. Here, we peek inside a SOTA foundation model of neural activity (Wang et al., 2025) as a physiologist might, characterizing each 'neuron' based on its temporal response properties to parametric stimuli. We analyze how different stimuli are represented in neural activity space by building decoding man- ifolds, and we analyze how different neurons are represented in stimulus-response space by building neural encoding manifolds. We find that the different processing stages of the model (i.e., the feedforward encoder, recurrent, and readout modules) each exhibit qualitatively different representational structures in these manifolds. The recurrent module shows a jump in capabilities over the encoder module by 'pushing apart' the representations of different temporal stimulus patterns; while the readout module achieves biological fidelity by using numerous specialized feature maps rather than biologically plausible mechanisms. Overall, we present this work as a study of the inner workings of a prominent neural foundation model, gaining insights into the biological relevance of its internals through the novel analysis of its neurons' joint temporal response patterns. 

---
# Bayesian Optimization for Function-Valued Responses under Min-Max Criteria 

**Authors**: Pouya Ahadi, Reza Marzban, Ali Adibi, Kamran Paynabar  

**Link**: [PDF](https://arxiv.org/pdf/2512.07868)  

**Abstract**: Bayesian optimization is widely used for optimizing expensive black box functions, but most existing approaches focus on scalar responses. In many scientific and engineering settings the response is functional, varying smoothly over an index such as time or wavelength, which makes classical formulations inadequate. Existing methods often minimize integrated error, which captures average performance but neglects worst case deviations. To address this limitation we propose min-max Functional Bayesian Optimization (MM-FBO), a framework that directly minimizes the maximum error across the functional domain. Functional responses are represented using functional principal component analysis, and Gaussian process surrogates are constructed for the principal component scores. Building on this representation, MM-FBO introduces an integrated uncertainty acquisition function that balances exploitation of worst case expected error with exploration across the functional domain. We provide two theoretical guarantees: a discretization bound for the worst case objective, and a consistency result showing that as the surrogate becomes accurate and uncertainty vanishes, the acquisition converges to the true min-max objective. We validate the method through experiments on synthetic benchmarks and physics inspired case studies involving electromagnetic scattering by metaphotonic devices and vapor phase infiltration. Results show that MM-FBO consistently outperforms existing baselines and highlights the importance of explicitly modeling functional uncertainty in Bayesian optimization. 

---
# LLM-Generated Counterfactual Stress Scenarios for Portfolio Risk Simulation via Hybrid Prompt-RAG Pipeline 

**Authors**: Masoud Soleimani  

**Link**: [PDF](https://arxiv.org/pdf/2512.07867)  

**Abstract**: We develop a transparent and fully auditable LLM-based pipeline for macro-financial stress testing, combining structured prompting with optional retrieval of country fundamentals and news. The system generates machine-readable macroeconomic scenarios for the G7, which cover GDP growth, inflation, and policy rates, and are translated into portfolio losses through a factor-based mapping that enables Value-at-Risk and Expected Shortfall assessment relative to classical econometric baselines. Across models, countries, and retrieval settings, the LLMs produce coherent and country-specific stress narratives, yielding stable tail-risk amplification with limited sensitivity to retrieval choices. Comprehensive plausibility checks, scenario diagnostics, and ANOVA-based variance decomposition show that risk variation is driven primarily by portfolio composition and prompt design rather than by the retrieval mechanism. The pipeline incorporates snapshotting, deterministic modes, and hash-verified artifacts to ensure reproducibility and auditability. Overall, the results demonstrate that LLM-generated macro scenarios, when paired with transparent structure and rigorous validation, can provide a scalable and interpretable complement to traditional stress-testing frameworks. 

---
# Command & Control (C2) Traffic Detection Via Algorithm Generated Domain (Dga) Classification Using Deep Learning And Natural Language Processing 

**Authors**: Maria Milena Araujo Felix  

**Link**: [PDF](https://arxiv.org/pdf/2512.07866)  

**Abstract**: The sophistication of modern malware, specifically regarding communication with Command and Control (C2) servers, has rendered static blacklist-based defenses obsolete. The use of Domain Generation Algorithms (DGA) allows attackers to generate thousands of dynamic addresses daily, hindering blocking by traditional firewalls. This paper aims to propose and evaluate a method for detecting DGA domains using Deep Learning and Natural Language Processing (NLP) techniques. The methodology consisted of collecting a hybrid database containing 50,000 legitimate and 50,000 malicious domains, followed by the extraction of lexical features and the training of a Recurrent Neural Network (LSTM). Results demonstrated that while statistical entropy analysis is effective for simple DGAs, the Neural Network approach presents superiority in detecting complex patterns, reaching 97.2% accuracy and reducing the false positive rate in ambiguous lawful traffic scenarios. 

---
# GPU Memory Prediction for Multimodal Model Training 

**Authors**: Jinwoo Jeong, Minchul Kang, Younghun Go, Changyong Shin, Hyunho Lee, Junho Yoon, Gyeongsik Yang, Chuck Yoo  

**Link**: [PDF](https://arxiv.org/pdf/2512.07853)  

**Abstract**: As deep learning models in agentic AI systems grow in scale and complexity, GPU memory requirements increase and often exceed the available GPU memory capacity, so that out-of-memory (OoM) errors occur. It is well known that OoM interrupts the whole training itself and wastes substantial computational resources. Therefore, to prevent OoM, accurate prediction of GPU memory usage is essential. However, previous studies focus only on unimodal architectures and fail to generalize to multimodal models, even though the multimodal models are a common choice in agentic AI systems. To address this limitation, we propose a framework that predicts the peak GPU memory usage by analyzing the model architecture and training behavior of multimodal models. Specifically, the framework decomposes the multimodal model into its constituent layers and applies factorization to estimate the memory usage of each layer. Our evaluation shows that our framework achieves high prediction accuracy of ~8.7% average MAPE. 

---
# SABER: Small Actions, Big Errors - Safeguarding Mutating Steps in LLM Agents 

**Authors**: Alejandro Cuadron, Pengfei Yu, Yang Liu, Arpit Gupta  

**Link**: [PDF](https://arxiv.org/pdf/2512.07850)  

**Abstract**: Despite rapid progress in LLM agents, performance on long-horizon, tool-using tasks remains fragile. To better understand this fragility, we ask a simple question: \emph{do all actions contribute equally to failure?} Analyzing execution traces on $\tau$-Bench (Airline/Retail) and SWE-Bench Verified, we decompose trajectories into \emph{mutating} (environment-changing) vs.\ non-mutating steps and formalize \emph{decisive deviations}, earliest action, level divergences that flip success to failure. A logistic regression reveals that each additional deviation in a mutating action reduces the odds of success by upto $92\%$ on Airline and upto $96\%$ on Retail for SoTA models. In contrast, deviations in non-mutating actions have little to no effect. Errors also grow with context length as agents drift from role and act on stale constraints. Motivated by these observations, we introduce \cm{}, a model-agnostic, gradient-free, test-time safeguard that (i) adds mutation-gated verification, (ii) injects \emph{Targeted Reflection} before mutating steps, and (iii) performs block-based context cleaning. \cm{} delivers consistent gains, e.g., Qwen3-Thinking: +28\% \emph{relative} on Airline, +11\% on Retail, and +7\% on SWE-Bench Verified; Claude: +9\%/+7\%. We further identify ceiling effects in $\tau$-Bench, where annotation errors and underspecified tasks artificially cap model performance. To address this, we release $\tau$-Bench Verified, which restores benchmark headroom through targeted revisions. Our results argue for action-level analysis, targeted safeguards, and reliable evaluations as prerequisites for robust multi-turn agents. 

---
# MixLM: High-Throughput and Effective LLM Ranking via Text-Embedding Mix-Interaction 

**Authors**: Guoyao Li, Ran He, Shusen Jing, Kayhan Behdin, Yubo Wang, Sundara Raman Ramachandran, Chanh Nguyen, Jian Sheng, Xiaojing Ma, Chuanrui Zhu, Sriram Vasudevan, Muchen Wu, Sayan Ghosh, Lin Su, Qingquan Song, Xiaoqing Wang, Zhipeng Wang, Qing Lan, Yanning Chen, Jingwei Wu, Luke Simon, Wenjing Zhang, Qi Guo, Fedor Borisyuk  

**Link**: [PDF](https://arxiv.org/pdf/2512.07846)  

**Abstract**: Large language models (LLMs) excel at capturing semantic nuances and therefore show impressive relevance ranking performance in modern recommendation and search systems. However, they suffer from high computational overhead under industrial latency and throughput requirements. In particular, cross-encoder ranking systems often create long context prefill-heavy workloads, as the model has to be presented with the user, query and item information. To this end, we propose MixLM, a novel LLM-based ranking framework, which significantly improves the system throughput via reducing the input context length, while preserving the semantic strength of cross-encoder rankers. In contrast to a standard ranking system where the context is presented to the model as pure text, we propose to use mix-interaction, a mixture of text and embedding tokens to represent the input. Specifically, MixLM encodes all items in the catalog into a few embedding tokens and stores in a nearline cache. The encoded item descriptions are used during online inference, effectively reducing the item length from a few thousand text tokens to a few embedding tokens. We share insights from deploying our MixLM framework to a real-world search application at LinkedIn, including a detailed discussion of our training pipelines, as well as a thorough analysis of our online serving infrastructure optimization. Comparing with strong baselines, MixLM increased throughput by 10.0x under the same latency budget, while maintaining relevance metrics. The efficiency gains delivered by MixLM enabled full-traffic deployment of LLM-powered search, which resulted in a significant 0.47% increase in Daily Active Users (DAU) in online A/B tests. 

---
# AudioScene: Integrating Object-Event Audio into 3D Scenes 

**Authors**: Shuaihang Yuan, Congcong Wen, Muhammad Shafique, Anthony Tzes, Yi Fang  

**Link**: [PDF](https://arxiv.org/pdf/2512.07845)  

**Abstract**: The rapid advances in audio analysis underscore its vast potential for humancomputer interaction, environmental monitoring, and public safety; yet, existing audioonly datasets often lack spatial context. To address this gap, we present two novel audiospatial scene datasets, AudioScanNet and AudioRoboTHOR, designed to explore audioconditioned tasks within 3D environments. By integrating audio clips with spatially aligned 3D scenes, our datasets enable research on how audio signals interact with spatial context. To associate audio events with corresponding spatial information, we leverage the common sense reasoning ability of large language models and supplement them with rigorous human verification, This approach offers greater scalability compared to purely manual annotation while maintaining high standards of accuracy, completeness, and diversity, quantified through inter annotator agreement and performance on two benchmark tasks audio based 3D visual grounding and audio based robotic zeroshot navigation. The results highlight the limitations of current audiocentric methods and underscore the practical challenges and significance of our datasets in advancing audio guided spatial learning. 

---
# Space Alignment Matters: The Missing Piece for Inducing Neural Collapse in Long-Tailed Learning 

**Authors**: Jinping Wang, Zhiqiang Gao, Zhiwu Xie  

**Link**: [PDF](https://arxiv.org/pdf/2512.07844)  

**Abstract**: Recent studies on Neural Collapse (NC) reveal that, under class-balanced conditions, the class feature means and classifier weights spontaneously align into a simplex equiangular tight frame (ETF). In long-tailed regimes, however, severe sample imbalance tends to prevent the emergence of the NC phenomenon, resulting in poor generalization performance. Current efforts predominantly seek to recover the ETF geometry by imposing constraints on features or classifier weights, yet overlook a critical problem: There is a pronounced misalignment between the feature and the classifier weight spaces. In this paper, we theoretically quantify the harm of such misalignment through an optimal error exponent analysis. Built on this insight, we propose three explicit alignment strategies that plug-and-play into existing long-tail methods without architectural change. Extensive experiments on the CIFAR-10-LT, CIFAR-100-LT, and ImageNet-LT datasets consistently boost examined baselines and achieve the state-of-the-art performances. 

---
# ThreadWeaver: Adaptive Threading for Efficient Parallel Reasoning in Language Models 

**Authors**: Long Lian, Sida Wang, Felix Juefei-Xu, Tsu-Jui Fu, Xiuyu Li, Adam Yala, Trevor Darrell, Alane Suhr, Yuandong Tian, Xi Victoria Lin  

**Link**: [PDF](https://arxiv.org/pdf/2512.07843)  

**Abstract**: Scaling inference-time computation has enabled Large Language Models (LLMs) to achieve strong reasoning performance, but inherently sequential decoding leads to substantial latency, especially on complex tasks. Recent work on adaptive parallel reasoning aims to improve inference efficiency by decomposing the problem-solving process into concurrent reasoning threads when beneficial. However, existing methods on realistic tasks are either limited to supervised behavior cloning or exhibit significant accuracy drops compared to widely-used sequential long chain-of-thought (CoT) baselines. Moreover, many require customized inference engines, complicating deployment. We introduce ThreadWeaver, a framework for adaptive parallel reasoning that achieves accuracy on par with popular sequential reasoning models of comparable size while significantly reducing inference latency. ThreadWeaver's performance stems from three key innovations: 1) a two-stage parallel trajectory generator that produces large-scale, high-quality CoT data with parallel annotations for supervised fine-tuning; 2) a trie-based training-inference co-design that enables parallel reasoning on any off-the-shelf autoregressive inference engine without modifying position embeddings or KV caches; and 3) a parallelization-aware reinforcement learning framework that teaches the model to balance accuracy with effective parallelization. Across six challenging mathematical reasoning benchmarks, ThreadWeaver trained atop Qwen3-8B achieves accuracy comparable to cutting-edge sequential reasoning models (71.9% on average and 79.9% on AIME24) while delivering up to 1.53x average speedup in token latency, establishing a new Pareto frontier between accuracy and efficiency. 

---
# Automating High Energy Physics Data Analysis with LLM-Powered Agents 

**Authors**: Eli Gendreau-Distler, Joshua Ho, Dongwon Kim, Luc Tomas Le Pottier, Haichen Wang, Chengxi Yang  

**Link**: [PDF](https://arxiv.org/pdf/2512.07785)  

**Abstract**: We present a proof-of-principle study demonstrating the use of large language model (LLM) agents to automate a representative high energy physics (HEP) analysis. Using the Higgs boson diphoton cross-section measurement as a case study with ATLAS Open Data, we design a hybrid system that combines an LLM-based supervisor-coder agent with the Snakemake workflow manager. In this architecture, the workflow manager enforces reproducibility and determinism, while the agent autonomously generates, executes, and iteratively corrects analysis code in response to user instructions. We define quantitative evaluation metrics including success rate, error distribution, costs per specific task, and average number of API calls, to assess agent performance across multi-stage workflows. To characterize variability across architectures, we benchmark a representative selection of state-of-the-art LLMs spanning the Gemini and GPT-5 series, the Claude family, and leading open-weight models. While the workflow manager ensures deterministic execution of all analysis steps, the final outputs still show stochastic variation. Although we set the temperature to zero, other sampling parameters (e.g., top-p, top-k) remained at their defaults, and some reasoning-oriented models internally adjust these settings. Consequently, the models do not produce fully deterministic results. This study establishes the first LLM-agent-driven automated data-analysis framework in HEP, enabling systematic benchmarking of model capabilities, stability, and limitations in real-world scientific computing environments. The baseline code used in this work is available at this https URL. This work was accepted as a poster at the Machine Learning and the Physical Sciences (ML4PS) workshop at NeurIPS 2025. The initial submission was made on August 30, 2025. 

---
# MELLA: Bridging Linguistic Capability and Cultural Groundedness for Low-Resource Language MLLMs 

**Authors**: Yufei Gao, Jiaying Fei, Nuo Chen, Ruirui Chen, Guohang Yan, Yunshi Lan, Botian Shi  

**Link**: [PDF](https://arxiv.org/pdf/2508.05502)  

**Abstract**: Multimodal Large Language Models (MLLMs) have shown remarkable performance in high-resource languages. However, their effectiveness diminishes significantly in the contexts of low-resource languages. Current multilingual enhancement methods are often limited to text modality or rely solely on machine translation. While such approaches help models acquire basic linguistic capabilities and produce "thin descriptions", they neglect the importance of multimodal informativeness and cultural groundedness, both of which are crucial for serving low-resource language users effectively. To bridge this gap, in this study, we identify two significant objectives for a truly effective MLLM in low-resource language settings, namely 1) linguistic capability and 2) cultural groundedness, placing special emphasis on cultural awareness. To achieve these dual objectives, we propose a dual-source strategy that guides the collection of data tailored to each goal, sourcing native web alt-text for culture and MLLM-generated captions for linguistics. As a concrete implementation, we introduce MELLA, a multimodal, multilingual dataset. Experiment results show that after fine-tuning on MELLA, there is a general performance improvement for the eight languages on various MLLM backbones, with models producing "thick descriptions". We verify that the performance gains are from both cultural knowledge enhancement and linguistic capability enhancement. Our dataset can be found at this https URL. 

---
