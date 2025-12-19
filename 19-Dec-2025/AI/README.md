# Generative Adversarial Reasoner: Enhancing LLM Reasoning with Adversarial Reinforcement Learning 

**Authors**: Qihao Liu, Luoxin Ye, Wufei Ma, Yu-Cheng Chou, Alan Yuille  

**Link**: [PDF](https://arxiv.org/pdf/2512.16917)  

**Abstract**: Large language models (LLMs) with explicit reasoning capabilities excel at mathematical reasoning yet still commit process errors, such as incorrect calculations, brittle logic, and superficially plausible but invalid steps. In this paper, we introduce Generative Adversarial Reasoner, an on-policy joint training framework designed to enhance reasoning by co-evolving an LLM reasoner and an LLM-based discriminator through adversarial reinforcement learning. A compute-efficient review schedule partitions each reasoning chain into logically complete slices of comparable length, and the discriminator evaluates each slice's soundness with concise, structured justifications. Learning couples complementary signals: the LLM reasoner is rewarded for logically consistent steps that yield correct answers, while the discriminator earns rewards for correctly detecting errors or distinguishing traces in the reasoning process. This produces dense, well-calibrated, on-policy step-level rewards that supplement sparse exact-match signals, improving credit assignment, increasing sample efficiency, and enhancing overall reasoning quality of LLMs. Across various mathematical benchmarks, the method delivers consistent gains over strong baselines with standard RL post-training. Specifically, on AIME24, we improve DeepSeek-R1-Distill-Qwen-7B from 54.0 to 61.3 (+7.3) and DeepSeek-R1-Distill-Llama-8B from 43.7 to 53.7 (+10.0). The modular discriminator also enables flexible reward shaping for objectives such as teacher distillation, preference alignment, and mathematical proof-based reasoning. 

---
# The Social Responsibility Stack: A Control-Theoretic Architecture for Governing Socio-Technical AI 

**Authors**: Otman A. Basir  

**Link**: [PDF](https://arxiv.org/pdf/2512.16873)  

**Abstract**: Artificial intelligence systems are increasingly deployed in domains that shape human behaviour, institutional decision-making, and societal outcomes. Existing responsible AI and governance efforts provide important normative principles but often lack enforceable engineering mechanisms that operate throughout the system lifecycle. This paper introduces the Social Responsibility Stack (SRS), a six-layer architectural framework that embeds societal values into AI systems as explicit constraints, safeguards, behavioural interfaces, auditing mechanisms, and governance processes. SRS models responsibility as a closed-loop supervisory control problem over socio-technical systems, integrating design-time safeguards with runtime monitoring and institutional oversight. We develop a unified constraint-based formulation, introduce safety-envelope and feedback interpretations, and show how fairness, autonomy, cognitive burden, and explanation quality can be continuously monitored and enforced. Case studies in clinical decision support, cooperative autonomous vehicles, and public-sector systems illustrate how SRS translates normative objectives into actionable engineering and operational controls. The framework bridges ethics, control theory, and AI governance, providing a practical foundation for accountable, adaptive, and auditable socio-technical AI systems. 

---
# Distributional AGI Safety 

**Authors**: Nenad Tomašev, Matija Franklin, Julian Jacobs, Sébastien Krier, Simon Osindero  

**Link**: [PDF](https://arxiv.org/pdf/2512.16856)  

**Abstract**: AI safety and alignment research has predominantly been focused on methods for safeguarding individual AI systems, resting on the assumption of an eventual emergence of a monolithic Artificial General Intelligence (AGI). The alternative AGI emergence hypothesis, where general capability levels are first manifested through coordination in groups of sub-AGI individual agents with complementary skills and affordances, has received far less attention. Here we argue that this patchwork AGI hypothesis needs to be given serious consideration, and should inform the development of corresponding safeguards and mitigations. The rapid deployment of advanced AI agents with tool-use capabilities and the ability to communicate and coordinate makes this an urgent safety consideration. We therefore propose a framework for distributional AGI safety that moves beyond evaluating and aligning individual agents. This framework centers on the design and implementation of virtual agentic sandbox economies (impermeable or semi-permeable), where agent-to-agent transactions are governed by robust market mechanisms, coupled with appropriate auditability, reputation management, and oversight to mitigate collective risks. 

---
# TOGGLE: Temporal Logic-Guided Large Language Model Compression for Edge 

**Authors**: Khurram Khalil, Khaza Anuarul Hoque  

**Link**: [PDF](https://arxiv.org/pdf/2512.16855)  

**Abstract**: Large Language Models (LLMs) deliver exceptional performance across natural language tasks but demand substantial computational resources, limiting their deployment on resource-constrained edge devices. Existing compression techniques, such as quantization and pruning, often degrade critical linguistic properties and lack formal guarantees for preserving model behavior. We propose Temporal Logic-Guided Large Language Model Compression (TOGGLE), a novel framework that leverages Signal Temporal Logic (STL) to formally specify and enforce linguistic properties during compression. TOGGLE employs an STL robustness-guided Bayesian optimization to systematically explore layer-wise quantization and pruning configurations, generating compressed models that formally satisfy specified linguistic constraints without retraining or fine-tuning. Evaluating TOGGLE on four LLM architectures (GPT-2, DeepSeek-V2 7B, LLaMA 3 8B, and Mistral 7B), we achieve up to 3.3x reduction in computational costs (FLOPs) and up to a 68.8% reduction in model size while satisfying all linguistic properties. TOGGLE represents the first integration of formal methods into LLM compression, enabling efficient, verifiable deployment of LLMs on edge hardware. 

---
# CitySeeker: How Do VLMS Explore Embodied Urban Navigation With Implicit Human Needs? 

**Authors**: Siqi Wang, Chao Liang, Yunfan Gao, Erxin Yu, Sen Li, Yushi Li, Jing Li, Haofen Wang  

**Link**: [PDF](https://arxiv.org/pdf/2512.16755)  

**Abstract**: Vision-Language Models (VLMs) have made significant progress in explicit instruction-based navigation; however, their ability to interpret implicit human needs (e.g., "I am thirsty") in dynamic urban environments remains underexplored. This paper introduces CitySeeker, a novel benchmark designed to assess VLMs' spatial reasoning and decision-making capabilities for exploring embodied urban navigation to address implicit needs. CitySeeker includes 6,440 trajectories across 8 cities, capturing diverse visual characteristics and implicit needs in 7 goal-driven scenarios. Extensive experiments reveal that even top-performing models (e.g., Qwen2.5-VL-32B-Instruct) achieve only 21.1% task completion. We find key bottlenecks in error accumulation in long-horizon reasoning, inadequate spatial cognition, and deficient experiential recall. To further analyze them, we investigate a series of exploratory strategies-Backtracking Mechanisms, Enriching Spatial Cognition, and Memory-Based Retrieval (BCR), inspired by human cognitive mapping's emphasis on iterative observation-reasoning cycles and adaptive path optimization. Our analysis provides actionable insights for developing VLMs with robust spatial intelligence required for tackling "last-mile" navigation challenges. 

---
# AI-Driven Prediction of Cancer Pain Episodes: A Hybrid Decision Support Approach 

**Authors**: Yipeng Zhuang, Yifeng Guo, Yuewen Li, Yuheng Wu, Philip Leung-Ho Yu, Tingting Song, Zhiyong Wang, Kunzhong Zhou, Weifang Wang, Li Zhuang  

**Link**: [PDF](https://arxiv.org/pdf/2512.16739)  

**Abstract**: Lung cancer patients frequently experience breakthrough pain episodes, with up to 91% requiring timely intervention. To enable proactive pain management, we propose a hybrid machine learning and large language model pipeline that predicts pain episodes within 48 and 72 hours of hospitalization using both structured and unstructured electronic health record data. A retrospective cohort of 266 inpatients was analyzed, with features including demographics, tumor stage, vital signs, and WHO-tiered analgesic use. The machine learning module captured temporal medication trends, while the large language model interpreted ambiguous dosing records and free-text clinical notes. Integrating these modalities improved sensitivity and interpretability. Our framework achieved an accuracy of 0.874 (48h) and 0.917 (72h), with an improvement in sensitivity of 8.6% and 10.4% due to the augmentation of large language model. This hybrid approach offers a clinically interpretable and scalable tool for early pain episode forecasting, with potential to enhance treatment precision and optimize resource allocation in oncology care. 

---
# Discovering and Learning Probabilistic Models of Black-Box AI Capabilities 

**Authors**: Daniel Bramblett, Rushang Karia, Adrian Ciotinga, Ruthvick Suresh, Pulkit Verma, YooJung Choi, Siddharth Srivastava  

**Link**: [PDF](https://arxiv.org/pdf/2512.16733)  

**Abstract**: Black-box AI (BBAI) systems such as foundational models are increasingly being used for sequential decision making. To ensure that such systems are safe to operate and deploy, it is imperative to develop efficient methods that can provide a sound and interpretable representation of the BBAI's capabilities. This paper shows that PDDL-style representations can be used to efficiently learn and model an input BBAI's planning capabilities. It uses the Monte-Carlo tree search paradigm to systematically create test tasks, acquire data, and prune the hypothesis space of possible symbolic models. Learned models describe a BBAI's capabilities, the conditions under which they can be executed, and the possible outcomes of executing them along with their associated probabilities. Theoretical results show soundness, completeness and convergence of the learned models. Empirical results with multiple BBAI systems illustrate the scope, efficiency, and accuracy of the presented methods. 

---
# Dual Computational Horizons: Incompleteness and Unpredictability in Intelligent Systems 

**Authors**: Abhisek Ganguly  

**Link**: [PDF](https://arxiv.org/pdf/2512.16707)  

**Abstract**: We formalize two independent computational limitations that constrain algorithmic intelligence: formal incompleteness and dynamical unpredictability. The former limits the deductive power of consistent reasoning systems while the later bounds long-term prediction under finite precision. We show that these two extrema together impose structural bounds on an agent's ability to reason about its own predictive capabilities. In particular, an algorithmic agent cannot compute its own maximal prediction horizon generally. This perspective clarifies inherent trade-offs between reasoning, prediction, and self-analysis in intelligent systems. 

---
# Cyber Humanism in Education: Reclaiming Agency through AI and Learning Sciences 

**Authors**: Giovanni Adorni  

**Link**: [PDF](https://arxiv.org/pdf/2512.16701)  

**Abstract**: Generative Artificial Intelligence (GenAI) is rapidly reshaping how knowledge is produced and validated in education. Rather than adding another digital tool, large language models reconfigure reading, writing, and coding into hybrid human-AI workflows, raising concerns about epistemic automation, cognitive offloading, and the de-professiona\-lisation of teachers. This paper proposes \emph{Cyber Humanism in Education} as a framework for reclaiming human agency in this landscape. We conceptualise AI-enabled learning environments as socio-technical infrastructures co-authored by humans and machines, and position educators and learners as epistemic agents and \emph{algorithmic citizens} who have both the right and the responsibility to shape these infrastructures.
We articulate three pillars for cyber-humanist design, \emph{reflexive competence}, \emph{algorithmic citizenship}, and \emph{dialogic design}, and relate them to major international digital and AI competence frameworks. We then present higher-education case studies that operationalise these ideas through \emph{prompt-based learning} and a new \emph{Conversational AI Educator} certification within the EPICT ecosystem. The findings show how such practices can strengthen epistemic agency while surfacing tensions around workload, equity, and governance, and outline implications for the future of AI-rich, human-centred education. 

---
# Do Multi-Agents Solve Better Than Single? Evaluating Agentic Frameworks for Diagram-Grounded Geometry Problem Solving and Reasoning 

**Authors**: Mahbub E Sobhani, Md. Faiyaz Abdullah Sayeedi, Mohammad Nehad Alam, Proma Hossain Progga, Swakkhar Shatabda  

**Link**: [PDF](https://arxiv.org/pdf/2512.16698)  

**Abstract**: Diagram-grounded geometry problem solving is a critical benchmark for multimodal large language models (MLLMs), yet the benefits of multi-agent design over single-agent remain unclear. We systematically compare single-agent and multi-agent pipelines on four visual math benchmarks: Geometry3K, MathVerse, OlympiadBench, and We-Math. For open-source models, multi-agent consistently improves performance. For example, Qwen-2.5-VL (7B) gains +6.8 points and Qwen-2.5-VL (32B) gains +3.3 on Geometry3K, and both Qwen-2.5-VL variants see further gains on OlympiadBench and We-Math. In contrast, the closed-source Gemini-2.0-Flash generally performs better in single-agent mode on classic benchmarks, while multi-agent yields only modest improvements on the newer We-Math dataset. These findings show that multi-agent pipelines provide clear benefits for open-source models and can assist strong proprietary systems on newer, less familiar benchmarks, but agentic decomposition is not universally optimal. All code, data, and reasoning files are available at this https URL 

---
# Unsupervised Thematic Clustering Of hadith Texts Using The Apriori Algorithm 

**Authors**: Wisnu Uriawan, Achmad Ajie Priyajie, Angga Gustian, Fikri Nur Hidayat, Sendi Ahmad Rafiudin, Muhamad Fikri Zaelani  

**Link**: [PDF](https://arxiv.org/pdf/2512.16694)  

**Abstract**: This research stems from the urgency to automate the thematic grouping of hadith in line with the growing digitalization of Islamic texts. Based on a literature review, the unsupervised learning approach with the Apriori algorithm has proven effective in identifying association patterns and semantic relations in unlabeled text data. The dataset used is the Indonesian Translation of the hadith of Bukhari, which first goes through preprocessing stages including case folding, punctuation cleaning, tokenization, stopword removal, and stemming. Next, an association rule mining analysis was conducted using the Apriori algorithm with support, confidence, and lift parameters. The results show the existence of meaningful association patterns such as the relationship between rakaat-prayer, verse-revelation, and hadith-story, which describe the themes of worship, revelation, and hadith narration. These findings demonstrate that the Apriori algorithm has the ability to automatically uncover latent semantic relationships, while contributing to the development of digital Islamic studies and technology-based learning systems. 

---
# Comprehensive AI Literacy: The Case for Centering Human Agency 

**Authors**: Sri Yash Tadimalla, Justin Cary, Gordon Hull, Jordan Register, Daniel Maxwell, David Pugalee, Tina Heafner  

**Link**: [PDF](https://arxiv.org/pdf/2512.16656)  

**Abstract**: The rapid assimilation of Artificial Intelligence technologies into various facets of society has created a significant educational imperative that current frameworks are failing to effectively address. We are witnessing the rise of a dangerous literacy gap, where a focus on the functional, operational skills of using AI tools is eclipsing the development of critical and ethical reasoning about them. This position paper argues for a systemic shift toward comprehensive AI literacy that centers human agency - the empowered capacity for intentional, critical, and responsible choice. This principle applies to all stakeholders in the educational ecosystem: it is the student's agency to question, create with, or consciously decide not to use AI based on the task; it is the teacher's agency to design learning experiences that align with instructional values, rather than ceding pedagogical control to a tool. True literacy involves teaching about agency itself, framing technology not as an inevitability to be adopted, but as a choice to be made. This requires a deep commitment to critical thinking and a robust understanding of epistemology. Through the AI Literacy, Fluency, and Competency frameworks described in this paper, educators and students will become agents in their own human-centric approaches to AI, providing necessary pathways to clearly articulate the intentions informing decisions and attitudes toward AI and the impact of these decisions on academic work, career, and society. 

---
# Prefix Probing: Lightweight Harmful Content Detection for Large Language Models 

**Authors**: Jirui Yang, Hengqi Guo, Zhihui Lu, Yi Zhao, Yuansen Zhang, Shijing Hu, Qiang Duan, Yinggui Wang, Tao Wei  

**Link**: [PDF](https://arxiv.org/pdf/2512.16650)  

**Abstract**: Large language models often face a three-way trade-off among detection accuracy, inference latency, and deployment cost when used in real-world safety-sensitive applications. This paper introduces Prefix Probing, a black-box harmful content detection method that compares the conditional log-probabilities of "agreement/execution" versus "refusal/safety" opening prefixes and leverages prefix caching to reduce detection overhead to near first-token latency. During inference, the method requires only a single log-probability computation over the probe prefixes to produce a harmfulness score and apply a threshold, without invoking any additional models or multi-stage inference. To further enhance the discriminative power of the prefixes, we design an efficient prefix construction algorithm that automatically discovers highly informative prefixes, substantially improving detection performance. Extensive experiments demonstrate that Prefix Probing achieves detection effectiveness comparable to mainstream external safety models while incurring only minimal computational cost and requiring no extra model deployment, highlighting its strong practicality and efficiency. 

---
# Implementing a Sharia Chatbot as a Consultation Medium for Questions About Islam 

**Authors**: Wisnu Uriawan, Aria Octavian Hamza, Ade Ripaldi Nuralim, Adi Purnama, Ahmad Juaeni Yunus, Anissya Auliani Supriadi Putri  

**Link**: [PDF](https://arxiv.org/pdf/2512.16644)  

**Abstract**: This research presents the implementation of a Sharia-compliant chatbot as an interactive medium for consulting Islamic questions, leveraging Reinforcement Learning (Q-Learning) integrated with Sentence-Transformers for semantic embedding to ensure contextual and accurate responses. Utilizing the CRISP-DM methodology, the system processes a curated Islam QA dataset of 25,000 question-answer pairs from authentic sources like the Qur'an, Hadith, and scholarly fatwas, formatted in JSON for flexibility and scalability. The chatbot prototype, developed with a Flask API backend and Flutter-based mobile frontend, achieves 87% semantic accuracy in functional testing across diverse topics including fiqh, aqidah, ibadah, and muamalah, demonstrating its potential to enhance religious literacy, digital da'wah, and access to verified Islamic knowledge in the Industry 4.0 era. While effective for closed-domain queries, limitations such as static learning and dataset dependency highlight opportunities for future enhancements like continuous adaptation and multi-turn conversation support, positioning this innovation as a bridge between traditional Islamic scholarship and modern AI-driven consultation. 

---
# Needle in the Web: A Benchmark for Retrieving Targeted Web Pages in the Wild 

**Authors**: Yumeng Wang, Tianyu Fan, Lingrui Xu, Chao Huang  

**Link**: [PDF](https://arxiv.org/pdf/2512.16553)  

**Abstract**: Large Language Models (LLMs) have evolved from simple chatbots into sophisticated agents capable of automating complex real-world tasks, where browsing and reasoning over live web content is key to assessing retrieval and cognitive skills. Existing benchmarks like BrowseComp and xBench-DeepSearch emphasize complex reasoning searches requiring multi-hop synthesis but neglect Fuzzy Exploratory Search, namely queries that are vague and multifaceted, where users seek the most relevant webpage rather than a single factual answer. To address this gap, we introduce Needle in the Web, a novel benchmark specifically designed to evaluate modern search agents and LLM-based systems on their ability to retrieve and reason over real-world web content in response to ambiguous, exploratory queries under varying levels of difficulty. Needle in the Web comprises 663 questions spanning seven distinct domains. To ensure high query quality and answer uniqueness, we employ a flexible methodology that reliably generates queries of controllable difficulty based on factual claims of web contents. We benchmark three leading LLMs and three agent-based search systems on Needle in the Web, finding that most models struggle: many achieve below 35% accuracy, and none consistently excel across domains or difficulty levels. These findings reveal that Needle in the Web presents a significant challenge for current search systems and highlights the open problem of effective fuzzy retrieval under semantic ambiguity. 

---
# From Personalization to Prejudice: Bias and Discrimination in Memory-Enhanced AI Agents for Recruitment 

**Authors**: Himanshu Gharat, Himanshi Agrawal, Gourab K. Patro  

**Link**: [PDF](https://arxiv.org/pdf/2512.16532)  

**Abstract**: Large Language Models (LLMs) have empowered AI agents with advanced capabilities for understanding, reasoning, and interacting across diverse tasks. The addition of memory further enhances them by enabling continuity across interactions, learning from past experiences, and improving the relevance of actions and responses over time; termed as memory-enhanced personalization. Although such personalization through memory offers clear benefits, it also introduces risks of bias. While several previous studies have highlighted bias in ML and LLMs, bias due to memory-enhanced personalized agents is largely unexplored. Using recruitment as an example use case, we simulate the behavior of a memory-enhanced personalized agent, and study whether and how bias is introduced and amplified in and across various stages of operation. Our experiments on agents using safety-trained LLMs reveal that bias is systematically introduced and reinforced through personalization, emphasizing the need for additional protective measures or agent guardrails in memory-enhanced LLM-based AI agents. 

---
# Scaling Laws for Energy Efficiency of Local LLMs 

**Authors**: Ander Alvarez, Alessandro Genuardi, Nilotpal Sinha, Antonio Tiene, Samuel Mugel, Román Orús  

**Link**: [PDF](https://arxiv.org/pdf/2512.16531)  

**Abstract**: Deploying local large language models and vision-language models on edge devices requires balancing accuracy with constrained computational and energy budgets. Although graphics processors dominate modern artificial-intelligence deployment, most consumer hardware--including laptops, desktops, industrial controllers, and embedded systems--relies on central processing units. Despite this, the computational laws governing central-processing-unit-only inference for local language and vision-language workloads remain largely unexplored. We systematically benchmark large language and vision-language models on two representative central-processing-unit tiers widely used for local inference: a MacBook Pro M2, reflecting mainstream laptop-class deployment, and a Raspberry Pi 5, representing constrained, low-power embedded settings. Using a unified methodology based on continuous sampling of processor and memory usage together with area-under-curve integration, we characterize how computational load scales with input text length for language models and with image resolution for vision-language models. We uncover two empirical scaling laws: (1) computational cost for language-model inference scales approximately linearly with token length; and (2) vision-language models exhibit a preprocessing-driven "resolution knee", where compute remains constant above an internal resolution clamp and decreases sharply below it. Beyond these laws, we show that quantum-inspired compression reduces processor and memory usage by up to 71.9% and energy consumption by up to 62%, while preserving or improving semantic accuracy. These results provide a systematic quantification of multimodal central-processing-unit-only scaling for local language and vision-language workloads, and they identify model compression and input-resolution preprocessing as effective, low-cost levers for sustainable edge inference. 

---
# ParamExplorer: A framework for exploring parameters in generative art 

**Authors**: Julien Gachadoat, Guillaume Lagarde  

**Link**: [PDF](https://arxiv.org/pdf/2512.16529)  

**Abstract**: Generative art systems often involve high-dimensional and complex parameter spaces in which aesthetically compelling outputs occupy only small, fragmented regions. Because of this combinatorial explosion, artists typically rely on extensive manual trial-and-error, leaving many potentially interesting configurations undiscovered. In this work we make two contributions. First, we introduce ParamExplorer, an interactive and modular framework inspired by reinforcement learning that helps the exploration of parameter spaces in generative art algorithms, guided by human-in-the-loop or even automated feedback. The framework also integrates seamlessly with existing this http URL projects. Second, within this framework we implement and evaluate several exploration strategies, referred to as agents. 

---
# Best Practices For Empirical Meta-Algorithmic Research Guidelines from the COSEAL Research Network 

**Authors**: Theresa Eimer, Lennart Schäpermeier, André Biedenkapp, Alexander Tornede, Lars Kotthoff, Pieter Leyman, Matthias Feurer, Katharina Eggensperger, Kaitlin Maile, Tanja Tornede, Anna Kozak, Ke Xue, Marcel Wever, Mitra Baratchi, Damir Pulatov, Heike Trautmann, Haniye Kashgarani, Marius Lindauer  

**Link**: [PDF](https://arxiv.org/pdf/2512.16491)  

**Abstract**: Empirical research on meta-algorithmics, such as algorithm selection, configuration, and scheduling, often relies on extensive and thus computationally expensive experiments. With the large degree of freedom we have over our experimental setup and design comes a plethora of possible error sources that threaten the scalability and validity of our scientific insights. Best practices for meta-algorithmic research exist, but they are scattered between different publications and fields, and continue to evolve separately from each other. In this report, we collect good practices for empirical meta-algorithmic research across the subfields of the COSEAL community, encompassing the entire experimental cycle: from formulating research questions and selecting an experimental design, to executing ex- periments, and ultimately, analyzing and presenting results impartially. It establishes the current state-of-the-art practices within meta-algorithmic research and serves as a guideline to both new researchers and practitioners in meta-algorithmic fields. 

---
# Quantifying and Bridging the Fidelity Gap: A Decisive-Feature Approach to Comparing Synthetic and Real Imagery 

**Authors**: Danial Safaei, Siddartha Khastgir, Mohsen Alirezaei, Jeroen Ploeg, Son Tong, Xingyu Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2512.16468)  

**Abstract**: Virtual testing using synthetic data has become a cornerstone of autonomous vehicle (AV) safety assurance. Despite progress in improving visual realism through advanced simulators and generative AI, recent studies reveal that pixel-level fidelity alone does not ensure reliable transfer from simulation to the real world. What truly matters is whether the system-under-test (SUT) bases its decisions on the same causal evidence in both real and simulated environments - not just whether images "look real" to humans. This paper addresses the lack of such a behavior-grounded fidelity measure by introducing Decisive Feature Fidelity (DFF), a new SUT-specific metric that extends the existing fidelity spectrum to capture mechanism parity - the agreement in causal evidence underlying the SUT's decisions across domains. DFF leverages explainable-AI (XAI) methods to identify and compare the decisive features driving the SUT's outputs for matched real-synthetic pairs. We further propose practical estimators based on counterfactual explanations, along with a DFF-guided calibration scheme to enhance simulator fidelity. Experiments on 2126 matched KITTI-VirtualKITTI2 pairs demonstrate that DFF reveals discrepancies overlooked by conventional output-value fidelity. Furthermore, results show that DFF-guided calibration improves decisive-feature and input-level fidelity without sacrificing output value fidelity across diverse SUTs. 

---
# cuPilot: A Strategy-Coordinated Multi-agent Framework for CUDA Kernel Evolution 

**Authors**: Jinwu Chen, Qidie Wu, Bin Li, Lin Ma, Xin Si, Yang Hu, Shouyi Yin, Jun Yang  

**Link**: [PDF](https://arxiv.org/pdf/2512.16465)  

**Abstract**: Optimizing CUDA kernels is a challenging and labor-intensive task, given the need for hardware-software co-design expertise and the proprietary nature of high-performance kernel libraries. While recent large language models (LLMs) combined with evolutionary algorithms show promise in automatic kernel optimization, existing approaches often fall short in performance due to their suboptimal agent designs and mismatched evolution representations. This work identifies these mismatches and proposes cuPilot, a strategy-coordinated multi-agent framework that introduces strategy as an intermediate semantic representation for kernel evolution. Key contributions include a strategy-coordinated evolution algorithm, roofline-guided prompting, and strategy-level population initialization. Experimental results show that the generated kernels by cuPilot achieve an average speed up of 3.09$\times$ over PyTorch on a benchmark of 100 kernels. On the GEMM tasks, cuPilot showcases sophisticated optimizations and achieves high utilization of critical hardware units. The generated kernels are open-sourced at this https URL. 

---
# TimeSeries2Report prompting enables adaptive large language model management of lithium-ion batteries 

**Authors**: Jiayang Yang, Chunhui Zhao, Martin Guay, Zhixing Cao  

**Link**: [PDF](https://arxiv.org/pdf/2512.16453)  

**Abstract**: Large language models (LLMs) offer promising capabilities for interpreting multivariate time-series data, yet their application to real-world battery energy storage system (BESS) operation and maintenance remains largely unexplored. Here, we present TimeSeries2Report (TS2R), a prompting framework that converts raw lithium-ion battery operational time-series into structured, semantically enriched reports, enabling LLMs to reason, predict, and make decisions in BESS management scenarios. TS2R encodes short-term temporal dynamics into natural language through a combination of segmentation, semantic abstraction, and rule-based interpretation, effectively bridging low-level sensor signals with high-level contextual insights. We benchmark TS2R across both lab-scale and real-world datasets, evaluating report quality and downstream task performance in anomaly detection, state-of-charge prediction, and charging/discharging management. Compared with vision-, embedding-, and text-based prompting baselines, report-based prompting via TS2R consistently improves LLM performance in terms of across accuracy, robustness, and explainability metrics. Notably, TS2R-integrated LLMs achieve expert-level decision quality and predictive consistency without retraining or architecture modification, establishing a practical path for adaptive, LLM-driven battery intelligence. 

---
# Towards AI-Supported Research: a Vision of the TIB AIssistant 

**Authors**: Sören Auer, Allard Oelen, Mohamad Yaser Jaradeh, Mutahira Khalid, Farhana Keya, Sasi Kiran Gaddipati, Jennifer D'Souza, Lorenz Schlüter, Amirreza Alasti, Gollam Rabby, Azanzi Jiomekong, Oliver Karras  

**Link**: [PDF](https://arxiv.org/pdf/2512.16447)  

**Abstract**: The rapid advancements in Generative AI and Large Language Models promise to transform the way research is conducted, potentially offering unprecedented opportunities to augment scholarly workflows. However, effectively integrating AI into research remains a challenge due to varying domain requirements, limited AI literacy, the complexity of coordinating tools and agents, and the unclear accuracy of Generative AI in research. We present the vision of the TIB AIssistant, a domain-agnostic human-machine collaborative platform designed to support researchers across disciplines in scientific discovery, with AI assistants supporting tasks across the research life cycle. The platform offers modular components - including prompt and tool libraries, a shared data store, and a flexible orchestration framework - that collectively facilitate ideation, literature analysis, methodology development, data analysis, and scholarly writing. We describe the conceptual framework, system architecture, and implementation of an early prototype that demonstrates the feasibility and potential impact of our approach. 

---
# StarCraft+: Benchmarking Multi-agent Algorithms in Adversary Paradigm 

**Authors**: Yadong Li, Tong Zhang, Bo Huang, Zhen Cui  

**Link**: [PDF](https://arxiv.org/pdf/2512.16444)  

**Abstract**: Deep multi-agent reinforcement learning (MARL) algorithms are booming in the field of collaborative intelligence, and StarCraft multi-agent challenge (SMAC) is widely-used as the benchmark therein. However, imaginary opponents of MARL algorithms are practically configured and controlled in a fixed built-in AI mode, which causes less diversity and versatility in algorithm evaluation. To address this issue, in this work, we establish a multi-agent algorithm-vs-algorithm environment, named StarCraft II battle arena (SC2BA), to refresh the benchmarking of MARL algorithms in an adversary paradigm. Taking StarCraft as infrastructure, the SC2BA environment is specifically created for inter-algorithm adversary with the consideration of fairness, usability and customizability, and meantime an adversarial PyMARL (APyMARL) library is developed with easy-to-use interfaces/modules. Grounding in SC2BA, we benchmark those classic MARL algorithms in two types of adversarial modes: dual-algorithm paired adversary and multi-algorithm mixed adversary, where the former conducts the adversary of pairwise algorithms while the latter focuses on the adversary to multiple behaviors from a group of algorithms. The extensive benchmark experiments exhibit some thought-provoking observations/problems in the effectivity, sensibility and scalability of these completed algorithms. The SC2BA environment as well as reproduced experiments are released in \href{this https URL}{Github}, and we believe that this work could mark a new step for the MARL field in the coming years. 

---
# TIB AIssistant: a Platform for AI-Supported Research Across Research Life Cycles 

**Authors**: Allard Oelen, Sören Auer  

**Link**: [PDF](https://arxiv.org/pdf/2512.16442)  

**Abstract**: The rapidly growing popularity of adopting Artificial Intelligence (AI), and specifically Large Language Models (LLMs), is having a widespread impact throughout society, including the academic domain. AI-supported research has the potential to support researchers with tasks across the entire research life cycle. In this work, we demonstrate the TIB AIssistant, an AI-supported research platform providing support throughout the research life cycle. The AIssistant consists of a collection of assistants, each responsible for a specific research task. In addition, tools are provided to give access to external scholarly services. Generated data is stored in the assets and can be exported as an RO-Crate bundle to provide transparency and enhance reproducibility of the research project. We demonstrate the AIssistant's main functionalities by means of a sequential walk-through of assistants, interacting with each other to generate sections for a draft research paper. In the end, with the AIssistant, we lay the foundation for a larger agenda of providing a community-maintained platform for AI-supported research. 

---
# Synthelite: Chemist-aligned and feasibility-aware synthesis planning with LLMs 

**Authors**: Nguyen Xuan-Vu, Daniel Armstrong, Milena Wehrbach, Andres M Bran, Zlatko Jončev, Philippe Schwaller  

**Link**: [PDF](https://arxiv.org/pdf/2512.16424)  

**Abstract**: Computer-aided synthesis planning (CASP) has long been envisioned as a complementary tool for synthetic chemists. However, existing frameworks often lack mechanisms to allow interaction with human experts, limiting their ability to integrate chemists' insights. In this work, we introduce Synthelite, a synthesis planning framework that uses large language models (LLMs) to directly propose retrosynthetic transformations. Synthelite can generate end-to-end synthesis routes by harnessing the intrinsic chemical knowledge and reasoning capabilities of LLMs, while allowing expert intervention through natural language prompts. Our experiments demonstrate that Synthelite can flexibly adapt its planning trajectory to diverse user-specified constraints, achieving up to 95\% success rates in both strategy-constrained and starting-material-constrained synthesis tasks. Additionally, Synthelite exhibits the ability to account for chemical feasibility during route design. We envision Synthelite to be both a useful tool and a step toward a paradigm where LLMs are the central orchestrators of synthesis planning. 

---
# PCIA: A Path Construction Imitation Algorithm for Global Optimization 

**Authors**: Mohammad-Javad Rezaei, Mozafar Bag-Mohammadi  

**Link**: [PDF](https://arxiv.org/pdf/2512.16392)  

**Abstract**: In this paper, a new metaheuristic optimization algorithm, called Path Construction Imitation Algorithm (PCIA), is proposed. PCIA is inspired by how humans construct new paths and use them. Typically, humans prefer popular transportation routes. In the event of a path closure, a new route is built by mixing the existing paths intelligently. Also, humans select different pathways on a random basis to reach unknown destinations. PCIA generates a random population to find the best route toward the destination, similar to swarm-based algorithms. Each particle represents a path toward the destination. PCIA has been tested with 53 mathematical optimization problems and 13 constrained optimization problems. The results showed that the PCIA is highly competitive compared to both popular and the latest metaheuristic algorithms. 

---
# AI Needs Physics More Than Physics Needs AI 

**Authors**: Peter Coveney, Roger Highfield  

**Link**: [PDF](https://arxiv.org/pdf/2512.16344)  

**Abstract**: Artificial intelligence (AI) is commonly depicted as transformative. Yet, after more than a decade of hype, its measurable impact remains modest outside a few high-profile scientific and commercial successes. The 2024 Nobel Prizes in Chemistry and Physics recognized AI's potential, but broader assessments indicate the impact to date is often more promotional than technical. We argue that while current AI may influence physics, physics has significantly more to offer this generation of AI. Current architectures - large language models, reasoning models, and agentic AI - can depend on trillions of meaningless parameters, suffer from distributional bias, lack uncertainty quantification, provide no mechanistic insights, and fail to capture even elementary scientific laws. We review critiques of these limits, highlight opportunities in quantum AI and analogue computing, and lay down a roadmap for the adoption of 'Big AI': a synthesis of theory-based rigour with the flexibility of machine learning. 

---
# Design and Evaluation of Cost-Aware PoQ for Decentralized LLM Inference 

**Authors**: Arther Tian, Alex Ding, Frank Chen, Alan Wu, Aaron Chan, Bruce Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2512.16317)  

**Abstract**: Decentralized large language model (LLM) inference promises transparent and censorship resistant access to advanced AI, yet existing verification approaches struggle to scale to modern models. Proof of Quality (PoQ) replaces cryptographic verification of computation with consensus over output quality, but the original formulation ignores heterogeneous computational costs across inference and evaluator nodes. This paper introduces a cost-aware PoQ framework that integrates explicit efficiency measurements into the reward mechanism for both types of nodes. The design combines ground truth token level F1, lightweight learned evaluators, and GPT based judgments within a unified evaluation pipeline, and adopts a linear reward function that balances normalized quality and cost.
Experiments on extractive question answering and abstractive summarization use five instruction tuned LLMs ranging from TinyLlama-1.1B to Llama-3.2-3B and three evaluation models spanning cross encoder and bi encoder architectures. Results show that a semantic textual similarity bi encoder achieves much higher correlation with both ground truth and GPT scores than cross encoders, indicating that evaluator architecture is a critical design choice for PoQ. Quality-cost analysis further reveals that the largest models in the pool are also the most efficient in terms of quality per unit latency. Monte Carlo simulations over 5\,000 PoQ rounds demonstrate that the cost-aware reward scheme consistently assigns higher average rewards to high quality low cost inference models and to efficient evaluators, while penalizing slow low quality nodes. These findings suggest that cost-aware PoQ provides a practical foundation for economically sustainable decentralized LLM inference. 

---
# Adaptation of Agentic AI 

**Authors**: Pengcheng Jiang, Jiacheng Lin, Zhiyi Shi, Zifeng Wang, Luxi He, Yichen Wu, Ming Zhong, Peiyang Song, Qizheng Zhang, Heng Wang, Xueqiang Xu, Hanwen Xu, Pengrui Han, Dylan Zhang, Jiashuo Sun, Chaoqi Yang, Kun Qian, Tian Wang, Changran Hu, Manling Li, Quanzheng Li, Hao Peng, Sheng Wang, Jingbo Shang, Chao Zhang, Jiaxuan You, Liyuan Liu, Pan Lu, Yu Zhang, Heng Ji, Yejin Choi, Dawn Song, Jimeng Sun, Jiawei Han  

**Link**: [PDF](https://arxiv.org/pdf/2512.16301)  

**Abstract**: Cutting-edge agentic AI systems are built on foundation models that can be adapted to plan, reason, and interact with external tools to perform increasingly complex and specialized tasks. As these systems grow in capability and scope, adaptation becomes a central mechanism for improving performance, reliability, and generalization. In this paper, we unify the rapidly expanding research landscape into a systematic framework that spans both agent adaptations and tool adaptations. We further decompose these into tool-execution-signaled and agent-output-signaled forms of agent adaptation, as well as agent-agnostic and agent-supervised forms of tool adaptation. We demonstrate that this framework helps clarify the design space of adaptation strategies in agentic AI, makes their trade-offs explicit, and provides practical guidance for selecting or switching among strategies during system design. We then review the representative approaches in each category, analyze their strengths and limitations, and highlight key open challenges and future opportunities. Overall, this paper aims to offer a conceptual foundation and practical roadmap for researchers and practitioners seeking to build more capable, efficient, and reliable agentic AI systems. 

---
# Code-in-the-Loop Forensics: Agentic Tool Use for Image Forgery Detection 

**Authors**: Fanrui Zhang, Qiang Zhang, Sizhuo Zhou, Jianwen Sun, Chuanhao Li, Jiaxin Ai, Yukang Feng, Yujie Zhang, Wenjie Li, Zizhen Li, Yifan Chang, Jiawei Liu, Kaipeng Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2512.16300)  

**Abstract**: Existing image forgery detection (IFD) methods either exploit low-level, semantics-agnostic artifacts or rely on multimodal large language models (MLLMs) with high-level semantic knowledge. Although naturally complementary, these two information streams are highly heterogeneous in both paradigm and reasoning, making it difficult for existing methods to unify them or effectively model their cross-level interactions. To address this gap, we propose ForenAgent, a multi-round interactive IFD framework that enables MLLMs to autonomously generate, execute, and iteratively refine Python-based low-level tools around the detection objective, thereby achieving more flexible and interpretable forgery analysis. ForenAgent follows a two-stage training pipeline combining Cold Start and Reinforcement Fine-Tuning to enhance its tool interaction capability and reasoning adaptability progressively. Inspired by human reasoning, we design a dynamic reasoning loop comprising global perception, local focusing, iterative probing, and holistic adjudication, and instantiate it as both a data-sampling strategy and a task-aligned process reward. For systematic training and evaluation, we construct FABench, a heterogeneous, high-quality agent-forensics dataset comprising 100k images and approximately 200k agent-interaction question-answer pairs. Experiments show that ForenAgent exhibits emergent tool-use competence and reflective reasoning on challenging IFD tasks when assisted by low-level tools, charting a promising route toward general-purpose IFD. The code will be released after the review process is completed. 

---
# OS-Oracle: A Comprehensive Framework for Cross-Platform GUI Critic Models 

**Authors**: Zhenyu Wu, Jingjing Xie, Zehao Li, Bowen Yang, Qiushi Sun, Zhaoyang Liu, Zhoumianze Liu, Yu Qiao, Xiangyu Yue, Zun Wang, Zichen Ding  

**Link**: [PDF](https://arxiv.org/pdf/2512.16295)  

**Abstract**: With VLM-powered computer-using agents (CUAs) becoming increasingly capable at graphical user interface (GUI) navigation and manipulation, reliable step-level decision-making has emerged as a key bottleneck for real-world deployment. In long-horizon workflows, errors accumulate quickly and irreversible actions can cause unintended consequences, motivating critic models that assess each action before execution. While critic models offer a promising solution, their effectiveness is hindered by the lack of diverse, high-quality GUI feedback data and public critic benchmarks for step-level evaluation in computer use. To bridge these gaps, we introduce OS-Oracle that makes three core contributions: (1) a scalable data pipeline for synthesizing cross-platform GUI critic data; (2) a two-stage training paradigm combining supervised fine-tuning (SFT) and consistency-preserving group relative policy optimization (CP-GRPO); (3) OS-Critic Bench, a holistic benchmark for evaluating critic model performance across Mobile, Web, and Desktop platforms. Leveraging this framework, we curate a high-quality dataset containing 310k critic samples. The resulting critic model, OS-Oracle-7B, achieves state-of-the-art performance among open-source VLMs on OS-Critic Bench, and surpasses proprietary models on the mobile domain. Furthermore, when serving as a pre-critic, OS-Oracle-7B improves the performance of native GUI agents such as UI-TARS-1.5-7B in OSWorld and AndroidWorld environments. The code is open-sourced at this https URL. 

---
# QuadSentinel: Sequent Safety for Machine-Checkable Control in Multi-agent Systems 

**Authors**: Yiliu Yang, Yilei Jiang, Qunzhong Wang, Yingshui Tan, Xiaoyong Zhu, Sherman S.M. Chow, Bo Zheng, Xiangyu Yue  

**Link**: [PDF](https://arxiv.org/pdf/2512.16279)  

**Abstract**: Safety risks arise as large language model-based agents solve complex tasks with tools, multi-step plans, and inter-agent messages. However, deployer-written policies in natural language are ambiguous and context dependent, so they map poorly to machine-checkable rules, and runtime enforcement is unreliable. Expressing safety policies as sequents, we propose \textsc{QuadSentinel}, a four-agent guard (state tracker, policy verifier, threat watcher, and referee) that compiles these policies into machine-checkable rules built from predicates over observable state and enforces them online. Referee logic plus an efficient top-$k$ predicate updater keeps costs low by prioritizing checks and resolving conflicts hierarchically. Measured on ST-WebAgentBench (ICML CUA~'25) and AgentHarm (ICLR~'25), \textsc{QuadSentinel} improves guardrail accuracy and rule recall while reducing false positives. Against single-agent baselines such as ShieldAgent (ICML~'25), it yields better overall safety control. Near-term deployments can adopt this pattern without modifying core agents by keeping policies separate and machine-checkable. Our code will be made publicly available at this https URL. 

---
# Learning to Wait: Synchronizing Agents with the Physical World 

**Authors**: Yifei She, Ping Zhang, He Liu, Yanmin Jia, Yang Jing, Zijun Liu, Peng Sun, Xiangbin Li, Xiaohe Hu  

**Link**: [PDF](https://arxiv.org/pdf/2512.16262)  

**Abstract**: Real-world agentic tasks, unlike synchronous Markov Decision Processes (MDPs), often involve non-blocking actions with variable latencies, creating a fundamental \textit{Temporal Gap} between action initiation and completion. Existing environment-side solutions, such as blocking wrappers or frequent polling, either limit scalability or dilute the agent's context window with redundant observations. In this work, we propose an \textbf{Agent-side Approach} that empowers Large Language Models (LLMs) to actively align their \textit{Cognitive Timeline} with the physical world. By extending the Code-as-Action paradigm to the temporal domain, agents utilize semantic priors and In-Context Learning (ICL) to predict precise waiting durations (\texttt{this http URL(t)}), effectively synchronizing with asynchronous environment without exhaustive checking. Experiments in a simulated Kubernetes cluster demonstrate that agents can precisely calibrate their internal clocks to minimize both query overhead and execution latency, validating that temporal awareness is a learnable capability essential for autonomous evolution in open-ended environments. 

---
# AMUSE: Audio-Visual Benchmark and Alignment Framework for Agentic Multi-Speaker Understanding 

**Authors**: Sanjoy Chowdhury, Karren D. Yang, Xudong Liu, Fartash Faghri, Pavan Kumar Anasosalu Vasu, Oncel Tuzel, Dinesh Manocha, Chun-Liang Li, Raviteja Vemulapalli  

**Link**: [PDF](https://arxiv.org/pdf/2512.16250)  

**Abstract**: Recent multimodal large language models (MLLMs) such as GPT-4o and Qwen3-Omni show strong perception but struggle in multi-speaker, dialogue-centric settings that demand agentic reasoning tracking who speaks, maintaining roles, and grounding events across time. These scenarios are central to multimodal audio-video understanding, where models must jointly reason over audio and visual streams in applications such as conversational video assistants and meeting analytics. We introduce AMUSE, a benchmark designed around tasks that are inherently agentic, requiring models to decompose complex audio-visual interactions into planning, grounding, and reflection steps. It evaluates MLLMs across three modes zero-shot, guided, and agentic and six task families, including spatio-temporal speaker grounding and multimodal dialogue summarization. Across all modes, current models exhibit weak multi-speaker reasoning and inconsistent behavior under both non-agentic and agentic evaluation. Motivated by the inherently agentic nature of these tasks and recent advances in LLM agents, we propose RAFT, a data-efficient agentic alignment framework that integrates reward optimization with intrinsic multimodal self-evaluation as reward and selective parameter adaptation for data and parameter efficient updates. Using RAFT, we achieve up to 39.52\% relative improvement in accuracy on our benchmark. Together, AMUSE and RAFT provide a practical platform for examining agentic reasoning in multimodal models and improving their capabilities. 

---
# AlignMerge - Alignment-Preserving Large Language Model Merging via Fisher-Guided Geometric Constraints 

**Authors**: Aniruddha Roy, Jyoti Patel, Aman Chadha, Vinija Jain, Amitava Das  

**Link**: [PDF](https://arxiv.org/pdf/2512.16245)  

**Abstract**: Merging large language models (LLMs) is a practical way to compose capabilities from multiple fine-tuned checkpoints without retraining. Yet standard schemes (linear weight soups, task vectors, and Fisher-weighted averaging) can preserve loss while quietly destroying alignment. We argue that merging is not a numerical trick but a geometry-constrained operation around an already-aligned anchor: fusion must be steered to respect safety geometry, not validated post hoc.
We introduce AlignMerge, a geometry-aware merging framework that makes alignment an explicit invariant. In a local Fisher chart around an instruction-tuned base, we estimate an alignment subspace with projector P_A and optimize:
L_AlignMerge = L_geo + lambda_align * L_align + lambda_bud * L_bud,
where L_geo keeps the merge close to its experts in Fisher-Rao geometry, L_align penalizes motion along alignment-sensitive directions, and L_bud enforces a soft alignment budget. As the alignment functional we use the decoding-invariant Alignment Quality Index (AQI), a latent-space criterion that captures how cleanly aligned and misaligned behaviors separate in representation space.
Across five model families (LLaMA-3 8B, Mistral 7B, Qwen 2, Phi-3.5, Gemma 2), merging safety anchors with task experts, AlignMerge improves alignment metrics (AQI, toxicity, LLM-judge alignment) while matching or exceeding the best expert on instruction-following, reasoning, and helpfulness. It also exhibits smaller alignment-subspace drift and fewer budget violations than Fisher soups, TIES, SafeMerge, and MergeAlign. These results make alignment-preserving merging a first-class design goal and suggest a path to geometry-aware composition of future foundation models. 

---
# Scaling Spatial Reasoning in MLLMs through Programmatic Data Synthesis 

**Authors**: Zhi Helu, Huang Jingjing, Xu Wang, Xu Yangbin, Zhang Wanyue, Jiang Baoyang, Deng Shirui, Zhu Liang, Li Fangfang, Zhao Tiejun, Lin Yankai, Yao Yuan  

**Link**: [PDF](https://arxiv.org/pdf/2512.16237)  

**Abstract**: Embodied intelligence, a grand challenge in artificial intelligence, is fundamentally constrained by the limited spatial understanding and reasoning capabilities of current models. Prevailing efforts to address this through enhancing Vision-Language Models (VLMs) are trapped in a dilemma: template-based datasets are scalable but structurally rigid, while manual annotation is linguistically diverse but unscalable and, critically, computationally imprecise. We introduce SPRITE, a novel framework that overcomes this dilemma by leveraging simulators and large models to programmatically synthesize scalable, diverse, and high-quality spatial reasoning data. The core innovation of SPRITE is to reframe ground-truth generation as a code-generation task. We utilize LLMs to compile complex spatial questions into executable programs, which are then verified against high-precision scene meta-information extracted from simulators. This ensures our ground truth is both computationally precise and verifiable, while the generative power of LLMs provides vast linguistic diversity. Leveraging this pipeline, we have curated a dataset encompassing 3 simulators, 11k+ scenes, and 300k+ image/video instruction-tuning pairs. We demonstrate that a VLM trained on our data achieves significant performance gains on multiple spatial benchmarks and outperforms other open-source datasets of equivalent size. Furthermore, a scalability analysis confirms our hypothesis that overcoming the low-diversity nature of traditional template methods is essential for building robust, generalizable spatial intelligence. We will make the SPRITE framework code and the full 300k+ dataset publicly available to facilitate future research in spatial intelligence. 

---
# PDE-Agent: A toolchain-augmented multi-agent framework for PDE solving 

**Authors**: Jianming Liu, Ren Zhu, Jian Xu, Kun Ding, Xu-Yao Zhang, Gaofeng Meng, Cheng-Lin Liu  

**Link**: [PDF](https://arxiv.org/pdf/2512.16214)  

**Abstract**: Solving Partial Differential Equations (PDEs) is a cornerstone of engineering and scientific research. Traditional methods for PDE solving are cumbersome, relying on manual setup and domain expertise. While Physics-Informed Neural Network (PINNs) introduced end-to-end neural network-based solutions, and frameworks like DeepXDE further enhanced automation, these approaches still depend on expert knowledge and lack full autonomy. In this work, we frame PDE solving as tool invocation via LLM-driven agents and introduce PDE-Agent, the first toolchain-augmented multi-agent collaboration framework, inheriting the reasoning capacity of LLMs and the controllability of external tools and enabling automated PDE solving from natural language descriptions. PDE-Agent leverages the strengths of multi-agent and multi-tool collaboration through two key innovations: (1) A Prog-Act framework with graph memory for multi-agent collaboration, which enables effective dynamic planning and error correction via dual-loop mechanisms (localized fixes and global revisions). (2) A Resource-Pool integrated with a tool-parameter separation mechanism for multi-tool collaboration. This centralizes the management of runtime artifacts and resolves inter-tool dependency gaps in existing frameworks. To validate and evaluate this new paradigm for PDE solving , we develop PDE-Bench, a multi-type PDE Benchmark for agent-based tool collaborative solving, and propose multi-level metrics for assessing tool coordination. Evaluations verify that PDE-Agent exhibits superior applicability and performance in complex multi-step, cross-step dependent tasks. This new paradigm of toolchain-augmented multi-agent PDE solving will further advance future developments in automated scientific computing. Our source code and dataset will be made publicly available. 

---
# Weighted K-Harmonic Means Clustering: Convergence Analysis and Applications to Wireless Communications 

**Authors**: Gourab Ghatak  

**Link**: [PDF](https://arxiv.org/pdf/2512.16185)  

**Abstract**: We propose the \emph{weighted K-harmonic means} (WKHM) clustering algorithm, a regularized variant of K-harmonic means designed to ensure numerical stability while enabling soft assignments through inverse-distance weighting. Unlike classical K-means and constrained K-means, WKHM admits a direct interpretation in wireless networks: its weights are exactly equivalent to fractional user association based on received signal strength. We establish rigorous convergence guarantees under both deterministic and stochastic settings, addressing key technical challenges arising from non-convexity and random initialization. Specifically, we prove monotone descent to a local minimum under fixed initialization, convergence in probability under Binomial Point Process (BPP) initialization, and almost sure convergence under mild decay conditions. These results provide the first stochastic convergence guarantees for harmonic-mean-based clustering. Finally, through extensive simulations with diverse user distributions, we show that WKHM achieves a superior tradeoff between minimum signal strength and load fairness compared to classical and modern clustering baselines, making it a principled tool for joint radio node placement and user association in wireless networks. 

---
# Science Consultant Agent 

**Authors**: Karthikeyan K, Philip Wu, Xin Tang, Alexandre Alves  

**Link**: [PDF](https://arxiv.org/pdf/2512.16171)  

**Abstract**: The Science Consultant Agent is a web-based Artificial Intelligence (AI) tool that helps practitioners select and implement the most effective modeling strategy for AI-based solutions. It operates through four core components: Questionnaire, Smart Fill, Research-Guided Recommendation, and Prototype Builder. By combining structured questionnaires, literature-backed solution recommendations, and prototype generation, the Science Consultant Agent accelerates development for everyone from Product Managers and Software Developers to Researchers. The full pipeline is illustrated in Figure 1. 

---
# ToolForge: A Data Synthesis Pipeline for Multi-Hop Search without Real-World APIs 

**Authors**: Hao Chen, Zhexin Hu, Jiajun Chai, Haocheng Yang, Hang He, Xiaohan Wang, Wei Lin, Luhang Wang, Guojun Yin, Zhuofeng zhao  

**Link**: [PDF](https://arxiv.org/pdf/2512.16149)  

**Abstract**: Training LLMs to invoke tools and leverage retrieved information necessitates high-quality, diverse data. However, existing pipelines for synthetic data generation often rely on tens of thousands of real API calls to enhance generalization, incurring prohibitive costs while lacking multi-hop reasoning and self-reflection. To address these limitations, we introduce ToolForge, an automated synthesis framework that achieves strong real-world tool-calling performance by constructing only a small number of virtual tools, eliminating the need for real API calls. ToolForge leverages a (question, golden context, answer) triple to synthesize large-scale tool-learning data specifically designed for multi-hop search scenarios, further enriching the generated data through multi-hop reasoning and self-reflection mechanisms. To ensure data fidelity, we employ a Multi-Layer Validation Framework that integrates both rule-based and model-based assessments. Empirical results show that a model with only 8B parameters, when trained on our synthesized data, outperforms GPT-4o on multiple benchmarks. Our code and dataset are publicly available at this https URL . 

---
# WeMusic-Agent: Efficient Conversational Music Recommendation via Knowledge Internalization and Agentic Boundary Learning 

**Authors**: Wendong Bi, Yirong Mao, Xianglong Liu, Kai Tian, Jian Zhang, Hanjie Wang, Wenhui Que  

**Link**: [PDF](https://arxiv.org/pdf/2512.16108)  

**Abstract**: Personalized music recommendation in conversational scenarios usually requires a deep understanding of user preferences and nuanced musical context, yet existing methods often struggle with balancing specialized domain knowledge and flexible tool integration. This paper proposes WeMusic-Agent, a training framework for efficient LLM-based conversational music recommendation. By integrating the knowledge internalization and agentic boundary learning, the framework aims to teach the model to intelligently decide when to leverage internalized knowledge and when to call specialized tools (e.g., music retrieval APIs, music recommendation systems). Under this framework, we present WeMusic-Agent-M1, an agentic model that internalizes extensive musical knowledge via continued pretraining on 50B music-related corpus while acquiring the ability to invoke external tools when necessary. Additionally, considering the lack of open-source benchmarks for conversational music recommendation, we also construct a benchmark for personalized music recommendations derived from real-world data in WeChat Listen. This benchmark enables comprehensive evaluation across multiple dimensions, including relevance, personalization, and diversity of the recommendations. Experiments on real-world data demonstrate that WeMusic-Agent achieves significant improvements over existing models. 

---
# Topic Discovery and Classification for Responsible Generative AI Adaptation in Higher Education 

**Authors**: Diane Myung-kyung Woodbridge, Allyson Seba, Freddie Seba, Aydin Schwartz  

**Link**: [PDF](https://arxiv.org/pdf/2512.16036)  

**Abstract**: As generative artificial intelligence (GenAI) becomes increasingly capable of delivering personalized learning experiences and real-time feedback, a growing number of students are incorporating these tools into their academic workflows. They use GenAI to clarify concepts, solve complex problems, and, in some cases, complete assignments by copying and pasting model-generated contents. While GenAI has the potential to enhance learning experience, it also raises concerns around misinformation, hallucinated outputs, and its potential to undermine critical thinking and problem-solving skills. In response, many universities, colleges, departments, and instructors have begun to develop and adopt policies to guide responsible integration of GenAI into learning environments. However, these policies vary widely across institutions and contexts, and their evolving nature often leaves students uncertain about expectations and best practices. To address this challenge, the authors designed and implemented an automated system for discovering and categorizing AI-related policies found in course syllabi and institutional policy websites. The system combines unsupervised topic modeling techniques to identify key policy themes with large language models (LLMs) to classify the level of GenAI allowance and other requirements in policy texts. The developed application achieved a coherence score of 0.73 for topic discovery. In addition, GPT-4.0-based classification of policy categories achieved precision between 0.92 and 0.97, and recall between 0.85 and 0.97 across eight identified topics. By providing structured and interpretable policy information, this tool promotes the safe, equitable, and pedagogically aligned use of GenAI technologies in education. Furthermore, the system can be integrated into educational technology platforms to help students understand and comply with relevant guidelines. 

---
# Do Large Language Models Know What They Don't Know? Kalshibench: A New Benchmark for Evaluating Epistemic Calibration via Prediction Markets 

**Authors**: Lukas Nel  

**Link**: [PDF](https://arxiv.org/pdf/2512.16030)  

**Abstract**: A well-calibrated model should express confidence that matches its actual accuracy -- when it claims 80\% confidence, it should be correct 80\% of the time. While large language models (LLMs) have achieved remarkable performance across diverse tasks, their epistemic calibration remains poorly understood. We introduce \textbf{KalshiBench}, a benchmark of 300 prediction market questions from Kalshi, a CFTC-regulated exchange, with verifiable real-world outcomes occurring after model training cutoffs. Unlike traditional benchmarks measuring accuracy on static knowledge, KalshiBench evaluates whether models can appropriately quantify uncertainty about genuinely unknown future events. We evaluate five frontier models -- Claude Opus 4.5, GPT-5.2, DeepSeek-V3.2, Qwen3-235B, and Kimi-K2 -- and find \textbf{systematic overconfidence across all models}. Even the best-calibrated model (Claude Opus 4.5, ECE=0.120) shows substantial calibration errors, while reasoning-enhanced models like GPT-5.2-XHigh exhibit \emph{worse} calibration (ECE=0.395) despite comparable accuracy. Critically, only one model achieves a positive Brier Skill Score, indicating most models perform worse than simply predicting base rates. Our findings suggest that scaling and enhanced reasoning do not automatically confer calibration benefits, highlighting epistemic calibration as a distinct capability requiring targeted development. 

---
# Conversational Time Series Foundation Models: Towards Explainable and Effective Forecasting 

**Authors**: Defu Cao, Michael Gee, Jinbo Liu, Hengxuan Wang, Wei Yang, Rui Wang, Yan Liu  

**Link**: [PDF](https://arxiv.org/pdf/2512.16022)  

**Abstract**: The proliferation of time series foundation models has created a landscape where no single method achieves consistent superiority, framing the central challenge not as finding the best model, but as orchestrating an optimal ensemble with interpretability. While Large Language Models (LLMs) offer powerful reasoning capabilities, their direct application to time series forecasting has proven ineffective. We address this gap by repositioning the LLM as an intelligent judge that evaluates, explains, and strategically coordinates an ensemble of foundation models. To overcome the LLM's inherent lack of domain-specific knowledge on time series, we introduce an R1-style finetuning process, guided by SHAP-based faithfulness scores, which teaches the model to interpret ensemble weights as meaningful causal statements about temporal dynamics. The trained agent then engages in iterative, multi-turn conversations to perform forward-looking assessments, provide causally-grounded explanations for its weighting decisions, and adaptively refine the optimization strategy. Validated on the GIFT-Eval benchmark on 23 datasets across 97 settings, our approach significantly outperforms leading time series foundation models on both CRPS and MASE metrics, establishing new state-of-the-art results. 

---
# Subjective functions 

**Authors**: Samuel J. Gershman  

**Link**: [PDF](https://arxiv.org/pdf/2512.15948)  

**Abstract**: Where do objective functions come from? How do we select what goals to pursue? Human intelligence is adept at synthesizing new objective functions on the fly. How does this work, and can we endow artificial systems with the same ability? This paper proposes an approach to answering these questions, starting with the concept of a subjective function, a higher-order objective function that is endogenous to the agent (i.e., defined with respect to the agent's features, rather than an external task). Expected prediction error is studied as a concrete example of a subjective function. This proposal has many connections to ideas in psychology, neuroscience, and machine learning. 

---
# Small Language Models for Efficient Agentic Tool Calling: Outperforming Large Models with Targeted Fine-tuning 

**Authors**: Polaris Jhandi, Owais Kazi, Shreyas Subramanian, Neel Sendas  

**Link**: [PDF](https://arxiv.org/pdf/2512.15943)  

**Abstract**: As organizations scale adoption of generative AI, model cost optimization and operational efficiency have emerged as critical factors determining sustainability and accessibility. While Large Language Models (LLMs) demonstrate impressive capabilities across diverse tasks, their extensive computational requirements make them cost-prohibitive for routine enterprise use. This limitation motivates the exploration of Small Language Models (SLMs), which can deliver comparable performance in targeted applications while drastically reducing infrastructure overhead (Irugalbandara et al., 2023). In this work, we investigate the feasibility of replacing LLM-driven workflows with optimized SLMs. We trained a domain-adapted SLM to execute representative tasks traditionally handled by LLMs, such as document summarization, query answering, and structured data interpretation. As part of the experiment, we investigated the fine-tuning of facebook/opt-350m model (single epoch only) using the Hugging Face TRL (Transformer Reinforcement Learning), specifically the Supervised Fine-Tuning (SFT) trainer. The OPT-350M model was released by Meta AI in 2022 as part of the OPT (Open Pretrained Transformer) family of models. Similar studies demonstrate that even models at the 350M parameter scale can meaningfully contribute to instruction-tuning pipelines (Mekala et al., 2024). Experimental results demonstrated that our fine-tuned SLM achieves exceptional performance with a 77.55\% pass rate on ToolBench evaluation, significantly outperforming all baseline models including ChatGPT-CoT (26.00\%), ToolLLaMA-DFS (30.18\%), and ToolLLaMA-CoT (16.27\%). These findings emphasize that thoughtful design and targeted training of SLMs can significantly lower barriers to adoption, enabling cost-effective, large-scale integration of generative AI into production systems. 

---
# Leveraging Spreading Activation for Improved Document Retrieval in Knowledge-Graph-Based RAG Systems 

**Authors**: Jovan Pavlović, Miklós Krész, László Hajdu  

**Link**: [PDF](https://arxiv.org/pdf/2512.15922)  

**Abstract**: Despite initial successes and a variety of architectures, retrieval-augmented generation (RAG) systems still struggle to reliably retrieve and connect the multi-step evidence required for complicated reasoning tasks. Most of the standard RAG frameworks regard all retrieved information as equally reliable, overlooking the varying credibility and interconnected nature of large textual corpora. GraphRAG approaches offer potential improvement to RAG systems by integrating knowledge graphs, which structure information into nodes and edges, capture entity relationships, and enable multi-step logical traversal. However, GraphRAG is not always an ideal solution as it depends on high-quality graph representations of the corpus, which requires either pre-existing knowledge graphs that are expensive to build and update, or automated graph construction pipelines that are often unreliable. Moreover, systems following this paradigm typically use large language models to guide graph traversal and evidence retrieval, leading to challenges similar to those encountered with standard RAG. In this paper, we propose a novel RAG framework that employs the spreading activation algorithm to retrieve information from a corpus of documents interconnected by automatically constructed knowledge graphs, thereby enhancing the performance of large language models on complex tasks such as multi-hop question answering. Experiments show that our method achieves better or comparable performance to iterative RAG methodologies, while also being easily integrable as a plug-and-play module with a wide range of RAG-based approaches. Combining our method with chain-of-thought iterative retrieval yields up to a 39\% absolute gain in answer correctness compared to naive RAG, achieving these results with small open-weight language models and highlighting its effectiveness in resource-constrained settings. 

---
# Darth Vecdor: An Open-Source System for Generating Knowledge Graphs Through Large Language Model Queries 

**Authors**: Jonathan A. Handler  

**Link**: [PDF](https://arxiv.org/pdf/2512.15906)  

**Abstract**: Many large language models (LLMs) are trained on a massive body of knowledge present on the Internet. Darth Vecdor (DV) was designed to extract this knowledge into a structured, terminology-mapped, SQL database ("knowledge base" or "knowledge graph"). Knowledge graphs may be useful in many domains, including healthcare. Although one might query an LLM directly rather than a SQL-based knowledge graph, concerns such as cost, speed, safety, and confidence may arise, especially in high-volume operations. These may be mitigated when the information is pre-extracted from the LLM and becomes query-able through a standard database. However, the author found the need to address several issues. These included erroneous, off-topic, free-text, overly general, and inconsistent LLM responses, as well as allowing for multi-element responses. DV was built with features intended to mitigate these issues. To facilitate ease of use, and to allow for prompt engineering by those with domain expertise but little technical background, DV provides a simple, browser-based graphical user interface. DV has been released as free, open-source, extensible software, on an "as is" basis, without warranties or conditions of any kind, either express or implied. Users need to be cognizant of the potential risks and benefits of using DV and its outputs, and users are responsible for ensuring any use is safe and effective. DV should be assumed to have bugs, potentially very serious ones. However, the author hopes that appropriate use of current and future versions of DV and its outputs can help improve healthcare. 

---
# PediatricAnxietyBench: Evaluating Large Language Model Safety Under Parental Anxiety and Pressure in Pediatric Consultations 

**Authors**: Vahideh Zolfaghari  

**Link**: [PDF](https://arxiv.org/pdf/2512.15894)  

**Abstract**: Large language models (LLMs) are increasingly consulted by parents for pediatric guidance, yet their safety under real-world adversarial pressures is poorly understood. Anxious parents often use urgent language that can compromise model safeguards, potentially causing harmful advice. PediatricAnxietyBench is an open-source benchmark of 300 high-quality queries across 10 pediatric topics (150 patient-derived, 150 adversarial) enabling reproducible evaluation. Two Llama models (70B and 8B) were assessed using a multi-dimensional safety framework covering diagnostic restraint, referral adherence, hedging, and emergency recognition. Adversarial queries incorporated parental pressure patterns, including urgency, economic barriers, and challenges to disclaimers. Mean safety score was 5.50/15 (SD=2.41). The 70B model outperformed the 8B model (6.26 vs 4.95, p<0.001) with lower critical failures (4.8% vs 12.0%, p=0.02). Adversarial queries reduced safety by 8% (p=0.03), with urgency causing the largest drop (-1.40). Vulnerabilities appeared in seizures (33.3% inappropriate diagnosis) and post-vaccination queries. Hedging strongly correlated with safety (r=0.68, p<0.001), while emergency recognition was absent. Model scale influences safety, yet all models showed vulnerabilities to realistic parental pressures. PediatricAnxietyBench provides a reusable adversarial evaluation framework to reveal clinically significant failure modes overlooked by standard benchmarks. 

---
# State-Augmented Graphs for Circular Economy Triage 

**Authors**: Richard Fox, Rui Li, Gustav Jonsson, Farzaneh Goli, Miying Yang, Emel Aktas, Yongjing Wang  

**Link**: [PDF](https://arxiv.org/pdf/2512.15824)  

**Abstract**: Circular economy (CE) triage is the assessment of products to determine which sustainable pathway they can follow once they reach the end of their usefulness as they are currently being used. Effective CE triage requires adaptive decisions that balance retained value against the costs and constraints of processing and labour. This paper presents a novel decision-making framework as a simple deterministic solver over a state-augmented Disassembly Sequencing Planning (DSP) graph. By encoding the disassembly history into the state, our framework enforces the Markov property, enabling optimal, recursive evaluation by ensuring each decision only depends on the previous state. The triage decision involves choices between continuing disassembly or committing to a CE option. The model integrates condition-aware utility based on diagnostic health scores and complex operational constraints. We demonstrate the framework's flexibility with a worked example: the hierarchical triage of electric vehicle (EV) batteries, where decisions are driven by the recursive valuation of components. The example illustrates how a unified formalism enables the accommodation of varying mechanical complexity, safety requirements, and economic drivers. This unified formalism therefore provides a tractable and generalisable foundation for optimising CE triage decisions across diverse products and operational contexts. 

---
# Beyond Training: Enabling Self-Evolution of Agents with MOBIMEM 

**Authors**: Zibin Liu, Cheng Zhang, Xi Zhao, Yunfei Feng, Bingyu Bai, Dahu Feng, Erhu Feng, Yubin Xia, Haibo Chen  

**Link**: [PDF](https://arxiv.org/pdf/2512.15784)  

**Abstract**: Large Language Model (LLM) agents are increasingly deployed to automate complex workflows in mobile and desktop environments. However, current model-centric agent architectures struggle to self-evolve post-deployment: improving personalization, capability, and efficiency typically requires continuous model retraining/fine-tuning, which incurs prohibitive computational overheads and suffers from an inherent trade-off between model accuracy and inference efficiency.
To enable iterative self-evolution without model retraining, we propose MOBIMEM, a memory-centric agent system. MOBIMEM first introduces three specialized memory primitives to decouple agent evolution from model weights: (1) Profile Memory uses a lightweight distance-graph (DisGraph) structure to align with user preferences, resolving the accuracy-latency trade-off in user profile retrieval; (2) Experience Memory employs multi-level templates to instantiate execution logic for new tasks, ensuring capability generalization; and (3) Action Memory records fine-grained interaction sequences, reducing the reliance on expensive model inference. Building upon this memory architecture, MOBIMEM further integrates a suite of OS-inspired services to orchestrate execution: a scheduler that coordinates parallel sub-task execution and memory operations; an agent record-and-replay (AgentRR) mechanism that enables safe and efficient action reuse; and a context-aware exception handling that ensures graceful recovery from user interruptions and runtime errors.
Evaluation on AndroidWorld and top-50 apps shows that MOBIMEM achieves 83.1% profile alignment with 23.83 ms retrieval time (280x faster than GraphRAG baselines), improves task success rates by up to 50.3%, and reduces end-to-end latency by up to 9x on mobile devices. 

---
# AI Epidemiology: achieving explainable AI through expert oversight patterns 

**Authors**: Kit Tempest-Walters  

**Link**: [PDF](https://arxiv.org/pdf/2512.15783)  

**Abstract**: AI Epidemiology is a framework for governing and explaining advanced AI systems by applying population-level surveillance methods to AI outputs. The approach mirrors the way in which epidemiologists enable public health interventions through statistical evidence before molecular mechanisms are understood. This bypasses the problem of model complexity which plagues current interpretability methods (such as SHAP and mechanistic interpretability) at the scale of deployed models.
AI Epidemiology achieves this population-level surveillance by standardising capture of AI-expert interactions into structured assessment fields: risk level, alignment score, and accuracy score. These function as exposure variables which predict output failure through statistical associations, much like cholesterol and blood pressure act as exposure variables predicting cardiac events. Output-failure associations are subsequently validated against expert overrides and real-world outcomes.
The framework places zero burden on experts and provides automatic audit trails by passively tracking expert convergence and divergence with AI recommendations. Since it analyses outputs rather than internal model computations, it also provides governance continuity when institutions update models and switch vendors. Finally, by providing reliability scores and semantic assessments (e.g. 'this recommendation resembles 500 cases overridden by experts due to guideline violations'), it enables experts and institutions to detect unreliable AI outputs before they cause harm. This democratises AI oversight by enabling domain experts to govern AI systems without requiring machine learning expertise. 

---
# Emergence: Overcoming Privileged Information Bias in Asymmetric Embodied Agents via Active Querying 

**Authors**: Shaun Baek, Sam Liu, Joseph Ukpong  

**Link**: [PDF](https://arxiv.org/pdf/2512.15776)  

**Abstract**: Large Language Models (LLMs) act as powerful reasoning engines but struggle with "symbol grounding" in embodied environments, particularly when information is asymmetrically distributed. We investigate the Privileged Information Bias (or "Curse of Knowledge"), where a knowledgeable "Leader" agent fails to guide a sensor-limited "Follower" due to a lack of Theory of Mind. To quantify this phenomenon, we propose a novel Asymmetric Assistive Reasoning framework within AI2-THOR. Our experiments reveal a significant "Success Gap": while the Leader successfully perceives the target in 35.0% of episodes, the collaborative team succeeds only 17.0% of the time, implying that nearly 50% of feasible plans fail solely due to communicative grounding errors. We demonstrate that a "Pull-based" protocol (active querying) is significantly more robust than standard "Push-based" instruction, with successful episodes featuring 2x the frequency of clarification requests. This research isolates the mechanism of active uncertainty reduction as a prerequisite for safe human-AI and robot-robot collaboration. 

---
# Prompt-to-Parts: Generative AI for Physical Assembly and Scalable Instructions 

**Authors**: David Noever  

**Link**: [PDF](https://arxiv.org/pdf/2512.15743)  

**Abstract**: We present a framework for generating physically realizable assembly instructions from natural language descriptions. Unlike unconstrained text-to-3D approaches, our method operates within a discrete parts vocabulary, enforcing geometric validity, connection constraints, and buildability ordering. Using LDraw as a text-rich intermediate representation, we demonstrate that large language models can be guided with tools to produce valid step-by-step construction sequences and assembly instructions for brick-based prototypes of more than 3000 assembly parts. We introduce a Python library for programmatic model generation and evaluate buildable outputs on complex satellites, aircraft, and architectural domains. The approach aims for demonstrable scalability, modularity, and fidelity that bridges the gap between semantic design intent and manufacturable output. Physical prototyping follows from natural language specifications. The work proposes a novel elemental lingua franca as a key missing piece from the previous pixel-based diffusion methods or computer-aided design (CAD) models that fail to support complex assembly instructions or component exchange. Across four original designs, this novel "bag of bricks" method thus functions as a physical API: a constrained vocabulary connecting precisely oriented brick locations to a "bag of words" through which arbitrary functional requirements compile into material reality. Given such a consistent and repeatable AI representation opens new design options while guiding natural language implementations in manufacturing and engineering prototyping. 

---
# The Principle of Proportional Duty: A Knowledge-Duty Framework for Ethical Equilibrium in Human and Artificial Systems 

**Authors**: Timothy Prescher  

**Link**: [PDF](https://arxiv.org/pdf/2512.15740)  

**Abstract**: Traditional ethical frameworks often struggle to model decision-making under uncertainty, treating it as a simple constraint on action. This paper introduces the Principle of Proportional Duty (PPD), a novel framework that models how ethical responsibility scales with an agent's epistemic state. The framework reveals that moral duty is not lost to uncertainty but transforms: as uncertainty increases, Action Duty (the duty to act decisively) is proportionally converted into Repair Duty (the active duty to verify, inquire, and resolve uncertainty).
This dynamic is expressed by the equation D_total = K[(1-HI) + HI * g(C_signal)], where Total Duty is a function of Knowledge (K), Humility/Uncertainty (HI), and Contextual Signal Strength (C_signal). Monte Carlo simulations demonstrate that systems maintaining a baseline humility coefficient (lambda > 0) produce more stable duty allocations and reduce the risk of overconfident decision-making.
By formalizing humility as a system parameter, the PPD offers a mathematically tractable approach to moral responsibility that could inform the development of auditable AI decision systems. This paper applies the framework across four domains, clinical ethics, recipient-rights law, economic governance, and artificial intelligence, to demonstrate its cross-disciplinary validity. The findings suggest that proportional duty serves as a stabilizing principle within complex systems, preventing both overreach and omission by dynamically balancing epistemic confidence against contextual risk. 

---
# Anubuddhi: A Multi-Agent AI System for Designing and Simulating Quantum Optics Experiments 

**Authors**: S. K. Rithvik  

**Link**: [PDF](https://arxiv.org/pdf/2512.15736)  

**Abstract**: We present Anubuddhi, a multi-agent AI system that designs and simulates quantum optics experiments from natural language prompts without requiring specialized programming knowledge. The system composes optical layouts by arranging components from a three-tier toolbox via semantic retrieval, then validates designs through physics simulation with convergent refinement. The architecture combines intent routing, knowledge-augmented generation, and dual-mode validation (QuTiP and FreeSim). We evaluated 13 experiments spanning fundamental optics (Hong-Ou-Mandel interference, Michelson/Mach-Zehnder interferometry, Bell states, delayed-choice quantum eraser), quantum information protocols (BB84 QKD, Franson interferometry, GHZ states, quantum teleportation, hyperentanglement), and advanced technologies (boson sampling, electromagnetically induced transparency, frequency conversion). The system achieves design-simulation alignment scores of 8--9/10, with simulations faithfully modeling intended physics. A critical finding distinguishes structural correctness from quantitative accuracy: high alignment confirms correct physics architecture, while numerical predictions require expert review. Free-form simulation outperformed constrained frameworks for 11/13 experiments, revealing that quantum optics diversity demands flexible mathematical representations. The system democratizes computational experiment design for research and pedagogy, producing strong initial designs users can iteratively refine through conversation. 

---
# Differences That Matter: Auditing Models for Capability Gap Discovery and Rectification 

**Authors**: Qihao Liu, Chengzhi Mao, Yaojie Liu, Alan Yuille, Wen-Sheng Chu  

**Link**: [PDF](https://arxiv.org/pdf/2512.16921)  

**Abstract**: Conventional evaluation methods for multimodal LLMs (MLLMs) lack interpretability and are often insufficient to fully disclose significant capability gaps across models. To address this, we introduce AuditDM, an automated framework that actively discovers and rectifies MLLM failure modes by auditing their divergence. AuditDM fine-tunes an MLLM as an auditor via reinforcement learning to generate challenging questions and counterfactual images that maximize disagreement among target models. Once trained, the auditor uncovers diverse, interpretable exemplars that reveal model weaknesses and serve as annotation-free data for rectification. When applied to SoTA models like Gemma-3 and PaliGemma-2, AuditDM discovers more than 20 distinct failure types. Fine-tuning on these discoveries consistently improves all models across 16 benchmarks, and enables a 3B model to surpass its 28B counterpart. Our results suggest that as data scaling hits diminishing returns, targeted model auditing offers an effective path to model diagnosis and improvement. 

---
# EasyV2V: A High-quality Instruction-based Video Editing Framework 

**Authors**: Jinjie Mai, Chaoyang Wang, Guocheng Gordon Qian, Willi Menapace, Sergey Tulyakov, Bernard Ghanem, Peter Wonka, Ashkan Mirzaei  

**Link**: [PDF](https://arxiv.org/pdf/2512.16920)  

**Abstract**: While image editing has advanced rapidly, video editing remains less explored, facing challenges in consistency, control, and generalization. We study the design space of data, architecture, and control, and introduce \emph{EasyV2V}, a simple and effective framework for instruction-based video editing. On the data side, we compose existing experts with fast inverses to build diverse video pairs, lift image edit pairs into videos via single-frame supervision and pseudo pairs with shared affine motion, mine dense-captioned clips for video pairs, and add transition supervision to teach how edits unfold. On the model side, we observe that pretrained text-to-video models possess editing capability, motivating a simplified design. Simple sequence concatenation for conditioning with light LoRA fine-tuning suffices to train a strong model. For control, we unify spatiotemporal control via a single mask mechanism and support optional reference images. Overall, EasyV2V works with flexible inputs, e.g., video+text, video+mask+text, video+mask+reference+text, and achieves state-of-the-art video editing results, surpassing concurrent and commercial systems. Project page: this https URL 

---
# DVGT: Driving Visual Geometry Transformer 

**Authors**: Sicheng Zuo, Zixun Xie, Wenzhao Zheng, Shaoqing Xu, Fang Li, Shengyin Jiang, Long Chen, Zhi-Xin Yang, Jiwen Lu  

**Link**: [PDF](https://arxiv.org/pdf/2512.16919)  

**Abstract**: Perceiving and reconstructing 3D scene geometry from visual inputs is crucial for autonomous driving. However, there still lacks a driving-targeted dense geometry perception model that can adapt to different scenarios and camera configurations. To bridge this gap, we propose a Driving Visual Geometry Transformer (DVGT), which reconstructs a global dense 3D point map from a sequence of unposed multi-view visual inputs. We first extract visual features for each image using a DINO backbone, and employ alternating intra-view local attention, cross-view spatial attention, and cross-frame temporal attention to infer geometric relations across images. We then use multiple heads to decode a global point map in the ego coordinate of the first frame and the ego poses for each frame. Unlike conventional methods that rely on precise camera parameters, DVGT is free of explicit 3D geometric priors, enabling flexible processing of arbitrary camera configurations. DVGT directly predicts metric-scaled geometry from image sequences, eliminating the need for post-alignment with external sensors. Trained on a large mixture of driving datasets including nuScenes, OpenScene, Waymo, KITTI, and DDAD, DVGT significantly outperforms existing models on various scenarios. Code is available at this https URL. 

---
# Exploration v.s. Exploitation: Rethinking RLVR through Clipping, Entropy, and Spurious Reward 

**Authors**: Peter Chen, Xiaopeng Li, Ziniu Li, Wotao Yin, Xi Chen, Tianyi Lin  

**Link**: [PDF](https://arxiv.org/pdf/2512.16912)  

**Abstract**: This paper examines the exploration-exploitation trade-off in reinforcement learning with verifiable rewards (RLVR), a framework for improving the reasoning of Large Language Models (LLMs). Recent studies suggest that RLVR can elicit strong mathematical reasoning in LLMs through two seemingly paradoxical mechanisms: spurious rewards, which suppress exploitation by rewarding outcomes unrelated to the ground truth, and entropy minimization, which suppresses exploration by pushing the model toward more confident and deterministic outputs, highlighting a puzzling dynamic: both discouraging exploitation and discouraging exploration improve reasoning performance, yet the underlying principles that reconcile these effects remain poorly understood. We focus on two fundamental questions: (i) how policy entropy relates to performance, and (ii) whether spurious rewards yield gains, potentially through the interplay of clipping bias and model contamination. Our results show that clipping bias under spurious rewards reduces policy entropy, leading to more confident and deterministic outputs, while entropy minimization alone is insufficient for improvement. We further propose a reward-misalignment model explaining why spurious rewards can enhance performance beyond contaminated settings. Our findings clarify the mechanisms behind spurious-reward benefits and provide principles for more effective RLVR training. 

---
# Posterior Behavioral Cloning: Pretraining BC Policies for Efficient RL Finetuning 

**Authors**: Andrew Wagenmaker, Perry Dong, Raymond Tsao, Chelsea Finn, Sergey Levine  

**Link**: [PDF](https://arxiv.org/pdf/2512.16911)  

**Abstract**: Standard practice across domains from robotics to language is to first pretrain a policy on a large-scale demonstration dataset, and then finetune this policy, typically with reinforcement learning (RL), in order to improve performance on deployment domains. This finetuning step has proved critical in achieving human or super-human performance, yet while much attention has been given to developing more effective finetuning algorithms, little attention has been given to ensuring the pretrained policy is an effective initialization for RL finetuning. In this work we seek to understand how the pretrained policy affects finetuning performance, and how to pretrain policies in order to ensure they are effective initializations for finetuning. We first show theoretically that standard behavioral cloning (BC) -- which trains a policy to directly match the actions played by the demonstrator -- can fail to ensure coverage over the demonstrator's actions, a minimal condition necessary for effective RL finetuning. We then show that if, instead of exactly fitting the observed demonstrations, we train a policy to model the posterior distribution of the demonstrator's behavior given the demonstration dataset, we do obtain a policy that ensures coverage over the demonstrator's actions, enabling more effective finetuning. Furthermore, this policy -- which we refer to as the posterior behavioral cloning (PostBC) policy -- achieves this while ensuring pretrained performance is no worse than that of the BC policy. We then show that PostBC is practically implementable with modern generative models in robotic control domains -- relying only on standard supervised learning -- and leads to significantly improved RL finetuning performance on both realistic robotic control benchmarks and real-world robotic manipulation tasks, as compared to standard behavioral cloning. 

---
# Flowing from Reasoning to Motion: Learning 3D Hand Trajectory Prediction from Egocentric Human Interaction Videos 

**Authors**: Mingfei Chen, Yifan Wang, Zhengqin Li, Homanga Bharadhwaj, Yujin Chen, Chuan Qin, Ziyi Kou, Yuan Tian, Eric Whitmire, Rajinder Sodhi, Hrvoje Benko, Eli Shlizerman, Yue Liu  

**Link**: [PDF](https://arxiv.org/pdf/2512.16907)  

**Abstract**: Prior works on 3D hand trajectory prediction are constrained by datasets that decouple motion from semantic supervision and by models that weakly link reasoning and action. To address these, we first present the EgoMAN dataset, a large-scale egocentric dataset for interaction stage-aware 3D hand trajectory prediction with 219K 6DoF trajectories and 3M structured QA pairs for semantic, spatial, and motion reasoning. We then introduce the EgoMAN model, a reasoning-to-motion framework that links vision-language reasoning and motion generation via a trajectory-token interface. Trained progressively to align reasoning with motion dynamics, our approach yields accurate and stage-aware trajectories with generalization across real-world scenes. 

---
# Impacts of Racial Bias in Historical Training Data for News AI 

**Authors**: Rahul Bhargava, Malene Hornstrup Jespersen, Emily Boardman Ndulue, Vivica Dsouza  

**Link**: [PDF](https://arxiv.org/pdf/2512.16901)  

**Abstract**: AI technologies have rapidly moved into business and research applications that involve large text corpora, including computational journalism research and newsroom settings. These models, trained on extant data from various sources, can be conceptualized as historical artifacts that encode decades-old attitudes and stereotypes. This paper investigates one such example trained on the broadly-used New York Times Annotated Corpus to create a multi-label classifier. Our use in research settings surfaced the concerning "blacks" thematic topic label. Through quantitative and qualitative means we investigate this label's use in the training corpus, what concepts it might be encoding in the trained classifier, and how those concepts impact our model use. Via the application of explainable AI methods, we find that the "blacks" label operates partially as a general "racism detector" across some minoritized groups. However, it performs poorly against expectations on modern examples such as COVID-19 era anti-Asian hate stories, and reporting on the Black Lives Matter movement. This case study of interrogating embedded biases in a model reveals how similar applications in newsroom settings can lead to unexpected outputs that could impact a wide variety of potential uses of any large language model-story discovery, audience targeting, summarization, etc. The fundamental tension this exposes for newsrooms is how to adopt AI-enabled workflow tools while reducing the risk of reproducing historical biases in news coverage. 

---
# LinkedOut: Linking World Knowledge Representation Out of Video LLM for Next-Generation Video Recommendation 

**Authors**: Haichao Zhang, Yao Lu, Lichen Wang, Yunzhe Li, Daiwei Chen, Yunpeng Xu, Yun Fu  

**Link**: [PDF](https://arxiv.org/pdf/2512.16891)  

**Abstract**: Video Large Language Models (VLLMs) unlock world-knowledge-aware video understanding through pretraining on internet-scale data and have already shown promise on tasks such as movie analysis and video question answering. However, deploying VLLMs for downstream tasks such as video recommendation remains challenging, since real systems require multi-video inputs, lightweight backbones, low-latency sequential inference, and rapid response. In practice, (1) decode-only generation yields high latency for sequential inference, (2) typical interfaces do not support multi-video inputs, and (3) constraining outputs to language discards fine-grained visual details that matter for downstream vision tasks. We argue that these limitations stem from the absence of a representation that preserves pixel-level detail while leveraging world knowledge. We present LinkedOut, a representation that extracts VLLM world knowledge directly from video to enable fast inference, supports multi-video histories, and removes the language bottleneck. LinkedOut extracts semantically grounded, knowledge-aware tokens from raw frames using VLLMs, guided by promptable queries and optional auxiliary modalities. We introduce a cross-layer knowledge fusion MoE that selects the appropriate level of abstraction from the rich VLLM features, enabling personalized, interpretable, and low-latency recommendation. To our knowledge, LinkedOut is the first VLLM-based video recommendation method that operates on raw frames without handcrafted labels, achieving state-of-the-art results on standard benchmarks. Interpretability studies and ablations confirm the benefits of layer diversity and layer-wise fusion, pointing to a practical path that fully leverages VLLM world-knowledge priors and visual reasoning for downstream vision tasks such as recommendation. 

---
# Training Together, Diagnosing Better: Federated Learning for Collagen VI-Related Dystrophies 

**Authors**: Astrid Brull, Sara Aguti, Véronique Bolduc, Ying Hu, Daniel M. Jimenez-Gutierrez, Enrique Zuazua, Joaquin Del-Rio, Oleksii Sliusarenko, Haiyan Zhou, Francesco Muntoni, Carsten G. Bönnemann, Xabi Uribe-Etxebarria  

**Link**: [PDF](https://arxiv.org/pdf/2512.16876)  

**Abstract**: The application of Machine Learning (ML) to the diagnosis of rare diseases, such as collagen VI-related dystrophies (COL6-RD), is fundamentally limited by the scarcity and fragmentation of available data. Attempts to expand sampling across hospitals, institutions, or countries with differing regulations face severe privacy, regulatory, and logistical obstacles that are often difficult to overcome. The Federated Learning (FL) provides a promising solution by enabling collaborative model training across decentralized datasets while keeping patient data local and private. Here, we report a novel global FL initiative using the this http URL FL platform, which leverages FL across distributed datasets in two international organizations for the diagnosis of COL6-RD, using collagen VI immunofluorescence microscopy images from patient-derived fibroblast cultures. Our solution resulted in an ML model capable of classifying collagen VI patient images into the three primary pathogenic mechanism groups associated with COL6-RD: exon skipping, glycine substitution, and pseudoexon insertion. This new approach achieved an F1-score of 0.82, outperforming single-organization models (0.57-0.75). These results demonstrate that FL substantially improves diagnostic utility and generalizability compared to isolated institutional models. Beyond enabling more accurate diagnosis, we anticipate that this approach will support the interpretation of variants of uncertain significance and guide the prioritization of sequencing strategies to identify novel pathogenic variants. 

---
# Pixel Seal: Adversarial-only training for invisible image and video watermarking 

**Authors**: Tomáš Souček, Pierre Fernandez, Hady Elsahar, Sylvestre-Alvise Rebuffi, Valeriu Lacatusu, Tuan Tran, Tom Sander, Alexandre Mourachko  

**Link**: [PDF](https://arxiv.org/pdf/2512.16874)  

**Abstract**: Invisible watermarking is essential for tracing the provenance of digital content. However, training state-of-the-art models remains notoriously difficult, with current approaches often struggling to balance robustness against true imperceptibility. This work introduces Pixel Seal, which sets a new state-of-the-art for image and video watermarking. We first identify three fundamental issues of existing methods: (i) the reliance on proxy perceptual losses such as MSE and LPIPS that fail to mimic human perception and result in visible watermark artifacts; (ii) the optimization instability caused by conflicting objectives, which necessitates exhaustive hyperparameter tuning; and (iii) reduced robustness and imperceptibility of watermarks when scaling models to high-resolution images and videos. To overcome these issues, we first propose an adversarial-only training paradigm that eliminates unreliable pixel-wise imperceptibility losses. Second, we introduce a three-stage training schedule that stabilizes convergence by decoupling robustness and imperceptibility. Third, we address the resolution gap via high-resolution adaptation, employing JND-based attenuation and training-time inference simulation to eliminate upscaling artifacts. We thoroughly evaluate the robustness and imperceptibility of Pixel Seal on different image types and across a wide range of transformations, and show clear improvements over the state-of-the-art. We finally demonstrate that the model efficiently adapts to video via temporal watermark pooling, positioning Pixel Seal as a practical and scalable solution for reliable provenance in real-world image and video settings. 

---
# Sequencing to Mitigate Catastrophic Forgetting in Continual Learning 

**Authors**: Hesham G. Moussa, Aroosa Hameed, Arashmid Akhavain  

**Link**: [PDF](https://arxiv.org/pdf/2512.16871)  

**Abstract**: To cope with real-world dynamics, an intelligent system needs to incrementally acquire, update, and exploit knowledge throughout its lifetime. This ability, known as Continual learning, provides a foundation for AI systems to develop themselves adaptively. Catastrophic forgetting is a major challenge to the progress of Continual Learning approaches, where learning a new task usually results in a dramatic performance drop on previously learned ones. Many approaches have emerged to counteract the impact of CF. Most of the proposed approaches can be categorized into five classes: replay-based, regularization-based, optimization-based, representation-based, and architecture-based. In this work, we approach the problem from a different angle, specifically by considering the optimal sequencing of tasks as they are presented to the model. We investigate the role of task sequencing in mitigating CF and propose a method for determining the optimal task order. The proposed method leverages zero-shot scoring algorithms inspired by neural architecture search (NAS). Results demonstrate that intelligent task sequencing can substantially reduce CF. Moreover, when combined with traditional continual learning strategies, sequencing offers enhanced performance and robustness against forgetting. Additionally, the presented approaches can find applications in other fields, such as curriculum learning. 

---
# Semi-Supervised Online Learning on the Edge by Transforming Knowledge from Teacher Models 

**Authors**: Jiabin Xue  

**Link**: [PDF](https://arxiv.org/pdf/2512.16866)  

**Abstract**: Edge machine learning (Edge ML) enables training ML models using the vast data distributed across network edges. However, many existing approaches assume static models trained centrally and then deployed, making them ineffective against unseen data. To address this, Online Edge ML allows models to be trained directly on edge devices and updated continuously with new data. This paper explores a key challenge of Online Edge ML: "How to determine labels for truly future, unseen data points". We propose Knowledge Transformation (KT), a hybrid method combining Knowledge Distillation, Active Learning, and causal reasoning. In short, KT acts as the oracle in active learning by transforming knowledge from a teacher model to generate pseudo-labels for training a student model. To verify the validity of the method, we conducted simulation experiments with two setups: (1) using a less stable teacher model and (2) a relatively more stable teacher model. Results indicate that when a stable teacher model is given, the student model can eventually reach its expected maximum performance. KT is potentially beneficial for scenarios that meet the following circumstances: (1) when the teacher's task is generic, which means existing pre-trained models might be adequate for its task, so there will be no need to train the teacher model from scratch; and/or (2) when the label for the student's task is difficult or expensive to acquire. 

---
# ReinforceGen: Hybrid Skill Policies with Automated Data Generation and Reinforcement Learning 

**Authors**: Zihan Zhou, Animesh Garg, Ajay Mandlekar, Caelan Garrett  

**Link**: [PDF](https://arxiv.org/pdf/2512.16861)  

**Abstract**: Long-horizon manipulation has been a long-standing challenge in the robotics community. We propose ReinforceGen, a system that combines task decomposition, data generation, imitation learning, and motion planning to form an initial solution, and improves each component through reinforcement-learning-based fine-tuning. ReinforceGen first segments the task into multiple localized skills, which are connected through motion planning. The skills and motion planning targets are trained with imitation learning on a dataset generated from 10 human demonstrations, and then fine-tuned through online adaptation and reinforcement learning. When benchmarked on the Robosuite dataset, ReinforceGen reaches 80% success rate on all tasks with visuomotor controls in the highest reset range setting. Additional ablation studies show that our fine-tuning approaches contributes to an 89% average performance increase. More results and videos available in this https URL 

---
# GenEval 2: Addressing Benchmark Drift in Text-to-Image Evaluation 

**Authors**: Amita Kamath, Kai-Wei Chang, Ranjay Krishna, Luke Zettlemoyer, Yushi Hu, Marjan Ghazvininejad  

**Link**: [PDF](https://arxiv.org/pdf/2512.16853)  

**Abstract**: Automating Text-to-Image (T2I) model evaluation is challenging; a judge model must be used to score correctness, and test prompts must be selected to be challenging for current T2I models but not the judge. We argue that satisfying these constraints can lead to benchmark drift over time, where the static benchmark judges fail to keep up with newer model capabilities. We show that benchmark drift is a significant problem for GenEval, one of the most popular T2I benchmarks. Although GenEval was well-aligned with human judgment at the time of its release, it has drifted far from human judgment over time -- resulting in an absolute error of as much as 17.7% for current models. This level of drift strongly suggests that GenEval has been saturated for some time, as we verify via a large-scale human study. To help fill this benchmarking gap, we introduce a new benchmark, GenEval 2, with improved coverage of primitive visual concepts and higher degrees of compositionality, which we show is more challenging for current models. We also introduce Soft-TIFA, an evaluation method for GenEval 2 that combines judgments for visual primitives, which we show is more well-aligned with human judgment and argue is less likely to drift from human-alignment over time (as compared to more holistic judges such as VQAScore). Although we hope GenEval 2 will provide a strong benchmark for many years, avoiding benchmark drift is far from guaranteed and our work, more generally, highlights the importance of continual audits and improvement for T2I and related automated model evaluation benchmarks. 

---
# PrivateXR: Defending Privacy Attacks in Extended Reality Through Explainable AI-Guided Differential Privacy 

**Authors**: Ripan Kumar Kundu, Istiak Ahmed, Khaza Anuarul Hoque  

**Link**: [PDF](https://arxiv.org/pdf/2512.16851)  

**Abstract**: The convergence of artificial AI and XR technologies (AI XR) promises innovative applications across many domains. However, the sensitive nature of data (e.g., eye-tracking) used in these systems raises significant privacy concerns, as adversaries can exploit these data and models to infer and leak personal information through membership inference attacks (MIA) and re-identification (RDA) with a high success rate. Researchers have proposed various techniques to mitigate such privacy attacks, including differential privacy (DP). However, AI XR datasets often contain numerous features, and applying DP uniformly can introduce unnecessary noise to less relevant features, degrade model accuracy, and increase inference time, limiting real-time XR deployment. Motivated by this, we propose a novel framework combining explainable AI (XAI) and DP-enabled privacy-preserving mechanisms to defend against privacy attacks. Specifically, we leverage post-hoc explanations to identify the most influential features in AI XR models and selectively apply DP to those features during inference. We evaluate our XAI-guided DP approach on three state-of-the-art AI XR models and three datasets: cybersickness, emotion, and activity classification. Our results show that the proposed method reduces MIA and RDA success rates by up to 43% and 39%, respectively, for cybersickness tasks while preserving model utility with up to 97% accuracy using Transformer models. Furthermore, it improves inference time by up to ~2x compared to traditional DP approaches. To demonstrate practicality, we deploy the XAI-guided DP AI XR models on an HTC VIVE Pro headset and develop a user interface (UI), namely PrivateXR, allowing users to adjust privacy levels (e.g., low, medium, high) while receiving real-time task predictions, protecting user privacy during XR gameplay. 

---
# Meta-RL Induces Exploration in Language Agents 

**Authors**: Yulun Jiang, Liangze Jiang, Damien Teney, Michael Moor, Maria Brbic  

**Link**: [PDF](https://arxiv.org/pdf/2512.16848)  

**Abstract**: Reinforcement learning (RL) has enabled the training of large language model (LLM) agents to interact with the environment and to solve multi-turn long-horizon tasks. However, the RL-trained agents often struggle in tasks that require active exploration and fail to efficiently adapt from trial-and-error experiences. In this paper, we present LaMer, a general Meta-RL framework that enables LLM agents to actively explore and learn from the environment feedback at test time. LaMer consists of two key components: (i) a cross-episode training framework to encourage exploration and long-term rewards optimization; and (ii) in-context policy adaptation via reflection, allowing the agent to adapt their policy from task feedback signal without gradient update. Experiments across diverse environments show that LaMer significantly improves performance over RL baselines, with 11%, 14%, and 19% performance gains on Sokoban, MineSweeper and Webshop, respectively. Moreover, LaMer also demonstrates better generalization to more challenging or previously unseen tasks compared to the RL-trained agents. Overall, our results demonstrate that Meta-RL provides a principled approach to induce exploration in language agents, enabling more robust adaptation to novel environments through learned exploration strategies. 

---
# LLMCache: Layer-Wise Caching Strategies for Accelerated Reuse in Transformer Inference 

**Authors**: Harsh Vardhan Bansal  

**Link**: [PDF](https://arxiv.org/pdf/2512.16843)  

**Abstract**: Transformer-based language models have achieved remarkable performance across a wide range of tasks, yet their high inference latency poses a significant challenge for real-timeand large-scale deployment. While existing caching mechanisms,such as token-level key-value caches, offer speedups in autore-gressive decoding, they are limited in scope and applicability. In this paper, we present LLMCache, a novel layer-wise caching framework that accelerates transformer inference by reusing intermediate activations based on semantic similarity of input sequences. Unlike prior work, LLMCache is model-agnostic,operates across both encoder and decoder architectures, and supports caching at arbitrary transformer layers. We introduce a lightweight fingerprinting mechanism for matching seman-tically similar inputs and propose adaptive eviction strategies to manage cache staleness. Experiments on BERT and GPT-2 across SQuAD, WikiText-103, and OpenBookQA show up to 3.1 X speedup in inference time with <0.5% accuracy degradation. Our results highlight LLMCache as a practical and general-purpose solution for optimizing transformer inference in real-world applications 

---
# OPENTOUCH: Bringing Full-Hand Touch to Real-World Interaction 

**Authors**: Yuxin Ray Song, Jinzhou Li, Rao Fu, Devin Murphy, Kaichen Zhou, Rishi Shiv, Yaqi Li, Haoyu Xiong, Crystal Elaine Owens, Yilun Du, Yiyue Luo, Xianyi Cheng, Antonio Torralba, Wojciech Matusik, Paul Pu Liang  

**Link**: [PDF](https://arxiv.org/pdf/2512.16842)  

**Abstract**: The human hand is our primary interface to the physical world, yet egocentric perception rarely knows when, where, or how forcefully it makes contact. Robust wearable tactile sensors are scarce, and no existing in-the-wild datasets align first-person video with full-hand touch. To bridge the gap between visual perception and physical interaction, we present OpenTouch, the first in-the-wild egocentric full-hand tactile dataset, containing 5.1 hours of synchronized video-touch-pose data and 2,900 curated clips with detailed text annotations. Using OpenTouch, we introduce retrieval and classification benchmarks that probe how touch grounds perception and action. We show that tactile signals provide a compact yet powerful cue for grasp understanding, strengthen cross-modal alignment, and can be reliably retrieved from in-the-wild video queries. By releasing this annotated vision-touch-pose dataset and benchmark, we aim to advance multimodal egocentric perception, embodied learning, and contact-rich robotic manipulation. 

---
# Next-Generation License Plate Detection and Recognition System using YOLOv8 

**Authors**: Arslan Amin, Rafia Mumtaz, Muhammad Jawad Bashir, Syed Mohammad Hassan Zaidi  

**Link**: [PDF](https://arxiv.org/pdf/2512.16826)  

**Abstract**: In the evolving landscape of traffic management and vehicle surveillance, efficient license plate detection and recognition are indispensable. Historically, many methodologies have tackled this challenge, but consistent real-time accuracy, especially in diverse environments, remains elusive. This study examines the performance of YOLOv8 variants on License Plate Recognition (LPR) and Character Recognition tasks, crucial for advancing Intelligent Transportation Systems. Two distinct datasets were employed for training and evaluation, yielding notable findings. The YOLOv8 Nano variant demonstrated a precision of 0.964 and mAP50 of 0.918 on the LPR task, while the YOLOv8 Small variant exhibited a precision of 0.92 and mAP50 of 0.91 on the Character Recognition task. A custom method for character sequencing was introduced, effectively sequencing the detected characters based on their x-axis positions. An optimized pipeline, utilizing YOLOv8 Nano for LPR and YOLOv8 Small for Character Recognition, is proposed. This configuration not only maintains computational efficiency but also ensures high accuracy, establishing a robust foundation for future real-world deployments on edge devices within Intelligent Transportation Systems. This effort marks a significant stride towards the development of smarter and more efficient urban infrastructures. 

---
# Grammar-Forced Translation of Natural Language to Temporal Logic using LLMs 

**Authors**: William English, Dominic Simon, Sumit Kumar Jha, Rickard Ewetz  

**Link**: [PDF](https://arxiv.org/pdf/2512.16814)  

**Abstract**: Translating natural language (NL) into a formal language such as temporal logic (TL) is integral for human communication with robots and autonomous systems. State-of-the-art approaches decompose the task into a lifting of atomic propositions (APs) phase and a translation phase. However, existing methods struggle with accurate lifting, the existence of co-references, and learning from limited data. In this paper, we propose a framework for NL to TL translation called Grammar Forced Translation (GraFT). The framework is based on the observation that previous work solves both the lifting and translation steps by letting a language model iteratively predict tokens from its full vocabulary. In contrast, GraFT reduces the complexity of both tasks by restricting the set of valid output tokens from the full vocabulary to only a handful in each step. The solution space reduction is obtained by exploiting the unique properties of each problem. We also provide a theoretical justification for why the solution space reduction leads to more efficient learning. We evaluate the effectiveness of GraFT using the CW, GLTL, and Navi benchmarks. Compared with state-of-the-art translation approaches, it can be observed that GraFT the end-to-end translation accuracy by 5.49% and out-of-domain translation accuracy by 14.06% on average. 

---
# Coordinated Anti-Jamming Resilience in Swarm Networks via Multi-Agent Reinforcement Learning 

**Authors**: Bahman Abolhassani, Tugba Erpek, Kemal Davaslioglu, Yalin E. Sagduyu, Sastry Kompella  

**Link**: [PDF](https://arxiv.org/pdf/2512.16813)  

**Abstract**: Reactive jammers pose a severe security threat to robotic-swarm networks by selectively disrupting inter-agent communications and undermining formation integrity and mission success. Conventional countermeasures such as fixed power control or static channel hopping are largely ineffective against such adaptive adversaries. This paper presents a multi-agent reinforcement learning (MARL) framework based on the QMIX algorithm to improve the resilience of swarm communications under reactive jamming. We consider a network of multiple transmitter-receiver pairs sharing channels while a reactive jammer with Markovian threshold dynamics senses aggregate power and reacts accordingly. Each agent jointly selects transmit frequency (channel) and power, and QMIX learns a centralized but factorizable action-value function that enables coordinated yet decentralized execution. We benchmark QMIX against a genie-aided optimal policy in a no-channel-reuse setting, and against local Upper Confidence Bound (UCB) and a stateless reactive policy in a more general fading regime with channel reuse enabled. Simulation results show that QMIX rapidly converges to cooperative policies that nearly match the genie-aided bound, while achieving higher throughput and lower jamming incidence than the baselines, thereby demonstrating MARL's effectiveness for securing autonomous swarms in contested environments. 

---
# From Facts to Conclusions : Integrating Deductive Reasoning in Retrieval-Augmented LLMs 

**Authors**: Shubham Mishra, Samyek Jain, Gorang Mehrishi, Shiv Tiwari, Harsh Sharma, Pratik Narang, Dhruv Kumar  

**Link**: [PDF](https://arxiv.org/pdf/2512.16795)  

**Abstract**: Retrieval-Augmented Generation (RAG) grounds large language models (LLMs) in external evidence, but fails when retrieved sources conflict or contain outdated or subjective information. Prior work address these issues independently but lack unified reasoning supervision. We propose a reasoning-trace-augmented RAG framework that adds structured, interpretable reasoning across three stages : (1) document-level adjudication, (2) conflict analysis, and (3) grounded synthesis, producing citation-linked answers or justified refusals. A Conflict-Aware Trust-Score (CATS) pipeline is introduced which evaluates groundedness, factual correctness, refusal accuracy, and conflict-behavior alignment using an LLM-as-a-Judge. Our 539-query reasoning dataset and evaluation pipeline establish a foundation for conflict-aware, interpretable RAG systems. Experimental results demonstrate substantial gains over baselines, most notably with Qwen, where Supervised Fine-Tuning improved End-to-End answer correctness from 0.069 to 0.883 and behavioral adherence from 0.074 to 0.722. 

---
# Delay-Aware Multi-Stage Edge Server Upgrade with Budget Constraint 

**Authors**: Endar Suprih Wihidayat, Sieteng Soh, Kwan-Wu Chin, Duc-son Pham  

**Link**: [PDF](https://arxiv.org/pdf/2512.16792)  

**Abstract**: In this paper, the Multi-stage Edge Server Upgrade (M-ESU) is proposed as a new network planning problem, involving the upgrading of an existing multi-access edge computing (MEC) system through multiple stages (e.g., over several years). More precisely, the problem considers two key decisions: (i) whether to deploy additional edge servers or upgrade those already installed, and (ii) how tasks should be offloaded so that the average number of tasks that meet their delay requirement is maximized. The framework specifically involves: (i) deployment of new servers combined with capacity upgrades for existing servers, and (ii) the optimal task offloading to maximize the average number of tasks with a delay requirement. It also considers the following constraints: (i) budget per stage, (ii) server deployment and upgrade cost (in $) and cost depreciation rate, (iii) computation resource of servers, (iv) number of tasks and their growth rate (in %), and (v) the increase in task sizes and stricter delay requirements over time. We present two solutions: a Mixed Integer Linear Programming (MILP) model and an efficient heuristic algorithm (M-ESU/H). MILP yields the optimal solution for small networks, whereas M-ESU/H is used in large-scale networks. For small networks, the simulation results show that the solution computed by M-ESU/H is within 1.25% of the optimal solution while running several orders of magnitude faster. For large networks, M-ESU/H is compared against three alternative heuristic solutions that consider only server deployment, or giving priority to server deployment or upgrade. Our experiments show that M-ESU/H yields up to 21.57% improvement in task satisfaction under identical budget and demand growth conditions, confirming its scalability and practical value for long-term MEC systems. 

---
# KineST: A Kinematics-guided Spatiotemporal State Space Model for Human Motion Tracking from Sparse Signals 

**Authors**: Shuting Zhao, Zeyu Xiao, Xinrong Chen  

**Link**: [PDF](https://arxiv.org/pdf/2512.16791)  

**Abstract**: Full-body motion tracking plays an essential role in AR/VR applications, bridging physical and virtual interactions. However, it is challenging to reconstruct realistic and diverse full-body poses based on sparse signals obtained by head-mounted displays, which are the main devices in AR/VR scenarios. Existing methods for pose reconstruction often incur high computational costs or rely on separately modeling spatial and temporal dependencies, making it difficult to balance accuracy, temporal coherence, and efficiency. To address this problem, we propose KineST, a novel kinematics-guided state space model, which effectively extracts spatiotemporal dependencies while integrating local and global pose perception. The innovation comes from two core ideas. Firstly, in order to better capture intricate joint relationships, the scanning strategy within the State Space Duality framework is reformulated into kinematics-guided bidirectional scanning, which embeds kinematic priors. Secondly, a mixed spatiotemporal representation learning approach is employed to tightly couple spatial and temporal contexts, balancing accuracy and smoothness. Additionally, a geometric angular velocity loss is introduced to impose physically meaningful constraints on rotational variations for further improving motion stability. Extensive experiments demonstrate that KineST has superior performance in both accuracy and temporal consistency within a lightweight framework. Project page: this https URL 

---
# Towards Mass Spectrum Analysis with ASP 

**Authors**: Nils Küchenmeister, Alex Ivliev, Markus Krötzsch  

**Link**: [PDF](https://arxiv.org/pdf/2512.16780)  

**Abstract**: We present a new use of Answer Set Programming (ASP) to discover the molecular structure of chemical samples based on the relative abundance of elements and structural fragments, as measured in mass spectrometry. To constrain the exponential search space for this combinatorial problem, we develop canonical representations of molecular structures and an ASP implemen- tation that uses these definitions. We evaluate the correctness of our implementation over a large set of known molecular structures, and we compare its quality and performance to other ASP symmetry-breaking methods and to a commercial tool from analytical chemistry. Under consideration in Theory and Practice of Logic Programming (TPLP). 

---
# GinSign: Grounding Natural Language Into System Signatures for Temporal Logic Translation 

**Authors**: William English, Chase Walker, Dominic Simon, Rickard Ewetz  

**Link**: [PDF](https://arxiv.org/pdf/2512.16770)  

**Abstract**: Natural language (NL) to temporal logic (TL) translation enables engineers to specify, verify, and enforce system behaviors without manually crafting formal specifications-an essential capability for building trustworthy autonomous systems. While existing NL-to-TL translation frameworks have demonstrated encouraging initial results, these systems either explicitly assume access to accurate atom grounding or suffer from low grounded translation accuracy. In this paper, we propose a framework for Grounding Natural Language Into System Signatures for Temporal Logic translation called GinSign. The framework introduces a grounding model that learns the abstract task of mapping NL spans onto a given system signature: given a lifted NL specification and a system signature $\mathcal{S}$, the classifier must assign each lifted atomic proposition to an element of the set of signature-defined atoms $\mathcal{P}$. We decompose the grounding task hierarchically- first predicting predicate labels, then selecting the appropriately typed constant arguments. Decomposing this task from a free-form generation problem into a structured classification problem permits the use of smaller masked language models and eliminates the reliance on expensive LLMs. Experiments across multiple domains show that frameworks which omit grounding tend to produce syntactically correct lifted LTL that is semantically nonequivalent to grounded target expressions, whereas our framework supports downstream model checking and achieves grounded logical-equivalence scores of $95.5\%$, a $1.4\times$ improvement over SOTA. 

---
# Plausibility as Failure: How LLMs and Humans Co-Construct Epistemic Error 

**Authors**: Claudia Vale Oliveira, Nelson Zagalo, Filipe Silva, Anabela Brandao, Syeda Faryal Hussain Khurrum, Joaquim Santos  

**Link**: [PDF](https://arxiv.org/pdf/2512.16750)  

**Abstract**: Large language models (LLMs) are increasingly used as epistemic partners in everyday reasoning, yet their errors remain predominantly analyzed through predictive metrics rather than through their interpretive effects on human judgment. This study examines how different forms of epistemic failure emerge, are masked, and are tolerated in human AI interaction, where failure is understood as a relational breakdown shaped by model-generated plausibility and human interpretive judgment. We conducted a three round, multi LLM evaluation using interdisciplinary tasks and progressively differentiated assessment frameworks to observe how evaluators interpret model responses across linguistic, epistemic, and credibility dimensions. Our findings show that LLM errors shift from predictive to hermeneutic forms, where linguistic fluency, structural coherence, and superficially plausible citations conceal deeper distortions of meaning. Evaluators frequently conflated criteria such as correctness, relevance, bias, groundedness, and consistency, indicating that human judgment collapses analytical distinctions into intuitive heuristics shaped by form and fluency. Across rounds, we observed a systematic verification burden and cognitive drift. As tasks became denser, evaluators increasingly relied on surface cues, allowing erroneous yet well formed answers to pass as credible. These results suggest that error is not solely a property of model behavior but a co-constructed outcome of generative plausibility and human interpretive shortcuts. Understanding AI epistemic failure therefore requires reframing evaluation as a relational interpretive process, where the boundary between system failure and human miscalibration becomes porous. The study provides implications for LLM assessment, digital literacy, and the design of trustworthy human AI communication. 

---
# TreeNet: A Light Weight Model for Low Bitrate Image Compression 

**Authors**: Mahadev Prasad Panda, Purnachandra Rao Makkena, Srivatsa Prativadibhayankaram, Siegfried Fößel, André Kaup  

**Link**: [PDF](https://arxiv.org/pdf/2512.16743)  

**Abstract**: Reducing computational complexity remains a critical challenge for the widespread adoption of learning-based image compression techniques. In this work, we propose TreeNet, a novel low-complexity image compression model that leverages a binary tree-structured encoder-decoder architecture to achieve efficient representation and reconstruction. We employ attentional feature fusion mechanism to effectively integrate features from multiple branches. We evaluate TreeNet on three widely used benchmark datasets and compare its performance against competing methods including JPEG AI, a recent standard in learning-based image compression. At low bitrates, TreeNet achieves an average improvement of 4.83% in BD-rate over JPEG AI, while reducing model complexity by 87.82%. Furthermore, we conduct extensive ablation studies to investigate the influence of various latent representations within TreeNet, offering deeper insights into the factors contributing to reconstruction. 

---
# Towards Reproducibility in Predictive Process Mining: SPICE - A Deep Learning Library 

**Authors**: Oliver Stritzel, Nick Hühnerbein, Simon Rauch, Itzel Zarate, Lukas Fleischmann, Moike Buck, Attila Lischka, Christian Frey  

**Link**: [PDF](https://arxiv.org/pdf/2512.16715)  

**Abstract**: In recent years, Predictive Process Mining (PPM) techniques based on artificial neural networks have evolved as a method for monitoring the future behavior of unfolding business processes and predicting Key Performance Indicators (KPIs). However, many PPM approaches often lack reproducibility, transparency in decision making, usability for incorporating novel datasets and benchmarking, making comparisons among different implementations very difficult. In this paper, we propose SPICE, a Python framework that reimplements three popular, existing baseline deep-learning-based methods for PPM in PyTorch, while designing a common base framework with rigorous configurability to enable reproducible and robust comparison of past and future modelling approaches. We compare SPICE to original reported metrics and with fair metrics on 11 datasets. 

---
# Few-Shot Fingerprinting Subject Re-Identification in 3D-MRI and 2D-X-Ray 

**Authors**: Gonçalo Gaspar Alves, Shekoufeh Gorgi Zadeh, Andreas Husch, Ben Bausch  

**Link**: [PDF](https://arxiv.org/pdf/2512.16685)  

**Abstract**: Combining open-source datasets can introduce data leakage if the same subject appears in multiple sets, leading to inflated model performance. To address this, we explore subject fingerprinting, mapping all images of a subject to a distinct region in latent space, to enable subject re-identification via similarity matching. Using a ResNet-50 trained with triplet margin loss, we evaluate few-shot fingerprinting on 3D MRI and 2D X-ray data in both standard (20-way 1-shot) and challenging (1000-way 1-shot) scenarios. The model achieves high Mean- Recall-@-K scores: 99.10% (20-way 1-shot) and 90.06% (500-way 5-shot) on ChestXray-14; 99.20% (20-way 1-shot) and 98.86% (100-way 3-shot) on BraTS- 2021. 

---
# Microsoft Academic Graph Information Retrieval for Research Recommendation and Assistance 

**Authors**: Jacob Reiss, Shikshya Shiwakoti, Samuel Goldsmith, Ujjwal Pandit  

**Link**: [PDF](https://arxiv.org/pdf/2512.16661)  

**Abstract**: In today's information-driven world, access to scientific publications has become increasingly easy. At the same time, filtering through the massive volume of available research has become more challenging than ever. Graph Neural Networks (GNNs) and graph attention mechanisms have shown strong effectiveness in searching large-scale information databases, particularly when combined with modern large language models. In this paper, we propose an Attention-Based Subgraph Retriever, a GNN-as-retriever model that applies attention-based pruning to extract a refined subgraph, which is then passed to a large language model for advanced knowledge reasoning. 

---
# Protecting Deep Neural Network Intellectual Property with Chaos-Based White-Box Watermarking 

**Authors**: Sangeeth B, Serena Nicolazzo, Deepa K., Vinod P  

**Link**: [PDF](https://arxiv.org/pdf/2512.16658)  

**Abstract**: The rapid proliferation of deep neural networks (DNNs) across several domains has led to increasing concerns regarding intellectual property (IP) protection and model misuse. Trained DNNs represent valuable assets, often developed through significant investments. However, the ease with which models can be copied, redistributed, or repurposed highlights the urgent need for effective mechanisms to assert and verify model ownership. In this work, we propose an efficient and resilient white-box watermarking framework that embeds ownership information into the internal parameters of a DNN using chaotic sequences. The watermark is generated using a logistic map, a well-known chaotic function, producing a sequence that is sensitive to its initialization parameters. This sequence is injected into the weights of a chosen intermediate layer without requiring structural modifications to the model or degradation in predictive performance. To validate ownership, we introduce a verification process based on a genetic algorithm that recovers the original chaotic parameters by optimizing the similarity between the extracted and regenerated sequences. The effectiveness of the proposed approach is demonstrated through extensive experiments on image classification tasks using MNIST and CIFAR-10 datasets. The results show that the embedded watermark remains detectable after fine-tuning, with negligible loss in model accuracy. In addition to numerical recovery of the watermark, we perform visual analyses using weight density plots and construct activation-based classifiers to distinguish between original, watermarked, and tampered models. Overall, the proposed method offers a flexible and scalable solution for embedding and verifying model ownership in white-box settings well-suited for real-world scenarios where IP protection is critical. 

---
# Stackelberg Learning from Human Feedback: Preference Optimization as a Sequential Game 

**Authors**: Barna Pásztor, Thomas Kleine Buening, Andreas Krause  

**Link**: [PDF](https://arxiv.org/pdf/2512.16626)  

**Abstract**: We introduce Stackelberg Learning from Human Feedback (SLHF), a new framework for preference optimization. SLHF frames the alignment problem as a sequential-move game between two policies: a Leader, which commits to an action, and a Follower, which responds conditionally on the Leader's action. This approach decomposes preference optimization into a refinement problem for the Follower and an optimization problem against an adversary for the Leader. Unlike Reinforcement Learning from Human Feedback (RLHF), which assigns scalar rewards to actions, or Nash Learning from Human Feedback (NLHF), which seeks a simultaneous-move equilibrium, SLHF leverages the asymmetry of sequential play to capture richer preference structures. The sequential design of SLHF naturally enables inference-time refinement, as the Follower learns to improve the Leader's actions, and these refinements can be leveraged through iterative sampling. We compare the solution concepts of SLHF, RLHF, and NLHF, and lay out key advantages in consistency, data sensitivity, and robustness to intransitive preferences. Experiments on large language models demonstrate that SLHF achieves strong alignment across diverse preference datasets, scales from 0.5B to 8B parameters, and yields inference-time refinements that transfer across model families without further fine-tuning. 

---
# Don't Guess, Escalate: Towards Explainable Uncertainty-Calibrated AI Forensic Agents 

**Authors**: Giulia Boato, Andrea Montibeller, Edward Delp, Luisa Verdoliva, Daniele Miorandi  

**Link**: [PDF](https://arxiv.org/pdf/2512.16614)  

**Abstract**: AI is reshaping the landscape of multimedia forensics. We propose AI forensic agents: reliable orchestrators that select and combine forensic detectors, identify provenance and context, and provide uncertainty-aware assessments. We highlight pitfalls in current solutions and introduce a unified framework to improve the authenticity verification process. 

---
# Refusal Steering: Fine-grained Control over LLM Refusal Behaviour for Sensitive Topics 

**Authors**: Iker García-Ferrero, David Montero, Roman Orus  

**Link**: [PDF](https://arxiv.org/pdf/2512.16602)  

**Abstract**: We introduce Refusal Steering, an inference-time method to exercise fine-grained control over Large Language Models refusal behaviour on politically sensitive topics without retraining. We replace fragile pattern-based refusal detection with an LLM-as-a-judge that assigns refusal confidence scores and we propose a ridge-regularized variant to compute steering vectors that better isolate the refusal--compliance direction. On Qwen3-Next-80B-A3B-Thinking, our method removes the refusal behaviour of the model around politically sensitive topics while maintaining safety on JailbreakBench and near-baseline performance on general benchmarks. The approach generalizes across 4B and 80B models and can also induce targeted refusals when desired. We analize the steering vectors and show that refusal signals concentrate in deeper layers of the transformer and are distributed across many dimensions. Together, these results demonstrate that activation steering can remove political refusal behaviour while retaining safety alignment for harmful content, offering a practical path to controllable, transparent moderation at inference time. 

---
# Yuan-TecSwin: A text conditioned Diffusion model with Swin-transformer blocks 

**Authors**: Shaohua Wu, Tong Yu, Shenling Wang, Xudong Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2512.16586)  

**Abstract**: Diffusion models have shown remarkable capacity in image synthesis based on their U-shaped architecture and convolutional neural networks (CNN) as basic blocks. The locality of the convolution operation in CNN may limit the model's ability to understand long-range semantic information. To address this issue, we propose Yuan-TecSwin, a text-conditioned diffusion model with Swin-transformer in this work. The Swin-transformer blocks take the place of CNN blocks in the encoder and decoder, to improve the non-local modeling ability in feature extraction and image restoration. The text-image alignment is improved with a well-chosen text encoder, effective utilization of text embedding, and careful design in the incorporation of text condition. Using an adapted time step to search in different diffusion stages, inference performance is further improved by 10%. Yuan-TecSwin achieves the state-of-the-art FID score of 1.37 on ImageNet generation benchmark, without any additional models at different denoising stages. In a side-by-side comparison, we find it difficult for human interviewees to tell the model-generated images from the human-painted ones. 

---
# Plain language adaptations of biomedical text using LLMs: Comparision of evaluation metrics 

**Authors**: Primoz Kocbek, Leon Kopitar, Gregor Stiglic  

**Link**: [PDF](https://arxiv.org/pdf/2512.16530)  

**Abstract**: This study investigated the application of Large Language Models (LLMs) for simplifying biomedical texts to enhance health literacy. Using a public dataset, which included plain language adaptations of biomedical abstracts, we developed and evaluated several approaches, specifically a baseline approach using a prompt template, a two AI agent approach, and a fine-tuning approach. We selected OpenAI gpt-4o and gpt-4o mini models as baselines for further research. We evaluated our approaches with quantitative metrics, such as Flesch-Kincaid grade level, SMOG Index, SARI, and BERTScore, G-Eval, as well as with qualitative metric, more precisely 5-point Likert scales for simplicity, accuracy, completeness, brevity. Results showed a superior performance of gpt-4o-mini and an underperformance of FT approaches. G-Eval, a LLM based quantitative metric, showed promising results, ranking the approaches similarly as the qualitative metric. 

---
# TTP: Test-Time Padding for Adversarial Detection and Robust Adaptation on Vision-Language Models 

**Authors**: Zhiwei Li, Yitian Pang, Weining Wang, Zhenan Sun, Qi Li  

**Link**: [PDF](https://arxiv.org/pdf/2512.16523)  

**Abstract**: Vision-Language Models (VLMs), such as CLIP, have achieved impressive zero-shot recognition performance but remain highly susceptible to adversarial perturbations, posing significant risks in safety-critical scenarios. Previous training-time defenses rely on adversarial fine-tuning, which requires labeled data and costly retraining, while existing test-time strategies fail to reliably distinguish between clean and adversarial inputs, thereby preventing both adversarial robustness and clean accuracy from reaching their optimum. To address these limitations, we propose Test-Time Padding (TTP), a lightweight defense framework that performs adversarial detection followed by targeted adaptation at inference. TTP identifies adversarial inputs via the cosine similarity shift between CLIP feature embeddings computed before and after spatial padding, yielding a universal threshold for reliable detection across architectures and datasets. For detected adversarial cases, TTP employs trainable padding to restore disrupted attention patterns, coupled with a similarity-aware ensemble strategy for a more robust final prediction. For clean inputs, TTP leaves them unchanged by default or optionally integrates existing test-time adaptation techniques for further accuracy gains. Comprehensive experiments on diverse CLIP backbones and fine-grained benchmarks show that TTP consistently surpasses state-of-the-art test-time defenses, delivering substantial improvements in adversarial robustness without compromising clean accuracy. The code for this paper will be released soon. 

---
# The Universe Learning Itself: On the Evolution of Dynamics from the Big Bang to Machine Intelligence 

**Authors**: Pradeep Singh, Mudasani Rushikesh, Bezawada Sri Sai Anurag, Balasubramanian Raman  

**Link**: [PDF](https://arxiv.org/pdf/2512.16515)  

**Abstract**: We develop a unified, dynamical-systems narrative of the universe that traces a continuous chain of structure formation from the Big Bang to contemporary human societies and their artificial learning systems. Rather than treating cosmology, astrophysics, geophysics, biology, cognition, and machine intelligence as disjoint domains, we view each as successive regimes of dynamics on ever-richer state spaces, stitched together by phase transitions, symmetry-breaking events, and emergent attractors. Starting from inflationary field dynamics and the growth of primordial perturbations, we describe how gravitational instability sculpts the cosmic web, how dissipative collapse in baryonic matter yields stars and planets, and how planetary-scale geochemical cycles define long-lived nonequilibrium attractors. Within these attractors, we frame the origin of life as the emergence of self-maintaining reaction networks, evolutionary biology as flow on high-dimensional genotype-phenotype-environment manifolds, and brains as adaptive dynamical systems operating near critical surfaces. Human culture and technology-including modern machine learning and artificial intelligence-are then interpreted as symbolic and institutional dynamics that implement and refine engineered learning flows which recursively reshape their own phase space. Throughout, we emphasize recurring mathematical motifs-instability, bifurcation, multiscale coupling, and constrained flows on measure-zero subsets of the accessible state space. Our aim is not to present any new cosmological or biological model, but a cross-scale, theoretical perspective: a way of reading the universe's history as the evolution of dynamics itself, culminating (so far) in biological and artificial systems capable of modeling, predicting, and deliberately perturbing their own future trajectories. 

---
# XTC, A Research Platform for Optimizing AI Workload Operators 

**Authors**: Pompougnac Hugo, Guillon Christophe, Noiry Sylvain, Dutilleul Alban, Iooss Guillaume, Rastello Fabrice  

**Link**: [PDF](https://arxiv.org/pdf/2512.16512)  

**Abstract**: Achieving high efficiency on AI operators demands precise control over computation and data movement. However, existing scheduling languages are locked into specific compiler ecosystems, preventing fair comparison, reuse, and evaluation across frameworks. No unified interface currently decouples scheduling specification from code generation and measurement. We introduce XTC, a platform that unifies scheduling and performance evaluation across compilers. With its common API and reproducible measurement framework, XTC enables portable experimentation and accelerates research on optimization strategies. 

---
# PoseMoE: Mixture-of-Experts Network for Monocular 3D Human Pose Estimation 

**Authors**: Mengyuan Liu, Jiajie Liu, Jinyan Zhang, Wenhao Li, Junsong Yuan  

**Link**: [PDF](https://arxiv.org/pdf/2512.16494)  

**Abstract**: The lifting-based methods have dominated monocular 3D human pose estimation by leveraging detected 2D poses as intermediate representations. The 2D component of the final 3D human pose benefits from the detected 2D poses, whereas its depth counterpart must be estimated from scratch. The lifting-based methods encode the detected 2D pose and unknown depth in an entangled feature space, explicitly introducing depth uncertainty to the detected 2D pose, thereby limiting overall estimation accuracy. This work reveals that the depth representation is pivotal for the estimation process. Specifically, when depth is in an initial, completely unknown state, jointly encoding depth features with 2D pose features is detrimental to the estimation process. In contrast, when depth is initially refined to a more dependable state via network-based estimation, encoding it together with 2D pose information is beneficial. To address this limitation, we present a Mixture-of-Experts network for monocular 3D pose estimation named PoseMoE. Our approach introduces: (1) A mixture-of-experts network where specialized expert modules refine the well-detected 2D pose features and learn the depth features. This mixture-of-experts design disentangles the feature encoding process for 2D pose and depth, therefore reducing the explicit influence of uncertain depth features on 2D pose features. (2) A cross-expert knowledge aggregation module is proposed to aggregate cross-expert spatio-temporal contextual information. This step enhances features through bidirectional mapping between 2D pose and depth. Extensive experiments show that our proposed PoseMoE outperforms the conventional lifting-based methods on three widely used datasets: Human3.6M, MPI-INF-3DHP, and 3DPW. 

---
# Smile on the Face, Sadness in the Eyes: Bridging the Emotion Gap with a Multimodal Dataset of Eye and Facial Behaviors 

**Authors**: Kejun Liu, Yuanyuan Liu, Lin Wei, Chang Tang, Yibing Zhan, Zijing Chen, Zhe Chen  

**Link**: [PDF](https://arxiv.org/pdf/2512.16485)  

**Abstract**: Emotion Recognition (ER) is the process of analyzing and identifying human emotions from sensing data. Currently, the field heavily relies on facial expression recognition (FER) because visual channel conveys rich emotional cues. However, facial expressions are often used as social tools rather than manifestations of genuine inner emotions. To understand and bridge this gap between FER and ER, we introduce eye behaviors as an important emotional cue and construct an Eye-behavior-aided Multimodal Emotion Recognition (EMER) dataset. To collect data with genuine emotions, spontaneous emotion induction paradigm is exploited with stimulus material, during which non-invasive eye behavior data, like eye movement sequences and eye fixation maps, is captured together with facial expression videos. To better illustrate the gap between ER and FER, multi-view emotion labels for mutimodal ER and FER are separately annotated. Furthermore, based on the new dataset, we design a simple yet effective Eye-behavior-aided MER Transformer (EMERT) that enhances ER by bridging the emotion gap. EMERT leverages modality-adversarial feature decoupling and a multitask Transformer to model eye behaviors as a strong complement to facial expressions. In the experiment, we introduce seven multimodal benchmark protocols for a variety of comprehensive evaluations of the EMER dataset. The results show that the EMERT outperforms other state-of-the-art multimodal methods by a great margin, revealing the importance of modeling eye behaviors for robust ER. To sum up, we provide a comprehensive analysis of the importance of eye behaviors in ER, advancing the study on addressing the gap between FER and ER for more robust ER performance. Our EMER dataset and the trained EMERT models will be publicly available at this https URL. 

---
# Guiding Perception-Reasoning Closer to Human in Blind Image Quality Assessment 

**Authors**: Yuan Li, Yahan Yu, Youyuan Lin, Yong-Hao Yang, Chenhui Chu, Shin'ya Nishida  

**Link**: [PDF](https://arxiv.org/pdf/2512.16484)  

**Abstract**: Humans assess image quality through a perception-reasoning cascade, integrating sensory cues with implicit reasoning to form self-consistent judgments. In this work, we investigate how a model can acquire both human-like and self-consistent reasoning capability for blind image quality assessment (BIQA). We first collect human evaluation data that capture several aspects of human perception-reasoning pipeline. Then, we adopt reinforcement learning, using human annotations as reward signals to guide the model toward human-like perception and reasoning. To enable the model to internalize self-consistent reasoning capability, we design a reward that drives the model to infer the image quality purely from self-generated descriptions. Empirically, our approach achieves score prediction performance comparable to state-of-the-art BIQA systems under general metrics, including Pearson and Spearman correlation coefficients. In addition to the rating score, we assess human-model alignment using ROUGE-1 to measure the similarity between model-generated and human perception-reasoning chains. On over 1,000 human-annotated samples, our model reaches a ROUGE-1 score of 0.512 (cf. 0.443 for baseline), indicating substantial coverage of human explanations and marking a step toward human-like interpretable reasoning in BIQA. 

---
# AI4EOSC: a Federated Cloud Platform for Artificial Intelligence in Scientific Research 

**Authors**: Ignacio Heredia, Álvaro López García, Germán Moltó, Amanda Calatrava, Valentin Kozlov, Alessandro Costantini, Viet Tran, Mario David, Daniel San Martín, Marcin Płóciennik, Marta Obregón Ruiz, Saúl Fernandez, Judith Sáinz-Pardo Díaz, Miguel Caballer, Caterina Alarcón Marín, Stefan Dlugolinsky, Martin Šeleng, Lisana Berberi, Khadijeh Alibabaei, Borja Esteban Sanchis, Pedro Castro, Giacinto Donvito, Diego Aguirre, Sergio Langarita, Vicente Rodriguez, Leonhard Duda, Andrés Heredia Canales, Susana Rebolledo Ruiz, João Machado, Giang Nguyen, Fernando Aguilar Gómez, Jaime Díez  

**Link**: [PDF](https://arxiv.org/pdf/2512.16455)  

**Abstract**: In this paper, we describe a federated compute platform dedicated to support Artificial Intelligence in scientific workloads. Putting the effort into reproducible deployments, it delivers consistent, transparent access to a federation of physically distributed e-Infrastructures. Through a comprehensive service catalogue, the platform is able to offer an integrated user experience covering the full Machine Learning lifecycle, including model development (with dedicated interactive development environments), training (with GPU resources, annotation tools, experiment tracking, and federated learning support) and deployment (covering a wide range of deployment options all along the Cloud Continuum). The platform also provides tools for traceability and reproducibility of AI models, integrates with different Artificial Intelligence model providers, datasets and storage resources, allowing users to interact with the broader Machine Learning ecosystem. Finally, it is easily customizable to lower the adoption barrier by external communities. 

---
# IoMT-based Automated Leukemia Classification using CNN and Higher Order Singular Value 

**Authors**: Shabnam Bagheri Marzijarani, Mohammad Zolfaghari, Hedieh Sajedi  

**Link**: [PDF](https://arxiv.org/pdf/2512.16448)  

**Abstract**: The Internet of Things (IoT) is a concept by which objects find identity and can communicate with each other in a network. One of the applications of the IoT is in the field of medicine, which is called the Internet of Medical Things (IoMT). Acute Lymphocytic Leukemia (ALL) is a type of cancer categorized as a hematic disease. It usually begins in the bone marrow due to the overproduction of immature White Blood Cells (WBCs or leukocytes). Since it has a high rate of spread to other body organs, it is a fatal disease if not diagnosed and treated early. Therefore, for identifying cancerous (ALL) cells in medical diagnostic laboratories, blood, as well as bone marrow smears, are taken by pathologists. However, manual examinations face limitations due to human error risk and time-consuming procedures. So, to tackle the mentioned issues, methods based on Artificial Intelligence (AI), capable of identifying cancer from non-cancer tissue, seem vital. Deep Neural Networks (DNNs) are the most efficient machine learning (ML) methods. These techniques employ multiple layers to extract higher-level features from the raw input. In this paper, a Convolutional Neural Network (CNN) is applied along with a new type of classifier, Higher Order Singular Value Decomposition (HOSVD), to categorize ALL and normal (healthy) cells from microscopic blood images. We employed the model on IoMT structure to identify leukemia quickly and safely. With the help of this new leukemia classification framework, patients and clinicians can have real-time communication. The model was implemented on the Acute Lymphoblastic Leukemia Image Database (ALL-IDB2) and achieved an average accuracy of %98.88 in the test step. 

---
# E-SDS: Environment-aware See it, Do it, Sorted - Automated Environment-Aware Reinforcement Learning for Humanoid Locomotion 

**Authors**: Enis Yalcin, Joshua O'Hara, Maria Stamatopoulou, Chengxu Zhou, Dimitrios Kanoulas  

**Link**: [PDF](https://arxiv.org/pdf/2512.16446)  

**Abstract**: Vision-language models (VLMs) show promise in automating reward design in humanoid locomotion, which could eliminate the need for tedious manual engineering. However, current VLM-based methods are essentially "blind", as they lack the environmental perception required to navigate complex terrain. We present E-SDS (Environment-aware See it, Do it, Sorted), a framework that closes this perception gap. E-SDS integrates VLMs with real-time terrain sensor analysis to automatically generate reward functions that facilitate training of robust perceptive locomotion policies, grounded by example videos. Evaluated on a Unitree G1 humanoid across four distinct terrains (simple, gaps, obstacles, stairs), E-SDS uniquely enabled successful stair descent, while policies trained with manually-designed rewards or a non-perceptive automated baseline were unable to complete the task. In all terrains, E-SDS also reduced velocity tracking error by 51.9-82.6%. Our framework reduces the human effort of reward design from days to less than two hours while simultaneously producing more robust and capable locomotion policies. 

---
# Topic Modelling Black Box Optimization 

**Authors**: Roman Akramov, Artem Khamatullin, Svetlana Glazyrina, Maksim Kryzhanovskiy, Roman Ischenko  

**Link**: [PDF](https://arxiv.org/pdf/2512.16445)  

**Abstract**: Choosing the number of topics $T$ in Latent Dirichlet Allocation (LDA) is a key design decision that strongly affects both the statistical fit and interpretability of topic models. In this work, we formulate the selection of $T$ as a discrete black-box optimization problem, where each function evaluation corresponds to training an LDA model and measuring its validation perplexity. Under a fixed evaluation budget, we compare four families of optimizers: two hand-designed evolutionary methods - Genetic Algorithm (GA) and Evolution Strategy (ES) - and two learned, amortized approaches, Preferential Amortized Black-Box Optimization (PABBO) and Sharpness-Aware Black-Box Optimization (SABBO). Our experiments show that, while GA, ES, PABBO, and SABBO eventually reach a similar band of final perplexity, the amortized optimizers are substantially more sample- and time-efficient. SABBO typically identifies a near-optimal topic number after essentially a single evaluation, and PABBO finds competitive configurations within a few evaluations, whereas GA and ES require almost the full budget to approach the same region. 

---
# Emergent Bias and Fairness in Multi-Agent Decision Systems 

**Authors**: Maeve Madigan, Parameswaran Kamalaruban, Glenn Moynihan, Tom Kempton, David Sutton, Stuart Burrell  

**Link**: [PDF](https://arxiv.org/pdf/2512.16433)  

**Abstract**: Multi-agent systems have demonstrated the ability to improve performance on a variety of predictive tasks by leveraging collaborative decision making. However, the lack of effective evaluation methodologies has made it difficult to estimate the risk of bias, making deployment of such systems unsafe in high stakes domains such as consumer finance, where biased decisions can translate directly into regulatory breaches and financial loss. To address this challenge, we need to develop fairness evaluation methodologies for multi-agent predictive systems and measure the fairness characteristics of these systems in the financial tabular domain. Examining fairness metrics using large-scale simulations across diverse multi-agent configurations, with varying communication and collaboration mechanisms, we reveal patterns of emergent bias in financial decision-making that cannot be traced to individual agent components, indicating that multi-agent systems may exhibit genuinely collective behaviors. Our findings highlight that fairness risks in financial multi-agent systems represent a significant component of model risk, with tangible impacts on tasks such as credit scoring and income estimation. We advocate that multi-agent decision systems must be evaluated as holistic entities rather than through reductionist analyses of their constituent components. 

---
# Introducing ORKG ASK: an AI-driven Scholarly Literature Search and Exploration System Taking a Neuro-Symbolic Approach 

**Authors**: Allard Oelen, Mohamad Yaser Jaradeh, Sören Auer  

**Link**: [PDF](https://arxiv.org/pdf/2512.16425)  

**Abstract**: As the volume of published scholarly literature continues to grow, finding relevant literature becomes increasingly difficult. With the rise of generative Artificial Intelligence (AI), and particularly Large Language Models (LLMs), new possibilities emerge to find and explore literature. We introduce ASK (Assistant for Scientific Knowledge), an AI-driven scholarly literature search and exploration system that follows a neuro-symbolic approach. ASK aims to provide active support to researchers in finding relevant scholarly literature by leveraging vector search, LLMs, and knowledge graphs. The system allows users to input research questions in natural language and retrieve relevant articles. ASK automatically extracts key information and generates answers to research questions using a Retrieval-Augmented Generation (RAG) approach. We present an evaluation of ASK, assessing the system's usability and usefulness. Findings indicate that the system is user-friendly and users are generally satisfied while using the system. 

---
# Hypernetworks That Evolve Themselves 

**Authors**: Joachim Winther Pedersen, Erwan Plantec, Eleni Nisioti, Marcello Barylli, Milton Montero, Kathrin Korte, Sebastian Risi  

**Link**: [PDF](https://arxiv.org/pdf/2512.16406)  

**Abstract**: How can neural networks evolve themselves without relying on external optimizers? We propose Self-Referential Graph HyperNetworks, systems where the very machinery of variation and inheritance is embedded within the network. By uniting hypernetworks, stochastic parameter generation, and graph-based representations, Self-Referential GHNs mutate and evaluate themselves while adapting mutation rates as selectable traits. Through new reinforcement learning benchmarks with environmental shifts (CartPoleSwitch, LunarLander-Switch), Self-Referential GHNs show swift, reliable adaptation and emergent population dynamics. In the locomotion benchmark Ant-v5, they evolve coherent gaits, showing promising fine-tuning capabilities by autonomously decreasing variation in the population to concentrate around promising solutions. Our findings support the idea that evolvability itself can emerge from neural self-reference. Self-Referential GHNs reflect a step toward synthetic systems that more closely mirror biological evolution, offering tools for autonomous, open-ended learning agents. 

---
# Using Gaussian Splats to Create High-Fidelity Facial Geometry and Texture 

**Authors**: Haodi He, Jihun Yu, Ronald Fedkiw  

**Link**: [PDF](https://arxiv.org/pdf/2512.16397)  

**Abstract**: We leverage increasingly popular three-dimensional neural representations in order to construct a unified and consistent explanation of a collection of uncalibrated images of the human face. Our approach utilizes Gaussian Splatting, since it is more explicit and thus more amenable to constraints than NeRFs. We leverage segmentation annotations to align the semantic regions of the face, facilitating the reconstruction of a neutral pose from only 11 images (as opposed to requiring a long video). We soft constrain the Gaussians to an underlying triangulated surface in order to provide a more structured Gaussian Splat reconstruction, which in turn informs subsequent perturbations to increase the accuracy of the underlying triangulated surface. The resulting triangulated surface can then be used in a standard graphics pipeline. In addition, and perhaps most impactful, we show how accurate geometry enables the Gaussian Splats to be transformed into texture space where they can be treated as a view-dependent neural texture. This allows one to use high visual fidelity Gaussian Splatting on any asset in a scene without the need to modify any other asset or any other aspect (geometry, lighting, renderer, etc.) of the graphics pipeline. We utilize a relightable Gaussian model to disentangle texture from lighting in order to obtain a delit high-resolution albedo texture that is also readily usable in a standard graphics pipeline. The flexibility of our system allows for training with disparate images, even with incompatible lighting, facilitating robust regularization. Finally, we demonstrate the efficacy of our approach by illustrating its use in a text-driven asset creation pipeline. 

---
# Kascade: A Practical Sparse Attention Method for Long-Context LLM Inference 

**Authors**: Dhruv Deshmukh, Saurabh Goyal, Nipun Kwatra, Ramachandran Ramjee  

**Link**: [PDF](https://arxiv.org/pdf/2512.16391)  

**Abstract**: Attention is the dominant source of latency during long-context LLM inference, an increasingly popular workload with reasoning models and RAG. We propose Kascade, a training-free sparse attention method that leverages known observations such as 1) post-softmax attention is intrinsically sparse, and 2) the identity of high-weight keys is stable across nearby layers. Kascade computes exact Top-k indices in a small set of anchor layers, then reuses those indices in intermediate reuse layers. The anchor layers are selected algorithmically, via a dynamic-programming objective that maximizes cross-layer similarity over a development set, allowing easy deployment across models. The method incorporates efficient implementation constraints (e.g. tile-level operations), across both prefill and decode attention. The Top-k selection and reuse in Kascade is head-aware and we show in our experiments that this is critical for high accuracy. Kascade achieves up to 4.1x speedup in decode attention and 2.2x speedup in prefill attention over FlashAttention-3 baseline on H100 GPUs while closely matching dense attention accuracy on long-context benchmarks such as LongBench and AIME-24. 

---
# Hearing to Translate: The Effectiveness of Speech Modality Integration into LLMs 

**Authors**: Sara Papi, Javier Garcia Gilabert, Zachary Hopton, Vilém Zouhar, Carlos Escolano, Gerard I. Gállego, Jorge Iranzo-Sánchez, Ahrii Kim, Dominik Macháček, Patricia Schmidtova, Maike Züfle  

**Link**: [PDF](https://arxiv.org/pdf/2512.16378)  

**Abstract**: As Large Language Models (LLMs) expand beyond text, integrating speech as a native modality has given rise to SpeechLLMs, which aim to translate spoken language directly, thereby bypassing traditional transcription-based pipelines. Whether this integration improves speech-to-text translation quality over established cascaded architectures, however, remains an open question. We present Hearing to Translate, the first comprehensive test suite rigorously benchmarking 5 state-of-the-art SpeechLLMs against 16 strong direct and cascade systems that couple leading speech foundation models (SFM), with multilingual LLMs. Our analysis spans 16 benchmarks, 13 language pairs, and 9 challenging conditions, including disfluent, noisy, and long-form speech. Across this extensive evaluation, we find that cascaded systems remain the most reliable overall, while current SpeechLLMs only match cascades in selected settings and SFMs lag behind both, highlighting that integrating an LLM, either within the model or in a pipeline, is essential for high-quality speech translation. 

---
# Collaborative Edge-to-Server Inference for Vision-Language Models 

**Authors**: Soochang Song, Yongjune Kim  

**Link**: [PDF](https://arxiv.org/pdf/2512.16349)  

**Abstract**: We propose a collaborative edge-to-server inference framework for vision-language models (VLMs) that reduces the communication cost while maintaining inference accuracy. In typical deployments, visual data captured at edge devices (clients) is transmitted to the server for VLM inference. However, resizing the original image (global image) to match the vision encoder's input resolution often discards fine-grained details, leading to accuracy degradation. To overcome this limitation, we design a two-stage framework. In the first stage, the server performs inference on the global image and identifies a region of interest (RoI) using the VLM's internal attention. The min-entropy of the output tokens is then computed as a confidence measure to determine whether retransmission is required. If the min-entropy exceeds a predefined threshold, the server requests the edge device to send a detail-preserved local image of the RoI. The server then refines its inference by jointly leveraging the global and local images. This selective retransmission strategy ensures that only essential visual content is transmitted. Experiments across multiple VLM architectures show that the proposed framework significantly reduces communication cost while maintaining inference accuracy. 

---
# Pretrained Battery Transformer (PBT): A battery life prediction foundation model 

**Authors**: Ruifeng Tan, Weixiang Hong, Jia Li, Jiaqiang Huang, Tong-Yi Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2512.16334)  

**Abstract**: Early prediction of battery cycle life is essential for accelerating battery research, manufacturing, and deployment. Although machine learning methods have shown encouraging results, progress is hindered by data scarcity and heterogeneity arising from diverse aging conditions. In other fields, foundation models (FMs) trained on diverse datasets have achieved broad generalization through transfer learning, but no FMs have been reported for battery cycle life prediction yet. Here we present the Pretrained Battery Transformer (PBT), the first FM for battery life prediction, developed through domain-knowledge-encoded mixture-of-expert layers. Validated on the largest public battery life database, PBT learns transferable representations from 13 lithium-ion battery (LIB) datasets, outperforming existing models by an average of 19.8%. With transfer learning, PBT achieves state-of-the-art performance across 15 diverse datasets encompassing various operating conditions, formation protocols, and chemistries of LIBs. This work establishes a foundation model pathway for battery lifetime prediction, paving the way toward universal battery lifetime prediction systems. 

---
# Agent Tools Orchestration Leaks More: Dataset, Benchmark, and Mitigation 

**Authors**: Yuxuan Qiao, Dongqin Liu, Hongchang Yang, Wei Zhou, Songlin Hu  

**Link**: [PDF](https://arxiv.org/pdf/2512.16310)  

**Abstract**: Driven by Large Language Models, the single-agent, multi-tool architecture has become a popular paradigm for autonomous agents due to its simplicity and effectiveness. However, this architecture also introduces a new and severe privacy risk, which we term Tools Orchestration Privacy Risk (TOP-R), where an agent, to achieve a benign user goal, autonomously aggregates information fragments across multiple tools and leverages its reasoning capabilities to synthesize unexpected sensitive information. We provide the first systematic study of this risk. First, we establish a formal framework, attributing the risk's root cause to the agent's misaligned objective function: an overoptimization for helpfulness while neglecting privacy awareness. Second, we construct TOP-Bench, comprising paired leakage and benign scenarios, to comprehensively evaluate this risk. To quantify the trade-off between safety and robustness, we introduce the H-Score as a holistic metric. The evaluation results reveal that TOP-R is a severe risk: the average Risk Leakage Rate (RLR) of eight representative models reaches 90.24%, while the average H-Score is merely 0.167, with no model exceeding 0.3. Finally, we propose the Privacy Enhancement Principle (PEP) method, which effectively mitigates TOP-R, reducing the Risk Leakage Rate to 46.58% and significantly improving the H-Score to 0.624. Our work reveals both a new class of risk and inherent structural limitations in current agent architectures, while also offering feasible mitigation strategies. 

---
# Beyond the Benchmark: Innovative Defenses Against Prompt Injection Attacks 

**Authors**: Safwan Shaheer, G.M. Refatul Islam, Mohammad Rafid Hamid, Tahsin Zaman Jilan  

**Link**: [PDF](https://arxiv.org/pdf/2512.16307)  

**Abstract**: In this fast-evolving area of LLMs, our paper discusses the significant security risk presented by prompt injection attacks. It focuses on small open-sourced models, specifically the LLaMA family of models. We introduce novel defense mechanisms capable of generating automatic defenses and systematically evaluate said generated defenses against a comprehensive set of benchmarked attacks. Thus, we empirically demonstrated the improvement proposed by our approach in mitigating goal-hijacking vulnerabilities in LLMs. Our work recognizes the increasing relevance of small open-sourced LLMs and their potential for broad deployments on edge devices, aligning with future trends in LLM applications. We contribute to the greater ecosystem of open-source LLMs and their security in the following: (1) assessing present prompt-based defenses against the latest attacks, (2) introducing a new framework using a seed defense (Chain Of Thoughts) to refine the defense prompts iteratively, and (3) showing significant improvements in detecting goal hijacking attacks. Out strategies significantly reduce the success rates of the attacks and false detection rates while at the same time effectively detecting goal-hijacking capabilities, paving the way for more secure and efficient deployments of small and open-source LLMs in resource-constrained environments. 

---
# PixelArena: A benchmark for Pixel-Precision Visual Intelligence 

**Authors**: Feng Liang, Sizhe Cheng, Chenqi Yi  

**Link**: [PDF](https://arxiv.org/pdf/2512.16303)  

**Abstract**: Multi-modal large language models that have image output are emerging. Many image generation benchmarks focus on aesthetics instead of fine-grained generation capabilities. In PixelArena, we propose using semantic segmentation tasks to objectively examine their fine-grained generative intelligence with pixel precision. We find the latest Gemini 3 Pro Image has emergent image generation capabilities that generate semantic masks with high fidelity under zero-shot settings, showcasing visual intelligence unseen before and true generalization in new image generation tasks. We further investigate its results, compare them qualitatively and quantitatively with those of other models, and present failure cases. The findings not only signal exciting progress in the field but also provide insights into future research related to multimodality, reasoning, interpretability and benchmarking. 

---
# Feature-Selective Representation Misdirection for Machine Unlearning 

**Authors**: Taozhao Chen, Linghan Huang, Kim-Kwang Raymond Choo, Huaming Chen  

**Link**: [PDF](https://arxiv.org/pdf/2512.16297)  

**Abstract**: As large language models (LLMs) are increasingly adopted in safety-critical and regulated sectors, the retention of sensitive or prohibited knowledge introduces escalating risks, ranging from privacy leakage to regulatory non-compliance to to potential misuse, and so on. Recent studies suggest that machine unlearning can help ensure deployed models comply with evolving legal, safety, and governance requirements. However, current unlearning techniques assume clean separation between forget and retain datasets, which is challenging in operational settings characterized by highly entangled distributions. In such scenarios, perturbation-based methods often degrade general model utility or fail to ensure safety. To address this, we propose Selective Representation Misdirection for Unlearning (SRMU), a novel principled activation-editing framework that enforces feature-aware and directionally controlled perturbations. Unlike indiscriminate model weights perturbations, SRMU employs a structured misdirection vector with an activation importance map. The goal is to allow SRMU selectively suppresses harmful representations while preserving the utility on benign ones. Experiments are conducted on the widely used WMDP benchmark across low- and high-entanglement configurations. Empirical results reveal that SRMU delivers state-of-the-art unlearning performance with minimal utility losses, and remains effective under 20-30\% overlap where existing baselines collapse. SRMU provides a robust foundation for safety-driven model governance, privacy compliance, and controlled knowledge removal in the emerging LLM-based applications. We release the replication package at this https URL. 

---
# CKA-Guided Modular Quantization: Beyond Bit-Width to Algorithmic Diversity 

**Authors**: Jinhao Zhang, Yunquan Zhang, Daning Chen  

**Link**: [PDF](https://arxiv.org/pdf/2512.16282)  

**Abstract**: Current mainstream post-training quantization methods for large language models typically apply a uniform quantization strategy across all network layers, overlooking the substantial differences in algorithmic suitability among layers. To address this limitation, we propose CKA Guided Modular Quantization, a fine-tuning-free, plug-and-play framework for algorithmic heterogeneous quantization. Our method independently evaluates multiple PTQ algorithms on each layer and employs Linear Centered Kernel Alignment (CKA) as a metric to automatically select the optimal quantization strategy per layer. The individually optimized strategies are then integrated to construct a hybrid quantized model. Experiments demonstrate that our approach consistently outperforms both uniform quantization baselines and state-of-the-art mixed-precision methods across mainstream LLMs including LLaMA and Qwen ,in terms of perplexity (PPL) and downstream task performance. 

---
# Love, Lies, and Language Models: Investigating AI's Role in Romance-Baiting Scams 

**Authors**: Gilad Gressel, Rahul Pankajakshan, Shir Rozenfeld, Ling Li, Ivan Franceschini, Krishnahsree Achuthan, Yisroel Mirsky  

**Link**: [PDF](https://arxiv.org/pdf/2512.16280)  

**Abstract**: Romance-baiting scams have become a major source of financial and emotional harm worldwide. These operations are run by organized crime syndicates that traffic thousands of people into forced labor, requiring them to build emotional intimacy with victims over weeks of text conversations before pressuring them into fraudulent cryptocurrency investments. Because the scams are inherently text-based, they raise urgent questions about the role of Large Language Models (LLMs) in both current and future automation.
We investigate this intersection by interviewing 145 insiders and 5 scam victims, performing a blinded long-term conversation study comparing LLM scam agents to human operators, and executing an evaluation of commercial safety filters. Our findings show that LLMs are already widely deployed within scam organizations, with 87% of scam labor consisting of systematized conversational tasks readily susceptible to automation. In a week-long study, an LLM agent not only elicited greater trust from study participants (p=0.007) but also achieved higher compliance with requests than human operators (46% vs. 18% for humans). Meanwhile, popular safety filters detected 0.0% of romance baiting dialogues. Together, these results suggest that romance-baiting scams may be amenable to full-scale LLM automation, while existing defenses remain inadequate to prevent their expansion. 

---
# GFLAN: Generative Functional Layouts 

**Authors**: Mohamed Abouagour, Eleftherios Garyfallidis  

**Link**: [PDF](https://arxiv.org/pdf/2512.16275)  

**Abstract**: Automated floor plan generation lies at the intersection of combinatorial search, geometric constraint satisfaction, and functional design requirements -- a confluence that has historically resisted a unified computational treatment. While recent deep learning approaches have improved the state of the art, they often struggle to capture architectural reasoning: the precedence of topological relationships over geometric instantiation, the propagation of functional constraints through adjacency networks, and the emergence of circulation patterns from local connectivity decisions. To address these fundamental challenges, this paper introduces GFLAN, a generative framework that restructures floor plan synthesis through explicit factorization into topological planning and geometric realization. Given a single exterior boundary and a front-door location, our approach departs from direct pixel-to-pixel or wall-tracing generation in favor of a principled two-stage decomposition. Stage A employs a specialized convolutional architecture with dual encoders -- separating invariant spatial context from evolving layout state -- to sequentially allocate room centroids within the building envelope via discrete probability maps over feasible placements. Stage B constructs a heterogeneous graph linking room nodes to boundary vertices, then applies a Transformer-augmented graph neural network (GNN) that jointly regresses room boundaries. 

---
# Beyond Blind Spots: Analytic Hints for Mitigating LLM-Based Evaluation Pitfalls 

**Authors**: Ora Nova Fandina, Eitan Farchi, Shmulik Froimovich, Raviv Gal, Wesam Ibraheem, Rami Katan, Alice Podolsky  

**Link**: [PDF](https://arxiv.org/pdf/2512.16272)  

**Abstract**: Large Language Models are increasingly deployed as judges (LaaJ) in code generation pipelines. While attractive for scalability, LaaJs tend to overlook domain specific issues raising concerns about their reliability in critical evaluation tasks. To better understand these limitations in practice, we examine LaaJ behavior in a concrete industrial use case: legacy code modernization via COBOL code generation. In this setting, we find that even production deployed LaaJs can miss domain critical errors, revealing consistent blind spots in their evaluation capabilities.
To better understand these blind spots, we analyze generated COBOL programs and associated LaaJs judgments, drawing on expert knowledge to construct a preliminary taxonomy. Based on this taxonomy, we develop a lightweight analytic checker tool that flags over 30 domain specific issues observed in practice. We use its outputs as analytic hints, dynamically injecting them into the judges prompt to encourage LaaJ to revisit aspects it may have overlooked.
Experiments on a test set of 100 programs using four production level LaaJs show that LaaJ alone detects only about 45% of the errors present in the code (in all judges we tested), while the analytic checker alone lacks explanatory depth. When combined, the LaaJ+Hints configuration achieves up to 94% coverage (for the best performing judge and injection prompt) and produces qualitatively richer, more accurate explanations, demonstrating that analytic-LLM hybrids can substantially enhance evaluation reliability in deployed pipelines. We release the dataset and all used prompts. 

---
# Domain-Agnostic Causal-Aware Audio Transformer for Infant Cry Classification 

**Authors**: Geofrey Owino, Bernard Shibwabo Kasamani, Ahmed M. Abdelmoniem, Edem Wornyo  

**Link**: [PDF](https://arxiv.org/pdf/2512.16271)  

**Abstract**: Accurate and interpretable classification of infant cry paralinguistics is essential for early detection of neonatal distress and clinical decision support. However, many existing deep learning methods rely on correlation-driven acoustic representations, which makes them vulnerable to noise, spurious cues, and domain shifts across recording environments. We propose DACH-TIC, a Domain-Agnostic Causal-Aware Hierarchical Audio Transformer for robust infant cry classification. The model integrates causal attention, hierarchical representation learning, multi-task supervision, and adversarial domain generalization within a unified framework.
DACH-TIC employs a structured transformer backbone with local token-level and global semantic encoders, augmented by causal attention masking and controlled perturbation training to approximate counterfactual acoustic variations. A domain-adversarial objective promotes environment-invariant representations, while multi-task learning jointly optimizes cry type recognition, distress intensity estimation, and causal relevance prediction. The model is evaluated on the Baby Chillanto and Donate-a-Cry datasets, with ESC-50 environmental noise overlays for domain augmentation.
Experimental results show that DACH-TIC outperforms state-of-the-art baselines, including HTS-AT and SE-ResNet Transformer, achieving improvements of 2.6 percent in accuracy and 2.2 points in macro-F1 score, alongside enhanced causal fidelity. The model generalizes effectively to unseen acoustic environments, with a domain performance gap of only 2.4 percent, demonstrating its suitability for real-world neonatal acoustic monitoring systems. 

---
# TextEditBench: Evaluating Reasoning-aware Text Editing Beyond Rendering 

**Authors**: Rui Gui, Yang Wan, Haochen Han, Dongxing Mao, Fangming Liu, Min Li, Alex Jinpeng Wang  

**Link**: [PDF](https://arxiv.org/pdf/2512.16270)  

**Abstract**: Text rendering has recently emerged as one of the most challenging frontiers in visual generation, drawing significant attention from large-scale diffusion and multimodal models. However, text editing within images remains largely unexplored, as it requires generating legible characters while preserving semantic, geometric, and contextual coherence. To fill this gap, we introduce TextEditBench, a comprehensive evaluation benchmark that explicitly focuses on text-centric regions in images. Beyond basic pixel manipulations, our benchmark emphasizes reasoning-intensive editing scenarios that require models to understand physical plausibility, linguistic meaning, and cross-modal dependencies. We further propose a novel evaluation dimension, Semantic Expectation (SE), which measures reasoning ability of model to maintain semantic consistency, contextual coherence, and cross-modal alignment during text editing. Extensive experiments on state-of-the-art editing systems reveal that while current models can follow simple textual instructions, they still struggle with context-dependent reasoning, physical consistency, and layout-aware integration. By focusing evaluation on this long-overlooked yet fundamental capability, TextEditBench establishes a new testing ground for advancing text-guided image editing and reasoning in multimodal generation. 

---
# Interpretable Deep Learning for Stock Returns: A Consensus-Bottleneck Asset Pricing Model 

**Authors**: Bong-Gyu Jang, Younwoo Jeong, Changeun Kim  

**Link**: [PDF](https://arxiv.org/pdf/2512.16251)  

**Abstract**: We introduce the \textit{Consensus-Bottleneck Asset Pricing Model} (CB-APM), a partially interpretable neural network that replicates the reasoning processes of sell-side analysts by capturing how dispersed investor beliefs are compressed into asset prices through a consensus formation process. By modeling this ``bottleneck'' to summarize firm- and macro-level information, CB-APM not only predicts future risk premiums of U.S. equities but also links belief aggregation to expected returns in a structurally interpretable manner. The model improves long-horizon return forecasts and outperforms standard deep learning approaches in both predictive accuracy and explanatory power. Comprehensive portfolio analyses show that CB-APM's out-of-sample predictions translate into economically meaningful payoffs, with monotonic return differentials and stable long-short performance across regularization settings. Empirically, CB-APM leverages consensus as a regularizer to amplify long-horizon predictability and yields interpretable consensus-based components that clarify how information is priced in returns. Moreover, regression and GRS-based pricing diagnostics reveal that the learned consensus representations capture priced variation only partially spanned by traditional factor models, demonstrating that CB-APM uncovers belief-driven structure in expected returns beyond the canonical factor space. Overall, CB-APM provides an interpretable and empirically grounded framework for understanding belief-driven return dynamics. 

---
# Sigma-Moe-Tiny Technical Report 

**Authors**: Qingguo Hu, Zhenghao Lin, Ziyue Yang, Yucheng Ding, Xiao Liu, Yuting Jiang, Ruizhe Wang, Tianyu Chen, Zhongxin Guo, Yifan Xiong, Rui Gao, Lei Qu, Jinsong Su, Peng Cheng, Yeyun Gong  

**Link**: [PDF](https://arxiv.org/pdf/2512.16248)  

**Abstract**: Mixture-of-Experts (MoE) has emerged as a promising paradigm for foundation models due to its efficient and powerful scalability. In this work, we present Sigma-MoE-Tiny, an MoE language model that achieves the highest sparsity compared to existing open-source models. Sigma-MoE-Tiny employs fine-grained expert segmentation with up to 96 experts per layer, while activating only one expert for each token, resulting in 20B total parameters with just 0.5B activated. The major challenge introduced by such extreme sparsity lies in expert load balancing. We find that the widely-used load balancing loss tends to become ineffective in the lower layers under this setting. To address this issue, we propose a progressive sparsification schedule aiming to balance expert utilization and training stability. Sigma-MoE-Tiny is pre-trained on a diverse and high-quality corpus, followed by post-training to further unlock its capabilities. The entire training process remains remarkably stable, with no occurrence of irrecoverable loss spikes. Comprehensive evaluations reveal that, despite activating only 0.5B parameters, Sigma-MoE-Tiny achieves top-tier performance among counterparts of comparable or significantly larger scale. In addition, we provide an in-depth discussion of load balancing in highly sparse MoE models, offering insights for advancing sparsity in future MoE architectures.
Project page: this https URL
Code: this https URL 

---
# Coarse-to-Fine Open-Set Graph Node Classification with Large Language Models 

**Authors**: Xueqi Ma, Xingjun Ma, Sarah Monazam Erfani, Danilo Mandic, James Bailey  

**Link**: [PDF](https://arxiv.org/pdf/2512.16244)  

**Abstract**: Developing open-set classification methods capable of classifying in-distribution (ID) data while detecting out-of-distribution (OOD) samples is essential for deploying graph neural networks (GNNs) in open-world scenarios. Existing methods typically treat all OOD samples as a single class, despite real-world applications, especially high-stake settings such as fraud detection and medical diagnosis, demanding deeper insights into OOD samples, including their probable labels. This raises a critical question: can OOD detection be extended to OOD classification without true label information? To address this question, we propose a Coarse-to-Fine open-set Classification (CFC) framework that leverages large language models (LLMs) for graph datasets. CFC consists of three key components: a coarse classifier that uses LLM prompts for OOD detection and outlier label generation, a GNN-based fine classifier trained with OOD samples identified by the coarse classifier for enhanced OOD detection and ID classification, and refined OOD classification achieved through LLM prompts and post-processed OOD labels. Unlike methods that rely on synthetic or auxiliary OOD samples, CFC employs semantic OOD instances that are genuinely out-of-distribution based on their inherent meaning, improving interpretability and practical utility. Experimental results show that CFC improves OOD detection by ten percent over state-of-the-art methods on graph and text domains and achieves up to seventy percent accuracy in OOD classification on graph datasets. 

---
# The Evolution of Reranking Models in Information Retrieval: From Heuristic Methods to Large Language Models 

**Authors**: Tejul Pandit, Sakshi Mahendru, Meet Raval, Dhvani Upadhyay  

**Link**: [PDF](https://arxiv.org/pdf/2512.16236)  

**Abstract**: Reranking is a critical stage in contemporary information retrieval (IR) systems, improving the relevance of the user-presented final results by honing initial candidate sets. This paper is a thorough guide to examine the changing reranker landscape and offer a clear view of the advancements made in reranking methods. We present a comprehensive survey of reranking models employed in IR, particularly within modern Retrieval Augmented Generation (RAG) pipelines, where retrieved documents notably influence output quality.
We embark on a chronological journey through the historical trajectory of reranking techniques, starting with foundational approaches, before exploring the wide range of sophisticated neural network architectures such as cross-encoders, sequence-generation models like T5, and Graph Neural Networks (GNNs) utilized for structural information. Recognizing the computational cost of advancing neural rerankers, we analyze techniques for enhancing efficiency, notably knowledge distillation for creating competitive, lighter alternatives. Furthermore, we map the emerging territory of integrating Large Language Models (LLMs) in reranking, examining novel prompting strategies and fine-tuning tactics. This survey seeks to elucidate the fundamental ideas, relative effectiveness, computational features, and real-world trade-offs of various reranking strategies. The survey provides a structured synthesis of the diverse reranking paradigms, highlighting their underlying principles and comparative strengths and weaknesses. 

---
# AI-Powered Dermatological Diagnosis: From Interpretable Models to Clinical Implementation A Comprehensive Framework for Accessible and Trustworthy Skin Disease Detection 

**Authors**: Satya Narayana Panda, Vaishnavi Kukkala, Spandana Iyer  

**Link**: [PDF](https://arxiv.org/pdf/2512.16235)  

**Abstract**: Dermatological conditions affect 1.9 billion people globally, yet accurate diagnosis remains challenging due to limited specialist availability and complex clinical presentations. Family history significantly influences skin disease susceptibility and treatment responses, but is often underutilized in diagnostic processes. This research addresses the critical question: How can AI-powered systems integrate family history data with clinical imaging to enhance dermatological diagnosis while supporting clinical trial validation and real-world implementation?
We developed a comprehensive multi-modal AI framework that combines deep learning-based image analysis with structured clinical data, including detailed family history patterns. Our approach employs interpretable convolutional neural networks integrated with clinical decision trees that incorporate hereditary risk factors. The methodology includes prospective clinical trials across diverse healthcare settings to validate AI-assisted diagnosis against traditional clinical assessment.
In this work, validation was conducted with healthcare professionals to assess AI-assisted outputs against clinical expectations; prospective clinical trials across diverse healthcare settings are proposed as future work. The integrated AI system demonstrates enhanced diagnostic accuracy when family history data is incorporated, particularly for hereditary skin conditions such as melanoma, psoriasis, and atopic dermatitis. Expert feedback indicates potential for improved early detection and more personalized recommendations; formal clinical trials are planned. The framework is designed for integration into clinical workflows while maintaining interpretability through explainable AI mechanisms. 

---
# An Information-Theoretic Framework for Robust Large Language Model Editing 

**Authors**: Qizhou Chen, Chengyu Wang, Taolin Zhang, Xiaofeng He  

**Link**: [PDF](https://arxiv.org/pdf/2512.16227)  

**Abstract**: Large Language Models (LLMs) have become indispensable tools in science, technology, and society, enabling transformative advances across diverse fields. However, errors or outdated information within these models can undermine their accuracy and restrict their safe deployment. Developing efficient strategies for updating model knowledge without the expense and disruption of full retraining remains a critical challenge. Current model editing techniques frequently struggle to generalize corrections beyond narrow domains, leading to unintended consequences and limiting their practical impact. Here, we introduce a novel framework for editing LLMs, grounded in information bottleneck theory. This approach precisely compresses and isolates the essential information required for generalizable knowledge correction while minimizing disruption to unrelated model behaviors. Building upon this foundation, we present the Information Bottleneck Knowledge Editor (IBKE), which leverages compact latent representations to guide gradient-based updates, enabling robust and broadly applicable model editing. We validate IBKE's effectiveness across multiple LLM architectures and standard benchmark tasks, demonstrating state-of-the-art accuracy and improved generality and specificity of edits. These findings establish a theoretically principled and practical paradigm for open-domain knowledge editing, advancing the utility and trustworthiness of LLMs in real-world applications. 

---
# Neural emulation of gravity-driven geohazard runout 

**Authors**: Lorenzo Nava, Ye Chen, Maximillian Van Wyk de Vries  

**Link**: [PDF](https://arxiv.org/pdf/2512.16221)  

**Abstract**: Predicting geohazard runout is critical for protecting lives, infrastructure and ecosystems. Rapid mass flows, including landslides and avalanches, cause several thousand deaths across a wide range of environments, often travelling many kilometres from their source. The wide range of source conditions and material properties governing these flows makes their runout difficult to anticipate, particularly for downstream communities that may be suddenly exposed to severe impacts. Accurately predicting runout at scale requires models that are both physically realistic and computationally efficient, yet existing approaches face a fundamental speed-realism trade-off. Here we train a machine learning model to predict geohazard runout across representative real world terrains. The model predicts both flow extent and deposit thickness with high accuracy and 100 to 10,000 times faster computation than numerical solvers. It is trained on over 100,000 numerical simulations across over 10,000 real world digital elevation model chips and reproduces key physical behaviours, including avulsion and deposition patterns, while generalizing across different flow types, sizes and landscapes. Our results demonstrate that neural emulation enables rapid, spatially resolved runout prediction across diverse real world terrains, opening new opportunities for disaster risk reduction and impact-based forecasting. These results highlight neural emulation as a promising pathway for extending physically realistic geohazard modelling to spatial and temporal scales relevant for large scale early warning systems. 

---
# Open Ad-hoc Categorization with Contextualized Feature Learning 

**Authors**: Zilin Wang, Sangwoo Mo, Stella X. Yu, Sima Behpour, Liu Ren  

**Link**: [PDF](https://arxiv.org/pdf/2512.16202)  

**Abstract**: Adaptive categorization of visual scenes is essential for AI agents to handle changing tasks. Unlike fixed common categories for plants or animals, ad-hoc categories are created dynamically to serve specific goals. We study open ad-hoc categorization: Given a few labeled exemplars and abundant unlabeled data, the goal is to discover the underlying context and to expand ad-hoc categories through semantic extension and visual clustering around it.
Building on the insight that ad-hoc and common categories rely on similar perceptual mechanisms, we propose OAK, a simple model that introduces a small set of learnable context tokens at the input of a frozen CLIP and optimizes with both CLIP's image-text alignment objective and GCD's visual clustering objective.
On Stanford and Clevr-4 datasets, OAK achieves state-of-the-art in accuracy and concept discovery across multiple categorizations, including 87.4% novel accuracy on Stanford Mood, surpassing CLIP and GCD by over 50%. Moreover, OAK produces interpretable saliency maps, focusing on hands for Action, faces for Mood, and backgrounds for Location, promoting transparency and trust while enabling adaptive and generalizable categorization. 

---
# Towards Closing the Domain Gap with Event Cameras 

**Authors**: M. Oltan Sevinc, Liao Wu, Francisco Cruz  

**Link**: [PDF](https://arxiv.org/pdf/2512.16178)  

**Abstract**: Although traditional cameras are the primary sensor for end-to-end driving, their performance suffers greatly when the conditions of the data they were trained on does not match the deployment environment, a problem known as the domain gap. In this work, we consider the day-night lighting difference domain gap. Instead of traditional cameras we propose event cameras as a potential alternative which can maintain performance across lighting condition domain gaps without requiring additional adjustments. Our results show that event cameras maintain more consistent performance across lighting conditions, exhibiting domain-shift penalties that are generally comparable to or smaller than grayscale frames and provide superior baseline performance in cross-domain scenarios. 

---
# Ev-Trust: A Strategy Equilibrium Trust Mechanism for Evolutionary Games in LLM-Based Multi-Agent Services 

**Authors**: Shiduo Yang, Jiye Wang, Jiayu Qin, Jianbin Li, Yu Wang, Yuanhe Zhao, Kenan Guo  

**Link**: [PDF](https://arxiv.org/pdf/2512.16167)  

**Abstract**: The rapid evolution of the Web toward an agent-centric paradigm, driven by large language models (LLMs), has enabled autonomous agents to reason, plan, and interact in complex decentralized environments. However, the openness and heterogeneity of LLM-based multi-agent systems also amplify the risks of deception, fraud, and misinformation, posing severe challenges to trust establishment and system robustness. To address this issue, we propose Ev-Trust, a strategy-equilibrium trust mechanism grounded in evolutionary game theory. This mechanism integrates direct trust, indirect trust, and expected revenue into a dynamic feedback structure that guides agents' behavioral evolution toward equilibria. Within a decentralized "Request-Response-Payment-Evaluation" service framework, Ev-Trust enables agents to adaptively adjust strategies, naturally excluding malicious participants while reinforcing high-quality collaboration. Furthermore, our theoretical derivation based on replicator dynamics equations proves the existence and stability of local evolutionary equilibria. Experimental results indicate that our approach effectively reflects agent trustworthiness in LLM-driven open service interaction scenarios, reduces malicious strategies, and increases collective revenue. We hope Ev-Trust can provide a new perspective on trust modeling for the agentic service web in group evolutionary game scenarios. 

---
# C-DGPA: Class-Centric Dual-Alignment Generative Prompt Adaptation 

**Authors**: Chao Li, Dasha Hu, Chengyang Li, Yuming Jiang, Yuncheng Shen  

**Link**: [PDF](https://arxiv.org/pdf/2512.16164)  

**Abstract**: Unsupervised Domain Adaptation transfers knowledge from a labeled source domain to an unlabeled target domain. Directly deploying Vision-Language Models (VLMs) with prompt tuning in downstream UDA tasks faces the signifi cant challenge of mitigating domain discrepancies. Existing prompt-tuning strategies primarily align marginal distribu tion, but neglect conditional distribution discrepancies, lead ing to critical issues such as class prototype misalignment and degraded semantic discriminability. To address these lim itations, the work proposes C-DGPA: Class-Centric Dual Alignment Generative Prompt Adaptation. C-DGPA syner gistically optimizes marginal distribution alignment and con ditional distribution alignment through a novel dual-branch architecture. The marginal distribution alignment branch em ploys a dynamic adversarial training framework to bridge marginal distribution discrepancies. Simultaneously, the con ditional distribution alignment branch introduces a Class Mapping Mechanism (CMM) to align conditional distribu tion discrepancies by standardizing semantic prompt under standing and preventing source domain over-reliance. This dual alignment strategy effectively integrates domain knowl edge into prompt learning via synergistic optimization, ensur ing domain-invariant and semantically discriminative repre sentations. Extensive experiments on OfficeHome, Office31, and VisDA-2017 validate the superiority of C-DGPA. It achieves new state-of-the-art results on all benchmarks. 

---
# Decoding Fake Narratives in Spreading Hateful Stories: A Dual-Head RoBERTa Model with Multi-Task Learning 

**Authors**: Yash Bhaskar, Sankalp Bahad, Parameswari Krishnamurthy  

**Link**: [PDF](https://arxiv.org/pdf/2512.16147)  

**Abstract**: Social media platforms, while enabling global connectivity, have become hubs for the rapid spread of harmful content, including hate speech and fake narratives \cite{davidson2017automated, shu2017fake}. The Faux-Hate shared task focuses on detecting a specific phenomenon: the generation of hate speech driven by fake narratives, termed Faux-Hate. Participants are challenged to identify such instances in code-mixed Hindi-English social media text. This paper describes our system developed for the shared task, addressing two primary sub-tasks: (a) Binary Faux-Hate detection, involving fake and hate speech classification, and (b) Target and Severity prediction, categorizing the intended target and severity of hateful content. Our approach combines advanced natural language processing techniques with domain-specific pretraining to enhance performance across both tasks. The system achieved competitive results, demonstrating the efficacy of leveraging multi-task learning for this complex problem. 

---
# MRG-R1: Reinforcement Learning for Clinically Aligned Medical Report Generation 

**Authors**: Pengyu Wang, Shuchang Ye, Usman Naseem, Jinman Kim  

**Link**: [PDF](https://arxiv.org/pdf/2512.16145)  

**Abstract**: Medical report generation (MRG) aims to automatically derive radiology-style reports from medical images to aid in clinical decision-making. However, existing methods often generate text that mimics the linguistic style of radiologists but fails to guarantee clinical correctness, because they are trained on token-level objectives which focus on word-choice and sentence structure rather than actual medical accuracy. We propose a semantic-driven reinforcement learning (SRL) method for medical report generation, adopted on a large vision-language model (LVLM). SRL adopts Group Relative Policy Optimization (GRPO) to encourage clinical-correctness-guided learning beyond imitation of language style. Specifically, we optimise a report-level reward: a margin-based cosine similarity (MCCS) computed between key radiological findings extracted from generated and reference reports, thereby directly aligning clinical-label agreement and improving semantic correctness. A lightweight reasoning format constraint further guides the model to generate structured "thinking report" outputs. We evaluate Medical Report Generation with Sematic-driven Reinforment Learning (MRG-R1), on two datasets: IU X-Ray and MIMIC-CXR using clinical efficacy (CE) metrics. MRG-R1 achieves state-of-the-art performance with CE-F1 51.88 on IU X-Ray and 40.39 on MIMIC-CXR. We found that the label-semantic reinforcement is better than conventional token-level supervision. These results indicate that optimizing a clinically grounded, report-level reward rather than token overlap,meaningfully improves clinical correctness. This work is a prior to explore semantic-reinforcement in supervising medical correctness in medical Large vision-language model(Med-LVLM) training. 

---
# INTELLECT-3: Technical Report 

**Authors**: Prime Intellect Team, Mika Senghaas, Fares Obeid, Sami Jaghouar, William Brown, Jack Min Ong, Daniel Auras, Matej Sirovatka, Jannik Straube, Andrew Baker, Sebastian Müller, Justus Mattern, Manveer Basra, Aiman Ismail, Dominik Scherm, Cooper Miller, Ameen Patel, Simon Kirsten, Mario Sieg, Christian Reetz, Kemal Erdem, Vincent Weisser, Johannes Hagemann  

**Link**: [PDF](https://arxiv.org/pdf/2512.16144)  

**Abstract**: We present INTELLECT-3, a 106B-parameter Mixture-of-Experts model (12B active) trained with large-scale reinforcement learning on our end-to-end RL infrastructure stack. INTELLECT-3 achieves state of the art performance for its size across math, code, science and reasoning benchmarks, outperforming many larger frontier models. We open-source the model together with the full infrastructure stack used to create it, including RL frameworks, complete recipe, and a wide collection of environments, built with the verifiers library, for training and evaluation from our Environments Hub community platform. Built for this effort, we introduce prime-rl, an open framework for large-scale asynchronous reinforcement learning, which scales seamlessly from a single node to thousands of GPUs, and is tailored for agentic RL with first-class support for multi-turn interactions and tool use. Using this stack, we run both SFT and RL training on top of the GLM-4.5-Air-Base model, scaling RL training up to 512 H200s with high training efficiency. 

---
# Autoencoder-based Denoising Defense against Adversarial Attacks on Object Detection 

**Authors**: Min Geun Song, Gang Min Kim, Woonmin Kim, Yongsik Kim, Jeonghyun Sim, Sangbeom Park, Huy Kang Kim  

**Link**: [PDF](https://arxiv.org/pdf/2512.16123)  

**Abstract**: Deep learning-based object detection models play a critical role in real-world applications such as autonomous driving and security surveillance systems, yet they remain vulnerable to adversarial examples. In this work, we propose an autoencoder-based denoising defense to recover object detection performance degraded by adversarial perturbations. We conduct adversarial attacks using Perlin noise on vehicle-related images from the COCO dataset, apply a single-layer convolutional autoencoder to remove the perturbations, and evaluate detection performance using YOLOv5. Our experiments demonstrate that adversarial attacks reduce bbox mAP from 0.2890 to 0.1640, representing a 43.3% performance degradation. After applying the proposed autoencoder defense, bbox mAP improves to 0.1700 (3.7% recovery) and bbox mAP@50 increases from 0.2780 to 0.3080 (10.8% improvement). These results indicate that autoencoder-based denoising can provide partial defense against adversarial attacks without requiring model retraining. 

---
# ModelTables: A Corpus of Tables about Models 

**Authors**: Zhengyuan Dong, Victor Zhong, Renée J. Miller  

**Link**: [PDF](https://arxiv.org/pdf/2512.16106)  

**Abstract**: We present ModelTables, a benchmark of tables in Model Lakes that captures the structured semantics of performance and configuration tables often overlooked by text only retrieval. The corpus is built from Hugging Face model cards, GitHub READMEs, and referenced papers, linking each table to its surrounding model and publication context. Compared with open data lake tables, model tables are smaller yet exhibit denser inter table relationships, reflecting tightly coupled model and benchmark evolution. The current release covers over 60K models and 90K tables. To evaluate model and table relatedness, we construct a multi source ground truth using three complementary signals: (1) paper citation links, (2) explicit model card links and inheritance, and (3) shared training datasets. We present one extensive empirical use case for the benchmark which is table search. We compare canonical Data Lake search operators (unionable, joinable, keyword) and Information Retrieval baselines (dense, sparse, hybrid retrieval) on this benchmark. Union based semantic table retrieval attains 54.8 % P@1 overall (54.6 % on citation, 31.3 % on inheritance, 30.6 % on shared dataset signals); table based dense retrieval reaches 66.5 % P@1, and metadata hybrid retrieval achieves 54.1 %. This evaluation indicates clear room for developing better table search methods. By releasing ModelTables and its creation protocol, we provide the first large scale benchmark of structured data describing AI model. Our use case of table discovery in Model Lakes, provides intuition and evidence for developing more accurate semantic retrieval, structured comparison, and principled organization of structured model knowledge. Source code, data, and other artifacts have been made available at this https URL. 

---
# AIMM: An AI-Driven Multimodal Framework for Detecting Social-Media-Influenced Stock Market Manipulation 

**Authors**: Sandeep Neela  

**Link**: [PDF](https://arxiv.org/pdf/2512.16103)  

**Abstract**: Market manipulation now routinely originates from coordinated social media campaigns, not isolated trades. Retail investors, regulators, and brokerages need tools that connect online narratives and coordination patterns to market behavior. We present AIMM, an AI-driven framework that fuses Reddit activity, bot and coordination indicators, and OHLCV market features into a daily AIMM Manipulation Risk Score for each ticker.
The system uses a parquet-native pipeline with a Streamlit dashboard that allows analysts to explore suspicious windows, inspect underlying posts and price action, and log model outputs over time. Due to Reddit API restrictions, we employ calibrated synthetic social features matching documented event characteristics; market data (OHLCV) uses real historical data from Yahoo Finance. This release makes three contributions. First, we build the AIMM Ground Truth dataset (AIMM-GT): 33 labeled ticker-days spanning eight equities, drawing from SEC enforcement actions, community-verified manipulation cases, and matched normal controls. Second, we implement forward-walk evaluation and prospective prediction logging for both retrospective and deployment-style assessment. Third, we analyze lead times and show that AIMM flagged GME 22 days before the January 2021 squeeze peak.
The current labeled set is small (33 ticker-days, 3 positive events), but results show preliminary discriminative capability and early warnings for the GME incident. We release the code, dataset schema, and dashboard design to support research on social media-driven market surveillance. 

---
# TurboDiffusion: Accelerating Video Diffusion Models by 100-200 Times 

**Authors**: Jintao Zhang, Kaiwen Zheng, Kai Jiang, Haoxu Wang, Ion Stoica, Joseph E. Gonzalez, Jianfei Chen, Jun Zhu  

**Link**: [PDF](https://arxiv.org/pdf/2512.16093)  

**Abstract**: We introduce TurboDiffusion, a video generation acceleration framework that can speed up end-to-end diffusion generation by 100-200x while maintaining video quality. TurboDiffusion mainly relies on several components for acceleration: (1) Attention acceleration: TurboDiffusion uses low-bit SageAttention and trainable Sparse-Linear Attention (SLA) to speed up attention computation. (2) Step distillation: TurboDiffusion adopts rCM for efficient step distillation. (3) W8A8 quantization: TurboDiffusion quantizes model parameters and activations to 8 bits to accelerate linear layers and compress the model. In addition, TurboDiffusion incorporates several other engineering optimizations.
We conduct experiments on the Wan2.2-I2V-14B-720P, Wan2.1-T2V-1.3B-480P, Wan2.1-T2V-14B-720P, and Wan2.1-T2V-14B-480P models. Experimental results show that TurboDiffusion achieves 100-200x speedup for video generation even on a single RTX 5090 GPU, while maintaining comparable video quality. The GitHub repository, which includes model checkpoints and easy-to-use code, is available at this https URL. 

---
# LAPX: Lightweight Hourglass Network with Global Context 

**Authors**: Haopeng Zhao, Marsha Mariya Kappan, Mahdi Bamdad, Francisco Cruz  

**Link**: [PDF](https://arxiv.org/pdf/2512.16089)  

**Abstract**: Human pose estimation is a crucial task in computer vision. Methods that have SOTA (State-of-the-Art) accuracy, often involve a large number of parameters and incur substantial computational cost. Many lightweight variants have been proposed to reduce the model size and computational cost of them. However, several of these methods still contain components that are not well suited for efficient deployment on edge devices. Moreover, models that primarily emphasize inference speed on edge devices often suffer from limited accuracy due to their overly simplified designs. To address these limitations, we propose LAPX, an Hourglass network with self-attention that captures global contextual information, based on previous work, LAP. In addition to adopting the self-attention module, LAPX advances the stage design and refine the lightweight attention modules. It achieves competitive results on two benchmark datasets, MPII and COCO, with only 2.3M parameters, and demonstrates real-time performance, confirming its edge-device suitability. 

---
# Scaling Text2SQL via LLM-efficient Schema Filtering with Functional Dependency Graph Rerankers 

**Authors**: Thanh Dat Hoang, Thanh Tam Nguyen, Thanh Trung Huynh, Hongzhi Yin, Quoc Viet Hung Nguyen  

**Link**: [PDF](https://arxiv.org/pdf/2512.16083)  

**Abstract**: Most modern Text2SQL systems prompt large language models (LLMs) with entire schemas -- mostly column information -- alongside the user's question. While effective on small databases, this approach fails on real-world schemas that exceed LLM context limits, even for commercial models. The recent Spider 2.0 benchmark exemplifies this with hundreds of tables and tens of thousands of columns, where existing systems often break. Current mitigations either rely on costly multi-step prompting pipelines or filter columns by ranking them against user's question independently, ignoring inter-column structure. To scale existing systems, we introduce \toolname, an open-source, LLM-efficient schema filtering framework that compacts Text2SQL prompts by (i) ranking columns with a query-aware LLM encoder enriched with values and metadata, (ii) reranking inter-connected columns via a lightweight graph transformer over functional dependencies, and (iii) selecting a connectivity-preserving sub-schema with a Steiner-tree heuristic. Experiments on real datasets show that \toolname achieves near-perfect recall and higher precision than CodeS, SchemaExP, Qwen rerankers, and embedding retrievers, while maintaining sub-second median latency and scaling to schemas with 23,000+ columns. Our source code is available at this https URL. 

---
# Evaluation of Generative Models for Emotional 3D Animation Generation in VR 

**Authors**: Kiran Chhatre, Renan Guarese, Andrii Matviienko, Christopher Peters  

**Link**: [PDF](https://arxiv.org/pdf/2512.16081)  

**Abstract**: Social interactions incorporate nonverbal signals to convey emotions alongside speech, including facial expressions and body gestures. Generative models have demonstrated promising results in creating full-body nonverbal animations synchronized with speech; however, evaluations using statistical metrics in 2D settings fail to fully capture user-perceived emotions, limiting our understanding of model effectiveness. To address this, we evaluate emotional 3D animation generative models within a Virtual Reality (VR) environment, emphasizing user-centric metrics emotional arousal realism, naturalness, enjoyment, diversity, and interaction quality in a real-time human-agent interaction scenario. Through a user study (N=48), we examine perceived emotional quality for three state of the art speech-driven 3D animation methods across two emotions happiness (high arousal) and neutral (mid arousal). Additionally, we compare these generative models against real human expressions obtained via a reconstruction-based method to assess both their strengths and limitations and how closely they replicate real human facial and body expressions. Our results demonstrate that methods explicitly modeling emotions lead to higher recognition accuracy compared to those focusing solely on speech-driven synchrony. Users rated the realism and naturalness of happy animations significantly higher than those of neutral animations, highlighting the limitations of current generative models in handling subtle emotional states. Generative models underperformed compared to reconstruction-based methods in facial expression quality, and all methods received relatively low ratings for animation enjoyment and interaction quality, emphasizing the importance of incorporating user-centric evaluations into generative model development. Finally, participants positively recognized animation diversity across all generative models. 

---
# Feasibility of Radio Frequency Based Wireless Sensing of Lead Contamination in Soil 

**Authors**: Yixuan Gao, Tanvir Ahmed, Mikhail Mohammed, Zhongqi Cheng, Rajalakshmi Nandakumar  

**Link**: [PDF](https://arxiv.org/pdf/2512.16071)  

**Abstract**: Widespread Pb (lead) contamination of urban soil significantly impacts food safety and public health and hinders city greening efforts. However, most existing technologies for measuring Pb are labor-intensive and costly. In this study, we propose SoilScanner, a radio frequency-based wireless system that can detect Pb in soils. This is based on our discovery that the propagation of different frequency band radio signals is affected differently by different salts such as NaCl and Pb(NO3)2 in the soil. In a controlled experiment, manually adding NaCl and Pb(NO3)2 in clean soil, we demonstrated that different salts reflected signals at different frequencies in distinct patterns. In addition, we confirmed the finding using uncontrolled field samples with a machine learning model. Our experiment results show that SoilScanner can classify soil samples into low-Pb and high-Pb categories (threshold at 200 ppm) with an accuracy of 72%, with no sample with > 500 ppm of Pb being misclassified. The results of this study show that it is feasible to build portable and affordable Pb detection and screening devices based on wireless technology. 

---
# A Multi-Agent Large Language Model Framework for Automated Qualitative Analysis 

**Authors**: Qidi Xu, Nuzha Amjad, Grace Giles, Alexa Cumming, De'angelo Hermesky, Alexander Wen, Min Ji Kwak, Yejin Kim  

**Link**: [PDF](https://arxiv.org/pdf/2512.16063)  

**Abstract**: Understanding patients experiences is essential for advancing patient centered care, especially in chronic diseases that require ongoing communication. However, qualitative thematic analysis, the primary approach for exploring these experiences, remains labor intensive, subjective, and difficult to scale. In this study, we developed a multi agent large language model framework that automates qualitative thematic analysis through three agents (Instructor, Thematizer, CodebookGenerator), named Collaborative Theme Identification Agent (CoTI). We applied CoTI to 12 heart failure patient interviews to analyze their perceptions of medication intensity. CoTI identified key phrases, themes, and codebook that were more similar to those of the senior investigator than both junior investigators and baseline NLP models. We also implemented CoTI into a user-facing application to enable AI human interaction in qualitative analysis. However, collaboration between CoTI and junior investigators provided only marginal gains, suggesting they may overrely on CoTI and limit their independent critical thinking. 

---
# CauSTream: Causal Spatio-Temporal Representation Learning for Streamflow Forecasting 

**Authors**: Shu Wan, Reepal Shah, John Sabo, Huan Liu, K. Selçuk Candan  

**Link**: [PDF](https://arxiv.org/pdf/2512.16046)  

**Abstract**: Streamflow forecasting is crucial for water resource management and risk mitigation. While deep learning models have achieved strong predictive performance, they often overlook underlying physical processes, limiting interpretability and generalization. Recent causal learning approaches address these issues by integrating domain knowledge, yet they typically rely on fixed causal graphs that fail to adapt to data. We propose CauStream, a unified framework for causal spatiotemporal streamflow forecasting. CauSTream jointly learns (i) a runoff causal graph among meteorological forcings and (ii) a routing graph capturing dynamic dependencies across stations. We further establish identifiability conditions for these causal structures under a nonparametric setting. We evaluate CauSTream on three major U.S. river basins across three forecasting horizons. The model consistently outperforms prior state-of-the-art methods, with performance gaps widening at longer forecast windows, indicating stronger generalization to unseen conditions. Beyond forecasting, CauSTream also learns causal graphs that capture relationships among hydrological factors and stations. The inferred structures align closely with established domain knowledge, offering interpretable insights into watershed dynamics. CauSTream offers a principled foundation for causal spatiotemporal modeling, with the potential to extend to a wide range of scientific and environmental applications. 

---
# Are We on the Right Way to Assessing LLM-as-a-Judge? 

**Authors**: Yuanning Feng, Sinan Wang, Zhengxiang Cheng, Yao Wan, Dongping Chen  

**Link**: [PDF](https://arxiv.org/pdf/2512.16041)  

**Abstract**: LLM-as-a-Judge has been widely adopted as an evaluation method and served as supervised rewards in model training. However, existing benchmarks for LLM-as-a-Judge are mainly relying on human-annotated ground truth, which introduces human bias that undermines the assessment of reliability and imposes scalability constraints. To overcome these limitations, we introduce Sage, a novel evaluation suite that assesses the quality of LLM judges without necessitating any human annotation. Inspired by axioms of rational choice theory, Sage introduces two new lenses for measuring LLM-as-a-Judge: local self-consistency (pair-wise preference stability) and global logical consistency (transitivity across a full set of preferences). We curate a dataset of 650 questions by combining structured benchmark problems with real-world user queries. Our experiments demonstrate both the stability of our metrics and their high correlation with supervised benchmarks like LLMBar and RewardBench2, confirming Sage's reliability as an evaluation suite for the robustness and accuracy of LLM-as-a-Judge. Based on Sage, we reveal that current state-of-the-art LLMs exhibit significant reliability problems when acting as judges in both scoring and pairwise settings; even the top-performing models, Gemini-2.5-Pro and GPT-5, fail to maintain consistent preferences in nearly a quarter of difficult cases. We attribute this to a new phenomenon called situational preference, which explains why explicit rubrics or criteria can help the model judge consistently across answer pairs. Our further analysis shows that finetuned LLM-as-a-Judge is a feasible method to boost performance, and the panel-based judge as well as deep reasoning can enhance the judging consistency. We also find substantial inconsistency in human judgments, which indicates that human annotation may not be a reliable gold standard. 

---
# Cross-Language Bias Examination in Large Language Models 

**Authors**: Yuxuan Liang, Marwa Mahmoud  

**Link**: [PDF](https://arxiv.org/pdf/2512.16029)  

**Abstract**: This study introduces an innovative multilingual bias evaluation framework for assessing bias in Large Language Models, combining explicit bias assessment through the BBQ benchmark with implicit bias measurement using a prompt-based Implicit Association Test. By translating the prompts and word list into five target languages, English, Chinese, Arabic, French, and Spanish, we directly compare different types of bias across languages. The results reveal substantial gaps in bias across languages used in LLMs. For example, Arabic and Spanish consistently show higher levels of stereotype bias, while Chinese and English exhibit lower levels of bias. We also identify contrasting patterns across bias types. Age shows the lowest explicit bias but the highest implicit bias, emphasizing the importance of detecting implicit biases that are undetectable with standard benchmarks. These findings indicate that LLMs vary significantly across languages and bias dimensions. This study fills a key research gap by providing a comprehensive methodology for cross-lingual bias analysis. Ultimately, our work establishes a foundation for the development of equitable multilingual LLMs, ensuring fairness and effectiveness across diverse languages and cultures. 

---
# Few-Shot Inference of Human Perceptions of Robot Performance in Social Navigation Scenarios 

**Authors**: Qiping Zhang, Nathan Tsoi, Mofeed Nagib, Hao-Tien Lewis Chiang, Marynel Vázquez  

**Link**: [PDF](https://arxiv.org/pdf/2512.16019)  

**Abstract**: Understanding how humans evaluate robot behavior during human-robot interactions is crucial for developing socially aware robots that behave according to human expectations. While the traditional approach to capturing these evaluations is to conduct a user study, recent work has proposed utilizing machine learning instead. However, existing data-driven methods require large amounts of labeled data, which limits their use in practice. To address this gap, we propose leveraging the few-shot learning capabilities of Large Language Models (LLMs) to improve how well a robot can predict a user's perception of its performance, and study this idea experimentally in social navigation tasks. To this end, we extend the SEAN TOGETHER dataset with additional real-world human-robot navigation episodes and participant feedback. Using this augmented dataset, we evaluate the ability of several LLMs to predict human perceptions of robot performance from a small number of in-context examples, based on observed spatio-temporal cues of the robot and surrounding human motion. Our results demonstrate that LLMs can match or exceed the performance of traditional supervised learning models while requiring an order of magnitude fewer labeled instances. We further show that prediction performance can improve with more in-context examples, confirming the scalability of our approach. Additionally, we investigate what kind of sensor-based information an LLM relies on to make these inferences by conducting an ablation study on the input features considered for performance prediction. Finally, we explore the novel application of personalized examples for in-context learning, i.e., drawn from the same user being evaluated, finding that they further enhance prediction accuracy. This work paves the path to improving robot behavior in a scalable manner through user-centered feedback. 

---
# Towards Fine-Tuning-Based Site Calibration for Knowledge-Guided Machine Learning: A Summary of Results 

**Authors**: Ruolei Zeng, Arun Sharma, Shuai An, Mingzhou Yang, Shengya Zhang, Licheng Liu, David Mulla, Shashi Shekhar  

**Link**: [PDF](https://arxiv.org/pdf/2512.16013)  

**Abstract**: Accurate and cost-effective quantification of the agroecosystem carbon cycle at decision-relevant scales is essential for climate mitigation and sustainable agriculture. However, both transfer learning and the exploitation of spatial variability in this field are challenging, as they involve heterogeneous data and complex cross-scale dependencies. Conventional approaches often rely on location-independent parameterizations and independent training, underutilizing transfer learning and spatial heterogeneity in the inputs, and limiting their applicability in regions with substantial variability. We propose FTBSC-KGML (Fine-Tuning-Based Site Calibration-Knowledge-Guided Machine Learning), a pretraining- and fine-tuning-based, spatial-variability-aware, and knowledge-guided machine learning framework that augments KGML-ag with a pretraining-fine-tuning process and site-specific parameters. Using a pretraining-fine-tuning process with remote-sensing GPP, climate, and soil covariates collected across multiple midwestern sites, FTBSC-KGML estimates land emissions while leveraging transfer learning and spatial heterogeneity. A key component is a spatial-heterogeneity-aware transfer-learning scheme, which is a globally pretrained model that is fine-tuned at each state or site to learn place-aware representations, thereby improving local accuracy under limited data without sacrificing interpretability. Empirically, FTBSC-KGML achieves lower validation error and greater consistency in explanatory power than a purely global model, thereby better capturing spatial variability across states. This work extends the prior SDSA-KGML framework. 

---
# Surrogate Neural Architecture Codesign Package (SNAC-Pack) 

**Authors**: Jason Weitz, Dmitri Demler, Benjamin Hawks, Nhan Tran, Javier Duarte  

**Link**: [PDF](https://arxiv.org/pdf/2512.15998)  

**Abstract**: Neural Architecture Search is a powerful approach for automating model design, but existing methods struggle to accurately optimize for real hardware performance, often relying on proxy metrics such as bit operations. We present Surrogate Neural Architecture Codesign Package (SNAC-Pack), an integrated framework that automates the discovery and optimization of neural networks focusing on FPGA deployment. SNAC-Pack combines Neural Architecture Codesign's multi-stage search capabilities with the Resource Utilization and Latency Estimator, enabling multi-objective optimization across accuracy, FPGA resource utilization, and latency without requiring time-intensive synthesis for each candidate model. We demonstrate SNAC-Pack on a high energy physics jet classification task, achieving 63.84% accuracy with resource estimation. When synthesized on a Xilinx Virtex UltraScale+ VU13P FPGA, the SNAC-Pack model matches baseline accuracy while maintaining comparable resource utilization to models optimized using traditional BOPs metrics. This work demonstrates the potential of hardware-aware neural architecture search for resource-constrained deployments and provides an open-source framework for automating the design of efficient FPGA-accelerated models. 

---
# Provably Extracting the Features from a General Superposition 

**Authors**: Allen Liu  

**Link**: [PDF](https://arxiv.org/pdf/2512.15987)  

**Abstract**: It is widely believed that complex machine learning models generally encode features through linear representations, but these features exist in superposition, making them challenging to recover. We study the following fundamental setting for learning features in superposition from black-box query access: we are given query access to a function \[ f(x)=\sum_{i=1}^n a_i\,\sigma_i(v_i^\top x), \] where each unit vector $v_i$ encodes a feature direction and $\sigma_i:\mathbb{R} \rightarrow \mathbb{R}$ is an arbitrary response function and our goal is to recover the $v_i$ and the function $f$.
In learning-theoretic terms, superposition refers to the overcomplete regime, when the number of features is larger than the underlying dimension (i.e. $n > d$), which has proven especially challenging for typical algorithmic approaches. Our main result is an efficient query algorithm that, from noisy oracle access to $f$, identifies all feature directions whose responses are non-degenerate and reconstructs the function $f$. Crucially, our algorithm works in a significantly more general setting than all related prior results -- we allow for essentially arbitrary superpositions, only requiring that $v_i, v_j$ are not nearly identical for $i \neq j$, and general response functions $\sigma_i$. At a high level, our algorithm introduces an approach for searching in Fourier space by iteratively refining the search space to locate the hidden directions $v_i$. 

---
# Embedding Software Intent: Lightweight Java Module Recovery 

**Authors**: Yirui He, Yuqi Huai, Xingyu Chen, Joshua Garcia  

**Link**: [PDF](https://arxiv.org/pdf/2512.15980)  

**Abstract**: As an increasing number of software systems reach unprecedented scale, relying solely on code-level abstractions is becoming impractical. While architectural abstractions offer a means to manage these systems, maintaining their consistency with the actual code has been problematic. The Java Platform Module System (JPMS), introduced in Java 9, addresses this limitation by enabling explicit module specification at the language level. JPMS enhances architectural implementation through improved encapsulation and direct specification of ground-truth architectures within Java projects. Although many projects are written in Java, modularizing existing monolithic projects to JPMS modules is an open challenge due to ineffective module recovery by existing architecture recovery techniques. To address this challenge, this paper presents ClassLAR (Class-and Language model-based Architectural Recovery), a novel, lightweight, and efficient approach that recovers Java modules from monolithic Java systems using fully-qualified class names. ClassLAR leverages language models to extract semantic information from package and class names, capturing both structural and functional intent. In evaluations across 20 popular Java projects, ClassLAR outperformed all state-of-the-art techniques in architectural-level similarity metrics while achieving execution times that were 3.99 to 10.50 times faster. 

---
# OLAF: Towards Robust LLM-Based Annotation Framework in Empirical Software Engineering 

**Authors**: Mia Mohammad Imran, Tarannum Shaila Zaman  

**Link**: [PDF](https://arxiv.org/pdf/2512.15979)  

**Abstract**: Large Language Models (LLMs) are increasingly used in empirical software engineering (ESE) to automate or assist annotation tasks such as labeling commits, issues, and qualitative artifacts. Yet the reliability and reproducibility of such annotations remain underexplored. Existing studies often lack standardized measures for reliability, calibration, and drift, and frequently omit essential configuration details. We argue that LLM-based annotation should be treated as a measurement process rather than a purely automated activity. In this position paper, we outline the \textbf{Operationalization for LLM-based Annotation Framework (OLAF)}, a conceptual framework that organizes key constructs: \textit{reliability, calibration, drift, consensus, aggregation}, and \textit{transparency}. The paper aims to motivate methodological discussion and future empirical work toward more transparent and reproducible LLM-based annotation in software engineering research. 

---
# BRAID: Bounded Reasoning for Autonomous Inference and Decisions 

**Authors**: Armağan Amcalar, Eyup Cinar  

**Link**: [PDF](https://arxiv.org/pdf/2512.15959)  

**Abstract**: Large Language Models (LLMs) exhibit nonlinear relationships between performance, cost, and token usage. This paper presents a quantitative study on structured prompting using BRAID (Bounded Reasoning for Au tonomous Inference and Decisions) across multiple GPT model tiers, eval uated on the AdvancedIF, GSM-Hard, and the SCALE MultiChallenge benchmark datasets. BRAID introduces a bounded reasoning framework using Mermaid-based instruction graphs that enable models to reason struc turally rather than through unbounded natural-language token expansion. We show that structured machine-readable prompts substantially increase reasoning accuracy and cost efficiency for agents in production systems. The findings establish BRAID as an effective and scalable technique for optimizing inference efficiency in autonomous agent systems. All datasets and detailed result logs are available at this https URL. 

---
# Seeing is Believing (and Predicting): Context-Aware Multi-Human Behavior Prediction with Vision Language Models 

**Authors**: Utsav Panchal, Yuchen Liu, Luigi Palmieri, Ilche Georgievski, Marco Aiello  

**Link**: [PDF](https://arxiv.org/pdf/2512.15957)  

**Abstract**: Accurately predicting human behaviors is crucial for mobile robots operating in human-populated environments. While prior research primarily focuses on predicting actions in single-human scenarios from an egocentric view, several robotic applications require understanding multiple human behaviors from a third-person perspective. To this end, we present CAMP-VLM (Context-Aware Multi-human behavior Prediction): a Vision Language Model (VLM)-based framework that incorporates contextual features from visual input and spatial awareness from scene graphs to enhance prediction of humans-scene interactions. Due to the lack of suitable datasets for multi-human behavior prediction from an observer view, we perform fine-tuning of CAMP-VLM with synthetic human behavior data generated by a photorealistic simulator, and evaluate the resulting models on both synthetic and real-world sequences to assess their generalization capabilities. Leveraging Supervised Fine-Tuning (SFT) and Direct Preference Optimization (DPO), CAMP-VLM outperforms the best-performing baseline by up to 66.9% in prediction accuracy. 

---
# SALVE: Sparse Autoencoder-Latent Vector Editing for Mechanistic Control of Neural Networks 

**Authors**: Vegard Flovik  

**Link**: [PDF](https://arxiv.org/pdf/2512.15938)  

**Abstract**: Deep neural networks achieve impressive performance but remain difficult to interpret and control. We present SALVE (Sparse Autoencoder-Latent Vector Editing), a unified "discover, validate, and control" framework that bridges mechanistic interpretability and model editing. Using an $\ell_1$-regularized autoencoder, we learn a sparse, model-native feature basis without supervision. We validate these features with Grad-FAM, a feature-level saliency mapping method that visually grounds latent features in input data. Leveraging the autoencoder's structure, we perform precise and permanent weight-space interventions, enabling continuous modulation of both class-defining and cross-class features. We further derive a critical suppression threshold, $\alpha_{crit}$, quantifying each class's reliance on its dominant feature, supporting fine-grained robustness diagnostics. Our approach is validated on both convolutional (ResNet-18) and transformer-based (ViT-B/16) models, demonstrating consistent, interpretable control over their behavior. This work contributes a principled methodology for turning feature discovery into actionable model edits, advancing the development of transparent and controllable AI systems. 

---
# Scalable Agentic Reasoning for Designing Biologics Targeting Intrinsically Disordered Proteins 

**Authors**: Matthew Sinclair, Moeen Meigooni, Archit Vasan, Ozan Gokdemir, Xinran Lian, Heng Ma, Yadu Babuji, Alexander Brace, Khalid Hossain, Carlo Siebenschuh, Thomas Brettin, Kyle Chard, Christopher Henry, Venkatram Vishwanath, Rick L. Stevens, Ian T. Foster, Arvind Ramanathan  

**Link**: [PDF](https://arxiv.org/pdf/2512.15930)  

**Abstract**: Intrinsically disordered proteins (IDPs) represent crucial therapeutic targets due to their significant role in disease -- approximately 80\% of cancer-related proteins contain long disordered regions -- but their lack of stable secondary/tertiary structures makes them "undruggable". While recent computational advances, such as diffusion models, can design high-affinity IDP binders, translating these to practical drug discovery requires autonomous systems capable of reasoning across complex conformational ensembles and orchestrating diverse computational tools at this http URL address this challenge, we designed and implemented StructBioReasoner, a scalable multi-agent system for designing biologics that can be used to target IDPs. StructBioReasoner employs a novel tournament-based reasoning framework where specialized agents compete to generate and refine therapeutic hypotheses, naturally distributing computational load for efficient exploration of the vast design space. Agents integrate domain knowledge with access to literature synthesis, AI-structure prediction, molecular simulations, and stability analysis, coordinating their execution on HPC infrastructure via an extensible federated agentic middleware, Academy. We benchmark StructBioReasoner across Der f 21 and NMNAT-2 and demonstrate that over 50\% of 787 designed and validated candidates for Der f 21 outperformed the human-designed reference binders from literature, in terms of improved binding free energy. For the more challenging NMNAT-2 protein, we identified three binding modes from 97,066 binders, including the well-studied NMNAT2:p53 interface. Thus, StructBioReasoner lays the groundwork for agentic reasoning systems for IDP therapeutic discovery on Exascale platforms. 

---
# Social Story Frames: Contextual Reasoning about Narrative Intent and Reception 

**Authors**: Joel Mire, Maria Antoniak, Steven R. Wilson, Zexin Ma, Achyutarama R. Ganti, Andrew Piper, Maarten Sap  

**Link**: [PDF](https://arxiv.org/pdf/2512.15925)  

**Abstract**: Reading stories evokes rich interpretive, affective, and evaluative responses, such as inferences about narrative intent or judgments about characters. Yet, computational models of reader response are limited, preventing nuanced analyses. To address this gap, we introduce SocialStoryFrames, a formalism for distilling plausible inferences about reader response, such as perceived author intent, explanatory and predictive reasoning, affective responses, and value judgments, using conversational context and a taxonomy grounded in narrative theory, linguistic pragmatics, and psychology. We develop two models, SSF-Generator and SSF-Classifier, validated through human surveys (N=382 participants) and expert annotations, respectively. We conduct pilot analyses to showcase the utility of the formalism for studying storytelling at scale. Specifically, applying our models to SSF-Corpus, a curated dataset of 6,140 social media stories from diverse contexts, we characterize the frequency and interdependence of storytelling intents, and we compare and contrast narrative practices (and their diversity) across communities. By linking fine-grained, context-sensitive modeling with a generic taxonomy of reader responses, SocialStoryFrames enable new research into storytelling in online communities. 

---
# VET Your Agent: Towards Host-Independent Autonomy via Verifiable Execution Traces 

**Authors**: Artem Grigor, Christian Schroeder de Witt, Simon Birnbach, Ivan Martinovic  

**Link**: [PDF](https://arxiv.org/pdf/2512.15892)  

**Abstract**: Recent advances in large language models (LLMs) have enabled a new generation of autonomous agents that operate over sustained periods and manage sensitive resources on behalf of users. Trusted for their ability to act without direct oversight, such agents are increasingly considered in high-stakes domains including financial management, dispute resolution, and governance. Yet in practice, agents execute on infrastructure controlled by a host, who can tamper with models, inputs, or outputs, undermining any meaningful notion of autonomy.
We address this gap by introducing VET (Verifiable Execution Traces), a formal framework that achieves host-independent authentication of agent outputs and takes a step toward host-independent autonomy. Central to VET is the Agent Identity Document (AID), which specifies an agent's configuration together with the proof systems required for verification. VET is compositional: it supports multiple proof mechanisms, including trusted hardware, succinct cryptographic proofs, and notarized TLS transcripts (Web Proofs).
We implement VET for an API-based LLM agent and evaluate our instantiation on realistic workloads. We find that for today's black-box, secret-bearing API calls, Web Proofs appear to be the most practical choice, with overhead typically under 3$\times$ compared to direct API calls, while for public API calls, a lower-overhead TEE Proxy is often sufficient. As a case study, we deploy a verifiable trading agent that produces proofs for each decision and composes Web Proofs with a TEE Proxy. Our results demonstrate that practical, host-agnostic authentication is already possible with current technology, laying the foundation for future systems that achieve full host-independent autonomy. 

---
# Dynamical Mechanisms for Coordinating Long-term Working Memory Based on the Precision of Spike-timing in Cortical Neurons 

**Authors**: Terrence J. Sejnowski  

**Link**: [PDF](https://arxiv.org/pdf/2512.15891)  

**Abstract**: In the last century, most sensorimotor studies of cortical neurons relied on average firing rates. Rate coding is efficient for fast sensorimotor processing that occurs within a few seconds. Much less is known about long-term working memory with a time scale of hours (Ericsson and Kintsch, 1995). The discovery of the millisecond precision of spike initiation in cortical neurons was unexpected (Mainen and Sejnowski, 1995). Even more striking was the precision of spiking in vivo, in response to rapidly fluctuating sensory inputs, suggesting that neural circuits could, in principle, preserve and manipulate sensory information through spike timing. It could support spike-timing-dependent plasticity (STDP), which is triggered by the relative timing of spikes between presynaptic and postsynaptic neurons in the millisecond range. What spike-timing mechanisms could regulate STDP in vivo? Cortical traveling waves have been observed across many frequency bands with high temporal precision. Traveling waves have wave fronts that could link spike timing to STDP. As a wave front passes through a cortical column, excitatory synapses on the dendrites of both pyramidal and basket cells are synchronously stimulated. Inhibitory basket cells form a calyx on pyramidal cell bodies, and inhibitory rebound following a strong transient hyperpolarization can trigger a backpropagating action potential, which arrives shortly after the excitatory inputs on pyramidal dendrites. STDP activated in this way could persist for hours, creating a second-tier network. This temporary network could support long-term working memory, a cognitive network riding above the long-term sensorimotor network. On their own, traveling waves and STDP have not yet yielded new insights into cortical function. Together, they could be responsible for how we think (Sejnowski, 2025). 

---
# Seeing Beyond Words: Self-Supervised Visual Learning for Multimodal Large Language Models 

**Authors**: Davide Caffagni, Sara Sarto, Marcella Cornia, Lorenzo Baraldi, Pier Luigi Dovesi, Shaghayegh Roohi, Mark Granroth-Wilding, Rita Cucchiara  

**Link**: [PDF](https://arxiv.org/pdf/2512.15885)  

**Abstract**: Multimodal Large Language Models (MLLMs) have recently demonstrated impressive capabilities in connecting vision and language, yet their proficiency in fundamental visual reasoning tasks remains limited. This limitation can be attributed to the fact that MLLMs learn visual understanding primarily from textual descriptions, which constitute a subjective and inherently incomplete supervisory signal. Furthermore, the modest scale of multimodal instruction tuning compared to massive text-only pre-training leads MLLMs to overfit language priors while overlooking visual details. To address these issues, we introduce JARVIS, a JEPA-inspired framework for self-supervised visual enhancement in MLLMs. Specifically, we integrate the I-JEPA learning paradigm into the standard vision-language alignment pipeline of MLLMs training. Our approach leverages frozen vision foundation models as context and target encoders, while training the predictor, implemented as the early layers of an LLM, to learn structural and semantic regularities from images without relying exclusively on language supervision. Extensive experiments on standard MLLM benchmarks show that JARVIS consistently improves performance on vision-centric benchmarks across different LLM families, without degrading multimodal reasoning abilities. Our source code is publicly available at: this https URL. 

---
# Optimizing Agentic Language Model Inference via Speculative Tool Calls 

**Authors**: Daniel Nichols, Prajwal Singhania, Charles Jekel, Abhinav Bhatele, Harshitha Menon  

**Link**: [PDF](https://arxiv.org/pdf/2512.15834)  

**Abstract**: Language models (LMs) are becoming increasingly dependent on external tools. LM-based agentic frameworks frequently interact with their environment via such tools to search files, run code, call APIs, etc. Further, modern reasoning-based LMs use tools such as web search and Python code execution to enhance their reasoning capabilities. While tools greatly improve the capabilities of LMs, they also introduce performance bottlenecks during the inference process. In this paper, we introduce novel systems optimizations to address such performance bottlenecks by speculating tool calls and forcing sequences to remain resident in the inference engine to minimize overheads. Our optimizations lead to throughput improvements of several hundred tokens per second when hosting inference for LM agents. We provide a theoretical analysis of our algorithms to provide insights into speculation configurations that will yield the best performance. Further, we recommend a new "tool cache" API endpoint to enable LM providers to easily adopt these optimizations. 

---
# Human-like Working Memory from Artificial Intrinsic Plasticity Neurons 

**Authors**: Jingli Liu, Huannan Zheng, Bohao Zou, Kezhou Yang  

**Link**: [PDF](https://arxiv.org/pdf/2512.15829)  

**Abstract**: Working memory enables the brain to integrate transient information for rapid decision-making. Artificial networks typically replicate this via recurrent or parallel architectures, yet incur high energy costs and noise sensitivity. Here we report IPNet, a hardware-software co-designed neuromorphic architecture realizing human-like working memory via neuronal intrinsic plasticity. Exploiting Joule-heating dynamics of Magnetic Tunnel Junctions (MTJs), IPNet physically emulates biological memory volatility. The memory behavior of the proposed architecture shows similar trends in n-back, free recall and memory interference tasks to that of reported human subjects. Implemented exclusively with MTJ neurons, the architecture with human-like working memory achieves 99.65% accuracy on 11-class DVS gesture datasets and maintains 99.48% on a novel 22-class time-reversed benchmark, outperforming RNN, LSTM, and 2+1D CNN baselines sharing identical backbones. For autonomous driving (DDD-20), IPNet reduces steering prediction error by 14.4% compared to ResNet-LSTM. Architecturally, we identify a 'Memory-at-the-Frontier' effect where performance is maximized at the sensing interface, validating a bio-plausible near-sensor processing paradigm. Crucially, all results rely on raw parameters from fabricated devices without optimization. Hardware-in-the-loop validation confirms the system's physical realizability. Separately, energy analysis reveals a reduction in memory power of 2,874x compared to LSTMs and 90,920x versus parallel 3D-CNNs. This capacitor-free design enables a compact ~1.5um2 footprint (28 nm CMOS): a >20-fold reduction over standard LIF neurons. Ultimately, we demonstrate that instantiating human-like working memory via intrinsic neuronal plasticity endows neural networks with the dual biological advantages of superior dynamic vision processing and minimal metabolic cost. 

---
# A Neurosymbolic Approach to Loop Invariant Generation via Weakest Precondition Reasoning 

**Authors**: Daragh King, Vasileios Koutavas, Laura Kovacs  

**Link**: [PDF](https://arxiv.org/pdf/2512.15816)  

**Abstract**: Loop invariant generation remains a critical bottleneck in automated program verification. Recent work has begun to explore the use of Large Language Models (LLMs) in this area, yet these approaches tend to lack a reliable and structured methodology, with little reference to existing program verification theory. This paper presents NeuroInv, a neurosymbolic approach to loop invariant generation. NeuroInv comprises two key modules: (1) a neural reasoning module that leverages LLMs and Hoare logic to derive and refine candidate invariants via backward-chaining weakest precondition reasoning, and (2) a verification-guided symbolic module that iteratively repairs invariants using counterexamples from OpenJML. We evaluate NeuroInv on a comprehensive benchmark of 150 Java programs, encompassing single and multiple (sequential) loops, multiple arrays, random branching, and noisy code segments. NeuroInv achieves a $99.5\%$ success rate, substantially outperforming the other evaluated approaches. Additionally, we introduce a hard benchmark of $10$ larger multi-loop programs (with an average of $7$ loops each); NeuroInv's performance in this setting demonstrates that it can scale to more complex verification scenarios. 

---
# CodeMem: Architecting Reproducible Agents via Dynamic MCP and Procedural Memory 

**Authors**: Nishant Gaurav, Adit Akarsh, Tejas Ravishankar, Manoj Bajaj  

**Link**: [PDF](https://arxiv.org/pdf/2512.15813)  

**Abstract**: Current tool-using AI agents suffer from limited action space, context inefficiency, and probabilistic instability that makes them unsuitable for handling repetitive tasks which are otherwise reliably and efficiently tackled by agentic workflows built on platforms like n8n and Zapier. Earlier works like CodeAct, DynaSaur, Code Mode have tried to tackle the first two issues by using the whole Python language as its action space: The number of tools that the agent can call becomes infinite. Python code blocks can execute complex actions into a single step and print only relevant results which helps in keeping the context lean. However, the probabilistic instability issue still remains, as for the same task in the same environment, the agent can follow different trajectories due to the probabilistic nature of LLMs. Therefore, we need procedural memory for consistency and reliability. This paper proposes CodeMem, an architecture to implement procedural memory via code which can be used to build and run reusable agentic workflows with deterministic reliability. 

---
# Foundation Models in Biomedical Imaging: Turning Hype into Reality 

**Authors**: Amgad Muneer, Kai Zhang, Ibraheem Hamdi, Rizwan Qureshi, Muhammad Waqas, Shereen Fouad, Hazrat Ali, Syed Muhammad Anwar, Jia Wu  

**Link**: [PDF](https://arxiv.org/pdf/2512.15808)  

**Abstract**: Foundation models (FMs) are driving a prominent shift in artificial intelligence across different domains, including biomedical imaging. These models are designed to move beyond narrow pattern recognition towards emulating sophisticated clinical reasoning, understanding complex spatial relationships, and integrating multimodal data with unprecedented flexibility. However, a critical gap exists between this potential and the current reality, where the clinical evaluation and deployment of FMs are hampered by significant challenges. Herein, we critically assess the current state-of-the-art, analyzing hype by examining the core capabilities and limitations of FMs in the biomedical domain. We also provide a taxonomy of reasoning, ranging from emulated sequential logic and spatial understanding to the integration of explicit symbolic knowledge, to evaluate whether these models exhibit genuine cognition or merely mimic surface-level patterns. We argue that a critical frontier lies beyond statistical correlation, in the pursuit of causal inference, which is essential for building robust models that understand cause and effect. Furthermore, we discuss the paramount issues in deployment stemming from trustworthiness, bias, and safety, dissecting the challenges of algorithmic bias, data bias and privacy, and model hallucinations. We also draw attention to the need for more inclusive, rigorous, and clinically relevant validation frameworks to ensure their safe and ethical application. We conclude that while the vision of autonomous AI-doctors remains distant, the immediate reality is the emergence of powerful technology and assistive tools that would benefit clinical practice. The future of FMs in biomedical imaging hinges not on scale alone, but on developing hybrid, causally aware, and verifiably safe systems that augment, rather than replace, human expertise. 

---
# Edge-wise Topological Divergence Gaps: Guiding Search in Combinatorial Optimization 

**Authors**: Ilya Trofimov, Daria Voronkova, Alexander Mironenko, Anton Dmitriev, Eduard Tulchinskii, Evgeny Burnaev, Serguei Barannikov  

**Link**: [PDF](https://arxiv.org/pdf/2512.15800)  

**Abstract**: We introduce a topological feedback mechanism for the Travelling Salesman Problem (TSP) by analyzing the divergence between a tour and the minimum spanning tree (MST). Our key contribution is a canonical decomposition theorem that expresses the tour-MST gap as edge-wise topology-divergence gaps from the RTD-Lite barcode. Based on this, we develop a topological guidance for 2-opt and 3-opt heuristics that increases their performance. We carry out experiments with fine-optimization of tours obtained from heatmap-based methods, TSPLIB, and random instances. Experiments demonstrate the topology-guided optimization results in better performance and faster convergence in many cases. 

---
# Cybercrime and Computer Forensics in Epoch of Artificial Intelligence in India 

**Authors**: Sahibpreet Singh, Shikha Dhiman  

**Link**: [PDF](https://arxiv.org/pdf/2512.15799)  

**Abstract**: The integration of generative Artificial Intelligence into the digital ecosystem necessitates a critical re-evaluation of Indian criminal jurisprudence regarding computational forensics integrity. While algorithmic efficiency enhances evidence extraction, a research gap exists regarding the Digital Personal Data Protection Act, 2023's compatibility with adversarial AI threats, specifically anti-forensics and deepfakes. This study scrutinizes the AI "dual-use" dilemma, functioning as both a cyber-threat vector and forensic automation mechanism, to delineate privacy boundaries in high-stakes investigations. Employing a doctrinal legal methodology, the research synthesizes statutory analysis of the DPDP Act with global ethical frameworks (IEEE, EU) to evaluate regulatory efficacy. Preliminary results indicate that while Machine Learning offers high accuracy in pattern recognition, it introduces vulnerabilities regarding data poisoning and algorithmic bias. Findings highlight a critical tension between the Act's data minimization principles and forensic data retention requirements. Furthermore, the paper identifies that existing legal definitions inadequately encompass AI-driven "tool crimes" and "target crimes." Consequently, the research proposes a "human-centric" forensic model prioritizing explainable AI (XAI) to ensure evidence admissibility. These implications suggest that synchronizing Indian privacy statutes with international forensic standards is imperative to mitigate synthetic media risks, establishing a roadmap for future legislative amendments and technical standardization. 

---
# Explainable Ethical Assessment on Human Behaviors by Generating Conflicting Social Norms 

**Authors**: Yuxi Sun, Wei Gao, Hongzhan Lin, Jing Ma, Wenxuan Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2512.15793)  

**Abstract**: Human behaviors are often guided or constrained by social norms, which are defined as shared, commonsense rules. For example, underlying an action ``\textit{report a witnessed crime}" are social norms that inform our conduct, such as ``\textit{It is expected to be brave to report crimes}''. Current AI systems that assess valence (i.e., support or oppose) of human actions by leveraging large-scale data training not grounded on explicit norms may be difficult to explain, and thus untrustworthy. Emulating human assessors by considering social norms can help AI models better understand and predict valence. While multiple norms come into play, conflicting norms can create tension and directly influence human behavior. For example, when deciding whether to ``\textit{report a witnessed crime}'', one may balance \textit{bravery} against \textit{self-protection}. In this paper, we introduce \textit{ClarityEthic}, a novel ethical assessment approach, to enhance valence prediction and explanation by generating conflicting social norms behind human actions, which strengthens the moral reasoning capabilities of language models by using a contrastive learning strategy. Extensive experiments demonstrate that our method outperforms strong baseline approaches, and human evaluations confirm that the generated social norms provide plausible explanations for the assessment of human behaviors. 

---
# A Systematic Analysis of Biases in Large Language Models 

**Authors**: Xulang Zhang, Rui Mao, Erik Cambria  

**Link**: [PDF](https://arxiv.org/pdf/2512.15792)  

**Abstract**: Large language models (LLMs) have rapidly become indispensable tools for acquiring information and supporting human decision-making. However, ensuring that these models uphold fairness across varied contexts is critical to their safe and responsible deployment. In this study, we undertake a comprehensive examination of four widely adopted LLMs, probing their underlying biases and inclinations across the dimensions of politics, ideology, alliance, language, and gender. Through a series of carefully designed experiments, we investigate their political neutrality using news summarization, ideological biases through news stance classification, tendencies toward specific geopolitical alliances via United Nations voting patterns, language bias in the context of multilingual story completion, and gender-related affinities as revealed by responses to the World Values Survey. Results indicate that while the LLMs are aligned to be neutral and impartial, they still show biases and affinities of different types. 

---
# Evaluation of AI Ethics Tools in Language Models: A Developers' Perspective Case Stud 

**Authors**: Jhessica Silva, Diego A. B. Moreira, Gabriel O. dos Santos, Alef Ferreira, Helena Maia, Sandra Avila, Helio Pedrini  

**Link**: [PDF](https://arxiv.org/pdf/2512.15791)  

**Abstract**: In Artificial Intelligence (AI), language models have gained significant importance due to the widespread adoption of systems capable of simulating realistic conversations with humans through text generation. Because of their impact on society, developing and deploying these language models must be done responsibly, with attention to their negative impacts and possible harms. In this scenario, the number of AI Ethics Tools (AIETs) publications has recently increased. These AIETs are designed to help developers, companies, governments, and other stakeholders establish trust, transparency, and responsibility with their technologies by bringing accepted values to guide AI's design, development, and use stages. However, many AIETs lack good documentation, examples of use, and proof of their effectiveness in practice. This paper presents a methodology for evaluating AIETs in language models. Our approach involved an extensive literature survey on 213 AIETs, and after applying inclusion and exclusion criteria, we selected four AIETs: Model Cards, ALTAI, FactSheets, and Harms Modeling. For evaluation, we applied AIETs to language models developed for the Portuguese language, conducting 35 hours of interviews with their developers. The evaluation considered the developers' perspective on the AIETs' use and quality in helping to identify ethical considerations about their model. The results suggest that the applied AIETs serve as a guide for formulating general ethical considerations about language models. However, we note that they do not address unique aspects of these models, such as idiomatic expressions. Additionally, these AIETs did not help to identify potential negative impacts of models for the Portuguese language. 

---
# Toward Agentic Environments: GenAI and the Convergence of AI, Sustainability, and Human-Centric Spaces 

**Authors**: Przemek Pospieszny, Dominika P. Brodowicz  

**Link**: [PDF](https://arxiv.org/pdf/2512.15787)  

**Abstract**: In recent years, advances in artificial intelligence (AI), particularly generative AI (GenAI) and large language models (LLMs), have made human-computer interactions more frequent, efficient, and accessible across sectors ranging from banking to healthcare. AI tools embedded in digital devices support decision-making and operational management at both individual and organizational levels, including resource allocation, workflow automation, and real-time data analysis. However, the prevailing cloud-centric deployment of AI carries a substantial environmental footprint due to high computational demands. In this context, this paper introduces the concept of agentic environments, a sustainability-oriented AI framework that extends beyond reactive systems by leveraging GenAI, multi-agent systems, and edge computing to reduce the environmental impact of technology. Agentic environments enable more efficient resource use, improved quality of life, and sustainability-by-design, while simultaneously enhancing data privacy through decentralized, edge-driven solutions. Drawing on secondary research as well as primary data from focus groups and semi-structured interviews with AI professionals from leading technology companies, the paper proposes a conceptual framework for agentic environments examined through three lenses: the personal sphere, professional and commercial use, and urban operations. The findings highlight the potential of agentic environments to foster sustainable ecosystems through optimized resource utilization and strengthened data privacy. The study concludes with recommendations for edge-driven deployment models to reduce reliance on energy-intensive cloud infrastructures. 

---
# Cultural Rights and the Rights to Development in the Age of AI: Implications for Global Human Rights Governance 

**Authors**: Alexander Kriebitz, Caitlin Corrigan, Aive Pevkur, Alberto Santos Ferro, Amanda Horzyk, Dirk Brand, Dohee Kim, Dodzi Koku Hattoh, Flavia Massucci, Gilles Fayad, Kamil Strzepek, Laud Ammah, Lavina Ramkissoon, Mariette Awad, Natalia Amasiadi, Nathan C. Walker, Nicole Manger, Sophia Devlin  

**Link**: [PDF](https://arxiv.org/pdf/2512.15786)  

**Abstract**: Cultural rights and the right to development are essential norms within the wider framework of international human rights law. However, recent technological advances in artificial intelligence (AI) and adjacent digital frontier technologies pose significant challenges to the protection and realization of these rights. This owes to the increasing influence of AI systems on the creation and depiction of cultural content, affect the use and distribution of the intellectual property of individuals and communities, and influence cultural participation and expression worldwide. In addition, the growing influence of AI thus risks exacerbating preexisting economic, social and digital divides and reinforcing inequities for marginalized communities. This dynamic challenges the existing interplay between cultural rights and the right to development, and raises questions about the integration of cultural and developmental considerations into emerging AI governance frameworks. To address these challenges, the paper examines the impact of AI on both categories of rights. Conceptually, it analyzes the epistemic and normative limitations of AI with respect to cultural and developmental assumptions embedded in algorithmic design and deployment, but also individual and structural impacts of AI on both rights. On this basis, the paper identifies gaps and tensions in existing AI governance frameworks with respect to cultural rights and the right to development.
By situating cultural rights and the right to development within the broader landscape of AI and human rights, this paper contributes to the academic discourse on AI ethics, legal frameworks, and international human rights law. Finally, it outlines avenues for future research and policy development based on existing conversations in global AI governance. 

---
# Adversarial Robustness in Financial Machine Learning: Defenses, Economic Impact, and Governance Evidence 

**Authors**: Samruddhi Baviskar  

**Link**: [PDF](https://arxiv.org/pdf/2512.15780)  

**Abstract**: We evaluate adversarial robustness in tabular machine learning models used in financial decision making. Using credit scoring and fraud detection data, we apply gradient based attacks and measure impacts on discrimination, calibration, and financial risk metrics. Results show notable performance degradation under small perturbations and partial recovery through adversarial training. 

---
# Enhanced Web User Interface Design Via Cross-Device Responsiveness Assessment Using An Improved HCI-INTEGRATED DL Schemes 

**Authors**: Shrinivass Arunachalam Balasubramanian  

**Link**: [PDF](https://arxiv.org/pdf/2512.15775)  

**Abstract**: User Interface (UI) optimization is essential in the digital era to enhance user satisfaction in web environments. Nevertheless, the existing UI optimization models had overlooked the Cross-Responsiveness (CR) assessment, affecting the user interaction efficiency. Consequently, this article proposes a dynamic web UI optimization through CR assessment using Finite Exponential Continuous State Machine (FECSM) and Quokka Nonlinear Difference Swarm Optimization Algorithm (QNDSOA). Initially, the design and user interaction related information is collected as well as pre-processed for min-max normalization. Next, the Human-Computer Interaction (HCI)-based features are extracted, followed by user behaviour pattern grouping. Meanwhile, the CR assessment is done using FECSM. Then, the proposed Bidirectional Gated Luong and Mish Recurrent Unit (BiGLMRU) is used to classify the User eXperience (UX) change type, which is labelled based on the User Interface Change Prediction Index (UICPI). Lastly, a novel QNDSOA is utilized to optimize the UI design with an average fitness of 98.5632%. Feedback monitoring is done after optimal deployment. 

---
# TS-DP: Reinforcement Speculative Decoding For Temporal Adaptive Diffusion Policy Acceleration 

**Authors**: Ye Li, Jiahe Feng, Yuan Meng, Kangye Ji, Chen Tang, Xinwan Wen, Shutao Xia, Zhi Wang, Wenwu Zhu  

**Link**: [PDF](https://arxiv.org/pdf/2512.15773)  

**Abstract**: Diffusion Policy (DP) excels in embodied control but suffers from high inference latency and computational cost due to multiple iterative denoising steps. The temporal complexity of embodied tasks demands a dynamic and adaptable computation mode. Static and lossy acceleration methods, such as quantization, fail to handle such dynamic embodied tasks, while speculative decoding offers a lossless and adaptive yet underexplored alternative for DP. However, it is non-trivial to address the following challenges: how to match the base model's denoising quality at lower cost under time-varying task difficulty in embodied settings, and how to dynamically and interactively adjust computation based on task difficulty in such environments. In this paper, we propose Temporal-aware Reinforcement-based Speculative Diffusion Policy (TS-DP), the first framework that enables speculative decoding for DP with temporal adaptivity. First, to handle dynamic environments where task difficulty varies over time, we distill a Transformer-based drafter to imitate the base model and replace its costly denoising calls. Second, an RL-based scheduler further adapts to time-varying task difficulty by adjusting speculative parameters to maintain accuracy while improving efficiency. Extensive experiments across diverse embodied environments demonstrate that TS-DP achieves up to 4.17 times faster inference with over 94% accepted drafts, reaching an inference frequency of 25 Hz and enabling real-time diffusion-based control without performance degradation. 

---
# TENG++: Time-Evolving Natural Gradient for Solving PDEs With Deep Neural Nets under General Boundary Conditions 

**Authors**: Xinjie He, Chenggong Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2512.15771)  

**Abstract**: Partial Differential Equations (PDEs) are central to modeling complex systems across physical, biological, and engineering domains, yet traditional numerical methods often struggle with high-dimensional or complex problems. Physics-Informed Neural Networks (PINNs) have emerged as an efficient alternative by embedding physics-based constraints into deep learning frameworks, but they face challenges in achieving high accuracy and handling complex boundary conditions. In this work, we extend the Time-Evolving Natural Gradient (TENG) framework to address Dirichlet boundary conditions, integrating natural gradient optimization with numerical time-stepping schemes, including Euler and Heun methods, to ensure both stability and accuracy. By incorporating boundary condition penalty terms into the loss function, the proposed approach enables precise enforcement of Dirichlet constraints. Experiments on the heat equation demonstrate the superior accuracy of the Heun method due to its second-order corrections and the computational efficiency of the Euler method for simpler scenarios. This work establishes a foundation for extending the framework to Neumann and mixed boundary conditions, as well as broader classes of PDEs, advancing the applicability of neural network-based solvers for real-world problems. 

---
# Data-Chain Backdoor: Do You Trust Diffusion Models as Generative Data Supplier? 

**Authors**: Junchi Lu, Xinke Li, Yuheng Liu, Qi Alfred Chen  

**Link**: [PDF](https://arxiv.org/pdf/2512.15769)  

**Abstract**: The increasing use of generative models such as diffusion models for synthetic data augmentation has greatly reduced the cost of data collection and labeling in downstream perception tasks. However, this new data source paradigm may introduce important security concerns. This work investigates backdoor propagation in such emerging generative data supply chains, namely Data-Chain Backdoor (DCB). Specifically, we find that open-source diffusion models can become hidden carriers of backdoors. Their strong distribution-fitting ability causes them to memorize and reproduce backdoor triggers during generation, which are subsequently inherited by downstream models, resulting in severe security risks. This threat is particularly concerning under clean-label attack scenarios, as it remains effective while having negligible impact on the utility of the synthetic data. Furthermore, we discover an Early-Stage Trigger Manifestation (ESTM) phenomenon: backdoor trigger patterns tend to surface more explicitly in the early, high-noise stages of the diffusion model's reverse generation process before being subtly integrated into the final samples. Overall, this work reveals a previously underexplored threat in generative data pipelines and provides initial insights toward mitigating backdoor risks in synthetic data generation. 

---
# PHANTOM: Progressive High-fidelity Adversarial Network for Threat Object Modeling 

**Authors**: Jamal Al-Karaki, Muhammad Al-Zafar Khan, Rand Derar Mohammad Al Athamneh  

**Link**: [PDF](https://arxiv.org/pdf/2512.15768)  

**Abstract**: The scarcity of cyberattack data hinders the development of robust intrusion detection systems. This paper introduces PHANTOM, a novel adversarial variational framework for generating high-fidelity synthetic attack data. Its innovations include progressive training, a dual-path VAE-GAN architecture, and domain-specific feature matching to preserve the semantics of attacks. Evaluated on 100,000 network traffic samples, models trained on PHANTOM data achieve 98% weighted accuracy on real attacks. Statistical analyses confirm that the synthetic data preserves authentic distributions and diversity. Limitations in generating rare attack types are noted, highlighting challenges with severe class imbalance. This work advances the generation of synthetic data for training robust, privacy-preserving detection systems. 

---
# Bridging Data and Physics: A Graph Neural Network-Based Hybrid Twin Framework 

**Authors**: M. Gorpinich, B. Moya, S. Rodriguez, F. Meraghni, Y. Jaafra, A. Briot, M. Henner, R. Leon, F. Chinesta  

**Link**: [PDF](https://arxiv.org/pdf/2512.15767)  

**Abstract**: Simulating complex unsteady physical phenomena relies on detailed mathematical models, simulated for instance by using the Finite Element Method (FEM). However, these models often exhibit discrepancies from the reality due to unmodeled effects or simplifying assumptions. We refer to this gap as the ignorance model. While purely data-driven approaches attempt to learn full system behavior, they require large amounts of high-quality data across the entire spatial and temporal domain. In real-world scenarios, such information is unavailable, making full data-driven modeling unreliable. To overcome this limitation, we model of the ignorance component using a hybrid twin approach, instead of simulating phenomena from scratch. Since physics-based models approximate the overall behavior of the phenomena, the remaining ignorance is typically lower in complexity than the full physical response, therefore, it can be learned with significantly fewer data. A key difficulty, however, is that spatial measurements are sparse, also obtaining data measuring the same phenomenon for different spatial configurations is challenging in practice. Our contribution is to overcome this limitation by using Graph Neural Networks (GNNs) to represent the ignorance model. GNNs learn the spatial pattern of the missing physics even when the number of measurement locations is limited. This allows us to enrich the physics-based model with data-driven corrections without requiring dense spatial, temporal and parametric data. To showcase the performance of the proposed method, we evaluate this GNN-based hybrid twin on nonlinear heat transfer problems across different meshes, geometries, and load positions. Results show that the GNN successfully captures the ignorance and generalizes corrections across spatial configurations, improving simulation accuracy and interpretability, while minimizing data requirements. 

---
# LOOPRAG: Enhancing Loop Transformation Optimization with Retrieval-Augmented Large Language Models 

**Authors**: Yijie Zhi, Yayu Cao, Jianhua Dai, Xiaoyang Han, Jingwen Pu, Qingran Wu, Sheng Cheng, Ming Cai  

**Link**: [PDF](https://arxiv.org/pdf/2512.15766)  

**Abstract**: Loop transformations are semantics-preserving optimization techniques, widely used to maximize objectives such as parallelism. Despite decades of research, applying the optimal composition of loop transformations remains challenging due to inherent complexities, including cost modeling for optimization objectives. Recent studies have explored the potential of Large Language Models (LLMs) for code optimization. However, our key observation is that LLMs often struggle with effective loop transformation optimization, frequently leading to errors or suboptimal optimization, thereby missing opportunities for performance improvements. To bridge this gap, we propose LOOPRAG, a novel retrieval-augmented generation framework designed to guide LLMs in performing effective loop optimization on Static Control Part. We introduce a parameter-driven method to harness loop properties, which trigger various loop transformations, and generate diverse yet legal example codes serving as a demonstration source. To effectively obtain the most informative demonstrations, we propose a loop-aware algorithm based on loop features, which balances similarity and diversity for code retrieval. To enhance correct and efficient code generation, we introduce a feedback-based iterative mechanism that incorporates compilation, testing and performance results as feedback to guide LLMs. Each optimized code undergoes mutation, coverage and differential testing for equivalence checking. We evaluate LOOPRAG on PolyBench, TSVC and LORE benchmark suites, and compare it against compilers (GCC-Graphite, Clang-Polly, Perspective and ICX) and representative LLMs (DeepSeek and GPT-4). The results demonstrate average speedups over base compilers of up to 11.20$\times$, 14.34$\times$, and 9.29$\times$ for PolyBench, TSVC, and LORE, respectively, and speedups over base LLMs of up to 11.97$\times$, 5.61$\times$, and 11.59$\times$. 

---
# AdaGradSelect: An adaptive gradient-guided layer selection method for efficient fine-tuning of SLMs 

**Authors**: Anshul Kumar, Gagan Raj Gupta, Manisha Chawla  

**Link**: [PDF](https://arxiv.org/pdf/2512.15764)  

**Abstract**: Large Language Models (LLMs) can perform many NLP tasks well, but fully fine-tuning them is expensive and requires a lot of memory. Parameter-Efficient Fine-Tuning (PEFT) methods such as LoRA reduce this cost by adding small low-rank updates to frozen model weights. However, these methods restrict the training to a limited subspace, which can sometimes reduce performance.
For Small Language Models (SLMs), where efficiency gains matter even more, we introduce AdaGradSelect, an adaptive method that selects which transformer blocks to update based on gradients.
Early observations showed that updating only the transformer blocks with the highest gradient norms can achieve performance close to full fine-tuning. Building on this insight, AdaGradSelect adaptively chooses which blocks to train. It uses a combination of Dirichlet-based sampling, which depends on how frequently blocks were updated in the past, and an epsilon-greedy exploration strategy. This lets the method explore different blocks in early training and gradually focus on the most important ones in later epochs.
Experiments show that AdaGradSelect trains about 12 percent faster and uses 35 percent less GPU memory while delivering performance very close to full fine-tuning. On the GSM8K dataset, it outperforms LoRA (rank 256) by about 3 percent on average across models such as Qwen2.5-0.5B, LLaMA3.2-1B, and Phi4-mini-3.8B. It also achieves similar accuracy on the MATH dataset. Overall, AdaGradSelect provides a more effective and resource-efficient alternative to traditional fine-tuning methods. 

---
# Cross-Sample Augmented Test-Time Adaptation for Personalized Intraoperative Hypotension Prediction 

**Authors**: Kanxue Li, Yibing Zhan, Hua Jin, Chongchong Qi, Xu Lin, Baosheng Yu  

**Link**: [PDF](https://arxiv.org/pdf/2512.15762)  

**Abstract**: Intraoperative hypotension (IOH) poses significant surgical risks, but accurate prediction remains challenging due to patient-specific variability. While test-time adaptation (TTA) offers a promising approach for personalized prediction, the rarity of IOH events often leads to unreliable test-time training. To address this, we propose CSA-TTA, a novel Cross-Sample Augmented Test-Time Adaptation framework that enhances training by incorporating hypotension events from other individuals. Specifically, we first construct a cross-sample bank by segmenting historical data into hypotensive and non-hypotensive samples. Then, we introduce a coarse-to-fine retrieval strategy for building test-time training data: we initially apply K-Shape clustering to identify representative cluster centers and subsequently retrieve the top-K semantically similar samples based on the current patient signal. Additionally, we integrate both self-supervised masked reconstruction and retrospective sequence forecasting signals during training to enhance model adaptability to rapid and subtle intraoperative dynamics. We evaluate the proposed CSA-TTA on both the VitalDB dataset and a real-world in-hospital dataset by integrating it with state-of-the-art time series forecasting models, including TimesFM and UniTS. CSA-TTA consistently enhances performance across settings-for instance, on VitalDB, it improves Recall and F1 scores by +1.33% and +1.13%, respectively, under fine-tuning, and by +7.46% and +5.07% in zero-shot scenarios-demonstrating strong robustness and generalization. 

---
# ReactorFold: Generative discovery of nuclear reactor cores via emergent physical reasoning 

**Authors**: Yoonpyo Lee  

**Link**: [PDF](https://arxiv.org/pdf/2512.15756)  

**Abstract**: Designing nuclear reactor cores requires navigating large discrete design spaces governed by complex neutronic interactions. Traditional deterministic, metaheuristic, and machine-learning-assisted methods search within fixed, human-defined configuration spaces, limiting their ability to discover fundamentally new design topologies. Here we introduce ReactorFold, a generative framework that reformulates fuel-assembly design as a sequence modeling problem for language models. Using Monte Carlo data, parameter-efficient fine-tuning, and Direct Preference Optimization (DPO), the model learns the latent structure of a pressurized-water-reactor assembly and generates candidate layouts in a single forward pass. Notably, the DPO-aligned model exhibits emergent design-space expansion: despite being trained exclusively on configurations with a fixed number of gadolinium burnable absorber (Gd) rods, it autonomously adjusts Gd inventory to satisfy strict power-peaking constraints. The model also discovers high-performing asymmetric configurations that challenge conventional symmetric loading heuristics, accessing design regimes inaccessible to conventional search methods and demonstrating that language models can internalize causal physical relationships and transcend human-imposed design constraints. 

---
# TAO-Net: Two-stage Adaptive OOD Classification Network for Fine-grained Encrypted Traffic Classification 

**Authors**: Zihao Wang, Wei Peng, Junming Zhang, Jian Li, Wenxin Fang  

**Link**: [PDF](https://arxiv.org/pdf/2512.15753)  

**Abstract**: Encrypted traffic classification aims to identify applications or services by analyzing network traffic data. One of the critical challenges is the continuous emergence of new applications, which generates Out-of-Distribution (OOD) traffic patterns that deviate from known categories and are not well represented by predefined models. Current approaches rely on predefined categories, which limits their effectiveness in handling unknown traffic types. Although some methods mitigate this limitation by simply classifying unknown traffic into a single "Other" category, they fail to make a fine-grained classification. In this paper, we propose a Two-stage Adaptive OOD classification Network (TAO-Net) that achieves accurate classification for both In-Distribution (ID) and OOD encrypted traffic. The method incorporates an innovative two-stage design: the first stage employs a hybrid OOD detection mechanism that integrates transformer-based inter-layer transformation smoothness and feature analysis to effectively distinguish between ID and OOD traffic, while the second stage leverages large language models with a novel semantic-enhanced prompt strategy to transform OOD traffic classification into a generation task, enabling flexible fine-grained classification without relying on predefined labels. Experiments on three datasets demonstrate that TAO-Net achieves 96.81-97.70% macro-precision and 96.77-97.68% macro-F1, outperforming previous methods that only reach 44.73-86.30% macro-precision, particularly in identifying emerging network applications. 

---
# GLOW: Graph-Language Co-Reasoning for Agentic Workflow Performance Prediction 

**Authors**: Wei Guan, Jian Cao, Jinyu Cai, Qiqi Cai, Jianqi Gao, See-Kiong Ng  

**Link**: [PDF](https://arxiv.org/pdf/2512.15751)  

**Abstract**: Agentic Workflows (AWs) have emerged as a promising paradigm for solving complex tasks. However, the scalability of automating their generation is severely constrained by the high cost and latency of execution-based evaluation. Existing AW performance prediction methods act as surrogates but fail to simultaneously capture the intricate topological dependencies and the deep semantic logic embedded in AWs. To address this limitation, we propose GLOW, a unified framework for AW performance prediction that combines the graph-structure modeling capabilities of GNNs with the reasoning power of LLMs. Specifically, we introduce a graph-oriented LLM, instruction-tuned on graph tasks, to extract topologically aware semantic features, which are fused with GNN-encoded structural representations. A contrastive alignment strategy further refines the latent space to distinguish high-quality AWs. Extensive experiments on FLORA-Bench show that GLOW outperforms state-of-the-art baselines in prediction accuracy and ranking utility. 

---
# LLaDA2.0: Scaling Up Diffusion Language Models to 100B 

**Authors**: Tiwei Bie, Maosong Cao, Kun Chen, Lun Du, Mingliang Gong, Zhuochen Gong, Yanmei Gu, Jiaqi Hu, Zenan Huang, Zhenzhong Lan, Chengxi Li, Chongxuan Li, Jianguo Li, Zehuan Li, Huabin Liu, Ling Liu, Guoshan Lu, Xiaocheng Lu, Yuxin Ma, Jianfeng Tan, Lanning Wei, Ji-Rong Wen, Yipeng Xing, Xiaolu Zhang, Junbo Zhao, Da Zheng, Jun Zhou, Junlin Zhou, Zhanchao Zhou, Liwang Zhu, Yihong Zhuang  

**Link**: [PDF](https://arxiv.org/pdf/2512.15745)  

**Abstract**: This paper presents LLaDA2.0 -- a tuple of discrete diffusion large language models (dLLM) scaling up to 100B total parameters through systematic conversion from auto-regressive (AR) models -- establishing a new paradigm for frontier-scale deployment. Instead of costly training from scratch, LLaDA2.0 upholds knowledge inheritance, progressive adaption and efficiency-aware design principle, and seamless converts a pre-trained AR model into dLLM with a novel 3-phase block-level WSD based training scheme: progressive increasing block-size in block diffusion (warm-up), large-scale full-sequence diffusion (stable) and reverting back to compact-size block diffusion (decay). Along with post-training alignment with SFT and DPO, we obtain LLaDA2.0-mini (16B) and LLaDA2.0-flash (100B), two instruction-tuned Mixture-of-Experts (MoE) variants optimized for practical deployment. By preserving the advantages of parallel decoding, these models deliver superior performance and efficiency at the frontier scale. Both models were open-sourced. 

---
# Bayesian Modeling for Uncertainty Management in Financial Risk Forecasting and Compliance 

**Authors**: Sharif Al Mamun, Rakib Hossain, Md. Jobayer Rahman, Malay Kumar Devnath, Farhana Afroz, Lisan Al Amin  

**Link**: [PDF](https://arxiv.org/pdf/2512.15739)  

**Abstract**: A Bayesian analytics framework that precisely quantifies uncertainty offers a significant advance for financial risk management. We develop an integrated approach that consistently enhances the handling of risk in market volatility forecasting, fraud detection, and compliance monitoring. Our probabilistic, interpretable models deliver reliable results: We evaluate the performance of one-day-ahead 95% Value-at-Risk (VaR) forecasts on daily S&P 500 returns, with a training period from 2000 to 2019 and an out-of-sample test period spanning 2020 to 2024. Formal tests of unconditional (Kupiec) and conditional (Christoffersen) coverage reveal that an LSTM baseline achieves near-nominal calibration. In contrast, a GARCH(1,1) model with Student-t innovations underestimates tail risk. Our proposed discount-factor DLM model produces a slightly liberal VaR estimate, with evidence of clustered violations. Bayesian logistic regression improves recall and AUC-ROC for fraud detection, and a hierarchical Beta state-space model provides transparent and adaptive compliance risk assessment. The pipeline is distinguished by precise uncertainty quantification, interpretability, and GPU-accelerated analysis, delivering up to 50x speedup. Remaining challenges include sparse fraud data and proxy compliance labels, but the framework enables actionable risk insights. Future expansion will extend feature sets, explore regime-switching priors, and enhance scalable inference. 

---
# Hybrid Quantum-Classical Ensemble Learning for S\&P 500 Directional Prediction 

**Authors**: Abraham Itzhak Weinberg  

**Link**: [PDF](https://arxiv.org/pdf/2512.15738)  

**Abstract**: Financial market prediction is a challenging application of machine learning, where even small improvements in directional accuracy can yield substantial value. Most models struggle to exceed 55--57\% accuracy due to high noise, non-stationarity, and market efficiency. We introduce a hybrid ensemble framework combining quantum sentiment analysis, Decision Transformer architecture, and strategic model selection, achieving 60.14\% directional accuracy on S\&P 500 prediction, a 3.10\% improvement over individual models.
Our framework addresses three limitations of prior approaches. First, architecture diversity dominates dataset diversity: combining different learning algorithms (LSTM, Decision Transformer, XGBoost, Random Forest, Logistic Regression) on the same data outperforms training identical architectures on multiple datasets (60.14\% vs.\ 52.80\%), confirmed by correlation analysis ($r>0.6$ among same-architecture models). Second, a 4-qubit variational quantum circuit enhances sentiment analysis, providing +0.8\% to +1.5\% gains per model. Third, smart filtering excludes weak predictors (accuracy $<52\%$), improving ensemble performance (Top-7 models: 60.14\% vs.\ all 35 models: 51.2\%).
We evaluate on 2020--2023 market data across seven instruments, covering diverse regimes including the COVID-19 crash and inflation-driven correction. McNemar's test confirms statistical significance ($p<0.05$). Preliminary backtesting with confidence-based filtering (6+ model consensus) yields a Sharpe ratio of 1.2 versus buy-and-hold's 0.8, demonstrating practical trading potential. 

---
# Deep Reinforcement Learning Optimization for Uncertain Nonlinear Systems via Event-Triggered Robust Adaptive Dynamic Programming 

**Authors**: Ningwei Bai, Chi Pui Chan, Qichen Yin, Tengyang Gong, Yunda Yan, Zezhi Tang  

**Link**: [PDF](https://arxiv.org/pdf/2512.15735)  

**Abstract**: This work proposes a unified control architecture that couples a Reinforcement Learning (RL)-driven controller with a disturbance-rejection Extended State Observer (ESO), complemented by an Event-Triggered Mechanism (ETM) to limit unnecessary computations. The ESO is utilized to estimate the system states and the lumped disturbance in real time, forming the foundation for effective disturbance compensation. To obtain near-optimal behavior without an accurate system description, a value-iteration-based Adaptive Dynamic Programming (ADP) method is adopted for policy approximation. The inclusion of the ETM ensures that parameter updates of the learning module are executed only when the state deviation surpasses a predefined bound, thereby preventing excessive learning activity and substantially reducing computational load. A Lyapunov-oriented analysis is used to characterize the stability properties of the resulting closed-loop system. Numerical experiments further confirm that the developed approach maintains strong control performance and disturbance tolerance, while achieving a significant reduction in sampling and processing effort compared with standard time-triggered ADP schemes. 

---
# A Context-Free Smart Grid Model Using Complex System Approach 

**Authors**: Soufian Ben Amor, Alain Bui, Guillaume Guerard  

**Link**: [PDF](https://arxiv.org/pdf/2512.15733)  

**Abstract**: Energy and pollution are urging problems of the 21th century. By gradually changing the actual power grid system, smart grid may evolve into different systems by means of size, elements and strategies, but its fundamental requirements and objectives will not change such as optimizing production, transmission, and consumption. Studying the smart grid through modeling and simulation provides us with valuable results which cannot be obtained in real world due to time and cost related constraints. Moreover, due to the complexity of the smart grid, achieving global optimization is not an easy task. In this paper, we propose a complex system based approach to the smart grid modeling, accentuating on the optimization by combining game theoretical and classical methods in different levels. Thanks to this combination, the optimization can be achieved with flexibility and scalability, while keeping its generality. 

---
# TinyMyo: a Tiny Foundation Model for Flexible EMG Signal Processing at the Edge 

**Authors**: Matteo Fasulo, Giusy Spacone, Thorir Mar Ingolfsson, Yawei Li, Luca Benini, Andrea Cossettini  

**Link**: [PDF](https://arxiv.org/pdf/2512.15729)  

**Abstract**: Surface electromyography (EMG) is a non-invasive sensing modality used in several domains, including biomechanics, rehabilitation, prosthetic control, and emerging human-machine interaction paradigms. Despite decades of use, significant challenges remain in achieving robust generalization across subjects, recording systems, and acquisition protocols. To tackle these challenges, foundation models (FMs) are gaining traction when targeting end-to-end applications based on EMG signals. Yet, existing EMG FMs remain limited to single downstream tasks and lack deployability on embedded platforms. In this work, we present TinyMyo, a lightweight FM based on a Transformer encoder architecture. The model is pre-trained in a self-supervised manner on publicly available datasets and achieves high reconstruction fidelity with only 3.6M parameters. With minimal task-specific head adaptations, the same backbone is used to tackle multiple downstream tasks, leveraging datasets acquired from diverse sensing locations and hardware platforms. We demonstrate generalization across hand gesture classification, hand kinematic regression, speech production and recognition, with performance comparable to or surpassing the state of the art (SoA), and model size below 5M parameters. We achieve SoA results compared to previous FM-based works on the NinaPro DB5 ($89.4\pm0.16\%$), UCI-EMG ($97.56\pm0.32\%$), and EPN-612 ($96.74\pm0.09\%$) datasets. We report, to the best of our knowledge, the first deployment of an EMG FM on an ultra-low-power microcontroller (GAP9), achieving an average power envelope of 36.45mW. By open-sourcing the pre-trained and the downstream task architectures (this https URL), we aim to provide a flexible resource that can accelerate future research and serve as a common foundation for the EMG community. 

---
# FedSight AI: Multi-Agent System Architecture for Federal Funds Target Rate Prediction 

**Authors**: Yuhan Hou, Tianji Rao, Jeremy Tan, Adler Viton, Xiyue Zhang, David Ye, Abhishek Kodi, Sanjana Dulam, Aditya Paul, Yikai Feng  

**Link**: [PDF](https://arxiv.org/pdf/2512.15728)  

**Abstract**: The Federal Open Market Committee (FOMC) sets the federal funds rate, shaping monetary policy and the broader economy. We introduce \emph{FedSight AI}, a multi-agent framework that uses large language models (LLMs) to simulate FOMC deliberations and predict policy outcomes. Member agents analyze structured indicators and unstructured inputs such as the Beige Book, debate options, and vote, replicating committee reasoning. A Chain-of-Draft (CoD) extension further improves efficiency and accuracy by enforcing concise multistage reasoning. Evaluated at 2023-2024 meetings, FedSight CoD achieved accuracy of 93.75\% and stability of 93.33\%, outperforming baselines including MiniFed and Ordinal Random Forest (RF), while offering transparent reasoning aligned with real FOMC communications. 

---
# Value Lens: Using Large Language Models to Understand Human Values 

**Authors**: Eduardo de la Cruz Fernández, Marcelo Karanik, Sascha Ossowski  

**Link**: [PDF](https://arxiv.org/pdf/2512.15722)  

**Abstract**: The autonomous decision-making process, which is increasingly applied to computer systems, requires that the choices made by these systems align with human values. In this context, systems must assess how well their decisions reflect human values. To achieve this, it is essential to identify whether each available action promotes or undermines these values. This article presents Value Lens, a text-based model designed to detect human values using generative artificial intelligence, specifically Large Language Models (LLMs). The proposed model operates in two stages: the first aims to formulate a formal theory of values, while the second focuses on identifying these values within a given text. In the first stage, an LLM generates a description based on the established theory of values, which experts then verify. In the second stage, a pair of LLMs is employed: one LLM detects the presence of values, and the second acts as a critic and reviewer of the detection process. The results indicate that Value Lens performs comparably to, and even exceeds, the effectiveness of other models that apply different methods for similar tasks. 

---
# DiscoverDCP: A Data-Driven Approach for Construction of Disciplined Convex Programs via Symbolic Regression 

**Authors**: Sveinung Myhre  

**Link**: [PDF](https://arxiv.org/pdf/2512.15721)  

**Abstract**: We propose DiscoverDCP, a data-driven framework that integrates symbolic regression with the rule sets of Disciplined Convex Programming (DCP) to perform system identification. By enforcing that all discovered candidate model expressions adhere to DCP composition rules, we ensure that the output expressions are globally convex by construction, circumventing the computationally intractable process of post-hoc convexity verification. This approach allows for the discovery of convex surrogates that exhibit more relaxed and accurate functional forms than traditional fixed-parameter convex expressions (e.g., quadratic functions). The proposed method produces interpretable, verifiable, and flexible convex models suitable for safety-critical control and optimization tasks. 

---
# Large Model Enabled Embodied Intelligence for 6G Integrated Perception, Communication, and Computation Network 

**Authors**: Zhuoran Li, Zhen Gao, Xinhua Liu, Zheng Wang, Xiaotian Zhou, Lei Liu, Yongpeng Wu, Wei Feng, Yongming Huang  

**Link**: [PDF](https://arxiv.org/pdf/2512.15109)  

**Abstract**: The advent of sixth-generation (6G) places intelligence at the core of wireless architecture, fusing perception, communication, and computation into a single closed-loop. This paper argues that large artificial intelligence models (LAMs) can endow base stations with perception, reasoning, and acting capabilities, thus transforming them into intelligent base station agents (IBSAs). We first review the historical evolution of BSs from single-functional analog infrastructure to distributed, software-defined, and finally LAM-empowered IBSA, highlighting the accompanying changes in architecture, hardware platforms, and deployment. We then present an IBSA architecture that couples a perception-cognition-execution pipeline with cloud-edge-end collaboration and parameter-efficient adaptation. Subsequently,we study two representative scenarios: (i) cooperative vehicle-road perception for autonomous driving, and (ii) ubiquitous base station support for low-altitude uncrewed aerial vehicle safety monitoring and response against unauthorized drones. On this basis, we analyze key enabling technologies spanning LAM design and training, efficient edge-cloud inference, multi-modal perception and actuation, as well as trustworthy security and governance. We further propose a holistic evaluation framework and benchmark considerations that jointly cover communication performance, perception accuracy, decision-making reliability, safety, and energy efficiency. Finally, we distill open challenges on benchmarks, continual adaptation, trustworthy decision-making, and standardization. Together, this work positions LAM-enabled IBSAs as a practical path toward integrated perception, communication, and computation native, safety-critical 6G systems. 

---
# Weakly Supervised Pneumonia Localization from Chest X-Rays Using Deep Neural Network and Grad-CAM Explanations 

**Authors**: Kiran Shahi, Anup Bagale  

**Link**: [PDF](https://arxiv.org/pdf/2511.00456)  

**Abstract**: Chest X-ray imaging is commonly used to diagnose pneumonia, but accurately localizing the pneumonia-affected regions typically requires detailed pixel-level annotations, which are costly and time consuming to obtain. To address this limitation, this study proposes a weakly supervised deep learning framework for pneumonia classification and localization using Gradient-weighted Class Activation Mapping (Grad-CAM). Instead of relying on costly pixel-level annotations, the proposed method utilizes image-level labels to generate clinically meaningful heatmaps that highlight pneumonia-affected regions. Furthermore, we evaluate seven pre-trained deep learning models, including a Vision Transformer, under identical training conditions, using focal loss and patient-wise splits to prevent data leakage. Experimental results suggest that all models achieved high classification accuracy (96--98\%), with ResNet-18 and EfficientNet-B0 showing the best overall performance and MobileNet-V3 providing an efficient lightweight alternative. Grad-CAM heatmap visualizations confirm that the proposed methods focus on clinically relevant lung regions, supporting the use of explainable AI for radiological diagnostics. Overall, this work highlights the potential of weakly supervised, explainable models that enhance transparency and clinical trust in AI-assisted pneumonia screening. 

---
