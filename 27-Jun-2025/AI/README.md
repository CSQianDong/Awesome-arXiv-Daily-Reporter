# PsyLite Technical Report 

**Authors**: Fangjun Ding, Renyu Zhang, Xinyu Feng, Chengye Xie, Zheng Zhang, Yanting Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2506.21536)  

**Abstract**: With the rapid development of digital technology, AI-driven psychological counseling has gradually become an important research direction in the field of mental health. However, existing models still have deficiencies in dialogue safety, detailed scenario handling, and lightweight deployment. To address these issues, this study proposes PsyLite, a lightweight psychological counseling large language model agent developed based on the base model InternLM2.5-7B-chat. Through a two-stage training strategy (hybrid distillation data fine-tuning and ORPO preference optimization), PsyLite enhances the model's deep-reasoning ability, psychological counseling ability, and safe dialogue ability. After deployment using Ollama and Open WebUI, a custom workflow is created with Pipelines. An innovative conditional RAG is designed to introduce crosstalk humor elements at appropriate times during psychological counseling to enhance user experience and decline dangerous requests to strengthen dialogue safety. Evaluations show that PsyLite outperforms the baseline models in the Chinese general evaluation (CEval), psychological counseling professional evaluation (CPsyCounE), and dialogue safety evaluation (SafeDialBench), particularly in psychological counseling professionalism (CPsyCounE score improvement of 47.6\%) and dialogue safety (\safe{} score improvement of 2.4\%). Additionally, the model uses quantization technology (GGUF q4\_k\_m) to achieve low hardware deployment (5GB memory is sufficient for operation), providing a feasible solution for psychological counseling applications in resource-constrained environments. 

---
# Mind2Web 2: Evaluating Agentic Search with Agent-as-a-Judge 

**Authors**: Boyu Gou, Zanming Huang, Yuting Ning, Yu Gu, Michael Lin, Weijian Qi, Andrei Kopanev, Botao Yu, Bernal Jiménez Gutiérrez, Yiheng Shu, Chan Hee Song, Jiaman Wu, Shijie Chen, Hanane Nour Moussa, Tianshu Zhang, Jian Xie, Yifei Li, Tianci Xue, Zeyi Liao, Kai Zhang, Boyuan Zheng, Zhaowei Cai, Viktor Rozgic, Morteza Ziyadi, Huan Sun, Yu Su  

**Link**: [PDF](https://arxiv.org/pdf/2506.21506)  

**Abstract**: Agentic search such as Deep Research systems, where large language models autonomously browse the web, synthesize information, and return comprehensive citation-backed answers, represents a major shift in how users interact with web-scale information. While promising greater efficiency and cognitive offloading, the growing complexity and open-endedness of agentic search have outpaced existing evaluation benchmarks and methodologies, which largely assume short search horizons and static answers. In this paper, we introduce Mind2Web 2, a benchmark of 130 realistic, high-quality, and long-horizon tasks that require real-time web browsing and extensive information synthesis, constructed with over 1,000 hours of human labor. To address the challenge of evaluating time-varying and complex answers, we propose a novel Agent-as-a-Judge framework. Our method constructs task-specific judge agents based on a tree-structured rubric design to automatically assess both answer correctness and source attribution. We conduct a comprehensive evaluation of nine frontier agentic search systems and human performance, along with a detailed error analysis to draw insights for future development. The best-performing system, OpenAI Deep Research, can already achieve 50-70% of human performance while spending half the time, showing a great potential. Altogether, Mind2Web 2 provides a rigorous foundation for developing and benchmarking the next generation of agentic search systems. 

---
# Ad-Hoc Human-AI Coordination Challenge 

**Authors**: Tin Dizdarević, Ravi Hammond, Tobias Gessler, Anisoara Calinescu, Jonathan Cook, Matteo Gallici, Andrei Lupu, Jakob Nicolaus Foerster  

**Link**: [PDF](https://arxiv.org/pdf/2506.21490)  

**Abstract**: Achieving seamless coordination between AI agents and humans is crucial for real-world applications, yet it remains a significant open challenge. Hanabi is a cooperative card game featuring imperfect information, constrained communication, theory of mind requirements, and coordinated action -- making it an ideal testbed for human-AI coordination. However, its use for human-AI interaction has been limited by the challenges of human evaluation. In this work, we introduce the Ad-Hoc Human-AI Coordination Challenge (AH2AC2) to overcome the constraints of costly and difficult-to-reproduce human evaluations. We develop \textit{human proxy agents} on a large-scale human dataset that serve as robust, cheap, and reproducible human-like evaluation partners in AH2AC2. To encourage the development of data-efficient methods, we open-source a dataset of 3,079 games, deliberately limiting the amount of available human gameplay data. We present baseline results for both two- and three- player Hanabi scenarios. To ensure fair evaluation, we host the proxy agents through a controlled evaluation system rather than releasing them publicly. The code is available at \href{this https URL}{this https URL}. 

---
# Spatial Mental Modeling from Limited Views 

**Authors**: Baiqiao Yin, Qineng Wang, Pingyue Zhang, Jianshu Zhang, Kangrui Wang, Zihan Wang, Jieyu Zhang, Keshigeyan Chandrasegaran, Han Liu, Ranjay Krishna, Saining Xie, Manling Li, Jiajun Wu, Li Fei-Fei  

**Link**: [PDF](https://arxiv.org/pdf/2506.21458)  

**Abstract**: Can Vision Language Models (VLMs) imagine the full scene from just a few views, like humans do? Humans form spatial mental models, internal representations of unseen space, to reason about layout, perspective, and motion. Our new MindCube benchmark with 21,154 questions across 3,268 images exposes this critical gap, where existing VLMs exhibit near-random performance. Using MindCube, we systematically evaluate how well VLMs build robust spatial mental models through representing positions (cognitive mapping), orientations (perspective-taking), and dynamics (mental simulation for "what-if" movements). We then explore three approaches to help VLMs approximate spatial mental models, including unseen intermediate views, natural language reasoning chains, and cognitive maps. The significant improvement comes from a synergistic approach, "map-then-reason", that jointly trains the model to first generate a cognitive map and then reason upon it. By training models to reason over these internal maps, we boosted accuracy from 37.8% to 60.8% (+23.0%). Adding reinforcement learning pushed performance even further to 70.7% (+32.9%). Our key insight is that such scaffolding of spatial mental models, actively constructing and utilizing internal structured spatial representations with flexible reasoning processes, significantly improves understanding of unobservable space. 

---
# TableMoE: Neuro-Symbolic Routing for Structured Expert Reasoning in Multimodal Table Understanding 

**Authors**: Junwen Zhang, Pu Chen, Yin Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2506.21393)  

**Abstract**: Multimodal understanding of tables in real-world contexts is challenging due to the complexity of structure, symbolic density, and visual degradation (blur, skew, watermarking, incomplete structures or fonts, multi-span or hierarchically nested layouts). Existing multimodal large language models (MLLMs) struggle with such WildStruct conditions, resulting in limited performance and poor generalization. To address these challenges, we propose TableMoE, a neuro-symbolic Mixture-of-Connector-Experts (MoCE) architecture specifically designed for robust, structured reasoning over multimodal table data. TableMoE features an innovative Neuro-Symbolic Routing mechanism, which predicts latent semantic token roles (e.g., header, data cell, axis, formula) and dynamically routes table elements to specialized experts (Table-to-HTML, Table-to-JSON, Table-to-Code) using a confidence-aware gating strategy informed by symbolic reasoning graphs. To facilitate effective alignment-driven pretraining, we introduce the large-scale TableMoE-Align dataset, consisting of 1.2M table-HTML-JSON-code quadruples across finance, science, biomedicine and industry, utilized exclusively for model pretraining. For evaluation, we curate and release four challenging WildStruct benchmarks: WMMFinQA, WMMTatQA, WMMTabDialog, and WMMFinanceMath, designed specifically to stress-test models under real-world multimodal degradation and structural complexity. Experimental results demonstrate that TableMoE significantly surpasses existing state-of-the-art models. Extensive ablation studies validate each core component, emphasizing the critical role of Neuro-Symbolic Routing and structured expert alignment. Through qualitative analyses, we further showcase TableMoE's interpretability and enhanced robustness, underscoring the effectiveness of integrating neuro-symbolic reasoning for multimodal table understanding. 

---
# Active Inference AI Systems for Scientific Discovery 

**Authors**: Karthik Duraisamy  

**Link**: [PDF](https://arxiv.org/pdf/2506.21329)  

**Abstract**: The rapid evolution of artificial intelligence has led to expectations of transformative scientific discovery, yet current systems remain fundamentally limited by their operational architectures, brittle reasoning mechanisms, and their separation from experimental reality. Building on earlier work, we contend that progress in AI-driven science now depends on closing three fundamental gaps -- the abstraction gap, the reasoning gap, and the reality gap -- rather than on model size/data/test time compute. Scientific reasoning demands internal representations that support simulation of actions and response, causal structures that distinguish correlation from mechanism, and continuous calibration. We define active inference AI systems for scientific discovery as those that (i) maintain long-lived research memories grounded in causal self-supervised foundation models, (ii) symbolic or neuro-symbolic planners equipped with Bayesian guardrails, (iii) grow persistent knowledge graphs where thinking generates novel conceptual nodes, reasoning establishes causal edges, and real-world interaction prunes false connections while strengthening verified pathways, and (iv) refine their internal representations through closed-loop interaction with both high-fidelity simulators and automated laboratories - an operational loop where mental simulation guides action and empirical surprise reshapes understanding. In essence, we outline an architecture where discovery arises from the interplay between internal models that enable counterfactual reasoning and external validation that grounds hypotheses in reality. It is also argued that the inherent ambiguity in feedback from simulations and experiments, and underlying uncertainties makes human judgment indispensable, not as a temporary scaffold but as a permanent architectural component. 

---
# IXAII: An Interactive Explainable Artificial Intelligence Interface for Decision Support Systems 

**Authors**: Pauline Speckmann, Mario Nadj, Christian Janiesch  

**Link**: [PDF](https://arxiv.org/pdf/2506.21310)  

**Abstract**: Although several post-hoc methods for explainable AI have been developed, most are static and neglect the user perspective, limiting their effectiveness for the target audience. In response, we developed the interactive explainable intelligent system called IXAII that offers explanations from four explainable AI methods: LIME, SHAP, Anchors, and DiCE. Our prototype provides tailored views for five user groups and gives users agency over the explanations' content and their format. We evaluated IXAII through interviews with experts and lay users. Our results indicate that IXAII, which provides different explanations with multiple visualization options, is perceived as helpful to increase transparency. By bridging the gaps between explainable AI methods, interactivity, and practical implementation, we provide a novel perspective on AI explanation practices and human-AI interaction. 

---
# World-aware Planning Narratives Enhance Large Vision-Language Model Planner 

**Authors**: Junhao Shi, Zhaoye Fei, Siyin Wang, Qipeng Guo, Jingjing Gong, Xipeng QIu  

**Link**: [PDF](https://arxiv.org/pdf/2506.21230)  

**Abstract**: Large Vision-Language Models (LVLMs) show promise for embodied planning tasks but struggle with complex scenarios involving unfamiliar environments and multi-step goals. Current approaches rely on environment-agnostic imitation learning that disconnects instructions from environmental contexts, causing models to struggle with context-sensitive instructions and rely on supplementary cues rather than visual reasoning during long-horizon interactions. In this work, we propose World-Aware Planning Narrative Enhancement (WAP), a framework that infuses LVLMs with comprehensive environmental understanding through four cognitive capabilities (visual appearance modeling, spatial reasoning, functional abstraction, and syntactic grounding) while developing and evaluating models using only raw visual observations through curriculum learning. Evaluations on the EB-ALFRED benchmark demonstrate substantial improvements, with Qwen2.5-VL achieving a 60.7 absolute improvement in task success rates, particularly in commonsense reasoning (+60.0) and long-horizon planning (+70.0). Notably, our enhanced open-source models outperform proprietary systems like GPT-4o and Claude-3.5-Sonnet by a large margin. 

---
# Unveiling Causal Reasoning in Large Language Models: Reality or Mirage? 

**Authors**: Haoang Chi, He Li, Wenjing Yang, Feng Liu, Long Lan, Xiaoguang Ren, Tongliang Liu, Bo Han  

**Link**: [PDF](https://arxiv.org/pdf/2506.21215)  

**Abstract**: Causal reasoning capability is critical in advancing large language models (LLMs) toward strong artificial intelligence. While versatile LLMs appear to have demonstrated capabilities in understanding contextual causality and providing responses that obey the laws of causality, it remains unclear whether they perform genuine causal reasoning akin to humans. However, current evidence indicates the contrary. Specifically, LLMs are only capable of performing shallow (level-1) causal reasoning, primarily attributed to the causal knowledge embedded in their parameters, but they lack the capacity for genuine human-like (level-2) causal reasoning. To support this hypothesis, methodologically, we delve into the autoregression mechanism of transformer-based LLMs, revealing that it is not inherently causal. Empirically, we introduce a new causal Q&A benchmark called CausalProbe-2024, whose corpora are fresh and nearly unseen for the studied LLMs. The LLMs exhibit a significant performance drop on CausalProbe-2024 compared to earlier benchmarks, indicating the fact that they primarily engage in level-1 causal reasoning. To bridge the gap towards level-2 causal reasoning, we draw inspiration from the fact that human reasoning is usually facilitated by general knowledge and intended goals. We propose G^2-Reasoner, a method that incorporates general knowledge and goal-oriented prompts into LLMs' causal reasoning processes. Experiments demonstrate that G^2-Reasoner significantly enhances LLMs' causal reasoning capability, particularly in fresh and counterfactual contexts. This work sheds light on a new path for LLMs to advance towards genuine causal reasoning, going beyond level-1 and making strides towards level-2. 

---
# Beyond Reactive Safety: Risk-Aware LLM Alignment via Long-Horizon Simulation 

**Authors**: Chenkai Sun, Denghui Zhang, ChengXiang Zhai, Heng Ji  

**Link**: [PDF](https://arxiv.org/pdf/2506.20949)  

**Abstract**: Given the growing influence of language model-based agents on high-stakes societal decisions, from public policy to healthcare, ensuring their beneficial impact requires understanding the far-reaching implications of their suggestions. We propose a proof-of-concept framework that projects how model-generated advice could propagate through societal systems on a macroscopic scale over time, enabling more robust alignment. To assess the long-term safety awareness of language models, we also introduce a dataset of 100 indirect harm scenarios, testing models' ability to foresee adverse, non-obvious outcomes from seemingly harmless user prompts. Our approach achieves not only over 20% improvement on the new dataset but also an average win rate exceeding 70% against strong baselines on existing safety benchmarks (AdvBench, SafeRLHF, WildGuardMix), suggesting a promising direction for safer agents. 

---
# Dynamic Context-Aware Prompt Recommendation for Domain-Specific AI Applications 

**Authors**: Xinye Tang, Haijun Zhai, Chaitanya Belwal, Vineeth Thayanithi, Philip Baumann, Yogesh K Roy  

**Link**: [PDF](https://arxiv.org/pdf/2506.20815)  

**Abstract**: LLM-powered applications are highly susceptible to the quality of user prompts, and crafting high-quality prompts can often be challenging especially for domain-specific applications. This paper presents a novel dynamic context-aware prompt recommendation system for domain-specific AI applications. Our solution combines contextual query analysis, retrieval-augmented knowledge grounding, hierarchical skill organization, and adaptive skill ranking to generate relevant and actionable prompt suggestions.
The system leverages behavioral telemetry and a two-stage hierarchical reasoning process to dynamically select and rank relevant skills, and synthesizes prompts using both predefined and adaptive templates enhanced with few-shot learning. Experiments on real-world datasets demonstrate that our approach achieves high usefulness and relevance, as validated by both automated and expert evaluations. 

---
# MAGPIE: A dataset for Multi-AGent contextual PrIvacy Evaluation 

**Authors**: Gurusha Juneja, Alon Albalak, Wenyue Hua, William Yang Wang  

**Link**: [PDF](https://arxiv.org/pdf/2506.20737)  

**Abstract**: The proliferation of LLM-based agents has led to increasing deployment of inter-agent collaboration for tasks like scheduling, negotiation, resource allocation etc. In such systems, privacy is critical, as agents often access proprietary tools and domain-specific databases requiring strict confidentiality. This paper examines whether LLM-based agents demonstrate an understanding of contextual privacy. And, if instructed, do these systems preserve inference time user privacy in non-adversarial multi-turn conversation. Existing benchmarks to evaluate contextual privacy in LLM-agents primarily assess single-turn, low-complexity tasks where private information can be easily excluded. We first present a benchmark - MAGPIE comprising 158 real-life high-stakes scenarios across 15 domains. These scenarios are designed such that complete exclusion of private data impedes task completion yet unrestricted information sharing could lead to substantial losses. We then evaluate the current state-of-the-art LLMs on (a) their understanding of contextually private data and (b) their ability to collaborate without violating user privacy. Empirical experiments demonstrate that current models, including GPT-4o and Claude-2.7-Sonnet, lack robust understanding of contextual privacy, misclassifying private data as shareable 25.2\% and 43.6\% of the time. In multi-turn conversations, these models disclose private information in 59.9\% and 50.5\% of cases even under explicit privacy instructions. Furthermore, multi-agent systems fail to complete tasks in 71\% of scenarios. These results underscore that current models are not aligned towards both contextual privacy preservation and collaborative task-solving. 

---
# The Singapore Consensus on Global AI Safety Research Priorities 

**Authors**: Yoshua Bengio, Tegan Maharaj, Luke Ong, Stuart Russell, Dawn Song, Max Tegmark, Lan Xue, Ya-Qin Zhang, Stephen Casper, Wan Sie Lee, Sören Mindermann, Vanessa Wilfred, Vidhisha Balachandran, Fazl Barez, Michael Belinsky, Imane Bello, Malo Bourgon, Mark Brakel, Siméon Campos, Duncan Cass-Beggs, Jiahao Chen, Rumman Chowdhury, Kuan Chua Seah, Jeff Clune, Juntao Dai, Agnes Delaborde, Nouha Dziri, Francisco Eiras, Joshua Engels, Jinyu Fan, Adam Gleave, Noah Goodman, Fynn Heide, Dan Hendrycks, Cyrus Hodes, Bryan Low Kian Hsiang, Minlie Huang, Sami Jawhar, Wang Jingyu, Adam Tauman Kalai, Meindert Kamphuis, Mohan Kankanhalli, Subhash Kantamneni, Mathias Bonde Kirk, Thomas Kwa, Jeffrey Ladish, Kwok-Yan Lam, Wan Lee Sie, Taewhi Lee, Xiaojian Li, Jiajun Liu, Chaochao Lu, Yifan Mai, Richard Mallah, Julian Michael, Nick Moës, Simon Möller, Kihyuk Nam, Kwan Yee Ng, Mark Nitzberg, Besmira Nushi, Seán O hÉigeartaigh, Alejandro Ortega, Pierre Peigné, James Petrie, Benjamin Prud'Homme, Reihaneh Rabbany, Nayat Sanchez-Pi, Sarah Schwettmann, Buck Shlegeris, Saad Siddiqui, Aradhana Sinha, Martín Soto, Cheston Tan, Dong Ting, Robert Trager, Brian Tse, Anthony Tung K. H., Vanessa Wilfred, John Willes, Denise Wong, Wei Xu, Rongwu Xu, Yi Zeng, HongJiang Zhang, Djordje Žikelić  

**Link**: [PDF](https://arxiv.org/pdf/2506.20702)  

**Abstract**: Rapidly improving AI capabilities and autonomy hold significant promise of transformation, but are also driving vigorous debate on how to ensure that AI is safe, i.e., trustworthy, reliable, and secure. Building a trusted ecosystem is therefore essential -- it helps people embrace AI with confidence and gives maximal space for innovation while avoiding backlash.
The "2025 Singapore Conference on AI (SCAI): International Scientific Exchange on AI Safety" aimed to support research in this space by bringing together AI scientists across geographies to identify and synthesise research priorities in AI safety. This resulting report builds on the International AI Safety Report chaired by Yoshua Bengio and backed by 33 governments. By adopting a defence-in-depth model, this report organises AI safety research domains into three types: challenges with creating trustworthy AI systems (Development), challenges with evaluating their risks (Assessment), and challenges with monitoring and intervening after deployment (Control). 

---
# Whole-Body Conditioned Egocentric Video Prediction 

**Authors**: Yutong Bai, Danny Tran, Amir Bar, Yann LeCun, Trevor Darrell, Jitendra Malik  

**Link**: [PDF](https://arxiv.org/pdf/2506.21552)  

**Abstract**: We train models to Predict Ego-centric Video from human Actions (PEVA), given the past video and an action represented by the relative 3D body pose. By conditioning on kinematic pose trajectories, structured by the joint hierarchy of the body, our model learns to simulate how physical human actions shape the environment from a first-person point of view. We train an auto-regressive conditional diffusion transformer on Nymeria, a large-scale dataset of real-world egocentric video and body pose capture. We further design a hierarchical evaluation protocol with increasingly challenging tasks, enabling a comprehensive analysis of the model's embodied prediction and control abilities. Our work represents an initial attempt to tackle the challenges of modeling complex real-world environments and embodied agent behaviors with video prediction from the perspective of a human. 

---
# mTSBench: Benchmarking Multivariate Time Series Anomaly Detection and Model Selection at Scale 

**Authors**: Xiaona Zhou, Constantin Brif, Ismini Lourentzou  

**Link**: [PDF](https://arxiv.org/pdf/2506.21550)  

**Abstract**: Multivariate time series anomaly detection (MTS-AD) is critical in domains like healthcare, cybersecurity, and industrial monitoring, yet remains challenging due to complex inter-variable dependencies, temporal dynamics, and sparse anomaly labels. We introduce mTSBench, the largest benchmark to date for MTS-AD and unsupervised model selection, spanning 344 labeled time series across 19 datasets and 12 diverse application domains. mTSBench evaluates 24 anomaly detection methods, including large language model (LLM)-based detectors for multivariate time series, and systematically benchmarks unsupervised model selection techniques under standardized conditions. Consistent with prior findings, our results confirm that no single detector excels across datasets, underscoring the importance of model selection. However, even state-of-the-art selection methods remain far from optimal, revealing critical gaps. mTSBench provides a unified evaluation suite to enable rigorous, reproducible comparisons and catalyze future advances in adaptive anomaly detection and robust model selection. 

---
# HalluSegBench: Counterfactual Visual Reasoning for Segmentation Hallucination Evaluation 

**Authors**: Xinzhuo Li, Adheesh Juvekar, Xingyou Liu, Muntasir Wahed, Kiet A. Nguyen, Ismini Lourentzou  

**Link**: [PDF](https://arxiv.org/pdf/2506.21546)  

**Abstract**: Recent progress in vision-language segmentation has significantly advanced grounded visual understanding. However, these models often exhibit hallucinations by producing segmentation masks for objects not grounded in the image content or by incorrectly labeling irrelevant regions. Existing evaluation protocols for segmentation hallucination primarily focus on label or textual hallucinations without manipulating the visual context, limiting their capacity to diagnose critical failures. In response, we introduce HalluSegBench, the first benchmark specifically designed to evaluate hallucinations in visual grounding through the lens of counterfactual visual reasoning. Our benchmark consists of a novel dataset of 1340 counterfactual instance pairs spanning 281 unique object classes, and a set of newly introduced metrics that quantify hallucination sensitivity under visually coherent scene edits. Experiments on HalluSegBench with state-of-the-art vision-language segmentation models reveal that vision-driven hallucinations are significantly more prevalent than label-driven ones, with models often persisting in false segmentation, highlighting the need for counterfactual reasoning to diagnose grounding fidelity. 

---
# WorldVLA: Towards Autoregressive Action World Model 

**Authors**: Jun Cen, Chaohui Yu, Hangjie Yuan, Yuming Jiang, Siteng Huang, Jiayan Guo, Xin Li, Yibing Song, Hao Luo, Fan Wang, Deli Zhao, Hao Chen  

**Link**: [PDF](https://arxiv.org/pdf/2506.21539)  

**Abstract**: We present WorldVLA, an autoregressive action world model that unifies action and image understanding and generation. Our WorldVLA intergrates Vision-Language-Action (VLA) model and world model in one single framework. The world model predicts future images by leveraging both action and image understanding, with the purpose of learning the underlying physics of the environment to improve action generation. Meanwhile, the action model generates the subsequent actions based on image observations, aiding in visual understanding and in turn helps visual generation of the world model. We demonstrate that WorldVLA outperforms standalone action and world models, highlighting the mutual enhancement between the world model and the action model. In addition, we find that the performance of the action model deteriorates when generating sequences of actions in an autoregressive manner. This phenomenon can be attributed to the model's limited generalization capability for action prediction, leading to the propagation of errors from earlier actions to subsequent ones. To address this issue, we propose an attention mask strategy that selectively masks prior actions during the generation of the current action, which shows significant performance improvement in the action chunk generation task. 

---
# "What's Up, Doc?": Analyzing How Users Seek Health Information in Large-Scale Conversational AI Datasets 

**Authors**: Akshay Paruchuri, Maryam Aziz, Rohit Vartak, Ayman Ali, Best Uchehara, Xin Liu, Ishan Chatterjee, Monica Agrawal  

**Link**: [PDF](https://arxiv.org/pdf/2506.21532)  

**Abstract**: People are increasingly seeking healthcare information from large language models (LLMs) via interactive chatbots, yet the nature and inherent risks of these conversations remain largely unexplored. In this paper, we filter large-scale conversational AI datasets to achieve HealthChat-11K, a curated dataset of 11K real-world conversations composed of 25K user messages. We use HealthChat-11K and a clinician-driven taxonomy for how users interact with LLMs when seeking healthcare information in order to systematically study user interactions across 21 distinct health specialties. Our analysis reveals insights into the nature of how and why users seek health information, such as common interactions, instances of incomplete context, affective behaviors, and interactions (e.g., leading questions) that can induce sycophancy, underscoring the need for improvements in the healthcare support capabilities of LLMs deployed as conversational AI. Code and artifacts to retrieve our analyses and combine them into a curated dataset can be found here: this https URL 

---
# Potemkin Understanding in Large Language Models 

**Authors**: Marina Mancoridis, Bec Weeks, Keyon Vafa, Sendhil Mullainathan  

**Link**: [PDF](https://arxiv.org/pdf/2506.21521)  

**Abstract**: Large language models (LLMs) are regularly evaluated using benchmark datasets. But what justifies making inferences about an LLM's capabilities based on its answers to a curated set of questions? This paper first introduces a formal framework to address this question. The key is to note that the benchmarks used to test LLMs -- such as AP exams -- are also those used to test people. However, this raises an implication: these benchmarks are only valid tests if LLMs misunderstand concepts in ways that mirror human misunderstandings. Otherwise, success on benchmarks only demonstrates potemkin understanding: the illusion of understanding driven by answers irreconcilable with how any human would interpret a concept. We present two procedures for quantifying the existence of potemkins: one using a specially designed benchmark in three domains, the other using a general procedure that provides a lower-bound on their prevalence. We find that potemkins are ubiquitous across models, tasks, and domains. We also find that these failures reflect not just incorrect understanding, but deeper internal incoherence in concept representations. 

---
# skLEP: A Slovak General Language Understanding Benchmark 

**Authors**: Marek Šuppa, Andrej Ridzik, Daniel Hládek, Tomáš Javůrek, Viktória Ondrejová, Kristína Sásiková, Martin Tamajka, Marián Šimko  

**Link**: [PDF](https://arxiv.org/pdf/2506.21508)  

**Abstract**: In this work, we introduce skLEP, the first comprehensive benchmark specifically designed for evaluating Slovak natural language understanding (NLU) models. We have compiled skLEP to encompass nine diverse tasks that span token-level, sentence-pair, and document-level challenges, thereby offering a thorough assessment of model capabilities. To create this benchmark, we curated new, original datasets tailored for Slovak and meticulously translated established English NLU resources. Within this paper, we also present the first systematic and extensive evaluation of a wide array of Slovak-specific, multilingual, and English pre-trained language models using the skLEP tasks. Finally, we also release the complete benchmark data, an open-source toolkit facilitating both fine-tuning and evaluation of models, and a public leaderboard at this https URL in the hopes of fostering reproducibility and drive future research in Slovak NLU. 

---
# Process mining-driven modeling and simulation to enhance fault diagnosis in cyber-physical systems 

**Authors**: Francesco Vitale, Nicola Dall'Ora, Sebastiano Gaiardelli, Enrico Fraccaroli, Nicola Mazzocca, Franco Fummi  

**Link**: [PDF](https://arxiv.org/pdf/2506.21502)  

**Abstract**: Fault diagnosis in Cyber-Physical Systems (CPSs) is essential for ensuring system dependability and operational efficiency by accurately detecting anomalies and identifying their root causes. However, the manual modeling of faulty behaviors often demands extensive domain expertise and produces models that are complex, error-prone, and difficult to interpret. To address this challenge, we present a novel unsupervised fault diagnosis methodology that integrates collective anomaly detection in multivariate time series, process mining, and stochastic simulation. Initially, collective anomalies are detected from low-level sensor data using multivariate time-series analysis. These anomalies are then transformed into structured event logs, enabling the discovery of interpretable process models through process mining. By incorporating timing distributions into the extracted Petri nets, the approach supports stochastic simulation of faulty behaviors, thereby enhancing root cause analysis and behavioral understanding. The methodology is validated using the Robotic Arm Dataset (RoAD), a widely recognized benchmark in smart manufacturing. Experimental results demonstrate its effectiveness in modeling, simulating, and classifying faulty behaviors in CPSs. This enables the creation of comprehensive fault dictionaries that support predictive maintenance and the development of digital twins for industrial environments. 

---
# TITAN: Query-Token based Domain Adaptive Adversarial Learning 

**Authors**: Tajamul Ashraf, Janibul Bashir  

**Link**: [PDF](https://arxiv.org/pdf/2506.21484)  

**Abstract**: We focus on the source-free domain adaptive object detection (SF-DAOD) problem when source data is unavailable during adaptation and the model must adapt to an unlabeled target domain. The majority of approaches for the problem employ a self-supervised approach using a student-teacher (ST) framework where pseudo-labels are generated via a source-pretrained model for further fine-tuning. We observe that the performance of a student model often degrades drastically, due to the collapse of the teacher model, primarily caused by high noise in pseudo-labels, resulting from domain bias, discrepancies, and a significant domain shift across domains. To obtain reliable pseudo-labels, we propose a Target-based Iterative Query-Token Adversarial Network (TITAN), which separates the target images into two subsets: those similar to the source (easy) and those dissimilar (hard). We propose a strategy to estimate variance to partition the target domain. This approach leverages the insight that higher detection variances correspond to higher recall and greater similarity to the source domain. Also, we incorporate query-token-based adversarial modules into a student-teacher baseline framework to reduce the domain gaps between two feature representations. Experiments conducted on four natural imaging datasets and two challenging medical datasets have substantiated the superior performance of TITAN compared to existing state-of-the-art (SOTA) methodologies. We report an mAP improvement of +22.7, +22.2, +21.1, and +3.7 percent over the current SOTA on C2F, C2B, S2C, and K2C benchmarks, respectively. 

---
# SmoothSinger: A Conditional Diffusion Model for Singing Voice Synthesis with Multi-Resolution Architecture 

**Authors**: Kehan Sui, Jinxu Xiang, Fang Jin  

**Link**: [PDF](https://arxiv.org/pdf/2506.21478)  

**Abstract**: Singing voice synthesis (SVS) aims to generate expressive and high-quality vocals from musical scores, requiring precise modeling of pitch, duration, and articulation. While diffusion-based models have achieved remarkable success in image and video generation, their application to SVS remains challenging due to the complex acoustic and musical characteristics of singing, often resulting in artifacts that degrade naturalness. In this work, we propose SmoothSinger, a conditional diffusion model designed to synthesize high quality and natural singing voices. Unlike prior methods that depend on vocoders as a final stage and often introduce distortion, SmoothSinger refines low-quality synthesized audio directly in a unified framework, mitigating the degradation associated with two-stage pipelines. The model adopts a reference-guided dual-branch architecture, using low-quality audio from any baseline system as a reference to guide the denoising process, enabling more expressive and context-aware synthesis. Furthermore, it enhances the conventional U-Net with a parallel low-frequency upsampling path, allowing the model to better capture pitch contours and long term spectral dependencies. To improve alignment during training, we replace reference audio with degraded ground truth audio, addressing temporal mismatch between reference and target signals. Experiments on the Opencpop dataset, a large-scale Chinese singing corpus, demonstrate that SmoothSinger achieves state-of-the-art results in both objective and subjective evaluations. Extensive ablation studies confirm its effectiveness in reducing artifacts and improving the naturalness of synthesized voices. 

---
# Optimising 4th-Order Runge-Kutta Methods: A Dynamic Heuristic Approach for Efficiency and Low Storage 

**Authors**: Gavin Lee Goodship, Luis Miralles-Pechuan, Stephen O'Sullivan  

**Link**: [PDF](https://arxiv.org/pdf/2506.21465)  

**Abstract**: Extended Stability Runge-Kutta (ESRK) methods are crucial for solving large-scale computational problems in science and engineering, including weather forecasting, aerodynamic analysis, and complex biological modelling. However, balancing accuracy, stability, and computational efficiency remains challenging, particularly for high-order, low-storage schemes. This study introduces a hybrid Genetic Algorithm (GA) and Reinforcement Learning (RL) approach for automated heuristic discovery, optimising low-storage ESRK methods. Unlike traditional approaches that rely on manually designed heuristics or exhaustive numerical searches, our method leverages GA-driven mutations for search-space exploration and an RL-inspired state transition mechanism to refine heuristic selection dynamically. This enables systematic parameter reduction, preserving fourth-order accuracy while significantly improving computational this http URL proposed GA-RL heuristic optimisation framework is validated through rigorous testing on benchmark problems, including the 1D and 2D Brusselator systems and the steady-state Navier-Stokes equations. The best-performing heuristic achieves a 25\% reduction in IPOPT runtime compared to traditional ESRK optimisation processes while maintaining numerical stability and accuracy. These findings demonstrate the potential of adaptive heuristic discovery to improve resource efficiency in high-fidelity simulations and broaden the applicability of low-storage Runge-Kutta methods in real-world computational fluid dynamics, physics simulations, and other demanding fields. This work establishes a new paradigm in heuristic optimisation for numerical methods, opening pathways for further exploration using Deep RL and AutoML-based heuristic search 

---
# Domain Knowledge-Enhanced LLMs for Fraud and Concept Drift Detection 

**Authors**: Ali Şenol, Garima Agrawal, Huan Liu  

**Link**: [PDF](https://arxiv.org/pdf/2506.21443)  

**Abstract**: Detecting deceptive conversations on dynamic platforms is increasingly difficult due to evolving language patterns and Concept Drift (CD)\-i.e., semantic or topical shifts that alter the context or intent of interactions over time. These shifts can obscure malicious intent or mimic normal dialogue, making accurate classification challenging. While Large Language Models (LLMs) show strong performance in natural language tasks, they often struggle with contextual ambiguity and hallucinations in risk\-sensitive scenarios. To address these challenges, we present a Domain Knowledge (DK)\-Enhanced LLM framework that integrates pretrained LLMs with structured, task\-specific insights to perform fraud and concept drift detection. The proposed architecture consists of three main components: (1) a DK\-LLM module to detect fake or deceptive conversations; (2) a drift detection unit (OCDD) to determine whether a semantic shift has occurred; and (3) a second DK\-LLM module to classify the drift as either benign or fraudulent. We first validate the value of domain knowledge using a fake review dataset and then apply our full framework to SEConvo, a multiturn dialogue dataset that includes various types of fraud and spam attacks. Results show that our system detects fake conversations with high accuracy and effectively classifies the nature of drift. Guided by structured prompts, the LLaMA\-based implementation achieves 98\% classification accuracy. Comparative studies against zero\-shot baselines demonstrate that incorporating domain knowledge and drift awareness significantly improves performance, interpretability, and robustness in high\-stakes NLP applications. 

---
# Scalable Bayesian Low-Rank Adaptation of Large Language Models via Stochastic Variational Subspace Inference 

**Authors**: Colin Samplawski, Adam D. Cobb, Manoj Acharya, Ramneet Kaur, Susmit Jha  

**Link**: [PDF](https://arxiv.org/pdf/2506.21408)  

**Abstract**: Despite their widespread use, large language models (LLMs) are known to hallucinate incorrect information and be poorly calibrated. This makes the uncertainty quantification of these models of critical importance, especially in high-stakes domains, such as autonomy and healthcare. Prior work has made Bayesian deep learning-based approaches to this problem more tractable by performing inference over the low-rank adaptation (LoRA) parameters of a fine-tuned model. While effective, these approaches struggle to scale to larger LLMs due to requiring further additional parameters compared to LoRA. In this work we present $\textbf{Scala}$ble $\textbf{B}$ayesian $\textbf{L}$ow-Rank Adaptation via Stochastic Variational Subspace Inference (ScalaBL). We perform Bayesian inference in an $r$-dimensional subspace, for LoRA rank $r$. By repurposing the LoRA parameters as projection matrices, we are able to map samples from this subspace into the full weight space of the LLM. This allows us to learn all the parameters of our approach using stochastic variational inference. Despite the low dimensionality of our subspace, we are able to achieve competitive performance with state-of-the-art approaches while only requiring ${\sim}1000$ additional parameters. Furthermore, it allows us to scale up to the largest Bayesian LLM to date, with four times as a many base parameters as prior work. 

---
# Leveraging LLM-Assisted Query Understanding for Live Retrieval-Augmented Generation 

**Authors**: Guanting Dong, Xiaoxi Li, Yuyao Zhang, Mengjie Deng  

**Link**: [PDF](https://arxiv.org/pdf/2506.21384)  

**Abstract**: Real-world live retrieval-augmented generation (RAG) systems face significant challenges when processing user queries that are often noisy, ambiguous, and contain multiple intents. While RAG enhances large language models (LLMs) with external knowledge, current systems typically struggle with such complex inputs, as they are often trained or evaluated on cleaner data. This paper introduces Omni-RAG, a novel framework designed to improve the robustness and effectiveness of RAG systems in live, open-domain settings. Omni-RAG employs LLM-assisted query understanding to preprocess user inputs through three key modules: (1) Deep Query Understanding and Decomposition, which utilizes LLMs with tailored prompts to denoise queries (e.g., correcting spelling errors) and decompose multi-intent queries into structured sub-queries; (2) Intent-Aware Knowledge Retrieval, which performs retrieval for each sub-query from a corpus (i.e., FineWeb using OpenSearch) and aggregates the results; and (3) Reranking and Generation, where a reranker (i.e., BGE) refines document selection before a final response is generated by an LLM (i.e., Falcon-10B) using a chain-of-thought prompt. Omni-RAG aims to bridge the gap between current RAG capabilities and the demands of real-world applications, such as those highlighted by the SIGIR 2025 LiveRAG Challenge, by robustly handling complex and noisy queries. 

---
# Temporal-Aware Graph Attention Network for Cryptocurrency Transaction Fraud Detection 

**Authors**: Zhi Zheng, Bochuan Zhou, Yuping Song  

**Link**: [PDF](https://arxiv.org/pdf/2506.21382)  

**Abstract**: Cryptocurrency transaction fraud detection faces the dual challenges of increasingly complex transaction patterns and severe class imbalance. Traditional methods rely on manual feature engineering and struggle to capture temporal and structural dependencies in transaction networks. This paper proposes an Augmented Temporal-aware Graph Attention Network (ATGAT) that enhances detection performance through three modules: (1) designing an advanced temporal embedding module that fuses multi-scale time difference features with periodic position encoding; (2) constructing a temporal-aware triple attention mechanism that jointly optimizes structural, temporal, and global context attention; (3) employing weighted BCE loss to address class imbalance. Experiments on the Elliptic++ cryptocurrency dataset demonstrate that ATGAT achieves an AUC of 0.9130, representing a 9.2% improvement over the best traditional method XGBoost, 12.0% over GCN, and 10.0% over standard GAT. This method not only validates the enhancement effect of temporal awareness and triple attention mechanisms on graph neural networks, but also provides financial institutions with more reliable fraud detection tools, with its design principles generalizable to other temporal graph anomaly detection tasks. 

---
# Pay Attention to Small Weights 

**Authors**: Chao Zhou, Tom Jacobs, Advait Gadhikar, Rebekka Burkholz  

**Link**: [PDF](https://arxiv.org/pdf/2506.21374)  

**Abstract**: Finetuning large pretrained neural networks is known to be resource-intensive, both in terms of memory and computational cost. To mitigate this, a common approach is to restrict training to a subset of the model parameters. By analyzing the relationship between gradients and weights during finetuning, we observe a notable pattern: large gradients are often associated with small-magnitude weights. This correlation is more pronounced in finetuning settings than in training from scratch. Motivated by this observation, we propose NANOADAM, which dynamically updates only the small-magnitude weights during finetuning and offers several practical advantages: first, this criterion is gradient-free -- the parameter subset can be determined without gradient computation; second, it preserves large-magnitude weights, which are likely to encode critical features learned during pretraining, thereby reducing the risk of catastrophic forgetting; thirdly, it permits the use of larger learning rates and consistently leads to better generalization performance in experiments. We demonstrate this for both NLP and vision tasks. 

---
# Real-time and personalized product recommendations for large e-commerce platforms 

**Authors**: Matteo Tolloso, Davide Bacciu, Shahab Mokarizadeh, Marco Varesi  

**Link**: [PDF](https://arxiv.org/pdf/2506.21368)  

**Abstract**: We present a methodology to provide real-time and personalized product recommendations for large e-commerce platforms, specifically focusing on fashion retail. Our approach aims to achieve accurate and scalable recommendations with minimal response times, ensuring user satisfaction, leveraging Graph Neural Networks and parsimonious learning methodologies. Extensive experimentation with datasets from one of the largest e-commerce platforms demonstrates the effectiveness of our approach in forecasting purchase sequences and handling multi-interaction scenarios, achieving efficient personalized recommendations under real-world constraints. 

---
# rQdia: Regularizing Q-Value Distributions With Image Augmentation 

**Authors**: Sam Lerman, Jing Bi  

**Link**: [PDF](https://arxiv.org/pdf/2506.21367)  

**Abstract**: rQdia regularizes Q-value distributions with augmented images in pixel-based deep reinforcement learning. With a simple auxiliary loss, that equalizes these distributions via MSE, rQdia boosts DrQ and SAC on 9/12 and 10/12 tasks respectively in the MuJoCo Continuous Control Suite from pixels, and Data-Efficient Rainbow on 18/26 Atari Arcade environments. Gains are measured in both sample efficiency and longer-term training. Moreover, the addition of rQdia finally propels model-free continuous control from pixels over the state encoding baseline. 

---
# CA-I2P: Channel-Adaptive Registration Network with Global Optimal Selection 

**Authors**: Zhixin Cheng, Jiacheng Deng, Xinjun Li, Xiaotian Yin, Bohao Liao, Baoqun Yin, Wenfei Yang, Tianzhu Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2506.21364)  

**Abstract**: Detection-free methods typically follow a coarse-to-fine pipeline, extracting image and point cloud features for patch-level matching and refining dense pixel-to-point correspondences. However, differences in feature channel attention between images and point clouds may lead to degraded matching results, ultimately impairing registration accuracy. Furthermore, similar structures in the scene could lead to redundant correspondences in cross-modal matching. To address these issues, we propose Channel Adaptive Adjustment Module (CAA) and Global Optimal Selection Module (GOS). CAA enhances intra-modal features and suppresses cross-modal sensitivity, while GOS replaces local selection with global optimization. Experiments on RGB-D Scenes V2 and 7-Scenes demonstrate the superiority of our method, achieving state-of-the-art performance in image-to-point cloud registration. 

---
# A Systematic Review of Human-AI Co-Creativity 

**Authors**: Saloni Singh, Koen Hndriks, Drik Heylen, Kim Baraka  

**Link**: [PDF](https://arxiv.org/pdf/2506.21333)  

**Abstract**: The co creativity community is making significant progress in developing more sophisticated and tailored systems to support and enhance human creativity. Design considerations from prior work can serve as a valuable and efficient foundation for future systems. To support this effort, we conducted a systematic literature review of 62 papers on co-creative systems. These papers cover a diverse range of applications, including visual arts, design, and writing, where the AI acts not just as a tool but as an active collaborator in the creative process. From this review, we identified several key dimensions relevant to system design: phase of the creative process, creative task, proactive behavior of the system, user control, system embodiment, and AI model type. Our findings suggest that systems offering high user control lead to greater satisfaction, trust, and a stronger sense of ownership over creative outcomes. Furthermore, proactive systems, when adaptive and context sensitive, can enhance collaboration. We also extracted 24 design considerations, highlighting the value of encouraging users to externalize their thoughts and of increasing the system's social presence and transparency to foster trust. Despite recent advancements, important gaps remain, such as limited support for early creative phases like problem clarification, and challenges related to user adaptation to AI systems. 

---
# Holistic Surgical Phase Recognition with Hierarchical Input Dependent State Space Models 

**Authors**: Haoyang Wu, Tsun-Hsuan Wang, Mathias Lechner, Ramin Hasani, Jennifer A. Eckhoff, Paul Pak, Ozanan R. Meireles, Guy Rosman, Yutong Ban, Daniela Rus  

**Link**: [PDF](https://arxiv.org/pdf/2506.21330)  

**Abstract**: Surgical workflow analysis is essential in robot-assisted surgeries, yet the long duration of such procedures poses significant challenges for comprehensive video analysis. Recent approaches have predominantly relied on transformer models; however, their quadratic attention mechanism restricts efficient processing of lengthy surgical videos. In this paper, we propose a novel hierarchical input-dependent state space model that leverages the linear scaling property of state space models to enable decision making on full-length videos while capturing both local and global dynamics. Our framework incorporates a temporally consistent visual feature extractor, which appends a state space model head to a visual feature extractor to propagate temporal information. The proposed model consists of two key modules: a local-aggregation state space model block that effectively captures intricate local dynamics, and a global-relation state space model block that models temporal dependencies across the entire video. The model is trained using a hybrid discrete-continuous supervision strategy, where both signals of discrete phase labels and continuous phase progresses are propagated through the network. Experiments have shown that our method outperforms the current state-of-the-art methods by a large margin (+2.8% on Cholec80, +4.3% on MICCAI2016, and +12.9% on Heichole datasets). Code will be publicly available after paper acceptance. 

---
# On Uniform Weighted Deep Polynomial approximation 

**Authors**: Kingsley Yeon, Steven B. Damelin  

**Link**: [PDF](https://arxiv.org/pdf/2506.21306)  

**Abstract**: It is a classical result in rational approximation theory that certain non-smooth or singular functions, such as $|x|$ and $x^{1/p}$, can be efficiently approximated using rational functions with root-exponential convergence in terms of degrees of freedom \cite{Sta, GN}. In contrast, polynomial approximations admit only algebraic convergence by Jackson's theorem \cite{Lub2}. Recent work shows that composite polynomial architectures can recover exponential approximation rates even without smoothness \cite{KY}. In this work, we introduce and analyze a class of weighted deep polynomial approximants tailored for functions with asymmetric behavior-growing unbounded on one side and decaying on the other. By multiplying a learnable deep polynomial with a one-sided weight, we capture both local non-smoothness and global growth. We show numerically that this framework outperforms Taylor, Chebyshev, and standard deep polynomial approximants, even when all use the same number of parameters. To optimize these approximants in practice, we propose a stable graph-based parameterization strategy building on \cite{Jar}. 

---
# Exploring Adapter Design Tradeoffs for Low Resource Music Generation 

**Authors**: Atharva Mehta, Shivam Chauhan, Monojit Choudhury  

**Link**: [PDF](https://arxiv.org/pdf/2506.21298)  

**Abstract**: Fine-tuning large-scale music generation models, such as MusicGen and Mustango, is a computationally expensive process, often requiring updates to billions of parameters and, therefore, significant hardware resources. Parameter-Efficient Fine-Tuning (PEFT) techniques, particularly adapter-based methods, have emerged as a promising alternative, enabling adaptation with minimal trainable parameters while preserving model performance. However, the design choices for adapters, including their architecture, placement, and size, are numerous, and it is unclear which of these combinations would produce optimal adapters and why, for a given case of low-resource music genre. In this paper, we attempt to answer this question by studying various adapter configurations for two AI music models, MusicGen and Mustango, on two genres: Hindustani Classical and Turkish Makam music.
Our findings reveal distinct trade-offs: convolution-based adapters excel in capturing fine-grained local musical details such as ornamentations and short melodic phrases, while transformer-based adapters better preserve long-range dependencies crucial for structured improvisation. Additionally, we analyze computational resource requirements across different adapter scales, demonstrating how mid-sized adapters (40M parameters) achieve an optimal balance between expressivity and quality. Furthermore, we find that Mustango, a diffusion-based model, generates more diverse outputs with better adherence to the description in the input prompt while lacking in providing stability in notes, rhythm alignment, and aesthetics. Also, it is computationally intensive and requires significantly more time to train. In contrast, autoregressive models like MusicGen offer faster training and are more efficient, and can produce better quality output in comparison, but have slightly higher redundancy in their generations. 

---
# Detecting Referring Expressions in Visually Grounded Dialogue with Autoregressive Language Models 

**Authors**: Bram Willemsen, Gabriel Skantze  

**Link**: [PDF](https://arxiv.org/pdf/2506.21294)  

**Abstract**: In this paper, we explore the use of a text-only, autoregressive language modeling approach for the extraction of referring expressions from visually grounded dialogue. More specifically, the aim is to investigate the extent to which the linguistic context alone can inform the detection of mentions that have a (visually perceivable) referent in the visual context of the conversation. To this end, we adapt a pretrained large language model (LLM) to perform a relatively course-grained annotation of mention spans in unfolding conversations by demarcating mention span boundaries in text via next-token prediction. Our findings indicate that even when using a moderately sized LLM, relatively small datasets, and parameter-efficient fine-tuning, a text-only approach can be effective, highlighting the relative importance of the linguistic context for this task. Nevertheless, we argue that the task represents an inherently multimodal problem and discuss limitations fundamental to unimodal approaches. 

---
# Small Encoders Can Rival Large Decoders in Detecting Groundedness 

**Authors**: Istabrak Abbes, Gabriele Prato, Quentin Fournier, Fernando Rodriguez, Alaa Boukhary, Adam Elwood, Sarath Chandar  

**Link**: [PDF](https://arxiv.org/pdf/2506.21288)  

**Abstract**: Augmenting large language models (LLMs) with external context significantly improves their performance in natural language processing (NLP) tasks. However, LLMs struggle to answer queries reliably when the provided context lacks information, often resorting to ungrounded speculation or internal knowledge. Groundedness - generating responses strictly supported by the context - is essential for ensuring factual consistency and trustworthiness. This study focuses on detecting whether a given query is grounded in a document provided in context before the costly answer generation by LLMs. Such a detection mechanism can significantly reduce both inference time and resource consumption. We show that lightweight, task specific encoder models such as RoBERTa and NomicBERT, fine-tuned on curated datasets, can achieve accuracy comparable to state-of-the-art LLMs, such as Llama3 8B and GPT4o, in groundedness detection while reducing inference latency by orders of magnitude. The code is available at : this https URL 

---
# Hyperspherical Variational Autoencoders Using Efficient Spherical Cauchy Distribution 

**Authors**: Lukas Sablica, Kurt Hornik  

**Link**: [PDF](https://arxiv.org/pdf/2506.21278)  

**Abstract**: We propose a novel variational autoencoder (VAE) architecture that employs a spherical Cauchy (spCauchy) latent distribution. Unlike traditional Gaussian latent spaces or the widely used von Mises-Fisher (vMF) distribution, spCauchy provides a more natural hyperspherical representation of latent variables, better capturing directional data while maintaining flexibility. Its heavy-tailed nature prevents over-regularization, ensuring efficient latent space utilization while offering a more expressive representation. Additionally, spCauchy circumvents the numerical instabilities inherent to vMF, which arise from computing normalization constants involving Bessel functions. Instead, it enables a fully differentiable and efficient reparameterization trick via Möbius transformations, allowing for stable and scalable training. The KL divergence can be computed through a rapidly converging power series, eliminating concerns of underflow or overflow associated with evaluation of ratios of hypergeometric functions. These properties make spCauchy a compelling alternative for VAEs, offering both theoretical advantages and practical efficiency in high-dimensional generative modeling. 

---
# Integrating Vehicle Acoustic Data for Enhanced Urban Traffic Management: A Study on Speed Classification in Suzhou 

**Authors**: Pengfei Fan, Yuli Zhang, Xinheng Wang, Ruiyuan Jiang, Hankang Gu, Dongyao Jia, Shangbo Wang  

**Link**: [PDF](https://arxiv.org/pdf/2506.21269)  

**Abstract**: This study presents and publicly releases the Suzhou Urban Road Acoustic Dataset (SZUR-Acoustic Dataset), which is accompanied by comprehensive data-acquisition protocols and annotation guidelines to ensure transparency and reproducibility of the experimental workflow. To model the coupling between vehicular noise and driving speed, we propose a bimodal-feature-fusion deep convolutional neural network (BMCNN). During preprocessing, an adaptive denoising and normalization strategy is applied to suppress environmental background interference; in the network architecture, parallel branches extract Mel-frequency cepstral coefficients (MFCCs) and wavelet-packet energy features, which are subsequently fused via a cross-modal attention mechanism in the intermediate feature space to fully exploit time-frequency information. Experimental results demonstrate that BMCNN achieves a classification accuracy of 87.56% on the SZUR-Acoustic Dataset and 96.28% on the public IDMT-Traffic dataset. Ablation studies and robustness tests on the Suzhou dataset further validate the contributions of each module to performance improvement and overfitting mitigation. The proposed acoustics-based speed classification method can be integrated into smart-city traffic management systems for real-time noise monitoring and speed estimation, thereby optimizing traffic flow control, reducing roadside noise pollution, and supporting sustainable urban planning. 

---
# DiLoCoX: A Low-Communication Large-Scale Training Framework for Decentralized Cluster 

**Authors**: Ji Qi, WenPeng Zhu, Li Li, Ming Wu, YingJun Wu, Wu He, Xun Gao, Jason Zeng, Michael Heinrich  

**Link**: [PDF](https://arxiv.org/pdf/2506.21263)  

**Abstract**: The distributed training of foundation models, particularly large language models (LLMs), demands a high level of communication. Consequently, it is highly dependent on a centralized cluster with fast and reliable interconnects. Can we conduct training on slow networks and thereby unleash the power of decentralized clusters when dealing with models exceeding 100 billion parameters? In this paper, we propose DiLoCoX, a low-communication large-scale decentralized cluster training framework. It combines Pipeline Parallelism with Dual Optimizer Policy, One-Step-Delay Overlap of Communication and Local Training, and an Adaptive Gradient Compression Scheme. This combination significantly improves the scale of parameters and the speed of model pre-training. We justify the benefits of one-step-delay overlap of communication and local training, as well as the adaptive gradient compression scheme, through a theoretical analysis of convergence. Empirically, we demonstrate that DiLoCoX is capable of pre-training a 107B foundation model over a 1Gbps network. Compared to vanilla AllReduce, DiLoCoX can achieve a 357x speedup in distributed training while maintaining negligible degradation in model convergence. To the best of our knowledge, this is the first decentralized training framework successfully applied to models with over 100 billion parameters. 

---
# Agent-RewardBench: Towards a Unified Benchmark for Reward Modeling across Perception, Planning, and Safety in Real-World Multimodal Agents 

**Authors**: Tianyi Men, Zhuoran Jin, Pengfei Cao, Yubo Chen, Kang Liu, Jun Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2506.21252)  

**Abstract**: As Multimodal Large Language Models (MLLMs) advance, multimodal agents show promise in real-world tasks like web navigation and embodied intelligence. However, due to limitations in a lack of external feedback, these agents struggle with self-correction and generalization. A promising approach is to use reward models as external feedback, but there is no clear on how to select reward models for agents. Thus, there is an urgent need to build a reward bench targeted at agents. To address these challenges, we propose Agent-RewardBench, a benchmark designed to evaluate reward modeling ability in MLLMs. The benchmark is characterized by three key features: (1) Multiple dimensions and real-world agent scenarios evaluation. It covers perception, planning, and safety with 7 scenarios; (2) Step-level reward evaluation. It allows for the assessment of agent capabilities at the individual steps of a task, providing a more granular view of performance during the planning process; and (3) Appropriately difficulty and high-quality. We carefully sample from 10 diverse models, difficulty control to maintain task challenges, and manual verification to ensure the integrity of the data. Experiments demonstrate that even state-of-the-art multimodal models show limited performance, highlighting the need for specialized training in agent reward modeling. Code is available at github. 

---
# From On-chain to Macro: Assessing the Importance of Data Source Diversity in Cryptocurrency Market Forecasting 

**Authors**: Giorgos Demosthenous, Chryssis Georgiou, Eliada Polydorou  

**Link**: [PDF](https://arxiv.org/pdf/2506.21246)  

**Abstract**: This study investigates the impact of data source diversity on the performance of cryptocurrency forecasting models by integrating various data categories, including technical indicators, on-chain metrics, sentiment and interest metrics, traditional market indices, and macroeconomic indicators. We introduce the Crypto100 index, representing the top 100 cryptocurrencies by market capitalization, and propose a novel feature reduction algorithm to identify the most impactful and resilient features from diverse data sources. Our comprehensive experiments demonstrate that data source diversity significantly enhances the predictive performance of forecasting models across different time horizons. Key findings include the paramount importance of on-chain metrics for both short-term and long-term predictions, the growing relevance of traditional market indices and macroeconomic indicators for longer-term forecasts, and substantial improvements in model accuracy when diverse data sources are utilized. These insights help demystify the short-term and long-term driving factors of the cryptocurrency market and lay the groundwork for developing more accurate and resilient forecasting models. 

---
# $T^3$: Multi-level Tree-based Automatic Program Repair with Large Language Models 

**Authors**: Quanming Liu, Xupeng Bu, Zhichao Yan, Ru Li  

**Link**: [PDF](https://arxiv.org/pdf/2506.21211)  

**Abstract**: Automatic Program Repair (APR) is a core technology in software development and maintenance, with aims to enable automated defect repair with minimal human intervention. In recent years, the substantial advancements in Large Language Models (LLMs) and the Chain-of-Thought (CoT) techniques have significantly enhanced the reasoning capabilities of these models. However, due to the complex logic and multi-step reasoning ability needed, the application of CoT techniques in the APR domain remains insufficient. This study systematically evaluates the performance of several common CoT techniques in APR tasks and proposes an innovative framework $T^3$, which integrates the powerful reasoning capabilities of LLMs with tree search, effectively improving the precision of generating candidate repair solutions. Furthermore, $T^3$ provides valuable guidance for optimizing sample selection and repair strategies in APR tasks, establishing a robust framework for achieving efficient automated debugging. 

---
# BitMark for Infinity: Watermarking Bitwise Autoregressive Image Generative Models 

**Authors**: Louis Kerner, Michel Meintz, Bihe Zhao, Franziska Boenisch, Adam Dziedzic  

**Link**: [PDF](https://arxiv.org/pdf/2506.21209)  

**Abstract**: State-of-the-art text-to-image models like Infinity generate photorealistic images at an unprecedented speed. These models operate in a bitwise autoregressive manner over a discrete set of tokens that is practically infinite in size. However, their impressive generative power comes with a growing risk: as their outputs increasingly populate the Internet, they are likely to be scraped and reused as training data-potentially by the very same models. This phenomenon has been shown to lead to model collapse, where repeated training on generated content, especially from the models' own previous versions, causes a gradual degradation in performance. A promising mitigation strategy is watermarking, which embeds human-imperceptible yet detectable signals into generated images-enabling the identification of generated content. In this work, we introduce BitMark, a robust bitwise watermarking framework for Infinity. Our method embeds a watermark directly at the bit level of the token stream across multiple scales (also referred to as resolutions) during Infinity's image generation process. Our bitwise watermark subtly influences the bits to preserve visual fidelity and generation speed while remaining robust against a spectrum of removal techniques. Furthermore, it exhibits high radioactivity, i.e., when watermarked generated images are used to train another image generative model, this second model's outputs will also carry the watermark. The radioactive traces remain detectable even when only fine-tuning diffusion or image autoregressive models on images watermarked with our BitMark. Overall, our approach provides a principled step toward preventing model collapse in image generative models by enabling reliable detection of generated outputs. 

---
# Task-Aware KV Compression For Cost-Effective Long Video Understanding 

**Authors**: Minghao Qin, Yan Shu, Peitian Zhang, Kun Lun, Huaying Yuan, Juenjie Zhou, Shitao Xiao, Bo Zhao, Zheng Liu  

**Link**: [PDF](https://arxiv.org/pdf/2506.21184)  

**Abstract**: Long-video understanding (LVU) remains a severe challenge for existing multimodal large language models (MLLMs), primarily due to the prohibitive computational cost. Recent approaches have explored KV compression to mitigate this issue, but they often suffer from significant information loss at high compression ratios. In this paper, we introduce Video-X^2L, which flexibly preserves critical video information for each LVU task. Video-X^2L involves two key operations. The first one is called bi-level KV compression. During the MLLM's pre-filling stage, Video-X^2L generates two types of compressed KVs: low-compression KVs (L-KVs) to capture fine-grained video details and high-compression KVs (H-KVs) to offer compact video representations. The second one is called selective KV re-loading. During the MLLM's decoding stage, Video-X^2L selectively re-loads L-KVs for the most critical video chunks while using H-KVs for other less important ones. This allows the MLLM to fully utilize task-specific information while maintaining the overall compactness. Video-X^2L is simple yet effective: it is free from additional training and directly compatible with existing KV-compressible MLLMs. We evaluate Video-X^2L with a variety of popular LVU benchmarks, including VideoMME, MLVU, LongVideoBench, and VNBench. Our experiment result shows that Video-X^2L outperforms existing KV-compression methods by a huge advantage while substantially saving the computation cost. 

---
# Maintaining MTEB: Towards Long Term Usability and Reproducibility of Embedding Benchmarks 

**Authors**: Isaac Chung, Imene Kerboua, Marton Kardos, Roman Solomatin, Kenneth Enevoldsen  

**Link**: [PDF](https://arxiv.org/pdf/2506.21182)  

**Abstract**: The Massive Text Embedding Benchmark (MTEB) has become a standard evaluation platform for text embedding models. While previous work has established the core benchmark methodology, this paper focuses on the engineering aspects that ensure MTEB's continued reproducibility and extensibility. We present our approach to maintaining robust continuous integration pipelines that validate dataset integrity, automate test execution, and assess benchmark results' generalizability. We detail the design choices that collectively enhance reproducibility and usability. Furthermore, we discuss our strategies for handling community contributions and extending the benchmark with new tasks and datasets. These engineering practices have been instrumental in scaling MTEB to become more comprehensive while maintaining quality and, ultimately, relevance to the field. Our experiences offer valuable insights for benchmark maintainers facing similar challenges in ensuring reproducibility and usability in machine learning evaluation frameworks. The MTEB repository is available at: this https URL 

---
# A Hierarchical Deep Learning Approach for Minority Instrument Detection 

**Authors**: Dylan Sechet, Francesca Bugiotti, Matthieu Kowalski, Edouard d'Hérouville, Filip Langiewicz  

**Link**: [PDF](https://arxiv.org/pdf/2506.21167)  

**Abstract**: Identifying instrument activities within audio excerpts is vital in music information retrieval, with significant implications for music cataloging and discovery. Prior deep learning endeavors in musical instrument recognition have predominantly emphasized instrument classes with ample data availability. Recent studies have demonstrated the applicability of hierarchical classification in detecting instrument activities in orchestral music, even with limited fine-grained annotations at the instrument level. Based on the Hornbostel-Sachs classification, such a hierarchical classification system is evaluated using the MedleyDB dataset, renowned for its diversity and richness concerning various instruments and music genres. This work presents various strategies to integrate hierarchical structures into models and tests a new class of models for hierarchical music prediction. This study showcases more reliable coarse-level instrument detection by bridging the gap between detailed instrument identification and group-level recognition, paving the way for further advancements in this domain. 

---
# A Novel Framework for Integrating 3D Ultrasound into Percutaneous Liver Tumour Ablation 

**Authors**: Shuwei Xing, Derek W. Cool, David Tessier, Elvis C.S. Chen, Terry M. Peters, Aaron Fenster  

**Link**: [PDF](https://arxiv.org/pdf/2506.21162)  

**Abstract**: 3D ultrasound (US) imaging has shown significant benefits in enhancing the outcomes of percutaneous liver tumour ablation. Its clinical integration is crucial for transitioning 3D US into the therapeutic domain. However, challenges of tumour identification in US images continue to hinder its broader adoption. In this work, we propose a novel framework for integrating 3D US into the standard ablation workflow. We present a key component, a clinically viable 2D US-CT/MRI registration approach, leveraging 3D US as an intermediary to reduce registration complexity. To facilitate efficient verification of the registration workflow, we also propose an intuitive multimodal image visualization technique. In our study, 2D US-CT/MRI registration achieved a landmark distance error of approximately 2-4 mm with a runtime of 0.22s per image pair. Additionally, non-rigid registration reduced the mean alignment error by approximately 40% compared to rigid registration. Results demonstrated the efficacy of the proposed 2D US-CT/MRI registration workflow. Our integration framework advanced the capabilities of 3D US imaging in improving percutaneous tumour ablation, demonstrating the potential to expand the therapeutic role of 3D US in clinical interventions. 

---
# Transformer-Based Spatial-Temporal Counterfactual Outcomes Estimation 

**Authors**: He Li, Haoang Chi, Mingyu Liu, Wanrong Huang, Liyang Xu, Wenjing Yang  

**Link**: [PDF](https://arxiv.org/pdf/2506.21154)  

**Abstract**: The real world naturally has dimensions of time and space. Therefore, estimating the counterfactual outcomes with spatial-temporal attributes is a crucial problem. However, previous methods are based on classical statistical models, which still have limitations in performance and generalization. This paper proposes a novel framework for estimating counterfactual outcomes with spatial-temporal attributes using the Transformer, exhibiting stronger estimation ability. Under mild assumptions, the proposed estimator within this framework is consistent and asymptotically normal. To validate the effectiveness of our approach, we conduct simulation experiments and real data experiments. Simulation experiments show that our estimator has a stronger estimation capability than baseline methods. Real data experiments provide a valuable conclusion to the causal effect of conflicts on forest loss in Colombia. The source code is available at this https URL. 

---
# Robust Deep Learning for Myocardial Scar Segmentation in Cardiac MRI with Noisy Labels 

**Authors**: Aida Moafi, Danial Moafi, Evgeny M. Mirkes, Gerry P. McCann, Abbas S. Alatrany, Jayanth R. Arnold, Mostafa Mehdipour Ghazi  

**Link**: [PDF](https://arxiv.org/pdf/2506.21151)  

**Abstract**: The accurate segmentation of myocardial scars from cardiac MRI is essential for clinical assessment and treatment planning. In this study, we propose a robust deep-learning pipeline for fully automated myocardial scar detection and segmentation by fine-tuning state-of-the-art models. The method explicitly addresses challenges of label noise from semi-automatic annotations, data heterogeneity, and class imbalance through the use of Kullback-Leibler loss and extensive data augmentation. We evaluate the model's performance on both acute and chronic cases and demonstrate its ability to produce accurate and smooth segmentations despite noisy labels. In particular, our approach outperforms state-of-the-art models like nnU-Net and shows strong generalizability in an out-of-distribution test set, highlighting its robustness across various imaging conditions and clinical tasks. These results establish a reliable foundation for automated myocardial scar quantification and support the broader clinical adoption of deep learning in cardiac imaging. 

---
# Linearity-based neural network compression 

**Authors**: Silas Dobler, Florian Lemmerich  

**Link**: [PDF](https://arxiv.org/pdf/2506.21146)  

**Abstract**: In neural network compression, most current methods reduce unnecessary parameters by measuring importance and redundancy. To augment already highly optimized existing solutions, we propose linearity-based compression as a novel way to reduce weights in a neural network. It is based on the intuition that with ReLU-like activation functions, neurons that are almost always activated behave linearly, allowing for merging of subsequent layers. We introduce the theory underlying this compression and evaluate our approach experimentally. Our novel method achieves a lossless compression down to 1/4 of the original model size in over the majority of tested models. Applying our method on already importance-based pruned models shows very little interference between different types of compression, demonstrating the option of successful combination of techniques. Overall, our work lays the foundation for a new type of compression method that enables smaller and ultimately more efficient neural network models. 

---
# DBConformer: Dual-Branch Convolutional Transformer for EEG Decoding 

**Authors**: Ziwei Wang, Hongbin Wang, Tianwang Jia, Xingyi He, Siyang Li, Dongrui Wu  

**Link**: [PDF](https://arxiv.org/pdf/2506.21140)  

**Abstract**: Electroencephalography (EEG)-based brain-computer interfaces (BCIs) transform spontaneous/evoked neural activity into control commands for external communication. While convolutional neural networks (CNNs) remain the mainstream backbone for EEG decoding, their inherently short receptive field makes it difficult to capture long-range temporal dependencies and global inter-channel relationships. Recent CNN-Transformer (Conformers) hybrids partially address this issue, but most adopt a serial design, resulting in suboptimal integration of local and global features, and often overlook explicit channel-wise modeling. To address these limitations, we propose DBConformer, a dual-branch convolutional Transformer network tailored for EEG decoding. It integrates a temporal Conformer to model long-range temporal dependencies and a spatial Conformer to extract inter-channel interactions, capturing both temporal dynamics and spatial patterns in EEG signals. A lightweight channel attention module further refines spatial representations by assigning data-driven importance to EEG channels. Extensive experiments on five motor imagery (MI) datasets and two seizure detection datasets under three evaluation settings demonstrate that DBConformer consistently outperforms 10 competitive baseline models, with over eight times fewer parameters than the high-capacity EEG Conformer baseline. Further, the visualization results confirm that the features extracted by DBConformer are physiologically interpretable and aligned with sensorimotor priors in MI. The superior performance and interpretability of DBConformer make it reliable for robust and explainable EEG decoding. Code is publicized at this https URL. 

---
# How Good Are Synthetic Requirements ? Evaluating LLM-Generated Datasets for AI4RE 

**Authors**: Abdelkarim El-Hajjami, Camille Salinesi  

**Link**: [PDF](https://arxiv.org/pdf/2506.21138)  

**Abstract**: The shortage of publicly available, labeled requirements datasets remains a major barrier to advancing Artificial Intelligence for Requirements Engineering (AI4RE). While Large Language Models offer promising capabilities for synthetic data generation, systematic approaches to control and optimize the quality of generated requirements remain underexplored. This paper presents Synthline v1, an enhanced Product Line approach for generating synthetic requirements data that extends our earlier v0 version with advanced generation strategies and curation techniques. We investigate four research questions assessing how prompting strategies, automated prompt optimization, and post-generation curation affect data quality across four classification tasks: defect detection, functional vs. non-functional, quality vs. non-quality, and security vs. non-security. Our evaluation shows that multi-sample prompting significantly boosts both utility and diversity over single-sample generation, with F1-score gains from 6 to 44 points. The use of PACE (Prompt Actor-Critic Editing) for automated prompt optimization yields task-dependent results, greatly improving functional classification (+32.5 points) but reducing performance on others. Interestingly, similarity-based curation improves diversity but often harms classification performance, indicating that some redundancy may help ML models. Most importantly, our results show that synthetic requirements can match or outperform human-authored ones for specific tasks, with synthetic data surpassing human data for security (+7.8 points) and defect classification (+15.4 points). These findings offer practical insights for AI4RE and chart a viable path to mitigating dataset scarcity through systematic synthetic generation. 

---
# Curriculum-Guided Antifragile Reinforcement Learning for Secure UAV Deconfliction under Observation-Space Attacks 

**Authors**: Deepak Kumar Panda, Adolfo Perrusquia, Weisi Guo  

**Link**: [PDF](https://arxiv.org/pdf/2506.21129)  

**Abstract**: Reinforcement learning (RL) policies deployed in safety-critical systems, such as unmanned aerial vehicle (UAV) navigation in dynamic airspace, are vulnerable to out-ofdistribution (OOD) adversarial attacks in the observation space. These attacks induce distributional shifts that significantly degrade value estimation, leading to unsafe or suboptimal decision making rendering the existing policy fragile. To address this vulnerability, we propose an antifragile RL framework designed to adapt against curriculum of incremental adversarial perturbations. The framework introduces a simulated attacker which incrementally increases the strength of observation-space perturbations which enables the RL agent to adapt and generalize across a wider range of OOD observations and anticipate previously unseen attacks. We begin with a theoretical characterization of fragility, formally defining catastrophic forgetting as a monotonic divergence in value function distributions with increasing perturbation strength. Building on this, we define antifragility as the boundedness of such value shifts and derive adaptation conditions under which forgetting is stabilized. Our method enforces these bounds through iterative expert-guided critic alignment using Wasserstein distance minimization across incrementally perturbed observations. We empirically evaluate the approach in a UAV deconfliction scenario involving dynamic 3D obstacles. Results show that the antifragile policy consistently outperforms standard and robust RL baselines when subjected to both projected gradient descent (PGD) and GPS spoofing attacks, achieving up to 15% higher cumulative reward and over 30% fewer conflict events. These findings demonstrate the practical and theoretical viability of antifragile reinforcement learning for secure and resilient decision-making in environments with evolving threat scenarios. 

---
# Robust Policy Switching for Antifragile Reinforcement Learning for UAV Deconfliction in Adversarial Environments 

**Authors**: Deepak Kumar Panda, Weisi Guo  

**Link**: [PDF](https://arxiv.org/pdf/2506.21127)  

**Abstract**: The increasing automation of navigation for unmanned aerial vehicles (UAVs) has exposed them to adversarial attacks that exploit vulnerabilities in reinforcement learning (RL) through sensor manipulation. Although existing robust RL methods aim to mitigate such threats, their effectiveness has limited generalization to out-of-distribution shifts from the optimal value distribution, as they are primarily designed to handle fixed perturbation. To address this limitation, this paper introduces an antifragile RL framework that enhances adaptability to broader distributional shifts by incorporating a switching mechanism based on discounted Thompson sampling (DTS). This mechanism dynamically selects among multiple robust policies to minimize adversarially induced state-action-value distribution shifts. The proposed approach first derives a diverse ensemble of action robust policies by accounting for a range of perturbations in the policy space. These policies are then modeled as a multiarmed bandit (MAB) problem, where DTS optimally selects policies in response to nonstationary Bernoulli rewards, effectively adapting to evolving adversarial strategies. Theoretical framework has also been provided where by optimizing the DTS to minimize the overall regrets due to distributional shift, results in effective adaptation against unseen adversarial attacks thus inducing antifragility. Extensive numerical simulations validate the effectiveness of the proposed framework in complex navigation environments with multiple dynamic three-dimensional obstacles and with stronger projected gradient descent (PGD) and spoofing attacks. Compared to conventional robust, non-adaptive RL methods, the antifragile approach achieves superior performance, demonstrating shorter navigation path lengths and a higher rate of conflict-free navigation trajectories compared to existing robust RL techniques 

---
# Progtuning: Progressive Fine-tuning Framework for Transformer-based Language Models 

**Authors**: Xiaoshuang Ji, Zhendong Zhao, Xiaojun Chen, Xin Zhao, Zeyao Liu  

**Link**: [PDF](https://arxiv.org/pdf/2506.21119)  

**Abstract**: Fine-tuning is a promising technique for leveraging Transformer-based language models in downstream tasks. As model sizes continue to grow, updating all model parameters becomes increasingly costly. Parameter-efficient fine-tuning methods effectively address this issue by selectively updating a small subset of parameters. However, fine-tuning and most existing parameter-efficient fine-tuning methods require updating the same number of parameters as the initial size, ignoring the unequal contribution across Transformer blocks and leading to extremely inefficient allocation of computing resources. In this paper, we propose Progtuning, the novel fine-tuning framework combined with progressive learning for Transformer-based language models. Specifically, Progtuning progressively reduces the number of updated transformer blocks based on the contribution. Remarkably, Progtuning optimizes resource allocation and reduces the number of updated parameters by approximately 25\%, while still maintaining competitive performance. And it also exhibits high adaptability with parameter-efficient fine-tuning methods, demonstrating excellent performance across various adaptation scenarios. 

---
# IPFormer-VideoLLM: Enhancing Multi-modal Video Understanding for Multi-shot Scenes 

**Authors**: Yujia Liang, Jile Jiao, Zhicheng Wang, Xuetao Feng, Zixuan Ye, Yuan Wang, Hao Lu  

**Link**: [PDF](https://arxiv.org/pdf/2506.21116)  

**Abstract**: Video Large Language Models (VideoLLMs) have demonstrated remarkable understanding capabilities, but are found struggling to tackle multi-shot scenarios,e.g., video clips with varying camera angles or scene changes. This challenge can render failures such as instance identity forgetting and key frame negligence. In this work, we first attribute the challenge to the lack of multi-shot annotations among existing datasets and therefore we introduce a new dataset termed MultiClip-Bench, featuring dense descriptions and instruction-based question-answering pairs tailored for multi-shot scenarios. We empirically find that the training set significantly boosts the multi-shot performance, while the testing benchmark provides a reliable measure of the model capability in multi-shot scenarios. By further analyzing and discovering that current models only encode instance features in a discrete or lossy manner, at the risk of missing identity information, we then contribute a new model IPFormer-VideoLLM. Its key idea is the injection of instance-level features as instance prompts through an efficient attention-based connector. This allows for the aggregation of instance-specific information across scenes. Experiments demonstrate that our proposed dataset and model not only enhance the multi-scene video understanding significantly, but also offer distinct advantages across various video benchmarks. 

---
# PhishKey: A Novel Centroid-Based Approach for Enhanced Phishing Detection Using Adaptive HTML Component Extraction 

**Authors**: Felipe Castaño, Eduardo Fidalgo, Enrique Alegre, Rocio Alaiz-Rodríguez, Raul Orduna, Francesco Zola  

**Link**: [PDF](https://arxiv.org/pdf/2506.21106)  

**Abstract**: Phishing attacks pose a significant cybersecurity threat, evolving rapidly to bypass detection mechanisms and exploit human vulnerabilities. This paper introduces PhishKey to address the challenges of adaptability, robustness, and efficiency. PhishKey is a novel phishing detection method using automatic feature extraction from hybrid sources. PhishKey combines character-level processing with Convolutional Neural Networks (CNN) for URL classification, and a Centroid-Based Key Component Phishing Extractor (CAPE) for HTML content at the word level. CAPE reduces noise and ensures complete sample processing avoiding crop operations on the input data. The predictions from both modules are integrated using a soft-voting ensemble to achieve more accurate and reliable classifications. Experimental evaluations on four state-of-the-art datasets demonstrate the effectiveness of PhishKey. It achieves up to 98.70% F1 Score and shows strong resistance to adversarial manipulations such as injection attacks with minimal performance degradation. 

---
# Interpretable Hierarchical Concept Reasoning through Attention-Guided Graph Learning 

**Authors**: David Debot, Pietro Barbiero, Gabriele Dominici, Giuseppe Marra  

**Link**: [PDF](https://arxiv.org/pdf/2506.21102)  

**Abstract**: Concept-Based Models (CBMs) are a class of deep learning models that provide interpretability by explaining predictions through high-level concepts. These models first predict concepts and then use them to perform a downstream task. However, current CBMs offer interpretability only for the final task prediction, while the concept predictions themselves are typically made via black-box neural networks. To address this limitation, we propose Hierarchical Concept Memory Reasoner (H-CMR), a new CBM that provides interpretability for both concept and task predictions. H-CMR models relationships between concepts using a learned directed acyclic graph, where edges represent logic rules that define concepts in terms of other concepts. During inference, H-CMR employs a neural attention mechanism to select a subset of these rules, which are then applied hierarchically to predict all concepts and the final task. Experimental results demonstrate that H-CMR matches state-of-the-art performance while enabling strong human interaction through concept and model interventions. The former can significantly improve accuracy at inference time, while the latter can enhance data efficiency during training when background knowledge is available. 

---
# ComRAG: Retrieval-Augmented Generation with Dynamic Vector Stores for Real-time Community Question Answering in Industry 

**Authors**: Qinwen Chen, Wenbiao Tao, Zhiwei Zhu, Mingfan Xi, Liangzhong Guo, Yuan Wang, Wei Wang, Yunshi Lan  

**Link**: [PDF](https://arxiv.org/pdf/2506.21098)  

**Abstract**: Community Question Answering (CQA) platforms can be deemed as important knowledge bases in community, but effectively leveraging historical interactions and domain knowledge in real-time remains a challenge. Existing methods often underutilize external knowledge, fail to incorporate dynamic historical QA context, or lack memory mechanisms suited for industrial deployment. We propose ComRAG, a retrieval-augmented generation framework for real-time industrial CQA that integrates static knowledge with dynamic historical QA pairs via a centroid-based memory mechanism designed for retrieval, generation, and efficient storage. Evaluated on three industrial CQA datasets, ComRAG consistently outperforms all baselines--achieving up to 25.9% improvement in vector similarity, reducing latency by 8.7% to 23.3%, and lowering chunk growth from 20.23% to 2.06% over iterations. 

---
# FeDa4Fair: Client-Level Federated Datasets for Fairness Evaluation 

**Authors**: Xenia Heilmann, Luca Corbucci, Mattia Cerrato, Anna Monreale  

**Link**: [PDF](https://arxiv.org/pdf/2506.21095)  

**Abstract**: Federated Learning (FL) enables collaborative model training across multiple clients without sharing clients' private data. However, fairness remains a key concern, as biases in local clients' datasets can impact the entire federated system. Heterogeneous data distributions across clients may lead to models that are fairer for some clients than others. Although several fairness-enhancing solutions are present in the literature, most focus on mitigating bias for a single sensitive attribute, typically binary, overlooking the diverse and sometimes conflicting fairness needs of different clients. This limited perspective can limit the effectiveness of fairness interventions for the different clients. To support more robust and reproducible fairness research in FL, we aim to enable a consistent benchmarking of fairness-aware FL methods at both the global and client levels. In this paper, we contribute in three ways: (1) We introduce FeDa4Fair, a library to generate tabular datasets tailored to evaluating fair FL methods under heterogeneous client bias; (2) we release four bias-heterogeneous datasets and corresponding benchmarks to compare fairness mitigation methods in a controlled environment; (3) we provide ready-to-use functions for evaluating fairness outcomes for these datasets. 

---
# CovDocker: Benchmarking Covalent Drug Design with Tasks, Datasets, and Solutions 

**Authors**: Yangzhe Peng, Kaiyuan Gao, Liang He, Yuheng Cong, Haiguang Liu, Kun He, Lijun Wu  

**Link**: [PDF](https://arxiv.org/pdf/2506.21085)  

**Abstract**: Molecular docking plays a crucial role in predicting the binding mode of ligands to target proteins, and covalent interactions, which involve the formation of a covalent bond between the ligand and the target, are particularly valuable due to their strong, enduring binding nature. However, most existing docking methods and deep learning approaches hardly account for the formation of covalent bonds and the associated structural changes. To address this gap, we introduce a comprehensive benchmark for covalent docking, CovDocker, which is designed to better capture the complexities of covalent binding. We decompose the covalent docking process into three main tasks: reactive location prediction, covalent reaction prediction, and covalent docking. By adapting state-of-the-art models, such as Uni-Mol and Chemformer, we establish baseline performances and demonstrate the effectiveness of the benchmark in accurately predicting interaction sites and modeling the molecular transformations involved in covalent binding. These results confirm the role of the benchmark as a rigorous framework for advancing research in covalent drug design. It underscores the potential of data-driven approaches to accelerate the discovery of selective covalent inhibitors and addresses critical challenges in therapeutic development. 

---
# EgoAdapt: Adaptive Multisensory Distillation and Policy Learning for Efficient Egocentric Perception 

**Authors**: Sanjoy Chowdhury, Subrata Biswas, Sayan Nag, Tushar Nagarajan, Calvin Murdock, Ishwarya Ananthabhotla, Yijun Qian, Vamsi Krishna Ithapu, Dinesh Manocha, Ruohan Gao  

**Link**: [PDF](https://arxiv.org/pdf/2506.21080)  

**Abstract**: Modern perception models, particularly those designed for multisensory egocentric tasks, have achieved remarkable performance but often come with substantial computational costs. These high demands pose challenges for real-world deployment, especially in resource-constrained environments. In this paper, we introduce EgoAdapt, a framework that adaptively performs cross-modal distillation and policy learning to enable efficient inference across different egocentric perception tasks, including egocentric action recognition, active speaker localization, and behavior anticipation. Our proposed policy module is adaptable to task-specific action spaces, making it broadly applicable. Experimental results on three challenging egocentric datasets EPIC-Kitchens, EasyCom, and Aria Everyday Activities demonstrate that our method significantly enhances efficiency, reducing GMACs by up to 89.09%, parameters up to 82.02%, and energy up to 9.6x, while still on-par and in many cases outperforming, the performance of corresponding state-of-the-art models. 

---
# A Semi-supervised Scalable Unified Framework for E-commerce Query Classification 

**Authors**: Chunyuan Yuan, Chong Zhang, Zheng Fang, Ming Pang, Xue Jiang, Changping Peng, Zhangang Lin, Ching Law  

**Link**: [PDF](https://arxiv.org/pdf/2506.21049)  

**Abstract**: Query classification, including multiple subtasks such as intent and category prediction, is vital to e-commerce applications. E-commerce queries are usually short and lack context, and the information between labels cannot be used, resulting in insufficient prior information for modeling. Most existing industrial query classification methods rely on users' posterior click behavior to construct training samples, resulting in a Matthew vicious cycle. Furthermore, the subtasks of query classification lack a unified framework, leading to low efficiency for algorithm optimization.
In this paper, we propose a novel Semi-supervised Scalable Unified Framework (SSUF), containing multiple enhanced modules to unify the query classification tasks. The knowledge-enhanced module uses world knowledge to enhance query representations and solve the problem of insufficient query information. The label-enhanced module uses label semantics and semi-supervised signals to reduce the dependence on posterior labels. The structure-enhanced module enhances the label representation based on the complex label relations. Each module is highly pluggable, and input features can be added or removed as needed according to each subtask. We conduct extensive offline and online A/B experiments, and the results show that SSUF significantly outperforms the state-of-the-art models. 

---
# Improving Diffusion-Based Image Editing Faithfulness via Guidance and Scheduling 

**Authors**: Hansam Cho, Seoung Bum Kim  

**Link**: [PDF](https://arxiv.org/pdf/2506.21045)  

**Abstract**: Text-guided diffusion models have become essential for high-quality image synthesis, enabling dynamic image editing. In image editing, two crucial aspects are editability, which determines the extent of modification, and faithfulness, which reflects how well unaltered elements are preserved. However, achieving optimal results is challenging because of the inherent trade-off between editability and faithfulness. To address this, we propose Faithfulness Guidance and Scheduling (FGS), which enhances faithfulness with minimal impact on editability. FGS incorporates faithfulness guidance to strengthen the preservation of input image information and introduces a scheduling strategy to resolve misalignment between editability and faithfulness. Experimental results demonstrate that FGS achieves superior faithfulness while maintaining editability. Moreover, its compatibility with various editing methods enables precise, high-quality image edits across diverse tasks. 

---
# Efficient Skill Discovery via Regret-Aware Optimization 

**Authors**: He Zhang, Ming Zhou, Shaopeng Zhai, Ying Sun, Hui Xiong  

**Link**: [PDF](https://arxiv.org/pdf/2506.21044)  

**Abstract**: Unsupervised skill discovery aims to learn diverse and distinguishable behaviors in open-ended reinforcement learning. For existing methods, they focus on improving diversity through pure exploration, mutual information optimization, and learning temporal representation. Despite that they perform well on exploration, they remain limited in terms of efficiency, especially for the high-dimensional situations. In this work, we frame skill discovery as a min-max game of skill generation and policy learning, proposing a regret-aware method on top of temporal representation learning that expands the discovered skill space along the direction of upgradable policy strength. The key insight behind the proposed method is that the skill discovery is adversarial to the policy learning, i.e., skills with weak strength should be further explored while less exploration for the skills with converged strength. As an implementation, we score the degree of strength convergence with regret, and guide the skill discovery with a learnable skill generator. To avoid degeneration, skill generation comes from an up-gradable population of skill generators. We conduct experiments on environments with varying complexities and dimension sizes. Empirical results show that our method outperforms baselines in both efficiency and diversity. Moreover, our method achieves a 15% zero shot improvement in high-dimensional environments, compared to existing methods. 

---
# V2X-REALM: Vision-Language Model-Based Robust End-to-End Cooperative Autonomous Driving with Adaptive Long-Tail Modeling 

**Authors**: Junwei You, Pei Li, Zhuoyu Jiang, Zilin Huang, Rui Gan, Haotian Shi, Bin Ran  

**Link**: [PDF](https://arxiv.org/pdf/2506.21041)  

**Abstract**: Ensuring robust planning and decision-making under rare, diverse, and visually degraded long-tail scenarios remains a fundamental challenge for autonomous driving in urban environments. This issue becomes more critical in cooperative settings, where vehicles and infrastructure jointly perceive and reason across complex environments. To address this challenge, we propose V2X-REALM, a vision-language model (VLM)-based framework with adaptive multimodal learning for robust cooperative autonomous driving under long-tail scenarios. V2X-REALM introduces three core innovations: (i) a prompt-driven long-tail scenario generation and evaluation pipeline that leverages foundation models to synthesize realistic long-tail conditions such as snow and fog across vehicle- and infrastructure-side views, enriching training diversity efficiently; (ii) a gated multi-scenario adaptive attention module that modulates the visual stream using scenario priors to recalibrate ambiguous or corrupted features; and (iii) a multi-task scenario-aware contrastive learning objective that improves multimodal alignment and promotes cross-scenario feature separability. Extensive experiments demonstrate that V2X-REALM significantly outperforms existing baselines in robustness, semantic reasoning, safety, and planning accuracy under complex, challenging driving conditions, advancing the scalability of end-to-end cooperative autonomous driving. 

---
# Strict Subgoal Execution: Reliable Long-Horizon Planning in Hierarchical Reinforcement Learning 

**Authors**: Jaebak Hwang, Sanghyeon Lee, Jeongmo Kim, Seungyul Han  

**Link**: [PDF](https://arxiv.org/pdf/2506.21039)  

**Abstract**: Long-horizon goal-conditioned tasks pose fundamental challenges for reinforcement learning (RL), particularly when goals are distant and rewards are sparse. While hierarchical and graph-based methods offer partial solutions, they often suffer from subgoal infeasibility and inefficient planning. We introduce Strict Subgoal Execution (SSE), a graph-based hierarchical RL framework that enforces single-step subgoal reachability by structurally constraining high-level decision-making. To enhance exploration, SSE employs a decoupled exploration policy that systematically traverses underexplored regions of the goal space. Furthermore, a failure-aware path refinement, which refines graph-based planning by dynamically adjusting edge costs according to observed low-level success rates, thereby improving subgoal reliability. Experimental results across diverse long-horizon benchmarks demonstrate that SSE consistently outperforms existing goal-conditioned RL and hierarchical RL approaches in both efficiency and success rate. 

---
# Large Language Models Acing Chartered Accountancy 

**Authors**: Jatin Gupta, Akhil Sharma, Saransh Singhania, Mohammad Adnan, Sakshi Deo, Ali Imam Abidi, Keshav Gupta  

**Link**: [PDF](https://arxiv.org/pdf/2506.21031)  

**Abstract**: Advanced intelligent systems, particularly Large Language Models (LLMs), are significantly reshaping financial practices through advancements in Natural Language Processing (NLP). However, the extent to which these models effectively capture and apply domain-specific financial knowledge remains uncertain. Addressing a critical gap in the expansive Indian financial context, this paper introduces CA-Ben, a Chartered Accountancy benchmark specifically designed to evaluate the financial, legal, and quantitative reasoning capabilities of LLMs. CA-Ben comprises structured question-answer datasets derived from the rigorous examinations conducted by the Institute of Chartered Accountants of India (ICAI), spanning foundational, intermediate, and advanced CA curriculum stages. Six prominent LLMs i.e. GPT 4o, LLAMA 3.3 70B, LLAMA 3.1 405B, MISTRAL Large, Claude 3.5 Sonnet, and Microsoft Phi 4 were evaluated using standardized protocols. Results indicate variations in performance, with Claude 3.5 Sonnet and GPT-4o outperforming others, especially in conceptual and legal reasoning. Notable challenges emerged in numerical computations and legal interpretations. The findings emphasize the strengths and limitations of current LLMs, suggesting future improvements through hybrid reasoning and retrieval-augmented generation methods, particularly for quantitative analysis and accurate legal interpretation. 

---
# Multimodal Prompt Alignment for Facial Expression Recognition 

**Authors**: Fuyan Ma, Yiran He, Bin Sun, Shutao Li  

**Link**: [PDF](https://arxiv.org/pdf/2506.21017)  

**Abstract**: Prompt learning has been widely adopted to efficiently adapt vision-language models (VLMs) like CLIP for various downstream tasks. Despite their success, current VLM-based facial expression recognition (FER) methods struggle to capture fine-grained textual-visual relationships, which are essential for distinguishing subtle differences between facial expressions. To address this challenge, we propose a multimodal prompt alignment framework for FER, called MPA-FER, that provides fine-grained semantic guidance to the learning process of prompted visual features, resulting in more precise and interpretable representations. Specifically, we introduce a multi-granularity hard prompt generation strategy that utilizes a large language model (LLM) like ChatGPT to generate detailed descriptions for each facial expression. The LLM-based external knowledge is injected into the soft prompts by minimizing the feature discrepancy between the soft prompts and the hard prompts. To preserve the generalization abilities of the pretrained CLIP model, our approach incorporates prototype-guided visual feature alignment, ensuring that the prompted visual features from the frozen image encoder align closely with class-specific prototypes. Additionally, we propose a cross-modal global-local alignment module that focuses on expression-relevant facial features, further improving the alignment between textual and visual features. Extensive experiments demonstrate our framework outperforms state-of-the-art methods on three FER benchmark datasets, while retaining the benefits of the pretrained model and minimizing computational costs. 

---
# SAC: A Framework for Measuring and Inducing Personality Traits in LLMs with Dynamic Intensity Control 

**Authors**: Adithya Chittem, Aishna Shrivastava, Sai Tarun Pendela, Jagat Sesh Challa, Dhruv Kumar  

**Link**: [PDF](https://arxiv.org/pdf/2506.20993)  

**Abstract**: Large language models (LLMs) have gained significant traction across a wide range of fields in recent years. There is also a growing expectation for them to display human-like personalities during interactions. To meet this expectation, numerous studies have proposed methods for modelling LLM personalities through psychometric evaluations. However, most existing models face two major limitations: they rely on the Big Five (OCEAN) framework, which only provides coarse personality dimensions, and they lack mechanisms for controlling trait intensity. In this paper, we address this gap by extending the Machine Personality Inventory (MPI), which originally used the Big Five model, to incorporate the 16 Personality Factor (16PF) model, allowing expressive control over sixteen distinct traits. We also developed a structured framework known as Specific Attribute Control (SAC) for evaluating and dynamically inducing trait intensity in LLMs. Our method introduces adjective-based semantic anchoring to guide trait intensity expression and leverages behavioural questions across five intensity factors: \textit{Frequency}, \textit{Depth}, \textit{Threshold}, \textit{Effort}, and \textit{Willingness}. Through experimentation, we find that modelling intensity as a continuous spectrum yields substantially more consistent and controllable personality expression compared to binary trait toggling. Moreover, we observe that changes in target trait intensity systematically influence closely related traits in psychologically coherent directions, suggesting that LLMs internalize multi-dimensional personality structures rather than treating traits in isolation. Our work opens new pathways for controlled and nuanced human-machine interactions in domains such as healthcare, education, and interviewing processes, bringing us one step closer to truly human-like social machines. 

---
# Segment Anything in Pathology Images with Natural Language 

**Authors**: Zhixuan Chen, Junlin Hou, Liqi Lin, Yihui Wang, Yequan Bie, Xi Wang, Yanning Zhou, Ronald Cheong Kin Chan, Hao Chen  

**Link**: [PDF](https://arxiv.org/pdf/2506.20988)  

**Abstract**: Pathology image segmentation is crucial in computational pathology for analyzing histological features relevant to cancer diagnosis and prognosis. However, current methods face major challenges in clinical applications due to limited annotated data and restricted category definitions. To address these limitations, we propose PathSegmentor, the first text-prompted segmentation foundation model designed specifically for pathology images. We also introduce PathSeg , the largest and most comprehensive dataset for pathology segmentation, built from 17 public sources and containing 275k image-mask-label triples across 160 diverse categories. With PathSegmentor, users can perform semantic segmentation using natural language prompts, eliminating the need for laborious spatial inputs such as points or boxes. Extensive experiments demonstrate that PathSegmentor outperforms specialized models with higher accuracy and broader applicability, while maintaining a compact architecture. It significantly surpasses existing spatial- and text-prompted models by 0.145 and 0.429 in overall Dice scores, respectively, showing strong robustness in segmenting complex structures and generalizing to external datasets. Moreover, PathSegmentor's outputs enhance the interpretability of diagnostic models through feature importance estimation and imaging biomarker discovery, offering pathologists evidence-based support for clinical decision-making. This work advances the development of explainable AI in precision oncology. 

---
# Enhancing Homophily-Heterophily Separation: Relation-Aware Learning in Heterogeneous Graphs 

**Authors**: Ziyu Zheng, Yaming Yang, Ziyu Guan, Wei Zhao, Weigang Lu  

**Link**: [PDF](https://arxiv.org/pdf/2506.20980)  

**Abstract**: Real-world networks usually have a property of node heterophily, that is, the connected nodes usually have different features or different labels. This heterophily issue has been extensively studied in homogeneous graphs but remains under-explored in heterogeneous graphs, where there are multiple types of nodes and edges. Capturing node heterophily in heterogeneous graphs is very challenging since both node/edge heterogeneity and node heterophily should be carefully taken into consideration. Existing methods typically convert heterogeneous graphs into homogeneous ones to learn node heterophily, which will inevitably lose the potential heterophily conveyed by heterogeneous relations. To bridge this gap, we propose Relation-Aware Separation of Homophily and Heterophily (RASH), a novel contrastive learning framework that explicitly models high-order semantics of heterogeneous interactions and adaptively separates homophilic and heterophilic patterns. Particularly, RASH introduces dual heterogeneous hypergraphs to encode multi-relational bipartite subgraphs and dynamically constructs homophilic graphs and heterophilic graphs based on relation importance. A multi-relation contrastive loss is designed to align heterogeneous and homophilic/heterophilic views by maximizing mutual information. In this way, RASH simultaneously resolves the challenges of heterogeneity and heterophily in heterogeneous graphs. Extensive experiments on benchmark datasets demonstrate the effectiveness of RASH across various downstream tasks. The code is available at: this https URL. 

---
# From Cradle to Cane: A Two-Pass Framework for High-Fidelity Lifespan Face Aging 

**Authors**: Tao Liu, Dafeng Zhang, Gengchen Li, Shizhuo Liu, Yongqi Song, Senmao Li, Shiqi Yang, Boqian Li, Kai Wang, Yaxing Wang  

**Link**: [PDF](https://arxiv.org/pdf/2506.20977)  

**Abstract**: Face aging has become a crucial task in computer vision, with applications ranging from entertainment to healthcare. However, existing methods struggle with achieving a realistic and seamless transformation across the entire lifespan, especially when handling large age gaps or extreme head poses. The core challenge lies in balancing age accuracy and identity preservation--what we refer to as the Age-ID trade-off. Most prior methods either prioritize age transformation at the expense of identity consistency or vice versa. In this work, we address this issue by proposing a two-pass face aging framework, named Cradle2Cane, based on few-step text-to-image (T2I) diffusion models. The first pass focuses on solving age accuracy by introducing an adaptive noise injection (AdaNI) mechanism. This mechanism is guided by including prompt descriptions of age and gender for the given person as the textual condition. Also, by adjusting the noise level, we can control the strength of aging while allowing more flexibility in transforming the face. However, identity preservation is weakly ensured here to facilitate stronger age transformations. In the second pass, we enhance identity preservation while maintaining age-specific features by conditioning the model on two identity-aware embeddings (IDEmb): SVR-ArcFace and Rotate-CLIP. This pass allows for denoising the transformed image from the first pass, ensuring stronger identity preservation without compromising the aging accuracy. Both passes are jointly trained in an end-to-end way. Extensive experiments on the CelebA-HQ test dataset, evaluated through Face++ and Qwen-VL protocols, show that our Cradle2Cane outperforms existing face aging methods in age accuracy and identity consistency. 

---
# DFVEdit: Conditional Delta Flow Vector for Zero-shot Video Editing 

**Authors**: Lingling Cai, Kang Zhao, Hangjie Yuan, Xiang Wang, Yingya Zhang, Kejie Huang  

**Link**: [PDF](https://arxiv.org/pdf/2506.20967)  

**Abstract**: The advent of Video Diffusion Transformers (Video DiTs) marks a milestone in video generation. However, directly applying existing video editing methods to Video DiTs often incurs substantial computational overhead, due to resource-intensive attention modification or finetuning. To alleviate this problem, we present DFVEdit, an efficient zero-shot video editing method tailored for Video DiTs. DFVEdit eliminates the need for both attention modification and fine-tuning by directly operating on clean latents via flow transformation. To be more specific, we observe that editing and sampling can be unified under the continuous flow perspective. Building upon this foundation, we propose the Conditional Delta Flow Vector (CDFV) -- a theoretically unbiased estimation of DFV -- and integrate Implicit Cross Attention (ICA) guidance as well as Embedding Reinforcement (ER) to further enhance editing quality. DFVEdit excels in practical efficiency, offering at least 20x inference speed-up and 85\% memory reduction on Video DiTs compared to attention-engineering-based editing methods. Extensive quantitative and qualitative experiments demonstrate that DFVEdit can be seamlessly applied to popular Video DiTs (e.g., CogVideoX and Wan2.1), attaining state-of-the-art performance on structural fidelity, spatial-temporal consistency, and editing quality. 

---
# Parallels Between VLA Model Post-Training and Human Motor Learning: Progress, Challenges, and Trends 

**Authors**: Tian-Yu Xiang, Ao-Qun Jin, Xiao-Hu Zhou, Mei-Jiang Gui, Xiao-Liang Xie, Shi-Qi Liu, Shuang-Yi Wang, Sheng-Bin Duan, Fu-Chao Xie, Wen-Kai Wang, Si-Cheng Wang, Ling-Yun Li, Tian Tu, Zeng-Guang Hou  

**Link**: [PDF](https://arxiv.org/pdf/2506.20966)  

**Abstract**: Vision-language-action (VLA) models extend vision-language models (VLM) by integrating action generation modules for robotic manipulation. Leveraging strengths of VLM in vision perception and instruction understanding, VLA models exhibit promising generalization across diverse manipulation tasks. However, applications demanding high precision and accuracy reveal performance gaps without further adaptation. Evidence from multiple domains highlights the critical role of post-training to align foundational models with downstream applications, spurring extensive research on post-training VLA models. VLA model post-training aims to address the challenge of improving an embodiment's ability to interact with the environment for the given tasks, analogous to the process of humans motor skills acquisition. Accordingly, this paper reviews post-training strategies for VLA models through the lens of human motor learning, focusing on three dimensions: environments, embodiments, and tasks. A structured taxonomy is introduced aligned with human learning mechanisms: (1) enhancing environmental perception, (2) improving embodiment awareness, (3) deepening task comprehension, and (4) multi-component integration. Finally, key challenges and trends in post-training VLA models are identified, establishing a conceptual framework to guide future research. This work delivers both a comprehensive overview of current VLA model post-training methods from a human motor learning perspective and practical insights for VLA model development. (Project website: this https URL) 

---
# Evidence-based diagnostic reasoning with multi-agent copilot for human pathology 

**Authors**: Chengkuan Chen, Luca L. Weishaupt, Drew F. K. Williamson, Richard J. Chen, Tong Ding, Bowen Chen, Anurag Vaidya, Long Phi Le, Guillaume Jaume, Ming Y. Lu, Faisal Mahmood  

**Link**: [PDF](https://arxiv.org/pdf/2506.20964)  

**Abstract**: Pathology is experiencing rapid digital transformation driven by whole-slide imaging and artificial intelligence (AI). While deep learning-based computational pathology has achieved notable success, traditional models primarily focus on image analysis without integrating natural language instruction or rich, text-based context. Current multimodal large language models (MLLMs) in computational pathology face limitations, including insufficient training data, inadequate support and evaluation for multi-image understanding, and a lack of autonomous, diagnostic reasoning capabilities. To address these limitations, we introduce PathChat+, a new MLLM specifically designed for human pathology, trained on over 1 million diverse, pathology-specific instruction samples and nearly 5.5 million question answer turns. Extensive evaluations across diverse pathology benchmarks demonstrated that PathChat+ substantially outperforms the prior PathChat copilot, as well as both state-of-the-art (SOTA) general-purpose and other pathology-specific models. Furthermore, we present SlideSeek, a reasoning-enabled multi-agent AI system leveraging PathChat+ to autonomously evaluate gigapixel whole-slide images (WSIs) through iterative, hierarchical diagnostic reasoning, reaching high accuracy on DDxBench, a challenging open-ended differential diagnosis benchmark, while also capable of generating visually grounded, humanly-interpretable summary reports. 

---
# OmniEval: A Benchmark for Evaluating Omni-modal Models with Visual, Auditory, and Textual Inputs 

**Authors**: Yiman Zhang, Ziheng Luo, Qiangyu Yan, Wei He, Borui Jiang, Xinghao Chen, Kai Han  

**Link**: [PDF](https://arxiv.org/pdf/2506.20960)  

**Abstract**: In this paper, we introduce OmniEval, a benchmark for evaluating omni-modality models like MiniCPM-O 2.6, which encompasses visual, auditory, and textual inputs. Compared with existing benchmarks, our OmniEval has several distinctive features: (i) Full-modal collaboration: We design evaluation tasks that highlight the strong coupling between audio and video, requiring models to effectively leverage the collaborative perception of all modalities; (ii) Diversity of videos: OmniEval includes 810 audio-visual synchronized videos, 285 Chinese videos and 525 English videos; (iii) Diversity and granularity of tasks: OmniEval contains 2617 question-answer pairs, comprising 1412 open-ended questions and 1205 multiple-choice questions. These questions are divided into 3 major task types and 12 sub-task types to achieve comprehensive evaluation. Among them, we introduce a more granular video localization task named Grounding. Then we conduct experiments on OmniEval with several omni-modality models. We hope that our OmniEval can provide a platform for evaluating the ability to construct and understand coherence from the context of all modalities. Codes and data could be found at this https URL. 

---
# Antibody Design and Optimization with Multi-scale Equivariant Graph Diffusion Models for Accurate Complex Antigen Binding 

**Authors**: Jiameng Chen, Xiantao Cai, Jia Wu, Wenbin Hu  

**Link**: [PDF](https://arxiv.org/pdf/2506.20957)  

**Abstract**: Antibody design remains a critical challenge in therapeutic and diagnostic development, particularly for complex antigens with diverse binding interfaces. Current computational methods face two main limitations: (1) capturing geometric features while preserving symmetries, and (2) generalizing novel antigen interfaces. Despite recent advancements, these methods often fail to accurately capture molecular interactions and maintain structural integrity. To address these challenges, we propose \textbf{AbMEGD}, an end-to-end framework integrating \textbf{M}ulti-scale \textbf{E}quivariant \textbf{G}raph \textbf{D}iffusion for antibody sequence and structure co-design. Leveraging advanced geometric deep learning, AbMEGD combines atomic-level geometric features with residue-level embeddings, capturing local atomic details and global sequence-structure interactions. Its E(3)-equivariant diffusion method ensures geometric precision, computational efficiency, and robust generalizability for complex antigens. Furthermore, experiments using the SAbDab database demonstrate a 10.13\% increase in amino acid recovery, 3.32\% rise in improvement percentage, and a 0.062~Å reduction in root mean square deviation within the critical CDR-H3 region compared to DiffAb, a leading antibody design model. These results highlight AbMEGD's ability to balance structural integrity with improved functionality, establishing a new benchmark for sequence-structure co-design and affinity optimization. The code is available at: this https URL. 

---
# Consistent Zero-shot 3D Texture Synthesis Using Geometry-aware Diffusion and Temporal Video Models 

**Authors**: Donggoo Kang, Jangyeong Kim, Dasol Jeong, Junyoung Choi, Jeonga Wi, Hyunmin Lee, Joonho Gwon, Joonki Paik  

**Link**: [PDF](https://arxiv.org/pdf/2506.20946)  

**Abstract**: Current texture synthesis methods, which generate textures from fixed viewpoints, suffer from inconsistencies due to the lack of global context and geometric understanding. Meanwhile, recent advancements in video generation models have demonstrated remarkable success in achieving temporally consistent videos. In this paper, we introduce VideoTex, a novel framework for seamless texture synthesis that leverages video generation models to address both spatial and temporal inconsistencies in 3D textures. Our approach incorporates geometry-aware conditions, enabling precise utilization of 3D mesh structures. Additionally, we propose a structure-wise UV diffusion strategy, which enhances the generation of occluded areas by preserving semantic information, resulting in smoother and more coherent textures. VideoTex not only achieves smoother transitions across UV boundaries but also ensures high-quality, temporally stable textures across video frames. Extensive experiments demonstrate that VideoTex outperforms existing methods in texture fidelity, seam blending, and stability, paving the way for dynamic real-time applications that demand both visual quality and temporal coherence. 

---
# Interpretable Representation Learning for Additive Rule Ensembles 

**Authors**: Shahrzad Behzadimanesh, Pierre Le Bodic, Geoffrey I. Webb, Mario Boley  

**Link**: [PDF](https://arxiv.org/pdf/2506.20927)  

**Abstract**: Small additive ensembles of symbolic rules offer interpretable prediction models. Traditionally, these ensembles use rule conditions based on conjunctions of simple threshold propositions $x \geq t$ on a single input variable $x$ and threshold $t$, resulting geometrically in axis-parallel polytopes as decision regions. While this form ensures a high degree of interpretability for individual rules and can be learned efficiently using the gradient boosting approach, it relies on having access to a curated set of expressive and ideally independent input features so that a small ensemble of axis-parallel regions can describe the target variable well. Absent such features, reaching sufficient accuracy requires increasing the number and complexity of individual rules, which diminishes the interpretability of the model. Here, we extend classical rule ensembles by introducing logical propositions with learnable sparse linear transformations of input variables, i.e., propositions of the form $\mathbf{x}^\mathrm{T}\mathbf{w} \geq t$, where $\mathbf{w}$ is a learnable sparse weight vector, enabling decision regions as general polytopes with oblique faces. We propose a learning method using sequential greedy optimization based on an iteratively reweighted formulation of logistic regression. Experimental results demonstrate that the proposed method efficiently constructs rule ensembles with the same test risk as state-of-the-art methods while significantly reducing model complexity across ten benchmark datasets. 

---
# LLM-guided Chemical Process Optimization with a Multi-Agent Approach 

**Authors**: Tong Zeng, Srivathsan Badrinarayanan, Janghoon Ock, Cheng-Kai Lai, Amir Barati Farimani  

**Link**: [PDF](https://arxiv.org/pdf/2506.20921)  

**Abstract**: Chemical process optimization is crucial to maximize production efficiency and economic performance. Traditional methods, including gradient-based solvers, evolutionary algorithms, and parameter grid searches, become impractical when operating constraints are ill-defined or unavailable, requiring engineers to rely on subjective heuristics to estimate feasible parameter ranges. To address this constraint definition bottleneck, we present a multi-agent framework of large language model (LLM) agents that autonomously infer operating constraints from minimal process descriptions, then collaboratively guide optimization using the inferred constraints. Our AutoGen-based agentic framework employs OpenAI's o3 model, with specialized agents for constraint generation, parameter validation, simulation execution, and optimization guidance. Through two phases - autonomous constraint generation using embedded domain knowledge, followed by iterative multi-agent optimization - the framework eliminates the need for predefined operational bounds. Validated on the hydrodealkylation process across cost, yield, and yield-to-cost ratio metrics, the framework demonstrated competitive performance with conventional optimization methods while achieving better computational efficiency, requiring fewer iterations to converge. Our approach converged in under 20 minutes, achieving a 31-fold speedup over grid search. Beyond computational efficiency, the framework's reasoning-guided search demonstrates sophisticated process understanding, correctly identifying utility trade-offs, and applying domain-informed heuristics. This approach shows significant potential for optimization scenarios where operational constraints are poorly characterized or unavailable, particularly for emerging processes and retrofit applications. 

---
# Optimising Language Models for Downstream Tasks: A Post-Training Perspective 

**Authors**: Zhengyan Shi  

**Link**: [PDF](https://arxiv.org/pdf/2506.20917)  

**Abstract**: Language models (LMs) have demonstrated remarkable capabilities in NLP, yet adapting them efficiently and robustly to specific tasks remains challenging. As their scale and complexity grow, fine-tuning LMs on labelled data often underutilizes available unlabelled data, leads to overfitting on small task-specific sets, and imposes significant computational costs. These limitations hamper their application to the open-ended landscape of real-world language tasks.
This thesis proposes a series of methods to better adapt LMs to downstream applications. First, we explore strategies for extracting task-relevant knowledge from unlabelled data, introducing a novel continued pre-training technique that outperforms state-of-the-art semi-supervised approaches. Next, we present a parameter-efficient fine-tuning method that substantially reduces memory and compute costs while maintaining competitive performance. We also introduce improved supervised fine-tuning methods that enable LMs to better follow instructions, especially when labelled data is scarce, enhancing their performance across a range of NLP tasks, including open-ended generation. Finally, we develop new evaluation methods and benchmarks, such as multi-hop spatial reasoning tasks, to assess LM capabilities and adaptation more comprehensively.
Through extensive empirical studies across diverse NLP tasks, our results demonstrate that these approaches substantially improve LM robustness, efficiency, and generalization, making them more adaptable to a broad range of applications. These advances mark a significant step towards more robust and efficient LMs, bringing us closer to the goal of artificial general intelligence. 

---
# ZKPROV: A Zero-Knowledge Approach to Dataset Provenance for Large Language Models 

**Authors**: Mina Namazi, Alexander Nemecek, Erman Ayday  

**Link**: [PDF](https://arxiv.org/pdf/2506.20915)  

**Abstract**: As the deployment of large language models (LLMs) grows in sensitive domains, ensuring the integrity of their computational provenance becomes a critical challenge, particularly in regulated sectors such as healthcare, where strict requirements are applied in dataset usage. We introduce ZKPROV, a novel cryptographic framework that enables zero-knowledge proofs of LLM provenance. It allows users to verify that a model is trained on a reliable dataset without revealing sensitive information about it or its parameters. Unlike prior approaches that focus on complete verification of the training process (incurring significant computational cost) or depend on trusted execution environments, ZKPROV offers a distinct balance. Our method cryptographically binds a trained model to its authorized training dataset(s) through zero-knowledge proofs while avoiding proof of every training step. By leveraging dataset-signed metadata and compact model parameter commitments, ZKPROV provides sound and privacy-preserving assurances that the result of the LLM is derived from a model trained on the claimed authorized and relevant dataset. Experimental results demonstrate the efficiency and scalability of the ZKPROV in generating this proof and verifying it, achieving a practical solution for real-world deployments. We also provide formal security guarantees, proving that our approach preserves dataset confidentiality while ensuring trustworthy dataset provenance. 

---
# Omniwise: Predicting GPU Kernels Performance with LLMs 

**Authors**: Zixian Wang, Cole Ramos, Muhammad A. Awad, Keith Lowery  

**Link**: [PDF](https://arxiv.org/pdf/2506.20886)  

**Abstract**: In recent years, the rapid advancement of deep neural networks (DNNs) has revolutionized artificial intelligence, enabling models with unprecedented capabilities in understanding, generating, and processing complex data. These powerful architectures have transformed a wide range of downstream applications, tackling tasks beyond human reach. In this paper, we introduce Omniwise, the first end-to-end, self-supervised fine-tuning pipeline that applies large language models (LLMs) to GPU kernel performance prediction--a novel use case in performance profiling. Omniwise is model-agnostic and lightweight, achieving strong results even with a small 3B-parameter model. It can predict key performance metrics, including memory bandwidth, cache hit rates, GFLOPs, and arithmetic intensity, directly from kernel code without the need for code execution or profiling tools. Our approach achieves over 90% of predictions within 10% relative error on GPU kernels executed on AMD MI250 and MI300X architectures. In addition to the pipeline, we develop an online inference server and a Visual Studio Code plugin that seamlessly integrate LLM-based performance prediction into developers' workflows. 

---
# Complex Model Transformations by Reinforcement Learning with Uncertain Human Guidance 

**Authors**: Kyanna Dagenais, Istvan David  

**Link**: [PDF](https://arxiv.org/pdf/2506.20883)  

**Abstract**: Model-driven engineering problems often require complex model transformations (MTs), i.e., MTs that are chained in extensive sequences. Pertinent examples of such problems include model synchronization, automated model repair, and design space exploration. Manually developing complex MTs is an error-prone and often infeasible process. Reinforcement learning (RL) is an apt way to alleviate these issues. In RL, an autonomous agent explores the state space through trial and error to identify beneficial sequences of actions, such as MTs. However, RL methods exhibit performance issues in complex problems. In these situations, human guidance can be of high utility. In this paper, we present an approach and technical framework for developing complex MT sequences through RL, guided by potentially uncertain human advice. Our framework allows user-defined MTs to be mapped onto RL primitives, and executes them as RL programs to find optimal MT sequences. Our evaluation shows that human guidance, even if uncertain, substantially improves RL performance, and results in more efficient development of complex MTs. Through a trade-off between the certainty and timeliness of human advice, our method takes a step towards RL-driven human-in-the-loop engineering methods. 

---
# THIRDEYE: Cue-Aware Monocular Depth Estimation via Brain-Inspired Multi-Stage Fusion 

**Authors**: Calin Teodor Ioan  

**Link**: [PDF](https://arxiv.org/pdf/2506.20877)  

**Abstract**: Monocular depth estimation methods traditionally train deep models to infer depth directly from RGB pixels. This implicit learning often overlooks explicit monocular cues that the human visual system relies on, such as occlusion boundaries, shading, and perspective. Rather than expecting a network to discover these cues unaided, we present ThirdEye, a cue-aware pipeline that deliberately supplies each cue through specialised, pre-trained, and frozen networks. These cues are fused in a three-stage cortical hierarchy (V1->V2->V3) equipped with a key-value working-memory module that weights them by reliability. An adaptive-bins transformer head then produces a high-resolution disparity map. Because the cue experts are frozen, ThirdEye inherits large amounts of external supervision while requiring only modest fine-tuning. This extended version provides additional architectural detail, neuroscientific motivation, and an expanded experimental protocol; quantitative results will appear in a future revision. 

---
# Engineering RAG Systems for Real-World Applications: Design, Development, and Evaluation 

**Authors**: Md Toufique Hasan, Muhammad Waseem, Kai-Kristian Kemell, Ayman Asad Khan, Mika Saari, Pekka Abrahamsson  

**Link**: [PDF](https://arxiv.org/pdf/2506.20869)  

**Abstract**: Retrieval-Augmented Generation (RAG) systems are emerging as a key approach for grounding Large Language Models (LLMs) in external knowledge, addressing limitations in factual accuracy and contextual relevance. However, there is a lack of empirical studies that report on the development of RAG-based implementations grounded in real-world use cases, evaluated through general user involvement, and accompanied by systematic documentation of lessons learned. This paper presents five domain-specific RAG applications developed for real-world scenarios across governance, cybersecurity, agriculture, industrial research, and medical diagnostics. Each system incorporates multilingual OCR, semantic retrieval via vector embeddings, and domain-adapted LLMs, deployed through local servers or cloud APIs to meet distinct user needs. A web-based evaluation involving a total of 100 participants assessed the systems across six dimensions: (i) Ease of Use, (ii) Relevance, (iii) Transparency, (iv) Responsiveness, (v) Accuracy, and (vi) Likelihood of Recommendation. Based on user feedback and our development experience, we documented twelve key lessons learned, highlighting technical, operational, and ethical challenges affecting the reliability and usability of RAG systems in practice. 

---
# Generating Reliable Adverse event Profiles for Health through Automated Integrated Data (GRAPH-AID): A Semi-Automated Ontology Building Approach 

**Authors**: Srikar Reddy Gadusu, Larry Callahan, Samir Lababidi, Arunasri Nishtala, Sophia Healey, Hande McGinty  

**Link**: [PDF](https://arxiv.org/pdf/2506.20851)  

**Abstract**: As data and knowledge expand rapidly, adopting systematic methodologies for ontology generation has become crucial. With the daily increases in data volumes and frequent content changes, the demand for databases to store and retrieve information for the creation of knowledge graphs has become increasingly urgent. The previously established Knowledge Acquisition and Representation Methodology (KNARM) outlines a systematic approach to address these challenges and create knowledge graphs. However, following this methodology highlights the existing challenge of seamlessly integrating Neo4j databases with the Web Ontology Language (OWL). Previous attempts to integrate data from Neo4j into an ontology have been discussed, but these approaches often require an understanding of description logics (DL) syntax, which may not be familiar to many users. Thus, a more accessible method is necessary to bridge this gap. This paper presents a user-friendly approach that utilizes Python and its rdflib library to support ontology development. We showcase our novel approach through a Neo4j database we created by integrating data from the Food and Drug Administration (FDA) Adverse Event Reporting System (FAERS) database. Using this dataset, we developed a Python script that automatically generates the required classes and their axioms, facilitating a smoother integration process. This approach offers a practical solution to the challenges of ontology generation in the context of rapidly growing adverse drug event datasets, supporting improved drug safety monitoring and public health decision-making. 

---
# FixCLR: Negative-Class Contrastive Learning for Semi-Supervised Domain Generalization 

**Authors**: Ha Min Son, Shahbaz Rezaei, Xin Liu  

**Link**: [PDF](https://arxiv.org/pdf/2506.20841)  

**Abstract**: Semi-supervised domain generalization (SSDG) aims to solve the problem of generalizing to out-of-distribution data when only a few labels are available. Due to label scarcity, applying domain generalization methods often underperform. Consequently, existing SSDG methods combine semi-supervised learning methods with various regularization terms. However, these methods do not explicitly regularize to learn domains invariant representations across all domains, which is a key goal for domain generalization. To address this, we introduce FixCLR. Inspired by success in self-supervised learning, we change two crucial components to adapt contrastive learning for explicit domain invariance regularization: utilization of class information from pseudo-labels and using only a repelling term. FixCLR can also be added on top of most existing SSDG and semi-supervised methods for complementary performance improvements. Our research includes extensive experiments that have not been previously explored in SSDG studies. These experiments include benchmarking different improvements to semi-supervised methods, evaluating the performance of pretrained versus non-pretrained models, and testing on datasets with many domains. Overall, FixCLR proves to be an effective SSDG method, especially when combined with other semi-supervised methods. 

---
# Leveraging Vision-Language Models to Select Trustworthy Super-Resolution Samples Generated by Diffusion Models 

**Authors**: Cansu Korkmaz, Ahmet Murat Tekalp, Zafer Dogan  

**Link**: [PDF](https://arxiv.org/pdf/2506.20832)  

**Abstract**: Super-resolution (SR) is an ill-posed inverse problem with many feasible solutions consistent with a given low-resolution image. On one hand, regressive SR models aim to balance fidelity and perceptual quality to yield a single solution, but this trade-off often introduces artifacts that create ambiguity in information-critical applications such as recognizing digits or letters. On the other hand, diffusion models generate a diverse set of SR images, but selecting the most trustworthy solution from this set remains a challenge. This paper introduces a robust, automated framework for identifying the most trustworthy SR sample from a diffusion-generated set by leveraging the semantic reasoning capabilities of vision-language models (VLMs). Specifically, VLMs such as BLIP-2, GPT-4o, and their variants are prompted with structured queries to assess semantic correctness, visual quality, and artifact presence. The top-ranked SR candidates are then ensembled to yield a single trustworthy output in a cost-effective manner. To rigorously assess the validity of VLM-selected samples, we propose a novel Trustworthiness Score (TWS) a hybrid metric that quantifies SR reliability based on three complementary components: semantic similarity via CLIP embeddings, structural integrity using SSIM on edge maps, and artifact sensitivity through multi-level wavelet decomposition. We empirically show that TWS correlates strongly with human preference in both ambiguous and natural images, and that VLM-guided selections consistently yield high TWS values. Compared to conventional metrics like PSNR, LPIPS, which fail to reflect information fidelity, our approach offers a principled, scalable, and generalizable solution for navigating the uncertainty of the diffusion SR space. By aligning outputs with human expectations and semantic correctness, this work sets a new benchmark for trustworthiness in generative SR. 

---
# Uncovering Hidden Violent Tendencies in LLMs: A Demographic Analysis via Behavioral Vignettes 

**Authors**: Quintin Myers, Yanjun Gao  

**Link**: [PDF](https://arxiv.org/pdf/2506.20822)  

**Abstract**: Large language models (LLMs) are increasingly proposed for detecting and responding to violent content online, yet their ability to reason about morally ambiguous, real-world scenarios remains underexamined. We present the first study to evaluate LLMs using a validated social science instrument designed to measure human response to everyday conflict, namely the Violent Behavior Vignette Questionnaire (VBVQ). To assess potential bias, we introduce persona-based prompting that varies race, age, and geographic identity within the United States. Six LLMs developed across different geopolitical and organizational contexts are evaluated under a unified zero-shot setting. Our study reveals two key findings: (1) LLMs surface-level text generation often diverges from their internal preference for violent responses; (2) their violent tendencies vary across demographics, frequently contradicting established findings in criminology, social science, and psychology. 

---
# MultiFinRAG: An Optimized Multimodal Retrieval-Augmented Generation (RAG) Framework for Financial Question Answering 

**Authors**: Chinmay Gondhalekar, Urjitkumar Patel, Fang-Chun Yeh  

**Link**: [PDF](https://arxiv.org/pdf/2506.20821)  

**Abstract**: Financial documents--such as 10-Ks, 10-Qs, and investor presentations--span hundreds of pages and combine diverse modalities, including dense narrative text, structured tables, and complex figures. Answering questions over such content often requires joint reasoning across modalities, which strains traditional large language models (LLMs) and retrieval-augmented generation (RAG) pipelines due to token limitations, layout loss, and fragmented cross-modal context. We introduce MultiFinRAG, a retrieval-augmented generation framework purpose-built for financial QA. MultiFinRAG first performs multimodal extraction by grouping table and figure images into batches and sending them to a lightweight, quantized open-source multimodal LLM, which produces both structured JSON outputs and concise textual summaries. These outputs, along with narrative text, are embedded and indexed with modality-aware similarity thresholds for precise retrieval. A tiered fallback strategy then dynamically escalates from text-only to text+table+image contexts when necessary, enabling cross-modal reasoning while reducing irrelevant context. Despite running on commodity hardware, MultiFinRAG achieves 19 percentage points higher accuracy than ChatGPT-4o (free-tier) on complex financial QA tasks involving text, tables, images, and combined multimodal reasoning. 

---
# FINN-GL: Generalized Mixed-Precision Extensions for FPGA-Accelerated LSTMs 

**Authors**: Shashwat Khandelwal, Jakoba Petri-Koenig, Thomas B. Preußer, Michaela Blott, Shreejith Shanker  

**Link**: [PDF](https://arxiv.org/pdf/2506.20810)  

**Abstract**: Recurrent neural networks (RNNs), particularly LSTMs, are effective for time-series tasks like sentiment analysis and short-term stock prediction. However, their computational complexity poses challenges for real-time deployment in resource constrained environments. While FPGAs offer a promising platform for energy-efficient AI acceleration, existing tools mainly target feed-forward networks, and LSTM acceleration typically requires full custom implementation. In this paper, we address this gap by leveraging the open-source and extensible FINN framework to enable the generalized deployment of LSTMs on FPGAs. Specifically, we leverage the Scan operator from the Open Neural Network Exchange (ONNX) specification to model the recurrent nature of LSTM computations, enabling support for mixed quantisation within them and functional verification of LSTM-based models. Furthermore, we introduce custom transformations within the FINN compiler to map the quantised ONNX computation graph to hardware blocks from the HLS kernel library of the FINN compiler and Vitis HLS. We validate the proposed tool-flow by training a quantised ConvLSTM model for a mid-price stock prediction task using the widely used dataset and generating a corresponding hardware IP of the model using our flow, targeting the XCZU7EV device. We show that the generated quantised ConvLSTM accelerator through our flow achieves a balance between performance (latency) and resource consumption, while matching (or bettering) inference accuracy of state-of-the-art models with reduced precision. We believe that the generalisable nature of the proposed flow will pave the way for resource-efficient RNN accelerator designs on FPGAs. 

---
# GPU Kernel Scientist: An LLM-Driven Framework for Iterative Kernel Optimization 

**Authors**: Martin Andrews, Sam Witteveen  

**Link**: [PDF](https://arxiv.org/pdf/2506.20807)  

**Abstract**: Optimizing GPU kernels for high performance is a complex task, often demanding deep architectural knowledge, extensive profiling, and iterative experimentation. This challenge is amplified when targeting newer or less-documented GPU architectures where traditional development aids are scarce. This paper introduces an LLM-powered "GPU Kernel Scientist," an automated methodology for iteratively refining accelerator kernels.
Our methodology employs LLMs in a multi-stage, evolutionary process: (a) strategically selecting promising prior code versions as a basis for new iterations; (b) generating hypotheses for optimization experiments, based on existing code and assimilated knowledge from general GPU literature; and (c) autonomously implementing these experiments through code modification and subsequent submission to an external evaluation system, using only observed timing data as performance feedback. We detail how this approach navigates the challenges of the AMD MI300 target architecture and leverages LLMs to compensate for limited domain-specific human expertise.
Since quantitative results from an ongoing performance competition were embargoed on paper submission date, we present the architectural design, operational workflow, and qualitative insights, highlighting the potential of LLM-driven agents to democratise and accelerate GPU kernel optimization, especially in resource-constrained or rapidly evolving hardware environments. 

---
# Poster: Enhancing GNN Robustness for Network Intrusion Detection via Agent-based Analysis 

**Authors**: Zhonghao Zhan, Huichi Zhou, Hamed Haddadi  

**Link**: [PDF](https://arxiv.org/pdf/2506.20806)  

**Abstract**: Graph Neural Networks (GNNs) show great promise for Network Intrusion Detection Systems (NIDS), particularly in IoT environments, but suffer performance degradation due to distribution drift and lack robustness against realistic adversarial attacks. Current robustness evaluations often rely on unrealistic synthetic perturbations and lack demonstrations on systematic analysis of different kinds of adversarial attack, which encompass both black-box and white-box scenarios. This work proposes a novel approach to enhance GNN robustness and generalization by employing Large Language Models (LLMs) in an agentic pipeline as simulated cybersecurity expert agents. These agents scrutinize graph structures derived from network flow data, identifying and potentially mitigating suspicious or adversarially perturbed elements before GNN processing. Our experiments, using a framework designed for realistic evaluation and testing with a variety of adversarial attacks including a dataset collected from physical testbed experiments, demonstrate that integrating LLM analysis can significantly improve the resilience of GNN-based NIDS against challenges, showcasing the potential of LLM agent as a complementary layer in intrusion detection architectures. 

---
# The Ideation-Execution Gap: Execution Outcomes of LLM-Generated versus Human Research Ideas 

**Authors**: Chenglei Si, Tatsunori Hashimoto, Diyi Yang  

**Link**: [PDF](https://arxiv.org/pdf/2506.20803)  

**Abstract**: Large Language Models (LLMs) have shown promise in accelerating the scientific research pipeline. A key capability for this process is the ability to generate novel research ideas, and prior studies have found settings in which LLM-generated research ideas were judged as more novel than human-expert ideas. However, a good idea should not simply appear to be novel, it should also result in better research after being executed. To test whether AI-generated ideas lead to better research outcomes, we conduct an execution study by recruiting 43 expert researchers to execute randomly-assigned ideas, either written by experts or generated by an LLM. Each expert spent over 100 hours implementing the idea and wrote a 4-page short paper to document the experiments. All the executed projects are then reviewed blindly by expert NLP researchers. Comparing the review scores of the same ideas before and after execution, the scores of the LLM-generated ideas decrease significantly more than expert-written ideas on all evaluation metrics (novelty, excitement, effectiveness, and overall; p < 0.05), closing the gap between LLM and human ideas observed at the ideation stage. When comparing the aggregated review scores from the execution study, we even observe that for many metrics there is a flip in rankings where human ideas score higher than LLM ideas. This ideation-execution gap highlights the limitations of current LLMs in generating truly effective research ideas and the challenge of evaluating research ideas in the absence of execution outcomes. 

---
# Stochastic Parameter Decomposition 

**Authors**: Lucius Bushnaq, Dan Braun, Lee Sharkey  

**Link**: [PDF](https://arxiv.org/pdf/2506.20790)  

**Abstract**: A key step in reverse engineering neural networks is to decompose them into simpler parts that can be studied in relative isolation. Linear parameter decomposition -- a framework that has been proposed to resolve several issues with current decomposition methods -- decomposes neural network parameters into a sum of sparsely used vectors in parameter space. However, the current main method in this framework, Attribution-based Parameter Decomposition (APD), is impractical on account of its computational cost and sensitivity to hyperparameters. In this work, we introduce \textit{Stochastic Parameter Decomposition} (SPD), a method that is more scalable and robust to hyperparameters than APD, which we demonstrate by decomposing models that are slightly larger and more complex than was possible to decompose with APD. We also show that SPD avoids other issues, such as shrinkage of the learned parameters, and better identifies ground truth mechanisms in toy models. By bridging causal mediation analysis and network decomposition methods, this demonstration opens up new research possibilities in mechanistic interpretability by removing barriers to scaling linear parameter decomposition methods to larger models. We release a library for running SPD and reproducing our experiments at this https URL. 

---
# Agile Management for Machine Learning: A Systematic Mapping Study 

**Authors**: Lucas Romao, Hugo Villamizar, Romeu Oliveira, Silvio Alonso, Marcos Kalinowski  

**Link**: [PDF](https://arxiv.org/pdf/2506.20759)  

**Abstract**: [Context] Machine learning (ML)-enabled systems are present in our society, driving significant digital transformations. The dynamic nature of ML development, characterized by experimental cycles and rapid changes in data, poses challenges to traditional project management. Agile methods, with their flexibility and incremental delivery, seem well-suited to address this dynamism. However, it is unclear how to effectively apply these methods in the context of ML-enabled systems, where challenges require tailored approaches. [Goal] Our goal is to outline the state of the art in agile management for ML-enabled systems. [Method] We conducted a systematic mapping study using a hybrid search strategy that combines database searches with backward and forward snowballing iterations. [Results] Our study identified 27 papers published between 2008 and 2024. From these, we identified eight frameworks and categorized recommendations and practices into eight key themes, such as Iteration Flexibility, Innovative ML-specific Artifacts, and the Minimal Viable Model. The main challenge identified across studies was accurate effort estimation for ML-related tasks. [Conclusion] This study contributes by mapping the state of the art and identifying open gaps in the field. While relevant work exists, more robust empirical evaluation is still needed to validate these contributions. 

---
# Exploring the Effects of Chatbot Anthropomorphism and Human Empathy on Human Prosocial Behavior Toward Chatbots 

**Authors**: Jingshu Li, Zicheng Zhu, Renwen Zhang, Yi-Chieh Lee  

**Link**: [PDF](https://arxiv.org/pdf/2506.20748)  

**Abstract**: Chatbots are increasingly integrated into people's lives and are widely used to help people. Recently, there has also been growing interest in the reverse direction-humans help chatbots-due to a wide range of benefits including better chatbot performance, human well-being, and collaborative outcomes. However, little research has explored the factors that motivate people to help chatbots. To address this gap, we draw on the Computers Are Social Actors (CASA) framework to examine how chatbot anthropomorphism-including human-like identity, emotional expression, and non-verbal expression-influences human empathy toward chatbots and their subsequent prosocial behaviors and intentions. We also explore people's own interpretations of their prosocial behaviors toward chatbots. We conducted an online experiment (N = 244) in which chatbots made mistakes in a collaborative image labeling task and explained the reasons to participants. We then measured participants' prosocial behaviors and intentions toward the chatbots. Our findings revealed that human identity and emotional expression of chatbots increased participants' prosocial behavior and intention toward chatbots, with empathy mediating these effects. Qualitative analysis further identified two motivations for participants' prosocial behaviors: empathy for the chatbot and perceiving the chatbot as human-like. We discuss the implications of these results for understanding and promoting human prosocial behaviors toward chatbots. 

---
# Test-time Scaling Techniques in Theoretical Physics -- A Comparison of Methods on the TPBench Dataset 

**Authors**: Zhiqi Gao, Tianyi Li, Yurii Kvasiuk, Sai Chaitanya Tadepalli, Maja Rudolph, Daniel J.H. Chung, Frederic Sala, Moritz Münchmeyer  

**Link**: [PDF](https://arxiv.org/pdf/2506.20729)  

**Abstract**: Large language models (LLMs) have shown strong capabilities in complex reasoning, and test-time scaling techniques can enhance their performance with comparably low cost. Many of these methods have been developed and evaluated on mathematical reasoning benchmarks such as AIME. This paper investigates whether the lessons learned from these benchmarks generalize to the domain of advanced theoretical physics. We evaluate a range of common test-time scaling methods on the TPBench physics dataset and compare their effectiveness with results on AIME. To better leverage the structure of physics problems, we develop a novel, symbolic weak-verifier framework to improve parallel scaling results. Our empirical results demonstrate that this method significantly outperforms existing test-time scaling approaches on TPBench. We also evaluate our method on AIME, confirming its effectiveness in solving advanced mathematical problems. Our findings highlight the power of step-wise symbolic verification for tackling complex scientific problems. 

---
# On Convolutions, Intrinsic Dimension, and Diffusion Models 

**Authors**: Kin Kwan Leung, Rasa Hosseinzadeh, Gabriel Loaiza-Ganem  

**Link**: [PDF](https://arxiv.org/pdf/2506.20705)  

**Abstract**: The manifold hypothesis asserts that data of interest in high-dimensional ambient spaces, such as image data, lies on unknown low-dimensional submanifolds. Diffusion models (DMs) -- which operate by convolving data with progressively larger amounts of Gaussian noise and then learning to revert this process -- have risen to prominence as the most performant generative models, and are known to be able to learn distributions with low-dimensional support. For a given datum in one of these submanifolds, we should thus intuitively expect DMs to have implicitly learned its corresponding local intrinsic dimension (LID), i.e. the dimension of the submanifold it belongs to. Kamkari et al. (2024b) recently showed that this is indeed the case by linking this LID to the rate of change of the log marginal densities of the DM with respect to the amount of added noise, resulting in an LID estimator known as FLIPD. LID estimators such as FLIPD have a plethora of uses, among others they quantify the complexity of a given datum, and can be used to detect outliers, adversarial examples and AI-generated text. FLIPD achieves state-of-the-art performance at LID estimation, yet its theoretical underpinnings are incomplete since Kamkari et al. (2024b) only proved its correctness under the highly unrealistic assumption of affine submanifolds. In this work we bridge this gap by formally proving the correctness of FLIPD under realistic assumptions. Additionally, we show that an analogous result holds when Gaussian convolutions are replaced with uniform ones, and discuss the relevance of this result. 

---
# Diffusion Tree Sampling: Scalable inference-time alignment of diffusion models 

**Authors**: Vineet Jain, Kusha Sareen, Mohammad Pedramfar, Siamak Ravanbakhsh  

**Link**: [PDF](https://arxiv.org/pdf/2506.20701)  

**Abstract**: Adapting a pretrained diffusion model to new objectives at inference time remains an open problem in generative modeling. Existing steering methods suffer from inaccurate value estimation, especially at high noise levels, which biases guidance. Moreover, information from past runs is not reused to improve sample quality, resulting in inefficient use of compute. Inspired by the success of Monte Carlo Tree Search, we address these limitations by casting inference-time alignment as a search problem that reuses past computations. We introduce a tree-based approach that samples from the reward-aligned target density by propagating terminal rewards back through the diffusion chain and iteratively refining value estimates with each additional generation. Our proposed method, Diffusion Tree Sampling (DTS), produces asymptotically exact samples from the target distribution in the limit of infinite rollouts, and its greedy variant, Diffusion Tree Search (DTS$^\star$), performs a global search for high reward samples. On MNIST and CIFAR-10 class-conditional generation, DTS matches the FID of the best-performing baseline with up to $10\times$ less compute. In text-to-image generation and language completion tasks, DTS$^\star$ effectively searches for high reward samples that match best-of-N with up to $5\times$ less compute. By reusing information from previous generations, we get an anytime algorithm that turns additional compute into steadily better samples, providing a scalable approach for inference-time alignment of diffusion models. 

---
# IMC-PINN-FE: A Physics-Informed Neural Network for Patient-Specific Left Ventricular Finite Element Modeling with Image Motion Consistency and Biomechanical Parameter Estimation 

**Authors**: Siyu Mu, Wei Xuan Chan, Choon Hwai Yap  

**Link**: [PDF](https://arxiv.org/pdf/2506.20696)  

**Abstract**: Elucidating the biomechanical behavior of the myocardium is crucial for understanding cardiac physiology, but cannot be directly inferred from clinical imaging and typically requires finite element (FE) simulations. However, conventional FE methods are computationally expensive and often fail to reproduce observed cardiac motions. We propose IMC-PINN-FE, a physics-informed neural network (PINN) framework that integrates imaged motion consistency (IMC) with FE modeling for patient-specific left ventricular (LV) biomechanics. Cardiac motion is first estimated from MRI or echocardiography using either a pre-trained attention-based network or an unsupervised cyclic-regularized network, followed by extraction of motion modes. IMC-PINN-FE then rapidly estimates myocardial stiffness and active tension by fitting clinical pressure measurements, accelerating computation from hours to seconds compared to traditional inverse FE. Based on these parameters, it performs FE modeling across the cardiac cycle at 75x speedup. Through motion constraints, it matches imaged displacements more accurately, improving average Dice from 0.849 to 0.927, while preserving realistic pressure-volume behavior. IMC-PINN-FE advances previous PINN-FE models by introducing back-computation of material properties and better motion fidelity. Using motion from a single subject to reconstruct shape modes also avoids the need for large datasets and improves patient specificity. IMC-PINN-FE offers a robust and efficient approach for rapid, personalized, and image-consistent cardiac biomechanical modeling. 

---
# Evaluating PDE discovery methods for multiscale modeling of biological signals 

**Authors**: Andréa Ducos, Audrey Denizot, Thomas Guyet, Hugues Berry  

**Link**: [PDF](https://arxiv.org/pdf/2506.20694)  

**Abstract**: Biological systems are non-linear, include unobserved variables and the physical principles that govern their dynamics are partly unknown. This makes the characterization of their behavior very challenging. Notably, their activity occurs on multiple interdependent spatial and temporal scales that require linking mechanisms across scales. To address the challenge of bridging gaps between scales, we leverage partial differential equations (PDE) discovery. PDE discovery suggests meso-scale dynamics characteristics from micro-scale data. In this article, we present our framework combining particle-based simulations and PDE discovery and conduct preliminary experiments to assess equation discovery in controlled settings. We evaluate five state-of-the-art PDE discovery methods on particle-based simulations of calcium diffusion in astrocytes. The performances of the methods are evaluated on both the form of the discovered equation and the forecasted temporal variations of calcium concentration. Our results show that several methods accurately recover the diffusion term, highlighting the potential of PDE discovery for capturing macroscopic dynamics in biological systems from microscopic data. 

---
# U-R-VEDA: Integrating UNET, Residual Links, Edge and Dual Attention, and Vision Transformer for Accurate Semantic Segmentation of CMRs 

**Authors**: Racheal Mukisa, Arvind K. Bansal  

**Link**: [PDF](https://arxiv.org/pdf/2506.20689)  

**Abstract**: Artificial intelligence, including deep learning models, will play a transformative role in automated medical image analysis for the diagnosis of cardiac disorders and their management. Automated accurate delineation of cardiac images is the first necessary initial step for the quantification and automated diagnosis of cardiac disorders. In this paper, we propose a deep learning based enhanced UNet model, U-R-Veda, which integrates convolution transformations, vision transformer, residual links, channel-attention, and spatial attention, together with edge-detection based skip-connections for an accurate fully-automated semantic segmentation of cardiac magnetic resonance (CMR) images. The model extracts local-features and their interrelationships using a stack of combination convolution blocks, with embedded channel and spatial attention in the convolution block, and vision transformers. Deep embedding of channel and spatial attention in the convolution block identifies important features and their spatial localization. The combined edge information with channel and spatial attention as skip connection reduces information-loss during convolution transformations. The overall model significantly improves the semantic segmentation of CMR images necessary for improved medical image analysis. An algorithm for the dual attention module (channel and spatial attention) has been presented. Performance results show that U-R-Veda achieves an average accuracy of 95.2%, based on DSC metrics. The model outperforms the accuracy attained by other models, based on DSC and HD metrics, especially for the delineation of right-ventricle and left-ventricle-myocardium. 

---
# Progressive Size-Adaptive Federated Learning: A Comprehensive Framework for Heterogeneous Multi-Modal Data Systems 

**Authors**: Sajid Hussain, Muhammad Sohail, Nauman Ali Khan, Naima Iltaf, Ihtesham ul Islam  

**Link**: [PDF](https://arxiv.org/pdf/2506.20685)  

**Abstract**: Federated Learning (FL) has emerged as a transformative paradigm for distributed machine learning while preserving data privacy. However, existing approaches predominantly focus on model heterogeneity and aggregation techniques, largely overlooking the fundamental impact of dataset size characteristics on federated training dynamics. This paper introduces Size-Based Adaptive Federated Learning (SAFL), a novel progressive training framework that systematically organizes federated learning based on dataset size characteristics across heterogeneous multi-modal data. Our comprehensive experimental evaluation across 13 diverse datasets spanning 7 modalities (vision, text, time series, audio, sensor, medical vision, and multimodal) reveals critical insights: 1) an optimal dataset size range of 1000-1500 samples for federated learning effectiveness; 2) a clear modality performance hierarchy with structured data (time series, sensor) significantly outperforming unstructured data (text, multimodal); and 3) systematic performance degradation for large datasets exceeding 2000 samples. SAFL achieves an average accuracy of 87.68% across all datasets, with structured data modalities reaching 99%+ accuracy. The framework demonstrates superior communication efficiency, reducing total data transfer to 7.38 GB across 558 communications while maintaining high performance. Our real-time monitoring framework provides unprecedented insights into system resource utilization, network efficiency, and training dynamics. This work fills critical gaps in understanding how data characteristics should drive federated learning strategies, providing both theoretical insights and practical guidance for real-world FL deployments in neural network and learning systems. 

---
# Global and Local Contrastive Learning for Joint Representations from Cardiac MRI and ECG 

**Authors**: Alexander Selivanov, Philip Müller, Özgün Turgut, Nil Stolt-Ansó, Daniel Rückert  

**Link**: [PDF](https://arxiv.org/pdf/2506.20683)  

**Abstract**: An electrocardiogram (ECG) is a widely used, cost-effective tool for detecting electrical abnormalities in the heart. However, it cannot directly measure functional parameters, such as ventricular volumes and ejection fraction, which are crucial for assessing cardiac function. Cardiac magnetic resonance (CMR) is the gold standard for these measurements, providing detailed structural and functional insights, but is expensive and less accessible. To bridge this gap, we propose PTACL (Patient and Temporal Alignment Contrastive Learning), a multimodal contrastive learning framework that enhances ECG representations by integrating spatio-temporal information from CMR. PTACL uses global patient-level contrastive loss and local temporal-level contrastive loss. The global loss aligns patient-level representations by pulling ECG and CMR embeddings from the same patient closer together, while pushing apart embeddings from different patients. Local loss enforces fine-grained temporal alignment within each patient by contrasting encoded ECG segments with corresponding encoded CMR frames. This approach enriches ECG representations with diagnostic information beyond electrical activity and transfers more insights between modalities than global alignment alone, all without introducing new learnable weights. We evaluate PTACL on paired ECG-CMR data from 27,951 subjects in the UK Biobank. Compared to baseline approaches, PTACL achieves better performance in two clinically relevant tasks: (1) retrieving patients with similar cardiac phenotypes and (2) predicting CMR-derived cardiac function parameters, such as ventricular volumes and ejection fraction. Our results highlight the potential of PTACL to enhance non-invasive cardiac diagnostics using ECG. The code is available at: this https URL 

---
# Utility-Driven Speculative Decoding for Mixture-of-Experts 

**Authors**: Anish Saxena, Po-An Tsai, Hritvik Taneja, Aamer Jaleel, Moinuddin Qureshi  

**Link**: [PDF](https://arxiv.org/pdf/2506.20675)  

**Abstract**: GPU memory bandwidth is the main bottleneck for low-latency Large Language Model (LLM) inference. Speculative decoding leverages idle GPU compute by using a lightweight drafter to propose K tokens, which the LLM verifies in parallel, boosting token throughput. In conventional dense LLMs, all model weights are fetched each iteration, so speculation adds no latency overhead. Emerging Mixture of Experts (MoE) models activate only a subset of weights per token, greatly reducing data movement. However, we show that speculation is ineffective for MoEs: draft tokens collectively activate more weights, increasing data movement and verification time by 2-3x. When token throughput gains fail to offset this overhead, speculation causes slowdowns up to 1.5x, making it infeasible. Even when useful, the optimal K varies by task, model, and even between requests and iterations. Thus, despite widespread use in dense LLMs, speculation remains impractical in leading MoEs.
We present Cascade, a utility-driven framework that selectively enables speculation to avoid slowdowns and dynamically tunes K to accelerate MoE serving. Cascade uses a lightweight metric, speculation utility, the ratio of token gains to verification cost, which shows iteration-level locality, enabling periodic decisions via short test and longer set phases. For each request, Cascade disables speculation if utility drops below one during testing, and when utility exceeds one, tests multiple K-values to choose the utility-maximizing K for the set phase. We implement Cascade in vLLM and evaluate it on five popular MoEs with workloads spanning code, math, extraction, and mixed tasks. Cascade limits slowdown to 5% (vs. 1.5x) and improves throughput by 7-14% over static K, making speculative decoding practical for MoEs. 

---
# ClusterRCA: Network Failure Diagnosis in HPC Systems Using Multimodal Data 

**Authors**: Yongqian Sun, Xijie Pan, Xiao Xiong, Lei Tao, Jiaju Wang, Shenglin Zhang, Yuan Yuan, Yuqi Li, Kunlin Jian  

**Link**: [PDF](https://arxiv.org/pdf/2506.20673)  

**Abstract**: Network failure diagnosis is challenging yet critical for high-performance computing (HPC) systems. Existing methods cannot be directly applied to HPC scenarios due to data heterogeneity and lack of accuracy. This paper proposes a novel framework, called ClusterRCA, to localize culprit nodes and determine failure types by leveraging multimodal data. ClusterRCA extracts features from topologically connected network interface controller (NIC) pairs to analyze the diverse, multimodal data in HPC systems. To accurately localize culprit nodes and determine failure types, ClusterRCA combines classifier-based and graph-based approaches. A failure graph is constructed based on the output of the state classifier, and then it performs a customized random walk on the graph to localize the root cause. Experiments on datasets collected by a top-tier global HPC device vendor show ClusterRCA achieves high accuracy in diagnosing network failure for HPC systems. ClusterRCA also maintains robust performance across different application scenarios. 

---
# CBF-AFA: Chunk-Based Multi-SSL Fusion for Automatic Fluency Assessment 

**Authors**: Papa Séga Wade, Mihai Andries, Ioannis Kanellos, Thierry Moudenc  

**Link**: [PDF](https://arxiv.org/pdf/2506.20243)  

**Abstract**: Automatic fluency assessment (AFA) remains challenging, particularly in capturing speech rhythm, pauses, and disfluencies in non-native speakers. We introduce a chunk-based approach integrating self-supervised learning (SSL) models (Wav2Vec2, HuBERT, and WavLM) selected for their complementary strengths in phonetic, prosodic, and noisy speech modeling, with a hierarchical CNN-BiLSTM framework. Speech is segmented into breath-group chunks using Silero voice activity detection (Silero-VAD), enabling fine-grained temporal analysis while mitigating over-segmentation artifacts. SSL embeddings are fused via a learnable weighted mechanism, balancing acoustic and linguistic features, and enriched with chunk-level fluency markers (e.g., speech rate, pause durations, n-gram repetitions). The CNN-BiLSTM captures local and long-term dependencies across chunks. Evaluated on Avalinguo and Speechocean762, our approach improves F1-score by 2.8 and Pearson correlation by 6.2 points over single SSL baselines on Speechocean762, with gains of 4.2 F1-score and 4.0 Pearson points on Avalinguo, surpassing this http URL-based segmentation baselines. These findings highlight chunk-based multi-SSL fusion for robust fluency evaluation, though future work should explore generalization to dialects with irregular prosody. 

---
# DRAGON: Distributional Rewards Optimize Diffusion Generative Models 

**Authors**: Yatong Bai, Jonah Casebeer, Somayeh Sojoudi, Nicholas J. Bryan  

**Link**: [PDF](https://arxiv.org/pdf/2504.15217)  

**Abstract**: We present Distributional RewArds for Generative OptimizatioN (DRAGON), a versatile framework for fine-tuning media generation models towards a desired outcome. Compared with traditional reinforcement learning with human feedback (RLHF) or pairwise preference approaches such as direct preference optimization (DPO), DRAGON is more flexible. It can optimize reward functions that evaluate either individual examples or distributions of them, making it compatible with a broad spectrum of instance-wise, instance-to-distribution, and distribution-to-distribution rewards. Leveraging this versatility, we construct novel reward functions by selecting an encoder and a set of reference examples to create an exemplar distribution. When cross-modality encoders such as CLAP are used, the reference examples may be of a different modality (e.g., text versus audio). Then, DRAGON gathers online and on-policy generations, scores them to construct a positive demonstration set and a negative set, and leverages the contrast between the two sets to maximize the reward. For evaluation, we fine-tune an audio-domain text-to-music diffusion model with 20 different reward functions, including a custom music aesthetics model, CLAP score, Vendi diversity, and Frechet audio distance (FAD). We further compare instance-wise (per-song) and full-dataset FAD settings while ablating multiple FAD encoders and reference sets. Over all 20 target rewards, DRAGON achieves an 81.45% average win rate. Moreover, reward functions based on exemplar sets indeed enhance generations and are comparable to model-based rewards. With an appropriate exemplar set, DRAGON achieves a 60.95% human-voted music quality win rate without training on human preference annotations. As such, DRAGON exhibits a new approach to designing and optimizing reward functions for improving human-perceived quality. Sound examples at this https URL. 

---
