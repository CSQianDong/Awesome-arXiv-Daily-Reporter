# Attention Schema-based Attention Control (ASAC): A Cognitive-Inspired Approach for Attention Management in Transformers 

**Authors**: Krati Saxena, Federico Jurado Ruiz, Guido Manzi, Dianbo Liu, Alex Lamb  

**Link**: [PDF](https://arxiv.org/pdf/2509.16058)  

**Abstract**: Attention mechanisms have become integral in AI, significantly enhancing model performance and scalability by drawing inspiration from human cognition. Concurrently, the Attention Schema Theory (AST) in cognitive science posits that individuals manage their attention by creating a model of the attention itself, effectively allocating cognitive resources. Inspired by AST, we introduce ASAC (Attention Schema-based Attention Control), which integrates the attention schema concept into artificial neural networks. Our initial experiments focused on embedding the ASAC module within transformer architectures. This module employs a Vector-Quantized Variational AutoEncoder (VQVAE) as both an attention abstractor and controller, facilitating precise attention management. By explicitly modeling attention allocation, our approach aims to enhance system efficiency. We demonstrate ASAC's effectiveness in both the vision and NLP domains, highlighting its ability to improve classification accuracy and expedite the learning process. Our experiments with vision transformers across various datasets illustrate that the attention controller not only boosts classification accuracy but also accelerates learning. Furthermore, we have demonstrated the model's robustness and generalization capabilities across noisy and out-of-distribution datasets. In addition, we have showcased improved performance in multi-task settings. Quick experiments reveal that the attention schema-based module enhances resilience to adversarial attacks, optimizes attention to improve learning efficiency, and facilitates effective transfer learning and learning from fewer examples. These promising results establish a connection between cognitive science and machine learning, shedding light on the efficient utilization of attention mechanisms in AI systems. 

---
# Structured Information for Improving Spatial Relationships in Text-to-Image Generation 

**Authors**: Sander Schildermans, Chang Tian, Ying Jiao, Marie-Francine Moens  

**Link**: [PDF](https://arxiv.org/pdf/2509.15962)  

**Abstract**: Text-to-image (T2I) generation has advanced rapidly, yet faithfully capturing spatial relationships described in natural language prompts remains a major challenge. Prior efforts have addressed this issue through prompt optimization, spatially grounded generation, and semantic refinement. This work introduces a lightweight approach that augments prompts with tuple-based structured information, using a fine-tuned language model for automatic conversion and seamless integration into T2I pipelines. Experimental results demonstrate substantial improvements in spatial accuracy, without compromising overall image quality as measured by Inception Score. Furthermore, the automatically generated tuples exhibit quality comparable to human-crafted tuples. This structured information provides a practical and portable solution to enhance spatial relationships in T2I generation, addressing a key limitation of current large-scale generative systems. 

---
# EHR-MCP: Real-world Evaluation of Clinical Information Retrieval by Large Language Models via Model Context Protocol 

**Authors**: Kanato Masayoshi, Masahiro Hashimoto, Ryoichi Yokoyama, Naoki Toda, Yoshifumi Uwamino, Shogo Fukuda, Ho Namkoong, Masahiro Jinzaki  

**Link**: [PDF](https://arxiv.org/pdf/2509.15957)  

**Abstract**: Background: Large language models (LLMs) show promise in medicine, but their deployment in hospitals is limited by restricted access to electronic health record (EHR) systems. The Model Context Protocol (MCP) enables integration between LLMs and external tools.
Objective: To evaluate whether an LLM connected to an EHR database via MCP can autonomously retrieve clinically relevant information in a real hospital setting.
Methods: We developed EHR-MCP, a framework of custom MCP tools integrated with the hospital EHR database, and used GPT-4.1 through a LangGraph ReAct agent to interact with it. Six tasks were tested, derived from use cases of the infection control team (ICT). Eight patients discussed at ICT conferences were retrospectively analyzed. Agreement with physician-generated gold standards was measured.
Results: The LLM consistently selected and executed the correct MCP tools. Except for two tasks, all tasks achieved near-perfect accuracy. Performance was lower in the complex task requiring time-dependent calculations. Most errors arose from incorrect arguments or misinterpretation of tool results. Responses from EHR-MCP were reliable, though long and repetitive data risked exceeding the context window.
Conclusions: LLMs can retrieve clinical data from an EHR via MCP tools in a real hospital setting, achieving near-perfect performance in simple tasks while highlighting challenges in complex ones. EHR-MCP provides an infrastructure for secure, consistent data access and may serve as a foundation for hospital AI agents. Future work should extend beyond retrieval to reasoning, generation, and clinical impact assessment, paving the way for effective integration of generative AI into clinical practice. 

---
# A Comparative Study of Rule-Based and Data-Driven Approaches in Industrial Monitoring 

**Authors**: Giovanni De Gasperis, Sante Dino Facchini  

**Link**: [PDF](https://arxiv.org/pdf/2509.15848)  

**Abstract**: Industrial monitoring systems, especially when deployed in Industry 4.0 environments, are experiencing a shift in paradigm from traditional rule-based architectures to data-driven approaches leveraging machine learning and artificial intelligence. This study presents a comparison between these two methodologies, analyzing their respective strengths, limitations, and application scenarios, and proposes a basic framework to evaluate their key properties. Rule-based systems offer high interpretability, deterministic behavior, and ease of implementation in stable environments, making them ideal for regulated industries and safety-critical applications. However, they face challenges with scalability, adaptability, and performance in complex or evolving contexts. Conversely, data-driven systems excel in detecting hidden anomalies, enabling predictive maintenance and dynamic adaptation to new conditions. Despite their high accuracy, these models face challenges related to data availability, explainability, and integration complexity. The paper suggests hybrid solutions as a possible promising direction, combining the transparency of rule-based logic with the analytical power of machine learning. Our hypothesis is that the future of industrial monitoring lies in intelligent, synergic systems that leverage both expert knowledge and data-driven insights. This dual approach enhances resilience, operational efficiency, and trust, paving the way for smarter and more flexible industrial environments. 

---
# Building Data-Driven Occupation Taxonomies: A Bottom-Up Multi-Stage Approach via Semantic Clustering and Multi-Agent Collaboration 

**Authors**: Nan Li, Bo Kang, Tijl De Bie  

**Link**: [PDF](https://arxiv.org/pdf/2509.15786)  

**Abstract**: Creating robust occupation taxonomies, vital for applications ranging from job recommendation to labor market intelligence, is challenging. Manual curation is slow, while existing automated methods are either not adaptive to dynamic regional markets (top-down) or struggle to build coherent hierarchies from noisy data (bottom-up). We introduce CLIMB (CLusterIng-based Multi-agent taxonomy Builder), a framework that fully automates the creation of high-quality, data-driven taxonomies from raw job postings. CLIMB uses global semantic clustering to distill core occupations, then employs a reflection-based multi-agent system to iteratively build a coherent hierarchy. On three diverse, real-world datasets, we show that CLIMB produces taxonomies that are more coherent and scalable than existing methods and successfully capture unique regional characteristics. We release our code and datasets at this https URL. 

---
# Ontology Creation and Management Tools: the Case of Anatomical Connectivity 

**Authors**: Natallia Kokash, Bernard de Bono, Tom Gillespie  

**Link**: [PDF](https://arxiv.org/pdf/2509.15780)  

**Abstract**: We are developing infrastructure to support researchers in mapping data related to the peripheral nervous system and other physiological systems, with an emphasis on their relevance to the organs under investigation. The nervous system, a complex network of nerves and ganglia, plays a critical role in coordinating and transmitting signals throughout the body. To aid in this, we have created ApiNATOMY, a framework for the topological and semantic representation of multiscale physiological circuit maps. ApiNATOMY integrates a Knowledge Representation (KR) model and a suite of Knowledge Management (KM) tools. The KR model enables physiology experts to easily capture interactions between anatomical entities, while the KM tools help modelers convert high-level abstractions into detailed models of physiological processes, which can be integrated with external ontologies and knowledge graphs. 

---
# A Nascent Taxonomy of Machine Learning in Intelligent Robotic Process Automation 

**Authors**: Lukas Laakmann, Seyyid A. Ciftci, Christian Janiesch  

**Link**: [PDF](https://arxiv.org/pdf/2509.15730)  

**Abstract**: Robotic process automation (RPA) is a lightweight approach to automating business processes using software robots that emulate user actions at the graphical user interface level. While RPA has gained popularity for its cost-effective and timely automation of rule-based, well-structured tasks, its symbolic nature has inherent limitations when approaching more complex tasks currently performed by human agents. Machine learning concepts enabling intelligent RPA provide an opportunity to broaden the range of automatable tasks. In this paper, we conduct a literature review to explore the connections between RPA and machine learning and organize the joint concept intelligent RPA into a taxonomy. Our taxonomy comprises the two meta-characteristics RPA-ML integration and RPA-ML interaction. Together, they comprise eight dimensions: architecture and ecosystem, capabilities, data basis, intelligence level, and technical depth of integration as well as deployment environment, lifecycle phase, and user-robot relation. 

---
# CCrepairBench: A High-Fidelity Benchmark and Reinforcement Learning Framework for C++ Compilation Repair 

**Authors**: Weixuan Sun, Jucai Zhai, Dengfeng Liu, Xin Zhang, Xiaojun Wu, Qiaobo Hao, AIMgroup, Yang Fang, Jiuyang Tang  

**Link**: [PDF](https://arxiv.org/pdf/2509.15690)  

**Abstract**: The automated repair of C++ compilation errors presents a significant challenge, the resolution of which is critical for developer productivity. Progress in this domain is constrained by two primary factors: the scarcity of large-scale, high-fidelity datasets and the limitations of conventional supervised methods, which often fail to generate semantically correct this http URL paper addresses these gaps by introducing a comprehensive framework with three core contributions. First, we present CCrepair, a novel, large-scale C++ compilation error dataset constructed through a sophisticated generate-and-verify pipeline. Second, we propose a Reinforcement Learning (RL) paradigm guided by a hybrid reward signal, shifting the focus from mere compilability to the semantic quality of the fix. Finally, we establish the robust, two-stage evaluation system providing this signal, centered on an LLM-as-a-Judge whose reliability has been rigorously validated against the collective judgments of a panel of human experts. This integrated approach aligns the training objective with generating high-quality, non-trivial patches that are both syntactically and semantically correct. The effectiveness of our approach was demonstrated experimentally. Our RL-trained Qwen2.5-1.5B-Instruct model achieved performance comparable to a Qwen2.5-14B-Instruct model, validating the efficiency of our training paradigm. Our work provides the research community with a valuable new dataset and a more effective paradigm for training and evaluating robust compilation repair models, paving the way for more practical and reliable automated programming assistants. 

---
# MicroRCA-Agent: Microservice Root Cause Analysis Method Based on Large Language Model Agents 

**Authors**: Pan Tang, Shixiang Tang, Huanqi Pu, Zhiqing Miao, Zhixing Wang  

**Link**: [PDF](https://arxiv.org/pdf/2509.15635)  

**Abstract**: This paper presents MicroRCA-Agent, an innovative solution for microservice root cause analysis based on large language model agents, which constructs an intelligent fault root cause localization system with multimodal data fusion. The technical innovations are embodied in three key aspects: First, we combine the pre-trained Drain log parsing algorithm with multi-level data filtering mechanism to efficiently compress massive logs into high-quality fault features. Second, we employ a dual anomaly detection approach that integrates Isolation Forest unsupervised learning algorithms with status code validation to achieve comprehensive trace anomaly identification. Third, we design a statistical symmetry ratio filtering mechanism coupled with a two-stage LLM analysis strategy to enable full-stack phenomenon summarization across node-service-pod hierarchies. The multimodal root cause analysis module leverages carefully designed cross-modal prompts to deeply integrate multimodal anomaly information, fully exploiting the cross-modal understanding and logical reasoning capabilities of large language models to generate structured analysis results encompassing fault components, root cause descriptions, and reasoning trace. Comprehensive ablation studies validate the complementary value of each modal data and the effectiveness of the system architecture. The proposed solution demonstrates superior performance in complex microservice fault scenarios, achieving a final score of 50.71. The code has been released at: this https URL. 

---
# Stress Testing Deliberative Alignment for Anti-Scheming Training 

**Authors**: Bronson Schoen, Evgenia Nitishinskaya, Mikita Balesni, Axel Højmark, Felix Hofstätter, Jérémy Scheurer, Alexander Meinke, Jason Wolfe, Teun van der Weij, Alex Lloyd, Nicholas Goldowsky-Dill, Angela Fan, Andrei Matveiakin, Rusheb Shah, Marcus Williams, Amelia Glaese, Boaz Barak, Wojciech Zaremba, Marius Hobbhahn  

**Link**: [PDF](https://arxiv.org/pdf/2509.15541)  

**Abstract**: Highly capable AI systems could secretly pursue misaligned goals -- what we call "scheming". Because a scheming AI would deliberately try to hide its misaligned goals and actions, measuring and mitigating scheming requires different strategies than are typically used in ML. We propose that assessing anti-scheming interventions requires at least (1) testing propensity to scheme on far out-of-distribution (OOD) tasks, (2) evaluating whether lack of scheming is driven by situational awareness, and (3) checking for robustness to pre-existing misaligned goals. We use a broad category of "covert actions" -- such as secretly breaking rules or intentionally underperforming in tests -- as a proxy for scheming, and design evaluations for covert actions. We then stress-test deliberative alignment as a case study for anti-scheming. Across 26 OOD evaluations (180+ environments), deliberative alignment reduces covert action rates (OpenAI o3: 13%->0.4%) but does not fully eliminate them. Our mitigation is also able to largely stop agents from pursuing a hidden goal previously trained into the model, but we still find misbehavior after additional red-teaming. We find that models' chain-of-thought (CoT) often demonstrates awareness of being evaluated for alignment, and show causal evidence that this awareness decreases covert behavior, while unawareness increases it. Therefore, we cannot exclude that the observed reductions in covert action rates are at least partially driven by situational awareness. While we rely on human-legible CoT for training, studying situational awareness, and demonstrating clear evidence of misalignment, our ability to rely on this degrades as models continue to depart from reasoning in standard English. We encourage research into alignment mitigations for scheming and their assessment, especially for the adversarial case of deceptive alignment, which this paper does not address. 

---
# FragmentRetro: A Quadratic Retrosynthetic Method Based on Fragmentation Algorithms 

**Authors**: Yu Shee, Anthony M. Smaldone, Anton Morgunov, Gregory W. Kyro, Victor S. Batista  

**Link**: [PDF](https://arxiv.org/pdf/2509.15409)  

**Abstract**: Retrosynthesis, the process of deconstructing a target molecule into simpler precursors, is crucial for computer-aided synthesis planning (CASP). Widely adopted tree-search methods often suffer from exponential computational complexity. In this work, we introduce FragmentRetro, a novel retrosynthetic method that leverages fragmentation algorithms, specifically BRICS and r-BRICS, combined with stock-aware exploration and pattern fingerprint screening to achieve quadratic complexity. FragmentRetro recursively combines molecular fragments and verifies their presence in a building block set, providing sets of fragment combinations as retrosynthetic solutions. We present the first formal computational analysis of retrosynthetic methods, showing that tree search exhibits exponential complexity $O(b^h)$, DirectMultiStep scales as $O(h^6)$, and FragmentRetro achieves $O(h^2)$, where $h$ represents the number of heavy atoms in the target molecule and $b$ is the branching factor for tree search. Evaluations on PaRoutes, USPTO-190, and natural products demonstrate that FragmentRetro achieves high solved rates with competitive runtime, including cases where tree search fails. The method benefits from fingerprint screening, which significantly reduces substructure matching complexity. While FragmentRetro focuses on efficiently identifying fragment-based solutions rather than full reaction pathways, its computational advantages and ability to generate strategic starting candidates establish it as a powerful foundational component for scalable and automated synthesis planning. 

---
# Diagnostics of cognitive failures in multi-agent expert systems using dynamic evaluation protocols and subsequent mutation of the processing context 

**Authors**: Andrejs Sorstkins, Josh Bailey, Dr Alistair Baron  

**Link**: [PDF](https://arxiv.org/pdf/2509.15366)  

**Abstract**: The rapid evolution of neural architectures - from multilayer perceptrons to large-scale Transformer-based models - has enabled language models (LLMs) to exhibit emergent agentic behaviours when equipped with memory, planning, and external tool use. However, their inherent stochasticity and multi-step decision processes render classical evaluation methods inadequate for diagnosing agentic performance. This work introduces a diagnostic framework for expert systems that not only evaluates but also facilitates the transfer of expert behaviour into LLM-powered agents. The framework integrates (i) curated golden datasets of expert annotations, (ii) silver datasets generated through controlled behavioural mutation, and (iii) an LLM-based Agent Judge that scores and prescribes targeted improvements. These prescriptions are embedded into a vectorized recommendation map, allowing expert interventions to propagate as reusable improvement trajectories across multiple system instances. We demonstrate the framework on a multi-agent recruiter-assistant system, showing that it uncovers latent cognitive failures - such as biased phrasing, extraction drift, and tool misrouting - while simultaneously steering agents toward expert-level reasoning and style. The results establish a foundation for standardized, reproducible expert behaviour transfer in stochastic, tool-augmented LLM agents, moving beyond static evaluation to active expert system refinement. 

---
# Knowledge-Driven Hallucination in Large Language Models: An Empirical Study on Process Modeling 

**Authors**: Humam Kourani, Anton Antonov, Alessandro Berti, Wil M.P. van der Aalst  

**Link**: [PDF](https://arxiv.org/pdf/2509.15336)  

**Abstract**: The utility of Large Language Models (LLMs) in analytical tasks is rooted in their vast pre-trained knowledge, which allows them to interpret ambiguous inputs and infer missing information. However, this same capability introduces a critical risk of what we term knowledge-driven hallucination: a phenomenon where the model's output contradicts explicit source evidence because it is overridden by the model's generalized internal knowledge. This paper investigates this phenomenon by evaluating LLMs on the task of automated process modeling, where the goal is to generate a formal business process model from a given source artifact. The domain of Business Process Management (BPM) provides an ideal context for this study, as many core business processes follow standardized patterns, making it likely that LLMs possess strong pre-trained schemas for them. We conduct a controlled experiment designed to create scenarios with deliberate conflict between provided evidence and the LLM's background knowledge. We use inputs describing both standard and deliberately atypical process structures to measure the LLM's fidelity to the provided evidence. Our work provides a methodology for assessing this critical reliability issue and raises awareness of the need for rigorous validation of AI-generated artifacts in any evidence-based domain. 

---
# An Artificial Intelligence Driven Semantic Similarity-Based Pipeline for Rapid Literature 

**Authors**: Abhiyan Dhakal, Kausik Paudel, Sanjog Sigdel  

**Link**: [PDF](https://arxiv.org/pdf/2509.15292)  

**Abstract**: We propose an automated pipeline for performing literature reviews using semantic similarity. Unlike traditional systematic review systems or optimization based methods, this work emphasizes minimal overhead and high relevance by using transformer based embeddings and cosine similarity. By providing a paper title and abstract, it generates relevant keywords, fetches relevant papers from open access repository, and ranks them based on their semantic closeness to the input. Three embedding models were evaluated. A statistical thresholding approach is then applied to filter relevant papers, enabling an effective literature review pipeline. Despite the absence of heuristic feedback or ground truth relevance labels, the proposed system shows promise as a scalable and practical tool for preliminary research and exploratory analysis. 

---
# The Distribution Shift Problem in Transportation Networks using Reinforcement Learning and AI 

**Authors**: Federico Taschin, Abderrahmane Lazaraq, Ozan K. Tonguz, Inci Ozgunes  

**Link**: [PDF](https://arxiv.org/pdf/2509.15291)  

**Abstract**: The use of Machine Learning (ML) and Artificial Intelligence (AI) in smart transportation networks has increased significantly in the last few years. Among these ML and AI approaches, Reinforcement Learning (RL) has been shown to be a very promising approach by several authors. However, a problem with using Reinforcement Learning in Traffic Signal Control is the reliability of the trained RL agents due to the dynamically changing distribution of the input data with respect to the distribution of the data used for training. This presents a major challenge and a reliability problem for the trained network of AI agents and could have very undesirable and even detrimental consequences if a suitable solution is not found. Several researchers have tried to address this problem using different approaches. In particular, Meta Reinforcement Learning (Meta RL) promises to be an effective solution. In this paper, we evaluate and analyze a state-of-the-art Meta RL approach called MetaLight and show that, while under certain conditions MetaLight can indeed lead to reasonably good results, under some other conditions it might not perform well (with errors of up to 22%), suggesting that Meta RL schemes are often not robust enough and can even pose major reliability problems. 

---
# KNARsack: Teaching Neural Algorithmic Reasoners to Solve Pseudo-Polynomial Problems 

**Authors**: Stjepan Požgaj, Dobrik Georgiev, Marin Šilić, Petar Veličković  

**Link**: [PDF](https://arxiv.org/pdf/2509.15239)  

**Abstract**: Neural algorithmic reasoning (NAR) is a growing field that aims to embed algorithmic logic into neural networks by imitating classical algorithms. In this extended abstract, we detail our attempt to build a neural algorithmic reasoner that can solve Knapsack, a pseudo-polynomial problem bridging classical algorithms and combinatorial optimisation, but omitted in standard NAR benchmarks. Our neural algorithmic reasoner is designed to closely follow the two-phase pipeline for the Knapsack problem, which involves first constructing the dynamic programming table and then reconstructing the solution from it. The approach, which models intermediate states through dynamic programming supervision, achieves better generalization to larger problem instances than a direct-prediction baseline that attempts to select the optimal subset only from the problem inputs. 

---
# MICA: Multi-Agent Industrial Coordination Assistant 

**Authors**: Di Wen, Kunyu Peng, Junwei Zheng, Yufan Chen, Yitain Shi, Jiale Wei, Ruiping Liu, Kailun Yang, Rainer Stiefelhagen  

**Link**: [PDF](https://arxiv.org/pdf/2509.15237)  

**Abstract**: Industrial workflows demand adaptive and trustworthy assistance that can operate under limited computing, connectivity, and strict privacy constraints. In this work, we present MICA (Multi-Agent Industrial Coordination Assistant), a perception-grounded and speech-interactive system that delivers real-time guidance for assembly, troubleshooting, part queries, and maintenance. MICA coordinates five role-specialized language agents, audited by a safety checker, to ensure accurate and compliant support. To achieve robust step understanding, we introduce Adaptive Step Fusion (ASF), which dynamically blends expert reasoning with online adaptation from natural speech feedback. Furthermore, we establish a new multi-agent coordination benchmark across representative task categories and propose evaluation metrics tailored to industrial assistance, enabling systematic comparison of different coordination topologies. Our experiments demonstrate that MICA consistently improves task success, reliability, and responsiveness over baseline structures, while remaining deployable on practical offline hardware. Together, these contributions highlight MICA as a step toward deployable, privacy-preserving multi-agent assistants for dynamic factory environments. The source code will be made publicly available at this https URL. 

---
# RPG: A Repository Planning Graph for Unified and Scalable Codebase Generation 

**Authors**: Jane Luo, Xin Zhang, Steven Liu, Jie Wu, Yiming Huang, Yangyu Huang, Chengyu Yin, Ying Xin, Jianfeng Liu, Yuefeng Zhan, Hao Sun, Qi Chen, Scarlett Li, Mao Yang  

**Link**: [PDF](https://arxiv.org/pdf/2509.16198)  

**Abstract**: Large language models excel at function- and file-level code generation, yet generating complete repositories from scratch remains a fundamental challenge. This process demands coherent and reliable planning across proposal- and implementation-level stages, while natural language, due to its ambiguity and verbosity, is ill-suited for faithfully representing complex software structures. To address this, we introduce the Repository Planning Graph (RPG), a persistent representation that unifies proposal- and implementation-level planning by encoding capabilities, file structures, data flows, and functions in one graph. RPG replaces ambiguous natural language with an explicit blueprint, enabling long-horizon planning and scalable repository generation. Building on RPG, we develop ZeroRepo, a graph-driven framework for repository generation from scratch. It operates in three stages: proposal-level planning and implementation-level refinement to construct the graph, followed by graph-guided code generation with test validation. To evaluate this setting, we construct RepoCraft, a benchmark of six real-world projects with 1,052 tasks. On RepoCraft, ZeroRepo produces repositories averaging nearly 36K LOC, roughly 3.9$\times$ the strongest baseline (Claude Code) and about 64$\times$ other baselines. It attains 81.5% functional coverage and a 69.7% pass rate, exceeding Claude Code by 27.3 and 35.8 percentage points, respectively. Further analysis shows that RPG models complex dependencies, enables progressively more sophisticated planning through near-linear scaling, and enhances LLM understanding of repositories, thereby accelerating agent localization. 

---
# FocalCodec-Stream: Streaming Low-Bitrate Speech Coding via Causal Distillation 

**Authors**: Luca Della Libera, Cem Subakan, Mirco Ravanelli  

**Link**: [PDF](https://arxiv.org/pdf/2509.16195)  

**Abstract**: Neural audio codecs are a fundamental component of modern generative audio pipelines. Although recent codecs achieve strong low-bitrate reconstruction and provide powerful representations for downstream tasks, most are non-streamable, limiting their use in real-time applications. We present FocalCodec-Stream, a hybrid codec based on focal modulation that compresses speech into a single binary codebook at 0.55 - 0.80 kbps with a theoretical latency of 80 ms. Our approach combines multi-stage causal distillation of WavLM with targeted architectural improvements, including a lightweight refiner module that enhances quality under latency constraints. Experiments show that FocalCodec-Stream outperforms existing streamable codecs at comparable bitrates, while preserving both semantic and acoustic information. The result is a favorable trade-off between reconstruction quality, downstream task performance, latency, and efficiency. Code and checkpoints will be released at this https URL. 

---
# CultureScope: A Dimensional Lens for Probing Cultural Understanding in LLMs 

**Authors**: Jinghao Zhang, Sihang Jiang, Shiwei Guo, Shisong Chen, Yanghua Xiao, Hongwei Feng, Jiaqing Liang, Minggui HE, Shimin Tao, Hongxia Ma  

**Link**: [PDF](https://arxiv.org/pdf/2509.16188)  

**Abstract**: As large language models (LLMs) are increasingly deployed in diverse cultural environments, evaluating their cultural understanding capability has become essential for ensuring trustworthy and culturally aligned applications. However, most existing benchmarks lack comprehensiveness and are challenging to scale and adapt across different cultural contexts, because their frameworks often lack guidance from well-established cultural theories and tend to rely on expert-driven manual annotations. To address these issues, we propose CultureScope, the most comprehensive evaluation framework to date for assessing cultural understanding in LLMs. Inspired by the cultural iceberg theory, we design a novel dimensional schema for cultural knowledge classification, comprising 3 layers and 140 dimensions, which guides the automated construction of culture-specific knowledge bases and corresponding evaluation datasets for any given languages and cultures. Experimental results demonstrate that our method can effectively evaluate cultural understanding. They also reveal that existing large language models lack comprehensive cultural competence, and merely incorporating multilingual data does not necessarily enhance cultural understanding. All code and data files are available at this https URL 

---
# Accelerating Atomic Fine Structure Determination with Graph Reinforcement Learning 

**Authors**: M. Ding, V.-A. Darvariu, A. N. Ryabtsev, N. Hawes, J. C. Pickering  

**Link**: [PDF](https://arxiv.org/pdf/2509.16184)  

**Abstract**: Atomic data determined by analysis of observed atomic spectra are essential for plasma diagnostics. For each low-ionisation open d- and f-subshell atomic species, around $10^3$ fine structure level energies can be determined through years of analysis of $10^4$ observable spectral lines. We propose the automation of this task by casting the analysis procedure as a Markov decision process and solving it by graph reinforcement learning using reward functions learned on historical human decisions. In our evaluations on existing spectral line lists and theoretical calculations for Co II and Nd II-III, hundreds of level energies were computed within hours, agreeing with published values in 95% of cases for Co II and 54-87% for Nd II-III. As the current efficiency in atomic fine structure determination struggles to meet growing atomic data demands from astronomy and fusion science, our new artificial intelligence approach sets the stage for closing this gap. 

---
# Fast OTSU Thresholding Using Bisection Method 

**Authors**: Sai Varun Kodathala  

**Link**: [PDF](https://arxiv.org/pdf/2509.16179)  

**Abstract**: The Otsu thresholding algorithm represents a fundamental technique in image segmentation, yet its computational efficiency is severely limited by exhaustive search requirements across all possible threshold values. This work presents an optimized implementation that leverages the bisection method to exploit the unimodal characteristics of the between-class variance function. Our approach reduces the computational complexity from O(L) to O(log L) evaluations while preserving segmentation accuracy. Experimental validation on 48 standard test images demonstrates a 91.63% reduction in variance computations and 97.21% reduction in algorithmic iterations compared to conventional exhaustive search. The bisection method achieves exact threshold matches in 66.67% of test cases, with 95.83% exhibiting deviations within 5 gray levels. The algorithm maintains universal convergence within theoretical logarithmic bounds while providing deterministic performance guarantees suitable for real-time applications. This optimization addresses critical computational bottlenecks in large-scale image processing systems without compromising the theoretical foundations or segmentation quality of the original Otsu method. 

---
# Robust Vision-Language Models via Tensor Decomposition: A Defense Against Adversarial Attacks 

**Authors**: Het Patel, Muzammil Allie, Qian Zhang, Jia Chen, Evangelos E. Papalexakis  

**Link**: [PDF](https://arxiv.org/pdf/2509.16163)  

**Abstract**: Vision language models (VLMs) excel in multimodal understanding but are prone to adversarial attacks. Existing defenses often demand costly retraining or significant architecture changes. We introduce a lightweight defense using tensor decomposition suitable for any pre-trained VLM, requiring no retraining. By decomposing and reconstructing vision encoder representations, it filters adversarial noise while preserving meaning. Experiments with CLIP on COCO and Flickr30K show improved robustness. On Flickr30K, it restores 12.3\% performance lost to attacks, raising Recall@1 accuracy from 7.5\% to 19.8\%. On COCO, it recovers 8.1\% performance, improving accuracy from 3.8\% to 11.9\%. Analysis shows Tensor Train decomposition with low rank (8-32) and low residual strength ($\alpha=0.1-0.2$) is optimal. This method is a practical, plug-and-play solution with minimal overhead for existing VLMs. 

---
# Network-Based Detection of Autism Spectrum Disorder Using Sustainable and Non-invasive Salivary Biomarkers 

**Authors**: Janayna M. Fernandes, Robinson Sabino-Silva, Murillo G. Carneiro  

**Link**: [PDF](https://arxiv.org/pdf/2509.16126)  

**Abstract**: Autism Spectrum Disorder (ASD) lacks reliable biological markers, delaying early diagnosis. Using 159 salivary samples analyzed by ATR-FTIR spectroscopy, we developed GANet, a genetic algorithm-based network optimization framework leveraging PageRank and Degree for importance-based feature characterization. GANet systematically optimizes network structure to extract meaningful patterns from high-dimensional spectral data. It achieved superior performance compared to linear discriminant analysis, support vector machines, and deep learning models, reaching 0.78 accuracy, 0.61 sensitivity, 0.90 specificity, and a 0.74 harmonic mean. These results demonstrate GANet's potential as a robust, bio-inspired, non-invasive tool for precise ASD detection and broader spectral-based health applications. 

---
# DiffusionNFT: Online Diffusion Reinforcement with Forward Process 

**Authors**: Kaiwen Zheng, Huayu Chen, Haotian Ye, Haoxiang Wang, Qinsheng Zhang, Kai Jiang, Hang Su, Stefano Ermon, Jun Zhu, Ming-Yu Liu  

**Link**: [PDF](https://arxiv.org/pdf/2509.16117)  

**Abstract**: Online reinforcement learning (RL) has been central to post-training language models, but its extension to diffusion models remains challenging due to intractable likelihoods. Recent works discretize the reverse sampling process to enable GRPO-style training, yet they inherit fundamental drawbacks, including solver restrictions, forward-reverse inconsistency, and complicated integration with classifier-free guidance (CFG). We introduce Diffusion Negative-aware FineTuning (DiffusionNFT), a new online RL paradigm that optimizes diffusion models directly on the forward process via flow matching. DiffusionNFT contrasts positive and negative generations to define an implicit policy improvement direction, naturally incorporating reinforcement signals into the supervised learning objective. This formulation enables training with arbitrary black-box solvers, eliminates the need for likelihood estimation, and requires only clean images rather than sampling trajectories for policy optimization. DiffusionNFT is up to $25\times$ more efficient than FlowGRPO in head-to-head comparisons, while being CFG-free. For instance, DiffusionNFT improves the GenEval score from 0.24 to 0.98 within 1k steps, while FlowGRPO achieves 0.95 with over 5k steps and additional CFG employment. By leveraging multiple reward models, DiffusionNFT significantly boosts the performance of SD3.5-Medium in every benchmark tested. 

---
# Beyond Pointwise Scores: Decomposed Criteria-Based Evaluation of LLM Responses 

**Authors**: Fangyi Yu, Nabeel Seedat, Dasha Herrmannova, Frank Schilder, Jonathan Richard Schwarz  

**Link**: [PDF](https://arxiv.org/pdf/2509.16093)  

**Abstract**: Evaluating long-form answers in high-stakes domains such as law or medicine remains a fundamental challenge. Standard metrics like BLEU and ROUGE fail to capture semantic correctness, and current LLM-based evaluators often reduce nuanced aspects of answer quality into a single undifferentiated score. We introduce DeCE, a decomposed LLM evaluation framework that separates precision (factual accuracy and relevance) and recall (coverage of required concepts), using instance-specific criteria automatically extracted from gold answer requirements. DeCE is model-agnostic and domain-general, requiring no predefined taxonomies or handcrafted rubrics. We instantiate DeCE to evaluate different LLMs on a real-world legal QA task involving multi-jurisdictional reasoning and citation grounding. DeCE achieves substantially stronger correlation with expert judgments ($r=0.78$), compared to traditional metrics ($r=0.12$), pointwise LLM scoring ($r=0.35$), and modern multidimensional evaluators ($r=0.48$). It also reveals interpretable trade-offs: generalist models favor recall, while specialized models favor precision. Importantly, only 11.95% of LLM-generated criteria required expert revision, underscoring DeCE's scalability. DeCE offers an interpretable and actionable LLM evaluation framework in expert domains. 

---
# See&Trek: Training-Free Spatial Prompting for Multimodal Large Language Model 

**Authors**: Pengteng Li, Pinhao Song, Wuyang Li, Weiyu Guo, Huizai Yao, Yijie Xu, Dugang Liu, Hui Xiong  

**Link**: [PDF](https://arxiv.org/pdf/2509.16087)  

**Abstract**: We introduce SEE&TREK, the first training-free prompting framework tailored to enhance the spatial understanding of Multimodal Large Language Models (MLLMS) under vision-only constraints. While prior efforts have incorporated modalities like depth or point clouds to improve spatial reasoning, purely visualspatial understanding remains underexplored. SEE&TREK addresses this gap by focusing on two core principles: increasing visual diversity and motion reconstruction. For visual diversity, we conduct Maximum Semantic Richness Sampling, which employs an off-the-shell perception model to extract semantically rich keyframes that capture scene structure. For motion reconstruction, we simulate visual trajectories and encode relative spatial positions into keyframes to preserve both spatial relations and temporal coherence. Our method is training&GPU-free, requiring only a single forward pass, and can be seamlessly integrated into existing MLLM'S. Extensive experiments on the VSI-B ENCH and STI-B ENCH show that S EE &T REK consistently boosts various MLLM S performance across diverse spatial reasoning tasks with the most +3.5% improvement, offering a promising path toward stronger spatial intelligence. 

---
# Communications to Circulations: 3D Wind Field Retrieval and Real-Time Prediction Using 5G GNSS Signals and Deep Learning 

**Authors**: Yuchen Ye, Hong Liang, Chaoxia Yuan, Mingyu Li, Aoqi Zhou, Chunqing Shang, Hua Cai, Peixi Liu, Kezuan Wang, Yifeng Zheng  

**Link**: [PDF](https://arxiv.org/pdf/2509.16068)  

**Abstract**: Accurate atmospheric wind field information is crucial for various applications, including weather forecasting, aviation safety, and disaster risk reduction. However, obtaining high spatiotemporal resolution wind data remains challenging due to limitations in traditional in-situ observations and remote sensing techniques, as well as the computational expense and biases of numerical weather prediction (NWP) models. This paper introduces G-WindCast, a novel deep learning framework that leverages signal strength variations from 5G Global Navigation Satellite System (GNSS) signals to retrieve and forecast three-dimensional (3D) atmospheric wind fields. The framework utilizes Forward Neural Networks (FNN) and Transformer networks to capture complex, nonlinear, and spatiotemporal relationships between GNSS-derived features and wind dynamics. Our preliminary results demonstrate promising accuracy in both wind retrieval and short-term wind forecasting (up to 30 minutes lead time), with skill scores comparable to high-resolution NWP outputs in certain scenarios. The model exhibits robustness across different forecast horizons and pressure levels, and its predictions for wind speed and direction show superior agreement with observations compared to concurrent ERA5 reanalysis data. Furthermore, we show that the system can maintain excellent performance for localized forecasting even with a significantly reduced number of GNSS stations (e.g., around 100), highlighting its cost-effectiveness and scalability. This interdisciplinary approach underscores the transformative potential of exploiting non-traditional data sources and deep learning for advanced environmental monitoring and real-time atmospheric applications. 

---
# Compose by Focus: Scene Graph-based Atomic Skills 

**Authors**: Han Qi, Changhe Chen, Heng Yang  

**Link**: [PDF](https://arxiv.org/pdf/2509.16053)  

**Abstract**: A key requirement for generalist robots is compositional generalization - the ability to combine atomic skills to solve complex, long-horizon tasks. While prior work has primarily focused on synthesizing a planner that sequences pre-learned skills, robust execution of the individual skills themselves remains challenging, as visuomotor policies often fail under distribution shifts induced by scene composition. To address this, we introduce a scene graph-based representation that focuses on task-relevant objects and relations, thereby mitigating sensitivity to irrelevant variation. Building on this idea, we develop a scene-graph skill learning framework that integrates graph neural networks with diffusion-based imitation learning, and further combine "focused" scene-graph skills with a vision-language model (VLM) based task planner. Experiments in both simulation and real-world manipulation tasks demonstrate substantially higher success rates than state-of-the-art baselines, highlighting improved robustness and compositional generalization in long-horizon tasks. 

---
# Think, Verbalize, then Speak: Bridging Complex Thoughts and Comprehensible Speech 

**Authors**: Sang Hoon Woo, Sehun Lee, Kang-wook Kim, Gunhee Kim  

**Link**: [PDF](https://arxiv.org/pdf/2509.16028)  

**Abstract**: Spoken dialogue systems increasingly employ large language models (LLMs) to leverage their advanced reasoning capabilities. However, direct application of LLMs in spoken communication often yield suboptimal results due to mismatches between optimal textual and verbal delivery. While existing approaches adapt LLMs to produce speech-friendly outputs, their impact on reasoning performance remains underexplored. In this work, we propose Think-Verbalize-Speak, a framework that decouples reasoning from spoken delivery to preserve the full reasoning capacity of LLMs. Central to our method is verbalizing, an intermediate step that translates thoughts into natural, speech-ready text. We also introduce ReVerT, a latency-efficient verbalizer based on incremental and asynchronous summarization. Experiments across multiple benchmarks show that our method enhances speech naturalness and conciseness with minimal impact on reasoning. The project page with the dataset and the source code is available at this https URL 

---
# Session-Level Spoken Language Assessment with a Multimodal Foundation Model via Multi-Target Learning 

**Authors**: Hong-Yun Lin, Jhen-Ke Lin, Chung-Chun Wang, Hao-Chien Lu, Berlin Chen  

**Link**: [PDF](https://arxiv.org/pdf/2509.16025)  

**Abstract**: Spoken Language Assessment (SLA) estimates a learner's oral proficiency from spontaneous speech. The growing population of L2 English speakers has intensified the demand for reliable SLA, a critical component of Computer Assisted Language Learning (CALL). Existing efforts often rely on cascaded pipelines, which are prone to error propagation, or end-to-end models that often operate on a short audio window, which might miss discourse-level evidence. This paper introduces a novel multimodal foundation model approach that performs session-level evaluation in a single pass. Our approach couples multi-target learning with a frozen, Whisper ASR model-based speech prior for acoustic-aware calibration, allowing for jointly learning holistic and trait-level objectives of SLA without resorting to handcrafted features. By coherently processing the entire response session of an L2 speaker, the model excels at predicting holistic oral proficiency. Experiments conducted on the Speak & Improve benchmark demonstrate that our proposed approach outperforms the previous state-of-the-art cascaded system and exhibits robust cross-part generalization, producing a compact deployable grader that is tailored for CALL applications. 

---
# AI Methods for Permutation Circuit Synthesis Across Generic Topologies 

**Authors**: Victor Villar, Juan Cruz-Benito, Ismael Faro, David Kremer  

**Link**: [PDF](https://arxiv.org/pdf/2509.16020)  

**Abstract**: This paper investigates artificial intelligence (AI) methodologies for the synthesis and transpilation of permutation circuits across generic topologies. Our approach uses Reinforcement Learning (RL) techniques to achieve near-optimal synthesis of permutation circuits up to 25 qubits. Rather than developing specialized models for individual topologies, we train a foundational model on a generic rectangular lattice, and employ masking mechanisms to dynamically select subsets of topologies during the synthesis. This enables the synthesis of permutation circuits on any topology that can be embedded within the rectangular lattice, without the need to re-train the model. In this paper we show results for 5x5 lattice and compare them to previous AI topology-oriented models and classical methods, showing that they outperform classical heuristics, and match previous specialized AI models, and performs synthesis even for topologies that were not seen during training. We further show that the model can be fine tuned to strengthen the performance for selected topologies of interest. This methodology allows a single trained model to efficiently synthesize circuits across diverse topologies, allowing its practical integration into transpilation workflows. 

---
# Fed-PISA: Federated Voice Cloning via Personalized Identity-Style Adaptation 

**Authors**: Qi Wang, Shituo Ma, Guoxin Yu, Hanyang Peng, Yue Yu  

**Link**: [PDF](https://arxiv.org/pdf/2509.16010)  

**Abstract**: Voice cloning for Text-to-Speech (TTS) aims to generate expressive and personalized speech from text using limited data from a target speaker. Federated Learning (FL) offers a collaborative and privacy-preserving framework for this task, but existing approaches suffer from high communication costs and tend to suppress stylistic heterogeneity, resulting in insufficient personalization. To address these issues, we propose Fed-PISA, which stands for Federated Personalized Identity-Style Adaptation. To minimize communication costs, Fed-PISA introduces a disentangled Low-Rank Adaptation (LoRA) mechanism: the speaker's timbre is retained locally through a private ID-LoRA, while only a lightweight style-LoRA is transmitted to the server, thereby minimizing parameter exchange. To harness heterogeneity, our aggregation method, inspired by collaborative filtering, is introduced to create custom models for each client by learning from stylistically similar peers. Experiments show that Fed-PISA improves style expressivity, naturalness, and speaker similarity, outperforming standard federated baselines with minimal communication costs. 

---
# Towards Sharper Object Boundaries in Self-Supervised Depth Estimation 

**Authors**: Aurélien Cecille, Stefan Duffner, Franck Davoine, Rémi Agier, Thibault Neveu  

**Link**: [PDF](https://arxiv.org/pdf/2509.15987)  

**Abstract**: Accurate monocular depth estimation is crucial for 3D scene understanding, but existing methods often blur depth at object boundaries, introducing spurious intermediate 3D points. While achieving sharp edges usually requires very fine-grained supervision, our method produces crisp depth discontinuities using only self-supervision. Specifically, we model per-pixel depth as a mixture distribution, capturing multiple plausible depths and shifting uncertainty from direct regression to the mixture weights. This formulation integrates seamlessly into existing pipelines via variance-aware loss functions and uncertainty propagation. Extensive evaluations on KITTI and VKITTIv2 show that our method achieves up to 35% higher boundary sharpness and improves point cloud quality compared to state-of-the-art baselines. 

---
# EmoHeal: An End-to-End System for Personalized Therapeutic Music Retrieval from Fine-grained Emotions 

**Authors**: Xinchen Wan, Jinhua Liang, Huan Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2509.15986)  

**Abstract**: Existing digital mental wellness tools often overlook the nuanced emotional states underlying everyday challenges. For example, pre-sleep anxiety affects more than 1.5 billion people worldwide, yet current approaches remain largely static and "one-size-fits-all", failing to adapt to individual needs. In this work, we present EmoHeal, an end-to-end system that delivers personalized, three-stage supportive narratives. EmoHeal detects 27 fine-grained emotions from user text with a fine-tuned XLM-RoBERTa model, mapping them to musical parameters via a knowledge graph grounded in music therapy principles (GEMS, iso-principle). EmoHeal retrieves audiovisual content using the CLAMP3 model to guide users from their current state toward a calmer one ("match-guide-target"). A within-subjects study (N=40) demonstrated significant supportive effects, with participants reporting substantial mood improvement (M=4.12, p<0.001) and high perceived emotion recognition accuracy (M=4.05, p<0.001). A strong correlation between perceived accuracy and therapeutic outcome (r=0.72, p<0.001) validates our fine-grained approach. These findings establish the viability of theory-driven, emotion-aware digital wellness tools and provides a scalable AI blueprint for operationalizing music therapy principles. 

---
# Uncertainty-Based Smooth Policy Regularisation for Reinforcement Learning with Few Demonstrations 

**Authors**: Yujie Zhu, Charles A. Hepburn, Matthew Thorpe, Giovanni Montana  

**Link**: [PDF](https://arxiv.org/pdf/2509.15981)  

**Abstract**: In reinforcement learning with sparse rewards, demonstrations can accelerate learning, but determining when to imitate them remains challenging. We propose Smooth Policy Regularisation from Demonstrations (SPReD), a framework that addresses the fundamental question: when should an agent imitate a demonstration versus follow its own policy? SPReD uses ensemble methods to explicitly model Q-value distributions for both demonstration and policy actions, quantifying uncertainty for comparisons. We develop two complementary uncertainty-aware methods: a probabilistic approach estimating the likelihood of demonstration superiority, and an advantage-based approach scaling imitation by statistical significance. Unlike prevailing methods (e.g. Q-filter) that make binary imitation decisions, SPReD applies continuous, uncertainty-proportional regularisation weights, reducing gradient variance during training. Despite its computational simplicity, SPReD achieves remarkable gains in experiments across eight robotics tasks, outperforming existing approaches by up to a factor of 14 in complex tasks while maintaining robustness to demonstration quality and quantity. Our code is available at this https URL. 

---
# Shedding Light on Depth: Explainability Assessment in Monocular Depth Estimation 

**Authors**: Lorenzo Cirillo, Claudio Schiavella, Lorenzo Papa, Paolo Russo, Irene Amerini  

**Link**: [PDF](https://arxiv.org/pdf/2509.15980)  

**Abstract**: Explainable artificial intelligence is increasingly employed to understand the decision-making process of deep learning models and create trustworthiness in their adoption. However, the explainability of Monocular Depth Estimation (MDE) remains largely unexplored despite its wide deployment in real-world applications. In this work, we study how to analyze MDE networks to map the input image to the predicted depth map. More in detail, we investigate well-established feature attribution methods, Saliency Maps, Integrated Gradients, and Attention Rollout on different computationally complex models for MDE: METER, a lightweight network, and PixelFormer, a deep network. We assess the quality of the generated visual explanations by selectively perturbing the most relevant and irrelevant pixels, as identified by the explainability methods, and analyzing the impact of these perturbations on the model's output. Moreover, since existing evaluation metrics can have some limitations in measuring the validity of visual explanations for MDE, we additionally introduce the Attribution Fidelity. This metric evaluates the reliability of the feature attribution by assessing their consistency with the predicted depth map. Experimental results demonstrate that Saliency Maps and Integrated Gradients have good performance in highlighting the most important input features for MDE lightweight and deep models, respectively. Furthermore, we show that Attribution Fidelity effectively identifies whether an explainability method fails to produce reliable visual maps, even in scenarios where conventional metrics might suggest satisfactory results. 

---
# BEFT: Bias-Efficient Fine-Tuning of Language Models 

**Authors**: Baichuan Huang, Ananth Balashankar, Amir Aminifar  

**Link**: [PDF](https://arxiv.org/pdf/2509.15974)  

**Abstract**: Fine-tuning all-bias-terms stands out among various parameter-efficient fine-tuning (PEFT) techniques, owing to its out-of-the-box usability and competitive performance, especially in low-data regimes. Bias-only fine-tuning has the potential for unprecedented parameter efficiency. However, the link between fine-tuning different bias terms (i.e., bias terms in the query, key, or value projections) and downstream performance remains unclear. The existing approaches, e.g., based on the magnitude of bias change or empirical Fisher information, provide limited guidance for selecting the particular bias term for effective fine-tuning. In this paper, we propose an approach for selecting the bias term to be fine-tuned, forming the foundation of our bias-efficient fine-tuning (BEFT). We extensively evaluate our bias-efficient approach against other bias-selection approaches, across a wide range of large language models (LLMs) spanning encoder-only and decoder-only architectures from 110M to 6.7B parameters. Our results demonstrate the effectiveness and superiority of our bias-efficient approach on diverse downstream tasks, including classification, multiple-choice, and generation tasks. 

---
# RLinf: Flexible and Efficient Large-scale Reinforcement Learning via Macro-to-Micro Flow Transformation 

**Authors**: Chao Yu, Yuanqing Wang, Zhen Guo, Hao Lin, Si Xu, Hongzhi Zang, Quanlu Zhang, Yongji Wu, Chunyang Zhu, Junhao Hu, Zixiao Huang, Mingjie Wei, Yuqing Xie, Ke Yang, Bo Dai, Zhexuan Xu, Xiangyuan Wang, Xu Fu, Zhihao Liu, Kang Chen, Weilin Liu, Gang Liu, Boxun Li, Jianlei Yang, Zhi Yang, Guohao Dai, Yu Wang  

**Link**: [PDF](https://arxiv.org/pdf/2509.15965)  

**Abstract**: Reinforcement learning (RL) has demonstrated immense potential in advancing artificial general intelligence, agentic intelligence, and embodied intelligence. However, the inherent heterogeneity and dynamicity of RL workflows often lead to low hardware utilization and slow training on existing systems. In this paper, we present RLinf, a high-performance RL training system based on our key observation that the major roadblock to efficient RL training lies in system flexibility. To maximize flexibility and efficiency, RLinf is built atop a novel RL system design paradigm called macro-to-micro flow transformation (M2Flow), which automatically breaks down high-level, easy-to-compose RL workflows at both the temporal and spatial dimensions, and recomposes them into optimized execution flows. Supported by RLinf worker's adaptive communication capability, we devise context switching and elastic pipelining to realize M2Flow transformation, and a profiling-guided scheduling policy to generate optimal execution plans. Extensive evaluations on both reasoning RL and embodied RL tasks demonstrate that RLinf consistently outperforms state-of-the-art systems, achieving 1.1x-2.13x speedup in end-to-end training throughput. 

---
# MoE-CE: Enhancing Generalization for Deep Learning based Channel Estimation via a Mixture-of-Experts Framework 

**Authors**: Tianyu Li, Yan Xin, Jianzhong, Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2509.15964)  

**Abstract**: Reliable channel estimation (CE) is fundamental for robust communication in dynamic wireless environments, where models must generalize across varying conditions such as signal-to-noise ratios (SNRs), the number of resource blocks (RBs), and channel profiles. Traditional deep learning (DL)-based methods struggle to generalize effectively across such diverse settings, particularly under multitask and zero-shot scenarios. In this work, we propose MoE-CE, a flexible mixture-of-experts (MoE) framework designed to enhance the generalization capability of DL-based CE methods. MoE-CE provides an appropriate inductive bias by leveraging multiple expert subnetworks, each specialized in distinct channel characteristics, and a learned router that dynamically selects the most relevant experts per input. This architecture enhances model capacity and adaptability without a proportional rise in computational cost while being agnostic to the choice of the backbone model and the learning algorithm. Through extensive experiments on synthetic datasets generated under diverse SNRs, RB numbers, and channel profiles, including multitask and zero-shot evaluations, we demonstrate that MoE-CE consistently outperforms conventional DL approaches, achieving significant performance gains while maintaining efficiency. 

---
# Explainable AI for Maritime Autonomous Surface Ships (MASS): Adaptive Interfaces and Trustworthy Human-AI Collaboration 

**Authors**: Zhuoyue Zhang, Haitong Xu  

**Link**: [PDF](https://arxiv.org/pdf/2509.15959)  

**Abstract**: Autonomous navigation in maritime domains is accelerating alongside advances in artificial intelligence, sensing, and connectivity. Opaque decision-making and poorly calibrated human-automation interaction remain key barriers to safe adoption. This article synthesizes 100 studies on automation transparency for Maritime Autonomous Surface Ships (MASS) spanning situation awareness (SA), human factors, interface design, and regulation. We (i) map the Guidance-Navigation-Control stack to shore-based operational modes -- remote supervision (RSM) and remote control (RCM) -- and identify where human unsafe control actions (Human-UCAs) concentrate in handover and emergency loops; (ii) summarize evidence that transparency features (decision rationales, alternatives, confidence/uncertainty, and rule-compliance indicators) improve understanding and support trust calibration, though reliability and predictability often dominate trust; (iii) distill design strategies for transparency at three layers: sensor/SA acquisition and fusion, HMI/eHMI presentation (textual/graphical overlays, color coding, conversational and immersive UIs), and engineer-facing processes (resilient interaction design, validation, and standardization). We integrate methods for Human-UCA identification (STPA-Cog + IDAC), quantitative trust/SA assessment, and operator workload monitoring, and outline regulatory and rule-based implications including COLREGs formalization and route exchange. We conclude with an adaptive transparency framework that couples operator state estimation with explainable decision support to reduce cognitive overload and improve takeover timeliness. The review highlights actionable figure-of-merit displays (e.g., CPA/TCPA risk bars, robustness heatmaps), transparent model outputs (rule traceability, confidence), and training pipelines (HIL/MIL, simulation) as near-term levers for safer MASS operations. 

---
# Compose Yourself: Average-Velocity Flow Matching for One-Step Speech Enhancement 

**Authors**: Gang Yang, Yue Lei, Wenxin Tai, Jin Wu, Jia Chen, Ting Zhong, Fan Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2509.15952)  

**Abstract**: Diffusion and flow matching (FM) models have achieved remarkable progress in speech enhancement (SE), yet their dependence on multi-step generation is computationally expensive and vulnerable to discretization errors. Recent advances in one-step generative modeling, particularly MeanFlow, provide a promising alternative by reformulating dynamics through average velocity fields. In this work, we present COSE, a one-step FM framework tailored for SE. To address the high training overhead of Jacobian-vector product (JVP) computations in MeanFlow, we introduce a velocity composition identity to compute average velocity efficiently, eliminating expensive computation while preserving theoretical consistency and achieving competitive enhancement quality. Extensive experiments on standard benchmarks show that COSE delivers up to 5x faster sampling and reduces training cost by 40%, all without compromising speech quality. Code is available at this https URL. 

---
# ArchesClimate: Probabilistic Decadal Ensemble Generation With Flow Matching 

**Authors**: Graham Clyne, Guillaume Couairon, Guillaume Gastineau, Claire Monteleoni, Anastase Charantonis  

**Link**: [PDF](https://arxiv.org/pdf/2509.15942)  

**Abstract**: Climate projections have uncertainties related to components of the climate system and their interactions. A typical approach to quantifying these uncertainties is to use climate models to create ensembles of repeated simulations under different initial conditions. Due to the complexity of these simulations, generating such ensembles of projections is computationally expensive. In this work, we present ArchesClimate, a deep learning-based climate model emulator that aims to reduce this cost. ArchesClimate is trained on decadal hindcasts of the IPSL-CM6A-LR climate model at a spatial resolution of approximately 2.5x1.25 degrees. We train a flow matching model following ArchesWeatherGen, which we adapt to predict near-term climate. Once trained, the model generates states at a one-month lead time and can be used to auto-regressively emulate climate model simulations of any length. We show that for up to 10 years, these generations are stable and physically consistent. We also show that for several important climate variables, ArchesClimate generates simulations that are interchangeable with the IPSL model. This work suggests that climate model emulators could significantly reduce the cost of climate model simulations. 

---
# A Vision-Language-Action-Critic Model for Robotic Real-World Reinforcement Learning 

**Authors**: Shaopeng Zhai, Qi Zhang, Tianyi Zhang, Fuxian Huang, Haoran Zhang, Ming Zhou, Shengzhe Zhang, Litao Liu, Sixu Lin, Jiangmiao Pang  

**Link**: [PDF](https://arxiv.org/pdf/2509.15937)  

**Abstract**: Robotic real-world reinforcement learning (RL) with vision-language-action (VLA) models is bottlenecked by sparse, handcrafted rewards and inefficient exploration. We introduce VLAC, a general process reward model built upon InternVL and trained on large scale heterogeneous datasets. Given pairwise observations and a language goal, it outputs dense progress delta and done signal, eliminating task-specific reward engineering, and supports one-shot in-context transfer to unseen tasks and environments. VLAC is trained on vision-language datasets to strengthen perception, dialogic and reasoning capabilities, together with robot and human trajectories data that ground action generation and progress estimation, and additionally strengthened to reject irrelevant prompts as well as detect regression or stagnation by constructing large numbers of negative and semantically mismatched samples. With prompt control, a single VLAC model alternately generating reward and action tokens, unifying critic and policy. Deployed inside an asynchronous real-world RL loop, we layer a graded human-in-the-loop protocol (offline demonstration replay, return and explore, human guided explore) that accelerates exploration and stabilizes early learning. Across four distinct real-world manipulation tasks, VLAC lifts success rates from about 30\% to about 90\% within 200 real-world interaction episodes; incorporating human-in-the-loop interventions yields a further 50% improvement in sample efficiency and achieves up to 100% final success. 

---
# The Alignment Bottleneck 

**Authors**: Wenjun Cao  

**Link**: [PDF](https://arxiv.org/pdf/2509.15932)  

**Abstract**: Large language models improve with scale, yet feedback-based alignment still exhibits systematic deviations from intended behavior. Motivated by bounded rationality in economics and cognitive science, we view judgment as resource-limited and feedback as a constrained channel. On this basis, we model the loop as a two-stage cascade $U \to H \to Y$ given $S$, with cognitive capacity $C_{\text{cog}|S}$ and average total capacity $\bar{C}_{\text{tot}|S}$. Our main result is a capacity-coupled Alignment Performance Interval. It pairs a data size-independent Fano lower bound proved on a separable codebook mixture with a PAC-Bayes upper bound whose KL term is controlled by the same channel via $m \, \bar{C}_{\text{tot}|S}$. The PAC-Bayes bound becomes an upper bound on the same true risk when the canonical observable loss is used and the dataset is drawn from the same mixture. Under these matched conditions, both limits are governed by a single capacity. Consequences include that, with value complexity and capacity fixed, adding labels alone cannot cross the bound; attaining lower risk on more complex targets requires capacity that grows with $\log M$; and once useful signal saturates capacity, further optimization tends to fit channel regularities, consistent with reports of sycophancy and reward hacking. The analysis views alignment as interface engineering: measure and allocate limited capacity, manage task complexity, and decide where information is spent. 

---
# Enhancing Generative Auto-bidding with Offline Reward Evaluation and Policy Search 

**Authors**: Zhiyu Mou, Yiqin Lv, Miao Xu, Cheems Wang, Yixiu Mao, Qichen Ye, Chao Li, Rongquan Bai, Chuan Yu, Jian Xu, Bo Zheng  

**Link**: [PDF](https://arxiv.org/pdf/2509.15927)  

**Abstract**: Auto-bidding is an essential tool for advertisers to enhance their advertising performance. Recent progress has shown that AI-Generated Bidding (AIGB), which formulates the auto-bidding as a trajectory generation task and trains a conditional diffusion-based planner on offline data, achieves superior and stable performance compared to typical offline reinforcement learning (RL)-based auto-bidding methods. However, existing AIGB methods still encounter a performance bottleneck due to their neglect of fine-grained generation quality evaluation and inability to explore beyond static datasets. To address this, we propose AIGB-Pearl (\emph{Planning with EvAluator via RL}), a novel method that integrates generative planning and policy optimization. The key to AIGB-Pearl is to construct a non-bootstrapped \emph{trajectory evaluator} to assign rewards and guide policy search, enabling the planner to optimize its generation quality iteratively through interaction. Furthermore, to enhance trajectory evaluator accuracy in offline settings, we incorporate three key techniques: (i) a Large Language Model (LLM)-based architecture for better representational capacity, (ii) hybrid point-wise and pair-wise losses for better score learning, and (iii) adaptive integration of expert feedback for better generalization ability. Extensive experiments on both simulated and real-world advertising systems demonstrate the state-of-the-art performance of our approach. 

---
# Foundation Models as World Models: A Foundational Study in Text-Based GridWorlds 

**Authors**: Remo Sasso, Michelangelo Conserva, Dominik Jeurissen, Paulo Rauber  

**Link**: [PDF](https://arxiv.org/pdf/2509.15915)  

**Abstract**: While reinforcement learning from scratch has shown impressive results in solving sequential decision-making tasks with efficient simulators, real-world applications with expensive interactions require more sample-efficient agents. Foundation models (FMs) are natural candidates to improve sample efficiency as they possess broad knowledge and reasoning capabilities, but it is yet unclear how to effectively integrate them into the reinforcement learning framework. In this paper, we anticipate and, most importantly, evaluate two promising strategies. First, we consider the use of foundation world models (FWMs) that exploit the prior knowledge of FMs to enable training and evaluating agents with simulated interactions. Second, we consider the use of foundation agents (FAs) that exploit the reasoning capabilities of FMs for decision-making. We evaluate both approaches empirically in a family of grid-world environments that are suitable for the current generation of large language models (LLMs). Our results suggest that improvements in LLMs already translate into better FWMs and FAs; that FAs based on current LLMs can already provide excellent policies for sufficiently simple environments; and that the coupling of FWMs and reinforcement learning agents is highly promising for more complex settings with partial observability and stochastic elements. 

---
# An Equivariant Graph Network for Interpretable Nanoporous Materials Design 

**Authors**: Zhenhao Zhou, Salman Bin Kashif, Dawei Feng, Jin-Hu Dou, Kaihang Shi, Tao Deng, Zhenpeng Yao  

**Link**: [PDF](https://arxiv.org/pdf/2509.15908)  

**Abstract**: Nanoporous materials hold promise for diverse sustainable applications, yet their vast chemical space poses challenges for efficient design. Machine learning offers a compelling pathway to accelerate the exploration, but existing models lack either interpretability or fidelity for elucidating the correlation between crystal geometry and property. Here, we report a three-dimensional periodic space sampling method that decomposes large nanoporous structures into local geometrical sites for combined property prediction and site-wise contribution quantification. Trained with a constructed database and retrieved datasets, our model achieves state-of-the-art accuracy and data efficiency for property prediction on gas storage, separation, and electrical conduction. Meanwhile, this approach enables the interpretation of the prediction and allows for accurate identification of significant local sites for targeted properties. Through identifying transferable high-performance sites across diverse nanoporous frameworks, our model paves the way for interpretable, symmetry-aware nanoporous materials design, which is extensible to other materials, like molecular crystals and beyond. 

---
# Re-FRAME the Meeting Summarization SCOPE: Fact-Based Summarization and Personalization via Questions 

**Authors**: Frederic Kirstein, Sonu Kumar, Terry Ruas, Bela Gipp  

**Link**: [PDF](https://arxiv.org/pdf/2509.15901)  

**Abstract**: Meeting summarization with large language models (LLMs) remains error-prone, often producing outputs with hallucinations, omissions, and irrelevancies. We present FRAME, a modular pipeline that reframes summarization as a semantic enrichment task. FRAME extracts and scores salient facts, organizes them thematically, and uses these to enrich an outline into an abstractive summary. To personalize summaries, we introduce SCOPE, a reason-out-loud protocol that has the model build a reasoning trace by answering nine questions before content selection. For evaluation, we propose P-MESA, a multi-dimensional, reference-free evaluation framework to assess if a summary fits a target reader. P-MESA reliably identifies error instances, achieving >= 89% balanced accuracy against human annotations and strongly aligns with human severity ratings (r >= 0.70). On QMSum and FAME, FRAME reduces hallucination and omission by 2 out of 5 points (measured with MESA), while SCOPE improves knowledge fit and goal alignment over prompt-only baselines. Our findings advocate for rethinking summarization to improve control, faithfulness, and personalization. 

---
# From Data to Diagnosis: A Large, Comprehensive Bone Marrow Dataset and AI Methods for Childhood Leukemia Prediction 

**Authors**: Henning Höfener, Farina Kock, Martina Pontones, Tabita Ghete, David Pfrang, Nicholas Dickel, Meik Kunz, Daniela P. Schacherer, David A. Clunie, Andrey Fedorov, Max Westphal, Markus Metzler  

**Link**: [PDF](https://arxiv.org/pdf/2509.15895)  

**Abstract**: Leukemia diagnosis primarily relies on manual microscopic analysis of bone marrow morphology supported by additional laboratory parameters, making it complex and time consuming. While artificial intelligence (AI) solutions have been proposed, most utilize private datasets and only cover parts of the diagnostic pipeline. Therefore, we present a large, high-quality, publicly available leukemia bone marrow dataset spanning the entire diagnostic process, from cell detection to diagnosis. Using this dataset, we further propose methods for cell detection, cell classification, and diagnosis prediction. The dataset comprises 246 pediatric patients with diagnostic, clinical and laboratory information, over 40 000 cells with bounding box annotations and more than 28 000 of these with high-quality class labels, making it the most comprehensive dataset publicly available. Evaluation of the AI models yielded an average precision of 0.96 for the cell detection, an area under the curve of 0.98, and an F1-score of 0.61 for the 33-class cell classification, and a mean F1-score of 0.90 for the diagnosis prediction using predicted cell counts. While the proposed approaches demonstrate their usefulness for AI-assisted diagnostics, the dataset will foster further research and development in the field, ultimately contributing to more precise diagnoses and improved patient outcomes. 

---
# MoAngelo: Motion-Aware Neural Surface Reconstruction for Dynamic Scenes 

**Authors**: Mohamed Ebbed, Zorah Lähner  

**Link**: [PDF](https://arxiv.org/pdf/2509.15892)  

**Abstract**: Dynamic scene reconstruction from multi-view videos remains a fundamental challenge in computer vision. While recent neural surface reconstruction methods have achieved remarkable results in static 3D reconstruction, extending these approaches with comparable quality for dynamic scenes introduces significant computational and representational challenges. Existing dynamic methods focus on novel-view synthesis, therefore, their extracted meshes tend to be noisy. Even approaches aiming for geometric fidelity often result in too smooth meshes due to the ill-posedness of the problem. We present a novel framework for highly detailed dynamic reconstruction that extends the static 3D reconstruction method NeuralAngelo to work in dynamic settings. To that end, we start with a high-quality template scene reconstruction from the initial frame using NeuralAngelo, and then jointly optimize deformation fields that track the template and refine it based on the temporal sequence. This flexible template allows updating the geometry to include changes that cannot be modeled with the deformation field, for instance occluded parts or the changes in the topology. We show superior reconstruction accuracy in comparison to previous state-of-the-art methods on the ActorsHQ dataset. 

---
# Distribution-Aligned Decoding for Efficient LLM Task Adaptation 

**Authors**: Senkang Hu, Xudong Han, Jinqi Jiang, Yihang Tao, Zihan Fang, Sam Tak Wu Kwong, Yuguang Fang  

**Link**: [PDF](https://arxiv.org/pdf/2509.15888)  

**Abstract**: Adapting billion-parameter language models to a downstream task is still costly, even with parameter-efficient fine-tuning (PEFT). We re-cast task adaptation as output-distribution alignment: the objective is to steer the output distribution toward the task distribution directly during decoding rather than indirectly through weight updates. Building on this view, we introduce Steering Vector Decoding (SVD), a lightweight, PEFT-compatible, and theoretically grounded method. We start with a short warm-start fine-tune and extract a task-aware steering vector from the Kullback-Leibler (KL) divergence gradient between the output distribution of the warm-started and pre-trained models. This steering vector is then used to guide the decoding process to steer the model's output distribution towards the task distribution. We theoretically prove that SVD is first-order equivalent to the gradient step of full fine-tuning and derive a globally optimal solution for the strength of the steering vector. Across three tasks and nine benchmarks, SVD paired with four standard PEFT methods improves multiple-choice accuracy by up to 5 points and open-ended truthfulness by 2 points, with similar gains (1-2 points) on commonsense datasets without adding trainable parameters beyond the PEFT adapter. SVD thus offers a lightweight, theoretically grounded path to stronger task adaptation for large language models. 

---
# RACap: Relation-Aware Prompting for Lightweight Retrieval-Augmented Image Captioning 

**Authors**: Xiaosheng Long, Hanyu Wang, Zhentao Song, Kun Luo, Hongde Liu  

**Link**: [PDF](https://arxiv.org/pdf/2509.15883)  

**Abstract**: Recent retrieval-augmented image captioning methods incorporate external knowledge to compensate for the limitations in comprehending complex scenes. However, current approaches face challenges in relation modeling: (1) the representation of semantic prompts is too coarse-grained to capture fine-grained relationships; (2) these methods lack explicit modeling of image objects and their semantic relationships. To address these limitations, we propose RACap, a relation-aware retrieval-augmented model for image captioning, which not only mines structured relation semantics from retrieval captions, but also identifies heterogeneous objects from the image. RACap effectively retrieves structured relation features that contain heterogeneous visual information to enhance the semantic consistency and relational expressiveness. Experimental results show that RACap, with only 10.8M trainable parameters, achieves superior performance compared to previous lightweight captioning models. 

---
# Self-Supervised Cross-Modal Learning for Image-to-Point Cloud Registration 

**Authors**: Xingmei Wang, Xiaoyu Hu, Chengkai Huang, Ziyan Zeng, Guohao Nie, Quan Z. Sheng, Lina Yao  

**Link**: [PDF](https://arxiv.org/pdf/2509.15882)  

**Abstract**: Bridging 2D and 3D sensor modalities is critical for robust perception in autonomous systems. However, image-to-point cloud (I2P) registration remains challenging due to the semantic-geometric gap between texture-rich but depth-ambiguous images and sparse yet metrically precise point clouds, as well as the tendency of existing methods to converge to local optima. To overcome these limitations, we introduce CrossI2P, a self-supervised framework that unifies cross-modal learning and two-stage registration in a single end-to-end pipeline. First, we learn a geometric-semantic fused embedding space via dual-path contrastive learning, enabling annotation-free, bidirectional alignment of 2D textures and 3D structures. Second, we adopt a coarse-to-fine registration paradigm: a global stage establishes superpoint-superpixel correspondences through joint intra-modal context and cross-modal interaction modeling, followed by a geometry-constrained point-level refinement for precise registration. Third, we employ a dynamic training mechanism with gradient normalization to balance losses for feature alignment, correspondence refinement, and pose estimation. Extensive experiments demonstrate that CrossI2P outperforms state-of-the-art methods by 23.7% on the KITTI Odometry benchmark and by 37.9% on nuScenes, significantly improving both accuracy and robustness. 

---
# DeepMech: A Machine Learning Framework for Chemical Reaction Mechanism Prediction 

**Authors**: Manajit Das, Ajnabiul Hoque, Mayank Baranwal, Raghavan B. Sunoj  

**Link**: [PDF](https://arxiv.org/pdf/2509.15872)  

**Abstract**: Prediction of complete step-by-step chemical reaction mechanisms (CRMs) remains a major challenge. Whereas the traditional approaches in CRM tasks rely on expert-driven experiments or costly quantum chemical computations, contemporary deep learning (DL) alternatives ignore key intermediates and mechanistic steps and often suffer from hallucinations. We present DeepMech, an interpretable graph-based DL framework employing atom- and bond-level attention, guided by generalized templates of mechanistic operations (TMOps), to generate CRMs. Trained on our curated ReactMech dataset (~30K CRMs with 100K atom-mapped and mass-balanced elementary steps), DeepMech achieves 98.98+/-0.12% accuracy in predicting elementary steps and 95.94+/-0.21% in complete CRM tasks, besides maintaining high fidelity even in out-of-distribution scenarios as well as in predicting side and/or byproducts. Extension to multistep CRMs relevant to prebiotic chemistry, demonstrates the ability of DeepMech in effectively reconstructing pathways from simple primordial substrates to complex biomolecules such as serine and aldopentose. Attention analysis identifies reactive atoms/bonds in line with chemical intuition, rendering our model interpretable and suitable for reaction design. 

---
# EvoBrain: Dynamic Multi-channel EEG Graph Modeling for Time-evolving Brain Network 

**Authors**: Rikuto Kotoge, Zheng Chen, Tasuku Kimura, Yasuko Matsubara, Takufumi Yanagisawa, Haruhiko Kishima, Yasushi Sakurai  

**Link**: [PDF](https://arxiv.org/pdf/2509.15857)  

**Abstract**: Dynamic GNNs, which integrate temporal and spatial features in Electroencephalography (EEG) data, have shown great potential in automating seizure detection. However, fully capturing the underlying dynamics necessary to represent brain states, such as seizure and non-seizure, remains a non-trivial task and presents two fundamental challenges. First, most existing dynamic GNN methods are built on temporally fixed static graphs, which fail to reflect the evolving nature of brain connectivity during seizure progression. Second, current efforts to jointly model temporal signals and graph structures and, more importantly, their interactions remain nascent, often resulting in inconsistent performance. To address these challenges, we present the first theoretical analysis of these two problems, demonstrating the effectiveness and necessity of explicit dynamic modeling and time-then-graph dynamic GNN method. Building on these insights, we propose EvoBrain, a novel seizure detection model that integrates a two-stream Mamba architecture with a GCN enhanced by Laplacian Positional Encoding, following neurological insights. Moreover, EvoBrain incorporates explicitly dynamic graph structures, allowing both nodes and edges to evolve over time. Our contributions include (a) a theoretical analysis proving the expressivity advantage of explicit dynamic modeling and time-then-graph over other approaches, (b) a novel and efficient model that significantly improves AUROC by 23% and F1 score by 30%, compared with the dynamic GNN baseline, and (c) broad evaluations of our method on the challenging early seizure prediction tasks. 

---
# Diversity of Structured Domains via k-Kemeny Scores 

**Authors**: Piotr Faliszewski, Krzysztof Sornat, Stanisław Szufa, Tomasz Wąs  

**Link**: [PDF](https://arxiv.org/pdf/2509.15812)  

**Abstract**: In the k-Kemeny problem, we are given an ordinal election, i.e., a collection of votes ranking the candidates from best to worst, and we seek the smallest number of swaps of adjacent candidates that ensure that the election has at most k different rankings. We study this problem for a number of structured domains, including the single-peaked, single-crossing, group-separable, and Euclidean ones. We obtain two kinds of results: (1) We show that k-Kemeny remains intractable under most of these domains, even for k=2, and (2) we use k-Kemeny to rank these domains in terms of their diversity. 

---
# Best-of-L: Cross-Lingual Reward Modeling for Mathematical Reasoning 

**Authors**: Sara Rajaee, Rochelle Choenni, Ekaterina Shutova, Christof Monz  

**Link**: [PDF](https://arxiv.org/pdf/2509.15811)  

**Abstract**: While the reasoning abilities of large language models (LLMs) continue to advance, it remains unclear how such ability varies across languages in multilingual LLMs and whether different languages produce reasoning paths that complement each other. To investigate this question, we train a reward model to rank generated responses for a given question across languages. Our results show that our cross-lingual reward model substantially improves mathematical reasoning performance compared to using reward modeling within a single language, benefiting even high-resource languages. While English often exhibits the highest performance in multilingual models, we find that cross-lingual sampling particularly benefits English under low sampling budgets. Our findings reveal new opportunities to improve multilingual reasoning by leveraging the complementary strengths of diverse languages. 

---
# Instance Generation for Meta-Black-Box Optimization through Latent Space Reverse Engineering 

**Authors**: Chen Wang, Zeyuan Ma, Zhiguang Cao, Yue-Jiao Gong  

**Link**: [PDF](https://arxiv.org/pdf/2509.15810)  

**Abstract**: To relieve intensive human-expertise required to design optimization algorithms, recent Meta-Black-Box Optimization (MetaBBO) researches leverage generalization strength of meta-learning to train neural network-based algorithm design policies over a predefined training problem set, which automates the adaptability of the low-level optimizers on unseen problem instances. Currently, a common training problem set choice in existing MetaBBOs is well-known benchmark suites CoCo-BBOB. Although such choice facilitates the MetaBBO's development, problem instances in CoCo-BBOB are more or less limited in diversity, raising the risk of overfitting of MetaBBOs, which might further results in poor generalization. In this paper, we propose an instance generation approach, termed as \textbf{LSRE}, which could generate diverse training problem instances for MetaBBOs to learn more generalizable policies. LSRE first trains an autoencoder which maps high-dimensional problem features into a 2-dimensional latent space. Uniform-grid sampling in this latent space leads to hidden representations of problem instances with sufficient diversity. By leveraging a genetic-programming approach to search function formulas with minimal L2-distance to these hidden representations, LSRE reverse engineers a diversified problem set, termed as \textbf{Diverse-BBO}. We validate the effectiveness of LSRE by training various MetaBBOs on Diverse-BBO and observe their generalization performances on either synthetic or realistic scenarios. Extensive experimental results underscore the superiority of Diverse-BBO to existing training set choices in MetaBBOs. Further ablation studies not only demonstrate the effectiveness of design choices in LSRE, but also reveal interesting insights on instance diversity and MetaBBO's generalization. 

---
# CIDER: A Causal Cure for Brand-Obsessed Text-to-Image Models 

**Authors**: Fangjian Shen, Zifeng Liang, Chao Wang, Wushao Wen  

**Link**: [PDF](https://arxiv.org/pdf/2509.15803)  

**Abstract**: Text-to-image (T2I) models exhibit a significant yet under-explored "brand bias", a tendency to generate contents featuring dominant commercial brands from generic prompts, posing ethical and legal risks. We propose CIDER, a novel, model-agnostic framework to mitigate bias at inference-time through prompt refinement to avoid costly retraining. CIDER uses a lightweight detector to identify branded content and a Vision-Language Model (VLM) to generate stylistically divergent alternatives. We introduce the Brand Neutrality Score (BNS) to quantify this issue and perform extensive experiments on leading T2I models. Results show CIDER significantly reduces both explicit and implicit biases while maintaining image quality and aesthetic appeal. Our work offers a practical solution for more original and equitable content, contributing to the development of trustworthy generative AI. 

---
# ChronoForge-RL: Chronological Forging through Reinforcement Learning for Enhanced Video Understanding 

**Authors**: Kehua Chen  

**Link**: [PDF](https://arxiv.org/pdf/2509.15800)  

**Abstract**: Current state-of-the-art video understanding methods typically struggle with two critical challenges: (1) the computational infeasibility of processing every frame in dense video content and (2) the difficulty in identifying semantically significant frames through naive uniform sampling strategies. In this paper, we propose a novel video understanding framework, called ChronoForge-RL, which combines Temporal Apex Distillation (TAD) and KeyFrame-aware Group Relative Policy Optimization (KF-GRPO) to tackle these issues. Concretely, we introduce a differentiable keyframe selection mechanism that systematically identifies semantic inflection points through a three-stage process to enhance computational efficiency while preserving temporal information. Then, two particular modules are proposed to enable effective temporal reasoning: Firstly, TAD leverages variation scoring, inflection detection, and prioritized distillation to select the most informative frames. Secondly, we introduce KF-GRPO which implements a contrastive learning paradigm with a saliency-enhanced reward mechanism that explicitly incentivizes models to leverage both frame content and temporal relationships. Finally, our proposed ChronoForge-RL achieves 69.1% on VideoMME and 52.7% on LVBench compared to baseline methods, clearly surpassing previous approaches while enabling our 7B parameter model to achieve performance comparable to 72B parameter alternatives. 

---
# Hierarchical Reinforcement Learning with Low-Level MPC for Multi-Agent Control 

**Authors**: Max Studt, Georg Schildbach  

**Link**: [PDF](https://arxiv.org/pdf/2509.15799)  

**Abstract**: Achieving safe and coordinated behavior in dynamic, constraint-rich environments remains a major challenge for learning-based control. Pure end-to-end learning often suffers from poor sample efficiency and limited reliability, while model-based methods depend on predefined references and struggle to generalize. We propose a hierarchical framework that combines tactical decision-making via reinforcement learning (RL) with low-level execution through Model Predictive Control (MPC). For the case of multi-agent systems this means that high-level policies select abstract targets from structured regions of interest (ROIs), while MPC ensures dynamically feasible and safe motion. Tested on a predator-prey benchmark, our approach outperforms end-to-end and shielding-based RL baselines in terms of reward, safety, and consistency, underscoring the benefits of combining structured learning with model-based control. 

---
# Monte Carlo Tree Diffusion with Multiple Experts for Protein Design 

**Authors**: Xuefeng Liu, Mingxuan Cao, Songhao Jiang, Xiao Luo, Xiaotian Duan, Mengdi Wang, Tobin R. Sosnick, Jinbo Xu, Rick Stevens  

**Link**: [PDF](https://arxiv.org/pdf/2509.15796)  

**Abstract**: The goal of protein design is to generate amino acid sequences that fold into functional structures with desired properties. Prior methods combining autoregressive language models with Monte Carlo Tree Search (MCTS) struggle with long-range dependencies and suffer from an impractically large search space. We propose MCTD-ME, Monte Carlo Tree Diffusion with Multiple Experts, which integrates masked diffusion models with tree search to enable multi-token planning and efficient exploration. Unlike autoregressive planners, MCTD-ME uses biophysical-fidelity-enhanced diffusion denoising as the rollout engine, jointly revising multiple positions and scaling to large sequence spaces. It further leverages experts of varying capacities to enrich exploration, guided by a pLDDT-based masking schedule that targets low-confidence regions while preserving reliable residues. We propose a novel multi-expert selection rule (PH-UCT-ME) extends predictive-entropy UCT to expert ensembles. On the inverse folding task (CAMEO and PDB benchmarks), MCTD-ME outperforms single-expert and unguided baselines in both sequence recovery (AAR) and structural similarity (scTM), with gains increasing for longer proteins and benefiting from multi-expert guidance. More generally, the framework is model-agnostic and applicable beyond inverse folding, including de novo protein engineering and multi-objective molecular generation. 

---
# CBPNet: A Continual Backpropagation Prompt Network for Alleviating Plasticity Loss on Edge Devices 

**Authors**: Runjie Shao, Boyu Diao, Zijia An, Ruiqi Liu, Yongjun Xu  

**Link**: [PDF](https://arxiv.org/pdf/2509.15785)  

**Abstract**: To meet the demands of applications like robotics and autonomous driving that require real-time responses to dynamic environments, efficient continual learning methods suitable for edge devices have attracted increasing attention. In this transition, using frozen pretrained models with prompts has become a mainstream strategy to combat catastrophic forgetting. However, this approach introduces a new critical bottleneck: plasticity loss, where the model's ability to learn new knowledge diminishes due to the frozen backbone and the limited capacity of prompt parameters. We argue that the reduction in plasticity stems from a lack of update vitality in underutilized parameters during the training process. To this end, we propose the Continual Backpropagation Prompt Network (CBPNet), an effective and parameter efficient framework designed to restore the model's learning vitality. We innovatively integrate an Efficient CBP Block that counteracts plasticity decay by adaptively reinitializing these underutilized parameters. Experimental results on edge devices demonstrate CBPNet's effectiveness across multiple benchmarks. On Split CIFAR-100, it improves average accuracy by over 1% against a strong baseline, and on the more challenging Split ImageNet-R, it achieves a state of the art accuracy of 69.41%. This is accomplished by training additional parameters that constitute less than 0.2% of the backbone's size, validating our approach. 

---
# Ideal Registration? Segmentation is All You Need 

**Authors**: Xiang Chen, Fengting Zhang, Qinghao Liu, Min Liu, Kun Wu, Yaonan Wang, Hang Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2509.15784)  

**Abstract**: Deep learning has revolutionized image registration by its ability to handle diverse tasks while achieving significant speed advantages over conventional approaches. Current approaches, however, often employ globally uniform smoothness constraints that fail to accommodate the complex, regionally varying deformations characteristic of anatomical motion. To address this limitation, we propose SegReg, a Segmentation-driven Registration framework that implements anatomically adaptive regularization by exploiting region-specific deformation patterns. Our SegReg first decomposes input moving and fixed images into anatomically coherent subregions through segmentation. These localized domains are then processed by the same registration backbone to compute optimized partial deformation fields, which are subsequently integrated into a global deformation field. SegReg achieves near-perfect structural alignment (98.23% Dice on critical anatomies) using ground-truth segmentation, and outperforms existing methods by 2-12% across three clinical registration scenarios (cardiac, abdominal, and lung images) even with automatic segmentation. Our SegReg demonstrates a near-linear dependence of registration accuracy on segmentation quality, transforming the registration challenge into a segmentation problem. The source code will be released upon manuscript acceptance. 

---
# On Optimal Steering to Achieve Exact Fairness 

**Authors**: Mohit Sharma, Amit Jayant Deshpande, Chiranjib Bhattacharyya, Rajiv Ratn Shah  

**Link**: [PDF](https://arxiv.org/pdf/2509.15759)  

**Abstract**: To fix the 'bias in, bias out' problem in fair machine learning, it is important to steer feature distributions of data or internal representations of Large Language Models (LLMs) to ideal ones that guarantee group-fair outcomes. Previous work on fair generative models and representation steering could greatly benefit from provable fairness guarantees on the model output. We define a distribution as ideal if the minimizer of any cost-sensitive risk on it is guaranteed to have exact group-fair outcomes (e.g., demographic parity, equal opportunity)-in other words, it has no fairness-utility trade-off. We formulate an optimization program for optimal steering by finding the nearest ideal distribution in KL-divergence, and provide efficient algorithms for it when the underlying distributions come from well-known parametric families (e.g., normal, log-normal). Empirically, our optimal steering techniques on both synthetic and real-world datasets improve fairness without diminishing utility (and sometimes even improve utility). We demonstrate affine steering of LLM representations to reduce bias in multi-class classification, e.g., occupation prediction from a short biography in Bios dataset (De-Arteaga et al.). Furthermore, we steer internal representations of LLMs towards desired outputs so that it works equally well across different groups. 

---
# FloorSAM: SAM-Guided Floorplan Reconstruction with Semantic-Geometric Fusion 

**Authors**: Han Ye, Haofu Wang, Yunchi Zhang, Jiangjian Xiao, Yuqiang Jin, Jinyuan Liu, Wen-An Zhang, Uladzislau Sychou, Alexander Tuzikov, Vladislav Sobolevskii, Valerii Zakharov, Boris Sokolov, Minglei Fu  

**Link**: [PDF](https://arxiv.org/pdf/2509.15750)  

**Abstract**: Reconstructing building floor plans from point cloud data is key for indoor navigation, BIM, and precise measurements. Traditional methods like geometric algorithms and Mask R-CNN-based deep learning often face issues with noise, limited generalization, and loss of geometric details. We propose FloorSAM, a framework that integrates point cloud density maps with the Segment Anything Model (SAM) for accurate floor plan reconstruction from LiDAR data. Using grid-based filtering, adaptive resolution projection, and image enhancement, we create robust top-down density maps. FloorSAM uses SAM's zero-shot learning for precise room segmentation, improving reconstruction across diverse layouts. Room masks are generated via adaptive prompt points and multistage filtering, followed by joint mask and point cloud analysis for contour extraction and regularization. This produces accurate floor plans and recovers room topological relationships. Tests on Giblayout and ISPRS datasets show better accuracy, recall, and robustness than traditional methods, especially in noisy and complex settings. Code and materials: this http URL. 

---
# GP3: A 3D Geometry-Aware Policy with Multi-View Images for Robotic Manipulation 

**Authors**: Quanhao Qian, Guoyang Zhao, Gongjie Zhang, Jiuniu Wang, Ran Xu, Junlong Gao, Deli Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2509.15733)  

**Abstract**: Effective robotic manipulation relies on a precise understanding of 3D scene geometry, and one of the most straightforward ways to acquire such geometry is through multi-view observations. Motivated by this, we present GP3 -- a 3D geometry-aware robotic manipulation policy that leverages multi-view input. GP3 employs a spatial encoder to infer dense spatial features from RGB observations, which enable the estimation of depth and camera parameters, leading to a compact yet expressive 3D scene representation tailored for manipulation. This representation is fused with language instructions and translated into continuous actions via a lightweight policy head. Comprehensive experiments demonstrate that GP3 consistently outperforms state-of-the-art methods on simulated benchmarks. Furthermore, GP3 transfers effectively to real-world robots without depth sensors or pre-mapped environments, requiring only minimal fine-tuning. These results highlight GP3 as a practical, sensor-agnostic solution for geometry-aware robotic manipulation. 

---
# Once Upon a Time: Interactive Learning for Storytelling with Small Language Models 

**Authors**: Jonas Mayer Martins, Ali Hamza Bashir, Muhammad Rehan Khalid, Lisa Beinborn  

**Link**: [PDF](https://arxiv.org/pdf/2509.15714)  

**Abstract**: Children efficiently acquire language not just by listening, but by interacting with others in their social environment. Conversely, large language models are typically trained with next-word prediction on massive amounts of text. Motivated by this contrast, we investigate whether language models can be trained with less data by learning not only from next-word prediction but also from high-level, cognitively inspired feedback. We train a student model to generate stories, which a teacher model rates on readability, narrative coherence, and creativity. By varying the amount of pretraining before the feedback loop, we assess the impact of this interactive learning on formal and functional linguistic competence. We find that the high-level feedback is highly data efficient: With just 1 M words of input in interactive learning, storytelling skills can improve as much as with 410 M words of next-word prediction. 

---
# SGMAGNet: A Baseline Model for 3D Cloud Phase Structure Reconstruction on a New Passive Active Satellite Benchmark 

**Authors**: Chi Yang, Fu Wang, Xiaofei Yang, Hao Huang, Weijia Cao, Xiaowen Chu  

**Link**: [PDF](https://arxiv.org/pdf/2509.15706)  

**Abstract**: Cloud phase profiles are critical for numerical weather prediction (NWP), as they directly affect radiative transfer and precipitation processes. In this study, we present a benchmark dataset and a baseline framework for transforming multimodal satellite observations into detailed 3D cloud phase structures, aiming toward operational cloud phase profile retrieval and future integration with NWP systems to improve cloud microphysics parameterization. The multimodal observations consist of (1) high--spatiotemporal--resolution, multi-band visible (VIS) and thermal infrared (TIR) imagery from geostationary satellites, and (2) accurate vertical cloud phase profiles from spaceborne lidar (CALIOP\slash CALIPSO) and radar (CPR\slash CloudSat). The dataset consists of synchronized image--profile pairs across diverse cloud regimes, defining a supervised learning task: given VIS/TIR patches, predict the corresponding 3D cloud phase structure. We adopt SGMAGNet as the main model and compare it with several baseline architectures, including UNet variants and SegNet, all designed to capture multi-scale spatial patterns. Model performance is evaluated using standard classification metrics, including Precision, Recall, F1-score, and IoU. The results demonstrate that SGMAGNet achieves superior performance in cloud phase reconstruction, particularly in complex multi-layer and boundary transition regions. Quantitatively, SGMAGNet attains a Precision of 0.922, Recall of 0.858, F1-score of 0.763, and an IoU of 0.617, significantly outperforming all baselines across these key metrics. 

---
# Saccadic Vision for Fine-Grained Visual Classification 

**Authors**: Johann Schmidt, Sebastian Stober, Joachim Denzler, Paul Bodesheim  

**Link**: [PDF](https://arxiv.org/pdf/2509.15688)  

**Abstract**: Fine-grained visual classification (FGVC) requires distinguishing between visually similar categories through subtle, localized features - a task that remains challenging due to high intra-class variability and limited inter-class differences. Existing part-based methods often rely on complex localization networks that learn mappings from pixel to sample space, requiring a deep understanding of image content while limiting feature utility for downstream tasks. In addition, sampled points frequently suffer from high spatial redundancy, making it difficult to quantify the optimal number of required parts. Inspired by human saccadic vision, we propose a two-stage process that first extracts peripheral features (coarse view) and generates a sample map, from which fixation patches are sampled and encoded in parallel using a weight-shared encoder. We employ contextualized selective attention to weigh the impact of each fixation patch before fusing peripheral and focus representations. To prevent spatial collapse - a common issue in part-based methods - we utilize non-maximum suppression during fixation sampling to eliminate redundancy. Comprehensive evaluation on standard FGVC benchmarks (CUB-200-2011, NABirds, Food-101 and Stanford-Dogs) and challenging insect datasets (EU-Moths, Ecuador-Moths and AMI-Moths) demonstrates that our method achieves comparable performance to state-of-the-art approaches while consistently outperforming our baseline encoder. 

---
# KITE: Kernelized and Information Theoretic Exemplars for In-Context Learning 

**Authors**: Vaibhav Singh, Soumya Suvra Ghosal, Kapu Nirmal Joshua, Soumyabrata Pal, Sayak Ray Chowdhury  

**Link**: [PDF](https://arxiv.org/pdf/2509.15676)  

**Abstract**: In-context learning (ICL) has emerged as a powerful paradigm for adapting large language models (LLMs) to new and data-scarce tasks using only a few carefully selected task-specific examples presented in the prompt. However, given the limited context size of LLMs, a fundamental question arises: Which examples should be selected to maximize performance on a given user query? While nearest-neighbor-based methods like KATE have been widely adopted for this purpose, they suffer from well-known drawbacks in high-dimensional embedding spaces, including poor generalization and a lack of diversity. In this work, we study this problem of example selection in ICL from a principled, information theory-driven perspective. We first model an LLM as a linear function over input embeddings and frame the example selection task as a query-specific optimization problem: selecting a subset of exemplars from a larger example bank that minimizes the prediction error on a specific query. This formulation departs from traditional generalization-focused learning theoretic approaches by targeting accurate prediction for a specific query instance. We derive a principled surrogate objective that is approximately submodular, enabling the use of a greedy algorithm with an approximation guarantee. We further enhance our method by (i) incorporating the kernel trick to operate in high-dimensional feature spaces without explicit mappings, and (ii) introducing an optimal design-based regularizer to encourage diversity in the selected examples. Empirically, we demonstrate significant improvements over standard retrieval methods across a suite of classification tasks, highlighting the benefits of structure-aware, diverse example selection for ICL in real-world, label-scarce scenarios. 

---
# Inference Offloading for Cost-Sensitive Binary Classification at the Edge 

**Authors**: Vishnu Narayanan Moothedath, Umang Agarwal, Umeshraja N, James Richard Gross, Jaya Prakash Champati, Sharayu Moharir  

**Link**: [PDF](https://arxiv.org/pdf/2509.15674)  

**Abstract**: We focus on a binary classification problem in an edge intelligence system where false negatives are more costly than false positives. The system has a compact, locally deployed model, which is supplemented by a larger, remote model, which is accessible via the network by incurring an offloading cost. For each sample, our system first uses the locally deployed model for inference. Based on the output of the local model, the sample may be offloaded to the remote model. This work aims to understand the fundamental trade-off between classification accuracy and these offloading costs within such a hierarchical inference (HI) system. To optimize this system, we propose an online learning framework that continuously adapts a pair of thresholds on the local model's confidence scores. These thresholds determine the prediction of the local model and whether a sample is classified locally or offloaded to the remote model. We present a closed-form solution for the setting where the local model is calibrated. For the more general case of uncalibrated models, we introduce H2T2, an online two-threshold hierarchical inference policy, and prove it achieves sublinear regret. H2T2 is model-agnostic, requires no training, and learns in the inference phase using limited feedback. Simulations on real-world datasets show that H2T2 consistently outperforms naive and single-threshold HI policies, sometimes even surpassing offline optima. The policy also demonstrates robustness to distribution shifts and adapts effectively to mismatched classifiers. 

---
# TISDiSS: A Training-Time and Inference-Time Scalable Framework for Discriminative Source Separation 

**Authors**: Yongsheng Feng, Yuetonghui Xu, Jiehui Luo, Hongjia Liu, Xiaobing Li, Feng Yu, Wei Li  

**Link**: [PDF](https://arxiv.org/pdf/2509.15666)  

**Abstract**: Source separation is a fundamental task in speech, music, and audio processing, and it also provides cleaner and larger data for training generative models. However, improving separation performance in practice often depends on increasingly large networks, inflating training and deployment costs. Motivated by recent advances in inference-time scaling for generative modeling, we propose Training-Time and Inference-Time Scalable Discriminative Source Separation (TISDiSS), a unified framework that integrates early-split multi-loss supervision, shared-parameter design, and dynamic inference repetitions. TISDiSS enables flexible speed-performance trade-offs by adjusting inference depth without retraining additional models. We further provide systematic analyses of architectural and training choices and show that training with more inference repetitions improves shallow-inference performance, benefiting low-latency applications. Experiments on standard speech separation benchmarks demonstrate state-of-the-art performance with a reduced parameter count, establishing TISDiSS as a scalable and practical framework for adaptive source separation. 

---
# SightSound-R1: Cross-Modal Reasoning Distillation from Vision to Audio Language Models 

**Authors**: Qiaolin Wang, Xilin Jiang, Linyang He, Junkai Wu, Nima Mesgarani  

**Link**: [PDF](https://arxiv.org/pdf/2509.15661)  

**Abstract**: While large audio-language models (LALMs) have demonstrated state-of-the-art audio understanding, their reasoning capability in complex soundscapes still falls behind large vision-language models (LVLMs). Compared to the visual domain, one bottleneck is the lack of large-scale chain-of-thought audio data to teach LALM stepwise reasoning. To circumvent this data and modality gap, we present SightSound-R1, a cross-modal distillation framework that transfers advanced reasoning from a stronger LVLM teacher to a weaker LALM student on the same audio-visual question answering (AVQA) dataset. SightSound-R1 consists of three core steps: (i) test-time scaling to generate audio-focused chains of thought (CoT) from an LVLM teacher, (ii) audio-grounded validation to filter hallucinations, and (iii) a distillation pipeline with supervised fine-tuning (SFT) followed by Group Relative Policy Optimization (GRPO) for the LALM student. Results show that SightSound-R1 improves LALM reasoning performance both in the in-domain AVQA test set as well as in unseen auditory scenes and questions, outperforming both pretrained and label-only distilled baselines. Thus, we conclude that vision reasoning can be effectively transferred to audio models and scaled with abundant audio-visual data. 

---
# Chunk Knowledge Generation Model for Enhanced Information Retrieval: A Multi-task Learning Approach 

**Authors**: Jisu Kim, Jinhee Park, Changhyun Jeon, Jungwoo Choi, Keonwoo Kim, Minji Hong, Sehyun Kim  

**Link**: [PDF](https://arxiv.org/pdf/2509.15658)  

**Abstract**: Traditional query expansion techniques for addressing vocabulary mismatch problems in information retrieval are context-sensitive and may lead to performance degradation. As an alternative, document expansion research has gained attention, but existing methods such as Doc2Query have limitations including excessive preprocessing costs, increased index size, and reliability issues with generated content. To mitigate these problems and seek more structured and efficient alternatives, this study proposes a method that divides documents into chunk units and generates textual data for each chunk to simultaneously improve retrieval efficiency and accuracy. The proposed "Chunk Knowledge Generation Model" adopts a T5-based multi-task learning structure that simultaneously generates titles and candidate questions from each document chunk while extracting keywords from user queries. This approach maximizes computational efficiency by generating and extracting three types of semantic information in parallel through a single encoding and two decoding processes. The generated data is utilized as additional information in the retrieval system. GPT-based evaluation on 305 query-document pairs showed that retrieval using the proposed model achieved 95.41% accuracy at Top@10, demonstrating superior performance compared to document chunk-level retrieval. This study contributes by proposing an approach that simultaneously generates titles and candidate questions from document chunks for application in retrieval pipelines, and provides empirical evidence applicable to large-scale information retrieval systems by demonstrating improved retrieval accuracy through qualitative evaluation. 

---
# Toward Efficient Influence Function: Dropout as a Compression Tool 

**Authors**: Yuchen Zhang, Mohammad Mohammadi Amiri  

**Link**: [PDF](https://arxiv.org/pdf/2509.15651)  

**Abstract**: Assessing the impact the training data on machine learning models is crucial for understanding the behavior of the model, enhancing the transparency, and selecting training data. Influence function provides a theoretical framework for quantifying the effect of training data points on model's performance given a specific test data. However, the computational and memory costs of influence function presents significant challenges, especially for large-scale models, even when using approximation methods, since the gradients involved in computation are as large as the model itself. In this work, we introduce a novel approach that leverages dropout as a gradient compression mechanism to compute the influence function more efficiently. Our method significantly reduces computational and memory overhead, not only during the influence function computation but also in gradient compression process. Through theoretical analysis and empirical validation, we demonstrate that our method could preserves critical components of the data influence and enables its application to modern large-scale models. 

---
# Information Geometry of Variational Bayes 

**Authors**: Mohammad Emtiyaz Khan  

**Link**: [PDF](https://arxiv.org/pdf/2509.15641)  

**Abstract**: We highlight a fundamental connection between information geometry and variational Bayes (VB) and discuss its consequences for machine learning. Under certain conditions, a VB solution always requires estimation or computation of natural gradients. We show several consequences of this fact by using the natural-gradient descent algorithm of Khan and Rue (2023) called the Bayesian Learning Rule (BLR). These include (i) a simplification of Bayes' rule as addition of natural gradients, (ii) a generalization of quadratic surrogates used in gradient-based methods, and (iii) a large-scale implementation of VB algorithms for large language models. Neither the connection nor its consequences are new but we further emphasize the common origins of the two fields of information geometry and Bayes with a hope to facilitate more work at the intersection of the two fields. 

---
# Latent Zoning Network: A Unified Principle for Generative Modeling, Representation Learning, and Classification 

**Authors**: Zinan Lin, Enshu Liu, Xuefei Ning, Junyi Zhu, Wenyu Wang, Sergey Yekhanin  

**Link**: [PDF](https://arxiv.org/pdf/2509.15591)  

**Abstract**: Generative modeling, representation learning, and classification are three core problems in machine learning (ML), yet their state-of-the-art (SoTA) solutions remain largely disjoint. In this paper, we ask: Can a unified principle address all three? Such unification could simplify ML pipelines and foster greater synergy across tasks. We introduce Latent Zoning Network (LZN) as a step toward this goal. At its core, LZN creates a shared Gaussian latent space that encodes information across all tasks. Each data type (e.g., images, text, labels) is equipped with an encoder that maps samples to disjoint latent zones, and a decoder that maps latents back to data. ML tasks are expressed as compositions of these encoders and decoders: for example, label-conditional image generation uses a label encoder and image decoder; image embedding uses an image encoder; classification uses an image encoder and label decoder. We demonstrate the promise of LZN in three increasingly complex scenarios: (1) LZN can enhance existing models (image generation): When combined with the SoTA Rectified Flow model, LZN improves FID on CIFAR10 from 2.76 to 2.59-without modifying the training objective. (2) LZN can solve tasks independently (representation learning): LZN can implement unsupervised representation learning without auxiliary loss functions, outperforming the seminal MoCo and SimCLR methods by 9.3% and 0.2%, respectively, on downstream linear classification on ImageNet. (3) LZN can solve multiple tasks simultaneously (joint generation and classification): With image and label encoders/decoders, LZN performs both tasks jointly by design, improving FID and achieving SoTA classification accuracy on CIFAR10. The code and trained models are available at this https URL. The project website is at this https URL. 

---
# CFDA & CLIP at TREC iKAT 2025: Enhancing Personalized Conversational Search via Query Reformulation and Rank Fusion 

**Authors**: Yu-Cheng Chang, Guan-Wei Yeo, Quah Eugene, Fan-Jie Shih, Yuan-Ching Kuo, Tsung-En Yu, Hung-Chun Hsu, Ming-Feng Tsai, Chuan-Ju Wang  

**Link**: [PDF](https://arxiv.org/pdf/2509.15588)  

**Abstract**: The 2025 TREC Interactive Knowledge Assistance Track (iKAT) featured both interactive and offline submission tasks. The former requires systems to operate under real-time constraints, making robustness and efficiency as important as accuracy, while the latter enables controlled evaluation of passage ranking and response generation with pre-defined datasets. To address this, we explored query rewriting and retrieval fusion as core strategies. We built our pipelines around Best-of-$N$ selection and Reciprocal Rank Fusion (RRF) strategies to handle different submission tasks. Results show that reranking and fusion improve robustness while revealing trade-offs between effectiveness and efficiency across both tasks. 

---
# DivLogicEval: A Framework for Benchmarking Logical Reasoning Evaluation in Large Language Models 

**Authors**: Tsz Ting Chung, Lemao Liu, Mo Yu, Dit-Yan Yeung  

**Link**: [PDF](https://arxiv.org/pdf/2509.15587)  

**Abstract**: Logic reasoning in natural language has been recognized as an important measure of human intelligence for Large Language Models (LLMs). Popular benchmarks may entangle multiple reasoning skills and thus provide unfaithful evaluations on the logic reasoning skill. Meanwhile, existing logic reasoning benchmarks are limited in language diversity and their distributions are deviated from the distribution of an ideal logic reasoning benchmark, which may lead to biased evaluation results. This paper thereby proposes a new classical logic benchmark DivLogicEval, consisting of natural sentences composed of diverse statements in a counterintuitive way. To ensure a more reliable evaluation, we also introduce a new evaluation metric that mitigates the influence of bias and randomness inherent in LLMs. Through experiments, we demonstrate the extent to which logical reasoning is required to answer the questions in DivLogicEval and compare the performance of different popular LLMs in conducting logical reasoning. 

---
# Momentum-constrained Hybrid Heuristic Trajectory Optimization Framework with Residual-enhanced DRL for Visually Impaired Scenarios 

**Authors**: Yuting Zeng, Zhiwen Zheng, You Zhou, JiaLing Xiao, Yongbin Yu, Manping Fan, Bo Gong, Liyong Ren  

**Link**: [PDF](https://arxiv.org/pdf/2509.15582)  

**Abstract**: This paper proposes a momentum-constrained hybrid heuristic trajectory optimization framework (MHHTOF) tailored for assistive navigation in visually impaired scenarios, integrating trajectory sampling generation, optimization and evaluation with residual-enhanced deep reinforcement learning (DRL). In the first stage, heuristic trajectory sampling cluster (HTSC) is generated in the Frenet coordinate system using third-order interpolation with fifth-order polynomials and momentum-constrained trajectory optimization (MTO) constraints to ensure smoothness and feasibility. After first stage cost evaluation, the second stage leverages a residual-enhanced actor-critic network with LSTM-based temporal feature modeling to adaptively refine trajectory selection in the Cartesian coordinate system. A dual-stage cost modeling mechanism (DCMM) with weight transfer aligns semantic priorities across stages, supporting human-centered optimization. Experimental results demonstrate that the proposed LSTM-ResB-PPO achieves significantly faster convergence, attaining stable policy performance in approximately half the training iterations required by the PPO baseline, while simultaneously enhancing both reward outcomes and training stability. Compared to baseline method, the selected model reduces average cost and cost variance by 30.3% and 53.3%, and lowers ego and obstacle risks by over 77%. These findings validate the framework's effectiveness in enhancing robustness, safety, and real-time feasibility in complex assistive planning tasks. 

---
# Multimodal Learning for Fake News Detection in Short Videos Using Linguistically Verified Data and Heterogeneous Modality Fusion 

**Authors**: Shanghong Li, Chiam Wen Qi Ruth, Hong Xu, Fang Liu  

**Link**: [PDF](https://arxiv.org/pdf/2509.15578)  

**Abstract**: The rapid proliferation of short video platforms has necessitated advanced methods for detecting fake news. This need arises from the widespread influence and ease of sharing misinformation, which can lead to significant societal harm. Current methods often struggle with the dynamic and multimodal nature of short video content. This paper presents HFN, Heterogeneous Fusion Net, a novel multimodal framework that integrates video, audio, and text data to evaluate the authenticity of short video content. HFN introduces a Decision Network that dynamically adjusts modality weights during inference and a Weighted Multi-Modal Feature Fusion module to ensure robust performance even with incomplete data. Additionally, we contribute a comprehensive dataset VESV (VEracity on Short Videos) specifically designed for short video fake news detection. Experiments conducted on the FakeTT and newly collected VESV datasets demonstrate improvements of 2.71% and 4.14% in Marco F1 over state-of-the-art methods. This work establishes a robust solution capable of effectively identifying fake news in the complex landscape of short video platforms, paving the way for more reliable and comprehensive approaches in combating misinformation. 

---
# Relevance to Utility: Process-Supervised Rewrite for RAG 

**Authors**: Jaeyoung Kim, Jongho Kim, Seung-won Hwang, Seoho Song, Young-In Song  

**Link**: [PDF](https://arxiv.org/pdf/2509.15577)  

**Abstract**: Retrieval-Augmented Generation systems often suffer from a gap between optimizing retrieval relevance and generative utility: retrieved documents may be topically relevant but still lack the content needed for effective reasoning during generation. While existing "bridge" modules attempt to rewrite the retrieved text for better generation, we show how they fail to capture true document utility. In this work, we propose R2U, with a key distinction of directly optimizing to maximize the probability of generating a correct answer through process supervision. As such direct observation is expensive, we also propose approximating an efficient distillation pipeline by scaling the supervision from LLMs, which helps the smaller rewriter model generalize better. We evaluate our method across multiple open-domain question-answering benchmarks. The empirical results demonstrate consistent improvements over strong bridging baselines. 

---
# Towards Size-invariant Salient Object Detection: A Generic Evaluation and Optimization Approach 

**Authors**: Shilong Bao, Qianqian Xu, Feiran Li, Boyu Han, Zhiyong Yang, Xiaochun Cao, Qingming Huang  

**Link**: [PDF](https://arxiv.org/pdf/2509.15573)  

**Abstract**: This paper investigates a fundamental yet underexplored issue in Salient Object Detection (SOD): the size-invariant property for evaluation protocols, particularly in scenarios when multiple salient objects of significantly different sizes appear within a single image. We first present a novel perspective to expose the inherent size sensitivity of existing widely used SOD metrics. Through careful theoretical derivations, we show that the evaluation outcome of an image under current SOD metrics can be essentially decomposed into a sum of several separable terms, with the contribution of each term being directly proportional to its corresponding region size. Consequently, the prediction errors would be dominated by the larger regions, while smaller yet potentially more semantically important objects are often overlooked, leading to biased performance assessments and practical degradation. To address this challenge, a generic Size-Invariant Evaluation (SIEva) framework is proposed. The core idea is to evaluate each separable component individually and then aggregate the results, thereby effectively mitigating the impact of size imbalance across objects. Building upon this, we further develop a dedicated optimization framework (SIOpt), which adheres to the size-invariant principle and significantly enhances the detection of salient objects across a broad range of sizes. Notably, SIOpt is model-agnostic and can be seamlessly integrated with a wide range of SOD backbones. Theoretically, we also present generalization analysis of SOD methods and provide evidence supporting the validity of our new evaluation protocols. Finally, comprehensive experiments speak to the efficacy of our proposed approach. The code is available at this https URL. 

---
# Contrastive Learning with Spectrum Information Augmentation in Abnormal Sound Detection 

**Authors**: Xinxin Meng, Jiangtao Guo, Yunxiang Zhang, Shun Huang  

**Link**: [PDF](https://arxiv.org/pdf/2509.15570)  

**Abstract**: The outlier exposure method is an effective approach to address the unsupervised anomaly sound detection problem. The key focus of this method is how to make the model learn the distribution space of normal data. Based on biological perception and data analysis, it is found that anomalous audio and noise often have higher frequencies. Therefore, we propose a data augmentation method for high-frequency information in contrastive learning. This enables the model to pay more attention to the low-frequency information of the audio, which represents the normal operational mode of the machine. We evaluated the proposed method on the DCASE 2020 Task 2. The results showed that our method outperformed other contrastive learning methods used on this dataset. We also evaluated the generalizability of our method on the DCASE 2022 Task 2 dataset. 

---
# LiteLong: Resource-Efficient Long-Context Data Synthesis for LLMs 

**Authors**: Junlong Jia, Xing Wu, Chaochen Gao, Ziyang Chen, Zijia Lin, Zhongzhi Li, Weinong Wang, Haotian Xu, Donghui Jin, Debing Zhang, Binghui Guo  

**Link**: [PDF](https://arxiv.org/pdf/2509.15568)  

**Abstract**: High-quality long-context data is essential for training large language models (LLMs) capable of processing extensive documents, yet existing synthesis approaches using relevance-based aggregation face challenges of computational efficiency. We present LiteLong, a resource-efficient method for synthesizing long-context data through structured topic organization and multi-agent debate. Our approach leverages the BISAC book classification system to provide a comprehensive hierarchical topic organization, and then employs a debate mechanism with multiple LLMs to generate diverse, high-quality topics within this structure. For each topic, we use lightweight BM25 retrieval to obtain relevant documents and concatenate them into 128K-token training samples. Experiments on HELMET and Ruler benchmarks demonstrate that LiteLong achieves competitive long-context performance and can seamlessly integrate with other long-dependency enhancement methods. LiteLong makes high-quality long-context data synthesis more accessible by reducing both computational and data engineering costs, facilitating further research in long-context language training. 

---
# BTL-UI: Blink-Think-Link Reasoning Model for GUI Agent 

**Authors**: Shaojie Zhang, Ruoceng Zhang, Pei Fu, Shaokang Wang, Jiahui Yang, Xin Du, Shiqi Cui, Bin Qin, Ying Huang, Zhenbo Luo, Jian Luan  

**Link**: [PDF](https://arxiv.org/pdf/2509.15566)  

**Abstract**: In the field of AI-driven human-GUI interaction automation, while rapid advances in multimodal large language models and reinforcement fine-tuning techniques have yielded remarkable progress, a fundamental challenge persists: their interaction logic significantly deviates from natural human-GUI communication patterns. To fill this gap, we propose "Blink-Think-Link" (BTL), a brain-inspired framework for human-GUI interaction that mimics the human cognitive process between users and graphical interfaces. The system decomposes interactions into three biologically plausible phases: (1) Blink - rapid detection and attention to relevant screen areas, analogous to saccadic eye movements; (2) Think - higher-level reasoning and decision-making, mirroring cognitive planning; and (3) Link - generation of executable commands for precise motor control, emulating human action selection mechanisms. Additionally, we introduce two key technical innovations for the BTL framework: (1) Blink Data Generation - an automated annotation pipeline specifically optimized for blink data, and (2) BTL Reward -- the first rule-based reward mechanism that enables reinforcement learning driven by both process and outcome. Building upon this framework, we develop a GUI agent model named BTL-UI, which demonstrates consistent state-of-the-art performance across both static GUI understanding and dynamic interaction tasks in comprehensive benchmarks. These results provide conclusive empirical validation of the framework's efficacy in developing advanced GUI Agents. 

---
# Reward Hacking Mitigation using Verifiable Composite Rewards 

**Authors**: Mirza Farhan Bin Tarek, Rahmatollah Beheshti  

**Link**: [PDF](https://arxiv.org/pdf/2509.15557)  

**Abstract**: Reinforcement Learning from Verifiable Rewards (RLVR) has recently shown that large language models (LLMs) can develop their own reasoning without direct supervision. However, applications in the medical domain, specifically for question answering, are susceptible to significant reward hacking during the reasoning phase. Our work addresses two primary forms of this behavior: i) providing a final answer without preceding reasoning, and ii) employing non-standard reasoning formats to exploit the reward mechanism. To mitigate these, we introduce a composite reward function with specific penalties for these behaviors. Our experiments show that extending RLVR with our proposed reward model leads to better-formatted reasoning with less reward hacking and good accuracy compared to the baselines. This approach marks a step toward reducing reward hacking and enhancing the reliability of models utilizing RLVR. 

---
# Exploring Polyglot Harmony: On Multilingual Data Allocation for Large Language Models Pretraining 

**Authors**: Ping Guo, Yubing Ren, Binbin Liu, Fengze Liu, Haobin Lin, Yifan Zhang, Bingni Zhang, Taifeng Wang, Yin Zheng  

**Link**: [PDF](https://arxiv.org/pdf/2509.15556)  

**Abstract**: Large language models (LLMs) have become integral to a wide range of applications worldwide, driving an unprecedented global demand for effective multilingual capabilities. Central to achieving robust multilingual performance is the strategic allocation of language proportions within training corpora. However, determining optimal language ratios is highly challenging due to intricate cross-lingual interactions and sensitivity to dataset scale. This paper introduces Climb (Cross-Lingual Interaction-aware Multilingual Balancing), a novel framework designed to systematically optimize multilingual data allocation. At its core, Climb introduces a cross-lingual interaction-aware language ratio, explicitly quantifying each language's effective allocation by capturing inter-language dependencies. Leveraging this ratio, Climb proposes a principled two-step optimization procedure--first equalizing marginal benefits across languages, then maximizing the magnitude of the resulting language allocation vectors--significantly simplifying the inherently complex multilingual optimization problem. Extensive experiments confirm that Climb can accurately measure cross-lingual interactions across various multilingual settings. LLMs trained with Climb-derived proportions consistently achieve state-of-the-art multilingual performance, even achieving competitive performance with open-sourced LLMs trained with more tokens. 

---
# Diffusion-Based Cross-Modal Feature Extraction for Multi-Label Classification 

**Authors**: Tian Lan, Yiming Zheng, Jianxin Yin  

**Link**: [PDF](https://arxiv.org/pdf/2509.15553)  

**Abstract**: Multi-label classification has broad applications and depends on powerful representations capable of capturing multi-label interactions. We introduce \textit{Diff-Feat}, a simple but powerful framework that extracts intermediate features from pre-trained diffusion-Transformer models for images and text, and fuses them for downstream tasks. We observe that for vision tasks, the most discriminative intermediate feature along the diffusion process occurs at the middle step and is located in the middle block in Transformer. In contrast, for language tasks, the best feature occurs at the noise-free step and is located in the deepest block. In particular, we observe a striking phenomenon across varying datasets: a mysterious "Layer $12$" consistently yields the best performance on various downstream classification tasks for images (under DiT-XL/2-256$\times$256). We devise a heuristic local-search algorithm that pinpoints the locally optimal "image-text"$\times$"block-timestep" pair among a few candidates, avoiding an exhaustive grid search. A simple fusion-linear projection followed by addition-of the selected representations yields state-of-the-art performance: 98.6\% mAP on MS-COCO-enhanced and 45.7\% mAP on Visual Genome 500, surpassing strong CNN, graph, and Transformer baselines by a wide margin. t-SNE and clustering metrics further reveal that \textit{Diff-Feat} forms tighter semantic clusters than unimodal counterparts. The code is available at this https URL. 

---
# GUI-ARP: Enhancing Grounding with Adaptive Region Perception for GUI Agents 

**Authors**: Xianhang Ye, Yiqing Li, Wei Dai, Miancan Liu, Ziyuan Chen, Zhangye Han, Hongbo Min, Jinkui Ren, Xiantao Zhang, Wen Yang, Zhi Jin  

**Link**: [PDF](https://arxiv.org/pdf/2509.15532)  

**Abstract**: Existing GUI grounding methods often struggle with fine-grained localization in high-resolution screenshots. To address this, we propose GUI-ARP, a novel framework that enables adaptive multi-stage inference. Equipped with the proposed Adaptive Region Perception (ARP) and Adaptive Stage Controlling (ASC), GUI-ARP dynamically exploits visual attention for cropping task-relevant regions and adapts its inference strategy, performing a single-stage inference for simple cases and a multi-stage analysis for more complex scenarios. This is achieved through a two-phase training pipeline that integrates supervised fine-tuning with reinforcement fine-tuning based on Group Relative Policy Optimization (GRPO). Extensive experiments demonstrate that the proposed GUI-ARP achieves state-of-the-art performance on challenging GUI grounding benchmarks, with a 7B model reaching 60.8% accuracy on ScreenSpot-Pro and 30.9% on UI-Vision benchmark. Notably, GUI-ARP-7B demonstrates strong competitiveness against open-source 72B models (UI-TARS-72B at 38.1%) and proprietary models. 

---
# How do Language Models Generate Slang: A Systematic Comparison between Human and Machine-Generated Slang Usages 

**Authors**: Siyang Wu, Zhewei Sun  

**Link**: [PDF](https://arxiv.org/pdf/2509.15518)  

**Abstract**: Slang is a commonly used type of informal language that poses a daunting challenge to NLP systems. Recent advances in large language models (LLMs), however, have made the problem more approachable. While LLM agents are becoming more widely applied to intermediary tasks such as slang detection and slang interpretation, their generalizability and reliability are heavily dependent on whether these models have captured structural knowledge about slang that align well with human attested slang usages. To answer this question, we contribute a systematic comparison between human and machine-generated slang usages. Our evaluative framework focuses on three core aspects: 1) Characteristics of the usages that reflect systematic biases in how machines perceive slang, 2) Creativity reflected by both lexical coinages and word reuses employed by the slang usages, and 3) Informativeness of the slang usages when used as gold-standard examples for model distillation. By comparing human-attested slang usages from the Online Slang Dictionary (OSD) and slang generated by GPT-4o and Llama-3, we find significant biases in how LLMs perceive slang. Our results suggest that while LLMs have captured significant knowledge about the creative aspects of slang, such knowledge does not align with humans sufficiently to enable LLMs for extrapolative tasks such as linguistic analyses. 

---
# The (Short-Term) Effects of Large Language Models on Unemployment and Earnings 

**Authors**: Danqing Chen, Carina Kane, Austin Kozlowski, Nadav Kunievsky, James A. Evans  

**Link**: [PDF](https://arxiv.org/pdf/2509.15510)  

**Abstract**: Large Language Models have spread rapidly since the release of ChatGPT in late 2022, accompanied by claims of major productivity gains but also concerns about job displacement. This paper examines the short-run labor market effects of LLM adoption by comparing earnings and unemployment across occupations with differing levels of exposure to these technologies. Using a Synthetic Difference in Differences approach, we estimate the impact of LLM exposure on earnings and unemployment. Our findings show that workers in highly exposed occupations experienced earnings increases following ChatGPT's introduction, while unemployment rates remained unchanged. These results suggest that initial labor market adjustments to LLMs operate primarily through earnings rather than worker reallocation. 

---
# Explainable AI-Enhanced Supervisory Control for Robust Multi-Agent Robotic Systems 

**Authors**: Reza Pirayeshshirazinezhad, Nima Fathi  

**Link**: [PDF](https://arxiv.org/pdf/2509.15491)  

**Abstract**: We present an explainable AI-enhanced supervisory control framework for multi-agent robotics that combines (i) a timed-automata supervisor for safe, auditable mode switching, (ii) robust continuous control (Lyapunov-based controller for large-angle maneuver; sliding-mode controller (SMC) with boundary layers for precision and disturbance rejection), and (iii) an explainable predictor that maps mission context to gains and expected performance (energy, error). Monte Carlo-driven optimization provides the training data, enabling transparent real-time trade-offs.
We validated the approach in two contrasting domains, spacecraft formation flying and autonomous underwater vehicles (AUVs). Despite different environments (gravity/actuator bias vs. hydrodynamic drag/currents), both share uncertain six degrees of freedom (6-DOF) rigid-body dynamics, relative motion, and tight tracking needs, making them representative of general robotic systems. In the space mission, the supervisory logic selects parameters that meet mission criteria. In AUV leader-follower tests, the same SMC structure maintains a fixed offset under stochastic currents with bounded steady error. In spacecraft validation, the SMC controller achieved submillimeter alignment with 21.7% lower tracking error and 81.4% lower energy consumption compared to Proportional-Derivative PD controller baselines. At the same time, in AUV tests, SMC maintained bounded errors under stochastic currents. These results highlight both the portability and the interpretability of the approach for safety-critical, resource-constrained multi-agent robotics. 

---
# SmolRGPT: Efficient Spatial Reasoning for Warehouse Environments with 600M Parameters 

**Authors**: Abdarahmane Traore, Éric Hervet, Andy Couturier  

**Link**: [PDF](https://arxiv.org/pdf/2509.15490)  

**Abstract**: Recent advances in vision-language models (VLMs) have enabled powerful multimodal reasoning, but state-of-the-art approaches typically rely on extremely large models with prohibitive computational and memory requirements. This makes their deployment challenging in resource-constrained environments such as warehouses, robotics, and industrial applications, where both efficiency and robust spatial understanding are critical. In this work, we present SmolRGPT, a compact vision-language architecture that explicitly incorporates region-level spatial reasoning by integrating both RGB and depth cues. SmolRGPT employs a three-stage curriculum that progressively align visual and language features, enables spatial relationship understanding, and adapts to task-specific datasets. We demonstrate that with only 600M parameters, SmolRGPT achieves competitive results on challenging warehouse spatial reasoning benchmarks, matching or exceeding the performance of much larger alternatives. These findings highlight the potential for efficient, deployable multimodal intelligence in real-world settings without sacrificing core spatial reasoning capabilities. The code of the experimentation will be available at: this https URL 

---
# mucAI at BAREC Shared Task 2025: Towards Uncertainty Aware Arabic Readability Assessment 

**Authors**: Ahmed Abdou  

**Link**: [PDF](https://arxiv.org/pdf/2509.15485)  

**Abstract**: We present a simple, model-agnostic post-processing technique for fine-grained Arabic readability classification in the BAREC 2025 Shared Task (19 ordinal levels). Our method applies conformal prediction to generate prediction sets with coverage guarantees, then computes weighted averages using softmax-renormalized probabilities over the conformal sets. This uncertainty-aware decoding improves Quadratic Weighted Kappa (QWK) by reducing high-penalty misclassifications to nearer levels. Our approach shows consistent QWK improvements of 1-3 points across different base models. In the strict track, our submission achieves QWK scores of 84.9\%(test) and 85.7\% (blind test) for sentence level, and 73.3\% for document level. For Arabic educational assessment, this enables human reviewers to focus on a handful of plausible levels, combining statistical guarantees with practical usability. 

---
# Comparing Computational Pathology Foundation Models using Representational Similarity Analysis 

**Authors**: Vaibhav Mishra, William Lotter  

**Link**: [PDF](https://arxiv.org/pdf/2509.15482)  

**Abstract**: Foundation models are increasingly developed in computational pathology (CPath) given their promise in facilitating many downstream tasks. While recent studies have evaluated task performance across models, less is known about the structure and variability of their learned representations. Here, we systematically analyze the representational spaces of six CPath foundation models using techniques popularized in computational neuroscience. The models analyzed span vision-language contrastive learning (CONCH, PLIP, KEEP) and self-distillation (UNI (v2), Virchow (v2), Prov-GigaPath) approaches. Through representational similarity analysis using H&E image patches from TCGA, we find that UNI2 and Virchow2 have the most distinct representational structures, whereas Prov-Gigapath has the highest average similarity across models. Having the same training paradigm (vision-only vs. vision-language) did not guarantee higher representational similarity. The representations of all models showed a high slide-dependence, but relatively low disease-dependence. Stain normalization decreased slide-dependence for all models by a range of 5.5% (CONCH) to 20.5% (PLIP). In terms of intrinsic dimensionality, vision-language models demonstrated relatively compact representations, compared to the more distributed representations of vision-only models. These findings highlight opportunities to improve robustness to slide-specific features, inform model ensembling strategies, and provide insights into how training paradigms shape model representations. Our framework is extendable across medical imaging domains, where probing the internal representations of foundation models can help ensure effective development and deployment. 

---
# Self-supervised learning of imaging and clinical signatures using a multimodal joint-embedding predictive architecture 

**Authors**: Thomas Z. Li, Aravind R. Krishnan, Lianrui Zuo, John M. Still, Kim L. Sandler, Fabien Maldonado, Thomas A. Lasko, Bennett A. Landman  

**Link**: [PDF](https://arxiv.org/pdf/2509.15470)  

**Abstract**: The development of multimodal models for pulmonary nodule diagnosis is limited by the scarcity of labeled data and the tendency for these models to overfit on the training distribution. In this work, we leverage self-supervised learning from longitudinal and multimodal archives to address these challenges. We curate an unlabeled set of patients with CT scans and linked electronic health records from our home institution to power joint embedding predictive architecture (JEPA) pretraining. After supervised finetuning, we show that our approach outperforms an unregularized multimodal model and imaging-only model in an internal cohort (ours: 0.91, multimodal: 0.88, imaging-only: 0.73 AUC), but underperforms in an external cohort (ours: 0.72, imaging-only: 0.75 AUC). We develop a synthetic environment that characterizes the context in which JEPA may underperform. This work innovates an approach that leverages unlabeled multimodal medical archives to improve predictive models and demonstrates its advantages and limitations in pulmonary nodule diagnosis. 

---
# Incorporating Visual Cortical Lateral Connection Properties into CNN: Recurrent Activation and Excitatory-Inhibitory Separation 

**Authors**: Jin Hyun Park, Cheng Zhang, Yoonsuck Choe  

**Link**: [PDF](https://arxiv.org/pdf/2509.15460)  

**Abstract**: The original Convolutional Neural Networks (CNNs) and their modern updates such as the ResNet are heavily inspired by the mammalian visual system. These models include afferent connections (retina and LGN to the visual cortex) and long-range projections (connections across different visual cortical areas). However, in the mammalian visual system, there are connections within each visual cortical area, known as lateral (or horizontal) connections. These would roughly correspond to connections within CNN feature maps, and this important architectural feature is missing in current CNN models. In this paper, we present how such lateral connections can be modeled within the standard CNN framework, and test its benefits and analyze its emergent properties in relation to the biological visual system. We will focus on two main architectural features of lateral connections: (1) recurrent activation and (2) separation of excitatory and inhibitory connections. We show that recurrent CNN using weight sharing is equivalent to lateral connections, and propose a custom loss function to separate excitatory and inhibitory weights. The addition of these two leads to increased classification accuracy, and importantly, the activation properties and connection properties of the resulting model show properties similar to those observed in the biological visual system. We expect our approach to help align CNN closer to its biological counterpart and better understand the principles of visual cortical computation. 

---
# CAGE: Continuity-Aware edGE Network Unlocks Robust Floorplan Reconstruction 

**Authors**: Yiyi Liu, Chunyang Liu, Weiqin Jiao, Bojian Wu, Fashuai Li, Biao Xiong  

**Link**: [PDF](https://arxiv.org/pdf/2509.15459)  

**Abstract**: We present \textbf{CAGE} (\textit{Continuity-Aware edGE}) network, a \textcolor{red}{robust} framework for reconstructing vector floorplans directly from point-cloud density maps. Traditional corner-based polygon representations are highly sensitive to noise and incomplete observations, often resulting in fragmented or implausible layouts. Recent line grouping methods leverage structural cues to improve robustness but still struggle to recover fine geometric details. To address these limitations, we propose a \textit{native} edge-centric formulation, modeling each wall segment as a directed, geometrically continuous edge. This representation enables inference of coherent floorplan structures, ensuring watertight, topologically valid room boundaries while improving robustness and reducing artifacts. Towards this design, we develop a dual-query transformer decoder that integrates perturbed and latent queries within a denoising framework, which not only stabilizes optimization but also accelerates convergence. Extensive experiments on Structured3D and SceneCAD show that \textbf{CAGE} achieves state-of-the-art performance, with F1 scores of 99.1\% (rooms), 91.7\% (corners), and 89.3\% (angles). The method also demonstrates strong cross-dataset generalization, underscoring the efficacy of our architectural innovations. Code and pretrained models will be released upon acceptance. 

---
# Hierarchical Self-Attention: Generalizing Neural Attention Mechanics to Multi-Scale Problems 

**Authors**: Saeed Amizadeh, Sara Abdali, Yinheng Li, Kazuhito Koishida  

**Link**: [PDF](https://arxiv.org/pdf/2509.15448)  

**Abstract**: Transformers and their attention mechanism have been revolutionary in the field of Machine Learning. While originally proposed for the language data, they quickly found their way to the image, video, graph, etc. data modalities with various signal geometries. Despite this versatility, generalizing the attention mechanism to scenarios where data is presented at different scales from potentially different modalities is not straightforward. The attempts to incorporate hierarchy and multi-modality within transformers are largely based on ad hoc heuristics, which are not seamlessly generalizable to similar problems with potentially different structures. To address this problem, in this paper, we take a fundamentally different approach: we first propose a mathematical construct to represent multi-modal, multi-scale data. We then mathematically derive the neural attention mechanics for the proposed construct from the first principle of entropy minimization. We show that the derived formulation is optimal in the sense of being the closest to the standard Softmax attention while incorporating the inductive biases originating from the hierarchical/geometric information of the problem. We further propose an efficient algorithm based on dynamic programming to compute our derived attention mechanism. By incorporating it within transformers, we show that the proposed hierarchical attention mechanism not only can be employed to train transformer models in hierarchical/multi-modal settings from scratch, but it can also be used to inject hierarchical information into classical, pre-trained transformer models post training, resulting in more efficient models in zero-shot manner. 

---
# PILOT: Steering Synthetic Data Generation with Psychological & Linguistic Output Targeting 

**Authors**: Caitlin Cisar, Emily Sheffield, Joshua Drake, Alden Harrell, Subramanian Chidambaram, Nikita Nangia, Vinayak Arannil, Alex Williams  

**Link**: [PDF](https://arxiv.org/pdf/2509.15447)  

**Abstract**: Generative AI applications commonly leverage user personas as a steering mechanism for synthetic data generation, but reliance on natural language representations forces models to make unintended inferences about which attributes to emphasize, limiting precise control over outputs. We introduce PILOT (Psychological and Linguistic Output Targeting), a two-phase framework for steering large language models with structured psycholinguistic profiles. In Phase 1, PILOT translates natural language persona descriptions into multidimensional profiles with normalized scores across linguistic and psychological dimensions. In Phase 2, these profiles guide generation along measurable axes of variation. We evaluate PILOT across three state-of-the-art LLMs (Mistral Large 2, Deepseek-R1, LLaMA 3.3 70B) using 25 synthetic personas under three conditions: Natural-language Persona Steering (NPS), Schema-Based Steering (SBS), and Hybrid Persona-Schema Steering (HPS). Results demonstrate that schema-based approaches significantly reduce artificial-sounding persona repetition while improving output coherence, with silhouette scores increasing from 0.098 to 0.237 and topic purity from 0.773 to 0.957. Our analysis reveals a fundamental trade-off: SBS produces more concise outputs with higher topical consistency, while NPS offers greater lexical diversity but reduced predictability. HPS achieves a balance between these extremes, maintaining output variety while preserving structural consistency. Expert linguistic evaluation confirms that PILOT maintains high response quality across all conditions, with no statistically significant differences between steering approaches. 

---
# Implicit Kinodynamic Motion Retargeting for Human-to-humanoid Imitation Learning 

**Authors**: Xingyu Chen, Hanyu Wu, Sikai Wu, Mingliang Zhou, Diyun Xiang, Haodong Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2509.15443)  

**Abstract**: Human-to-humanoid imitation learning aims to learn a humanoid whole-body controller from human motion. Motion retargeting is a crucial step in enabling robots to acquire reference trajectories when exploring locomotion skills. However, current methods focus on motion retargeting frame by frame, which lacks scalability. Could we directly convert large-scale human motion into robot-executable motion through a more efficient approach? To address this issue, we propose Implicit Kinodynamic Motion Retargeting (IKMR), a novel efficient and scalable retargeting framework that considers both kinematics and dynamics. In kinematics, IKMR pretrains motion topology feature representation and a dual encoder-decoder architecture to learn a motion domain mapping. In dynamics, IKMR integrates imitation learning with the motion retargeting network to refine motion into physically feasible trajectories. After fine-tuning using the tracking results, IKMR can achieve large-scale physically feasible motion retargeting in real time, and a whole-body controller could be directly trained and deployed for tracking its retargeted trajectories. We conduct our experiments both in the simulator and the real robot on a full-size humanoid robot. Extensive experiments and evaluation results verify the effectiveness of our proposed framework. 

---
# Where Do I 'Add the Egg'?: Exploring Agency and Ownership in AI Creative Co-Writing Systems 

**Authors**: Dashiel Carrera, Jeb Thomas-Mitchell, Daniel Wigdor  

**Link**: [PDF](https://arxiv.org/pdf/2509.15440)  

**Abstract**: AI co-writing systems challenge long held ideals about agency and ownership in the creative process, thereby hindering widespread adoption. In order to address this, we investigate conceptions of agency and ownership in AI creative co-writing. Drawing on insights from a review of commercial systems, we developed three co-writing systems with identical functionality but distinct interface metaphors: agentic, tool-like, and magical. Through interviews with professional and non-professional writers (n = 18), we explored how these metaphors influenced participants' sense of control and authorship. Our analysis resulted in a taxonomy of agency and ownership subtypes and underscore how tool-like metaphors shift writers' expected points of control while agentic metaphors foreground conceptual contributions. We argue that interface metaphors not only guide expectations of control but also frame conceptions of authorship. We conclude with recommendations for the design of AI co-writing systems, emphasizing how metaphor shapes user experience and creative practice. 

---
# Dual-Mode Visual System for Brain-Computer Interfaces: Integrating SSVEP and P300 Responses 

**Authors**: Ekgari Kasawala, Surej Mouli  

**Link**: [PDF](https://arxiv.org/pdf/2509.15439)  

**Abstract**: In brain-computer interface (BCI) systems, steady-state visual evoked potentials (SSVEP) and P300 responses have achieved widespread implementation owing to their superior information transfer rates (ITR) and minimal training requirements. These neurophysiological signals have exhibited robust efficacy and versatility in external device control, demonstrating enhanced precision and scalability. However, conventional implementations predominantly utilise liquid crystal display (LCD)-based visual stimulation paradigms, which present limitations in practical deployment scenarios. This investigation presents the development and evaluation of a novel light-emitting diode (LED)-based dual stimulation apparatus designed to enhance SSVEP classification accuracy through the integration of both SSVEP and P300 paradigms. The system employs four distinct frequencies, 7 Hz, 8 Hz, 9 Hz, and 10 Hz, corresponding to forward, backward, right, and left directional controls, respectively. Oscilloscopic verification confirmed the precision of these stimulation frequencies. Real-time feature extraction was accomplished through the concurrent analysis of maximum Fast Fourier Transform (FFT) amplitude and P300 peak detection to ascertain user intent. Directional control was determined by the frequency exhibiting maximal amplitude characteristics. The visual stimulation hardware demonstrated minimal frequency deviation, with error differentials ranging from 0.15%to 0.20%across all frequencies. The implemented signal processing algorithm successfully discriminated all four stimulus frequencies whilst correlating them with their respective P300 event markers. Classification accuracy was evaluated based on correct task intention recognition. The proposed hybrid system achieved a mean classification accuracy of 86.25%, coupled with an average ITR of 42.08 bits per minute (bpm). 

---
# Impact of Phonetics on Speaker Identity in Adversarial Voice Attack 

**Authors**: Daniyal Kabir Dar, Qiben Yan, Li Xiao, Arun Ross  

**Link**: [PDF](https://arxiv.org/pdf/2509.15437)  

**Abstract**: Adversarial perturbations in speech pose a serious threat to automatic speech recognition (ASR) and speaker verification by introducing subtle waveform modifications that remain imperceptible to humans but can significantly alter system outputs. While targeted attacks on end-to-end ASR models have been widely studied, the phonetic basis of these perturbations and their effect on speaker identity remain underexplored. In this work, we analyze adversarial audio at the phonetic level and show that perturbations exploit systematic confusions such as vowel centralization and consonant substitutions. These distortions not only mislead transcription but also degrade phonetic cues critical for speaker verification, leading to identity drift. Using DeepSpeech as our ASR target, we generate targeted adversarial examples and evaluate their impact on speaker embeddings across genuine and impostor samples. Results across 16 phonetically diverse target phrases demonstrate that adversarial audio induces both transcription errors and identity drift, highlighting the need for phonetic-aware defenses to ensure the robustness of ASR and speaker recognition systems. 

---
# Region-Aware Deformable Convolutions 

**Authors**: Abolfazl Saheban Maleki, Maryam Imani  

**Link**: [PDF](https://arxiv.org/pdf/2509.15436)  

**Abstract**: We introduce Region-Aware Deformable Convolution (RAD-Conv), a new convolutional operator that enhances neural networks' ability to adapt to complex image structures. Unlike traditional deformable convolutions, which are limited to fixed quadrilateral sampling areas, RAD-Conv uses four boundary offsets per kernel element to create flexible, rectangular regions that dynamically adjust their size and shape to match image content. This approach allows precise control over the receptive field's width and height, enabling the capture of both local details and long-range dependencies, even with small 1x1 kernels. By decoupling the receptive field's shape from the kernel's structure, RAD-Conv combines the adaptability of attention mechanisms with the efficiency of standard convolutions. This innovative design offers a practical solution for building more expressive and efficient vision models, bridging the gap between rigid convolutional architectures and computationally costly attention-based methods. 

---
# ORCA: Agentic Reasoning For Hallucination and Adversarial Robustness in Vision-Language Models 

**Authors**: Chung-En Johnny Yu, Hsuan-Chih, Chen, Brian Jalaian, Nathaniel D. Bastian  

**Link**: [PDF](https://arxiv.org/pdf/2509.15435)  

**Abstract**: Large Vision-Language Models (LVLMs) exhibit strong multimodal capabilities but remain vulnerable to hallucinations from intrinsic errors and adversarial attacks from external exploitations, limiting their reliability in real-world applications. We present ORCA, an agentic reasoning framework that improves the factual accuracy and adversarial robustness of pretrained LVLMs through test-time structured inference reasoning with a suite of small vision models (less than 3B parameters). ORCA operates via an Observe--Reason--Critique--Act loop, querying multiple visual tools with evidential questions, validating cross-model inconsistencies, and refining predictions iteratively without access to model internals or retraining. ORCA also stores intermediate reasoning traces, which supports auditable decision-making. Though designed primarily to mitigate object-level hallucinations, ORCA also exhibits emergent adversarial robustness without requiring adversarial training or defense mechanisms. We evaluate ORCA across three settings: (1) clean images on hallucination benchmarks, (2) adversarially perturbed images without defense, and (3) adversarially perturbed images with defense applied. On the POPE hallucination benchmark, ORCA improves standalone LVLM performance by +3.64\% to +40.67\% across different subsets. Under adversarial perturbations on POPE, ORCA achieves an average accuracy gain of +20.11\% across LVLMs. When combined with defense techniques on adversarially perturbed AMBER images, ORCA further improves standalone LVLM performance, with gains ranging from +1.20\% to +48.00\% across evaluation metrics. These results demonstrate that ORCA offers a promising path toward building more reliable and robust multimodal systems. 

---
# Deep learning and abstractive summarisation for radiological reports: an empirical study for adapting the PEGASUS models' family with scarce data 

**Authors**: Claudio Benzoni, Martina Langhals, Martin Boeker, Luise Modersohn, Máté E. Maros  

**Link**: [PDF](https://arxiv.org/pdf/2509.15419)  

**Abstract**: Regardless of the rapid development of artificial intelligence, abstractive summarisation is still challenging for sensitive and data-restrictive domains like medicine. With the increasing number of imaging, the relevance of automated tools for complex medical text summarisation is expected to become highly relevant. In this paper, we investigated the adaptation via fine-tuning process of a non-domain-specific abstractive summarisation encoder-decoder model family, and gave insights to practitioners on how to avoid over- and underfitting. We used PEGASUS and PEGASUS-X, on a medium-sized radiological reports public dataset. For each model, we comprehensively evaluated two different checkpoints with varying sizes of the same training data. We monitored the models' performances with lexical and semantic metrics during the training history on the fixed-size validation set. PEGASUS exhibited different phases, which can be related to epoch-wise double-descent, or peak-drop-recovery behaviour. For PEGASUS-X, we found that using a larger checkpoint led to a performance detriment. This work highlights the challenges and risks of fine-tuning models with high expressivity when dealing with scarce training data, and lays the groundwork for future investigations into more robust fine-tuning strategies for summarisation models in specialised domains. 

---
# Exploring multimodal implicit behavior learning for vehicle navigation in simulated cities 

**Authors**: Eric Aislan Antonelo, Gustavo Claudio Karl Couto, Christian Möller  

**Link**: [PDF](https://arxiv.org/pdf/2509.15400)  

**Abstract**: Standard Behavior Cloning (BC) fails to learn multimodal driving decisions, where multiple valid actions exist for the same scenario. We explore Implicit Behavioral Cloning (IBC) with Energy-Based Models (EBMs) to better capture this multimodality. We propose Data-Augmented IBC (DA-IBC), which improves learning by perturbing expert actions to form the counterexamples of IBC training and using better initialization for derivative-free inference. Experiments in the CARLA simulator with Bird's-Eye View inputs demonstrate that DA-IBC outperforms standard IBC in urban driving tasks designed to evaluate multimodal behavior learning in a test environment. The learned energy landscapes are able to represent multimodal action distributions, which BC fails to achieve. 

---
# Generating Part-Based Global Explanations Via Correspondence 

**Authors**: Kunal Rathore, Prasad Tadepalli  

**Link**: [PDF](https://arxiv.org/pdf/2509.15393)  

**Abstract**: Deep learning models are notoriously opaque. Existing explanation methods often focus on localized visual explanations for individual images. Concept-based explanations, while offering global insights, require extensive annotations, incurring significant labeling cost. We propose an approach that leverages user-defined part labels from a limited set of images and efficiently transfers them to a larger dataset. This enables the generation of global symbolic explanations by aggregating part-based local explanations, ultimately providing human-understandable explanations for model decisions on a large scale. 

---
# Efficient and Versatile Model for Multilingual Information Retrieval of Islamic Text: Development and Deployment in Real-World Scenarios 

**Authors**: Vera Pavlova, Mohammed Makhlouf  

**Link**: [PDF](https://arxiv.org/pdf/2509.15380)  

**Abstract**: Despite recent advancements in Multilingual Information Retrieval (MLIR), a significant gap remains between research and practical deployment. Many studies assess MLIR performance in isolated settings, limiting their applicability to real-world scenarios. In this work, we leverage the unique characteristics of the Quranic multilingual corpus to examine the optimal strategies to develop an ad-hoc IR system for the Islamic domain that is designed to satisfy users' information needs in multiple languages. We prepared eleven retrieval models employing four training approaches: monolingual, cross-lingual, translate-train-all, and a novel mixed method combining cross-lingual and monolingual techniques. Evaluation on an in-domain dataset demonstrates that the mixed approach achieves promising results across diverse retrieval scenarios. Furthermore, we provide a detailed analysis of how different training configurations affect the embedding space and their implications for multilingual retrieval effectiveness. Finally, we discuss deployment considerations, emphasizing the cost-efficiency of deploying a single versatile, lightweight model for real-world MLIR applications. 

---
# Recent Advancements in Microscopy Image Enhancement using Deep Learning: A Survey 

**Authors**: Debasish Dutta, Neeharika Sonowal, Risheraj Barauh, Deepjyoti Chetia, Sanjib Kr Kalita  

**Link**: [PDF](https://arxiv.org/pdf/2509.15363)  

**Abstract**: Microscopy image enhancement plays a pivotal role in understanding the details of biological cells and materials at microscopic scales. In recent years, there has been a significant rise in the advancement of microscopy image enhancement, specifically with the help of deep learning methods. This survey paper aims to provide a snapshot of this rapidly growing state-of-the-art method, focusing on its evolution, applications, challenges, and future directions. The core discussions take place around the key domains of microscopy image enhancement of super-resolution, reconstruction, and denoising, with each domain explored in terms of its current trends and their practical utility of deep learning. 

---
# Beyond Spurious Signals: Debiasing Multimodal Large Language Models via Counterfactual Inference and Adaptive Expert Routing 

**Authors**: Zichen Wu, Hsiu-Yuan Huang, Yunfang Wu  

**Link**: [PDF](https://arxiv.org/pdf/2509.15361)  

**Abstract**: Multimodal Large Language Models (MLLMs) have shown substantial capabilities in integrating visual and textual information, yet frequently rely on spurious correlations, undermining their robustness and generalization in complex multimodal reasoning tasks. This paper addresses the critical challenge of superficial correlation bias in MLLMs through a novel causal mediation-based debiasing framework. Specially, we distinguishing core semantics from spurious textual and visual contexts via counterfactual examples to activate training-stage debiasing and employ a Mixture-of-Experts (MoE) architecture with dynamic routing to selectively engages modality-specific debiasing experts. Empirical evaluation on multimodal sarcasm detection and sentiment analysis tasks demonstrates that our framework significantly surpasses unimodal debiasing strategies and existing state-of-the-art models. 

---
# Emulating Human-like Adaptive Vision for Efficient and Flexible Machine Visual Perception 

**Authors**: Yulin Wang, Yang Yue, Yang Yue, Huanqian Wang, Haojun Jiang, Yizeng Han, Zanlin Ni, Yifan Pu, Minglei Shi, Rui Lu, Qisen Yang, Andrew Zhao, Zhuofan Xia, Shiji Song, Gao Huang  

**Link**: [PDF](https://arxiv.org/pdf/2509.15333)  

**Abstract**: Human vision is highly adaptive, efficiently sampling intricate environments by sequentially fixating on task-relevant regions. In contrast, prevailing machine vision models passively process entire scenes at once, resulting in excessive resource demands scaling with spatial-temporal input resolution and model size, yielding critical limitations impeding both future advancements and real-world application. Here we introduce AdaptiveNN, a general framework aiming to drive a paradigm shift from 'passive' to 'active, adaptive' vision models. AdaptiveNN formulates visual perception as a coarse-to-fine sequential decision-making process, progressively identifying and attending to regions pertinent to the task, incrementally combining information across fixations, and actively concluding observation when sufficient. We establish a theory integrating representation learning with self-rewarding reinforcement learning, enabling end-to-end training of the non-differentiable AdaptiveNN without additional supervision on fixation locations. We assess AdaptiveNN on 17 benchmarks spanning 9 tasks, including large-scale visual recognition, fine-grained discrimination, visual search, processing images from real driving and medical scenarios, language-driven embodied AI, and side-by-side comparisons with humans. AdaptiveNN achieves up to 28x inference cost reduction without sacrificing accuracy, flexibly adapts to varying task demands and resource budgets without retraining, and provides enhanced interpretability via its fixation patterns, demonstrating a promising avenue toward efficient, flexible, and interpretable computer vision. Furthermore, AdaptiveNN exhibits closely human-like perceptual behaviors in many cases, revealing its potential as a valuable tool for investigating visual cognition. Code is available at this https URL. 

---
# Collective Voice: Recovered-Peer Support Mediated by An LLM-Based Chatbot for Eating Disorder Recovery 

**Authors**: Ryuhaerang Choi, Taehan Kim, Subin Park, Seohyeon Yoo, Jennifer G. Kim, Sung-Ju Lee  

**Link**: [PDF](https://arxiv.org/pdf/2509.15289)  

**Abstract**: Peer recovery narratives provide unique benefits beyond professional or lay mentoring by fostering hope and sustained recovery in eating disorder (ED) contexts. Yet, such support is limited by the scarcity of peer-involved programs and potential drawbacks on recovered peers, including relapse risk. To address this, we designed RecoveryTeller, a chatbot adopting a recovered-peer persona that portrays itself as someone recovered from an ED. We examined whether such a persona can reproduce the support affordances of peer recovery narratives. We compared RecoveryTeller with a lay-mentor persona chatbot offering similar guidance but without a recovery background. We conducted a 20-day cross-over deployment study with 26 ED participants, each using both chatbots for 10 days. RecoveryTeller elicited stronger emotional resonance than a lay-mentor chatbot, yet tensions between emotional and epistemic trust led participants to view the two personas as complementary rather than substitutes. We provide design implications for mental health chatbot persona design. 

---
# Evaluating the Limitations of Local LLMs in Solving Complex Programming Challenges 

**Authors**: Kadin Matotek, Heather Cassel, Md Amiruzzaman, Linh B. Ngo  

**Link**: [PDF](https://arxiv.org/pdf/2509.15283)  

**Abstract**: This study examines the performance of today's open-source, locally hosted large-language models (LLMs) in handling complex competitive programming tasks with extended problem descriptions and contexts. Building on the original Framework for AI-driven Code Generation Evaluation (FACE), the authors retrofit the pipeline to work entirely offline through the Ollama runtime, collapsing FACE's sprawling per-problem directory tree into a handful of consolidated JSON files, and adding robust checkpointing so multi-day runs can resume after failures. The enhanced framework generates, submits, and records solutions for the full Kattis corpus of 3,589 problems across eight code-oriented models ranging from 6.7-9 billion parameters. The submission results show that the overall pass@1 accuracy is modest for the local models, with the best models performing at approximately half the acceptance rate of the proprietary models, Gemini 1.5 and ChatGPT-4. These findings expose a persistent gap between private, cost-controlled LLM deployments and state-of-the-art proprietary services, yet also highlight the rapid progress of open models and the practical benefits of an evaluation workflow that organizations can replicate on in-house hardware. 

---
# Partial Column Generation with Graph Neural Networks for Team Formation and Routing 

**Authors**: Giacomo Dall'Olio, Rainer Kolisch, Yaoxin Wu  

**Link**: [PDF](https://arxiv.org/pdf/2509.15275)  

**Abstract**: The team formation and routing problem is a challenging optimization problem with several real-world applications in fields such as airport, healthcare, and maintenance operations. To solve this problem, exact solution methods based on column generation have been proposed in the literature. In this paper, we propose a novel partial column generation strategy for settings with multiple pricing problems, based on predicting which ones are likely to yield columns with a negative reduced cost. We develop a machine learning model tailored to the team formation and routing problem that leverages graph neural networks for these predictions. Computational experiments demonstrate that applying our strategy enhances the solution method and outperforms traditional partial column generation approaches from the literature, particularly on hard instances solved under a tight time limit. 

---
# Large Vision Models Can Solve Mental Rotation Problems 

**Authors**: Sebastian Ray Mason, Anders Gjølbye, Phillip Chavarria Højbjerg, Lenka Tětková, Lars Kai Hansen  

**Link**: [PDF](https://arxiv.org/pdf/2509.15271)  

**Abstract**: Mental rotation is a key test of spatial reasoning in humans and has been central to understanding how perception supports cognition. Despite the success of modern vision transformers, it is still unclear how well these models develop similar abilities. In this work, we present a systematic evaluation of ViT, CLIP, DINOv2, and DINOv3 across a range of mental-rotation tasks, from simple block structures similar to those used by Shepard and Metzler to study human cognition, to more complex block figures, three types of text, and photo-realistic objects. By probing model representations layer by layer, we examine where and how these networks succeed. We find that i) self-supervised ViTs capture geometric structure better than supervised ViTs; ii) intermediate layers perform better than final layers; iii) task difficulty increases with rotation complexity and occlusion, mirroring human reaction times and suggesting similar constraints in embedding space representations. 

---
# PRISM: Phase-enhanced Radial-based Image Signature Mapping framework for fingerprinting AI-generated images 

**Authors**: Emanuele Ricco, Elia Onofri, Lorenzo Cima, Stefano Cresci, Roberto Di Pietro  

**Link**: [PDF](https://arxiv.org/pdf/2509.15270)  

**Abstract**: A critical need has emerged for generative AI: attribution methods. That is, solutions that can identify the model originating AI-generated content. This feature, generally relevant in multimodal applications, is especially sensitive in commercial settings where users subscribe to paid proprietary services and expect guarantees about the source of the content they receive. To address these issues, we introduce PRISM, a scalable Phase-enhanced Radial-based Image Signature Mapping framework for fingerprinting AI-generated images. PRISM is based on a radial reduction of the discrete Fourier transform that leverages amplitude and phase information to capture model-specific signatures. The output of the above process is subsequently clustered via linear discriminant analysis to achieve reliable model attribution in diverse settings, even if the model's internal details are inaccessible. To support our work, we construct PRISM-36K, a novel dataset of 36,000 images generated by six text-to-image GAN- and diffusion-based models. On this dataset, PRISM achieves an attribution accuracy of 92.04%. We additionally evaluate our method on four benchmarks from the literature, reaching an average accuracy of 81.60%. Finally, we evaluate our methodology also in the binary task of detecting real vs fake images, achieving an average accuracy of 88.41%. We obtain our best result on GenImage with an accuracy of 95.06%, whereas the original benchmark achieved 82.20%. Our results demonstrate the effectiveness of frequency-domain fingerprinting for cross-architecture and cross-dataset model attribution, offering a viable solution for enforcing accountability and trust in generative AI systems. 

---
# Modeling Transformers as complex networks to analyze learning dynamics 

**Authors**: Elisabetta Rocchetti  

**Link**: [PDF](https://arxiv.org/pdf/2509.15269)  

**Abstract**: The process by which Large Language Models (LLMs) acquire complex capabilities during training remains a key open question in mechanistic interpretability. This project investigates whether these learning dynamics can be characterized through the lens of Complex Network Theory (CNT). I introduce a novel methodology to represent a Transformer-based LLM as a directed, weighted graph where nodes are the model's computational components (attention heads and MLPs) and edges represent causal influence, measured via an intervention-based ablation technique. By tracking the evolution of this component-graph across 143 training checkpoints of the Pythia-14M model on a canonical induction task, I analyze a suite of graph-theoretic metrics. The results reveal that the network's structure evolves through distinct phases of exploration, consolidation, and refinement. Specifically, I identify the emergence of a stable hierarchy of information spreader components and a dynamic set of information gatherer components, whose roles reconfigure at key learning junctures. This work demonstrates that a component-level network perspective offers a powerful macroscopic lens for visualizing and understanding the self-organizing principles that drive the formation of functional circuits in LLMs. 

---
# Autoguided Online Data Curation for Diffusion Model Training 

**Authors**: Valeria Pais, Luis Oala, Daniele Faccio, Marco Aversa  

**Link**: [PDF](https://arxiv.org/pdf/2509.15267)  

**Abstract**: The costs of generative model compute rekindled promises and hopes for efficient data curation. In this work, we investigate whether recently developed autoguidance and online data selection methods can improve the time and sample efficiency of training generative diffusion models. We integrate joint example selection (JEST) and autoguidance into a unified code base for fast ablation and benchmarking. We evaluate combinations of data curation on a controlled 2-D synthetic data generation task as well as (3x64x64)-D image generation. Our comparisons are made at equal wall-clock time and equal number of samples, explicitly accounting for the overhead of selection. Across experiments, autoguidance consistently improves sample quality and diversity. Early AJEST (applying selection only at the beginning of training) can match or modestly exceed autoguidance alone in data efficiency on both tasks. However, its time overhead and added complexity make autoguidance or uniform random data selection preferable in most situations. These findings suggest that while targeted online selection can yield efficiency gains in early training, robust sample quality improvements are primarily driven by autoguidance. We discuss limitations and scope, and outline when data selection may be beneficial. 

---
# IEFS-GMB: Gradient Memory Bank-Guided Feature Selection Based on Information Entropy for EEG Classification of Neurological Disorders 

**Authors**: Liang Zhang, Hanyang Dong, Jia-Hong Gao, Yi Sun, Kuntao Xiao, Wanli Yang, Zhao Lv, Shurong Sheng  

**Link**: [PDF](https://arxiv.org/pdf/2509.15259)  

**Abstract**: Deep learning-based EEG classification is crucial for the automated detection of neurological disorders, improving diagnostic accuracy and enabling early intervention. However, the low signal-to-noise ratio of EEG signals limits model performance, making feature selection (FS) vital for optimizing representations learned by neural network encoders. Existing FS methods are seldom designed specifically for EEG diagnosis; many are architecture-dependent and lack interpretability, limiting their applicability. Moreover, most rely on single-iteration data, resulting in limited robustness to variability. To address these issues, we propose IEFS-GMB, an Information Entropy-based Feature Selection method guided by a Gradient Memory Bank. This approach constructs a dynamic memory bank storing historical gradients, computes feature importance via information entropy, and applies entropy-based weighting to select informative EEG features. Experiments on four public neurological disease datasets show that encoders enhanced with IEFS-GMB achieve accuracy improvements of 0.64% to 6.45% over baseline models. The method also outperforms four competing FS techniques and improves model interpretability, supporting its practical use in clinical settings. 

---
# Generative AI Meets Wireless Sensing: Towards Wireless Foundation Model 

**Authors**: Zheng Yang, Guoxuan Chi, Chenshu Wu, Hanyu Liu, Yuchong Gao, Yunhao Liu, Jie Xu, Tony Xiao Han  

**Link**: [PDF](https://arxiv.org/pdf/2509.15258)  

**Abstract**: Generative Artificial Intelligence (GenAI) has made significant advancements in fields such as computer vision (CV) and natural language processing (NLP), demonstrating its capability to synthesize high-fidelity data and improve generalization. Recently, there has been growing interest in integrating GenAI into wireless sensing systems. By leveraging generative techniques such as data augmentation, domain adaptation, and denoising, wireless sensing applications, including device localization, human activity recognition, and environmental monitoring, can be significantly improved. This survey investigates the convergence of GenAI and wireless sensing from two complementary perspectives. First, we explore how GenAI can be integrated into wireless sensing pipelines, focusing on two modes of integration: as a plugin to augment task-specific models and as a solver to directly address sensing tasks. Second, we analyze the characteristics of mainstream generative models, such as Generative Adversarial Networks (GANs), Variational Autoencoders (VAEs), and diffusion models, and discuss their applicability and unique advantages across various wireless sensing tasks. We further identify key challenges in applying GenAI to wireless sensing and outline a future direction toward a wireless foundation model: a unified, pre-trained design capable of scalable, adaptable, and efficient signal understanding across diverse sensing tasks. 

---
# A Multi-Scale Graph Neural Process with Cross-Drug Co-Attention for Drug-Drug Interactions Prediction 

**Authors**: Zimo Yan, Jie Zhang, Zheng Xie, Yiping Song, Hao Li  

**Link**: [PDF](https://arxiv.org/pdf/2509.15256)  

**Abstract**: Accurate prediction of drug-drug interactions (DDI) is crucial for medication safety and effective drug development. However, existing methods often struggle to capture structural information across different scales, from local functional groups to global molecular topology, and typically lack mechanisms to quantify prediction confidence. To address these limitations, we propose MPNP-DDI, a novel Multi-scale Graph Neural Process framework. The core of MPNP-DDI is a unique message-passing scheme that, by being iteratively applied, learns a hierarchy of graph representations at multiple scales. Crucially, a cross-drug co-attention mechanism then dynamically fuses these multi-scale representations to generate context-aware embeddings for interacting drug pairs, while an integrated neural process module provides principled uncertainty estimation. Extensive experiments demonstrate that MPNP-DDI significantly outperforms state-of-the-art baselines on benchmark datasets. By providing accurate, generalizable, and uncertainty-aware predictions built upon multi-scale structural features, MPNP-DDI represents a powerful computational tool for pharmacovigilance, polypharmacy risk assessment, and precision medicine. 

---
# Emotion-Aware Speech Generation with Character-Specific Voices for Comics 

**Authors**: Zhiwen Qian, Jinhua Liang, Huan Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2509.15253)  

**Abstract**: This paper presents an end-to-end pipeline for generating character-specific, emotion-aware speech from comics. The proposed system takes full comic volumes as input and produces speech aligned with each character's dialogue and emotional state. An image processing module performs character detection, text recognition, and emotion intensity recognition. A large language model performs dialogue attribution and emotion analysis by integrating visual information with the evolving plot context. Speech is synthesized through a text-to-speech model with distinct voice profiles tailored to each character and emotion. This work enables automated voiceover generation for comics, offering a step toward interactive and immersive comic reading experience. 

---
# Walk and Read Less: Improving the Efficiency of Vision-and-Language Navigation via Tuning-Free Multimodal Token Pruning 

**Authors**: Wenda Qin, Andrea Burns, Bryan A. Plummer, Margrit Betke  

**Link**: [PDF](https://arxiv.org/pdf/2509.15250)  

**Abstract**: Large models achieve strong performance on Vision-and-Language Navigation (VLN) tasks, but are costly to run in resource-limited environments. Token pruning offers appealing tradeoffs for efficiency with minimal performance loss by reducing model input size, but prior work overlooks VLN-specific challenges. For example, information loss from pruning can effectively increase computational cost due to longer walks. Thus, the inability to identify uninformative tokens undermines the supposed efficiency gains from pruning. To address this, we propose Navigation-Aware Pruning (NAP), which uses navigation-specific traits to simplify the pruning process by pre-filtering tokens into foreground and background. For example, image views are filtered based on whether the agent can navigate in that direction. We also extract navigation-relevant instructions using a Large Language Model. After filtering, we focus pruning on background tokens, minimizing information loss. To further help avoid increases in navigation length, we discourage backtracking by removing low-importance navigation nodes. Experiments on standard VLN benchmarks show NAP significantly outperforms prior work, preserving higher success rates while saving more than 50% FLOPS. 

---
# Causal Reasoning Elicits Controllable 3D Scene Generation 

**Authors**: Shen Chen, Ruiyu Zhao, Jiale Zhou, Zongkai Wu, Jenq-Neng Hwang, Lei Li  

**Link**: [PDF](https://arxiv.org/pdf/2509.15249)  

**Abstract**: Existing 3D scene generation methods often struggle to model the complex logical dependencies and physical constraints between objects, limiting their ability to adapt to dynamic and realistic environments. We propose CausalStruct, a novel framework that embeds causal reasoning into 3D scene generation. Utilizing large language models (LLMs), We construct causal graphs where nodes represent objects and attributes, while edges encode causal dependencies and physical constraints. CausalStruct iteratively refines the scene layout by enforcing causal order to determine the placement order of objects and applies causal intervention to adjust the spatial configuration according to physics-driven constraints, ensuring consistency with textual descriptions and real-world dynamics. The refined scene causal graph informs subsequent optimization steps, employing a Proportional-Integral-Derivative(PID) controller to iteratively tune object scales and positions. Our method uses text or images to guide object placement and layout in 3D scenes, with 3D Gaussian Splatting and Score Distillation Sampling improving shape accuracy and rendering stability. Extensive experiments show that CausalStruct generates 3D scenes with enhanced logical coherence, realistic spatial interactions, and robust adaptability. 

---
# Synthetic bootstrapped pretraining 

**Authors**: Zitong Yang, Aonan Zhang, Hong Liu, Tatsunori Hashimoto, Emmanuel Candès, Chong Wang, Ruoming Pang  

**Link**: [PDF](https://arxiv.org/pdf/2509.15248)  

**Abstract**: We introduce Synthetic Bootstrapped Pretraining (SBP), a language model (LM) pretraining procedure that first learns a model of relations between documents from the pretraining dataset and then leverages it to synthesize a vast new corpus for joint training. While the standard pretraining teaches LMs to learn causal correlations among tokens within a single document, it is not designed to efficiently model the rich, learnable inter-document correlations that can potentially lead to better performance. We validate SBP by designing a compute-matched pretraining setup and pretrain a 3B-parameter model on up to 1T tokens from scratch. We find SBP consistently improves upon a strong repetition baseline and delivers a significant fraction of performance improvement attainable by an oracle upper bound with access to 20x more unique data. Qualitative analysis reveals that the synthesized documents go beyond mere paraphrases -- SBP first abstracts a core concept from the seed material and then crafts a new narration on top of it. Besides strong empirical performance, SBP admits a natural Bayesian interpretation: the synthesizer implicitly learns to abstract the latent concepts shared between related documents. 

---
# GenCAD-3D: CAD Program Generation using Multimodal Latent Space Alignment and Synthetic Dataset Balancing 

**Authors**: Nomi Yu, Md Ferdous Alam, A. John Hart, Faez Ahmed  

**Link**: [PDF](https://arxiv.org/pdf/2509.15246)  

**Abstract**: CAD programs, structured as parametric sequences of commands that compile into precise 3D geometries, are fundamental to accurate and efficient engineering design processes. Generating these programs from nonparametric data such as point clouds and meshes remains a crucial yet challenging task, typically requiring extensive manual intervention. Current deep generative models aimed at automating CAD generation are significantly limited by imbalanced and insufficiently large datasets, particularly those lacking representation for complex CAD programs. To address this, we introduce GenCAD-3D, a multimodal generative framework utilizing contrastive learning for aligning latent embeddings between CAD and geometric encoders, combined with latent diffusion models for CAD sequence generation and retrieval. Additionally, we present SynthBal, a synthetic data augmentation strategy specifically designed to balance and expand datasets, notably enhancing representation of complex CAD geometries. Our experiments show that SynthBal significantly boosts reconstruction accuracy, reduces the generation of invalid CAD models, and markedly improves performance on high-complexity geometries, surpassing existing benchmarks. These advancements hold substantial implications for streamlining reverse engineering and enhancing automation in engineering design. We will publicly release our datasets and code, including a set of 51 3D-printed and laser-scanned parts on our project site. 

---
# Generating Plans for Belief-Desire-Intention (BDI) Agents Using Alternating-Time Temporal Logic (ATL) 

**Authors**: Dylan Léveillé  

**Link**: [PDF](https://arxiv.org/pdf/2509.15238)  

**Abstract**: Belief-Desire-Intention (BDI) is a framework for modelling agents based on their beliefs, desires, and intentions. Plans are a central component of BDI agents, and define sequences of actions that an agent must undertake to achieve a certain goal. Existing approaches to plan generation often require significant manual effort, and are mainly focused on single-agent systems. As a result, in this work, we have developed a tool that automatically generates BDI plans using Alternating-Time Temporal Logic (ATL). By using ATL, the plans generated accommodate for possible competition or cooperation between the agents in the system. We demonstrate the effectiveness of the tool by generating plans for an illustrative game that requires agent collaboration to achieve a shared goal. We show that the generated plans allow the agents to successfully attain this goal. 

---
# ChannelFlow-Tools: A Standardized Dataset Creation Pipeline for 3D Obstructed Channel Flows 

**Authors**: Shubham Kavane, Kajol Kulkarni, Harald Koestler  

**Link**: [PDF](https://arxiv.org/pdf/2509.15236)  

**Abstract**: We present ChannelFlow-Tools, a configuration-driven framework that standardizes the end-to-end path from programmatic CAD solid generation to ML-ready inputs and targets for 3D obstructed channel flows. The toolchain integrates geometry synthesis with feasibility checks, signed distance field (SDF) voxelization, automated solver orchestration on HPC (waLBerla LBM), and Cartesian resampling to co-registered multi-resolution tensors. A single Hydra/OmegaConf configuration governs all stages, enabling deterministic reproduction and controlled ablations. As a case study, we generate 10k+ scenes spanning Re=100-15000 with diverse shapes and poses. An end-to-end evaluation of storage trade-offs directly from the emitted artifacts, a minimal 3D U-Net at 128x32x32, and example surrogate models with dataset size illustrate that the standardized representations support reproducible ML training. ChannelFlow-Tools turns one-off dataset creation into a reproducible, configurable pipeline for CFD surrogate modeling. 

---
# Pre-Forgettable Models: Prompt Learning as a Native Mechanism for Unlearning 

**Authors**: Rutger Hendrix, Giovanni Patanè, Leonardo G. Russo, Simone Carnemolla, Giovanni Bellitto, Federica Proietto Salanitri, Concetto Spampinato, Matteo Pennisi  

**Link**: [PDF](https://arxiv.org/pdf/2509.15230)  

**Abstract**: Foundation models have transformed multimedia analysis by enabling robust and transferable representations across diverse modalities and tasks. However, their static deployment conflicts with growing societal and regulatory demands -- particularly the need to unlearn specific data upon request, as mandated by privacy frameworks such as the GDPR. Traditional unlearning approaches, including retraining, activation editing, or distillation, are often computationally expensive, fragile, and ill-suited for real-time or continuously evolving systems. In this paper, we propose a paradigm shift: rethinking unlearning not as a retroactive intervention but as a built-in capability. We introduce a prompt-based learning framework that unifies knowledge acquisition and removal within a single training phase. Rather than encoding information in model weights, our approach binds class-level semantics to dedicated prompt tokens. This design enables instant unlearning simply by removing the corresponding prompt -- without retraining, model modification, or access to original data. Experiments demonstrate that our framework preserves predictive performance on retained classes while effectively erasing forgotten ones. Beyond utility, our method exhibits strong privacy and security guarantees: it is resistant to membership inference attacks, and prompt removal prevents any residual knowledge extraction, even under adversarial conditions. This ensures compliance with data protection principles and safeguards against unauthorized access to forgotten information, making the framework suitable for deployment in sensitive and regulated environments. Overall, by embedding removability into the architecture itself, this work establishes a new foundation for designing modular, scalable and ethically responsive AI models. 

---
