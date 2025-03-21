# Deconstructing Long Chain-of-Thought: A Structured Reasoning Optimization Framework for Long CoT Distillation 

**Authors**: Yijia Luo, Yulin Song, Xingyao Zhang, Jiaheng Liu, Weixun Wang, GengRu Chen, Wenbo Su, Bo Zheng  

**Link**: [PDF](https://arxiv.org/pdf/2503.16385)  

**Abstract**: Recent advancements in large language models (LLMs) have demonstrated remarkable reasoning capabilities through long chain-of-thought (CoT) reasoning. The R1 distillation scheme has emerged as a promising approach for training cost-effective models with enhanced reasoning abilities. However, the underlying mechanisms driving its effectiveness remain unclear. This study examines the universality of distillation data and identifies key components that enable the efficient transfer of long-chain reasoning capabilities in LLM distillation. Our findings reveal that the effectiveness of long CoT reasoning distillation from teacher models like Qwen-QwQ degrades significantly on nonhomologous models, challenging the assumed universality of current distillation methods. To gain deeper insights into the structure and patterns of long CoT reasoning, we propose DLCoT (Deconstructing Long Chain-of-Thought), a distillation data enhancement framework. DLCoT consists of three key steps: (1) data segmentation to decompose complex long CoT structures, (2) simplification by eliminating unsolvable and redundant solutions, and (3) optimization of intermediate error states. Our approach significantly improves model performance and token efficiency, facilitating the development of high-performance LLMs. 

---
# OmniGeo: Towards a Multimodal Large Language Models for Geospatial Artificial Intelligence 

**Authors**: Long Yuan, Fengran Mo, Kaiyu Huang, Wenjie Wang, Wangyuxuan Zhai, Xiaoyu Zhu, You Li, Jinan Xu, Jian-Yun Nie  

**Link**: [PDF](https://arxiv.org/pdf/2503.16326)  

**Abstract**: The rapid advancement of multimodal large language models (LLMs) has opened new frontiers in artificial intelligence, enabling the integration of diverse large-scale data types such as text, images, and spatial information. In this paper, we explore the potential of multimodal LLMs (MLLM) for geospatial artificial intelligence (GeoAI), a field that leverages spatial data to address challenges in domains including Geospatial Semantics, Health Geography, Urban Geography, Urban Perception, and Remote Sensing. We propose a MLLM (OmniGeo) tailored to geospatial applications, capable of processing and analyzing heterogeneous data sources, including satellite imagery, geospatial metadata, and textual descriptions. By combining the strengths of natural language understanding and spatial reasoning, our model enhances the ability of instruction following and the accuracy of GeoAI systems. Results demonstrate that our model outperforms task-specific models and existing LLMs on diverse geospatial tasks, effectively addressing the multimodality nature while achieving competitive results on the zero-shot geospatial tasks. Our code will be released after publication. 

---
# GreenIQ: A Deep Search Platform for Comprehensive Carbon Market Analysis and Automated Report Generation 

**Authors**: Bisola Faith Kayode, Akinyemi Sadeeq Akintola, Oluwole Fagbohun, Egonna Anaesiuba-Bristol, Onyekachukwu Ojumah, Oluwagbade Odimayo, Toyese Oloyede, Aniema Inyang, Teslim Kazeem, Habeeb Alli, Udodirim Ibem Offia, Prisca Chinazor Amajuoyi  

**Link**: [PDF](https://arxiv.org/pdf/2503.16041)  

**Abstract**: This study introduces GreenIQ, an AI-powered deep search platform designed to revolutionise carbon market intelligence through autonomous analysis and automated report generation. Carbon markets operate across diverse regulatory landscapes, generating vast amounts of heterogeneous data from policy documents, industry reports, academic literature, and real-time trading platforms. Traditional research approaches remain labour-intensive, slow, and difficult to scale. GreenIQ addresses these limitations through a multi-agent architecture powered by Large Language Models (LLMs), integrating five specialised AI agents: a Main Researcher Agent for intelligent information retrieval, a Report Writing Agent for structured synthesis, a Final Reviewer Agent for accuracy verification, a Data Visualisation Agent for enhanced interpretability, and a Translator Agent for multilingual adaptation. The system achieves seamless integration of structured and unstructured information with AI-driven citation verification, ensuring high transparency and reliability. GreenIQ delivers a 99.2\% reduction in processing time and a 99.7\% cost reduction compared to traditional research methodologies. A novel AI persona-based evaluation framework involving 16 domain-specific AI personas highlights its superior cross-jurisdictional analytical capabilities and regulatory insight generation. GreenIQ sets new standards in AI-driven research synthesis, policy analysis, and sustainability finance by streamlining carbon market research. It offers an efficient and scalable framework for environmental and financial intelligence, enabling more accurate, timely, and cost-effective decision-making in complex regulatory landscapes 

---
# Attention Pruning: Automated Fairness Repair of Language Models via Surrogate Simulated Annealing 

**Authors**: Vishnu Asutosh Dasu, Md Rafi ur Rashid, Vipul Gupta, Saeid Tizpaz-Niari, Gang Tan  

**Link**: [PDF](https://arxiv.org/pdf/2503.15815)  

**Abstract**: This paper explores pruning attention heads as a post-processing bias mitigation method for large language models (LLMs). Modern AI systems such as LLMs are expanding into sensitive social contexts where fairness concerns become especially crucial. Since LLMs develop decision-making patterns by training on massive datasets of human-generated content, they naturally encode and perpetuate societal biases. While modifying training datasets and algorithms is expensive and requires significant resources; post-processing techniques-such as selectively deactivating neurons and attention heads in pre-trained LLMs-can provide feasible and effective approaches to improve fairness. However, identifying the optimal subset of parameters to prune presents a combinatorial challenge within LLMs' immense parameter space, requiring solutions that efficiently balance competing objectives across the frontiers of model fairness and utility.
To address the computational challenges, we explore a search-based program repair approach via randomized simulated annealing. Given the prohibitive evaluation costs in billion-parameter LLMs, we develop surrogate deep neural networks that efficiently model the relationship between attention head states (active/inactive) and their corresponding fairness/utility metrics. This allows us to perform optimization over the surrogate models and efficiently identify optimal subsets of attention heads for selective pruning rather than directly searching through the LLM parameter space. This paper introduces Attention Pruning, a fairness-aware surrogate simulated annealing approach to prune attention heads in LLMs that disproportionately contribute to bias while minimally impacting overall model utility. Our experiments show that Attention Pruning achieves up to $40\%$ reduction in gender bias and outperforms the state-of-the-art bias mitigation strategies. 

---
# Video-VoT-R1: An efficient video inference model integrating image packing and AoE architecture 

**Authors**: Cheng Li, Jiexiong Liu, Yixuan Chen, Yanqin Jia  

**Link**: [PDF](https://arxiv.org/pdf/2503.15807)  

**Abstract**: In the field of video-language pretraining, existing models face numerous challenges in terms of inference efficiency and multimodal data processing. This paper proposes a KunLunBaize-VoT-R1 video inference model based on a long-sequence image encoder, along with its training and application methods. By integrating image packing technology, the Autonomy-of-Experts (AoE) architecture, and combining the video of Thought (VoT), a large language model (LLM) trained with large-scale reinforcement learning, and multiple training techniques, the efficiency and accuracy of the model in video inference tasks are effectively improved. Experiments show that this model performs outstandingly in multiple tests, providing a new solution for video-language understanding. 

---
# Advancing Mobile GUI Agents: A Verifier-Driven Approach to Practical Deployment 

**Authors**: Gaole Dai, Shiqi Jiang, Ting Cao, Yuanchun Li, Yuqing Yang, Rui Tan, Mo Li, Lili Qiu  

**Link**: [PDF](https://arxiv.org/pdf/2503.15937)  

**Abstract**: We propose V-Droid, a mobile GUI task automation agent. Unlike previous mobile agents that utilize Large Language Models (LLMs) as generators to directly generate actions at each step, V-Droid employs LLMs as verifiers to evaluate candidate actions before making final decisions. To realize this novel paradigm, we introduce a comprehensive framework for constructing verifier-driven mobile agents: the discretized action space construction coupled with the prefilling-only workflow to accelerate the verification process, the pair-wise progress preference training to significantly enhance the verifier's decision-making capabilities, and the scalable human-agent joint annotation scheme to efficiently collect the necessary data at scale. V-Droid sets a new state-of-the-art task success rate across several public mobile task automation benchmarks: 59.5% on AndroidWorld, 38.3% on AndroidLab, and 49% on MobileAgentBench, surpassing existing agents by 9.5%, 2.1%, and 9%, respectively. Furthermore, V-Droid achieves an impressively low latency of 0.7 seconds per step, making it the first mobile agent capable of delivering near-real-time, effective decision-making capabilities. 

---
# Reinforcement Learning Environment with LLM-Controlled Adversary in D&D 5th Edition Combat 

**Authors**: Joseph Emmanuel DL Dayo, Michel Onasis S. Ogbinar, Prospero C. Naval Jr  

**Link**: [PDF](https://arxiv.org/pdf/2503.15726)  

**Abstract**: The objective of this study is to design and implement a reinforcement learning (RL) environment using D\&D 5E combat scenarios to challenge smaller RL agents through interaction with a robust adversarial agent controlled by advanced Large Language Models (LLMs) like GPT-4o and LLaMA 3 8B. This research employs Deep Q-Networks (DQN) for the smaller agents, creating a testbed for strategic AI development that also serves as an educational tool by simulating dynamic and unpredictable combat scenarios. We successfully integrated sophisticated language models into the RL framework, enhancing strategic decision-making processes. Our results indicate that while RL agents generally outperform LLM-controlled adversaries in standard metrics, the strategic depth provided by LLMs significantly enhances the overall AI capabilities in this complex, rule-based setting. The novelty of our approach and its implications for mastering intricate environments and developing adaptive strategies are discussed, alongside potential innovations in AI-driven interactive simulations. This paper aims to demonstrate how integrating LLMs can create more robust and adaptable AI systems, providing valuable insights for further research and educational applications. 

---
# R$^2$: A LLM Based Novel-to-Screenplay Generation Framework with Causal Plot Graphs 

**Authors**: Zefeng Lin, Yi Xiao, Zhiqiang Mo, Qifan Zhang, Jie Wang, Jiayang Chen, Jiajing Zhang, Hui Zhang, Zhengyi Liu, Xianyong Fang, Xiaohua Xu  

**Link**: [PDF](https://arxiv.org/pdf/2503.15655)  

**Abstract**: Automatically adapting novels into screenplays is important for the TV, film, or opera industries to promote products with low costs. The strong performances of large language models (LLMs) in long-text generation call us to propose a LLM based framework Reader-Rewriter (R$^2$) for this task. However, there are two fundamental challenges here. First, the LLM hallucinations may cause inconsistent plot extraction and screenplay generation. Second, the causality-embedded plot lines should be effectively extracted for coherent rewriting. Therefore, two corresponding tactics are proposed: 1) A hallucination-aware refinement method (HAR) to iteratively discover and eliminate the affections of hallucinations; and 2) a causal plot-graph construction method (CPC) based on a greedy cycle-breaking algorithm to efficiently construct plot lines with event causalities. Recruiting those efficient techniques, R$^2$ utilizes two modules to mimic the human screenplay rewriting process: The Reader module adopts a sliding window and CPC to build the causal plot graphs, while the Rewriter module generates first the scene outlines based on the graphs and then the screenplays. HAR is integrated into both modules for accurate inferences of LLMs. Experimental results demonstrate the superiority of R$^2$, which substantially outperforms three existing approaches (51.3%, 22.6%, and 57.1% absolute increases) in pairwise comparison at the overall win rate for GPT-4o. 

---
# Dialogic Learning in Child-Robot Interaction: A Hybrid Approach to Personalized Educational Content Generation 

**Authors**: Elena Malnatsky, Shenghui Wang, Koen V. Hindriks, Mike E.U. Ligthart  

**Link**: [PDF](https://arxiv.org/pdf/2503.15762)  

**Abstract**: Dialogic learning fosters motivation and deeper understanding in education through purposeful and structured dialogues. Foundational models offer a transformative potential for child-robot interactions, enabling the design of personalized, engaging, and scalable interactions. However, their integration into educational contexts presents challenges in terms of ensuring age-appropriate and safe content and alignment with pedagogical goals. We introduce a hybrid approach to designing personalized educational dialogues in child-robot interactions. By combining rule-based systems with LLMs for selective offline content generation and human validation, the framework ensures educational quality and developmental appropriateness. We illustrate this approach through a project aimed at enhancing reading motivation, in which a robot facilitated book-related dialogues. 

---
# DeepPsy-Agent: A Stage-Aware and Deep-Thinking Emotional Support Agent System 

**Authors**: Kai Chen, Zebing Sun  

**Link**: [PDF](https://arxiv.org/pdf/2503.15876)  

**Abstract**: This paper introduces DeepPsy-Agent, an innovative psychological support system that combines the three-stage helping theory in psychology with deep learning techniques. The system consists of two core components: (1) a multi-stage response-capable dialogue model (\textit{deeppsy-chat}), which enhances reasoning capabilities through stage-awareness and deep-thinking analysis to generate high-quality responses; and (2) a real-time stage transition detection model that identifies contextual shifts to guide the dialogue towards more effective intervention stages. Based on 30,000 real psychological hotline conversations, we employ AI-simulated dialogues and expert re-annotation strategies to construct a high-quality multi-turn dialogue dataset. Experimental results demonstrate that DeepPsy-Agent outperforms general-purpose large language models (LLMs) in key metrics such as problem exposure completeness, cognitive restructuring success rate, and action adoption rate. Ablation studies further validate the effectiveness of stage-awareness and deep-thinking modules, showing that stage information contributes 42.3\% to performance, while the deep-thinking module increases root-cause identification by 58.3\% and reduces ineffective suggestions by 72.1\%. This system addresses critical challenges in AI-based psychological support through dynamic dialogue management and deep reasoning, advancing intelligent mental health services. 

---
# Large Language Models for Water Distribution Systems Modeling and Decision-Making 

**Authors**: Yinon Goldshtein, Gal Perelman, Assaf Schuster, Avi Ostfeld  

**Link**: [PDF](https://arxiv.org/pdf/2503.16191)  

**Abstract**: The design, operations, and management of water distribution systems (WDS) involve complex mathematical models. These models are continually improving due to computational advancements, leading to better decision-making and more efficient WDS management. However, the significant time and effort required for modeling, programming, and analyzing results remain substantial challenges. Another issue is the professional burden, which confines the interaction with models, databases, and other sophisticated tools to a small group of experts, thereby causing non-technical stakeholders to depend on these experts or make decisions without modeling support. Furthermore, explaining model results is challenging even for experts, as it is often unclear which conditions cause the model to reach a certain state or recommend a specific policy. The recent advancements in Large Language Models (LLMs) open doors for a new stage in human-model interaction. This study proposes a framework of plain language interactions with hydraulic and water quality models based on LLM-EPANET architecture. This framework is tested with increasing levels of complexity of queries to study the ability of LLMs to interact with WDS models, run complex simulations, and report simulation results. The performance of the proposed framework is evaluated across several categories of queries and hyper-parameter configurations, demonstrating its potential to enhance decision-making processes in WDS management. 

---
# Entropy-based Exploration Conduction for Multi-step Reasoning 

**Authors**: Jinghan Zhang, Xiting Wang, Fengran Mo, Yeyang Zhou, Wanfu Gao, Kunpeng Liu  

**Link**: [PDF](https://arxiv.org/pdf/2503.15848)  

**Abstract**: In large language model (LLM) reasoning, multi-step processes have proven effective for solving complex tasks. However, the depth of exploration can significantly affect the reasoning performance. Existing methods to automatically decide the depth often bring high costs and lack flexibility, and thus undermine the model's reasoning accuracy. To address these issues, we propose Entropy-based Exploration Depth Conduction (Entro-duction), a novel method that dynamically adjusts the exploration depth during multi-step reasoning by monitoring LLM's output entropy and variance entropy. We employ these two metrics to capture the model's current uncertainty and the fluctuation of uncertainty across consecutive reasoning steps. Based on the observed changes, the LLM selects whether to deepen, expand or stop exploration according to the probability. In this way, we balance the reasoning accuracy and exploration effectiveness. Experimental results across four benchmark datasets demonstrate the efficacy of Entro-duction. We further conduct experiments and analysis on the components of Entro-duction to discuss their contributions to reasoning performance. 

---
# Bridging Technology and Humanities: Evaluating the Impact of Large Language Models on Social Sciences Research with DeepSeek-R1 

**Authors**: Peiran Gu, Fuhao Duan, Wenhao Li, Bochen Xu, Ying Cai, Teng Yao, Chenxun Zhuo, Tianming Liu, Bao Ge  

**Link**: [PDF](https://arxiv.org/pdf/2503.16304)  

**Abstract**: In recent years, the development of Large Language Models (LLMs) has made significant breakthroughs in the field of natural language processing and has gradually been applied to the field of humanities and social sciences research. LLMs have a wide range of application value in the field of humanities and social sciences because of its strong text understanding, generation and reasoning capabilities. In humanities and social sciences research, LLMs can analyze large-scale text data and make inferences.
This article analyzes the large language model DeepSeek-R1 from seven aspects: low-resource language translation, educational question-answering, student writing improvement in higher education, logical reasoning, educational measurement and psychometrics, public health policy analysis, and art this http URL we compare the answers given by DeepSeek-R1 in the seven aspects with the answers given by o1-preview. DeepSeek-R1 performs well in the humanities and social sciences, answering most questions correctly and logically, and can give reasonable analysis processes and explanations. Compared with o1-preview, it can automatically generate reasoning processes and provide more detailed explanations, which is suitable for beginners or people who need to have a detailed understanding of this knowledge, while o1-preview is more suitable for quick reading.
Through analysis, it is found that LLM has broad application potential in the field of humanities and social sciences, and shows great advantages in improving text analysis efficiency, language communication and other fields. LLM's powerful language understanding and generation capabilities enable it to deeply explore complex problems in the field of humanities and social sciences, and provide innovative tools for academic research and practical applications. 

---
# Using Language Models to Decipher the Motivation Behind Human Behaviors 

**Authors**: Yutong Xie, Qiaozhu Mei, Walter Yuan, Matthew O. Jackson  

**Link**: [PDF](https://arxiv.org/pdf/2503.15752)  

**Abstract**: AI presents a novel tool for deciphering the motivations behind human behaviors. We show that by varying prompts to a large language model, we can elicit a full range of human behaviors in a variety of different scenarios in terms of classic economic games. Then by analyzing which prompts are needed to elicit which behaviors, we can infer (decipher) the motivations behind the human behaviors. We also show how one can analyze the prompts to reveal relationships between the classic economic games, providing new insight into what different economic scenarios induce people to think about. We also show how this deciphering process can be used to understand differences in the behavioral tendencies of different populations. 

---
# Towards Lighter and Robust Evaluation for Retrieval Augmented Generation 

**Authors**: Alex-Razvan Ispas, Charles-Elie Simon, Fabien Caspani, Vincent Guigue  

**Link**: [PDF](https://arxiv.org/pdf/2503.16161)  

**Abstract**: Large Language Models are prompting us to view more NLP tasks from a generative perspective. At the same time, they offer a new way of accessing information, mainly through the RAG framework. While there have been notable improvements for the autoregressive models, overcoming hallucination in the generated answers remains a continuous problem. A standard solution is to use commercial LLMs, such as GPT4, to evaluate these algorithms. However, such frameworks are expensive and not very transparent. Therefore, we propose a study which demonstrates the interest of open-weight models for evaluating RAG hallucination. We develop a lightweight approach using smaller, quantized LLMs to provide an accessible and interpretable metric that gives continuous scores for the generated answer with respect to their correctness and faithfulness. This score allows us to question decisions' reliability and explore thresholds to develop a new AUC metric as an alternative to correlation with human judgment. 

---
# Unify and Triumph: Polyglot, Diverse, and Self-Consistent Generation of Unit Tests with LLMs 

**Authors**: Djamel Eddine Khelladi, Charly Reux, Mathieu Acher  

**Link**: [PDF](https://arxiv.org/pdf/2503.16144)  

**Abstract**: Large language model (LLM)-based test generation has gained attention in software engineering, yet most studies evaluate LLMs' ability to generate unit tests in a single attempt for a given language, missing the opportunity to leverage LLM diversity for more robust testing. This paper introduces PolyTest, a novel approach that enhances test generation by exploiting polyglot and temperature-controlled diversity. PolyTest systematically leverages these properties in two complementary ways: (1) Cross-lingual test generation, where tests are generated in multiple languages at zero temperature and then unified; (2) Diverse test sampling, where multiple test sets are generated within the same language at a higher temperature before unification. A key insight is that LLMs can generate diverse yet contradicting tests -- same input, different expected outputs -- across languages and generations. PolyTest mitigates inconsistencies by unifying test sets, fostering self-consistency and improving overall test quality. Unlike single-language or single-attempt approaches, PolyTest enhances testing without requiring on-the-fly execution, making it particularly beneficial for weaker-performing languages. We evaluate PolyTest on Llama3-70B, GPT-4o, and GPT-3.5 using EvalPlus, generating tests in five languages (Java, C, Python, JavaScript, and a CSV-based format) at temperature 0 and sampling multiple sets at temperature 1. We observe that LLMs frequently generate contradicting tests across settings, and that PolyTest significantly improves test quality across all considered metrics -- number of tests, passing rate, statement/branch coverage (up to +9.01%), and mutation score (up to +11.23%). Finally, PolyTest outperforms Pynguin in test generation, passing rate, and mutation score. 

---
# MathFusion: Enhancing Mathematic Problem-solving of LLM through Instruction Fusion 

**Authors**: Qizhi Pei, Lijun Wu, Zhuoshi Pan, Yu Li, Honglin Lin, Chenlin Ming, Xin Gao, Conghui He, Rui Yan  

**Link**: [PDF](https://arxiv.org/pdf/2503.16212)  

**Abstract**: Large Language Models (LLMs) have shown impressive progress in mathematical reasoning. While data augmentation is promising to enhance mathematical problem-solving ability, current approaches are predominantly limited to instance-level modifications-such as rephrasing or generating syntactic variations-which fail to capture and leverage the intrinsic relational structures inherent in mathematical knowledge. Inspired by human learning processes, where mathematical proficiency develops through systematic exposure to interconnected concepts, we introduce MathFusion, a novel framework that enhances mathematical reasoning through cross-problem instruction synthesis. MathFusion implements this through three fusion strategies: (1) sequential fusion, which chains related problems to model solution dependencies; (2) parallel fusion, which combines analogous problems to reinforce conceptual understanding; and (3) conditional fusion, which creates context-aware selective problems to enhance reasoning flexibility. By applying these strategies, we generate a new dataset, \textbf{MathFusionQA}, followed by fine-tuning models (DeepSeekMath-7B, Mistral-7B, Llama3-8B) on it. Experimental results demonstrate that MathFusion achieves substantial improvements in mathematical reasoning while maintaining high data efficiency, boosting performance by 18.0 points in accuracy across diverse benchmarks while requiring only 45K additional synthetic instructions, representing a substantial improvement over traditional single-instruction approaches. Our datasets, models, and code are publicly available at this https URL. 

---
# Tuning LLMs by RAG Principles: Towards LLM-native Memory 

**Authors**: Jiale Wei, Shuchi Wu, Ruochen Liu, Xiang Ying, Jingbo Shang, Fangbo Tao  

**Link**: [PDF](https://arxiv.org/pdf/2503.16071)  

**Abstract**: Memory, additional information beyond the training of large language models (LLMs), is crucial to various real-world applications, such as personal assistant. The two mainstream solutions to incorporate memory into the generation process are long-context LLMs and retrieval-augmented generation (RAG). In this paper, we first systematically compare these two types of solutions on three renovated/new datasets and show that (1) long-context solutions, although more expensive, shall be easier to capture the big picture and better answer queries which require considering the memory as a whole; and (2) when the queries concern specific information, RAG solutions shall be more competitive especially when the keywords can be explicitly matched. Therefore, we propose a novel method RAG-Tuned-LLM which fine-tunes a relative small (e.g., 7B) LLM using the data generated following the RAG principles, so it can combine the advantages of both solutions. Extensive experiments on three datasets demonstrate that RAG-Tuned-LLM can beat long-context LLMs and RAG methods across a wide range of query types. 

---
# The Lighthouse of Language: Enhancing LLM Agents via Critique-Guided Improvement 

**Authors**: Ruihan Yang, Fanghua Ye, Jian Li, Siyu Yuan, Yikai Zhang, Zhaopeng Tu, Xiaolong Li, Deqing Yang  

**Link**: [PDF](https://arxiv.org/pdf/2503.16024)  

**Abstract**: Large language models (LLMs) have recently transformed from text-based assistants to autonomous agents capable of planning, reasoning, and iteratively improving their actions. While numerical reward signals and verifiers can effectively rank candidate actions, they often provide limited contextual guidance. In contrast, natural language feedback better aligns with the generative capabilities of LLMs, providing richer and more actionable suggestions. However, parsing and implementing this feedback effectively can be challenging for LLM-based agents. In this work, we introduce Critique-Guided Improvement (CGI), a novel two-player framework, comprising an actor model that explores an environment and a critic model that generates detailed nature language feedback. By training the critic to produce fine-grained assessments and actionable revisions, and the actor to utilize these critiques, our approach promotes more robust exploration of alternative strategies while avoiding local optima. Experiments in three interactive environments show that CGI outperforms existing baselines by a substantial margin. Notably, even a small critic model surpasses GPT-4 in feedback quality. The resulting actor achieves state-of-the-art performance, demonstrating the power of explicit iterative guidance to enhance decision-making in LLM-based agents. 

---
# Autonomous AI imitators increase diversity in homogeneous information ecosystems 

**Authors**: Emil Bakkensen Johansen, Oliver Baumann  

**Link**: [PDF](https://arxiv.org/pdf/2503.16021)  

**Abstract**: Recent breakthroughs in large language models (LLMs) have facilitated autonomous AI agents capable of imitating human-generated content. This technological advancement raises fundamental questions about AI's potential impact on the diversity and democratic value of information ecosystems. Here, we introduce a large-scale simulation framework to examine AI-based imitation in news, a context critically influential for public discourse. By systematically testing two distinct imitation strategies across a range of information environments varying in initial diversity, we demonstrate that AI-generated articles do not uniformly homogenize content. Instead, AI's influence is strongly context-dependent: AI-generated articles can introduce valuable diversity in originally homogeneous news environments, while potentially diminishing diversity in contexts that initially display high heterogeneity. These results illustrate that the baseline diversity of an information space critically shapes AI's impact, challenging assumptions that AI-driven imitation uniformly threatens information diversity. Instead, when information is initially homogeneous, AI-driven imitation can expand perspectives, styles, and topics. This is especially important in news contexts, where information diversity fosters richer public debate by exposing citizens to alternative viewpoints, challenging biases, and preventing narrative monopolies, which is essential for a resilient democracy. 

---
# From Structured Prompts to Open Narratives: Measuring Gender Bias in LLMs Through Open-Ended Storytelling 

**Authors**: Evan Chen, Run-Jun Zhan, Yan-Bai Lin, Hung-Hsuan Chen  

**Link**: [PDF](https://arxiv.org/pdf/2503.15904)  

**Abstract**: Large Language Models (LLMs) have revolutionized natural language processing, yet concerns persist regarding their tendency to reflect or amplify social biases present in their training data. This study introduces a novel evaluation framework to uncover gender biases in LLMs, focusing on their occupational narratives. Unlike previous methods relying on structured scenarios or carefully crafted prompts, our approach leverages free-form storytelling to reveal biases embedded in the models. Systematic analyses show an overrepresentation of female characters across occupations in six widely used LLMs. Additionally, our findings reveal that LLM-generated occupational gender rankings align more closely with human stereotypes than actual labor statistics. These insights underscore the need for balanced mitigation strategies to ensure fairness while avoiding the reinforcement of new stereotypes. 

---
# Parameters vs. Context: Fine-Grained Control of Knowledge Reliance in Language Models 

**Authors**: Baolong Bi, Shenghua Liu, Yiwei Wang, Yilong Xu, Junfeng Fang, Lingrui Mei, Xueqi Cheng  

**Link**: [PDF](https://arxiv.org/pdf/2503.15888)  

**Abstract**: Retrieval-Augmented Generation (RAG) mitigates hallucinations in Large Language Models (LLMs) by integrating external knowledge. However, conflicts between parametric knowledge and retrieved context pose challenges, particularly when retrieved information is unreliable or the model's internal knowledge is outdated. In such cases, LLMs struggle to determine whether to rely more on their own parameters or the conflicted context. To address this, we propose **CK-PLUG**, a plug-and-play method for controlling LLMs' reliance on parametric and contextual knowledge. We introduce a novel knowledge consistency metric, Confidence Gain, which detects knowledge conflicts by measuring entropy shifts in token probability distributions after context insertion. CK-PLUG then enables fine-grained control over knowledge preference by adjusting the probability distribution of tokens with negative confidence gain through a single tuning parameter. Experiments demonstrate CK-PLUG's ability to significantly regulate knowledge reliance in counterfactual RAG scenarios while maintaining generation fluency and knowledge accuracy. For instance, on Llama3-8B, memory recall (MR) of RAG response can be adjusted within a broad range (9.9%-71.9%), compared to the baseline of 42.1%. Moreover, CK-PLUG supports adaptive control based on the model's confidence in both internal and external knowledge, achieving consistent performance improvements across various general RAG tasks. Our code is available at: $\href{this https URL}{\text{this https URL}}$. 

---
# Don't Fight Hallucinations, Use Them: Estimating Image Realism using NLI over Atomic Facts 

**Authors**: Elisei Rykov, Kseniia Petrushina, Kseniia Titova, Alexander Panchenko, Vasily Konovalov  

**Link**: [PDF](https://arxiv.org/pdf/2503.15948)  

**Abstract**: Quantifying the realism of images remains a challenging problem in the field of artificial intelligence. For example, an image of Albert Einstein holding a smartphone violates common-sense because modern smartphone were invented after Einstein's death. We introduce a novel method for assessing image realism using Large Vision-Language Models (LVLMs) and Natural Language Inference (NLI). Our approach is based on the premise that LVLMs may generate hallucinations when confronted with images that defy common sense. Using LVLM to extract atomic facts from these images, we obtain a mix of accurate facts and erroneous hallucinations. We proceed by calculating pairwise entailment scores among these facts, subsequently aggregating these values to yield a singular reality score. This process serves to identify contradictions between genuine facts and hallucinatory elements, signaling the presence of images that violate common sense. Our approach has achieved a new state-of-the-art performance in zero-shot mode on the WHOOPS! dataset. 

---
# CaKE: Circuit-aware Editing Enables Generalizable Knowledge Learners 

**Authors**: Yunzhi Yao, Jizhan Fang, Jia-Chen Gu, Ningyu Zhang, Shumin Deng, Huajun Chen, Nanyun Peng  

**Link**: [PDF](https://arxiv.org/pdf/2503.16356)  

**Abstract**: Knowledge Editing (KE) enables the modification of outdated or incorrect information in large language models (LLMs). While existing KE methods can update isolated facts, they struggle to generalize these updates to multi-hop reasoning tasks that depend on the modified knowledge. Through an analysis of reasoning circuits -- the neural pathways LLMs use for knowledge-based inference, we observe that current layer-localized KE approaches, such as MEMIT and WISE, which edit only single or a few model layers, struggle to effectively incorporate updated information into these reasoning pathways. To address this limitation, we propose CaKE (Circuit-aware Knowledge Editing), a novel method that enables more effective integration of updated knowledge in LLMs. CaKE leverages strategically curated data, guided by our circuits-based analysis, that enforces the model to utilize the modified knowledge, stimulating the model to develop appropriate reasoning circuits for newly integrated knowledge. Experimental results show that CaKE enables more accurate and consistent use of updated knowledge across related reasoning tasks, leading to an average of 20% improvement in multi-hop reasoning accuracy on MQuAKE dataset compared to existing KE methods. We release the code and data in this https URL. 

---
# Fùxì: A Benchmark for Evaluating Language Models on Ancient Chinese Text Understanding and Generation 

**Authors**: Shangqing Zhao, Yuhao Zhou, Yupei Ren, Zhe Chen, Chenghao Jia, Fang Zhe, Zhaogaung Long, Shu Liu, Man Lan  

**Link**: [PDF](https://arxiv.org/pdf/2503.15837)  

**Abstract**: Ancient Chinese text processing presents unique challenges for large language models (LLMs) due to its distinct linguistic features, complex structural constraints, and rich cultural context. While existing benchmarks have primarily focused on evaluating comprehension through multiple-choice questions, there remains a critical gap in assessing models' generative capabilities in classical Chinese. We introduce Fùxì, a comprehensive benchmark that evaluates both understanding and generation capabilities across 21 diverse tasks. Our benchmark distinguishes itself through three key contributions: (1) balanced coverage of both comprehension and generation tasks, including novel tasks like poetry composition and couplet completion, (2) specialized evaluation metrics designed specifically for classical Chinese text generation, combining rule-based verification with fine-tuned LLM evaluators, and (3) a systematic assessment framework that considers both linguistic accuracy and cultural authenticity. Through extensive evaluation of state-of-the-art LLMs, we reveal significant performance gaps between understanding and generation tasks, with models achieving promising results in comprehension but struggling considerably in generation tasks, particularly those requiring deep cultural knowledge and adherence to classical formats. Our findings highlight the current limitations in ancient Chinese text processing and provide insights for future model development. The benchmark, evaluation toolkit, and baseline results are publicly available to facilitate research in this domain. 

---
# Detecting LLM-Written Peer Reviews 

**Authors**: Vishisht Rao, Aounon Kumar, Himabindu Lakkaraju, Nihar B. Shah  

**Link**: [PDF](https://arxiv.org/pdf/2503.15772)  

**Abstract**: Editors of academic journals and program chairs of conferences require peer reviewers to write their own reviews. However, there is growing concern about the rise of lazy reviewing practices, where reviewers use large language models (LLMs) to generate reviews instead of writing them independently. Existing tools for detecting LLM-generated content are not designed to differentiate between fully LLM-generated reviews and those merely polished by an LLM. In this work, we employ a straightforward approach to identify LLM-generated reviews - doing an indirect prompt injection via the paper PDF to ask the LLM to embed a watermark. Our focus is on presenting watermarking schemes and statistical tests that maintain a bounded family-wise error rate, when a venue evaluates multiple reviews, with a higher power as compared to standard methods like Bonferroni correction. These guarantees hold without relying on any assumptions about human-written reviews. We also consider various methods for prompt injection including font embedding and jailbreaking. We evaluate the effectiveness and various tradeoffs of these methods, including different reviewer defenses. We find a high success rate in the embedding of our watermarks in LLM-generated reviews across models. We also find that our approach is resilient to common reviewer defenses, and that the bounds on error rates in our statistical tests hold in practice while having the power to flag LLM-generated reviews, while Bonferroni correction is infeasible. 

---
# AutoRedTeamer: Autonomous Red Teaming with Lifelong Attack Integration 

**Authors**: Andy Zhou, Kevin Wu, Francesco Pinto, Zhaorun Chen, Yi Zeng, Yu Yang, Shuang Yang, Sanmi Koyejo, James Zou, Bo Li  

**Link**: [PDF](https://arxiv.org/pdf/2503.15754)  

**Abstract**: As large language models (LLMs) become increasingly capable, security and safety evaluation are crucial. While current red teaming approaches have made strides in assessing LLM vulnerabilities, they often rely heavily on human input and lack comprehensive coverage of emerging attack vectors. This paper introduces AutoRedTeamer, a novel framework for fully automated, end-to-end red teaming against LLMs. AutoRedTeamer combines a multi-agent architecture with a memory-guided attack selection mechanism to enable continuous discovery and integration of new attack vectors. The dual-agent framework consists of a red teaming agent that can operate from high-level risk categories alone to generate and execute test cases and a strategy proposer agent that autonomously discovers and implements new attacks by analyzing recent research. This modular design allows AutoRedTeamer to adapt to emerging threats while maintaining strong performance on existing attack vectors. We demonstrate AutoRedTeamer's effectiveness across diverse evaluation settings, achieving 20% higher attack success rates on HarmBench against Llama-3.1-70B while reducing computational costs by 46% compared to existing approaches. AutoRedTeamer also matches the diversity of human-curated benchmarks in generating test cases, providing a comprehensive, scalable, and continuously evolving framework for evaluating the security of AI systems. 

---
# Safety Aware Task Planning via Large Language Models in Robotics 

**Authors**: Azal Ahmad Khan, Michael Andrev, Muhammad Ali Murtaza, Sergio Aguilera, Rui Zhang, Jie Ding, Seth Hutchinson, Ali Anwar  

**Link**: [PDF](https://arxiv.org/pdf/2503.15707)  

**Abstract**: The integration of large language models (LLMs) into robotic task planning has unlocked better reasoning capabilities for complex, long-horizon workflows. However, ensuring safety in LLM-driven plans remains a critical challenge, as these models often prioritize task completion over risk mitigation. This paper introduces SAFER (Safety-Aware Framework for Execution in Robotics), a multi-LLM framework designed to embed safety awareness into robotic task planning. SAFER employs a Safety Agent that operates alongside the primary task planner, providing safety feedback. Additionally, we introduce LLM-as-a-Judge, a novel metric leveraging LLMs as evaluators to quantify safety violations within generated task plans. Our framework integrates safety feedback at multiple stages of execution, enabling real-time risk assessment, proactive error correction, and transparent safety evaluation. We also integrate a control framework using Control Barrier Functions (CBFs) to ensure safety guarantees within SAFER's task planning. We evaluated SAFER against state-of-the-art LLM planners on complex long-horizon tasks involving heterogeneous robotic agents, demonstrating its effectiveness in reducing safety violations while maintaining task efficiency. We also verify the task planner and safety planner through actual hardware experiments involving multiple robots and a human. 

---
# Does Context Matter? ContextualJudgeBench for Evaluating LLM-based Judges in Contextual Settings 

**Authors**: Austin Xu, Srijan Bansal, Yifei Ming, Semih Yavuz, Shafiq Joty  

**Link**: [PDF](https://arxiv.org/pdf/2503.15620)  

**Abstract**: The large language model (LLM)-as-judge paradigm has been used to meet the demand for a cheap, reliable, and fast evaluation of model outputs during AI system development and post-deployment monitoring. While judge models -- LLMs finetuned to specialize in assessing and critiquing model outputs -- have been touted as general purpose evaluators, they are typically evaluated only on non-contextual scenarios, such as instruction following. The omission of contextual settings -- those where external information is used as context to generate an output -- is surprising given the increasing prevalence of retrieval-augmented generation (RAG) and summarization use cases. Contextual assessment is uniquely challenging, as evaluation often depends on practitioner priorities, leading to conditional evaluation criteria (e.g., comparing responses based on factuality and then considering completeness if they are equally factual). To address the gap, we propose ContextualJudgeBench, a judge benchmark with 2,000 challenging response pairs across eight splits inspired by real-world contextual evaluation scenarios. We build our benchmark with a multi-pronged data construction pipeline that leverages both existing human annotations and model-based perturbations. Our comprehensive study across 11 judge models and 9 general purpose models, reveals that the contextual information and its assessment criteria present a significant challenge to even state-of-the-art models. For example, OpenAI's o1, the best-performing model, barely reaches 55% consistent accuracy. 

---
# ChatGPT and U(X): A Rapid Review on Measuring the User Experience 

**Authors**: Katie Seaborn  

**Link**: [PDF](https://arxiv.org/pdf/2503.15808)  

**Abstract**: ChatGPT, powered by a large language model (LLM), has revolutionized everyday human-computer interaction (HCI) since its 2022 release. While now used by millions around the world, a coherent pathway for evaluating the user experience (UX) ChatGPT offers remains missing. In this rapid review (N = 58), I explored how ChatGPT UX has been approached quantitatively so far. I focused on the independent variables (IVs) manipulated, the dependent variables (DVs) measured, and the methods used for measurement. Findings reveal trends, gaps, and emerging consensus in UX assessments. This work offers a first step towards synthesizing existing approaches to measuring ChatGPT UX, urgent trajectories to advance standardization and breadth, and two preliminary frameworks aimed at guiding future research and tool development. I seek to elevate the field of ChatGPT UX by empowering researchers and practitioners in optimizing user interactions with ChatGPT and similar LLM-based systems. 

---
# Efficient but Vulnerable: Benchmarking and Defending LLM Batch Prompting Attack 

**Authors**: Murong Yue, Ziyu Yao  

**Link**: [PDF](https://arxiv.org/pdf/2503.15551)  

**Abstract**: Batch prompting, which combines a batch of multiple queries sharing the same context in one inference, has emerged as a promising solution to reduce inference costs. However, our study reveals a significant security vulnerability in batch prompting: malicious users can inject attack instructions into a batch, leading to unwanted interference across all queries, which can result in the inclusion of harmful content, such as phishing links, or the disruption of logical reasoning. In this paper, we construct BATCHSAFEBENCH, a comprehensive benchmark comprising 150 attack instructions of two types and 8k batch instances, to study the batch prompting vulnerability systematically. Our evaluation of both closed-source and open-weight LLMs demonstrates that all LLMs are susceptible to batch-prompting attacks. We then explore multiple defending approaches. While the prompting-based defense shows limited effectiveness for smaller LLMs, the probing-based approach achieves about 95% accuracy in detecting attacks. Additionally, we perform a mechanistic analysis to understand the attack and identify attention heads that are responsible for it. 

---
# Active management of battery degradation in wireless sensor network using deep reinforcement learning for group battery replacement 

**Authors**: Jong-Hyun Jeonga, Hongki Jo, Qiang Zhou, Tahsin Afroz Hoque Nishat, Lang Wu  

**Link**: [PDF](https://arxiv.org/pdf/2503.15865)  

**Abstract**: Wireless sensor networks (WSNs) have become a promising solution for structural health monitoring (SHM), especially in hard-to-reach or remote locations. Battery-powered WSNs offer various advantages over wired systems, however limited battery life has always been one of the biggest obstacles in practical use of the WSNs, regardless of energy harvesting methods. While various methods have been studied for battery health management, existing methods exclusively aim to extend lifetime of individual batteries, lacking a system level view. A consequence of applying such methods is that batteries in a WSN tend to fail at different times, posing significant difficulty on planning and scheduling of battery replacement trip. This study investigate a deep reinforcement learning (DRL) method for active battery degradation management by optimizing duty cycle of WSNs at the system level. This active management strategy effectively reduces earlier failure of battery individuals which enable group replacement without sacrificing WSN performances. A simulated environment based on a real-world WSN setup was developed to train a DRL agent and learn optimal duty cycle strategies. The performance of the strategy was validated in a long-term setup with various network sizes, demonstrating its efficiency and scalability. 

---
# Enforcing Cybersecurity Constraints for LLM-driven Robot Agents for Online Transactions 

**Authors**: Shraddha Pradipbhai Shah, Aditya Vilas Deshpande  

**Link**: [PDF](https://arxiv.org/pdf/2503.15546)  

**Abstract**: The integration of Large Language Models (LLMs) into autonomous robotic agents for conducting online transactions poses significant cybersecurity challenges. This study aims to enforce robust cybersecurity constraints to mitigate the risks associated with data breaches, transaction fraud, and system manipulation. The background focuses on the rise of LLM-driven robotic systems in e-commerce, finance, and service industries, alongside the vulnerabilities they introduce. A novel security architecture combining blockchain technology with multi-factor authentication (MFA) and real-time anomaly detection was implemented to safeguard transactions. Key performance metrics such as transaction integrity, response time, and breach detection accuracy were evaluated, showing improved security and system performance. The results highlight that the proposed architecture reduced fraudulent transactions by 90%, improved breach detection accuracy to 98%, and ensured secure transaction validation within a latency of 0.05 seconds. These findings emphasize the importance of cybersecurity in the deployment of LLM-driven robotic systems and suggest a framework adaptable to various online platforms. 

---
# PersonaAI: Leveraging Retrieval-Augmented Generation and Personalized Context for AI-Driven Digital Avatars 

**Authors**: Elvis Kimara, Kunle S. Oguntoye, Jian Sun  

**Link**: [PDF](https://arxiv.org/pdf/2503.15489)  

**Abstract**: This paper introduces PersonaAI, a cutting-edge application that leverages Retrieval-Augmented Generation (RAG) and the LLAMA model to create highly personalized digital avatars capable of accurately mimicking individual personalities. Designed as a cloud-based mobile application, PersonaAI captures user data seamlessly, storing it in a secure database for retrieval and analysis. The result is a system that provides context-aware, accurate responses to user queries, enhancing the potential of AI-driven personalization.
Why should you care? PersonaAI combines the scalability of RAG with the efficiency of prompt-engineered LLAMA3, offering a lightweight, sustainable alternative to traditional large language model (LLM) training methods. The system's novel approach to data collection, utilizing real-time user interactions via a mobile app, ensures enhanced context relevance while maintaining user privacy. By open-sourcing our implementation, we aim to foster adaptability and community-driven development.
PersonaAI demonstrates how AI can transform interactions by merging efficiency, scalability, and personalization, making it a significant step forward in the future of digital avatars and personalized AI. 

---
# From Divergence to Consensus: Evaluating the Role of Large Language Models in Facilitating Agreement through Adaptive Strategies 

**Authors**: Loukas Triantafyllopoulos, Dimitris Kalles  

**Link**: [PDF](https://arxiv.org/pdf/2503.15521)  

**Abstract**: Achieving consensus in group decision-making often involves overcoming significant challenges, particularly in reconciling diverse perspectives and mitigating biases that hinder agreement. Traditional methods relying on human facilitators are often constrained by scalability and efficiency, especially in large-scale, fast-paced discussions. To address these challenges, this study proposes a novel framework employing large language models (LLMs) as automated facilitators within a custom-built multi-user chat system. Leveraging cosine similarity as a core metric, this approach evaluates the ability of three state-of-the-art LLMs- ChatGPT 4.0, Mistral Large 2, and AI21 Jamba Instruct- to synthesize consensus proposals that align with participants' viewpoints. Unlike conventional techniques, the system integrates adaptive facilitation strategies, including clarifying misunderstandings, summarizing discussions, and proposing compromises, enabling the LLMs to iteratively refine consensus proposals based on user feedback. Experimental results demonstrate the superiority of ChatGPT 4.0, which achieves higher alignment with participant opinions, requiring fewer iterations to reach consensus compared to its counterparts. Moreover, analysis reveals the nuanced performance of the models across various sustainability-focused discussion topics, such as climate action, quality education, good health and well-being, and access to clean water and sanitation. These findings highlight the transformative potential of LLM-driven facilitation for improving collective decision-making processes and underscore the importance of advancing evaluation metrics and cross-cultural adaptability in future research. 

---
# Stop Overthinking: A Survey on Efficient Reasoning for Large Language Models 

**Authors**: Yang Sui, Yu-Neng Chuang, Guanchu Wang, Jiamu Zhang, Tianyi Zhang, Jiayi Yuan, Hongyi Liu, Andrew Wen, Shaochen, Zhong, Hanjie Chen, Xia Hu  

**Link**: [PDF](https://arxiv.org/pdf/2503.16419)  

**Abstract**: Large Language Models (LLMs) have demonstrated remarkable capabilities in complex tasks. Recent advancements in Large Reasoning Models (LRMs), such as OpenAI o1 and DeepSeek-R1, have further improved performance in System-2 reasoning domains like mathematics and programming by harnessing supervised fine-tuning (SFT) and reinforcement learning (RL) techniques to enhance the Chain-of-Thought (CoT) reasoning. However, while longer CoT reasoning sequences improve performance, they also introduce significant computational overhead due to verbose and redundant outputs, known as the "overthinking phenomenon". In this paper, we provide the first structured survey to systematically investigate and explore the current progress toward achieving efficient reasoning in LLMs. Overall, relying on the inherent mechanism of LLMs, we categorize existing works into several key directions: (1) model-based efficient reasoning, which considers optimizing full-length reasoning models into more concise reasoning models or directly training efficient reasoning models; (2) reasoning output-based efficient reasoning, which aims to dynamically reduce reasoning steps and length during inference; (3) input prompts-based efficient reasoning, which seeks to enhance reasoning efficiency based on input prompt properties such as difficulty or length control. Additionally, we introduce the use of efficient data for training reasoning models, explore the reasoning capabilities of small language models, and discuss evaluation methods and benchmarking. 

---
# Cultural Alignment in Large Language Models Using Soft Prompt Tuning 

**Authors**: Reem I. Masoud, Martin Ferianc, Philip Treleaven, Miguel Rodrigues  

**Link**: [PDF](https://arxiv.org/pdf/2503.16094)  

**Abstract**: Large Language Model (LLM) alignment conventionally relies on supervised fine-tuning or reinforcement learning based alignment frameworks. These methods typically require labeled or preference datasets and involve updating model weights to align the LLM with the training objective or reward model. Meanwhile, in social sciences such as cross-cultural studies, factor analysis is widely used to uncover underlying dimensions or latent variables that explain observed patterns in survey data. The non-differentiable nature of these measurements deriving from survey data renders the former alignment methods infeasible for alignment with cultural dimensions. To overcome this, we propose a parameter efficient strategy that combines soft prompt tuning, which freezes the model parameters while modifying the input prompt embeddings, with Differential Evolution (DE), a black-box optimization method for cases where a differentiable objective is unattainable. This strategy ensures alignment consistency without the need for preference data or model parameter updates, significantly enhancing efficiency and mitigating overfitting. Our method demonstrates significant improvements in LLama-3-8B-Instruct's cultural dimensions across multiple regions, outperforming both the Naive LLM and the In-context Learning (ICL) baseline, and effectively bridges computational models with human cultural nuances. 

---
# Automatically Generating Chinese Homophone Words to Probe Machine Translation Estimation Systems 

**Authors**: Shenbin Qian, Constantin Orăsan, Diptesh Kanojia, Félix do Carmo  

**Link**: [PDF](https://arxiv.org/pdf/2503.16158)  

**Abstract**: Evaluating machine translation (MT) of user-generated content (UGC) involves unique challenges such as checking whether the nuance of emotions from the source are preserved in the target text. Recent studies have proposed emotion-related datasets, frameworks and models to automatically evaluate MT quality of Chinese UGC, without relying on reference translations. However, whether these models are robust to the challenge of preserving emotional nuances has been left largely unexplored. To address this gap, we introduce a novel method inspired by information theory which generates challenging Chinese homophone words related to emotions, by leveraging the concept of self-information. Our approach generates homophones that were observed to cause translation errors in emotion preservation, and exposes vulnerabilities in MT systems and their evaluation methods when tackling emotional UGC. We evaluate the efficacy of our method using human evaluation for the quality of these generated homophones, and compare it with an existing one, showing that our method achieves higher correlation with human judgments. The generated Chinese homophones, along with their manual translations, are utilized to generate perturbations and to probe the robustness of existing quality evaluation models, including models trained using multi-task learning, fine-tuned variants of multilingual language models, as well as large language models (LLMs). Our results indicate that LLMs with larger size exhibit higher stability and robustness to such perturbations. We release our data and code for reproducibility and further research. 

---
# Fin-R1: A Large Language Model for Financial Reasoning through Reinforcement Learning 

**Authors**: Zhaowei Liu, Xin Guo, Fangqi Lou, Lingfeng Zeng, Jinyi Niu, Zixuan Wang, Jiajie Xu, Weige Cai, Ziwei Yang, Xueqian Zhao, Chao Li, Sheng Xu, Dezhi Chen, Yun Chen, Zuo Bai, Liwen Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2503.16252)  

**Abstract**: Reasoning large language models are rapidly evolving across various domains. However, their capabilities in handling complex financial tasks still require in-depth exploration. In this paper, we introduce Fin-R1, a reasoning large language model specifically designed for the financial sector. Fin-R1 is built using a two-stage architecture, leveraging a financial reasoning dataset distilled and processed based on DeepSeek-R1. Through supervised fine-tuning (SFT) and reinforcement learning (RL) training, it demonstrates performance close to DeepSeek-R1 with a parameter size of 7 billion across a range of financial reasoning tasks. It achieves the state-of-the-art (SOTA) in the FinQA and ConvFinQA tasks between those LLMs in our evaluation, surpassing larger models in other tasks as well. Fin-R1 showcases strong reasoning and decision-making capabilities, providing solutions to various problems encountered in the financial domain. Our code is available at this https URL. 

---
# Corrective In-Context Learning: Evaluating Self-Correction in Large Language Models 

**Authors**: Mario Sanz-Guerrero, Katharina von der Wense  

**Link**: [PDF](https://arxiv.org/pdf/2503.16022)  

**Abstract**: In-context learning (ICL) has transformed the use of large language models (LLMs) for NLP tasks, enabling few-shot learning by conditioning on labeled examples without finetuning. Despite its effectiveness, ICL is prone to errors, especially for challenging examples. With the goal of improving the performance of ICL, we propose corrective in-context learning (CICL), an approach that incorporates a model's incorrect predictions alongside ground truth corrections into the prompt, aiming to enhance classification accuracy through self-correction. However, contrary to our hypothesis, extensive experiments on text classification tasks demonstrate that CICL consistently underperforms standard ICL, with performance degrading as the proportion of corrections in the prompt increases. Our findings indicate that CICL introduces confusion by disrupting the model's task understanding, rather than refining its predictions. Additionally, we observe that presenting harder examples in standard ICL does not improve performance, suggesting that example difficulty alone may not be a reliable criterion for effective selection. By presenting these negative results, we provide important insights into the limitations of self-corrective mechanisms in LLMs and offer directions for future research. 

---
# Evaluating Test-Time Scaling LLMs for Legal Reasoning: OpenAI o1, DeepSeek-R1, and Beyond 

**Authors**: Yaoyao Yu, Leilei Gan, Yinghao Hu, Bin Wei, Kun Kuang, Fei Wu  

**Link**: [PDF](https://arxiv.org/pdf/2503.16040)  

**Abstract**: Recently, Test-Time Scaling Large Language Models (LLMs), such as DeepSeek-R1 and OpenAI o1, have demonstrated exceptional capabilities across various domains and tasks, particularly in reasoning. While these models have shown impressive performance on general language tasks, their effectiveness in specialized fields like legal remains unclear. To address this, we present a preliminary evaluation of LLMs in various legal scenarios, covering both Chinese and English legal tasks. Our analysis includes 9 LLMs and 17 legal tasks, with a focus on newly published and more complex challenges such as multi-defendant legal judgments and legal argument reasoning. Our findings indicate that, despite DeepSeek-R1 and OpenAI o1 being among the most powerful models, their legal reasoning capabilities are still lacking. Specifically, these models score below 80\% on seven Chinese legal reasoning tasks and below 80\% on two English legal reasoning tasks. This suggests that, even among the most advanced reasoning models, legal reasoning abilities remain underdeveloped. 

---
# MKG-Rank: Enhancing Large Language Models with Knowledge Graph for Multilingual Medical Question Answering 

**Authors**: Feiyang Li, Yingjian Chen, Haoran Liu, Rui Yang, Han Yuan, Yuang Jiang, Tianxiao Li, Edison Marrese Taylor, Hossein Rouhizadeh, Yusuke Iwasawa, Douglas Teodoro, Yutaka Matsuo, Irene Li  

**Link**: [PDF](https://arxiv.org/pdf/2503.16131)  

**Abstract**: Large Language Models (LLMs) have shown remarkable progress in medical question answering (QA), yet their effectiveness remains predominantly limited to English due to imbalanced multilingual training data and scarce medical resources for low-resource languages. To address this critical language gap in medical QA, we propose Multilingual Knowledge Graph-based Retrieval Ranking (MKG-Rank), a knowledge graph-enhanced framework that enables English-centric LLMs to perform multilingual medical QA. Through a word-level translation mechanism, our framework efficiently integrates comprehensive English-centric medical knowledge graphs into LLM reasoning at a low cost, mitigating cross-lingual semantic distortion and achieving precise medical QA across language barriers. To enhance efficiency, we introduce caching and multi-angle ranking strategies to optimize the retrieval process, significantly reducing response times and prioritizing relevant medical knowledge. Extensive evaluations on multilingual medical QA benchmarks across Chinese, Japanese, Korean, and Swahili demonstrate that MKG-Rank consistently outperforms zero-shot LLMs, achieving maximum 33.89% increase in accuracy, while maintaining an average retrieval time of only 0.0009 seconds. 

---
# From Chaos to Order: The Atomic Reasoner Framework for Fine-grained Reasoning in Large Language Models 

**Authors**: Jinyi Liu, Yan Zheng, Rong Cheng, Qiyu Wu, Wei Guo, Fei Ni, Hebin Liang, Yifu Yuan, Hangyu Mao, Fuzheng Zhang, Jianye Hao  

**Link**: [PDF](https://arxiv.org/pdf/2503.15944)  

**Abstract**: Recent advances in large language models (LLMs) have shown remarkable progress, yet their capacity for logical ``slow-thinking'' reasoning persists as a critical research frontier. Current inference scaling paradigms suffer from two fundamental constraints: fragmented thought flows compromising logical coherence, and intensively computational complexity that escalates with search space dimensions. To overcome these limitations, we present \textbf{Atomic Reasoner} (\textbf{AR}), a cognitive inference strategy that enables fine-grained reasoning through systematic atomic-level operations. AR decomposes the reasoning process into atomic cognitive units, employing a cognitive routing mechanism to dynamically construct reasoning representations and orchestrate inference pathways. This systematic methodology implements stepwise, structured cognition, which ensures logical coherence while significantly reducing cognitive load, effectively simulating the cognitive patterns observed in human deep thinking processes. Extensive experimental results demonstrate AR's superior reasoning capabilities without the computational burden of exhaustive solution searches, particularly excelling in linguistic logic puzzles. These findings substantiate AR's effectiveness in enhancing LLMs' capacity for robust, long-sequence logical reasoning and deliberation. 

---
# ECKGBench: Benchmarking Large Language Models in E-commerce Leveraging Knowledge Graph 

**Authors**: Langming Liu, Haibin Chen, Yuhao Wang, Yujin Yuan, Shilei Liu, Wenbo Su, Xiangyu Zhao, Bo Zheng  

**Link**: [PDF](https://arxiv.org/pdf/2503.15990)  

**Abstract**: Large language models (LLMs) have demonstrated their capabilities across various NLP tasks. Their potential in e-commerce is also substantial, evidenced by practical implementations such as platform search, personalized recommendations, and customer service. One primary concern associated with LLMs is their factuality (e.g., hallucination), which is urgent in e-commerce due to its significant impact on user experience and revenue. Despite some methods proposed to evaluate LLMs' factuality, issues such as lack of reliability, high consumption, and lack of domain expertise leave a gap between effective assessment in e-commerce. To bridge the evaluation gap, we propose ECKGBench, a dataset specifically designed to evaluate the capacities of LLMs in e-commerce knowledge. Specifically, we adopt a standardized workflow to automatically generate questions based on a large-scale knowledge graph, guaranteeing sufficient reliability. We employ the simple question-answering paradigm, substantially improving the evaluation efficiency by the least input and output tokens. Furthermore, we inject abundant e-commerce expertise in each evaluation stage, including human annotation, prompt design, negative sampling, and verification. Besides, we explore the LLMs' knowledge boundaries in e-commerce from a novel perspective. Through comprehensive evaluations of several advanced LLMs on ECKGBench, we provide meticulous analysis and insights into leveraging LLMs for e-commerce. 

---
# Uncertainty Quantification and Confidence Calibration in Large Language Models: A Survey 

**Authors**: Xiaoou Liu, Tiejin Chen, Longchao Da, Chacha Chen, Zhen Lin, Hua Wei  

**Link**: [PDF](https://arxiv.org/pdf/2503.15850)  

**Abstract**: Large Language Models (LLMs) excel in text generation, reasoning, and decision-making, enabling their adoption in high-stakes domains such as healthcare, law, and transportation. However, their reliability is a major concern, as they often produce plausible but incorrect responses. Uncertainty quantification (UQ) enhances trustworthiness by estimating confidence in outputs, enabling risk mitigation and selective prediction. However, traditional UQ methods struggle with LLMs due to computational constraints and decoding inconsistencies. Moreover, LLMs introduce unique uncertainty sources, such as input ambiguity, reasoning path divergence, and decoding stochasticity, that extend beyond classical aleatoric and epistemic uncertainty. To address this, we introduce a new taxonomy that categorizes UQ methods based on computational efficiency and uncertainty dimensions (input, reasoning, parameter, and prediction uncertainty). We evaluate existing techniques, assess their real-world applicability, and identify open challenges, emphasizing the need for scalable, interpretable, and robust UQ approaches to enhance LLM reliability. 

---
# Adaptive Group Policy Optimization: Towards Stable Training and Token-Efficient Reasoning 

**Authors**: Chen Li, Nazhou Liu, Kai Yang  

**Link**: [PDF](https://arxiv.org/pdf/2503.15952)  

**Abstract**: Since DeepSeek-R1 popularized, Group Relative Policy Optimization (GRPO) has become the core part of Reasoning LLMs training. However, we find some deficiency that influences RL stability and inference efficiency. Thus, we propose Adaptive Group Policy Optimization (AGPO) which contains two simple but effective modifications: a revised advantage estimation method to mitigate zero-variance situations; a length-based reward, incentivizing the model to avoid overthinking. The experiments demonstrate our methods achieve more stable training and comparable or superior performance with significantly fewer tokens in reasoning steps. 

---
# LLM Braces: Straightening Out LLM Predictions with Relevant Sub-Updates 

**Authors**: Ying Shen, Lifu Huang  

**Link**: [PDF](https://arxiv.org/pdf/2503.16334)  

**Abstract**: Recent findings reveal that much of the knowledge in a Transformer-based Large Language Model (LLM) is encoded in its feed-forward (FFN) layers, where each FNN layer can be interpreted as the summation of sub-updates, each corresponding to a weighted column vector from the FFN's value parameter matrix that often encodes human-interpretable concepts. In light of this, we hypothesize that model performance and behaviors can be further enhanced and controlled by modulating the contributions of these sub-updates based on their relevance to the input or target output style, and propose LLMBRACES, a novel and efficient method that computes relevance scores associated with value vectors in FFN layers and leverages these scores to dynamically adjust the contribution of sub-updates. By optimizing sub-update contributions, LLMBRACES refines the prediction process, leading to more accurate and reliable outputs, much like a 'brace' providing support and stability. Moreover, LLMBRACES can be extended to support conditional control over generation characteristics, such as sentiment, thereby offering fine-grained steering of LLM outputs. Extensive experiments on various LLMs-including Qwen2.5-1.5B, Llama2-7B, and Llama3-8B-demonstrate that LLMBRACES outperforms baseline approaches in both fine-tuning and zero-shot settings while requiring significantly fewer tunable parameters, up to 75% fewer compared to LoRA. Furthermore, LLMBRACES excels in sentiment-controlled generation and toxicity reduction, highlighting its potential for flexible, controlled text generation across applications. 

---
# Reinforcement Learning for Reasoning in Small LLMs: What Works and What Doesn't 

**Authors**: Quy-Anh Dang, Chris Ngo  

**Link**: [PDF](https://arxiv.org/pdf/2503.16219)  

**Abstract**: Enhancing the reasoning capabilities of large language models (LLMs) typically relies on massive computational resources and extensive datasets, limiting accessibility for resource-constrained settings. Our study investigates the potential of reinforcement learning (RL) to improve reasoning in small LLMs, focusing on a 1.5-billion-parameter model, DeepSeek-R1-Distill-Qwen-1.5B, under strict constraints: training on 4 NVIDIA A40 GPUs (48 GB VRAM each) within 24 hours. Adapting the Group Relative Policy Optimization (GRPO) algorithm and curating a compact, high-quality mathematical reasoning dataset, we conducted three experiments to explore model behavior and performance. Our results demonstrate rapid reasoning gains - e.g., AMC23 accuracy rising from 63% to 80% and AIME24 reaching 46.7%, surpassing o1-preview - using only 7,000 samples and a $42 training cost, compared to thousands of dollars for baseline models. However, challenges such as optimization instability and length constraints emerged with prolonged training. These findings highlight the efficacy of RL-based fine-tuning for small LLMs, offering a cost-effective alternative to large-scale approaches. We release our code and datasets as open-source resources, providing insights into trade-offs and laying a foundation for scalable, reasoning-capable LLMs in resource-limited environments. All are available at this https URL. 

---
# CodeReviewQA: The Code Review Comprehension Assessment for Large Language Models 

**Authors**: Hong Yi Lin, Chunhua Liu, Haoyu Gao, Patanamon Thongtanunam, Christoph Treude  

**Link**: [PDF](https://arxiv.org/pdf/2503.16167)  

**Abstract**: State-of-the-art large language models (LLMs) have demonstrated impressive code generation capabilities but struggle with real-world software engineering tasks, such as revising source code to address code reviews, hindering their practical use. Code review comments are often implicit, ambiguous, and colloquial, requiring models to grasp both code and human intent. This challenge calls for evaluating large language models' ability to bridge both technical and conversational contexts. While existing work has employed the automated code refinement (ACR) task to resolve these comments, current evaluation methods fall short, relying on text matching metrics that provide limited insight into model failures and remain susceptible to training data contamination. To address these limitations, we introduce a novel evaluation benchmark, $\textbf{CodeReviewQA}$ that enables us to conduct fine-grained assessment of model capabilities and mitigate data contamination risks. In CodeReviewQA, we decompose the generation task of code refinement into $\textbf{three essential reasoning steps}$: $\textit{change type recognition}$ (CTR), $\textit{change localisation}$ (CL), and $\textit{solution identification}$ (SI). Each step is reformulated as multiple-choice questions with varied difficulty levels, enabling precise assessment of model capabilities, while mitigating data contamination risks. Our comprehensive evaluation spans 72 recently released large language models on $\textbf{900 manually curated, high-quality examples}$ across nine programming languages. Our results show that CodeReviewQA is able to expose specific model weaknesses in code review comprehension, disentangled from their generative automated code refinement results. 

---
# Personalized Attacks of Social Engineering in Multi-turn Conversations -- LLM Agents for Simulation and Detection 

**Authors**: Tharindu Kumarage, Cameron Johnson, Jadie Adams, Lin Ai, Matthias Kirchner, Anthony Hoogs, Joshua Garland, Julia Hirschberg, Arslan Basharat, Huan Liu  

**Link**: [PDF](https://arxiv.org/pdf/2503.15552)  

**Abstract**: The rapid advancement of conversational agents, particularly chatbots powered by Large Language Models (LLMs), poses a significant risk of social engineering (SE) attacks on social media platforms. SE detection in multi-turn, chat-based interactions is considerably more complex than single-instance detection due to the dynamic nature of these conversations. A critical factor in mitigating this threat is understanding the mechanisms through which SE attacks operate, specifically how attackers exploit vulnerabilities and how victims' personality traits contribute to their susceptibility. In this work, we propose an LLM-agentic framework, SE-VSim, to simulate SE attack mechanisms by generating multi-turn conversations. We model victim agents with varying personality traits to assess how psychological profiles influence susceptibility to manipulation. Using a dataset of over 1000 simulated conversations, we examine attack scenarios in which adversaries, posing as recruiters, funding agencies, and journalists, attempt to extract sensitive information. Based on this analysis, we present a proof of concept, SE-OmniGuard, to offer personalized protection to users by leveraging prior knowledge of the victims personality, evaluating attack strategies, and monitoring information exchanges in conversations to identify potential SE attempts. 

---
# Enhancing Pancreatic Cancer Staging with Large Language Models: The Role of Retrieval-Augmented Generation 

**Authors**: Hisashi Johno, Yuki Johno, Akitomo Amakawa, Junichi Sato, Ryota Tozuka, Atsushi Komaba, Hiroaki Watanabe, Hiroki Watanabe, Chihiro Goto, Hiroyuki Morisaka, Hiroshi Onishi, Kazunori Nakamoto  

**Link**: [PDF](https://arxiv.org/pdf/2503.15664)  

**Abstract**: Purpose: Retrieval-augmented generation (RAG) is a technology to enhance the functionality and reliability of large language models (LLMs) by retrieving relevant information from reliable external knowledge (REK). RAG has gained interest in radiology, and we previously reported the utility of NotebookLM, an LLM with RAG (RAG-LLM), for lung cancer staging. However, since the comparator LLM differed from NotebookLM's internal model, it remained unclear whether its advantage stemmed from RAG or inherent model differences. To better isolate RAG's impact and assess its utility across different cancers, we compared NotebookLM with its internal LLM, Gemini 2.0 Flash, in a pancreatic cancer staging experiment.
Materials and Methods: A summary of Japan's pancreatic cancer staging guidelines was used as REK. We compared three groups - REK+/RAG+ (NotebookLM with REK), REK+/RAG- (Gemini 2.0 Flash with REK), and REK-/RAG- (Gemini 2.0 Flash without REK) - in staging 100 fictional pancreatic cancer cases based on CT findings. Staging criteria included TNM classification, local invasion factors, and resectability classification. In REK+/RAG+, retrieval accuracy was quantified based on the sufficiency of retrieved REK excerpts.
Results: REK+/RAG+ achieved a staging accuracy of 70%, outperforming REK+/RAG- (38%) and REK-/RAG- (35%). For TNM classification, REK+/RAG+ attained 80% accuracy, exceeding REK+/RAG- (55%) and REK-/RAG- (50%). Additionally, REK+/RAG+ explicitly presented retrieved REK excerpts, achieving a retrieval accuracy of 92%.
Conclusion: NotebookLM, a RAG-LLM, outperformed its internal LLM, Gemini 2.0 Flash, in a pancreatic cancer staging experiment, suggesting that RAG may improve LLM's staging accuracy. Furthermore, its ability to retrieve and present REK excerpts provides transparency for physicians, highlighting its applicability for clinical diagnosis and classification. 

---
# Representing data in words 

**Authors**: Amandine M. Caut, Amy Rouillard, Beimnet Zenebe, Matthias Green, Ágúst Pálmason Morthens, David J. T. Sumpter  

**Link**: [PDF](https://arxiv.org/pdf/2503.15509)  

**Abstract**: An important part of data science is the use of visualisations to display data in a way that is easy to digest. Visualisations often rely on underlying statistical or machine learning models -- ranging from basic calculations like category means to advanced methods such as principal component analysis of multidimensional datasets -- to convey insights. We introduce an analogous concept for word descriptions of data, which we call wordalisations. Wordalisations describe data in easy to digest words, without necessarily reporting numerical values from the data. We show how to create wordalisations using large language models, through prompt templates engineered according to a task-agnostic structure which can be used to automatically generate prompts from data. We show how to produce reliable and engaging texts on three application areas: scouting football players, personality tests, and international survey data. Using the model cards framework, we emphasise the importance of clearly stating the model we are imposing on the data when creating the wordalisation, detailing how numerical values are translated into words, incorporating background information into prompts for the large language model, and documenting the limitations of the wordalisations. We argue that our model cards approach is a more appropriate framework for setting best practices in wordalisation of data than performance tests on benchmark datasets. 

---
# Agreeing to Interact in Human-Robot Interaction using Large Language Models and Vision Language Models 

**Authors**: Kazuhiro Sasabuchi, Naoki Wake, Atsushi Kanehira, Jun Takamatsu, Katsushi Ikeuchi  

**Link**: [PDF](https://arxiv.org/pdf/2503.15491)  

**Abstract**: In human-robot interaction (HRI), the beginning of an interaction is often complex. Whether the robot should communicate with the human is dependent on several situational factors (e.g., the current human's activity, urgency of the interaction, etc.). We test whether large language models (LLM) and vision language models (VLM) can provide solutions to this problem. We compare four different system-design patterns using LLMs and VLMs, and test on a test set containing 84 human-robot situations. The test set mixes several publicly available datasets and also includes situations where the appropriate action to take is open-ended. Our results using the GPT-4o and Phi-3 Vision model indicate that LLMs and VLMs are capable of handling interaction beginnings when the desired actions are clear, however, challenge remains in the open-ended situations where the model must balance between the human and robot situation. 

---
# LLM-Aided Customizable Profiling of Code Data Based On Programming Language Concepts 

**Authors**: Pankaj Thorat, Adnan Qidwai, Adrija Dhar, Aishwariya Chakraborty, Anand Eswaran, Hima Patel, Praveen Jayachandran  

**Link**: [PDF](https://arxiv.org/pdf/2503.15571)  

**Abstract**: Data profiling is critical in machine learning for generating descriptive statistics, supporting both deeper understanding and downstream tasks like data valuation and curation. This work addresses profiling specifically in the context of code datasets for Large Language Models (code-LLMs), where data quality directly influences tasks such as code generation and summarization. Characterizing code datasets in terms of programming language concepts enables better insights and targeted data curation. Our proposed methodology decomposes code data profiling into two phases: (1) an offline phase where LLMs are leveraged to derive and learn rules for extracting syntactic and semantic concepts across various programming languages, including previously unseen or low-resource languages, and (2) an online deterministic phase applying these derived rules for efficient real-time analysis. This hybrid approach is customizable, extensible to new syntactic and semantic constructs, and scalable to multiple languages. Experimentally, our LLM-aided method achieves a mean accuracy of 90.33% for syntactic extraction rules and semantic classification accuracies averaging 80% and 77% across languages and semantic concepts, respectively. 

---
