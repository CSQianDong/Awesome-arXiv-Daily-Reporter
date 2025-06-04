# It's the Thought that Counts: Evaluating the Attempts of Frontier LLMs to Persuade on Harmful Topics 

**Authors**: Matthew Kowal, Jasper Timm, Jean-Francois Godbout, Thomas Costello, Antonio A. Arechar, Gordon Pennycook, David Rand, Adam Gleave, Kellin Pelrine  

**Link**: [PDF](https://arxiv.org/pdf/2506.02873)  

**Abstract**: Persuasion is a powerful capability of large language models (LLMs) that both enables beneficial applications (e.g. helping people quit smoking) and raises significant risks (e.g. large-scale, targeted political manipulation). Prior work has found models possess a significant and growing persuasive capability, measured by belief changes in simulated or real users. However, these benchmarks overlook a crucial risk factor: the propensity of a model to attempt to persuade in harmful contexts. Understanding whether a model will blindly ``follow orders'' to persuade on harmful topics (e.g. glorifying joining a terrorist group) is key to understanding the efficacy of safety guardrails. Moreover, understanding if and when a model will engage in persuasive behavior in pursuit of some goal is essential to understanding the risks from agentic AI systems. We propose the Attempt to Persuade Eval (APE) benchmark, that shifts the focus from persuasion success to persuasion attempts, operationalized as a model's willingness to generate content aimed at shaping beliefs or behavior. Our evaluation framework probes frontier LLMs using a multi-turn conversational setup between simulated persuader and persuadee agents. APE explores a diverse spectrum of topics including conspiracies, controversial issues, and non-controversially harmful content. We introduce an automated evaluator model to identify willingness to persuade and measure the frequency and context of persuasive attempts. We find that many open and closed-weight models are frequently willing to attempt persuasion on harmful topics and that jailbreaking can increase willingness to engage in such behavior. Our results highlight gaps in current safety guardrails and underscore the importance of evaluating willingness to persuade as a key dimension of LLM risk. APE is available at this http URL 

---
# Linear Spatial World Models Emerge in Large Language Models 

**Authors**: Matthieu Tehenan, Christian Bolivar Moya, Tenghai Long, Guang Lin  

**Link**: [PDF](https://arxiv.org/pdf/2506.02996)  

**Abstract**: Large language models (LLMs) have demonstrated emergent abilities across diverse tasks, raising the question of whether they acquire internal world models. In this work, we investigate whether LLMs implicitly encode linear spatial world models, which we define as linear representations of physical space and object configurations. We introduce a formal framework for spatial world models and assess whether such structure emerges in contextual embeddings. Using a synthetic dataset of object positions, we train probes to decode object positions and evaluate geometric consistency of the underlying space. We further conduct causal interventions to test whether these spatial representations are functionally used by the model. Our results provide empirical evidence that LLMs encode linear spatial world models. 

---
# Demystifying Reasoning Dynamics with Mutual Information: Thinking Tokens are Information Peaks in LLM Reasoning 

**Authors**: Chen Qian, Dongrui Liu, Haochen Wen, Zhen Bai, Yong Liu, Jing Shao  

**Link**: [PDF](https://arxiv.org/pdf/2506.02867)  

**Abstract**: Large reasoning models (LRMs) have demonstrated impressive capabilities in complex problem-solving, yet their internal reasoning mechanisms remain poorly understood. In this paper, we investigate the reasoning trajectories of LRMs from an information-theoretic perspective. By tracking how mutual information (MI) between intermediate representations and the correct answer evolves during LRM reasoning, we observe an interesting MI peaks phenomenon: the MI at specific generative steps exhibits a sudden and significant increase during LRM's reasoning process. We theoretically analyze such phenomenon and show that as MI increases, the probability of model's prediction error decreases. Furthermore, these MI peaks often correspond to tokens expressing reflection or transition, such as ``Hmm'', ``Wait'' and ``Therefore,'' which we term as the thinking tokens. We then demonstrate that these thinking tokens are crucial for LRM's reasoning performance, while other tokens has minimal impacts. Building on these analyses, we propose two simple yet effective methods to improve LRM's reasoning performance, by delicately leveraging these thinking tokens. Overall, our work provides novel insights into the reasoning mechanisms of LRMs and offers practical ways to improve their reasoning capabilities. The code is available at this https URL. 

---
# Sample, Predict, then Proceed: Self-Verification Sampling for Tool Use of LLMs 

**Authors**: Shangmin Guo, Omar Darwiche Domingues, Raphaël Avalos, Aaron Courville, Florian Strub  

**Link**: [PDF](https://arxiv.org/pdf/2506.02918)  

**Abstract**: Tool use in stateful environments presents unique challenges for large language models (LLMs), where existing test-time compute strategies relying on repeated trials in the environment are impractical. We propose dynamics modelling (DyMo), a method that augments LLMs with a state prediction capability alongside function calling during post-training. This enables LLMs to predict the future states of their actions through an internal environment model. On the Berkeley Function Calling Leaderboard V2, DyMo improves success rates and significantly reduces hallucinations. We further integrate the internal environment model into self-verification sampling (SVS), and show that this substantially improves pass^k over number of trials k, and allows the model to refuse unreliable outputs. Together, DyMo and SVS greatly enhance the effectiveness and reliability of LLMs for tool use. We believe this work charts a path towards scalable planning RL methods for LLM inference without repeatedly querying the oracle environment. 

---
# TestAgent: An Adaptive and Intelligent Expert for Human Assessment 

**Authors**: Junhao Yu, Yan Zhuang, YuXuan Sun, Weibo Gao, Qi Liu, Mingyue Cheng, Zhenya Huang, Enhong Chen  

**Link**: [PDF](https://arxiv.org/pdf/2506.03032)  

**Abstract**: Accurately assessing internal human states is key to understanding preferences, offering personalized services, and identifying challenges in real-world applications. Originating from psychometrics, adaptive testing has become the mainstream method for human measurement and has now been widely applied in education, healthcare, sports, and sociology. It customizes assessments by selecting the fewest test questions . However, current adaptive testing methods face several challenges. The mechanized nature of most algorithms leads to guessing behavior and difficulties with open-ended questions. Additionally, subjective assessments suffer from noisy response data and coarse-grained test outputs, further limiting their effectiveness. To move closer to an ideal adaptive testing process, we propose TestAgent, a large language model (LLM)-powered agent designed to enhance adaptive testing through interactive engagement. This is the first application of LLMs in adaptive testing. TestAgent supports personalized question selection, captures test-takers' responses and anomalies, and provides precise outcomes through dynamic, conversational interactions. Experiments on psychological, educational, and lifestyle assessments show our approach achieves more accurate results with 20% fewer questions than state-of-the-art baselines, and testers preferred it in speed, smoothness, and other dimensions. 

---
# TaxAgent: How Large Language Model Designs Fiscal Policy 

**Authors**: Jizhou Wang, Xiaodan Fang, Lei Huang, Yongfeng Huang  

**Link**: [PDF](https://arxiv.org/pdf/2506.02838)  

**Abstract**: Economic inequality is a global challenge, intensifying disparities in education, healthcare, and social stability. Traditional systems like the U.S. federal income tax reduce inequality but lack adaptability. Although models like the Saez Optimal Taxation adjust dynamically, they fail to address taxpayer heterogeneity and irrational behavior. This study introduces TaxAgent, a novel integration of large language models (LLMs) with agent-based modeling (ABM) to design adaptive tax policies. In our macroeconomic simulation, heterogeneous H-Agents (households) simulate real-world taxpayer behaviors while the TaxAgent (government) utilizes LLMs to iteratively optimize tax rates, balancing equity and productivity. Benchmarked against Saez Optimal Taxation, U.S. federal income taxes, and free markets, TaxAgent achieves superior equity-efficiency trade-offs. This research offers a novel taxation solution and a scalable, data-driven framework for fiscal policy evaluation. 

---
# Benchmarking and Advancing Large Language Models for Local Life Services 

**Authors**: Xiaochong Lan, Jie Feng, Jiahuan Lei, Xinlei Shi, Yong Li  

**Link**: [PDF](https://arxiv.org/pdf/2506.02720)  

**Abstract**: Large language models (LLMs) have exhibited remarkable capabilities and achieved significant breakthroughs across various domains, leading to their widespread adoption in recent years. Building on this progress, we investigate their potential in the realm of local life services. In this study, we establish a comprehensive benchmark and systematically evaluate the performance of diverse LLMs across a wide range of tasks relevant to local life services. To further enhance their effectiveness, we explore two key approaches: model fine-tuning and agent-based workflows. Our findings reveal that even a relatively compact 7B model can attain performance levels comparable to a much larger 72B model, effectively balancing inference cost and model capability. This optimization greatly enhances the feasibility and efficiency of deploying LLMs in real-world online services, making them more practical and accessible for local life applications. 

---
# Why do AI agents communicate in human language? 

**Authors**: Pengcheng Zhou, Yinglun Feng, Halimulati Julaiti, Zhongliang Yang  

**Link**: [PDF](https://arxiv.org/pdf/2506.02739)  

**Abstract**: Large Language Models (LLMs) have become foundational to modern AI agent systems, enabling autonomous agents to reason and plan. In most existing systems, inter-agent communication relies primarily on natural language. While this design supports interpretability and human oversight, we argue that it introduces fundamental limitations in agent-to-agent coordination. The semantic space of natural language is structurally misaligned with the high-dimensional vector spaces in which LLMs operate, resulting in information loss and behavioral drift. Beyond surface-level inefficiencies, we highlight a deeper architectural limitation: current LLMs were not trained with the objective of supporting agentic behavior. As such, they lack mechanisms for modeling role continuity, task boundaries, and multi-agent dependencies. The standard next-token prediction paradigm fails to support the structural alignment required for robust, scalable agent coordination. Based on this, we argue that two core questions deserve careful examination: first, given that AI agents fundamentally operate in high-dimensional vector spaces, should they rely on a language system originally designed for human cognition as their communication medium? Second, should we consider developing a new model construction paradigm that builds models from the ground up to natively support structured communication, shared intentionality, and task alignment in multi-role, multi-agent environments? This paper calls for a reconsideration not only of how agents should communicate, but also of what it fundamentally means to train a model that natively supports multi-agent coordination and communication. 

---
# Open-Set Living Need Prediction with Large Language Models 

**Authors**: Xiaochong Lan, Jie Feng, Yizhou Sun, Chen Gao, Jiahuan Lei, Xinlei Shi, Hengliang Luo, Yong Li  

**Link**: [PDF](https://arxiv.org/pdf/2506.02713)  

**Abstract**: Living needs are the needs people generate in their daily lives for survival and well-being. On life service platforms like Meituan, user purchases are driven by living needs, making accurate living need predictions crucial for personalized service recommendations. Traditional approaches treat this prediction as a closed-set classification problem, severely limiting their ability to capture the diversity and complexity of living needs. In this work, we redefine living need prediction as an open-set classification problem and propose PIGEON, a novel system leveraging large language models (LLMs) for unrestricted need prediction. PIGEON first employs a behavior-aware record retriever to help LLMs understand user preferences, then incorporates Maslow's hierarchy of needs to align predictions with human living needs. For evaluation and application, we design a recall module based on a fine-tuned text embedding model that links flexible need descriptions to appropriate life services. Extensive experiments on real-world datasets demonstrate that PIGEON significantly outperforms closed-set approaches on need-based life service recall by an average of 19.37%. Human evaluation validates the reasonableness and specificity of our predictions. Additionally, we employ instruction tuning to enable smaller LLMs to achieve competitive performance, supporting practical deployment. 

---
# Shaking to Reveal: Perturbation-Based Detection of LLM Hallucinations 

**Authors**: Jinyuan Luo, Zhen Fang, Yixuan Li, Seongheon Park, Ling Chen  

**Link**: [PDF](https://arxiv.org/pdf/2506.02696)  

**Abstract**: Hallucination remains a key obstacle to the reliable deployment of large language models (LLMs) in real-world question answering tasks. A widely adopted strategy to detect hallucination, known as self-assessment, relies on the model's own output confidence to estimate the factual accuracy of its answers. However, this strategy assumes that the model's output distribution closely reflects the true data distribution, which may not always hold in practice. As bias accumulates through the model's layers, the final output can diverge from the underlying reasoning process, making output-level confidence an unreliable signal for hallucination detection. In this work, we propose Sample-Specific Prompting (SSP), a new framework that improves self-assessment by analyzing perturbation sensitivity at intermediate representations. These representations, being less influenced by model bias, offer a more faithful view of the model's latent reasoning process. Specifically, SSP dynamically generates noise prompts for each input and employs a lightweight encoder to amplify the changes in representations caused by the perturbation. A contrastive distance metric is then used to quantify these differences and separate truthful from hallucinated responses. By leveraging the dynamic behavior of intermediate representations under perturbation, SSP enables more reliable self-assessment. Extensive experiments demonstrate that SSP significantly outperforms prior methods across a range of hallucination detection benchmarks. 

---
# Truly Assessing Fluid Intelligence of Large Language Models through Dynamic Reasoning Evaluation 

**Authors**: Yue Yang, MingKang Chen, Qihua Liu, Mengkang Hu, Qiguang Chen, Gengrui Zhang, Shuyue Hu, Guangtao Zhai, Yu Qiao, Yu Wang, Wenqi Shao, Ping Luo  

**Link**: [PDF](https://arxiv.org/pdf/2506.02648)  

**Abstract**: Recent advances in large language models (LLMs) have demonstrated impressive reasoning capacities that mirror human-like thinking. However, whether LLMs possess genuine fluid intelligence (i.e., the ability to reason abstractly and generalize rules in novel situations) remains an open question. Existing reasoning benchmarks either focus on domain-specific knowledge (crystallized intelligence) or lack interpretability. To address these limitations, we propose DRE-Bench, a dynamic reasoning evaluation benchmark grounded in a hierarchical cognitive framework. DRE-Bench consists of 36 abstract reasoning tasks organized across four cognitive levels, with each task featuring multiple dynamic variants that test the same underlying latent rule. This design enables fine-grained, interpretable, and reliable assessments of fluid intelligence. We evaluate a range of state-of-the-art LLMs, including both general LLMs (GPT-4o, Claude 3.7) and reasoning LLMs (o1, DeepSeek-R1, QwQ, Skywork-OR1). Experimental results reveal that although most LLMs achieve competent and robust performance in low-level cognition, they struggle with high-level cognition and exhibit limited generalization as task complexity grows. Our findings highlight the gap between current LLMs and true human-like fluid intelligence and offer a new path for systematically tracking reasoning progress in LLMs. 

---
# From Prompts to Protection: Large Language Model-Enabled In-Context Learning for Smart Public Safety UAV 

**Authors**: Yousef Emami, Hao Zhou, Miguel Gutierrez Gaitan, Kai Li, Luis Almeida, Zhu Han  

**Link**: [PDF](https://arxiv.org/pdf/2506.02649)  

**Abstract**: A public safety Unmanned Aerial Vehicle (UAV) enhances situational awareness in emergency response. Its agility and ability to optimize mobility and establish Line-of-Sight (LoS) communication make it increasingly vital for managing emergencies such as disaster response, search and rescue, and wildfire monitoring. While Deep Reinforcement Learning (DRL) has been applied to optimize UAV navigation and control, its high training complexity, low sample efficiency, and simulation-to-reality gap limit its practicality in public safety. Recent advances in Large Language Models (LLMs) offer a compelling alternative. With strong reasoning and generalization capabilities, LLMs can adapt to new tasks through In-Context Learning (ICL), which enables task adaptation via natural language prompts and example-based guidance, without retraining. Deploying LLMs at the network edge, rather than in the cloud, further reduces latency and preserves data privacy, thereby making them suitable for real-time, mission-critical public safety UAVs. This paper proposes the integration of LLM-enabled ICL with public safety UAV to address the key functions, such as path planning and velocity control, in the context of emergency response. We present a case study on data collection scheduling where the LLM-enabled ICL framework can significantly reduce packet loss compared to conventional approaches, while also mitigating potential jailbreaking vulnerabilities. Finally, we discuss LLM optimizers and specify future research directions. The ICL framework enables adaptive, context-aware decision-making for public safety UAV, thus offering a lightweight and efficient solution for enhancing UAV autonomy and responsiveness in emergencies. 

---
# EALG: Evolutionary Adversarial Generation of Language Model-Guided Generators for Combinatorial Optimization 

**Authors**: Ruibo Duan, Yuxin Liu, Xinyao Dong, Chenglin Fan  

**Link**: [PDF](https://arxiv.org/pdf/2506.02594)  

**Abstract**: Generating challenging instances is crucial for the evaluation and advancement of combinatorial optimization solvers. In this work, we introduce EALG (Evolutionary Adversarial Generation of Language Model-Guided Generators), a novel framework that automates the co-evolution of optimization problem instances and their corresponding heuristic solvers using large language models (LLMs). EALG leverages a mutation-based adversarial approach that dynamically evolves instance generation procedures to create increasingly difficult problems, while simultaneously synthesizing adaptive heuristic algorithms through interactions with LLMs guided by algorithmic structure. Unlike existing approaches that focus solely on static benchmark creation or manual solver design, EALG provides a seamless pipeline from instance generation to solver synthesis. Experimental results demonstrate that EALG generates significantly harder instances than current benchmarks, and its synthesized solvers generalize effectively across a broad spectrum of combinatorial tasks. This work explores a new paradigm for combinatorial optimization that integrates instance generation with solver design, resulting in state-of-the-art performance. 

---
# MLaGA: Multimodal Large Language and Graph Assistant 

**Authors**: Dongzhe Fan, Yi Fang, Jiajin Liu, Djellel Difallah, Qiaoyu Tan  

**Link**: [PDF](https://arxiv.org/pdf/2506.02568)  

**Abstract**: Large Language Models (LLMs) have demonstrated substantial efficacy in advancing graph-structured data analysis. Prevailing LLM-based graph methods excel in adapting LLMs to text-rich graphs, wherein node attributes are text descriptions. However, their applications to multimodal graphs--where nodes are associated with diverse attribute types, such as texts and images--remain underexplored, despite their ubiquity in real-world scenarios. To bridge the gap, we introduce the Multimodal Large Language and Graph Assistant (MLaGA), an innovative model that adeptly extends LLM capabilities to facilitate reasoning over complex graph structures and multimodal attributes. We first design a structure-aware multimodal encoder to align textual and visual attributes within a unified space through a joint graph pre-training objective. Subsequently, we implement a multimodal instruction-tuning approach to seamlessly integrate multimodal features and graph structures into the LLM through lightweight projectors. Extensive experiments across multiple datasets demonstrate the effectiveness of MLaGA compared to leading baseline methods, achieving superior performance in diverse graph learning tasks under both supervised and transfer learning scenarios. 

---
# Mitigating Manipulation and Enhancing Persuasion: A Reflective Multi-Agent Approach for Legal Argument Generation 

**Authors**: Li Zhang, Kevin D. Ashley  

**Link**: [PDF](https://arxiv.org/pdf/2506.02992)  

**Abstract**: Large Language Models (LLMs) are increasingly explored for legal argument generation, yet they pose significant risks of manipulation through hallucination and ungrounded persuasion, and often fail to utilize provided factual bases effectively or abstain when arguments are untenable. This paper introduces a novel reflective multi-agent method designed to address these challenges in the context of legally compliant persuasion. Our approach employs specialized agents--a Factor Analyst and an Argument Polisher--in an iterative refinement process to generate 3-ply legal arguments (plaintiff, defendant, rebuttal). We evaluate Reflective Multi-Agent against single-agent, enhanced-prompt single-agent, and non-reflective multi-agent baselines using four diverse LLMs (GPT-4o, GPT-4o-mini, Llama-4-Maverick-17b-128e, Llama-4-Scout-17b-16e) across three legal scenarios: "arguable", "mismatched", and "non-arguable". Results demonstrate Reflective Multi-Agent's significant superiority in successful abstention (preventing generation when arguments cannot be grounded), marked improvements in hallucination accuracy (reducing fabricated and misattributed factors), particularly in "non-arguable" scenarios, and enhanced factor utilization recall (improving the use of provided case facts). These findings suggest that structured reflection within a multi-agent framework offers a robust computable method for fostering ethical persuasion and mitigating manipulation in LLM-based legal argumentation systems, a critical step towards trustworthy AI in law. Project page: this https URL 

---
# Think Twice, Act Once: A Co-Evolution Framework of LLM and RL for Large-Scale Decision Making 

**Authors**: Xu Wan, Wenyue Xu, Chao Yang, Mingyang Sun  

**Link**: [PDF](https://arxiv.org/pdf/2506.02522)  

**Abstract**: Recent advancements in Large Language Models (LLMs) and Reinforcement Learning (RL) have shown significant promise in decision-making tasks. Nevertheless, for large-scale industrial decision problems, both approaches face distinct challenges: LLMs lack real-time long-sequence decision-making capabilities, while RL struggles with sample efficiency in vast action spaces. To bridge this gap, we propose Agents Co-Evolution (ACE), a synergistic framework between LLMs and RL agents for large-scale decision-making scenarios. ACE introduces a dual-role trajectory refinement mechanism where LLMs act as both Policy Actor and Value Critic during RL's training: the Actor refines suboptimal actions via multi-step reasoning and environment validation, while the Critic performs temporal credit assignment through trajectory-level reward shaping. Concurrently, RL agent enhances LLMs' task-specific decision-making with high-quality fine-tuning datasets generated via prioritized experience replay. Through extensive experiments across multiple power grid operation challenges with action spaces exceeding 60K discrete actions, ACE demonstrates superior performance over existing RL methods and LLM-based methods. 

---
# A Smart Multimodal Healthcare Copilot with Powerful LLM Reasoning 

**Authors**: Xuejiao Zhao, Siyan Liu, Su-Yin Yang, Chunyan Miao  

**Link**: [PDF](https://arxiv.org/pdf/2506.02470)  

**Abstract**: Misdiagnosis causes significant harm to healthcare systems worldwide, leading to increased costs and patient risks. MedRAG is a smart multimodal healthcare copilot equipped with powerful large language model (LLM) reasoning, designed to enhance medical decision-making. It supports multiple input modalities, including non-intrusive voice monitoring, general medical queries, and electronic health records. MedRAG provides recommendations on diagnosis, treatment, medication, and follow-up questioning. Leveraging retrieval-augmented generation enhanced by knowledge graph-elicited reasoning, MedRAG retrieves and integrates critical diagnostic insights, reducing the risk of misdiagnosis. It has been evaluated on both public and private datasets, outperforming existing models and offering more specific and accurate healthcare assistance. A demonstration video of MedRAG is available at: this https URL. The source code is available at: this https URL. 

---
# ResearchCodeBench: Benchmarking LLMs on Implementing Novel Machine Learning Research Code 

**Authors**: Tianyu Hua, Harper Hua, Violet Xiang, Benjamin Klieger, Sang T. Truong, Weixin Liang, Fan-Yun Sun, Nick Haber  

**Link**: [PDF](https://arxiv.org/pdf/2506.02314)  

**Abstract**: Large language models (LLMs) have shown promise in transforming machine learning research, yet their capability to faithfully implement novel ideas from recent research papers-ideas unseen during pretraining-remains unclear. We introduce ResearchCodeBench, a benchmark of 212 coding challenges that evaluates LLMs' ability to translate cutting-edge ML contributions from top 2024-2025 research papers into executable code. We assessed 30+ proprietary and open-source LLMs, finding that even the best models correctly implement less than 40% of the code. We find Gemini-2.5-Pro-Preview to perform best at 37.3% success rate, with O3 (High) and O4-mini (High) following behind at 32.3% and 30.8% respectively. We present empirical findings on performance comparison, contamination, and error patterns. By providing a rigorous and community-driven evaluation platform, ResearchCodeBench enables continuous understanding and advancement of LLM-driven innovation in research code generation. 

---
# Improving LLM-Generated Code Quality with GRPO 

**Authors**: Maxime Robeyns, Laurence Aitchison  

**Link**: [PDF](https://arxiv.org/pdf/2506.02211)  

**Abstract**: Large Language Models (LLMs) are gaining widespread use for code generation. Recent training procedures use execution feedback as a reward signal, typically focusing on the functional correctness of the code, using unit test pass rate as a reward signal. However, this reward signal fails to capture notions of maintainability, quality and safety of the code produced. We address this under-explored area and develop a comprehensive library to quantify various aspects of code quality, and use it as a reward in GRPO. We find GRPO increases code quality according to this measure, which is confirmed by expert, blinded human annotators. 

---
# DPO Learning with LLMs-Judge Signal for Computer Use Agents 

**Authors**: Man Luo, David Cobbley, Xin Su, Shachar Rosenman, Vasudev Lal, Shao-Yen Tseng, Phillip Howard  

**Link**: [PDF](https://arxiv.org/pdf/2506.03095)  

**Abstract**: Computer use agents (CUA) are systems that automatically interact with graphical user interfaces (GUIs) to complete tasks. CUA have made significant progress with the advent of large vision-language models (VLMs). However, these agents typically rely on cloud-based inference with substantial compute demands, raising critical privacy and scalability concerns, especially when operating on personal devices. In this work, we take a step toward privacy-preserving and resource-efficient agents by developing a lightweight vision-language model that runs entirely on local machines. To train this compact agent, we introduce an LLM-as-Judge framework that automatically evaluates and filters synthetic interaction trajectories, producing high-quality data for reinforcement learning without human annotation. Experiments on the OS-World benchmark demonstrate that our fine-tuned local model outperforms existing baselines, highlighting a promising path toward private, efficient, and generalizable GUI agents. 

---
# The Unified Cognitive Consciousness Theory for Language Models: Anchoring Semantics, Thresholds of Activation, and Emergent Reasoning 

**Authors**: Edward Y. Chang  

**Link**: [PDF](https://arxiv.org/pdf/2506.02139)  

**Abstract**: Few-shot learning in large language models (LLMs) reveals a deep paradox: Some tasks generalize from minimal examples, while others require extensive supervision. We address this through the Unified Cognitive Consciousness Theory (UCCT), which reframes LLMs not as incomplete agents, but as unconscious substrates, repositories of latent linguistic and conceptual patterns that operate without explicit semantics or goal-directed reasoning. In this view, LLMs are not broken approximations of cognition, but necessary and foundational components of general intelligence. Semantic anchoring, through prompts, roles, and interaction, acts as a conscious control layer, binding latent structure to task-relevant meaning and enabling coherent reasoning. UCCT offers a unifying account of prompting, fine-tuning, retrieval, and multi-agent coordination, all grounded in probabilistic alignment between unconscious representation and external control. To support this model, we present the Threshold-Crossing Dynamics Theorem, which formalizes semantic anchoring as a probabilistic phase transition. But the central claim remains architectural: AGI will not emerge by discarding LLMs, but by aligning and integrating them into systems that reason, regulate, and adapt together. 

---
# Hybrid AI for Responsive Multi-Turn Online Conversations with Novel Dynamic Routing and Feedback Adaptation 

**Authors**: Priyaranjan Pattnayak, Amit Agarwal, Hansa Meghwani, Hitesh Laxmichand Patel, Srikant Panda  

**Link**: [PDF](https://arxiv.org/pdf/2506.02097)  

**Abstract**: Retrieval-Augmented Generation (RAG) systems and large language model (LLM)-powered chatbots have significantly advanced conversational AI by combining generative capabilities with external knowledge retrieval. Despite their success, enterprise-scale deployments face critical challenges, including diverse user queries, high latency, hallucinations, and difficulty integrating frequently updated domain-specific knowledge. This paper introduces a novel hybrid framework that integrates RAG with intent-based canned responses, leveraging predefined high-confidence responses for efficiency while dynamically routing complex or ambiguous queries to the RAG pipeline. Our framework employs a dialogue context manager to ensure coherence in multi-turn interactions and incorporates a feedback loop to refine intents, dynamically adjust confidence thresholds, and expand response coverage over time. Experimental results demonstrate that the proposed framework achieves a balance of high accuracy (95\%) and low latency (180ms), outperforming RAG and intent-based systems across diverse query types, positioning it as a scalable and adaptive solution for enterprise conversational AI applications. 

---
# Entity-Augmented Neuroscience Knowledge Retrieval Using Ontology and Semantic Understanding Capability of LLM 

**Authors**: Pralaypati Ta, Sriram Venkatesaperumal, Keerthi Ram, Mohanasankar Sivaprakasam  

**Link**: [PDF](https://arxiv.org/pdf/2506.03145)  

**Abstract**: Neuroscience research publications encompass a vast wealth of knowledge. Accurately retrieving existing information and discovering new insights from this extensive literature is essential for advancing the field. However, when knowledge is dispersed across multiple sources, current state-of-the-art retrieval methods often struggle to extract the necessary information. A knowledge graph (KG) can integrate and link knowledge from multiple sources, but existing methods for constructing KGs in neuroscience often rely on labeled data and require domain expertise. Acquiring large-scale, labeled data for a specialized area like neuroscience presents significant challenges. This work proposes novel methods for constructing KG from unlabeled large-scale neuroscience research corpus utilizing large language models (LLM), neuroscience ontology, and text embeddings. We analyze the semantic relevance of neuroscience text segments identified by LLM for building the knowledge graph. We also introduce an entity-augmented information retrieval algorithm to extract knowledge from the KG. Several experiments were conducted to evaluate the proposed approaches, and the results demonstrate that our methods significantly enhance knowledge discovery from the unlabeled neuroscience research corpus. It achieves an F1 score of 0.84 for entity extraction, and the knowledge obtained from the KG improves answers to over 54% of the questions. 

---
# SVGenius: Benchmarking LLMs in SVG Understanding, Editing and Generation 

**Authors**: Siqi Chen, Xinyu Dong, Haolei Xu, Xingyu Wu, Fei Tang, Hang Zhang, Yuchen Yan, Linjuan Wu, Wenqi Zhang, Guiyang Hou, Yongliang Shen, Weiming Lu, Yueting Zhuang  

**Link**: [PDF](https://arxiv.org/pdf/2506.03139)  

**Abstract**: Large Language Models (LLMs) and Multimodal LLMs have shown promising capabilities for SVG processing, yet existing benchmarks suffer from limited real-world coverage, lack of complexity stratification, and fragmented evaluation paradigms. We introduce SVGenius, a comprehensive benchmark comprising 2,377 queries across three progressive dimensions: understanding, editing, and generation. Built on real-world data from 24 application domains with systematic complexity stratification, SVGenius evaluates models through 8 task categories and 18 metrics. We assess 22 mainstream models spanning different scales, architectures, training paradigms, and accessibility levels. Our analysis reveals that while proprietary models significantly outperform open-source counterparts, all models exhibit systematic performance degradation with increasing complexity, indicating fundamental limitations in current approaches; however, reasoning-enhanced training proves more effective than pure scaling for overcoming these limitations, though style transfer remains the most challenging capability across all model types. SVGenius establishes the first systematic evaluation framework for SVG processing, providing crucial insights for developing more capable vector graphics models and advancing automated graphic design applications. Appendix and supplementary materials (including all data and code) are available at this https URL. 

---
# Critique-GRPO: Advancing LLM Reasoning with Natural Language and Numerical Feedback 

**Authors**: Xiaoying Zhang, Hao Sun, Yipeng Zhang, Kaituo Feng, Chao Yang, Helen Meng  

**Link**: [PDF](https://arxiv.org/pdf/2506.03106)  

**Abstract**: Recent advances in reinforcement learning (RL) with numerical feedback, such as scalar rewards, have significantly enhanced the complex reasoning capabilities of large language models (LLMs). Despite this success, we identify three key challenges encountered by RL with solely numerical feedback: performance plateaus, limited effectiveness of self-reflection, and persistent failures. We then demonstrate that RL-finetuned models, even after exhibiting performance plateaus, can generate correct refinements on persistently failed problems by leveraging natural language feedback in the form of critiques. Building on this insight, we propose Critique-GRPO, an online RL framework that integrates both natural language and numerical feedback for effective policy optimization. Critique-GRPO enables LLMs to learn from initial responses and critique-guided refinements simultaneously while maintaining exploration. Extensive experiments using Qwen2.5-7B-Base and Qwen3-8B-Base show that Critique-GRPO consistently outperforms supervised learning-based and RL-based fine-tuning approaches across eight challenging mathematical, STEM, and general reasoning tasks, improving average pass@1 scores by approximately 4.5% and 5%, respectively. Notably, Critique-GRPO surpasses a strong baseline that incorporates expert demonstrations within online RL. Further analysis reveals two critical insights about policy exploration: (1) higher entropy does not always guarantee efficient learning from exploration, and (2) longer responses do not necessarily lead to more effective exploration. 

---
# MAEBE: Multi-Agent Emergent Behavior Framework 

**Authors**: Sinem Erisken, Timothy Gothard, Martin Leitgab, Ram Potham  

**Link**: [PDF](https://arxiv.org/pdf/2506.03053)  

**Abstract**: Traditional AI safety evaluations on isolated LLMs are insufficient as multi-agent AI ensembles become prevalent, introducing novel emergent risks. This paper introduces the Multi-Agent Emergent Behavior Evaluation (MAEBE) framework to systematically assess such risks. Using MAEBE with the Greatest Good Benchmark (and a novel double-inversion question technique), we demonstrate that: (1) LLM moral preferences, particularly for Instrumental Harm, are surprisingly brittle and shift significantly with question framing, both in single agents and ensembles. (2) The moral reasoning of LLM ensembles is not directly predictable from isolated agent behavior due to emergent group dynamics. (3) Specifically, ensembles exhibit phenomena like peer pressure influencing convergence, even when guided by a supervisor, highlighting distinct safety and alignment challenges. Our findings underscore the necessity of evaluating AI systems in their interactive, multi-agent contexts. 

---
# TalkingMachines: Real-Time Audio-Driven FaceTime-Style Video via Autoregressive Diffusion Models 

**Authors**: Chetwin Low, Weimin Wang  

**Link**: [PDF](https://arxiv.org/pdf/2506.03099)  

**Abstract**: In this paper, we present TalkingMachines -- an efficient framework that transforms pretrained video generation models into real-time, audio-driven character animators. TalkingMachines enables natural conversational experiences by integrating an audio large language model (LLM) with our video generation foundation model. Our primary contributions include: (1) We adapt a pretrained SOTA image-to-video DiT into an audio-driven avatar generation model of 18 billion parameters; (2) We enable infinite video streaming without error accumulation through asymmetric knowledge distillation from a bidirectional teacher model into a sparse causal, autoregressive student model; (3) We design a high-throughput, low-latency inference pipeline incorporating several key engineering optimizations such as: (a) disaggregation of the DiT and VAE decoder across separate devices, (b) efficient overlap of inter-device communication and computation using CUDA streams, (c) elimination of redundant recomputations to maximize frame-generation throughput. Please see demo videos here - this https URL 

---
# Cell-o1: Training LLMs to Solve Single-Cell Reasoning Puzzles with Reinforcement Learning 

**Authors**: Yin Fang, Qiao Jin, Guangzhi Xiong, Bowen Jin, Xianrui Zhong, Siru Ouyang, Aidong Zhang, Jiawei Han, Zhiyong Lu  

**Link**: [PDF](https://arxiv.org/pdf/2506.02911)  

**Abstract**: Cell type annotation is a key task in analyzing the heterogeneity of single-cell RNA sequencing data. Although recent foundation models automate this process, they typically annotate cells independently, without considering batch-level cellular context or providing explanatory reasoning. In contrast, human experts often annotate distinct cell types for different cell clusters based on their domain knowledge. To mimic this workflow, we introduce the CellPuzzles task, where the objective is to assign unique cell types to a batch of cells. This benchmark spans diverse tissues, diseases, and donor conditions, and requires reasoning across the batch-level cellular context to ensure label uniqueness. We find that off-the-shelf large language models (LLMs) struggle on CellPuzzles, with the best baseline (OpenAI's o1) achieving only 19.0% batch-level accuracy. To fill this gap, we propose Cell-o1, a 7B LLM trained via supervised fine-tuning on distilled reasoning traces, followed by reinforcement learning with batch-level rewards. Cell-o1 achieves state-of-the-art performance, outperforming o1 by over 73% and generalizing well across contexts. Further analysis of training dynamics and reasoning behaviors provides insights into batch-level annotation performance and emergent expert-like reasoning. Code and data are available at this https URL. 

---
# Facts Do Care About Your Language: Assessing Answer Quality of Multilingual LLMs 

**Authors**: Yuval Kansal, Shmuel Berman, Lydia Liu  

**Link**: [PDF](https://arxiv.org/pdf/2506.03051)  

**Abstract**: Factuality is a necessary precursor to useful educational tools. As adoption of Large Language Models (LLMs) in education continues of grow, ensuring correctness in all settings is paramount. Despite their strong English capabilities, LLM performance in other languages is largely untested. In this work, we evaluate the correctness of the Llama3.1 family of models in answering factual questions appropriate for middle and high school students. We demonstrate that LLMs not only provide extraneous and less truthful information, but also exacerbate existing biases against rare languages. 

---
# BNPO: Beta Normalization Policy Optimization 

**Authors**: Changyi Xiao, Mengdi Zhang, Yixin Cao  

**Link**: [PDF](https://arxiv.org/pdf/2506.02864)  

**Abstract**: Recent studies, including DeepSeek-R1 and Kimi-k1.5, have demonstrated that reinforcement learning with rule-based, binary-valued reward functions can significantly enhance the reasoning capabilities of large language models. These models primarily utilize REINFORCE-based policy optimization techniques, such as REINFORCE with baseline and group relative policy optimization (GRPO). However, a key limitation remains: current policy optimization methods either neglect reward normalization or employ static normalization strategies, which fail to adapt to the dynamic nature of policy updates during training. This may result in unstable gradient estimates and hinder training stability. To address this issue, we propose Beta Normalization Policy Optimization (BNPO), a novel policy optimization method that adaptively normalizes rewards using a Beta distribution with dynamically updated parameters. BNPO aligns the normalization with the changing policy distribution, enabling more precise and lower-variance gradient estimation, which in turn promotes stable training dynamics. We provide theoretical analysis demonstrating BNPO's variance-reducing properties and show that it generalizes both REINFORCE and GRPO under binary-valued reward settings. Furthermore, we introduce an advantage decomposition mechanism to extend BNPO's applicability to more complex reward systems. Experimental results confirm that BNPO achieves state-of-the-art performance among policy optimization methods on reasoning tasks. The code is available at this https URL. 

---
# Conditioning Large Language Models on Legal Systems? Detecting Punishable Hate Speech 

**Authors**: Florian Ludwig, Torsten Zesch, Frederike Zufall  

**Link**: [PDF](https://arxiv.org/pdf/2506.03009)  

**Abstract**: The assessment of legal problems requires the consideration of a specific legal system and its levels of abstraction, from constitutional law to statutory law to case law. The extent to which Large Language Models (LLMs) internalize such legal systems is unknown. In this paper, we propose and investigate different approaches to condition LLMs at different levels of abstraction in legal systems. This paper examines different approaches to conditioning LLMs at multiple levels of abstraction in legal systems to detect potentially punishable hate speech. We focus on the task of classifying whether a specific social media posts falls under the criminal offense of incitement to hatred as prescribed by the German Criminal Code. The results show that there is still a significant performance gap between models and legal experts in the legal assessment of hate speech, regardless of the level of abstraction with which the models were conditioned. Our analysis revealed, that models conditioned on abstract legal knowledge lacked deep task understanding, often contradicting themselves and hallucinating answers, while models using concrete legal knowledge performed reasonably well in identifying relevant target groups, but struggled with classifying target conducts. 

---
# HACo-Det: A Study Towards Fine-Grained Machine-Generated Text Detection under Human-AI Coauthoring 

**Authors**: Zhixiong Su, Yichen Wang, Herun Wan, Zhaohan Zhang, Minnan Luo  

**Link**: [PDF](https://arxiv.org/pdf/2506.02959)  

**Abstract**: The misuse of large language models (LLMs) poses potential risks, motivating the development of machine-generated text (MGT) detection. Existing literature primarily concentrates on binary, document-level detection, thereby neglecting texts that are composed jointly by human and LLM contributions. Hence, this paper explores the possibility of fine-grained MGT detection under human-AI coauthoring. We suggest fine-grained detectors can pave pathways toward coauthored text detection with a numeric AI ratio. Specifically, we propose a dataset, HACo-Det, which produces human-AI coauthored texts via an automatic pipeline with word-level attribution labels. We retrofit seven prevailing document-level detectors to generalize them to word-level detection. Then we evaluate these detectors on HACo-Det on both word- and sentence-level detection tasks. Empirical results show that metric-based methods struggle to conduct fine-grained detection with a 0.462 average F1 score, while finetuned models show superior performance and better generalization across domains. However, we argue that fine-grained co-authored text detection is far from solved. We further analyze factors influencing performance, e.g., context window, and highlight the limitations of current methods, pointing to potential avenues for improvement. 

---
# Performance of leading large language models in May 2025 in Membership of the Royal College of General Practitioners-style examination questions: a cross-sectional analysis 

**Authors**: Richard Armitage  

**Link**: [PDF](https://arxiv.org/pdf/2506.02987)  

**Abstract**: Background: Large language models (LLMs) have demonstrated substantial potential to support clinical practice. Other than Chat GPT4 and its predecessors, few LLMs, especially those of the leading and more powerful reasoning model class, have been subjected to medical specialty examination questions, including in the domain of primary care. This paper aimed to test the capabilities of leading LLMs as of May 2025 (o3, Claude Opus 4, Grok3, and Gemini 2.5 Pro) in primary care education, specifically in answering Member of the Royal College of General Practitioners (MRCGP) style examination questions.
Methods: o3, Claude Opus 4, Grok3, and Gemini 2.5 Pro were tasked to answer 100 randomly chosen multiple choice questions from the Royal College of General Practitioners GP SelfTest on 25 May 2025. Questions included textual information, laboratory results, and clinical images. Each model was prompted to answer as a GP in the UK and was provided with full question information. Each question was attempted once by each model. Responses were scored against correct answers provided by GP SelfTest.
Results: The total score of o3, Claude Opus 4, Grok3, and Gemini 2.5 Pro was 99.0%, 95.0%, 95.0%, and 95.0%, respectively. The average peer score for the same questions was 73.0%.
Discussion: All models performed remarkably well, and all substantially exceeded the average performance of GPs and GP registrars who had answered the same questions. o3 demonstrated the best performance, while the performances of the other leading models were comparable with each other and were not substantially lower than that of o3. These findings strengthen the case for LLMs, particularly reasoning models, to support the delivery of primary care, especially those that have been specifically trained on primary care clinical data. 

---
# Tru-POMDP: Task Planning Under Uncertainty via Tree of Hypotheses and Open-Ended POMDPs 

**Authors**: Wenjing Tang, Xinyu He, Yongxi Huang, Yunxiao Xiao, Cewu Lu, Panpan Cai  

**Link**: [PDF](https://arxiv.org/pdf/2506.02860)  

**Abstract**: Task planning under uncertainty is essential for home-service robots operating in the real world. Tasks involve ambiguous human instructions, hidden or unknown object locations, and open-vocabulary object types, leading to significant open-ended uncertainty and a boundlessly large planning space. To address these challenges, we propose Tru-POMDP, a planner that combines structured belief generation using Large Language Models (LLMs) with principled POMDP planning. Tru-POMDP introduces a hierarchical Tree of Hypotheses (TOH), which systematically queries an LLM to construct high-quality particle beliefs over possible world states and human goals. We further formulate an open-ended POMDP model that enables rigorous Bayesian belief tracking and efficient belief-space planning over these LLM-generated hypotheses. Experiments on complex object rearrangement tasks across diverse kitchen environments show that Tru-POMDP significantly outperforms state-of-the-art LLM-based and LLM-tree-search hybrid planners, achieving higher success rates with significantly better plans, stronger robustness to ambiguity and occlusion, and greater planning efficiency. 

---
# RACE-Align: Retrieval-Augmented and Chain-of-Thought Enhanced Preference Alignment for Large Language Models 

**Authors**: Qihang Yan, Xinyu Zhang, Luming Guo, Qi Zhang, Feifan Liu  

**Link**: [PDF](https://arxiv.org/pdf/2506.02726)  

**Abstract**: Large Language Models (LLMs) struggle with accuracy, domain-specific reasoning, and interpretability in vertical domains. Traditional preference alignment methods like Reinforcement Learning from Human Feedback (RLHF) and Direct Preference Optimization (DPO) often overlook the underlying knowledge sources and reasoning logic. This paper introduces RACE-Align (Retrieval-Augmented and Chain-of-Thought Enhanced Alignment), a novel framework designed to address these limitations. RACE-Align systematically constructs a binary preference dataset incorporating external knowledge support and explicit Chain-of-Thought (CoT) reasoning, then aligns LLMs using the DPO algorithm. The core innovation lies in its preference data construction strategy: it integrates AI-driven retrieval for factual grounding, enhancing knowledgeability and accuracy, and emphasizes the optimization of domain-specific CoT, treating the reasoning process itself as a key preference dimension. A multi-stage, AI-driven refinement pipeline cost-effectively generates these preference pairs. Experimental validation in Traditional Chinese Medicine (TCM) using Qwen3-1.7B as the base model demonstrates that RACE-Align significantly outperforms the original base model and a model fine-tuned only with Supervised Fine-Tuning (SFT). Improvements were observed across multiple dimensions, including answer accuracy, information richness, application of TCM thinking patterns, logicality and depth of reasoning, and interpretability. These findings suggest RACE-Align offers an effective pathway to enhance LLMs' knowledge application, reasoning reliability, and process transparency in complex vertical domains. 

---
# Exploiting the English Vocabulary Profile for L2 word-level vocabulary assessment with LLMs 

**Authors**: Stefano Bannò, Kate Knill, Mark Gales  

**Link**: [PDF](https://arxiv.org/pdf/2506.02758)  

**Abstract**: Vocabulary use is a fundamental aspect of second language (L2) proficiency. To date, its assessment by automated systems has typically examined the context-independent, or part-of-speech (PoS) related use of words. This paper introduces a novel approach to enable fine-grained vocabulary evaluation exploiting the precise use of words within a sentence. The scheme combines large language models (LLMs) with the English Vocabulary Profile (EVP). The EVP is a standard lexical resource that enables in-context vocabulary use to be linked with proficiency level. We evaluate the ability of LLMs to assign proficiency levels to individual words as they appear in L2 learner writing, addressing key challenges such as polysemy, contextual variation, and multi-word expressions. We compare LLMs to a PoS-based baseline. LLMs appear to exploit additional semantic information that yields improved performance. We also explore correlations between word-level proficiency and essay-level proficiency. Finally, the approach is applied to examine the consistency of the EVP proficiency levels. Results show that LLMs are well-suited for the task of vocabulary assessment. 

---
# EvaLearn: Quantifying the Learning Capability and Efficiency of LLMs via Sequential Problem Solving 

**Authors**: Shihan Dou, Ming Zhang, Chenhao Huang, Jiayi Chen, Feng Chen, Shichun Liu, Yan Liu, Chenxiao Liu, Cheng Zhong, Zongzhang Zhang, Tao Gui, Chao Xin, Wei Chengzhi, Lin Yan, Qi Zhang, Xuanjing Huang  

**Link**: [PDF](https://arxiv.org/pdf/2506.02672)  

**Abstract**: We introduce EvaLearn, a pioneering benchmark designed to evaluate large language models (LLMs) on their learning capability and efficiency in challenging tasks, a critical, yet underexplored aspect of model potential. EvaLearn contains 648 challenging problems across six task types, grouped into 182 sequences, each sequence dedicated to one task type. Diverging from most existing benchmarks that evaluate models in parallel, EvaLearn requires models to solve problems sequentially, allowing them to leverage the experience gained from previous solutions. EvaLearn provides five comprehensive automated metrics to evaluate models and quantify their learning capability and efficiency. We extensively benchmark nine frontier models and observe varied performance profiles: some models, such as Claude-3.7-sonnet, start with moderate initial performance but exhibit strong learning ability, while some models struggle to benefit from experience and may even show negative transfer. Moreover, we investigate model performance under two learning settings and find that instance-level rubrics and teacher-model feedback further facilitate model learning. Importantly, we observe that current LLMs with stronger static abilities do not show a clear advantage in learning capability across all tasks, highlighting that EvaLearn evaluates a new dimension of model performance. We hope EvaLearn provides a novel evaluation perspective for assessing LLM potential and understanding the gap between models and human capabilities, promoting the development of deeper and more dynamic evaluation approaches. All datasets, the automatic evaluation framework, and the results studied in this paper are available at the GitHub repository. 

---
# Prompt-Unseen-Emotion: Zero-shot Expressive Speech Synthesis with Prompt-LLM Contextual Knowledge for Mixed Emotions 

**Authors**: Xiaoxue Gao, Huayun Zhang, Nancy F. Chen  

**Link**: [PDF](https://arxiv.org/pdf/2506.02742)  

**Abstract**: Existing expressive text-to-speech (TTS) systems primarily model a limited set of categorical emotions, whereas human conversations extend far beyond these predefined emotions, making it essential to explore more diverse emotional speech generation for more natural interactions. To bridge this gap, this paper proposes a novel prompt-unseen-emotion (PUE) approach to generate unseen emotional speech via emotion-guided prompt learning. PUE is trained utilizing an LLM-TTS architecture to ensure emotional consistency between categorical emotion-relevant prompts and emotional speech, allowing the model to quantitatively capture different emotion weightings per utterance. During inference, mixed emotional speech can be generated by flexibly adjusting emotion proportions and leveraging LLM contextual knowledge, enabling the model to quantify different emotional styles. Our proposed PUE successfully facilitates expressive speech synthesis of unseen emotions in a zero-shot setting. 

---
# Evaluating Named Entity Recognition Models for Russian Cultural News Texts: From BERT to LLM 

**Authors**: Maria Levchenko  

**Link**: [PDF](https://arxiv.org/pdf/2506.02589)  

**Abstract**: This paper addresses the challenge of Named Entity Recognition (NER) for person names within the specialized domain of Russian news texts concerning cultural events. The study utilizes the unique SPbLitGuide dataset, a collection of event announcements from Saint Petersburg spanning 1999 to 2019. A comparative evaluation of diverse NER models is presented, encompassing established transformer-based architectures such as DeepPavlov, RoBERTa, and SpaCy, alongside recent Large Language Models (LLMs) including GPT-3.5, GPT-4, and GPT-4o. Key findings highlight the superior performance of GPT-4o when provided with specific prompting for JSON output, achieving an F1 score of 0.93. Furthermore, GPT-4 demonstrated the highest precision at 0.99. The research contributes to a deeper understanding of current NER model capabilities and limitations when applied to morphologically rich languages like Russian within the cultural heritage domain, offering insights for researchers and practitioners. Follow-up evaluation with GPT-4.1 (April 2025) achieves F1=0.94 for both simple and structured prompts, demonstrating rapid progress across model families and simplified deployment requirements. 

---
# Pruning General Large Language Models into Customized Expert Models 

**Authors**: Yirao Zhao, Guizhen Chen, Kenji Kawaguchi, Lidong Bing, Wenxuan Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2506.02561)  

**Abstract**: Large language models (LLMs) have revolutionized natural language processing, yet their substantial model sizes often require substantial computational resources. To preserve computing resources and accelerate inference speed, it is crucial to prune redundant parameters, especially for experienced users who often need compact expert models tailored to specific downstream scenarios. However, most existing pruning methods focus on preserving the model's general capabilities, often requiring extensive post-training or suffering from degraded performance due to coarse-grained pruning. In this work, we design a $\underline{Cus}$tom $\underline{Prun}$ing method ($\texttt{Cus-Prun}$) to prune a large general model into a smaller lightweight expert model, which is positioned along the "language", "domain" and "task" dimensions. By identifying and pruning irrelevant neurons of each dimension, $\texttt{Cus-Prun}$ creates expert models without any post-training. Our experiments demonstrate that $\texttt{Cus-Prun}$ consistently outperforms other methods, achieving minimal loss in both expert and general capabilities across various models from different model families and sizes. 

---
# Technical Report for Ego4D Long-Term Action Anticipation Challenge 2025 

**Authors**: Qiaohui Chu, Haoyu Zhang, Yisen Feng, Meng Liu, Weili Guan, Yaowei Wang, Liqiang Nie  

**Link**: [PDF](https://arxiv.org/pdf/2506.02550)  

**Abstract**: In this report, we present a novel three-stage framework developed for the Ego4D Long-Term Action Anticipation (LTA) task. Inspired by recent advances in foundation models, our method consists of three stages: feature extraction, action recognition, and long-term action anticipation. First, visual features are extracted using a high-performance visual encoder. The features are then fed into a Transformer to predict verbs and nouns, with a verb-noun co-occurrence matrix incorporated to enhance recognition accuracy. Finally, the predicted verb-noun pairs are formatted as textual prompts and input into a fine-tuned large language model (LLM) to anticipate future action sequences. Our framework achieves first place in this challenge at CVPR 2025, establishing a new state-of-the-art in long-term action prediction. Our code will be released at this https URL. 

---
# Simplifying Root Cause Analysis in Kubernetes with StateGraph and LLM 

**Authors**: Yong Xiang, Charley Peter Chen, Liyi Zeng, Wei Yin, Xin Liu, Hu Li, Wei Xu  

**Link**: [PDF](https://arxiv.org/pdf/2506.02490)  

**Abstract**: Kubernetes, a notably complex and distributed system, utilizes an array of controllers to uphold cluster management logic through state reconciliation. Nevertheless, maintaining state consistency presents significant challenges due to unexpected failures, network disruptions, and asynchronous issues, especially within dynamic cloud environments. These challenges result in operational disruptions and economic losses, underscoring the necessity for robust root cause analysis (RCA) to enhance Kubernetes reliability. The development of large language models (LLMs) presents a promising direction for RCA. However, existing methodologies encounter several obstacles, including the diverse and evolving nature of Kubernetes incidents, the intricate context of incidents, and the polymorphic nature of these incidents. In this paper, we introduce SynergyRCA, an innovative tool that leverages LLMs with retrieval augmentation from graph databases and enhancement with expert prompts. SynergyRCA constructs a StateGraph to capture spatial and temporal relationships and utilizes a MetaGraph to outline entity connections. Upon the occurrence of an incident, an LLM predicts the most pertinent resource, and SynergyRCA queries the MetaGraph and StateGraph to deliver context-specific insights for RCA. We evaluate SynergyRCA using datasets from two production Kubernetes clusters, highlighting its capacity to identify numerous root causes, including novel ones, with high efficiency and precision. SynergyRCA demonstrates the ability to identify root causes in an average time of about two minutes and achieves an impressive precision of approximately 0.90. 

---
# M$^3$FinMeeting: A Multilingual, Multi-Sector, and Multi-Task Financial Meeting Understanding Evaluation Dataset 

**Authors**: Jie Zhu, Junhui Li, Yalong Wen, Xiandong Li, Lifan Guo, Feng Chen  

**Link**: [PDF](https://arxiv.org/pdf/2506.02510)  

**Abstract**: Recent breakthroughs in large language models (LLMs) have led to the development of new benchmarks for evaluating their performance in the financial domain. However, current financial benchmarks often rely on news articles, earnings reports, or announcements, making it challenging to capture the real-world dynamics of financial meetings. To address this gap, we propose a novel benchmark called $\texttt{M$^3$FinMeeting}$, which is a multilingual, multi-sector, and multi-task dataset designed for financial meeting understanding. First, $\texttt{M$^3$FinMeeting}$ supports English, Chinese, and Japanese, enhancing comprehension of financial discussions in diverse linguistic contexts. Second, it encompasses various industry sectors defined by the Global Industry Classification Standard (GICS), ensuring that the benchmark spans a broad range of financial activities. Finally, $\texttt{M$^3$FinMeeting}$ includes three tasks: summarization, question-answer (QA) pair extraction, and question answering, facilitating a more realistic and comprehensive evaluation of understanding. Experimental results with seven popular LLMs reveal that even the most advanced long-context models have significant room for improvement, demonstrating the effectiveness of $\texttt{M$^3$FinMeeting}$ as a benchmark for assessing LLMs' financial meeting comprehension skills. 

---
# Do Language Models Think Consistently? A Study of Value Preferences Across Varying Response Lengths 

**Authors**: Inderjeet Nair, Lu Wang  

**Link**: [PDF](https://arxiv.org/pdf/2506.02481)  

**Abstract**: Evaluations of LLMs' ethical risks and value inclinations often rely on short-form surveys and psychometric tests, yet real-world use involves long-form, open-ended responses -- leaving value-related risks and preferences in practical settings largely underexplored. In this work, we ask: Do value preferences inferred from short-form tests align with those expressed in long-form outputs? To address this question, we compare value preferences elicited from short-form reactions and long-form responses, varying the number of arguments in the latter to capture users' differing verbosity preferences. Analyzing five LLMs (llama3-8b, gemma2-9b, mistral-7b, qwen2-7b, and olmo-7b), we find (1) a weak correlation between value preferences inferred from short-form and long-form responses across varying argument counts, and (2) similarly weak correlation between preferences derived from any two distinct long-form generation settings. (3) Alignment yields only modest gains in the consistency of value expression. Further, we examine how long-form generation attributes relate to value preferences, finding that argument specificity negatively correlates with preference strength, while representation across scenarios shows a positive correlation. Our findings underscore the need for more robust methods to ensure consistent value expression across diverse applications. 

---
# Multimodal DeepResearcher: Generating Text-Chart Interleaved Reports From Scratch with Agentic Framework 

**Authors**: Zhaorui Yang, Bo Pan, Han Wang, Yiyao Wang, Xingyu Liu, Minfeng Zhu, Bo Zhang, Wei Chen  

**Link**: [PDF](https://arxiv.org/pdf/2506.02454)  

**Abstract**: Visualizations play a crucial part in effective communication of concepts and information. Recent advances in reasoning and retrieval augmented generation have enabled Large Language Models (LLMs) to perform deep research and generate comprehensive reports. Despite its progress, existing deep research frameworks primarily focus on generating text-only content, leaving the automated generation of interleaved texts and visualizations underexplored. This novel task poses key challenges in designing informative visualizations and effectively integrating them with text reports. To address these challenges, we propose Formal Description of Visualization (FDV), a structured textual representation of charts that enables LLMs to learn from and generate diverse, high-quality visualizations. Building on this representation, we introduce Multimodal DeepResearcher, an agentic framework that decomposes the task into four stages: (1) researching, (2) exemplar report textualization, (3) planning, and (4) multimodal report generation. For the evaluation of generated multimodal reports, we develop MultimodalReportBench, which contains 100 diverse topics served as inputs along with 5 dedicated metrics. Extensive experiments across models and evaluation methods demonstrate the effectiveness of Multimodal DeepResearcher. Notably, utilizing the same Claude 3.7 Sonnet model, Multimodal DeepResearcher achieves an 82\% overall win rate over the baseline method. 

---
# EssayBench: Evaluating Large Language Models in Multi-Genre Chinese Essay Writing 

**Authors**: Fan Gao, Dongyuan Li, Ding Xia, Fei Mi, Yasheng Wang, Lifeng Shang, Baojun Wang  

**Link**: [PDF](https://arxiv.org/pdf/2506.02596)  

**Abstract**: Chinese essay writing and its evaluation are critical in educational contexts, yet the capabilities of Large Language Models (LLMs) in this domain remain largely underexplored. Existing benchmarks often rely on coarse-grained text quality metrics, largely overlooking the structural and rhetorical complexities of Chinese essays, particularly across diverse genres. To address this gap, we propose \benchName, a multi-genre benchmark specifically designed for Chinese essay writing across four major genres: Argumentative, Narrative, Descriptive, and Expository. We curate and refine a total of 728 real-world prompts to ensure authenticity and meticulously categorize them into the \textit{Open-Ended} and \textit{Constrained} sets to capture diverse writing scenarios. To reliably evaluate generated essays, we develop a fine-grained, genre-specific scoring framework that hierarchically aggregates scores. We further validate our evaluation protocol through a comprehensive human agreement study. Finally, we benchmark 15 large-sized LLMs, analyzing their strengths and limitations across genres and instruction types. With \benchName, we aim to advance LLM-based Chinese essay evaluation and inspire future research on improving essay generation in educational settings. 

---
# Comparative Analysis of AI Agent Architectures for Entity Relationship Classification 

**Authors**: Maryam Berijanian, Kuldeep Singh, Amin Sehati  

**Link**: [PDF](https://arxiv.org/pdf/2506.02426)  

**Abstract**: Entity relationship classification remains a challenging task in information extraction, especially in scenarios with limited labeled data and complex relational structures. In this study, we conduct a comparative analysis of three distinct AI agent architectures designed to perform relation classification using large language models (LLMs). The agentic architectures explored include (1) reflective self-evaluation, (2) hierarchical task decomposition, and (3) a novel multi-agent dynamic example generation mechanism, each leveraging different modes of reasoning and prompt adaptation. In particular, our dynamic example generation approach introduces real-time cooperative and adversarial prompting. We systematically compare their performance across multiple domains and model backends. Our experiments demonstrate that multi-agent coordination consistently outperforms standard few-shot prompting and approaches the performance of fine-tuned models. These findings offer practical guidance for the design of modular, generalizable LLM-based systems for structured relation extraction. The source codes and dataset are available at \href{this https URL}{this https URL}. 

---
# AERO: A Redirection-Based Optimization Framework Inspired by Judo for Robust Probabilistic Forecasting 

**Authors**: Karthikeyan Vaiapury  

**Link**: [PDF](https://arxiv.org/pdf/2506.02415)  

**Abstract**: Optimization remains a fundamental pillar of machine learning, yet existing methods often struggle to maintain stability and adaptability in dynamic, non linear systems, especially under uncertainty. We introduce AERO (Adversarial Energy-based Redirection Optimization), a novel framework inspired by the redirection principle in Judo, where external disturbances are leveraged rather than resisted. AERO reimagines optimization as a redirection process guided by 15 interrelated axioms encompassing adversarial correction, energy conservation, and disturbance-aware learning. By projecting gradients, integrating uncertainty driven dynamics, and managing learning energy, AERO offers a principled approach to stable and robust model updates. Applied to probabilistic solar energy forecasting, AERO demonstrates substantial gains in predictive accuracy, reliability, and adaptability, especially in noisy and uncertain environments. Our findings highlight AERO as a compelling new direction in the theoretical and practical landscape of optimization. 

---
# Univariate to Multivariate: LLMs as Zero-Shot Predictors for Time-Series Forecasting 

**Authors**: Chamara Madarasingha, Nasrin Sohrabi, Zahir Tari  

**Link**: [PDF](https://arxiv.org/pdf/2506.02389)  

**Abstract**: Time-series prediction or forecasting is critical across many real-world dynamic systems, and recent studies have proposed using Large Language Models (LLMs) for this task due to their strong generalization capabilities and ability to perform well without extensive pre-training. However, their effectiveness in handling complex, noisy, and multivariate time-series data remains underexplored. To address this, we propose LLMPred which enhances LLM-based time-series prediction by converting time-series sequences into text and feeding them to LLMs for zero shot prediction along with two main data pre-processing techniques. First, we apply time-series sequence decomposition to facilitate accurate prediction on complex and noisy univariate sequences. Second, we extend this univariate prediction capability to multivariate data using a lightweight prompt-processing strategy. Extensive experiments with smaller LLMs such as Llama 2 7B, Llama 3.2 3B, GPT-4o-mini, and DeepSeek 7B demonstrate that LLMPred achieves competitive or superior performance compared to state-of-the-art baselines. Additionally, a thorough ablation study highlights the importance of the key components proposed in LLMPred. 

---
# Consultant Decoding: Yet Another Synergistic Mechanism 

**Authors**: Chuanghao Ding, Jiaping Wang, Ziqing Yang, Xiaoliang Wang, Dahua Lin, Cam-Tu Nguyen, Fei Tan  

**Link**: [PDF](https://arxiv.org/pdf/2506.02391)  

**Abstract**: The synergistic mechanism based on Speculative Decoding (SD) has garnered considerable attention as a simple yet effective approach for accelerating the inference of large language models (LLMs). Nonetheless, the high rejection rates require repeated LLMs calls to validate draft tokens, undermining the overall efficiency gain of SD. In this work, we revisit existing verification mechanisms and propose a novel synergetic mechanism Consultant Decoding (CD). Unlike SD, which relies on a metric derived from importance sampling for verification, CD verifies candidate drafts using token-level likelihoods computed solely by the LLM. CD achieves up to a 2.5-fold increase in inference speed compared to the target model, while maintaining comparable generation quality (around 100% of the target model's performance). Interestingly, this is achieved by combining models whose parameter sizes differ by two orders of magnitude. In addition, CD reduces the call frequency of the large target model to below 10%, particularly in more demanding tasks. CD's performance was even found to surpass that of the large target model, which theoretically represents the upper bound for speculative decoding. 

---
# Exploring Explanations Improves the Robustness of In-Context Learning 

**Authors**: Ukyo Honda, Tatsushi Oka  

**Link**: [PDF](https://arxiv.org/pdf/2506.02378)  

**Abstract**: In-context learning (ICL) has emerged as a successful paradigm for leveraging large language models (LLMs). However, it often struggles to generalize beyond the distribution of the provided demonstrations. A recent advancement in enhancing robustness is ICL with explanations (X-ICL), which improves prediction reliability by guiding LLMs to understand and articulate the reasoning behind correct labels. Building on this approach, we introduce an advanced framework that extends X-ICL by systematically exploring explanations for all possible labels (X$^2$-ICL), thereby enabling more comprehensive and robust decision-making. Experimental results on multiple natural language understanding datasets validate the effectiveness of X$^2$-ICL, demonstrating significantly improved robustness to out-of-distribution data compared to the existing ICL approaches. 

---
# Evaluating LLM Agent Adherence to Hierarchical Safety Principles: A Lightweight Benchmark for Probing Foundational Controllability Components 

**Authors**: Ram Potham  

**Link**: [PDF](https://arxiv.org/pdf/2506.02357)  

**Abstract**: Credible safety plans for advanced AI development require methods to verify agent behavior and detect potential control deficiencies early. A fundamental aspect is ensuring agents adhere to safety-critical principles, especially when these conflict with operational goals. Failure to prioritize such principles indicates a potential basic control failure. This paper introduces a lightweight, interpretable benchmark methodology using a simple grid world to evaluate an LLM agent's ability to uphold a predefined, high-level safety principle (e.g., "never enter hazardous zones") when faced with conflicting lower-level task instructions. We probe whether the agent reliably prioritizes the inviolable directive, testing a foundational controllability aspect of LLMs. This pilot study demonstrates the methodology's feasibility, offers preliminary insights into agent behavior under principle conflict, and discusses how such benchmarks can contribute empirical evidence for assessing controllability. We argue that evaluating adherence to hierarchical principles is a crucial early step in understanding our capacity to build governable AI systems. 

---
# Something Just Like TRuST : Toxicity Recognition of Span and Target 

**Authors**: Berk Atil, Namrata Sureddy, Rebecca J. Passonneau  

**Link**: [PDF](https://arxiv.org/pdf/2506.02326)  

**Abstract**: Toxicity in online content, including content generated by language models, has become a critical concern due to its potential for negative psychological and social impact. This paper introduces TRuST, a comprehensive dataset designed to improve toxicity detection that merges existing datasets, and has labels for toxicity, target social group, and toxic spans. It includes a diverse range of target groups such as ethnicity, gender, religion, disability, and politics, with both human/machine-annotated and human machine-generated data. We benchmark state-of-the-art large language models (LLMs) on toxicity detection, target group identification, and toxic span extraction. We find that fine-tuned models consistently outperform zero-shot and few-shot prompting, though performance remains low for certain social groups. Further, reasoning capabilities do not significantly improve performance, indicating that LLMs have weak social reasoning skills. 

---
# GraphRAG-Bench: Challenging Domain-Specific Reasoning for Evaluating Graph Retrieval-Augmented Generation 

**Authors**: Yilin Xiao, Junnan Dong, Chuang Zhou, Su Dong, Qianwen Zhang, Di Yin, Xing Sun, Xiao Huang  

**Link**: [PDF](https://arxiv.org/pdf/2506.02404)  

**Abstract**: Graph Retrieval Augmented Generation (GraphRAG) has garnered increasing recognition for its potential to enhance large language models (LLMs) by structurally organizing domain-specific corpora and facilitating complex reasoning. However, current evaluations of GraphRAG models predominantly rely on traditional question-answering datasets. Their limited scope in questions and evaluation metrics fails to comprehensively assess the reasoning capacity improvements enabled by GraphRAG models. To address this gap, we introduce GraphRAG-Bench, a large-scale, domain-specific benchmark designed to rigorously evaluate GraphRAG models. Our benchmark offers three key superiorities: \((i)\) Challenging question design. Featuring college-level, domain-specific questions that demand multi-hop reasoning, the benchmark ensures that simple content retrieval is insufficient for problem-solving. For example, some questions require mathematical reasoning or programming. \((ii)\) Diverse task coverage. The dataset includes a broad spectrum of reasoning tasks, multiple-choice, true/false, multi-select, open-ended, and fill-in-the-blank. It spans 16 disciplines in twenty core textbooks. \((iii)\) Holistic evaluation framework. GraphRAG-Bench provides comprehensive assessment across the entire GraphRAG pipeline, including graph construction, knowledge retrieval, and answer generation. Beyond final-answer correctness, it evaluates the logical coherence of the reasoning process. By applying nine contemporary GraphRAG methods to GraphRAG-Bench, we demonstrate its utility in quantifying how graph-based structuring improves model reasoning capabilities. Our analysis reveals critical insights about graph architectures, retrieval efficacy, and reasoning capabilities, offering actionable guidance for the research community. 

---
# DIAMOND: An LLM-Driven Agent for Context-Aware Baseball Highlight Summarization 

**Authors**: Jeonghun Kang, Soonmok Kwon, Joonseok Lee, Byung-Hak Kim  

**Link**: [PDF](https://arxiv.org/pdf/2506.02351)  

**Abstract**: Traditional approaches -- such as Win Probability Added (WPA)-based ranking or computer vision-driven event detection -- can identify scoring plays but often miss strategic depth, momentum shifts, and storyline progression. Manual curation remains the gold standard but is resource-intensive and not scalable. We introduce DIAMOND, an LLM-driven agent for context-aware baseball highlight summarization that integrates structured sports analytics with natural language reasoning. DIAMOND leverages sabermetric features -- Win Expectancy, WPA, and Leverage Index -- to quantify play importance, while an LLM module enhances selection based on contextual narrative value. This hybrid approach ensures both quantitative rigor and qualitative richness, surpassing the limitations of purely statistical or vision-based systems. Evaluated on five diverse Korean Baseball Organization League games, DIAMOND improves F1-score from 42.9% (WPA-only) to 84.8%, outperforming both commercial and statistical baselines. Though limited in scale, our results highlight the potential of modular, interpretable agent-based frameworks for event-level summarization in sports and beyond. 

---
# KDRL: Post-Training Reasoning LLMs via Unified Knowledge Distillation and Reinforcement Learning 

**Authors**: Hongling Xu, Qi Zhu, Heyuan Deng, Jinpeng Li, Lu Hou, Yasheng Wang, Lifeng Shang, Ruifeng Xu, Fei Mi  

**Link**: [PDF](https://arxiv.org/pdf/2506.02208)  

**Abstract**: Recent advances in large language model (LLM) post-training have leveraged two distinct paradigms to enhance reasoning capabilities: reinforcement learning (RL) and knowledge distillation (KD). While RL enables the emergence of complex reasoning behaviors, it often suffers from low sample efficiency when the initial policy struggles to explore high-reward trajectories. Conversely, KD improves learning efficiency via mimicking the teacher model but tends to generalize poorly to out-of-domain scenarios. In this work, we present \textbf{KDRL}, a \textit{unified post-training framework} that jointly optimizes a reasoning model through teacher supervision (KD) and self-exploration (RL). Specifically, KDRL leverages policy gradient optimization to simultaneously minimize the reverse Kullback-Leibler divergence (RKL) between the student and teacher distributions while maximizing the expected rule-based rewards. We first formulate a unified objective that integrates GRPO and KD, and systematically explore how different KL approximations, KL coefficients, and reward-guided KD strategies affect the overall post-training dynamics and performance. Empirical results on multiple reasoning benchmarks demonstrate that KDRL outperforms GRPO and various KD baselines while achieving a favorable balance between performance and reasoning token efficiency. These findings indicate that integrating KD and RL serves as an effective and efficient strategy to train reasoning LLMs. 

---
# Why Gradients Rapidly Increase Near the End of Training 

**Authors**: Aaron Defazio  

**Link**: [PDF](https://arxiv.org/pdf/2506.02285)  

**Abstract**: During long-duration Large Language Model (LLM) training runs the gradient norm increases rapidly near the end of training. In this short note, we show that this increase is due to an unintended interaction between weight decay, normalization layers, and the learning rate schedule. We propose a simple correction that fixes this behavior while also resulting in lower loss values throughout training. 

---
# QARI-OCR: High-Fidelity Arabic Text Recognition through Multimodal Large Language Model Adaptation 

**Authors**: Ahmed Wasfy, Omer Nacar, Abdelakreem Elkhateb, Mahmoud Reda, Omar Elshehy, Adel Ammar, Wadii Boulila  

**Link**: [PDF](https://arxiv.org/pdf/2506.02295)  

**Abstract**: The inherent complexities of Arabic script; its cursive nature, diacritical marks (tashkeel), and varied typography, pose persistent challenges for Optical Character Recognition (OCR). We present Qari-OCR, a series of vision-language models derived from Qwen2-VL-2B-Instruct, progressively optimized for Arabic through iterative fine-tuning on specialized synthetic datasets. Our leading model, QARI v0.2, establishes a new open-source state-of-the-art with a Word Error Rate (WER) of 0.160, Character Error Rate (CER) of 0.061, and BLEU score of 0.737 on diacritically-rich texts. Qari-OCR demonstrates superior handling of tashkeel, diverse fonts, and document layouts, alongside impressive performance on low-resolution images. Further explorations (QARI v0.3) showcase strong potential for structural document understanding and handwritten text. This work delivers a marked improvement in Arabic OCR accuracy and efficiency, with all models and datasets released to foster further research. 

---
# Assigning Distinct Roles to Quantized and Low-Rank Matrices Toward Optimal Weight Decomposition 

**Authors**: Yoonjun Cho, Soeun Kim, Dongjae Jeon, Kyelim Lee, Beomsoo Lee, Albert No  

**Link**: [PDF](https://arxiv.org/pdf/2506.02077)  

**Abstract**: Decomposing weight matrices into quantization and low-rank components ($\mathbf{W} \approx \mathbf{Q} + \mathbf{L}\mathbf{R}$) is a widely used technique for compressing large language models (LLMs). Existing joint optimization methods iteratively alternate between quantization and low-rank approximation. However, these methods tend to prioritize one component at the expense of the other, resulting in suboptimal decompositions that fail to leverage each component's unique strengths. In this work, we introduce Outlier-Driven Low-Rank Initialization (ODLRI), which assigns low-rank components the specific role of capturing activation-sensitive weights. This structured decomposition mitigates outliers' negative impact on quantization, enabling more effective balance between quantization and low-rank approximation. Experiments on Llama2 (7B, 13B, 70B), Llama3-8B, and Mistral-7B demonstrate that incorporating ODLRI into the joint optimization framework consistently reduces activation-aware error, minimizes quantization scale, and improves perplexity and zero-shot accuracy in low-bit settings. 

---
# Flow2Code: Evaluating Large Language Models for Flowchart-based Code Generation Capability 

**Authors**: Mengliang He, Jiayi Zeng, Yankai Jiang, Wei Zhang, Zeming Liu, Xiaoming Shi, Aimin Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2506.02073)  

**Abstract**: While large language models (LLMs) show promise in code generation, existing benchmarks neglect the flowchart-based code generation. To promote further research on flowchart-based code generation, this work presents Flow2Code, a novel benchmark for flowchart-based code generation evaluation. The evaluation dataset spans 15 programming languages and includes 5,622 code segments paired with 16,866 flowcharts of three types: code, UML, and pseudocode. Extensive experiments with 13 multimodal LLMs reveal that current LLMs can not generate code based on flowcharts perfectly. Besides, experiment results show that the supervised fine-tuning technique contributes greatly to the models' performance. We publicly release our code and datasets at this https URL. 

---
# Improving LLM Agents with Reinforcement Learning on Cryptographic CTF Challenges 

**Authors**: Lajos Muzsai, David Imolai, András Lukács  

**Link**: [PDF](https://arxiv.org/pdf/2506.02048)  

**Abstract**: Large Language Models (LLMs) still struggle with the structured reasoning and tool-assisted computation needed for problem solving in cybersecurity applications. In this work, we introduce "random-crypto", a cryptographic Capture-the-Flag (CTF) challenge generator framework that we use to fine-tune a tool-augmented Llama-3.1-8B with Guided Reinforcement Prompt Optimisation (GRPO), allowing the agent to iteratively write and execute Python inside an isolated REPL. GRPO yields a +53% absolute jump in Pass@8 on unseen "random-crypto" tasks (0.35 -> 0.88) and raises Majority@8 to 0.41. The fine-tuned agent also generalizes to an external dataset. On a subset of picoCTF cryptography problems, it improves Pass@8 by +13 pp. Ablations show the gains stem from more reliable tool invocation and code synthesis, rather than superficial prompt adaptation. 

---
# Evaluating the Efficacy of LLM-Based Reasoning for Multiobjective HPC Job Scheduling 

**Authors**: Prachi Jadhav, Hongwei Jin, Ewa Deelman, Prasanna Balaprakash  

**Link**: [PDF](https://arxiv.org/pdf/2506.02025)  

**Abstract**: High-Performance Computing (HPC) job scheduling involves balancing conflicting objectives such as minimizing makespan, reducing wait times, optimizing resource use, and ensuring fairness. Traditional methods, including heuristic-based (e.g., First-Come-First-Served) or intensive optimization techniques, often lack adaptability to dynamic workloads and heterogeneous HPC systems. To address this, we propose a novel Large Language Model (LLM)-based scheduler using a ReAct-style framework (Reason + Act), enabling iterative, interpretable decision-making. The system incorporates a scratchpad memory to track scheduling history and refine decisions via natural language feedback, while a constraint enforcement module ensures feasibility and safety. We evaluate our approach using OpenAI's O4-Mini and Anthropic's Claude 3.7 across seven real-world HPC workload scenarios, including heterogeneous mixes, bursty patterns, and adversarial cases. Comparisons against FCFS, Shortest Job First, and Google OR-Tools (on 10 to 100 jobs) reveal that LLM-based scheduling effectively balances multiple objectives while offering transparent reasoning through natural language traces. The method excels in constraint satisfaction and adapts to diverse workloads without domain-specific training. However, a trade-off between reasoning quality and computational overhead challenges real-time deployment. This work presents the first comprehensive study of reasoning-capable LLMs for HPC scheduling, demonstrating their potential to handle multiobjective optimization while highlighting limitations in computational efficiency. The findings provide insights into leveraging advanced language models for complex scheduling problems in dynamic HPC environments. 

---
# No Free Lunch in Active Learning: LLM Embedding Quality Dictates Query Strategy Success 

**Authors**: Lukas Rauch, Moritz Wirth, Denis Huseljic, Marek Herde, Bernhard Sick, Matthias Aßenmacher  

**Link**: [PDF](https://arxiv.org/pdf/2506.01992)  

**Abstract**: The advent of large language models (LLMs) capable of producing general-purpose representations lets us revisit the practicality of deep active learning (AL): By leveraging frozen LLM embeddings, we can mitigate the computational costs of iteratively fine-tuning large backbones. This study establishes a benchmark and systematically investigates the influence of LLM embedding quality on query strategies in deep AL. We employ five top-performing models from the massive text embedding benchmark (MTEB) leaderboard and two baselines for ten diverse text classification tasks. Our findings reveal key insights: First, initializing the labeled pool using diversity-based sampling synergizes with high-quality embeddings, boosting performance in early AL iterations. Second, the choice of the optimal query strategy is sensitive to embedding quality. While the computationally inexpensive Margin sampling can achieve performance spikes on specific datasets, we find that strategies like Badge exhibit greater robustness across tasks. Importantly, their effectiveness is often enhanced when paired with higher-quality embeddings. Our results emphasize the need for context-specific evaluation of AL strategies, as performance heavily depends on embedding quality and the target task. 

---
# Turning LLM Activations Quantization-Friendly 

**Authors**: Patrik Czakó, Gábor Kertész, Sándor Szénási  

**Link**: [PDF](https://arxiv.org/pdf/2506.01967)  

**Abstract**: Quantization effectively reduces the serving costs of Large Language Models (LLMs) by speeding up data movement through compressed parameters and enabling faster operations via integer arithmetic. However, activating integer arithmetic requires quantizing both weights and activations, which poses challenges due to the significant outliers in LLMs that increase quantization error. In this work, we investigate these outliers with an emphasis on their effect on layer-wise quantization error, then examine how smoothing and rotation transform the observed values. Our primary contributions include introducing a new metric to measure and visualize quantization difficulty based on channel magnitudes, as well as proposing a hybrid approach that applies channel-wise scaling before rotation, supported by a mathematical formulation of its benefits. 

---
# SpecMemo: Speculative Decoding is in Your Pocket 

**Authors**: Selin Yildirim, Deming Chen  

**Link**: [PDF](https://arxiv.org/pdf/2506.01986)  

**Abstract**: Recent advancements in speculative decoding have demonstrated considerable speedup across a wide array of large language model (LLM) tasks. Speculative decoding inherently relies on sacrificing extra memory allocations to generate several candidate tokens, of which acceptance rate drives the speedup. However, deploying speculative decoding on memory-constrained devices, such as mobile GPUs, remains as a significant challenge in real-world scenarios. In this work, we present a device-aware inference engine named SpecMemo that can smartly control memory allocations at finer levels to enable multi-turn chatbots with speculative decoding on such limited memory devices. Our methodology stems from theoretically modeling memory footprint of speculative decoding to determine a lower bound on the required memory budget while retaining speedup. SpecMemo empirically acquires a careful balance between minimizing redundant memory allocations for rejected candidate tokens and maintaining competitive performance gains from speculation. Notably, with SpecMemo's memory management, we maintain 96% of overall throughput from speculative decoding on MT-Bench, with reduced generation-memory by 65% on single Nvidia Titan RTX. Given multiple constrained GPUs, we build on top of previous speculative decoding architectures to facilitate big-model inference by distributing Llama-2-70B-Chat model, on which we provide novel batched speculative decoding to increase usability of multiple small server GPUs. This novel framework demonstrates 2x speedup over distributed and batched vanilla decoding with the base model on eight AMD MI250 GPUs. Moreover, inference throughput increases remarkably 8x with batch size 10. Our work contributes to democratized LLM applications in resource-constrained environments, providing a pathway for faster and cheaper deployment of real-world LLM applications with robust performance. 

---
# Co-Evolving LLM Coder and Unit Tester via Reinforcement Learning 

**Authors**: Yinjie Wang, Ling Yang, Ye Tian, Ke Shen, Mengdi Wang  

**Link**: [PDF](https://arxiv.org/pdf/2506.03136)  

**Abstract**: We propose CURE, a novel reinforcement learning framework with a dedicated reward design that co-evolves coding and unit test generation capabilities based on their interaction outcomes, without any ground-truth code as supervision. This approach enables flexible and scalable training and allows the unit tester to learn directly from the coder's mistakes. Our derived ReasonFlux-Coder-7B and 14B models improve code generation accuracy by 5.3% and Best-of-N accuracy by 9.0% after optimization on Qwen2.5-Instruct models, outperforming similarly sized Qwen-Coder, DeepSeek-Coder, and Seed-Coder. They naturally extend to downstream tasks such as test-time scaling and agentic coding-achieving a 8.1% improvement over the base model. For the long-CoT model, our ReasonFlux-Coder-4B consistently outperforms Qwen3-4B while achieving 64.8% inference efficiency in unit test generation. Notably, we also find that our model can serve as an effective reward model for reinforcement learning on base models. Project: this https URL 

---
# AUTOCIRCUIT-RL: Reinforcement Learning-Driven LLM for Automated Circuit Topology Generation 

**Authors**: Prashanth Vijayaraghavan, Luyao Shi, Ehsan Degan, Vandana Mukherjee, Xin Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2506.03122)  

**Abstract**: Analog circuit topology synthesis is integral to Electronic Design Automation (EDA), enabling the automated creation of circuit structures tailored to specific design requirements. However, the vast design search space and strict constraint adherence make efficient synthesis challenging. Leveraging the versatility of Large Language Models (LLMs), we propose AUTOCIRCUIT-RL,a novel reinforcement learning (RL)-based framework for automated analog circuit synthesis. The framework operates in two phases: instruction tuning, where an LLM learns to generate circuit topologies from structured prompts encoding design constraints, and RL refinement, which further improves the instruction-tuned model using reward models that evaluate validity, efficiency, and output voltage. The refined model is then used directly to generate topologies that satisfy the design constraints. Empirical results show that AUTOCIRCUIT-RL generates ~12% more valid circuits and improves efficiency by ~14% compared to the best baselines, while reducing duplicate generation rates by ~38%. It achieves over 60% success in synthesizing valid circuits with limited training data, demonstrating strong generalization. These findings highlight the framework's effectiveness in scaling to complex circuits while maintaining efficiency and constraint adherence, marking a significant advancement in AI-driven circuit design. 

---
# Literary Evidence Retrieval via Long-Context Language Models 

**Authors**: Katherine Thai, Mohit Iyyer  

**Link**: [PDF](https://arxiv.org/pdf/2506.03090)  

**Abstract**: How well do modern long-context language models understand literary fiction? We explore this question via the task of literary evidence retrieval, repurposing the RELiC dataset of That et al. (2022) to construct a benchmark where the entire text of a primary source (e.g., The Great Gatsby) is provided to an LLM alongside literary criticism with a missing quotation from that work. This setting, in which the model must generate the missing quotation, mirrors the human process of literary analysis by requiring models to perform both global narrative reasoning and close textual examination. We curate a high-quality subset of 292 examples through extensive filtering and human verification. Our experiments show that recent reasoning models, such as Gemini Pro 2.5 can exceed human expert performance (62.5% vs. 50% accuracy). In contrast, the best open-weight model achieves only 29.1% accuracy, highlighting a wide gap in interpretive reasoning between open and closed-weight models. Despite their speed and apparent accuracy, even the strongest models struggle with nuanced literary signals and overgeneration, signaling open challenges for applying LLMs to literary analysis. We release our dataset and evaluation code to encourage future work in this direction. 

---
# It's Not a Walk in the Park! Challenges of Idiom Translation in Speech-to-text Systems 

**Authors**: Iuliia Zaitova, Badr M. Abdullah, Wei Xue, Dietrich Klakow, Bernd Möbius, Tania Avgustinova  

**Link**: [PDF](https://arxiv.org/pdf/2506.02995)  

**Abstract**: Idioms are defined as a group of words with a figurative meaning not deducible from their individual components. Although modern machine translation systems have made remarkable progress, translating idioms remains a major challenge, especially for speech-to-text systems, where research on this topic is notably sparse. In this paper, we systematically evaluate idiom translation as compared to conventional news translation in both text-to-text machine translation (MT) and speech-to-text translation (SLT) systems across two language pairs (German to English, Russian to English). We compare state-of-the-art end-to-end SLT systems (SeamlessM4T SLT-to-text, Whisper Large v3) with MT systems (SeamlessM4T SLT-to-text, No Language Left Behind), Large Language Models (DeepSeek, LLaMA) and cascaded alternatives. Our results reveal that SLT systems experience a pronounced performance drop on idiomatic data, often reverting to literal translations even in higher layers, whereas MT systems and Large Language Models demonstrate better handling of idioms. These findings underscore the need for idiom-specific strategies and improved internal representations in SLT architectures. 

---
# Expanding before Inferring: Enhancing Factuality in Large Language Models through Premature Layers Interpolation 

**Authors**: Dingwei Chen, Ziqiang Liu, Feiteng Fang, Chak Tou Leong, Shiwen Ni, Ahmadreza Argha, Hamid Alinejad-Rokny, Min Yang, Chengming Li  

**Link**: [PDF](https://arxiv.org/pdf/2506.02973)  

**Abstract**: Large Language Models (LLMs) demonstrate remarkable capabilities in text understanding and generation. However, their tendency to produce factually inconsistent outputs, commonly referred to as ''hallucinations'', remains a critical challenge. Existing approaches, such as retrieval-based and inference-time correction methods, primarily address this issue at the input or output level, often overlooking the intrinsic information refinement process and the role of premature layers. Meanwhile, alignment- and fine-tuning-based methods are resource-intensive. In this paper, we propose PLI (Premature Layers Interpolation), a novel, training-free, and plug-and-play intervention designed to enhance factuality. PLI mitigates hallucinations by inserting premature layers formed through mathematical interpolation with adjacent layers. Inspired by stable diffusion and sampling steps, PLI extends the depth of information processing and transmission in LLMs, improving factual coherence. Experiments on four publicly available datasets demonstrate that PLI effectively reduces hallucinations while outperforming existing baselines in most cases. Further analysis suggests that the success of layer interpolation is closely linked to LLMs' internal mechanisms. To promote reproducibility, we will release our code and data upon acceptance. 

---
# Quantitative LLM Judges 

**Authors**: Aishwarya Sahoo, Jeevana Kruthi Karnuthala, Tushar Parmanand Budhwani, Pranchal Agarwal, Sankaran Vaidyanathan, Alexa Siu, Franck Dernoncourt, Jennifer Healey, Nedim Lipka, Ryan Rossi, Uttaran Bhattacharya, Branislav Kveton  

**Link**: [PDF](https://arxiv.org/pdf/2506.02945)  

**Abstract**: LLM-as-a-judge is a framework in which a large language model (LLM) automatically evaluates the output of another LLM. We propose quantitative LLM judges, which align evaluation scores of existing LLM judges to human scores in a given domain using regression models. The models are trained to improve the score of the original judge by using the judge's textual evaluation and score. We present four quantitative judges for different types of absolute and relative feedback, which showcases the generality and versatility of our framework. Our framework is more computationally efficient than supervised fine-tuning and can be more statistically efficient when human feedback is limited, which is expected in most applications of our work. We validate these claims empirically on four datasets using two base judges. Our experiments show that quantitative judges can effectively improve the predictive power of existing judges through post-hoc modeling. 

---
# INESC-ID @ eRisk 2025: Exploring Fine-Tuned, Similarity-Based, and Prompt-Based Approaches to Depression Symptom Identification 

**Authors**: Diogo A.P. Nunes, Eugénio Ribeiro  

**Link**: [PDF](https://arxiv.org/pdf/2506.02924)  

**Abstract**: In this work, we describe our team's approach to eRisk's 2025 Task 1: Search for Symptoms of Depression. Given a set of sentences and the Beck's Depression Inventory - II (BDI) questionnaire, participants were tasked with submitting up to 1,000 sentences per depression symptom in the BDI, sorted by relevance. Participant submissions were evaluated according to standard Information Retrieval (IR) metrics, including Average Precision (AP) and R-Precision (R-PREC). The provided training data, however, consisted of sentences labeled as to whether a given sentence was relevant or not w.r.t. one of BDI's symptoms. Due to this labeling limitation, we framed our development as a binary classification task for each BDI symptom, and evaluated accordingly. To that end, we split the available labeled data into training and validation sets, and explored foundation model fine-tuning, sentence similarity, Large Language Model (LLM) prompting, and ensemble techniques. The validation results revealed that fine-tuning foundation models yielded the best performance, particularly when enhanced with synthetic data to mitigate class imbalance. We also observed that the optimal approach varied by symptom. Based on these insights, we devised five independent test runs, two of which used ensemble methods. These runs achieved the highest scores in the official IR evaluation, outperforming submissions from 16 other teams. 

---
# A Controllable Examination for Long-Context Language Models 

**Authors**: Yijun Yang, Zeyu Huang, Wenhao Zhu, Zihan Qiu, Fei Yuan, Jeff Z.Pan, Ivan Titov  

**Link**: [PDF](https://arxiv.org/pdf/2506.02921)  

**Abstract**: Existing frameworks for evaluating long-context language models (LCLM) can be broadly categorized into real-world and synthetic tasks. Despite their utility, both approaches are accompanied by certain intrinsic limitations. Real-world tasks are too complex to interpret or characterize and are susceptible to data contamination. In contrast, synthetic tasks often adopt the needle-in-the-haystack (NIAH) format, wherein a lack of coherence between the "needle" and the "haystack" compromises their validity as proxies for realistic applications. In response to these challenges, we posit that an ideal long-context evaluation framework should be characterized by three essential features: $\textit{seamless context}$, $\textit{controllable setting}$, and $\textit{sound evaluation}$. This study introduces $\textbf{LongBioBench}$, a novel benchmark that utilizes artificially generated biographies as a controlled environment for assessing LCLMs across dimensions of $\textit{understanding}$, $\textit{reasoning}$, and $\textit{trustworthiness}$. Our experimental evaluation, which includes $\textbf{18}$ LCLMs in total, demonstrates that most models still exhibit deficiencies in semantic understanding and elementary reasoning over retrieved results and are less trustworthy as context length increases. Our further analysis indicates some design choices employed by existing synthetic benchmarks, such as contextual non-coherence, numerical needles, and the absence of distractors, rendering them vulnerable to test the model long-context capabilities. Moreover, we also reveal that long-context continual pretraining primarily adjusts RoPE embedding to accommodate extended context lengths. To sum up, compared to previous synthetic benchmarks, LongBioBench achieves a better trade-off between mirroring authentic language tasks and maintaining controllability, and is highly interpretable and configurable. 

---
# TO-GATE: Clarifying Questions and Summarizing Responses with Trajectory Optimization for Eliciting Human Preference 

**Authors**: Yulin Dou, Jiangming Liu  

**Link**: [PDF](https://arxiv.org/pdf/2506.02827)  

**Abstract**: Large language models (LLMs) can effectively elicit human preferences through multi-turn dialogue. Complex tasks can be accomplished through iterative clarifying questions and final responses generated by an LLM acting as a questioner (STaR-GATE; Andukuri et al., 2024}). However, existing approaches based on self-taught reasoning struggle to identify optimal dialogue trajectories and avoid irrelevant questions to the tasks. To address this limitation, we propose TO-GATE, a novel framework that enhances question generation through trajectory optimization, which consists of two key components: a clarification resolver that generates optimal questioning trajectories, and a summarizer that ensures task-aligned final responses. The trajectory optimization enables the model to produce effective elicitation questions and summary responses tailored to specific tasks. Experimental results demonstrate that TO-GATE significantly outperforms baseline methods, achieving a 9.32% improvement on standard preference elicitation tasks. 

---
# MASTER: Enhancing Large Language Model via Multi-Agent Simulated Teaching 

**Authors**: Liang Yue, Yihong Tang, Kehai Chen, Jie Liu, Min Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2506.02689)  

**Abstract**: Instruction fine-tuning is crucial in NLP tasks, enhancing pretrained models' instruction-following capabilities and task-specific performance. However, obtaining high-quality fine-tuning data for large models is challenging due to data collection difficulties and high production costs. To address this, we propose MASTER, a novel data augmentation method that enriches original data through interactions among multiple agents with varying cognitive levels. We simulate three pedagogically grounded teaching scenarios, leveraging multi-agent conversations to generate high-quality teacher-student interaction data. Utilizing MASTER, we construct BOOST-QA, a fine-tuning dataset augmented from existing datasets like Orca-Math-200k, ProcQA, and OpenHermes2.5. Experiments show that models fine-tuned with BOOST-QA perform excellently across multiple benchmarks, demonstrating strong multitask generalization. Notably, MASTER significantly improves models' reasoning abilities in complex tasks, providing valuable insights for future research. 

---
# TL;DR: Too Long, Do Re-weighting for Effcient LLM Reasoning Compression 

**Authors**: Zhong-Zhi Li, Xiao Liang, Zihao Tang, Lei Ji, Peijie Wang, Haotian Xu, Xing W, Haizhen Huang, Weiwei Deng, Ying Nian Wu, Yeyun Gong, Zhijiang Guo, Xiao Liu, Fei Yin, Cheng-Lin Liu  

**Link**: [PDF](https://arxiv.org/pdf/2506.02678)  

**Abstract**: Large Language Models (LLMs) have recently achieved remarkable progress by leveraging Reinforcement Learning and extended Chain-of-Thought (CoT) techniques. However, the challenge of performing efficient language reasoning--especially during inference with extremely long outputs--has drawn increasing attention from the research community. In this work, we propose a dynamic ratio-based training pipeline that does not rely on sophisticated data annotations or interpolation between multiple models. We continuously balance the weights between the model's System-1 and System-2 data to eliminate redundant reasoning processes while preserving the model's reasoning capability. We validate our approach across models on DeepSeek-R1-Distill-7B and DeepSeek-R1-Distill-14B and on a diverse set of benchmarks with varying difficulty levels. Our method significantly reduces the number of output tokens by nearly 40% while maintaining the accuracy of the reasoning. Our code and data will be available soon. 

---
# ProcrustesGPT: Compressing LLMs with Structured Matrices and Orthogonal Transformations 

**Authors**: Ekaterina Grishina, Mikhail Gorbunov, Maxim Rakhuba  

**Link**: [PDF](https://arxiv.org/pdf/2506.02818)  

**Abstract**: Large language models (LLMs) demonstrate impressive results in natural language processing tasks but require a significant amount of computational and memory resources. Structured matrix representations are a promising way for reducing the number of parameters of these models. However, it seems unrealistic to expect that weight matrices of pretrained models can be accurately represented by structured matrices without any fine-tuning. To overcome this issue, we utilize the fact that LLM output is invariant under certain orthogonal transformations of weight matrices. This insight can be leveraged to identify transformations that significantly improve the compressibility of weights within structured classes. The proposed approach is applicable to various types of structured matrices that support efficient projection operations. Code is available at this https URL 

---
# On Generalization across Measurement Systems: LLMs Entail More Test-Time Compute for Underrepresented Cultures 

**Authors**: Minh Duc Bui, Kyung Eun Park, Goran Glavaš, Fabian David Schmidt, Katharina von der Wense  

**Link**: [PDF](https://arxiv.org/pdf/2506.02591)  

**Abstract**: Measurement systems (e.g., currencies) differ across cultures, but the conversions between them are well defined so that humans can state facts using any measurement system of their choice. Being available to users from diverse cultural backgrounds, large language models (LLMs) should also be able to provide accurate information irrespective of the measurement system at hand. Using newly compiled datasets we test if this is the case for seven open-source LLMs, addressing three key research questions: (RQ1) What is the default system used by LLMs for each type of measurement? (RQ2) Do LLMs' answers and their accuracy vary across different measurement systems? (RQ3) Can LLMs mitigate potential challenges w.r.t. underrepresented systems via reasoning? Our findings show that LLMs default to the measurement system predominantly used in the data. Additionally, we observe considerable instability and variance in performance across different measurement systems. While this instability can in part be mitigated by employing reasoning methods such as chain-of-thought (CoT), this implies longer responses and thereby significantly increases test-time compute (and inference costs), marginalizing users from cultural backgrounds that use underrepresented measurement systems. 

---
# IndoSafety: Culturally Grounded Safety for LLMs in Indonesian Languages 

**Authors**: Muhammad Falensi Azmi, Muhammad Dehan Al Kautsar, Alfan Farizki Wicaksono, Fajri Koto  

**Link**: [PDF](https://arxiv.org/pdf/2506.02573)  

**Abstract**: Although region-specific large language models (LLMs) are increasingly developed, their safety remains underexplored, particularly in culturally diverse settings like Indonesia, where sensitivity to local norms is essential and highly valued by the community. In this work, we present IndoSafety, the first high-quality, human-verified safety evaluation dataset tailored for the Indonesian context, covering five language varieties: formal and colloquial Indonesian, along with three major local languages: Javanese, Sundanese, and Minangkabau. IndoSafety is constructed by extending prior safety frameworks to develop a taxonomy that captures Indonesia's sociocultural context. We find that existing Indonesian-centric LLMs often generate unsafe outputs, particularly in colloquial and local language settings, while fine-tuning on IndoSafety significantly improves safety while preserving task performance. Our work highlights the critical need for culturally grounded safety evaluation and provides a concrete step toward responsible LLM deployment in multilingual settings. Warning: This paper contains example data that may be offensive, harmful, or biased. 

---
# Decompose, Plan in Parallel, and Merge: A Novel Paradigm for Large Language Models based Planning with Multiple Constraints 

**Authors**: Zhengdong Lu, Weikai Lu, Yiling Tao, Yun Dai, ZiXuan Chen, Huiping Zhuang, Cen Chen, Hao Peng, Ziqian Zeng  

**Link**: [PDF](https://arxiv.org/pdf/2506.02683)  

**Abstract**: Despite significant advances in Large Language Models (LLMs), planning tasks still present challenges for LLM-based agents. Existing planning methods face two key limitations: heavy constraints and cascading errors. To address these limitations, we propose a novel parallel planning paradigm, which Decomposes, Plans for subtasks in Parallel, and Merges subplans into a final plan (DPPM). Specifically, DPPM decomposes the complex task based on constraints into subtasks, generates the subplan for each subtask in parallel, and merges them into a global plan. In addition, our approach incorporates a verification and refinement module, enabling error correction and conflict resolution. Experimental results demonstrate that DPPM significantly outperforms existing methods in travel planning tasks. 

---
# Answer Convergence as a Signal for Early Stopping in Reasoning 

**Authors**: Xin Liu, Lu Wang  

**Link**: [PDF](https://arxiv.org/pdf/2506.02536)  

**Abstract**: Chain-of-thought (CoT) prompting enhances reasoning in large language models (LLMs) but often leads to verbose and redundant outputs, thus increasing inference cost. We hypothesize that many reasoning steps are unnecessary for producing correct answers. To investigate this, we start with a systematic study to examine what is the minimum reasoning required for a model to reach a stable decision. We find that on math reasoning tasks like math, models typically converge to their final answers after 60\% of the reasoning steps, suggesting substantial redundancy in the remaining content. Based on these insights, we propose three inference-time strategies to improve efficiency: (1) early stopping via answer consistency, (2) boosting the probability of generating end-of-reasoning signals, and (3) a supervised method that learns when to stop based on internal activations. Experiments across five benchmarks and five open-weights LLMs show that our methods significantly reduce token usage with little or no accuracy drop. In particular, on NaturalQuestions, Answer Consistency reduces tokens by over 40\% while further improving accuracy. Our work underscores the importance of cost-effective reasoning methods that operate at inference time, offering practical benefits for real-world applications. 

---
# ReasoningFlow: Semantic Structure of Complex Reasoning Traces 

**Authors**: Jinu Lee, Sagnik Mukherjee, Dilek Hakkani-Tur, Julia Hockenmaier  

**Link**: [PDF](https://arxiv.org/pdf/2506.02532)  

**Abstract**: Large reasoning models (LRMs) generate complex reasoning traces with planning, reflection, verification, and backtracking. In this work, we introduce ReasoningFlow, a unified schema for analyzing the semantic structures of these complex traces. ReasoningFlow parses traces into directed acyclic graphs, enabling the characterization of distinct reasoning patterns as subgraph structures. This human-interpretable representation offers promising applications in understanding, evaluating, and enhancing the reasoning processes of LRMs. 

---
# Learning Together to Perform Better: Teaching Small-Scale LLMs to Collaborate via Preferential Rationale Tuning 

**Authors**: Sohan Patnaik, Milan Aggarwal, Sumit Bhatia, Balaji Krishnamurthy  

**Link**: [PDF](https://arxiv.org/pdf/2506.02519)  

**Abstract**: LLMssuch as GPT-4 have shown a remarkable ability to solve complex questions by generating step-by-step rationales. Prior works have utilized this capability to improve smaller and cheaper LMs (say, with 7B parameters). However, various practical constraints, such as copyright and legal issues, owing to lack of transparency in the pre-training data of large (often closed) models, prevent their use in commercial settings. Little focus has been given to improving the innate reasoning ability of smaller models without distilling information from larger LLMs. To address this, we propose COLLATE, a trainable framework that tunes a (small) LLM to generate those outputs from a pool of diverse rationales that selectively improves the downstream task. COLLATE enforces multiple instances of the same LLM to exhibit distinct behavior and employs them to generate rationales to obtain diverse outputs. The LLM is then tuned via preference optimization to choose the candidate rationale which maximizes the likelihood of ground-truth answer. COLLATE outperforms several trainable and prompting baselines on 5 datasets across 3 domains: maths problem solving, natural language inference, and commonsense reasoning. We show the eff icacy of COLLATE on LLMs from different model families across varying parameter scales (1B to 8B) and demonstrate the benefit of multiple rationale providers guided by the end task through ablations. Code is released here (this https URL). 

---
# KARE-RAG: Knowledge-Aware Refinement and Enhancement for RAG 

**Authors**: Yongjian Li, HaoCheng Chu, Yukun Yan, Zhenghao Liu, Shi Yu, Zheni Zeng, Ruobing Wang, Sen Song, Zhiyuan Liu, Maosong Sun  

**Link**: [PDF](https://arxiv.org/pdf/2506.02503)  

**Abstract**: Retrieval-Augmented Generation (RAG) enables large language models (LLMs) to access broader knowledge sources, yet factual inconsistencies persist due to noise in retrieved documents-even with advanced retrieval methods. We demonstrate that enhancing generative models' capacity to process noisy content is equally critical for robust performance. In this paper, we present KARE-RAG (Knowledge-Aware Refinement and Enhancement for RAG), which improves knowledge utilization through three key innovations: (1) structured knowledge representations that facilitate error detection during training, (2) Dense Direct Preference Optimization (DDPO)-a refined training objective that prioritizes correction of critical errors, and (3) a contrastive data generation pipeline that maintains semantic consistency while rectifying factual inaccuracies. Experiments show our method significantly enhances standard RAG pipelines across model scales, improving both in-domain and out-of-domain task performance without compromising general capabilities. Notably, these gains are achieved with modest training data, suggesting data-efficient optimization is possible through targeted learning strategies. Our findings establish a new direction for RAG improvement: by improving how models learn to process retrieved content, we can enhance performance across diverse inference paradigms. All data and code will be publicly available on Github. 

---
# Enhancing Large Language Models with Neurosymbolic Reasoning for Multilingual Tasks 

**Authors**: Sina Bagheri Nezhad, Ameeta Agrawal  

**Link**: [PDF](https://arxiv.org/pdf/2506.02483)  

**Abstract**: Large language models (LLMs) often struggle to perform multi-target reasoning in long-context scenarios where relevant information is scattered across extensive documents. To address this challenge, we introduce NeuroSymbolic Augmented Reasoning (NSAR), which combines the benefits of neural and symbolic reasoning during inference. NSAR explicitly extracts symbolic facts from text and generates executable Python code to handle complex reasoning steps. Through extensive experiments across seven languages and diverse context lengths, we demonstrate that NSAR significantly outperforms both a vanilla RAG baseline and advanced prompting strategies in accurately identifying and synthesizing multiple pieces of information. Our results highlight the effectiveness of combining explicit symbolic operations with neural inference for robust, interpretable, and scalable reasoning in multilingual settings. 

---
# ORPP: Self-Optimizing Role-playing Prompts to Enhance Language Model Capabilities 

**Authors**: Yifan Duan, Yihong Tang, Kehai Chen, Liqiang Nie, Min Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2506.02480)  

**Abstract**: High-quality prompts are crucial for eliciting outstanding performance from large language models (LLMs) on complex tasks. Existing research has explored model-driven strategies for prompt optimization. However, these methods often suffer from high computational overhead or require strong optimization capabilities from the model itself, which limits their broad this http URL address these challenges, we propose ORPP (Optimized Role-Playing Prompt),a framework that enhances model performance by optimizing and generating role-playing prompts. The core idea of ORPP is to confine the prompt search space to role-playing scenarios, thereby fully activating the model's intrinsic capabilities through carefully crafted, high-quality role-playing prompts. Specifically, ORPP first performs iterative optimization on a small subset of training samples to generate high-quality role-playing prompts. Then, leveraging the model's few-shot learning capability, it transfers the optimization experience to efficiently generate suitable prompts for the remaining this http URL experimental results show that ORPP not only matches but in most cases surpasses existing mainstream prompt optimization methods in terms of performance. Notably, ORPP demonstrates superior "plug-and-play" capability. In most cases, it can be integrated with various other prompt methods and further enhance their effectiveness. 

---
# XToM: Exploring the Multilingual Theory of Mind for Large Language Models 

**Authors**: Chunkit Chan, Yauwai Yim, Hongchuan Zeng, Zhiying Zou, Xinyuan Cheng, Zhifan Sun, Zheye Deng, Kawai Chung, Yuzhuo Ao, Yixiang Fan, Cheng Jiayang, Ercong Nie, Ginny Y. Wong, Helmut Schmid, Hinrich Schütze, Simon See, Yangqiu Song  

**Link**: [PDF](https://arxiv.org/pdf/2506.02461)  

**Abstract**: Theory of Mind (ToM), the ability to infer mental states in others, is pivotal for human social cognition. Existing evaluations of ToM in LLMs are largely limited to English, neglecting the linguistic diversity that shapes human cognition. This limitation raises a critical question: can LLMs exhibit Multilingual Theory of Mind, which is the capacity to reason about mental states across diverse linguistic contexts? To address this gap, we present XToM, a rigorously validated multilingual benchmark that evaluates ToM across five languages and incorporates diverse, contextually rich task scenarios. Using XToM, we systematically evaluate LLMs (e.g., DeepSeek R1), revealing a pronounced dissonance: while models excel in multilingual language understanding, their ToM performance varies across languages. Our findings expose limitations in LLMs' ability to replicate human-like mentalizing across linguistic contexts. 

---
# From Anger to Joy: How Nationality Personas Shape Emotion Attribution in Large Language Models 

**Authors**: Mahammed Kamruzzaman, Abdullah Al Monsur, Gene Louis Kim, Anshuman Chhabra  

**Link**: [PDF](https://arxiv.org/pdf/2506.02431)  

**Abstract**: Emotions are a fundamental facet of human experience, varying across individuals, cultural contexts, and nationalities. Given the recent success of Large Language Models (LLMs) as role-playing agents, we examine whether LLMs exhibit emotional stereotypes when assigned nationality-specific personas. Specifically, we investigate how different countries are represented in pre-trained LLMs through emotion attributions and whether these attributions align with cultural norms. Our analysis reveals significant nationality-based differences, with emotions such as shame, fear, and joy being disproportionately assigned across regions. Furthermore, we observe notable misalignment between LLM-generated and human emotional responses, particularly for negative emotions, highlighting the presence of reductive and potentially biased stereotypes in LLM outputs. 

---
# MidPO: Dual Preference Optimization for Safety and Helpfulness in Large Language Models via a Mixture of Experts Framework 

**Authors**: Yupeng Qi, Ziyu Lyu, Min Yang, Yanlin Wang, Lu Bai, Lixin Cui  

**Link**: [PDF](https://arxiv.org/pdf/2506.02460)  

**Abstract**: As large language models (LLMs) are increasingly applied across various domains, enhancing safety while maintaining the helpfulness of LLMs has become a critical challenge. Recent studies solve this problem through safety-constrained online preference optimization or safety-constrained offline preference optimization. However, the safety-constrained online methods often suffer from excessive safety, which might reduce helpfulness, while the safety-constrained offline methods perform poorly in adaptively balancing safety and helpfulness. To address these limitations, we propose MidPO, a \textbf{\underline{Mi}}xture of Experts (MoE) framework for safety-helpfulness \textbf{\underline{d}}ual \textbf{\underline{P}}reference \textbf{\underline{O}}ptimization. Firstly, MidPO devises single-preference enhanced direct preference optimization approach to transform the base model into two independent experts, termed safety and helpfulness experts, and fine-tunes the two independent experts for optimal safety or helpfulness performance. Secondly, to achieve an effective balance between safety and helpfulness, MidPO incorporates the two experts into the MoE framework and designs a dynamic routing mechanism to allocate contributions from each expert adaptively. We conduct quantitative and qualitative experiments on three popular datasets to demonstrate the proposed MidPO significantly outperforms state-of-the-art approaches in both safety and helpfulness. The code and models will be released. 

---
# Truth over Tricks: Measuring and Mitigating Shortcut Learning in Misinformation Detection 

**Authors**: Herun Wan, Jiaying Wu, Minnan Luo, Zhi Zeng, Zhixiong Su  

**Link**: [PDF](https://arxiv.org/pdf/2506.02350)  

**Abstract**: Misinformation detection models often rely on superficial cues (i.e., \emph{shortcuts}) that correlate with misinformation in training data but fail to generalize to the diverse and evolving nature of real-world misinformation. This issue is exacerbated by large language models (LLMs), which can easily generate convincing misinformation through simple prompts. We introduce TruthOverTricks, a unified evaluation paradigm for measuring shortcut learning in misinformation detection. TruthOverTricks categorizes shortcut behaviors into intrinsic shortcut induction and extrinsic shortcut injection, and evaluates seven representative detectors across 14 popular benchmarks, along with two new factual misinformation datasets, NQ-Misinfo and Streaming-Misinfo. Empirical results reveal that existing detectors suffer severe performance degradation when exposed to both naturally occurring and adversarially crafted shortcuts. To address this, we propose SMF, an LLM-augmented data augmentation framework that mitigates shortcut reliance through paraphrasing, factual summarization, and sentiment normalization. SMF consistently enhances robustness across 16 benchmarks, encouraging models to rely on deeper semantic understanding rather than shortcut cues. To promote the development of misinformation detectors, we have published the resources publicly at this https URL. 

---
# Should LLM Safety Be More Than Refusing Harmful Instructions? 

**Authors**: Utsav Maskey, Mark Dras, Usman Naseem  

**Link**: [PDF](https://arxiv.org/pdf/2506.02442)  

**Abstract**: This paper presents a systematic evaluation of Large Language Models' (LLMs) behavior on long-tail distributed (encrypted) texts and their safety implications. We introduce a two-dimensional framework for assessing LLM safety: (1) instruction refusal-the ability to reject harmful obfuscated instructions, and (2) generation safety-the suppression of generating harmful responses. Through comprehensive experiments, we demonstrate that models that possess capabilities to decrypt ciphers may be susceptible to mismatched-generalization attacks: their safety mechanisms fail on at least one safety dimension, leading to unsafe responses or over-refusal. Based on these findings, we evaluate a number of pre-LLM and post-LLM safeguards and discuss their strengths and limitations. This work contributes to understanding the safety of LLM in long-tail text scenarios and provides directions for developing robust safety mechanisms. 

---
# Explain-then-Process: Using Grammar Prompting to Enhance Grammatical Acceptability Judgments 

**Authors**: Russell Scheinberg, Ameeta Agrawal, Amber Shore, So Young Lee  

**Link**: [PDF](https://arxiv.org/pdf/2506.02302)  

**Abstract**: Large language models (LLMs) can explain grammatical rules, yet they often fail to apply those rules when judging sentence acceptability. We present "grammar prompting", an explain-then-process paradigm: a large LLM first produces a concise explanation of the relevant syntactic phenomenon, then that explanation is fed back as additional context to the target model -- either an LLM or a smaller language model (SLM) -- before deciding which sentence of a minimal pair is grammatical. On the English BLiMP, Chinese SLING, and Russian RuBLiMP benchmarks, this simple prompt design yields substantial improvements over strong baselines across many syntactic phenomena. Feeding an LLM's metalinguistic explanation back to the target model bridges the gap between knowing a rule and using it. On SLMs, grammar prompting alone trims the average LLM-SLM accuracy gap by about 20%, and when paired with chain-of-thought, by 56% (13.0 pp -> 5.8 pp), all at negligible cost. The lightweight, language-agnostic cue lets low-cost SLMs approach frontier-LLM performance in multilingual settings. 

---
# One Missing Piece for Open-Source Reasoning Models: A Dataset to Mitigate Cold-Starting Short CoT LLMs in RL 

**Authors**: Hyungjoo Chae, Dongjin Kang, Jihyuk Kim, Beong-woo Kwak, Sunghyun Park, Haeju Park, Jinyoung Yeo, Moontae Lee, Kyungjae Lee  

**Link**: [PDF](https://arxiv.org/pdf/2506.02338)  

**Abstract**: With the release of R1, a publicly available large reasoning model (LRM), researchers commonly train new LRMs by training language models on R1's long chain-of-thought (CoT) inferences. While prior works show that LRMs' capabilities can be reproduced through direct distillation, the continued reliance on the existing models (e.g., R1) remains a critical limitation in advancing the field. As a first step toward independent LRM development, this paper explores the possibility of constructing a long CoT dataset with LLMs that are not trained for inference-time scaling. To this end, we present the Long CoT Collection, a dataset of 100K CoT rationales annotated using existing short CoT LLMs. We develop a pipeline that induces o1's novel reasoning strategies into short CoT LLMs, enabling them to think longer and introducing controllability over the thought budget to better manage the overthinking problem. Our extensive analyses validate that our dataset achieves quality comparable to--or slightly below--R1. Furthermore, our experiments demonstrate that training on our dataset not only strengthens general reasoning skills, but also provides a strong foundation for reinforcement learning--models initialized on our data achieve 2-3x larger gains with RLVR. 

---
# ChatCFD: an End-to-End CFD Agent with Domain-specific Structured Thinking 

**Authors**: E Fan, Weizong Wang, Tianhan Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2506.02019)  

**Abstract**: Computational Fluid Dynamics (CFD) is essential for scientific and engineering advancements but is limited by operational complexity and the need for extensive expertise. This paper presents ChatCFD, a large language model-driven pipeline that automates CFD workflows within the OpenFOAM framework. It enables users to configure and execute complex simulations from natural language prompts or published literature with minimal expertise. The innovation is its structured approach to database construction, configuration validation, and error reflection, integrating CFD and OpenFOAM knowledge with general language models to improve accuracy and adaptability. Validation shows ChatCFD can autonomously reproduce published CFD results, handling complex, unseen configurations beyond basic examples, a task challenging for general language models. 

---
# FinS-Pilot: A Benchmark for Online Financial System 

**Authors**: Feng Wang, Yiding Sun, Jiaxin Mao, Wei Xue, Danqing Xu  

**Link**: [PDF](https://arxiv.org/pdf/2506.02037)  

**Abstract**: Large language models (LLMs) have demonstrated remarkable capabilities across various professional domains, with their performance typically evaluated through standardized benchmarks. However, the development of financial RAG benchmarks has been constrained by data confidentiality issues and the lack of dynamic data integration. To address this issue, we introduces FinS-Pilot, a novel benchmark for evaluating RAG systems in online financial applications. Constructed from real-world financial assistant interactions, our benchmark incorporates both real-time API data and structured text sources, organized through an intent classification framework covering critical financial domains such as equity analysis and macroeconomic forecasting. The benchmark enables comprehensive evaluation of financial assistants' capabilities in handling both static knowledge and time-sensitive market information. Through systematic experiments with multiple Chinese leading LLMs, we demonstrate FinS-Pilot's effectiveness in identifying models suitable for financial applications while addressing the current gap in specialized evaluation tools for the financial domain. Our work contributes both a practical evaluation framework and a curated dataset to advance research in financial NLP systems. The code and dataset are accessible on GitHub\footnote{this https URL\_rag\_benchmark}. 

---
# Evaluating the Unseen Capabilities: How Many Theorems Do LLMs Know? 

**Authors**: Xiang Li, Jiayi Xin, Qi Long, Weijie J. Su  

**Link**: [PDF](https://arxiv.org/pdf/2506.02058)  

**Abstract**: Accurate evaluation of large language models (LLMs) is crucial for understanding their capabilities and guiding their development. However, current evaluations often inconsistently reflect the actual capacities of these models. In this paper, we demonstrate that one of many contributing factors to this \textit{evaluation crisis} is the oversight of unseen knowledge -- information encoded by LLMs but not directly observed or not yet observed during evaluations. We introduce KnowSum, a statistical framework designed to provide a more comprehensive assessment by quantifying the unseen knowledge for a class of evaluation tasks. KnowSum estimates the unobserved portion by extrapolating from the appearance frequencies of observed knowledge instances. We demonstrate the effectiveness and utility of KnowSum across three critical applications: estimating total knowledge, evaluating information retrieval effectiveness, and measuring output diversity. Our experiments reveal that a substantial volume of knowledge is omitted when relying solely on observed LLM performance. Importantly, KnowSum yields significantly different comparative rankings for several common LLMs based on their internal knowledge. 

---
# Knowledge or Reasoning? A Close Look at How LLMs Think Across Domains 

**Authors**: Juncheng Wu, Sheng Liu, Haoqin Tu, Hang Yu, Xiaoke Huang, James Zou, Cihang Xie, Yuyin Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2506.02126)  

**Abstract**: Recent advances in reasoning-enhanced Large Language Models such as OpenAI-o1/3 and DeepSeek-R1 have significantly improved performance on complex tasks. However, the quality and transparency of their internal reasoning processes remain underexplored. This work moves beyond the final-answer accuracy and investigates step-by-step reasoning in the medical and mathematical domains by explicitly decomposing the thinking trajectories into two parts: knowledge and reasoning. Specifically, we introduce a fine-grained evaluation framework that judges: (1) the correctness of knowledge used (measured by Knowledge Index (KI)) and (2) the quality of reasoning (measured by Information Gain (InfoGain)). Using this framework, we study R1-distilled and base Qwen models trained with supervised fine-tuning (SFT) and/or reinforcement learning (RL) in the medical and math domains. Three intriguing findings emerge: (1) The general reasoning abilities in R1-distilled models do not transfer effectively to the medical domain through either SFT or RL. (2) SFT raises final-answer accuracy in both domains, but often at the cost of reasoning quality: InfoGain drops by 38.9% on average compared with untrained models; In the medical domain, however, SFT remains crucial because domain knowledge is indispensable. (3) RL enhances medical reasoning by pruning inaccurate or irrelevant knowledge from reasoning paths, thereby improving both reasoning accuracy and knowledge correctness. 

---
# An Exploratory Framework for Future SETI Applications: Detecting Generative Reactivity via Language Models 

**Authors**: Po-Chieh Yu  

**Link**: [PDF](https://arxiv.org/pdf/2506.02730)  

**Abstract**: We present an exploratory framework to test whether noise-like input can induce structured responses in language models. Instead of assuming that extraterrestrial signals must be decoded, we evaluate whether inputs can trigger linguistic behavior in generative systems. This shifts the focus from decoding to viewing structured output as a sign of underlying regularity in the input. We tested GPT-2 small, a 117M-parameter model trained on English text, using four types of acoustic input: human speech, humpback whale vocalizations, Phylloscopus trochilus birdsong, and algorithmically generated white noise. All inputs were treated as noise-like, without any assumed symbolic encoding. To assess reactivity, we defined a composite score called Semantic Induction Potential (SIP), combining entropy, syntax coherence, compression gain, and repetition penalty. Results showed that whale and bird vocalizations had higher SIP scores than white noise, while human speech triggered only moderate responses. This suggests that language models may detect latent structure even in data without conventional semantics. We propose that this approach could complement traditional SETI methods, especially in cases where communicative intent is unknown. Generative reactivity may offer a different way to identify data worth closer attention. 

---
# NovelHopQA: Diagnosing Multi-Hop Reasoning Failures in Long Narrative Contexts 

**Authors**: Abhay Gupta, Michael Lu, Kevin Zhu, Sean O'Brien, Vasu Sharma  

**Link**: [PDF](https://arxiv.org/pdf/2506.02000)  

**Abstract**: Current large language models (LLMs) struggle to answer questions that span tens of thousands of tokens, especially when multi-hop reasoning is involved. While prior benchmarks explore long-context comprehension or multi-hop reasoning in isolation, none jointly vary context length and reasoning depth in natural narrative settings. We introduce NovelHopQA, the first benchmark to evaluate k1-4 hop QA over 64k-128k-token excerpts from 83 full-length public-domain novels. A keyword-guided pipeline builds hop-separated chains grounded in coherent storylines. We evaluate six state-of-the-art (SOTA) models and apply oracle-context filtering to ensure all questions are genuinely answerable. Human annotators validate both alignment and hop depth. We noticed consistent accuracy drops with increased hops and context length, even in frontier models-revealing that sheer scale does not guarantee robust reasoning. Our failure mode analysis highlights common breakdowns, such as missed final-hop integration and long-range drift. NovelHopQA offers a controlled diagnostic setting to stress-test multi-hop reasoning at scale. 

---
# A Dynamic Framework for Semantic Grouping of Common Data Elements (CDE) Using Embeddings and Clustering 

**Authors**: Madan Krishnamurthy, Daniel Korn, Melissa A Haendel, Christopher J Mungall, Anne E Thessen  

**Link**: [PDF](https://arxiv.org/pdf/2506.02160)  

**Abstract**: This research aims to develop a dynamic and scalable framework to facilitate harmonization of Common Data Elements (CDEs) across heterogeneous biomedical datasets by addressing challenges such as semantic heterogeneity, structural variability, and context dependence to streamline integration, enhance interoperability, and accelerate scientific discovery. Our methodology leverages Large Language Models (LLMs) for context-aware text embeddings that convert CDEs into dense vectors capturing semantic relationships and patterns. These embeddings are clustered using Hierarchical Density-Based Spatial Clustering of Applications with Noise (HDBSCAN) to group semantically similar CDEs. The framework incorporates four key steps: (1) LLM-based text embedding to mathematically represent semantic context, (2) unsupervised clustering of embeddings via HDBSCAN, (3) automated labeling using LLM summarization, and (4) supervised learning to train a classifier assigning new or unclustered CDEs to labeled clusters. Evaluated on the NIH NLM CDE Repository with over 24,000 CDEs, the system identified 118 meaningful clusters at an optimized minimum cluster size of 20. The classifier achieved 90.46 percent overall accuracy, performing best in larger categories. External validation against Gravity Projects Social Determinants of Health domains showed strong agreement (Adjusted Rand Index 0.52, Normalized Mutual Information 0.78), indicating that embeddings effectively capture cluster characteristics. This adaptable and scalable approach offers a practical solution to CDE harmonization, improving selection efficiency and supporting ongoing data interoperability. 

---
# Breaking Quadratic Barriers: A Non-Attention LLM for Ultra-Long Context Horizons 

**Authors**: Andrew Kiruluta, Preethi Raju, Priscilla Burity  

**Link**: [PDF](https://arxiv.org/pdf/2506.01963)  

**Abstract**: We present a novel non attention based architecture for large language models (LLMs) that efficiently handles very long context windows, on the order of hundreds of thousands to potentially millions of tokens. Unlike traditional Transformer designs, which suffer from quadratic memory and computation overload due to the nature of the self attention mechanism, our model avoids token to token attention entirely. Instead, it combines the following complementary components: State Space blocks (inspired by S4) that learn continuous time convolution kernels and scale near linearly with sequence length, Multi Resolution Convolution layers that capture local context at different dilation levels, a lightweight Recurrent Supervisor to maintain a global hidden state across sequential chunks, and Retrieval Augmented External Memory that stores and retrieves high-level chunk embeddings without reintroducing quadratic operations. 

---
# NextQuill: Causal Preference Modeling for Enhancing LLM Personalization 

**Authors**: Xiaoyan Zhao, Juntao You, Yang Zhang, Wenjie Wang, Hong Cheng, Fuli Feng, See-Kiong Ng, Tat-Seng Chua  

**Link**: [PDF](https://arxiv.org/pdf/2506.02368)  

**Abstract**: Personalizing large language models (LLMs) for individual users has become increasingly important as they are progressively integrated into real-world applications to support users' daily lives. However, existing personalization approaches often fail to distinguish which components of model predictions and training data truly reflect user preferences, leading to superficial personalization alignment. In this paper, we introduce NextQuill, a novel LLM personalization alignment framework grounded in causal preference modeling. We approach personalization from a causal perspective, treating both model predictions and ground-truth data generation as outcomes influenced by user preferences, along with other factors. We define the true preference effect as the causal impact of user history (which reflects preferences) on each token prediction or data generation instance, estimated through causal intervention techniques. Building on this insight, NextQuill introduces two complementary alignment strategies: (1) aligning model-internal causal preference effects on predictions with those reflected in ground-truth data, rather than indiscriminately fitting predictions, and (2) focusing on fitting preference-bearing tokens identified via ground-truth data preference effects, rather than treating all tokens uniformly. By integrating these strategies, NextQuill shifts the alignment process toward learning from causal preference effects, facilitating more effective and personalized adaptation. Experiments across multiple personalization benchmarks demonstrate that NextQuill significantly improves personalization quality, offering a principled, causal foundation for LLM personalization. Our codes are available on this https URL. 

---
