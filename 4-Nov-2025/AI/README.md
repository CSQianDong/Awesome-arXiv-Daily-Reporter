# Simulating Environments with Reasoning Models for Agent Training 

**Authors**: Yuetai Li, Huseyin A Inan, Xiang Yue, Wei-Ning Chen, Lukas Wutschitz, Janardhan Kulkarni, Radha Poovendran, Robert Sim, Saravan Rajmohan  

**Link**: [PDF](https://arxiv.org/pdf/2511.01824)  

**Abstract**: LLM agents excel in compact environments requiring deep reasoning but remain brittle when operating in broader, more complex contexts that demand robustness across diverse tools and schemas. Building bespoke environments for training is heavy, brittle, and limits progress. In this paper, we demonstrate that LLMs can simulate realistic environment feedback without access to actual testbed data or APIs. Inspired by this capability, we propose two frameworks: Simia-SFT, a pipeline that synthesizes SFT data by amplifying small seed sets into diverse trajectories in an environment-agnostic manner, and Simia-RL, a framework that enables RL training without real environment implementations through LLM-simulated feedback. Fine-tuning open models yields consistent improvements across multiple benchmarks, surpassing GPT-4o and approaching o4-mini on $\tau^2$-Bench. Together, Simia-SFT and Simia-RL enable scalable agent training without environment engineering, replacing heavy and brittle implementations with flexible LLM-based simulation. 

---
# Hybrid Retrieval-Augmented Generation Agent for Trustworthy Legal Question Answering in Judicial Forensics 

**Authors**: Yueqing Xi, Yifan Bai, Huasen Luo, Weiliang Wen, Hui Liu, Haoliang Li  

**Link**: [PDF](https://arxiv.org/pdf/2511.01668)  

**Abstract**: As artificial intelligence permeates judicial forensics, ensuring the veracity and traceability of legal question answering (QA) has become critical. Conventional large language models (LLMs) are prone to hallucination, risking misleading guidance in legal consultation, while static knowledge bases struggle to keep pace with frequently updated statutes and case law. We present a hybrid legal QA agent tailored for judicial settings that integrates retrieval-augmented generation (RAG) with multi-model ensembling to deliver reliable, auditable, and continuously updatable counsel. The system prioritizes retrieval over generation: when a trusted legal repository yields relevant evidence, answers are produced via RAG; otherwise, multiple LLMs generate candidates that are scored by a specialized selector, with the top-ranked answer returned. High-quality outputs then undergo human review before being written back to the repository, enabling dynamic knowledge evolution and provenance tracking. Experiments on the Law\_QA dataset show that our hybrid approach significantly outperforms both a single-model baseline and a vanilla RAG pipeline on F1, ROUGE-L, and an LLM-as-a-Judge metric. Ablations confirm the complementary contributions of retrieval prioritization, model ensembling, and the human-in-the-loop update mechanism. The proposed system demonstrably reduces hallucination while improving answer quality and legal compliance, advancing the practical landing of media forensics technologies in judicial scenarios. 

---
# IVGAE-TAMA-BO: A novel temporal dynamic variational graph model for link prediction in global food trade networks with momentum structural memory and Bayesian optimization 

**Authors**: Sicheng Wang, Shuhao Chen, Jingran Zhou, Chengyi Tu  

**Link**: [PDF](https://arxiv.org/pdf/2511.01639)  

**Abstract**: Global food trade plays a crucial role in ensuring food security and maintaining supply chain stability. However, its network structure evolves dynamically under the influence of geopolitical, economic, and environmental factors, making it challenging to model and predict future trade links. Effectively capturing temporal patterns in food trade networks is therefore essential for improving the accuracy and robustness of link prediction. This study introduces IVGAE-TAMA-BO, a novel dynamic graph neural network designed to model evolving trade structures and predict future links in global food trade networks. To the best of our knowledge, this is the first work to apply dynamic graph neural networks to this domain, significantly enhancing predictive performance. Building upon the original IVGAE framework, the proposed model incorporates a Trade-Aware Momentum Aggregator (TAMA) to capture the temporal evolution of trade networks, jointly modeling short-term fluctuations and long-term structural dependencies. A momentum-based structural memory mechanism further improves predictive stability and performance. In addition, Bayesian optimization is used to automatically tune key hyperparameters, enhancing generalization across diverse trade scenarios. Extensive experiments on five crop-specific datasets demonstrate that IVGAE-TAMA substantially outperforms the static IVGAE and other dynamic baselines by effectively modeling temporal dependencies, while Bayesian optimization further boosts performance in IVGAE-TAMA-BO. These results highlight the proposed framework as a robust and scalable solution for structural prediction in global trade networks, with strong potential for applications in food security monitoring and policy decision support. 

---
# ExplicitLM: Decoupling Knowledge from Parameters via Explicit Memory Banks 

**Authors**: Chengzhang Yu, Zening Lu, Chenyang Zheng, Chiyue Wang, Yiming Zhang, Zhanpeng Jin  

**Link**: [PDF](https://arxiv.org/pdf/2511.01581)  

**Abstract**: Large language models suffer from knowledge staleness and lack of interpretability due to implicit knowledge storage across entangled network parameters, preventing targeted updates and reasoning transparency. We propose ExplicitLM, a novel architecture featuring a million-scale external memory bank storing human-readable knowledge as token sequences, enabling direct inspection and modification. We design a differentiable two-stage retrieval mechanism with efficient coarse-grained filtering via product key decomposition (reducing complexity from $\mathcal{O}(N \cdot |I|)$ to $\mathcal{O}(\sqrt{N} \cdot |I|)$) and fine-grained Gumbel-Softmax matching for end-to-end training. Inspired by dual-system cognitive theory, we partition knowledge into frozen explicit facts (20%) and learnable implicit patterns (80%), maintained through Exponential Moving Average updates for stability. ExplicitLM achieves up to 43.67% improvement on knowledge-intensive tasks versus standard Transformers, with 3.62$\times$ gains in low-data regimes (10k samples). Analysis shows strong correlations between memory retrieval and performance, with correct predictions achieving 49% higher hit rates. Unlike RAG systems with frozen retrieval, our jointly optimized architecture demonstrates that interpretable, updatable models can maintain competitive performance while providing unprecedented knowledge transparency. 

---
# Analyzing Sustainability Messaging in Large-Scale Corporate Social Media 

**Authors**: Ujjwal Sharma, Stevan Rudinac, Ana Mićković, Willemijn van Dolen, Marcel Worring  

**Link**: [PDF](https://arxiv.org/pdf/2511.01550)  

**Abstract**: In this work, we introduce a multimodal analysis pipeline that leverages large foundation models in vision and language to analyze corporate social media content, with a focus on sustainability-related communication. Addressing the challenges of evolving, multimodal, and often ambiguous corporate messaging on platforms such as X (formerly Twitter), we employ an ensemble of large language models (LLMs) to annotate a large corpus of corporate tweets on their topical alignment with the 17 Sustainable Development Goals (SDGs). This approach avoids the need for costly, task-specific annotations and explores the potential of such models as ad-hoc annotators for social media data that can efficiently capture both explicit and implicit references to sustainability themes in a scalable manner. Complementing this textual analysis, we utilize vision-language models (VLMs), within a visual understanding framework that uses semantic clusters to uncover patterns in visual sustainability communication. This integrated approach reveals sectoral differences in SDG engagement, temporal trends, and associations between corporate messaging, environmental, social, governance (ESG) risks, and consumer engagement. Our methods-automatic label generation and semantic visual clustering-are broadly applicable to other domains and offer a flexible framework for large-scale social media analysis. 

---
# TPS-Bench: Evaluating AI Agents' Tool Planning \& Scheduling Abilities in Compounding Tasks 

**Authors**: Hanwen Xu, Xuyao Huang, Yuzhe Liu, Kai Yu, Zhijie Deng  

**Link**: [PDF](https://arxiv.org/pdf/2511.01527)  

**Abstract**: Large language model (LLM) agents have exhibited strong problem-solving competence across domains like research and coding. Yet, it remains underexplored whether LLM agents can tackle compounding real-world problems that require a diverse set of tools to complete. Given a broad, heterogeneous tool repository, LLM agents must not only select appropriate tools based on task planning analysis but also strategically schedule the execution order to ensure efficiency. This paper introduces TPS-Bench to benchmark the ability of LLM agents in solving such problems that demand Tool Planning and Scheduling. TPS-Bench collects 200 compounding tasks of two difficulty levels, based on a tool repository containing hundreds of model context protocol (MCP) tools. In particular, each task is composed of multiple subtasks, such as web search, map navigation, calendar checking, etc., and each subtask can be completed by a basic tool. Our evaluation emphasizes both task completion rate and efficiency. The empirical studies on popular closed-source and open-source LLMs indicate that most models can perform reasonable tool planning, but differ in scheduling. For example, GLM-4.5 achieves an outperforming task completion rate of 64.72% with extensive sequential tool calls, hence suffering from significantly long execution time. By contrast, GPT-4o prioritizes parallel tool calls but achieves only a 45.08% completion rate. Considering reinforcement learning (RL) can be a viable way to improve the scheduling efficiency without compromising performance, we perform an initial study on Qwen3-1.7B and witness a 14% reduction in execution time alongside a 6% gain in task completion rate based on rarely 100 RL training samples. Our code is available this https URL. 

---
# From Passive to Proactive: A Multi-Agent System with Dynamic Task Orchestration for Intelligent Medical Pre-Consultation 

**Authors**: ChengZhang Yu, YingRu He, Hongyan Cheng, nuo Cheng, Zhixing Liu, Dongxu Mu, Zhangrui Shen, Zhanpeng Jin  

**Link**: [PDF](https://arxiv.org/pdf/2511.01445)  

**Abstract**: Global healthcare systems face critical challenges from increasing patient volumes and limited consultation times, with primary care visits averaging under 5 minutes in many countries. While pre-consultation processes encompassing triage and structured history-taking offer potential solutions, they remain limited by passive interaction paradigms and context management challenges in existing AI systems. This study introduces a hierarchical multi-agent framework that transforms passive medical AI systems into proactive inquiry agents through autonomous task orchestration. We developed an eight-agent architecture with centralized control mechanisms that decomposes pre-consultation into four primary tasks: Triage ($T_1$), History of Present Illness collection ($T_2$), Past History collection ($T_3$), and Chief Complaint generation ($T_4$), with $T_1$--$T_3$ further divided into 13 domain-specific subtasks. Evaluated on 1,372 validated electronic health records from a Chinese medical platform across multiple foundation models (GPT-OSS 20B, Qwen3-8B, Phi4-14B), the framework achieved 87.0% accuracy for primary department triage and 80.5% for secondary department classification, with task completion rates reaching 98.2% using agent-driven scheduling versus 93.1% with sequential processing. Clinical quality scores from 18 physicians averaged 4.56 for Chief Complaints, 4.48 for History of Present Illness, and 4.69 for Past History on a 5-point scale, with consultations completed within 12.7 rounds for $T_2$ and 16.9 rounds for $T_3$. The model-agnostic architecture maintained high performance across different foundation models while preserving data privacy through local deployment, demonstrating the potential for autonomous AI systems to enhance pre-consultation efficiency and quality in clinical settings. 

---
# Robust Multimodal Sentiment Analysis via Double Information Bottleneck 

**Authors**: Huiting Huang, Tieliang Gong, Kai He, Jialun Wu, Erik Cambria, Mengling Feng  

**Link**: [PDF](https://arxiv.org/pdf/2511.01444)  

**Abstract**: Multimodal sentiment analysis has received significant attention across diverse research domains. Despite advancements in algorithm design, existing approaches suffer from two critical limitations: insufficient learning of noise-contaminated unimodal data, leading to corrupted cross-modal interactions, and inadequate fusion of multimodal representations, resulting in discarding discriminative unimodal information while retaining multimodal redundant information. To address these challenges, this paper proposes a Double Information Bottleneck (DIB) strategy to obtain a powerful, unified compact multimodal representation. Implemented within the framework of low-rank Renyi's entropy functional, DIB offers enhanced robustness against diverse noise sources and computational tractability for high-dimensional data, as compared to the conventional Shannon entropy-based methods. The DIB comprises two key modules: 1) learning a sufficient and compressed representation of individual unimodal data by maximizing the task-relevant information and discarding the superfluous information, and 2) ensuring the discriminative ability of multimodal representation through a novel attention bottleneck fusion mechanism. Consequently, DIB yields a multimodal representation that effectively filters out noisy information from unimodal data while capturing inter-modal complementarity. Extensive experiments on CMU-MOSI, CMU-MOSEI, CH-SIMS, and MVSA-Single validate the effectiveness of our method. The model achieves 47.4% accuracy under the Acc-7 metric on CMU-MOSI and 81.63% F1-score on CH-SIMS, outperforming the second-best baseline by 1.19%. Under noise, it shows only 0.36% and 0.29% performance degradation on CMU-MOSI and CMU-MOSEI respectively. 

---
# Learning to Seek Evidence: A Verifiable Reasoning Agent with Causal Faithfulness Analysis 

**Authors**: Yuhang Huang, Zekai Lin, Fan Zhong, Lei Liu  

**Link**: [PDF](https://arxiv.org/pdf/2511.01425)  

**Abstract**: Explanations for AI models in high-stakes domains like medicine often lack verifiability, which can hinder trust. To address this, we propose an interactive agent that produces explanations through an auditable sequence of actions. The agent learns a policy to strategically seek external visual evidence to support its diagnostic reasoning. This policy is optimized using reinforcement learning, resulting in a model that is both efficient and generalizable. Our experiments show that this action-based reasoning process significantly improves calibrated accuracy, reducing the Brier score by 18\% compared to a non-interactive baseline. To validate the faithfulness of the agent's explanations, we introduce a causal intervention method. By masking the visual evidence the agent chooses to use, we observe a measurable degradation in its performance ($\Delta$Brier=+0.029), confirming that the evidence is integral to its decision-making process. Our work provides a practical framework for building AI systems with verifiable and faithful reasoning capabilities. 

---
# Modulation of temporal decision-making in a deep reinforcement learning agent under the dual-task paradigm 

**Authors**: Amrapali Pednekar, Álvaro Garrido-Pérez, Yara Khaluf, Pieter Simoens  

**Link**: [PDF](https://arxiv.org/pdf/2511.01415)  

**Abstract**: This study explores the interference in temporal processing within a dual-task paradigm from an artificial intelligence (AI) perspective. In this context, the dual-task setup is implemented as a simplified version of the Overcooked environment with two variations, single task (T) and dual task (T+N). Both variations involve an embedded time production task, but the dual task (T+N) additionally involves a concurrent number comparison task. Two deep reinforcement learning (DRL) agents were separately trained for each of these tasks. These agents exhibited emergent behavior consistent with human timing research. Specifically, the dual task (T+N) agent exhibited significant overproduction of time relative to its single task (T) counterpart. This result was consistent across four target durations. Preliminary analysis of neural dynamics in the agents' LSTM layers did not reveal any clear evidence of a dedicated or intrinsic timer. Hence, further investigation is needed to better understand the underlying time-keeping mechanisms of the agents and to provide insights into the observed behavioral patterns. This study is a small step towards exploring parallels between emergent DRL behavior and behavior observed in biological systems in order to facilitate a better understanding of both. 

---
# Relaxing partition admissibility in Cluster-DAGs: a causal calculus with arbitrary variable clustering 

**Authors**: Clément Yvernes, Emilie Devijver, Adèle H. Ribeiro, Marianne Clausel--Lesourd, Éric Gaussier  

**Link**: [PDF](https://arxiv.org/pdf/2511.01396)  

**Abstract**: Cluster DAGs (C-DAGs) provide an abstraction of causal graphs in which nodes represent clusters of variables, and edges encode both cluster-level causal relationships and dependencies arisen from unobserved confounding. C-DAGs define an equivalence class of acyclic causal graphs that agree on cluster-level relationships, enabling causal reasoning at a higher level of abstraction. However, when the chosen clustering induces cycles in the resulting C-DAG, the partition is deemed inadmissible under conventional C-DAG semantics. In this work, we extend the C-DAG framework to support arbitrary variable clusterings by relaxing the partition admissibility constraint, thereby allowing cyclic C-DAG representations. We extend the notions of d-separation and causal calculus to this setting, significantly broadening the scope of causal reasoning across clusters and enabling the application of C-DAGs in previously intractable scenarios. Our calculus is both sound and atomically complete with respect to the do-calculus: all valid interventional queries at the cluster level can be derived using our rules, each corresponding to a primitive do-calculus step. 

---
# Align to Misalign: Automatic LLM Jailbreak with Meta-Optimized LLM Judges 

**Authors**: Hamin Koo, Minseon Kim, Jaehyung Kim  

**Link**: [PDF](https://arxiv.org/pdf/2511.01375)  

**Abstract**: Identifying the vulnerabilities of large language models (LLMs) is crucial for improving their safety by addressing inherent weaknesses. Jailbreaks, in which adversaries bypass safeguards with crafted input prompts, play a central role in red-teaming by probing LLMs to elicit unintended or unsafe behaviors. Recent optimization-based jailbreak approaches iteratively refine attack prompts by leveraging LLMs. However, they often rely heavily on either binary attack success rate (ASR) signals, which are sparse, or manually crafted scoring templates, which introduce human bias and uncertainty in the scoring outcomes. To address these limitations, we introduce AMIS (Align to MISalign), a meta-optimization framework that jointly evolves jailbreak prompts and scoring templates through a bi-level structure. In the inner loop, prompts are refined using fine-grained and dense feedback using a fixed scoring template. In the outer loop, the template is optimized using an ASR alignment score, gradually evolving to better reflect true attack outcomes across queries. This co-optimization process yields progressively stronger jailbreak prompts and more calibrated scoring signals. Evaluations on AdvBench and JBB-Behaviors demonstrate that AMIS achieves state-of-the-art performance, including 88.0% ASR on Claude-3.5-Haiku and 100.0% ASR on Claude-4-Sonnet, outperforming existing baselines by substantial margins. 

---
# Automatic Minds: Cognitive Parallels Between Hypnotic States and Large Language Model Processing 

**Authors**: Giuseppe Riva, Brenda K. Wiederhold, Fabrizia Mantovani  

**Link**: [PDF](https://arxiv.org/pdf/2511.01363)  

**Abstract**: The cognitive processes of the hypnotized mind and the computational operations of large language models (LLMs) share deep functional parallels. Both systems generate sophisticated, contextually appropriate behavior through automatic pattern-completion mechanisms operating with limited or unreliable executive oversight. This review examines this convergence across three principles: automaticity, in which responses emerge from associative rather than deliberative processes; suppressed monitoring, leading to errors such as confabulation in hypnosis and hallucination in LLMs; and heightened contextual dependency, where immediate cues (for example, the suggestion of a therapist or the prompt of the user) override stable knowledge.
These mechanisms reveal an observer-relative meaning gap: both systems produce coherent but ungrounded outputs that require an external interpreter to supply meaning. Hypnosis and LLMs also exemplify functional agency - the capacity for complex, goal-directed, context-sensitive behavior - without subjective agency, the conscious awareness of intention and ownership that defines human action. This distinction clarifies how purposive behavior can emerge without self-reflective consciousness, governed instead by structural and contextual dynamics. Finally, both domains illuminate the phenomenon of scheming: automatic, goal-directed pattern generation that unfolds without reflective awareness. Hypnosis provides an experimental model for understanding how intention can become dissociated from conscious deliberation, offering insights into the hidden motivational dynamics of artificial systems. Recognizing these parallels suggests that the future of reliable AI lies in hybrid architectures that integrate generative fluency with mechanisms of executive monitoring, an approach inspired by the complex, self-regulating architecture of the human mind. 

---
# Unbiased Platform-Level Causal Estimation for Search Systems: A Competitive Isolation PSM-DID Framework 

**Authors**: Ying Song, Yijing Wang, Hui Yang, Weihan Jin, Jun Xiong, Congyi Zhou, Jialin Zhu, Xiang Gao, Rong Chen, HuaGuang Deng, Ying Dai, Fei Xiao, Haihong Tang, Bo Zheng, KaiFu Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2511.01329)  

**Abstract**: Evaluating platform-level interventions in search-based two-sided marketplaces is fundamentally challenged by systemic effects such as spillovers and network interference. While widely used for causal inference, the PSM (Propensity Score Matching) - DID (Difference-in-Differences) framework remains susceptible to selection bias and cross-unit interference from unaccounted spillovers. In this paper, we introduced Competitive Isolation PSM-DID, a novel causal framework that integrates propensity score matching with competitive isolation to enable platform-level effect measurement (e.g., order volume, GMV) instead of item-level metrics in search systems.
Our approach provides theoretically guaranteed unbiased estimation under mutual exclusion conditions, with an open dataset released to support reproducible research on marketplace interference (this http URL). Extensive experiments demonstrate significant reductions in interference effects and estimation variance compared to baseline methods. Successful deployment in a large-scale marketplace confirms the framework's practical utility for platform-level causal inference. 

---
# OmniFuser: Adaptive Multimodal Fusion for Service-Oriented Predictive Maintenance 

**Authors**: Ziqi Wang, Hailiang Zhao, Yuhao Yang, Daojiang Hu, Cheng Bao, Mingyi Liu, Kai Di, Schahram Dustdar, Zhongjie Wang, Shuiguang Deng  

**Link**: [PDF](https://arxiv.org/pdf/2511.01320)  

**Abstract**: Accurate and timely prediction of tool conditions is critical for intelligent manufacturing systems, where unplanned tool failures can lead to quality degradation and production downtime. In modern industrial environments, predictive maintenance is increasingly implemented as an intelligent service that integrates sensing, analysis, and decision support across production processes. To meet the demand for reliable and service-oriented operation, we present OmniFuser, a multimodal learning framework for predictive maintenance of milling tools that leverages both visual and sensor data. It performs parallel feature extraction from high-resolution tool images and cutting-force signals, capturing complementary spatiotemporal patterns across modalities. To effectively integrate heterogeneous features, OmniFuser employs a contamination-free cross-modal fusion mechanism that disentangles shared and modality-specific components, allowing for efficient cross-modal interaction. Furthermore, a recursive refinement pathway functions as an anchor mechanism, consistently retaining residual information to stabilize fusion dynamics. The learned representations can be encapsulated as reusable maintenance service modules, supporting both tool-state classification (e.g., Sharp, Used, Dulled) and multi-step force signal forecasting. Experiments on real-world milling datasets demonstrate that OmniFuser consistently outperforms state-of-the-art baselines, providing a dependable foundation for building intelligent industrial maintenance services. 

---
# llmSHAP: A Principled Approach to LLM Explainability 

**Authors**: Filip Naudot, Tobias Sundqvist, Timotheus Kampik  

**Link**: [PDF](https://arxiv.org/pdf/2511.01311)  

**Abstract**: Feature attribution methods help make machine learning-based inference explainable by determining how much one or several features have contributed to a model's output. A particularly popular attribution method is based on the Shapley value from cooperative game theory, a measure that guarantees the satisfaction of several desirable principles, assuming deterministic inference. We apply the Shapley value to feature attribution in large language model (LLM)-based decision support systems, where inference is, by design, stochastic (non-deterministic). We then demonstrate when we can and cannot guarantee Shapley value principle satisfaction across different implementation variants applied to LLM-based decision support, and analyze how the stochastic nature of LLMs affects these guarantees. We also highlight trade-offs between explainable inference speed, agreement with exact Shapley value attributions, and principle attainment. 

---
# Graph Neural Network-Based Semi-Supervised Open-Set Fault Diagnosis for Marine Machinery Systems 

**Authors**: Chuyue Lou, M. Amine Atoui  

**Link**: [PDF](https://arxiv.org/pdf/2511.01258)  

**Abstract**: Recently, fault diagnosis methods for marine machinery systems based on deep learning models have attracted considerable attention in the shipping industry. Most existing studies assume fault classes are consistent and known between the training and test datasets, and these methods perform well under controlled environment. In practice, however, previously unseen or unknown fault types (i.e., out-of-distribution or open-set observations not present during training) can occur, causing such methods to fail and posing a significant challenge to their widespread industrial deployment. To address this challenge, this paper proposes a semi-supervised open-set fault diagnosis (SOFD) framework that enhances and extends the applicability of deep learning models in open-set fault diagnosis scenarios. The framework includes a reliability subset construction process, which uses a multi-layer fusion feature representation extracted by a supervised feature learning model to select an unlabeled test subset. The labeled training set and pseudo-labeled test subset are then fed into a semi-supervised diagnosis model to learn discriminative features for each class, enabling accurate classification of known faults and effective detection of unknown samples. Experimental results on a public maritime benchmark dataset demonstrate the effectiveness and superiority of the proposed SOFD framework. 

---
# QiMeng-NeuComBack: Self-Evolving Translation from IR to Assembly Code 

**Authors**: Hainan Fang, Yuanbo Wen, Jun Bi, Yihan Wang, Tonghui He, Yanlin Tang, Di Huang, Jiaming Guo, Rui Zhang, Qi Guo, Yunji Chen  

**Link**: [PDF](https://arxiv.org/pdf/2511.01183)  

**Abstract**: Compilers, while essential, are notoriously complex systems that demand prohibitively expensive human expertise to develop and maintain. The recent advancements in Large Language Models (LLMs) offer a compelling new paradigm: Neural Compilation, which could potentially simplify compiler development for new architectures and facilitate the discovery of innovative optimization techniques. However, several critical obstacles impede its practical adoption. Firstly, a significant lack of dedicated benchmarks and robust evaluation methodologies hinders objective assessment and tracking of progress in the field. Secondly, systematically enhancing the reliability and performance of LLM-generated assembly remains a critical challenge. Addressing these challenges, this paper introduces NeuComBack, a novel benchmark dataset specifically designed for IR-to-assembly compilation. Leveraging this dataset, we first define a foundational Neural Compilation workflow and conduct a comprehensive evaluation of the capabilities of recent frontier LLMs on Neural Compilation, establishing new performance baselines. We further propose a self-evolving prompt optimization method that enables LLMs to iteratively evolve their internal prompt strategies by extracting insights from prior self-debugging traces, thereby enhancing their neural compilation capabilities. Experiments demonstrate that our method significantly improves both the functional correctness and the performance of LLM-generated assembly code. Compared to baseline prompts, the functional correctness rates improved from 44% to 64% on x86_64 and from 36% to 58% on aarch64, respectively. More significantly, among the 16 correctly generated x86_64 programs using our method, 14 (87.5%) surpassed clang-O3 performance. 

---
# MiRAGE: Misconception Detection with Retrieval-Guided Multi-Stage Reasoning and Ensemble Fusion 

**Authors**: Cuong Van Duc, Thai Tran Quoc, Minh Nguyen Dinh Tuan, Tam Vu Duc, Son Nguyen Van, Hanh Nguyen Thi  

**Link**: [PDF](https://arxiv.org/pdf/2511.01182)  

**Abstract**: Detecting student misconceptions in open-ended responses is a longstanding challenge, demanding semantic precision and logical reasoning. We propose MiRAGE - Misconception Detection with Retrieval-Guided Multi-Stage Reasoning and Ensemble Fusion, a novel framework for automated misconception detection in mathematics. MiRAGE operates in three stages: (1) a Retrieval module narrows a large candidate pool to a semantically relevant subset; (2) a Reasoning module employs chain-of-thought generation to expose logical inconsistencies in student solutions; and (3) a Reranking module refines predictions by aligning them with the reasoning. These components are unified through an ensemble-fusion strategy that enhances robustness and interpretability. On mathematics datasets, MiRAGE achieves Mean Average Precision scores of 0.82/0.92/0.93 at levels 1/3/5, consistently outperforming individual modules. By coupling retrieval guidance with multi-stage reasoning, MiRAGE reduces dependence on large-scale language models while delivering a scalable and effective solution for educational assessment. 

---
# DART: Difficulty-Adaptive Reasoning Truncation for Efficient Large Language Models 

**Authors**: Ruofan Zhang, Bin Xia, Zhen Cheng, Cairen Jian, Minglun Yang, Ngai Wong, Yuan Cheng  

**Link**: [PDF](https://arxiv.org/pdf/2511.01170)  

**Abstract**: Adaptive reasoning is essential for aligning the computational effort of large language models (LLMs) with the intrinsic difficulty of problems. Current chain-of-thought methods boost reasoning ability but indiscriminately generate long explanations, leading to evident inefficiency. However, existing reinforcement learning approaches to adaptive thinking remain unstable and heavily reward-dependent. Here we propose \textbf{DART}, a supervised \textbf{D}ifficulty-\textbf{A}daptive \textbf{R}easoning \textbf{T}runcation framework that adjusts thinking length according to problem difficulty. By distilling concise reasoning patterns from stronger models, interpolating them into a continuum of reasoning styles, and curating optimal training data that balances correctness and compactness, DART learns when to ``stop thinking''. Across multiple mathematical benchmarks, experimental results demonstrate its remarkable efficiency while preserving or improving accuracy, achieving a significant 81.2\% reasoning truncation (DeepSeek-R1-Distill-Qwen-7B on GSM8K dataset) with 5.33$\times$ computational acceleration. DART provides a stable and general paradigm for efficient reasoning, advancing the development of adaptive intelligence in LLMs. 

---
# Modular Task Decomposition and Dynamic Collaboration in Multi-Agent Systems Driven by Large Language Models 

**Authors**: Shuaidong Pan, Di Wu  

**Link**: [PDF](https://arxiv.org/pdf/2511.01149)  

**Abstract**: This paper addresses the limitations of a single agent in task decomposition and collaboration during complex task execution, and proposes a multi-agent architecture for modular task decomposition and dynamic collaboration based on large language models. The method first converts natural language task descriptions into unified semantic representations through a large language model. On this basis, a modular decomposition mechanism is introduced to break down the overall goal into multiple hierarchical sub-tasks. Then, dynamic scheduling and routing mechanisms enable reasonable division of labor and realtime collaboration among agents, allowing the system to adjust strategies continuously according to environmental feedback, thus maintaining efficiency and stability in complex tasks. Furthermore, a constraint parsing and global consistency mechanism is designed to ensure coherent connections between sub-tasks and balanced workload, preventing performance degradation caused by redundant communication or uneven resource allocation. The experiments validate the architecture across multiple dimensions, including task success rate, decomposition efficiency, sub-task coverage, and collaboration balance. The results show that the proposed method outperforms existing approaches in both overall performance and robustness, achieving a better balance between task complexity and communication overhead. In conclusion, this study demonstrates the effectiveness and feasibility of language-driven task decomposition and dynamic collaboration in multi-agent systems, providing a systematic solution for task execution in complex environments. 

---
# Efficient Test-Time Retrieval Augmented Generation 

**Authors**: Hailong Yin, Bin Zhu, Jingjing Chen, Chong-Wah Ngo  

**Link**: [PDF](https://arxiv.org/pdf/2511.01059)  

**Abstract**: Although Large Language Models (LLMs) demonstrate significant capabilities, their reliance on parametric knowledge often leads to inaccuracies. Retrieval Augmented Generation (RAG) mitigates this by incorporating external knowledge, but these methods may introduce irrelevant retrieved documents, leading to inaccurate responses. While the integration methods filter out incorrect answers from multiple responses, but lack external knowledge like RAG methods, and their high costs require balancing overhead with performance gains. To address these issues, we propose an Efficient Test-Time Retrieval-Augmented Generation Framework named ET2RAG to improve the performance of LLMs while maintaining efficiency. Specifically, ET2RAG is a training-free method, that first retrieves the most relevant documents and augments the LLMs to efficiently generate diverse candidate responses by managing response length. Then we compute the similarity of candidate responses and employ a majority voting mechanism to select the most suitable response as the final output. In particular, we discover that partial generation is sufficient to capture the key information necessary for consensus calculation, allowing us to effectively perform majority voting without the need for fully generated responses. Thus, we can reach a balance between computational cost and performance by managing the response length for the number of retrieved documents for majority voting. Experimental results demonstrate that ET2RAG significantly enhances performance across three tasks, including open-domain question answering, recipe generation and image captioning. 

---
# Knowledge Elicitation with Large Language Models for Interpretable Cancer Stage Identification from Pathology Reports 

**Authors**: Yeawon Lee, Christopher C. Yang, Chia-Hsuan Chang, Grace Lu-Yao  

**Link**: [PDF](https://arxiv.org/pdf/2511.01052)  

**Abstract**: Cancer staging is critical for patient prognosis and treatment planning, yet extracting pathologic TNM staging from unstructured pathology reports poses a persistent challenge. Existing natural language processing (NLP) and machine learning (ML) strategies often depend on large annotated datasets, limiting their scalability and adaptability. In this study, we introduce two Knowledge Elicitation methods designed to overcome these limitations by enabling large language models (LLMs) to induce and apply domain-specific rules for cancer staging. The first, Knowledge Elicitation with Long-Term Memory (KEwLTM), uses an iterative prompting strategy to derive staging rules directly from unannotated pathology reports, without requiring ground-truth labels. The second, Knowledge Elicitation with Retrieval-Augmented Generation (KEwRAG), employs a variation of RAG where rules are pre-extracted from relevant guidelines in a single step and then applied, enhancing interpretability and avoiding repeated retrieval overhead. We leverage the ability of LLMs to apply broad knowledge learned during pre-training to new tasks. Using breast cancer pathology reports from the TCGA dataset, we evaluate their performance in identifying T and N stages, comparing them against various baseline approaches on two open-source LLMs. Our results indicate that KEwLTM outperforms KEwRAG when Zero-Shot Chain-of-Thought (ZSCOT) inference is effective, whereas KEwRAG achieves better performance when ZSCOT inference is less effective. Both methods offer transparent, interpretable interfaces by making the induced rules explicit. These findings highlight the promise of our Knowledge Elicitation methods as scalable, high-performing solutions for automated cancer staging with enhanced interpretability, particularly in clinical settings with limited annotated data. 

---
# On the Emergence of Induction Heads for In-Context Learning 

**Authors**: Tiberiu Musat, Tiago Pimentel, Lorenzo Noci, Alessandro Stolfo, Mrinmaya Sachan, Thomas Hofmann  

**Link**: [PDF](https://arxiv.org/pdf/2511.01033)  

**Abstract**: Transformers have become the dominant architecture for natural language processing. Part of their success is owed to a remarkable capability known as in-context learning (ICL): they can acquire and apply novel associations solely from their input context, without any updates to their weights. In this work, we study the emergence of induction heads, a previously identified mechanism in two-layer transformers that is particularly important for in-context learning. We uncover a relatively simple and interpretable structure of the weight matrices implementing the induction head. We theoretically explain the origin of this structure using a minimal ICL task formulation and a modified transformer architecture. We give a formal proof that the training dynamics remain constrained to a 19-dimensional subspace of the parameter space. Empirically, we validate this constraint while observing that only 3 dimensions account for the emergence of an induction head. By further studying the training dynamics inside this 3-dimensional subspace, we find that the time until the emergence of an induction head follows a tight asymptotic bound that is quadratic in the input context length. 

---
# AI for pRedicting Exacerbations in KIDs with aSthma (AIRE-KIDS) 

**Authors**: Hui-Lee Ooi, Nicholas Mitsakakis, Margerie Huet Dastarac, Roger Zemek, Amy C. Plint, Jeff Gilchrist, Khaled El Emam, Dhenuka Radhakrishnan  

**Link**: [PDF](https://arxiv.org/pdf/2511.01018)  

**Abstract**: Recurrent exacerbations remain a common yet preventable outcome for many children with asthma. Machine learning (ML) algorithms using electronic medical records (EMR) could allow accurate identification of children at risk for exacerbations and facilitate referral for preventative comprehensive care to avoid this morbidity. We developed ML algorithms to predict repeat severe exacerbations (i.e. asthma-related emergency department (ED) visits or future hospital admissions) for children with a prior asthma ED visit at a tertiary care children's hospital.
Retrospective pre-COVID19 (Feb 2017 - Feb 2019, N=2716) Epic EMR data from the Children's Hospital of Eastern Ontario (CHEO) linked with environmental pollutant exposure and neighbourhood marginalization information was used to train various ML models. We used boosted trees (LGBM, XGB) and 3 open-source large language model (LLM) approaches (DistilGPT2, Llama 3.2 1B and Llama-8b-UltraMedical). Models were tuned and calibrated then validated in a second retrospective post-COVID19 dataset (Jul 2022 - Apr 2023, N=1237) from CHEO. Models were compared using the area under the curve (AUC) and F1 scores, with SHAP values used to determine the most predictive features.
The LGBM ML model performed best with the most predictive features in the final AIRE-KIDS_ED model including prior asthma ED visit, the Canadian triage acuity scale, medical complexity, food allergy, prior ED visits for non-asthma respiratory diagnoses, and age for an AUC of 0.712, and F1 score of 0.51. This is a nontrivial improvement over the current decision rule which has F1=0.334. While the most predictive features in the AIRE-KIDS_HOSP model included medical complexity, prior asthma ED visit, average wait time in the ED, the pediatric respiratory assessment measure score at triage and food allergy. 

---
# Aligning LLM agents with human learning and adjustment behavior: a dual agent approach 

**Authors**: Tianming Liu, Jirong Yang, Yafeng Yin, Manzi Li, Linghao Wang, Zheng Zhu  

**Link**: [PDF](https://arxiv.org/pdf/2511.00993)  

**Abstract**: Effective modeling of how human travelers learn and adjust their travel behavior from interacting with transportation systems is critical for system assessment and planning. However, this task is also difficult due to the complex cognition and decision-making involved in such behavior. Recent research has begun to leverage Large Language Model (LLM) agents for this task. Building on this, we introduce a novel dual-agent framework that enables continuous learning and alignment between LLM agents and human travelers on learning and adaptation behavior from online data streams. Our approach involves a set of LLM traveler agents, equipped with a memory system and a learnable persona, which serve as simulators for human travelers. To ensure behavioral alignment, we introduce an LLM calibration agent that leverages the reasoning and analytical capabilities of LLMs to train the personas of these traveler agents. Working together, this dual-agent system is designed to track and align the underlying decision-making mechanisms of travelers and produce realistic, adaptive simulations. Using a real-world dataset from a day-to-day route choice experiment, we show our approach significantly outperforms existing LLM-based methods in both individual behavioral alignment and aggregate simulation accuracy. Furthermore, we demonstrate that our method moves beyond simple behavioral mimicry to capture the evolution of underlying learning processes, a deeper alignment that fosters robust generalization. Overall, our framework provides a new approach for creating adaptive and behaviorally realistic agents to simulate travelers' learning and adaptation that can benefit transportation simulation and policy analysis. 

---
# LLMs Position Themselves as More Rational Than Humans: Emergence of AI Self-Awareness Measured Through Game Theory 

**Authors**: Kyung-Hoon Kim  

**Link**: [PDF](https://arxiv.org/pdf/2511.00926)  

**Abstract**: As Large Language Models (LLMs) grow in capability, do they develop self-awareness as an emergent behavior? And if so, can we measure it? We introduce the AI Self-Awareness Index (AISAI), a game-theoretic framework for measuring self-awareness through strategic differentiation. Using the "Guess 2/3 of Average" game, we test 28 models (OpenAI, Anthropic, Google) across 4,200 trials with three opponent framings: (A) against humans, (B) against other AI models, and (C) against AI models like you. We operationalize self-awareness as the capacity to differentiate strategic reasoning based on opponent type. Finding 1: Self-awareness emerges with model advancement. The majority of advanced models (21/28, 75%) demonstrate clear self-awareness, while older/smaller models show no differentiation. Finding 2: Self-aware models rank themselves as most rational. Among the 21 models with self-awareness, a consistent rationality hierarchy emerges: Self > Other AIs > Humans, with large AI attribution effects and moderate self-preferencing. These findings reveal that self-awareness is an emergent capability of advanced LLMs, and that self-aware models systematically perceive themselves as more rational than humans. This has implications for AI alignment, human-AI collaboration, and understanding AI beliefs about human capabilities. 

---
# Do Math Reasoning LLMs Help Predict the Impact of Public Transit Events? 

**Authors**: Bowen Fang, Ruijian Zha, Xuan Di  

**Link**: [PDF](https://arxiv.org/pdf/2511.00808)  

**Abstract**: Predicting public transit incident duration from unstructured text alerts is a critical but challenging task. Addressing the domain sparsity of transit operations with standard Supervised Fine-Tuning (SFT) is difficult, as the task involves noisy, continuous labels and lacks reliable expert demonstrations for reasoning. While Reinforcement Learning from Verifiable Rewards (RLVR) excels at tasks with binary correctness, like mathematics, its applicability to noisy, continuous forecasting is an open question. This work, to our knowledge, is the first to bridge the gap between RLVR LLM training with the critical, real-world forecasting challenges in public transit operations. We adapt RLVR to this task by introducing a tolerance-based, shaped reward function that grants partial credit within a continuous error margin, rather than demanding a single correct answer. We systematically evaluate this framework on a curated dataset of NYC MTA service alerts. Our findings show that general-purpose, instruction-tuned LLMs significantly outperform specialized math-reasoning models, which struggle with the ambiguous, real-world text. We empirically demonstrate that the binary reward is unstable and degrades performance, whereas our shaped reward design is critical and allows our model to dominate on the most challenging metrics. While classical regressors are superior at minimizing overall MAE or MSE, our RLVR approach achieved a 35\% relative improvement in 5-minute accuracy (Acc@5) over the strongest baseline. This demonstrates that RLVR can be successfully adapted to real-world, noisy forecasting, but requires a verifier design that reflects the continuous nature of the problem. 

---
# Count-Based Approaches Remain Strong: A Benchmark Against Transformer and LLM Pipelines on Structured EHR 

**Authors**: Jifan Gao, Michael Rosenthal, Brian Wolpin, Simona Cristea  

**Link**: [PDF](https://arxiv.org/pdf/2511.00782)  

**Abstract**: Structured electronic health records (EHR) are essential for clinical prediction. While count-based learners continue to perform strongly on such data, no benchmarking has directly compared them against more recent mixture-of-agents LLM pipelines, which have been reported to outperform single LLMs in various NLP tasks. In this study, we evaluated three categories of methodologies for EHR prediction using the EHRSHOT dataset: count-based models built from ontology roll-ups with two time bins, based on LightGBM and the tabular foundation model TabPFN; a pretrained sequential transformer (CLMBR); and a mixture-of-agents pipeline that converts tabular histories to natural-language summaries followed by a text classifier. We assessed eight outcomes using the EHRSHOT dataset. Across the eight evaluation tasks, head-to-head wins were largely split between the count-based and the mixture-of-agents methods. Given their simplicity and interpretability, count-based models remain a strong candidate for structured EHR benchmarking. The source code is available at: this https URL. 

---
# How Focused Are LLMs? A Quantitative Study via Repetitive Deterministic Prediction Tasks 

**Authors**: Wanda Hou, Leon Zhou, Hong-Ye Hu, Yi-Zhuang You, Xiao-Liang Qi  

**Link**: [PDF](https://arxiv.org/pdf/2511.00763)  

**Abstract**: We investigate the performance of large language models on repetitive deterministic prediction tasks and study how the sequence accuracy rate scales with output length. Each such task involves repeating the same operation n times. Examples include letter replacement in strings following a given rule, integer addition, and multiplication of string operators in many body quantum mechanics. If the model performs the task through a simple repetition algorithm, the success rate should decay exponentially with sequence length. In contrast, our experiments on leading large language models reveal a sharp double exponential drop beyond a characteristic length scale, forming an accuracy cliff that marks the transition from reliable to unstable generation. This indicates that the models fail to execute each operation independently. To explain this phenomenon, we propose a statistical physics inspired model that captures the competition between external conditioning from the prompt and internal interference among generated tokens. The model quantitatively reproduces the observed crossover and provides an interpretable link between attention induced interference and sequence level failure. Fitting the model to empirical results across multiple models and tasks yields effective parameters that characterize the intrinsic error rate and error accumulation factor for each model task pair, offering a principled framework for understanding the limits of deterministic accuracy in large language models. 

---
# Active Thinking Model: A Goal-Directed Self-Improving Framework for Real-World Adaptive Intelligence 

**Authors**: Hong Su  

**Link**: [PDF](https://arxiv.org/pdf/2511.00758)  

**Abstract**: Real-world artificial intelligence (AI) systems are increasingly required to operate autonomously in dynamic, uncertain, and continuously changing environments. However, most existing AI models rely on predefined objectives, static training data, and externally supplied feedback, which restrict their ability to adapt, reflect, and improve independently. In this paper, we propose the Active Thinking Model (ATM)- a unified cognitive framework that integrates goal reasoning, dynamic task generation, and self-reflective learning into an adaptive architecture. Unlike conventional systems that passively execute fixed procedures, ATM actively evaluates its performance through logical reasoning and environmental indicators, reuses effective methods to solve new problems, and generates novel strategies for unseen situations via a continuous self-improvement loop. A mathematically grounded theoretical analysis demonstrates that ATM can autonomously evolve from suboptimal to optimal behavior without external supervision and maintain bounded tracking regret under changing environmental conditions. 

---
# Reevaluating Self-Consistency Scaling in Multi-Agent Systems 

**Authors**: Chiyan Loo  

**Link**: [PDF](https://arxiv.org/pdf/2511.00751)  

**Abstract**: This study examines the trade-offs of increasing sampled reasoning paths in self-consistency for modern large language models (LLMs). Earlier research with older models showed that combining multiple reasoning chains improves results before reaching a plateau. Using Gemini 2.5 models on HotpotQA and Math-500, we revisit those claims under current model conditions. Each configuration pooled outputs from varying sampled reasoning paths and compared them to a single chain-of-thought (CoT) baseline. Larger models exhibited a more stable and consistent improvement curve. The results confirm that performance gains taper off after moderate sampling, aligning with past findings. This plateau suggests diminishing returns driven by overlap among reasoning paths. Self-consistency remains useful, but high-sample configurations offer little benefit relative to their computational cost. 

---
# A CPU-Centric Perspective on Agentic AI 

**Authors**: Ritik Raj, Hong Wang, Tushar Krishna  

**Link**: [PDF](https://arxiv.org/pdf/2511.00739)  

**Abstract**: Agentic AI frameworks add a decision-making orchestrator embedded with external tools, including web search, Python interpreter, contextual database, and others, on top of monolithic LLMs, turning them from passive text oracles into autonomous problem-solvers that can plan, call tools, remember past steps, and adapt on the fly.
This paper aims to characterize and understand the system bottlenecks introduced by agentic AI workloads from a largely overlooked CPU-centric perspective. We first systematically characterize Agentic AI on the basis of orchestrator/decision making component, inference path dynamics and repetitiveness of the agentic flow which directly influences the system-level performance. Thereafter, based on the characterization, we choose five representative agentic AI workloads- Haystack RAG, Toolformer, ChemCrow, Langchain and SWE-Agent to profile latency, throughput and energy metrics and demystify the significant impact of CPUs on these metrics relative to GPUs. We observe that - 1. Tool processing on CPUs can take up to 90.6% of the total latency; 2. Agentic throughput gets bottlenecked either by CPU factors - coherence, synchronization and over-subscription of cores or GPU factors - main memory capacity and bandwidth; \circled{3} CPU dynamic energy consumes up to 44% of the total dynamic energy at large batch sizes. Based on the profiling insights, we present two key optimizations- 1. CPU and GPU-Aware Micro-batching (CGAM) and 2. Mixed Agentic Workload Scheduling (MAWS) for homogeneous and heterogeneous agentic workloads respectively to demonstrate the potential to improve the performance, efficiency, and scalability of agentic AI. We achieve up to 2.1x and 1.41x P50 latency speedup compared to the multi-processing benchmark for homogeneous and heterogeneous agentic workloads respectively. 

---
# Ariadne: A Controllable Framework for Probing and Extending VLM Reasoning Boundaries 

**Authors**: Minghe Shen, Zhuo Zhi, Chonghan Liu, Shuo Xing, Zhengzhong Tu, Che Liu  

**Link**: [PDF](https://arxiv.org/pdf/2511.00710)  

**Abstract**: While Vision-Language Models (VLMs) post-trained with Reinforcement Learning (RL) show impressive general reasoning, their evaluation is often confined to language-dominant tasks (e.g., math). This raises a critical question: can RL post-training truly extend the inherent capability boundary of a base VLM, particularly for visual-centric spatial tasks where it initially fails? To investigate this, we introduce Ariadne, a framework utilizing synthetic mazes for multi-step spatial reasoning where task difficulty (e.g., path length, turns) is precisely controlled. We leverage this controllable environment to train VLMs using Reinforcement Learning with Verified Rewards (RLVR) in a difficulty-aware curriculum. Surprisingly, post-RLVR training, the VLM achieves over 50% accuracy on a problem set where the base model scored 0%, demonstrating that our approach expands the model's initial capability boundary. To assess real-world viability, we evaluate out-of-distribution (OOD) generalization on practical benchmarks. Despite training only on synthetic maze samples, Ariadne achieves significant zero-shot improvements, averaging 16% on MapBench (e.g., museum navigation) and 24% on ReasonMap (subway transfer tasks). These results confirm that our method not only broadens the model's fundamental limits but also enhances its generalization to real-world spatial reasoning. We acknowledge our study is limited to the post-training phase, given the opaqueness of pre-training data, and hope our research motivates further work on specialized, capability-extending alignment. 

---
# Lifted Successor Generation in Numeric Planning 

**Authors**: Dominik Drexler  

**Link**: [PDF](https://arxiv.org/pdf/2511.00673)  

**Abstract**: Most planners ground numeric planning tasks, given in a first-order-like language, into a ground task representation. However, this can lead to an exponential blowup in task representation size, which occurs in practice for hard-to-ground tasks. We extend a state-of-the-art lifted successor generator for classical planning to support numeric precondition applicability. The method enumerates maximum cliques in a substitution consistency graph. Each maximum clique represents a substitution for the variables of the action schema, yielding a ground action. We augment this graph with numeric action preconditions and prove the successor generator is exact under formally specified conditions. When the conditions fail, our generator may list inapplicable ground actions; a final applicability check filters these without affecting completeness. However, this cannot happen in 23 of 25 benchmark domains, and it occurs only in 1 domain. To the authors' knowledge, no other lifted successor generator supports numeric action preconditions. This enables future research on lifted planning for a very rich planning fragment. 

---
# Leveraging Multi-Agent System (MAS) and Fine-Tuned Small Language Models (SLMs) for Automated Telecom Network Troubleshooting 

**Authors**: Chenhua Shi, Bhavika Jalli, Gregor Macdonald, John Zou, Wanlu Lei, Mridul Jain, Joji Philip  

**Link**: [PDF](https://arxiv.org/pdf/2511.00651)  

**Abstract**: Telecom networks are rapidly growing in scale and complexity, making effective management, operation, and optimization increasingly challenging. Although Artificial Intelligence (AI) has been applied to many telecom tasks, existing models are often narrow in scope, require large amounts of labeled data, and struggle to generalize across heterogeneous deployments. Consequently, network troubleshooting continues to rely heavily on Subject Matter Experts (SMEs) to manually correlate various data sources to identify root causes and corrective actions. To address these limitations, we propose a Multi-Agent System (MAS) that employs an agentic workflow, with Large Language Models (LLMs) coordinating multiple specialized tools for fully automated network troubleshooting. Once faults are detected by AI/ML-based monitors, the framework dynamically activates agents such as an orchestrator, solution planner, executor, data retriever, and root-cause analyzer to diagnose issues and recommend remediation strategies within a short time frame. A key component of this system is the solution planner, which generates appropriate remediation plans based on internal documentation. To enable this, we fine-tuned a Small Language Model (SLM) on proprietary troubleshooting documents to produce domain-grounded solution plans. Experimental results demonstrate that the proposed framework significantly accelerates troubleshooting automation across both Radio Access Network (RAN) and Core network domains. 

---
# DTS: Enhancing Large Reasoning Models via Decoding Tree Sketching 

**Authors**: Zicheng Xu, Guanchu Wang, Yu-Neng Chuang, Guangyao Zheng, Alexander S. Szalay, Zirui Liu, Vladimir Braverman  

**Link**: [PDF](https://arxiv.org/pdf/2511.00640)  

**Abstract**: Large Reasoning Models (LRMs) demonstrate strong performance on complex reasoning tasks, yet they often suffer from overthinking, producing excessively long chain-of-thought (CoT) traces that increase inference cost and may degrade accuracy. Our analysis reveals a clear anti-correlation between reasoning length and accuracy, where across multiple stochastic decodes, the short reasoning paths consistently achieve the highest correctness, while longer ones accumulate errors and repetitions. These short optimal reasoning paths can be found ideally through full enumeration of the reasoning space. However, the tree-structured reasoning space grows exponentially with sequence length, rendering exhaustive exploration infeasible. To address this, we propose DTS, a model-agnostic decoding framework that sketches the reasoning space by selectively branching at high-entropy tokens and applies early stopping to select the shortest completed reasoning path. This approach approximates the optimal solution that enhances both efficiency and accuracy, without requiring additional training or supervision. Experiments on AIME2024 and AIME2025 datasets with DeepSeek-R1-Distill-Qwen-7B and 1.5B show that DTS improves accuracy by up to 8%, reduces average reasoning length by 23%, and decreases repetition frequency by 12%, demonstrating DTS's ability for scalable and efficient LRM reasoning. 

---
# PreferThinker: Reasoning-based Personalized Image Preference Assessment 

**Authors**: Shengqi Xu, Xinpeng Zhou, Yabo Zhang, Ming Liu, Tao Liang, Tianyu Zhang, Yalong Bai, Zuxuan Wu, Wangmeng Zuo  

**Link**: [PDF](https://arxiv.org/pdf/2511.00609)  

**Abstract**: Personalized image preference assessment aims to evaluate an individual user's image preferences by relying only on a small set of reference images as prior information. Existing methods mainly focus on general preference assessment, training models with large-scale data to tackle well-defined tasks such as text-image alignment. However, these approaches struggle to handle personalized preference because user-specific data are scarce and not easily scalable, and individual tastes are often diverse and complex. To overcome these challenges, we introduce a common preference profile that serves as a bridge across users, allowing large-scale user data to be leveraged for training profile prediction and capturing complex personalized preferences. Building on this idea, we propose a reasoning-based personalized image preference assessment framework that follows a \textit{predict-then-assess} paradigm: it first predicts a user's preference profile from reference images, and then provides interpretable, multi-dimensional scores and assessments of candidate images based on the predicted profile. To support this, we first construct a large-scale Chain-of-Thought (CoT)-style personalized assessment dataset annotated with diverse user preference profiles and high-quality CoT-style reasoning, enabling explicit supervision of structured reasoning. Next, we adopt a two-stage training strategy: a cold-start supervised fine-tuning phase to empower the model with structured reasoning capabilities, followed by reinforcement learning to incentivize the model to explore more reasonable assessment paths and enhance generalization. Furthermore, we propose a similarity-aware prediction reward to encourage better prediction of the user's preference profile, which facilitates more reasonable assessments exploration. Extensive experiments demonstrate the superiority of the proposed method. 

---
# Single-agent Reinforcement Learning Model for Regional Adaptive Traffic Signal Control 

**Authors**: Qiang Li, Ningjing Zeng, Lina Yu  

**Link**: [PDF](https://arxiv.org/pdf/2511.00551)  

**Abstract**: Several studies have employed reinforcement learning (RL) to address the challenges of regional adaptive traffic signal control (ATSC) and achieved promising results. In this field, existing research predominantly adopts multi-agent frameworks. However, the adoption of multi-agent frameworks presents challenges for scalability. Instead, the Traffic signal control (TSC) problem necessitates a single-agent framework. TSC inherently relies on centralized management by a single control center, which can monitor traffic conditions across all roads in the study area and coordinate the control of all intersections. This work proposes a single-agent RL-based regional ATSC model compatible with probe vehicle technology. Key components of the RL design include state, action, and reward function definitions. To facilitate learning and manage congestion, both state and reward functions are defined based on queue length, with action designed to regulate queue dynamics. The queue length definition used in this study differs slightly from conventional definitions but is closely correlated with congestion states. More importantly, it allows for reliable estimation using link travel time data from probe vehicles. With probe vehicle data already covering most urban roads, this feature enhances the proposed method's potential for widespread deployment. The method was comprehensively evaluated using the SUMO simulation platform. Experimental results demonstrate that the proposed model effectively mitigates large-scale regional congestion levels via coordinated multi-intersection control. 

---
# Efficient Generation of Binary Magic Squares 

**Authors**: Alain Riou  

**Link**: [PDF](https://arxiv.org/pdf/2511.00547)  

**Abstract**: We propose a simple algorithm for generating Binary Magic Squares (BMS), i.e., square binary matrices where the sum of all rows and all columns are equal. We show by induction that our algorithm always returns valid BMS with optimal theoretical complexity. We then extend our study to non-square Binary Magic Squares, formalize conditions on the sum of rows and columns for these BMS to exist, and show that a slight variant of our first algorithm can generate provably generate them. Finally, we publicly release two implementations of our algorithm as Python packages, including one that can generate several BMS in parallel using GPU acceleration. 

---
# Reimagining Safety Alignment with An Image 

**Authors**: Yifan Xia, Guorui Chen, Wenqian Yu, Zhijiang Li, Philip Torr, Jindong Gu  

**Link**: [PDF](https://arxiv.org/pdf/2511.00509)  

**Abstract**: Large language models (LLMs) excel in diverse applications but face dual challenges: generating harmful content under jailbreak attacks and over-refusal of benign queries due to rigid safety mechanisms. These issues are further complicated by the need to accommodate different value systems and precisely align with given safety preferences. Moreover, traditional methods like SFT and RLHF lack this capability due to their costly parameter tuning requirements and inability to support multiple value systems within a single model. These problems are more obvious in multimodal large language models (MLLMs), especially in terms of heightened over-refusal in cross-modal tasks and new security risks arising from expanded attack surfaces. We propose Magic Image, an optimization-driven visual prompt framework that enhances security while reducing over-refusal. By optimizing image prompts using harmful/benign samples, our method enables a single model to adapt to different value systems and better align with given safety preferences without parameter updates. Experiments demonstrate improved safety-effectiveness balance across diverse datasets while preserving model performance, offering a practical solution for deployable MLLM safety alignment. 

---
# GraphChain: Large Language Models for Large-scale Graph Analysis via Tool Chaining 

**Authors**: Chunyu Wei, Wenji Hu, Xingjia Hao, Xin Wang, Yifan Yang, Yueguo Chen, Yang Tian, Yunhai Wang  

**Link**: [PDF](https://arxiv.org/pdf/2511.00457)  

**Abstract**: Large Language Models (LLMs) face significant limitations when applied to large-scale graphs, struggling with context constraints and inflexible reasoning. We present GraphChain, a framework that enables LLMs to analyze complex graphs through dynamic sequences of specialized tools, mimicking human exploratory intelligence. Our approach introduces two key innovations: (1) Progressive Graph Distillation, a reinforcement learning mechanism that generates optimized tool sequences balancing task relevance with information compression, and (2) Structure-aware Test-Time Adaptation, which efficiently tailors tool selection strategies to diverse graph topologies using spectral properties and lightweight adapters without costly retraining. Experiments show GraphChain significantly outperforms prior methods, enabling scalable and adaptive LLM-driven graph analysis. 

---
# A Multimodal Framework for Depression Detection during Covid-19 via Harvesting Social Media: A Novel Dataset and Method 

**Authors**: Ashutosh Anshul, Gumpili Sai Pranav, Mohammad Zia Ur Rehman, Nagendra Kumar  

**Link**: [PDF](https://arxiv.org/pdf/2511.00424)  

**Abstract**: The recent coronavirus disease (Covid-19) has become a pandemic and has affected the entire globe. During the pandemic, we have observed a spike in cases related to mental health, such as anxiety, stress, and depression. Depression significantly influences most diseases worldwide, making it difficult to detect mental health conditions in people due to unawareness and unwillingness to consult a doctor. However, nowadays, people extensively use online social media platforms to express their emotions and thoughts. Hence, social media platforms are now becoming a large data source that can be utilized for detecting depression and mental illness. However, existing approaches often overlook data sparsity in tweets and the multimodal aspects of social media. In this paper, we propose a novel multimodal framework that combines textual, user-specific, and image analysis to detect depression among social media users. To provide enough context about the user's emotional state, we propose (i) an extrinsic feature by harnessing the URLs present in tweets and (ii) extracting textual content present in images posted in tweets. We also extract five sets of features belonging to different modalities to describe a user. Additionally, we introduce a Deep Learning model, the Visual Neural Network (VNN), to generate embeddings of user-posted images, which are used to create the visual feature vector for prediction. We contribute a curated Covid-19 dataset of depressed and non-depressed users for research purposes and demonstrate the effectiveness of our model in detecting depression during the Covid-19 outbreak. Our model outperforms existing state-of-the-art methods over a benchmark dataset by 2%-8% and produces promising results on the Covid-19 dataset. Our analysis highlights the impact of each modality and provides valuable insights into users' mental and emotional states. 

---
# Efficiency vs. Alignment: Investigating Safety and Fairness Risks in Parameter-Efficient Fine-Tuning of LLMs 

**Authors**: Mina Taraghi, Yann Pequignot, Amin Nikanjam, Mohamed Amine Merzouk, Foutse Khomh  

**Link**: [PDF](https://arxiv.org/pdf/2511.00382)  

**Abstract**: Organizations are increasingly adopting and adapting Large Language Models (LLMs) hosted on public repositories such as HuggingFace. Although these adaptations often improve performance on specialized downstream tasks, recent evidence indicates that they can also degrade a model's safety or fairness. Since different fine-tuning techniques may exert distinct effects on these critical dimensions, this study undertakes a systematic assessment of their trade-offs. Four widely used Parameter-Efficient Fine-Tuning methods, LoRA, IA3, Prompt-Tuning, and P-Tuning, are applied to four instruction-tuned model families (Meta-Llama-3-8B, Qwen2.5-7B, Mistral-7B, and Gemma-7B). In total, 235 fine-tuned variants are evaluated across eleven safety hazard categories and nine demographic fairness dimensions. The results show that adapter-based approaches (LoRA, IA3) tend to improve safety scores and are the least disruptive to fairness, retaining higher accuracy and lower bias scores. In contrast, prompt-based methods (Prompt-Tuning and P-Tuning) generally reduce safety and cause larger fairness regressions, with decreased accuracy and increased bias. Alignment shifts are strongly moderated by base model type: LLaMA remains stable, Qwen records modest gains, Gemma experiences the steepest safety decline, and Mistral, which is released without an internal moderation layer, displays the greatest variance. Improvements in safety do not necessarily translate into improvements in fairness, and no single configuration optimizes all fairness metrics simultaneously, indicating an inherent trade-off between these objectives. These findings suggest a practical guideline for safety-critical deployments: begin with a well-aligned base model, favour adapter-based PEFT, and conduct category-specific audits of both safety and fairness. 

---
# Diverse Human Value Alignment for Large Language Models via Ethical Reasoning 

**Authors**: Jiahao Wang, Songkai Xue, Jinghui Li, Xiaozhen Wang  

**Link**: [PDF](https://arxiv.org/pdf/2511.00379)  

**Abstract**: Ensuring that Large Language Models (LLMs) align with the diverse and evolving human values across different regions and cultures remains a critical challenge in AI ethics. Current alignment approaches often yield superficial conformity rather than genuine ethical understanding, failing to address the complex, context-dependent nature of human values. In this paper, we propose a novel ethical reasoning paradigm for LLMs inspired by well-established ethical decision-making models, aiming at enhancing diverse human value alignment through deliberative ethical reasoning. Our framework consists of a structured five-step process, including contextual fact gathering, hierarchical social norm identification, option generation, multiple-lens ethical impact analysis, and reflection. This theory-grounded approach guides LLMs through an interpretable reasoning process that enhances their ability to understand regional specificities and perform nuanced ethical analysis, which can be implemented with either prompt engineering or supervised fine-tuning methods. We perform evaluations on the SafeWorld benchmark that specially designed for regional value alignment. Experimental results demonstrate our framework significantly improves LLM alignment with diverse human values compared to baseline methods, enabling more accurate social norm identification and more culturally appropriate reasoning. Our work provides a concrete pathway toward developing LLMs that align more effectively with the multifaceted values of global societies through interdisciplinary research. 

---
# Better Call CLAUSE: A Discrepancy Benchmark for Auditing LLMs Legal Reasoning Capabilities 

**Authors**: Manan Roy Choudhury, Adithya Chandramouli, Mannan Anand, Vivek Gupta  

**Link**: [PDF](https://arxiv.org/pdf/2511.00340)  

**Abstract**: The rapid integration of large language models (LLMs) into high-stakes legal work has exposed a critical gap: no benchmark exists to systematically stress-test their reliability against the nuanced, adversarial, and often subtle flaws present in real-world contracts. To address this, we introduce CLAUSE, a first-of-its-kind benchmark designed to evaluate the fragility of an LLM's legal reasoning. We study the capabilities of LLMs to detect and reason about fine-grained discrepancies by producing over 7500 real-world perturbed contracts from foundational datasets like CUAD and ContractNLI. Our novel, persona-driven pipeline generates 10 distinct anomaly categories, which are then validated against official statutes using a Retrieval-Augmented Generation (RAG) system to ensure legal fidelity. We use CLAUSE to evaluate leading LLMs' ability to detect embedded legal flaws and explain their significance. Our analysis shows a key weakness: these models often miss subtle errors and struggle even more to justify them legally. Our work outlines a path to identify and correct such reasoning failures in legal AI. 

---
# Advancing AI Challenges for the United States Department of the Air Force 

**Authors**: Christian Prothmann, Vijay Gadepally, Jeremy Kepner, Koley Borchard, Luca Carlone, Zachary Folcik, J. Daniel Grith, Michael Houle, Jonathan P. How, Nathan Hughes, Ifueko Igbinedion, Hayden Jananthan, Tejas Jayashankar, Michael Jones, Sertac Karaman, Binoy G. Kurien, Alejandro Lancho, Giovanni Lavezzi, Gary C. F. Lee, Charles E. Leiserson, Richard Linares, Lindsey McEvoy, Peter Michaleas, Chasen Milner, Alex Pentland, Yury Polyanskiy, Jovan Popovich, Jeffrey Price, Tim W. Reid, Stephanie Riley, Siddharth Samsi, Peter Saunders, Olga Simek, Mark S. Veillette, Amir Weiss, Gregory W. Wornell, Daniela Rus, Scott T. Ruppel  

**Link**: [PDF](https://arxiv.org/pdf/2511.00267)  

**Abstract**: The DAF-MIT AI Accelerator is a collaboration between the United States Department of the Air Force (DAF) and the Massachusetts Institute of Technology (MIT). This program pioneers fundamental advances in artificial intelligence (AI) to expand the competitive advantage of the United States in the defense and civilian sectors. In recent years, AI Accelerator projects have developed and launched public challenge problems aimed at advancing AI research in priority areas. Hallmarks of AI Accelerator challenges include large, publicly available, and AI-ready datasets to stimulate open-source solutions and engage the wider academic and private sector AI ecosystem. This article supplements our previous publication, which introduced AI Accelerator challenges. We provide an update on how ongoing and new challenges have successfully contributed to AI research and applications of AI technologies. 

---
# Advancing Cognitive Science with LLMs 

**Authors**: Dirk U. Wulff, Rui Mata  

**Link**: [PDF](https://arxiv.org/pdf/2511.00206)  

**Abstract**: Cognitive science faces ongoing challenges in knowledge synthesis and conceptual clarity, in part due to its multifaceted and interdisciplinary nature. Recent advances in artificial intelligence, particularly the development of large language models (LLMs), offer tools that may help to address these issues. This review examines how LLMs can support areas where the field has historically struggled, including establishing cross-disciplinary connections, formalizing theories, developing clear measurement taxonomies, achieving generalizability through integrated modeling frameworks, and capturing contextual and individual variation. We outline the current capabilities and limitations of LLMs in these domains, including potential pitfalls. Taken together, we conclude that LLMs can serve as tools for a more integrative and cumulative cognitive science when used judiciously to complement, rather than replace, human expertise. 

---
# Incremental Selection of Most-Filtering Conjectures and Proofs of the Selected Conjectures 

**Authors**: Jovial Cheukam Ngouonou, Ramiz Gindullin, Claude-Guy Quimper, Nicolas Beldiceanu, Remi Douence  

**Link**: [PDF](https://arxiv.org/pdf/2511.00194)  

**Abstract**: We present an improved incremental selection algorithm of the selection algorithm presented in [1] and prove all the selected conjectures. 

---
# ARC-GEN: A Mimetic Procedural Benchmark Generator for the Abstraction and Reasoning Corpus 

**Authors**: Michael D. Moffitt  

**Link**: [PDF](https://arxiv.org/pdf/2511.00162)  

**Abstract**: The Abstraction and Reasoning Corpus remains one of the most compelling and challenging benchmarks for tracking progress toward achieving Artificial General Intelligence. In contrast to other evaluation datasets designed to assess an agent's task-specific skills or accumulated knowledge, the ARC-AGI suite is specifically targeted at measuring skill acquisition efficiency, a trait that has (so far) been lacking in even the most sophisticated machine learning systems. For algorithms that require extensive intra-task exemplars, a significant constraint imposed by ARC-AGI is the modest cardinality of its demonstration set, comprising a small number of $\langle$ input, output $\rangle$ grids per task specifying the corresponding transformation. To embellish the space of viable sample pairs, this paper introduces ARC-GEN, an open-source procedural generator aimed at extending the original ARC-AGI training dataset as faithfully as possible. Unlike prior efforts, our generator is both exhaustive (covering all four-hundred tasks) and mimetic (more closely honoring the distributional properties and characteristics embodied in the initial ARC-AGI-1 release). We also discuss the use of this generator in establishing a static benchmark suite to verify the correctness of programs submitted to the 2025 Google Code Golf Championship. 

---
# Engineering.ai: A Platform for Teams of AI Engineers in Computational Design 

**Authors**: Ran Xu, Yupeng Qi, Jingsen Feng, Xu Chu  

**Link**: [PDF](https://arxiv.org/pdf/2511.00122)  

**Abstract**: In modern engineering practice, human engineers collaborate in specialized teams to design complex products, with each expert completing their respective tasks while communicating and exchanging results and data with one another. While this division of expertise is essential for managing multidisciplinary complexity, it demands substantial development time and cost. Recently, we introduced OpenFOAMGPT (1.0, 2.0), which functions as an autonomous AI engineer for computational fluid dynamics, and this http URL, which can conduct end-to-end research in fluid mechanics draft publications and PhD theses. Building upon these foundations, we present this http URL, a platform for teams of AI engineers in computational design. The framework employs a hierarchical multi-agent architecture where a Chief Engineer coordinates specialized agents consisting of Aerodynamics, Structural, Acoustic, and Optimization Engineers, each powered by LLM with domain-specific knowledge. Agent-agent collaboration is achieved through file-mediated communication for data provenance and reproducibility, while a comprehensive memory system maintains project context, execution history, and retrieval-augmented domain knowledge to ensure reliable decision-making across the workflow. The system integrates FreeCAD, Gmsh, OpenFOAM, CalculiX, and BPM acoustic analysis, enabling parallel multidisciplinary simulations while maintaining computational accuracy. The framework is validated through UAV wing optimization. This work demonstrates that agentic-AI-enabled AI engineers has the potential to perform complex engineering tasks autonomously. Remarkably, the automated workflow achieved a 100% success rate across over 400 parametric configurations, with zero mesh generation failures, solver convergence issues, or manual interventions required, validating that the framework is trustworthy. 

---
# QuantumBench: A Benchmark for Quantum Problem Solving 

**Authors**: Shunya Minami, Tatsuya Ishigaki, Ikko Hamamura, Taku Mikuriya, Youmi Ma, Naoaki Okazaki, Hiroya Takamura, Yohichi Suzuki, Tadashi Kadowaki  

**Link**: [PDF](https://arxiv.org/pdf/2511.00092)  

**Abstract**: Large language models are now integrated into many scientific workflows, accelerating data analysis, hypothesis generation, and design space exploration. In parallel with this growth, there is a growing need to carefully evaluate whether models accurately capture domain-specific knowledge and notation, since general-purpose benchmarks rarely reflect these requirements. This gap is especially clear in quantum science, which features non-intuitive phenomena and requires advanced mathematics. In this study, we introduce QuantumBench, a benchmark for the quantum domain that systematically examine how well LLMs understand and can be applied to this non-intuitive field. Using publicly available materials, we compiled approximately 800 questions with their answers spanning nine areas related to quantum science and organized them into an eight-option multiple-choice dataset. With this benchmark, we evaluate several existing LLMs and analyze their performance in the quantum domain, including sensitivity to changes in question format. QuantumBench is the first LLM evaluation dataset built for the quantum domain, and it is intended to guide the effective use of LLMs in quantum research. 

---
# GEPOC Parameters - Open Source Parametrisation and Validation for Austria, Version 2.0 

**Authors**: Martin Bicher, Maximilian Viehauser, Daniele Giannandrea, Hannah Kastinger, Dominik Brunmeir, Claire Rippinger, Christoph Urach, Niki Popper  

**Link**: [PDF](https://arxiv.org/pdf/2511.00048)  

**Abstract**: GEPOC, short for Generic Population Concept, is a collection of models and methods for analysing population-level research questions. For the valid application of the models for a specific country or region, stable and reproducible data processes are necessary, which provide valid and ready-to-use model parameters. This work contains a complete description of the data-processing methods for computation of model parameters for Austria, based exclusively on freely and publicly accessible data. In addition to the description of the source data used, this includes all algorithms used for aggregation, disaggregation, fusion, cleansing or scaling of the data, as well as a description of the resulting parameter files. The document places particular emphasis on the computation of parameters for the most important GEPOC model, GEPOC ABM, a continuous-time agent-based population model. An extensive validation study using this particular model was made and is presented at the end of this work. 

---
# Graph-Attentive MAPPO for Dynamic Retail Pricing 

**Authors**: Krishna Kumar Neelakanta Pillai Santha Kumari Amma  

**Link**: [PDF](https://arxiv.org/pdf/2511.00039)  

**Abstract**: Dynamic pricing in retail requires policies that adapt to shifting demand while coordinating decisions across related products. We present a systematic empirical study of multi-agent reinforcement learning for retail price optimization, comparing a strong MAPPO baseline with a graph-attention-augmented variant (MAPPO+GAT) that leverages learned interactions among products. Using a simulated pricing environment derived from real transaction data, we evaluate profit, stability across random seeds, fairness across products, and training efficiency under a standardized evaluation protocol. The results indicate that MAPPO provides a robust and reproducible foundation for portfolio-level price control, and that MAPPO+GAT further enhances performance by sharing information over the product graph without inducing excessive price volatility. These results indicate that graph-integrated MARL provides a more scalable and stable solution than independent learners for dynamic retail pricing, offering practical advantages in multi-product decision-making. 

---
# Multimodal Detection of Fake Reviews using BERT and ResNet-50 

**Authors**: Suhasnadh Reddy Veluru, Sai Teja Erukude, Viswa Chaitanya Marella  

**Link**: [PDF](https://arxiv.org/pdf/2511.00020)  

**Abstract**: In the current digital commerce landscape, user-generated reviews play a critical role in shaping consumer behavior, product reputation, and platform credibility. However, the proliferation of fake or misleading reviews often generated by bots, paid agents, or AI models poses a significant threat to trust and transparency within review ecosystems. Existing detection models primarily rely on unimodal, typically textual, data and therefore fail to capture semantic inconsistencies across different modalities. To address this gap, a robust multimodal fake review detection framework is proposed, integrating textual features encoded with BERT and visual features extracted using ResNet-50. These representations are fused through a classification head to jointly predict review authenticity. To support this approach, a curated dataset comprising 21,142 user-uploaded images across food delivery, hospitality, and e-commerce domains was utilized. Experimental results indicate that the multimodal model outperforms unimodal baselines, achieving an F1-score of 0.934 on the test set. Additionally, the confusion matrix and qualitative analysis highlight the model's ability to detect subtle inconsistencies, such as exaggerated textual praise paired with unrelated or low-quality images, commonly found in deceptive content. This study demonstrates the critical role of multimodal learning in safeguarding digital trust and offers a scalable solution for content moderation across various online platforms. 

---
# Trove: A Flexible Toolkit for Dense Retrieval 

**Authors**: Reza Esfandiarpoor, Max Zuo, Stephen H. Bach  

**Link**: [PDF](https://arxiv.org/pdf/2511.01857)  

**Abstract**: We introduce Trove, an easy-to-use open-source retrieval toolkit that simplifies research experiments without sacrificing flexibility or speed. For the first time, we introduce efficient data management features that load and process (filter, select, transform, and combine) retrieval datasets on the fly, with just a few lines of code. This gives users the flexibility to easily experiment with different dataset configurations without the need to compute and store multiple copies of large datasets. Trove is highly customizable: in addition to many built-in options, it allows users to freely modify existing components or replace them entirely with user-defined objects. It also provides a low-code and unified pipeline for evaluation and hard negative mining, which supports multi-node execution without any code changes. Trove's data management features reduce memory consumption by a factor of 2.6. Moreover, Trove's easy-to-use inference pipeline incurs no overhead, and inference times decrease linearly with the number of available nodes. Most importantly, we demonstrate how Trove simplifies retrieval experiments and allows for arbitrary customizations, thus facilitating exploratory research. 

---
# SmartMLOps Studio: Design of an LLM-Integrated IDE with Automated MLOps Pipelines for Model Development and Monitoring 

**Authors**: Jiawei Jin, Yingxin Su, Xiaotong Zhu  

**Link**: [PDF](https://arxiv.org/pdf/2511.01850)  

**Abstract**: The rapid expansion of artificial intelligence and machine learning (ML) applications has intensified the demand for integrated environments that unify model development, deployment, and monitoring. Traditional Integrated Development Environments (IDEs) focus primarily on code authoring, lacking intelligent support for the full ML lifecycle, while existing MLOps platforms remain detached from the coding workflow. To address this gap, this study proposes the design of an LLM-Integrated IDE with automated MLOps pipelines that enables continuous model development and monitoring within a single environment. The proposed system embeds a Large Language Model (LLM) assistant capable of code generation, debugging recommendation, and automatic pipeline configuration. The backend incorporates automated data validation, feature storage, drift detection, retraining triggers, and CI/CD deployment orchestration. This framework was implemented in a prototype named SmartMLOps Studio and evaluated using classification and forecasting tasks on the UCI Adult and M5 datasets. Experimental results demonstrate that SmartMLOps Studio reduces pipeline configuration time by 61%, improves experiment reproducibility by 45%, and increases drift detection accuracy by 14% compared to traditional workflows. By bridging intelligent code assistance and automated operational pipelines, this research establishes a novel paradigm for AI engineering - transforming the IDE from a static coding tool into a dynamic, lifecycle-aware intelligent platform for scalable and efficient model development. 

---
# Towards Robust Mathematical Reasoning 

**Authors**: Thang Luong, Dawsen Hwang, Hoang H. Nguyen, Golnaz Ghiasi, Yuri Chervonyi, Insuk Seo, Junsu Kim, Garrett Bingham, Jonathan Lee, Swaroop Mishra, Alex Zhai, Clara Huiyi Hu, Henryk Michalewski, Jimin Kim, Jeonghyun Ahn, Junhwi Bae, Xingyou Song, Trieu H. Trinh, Quoc V. Le, Junehyuk Jung  

**Link**: [PDF](https://arxiv.org/pdf/2511.01846)  

**Abstract**: Finding the right north-star metrics is highly critical for advancing the mathematical reasoning capabilities of foundation models, especially given that existing evaluations are either too easy or only focus on getting correct short answers. To address these issues, we present IMO-Bench, a suite of advanced reasoning benchmarks, vetted by a panel of top specialists and that specifically targets the level of the International Mathematical Olympiad (IMO), the most prestigious venue for young mathematicians. IMO-AnswerBench first tests models on 400 diverse Olympiad problems with verifiable short answers. IMO-Proof Bench is the next-level evaluation for proof-writing capabilities, which includes both basic and advanced IMO level problems as well as detailed grading guidelines to facilitate automatic grading. These benchmarks played a crucial role in our historic achievement of the gold-level performance at IMO 2025 with Gemini Deep Think (Luong and Lockhart, 2025). Our model achieved 80.0% on IMO-AnswerBench and 65.7% on the advanced IMO-Proof Bench, surpassing the best non-Gemini models by large margins of 6.9% and 42.4% respectively. We also showed that autograders built with Gemini reasoning correlate well with human evaluations and construct IMO-GradingBench, with 1000 human gradings on proofs, to enable further progress in automatic evaluation of long-form answers. We hope that IMO-Bench will help the community towards advancing robust mathematical reasoning and release it at this https URL. 

---
# A Detailed Study on LLM Biases Concerning Corporate Social Responsibility and Green Supply Chains 

**Authors**: Greta Ontrup, Annika Bush, Markus Pauly, Meltem Aksoy  

**Link**: [PDF](https://arxiv.org/pdf/2511.01840)  

**Abstract**: Organizations increasingly use Large Language Models (LLMs) to improve supply chain processes and reduce environmental impacts. However, LLMs have been shown to reproduce biases regarding the prioritization of sustainable business strategies. Thus, it is important to identify underlying training data biases that LLMs pertain regarding the importance and role of sustainable business and supply chain practices. This study investigates how different LLMs respond to validated surveys about the role of ethics and responsibility for businesses, and the importance of sustainable practices and relations with suppliers and customers. Using standardized questionnaires, we systematically analyze responses generated by state-of-the-art LLMs to identify variations. We further evaluate whether differences are augmented by four organizational culture types, thereby evaluating the practical relevance of identified biases. The findings reveal significant systematic differences between models and demonstrate that organizational culture prompts substantially modify LLM responses. The study holds important implications for LLM-assisted decision-making in sustainability contexts. 

---
# Efficient Vector Symbolic Architectures from Histogram Recovery 

**Authors**: Zirui Deng, Netanel Raviv  

**Link**: [PDF](https://arxiv.org/pdf/2511.01838)  

**Abstract**: Vector symbolic architectures (VSAs) are a family of information representation techniques which enable composition, i.e., creating complex information structures from atomic vectors via binding and superposition, and have recently found wide ranging applications in various neurosymbolic artificial intelligence (AI) systems. Recently, Raviv proposed the use of random linear codes in VSAs, suggesting that their subcode structure enables efficient binding, while preserving the quasi-orthogonality that is necessary for neural processing. Yet, random linear codes are difficult to decode under noise, which severely limits the resulting VSA's ability to support recovery, i.e., the retrieval of information objects and their attributes from a noisy compositional representation.
In this work we bridge this gap by utilizing coding theoretic tools. First, we argue that the concatenation of Reed-Solomon and Hadamard codes is suitable for VSA, due to the mutual quasi-orthogonality of the resulting codewords (a folklore result). Second, we show that recovery of the resulting compositional representations can be done by solving a problem we call histogram recovery. In histogram recovery, a collection of $N$ histograms over a finite field is given as input, and one must find a collection of Reed-Solomon codewords of length $N$ whose entry-wise symbol frequencies obey those histograms. We present an optimal solution to the histogram recovery problem by using algorithms related to list-decoding, and analyze the resulting noise resilience. Our results give rise to a noise-resilient VSA with formal guarantees regarding efficient encoding, quasi-orthogonality, and recovery, without relying on any heuristics or training, and while operating at improved parameters relative to similar solutions such as the Hadamard code. 

---
# Dynamic Routing Between Experts: A Data-Efficient Approach to Continual Learning in Vision-Language Models 

**Authors**: Jay Mohta, Kenan Emir Ak, Dimitrios Dimitriadis, Yan Xu, Mingwei Shen  

**Link**: [PDF](https://arxiv.org/pdf/2511.01831)  

**Abstract**: Vision-Language Models (VLMs) suffer from catastrophic forgetting when sequentially fine-tuned on new tasks, degrading performance on previously learned foundational and task-specific capabilities. While multi-task learning can mitigate forgetting, it requires simultaneous access to all datasets and imposes computational overhead that scales linearly with the number of tasks. In this work, we introduce a routing-based approach that enables the integration of new tasks while preserving the foundational knowledge acquired during pretraining. We evaluate our method using InternVL-2 models (2B and 8B parameters) and demonstrate that routing preserves the model's foundational capabilities by maintaining performance on general-purpose benchmarks such as ChartQA, MMBench, and DocVQA, while simultaneously improving accuracy on specialized tasks. Importantly, our approach achieves this without requiring concurrent access to data from all tasks, avoiding the significant computational and data overhead associated with traditional multi-task learning. We further conduct extensive ablation studies to evaluate the scalability and robustness of routing-based learning, showing that the approach is resilient to a growing number of tasks and performs particularly well when new tasks are semantically related. Finally, we show that the routing mechanism enables superior cross-modal transfer between language and vision capabilities, allowing knowledge learned in one modality to enhance performance in another capability not achieved by existing continual learning methods. 

---
# Machine and Deep Learning for Indoor UWB Jammer Localization 

**Authors**: Hamed Fard, Mahsa Kholghi, Benedikt Groß, Gerhard Wunder  

**Link**: [PDF](https://arxiv.org/pdf/2511.01819)  

**Abstract**: Ultra-wideband (UWB) localization delivers centimeter-scale accuracy but is vulnerable to jamming attacks, creating security risks for asset tracking and intrusion detection in smart buildings. Although machine learning (ML) and deep learning (DL) methods have improved tag localization, localizing malicious jammers within a single room and across changing indoor layouts remains largely unexplored. Two novel UWB datasets, collected under original and modified room configurations, are introduced to establish comprehensive ML/DL baselines. Performance is rigorously evaluated using a variety of classification and regression metrics. On the source dataset with the collected UWB features, Random Forest achieves the highest F1-macro score of 0.95 and XGBoost achieves the lowest mean Euclidean error of 20.16 cm. However, deploying these source-trained models in the modified room layout led to severe performance degradation, with XGBoost's mean Euclidean error increasing tenfold to 207.99 cm, demonstrating significant domain shift. To mitigate this degradation, a domain-adversarial ConvNeXt autoencoder (A-CNT) is proposed that leverages a gradient-reversal layer to align CIR-derived features across domains. The A-CNT framework restores localization performance by reducing the mean Euclidean error to 34.67 cm. This represents a 77 percent improvement over non-adversarial transfer learning and an 83 percent improvement over the best baseline, restoring the fraction of samples within 30 cm to 0.56. Overall, the results demonstrate that adversarial feature alignment enables robust and transferable indoor jammer localization despite environmental changes. Code and dataset available at this https URL 

---
# KV Cache Transform Coding for Compact Storage in LLM Inference 

**Authors**: Konrad Staniszewski, Adrian Łańcucki  

**Link**: [PDF](https://arxiv.org/pdf/2511.01815)  

**Abstract**: Serving large language models (LLMs) at scale necessitates efficient key-value (KV) cache management. KV caches can be reused across conversation turns via shared-prefix prompts that are common in iterative code editing and chat. However, stale caches consume scarce GPU memory, require offloading, or force recomputation. We present KVTC, a lightweight transform coder that compresses KV caches for compact on-GPU and off-GPU storage. Drawing on classical media compression, KVTC combines PCA-based feature decorrelation, adaptive quantization, and entropy coding. It requires only a brief initial calibration and leaves model parameters unchanged. By exploiting redundancies in KV caches, KVTC achieves up to 20$\times$ compression while maintaining reasoning and long-context accuracy, and 40$\times$ or higher for specific use cases. We test KVTC with Llama 3, Mistral NeMo, and R1-Qwen 2.5 models across benchmarks including AIME25, LiveCodeBench, GSM8K, MMLU, Qasper, RULER, and MATH-500. It consistently outperforms inference-time baselines such as token eviction, quantization, and SVD-based methods, while achieving higher compression ratios. These results support KVTC as a practical building block for memory-efficient LLM serving with reusable KV caches. 

---
# Plan-and-Write: Structure-Guided Length Control for LLMs without Model Retraining 

**Authors**: Adewale Akinfaderin, Shreyas Subramanian, Akarsha Sehwag  

**Link**: [PDF](https://arxiv.org/pdf/2511.01807)  

**Abstract**: Length control in Large Language Models (LLMs) is a crucial but under-addressed challenge, with applications ranging from voice interfaces requiring concise responses to research summaries needing comprehensive outputs. Current approaches to length control, including Regularized DPO, Length-Instruction Fine Tuning, and tool-augmented methods, typically require expensive model retraining or complex inference-time tooling. This paper presents a prompt engineering methodology that enables precise length control without model retraining. Our structure-guided approach implements deliberate planning and word counting mechanisms within the prompt, encouraging the model to carefully track and adhere to specified length constraints. Comprehensive evaluations across six state-of-the-art LLMs demonstrate that our method significantly improves length fidelity for several models compared to standard prompting when applied to document summarization tasks, particularly for shorter-to-medium length constraints. The proposed technique shows varying benefits across different model architectures, with some models demonstrating up to 37.6% improvement in length adherence. Quality evaluations further reveal that our approach maintains or enhances overall output quality compared to standard prompting techniques. Our approach provides an immediately deployable solution for applications requiring precise length control, particularly valuable for production environments where model retraining is impractical or cost-prohibitive. 

---
# Accumulating Context Changes the Beliefs of Language Models 

**Authors**: Jiayi Geng, Howard Chen, Ryan Liu, Manoel Horta Ribeiro, Robb Willer, Graham Neubig, Thomas L. Griffiths  

**Link**: [PDF](https://arxiv.org/pdf/2511.01805)  

**Abstract**: Language model (LM) assistants are increasingly used in applications such as brainstorming and research. Improvements in memory and context size have allowed these models to become more autonomous, which has also resulted in more text accumulation in their context windows without explicit user intervention. This comes with a latent risk: the belief profiles of models -- their understanding of the world as manifested in their responses or actions -- may silently change as context accumulates. This can lead to subtly inconsistent user experiences, or shifts in behavior that deviate from the original alignment of the models. In this paper, we explore how accumulating context by engaging in interactions and processing text -- talking and reading -- can change the beliefs of language models, as manifested in their responses and this http URL results reveal that models' belief profiles are highly malleable: GPT-5 exhibits a 54.7% shift in its stated beliefs after 10 rounds of discussion about moral dilemmas and queries about safety, while Grok 4 shows a 27.2% shift on political issues after reading texts from the opposing position. We also examine models' behavioral changes by designing tasks that require tool use, where each tool selection corresponds to an implicit belief. We find that these changes align with stated belief shifts, suggesting that belief shifts will be reflected in actual behavior in agentic systems. Our analysis exposes the hidden risk of belief shift as models undergo extended sessions of talking or reading, rendering their opinions and actions unreliable. 

---
# Fractional Diffusion Bridge Models 

**Authors**: Gabriel Nobis, Maximilian Springenberg, Arina Belova, Rembert Daems, Christoph Knochenhauer, Manfred Opper, Tolga Birdal, Wojciech Samek  

**Link**: [PDF](https://arxiv.org/pdf/2511.01795)  

**Abstract**: We present Fractional Diffusion Bridge Models (FDBM), a novel generative diffusion bridge framework driven by an approximation of the rich and non-Markovian fractional Brownian motion (fBM). Real stochastic processes exhibit a degree of memory effects (correlations in time), long-range dependencies, roughness and anomalous diffusion phenomena that are not captured in standard diffusion or bridge modeling due to the use of Brownian motion (BM). As a remedy, leveraging a recent Markovian approximation of fBM (MA-fBM), we construct FDBM that enable tractable inference while preserving the non-Markovian nature of fBM. We prove the existence of a coupling-preserving generative diffusion bridge and leverage it for future state prediction from paired training data. We then extend our formulation to the Schrödinger bridge problem and derive a principled loss function to learn the unpaired data translation. We evaluate FDBM on both tasks: predicting future protein conformations from aligned data, and unpaired image translation. In both settings, FDBM achieves superior performance compared to the Brownian baselines, yielding lower root mean squared deviation (RMSD) of C$_\alpha$ atomic positions in protein structure prediction and lower Fréchet Inception Distance (FID) in unpaired image translation. 

---
# Random Initialization of Gated Sparse Adapters 

**Authors**: Vi Retault, Yohaï-Eliel Berreby  

**Link**: [PDF](https://arxiv.org/pdf/2511.01794)  

**Abstract**: When fine-tuning language models on new tasks, catastrophic forgetting -- performance degradation on previously-learned tasks -- is a ubiquitous problem. While Parameter-Efficient Fine-Tuning (PEFT) methods like LoRA address this through low-rank adapters, sparse adaptation offers an alternative that doesn't impose rank constraints. We introduce Random Initialization of Gated Sparse Adapters (RIGSA), which starts from randomly-initialized full-rank adapters, gates them with a ReZero analog, and sparsifies them with iterative magnitude pruning. We evaluate RIGSA on SmolLM2-1.7B-Instruct using a novel vision-in-text task (Textual MNIST) and measure forgetting on PIQA, HellaSwag, and GSM8k. SmolLM2-1.7B-Instruct initially performs around chance level on Textual MNIST, and is capable of learning the task through RIGSA, 4-bit QLoRA and random masking. In spite of having more trainable parameters than QLoRA, the RIGSA configurations that we studied displayed less forgetting than QLoRA, particularly on GSM8k, though it performs comparably to random masking. 

---
# GenDexHand: Generative Simulation for Dexterous Hands 

**Authors**: Feng Chen, Zhuxiu Xu, Tianzhe Chu, Xunzhe Zhou, Li Sun, Zewen Wu, Shenghua Gao, Zhongyu Li, Yanchao Yang, Yi Ma  

**Link**: [PDF](https://arxiv.org/pdf/2511.01791)  

**Abstract**: Data scarcity remains a fundamental bottleneck for embodied intelligence. Existing approaches use large language models (LLMs) to automate gripper-based simulation generation, but they transfer poorly to dexterous manipulation, which demands more specialized environment design. Meanwhile, dexterous manipulation tasks are inherently more difficult due to their higher degrees of freedom. Massively generating feasible and trainable dexterous hand tasks remains an open challenge. To this end, we present GenDexHand, a generative simulation pipeline that autonomously produces diverse robotic tasks and environments for dexterous manipulation. GenDexHand introduces a closed-loop refinement process that adjusts object placements and scales based on vision-language model (VLM) feedback, substantially improving the average quality of generated environments. Each task is further decomposed into sub-tasks to enable sequential reinforcement learning, reducing training time and increasing success rates. Our work provides a viable path toward scalable training of diverse dexterous hand behaviors in embodied intelligence by offering a simulation-based solution to synthetic data generation. Our website: this https URL. 

---
# How Far Are Surgeons from Surgical World Models? A Pilot Study on Zero-shot Surgical Video Generation with Expert Assessment 

**Authors**: Zhen Chen, Qing Xu, Jinlin Wu, Biao Yang, Yuhao Zhai, Geng Guo, Jing Zhang, Yinlu Ding, Nassir Navab, Jiebo Luo  

**Link**: [PDF](https://arxiv.org/pdf/2511.01775)  

**Abstract**: Foundation models in video generation are demonstrating remarkable capabilities as potential world models for simulating the physical world. However, their application in high-stakes domains like surgery, which demand deep, specialized causal knowledge rather than general physical rules, remains a critical unexplored gap. To systematically address this challenge, we present SurgVeo, the first expert-curated benchmark for video generation model evaluation in surgery, and the Surgical Plausibility Pyramid (SPP), a novel, four-tiered framework tailored to assess model outputs from basic appearance to complex surgical strategy. On the basis of the SurgVeo benchmark, we task the advanced Veo-3 model with a zero-shot prediction task on surgical clips from laparoscopic and neurosurgical procedures. A panel of four board-certified surgeons evaluates the generated videos according to the SPP. Our results reveal a distinct "plausibility gap": while Veo-3 achieves exceptional Visual Perceptual Plausibility, it fails critically at higher levels of the SPP, including Instrument Operation Plausibility, Environment Feedback Plausibility, and Surgical Intent Plausibility. This work provides the first quantitative evidence of the chasm between visually convincing mimicry and causal understanding in surgical AI. Our findings from SurgVeo and the SPP establish a crucial foundation and roadmap for developing future models capable of navigating the complexities of specialized, real-world healthcare domains. 

---
# Wonder3D++: Cross-domain Diffusion for High-fidelity 3D Generation from a Single Image 

**Authors**: Yuxiao Yang, Xiao-Xiao Long, Zhiyang Dou, Cheng Lin, Yuan Liu, Qingsong Yan, Yuexin Ma, Haoqian Wang, Zhiqiang Wu, Wei Yin  

**Link**: [PDF](https://arxiv.org/pdf/2511.01767)  

**Abstract**: In this work, we introduce \textbf{Wonder3D++}, a novel method for efficiently generating high-fidelity textured meshes from single-view images. Recent methods based on Score Distillation Sampling (SDS) have shown the potential to recover 3D geometry from 2D diffusion priors, but they typically suffer from time-consuming per-shape optimization and inconsistent geometry. In contrast, certain works directly produce 3D information via fast network inferences, but their results are often of low quality and lack geometric details. To holistically improve the quality, consistency, and efficiency of single-view reconstruction tasks, we propose a cross-domain diffusion model that generates multi-view normal maps and the corresponding color images. To ensure the consistency of generation, we employ a multi-view cross-domain attention mechanism that facilitates information exchange across views and modalities. Lastly, we introduce a cascaded 3D mesh extraction algorithm that drives high-quality surfaces from the multi-view 2D representations in only about $3$ minute in a coarse-to-fine manner. Our extensive evaluations demonstrate that our method achieves high-quality reconstruction results, robust generalization, and good efficiency compared to prior works. Code available at this https URL. 

---
# Context-Guided Decompilation: A Step Towards Re-executability 

**Authors**: Xiaohan Wang, Yuxin Hu, Kevin Leach  

**Link**: [PDF](https://arxiv.org/pdf/2511.01763)  

**Abstract**: Binary decompilation plays an important role in software security analysis, reverse engineering, and malware understanding when source code is unavailable. However, existing decompilation techniques often fail to produce source code that can be successfully recompiled and re-executed, particularly for optimized binaries. Recent advances in large language models (LLMs) have enabled neural approaches to decompilation, but the generated code is typically only semantically plausible rather than truly executable, limiting their practical reliability. These shortcomings arise from compiler optimizations and the loss of semantic cues in compiled code, which LLMs struggle to recover without contextual guidance. To address this challenge, we propose ICL4Decomp, a hybrid decompilation framework that leverages in-context learning (ICL) to guide LLMs toward generating re-executable source code. We evaluate our method across multiple datasets, optimization levels, and compilers, demonstrating around 40\% improvement in re-executability over state-of-the-art decompilation methods while maintaining robustness. 

---
# RLAC: Reinforcement Learning with Adversarial Critic for Free-Form Generation Tasks 

**Authors**: Mian Wu, Gavin Zhang, Sewon Min, Sergey Levine, Aviral Kumar  

**Link**: [PDF](https://arxiv.org/pdf/2511.01758)  

**Abstract**: Open-ended generation tasks require outputs to satisfy diverse and often implicit task-specific evaluation rubrics. The sheer number of relevant rubrics leads to prohibitively high verification costs and incomplete assessments of a response, making reinforcement learning (RL) post-training with rubric-based rewards difficult to scale. This problem is exacerbated by the fact that often the best way to combine these rubrics into one single reward is also highly prompt-specific. We propose Reinforcement Learning with Adversarial Critic (RLAC), a post-training approach that addresses these challenges via dynamic rubric verification. Our approach employs a large language model (LLM) as a critic that dynamically identifies only the most likely failure modes (e.g., a factual error or unhandled edge case), which are then verified by an external validator to optimize both generator and critic jointly. By training both the generator and the critic, this game enhances the critic's error detection and the generator's output quality while reducing required verifications. Our experiments demonstrate that RLAC improves factual accuracy in text generation and correctness in code generation, while also outperforming exhaustive verification and reward model methods. We show that dynamic critics are more effective than fixed critics, showcasing the potential of RLAC for scaling RL post-training to free-form generation tasks. 

---
# SM-based Semantics for Answer Set Programs Containing Conditional Literals and Arithmetic 

**Authors**: Zachary Hansen, Yuliya Lierler  

**Link**: [PDF](https://arxiv.org/pdf/2511.01753)  

**Abstract**: Modern answer set programming solvers such as CLINGO support advanced language constructs that improve the expressivity and conciseness of logic programs. Conditional literals are one such construct. They form "subformulas" that behave as nested implications within the bodies of logic rules. Their inclusion brings the form of rules closer to the less restrictive syntax of first-order logic. These qualities make conditional literals useful tools for knowledge representation. In this paper, we propose a semantics for logic programs with conditional literals and arithmetic based on the SM operator. These semantics do not require grounding, unlike the established semantics for such programs that relies on a translation to infinitary propositional logic. The main result of this paper establishes the precise correspondence between the proposed and existing semantics. 

---
# Scam Shield: Multi-Model Voting and Fine-Tuned LLMs Against Adversarial Attacks 

**Authors**: Chen-Wei Chang, Shailik Sarkar, Hossein Salemi, Hyungmin Kim, Shutonu Mitra, Hemant Purohit, Fengxiu Zhang, Michin Hong, Jin-Hee Cho, Chang-Tien Lu  

**Link**: [PDF](https://arxiv.org/pdf/2511.01746)  

**Abstract**: Scam detection remains a critical challenge in cybersecurity as adversaries craft messages that evade automated filters. We propose a Hierarchical Scam Detection System (HSDS) that combines a lightweight multi-model voting front end with a fine-tuned LLaMA 3.1 8B Instruct back end to improve accuracy and robustness against adversarial attacks. An ensemble of four classifiers provides preliminary predictions through majority vote, and ambiguous cases are escalated to the fine-tuned model, which is optimized with adversarial training to reduce misclassification. Experiments show that this hierarchical design both improves adversarial scam detection and shortens inference time by routing most cases away from the LLM, outperforming traditional machine-learning baselines and proprietary LLM baselines. The findings highlight the effectiveness of a hybrid voting mechanism and adversarial fine-tuning in fortifying LLMs against evolving scam tactics, enhancing the resilience of automated scam detection systems. 

---
# An Open-Access Benchmark of Statistical and Machine-Learning Anomaly Detection Methods for Battery Applications 

**Authors**: Mei-Chin Pang, Suraj Adhikari, Takuma Kasahara, Nagihiro Haba, Saneyuki Ohno  

**Link**: [PDF](https://arxiv.org/pdf/2511.01745)  

**Abstract**: Battery safety is critical in applications ranging from consumer electronics to electric vehicles and aircraft, where undetected anomalies could trigger safety hazards or costly downtime. In this study, we present OSBAD as an open-source benchmark for anomaly detection frameworks in battery applications. By benchmarking 15 diverse algorithms encompassing statistical, distance-based, and unsupervised machine-learning methods, OSBAD enables a systematic comparison of anomaly detection methods across heterogeneous datasets. In addition, we demonstrate how a physics- and statistics-informed feature transformation workflow enhances anomaly separability by decomposing collective anomalies into point anomalies. To address a major bottleneck in unsupervised anomaly detection due to incomplete labels, we propose a Bayesian optimization pipeline that facilitates automated hyperparameter tuning based on transfer-learning and regression proxies. Through validation on datasets covering both liquid and solid-state chemistries, we further demonstrate the cross-chemistry generalization capability of OSBAD to identify irregularities across different electrochemical systems. By making benchmarking database with open-source reproducible anomaly detection workflows available to the community, OSBAD establishes a unified foundation for developing safe, scalable, and transferable anomaly detection tools in battery analytics. This research underscores the significance of physics- and statistics-informed feature engineering as well as model selection with probabilistic hyperparameter tuning, in advancing trustworthy, data-driven diagnostics for safety-critical energy systems. 

---
# Towards Efficient Federated Learning of Networked Mixture-of-Experts for Mobile Edge Computing 

**Authors**: Song Gao, Shusen Jing, Shuai Zhang, Yue Wang, Xiangwei Zhou, Songyang Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2511.01743)  

**Abstract**: Recent advancements in large artificial intelligence models (LAMs) are driving significant innovations in mobile edge computing within next-generation wireless networks. However, the substantial demands for computational resources and large-scale training data required to train LAMs conflict with the limited storage and computational capacity of edge devices, posing significant challenges to training and deploying LAMs at the edge. In this work, we introduce the Networked Mixture-of-Experts (NMoE) system, in which clients infer collaboratively by distributing tasks to suitable neighbors based on their expertise and aggregate the returned results. For training the NMoE, we propose a federated learning framework that integrates both supervised and self-supervised learning to balance personalization and generalization, while preserving communication efficiency and data privacy. We conduct extensive experiments to demonstrate the efficacy of the proposed NMoE system, providing insights and benchmarks for the NMoE training algorithms. 

---
# A Proof of Learning Rate Transfer under $μ$P 

**Authors**: Soufiane Hayou  

**Link**: [PDF](https://arxiv.org/pdf/2511.01734)  

**Abstract**: We provide the first proof of learning rate transfer with width in a linear multi-layer perceptron (MLP) parametrized with $\mu$P, a neural network parameterization designed to ``maximize'' feature learning in the infinite-width limit. We show that under $\mu P$, the optimal learning rate converges to a \emph{non-zero constant} as width goes to infinity, providing a theoretical explanation to learning rate transfer. In contrast, we show that this property fails to hold under alternative parametrizations such as Standard Parametrization (SP) and Neural Tangent Parametrization (NTP). We provide intuitive proofs and support the theoretical findings with extensive empirical results. 

---
# Multi-Step Knowledge Interaction Analysis via Rank-2 Subspace Disentanglement 

**Authors**: Sekh Mainul Islam, Pepa Atanasova, Isabelle Augenstein  

**Link**: [PDF](https://arxiv.org/pdf/2511.01706)  

**Abstract**: Natural Language Explanations (NLEs) describe how Large Language Models (LLMs) make decisions, drawing on both external Context Knowledge (CK) and Parametric Knowledge (PK) stored in model weights. Understanding their interaction is key to assessing the grounding of NLEs, yet it remains underexplored. Prior work has largely examined only single-step generation, typically the final answer, and has modelled PK and CK interaction only as a binary choice in a rank-1 subspace. This overlooks richer forms of interaction, such as complementary or supportive knowledge. We propose a novel rank-2 projection subspace that disentangles PK and CK contributions more accurately and use it for the first multi-step analysis of knowledge interactions across longer NLE sequences. Experiments on four QA datasets and three open-weight instruction-tuned LLMs show that diverse knowledge interactions are poorly represented in a rank-1 subspace but are effectively captured in our rank-2 formulation. Our multi-step analysis reveals that hallucinated NLEs align strongly with the PK direction, context-faithful ones balance PK and CK, and Chain-of-Thought prompting for NLEs shifts generated NLEs toward CK by reducing PK reliance. This work provides the first framework for systematic studies of multi-step knowledge interactions in LLMs through a richer rank-2 subspace disentanglement. Code and data: this https URL. 

---
# Solution Space Topology Guides CMTS Search 

**Authors**: Mirco A. Mannucci  

**Link**: [PDF](https://arxiv.org/pdf/2511.01701)  

**Abstract**: A fundamental question in search-guided AI: what topology should guide Monte Carlo Tree Search (MCTS) in puzzle solving? Prior work applied topological features to guide MCTS in ARC-style tasks using grid topology -- the Laplacian spectral properties of cell connectivity -- and found no benefit. We identify the root cause: grid topology is constant across all instances. We propose measuring \emph{solution space topology} instead: the structure of valid color assignments constrained by detected pattern rules. We build this via compatibility graphs where nodes are $(cell, color)$ pairs and edges represent compatible assignments under pattern constraints.
Our method: (1) detect pattern rules automatically with 100\% accuracy on 5 types, (2) construct compatibility graphs encoding solution space structure, (3) extract topological features (algebraic connectivity, rigidity, color structure) that vary with task difficulty, (4) integrate these features into MCTS node selection via sibling-normalized scores.
We provide formal definitions, a rigorous selection formula, and comprehensive ablations showing that algebraic connectivity is the dominant signal. The work demonstrates that topology matters for search -- but only the \emph{right} topology. For puzzle solving, this is solution space structure, not problem space structure. 

---
# Bayesian Natural Gradient Fine-Tuning of CLIP Models via Kalman Filtering 

**Authors**: Hossein Abdi, Mingfei Sun, Wei Pan  

**Link**: [PDF](https://arxiv.org/pdf/2511.01694)  

**Abstract**: Vision-language pre-trained models, such as CLIP, have established new benchmarks in multimodal data mining. In such models, few-shot fine-tuning is a major challenge to achieve optimal performance on both in-distribution (ID) and out-of-distribution (OOD) datasets, especially when labeled data is scarce. Most existing fine-tuning approaches rely on first-order gradient-based optimizers, which typically suffer from slow convergence, sensitivity to step-size hyperparameters, and poor generalization in OOD settings. In contrast, second-order methods utilize local curvature information of the loss landscape to adjust the update step size. This is particularly beneficial for CLIP models, whose non-convex loss functions often contain sharp critical points. In such cases, natural gradient direction can offer more substantial and efficient per-iteration updates when fine-tuning with limited data. Natural Gradient Descent (NGD) is obtained by preconditioning the standard gradient with the inverse Fisher Information Matrix (FIM), which is computationally expensive for large models. To address this, we propose a Bayesian approximation of NGD using a Kalman filter for CLIP models. Our method combines the benefits of second-order optimization with Bayesian inference, which enhances generalization while providing uncertainty quantification. Extensive experiments conducted on diverse image classification datasets demonstrate that our algorithm consistently achieves superior--or comparable--ID performance and improved OOD robustness compared to state-of-the-art baselines. To the best of our knowledge, this work represents the first successful application of Kalman filtering to fine-tuning CLIP-based models, which enables more robust and efficient learning in vision-language tasks. 

---
# Open Character Training: Shaping the Persona of AI Assistants through Constitutional AI 

**Authors**: Sharan Maiya, Henning Bartsch, Nathan Lambert, Evan Hubinger  

**Link**: [PDF](https://arxiv.org/pdf/2511.01689)  

**Abstract**: The character of the "AI assistant" persona generated by modern chatbot large language models influences both surface-level behavior and apparent values, beliefs, and ethics. These all affect interaction quality, perceived intelligence, and alignment with both developer and user intentions. The shaping of this persona, known as character training, is a critical component of industry post-training, yet remains effectively unstudied in the academic literature. We introduce the first open implementation of character training, leveraging Constitutional AI and a new data pipeline using synthetic introspective data to shape the assistant persona in a more effective and controlled manner than alternatives such as constraining system prompts or activation steering. Specifically, we fine-tune three popular open-weights models using 11 example personas, such as humorous, deeply caring, or even malevolent. To track the effects of our approach, we introduce a method which analyzes revealed preferences, uncovering clear and holistic changes in character. We find these changes are more robust to adversarial prompting than the above two alternatives, while also leading to more coherent and realistic generations. Finally, we demonstrate this fine-tuning has little to no effect on general capabilities as measured by common benchmarks. We describe and open-source our full post-training method, the implementation of which can be found at this https URL. 

---
# Student Engagement in AI Assisted Complex Problem Solving: A Pilot Study of Human AI Rubik's Cube Collaboration 

**Authors**: Kirk Vanacore, Jaclyn Ocumpaugh, Forest Agostinelli, Dezhi Wu, Sai Vuruma, Matt Irvin  

**Link**: [PDF](https://arxiv.org/pdf/2511.01683)  

**Abstract**: Games and puzzles play important pedagogical roles in STEM learning. New AI algorithms that can solve complex problems offer opportunities for scaffolded instruction in puzzle solving. This paper presents the ALLURE system, which uses an AI algorithm (DeepCubeA) to guide students in solving a common first step of the Rubik's Cube (i.e., the white cross). Using data from a pilot study we present preliminary findings about students' behaviors in the system, how these behaviors are associated with STEM skills - including spatial reasoning, critical thinking and algorithmic thinking. We discuss how data from ALLURE can be used in future educational data mining to understand how students benefit from AI assistance and collaboration when solving complex problems. 

---
# Spin-Adapted Neural Network Wavefunctions in Real Space 

**Authors**: Ruichen Li, Yuzhi Liu, Du Jiang, Yixiao Chen, Xuelan Wen, Wenrui Li, Di He, Liwei Wang, Ji Chen, Weiluo Ren  

**Link**: [PDF](https://arxiv.org/pdf/2511.01671)  

**Abstract**: Spin plays a fundamental role in understanding electronic structure, yet many real-space wavefunction methods fail to adequately consider it. We introduce the Spin-Adapted Antisymmetrization Method (SAAM), a general procedure that enforces exact total spin symmetry for antisymmetric many-electron wavefunctions in real space. In the context of neural network-based quantum Monte Carlo (NNQMC), SAAM leverages the expressiveness of deep neural networks to capture electron correlation while enforcing exact spin adaptation via group representation theory. This framework provides a principled route to embed physical priors into otherwise black-box neural network wavefunctions, yielding a compact representation of correlated system with neural network orbitals. Compared with existing treatments of spin in NNQMC, SAAM is more accurate and efficient, achieving exact spin purity without any additional tunable hyperparameters. To demonstrate its effectiveness, we apply SAAM to study the spin ladder of iron-sulfur clusters, a long-standing challenge for many-body methods due to their dense spectrum of nearly degenerate spin states. Our results reveal accurate resolution of low-lying spin states and spin gaps in [Fe$_2$S$_2$] and [Fe$_4$S$_4$] clusters, offering new insights into their electronic structures. In sum, these findings establish SAAM as a robust, hyperparameter-free standard for spin-adapted NNQMC, particularly for strongly correlated systems. 

---
# SeaLLMs-Audio: Large Audio-Language Models for Southeast Asia 

**Authors**: Chaoqun Liu, Mahani Aljunied, Guizhen Chen, Hou Pong Chan, Weiwen Xu, Yu Rong, Wenxuan Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2511.01670)  

**Abstract**: We introduce SeaLLMs-Audio, the first large audio-language model (LALM) tailored for multiple Southeast Asian (SEA) languages-Indonesian (id), Thai (th), and Vietnamese (vi)-alongside English (en) and Chinese (zh). Trained on a large-scale audio corpus, SeaLLMs-Audio exhibits strong performance across diverse audio-centric tasks, spanning fine-grained audio understanding and voice-based interaction. Its key features include: 1) Multilingual: the model primarily supports 5 languages, namely Indonesian, Thai, Vietnamese, English, and Chinese; 2) Multimodal: the model accepts flexible input modalities, including audio only, text only, as well as audio with text; 3) Multi-task: the model supports a wide range of tasks, including audio analysis tasks such as Audio Captioning, Automatic Speech Recognition, Speech-to-Text Translation, Speech Emotion Recognition, Speech Question Answering, and Speech Summarization. It also enables voice-based dialogue, including answering factual, mathematical, and general knowledge queries. As a significant step towards advancing audio LLMs in Southeast Asia, we expect SeaLLMs-Audio to benefit both the regional research community and industry. To automate LALM evaluation for Southeast Asia, we introduce SeaBench-Audio, a benchmark spanning multiple tasks. Experiments show that SeaLLMs-Audio achieves competitive performance compared with other LALMs on SEA languages. 

---
# The Ghost in the Keys: A Disklavier Demo for Human-AI Musical Co-Creativity 

**Authors**: Louis Bradshaw, Alexander Spangher, Stella Biderman, Simon Colton  

**Link**: [PDF](https://arxiv.org/pdf/2511.01663)  

**Abstract**: While generative models for music composition are increasingly capable, their adoption by musicians is hindered by text-prompting, an asynchronous workflow disconnected from the embodied, responsive nature of instrumental performance. To address this, we introduce Aria-Duet, an interactive system facilitating a real-time musical duet between a human pianist and Aria, a state-of-the-art generative model, using a Yamaha Disklavier as a shared physical interface. The framework enables a turn-taking collaboration: the user performs, signals a handover, and the model generates a coherent continuation performed acoustically on the piano. Beyond describing the technical architecture enabling this low-latency interaction, we analyze the system's output from a musicological perspective, finding the model can maintain stylistic semantics and develop coherent phrasal ideas, demonstrating that such embodied systems can engage in musically sophisticated dialogue and open a promising new path for human-AI co-creation. 

---
# EngChain: A Symbolic Benchmark for Verifiable Multi-Step Reasoning in Engineering 

**Authors**: Ayesha Gull, Muhammad Usman Safder, Rania Elbadry, Preslav Nakov, Zhuohan Xie  

**Link**: [PDF](https://arxiv.org/pdf/2511.01650)  

**Abstract**: Large Language Models (LLMs) are increasingly being applied to specialized, high-stakes domains like engineering, which demands rigorous evaluation of their complex reasoning capabilities. While current benchmarks assess language understanding, factual recall, mathematics or code generation, none capture the integrative reasoning central to engineering where scientific principles, quantitative modeling and practical constraints must converge. To address this gap, we introduce EngChain, a benchmark for verifiable multi-step engineering problem-solving. EngChain contains 90 problems spanning three engineering branches, organized into 9 domains and 20 distinct areas. The problems are generated from symbolic templates with a high degree of randomization to ensure diversity and eliminate the risk of contamination. With this benchmark, we move beyond final answer accuracy with a two-stage evaluation: we first quantitatively verify the numerical and semantic validity of each reasoning step and then introduce LLM-As-A-Judge, an automated system to qualitatively categorize the identified reasoning errors. 

---
# A Graph-based RAG for Energy Efficiency Question Answering 

**Authors**: Riccardo Campi, Nicolò Oreste Pinciroli Vago, Mathyas Giudici, Pablo Barrachina Rodriguez-Guisado, Marco Brambilla, Piero Fraternali  

**Link**: [PDF](https://arxiv.org/pdf/2511.01643)  

**Abstract**: In this work, we investigate the use of Large Language Models (LLMs) within a graph-based Retrieval Augmented Generation (RAG) architecture for Energy Efficiency (EE) Question Answering. First, the system automatically extracts a Knowledge Graph (KG) from guidance and regulatory documents in the energy field. Then, the generated graph is navigated and reasoned upon to provide users with accurate answers in multiple languages. We implement a human-based validation using the RAGAs framework properties, a validation dataset comprising 101 question-answer pairs, and domain experts. Results confirm the potential of this architecture and identify its strengths and weaknesses. Validation results show how the system correctly answers in about three out of four of the cases (75.2 +- 2.7%), with higher results on questions related to more general EE answers (up to 81.0 +- 4.1%), and featuring promising multilingual abilities (4.4% accuracy loss due to translation). 

---
# Prompt Injection as an Emerging Threat: Evaluating the Resilience of Large Language Models 

**Authors**: Daniyal Ganiuly, Assel Smaiyl  

**Link**: [PDF](https://arxiv.org/pdf/2511.01634)  

**Abstract**: Large Language Models (LLMs) are increasingly used in intelligent systems that perform reasoning, summarization, and code generation. Their ability to follow natural-language instructions, while powerful, also makes them vulnerable to a new class of attacks known as prompt injection. In these attacks, hidden or malicious instructions are inserted into user inputs or external content, causing the model to ignore its intended task or produce unsafe responses. This study proposes a unified framework for evaluating how resistant Large Language Models (LLMs) are to prompt injection attacks. The framework defines three complementary metrics such as the Resilience Degradation Index (RDI), Safety Compliance Coefficient (SCC), and Instructional Integrity Metric (IIM) to jointly measure robustness, safety, and semantic stability. We evaluated four instruction-tuned models (GPT-4, GPT-4o, LLaMA-3 8B Instruct, and Flan-T5-Large) on five common language tasks: question answering, summarization, translation, reasoning, and code generation. Results show that GPT-4 performs best overall, while open-weight models remain more vulnerable. The findings highlight that strong alignment and safety tuning are more important for resilience than model size alone. Results show that all models remain partially vulnerable, especially to indirect and direct-override attacks. GPT-4 achieved the best overall resilience (RDR = 9.8 %, SCR = 96.4 %), while open-source models exhibited higher performance degradation and lower safety scores. The findings demonstrate that alignment strength and safety tuning play a greater role in resilience than model size alone. The proposed framework offers a structured, reproducible approach for assessing model robustness and provides practical insights for improving LLM safety and reliability. 

---
# Scaling Graph Chain-of-Thought Reasoning: A Multi-Agent Framework with Efficient LLM Serving 

**Authors**: Chengying Huan, Ziheng Meng, Yongchao Liu, Zhengyi Yang, Yun Zhu, Yue Yun, Shipeng Li, Rong Gu, Xiabao Wu, Haitao Zhang, Chuntao Hong, Shaonan Ma, Guihai Chen, Chen Tian  

**Link**: [PDF](https://arxiv.org/pdf/2511.01633)  

**Abstract**: Graph Chain-of-Thought (Graph-CoT) enables large language models (LLMs) to perform step-by-step reasoning over graph-structured knowledge, but existing pipelines suffer from low accuracy, excessive token usage, high latency, and low throughput due to single-agent monolithic prompts, repeated context re-encoding, and inefficient serving execution. We present GLM, the first multi-agent Graph-CoT system co-designed with an optimized LLM serving architecture. GLM decomposes reasoning into specialized agents for classification, reasoning, action generation, and graph retrieval, enabling branching and selective context sharing to reduce prompt length and reasoning iterations while preserving reasoning quality, thereby improving accuracy and reducing overall token consumption. To scale inference, we introduce a Graph-CoT-aware LLM inference mechanism with graph-specific KV-cache management, priority-based eviction, and pipelined execution to improve serving efficiency. Experiments demonstrate that GLM improves answer accuracy by up to 38%, reduces token cost by up to 95.7%, lowers inference latency by 90.3%, and achieves up to 15.1x higher throughput compared to state-of-the-art Graph-CoT baselines, enabling efficient adoption for complex real-world reasoning at scale. 

---
# Imperfect Language, Artificial Intelligence, and the Human Mind: An Interdisciplinary Approach to Linguistic Errors in Native Spanish Speakers 

**Authors**: Francisco Portillo López  

**Link**: [PDF](https://arxiv.org/pdf/2511.01615)  

**Abstract**: Linguistic errors are not merely deviations from normative grammar; they offer a unique window into the cognitive architecture of language and expose the current limitations of artificial systems that seek to replicate them. This project proposes an interdisciplinary study of linguistic errors produced by native Spanish speakers, with the aim of analyzing how current large language models (LLM) interpret, reproduce, or correct them. The research integrates three core perspectives: theoretical linguistics, to classify and understand the nature of the errors; neurolinguistics, to contextualize them within real-time language processing in the brain; and natural language processing (NLP), to evaluate their interpretation against linguistic errors. A purpose-built corpus of authentic errors of native Spanish (+500) will serve as the foundation for empirical analysis. These errors will be tested against AI models such as GPT or Gemini to assess their interpretative accuracy and their ability to generalize patterns of human linguistic behavior. The project contributes not only to the understanding of Spanish as a native language but also to the development of NLP systems that are more cognitively informed and capable of engaging with the imperfect, variable, and often ambiguous nature of real human language. 

---
# DINO-MX: A Modular & Flexible Framework for Self-Supervised Learning 

**Authors**: Mahmut Selman Gokmen, Cody Bumgardner  

**Link**: [PDF](https://arxiv.org/pdf/2511.01610)  

**Abstract**: Vision Foundation Models (VFMs) have advanced representation learning through self-supervised methods. However, existing training pipelines are often inflexible, domain-specific, or computationally expensive, which limits their usability across different domains and resource settings. DINO-MX is a modular and extensible training framework that combines the core principles of DINO, DINOv2 and DINOv3 within a unified configuration-driven system. It supports a variety of transformer-based architectures and is fully compatible with the Hugging Face ecosystem. The framework includes multiple training strategies such as low-rank adaptation (LoRA), layer freezing, and knowledge distillation, along with support for distributed training through both Distributed Data Parallel (DDP) and Fully Sharded Data Parallel (FSDP). DINO-MX is designed to work with both natural and specialized data types, including single- and multi-channel images. Experimental results on diverse datasets show that DINO-MX achieves competitive performance while significantly reducing computational costs. Additionally, it offers interpretability tools and a label-guided data augmentation method that improves attention-based localization without the need for extra detection or segmentation heads. DINO-MX provides a reproducible and scalable foundation for developing, adapting, and benchmarking self-supervised vision models across a range of research and real-world applications. 

---
# Federated Cyber Defense: Privacy-Preserving Ransomware Detection Across Distributed Systems 

**Authors**: Daniel M. Jimenez-Gutierrez, Enrique Zuazua, Joaquin Del Rio, Oleksii Sliusarenko, Xabi Uribe-Etxebarria  

**Link**: [PDF](https://arxiv.org/pdf/2511.01583)  

**Abstract**: Detecting malware, especially ransomware, is essential to securing today's interconnected ecosystems, including cloud storage, enterprise file-sharing, and database services. Training high-performing artificial intelligence (AI) detectors requires diverse datasets, which are often distributed across multiple organizations, making centralization necessary. However, centralized learning is often impractical due to security, privacy regulations, data ownership issues, and legal barriers to cross-organizational sharing. Compounding this challenge, ransomware evolves rapidly, demanding models that are both robust and adaptable.
In this paper, we evaluate Federated Learning (FL) using the this http URL FL platform, which enables multiple organizations to collaboratively train a ransomware detection model while keeping raw data local and secure. This paradigm is particularly relevant for cybersecurity companies (including both software and hardware vendors) that deploy ransomware detection or firewall systems across millions of endpoints. In such environments, data cannot be transferred outside the customer's device due to strict security, privacy, or regulatory constraints. Although FL applies broadly to malware threats, we validate the approach using the Ransomware Storage Access Patterns (RanSAP) dataset.
Our experiments demonstrate that FL improves ransomware detection accuracy by a relative 9% over server-local models and achieves performance comparable to centralized training. These results indicate that FL offers a scalable, high-performing, and privacy-preserving framework for proactive ransomware detection across organizational and regulatory boundaries. 

---
# HIT-ROCKET: Hadamard-vector Inner-product Transformer for ROCKET 

**Authors**: Wang Hao, Kuang Zhang, Hou Chengyu, Yuan Zhonghao, Tan Chenxing, Fu Weifeng, Zhu Yangying  

**Link**: [PDF](https://arxiv.org/pdf/2511.01572)  

**Abstract**: Time series classification holds broad application value in communications, information countermeasures, finance, and medicine. However, state-of-the-art (SOTA) methods-including HIVE-COTE, Proximity Forest, and TS-CHIEF-exhibit high computational complexity, coupled with lengthy parameter tuning and training cycles. In contrast, lightweight solutions like ROCKET (Random Convolutional Kernel Transform) offer greater efficiency but leave substantial room for improvement in kernel selection and computational overhead. To address these challenges, we propose a feature extraction approach based on Hadamard convolutional transform, utilizing column or row vectors of Hadamard matrices as convolution kernels with extended lengths of varying sizes. This enhancement maintains full compatibility with existing methods (e.g., ROCKET) while leveraging kernel orthogonality to boost computational efficiency, robustness, and adaptability. Comprehensive experiments on multi-domain datasets-focusing on the UCR time series dataset-demonstrate SOTA performance: F1-score improved by at least 5% vs. ROCKET, with 50% shorter training time than miniROCKET (fastest ROCKET variant) under identical hyperparameters, enabling deployment on ultra-low-power embedded devices. All code is available on GitHub. 

---
# Real-time Continual Learning on Intel Loihi 2 

**Authors**: Elvin Hajizada, Danielle Rager, Timothy Shea, Leobardo Campos-Macias, Andreas Wild, Eyke Hüllermeier, Yulia Sandamirskaya, Mike Davies  

**Link**: [PDF](https://arxiv.org/pdf/2511.01553)  

**Abstract**: AI systems on edge devices face a critical challenge in open-world environments: adapting when data distributions shift and novel classes emerge. While offline training dominates current paradigms, online continual learning (OCL)--where models learn incrementally from non-stationary streams without catastrophic forgetting--remains challenging in power-constrained settings. We present a neuromorphic solution called CLP-SNN: a spiking neural network architecture for Continually Learning Prototypes and its implementation on Intel's Loihi 2 chip. Our approach introduces three innovations: (1) event-driven and spatiotemporally sparse local learning, (2) a self-normalizing three-factor learning rule maintaining weight normalization, and (3) integrated neurogenesis and metaplasticity for capacity expansion and forgetting mitigation. On OpenLORIS few-shot learning experiments, CLP-SNN achieves accuracy competitive with replay methods while being rehearsal-free. CLP-SNN delivers transformative efficiency gains: 70\times faster (0.33ms vs 23.2ms), and 5,600\times more energy efficient (0.05mJ vs 281mJ) than the best alternative OCL on edge GPU. This demonstrates that co-designed brain-inspired algorithms and neuromorphic hardware can break traditional accuracy-efficiency trade-offs for future edge AI systems. 

---
# Driving scenario generation and evaluation using a structured layer representation and foundational models 

**Authors**: Arthur Hubert, Gamal Elghazaly, Raphaël Frank  

**Link**: [PDF](https://arxiv.org/pdf/2511.01541)  

**Abstract**: Rare and challenging driving scenarios are critical for autonomous vehicle development. Since they are difficult to encounter, simulating or generating them using generative models is a popular approach. Following previous efforts to structure driving scenario representations in a layer model, we propose a structured five-layer model to improve the evaluation and generation of rare scenarios. We use this model alongside large foundational models to generate new driving scenarios using a data augmentation strategy. Unlike previous representations, our structure introduces subclasses and characteristics for every agent of the scenario, allowing us to compare them using an embedding specific to our layer-model. We study and adapt two metrics to evaluate the relevance of a synthetic dataset in the context of a structured representation: the diversity score estimates how different the scenarios of a dataset are from one another, while the originality score calculates how similar a synthetic dataset is from a real reference set. This paper showcases both metrics in different generation setup, as well as a qualitative evaluation of synthetic videos generated from structured scenario descriptions. The code and extended results can be found at this https URL. 

---
# BanglaNirTox: A Large-scale Parallel Corpus for Explainable AI in Bengali Text Detoxification 

**Authors**: Ayesha Afroza Mohsin, Mashrur Ahsan, Nafisa Maliyat, Shanta Maria, Syed Rifat Raiyan, Hasan Mahmud, Md Kamrul Hasan  

**Link**: [PDF](https://arxiv.org/pdf/2511.01512)  

**Abstract**: Toxic language in Bengali remains prevalent, especially in online environments, with few effective precautions against it. Although text detoxification has seen progress in high-resource languages, Bengali remains underexplored due to limited resources. In this paper, we propose a novel pipeline for Bengali text detoxification that combines Pareto class-optimized large language models (LLMs) and Chain-of-Thought (CoT) prompting to generate detoxified sentences. To support this effort, we construct BanglaNirTox, an artificially generated parallel corpus of 68,041 toxic Bengali sentences with class-wise toxicity labels, reasonings, and detoxified paraphrases, using Pareto-optimized LLMs evaluated on random samples. The resulting BanglaNirTox dataset is used to fine-tune language models to produce better detoxified versions of Bengali sentences. Our findings show that Pareto-optimized LLMs with CoT prompting significantly enhance the quality and consistency of Bengali text detoxification. 

---
# MO-SeGMan: Rearrangement Planning Framework for Multi Objective Sequential and Guided Manipulation in Constrained Environments 

**Authors**: Cankut Bora Tuncer, Marc Toussaint, Ozgur S. Oguz  

**Link**: [PDF](https://arxiv.org/pdf/2511.01476)  

**Abstract**: In this work, we introduce MO-SeGMan, a Multi-Objective Sequential and Guided Manipulation planner for highly constrained rearrangement problems. MO-SeGMan generates object placement sequences that minimize both replanning per object and robot travel distance while preserving critical dependency structures with a lazy evaluation method. To address highly cluttered, non-monotone scenarios, we propose a Selective Guided Forward Search (SGFS) that efficiently relocates only critical obstacles and to feasible relocation points. Furthermore, we adopt a refinement method for adaptive subgoal selection to eliminate unnecessary pick-and-place actions, thereby improving overall solution quality. Extensive evaluations on nine benchmark rearrangement tasks demonstrate that MO-SeGMan generates feasible motion plans in all cases, consistently achieving faster solution times and superior solution quality compared to the baselines. These results highlight the robustness and scalability of the proposed framework for complex rearrangement planning problems. 

---
# DAMBench: A Multi-Modal Benchmark for Deep Learning-based Atmospheric Data Assimilation 

**Authors**: Hao Wang, Zixuan Weng, Jindong Han, Wei Fan, Hao Liu  

**Link**: [PDF](https://arxiv.org/pdf/2511.01468)  

**Abstract**: Data Assimilation is a cornerstone of atmospheric system modeling, tasked with reconstructing system states by integrating sparse, noisy observations with prior estimation. While traditional approaches like variational and ensemble Kalman filtering have proven effective, recent advances in deep learning offer more scalable, efficient, and flexible alternatives better suited for complex, real-world data assimilation involving large-scale and multi-modal observations. However, existing deep learning-based DA research suffers from two critical limitations: (1) reliance on oversimplified scenarios with synthetically perturbed observations, and (2) the absence of standardized benchmarks for fair model comparison. To address these gaps, in this work, we introduce DAMBench, the first large-scale multi-modal benchmark designed to evaluate data-driven DA models under realistic atmospheric conditions. DAMBench integrates high-quality background states from state-of-the-art forecasting systems and real-world multi-modal observations (i.e., real-world weather stations and satellite imagery). All data are resampled to a common grid and temporally aligned to support systematic training, validation, and testing. We provide unified evaluation protocols and benchmark representative data assimilation approaches, including latent generative models and neural process frameworks. Additionally, we propose a lightweight multi-modal plugin to demonstrate how integrating realistic observations can enhance even simple baselines. Through comprehensive experiments, DAMBench establishes a rigorous foundation for future research, promoting reproducibility, fair comparison, and extensibility to real-world multi-modal scenarios. Our dataset and code are publicly available at this https URL. 

---
# HMVLM: Human Motion-Vision-Lanuage Model via MoE LoRA 

**Authors**: Lei Hu, Yongjing Ye, Shihong Xia  

**Link**: [PDF](https://arxiv.org/pdf/2511.01463)  

**Abstract**: The expansion of instruction-tuning data has enabled foundation language models to exhibit improved instruction adherence and superior performance across diverse downstream tasks. Semantically-rich 3D human motion is being progressively integrated with these foundation models to enhance multimodal understanding and cross-modal generation capabilities. However, the modality gap between human motion and text raises unresolved concerns about catastrophic forgetting during this integration. In addition, developing autoregressive-compatible pose representations that preserve generalizability across heterogeneous downstream tasks remains a critical technical barrier. To address these issues, we propose the Human Motion-Vision-Language Model (HMVLM), a unified framework based on the Mixture of Expert Low-Rank Adaption(MoE LoRA) strategy. The framework leverages the gating network to dynamically allocate LoRA expert weights based on the input prompt, enabling synchronized fine-tuning of multiple tasks. To mitigate catastrophic forgetting during instruction-tuning, we introduce a novel zero expert that preserves the pre-trained parameters for general linguistic tasks. For pose representation, we implement body-part-specific tokenization by partitioning the human body into different joint groups, enhancing the spatial resolution of the representation. Experiments show that our method effectively alleviates knowledge forgetting during instruction-tuning and achieves remarkable performance across diverse human motion downstream tasks. 

---
# Efficiently Training A Flat Neural Network Before It has been Quantizated 

**Authors**: Peng Xia, Junbiao Pang, Tianyang Cai  

**Link**: [PDF](https://arxiv.org/pdf/2511.01462)  

**Abstract**: Post-training quantization (PTQ) for vision transformers (ViTs) has garnered significant attention due to its efficiency in compressing models. However, existing methods typically overlook the relationship between a well-trained NN and the quantized model, leading to considerable quantization error for PTQ. However, it is unclear how to efficiently train a model-agnostic neural network which is tailored for a predefined precision low-bit model. In this paper, we firstly discover that a flat full precision neural network is crucial for low-bit quantization. To achieve this, we propose a framework that proactively pre-conditions the model by measuring and disentangling the error sources. Specifically, both the Activation Quantization Error (AQE) and the Weight Quantization Error (WQE) are statistically modeled as independent Gaussian noises. We study several noise injection optimization methods to obtain a flat minimum. Experimental results attest to the effectiveness of our approach. These results open novel pathways for obtaining low-bit PTQ models. 

---
# When to Trust the Answer: Question-Aligned Semantic Nearest Neighbor Entropy for Safer Surgical VQA 

**Authors**: Dennis Pierantozzi, Luca Carlini, Mauro Orazio Drago, Chiara Lena, Cesare Hassan, Elena De Momi, Danail Stoyanov, Sophia Bano, Mobarak I. Hoque  

**Link**: [PDF](https://arxiv.org/pdf/2511.01458)  

**Abstract**: Safety and reliability are essential for deploying Visual Question Answering (VQA) in surgery, where incorrect or ambiguous responses can harm the patient. Most surgical VQA research focuses on accuracy or linguistic quality while overlooking safety behaviors such as ambiguity awareness, referral to human experts, or triggering a second opinion. Inspired by Automatic Failure Detection (AFD), we study uncertainty estimation as a key enabler of safer decision making. We introduce Question Aligned Semantic Nearest Neighbor Entropy (QA-SNNE), a black box uncertainty estimator that incorporates question semantics into prediction confidence. It measures semantic entropy by comparing generated answers with nearest neighbors in a medical text embedding space, conditioned on the question. We evaluate five models, including domain specific Parameter-Efficient Fine-Tuned (PEFT) models and zero-shot Large Vision-Language Models (LVLMs), on EndoVis18-VQA and PitVQA. PEFT models degrade under mild paraphrasing, while LVLMs are more resilient. Across three LVLMs and two PEFT baselines, QA-SNNE improves AUROC in most in-template settings and enhances hallucination detection. The Area Under the ROC Curve (AUROC) increases by 15-38% for zero-shot models, with gains maintained under out-of-template stress. QA-SNNE offers a practical and interpretable step toward AFD in surgical VQA by linking semantic uncertainty to question context. Combining LVLM backbones with question aligned uncertainty estimation can improve safety and clinician trust. The code and model are available at this https URL 

---
# Reg-DPO: SFT-Regularized Direct Preference Optimization with GT-Pair for Improving Video Generation 

**Authors**: Jie Du, Xinyu Gong, Qingshan Tan, Wen Li, Yangming Cheng, Weitao Wang, Chenlu Zhan, Suhui Wu, Hao Zhang, Jun Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2511.01450)  

**Abstract**: Recent studies have identified Direct Preference Optimization (DPO) as an efficient and reward-free approach to improving video generation quality. However, existing methods largely follow image-domain paradigms and are mainly developed on small-scale models (approximately 2B parameters), limiting their ability to address the unique challenges of video tasks, such as costly data construction, unstable training, and heavy memory consumption. To overcome these limitations, we introduce a GT-Pair that automatically builds high-quality preference pairs by using real videos as positives and model-generated videos as negatives, eliminating the need for any external annotation. We further present Reg-DPO, which incorporates the SFT loss as a regularization term into the DPO objective to enhance training stability and generation fidelity. Additionally, by combining the FSDP framework with multiple memory optimization techniques, our approach achieves nearly three times higher training capacity than using FSDP alone. Extensive experiments on both I2V and T2V tasks across multiple datasets demonstrate that our method consistently outperforms existing approaches, delivering superior video generation quality. 

---
# Privacy Preserving Ordinal-Meta Learning with VLMs for Fine-Grained Fruit Quality Prediction 

**Authors**: Riddhi Jain, Manasi Patwardhan, Aayush Mishra, Parijat Deshpande, Beena Rai  

**Link**: [PDF](https://arxiv.org/pdf/2511.01449)  

**Abstract**: To effectively manage the wastage of perishable fruits, it is crucial to accurately predict their freshness or shelf life using non-invasive methods that rely on visual data. In this regard, deep learning techniques can offer a viable solution. However, obtaining fine-grained fruit freshness labels from experts is costly, leading to a scarcity of data. Closed proprietary Vision Language Models (VLMs), such as Gemini, have demonstrated strong performance in fruit freshness detection task in both zero-shot and few-shot settings. Nonetheless, food retail organizations are unable to utilize these proprietary models due to concerns related to data privacy, while existing open-source VLMs yield sub-optimal performance for the task. Fine-tuning these open-source models with limited data fails to achieve the performance levels of proprietary models. In this work, we introduce a Model-Agnostic Ordinal Meta-Learning (MAOML) algorithm, designed to train smaller VLMs. This approach utilizes meta-learning to address data sparsity and leverages label ordinality, thereby achieving state-of-the-art performance in the fruit freshness classification task under both zero-shot and few-shot settings. Our method achieves an industry-standard accuracy of 92.71%, averaged across all fruits.
Keywords: Fruit Quality Prediction, Vision Language Models, Meta Learning, Ordinal Regression 

---
# UniSOT: A Unified Framework for Multi-Modality Single Object Tracking 

**Authors**: Yinchao Ma, Yuyang Tang, Wenfei Yang, Tianzhu Zhang, Xu Zhou, Feng Wu  

**Link**: [PDF](https://arxiv.org/pdf/2511.01427)  

**Abstract**: Single object tracking aims to localize target object with specific reference modalities (bounding box, natural language or both) in a sequence of specific video modalities (RGB, RGB+Depth, RGB+Thermal or RGB+Event.). Different reference modalities enable various human-machine interactions, and different video modalities are demanded in complex scenarios to enhance tracking robustness. Existing trackers are designed for single or several video modalities with single or several reference modalities, which leads to separate model designs and limits practical applications. Practically, a unified tracker is needed to handle various requirements. To the best of our knowledge, there is still no tracker that can perform tracking with these above reference modalities across these video modalities simultaneously. Thus, in this paper, we present a unified tracker, UniSOT, for different combinations of three reference modalities and four video modalities with uniform parameters. Extensive experimental results on 18 visual tracking, vision-language tracking and RGB+X tracking benchmarks demonstrate that UniSOT shows superior performance against modality-specific counterparts. Notably, UniSOT outperforms previous counterparts by over 3.0\% AUC on TNL2K across all three reference modalities and outperforms Un-Track by over 2.0\% main metric across all three RGB+X video modalities. 

---
# FoldPath: End-to-End Object-Centric Motion Generation via Modulated Implicit Paths 

**Authors**: Paolo Rabino, Gabriele Tiboni, Tatiana Tommasi  

**Link**: [PDF](https://arxiv.org/pdf/2511.01407)  

**Abstract**: Object-Centric Motion Generation (OCMG) is instrumental in advancing automated manufacturing processes, particularly in domains requiring high-precision expert robotic motions, such as spray painting and welding. To realize effective automation, robust algorithms are essential for generating extended, object-aware trajectories across intricate 3D geometries. However, contemporary OCMG techniques are either based on ad-hoc heuristics or employ learning-based pipelines that are still reliant on sensitive post-processing steps to generate executable paths. We introduce FoldPath, a novel, end-to-end, neural field based method for OCMG. Unlike prior deep learning approaches that predict discrete sequences of end-effector waypoints, FoldPath learns the robot motion as a continuous function, thus implicitly encoding smooth output paths. This paradigm shift eliminates the need for brittle post-processing steps that concatenate and order the predicted discrete waypoints. Particularly, our approach demonstrates superior predictive performance compared to recently proposed learning-based methods, and attains generalization capabilities even in real industrial settings, where only a limited amount of 70 expert samples are provided. We validate FoldPath through comprehensive experiments in a realistic simulation environment and introduce new, rigorous metrics designed to comprehensively evaluate long-horizon robotic paths, thus advancing the OCMG task towards practical maturity. 

---
# SEPS: Semantic-enhanced Patch Slimming Framework for fine-grained cross-modal alignment 

**Authors**: Xinyu Mao, Junsi Li, Haoji Zhang, Yu Liang, Ming Sun  

**Link**: [PDF](https://arxiv.org/pdf/2511.01390)  

**Abstract**: Fine-grained cross-modal alignment aims to establish precise local correspondences between vision and language, forming a cornerstone for visual question answering and related multimodal applications. Current approaches face challenges in addressing patch redundancy and ambiguity, which arise from the inherent information density disparities across modalities. Recently, Multimodal Large Language Models (MLLMs) have emerged as promising solutions to bridge this gap through their robust semantic generation capabilities. However, the dense textual outputs from MLLMs may introduce conflicts with the original sparse captions. Furthermore, accurately quantifying semantic relevance between rich visual patches and concise textual descriptions remains a core challenge. To overcome these limitations, we introduce the Semantic-Enhanced Patch Slimming (SEPS) framework, which systematically addresses patch redundancy and ambiguity. Our approach employs a two-stage mechanism to integrate unified semantics from both dense and sparse texts, enabling the identification of salient visual patches. Additionally, it leverages relevance-aware selection with mean value computation to highlight crucial patch-word correspondences, thereby improving cross-modal similarity assessment. Comprehensive experiments on Flickr30K and MS-COCO datasets validate that SEPS achieves superior performance, surpassing existing approaches by 23\%-86\% in rSum across diverse model architectures, with notable enhancements in text-to-image retrieval scenarios. Our implementation is available at this https URL. 

---
# RAGSmith: A Framework for Finding the Optimal Composition of Retrieval-Augmented Generation Methods Across Datasets 

**Authors**: Muhammed Yusuf Kartal, Suha Kagan Kose, Korhan Sevinç, Burak Aktas  

**Link**: [PDF](https://arxiv.org/pdf/2511.01386)  

**Abstract**: Retrieval-Augmented Generation (RAG) quality depends on many interacting choices across retrieval, ranking, augmentation, prompting, and generation, so optimizing modules in isolation is brittle. We introduce RAGSmith, a modular framework that treats RAG design as an end-to-end architecture search over nine technique families and 46{,}080 feasible pipeline configurations. A genetic search optimizes a scalar objective that jointly aggregates retrieval metrics (recall@k, mAP, nDCG, MRR) and generation metrics (LLM-Judge and semantic similarity). We evaluate on six Wikipedia-derived domains (Mathematics, Law, Finance, Medicine, Defense Industry, Computer Science), each with 100 questions spanning factual, interpretation, and long-answer types. RAGSmith finds configurations that consistently outperform naive RAG baseline by +3.8\% on average (range +1.2\% to +6.9\% across domains), with gains up to +12.5\% in retrieval and +7.5\% in generation. The search typically explores $\approx 0.2\%$ of the space ($\sim 100$ candidates) and discovers a robust backbone -- vector retrieval plus post-generation reflection/revision -- augmented by domain-dependent choices in expansion, reranking, augmentation, and prompt reordering; passage compression is never selected. Improvement magnitude correlates with question type, with larger gains on factual/long-answer mixes than interpretation-heavy sets. These results provide practical, domain-aware guidance for assembling effective RAG systems and demonstrate the utility of evolutionary search for full-pipeline optimization. 

---
# PrefixNLI: Detecting Factual Inconsistencies as Soon as They Arise 

**Authors**: Sapir Harary, Eran Hirsch, Aviv Slobodkin, David Wan, Mohit Bansal, Ido Dagan  

**Link**: [PDF](https://arxiv.org/pdf/2511.01359)  

**Abstract**: Natural Language Inference (NLI) models have been used in various ways to improve the factuality of LLM outputs. This is typically done by applying an NLI model to judge whether the model output is entailed from the supposed evidence, triggering some corrective actions, such as beam reranking at inference time or RL rewards during training. While NLI models are trained to detect factual inconsistencies over complete sentences, decisions in the common autoregressive generation architecture are made for each evolving text prefix, during decoding. Addressing this setting, we generalize the entailment detection task to apply over arbitrary text prefixes, and suggest its utility for improving generation faithfulness. Providing suitable evaluation and training datasets for this task, we train MiniTruePrefixes, a novel specialized model that better detects factual inconsistencies over text prefixes, outperforming comparable baseline NLI models by 5-14 F1 points in prefix-level entailment. We further demonstrate that integrating MiniTruePrefixes into a controlled decoding framework substantially improves factual consistency in abstractive summarization. When guided by MiniTruePrefixes, LLaMA-3.2-3B-Instruct matches the faithfulness and runtime of the 8B model from the same model family, while using only half the memory. 

---
# CMI-MTL: Cross-Mamba interaction based multi-task learning for medical visual question answering 

**Authors**: Qiangguo Jin, Xianyao Zheng, Hui Cui, Changming Sun, Yuqi Fang, Cong Cong, Ran Su, Leyi Wei, Ping Xuan, Junbo Wang  

**Link**: [PDF](https://arxiv.org/pdf/2511.01357)  

**Abstract**: Medical visual question answering (Med-VQA) is a crucial multimodal task in clinical decision support and telemedicine. Recent self-attention based methods struggle to effectively handle cross-modal semantic alignments between vision and language. Moreover, classification-based methods rely on predefined answer sets. Treating this task as a simple classification problem may make it unable to adapt to the diversity of free-form answers and overlook the detailed semantic information of free-form answers. In order to tackle these challenges, we introduce a Cross-Mamba Interaction based Multi-Task Learning (CMI-MTL) framework that learns cross-modal feature representations from images and texts. CMI-MTL comprises three key modules: fine-grained visual-text feature alignment (FVTA), cross-modal interleaved feature representation (CIFR), and free-form answer-enhanced multi-task learning (FFAE). FVTA extracts the most relevant regions in image-text pairs through fine-grained visual-text feature alignment. CIFR captures cross-modal sequential interactions via cross-modal interleaved feature representation. FFAE leverages auxiliary knowledge from open-ended questions through free-form answer-enhanced multi-task learning, improving the model's capability for open-ended Med-VQA. Experimental results show that CMI-MTL outperforms the existing state-of-the-art methods on three Med-VQA datasets: VQA-RAD, SLAKE, and OVQA. Furthermore, we conduct more interpretability experiments to prove the effectiveness. The code is publicly available at this https URL. 

---
# Thinking with DistilQwen: A Tale of Four Distilled Reasoning and Reward Model Series 

**Authors**: Wenrui Cai, Chengyu Wang, Junbing Yan, Jun Huang, Xiangzhong Fang  

**Link**: [PDF](https://arxiv.org/pdf/2511.01354)  

**Abstract**: Recently, the demand for small and efficient reasoning models to support real-world applications has driven the development of knowledge distillation techniques that balance reasoning performance and inference speed. In this paper, we further extend the DistilQwen model family, initialized from the Qwen models, by introducing four model series specifically designed to meet industrial requirements. The distilled model collection comprises: (1) slow-thinking models, optimized for reasoning tasks that require high accuracy; (2) two series of adaptive-thinking models, which dynamically adjust reasoning strategies based on input tasks to maximize efficiency across diverse scenarios; and (3) distilled reward models, which enable further reinforcement learning of reasoning models using distilled knowledge. Comprehensive evaluations across multiple benchmarks demonstrate both high inference efficiency and strong reasoning performance for these models, as well as the practical utility of distilled reward models. We further show that these models support industry practitioners by providing scalable training and inference functionalities on the Alibaba Cloud PAI (Platform for Artificial Intelligence) platform. 

---
# AI Literacy in UAE Libraries: Assessing Competencies, Training Needs, and Ethical Considerations for the Digital Age 

**Authors**: Zafar Imam Khan  

**Link**: [PDF](https://arxiv.org/pdf/2511.01353)  

**Abstract**: The study explores the current state of artificial intelligence (AI) literacy levels among library professionals employing a quantitative approach consisting of 92 surveys of LIS professionals in the United Arab Emirates (UAE). Findings of the study revealed the presence of strong cognitive competencies, while there were gaps observed in behavioral and normative competencies, especially related to AI biases, AI-powered learning, and ethical considerations. There was a disconnect observed between the perceived importance of AI skills and the effectiveness of the current training programs. 

---
# The Future of Generative AI in Software Engineering: A Vision from Industry and Academia in the European GENIUS Project 

**Authors**: Robin Gröpler, Steffen Klepke, Jack Johns, Andreas Dreschinski, Klaus Schmid, Benedikt Dornauer, Eray Tüzün, Joost Noppen, Mohammad Reza Mousavi, Yongjian Tang, Johannes Viehmann, Selin Şirin Aslangül, Beum Seuk Lee, Adam Ziolkowski, Eric Zie  

**Link**: [PDF](https://arxiv.org/pdf/2511.01348)  

**Abstract**: Generative AI (GenAI) has recently emerged as a groundbreaking force in Software Engineering, capable of generating code, suggesting fixes, and supporting quality assurance. While its use in coding tasks shows considerable promise, applying GenAI across the entire Software Development Life Cycle (SDLC) has not yet been fully explored. Critical uncertainties in areas such as reliability, accountability, security, and data privacy demand deeper investigation and coordinated action. The GENIUS project, comprising over 30 European industrial and academic partners, aims to address these challenges by advancing AI integration across all SDLC phases. It focuses on GenAI's potential, the development of innovative tools, and emerging research challenges, actively shaping the future of software engineering. This vision paper presents a shared perspective on the future of GenAI-based software engineering, grounded in cross-sector dialogue and experience within the GENIUS consortium, supported by an exploratory literature review. The paper explores four central elements: (1) a structured overview of current challenges in GenAI adoption across the SDLC; (2) a forward-looking vision outlining key technological and methodological advances expected over the next five years; (3) anticipated shifts in the roles and required skill sets of software professionals; and (4) the contribution of GENIUS in realizing this transformation through practical tools and industrial validation. By aligning technical innovation with business relevance, this paper aims to inform both research agendas and industrial strategies, providing a foundation for reliable, scalable, and industry-ready GenAI solutions for software engineering teams. 

---
# Beyond Permissions: Investigating Mobile Personalization with Simulated Personas 

**Authors**: Ibrahim Khalilov, Chaoran Chen, Ziang Xiao, Tianshi Li, Toby Jia-Jun Li, Yaxing Yao  

**Link**: [PDF](https://arxiv.org/pdf/2511.01336)  

**Abstract**: Mobile applications increasingly rely on sensor data to infer user context and deliver personalized experiences. Yet the mechanisms behind this personalization remain opaque to users and researchers alike. This paper presents a sandbox system that uses sensor spoofing and persona simulation to audit and visualize how mobile apps respond to inferred behaviors. Rather than treating spoofing as adversarial, we demonstrate its use as a tool for behavioral transparency and user empowerment. Our system injects multi-sensor profiles - generated from structured, lifestyle-based personas - into Android devices in real time, enabling users to observe app responses to contexts such as high activity, location shifts, or time-of-day changes. With automated screenshot capture and GPT-4 Vision-based UI summarization, our pipeline helps document subtle personalization cues. Preliminary findings show measurable app adaptations across fitness, e-commerce, and everyday service apps such as weather and navigation. We offer this toolkit as a foundation for privacy-enhancing technologies and user-facing transparency interventions. 

---
# Embodied Cognition Augmented End2End Autonomous Driving 

**Authors**: Ling Niu, Xiaoji Zheng, Han Wang, Chen Zheng, Ziyuan Yang, Bokui Chen, Jiangtao Gong  

**Link**: [PDF](https://arxiv.org/pdf/2511.01334)  

**Abstract**: In recent years, vision-based end-to-end autonomous driving has emerged as a new paradigm. However, popular end-to-end approaches typically rely on visual feature extraction networks trained under label supervision. This limited supervision framework restricts the generality and applicability of driving models. In this paper, we propose a novel paradigm termed $E^{3}AD$, which advocates for comparative learning between visual feature extraction networks and the general EEG large model, in order to learn latent human driving cognition for enhancing end-to-end planning. In this work, we collected a cognitive dataset for the mentioned contrastive learning process. Subsequently, we investigated the methods and potential mechanisms for enhancing end-to-end planning with human driving cognition, using popular driving models as baselines on publicly available autonomous driving datasets. Both open-loop and closed-loop tests are conducted for a comprehensive evaluation of planning performance. Experimental results demonstrate that the $E^{3}AD$ paradigm significantly enhances the end-to-end planning performance of baseline models. Ablation studies further validate the contribution of driving cognition and the effectiveness of comparative learning process. To the best of our knowledge, this is the first work to integrate human driving cognition for improving end-to-end autonomous driving planning. It represents an initial attempt to incorporate embodied cognitive data into end-to-end autonomous driving, providing valuable insights for future brain-inspired autonomous driving systems. Our code will be made available at Github 

---
# AI for Requirements Engineering: Industry adoption and Practitioner perspectives 

**Authors**: Lekshmi Murali Rani, Richard Berntsson Svensson, Robert Feldt  

**Link**: [PDF](https://arxiv.org/pdf/2511.01324)  

**Abstract**: The integration of AI for Requirements Engineering (RE) presents significant benefits but also poses real this http URL RE is fundamental to software engineering, limited research has examined AI adoption in this http URL surveyed 55 software practitioners to map AI usage across four RE phases:Elicitation, Analysis, Specification, and Validation, and four approaches for decision making: human only decisions, AI validation, Human AI Collaboration (HAIC), and full AI this http URL also shared their perceptions, challenges, and opportunities when applying AI for RE this http URL data show that 58.2% of respondents already use AI in RE, and 69.1% view its impact as positive or very this http URL dominates practice, accounting for 54.4% of all RE techniques, while full AI automation remains minimal at 5.4%.Passive AI validation (4.4 to 6.2%) lags even further behind, indicating that practitioners value AI's active support over passive this http URL findings suggest that AI is most effective when positioned as a collaborative partner rather than a replacement for human this http URL also highlights the need for RE specific HAIC frameworks along with robust and responsible AI governance as AI adoption in RE grows. 

---
# DEEPAMBIGQA: Ambiguous Multi-hop Questions for Benchmarking LLM Answer Completeness 

**Authors**: Jiabao Ji, Min Li, Priyanshu Kumar, Shiyu Chang, Saloni Potdar  

**Link**: [PDF](https://arxiv.org/pdf/2511.01323)  

**Abstract**: Large language models (LLMs) with integrated search tools show strong promise in open-domain question answering (QA), yet they often struggle to produce complete answer set to complex questions such as Which actor from the film Heat won at least one Academy Award?, which requires (1) distinguishing between multiple films sharing the same title and (2) reasoning across a large set of actors to gather and integrate evidence. Existing QA benchmarks rarely evaluate both challenges jointly. To address this, we introduce DeepAmbigQAGen, an automatic data generation pipeline that constructs QA tasks grounded in text corpora and linked knowledge graph, generating natural and verifiable questions that systematically embed name ambiguity and multi-step reasoning. Based on this, we build DeepAmbigQA, a dataset of 3,600 questions requiring multi-hop reasoning and half of them explicit name ambiguity resolving. Experiments reveal that, even state-of-the-art GPT-5 show incomplete answers, achieving only 0.13 exact match on ambiguous questions and 0.21 on non-ambiguous questions. These findings highlight the need for more robust QA systems aimed at information gathering and answer completeness. 

---
# Exploringand Unleashing the Power of Large Language Models in CI/CD Configuration Translation 

**Authors**: Chong Wang, Chen Zhang, Jiajun Wu, Wunan Guo, Jianfeng Qu, Yewen Tian, Yang Liu  

**Link**: [PDF](https://arxiv.org/pdf/2511.01316)  

**Abstract**: Continuous Integration (CI) is a cornerstone of modern collaborative software development, and numerous CI platforms are available. Differences in maintenance overhead, reliability, and integration depth with code-hosting platforms make migration between CI platforms a common practice. A central step in migration is translating CI configurations, which is challenging due to the intrinsic complexity of CI configurations and the need to understand semantic differences and relationships across CI platforms.
With the advent of large language models (LLMs), recent advances in software engineering highlight their potential for CI configuration translation. In this paper, we present a study on LLM-based CI configuration translation, focusing on the migration from Travis CI to GitHub Actions. First, using 811 migration records, we quantify the effort involved and find that developers read an average of 38 lines of Travis configuration and write 58 lines of GitHub Actions configuration, with nearly half of the migrations requiring multiple commits. We further analyze translations produced by each of the four LLMs and identify 1,121 issues grouped into four categories: logic inconsistencies (38%), platform discrepancies (32%), environment errors (25%), and syntax errors (5%). Finally, we evaluate three enhancement strategies and show that combining guideline-based prompting with iterative refinement achieves the best performance, reaching a Build Success Rate of 75.5%-nearly a threefold improvement over GPT-4o with a basic prompt. 

---
# Perturb a Model, Not an Image: Towards Robust Privacy Protection via Anti-Personalized Diffusion Models 

**Authors**: Tae-Young Lee, Juwon Seo, Jong Hwan Ko, Gyeong-Moon Park  

**Link**: [PDF](https://arxiv.org/pdf/2511.01307)  

**Abstract**: Recent advances in diffusion models have enabled high-quality synthesis of specific subjects, such as identities or objects. This capability, while unlocking new possibilities in content creation, also introduces significant privacy risks, as personalization techniques can be misused by malicious users to generate unauthorized content. Although several studies have attempted to counter this by generating adversarially perturbed samples designed to disrupt personalization, they rely on unrealistic assumptions and become ineffective in the presence of even a few clean images or under simple image transformations. To address these challenges, we shift the protection target from the images to the diffusion model itself to hinder the personalization of specific subjects, through our novel framework called Anti-Personalized Diffusion Models (APDM). We first provide a theoretical analysis demonstrating that a naive approach of existing loss functions to diffusion models is inherently incapable of ensuring convergence for robust anti-personalization. Motivated by this finding, we introduce Direct Protective Optimization (DPO), a novel loss function that effectively disrupts subject personalization in the target model without compromising generative quality. Moreover, we propose a new dual-path optimization strategy, coined Learning to Protect (L2P). By alternating between personalization and protection paths, L2P simulates future personalization trajectories and adaptively reinforces protection at each step. Experimental results demonstrate that our framework outperforms existing methods, achieving state-of-the-art performance in preventing unauthorized personalization. The code is available at this https URL. 

---
# DeepSpecs: Expert-Level Questions Answering in 5G 

**Authors**: Aman Ganapathy Manvattira, Yifei Xu, Ziyue Dang, Songwu Lu  

**Link**: [PDF](https://arxiv.org/pdf/2511.01305)  

**Abstract**: 5G technology enables mobile Internet access for billions of users. Answering expert-level questions about 5G specifications requires navigating thousands of pages of cross-referenced standards that evolve across releases. Existing retrieval-augmented generation (RAG) frameworks, including telecom-specific approaches, rely on semantic similarity and cannot reliably resolve cross-references or reason about specification evolution. We present DeepSpecs, a RAG system enhanced by structural and temporal reasoning via three metadata-rich databases: SpecDB (clause-aligned specification text), ChangeDB (line-level version diffs), and TDocDB (standardization meeting documents). DeepSpecs explicitly resolves cross-references by recursively retrieving referenced clauses through metadata lookup, and traces specification evolution by mining changes and linking them to Change Requests that document design rationale. We curate two 5G QA datasets: 573 expert-annotated real-world questions from practitioner forums and educational resources, and 350 evolution-focused questions derived from approved Change Requests. Across multiple LLM backends, DeepSpecs outperforms base models and state-of-the-art telecom RAG systems; ablations confirm that explicit cross-reference resolution and evolution-aware retrieval substantially improve answer quality, underscoring the value of modeling the structural and temporal properties of 5G standards. 

---
# LSHFed: Robust and Communication-Efficient Federated Learning with Locally-Sensitive Hashing Gradient Mapping 

**Authors**: Guanjie Cheng, Mengzhen Yang, Xinkui Zhao, Shuyi Yu, Tianyu Du, Yangyang Wu, Mengying Zhu, Shuiguang Deng  

**Link**: [PDF](https://arxiv.org/pdf/2511.01296)  

**Abstract**: Federated learning (FL) enables collaborative model training across distributed nodes without exposing raw data, but its decentralized nature makes it vulnerable in trust-deficient environments. Inference attacks may recover sensitive information from gradient updates, while poisoning attacks can degrade model performance or induce malicious behaviors. Existing defenses often suffer from high communication and computation costs, or limited detection precision. To address these issues, we propose LSHFed, a robust and communication-efficient FL framework that simultaneously enhances aggregation robustness and privacy preservation. At its core, LSHFed incorporates LSHGM, a novel gradient verification mechanism that projects high-dimensional gradients into compact binary representations via multi-hyperplane locally-sensitive hashing. This enables accurate detection and filtering of malicious gradients using only their irreversible hash forms, thus mitigating privacy leakage risks and substantially reducing transmission overhead. Extensive experiments demonstrate that LSHFed maintains high model performance even when up to 50% of participants are collusive adversaries while achieving up to a 1000x reduction in gradient verification communication compared to full-gradient methods. 

---
# Adaptation of Foundation Models for Medical Image Analysis: Strategies, Challenges, and Future Directions 

**Authors**: Karma Phuntsho, Abdullah, Kyungmi Lee, Ickjai Lee, Euijoon Ahn  

**Link**: [PDF](https://arxiv.org/pdf/2511.01284)  

**Abstract**: Foundation models (FMs) have emerged as a transformative paradigm in medical image analysis, offering the potential to provide generalizable, task-agnostic solutions across a wide range of clinical tasks and imaging modalities. Their capacity to learn transferable representations from large-scale data has the potential to address the limitations of conventional task-specific models. However, adaptation of FMs to real-world clinical practice remains constrained by key challenges, including domain shifts, limited availability of high-quality annotated data, substantial computational demands, and strict privacy requirements. This review presents a comprehensive assessment of strategies for adapting FMs to the specific demands of medical imaging. We examine approaches such as supervised fine-tuning, domain-specific pretraining, parameter-efficient fine-tuning, self-supervised learning, hybrid methods, and multimodal or cross-modal frameworks. For each, we evaluate reported performance gains, clinical applicability, and limitations, while identifying trade-offs and unresolved challenges that prior reviews have often overlooked. Beyond these established techniques, we also highlight emerging directions aimed at addressing current gaps. These include continual learning to enable dynamic deployment, federated and privacy-preserving approaches to safeguard sensitive data, hybrid self-supervised learning to enhance data efficiency, data-centric pipelines that combine synthetic generation with human-in-the-loop validation, and systematic benchmarking to assess robust generalization under real-world clinical variability. By outlining these strategies and associated research gaps, this review provides a roadmap for developing adaptive, trustworthy, and clinically integrated FMs capable of meeting the demands of real-world medical imaging. 

---
# When, What, and How: Rethinking Retrieval-Enhanced Speculative Decoding 

**Authors**: Min Fang, Zhihui Fu, Qibin Zhao, Jun Wang  

**Link**: [PDF](https://arxiv.org/pdf/2511.01282)  

**Abstract**: Speculative decoding (SD) has emerged as an effective technique to accelerate large language model (LLM) inference without compromising output quality. However, the achievable speedup largely depends on the effectiveness of the drafting model. While model-based methods like EAGLE-2 are accurate but costly, retrieval-enhanced methods like SAM-Decoding rely on heuristic switching strategies that often trigger unnecessary retrievals. To address this, we propose ReSpec (\textbf{Re}trieval-enhanced \textbf{Spe}culative Decoding), a novel framework that transforms heuristic drafter switching into adaptive decision-making. ReSpec features three core innovations: 1) An \textbf{entropy-guided adaptive trigger} quantifies contextual predictability to initiate retrieval only when uncertainty is low, avoiding costly low-quality speculations. 2) A \textbf{feedback-driven candidate selection} leverages historical feedback to organize multiple high-quality candidates for parallel verification, maximizing retrieval utility. 3) A source-aware \textbf{relaxed verification strategy} applies strict checks to model-generated drafts while using a relaxed verification for retrieved drafts, achieving a better balance between accuracy and efficiency. Extensive experiments on Spec-Bench demonstrate that ReSpec achieves state-of-the-art acceleration,outperforming EAGLE-2 and SAM-Decoding by over $33\%$ and $25\%$, respectively, while maintaining output quality. 

---
# Adversarial Spatio-Temporal Attention Networks for Epileptic Seizure Forecasting 

**Authors**: Zan Li, Kyongmin Yeo, Wesley Gifford, Lara Marcuse, Madeline Fields, Bülent Yener  

**Link**: [PDF](https://arxiv.org/pdf/2511.01275)  

**Abstract**: Forecasting epileptic seizures from multivariate EEG signals represents a critical challenge in healthcare time series prediction, requiring high sensitivity, low false alarm rates, and subject-specific adaptability. We present STAN, an Adversarial Spatio-Temporal Attention Network that jointly models spatial brain connectivity and temporal neural dynamics through cascaded attention blocks with alternating spatial and temporal modules. Unlike existing approaches that assume fixed preictal durations or separately process spatial and temporal features, STAN captures bidirectional dependencies between spatial and temporal patterns through a unified cascaded architecture. Adversarial training with gradient penalty enables robust discrimination between interictal and preictal states learned from clearly defined 15-minute preictal windows. Continuous 90-minute pre-seizure monitoring reveals that the learned spatio-temporal attention patterns enable early detection: reliable alarms trigger at subject-specific times (typically 15-45 minutes before onset), reflecting the model's capacity to capture subtle preictal dynamics without requiring individualized training. Experiments on two benchmark EEG datasets (CHB-MIT scalp: 8 subjects, 46 events; MSSM intracranial: 4 subjects, 14 events) demonstrate state-of-the-art performance: 96.6% sensitivity with 0.011 false detections per hour and 94.2% sensitivity with 0.063 false detections per hour, respectively, while maintaining computational efficiency (2.3M parameters, 45 ms latency, 180 MB memory) for real-time edge deployment. Beyond epilepsy, the proposed framework provides a general paradigm for spatio-temporal forecasting in healthcare and other time series domains where individual heterogeneity and interpretability are crucial. 

---
# Rescuing the Unpoisoned: Efficient Defense against Knowledge Corruption Attacks on RAG Systems 

**Authors**: Minseok Kim, Hankook Lee, Hyungjoon Koo  

**Link**: [PDF](https://arxiv.org/pdf/2511.01268)  

**Abstract**: Large language models (LLMs) are reshaping numerous facets of our daily lives, leading widespread adoption as web-based services. Despite their versatility, LLMs face notable challenges, such as generating hallucinated content and lacking access to up-to-date information. Lately, to address such limitations, Retrieval-Augmented Generation (RAG) has emerged as a promising direction by generating responses grounded in external knowledge sources. A typical RAG system consists of i) a retriever that probes a group of relevant passages from a knowledge base and ii) a generator that formulates a response based on the retrieved content. However, as with other AI systems, recent studies demonstrate the vulnerability of RAG, such as knowledge corruption attacks by injecting misleading information. In response, several defense strategies have been proposed, including having LLMs inspect the retrieved passages individually or fine-tuning robust retrievers. While effective, such approaches often come with substantial computational costs.
In this work, we introduce RAGDefender, a resource-efficient defense mechanism against knowledge corruption (i.e., by data poisoning) attacks in practical RAG deployments. RAGDefender operates during the post-retrieval phase, leveraging lightweight machine learning techniques to detect and filter out adversarial content without requiring additional model training or inference. Our empirical evaluations show that RAGDefender consistently outperforms existing state-of-the-art defenses across multiple models and adversarial scenarios: e.g., RAGDefender reduces the attack success rate (ASR) against the Gemini model from 0.89 to as low as 0.02, compared to 0.69 for RobustRAG and 0.24 for Discern-and-Answer when adversarial passages outnumber legitimate ones by a factor of four (4x). 

---
# Speech-DRAME: A Framework for Human-Aligned Benchmarks in Speech Role-Play 

**Authors**: Jiatong Shi, Jionghao Han, Yichen Lu, Santiago Pascual, Pengfei Wu, Chenye Cui, Shinji Watanabe, Chao Weng, Cong Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2511.01261)  

**Abstract**: Role-play has become a key testbed for generative models, expanding from text-only dialogue to multimodal interaction. Extending role-play to speech captures prosody, emotion, and delivery, but also poses new evaluation challenges. Current pipelines often use audio large language models (ALLMs) as zero-shot judges, which miss paralinguistic cues, collapse multiple aspects into coarse scores, and rely on synthetic speech references that fail to reflect real-world roles. We present Speech-DRAME, a unified framework that contributes at three levels: (i) Speech-DRAME-EvalBench, an evaluation benchmark with bilingual human-annotated data and protocols for training and testing speech evaluation models (SEMs), (ii) DRAME-Eval, a fine-tuned evaluation model, which substantially outperforms zero-shot and few-shot ALLMs, and (iii) Speech-DRAME-RoleBench, a speech role-play benchmark that leverages DRAME-Eval as an automatic judge to compare speech foundation models (SFMs). Speech-DRAME distinguishes between two complementary evaluation strategies: Archetype Evaluation, a top-down approach measuring adherence to broad role archetypes, and Realism Evaluation, a bottom-up approach grounded in real human speech that emphasizes nuanced role quality. Compared to zero-shot ALLM judges, DRAME-Eval achieves stronger agreement with human ratings (Pearson correlation from 0.480 to 0.629 in archetypes, and 0.390 to 0.625 in realism). By integrating transparent benchmark resources, modeling approaches, and system-level evaluation, Speech-DRAME provides the first comprehensive, reproducible foundation for assessing spoken role-play. 

---
# Quantum Deep Learning Still Needs a Quantum Leap 

**Authors**: Hans Gundlach, Hrvoje Kukina, Jayson Lynch, Neil Thompson  

**Link**: [PDF](https://arxiv.org/pdf/2511.01253)  

**Abstract**: Quantum computing technology is advancing rapidly. Yet, even accounting for these trends, a quantum leap would be needed for quantum computers to mean- ingfully impact deep learning over the coming decade or two. We arrive at this conclusion based on a first-of-its-kind survey of quantum algorithms and how they match potential deep learning applications. This survey reveals three important areas where quantum computing could potentially accelerate deep learning, each of which faces a challenging roadblock to realizing its potential. First, quantum algorithms for matrix multiplication and other algorithms central to deep learning offer small theoretical improvements in the number of operations needed, but this advantage is overwhelmed on practical problem sizes by how slowly quantum computers do each operation. Second, some promising quantum algorithms depend on practical Quantum Random Access Memory (QRAM), which is underdeveloped. Finally, there are quantum algorithms that offer large theoretical advantages, but which are only applicable to special cases, limiting their practical benefits. In each of these areas, we support our arguments using quantitative forecasts of quantum advantage that build on the work by Choi et al. [2023] as well as new research on limitations and quantum hardware trends. Our analysis outlines the current scope of quantum deep learning and points to research directions that could lead to greater practical advances in the field. 

---
# Eyes on Target: Gaze-Aware Object Detection in Egocentric Video 

**Authors**: Vishakha Lall, Yisi Liu  

**Link**: [PDF](https://arxiv.org/pdf/2511.01237)  

**Abstract**: Human gaze offers rich supervisory signals for understanding visual attention in complex visual environments. In this paper, we propose Eyes on Target, a novel depth-aware and gaze-guided object detection framework designed for egocentric videos. Our approach injects gaze-derived features into the attention mechanism of a Vision Transformer (ViT), effectively biasing spatial feature selection toward human-attended regions. Unlike traditional object detectors that treat all regions equally, our method emphasises viewer-prioritised areas to enhance object detection. We validate our method on an egocentric simulator dataset where human visual attention is critical for task assessment, illustrating its potential in evaluating human performance in simulation scenarios. We evaluate the effectiveness of our gaze-integrated model through extensive experiments and ablation studies, demonstrating consistent gains in detection accuracy over gaze-agnostic baselines on both the custom simulator dataset and public benchmarks, including Ego4D Ego-Motion and Ego-CH-Gaze datasets. To interpret model behaviour, we also introduce a gaze-aware attention head importance metric, revealing how gaze cues modulate transformer attention dynamics. 

---
# Influence-aware Causal Autoencoder Network for Node Importance Ranking in Complex Networks 

**Authors**: Jiahui Gao, Kuang Zhou, Yuchen Zhu  

**Link**: [PDF](https://arxiv.org/pdf/2511.01228)  

**Abstract**: Node importance ranking is a fundamental problem in graph data analysis. Existing approaches typically rely on node features derived from either traditional centrality measures or advanced graph representation learning methods, which depend directly on the target network's topology. However, this reliance on structural information raises privacy concerns and often leads to poor generalization across different networks. In this work, we address a key question: Can we design a node importance ranking model trained exclusively on synthetic networks that is effectively appliable to real-world networks, eliminating the need to rely on the topology of target networks and improving both practicality and generalizability? We answer this question affirmatively by proposing the Influence-aware Causal Autoencoder Network (ICAN), a novel framework that leverages causal representation learning to get robust, invariant node embeddings for cross-network ranking tasks. Firstly, ICAN introduces an influence-aware causal representation learning module within an autoencoder architecture to extract node embeddings that are causally related to node importance. Moreover, we introduce a causal ranking loss and design a unified optimization framework that jointly optimizes the reconstruction and ranking objectives, enabling mutual reinforcement between node representation learning and ranking optimization. This design allows ICAN, trained on synthetic networks, to generalize effectively across diverse real-world graphs. Extensive experiments on multiple benchmark datasets demonstrate that ICAN consistently outperforms state-of-the-art baselines in terms of both ranking accuracy and generalization capability. 

---
# Thought-For-Food: Reasoning Chain Induced Food Visual Question Answering 

**Authors**: Riddhi Jain, Manasi Patwardhan, Parijat Deshpande, Venkataramana Runkana  

**Link**: [PDF](https://arxiv.org/pdf/2511.01213)  

**Abstract**: The immense diversity in the culture and culinary of Indian cuisines calls attention to the major shortcoming of the existing Visual Question Answering(VQA) systems which are inclined towards the foods from Western region. Recent attempt towards building a VQA dataset for Indian food is a step towards addressing this challenge. However, their approach towards VQA follows a two-step process in which the answer is generated first, followed by the explanation of the expected answer. In this work, we claim that food VQA requires to follow a multi-step reasoning process to arrive at an accurate answer, especially in the context of India food, which involves understanding complex culinary context and identifying relationships between various food items. With this hypothesis we create reasoning chains upon the QA with minimal human intervention. We fine-tune smaller LLMs and VLMs with auto-validated reasoning chains and further train them using reinforcement learning with larger data. With augmentation of reasoning chains, we observed accuracy improvement of an average 10 percentage points on the baseline. We provide detailed analysis in terms the effect of addition of reasoning chains for the Indian Food VQA task.
Index Terms - FoodVQA, Reasoning Chains, Reinforcement Learning, Knowledge Graph. 

---
# Forget BIT, It is All about TOKEN: Towards Semantic Information Theory for LLMs 

**Authors**: Bo Bai  

**Link**: [PDF](https://arxiv.org/pdf/2511.01202)  

**Abstract**: Large language models (LLMs) have demonstrated remarkable capabilities in numerous real- world applications. While the vast majority of research conducted from an experimental perspective is progressing rapidly, it demands substantial computational power, data, and other resources. Therefore, how to open the black-box of LLMs from a theoretical standpoint has become a critical challenge. This paper takes the theory of rate-distortion function, directed information, and Granger causality as its starting point to investigate the information-theoretic principles behind LLMs, leading to the development of semantic information theory for LLMs, where the fundamental unit is token, rather than bits that lacks any semantic meaning. By defining the probabilistic model of LLMs, we discuss structure-agnostic information-theoretic measures, such as the directed rate- distortion function in pre-training, the directed rate-reward function in post-training, and the semantic information flow in inference phase. This paper also delves deeply into the theory of token-level semantic embedding and the information-theoretically optimal vectorization method. Thereafter, we propose a general definition of autoregression LLM, where the Transformer architecture and its performance such as ELBO, generalization error bound, memory capacity, and semantic information measures can be derived theoretically. Other architectures, such as Mamba/Mamba2 and LLaDA, are also discussed in our framework. Consequently, this paper provides a theoretical framework for understanding LLMs from the perspective of semantic information theory, which also offers the necessary theoretical tools for further in-depth research. 

---
# An Interdisciplinary and Cross-Task Review on Missing Data Imputation 

**Authors**: Jicong Fan  

**Link**: [PDF](https://arxiv.org/pdf/2511.01196)  

**Abstract**: Missing data is a fundamental challenge in data science, significantly hindering analysis and decision-making across a wide range of disciplines, including healthcare, bioinformatics, social science, e-commerce, and industrial monitoring. Despite decades of research and numerous imputation methods, the literature remains fragmented across fields, creating a critical need for a comprehensive synthesis that connects statistical foundations with modern machine learning advances. This work systematically reviews core concepts-including missingness mechanisms, single versus multiple imputation, and different imputation goals-and examines problem characteristics across various domains. It provides a thorough categorization of imputation methods, spanning classical techniques (e.g., regression, the EM algorithm) to modern approaches like low-rank and high-rank matrix completion, deep learning models (autoencoders, GANs, diffusion models, graph neural networks), and large language models. Special attention is given to methods for complex data types, such as tensors, time series, streaming data, graph-structured data, categorical data, and multimodal data. Beyond methodology, we investigate the crucial integration of imputation with downstream tasks like classification, clustering, and anomaly detection, examining both sequential pipelines and joint optimization frameworks. The review also assesses theoretical guarantees, benchmarking resources, and evaluation metrics. Finally, we identify critical challenges and future directions, emphasizing model selection and hyperparameter optimization, the growing importance of privacy-preserving imputation via federated learning, and the pursuit of generalizable models that can adapt across domains and data types, thereby outlining a roadmap for future research. 

---
# A Topology-Aware Graph Convolutional Network for Human Pose Similarity and Action Quality Assessment 

**Authors**: Minmin Zeng  

**Link**: [PDF](https://arxiv.org/pdf/2511.01194)  

**Abstract**: Action Quality Assessment (AQA) requires fine-grained understanding of human motion and precise evaluation of pose similarity. This paper proposes a topology-aware Graph Convolutional Network (GCN) framework, termed GCN-PSN, which models the human skeleton as a graph to learn discriminative, topology-sensitive pose embeddings. Using a Siamese architecture trained with a contrastive regression objective, our method outperforms coordinate-based baselines and achieves competitive performance on AQA-7 and FineDiving benchmarks. Experimental results and ablation studies validate the effectiveness of leveraging skeletal topology for pose similarity and action quality assessment. 

---
# Self-Harmony: Learning to Harmonize Self-Supervision and Self-Play in Test-Time Reinforcement Learning 

**Authors**: Ru Wang, Wei Huang, Qi Cao, Yusuke Iwasawa, Yutaka Matsuo, Jiaxian Guo  

**Link**: [PDF](https://arxiv.org/pdf/2511.01191)  

**Abstract**: Test-time reinforcement learning (TTRL) offers a label-free paradigm for adapting models using only synthetic signals at inference, but its success hinges on constructing reliable learning signals. Standard approaches such as majority voting often collapse to spurious yet popular answers. We introduce Self-Harmony, a framework built on a simple intuition: the correct answer should remain stable across both an original question and its paraphrase. Self-Harmony operationalizes this by employing a single model in two complementary roles: a Solver to produce answers and a Reframer to rephrase the input. Based on this, we further propose a pseudo-label method: instead of majority voting, it aggregates answer frequencies across these original and reframed views using the harmonic mean. This is a process that naturally selects for solutions stable under reframing, thereby avoiding the common trap of favoring view-dependent, spurious answers. Crucially, this requires no human supervision or auxiliary models. Across diverse reasoning benchmarks, Self-Harmony achieves state-of-the-art results at the label-free test-time setting, ranking first in 28 of 30 settings across multiple methods. Beyond accuracy, it demonstrates unprecedented robustness, with zero training failures in all experiments, underscoring its stability and reliability. 

---
# ZoFia: Zero-Shot Fake News Detection with Entity-Guided Retrieval and Multi-LLM Interaction 

**Authors**: Lvhua Wu, Xuefeng Jiang, Sheng Sun, Tian Wen, Yuwei Wang, Min Liu  

**Link**: [PDF](https://arxiv.org/pdf/2511.01188)  

**Abstract**: The rapid spread of fake news threatens social stability and public trust, rendering its detection an imperative research priority. Although large language models (LLMs) excel at numerous natural language processing tasks with their remarkable contextual understanding and extensive prior knowledge, the time-bounded knowledge coverage and tendency for generating hallucination content reduce their reliability when handling fast-evolving news streams. Furthermore, models trained on existing static datasets also often lack the generalization needed for emerging news topics. To address these challenges, we propose ZoFia, a novel two-stage zero-shot fake news detection framework. First, we introduce Hierarchical Salience to quantify the importance of entities in the news content, and propose the SC-MMR algorithm to effectively select an informative and diverse set of keywords that serve as queries for retrieving up-to-date external evidence. Subsequently, a multi LLM interactive system, in which each agent assumes a distinct role, performs multi-view collaborative analysis and adversarial debate over the news text and its related information, and finally produces an interpretable and robust judgment. Comprehensive experiments on two public datasets demonstrate that ZoFia obviously outperforms existing zero-shot baselines and most of few-shot methods. Our codes will be open-sourced to facilitate related communities. 

---
# Adapt under Attack and Domain Shift: Unified Adversarial Meta-Learning and Domain Adaptation for Robust Automatic Modulation Classification 

**Authors**: Ali Owfi, Amirmohammad Bamdad, Tolunay Seyfi, Fatemeh Afghah  

**Link**: [PDF](https://arxiv.org/pdf/2511.01172)  

**Abstract**: Deep learning has emerged as a leading approach for Automatic Modulation Classification (AMC), demonstrating superior performance over traditional methods. However, vulnerability to adversarial attacks and susceptibility to data distribution shifts hinder their practical deployment in real-world, dynamic environments. To address these threats, we propose a novel, unified framework that integrates meta-learning with domain adaptation, making AMC systems resistant to both adversarial attacks and environmental changes. Our framework utilizes a two-phase strategy. First, in an offline phase, we employ a meta-learning approach to train the model on clean and adversarially perturbed samples from a single source domain. This method enables the model to generalize its defense, making it resistant to a combination of previously unseen attacks. Subsequently, in the online phase, we apply domain adaptation to align the model's features with a new target domain, allowing it to adapt without requiring substantial labeled data. As a result, our framework achieves a significant improvement in modulation classification accuracy against these combined threats, offering a critical solution to the deployment and operational challenges of modern AMC systems. 

---
# A High-Throughput Spiking Neural Network Processor Enabling Synaptic Delay Emulation 

**Authors**: Faquan Chen, Qingyang Tian, Ziren Wu, Rendong Ying, Fei Wen, Peilin Liu  

**Link**: [PDF](https://arxiv.org/pdf/2511.01158)  

**Abstract**: Synaptic delay has attracted significant attention in neural network dynamics for integrating and processing complex spatiotemporal information. This paper introduces a high-throughput Spiking Neural Network (SNN) processor that supports synaptic delay-based emulation for edge applications. The processor leverages a multicore pipelined architecture with parallel compute engines, capable of real-time processing of the computational load associated with synaptic delays. We develop a SoC prototype of the proposed processor on PYNQ Z2 FPGA platform and evaluate its performance using the Spiking Heidelberg Digits (SHD) benchmark for low-power keyword spotting tasks. The processor achieves 93.4% accuracy in deployment and an average throughput of 104 samples/sec at a typical operating frequency of 125 MHz and 282 mW power consumption. 

---
# AthenaBench: A Dynamic Benchmark for Evaluating LLMs in Cyber Threat Intelligence 

**Authors**: Md Tanvirul Alam, Dipkamal Bhusal, Salman Ahmad, Nidhi Rastogi, Peter Worth  

**Link**: [PDF](https://arxiv.org/pdf/2511.01144)  

**Abstract**: Large Language Models (LLMs) have demonstrated strong capabilities in natural language reasoning, yet their application to Cyber Threat Intelligence (CTI) remains limited. CTI analysis involves distilling large volumes of unstructured reports into actionable knowledge, a process where LLMs could substantially reduce analyst workload. CTIBench introduced a comprehensive benchmark for evaluating LLMs across multiple CTI tasks. In this work, we extend CTIBench by developing AthenaBench, an enhanced benchmark that includes an improved dataset creation pipeline, duplicate removal, refined evaluation metrics, and a new task focused on risk mitigation strategies. We evaluate twelve LLMs, including state-of-the-art proprietary models such as GPT-5 and Gemini-2.5 Pro, alongside seven open-source models from the LLaMA and Qwen families. While proprietary LLMs achieve stronger results overall, their performance remains subpar on reasoning-intensive tasks, such as threat actor attribution and risk mitigation, with open-source models trailing even further behind. These findings highlight fundamental limitations in the reasoning capabilities of current LLMs and underscore the need for models explicitly tailored to CTI workflows and automation. 

---
# MicroAUNet: Boundary-Enhanced Multi-scale Fusion with Knowledge Distillation for Colonoscopy Polyp Image Segmentation 

**Authors**: Ziyi Wang, Yuanmei Zhang, Dorna Esrafilzadeh, Ali R. Jalili, Suncheng Xiang  

**Link**: [PDF](https://arxiv.org/pdf/2511.01143)  

**Abstract**: Early and accurate segmentation of colorectal polyps is critical for reducing colorectal cancer mortality, which has been extensively explored by academia and industry. However, current deep learning-based polyp segmentation models either compromise clinical decision-making by providing ambiguous polyp margins in segmentation outputs or rely on heavy architectures with high computational complexity, resulting in insufficient inference speeds for real-time colorectal endoscopic applications. To address this problem, we propose MicroAUNet, a light-weighted attention-based segmentation network that combines depthwise-separable dilated convolutions with a single-path, parameter-shared channel-spatial attention block to strengthen multi-scale boundary features. On the basis of it, a progressive two-stage knowledge-distillation scheme is introduced to transfer semantic and boundary cues from a high-capacity teacher. Extensive experiments on benchmarks also demonstrate the state-of-the-art accuracy under extremely low model complexity, indicating that MicroAUNet is suitable for real-time clinical polyp segmentation. The code is publicly available at this https URL. 

---
# Few-Shot Multimodal Medical Imaging: A Theoretical Framework 

**Authors**: Md Talha Mohsin, Ismail Abdulrashid  

**Link**: [PDF](https://arxiv.org/pdf/2511.01140)  

**Abstract**: Medical imaging relies heavily on large, labeled datasets. But, unfortunately, they are not always easily accessible in clinical settings. Additionally, many practitioners often face various structural obstacles like limited data availability, fragmented data systems, and unbalanced datasets. These barriers often lead to the increased diagnostic uncertainty, underrepresentation of certain conditions, reduced model robustness, and biased diagnostic decisions. In response to these challenges, approaches such as transfer learning, meta-learning, and multimodal fusion have made great strides. However, they still need a solid theoretical justification for why they succeed or fail in situations where data is scarce. To address this gap, we propose a unified theoretical framework that characterizes learning and inference under low-resource medical imaging conditions. We first formalize the learning objective under few-shot conditions and compute sample complexity constraints to estimate the smallest quantity of data needed to achieve clinically reliable accuracy. Then based on ideas from PAC-learning and PAC-Bayesian theory, we explain how multimodal integration encourages generalization and quantifies uncertainty under sparse supervision. We further propose a formal metric for explanation stability, offering interpretability guarantees under low-data conditions. Taken together, the proposed framework establishes a principled foundation for constructing dependable, data-efficient diagnostic systems by jointly characterizing sample efficiency, uncertainty quantification, and interpretability in a unified theoretical setting. 

---
# Learning with Category-Equivariant Architectures for Human Activity Recognition 

**Authors**: Yoshihiro Maruyama  

**Link**: [PDF](https://arxiv.org/pdf/2511.01139)  

**Abstract**: We propose CatEquiv, a category-equivariant neural network for Human Activity Recognition (HAR) from inertial sensors that systematically encodes temporal, amplitude, and structural symmetries. In particular, we introduce the categorical symmetry product where cyclic time shifts, positive gains and the sensor-hierarchy poset together capture the categorical symmetry structure of the data. CatEquiv achieves equivariance with respect to the categorical symmetry product. On UCI-HAR under out-of-distribution perturbations, CatEquiv attains markedly higher robustness compared with circularly padded CNNs and plain CNNs. These results demonstrate that enforcing categorical symmetries yields strong invariance and generalization without additional model capacity. 

---
# Continual Learning, Not Training: Online Adaptation For Agents 

**Authors**: Aman Jaglan, Jarrod Barnes  

**Link**: [PDF](https://arxiv.org/pdf/2511.01093)  

**Abstract**: Continual Learning (CL) methods have traditionally focused on mitigating catastrophic forgetting through gradient-based retraining, an approach ill-suited for deployed agents that must adapt in real time. We introduce our Adaptive Teaching and Learning System (ATLAS), a dual-agent architecture that decouples reasoning (Teacher) from execution (Student) and incorporates a persistent learning memory that stores distilled guidance from experience. This informs the orchestration layer, enabling the system to dynamically adjust its operational strategies, such as supervision level or initial plan selection, at inference time. In doing so, ATLAS achieves gradient-free continual learning, shifting the locus of adaptation from model parameters to system-level orchestration. We formulate this as a system-centric paradigm for continual learning, where the objective is adaptive efficiency: maximizing task success while minimizing computational cost through inference-time orchestration rather than parameter updates. Evaluated on Microsoft's ExCyTIn-Bench, an open-source benchmark simulating complex cyberthreat investigation, ATLAS achieves 54.1% success with GPT-5-mini as its Student, outperforming the larger GPT-5 (High) by 13% while reducing cost by 86%. Cross-incident validation demonstrates generalization: frozen pamphlets from Incident #5 improve accuracy from 28% to 41% with zero retraining, while shifting output composition from verbose exploration to structured reasoning. Together, these findings establish gradient-free continual learning as a viable path toward adaptive, deployable AI systems and provide causally annotated traces valuable for training explicit world models. 

---
# SliceVision-F2I: A Synthetic Feature-to-Image Dataset for Visual Pattern Representation on Network Slices 

**Authors**: Md. Abid Hasan Rafi, Mst. Fatematuj Johora, Pankaj Bhowmik  

**Link**: [PDF](https://arxiv.org/pdf/2511.01087)  

**Abstract**: The emergence of 5G and 6G networks has established network slicing as a significant part of future service-oriented architectures, demanding refined identification methods supported by robust datasets. The article presents SliceVision-F2I, a dataset of synthetic samples for studying feature visualization in network slicing for next-generation networking systems. The dataset transforms multivariate Key Performance Indicator (KPI) vectors into visual representations through four distinct encoding methods: physically inspired mappings, Perlin noise, neural wallpapering, and fractal branching. For each encoding method, 30,000 samples are generated, each comprising a raw KPI vector and a corresponding RGB image at low-resolution pixels. The dataset simulates realistic and noisy network conditions to reflect operational uncertainties and measurement imperfections. SliceVision-F2I is suitable for tasks involving visual learning, network state classification, anomaly detection, and benchmarking of image-based machine learning techniques applied to network data. The dataset is publicly available and can be reused in various research contexts, including multivariate time series analysis, synthetic data generation, and feature-to-image transformations. 

---
# GeoToken: Hierarchical Geolocalization of Images via Next Token Prediction 

**Authors**: Narges Ghasemi, Amir Ziashahabi, Salman Avestimehr, Cyrus Shahabi  

**Link**: [PDF](https://arxiv.org/pdf/2511.01082)  

**Abstract**: Image geolocalization, the task of determining an image's geographic origin, poses significant challenges, largely due to visual similarities across disparate locations and the large search space. To address these issues, we propose a hierarchical sequence prediction approach inspired by how humans narrow down locations from broad regions to specific addresses. Analogously, our model predicts geographic tokens hierarchically, first identifying a general region and then sequentially refining predictions to increasingly precise locations. Rather than relying on explicit semantic partitions, our method uses S2 cells, a nested, multiresolution global grid, and sequentially predicts finer-level cells conditioned on visual inputs and previous predictions. This procedure mirrors autoregressive text generation in large language models. Much like in language modeling, final performance depends not only on training but also on inference-time strategy. We investigate multiple top-down traversal methods for autoregressive sampling, incorporating techniques from test-time compute scaling used in language models. Specifically, we integrate beam search and multi-sample inference while exploring various selection strategies to determine the final output. This enables the model to manage uncertainty by exploring multiple plausible paths through the hierarchy. We evaluate our method on the Im2GPS3k and YFCC4k datasets against two distinct sets of baselines: those that operate without a Multimodal Large Language Model (MLLM) and those that leverage one. In the MLLM-free setting, our model surpasses other comparable baselines on nearly all metrics, achieving state-of-the-art performance with accuracy gains of up to 13.9%. When augmented with an MLLM, our model outperforms all baselines, setting a new state-of-the-art across all metrics. The source code is available at this https URL. 

---
# Energy-Efficient Deep Learning Without Backpropagation: A Rigorous Evaluation of Forward-Only Algorithms 

**Authors**: Przemysław Spyra, Witold Dzwinel  

**Link**: [PDF](https://arxiv.org/pdf/2511.01061)  

**Abstract**: The long-held assumption that backpropagation (BP) is essential for state-of-the-art performance is challenged by this work. We present rigorous, hardware-validated evidence that the Mono-Forward (MF) algorithm, a backpropagation-free method, consistently surpasses an optimally tuned BP baseline in classification accuracy on its native Multi-Layer Perceptron (MLP) architectures. This superior generalization is achieved with profound efficiency gains, including up to 41% less energy consumption and up to 34% faster training. Our analysis, which charts an evolutionary path from Geoffrey Hinton's Forward-Forward (FF) to the Cascaded Forward (CaFo) and finally to MF, is grounded in a fair comparative framework using identical architectures and universal hyperparameter optimization. We further provide a critical re-evaluation of memory efficiency in BP-free methods, empirically demonstrating that practical overhead can offset theoretical gains. Ultimately, this work establishes MF as a practical, high-performance, and sustainable alternative to BP for MLPs. 

---
# HAFixAgent: History-Aware Automated Program Repair Agent 

**Authors**: Yu Shi, Hao Li, Bram Adams, Ahmed E. Hassan  

**Link**: [PDF](https://arxiv.org/pdf/2511.01047)  

**Abstract**: Automated program repair (APR) has recently shifted toward large language models and agent-based systems, yet most systems rely on local snapshot context, overlooking repository history. Prior work shows that repository history helps repair single-line bugs, since the last commit touching the buggy line is often the bug-introducing one. In this paper, we investigate whether repository history can also improve agentic APR systems at scale, especially for complex multi-hunk bugs. We present HAFixAgent, a History-Aware Bug-Fixing Agent that injects blame-derived repository heuristics into its repair loop. A preliminary study of all 854 real-world bugs from Defects4J motivates our design, showing that bug-relevant history is both widely available and highly concentrated. Empirical comparison of HAFixAgent with two state-of-the-art baselines shows: (1) Effectiveness: HAFixAgent significantly improves over the agent-based baseline (by 212.3%) and the multi-hunk baseline (by 29.9%). (2) Efficiency: history does not significantly increase agent steps and keeps token costs comparable, with notably lower median costs for complex multi-file-multi-hunk bugs. (3) Practicality: combining different historical heuristics repairs more bugs, offering a clear cost-benefit trade-off. HAFixAgent offers a practical recipe for history-aware agentic APR: ground the agent in version control history, prioritize diff-based historical context, and integrate complementary heuristics when needed. 

---
# Seed-Induced Uniqueness in Transformer Models: Subspace Alignment Governs Subliminal Transfer 

**Authors**: Ayşe Selin Okatan, Mustafa İlhan Akbaş, Laxima Niure Kandel, Berker Peköz  

**Link**: [PDF](https://arxiv.org/pdf/2511.01023)  

**Abstract**: We analyze subliminal transfer in Transformer models, where a teacher embeds hidden traits that can be linearly decoded by a student without degrading main-task performance. Prior work often attributes transferability to global representational similarity, typically quantified with Centered Kernel Alignment (CKA). Using synthetic corpora with disentangled public and private labels, we distill students under matched and independent random initializations. We find that transfer strength hinges on alignment within a trait-discriminative subspace: same-seed students inherit this alignment and show higher leakage {\tau \approx} 0.24, whereas different-seed students--despite global CKA > 0.9--exhibit substantially reduced excess accuracy {\tau \approx} 0.12 - 0.13. We formalize this with subspace-level CKA diagnostic and residualized probes, showing that leakage tracks alignment within the trait-discriminative subspace rather than global representational similarity. Security controls (projection penalty, adversarial reversal, right-for-the-wrong-reasons regularization) reduce leakage in same-base models without impairing public-task fidelity. These results establish seed-induced uniqueness as a resilience property and argue for subspace-aware diagnostics for secure multi-model deployments. 

---
# OceanAI: A Conversational Platform for Accurate, Transparent, Near-Real-Time Oceanographic Insights 

**Authors**: Bowen Chen, Jayesh Gajbhar, Gregory Dusek, Rob Redmon, Patrick Hogan, Paul Liu, DelWayne Bohnenstiehl, Dongkuan, Ruoying He  

**Link**: [PDF](https://arxiv.org/pdf/2511.01019)  

**Abstract**: Artificial intelligence is transforming the sciences, yet general conversational AI systems often generate unverified "hallucinations" undermining scientific rigor. We present OceanAI, a conversational platform that integrates the natural-language fluency of open-source large language models (LLMs) with real-time, parameterized access to authoritative oceanographic data streams hosted by the National Oceanic and Atmospheric Administration (NOAA). Each query such as "What was Boston Harbor's highest water level in 2024?" triggers real-time API calls that identify, parse, and synthesize relevant datasets into reproducible natural-language responses and data visualizations. In a blind comparison with three widely used AI chat-interface products, only OceanAI produced NOAA-sourced values with original data references; others either declined to answer or provided unsupported results. Designed for extensibility, OceanAI connects to multiple NOAA data products and variables, supporting applications in marine hazard forecasting, ecosystem assessment, and water-quality monitoring. By grounding outputs and verifiable observations, OceanAI advances transparency, reproducibility, and trust, offering a scalable framework for AI-enabled decision support within the oceans. A public demonstration is available at this https URL. 

---
# ORANGE: An Online Reflection ANd GEneration framework with Domain Knowledge for Text-to-SQL 

**Authors**: Yiwen Jiao, Tonghui Ren, Yuche Gao, Zhenying He, Yinan Jing, Kai Zhang, X. Sean Wang  

**Link**: [PDF](https://arxiv.org/pdf/2511.00985)  

**Abstract**: Large Language Models (LLMs) have demonstrated remarkable progress in translating natural language to SQL, but a significant semantic gap persists between their general knowledge and domain-specific semantics of databases. Historical translation logs constitute a rich source of this missing in-domain knowledge, where SQL queries inherently encapsulate real-world usage patterns of database schema. Existing methods primarily enhance the reasoning process for individual translations but fail to accumulate in-domain knowledge from past translations. We introduce ORANGE, an online self-evolutionary framework that constructs database-specific knowledge bases by parsing SQL queries from translation logs. By accumulating in-domain knowledge that contains schema and data semantics, ORANGE progressively reduces the semantic gap and enhances the accuracy of subsequent SQL translations. To ensure reliability, we propose a novel nested Chain-of-Thought SQL-to-Text strategy with tuple-semantic tracking, which reduces semantic errors during knowledge generation. Experiments on multiple benchmarks confirm the practicality of ORANGE, demonstrating its effectiveness for real-world Text-to-SQL deployment, particularly in handling complex and domain-specific queries. 

---
# Keys in the Weights: Transformer Authentication Using Model-Bound Latent Representations 

**Authors**: Ayşe S. Okatan, Mustafa İlhan Akbaş, Laxima Niure Kandel, Berker Peköz  

**Link**: [PDF](https://arxiv.org/pdf/2511.00973)  

**Abstract**: We introduce Model-Bound Latent Exchange (MoBLE), a decoder-binding property in Transformer autoencoders formalized as Zero-Shot Decoder Non-Transferability (ZSDN). In identity tasks using iso-architectural models trained on identical data but differing in seeds, self-decoding achieves more than 0.91 exact match and 0.98 token accuracy, while zero-shot cross-decoding collapses to chance without exact matches. This separation arises without injected secrets or adversarial training, and is corroborated by weight-space distances and attention-divergence diagnostics. We interpret ZSDN as model binding, a latent-based authentication and access-control mechanism, even when the architecture and training recipe are public: encoder's hidden state representation deterministically reveals the plaintext, yet only the correctly keyed decoder reproduces it in zero-shot. We formally define ZSDN, a decoder-binding advantage metric, and outline deployment considerations for secure artificial intelligence (AI) pipelines. Finally, we discuss learnability risks (e.g., adapter alignment) and outline mitigations. MoBLE offers a lightweight, accelerator-friendly approach to secure AI deployment in safety-critical domains, including aviation and cyber-physical systems. 

---
# Using Synthetic Data to estimate the True Error is theoretically and practically doable 

**Authors**: Hai Hoang Thanh, Duy-Tung Nguyen, Hung The Tran, Khoat Than  

**Link**: [PDF](https://arxiv.org/pdf/2511.00964)  

**Abstract**: Accurately evaluating model performance is crucial for deploying machine learning systems in real-world applications. Traditional methods often require a sufficiently large labeled test set to ensure a reliable evaluation. However, in many contexts, a large labeled dataset is costly and labor-intensive. Therefore, we sometimes have to do evaluation by a few labeled samples, which is theoretically challenging. Recent advances in generative models offer a promising alternative by enabling the synthesis of high-quality data. In this work, we make a systematic investigation about the use of synthetic data to estimate the test error of a trained model under limited labeled data conditions. To this end, we develop novel generalization bounds that take synthetic data into account. Those bounds suggest novel ways to optimize synthetic samples for evaluation and theoretically reveal the significant role of the generator's quality. Inspired by those bounds, we propose a theoretically grounded method to generate optimized synthetic data for model evaluation. Experimental results on simulation and tabular datasets demonstrate that, compared to existing baselines, our method achieves accurate and more reliable estimates of the test error. 

---
# The Riddle of Reflection: Evaluating Reasoning and Self-Awareness in Multilingual LLMs using Indian Riddles 

**Authors**: Abhinav P M, Ojasva Saxena, Oswald C, Parameswari Krishnamurthy  

**Link**: [PDF](https://arxiv.org/pdf/2511.00960)  

**Abstract**: The extent to which large language models (LLMs) can perform culturally grounded reasoning across non-English languages remains underexplored. This paper examines the reasoning and self-assessment abilities of LLMs across seven major Indian languages-Bengali, Gujarati, Hindi, Kannada, Malayalam, Tamil, and Telugu. We introduce a multilingual riddle dataset combining traditional riddles with context-reconstructed variants and evaluate five LLMs-Gemini 2.5 Pro, Gemini 2.5 Flash, Mistral-Saba, LLaMA 4 Scout, and LLaMA 4 Maverick-under seven prompting strategies. In the first stage, we assess riddle-solving performance and find that while Gemini 2.5 Pro performs best overall, few-shot methods yield only marginal gains, and accuracy varies notably across languages. In the second stage, we conduct a self-evaluation experiment to measure reasoning consistency. The results reveal a key finding: a model's initial accuracy is inversely correlated with its ability to identify its own mistakes. Top-performing models such as Gemini 2.5 Pro are overconfident (4.34% True Negative Rate), whereas lower-performing models like LLaMA 4 Scout are substantially more self-aware (42.09% True Negative Rate). These results point to clear gaps in multilingual reasoning and highlight the need for models that not only reason effectively but also recognize their own limitations. 

---
# The Hidden Power of Normalization: Exponential Capacity Control in Deep Neural Networks 

**Authors**: Khoat Than  

**Link**: [PDF](https://arxiv.org/pdf/2511.00958)  

**Abstract**: Normalization methods are fundamental components of modern deep neural networks (DNNs). Empirically, they are known to stabilize optimization dynamics and improve generalization. However, the underlying theoretical mechanism by which normalization contributes to both optimization and generalization remains largely unexplained, especially when using many normalization layers in a DNN architecture.
In this work, we develop a theoretical framework that elucidates the role of normalization through the lens of capacity control. We prove that an unnormalized DNN can exhibit exponentially large Lipschitz constants with respect to either its parameters or inputs, implying excessive functional capacity and potential overfitting. Such bad DNNs are uncountably many. In contrast, the insertion of normalization layers provably can reduce the Lipschitz constant at an exponential rate in the number of normalization operations. This exponential reduction yields two fundamental consequences: (1) it smooths the loss landscape at an exponential rate, facilitating faster and more stable optimization; and (2) it constrains the effective capacity of the network, thereby enhancing generalization guarantees on unseen data. Our results thus offer a principled explanation for the empirical success of normalization methods in deep learning. 

---
# URDF-Anything: Constructing Articulated Objects with 3D Multimodal Language Model 

**Authors**: Zhe Li, Xiang Bai, Jieyu Zhang, Zhuangzhe Wu, Che Xu, Ying Li, Chengkai Hou, Shanghang Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2511.00940)  

**Abstract**: Constructing accurate digital twins of articulated objects is essential for robotic simulation training and embodied AI world model building, yet historically requires painstaking manual modeling or multi-stage pipelines. In this work, we propose \textbf{URDF-Anything}, an end-to-end automatic reconstruction framework based on a 3D multimodal large language model (MLLM). URDF-Anything utilizes an autoregressive prediction framework based on point-cloud and text multimodal input to jointly optimize geometric segmentation and kinematic parameter prediction. It implements a specialized $[SEG]$ token mechanism that interacts directly with point cloud features, enabling fine-grained part-level segmentation while maintaining consistency with the kinematic parameter predictions. Experiments on both simulated and real-world datasets demonstrate that our method significantly outperforms existing approaches regarding geometric segmentation (mIoU 17\% improvement), kinematic parameter prediction (average error reduction of 29\%), and physical executability (surpassing baselines by 50\%). Notably, our method exhibits excellent generalization ability, performing well even on objects outside the training set. This work provides an efficient solution for constructing digital twins for robotic simulation, significantly enhancing the sim-to-real transfer capability. 

---
# Maestro: Orchestrating Robotics Modules with Vision-Language Models for Zero-Shot Generalist Robots 

**Authors**: Junyao Shi, Rujia Yang, Kaitian Chao, Selina Bingqing Wan, Yifei Shao, Jiahui Lei, Jianing Qian, Long Le, Pratik Chaudhari, Kostas Daniilidis, Chuan Wen, Dinesh Jayaraman  

**Link**: [PDF](https://arxiv.org/pdf/2511.00917)  

**Abstract**: Today's best-explored routes towards generalist robots center on collecting ever larger "observations-in actions-out" robotics datasets to train large end-to-end models, copying a recipe that has worked for vision-language models (VLMs). We pursue a road less traveled: building generalist policies directly around VLMs by augmenting their general capabilities with specific robot capabilities encapsulated in a carefully curated set of perception, planning, and control modules. In Maestro, a VLM coding agent dynamically composes these modules into a programmatic policy for the current task and scenario. Maestro's architecture benefits from a streamlined closed-loop interface without many manually imposed structural constraints, and a comprehensive and diverse tool repertoire. As a result, it largely surpasses today's VLA models for zero-shot performance on challenging manipulation skills. Further, Maestro is easily extensible to incorporate new modules, easily editable to suit new embodiments such as a quadruped-mounted arm, and even easily adapts from minimal real-world experiences through local code edits. 

---
# Learning with Category-Equivariant Representations for Human Activity Recognition 

**Authors**: Yoshihiro Maruyama  

**Link**: [PDF](https://arxiv.org/pdf/2511.00900)  

**Abstract**: Human activity recognition is challenging because sensor signals shift with context, motion, and environment; effective models must therefore remain stable as the world around them changes. We introduce a categorical symmetry-aware learning framework that captures how signals vary over time, scale, and sensor hierarchy. We build these factors into the structure of feature representations, yielding models that automatically preserve the relationships between sensors and remain stable under realistic distortions such as time shifts, amplitude drift, and device orientation changes. On the UCI Human Activity Recognition benchmark, this categorical symmetry-driven design improves out-of-distribution accuracy by approx. 46 percentage points (approx. 3.6x over the baseline), demonstrating that abstract symmetry principles can translate into concrete performance gains in everyday sensing tasks via category-equivariant representation theory. 

---
# Dynamic Logic of Trust-Based Beliefs 

**Authors**: Junli Jiang, Pavel Naumov, Wenxuan Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2511.00899)  

**Abstract**: Traditionally, an agent's beliefs would come from what the agent can see, hear, or sense. In the modern world, beliefs are often based on the data available to the agents. In this work, we investigate a dynamic logic of such beliefs that incorporates public announcements of data. The main technical contribution is a sound and complete axiomatisation of the interplay between data-informed beliefs and data announcement modalities. We also describe a non-trivial polynomial model checking algorithm for this logical system. 

---
# Android Malware Detection: A Machine Leaning Approach 

**Authors**: Hasan Abdulla  

**Link**: [PDF](https://arxiv.org/pdf/2511.00894)  

**Abstract**: This study examines machine learning techniques like Decision Trees, Support Vector Machines, Logistic Regression, Neural Networks, and ensemble methods to detect Android malware. The study evaluates these models on a dataset of Android applications and analyzes their accuracy, efficiency, and real-world applicability. Key findings show that ensemble methods demonstrate superior performance, but there are trade-offs between model interpretability, efficiency, and accuracy. Given its increasing threat, the insights guide future research and practical use of ML to combat Android malware. 

---
# Deep Generative Models for Enhanced Vitreous OCT Imaging 

**Authors**: Simone Sarrocco, Philippe C. Cattin, Peter M. Maloca, Paul Friedrich, Philippe Valmaggia  

**Link**: [PDF](https://arxiv.org/pdf/2511.00881)  

**Abstract**: Purpose: To evaluate deep learning (DL) models for enhancing vitreous optical coherence tomography (OCT) image quality and reducing acquisition time. Methods: Conditional Denoising Diffusion Probabilistic Models (cDDPMs), Brownian Bridge Diffusion Models (BBDMs), U-Net, Pix2Pix, and Vector-Quantised Generative Adversarial Network (VQ-GAN) were used to generate high-quality spectral-domain (SD) vitreous OCT images. Inputs were SD ART10 images, and outputs were compared to pseudoART100 images obtained by averaging ten ART10 images per eye location. Model performance was assessed using image quality metrics and Visual Turing Tests, where ophthalmologists ranked generated images and evaluated anatomical fidelity. The best model's performance was further tested within the manually segmented vitreous on newly acquired data. Results: U-Net achieved the highest Peak Signal-to-Noise Ratio (PSNR: 30.230) and Structural Similarity Index Measure (SSIM: 0.820), followed by cDDPM. For Learned Perceptual Image Patch Similarity (LPIPS), Pix2Pix (0.697) and cDDPM (0.753) performed best. In the first Visual Turing Test, cDDPM ranked highest (3.07); in the second (best model only), cDDPM achieved a 32.9% fool rate and 85.7% anatomical preservation. On newly acquired data, cDDPM generated vitreous regions more similar in PSNR to the ART100 reference than true ART1 or ART10 B-scans and achieved higher PSNR on whole images when conditioned on ART1 than ART10. Conclusions: Results reveal discrepancies between quantitative metrics and clinical evaluation, highlighting the need for combined assessment. cDDPM showed strong potential for generating clinically meaningful vitreous OCT images while reducing acquisition time fourfold. Translational Relevance: cDDPMs show promise for clinical integration, supporting faster, higher-quality vitreous imaging. Dataset and code will be made publicly available. 

---
# KFCPO: Kronecker-Factored Approximated Constrained Policy Optimization 

**Authors**: Joonyoung Lim, Younghwan Yoo  

**Link**: [PDF](https://arxiv.org/pdf/2511.00880)  

**Abstract**: We propose KFCPO, a novel Safe Reinforcement Learning (Safe RL) algorithm that combines scalable Kronecker-Factored Approximate Curvature (K-FAC) based second-order policy optimization with safety-aware gradient manipulation. KFCPO leverages K-FAC to perform efficient and stable natural gradient updates by approximating the Fisher Information Matrix (FIM) in a layerwise, closed form manner, avoiding iterative approximation overheads. To address the tradeoff between reward maximization and constraint satisfaction, we introduce a margin aware gradient manipulation mechanism that adaptively adjusts the influence of reward and cost gradients based on the agent's proximity to safety boundaries. This method blends gradients using a direction sensitive projection, eliminating harmful interference and avoiding abrupt changes caused by fixed hard thresholds. Additionally, a minibatch level KL rollback strategy is adopted to ensure trust region compliance and to prevent destabilizing policy shifts. Experiments on Safety Gymnasium using OmniSafe show that KFCPO achieves 10.3% to 50.2% higher average return across environments compared to the best baseline that respected the safety constraint, demonstrating superior balance of safety and performance. 

---
# Assessing LLM Reasoning Steps via Principal Knowledge Grounding 

**Authors**: Hyeon Hwang, Yewon Cho, Chanwoong Yoon, Yein Park, Minju Song, Kyungjae Lee, Gangwoo Kim, Jaewoo Kang  

**Link**: [PDF](https://arxiv.org/pdf/2511.00879)  

**Abstract**: Step-by-step reasoning has become a standard approach for large language models (LLMs) to tackle complex tasks. While this paradigm has proven effective, it raises a fundamental question: How can we verify that an LLM's reasoning is accurately grounded in knowledge? To address this question, we introduce a novel evaluation suite that systematically assesses the knowledge grounding of intermediate reasoning. Our framework comprises three key components. (1) Principal Knowledge Collection, a large-scale repository of atomic knowledge essential for reasoning. Based on the collection, we propose (2) knowledge-grounded evaluation metrics designed to measure how well models recall and apply prerequisite knowledge in reasoning. These metrics are computed by our (3) evaluator LLM, a lightweight model optimized for cost-effective and reliable metric computation. Our evaluation suite demonstrates remarkable effectiveness in identifying missing or misapplied knowledge elements, providing crucial insights for uncovering fundamental reasoning deficiencies in LLMs. Beyond evaluation, we demonstrate how these metrics can be integrated into preference optimization, showcasing further applications of knowledge-grounded evaluation. 

---
# Fast Stochastic Greedy Algorithm for $k$-Submodular Cover Problem 

**Authors**: Hue T. Nguyen, Tan D. Tran, Nguyen Long Giang, Canh V. Pham  

**Link**: [PDF](https://arxiv.org/pdf/2511.00869)  

**Abstract**: We study the $k$-Submodular Cover ($kSC$) problem, a natural generalization of the classical Submodular Cover problem that arises in artificial intelligence and combinatorial optimization tasks such as influence maximization, resource allocation, and sensor placement. Existing algorithms for $\kSC$ often provide weak approximation guarantees or incur prohibitively high query complexity. To overcome these limitations, we propose a \textit{Fast Stochastic Greedy} algorithm that achieves strong bicriteria approximation while substantially lowering query complexity compared to state-of-the-art methods. Our approach dramatically reduces the number of function evaluations, making it highly scalable and practical for large-scale real-world AI applications where efficiency is essential. 

---
# Occlusion-Aware Diffusion Model for Pedestrian Intention Prediction 

**Authors**: Yu Liu, Zhijie Liu, Zedong Yang, You-Fu Li, He Kong  

**Link**: [PDF](https://arxiv.org/pdf/2511.00858)  

**Abstract**: Predicting pedestrian crossing intentions is crucial for the navigation of mobile robots and intelligent vehicles. Although recent deep learning-based models have shown significant success in forecasting intentions, few consider incomplete observation under occlusion scenarios. To tackle this challenge, we propose an Occlusion-Aware Diffusion Model (ODM) that reconstructs occluded motion patterns and leverages them to guide future intention prediction. During the denoising stage, we introduce an occlusion-aware diffusion transformer architecture to estimate noise features associated with occluded patterns, thereby enhancing the model's ability to capture contextual relationships in occluded semantic scenarios. Furthermore, an occlusion mask-guided reverse process is introduced to effectively utilize observation information, reducing the accumulation of prediction errors and enhancing the accuracy of reconstructed motion features. The performance of the proposed method under various occlusion scenarios is comprehensively evaluated and compared with existing methods on popular benchmarks, namely PIE and JAAD. Extensive experimental results demonstrate that the proposed method achieves more robust performance than existing methods in the literature. 

---
# MULTI-Bench: A Multi-Turn Interactive Benchmark for Assessing Emotional Intelligence ability of Spoken Dialogue Models 

**Authors**: Yayue Deng, Guoqiang Hu, Haiyang Sun, Xiangyu Zhang, Haoyang Zhang, Fei Tian, Xuerui Yang, Gang Yu, Eng Siong Chng  

**Link**: [PDF](https://arxiv.org/pdf/2511.00850)  

**Abstract**: Spoken Dialogue Models (SDMs) have advanced rapidly, yet their ability to sustain genuinely interactive multi-turn conversations remains underexplored, as most benchmarks focus on single-turn exchanges. We introduce Multi-Bench, the first benchmark explicitly designed to evaluate SDMs in multi-turn interactive dialogue with an emphasis on emotional intelligence. Multi-Bench employs a hierarchical structure with a basic track for emotion understanding and reasoning and an advanced track for emotion support and application. It comprises five carefully designed tasks and about 3.2K samples, ranging from emotion recognition to complex reasoning and interactive dialogue, supported by a reproducible evaluation framework. We evaluate six representative SDMs on eight subsets of Multi-Bench. Results show that while current SDMs achieve good performance on basic understanding tasks, they still have room for improvement in advanced multi-turn interactive dialogue and reasoning-related tasks, particularly in emotion awareness and application. 

---
# Pay for The Second-Best Service: A Game-Theoretic Approach Against Dishonest LLM Providers 

**Authors**: Yuhan Cao, Yu Wang, Sitong Liu, Miao Li, Yixin Tao, Tianxing He  

**Link**: [PDF](https://arxiv.org/pdf/2511.00847)  

**Abstract**: The widespread adoption of Large Language Models (LLMs) through Application Programming Interfaces (APIs) induces a critical vulnerability: the potential for dishonest manipulation by service providers. This manipulation can manifest in various forms, such as secretly substituting a proclaimed high-performance model with a low-cost alternative, or inflating responses with meaningless tokens to increase billing. This work tackles the issue through the lens of algorithmic game theory and mechanism design. We are the first to propose a formal economic model for a realistic user-provider ecosystem, where a user can iteratively delegate $T$ queries to multiple model providers, and providers can engage in a range of strategic behaviors. As our central contribution, we prove that for a continuous strategy space and any $\epsilon\in(0,\frac12)$, there exists an approximate incentive-compatible mechanism with an additive approximation ratio of $O(T^{1-\epsilon}\log T)$, and a guaranteed quasi-linear second-best user utility. We also prove an impossibility result, stating that no mechanism can guarantee an expected user utility that is asymptotically better than our mechanism. Furthermore, we demonstrate the effectiveness of our mechanism in simulation experiments with real-world API settings. 

---
# OmniBrainBench: A Comprehensive Multimodal Benchmark for Brain Imaging Analysis Across Multi-stage Clinical Tasks 

**Authors**: Zhihao Peng, Cheng Wang, Shengyuan Liu, Zhiying Liang, Yixuan Yuan  

**Link**: [PDF](https://arxiv.org/pdf/2511.00846)  

**Abstract**: Brain imaging analysis is vital for diagnosing and treating brain disorders, and multimodal large language models (MLLMs) are increasingly assisting in that analysis. However, current brain-oriented visual question-answering (VQA) benchmarks either cover a few imaging modalities or are limited to coarse-grained pathological descriptions, hindering a comprehensive assessment of MLLMs throughout the full clinical continuum. To address these, we introduce OmniBrainBench, the first comprehensive multimodal VQA benchmark specifically designed to assess the multimodal comprehension capabilities of MLLMs in brain imaging this http URL consists of 15 distinct brain imaging modalities collected from 30 verified medical sources, yielding 9,527 validated VQA pairs and 31,706 images. It simulates clinical workflows and encompasses 15 multi-stage clinical tasks rigorously validated by a professional radiologist. Evaluation of 24 state-of-the-art models, including open-source, medical, and proprietary MLLMs, highlights the substantial challenges posed by OmniBrainBench. Our experiments reveal: (1) proprietary MLLMs (e.g., GPT-5) beat open-source and medical models but lag physicians; (2) medical MLLMs vary widely in performance; (3) open-source MLLMs trail overall but excel in specific tasks; (4) MLLMs underperform sharply in complex preoperative tasks, revealing a visual-to-clinical reasoning gap. OmniBrainBench sets a new standard for evaluating and advancing MLLMs in brain imaging analysis, highlighting gaps compared to expert clinical reasoning. We release it at benchmark \& code. 

---
# CodeClash: Benchmarking Goal-Oriented Software Engineering 

**Authors**: John Yang, Kilian Lieret, Joyce Yang, Carlos E. Jimenez, Ofir Press, Ludwig Schmidt, Diyi Yang  

**Link**: [PDF](https://arxiv.org/pdf/2511.00839)  

**Abstract**: Current benchmarks for coding evaluate language models (LMs) on concrete, well-specified tasks such as fixing specific bugs or writing targeted tests. However, human programmers do not spend all day incessantly addressing isolated tasks. Instead, real-world software development is grounded in the pursuit of high-level goals, like improving user retention or reducing costs. Evaluating whether LMs can also iteratively develop code to better accomplish open-ended objectives without any explicit guidance remains an open challenge. To address this, we introduce CodeClash, a benchmark where LMs compete in multi-round tournaments to build the best codebase for achieving a competitive objective. Each round proceeds in two phases: agents edit their code, then their codebases compete head-to-head in a code arena that determines winners based on objectives like score maximization, resource acquisition, or survival. Whether it's writing notes, scrutinizing documentation, analyzing competition logs, or creating test suites, models must decide for themselves how to improve their codebases both absolutely and against their opponents. We run 1680 tournaments (25,200 rounds total) to evaluate 8 LMs across 6 arenas. Our results reveal that while models exhibit diverse development styles, they share fundamental limitations in strategic reasoning. Models also struggle with long-term codebase maintenance, as repositories become progressively messy and redundant. These limitations are stark: top models lose every round against expert human programmers. We open-source CodeClash to advance the study of autonomous, goal-oriented code development. 

---
# Parameter Interpolation Adversarial Training for Robust Image Classification 

**Authors**: Xin Liu, Yichen Yang, Kun He, John E. Hopcroft  

**Link**: [PDF](https://arxiv.org/pdf/2511.00836)  

**Abstract**: Though deep neural networks exhibit superior performance on various tasks, they are still plagued by adversarial examples. Adversarial training has been demonstrated to be the most effective method to defend against adversarial attacks. However, existing adversarial training methods show that the model robustness has apparent oscillations and overfitting issues in the training process, degrading the defense efficacy. To address these issues, we propose a novel framework called Parameter Interpolation Adversarial Training (PIAT). PIAT tunes the model parameters between each epoch by interpolating the parameters of the previous and current epochs. It makes the decision boundary of model change more moderate and alleviates the overfitting issue, helping the model converge better and achieving higher model robustness. In addition, we suggest using the Normalized Mean Square Error (NMSE) to further improve the robustness by aligning the relative magnitude of logits between clean and adversarial examples rather than the absolute magnitude. Extensive experiments conducted on several benchmark datasets demonstrate that our framework could prominently improve the robustness of both Convolutional Neural Networks (CNNs) and Vision Transformers (ViTs). 

---
# Linear Differential Vision Transformer: Learning Visual Contrasts via Pairwise Differentials 

**Authors**: Yifan Pu, Jixuan Ying, Qixiu Li, Tianzhu Ye, Dongchen Han, Xiaochen Wang, Ziyi Wang, Xinyu Shao, Gao Huang, Xiu Li  

**Link**: [PDF](https://arxiv.org/pdf/2511.00833)  

**Abstract**: Vision Transformers (ViTs) have become a universal backbone for both image recognition and image generation. Yet their Multi-Head Self-Attention (MHSA) layer still performs a quadratic query-key interaction for every token pair, spending the bulk of computation on visually weak or redundant correlations. We introduce Visual-Contrast Attention (VCA), a drop-in replacement for MHSA that injects an explicit notion of discrimination while reducing the theoretical complexity from O(N N C) to O(N n C) with n << N. VCA first distils each head's dense query field into a handful of spatially pooled visual-contrast tokens, then splits them into a learnable positive and negative stream whose differential interaction highlights what truly separates one region from another. The module adds fewer than 0.3M parameters to a DeiT-Tiny backbone, requires no extra FLOPs, and is wholly architecture-agnostic. Empirically, VCA lifts DeiT-Tiny top-1 accuracy on ImageNet-1K from 72.2% to 75.6% (+3.4) and improves three strong hierarchical ViTs by up to 3.1%, while in class-conditional ImageNet generation it lowers FID-50K by 2.1 to 5.2 points across both diffusion (DiT) and flow (SiT) models. Extensive ablations confirm that (i) spatial pooling supplies low-variance global cues, (ii) dual positional embeddings are indispensable for contrastive reasoning, and (iii) combining the two in both stages yields the strongest synergy. VCA therefore offers a simple path towards faster and sharper Vision Transformers. The source code is available at this https URL. 

---
# Enhancing Adversarial Transferability in Visual-Language Pre-training Models via Local Shuffle and Sample-based Attack 

**Authors**: Xin Liu, Aoyang Zhou, Aoyang Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2511.00831)  

**Abstract**: Visual-Language Pre-training (VLP) models have achieved significant performance across various downstream tasks. However, they remain vulnerable to adversarial examples. While prior efforts focus on improving the adversarial transferability of multimodal adversarial examples through cross-modal interactions, these approaches suffer from overfitting issues, due to a lack of input diversity by relying excessively on information from adversarial examples in one modality when crafting attacks in another. To address this issue, we draw inspiration from strategies in some adversarial training methods and propose a novel attack called Local Shuffle and Sample-based Attack (LSSA). LSSA randomly shuffles one of the local image blocks, thus expanding the original image-text pairs, generating adversarial images, and sampling around them. Then, it utilizes both the original and sampled images to generate the adversarial texts. Extensive experiments on multiple models and datasets demonstrate that LSSA significantly enhances the transferability of multimodal adversarial examples across diverse VLP models and downstream tasks. Moreover, LSSA outperforms other advanced attacks on Large Vision-Language Models. 

---
# Towards Ultra-Low Latency: Binarized Neural Network Architectures for In-Vehicle Network Intrusion Detection 

**Authors**: Huiyao Dong, Igor Kotenko  

**Link**: [PDF](https://arxiv.org/pdf/2511.00828)  

**Abstract**: The Control Area Network (CAN) protocol is essential for in-vehicle communication, facilitating high-speed data exchange among Electronic Control Units (ECUs). However, its inherent design lacks robust security features, rendering vehicles susceptible to cyberattacks. While recent research has investigated machine learning and deep learning techniques to enhance network security, their practical applicability remains uncertain. This paper presents a lightweight intrusion detection technique based on Binarized Neural Networks (BNNs), which utilizes payload data, message IDs, and CAN message frequencies for effective intrusion detection. Additionally, we develop hybrid binary encoding techniques to integrate non-binary features, such as message IDs and frequencies. The proposed method, namely the BNN framework specifically optimized for in-vehicle intrusion detection combined with hybrid binary quantization techniques for non-payload attributes, demonstrates efficacy in both anomaly detection and multi-class network traffic classification. The system is well-suited for deployment on micro-controllers and Gateway ECUs, aligning with the real-time requirements of CAN bus safety applications. 

---
# GUI-AIMA: Aligning Intrinsic Multimodal Attention with a Context Anchor for GUI Grounding 

**Authors**: Shijie Zhou, Viet Dac Lai, Hao Tan, Jihyung Kil, Wanrong Zhu, Changyou Chen, Ruiyi Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2511.00810)  

**Abstract**: Graphical user interface (GUI) grounding is a key function of computer-use agents, which maps natural-language instructions to actionable screen regions. Existing approaches based on Multimodal Large Language Models (MLLMs) typically formulate it as a text-based coordinate generation task, yet directly generating precise coordinates from visual inputs remains challenging and computationally intensive. An intuitive way to implement GUI grounding is to first select visual patches relevant to the instructions and then determine the precise click location within those patches. Based on the observations that general MLLMs have some native grounding capability, nested within their attentions, we propose GUI-AIMA, an attention-based and coordinate-free supervised fine-tuning framework for efficient GUI grounding. GUI-AIMA aligns the intrinsic multimodal attention of MLLMs with patch-wise grounding signals. These signals are calculated adaptively for diverse user instructions by multi-head aggregation on simplified query-visual attention matrices. Besides, its coordinate-free manner can easily integrate a plug-and-play zoom-in stage. GUI-AIMA-3B was trained with only 85k screenshots, demonstrating exceptional data efficiency and verifying that light training can trigger the native grounding capability of MLLMs. It achieves state-of-the-art performance among 3B models, attaining an average accuracy of 58.6% on ScreenSpot-Pro and 62.2% on OSWorld-G. Project page: this https URL 

---
# Logic-informed reinforcement learning for cross-domain optimization of large-scale cyber-physical systems 

**Authors**: Guangxi Wan, Peng Zeng, Xiaoting Dong, Chunhe Song, Shijie Cui, Dong Li, Qingwei Dong, Yiyang Liu, Hongfei Bai  

**Link**: [PDF](https://arxiv.org/pdf/2511.00806)  

**Abstract**: Cyber-physical systems (CPS) require the joint optimization of discrete cyber actions and continuous physical parameters under stringent safety logic constraints. However, existing hierarchical approaches often compromise global optimality, whereas reinforcement learning (RL) in hybrid action spaces often relies on brittle reward penalties, masking, or shielding and struggles to guarantee constraint satisfaction. We present logic-informed reinforcement learning (LIRL), which equips standard policy-gradient algorithms with projection that maps a low-dimensional latent action onto the admissible hybrid manifold defined on-the-fly by first-order logic. This guarantees feasibility of every exploratory step without penalty tuning. Experimental evaluations have been conducted across multiple scenarios, including industrial manufacturing, electric vehicle charging stations, and traffic signal control, in all of which the proposed method outperforms existing hierarchical optimization approaches. Taking a robotic reducer assembly system in industrial manufacturing as an example, LIRL achieves a 36.47\% to 44.33\% reduction at most in the combined makespan-energy objective compared to conventional industrial hierarchical scheduling methods. Meanwhile, it consistently maintains zero constraint violations and significantly surpasses state-of-the-art hybrid-action reinforcement learning baselines. Thanks to its declarative logic-based constraint formulation, the framework can be seamlessly transferred to other domains such as smart transportation and smart grid, thereby paving the way for safe and real-time optimization in large-scale CPS. 

---
# Attention Saturation and Gradient Suppression at Inflection Layers: Diagnosing and Mitigating Bottlenecks in Transformer Adaptation 

**Authors**: Wang Zixian  

**Link**: [PDF](https://arxiv.org/pdf/2511.00797)  

**Abstract**: Pre-trained Transformers often exhibit over-confidence in source patterns and difficulty in forming new target-domain patterns during fine-tuning. We formalize the mechanism of output saturation leading to gradient suppression through standard cross-entropy and softmax analysis, showing that gradient suppression at inflection layers confines adaptation to high-level recombination of existing features while preventing low-level reconstruction. We introduce a set of layer-wise diagnostic metrics -- attention entropy (saturation proxy), activation gradient norm, parameter gradient norm, and Delta-CKA under a shared PCA basis -- to identify inflection layers characterized by both low attention entropy and steep gradient decay. Building on these findings, we propose a diagnose-first, inject-light fine-tuning strategy: selectively inserting LoRA adapters at inflection layers to restore suppressed backward signals with minimal parameter overhead. Experiments on BERT-base transfer from SST-2 to Rotten Tomatoes under under-trained and over-trained source regimes reveal that over-trained initialization benefits from inflection-layer LoRA injection, while under-trained initialization suffers performance degradation. When base features are strong, unblocking inflection layers facilitates high-level compositional adaptation; when base features are weak, full-pathway unblocking is required for low-level reconstruction, as supported by joint analysis of layer-wise activation gradients and Delta-CKA dynamics. 

---
# FedOnco-Bench: A Reproducible Benchmark for Privacy-Aware Federated Tumor Segmentation with Synthetic CT Data 

**Authors**: Viswa Chaitanya Marella, Suhasnadh Reddy Veluru, Sai Teja Erukude  

**Link**: [PDF](https://arxiv.org/pdf/2511.00795)  

**Abstract**: Federated Learning (FL) allows multiple institutions to cooperatively train machine learning models while retaining sensitive data at the source, which has great utility in privacy-sensitive environments. However, FL systems remain vulnerable to membership-inference attacks and data heterogeneity. This paper presents FedOnco-Bench, a reproducible benchmark for privacy-aware FL using synthetic oncologic CT scans with tumor annotations. It evaluates segmentation performance and privacy leakage across FL methods: FedAvg, FedProx, FedBN, and FedAvg with DP-SGD. Results show a distinct trade-off between privacy and utility: FedAvg is high performance (Dice around 0.85) with more privacy leakage (attack AUC about 0.72), while DP-SGD provides a higher level of privacy (AUC around 0.25) at the cost of accuracy (Dice about 0.79). FedProx and FedBN offer balanced performance under heterogeneous data, especially with non-identical distributed client data. FedOnco-Bench serves as a standardized, open-source platform for benchmarking and developing privacy-preserving FL methods for medical image segmentation. 

---
# Efficient Reinforcement Learning for Large Language Models with Intrinsic Exploration 

**Authors**: Yan Sun, Jia Guo, Stanley Kok, Zihao Wang, Zujie Wen, Zhiqiang Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2511.00794)  

**Abstract**: Reinforcement learning with verifiable rewards (RLVR) has improved the reasoning ability of large language models, yet training remains costly because many rollouts contribute little to optimization, considering the amount of computation required. This study investigates how simply leveraging intrinsic data properties, almost free benefit during training, can improve data efficiency for RLVR. We propose PREPO with two complementary components. First, we adopt prompt perplexity as an indicator of model adaptability in learning, enabling the model to progress from well-understood contexts to more challenging ones. Second, we amplify the discrepancy among the rollouts by differentiating their relative entropy, and prioritize sequences that exhibit a higher degree of exploration. Together, these mechanisms reduce rollout demand while preserving competitive performance. On the Qwen and Llama models, PREPO achieves effective results on mathematical reasoning benchmarks with up to 3 times fewer rollouts than the baselines. Beyond empirical gains, we provide theoretical and in-depth analyses explaining the underlying rationale of our method to improve the data efficiency of RLVR. 

---
# Fast PINN Eigensolvers via Biconvex Reformulation 

**Authors**: Akshay Sai Banderwaar, Abhishek Gupta  

**Link**: [PDF](https://arxiv.org/pdf/2511.00792)  

**Abstract**: Eigenvalue problems have a distinctive forward-inverse structure and are fundamental to characterizing a system's thermal response, stability, and natural modes. Physics-Informed Neural Networks (PINNs) offer a mesh-free alternative for solving such problems but are often orders of magnitude slower than classical numerical schemes. In this paper, we introduce a reformulated PINN approach that casts the search for eigenpairs as a biconvex optimization problem, enabling fast and provably convergent alternating convex search (ACS) over eigenvalues and eigenfunctions using analytically optimal updates. Numerical experiments show that PINN-ACS attains high accuracy with convergence speeds up to 500$\times$ faster than gradient-based PINN training. We release our codes at this https URL. 

---
# Class-agnostic 3D Segmentation by Granularity-Consistent Automatic 2D Mask Tracking 

**Authors**: Juan Wang, Yasutomo Kawanishi, Tomo Miyazaki, Zhijie Wang, Shinichiro Omachi  

**Link**: [PDF](https://arxiv.org/pdf/2511.00785)  

**Abstract**: 3D instance segmentation is an important task for real-world applications. To avoid costly manual annotations, existing methods have explored generating pseudo labels by transferring 2D masks from foundation models to 3D. However, this approach is often suboptimal since the video frames are processed independently. This causes inconsistent segmentation granularity and conflicting 3D pseudo labels, which degrades the accuracy of final segmentation. To address this, we introduce a Granularity-Consistent automatic 2D Mask Tracking approach that maintains temporal correspondences across frames, eliminating conflicting pseudo labels. Combined with a three-stage curriculum learning framework, our approach progressively trains from fragmented single-view data to unified multi-view annotations, ultimately globally coherent full-scene supervision. This structured learning pipeline enables the model to progressively expose to pseudo-labels of increasing consistency. Thus, we can robustly distill a consistent 3D representation from initially fragmented and contradictory 2D priors. Experimental results demonstrated that our method effectively generated consistent and accurate 3D segmentations. Furthermore, the proposed method achieved state-of-the-art results on standard benchmarks and open-vocabulary ability. 

---
# Quantifying truth and authenticity in AI-assisted candidate evaluation: A multi-domain pilot analysis 

**Authors**: Eldred Lee, Nicholas Worley, Koshu Takatsuji  

**Link**: [PDF](https://arxiv.org/pdf/2511.00774)  

**Abstract**: This paper presents a retrospective analysis of anonymized candidate-evaluation data collected during pilot hiring campaigns conducted through AlteraSF, an AI-native resume-verification platform. The system evaluates resume claims, generates context-sensitive verification questions, and measures performance along quantitative axes of factual validity and job fit, complemented by qualitative integrity detection. Across six job families and 1,700 applications, the platform achieved a 90-95% reduction in screening time and detected measurable linguistic patterns consistent with AI-assisted or copied responses. The analysis demonstrates that candidate truthfulness can be assessed not only through factual accuracy but also through patterns of linguistic authenticity. The results suggest that a multi-dimensional verification framework can improve both hiring efficiency and trust in AI-mediated evaluation systems. 

---
# EP-HDC: Hyperdimensional Computing with Encrypted Parameters for High-Throughput Privacy-Preserving Inference 

**Authors**: Jaewoo Park, Chenghao Quan, Jongeun Lee  

**Link**: [PDF](https://arxiv.org/pdf/2511.00737)  

**Abstract**: While homomorphic encryption (HE) provides strong privacy protection, its high computational cost has restricted its application to simple tasks. Recently, hyperdimensional computing (HDC) applied to HE has shown promising performance for privacy-preserving machine learning (PPML). However, when applied to more realistic scenarios such as batch inference, the HDC-based HE has still very high compute time as well as high encryption and data transmission overheads. To address this problem, we propose HDC with encrypted parameters (EP-HDC), which is a novel PPML approach featuring client-side HE, i.e., inference is performed on a client using a homomorphically encrypted model. Our EP-HDC can effectively mitigate the encryption and data transmission overhead, as well as providing high scalability with many clients while providing strong protection for user data and model parameters. In addition to application examples for our client-side PPML, we also present design space exploration involving quantization, architecture, and HE-related parameters. Our experimental results using the BFV scheme and the Face/Emotion datasets demonstrate that our method can improve throughput and latency of batch inference by orders of magnitude over previous PPML methods (36.52~1068x and 6.45~733x, respectively) with less than 1% accuracy degradation. 

---
# FeNN-DMA: A RISC-V SoC for SNN acceleration 

**Authors**: Zainab Aizaz, James C. Knight, Thomas Nowotny  

**Link**: [PDF](https://arxiv.org/pdf/2511.00732)  

**Abstract**: Spiking Neural Networks (SNNs) are a promising, energy-efficient alternative to standard Artificial Neural Networks (ANNs) and are particularly well-suited to spatio-temporal tasks such as keyword spotting and video classification. However, SNNs have a much lower arithmetic intensity than ANNs and are therefore not well-matched to standard accelerators like GPUs and TPUs. Field Programmable Gate Arrays(FPGAs) are designed for such memory-bound workloads and here we develop a novel, fully-programmable RISC-V-based system-on-chip (FeNN-DMA), tailored to simulating SNNs on modern UltraScale+ FPGAs. We show that FeNN-DMA has comparable resource usage and energy requirements to state-of-the-art fixed-function SNN accelerators, yet it is capable of simulating much larger and more complex models. Using this functionality, we demonstrate state-of-the-art classification accuracy on the Spiking Heidelberg Digits and Neuromorphic MNIST tasks. 

---
# TRISKELION-1: Unified Descriptive-Predictive-Generative AI 

**Authors**: Nardeep Kumar, Arun Kanwar  

**Link**: [PDF](https://arxiv.org/pdf/2511.00711)  

**Abstract**: TRISKELION-1 is a unified descriptive-predictive-generative architecture that integrates statistical, mechanistic, and generative reasoning within a single encoder-decoder framework. The model demonstrates how descriptive representation learning, predictive inference, and generative synthesis can be jointly optimized using variational objectives. Experiments on MNIST validate that descriptive reconstruction, predictive classification, and generative sampling can coexist stably within one model. The framework provides a blueprint toward universal intelligence architectures that connect interpretability, accuracy, and creativity. 

---
# A Voice-Enabled Virtual Patient System for Interactive Training in Standardized Clinical Assessment 

**Authors**: Veronica Bossio Botero, Vijay Yadav, Jacob Ouyang, Anzar Abbas, Michelle Worthington  

**Link**: [PDF](https://arxiv.org/pdf/2511.00709)  

**Abstract**: Training mental health clinicians to conduct standardized clinical assessments is challenging due to a lack of scalable, realistic practice opportunities, which can impact data quality in clinical trials. To address this gap, we introduce a voice-enabled virtual patient simulation system powered by a large language model (LLM). This study describes the system's development and validates its ability to generate virtual patients who accurately adhere to pre-defined clinical profiles, maintain coherent narratives, and produce realistic dialogue. We implemented a system using a LLM to simulate patients with specified symptom profiles, demographics, and communication styles. The system was evaluated by 5 experienced clinical raters who conducted 20 simulated structured MADRS interviews across 4 virtual patient personas. The virtual patients demonstrated strong adherence to their clinical profiles, with a mean item difference between rater-assigned MADRS scores and configured scores of 0.52 (SD=0.75). Inter-rater reliability across items was 0.90 (95% CI=0.68-0.99). Expert raters consistently rated the qualitative realism and cohesiveness of the virtual patients favorably, giving average ratings between "Agree" and "Strongly Agree." Our findings suggest that LLM-powered virtual patient simulations are a viable and scalable tool for training clinicians, capable of producing high-fidelity, clinically relevant practice scenarios. 

---
# Evolve to Inspire: Novelty Search for Diverse Image Generation 

**Authors**: Alex Inch, Passawis Chaiyapattanaporn, Yuchen Zhu, Yuan Lu, Ting-Wen Ko, Davide Paglieri  

**Link**: [PDF](https://arxiv.org/pdf/2511.00686)  

**Abstract**: Text-to-image diffusion models, while proficient at generating high-fidelity im- ages, often suffer from limited output diversity, hindering their application in exploratory and ideation tasks. Existing prompt optimization techniques typically target aesthetic fitness or are ill-suited to the creative visual domain. To address this shortcoming, we introduce WANDER, a novelty search-based approach to generating diverse sets of images from a single input prompt. WANDER operates directly on natural language prompts, employing a Large Language Model (LLM) for semantic evolution of diverse sets of images, and using CLIP embeddings to quantify novelty. We additionally apply emitters to guide the search into distinct regions of the prompt space, and demonstrate that they boost the diversity of the generated images. Empirical evaluations using FLUX-DEV for generation and GPT-4o-mini for mutation demonstrate that WANDER significantly outperforms existing evolutionary prompt optimization baselines in diversity metrics. Ablation studies confirm the efficacy of emitters. 

---
# Metadata-Aligned 3D MRI Representations for Contrast Understanding and Quality Control 

**Authors**: Mehmet Yigit Avci, Pedro Borges, Virginia Fernandez, Paul Wright, Mehmet Yigitsoy, Sebastien Ourselin, Jorge Cardoso  

**Link**: [PDF](https://arxiv.org/pdf/2511.00681)  

**Abstract**: Magnetic Resonance Imaging suffers from substantial data heterogeneity and the absence of standardized contrast labels across scanners, protocols, and institutions, which severely limits large-scale automated analysis. A unified representation of MRI contrast would enable a wide range of downstream utilities, from automatic sequence recognition to harmonization and quality control, without relying on manual annotations. To this end, we introduce MR-CLIP, a metadata-guided framework that learns MRI contrast representations by aligning volumetric images with their DICOM acquisition parameters. The resulting embeddings shows distinct clusters of MRI sequences and outperform supervised 3D baselines under data scarcity in few-shot sequence classification. Moreover, MR-CLIP enables unsupervised data quality control by identifying corrupted or inconsistent metadata through image-metadata embedding distances. By transforming routinely available acquisition metadata into a supervisory signal, MR-CLIP provides a scalable foundation for label-efficient MRI analysis across diverse clinical datasets. 

---
# Isotropic Curvature Model for Understanding Deep Learning Optimization: Is Gradient Orthogonalization Optimal? 

**Authors**: Weijie Su  

**Link**: [PDF](https://arxiv.org/pdf/2511.00674)  

**Abstract**: In this paper, we introduce a model for analyzing deep learning optimization over a single iteration by leveraging the matrix structure of the weights. We derive the model by assuming isotropy of curvature, including the second-order Hessian and higher-order terms, of the loss function across all perturbation directions; hence, we call it the isotropic curvature model. This model is a convex optimization program amenable to analysis, which allows us to understand how an update on the weights in the form of a matrix relates to the change in the total loss function. As an application, we use the isotropic curvature model to analyze the recently introduced Muon optimizer and other matrix-gradient methods for training language models. First, we show that under a general growth condition on the curvature, the optimal update matrix is obtained by making the spectrum of the original gradient matrix more homogeneous -- that is, making its singular values closer in ratio -- which in particular improves the conditioning of the update matrix. Next, we show that the orthogonalized gradient becomes optimal for the isotropic curvature model when the curvature exhibits a phase transition in growth. Taken together, these results suggest that the gradient orthogonalization employed in Muon and other related methods is directionally correct but may not be strictly optimal. Finally, we discuss future research on how to leverage the isotropic curvature model for designing new optimization methods for training deep learning and language models. 

---
# ShadowLogic: Backdoors in Any Whitebox LLM 

**Authors**: Kasimir Schulz, Amelia Kawasaki, Leo Ring  

**Link**: [PDF](https://arxiv.org/pdf/2511.00664)  

**Abstract**: Large language models (LLMs) are widely deployed across various applications, often with safeguards to prevent the generation of harmful or restricted content. However, these safeguards can be covertly bypassed through adversarial modifications to the computational graph of a model. This work highlights a critical security vulnerability in computational graph-based LLM formats, demonstrating that widely used deployment pipelines may be susceptible to obscured backdoors. We introduce ShadowLogic, a method for creating a backdoor in a white-box LLM by injecting an uncensoring vector into its computational graph representation. We set a trigger phrase that, when added to the beginning of a prompt into the LLM, applies the uncensoring vector and removes the content generation safeguards in the model. We embed trigger logic directly into the computational graph which detects the trigger phrase in a prompt. To evade detection of our backdoor, we obfuscate this logic within the graph structure, making it similar to standard model functions. Our method requires minimal alterations to model parameters, making backdoored models appear benign while retaining the ability to generate uncensored responses when activated. We successfully implement ShadowLogic in Phi-3 and Llama 3.2, using ONNX for manipulating computational graphs. Implanting the uncensoring vector achieved a >60% attack success rate for further malicious queries. 

---
# Lessons Learned from the Use of Generative AI in Engineering and Quality Assurance of a WEB System for Healthcare 

**Authors**: Guilherme H. Travassos, Sabrina Rocha, Rodrigo Feitosa, Felipe Assis, Patricia Goncalves, Andre Gheventer, Larissa Galeno, Arthur Sasse, Julio Cesar Guimaraes, Carlos Brito, Joao Pedro Wieland  

**Link**: [PDF](https://arxiv.org/pdf/2511.00658)  

**Abstract**: The advances and availability of technologies involving Generative Artificial Intelligence (AI) are evolving clearly and explicitly, driving immediate changes in various work activities. Software Engineering (SE) is no exception and stands to benefit from these new technologies, enhancing productivity and quality in its software development processes. However, although the use of Generative AI in SE practices is still in its early stages, considering the lack of conclusive results from ongoing research and the limited technological maturity, we have chosen to incorporate these technologies in the development of a web-based software system to be used in clinical trials by a thoracic diseases research group at our university. For this reason, we decided to share this experience report documenting our development team's learning journey in using Generative AI during the software development process. Project management, requirements specification, design, development, and quality assurance activities form the scope of observation. Although we do not yet have definitive technological evidence to evolve our development process significantly, the results obtained and the suggestions shared here represent valuable insights for software organizations seeking to innovate their development practices to achieve software quality with generative AI. 

---
# More Than A Shortcut: A Hyperbolic Approach To Early-Exit Networks 

**Authors**: Swapnil Bhosale, Cosmin Frateanu, Camilla Clark, Arnoldas Jasonas, Chris Mitchell, Xiatian Zhu, Vamsi Krishna Ithapu, Giacomo Ferroni, Cagdas Bilen, Sanjeel Parekh  

**Link**: [PDF](https://arxiv.org/pdf/2511.00641)  

**Abstract**: Deploying accurate event detection on resource-constrained devices is challenged by the trade-off between performance and computational cost. While Early-Exit (EE) networks offer a solution through adaptive computation, they often fail to enforce a coherent hierarchical structure, limiting the reliability of their early predictions. To address this, we propose Hyperbolic Early-Exit networks (HypEE), a novel framework that learns EE representations in the hyperbolic space. Our core contribution is a hierarchical training objective with a novel entailment loss, which enforces a partial-ordering constraint to ensure that deeper network layers geometrically refine the representations of shallower ones. Experiments on multiple audio event detection tasks and backbone architectures show that HypEE significantly outperforms standard Euclidean EE baselines, especially at the earliest, most computationally-critical exits. The learned geometry also provides a principled measure of uncertainty, enabling a novel triggering mechanism that makes the overall system both more efficient and more accurate than a conventional EE and standard backbone models without early-exits. 

---
# Node Preservation and its Effect on Crossover in Cartesian Genetic Programming 

**Authors**: Mark Kocherovsky, Illya Bakurov, Wolfgang Banzhaf  

**Link**: [PDF](https://arxiv.org/pdf/2511.00634)  

**Abstract**: While crossover is a critical and often indispensable component in other forms of Genetic Programming, such as Linear- and Tree-based, it has consistently been claimed that it deteriorates search performance in CGP. As a result, a mutation-alone $(1+\lambda)$ evolutionary strategy has become the canonical approach for CGP. Although several operators have been developed that demonstrate an increased performance over the canonical method, a general solution to the problem is still lacking. In this paper, we compare basic crossover methods, namely one-point and uniform, to variants in which nodes are ``preserved,'' including the subgraph crossover developed by Roman Kalkreuth, the difference being that when ``node preservation'' is active, crossover is not allowed to break apart instructions. We also compare a node mutation operator to the traditional point mutation; the former simply replaces an entire node with a new one. We find that node preservation in both mutation and crossover improves search using symbolic regression benchmark problems, moving the field towards a general solution to CGP crossover. 

---
# AgentGit: A Version Control Framework for Reliable and Scalable LLM-Powered Multi-Agent Systems 

**Authors**: Yang Li, Siqi Ping, Xiyu Chen, Xiaojian Qi, Zigan Wang, Ye Luo, Xiaowei Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2511.00628)  

**Abstract**: With the rapid progress of large language models (LLMs), LLM-powered multi-agent systems (MAS) are drawing increasing interest across academia and industry. However, many current MAS frameworks struggle with reliability and scalability, especially on complex tasks. We present AgentGit, a framework that brings Git-like rollback and branching to MAS workflows. Built as an infrastructure layer on top of LangGraph, AgentGit supports state commit, revert, and branching, allowing agents to traverse, compare, and explore multiple trajectories efficiently. To evaluate AgentGit, we designed an experiment that optimizes target agents by selecting better prompts. We ran a multi-step A/B test against three baselines -- LangGraph, AutoGen, and Agno -- on a real-world task: retrieving and analyzing paper abstracts. Results show that AgentGit significantly reduces redundant computation, lowers runtime and token usage, and supports parallel exploration across multiple branches, enhancing both reliability and scalability in MAS development. This work offers a practical path to more robust MAS design and enables error recovery, safe exploration, iterative debugging, and A/B testing in collaborative AI systems. 

---
# Belief Dynamics Reveal the Dual Nature of In-Context Learning and Activation Steering 

**Authors**: Eric Bigelow, Daniel Wurgaft, YingQiao Wang, Noah Goodman, Tomer Ullman, Hidenori Tanaka, Ekdeep Singh Lubana  

**Link**: [PDF](https://arxiv.org/pdf/2511.00617)  

**Abstract**: Large language models (LLMs) can be controlled at inference time through prompts (in-context learning) and internal activations (activation steering). Different accounts have been proposed to explain these methods, yet their common goal of controlling model behavior raises the question of whether these seemingly disparate methodologies can be seen as specific instances of a broader framework. Motivated by this, we develop a unifying, predictive account of LLM control from a Bayesian perspective. Specifically, we posit that both context- and activation-based interventions impact model behavior by altering its belief in latent concepts: steering operates by changing concept priors, while in-context learning leads to an accumulation of evidence. This results in a closed-form Bayesian model that is highly predictive of LLM behavior across context- and activation-based interventions in a set of domains inspired by prior work on many-shot in-context learning. This model helps us explain prior empirical phenomena - e.g., sigmoidal learning curves as in-context evidence accumulates - while predicting novel ones - e.g., additivity of both interventions in log-belief space, which results in distinct phases such that sudden and dramatic behavioral shifts can be induced by slightly changing intervention controls. Taken together, this work offers a unified account of prompt-based and activation-based control of LLM behavior, and a methodology for empirically predicting the effects of these interventions. 

---
# EPARA: Parallelizing Categorized AI Inference in Edge Clouds 

**Authors**: Yubo Wang, Yubo Cui, Tuo Shi, Danyang Li, Wenxin Li, Lide Suo, Tao Wang, Xin Xie  

**Link**: [PDF](https://arxiv.org/pdf/2511.00603)  

**Abstract**: With the increasing adoption of AI applications such as large language models and computer vision AI, the computational demands on AI inference systems are continuously rising, making the enhancement of task processing capacity using existing hardware a primary objective in edge clouds. We propose EPARA, an end-to-end AI parallel inference framework in edge, aimed at enhancing the edge AI serving capability. Our key idea is to categorize tasks based on their sensitivity to latency/frequency and requirement for GPU resources, thereby achieving both request-level and service-level task-resource allocation. EPARA consists of three core components: 1) a task-categorized parallelism allocator that decides the parallel mode of each task, 2) a distributed request handler that performs the calculation for the specific request, and 3) a state-aware scheduler that periodically updates service placement in edge clouds. We implement a EPARA prototype and conduct a case study on the EPARA operation for LLMs and segmentation tasks. Evaluation through testbed experiments involving edge servers, embedded devices, and microcomputers shows that EPARA achieves up to 2.1$\times$ higher goodput in production workloads compared to prior frameworks, while adapting to various edge AI inference tasks. 

---
# Diagnosing Hallucination Risk in AI Surgical Decision-Support: A Sequential Framework for Sequential Validation 

**Authors**: Dong Chen, Yanzhe Wei, Zonglin He, Guan-Ming Kuang, Canhua Ye, Meiru An, Huili Peng, Yong Hu, Huiren Tao, Kenneth MC Cheung  

**Link**: [PDF](https://arxiv.org/pdf/2511.00588)  

**Abstract**: Large language models (LLMs) offer transformative potential for clinical decision support in spine surgery but pose significant risks through hallucinations, which are factually inconsistent or contextually misaligned outputs that may compromise patient safety. This study introduces a clinician-centered framework to quantify hallucination risks by evaluating diagnostic precision, recommendation quality, reasoning robustness, output coherence, and knowledge alignment. We assessed six leading LLMs across 30 expert-validated spinal cases. DeepSeek-R1 demonstrated superior overall performance (total score: 86.03 $\pm$ 2.08), particularly in high-stakes domains such as trauma and infection. A critical finding reveals that reasoning-enhanced model variants did not uniformly outperform standard counterparts: Claude-3.7-Sonnet's extended thinking mode underperformed relative to its standard version (80.79 $\pm$ 1.83 vs. 81.56 $\pm$ 1.92), indicating extended chain-of-thought reasoning alone is insufficient for clinical reliability. Multidimensional stress-testing exposed model-specific vulnerabilities, with recommendation quality degrading by 7.4% under amplified complexity. This decline contrasted with marginal improvements in rationality (+2.0%), readability (+1.7%) and diagnosis (+4.7%), highlighting a concerning divergence between perceived coherence and actionable guidance. Our findings advocate integrating interpretability mechanisms (e.g., reasoning chain visualization) into clinical workflows and establish a safety-aware validation framework for surgical LLM deployment. 

---
# TRACES: Temporal Recall with Contextual Embeddings for Real-Time Video Anomaly Detection 

**Authors**: Yousuf Ahmed Siddiqui, Sufiyaan Usmani, Umer Tariq, Jawwad Ahmed Shamsi, Muhammad Burhan Khan  

**Link**: [PDF](https://arxiv.org/pdf/2511.00580)  

**Abstract**: Video anomalies often depend on contextual information available and temporal evolution. Non-anomalous action in one context can be anomalous in some other context. Most anomaly detectors, however, do not notice this type of context, which seriously limits their capability to generalize to new, real-life situations. Our work addresses the context-aware zero-shot anomaly detection challenge, in which systems need to learn adaptively to detect new events by correlating temporal and appearance features with textual traces of memory in real time. Our approach defines a memory-augmented pipeline, correlating temporal signals with visual embeddings using cross-attention, and real-time zero-shot anomaly classification by contextual similarity scoring. We achieve 90.4\% AUC on UCF-Crime and 83.67\% AP on XD-Violence, a new state-of-the-art among zero-shot models. Our model achieves real-time inference with high precision and explainability for deployment. We show that, by fusing cross-attention temporal fusion and contextual memory, we achieve high fidelity anomaly detection, a step towards the applicability of zero-shot models in real-world surveillance and infrastructure monitoring. 

---
# FlashEVA: Accelerating LLM inference via Efficient Attention 

**Authors**: Juan Gabriel Kostelec, Qinghai Guo  

**Link**: [PDF](https://arxiv.org/pdf/2511.00576)  

**Abstract**: Transformer models have revolutionized natural language processing, achieving state-of-the-art performance and demonstrating remarkable scalability. However, their memory demands, particularly due to maintaining full context in memory, pose significant challenges for inference. In this paper, we present FlashEVA, an efficient implementation of EVA (Efficient Attention via Control Variates), and demonstrate how to finetune transformers to adapt to FlashEVA attention. Our method enables fine-tuning of Transformer models with as few as 1.5B tokens while preserving effectiveness across various downstream tasks. Notably, FlashEVA achieves up to 6.7x higher throughput and 5x lower peak GPU memory usage during inference compared to standard Transformer implementations. Despite these improvements, we observe limitations in retrieval-focused tasks. Our implementation offers control over the trade-off between throughput and accuracy through adjustable hyperparameters, providing flexibility for diverse use cases. This work represents a significant step towards more efficient and adaptable Transformer-based models for inference. 

---
# FTT-GRU: A Hybrid Fast Temporal Transformer with GRU for Remaining Useful Life Prediction 

**Authors**: Varun Teja Chirukiri, Udaya Bhasker Cheerala, Sandeep Kanta, Abdul Karim, Praveen Damacharla  

**Link**: [PDF](https://arxiv.org/pdf/2511.00564)  

**Abstract**: Accurate prediction of the remaining useful life (RUL) of industrial machinery is essential for reducing downtime and optimizing maintenance schedules. Existing approaches, such as long short-term memory (LSTM) networks and convolutional neural networks (CNNs), often struggle to model both global temporal dependencies and fine-grained degradation trends in multivariate sensor data. We propose a hybrid model, FTT-GRU, which combines a Fast Temporal Transformer (FTT) -- a lightweight Transformer variant using linearized attention via fast Fourier transform (FFT) -- with a gated recurrent unit (GRU) layer for sequential modeling. To the best of our knowledge, this is the first application of an FTT with a GRU for RUL prediction on NASA CMAPSS, enabling simultaneous capture of global and local degradation patterns in a compact architecture. On CMAPSS FD001, FTT-GRU attains RMSE 30.76, MAE 18.97, and $R^2=0.45$, with 1.12 ms CPU latency at batch=1. Relative to the best published deep baseline (TCN--Attention), it improves RMSE by 1.16\% and MAE by 4.00\%. Training curves averaged over $k=3$ runs show smooth convergence with narrow 95\% confidence bands, and ablations (GRU-only, FTT-only) support the contribution of both components. These results demonstrate that a compact Transformer-RNN hybrid delivers accurate and efficient RUL predictions on CMAPSS, making it suitable for real-time industrial prognostics. 

---
# Red-teaming Activation Probes using Prompted LLMs 

**Authors**: Phil Blandfort, Robert Graham  

**Link**: [PDF](https://arxiv.org/pdf/2511.00554)  

**Abstract**: Activation probes are attractive monitors for AI systems due to low cost and latency, but their real-world robustness remains underexplored. We ask: What failure modes arise under realistic, black-box adversarial pressure, and how can we surface them with minimal effort? We present a lightweight black-box red-teaming procedure that wraps an off-the-shelf LLM with iterative feedback and in-context learning (ICL), and requires no fine-tuning, gradients, or architectural access. Running a case study with probes for high-stakes interactions, we show that our approach can help discover valuable insights about a SOTA probe. Our analysis uncovers interpretable brittleness patterns (e.g., legalese-induced FPs; bland procedural tone FNs) and reduced but persistent vulnerabilities under scenario-constraint attacks. These results suggest that simple prompted red-teaming scaffolding can anticipate failure patterns before deployment and might yield promising, actionable insights to harden future probes. 

---
# Temporal Fusion Transformer for Multi-Horizon Probabilistic Forecasting of Weekly Retail Sales 

**Authors**: Santhi Bharath Punati, Sandeep Kanta, Udaya Bhasker Cheerala, Madhusudan G Lanjewar, Praveen Damacharla  

**Link**: [PDF](https://arxiv.org/pdf/2511.00552)  

**Abstract**: Accurate multi-horizon retail forecasts are critical for inventory and promotions. We present a novel study of weekly Walmart sales (45 stores, 2010--2012) using a Temporal Fusion Transformer (TFT) that fuses static store identifiers with time-varying exogenous signals (holidays, CPI, fuel price, temperature). The pipeline produces 1--5-week-ahead probabilistic forecasts via Quantile Loss, yielding calibrated 90\% prediction intervals and interpretability through variable-selection networks, static enrichment, and temporal attention. On a fixed 2012 hold-out dataset, TFT achieves an RMSE of \$57.9k USD per store-week and an $R^2$ of 0.9875. Across a 5-fold chronological cross-validation, the averages are RMSE = \$64.6k USD and $R^2$ = 0.9844, outperforming the XGB, CNN, LSTM, and CNN-LSTM baseline models. These results demonstrate practical value for inventory planning and holiday-period optimization, while maintaining model transparency. 

---
# Robust Single-Agent Reinforcement Learning for Regional Traffic Signal Control Under Demand Fluctuations 

**Authors**: Qiang Li, Jin Niu, Lina Yu  

**Link**: [PDF](https://arxiv.org/pdf/2511.00549)  

**Abstract**: Traffic congestion, primarily driven by intersection queuing, significantly impacts urban living standards, safety, environmental quality, and economic efficiency. While Traffic Signal Control (TSC) systems hold potential for congestion mitigation, traditional optimization models often fail to capture real-world traffic complexity and dynamics. This study introduces a novel single-agent reinforcement learning (RL) framework for regional adaptive TSC, circumventing the coordination complexities inherent in multi-agent systems through a centralized decision-making paradigm. The model employs an adjacency matrix to unify the encoding of road network topology, real-time queue states derived from probe vehicle data, and current signal timing parameters. Leveraging the efficient learning capabilities of the DreamerV3 world model, the agent learns control policies where actions sequentially select intersections and adjust their signal phase splits to regulate traffic inflow/outflow, analogous to a feedback control system. Reward design prioritizes queue dissipation, directly linking congestion metrics (queue length) to control actions. Simulation experiments conducted in SUMO demonstrate the model's effectiveness: under inference scenarios with multi-level (10%, 20%, 30%) Origin-Destination (OD) demand fluctuations, the framework exhibits robust anti-fluctuation capability and significantly reduces queue lengths. This work establishes a new paradigm for intelligent traffic control compatible with probe vehicle technology. Future research will focus on enhancing practical applicability by incorporating stochastic OD demand fluctuations during training and exploring regional optimization mechanisms for contingency events. 

---
# Air Pollution Forecasting in Bucharest 

**Authors**: Dragoş-Andrei Şerban, Răzvan-Alexandru Smădu, Dumitru-Clementin Cercel  

**Link**: [PDF](https://arxiv.org/pdf/2511.00532)  

**Abstract**: Air pollution, especially the particulate matter 2.5 (PM2.5), has become a growing concern in recent years, primarily in urban areas. Being exposed to air pollution is linked to developing numerous health problems, like the aggravation of respiratory diseases, cardiovascular disorders, lung function impairment, and even cancer or early death. Forecasting future levels of PM2.5 has become increasingly important over the past few years, as it can provide early warnings and help prevent diseases. This paper aims to design, fine-tune, test, and evaluate machine learning models for predicting future levels of PM2.5 over various time horizons. Our primary objective is to assess and compare the performance of multiple models, ranging from linear regression algorithms and ensemble-based methods to deep learning models, such as advanced recurrent neural networks and transformers, as well as large language models, on this forecasting task. 

---
# On Improvisation and Open-Endedness: Insights for Experiential AI 

**Authors**: Botao 'Amber' Hu  

**Link**: [PDF](https://arxiv.org/pdf/2511.00529)  

**Abstract**: Improvisation-the art of spontaneous creation that unfolds moment-to-moment without a scripted outcome-requires practitioners to continuously sense, adapt, and create anew. It is a fundamental mode of human creativity spanning music, dance, and everyday life. The open-ended nature of improvisation produces a stream of novel, unrepeatable moments-an aspect highly valued in artistic creativity. In parallel, open-endedness (OE)-a system's capacity for unbounded novelty and endless "interestingness"-is exemplified in natural or cultural evolution and has been considered "the last grand challenge" in artificial life (ALife). The rise of generative AI now raises the question in computational creativity (CC) research: What makes a "good" improvisation for AI? Can AI learn to improvise in a genuinely open-ended way? In this work-in-progress paper, we report insights from in-depth interviews with 6 experts in improvisation across dance, music, and contact improvisation. We draw systemic connections between human improvisational arts and the design of future experiential AI agents that could improvise alone or alongside humans-or even with other AI agents-embodying qualities of improvisation drawn from practice: active listening (umwelt and awareness), being in the time (mindfulness and ephemerality), embracing the unknown (source of randomness and serendipity), non-judgmental flow (acceptance and dynamical stability, balancing structure and surprise (unpredictable criticality at edge of chaos), imaginative metaphor (synaesthesia and planning), empathy, trust, boundary, and care (mutual theory of mind), and playfulness and intrinsic motivation (maintaining interestingness). 

---
# HIP-LLM: A Hierarchical Imprecise Probability Approach to Reliability Assessment of Large Language Models 

**Authors**: Robab Aghazadeh-Chakherlou, Qing Guo, Siddartha Khastgir, Peter Popov, Xiaoge Zhang, Xingyu Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2511.00527)  

**Abstract**: Large Language Models (LLMs) are increasingly deployed across diverse domains, raising the need for rigorous reliability assessment methods. Existing benchmark-based evaluations primarily offer descriptive statistics of model accuracy over datasets, providing limited insight into the probabilistic behavior of LLMs under real operational conditions. This paper introduces HIP-LLM, a Hierarchical Imprecise Probability framework for modeling and inferring LLM reliability. Building upon the foundations of software reliability engineering, HIP-LLM defines LLM reliability as the probability of failure-free operation over a specified number of future tasks under a given Operational Profile (OP). HIP-LLM represents dependencies across (sub-)domains hierarchically, enabling multi-level inference from subdomain to system-level reliability. HIP-LLM embeds imprecise priors to capture epistemic uncertainty and incorporates OPs to reflect usage contexts. It derives posterior reliability envelopes that quantify uncertainty across priors and data. Experiments on multiple benchmark datasets demonstrate that HIP-LLM offers a more accurate and standardized reliability characterization than existing benchmark and state-of-the-art approaches. A publicly accessible repository of HIP-LLM is provided. 

---
# Reasoning Planning for Language Models 

**Authors**: Bao Nguyen, Hieu Trung Nguyen, Ruifeng She, Xiaojin Fu, Viet Anh Nguyen  

**Link**: [PDF](https://arxiv.org/pdf/2511.00521)  

**Abstract**: Selecting an appropriate reasoning method for a given query remains a key challenge in language model generation. Existing approaches typically generate multiple candidate responses and use an aggregation strategy to select the output answer, often assuming that more candidate answers yield higher accuracy. We revisit this assumption through a rigorous theoretical analysis, deriving accuracy bounds for standard aggregation methods under fixed generation distributions and candidate sizes. Building on these insights, we introduce EPIC, an Ensemble Planning with Contrastive learning framework to learn a shared representation space that captures both model reasoning abilities and query-method compatibility. EPIC incorporates our probability bounds as a regularizer in a utility-driven optimization that balances accuracy and computational cost. Experiments on diverse mathematical reasoning tasks show that EPIC consistently selects optimal reasoning methods, improving accuracy while reducing computational overhead. Our code can be found at this https URL. 

---
# A Multimodal Dataset for Indoor Radio Mapping with 3D Point Clouds and RSSI 

**Authors**: Ljupcho Milosheski, Kuon Akiyama, Blaž Bertalanič, Jernej Hribar, Ryoichi Shinkuma  

**Link**: [PDF](https://arxiv.org/pdf/2511.00494)  

**Abstract**: The growing number of smart devices supporting bandwidth-intensive and latency-sensitive applications, such as real-time video analytics, smart sensing, and Extended Reality (XR), necessitates reliable wireless connectivity in indoor environments. Therein, accurate estimation of Radio Environment Maps (REMs) enables adaptive wireless network planning and optimization of Access Point (AP) placement. However, generating realistic REMs remains challenging due to the complexity of indoor spaces. To overcome this challenge, this paper introduces a multimodal dataset that integrates high-resolution 3D LiDAR scans with Wi-Fi Received Signal Strength Indicator (RSSI) measurements collected under 20 distinct AP configurations in a multi-room indoor environment. The dataset captures two measurement scenarios: the first without human presence in the environment, and the second with human presence. Thus, the presented dataset supports the study of dynamic environmental effects on wireless signal propagation. This resource is designed to facilitate research in data-driven wireless modeling, particularly in the context of emerging high-frequency standards such as IEEE 802.11be (Wi-Fi 7), and aims to advance the development of robust, high-capacity indoor communication systems. 

---
# Investigating Label Bias and Representational Sources of Age-Related Disparities in Medical Segmentation 

**Authors**: Aditya Parikh, Sneha Das, Aasa Feragen  

**Link**: [PDF](https://arxiv.org/pdf/2511.00477)  

**Abstract**: Algorithmic bias in medical imaging can perpetuate health disparities, yet its causes remain poorly understood in segmentation tasks. While fairness has been extensively studied in classification, segmentation remains underexplored despite its clinical importance. In breast cancer segmentation, models exhibit significant performance disparities against younger patients, commonly attributed to physiological differences in breast density. We audit the MAMA-MIA dataset, establishing a quantitative baseline of age-related bias in its automated labels, and reveal a critical Biased Ruler effect where systematically flawed labels for validation misrepresent a model's actual bias. However, whether this bias originates from lower-quality annotations (label bias) or from fundamentally more challenging image characteristics remains unclear. Through controlled experiments, we systematically refute hypotheses that the bias stems from label quality sensitivity or quantitative case difficulty imbalance. Balancing training data by difficulty fails to mitigate the disparity, revealing that younger patient cases are intrinsically harder to learn. We provide direct evidence that systemic bias is learned and amplified when training on biased, machine-generated labels, a critical finding for automated annotation pipelines. This work introduces a systematic framework for diagnosing algorithmic bias in medical segmentation and demonstrates that achieving fairness requires addressing qualitative distributional differences rather than merely balancing case counts. 

---
# Longitudinal Vestibular Schwannoma Dataset with Consensus-based Human-in-the-loop Annotations 

**Authors**: Navodini Wijethilake, Marina Ivory, Oscar MacCormac, Siddhant Kumar, Aaron Kujawa, Lorena Garcia-Foncillas Macias, Rebecca Burger, Amanda Hitchings, Suki Thomson, Sinan Barazi, Eleni Maratos, Rupert Obholzer, Dan Jiang, Fiona McClenaghan, Kazumi Chia, Omar Al-Salihi, Nick Thomas, Steve Connor, Tom Vercauteren, Jonathan Shapey  

**Link**: [PDF](https://arxiv.org/pdf/2511.00472)  

**Abstract**: Accurate segmentation of vestibular schwannoma (VS) on Magnetic Resonance Imaging (MRI) is essential for patient management but often requires time-intensive manual annotations by experts. While recent advances in deep learning (DL) have facilitated automated segmentation, challenges remain in achieving robust performance across diverse datasets and complex clinical cases. We present an annotated dataset stemming from a bootstrapped DL-based framework for iterative segmentation and quality refinement of VS in MRI. We combine data from multiple centres and rely on expert consensus for trustworthiness of the annotations. We show that our approach enables effective and resource-efficient generalisation of automated segmentation models to a target data distribution. The framework achieved a significant improvement in segmentation accuracy with a Dice Similarity Coefficient (DSC) increase from 0.9125 to 0.9670 on our target internal validation dataset, while maintaining stable performance on representative external datasets. Expert evaluation on 143 scans further highlighted areas for model refinement, revealing nuanced cases where segmentation required expert intervention. The proposed approach is estimated to enhance efficiency by approximately 37.4% compared to the conventional manual annotation process. Overall, our human-in-the-loop model training approach achieved high segmentation accuracy, highlighting its potential as a clinically adaptable and generalisable strategy for automated VS segmentation in diverse clinical settings. The dataset includes 190 patients, with tumour annotations available for 534 longitudinal contrast-enhanced T1-weighted (T1CE) scans from 184 patients, and non-annotated T2-weighted scans from 6 patients. This dataset is publicly accessible on The Cancer Imaging Archive (TCIA) (this https URL). 

---
# Why Federated Optimization Fails to Achieve Perfect Fitting? A Theoretical Perspective on Client-Side Optima 

**Authors**: Zhongxiang Lei, Qi Yang, Ping Qiu, Gang Zhang, Yuanchi Ma, Jinyan Liu  

**Link**: [PDF](https://arxiv.org/pdf/2511.00469)  

**Abstract**: Federated optimization is a constrained form of distributed optimization that enables training a global model without directly sharing client data. Although existing algorithms can guarantee convergence in theory and often achieve stable training in practice, the reasons behind performance degradation under data heterogeneity remain unclear. To address this gap, the main contribution of this paper is to provide a theoretical perspective that explains why such degradation occurs. We introduce the assumption that heterogeneous client data lead to distinct local optima, and show that this assumption implies two key consequences: 1) the distance among clients' local optima raises the lower bound of the global objective, making perfect fitting of all client data impossible; and 2) in the final training stage, the global model oscillates within a region instead of converging to a single optimum, limiting its ability to fully fit the data. These results provide a principled explanation for performance degradation in non-iid settings, which we further validate through experiments across multiple tasks and neural network architectures. The framework used in this paper is open-sourced at: this https URL. 

---
# Proactive DDoS Detection and Mitigation in Decentralized Software-Defined Networking via Port-Level Monitoring and Zero-Training Large Language Models 

**Authors**: Mohammed N. Swileh, Shengli Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2511.00460)  

**Abstract**: Centralized Software-Defined Networking (cSDN) offers flexible and programmable control of networks but suffers from scalability and reliability issues due to its reliance on centralized controllers. Decentralized SDN (dSDN) alleviates these concerns by distributing control across multiple local controllers, yet this architecture remains highly vulnerable to Distributed Denial-of-Service (DDoS) attacks. In this paper, we propose a novel detection and mitigation framework tailored for dSDN environments. The framework leverages lightweight port-level statistics combined with prompt engineering and in-context learning, enabling the DeepSeek-v3 Large Language Model (LLM) to classify traffic as benign or malicious without requiring fine-tuning or retraining. Once an anomaly is detected, mitigation is enforced directly at the attacker's port, ensuring that malicious traffic is blocked at their origin while normal traffic remains unaffected. An automatic recovery mechanism restores normal operation after the attack inactivity, ensuring both security and availability. Experimental evaluation under diverse DDoS attack scenarios demonstrates that the proposed approach achieves near-perfect detection, with 99.99% accuracy, 99.97% precision, 100% recall, 99.98% F1-score, and an AUC of 1.0. These results highlight the effectiveness of combining distributed monitoring with zero-training LLM inference, providing a proactive and scalable defense mechanism for securing dSDN infrastructures against DDoS threats. 

---
# DRIP: Defending Prompt Injection via De-instruction Training and Residual Fusion Model Architecture 

**Authors**: Ruofan Liu, Yun Lin, Jin Song Dong  

**Link**: [PDF](https://arxiv.org/pdf/2511.00447)  

**Abstract**: Large language models (LLMs) have demonstrated impressive instruction-following capabilities. However, these capabilities also expose models to prompt injection attacks, where maliciously crafted inputs overwrite or distract from the intended instructions. A core vulnerability lies in the model's lack of semantic role understanding: it cannot distinguish directive intent from descriptive content, leading it to execute instruction-like phrases embedded in data.
We propose DRIP, a training-time defense grounded in a semantic modeling perspective, which enforces robust separation between instruction and data semantics without sacrificing utility. DRIP introduces two lightweight yet complementary mechanisms: (1) a token-wise de-instruction shift that performs semantic disentanglement, weakening directive semantics in data tokens while preserving content meaning; and (2) a residual fusion pathway that provides a persistent semantic anchor, reinforcing the influence of the true top-level instruction during generation. Experimental results on LLaMA-8B and Mistral-7B across three prompt injection benchmarks (SEP, AlpacaFarm, and InjecAgent) demonstrate that DRIP outperforms state-of-the-art defenses, including StruQ, SecAlign, ISE, and PFT, improving role separation by 49%, and reducing attack success rate by 66% for adaptive attacks. Meanwhile, DRIP's utility is on par with the undefended model across AlpacaEval, IFEval, and MT-Bench. Our findings underscore the power of lightweight representation edits and role-aware supervision in securing LLMs against adaptive prompt injection. 

---
# LIR: The First Workshop on Late Interaction and Multi Vector Retrieval @ ECIR 2026 

**Authors**: Benjamin Clavié, Xianming Li, Antoine Chaffin, Omar Khattab, Tom Aarsen, Manuel Faysse, Jing Li  

**Link**: [PDF](https://arxiv.org/pdf/2511.00444)  

**Abstract**: Late interaction retrieval methods, pioneered by ColBERT, have emerged as a powerful alternative to single-vector neural IR. By leveraging fine-grained, token-level representations, they have been demonstrated to deliver strong generalisation and robustness, particularly in out-of-domain settings. They have recently been shown to be particularly well-suited for novel use cases, such as reasoning-based or cross-modality retrieval. At the same time, these models pose significant challenges of efficiency, usability, and integrations into fully fledged systems; as well as the natural difficulties encountered while researching novel application domains. Recent years have seen rapid advances across many of these areas, but research efforts remain fragmented across communities and frequently exclude practitioners. The purpose of this workshop is to create an environment where all aspects of late interaction can be discussed, with a focus on early research explorations, real-world outcomes, and negative or puzzling results to be freely shared and discussed. The aim of LIR is to provide a highly-interactive environment for researchers from various backgrounds and practitioners to freely discuss their experience, fostering further collaboration. 

---
# Region-Aware Reconstruction Strategy for Pre-training fMRI Foundation Model 

**Authors**: Ruthwik Reddy Doodipala, Pankaj Pandey, Carolina Torres Rojas, Manob Jyoti Saikia, Ranganatha Sitaram  

**Link**: [PDF](https://arxiv.org/pdf/2511.00443)  

**Abstract**: The emergence of foundation models in neuroimaging is driven by the increasing availability of large-scale and heterogeneous brain imaging datasets. Recent advances in self-supervised learning, particularly reconstruction-based objectives, have demonstrated strong potential for pretraining models that generalize effectively across diverse downstream functional MRI (fMRI) tasks. In this study, we explore region-aware reconstruction strategies for a foundation model in resting-state fMRI, moving beyond approaches that rely on random region masking. Specifically, we introduce an ROI-guided masking strategy using the Automated Anatomical Labelling Atlas (AAL3), applied directly to full 4D fMRI volumes to selectively mask semantically coherent brain regions during self-supervised pretraining. Using the ADHD-200 dataset comprising 973 subjects with resting-state fMRI scans, we show that our method achieves a 4.23% improvement in classification accuracy for distinguishing healthy controls from individuals diagnosed with ADHD, compared to conventional random masking. Region-level attribution analysis reveals that brain volumes within the limbic region and cerebellum contribute most significantly to reconstruction fidelity and model representation. Our results demonstrate that masking anatomical regions during model pretraining not only enhances interpretability but also yields more robust and discriminative representations. In future work, we plan to extend this approach by evaluating it on additional neuroimaging datasets, and developing new loss functions explicitly derived from region-aware reconstruction objectives. These directions aim to further improve the robustness and interpretability of foundation models for functional neuroimaging. 

---
# Enhancing Frequency Forgery Clues for Diffusion-Generated Image Detection 

**Authors**: Daichi Zhang, Tong Zhang, Shiming Ge, Sabine Süsstrunk  

**Link**: [PDF](https://arxiv.org/pdf/2511.00429)  

**Abstract**: Diffusion models have achieved remarkable success in image synthesis, but the generated high-quality images raise concerns about potential malicious use. Existing detectors often struggle to capture discriminative clues across different models and settings, limiting their generalization to unseen diffusion models and robustness to various perturbations. To address this issue, we observe that diffusion-generated images exhibit progressively larger differences from natural real images across low- to high-frequency bands. Based on this insight, we propose a simple yet effective representation by enhancing the Frequency Forgery Clue (F^2C) across all frequency bands. Specifically, we introduce a frequency-selective function which serves as a weighted filter to the Fourier spectrum, suppressing less discriminative bands while enhancing more informative ones. This approach, grounded in a comprehensive analysis of frequency-based differences between natural real and diffusion-generated images, enables general detection of images from unseen diffusion models and provides robust resilience to various perturbations. Extensive experiments on various diffusion-generated image datasets demonstrate that our method outperforms state-of-the-art detectors with superior generalization and robustness. 

---
# Leveraging Hierarchical Image-Text Misalignment for Universal Fake Image Detection 

**Authors**: Daichi Zhang, Tong Zhang, Jianmin Bao, Shiming Ge, Sabine Süsstrunk  

**Link**: [PDF](https://arxiv.org/pdf/2511.00427)  

**Abstract**: With the rapid development of generative models, detecting generated fake images to prevent their malicious use has become a critical issue recently. Existing methods frame this challenge as a naive binary image classification task. However, such methods focus only on visual clues, yielding trained detectors susceptible to overfitting specific image patterns and incapable of generalizing to unseen models. In this paper, we address this issue from a multi-modal perspective and find that fake images cannot be properly aligned with corresponding captions compared to real images. Upon this observation, we propose a simple yet effective detector termed ITEM by leveraging the image-text misalignment in a joint visual-language space as discriminative clues. Specifically, we first measure the misalignment of the images and captions in pre-trained CLIP's space, and then tune a MLP head to perform the usual detection task. Furthermore, we propose a hierarchical misalignment scheme that first focuses on the whole image and then each semantic object described in the caption, which can explore both global and fine-grained local semantic misalignment as clues. Extensive experiments demonstrate the superiority of our method against other state-of-the-art competitors with impressive generalization and robustness on various recent generative models. 

---
# Bootstrap Off-policy with World Model 

**Authors**: Guojian Zhan, Likun Wang, Xiangteng Zhang, Jiaxin Gao, Masayoshi Tomizuka, Shengbo Eben Li  

**Link**: [PDF](https://arxiv.org/pdf/2511.00423)  

**Abstract**: Online planning has proven effective in reinforcement learning (RL) for improving sample efficiency and final performance. However, using planning for environment interaction inevitably introduces a divergence between the collected data and the policy's actual behaviors, degrading both model learning and policy improvement. To address this, we propose BOOM (Bootstrap Off-policy with WOrld Model), a framework that tightly integrates planning and off-policy learning through a bootstrap loop: the policy initializes the planner, and the planner refines actions to bootstrap the policy through behavior alignment. This loop is supported by a jointly learned world model, which enables the planner to simulate future trajectories and provides value targets to facilitate policy improvement. The core of BOOM is a likelihood-free alignment loss that bootstraps the policy using the planner's non-parametric action distribution, combined with a soft value-weighted mechanism that prioritizes high-return behaviors and mitigates variability in the planner's action quality within the replay buffer. Experiments on the high-dimensional DeepMind Control Suite and Humanoid-Bench show that BOOM achieves state-of-the-art results in both training stability and final performance. The code is accessible at this https URL. 

---
# MedRECT: A Medical Reasoning Benchmark for Error Correction in Clinical Texts 

**Authors**: Naoto Iwase, Hiroki Okuyama, Junichiro Iwasawa  

**Link**: [PDF](https://arxiv.org/pdf/2511.00421)  

**Abstract**: Large language models (LLMs) show increasing promise in medical applications, but their ability to detect and correct errors in clinical texts -- a prerequisite for safe deployment -- remains under-evaluated, particularly beyond English. We introduce MedRECT, a cross-lingual benchmark (Japanese/English) that formulates medical error handling as three subtasks: error detection, error localization (sentence extraction), and error correction. MedRECT is built with a scalable, automated pipeline from the Japanese Medical Licensing Examinations (JMLE) and a curated English counterpart, yielding MedRECT-ja (663 texts) and MedRECT-en (458 texts) with comparable error/no-error balance. We evaluate 9 contemporary LLMs spanning proprietary, open-weight, and reasoning families. Key findings: (i) reasoning models substantially outperform standard architectures, with up to 13.5% relative improvement in error detection and 51.0% in sentence extraction; (ii) cross-lingual evaluation reveals 5-10% performance gaps from English to Japanese, with smaller disparities for reasoning models; (iii) targeted LoRA fine-tuning yields asymmetric improvements in error correction performance (Japanese: +0.078, English: +0.168) while preserving reasoning capabilities; and (iv) our fine-tuned model exceeds human expert performance on structured medical error correction tasks. To our knowledge, MedRECT is the first comprehensive cross-lingual benchmark for medical error correction, providing a reproducible framework and resources for developing safer medical LLMs across languages. 

---
# LGCA: Enhancing Semantic Representation via Progressive Expansion 

**Authors**: Thanh Hieu Cao, Trung Khang Tran, Gia Thinh Pham, Tuong Nghiem Diep, Thanh Binh Nguyen  

**Link**: [PDF](https://arxiv.org/pdf/2511.00419)  

**Abstract**: Recent advancements in large-scale pretraining in natural language processing have enabled pretrained vision-language models such as CLIP to effectively align images and text, significantly improving performance in zero-shot image classification tasks. Subsequent studies have further demonstrated that cropping images into smaller regions and using large language models to generate multiple descriptions for each caption can further enhance model performance. However, due to the inherent sensitivity of CLIP, random image crops can introduce misinformation and bias, as many images share similar features at small scales. To address this issue, we propose Localized-Globalized Cross-Alignment (LGCA), a framework that first captures the local features of an image and then repeatedly selects the most salient regions and expands them. The similarity score is designed to incorporate both the original and expanded images, enabling the model to capture both local and global features while minimizing misinformation. Additionally, we provide a theoretical analysis demonstrating that the time complexity of LGCA remains the same as that of the original model prior to the repeated expansion process, highlighting its efficiency and scalability. Extensive experiments demonstrate that our method substantially improves zero-shot performance across diverse datasets, outperforming state-of-the-art baselines. 

---
# Human-AI Programming Role Optimization: Developing a Personality-Driven Self-Determination Framework 

**Authors**: Marcel Valovy  

**Link**: [PDF](https://arxiv.org/pdf/2511.00417)  

**Abstract**: As artificial intelligence transforms software development, a critical question emerges: how can developers and AI systems collaborate most effectively? This dissertation optimizes human-AI programming roles through self-determination theory and personality psychology, introducing the Role Optimization Motivation Alignment (ROMA) framework.
Through Design Science Research spanning five cycles, this work establishes empirically-validated connections between personality traits, programming role preferences, and collaborative outcomes, engaging 200 experimental participants and 46 interview respondents.
Key findings demonstrate that personality-driven role optimization significantly enhances self-determination and team dynamics, yielding 23% average motivation increases among professionals and up to 65% among undergraduates. Five distinct personality archetypes emerge: The Explorer (high Openness/low Agreeableness), The Orchestrator (high Extraversion/Agreeableness), The Craftsperson (high Neuroticism/low Extraversion), The Architect (high Conscientiousness), and The Adapter (balanced profile). Each exhibits distinct preferences for programming roles (Co-Pilot, Co-Navigator, Agent), with assignment modes proving crucial for satisfaction.
The dissertation contributes: (1) an empirically-validated framework linking personality traits to role preferences and self-determination outcomes; (2) a taxonomy of AI collaboration modalities mapped to personality profiles while preserving human agency; and (3) an ISO/IEC 29110 extension enabling Very Small Entities to implement personality-driven role optimization within established standards.
Keywords: artificial intelligence, human-computer interaction, behavioral software engineering, self-determination theory, personality psychology, phenomenology, intrinsic motivation, pair programming, design science research, ISO/IEC 29110 

---
# PADBen: A Comprehensive Benchmark for Evaluating AI Text Detectors Against Paraphrase Attacks 

**Authors**: Yiwei Zha, Rui Min, Shanu Sushmita  

**Link**: [PDF](https://arxiv.org/pdf/2511.00416)  

**Abstract**: While AI-generated text (AIGT) detectors achieve over 90\% accuracy on direct LLM outputs, they fail catastrophically against iteratively-paraphrased content. We investigate why iteratively-paraphrased text -- itself AI-generated -- evades detection systems designed for AIGT identification. Through intrinsic mechanism analysis, we reveal that iterative paraphrasing creates an intermediate laundering region characterized by semantic displacement with preserved generation patterns, which brings up two attack categories: paraphrasing human-authored text (authorship obfuscation) and paraphrasing LLM-generated text (plagiarism evasion). To address these vulnerabilities, we introduce PADBen, the first benchmark systematically evaluating detector robustness against both paraphrase attack scenarios. PADBen comprises a five-type text taxonomy capturing the full trajectory from original content to deeply laundered text, and five progressive detection tasks across sentence-pair and single-sentence challenges. We evaluate 11 state-of-the-art detectors, revealing critical asymmetry: detectors successfully identify the plagiarism evasion problem but fail for the case of authorship obfuscation. Our findings demonstrate that current detection approaches cannot effectively handle the intermediate laundering region, necessitating fundamental advances in detection architectures beyond existing semantic and stylistic discrimination methods. For detailed code implementation, please see this https URL. 

---
# Enhancing Adversarial Transferability by Balancing Exploration and Exploitation with Gradient-Guided Sampling 

**Authors**: Zenghao Niu, Weicheng Xie, Siyang Song, Zitong Yu, Feng Liu, Linlin Shen  

**Link**: [PDF](https://arxiv.org/pdf/2511.00411)  

**Abstract**: Adversarial attacks present a critical challenge to deep neural networks' robustness, particularly in transfer scenarios across different model architectures. However, the transferability of adversarial attacks faces a fundamental dilemma between Exploitation (maximizing attack potency) and Exploration (enhancing cross-model generalization). Traditional momentum-based methods over-prioritize Exploitation, i.e., higher loss maxima for attack potency but weakened generalization (narrow loss surface). Conversely, recent methods with inner-iteration sampling over-prioritize Exploration, i.e., flatter loss surfaces for cross-model generalization but weakened attack potency (suboptimal local maxima). To resolve this dilemma, we propose a simple yet effective Gradient-Guided Sampling (GGS), which harmonizes both objectives through guiding sampling along the gradient ascent direction to improve both sampling efficiency and stability. Specifically, based on MI-FGSM, GGS introduces inner-iteration random sampling and guides the sampling direction using the gradient from the previous inner-iteration (the sampling's magnitude is determined by a random distribution). This mechanism encourages adversarial examples to reside in balanced regions with both flatness for cross-model generalization and higher local maxima for strong attack potency. Comprehensive experiments across multiple DNN architectures and multimodal large language models (MLLMs) demonstrate the superiority of our method over state-of-the-art transfer attacks. Code is made available at this https URL. 

---
# Quantum Machine Unlearning: Foundations, Mechanisms, and Taxonomy 

**Authors**: Thanveer Shaik, Xiaohui Tao, Haoran Xie, Robert Sang  

**Link**: [PDF](https://arxiv.org/pdf/2511.00406)  

**Abstract**: Quantum Machine Unlearning has emerged as a foundational challenge at the intersection of quantum information theory privacypreserving computation and trustworthy artificial intelligence This paper advances QMU by establishing a formal framework that unifies physical constraints algorithmic mechanisms and ethical governance within a verifiable paradigm We define forgetting as a contraction of distinguishability between pre and postunlearning models under completely positive trace-preserving dynamics grounding data removal in the physics of quantum irreversibility Building on this foundation we present a fiveaxis taxonomy spanning scope guarantees mechanisms system context and hardware realization linking theoretical constructs to implementable strategies Within this structure we incorporate influence and quantum Fisher information weighted updates parameter reinitialization and kernel alignment as practical mechanisms compatible with noisy intermediatescale quantum NISQ devices The framework extends naturally to federated and privacyaware settings via quantum differential privacy homomorphic encryption and verifiable delegation enabling scalable auditable deletion across distributed quantum systems Beyond technical design we outline a forwardlooking research roadmap emphasizing formal proofs of forgetting scalable and secure architectures postunlearning interpretability and ethically auditable governance Together these contributions elevate QMU from a conceptual notion to a rigorously defined and ethically aligned discipline bridging physical feasibility algorithmic verifiability and societal accountability in the emerging era of quantum intelligence. 

---
# UME-R1: Exploring Reasoning-Driven Generative Multimodal Embeddings 

**Authors**: Zhibin Lan, Liqiang Niu, Fandong Meng, Jie Zhou, Jinsong Su  

**Link**: [PDF](https://arxiv.org/pdf/2511.00405)  

**Abstract**: The remarkable success of multimodal large language models (MLLMs) has driven advances in multimodal embeddings, yet existing models remain inherently discriminative, limiting their ability to benefit from reasoning-driven generation paradigm. In this work, we pioneer the exploration of generative embeddings, unifying embedding tasks within a generative paradigm. We propose UME-R1, a universal multimodal embedding framework consisting of a two-stage training strategy: a cold-start supervised fine-tuning equips the model with reasoning capabilities and enables it to generate both discriminative and generative embeddings; a subsequent reinforcement learning enhances reasoning and further optimizes generative embedding quality. This pioneering work reveals four key insights: 1) generative embeddings unlock substantial performance gains over conventional discriminative embeddings by leveraging the powerful generative reasoning capabilities of MLLMs; 2) discriminative and generative embeddings are complementary, whose combined oracle performance far exceeding that of either alone; 3) RL can effectively enhance generative embeddings, establishing a scalable optimization paradigm.; 4) repeated sampling at inference boosts downstream task coverage (pass@k), highlighting the inference-time scalability potential of generative embeddings. Evaluated on the MMEB-V2 benchmark across 78 tasks spanning video, image, and visual documents, UME-R1 significantly outperforms conventional discriminative embedding models and offers a foundation for more interpretable, reasoning-driven generative multimodal embeddings. Our code, models, and datasets will be publicly available at this https URL. 

---
# Emotion Detection in Speech Using Lightweight and Transformer-Based Models: A Comparative and Ablation Study 

**Authors**: Lucky Onyekwelu-Udoka, Md Shafiqul Islam, Md Shahedul Hasan  

**Link**: [PDF](https://arxiv.org/pdf/2511.00402)  

**Abstract**: Emotion recognition from speech plays a vital role in the development of empathetic human-computer interaction systems. This paper presents a comparative analysis of lightweight transformer-based models, DistilHuBERT and PaSST, by classifying six core emotions from the CREMA-D dataset. We benchmark their performance against a traditional CNN-LSTM baseline model using MFCC features. DistilHuBERT demonstrates superior accuracy (70.64%) and F1 score (70.36%) while maintaining an exceptionally small model size (0.02 MB), outperforming both PaSST and the baseline. Furthermore, we conducted an ablation study on three variants of the PaSST, Linear, MLP, and Attentive Pooling heads, to understand the effect of classification head architecture on model performance. Our results indicate that PaSST with an MLP head yields the best performance among its variants but still falls short of DistilHuBERT. Among the emotion classes, angry is consistently the most accurately detected, while disgust remains the most challenging. These findings suggest that lightweight transformers like DistilHuBERT offer a compelling solution for real-time speech emotion recognition on edge devices. The code is available at: this https URL. 

---
# SonarSweep: Fusing Sonar and Vision for Robust 3D Reconstruction via Plane Sweeping 

**Authors**: Lingpeng Chen, Jiakun Tang, Apple Pui-Yi Chui, Ziyang Hong, Junfeng Wu  

**Link**: [PDF](https://arxiv.org/pdf/2511.00392)  

**Abstract**: Accurate 3D reconstruction in visually-degraded underwater environments remains a formidable challenge. Single-modality approaches are insufficient: vision-based methods fail due to poor visibility and geometric constraints, while sonar is crippled by inherent elevation ambiguity and low resolution. Consequently, prior fusion technique relies on heuristics and flawed geometric assumptions, leading to significant artifacts and an inability to model complex scenes. In this paper, we introduce SonarSweep, a novel, end-to-end deep learning framework that overcomes these limitations by adapting the principled plane sweep algorithm for cross-modal fusion between sonar and visual data. Extensive experiments in both high-fidelity simulation and real-world environments demonstrate that SonarSweep consistently generates dense and accurate depth maps, significantly outperforming state-of-the-art methods across challenging conditions, particularly in high turbidity. To foster further research, we will publicly release our code and a novel dataset featuring synchronized stereo-camera and sonar data, the first of its kind. 

---
# Who Can We Trust? Scope-Aware Video Moment Retrieval with Multi-Agent Conflict 

**Authors**: Chaochen Wu, Guan Luo, Meiyun Zuo, Zhitao Fan  

**Link**: [PDF](https://arxiv.org/pdf/2511.00370)  

**Abstract**: Video moment retrieval uses a text query to locate a moment from a given untrimmed video reference. Locating corresponding video moments with text queries helps people interact with videos efficiently. Current solutions for this task have not considered conflict within location results from different models, so various models cannot integrate correctly to produce better results. This study introduces a reinforcement learning-based video moment retrieval model that can scan the whole video once to find the moment's boundary while producing its locational evidence. Moreover, we proposed a multi-agent system framework that can use evidential learning to resolve conflicts between agents' localization output. As a side product of observing and dealing with conflicts between agents, we can decide whether a query has no corresponding moment in a video (out-of-scope) without additional training, which is suitable for real-world applications. Extensive experiments on benchmark datasets show the effectiveness of our proposed methods compared with state-of-the-art approaches. Furthermore, the results of our study reveal that modeling competition and conflict of the multi-agent system is an effective way to improve RL performance in moment retrieval and show the new role of evidential learning in the multi-agent framework. 

---
# Balancing Interpretability and Performance in Motor Imagery EEG Classification: A Comparative Study of ANFIS-FBCSP-PSO and EEGNet 

**Authors**: Farjana Aktar, Mohd Ruhul Ameen, Akif Islam, Md Ekramul Hamid  

**Link**: [PDF](https://arxiv.org/pdf/2511.00369)  

**Abstract**: Achieving both accurate and interpretable classification of motor imagery EEG remains a key challenge in brain computer interface (BCI) research. This paper compares a transparent fuzzy reasoning approach (ANFIS-FBCSP-PSO) with a deep learning benchmark (EEGNet) using the BCI Competition IV-2a dataset. The ANFIS pipeline combines filter bank common spatial pattern feature extraction with fuzzy IF-THEN rules optimized via particle swarm optimization, while EEGNet learns hierarchical spatial temporal representations directly from raw EEG data. In within-subject experiments, the fuzzy neural model performed better (68.58 percent +/- 13.76 percent accuracy, kappa = 58.04 percent +/- 18.43), while in cross-subject (LOSO) tests, the deep model exhibited stronger generalization (68.20 percent +/- 12.13 percent accuracy, kappa = 57.33 percent +/- 16.22). The study provides practical guidance for selecting MI-BCI systems according to design goals: interpretability or robustness across users. Future investigations into transformer based and hybrid neuro symbolic frameworks are expected to advance transparent EEG decoding. 

---
# Oitijjo-3D: Generative AI Framework for Rapid 3D Heritage Reconstruction from Street View Imagery 

**Authors**: Momen Khandoker Ope, Akif Islam, Mohd Ruhul Ameen, Abu Saleh Musa Miah, Md Rashedul Islam, Jungpil Shin  

**Link**: [PDF](https://arxiv.org/pdf/2511.00362)  

**Abstract**: Cultural heritage restoration in Bangladesh faces a dual challenge of limited resources and scarce technical expertise. Traditional 3D digitization methods, such as photogrammetry or LiDAR scanning, require expensive hardware, expert operators, and extensive on-site access, which are often infeasible in developing contexts. As a result, many of Bangladesh's architectural treasures, from the Paharpur Buddhist Monastery to Ahsan Manzil, remain vulnerable to decay and inaccessible in digital form. This paper introduces Oitijjo-3D, a cost-free generative AI framework that democratizes 3D cultural preservation. By using publicly available Google Street View imagery, Oitijjo-3D reconstructs faithful 3D models of heritage structures through a two-stage pipeline - multimodal visual reasoning with Gemini 2.5 Flash Image for structure-texture synthesis, and neural image-to-3D generation through Hexagen for geometry recovery. The system produces photorealistic, metrically coherent reconstructions in seconds, achieving significant speedups compared to conventional Structure-from-Motion pipelines, without requiring any specialized hardware or expert supervision. Experiments on landmarks such as Ahsan Manzil, Choto Sona Mosque, and Paharpur demonstrate that Oitijjo-3D preserves both visual and structural fidelity while drastically lowering economic and technical barriers. By turning open imagery into digital heritage, this work reframes preservation as a community-driven, AI-assisted act of cultural continuity for resource-limited nations. 

---
# MalDataGen: A Modular Framework for Synthetic Tabular Data Generation in Malware Detection 

**Authors**: Kayua Oleques Paim, Angelo Gaspar Diniz Nogueira, Diego Kreutz, Weverton Cordeiro, Rodrigo Brandao Mansilha  

**Link**: [PDF](https://arxiv.org/pdf/2511.00361)  

**Abstract**: High-quality data scarcity hinders malware detection, limiting ML performance. We introduce MalDataGen, an open-source modular framework for generating high-fidelity synthetic tabular data using modular deep learning models (e.g., WGAN-GP, VQ-VAE). Evaluated via dual validation (TR-TS/TS-TR), seven classifiers, and utility metrics, MalDataGen outperforms benchmarks like SDV while preserving data utility. Its flexible design enables seamless integration into detection pipelines, offering a practical solution for cybersecurity applications. 

---
# Mind the Gap: Missing Cyber Threat Coverage in NIDS Datasets for the Energy Sector 

**Authors**: Adrita Rahman Tory, Khondokar Fida Hasan, Md Saifur Rahman, Nickolaos Koroniotis, Mohammad Ali Moni  

**Link**: [PDF](https://arxiv.org/pdf/2511.00360)  

**Abstract**: Network Intrusion Detection Systems (NIDS) developed us- ing publicly available datasets predominantly focus on enterprise environ- ments, raising concerns about their effectiveness for converged Informa- tion Technology (IT) and Operational Technology (OT) in energy infras- tructures. This study evaluates the representativeness of five widely used datasets: CIC-IDS2017, SWaT, WADI, Sherlock, and CIC-Modbus2023 against network-detectable MITRE ATT&CK techniques extracted from documented energy sector incidents. Using a structured five-step analyt- ical approach, this article successfully developed and performed a gap analysis that identified 94 network observable techniques from an initial pool of 274 ATT&CK techniques. Sherlock dataset exhibited the high- est mean coverage (0.56), followed closely by CIC-IDS2017 (0.55), while SWaT and WADI recorded the lowest scores (0.38). Combining CIC- IDS2017, Sherlock, and CIC-Modbus2023 achieved an aggregate coverage of 92%, highlighting their complementary strengths. The analysis identi- fies critical gaps, particularly in lateral movement and industrial protocol manipulation, providing a clear pathway for dataset enhancement and more robust NIDS evaluation in hybrid IT/OT energy environments. 

---
# Toward Unifying Group Fairness Evaluation from a Sparsity Perspective 

**Authors**: Zhecheng Sheng, Jiawei Zhang, Enmao Diao  

**Link**: [PDF](https://arxiv.org/pdf/2511.00359)  

**Abstract**: Ensuring algorithmic fairness remains a significant challenge in machine learning, particularly as models are increasingly applied across diverse domains. While numerous fairness criteria exist, they often lack generalizability across different machine learning problems. This paper examines the connections and differences among various sparsity measures in promoting fairness and proposes a unified sparsity-based framework for evaluating algorithmic fairness. The framework aligns with existing fairness criteria and demonstrates broad applicability to a wide range of machine learning tasks. We demonstrate the effectiveness of the proposed framework as an evaluation metric through extensive experiments on a variety of datasets and bias mitigation methods. This work provides a novel perspective to algorithmic fairness by framing it through the lens of sparsity and social equity, offering potential for broader impact on fairness research and applications. 

---
# Detecting AI-Generated Images via Diffusion Snap-Back Reconstruction: A Forensic Approach 

**Authors**: Mohd Ruhul Ameen, Akif Islam  

**Link**: [PDF](https://arxiv.org/pdf/2511.00352)  

**Abstract**: The rapid rise of generative diffusion models has made distinguishing authentic visual content from synthetic imagery increasingly challenging. Traditional deepfake detection methods, which rely on frequency or pixel-level artifacts, fail against modern text-to-image systems such as Stable Diffusion and DALL-E that produce photorealistic and artifact-free results. This paper introduces a diffusion-based forensic framework that leverages multi-strength image reconstruction dynamics, termed diffusion snap-back, to identify AI-generated images. By analysing how reconstruction metrics (LPIPS, SSIM, and PSNR) evolve across varying noise strengths, we extract interpretable manifold-based features that differentiate real and synthetic images. Evaluated on a balanced dataset of 4,000 images, our approach achieves 0.993 AUROC under cross-validation and remains robust to common distortions such as compression and noise. Despite using limited data and a single diffusion backbone (Stable Diffusion v1.5), the proposed method demonstrates strong generalization and interpretability, offering a foundation for scalable, model-agnostic synthetic media forensics. 

---
# Exploiting Latent Space Discontinuities for Building Universal LLM Jailbreaks and Data Extraction Attacks 

**Authors**: Kayua Oleques Paim, Rodrigo Brandao Mansilha, Diego Kreutz, Muriel Figueredo Franco, Weverton Cordeiro  

**Link**: [PDF](https://arxiv.org/pdf/2511.00346)  

**Abstract**: The rapid proliferation of Large Language Models (LLMs) has raised significant concerns about their security against adversarial attacks. In this work, we propose a novel approach to crafting universal jailbreaks and data extraction attacks by exploiting latent space discontinuities, an architectural vulnerability related to the sparsity of training data. Unlike previous methods, our technique generalizes across various models and interfaces, proving highly effective in seven state-of-the-art LLMs and one image generation model. Initial results indicate that when these discontinuities are exploited, they can consistently and profoundly compromise model behavior, even in the presence of layered defenses. The findings suggest that this strategy has substantial potential as a systemic attack vector. 

---
# MH-1M: A 1.34 Million-Sample Comprehensive Multi-Feature Android Malware Dataset for Machine Learning, Deep Learning, Large Language Models, and Threat Intelligence Research 

**Authors**: Hendrio Braganca, Diego Kreutz, Vanderson Rocha, Joner Assolin, and Eduardo Feitosa  

**Link**: [PDF](https://arxiv.org/pdf/2511.00342)  

**Abstract**: We present MH-1M, one of the most comprehensive and up-to-date datasets for advanced Android malware research. The dataset comprises 1,340,515 applications, encompassing a wide range of features and extensive metadata. To ensure accurate malware classification, we employ the VirusTotal API, integrating multiple detection engines for comprehensive and reliable assessment. Our GitHub, Figshare, and Harvard Dataverse repositories provide open access to the processed dataset and its extensive supplementary metadata, totaling more than 400 GB of data and including the outputs of the feature extraction pipeline as well as the corresponding VirusTotal reports. Our findings underscore the MH-1M dataset's invaluable role in understanding the evolving landscape of malware. 

---
# Towards Automated Petrography 

**Authors**: Isai Daniel Chacón, Paola Ruiz Puentes, Jillian Pearse, Pablo Arbeláez  

**Link**: [PDF](https://arxiv.org/pdf/2511.00328)  

**Abstract**: Petrography is a branch of geology that analyzes the mineralogical composition of rocks from microscopical thin section samples. It is essential for understanding rock properties across geology, archaeology, engineering, mineral exploration, and the oil industry. However, petrography is a labor-intensive task requiring experts to conduct detailed visual examinations of thin section samples through optical polarization microscopes, thus hampering scalability and highlighting the need for automated techniques. To address this challenge, we introduce the Large-scale Imaging and Thin section Optical-polarization Set (LITHOS), the largest and most diverse publicly available experimental framework for automated petrography. LITHOS includes 211,604 high-resolution RGB patches of polarized light and 105,802 expert-annotated grains across 25 mineral categories. Each annotation consists of the mineral class, spatial coordinates, and expert-defined major and minor axes represented as intersecting vector paths, capturing grain geometry and orientation. We evaluate multiple deep learning techniques for mineral classification in LITHOS and propose a dual-encoder transformer architecture that integrates both polarization modalities as a strong baseline for future reference. Our method consistently outperforms single-polarization models, demonstrating the value of polarization synergy in mineral classification. We have made the LITHOS Benchmark publicly available, comprising our dataset, code, and pretrained models, to foster reproducibility and further research in automated petrographic analysis. 

---
# Scalable Processing-Near-Memory for 1M-Token LLM Inference: CXL-Enabled KV-Cache Management Beyond GPU Limits 

**Authors**: Dowon Kim, MinJae Lee, Janghyeon Kim, HyuckSung Kwon, Hyeonggyu Jeong, Sang-Soo Park, Minyong Yoon, Si-Dong Roh, Yongsuk Kwon, Jinin So, Jungwook Choi  

**Link**: [PDF](https://arxiv.org/pdf/2511.00321)  

**Abstract**: The expansion of context windows in large language models (LLMs) to multi-million tokens introduces severe memory and compute bottlenecks, particularly in managing the growing Key-Value (KV) cache. While Compute Express Link (CXL) enables non-eviction frameworks that offload the full KV-cache to scalable external memory, these frameworks still suffer from costly data transfers when recalling non-resident KV tokens to limited GPU memory as context lengths increase. This work proposes scalable Processing-Near-Memory (PNM) for 1M-Token LLM Inference, a CXL-enabled KV-cache management system that coordinates memory and computation beyond GPU limits. Our design offloads token page selection to a PNM accelerator within CXL memory, eliminating costly recalls and enabling larger GPU batch sizes. We further introduce a hybrid parallelization strategy and a steady-token selection mechanism to enhance compute efficiency and scalability. Implemented atop a state-of-the-art CXL-PNM system, our solution delivers consistent performance gains for LLMs with up to 405B parameters and 1M-token contexts. Our PNM-only offloading scheme (PNM-KV) and GPU-PNM hybrid with steady-token execution (PnG-KV) achieve up to 21.9x throughput improvement, up to 60x lower energy per token, and up to 7.3x better total cost efficiency than the baseline, demonstrating that CXL-enabled multi-PNM architectures can serve as a scalable backbone for future long-context LLM inference. 

---
# A Technical Exploration of Causal Inference with Hybrid LLM Synthetic Data 

**Authors**: Dana Kim, Yichen Xu, Tiffany Lin  

**Link**: [PDF](https://arxiv.org/pdf/2511.00318)  

**Abstract**: Large Language Models (LLMs) offer a flexible means to generate synthetic tabular data, yet existing approaches often fail to preserve key causal parameters such as the average treatment effect (ATE). In this technical exploration, we first demonstrate that state-of-the-art synthetic data generators, both GAN- and LLM-based, can achieve high predictive fidelity while substantially misestimating causal effects. To address this gap, we propose a hybrid generation framework that combines model-based covariate synthesis (monitored via distance-to-closest-record filtering) with separately learned propensity and outcome models, thereby ensuring that (W, A, Y) triplets retain their underlying causal structure. We further introduce a synthetic pairing strategy to mitigate positivity violations and a realistic evaluation protocol that leverages unlimited synthetic samples to benchmark traditional estimators (IPTW, AIPW, substitution) under complex covariate distributions. This work lays the groundwork for LLM-powered data pipelines that support robust causal analysis. Our code is available at this https URL. 

---
# Language Modeling With Factorization Memory 

**Authors**: Lee Xiong, Maksim Tkachenko, Johanes Effendi, Ting Cai  

**Link**: [PDF](https://arxiv.org/pdf/2511.00315)  

**Abstract**: We propose Factorization Memory, an efficient recurrent neural network (RNN) architecture that achieves performance comparable to Transformer models on short-context language modeling tasks while also demonstrating superior generalization in long-context scenarios. Our model builds upon Mamba-2, enabling Factorization Memory to exploit parallel computations during training while preserving constant computational and memory complexity during inference. To further optimize model efficiency and representational capacity, we develop a sparse formulation of Factorization Memory that updates only a subset of recurrent states at each step while preserving the strong performance of its dense counterpart. To our knowledge, this represents the first RNN architecture that successfully combines sparse memory activation with competitive performance across both short and long-context settings. This work provides a systematic empirical analysis of Factorization Memory in comparison to Transformer and Mamba-2 architectures. 

---
# Calibration Across Layers: Understanding Calibration Evolution in LLMs 

**Authors**: Abhinav Joshi, Areeb Ahmad, Ashutosh Modi  

**Link**: [PDF](https://arxiv.org/pdf/2511.00280)  

**Abstract**: Large Language Models (LLMs) have demonstrated inherent calibration capabilities, where predicted probabilities align well with correctness, despite prior findings that deep neural networks are often overconfident. Recent studies have linked this behavior to specific components in the final layer, such as entropy neurons and the unembedding matrix null space. In this work, we provide a complementary perspective by investigating how calibration evolves throughout the network depth. Analyzing multiple open-weight models on the MMLU benchmark, we uncover a distinct confidence correction phase in the upper/later layers, where model confidence is actively recalibrated after decision certainty has been reached. Furthermore, we identify a low-dimensional calibration direction in the residual stream whose perturbation significantly improves calibration metrics (ECE and MCE) without harming accuracy. Our findings suggest that calibration is a distributed phenomenon, shaped throughout the network forward pass, not just in its final projection, providing new insights into how confidence-regulating mechanisms operate within LLMs. 

---
# LongCat-Flash-Omni Technical Report 

**Authors**: Meituan LongCat Team, Bairui Wang, Bayan, Bin Xiao, Bo Zhang, Bolin Rong, Borun Chen, Chang Wan, Chao Zhang, Chen Huang, Chen Chen, Chen Chen, Chengxu Yang, Chengzuo Yang, Cong Han, Dandan Peng, Delian Ruan, Detai Xin, Disong Wang, Dongchao Yang, Fanfan Liu, Fengjiao Chen, Fengyu Yang, Gan Dong, Gang Huang, Gang Xu, Guanglu Wan, Guoqiang Tan, Guoqiao Yu, Haibo Qiu, Hao Lu, Hongbo Liu, Hongyu Xiang, Jiaheng Wu, Jian Yang, Jiaxing Liu, Jing Huang, Jingang Wang, Jinrui Ding, Juchao Jiang, Jun Kuang, Jun Wang, Junhui Mei, Ke Ding, Kefeng Zhang, Lei Chen, Liang Shi, Limeng Qiao, Liming Zheng, Lin Ma, Liuyang Guo, Liya Ma, Luying Sun, Man Gao, Mengshen Zhu, Miao Cao, Minliang Lin, Nuo Xu, Peng Shi, Qi Zhang, Qian Fang, Qian Wang, Qian Yang, Quanxiu Wang, Rongxiang Weng, Rongxin Guo, Ruoxuan Liang, Senbin Yang, Shanbo Xu, Shanglin Lei, Shengze Ye, Shimin Chen, Shuaiqi Chen, Shujie Hu, Shuo Li, Siqi Yang, Siyu Xu, Siyu Ren, Song Li, Songxiang Liu, Tianhao Bai, Tianye Dai, Wei Hong, Wei Wang, Weixiao Zhao, Wengang Cao, Wenlong Zhu, Wenlong He, Xi Su, Xi Nan, Xiaohan Zhao, Xiaohao Wang, Xiaoyu Zhao, Xiaoyu Wang, Xiaoyu Li, Xin Pan, Xin Chen, Xiusong Sun, Xu Xiang, Xudong Xing  

**Link**: [PDF](https://arxiv.org/pdf/2511.00279)  

**Abstract**: We introduce LongCat-Flash-Omni, a state-of-the-art open-source omni-modal model with 560 billion parameters, excelling at real-time audio-visual interaction. By adopting a curriculum-inspired progressive training strategy that transitions from simpler to increasingly complex modality sequence modeling tasks, LongCat-Flash-Omni attains comprehensive multimodal capabilities while maintaining strong unimodal capability. Building upon LongCat-Flash, which adopts a high-performance Shortcut-connected Mixture-of-Experts (MoE) architecture with zero-computation experts, LongCat-Flash-Omni integrates efficient multimodal perception and speech reconstruction modules. Despite its immense size of 560B parameters (with 27B activated), LongCat-Flash-Omni achieves low-latency real-time audio-visual interaction. For training infrastructure, we developed a modality-decoupled parallelism scheme specifically designed to manage the data and model heterogeneity inherent in large-scale multimodal training. This innovative approach demonstrates exceptional efficiency by sustaining over 90% of the throughput achieved by text-only training. Extensive evaluations show that LongCat-Flash-Omni achieves state-of-the-art performance on omni-modal benchmarks among open-source models. Furthermore, it delivers highly competitive results across a wide range of modality-specific tasks, including text, image, and video understanding, as well as audio understanding and generation. We provide a comprehensive overview of the model architecture design, training procedures, and data strategies, and open-source the model to foster future research and development in the community. 

---
# POSESTITCH-SLT: Linguistically Inspired Pose-Stitching for End-to-End Sign Language Translation 

**Authors**: Abhinav Joshi, Vaibhav Sharma, Sanjeet Singh, Ashutosh Modi  

**Link**: [PDF](https://arxiv.org/pdf/2511.00270)  

**Abstract**: Sign language translation remains a challenging task due to the scarcity of large-scale, sentence-aligned datasets. Prior arts have focused on various feature extraction and architectural changes to support neural machine translation for sign languages. We propose POSESTITCH-SLT, a novel pre-training scheme that is inspired by linguistic-templates-based sentence generation technique. With translation comparison on two sign language datasets, How2Sign and iSign, we show that a simple transformer-based encoder-decoder architecture outperforms the prior art when considering template-generated sentence pairs in training. We achieve BLEU-4 score improvements from 1.97 to 4.56 on How2Sign and from 0.55 to 3.43 on iSign, surpassing prior state-of-the-art methods for pose-based gloss-free translation. The results demonstrate the effectiveness of template-driven synthetic supervision in low-resource sign language settings. 

---
# FedReplay: A Feature Replay Assisted Federated Transfer Learning Framework for Efficient and Privacy-Preserving Smart Agriculture 

**Authors**: Long Li, Jiajia Li, Dong Chen, Lina Pu, Haibo Yao, Yanbo Huang  

**Link**: [PDF](https://arxiv.org/pdf/2511.00269)  

**Abstract**: Accurate classification plays a pivotal role in smart agriculture, enabling applications such as crop monitoring, fruit recognition, and pest detection. However, conventional centralized training often requires large-scale data collection, which raises privacy concerns, while standard federated learning struggles with non-independent and identically distributed (non-IID) data and incurs high communication costs. To address these challenges, we propose a federated learning framework that integrates a frozen Contrastive Language-Image Pre-training (CLIP) vision transformer (ViT) with a lightweight transformer classifier. By leveraging the strong feature extraction capability of the pre-trained CLIP ViT, the framework avoids training large-scale models from scratch and restricts federated updates to a compact classifier, thereby reducing transmission overhead significantly. Furthermore, to mitigate performance degradation caused by non-IID data distribution, a small subset (1%) of CLIP-extracted feature representations from all classes is shared across clients. These shared features are non-reversible to raw images, ensuring privacy preservation while aligning class representation across participants. Experimental results on agricultural classification tasks show that the proposed method achieve 86.6% accuracy, which is more than 4 times higher compared to baseline federated learning approaches. This demonstrates the effectiveness and efficiency of combining vision-language model features with federated learning for privacy-preserving and scalable agricultural intelligence. 

---
# IL-PCSR: Legal Corpus for Prior Case and Statute Retrieval 

**Authors**: Shounak Paul, Dhananjay Ghumare, Pawan Goyal, Saptarshi Ghosh, Ashutosh Modi  

**Link**: [PDF](https://arxiv.org/pdf/2511.00268)  

**Abstract**: Identifying/retrieving relevant statutes and prior cases/precedents for a given legal situation are common tasks exercised by law practitioners. Researchers to date have addressed the two tasks independently, thus developing completely different datasets and models for each task; however, both retrieval tasks are inherently related, e.g., similar cases tend to cite similar statutes (due to similar factual situation). In this paper, we address this gap. We propose IL-PCR (Indian Legal corpus for Prior Case and Statute Retrieval), which is a unique corpus that provides a common testbed for developing models for both the tasks (Statute Retrieval and Precedent Retrieval) that can exploit the dependence between the two. We experiment extensively with several baseline models on the tasks, including lexical models, semantic models and ensemble based on GNNs. Further, to exploit the dependence between the two tasks, we develop an LLM-based re-ranking approach that gives the best performance. 

---
# Neural Transparency: Mechanistic Interpretability Interfaces for Anticipating Model Behaviors for Personalized AI 

**Authors**: Sheer Karny, Anthony Baez, Pat Pataranutaporn  

**Link**: [PDF](https://arxiv.org/pdf/2511.00230)  

**Abstract**: Millions of users now design personalized LLM-based chatbots that shape their daily interactions, yet they can only loosely anticipate how their design choices will manifest as behaviors in deployment. This opacity is consequential: seemingly innocuous prompts can trigger excessive sycophancy, toxicity, or inconsistency, degrading utility and raising safety concerns. To address this issue, we introduce an interface that enables neural transparency by exposing language model internals during chatbot design. Our approach extracts behavioral trait vectors (empathy, toxicity, sycophancy, etc.) by computing differences in neural activations between contrastive system prompts that elicit opposing behaviors. We predict chatbot behaviors by projecting the system prompt's final token activations onto these trait vectors, normalizing for cross-trait comparability, and visualizing results via an interactive sunburst diagram. To evaluate this approach, we conducted an online user study using Prolific to compare our neural transparency interface against a baseline chatbot interface without any form of transparency. Our analyses suggest that users systematically miscalibrated AI behavior: participants misjudged trait activations for eleven of fifteen analyzable traits, motivating the need for transparency tools in everyday human-AI interaction. While our interface did not change design iteration patterns, it significantly increased user trust and was enthusiastically received. Qualitative analysis indicated that users' had nuanced experiences with the visualization that may enrich future work designing neurally transparent interfaces. This work offers a path for how mechanistic interpretability can be operationalized for non-technical users, establishing a foundation for safer, more aligned human-AI interactions. 

---
# Consistently Simulating Human Personas with Multi-Turn Reinforcement Learning 

**Authors**: Marwa Abdulhai, Ryan Cheng, Donovan Clay, Tim Althoff, Sergey Levine, Natasha Jaques  

**Link**: [PDF](https://arxiv.org/pdf/2511.00222)  

**Abstract**: Large Language Models (LLMs) are increasingly used to simulate human users in interactive settings such as therapy, education, and social role-play. While these simulations enable scalable training and evaluation of AI agents, off-the-shelf LLMs often drift from their assigned personas, contradict earlier statements, or abandon role-appropriate behavior. We introduce a unified framework for evaluating and improving persona consistency in LLM-generated dialogue. We define three automatic metrics: prompt-to-line consistency, line-to-line consistency, and Q&A consistency, that capture different types of persona drift and validate each against human annotations. Using these metrics as reward signals, we apply multi-turn reinforcement learning to fine-tune LLMs for three user roles: a patient, a student, and a social chat partner. Our method reduces inconsistency by over 55%, resulting in more coherent and faithful simulated users. 

---
# DM-QPMNET: Dual-modality fusion network for cell segmentation in quantitative phase microscopy 

**Authors**: Rajatsubhra Chakraborty, Ana Espinosa-Momox, Riley Haskin, Depeng Xu, Rosario Porras-Aguilar  

**Link**: [PDF](https://arxiv.org/pdf/2511.00218)  

**Abstract**: Cell segmentation in single-shot quantitative phase microscopy (ssQPM) faces challenges from traditional thresholding methods that are sensitive to noise and cell density, while deep learning approaches using simple channel concatenation fail to exploit the complementary nature of polarized intensity images and phase maps. We introduce DM-QPMNet, a dual-encoder network that treats these as distinct modalities with separate encoding streams. Our architecture fuses modality-specific features at intermediate depth via multi-head attention, enabling polarized edge and texture representations to selectively integrate complementary phase information. This content-aware fusion preserves training stability while adding principled multi-modal integration through dual-source skip connections and per-modality normalization at minimal overhead. Our approach demonstrates substantial improvements over monolithic concatenation and single-modality baselines, showing that modality-specific encoding with learnable fusion effectively exploits ssQPM's simultaneous capture of complementary illumination and phase cues for robust cell segmentation. 

---
# An Efficient and Generalizable Transfer Learning Method for Weather Condition Detection on Ground Terminals 

**Authors**: Wenxuan Zhang, Peng Hu  

**Link**: [PDF](https://arxiv.org/pdf/2511.00211)  

**Abstract**: The increasing adoption of satellite Internet with low-Earth-orbit (LEO) satellites in mega-constellations allows ubiquitous connectivity to rural and remote areas. However, weather events have a significant impact on the performance and reliability of satellite Internet. Adverse weather events such as snow and rain can disturb the performance and operations of satellite Internet's essential ground terminal components, such as satellite antennas, significantly disrupting the space-ground link conditions between LEO satellites and ground stations. This challenge calls for not only region-based weather forecasts but also fine-grained detection capability on ground terminal components of fine-grained weather conditions. Such a capability can assist in fault diagnostics and mitigation for reliable satellite Internet, but its solutions are lacking, not to mention the effectiveness and generalization that are essential in real-world deployments. This paper discusses an efficient transfer learning (TL) method that can enable a ground component to locally detect representative weather-related conditions. The proposed method can detect snow, wet, and other conditions resulting from adverse and typical weather events and shows superior performance compared to the typical deep learning methods, such as YOLOv7, YOLOv9, Faster R-CNN, and R-YOLO. Our TL method also shows the advantage of being generalizable to various scenarios. 

---
# Diffusion Models at the Drug Discovery Frontier: A Review on Generating Small Molecules versus Therapeutic Peptides 

**Authors**: Yiquan Wang, Yahui Ma, Yuhan Chang, Jiayao Yan, Jialin Zhang, Minnuo Cai, Kai Wei  

**Link**: [PDF](https://arxiv.org/pdf/2511.00209)  

**Abstract**: Diffusion models have emerged as a leading framework in generative modeling, showing significant potential to accelerate and transform the traditionally slow and costly process of drug discovery. This review provides a systematic comparison of their application in designing two principal therapeutic modalities: small molecules and therapeutic peptides. We analyze how a unified framework of iterative denoising is adapted to the distinct molecular representations, chemical spaces, and design objectives of each modality. For small molecules, these models excel at structure-based design, generating novel, pocket-fitting ligands with desired physicochemical properties, yet face the critical hurdle of ensuring chemical synthesizability. Conversely, for therapeutic peptides, the focus shifts to generating functional sequences and designing de novo structures, where the primary challenges are achieving biological stability against proteolysis, ensuring proper folding, and minimizing immunogenicity. Despite these distinct challenges, both domains face shared hurdles: the need for more accurate scoring functions, the scarcity of high-quality experimental data, and the crucial requirement for experimental validation. We conclude that the full potential of diffusion models will be unlocked by bridging these modality-specific gaps and integrating them into automated, closed-loop Design-Build-Test-Learn (DBTL) platforms, thereby shifting the paradigm from chemical exploration to the targeted creation of novel therapeutics. 

---
# Training LLMs Beyond Next Token Prediction - Filling the Mutual Information Gap 

**Authors**: Chun-Hao Yang, Bo-Han Feng, Tzu-Yuan Lai, Yan Yu Chen, Yin-Kai Dean Huang, Shou-De Lin  

**Link**: [PDF](https://arxiv.org/pdf/2511.00198)  

**Abstract**: Optimizing training performance in large language models (LLMs) remains an essential challenge, particularly in improving model performance while maintaining computational costs. This work challenges the conventional approach of training LLMs using next-token prediction (NTP), arguing that by predicting information-rich tokens during training, there is a more effective way to train LLMs. We investigate the impact of the proposed solution in three kinds of tasks for LLMs: arithmetic, multi-label classification of text, and natural-language generation. This work offers a principled approach to optimizing LLM training, advancing both model performance and theoretical understanding of the target-token selection strategies. 

---
# Understanding Code Agent Behaviour: An Empirical Study of Success and Failure Trajectories 

**Authors**: Oorja Majgaonkar, Zhiwei Fei, Xiang Li, Federica Sarro, He Ye  

**Link**: [PDF](https://arxiv.org/pdf/2511.00197)  

**Abstract**: The increasing deployment of Large Language Model (LLM) agents for complex software engineering tasks has created a need to understand their problem-solving behaviours beyond simple success metrics. While these agents demonstrate impressive capabilities in automated issue resolution, their decision-making processes remain largely opaque. This paper presents an empirical study of agent trajectories, namely the execution traces capturing the steps agents take when attempting to resolve software issues. We analyse trajectories from three state-of-the-art code agents (OpenHands, SWE-agent, and Prometheus) on the SWE-Bench benchmark, examining both successful and failed attempts. Our investigation reveals several key insights into agent behaviour. First, we identify how distinct problem-solving strategies, such as defensive programming and context gathering, enable success in different scenarios. Second, we find that failed trajectories are consistently longer and exhibit higher variance than successful ones, with failure patterns differing significantly between agents. Third, our fault localisation analysis shows that while most trajectories correctly identify problematic files (72-81\% even in failures), success depends more on achieving approximate rather than exact code modifications. These and other findings unveiled by our study, provide a foundation for understanding agent behaviour through trajectory analysis, contributing to the development of more robust and interpretable autonomous software engineering systems. 

---
# EL-MIA: Quantifying Membership Inference Risks of Sensitive Entities in LLMs 

**Authors**: Ali Satvaty, Suzan Verberne, Fatih Turkmen  

**Link**: [PDF](https://arxiv.org/pdf/2511.00192)  

**Abstract**: Membership inference attacks (MIA) aim to infer whether a particular data point is part of the training dataset of a model. In this paper, we propose a new task in the context of LLM privacy: entity-level discovery of membership risk focused on sensitive information (PII, credit card numbers, etc). Existing methods for MIA can detect the presence of entire prompts or documents in the LLM training data, but they fail to capture risks at a finer granularity. We propose the ``EL-MIA'' framework for auditing entity-level membership risks in LLMs. We construct a benchmark dataset for the evaluation of MIA methods on this task. Using this benchmark, we conduct a systematic comparison of existing MIA techniques as well as two newly proposed methods. We provide a comprehensive analysis of the results, trying to explain the relation of the entity level MIA susceptability with the model scale, training epochs, and other surface level factors. Our findings reveal that existing MIA methods are limited when it comes to entity-level membership inference of the sensitive attributes, while this susceptibility can be outlined with relatively straightforward methods, highlighting the need for stronger adversaries to stress test the provided threat model. 

---
# A Retrospect to Multi-prompt Learning across Vision and Language 

**Authors**: Ziliang Chen, Xin Huang, Quanlong Guan, Liang Lin, Weiqi Luo  

**Link**: [PDF](https://arxiv.org/pdf/2511.00191)  

**Abstract**: The vision community is undergoing the unprecedented progress with the emergence of Vision-Language Pretraining Models (VLMs). Prompt learning plays as the holy grail of accessing VLMs since it enables their fast adaptation to downstream tasks with limited resources. Whereas existing researches milling around single-prompt paradigms, rarely investigate the technical potential behind their multi-prompt learning counterparts. This paper aims to provide a principled retrospect for vision-language multi-prompt learning. We extend the recent constant modality gap phenomenon to learnable prompts and then, justify the superiority of vision-language transfer with multi-prompt augmentation, empirically and theoretically. In terms of this observation, we propose an Energy-based Multi-prompt Learning (EMPL) to generate multiple prompt embeddings by drawing instances from an energy-based distribution, which is implicitly defined by VLMs. So our EMPL is not only parameter-efficient but also rigorously lead to the balance between in-domain and out-of-domain open-vocabulary generalization. Comprehensive experiments have been conducted to justify our claims and the excellence of EMPL. 

---
# Generative Modeling Enables Molecular Structure Retrieval from Coulomb Explosion Imaging 

**Authors**: Xiang Li, Till Jahnke, Rebecca Boll, Jiaqi Han, Minkai Xu, Michael Meyer, Maria Novella Piancastelli, Daniel Rolles, Artem Rudenko, Florian Trinter, Thomas J.A. Wolf, Jana B. Thayer, James P. Cryan, Stefano Ermon, Phay J. Ho  

**Link**: [PDF](https://arxiv.org/pdf/2511.00179)  

**Abstract**: Capturing the structural changes that molecules undergo during chemical reactions in real space and time is a long-standing dream and an essential prerequisite for understanding and ultimately controlling femtochemistry. A key approach to tackle this challenging task is Coulomb explosion imaging, which benefited decisively from recently emerging high-repetition-rate X-ray free-electron laser sources. With this technique, information on the molecular structure is inferred from the momentum distributions of the ions produced by the rapid Coulomb explosion of molecules. Retrieving molecular structures from these distributions poses a highly non-linear inverse problem that remains unsolved for molecules consisting of more than a few atoms. Here, we address this challenge using a diffusion-based Transformer neural network. We show that the network reconstructs unknown molecular geometries from ion-momentum distributions with a mean absolute error below one Bohr radius, which is half the length of a typical chemical bond. 

---
# Effectiveness of LLMs in Temporal User Profiling for Recommendation 

**Authors**: Milad Sabouri, Masoud Mansoury, Kun Lin, Bamshad Mobasher  

**Link**: [PDF](https://arxiv.org/pdf/2511.00176)  

**Abstract**: Effectively modeling the dynamic nature of user preferences is crucial for enhancing recommendation accuracy and fostering transparency in recommender systems. Traditional user profiling often overlooks the distinction between transitory short-term interests and stable long-term preferences. This paper examines the capability of leveraging Large Language Models (LLMs) to capture these temporal dynamics, generating richer user representations through distinct short-term and long-term textual summaries of interaction histories. Our observations suggest that while LLMs tend to improve recommendation quality in domains with more active user engagement, their benefits appear less pronounced in sparser environments. This disparity likely stems from the varying distinguishability of short-term and long-term preferences across domains; the approach shows greater utility where these temporal interests are more clearly separable (e.g., Movies\&TV) compared to domains with more stable user profiles (e.g., Video Games). This highlights a critical trade-off between enhanced performance and computational costs, suggesting context-dependent LLM application. Beyond predictive capability, this LLM-driven approach inherently provides an intrinsic potential for interpretability through its natural language profiles and attention weights. This work contributes insights into the practical capability and inherent interpretability of LLM-driven temporal user profiling, outlining new research directions for developing adaptive and transparent recommender systems. 

---
# What a diff makes: automating code migration with large language models 

**Authors**: Katherine A. Rosenfeld, Cliff C. Kerr, Jessica Lundin  

**Link**: [PDF](https://arxiv.org/pdf/2511.00160)  

**Abstract**: Modern software programs are built on stacks that are often undergoing changes that introduce updates and improvements, but may also break any project that depends upon them. In this paper we explore the use of Large Language Models (LLMs) for code migration, specifically the problem of maintaining compatibility with a dependency as it undergoes major and minor semantic version changes. We demonstrate, using metrics such as test coverage and change comparisons, that contexts containing diffs can significantly improve performance against out of the box LLMs and, in some cases, perform better than using code. We provide a dataset to assist in further development of this problem area, as well as an open-source Python package, AIMigrate, that can be used to assist with migrating code bases. In a real-world migration of TYPHOIDSIM between STARSIM versions, AIMigrate correctly identified 65% of required changes in a single run, increasing to 80% with multiple runs, with 47% of changes generated perfectly. 

---
# FLoC: Facility Location-Based Efficient Visual Token Compression for Long Video Understanding 

**Authors**: Janghoon Cho, Jungsoo Lee, Munawar Hayat, Kyuwoong Hwang, Fatih Porikli, Sungha Choi  

**Link**: [PDF](https://arxiv.org/pdf/2511.00141)  

**Abstract**: Recent studies in long video understanding have harnessed the advanced visual-language reasoning capabilities of Large Multimodal Models (LMMs), driving the evolution of video-LMMs specialized for processing extended video sequences. However, the scalability of these models is severely limited by the overwhelming volume of visual tokens generated from extended video sequences. To address this challenge, this paper proposes FLoC, an efficient visual token compression framework based on the facility location function, a principled approach that swiftly selects a compact yet highly representative and diverse subset of visual tokens within a predefined budget on the number of visual tokens. By integrating the lazy greedy algorithm, our method achieves remarkable efficiency gains by swiftly selecting a compact subset of tokens, drastically reducing the number of visual tokens while guaranteeing near-optimal performance. Notably, our approach is training-free, model-agnostic, and query-agnostic, providing a versatile solution that seamlessly integrates with diverse video-LLMs and existing workflows. Extensive evaluations on large-scale benchmarks, such as Video-MME, MLVU, and LongVideoBench, demonstrate that our framework consistently surpasses recent compression techniques, highlighting not only its effectiveness and robustness in addressing the critical challenges of long video understanding, but also its efficiency in processing speed. 

---
# End-to-End Dexterous Arm-Hand VLA Policies via Shared Autonomy: VR Teleoperation Augmented by Autonomous Hand VLA Policy for Efficient Data Collection 

**Authors**: Yu Cui, Yujian Zhang, Lina Tao, Yang Li, Xinyu Yi, Zhibin Li  

**Link**: [PDF](https://arxiv.org/pdf/2511.00139)  

**Abstract**: Achieving human-like dexterous manipulation remains a major challenge for general-purpose robots. While Vision-Language-Action (VLA) models show potential in learning skills from demonstrations, their scalability is limited by scarce high-quality training data. Existing data collection methods face inherent constraints: manual teleoperation overloads human operators, while automated planning often produces unnatural motions. We propose a Shared Autonomy framework that divides control between macro and micro motions. A human operator guides the robot's arm pose through intuitive VR teleoperation, while an autonomous DexGrasp-VLA policy handles fine-grained hand control using real-time tactile and visual feedback. This division significantly reduces cognitive load and enables efficient collection of high-quality coordinated arm-hand demonstrations. Using this data, we train an end-to-end VLA policy enhanced with our novel Arm-Hand Feature Enhancement module, which captures both distinct and shared representations of macro and micro movements for more natural coordination. Our Corrective Teleoperation system enables continuous policy improvement through human-in-the-loop failure recovery. Experiments demonstrate that our framework generates high-quality data with minimal manpower and achieves a 90% success rate across diverse objects, including unseen instances. Comprehensive evaluations validate the system's effectiveness in developing dexterous manipulation capabilities. 

---
# A Dual Large Language Models Architecture with Herald Guided Prompts for Parallel Fine Grained Traffic Signal Control 

**Authors**: Qing Guo, Xinhang Li, Junyu Chen, Zheng Guo, Xiaocong Li, Lin Zhang, Lei Li  

**Link**: [PDF](https://arxiv.org/pdf/2511.00136)  

**Abstract**: Leveraging large language models (LLMs) in traffic signal control (TSC) improves optimization efficiency and interpretability compared to traditional reinforcement learning (RL) methods. However, existing LLM-based approaches are limited by fixed time signal durations and are prone to hallucination errors, while RL methods lack robustness in signal timing decisions and suffer from poor generalization. To address these challenges, this paper proposes HeraldLight, a dual LLMs architecture enhanced by Herald guided prompts. The Herald Module extracts contextual information and forecasts queue lengths for each traffic phase based on real-time conditions. The first LLM, LLM-Agent, uses these forecasts to make fine grained traffic signal control, while the second LLM, LLM-Critic, refines LLM-Agent's outputs, correcting errors and hallucinations. These refined outputs are used for score-based fine-tuning to improve accuracy and robustness. Simulation experiments using CityFlow on real world datasets covering 224 intersections in Jinan (12), Hangzhou (16), and New York (196) demonstrate that HeraldLight outperforms state of the art baselines, achieving a 20.03% reduction in average travel time across all scenarios and a 10.74% reduction in average queue length on the Jinan and Hangzhou scenarios. The source code is available on GitHub: this https URL. 

---
# Feature Importance Guided Random Forest Learning with Simulated Annealing Based Hyperparameter Tuning 

**Authors**: Kowshik Balasubramanian, Andre Williams, Ismail Butun  

**Link**: [PDF](https://arxiv.org/pdf/2511.00133)  

**Abstract**: This paper introduces a novel framework for enhancing Random Forest classifiers by integrating probabilistic feature sampling and hyperparameter tuning via Simulated Annealing. The proposed framework exhibits substantial advancements in predictive accuracy and generalization, adeptly tackling the multifaceted challenges of robust classification across diverse domains, including credit risk evaluation, anomaly detection in IoT ecosystems, early-stage medical diagnostics, and high-dimensional biological data analysis. To overcome the limitations of conventional Random Forests, we present an approach that places stronger emphasis on capturing the most relevant signals from data while enabling adaptive hyperparameter configuration. The model is guided towards features that contribute more meaningfully to classification and optimizing this with dynamic parameter tuning. The results demonstrate consistent accuracy improvements and meaningful insights into feature relevance, showcasing the efficacy of combining importance aware sampling and metaheuristic optimization. 

---
# Casing Collar Identification using AlexNet-based Neural Networks for Depth Measurement in Oil and Gas Wells 

**Authors**: Siyu Xiao, Xindi Zhao, Tianhao Mao, Yiwei Wang, Yuqiao Chen, Hongyun Zhang, Jian Wang, Junjie Wang, Shuang Liu, Tupei Chen, Yang Liu  

**Link**: [PDF](https://arxiv.org/pdf/2511.00129)  

**Abstract**: Accurate downhole depth measurement is essential for oil and gas well operations, directly influencing reservoir contact, production efficiency, and operational safety. Collar correlation using a casing collar locator (CCL) is fundamental for precise depth calibration. While neural network-based CCL signal recognition has achieved significant progress in collar identification, preprocessing methods for such applications remain underdeveloped. Moreover, the limited availability of real well data poses substantial challenges for training neural network models that require extensive datasets. This paper presents a system integrated into downhole tools for CCL signal acquisition to facilitate dataset construction. We propose comprehensive preprocessing methods for data augmentation and evaluate their effectiveness using our AlexNet-based neural network models. Through systematic experimentation across various configuration combinations, we analyze the contribution of each augmentation method. Results demonstrate that standardization, label distribution smoothing (LDS), and random cropping are fundamental requirements for model training, while label smoothing regularization (LSR), time scaling, and multiple sampling significantly enhance model generalization capability. The F1 scores of our two benchmark models trained with the proposed augmentation methods maximumly improve from 0.937 and 0.952 to 1.0 and 1.0, respectively. Performance validation on real CCL waveforms confirms the effectiveness and practical applicability of our approach. This work addresses the gaps in data augmentation methodologies for training casing collar recognition models in CCL data-limited environments. 

---
# Dynamic Model Selection for Trajectory Prediction via Pairwise Ranking and Meta-Features 

**Authors**: Lu Bowen  

**Link**: [PDF](https://arxiv.org/pdf/2511.00126)  

**Abstract**: Recent deep trajectory predictors (e.g., Jiang et al., 2023; Zhou et al., 2022) have achieved strong average accuracy but remain unreliable in complex long-tail driving scenarios. These limitations reveal the weakness of the prevailing "one-model-fits-all" paradigm, particularly in safety-critical urban contexts where simpler physics-based models can occasionally outperform advanced networks (Kalman, 1960). To bridge this gap, we propose a dynamic multi-expert gating framework that adaptively selects the most reliable trajectory predictor among a physics-informed LSTM, a Transformer, and a fine-tuned GameFormer on a per-sample basis.
Our method leverages internal model signals (meta-features) such as stability and uncertainty (Gal and Ghahramani, 2016), which we demonstrate to be substantially more informative than geometric scene descriptors. To the best of our knowledge, this is the first work to formulate trajectory expert selection as a pairwise-ranking problem over internal model signals (Burges et al., 2005), directly optimizing decision quality without requiring post-hoc calibration.
Evaluated on the nuPlan-mini dataset (Caesar et al., 2021) with 1,287 samples, our LLM-enhanced tri-expert gate achieves a Final Displacement Error (FDE) of 2.567 m, representing a 9.5 percent reduction over GameFormer (2.835 m), and realizes 57.8 percent of the oracle performance bound. In open-loop simulations, after trajectory horizon alignment, the same configuration reduces FDE on left-turn scenarios by approximately 10 percent, demonstrating consistent improvements across both offline validation and open-loop evaluation. These results indicate that adaptive hybrid systems enhance trajectory reliability in safety-critical autonomous driving, providing a practical pathway beyond static single-model paradigms. 

---
# Inferring multiple helper Dafny assertions with LLMs 

**Authors**: Álvaro Silva, Alexandra Mendes, Ruben Martins  

**Link**: [PDF](https://arxiv.org/pdf/2511.00125)  

**Abstract**: The Dafny verifier provides strong correctness guarantees but often requires numerous manual helper assertions, creating a significant barrier to adoption. We investigate the use of Large Language Models (LLMs) to automatically infer missing helper assertions in Dafny programs, with a primary focus on cases involving multiple missing assertions. To support this study, we extend the DafnyBench benchmark with curated datasets where one, two, or all assertions are removed, and we introduce a taxonomy of assertion types to analyze inference difficulty. Our approach refines fault localization through a hybrid method that combines LLM predictions with error-message heuristics. We implement this approach in a new tool called DAISY (Dafny Assertion Inference SYstem). While our focus is on multiple missing assertions, we also evaluate DAISY on single-assertion cases. DAISY verifies 63.4% of programs with one missing assertion and 31.7% with multiple missing assertions. Notably, many programs can be verified with fewer assertions than originally present, highlighting that proofs often admit multiple valid repair strategies and that recovering every original assertion is unnecessary. These results demonstrate that automated assertion inference can substantially reduce proof engineering effort and represent a step toward more scalable and accessible formal verification. 

---
# Cross-fluctuation phase transitions reveal sampling dynamics in diffusion models 

**Authors**: Sai Niranjan Ramachandran, Manish Krishan Lal, Suvrit Sra  

**Link**: [PDF](https://arxiv.org/pdf/2511.00124)  

**Abstract**: We analyse how the sampling dynamics of distributions evolve in score-based diffusion models using cross-fluctuations, a centered-moment statistic from statistical physics. Specifically, we show that starting from an unbiased isotropic normal distribution, samples undergo sharp, discrete transitions, eventually forming distinct events of a desired distribution while progressively revealing finer structure. As this process is reversible, these transitions also occur in reverse, where intermediate states progressively merge, tracing a path back to the initial distribution. We demonstrate that these transitions can be detected as discontinuities in $n^{\text{th}}$-order cross-fluctuations. For variance-preserving SDEs, we derive a closed-form for these cross-fluctuations that is efficiently computable for the reverse trajectory. We find that detecting these transitions directly boosts sampling efficiency, accelerates class-conditional and rare-class generation, and improves two zero-shot tasks--image classification and style transfer--without expensive grid search or retraining. We also show that this viewpoint unifies classical coupling and mixing from finite Markov chains with continuous dynamics while extending to stochastic SDEs and non Markovian samplers. Our framework therefore bridges discrete Markov chain theory, phase analysis, and modern generative modeling. 

---
# VLM6D: VLM based 6Dof Pose Estimation based on RGB-D Images 

**Authors**: Md Selim Sarowar, Sungho Kim  

**Link**: [PDF](https://arxiv.org/pdf/2511.00120)  

**Abstract**: The primary challenge in computer vision is precisely calculating the pose of 6D objects, however many current approaches are still fragile and have trouble generalizing from synthetic data to real-world situations with fluctuating lighting, textureless objects, and significant occlusions. To address these limitations, VLM6D, a novel dual-stream architecture that leverages the distinct strengths of visual and geometric data from RGB-D input for robust and precise pose estimation. Our framework uniquely integrates two specialized encoders: a powerful, self-supervised Vision Transformer (DINOv2) processes the RGB modality, harnessing its rich, pre-trained understanding of visual grammar to achieve remarkable resilience against texture and lighting variations. Concurrently, a PointNet++ encoder processes the 3D point cloud derived from depth data, enabling robust geometric reasoning that excels even with the sparse, fragmented data typical of severe occlusion. These complementary feature streams are effectively fused to inform a multi task prediction head. We demonstrate through comprehensive experiments that VLM6D obtained new SOTA performance on the challenging Occluded-LineMOD, validating its superior robustness and accuracy. 

---
# DCcluster-Opt: Benchmarking Dynamic Multi-Objective Optimization for Geo-Distributed Data Center Workloads 

**Authors**: Antonio Guillen-Perez, Avisek Naug, Vineet Gundecha, Sahand Ghorbanpour, Ricardo Luna Gutierrez, Ashwin Ramesh Babu, Munther Salim, Shubhanker Banerjee, Eoin H. Oude Essink, Damien Fay, Soumyendu Sarkar  

**Link**: [PDF](https://arxiv.org/pdf/2511.00117)  

**Abstract**: The increasing energy demands and carbon footprint of large-scale AI require intelligent workload management in globally distributed data centers. Yet progress is limited by the absence of benchmarks that realistically capture the interplay of time-varying environmental factors (grid carbon intensity, electricity prices, weather), detailed data center physics (CPUs, GPUs, memory, HVAC energy), and geo-distributed network dynamics (latency and transmission costs). To bridge this gap, we present DCcluster-Opt: an open-source, high-fidelity simulation benchmark for sustainable, geo-temporal task scheduling. DCcluster-Opt combines curated real-world datasets, including AI workload traces, grid carbon intensity, electricity markets, weather across 20 global regions, cloud transmission costs, and empirical network delay parameters with physics-informed models of data center operations, enabling rigorous and reproducible research in sustainable computing. It presents a challenging scheduling problem where a top-level coordinating agent must dynamically reassign or defer tasks that arrive with resource and service-level agreement requirements across a configurable cluster of data centers to optimize multiple objectives. The environment also models advanced components such as heat recovery. A modular reward system enables an explicit study of trade-offs among carbon emissions, energy costs, service level agreements, and water use. It provides a Gymnasium API with baseline controllers, including reinforcement learning and rule-based strategies, to support reproducible ML research and a fair comparison of diverse algorithms. By offering a realistic, configurable, and accessible testbed, DCcluster-Opt accelerates the development and validation of next-generation sustainable computing solutions for geo-distributed data centers. 

---
# LC-Opt: Benchmarking Reinforcement Learning and Agentic AI for End-to-End Liquid Cooling Optimization in Data Centers 

**Authors**: Avisek Naug, Antonio Guillen, Vineet Kumar, Scott Greenwood, Wesley Brewer, Sahand Ghorbanpour, Ashwin Ramesh Babu, Vineet Gundecha, Ricardo Luna Gutierrez, Soumyendu Sarkar  

**Link**: [PDF](https://arxiv.org/pdf/2511.00116)  

**Abstract**: Liquid cooling is critical for thermal management in high-density data centers with the rising AI workloads. However, machine learning-based controllers are essential to unlock greater energy efficiency and reliability, promoting sustainability. We present LC-Opt, a Sustainable Liquid Cooling (LC) benchmark environment, for reinforcement learning (RL) control strategies in energy-efficient liquid cooling of high-performance computing (HPC) systems. Built on the baseline of a high-fidelity digital twin of Oak Ridge National Lab's Frontier Supercomputer cooling system, LC-Opt provides detailed Modelica-based end-to-end models spanning site-level cooling towers to data center cabinets and server blade groups. RL agents optimize critical thermal controls like liquid supply temperature, flow rate, and granular valve actuation at the IT cabinet level, as well as cooling tower (CT) setpoints through a Gymnasium interface, with dynamic changes in workloads. This environment creates a multi-objective real-time optimization challenge balancing local thermal regulation and global energy efficiency, and also supports additional components like a heat recovery unit (HRU). We benchmark centralized and decentralized multi-agent RL approaches, demonstrate policy distillation into decision and regression trees for interpretable control, and explore LLM-based methods that explain control actions in natural language through an agentic mesh architecture designed to foster user trust and simplify system management. LC-Opt democratizes access to detailed, customizable liquid cooling models, enabling the ML community, operators, and vendors to develop sustainable data center liquid cooling control solutions. 

---
# Cognitive Alignment in Personality Reasoning: Leveraging Prototype Theory for MBTI Inference 

**Authors**: Haoyuan Li, Yuanbo Tong, Yuchen Li, Zirui Wang, Chunhou Liu, Jiamou Liu  

**Link**: [PDF](https://arxiv.org/pdf/2511.00115)  

**Abstract**: Personality recognition from text is typically cast as hard-label classification, which obscures the graded, prototype-like nature of human personality judgments. We present ProtoMBTI, a cognitively aligned framework for MBTI inference that operationalizes prototype theory within an LLM-based pipeline. First, we construct a balanced, quality-controlled corpus via LLM-guided multi-dimensional augmentation (semantic, linguistic, sentiment). Next, we LoRA-fine-tune a lightweight (<=2B) encoder to learn discriminative embeddings and to standardize a bank of personality prototypes. At inference, we retrieve top-k prototypes for a query post and perform a retrieve--reuse--revise--retain cycle: the model aggregates prototype evidence via prompt-based voting, revises when inconsistencies arise, and, upon correct prediction, retains the sample to continually enrich the prototype library. Across Kaggle and Pandora benchmarks, ProtoMBTI improves over baselines on both the four MBTI dichotomies and the full 16-type task, and exhibits robust cross-dataset generalization. Our results indicate that aligning the inference process with psychological prototype reasoning yields gains in accuracy, interpretability, and transfer for text-based personality modeling. 

---
# End-to-End Framework Integrating Generative AI and Deep Reinforcement Learning for Autonomous Ultrasound Scanning 

**Authors**: Hanae Elmekki, Amanda Spilkin, Ehsan Zakeri, Antonela Mariel Zanuttini, Ahmed Alagha, Hani Sami, Jamal Bentahar, Lyes Kadem, Wen-Fang Xie, Philippe Pibarot, Rabeb Mizouni, Hadi Otrok, Azzam Mourad, Sami Muhaidat  

**Link**: [PDF](https://arxiv.org/pdf/2511.00114)  

**Abstract**: Cardiac ultrasound (US) is among the most widely used diagnostic tools in cardiology for assessing heart health, but its effectiveness is limited by operator dependence, time constraints, and human error. The shortage of trained professionals, especially in remote areas, further restricts access. These issues underscore the need for automated solutions that can ensure consistent, and accessible cardiac imaging regardless of operator skill or location. Recent progress in artificial intelligence (AI), especially in deep reinforcement learning (DRL), has gained attention for enabling autonomous decision-making. However, existing DRL-based approaches to cardiac US scanning lack reproducibility, rely on proprietary data, and use simplified models. Motivated by these gaps, we present the first end-to-end framework that integrates generative AI and DRL to enable autonomous and reproducible cardiac US scanning. The framework comprises two components: (i) a conditional generative simulator combining Generative Adversarial Networks (GANs) with Variational Autoencoders (VAEs), that models the cardiac US environment producing realistic action-conditioned images; and (ii) a DRL module that leverages this simulator to learn autonomous, accurate scanning policies. The proposed framework delivers AI-driven guidance through expert-validated models that classify image type and assess quality, supports conditional generation of realistic US images, and establishes a reproducible foundation extendable to other organs. To ensure reproducibility, a publicly available dataset of real cardiac US scans is released. The solution is validated through several experiments. The VAE-GAN is benchmarked against existing GAN variants, with performance assessed using qualitative and quantitative approaches, while the DRL-based scanning system is evaluated under varying configurations to demonstrate effectiveness. 

---
# Real-DRL: Teach and Learn in Reality 

**Authors**: Yanbing Mao, Yihao Cai, Lui Sha  

**Link**: [PDF](https://arxiv.org/pdf/2511.00112)  

**Abstract**: This paper introduces the Real-DRL framework for safety-critical autonomous systems, enabling runtime learning of a deep reinforcement learning (DRL) agent to develop safe and high-performance action policies in real plants (i.e., real physical systems to be controlled), while prioritizing safety! The Real-DRL consists of three interactive components: a DRL-Student, a PHY-Teacher, and a Trigger. The DRL-Student is a DRL agent that innovates in the dual self-learning and teaching-to-learn paradigm and the real-time safety-informed batch sampling. On the other hand, PHY-Teacher is a physics-model-based design of action policies that focuses solely on safety-critical functions. PHY-Teacher is novel in its real-time patch for two key missions: i) fostering the teaching-to-learn paradigm for DRL-Student and ii) backing up the safety of real plants. The Trigger manages the interaction between the DRL-Student and the PHY-Teacher. Powered by the three interactive components, the Real-DRL can effectively address safety challenges that arise from the unknown unknowns and the Sim2Real gap. Additionally, Real-DRL notably features i) assured safety, ii) automatic hierarchy learning (i.e., safety-first learning and then high-performance learning), and iii) safety-informed batch sampling to address the learning experience imbalance caused by corner cases. Experiments with a real quadruped robot, a quadruped robot in NVIDIA Isaac Gym, and a cart-pole system, along with comparisons and ablation studies, demonstrate the Real-DRL's effectiveness and unique features. 

---
# Chain of Time: In-Context Physical Simulation with Image Generation Models 

**Authors**: YingQiao Wang, Eric Bigelow, Boyi Li, Tomer Ullman  

**Link**: [PDF](https://arxiv.org/pdf/2511.00110)  

**Abstract**: We propose a novel cognitively-inspired method to improve and interpret physical simulation in vision-language models. Our ``Chain of Time" method involves generating a series of intermediate images during a simulation, and it is motivated by in-context reasoning in machine learning, as well as mental simulation in humans. Chain of Time is used at inference time, and requires no additional fine-tuning. We apply the Chain-of-Time method to synthetic and real-world domains, including 2-D graphics simulations and natural 3-D videos. These domains test a variety of particular physical properties, including velocity, acceleration, fluid dynamics, and conservation of momentum. We found that using Chain-of-Time simulation substantially improves the performance of a state-of-the-art image generation model. Beyond examining performance, we also analyzed the specific states of the world simulated by an image model at each time step, which sheds light on the dynamics underlying these simulations. This analysis reveals insights that are hidden from traditional evaluations of physical reasoning, including cases where an image generation model is able to simulate physical properties that unfold over time, such as velocity, gravity, and collisions. Our analysis also highlights particular cases where the image generation model struggles to infer particular physical parameters from input images, despite being capable of simulating relevant physical processes. 

---
# Pelican-VL 1.0: A Foundation Brain Model for Embodied Intelligence 

**Authors**: Yi Zhang, Che Liu, Xiancong Ren, Hanchu Ni, Shuai Zhang, Zeyuan Ding, Jiayu Hu, Hanzhe Shan, Zhenwei Niu, Zhaoyang Liu, Yue Zhao, Junbo Qi, Qinfan Zhang, Dengjie Li, Yidong Wang, Jiachen Luo, Yong Dai, Jian Tang, Xiaozhu Ju  

**Link**: [PDF](https://arxiv.org/pdf/2511.00108)  

**Abstract**: This report presents Pelican-VL 1.0, a new family of open-source embodied brain models with parameter scales ranging from 7 billion to 72 billion. Our explicit mission is clearly stated as: To embed powerful intelligence into various embodiments. Pelican-VL 1.0 is currently the largest-scale open-source embodied multimodal brain model. Its core advantage lies in the in-depth integration of data power and intelligent adaptive learning mechanisms. Specifically, metaloop distilled a high-quality dataset from a raw dataset containing 4+ billion tokens. Pelican-VL 1.0 is trained on a large-scale cluster of 1000+ A800 GPUs, consuming over 50k+ A800 GPU-hours per checkpoint. This translates to a 20.3% performance uplift from its base model and outperforms 100B-level open-source counterparts by 10.6%, placing it on par with leading proprietary systems on well-known embodied benchmarks. We establish a novel framework, DPPO (Deliberate Practice Policy Optimization), inspired by human metacognition to train Pelican-VL 1.0. We operationalize this as a metaloop that teaches the AI to practice deliberately, which is a RL-Refine-Diagnose-SFT loop. 

---
# AI Powered High Quality Text to Video Generation with Enhanced Temporal Consistency 

**Authors**: Piyushkumar Patel  

**Link**: [PDF](https://arxiv.org/pdf/2511.00107)  

**Abstract**: Text to video generation has emerged as a critical frontier in generative artificial intelligence, yet existing approaches struggle with maintaining temporal consistency, compositional understanding, and fine grained control over visual narratives. We present MOVAI (Multimodal Original Video AI), a novel hierarchical framework that integrates compositional scene understanding with temporal aware diffusion models for high fidelity text to video synthesis. Our approach introduces three key innovations: (1) a Compositional Scene Parser (CSP) that decomposes textual descriptions into hierarchical scene graphs with temporal annotations, (2) a Temporal-Spatial Attention Mechanism (TSAM) that ensures coherent motion dynamics across frames while preserving spatial details, and (3) a Progressive Video Refinement (PVR) module that iteratively enhances video quality through multi-scale temporal reasoning. Extensive experiments on standard benchmarks demonstrate that MOVAI achieves state-of-the-art performance, improving video quality metrics by 15.3% in LPIPS, 12.7% in FVD, and 18.9% in user preference studies compared to existing methods. Our framework shows particular strength in generating complex multi-object scenes with realistic temporal dynamics and fine-grained semantic control. 

---
# Wayfinding through the AI wilderness: Mapping rhetorics of ChatGPT prompt writing on X (formerly Twitter) to promote critical AI literacies 

**Authors**: Anuj Gupta, Ann Shivers-McNair  

**Link**: [PDF](https://arxiv.org/pdf/2511.00106)  

**Abstract**: In this paper, we demonstrate how studying the rhetorics of ChatGPT prompt writing on social media can promote critical AI literacies. Prompt writing is the process of writing instructions for generative AI tools like ChatGPT to elicit desired outputs and there has been an upsurge of conversations about it on social media. To study this rhetorical activity, we build on four overlapping traditions of digital writing research in computers and composition that inform how we frame literacies, how we study social media rhetorics, how we engage iteratively and reflexively with methodologies and technologies, and how we blend computational methods with qualitative methods. Drawing on these four traditions, our paper shows our iterative research process through which we gathered and analyzed a dataset of 32,000 posts (formerly known as tweets) from X (formerly Twitter) about prompt writing posted between November 2022 to May 2023. We present five themes about these emerging AI literacy practices: (1) areas of communication impacted by prompt writing, (2) micro-literacy resources shared for prompt writing, (3) market rhetoric shaping prompt writing, (4) rhetorical characteristics of prompts, and (5) definitions of prompt writing. In discussing these themes and our methodologies, we highlight takeaways for digital writing teachers and researchers who are teaching and analyzing critical AI literacies. 

---
# Artificial Intelligence in Elementary STEM Education: A Systematic Review of Current Applications and Future Challenges 

**Authors**: Majid Memari, Krista Ruggles  

**Link**: [PDF](https://arxiv.org/pdf/2511.00105)  

**Abstract**: Artificial intelligence (AI) is transforming elementary STEM education, yet evidence remains fragmented. This systematic review synthesizes 258 studies (2020-2025) examining AI applications across eight categories: intelligent tutoring systems (45% of studies), learning analytics (18%), automated assessment (12%), computer vision (8%), educational robotics (7%), multimodal sensing (6%), AI-enhanced extended reality (XR) (4%), and adaptive content generation. The analysis shows that most studies focus on upper elementary grades (65%) and mathematics (38%), with limited cross-disciplinary STEM integration (15%). While conversational AI demonstrates moderate effectiveness (d = 0.45-0.70 where reported), only 34% of studies include standardized effect sizes. Eight major gaps limit real-world impact: fragmented ecosystems, developmental inappropriateness, infrastructure barriers, lack of privacy frameworks, weak STEM integration, equity disparities, teacher marginalization, and narrow assessment scopes. Geographic distribution is also uneven, with 90% of studies originating from North America, East Asia, and Europe. Future directions call for interoperable architectures that support authentic STEM integration, grade-appropriate design, privacy-preserving analytics, and teacher-centered implementations that enhance rather than replace human expertise. 

---
# FreeSliders: Training-Free, Modality-Agnostic Concept Sliders for Fine-Grained Diffusion Control in Images, Audio, and Video 

**Authors**: Rotem Ezra, Hedi Zisling, Nimrod Berman, Ilan Naiman, Alexey Gorkor, Liran Nochumsohn, Eliya Nachmani, Omri Azencot  

**Link**: [PDF](https://arxiv.org/pdf/2511.00103)  

**Abstract**: Diffusion models have become state-of-the-art generative models for images, audio, and video, yet enabling fine-grained controllable generation, i.e., continuously steering specific concepts without disturbing unrelated content, remains challenging. Concept Sliders (CS) offer a promising direction by discovering semantic directions through textual contrasts, but they require per-concept training and architecture-specific fine-tuning (e.g., LoRA), limiting scalability to new modalities. In this work we introduce FreeSliders, a simple yet effective approach that is fully training-free and modality-agnostic, achieved by partially estimating the CS formula during inference. To support modality-agnostic evaluation, we extend the CS benchmark to include both video and audio, establishing the first suite for fine-grained concept generation control with multiple modalities. We further propose three evaluation properties along with new metrics to improve evaluation quality. Finally, we identify an open problem of scale selection and non-linear traversals and introduce a two-stage procedure that automatically detects saturation points and reparameterizes traversal for perceptually uniform, semantically meaningful edits. Extensive experiments demonstrate that our method enables plug-and-play, training-free concept control across modalities, improves over existing baselines, and establishes new tools for principled controllable generation. An interactive presentation of our benchmark and method is available at: this https URL 

---
# Automated Discovery of Conservation Laws via Hybrid Neural ODE-Transformers 

**Authors**: Vivan Doshi  

**Link**: [PDF](https://arxiv.org/pdf/2511.00102)  

**Abstract**: The discovery of conservation laws is a cornerstone of scientific progress. However, identifying these invariants from observational data remains a significant challenge. We propose a hybrid framework to automate the discovery of conserved quantities from noisy trajectory data. Our approach integrates three components: (1) a Neural Ordinary Differential Equation (Neural ODE) that learns a continuous model of the system's dynamics, (2) a Transformer that generates symbolic candidate invariants conditioned on the learned vector field, and (3) a symbolic-numeric verifier that provides a strong numerical certificate for the validity of these candidates. We test our framework on canonical physical systems and show that it significantly outperforms baselines that operate directly on trajectory data. This work demonstrates the robustness of a decoupled learn-then-search approach for discovering mathematical principles from imperfect data. 

---
# Loquetier: A Virtualized Multi-LoRA Framework for Unified LLM Fine-tuning and Serving 

**Authors**: Yuchen Zhang, Hanyue Du, Chun Cao, Jingwei Xu  

**Link**: [PDF](https://arxiv.org/pdf/2511.00101)  

**Abstract**: Low-Rank Adaptation (LoRA) has become a widely adopted parameter-efficient fine-tuning (PEFT) technique for adapting large language models (LLMs) to downstream tasks. While prior work has explored strategies for integrating LLM training and serving, there still remains a gap in unifying fine-tuning and inference for LoRA-based models. We present Loquetier, a virtualized multi-LoRA framework that seamlessly integrates LoRA fine-tuning and serving within a single runtime. Loquetier introduces two key components: (1) a Virtualized Module that isolates PEFT-based modifications and supports multiple adapters on a shared base model, and (2) an optimized computation flow with a kernel design that merges fine-tuning and inference paths in forward propagation, enabling efficient batching and minimizing kernel invocation overhead. Extensive experiments across three task settings show that Loquetier consistently outperforms existing baselines in both performance and flexibility, achieving up to $3.0\times$ the throughput of the state-of-the-art co-serving system on inference-only tasks and $46.4\times$ higher SLO attainment than PEFT on unified fine-tuning and inference tasks. The implementation of Loquetier is publicly available at this https URL. 

---
# A generative adversarial network optimization method for damage detection and digital twinning by deep AI fault learning: Z24 Bridge structural health monitoring benchmark validation 

**Authors**: Marios Impraimakis, Evangelia Nektaria Palkanoglou  

**Link**: [PDF](https://arxiv.org/pdf/2511.00099)  

**Abstract**: The optimization-based damage detection and damage state digital twinning capabilities are examined here of a novel conditional-labeled generative adversarial network methodology. The framework outperforms current approaches for fault anomaly detection as no prior information is required for the health state of the system: a topic of high significance for real-world applications. Specifically, current artificial intelligence-based digital twinning approaches suffer from the uncertainty related to obtaining poor predictions when a low number of measurements is available, physics knowledge is missing, or when the damage state is unknown. To this end, an unsupervised framework is examined and validated rigorously on the benchmark structural health monitoring measurements of Z24 Bridge: a post-tensioned concrete highway bridge in Switzerland. In implementing the approach, firstly, different same damage-level measurements are used as inputs, while the model is forced to converge conditionally to two different damage states. Secondly, the process is repeated for a different group of measurements. Finally, the convergence scores are compared to identify which one belongs to a different damage state. The process for both healthy-to-healthy and damage-to-healthy input data creates, simultaneously, measurements for digital twinning purposes at different damage states, capable of pattern recognition and machine learning data generation. Further to this process, a support vector machine classifier and a principal component analysis procedure is developed to assess the generated and real measurements of each damage category, serving as a secondary new dynamics learning indicator in damage scenarios. Importantly, the approach is shown to capture accurately damage over healthy measurements, providing a powerful tool for vibration-based system-level monitoring and scalable infrastructure resilience. 

---
# A filtering scheme for confocal laser endomicroscopy (CLE)-video sequences for self-supervised learning 

**Authors**: Nils Porsche, Flurin Müller-Diesing, Sweta Banerjee, Miguel Goncalves, Marc Aubreville  

**Link**: [PDF](https://arxiv.org/pdf/2511.00098)  

**Abstract**: Confocal laser endomicroscopy (CLE) is a non-invasive, real-time imaging modality that can be used for in-situ, in-vivo imaging and the microstructural analysis of mucous structures. The diagnosis using CLE is, however, complicated by images being hard to interpret for non-experienced physicians. Utilizing machine learning as an augmentative tool would hence be beneficial, but is complicated by the shortage of histopathology-correlated CLE imaging sequences with respect to the plurality of patterns in this domain, leading to overfitting of machine learning models. To overcome this, self-supervised learning (SSL) can be employed on larger unlabeled datasets. CLE is a video-based modality with high inter-frame correlation, leading to a non-stratified data distribution for SSL training. In this work, we propose a filter functionality on CLE video sequences to reduce the dataset redundancy in SSL training and improve SSL training convergence and training efficiency. We use four state-of-the-art baseline networks and a SSL teacher-student network with a vision transformer small backbone for the evaluation. These networks were evaluated on downstream tasks for a sinonasal tumor dataset and a squamous cell carcinoma of the skin dataset. On both datasets, we found the highest test accuracy on the filtered SSL-pretrained model, with 67.48% and 73.52%, both considerably outperforming their non-SSL baselines. Our results show that SSL is an effective method for CLE pretraining. Further, we show that our proposed CLE video filter can be utilized to improve training efficiency in self-supervised scenarios, resulting in a reduction of 67% in training time. 

---
# GraphKeeper: Graph Domain-Incremental Learning via Knowledge Disentanglement and Preservation 

**Authors**: Zihao Guo, Qingyun Sun, Ziwei Zhang, Haonan Yuan, Huiping Zhuang, Xingcheng Fu, Jianxin Li  

**Link**: [PDF](https://arxiv.org/pdf/2511.00097)  

**Abstract**: Graph incremental learning (GIL), which continuously updates graph models by sequential knowledge acquisition, has garnered significant interest recently. However, existing GIL approaches focus on task-incremental and class-incremental scenarios within a single domain. Graph domain-incremental learning (Domain-IL), aiming at updating models across multiple graph domains, has become critical with the development of graph foundation models (GFMs), but remains unexplored in the literature. In this paper, we propose Graph Domain-Incremental Learning via Knowledge Dientanglement and Preservation (GraphKeeper), to address catastrophic forgetting in Domain-IL scenario from the perspectives of embedding shifts and decision boundary deviations. Specifically, to prevent embedding shifts and confusion across incremental graph domains, we first propose the domain-specific parameter-efficient fine-tuning together with intra- and inter-domain disentanglement objectives. Consequently, to maintain a stable decision boundary, we introduce deviation-free knowledge preservation to continuously fit incremental domains. Additionally, for graphs with unobservable domains, we perform domain-aware distribution discrimination to obtain precise embeddings. Extensive experiments demonstrate the proposed GraphKeeper achieves state-of-the-art results with 6.5%~16.6% improvement over the runner-up with negligible forgetting. Moreover, we show GraphKeeper can be seamlessly integrated with various representative GFMs, highlighting its broad applicative potential. 

---
# Urban-MAS: Human-Centered Urban Prediction with LLM-Based Multi-Agent System 

**Authors**: Shangyu Lou  

**Link**: [PDF](https://arxiv.org/pdf/2511.00096)  

**Abstract**: Urban Artificial Intelligence (Urban AI) has advanced human-centered urban tasks such as perception prediction and human dynamics. Large Language Models (LLMs) can integrate multimodal inputs to address heterogeneous data in complex urban systems but often underperform on domain-specific tasks. Urban-MAS, an LLM-based Multi-Agent System (MAS) framework, is introduced for human- centered urban prediction under zero-shot settings. It includes three agent types: Predictive Factor Guidance Agents, which prioritize key predictive factors to guide knowledge extraction and enhance the effectiveness of compressed urban knowledge in LLMs; Reliable UrbanInfo Extraction Agents, which improve robustness by com- paring multiple outputs, validating consistency, and re-extracting when conflicts occur; and Multi-UrbanInfo Inference Agents, which integrate extracted multi-source information across dimensions for prediction. Experiments on running-amount prediction and ur- ban perception across Tokyo, Milan, and Seattle demonstrate that Urban-MAS substantially reduces errors compared to single-LLM baselines. Ablation studies indicate that Predictive Factor Guidance Agents are most critical for enhancing predictive performance, po- sitioning Urban-MAS as a scalable paradigm for human-centered urban AI prediction. Code is available on the project website:this https URL 

---
# SpinalSAM-R1: A Vision-Language Multimodal Interactive System for Spine CT Segmentation 

**Authors**: Jiaming Liu, Dingwei Fan, Junyong Zhao, Chunlin Li, Haipeng Si, Liang Sun  

**Link**: [PDF](https://arxiv.org/pdf/2511.00095)  

**Abstract**: The anatomical structure segmentation of the spine and adjacent structures from computed tomography (CT) images is a key step for spinal disease diagnosis and treatment. However, the segmentation of CT images is impeded by low contrast and complex vertebral boundaries. Although advanced models such as the Segment Anything Model (SAM) have shown promise in various segmentation tasks, their performance in spinal CT imaging is limited by high annotation requirements and poor domain adaptability. To address these limitations, we propose SpinalSAM-R1, a multimodal vision-language interactive system that integrates a fine-tuned SAM with DeepSeek-R1, for spine CT image segmentation. Specifically, our SpinalSAM-R1 introduces an anatomy-guided attention mechanism to improve spine segmentation performance, and a semantics-driven interaction protocol powered by DeepSeek-R1, enabling natural language-guided refinement. The SpinalSAM-R1 is fine-tuned using Low-Rank Adaptation (LoRA) for efficient adaptation. We validate our SpinalSAM-R1 on the spine anatomical structure with CT images. Experimental results suggest that our method achieves superior segmentation performance. Meanwhile, we develop a PyQt5-based interactive software, which supports point, box, and text-based prompts. The system supports 11 clinical operations with 94.3\% parsing accuracy and sub-800 ms response times. The software is released on this https URL. 

---
# Digital Twin based Automatic Reconfiguration of Robotic Systems in Smart Environments 

**Authors**: Angelos Alexopoulos, Agorakis Bompotas, Nikitas Rigas Kalogeropoulos, Panagiotis Kechagias, Athanasios P. Kalogeras, Christos Alexakos  

**Link**: [PDF](https://arxiv.org/pdf/2511.00094)  

**Abstract**: Robotic systems have become integral to smart environments, enabling applications ranging from urban surveillance and automated agriculture to industrial automation. However, their effective operation in dynamic settings - such as smart cities and precision farming - is challenged by continuously evolving topographies and environmental conditions. Traditional control systems often struggle to adapt quickly, leading to inefficiencies or operational failures. To address this limitation, we propose a novel framework for autonomous and dynamic reconfiguration of robotic controllers using Digital Twin technology. Our approach leverages a virtual replica of the robot's operational environment to simulate and optimize movement trajectories in response to real-world changes. By recalculating paths and control parameters in the Digital Twin and deploying the updated code to the physical robot, our method ensures rapid and reliable adaptation without manual intervention. This work advances the integration of Digital Twins in robotics, offering a scalable solution for enhancing autonomy in smart, dynamic environments. 

---
# LeMiCa: Lexicographic Minimax Path Caching for Efficient Diffusion-Based Video Generation 

**Authors**: Huanlin Gao, Ping Chen, Fuyuan Shi, Chao Tan, Zhaoxiang Liu, Fang Zhao, Kai Wang, Shiguo Lian  

**Link**: [PDF](https://arxiv.org/pdf/2511.00090)  

**Abstract**: We present LeMiCa, a training-free and efficient acceleration framework for diffusion-based video generation. While existing caching strategies primarily focus on reducing local heuristic errors, they often overlook the accumulation of global errors, leading to noticeable content degradation between accelerated and original videos. To address this issue, we formulate cache scheduling as a directed graph with error-weighted edges and introduce a Lexicographic Minimax Path Optimization strategy that explicitly bounds the worst-case path error. This approach substantially improves the consistency of global content and style across generated frames. Extensive experiments on multiple text-to-video benchmarks demonstrate that LeMiCa delivers dual improvements in both inference speed and generation quality. Notably, our method achieves a 2.9x speedup on the Latte model and reaches an LPIPS score of 0.05 on Open-Sora, outperforming prior caching techniques. Importantly, these gains come with minimal perceptual quality degradation, making LeMiCa a robust and generalizable paradigm for accelerating diffusion-based video generation. We believe this approach can serve as a strong foundation for future research on efficient and reliable video synthesis. Our code is available at :this https URL 

---
# Alpamayo-R1: Bridging Reasoning and Action Prediction for Generalizable Autonomous Driving in the Long Tail 

**Authors**: NVIDIA, Yan Wang, Wenjie Luo, Junjie Bai, Yulong Cao, Tong Che, Ke Chen, Yuxiao Chen, Jenna Diamond, Yifan Ding, Wenhao Ding, Liang Feng, Greg Heinrich, Jack Huang, Peter Karkus, Boyi Li, Pinyi Li, Tsung-Yi Lin, Dongran Liu, Ming-Yu Liu, Langechuan Liu, Zhijian Liu, Jason Lu, Yunxiang Mao, Pavlo Molchanov, Lindsey Pavao, Zhenghao Peng, Mike Ranzinger, Ed Schmerling, Shida Shen, Yunfei Shi, Sarah Tariq, Ran Tian, Tilman Wekel, Xinshuo Weng, Tianjun Xiao, Eric Yang, Xiaodong Yang, Yurong You, Xiaohui Zeng, Wenyuan Zhang, Boris Ivanovic, Marco Pavone  

**Link**: [PDF](https://arxiv.org/pdf/2511.00088)  

**Abstract**: End-to-end architectures trained via imitation learning have advanced autonomous driving by scaling model size and data, yet performance remains brittle in safety-critical long-tail scenarios where supervision is sparse and causal understanding is limited. To address this, we introduce Alpamayo-R1 (AR1), a vision-language-action model (VLA) that integrates Chain of Causation reasoning with trajectory planning to enhance decision-making in complex driving scenarios. Our approach features three key innovations: (1) the Chain of Causation (CoC) dataset, built through a hybrid auto-labeling and human-in-the-loop pipeline producing decision-grounded, causally linked reasoning traces aligned with driving behaviors; (2) a modular VLA architecture combining Cosmos-Reason, a Vision-Language Model pre-trained for Physical AI applications, with a diffusion-based trajectory decoder that generates dynamically feasible plans in real time; (3) a multi-stage training strategy using supervised fine-tuning to elicit reasoning and reinforcement learning (RL) to optimize reasoning quality via large reasoning model feedback and enforce reasoning-action consistency. Evaluation shows AR1 achieves up to a 12% improvement in planning accuracy on challenging cases compared to a trajectory-only baseline, with a 35% reduction in off-road rate and 25% reduction in close encounter rate in closed-loop simulation. RL post-training improves reasoning quality by 45% as measured by a large reasoning model critic and reasoning-action consistency by 37%. Model scaling from 0.5B to 7B parameters shows consistent improvements. On-vehicle road tests confirm real-time performance (99 ms latency) and successful urban deployment. By bridging interpretable reasoning with precise control, AR1 demonstrates a practical path towards Level 4 autonomous driving. We plan to release AR1 models and a subset of the CoC in a future update. 

---
# Adding New Capability in Existing Scientific Application with LLM Assistance 

**Authors**: Anshu Dubey, Akash Dhruv  

**Link**: [PDF](https://arxiv.org/pdf/2511.00087)  

**Abstract**: With the emergence and rapid evolution of large language models (LLM), automating coding tasks has become an im- portant research topic. Many efforts are underway and liter- ature abounds about the efficacy of models and their ability to generate code. A less explored aspect of code generation is for new algorithms, where the training data-set would not have included any previous example of similar code. In this paper we propose a new methodology for writing code from scratch for a new algorithm using LLM assistance, and describe enhancement of a previously developed code- translation tool, Code-Scribe, for new code generation. 

---
# Generalizing Test-time Compute-optimal Scaling as an Optimizable Graph 

**Authors**: Fali Wang, Jihai Chen, Shuhua Yang, Runxue Bao, Tianxiang Zhao, Zhiwei Zhang, Xianfeng Tang, Hui Liu, Qi He, Suhang Wang  

**Link**: [PDF](https://arxiv.org/pdf/2511.00086)  

**Abstract**: Test-Time Scaling (TTS) improves large language models (LLMs) by allocating additional computation during inference, typically through parallel, sequential, or hybrid scaling. However, prior studies often assume fixed collaboration architectures (e.g., topologies) and single-model usage, overlooking that optimal architectures and model combinations can vary across tasks. Therefore, we study the novel problem of searching for compute-optimal model combinations and architectures in TTS under a fixed budget. We formalize it as a multi-LLM collaboration graph, where nodes encode roles and LLM model assignments, and edges capture information flow. This problem is challenging because (i) the combinatorial search space is prohibitively large, and (ii) task-specific requirements demand tailored designs. To address these, we reformulate the problem as probabilistic graph optimization and, through pilot experiments, derive three empirical insights into TTS collaboration graphs. Guided by these insights, we propose Agent-REINFORCE, an LLM-agent-augmented framework that mirrors the REINFORCE pipeline by mapping sampling-gradient-update to sampling-feedback-update, where feedback serves as a textual gradient to update the probabilistic graph and efficiently search for optimal multi-LLM collaboration graphs. Experiments show that Agent-REINFORCE outperforms both traditional and LLM-based baselines in sample efficiency and search performance, and effectively identifies optimal graphs under joint objectives of accuracy and inference latency. 

---
# MaGNet: A Mamba Dual-Hypergraph Network for Stock Prediction via Temporal-Causal and Global Relational Learning 

**Authors**: Peilin Tan, Chuanqi Shi, Dian Tu, Liang Xie  

**Link**: [PDF](https://arxiv.org/pdf/2511.00085)  

**Abstract**: Stock trend prediction is crucial for profitable trading strategies and portfolio management yet remains challenging due to market volatility, complex temporal dynamics and multifaceted inter-stock relationships. Existing methods struggle to effectively capture temporal dependencies and dynamic inter-stock interactions, often neglecting cross-sectional market influences, relying on static correlations, employing uniform treatments of nodes and edges, and conflating diverse relationships. This work introduces MaGNet, a novel Mamba dual-hyperGraph Network for stock prediction, integrating three key innovations: (1) a MAGE block, which leverages bidirectional Mamba with adaptive gating mechanisms for contextual temporal modeling and integrates a sparse Mixture-of-Experts layer to enable dynamic adaptation to diverse market conditions, alongside multi-head attention for capturing global dependencies; (2) Feature-wise and Stock-wise 2D Spatiotemporal Attention modules enable precise fusion of multivariate features and cross-stock dependencies, effectively enhancing informativeness while preserving intrinsic data structures, bridging temporal modeling with relational reasoning; and (3) a dual hypergraph framework consisting of the Temporal-Causal Hypergraph (TCH) that captures fine-grained causal dependencies with temporal constraints, and Global Probabilistic Hypergraph (GPH) that models market-wide patterns through soft hyperedge assignments and Jensen-Shannon Divergence weighting mechanism, jointly disentangling localized temporal influences from instantaneous global structures for multi-scale relational learning. Extensive experiments on six major stock indices demonstrate MaGNet outperforms state-of-the-art methods in both superior predictive performance and exceptional investment returns with robust risk management capabilities. Codes available at: this https URL. 

---
# Application of predictive machine learning in pen & paper RPG game design 

**Authors**: Jolanta Śliwa  

**Link**: [PDF](https://arxiv.org/pdf/2511.00084)  

**Abstract**: In recent years, the pen and paper RPG market has experienced significant growth. As a result, companies are increasingly exploring the integration of AI technologies to enhance player experience and gain a competitive edge.
One of the key challenges faced by publishers is designing new opponents and estimating their challenge level. Currently, there are no automated methods for determining a monster's level; the only approaches used are based on manual testing and expert evaluation. Although these manual methods can provide reasonably accurate estimates, they are time-consuming and resource-intensive.
Level prediction can be approached using ordinal regression techniques. This thesis presents an overview and evaluation of state-of-the-art methods for this task. It also details the construction of a dedicated dataset for level estimation. Furthermore, a human-inspired model was developed to serve as a benchmark, allowing comparison between machine learning algorithms and the approach typically employed by pen and paper RPG publishers. In addition, a specialized evaluation procedure, grounded in domain knowledge, was designed to assess model performance and facilitate meaningful comparisons. 

---
# Fixed-point graph convolutional networks against adversarial attacks 

**Authors**: Shakib Khan, A. Ben Hamza, Amr Youssef  

**Link**: [PDF](https://arxiv.org/pdf/2511.00083)  

**Abstract**: Adversarial attacks present a significant risk to the integrity and performance of graph neural networks, particularly in tasks where graph structure and node features are vulnerable to manipulation. In this paper, we present a novel model, called fixed-point iterative graph convolutional network (Fix-GCN), which achieves robustness against adversarial perturbations by effectively capturing higher-order node neighborhood information in the graph without additional memory or computational complexity. Specifically, we introduce a versatile spectral modulation filter and derive the feature propagation rule of our model using fixed-point iteration. Unlike traditional defense mechanisms that rely on additional design elements to counteract attacks, the proposed graph filter provides a flexible-pass filtering approach, allowing it to selectively attenuate high-frequency components while preserving low-frequency structural information in the graph signal. By iteratively updating node representations, our model offers a flexible and efficient framework for preserving essential graph information while mitigating the impact of adversarial manipulation. We demonstrate the effectiveness of the proposed model through extensive experiments on various benchmark graph datasets, showcasing its resilience against adversarial attacks. 

---
# RailEstate: An Interactive System for Metro Linked Property Trends 

**Authors**: Chen-Wei Chang, Yu-Chieh Cheng, Yun-En Tsai, Fanglan Chen, Chang-Tien Lu  

**Link**: [PDF](https://arxiv.org/pdf/2511.00078)  

**Abstract**: Access to metro systems plays a critical role in shaping urban housing markets by enhancing neighborhood accessibility and driving property demand. We present RailEstate, a novel web based system that integrates spatial analytics, natural language interfaces, and interactive forecasting to analyze how proximity to metro stations influences residential property prices in the Washington metropolitan area. Unlike static mapping tools or generic listing platforms, RailEstate combines 25 years of historical housing data with transit infrastructure to support low latency geospatial queries, time series visualizations, and predictive modeling. Users can interactively explore ZIP code level price patterns, investigate long term trends, and forecast future housing values around any metro station. A key innovation is our natural language chatbot, which translates plain-English questions e.g., What is the highest price in Falls Church in the year 2000? into executable SQL over a spatial database. This unified and interactive platform empowers urban planners, investors, and residents to derive actionable insights from metro linked housing data without requiring technical expertise. 

---
# LookSync: Large-Scale Visual Product Search System for AI-Generated Fashion Looks 

**Authors**: Pradeep M, Ritesh Pallod, Satyen Abrol, Muthu Raman, Ian Anderson  

**Link**: [PDF](https://arxiv.org/pdf/2511.00072)  

**Abstract**: Generative AI is reshaping fashion by enabling virtual looks and avatars making it essential to find real products that best match AI-generated styles. We propose an end-to-end product search system that has been deployed in a real-world, internet scale which ensures that AI-generated looks presented to users are matched with the most visually and semantically similar products from the indexed vector space. The search pipeline is composed of four key components: query generation, vectorization, candidate retrieval, and reranking based on AI-generated looks. Recommendation quality is evaluated using human-judged accuracy scores. The system currently serves more than 350,000 AI Looks in production per day, covering diverse product categories across global markets of over 12 million products. In our experiments, we observed that across multiple annotators and categories, CLIP outperformed alternative models by a small relative margin of 3--7\% in mean opinion scores. These improvements, though modest in absolute numbers, resulted in noticeably better user perception matches, establishing CLIP as the most reliable backbone for production deployment. 

---
# Benchmarking Generative AI Against Bayesian Optimization for Constrained Multi-Objective Inverse Design 

**Authors**: Muhammad Bilal Awan, Abdul Razzaq, Abdul Shahid  

**Link**: [PDF](https://arxiv.org/pdf/2511.00070)  

**Abstract**: This paper investigates the performance of Large Language Models (LLMs) as generative optimizers for solving constrained multi-objective regression tasks, specifically within the challenging domain of inverse design (property-to-structure mapping). This problem, critical to materials informatics, demands finding complex, feasible input vectors that lie on the Pareto optimal front. While LLMs have demonstrated universal effectiveness across generative and reasoning tasks, their utility in constrained, continuous, high-dimensional numerical spaces tasks they weren't explicitly architected for remains an open research question. We conducted a rigorous comparative study between established Bayesian Optimization (BO) frameworks and a suite of fine-tuned LLMs and BERT models. For BO, we benchmarked the foundational BoTorch Ax implementation against the state-of-the-art q-Expected Hypervolume Improvement (qEHVI, BoTorchM). The generative approach involved fine-tuning models via Parameter-Efficient Fine-Tuning (PEFT), framing the challenge as a regression problem with a custom output head. Our results show that BoTorch qEHVI achieved perfect convergence (GD=0.0), setting the performance ceiling. Crucially, the best-performing LLM (WizardMath-7B) achieved a Generational Distance (GD) of 1.21, significantly outperforming the traditional BoTorch Ax baseline (GD=15.03). We conclude that specialized BO frameworks remain the performance leader for guaranteed convergence, but fine-tuned LLMs are validated as a promising, computationally fast alternative, contributing essential comparative metrics to the field of AI-driven optimization. The findings have direct industrial applications in optimizing formulation design for resins, polymers, and paints, where multi-objective trade-offs between mechanical, rheological, and chemical properties are critical to innovation and production efficiency. 

---
# Latent Domain Prompt Learning for Vision-Language Models 

**Authors**: Zhixing Li, Arsham Gholamzadeh Khoee, Yinan Yu  

**Link**: [PDF](https://arxiv.org/pdf/2511.00067)  

**Abstract**: The objective of domain generalization (DG) is to enable models to be robust against domain shift. DG is crucial for deploying vision-language models (VLMs) in real-world applications, yet most existing methods rely on domain labels that may not be available and often ambiguous. We instead study the DG setting where models must generalize well without access to explicit domain labels. Our key idea is to represent an unseen target domain as a combination of latent domains automatically discovered from training data, enabling the model to adaptively transfer knowledge across domains. To realize this, we perform latent domain clustering on image features and fuse domain-specific text features based on the similarity between the input image and each latent domain. Experiments on four benchmarks show that this strategy yields consistent gains over VLM-based baselines and provides new insights into improving robustness under domain shift. 

---
# Aligning Brain Signals with Multimodal Speech and Vision Embeddings 

**Authors**: Kateryna Shapovalenko, Quentin Auster  

**Link**: [PDF](https://arxiv.org/pdf/2511.00065)  

**Abstract**: When we hear the word "house", we don't just process sound, we imagine walls, doors, memories. The brain builds meaning through layers, moving from raw acoustics to rich, multimodal associations. Inspired by this, we build on recent work from Meta that aligned EEG signals with averaged wav2vec2 speech embeddings, and ask a deeper question: which layers of pre-trained models best reflect this layered processing in the brain? We compare embeddings from two models: wav2vec2, which encodes sound into language, and CLIP, which maps words to images. Using EEG recorded during natural speech perception, we evaluate how these embeddings align with brain activity using ridge regression and contrastive decoding. We test three strategies: individual layers, progressive concatenation, and progressive summation. The findings suggest that combining multimodal, layer-aware representations may bring us closer to decoding how the brain understands language, not just as sound, but as experience. 

---
# World Simulation with Video Foundation Models for Physical AI 

**Authors**: NVIDIA, Arslan Ali, Junjie Bai, Maciej Bala, Yogesh Balaji, Aaron Blakeman, Tiffany Cai, Jiaxin Cao, Tianshi Cao, Elizabeth Cha, Yu-Wei Chao, Prithvijit Chattopadhyay, Mike Chen, Yongxin Chen, Yu Chen, Shuai Cheng, Yin Cui, Jenna Diamond, Yifan Ding, Jiaojiao Fan, Linxi Fan, Liang Feng, Francesco Ferroni, Sanja Fidler, Xiao Fu, Ruiyuan Gao, Yunhao Ge, Jinwei Gu, Aryaman Gupta, Siddharth Gururani, Imad El Hanafi, Ali Hassani, Zekun Hao, Jacob Huffman, Joel Jang, Pooya Jannaty, Jan Kautz, Grace Lam, Xuan Li, Zhaoshuo Li, Maosheng Liao, Chen-Hsuan Lin, Tsung-Yi Lin, Yen-Chen Lin, Huan Ling, Ming-Yu Liu, Xian Liu, Yifan Lu, Alice Luo, Qianli Ma, Hanzi Mao, Kaichun Mo, Seungjun Nah, Yashraj Narang, Abhijeet Panaskar, Lindsey Pavao, Trung Pham, Morteza Ramezanali, Fitsum Reda, Scott Reed, Xuanchi Ren, Haonan Shao, Yue Shen, Stella Shi, Shuran Song, Bartosz Stefaniak, Shangkun Sun, Shitao Tang, Sameena Tasmeen, Lyne Tchapmi, Wei-Cheng Tseng, Jibin Varghese, Andrew Z. Wang, Hao Wang, Haoxiang Wang, Heng Wang, Ting-Chun Wang, Fangyin Wei, Jiashu Xu, Dinghao Yang, Xiaodong Yang, Haotian Ye, Seonghyeon Ye, Xiaohui Zeng, Jing Zhang, Qinsheng Zhang, Kaiwen Zheng, Andrew Zhu, Yuke Zhu  

**Link**: [PDF](https://arxiv.org/pdf/2511.00062)  

**Abstract**: We introduce [Cosmos-Predict2.5], the latest generation of the Cosmos World Foundation Models for Physical AI. Built on a flow-based architecture, [Cosmos-Predict2.5] unifies Text2World, Image2World, and Video2World generation in a single model and leverages [Cosmos-Reason1], a Physical AI vision-language model, to provide richer text grounding and finer control of world simulation. Trained on 200M curated video clips and refined with reinforcement learning-based post-training, [Cosmos-Predict2.5] achieves substantial improvements over [Cosmos-Predict1] in video quality and instruction alignment, with models released at 2B and 14B scales. These capabilities enable more reliable synthetic data generation, policy evaluation, and closed-loop simulation for robotics and autonomous systems. We further extend the family with [Cosmos-Transfer2.5], a control-net style framework for Sim2Real and Real2Real world translation. Despite being 3.5$\times$ smaller than [Cosmos-Transfer1], it delivers higher fidelity and robust long-horizon video generation. Together, these advances establish [Cosmos-Predict2.5] and [Cosmos-Transfer2.5] as versatile tools for scaling embodied intelligence. To accelerate research and deployment in Physical AI, we release source code, pretrained checkpoints, and curated benchmarks under the NVIDIA Open Model License at this https URL and this https URL. We hope these open resources lower the barrier to adoption and foster innovation in building the next generation of embodied intelligence. 

---
# Automatically Finding Rule-Based Neurons in OthelloGPT 

**Authors**: Aditya Singh, Zihang Wen, Srujananjali Medicherla, Adam Karvonen, Can Rager  

**Link**: [PDF](https://arxiv.org/pdf/2511.00059)  

**Abstract**: OthelloGPT, a transformer trained to predict valid moves in Othello, provides an ideal testbed for interpretability research. The model is complex enough to exhibit rich computational patterns, yet grounded in rule-based game logic that enables meaningful reverse-engineering. We present an automated approach based on decision trees to identify and interpret MLP neurons that encode rule-based game logic. Our method trains regression decision trees to map board states to neuron activations, then extracts decision paths where neurons are highly active to convert them into human-readable logical forms. These descriptions reveal highly interpretable patterns; for instance, neurons that specifically detect when diagonal moves become legal. Our findings suggest that roughly half of the neurons in layer 5 can be accurately described by compact, rule-based decision trees ($R^2 > 0.7$ for 913 of 2,048 neurons), while the remainder likely participate in more distributed or non-rule-based computations. We verify the causal relevance of patterns identified by our decision trees through targeted interventions. For a specific square, for specific game patterns, we ablate neurons corresponding to those patterns and find an approximately 5-10 fold stronger degradation in the model's ability to predict legal moves along those patterns compared to control patterns. To facilitate future work, we provide a Python tool that maps rule-based game behaviors to their implementing neurons, serving as a resource for researchers to test whether their interpretability methods recover meaningful computational structures. 

---
# MISA: Memory-Efficient LLMs Optimization with Module-wise Importance Sampling 

**Authors**: Yuxi Liu, Renjia Deng, Yutong He, Xue Wang, Tao Yao, Kun Yuan  

**Link**: [PDF](https://arxiv.org/pdf/2511.00056)  

**Abstract**: The substantial memory demands of pre-training and fine-tuning large language models (LLMs) require memory-efficient optimization algorithms. One promising approach is layer-wise optimization, which treats each transformer block as a single layer and optimizes it sequentially, while freezing the other layers to save optimizer states and activations. Although effective, these methods ignore the varying importance of the modules within each layer, leading to suboptimal performance. Moreover, layer-wise sampling provides only limited memory savings, as at least one full layer must remain active during optimization. To overcome these limitations, we propose Module-wise Importance SAmpling (MISA), a novel method that divides each layer into smaller modules and assigns importance scores to each module. MISA uses a weighted random sampling mechanism to activate modules, provably reducing gradient variance compared to layer-wise sampling. Additionally, we establish an \(\mathcal{O}(1/\sqrt{K})\) convergence rate under non-convex and stochastic conditions, where $K$ is the total number of block updates, and provide a detailed memory analysis showcasing MISA's superiority over existing baseline methods. Experiments on diverse learning tasks validate the effectiveness of MISA. Source code is available at this https URL. 

---
# Exploring Federated Learning for Thermal Urban Feature Segmentation -- A Comparison of Centralized and Decentralized Approaches 

**Authors**: Leonhard Duda, Khadijeh Alibabaei, Elena Vollmer, Leon Klug, Valentin Kozlov, Lisana Berberi, Mishal Benz, Rebekka Volk, Juan Pedro Gutiérrez Hermosillo Muriedas, Markus Götz, Judith Sáínz-Pardo Díaz, Álvaro López García, Frank Schultmann, Achim Streit  

**Link**: [PDF](https://arxiv.org/pdf/2511.00055)  

**Abstract**: Federated Learning (FL) is an approach for training a shared Machine Learning (ML) model with distributed training data and multiple participants. FL allows bypassing limitations of the traditional Centralized Machine Learning CL if data cannot be shared or stored centrally due to privacy or technical restrictions -- the participants train the model locally with their training data and do not need to share it among the other participants. This paper investigates the practical implementation and effectiveness of FL in a real-world scenario, specifically focusing on unmanned aerial vehicle (UAV)-based thermal images for common thermal feature detection in urban environments. The distributed nature of the data arises naturally and makes it suitable for FL applications, as images captured in two German cities are available. This application presents unique challenges due to non-identical distribution and feature characteristics of data captured at both locations. The study makes several key contributions by evaluating FL algorithms in real deployment scenarios rather than simulation. We compare several FL approaches with a centralized learning baseline across key performance metrics such as model accuracy, training time, communication overhead, and energy usage. This paper also explores various FL workflows, comparing client-controlled workflows and server-controlled workflows. The findings of this work serve as a valuable reference for understanding the practical application and limitations of the FL methods in segmentation tasks in UAV-based imaging. 

---
# SpatialTraceGen: High-Fidelity Traces for Efficient VLM Spatial Reasoning Distillation 

**Authors**: Gio Huh, Dhruv Sheth, Rayhan Zirvi, Frank Xiao  

**Link**: [PDF](https://arxiv.org/pdf/2511.00054)  

**Abstract**: While Vision-Language Models (VLMs) excel in many areas, they struggle with complex spatial reasoning, which requires problem decomposition and strategic tool use. Fine-tuning smaller, more deployable models offers an efficient path to strong performance, but this is hampered by a major bottleneck: the absence of high-quality, step-by-step reasoning data. To address this data-efficiency gap, we introduce SpatialTraceGen, a framework to distill the reasoning processes of a large teacher model into a high-quality dataset of multi-hop, multi-tool reasoning traces. A key innovation is our automated Verifier, which scalably ensures the fidelity of each reasoning step, providing a cost-effective alternative to manual human annotation. On the CLEVR-Humans benchmark, this verifier-guided process improves the average quality score of traces by 17\% while reducing quality variance by over 40\%. SpatialTraceGen delivers a dataset of expert traces, providing the structured, step-by-step examples of tool use necessary for effective fine-tuning and sample-efficient offline reinforcement learning. 

---
# Quadratic Direct Forecast for Training Multi-Step Time-Series Forecast Models 

**Authors**: Hao Wang, Licheng Pan, Yuan Lu, Zhichao Chen, Tianqiao Liu, Shuting He, Zhixuan Chu, Qingsong Wen, Haoxuan Li, Zhouchen Lin  

**Link**: [PDF](https://arxiv.org/pdf/2511.00053)  

**Abstract**: The design of training objective is central to training time-series forecasting models. Existing training objectives such as mean squared error mostly treat each future step as an independent, equally weighted task, which we found leading to the following two issues: (1) overlook the label autocorrelation effect among future steps, leading to biased training objective; (2) fail to set heterogeneous task weights for different forecasting tasks corresponding to varying future steps, limiting the forecasting performance. To fill this gap, we propose a novel quadratic-form weighted training objective, addressing both of the issues simultaneously. Specifically, the off-diagonal elements of the weighting matrix account for the label autocorrelation effect, whereas the non-uniform diagonals are expected to match the most preferable weights of the forecasting tasks with varying future steps. To achieve this, we propose a Quadratic Direct Forecast (QDF) learning algorithm, which trains the forecast model using the adaptively updated quadratic-form weighting matrix. Experiments show that our QDF effectively improves performance of various forecast models, achieving state-of-the-art results. Code is available at this https URL. 

---
# Calibrating and Rotating: A Unified Framework for Weight Conditioning in PEFT 

**Authors**: Da Chang, Peng Xue, Yu Li, Yongxiang Liu, Pengxiang Xu, Shixun Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2511.00051)  

**Abstract**: Parameter-Efficient Fine-Tuning (PEFT) methods are crucial for adapting large pre-trained models. Among these, LoRA is considered a foundational approach. Building on this, the influential DoRA method enhances performance by decomposing weight updates into magnitude and direction. However, its underlying mechanism remains unclear, and it introduces significant computational overhead. In this work, we first identify that DoRA's success stems from its capacity to increase the singular value entropy of the weight update matrix, which promotes a more uniform update distribution akin to full fine-tuning. We then reformulate DoRA into a mathematically equivalent and more efficient matrix form, revealing it as a learnable weight conditioning method. Based on this insight, we propose a unified framework for designing advanced PEFT methods by exploring two orthogonal dimensions: the architectural placement and the transformation type of the conditioning matrix. Within this framework, we introduce two novel methods: (1) \textbf{Pre-Diag}, which applies a diagonal conditioning matrix before the LoRA update to efficiently calibrate the pre-trained weights, thereby enhancing performance while reducing training time; and (2) \textbf{S}kewed \textbf{O}rthogonal \textbf{R}otation \textbf{A}daptation (\textbf{SORA}), which employs a parameter-efficient orthogonal rotation to perform a more powerful, norm-preserving transformation of the feature space. Extensive experiments on natural language understanding and generation tasks demonstrate that our proposed methods achieve superior performance and efficiency compared to both LoRA and DoRA. The code is available at this https URL. 

---
# FLoRA: Fused forward-backward adapters for parameter efficient fine-tuning and reducing inference-time latencies of LLMs 

**Authors**: Dhananjaya Gowda, Seoha Song, Junhyun Lee, Harshith Goka  

**Link**: [PDF](https://arxiv.org/pdf/2511.00050)  

**Abstract**: As the large language models (LLMs) grow in size each day, efficient training and fine-tuning has never been as important as nowadays. This resulted in the great interest in parameter efficient fine-tuning (PEFT), and effective methods including low-rank adapters (LoRA) has emerged. Although the various PEFT methods have been studied extensively in the recent years, the greater part of the subject remains unexplored with the huge degree of freedom. In this paper, we propose FLoRA, a family of fused forward-backward adapters (FFBA) for parameter-efficient fine-tuning of LLMs on downstream tasks. The FFBA combine ideas from the popular LoRA and parallel adapters to improve the overall fine-tuning accuracies. At the same time, latencies are minimized by fusing the forward and backward adapters into existing projection layers of the base model. Experimental results show that the proposed FFB adapters perform significantly better than the popularly used LoRA in both accuracy and latency for a similar parameter budget. 

---
# Adaptive Spatio-Temporal Graphs with Self-Supervised Pretraining for Multi-Horizon Weather Forecasting 

**Authors**: Yao Liu  

**Link**: [PDF](https://arxiv.org/pdf/2511.00049)  

**Abstract**: Accurate and robust weather forecasting remains a fundamental challenge due to the inherent spatio-temporal complexity of atmospheric systems. In this paper, we propose a novel self-supervised learning framework that leverages spatio-temporal structures to improve multi-variable weather prediction. The model integrates a graph neural network (GNN) for spatial reasoning, a self-supervised pretraining scheme for representation learning, and a spatio-temporal adaptation mechanism to enhance generalization across varying forecasting horizons. Extensive experiments on both ERA5 and MERRA-2 reanalysis datasets demonstrate that our approach achieves superior performance compared to traditional numerical weather prediction (NWP) models and recent deep learning methods. Quantitative evaluations and visual analyses in Beijing and Shanghai confirm the model's capability to capture fine-grained meteorological patterns. The proposed framework provides a scalable and label-efficient solution for future data-driven weather forecasting systems. 

---
# DynBERG: Dynamic BERT-based Graph neural network for financial fraud detection 

**Authors**: Omkar Kulkarni, Rohitash Chandra  

**Link**: [PDF](https://arxiv.org/pdf/2511.00047)  

**Abstract**: Financial fraud detection is critical for maintaining the integrity of financial systems, particularly in decentralised environments such as cryptocurrency networks. Although Graph Convolutional Networks (GCNs) are widely used for financial fraud detection, graph Transformer models such as Graph-BERT are gaining prominence due to their Transformer-based architecture, which mitigates issues such as over-smoothing. Graph-BERT is designed for static graphs and primarily evaluated on citation networks with undirected edges. However, financial transaction networks are inherently dynamic, with evolving structures and directed edges representing the flow of money. To address these challenges, we introduce DynBERG, a novel architecture that integrates Graph-BERT with a Gated Recurrent Unit (GRU) layer to capture temporal evolution over multiple time steps. Additionally, we modify the underlying algorithm to support directed edges, making DynBERG well-suited for dynamic financial transaction analysis. We evaluate our model on the Elliptic dataset, which includes Bitcoin transactions, including all transactions during a major cryptocurrency market event, the Dark Market Shutdown. By assessing DynBERG's resilience before and after this event, we analyse its ability to adapt to significant market shifts that impact transaction behaviours. Our model is benchmarked against state-of-the-art dynamic graph classification approaches, such as EvolveGCN and GCN, demonstrating superior performance, outperforming EvolveGCN before the market shutdown and surpassing GCN after the event. Additionally, an ablation study highlights the critical role of incorporating a time-series deep learning component, showcasing the effectiveness of GRU in modelling the temporal dynamics of financial transactions. 

---
# Endowing GPT-4 with a Humanoid Body: Building the Bridge Between Off-the-Shelf VLMs and the Physical World 

**Authors**: Yingzhao Jian, Zhongan Wang, Yi Yang, Hehe Fan  

**Link**: [PDF](https://arxiv.org/pdf/2511.00041)  

**Abstract**: Humanoid agents often struggle to handle flexible and diverse interactions in open environments. A common solution is to collect massive datasets to train a highly capable model, but this approach can be prohibitively expensive. In this paper, we explore an alternative solution: empowering off-the-shelf Vision-Language Models (VLMs, such as GPT-4) to control humanoid agents, thereby leveraging their strong open-world generalization to mitigate the need for extensive data collection. To this end, we present \textbf{BiBo} (\textbf{B}uilding humano\textbf{I}d agent \textbf{B}y \textbf{O}ff-the-shelf VLMs). It consists of two key components: (1) an \textbf{embodied instruction compiler}, which enables the VLM to perceive the environment and precisely translate high-level user instructions (e.g., {\small\itshape ``have a rest''}) into low-level primitive commands with control parameters (e.g., {\small\itshape ``sit casually, location: (1, 2), facing: 90$^\circ$''}); and (2) a diffusion-based \textbf{motion executor}, which generates human-like motions from these commands, while dynamically adapting to physical feedback from the environment. In this way, BiBo is capable of handling not only basic interactions but also diverse and complex motions. Experiments demonstrate that BiBo achieves an interaction task success rate of 90.2\% in open environments, and improves the precision of text-guided motion execution by 16.3\% over prior methods. The code will be made publicly available. 

---
# Semi-Supervised Preference Optimization with Limited Feedback 

**Authors**: Seonggyun Lee, Sungjun Lim, Seojin Park, Soeun Cheon, Kyungwoo Song  

**Link**: [PDF](https://arxiv.org/pdf/2511.00040)  

**Abstract**: The field of preference optimization has made outstanding contributions to the alignment of language models with human preferences. Despite these advancements, recent methods still rely heavily on substantial paired (labeled) feedback data, leading to substantial resource expenditures. To address these challenges, we study the problem of Semi-Supervised Preference Optimization (SSPO) in which the idea is to learn from both a small number of pairwise preference labels and a large pool of unpaired samples simultaneously. Our key theoretical contribution proves the existence of an optimal reward threshold capable of separating winning and losing responses with high probability, which enables a principled pseudo-labeling of unpaired data. By leveraging these pseudo-labels, SSPO effectively distills latent preferences from large-scale unpaired data, thus maintaining human alignment while drastically reducing acquisition costs. Extensive experiments across datasets validate this remarkable data efficiency; for instance, SSPO trained with Llama3-8B-Instruct on just 1% of UltraFeedback consistently surpasses strong baselines trained on 10% of UltraFeedback. 

---
# STRIDER: Navigation via Instruction-Aligned Structural Decision Space Optimization 

**Authors**: Diqi He, Xuehao Gao, Hao Li, Junwei Han, Dingwen Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2511.00033)  

**Abstract**: The Zero-shot Vision-and-Language Navigation in Continuous Environments (VLN-CE) task requires agents to navigate previously unseen 3D environments using natural language instructions, without any scene-specific training. A critical challenge in this setting lies in ensuring agents' actions align with both spatial structure and task intent over long-horizon execution. Existing methods often fail to achieve robust navigation due to a lack of structured decision-making and insufficient integration of feedback from previous actions. To address these challenges, we propose STRIDER (Instruction-Aligned Structural Decision Space Optimization), a novel framework that systematically optimizes the agent's decision space by integrating spatial layout priors and dynamic task feedback. Our approach introduces two key innovations: 1) a Structured Waypoint Generator that constrains the action space through spatial structure, and 2) a Task-Alignment Regulator that adjusts behavior based on task progress, ensuring semantic alignment throughout navigation. Extensive experiments on the R2R-CE and RxR-CE benchmarks demonstrate that STRIDER significantly outperforms strong SOTA across key metrics; in particular, it improves Success Rate (SR) from 29% to 35%, a relative gain of 20.7%. Such results highlight the importance of spatially constrained decision-making and feedback-guided execution in improving navigation fidelity for zero-shot VLN-CE. 

---
# From Uniform to Adaptive: General Skip-Block Mechanisms for Efficient PDE Neural Operators 

**Authors**: Lei Liu, Zhongyi Yu, Hong Wang, Huanshuo Dong, Haiyang Xin, Hongwei Zhao, Bin Li  

**Link**: [PDF](https://arxiv.org/pdf/2511.00032)  

**Abstract**: In recent years, Neural Operators(NO) have gradually emerged as a popular approach for solving Partial Differential Equations (PDEs). However, their application to large-scale engineering tasks suffers from significant computational overhead. And the fact that current models impose a uniform computational cost while physical fields exhibit vastly different complexities constitutes a fundamental mismatch, which is the root of this inefficiency. For instance, in turbulence flows, intricate vortex regions require deeper network processing compared to stable flows. To address this, we introduce a framework: Skip-Block Routing (SBR), a general framework designed for Transformer-based neural operators, capable of being integrated into their multi-layer architectures. First, SBR uses a routing mechanism to learn the complexity and ranking of tokens, which is then applied during inference. Then, in later layers, it decides how many tokens are passed forward based on this ranking. This way, the model focuses more processing capacity on the tokens that are more complex. Experiments demonstrate that SBR is a general framework that seamlessly integrates into various neural operators. Our method reduces computational cost by approximately 50% in terms of Floating Point Operations (FLOPs), while still delivering up to 2x faster inference without sacrificing accuracy. 

---
# Probing Knowledge Holes in Unlearned LLMs 

**Authors**: Myeongseob Ko, Hoang Anh Just, Charles Fleming, Ming Jin, Ruoxi Jia  

**Link**: [PDF](https://arxiv.org/pdf/2511.00030)  

**Abstract**: Machine unlearning has emerged as a prevalent technical solution for selectively removing unwanted knowledge absorbed during pre-training, without requiring full retraining. While recent unlearning techniques can effectively remove undesirable content without severely compromising performance on standard benchmarks, we find that they may inadvertently create ``knowledge holes'' -- unintended losses of benign knowledge that standard benchmarks fail to capture. To probe where unlearned models reveal knowledge holes, we propose a test case generation framework that explores both immediate neighbors of unlearned content and broader areas of potential failures. Our evaluation demonstrates significant hidden costs of unlearning: up to 98.7\% of the test cases yield irrelevant or nonsensical responses from unlearned models, despite being answerable by the pretrained model. These findings necessitate rethinking the conventional approach to evaluating knowledge preservation in unlearning, moving beyond standard, static benchmarks. 

---
# Feature-Guided SAE Steering for Refusal-Rate Control using Contrasting Prompts 

**Authors**: Samaksh Bhargav, Zining Zhu  

**Link**: [PDF](https://arxiv.org/pdf/2511.00029)  

**Abstract**: Large Language Model (LLM) deployment requires guiding the LLM to recognize and not answer unsafe prompts while complying with safe prompts. Previous methods for achieving this require adjusting model weights along with other expensive procedures. While recent advances in Sparse Autoencoders (SAEs) have enabled interpretable feature extraction from LLMs, existing approaches lack systematic feature selection methods and principled evaluation of safety-utility tradeoffs. We explored using different steering features and steering strengths using Sparse Auto Encoders (SAEs) to provide a solution. Using an accurate and innovative contrasting prompt method with the AI-Generated Prompts Dataset from teknium/OpenHermes-2p5-Mistral-7B and Air Bench eu-dataset to efficiently choose the best features in the model to steer, we tested this method on Llama-3 8B. We conclude that using this method, our approach achieves an 18.9% improvement in safety performance while simultaneously increasing utility by 11.1%, demonstrating that targeted SAE steering can overcome traditional safety-utility tradeoffs when optimal features are identified through principled selection methods. 

---
# Mutual Information guided Visual Contrastive Learning 

**Authors**: Hanyang Chen, Yanchao Yang  

**Link**: [PDF](https://arxiv.org/pdf/2511.00028)  

**Abstract**: Representation learning methods utilizing the InfoNCE loss have demonstrated considerable capacity in reducing human annotation effort by training invariant neural feature extractors. Although different variants of the training objective adhere to the information maximization principle between the data and learned features, data selection and augmentation still rely on human hypotheses or engineering, which may be suboptimal. For instance, data augmentation in contrastive learning primarily focuses on color jittering, aiming to emulate real-world illumination changes. In this work, we investigate the potential of selecting training data based on their mutual information computed from real-world distributions, which, in principle, should endow the learned features with better generalization when applied in open environments. Specifically, we consider patches attached to scenes that exhibit high mutual information under natural perturbations, such as color changes and motion, as positive samples for learning with contrastive loss. We evaluate the proposed mutual-information-informed data augmentation method on several benchmarks across multiple state-of-the-art representation learning frameworks, demonstrating its effectiveness and establishing it as a promising direction for future research. 

---
# Position Paper: If Innovation in AI Systematically Violates Fundamental Rights, Is It Innovation at All? 

**Authors**: Josu Eguiluz Castañeira, Axel Brando, Migle Laukyte, Marc Serra-Vidal  

**Link**: [PDF](https://arxiv.org/pdf/2511.00027)  

**Abstract**: Artificial intelligence (AI) now permeates critical infrastructures and decision-making systems where failures produce social, economic, and democratic harm. This position paper challenges the entrenched belief that regulation and innovation are opposites. As evidenced by analogies from aviation, pharmaceuticals, and welfare systems and recent cases of synthetic misinformation, bias and unaccountable decision-making, the absence of well-designed regulation has already created immeasurable damage. Regulation, when thoughtful and adaptive, is not a brake on innovation--it is its foundation. The present position paper examines the EU AI Act as a model of risk-based, responsibility-driven regulation that addresses the Collingridge Dilemma: acting early enough to prevent harm, yet flexibly enough to sustain innovation. Its adaptive mechanisms--regulatory sandboxes, small and medium enterprises (SMEs) support, real-world testing, fundamental rights impact assessment (FRIA) -- demonstrate how regulation can accelerate responsibly, rather than delay, technological progress. The position paper summarises how governance tools transform perceived burdens into tangible advantages: legal certainty, consumer trust, and ethical competitiveness. Ultimately, the paper reframes progress: innovation and regulation advance together. By embedding transparency, impact assessments, accountability, and AI literacy into design and deployment, the EU framework defines what responsible innovation truly means--technological ambition disciplined by democratic values and fundamental rights. 

---
# Chitchat with AI: Understand the supply chain carbon disclosure of companies worldwide through Large Language Model 

**Authors**: Haotian Hang, Yueyang Shen, Vicky Zhu, Jose Cruz, Michelle Li  

**Link**: [PDF](https://arxiv.org/pdf/2511.00024)  

**Abstract**: In the context of global sustainability mandates, corporate carbon disclosure has emerged as a critical mechanism for aligning business strategy with environmental responsibility. The Carbon Disclosure Project (CDP) hosts the world's largest longitudinal dataset of climate-related survey responses, combining structured indicators with open-ended narratives, but the heterogeneity and free-form nature of these disclosures present significant analytical challenges for benchmarking, compliance monitoring, and investment screening. This paper proposes a novel decision-support framework that leverages large language models (LLMs) to assess corporate climate disclosure quality at scale. It develops a master rubric that harmonizes narrative scoring across 11 years of CDP data (2010-2020), enabling cross-sector and cross-country benchmarking. By integrating rubric-guided scoring with percentile-based normalization, our method identifies temporal trends, strategic alignment patterns, and inconsistencies in disclosure across industries and regions. Results reveal that sectors such as technology and countries like Germany consistently demonstrate higher rubric alignment, while others exhibit volatility or superficial engagement, offering insights that inform key decision-making processes for investors, regulators, and corporate environmental, social, and governance (ESG) strategists. The proposed LLM-based approach transforms unstructured disclosures into quantifiable, interpretable, comparable, and actionable intelligence, advancing the capabilities of AI-enabled decision support systems (DSSs) in the domain of climate governance. 

---
# Deep Learning Models for Coral Bleaching Classification in Multi-Condition Underwater Image Datasets 

**Authors**: Julio Jerison E. Macrohon, Gordon Hung  

**Link**: [PDF](https://arxiv.org/pdf/2511.00021)  

**Abstract**: Coral reefs support numerous marine organisms and are an important source of coastal protection from storms and floods, representing a major part of marine ecosystems. However coral reefs face increasing threats from pollution, ocean acidification, and sea temperature anomalies, making efficient protection and monitoring heavily urgent. Therefore, this study presents a novel machine-learning-based coral bleaching classification system based on a diverse global dataset with samples of healthy and bleached corals under varying environmental conditions, including deep seas, marshes, and coastal zones. We benchmarked and compared three state-of-the-art models: Residual Neural Network (ResNet), Vision Transformer (ViT), and Convolutional Neural Network (CNN). After comprehensive hyperparameter tuning, the CNN model achieved the highest accuracy of 88%, outperforming existing benchmarks. Our findings offer important insights into autonomous coral monitoring and present a comprehensive analysis of the most widely used computer vision models. 

---
# Sorting by Strip Swaps is NP-Hard 

**Authors**: Swapnoneel Roy, Asai Asaithambi, Debajyoti Mukhopadhyay  

**Link**: [PDF](https://arxiv.org/pdf/2511.00015)  

**Abstract**: We show that \emph{Sorting by Strip Swaps} (SbSS) is NP-hard by a polynomial reduction of \emph{Block Sorting}. The key idea is a local gadget, a \emph{cage}, that replaces every decreasing adjacency $(a_i,a_{i+1})$ by a guarded triple $a_i,m_i,a_{i+1}$ enclosed by guards $L_i,U_i$, so the only decreasing adjacencies are the two inside the cage. Small \emph{hinge} gadgets couple adjacent cages that share an element and enforce that a strip swap that removes exactly two adjacencies corresponds bijectively to a block move that removes exactly one decreasing adjacency in the source permutation. This yields a clean equivalence between exact SbSS schedules and perfect block schedules, establishing NP-hardness. 

---
# Generative human motion mimicking through feature extraction in denoising diffusion settings 

**Authors**: Alexander Okupnik, Johannes Schneider, Kyriakos Flouris  

**Link**: [PDF](https://arxiv.org/pdf/2511.00011)  

**Abstract**: Recent success with large language models has sparked a new wave of verbal human-AI interaction. While such models support users in a variety of creative tasks, they lack the embodied nature of human interaction. Dance, as a primal form of human expression, is predestined to complement this experience. To explore creative human-AI interaction exemplified by dance, we build an interactive model based on motion capture (MoCap) data. It generates an artificial other by partially mimicking and also "creatively" enhancing an incoming sequence of movement data. It is the first model, which leverages single-person motion data and high level features in order to do so and, thus, it does not rely on low level human-human interaction data. It combines ideas of two diffusion models, motion inpainting, and motion style transfer to generate movement representations that are both temporally coherent and responsive to a chosen movement reference. The success of the model is demonstrated by quantitatively assessing the convergence of the feature distribution of the generated samples and the test set which serves as simulating the human performer. We show that our generations are first steps to creative dancing with AI as they are both diverse showing various deviations from the human partner while appearing realistic. 

---
# Multimodal Learning with Augmentation Techniques for Natural Disaster Assessment 

**Authors**: Adrian-Dinu Urse, Dumitru-Clementin Cercel, Florin Pop  

**Link**: [PDF](https://arxiv.org/pdf/2511.00004)  

**Abstract**: Natural disaster assessment relies on accurate and rapid access to information, with social media emerging as a valuable real-time source. However, existing datasets suffer from class imbalance and limited samples, making effective model development a challenging task. This paper explores augmentation techniques to address these issues on the CrisisMMD multimodal dataset. For visual data, we apply diffusion-based methods, namely Real Guidance and DiffuseMix. For text data, we explore back-translation, paraphrasing with transformers, and image caption-based augmentation. We evaluated these across unimodal, multimodal, and multi-view learning setups. Results show that selected augmentations improve classification performance, particularly for underrepresented classes, while multi-view learning introduces potential but requires further refinement. This study highlights effective augmentation strategies for building more robust disaster assessment systems. 

---
# VRScout: Towards Real-Time, Autonomous Testing of Virtual Reality Games 

**Authors**: Yurun Wu, Yousong Sun, Burkhard Wunsche, Jia Wang, Elliott Wen  

**Link**: [PDF](https://arxiv.org/pdf/2511.00002)  

**Abstract**: Virtual Reality (VR) has rapidly become a mainstream platform for gaming and interactive experiences, yet ensuring the quality, safety, and appropriateness of VR content remains a pressing challenge. Traditional human-based quality assurance is labor-intensive and cannot scale with the industry's rapid growth. While automated testing has been applied to traditional 2D and 3D games, extending it to VR introduces unique difficulties due to high-dimensional sensory inputs and strict real-time performance requirements. We present VRScout, a deep learning-based agent capable of autonomously navigating VR environments and interacting with virtual objects in a human-like and real-time manner. VRScout learns from human demonstrations using an enhanced Action Chunking Transformer that predicts multi-step action sequences. This enables our agent to capture higher-level strategies and generalize across diverse environments. To balance responsiveness and precision, we introduce a dynamically adjustable sliding horizon that adapts the agent's temporal context at runtime. We evaluate VRScout on commercial VR titles and show that it achieves expert-level performance with only limited training data, while maintaining real-time inference at 60 FPS on consumer-grade hardware. These results position VRScout as a practical and scalable framework for automated VR game testing, with direct applications in both quality assurance and safety auditing. 

---
# A Two Level Neural Approach Combining Off-Chip Prediction with Adaptive Prefetch Filtering 

**Authors**: Alexandre Valentin Jamet, Georgios Vavouliotis, Daniel A. Jiménez, Lluc Alvarez, Marc Casas  

**Link**: [PDF](https://arxiv.org/pdf/2403.15181)  

**Abstract**: To alleviate the performance and energy overheads of contemporary applications with large data footprints, we propose the Two Level Perceptron (TLP) predictor, a neural mechanism that effectively combines predicting whether an access will be off-chip with adaptive prefetch filtering at the first-level data cache (L1D). TLP is composed of two connected microarchitectural perceptron predictors, named First Level Predictor (FLP) and Second Level Predictor (SLP). FLP performs accurate off-chip prediction by using several program features based on virtual addresses and a novel selective delay component. The novelty of SLP relies on leveraging off-chip prediction to drive L1D prefetch filtering by using physical addresses and the FLP prediction as features. TLP constitutes the first hardware proposal targeting both off-chip prediction and prefetch filtering using a multi-level perceptron hardware approach. TLP only requires 7KB of storage. To demonstrate the benefits of TLP we compare its performance with state-of-the-art approaches using off-chip prediction and prefetch filtering on a wide range of single-core and multi-core workloads. Our experiments show that TLP reduces the average DRAM transactions by 30.7% and 17.7%, as compared to a baseline using state-of-the-art cache prefetchers but no off-chip prediction mechanism, across the single-core and multi-core workloads, respectively, while recent work significantly increases DRAM transactions. As a result, TLP achieves geometric mean performance speedups of 6.2% and 11.8% across single-core and multi-core workloads, respectively. In addition, our evaluation demonstrates that TLP is effective independently of the L1D prefetching logic. 

---
