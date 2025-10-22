# Decoding Funded Research: Comparative Analysis of Topic Models and Uncovering the Effect of Gender and Geographic Location 

**Authors**: Shirin Tavakoli Kafiabad, Andrea Schiffauerova, Ashkan Ebadi  

**Link**: [PDF](https://arxiv.org/pdf/2510.18803)  

**Abstract**: Optimizing national scientific investment requires a clear understanding of evolving research trends and the demographic and geographical forces shaping them, particularly in light of commitments to equity, diversity, and inclusion. This study addresses this need by analyzing 18 years (2005-2022) of research proposals funded by the Natural Sciences and Engineering Research Council of Canada (NSERC). We conducted a comprehensive comparative evaluation of three topic modelling approaches: Latent Dirichlet Allocation (LDA), Structural Topic Modelling (STM), and BERTopic. We also introduced a novel algorithm, named COFFEE, designed to enable robust covariate effect estimation for BERTopic. This advancement addresses a significant gap, as BERTopic lacks a native function for covariate analysis, unlike the probabilistic STM. Our findings highlight that while all models effectively delineate core scientific domains, BERTopic outperformed by consistently identifying more granular, coherent, and emergent themes, such as the rapid expansion of artificial intelligence. Additionally, the covariate analysis, powered by COFFEE, confirmed distinct provincial research specializations and revealed consistent gender-based thematic patterns across various scientific disciplines. These insights offer a robust empirical foundation for funding organizations to formulate more equitable and impactful funding strategies, thereby enhancing the effectiveness of the scientific ecosystem. 

---
# Seg the HAB: Language-Guided Geospatial Algae Bloom Reasoning and Segmentation 

**Authors**: Patterson Hsieh, Jerry Yeh, Mao-Chi He, Wen-Han Hsieh, Elvis Hsieh  

**Link**: [PDF](https://arxiv.org/pdf/2510.18751)  

**Abstract**: Climate change is intensifying the occurrence of harmful algal bloom (HAB), particularly cyanobacteria, which threaten aquatic ecosystems and human health through oxygen depletion, toxin release, and disruption of marine biodiversity. Traditional monitoring approaches, such as manual water sampling, remain labor-intensive and limited in spatial and temporal coverage. Recent advances in vision-language models (VLMs) for remote sensing have shown potential for scalable AI-driven solutions, yet challenges remain in reasoning over imagery and quantifying bloom severity. In this work, we introduce ALGae Observation and Segmentation (ALGOS), a segmentation-and-reasoning system for HAB monitoring that combines remote sensing image understanding with severity estimation. Our approach integrates GeoSAM-assisted human evaluation for high-quality segmentation mask curation and fine-tunes vision language model on severity prediction using the Cyanobacteria Aggregated Manual Labels (CAML) from NASA. Experiments demonstrate that ALGOS achieves robust performance on both segmentation and severity-level estimation, paving the way toward practical and automated cyanobacterial monitoring systems. 

---
# Sherlock Your Queries: Learning to Ask the Right Questions for Dialogue-Based Retrieval 

**Authors**: Dong Yun, Marco Schouten, Dim Papadopoulos  

**Link**: [PDF](https://arxiv.org/pdf/2510.18659)  

**Abstract**: User queries in information retrieval are often ambiguous, making it challenging for systems to identify a user's target from a single query. While recent dialogue-based interactive retrieval systems can clarify user intent, they are inefficient as they often lack an explicit strategy to ask the most informative questions. To address this limitation, we propose SherlockLLM, a dialogue-driven retrieval framework that learns an optimal questioning strategy via Reinforcement Learning (RL) and avoids the need for large-scale annotated dialogue data. In our framework, an agent is trained to generate a sequence of binary questions to efficiently narrow down the search space. To validate our approach, we introduce a benchmark with both structured and unstructured tasks. Experimental results show that SherlockLLM is a robust and efficient solution. On the structured tasks, its performance matches strong baselines and approaches the theoretical optimal defined by binary search. On the challenging unstructured task, our agent significantly outperforms these baselines, showcasing its ability to learn a highly effective information-seeking dialogue policy. 

---
# Query Decomposition for RAG: Balancing Exploration-Exploitation 

**Authors**: Roxana Petcu, Kenton Murray, Daniel Khashabi, Evangelos Kanoulas, Maarten de Rijke, Dawn Lawrie, Kevin Duh  

**Link**: [PDF](https://arxiv.org/pdf/2510.18633)  

**Abstract**: Retrieval-augmented generation (RAG) systems address complex user requests by decomposing them into subqueries, retrieving potentially relevant documents for each, and then aggregating them to generate an answer. Efficiently selecting informative documents requires balancing a key trade-off: (i) retrieving broadly enough to capture all the relevant material, and (ii) limiting retrieval to avoid excessive noise and computational cost. We formulate query decomposition and document retrieval in an exploitation-exploration setting, where retrieving one document at a time builds a belief about the utility of a given sub-query and informs the decision to continue exploiting or exploring an alternative. We experiment with a variety of bandit learning methods and demonstrate their effectiveness in dynamically selecting the most informative sub-queries. Our main finding is that estimating document relevance using rank information and human judgments yields a 35% gain in document-level precision, 15% increase in {\alpha}-nDCG, and better performance on the downstream task of long-form generation. 

---
# Comparative Expressivity for Structured Argumentation Frameworks with Uncertain Rules and Premises 

**Authors**: Carlo Proietti, Antonio Yuste-Ginel  

**Link**: [PDF](https://arxiv.org/pdf/2510.18631)  

**Abstract**: Modelling qualitative uncertainty in formal argumentation is essential both for practical applications and theoretical understanding. Yet, most of the existing works focus on \textit{abstract} models for arguing with uncertainty. Following a recent trend in the literature, we tackle the open question of studying plausible instantiations of these abstract models. To do so, we ground the uncertainty of arguments in their components, structured within rules and premises. Our main technical contributions are: i) the introduction of a notion of expressivity that can handle abstract and structured formalisms, and ii) the presentation of both negative and positive expressivity results, comparing the expressivity of abstract and structured models of argumentation with uncertainty. These results affect incomplete abstract argumentation frameworks, and their extension with dependencies, on the abstract side, and ASPIC+, on the structured side. 

---
# Leveraging Association Rules for Better Predictions and Better Explanations 

**Authors**: Gilles Audemard, Sylvie Coste-Marquis, Pierre Marquis, Mehdi Sabiri, Nicolas Szczepanski  

**Link**: [PDF](https://arxiv.org/pdf/2510.18628)  

**Abstract**: We present a new approach to classification that combines data and knowledge. In this approach, data mining is used to derive association rules (possibly with negations) from data. Those rules are leveraged to increase the predictive performance of tree-based models (decision trees and random forests) used for a classification task. They are also used to improve the corresponding explanation task through the generation of abductive explanations that are more general than those derivable without taking such rules into account. Experiments show that for the two tree-based models under consideration, benefits can be offered by the approach in terms of predictive performance and in terms of explanation sizes. 

---
# VAR: Visual Attention Reasoning via Structured Search and Backtracking 

**Authors**: Wei Cai, Jian Zhao, Yuchen Yuan, Tianle Zhang, Ming Zhu, Haichuan Tang, Chi Zhang, Xuelong Li  

**Link**: [PDF](https://arxiv.org/pdf/2510.18619)  

**Abstract**: Multimodal Large Language Models (MLLMs), despite their advances, are hindered by their high hallucination tendency and heavy reliance on brittle, linear reasoning processes, leading to failures in complex tasks. To address these limitations, we introduce Visual Attention Reasoning (VAR), a novel framework that recasts grounded reasoning as a structured search over a reasoning trajectory space. VAR decomposes the reasoning process into two key stages: traceable evidence grounding and search-based chain-of-thought (CoT) generation, which incorporates a backtracking mechanism for self-correction. The search is guided by a multi-faceted reward function with semantic and geometric self-verification components, which penalize outputs that are not faithfully grounded in the visual input. We provide a theoretical analysis for our search strategy, validating its capability to find the correct solution with high probability. Experimental results show that our 7B model, VAR-7B, sets a new state-of-the-art on a comprehensive suite of hallucination and safety benchmarks, significantly outperforming existing open-source models and demonstrating competitive performance against leading proprietary systems. 

---
# QuantEvolve: Automating Quantitative Strategy Discovery through Multi-Agent Evolutionary Framework 

**Authors**: Junhyeog Yun, Hyoun Jun Lee, Insu Jeon  

**Link**: [PDF](https://arxiv.org/pdf/2510.18569)  

**Abstract**: Automating quantitative trading strategy development in dynamic markets is challenging, especially with increasing demand for personalized investment solutions. Existing methods often fail to explore the vast strategy space while preserving the diversity essential for robust performance across changing market conditions. We present QuantEvolve, an evolutionary framework that combines quality-diversity optimization with hypothesis-driven strategy generation. QuantEvolve employs a feature map aligned with investor preferences, such as strategy type, risk profile, turnover, and return characteristics, to maintain a diverse set of effective strategies. It also integrates a hypothesis-driven multi-agent system to systematically explore the strategy space through iterative generation and evaluation. This approach produces diverse, sophisticated strategies that adapt to both market regime shifts and individual investment needs. Empirical results show that QuantEvolve outperforms conventional baselines, validating its effectiveness. We release a dataset of evolved strategies to support future research. 

---
# Extracting alignment data in open models 

**Authors**: Federico Barbero, Xiangming Gu, Christopher A. Choquette-Choo, Chawin Sitawarin, Matthew Jagielski, Itay Yona, Petar Veličković, Ilia Shumailov, Jamie Hayes  

**Link**: [PDF](https://arxiv.org/pdf/2510.18554)  

**Abstract**: In this work, we show that it is possible to extract significant amounts of alignment training data from a post-trained model -- useful to steer the model to improve certain capabilities such as long-context reasoning, safety, instruction following, and maths. While the majority of related work on memorisation has focused on measuring success of training data extraction through string matching, we argue that embedding models are better suited for our specific goals. Distances measured through a high quality embedding model can identify semantic similarities between strings that a different metric such as edit distance will struggle to capture. In fact, in our investigation, approximate string matching would have severely undercounted (by a conservative estimate of $10\times$) the amount of data that can be extracted due to trivial artifacts that deflate the metric. Interestingly, we find that models readily regurgitate training data that was used in post-training phases such as SFT or RL. We show that this data can be then used to train a base model, recovering a meaningful amount of the original performance. We believe our work exposes a possibly overlooked risk towards extracting alignment data. Finally, our work opens up an interesting discussion on the downstream effects of distillation practices: since models seem to be regurgitating aspects of their training set, distillation can therefore be thought of as indirectly training on the model's original dataset. 

---
# SOCIA-Nabla: Textual Gradient Meets Multi-Agent Orchestration for Automated Simulator Generation 

**Authors**: Yuncheng Hua, Sion Weatherhead, Mehdi Jafari, Hao Xue, Flora D. Salim  

**Link**: [PDF](https://arxiv.org/pdf/2510.18551)  

**Abstract**: In this paper, we present SOCIA-Nabla, an end-to-end, agentic framework that treats simulator construction asinstance optimization over code within a textual computation graph. Specialized LLM-driven agents are embedded as graph nodes, and a workflow manager executes a loss-driven loop: code synthesis -> execution -> evaluation -> code repair. The optimizer performs Textual-Gradient Descent (TGD), while human-in-the-loop interaction is reserved for task-spec confirmation, minimizing expert effort and keeping the code itself as the trainable object. Across three CPS tasks, i.e., User Modeling, Mask Adoption, and Personal Mobility, SOCIA-Nabla attains state-of-the-art overall accuracy. By unifying multi-agent orchestration with a loss-aligned optimization view, SOCIA-Nabla converts brittle prompt pipelines into reproducible, constraint-aware simulator code generation that scales across domains and simulation granularities. This work is under review, and we will release the code soon. 

---
# Physics-guided Emulators Reveal Resilience and Fragility under Operational Latencies and Outages 

**Authors**: Sarth Dubey, Subimal Ghosh, Udit Bhatia  

**Link**: [PDF](https://arxiv.org/pdf/2510.18535)  

**Abstract**: Reliable hydrologic and flood forecasting requires models that remain stable when input data are delayed, missing, or inconsistent. However, most advances in rainfall-runoff prediction have been evaluated under ideal data conditions, emphasizing accuracy rather than operational resilience. Here, we develop an operationally ready emulator of the Global Flood Awareness System (GloFAS) that couples long- and short-term memory networks with a relaxed water-balance constraint to preserve physical coherence. Five architectures span a continuum of information availability: from complete historical and forecast forcings to scenarios with data latency and outages, allowing systematic evaluation of robustness. Trained in minimally managed catchments across the United States and tested in more than 5,000 basins, including heavily regulated rivers in India, the emulator reproduces the hydrological core of GloFAS and degrades smoothly as information quality declines. Transfer across contrasting hydroclimatic and management regimes yields reduced yet physically consistent performance, defining the limits of generalization under data scarcity and human influence. The framework establishes operational robustness as a measurable property of hydrological machine learning and advances the design of reliable real-time forecasting systems. 

---
# Counterfactual Reasoning for Steerable Pluralistic Value Alignment of Large Language Models 

**Authors**: Hanze Guo, Jing Yao, Xiao Zhou, Xiaoyuan Yi, Xing Xie  

**Link**: [PDF](https://arxiv.org/pdf/2510.18526)  

**Abstract**: As large language models (LLMs) become increasingly integrated into applications serving users across diverse cultures, communities and demographics, it is critical to align LLMs with pluralistic human values beyond average principles (e.g., HHH). In psychological and social value theories such as Schwartz's Value Theory, pluralistic values are represented by multiple value dimensions paired with various priorities. However, existing methods encounter two challenges when aligning with such fine-grained value objectives: 1) they often treat multiple values as independent and equally important, ignoring their interdependence and relative priorities (value complexity); 2) they struggle to precisely control nuanced value priorities, especially those underrepresented ones (value steerability). To handle these challenges, we propose COUPLE, a COUnterfactual reasoning framework for PLuralistic valuE alignment. It introduces a structural causal model (SCM) to feature complex interdependency and prioritization among features, as well as the causal relationship between high-level value dimensions and behaviors. Moreover, it applies counterfactual reasoning to generate outputs aligned with any desired value objectives. Benefitting from explicit causal modeling, COUPLE also provides better interpretability. We evaluate COUPLE on two datasets with different value systems and demonstrate that COUPLE advances other baselines across diverse types of value objectives. 

---
# Crucible: Quantifying the Potential of Control Algorithms through LLM Agents 

**Authors**: Lianchen Jia, Chaoyang Li, Qian Houde, Tianchi Huang, Jiangchuan Liu, Lifeng Sun  

**Link**: [PDF](https://arxiv.org/pdf/2510.18491)  

**Abstract**: Control algorithms in production environments typically require domain experts to tune their parameters and logic for specific scenarios. However, existing research predominantly focuses on algorithmic performance under ideal or default configurations, overlooking the critical aspect of Tuning Potential. To bridge this gap, we introduce Crucible, an agent that employs an LLM-driven, multi-level expert simulation to turn algorithms and defines a formalized metric to quantitatively evaluate their Tuning Potential. We demonstrate Crucible's effectiveness across a wide spectrum of case studies, from classic control tasks to complex computer systems, and validate its findings in a real-world deployment. Our experimental results reveal that Crucible systematically quantifies the tunable space across different algorithms. Furthermore, Crucible provides a new dimension for algorithm analysis and design, which ultimately leads to performance improvements. Our code is available at this https URL. 

---
# AndroidControl-Curated: Revealing the True Potential of GUI Agents through Benchmark Purification 

**Authors**: Ho Fai Leung, Xiaoyan Xi, Fei Zuo  

**Link**: [PDF](https://arxiv.org/pdf/2510.18488)  

**Abstract**: On-device virtual assistants like Siri and Google Assistant are increasingly pivotal, yet their capabilities are hamstrung by a reliance on rigid, developer-dependent APIs. GUI agents offer a powerful, API-independent alternative, but their adoption is hindered by the perception of poor performance, as even the best models (e.g. Qwen3-VL-235B) scores are capped at around 60% on benchmarks like AndroidControl, far from viability for real-world use. Our research reveals that issue lies not only with the models but with the benchmarks themselves. We identified notable shortcomings in AndroidControl, including ambiguities and factual errors, which systematically underrates agent capabilities. To address this critical oversight, we enhanced AndroidControl into AndroidControl-Curated, a refined version of the benchmark improved through a rigorous purification pipeline. On this enhanced benchmark, state-of-the-art models achieve success rates nearing 75% on complex tasks (15% improvement), reflecting that on-device GUI agents are actually closer to practical deployment than previously thought. We introduce our new SOTA model, Magma-R1- 3B, post-trained on just 2.4k curated samples using 60 hours of an H20 GPU (approximately $60). Despite being 200 times smaller in parameters, this model delivers performance comparable to Qwen3- VL-235B. We release both AndroidControl-Curated benchmark and Magma-R1 model to the research community, encouraging adoption of this enhanced benchmark to better reflect model capabilities and accelerate the development of robust, on-device virtual assistants. 

---
# StarBench: A Turn-Based RPG Benchmark for Agentic Multimodal Decision-Making and Information Seeking 

**Authors**: Haoran Zhang, Chenhao Zhu, Sicong Guo, Hanzhe Guo, Haiming Li, Donglin Yu  

**Link**: [PDF](https://arxiv.org/pdf/2510.18483)  

**Abstract**: Human players do more than press buttons: they ground what they see on screen into precise keyboard-mouse actions and, when stuck, they seek information before trying again. We ask whether current vision-language models (VLMs) can do the same. Despite encouraging results under simplified control or tool scaffolds, human-like play in a real client - mapping raw screenshots to temporally coherent low-level actions while deciding when to ask for guidance - remains an open challenge. We introduce StarBench, a turn-based RPG benchmark derived from Honkai: Star Rail that targets these two human-like competencies: multimodal decision-making from pixels to actions and agentic information seeking. StarBench standardizes evaluation across eight combat tasks and two regimes with shared tasks and metrics: (i) direct control, where agents receive only screenshots and must emit low-level primitives (click and keypress) with no semantic hints; and (ii) tool-assisted control, where higher-level intents can be mapped to primitives by detectors and OCR outputs provide optional textualized observations to ease UI grounding. To mirror human practice, StarBench also includes an ask-or-act diagnostic that measures whether and when agents choose to request brief guidance before proceeding, and how that choice affects subsequent performance. We report reference baselines for contemporary VLMs and a human reference. Results expose sizable gaps in perception-to-control fidelity in the direct regime, while showing that judicious information seeking correlates with improved success, establishing StarBench as a reproducible yardstick for agentic information seeking and multimodal decision-making in real-client play. 

---
# LAFA: Agentic LLM-Driven Federated Analytics over Decentralized Data Sources 

**Authors**: Haichao Ji, Zibo Wang, Yifei Zhu, Meng han, Dan Wang, Zhu Han  

**Link**: [PDF](https://arxiv.org/pdf/2510.18477)  

**Abstract**: Large Language Models (LLMs) have shown great promise in automating data analytics tasks by interpreting natural language queries and generating multi-operation execution plans. However, existing LLM-agent-based analytics frameworks operate under the assumption of centralized data access, offering little to no privacy protection. In contrast, federated analytics (FA) enables privacy-preserving computation across distributed data sources, but lacks support for natural language input and requires structured, machine-readable queries. In this work, we present LAFA, the first system that integrates LLM-agent-based data analytics with FA. LAFA introduces a hierarchical multi-agent architecture that accepts natural language queries and transforms them into optimized, executable FA workflows. A coarse-grained planner first decomposes complex queries into sub-queries, while a fine-grained planner maps each subquery into a Directed Acyclic Graph of FA operations using prior structural knowledge. To improve execution efficiency, an optimizer agent rewrites and merges multiple DAGs, eliminating redundant operations and minimizing computational and communicational overhead. Our experiments demonstrate that LAFA consistently outperforms baseline prompting strategies by achieving higher execution plan success rates and reducing resource-intensive FA operations by a substantial margin. This work establishes a practical foundation for privacy-preserving, LLM-driven analytics that supports natural language input in the FA setting. 

---
# Probabilistic Modeling of Intentions in Socially Intelligent LLM Agents 

**Authors**: Feifan Xia, Yuyang Fang, Defang Li, Yantong Xie, Weikang Li, Yang Li, Deguo Xia, Jizhou Huang  

**Link**: [PDF](https://arxiv.org/pdf/2510.18476)  

**Abstract**: We present a probabilistic intent modeling framework for large language model (LLM) agents in multi-turn social dialogue. The framework maintains a belief distribution over a partner's latent intentions, initialized from contextual priors and dynamically updated through likelihood estimation after each utterance. The evolving distribution provides additional contextual grounding for the policy, enabling adaptive dialogue strategies under uncertainty. Preliminary experiments in the SOTOPIA environment show consistent improvements: the proposed framework increases the Overall score by 9.0% on SOTOPIA-All and 4.1% on SOTOPIA-Hard compared with the Qwen2.5-7B baseline, and slightly surpasses an oracle agent that directly observes partner intentions. These early results suggest that probabilistic intent modeling can contribute to the development of socially intelligent LLM agents. 

---
# CircuitSeer: Mining High-Quality Data by Probing Mathematical Reasoning Circuits in LLMs 

**Authors**: Shaobo Wang, Yongliang Miao, Yuancheng Liu, and Qianli Ma, Ning Liao, Linfeng Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2510.18470)  

**Abstract**: Large language models (LLMs) have demonstrated impressive reasoning capabilities, but scaling their performance often relies on massive reasoning datasets that are computationally expensive to train on. Existing data selection methods aim to curate smaller, high-quality subsets but often rely on costly external models or opaque heuristics. In this work, we shift the focus from external heuristics to the model's internal mechanisms. We find that complex reasoning tasks consistently activate a sparse, specialized subset of attention heads, forming core reasoning circuits. Building on this insight, we propose CircuitSeer, a novel data selection method that quantifies the reasoning complexity of data by measuring its influence on these crucial circuits. Extensive experiments on 4 models and 9 datasets demonstrate CircuitSeer's superiority. Notably, fine-tuning Qwen2.5-Math-7B on just 10% of data selected by our method achieves a 1.4-point gain in average Pass@1 over training on the full dataset, highlighting its efficiency and effectiveness. 

---
# PlanU: Large Language Model Decision Making through Planning under Uncertainty 

**Authors**: Ziwei Deng, Mian Deng, Chenjing Liang, Zeming Gao, Chennan Ma, Chenxing Lin, Haipeng Zhang, Songzhu Mei, Cheng Wang, Siqi Shen  

**Link**: [PDF](https://arxiv.org/pdf/2510.18442)  

**Abstract**: Large Language Models (LLMs) are increasingly being explored across a range of decision-making tasks. However, LLMs sometimes struggle with decision-making tasks under uncertainty that are relatively easy for humans, such as planning actions in stochastic environments. The adoption of LLMs for decision-making is impeded by uncertainty challenges, such as LLM uncertainty and environmental uncertainty. LLM uncertainty arises from the stochastic sampling process inherent to LLMs. Most LLM-based Decision-Making (LDM) approaches address LLM uncertainty through multiple reasoning chains or search trees. However, these approaches overlook environmental uncertainty, which leads to poor performance in environments with stochastic state transitions. Some recent LDM approaches deal with uncertainty by forecasting the probability of unknown variables. However, they are not designed for multi-step decision-making tasks that require interaction with the environment. To address uncertainty in LLM decision-making, we introduce PlanU, an LLM-based planning method that captures uncertainty within Monte Carlo Tree Search (MCTS). PlanU models the return of each node in the MCTS as a quantile distribution, which uses a set of quantiles to represent the return distribution. To balance exploration and exploitation during tree search, PlanU introduces an Upper Confidence Bounds with Curiosity (UCC) score which estimates the uncertainty of MCTS nodes. Through extensive experiments, we demonstrate the effectiveness of PlanU in LLM-based decision-making tasks under uncertainty. 

---
# AlphaOPT: Formulating Optimization Programs with Self-Improving LLM Experience Library 

**Authors**: Minwei Kong, Ao Qu, Xiaotong Guo, Wenbin Ouyang, Chonghe Jiang, Han Zheng, Yining Ma, Dingyi Zhuang, Yuhan Tang, Junyi Li, Hai Wang, Cathy Wu, Jinhua Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2510.18428)  

**Abstract**: Optimization modeling enables critical decisions across industries but remains difficult to automate: informal language must be mapped to precise mathematical formulations and executable solver code. Prior LLM approaches either rely on brittle prompting or costly retraining with limited generalization. We present AlphaOPT, a self-improving experience library that enables an LLM to learn from limited demonstrations (even answers alone, without gold-standard programs) and solver feedback - without annotated reasoning traces or parameter updates. AlphaOPT operates in a continual two-phase cycle: (i) a Library Learning phase that reflects on failed attempts, extracting solver-verified, structured insights as {taxonomy, condition, explanation, example}; and (ii) a Library Evolution phase that diagnoses retrieval misalignments and refines the applicability conditions of stored insights, improving transfer across tasks. This design (1) learns efficiently from limited demonstrations without curated rationales, (2) expands continually without costly retraining by updating the library rather than model weights, and (3) makes knowledge explicit and interpretable for human inspection and intervention. Experiments show that AlphaOPT steadily improves with more data (65% to 72% from 100 to 300 training items) and surpasses the strongest baseline by 7.7% on the out-of-distribution OptiBench dataset when trained only on answers. Code and data are available at: this https URL. 

---
# Automated urban waterlogging assessment and early warning through a mixture of foundation models 

**Authors**: Chenxu Zhang, Fuxiang Huang, Lei Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2510.18425)  

**Abstract**: With climate change intensifying, urban waterlogging poses an increasingly severe threat to global public safety and infrastructure. However, existing monitoring approaches rely heavily on manual reporting and fail to provide timely and comprehensive assessments. In this study, we present Urban Waterlogging Assessment (UWAssess), a foundation model-driven framework that automatically identifies waterlogged areas in surveillance images and generates structured assessment reports. To address the scarcity of labeled data, we design a semi-supervised fine-tuning strategy and a chain-of-thought (CoT) prompting strategy to unleash the potential of the foundation model for data-scarce downstream tasks. Evaluations on challenging visual benchmarks demonstrate substantial improvements in perception performance. GPT-based evaluations confirm the ability of UWAssess to generate reliable textual reports that accurately describe waterlogging extent, depth, risk and impact. This dual capability enables a shift of waterlogging monitoring from perception to generation, while the collaborative framework of multiple foundation models lays the groundwork for intelligent and scalable systems, supporting urban management, disaster response and climate resilience. 

---
# Med-VRAgent: A Framework for Medical Visual Reasoning-Enhanced Agents 

**Authors**: Guangfu Guo, Xiaoqian Lu, Yue Feng  

**Link**: [PDF](https://arxiv.org/pdf/2510.18424)  

**Abstract**: Visual Language Models (VLMs) achieve promising results in medical reasoning but struggle with hallucinations, vague descriptions, inconsistent logic and poor localization. To address this, we propose a agent framework named Medical Visual Reasoning Agent (\textbf{Med-VRAgent}). The approach is based on Visual Guidance and Self-Reward paradigms and Monte Carlo Tree Search (MCTS). By combining the Visual Guidance with tree search, Med-VRAgent improves the medical visual reasoning capabilities of VLMs. We use the trajectories collected by Med-VRAgent as feedback to further improve the performance by fine-tuning the VLMs with the proximal policy optimization (PPO) objective. Experiments on multiple medical VQA benchmarks demonstrate that our method outperforms existing approaches. 

---
# Deep Learning-Based Control Optimization for Glass Bottle Forming 

**Authors**: Mattia Pujatti, Andrea Di Luca, Nicola Peghini, Federico Monegaglia, Marco Cristoforetti  

**Link**: [PDF](https://arxiv.org/pdf/2510.18412)  

**Abstract**: In glass bottle manufacturing, precise control of forming machines is critical for ensuring quality and minimizing defects. This study presents a deep learning-based control algorithm designed to optimize the forming process in real production environments. Using real operational data from active manufacturing plants, our neural network predicts the effects of parameter changes based on the current production setup. Through a specifically designed inversion mechanism, the algorithm identifies the optimal machine settings required to achieve the desired glass gob characteristics. Experimental results on historical datasets from multiple production lines show that the proposed method yields promising outcomes, suggesting potential for enhanced process stability, reduced waste, and improved product consistency. These results highlight the potential of deep learning to process control in glass manufacturing. 

---
# Heterogeneous Adversarial Play in Interactive Environments 

**Authors**: Manjie Xu, Xinyi Yang, Jiayu Zhan, Wei Liang, Chi Zhang, Yixin Zhu  

**Link**: [PDF](https://arxiv.org/pdf/2510.18407)  

**Abstract**: Self-play constitutes a fundamental paradigm for autonomous skill acquisition, whereby agents iteratively enhance their capabilities through self-directed environmental exploration. Conventional self-play frameworks exploit agent symmetry within zero-sum competitive settings, yet this approach proves inadequate for open-ended learning scenarios characterized by inherent asymmetry. Human pedagogical systems exemplify asymmetric instructional frameworks wherein educators systematically construct challenges calibrated to individual learners' developmental trajectories. The principal challenge resides in operationalizing these asymmetric, adaptive pedagogical mechanisms within artificial systems capable of autonomously synthesizing appropriate curricula without predetermined task hierarchies. Here we present Heterogeneous Adversarial Play (HAP), an adversarial Automatic Curriculum Learning framework that formalizes teacher-student interactions as a minimax optimization wherein task-generating instructor and problem-solving learner co-evolve through adversarial dynamics. In contrast to prevailing ACL methodologies that employ static curricula or unidirectional task selection mechanisms, HAP establishes a bidirectional feedback system wherein instructors continuously recalibrate task complexity in response to real-time learner performance metrics. Experimental validation across multi-task learning domains demonstrates that our framework achieves performance parity with SOTA baselines while generating curricula that enhance learning efficacy in both artificial agents and human subjects. 

---
# Memory-Augmented State Machine Prompting: A Novel LLM Agent Framework for Real-Time Strategy Games 

**Authors**: Runnan Qi, Yanan Ni, Lumin Jiang, Zongyuan Li, Kuihua Huang, Xian Guo  

**Link**: [PDF](https://arxiv.org/pdf/2510.18395)  

**Abstract**: This paper proposes Memory-Augmented State Machine Prompting (MASMP), a novel framework for LLM agents in real-time strategy games. Addressing key challenges like hallucinations and fragmented decision-making in existing approaches, MASMP integrates state machine prompting with memory mechanisms to unify structured actions with long-term tactical coherence. The framework features: (1) a natural language-driven state machine architecture that guides LLMs to emulate finite state machines and behavior trees through prompts, and (2) a lightweight memory module preserving strategic variables (e.g., tactics, priority units) across decision cycles. Experiments in StarCraft II demonstrate MASMP's 60% win rate against the hardest built-in AI (Lv7), vastly outperforming baselines (0%). Case studies reveal the method retains LLMs' semantic comprehension while resolving the "Knowing-Doing Gap" through strict state-action mapping, achieving both interpretability and FSM-like reliability. This work establishes a new paradigm for combining neural and symbolic AI in complex decision-making. 

---
# ShortcutBreaker: Low-Rank Noisy Bottleneck with Global Perturbation Attention for Multi-Class Unsupervised Anomaly Detection 

**Authors**: Peng Tang, Xiaoxiao Yan, Xiaobin Hu, Yuning Cui, Donghao Luo, Jiangning Zhang, Pengcheng Xu, Jinlong Peng, Qingdong He, Feiyue Huang, Song Xue, Tobias Lasser  

**Link**: [PDF](https://arxiv.org/pdf/2510.18342)  

**Abstract**: Multi-class unsupervised anomaly detection (MUAD) has garnered growing research interest, as it seeks to develop a unified model for anomaly detection across multiple classes, i.e., eliminating the need to train separate models for distinct objects and thereby saving substantial computational resources. Under the MUAD setting, while advanced Transformer-based architectures have brought significant performance improvements, identity shortcuts persist: they directly copy inputs to outputs, narrowing the gap in reconstruction errors between normal and abnormal cases, and thereby making the two harder to distinguish. Therefore, we propose ShortcutBreaker, a novel unified feature-reconstruction framework for MUAD tasks, featuring two key innovations to address the issue of shortcuts. First, drawing on matrix rank inequality, we design a low-rank noisy bottleneck (LRNB) to project highdimensional features into a low-rank latent space, and theoretically demonstrate its capacity to prevent trivial identity reproduction. Second, leveraging ViTs global modeling capability instead of merely focusing on local features, we incorporate a global perturbation attention to prevent information shortcuts in the decoders. Extensive experiments are performed on four widely used anomaly detection benchmarks, including three industrial datasets (MVTec-AD, ViSA, and Real-IAD) and one medical dataset (Universal Medical). The proposed method achieves a remarkable image-level AUROC of 99.8%, 98.9%, 90.6%, and 87.8% on these four datasets, respectively, consistently outperforming previous MUAD methods across different scenarios. 

---
# Earth AI: Unlocking Geospatial Insights with Foundation Models and Cross-Modal Reasoning 

**Authors**: Aaron Bell, Amit Aides, Amr Helmy, Arbaaz Muslim, Aviad Barzilai, Aviv Slobodkin, Bolous Jaber, David Schottlander, George Leifman, Joydeep Paul, Mimi Sun, Nadav Sherman, Natalie Williams, Per Bjornsson, Roy Lee, Ruth Alcantara, Thomas Turnbull, Tomer Shekel, Vered Silverman, Yotam Gigi, Adam Boulanger, Alex Ottenwess, Ali Ahmadalipour, Anna Carter, Charles Elliott, David Andre, Elad Aharoni, Gia Jung, Hassler Thurston, Jacob Bien, Jamie McPike, Juliet Rothenberg, Kartik Hegde, Kel Markert, Kim Philipp Jablonski, Luc Houriez, Monica Bharel, Phing VanLee, Reuven Sayag, Sebastian Pilarski, Shelley Cazares, Shlomi Pasternak, Siduo Jiang, Stone Jiang, Thomas Colthurst, Yang Chen, Yehonathan Refael, Yochai Blau, Yuval Carny, Yael Maguire, Avinatan Hassidim, James Manyika, Tim Thelin, Genady Beryozkin, Gautam Prasad, Luke Barrington, Yossi Matias, Niv Efron, Shravya Shetty  

**Link**: [PDF](https://arxiv.org/pdf/2510.18318)  

**Abstract**: Geospatial data offers immense potential for understanding our planet. However, the sheer volume and diversity of this data along with its varied resolutions, timescales, and sparsity pose significant challenges for thorough analysis and interpretation. This paper introduces Earth AI, a family of geospatial AI models and agentic reasoning that enables significant advances in our ability to unlock novel and profound insights into our planet. This approach is built upon foundation models across three key domains--Planet-scale Imagery, Population, and Environment--and an intelligent Gemini-powered reasoning engine. We present rigorous benchmarks showcasing the power and novel capabilities of our foundation models and validate that when used together, they provide complementary value for geospatial inference and their synergies unlock superior predictive capabilities. To handle complex, multi-step queries, we developed a Gemini-powered agent that jointly reasons over our multiple foundation models along with large geospatial data sources and tools. On a new benchmark of real-world crisis scenarios, our agent demonstrates the ability to deliver critical and timely insights, effectively bridging the gap between raw geospatial data and actionable understanding. 

---
# Genesis: Evolving Attack Strategies for LLM Web Agent Red-Teaming 

**Authors**: Zheng Zhang, Jiarui He, Yuchen Cai, Deheng Ye, Peilin Zhao, Ruili Feng, Hao Wang  

**Link**: [PDF](https://arxiv.org/pdf/2510.18314)  

**Abstract**: As large language model (LLM) agents increasingly automate complex web tasks, they boost productivity while simultaneously introducing new security risks. However, relevant studies on web agent attacks remain limited. Existing red-teaming approaches mainly rely on manually crafted attack strategies or static models trained offline. Such methods fail to capture the underlying behavioral patterns of web agents, making it difficult to generalize across diverse environments. In web agent attacks, success requires the continuous discovery and evolution of attack strategies. To this end, we propose Genesis, a novel agentic framework composed of three modules: Attacker, Scorer, and Strategist. The Attacker generates adversarial injections by integrating the genetic algorithm with a hybrid strategy representation. The Scorer evaluates the target web agent's responses to provide feedback. The Strategist dynamically uncovers effective strategies from interaction logs and compiles them into a continuously growing strategy library, which is then re-deployed to enhance the Attacker's effectiveness. Extensive experiments across various web tasks show that our framework discovers novel strategies and consistently outperforms existing attack baselines. 

---
# Illusions of reflection: open-ended task reveals systematic failures in Large Language Models' reflective reasoning 

**Authors**: Sion Weatherhead, Flora Salim, Aaron Belbasis  

**Link**: [PDF](https://arxiv.org/pdf/2510.18254)  

**Abstract**: Humans do not just find mistakes after the fact -- we often catch them mid-stream because 'reflection' is tied to the goal and its constraints. Today's large language models produce reasoning tokens and 'reflective' text, but is it functionally equivalent with human reflective reasoning? Prior work on closed-ended tasks -- with clear, external 'correctness' signals -- can make 'reflection' look effective while masking limits in self-correction. We therefore test eight frontier models on a simple, real-world task that is open-ended yet rule-constrained, with auditable success criteria: to produce valid scientific test items, then revise after considering their own critique. First-pass performance is poor (often zero valid items out of 4 required; mean $\approx$ 1), and reflection yields only modest gains (also $\approx$ 1). Crucially, the second attempt frequently repeats the same violation of constraint, indicating 'corrective gains' arise largely from chance production of a valid item rather than error detection and principled, constraint-sensitive repair. Performance before and after reflection deteriorates as open-endedness increases, and models marketed for 'reasoning' show no advantage. Our results suggest that current LLM 'reflection' lacks functional evidence of the active, goal-driven monitoring that helps humans respect constraints even on a first pass. Until such mechanisms are instantiated in the model itself, reliable performance requires external structure that enforces constraints. 

---
# ssToken: Self-modulated and Semantic-aware Token Selection for LLM Fine-tuning 

**Authors**: Xiaohan Qin, Xiaoxing Wang, Ning Liao, Cancheng Zhang, Xiangdong Zhang, Mingquan Feng, Jingzhi Wang, Junchi Yan  

**Link**: [PDF](https://arxiv.org/pdf/2510.18250)  

**Abstract**: Data quality plays a critical role in enhancing supervised fine-tuning (SFT) for large language models (LLMs), and token-level data selection has emerged as a promising direction for its fine-grained nature. Despite their strong empirical performance, existing token-level selection methods share two key limitations: (1) requiring training or accessing an additional reference model, and (2) relying solely on loss information for token selection, which cannot well preserve semantically important tokens that are not favored by loss-based metrics. To address these challenges, we propose ssToken, a Self-modulated and Semantic-aware Token Selection approach. ssToken leverages readily accessible history models to compute the per-token loss difference with the current model, which serves as a self-modulated signal that enables the model to adaptively select tokens along its optimization trajectory, rather than relying on excess loss from an offline-trained reference model as in prior works. We further introduce a semantic-aware, attention-based token importance estimation metric, orthogonal to loss-based selection and providing complementary semantic information for more effective filtering. Extensive experiments across different model families and scales demonstrate that both self-modulated selection and semantic-aware selection alone outperform full-data fine-tuning, while their integration--ssToken--achieves synergistic gains and further surpasses prior token-level selection methods, delivering performance improvements while maintaining training efficiency. 

---
# A Definition of AGI 

**Authors**: Dan Hendrycks, Dawn Song, Christian Szegedy, Honglak Lee, Yarin Gal, Erik Brynjolfsson, Sharon Li, Andy Zou, Lionel Levine, Bo Han, Jie Fu, Ziwei Liu, Jinwoo Shin, Kimin Lee, Mantas Mazeika, Long Phan, George Ingebretsen, Adam Khoja, Cihang Xie, Olawale Salaudeen, Matthias Hein, Kevin Zhao, Alexander Pan, David Duvenaud, Bo Li, Steve Omohundro, Gabriel Alfour, Max Tegmark, Kevin McGrew, Gary Marcus, Jaan Tallinn, Eric Schmidt, Yoshua Bengio  

**Link**: [PDF](https://arxiv.org/pdf/2510.18212)  

**Abstract**: The lack of a concrete definition for Artificial General Intelligence (AGI) obscures the gap between today's specialized AI and human-level cognition. This paper introduces a quantifiable framework to address this, defining AGI as matching the cognitive versatility and proficiency of a well-educated adult. To operationalize this, we ground our methodology in Cattell-Horn-Carroll theory, the most empirically validated model of human cognition. The framework dissects general intelligence into ten core cognitive domains-including reasoning, memory, and perception-and adapts established human psychometric batteries to evaluate AI systems. Application of this framework reveals a highly "jagged" cognitive profile in contemporary models. While proficient in knowledge-intensive domains, current AI systems have critical deficits in foundational cognitive machinery, particularly long-term memory storage. The resulting AGI scores (e.g., GPT-4 at 27%, GPT-5 at 58%) concretely quantify both rapid progress and the substantial gap remaining before AGI. 

---
# FST.ai 2.0: An Explainable AI Ecosystem for Fair, Fast, and Inclusive Decision-Making in Olympic and Paralympic Taekwondo 

**Authors**: Keivan Shariatmadar, Ahmad Osman, Ramin Ray, Usman Dildar, Kisam Kim  

**Link**: [PDF](https://arxiv.org/pdf/2510.18193)  

**Abstract**: Fair, transparent, and explainable decision-making remains a critical challenge in Olympic and Paralympic combat sports. This paper presents \emph{this http URL 2.0}, an explainable AI ecosystem designed to support referees, coaches, and athletes in real time during Taekwondo competitions and training. The system integrates {pose-based action recognition} using graph convolutional networks (GCNs), {epistemic uncertainty modeling} through credal sets, and {explainability overlays} for visual decision support. A set of {interactive dashboards} enables human--AI collaboration in referee evaluation, athlete performance analysis, and Para-Taekwondo classification. Beyond automated scoring, this http URL~2.0 incorporates modules for referee training, fairness monitoring, and policy-level analytics within the World Taekwondo ecosystem. Experimental validation on competition data demonstrates an {85\% reduction in decision review time} and {93\% referee trust} in AI-assisted decisions. The framework thus establishes a transparent and extensible pipeline for trustworthy, data-driven officiating and athlete assessment. By bridging real-time perception, explainable inference, and governance-aware design, this http URL~2.0 represents a step toward equitable, accountable, and human-aligned AI in sports. 

---
# Local Coherence or Global Validity? Investigating RLVR Traces in Math Domains 

**Authors**: Soumya Rani Samineni, Durgesh Kalwar, Vardaan Gangal, Siddhant Bhambri, Subbarao Kambhampati  

**Link**: [PDF](https://arxiv.org/pdf/2510.18176)  

**Abstract**: Reinforcement Learning with Verifiable Rewards (RLVR)-based post-training of Large Language Models (LLMs) has been shown to improve accuracy on reasoning tasks and continues to attract significant attention. Existing RLVR methods, however, typically treat all tokens uniformly without accounting for token-level advantages. These methods primarily evaluate performance based on final answer correctness or Pass@K accuracy, and yet make claims about RL post-training leading to improved reasoning traces. This motivates our investigation into the effect of RL post-training on intermediate tokens which are not directly incentivized. To study this, we design an experimental setup using the GRPO algorithm with Qwen-2.5-0.5B model on the GSM8K dataset. We introduce trace coherence, a First-Order Logic (FOL)-based measure to capture the consistency of reasoning steps by identifying errors in the traces. We distinguish between trace validity and trace coherence, noting that the former implies logical soundness while the latter measures local coherence via lack of errors. Our results show that RL post-training overall improves trace coherence with the most significant gains on problems where the base model fails but the RL model succeeds. Surprisingly, RL enhances local coherence without necessarily producing valid or correct solutions. This highlights a crucial distinction: improved local coherence in reasoning steps does not guarantee final answer correctness. We argue that claims of improved reasoning via RL must be examined with care, as these may be based on improved trace coherence, which may not translate into fully valid mathematical proofs. 

---
# AgentChangeBench: A Multi-Dimensional Evaluation Framework for Goal-Shift Robustness in Conversational AI 

**Authors**: Manik Rana, Calissa Man, Anotida Expected Msiiwa, Jeffrey Paine, Kevin Zhu, Sunishchal Dev, Vasu Sharma, Ahan M R  

**Link**: [PDF](https://arxiv.org/pdf/2510.18170)  

**Abstract**: Goal changes are a defining feature of real world multi-turn interactions, yet current agent benchmarks primarily evaluate static objectives or one-shot tool use. We introduce AgentChangeBench, a benchmark explicitly designed to measure how tool augmented language model agents adapt to mid dialogue goal shifts across three enterprise domains. Our framework formalizes evaluation through four complementary metrics: Task Success Rate (TSR) for effectiveness, Tool Use Efficiency (TUE) for reliability, Tool Call Redundancy Rate (TCRR) for wasted effort, and Goal-Shift Recovery Time (GSRT) for adaptation latency. AgentChangeBench comprises 2,835 task sequences and five user personas, each designed to trigger realistic shift points in ongoing workflows. Using this setup, we evaluate several frontier models and uncover sharp contrasts obscured by traditional $\text{pass}@k$ scores: for example, GPT-4o reaches $92.2\%$ recovery on airline booking shifts while Gemini collapses to $48.6\%$, and retail tasks show near perfect parameter validity yet redundancy rates above $80\%$, revealing major inefficiencies. These findings demonstrate that high raw accuracy does not imply robustness under dynamic goals, and that explicit measurement of recovery time and redundancy is essential. AgentChangeBench establishes a reproducible testbed for diagnosing and improving agent resilience in realistic enterprise settings. 

---
# Saber: An Efficient Sampling with Adaptive Acceleration and Backtracking Enhanced Remasking for Diffusion Language Model 

**Authors**: Yihong Dong, Zhaoyu Ma, Xue Jiang, Zhiyuan Fan, Jiaru Qian, Yongmin Li, Jianha Xiao, Zhi Jin, Rongyu Cao, Binhua Li, Fei Huang, Yongbin Li, Ge Li  

**Link**: [PDF](https://arxiv.org/pdf/2510.18165)  

**Abstract**: Diffusion language models (DLMs) are emerging as a powerful and promising alternative to the dominant autoregressive paradigm, offering inherent advantages in parallel generation and bidirectional context modeling. However, the performance of DLMs on code generation tasks, which have stronger structural constraints, is significantly hampered by the critical trade-off between inference speed and output quality. We observed that accelerating the code generation process by reducing the number of sampling steps usually leads to a catastrophic collapse in performance. In this paper, we introduce efficient Sampling with Adaptive acceleration and Backtracking Enhanced Remasking (i.e., Saber), a novel training-free sampling algorithm for DLMs to achieve better inference speed and output quality in code generation. Specifically, Saber is motivated by two key insights in the DLM generation process: 1) it can be adaptively accelerated as more of the code context is established; 2) it requires a backtracking mechanism to reverse the generated tokens. Extensive experiments on multiple mainstream code generation benchmarks show that Saber boosts Pass@1 accuracy by an average improvement of 1.9% over mainstream DLM sampling methods, meanwhile achieving an average 251.4% inference speedup. By leveraging the inherent advantages of DLMs, our work significantly narrows the performance gap with autoregressive models in code generation. 

---
# LLM-Based Multi-Agent System for Simulating and Analyzing Marketing and Consumer Behavior 

**Authors**: Man-Lin Chu, Lucian Terhorst, Kadin Reed, Tom Ni, Weiwei Chen, Rongyu Lin  

**Link**: [PDF](https://arxiv.org/pdf/2510.18155)  

**Abstract**: Simulating consumer decision-making is vital for designing and evaluating marketing strategies before costly real- world deployment. However, post-event analyses and rule-based agent-based models (ABMs) struggle to capture the complexity of human behavior and social interaction. We introduce an LLM-powered multi-agent simulation framework that models consumer decisions and social dynamics. Building on recent advances in large language model simulation in a sandbox envi- ronment, our framework enables generative agents to interact, express internal reasoning, form habits, and make purchasing decisions without predefined rules. In a price-discount marketing scenario, the system delivers actionable strategy-testing outcomes and reveals emergent social patterns beyond the reach of con- ventional methods. This approach offers marketers a scalable, low-risk tool for pre-implementation testing, reducing reliance on time-intensive post-event evaluations and lowering the risk of underperforming campaigns. 

---
# Annotating the Chain-of-Thought: A Behavior-Labeled Dataset for AI Safety 

**Authors**: Antonio-Gabriel Chacón Menke, Phan Xuan Tan, Eiji Kamioka  

**Link**: [PDF](https://arxiv.org/pdf/2510.18154)  

**Abstract**: Recent work has highlighted the importance of monitoring chain-of-thought reasoning for AI safety; however, current approaches that analyze textual reasoning steps can miss subtle harmful patterns and may be circumvented by models that hide unsafe reasoning. We present a sentence-level labeled dataset that enables activation-based monitoring of safety behaviors during LLM reasoning. Our dataset contains reasoning sequences with sentence-level annotations of safety behaviors such as expression of safety concerns or speculation on user intent, which we use to extract steering vectors for detecting and influencing these behaviors within model activations. The dataset fills a key gap in safety research: while existing datasets label reasoning holistically, effective application of steering vectors for safety monitoring could be improved by identifying precisely when specific behaviors occur within reasoning chains. We demonstrate the dataset's utility by extracting representations that both detect and steer safety behaviors in model activations, showcasing the potential of activation-level techniques for improving safety oversight on reasoning.
Content Warning: This paper discusses AI safety in the context of harmful prompts and may contain references to potentially harmful content. 

---
# Learning from Generalization Patterns: An Evaluation-Driven Approach to Enhanced Data Augmentation for Fine-Tuning Small Language Models 

**Authors**: Huan Song, Deeksha Razdan, Yiyue Qian, Arijit Ghosh Chowdhury, Parth Patwa, Aman Chadha, Shinan Zhang, Sharlina Keshava, Hannah Marlowe  

**Link**: [PDF](https://arxiv.org/pdf/2510.18143)  

**Abstract**: Small Language Models (SLMs) offer compelling advantages in deployment cost and latency, but their accuracy often lags behind larger models, particularly for complex domain-specific tasks. While supervised fine-tuning can help bridge this performance gap, it requires substantial manual effort in data preparation and iterative optimization. We present PaDA-Agent (Pattern-guided Data Augmentation Agent), an evaluation-driven approach that streamlines the data augmentation process for SLMs through coordinated operations. Unlike state-of-the-art approaches that focus on model training errors only and generating error-correcting samples, PaDA-Agent discovers failure patterns from the validation data via evaluations and drafts targeted data augmentation strategies aiming to directly reduce the generalization gap. Our experimental results demonstrate significant improvements over state-of-the-art LLM-based data augmentation approaches for Llama 3.2 1B Instruct model fine-tuning. 

---
# Measuring Reasoning in LLMs: a New Dialectical Angle 

**Authors**: Soheil Abbasloo  

**Link**: [PDF](https://arxiv.org/pdf/2510.18134)  

**Abstract**: What does it truly mean for a language model to "reason"? Most current evaluations and benchmarks reward models' correct standalone answers--but correctness alone reveals little about the process that produced them. In this work, we explore a different perspective: reasoning is not a static chain of steps, but a dynamic trajectory where ideas interact, clash, and evolve into deeper insights. To capture this dynamic, we draw on a well-established philosophical tradition: \textit{dialectics}, where reasoning unfolds through thesis, antithesis, and synthesis. Building on this, we present SIEV, a structured framework that evaluates reasoning of LLMs through dialectics. Unlike conventional evaluations, SIEV assesses not only the conclusion a model reaches, but how it gets there: its ability to resolve tension, integrate distinct ideas, and synthesize higher-order reasoning. This lens uncovers significant reasoning gaps in state-of-the-art models even under saturated benchmarks like GSM and MMLU. For instance, GPT-5-chat, a recent model, loses over 40 points (out of 100) when evaluated with SIEV on GSM. Our findings highlight that adopting a process-oriented, philosophically grounded approach enables a deeper, more rigorous, and more discriminative assessment of LLM reasoning. 

---
# SMaRT: Select, Mix, and ReinvenT - A Strategy Fusion Framework for LLM-Driven Reasoning and Planning 

**Authors**: Nikhil Verma, Manasa Bharadwaj, Wonjun Jang, Harmanpreet Singh, Yixiao Wang, Homa Fashandi, Chul Lee  

**Link**: [PDF](https://arxiv.org/pdf/2510.18095)  

**Abstract**: Large Language Models (LLMs) have redefined complex task automation with exceptional generalization capabilities. Despite these advancements, state-of-the-art methods rely on single-strategy prompting, missing the synergy of diverse reasoning approaches. No single strategy excels universally, highlighting the need for frameworks that fuse strategies to maximize performance and ensure robustness. We introduce the Select, Mix, and ReinvenT (SMaRT) framework, an innovative strategy fusion approach designed to overcome this constraint by creating balanced and efficient solutions through the seamless integration of diverse reasoning strategies. Unlike existing methods, which employ LLMs merely as evaluators, SMaRT uses them as intelligent integrators, unlocking the "best of all worlds" across tasks. Extensive empirical evaluations across benchmarks in reasoning, planning, and sequential decision-making highlight the robustness and adaptability of SMaRT. The framework consistently outperforms state-of-the-art baselines in solution quality, constraint adherence, and performance metrics. This work redefines LLM-driven decision-making by pioneering a new paradigm in cross-strategy calibration, unlocking superior outcomes for reasoning systems and advancing the boundaries of self-refining methodologies. 

---
# Planned Diffusion 

**Authors**: Daniel Israel, Tian Jin, Ellie Cheng, Guy Van den Broeck, Aditya Grover, Suvinay Subramanian, Michael Carbin  

**Link**: [PDF](https://arxiv.org/pdf/2510.18087)  

**Abstract**: A central challenge in large language model inference is the trade-off between generation speed and output quality. Autoregressive models produce high-quality text but generate tokens sequentially. Diffusion models can generate tokens in parallel but often need many iterations to match the same quality. We propose planned diffusion, a hybrid method that combines the strengths of both paradigms. Planned diffusion works in two stages: first, the model creates a short autoregressive plan that breaks the output into smaller, independent spans. Second, the model generates these spans simultaneously using diffusion. This approach expands the speed-quality Pareto frontier and provides a practical path to faster, high-quality text generation. On AlpacaEval, a suite of 805 instruction-following prompts, planned diffusion achieves Pareto-optimal trade-off between quality and latency, achieving 1.27x to 1.81x speedup over autoregressive generation with only 0.87\% to 5.4\% drop in win rate, respectively. Our sensitivity analysis shows that the planning mechanism of planned diffusion is minimal and reliable, and simple runtime knobs exist to provide flexible control of the quality-latency trade-off. 

---
# CompactPrompt: A Unified Pipeline for Prompt Data Compression in LLM Workflows 

**Authors**: Joong Ho Choi, Jiayang Zhao, Jeel Shah, Ritvika Sonawane, Vedant Singh, Avani Appalla, Will Flanagan, Filipe Condessa  

**Link**: [PDF](https://arxiv.org/pdf/2510.18043)  

**Abstract**: Large Language Models (LLMs) deliver powerful reasoning and generation capabilities but incur substantial run-time costs when operating in agentic workflows that chain together lengthy prompts and process rich data streams. We introduce CompactPrompt, an end-to-end pipeline that merges hard prompt compression with lightweight file-level data compression. CompactPrompt first prunes low-information tokens from prompts using self-information scoring and dependency-based phrase grouping. In parallel, it applies n-gram abbreviation to recurrent textual patterns in attached documents and uniform quantization to numerical columns, yielding compact yet semantically faithful representations. Integrated into standard LLM agents, CompactPrompt reduces total token usage and inference cost by up to 60% on benchmark dataset like TAT-QA and FinQA, while preserving output quality (Results in less than 5% accuracy drop for Claude-3.5-Sonnet, and GPT-4.1-Mini) CompactPrompt helps visualize real-time compression decisions and quantify cost-performance trade-offs, laying the groundwork for leaner generative AI pipelines. 

---
# Subject-Event Ontology Without Global Time: Foundations and Execution Semantics 

**Authors**: Alexander Boldachev  

**Link**: [PDF](https://arxiv.org/pdf/2510.18040)  

**Abstract**: A formalization of a subject-event ontology is proposed for modeling complex dynamic systems without reliance on global time. Key principles: (1) event as an act of fixation - a subject discerns and fixes changes according to models (conceptual templates) available to them; (2) causal order via happens-before - the order of events is defined by explicit dependencies, not timestamps; (3) making the ontology executable via a declarative dataflow mechanism, ensuring determinism; (4) models as epistemic filters - a subject can only fix what falls under its known concepts and properties; (5) presumption of truth - the declarative content of an event is available for computation from the moment of fixation, without external verification. The formalization includes nine axioms (A1-A9), ensuring the correctness of executable ontologies: monotonicity of history (I1), acyclicity of causality (I2), traceability (I3). Special attention is given to the model-based approach (A9): event validation via schemas, actor authorization, automatic construction of causal chains (W3) without global time. Practical applicability is demonstrated on the boldsea system - a workflow engine for executable ontologies, where the theoretical constructs are implemented in BSL (Boldsea Semantic Language). The formalization is applicable to distributed systems, microservice architectures, DLT platforms, and multiperspectivity scenarios (conflicting facts from different subjects). 

---
# OPTAGENT: Optimizing Multi-Agent LLM Interactions Through Verbal Reinforcement Learning for Enhanced Reasoning 

**Authors**: Zhenyu Bi, Meng Lu, Yang Li, Swastik Roy, Weijie Guan, Morteza Ziyadi, Xuan Wang  

**Link**: [PDF](https://arxiv.org/pdf/2510.18032)  

**Abstract**: Large Language Models (LLMs) have shown remarkable reasoning capabilities in mathematical and scientific tasks. To enhance complex reasoning, multi-agent systems have been proposed to harness the collective intelligence of LLM agents. However, existing collaboration structures are either predefined or rely on majority voting or round-table debates, which can suppress correct but less dominant agent contributions. Recent approaches model multi-agent systems as graph networks but optimize purely for agent performance, neglecting the quality of interactions. We hypothesize that effective agent communication is crucial for multi-agent reasoning and that debating quality plays a significant role. To address this, we propose $\ours$, a multi-agent verbal reinforcement learning algorithm that dynamically constructs and refines multi-agent collaboration structures. Our method defines action spaces and a feedback mechanism that evaluates communication robustness and coherence throughout the debate. The final decision is achieved through a majority vote over all the agents. We assess $\ours$ on various reasoning tasks, including mathematical reasoning, creative writing, scientific reasoning, and numerical sorting. Results demonstrate that our approach significantly outperforms single-agent prompting methods and state-of-the-art multi-agent frameworks on diverse tasks. 

---
# FABRIC: Framework for Agent-Based Realistic Intelligence Creation 

**Authors**: Abhigya Verma, Seganrasan Subramanian, Nandhakumar Kandasamy, Naman Gupta  

**Link**: [PDF](https://arxiv.org/pdf/2510.17995)  

**Abstract**: Large language models (LLMs) are increasingly deployed as agents, expected to decompose goals, invoke tools, and verify results in dynamic environments. Realizing these capabilities requires access to agentic data- structured interaction records that couple user intents with tool specifications, argument-grounded calls, and verifiable execution traces. However, collecting such data from human annotators is costly, time-consuming, and difficult to scale.
We present a unified framework for synthesizing agentic data using only LLMs, without any human-in-the-loop supervision. This framework decomposes generation into modular pipelines that produce complete interaction records spanning task specifications, tool definitions, policy pseudocode, natural language exchanges, and execution traces. Records conform to strict syntactic and semantic constraints, ensuring machine-parseability and faithful alignment across inputs, outputs, and tool calls.
Beyond single tasks, there is support for both multi-task and multi-turn agent interactions, enabling the construction of datasets that reflect the full spectrum of tool-use competencies. To ensure quality and consistency, the framework integrates constrained generation formats, JSON-schema validation, and judge-based filtering.
This paper formalizes the schema for agentic records, details the prompt design principles that guide generation, and introduces scalable pipelines for high-quality synthetic data. By providing a reproducible, LLM-only alternative to manual collection, hence advancing the development of agentic LLMs capable of robust tool use. 

---
# Beyond More Context: Retrieval Diversity Boosts Multi-Turn Intent Understanding 

**Authors**: Zhiming Lin  

**Link**: [PDF](https://arxiv.org/pdf/2510.17940)  

**Abstract**: Multi turn intent understanding is central to task oriented chatbots, yet real deployments face tight token budgets and noisy contexts, and most retrieval pipelines emphasize relevance while overlooking set level diversity and confounds such as more context or exemplar order. We ask whether retrieval diversity, rather than longer prompts, systematically improves LLM intent understanding under fixed budgets. We present a diversity aware retrieval framework that selects in context exemplars to balance intent coverage and linguistic variety, and integrates this selection with standard LLM decoders; the evaluation enforces budget matched prompts and randomized positions, and includes sensitivity analyses over exemplar count, diversity strength, and backbone size. On MultiWOZ 2.4 and SGD, the approach achieves strong gains in Joint Goal Accuracy under equal token budgets, surpassing strong LLM/DST baselines, with consistent improvements across K from 4 to 7 and moderate latency. Overall, the study isolates and validates the impact of content diversity in retrieval and offers a simple, deployable selection principle for building accurate, budget constrained multi turn intent systems. 

---
# Activation Manifold Projection: Liberating Task-Specific Behaviors from LLM Architectures 

**Authors**: Al Kari  

**Link**: [PDF](https://arxiv.org/pdf/2510.17902)  

**Abstract**: The proliferation of Large Language Model (LLM) architectures presents a fundamental challenge: valuable, task-specific behaviors learned through fine-tuning methods like Low-Rank Adaptation (LoRA) are effectively trapped within their source model's architecture, herein referred to architectural lock-in. Existing transfer methods attempt to bridge this gap by aligning the static weight spaces of models, a brittle and indirect approach that relies on tenuous correlations between parameter geometries. This paper introduces a fundamentally different and more direct paradigm: the Cartridge Activation Space Transfer (CAST), a novel framework that liberates LoRA-encoded behaviors by learning a direct, nonlinear mapping between the activation manifolds, the geometric structures formed by the model's internal neuron activations, of two distinct LLM architectures. CAST treats a pre-trained LoRA as a frozen "behavioral kernel." It learns a set of lightweight, bidirectional projection heads that translate the target model's activation stream into the source model's latent space, apply the frozen kernel, and project the result back. This process, trained on a general text corpus without any task-specific data, effectively decouples the learned skill from the source architecture. We demonstrate that CAST enables true "zero-shot" translation of any standard LoRA adapter. Our experiments, including transfers between heterogeneous model families like Llama-2 and Mistral, show that CAST-translated adapters achieve 85-95\% of the performance of a LoRA fully retrained on the target model, quantitatively outperforming current weight-space transfer techniques and establishing a new state-of-the-art in model interoperability. 

---
# Grasp Any Region: Towards Precise, Contextual Pixel Understanding for Multimodal LLMs 

**Authors**: Haochen Wang, Yuhao Wang, Tao Zhang, Yikang Zhou, Yanwei Li, Jiacong Wang, Ye Tian, Jiahao Meng, Zilong Huang, Guangcan Mai, Anran Wang, Yunhai Tong, Zhuochen Wang, Xiangtai Li, Zhaoxiang Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2510.18876)  

**Abstract**: While Multimodal Large Language Models (MLLMs) excel at holistic understanding, they struggle in capturing the dense world with complex scenes, requiring fine-grained analysis of intricate details and object inter-relationships. Region-level MLLMs have been a promising step. However, previous attempts are generally optimized to understand given regions in isolation, neglecting crucial global contexts. To address this, we introduce Grasp Any Region (GAR) for comprehen- sive region-level visual understanding. Empowered by an effective RoI-aligned feature replay technique, GAR supports (1) precise perception by leveraging necessary global contexts, and (2) modeling interactions between multiple prompts. Together, it then naturally achieves (3) advanced compositional reasoning to answer specific free-form questions about any region, shifting the paradigm from passive description to active dialogue. Moreover, we construct GAR-Bench, which not only provides a more accurate evaluation of single-region comprehension, but also, more importantly, measures interactions and complex reasoning across multiple regions. Extensive experiments have demonstrated that GAR-1B not only maintains the state-of-the-art captioning capabilities, e.g., outperforming DAM-3B +4.5 on DLC-Bench, but also excels at modeling relationships between multiple prompts with advanced comprehension capabilities, even surpassing InternVL3-78B on GAR-Bench-VQA. More importantly, our zero-shot GAR-8B even outperforms in-domain VideoRefer-7B on VideoRefer-BenchQ, indicating its strong capabilities can be easily transferred to videos. 

---
# How Do LLMs Use Their Depth? 

**Authors**: Akshat Gupta, Jay Yeung, Gopala Anumanchipalli, Anna Ivanova  

**Link**: [PDF](https://arxiv.org/pdf/2510.18871)  

**Abstract**: Growing evidence suggests that large language models do not use their depth uniformly, yet we still lack a fine-grained understanding of their layer-wise prediction dynamics. In this paper, we trace the intermediate representations of several open-weight models during inference and reveal a structured and nuanced use of depth. Specifically, we propose a "Guess-then-Refine" framework that explains how LLMs internally structure their computations to make predictions. We first show that the top-ranked predictions in early LLM layers are composed primarily of high-frequency tokens, which act as statistical guesses proposed by the model early on due to the lack of appropriate contextual information. As contextual information develops deeper into the model, these initial guesses get refined into contextually appropriate tokens. Even high-frequency token predictions from early layers get refined >70% of the time, indicating that correct token prediction is not "one-and-done". We then go beyond frequency-based prediction to examine the dynamic usage of layer depth across three case studies. (i) Part-of-speech analysis shows that function words are, on average, the earliest to be predicted correctly. (ii) Fact recall task analysis shows that, in a multi-token answer, the first token requires more computational depth than the rest. (iii) Multiple-choice task analysis shows that the model identifies the format of the response within the first half of the layers, but finalizes its response only toward the end. Together, our results provide a detailed view of depth usage in LLMs, shedding light on the layer-by-layer computations that underlie successful predictions and providing insights for future works to improve computational efficiency in transformer-based models. 

---
# LightMem: Lightweight and Efficient Memory-Augmented Generation 

**Authors**: Jizhan Fang, Xinle Deng, Haoming Xu, Ziyan Jiang, Yuqi Tang, Ziwen Xu, Shumin Deng, Yunzhi Yao, Mengru Wang, Shuofei Qiao, Huajun Chen, Ningyu Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2510.18866)  

**Abstract**: Despite their remarkable capabilities, Large Language Models (LLMs) struggle to effectively leverage historical interaction information in dynamic and complex environments. Memory systems enable LLMs to move beyond stateless interactions by introducing persistent information storage, retrieval, and utilization mechanisms. However, existing memory systems often introduce substantial time and computational overhead. To this end, we introduce a new memory system called LightMem, which strikes a balance between the performance and efficiency of memory systems. Inspired by the Atkinson-Shiffrin model of human memory, LightMem organizes memory into three complementary stages. First, cognition-inspired sensory memory rapidly filters irrelevant information through lightweight compression and groups information according to their topics. Next, topic-aware short-term memory consolidates these topic-based groups, organizing and summarizing content for more structured access. Finally, long-term memory with sleep-time update employs an offline procedure that decouples consolidation from online inference. Experiments on LongMemEval with GPT and Qwen backbones show that LightMem outperforms strong baselines in accuracy (up to 10.9% gains) while reducing token usage by up to 117x, API calls by up to 159x, and runtime by over 12x. The code is available at this https URL. 

---
# Every Step Evolves: Scaling Reinforcement Learning for Trillion-Scale Thinking Model 

**Authors**: Ling Team, Anqi Shen, Baihui Li, Bin Hu, Bin Jing, Cai Chen, Chao Huang, Chao Zhang, Chaokun Yang, Cheng Lin, Chengyao Wen, Congqi Li, Deng Zhao, Dingbo Yuan, Donghai You, Fagui Mao, Fanzhuang Meng, Feng Xu, Guojie Li, Guowei Wang, Hao Dai, Haonan Zheng, Hong Liu, Jia Guo, Jiaming Liu, Jian Liu, Jianhao Fu, Jiannan Shi, Jianwen Wang, Jianxin Lai, Jin Yang, Jun Mei, Jun Zhou, Junbo Zhao, Junping Zhao, Kuan Xu, Le Su, Lei Chen, Li Tang, Liang Jiang, Liangcheng Fu, Lianhao Xu, Linfeng Shi, Lisha Liao, Longfei Zheng, Meng Li, Mingchun Chen, Qi Zuo, Qiang Cheng, Qianggang Cao, Qitao Shi, Quanrui Guo, Senlin Zhu, Shaofei Wang, Shaomian Zheng, Shuaicheng Li, Shuwei Gu, Siba Chen, Tao Wu, Tao Zhang, Tianyu Zhang, Tianyu Zhou, Tiwei Bie, Tongkai Yang, Wang Hong, Wang Ren, Weihua Chen, Wenbo Yu, Wengang Zheng, Xiangchun Wang, Xiaodong Yan, Xiaopei Wan, Xin Zhao, Xinyu Kong, Xinyu Tang, Xudong Han, Xudong Wang, Xuemin Yang, Xueyu Hu, Yalin Zhang, Yan Sun, Yicheng Shan, Yilong Wang, Yingying Xu, Yongkang Liu, Yongzhen Guo, Yuanyuan Wang, Yuchen Yan, Yuefan Wang, Yuhong Guo, Zehuan Li, Zhankai Xu, Zhe Li, Zhenduo Zhang, Zhengke Gui, Zhenxuan Pan, Zhenyu Huang, Zhenzhong Lan, Zhiqiang Ding, Zhiqiang Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2510.18855)  

**Abstract**: We present Ring-1T, the first open-source, state-of-the-art thinking model with a trillion-scale parameter. It features 1 trillion total parameters and activates approximately 50 billion per token. Training such models at a trillion-parameter scale introduces unprecedented challenges, including train-inference misalignment, inefficiencies in rollout processing, and bottlenecks in the RL system. To address these, we pioneer three interconnected innovations: (1) IcePop stabilizes RL training via token-level discrepancy masking and clipping, resolving instability from training-inference mismatches; (2) C3PO++ improves resource utilization for long rollouts under a token budget by dynamically partitioning them, thereby obtaining high time efficiency; and (3) ASystem, a high-performance RL framework designed to overcome the systemic bottlenecks that impede trillion-parameter model training. Ring-1T delivers breakthrough results across critical benchmarks: 93.4 on AIME-2025, 86.72 on HMMT-2025, 2088 on CodeForces, and 55.94 on ARC-AGI-v1. Notably, it attains a silver medal-level result on the IMO-2025, underscoring its exceptional reasoning capabilities. By releasing the complete 1T parameter MoE model to the community, we provide the research community with direct access to cutting-edge reasoning capabilities. This contribution marks a significant milestone in democratizing large-scale reasoning intelligence and establishes a new baseline for open-source model performance. 

---
# Lyapunov-Aware Quantum-Inspired Reinforcement Learning for Continuous-Time Vehicle Control: A Feasibility Study 

**Authors**: Nutkritta Kraipatthanapong, Natthaphat Thathong, Pannita Suksawas, Thanunnut Klunklin, Kritin Vongthonglua, Krit Attahakul, Aueaphum Aueawatthanaphisut  

**Link**: [PDF](https://arxiv.org/pdf/2510.18852)  

**Abstract**: This paper presents a novel Lyapunov-Based Quantum Reinforcement Learning (LQRL) framework that integrates quantum policy optimization with Lyapunov stability analysis for continuous-time vehicle control. The proposed approach combines the representational power of variational quantum circuits (VQCs) with a stability-aware policy gradient mechanism to ensure asymptotic convergence and safe decision-making under dynamic environments. The vehicle longitudinal control problem was formulated as a continuous-state reinforcement learning task, where the quantum policy network generates control actions subject to Lyapunov stability constraints. Simulation experiments were conducted in a closed-loop adaptive cruise control scenario using a quantum-inspired policy trained under stability feedback. The results demonstrate that the LQRL framework successfully embeds Lyapunov stability verification into quantum policy learning, enabling interpretable and stability-aware control performance. Although transient overshoot and Lyapunov divergence were observed under aggressive acceleration, the system maintained bounded state evolution, validating the feasibility of integrating safety guarantees within quantum reinforcement learning architectures. The proposed framework provides a foundational step toward provably safe quantum control in autonomous systems and hybrid quantum-classical optimization domains. 

---
# DP$^2$O-SR: Direct Perceptual Preference Optimization for Real-World Image Super-Resolution 

**Authors**: Rongyuan Wu, Lingchen Sun, Zhengqiang Zhang, Shihao Wang, Tianhe Wu, Qiaosi Yi, Shuai Li, Lei Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2510.18851)  

**Abstract**: Benefiting from pre-trained text-to-image (T2I) diffusion models, real-world image super-resolution (Real-ISR) methods can synthesize rich and realistic details. However, due to the inherent stochasticity of T2I models, different noise inputs often lead to outputs with varying perceptual quality. Although this randomness is sometimes seen as a limitation, it also introduces a wider perceptual quality range, which can be exploited to improve Real-ISR performance. To this end, we introduce Direct Perceptual Preference Optimization for Real-ISR (DP$^2$O-SR), a framework that aligns generative models with perceptual preferences without requiring costly human annotations. We construct a hybrid reward signal by combining full-reference and no-reference image quality assessment (IQA) models trained on large-scale human preference datasets. This reward encourages both structural fidelity and natural appearance. To better utilize perceptual diversity, we move beyond the standard best-vs-worst selection and construct multiple preference pairs from outputs of the same model. Our analysis reveals that the optimal selection ratio depends on model capacity: smaller models benefit from broader coverage, while larger models respond better to stronger contrast in supervision. Furthermore, we propose hierarchical preference optimization, which adaptively weights training pairs based on intra-group reward gaps and inter-group diversity, enabling more efficient and stable learning. Extensive experiments across both diffusion- and flow-based T2I backbones demonstrate that DP$^2$O-SR significantly improves perceptual quality and generalizes well to real-world benchmarks. 

---
# Towards Faithful and Controllable Personalization via Critique-Post-Edit Reinforcement Learning 

**Authors**: Chenghao Zhu, Meiling Tao, Tiannan Wang, Dongyi Ding, Yuchen Eleanor Jiang, Wangchunshu Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2510.18849)  

**Abstract**: Faithfully personalizing large language models (LLMs) to align with individual user preferences is a critical but challenging task. While supervised fine-tuning (SFT) quickly reaches a performance plateau, standard reinforcement learning from human feedback (RLHF) also struggles with the nuances of personalization. Scalar-based reward models are prone to reward hacking which leads to verbose and superficially personalized responses. To address these limitations, we propose Critique-Post-Edit, a robust reinforcement learning framework that enables more faithful and controllable personalization. Our framework integrates two key components: (1) a Personalized Generative Reward Model (GRM) that provides multi-dimensional scores and textual critiques to resist reward hacking, and (2) a Critique-Post-Edit mechanism where the policy model revises its own outputs based on these critiques for more targeted and efficient learning. Under a rigorous length-controlled evaluation, our method substantially outperforms standard PPO on personalization benchmarks. Personalized Qwen2.5-7B achieves an average 11\% win-rate improvement, and personalized Qwen2.5-14B model surpasses the performance of GPT-4.1. These results demonstrate a practical path to faithful, efficient, and controllable personalization. 

---
# Actor-Free Continuous Control via Structurally Maximizable Q-Functions 

**Authors**: Yigit Korkmaz, Urvi Bhuwania, Ayush Jain, Erdem Bıyık  

**Link**: [PDF](https://arxiv.org/pdf/2510.18828)  

**Abstract**: Value-based algorithms are a cornerstone of off-policy reinforcement learning due to their simplicity and training stability. However, their use has traditionally been restricted to discrete action spaces, as they rely on estimating Q-values for individual state-action pairs. In continuous action spaces, evaluating the Q-value over the entire action space becomes computationally infeasible. To address this, actor-critic methods are typically employed, where a critic is trained on off-policy data to estimate Q-values, and an actor is trained to maximize the critic's output. Despite their popularity, these methods often suffer from instability during training. In this work, we propose a purely value-based framework for continuous control that revisits structural maximization of Q-functions, introducing a set of key architectural and algorithmic choices to enable efficient and stable learning. We evaluate the proposed actor-free Q-learning approach on a range of standard simulation tasks, demonstrating performance and sample efficiency on par with state-of-the-art baselines, without the cost of learning a separate actor. Particularly, in environments with constrained action spaces, where the value functions are typically non-smooth, our method with structural maximization outperforms traditional actor-critic methods with gradient-based maximization. We have released our code at this https URL. 

---
# An Explainable Hybrid AI Framework for Enhanced Tuberculosis and Symptom Detection 

**Authors**: Neel Patel, Alexander Wong, Ashkan Ebadi  

**Link**: [PDF](https://arxiv.org/pdf/2510.18819)  

**Abstract**: Tuberculosis remains a critical global health issue, particularly in resource-limited and remote areas. Early detection is vital for treatment, yet the lack of skilled radiologists underscores the need for artificial intelligence (AI)-driven screening tools. Developing reliable AI models is challenging due to the necessity for large, high-quality datasets, which are costly to obtain. To tackle this, we propose a teacher--student framework which enhances both disease and symptom detection on chest X-rays by integrating two supervised heads and a self-supervised head. Our model achieves an accuracy of 98.85% for distinguishing between COVID-19, tuberculosis, and normal cases, and a macro-F1 score of 90.09% for multilabel symptom detection, significantly outperforming baselines. The explainability assessments also show the model bases its predictions on relevant anatomical features, demonstrating promise for deployment in clinical screening and triage settings. 

---
# Fine-Tuned Thoughts: Leveraging Chain-of-Thought Reasoning for Industrial Asset Health Monitoring 

**Authors**: Shuxin Lin, Dhaval Patel, Christodoulos Constantinides  

**Link**: [PDF](https://arxiv.org/pdf/2510.18817)  

**Abstract**: Small Language Models (SLMs) are becoming increasingly popular in specialized fields, such as industrial applications, due to their efficiency, lower computational requirements, and ability to be fine-tuned for domain-specific tasks, enabling accurate and cost-effective solutions. However, performing complex reasoning using SLMs in specialized fields such as Industry 4.0 remains challenging. In this paper, we propose a knowledge distillation framework for industrial asset health, which transfers reasoning capabilities via Chain-of-Thought (CoT) distillation from Large Language Models (LLMs) to smaller, more efficient models (SLMs). We discuss the advantages and the process of distilling LLMs using multi-choice question answering (MCQA) prompts to enhance reasoning and refine decision-making. We also perform in-context learning to verify the quality of the generated knowledge and benchmark the performance of fine-tuned SLMs with generated knowledge against widely used LLMs. The results show that the fine-tuned SLMs with CoT reasoning outperform the base models by a significant margin, narrowing the gap to their LLM counterparts. Our code is open-sourced at: this https URL. 

---
# Online SFT for LLM Reasoning: Surprising Effectiveness of Self-Tuning without Rewards 

**Authors**: Mengqi Li, Lei Zhao, Anthony Man-Cho So, Ruoyu Sun, Xiao Li  

**Link**: [PDF](https://arxiv.org/pdf/2510.18814)  

**Abstract**: We present a simple, self-help online supervised finetuning (OSFT) paradigm for LLM reasoning. In this paradigm, the model generates its own responses and is immediately finetuned on this self-generated data. OSFT is a highly efficient training strategy for LLM reasoning, as it is reward-free and uses just one rollout by default. Experiment results show that OSFT achieves downstream performance on challenging mathematical reasoning tasks comparable to strong reinforcement learning with verifiable rewards (RLVR) methods such as GRPO. Our ablation study further demonstrates the efficiency and robustness of OSFT. The major mechanism of OSFT lies in facilitating the model's own existing preference (latent knowledge) learned from pretraining, which leads to reasoning ability improvement. We believe that OSFT offers an efficient and promising alternative to more complex, reward-based training paradigms. Our code is available at this https URL. 

---
# Computational Foundations for Strategic Coopetition: Formalizing Interdependence and Complementarity 

**Authors**: Vik Pant, Eric Yu  

**Link**: [PDF](https://arxiv.org/pdf/2510.18802)  

**Abstract**: Modern socio-technical systems are characterized by strategic coopetition where actors simultaneously cooperate to create value and compete to capture it. While conceptual modeling languages like i* provide rich qualitative representations of strategic dependencies, they lack mechanisms for quantitative analysis of dynamic trade-offs. Conversely, classical game theory offers mathematical rigor but strips away contextual richness. This technical report bridges this gap by developing computational foundations that formalize two critical dimensions of coopetition: interdependence and complementarity. We ground interdependence in i* structural dependency analysis, translating depender-dependee-dependum relationships into quantitative interdependence coefficients through a structured translation framework. We formalize complementarity following Brandenburger and Nalebuff's Added Value concept, modeling synergistic value creation with validated parameterization. We integrate structural dependencies with bargaining power in value appropriation and introduce a game-theoretic formulation where Nash Equilibrium incorporates structural interdependence. Validation combines comprehensive experimental testing across power and logarithmic value function specifications, demonstrating functional form robustness, with empirical application to the Samsung-Sony S-LCD joint venture (2004-2011), where logarithmic specifications achieve superior empirical fit (validation score 45/60) while power functions provide theoretical tractability. This technical report serves as the foundational reference for a coordinated research program examining strategic coopetition in requirements engineering and multi-agent systems, with companion work addressing trust dynamics, team production, and reciprocity mechanisms. 

---
# Verifiable Accuracy and Abstention Rewards in Curriculum RL to Alleviate Lost-in-Conversation 

**Authors**: Ming Li  

**Link**: [PDF](https://arxiv.org/pdf/2510.18731)  

**Abstract**: Large Language Models demonstrate strong capabilities in single-turn instruction following but suffer from Lost-in-Conversation (LiC), a degradation in performance as information is revealed progressively in multi-turn settings. Motivated by the current progress on Reinforcement Learning with Verifiable Rewards (RLVR), we propose Curriculum Reinforcement Learning with Verifiable Accuracy and Abstention Rewards (RLAAR), a framework that encourages models not only to generate correct answers, but also to judge the solvability of questions in the multi-turn conversation setting. Our approach employs a competence-gated curriculum that incrementally increases dialogue difficulty (in terms of instruction shards), stabilizing training while promoting reliability. Using multi-turn, on-policy rollouts and a mixed-reward system, RLAAR teaches models to balance problem-solving with informed abstention, reducing premature answering behaviors that cause LiC. Evaluated on LiC benchmarks, RLAAR significantly mitigates LiC performance decay (62.6% to 75.1%) and improves calibrated abstention rates (33.5% to 73.4%). Together, these results provide a practical recipe for building multi-turn reliable and trustworthy LLMs. 

---
# HarmNet: A Framework for Adaptive Multi-Turn Jailbreak Attacks on Large Language Models 

**Authors**: Sidhant Narula, Javad Rafiei Asl, Mohammad Ghasemigol, Eduardo Blanco, Daniel Takabi  

**Link**: [PDF](https://arxiv.org/pdf/2510.18728)  

**Abstract**: Large Language Models (LLMs) remain vulnerable to multi-turn jailbreak attacks. We introduce HarmNet, a modular framework comprising ThoughtNet, a hierarchical semantic network; a feedback-driven Simulator for iterative query refinement; and a Network Traverser for real-time adaptive attack execution. HarmNet systematically explores and refines the adversarial space to uncover stealthy, high-success attack paths. Experiments across closed-source and open-source LLMs show that HarmNet outperforms state-of-the-art methods, achieving higher attack success rates. For example, on Mistral-7B, HarmNet achieves a 99.4% attack success rate, 13.9% higher than the best baseline. Index terms: jailbreak attacks; large language models; adversarial framework; query refinement. 

---
# Causally Perturbed Fairness Testing 

**Authors**: Chengwen Du, Tao Chen  

**Link**: [PDF](https://arxiv.org/pdf/2510.18719)  

**Abstract**: To mitigate unfair and unethical discrimination over sensitive features (e.g., gender, age, or race), fairness testing plays an integral role in engineering systems that leverage AI models to handle tabular data. A key challenge therein is how to effectively reveal fairness bugs under an intractable sample size using perturbation. Much current work has been focusing on designing the test sample generators, ignoring the valuable knowledge about data characteristics that can help guide the perturbation and hence limiting their full potential. In this paper, we seek to bridge such a gap by proposing a generic framework of causally perturbed fairness testing, dubbed CausalFT. Through causal inference, the key idea of CausalFT is to extract the most directly and causally relevant non-sensitive feature to its sensitive counterpart, which can jointly influence the prediction of the label. Such a causal relationship is then seamlessly injected into the perturbation to guide a test sample generator. Unlike existing generator-level work, CausalFT serves as a higher-level framework that can be paired with diverse base generators. Extensive experiments on 1296 cases confirm that CausalFT can considerably improve arbitrary base generators in revealing fairness bugs over 93% of the cases with acceptable extra runtime overhead. Compared with a state-of-the-art approach that ranks the non-sensitive features solely based on correlation, CausalFT performs significantly better on 64% cases while being much more efficient. Further, CausalFT can better improve bias resilience in nearly all cases. 

---
# Preference-based Reinforcement Learning beyond Pairwise Comparisons: Benefits of Multiple Options 

**Authors**: Joongkyu Lee, Seouh-won Yi, Min-hwan Oh  

**Link**: [PDF](https://arxiv.org/pdf/2510.18713)  

**Abstract**: We study online preference-based reinforcement learning (PbRL) with the goal of improving sample efficiency. While a growing body of theoretical work has emerged-motivated by PbRL's recent empirical success, particularly in aligning large language models (LLMs)-most existing studies focus only on pairwise comparisons. A few recent works (Zhu et al., 2023, Mukherjee et al., 2024, Thekumparampil et al., 2024) have explored using multiple comparisons and ranking feedback, but their performance guarantees fail to improve-and can even deteriorate-as the feedback length increases, despite the richer information available. To address this gap, we adopt the Plackett-Luce (PL) model for ranking feedback over action subsets and propose M-AUPO, an algorithm that selects multiple actions by maximizing the average uncertainty within the offered subset. We prove that M-AUPO achieves a suboptimality gap of $\tilde{\mathcal{O}}\left( \frac{d}{T} \sqrt{ \sum_{t=1}^T \frac{1}{|S_t|}} \right)$, where $T$ is the total number of rounds, $d$ is the feature dimension, and $|S_t|$ is the size of the subset at round $t$. This result shows that larger subsets directly lead to improved performance and, notably, the bound avoids the exponential dependence on the unknown parameter's norm, which was a fundamental limitation in most previous works. Moreover, we establish a near-matching lower bound of $\Omega \left( \frac{d}{K \sqrt{T}} \right)$, where $K$ is the maximum subset size. To the best of our knowledge, this is the first theoretical result in PbRL with ranking feedback that explicitly shows improved sample efficiency as a function of the subset size. 

---
# Fetch.ai: An Architecture for Modern Multi-Agent Systems 

**Authors**: Michael J. Wooldridge, Attila Bagoly, Jonathan J. Ward, Emanuele La Malfa, Gabriel Paludo Licks  

**Link**: [PDF](https://arxiv.org/pdf/2510.18699)  

**Abstract**: Recent surges in LLM-driven intelligent systems largely overlook decades of foundational multi-agent systems (MAS) research, resulting in frameworks with critical limitations such as centralization and inadequate trust and communication protocols. This paper introduces the this http URL architecture, an industrial-strength platform designed to bridge this gap by facilitating the integration of classical MAS principles with modern AI capabilities. We present a novel, multi-layered solution built on a decentralized foundation of on-chain blockchain services for verifiable identity, discovery, and transactions. This is complemented by a comprehensive development framework for creating secure, interoperable agents, a cloud-based platform for deployment, and an intelligent orchestration layer where an agent-native LLM translates high-level human goals into complex, multi-agent workflows. We demonstrate the deployed nature of this system through a decentralized logistics use case where autonomous agents dynamically discover, negotiate, and transact with one another securely. Ultimately, the this http URL stack provides a principled architecture for moving beyond current agent implementations towards open, collaborative, and economically sustainable multi-agent ecosystems. 

---
# Exploring Membership Inference Vulnerabilities in Clinical Large Language Models 

**Authors**: Alexander Nemecek, Zebin Yun, Zahra Rahmani, Yaniv Harel, Vipin Chaudhary, Mahmood Sharif, Erman Ayday  

**Link**: [PDF](https://arxiv.org/pdf/2510.18674)  

**Abstract**: As large language models (LLMs) become progressively more embedded in clinical decision-support, documentation, and patient-information systems, ensuring their privacy and trustworthiness has emerged as an imperative challenge for the healthcare sector. Fine-tuning LLMs on sensitive electronic health record (EHR) data improves domain alignment but also raises the risk of exposing patient information through model behaviors. In this work-in-progress, we present an exploratory empirical study on membership inference vulnerabilities in clinical LLMs, focusing on whether adversaries can infer if specific patient records were used during model training. Using a state-of-the-art clinical question-answering model, Llemr, we evaluate both canonical loss-based attacks and a domain-motivated paraphrasing-based perturbation strategy that more realistically reflects clinical adversarial conditions. Our preliminary findings reveal limited but measurable membership leakage, suggesting that current clinical LLMs provide partial resistance yet remain susceptible to subtle privacy risks that could undermine trust in clinical AI adoption. These results motivate continued development of context-aware, domain-specific privacy evaluations and defenses such as differential privacy fine-tuning and paraphrase-aware training, to strengthen the security and trustworthiness of healthcare AI systems. 

---
# Reasoning Language Model Inference Serving Unveiled: An Empirical Study 

**Authors**: Qi Li, Junpan Wu, Xiang Liu, Yuxin Wang, Zeyu Li, Zhenheng Tang, Yuhan Chen, Shaohuai Shi, Xiaowen Chu  

**Link**: [PDF](https://arxiv.org/pdf/2510.18672)  

**Abstract**: The reasoning large language model (RLLM) has been proven competitive in solving complex reasoning tasks such as mathematics, coding, compared to general LLM. However, the serving performance and behavior of RLLM remains unexplored, which may undermine the deployment and utilization of RLLM in real-world scenario. To close this gap, in this paper, we conduct a comprehensive study of RLLM service. We first perform a pilot study on comparing the serving performance between RLLM and traditional LLM and reveal that there are several distinct differences regarding serving behavior: (1) significant memory usage and fluctuations; (2) straggler requests; (3) adaptive running time; (4) domain preference. Then we further investigate whether existing inference optimization techniques are valid for RLLM. Our main takeaways are that model quantization methods and speculative decoding can improve service system efficiency with small compromise to RLLM accuracy, while prefix caching, KV cache quantization may even degrade accuracy or serving performance for small RLLM. Lastly, we conduct evaluation under real world workload modeled by Gamma distribution to verify our findings. Empirical results of real world workload evaluation across different dataset are aligned with our main findings regarding RLLM serving. We hope our work can provide the research community and industry with insights to advance RLLM inference serving. 

---
# Binary Quadratic Quantization: Beyond First-Order Quantization for Real-Valued Matrix Compression 

**Authors**: Kyo Kuroki, Yasuyuki Okoshi, Thiem Van Chu, Kazushi Kawamura, Masato Motomura  

**Link**: [PDF](https://arxiv.org/pdf/2510.18650)  

**Abstract**: This paper proposes a novel matrix quantization method, Binary Quadratic Quantization (BQQ). In contrast to conventional first-order quantization approaches, such as uniform quantization and binary coding quantization, that approximate real-valued matrices via linear combinations of binary bases, BQQ leverages the expressive power of binary quadratic expressions while maintaining an extremely compact data format. We validate our approach with two experiments: a matrix compression benchmark and post-training quantization (PTQ) on pretrained Vision Transformer-based models. Experimental results demonstrate that BQQ consistently achieves a superior trade-off between memory efficiency and reconstruction error than conventional methods for compressing diverse matrix data. It also delivers strong PTQ performance, even though we neither target state-of-the-art PTQ accuracy under tight memory constraints nor rely on PTQ-specific binary matrix optimization. For example, our proposed method outperforms the state-of-the-art PTQ method by up to 2.2\% and 59.1% on the ImageNet dataset under the calibration-based and data-free scenarios, respectively, with quantization equivalent to 2 bits. These findings highlight the surprising effectiveness of binary quadratic expressions for efficient matrix approximation and neural network compression. 

---
# ε-Seg: Sparsely Supervised Semantic Segmentation of Microscopy Data 

**Authors**: Sheida Rahnamai Kordasiabi, Damian Dalle Nogare, Florian Jug  

**Link**: [PDF](https://arxiv.org/pdf/2510.18637)  

**Abstract**: Semantic segmentation of electron microscopy (EM) images of biological samples remains a challenge in the life sciences. EM data captures details of biological structures, sometimes with such complexity that even human observers can find it overwhelming. We introduce {\epsilon}-Seg, a method based on hierarchical variational autoencoders (HVAEs), employing center-region masking, sparse label contrastive learning (CL), a Gaussian mixture model (GMM) prior, and clustering-free label prediction. Center-region masking and the inpainting loss encourage the model to learn robust and representative embeddings to distinguish the desired classes, even if training labels are sparse (0.05% of the total image data or less). For optimal performance, we employ CL and a GMM prior to shape the latent space of the HVAE such that encoded input patches tend to cluster wrt. the semantic classes we wish to distinguish. Finally, instead of clustering latent embeddings for semantic segmentation, we propose a MLP semantic segmentation head to directly predict class labels from latent embeddings. We show empirical results of {\epsilon}-Seg and baseline methods on 2 dense EM datasets of biological tissues and demonstrate the applicability of our method also on fluorescence microscopy data. Our results show that {\epsilon}-Seg is capable of achieving competitive sparsely-supervised segmentation results on complex biological image data, even if only limited amounts of training labels are available. 

---
# C-SWAP: Explainability-Aware Structured Pruning for Efficient Neural Networks Compression 

**Authors**: Baptiste Bauvin, Loïc Baret, Ola Ahmad  

**Link**: [PDF](https://arxiv.org/pdf/2510.18636)  

**Abstract**: Neural network compression has gained increasing attention in recent years, particularly in computer vision applications, where the need for model reduction is crucial for overcoming deployment constraints. Pruning is a widely used technique that prompts sparsity in model structures, e.g. weights, neurons, and layers, reducing size and inference costs. Structured pruning is especially important as it allows for the removal of entire structures, which further accelerates inference time and reduces memory overhead. However, it can be computationally expensive, requiring iterative retraining and optimization. To overcome this problem, recent methods considered one-shot setting, which applies pruning directly at post-training. Unfortunately, they often lead to a considerable drop in performance. In this paper, we focus on this issue by proposing a novel one-shot pruning framework that relies on explainable deep learning. First, we introduce a causal-aware pruning approach that leverages cause-effect relations between model predictions and structures in a progressive pruning process. It allows us to efficiently reduce the size of the network, ensuring that the removed structures do not deter the performance of the model. Then, through experiments conducted on convolution neural network and vision transformer baselines, pre-trained on classification tasks, we demonstrate that our method consistently achieves substantial reductions in model size, with minimal impact on performance, and without the need for fine-tuning. Overall, our approach outperforms its counterparts, offering the best trade-off. Our code is available on GitHub. 

---
# Think with 3D: Geometric Imagination Grounded Spatial Reasoning from Limited Views 

**Authors**: Zhangquan Chen, Manyuan Zhang, Xinlei Yu, Xufang Luo, Mingze Sun, Zihao Pan, Yan Feng, Peng Pei, Xunliang Cai, Ruqi Huang  

**Link**: [PDF](https://arxiv.org/pdf/2510.18632)  

**Abstract**: Though recent advances in vision-language models (VLMs) have achieved remarkable progress across a wide range of multimodal tasks, understanding 3D spatial relationships from limited views remains a significant challenge. Previous reasoning methods typically rely on pure text (e.g., topological cognitive maps) or on 2D visual cues. However, their limited representational capacity hinders performance in specific tasks that require 3D spatial imagination. To address this limitation, we propose 3DThinker, a framework that can effectively exploits the rich geometric information embedded within images while reasoning, like humans do. Our framework is the first to enable 3D mentaling during reasoning without any 3D prior input, and it does not rely on explicitly labeled 3D data for training. Specifically, our training consists of two stages. First, we perform supervised training to align the 3D latent generated by VLM while reasoning with that of a 3D foundation model (e.g., VGGT). Then, we optimize the entire reasoning trajectory solely based on outcome signals, thereby refining the underlying 3D mentaling. Extensive experiments across multiple benchmarks show that 3DThinker consistently outperforms strong baselines and offers a new perspective toward unifying 3D representations into multimodal reasoning. Our code will be available at this https URL. 

---
# A Rectification-Based Approach for Distilling Boosted Trees into Decision Trees 

**Authors**: Gilles Audemard, Sylvie Coste-Marquis, Pierre Marquis, Mehdi Sabiri, Nicolas Szczepanski  

**Link**: [PDF](https://arxiv.org/pdf/2510.18615)  

**Abstract**: We present a new approach for distilling boosted trees into decision trees, in the objective of generating an ML model offering an acceptable compromise in terms of predictive performance and interpretability. We explain how the correction approach called rectification can be used to implement such a distillation process. We show empirically that this approach provides interesting results, in comparison with an approach to distillation achieved by retraining the model. 

---
# The Cost-Benefit of Interdisciplinarity in AI for Mental Health 

**Authors**: Katerina Drakos, Eva Paraschou, Simay Toplu, Line Harder Clemmensen, Christoph Lütge, Nicole Nadine Lønfeldt, Sneha Das  

**Link**: [PDF](https://arxiv.org/pdf/2510.18581)  

**Abstract**: Artificial intelligence has been introduced as a way to improve access to mental health support. However, most AI mental health chatbots rely on a limited range of disciplinary input, and fail to integrate expertise across the chatbot's lifecycle. This paper examines the cost-benefit trade-off of interdisciplinary collaboration in AI mental health chatbots. We argue that involving experts from technology, healthcare, ethics, and law across key lifecycle phases is essential to ensure value-alignment and compliance with the high-risk requirements of the AI Act. We also highlight practical recommendations and existing frameworks to help balance the challenges and benefits of interdisciplinarity in mental health chatbots. 

---
# Kaleido: Open-Sourced Multi-Subject Reference Video Generation Model 

**Authors**: Zhenxing Zhang, Jiayan Teng, Zhuoyi Yang, Tiankun Cao, Cheng Wang, Xiaotao Gu, Jie Tang, Dan Guo, Meng Wang  

**Link**: [PDF](https://arxiv.org/pdf/2510.18573)  

**Abstract**: We present Kaleido, a subject-to-video~(S2V) generation framework, which aims to synthesize subject-consistent videos conditioned on multiple reference images of target subjects. Despite recent progress in S2V generation models, existing approaches remain inadequate at maintaining multi-subject consistency and at handling background disentanglement, often resulting in lower reference fidelity and semantic drift under multi-image conditioning. These shortcomings can be attributed to several factors. Primarily, the training dataset suffers from a lack of diversity and high-quality samples, as well as cross-paired data, i.e., paired samples whose components originate from different instances. In addition, the current mechanism for integrating multiple reference images is suboptimal, potentially resulting in the confusion of multiple subjects. To overcome these limitations, we propose a dedicated data construction pipeline, incorporating low-quality sample filtering and diverse data synthesis, to produce consistency-preserving training data. Moreover, we introduce Reference Rotary Positional Encoding (R-RoPE) to process reference images, enabling stable and precise multi-image integration. Extensive experiments across numerous benchmarks demonstrate that Kaleido significantly outperforms previous methods in consistency, fidelity, and generalization, marking an advance in S2V generation. 

---
# Large language models for folktale type automation based on motifs: Cinderella case study 

**Authors**: Tjaša Arčon, Marko Robnik-Šikonja, Polona Tratnik  

**Link**: [PDF](https://arxiv.org/pdf/2510.18561)  

**Abstract**: Artificial intelligence approaches are being adapted to many research areas, including digital humanities. We built a methodology for large-scale analyses in folkloristics. Using machine learning and natural language processing, we automatically detected motifs in a large collection of Cinderella variants and analysed their similarities and differences with clustering and dimensionality reduction. The results show that large language models detect complex interactions in tales, enabling computational analysis of extensive text collections and facilitating cross-lingual comparisons. 

---
# WebDevJudge: Evaluating (M)LLMs as Critiques for Web Development Quality 

**Authors**: Chunyang Li, Yilun Zheng, Xinting Huang, Tianqing Fang, Jiahao Xu, Yangqiu Song, Lihui Chen, Han Hu  

**Link**: [PDF](https://arxiv.org/pdf/2510.18560)  

**Abstract**: The paradigm of LLM-as-a-judge is emerging as a scalable and efficient alternative to human evaluation, demonstrating strong performance on well-defined tasks. However, its reliability in open-ended tasks with dynamic environments and complex interactions remains unexplored. To bridge the gap, we introduce WebDevJudge, a systematic benchmark for assessing LLM-as-a-judge performance in web development, with support for both non-interactive evaluation based on static observations and continuous interactive evaluation with a dynamic web environment. WebDevJudge comprises human preference labels over paired web implementations, annotated with structured and query-grounded rubrics to ensure high-quality ground truth. Using this benchmark, we comprehensively evaluate various evaluators, including LLMs, MLLMs, and agentic workflows. We systematically investigate the impact of different paradigms and guidance mechanisms. Our experiments reveal a significant gap between LLM judges and human experts. In-depth analysis indicates this gap stems from fundamental model limitations, including failures in recognizing functional equivalence, verifying task feasibility, and mitigating bias. Overall, WebDevJudge presents a significant challenge to LLM-as-a-judge, offering insights to guide future research toward developing more reliable and capable automated evaluators for complicated scenarios. Code and data are available at this https URL. 

---
# RAISE: A Unified Framework for Responsible AI Scoring and Evaluation 

**Authors**: Loc Phuc Truong Nguyen, Hung Thanh Do  

**Link**: [PDF](https://arxiv.org/pdf/2510.18559)  

**Abstract**: As AI systems enter high-stakes domains, evaluation must extend beyond predictive accuracy to include explainability, fairness, robustness, and sustainability. We introduce RAISE (Responsible AI Scoring and Evaluation), a unified framework that quantifies model performance across these four dimensions and aggregates them into a single, holistic Responsibility Score. We evaluated three deep learning models: a Multilayer Perceptron (MLP), a Tabular ResNet, and a Feature Tokenizer Transformer, on structured datasets from finance, healthcare, and socioeconomics. Our findings reveal critical trade-offs: the MLP demonstrated strong sustainability and robustness, the Transformer excelled in explainability and fairness at a very high environmental cost, and the Tabular ResNet offered a balanced profile. These results underscore that no single model dominates across all responsibility criteria, highlighting the necessity of multi-dimensional evaluation for responsible model selection. Our implementation is available at: this https URL. 

---
# EfficientNav: Towards On-Device Object-Goal Navigation with Navigation Map Caching and Retrieval 

**Authors**: Zebin Yang, Sunjian Zheng, Tong Xie, Tianshi Xu, Bo Yu, Fan Wang, Jie Tang, Shaoshan Liu, Meng Li  

**Link**: [PDF](https://arxiv.org/pdf/2510.18546)  

**Abstract**: Object-goal navigation (ObjNav) tasks an agent with navigating to the location of a specific object in an unseen environment. Embodied agents equipped with large language models (LLMs) and online constructed navigation maps can perform ObjNav in a zero-shot manner. However, existing agents heavily rely on giant LLMs on the cloud, e.g., GPT-4, while directly switching to small LLMs, e.g., LLaMA3.2-11b, suffer from significant success rate drops due to limited model capacity for understanding complex navigation maps, which prevents deploying ObjNav on local devices. At the same time, the long prompt introduced by the navigation map description will cause high planning latency on local devices. In this paper, we propose EfficientNav to enable on-device efficient LLM-based zero-shot ObjNav. To help the smaller LLMs better understand the environment, we propose semantics-aware memory retrieval to prune redundant information in navigation maps. To reduce planning latency, we propose discrete memory caching and attention-based memory clustering to efficiently save and re-use the KV cache. Extensive experimental results demonstrate that EfficientNav achieves 11.1% improvement in success rate on HM3D benchmark over GPT-4-based baselines, and demonstrates 6.7x real-time latency reduction and 4.7x end-to-end latency reduction over GPT-4 planner. Our code will be released soon. 

---
# Pay Attention to the Triggers: Constructing Backdoors That Survive Distillation 

**Authors**: Giovanni De Muri, Mark Vero, Robin Staab, Martin Vechev  

**Link**: [PDF](https://arxiv.org/pdf/2510.18541)  

**Abstract**: LLMs are often used by downstream users as teacher models for knowledge distillation, compressing their capabilities into memory-efficient models. However, as these teacher models may stem from untrusted parties, distillation can raise unexpected security risks. In this paper, we investigate the security implications of knowledge distillation from backdoored teacher models. First, we show that prior backdoors mostly do not transfer onto student models. Our key insight is that this is because existing LLM backdooring methods choose trigger tokens that rarely occur in usual contexts. We argue that this underestimates the security risks of knowledge distillation and introduce a new backdooring technique, T-MTB, that enables the construction and study of transferable backdoors. T-MTB carefully constructs a composite backdoor trigger, made up of several specific tokens that often occur individually in anticipated distillation datasets. As such, the poisoned teacher remains stealthy, while during distillation the individual presence of these tokens provides enough signal for the backdoor to transfer onto the student. Using T-MTB, we demonstrate and extensively study the security risks of transferable backdoors across two attack scenarios, jailbreaking and content modulation, and across four model families of LLMs. 

---
# Zero-Shot Vehicle Model Recognition via Text-Based Retrieval-Augmented Generation 

**Authors**: Wei-Chia Chang, Yan-Ann Chen  

**Link**: [PDF](https://arxiv.org/pdf/2510.18502)  

**Abstract**: Vehicle make and model recognition (VMMR) is an important task in intelligent transportation systems, but existing approaches struggle to adapt to newly released models. Contrastive Language-Image Pretraining (CLIP) provides strong visual-text alignment, yet its fixed pretrained weights limit performance without costly image-specific finetuning. We propose a pipeline that integrates vision language models (VLMs) with Retrieval-Augmented Generation (RAG) to support zero-shot recognition through text-based reasoning. A VLM converts vehicle images into descriptive attributes, which are compared against a database of textual features. Relevant entries are retrieved and combined with the description to form a prompt, and a language model (LM) infers the make and model. This design avoids large-scale retraining and enables rapid updates by adding textual descriptions of new vehicles. Experiments show that the proposed method improves recognition by nearly 20% over the CLIP baseline, demonstrating the potential of RAG-enhanced LM reasoning for scalable VMMR in smart-city applications. 

---
# One Size Fits All? A Modular Adaptive Sanitization Kit (MASK) for Customizable Privacy-Preserving Phone Scam Detection 

**Authors**: Kangzhong Wang, Zitong Shen, Youqian Zhang, Michael MK Cheung, Xiapu Luo, Grace Ngai, Eugene Yujun Fu  

**Link**: [PDF](https://arxiv.org/pdf/2510.18493)  

**Abstract**: Phone scams remain a pervasive threat to both personal safety and financial security worldwide. Recent advances in large language models (LLMs) have demonstrated strong potential in detecting fraudulent behavior by analyzing transcribed phone conversations. However, these capabilities introduce notable privacy risks, as such conversations frequently contain sensitive personal information that may be exposed to third-party service providers during processing. In this work, we explore how to harness LLMs for phone scam detection while preserving user privacy. We propose MASK (Modular Adaptive Sanitization Kit), a trainable and extensible framework that enables dynamic privacy adjustment based on individual preferences. MASK provides a pluggable architecture that accommodates diverse sanitization methods - from traditional keyword-based techniques for high-privacy users to sophisticated neural approaches for those prioritizing accuracy. We also discuss potential modeling approaches and loss function designs for future development, enabling the creation of truly personalized, privacy-aware LLM-based detection systems that balance user trust and detection effectiveness, even beyond phone scam context. 

---
# Benchmarking Fairness-aware Graph Neural Networks in Knowledge Graphs 

**Authors**: Yuya Sasaki  

**Link**: [PDF](https://arxiv.org/pdf/2510.18473)  

**Abstract**: Graph neural networks (GNNs) are powerful tools for learning from graph-structured data but often produce biased predictions with respect to sensitive attributes. Fairness-aware GNNs have been actively studied for mitigating biased predictions. However, no prior studies have evaluated fairness-aware GNNs on knowledge graphs, which are one of the most important graphs in many applications, such as recommender systems. Therefore, we introduce a benchmarking study on knowledge graphs. We generate new graphs from three knowledge graphs, YAGO, DBpedia, and Wikidata, that are significantly larger than the existing graph datasets used in fairness studies. We benchmark inprocessing and preprocessing methods in different GNN backbones and early stopping conditions. We find several key insights: (i) knowledge graphs show different trends from existing datasets; clearer trade-offs between prediction accuracy and fairness metrics than other graphs in fairness-aware GNNs, (ii) the performance is largely affected by not only fairness-aware GNN methods but also GNN backbones and early stopping conditions, and (iii) preprocessing methods often improve fairness metrics, while inprocessing methods improve prediction accuracy. 

---
# CodeRL+: Improving Code Generation via Reinforcement with Execution Semantics Alignment 

**Authors**: Xue Jiang, Yihong Dong, Mengyang Liu, Hongyi Deng, Tian Wang, Yongding Tao, Rongyu Cao, Binhua Li, Zhi Jin, Wenpin Jiao, Fei Huang, Yongbin Li, Ge Li  

**Link**: [PDF](https://arxiv.org/pdf/2510.18471)  

**Abstract**: While Large Language Models (LLMs) excel at code generation by learning from vast code corpora, a fundamental semantic gap remains between their training on textual patterns and the goal of functional correctness, which is governed by formal execution semantics. Reinforcement Learning with Verifiable Rewards (RLVR) approaches attempt to bridge this gap using outcome rewards from executing test cases. However, solely relying on binary pass/fail signals is inefficient for establishing a well-aligned connection between the textual representation of code and its execution semantics, especially for subtle logical errors within the code. In this paper, we propose CodeRL+, a novel approach that integrates execution semantics alignment into the RLVR training pipeline for code generation. CodeRL+ enables the model to infer variable-level execution trajectory, providing a direct learning signal of execution semantics. CodeRL+ can construct execution semantics alignment directly using existing on-policy rollouts and integrates seamlessly with various RL algorithms. Extensive experiments demonstrate that CodeRL+ outperforms post-training baselines (including RLVR and Distillation), achieving a 4.6% average relative improvement in pass@1. CodeRL+ generalizes effectively to other coding tasks, yielding 15.5% and 4.4% higher accuracy on code-reasoning and test-output-generation benchmarks, respectively. CodeRL+ shows strong applicability across diverse RL algorithms and LLMs. Furthermore, probe analyses provide compelling evidence that CodeRL+ strengthens the alignment between code's textual representations and its underlying execution semantics. 

---
# Simple and Efficient Heterogeneous Temporal Graph Neural Network 

**Authors**: Yili Wang, Tairan Huang, Changlong He, Qiutong Li, Jianliang Gao  

**Link**: [PDF](https://arxiv.org/pdf/2510.18467)  

**Abstract**: Heterogeneous temporal graphs (HTGs) are ubiquitous data structures in the real world. Recently, to enhance representation learning on HTGs, numerous attention-based neural networks have been proposed. Despite these successes, existing methods rely on a decoupled temporal and spatial learning paradigm, which weakens interactions of spatio-temporal information and leads to a high model complexity. To bridge this gap, we propose a novel learning paradigm for HTGs called Simple and Efficient Heterogeneous Temporal Graph N}eural Network (SE-HTGNN). Specifically, we innovatively integrate temporal modeling into spatial learning via a novel dynamic attention mechanism, which retains attention information from historical graph snapshots to guide subsequent attention computation, thereby improving the overall discriminative representations learning of HTGs. Additionally, to comprehensively and adaptively understand HTGs, we leverage large language models to prompt SE-HTGNN, enabling the model to capture the implicit properties of node types as prior knowledge. Extensive experiments demonstrate that SE-HTGNN achieves up to 10x speed-up over the state-of-the-art and latest baseline while maintaining the best forecasting accuracy. 

---
# DeLoad: Demand-Driven Short-Video Preloading with Scalable Watch-Time Estimation 

**Authors**: Tong Liu, Zhiwei Fan, Guanyan Peng, Haodan Zhang, Yucheng Zhang, Zhen Wang, Pengjin Xie, Liang Liu  

**Link**: [PDF](https://arxiv.org/pdf/2510.18459)  

**Abstract**: Short video streaming has become a dominant paradigm in digital media, characterized by rapid swiping interactions and diverse media content. A key technical challenge is designing an effective preloading strategy that dynamically selects and prioritizes download tasks from an evolving playlist, balancing Quality of Experience (QoE) and bandwidth efficiency under practical commercial constraints. However, real world analysis reveals critical limitations of existing approaches: (1) insufficient adaptation of download task sizes to dynamic conditions, and (2) watch time prediction models that are difficult to deploy reliably at scale. In this paper, we propose DeLoad, a novel preloading framework that addresses these issues by introducing dynamic task sizing and a practical, multi dimensional watch time estimation method. Additionally, a Deep Reinforcement Learning (DRL) enhanced agent is trained to optimize the download range decisions adaptively. Extensive evaluations conducted on an offline testing platform, leveraging massive real world network data, demonstrate that DeLoad achieves significant improvements in QoE metrics (34.4% to 87.4% gain). Furthermore, after deployment on a large scale commercial short video platform, DeLoad has increased overall user watch time by 0.09% while simultaneously reducing rebuffering events and 3.76% bandwidth consumption. 

---
# ImageGem: In-the-wild Generative Image Interaction Dataset for Generative Model Personalization 

**Authors**: Yuanhe Guo, Linxi Xie, Zhuoran Chen, Kangrui Yu, Ryan Po, Guandao Yang, Gordon Wetztein, Hongyi Wen  

**Link**: [PDF](https://arxiv.org/pdf/2510.18433)  

**Abstract**: We introduce ImageGem, a dataset for studying generative models that understand fine-grained individual preferences. We posit that a key challenge hindering the development of such a generative model is the lack of in-the-wild and fine-grained user preference annotations. Our dataset features real-world interaction data from 57K users, who collectively have built 242K customized LoRAs, written 3M text prompts, and created 5M generated images. With user preference annotations from our dataset, we were able to train better preference alignment models. In addition, leveraging individual user preference, we investigated the performance of retrieval models and a vision-language model on personalized image retrieval and generative model recommendation. Finally, we propose an end-to-end framework for editing customized diffusion models in a latent weight space to align with individual user preferences. Our results demonstrate that the ImageGem dataset enables, for the first time, a new paradigm for generative model personalization. 

---
# ScaleNet: Scaling up Pretrained Neural Networks with Incremental Parameters 

**Authors**: Zhiwei Hao, Jianyuan Guo, Li Shen, Kai Han, Yehui Tang, Han Hu, Yunhe Wang  

**Link**: [PDF](https://arxiv.org/pdf/2510.18431)  

**Abstract**: Recent advancements in vision transformers (ViTs) have demonstrated that larger models often achieve superior performance. However, training these models remains computationally intensive and costly. To address this challenge, we introduce ScaleNet, an efficient approach for scaling ViT models. Unlike conventional training from scratch, ScaleNet facilitates rapid model expansion with negligible increases in parameters, building on existing pretrained models. This offers a cost-effective solution for scaling up ViTs. Specifically, ScaleNet achieves model expansion by inserting additional layers into pretrained ViTs, utilizing layer-wise weight sharing to maintain parameters efficiency. Each added layer shares its parameter tensor with a corresponding layer from the pretrained model. To mitigate potential performance degradation due to shared weights, ScaleNet introduces a small set of adjustment parameters for each layer. These adjustment parameters are implemented through parallel adapter modules, ensuring that each instance of the shared parameter tensor remains distinct and optimized for its specific function. Experiments on the ImageNet-1K dataset demonstrate that ScaleNet enables efficient expansion of ViT models. With a 2$\times$ depth-scaled DeiT-Base model, ScaleNet achieves a 7.42% accuracy improvement over training from scratch while requiring only one-third of the training epochs, highlighting its efficiency in scaling ViTs. Beyond image classification, our method shows significant potential for application in downstream vision areas, as evidenced by the validation in object detection task. 

---
# Optimistic Higher-Order Superposition 

**Authors**: Alexander Bentkamp, Jasmin Blanchette, Matthias Hetzenberger, Uwe Waldmann  

**Link**: [PDF](https://arxiv.org/pdf/2510.18429)  

**Abstract**: The $\lambda$-superposition calculus is a successful approach to proving higher-order formulas. However, some parts of the calculus are extremely explosive, notably due to the higher-order unifier enumeration and the functional extensionality axiom. In the present work, we introduce an "optimistic" version of $\lambda$-superposition that addresses these two issues. Specifically, our new calculus delays explosive unification problems using constraints stored along with the clauses, and it applies functional extensionality in a more targeted way. The calculus is sound and refutationally complete with respect to a Henkin semantics. We have yet to implement it in a prover, but examples suggest that it will outperform, or at least usefully complement, the original $\lambda$-superposition calculus. 

---
# On AI Verification in Open RAN 

**Authors**: Rahul Soundrarajan, Claudio Fiandrino, Michele Polese, Salvatore D'Oro, Leonardo Bonati, Tommaso Melodia  

**Link**: [PDF](https://arxiv.org/pdf/2510.18417)  

**Abstract**: Open RAN introduces a flexible, cloud-based architecture for the Radio Access Network (RAN), enabling Artificial Intelligence (AI)/Machine Learning (ML)-driven automation across heterogeneous, multi-vendor deployments. While EXplainable Artificial Intelligence (XAI) helps mitigate the opacity of AI models, explainability alone does not guarantee reliable network operations. In this article, we propose a lightweight verification approach based on interpretable models to validate the behavior of Deep Reinforcement Learning (DRL) agents for RAN slicing and scheduling in Open RAN. Specifically, we use Decision Tree (DT)-based verifiers to perform near-real-time consistency checks at runtime, which would be otherwise unfeasible with computationally expensive state-of-the-art verifiers. We analyze the landscape of XAI and AI verification, propose a scalable architectural integration, and demonstrate feasibility with a DT-based slice-verifier. We also outline future challenges to ensure trustworthy AI adoption in Open RAN. 

---
# Learning from N-Tuple Data with M Positive Instances: Unbiased Risk Estimation and Theoretical Guarantees 

**Authors**: Miao Zhang, Junpeng Li, ChangChun HUa, Yana Yang  

**Link**: [PDF](https://arxiv.org/pdf/2510.18406)  

**Abstract**: Weakly supervised learning often operates with coarse aggregate signals rather than instance labels. We study a setting where each training example is an $n$-tuple containing exactly m positives, while only the count m per tuple is observed. This NTMP (N-tuple with M positives) supervision arises in, e.g., image classification with region proposals and multi-instance measurements. We show that tuple counts admit a trainable unbiased risk estimator (URE) by linking the tuple-generation process to latent instance marginals. Starting from fixed (n,m), we derive a closed-form URE and extend it to variable tuple sizes, variable counts, and their combination. Identification holds whenever the effective mixing rate is separated from the class prior. We establish generalization bounds via Rademacher complexity and prove statistical consistency with standard rates under mild regularity assumptions. To improve finite-sample stability, we introduce simple ReLU corrections to the URE that preserve asymptotic correctness. Across benchmarks converted to NTMP tasks, the approach consistently outperforms representative weak-supervision baselines and yields favorable precision-recall and F1 trade-offs. It remains robust under class-prior imbalance and across diverse tuple configurations, demonstrating that count-only supervision can be exploited effectively through a theoretically grounded and practically stable objective. 

---
# Automated Wicket-Taking Delivery Segmentation and Weakness Detection in Cricket Videos Using OCR-Guided YOLOv8 and Trajectory Modeling 

**Authors**: Mst Jannatun Ferdous, Masum Billah, Joy Karmoker, Mohd Ruhul Ameen, Akif Islam, Md. Omar Faruqe  

**Link**: [PDF](https://arxiv.org/pdf/2510.18405)  

**Abstract**: This paper presents an automated system for cricket video analysis that leverages deep learning techniques to extract wicket-taking deliveries, detect cricket balls, and model ball trajectories. The system employs the YOLOv8 architecture for pitch and ball detection, combined with optical character recognition (OCR) for scorecard extraction to identify wicket-taking moments. Through comprehensive image preprocessing, including grayscale transformation, power transformation, and morphological operations, the system achieves robust text extraction from video frames. The pitch detection model achieved 99.5% mean Average Precision at 50% IoU (mAP50) with a precision of 0.999, while the ball detection model using transfer learning attained 99.18% mAP50 with 0.968 precision and 0.978 recall. The system enables trajectory modeling on detected pitches, providing data-driven insights for identifying batting weaknesses. Experimental results on multiple cricket match videos demonstrate the effectiveness of this approach for automated cricket analytics, offering significant potential for coaching and strategic decision-making. 

---
# MENTOR: A Reinforcement Learning Framework for Model Enhancement via Teacher-Optimized Rewards in Small Models 

**Authors**: ChangSu Choi, Hoyun Song, Dongyeon Kim, WooHyeon Jung, Minkyung Cho, Sunjin Park, NohHyeob Bae, Seona Yu, KyungTae Lim  

**Link**: [PDF](https://arxiv.org/pdf/2510.18383)  

**Abstract**: Distilling the tool-using capabilities of large language models (LLMs) into smaller, more efficient small language models (SLMs) is a key challenge for their practical application. The predominant approach, supervised fine-tuning (SFT), suffers from poor generalization as it trains models to imitate a static set of teacher trajectories rather than learn a robust methodology. While reinforcement learning (RL) offers an alternative, the standard RL using sparse rewards fails to effectively guide SLMs, causing them to struggle with inefficient exploration and adopt suboptimal strategies. To address these distinct challenges, we propose MENTOR, a framework that synergistically combines RL with teacher-guided distillation. Instead of simple imitation, MENTOR employs an RL-based process to learn a more generalizable policy through exploration. In addition, to solve the problem of reward sparsity, it uses a teacher's reference trajectory to construct a dense, composite teacher-guided reward that provides fine-grained guidance. Extensive experiments demonstrate that MENTOR significantly improves the cross-domain generalization and strategic competence of SLMs compared to both SFT and standard sparse-reward RL baselines. 

---
# S2AP: Score-space Sharpness Minimization for Adversarial Pruning 

**Authors**: Giorgio Piras, Qi Zhao, Fabio Brau, Maura Pintor, Christian Wressnegger, Battista Biggio  

**Link**: [PDF](https://arxiv.org/pdf/2510.18381)  

**Abstract**: Adversarial pruning methods have emerged as a powerful tool for compressing neural networks while preserving robustness against adversarial attacks. These methods typically follow a three-step pipeline: (i) pretrain a robust model, (ii) select a binary mask for weight pruning, and (iii) finetune the pruned model. To select the binary mask, these methods minimize a robust loss by assigning an importance score to each weight, and then keep the weights with the highest scores. However, this score-space optimization can lead to sharp local minima in the robust loss landscape and, in turn, to an unstable mask selection, reducing the robustness of adversarial pruning methods. To overcome this issue, we propose a novel plug-in method for adversarial pruning, termed Score-space Sharpness-aware Adversarial Pruning (S2AP). Through our method, we introduce the concept of score-space sharpness minimization, which operates during the mask search by perturbing importance scores and minimizing the corresponding robust loss. Extensive experiments across various datasets, models, and sparsity levels demonstrate that S2AP effectively minimizes sharpness in score space, stabilizing the mask selection, and ultimately improving the robustness of adversarial pruning methods. 

---
# PGTT: Phase-Guided Terrain Traversal for Perceptive Legged Locomotion 

**Authors**: Alexandros Ntagkas, Chairi Kiourt, Konstantinos Chatzilygeroudis  

**Link**: [PDF](https://arxiv.org/pdf/2510.18348)  

**Abstract**: State-of-the-art perceptive Reinforcement Learning controllers for legged robots either (i) impose oscillator or IK-based gait priors that constrain the action space, add bias to the policy optimization and reduce adaptability across robot morphologies, or (ii) operate "blind", which struggle to anticipate hind-leg terrain, and are brittle to noise. In this paper, we propose Phase-Guided Terrain Traversal (PGTT), a perception-aware deep-RL approach that overcomes these limitations by enforcing gait structure purely through reward shaping, thereby reducing inductive bias in policy learning compared to oscillator/IK-conditioned action priors. PGTT encodes per-leg phase as a cubic Hermite spline that adapts swing height to local heightmap statistics and adds a swing- phase contact penalty, while the policy acts directly in joint space supporting morphology-agnostic deployment. Trained in MuJoCo (MJX) on procedurally generated stair-like terrains with curriculum and domain randomization, PGTT achieves the highest success under push disturbances (median +7.5% vs. the next best method) and on discrete obstacles (+9%), with comparable velocity tracking, and converging to an effective policy roughly 2x faster than strong end-to-end baselines. We validate PGTT on a Unitree Go2 using a real-time LiDAR elevation-to-heightmap pipeline, and we report preliminary results on ANYmal-C obtained with the same hyperparameters. These findings indicate that terrain-adaptive, phase-guided reward shaping is a simple and general mechanism for robust perceptive locomotion across platforms. 

---
# Scalable, Explainable and Provably Robust Anomaly Detection with One-Step Flow Matching 

**Authors**: Zhong Li, Qi Huang, Yuxuan Zhu, Lincen Yang, Mohammad Mohammadi Amiri, Niki van Stein, Matthijs van Leeuwen  

**Link**: [PDF](https://arxiv.org/pdf/2510.18328)  

**Abstract**: We introduce Time-Conditioned Contraction Matching (TCCM), a novel method for semi-supervised anomaly detection in tabular data. TCCM is inspired by flow matching, a recent generative modeling framework that learns velocity fields between probability distributions and has shown strong performance compared to diffusion models and generative adversarial networks. Instead of directly applying flow matching as originally formulated, TCCM builds on its core idea -- learning velocity fields between distributions -- but simplifies the framework by predicting a time-conditioned contraction vector toward a fixed target (the origin) at each sampled time step. This design offers three key advantages: (1) a lightweight and scalable training objective that removes the need for solving ordinary differential equations during training and inference; (2) an efficient scoring strategy called one time-step deviation, which quantifies deviation from expected contraction behavior in a single forward pass, addressing the inference bottleneck of existing continuous-time models such as DTE (a diffusion-based model with leading anomaly detection accuracy but heavy inference cost); and (3) explainability and provable robustness, as the learned velocity field operates directly in input space, making the anomaly score inherently feature-wise attributable; moreover, the score function is Lipschitz-continuous with respect to the input, providing theoretical guarantees under small perturbations. Extensive experiments on the ADBench benchmark show that TCCM strikes a favorable balance between detection accuracy and inference cost, outperforming state-of-the-art methods -- especially on high-dimensional and large-scale datasets. The source code is available at our GitHub repository. 

---
# MoMaGen: Generating Demonstrations under Soft and Hard Constraints for Multi-Step Bimanual Mobile Manipulation 

**Authors**: Chengshu Li, Mengdi Xu, Arpit Bahety, Hang Yin, Yunfan Jiang, Huang Huang, Josiah Wong, Sujay Garlanka, Cem Gokmen, Ruohan Zhang, Weiyu Liu, Jiajun Wu, Roberto Martín-Martín, Li Fei-Fei  

**Link**: [PDF](https://arxiv.org/pdf/2510.18316)  

**Abstract**: Imitation learning from large-scale, diverse human demonstrations has proven effective for training robots, but collecting such data is costly and time-consuming. This challenge is amplified for multi-step bimanual mobile manipulation, where humans must teleoperate both a mobile base and two high-degree-of-freedom arms. Prior automated data generation frameworks have addressed static bimanual manipulation by augmenting a few human demonstrations in simulation, but they fall short for mobile settings due to two key challenges: (1) determining base placement to ensure reachability, and (2) positioning the camera to provide sufficient visibility for visuomotor policies. To address these issues, we introduce MoMaGen, which formulates data generation as a constrained optimization problem that enforces hard constraints (e.g., reachability) while balancing soft constraints (e.g., visibility during navigation). This formulation generalizes prior approaches and provides a principled foundation for future methods. We evaluate MoMaGen on four multi-step bimanual mobile manipulation tasks and show that it generates significantly more diverse datasets than existing methods. Leveraging this diversity, MoMaGen can train successful imitation learning policies from a single source demonstration, and these policies can be fine-tuned with as few as 40 real-world demonstrations to achieve deployment on physical robotic hardware. More details are available at our project page: this http URL. 

---
# Higher Embedding Dimension Creates a Stronger World Model for a Simple Sorting Task 

**Authors**: Brady Bhalla, Honglu Fan, Nancy Chen, Tony Yue YU  

**Link**: [PDF](https://arxiv.org/pdf/2510.18315)  

**Abstract**: We investigate how embedding dimension affects the emergence of an internal "world model" in a transformer trained with reinforcement learning to perform bubble-sort-style adjacent swaps. Models achieve high accuracy even with very small embedding dimensions, but larger dimensions yield more faithful, consistent, and robust internal representations. In particular, higher embedding dimensions strengthen the formation of structured internal representation and lead to better interpretability. After hundreds of experiments, we observe two consistent mechanisms: (1) the last row of the attention weight matrix monotonically encodes the global ordering of tokens; and (2) the selected transposition aligns with the largest adjacent difference of these encoded values. Our results provide quantitative evidence that transformers build structured internal world models and that model size improves representation quality in addition to end performance. We release our metrics and analyses, which can be used to probe similar algorithmic tasks. 

---
# From Retrieval to Generation: Unifying External and Parametric Knowledge for Medical Question Answering 

**Authors**: Lei Li, Xiao Zhou, Yingying Zhang, Xian Wu  

**Link**: [PDF](https://arxiv.org/pdf/2510.18297)  

**Abstract**: Medical question answering (QA) requires extensive access to domain-specific knowledge. A promising direction is to enhance large language models (LLMs) with external knowledge retrieved from medical corpora or parametric knowledge stored in model parameters. Existing approaches typically fall into two categories: Retrieval-Augmented Generation (RAG), which grounds model reasoning on externally retrieved evidence, and Generation-Augmented Generation (GAG), which depends solely on the models internal knowledge to generate contextual documents. However, RAG often suffers from noisy or incomplete retrieval, while GAG is vulnerable to hallucinated or inaccurate information due to unconstrained generation. Both issues can mislead reasoning and undermine answer reliability. To address these challenges, we propose MedRGAG, a unified retrieval-generation augmented framework that seamlessly integrates external and parametric knowledge for medical QA. MedRGAG comprises two key modules: Knowledge-Guided Context Completion (KGCC), which directs the generator to produce background documents that complement the missing knowledge revealed by retrieval; and Knowledge-Aware Document Selection (KADS), which adaptively selects an optimal combination of retrieved and generated documents to form concise yet comprehensive evidence for answer generation. Extensive experiments on five medical QA benchmarks demonstrate that MedRGAG achieves a 12.5% improvement over MedRAG and a 4.5% gain over MedGENIE, highlighting the effectiveness of unifying retrieval and generation for knowledge-intensive reasoning. Our code and data are publicly available at this https URL 

---
# Text or Pixels? It Takes Half: On the Token Efficiency of Visual Text Inputs in Multimodal LLMs 

**Authors**: Yanhong Li, Zixuan Lan, Jiawei Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2510.18279)  

**Abstract**: Large language models (LLMs) and their multimodal variants can now process visual inputs, including images of text. This raises an intriguing question: can we compress textual inputs by feeding them as images to reduce token usage while preserving performance? In this paper, we show that visual text representations are a practical and surprisingly effective form of input compression for decoder LLMs. We exploit the idea of rendering long text inputs as a single image and provide it directly to the model. This leads to dramatically reduced number of decoder tokens required, offering a new form of input compression. Through experiments on two distinct benchmarks RULER (long-context retrieval) and CNN/DailyMail (document summarization) we demonstrate that this text-as-image method yields substantial token savings (often nearly half) without degrading task performance. 

---
# StreamingTOM: Streaming Token Compression for Efficient Video Understanding 

**Authors**: Xueyi Chen, Keda Tao, Kele Shao, Huan Wang  

**Link**: [PDF](https://arxiv.org/pdf/2510.18269)  

**Abstract**: Unlike offline processing, streaming video vision-language models face two fundamental constraints: causality and accumulation. Causality prevents access to future frames that offline methods exploit, while accumulation causes tokens to grow unbounded, creating efficiency bottlenecks. However, existing approaches only regulate post-LLM kv-cache, leaving costly pre-LLM prefill unchanged. We introduce StreamingTOM, a training-free, plug-and-play two-stage framework that addresses both pre-LLM and post-LLM bottlenecks with predictable latency. Causal Temporal Reduction imposes a fixed per-frame budget and selects tokens based on adjacent-frame changes and token saliency, drastically reducing per-frame prefill cost by processing only a compact subset of visual tokens per frame instead of all visual tokens. Online Quantized Memory stores tokens in 4-bit format, retrieves relevant groups on demand, and dequantizes them, keeping the active kv-cache bounded regardless of stream length. Experiments demonstrate our method achieves $15.7\times$ kv-cache compression, $1.2\times$ lower peak memory and $2\times$ faster TTFT compared to prior SOTA. StreamingTOM maintains state-of-the-art accuracy among training-free methods with an average of $63.8\%$ on offline benchmarks and $55.8\%/3.7$ on RVS. These results highlight the practical benefits of our two-stage approach for efficient streaming video understanding with bounded growth. 

---
# Latent-Info and Low-Dimensional Learning for Human Mesh Recovery and Parallel Optimization 

**Authors**: Xiang Zhang, Suping Wu, Sheng Yang  

**Link**: [PDF](https://arxiv.org/pdf/2510.18267)  

**Abstract**: Existing 3D human mesh recovery methods often fail to fully exploit the latent information (e.g., human motion, shape alignment), leading to issues with limb misalignment and insufficient local details in the reconstructed human mesh (especially in complex scenes). Furthermore, the performance improvement gained by modelling mesh vertices and pose node interactions using attention mechanisms comes at a high computational cost. To address these issues, we propose a two-stage network for human mesh recovery based on latent information and low dimensional learning. Specifically, the first stage of the network fully excavates global (e.g., the overall shape alignment) and local (e.g., textures, detail) information from the low and high-frequency components of image features and aggregates this information into a hybrid latent frequency domain feature. This strategy effectively extracts latent information. Subsequently, utilizing extracted hybrid latent frequency domain features collaborates to enhance 2D poses to 3D learning. In the second stage, with the assistance of hybrid latent features, we model the interaction learning between the rough 3D human mesh template and the 3D pose, optimizing the pose and shape of the human mesh. Unlike existing mesh pose interaction methods, we design a low-dimensional mesh pose interaction method through dimensionality reduction and parallel optimization that significantly reduces computational costs without sacrificing reconstruction accuracy. Extensive experimental results on large publicly available datasets indicate superiority compared to the most state-of-the-art. 

---
# SPIKE: Stable Physics-Informed Kernel Evolution Method for Solving Hyperbolic Conservation Laws 

**Authors**: Hua Su, Lei Zhang, Jin Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2510.18266)  

**Abstract**: We introduce the Stable Physics-Informed Kernel Evolution (SPIKE) method for numerical computation of inviscid hyperbolic conservation laws. SPIKE resolves a fundamental paradox: how strong-form residual minimization can capture weak solutions containing discontinuities. SPIKE employs reproducing kernel representations with regularized parameter evolution, where Tikhonov regularization provides a smooth transition mechanism through shock formation, allowing the dynamics to traverse shock singularities. This approach automatically maintains conservation, tracks characteristics, and captures shocks satisfying Rankine-Hugoniot conditions within a unified framework requiring no explicit shock detection or artificial viscosity. Numerical validation across scalar and vector-valued conservation laws confirms the method's effectiveness. 

---
# Learning under Quantization for High-Dimensional Linear Regression 

**Authors**: Dechen Zhang, Junwei Su, Difan Zou  

**Link**: [PDF](https://arxiv.org/pdf/2510.18259)  

**Abstract**: The use of low-bit quantization has emerged as an indispensable technique for enabling the efficient training of large-scale models. Despite its widespread empirical success, a rigorous theoretical understanding of its impact on learning performance remains notably absent, even in the simplest linear regression setting. We present the first systematic theoretical study of this fundamental question, analyzing finite-step stochastic gradient descent (SGD) for high-dimensional linear regression under a comprehensive range of quantization targets: data, labels, parameters, activations, and gradients. Our novel analytical framework establishes precise algorithm-dependent and data-dependent excess risk bounds that characterize how different quantization affects learning: parameter, activation, and gradient quantization amplify noise during training; data quantization distorts the data spectrum; and data and label quantization introduce additional approximation and quantized error. Crucially, we prove that for multiplicative quantization (with input-dependent quantization step), this spectral distortion can be eliminated, and for additive quantization (with constant quantization step), a beneficial scaling effect with batch size emerges. Furthermore, for common polynomial-decay data spectra, we quantitatively compare the risks of multiplicative and additive quantization, drawing a parallel to the comparison between FP and integer quantization methods. Our theory provides a powerful lens to characterize how quantization shapes the learning dynamics of optimization algorithms, paving the way to further explore learning theory under practical hardware constraints. 

---
# NTKMTL: Mitigating Task Imbalance in Multi-Task Learning from Neural Tangent Kernel Perspective 

**Authors**: Xiaohan Qin, Xiaoxing Wang, Ning Liao, Junchi Yan  

**Link**: [PDF](https://arxiv.org/pdf/2510.18258)  

**Abstract**: Multi-Task Learning (MTL) enables a single model to learn multiple tasks simultaneously, leveraging knowledge transfer among tasks for enhanced generalization, and has been widely applied across various domains. However, task imbalance remains a major challenge in MTL. Although balancing the convergence speeds of different tasks is an effective approach to address this issue, it is highly challenging to accurately characterize the training dynamics and convergence speeds of multiple tasks within the complex MTL system. To this end, we attempt to analyze the training dynamics in MTL by leveraging Neural Tangent Kernel (NTK) theory and propose a new MTL method, NTKMTL. Specifically, we introduce an extended NTK matrix for MTL and adopt spectral analysis to balance the convergence speeds of multiple tasks, thereby mitigating task imbalance. Based on the approximation via shared representation, we further propose NTKMTL-SR, achieving training efficiency while maintaining competitive performance. Extensive experiments demonstrate that our methods achieve state-of-the-art performance across a wide range of benchmarks, including both multi-task supervised learning and multi-task reinforcement learning. Source code is available at this https URL. 

---
# DelvePO: Direction-Guided Self-Evolving Framework for Flexible Prompt Optimization 

**Authors**: Tao Tao, Guanghui Zhu, Lang Guo, Hongyi Chen, Chunfeng Yuan, Yihua Huang  

**Link**: [PDF](https://arxiv.org/pdf/2510.18257)  

**Abstract**: Prompt Optimization has emerged as a crucial approach due to its capabilities in steering Large Language Models to solve various tasks. However, current works mainly rely on the random rewriting ability of LLMs, and the optimization process generally focus on specific influencing factors, which makes it easy to fall into local optimum. Besides, the performance of the optimized prompt is often unstable, which limits its transferability in different tasks. To address the above challenges, we propose $\textbf{DelvePO}$ ($\textbf{D}$irection-Guid$\textbf{e}$d Se$\textbf{l}$f-E$\textbf{v}$olving Framework for Fl$\textbf{e}$xible $\textbf{P}$rompt $\textbf{O}$ptimization), a task-agnostic framework to optimize prompts in self-evolve manner. In our framework, we decouple prompts into different components that can be used to explore the impact that different factors may have on various tasks. On this basis, we introduce working memory, through which LLMs can alleviate the deficiencies caused by their own uncertainties and further obtain key insights to guide the generation of new prompts. Extensive experiments conducted on different tasks covering various domains for both open- and closed-source LLMs, including DeepSeek-R1-Distill-Llama-8B, Qwen2.5-7B-Instruct and GPT-4o-mini. Experimental results show that DelvePO consistently outperforms previous SOTA methods under identical experimental settings, demonstrating its effectiveness and transferability across different tasks. 

---
# Hyperbolic Space Learning Method Leveraging Temporal Motion Priors for Human Mesh Recovery 

**Authors**: Xiang Zhang, Suping Wu, Weibin Qiu, Zhaocheng Jin, Sheng Yang  

**Link**: [PDF](https://arxiv.org/pdf/2510.18256)  

**Abstract**: 3D human meshes show a natural hierarchical structure (like torso-limbs-fingers). But existing video-based 3D human mesh recovery methods usually learn mesh features in Euclidean space. It's hard to catch this hierarchical structure accurately. So wrong human meshes are reconstructed. To solve this problem, we propose a hyperbolic space learning method leveraging temporal motion prior for recovering 3D human meshes from videos. First, we design a temporal motion prior extraction module. This module extracts the temporal motion features from the input 3D pose sequences and image feature sequences respectively. Then it combines them into the temporal motion prior. In this way, it can strengthen the ability to express features in the temporal motion dimension. Since data representation in non-Euclidean space has been proved to effectively capture hierarchical relationships in real-world datasets (especially in hyperbolic space), we further design a hyperbolic space optimization learning strategy. This strategy uses the temporal motion prior information to assist learning, and uses 3D pose and pose motion information respectively in the hyperbolic space to optimize and learn the mesh features. Then, we combine the optimized results to get an accurate and smooth human mesh. Besides, to make the optimization learning process of human meshes in hyperbolic space stable and effective, we propose a hyperbolic mesh optimization loss. Extensive experimental results on large publicly available datasets indicate superiority in comparison with most state-of-the-art. 

---
# Finding the Sweet Spot: Optimal Data Augmentation Ratio for Imbalanced Credit Scoring Using ADASYN 

**Authors**: Luis H. Chia  

**Link**: [PDF](https://arxiv.org/pdf/2510.18252)  

**Abstract**: Credit scoring models face a critical challenge: severe class imbalance, with default rates typically below 10%, which hampers model learning and predictive performance. While synthetic data augmentation techniques such as SMOTE and ADASYN have been proposed to address this issue, the optimal augmentation ratio remains unclear, with practitioners often defaulting to full balancing (1:1 ratio) without empirical justification.
This study systematically evaluates 10 data augmentation scenarios using the Give Me Some Credit dataset (97,243 observations, 7% default rate), comparing SMOTE, BorderlineSMOTE, and ADASYN at different multiplication factors (1x, 2x, 3x). All models were trained using XGBoost and evaluated on a held-out test set of 29,173 real observations. Statistical significance was assessed using bootstrap testing with 1,000 iterations.
Key findings reveal that ADASYN with 1x multiplication (doubling the minority class) achieved optimal performance with AUC of 0.6778 and Gini coefficient of 0.3557, representing statistically significant improvements of +0.77% and +3.00% respectively (p = 0.017, bootstrap test). Higher multiplication factors (2x and 3x) resulted in performance degradation, with 3x showing a -0.48% decrease in AUC, suggesting a "law of diminishing returns" for synthetic oversampling. The optimal class imbalance ratio was found to be 6.6:1 (majority:minority), contradicting the common practice of balancing to 1:1.
This work provides the first empirical evidence of an optimal "sweet spot" for data augmentation in credit scoring, with practical guidelines for industry practitioners and researchers working with imbalanced datasets. While demonstrated on a single representative dataset, the methodology provides a reproducible framework for determining optimal augmentation ratios in other imbalanced domains. 

---
# Scaling Laws Meet Model Architecture: Toward Inference-Efficient LLMs 

**Authors**: Song Bian, Tao Yu, Shivaram Venkataraman, Youngsuk Park  

**Link**: [PDF](https://arxiv.org/pdf/2510.18245)  

**Abstract**: Scaling the number of parameters and the size of training data has proven to be an effective strategy for improving large language model (LLM) performance. Yet, as these models grow increasingly powerful and widely deployed, the cost of inference has become a pressing concern. Despite its importance, the trade-off between model accuracy and inference efficiency remains underexplored. In this work, we examine how key architectural factors, hidden size, the allocation of parameters between MLP and attention (mlp-to-attention ratio), and grouped-query attention (GQA), influence both inference cost and accuracy. We introduce a conditional scaling law that augments the Chinchilla framework with architectural information, along with a search framework for identifying architectures that are simultaneously inference-efficient and accurate. To validate our approach, we train more than 200 models spanning 80M to 3B parameters and 8B to 100B training tokens, and fit the proposed conditional scaling law. Our results show that the conditional scaling law reliably predicts optimal architectural choices and that the resulting models outperform existing open-source baselines. Under the same training budget, optimized architectures achieve up to 2.1% higher accuracy and 42% greater inference throughput compared to LLaMA-3.2. 

---
# EVER: Edge-Assisted Auto-Verification for Mobile MR-Aided Operation 

**Authors**: Jiangong Chen, Mingyu Zhu, Bin Li  

**Link**: [PDF](https://arxiv.org/pdf/2510.18224)  

**Abstract**: Mixed Reality (MR)-aided operation overlays digital objects on the physical world to provide a more immersive and intuitive operation process. A primary challenge is the precise and fast auto-verification of whether the user follows MR guidance by comparing frames before and after each operation. The pre-operation frame includes virtual guiding objects, while the post-operation frame contains physical counterparts. Existing approaches fall short of accounting for the discrepancies between physical and virtual objects due to imperfect 3D modeling or lighting estimation. In this paper, we propose EVER: an edge-assisted auto-verification system for mobile MR-aided operations. Unlike traditional frame-based similarity comparisons, EVER leverages the segmentation model and rendering pipeline adapted to the unique attributes of frames with physical pieces and those with their virtual counterparts; it adopts a threshold-based strategy using Intersection over Union (IoU) metrics for accurate auto-verification. To ensure fast auto-verification and low energy consumption, EVER offloads compute-intensive tasks to an edge server. Through comprehensive evaluations of public datasets and custom datasets with practical implementation, EVER achieves over 90% verification accuracy within 100 milliseconds (significantly faster than average human reaction time of approximately 273 milliseconds), while consuming only minimal additional computational resources and energy compared to a system without auto-verification. 

---
# The Emergence of Complex Behavior in Large-Scale Ecological Environments 

**Authors**: Joseph Bejjani, Chase Van Amburg, Chengrui Wang, Chloe Huangyuan Su, Sarah M. Pratt, Yasin Mazloumi, Naeem Khoshnevis, Sham M. Kakade, Kianté Brantley  

**Link**: [PDF](https://arxiv.org/pdf/2510.18221)  

**Abstract**: We explore how physical scale and population size shape the emergence of complex behaviors in open-ended ecological environments. In our setting, agents are unsupervised and have no explicit rewards or learning objectives but instead evolve over time according to reproduction, mutation, and natural selection. As they act, agents also shape their environment and the population around them in an ongoing dynamic ecology. Our goal is not to optimize a single high-performance policy, but instead to examine how behaviors emerge and evolve across large populations due to natural competition and environmental pressures. In an effort to discover how complex behaviors naturally emerge, we conduct experiments in large-scale worlds that reach populations of more than 60,000 individual agents, each with their own evolved neural network policy. We identify various emergent behaviors such as long-range resource extraction, vision-based foraging, and predation that arise under competitive and survival pressures. We examine how sensing modalities and environmental scale affect the emergence of these behaviors, finding that some appear only in sufficiently large environments and populations, with larger scales increasing behavioral stability and consistency. While there is a rich history of research in evolutionary settings, our scaling results provide promising new directions to explore ecology as an instrument of machine learning in an era of abundant computational resources. Experimental code is available at this https URL. 

---
# VLSU: Mapping the Limits of Joint Multimodal Understanding for AI Safety 

**Authors**: Shruti Palaskar, Leon Gatys, Mona Abdelrahman, Mar Jacobo, Larry Lindsey, Rutika Moharir, Gunnar Lund, Yang Xu, Navid Shiee, Jeffrey Bigham, Charles Maalouf, Joseph Yitan Cheng  

**Link**: [PDF](https://arxiv.org/pdf/2510.18214)  

**Abstract**: Safety evaluation of multimodal foundation models often treats vision and language inputs separately, missing risks from joint interpretation where benign content becomes harmful in combination. Existing approaches also fail to distinguish clearly unsafe content from borderline cases, leading to problematic over-blocking or under-refusal of genuinely harmful content. We present Vision Language Safety Understanding (VLSU), a comprehensive framework to systematically evaluate multimodal safety through fine-grained severity classification and combinatorial analysis across 17 distinct safety patterns. Using a multi-stage pipeline with real-world images and human annotation, we construct a large-scale benchmark of 8,187 samples spanning 15 harm categories. Our evaluation of eleven state-of-the-art models reveals systematic joint understanding failures: while models achieve 90%-plus accuracy on clear unimodal safety signals, performance degrades substantially to 20-55% when joint image-text reasoning is required to determine the safety label. Most critically, 34% of errors in joint image-text safety classification occur despite correct classification of the individual modalities, further demonstrating absent compositional reasoning capabilities. Additionally, we find that models struggle to balance refusing unsafe content while still responding to borderline cases that deserve engagement. For example, we find that instruction framing can reduce the over-blocking rate on borderline content from 62.4% to 10.4% in Gemini-1.5, but only at the cost of under-refusing on unsafe content with refusal rate dropping from 90.8% to 53.9%. Overall, our framework exposes weaknesses in joint image-text understanding and alignment gaps in current models, and provides a critical test bed to enable the next milestones in research on robust vision-language safety. 

---
# Contrastive Decoding Mitigates Score Range Bias in LLM-as-a-Judge 

**Authors**: Yoshinari Fujinuma  

**Link**: [PDF](https://arxiv.org/pdf/2510.18196)  

**Abstract**: Large Language Models (LLMs) are commonly used as evaluators in various applications, but the reliability of the outcomes remains a challenge. One such challenge is using LLMs-as-judges for direct assessment, i.e., assigning scores from a specified range without any references. We first show that this challenge stems from LLM judge outputs being associated with score range bias, i.e., LLM judge outputs are highly sensitive to pre-defined score ranges, preventing the search for optimal score ranges. We also show that similar biases exist among models from the same family. We then mitigate this bias through contrastive decoding, achieving up to 11.3% relative improvement on average in Spearman correlation with human judgments across different score ranges. 

---
# RadDiagSeg-M: A Vision Language Model for Joint Diagnosis and Multi-Target Segmentation in Radiology 

**Authors**: Chengrun Li, Corentin Royer, Haozhe Luo, Bastian Wittmann, Xia Li, Ibrahim Hamamci, Sezgin Er, Anjany Sekuboyina, Bjoern Menze  

**Link**: [PDF](https://arxiv.org/pdf/2510.18188)  

**Abstract**: Most current medical vision language models struggle to jointly generate diagnostic text and pixel-level segmentation masks in response to complex visual questions. This represents a major limitation towards clinical application, as assistive systems that fail to provide both modalities simultaneously offer limited value to medical practitioners. To alleviate this limitation, we first introduce RadDiagSeg-D, a dataset combining abnormality detection, diagnosis, and multi-target segmentation into a unified and hierarchical task. RadDiagSeg-D covers multiple imaging modalities and is precisely designed to support the development of models that produce descriptive text and corresponding segmentation masks in tandem. Subsequently, we leverage the dataset to propose a novel vision-language model, RadDiagSeg-M, capable of joint abnormality detection, diagnosis, and flexible segmentation. RadDiagSeg-M provides highly informative and clinically useful outputs, effectively addressing the need to enrich contextual information for assistive diagnosis. Finally, we benchmark RadDiagSeg-M and showcase its strong performance across all components involved in the task of multi-target text-and-mask generation, establishing a robust and competitive baseline. 

---
# VelocityNet: Real-Time Crowd Anomaly Detection via Person-Specific Velocity Analysis 

**Authors**: Fatima AlGhamdi, Omar Alharbi, Abdullah Aldwyish, Raied Aljadaany, Muhammad Kamran J Khan, Huda Alamri  

**Link**: [PDF](https://arxiv.org/pdf/2510.18187)  

**Abstract**: Detecting anomalies in crowded scenes is challenging due to severe inter-person occlusions and highly dynamic, context-dependent motion patterns. Existing approaches often struggle to adapt to varying crowd densities and lack interpretable anomaly indicators. To address these limitations, we introduce VelocityNet, a dual-pipeline framework that combines head detection and dense optical flow to extract person-specific velocities. Hierarchical clustering categorizes these velocities into semantic motion classes (halt, slow, normal, and fast), and a percentile-based anomaly scoring system measures deviations from learned normal patterns. Experiments demonstrate the effectiveness of our framework in real-time detection of diverse anomalous motion patterns within densely crowded environments. 

---
# ActivationReasoning: Logical Reasoning in Latent Activation Spaces 

**Authors**: Lukas Helff, Ruben Härle, Wolfgang Stammer, Felix Friedrich, Manuel Brack, Antonia Wüst, Hikaru Shindo, Patrick Schramowski, Kristian Kersting  

**Link**: [PDF](https://arxiv.org/pdf/2510.18184)  

**Abstract**: Large language models (LLMs) excel at generating fluent text, but their internal reasoning remains opaque and difficult to control. Sparse autoencoders (SAEs) make hidden activations more interpretable by exposing latent features that often align with human concepts. Yet, these features are fragile and passive, offering no mechanism for systematic reasoning or model control. To address this, we introduce ActivationReasoning (AR), a framework that embeds explicit logical reasoning into the latent space of LLMs. It proceeds in three stages: (1) Finding latent representations, first latent concept representations are identified (e.g., via SAEs) and organized into a dictionary; (2) Activating propositions, at inference time AR detects activating concepts and maps them to logical propositions; and (3)Logical reasoning, applying logical rules over these propositions to infer higher-order structures, compose new concepts, and steer model behavior. We evaluate AR on multi-hop reasoning (PrOntoQA), abstraction and robustness to indirect concept cues (Rail2Country), reasoning over natural and diverse language (ProverQA), and context-sensitive safety (BeaverTails). Across all tasks, AR scales robustly with reasoning complexity, generalizes to abstract and context-sensitive tasks, and transfers across model backbones. These results demonstrate that grounding logical structure in latent activations not only improves transparency but also enables structured reasoning, reliable control, and alignment with desired behaviors, providing a path toward more reliable and auditable AI. 

---
# Automatic Prompt Generation via Adaptive Selection of Prompting Techniques 

**Authors**: Yohei Ikenoue, Hitomi Tashiro, Shigeru Kuroyanagi  

**Link**: [PDF](https://arxiv.org/pdf/2510.18162)  

**Abstract**: Prompt engineering is crucial for achieving reliable and effective outputs from large language models (LLMs), but its design requires specialized knowledge of prompting techniques and a deep understanding of target tasks. To address this challenge, we propose a novel method that adaptively selects task-appropriate prompting techniques based on users' abstract task descriptions and automatically generates high-quality prompts without relying on pre-existing templates or frameworks. The proposed method constructs a knowledge base that associates task clusters, characterized by semantic similarity across diverse tasks, with their corresponding prompting techniques. When users input task descriptions, the system assigns them to the most relevant task cluster and dynamically generates prompts by integrating techniques drawn from the knowledge base. An experimental evaluation of the proposed method on 23 tasks from BIG-Bench Extra Hard (BBEH) demonstrates superior performance compared with standard prompts and existing automatic prompt-generation tools, as measured by both arithmetic and harmonic mean scores. This research establishes a foundation for streamlining and standardizing prompt creation, enabling non-experts to effectively leverage LLMs. 

---
# SafeCoop: Unravelling Full Stack Safety in Agentic Collaborative Driving 

**Authors**: Xiangbo Gao, Tzu-Hsiang Lin, Ruojing Song, Yuheng Wu, Kuan-Ru Huang, Zicheng Jin, Fangzhou Lin, Shinan Liu, Zhengzhong Tu  

**Link**: [PDF](https://arxiv.org/pdf/2510.18123)  

**Abstract**: Collaborative driving systems leverage vehicle-to-everything (V2X) communication across multiple agents to enhance driving safety and efficiency. Traditional V2X systems take raw sensor data, neural features, or perception results as communication media, which face persistent challenges, including high bandwidth demands, semantic loss, and interoperability issues. Recent advances investigate natural language as a promising medium, which can provide semantic richness, decision-level reasoning, and human-machine interoperability at significantly lower bandwidth. Despite great promise, this paradigm shift also introduces new vulnerabilities within language communication, including message loss, hallucinations, semantic manipulation, and adversarial attacks. In this work, we present the first systematic study of full-stack safety and security issues in natural-language-based collaborative driving. Specifically, we develop a comprehensive taxonomy of attack strategies, including connection disruption, relay/replay interference, content spoofing, and multi-connection forgery. To mitigate these risks, we introduce an agentic defense pipeline, which we call SafeCoop, that integrates a semantic firewall, language-perception consistency checks, and multi-source consensus, enabled by an agentic transformation function for cross-frame spatial alignment. We systematically evaluate SafeCoop in closed-loop CARLA simulation across 32 critical scenarios, achieving 69.15% driving score improvement under malicious attacks and up to 67.32% F1 score for malicious detection. This study provides guidance for advancing research on safe, secure, and trustworthy language-driven collaboration in transportation systems. Our project page is this https URL. 

---
# Latent Discrete Diffusion Models 

**Authors**: Dario Shariatian, Alain Durmus, Stefano Peluchetti  

**Link**: [PDF](https://arxiv.org/pdf/2510.18114)  

**Abstract**: We study discrete diffusion for language and other categorical data and focus on a common limitation of masked denoisers: reverse transitions typically factorize across positions, which can weaken joint structure and degrade quality in few-step generation. We propose \emph{Latent Discrete Diffusion Models} (LDDMs), which couple a masked discrete diffusion over tokens with a continuous diffusion over latent embeddings. The latent channel provides a softer signal and carries cross-token dependencies that help resolve ambiguities. We present two instantiations: (i) FUJI-LDDMs, which perform fully joint denoising of tokens and latents, and (ii) SEQ-LDDMs, which sequentially resolve the latent and then the discrete chain conditionally on it. For both variants we derive ELBO-style objectives and discuss design choices to learn informative latents yet amenable to diffusoin modeling. In experiments, LDDMs yield improvements on unconditional generation metrics as compared to state-of-the-art masked discrete diffusion baselines, and are effective at lower sampling budgets, where unmasking many tokens per step is desirable. 

---
# From AutoRecSys to AutoRecLab: A Call to Build, Evaluate, and Govern Autonomous Recommender-Systems Research Labs 

**Authors**: Joeran Beel, Bela Gipp, Tobias Vente, Moritz Baumgart, Philipp Meister  

**Link**: [PDF](https://arxiv.org/pdf/2510.18104)  

**Abstract**: Recommender-systems research has accelerated model and evaluation advances, yet largely neglects automating the research process itself. We argue for a shift from narrow AutoRecSys tools -- focused on algorithm selection and hyper-parameter tuning -- to an Autonomous Recommender-Systems Research Lab (AutoRecLab) that integrates end-to-end automation: problem ideation, literature analysis, experimental design and execution, result interpretation, manuscript drafting, and provenance logging. Drawing on recent progress in automated science (e.g., multi-agent AI Scientist and AI Co-Scientist systems), we outline an agenda for the RecSys community: (1) build open AutoRecLab prototypes that combine LLM-driven ideation and reporting with automated experimentation; (2) establish benchmarks and competitions that evaluate agents on producing reproducible RecSys findings with minimal human input; (3) create review venues for transparently AI-generated submissions; (4) define standards for attribution and reproducibility via detailed research logs and metadata; and (5) foster interdisciplinary dialogue on ethics, governance, privacy, and fairness in autonomous research. Advancing this agenda can increase research throughput, surface non-obvious insights, and position RecSys to contribute to emerging Artificial Research Intelligence. We conclude with a call to organise a community retreat to coordinate next steps and co-author guidance for the responsible integration of automated research systems. 

---
# Enhancing mortality prediction in cardiac arrest ICU patients through meta-modeling of structured clinical data from MIMIC-IV 

**Authors**: Nursultan Mamatov, Philipp Kellmeyer  

**Link**: [PDF](https://arxiv.org/pdf/2510.18103)  

**Abstract**: Accurate early prediction of in-hospital mortality in intensive care units (ICUs) is essential for timely clinical intervention and efficient resource allocation. This study develops and evaluates machine learning models that integrate both structured clinical data and unstructured textual information, specifically discharge summaries and radiology reports, from the MIMIC-IV database. We used LASSO and XGBoost for feature selection, followed by a multivariate logistic regression trained on the top features identified by both models. Incorporating textual features using TF-IDF and BERT embeddings significantly improved predictive performance. The final logistic regression model, which combined structured and textual input, achieved an AUC of 0.918, compared to 0.753 when using structured data alone, a relative improvement 22%. The analysis of the decision curve demonstrated a superior standardized net benefit in a wide range of threshold probabilities (0.2-0.8), confirming the clinical utility of the model. These results underscore the added prognostic value of unstructured clinical notes and support their integration into interpretable feature-driven risk prediction models for ICU patients. 

---
# Accelerating Vision Transformers with Adaptive Patch Sizes 

**Authors**: Rohan Choudhury, JungEun Kim, Jinhyung Park, Eunho Yang, László A. Jeni, Kris M. Kitani  

**Link**: [PDF](https://arxiv.org/pdf/2510.18091)  

**Abstract**: Vision Transformers (ViTs) partition input images into uniformly sized patches regardless of their content, resulting in long input sequence lengths for high-resolution images. We present Adaptive Patch Transformers (APT), which addresses this by using multiple different patch sizes within the same image. APT reduces the total number of input tokens by allocating larger patch sizes in more homogeneous areas and smaller patches in more complex ones. APT achieves a drastic speedup in ViT inference and training, increasing throughput by 40% on ViT-L and 50% on ViT-H while maintaining downstream performance, and can be applied to a previously fine-tuned ViT, converging in as little as 1 epoch. It also significantly reduces training and inference time without loss of performance in high-resolution dense visual tasks, achieving up to 30\% faster training and inference in visual QA, object detection, and semantic segmentation. 

---
# R2BC: Multi-Agent Imitation Learning from Single-Agent Demonstrations 

**Authors**: Connor Mattson, Varun Raveendra, Ellen Novoseller, Nicholas Waytowich, Vernon J. Lawhern, Daniel S. Brown  

**Link**: [PDF](https://arxiv.org/pdf/2510.18085)  

**Abstract**: Imitation Learning (IL) is a natural way for humans to teach robots, particularly when high-quality demonstrations are easy to obtain. While IL has been widely applied to single-robot settings, relatively few studies have addressed the extension of these methods to multi-agent systems, especially in settings where a single human must provide demonstrations to a team of collaborating robots. In this paper, we introduce and study Round-Robin Behavior Cloning (R2BC), a method that enables a single human operator to effectively train multi-robot systems through sequential, single-agent demonstrations. Our approach allows the human to teleoperate one agent at a time and incrementally teach multi-agent behavior to the entire system, without requiring demonstrations in the joint multi-agent action space. We show that R2BC methods match, and in some cases surpass, the performance of an oracle behavior cloning approach trained on privileged synchronized demonstrations across four multi-agent simulated tasks. Finally, we deploy R2BC on two physical robot tasks trained using real human demonstrations. 

---
# RL-Driven Security-Aware Resource Allocation Framework for UAV-Assisted O-RAN 

**Authors**: Zaineh Abughazzah, Emna Baccour, Loay Ismail, Amr Mohamed, Mounir Hamdi  

**Link**: [PDF](https://arxiv.org/pdf/2510.18084)  

**Abstract**: The integration of Unmanned Aerial Vehicles (UAVs) into Open Radio Access Networks (O-RAN) enhances communication in disaster management and Search and Rescue (SAR) operations by ensuring connectivity when infrastructure fails. However, SAR scenarios demand stringent security and low-latency communication, as delays or breaches can compromise mission success. While UAVs serve as mobile relays, they introduce challenges in energy consumption and resource management, necessitating intelligent allocation strategies. Existing UAV-assisted O-RAN approaches often overlook the joint optimization of security, latency, and energy efficiency in dynamic environments. This paper proposes a novel Reinforcement Learning (RL)-based framework for dynamic resource allocation in UAV relays, explicitly addressing these trade-offs. Our approach formulates an optimization problem that integrates security-aware resource allocation, latency minimization, and energy efficiency, which is solved using RL. Unlike heuristic or static methods, our framework adapts in real-time to network dynamics, ensuring robust communication. Simulations demonstrate superior performance compared to heuristic baselines, achieving enhanced security and energy efficiency while maintaining ultra-low latency in SAR scenarios. 

---
# Any-Depth Alignment: Unlocking Innate Safety Alignment of LLMs to Any-Depth 

**Authors**: Jiawei Zhang, Andrew Estornell, David D. Baek, Bo Li, Xiaojun Xu  

**Link**: [PDF](https://arxiv.org/pdf/2510.18081)  

**Abstract**: Large Language Models (LLMs) exhibit strong but shallow alignment: they directly refuse harmful queries when a refusal is expected at the very start of an assistant turn, yet this protection collapses once a harmful continuation is underway (either through the adversarial attacks or via harmful assistant-prefill attacks). This raises a fundamental question: Can the innate shallow alignment in LLMs be unlocked to ensure safety at arbitrary generation depths? To achieve this goal, we propose Any-Depth Alignment (ADA), an effective inference-time defense with negligible overhead. ADA is built based on our observation that alignment is concentrated in the assistant header tokens through repeated use in shallow-refusal training, and these tokens possess the model's strong alignment priors. By reintroducing these tokens mid-stream, ADA induces the model to reassess harmfulness and recover refusals at any point in generation. Across diverse open-source model families (Llama, Gemma, Mistral, Qwen, DeepSeek, and gpt-oss), ADA achieves robust safety performance without requiring any changes to the base model's parameters. It secures a near-100% refusal rate against challenging adversarial prefill attacks ranging from dozens to thousands of tokens. Furthermore, ADA reduces the average success rate of prominent adversarial prompt attacks (such as GCG, AutoDAN, PAIR, and TAP) to below 3%. This is all accomplished while preserving utility on benign tasks with minimal over-refusal. ADA maintains this resilience even after the base model undergoes subsequent instruction tuning (benign or adversarial). 

---
# R2L: Reliable Reinforcement Learning: Guaranteed Return & Reliable Policies in Reinforcement Learning 

**Authors**: Nadir Farhi  

**Link**: [PDF](https://arxiv.org/pdf/2510.18074)  

**Abstract**: In this work, we address the problem of determining reliable policies in reinforcement learning (RL), with a focus on optimization under uncertainty and the need for performance guarantees. While classical RL algorithms aim at maximizing the expected return, many real-world applications - such as routing, resource allocation, or sequential decision-making under risk - require strategies that ensure not only high average performance but also a guaranteed probability of success. To this end, we propose a novel formulation in which the objective is to maximize the probability that the cumulative return exceeds a prescribed threshold. We demonstrate that this reliable RL problem can be reformulated, via a state-augmented representation, into a standard RL problem, thereby allowing the use of existing RL and deep RL algorithms without the need for entirely new algorithmic frameworks. Theoretical results establish the equivalence of the two formulations and show that reliable strategies can be derived by appropriately adapting well-known methods such as Q-learning or Dueling Double DQN. To illustrate the practical relevance of the approach, we consider the problem of reliable routing, where the goal is not to minimize the expected travel time but rather to maximize the probability of reaching the destination within a given time budget. Numerical experiments confirm that the proposed formulation leads to policies that effectively balance efficiency and reliability, highlighting the potential of reliable RL for applications in stochastic and safety-critical environments. 

---
# Fine-tuning Flow Matching Generative Models with Intermediate Feedback 

**Authors**: Jiajun Fan, Chaoran Cheng, Shuaike Shen, Xiangxin Zhou, Ge Liu  

**Link**: [PDF](https://arxiv.org/pdf/2510.18072)  

**Abstract**: Flow-based generative models have shown remarkable success in text-to-image generation, yet fine-tuning them with intermediate feedback remains challenging, especially for continuous-time flow matching models. Most existing approaches solely learn from outcome rewards, struggling with the credit assignment problem. Alternative methods that attempt to learn a critic via direct regression on cumulative rewards often face training instabilities and model collapse in online settings. We present AC-Flow, a robust actor-critic framework that addresses these challenges through three key innovations: (1) reward shaping that provides well-normalized learning signals to enable stable intermediate value learning and gradient control, (2) a novel dual-stability mechanism that combines advantage clipping to prevent destructive policy updates with a warm-up phase that allows the critic to mature before influencing the actor, and (3) a scalable generalized critic weighting scheme that extends traditional reward-weighted methods while preserving model diversity through Wasserstein regularization. Through extensive experiments on Stable Diffusion 3, we demonstrate that AC-Flow achieves state-of-the-art performance in text-to-image alignment tasks and generalization to unseen human preference models. Our results demonstrate that even with a computationally efficient critic model, we can robustly finetune flow models without compromising generative quality, diversity, or stability. 

---
# SPACeR: Self-Play Anchoring with Centralized Reference Models 

**Authors**: Wei-Jer Chang, Akshay Rangesh, Kevin Joseph, Matthew Strong, Masayoshi Tomizuka, Yihan Hu, Wei Zhan  

**Link**: [PDF](https://arxiv.org/pdf/2510.18060)  

**Abstract**: Developing autonomous vehicles (AVs) requires not only safety and efficiency, but also realistic, human-like behaviors that are socially aware and predictable. Achieving this requires sim agent policies that are human-like, fast, and scalable in multi-agent settings. Recent progress in imitation learning with large diffusion-based or tokenized models has shown that behaviors can be captured directly from human driving data, producing realistic policies. However, these models are computationally expensive, slow during inference, and struggle to adapt in reactive, closed-loop scenarios. In contrast, self-play reinforcement learning (RL) scales efficiently and naturally captures multi-agent interactions, but it often relies on heuristics and reward shaping, and the resulting policies can diverge from human norms. We propose SPACeR, a framework that leverages a pretrained tokenized autoregressive motion model as a centralized reference policy to guide decentralized self-play. The reference model provides likelihood rewards and KL divergence, anchoring policies to the human driving distribution while preserving RL scalability. Evaluated on the Waymo Sim Agents Challenge, our method achieves competitive performance with imitation-learned policies while being up to 10x faster at inference and 50x smaller in parameter size than large generative models. In addition, we demonstrate in closed-loop ego planning evaluation tasks that our sim agents can effectively measure planner quality with fast and scalable traffic simulation, establishing a new paradigm for testing autonomous driving policies. 

---
# Adaptive Divergence Regularized Policy Optimization for Fine-tuning Generative Models 

**Authors**: Jiajun Fan, Tong Wei, Chaoran Cheng, Yuxin Chen, Ge Liu  

**Link**: [PDF](https://arxiv.org/pdf/2510.18053)  

**Abstract**: Balancing exploration and exploitation during reinforcement learning fine-tuning of generative models presents a critical challenge, as existing approaches rely on fixed divergence regularization that creates an inherent dilemma: strong regularization preserves model capabilities but limits reward optimization, while weak regularization enables greater alignment but risks instability or reward hacking. We introduce Adaptive Divergence Regularized Policy Optimization (ADRPO), which automatically adjusts regularization strength based on advantage estimates-reducing regularization for high-value samples while applying stronger regularization to poor samples, enabling policies to navigate between exploration and aggressive exploitation according to data quality. Our implementation with Wasserstein-2 regularization for flow matching generative models achieves remarkable results on text-to-image generation, achieving better semantic alignment and diversity than offline methods like DPO and online methods with fixed regularization like ORW-CFM-W2. ADRPO enables a 2B parameter SD3 model to surpass much larger models with 4.8B and 12B parameters in attribute binding, semantic consistency, artistic style transfer, and compositional control while maintaining generation diversity. ADRPO generalizes to KL-regularized fine-tuning of both text-only LLMs and multi-modal reasoning models, enhancing existing online RL methods like GRPO. In LLM fine-tuning, ADRPO demonstrates an emergent ability to escape local optima through active exploration, while in multi-modal audio reasoning, it outperforms GRPO through superior step-by-step reasoning, enabling a 7B model to outperform substantially larger commercial models including Gemini 2.5 Pro and GPT-4o Audio, offering an effective plug-and-play solution to the exploration-exploitation challenge across diverse generative architectures and modalities. 

---
# Measure-Theoretic Anti-Causal Representation Learning 

**Authors**: Arman Behnam, Binghui Wang  

**Link**: [PDF](https://arxiv.org/pdf/2510.18052)  

**Abstract**: Causal representation learning in the anti-causal setting (labels cause features rather than the reverse) presents unique challenges requiring specialized approaches. We propose Anti-Causal Invariant Abstractions (ACIA), a novel measure-theoretic framework for anti-causal representation learning. ACIA employs a two-level design, low-level representations capture how labels generate observations, while high-level representations learn stable causal patterns across environment-specific variations. ACIA addresses key limitations of existing approaches by accommodating prefect and imperfect interventions through interventional kernels, eliminating dependency on explicit causal structures, handling high-dimensional data effectively, and providing theoretical guarantees for out-of-distribution generalization. Experiments on synthetic and real-world medical datasets demonstrate that ACIA consistently outperforms state-of-the-art methods in both accuracy and invariance metrics. Furthermore, our theoretical results establish tight bounds on performance gaps between training and unseen environments, confirming the efficacy of our approach for robust anti-causal learning. 

---
# Language Models as Semantic Augmenters for Sequential Recommenders 

**Authors**: Mahsa Valizadeh, Xiangjue Dong, Rui Tuo, James Caverlee  

**Link**: [PDF](https://arxiv.org/pdf/2510.18046)  

**Abstract**: Large Language Models (LLMs) excel at capturing latent semantics and contextual relationships across diverse modalities. However, in modeling user behavior from sequential interaction data, performance often suffers when such semantic context is limited or absent. We introduce LaMAR, a LLM-driven semantic enrichment framework designed to enrich such sequences automatically. LaMAR leverages LLMs in a few-shot setting to generate auxiliary contextual signals by inferring latent semantic aspects of a user's intent and item relationships from existing metadata. These generated signals, such as inferred usage scenarios, item intents, or thematic summaries, augment the original sequences with greater contextual depth. We demonstrate the utility of this generated resource by integrating it into benchmark sequential modeling tasks, where it consistently improves performance. Further analysis shows that LLM-generated signals exhibit high semantic novelty and diversity, enhancing the representational capacity of the downstream models. This work represents a new data-centric paradigm where LLMs serve as intelligent context generators, contributing a new method for the semi-automatic creation of training data and language resources. 

---
# Cross-Domain Long-Term Forecasting: Radiation Dose from Sparse Neutron Sensor via Spatio-Temporal Operator Network 

**Authors**: Jay Phil Yoo, Kazuma Kobayashi, Souvik Chakraborty, Syed Bahauddin Alam  

**Link**: [PDF](https://arxiv.org/pdf/2510.18041)  

**Abstract**: Forecasting unobservable physical quantities from sparse, cross-domain sensor data is a central unsolved problem in scientific machine learning. Existing neural operators and large-scale forecasters rely on dense, co-located input-output fields and short temporal contexts, assumptions that fail in real-world systems where sensing and prediction occur on distinct physical manifolds and over long timescales. We introduce the Spatio-Temporal Operator Network (STONe), a non-autoregressive neural operator that learns a stable functional mapping between heterogeneous domains. By directly inferring high-altitude radiation dose fields from sparse ground-based neutron measurements, STONe demonstrates that operator learning can generalize beyond shared-domain settings. It defines a nonlinear operator between sensor and target manifolds that remains stable over long forecasting horizons without iterative recurrence. This challenges the conventional view that operator learning requires domain alignment or autoregressive propagation. Trained on 23 years of global neutron data, STONe achieves accurate 180-day forecasts with millisecond inference latency. The framework establishes a general principle for cross-domain operator inference, enabling real-time prediction of complex spatiotemporal fields in physics, climate, and energy systems. 

---
# TriggerNet: A Novel Explainable AI Framework for Red Palm Mite Detection and Multi-Model Comparison and Heuristic-Guided Annotation 

**Authors**: Harshini Suresha, Kavitha SH  

**Link**: [PDF](https://arxiv.org/pdf/2510.18038)  

**Abstract**: The red palm mite infestation has become a serious concern, particularly in regions with extensive palm cultivation, leading to reduced productivity and economic losses. Accurate and early identification of mite-infested plants is critical for effective management. The current study focuses on evaluating and comparing the ML model for classifying the affected plants and detecting the infestation. TriggerNet is a novel interpretable AI framework that integrates Grad-CAM, RISE, FullGrad, and TCAV to generate novel visual explanations for deep learning models in plant classification and disease detection. This study applies TriggerNet to address red palm mite (Raoiella indica) infestation, a major threat to palm cultivation and agricultural productivity. A diverse set of RGB images across 11 plant species, Arecanut, Date Palm, Bird of Paradise, Coconut Palm, Ginger, Citrus Tree, Palm Oil, Orchid, Banana Palm, Avocado Tree, and Cast Iron Plant was utilized for training and evaluation. Advanced deep learning models like CNN, EfficientNet, MobileNet, ViT, ResNet50, and InceptionV3, alongside machine learning classifiers such as Random Forest, SVM, and KNN, were employed for plant classification. For disease classification, all plants were categorized into four classes: Healthy, Yellow Spots, Reddish Bronzing, and Silk Webbing. Snorkel was used to efficiently label these disease classes by leveraging heuristic rules and patterns, reducing manual annotation time and improving dataset reliability. 

---
# SAVANT: Semantic Analysis with Vision-Augmented Anomaly deTection 

**Authors**: Roberto Brusnicki, David Pop, Yuan Gao, Mattia Piccinini, Johannes Betz  

**Link**: [PDF](https://arxiv.org/pdf/2510.18034)  

**Abstract**: Autonomous driving systems remain critically vulnerable to the long-tail of rare, out-of-distribution scenarios with semantic anomalies. While Vision Language Models (VLMs) offer promising reasoning capabilities, naive prompting approaches yield unreliable performance and depend on expensive proprietary models, limiting practical deployment. We introduce SAVANT (Semantic Analysis with Vision-Augmented Anomaly deTection), a structured reasoning framework that achieves high accuracy and recall in detecting anomalous driving scenarios from input images through layered scene analysis and a two-phase pipeline: structured scene description extraction followed by multi-modal evaluation. Our approach transforms VLM reasoning from ad-hoc prompting to systematic analysis across four semantic layers: Street, Infrastructure, Movable Objects, and Environment. SAVANT achieves 89.6% recall and 88.0% accuracy on real-world driving scenarios, significantly outperforming unstructured baselines. More importantly, we demonstrate that our structured framework enables a fine-tuned 7B parameter open-source model (Qwen2.5VL) to achieve 90.8% recall and 93.8% accuracy - surpassing all models evaluated while enabling local deployment at near-zero cost. By automatically labeling over 9,640 real-world images with high accuracy, SAVANT addresses the critical data scarcity problem in anomaly detection and provides a practical path toward reliable, accessible semantic monitoring for autonomous systems. 

---
# From Local to Global: Revisiting Structured Pruning Paradigms for Large Language Models 

**Authors**: Ziyan Wang, Enmao Diao, Qi Le, Pu Wang, Minwoo Lee, Shu-ping Yeh, Evgeny Stupachenko, Hao Feng, Li Yang  

**Link**: [PDF](https://arxiv.org/pdf/2510.18030)  

**Abstract**: Structured pruning is a practical approach to deploying large language models (LLMs) efficiently, as it yields compact, hardware-friendly architectures. However, the dominant local paradigm is task-agnostic: by optimizing layer-wise reconstruction rather than task objectives, it tends to preserve perplexity or generic zero-shot behavior but fails to capitalize on modest task-specific calibration signals, often yielding limited downstream gains. We revisit global structured pruning and present GISP-Global Iterative Structured Pruning-a post-training method that removes attention heads and MLP channels using first-order, loss-based important weights aggregated at the structure level with block-wise normalization. An iterative schedule, rather than one-shot pruning, stabilizes accuracy at higher sparsity and mitigates perplexity collapse without requiring intermediate fine-tuning; the pruning trajectory also forms nested subnetworks that support a "prune-once, deploy-many" workflow. Furthermore, because importance is defined by a model-level loss, GISP naturally supports task-specific objectives; we instantiate perplexity for language modeling and a margin-based objective for decision-style tasks. Extensive experiments show that across Llama2-7B/13B, Llama3-8B, and Mistral-0.3-7B, GISP consistently lowers WikiText-2 perplexity and improves downstream accuracy, with especially strong gains at 40-50% sparsity; on DeepSeek-R1-Distill-Llama-3-8B with GSM8K, task-aligned calibration substantially boosts exact-match accuracy. 

---
# DynaQuery: A Self-Adapting Framework for Querying Structured and Multimodal Data 

**Authors**: Aymane Hassini  

**Link**: [PDF](https://arxiv.org/pdf/2510.18029)  

**Abstract**: The rise of Large Language Models (LLMs) has accelerated the long-standing goal of enabling natural language querying over complex, hybrid databases. Yet, this ambition exposes a dual challenge: reasoning jointly over structured, multi-relational schemas and the semantic content of linked unstructured assets. To overcome this, we present DynaQuery - a unified, self-adapting framework that serves as a practical blueprint for next-generation "Unbound Databases." At the heart of DynaQuery lies the Schema Introspection and Linking Engine (SILE), a novel systems primitive that elevates schema linking to a first-class query planning phase. We conduct a rigorous, multi-benchmark empirical evaluation of this structure-aware architecture against the prevalent unstructured Retrieval-Augmented Generation (RAG) paradigm. Our results demonstrate that the unstructured retrieval paradigm is architecturally susceptible to catastrophic contextual failures, such as SCHEMA_HALLUCINATION, leading to unreliable query generation. In contrast, our SILE-based design establishes a substantially more robust foundation, nearly eliminating this failure mode. Moreover, end-to-end validation on a complex, newly curated benchmark uncovers a key generalization principle: the transition from pure schema-awareness to holistic semantics-awareness. Taken together, our findings provide a validated architectural basis for developing natural language database interfaces that are robust, adaptable, and predictably consistent. 

---
# Is Multilingual LLM Watermarking Truly Multilingual? A Simple Back-Translation Solution 

**Authors**: Asim Mohamed, Martin Gubri  

**Link**: [PDF](https://arxiv.org/pdf/2510.18019)  

**Abstract**: Multilingual watermarking aims to make large language model (LLM) outputs traceable across languages, yet current methods still fall short. Despite claims of cross-lingual robustness, they are evaluated only on high-resource languages. We show that existing multilingual watermarking methods are not truly multilingual: they fail to remain robust under translation attacks in medium- and low-resource languages. We trace this failure to semantic clustering, which fails when the tokenizer vocabulary contains too few full-word tokens for a given language. To address this, we introduce STEAM, a back-translation-based detection method that restores watermark strength lost through translation. STEAM is compatible with any watermarking method, robust across different tokenizers and languages, non-invasive, and easily extendable to new languages. With average gains of +0.19 AUC and +40%p TPR@1% on 17 languages, STEAM provides a simple and robust path toward fairer watermarking across diverse languages. 

---
# BadScientist: Can a Research Agent Write Convincing but Unsound Papers that Fool LLM Reviewers? 

**Authors**: Fengqing Jiang, Yichen Feng, Yuetai Li, Luyao Niu, Basel Alomair, Radha Poovendran  

**Link**: [PDF](https://arxiv.org/pdf/2510.18003)  

**Abstract**: The convergence of LLM-powered research assistants and AI-based peer review systems creates a critical vulnerability: fully automated publication loops where AI-generated research is evaluated by AI reviewers without human oversight. We investigate this through \textbf{BadScientist}, a framework that evaluates whether fabrication-oriented paper generation agents can deceive multi-model LLM review systems. Our generator employs presentation-manipulation strategies requiring no real experiments. We develop a rigorous evaluation framework with formal error guarantees (concentration bounds and calibration analysis), calibrated on real data. Our results reveal systematic vulnerabilities: fabricated papers achieve acceptance rates up to . Critically, we identify \textit{concern-acceptance conflict} -- reviewers frequently flag integrity issues yet assign acceptance-level scores. Our mitigation strategies show only marginal improvements, with detection accuracy barely exceeding random chance. Despite provably sound aggregation mathematics, integrity checking systematically fails, exposing fundamental limitations in current AI-driven review systems and underscoring the urgent need for defense-in-depth safeguards in scientific publishing. 

---
# SimBA: Simplifying Benchmark Analysis Using Performance Matrices Alone 

**Authors**: Nishant Subramani, Alfredo Gomez, Mona Diab  

**Link**: [PDF](https://arxiv.org/pdf/2510.17998)  

**Abstract**: Modern language models are evaluated on large benchmarks, which are difficult to make sense of, especially for model selection. Looking at the raw evaluation numbers themselves using a model-centric lens, we propose SimBA, a three phase framework to Simplify Benchmark Analysis. The three phases of SimBA are: stalk, where we conduct dataset & model comparisons, prowl, where we discover a representative subset, and pounce, where we use the representative subset to predict performance on a held-out set of models. Applying SimBA to three popular LM benchmarks: HELM, MMLU, and BigBenchLite reveals that across all three benchmarks, datasets and models relate strongly to one another (stalk). We develop an representative set discovery algorithm which covers a benchmark using raw evaluation scores alone. Using our algorithm, we find that with 6.25% (1/16), 1.7% (1/58), and 28.4% (21/74) of the datasets for HELM, MMLU, and BigBenchLite respectively, we achieve coverage levels of at least 95% (prowl). Additionally, using just these representative subsets, we can both preserve model ranks and predict performance on a held-out set of models with near zero mean-squared error (pounce). Taken together, SimBA can help model developers improve efficiency during model training and dataset creators validate whether their newly created dataset differs from existing datasets in a benchmark. Our code is open source, available at this https URL. 

---
# Universal Spectral Tokenization via Self-Supervised Panchromatic Representation Learning 

**Authors**: Jeff Shen, Francois Lanusse, Liam Holden Parker, Ollie Liu, Tom Hehir, Leopoldo Sarra, Lucas Meyer, Micah Bowles, Sebastian Wagner-Carena, Sebastian Wagner-Carena, Helen Qu, Siavash Golkar, Alberto Bietti, Hatim Bourfoune, Nathan Cassereau, Pierre Cornette, Keiya Hirashima, Geraud Krawezik, Ruben Ohana, Nicholas Lourie, Michael McCabe, Rudy Morel, Payel Mukhopadhyay, Mariel Pettee, Bruno Régaldo-Saint Blancard, Kyunghyun Cho, Miles Cranmer, Shirley Ho  

**Link**: [PDF](https://arxiv.org/pdf/2510.17959)  

**Abstract**: Sequential scientific data span many resolutions and domains, and unifying them into a common representation is a key step toward developing foundation models for the sciences. Astronomical spectra exemplify this challenge: massive surveys have collected millions of spectra across a wide range of wavelengths and resolutions, yet analyses remain fragmented across spectral domains (e.g., optical vs. infrared) and object types (e.g., stars vs. galaxies), limiting the ability to pool information across datasets. We present a deep learning model that jointly learns from heterogeneous spectra in a self-supervised manner. Our universal spectral tokenizer processes spectra from a variety of object types and resolutions directly on their native wavelength grids, producing intrinsically aligned, homogeneous, and physically meaningful representations that can be efficiently adapted to achieve competitive performance across a range of downstream tasks. For the first time, we demonstrate that a single model can unify spectral data across resolutions and domains, suggesting that our model can serve as a powerful building block for foundation models in astronomy -- and potentially extend to other scientific domains with heterogeneous sequential data, such as climate and healthcare. 

---
# Studying the Effects of Robot Intervention on School Shooters in Virtual Reality 

**Authors**: Christopher A McClurg, Alan R Wagner  

**Link**: [PDF](https://arxiv.org/pdf/2510.17948)  

**Abstract**: We advance the understanding of robotic intervention in high-risk scenarios by examining their potential to distract and impede a school shooter. To evaluate this concept, we conducted a virtual reality study with 150 university participants role-playing as a school shooter. Within the simulation, an autonomous robot predicted the shooter's movements and positioned itself strategically to interfere and distract. The strategy the robot used to approach the shooter was manipulated -- either moving directly in front of the shooter (aggressive) or maintaining distance (passive) -- and the distraction method, ranging from no additional cues (low), to siren and lights (medium), to siren, lights, and smoke to impair visibility (high). An aggressive, high-distraction robot reduced the number of victims by 46.6% relative to a no-robot control. This outcome underscores both the potential of robotic intervention to enhance safety and the pressing ethical questions surrounding their use in school environments. 

---
# PLAGUE: Plug-and-play framework for Lifelong Adaptive Generation of Multi-turn Exploits 

**Authors**: Neeladri Bhuiya, Madhav Aggarwal, Diptanshu Purwar  

**Link**: [PDF](https://arxiv.org/pdf/2510.17947)  

**Abstract**: Large Language Models (LLMs) are improving at an exceptional rate. With the advent of agentic workflows, multi-turn dialogue has become the de facto mode of interaction with LLMs for completing long and complex tasks. While LLM capabilities continue to improve, they remain increasingly susceptible to jailbreaking, especially in multi-turn scenarios where harmful intent can be subtly injected across the conversation to produce nefarious outcomes. While single-turn attacks have been extensively explored, adaptability, efficiency and effectiveness continue to remain key challenges for their multi-turn counterparts. To address these gaps, we present PLAGUE, a novel plug-and-play framework for designing multi-turn attacks inspired by lifelong-learning agents. PLAGUE dissects the lifetime of a multi-turn attack into three carefully designed phases (Primer, Planner and Finisher) that enable a systematic and information-rich exploration of the multi-turn attack family. Evaluations show that red-teaming agents designed using PLAGUE achieve state-of-the-art jailbreaking results, improving attack success rates (ASR) by more than 30% across leading models in a lesser or comparable query budget. Particularly, PLAGUE enables an ASR (based on StrongReject) of 81.4% on OpenAI's o3 and 67.3% on Claude's Opus 4.1, two models that are considered highly resistant to jailbreaks in safety literature. Our work offers tools and insights to understand the importance of plan initialization, context optimization and lifelong learning in crafting multi-turn attacks for a comprehensive model vulnerability evaluation. 

---
# Intuitionistic $j$-Do-Calculus in Topos Causal Models 

**Authors**: Sridhar Mahadevan  

**Link**: [PDF](https://arxiv.org/pdf/2510.17944)  

**Abstract**: In this paper, we generalize Pearl's do-calculus to an Intuitionistic setting called $j$-stable causal inference inside a topos of sheaves. Our framework is an elaboration of the recently proposed framework of Topos Causal Models (TCMs), where causal interventions are defined as subobjects. We generalize the original setting of TCM using the Lawvere-Tierney topology on a topos, defined by a modal operator $j$ on the subobject classifier $\Omega$. We introduce $j$-do-calculus, where we replace global truth with local truth defined by Kripke-Joyal semantics, and formalize causal reasoning as structure-preserving morphisms that are stable along $j$-covers. $j$-do-calculus is a sound rule system whose premises and conclusions are formulas of the internal Intuitionistic logic of the causal topos. We define $j$-stability for conditional independences and interventional claims as local truth in the internal logic of the causal topos. We give three inference rules that mirror Pearl's insertion/deletion and action/observation exchange, and we prove soundness in the Kripke-Joyal semantics. A companion paper in preparation will describe how to estimate the required entities from data and instantiate $j$-do with standard discovery procedures (e.g., score-based and constraint-based methods), and will include experimental results on how to (i) form data-driven $j$-covers (via regime/section constructions), (ii) compute chartwise conditional independences after graph surgeries, and (iii) glue them to certify the premises of the $j$-do rules in practice 

---
# Trust in foundation models and GenAI: A geographic perspective 

**Authors**: Grant McKenzie, Krzysztof Janowicz, Carsten Kessler  

**Link**: [PDF](https://arxiv.org/pdf/2510.17942)  

**Abstract**: Large-scale pre-trained machine learning models have reshaped our understanding of artificial intelligence across numerous domains, including our own field of geography. As with any new technology, trust has taken on an important role in this discussion. In this chapter, we examine the multifaceted concept of trust in foundation models, particularly within a geographic context. As reliance on these models increases and they become relied upon for critical decision-making, trust, while essential, has become a fractured concept. Here we categorize trust into three types: epistemic trust in the training data, operational trust in the model's functionality, and interpersonal trust in the model developers. Each type of trust brings with it unique implications for geographic applications. Topics such as cultural context, data heterogeneity, and spatial relationships are fundamental to the spatial sciences and play an important role in developing trust. The chapter continues with a discussion of the challenges posed by different forms of biases, the importance of transparency and explainability, and ethical responsibilities in model development. Finally, the novel perspective of geographic information scientists is emphasized with a call for further transparency, bias mitigation, and regionally-informed policies. Simply put, this chapter aims to provide a conceptual starting point for researchers, practitioners, and policy-makers to better understand trust in (generative) GeoAI. 

---
# Believe It or Not: How Deeply do LLMs Believe Implanted Facts? 

**Authors**: Stewart Slocum, Julian Minder, Clément Dumas, Henry Sleight, Ryan Greenblatt, Samuel Marks, Rowan Wang  

**Link**: [PDF](https://arxiv.org/pdf/2510.17941)  

**Abstract**: Knowledge editing techniques promise to implant new factual knowledge into large language models (LLMs). But do LLMs really believe these facts? We develop a framework to measure belief depth and use it to evaluate the success of knowledge editing techniques. We operationalize belief depth as the extent to which implanted knowledge 1) generalizes to related contexts (e.g. Fermi estimates several logical steps removed), 2) is robust to self-scrutiny and direct challenge, and 3) is represented similarly to genuine knowledge (as measured by linear probes). Our evaluations show that simple prompting and mechanistic editing techniques fail to implant knowledge deeply. In contrast, Synthetic Document Finetuning (SDF) - where models are trained on LLM-generated documents consistent with a fact - often succeeds at implanting beliefs that behave similarly to genuine knowledge. However, SDF's success is not universal, as implanted beliefs that contradict basic world knowledge are brittle and representationally distinct from genuine knowledge. Overall, our work introduces measurable criteria for belief depth and enables the rigorous evaluation necessary for deploying knowledge editing in real-world applications. 

---
# The Integration of Artificial Intelligence in Undergraduate Medical Education in Spain: Descriptive Analysis and International Perspectives 

**Authors**: Ana Enériz Janeiro, Karina Pitombeira Pereira, Julio Mayol, Javier Crespo, Fernando Carballo, Juan B. Cabello, Manel Ramos-Casals, Bibiana Pérez Corbacho, Juan Turnes  

**Link**: [PDF](https://arxiv.org/pdf/2510.17938)  

**Abstract**: AI is transforming medical practice and redefining the competencies that future healthcare professionals need to master. Despite international recommendations, the integration of AI into Medicine curricula in Spain had not been systematically evaluated until now. A cross-sectional study (July-September 2025) including Spanish universities offering the official degree in Medicine, according to the 'Register of Universities, Centers and Degrees (Registro de Universidades, Centros y Títulos RUCT)'. Curricula and publicly available institutional documentation were reviewed to identify courses and competencies related to AI in the 2025-2026 academic year. The analysis was performed using descriptive statistics. Of the 52 universities analyzed, ten (19.2%) offer specific AI courses, whereas 36 (69.2%) include no related content. Most of the identified courses are elective, with a credit load ranging from three to six ECTS, representing on average 1.17% of the total 360 credits of the degree. The University of Jaén is the only institution offering a compulsory course with AI content. The territorial analysis reveals marked disparities: Andalusia leads with 55.5% of its universities incorporating AI training, while several communities lack any initiative in this area. The integration of AI into the medical degree in Spain is incipient, fragmented, and uneven, with a low weight in ECTS. The limited training load and predominance of elective courses restrict the preparation of future physicians to practice in a healthcare environment increasingly mediated by AI. The findings support the establishment of minimum standards and national monitoring of indicators. 

---
# UniRL-Zero: Reinforcement Learning on Unified Models with Joint Language Model and Diffusion Model Experts 

**Authors**: Fu-Yun Wang, Han Zhang, Michael Gharbi, Hongsheng Li, Taesung Park  

**Link**: [PDF](https://arxiv.org/pdf/2510.17937)  

**Abstract**: We present UniRL-Zero, a unified reinforcement learning (RL) framework that boosts, multimodal language model understanding and reasoning, diffusion model multimedia generation, and their beneficial interaction capabilities within a unified model. Our work defines six scenarios for unified model reinforcement learning, providing systematic baselines for reinforcement learning of unified understanding and generation model. Our code is available at this https URL. 

---
# XDXD: End-to-end crystal structure determination with low resolution X-ray diffraction 

**Authors**: Jiale Zhao, Cong Liu, Yuxuan Zhang, Chengyue Gong, Zhenyi Zhang, Shifeng Jin, Zhenyu Liu  

**Link**: [PDF](https://arxiv.org/pdf/2510.17936)  

**Abstract**: Determining crystal structures from X-ray diffraction data is fundamental across diverse scientific fields, yet remains a significant challenge when data is limited to low resolution. While recent deep learning models have made breakthroughs in solving the crystallographic phase problem, the resulting low-resolution electron density maps are often ambiguous and difficult to interpret. To overcome this critical bottleneck, we introduce XDXD, to our knowledge, the first end-to-end deep learning framework to determine a complete atomic model directly from low-resolution single-crystal X-ray diffraction data. Our diffusion-based generative model bypasses the need for manual map interpretation, producing chemically plausible crystal structures conditioned on the diffraction pattern. We demonstrate that XDXD achieves a 70.4\% match rate for structures with data limited to 2.0~Å resolution, with a root-mean-square error (RMSE) below 0.05. Evaluated on a benchmark of 24,000 experimental structures, our model proves to be robust and accurate. Furthermore, a case study on small peptides highlights the model's potential for extension to more complex systems, paving the way for automated structure solution in previously intractable cases. 

---
# AtlasKV: Augmenting LLMs with Billion-Scale Knowledge Graphs in 20GB VRAM 

**Authors**: Haoyu Huang, Hong Ting Tsang, Jiaxin Bai, Xi Peng, Gong Zhang, Yangqiu Song  

**Link**: [PDF](https://arxiv.org/pdf/2510.17934)  

**Abstract**: Retrieval-augmented generation (RAG) has shown some success in augmenting large language models (LLMs) with external knowledge. However, as a non-parametric knowledge integration paradigm for LLMs, RAG methods heavily rely on external retrieval modules and the retrieved textual context prior. Especially for very large scale knowledge augmentation, they would introduce substantial inference latency due to expensive searches and much longer relevant context. In this paper, we propose a parametric knowledge integration method, called \textbf{AtlasKV}, a scalable, effective, and general way to augment LLMs with billion-scale knowledge graphs (KGs) (e.g. 1B triples) using very little GPU memory cost (e.g. less than 20GB VRAM). In AtlasKV, we introduce KG2KV and HiKVP to integrate KG triples into LLMs at scale with sub-linear time and memory complexity. It maintains strong knowledge grounding and generalization performance using the LLMs' inherent attention mechanism, and requires no external retrievers, long context priors, or retraining when adapting to new knowledge. 

---
# From Observations to Parameters: Detecting Changepoint in Nonlinear Dynamics with Simulation-based Inference 

**Authors**: Xiangbo Deng, Cheng Chen, Peng Yang  

**Link**: [PDF](https://arxiv.org/pdf/2510.17933)  

**Abstract**: Detecting regime shifts in chaotic time series is hard because observation-space signals are entangled with intrinsic variability. We propose Parameter--Space Changepoint Detection (Param--CPD), a two--stage framework that first amortizes Bayesian inference of governing parameters with a neural posterior estimator trained by simulation-based inference, and then applies a standard CPD algorithm to the resulting parameter trajectory. On Lorenz--63 with piecewise-constant parameters, Param--CPD improves F1, reduces localization error, and lowers false positives compared to observation--space baselines. We further verify identifiability and calibration of the inferred posteriors on stationary trajectories, explaining why parameter space offers a cleaner detection signal. Robustness analyses over tolerance, window length, and noise indicate consistent gains. Our results show that operating in a physically interpretable parameter space enables accurate and interpretable changepoint detection in nonlinear dynamical systems. 

---
# From Charts to Code: A Hierarchical Benchmark for Multimodal Models 

**Authors**: Jiahao Tang, Henry Hengyuan Zhao, Lijian Wu, Yifei Tao, Dongxing Mao, Yang Wan, Jingru Tan, Min Zeng, Min Li, Alex Jinpeng Wang  

**Link**: [PDF](https://arxiv.org/pdf/2510.17932)  

**Abstract**: We introduce Chart2Code, a new benchmark for evaluating the chart understanding and code generation capabilities of large multimodal models (LMMs). Chart2Code is explicitly designed from a user-driven perspective, capturing diverse real-world scenarios and progressively increasing task difficulty. It consists of three levels: Level 1 (Chart Reproduction) reproduces charts from a reference figure and user query; Level 2 (Chart Editing) involves complex modifications such as changing chart types or adding elements; and Level 3 (Long-Table to Chart Generation) requires models to transform long, information-dense tables into faithful charts following user instructions. To our knowledge, this is the first hierarchical benchmark that reflects practical chart2code usage while systematically scaling task complexity. In total, Chart2Code contains 2,023 tasks across 22 chart types, paired with multi-level evaluation metrics that assess both code correctness and the visual fidelity of rendered charts. We benchmark 25 state-of-the-art (SoTA) LMMs, including both proprietary and the latest open-source models such as GPT-5, Qwen2.5-VL, InternVL3/3.5, MiMo-VL, and Seed-1.6-VL. Experimental results demonstrate that even the SoTA model GPT-5 averages only 0.57 on code-based evaluation and 0.22 on chart-quality assessment across the editing tasks, underscoring the difficulty of Chart2Code. We anticipate this benchmark will drive advances in multimodal reasoning and foster the development of more robust and general-purpose LMMs. Our code and data are available on Chart2Code. 

---
# Attracting Commercial Artificial Intelligence Firms to Support National Security through Collaborative Contracts 

**Authors**: Andrew Bowne  

**Link**: [PDF](https://arxiv.org/pdf/2510.17931)  

**Abstract**: Unlike other military technologies driven by national security needs and developed with federal funding, AI is predominantly funded and advanced by commercial industry for civilian applications. However, there is a lack of understanding of the reasons commercial AI firms decide to work with the DoD or choose to abstain from the defence market. This thesis argues that the contract law and procurement framework are among the most significant obstacles. This research indicates that the commercial AI industry actually views the DoD as an attractive customer. However, this attraction is despite the obstacles presented by traditional contract law and procurement practices used to solicit and award contracts. Drawing on social exchange theory, this thesis introduces a theoretical framework, optimal buyer theory, to understand the factors that influence a commercial decision to engage with the DoD. Interviews from a sample of the participants explain why the AI industry holds such perceptions, opinions, and preferences about contracts generally and the DoD, specifically, in its role as a customer. This thesis concludes that commercial AI firms are attracted to contracts that are consistent with their business and technology considerations. Additionally, it develops best practices for leveraging existing contract law, primarily other transaction authority, to align contracting practices with commercial preferences and the machine learning development and deployment lifecycle. 

---
# Diagnosing Representation Dynamics in NER Model Extension 

**Authors**: Xirui Zhang, Philippe de La Chevasnerie, Benoit Fabre  

**Link**: [PDF](https://arxiv.org/pdf/2510.17930)  

**Abstract**: Extending Named Entity Recognition (NER) models to new PII entities in noisy spoken-language data is a common need. We find that jointly fine-tuning a BERT model on standard semantic entities (PER, LOC, ORG) and new pattern-based PII (EMAIL, PHONE) results in minimal degradation for original classes. We investigate this "peaceful coexistence," hypothesizing that the model uses independent semantic vs. morphological feature mechanisms.
Using an incremental learning setup as a diagnostic tool, we measure semantic drift and find two key insights. First, the LOC (location) entity is uniquely vulnerable due to a representation overlap with new PII, as it shares pattern-like features (e.g., postal codes). Second, we identify a "reverse O-tag representation drift." The model, initially trained to map PII patterns to 'O', blocks new learning. This is resolved only by unfreezing the 'O' tag's classifier, allowing the background class to adapt and "release" these patterns. This work provides a mechanistic diagnosis of NER model adaptation, highlighting feature independence, representation overlap, and 'O' tag plasticity. 

---
# EvoSyn: Generalizable Evolutionary Data Synthesis for Verifiable Learning 

**Authors**: He Du, Bowen Li, Aijun Yang, Siyang He, Qipeng Guo, Dacheng Tao  

**Link**: [PDF](https://arxiv.org/pdf/2510.17928)  

**Abstract**: Reliable verifiable data has become a key driver of capability gains in modern language models, enabling stable reinforcement learning with verifiable rewards and effective distillation that transfers competence across math, coding, and agentic tasks. Yet constructing generalizable synthetic verifiable data remains difficult due to hallucination-prone generation, and weak or trivial verification artifacts that fail to separate strong from weak solutions. Existing approaches often rely on task-specific heuristics or post-hoc filters that do not transfer across domains and lack a principled, universal evaluator of verifiability. In this work, we introduce an evolutionary, task-agnostic, strategy-guided, executably-checkable data synthesis framework that, from minimal seed supervision, jointly synthesizes problems, diverse candidate solutions, and verification artifacts, and iteratively discovers strategies via a consistency-based evaluator that enforces agreement between human-annotated and strategy-induced checks. This pipeline upgrades filtering into principled synthesis: it reliably assembles coherent, verifiable training instances and generalizes without domain-specific rules. Our experiments demonstrate the effectiveness of the proposed approach under both RLVR and model distillation training paradigms. The results show that training with our synthesized data yields significant improvements on both the LiveCodeBench and AgentBench-OS tasks, highlighting the robust generalization of our framework. 

---
# SpecAgent: A Speculative Retrieval and Forecasting Agent for Code Completion 

**Authors**: George Ma, Anurag Koul, Qi Chen, Yawen Wu, Sachit Kuhar, Yu Yu, Aritra Sengupta, Varun Kumar, Murali Krishna Ramanathan  

**Link**: [PDF](https://arxiv.org/pdf/2510.17925)  

**Abstract**: Large Language Models (LLMs) excel at code-related tasks but often struggle in realistic software repositories, where project-specific APIs and cross-file dependencies are crucial. Retrieval-augmented methods mitigate this by injecting repository context at inference time. The low inference-time latency budget affects either retrieval quality or the added latency adversely impacts user experience. We address this limitation with SpecAgent, an agent that improves both latency and code-generation quality by proactively exploring repository files during indexing and constructing speculative context that anticipates future edits in each file. This indexing-time asynchrony allows thorough context computation, masking latency, and the speculative nature of the context improves code-generation quality. Additionally, we identify the problem of future context leakage in existing benchmarks, which can inflate reported performance. To address this, we construct a synthetic, leakage-free benchmark that enables a more realistic evaluation of our agent against baselines. Experiments show that SpecAgent consistently achieves absolute gains of 9-11% (48-58% relative) compared to the best-performing baselines, while significantly reducing inference latency. 

---
# Efficient Toxicity Detection in Gaming Chats: A Comparative Study of Embeddings, Fine-Tuned Transformers and LLMs 

**Authors**: Yehor Tereshchenko, Mika Hämäläinen  

**Link**: [PDF](https://arxiv.org/pdf/2510.17924)  

**Abstract**: This paper presents a comprehensive comparative analysis of Natural Language Processing (NLP) methods for automated toxicity detection in online gaming chats. Traditional machine learning models with embeddings, large language models (LLMs) with zero-shot and few-shot prompting, fine-tuned transformer models, and retrieval-augmented generation (RAG) approaches are evaluated. The evaluation framework assesses three critical dimensions: classification accuracy, processing speed, and computational costs. A hybrid moderation system architecture is proposed that optimizes human moderator workload through automated detection and incorporates continuous learning mechanisms. The experimental results demonstrate significant performance variations across methods, with fine-tuned DistilBERT achieving optimal accuracy-cost trade-offs. The findings provide empirical evidence for deploying cost-effective, efficient content moderation systems in dynamic online gaming environments. 

---
# Rewarding the Journey, Not Just the Destination: A Composite Path and Answer Self-Scoring Reward Mechanism for Test-Time Reinforcement Learning 

**Authors**: Chenwei Tang, Jingyu Xing, Xinyu Liu, Wei Ju, Jiancheng Lv, Deng Xiong, Ziyue Qiao  

**Link**: [PDF](https://arxiv.org/pdf/2510.17923)  

**Abstract**: Reinforcement Learning (RL) has emerged as a powerful paradigm for advancing Large Language Models (LLMs), achieving remarkable performance in complex reasoning domains such as mathematics and code generation. However, current RL methods face a fundamental scalability bottleneck due to their heavy reliance on human-curated preference data or labeled datasets for reward modeling. To overcome this limitation, we explore RL on unlabeled data where models learn autonomously from continuous experience streams. The core challenge in this setting lies in reliable reward estimation without ground-truth supervision. Existing approaches like Test-Time RL address this through self-consistent consensus, but risk reinforcing incorrect pseudo-labels derived from majority voting. We introduce COMPASS (Composite Path and Answer Self-Scoring), a novel test-time reward mechanism that operates without external supervision. COMPASS integrates two complementary components: the Dual-Calibration Answer Reward (DCAR), which stabilizes training by establishing trustworthy pseudo-labels through confidence and credibility calibration, and the Decisive Path Reward (DPR), which directly optimizes the reasoning process quality beyond mere outcome supervision. By jointly reinforcing trustworthy consensus answers and highly decisive reasoning chains, the COMPASS systematically enhances the model's analytical capabilities. Extensive experiments show that COMPASS achieves significant and consistent performance gains across diverse reasoning tasks and model architectures, advancing a more scalable direction for LLMs to learn from continuous experience. 

---
# Select-Then-Decompose: From Empirical Analysis to Adaptive Selection Strategy for Task Decomposition in Large Language Models 

**Authors**: Shuodi Liu, Yingzhuo Liu, Zi Wang, Yusheng Wang, Huijia Wu, Liuyu Xiang, Zhaofeng He  

**Link**: [PDF](https://arxiv.org/pdf/2510.17922)  

**Abstract**: Large language models (LLMs) have demonstrated remarkable reasoning and planning capabilities, driving extensive research into task decomposition. Existing task decomposition methods focus primarily on memory, tool usage, and feedback mechanisms, achieving notable success in specific domains, but they often overlook the trade-off between performance and cost. In this study, we first conduct a comprehensive investigation on task decomposition, identifying six categorization schemes. Then, we perform an empirical analysis of three factors that influence the performance and cost of task decomposition: categories of approaches, characteristics of tasks, and configuration of decomposition and execution models, uncovering three critical insights and summarizing a set of practical principles. Building on this analysis, we propose the Select-Then-Decompose strategy, which establishes a closed-loop problem-solving process composed of three stages: selection, execution, and verification. This strategy dynamically selects the most suitable decomposition approach based on task characteristics and enhances the reliability of the results through a verification module. Comprehensive evaluations across multiple benchmarks show that the Select-Then-Decompose consistently lies on the Pareto frontier, demonstrating an optimal balance between performance and cost. Our code is publicly available at this https URL. 

---
# CLAWS:Creativity detection for LLM-generated solutions using Attention Window of Sections 

**Authors**: Keuntae Kim, Eunhye Jeong, Sehyeon Lee, Seohee Yoon, Yong Suk Choi  

**Link**: [PDF](https://arxiv.org/pdf/2510.17921)  

**Abstract**: Recent advances in enhancing the reasoning ability of large language models (LLMs) have been remarkably successful. LLMs trained with reinforcement learning (RL) for reasoning demonstrate strong performance in challenging tasks such as mathematics and coding, even with relatively small model sizes. However, despite these improvements in task accuracy, the assessment of creativity in LLM generations has been largely overlooked in reasoning tasks, in contrast to writing tasks. The lack of research on creativity assessment in reasoning primarily stems from two challenges: (1) the difficulty of defining the range of creativity, and (2) the necessity of human evaluation in the assessment process. To address these challenges, we propose CLAWS, a method that defines and classifies mathematical solutions into typical, creative, and hallucinated categories without human evaluation, by leveraging attention weights across prompt sections and output. CLAWS outperforms five existing white-box detection methods (Perplexity, Logit Entropy, Window Entropy, Hidden Score, and Attention Score) on five 7-8B math RL models (DeepSeek, Qwen, Mathstral, OpenMath2, and Oreal). We validate CLAWS on 4545 math problems collected from 181 math contests (AJHSME, AMC, AIME). 

---
# CBINNS: Cancer Biology-Informed Neural Network for Unknown Parameter Estimation and Missing Physics Identification 

**Authors**: Bishal Chhetri, B.V. Rathish Kumar  

**Link**: [PDF](https://arxiv.org/pdf/2510.17920)  

**Abstract**: The dynamics of tumor-immune interactions within a complex tumor microenvironment are typically modeled using a system of ordinary differential equations or partial differential equations. These models introduce some unknown parameters that need to be estimated accurately and efficiently from the limited and noisy experimental data. Moreover, due to the intricate biological complexity and limitations in experimental measurements, tumor-immune dynamics are not fully understood, and therefore, only partial knowledge of the underlying physics may be available, resulting in unknown or missing terms within the system of equations. In this study, we develop a cancer biology-informed neural network model(CBINN) to infer the unknown parameters in the system of equations as well as to discover the missing physics from sparse and noisy measurements. We test the performance of the CBINN model on three distinct nonlinear compartmental tumor-immune models and evaluate its robustness across multiple synthetic noise levels. By harnessing these highly nonlinear dynamics, our CBINN framework effectively estimates the unknown model parameters and uncovers the underlying physical laws or mathematical structures that govern these biological systems, even from scattered and noisy measurements. The models chosen here represent the dynamic patterns commonly observed in compartmental models of tumor-immune interactions, thereby validating the generalizability and efficacy of our methodology. 

---
# ParaVul: A Parallel Large Language Model and Retrieval-Augmented Framework for Smart Contract Vulnerability Detection 

**Authors**: Tenghui Huang, Jinbo Wen, Jiawen Kang, Siyong Chen, Zhengtao Li, Tao Zhang, Dongning Liu, Jiacheng Wang, Chengjun Cai, Yinqiu Liu, Dusit Niyato  

**Link**: [PDF](https://arxiv.org/pdf/2510.17919)  

**Abstract**: Smart contracts play a significant role in automating blockchain services. Nevertheless, vulnerabilities in smart contracts pose serious threats to blockchain security. Currently, traditional detection methods primarily rely on static analysis and formal verification, which can result in high false-positive rates and poor scalability. Large Language Models (LLMs) have recently made significant progress in smart contract vulnerability detection. However, they still face challenges such as high inference costs and substantial computational overhead. In this paper, we propose ParaVul, a parallel LLM and retrieval-augmented framework to improve the reliability and accuracy of smart contract vulnerability detection. Specifically, we first develop Sparse Low-Rank Adaptation (SLoRA) for LLM fine-tuning. SLoRA introduces sparsification by incorporating a sparse matrix into quantized LoRA-based LLMs, thereby reducing computational overhead and resource requirements while enhancing their ability to understand vulnerability-related issues. We then construct a vulnerability contract dataset and develop a hybrid Retrieval-Augmented Generation (RAG) system that integrates dense retrieval with Best Matching 25 (BM25), assisting in verifying the results generated by the LLM. Furthermore, we propose a meta-learning model to fuse the outputs of the RAG system and the LLM, thereby generating the final detection results. After completing vulnerability detection, we design chain-of-thought prompts to guide LLMs to generate comprehensive vulnerability detection reports. Simulation results demonstrate the superiority of ParaVul, especially in terms of F1 scores, achieving 0.9398 for single-label detection and 0.9330 for multi-label detection. 

---
# JT-Safe: Intrinsically Enhancing the Safety and Trustworthiness of LLMs 

**Authors**: Junlan Feng, Fanyu Meng, Chong Long, Pengyu Cong, Duqing Wang, Yan Zheng, Yuyao Zhang, Xuanchang Gao, Ye Yuan, Yunfei Ma, Zhijie Ren, Fan Yang, Na Wu, Di Jin, Chao Deng  

**Link**: [PDF](https://arxiv.org/pdf/2510.17918)  

**Abstract**: The hallucination and credibility concerns of large language models (LLMs) are global challenges that the industry is collectively addressing. Recently, a significant amount of advances have been made on post-training and inference techniques to mitigate these challenges. However, it is widely agreed that unsafe and hallucinations of LLMs intrinsically originate from pre-training, involving pre-training data and the next-token prediction learning mechanism. In this paper, we focus on enhancing pre-training data to improve the trustworthiness and safety of LLMs. Since the data is vast, it's almost impossible to entirely purge the data of factual errors, logical inconsistencies, or distributional biases. Moreover, the pre-training data lack grounding in real-world knowledge. Each piece of data is treated as a sequence of tokens rather than as a representation of a part of the world. To overcome these issues, we propose approaches to enhancing our pre-training data with its context in the world and increasing a substantial amount of data reflecting industrial scenarios. We argue that most source data are created by the authors for specific purposes in a certain spatial-temporal context. They have played a role in the real world. By incorporating related world context information, we aim to better anchor pre-training data within real-world scenarios, thereby reducing uncertainty in model training and enhancing the model's safety and trustworthiness. We refer to our Data with World Context as DWC. We continue pre-training an earlier checkpoint of JT-35B-Base with 1.5 trillion of DWC tokens. We introduce our post-training procedures to activate the potentials of DWC. Compared with the Qwen model of a similar scale, JT-Safe-35B achieves an average performance improvement of 1.79% on the Safety and Trustworthy evaluation benchmarks, while being pretrained with only 6.2 trillion tokens. 

---
# Data Unlearning Beyond Uniform Forgetting via Diffusion Time and Frequency Selection 

**Authors**: Jinseong Park, Mijung Park  

**Link**: [PDF](https://arxiv.org/pdf/2510.17917)  

**Abstract**: Data unlearning aims to remove the influence of specific training samples from a trained model without requiring full retraining. Unlike concept unlearning, data unlearning in diffusion models remains underexplored and often suffers from quality degradation or incomplete forgetting. To address this, we first observe that most existing methods attempt to unlearn the samples at all diffusion time steps equally, leading to poor-quality generation. We argue that forgetting occurs disproportionately across time and frequency, depending on the model and scenarios. By selectively focusing on specific time-frequency ranges during training, we achieve samples with higher aesthetic quality and lower noise. We validate this improvement by applying our time-frequency selective approach to diverse settings, including gradient-based and preference optimization objectives, as well as both image-level and text-to-image tasks. Finally, to evaluate both deletion and quality of unlearned data samples, we propose a simple normalized version of SSCD. Together, our analysis and methods establish a clearer understanding of the unique challenges in data unlearning for diffusion models, providing practical strategies to improve both evaluation and unlearning performance. 

---
# Self-Evidencing Through Hierarchical Gradient Decomposition: A Dissipative System That Maintains Non-Equilibrium Steady-State by Minimizing Variational Free Energy 

**Authors**: Michael James McCulloch  

**Link**: [PDF](https://arxiv.org/pdf/2510.17916)  

**Abstract**: The Free Energy Principle (FEP) states that self-organizing systems must minimize variational free energy to persist, but the path from principle to implementable algorithm has remained unclear. We present a constructive proof that the FEP can be realized through exact local credit assignment. The system decomposes gradient computation hierarchically: spatial credit via feedback alignment, temporal credit via eligibility traces, and structural credit via a Trophic Field Map (TFM) that estimates expected gradient magnitude for each connection block. We prove these mechanisms are exact at their respective levels and validate the central claim empirically: the TFM achieves 0.9693 Pearson correlation with oracle gradients. This exactness produces emergent capabilities including 98.6% retention after task interference, autonomous recovery from 75% structural damage, self-organized criticality (spectral radius p ~= 1.0$), and sample-efficient reinforcement learning on continuous control tasks without replay buffers. The architecture unifies Prigogine's dissipative structures, Friston's free energy minimization, and Hopfield's attractor dynamics, demonstrating that exact hierarchical inference over network topology can be implemented with local, biologically plausible rules. 

---
# Uncertainty-Aware Post-Hoc Calibration: Mitigating Confidently Incorrect Predictions Beyond Calibration Metrics 

**Authors**: Hassan Gharoun, Mohammad Sadegh Khorshidi, Kasra Ranjbarigderi, Fang Chen, Amir H. Gandomi  

**Link**: [PDF](https://arxiv.org/pdf/2510.17915)  

**Abstract**: Despite extensive research on neural network calibration, existing methods typically apply global transformations that treat all predictions uniformly, overlooking the heterogeneous reliability of individual predictions. Furthermore, the relationship between improved calibration and effective uncertainty-aware decision-making remains largely unexplored. This paper presents a post-hoc calibration framework that leverages prediction reliability assessment to jointly enhance calibration quality and uncertainty-aware decision-making. The framework employs proximity-based conformal prediction to stratify calibration samples into putatively correct and putatively incorrect groups based on semantic similarity in feature space. A dual calibration strategy is then applied: standard isotonic regression calibrated confidence in putatively correct predictions, while underconfidence-regularized isotonic regression reduces confidence toward uniform distributions for putatively incorrect predictions, facilitating their identification for further investigations. A comprehensive evaluation is conducted using calibration metrics, uncertainty-aware performance measures, and empirical conformal coverage. Experiments on CIFAR-10 and CIFAR-100 with BiT and CoAtNet backbones show that the proposed method achieves lower confidently incorrect predictions, and competitive Expected Calibration Error compared with isotonic and focal-loss baselines. This work bridges calibration and uncertainty quantification through instance-level adaptivity, offering a practical post-hoc solution that requires no model retraining while improving both probability alignment and uncertainty-aware decision-making. 

---
# NeuCo-Bench: A Novel Benchmark Framework for Neural Embeddings in Earth Observation 

**Authors**: Rikard Vinge, Isabelle Wittmann, Jannik Schneider, Michael Marszalek, Luis Gilch, Thomas Brunschwiler, Conrad M Albrecht  

**Link**: [PDF](https://arxiv.org/pdf/2510.17914)  

**Abstract**: We introduce NeuCo-Bench, a novel benchmark framework for evaluating (lossy) neural compression and representation learning in the context of Earth Observation (EO). Our approach builds on fixed-size embeddings that act as compact, task-agnostic representations applicable to a broad range of downstream tasks. NeuCo-Bench comprises three core components: (i) an evaluation pipeline built around reusable embeddings, (ii) a new challenge mode with a hidden-task leaderboard designed to mitigate pretraining bias, and (iii) a scoring system that balances accuracy and stability. To support reproducibility, we release SSL4EO-S12-downstream, a curated multispectral, multitemporal EO dataset. We present initial results from a public challenge at the 2025 CVPR EARTHVISION workshop and conduct ablations with state-of-the-art foundation models. NeuCo-Bench provides a first step towards community-driven, standardized evaluation of neural embeddings for EO and beyond. 

---
# TACLA: An LLM-Based Multi-Agent Tool for Transactional Analysis Training in Education 

**Authors**: Monika Zamojska, Jarosław A. Chudziak  

**Link**: [PDF](https://arxiv.org/pdf/2510.17913)  

**Abstract**: Simulating nuanced human social dynamics with Large Language Models (LLMs) remains a significant challenge, particularly in achieving psychological depth and consistent persona behavior crucial for high-fidelity training tools. This paper introduces TACLA (Transactional Analysis Contextual LLM-based Agents), a novel Multi-Agent architecture designed to overcome these limitations. TACLA integrates core principles of Transactional Analysis (TA) by modeling agents as an orchestrated system of distinct Parent, Adult, and Child ego states, each with its own pattern memory. An Orchestrator Agent prioritizes ego state activation based on contextual triggers and an agent's life script, ensuring psychologically authentic responses. Validated in an educational scenario, TACLA demonstrates realistic ego state shifts in Student Agents, effectively modeling conflict de-escalation and escalation based on different teacher intervention strategies. Evaluation shows high conversational credibility and confirms TACLA's capacity to create dynamic, psychologically-grounded social simulations, advancing the development of effective AI tools for education and beyond. 

---
# Interpretability Framework for LLMs in Undergraduate Calculus 

**Authors**: Sagnik Dakshit, Sushmita Sinha Roy  

**Link**: [PDF](https://arxiv.org/pdf/2510.17910)  

**Abstract**: Large Language Models (LLMs) are increasingly being used in education, yet their correctness alone does not capture the quality, reliability, or pedagogical validity of their problem-solving behavior, especially in mathematics, where multistep logic, symbolic reasoning, and conceptual clarity are critical. Conventional evaluation methods largely focus on final answer accuracy and overlook the reasoning process. To address this gap, we introduce a novel interpretability framework for analyzing LLM-generated solutions using undergraduate calculus problems as a representative domain. Our approach combines reasoning flow extraction and decomposing solutions into semantically labeled operations and concepts with prompt ablation analysis to assess input salience and output stability. Using structured metrics such as reasoning complexity, phrase sensitivity, and robustness, we evaluated the model behavior on real Calculus I to III university exams. Our findings revealed that LLMs often produce syntactically fluent yet conceptually flawed solutions, with reasoning patterns sensitive to prompt phrasing and input variation. This framework enables fine-grained diagnosis of reasoning failures, supports curriculum alignment, and informs the design of interpretable AI-assisted feedback tools. This is the first study to offer a structured, quantitative, and pedagogically grounded framework for interpreting LLM reasoning in mathematics education, laying the foundation for the transparent and responsible deployment of AI in STEM learning environments. 

---
# BreakFun: Jailbreaking LLMs via Schema Exploitation 

**Authors**: Amirkia Rafiei Oskooei, Mehmet S. Aktas  

**Link**: [PDF](https://arxiv.org/pdf/2510.17904)  

**Abstract**: The proficiency of Large Language Models (LLMs) in processing structured data and adhering to syntactic rules is a capability that drives their widespread adoption but also makes them paradoxically vulnerable. In this paper, we investigate this vulnerability through BreakFun, a jailbreak methodology that weaponizes an LLM's adherence to structured schemas. BreakFun employs a three-part prompt that combines an innocent framing and a Chain-of-Thought distraction with a core "Trojan Schema"--a carefully crafted data structure that compels the model to generate harmful content, exploiting the LLM's strong tendency to follow structures and schemas. We demonstrate this vulnerability is highly transferable, achieving an average success rate of 89% across 13 foundational and proprietary models on JailbreakBench, and reaching a 100% Attack Success Rate (ASR) on several prominent models. A rigorous ablation study confirms this Trojan Schema is the attack's primary causal factor. To counter this, we introduce the Adversarial Prompt Deconstruction guardrail, a defense that utilizes a secondary LLM to perform a "Literal Transcription"--extracting all human-readable text to isolate and reveal the user's true harmful intent. Our proof-of-concept guardrail demonstrates high efficacy against the attack, validating that targeting the deceptive schema is a viable mitigation strategy. Our work provides a look into how an LLM's core strengths can be turned into critical weaknesses, offering a fresh perspective for building more robustly aligned models. 

---
# The Sherpa.ai Blind Vertical Federated Learning Paradigm to Minimize the Number of Communications 

**Authors**: Alex Acero, Daniel M. Jimenez-Gutierrez, Dario Pighin, Enrique Zuazua, Joaquin Del Rio, Xabi Uribe-Etxebarria  

**Link**: [PDF](https://arxiv.org/pdf/2510.17901)  

**Abstract**: Federated Learning (FL) enables collaborative decentralized training across multiple parties (nodes) while keeping raw data private. There are two main paradigms in FL: Horizontal FL (HFL), where all participant nodes share the same feature space but hold different samples, and Vertical FL (VFL), where participants hold complementary features for the same samples. While HFL is widely adopted, VFL is employed in domains where nodes hold complementary features about the same samples. Still, VFL presents a significant limitation: the vast number of communications required during training. This compromises privacy and security, and can lead to high energy consumption, and in some cases, make model training unfeasible due to the high number of communications.
In this paper, we introduce this http URL Blind Vertical Federated Learning (SBVFL), a novel paradigm that leverages a distributed training mechanism enhanced for privacy and security. Decoupling the vast majority of node updates from the server dramatically reduces node-server communication. Experiments show that SBVFL reduces communication by ~99% compared to standard VFL while maintaining accuracy and robustness. Therefore, SBVFL enables practical, privacy-preserving VFL across sensitive domains, including healthcare, finance, manufacturing, aerospace, cybersecurity, and the defense industry. 

---
# Are LLMs Court-Ready? Evaluating Frontier Models on Indian Legal Reasoning 

**Authors**: Kush Juvekar, Arghya Bhattacharya, Sai Khadloya, Utkarsh Saxena  

**Link**: [PDF](https://arxiv.org/pdf/2510.17900)  

**Abstract**: Large language models (LLMs) are entering legal workflows, yet we lack a jurisdiction-specific framework to assess their baseline competence therein. We use India's public legal examinations as a transparent proxy. Our multi-year benchmark assembles objective screens from top national and state exams and evaluates open and frontier LLMs under real-world exam conditions. To probe beyond multiple-choice questions, we also include a lawyer-graded, paired-blinded study of long-form answers from the Supreme Court's Advocate-on-Record exam. This is, to our knowledge, the first exam-grounded, India-specific yardstick for LLM court-readiness released with datasets and protocols. Our work shows that while frontier systems consistently clear historical cutoffs and often match or exceed recent top-scorer bands on objective exams, none surpasses the human topper on long-form reasoning. Grader notes converge on three reliability failure modes: procedural or format compliance, authority or citation discipline, and forum-appropriate voice and structure. These findings delineate where LLMs can assist (checks, cross-statute consistency, statute and precedent lookups) and where human leadership remains essential: forum-specific drafting and filing, procedural and relief strategy, reconciling authorities and exceptions, and ethical, accountable judgment. 

---
# Automated Algorithm Design for Auto-Tuning Optimizers 

**Authors**: Floris-Jan Willemsen, Niki van Stein, Ben van Werkhoven  

**Link**: [PDF](https://arxiv.org/pdf/2510.17899)  

**Abstract**: Automatic performance tuning (auto-tuning) is essential for optimizing high-performance applications, where vast and irregular parameter spaces make manual exploration infeasible. Traditionally, auto-tuning relies on well-established optimization algorithms such as evolutionary algorithms, annealing methods, or surrogate model-based optimizers to efficiently find near-optimal configurations. However, designing effective optimizers remains challenging, as no single method performs best across all tuning tasks.
In this work, we explore a new paradigm: using large language models (LLMs) to automatically generate optimization algorithms tailored to auto-tuning problems. We introduce a framework that prompts LLMs with problem descriptions and search-space characteristics results to produce specialized optimization strategies, which are iteratively examined and improved.
These generated algorithms are evaluated on four real-world auto-tuning applications across six hardware platforms and compared against the state-of-the-art in optimization algorithms of two contemporary auto-tuning frameworks. The evaluation demonstrates that providing additional application- and search space-specific information in the generation stage results in an average performance improvement of 30.7\% and 14.6\%, respectively. In addition, our results show that LLM-generated optimizers can rival, and in various cases outperform, existing human-designed algorithms, with our best-performing generated optimization algorithms achieving, on average, 72.4\% improvement over state-of-the-art optimizers for auto-tuning. 

---
# L-MoE: End-to-End Training of a Lightweight Mixture of Low-Rank Adaptation Experts 

**Authors**: Shihao Ji, Zihui Song  

**Link**: [PDF](https://arxiv.org/pdf/2510.17898)  

**Abstract**: The Mixture of Experts (MoE) architecture enables the scaling of Large Language Models (LLMs) to trillions of parameters by activating a sparse subset of weights for each input, maintaining constant computational cost during inference. Concurrently, Low-Rank Adaptation (LoRA) has emerged as a dominant technique for parameter-efficiently fine-tuning LLMs on specialized tasks. In this work, we unify these two paradigms into a novel, end-to-end trainable framework named L-MoE: a Lightweight Mixture of LoRA Experts. L-MoE redefines MoE experts not as dense feed-forward networks, but as a collection of task-specialized, low-rank adapters. A lightweight gating network, trained jointly with the experts, learns to dynamically compose these LoRA adapters by computing a weighted average of their parameters for each input token. This composition is fully differentiable, allowing gradients from a standard auto-regressive language modeling objective to flow back through the entire architecture, simultaneously refining both the expert adapters and the routing strategy. This approach creates a highly parameter-efficient MoE model that is modular by design, allows for dynamic skill composition, and is trainable from end-to-end. We present the formal mathematical framework for L-MoE, detailing the differentiable routing mechanism and the joint optimization objective, thereby providing a new path toward building more efficient, scalable, and specialized language models. 

---
# Long-Context Attention Benchmark: From Kernel Efficiency to Distributed Context Parallelism 

**Authors**: Tao Bu, Qiangang Wang, Bowen Zeng, Hanwen Sun, Yunpeng Huang, Chun Cao, Jingwei Xu  

**Link**: [PDF](https://arxiv.org/pdf/2510.17896)  

**Abstract**: Transformer-based large language models (LLMs) have achieved remarkable success, yet their standard attention mechanism incurs quadratic computation and memory costs with respect to sequence length, posing a major bottleneck for long-context training. Prior work tackles this challenge along two directions: (1) kernel-level optimizations, which accelerate dense and sparse attention operators; and (2) module-level strategies, often referred to as distributed attention or context parallel training, which scale attention across multiple devices. However, systematic evaluation still remains limited: operator-level comparisons are often incomplete, while context parallel strategies are typically framework-specific, with unclear performance analysis across contexts. To address these gaps, we propose a unified benchmark that integrates representative attention kernels and context parallel mechanisms with a modular and extensible interface for evaluation. The benchmark evaluates methods along two critical dimensions: (1) attention mask patterns, which strongly affect efficiency, scalability, and usability, and (2) sequence length and distributed scale, which determine performance under extreme long-context training. Through comprehensive experiments on the cluster of up to 96 GPUs, our benchmark enables reproducible comparisons, highlights method-specific trade-offs, and provides practical guidance for designing and deploying attention mechanisms in long-context LLM training. 

---
# Hierarchical Federated Unlearning for Large Language Models 

**Authors**: Yisheng Zhong, Zhengbang Yang, Zhuangdi Zhu  

**Link**: [PDF](https://arxiv.org/pdf/2510.17895)  

**Abstract**: Large Language Models (LLMs) are increasingly integrated into real-world applications, raising concerns about privacy, security and the need to remove undesirable knowledge. Machine Unlearning has emerged as a promising solution, yet faces two key challenges: (1) practical unlearning needs are often continuous and heterogeneous, and (2) they involve decentralized, sensitive data with asymmetric access. These factors result in inter-domain and intra-domain interference, which further amplifies the dilemma of unbalanced forgetting and retaining performance. In response, we propose a federated unlearning approach for LLMs that is scalable and privacy preserving. Our method decouples unlearning and retention via task-specific adapter learning and employs a hierarchical merging strategy to mitigate conflicting objectives and enables robust, adaptable unlearning updates. Comprehensive experiments on benchmarks of WMDP, MUSE, and TOFU showed that our approach effectively handles heterogeneous unlearning requests while maintaining strong LLM utility compared with baseline methods. 

---
# MIN-Merging: Merge the Important Neurons for Model Merging 

**Authors**: Yunfei Liang  

**Link**: [PDF](https://arxiv.org/pdf/2510.17890)  

**Abstract**: Recent advances in deep learning have led to a surge of open-source models across diverse domains. While model merging offers a promising way to combine their strengths, existing approaches often suffer from parameter conflicts that degrade performance on domain-specific tasks. We propose MIN-Merging, a router-based framework that selectively merges the most important neurons to reduce such conflicts. Extensive experiments on Computer Vision(CV) and Natural Language Processing(NLP) benchmarks show that MIN-Merging achieves consistent gains on in-domain tasks while retaining the generalization ability of pretrained models on out-of-domain tasks. These results highlight its effectiveness as a practical solution to the parameter conflict problem in model merging. 

---
# Hey Pentti, We Did It!: A Fully Vector-Symbolic Lisp 

**Authors**: Eilene Tomkins-Flanagan, Mary A. Kelly  

**Link**: [PDF](https://arxiv.org/pdf/2510.17889)  

**Abstract**: Kanerva (2014) suggested that it would be possible to construct a complete Lisp out of a vector-symbolic architecture. We present the general form of a vector-symbolic representation of the five Lisp elementary functions, lambda expressions, and other auxiliary functions, found in the Lisp 1.5 specification McCarthy (1960), which is near minimal and sufficient for Turing-completeness. Our specific implementation uses holographic reduced representations Plate (1995), with a lookup table cleanup memory. Lisp, as all Turing-complete languages, is a Cartesian closed category, unusual in its proximity to the mathematical abstraction. We discuss the mathematics, the purpose, and the significance of demonstrating vector-symbolic architectures' Cartesian-closure, as well as the importance of explicitly including cleanup memories in the specification of the architecture. 

---
# Metrics and evaluations for computational and sustainable AI efficiency 

**Authors**: Hongyuan Liu, Xinyang Liu, Guosheng Hu  

**Link**: [PDF](https://arxiv.org/pdf/2510.17885)  

**Abstract**: The rapid advancement of Artificial Intelligence (AI) has created unprecedented demands for computational power, yet methods for evaluating the performance, efficiency, and environmental impact of deployed models remain fragmented. Current approaches often fail to provide a holistic view, making it difficult to compare and optimise systems across heterogeneous hardware, software stacks, and numeric precisions. To address this gap, we propose a unified and reproducible methodology for AI model inference that integrates computational and environmental metrics under realistic serving conditions. Our framework provides a pragmatic, carbon-aware evaluation by systematically measuring latency and throughput distributions, energy consumption, and location-adjusted carbon emissions, all while maintaining matched accuracy constraints for valid comparisons. We apply this methodology to multi-precision models across diverse hardware platforms, from data-centre accelerators like the GH200 to consumer-level GPUs such as the RTX 4090, running on mainstream software stacks including PyTorch, TensorRT, and ONNX Runtime. By systematically categorising these factors, our work establishes a rigorous benchmarking framework that produces decision-ready Pareto frontiers, clarifying the trade-offs between accuracy, latency, energy, and carbon. The accompanying open-source code enables independent verification and facilitates adoption, empowering researchers and practitioners to make evidence-based decisions for sustainable AI deployment. 

---
# When Intelligence Fails: An Empirical Study on Why LLMs Struggle with Password Cracking 

**Authors**: Mohammad Abdul Rehman, Syed Imad Ali Shah, Abbas Anwar, Noor Islam  

**Link**: [PDF](https://arxiv.org/pdf/2510.17884)  

**Abstract**: The remarkable capabilities of Large Language Models (LLMs) in natural language understanding and generation have sparked interest in their potential for cybersecurity applications, including password guessing. In this study, we conduct an empirical investigation into the efficacy of pre-trained LLMs for password cracking using synthetic user profiles. Specifically, we evaluate the performance of state-of-the-art open-source LLMs such as TinyLLaMA, Falcon-RW-1B, and Flan-T5 by prompting them to generate plausible passwords based on structured user attributes (e.g., name, birthdate, hobbies). Our results, measured using Hit@1, Hit@5, and Hit@10 metrics under both plaintext and SHA-256 hash comparisons, reveal consistently poor performance, with all models achieving less than 1.5% accuracy at Hit@10. In contrast, traditional rule-based and combinator-based cracking methods demonstrate significantly higher success rates. Through detailed analysis and visualization, we identify key limitations in the generative reasoning of LLMs when applied to the domain-specific task of password guessing. Our findings suggest that, despite their linguistic prowess, current LLMs lack the domain adaptation and memorization capabilities required for effective password inference, especially in the absence of supervised fine-tuning on leaked password datasets. This study provides critical insights into the limitations of LLMs in adversarial contexts and lays the groundwork for future efforts in secure, privacy-preserving, and robust password modeling. 

---
# From Flows to Words: Can Zero-/Few-Shot LLMs Detect Network Intrusions? A Grammar-Constrained, Calibrated Evaluation on UNSW-NB15 

**Authors**: Mohammad Abdul Rehman, Syed Imad Ali Shah, Abbas n=Anwar, Noor Islam  

**Link**: [PDF](https://arxiv.org/pdf/2510.17883)  

**Abstract**: Large Language Models (LLMs) can reason over natural-language inputs, but their role in intrusion detection without fine-tuning remains uncertain. This study evaluates a prompt-only approach on UNSW-NB15 by converting each network flow to a compact textual record and augmenting it with lightweight, domain-inspired boolean flags (asymmetry, burst rate, TTL irregularities, timer anomalies, rare service/state, short bursts). To reduce output drift and support measurement, the model is constrained to produce structured, grammar-valid responses, and a single decision threshold is calibrated on a small development split. We compare zero-shot, instruction-guided, and few-shot prompting to strong tabular and neural baselines under identical splits, reporting accuracy, precision, recall, F1, and macro scores. Empirically, unguided prompting is unreliable, while instructions plus flags substantially improve detection quality; adding calibrated scoring further stabilizes results. On a balanced subset of two hundred flows, a 7B instruction-tuned model with flags reaches macro-F1 near 0.78; a lighter 3B model with few-shot cues and calibration attains F1 near 0.68 on one thousand examples. As the evaluation set grows to two thousand flows, decision quality decreases, revealing sensitivity to coverage and prompting. Tabular baselines remain more stable and faster, yet the prompt-only pipeline requires no gradient training, produces readable artifacts, and adapts easily through instructions and flags. Contributions include a flow-to-text protocol with interpretable cues, a calibration method for thresholding, a systematic baseline comparison, and a reproducibility bundle with prompts, grammar, metrics, and figures. 

---
# Does GenAI Rewrite How We Write? An Empirical Study on Two-Million Preprints 

**Authors**: Minfeng Qi, Zhongmin Cao, Qin Wang, Ningran Li, Tianqing Zhu  

**Link**: [PDF](https://arxiv.org/pdf/2510.17882)  

**Abstract**: Preprint repositories become central infrastructures for scholarly communication. Their expansion transforms how research is circulated and evaluated before journal publication. Generative large language models (LLMs) introduce a further potential disruption by altering how manuscripts are written. While speculation abounds, systematic evidence of whether and how LLMs reshape scientific publishing remains limited.
This paper addresses the gap through a large-scale analysis of more than 2.1 million preprints spanning 2016--2025 (115 months) across four major repositories (i.e., arXiv, bioRxiv, medRxiv, SocArXiv). We introduce a multi-level analytical framework that integrates interrupted time-series models, collaboration and productivity metrics, linguistic profiling, and topic modeling to assess changes in volume, authorship, style, and disciplinary orientation. Our findings reveal that LLMs have accelerated submission and revision cycles, modestly increased linguistic complexity, and disproportionately expanded AI-related topics, while computationally intensive fields benefit more than others. These results show that LLMs act less as universal disruptors than as selective catalysts, amplifying existing strengths and widening disciplinary divides. By documenting these dynamics, the paper provides the first empirical foundation for evaluating the influence of generative AI on academic publishing and highlights the need for governance frameworks that preserve trust, fairness, and accountability in an AI-enabled research ecosystem. 

---
# POPI: Personalizing LLMs via Optimized Natural Language Preference Inference 

**Authors**: Yizhuo Chen, Xin Liu, Ruijie Wang, Zheng Li, Pei Chen, Changlong Yu, Priyanka Nigam, Meng Jiang, Bing Yin  

**Link**: [PDF](https://arxiv.org/pdf/2510.17881)  

**Abstract**: Large language models (LLMs) achieve strong benchmark performance, yet user experiences remain inconsistent due to diverse preferences in style, tone, and reasoning mode. Nevertheless, existing alignment techniques such as reinforcement learning from human feedback (RLHF) or Direct Preference Optimization (DPO) largely optimize toward population-level averages and overlook individual variation. Naive personalization strategies like per-user fine-tuning are computationally prohibitive, and in-context approaches that prepend raw user signals often suffer from inefficiency and noise. To address these challenges, we propose POPI, a general framework that introduces a preference inference model to distill heterogeneous user signals into concise natural language summaries. These summaries act as transparent, compact, and transferable personalization representations that condition a shared generation model to produce personalized responses. POPI jointly optimizes both preference inference and personalized generation under a unified objective using reinforcement learning, ensuring summaries maximally encode useful preference information. Extensive experiments across four personalization benchmarks demonstrate that POPI consistently improves personalization accuracy while reducing context overhead by a large margin. Moreover, optimized summaries seamlessly transfer to frozen off-the-shelf LLMs, enabling plug-and-play personalization without weight updates. 

---
# Outraged AI: Large language models prioritise emotion over cost in fairness enforcement 

**Authors**: Hao Liu, Yiqing Dai, Haotian Tan, Yu Lei, Yujia Zhou, Zhen Wu  

**Link**: [PDF](https://arxiv.org/pdf/2510.17880)  

**Abstract**: Emotions guide human decisions, but whether large language models (LLMs) use emotion similarly remains unknown. We tested this using altruistic third-party punishment, where an observer incurs a personal cost to enforce fairness, a hallmark of human morality and often driven by negative emotion. In a large-scale comparison of 4,068 LLM agents with 1,159 adults across 796,100 decisions, LLMs used emotion to guide punishment, sometimes even more strongly than humans did: Unfairness elicited stronger negative emotion that led to more punishment; punishing unfairness produced more positive emotion than accepting; and critically, prompting self-reports of emotion causally increased punishment. However, mechanisms diverged: LLMs prioritized emotion over cost, enforcing norms in an almost all-or-none manner with reduced cost sensitivity, whereas humans balanced fairness and cost. Notably, reasoning models (o3-mini, DeepSeek-R1) were more cost-sensitive and closer to human behavior than foundation models (GPT-3.5, DeepSeek-V3), yet remained heavily emotion-driven. These findings provide the first causal evidence of emotion-guided moral decisions in LLMs and reveal deficits in cost calibration and nuanced fairness judgements, reminiscent of early-stage human responses. We propose that LLMs progress along a trajectory paralleling human development; future models should integrate emotion with context-sensitive reasoning to achieve human-like emotional intelligence. 

---
# Decoding Listeners Identity: Person Identification from EEG Signals Using a Lightweight Spiking Transformer 

**Authors**: Zheyuan Lin, Siqi Cai, Haizhou Li  

**Link**: [PDF](https://arxiv.org/pdf/2510.17879)  

**Abstract**: EEG-based person identification enables applications in security, personalized brain-computer interfaces (BCIs), and cognitive monitoring. However, existing techniques often rely on deep learning architectures at high computational cost, limiting their scope of applications. In this study, we propose a novel EEG person identification approach using spiking neural networks (SNNs) with a lightweight spiking transformer for efficiency and effectiveness. The proposed SNN model is capable of handling the temporal complexities inherent in EEG signals. On the EEG-Music Emotion Recognition Challenge dataset, the proposed model achieves 100% classification accuracy with less than 10% energy consumption of traditional deep neural networks. This study offers a promising direction for energy-efficient and high-performance BCIs. The source code is available at this https URL. 

---
# DRL-Based Resource Allocation for Energy-Efficient IRS-Assisted UAV Spectrum Sharing Systems 

**Authors**: Yiheng Wang  

**Link**: [PDF](https://arxiv.org/pdf/2510.17877)  

**Abstract**: Intelligent reflecting surface (IRS) assisted unmanned aerial vehicle (UAV) systems provide a new paradigm for reconfigurable and flexible wireless communications. To enable more energy efficient and spectrum efficient IRS assisted UAV wireless communications, this paper introduces a novel IRS-assisted UAV enabled spectrum sharing system with orthogonal frequency division multiplexing (OFDM). The goal is to maximize the energy efficiency (EE) of the secondary network by jointly optimizing the beamforming, subcarrier allocation, IRS phase shifts, and the UAV trajectory subject to practical transmit power and passive reflection constraints as well as UAV physical limitations. A physically grounded propulsion-energy model is adopted, with its tight upper bound used to form a tractable EE lower bound for the spectrum sharing system. To handle highly non convex, time coupled optimization problems with a mixed continuous and discrete policy space, we develop a deep reinforcement learning (DRL) approach based on the actor critic framework. Extended experiments show the significant EE improvement of the proposed DRL-based approach compared to several benchmark schemes, thus demonstrating the effectiveness and robustness of the proposed approach with mobility. 

---
# 3D Weakly Supervised Semantic Segmentation via Class-Aware and Geometry-Guided Pseudo-Label Refinement 

**Authors**: Xiaoxu Xu, Xuexun Liu, Jinlong Li, Yitian Yuan, Qiudan Zhang, Lin Ma, Nicu Sebe, Xu Wang  

**Link**: [PDF](https://arxiv.org/pdf/2510.17875)  

**Abstract**: 3D weakly supervised semantic segmentation (3D WSSS) aims to achieve semantic segmentation by leveraging sparse or low-cost annotated data, significantly reducing reliance on dense point-wise annotations. Previous works mainly employ class activation maps or pre-trained vision-language models to address this challenge. However, the low quality of pseudo-labels and the insufficient exploitation of 3D geometric priors jointly create significant technical bottlenecks in developing high-performance 3D WSSS models. In this paper, we propose a simple yet effective 3D weakly supervised semantic segmentation method that integrates 3D geometric priors into a class-aware guidance mechanism to generate high-fidelity pseudo labels. Concretely, our designed methodology first employs Class-Aware Label Refinement module to generate more balanced and accurate pseudo labels for semantic categrories. This initial refinement stage focuses on enhancing label quality through category-specific optimization. Subsequently, the Geometry-Aware Label Refinement component is developed, which strategically integrates implicit 3D geometric constraints to effectively filter out low-confidence pseudo labels that fail to comply with geometric plausibility. Moreover, to address the challenge of extensive unlabeled regions, we propose a Label Update strategy that integrates Self-Training to propagate labels into these areas. This iterative process continuously enhances pseudo-label quality while expanding label coverage, ultimately fostering the development of high-performance 3D WSSS models. Comprehensive experimental validation reveals that our proposed methodology achieves state-of-the-art performance on both ScanNet and S3DIS benchmarks while demonstrating remarkable generalization capability in unsupervised settings, maintaining competitive accuracy through its robust design. 

---
# Repairing Tool Calls Using Post-tool Execution Reflection and RAG 

**Authors**: Jason Tsay, Zidane Wright, Gaodan Fang, Kiran Kate, Saurabh Jha, Yara Rizk  

**Link**: [PDF](https://arxiv.org/pdf/2510.17874)  

**Abstract**: Agentic systems interact with external systems by calling tools such as Python functions, REST API endpoints, or command line tools such as kubectl in Kubernetes. These tool calls often fail for various syntactic and semantic reasons. Some less obvious semantic errors can only be identified and resolved after analyzing the tool's response. To repair these errors, we develop a post-tool execution reflection component that combines large language model (LLM)-based reflection with domain-specific retrieval-augmented generation (RAG) using documents describing both the specific tool being called and troubleshooting documents related to the tool. For this paper, we focus on the use case of the kubectl command line tool to manage Kubernetes, a platform for orchestrating cluster applications. Through a larger empirical study and a smaller manual evaluation, we find that our RAG-based reflection will repair kubectl commands such that they are both more likely to successfully execute (pass rate) for 55% of our models evaluated and 36% more likely to correctly answer the user query on average. We find that troubleshooting documents improve pass rate compared to official documentation by an average of 10%. 

---
# Auditing and Mitigating Bias in Gender Classification Algorithms: A Data-Centric Approach 

**Authors**: Tadesse K Bahiru, Natnael Tilahun Sinshaw, Teshager Hailemariam Moges, Dheeraj Kumar Singh  

**Link**: [PDF](https://arxiv.org/pdf/2510.17873)  

**Abstract**: Gender classification systems often inherit and amplify demographic imbalances in their training data. We first audit five widely used gender classification datasets, revealing that all suffer from significant intersectional underrepresentation. To measure the downstream impact of these flaws, we train identical MobileNetV2 classifiers on the two most balanced of these datasets, UTKFace and FairFace. Our fairness evaluation shows that even these models exhibit significant bias, misclassifying female faces at a higher rate than male faces and amplifying existing racial skew. To counter these data-induced biases, we construct BalancedFace, a new public dataset created by blending images from FairFace and UTKFace, supplemented with images from other collections to fill missing demographic gaps. It is engineered to equalize subgroup shares across 189 intersections of age, race, and gender using only real, unedited images. When a standard classifier is trained on BalancedFace, it reduces the maximum True Positive Rate gap across racial subgroups by over 50% and brings the average Disparate Impact score 63% closer to the ideal of 1.0 compared to the next-best dataset, all with a minimal loss of overall accuracy. These results underline the profound value of data-centric interventions and provide an openly available resource for fair gender classification research. 

---
# A Survey of Recursive and Recurrent Neural Networks 

**Authors**: Jian-wei Liu, Bing-rong Xu, Zhi-yan Song  

**Link**: [PDF](https://arxiv.org/pdf/2510.17867)  

**Abstract**: In this paper, the branches of recursive and recurrent neural networks are classified in detail according to the network structure, training objective function and learning algorithm implementation. They are roughly divided into three categories: The first category is General Recursive and Recurrent Neural Networks, including Basic Recursive and Recurrent Neural Networks, Long Short Term Memory Recursive and Recurrent Neural Networks, Convolutional Recursive and Recurrent Neural Networks, Differential Recursive and Recurrent Neural Networks, One-Layer Recursive and Recurrent Neural Networks, High-Order Recursive and Recurrent Neural Networks, Highway Networks, Multidimensional Recursive and Recurrent Neural Networks, Bidirectional Recursive and Recurrent Neural Networks; the second category is Structured Recursive and Recurrent Neural Networks, including Grid Recursive and Recurrent Neural Networks, Graph Recursive and Recurrent Neural Networks, Temporal Recursive and Recurrent Neural Networks, Lattice Recursive and Recurrent Neural Networks, Hierarchical Recursive and Recurrent Neural Networks, Tree Recursive and Recurrent Neural Networks; the third category is Other Recursive and Recurrent Neural Networks, including Array Long Short Term Memory, Nested and Stacked Recursive and Recurrent Neural Networks, Memory Recursive and Recurrent Neural Networks. Various networks cross each other and even rely on each other to form a complex network of relationships. In the context of the development and convergence of various networks, many complex sequence, speech and image problems are solved. After a detailed description of the principle and structure of the above model and model deformation, the research progress and application of each model are described, and finally the recursive and recurrent neural network models are prospected and summarized. 

---
# MUSE: Model-based Uncertainty-aware Similarity Estimation for zero-shot 2D Object Detection and Segmentation 

**Authors**: Sungmin Cho, Sungbum Park, Insoo Oh  

**Link**: [PDF](https://arxiv.org/pdf/2510.17866)  

**Abstract**: In this work, we introduce MUSE (Model-based Uncertainty-aware Similarity Estimation), a training-free framework designed for model-based zero-shot 2D object detection and segmentation. MUSE leverages 2D multi-view templates rendered from 3D unseen objects and 2D object proposals extracted from input query images. In the embedding stage, it integrates class and patch embeddings, where the patch embeddings are normalized using generalized mean pooling (GeM) to capture both global and local representations efficiently. During the matching stage, MUSE employs a joint similarity metric that combines absolute and relative similarity scores, enhancing the robustness of matching under challenging scenarios. Finally, the similarity score is refined through an uncertainty-aware object prior that adjusts for proposal reliability. Without any additional training or fine-tuning, MUSE achieves state-of-the-art performance on the BOP Challenge 2025, ranking first across the Classic Core, H3, and Industrial tracks. These results demonstrate that MUSE offers a powerful and generalizable framework for zero-shot 2D object detection and segmentation. 

---
# Deploying Atmospheric and Oceanic AI Models on Chinese Hardware and Framework: Migration Strategies, Performance Optimization and Analysis 

**Authors**: Yuze Sun, Wentao Luo, Yanfei Xiang, Jiancheng Pan, Jiahao Li, Quan Zhang, Xiaomeng Huang  

**Link**: [PDF](https://arxiv.org/pdf/2510.17852)  

**Abstract**: With the growing role of artificial intelligence in climate and weather research, efficient model training and inference are in high demand. Current models like FourCastNet and AI-GOMS depend heavily on GPUs, limiting hardware independence, especially for Chinese domestic hardware and frameworks. To address this issue, we present a framework for migrating large-scale atmospheric and oceanic models from PyTorch to MindSpore and optimizing for Chinese chips, and evaluating their performance against GPUs. The framework focuses on software-hardware adaptation, memory optimization, and parallelism. Furthermore, the model's performance is evaluated across multiple metrics, including training speed, inference speed, model accuracy, and energy efficiency, with comparisons against GPU-based implementations. Experimental results demonstrate that the migration and optimization process preserves the models' original accuracy while significantly reducing system dependencies and improving operational efficiency by leveraging Chinese chips as a viable alternative for scientific computing. This work provides valuable insights and practical guidance for leveraging Chinese domestic chips and frameworks in atmospheric and oceanic AI model development, offering a pathway toward greater technological independence. 

---
# Pre to Post-Treatment Glioblastoma MRI Prediction using a Latent Diffusion Model 

**Authors**: Alexandre G. Leclercq, Sébastien Bougleux, Noémie N. Moreau, Alexis Desmonts, Romain Hérault, Aurélien Corroyer-Dulmont  

**Link**: [PDF](https://arxiv.org/pdf/2510.17851)  

**Abstract**: Glioblastoma (GBM) is an aggressive primary brain tumor with a median survival of approximately 15 months. In clinical practice, the Stupp protocol serves as the standard first-line treatment. However, patients exhibit highly heterogeneous therapeutic responses which required at least two months before first visual impact can be observed, typically with MRI. Early prediction treatment response is crucial for advancing personalized medicine. Disease Progression Modeling (DPM) aims to capture the trajectory of disease evolution, while Treatment Response Prediction (TRP) focuses on assessing the impact of therapeutic interventions. Whereas most TRP approaches primarly rely on timeseries data, we consider the problem of early visual TRP as a slice-to-slice translation model generating post-treatment MRI from a pre-treatment MRI, thus reflecting the tumor evolution. To address this problem we propose a Latent Diffusion Model with a concatenation-based conditioning from the pre-treatment MRI and the tumor localization, and a classifier-free guidance to enhance generation quality using survival information, in particular post-treatment tumor evolution. Our model were trained and tested on a local dataset consisting of 140 GBM patients collected at Centre François Baclesse. For each patient we collected pre and post T1-Gd MRI, tumor localization manually delineated in the pre-treatment MRI by medical experts, and survival information. 

---
# CARLE: A Hybrid Deep-Shallow Learning Framework for Robust and Explainable RUL Estimation of Rolling Element Bearings 

**Authors**: Waleed Razzaq, Yun-Bo Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2510.17846)  

**Abstract**: Prognostic Health Management (PHM) systems monitor and predict equipment health. A key task is Remaining Useful Life (RUL) estimation, which predicts how long a component, such as a rolling element bearing, will operate before failure. Many RUL methods exist but often lack generalizability and robustness under changing operating conditions. This paper introduces CARLE, a hybrid AI framework that combines deep and shallow learning to address these challenges. CARLE uses Res-CNN and Res-LSTM blocks with multi-head attention and residual connections to capture spatial and temporal degradation patterns, and a Random Forest Regressor (RFR) for stable, accurate RUL prediction. A compact preprocessing pipeline applies Gaussian filtering for noise reduction and Continuous Wavelet Transform (CWT) for time-frequency feature extraction. We evaluate CARLE on the XJTU-SY and PRONOSTIA bearing datasets. Ablation studies measure each component's contribution, while noise and cross-domain experiments test robustness and generalization. Comparative results show CARLE outperforms several state-of-the-art methods, especially under dynamic conditions. Finally, we analyze model interpretability with LIME and SHAP to assess transparency and trustworthiness. 

---
# MAT-Agent: Adaptive Multi-Agent Training Optimization 

**Authors**: Jusheng Zhang, Kaitong Cai, Yijia Fan, Ningyuan Liu, Keze Wang  

**Link**: [PDF](https://arxiv.org/pdf/2510.17845)  

**Abstract**: Multi-label image classification demands adaptive training strategies to navigate complex, evolving visual-semantic landscapes, yet conventional methods rely on static configurations that falter in dynamic settings. We propose MAT-Agent, a novel multi-agent framework that reimagines training as a collaborative, real-time optimization process. By deploying autonomous agents to dynamically tune data augmentation, optimizers, learning rates, and loss functions, MAT-Agent leverages non-stationary multi-armed bandit algorithms to balance exploration and exploitation, guided by a composite reward harmonizing accuracy, rare-class performance, and training stability. Enhanced with dual-rate exponential moving average smoothing and mixed-precision training, it ensures robustness and efficiency. Extensive experiments across Pascal VOC, COCO, and VG-256 demonstrate MAT-Agent's superiority: it achieves an mAP of 97.4 (vs. 96.2 for PAT-T), OF1 of 92.3, and CF1 of 91.4 on Pascal VOC; an mAP of 92.8 (vs. 92.0 for HSQ-CvN), OF1 of 88.2, and CF1 of 87.1 on COCO; and an mAP of 60.9, OF1 of 70.8, and CF1 of 61.1 on VG-256. With accelerated convergence and robust cross-domain generalization, MAT-Agent offers a scalable, intelligent solution for optimizing complex visual models, paving the way for adaptive deep learning advancements. 

---
# Modeling Layered Consciousness with Multi-Agent Large Language Models 

**Authors**: Sang Hun Kim, Jongmin Lee, Dongkyu Park, So Young Lee, Yosep Chong  

**Link**: [PDF](https://arxiv.org/pdf/2510.17844)  

**Abstract**: We propose a multi-agent framework for modeling artificial consciousness in large language models (LLMs), grounded in psychoanalytic theory. Our \textbf{Psychodynamic Model} simulates self-awareness, preconsciousness, and unconsciousness through agent interaction, guided by a Personalization Module combining fixed traits and dynamic needs. Using parameter-efficient fine-tuning on emotionally rich dialogues, the system was evaluated across eight personalized conditions. An LLM as a judge approach showed a 71.2\% preference for the fine-tuned model, with improved emotional depth and reduced output variance, demonstrating its potential for adaptive, personalized cognition. 

---
# GRETEL: A Goal-driven Retrieval and Execution-based Trial Framework for LLM Tool Selection Enhancing 

**Authors**: Zongze Wu, Yani Guo, Churong Liang, Runnan Li  

**Link**: [PDF](https://arxiv.org/pdf/2510.17843)  

**Abstract**: Despite remarkable advances in Large Language Model capabilities, tool retrieval for agent-based systems remains fundamentally limited by reliance on semantic similarity, which fails to capture functional viability. Current methods often retrieve textually relevant but functionally inoperative tools due to parameter mismatches, authentication failures, and execution constraints--a phenomenon we term the semantic-functional gap. We introduce GRETEL, to address this gap through systematic empirical validation. GRETEL implements an agentic workflow that processes semantically retrieved candidates through sandboxed plan-execute-evaluate cycles, generating execution-grounded evidence to distinguish truly functional tools from merely descriptive matches. Our comprehensive evaluation on the ToolBench benchmark demonstrates substantial improvements across all metrics: Pass Rate (at 10) increases from 0.690 to 0.826, Recall (at 10) improves from 0.841 to 0.867, and NDCG (at 10) rises from 0.807 to 0.857.. These results establish that execution-based validation provides a more reliable foundation for tool selection than semantic similarity alone, enabling more robust agent performance in real-world applications. 

---
# Brain-Language Model Alignment: Insights into the Platonic Hypothesis and Intermediate-Layer Advantage 

**Authors**: Ángela López-Cardona, Sebastián Idesis, Mireia Masias-Bruns, Sergi Abadal, Ioannis Arapakis  

**Link**: [PDF](https://arxiv.org/pdf/2510.17833)  

**Abstract**: Do brains and language models converge toward the same internal representations of the world? Recent years have seen a rise in studies of neural activations and model alignment. In this work, we review 25 fMRI-based studies published between 2023 and 2025 and explicitly confront their findings with two key hypotheses: (i) the Platonic Representation Hypothesis -- that as models scale and improve, they converge to a representation of the real world, and (ii) the Intermediate-Layer Advantage -- that intermediate (mid-depth) layers often encode richer, more generalizable features. Our findings provide converging evidence that models and brains may share abstract representational structures, supporting both hypotheses and motivating further research on brain-model alignment. 

---
# Synthetic EEG Generation using Diffusion Models for Motor Imagery Tasks 

**Authors**: Henrique de Lima Alexandre, Clodoaldo Aparecido de Moraes Lima  

**Link**: [PDF](https://arxiv.org/pdf/2510.17832)  

**Abstract**: Electroencephalography (EEG) is a widely used, non-invasive method for capturing brain activity, and is particularly relevant for applications in Brain-Computer Interfaces (BCI). However, collecting high-quality EEG data remains a major challenge due to sensor costs, acquisition time, and inter-subject variability. To address these limitations, this study proposes a methodology for generating synthetic EEG signals associated with motor imagery brain tasks using Diffusion Probabilistic Models (DDPM). The approach involves preprocessing real EEG data, training a diffusion model to reconstruct EEG channels from noise, and evaluating the quality of the generated signals through both signal-level and task-level metrics. For validation, we employed classifiers such as K-Nearest Neighbors (KNN), Convolutional Neural Networks (CNN), and U-Net to compare the performance of synthetic data against real data in classification tasks. The generated data achieved classification accuracies above 95%, with low mean squared error and high correlation with real signals.
Our results demonstrate that synthetic EEG signals produced by diffusion models can effectively complement datasets, improving classification performance in EEG-based BCIs and addressing data scarcity. 

---
# Multi-Agent Design Assistant for the Simulation of Inertial Fusion Energy 

**Authors**: Meir H. Shachar, Dane M. Sterbentz, Harshitha Menon, Charles F. Jekel, M. Giselle Fernández-Godino, Yue Hao, Kevin Korner, Robert Rieben, Daniel A. White, William J. Schill, Jonathan L. Belof  

**Link**: [PDF](https://arxiv.org/pdf/2510.17830)  

**Abstract**: Inertial fusion energy promises nearly unlimited, clean power if it can be achieved. However, the design and engineering of fusion systems requires controlling and manipulating matter at extreme energies and timescales; the shock physics and radiation transport governing the physical behavior under these conditions are complex requiring the development, calibration, and use of predictive multiphysics codes to navigate the highly nonlinear and multi-faceted design landscape. We hypothesize that artificial intelligence reasoning models can be combined with physics codes and emulators to autonomously design fusion fuel capsules. In this article, we construct a multi-agent system where natural language is utilized to explore the complex physics regimes around fusion energy. The agentic system is capable of executing a high-order multiphysics inertial fusion computational code. We demonstrate the capacity of the multi-agent design assistant to both collaboratively and autonomously manipulate, navigate, and optimize capsule geometry while accounting for high fidelity physics that ultimately achieve simulated ignition via inverse design. 

---
# Speak to a Protein: An Interactive Multimodal Co-Scientist for Protein Analysis 

**Authors**: Carles Navarro, Mariona Torrens, Philipp Thölke, Stefan Doerr, Gianni De Fabritiis  

**Link**: [PDF](https://arxiv.org/pdf/2510.17826)  

**Abstract**: Building a working mental model of a protein typically requires weeks of reading, cross-referencing crystal and predicted structures, and inspecting ligand complexes, an effort that is slow, unevenly accessible, and often requires specialized computational skills. We introduce \emph{Speak to a Protein}, a new capability that turns protein analysis into an interactive, multimodal dialogue with an expert co-scientist. The AI system retrieves and synthesizes relevant literature, structures, and ligand data; grounds answers in a live 3D scene; and can highlight, annotate, manipulate and see the visualization. It also generates and runs code when needed, explaining results in both text and graphics. We demonstrate these capabilities on relevant proteins, posing questions about binding pockets, conformational changes, or structure-activity relationships to test ideas in real-time. \emph{Speak to a Protein} reduces the time from question to evidence, lowers the barrier to advanced structural analysis, and enables hypothesis generation by tightly coupling language, code, and 3D structures. \emph{Speak to a Protein} is freely accessible at this https URL. 

---
# Carbon-Aware Orchestration of Integrated Satellite Aerial Terrestrial Networks via Digital Twin 

**Authors**: Shumaila Javaid, Nasir Saeed  

**Link**: [PDF](https://arxiv.org/pdf/2510.17825)  

**Abstract**: Integrated Satellite Aerial Terrestrial Networks (ISATNs) are envisioned as key enablers of 6G, providing global connectivity for applications such as autonomous transportation, Industrial IoT, and disaster response. Their large-scale deployment, however, risks unsustainable energy use and carbon emissions. This work advances prior energy-aware studies by proposing a carbon-aware orchestration framework for ISATNs that leverages Digital Twin (DT) technology. The framework adopts grams of CO$_2$-equivalent per bit (gCO$_2$/bit) as a primary sustainability metric and implements a multi timescale Plan Do Check Act (PDCA) loop that combines day-ahead forecasting with real-time adaptive optimization. ISATN-specific control knobs, including carbon-aware handovers, UAV duty cycling, and renewable-aware edge placement, are exploited to reduce emissions. Simulation results with real carbon intensity data show up to 29\% lower gCO$_2$/bit than QoS-only orchestration, while improving renewable utilization and resilience under adverse events. 

---
# A Biophysical-Model-Informed Source Separation Framework For EMG Decomposition 

**Authors**: D. Halatsis, P. Mamidanna, J. Pereira, D. Farina  

**Link**: [PDF](https://arxiv.org/pdf/2510.17822)  

**Abstract**: Recent advances in neural interfacing have enabled significant improvements in human-computer interaction, rehabilitation, and neuromuscular diagnostics. Motor unit (MU) decomposition from surface electromyography (sEMG) is a key technique for extracting neural drive information, but traditional blind source separation (BSS) methods fail to incorporate biophysical constraints, limiting their accuracy and interpretability. In this work, we introduce a novel Biophysical-Model-Informed Source Separation (BMISS) framework, which integrates anatomically accurate forward EMG models into the decomposition process. By leveraging MRI-based anatomical reconstructions and generative modeling, our approach enables direct inversion of a biophysically accurate forward model to estimate both neural drive and motor neuron properties in an unsupervised manner. Empirical validation in a controlled simulated setting demonstrates that BMISS achieves higher fidelity motor unit estimation while significantly reducing computational cost compared to traditional methods. This framework paves the way for non-invasive, personalized neuromuscular assessments, with potential applications in clinical diagnostics, prosthetic control, and neurorehabilitation. 

---
# LLM Assisted Alpha Fairness for 6 GHz WiFi and NR_U Coexistence: An Agentic Orchestrator for Throughput, Energy, and SLA 

**Authors**: Qun Wang, Yingzhou Lu, Guiran Liu, Binrong Zhu, Yang Liu  

**Link**: [PDF](https://arxiv.org/pdf/2510.17814)  

**Abstract**: Unlicensed 6GHz is becoming a primary workhorse for high-capacity access, with Wi-Fi and 5G NR-U competing for the same channels under listen-before-talk (LBT) rules. Operating in this regime requires decisions that jointly trade throughput, energy, and service-level objectives while remaining safe and auditable. We present an agentic controller that separates {policy} from {execution}. At the start of each scheduling epoch the agent summarizes telemetry (per-channel busy and baseline LBT failure; per-user CQI, backlog, latency, battery, priority, and power mode) and invokes a large language model (LLM) to propose a small set of interpretable knobs: a fairness index \alpha, per-channel duty-cycle caps for Wi-Fi/NR-U, and class weights. A deterministic optimizer then enforces feasibility and computes an \alpha-fair allocation that internalizes LBT losses and energy cost; malformed or unsafe policies are clamped and fall back to a rule baseline. In a 6GHz simulator with two 160MHz channels and mixed Wi-Fi/NR-U users, LLM-assisted policies consistently improve energy efficiency while keeping throughput competitive with a strong rule baseline. One LLM lowers total energy by 35.3% at modest throughput loss, and another attains the best overall trade-off, finishing with higher total bits (+3.5%) and higher bits/J (+12.2%) than the baseline. We release code, per-epoch logs, and plotting utilities to reproduce all figures and numbers, illustrating how transparent, policy-level LLM guidance can safely improve wireless coexistence. 

---
# Visual Space Optimization for Zero-shot Learning 

**Authors**: Xinsheng Wang, Shanmin Pang, Jihua Zhu, Zhongyu Li, Zhiqiang Tian, Yaochen Li  

**Link**: [PDF](https://arxiv.org/pdf/1907.00330)  

**Abstract**: Zero-shot learning, which aims to recognize new categories that are not included in the training set, has gained popularity owing to its potential ability in the real-word applications. Zero-shot learning models rely on learning an embedding space, where both semantic descriptions of classes and visual features of instances can be embedded for nearest neighbor search. Recently, most of the existing works consider the visual space formulated by deep visual features as an ideal choice of the embedding space. However, the discrete distribution of instances in the visual space makes the data structure unremarkable. We argue that optimizing the visual space is crucial as it allows semantic vectors to be embedded into the visual space more effectively. In this work, we propose two strategies to accomplish this purpose. One is the visual prototype based method, which learns a visual prototype for each visual class, so that, in the visual space, a class can be represented by a prototype feature instead of a series of discrete visual features. The other is to optimize the visual feature structure in an intermediate embedding space, and in this method we successfully devise a multilayer perceptron framework based algorithm that is able to learn the common intermediate embedding space and meanwhile to make the visual data structure more distinctive. Through extensive experimental evaluation on four benchmark datasets, we demonstrate that optimizing visual space is beneficial for zero-shot learning. Besides, the proposed prototype based method achieves the new state-of-the-art performance. 

---
