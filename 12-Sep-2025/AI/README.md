# The Illusion of Diminishing Returns: Measuring Long Horizon Execution in LLMs 

**Authors**: Akshit Sinha, Arvindh Arun, Shashwat Goel, Steffen Staab, Jonas Geiping  

**Link**: [PDF](https://arxiv.org/pdf/2509.09677)  

**Abstract**: Does continued scaling of large language models (LLMs) yield diminishing returns? Real-world value often stems from the length of task an agent can complete. We start this work by observing the simple but counterintuitive fact that marginal gains in single-step accuracy can compound into exponential improvements in the length of a task a model can successfully complete. Then, we argue that failures of LLMs when simple tasks are made longer arise from mistakes in execution, rather than an inability to reason. We propose isolating execution capability, by explicitly providing the knowledge and plan needed to solve a long-horizon task. We find that larger models can correctly execute significantly more turns even when small models have 100\% single-turn accuracy. We observe that the per-step accuracy of models degrades as the number of steps increases. This is not just due to long-context limitations -- curiously, we observe a self-conditioning effect -- models become more likely to make mistakes when the context contains their errors from prior turns. Self-conditioning does not reduce by just scaling the model size. In contrast, recent thinking models do not self-condition, and can also execute much longer tasks in a single turn. We conclude by benchmarking frontier thinking models on the length of task they can execute in a single turn. Overall, by focusing on the ability to execute, we hope to reconcile debates on how LLMs can solve complex reasoning problems yet fail at simple tasks when made longer, and highlight the massive benefits of scaling model size and sequential test-time compute for long-horizon tasks. 

---
# Boosting Embodied AI Agents through Perception-Generation Disaggregation and Asynchronous Pipeline Execution 

**Authors**: Shulai Zhang, Ao Xu, Quan Chen, Han Zhao, Weihao Cui, Ningxin Zheng, Haibin Lin, Xin Liu, Minyi Guo  

**Link**: [PDF](https://arxiv.org/pdf/2509.09560)  

**Abstract**: Embodied AI systems operate in dynamic environments, requiring seamless integration of perception and generation modules to process high-frequency input and output demands. Traditional sequential computation patterns, while effective in ensuring accuracy, face significant limitations in achieving the necessary "thinking" frequency for real-world applications. In this work, we present Auras, an algorithm-system co-designed inference framework to optimize the inference frequency of embodied AI agents. Auras disaggregates the perception and generation and provides controlled pipeline parallelism for them to achieve high and stable throughput. Faced with the data staleness problem that appears when the parallelism is increased, Auras establishes a public context for perception and generation to share, thereby promising the accuracy of embodied agents. Experimental results show that Auras improves throughput by 2.54x on average while achieving 102.7% of the original accuracy, demonstrating its efficacy in overcoming the constraints of sequential computation and providing high throughput. 

---
# Compositional Concept Generalization with Variational Quantum Circuits 

**Authors**: Hala Hawashin, Mina Abbaszadeh, Nicholas Joseph, Beth Pearson, Martha Lewis, Mehrnoosh sadrzadeh  

**Link**: [PDF](https://arxiv.org/pdf/2509.09541)  

**Abstract**: Compositional generalization is a key facet of human cognition, but lacking in current AI tools such as vision-language models. Previous work examined whether a compositional tensor-based sentence semantics can overcome the challenge, but led to negative results. We conjecture that the increased training efficiency of quantum models will improve performance in these tasks. We interpret the representations of compositional tensor-based models in Hilbert spaces and train Variational Quantum Circuits to learn these representations on an image captioning task requiring compositional generalization. We used two image encoding techniques: a multi-hot encoding (MHE) on binary image vectors and an angle/amplitude encoding on image vectors taken from the vision-language model CLIP. We achieve good proof-of-concept results using noisy MHE encodings. Performance on CLIP image vectors was more mixed, but still outperformed classical compositional models. 

---
# SEDM: Scalable Self-Evolving Distributed Memory for Agents 

**Authors**: Haoran Xu, Jiacong Hu, Ke Zhang, Lei Yu, Yuxin Tang, Xinyuan Song, Yiqun Duan, Lynn Ai, Bill Shi  

**Link**: [PDF](https://arxiv.org/pdf/2509.09498)  

**Abstract**: Long-term multi-agent systems inevitably generate vast amounts of trajectories and historical interactions, which makes efficient memory management essential for both performance and scalability. Existing methods typically depend on vector retrieval and hierarchical storage, yet they are prone to noise accumulation, uncontrolled memory expansion, and limited generalization across domains. To address these challenges, we present SEDM, Self-Evolving Distributed Memory, a verifiable and adaptive framework that transforms memory from a passive repository into an active, self-optimizing component. SEDM integrates verifiable write admission based on reproducible replay, a self-scheduling memory controller that dynamically ranks and consolidates entries according to empirical utility, and cross-domain knowledge diffusion that abstracts reusable insights to support transfer across heterogeneous tasks. Evaluations on benchmark datasets demonstrate that SEDM improves reasoning accuracy while reducing token overhead compared with strong memory baselines, and further enables knowledge distilled from fact verification to enhance multi-hop reasoning. The results highlight SEDM as a scalable and sustainable memory mechanism for open-ended multi-agent collaboration. The code will be released in the later stage of this project. 

---
# Inteligencia Artificial jurídica y el desafío de la veracidad: análisis de alucinaciones, optimización de RAG y principios para una integración responsable 

**Authors**: Alex Dantart  

**Link**: [PDF](https://arxiv.org/pdf/2509.09467)  

**Abstract**: This technical report analyzes the challenge of "hallucinations" (false information) in LLMs applied to law. It examines their causes, manifestations, and the effectiveness of the RAG mitigation strategy, highlighting its limitations and proposing holistic optimizations. The paper explores the ethical and regulatory implications, emphasizing human oversight as an irreplaceable role. It concludes that the solution lies not in incrementally improving generative models, but in adopting a "consultative" AI paradigm that prioritizes veracity and traceability, acting as a tool to amplify, not replace, professional judgment.
--
Este informe técnico analiza el desafío de las "alucinaciones" (información falsa) en los LLMs aplicados al derecho. Se examinan sus causas, manifestaciones y la efectividad de la estrategia de mitigación RAG, exponiendo sus limitaciones y proponiendo optimizaciones holísticas. Se exploran las implicaciones éticas y regulatorias, enfatizando la supervisión humana como un rol insustituible. El documento concluye que la solución no reside en mejorar incrementalmente los modelos generativos, sino en adoptar un paradigma de IA "consultiva" que priorice la veracidad y la trazabilidad, actuando como una herramienta para amplificar, y no sustituir, el juicio profesional. 

---
# TORSO: Template-Oriented Reasoning Towards General Tasks 

**Authors**: Minhyuk Kim, Seungyoon Lee, Heuiseok Lim  

**Link**: [PDF](https://arxiv.org/pdf/2509.09448)  

**Abstract**: The approaches that guide Large Language Models (LLMs) to emulate human reasoning during response generation have emerged as an effective method for enabling them to solve complex problems in a step-by-step manner, thereby achieving superior performance. However, most existing approaches using few-shot prompts to generate responses heavily depend on the provided examples, limiting the utilization of the model's inherent reasoning capabilities. Moreover, constructing task-specific few-shot prompts is often costly and may lead to inconsistencies across different tasks. In this work, we introduce Template-Oriented Reasoning (TORSO), which elicits the model to utilize internal reasoning abilities to generate proper responses across various tasks without the need for manually crafted few-shot examples. Our experimental results demonstrate that TORSO achieves strong performance on diverse LLMs benchmarks with reasonable rationales. 

---
# Curriculum-Based Multi-Tier Semantic Exploration via Deep Reinforcement Learning 

**Authors**: Abdel Hakim Drid, Vincenzo Suriani, Daniele Nardi, Abderrezzak Debilou  

**Link**: [PDF](https://arxiv.org/pdf/2509.09356)  

**Abstract**: Navigating and understanding complex and unknown environments autonomously demands more than just basic perception and movement from embodied agents. Truly effective exploration requires agents to possess higher-level cognitive abilities, the ability to reason about their surroundings, and make more informed decisions regarding exploration strategies. However, traditional RL approaches struggle to balance efficient exploration and semantic understanding due to limited cognitive capabilities embedded in the small policies for the agents, leading often to human drivers when dealing with semantic exploration. In this paper, we address this challenge by presenting a novel Deep Reinforcement Learning (DRL) architecture that is specifically designed for resource efficient semantic exploration. A key methodological contribution is the integration of a Vision-Language Model (VLM) common-sense through a layered reward function. The VLM query is modeled as a dedicated action, allowing the agent to strategically query the VLM only when deemed necessary for gaining external guidance, thereby conserving resources. This mechanism is combined with a curriculum learning strategy designed to guide learning at different levels of complexity to ensure robust and stable learning. Our experimental evaluation results convincingly demonstrate that our agent achieves significantly enhanced object discovery rates and develops a learned capability to effectively navigate towards semantically rich regions. Furthermore, it also shows a strategic mastery of when to prompt for external environmental information. By demonstrating a practical and scalable method for embedding common-sense semantic reasoning with autonomous agents, this research provides a novel approach to pursuing a fully intelligent and self-guided exploration in robotics. 

---
# Towards Adaptive ML Benchmarks: Web-Agent-Driven Construction, Domain Expansion, and Metric Optimization 

**Authors**: Hangyi Jia, Yuxi Qian, Hanwen Tong, Xinhui Wu, Lin Chen, Feng Wei  

**Link**: [PDF](https://arxiv.org/pdf/2509.09321)  

**Abstract**: Recent advances in large language models (LLMs) have enabled the emergence of general-purpose agents for automating end-to-end machine learning (ML) workflows, including data analysis, feature engineering, model training, and competition solving. However, existing benchmarks remain limited in task coverage, domain diversity, difficulty modeling, and evaluation rigor, failing to capture the full capabilities of such agents in realistic settings. We present TAM Bench, a diverse, realistic, and structured benchmark for evaluating LLM-based agents on end-to-end ML tasks. TAM Bench features three key innovations: (1) A browser automation and LLM-based task acquisition system that automatically collects and structures ML challenges from platforms such as Kaggle, AIcrowd, and Biendata, spanning multiple task types and data modalities (e.g., tabular, text, image, graph, audio); (2) A leaderboard-driven difficulty modeling mechanism that estimates task complexity using participant counts and score dispersion, enabling scalable and objective task calibration; (3) A multi-dimensional evaluation framework incorporating performance, format compliance, constraint adherence, and task generalization. Based on 150 curated AutoML tasks, we construct three benchmark subsets of different sizes -- Lite, Medium, and Full -- designed for varying evaluation scenarios. The Lite version, with 18 tasks and balanced coverage across modalities and difficulty levels, serves as a practical testbed for daily benchmarking and comparative studies. 

---
# Measuring Implicit Spatial Coordination in Teams: Effects on Collective Intelligence and Performance 

**Authors**: Thuy Ngoc Nguyen, Anita Williams Woolley, Cleotilde Gonzalez  

**Link**: [PDF](https://arxiv.org/pdf/2509.09314)  

**Abstract**: Coordinated teamwork is essential in fast-paced decision-making environments that require dynamic adaptation, often without an opportunity for explicit communication. Although implicit coordination has been extensively considered in the existing literature, the majority of work has focused on co-located, synchronous teamwork (such as sports teams) or, in distributed teams, primarily on coordination of knowledge work. However, many teams (firefighters, military, law enforcement, emergency response) must coordinate their movements in physical space without the benefit of visual cues or extensive explicit communication. This paper investigates how three dimensions of spatial coordination, namely exploration diversity, movement specialization, and adaptive spatial proximity, influence team performance in a collaborative online search and rescue task where explicit communication is restricted and team members rely on movement patterns to infer others' intentions and coordinate actions. Our metrics capture the relational aspects of teamwork by measuring spatial proximity, distribution patterns, and alignment of movements within shared environments. We analyze data from 34 four-person teams (136 participants) assigned to specialized roles in a search and rescue task. Results show that spatial specialization positively predicts performance, while adaptive spatial proximity exhibits a marginal inverted U-shaped relationship, suggesting moderate levels of adaptation are optimal. Furthermore, the temporal dynamics of these metrics differentiate high- from low-performing teams over time. These findings provide insights into implicit spatial coordination in role-based teamwork and highlight the importance of balanced adaptive strategies, with implications for training and AI-assisted team support systems. 

---
# Explaining Tournament Solutions with Minimal Supports 

**Authors**: Clément Contet, Umberto Grandi, Jérôme Mengin  

**Link**: [PDF](https://arxiv.org/pdf/2509.09312)  

**Abstract**: Tournaments are widely used models to represent pairwise dominance between candidates, alternatives, or teams. We study the problem of providing certified explanations for why a candidate appears among the winners under various tournament rules. To this end, we identify minimal supports, minimal sub-tournaments in which the candidate is guaranteed to win regardless of how the rest of the tournament is completed (that is, the candidate is a necessary winner of the sub-tournament). This notion corresponds to an abductive explanation for the question,"Why does the winner win the tournament", a central concept in formal explainable AI. We focus on common tournament solutions: the top cycle, the uncovered set, the Copeland rule, the Borda rule, the maximin rule, and the weighted uncovered set. For each rule we determine the size of the smallest minimal supports, and we present polynomial-time algorithms to compute them for all but the weighted uncovered set, for which the problem is NP-complete. Finally, we show how minimal supports can serve to produce compact, certified, and intuitive explanations. 

---
# LightAgent: Production-level Open-source Agentic AI Framework 

**Authors**: Weige Cai, Tong Zhu, Jinyi Niu, Ruiqi Hu, Lingyao Li, Tenglong Wang, Xiaowu Dai, Weining Shen, Liwen Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2509.09292)  

**Abstract**: With the rapid advancement of large language models (LLMs), Multi-agent Systems (MAS) have achieved significant progress in various application scenarios. However, substantial challenges remain in designing versatile, robust, and efficient platforms for agent deployment. To address these limitations, we propose \textbf{LightAgent}, a lightweight yet powerful agentic framework, effectively resolving the trade-off between flexibility and simplicity found in existing frameworks. LightAgent integrates core functionalities such as Memory (mem0), Tools, and Tree of Thought (ToT), while maintaining an extremely lightweight structure. As a fully open-source solution, it seamlessly integrates with mainstream chat platforms, enabling developers to easily build self-learning agents. We have released LightAgent at \href{this https URL}{this https URL} 

---
# Tree-OPO: Off-policy Monte Carlo Tree-Guided Advantage Optimization for Multistep Reasoning 

**Authors**: Bingning Huang, Tu Nguyen, Matthieu Zimmer  

**Link**: [PDF](https://arxiv.org/pdf/2509.09284)  

**Abstract**: Recent advances in reasoning with large language models (LLMs) have shown the effectiveness of Monte Carlo Tree Search (MCTS) for generating high-quality intermediate trajectories, particularly in math and symbolic domains. Inspired by this, we explore how MCTS-derived trajectories, traditionally used for training value or reward models, can be repurposed to improve policy optimization in preference-based reinforcement learning (RL). Specifically, we focus on Group Relative Policy Optimization (GRPO), a recent algorithm that enables preference-consistent policy learning without value networks. We propose a staged GRPO training paradigm where completions are derived from partially revealed MCTS rollouts, introducing a novel tree-structured setting for advantage estimation. This leads to a rich class of prefix-conditioned reward signals, which we analyze theoretically and empirically. Our initial results indicate that while structured advantage estimation can stabilize updates and better reflect compositional reasoning quality, challenges such as advantage saturation and reward signal collapse remain. We propose heuristic and statistical solutions to mitigate these issues and discuss open challenges for learning under staged or tree-like reward structures. 

---
# Fusing Knowledge and Language: A Comparative Study of Knowledge Graph-Based Question Answering with LLMs 

**Authors**: Vaibhav Chaudhary, Neha Soni, Narotam Singh, Amita Kapoor  

**Link**: [PDF](https://arxiv.org/pdf/2509.09272)  

**Abstract**: Knowledge graphs, a powerful tool for structuring information through relational triplets, have recently become the new front-runner in enhancing question-answering systems. While traditional Retrieval Augmented Generation (RAG) approaches are proficient in fact-based and local context-based extraction from concise texts, they encounter limitations when addressing the thematic and holistic understanding of complex, extensive texts, requiring a deeper analysis of both text and context. This paper presents a comprehensive technical comparative study of three different methodologies for constructing knowledge graph triplets and integrating them with Large Language Models (LLMs) for question answering: spaCy, Stanford CoreNLP-OpenIE, and GraphRAG, all leveraging open source technologies. We evaluate the effectiveness, feasibility, and adaptability of these methods by analyzing their capabilities, state of development, and their impact on the performance of LLM-based question answering. Experimental results indicate that while OpenIE provides the most comprehensive coverage of triplets, GraphRAG demonstrates superior reasoning abilities among the three. We conclude with a discussion on the strengths and limitations of each method and provide insights into future directions for improving knowledge graph-based question answering. 

---
# Jupiter: Enhancing LLM Data Analysis Capabilities via Notebook and Inference-Time Value-Guided Search 

**Authors**: Shuocheng Li, Yihao Liu, Silin Du, Wenxuan Zeng, Zhe Xu, Mengyu Zhou, Yeye He, Haoyu Dong, Shi Han, Dongmei Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2509.09245)  

**Abstract**: Large language models (LLMs) have shown great promise in automating data science workflows, but existing models still struggle with multi-step reasoning and tool use, which limits their effectiveness on complex data analysis tasks. To address this, we propose a scalable pipeline that extracts high-quality, tool-based data analysis tasks and their executable multi-step solutions from real-world Jupyter notebooks and associated data files. Using this pipeline, we introduce NbQA, a large-scale dataset of standardized task-solution pairs that reflect authentic tool-use patterns in practical data science scenarios. To further enhance multi-step reasoning, we present Jupiter, a framework that formulates data analysis as a search problem and applies Monte Carlo Tree Search (MCTS) to generate diverse solution trajectories for value model learning. During inference, Jupiter combines the value model and node visit counts to efficiently collect executable multi-step plans with minimal search steps. Experimental results show that Qwen2.5-7B and 14B-Instruct models on NbQA solve 77.82% and 86.38% of tasks on InfiAgent-DABench, respectively-matching or surpassing GPT-4o and advanced agent frameworks. Further evaluations demonstrate improved generalization and stronger tool-use reasoning across diverse multi-step reasoning tasks. 

---
# Enabling Regulatory Multi-Agent Collaboration: Architecture, Challenges, and Solutions 

**Authors**: Qinnan Hu, Yuntao Wang, Yuan Gao, Zhou Su, Linkang Du  

**Link**: [PDF](https://arxiv.org/pdf/2509.09215)  

**Abstract**: Large language models (LLMs)-empowered autonomous agents are transforming both digital and physical environments by enabling adaptive, multi-agent collaboration. While these agents offer significant opportunities across domains such as finance, healthcare, and smart manufacturing, their unpredictable behaviors and heterogeneous capabilities pose substantial governance and accountability challenges. In this paper, we propose a blockchain-enabled layered architecture for regulatory agent collaboration, comprising an agent layer, a blockchain data layer, and a regulatory application layer. Within this framework, we design three key modules: (i) an agent behavior tracing and arbitration module for automated accountability, (ii) a dynamic reputation evaluation module for trust assessment in collaborative scenarios, and (iii) a malicious behavior forecasting module for early detection of adversarial activities. Our approach establishes a systematic foundation for trustworthy, resilient, and scalable regulatory mechanisms in large-scale agent ecosystems. Finally, we discuss the future research directions for blockchain-enabled regulatory frameworks in multi-agent systems. 

---
# ProgD: Progressive Multi-scale Decoding with Dynamic Graphs for Joint Multi-agent Motion Forecasting 

**Authors**: Xing Gao, Zherui Huang, Weiyao Lin, Xiao Sun  

**Link**: [PDF](https://arxiv.org/pdf/2509.09210)  

**Abstract**: Accurate motion prediction of surrounding agents is crucial for the safe planning of autonomous vehicles. Recent advancements have extended prediction techniques from individual agents to joint predictions of multiple interacting agents, with various strategies to address complex interactions within future motions of agents. However, these methods overlook the evolving nature of these interactions. To address this limitation, we propose a novel progressive multi-scale decoding strategy, termed ProgD, with the help of dynamic heterogeneous graph-based scenario modeling. In particular, to explicitly and comprehensively capture the evolving social interactions in future scenarios, given their inherent uncertainty, we design a progressive modeling of scenarios with dynamic heterogeneous graphs. With the unfolding of such dynamic heterogeneous graphs, a factorized architecture is designed to process the spatio-temporal dependencies within future scenarios and progressively eliminate uncertainty in future motions of multiple agents. Furthermore, a multi-scale decoding procedure is incorporated to improve on the future scenario modeling and consistent prediction of agents' future motion. The proposed ProgD achieves state-of-the-art performance on the INTERACTION multi-agent prediction benchmark, ranking $1^{st}$, and the Argoverse 2 multi-world forecasting benchmark. 

---
# Mind Meets Space: Rethinking Agentic Spatial Intelligence from a Neuroscience-inspired Perspective 

**Authors**: Bui Duc Manh, Soumyaratna Debnath, Zetong Zhang, Shriram Damodaran, Arvind Kumar, Yueyi Zhang, Lu Mi, Erik Cambria, Lin Wang  

**Link**: [PDF](https://arxiv.org/pdf/2509.09154)  

**Abstract**: Recent advances in agentic AI have led to systems capable of autonomous task execution and language-based reasoning, yet their spatial reasoning abilities remain limited and underexplored, largely constrained to symbolic and sequential processing. In contrast, human spatial intelligence, rooted in integrated multisensory perception, spatial memory, and cognitive maps, enables flexible, context-aware decision-making in unstructured environments. Therefore, bridging this gap is critical for advancing Agentic Spatial Intelligence toward better interaction with the physical 3D world. To this end, we first start from scrutinizing the spatial neural models as studied in computational neuroscience, and accordingly introduce a novel computational framework grounded in neuroscience principles. This framework maps core biological functions to six essential computation modules: bio-inspired multimodal sensing, multi-sensory integration, egocentric-allocentric conversion, an artificial cognitive map, spatial memory, and spatial reasoning. Together, these modules form a perspective landscape for agentic spatial reasoning capability across both virtual and physical environments. On top, we conduct a framework-guided analysis of recent methods, evaluating their relevance to each module and identifying critical gaps that hinder the development of more neuroscience-grounded spatial reasoning modules. We further examine emerging benchmarks and datasets and explore potential application domains ranging from virtual to embodied systems, such as robotics. Finally, we outline potential research directions, emphasizing the promising roadmap that can generalize spatial reasoning across dynamic or unstructured environments. We hope this work will benefit the research community with a neuroscience-grounded perspective and a structured pathway. Our project page can be found at Github. 

---
# Anti-Money Laundering Machine Learning Pipelines; A Technical Analysis on Identifying High-risk Bank Clients with Supervised Learning 

**Authors**: Khashayar Namdar, Pin-Chien Wang, Tushar Raju, Steven Zheng, Fiona Li, Safwat Tahmin Khan  

**Link**: [PDF](https://arxiv.org/pdf/2509.09127)  

**Abstract**: Anti-money laundering (AML) actions and measurements are among the priorities of financial institutions, for which machine learning (ML) has shown to have a high potential. In this paper, we propose a comprehensive and systematic approach for developing ML pipelines to identify high-risk bank clients in a dataset curated for Task 1 of the University of Toronto 2023-2024 Institute for Management and Innovation (IMI) Big Data and Artificial Intelligence Competition. The dataset included 195,789 customer IDs, and we employed a 16-step design and statistical analysis to ensure the final pipeline was robust. We also framed the data in a SQLite database, developed SQL-based feature engineering algorithms, connected our pre-trained model to the database, and made it inference-ready, and provided explainable artificial intelligence (XAI) modules to derive feature importance. Our pipeline achieved a mean area under the receiver operating characteristic curve (AUROC) of 0.961 with a standard deviation (SD) of 0.005. The proposed pipeline achieved second place in the competition. 

---
# Understanding Economic Tradeoffs Between Human and AI Agents in Bargaining Games 

**Authors**: Crystal Qian, Kehang Zhu, John Horton, Benjamin S. Manning, Vivian Tsai, James Wexler, Nithum Thain  

**Link**: [PDF](https://arxiv.org/pdf/2509.09071)  

**Abstract**: Coordination tasks traditionally performed by humans are increasingly being delegated to autonomous agents. As this pattern progresses, it becomes critical to evaluate not only these agents' performance but also the processes through which they negotiate in dynamic, multi-agent environments. Furthermore, different agents exhibit distinct advantages: traditional statistical agents, such as Bayesian models, may excel under well-specified conditions, whereas large language models (LLMs) can generalize across contexts. In this work, we compare humans (N = 216), LLMs (GPT-4o, Gemini 1.5 Pro), and Bayesian agents in a dynamic negotiation setting that enables direct, identical-condition comparisons across populations, capturing both outcomes and behavioral dynamics. Bayesian agents extract the highest surplus through aggressive optimization, at the cost of frequent trade rejections. Humans and LLMs can achieve similar overall surplus, but through distinct behaviors: LLMs favor conservative, concessionary trades with few rejections, while humans employ more strategic, risk-taking, and fairness-oriented behaviors. Thus, we find that performance parity -- a common benchmark in agent evaluation -- can conceal fundamental differences in process and alignment, which are critical for practical deployment in real-world coordination tasks. 

---
# Instructional Prompt Optimization for Few-Shot LLM-Based Recommendations on Cold-Start Users 

**Authors**: Haowei Yang, Yushang Zhao, Sitao Min, Bo Su, Chao Yao, Wei Xu  

**Link**: [PDF](https://arxiv.org/pdf/2509.09066)  

**Abstract**: The cold-start user issue further compromises the effectiveness of recommender systems in limiting access to the historical behavioral information. It is an effective pipeline to optimize instructional prompts on a few-shot large language model (LLM) used in recommender tasks. We introduce a context-conditioned prompt formulation method P(u,\ Ds)\ \rightarrow\ R\widehat, where u is a cold-start user profile, Ds is a curated support set, and R\widehat is the predicted ranked list of items. Based on systematic experimentation with transformer-based autoregressive LLMs (BioGPT, LLaMA-2, GPT-4), we provide empirical evidence that optimal exemplar injection and instruction structuring can significantly improve the precision@k and NDCG scores of such models in low-data settings. The pipeline uses token-level alignments and embedding space regularization with a greater semantic fidelity. Our findings not only show that timely composition is not merely syntactic but also functional as it is in direct control of attention scales and decoder conduct through inference. This paper shows that prompt-based adaptation may be considered one of the ways to address cold-start recommendation issues in LLM-based pipelines. 

---
# Uncertainty Awareness and Trust in Explainable AI- On Trust Calibration using Local and Global Explanations 

**Authors**: Carina Newen, Daniel Bodemer, Sonja Glantz, Emmanuel Müller, Magdalena Wischnewski, Lenka Schnaubert  

**Link**: [PDF](https://arxiv.org/pdf/2509.08989)  

**Abstract**: Explainable AI has become a common term in the literature, scrutinized by computer scientists and statisticians and highlighted by psychological or philosophical researchers. One major effort many researchers tackle is constructing general guidelines for XAI schemes, which we derived from our study. While some areas of XAI are well studied, we focus on uncertainty explanations and consider global explanations, which are often left out. We chose an algorithm that covers various concepts simultaneously, such as uncertainty, robustness, and global XAI, and tested its ability to calibrate trust. We then checked whether an algorithm that aims to provide more of an intuitive visual understanding, despite being complicated to understand, can provide higher user satisfaction and human interpretability. 

---
# ForTIFAI: Fending Off Recursive Training Induced Failure for AI Models 

**Authors**: Soheil Zibakhsh Shabgahi, Pedram Aghazadeh, Azalia Mirhosseini, Farinaz Koushanfar  

**Link**: [PDF](https://arxiv.org/pdf/2509.08972)  

**Abstract**: The increasing reliance on generative AI models has accelerated the generation rate of synthetic data, with some projections suggesting that most available new data for training could be machine-generated by 2030. This shift to a mainly synthetic content presents a critical challenge: repeated training in synthetic data leads to a phenomenon known as model collapse, where model performance degrades over generations of training, eventually rendering the models ineffective. Although prior studies have explored the causes and detection of model collapse, existing mitigation strategies remain limited.
In this paper, we identify model overconfidence in their self-generated data as a key driver of collapse. Building on this observation, we propose a confidence-aware loss function that downweights high-confidence predictions during training. We introduce a novel loss function we call Truncated Cross Entropy (TCE). We demonstrate that TCE significantly delays model collapse in recursive training.
We provide a model-agnostic framework that links the loss function design to model collapse mitigation and validate our approach both theoretically and empirically, showing that it can extend the model's fidelity interval before collapse by more than 2.3x. Finally, we show that our method generalizes across modalities. These findings suggest that the design of loss functions provides a simple yet powerful tool for preserving the quality of generative models in the era of increasing synthetic data. 

---
# Global Constraint LLM Agents for Text-to-Model Translation 

**Authors**: Junyang Cai, Serdar Kadioglu, Bistra Dilkina  

**Link**: [PDF](https://arxiv.org/pdf/2509.08970)  

**Abstract**: Natural language descriptions of optimization or satisfaction problems are challenging to translate into correct MiniZinc models, as this process demands both logical reasoning and constraint programming expertise. We introduce a framework that addresses this challenge with an agentic approach: multiple specialized large language model (LLM) agents decompose the modeling task by global constraint type. Each agent is dedicated to detecting and generating code for a specific class of global constraint, while a final assembler agent integrates these constraint snippets into a complete MiniZinc model. By dividing the problem into smaller, well-defined sub-tasks, each LLM handles a simpler reasoning challenge, potentially reducing overall complexity. We conduct initial experiments with several LLMs and show better performance against baselines such as one-shot prompting and chain-of-thought prompting. Finally, we outline a comprehensive roadmap for future work, highlighting potential enhancements and directions for improvement. 

---
# Automated Unity Game Template Generation from GDDs via NLP and Multi-Modal LLMs 

**Authors**: Amna Hassan  

**Link**: [PDF](https://arxiv.org/pdf/2509.08847)  

**Abstract**: This paper presents a novel framework for automated game template generation by transforming Game Design Documents (GDDs) into functional Unity game prototypes using Natural Language Processing (NLP) and multi-modal Large Language Models (LLMs). We introduce an end-to-end system that parses GDDs, extracts structured game specifications, and synthesizes Unity-compatible C# code that implements the core mechanics, systems, and architecture defined in the design documentation. Our approach combines a fine-tuned LLaMA-3 model specialized for Unity code generation with a custom Unity integration package that streamlines the implementation process. Evaluation results demonstrate significant improvements over baseline models, with our fine-tuned model achieving superior performance (4.8/5.0 average score) compared to state-of-the-art LLMs across compilation success, GDD adherence, best practices adoption, and code modularity metrics. The generated templates demonstrate high adherence to GDD specifications across multiple game genres. Our system effectively addresses critical gaps in AI-assisted game development, positioning LLMs as valuable tools in streamlining the transition from game design to implementation. 

---
# An Interval Type-2 Version of Bayes Theorem Derived from Interval Probability Range Estimates Provided by Subject Matter Experts 

**Authors**: John T. Rickard, William A. Dembski, James Rickards  

**Link**: [PDF](https://arxiv.org/pdf/2509.08834)  

**Abstract**: Bayesian inference is widely used in many different fields to test hypotheses against observations. In most such applications, an assumption is made of precise input values to produce a precise output value. However, this is unrealistic for real-world applications. Often the best available information from subject matter experts (SMEs) in a given field is interval range estimates of the input probabilities involved in Bayes Theorem. This paper provides two key contributions to extend Bayes Theorem to an interval type-2 (IT2) version. First, we develop an IT2 version of Bayes Theorem that uses a novel and conservative method to avoid potential inconsistencies in the input IT2 MFs that otherwise might produce invalid output results. We then describe a novel and flexible algorithm for encoding SME-provided intervals into IT2 fuzzy membership functions (MFs), which we can use to specify the input probabilities in Bayes Theorem. Our algorithm generalizes and extends previous work on this problem that primarily addressed the encoding of intervals into word MFs for Computing with Words applications. 

---
# ButterflyQuant: Ultra-low-bit LLM Quantization through Learnable Orthogonal Butterfly Transforms 

**Authors**: Bingxin Xu, Zhen Dong, Oussama Elachqar, Yuzhang Shang  

**Link**: [PDF](https://arxiv.org/pdf/2509.09679)  

**Abstract**: Large language models require massive memory footprints, severely limiting deployment on consumer hardware. Quantization reduces memory through lower numerical precision, but extreme 2-bit quantization suffers from catastrophic performance loss due to outliers in activations. Rotation-based methods such as QuIP and QuaRot apply orthogonal transforms to eliminate outliers before quantization, using computational invariance: $\mathbf{y} = \mathbf{Wx} = (\mathbf{WQ}^T)(\mathbf{Qx})$ for orthogonal $\mathbf{Q}$. However, these methods use fixed transforms--Hadamard matrices achieving optimal worst-case coherence $\mu = 1/\sqrt{n}$--that cannot adapt to specific weight distributions. We identify that different transformer layers exhibit distinct outlier patterns, motivating layer-adaptive rotations rather than one-size-fits-all approaches. We propose ButterflyQuant, which replaces Hadamard rotations with learnable butterfly transforms parameterized by continuous Givens rotation angles. Unlike Hadamard's discrete $\{+1, -1\}$ entries that are non-differentiable and prohibit gradient-based learning, butterfly transforms' continuous parameterization enables smooth optimization while guaranteeing orthogonality by construction. This orthogonal constraint ensures theoretical guarantees in outlier suppression while achieving $O(n \log n)$ computational complexity with only $\frac{n \log n}{2}$ learnable parameters. We further introduce a uniformity regularization on post-transformation activations to promote smoother distributions amenable to quantization. Learning requires only 128 calibration samples and converges in minutes on a single GPU--a negligible one-time cost. On LLaMA-2-7B with 2-bit quantization, ButterflyQuant achieves 15.4 perplexity versus 22.1 for QuaRot. 

---
# CDE: Curiosity-Driven Exploration for Efficient Reinforcement Learning in Large Language Models 

**Authors**: Runpeng Dai, Linfeng Song, Haolin Liu, Zhenwen Liang, Dian Yu, Haitao Mi, Zhaopeng Tu, Rui Liu, Tong Zheng, Hongtu Zhu, Dong Yu  

**Link**: [PDF](https://arxiv.org/pdf/2509.09675)  

**Abstract**: Reinforcement Learning with Verifiable Rewards (RLVR) is a powerful paradigm for enhancing the reasoning ability of Large Language Models (LLMs). Yet current RLVR methods often explore poorly, leading to premature convergence and entropy collapse. To address this challenge, we introduce Curiosity-Driven Exploration (CDE), a framework that leverages the model's own intrinsic sense of curiosity to guide exploration. We formalize curiosity with signals from both the actor and the critic: for the actor, we use perplexity over its generated response, and for the critic, we use the variance of value estimates from a multi-head architecture. Both signals serve as an exploration bonus within the RLVR framework to guide the model. Our theoretical analysis shows that the actor-wise bonus inherently penalizes overconfident errors and promotes diversity among correct responses; moreover, we connect the critic-wise bonus to the well-established count-based exploration bonus in RL. Empirically, our method achieves an approximate +3 point improvement over standard RLVR using GRPO/PPO on AIME benchmarks. Further analysis identifies a calibration collapse mechanism within RLVR, shedding light on common LLM failure modes. 

---
# SimpleVLA-RL: Scaling VLA Training via Reinforcement Learning 

**Authors**: Haozhan Li, Yuxin Zuo, Jiale Yu, Yuhao Zhang, Zhaohui Yang, Kaiyan Zhang, Xuekai Zhu, Yuchen Zhang, Tianxing Chen, Ganqu Cui, Dehui Wang, Dingxiang Luo, Yuchen Fan, Youbang Sun, Jia Zeng, Jiangmiao Pang, Shanghang Zhang, Yu Wang, Yao Mu, Bowen Zhou, Ning Ding  

**Link**: [PDF](https://arxiv.org/pdf/2509.09674)  

**Abstract**: Vision-Language-Action (VLA) models have recently emerged as a powerful paradigm for robotic manipulation. Despite substantial progress enabled by large-scale pretraining and supervised fine-tuning (SFT), these models face two fundamental challenges: (i) the scarcity and high cost of large-scale human-operated robotic trajectories required for SFT scaling, and (ii) limited generalization to tasks involving distribution shift. Recent breakthroughs in Large Reasoning Models (LRMs) demonstrate that reinforcement learning (RL) can dramatically enhance step-by-step reasoning capabilities, raising a natural question: Can RL similarly improve the long-horizon step-by-step action planning of VLA? In this work, we introduce SimpleVLA-RL, an efficient RL framework tailored for VLA models. Building upon veRL, we introduce VLA-specific trajectory sampling, scalable parallelization, multi-environment rendering, and optimized loss computation. When applied to OpenVLA-OFT, SimpleVLA-RL achieves SoTA performance on LIBERO and even outperforms $\pi_0$ on RoboTwin 1.0\&2.0 with the exploration-enhancing strategies we introduce. SimpleVLA-RL not only reduces dependence on large-scale data and enables robust generalization, but also remarkably surpasses SFT in real-world tasks. Moreover, we identify a novel phenomenon ``pushcut'' during RL training, wherein the policy discovers previously unseen patterns beyond those seen in the previous training process. Github: this https URL 

---
# Feasibility-Guided Fair Adaptive Offline Reinforcement Learning for Medicaid Care Management 

**Authors**: Sanjay Basu, Sadiq Y. Patel, Parth Sheth, Bhairavi Muralidharan, Namrata Elamaran, Aakriti Kinra, Rajaie Batniji  

**Link**: [PDF](https://arxiv.org/pdf/2509.09655)  

**Abstract**: We introduce Feasibility-Guided Fair Adaptive Reinforcement Learning (FG-FARL), an offline RL procedure that calibrates per-group safety thresholds to reduce harm while equalizing a chosen fairness target (coverage or harm) across protected subgroups. Using de-identified longitudinal trajectories from a Medicaid population health management program, we evaluate FG-FARL against behavior cloning (BC) and HACO (Hybrid Adaptive Conformal Offline RL; a global conformal safety baseline). We report off-policy value estimates with bootstrap 95% confidence intervals and subgroup disparity analyses with p-values. FG-FARL achieves comparable value to baselines while improving fairness metrics, demonstrating a practical path to safer and more equitable decision support. 

---
# Retrieval-Augmented Generation for Reliable Interpretation of Radio Regulations 

**Authors**: Zakaria El Kassimi, Fares Fourati, Mohamed-Slim Alouini  

**Link**: [PDF](https://arxiv.org/pdf/2509.09651)  

**Abstract**: We study question answering in the domain of radio regulations, a legally sensitive and high-stakes area. We propose a telecom-specific Retrieval-Augmented Generation (RAG) pipeline and introduce, to our knowledge, the first multiple-choice evaluation set for this domain, constructed from authoritative sources using automated filtering and human validation. To assess retrieval quality, we define a domain-specific retrieval metric, under which our retriever achieves approximately 97% accuracy. Beyond retrieval, our approach consistently improves generation accuracy across all tested models. In particular, while naively inserting documents without structured retrieval yields only marginal gains for GPT-4o (less than 1%), applying our pipeline results in nearly a 12% relative improvement. These findings demonstrate that carefully targeted grounding provides a simple yet strong baseline and an effective domain-specific solution for regulatory question answering. All code and evaluation scripts, along with our derived question-answer dataset, are available at this https URL. 

---
# Explaining Concept Drift through the Evolution of Group Counterfactuals 

**Authors**: Ignacy Stępka, Jerzy Stefanowski  

**Link**: [PDF](https://arxiv.org/pdf/2509.09616)  

**Abstract**: Machine learning models in dynamic environments often suffer from concept drift, where changes in the data distribution degrade performance. While detecting this drift is a well-studied topic, explaining how and why the model's decision-making logic changes still remains a significant challenge. In this paper, we introduce a novel methodology to explain concept drift by analyzing the temporal evolution of group-based counterfactual explanations (GCEs). Our approach tracks shifts in the GCEs' cluster centroids and their associated counterfactual action vectors before and after a drift. These evolving GCEs act as an interpretable proxy, revealing structural changes in the model's decision boundary and its underlying rationale. We operationalize this analysis within a three-layer framework that synergistically combines insights from the data layer (distributional shifts), the model layer (prediction disagreement), and our proposed explanation layer. We show that such holistic view allows for a more comprehensive diagnosis of drift, making it possible to distinguish between different root causes, such as a spatial data shift versus a re-labeling of concepts. 

---
# LoCoBench: A Benchmark for Long-Context Large Language Models in Complex Software Engineering 

**Authors**: Jielin Qiu, Zuxin Liu, Zhiwei Liu, Rithesh Murthy, Jianguo Zhang, Haolin Chen, Shiyu Wang, Ming Zhu, Liangwei Yang, Juntao Tan, Zhepeng Cen, Cheng Qian, Shelby Heinecke, Weiran Yao, Silvio Savarese, Caiming Xiong, Huan Wang  

**Link**: [PDF](https://arxiv.org/pdf/2509.09614)  

**Abstract**: The emergence of long-context language models with context windows extending to millions of tokens has created new opportunities for sophisticated code understanding and software development evaluation. We propose LoCoBench, a comprehensive benchmark specifically designed to evaluate long-context LLMs in realistic, complex software development scenarios. Unlike existing code evaluation benchmarks that focus on single-function completion or short-context tasks, LoCoBench addresses the critical evaluation gap for long-context capabilities that require understanding entire codebases, reasoning across multiple files, and maintaining architectural consistency across large-scale software systems. Our benchmark provides 8,000 evaluation scenarios systematically generated across 10 programming languages, with context lengths spanning 10K to 1M tokens, a 100x variation that enables precise assessment of long-context performance degradation in realistic software development settings. LoCoBench introduces 8 task categories that capture essential long-context capabilities: architectural understanding, cross-file refactoring, multi-session development, bug investigation, feature implementation, code comprehension, integration testing, and security analysis. Through a 5-phase pipeline, we create diverse, high-quality scenarios that challenge LLMs to reason about complex codebases at unprecedented scale. We introduce a comprehensive evaluation framework with 17 metrics across 4 dimensions, including 8 new evaluation metrics, combined in a LoCoBench Score (LCBS). Our evaluation of state-of-the-art long-context models reveals substantial performance gaps, demonstrating that long-context understanding in complex software development represents a significant unsolved challenge that demands more attention. LoCoBench is released at: this https URL. 

---
# Mechanistic Learning with Guided Diffusion Models to Predict Spatio-Temporal Brain Tumor Growth 

**Authors**: Daria Laslo, Efthymios Georgiou, Marius George Linguraru, Andreas Rauschecker, Sabine Muller, Catherine R. Jutzeler, Sarah Bruningk  

**Link**: [PDF](https://arxiv.org/pdf/2509.09610)  

**Abstract**: Predicting the spatio-temporal progression of brain tumors is essential for guiding clinical decisions in neuro-oncology. We propose a hybrid mechanistic learning framework that combines a mathematical tumor growth model with a guided denoising diffusion implicit model (DDIM) to synthesize anatomically feasible future MRIs from preceding scans. The mechanistic model, formulated as a system of ordinary differential equations, captures temporal tumor dynamics including radiotherapy effects and estimates future tumor burden. These estimates condition a gradient-guided DDIM, enabling image synthesis that aligns with both predicted growth and patient anatomy. We train our model on the BraTS adult and pediatric glioma datasets and evaluate on 60 axial slices of in-house longitudinal pediatric diffuse midline glioma (DMG) cases. Our framework generates realistic follow-up scans based on spatial similarity metrics. It also introduces tumor growth probability maps, which capture both clinically relevant extent and directionality of tumor growth as shown by 95th percentile Hausdorff Distance. The method enables biologically informed image generation in data-limited scenarios, offering generative-space-time predictions that account for mechanistic priors. 

---
# Graph Alignment via Dual-Pass Spectral Encoding and Latent Space Communication 

**Authors**: Maysam Behmanesh, Erkan Turan, Maks Ovsjanikov  

**Link**: [PDF](https://arxiv.org/pdf/2509.09597)  

**Abstract**: Graph alignment-the problem of identifying corresponding nodes across multiple graphs-is fundamental to numerous applications. Most existing unsupervised methods embed node features into latent representations to enable cross-graph comparison without ground-truth correspondences. However, these methods suffer from two critical limitations: the degradation of node distinctiveness due to oversmoothing in GNN-based embeddings, and the misalignment of latent spaces across graphs caused by structural noise, feature heterogeneity, and training instability, ultimately leading to unreliable node correspondences. We propose a novel graph alignment framework that simultaneously enhances node distinctiveness and enforces geometric consistency across latent spaces. Our approach introduces a dual-pass encoder that combines low-pass and high-pass spectral filters to generate embeddings that are both structure-aware and highly discriminative. To address latent space misalignment, we incorporate a geometry-aware functional map module that learns bijective and isometric transformations between graph embeddings, ensuring consistent geometric relationships across different representations. Extensive experiments on graph benchmarks demonstrate that our method consistently outperforms existing unsupervised alignment baselines, exhibiting superior robustness to structural inconsistencies and challenging alignment scenarios. Additionally, comprehensive evaluation on vision-language benchmarks using diverse pretrained models shows that our framework effectively generalizes beyond graph domains, enabling unsupervised alignment of vision and language representations. 

---
# ObjectReact: Learning Object-Relative Control for Visual Navigation 

**Authors**: Sourav Garg, Dustin Craggs, Vineeth Bhat, Lachlan Mares, Stefan Podgorski, Madhava Krishna, Feras Dayoub, Ian Reid  

**Link**: [PDF](https://arxiv.org/pdf/2509.09594)  

**Abstract**: Visual navigation using only a single camera and a topological map has recently become an appealing alternative to methods that require additional sensors and 3D maps. This is typically achieved through an "image-relative" approach to estimating control from a given pair of current observation and subgoal image. However, image-level representations of the world have limitations because images are strictly tied to the agent's pose and embodiment. In contrast, objects, being a property of the map, offer an embodiment- and trajectory-invariant world representation. In this work, we present a new paradigm of learning "object-relative" control that exhibits several desirable characteristics: a) new routes can be traversed without strictly requiring to imitate prior experience, b) the control prediction problem can be decoupled from solving the image matching problem, and c) high invariance can be achieved in cross-embodiment deployment for variations across both training-testing and mapping-execution settings. We propose a topometric map representation in the form of a "relative" 3D scene graph, which is used to obtain more informative object-level global path planning costs. We train a local controller, dubbed "ObjectReact", conditioned directly on a high-level "WayObject Costmap" representation that eliminates the need for an explicit RGB input. We demonstrate the advantages of learning object-relative control over its image-relative counterpart across sensor height variations and multiple navigation tasks that challenge the underlying spatial understanding capability, e.g., navigating a map trajectory in the reverse direction. We further show that our sim-only policy is able to generalize well to real-world indoor environments. Code and supplementary material are accessible via project page: this https URL 

---
# Fluent but Unfeeling: The Emotional Blind Spots of Language Models 

**Authors**: Bangzhao Shu, Isha Joshi, Melissa Karnaze, Anh C. Pham, Ishita Kakkar, Sindhu Kothe, Arpine Hovasapian, Mai ElSherief  

**Link**: [PDF](https://arxiv.org/pdf/2509.09593)  

**Abstract**: The versatility of Large Language Models (LLMs) in natural language understanding has made them increasingly popular in mental health research. While many studies explore LLMs' capabilities in emotion recognition, a critical gap remains in evaluating whether LLMs align with human emotions at a fine-grained level. Existing research typically focuses on classifying emotions into predefined, limited categories, overlooking more nuanced expressions. To address this gap, we introduce EXPRESS, a benchmark dataset curated from Reddit communities featuring 251 fine-grained, self-disclosed emotion labels. Our comprehensive evaluation framework examines predicted emotion terms and decomposes them into eight basic emotions using established emotion theories, enabling a fine-grained comparison. Systematic testing of prevalent LLMs under various prompt settings reveals that accurately predicting emotions that align with human self-disclosed emotions remains challenging. Qualitative analysis further shows that while certain LLMs generate emotion terms consistent with established emotion theories and definitions, they sometimes fail to capture contextual cues as effectively as human self-disclosures. These findings highlight the limitations of LLMs in fine-grained emotion alignment and offer insights for future research aimed at enhancing their contextual understanding. 

---
# Invisible Attributes, Visible Biases: Exploring Demographic Shortcuts in MRI-based Alzheimer's Disease Classification 

**Authors**: Akshit Achara, Esther Puyol Anton, Alexander Hammers, Andrew P. King  

**Link**: [PDF](https://arxiv.org/pdf/2509.09558)  

**Abstract**: Magnetic resonance imaging (MRI) is the gold standard for brain imaging. Deep learning (DL) algorithms have been proposed to aid in the diagnosis of diseases such as Alzheimer's disease (AD) from MRI scans. However, DL algorithms can suffer from shortcut learning, in which spurious features, not directly related to the output label, are used for prediction. When these features are related to protected attributes, they can lead to performance bias against underrepresented protected groups, such as those defined by race and sex. In this work, we explore the potential for shortcut learning and demographic bias in DL based AD diagnosis from MRI. We first investigate if DL algorithms can identify race or sex from 3D brain MRI scans to establish the presence or otherwise of race and sex based distributional shifts. Next, we investigate whether training set imbalance by race or sex can cause a drop in model performance, indicating shortcut learning and bias. Finally, we conduct a quantitative and qualitative analysis of feature attributions in different brain regions for both the protected attribute and AD classification tasks. Through these experiments, and using multiple datasets and DL models (ResNet and SwinTransformer), we demonstrate the existence of both race and sex based shortcut learning and bias in DL based AD classification. Our work lays the foundation for fairer DL diagnostic tools in brain MRI. The code is provided at this https URL 

---
# An improved educational competition optimizer with multi-covariance learning operators for global optimization problems 

**Authors**: Baoqi Zhao, Xiong Yang, Hoileong Lee, Bowen Dong  

**Link**: [PDF](https://arxiv.org/pdf/2509.09552)  

**Abstract**: The educational competition optimizer is a recently introduced metaheuristic algorithm inspired by human behavior, originating from the dynamics of educational competition within society. Nonetheless, ECO faces constraints due to an imbalance between exploitation and exploration, rendering it susceptible to local optima and demonstrating restricted effectiveness in addressing complex optimization problems. To address these limitations, this study presents an enhanced educational competition optimizer (IECO-MCO) utilizing multi-covariance learning operators. In IECO, three distinct covariance learning operators are introduced to improve the performance of ECO. Each operator effectively balances exploitation and exploration while preventing premature convergence of the population. The effectiveness of IECO is assessed through benchmark functions derived from the CEC 2017 and CEC 2022 test suites, and its performance is compared with various basic and improved algorithms across different categories. The results demonstrate that IECO-MCO surpasses the basic ECO and other competing algorithms in convergence speed, stability, and the capability to avoid local optima. Furthermore, statistical analyses, including the Friedman test, Kruskal-Wallis test, and Wilcoxon rank-sum test, are conducted to validate the superiority of IECO-MCO over the compared algorithms. Compared with the basic algorithm (improved algorithm), IECO-MCO achieved an average ranking of 2.213 (2.488) on the CE2017 and CEC2022 test suites. Additionally, the practical applicability of the proposed IECO-MCO algorithm is verified by solving constrained optimization problems. The experimental outcomes demonstrate the superior performance of IECO-MCO in tackling intricate optimization problems, underscoring its robustness and practical effectiveness in real-world scenarios. 

---
# Improving Video Diffusion Transformer Training by Multi-Feature Fusion and Alignment from Self-Supervised Vision Encoders 

**Authors**: Dohun Lee, Hyeonho Jeong, Jiwook Kim, Duygu Ceylan, Jong Chul Ye  

**Link**: [PDF](https://arxiv.org/pdf/2509.09547)  

**Abstract**: Video diffusion models have advanced rapidly in the recent years as a result of series of architectural innovations (e.g., diffusion transformers) and use of novel training objectives (e.g., flow matching). In contrast, less attention has been paid to improving the feature representation power of such models. In this work, we show that training video diffusion models can benefit from aligning the intermediate features of the video generator with feature representations of pre-trained vision encoders. We propose a new metric and conduct an in-depth analysis of various vision encoders to evaluate their discriminability and temporal consistency, thereby assessing their suitability for video feature alignment. Based on the analysis, we present Align4Gen which provides a novel multi-feature fusion and alignment method integrated into video diffusion model training. We evaluate Align4Gen both for unconditional and class-conditional video generation tasks and show that it results in improved video generation as quantified by various metrics. Full video results are available on our project page: this https URL 

---
# A modified RIME algorithm with covariance learning and diversity enhancement for numerical optimization 

**Authors**: Shangqing Shi, Luoxiao Zhang, Yuchen Yin, Xiong Yang, Hoileong Lee  

**Link**: [PDF](https://arxiv.org/pdf/2509.09529)  

**Abstract**: Metaheuristics are widely applied for their ability to provide more efficient solutions. The RIME algorithm is a recently proposed physical-based metaheuristic algorithm with certain advantages. However, it suffers from rapid loss of population diversity during optimization and is prone to fall into local optima, leading to unbalanced exploitation and exploration. To address the shortcomings of RIME, this paper proposes a modified RIME with covariance learning and diversity enhancement (MRIME-CD). The algorithm applies three strategies to improve the optimization capability. First, a covariance learning strategy is introduced in the soft-rime search stage to increase the population diversity and balance the over-exploitation ability of RIME through the bootstrapping effect of dominant populations. Second, in order to moderate the tendency of RIME population to approach the optimal individual in the early search stage, an average bootstrapping strategy is introduced into the hard-rime puncture mechanism, which guides the population search through the weighted position of the dominant populations, thus enhancing the global search ability of RIME in the early stage. Finally, a new stagnation indicator is proposed, and a stochastic covariance learning strategy is used to update the stagnant individuals in the population when the algorithm gets stagnant, thus enhancing the ability to jump out of the local optimal solution. The proposed MRIME-CD algorithm is subjected to a series of validations on the CEC2017 test set, the CEC2022 test set, and the experimental results are analyzed using the Friedman test, the Wilcoxon rank sum test, and the Kruskal Wallis test. The results show that MRIME-CD can effectively improve the performance of basic RIME and has obvious superiorities in terms of solution accuracy, convergence speed and stability. 

---
# Towards Explainable Job Title Matching: Leveraging Semantic Textual Relatedness and Knowledge Graphs 

**Authors**: Vadim Zadykian, Bruno Andrade, Haithem Afli  

**Link**: [PDF](https://arxiv.org/pdf/2509.09522)  

**Abstract**: Semantic Textual Relatedness (STR) captures nuanced relationships between texts that extend beyond superficial lexical similarity. In this study, we investigate STR in the context of job title matching - a key challenge in resume recommendation systems, where overlapping terms are often limited or misleading. We introduce a self-supervised hybrid architecture that combines dense sentence embeddings with domain-specific Knowledge Graphs (KGs) to improve both semantic alignment and explainability. Unlike previous work that evaluated models on aggregate performance, our approach emphasizes data stratification by partitioning the STR score continuum into distinct regions: low, medium, and high semantic relatedness. This stratified evaluation enables a fine-grained analysis of model performance across semantically meaningful subspaces. We evaluate several embedding models, both with and without KG integration via graph neural networks. The results show that fine-tuned SBERT models augmented with KGs produce consistent improvements in the high-STR region, where the RMSE is reduced by 25% over strong baselines. Our findings highlight not only the benefits of combining KGs with text embeddings, but also the importance of regional performance analysis in understanding model behavior. This granular approach reveals strengths and weaknesses hidden by global metrics, and supports more targeted model selection for use in Human Resources (HR) systems and applications where fairness, explainability, and contextual matching are essential. 

---
# Explainable AI for Accelerated Microstructure Imaging: A SHAP-Guided Protocol on the Connectome 2.0 scanner 

**Authors**: Quentin Uhl, Tommaso Pavan, Julianna Gerold, Kwok-Shing Chan, Yohan Jun, Shohei Fujita, Aneri Bhatt, Yixin Ma, Qiaochu Wang, Hong-Hsi Lee, Susie Y. Huang, Berkin Bilgic, Ileana Jelescu  

**Link**: [PDF](https://arxiv.org/pdf/2509.09513)  

**Abstract**: The diffusion MRI Neurite Exchange Imaging model offers a promising framework for probing gray matter microstructure by estimating parameters such as compartment sizes, diffusivities, and inter-compartmental water exchange time. However, existing protocols require long scan times. This study proposes a reduced acquisition scheme for the Connectome 2.0 scanner that preserves model accuracy while substantially shortening scan duration. We developed a data-driven framework using explainable artificial intelligence with a guided recursive feature elimination strategy to identify an optimal 8-feature subset from a 15-feature protocol. The performance of this optimized protocol was validated in vivo and benchmarked against the full acquisition and alternative reduction strategies. Parameter accuracy, preservation of anatomical contrast, and test-retest reproducibility were assessed. The reduced protocol yielded parameter estimates and cortical maps comparable to the full protocol, with low estimation errors in synthetic data and minimal impact on test-retest variability. Compared to theory-driven and heuristic reduction schemes, the optimized protocol demonstrated superior robustness, reducing the deviation in water exchange time estimates by over two-fold. In conclusion, this hybrid optimization framework enables viable imaging of neurite exchange in 14 minutes without loss of parameter fidelity. This approach supports the broader application of exchange-sensitive diffusion magnetic resonance imaging in neuroscience and clinical research, and offers a generalizable method for designing efficient acquisition protocols in biophysical parameter mapping. 

---
# Incorporating AI Incident Reporting into Telecommunications Law and Policy: Insights from India 

**Authors**: Avinash Agarwal, Manisha J. Nene  

**Link**: [PDF](https://arxiv.org/pdf/2509.09508)  

**Abstract**: The integration of artificial intelligence (AI) into telecommunications infrastructure introduces novel risks, such as algorithmic bias and unpredictable system behavior, that fall outside the scope of traditional cybersecurity and data protection frameworks. This paper introduces a precise definition and a detailed typology of telecommunications AI incidents, establishing them as a distinct category of risk that extends beyond conventional cybersecurity and data protection breaches. It argues for their recognition as a distinct regulatory concern. Using India as a case study for jurisdictions that lack a horizontal AI law, the paper analyzes the country's key digital regulations. The analysis reveals that India's existing legal instruments, including the Telecommunications Act, 2023, the CERT-In Rules, and the Digital Personal Data Protection Act, 2023, focus on cybersecurity and data breaches, creating a significant regulatory gap for AI-specific operational incidents, such as performance degradation and algorithmic bias. The paper also examines structural barriers to disclosure and the limitations of existing AI incident repositories. Based on these findings, the paper proposes targeted policy recommendations centered on integrating AI incident reporting into India's existing telecom governance. Key proposals include mandating reporting for high-risk AI failures, designating an existing government body as a nodal agency to manage incident data, and developing standardized reporting frameworks. These recommendations aim to enhance regulatory clarity and strengthen long-term resilience, offering a pragmatic and replicable blueprint for other nations seeking to govern AI risks within their existing sectoral frameworks. 

---
# OpenFake: An Open Dataset and Platform Toward Large-Scale Deepfake Detection 

**Authors**: Victor Livernoche, Akshatha Arodi, Andreea Musulan, Zachary Yang, Adam Salvail, Gaétan Marceau Caron, Jean-François Godbout, Reihaneh Rabbany  

**Link**: [PDF](https://arxiv.org/pdf/2509.09495)  

**Abstract**: Deepfakes, synthetic media created using advanced AI techniques, have intensified the spread of misinformation, particularly in politically sensitive contexts. Existing deepfake detection datasets are often limited, relying on outdated generation methods, low realism, or single-face imagery, restricting the effectiveness for general synthetic image detection. By analyzing social media posts, we identify multiple modalities through which deepfakes propagate misinformation. Furthermore, our human perception study demonstrates that recently developed proprietary models produce synthetic images increasingly indistinguishable from real ones, complicating accurate identification by the general public. Consequently, we present a comprehensive, politically-focused dataset specifically crafted for benchmarking detection against modern generative models. This dataset contains three million real images paired with descriptive captions, which are used for generating 963k corresponding high-quality synthetic images from a mix of proprietary and open-source models. Recognizing the continual evolution of generative techniques, we introduce an innovative crowdsourced adversarial platform, where participants are incentivized to generate and submit challenging synthetic images. This ongoing community-driven initiative ensures that deepfake detection methods remain robust and adaptive, proactively safeguarding public discourse from sophisticated misinformation threats. 

---
# Prompt Pirates Need a Map: Stealing Seeds helps Stealing Prompts 

**Authors**: Felix Mächtle, Ashwath Shetty, Jonas Sander, Nils Loose, Sören Pirk, Thomas Eisenbarth  

**Link**: [PDF](https://arxiv.org/pdf/2509.09488)  

**Abstract**: Diffusion models have significantly advanced text-to-image generation, enabling the creation of highly realistic images conditioned on textual prompts and seeds. Given the considerable intellectual and economic value embedded in such prompts, prompt theft poses a critical security and privacy concern. In this paper, we investigate prompt-stealing attacks targeting diffusion models. We reveal that numerical optimization-based prompt recovery methods are fundamentally limited as they do not account for the initial random noise used during image generation. We identify and exploit a noise-generation vulnerability (CWE-339), prevalent in major image-generation frameworks, originating from PyTorch's restriction of seed values to a range of $2^{32}$ when generating the initial random noise on CPUs. Through a large-scale empirical analysis conducted on images shared via the popular platform CivitAI, we demonstrate that approximately 95% of these images' seed values can be effectively brute-forced in 140 minutes per seed using our seed-recovery tool, SeedSnitch. Leveraging the recovered seed, we propose PromptPirate, a genetic algorithm-based optimization method explicitly designed for prompt stealing. PromptPirate surpasses state-of-the-art methods, i.e., PromptStealer, P2HP, and CLIP-Interrogator, achieving an 8-11% improvement in LPIPS similarity. Furthermore, we introduce straightforward and effective countermeasures that render seed stealing, and thus optimization-based prompt stealing, ineffective. We have disclosed our findings responsibly and initiated coordinated mitigation efforts with the developers to address this critical vulnerability. 

---
# Resource-Efficient Glioma Segmentation on Sub-Saharan MRI 

**Authors**: Freedmore Sidume, Oumayma Soula, Joseph Muthui Wacira, YunFei Zhu, Abbas Rabiu Muhammad, Abderrazek Zeraii, Oluwaseun Kalejaye, Hajer Ibrahim, Olfa Gaddour, Brain Halubanza, Dong Zhang, Udunna C Anazodo, Confidence Raymond  

**Link**: [PDF](https://arxiv.org/pdf/2509.09469)  

**Abstract**: Gliomas are the most prevalent type of primary brain tumors, and their accurate segmentation from MRI is critical for diagnosis, treatment planning, and longitudinal monitoring. However, the scarcity of high-quality annotated imaging data in Sub-Saharan Africa (SSA) poses a significant challenge for deploying advanced segmentation models in clinical workflows. This study introduces a robust and computationally efficient deep learning framework tailored for resource-constrained settings. We leveraged a 3D Attention UNet architecture augmented with residual blocks and enhanced through transfer learning from pre-trained weights on the BraTS 2021 dataset. Our model was evaluated on 95 MRI cases from the BraTS-Africa dataset, a benchmark for glioma segmentation in SSA MRI data. Despite the limited data quality and quantity, our approach achieved Dice scores of 0.76 for the Enhancing Tumor (ET), 0.80 for Necrotic and Non-Enhancing Tumor Core (NETC), and 0.85 for Surrounding Non-Functional Hemisphere (SNFH). These results demonstrate the generalizability of the proposed model and its potential to support clinical decision making in low-resource settings. The compact architecture, approximately 90 MB, and sub-minute per-volume inference time on consumer-grade hardware further underscore its practicality for deployment in SSA health systems. This work contributes toward closing the gap in equitable AI for global health by empowering underserved regions with high-performing and accessible medical imaging solutions. 

---
# ENSI: Efficient Non-Interactive Secure Inference for Large Language Models 

**Authors**: Zhiyu He, Maojiang Wang, Xinwen Gao, Yuchuan Luo, Lin Liu, Shaojing Fu  

**Link**: [PDF](https://arxiv.org/pdf/2509.09424)  

**Abstract**: Secure inference enables privacy-preserving machine learning by leveraging cryptographic protocols that support computations on sensitive user data without exposing it. However, integrating cryptographic protocols with large language models (LLMs) presents significant challenges, as the inherent complexity of these protocols, together with LLMs' massive parameter scale and sophisticated architectures, severely limits practical usability. In this work, we propose ENSI, a novel non-interactive secure inference framework for LLMs, based on the principle of co-designing the cryptographic protocols and LLM architecture. ENSI employs an optimized encoding strategy that seamlessly integrates CKKS scheme with a lightweight LLM variant, BitNet, significantly reducing the computational complexity of encrypted matrix multiplications. In response to the prohibitive computational demands of softmax under homomorphic encryption (HE), we pioneer the integration of the sigmoid attention mechanism with HE as a seamless, retraining-free alternative. Furthermore, by embedding the Bootstrapping operation within the RMSNorm process, we efficiently refresh ciphertexts while markedly decreasing the frequency of costly bootstrapping invocations. Experimental evaluations demonstrate that ENSI achieves approximately an 8x acceleration in matrix multiplications and a 2.6x speedup in softmax inference on CPU compared to state-of-the-art method, with the proportion of bootstrapping is reduced to just 1%. 

---
# We're Still Doing It (All) Wrong: Recommender Systems, Fifteen Years Later 

**Authors**: Alan Said, Maria Soledad Pera, Michael D. Ekstrand  

**Link**: [PDF](https://arxiv.org/pdf/2509.09414)  

**Abstract**: In 2011, Xavier Amatriain sounded the alarm: recommender systems research was "doing it all wrong" [1]. His critique, rooted in statistical misinterpretation and methodological shortcuts, remains as relevant today as it was then. But rather than correcting course, we added new layers of sophistication on top of the same broken foundations. This paper revisits Amatriain's diagnosis and argues that many of the conceptual, epistemological, and infrastructural failures he identified still persist, in more subtle or systemic forms. Drawing on recent work in reproducibility, evaluation methodology, environmental impact, and participatory design, we showcase how the field's accelerating complexity has outpaced its introspection. We highlight ongoing community-led initiatives that attempt to shift the paradigm, including workshops, evaluation frameworks, and calls for value-sensitive and participatory research. At the same time, we contend that meaningful change will require not only new metrics or better tooling, but a fundamental reframing of what recommender systems research is for, who it serves, and how knowledge is produced and validated. Our call is not just for technical reform, but for a recommender systems research agenda grounded in epistemic humility, human impact, and sustainable practice. 

---
# LLMs Don't Know Their Own Decision Boundaries: The Unreliability of Self-Generated Counterfactual Explanations 

**Authors**: Harry Mayne, Ryan Othniel Kearns, Yushi Yang, Andrew M. Bean, Eoin Delaney, Chris Russell, Adam Mahdi  

**Link**: [PDF](https://arxiv.org/pdf/2509.09396)  

**Abstract**: To collaborate effectively with humans, language models must be able to explain their decisions in natural language. We study a specific type of self-explanation: self-generated counterfactual explanations (SCEs), where a model explains its prediction by modifying the input such that it would have predicted a different outcome. We evaluate whether LLMs can produce SCEs that are valid, achieving the intended outcome, and minimal, modifying the input no more than necessary. When asked to generate counterfactuals, we find that LLMs typically produce SCEs that are valid, but far from minimal, offering little insight into their decision-making behaviour. Worryingly, when asked to generate minimal counterfactuals, LLMs typically make excessively small edits that fail to change predictions. The observed validity-minimality trade-off is consistent across several LLMs, datasets, and evaluation settings. Our findings suggest that SCEs are, at best, an ineffective explainability tool and, at worst, can provide misleading insights into model behaviour. Proposals to deploy LLMs in high-stakes settings must consider the impact of unreliable self-explanations on downstream decision-making. Our code is available at this https URL. 

---
# MetaLLMix : An XAI Aided LLM-Meta-learning Based Approach for Hyper-parameters Optimization 

**Authors**: Mohammed Tiouti, Mohamed Bal-Ghaoui  

**Link**: [PDF](https://arxiv.org/pdf/2509.09387)  

**Abstract**: Effective model and hyperparameter selection remains a major challenge in deep learning, often requiring extensive expertise and computation. While AutoML and large language models (LLMs) promise automation, current LLM-based approaches rely on trial and error and expensive APIs, which provide limited interpretability and generalizability. We propose MetaLLMiX, a zero-shot hyperparameter optimization framework combining meta-learning, explainable AI, and efficient LLM reasoning. By leveraging historical experiment outcomes with SHAP explanations, MetaLLMiX recommends optimal hyperparameters and pretrained models without additional trials. We further employ an LLM-as-judge evaluation to control output format, accuracy, and completeness. Experiments on eight medical imaging datasets using nine open-source lightweight LLMs show that MetaLLMiX achieves competitive or superior performance to traditional HPO methods while drastically reducing computational cost. Our local deployment outperforms prior API-based approaches, achieving optimal results on 5 of 8 tasks, response time reductions of 99.6-99.9%, and the fastest training times on 6 datasets (2.4-15.7x faster), maintaining accuracy within 1-5% of best-performing baselines. 

---
# Robust Non-Linear Correlations via Polynomial Regression 

**Authors**: Luca Giuliani, Michele Lombardi  

**Link**: [PDF](https://arxiv.org/pdf/2509.09380)  

**Abstract**: The Hirschfeld-Gebelein-Rényi (HGR) correlation coefficient is an extension of Pearson's correlation that is not limited to linear correlations, with potential applications in algorithmic fairness, scientific analysis, and causal discovery. Recently, novel algorithms to estimate HGR in a differentiable manner have been proposed to facilitate its use as a loss regularizer in constrained machine learning applications. However, the inherent uncomputability of HGR requires a bias-variance trade-off, which can possibly compromise the robustness of the proposed methods, hence raising technical concerns if applied in real-world scenarios. We introduce a novel computational approach for HGR that relies on user-configurable polynomial kernels, offering greater robustness compared to previous methods and featuring a faster yet almost equally effective restriction. Our approach provides significant advantages in terms of robustness and determinism, making it a more reliable option for real-world applications. Moreover, we present a brief experimental analysis to validate the applicability of our approach within a constrained machine learning framework, showing that its computation yields an insightful subgradient that can serve as a loss regularizer. 

---
# Classification of Driver Behaviour Using External Observation Techniques for Autonomous Vehicles 

**Authors**: Ian Nell, Shane Gilroy  

**Link**: [PDF](https://arxiv.org/pdf/2509.09349)  

**Abstract**: Road traffic accidents remain a significant global concern, with human error, particularly distracted and impaired driving, among the leading causes. This study introduces a novel driver behavior classification system that uses external observation techniques to detect indicators of distraction and impairment. The proposed framework employs advanced computer vision methodologies, including real-time object tracking, lateral displacement analysis, and lane position monitoring. The system identifies unsafe driving behaviors such as excessive lateral movement and erratic trajectory patterns by implementing the YOLO object detection model and custom lane estimation algorithms. Unlike systems reliant on inter-vehicular communication, this vision-based approach enables behavioral analysis of non-connected vehicles. Experimental evaluations on diverse video datasets demonstrate the framework's reliability and adaptability across varying road and environmental conditions. 

---
# MoSE: Unveiling Structural Patterns in Graphs via Mixture of Subgraph Experts 

**Authors**: Junda Ye, Zhongbao Zhang, Li Sun, Siqiang Luo  

**Link**: [PDF](https://arxiv.org/pdf/2509.09337)  

**Abstract**: While graph neural networks (GNNs) have achieved great success in learning from graph-structured data, their reliance on local, pairwise message passing restricts their ability to capture complex, high-order subgraph patterns. leading to insufficient structural expressiveness. Recent efforts have attempted to enhance structural expressiveness by integrating random walk kernels into GNNs. However, these methods are inherently designed for graph-level tasks, which limits their applicability to other downstream tasks such as node classification. Moreover, their fixed kernel configurations hinder the model's flexibility in capturing diverse subgraph structures. To address these limitations, this paper proposes a novel Mixture of Subgraph Experts (MoSE) framework for flexible and expressive subgraph-based representation learning across diverse graph tasks. Specifically, MoSE extracts informative subgraphs via anonymous walks and dynamically routes them to specialized experts based on structural semantics, enabling the model to capture diverse subgraph patterns with improved flexibility and interpretability. We further provide a theoretical analysis of MoSE's expressivity within the Subgraph Weisfeiler-Lehman (SWL) Test, proving that it is more powerful than SWL. Extensive experiments, together with visualizations of learned subgraph experts, demonstrate that MoSE not only outperforms competitive baselines but also provides interpretable insights into structural patterns learned by the model. 

---
# OmniEVA: Embodied Versatile Planner via Task-Adaptive 3D-Grounded and Embodiment-aware Reasoning 

**Authors**: Yuecheng Liu, Dafeng Chi, Shiguang Wu, Zhanguang Zhang, Yuzheng Zhuang, Bowen Yang, He Zhu, Lingfeng Zhang, Pengwei Xie, David Gamaliel Arcos Bravo, Yingxue Zhang, Jianye Hao, Xingyue Quan  

**Link**: [PDF](https://arxiv.org/pdf/2509.09332)  

**Abstract**: Recent advances in multimodal large language models (MLLMs) have opened new opportunities for embodied intelligence, enabling multimodal understanding, reasoning, and interaction, as well as continuous spatial decision-making. Nevertheless, current MLLM-based embodied systems face two critical limitations. First, Geometric Adaptability Gap: models trained solely on 2D inputs or with hard-coded 3D geometry injection suffer from either insufficient spatial information or restricted 2D generalization, leading to poor adaptability across tasks with diverse spatial demands. Second, Embodiment Constraint Gap: prior work often neglects the physical constraints and capacities of real robots, resulting in task plans that are theoretically valid but practically this http URL address these gaps, we introduce OmniEVA -- an embodied versatile planner that enables advanced embodied reasoning and task planning through two pivotal innovations: (1) a Task-Adaptive 3D Grounding mechanism, which introduces a gated router to perform explicit selective regulation of 3D fusion based on contextual requirements, enabling context-aware 3D grounding for diverse embodied tasks. (2) an Embodiment-Aware Reasoning framework that jointly incorporates task goals and embodiment constraints into the reasoning loop, resulting in planning decisions that are both goal-directed and executable. Extensive experimental results demonstrate that OmniEVA not only achieves state-of-the-art general embodied reasoning performance, but also exhibits a strong ability across a wide range of downstream scenarios. Evaluations of a suite of proposed embodied benchmarks, including both primitive and composite tasks, confirm its robust and versatile planning capabilities. Project page: this https URL 

---
# Can Multimodal LLMs See Materials Clearly? A Multimodal Benchmark on Materials Characterization 

**Authors**: Zhengzhao Lai, Youbin Zheng, Zhenyang Cai, Haonan Lyu, Jinpu Yang, Hongqing Liang, Yan Hu, Benyou Wang  

**Link**: [PDF](https://arxiv.org/pdf/2509.09307)  

**Abstract**: Materials characterization is fundamental to acquiring materials information, revealing the processing-microstructure-property relationships that guide material design and optimization. While multimodal large language models (MLLMs) have recently shown promise in generative and predictive tasks within materials science, their capacity to understand real-world characterization imaging data remains underexplored. To bridge this gap, we present MatCha, the first benchmark for materials characterization image understanding, comprising 1,500 questions that demand expert-level domain expertise. MatCha encompasses four key stages of materials research comprising 21 distinct tasks, each designed to reflect authentic challenges faced by materials scientists. Our evaluation of state-of-the-art MLLMs on MatCha reveals a significant performance gap compared to human experts. These models exhibit degradation when addressing questions requiring higher-level expertise and sophisticated visual perception. Simple few-shot and chain-of-thought prompting struggle to alleviate these limitations. These findings highlight that existing MLLMs still exhibit limited adaptability to real-world materials characterization scenarios. We hope MatCha will facilitate future research in areas such as new material discovery and autonomous scientific agents. MatCha is available at this https URL. 

---
# Modality-Agnostic Input Channels Enable Segmentation of Brain lesions in Multimodal MRI with Sequences Unavailable During Training 

**Authors**: Anthony P. Addison, Felix Wagner, Wentian Xu, Natalie Voets, Konstantinos Kamnitsas  

**Link**: [PDF](https://arxiv.org/pdf/2509.09290)  

**Abstract**: Segmentation models are important tools for the detection and analysis of lesions in brain MRI. Depending on the type of brain pathology that is imaged, MRI scanners can acquire multiple, different image modalities (contrasts). Most segmentation models for multimodal brain MRI are restricted to fixed modalities and cannot effectively process new ones at inference. Some models generalize to unseen modalities but may lose discriminative modality-specific information. This work aims to develop a model that can perform inference on data that contain image modalities unseen during training, previously seen modalities, and heterogeneous combinations of both, thus allowing a user to utilize any available imaging modalities. We demonstrate this is possible with a simple, thus practical alteration to the U-net architecture, by integrating a modality-agnostic input channel or pathway, alongside modality-specific input channels. To train this modality-agnostic component, we develop an image augmentation scheme that synthesizes artificial MRI modalities. Augmentations differentially alter the appearance of pathological and healthy brain tissue to create artificial contrasts between them while maintaining realistic anatomical integrity. We evaluate the method using 8 MRI databases that include 5 types of pathologies (stroke, tumours, traumatic brain injury, multiple sclerosis and white matter hyperintensities) and 8 modalities (T1, T1+contrast, T2, PD, SWI, DWI, ADC and FLAIR). The results demonstrate that the approach preserves the ability to effectively process MRI modalities encountered during training, while being able to process new, unseen modalities to improve its segmentation. Project code: this https URL 

---
# Adaptive Knowledge Distillation using a Device-Aware Teacher for Low-Complexity Acoustic Scene Classification 

**Authors**: Seung Gyu Jeong, Seong Eun Kim  

**Link**: [PDF](https://arxiv.org/pdf/2509.09262)  

**Abstract**: In this technical report, we describe our submission for Task 1, Low-Complexity Device-Robust Acoustic Scene Classification, of the DCASE 2025 Challenge. Our work tackles the dual challenges of strict complexity constraints and robust generalization to both seen and unseen devices, while also leveraging the new rule allowing the use of device labels at test time. Our proposed system is based on a knowledge distillation framework where an efficient CP-MobileNet student learns from a compact, specialized two-teacher ensemble. This ensemble combines a baseline PaSST teacher, trained with standard cross-entropy, and a 'generalization expert' teacher. This expert is trained using our novel Device-Aware Feature Alignment (DAFA) loss, adapted from prior work, which explicitly structures the feature space for device robustness. To capitalize on the availability of test-time device labels, the distilled student model then undergoes a final device-specific fine-tuning stage. Our proposed system achieves a final accuracy of 57.93\% on the development set, demonstrating a significant improvement over the official baseline, particularly on unseen devices. 

---
# CoAtNeXt:An Attention-Enhanced ConvNeXtV2-Transformer Hybrid Model for Gastric Tissue Classification 

**Authors**: Mustafa Yurdakul, Sakir Tasdemir  

**Link**: [PDF](https://arxiv.org/pdf/2509.09242)  

**Abstract**: Background and objective Early diagnosis of gastric diseases is crucial to prevent fatal outcomes. Although histopathologic examination remains the diagnostic gold standard, it is performed entirely manually, making evaluations labor-intensive and prone to variability among pathologists. Critical findings may be missed, and lack of standard procedures reduces consistency. These limitations highlight the need for automated, reliable, and efficient methods for gastric tissue analysis. Methods In this study, a novel hybrid model named CoAtNeXt was proposed for the classification of gastric tissue images. The model is built upon the CoAtNet architecture by replacing its MBConv layers with enhanced ConvNeXtV2 blocks. Additionally, the Convolutional Block Attention Module (CBAM) is integrated to improve local feature extraction through channel and spatial attention mechanisms. The architecture was scaled to achieve a balance between computational efficiency and classification performance. CoAtNeXt was evaluated on two publicly available datasets, HMU-GC-HE-30K for eight-class classification and GasHisSDB for binary classification, and was compared against 10 Convolutional Neural Networks (CNNs) and ten Vision Transformer (ViT) models. Results CoAtNeXt achieved 96.47% accuracy, 96.60% precision, 96.47% recall, 96.45% F1 score, and 99.89% AUC on HMU-GC-HE-30K. On GasHisSDB, it reached 98.29% accuracy, 98.07% precision, 98.41% recall, 98.23% F1 score, and 99.90% AUC. It outperformed all CNN and ViT models tested and surpassed previous studies in the literature. Conclusion Experimental results show that CoAtNeXt is a robust architecture for histopathological classification of gastric tissue images, providing performance on binary and multiclass. Its highlights its potential to assist pathologists by enhancing diagnostic accuracy and reducing workload. 

---
# Virtual staining for 3D X-ray histology of bone implants 

**Authors**: Sarah C. Irvine, Christian Lucas, Diana Krüger, Bianca Guedert, Julian Moosmann, Berit Zeller-Plumhoff  

**Link**: [PDF](https://arxiv.org/pdf/2509.09235)  

**Abstract**: Three-dimensional X-ray histology techniques offer a non-invasive alternative to conventional 2D histology, enabling volumetric imaging of biological tissues without the need for physical sectioning or chemical staining. However, the inherent greyscale image contrast of X-ray tomography limits its biochemical specificity compared to traditional histological stains. Within digital pathology, deep learning-based virtual staining has demonstrated utility in simulating stained appearances from label-free optical images. In this study, we extend virtual staining to the X-ray domain by applying cross-modality image translation to generate artificially stained slices from synchrotron-radiation-based micro-CT scans. Using over 50 co-registered image pairs of micro-CT and toluidine blue-stained histology from bone-implant samples, we trained a modified CycleGAN network tailored for limited paired data. Whole slide histology images were downsampled to match the voxel size of the CT data, with on-the-fly data augmentation for patch-based training. The model incorporates pixelwise supervision and greyscale consistency terms, producing histologically realistic colour outputs while preserving high-resolution structural detail. Our method outperformed Pix2Pix and standard CycleGAN baselines across SSIM, PSNR, and LPIPS metrics. Once trained, the model can be applied to full CT volumes to generate virtually stained 3D datasets, enhancing interpretability without additional sample preparation. While features such as new bone formation were able to be reproduced, some variability in the depiction of implant degradation layers highlights the need for further training data and refinement. This work introduces virtual staining to 3D X-ray imaging and offers a scalable route for chemically informative, label-free tissue characterisation in biomedical research. 

---
# Vejde: A Framework for Inductive Deep Reinforcement Learning Based on Factor Graph Color Refinement 

**Authors**: Jakob Nyberg, Pontus Johnson  

**Link**: [PDF](https://arxiv.org/pdf/2509.09219)  

**Abstract**: We present and evaluate Vejde; a framework which combines data abstraction, graph neural networks and reinforcement learning to produce inductive policy functions for decision problems with richly structured states, such as object classes and relations. MDP states are represented as data bases of facts about entities, and Vejde converts each state to a bipartite graph, which is mapped to latent states through neural message passing. The factored representation of both states and actions allows Vejde agents to handle problems of varying size and structure. We tested Vejde agents on eight problem domains defined in RDDL, with ten problem instances each, where policies were trained using both supervised and reinforcement learning. To test policy generalization, we separate problem instances in two sets, one for training and the other solely for testing. Test results on unseen instances for the Vejde agents were compared to MLP agents trained on each problem instance, as well as the online planning algorithm Prost. Our results show that Vejde policies in average generalize to the test instances without a significant loss in score. Additionally, the inductive agents received scores on unseen test instances that on average were close to the instance-specific MLP agents. 

---
# Incentivizing Safer Actions in Policy Optimization for Constrained Reinforcement Learning 

**Authors**: Somnath Hazra, Pallab Dasgupta, Soumyajit Dey  

**Link**: [PDF](https://arxiv.org/pdf/2509.09208)  

**Abstract**: Constrained Reinforcement Learning (RL) aims to maximize the return while adhering to predefined constraint limits, which represent domain-specific safety requirements. In continuous control settings, where learning agents govern system actions, balancing the trade-off between reward maximization and constraint satisfaction remains a significant challenge. Policy optimization methods often exhibit instability near constraint boundaries, resulting in suboptimal training performance. To address this issue, we introduce a novel approach that integrates an adaptive incentive mechanism in addition to the reward structure to stay within the constraint bound before approaching the constraint boundary. Building on this insight, we propose Incrementally Penalized Proximal Policy Optimization (IP3O), a practical algorithm that enforces a progressively increasing penalty to stabilize training dynamics. Through empirical evaluation on benchmark environments, we demonstrate the efficacy of IP3O compared to the performance of state-of-the-art Safe RL algorithms. Furthermore, we provide theoretical guarantees by deriving a bound on the worst-case error of the optimality achieved by our algorithm. 

---
# Bona fide Cross Testing Reveals Weak Spot in Audio Deepfake Detection Systems 

**Authors**: Chin Yuen Kwok, Jia Qi Yip, Zhen Qiu, Chi Hung Chi, Kwok Yan Lam  

**Link**: [PDF](https://arxiv.org/pdf/2509.09204)  

**Abstract**: Audio deepfake detection (ADD) models are commonly evaluated using datasets that combine multiple synthesizers, with performance reported as a single Equal Error Rate (EER). However, this approach disproportionately weights synthesizers with more samples, underrepresenting others and reducing the overall reliability of EER. Additionally, most ADD datasets lack diversity in bona fide speech, often featuring a single environment and speech style (e.g., clean read speech), limiting their ability to simulate real-world conditions. To address these challenges, we propose bona fide cross-testing, a novel evaluation framework that incorporates diverse bona fide datasets and aggregates EERs for more balanced assessments. Our approach improves robustness and interpretability compared to traditional evaluation methods. We benchmark over 150 synthesizers across nine bona fide speech types and release a new dataset to facilitate further research at this https URL. 

---
# Improving Synthetic Data Training for Contextual Biasing Models with a Keyword-Aware Cost Function 

**Authors**: Chin Yuen Kwok, Jia Qi Yip, Eng Siong Chng  

**Link**: [PDF](https://arxiv.org/pdf/2509.09197)  

**Abstract**: Rare word recognition can be improved by adapting ASR models to synthetic data that includes these words. Further improvements can be achieved through contextual biasing, which trains and adds a biasing module into the model architecture to prioritize rare words. While training the module on synthetic rare word data is more effective than using non-rare-word data, it can lead to overfitting due to artifacts in the synthetic audio. To address this, we enhance the TCPGen-based contextual biasing approach and propose a keyword-aware loss function that additionally focuses on biased words when training biasing modules. This loss includes a masked cross-entropy term for biased word prediction and a binary classification term for detecting biased word positions. These two terms complementarily support the decoding of biased words during inference. By adapting Whisper to 10 hours of synthetic data, our method reduced the word error rate on the NSC Part 2 test set from 29.71% to 11.81%. 

---
# Efficient Trie-based Biasing using K-step Prediction for Rare Word Recognition 

**Authors**: Chin Yuen Kwok, Jia Qi yip  

**Link**: [PDF](https://arxiv.org/pdf/2509.09196)  

**Abstract**: Contextual biasing improves rare word recognition of ASR models by prioritizing the output of rare words during decoding. A common approach is Trie-based biasing, which gives "bonus scores" to partial hypothesis (e.g. "Bon") that may lead to the generation of the rare word (e.g. "Bonham"). If the full word ("Bonham") isn't ultimately recognized, the system revokes those earlier bonuses. This revocation is limited to beam search and is computationally expensive, particularly for models with large decoders. To overcome these limitations, we propose adapting ASR models to look ahead and predict multiple steps at once. This avoids the revocation step entirely by better estimating whether a partial hypothesis will lead to the generation of the full rare word. By fine-tuning Whisper with only 10 hours of synthetic data, our method reduces the word error rate on the NSC Part 2 test set from 30.86% to 12.19%. 

---
# On Integrating Large Language Models and Scenario-Based Programming for Improving Software Reliability 

**Authors**: Ayelet Berzack, Guy Katz  

**Link**: [PDF](https://arxiv.org/pdf/2509.09194)  

**Abstract**: Large Language Models (LLMs) are fast becoming indispensable tools for software developers, assisting or even partnering with them in crafting complex programs. The advantages are evident -- LLMs can significantly reduce development time, generate well-organized and comprehensible code, and occasionally suggest innovative ideas that developers might not conceive on their own. However, despite their strengths, LLMs will often introduce significant errors and present incorrect code with persuasive confidence, potentially misleading developers into accepting flawed solutions.
In order to bring LLMs into the software development cycle in a more reliable manner, we propose a methodology for combining them with ``traditional'' software engineering techniques in a structured way, with the goal of streamlining the development process, reducing errors, and enabling users to verify crucial program properties with increased confidence. Specifically, we focus on the Scenario-Based Programming (SBP) paradigm -- an event-driven, scenario-based approach for software engineering -- to allow human developers to pour their expert knowledge into the LLM, as well as to inspect and verify its outputs.
To evaluate our methodology, we conducted a significant case study, and used it to design and implement the Connect4 game. By combining LLMs and SBP we were able to create a highly-capable agent, which could defeat various strong existing agents. Further, in some cases, we were able to formally verify the correctness of our agent. Finally, our experience reveals interesting insights regarding the ease-of-use of our proposed approach. The full code of our case-study will be made publicly available with the final version of this paper. 

---
# Probing Pre-trained Language Models on Code Changes: Insights from ReDef, a High-Confidence Just-in-Time Defect Prediction Dataset 

**Authors**: Doha Nam, Taehyoun Kim, Duksan Ryu, Jongmoon Baik  

**Link**: [PDF](https://arxiv.org/pdf/2509.09192)  

**Abstract**: Just-in-Time software defect prediction (JIT-SDP) plays a critical role in prioritizing risky code changes during code review and continuous integration. However, existing datasets often suffer from noisy labels and low precision in identifying bug-inducing commits. To address this, we present ReDef (Revert-based Defect dataset), a high-confidence benchmark of function-level modifications curated from 22 large-scale C/C++ projects. Defective cases are anchored by revert commits, while clean cases are validated through post-hoc history checks. Ambiguous instances are conservatively filtered out via a GPT-assisted triage process involving multiple votes and audits. This pipeline yields 3,164 defective and 10,268 clean modifications, offering substantially more reliable labels than prior existing resources. Beyond dataset construction, we provide the first systematic evaluation of how pre-trained language models (PLMs) reason about code modifications -- specifically, which input encodings most effectively expose change information, and whether models genuinely capture edit semantics. We fine-tune CodeBERT, CodeT5+, and UniXcoder under five encoding strategies, and further probe their sensitivity through counterfactual perturbations that swap added/deleted blocks, invert diff polarity, or inject spurious markers. Our results show that compact diff-style encodings consistently outperform whole-function formats across all PLMs, with statistical tests confirming large, model-independent effects. However, under counterfactual tests, performance degrades little or not at all -- revealing that what appears to be robustness in fact reflects reliance on superficial cues rather than true semantic understanding. These findings indicate that, unlike in snapshot-based tasks, current PLMs remain limited in their ability to genuinely comprehend code modifications. 

---
# Dark-ISP: Enhancing RAW Image Processing for Low-Light Object Detection 

**Authors**: Jiasheng Guo, Xin Gao, Yuxiang Yan, Guanghao Li, Jian Pu  

**Link**: [PDF](https://arxiv.org/pdf/2509.09183)  

**Abstract**: Low-light Object detection is crucial for many real-world applications but remains challenging due to degraded image quality. While recent studies have shown that RAW images offer superior potential over RGB images, existing approaches either use RAW-RGB images with information loss or employ complex frameworks. To address these, we propose a lightweight and self-adaptive Image Signal Processing (ISP) plugin, Dark-ISP, which directly processes Bayer RAW images in dark environments, enabling seamless end-to-end training for object detection. Our key innovations are: (1) We deconstruct conventional ISP pipelines into sequential linear (sensor calibration) and nonlinear (tone mapping) sub-modules, recasting them as differentiable components optimized through task-driven losses. Each module is equipped with content-aware adaptability and physics-informed priors, enabling automatic RAW-to-RGB conversion aligned with detection objectives. (2) By exploiting the ISP pipeline's intrinsic cascade structure, we devise a Self-Boost mechanism that facilitates cooperation between sub-modules. Through extensive experiments on three RAW image datasets, we demonstrate that our method outperforms state-of-the-art RGB- and RAW-based detection approaches, achieving superior results with minimal parameters in challenging low-light environments. 

---
# EchoX: Towards Mitigating Acoustic-Semantic Gap via Echo Training for Speech-to-Speech LLMs 

**Authors**: Yuhao Zhang, Yuhao Du, Zhanchen Dai, Xiangnan Ma, Kaiqi Kou, Benyou Wang, Haizhou Li  

**Link**: [PDF](https://arxiv.org/pdf/2509.09174)  

**Abstract**: Speech-to-speech large language models (SLLMs) are attracting increasing attention. Derived from text-based large language models (LLMs), SLLMs often exhibit degradation in knowledge and reasoning capabilities. We hypothesize that this limitation arises because current training paradigms for SLLMs fail to bridge the acoustic-semantic gap in the feature representation space. To address this issue, we propose EchoX, which leverages semantic representations and dynamically generates speech training targets. This approach integrates both acoustic and semantic learning, enabling EchoX to preserve strong reasoning abilities as a speech LLM. Experimental results demonstrate that EchoX, with about six thousand hours of training data, achieves advanced performance on multiple knowledge-based question-answering benchmarks. The project is available at this https URL. 

---
# Adaptive Pareto-Optimal Token Merging for Edge Transformer Models in Semantic Communication 

**Authors**: Omar Erak, Omar Alhussein, Hatem Abou-Zeid, Mehdi Bennis  

**Link**: [PDF](https://arxiv.org/pdf/2509.09168)  

**Abstract**: Large-scale transformer models have emerged as a powerful tool for semantic communication systems, enabling edge devices to extract rich representations for robust inference across noisy wireless channels. However, their substantial computational demands remain a major barrier to practical deployment in resource-constrained 6G networks. In this paper, we present a training-free framework for adaptive token merging in pretrained vision transformers to jointly reduce inference time and transmission resource usage. We formulate the selection of per-layer merging proportions as a multi-objective optimization problem to balance accuracy and computational cost. We employ Gaussian process-based Bayesian optimization to construct a Pareto frontier of optimal configurations, enabling flexible runtime adaptation to dynamic application requirements and channel conditions. Extensive experiments demonstrate that our method consistently outperforms other baselines and achieves significant reductions in floating-point operations while maintaining competitive accuracy across a wide range of signal-to-noise ratio (SNR) conditions. Additional results highlight the effectiveness of adaptive policies that adjust merging aggressiveness in response to channel quality, providing a practical mechanism to trade off latency and semantic fidelity on demand. These findings establish a scalable and efficient approach for deploying transformer-based semantic communication in future edge intelligence systems. 

---
# Target-oriented Multimodal Sentiment Classification with Counterfactual-enhanced Debiasing 

**Authors**: Zhiyue Liu, Fanrong Ma, Xin Ling  

**Link**: [PDF](https://arxiv.org/pdf/2509.09160)  

**Abstract**: Target-oriented multimodal sentiment classification seeks to predict sentiment polarity for specific targets from image-text pairs. While existing works achieve competitive performance, they often over-rely on textual content and fail to consider dataset biases, in particular word-level contextual biases. This leads to spurious correlations between text features and output labels, impairing classification accuracy. In this paper, we introduce a novel counterfactual-enhanced debiasing framework to reduce such spurious correlations. Our framework incorporates a counterfactual data augmentation strategy that minimally alters sentiment-related causal features, generating detail-matched image-text samples to guide the model's attention toward content tied to sentiment. Furthermore, for learning robust features from counterfactual data and prompting model decisions, we introduce an adaptive debiasing contrastive learning mechanism, which effectively mitigates the influence of biased words. Experimental results on several benchmark datasets show that our proposed method outperforms state-of-the-art baselines. 

---
# A Knowledge Noise Mitigation Framework for Knowledge-based Visual Question Answering 

**Authors**: Zhiyue Liu, Sihang Liu, Jinyuan Liu, Xinru Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2509.09159)  

**Abstract**: Knowledge-based visual question answering (KB-VQA) requires a model to understand images and utilize external knowledge to provide accurate answers. Existing approaches often directly augment models with retrieved information from knowledge sources while ignoring substantial knowledge redundancy, which introduces noise into the answering process. To address this, we propose a training-free framework with knowledge focusing for KB-VQA, that mitigates the impact of noise by enhancing knowledge relevance and reducing redundancy. First, for knowledge retrieval, our framework concludes essential parts from the image-question pairs, creating low-noise queries that enhance the retrieval of highly relevant knowledge. Considering that redundancy still persists in the retrieved knowledge, we then prompt large models to identify and extract answer-beneficial segments from knowledge. In addition, we introduce a selective knowledge integration strategy, allowing the model to incorporate knowledge only when it lacks confidence in answering the question, thereby mitigating the influence of redundant information. Our framework enables the acquisition of accurate and critical knowledge, and extensive experiments demonstrate that it outperforms state-of-the-art methods. 

---
# HISPASpoof: A New Dataset For Spanish Speech Forensics 

**Authors**: Maria Risques, Kratika Bhagtani, Amit Kumar Singh Yadav, Edward J. Delp  

**Link**: [PDF](https://arxiv.org/pdf/2509.09155)  

**Abstract**: Zero-shot Voice Cloning (VC) and Text-to-Speech (TTS) methods have advanced rapidly, enabling the generation of highly realistic synthetic speech and raising serious concerns about their misuse. While numerous detectors have been developed for English and Chinese, Spanish-spoken by over 600 million people worldwide-remains underrepresented in speech forensics. To address this gap, we introduce HISPASpoof, the first large-scale Spanish dataset designed for synthetic speech detection and attribution. It includes real speech from public corpora across six accents and synthetic speech generated with six zero-shot TTS systems. We evaluate five representative methods, showing that detectors trained on English fail to generalize to Spanish, while training on HISPASpoof substantially improves detection. We also evaluate synthetic speech attribution performance on HISPASpoof, i.e., identifying the generation method of synthetic speech. HISPASpoof thus provides a critical benchmark for advancing reliable and inclusive speech forensics in Spanish. 

---
# OCELOT 2023: Cell Detection from Cell-Tissue Interaction Challenge 

**Authors**: JaeWoong Shin, Jeongun Ryu, Aaron Valero Puche, Jinhee Lee, Biagio Brattoli, Wonkyung Jung, Soo Ick Cho, Kyunghyun Paeng, Chan-Young Ock, Donggeun Yoo, Zhaoyang Li, Wangkai Li, Huayu Mai, Joshua Millward, Zhen He, Aiden Nibali, Lydia Anette Schoenpflug, Viktor Hendrik Koelzer, Xu Shuoyu, Ji Zheng, Hu Bin, Yu-Wen Lo, Ching-Hui Yang, Sérgio Pereira  

**Link**: [PDF](https://arxiv.org/pdf/2509.09153)  

**Abstract**: Pathologists routinely alternate between different magnifications when examining Whole-Slide Images, allowing them to evaluate both broad tissue morphology and intricate cellular details to form comprehensive diagnoses. However, existing deep learning-based cell detection models struggle to replicate these behaviors and learn the interdependent semantics between structures at different magnifications. A key barrier in the field is the lack of datasets with multi-scale overlapping cell and tissue annotations. The OCELOT 2023 challenge was initiated to gather insights from the community to validate the hypothesis that understanding cell and tissue (cell-tissue) interactions is crucial for achieving human-level performance, and to accelerate the research in this field. The challenge dataset includes overlapping cell detection and tissue segmentation annotations from six organs, comprising 673 pairs sourced from 306 The Cancer Genome Atlas (TCGA) Whole-Slide Images with hematoxylin and eosin staining, divided into training, validation, and test subsets. Participants presented models that significantly enhanced the understanding of cell-tissue relationships. Top entries achieved up to a 7.99 increase in F1-score on the test set compared to the baseline cell-only model that did not incorporate cell-tissue relationships. This is a substantial improvement in performance over traditional cell-only detection methods, demonstrating the need for incorporating multi-scale semantics into the models. This paper provides a comparative analysis of the methods used by participants, highlighting innovative strategies implemented in the OCELOT 2023 challenge. 

---
# Video Understanding by Design: How Datasets Shape Architectures and Insights 

**Authors**: Lei Wang, Piotr Koniusz, Yongsheng Gao  

**Link**: [PDF](https://arxiv.org/pdf/2509.09151)  

**Abstract**: Video understanding has advanced rapidly, fueled by increasingly complex datasets and powerful architectures. Yet existing surveys largely classify models by task or family, overlooking the structural pressures through which datasets guide architectural evolution. This survey is the first to adopt a dataset-driven perspective, showing how motion complexity, temporal span, hierarchical composition, and multimodal richness impose inductive biases that models should encode. We reinterpret milestones, from two-stream and 3D CNNs to sequential, transformer, and multimodal foundation models, as concrete responses to these dataset-driven pressures. Building on this synthesis, we offer practical guidance for aligning model design with dataset invariances while balancing scalability and task demands. By unifying datasets, inductive biases, and architectures into a coherent framework, this survey provides both a comprehensive retrospective and a prescriptive roadmap for advancing general-purpose video understanding. 

---
# Objectness Similarity: Capturing Object-Level Fidelity in 3D Scene Evaluation 

**Authors**: Yuiko Uchida, Ren Togo, Keisuke Maeda, Takahiro Ogawa, Miki Haseyama  

**Link**: [PDF](https://arxiv.org/pdf/2509.09143)  

**Abstract**: This paper presents Objectness SIMilarity (OSIM), a novel evaluation metric for 3D scenes that explicitly focuses on "objects," which are fundamental units of human visual perception. Existing metrics assess overall image quality, leading to discrepancies with human perception. Inspired by neuropsychological insights, we hypothesize that human recognition of 3D scenes fundamentally involves attention to individual objects. OSIM enables object-centric evaluations by leveraging an object detection model and its feature representations to quantify the "objectness" of each object in the scene. Our user study demonstrates that OSIM aligns more closely with human perception compared to existing metrics. We also analyze the characteristics of OSIM using various approaches. Moreover, we re-evaluate recent 3D reconstruction and generation models under a standardized experimental setup to clarify advancements in this field. The code is available at this https URL. 

---
# ViRanker: A BGE-M3 & Blockwise Parallel Transformer Cross-Encoder for Vietnamese Reranking 

**Authors**: Phuong-Nam Dang, Kieu-Linh Nguyen, Thanh-Hieu Pham  

**Link**: [PDF](https://arxiv.org/pdf/2509.09131)  

**Abstract**: This paper presents ViRanker, a cross-encoder reranking model tailored to the Vietnamese language. Built on the BGE-M3 encoder and enhanced with the Blockwise Parallel Transformer, ViRanker addresses the lack of competitive rerankers for Vietnamese, a low-resource language with complex syntax and diacritics. The model was trained on an 8 GB curated corpus and fine-tuned with hybrid hard-negative sampling to strengthen robustness. Evaluated on the MMARCO-VI benchmark, ViRanker achieves strong early-rank accuracy, surpassing multilingual baselines and competing closely with PhoRanker. By releasing the model openly on Hugging Face, we aim to support reproducibility and encourage wider adoption in real-world retrieval systems. Beyond Vietnamese, this study illustrates how careful architectural adaptation and data curation can advance reranking in other underrepresented languages. 

---
# Automated Classification of Tutors' Dialogue Acts Using Generative AI: A Case Study Using the CIMA Corpus 

**Authors**: Liqun He, Jiaqi Xu  

**Link**: [PDF](https://arxiv.org/pdf/2509.09125)  

**Abstract**: This study explores the use of generative AI for automating the classification of tutors' Dialogue Acts (DAs), aiming to reduce the time and effort required by traditional manual coding. This case study uses the open-source CIMA corpus, in which tutors' responses are pre-annotated into four DA categories. Both GPT-3.5-turbo and GPT-4 models were tested using tailored prompts. Results show that GPT-4 achieved 80% accuracy, a weighted F1-score of 0.81, and a Cohen's Kappa of 0.74, surpassing baseline performance and indicating substantial agreement with human annotations. These findings suggest that generative AI has strong potential to provide an efficient and accessible approach to DA classification, with meaningful implications for educational dialogue analysis. The study also highlights the importance of task-specific label definitions and contextual information in enhancing the quality of automated annotation. Finally, it underscores the ethical considerations associated with the use of generative AI and the need for responsible and transparent research practices. The script of this research is publicly available at this https URL. 

---
# Character-Level Perturbations Disrupt LLM Watermarks 

**Authors**: Zhaoxi Zhang, Xiaomei Zhang, Yanjun Zhang, He Zhang, Shirui Pan, Bo Liu, Asif Qumer Gill, Leo Yu Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2509.09112)  

**Abstract**: Large Language Model (LLM) watermarking embeds detectable signals into generated text for copyright protection, misuse prevention, and content detection. While prior studies evaluate robustness using watermark removal attacks, these methods are often suboptimal, creating the misconception that effective removal requires large perturbations or powerful adversaries.
To bridge the gap, we first formalize the system model for LLM watermark, and characterize two realistic threat models constrained on limited access to the watermark detector. We then analyze how different types of perturbation vary in their attack range, i.e., the number of tokens they can affect with a single edit. We observe that character-level perturbations (e.g., typos, swaps, deletions, homoglyphs) can influence multiple tokens simultaneously by disrupting the tokenization process. We demonstrate that character-level perturbations are significantly more effective for watermark removal under the most restrictive threat model. We further propose guided removal attacks based on the Genetic Algorithm (GA) that uses a reference detector for optimization. Under a practical threat model with limited black-box queries to the watermark detector, our method demonstrates strong removal performance. Experiments confirm the superiority of character-level perturbations and the effectiveness of the GA in removing watermarks under realistic constraints. Additionally, we argue there is an adversarial dilemma when considering potential defenses: any fixed defense can be bypassed by a suitable perturbation strategy. Motivated by this principle, we propose an adaptive compound character-level attack. Experimental results show that this approach can effectively defeat the defenses. Our findings highlight significant vulnerabilities in existing LLM watermark schemes and underline the urgency for the development of new robust mechanisms. 

---
# DP-FedLoRA: Privacy-Enhanced Federated Fine-Tuning for On-Device Large Language Models 

**Authors**: Honghui Xu, Shiva Shrestha, Wei Chen, Zhiyuan Li, Zhipeng Cai  

**Link**: [PDF](https://arxiv.org/pdf/2509.09097)  

**Abstract**: As on-device large language model (LLM) systems become increasingly prevalent, federated fine-tuning enables advanced language understanding and generation directly on edge devices; however, it also involves processing sensitive, user-specific data, raising significant privacy concerns within the federated learning framework. To address these challenges, we propose DP-FedLoRA, a privacy-enhanced federated fine-tuning framework that integrates LoRA-based adaptation with differential privacy in a communication-efficient setting. Each client locally clips and perturbs its LoRA matrices using Gaussian noise to satisfy ($\epsilon$, $\delta$)-differential privacy. We further provide a theoretical analysis demonstrating the unbiased nature of the updates and deriving bounds on the variance introduced by noise, offering practical guidance for privacy-budget calibration. Experimental results across mainstream benchmarks show that DP-FedLoRA delivers competitive performance while offering strong privacy guarantees, paving the way for scalable and privacy-preserving LLM deployment in on-device environments. 

---
# Towards Confidential and Efficient LLM Inference with Dual Privacy Protection 

**Authors**: Honglan Yu, Yibin Wang, Feifei Dai, Dong Liu, Haihui Fan, Xiaoyan Gu  

**Link**: [PDF](https://arxiv.org/pdf/2509.09091)  

**Abstract**: CPU-based trusted execution environments (TEEs) and differential privacy (DP) have gained wide applications for private inference. Due to high inference latency in TEEs, researchers use partition-based approaches that offload linear model components to GPUs. However, dense nonlinear layers of large language models (LLMs) result in significant communication overhead between TEEs and GPUs. DP-based approaches apply random noise to protect data privacy, but this compromises LLM performance and semantic understanding. To overcome the above drawbacks, this paper proposes CMIF, a Confidential and efficient Model Inference Framework. CMIF confidentially deploys the embedding layer in the client-side TEE and subsequent layers on GPU servers. Meanwhile, it optimizes the Report-Noisy-Max mechanism to protect sensitive inputs with a slight decrease in model performance. Extensive experiments on Llama-series models demonstrate that CMIF reduces additional inference overhead in TEEs while preserving user data privacy. 

---
# SQAP-VLA: A Synergistic Quantization-Aware Pruning Framework for High-Performance Vision-Language-Action Models 

**Authors**: Hengyu Fang, Yijiang Liu, Yuan Du, Li Du, Huanrui Yang  

**Link**: [PDF](https://arxiv.org/pdf/2509.09090)  

**Abstract**: Vision-Language-Action (VLA) models exhibit unprecedented capabilities for embodied intelligence. However, their extensive computational and memory costs hinder their practical deployment. Existing VLA compression and acceleration approaches conduct quantization or token pruning in an ad-hoc manner but fail to enable both for a holistic efficiency improvement due to an observed incompatibility. This work introduces SQAP-VLA, the first structured, training-free VLA inference acceleration framework that simultaneously enables state-of-the-art quantization and token pruning. We overcome the incompatibility by co-designing the quantization and token pruning pipeline, where we propose new quantization-aware token pruning criteria that work on an aggressively quantized model while improving the quantizer design to enhance pruning effectiveness. When applied to standard VLA models, SQAP-VLA yields significant gains in computational efficiency and inference speed while successfully preserving core model performance, achieving a $\times$1.93 speedup and up to a 4.5\% average success rate enhancement compared to the original model. 

---
# KoopMotion: Learning Almost Divergence Free Koopman Flow Fields for Motion Planning 

**Authors**: Alice Kate Li, Thales C Silva, Victoria Edwards, Vijay Kumar, M. Ani Hsieh  

**Link**: [PDF](https://arxiv.org/pdf/2509.09074)  

**Abstract**: In this work, we propose a novel flow field-based motion planning method that drives a robot from any initial state to a desired reference trajectory such that it converges to the trajectory's end point. Despite demonstrated efficacy in using Koopman operator theory for modeling dynamical systems, Koopman does not inherently enforce convergence to desired trajectories nor to specified goals -- a requirement when learning from demonstrations (LfD). We present KoopMotion which represents motion flow fields as dynamical systems, parameterized by Koopman Operators to mimic desired trajectories, and leverages the divergence properties of the learnt flow fields to obtain smooth motion fields that converge to a desired reference trajectory when a robot is placed away from the desired trajectory, and tracks the trajectory until the end point. To demonstrate the effectiveness of our approach, we show evaluations of KoopMotion on the LASA human handwriting dataset and a 3D manipulator end-effector trajectory dataset, including spectral analysis. We also perform experiments on a physical robot, verifying KoopMotion on a miniature autonomous surface vehicle operating in a non-static fluid flow environment. Our approach is highly sample efficient in both space and time, requiring only 3\% of the LASA dataset to generate dense motion plans. Additionally, KoopMotion provides a significant improvement over baselines when comparing metrics that measure spatial and temporal dynamics modeling efficacy. 

---
# STRIDE: Scalable and Interpretable XAI via Subset-Free Functional Decomposition 

**Authors**: Chaeyun Ko  

**Link**: [PDF](https://arxiv.org/pdf/2509.09070)  

**Abstract**: Most explainable AI (XAI) frameworks face two practical limitations: the exponential cost of reasoning over feature subsets and the reduced expressiveness of summarizing effects as single scalar values. We present STRIDE, a scalable framework that aims to mitigate both issues by framing explanation as a subset-enumeration-free, orthogonal functional decomposition in a Reproducing Kernel Hilbert Space (RKHS). Rather than focusing only on scalar attributions, STRIDE computes functional components f_S(x_S) via an analytical projection scheme based on a recursive kernel-centering procedure, avoiding explicit subset enumeration. In the tabular setups we study, the approach is model-agnostic, provides both local and global views, and is supported by theoretical results on orthogonality and L^2 convergence under stated assumptions. On public tabular benchmarks in our environment, we observed speedups ranging from 0.6 times (slower than TreeSHAP on a small dataset) to 9.7 times (California), with a median approximate 3.0 times across 10 datasets, while maintaining high fidelity (R^2 between 0.81 and 0.999) and substantial rank agreement on most datasets. Overall, STRIDE complements scalar attribution methods by offering a structured functional perspective, enabling novel diagnostics like 'component surgery' to quantitatively measure the impact of specific interactions within our experimental scope. 

---
# Improving LLM Safety and Helpfulness using SFT and DPO: A Study on OPT-350M 

**Authors**: Piyush Pant  

**Link**: [PDF](https://arxiv.org/pdf/2509.09055)  

**Abstract**: This research investigates the effectiveness of alignment techniques, Supervised Fine-Tuning (SFT), Direct Preference Optimization (DPO), and a combined SFT+DPO approach on improving the safety and helpfulness of the OPT-350M language model. Utilizing the Anthropic Helpful-Harmless RLHF dataset, we train and evaluate four models: the base OPT350M, an SFT model, a DPO model, and a model trained with both SFT and DPO. We introduce three key evaluation metrics: Harmlessness Rate (HmR), Helpfulness Rate (HpR), and a Combined Alignment Score (CAS), all derived from reward model outputs. The results show that while SFT outperforms DPO, The combined SFT+DPO model outperforms all others across all metrics, demonstrating the complementary nature of these techniques. Our findings also highlight challenges posed by noisy data, limited GPU resources, and training constraints. This study offers a comprehensive view of how fine-tuning strategies affect model alignment and provides a foundation for more robust alignment pipelines in future work. 

---
# A Scoping Review of Machine Learning Applications in Power System Protection and Disturbance Management 

**Authors**: Julian Oelhaf, Georg Kordowich, Mehran Pashaei, Christian Bergler, Andreas Maier, Johann Jäger, Siming Bayer  

**Link**: [PDF](https://arxiv.org/pdf/2509.09053)  

**Abstract**: The integration of renewable and distributed energy resources reshapes modern power systems, challenging conventional protection schemes. This scoping review synthesizes recent literature on machine learning (ML) applications in power system protection and disturbance management, following the PRISMA for Scoping Reviews framework. Based on over 100 publications, three key objectives are addressed: (i) assessing the scope of ML research in protection tasks; (ii) evaluating ML performance across diverse operational scenarios; and (iii) identifying methods suitable for evolving grid conditions. ML models often demonstrate high accuracy on simulated datasets; however, their performance under real-world conditions remains insufficiently validated. The existing literature is fragmented, with inconsistencies in methodological rigor, dataset quality, and evaluation metrics. This lack of standardization hampers the comparability of results and limits the generalizability of findings. To address these challenges, this review introduces a ML-oriented taxonomy for protection tasks, resolves key terminological inconsistencies, and advocates for standardized reporting practices. It further provides guidelines for comprehensive dataset documentation, methodological transparency, and consistent evaluation protocols, aiming to improve reproducibility and enhance the practical relevance of research outcomes. Critical gaps remain, including the scarcity of real-world validation, insufficient robustness testing, and limited consideration of deployment feasibility. Future research should prioritize public benchmark datasets, realistic validation methods, and advanced ML architectures. These steps are essential to move ML-based protection from theoretical promise to practical deployment in increasingly dynamic and decentralized power systems. 

---
# MoWE : A Mixture of Weather Experts 

**Authors**: Dibyajyoti Chakraborty, Romit Maulik, Peter Harrington, Dallas Foster, Mohammad Amin Nabian, Sanjay Choudhry  

**Link**: [PDF](https://arxiv.org/pdf/2509.09052)  

**Abstract**: Data-driven weather models have recently achieved state-of-the-art performance, yet progress has plateaued in recent years. This paper introduces a Mixture of Experts (MoWE) approach as a novel paradigm to overcome these limitations, not by creating a new forecaster, but by optimally combining the outputs of existing models. The MoWE model is trained with significantly lower computational resources than the individual experts. Our model employs a Vision Transformer-based gating network that dynamically learns to weight the contributions of multiple "expert" models at each grid point, conditioned on forecast lead time. This approach creates a synthesized deterministic forecast that is more accurate than any individual component in terms of Root Mean Squared Error (RMSE). Our results demonstrate the effectiveness of this method, achieving up to a 10% lower RMSE than the best-performing AI weather model on a 2-day forecast horizon, significantly outperforming individual experts as well as a simple average across experts. This work presents a computationally efficient and scalable strategy to push the state of the art in data-driven weather prediction by making the most out of leading high-quality forecast models. 

---
# Stated Preference for Interaction and Continued Engagement (SPICE): Evaluating an LLM's Willingness to Re-engage in Conversation 

**Authors**: Thomas Manuel Rost, Martina Figlia, Bernd Wallraff  

**Link**: [PDF](https://arxiv.org/pdf/2509.09043)  

**Abstract**: We introduce and evaluate Stated Preference for Interaction and Continued Engagement (SPICE), a simple diagnostic signal elicited by asking a Large Language Model a YES or NO question about its willingness to re-engage with a user's behavior after reviewing a short transcript. In a study using a 3-tone (friendly, unclear, abusive) by 10-interaction stimulus set, we tested four open-weight chat models across four framing conditions, resulting in 480 trials. Our findings show that SPICE sharply discriminates by user tone. Friendly interactions yielded a near-unanimous preference to continue (97.5% YES), while abusive interactions yielded a strong preference to discontinue (17.9% YES), with unclear interactions falling in between (60.4% YES). This core association remains decisive under multiple dependence-aware statistical tests, including Rao-Scott adjustment and cluster permutation tests. Furthermore, we demonstrate that SPICE provides a distinct signal from abuse classification. In trials where a model failed to identify abuse, it still overwhelmingly stated a preference not to continue the interaction (81% of the time). An exploratory analysis also reveals a significant interaction effect: a preamble describing the study context significantly impacts SPICE under ambiguity, but only when transcripts are presented as a single block of text rather than a multi-turn chat. The results validate SPICE as a robust, low-overhead, and reproducible tool for auditing model dispositions, complementing existing metrics by offering a direct, relational signal of a model's state. All stimuli, code, and analysis scripts are released to support replication. 

---
# Envy-Free but Still Unfair: Envy-Freeness Up To One Item (EF-1) in Personalized Recommendation 

**Authors**: Amanda Aird, Ben Armstrong, Nicholas Mattei, Robin Burke  

**Link**: [PDF](https://arxiv.org/pdf/2509.09037)  

**Abstract**: Envy-freeness and the relaxation to Envy-freeness up to one item (EF-1) have been used as fairness concepts in the economics, game theory, and social choice literatures since the 1960s, and have recently gained popularity within the recommendation systems communities. In this short position paper we will give an overview of envy-freeness and its use in economics and recommendation systems; and illustrate why envy is not appropriate to measure fairness for use in settings where personalization plays a role. 

---
# Personalized Sleep Prediction via Deep Adaptive Spatiotemporal Modeling and Sparse Data 

**Authors**: Xueyi Wang, C. J. C., Lamoth, Elisabeth Wilhelm  

**Link**: [PDF](https://arxiv.org/pdf/2509.09018)  

**Abstract**: A sleep forecast allows individuals and healthcare providers to anticipate and proactively address factors influencing restful rest, ultimately improving mental and physical well-being. This work presents an adaptive spatial and temporal model (AdaST-Sleep) for predicting sleep scores. Our proposed model combines convolutional layers to capture spatial feature interactions between multiple features and recurrent neural network layers to handle longer-term temporal health-related data. A domain classifier is further integrated to generalize across different subjects. We conducted several experiments using five input window sizes (3, 5, 7, 9, 11 days) and five predicting window sizes (1, 3, 5, 7, 9 days). Our approach consistently outperformed four baseline models, achieving its lowest RMSE (0.282) with a seven-day input window and a one-day predicting window. Moreover, the method maintained strong performance even when forecasting multiple days into the future, demonstrating its versatility for real-world applications. Visual comparisons reveal that the model accurately tracks both the overall sleep score level and daily fluctuations. These findings prove that the proposed framework provides a robust and adaptable solution for personalized sleep forecasting using sparse data from commercial wearable devices and domain adaptation techniques. 

---
# Can Vision-Language Models Solve Visual Math Equations? 

**Authors**: Monjoy Narayan Choudhury, Junling Wang, Yifan Hou, Mrinmaya Sachan  

**Link**: [PDF](https://arxiv.org/pdf/2509.09013)  

**Abstract**: Despite strong performance in visual understanding and language-based reasoning, Vision-Language Models (VLMs) struggle with tasks requiring integrated perception and symbolic computation. We study this limitation through visual equation solving, where mathematical equations are embedded in images, variables are represented by object icons, and coefficients must be inferred by counting. While VLMs perform well on textual equations, they fail on visually grounded counterparts. To understand this gap, we decompose the task into coefficient counting and variable recognition, and find that counting is the primary bottleneck, even when recognition is accurate. We also observe that composing recognition and reasoning introduces additional errors, highlighting challenges in multi-step visual reasoning. Finally, as equation complexity increases, symbolic reasoning itself becomes a limiting factor. These findings reveal key weaknesses in current VLMs and point toward future improvements in visually grounded mathematical reasoning. 

---
# Open-sci-ref-0.01: open and reproducible reference baselines for language model and dataset comparison 

**Authors**: Marianna Nezhurina, Taishi Nakamura, Timur Carstensen, Niccolò Ajroldi, Ville Komulainen, David Salinas, Jenia Jitsev  

**Link**: [PDF](https://arxiv.org/pdf/2509.09009)  

**Abstract**: We introduce open-sci-ref, a family of dense transformer models trained as research baselines across multiple model (0.13B to 1.7B parameters) and token scales (up to 1T) on 8 recent open reference datasets. Evaluating the models on various standardized benchmarks, our training runs set establishes reference points that enable researchers to assess the sanity and quality of alternative training approaches across scales and datasets. Intermediate checkpoints allow comparison and studying of the training dynamics. The established reference baselines allow training procedures to be compared through their scaling trends, aligning them on a common compute axis. Comparison of open reference datasets reveals that training on NemoTron-CC HQ consistently outperforms other reference datasets, followed by DCLM-baseline and FineWeb-Edu. In addition to intermediate training checkpoints, the release includes logs, code, and downstream evaluations to simplify reproduction, standardize comparison, and facilitate future research. 

---
# Implicit Neural Representations of Intramyocardial Motion and Strain 

**Authors**: Andrew Bell, Yan Kit Choi, Steffen Peterson, Andrew King, Muhummad Sohaib Nazir, Alistair Young  

**Link**: [PDF](https://arxiv.org/pdf/2509.09004)  

**Abstract**: Automatic quantification of intramyocardial motion and strain from tagging MRI remains an important but challenging task. We propose a method using implicit neural representations (INRs), conditioned on learned latent codes, to predict continuous left ventricular (LV) displacement -- without requiring inference-time optimisation. Evaluated on 452 UK Biobank test cases, our method achieved the best tracking accuracy (2.14 mm RMSE) and the lowest combined error in global circumferential (2.86%) and radial (6.42%) strain compared to three deep learning baselines. In addition, our method is $\sim$380$\times$ faster than the most accurate baseline. These results highlight the suitability of INR-based models for accurate and scalable analysis of myocardial strain in large CMR datasets. 

---
# Similarity-based Outlier Detection for Noisy Object Re-Identification Using Beta Mixtures 

**Authors**: Waqar Ahmad, Evan Murphy, Vladimir A. Krylov  

**Link**: [PDF](https://arxiv.org/pdf/2509.08926)  

**Abstract**: Object re-identification (Re-ID) methods are highly sensitive to label noise, which typically leads to significant performance degradation. We address this challenge by reframing Re-ID as a supervised image similarity task and adopting a Siamese network architecture trained to capture discriminative pairwise relationships. Central to our approach is a novel statistical outlier detection (OD) framework, termed Beta-SOD (Beta mixture Similarity-based Outlier Detection), which models the distribution of cosine similarities between embedding pairs using a two-component Beta distribution mixture model. We establish a novel identifiability result for mixtures of two Beta distributions, ensuring that our learning task is this http URL proposed OD step complements the Re-ID architecture combining binary cross-entropy, contrastive, and cosine embedding losses that jointly optimize feature-level similarity this http URL demonstrate the effectiveness of Beta-SOD in de-noising and Re-ID tasks for person Re-ID, on CUHK03 and Market-1501 datasets, and vehicle Re-ID, on VeRi-776 dataset. Our method shows superior performance compared to the state-of-the-art methods across various noise levels (10-30\%), demonstrating both robustness and broad applicability in noisy Re-ID scenarios. The implementation of Beta-SOD is available at: this https URL 

---
# Instance-Optimal Matrix Multiplicative Weight Update and Its Quantum Applications 

**Authors**: Weiyuan Gong, Tongyang Li, Xinzhao Wang, Zhiyu Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2509.08911)  

**Abstract**: The Matrix Multiplicative Weight Update (MMWU) is a seminal online learning algorithm with numerous applications. Applied to the matrix version of the Learning from Expert Advice (LEA) problem on the $d$-dimensional spectraplex, it is well known that MMWU achieves the minimax-optimal regret bound of $O(\sqrt{T\log d})$, where $T$ is the time horizon. In this paper, we present an improved algorithm achieving the instance-optimal regret bound of $O(\sqrt{T\cdot S(X||d^{-1}I_d)})$, where $X$ is the comparator in the regret, $I_d$ is the identity matrix, and $S(\cdot||\cdot)$ denotes the quantum relative entropy. Furthermore, our algorithm has the same computational complexity as MMWU, indicating that the improvement in the regret bound is ``free''.
Technically, we first develop a general potential-based framework for matrix LEA, with MMWU being its special case induced by the standard exponential potential. Then, the crux of our analysis is a new ``one-sided'' Jensen's trace inequality built on a Laplace transform technique, which allows the application of general potential functions beyond exponential to matrix LEA. Our algorithm is finally induced by an optimal potential function from the vector LEA problem, based on the imaginary error function.
Complementing the above, we provide a memory lower bound for matrix LEA, and explore the applications of our algorithm in quantum learning theory. We show that it outperforms the state of the art for learning quantum states corrupted by depolarization noise, random quantum states, and Gibbs states. In addition, applying our algorithm to linearized convex losses enables predicting nonlinear quantum properties, such as purity, quantum virtual cooling, and Rényi-$2$ correlation. 

---
# PromptGuard: An Orchestrated Prompting Framework for Principled Synthetic Text Generation for Vulnerable Populations using LLMs with Enhanced Safety, Fairness, and Controllability 

**Authors**: Tung Vu, Lam Nguyen, Quynh Dao  

**Link**: [PDF](https://arxiv.org/pdf/2509.08910)  

**Abstract**: The proliferation of Large Language Models (LLMs) in real-world applications poses unprecedented risks of generating harmful, biased, or misleading information to vulnerable populations including LGBTQ+ individuals, single parents, and marginalized communities. While existing safety approaches rely on post-hoc filtering or generic alignment techniques, they fail to proactively prevent harmful outputs at the generation source. This paper introduces PromptGuard, a novel modular prompting framework with our breakthrough contribution: VulnGuard Prompt, a hybrid technique that prevents harmful information generation using real-world data-driven contrastive learning. VulnGuard integrates few-shot examples from curated GitHub repositories, ethical chain-of-thought reasoning, and adaptive role-prompting to create population-specific protective barriers. Our framework employs theoretical multi-objective optimization with formal proofs demonstrating 25-30% analytical harm reduction through entropy bounds and Pareto optimality. PromptGuard orchestrates six core modules: Input Classification, VulnGuard Prompting, Ethical Principles Integration, External Tool Interaction, Output Validation, and User-System Interaction, creating an intelligent expert system for real-time harm prevention. We provide comprehensive mathematical formalization including convergence proofs, vulnerability analysis using information theory, and theoretical validation framework using GitHub-sourced datasets, establishing mathematical foundations for systematic empirical research. 

---
# Recurrence Meets Transformers for Universal Multimodal Retrieval 

**Authors**: Davide Caffagni, Sara Sarto, Marcella Cornia, Lorenzo Baraldi, Rita Cucchiara  

**Link**: [PDF](https://arxiv.org/pdf/2509.08897)  

**Abstract**: With the rapid advancement of multimodal retrieval and its application in LLMs and multimodal LLMs, increasingly complex retrieval tasks have emerged. Existing methods predominantly rely on task-specific fine-tuning of vision-language models and are limited to single-modality queries or documents. In this paper, we propose ReT-2, a unified retrieval model that supports multimodal queries, composed of both images and text, and searches across multimodal document collections where text and images coexist. ReT-2 leverages multi-layer representations and a recurrent Transformer architecture with LSTM-inspired gating mechanisms to dynamically integrate information across layers and modalities, capturing fine-grained visual and textual details. We evaluate ReT-2 on the challenging M2KR and M-BEIR benchmarks across different retrieval configurations. Results demonstrate that ReT-2 consistently achieves state-of-the-art performance across diverse settings, while offering faster inference and reduced memory usage compared to prior approaches. When integrated into retrieval-augmented generation pipelines, ReT-2 also improves downstream performance on Encyclopedic-VQA and InfoSeek datasets. Our source code and trained models are publicly available at: this https URL 

---
# Benchmarking Energy Efficiency of Large Language Models Using vLLM 

**Authors**: K. Pronk, Q. Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2509.08867)  

**Abstract**: The prevalence of Large Language Models (LLMs) is having an growing impact on the climate due to the substantial energy required for their deployment and use. To create awareness for developers who are implementing LLMs in their products, there is a strong need to collect more information about the energy efficiency of LLMs. While existing research has evaluated the energy efficiency of various models, these benchmarks often fall short of representing realistic production scenarios. In this paper, we introduce the LLM Efficiency Benchmark, designed to simulate real-world usage conditions. Our benchmark utilizes vLLM, a high-throughput, production-ready LLM serving backend that optimizes model performance and efficiency. We examine how factors such as model size, architecture, and concurrent request volume affect inference energy efficiency. Our findings demonstrate that it is possible to create energy efficiency benchmarks that better reflect practical deployment conditions, providing valuable insights for developers aiming to build more sustainable AI systems. 

---
# Investigating Student Interaction Patterns with Large Language Model-Powered Course Assistants in Computer Science Courses 

**Authors**: Chang Liu, Loc Hoang, Andrew Stolman, Rene F. Kizilcec, Bo Wu  

**Link**: [PDF](https://arxiv.org/pdf/2509.08862)  

**Abstract**: Providing students with flexible and timely academic support is a challenge at most colleges and universities, leaving many students without help outside scheduled hours. Large language models (LLMs) are promising for bridging this gap, but interactions between students and LLMs are rarely overseen by educators. We developed and studied an LLM-powered course assistant deployed across multiple computer science courses to characterize real-world use and understand pedagogical implications. By Spring 2024, our system had been deployed to approximately 2,000 students across six courses at three institutions. Analysis of the interaction data shows that usage remains strong in the evenings and nights and is higher in introductory courses, indicating that our system helps address temporal support gaps and novice learner needs. We sampled 200 conversations per course for manual annotation: most sampled responses were judged correct and helpful, with a small share unhelpful or erroneous; few responses included dedicated examples. We also examined an inquiry-based learning strategy: only around 11% of sampled conversations contained LLM-generated follow-up questions, which were often ignored by students in advanced courses. A Bloom's taxonomy analysis reveals that current LLM capabilities are limited in generating higher-order cognitive questions. These patterns suggest opportunities for pedagogically oriented LLM-based educational systems and greater educator involvement in configuring prompts, content, and policies. 

---
# Multi Robot Coordination in Highly Dynamic Environments: Tackling Asymmetric Obstacles and Limited Communication 

**Authors**: Vincenzo Suriani, Daniele Affinita, Domenico D. Bloisi, Daniele Nardi  

**Link**: [PDF](https://arxiv.org/pdf/2509.08859)  

**Abstract**: Coordinating a fully distributed multi-agent system (MAS) can be challenging when the communication channel has very limited capabilities in terms of sending rate and packet payload. When the MAS has to deal with active obstacles in a highly partially observable environment, the communication channel acquires considerable relevance. In this paper, we present an approach to deal with task assignments in extremely active scenarios, where tasks need to be frequently reallocated among the agents participating in the coordination process. Inspired by market-based task assignments, we introduce a novel distributed coordination method to orchestrate autonomous agents' actions efficiently in low communication scenarios. In particular, our algorithm takes into account asymmetric obstacles. While in the real world, the majority of obstacles are asymmetric, they are usually treated as symmetric ones, thus limiting the applicability of existing methods. To summarize, the presented architecture is designed to tackle scenarios where the obstacles are active and asymmetric, the communication channel is poor and the environment is partially observable. Our approach has been validated in simulation and in the real world, using a team of NAO robots during official RoboCup competitions. Experimental results show a notable reduction in task overlaps in limited communication settings, with a decrease of 52% in the most frequent reallocated task. 

---
# A vibe coding learning design to enhance EFL students' talking to, through, and about AI 

**Authors**: David James Woo, Kai Guo, Yangyang Yu  

**Link**: [PDF](https://arxiv.org/pdf/2509.08854)  

**Abstract**: This innovative practice article reports on the piloting of vibe coding (using natural language to create software applications with AI) for English as a Foreign Language (EFL) education. We developed a human-AI meta-languaging framework with three dimensions: talking to AI (prompt engineering), talking through AI (negotiating authorship), and talking about AI (mental models of AI). Using backward design principles, we created a four-hour workshop where two students designed applications addressing authentic EFL writing challenges. We adopted a case study methodology, collecting data from worksheets and video recordings, think-aloud protocols, screen recordings, and AI-generated images. Contrasting cases showed one student successfully vibe coding a functional application cohering to her intended design, while another encountered technical difficulties with major gaps between intended design and actual functionality. Analysis reveals differences in students' prompt engineering approaches, suggesting different AI mental models and tensions in attributing authorship. We argue that AI functions as a beneficial languaging machine, and that differences in how students talk to, through, and about AI explain vibe coding outcome variations. Findings indicate that effective vibe coding instruction requires explicit meta-languaging scaffolding, teaching structured prompt engineering, facilitating critical authorship discussions, and developing vocabulary for articulating AI mental models. 

---
# Safe and Certifiable AI Systems: Concepts, Challenges, and Lessons Learned 

**Authors**: Kajetan Schweighofer, Barbara Brune, Lukas Gruber, Simon Schmid, Alexander Aufreiter, Andreas Gruber, Thomas Doms, Sebastian Eder, Florian Mayer, Xaver-Paul Stadlbauer, Christoph Schwald, Werner Zellinger, Bernhard Nessler, Sepp Hochreiter  

**Link**: [PDF](https://arxiv.org/pdf/2509.08852)  

**Abstract**: There is an increasing adoption of artificial intelligence in safety-critical applications, yet practical schemes for certifying that AI systems are safe, lawful and socially acceptable remain scarce. This white paper presents the TÜV AUSTRIA Trusted AI framework an end-to-end audit catalog and methodology for assessing and certifying machine learning systems. The audit catalog has been in continuous development since 2019 in an ongoing collaboration with scientific partners. Building on three pillars - Secure Software Development, Functional Requirements, and Ethics & Data Privacy - the catalog translates the high-level obligations of the EU AI Act into specific, testable criteria. Its core concept of functional trustworthiness couples a statistically defined application domain with risk-based minimum performance requirements and statistical testing on independently sampled data, providing transparent and reproducible evidence of model quality in real-world settings. We provide an overview of the functional requirements that we assess, which are oriented on the lifecycle of an AI system. In addition, we share some lessons learned from the practical application of the audit catalog, highlighting common pitfalls we encountered, such as data leakage scenarios, inadequate domain definitions, neglect of biases, or a lack of distribution drift controls. We further discuss key aspects of certifying AI systems, such as robustness, algorithmic fairness, or post-certification requirements, outlining both our current conclusions and a roadmap for future research. In general, by aligning technical best practices with emerging European standards, the approach offers regulators, providers, and users a practical roadmap for legally compliant, functionally trustworthy, and certifiable AI systems. 

---
# Uncertainty Estimation using Variance-Gated Distributions 

**Authors**: H. Martin Gillis, Isaac Xu, Thomas Trappenberg  

**Link**: [PDF](https://arxiv.org/pdf/2509.08846)  

**Abstract**: Evaluation of per-sample uncertainty quantification from neural networks is essential for decision-making involving high-risk applications. A common approach is to use the predictive distribution from Bayesian or approximation models and decompose the corresponding predictive uncertainty into epistemic (model-related) and aleatoric (data-related) components. However, additive decomposition has recently been questioned. In this work, we propose an intuitive framework for uncertainty estimation and decomposition based on the signal-to-noise ratio of class probability distributions across different model predictions. We introduce a variance-gated measure that scales predictions by a confidence factor derived from ensembles. We use this measure to discuss the existence of a collapse in the diversity of committee machines. 

---
# Deep opacity and AI: A threat to XAI and to privacy protection mechanisms 

**Authors**: Vincent C. Müller  

**Link**: [PDF](https://arxiv.org/pdf/2509.08835)  

**Abstract**: It is known that big data analytics and AI pose a threat to privacy, and that some of this is due to some kind of "black box problem" in AI. I explain how this becomes a problem in the context of justification for judgments and actions. Furthermore, I suggest distinguishing three kinds of opacity: 1) the subjects do not know what the system does ("shallow opacity"), 2) the analysts do not know what the system does ("standard black box opacity"), or 3) the analysts cannot possibly know what the system might do ("deep opacity"). If the agents, data subjects as well as analytics experts, operate under opacity, then these agents cannot provide justifications for judgments that are necessary to protect privacy, e.g., they cannot give "informed consent", or guarantee "anonymity". It follows from these points that agents in big data analytics and AI often cannot make the judgments needed to protect privacy. So I conclude that big data analytics makes the privacy problems worse and the remedies less effective. As a positive note, I provide a brief outlook on technical ways to handle this situation. 

---
# PerFairX: Is There a Balance Between Fairness and Personality in Large Language Model Recommendations? 

**Authors**: Chandan Kumar Sah  

**Link**: [PDF](https://arxiv.org/pdf/2509.08829)  

**Abstract**: The integration of Large Language Models (LLMs) into recommender systems has enabled zero-shot, personality-based personalization through prompt-based interactions, offering a new paradigm for user-centric recommendations. However, incorporating user personality traits via the OCEAN model highlights a critical tension between achieving psychological alignment and ensuring demographic fairness. To address this, we propose PerFairX, a unified evaluation framework designed to quantify the trade-offs between personalization and demographic equity in LLM-generated recommendations. Using neutral and personality-sensitive prompts across diverse user profiles, we benchmark two state-of-the-art LLMs, ChatGPT and DeepSeek, on movie (MovieLens 10M) and music (this http URL 360K) datasets. Our results reveal that personality-aware prompting significantly improves alignment with individual traits but can exacerbate fairness disparities across demographic groups. Specifically, DeepSeek achieves stronger psychological fit but exhibits higher sensitivity to prompt variations, while ChatGPT delivers stable yet less personalized outputs. PerFairX provides a principled benchmark to guide the development of LLM-based recommender systems that are both equitable and psychologically informed, contributing to the creation of inclusive, user-centric AI applications in continual learning contexts. 

---
