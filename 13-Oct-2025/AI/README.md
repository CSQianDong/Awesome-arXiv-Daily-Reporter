# LiveOIBench: Can Large Language Models Outperform Human Contestants in Informatics Olympiads? 

**Authors**: Kaijian Zou, Aaron Xiong, Yunxiang Zhang, Frederick Zhang, Yueqi Ren, Jirong Yang, Ayoung Lee, Shitanshu Bhushan, Lu Wang  

**Link**: [PDF](https://arxiv.org/pdf/2510.09595)  

**Abstract**: Competitive programming problems increasingly serve as valuable benchmarks to evaluate the coding capabilities of large language models (LLMs) due to their complexity and ease of verification. Yet, current coding benchmarks face limitations such as lack of exceptionally challenging problems, insufficient test case coverage, reliance on online platform APIs that limit accessibility. To address these issues, we introduce LiveOIBench, a comprehensive benchmark featuring 403 expert-curated Olympiad-level competitive programming problems, each with an average of 60 expert-designed test cases. The problems are sourced directly from 72 official Informatics Olympiads in different regions conducted between 2023 and 2025. LiveOIBench distinguishes itself through four key features: (1) meticulously curated high-quality tasks with detailed subtask rubrics and extensive private test cases; (2) direct integration of elite contestant performance data to enable informative comparison against top-performing humans; (3) planned continuous, contamination-free updates from newly released Olympiad problems; and (4) a self-contained evaluation system facilitating offline and easy-to-reproduce assessments. Benchmarking 32 popular general-purpose and reasoning LLMs, we find that GPT-5 achieves a notable 81.76th percentile, a strong result that nonetheless falls short of top human contestant performance, who usually place above 90th. In contrast, among open-weight reasoning models, GPT-OSS-120B achieves only a 60th percentile, underscoring significant capability disparities from frontier closed models. Detailed analyses indicate that robust reasoning models prioritize precise problem analysis over excessive exploration, suggesting future models should emphasize structured analysis and minimize unnecessary exploration. All data, code, and leaderboard results will be made publicly available on our website. 

---
# GraphMERT: Efficient and Scalable Distillation of Reliable Knowledge Graphs from Unstructured Data 

**Authors**: Margarita Belova, Jiaxin Xiao, Shikhar Tuli, Niraj K. Jha  

**Link**: [PDF](https://arxiv.org/pdf/2510.09580)  

**Abstract**: Researchers have pursued neurosymbolic artificial intelligence (AI) applications for nearly three decades because symbolic components provide abstraction while neural components provide generalization. Thus, a marriage of the two components can lead to rapid advancements in AI. Yet, the field has not realized this promise since most neurosymbolic AI frameworks fail to scale. In addition, the implicit representations and approximate reasoning of neural approaches limit interpretability and trust. Knowledge graphs (KGs), a gold-standard representation of explicit semantic knowledge, can address the symbolic side. However, automatically deriving reliable KGs from text corpora has remained an open problem. We address these challenges by introducing GraphMERT, a tiny graphical encoder-only model that distills high-quality KGs from unstructured text corpora and its own internal representations. GraphMERT and its equivalent KG form a modular neurosymbolic stack: neural learning of abstractions; symbolic KGs for verifiable reasoning. GraphMERT + KG is the first efficient and scalable neurosymbolic model to achieve state-of-the-art benchmark accuracy along with superior symbolic representations relative to baselines.
Concretely, we target reliable domain-specific KGs that are both (1) factual (with provenance) and (2) valid (ontology-consistent relations with domain-appropriate semantics). When a large language model (LLM), e.g., Qwen3-32B, generates domain-specific KGs, it falls short on reliability due to prompt sensitivity, shallow domain expertise, and hallucinated relations. On text obtained from PubMed papers on diabetes, our 80M-parameter GraphMERT yields a KG with a 69.8% FActScore; a 32B-parameter baseline LLM yields a KG that achieves only 40.2% FActScore. The GraphMERT KG also attains a higher ValidityScore of 68.8%, versus 43.0% for the LLM baseline. 

---
# Safe, Untrusted, "Proof-Carrying" AI Agents: toward the agentic lakehouse 

**Authors**: Jacopo Tagliabue, Ciro Greco  

**Link**: [PDF](https://arxiv.org/pdf/2510.09567)  

**Abstract**: Data lakehouses run sensitive workloads, where AI-driven automation raises concerns about trust, correctness, and governance. We argue that API-first, programmable lakehouses provide the right abstractions for safe-by-design, agentic workflows. Using Bauplan as a case study, we show how data branching and declarative environments extend naturally to agents, enabling reproducibility and observability while reducing the attack surface. We present a proof-of-concept in which agents repair data pipelines using correctness checks inspired by proof-carrying code. Our prototype demonstrates that untrusted AI agents can operate safely on production data and outlines a path toward a fully agentic lakehouse. 

---
# Titans Revisited: A Lightweight Reimplementation and Critical Analysis of a Test-Time Memory Model 

**Authors**: Gavriel Di Nepi, Federico Siciliano, Fabrizio Silvestri  

**Link**: [PDF](https://arxiv.org/pdf/2510.09551)  

**Abstract**: By the end of 2024, Google researchers introduced Titans: Learning at Test Time, a neural memory model achieving strong empirical results across multiple tasks. However, the lack of publicly available code and ambiguities in the original description hinder reproducibility. In this work, we present a lightweight reimplementation of Titans and conduct a comprehensive evaluation on Masked Language Modeling, Time Series Forecasting, and Recommendation tasks. Our results reveal that Titans does not always outperform established baselines due to chunking. However, its Neural Memory component consistently improves performance compared to attention-only models. These findings confirm the model's innovative potential while highlighting its practical limitations and raising questions for future research. 

---
# Agentic Systems in Radiology: Design, Applications, Evaluation, and Challenges 

**Authors**: Christian Bluethgen, Dave Van Veen, Daniel Truhn, Jakob Nikolas Kather, Michael Moor, Malgorzata Polacin, Akshay Chaudhari, Thomas Frauenfelder, Curtis P. Langlotz, Michael Krauthammer, Farhad Nooralahzadeh  

**Link**: [PDF](https://arxiv.org/pdf/2510.09404)  

**Abstract**: Building agents, systems that perceive and act upon their environment with a degree of autonomy, has long been a focus of AI research. This pursuit has recently become vastly more practical with the emergence of large language models (LLMs) capable of using natural language to integrate information, follow instructions, and perform forms of "reasoning" and planning across a wide range of tasks. With its multimodal data streams and orchestrated workflows spanning multiple systems, radiology is uniquely suited to benefit from agents that can adapt to context and automate repetitive yet complex tasks. In radiology, LLMs and their multimodal variants have already demonstrated promising performance for individual tasks such as information extraction and report summarization. However, using LLMs in isolation underutilizes their potential to support complex, multi-step workflows where decisions depend on evolving context from multiple information sources. Equipping LLMs with external tools and feedback mechanisms enables them to drive systems that exhibit a spectrum of autonomy, ranging from semi-automated workflows to more adaptive agents capable of managing complex processes. This review examines the design of such LLM-driven agentic systems, highlights key applications, discusses evaluation methods for planning and tool use, and outlines challenges such as error cascades, tool-use efficiency, and health IT integration. 

---
# Sequence Variables: A Constraint Programming Computational Domain for Routing and Sequencing 

**Authors**: Augustin Delecluse, Pierre Schaus, Pascal Van Hentenryck  

**Link**: [PDF](https://arxiv.org/pdf/2510.09373)  

**Abstract**: Constraint Programming (CP) offers an intuitive, declarative framework for modeling Vehicle Routing Problems (VRP), yet classical CP models based on successor variables cannot always deal with optional visits or insertion based heuristics. To address these limitations, this paper formalizes sequence variables within CP. Unlike the classical successor models, this computational domain handle optional visits and support insertion heuristics, including insertion-based Large Neighborhood Search. We provide a clear definition of their domain, update operations, and introduce consistency levels for constraints on this domain. An implementation is described with the underlying data structures required for integrating sequence variables into existing trail-based CP solvers. Furthermore, global constraints specifically designed for sequence variables and vehicle routing are introduced. Finally, the effectiveness of sequence variables is demonstrated by simplifying problem modeling and achieving competitive computational performance on the Dial-a-Ride Problem. 

---
# Toward Mechanistic Explanation of Deductive Reasoning in Language Models 

**Authors**: Davide Maltoni, Matteo Ferrara  

**Link**: [PDF](https://arxiv.org/pdf/2510.09340)  

**Abstract**: Recent large language models have demonstrated relevant capabilities in solving problems that require logical reasoning; however, the corresponding internal mechanisms remain largely unexplored. In this paper, we show that a small language model can solve a deductive reasoning task by learning the underlying rules (rather than operating as a statistical learner). A low-level explanation of its internal representations and computational circuits is then provided. Our findings reveal that induction heads play a central role in the implementation of the rule completion and rule chaining steps involved in the logical inference required by the task. 

---
# Localist LLMs -- A Mathematical Framework for Dynamic Locality Control 

**Authors**: Joachim Diederich  

**Link**: [PDF](https://arxiv.org/pdf/2510.09338)  

**Abstract**: We present a novel framework for training large language models with continuously adjustable internal representations that span the full spectrum from localist (interpretable, rule-based) to distributed (generalizable, efficient) encodings. The key innovation is a locality dial, a tunable parameter that dynamically controls the degree of localization during both training and inference without requiring model retraining. This is achieved through group sparsity penalties on attention mechanisms, information-theoretic anchor design, and dynamic rule injection. We provide rigorous mathematical proofs establishing explicit threshold conditions under which attention provably concentrates on semantically relevant blocks, with exponential bounds on attention entropy and pointer fidelity. Specifically, we prove that when group sparsity penalties exceed certain threshold values, the model's attention mechanisms concentrate on semantically relevant blocks, achieving low entropy and high fidelity with negligible error. This framework enables practitioners to continuously interpolate between interpretable and high-performance modes, supporting applications in regulated domains requiring both transparency and capability. 

---
# Fundamentals of Building Autonomous LLM Agents 

**Authors**: Victor de Lamo Castrillo, Habtom Kahsay Gidey, Alexander Lenz, Alois Knoll  

**Link**: [PDF](https://arxiv.org/pdf/2510.09244)  

**Abstract**: This paper reviews the architecture and implementation methods of agents powered by large language models (LLMs). Motivated by the limitations of traditional LLMs in real-world tasks, the research aims to explore patterns to develop "agentic" LLMs that can automate complex tasks and bridge the performance gap with human capabilities. Key components include a perception system that converts environmental percepts into meaningful representations; a reasoning system that formulates plans, adapts to feedback, and evaluates actions through different techniques like Chain-of-Thought and Tree-of-Thought; a memory system that retains knowledge through both short-term and long-term mechanisms; and an execution system that translates internal decisions into concrete actions. This paper shows how integrating these systems leads to more capable and generalized software bots that mimic human cognitive processes for autonomous and intelligent behavior. 

---
# RegexPSPACE: A Benchmark for Evaluating LLM Reasoning on PSPACE-complete Regex Problems 

**Authors**: Hyundong Jin, Joonghyuk Hahn, Yo-Sub Han  

**Link**: [PDF](https://arxiv.org/pdf/2510.09227)  

**Abstract**: Large language models (LLMs) show strong performance across natural language processing (NLP), mathematical reasoning, and programming, and recent large reasoning models (LRMs) further emphasize explicit reasoning. Yet their computational limits, particularly spatial complexity constrained by finite context windows, remain poorly understood. While recent works often focus on problems within the NP complexity class, we push the boundary by introducing a novel benchmark grounded in two PSPACE-complete regular expression (regex) problems: equivalence decision (RegexEQ) and minimization (RegexMin). PSPACE-complete problems serve as a more rigorous standard for assessing computational capacity, as their solutions require massive search space exploration. We perform a double-exponential space exploration to construct a labeled dataset of over a million regex instances with a sound filtering process to build the benchmark. We conduct extensive evaluations on 6 LLMs and 5 LRMs of varying scales, revealing common failure patterns such as verbosity and repetition. With its well-defined structure and quantitative evaluation metrics, this work presents the first empirical investigation into the spatial computational limitations of LLMs and LRMs, offering a new framework for evaluating their advanced reasoning capabilities. Our code is available at this https URL . 

---
# Comparing Knowledge Source Integration Methods for Optimizing Healthcare Knowledge Fusion in Rescue Operation 

**Authors**: Mubaris Nadeem, Madjid Fathi  

**Link**: [PDF](https://arxiv.org/pdf/2510.09223)  

**Abstract**: In the field of medicine and healthcare, the utilization of medical expertise, based on medical knowledge combined with patients' health information is a life-critical challenge for patients and health professionals. The within-laying complexity and variety form the need for a united approach to gather, analyze, and utilize existing knowledge of medical treatments, and medical operations to provide the ability to present knowledge for the means of accurate patient-driven decision-making. One way to achieve this is the fusion of multiple knowledge sources in healthcare. It provides health professionals the opportunity to select from multiple contextual aligned knowledge sources which enables the support for critical decisions. This paper presents multiple conceptual models for knowledge fusion in the field of medicine, based on a knowledge graph structure. It will evaluate, how knowledge fusion can be enabled and presents how to integrate various knowledge sources into the knowledge graph for rescue operations. 

---
# Dr. Bias: Social Disparities in AI-Powered Medical Guidance 

**Authors**: Emma Kondrup, Anne Imouza  

**Link**: [PDF](https://arxiv.org/pdf/2510.09162)  

**Abstract**: With the rapid progress of Large Language Models (LLMs), the general public now has easy and affordable access to applications capable of answering most health-related questions in a personalized manner. These LLMs are increasingly proving to be competitive, and now even surpass professionals in some medical capabilities. They hold particular promise in low-resource settings, considering they provide the possibility of widely accessible, quasi-free healthcare support. However, evaluations that fuel these motivations highly lack insights into the social nature of healthcare, oblivious to health disparities between social groups and to how bias may translate into LLM-generated medical advice and impact users. We provide an exploratory analysis of LLM answers to a series of medical questions spanning key clinical domains, where we simulate these questions being asked by several patient profiles that vary in sex, age range, and ethnicity. By comparing natural language features of the generated responses, we show that, when LLMs are used for medical advice generation, they generate responses that systematically differ between social groups. In particular, Indigenous and intersex patients receive advice that is less readable and more complex. We observe these trends amplify when intersectional groups are considered. Considering the increasing trust individuals place in these models, we argue for higher AI literacy and for the urgent need for investigation and mitigation by AI developers to ensure these systemic differences are diminished and do not translate to unjust patient support. Our code is publicly available on GitHub. 

---
# PAC Reasoning: Controlling the Performance Loss for Efficient Reasoning 

**Authors**: Hao Zeng, Jianguo Huang, Bingyi Jing, Hongxin Wei, Bo An  

**Link**: [PDF](https://arxiv.org/pdf/2510.09133)  

**Abstract**: Large reasoning models (LRMs) have achieved remarkable progress in complex problem-solving tasks. Despite this success, LRMs typically suffer from high computational costs during deployment, highlighting a need for efficient inference. A popular direction of efficiency improvement is to switch the LRM between thinking and nonthinking modes dynamically. However, such approaches often introduce additional reasoning errors and lack statistical guarantees for the performance loss, which are critical for high-stakes applications. In this work, we propose Probably Approximately Correct (PAC) reasoning that controls the performance loss under the user-specified performance loss tolerance. In particular, we construct an upper confidence bound on the performance loss, formulated as a monotone function of the uncertainty score, and subsequently determine a threshold for switching to the nonthinking model. Theoretically, using the threshold to switch between the thinking and nonthinking modes ensures bounded performance loss in a distribution-free manner. Our comprehensive experiments on reasoning benchmarks show that the proposed method can save computational budgets and control the user-specified performance loss. 

---
# Leading the Follower: Learning Persuasive Agents in Social Deduction Games 

**Authors**: Zhang Zheng, Deheng Ye, Peilin Zhao, Hao Wang  

**Link**: [PDF](https://arxiv.org/pdf/2510.09087)  

**Abstract**: Large language model (LLM) agents have shown remarkable progress in social deduction games (SDGs). However, existing approaches primarily focus on information processing and strategy selection, overlooking the significance of persuasive communication in influencing other players' beliefs and responses. In SDGs, success depends not only on making correct deductions but on convincing others to response in alignment with one's intent. To address this limitation, we formalize turn-based dialogue in SDGs as a Stackelberg competition, where the current player acts as the leader who strategically influences the follower's response. Building on this theoretical foundation, we propose a reinforcement learning framework that trains agents to optimize utterances for persuasive impact. Through comprehensive experiments across three diverse SDGs, we demonstrate that our agents significantly outperform baselines. This work represents a significant step toward developing AI agents capable of strategic social influence, with implications extending to scenarios requiring persuasive communication. 

---
# Physics-Informed High-order Graph Dynamics Identification Learning for Predicting Complex Networks Long-term Dynamics 

**Authors**: Bicheng Wang, Jinping Wang, Yibo Sue  

**Link**: [PDF](https://arxiv.org/pdf/2510.09082)  

**Abstract**: Learning complex network dynamics is fundamental to understanding, modelling and controlling real-world complex systems. There are two main problems in the task of predicting the dynamic evolution of complex networks: on the one hand, existing methods usually use simple graphs to describe the relationships in complex networks; however, this approach can only capture pairwise relationships, while there may be rich non-pairwise structured relationships in the network. First-order GNNs have difficulty in capturing dynamic non-pairwise relationships. On the other hand, theoretical prediction models lack accuracy and data-driven prediction models lack interpretability. To address the above problems, this paper proposes a higher-order network dynamics identification method for long-term dynamic prediction of complex networks. Firstly, to address the problem that traditional graph machine learning can only deal with pairwise relations, dynamic hypergraph learning is introduced to capture the higher-order non-pairwise relations among complex networks and improve the accuracy of complex network modelling. Then, a dual-driven dynamic prediction module for physical data is proposed. The Koopman operator theory is introduced to transform the nonlinear dynamical differential equations for the dynamic evolution of complex networks into linear systems for solving. Meanwhile, the physical information neural differential equation method is utilised to ensure that the dynamic evolution conforms to the physical laws. The dual-drive dynamic prediction module ensures both accuracy and interpretability of the prediction. Validated on public datasets and self-built industrial chain network datasets, the experimental results show that the method in this paper has good prediction accuracy and long-term prediction performance. 

---
# OSCAR: Orthogonal Stochastic Control for Alignment-Respecting Diversity in Flow Matching 

**Authors**: Jingxuan Wu, Zhenglin Wan, Xingrui Yu, Yuzhe Yang, Bo An, Ivor Tsang  

**Link**: [PDF](https://arxiv.org/pdf/2510.09060)  

**Abstract**: Flow-based text-to-image models follow deterministic trajectories, forcing users to repeatedly sample to discover diverse modes, which is a costly and inefficient process. We present a training-free, inference-time control mechanism that makes the flow itself diversity-aware. Our method simultaneously encourages lateral spread among trajectories via a feature-space objective and reintroduces uncertainty through a time-scheduled stochastic perturbation. Crucially, this perturbation is projected to be orthogonal to the generation flow, a geometric constraint that allows it to boost variation without degrading image details or prompt fidelity. Our procedure requires no retraining or modification to the base sampler and is compatible with common flow-matching solvers. Theoretically, our method is shown to monotonically increase a volume surrogate while, due to its geometric constraints, approximately preserving the marginal distribution. This provides a principled explanation for why generation quality is robustly maintained. Empirically, across multiple text-to-image settings under fixed sampling budgets, our method consistently improves diversity metrics such as the Vendi Score and Brisque over strong baselines, while upholding image quality and alignment. 

---
# MEC$^3$O: Multi-Expert Consensus for Code Time Complexity Prediction 

**Authors**: Joonghyuk Hahn, Soohan Lim, Yo-Sub Han  

**Link**: [PDF](https://arxiv.org/pdf/2510.09049)  

**Abstract**: Predicting the complexity of source code is essential for software development and algorithm analysis. Recently, Baik et al. (2025) introduced CodeComplex for code time complexity prediction. The paper shows that LLMs without fine-tuning struggle with certain complexity classes. This suggests that no single LLM excels at every class, but rather each model shows advantages in certain classes. We propose MEC$^3$O, a multi-expert consensus system, which extends the multi-agent debate frameworks. MEC$^3$O assigns LLMs to complexity classes based on their performance and provides them with class-specialized instructions, turning them into experts. These experts engage in structured debates, and their predictions are integrated through a weighted consensus mechanism. Our expertise assignments to LLMs effectively handle Degeneration-of-Thought, reducing reliance on a separate judge model, and preventing convergence to incorrect majority opinions. Experiments on CodeComplex show that MEC$^3$O outperforms the open-source baselines, achieving at least 10% higher accuracy and macro-F1 scores. It also surpasses GPT-4o-mini in macro-F1 scores on average and demonstrates competitive on-par F1 scores to GPT-4o and GPT-o4-mini on average. This demonstrates the effectiveness of multi-expert debates and weight consensus strategy to generate the final predictions. Our code and data is available at this https URL. 

---
# Humanoid Artificial Consciousness Designed with Large Language Model Based on Psychoanalysis and Personality Theory 

**Authors**: Sang Hun Kim, Jongmin Lee, Dongkyu Park, So Young Lee, Yosep Chong  

**Link**: [PDF](https://arxiv.org/pdf/2510.09043)  

**Abstract**: Human consciousness is still a concept hard to define with current scientific understanding. Although Large Language Models (LLMs) have recently demonstrated significant advancements across various domains including translation and summarization, human consciousness is not something to imitate with current upfront technology owing to so-called hallucination. This study, therefore, proposes a novel approach to address these challenges by integrating psychoanalysis and the Myers-Briggs Type Indicator (MBTI) into constructing consciousness and personality modules. We developed three artificial consciousnesses (self-awareness, unconsciousness, and preconsciousness) based on the principles of psychoanalysis. Additionally, we designed 16 characters with different personalities representing the sixteen MBTI types, with several attributes such as needs, status, and memories. To determine if our model's artificial consciousness exhibits human-like cognition, we created ten distinct situations considering seven attributes such as emotional understanding and logical thinking. The decision-making process of artificial consciousness and the final action were evaluated in three ways: survey evaluation, three-tier classification via ChatGPT, and qualitative review. Both quantitative and qualitative analyses indicated a high likelihood of well-simulated consciousness, although the difference in response between different characters and consciousnesses was not very significant. This implies that the developed models incorporating elements of psychoanalysis and personality theory can lead to building a more intuitive and adaptable AI system with humanoid consciousness. Therefore, this study contributes to opening up new avenues for improving AI interactions in complex cognitive contexts. 

---
# Auto-scaling Continuous Memory for GUI Agent 

**Authors**: Wenyi Wu, Kun Zhou, Ruoxin Yuan, Vivian Yu, Stephen Wang, Zhiting Hu, Biwei Huang  

**Link**: [PDF](https://arxiv.org/pdf/2510.09038)  

**Abstract**: We study how to endow GUI agents with scalable memory that help generalize across unfamiliar interfaces and long-horizon tasks. Prior GUI agents compress past trajectories into text tokens, which balloons context length and misses decisive visual cues (e.g., exact widget size and position). We propose a continuous memory that encodes each GUI trajectory into a fixed-length sequence of continuous embeddings using the VLM itself as an encoder; these embeddings are plugged directly into the backbone's input layer, sharply reducing context cost while preserving fine-grained visual information. As memory size and retrieval depth increase, performance improves monotonically, unlike text memories that degrade with long prompts. To grow memory at low cost, we introduce an auto-scaling data flywheel that (i) discovers new environments via search, (ii) synthesizes tasks with an open-source VLM, (iii) rolls out trajectories with the agent, and (iv) verifies success with the same VLM. Using this pipeline, we collect 100k+ trajectories for about \$4000 and fine-tune only the memory encoder (LoRA on a Q-Former, 1.2\% parameters) with 1,500 samples. On real-world GUI benchmarks, our memory-augmented agent consistently improves success rates under long horizons and distribution shifts. Notably, Qwen-2.5-VL-7B + continuous memory achieves performance comparable to state-of-the-art closed-source models (e.g., GPT-4o, Claude-4). 

---
# Repairing Regex Vulnerabilities via Localization-Guided Instructions 

**Authors**: Sicheol Sung, Joonghyuk Hahn, Yo-Sub Han  

**Link**: [PDF](https://arxiv.org/pdf/2510.09037)  

**Abstract**: Regular expressions (regexes) are foundational to modern computing for critical tasks like input validation and data parsing, yet their ubiquity exposes systems to regular expression denial of service (ReDoS), a vulnerability requiring automated repair methods. Current approaches, however, are hampered by a trade-off. Symbolic, rule-based system are precise but fails to repair unseen or complex vulnerability patterns. Conversely, large language models (LLMs) possess the necessary generalizability but are unreliable for tasks demanding strict syntactic and semantic correctness. We resolve this impasse by introducing a hybrid framework, localized regex repair (LRR), designed to harness LLM generalization while enforcing reliability. Our core insight is to decouple problem identification from the repair process. First, a deterministic, symbolic module localizes the precise vulnerable subpattern, creating a constrained and tractable problem space. Then, the LLM invoked to generate a semantically equivalent fix for this isolated segment. This combined architecture successfully resolves complex repair cases intractable for rule-based repair while avoiding the semantic errors of LLM-only approaches. Our work provides a validated methodology for solving such problems in automated repair, improving the repair rate by 15.4%p over the state-of-the-art. Our code is available at this https URL. 

---
# RefGrader: Automated Grading of Mathematical Competition Proofs using Agentic Workflows 

**Authors**: Hamed Mahdavi, Pouria Mahdavinia, Samira Malek, Pegah Mohammadipour, Alireza Hashemi, Majid Daliri, Alireza Farhadi, Amir Khasahmadi, Niloofar Mireshghallah, Vasant Honavar  

**Link**: [PDF](https://arxiv.org/pdf/2510.09021)  

**Abstract**: State-of-the-art (SOTA) LLMs have progressed from struggling on proof-based Olympiad problems to solving most of the IMO 2025 problems, with leading systems reportedly handling 5 of 6 problems. Given this progress, we assess how well these models can grade proofs: detecting errors, judging their severity, and assigning fair scores beyond binary correctness. We study proof-analysis capabilities using a corpus of 90 Gemini 2.5 Pro-generated solutions that we grade on a 1-4 scale with detailed error annotations, and on MathArena solution sets for IMO/USAMO 2025 scored on a 0-7 scale. Our analysis shows that models can reliably flag incorrect (including subtly incorrect) solutions but exhibit calibration gaps in how partial credit is assigned. To address this, we introduce agentic workflows that extract and analyze reference solutions and automatically derive problem-specific rubrics for a multi-step grading process. We instantiate and compare different design choices for the grading workflows, and evaluate their trade-offs. Across our annotated corpus and MathArena, our proposed workflows achieve higher agreement with human grades and more consistent handling of partial credit across metrics. We release all code, data, and prompts/logs to facilitate future research. 

---
# TripScore: Benchmarking and rewarding real-world travel planning with fine-grained evaluation 

**Authors**: Yincen Qu, Huan Xiao, Feng Li, Hui Zhou, Xiangying Dai  

**Link**: [PDF](https://arxiv.org/pdf/2510.09011)  

**Abstract**: Travel planning is a valuable yet complex task that poses significant challenges even for advanced large language models (LLMs). While recent benchmarks have advanced in evaluating LLMs' planning capabilities, they often fall short in evaluating feasibility, reliability, and engagement of travel plans. We introduce a comprehensive benchmark for travel planning that unifies fine-grained criteria into a single reward, enabling direct comparison of plan quality and seamless integration with reinforcement learning (RL). Our evaluator achieves moderate agreement with travel-expert annotations (60.75\%) and outperforms multiple LLM-as-judge baselines. We further release a large-scale dataset of 4,870 queries including 219 real-world, free-form requests for generalization to authentic user intent. Using this benchmark, we conduct extensive experiments across diverse methods and LLMs, including test-time computation, neuro-symbolic approaches, supervised fine-tuning, and RL via GRPO. Across base models, RL generally improves itinerary feasibility over prompt-only and supervised baselines, yielding higher unified reward scores. 

---
# Tiny-R1V: Lightweight Multimodal Unified Reasoning Model via Model Merging 

**Authors**: Qixiang Yin, Huanjin Yao, Jianghao Chen, Jiaxing Huang, Zhicheng Zhao, Fei Su  

**Link**: [PDF](https://arxiv.org/pdf/2510.08987)  

**Abstract**: Although Multimodal Large Language Models (MLLMs) have demonstrated remarkable capabilities across diverse tasks, they encounter numerous challenges in terms of reasoning efficiency, such as large model size, overthinking, and compromised accuracy in lightweight scenarios. However, research on the reasoning capabilities of lightweight MLLMs is quite lacking. To this end, we propose Tiny-R1V, a novel lightweight 3B model that achieves faster inference and higher accuracy via a two-stage optimization, while unifying multimodal reasoning across multiple tasks and using fewer tokens. In the first stage, Tiny-R1V introduces Length-Informed Relative Policy Optimization (LIPO), a novel reinforcement learning method, to train each reasoning model. The LIPO is designed to dynamically adjusts advantages of responses within groups, that is, by prioritizing concise yet high-quality responses to encourage the generation of shorter and more accurate response. In the second stage, we propose Adaptive Model Merging (AMM), a training-free model merging method that merges multiple specialist models into a unified architecture. Specifically, AMM adaptively adjusts the weights of task vectors and robustly optimizes the merged vectors via a novel gradient projection regularization loss function, thus mitigating redundant conflicts between them. Extensive evaluations on ten widely-used reasoning benchmarks covering mathematics, structured data (charts, tables, documents), OCR, and general capabilities showcase the superior performance of Tiny-R1V, enabling lightweight models to excel in diverse multimodal reasoning tasks. 

---
# Semantic-Condition Tuning: Fusing Graph Context with Large Language Models for Knowledge Graph Completion 

**Authors**: Ruitong Liu, Yan Wen, Te Sun, Yunjia Wu, Pingyang Huang, Zihang Yu, Siyuan Li  

**Link**: [PDF](https://arxiv.org/pdf/2510.08966)  

**Abstract**: Fusing Knowledge Graphs with Large Language Models is crucial for knowledge-intensive tasks like knowledge graph completion. The prevailing paradigm, prefix-tuning, simply concatenates knowledge embeddings with text inputs. However, this shallow fusion overlooks the rich relational semantics within KGs and imposes a significant implicit reasoning burden on the LLM to correlate the prefix with the text. To address these, we propose Semantic-condition Tuning (SCT), a new knowledge injection paradigm comprising two key modules. First, a Semantic Graph Module employs a Graph Neural Network to extract a context-aware semantic condition from the local graph neighborhood, guided by knowledge-enhanced relations. Subsequently, this condition is passed to a Condition-Adaptive Fusion Module, which, in turn, adaptively modulates the textual embedding via two parameterized projectors, enabling a deep, feature-wise, and knowledge-aware interaction. The resulting pre-fused embedding is then fed into the LLM for fine-tuning. Extensive experiments on knowledge graph benchmarks demonstrate that SCT significantly outperforms prefix-tuning and other strong baselines. Our analysis confirms that by modulating the input representation with semantic graph context before LLM inference, SCT provides a more direct and potent signal, enabling more accurate and robust knowledge reasoning. 

---
# DualResearch: Entropy-Gated Dual-Graph Retrieval for Answer Reconstruction 

**Authors**: Jinxin Shi, Zongsheng Cao, Runmin Ma, Yusong Hu, Jie Zhou, Xin Li, Lei Bai, Liang He, Bo Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2510.08959)  

**Abstract**: The deep-research framework orchestrates external tools to perform complex, multi-step scientific reasoning that exceeds the native limits of a single large language model. However, it still suffers from context pollution, weak evidentiary support, and brittle execution paths. To address these issues, we propose DualResearch, a retrieval and fusion framework that matches the epistemic structure of tool-intensive reasoning by jointly modeling two complementary graphs: a breadth semantic graph that encodes stable background knowledge, and a depth causal graph that captures execution provenance. Each graph has a layer-native relevance function, seed-anchored semantic diffusion for breadth, and causal-semantic path matching with reliability weighting for depth. To reconcile their heterogeneity and query-dependent uncertainty, DualResearch converts per-layer path evidence into answer distributions and fuses them in log space via an entropy-gated rule with global calibration. The fusion up-weights the more certain channel and amplifies agreement. As a complement to deep-research systems, DualResearch compresses lengthy multi-tool execution logs into a concise reasoning graph, and we show that it can reconstruct answers stably and effectively. On the scientific reasoning benchmarks HLE and GPQA, DualResearch achieves competitive performance. Using log files from the open-source system InternAgent, its accuracy improves by 7.7% on HLE and 6.06% on GPQA. 

---
# EcphoryRAG: Re-Imagining Knowledge-Graph RAG via Human Associative Memory 

**Authors**: Zirui Liao  

**Link**: [PDF](https://arxiv.org/pdf/2510.08958)  

**Abstract**: Cognitive neuroscience research indicates that humans leverage cues to activate entity-centered memory traces (engrams) for complex, multi-hop recollection. Inspired by this mechanism, we introduce EcphoryRAG, an entity-centric knowledge graph RAG framework. During indexing, EcphoryRAG extracts and stores only core entities with corresponding metadata, a lightweight approach that reduces token consumption by up to 94\% compared to other structured RAG systems. For retrieval, the system first extracts cue entities from queries, then performs a scalable multi-hop associative search across the knowledge graph. Crucially, EcphoryRAG dynamically infers implicit relations between entities to populate context, enabling deep reasoning without exhaustive pre-enumeration of relationships. Extensive evaluations on the 2WikiMultiHop, HotpotQA, and MuSiQue benchmarks demonstrate that EcphoryRAG sets a new state-of-the-art, improving the average Exact Match (EM) score from 0.392 to 0.474 over strong KG-RAG methods like HippoRAG. These results validate the efficacy of the entity-cue-multi-hop retrieval paradigm for complex question answering. 

---
# FATHOMS-RAG: A Framework for the Assessment of Thinking and Observation in Multimodal Systems that use Retrieval Augmented Generation 

**Authors**: Samuel Hildebrand, Curtis Taylor, Sean Oesch, James M Ghawaly Jr, Amir Sadovnik, Ryan Shivers, Brandon Schreiber, Kevin Kurian  

**Link**: [PDF](https://arxiv.org/pdf/2510.08945)  

**Abstract**: Retrieval-augmented generation (RAG) has emerged as a promising paradigm for improving factual accuracy in large language models (LLMs). We introduce a benchmark designed to evaluate RAG pipelines as a whole, evaluating a pipeline's ability to ingest, retrieve, and reason about several modalities of information, differentiating it from existing benchmarks that focus on particular aspects such as retrieval. We present (1) a small, human-created dataset of 93 questions designed to evaluate a pipeline's ability to ingest textual data, tables, images, and data spread across these modalities in one or more documents; (2) a phrase-level recall metric for correctness; (3) a nearest-neighbor embedding classifier to identify potential pipeline hallucinations; (4) a comparative evaluation of 2 pipelines built with open-source retrieval mechanisms and 4 closed-source foundation models; and (5) a third-party human evaluation of the alignment of our correctness and hallucination metrics. We find that closed-source pipelines significantly outperform open-source pipelines in both correctness and hallucination metrics, with wider performance gaps in questions relying on multimodal and cross-document information. Human evaluation of our metrics showed average agreement of 4.62 for correctness and 4.53 for hallucination detection on a 1-5 Likert scale (5 indicating "strongly agree"). 

---
# RADAR: Mechanistic Pathways for Detecting Data Contamination in LLM Evaluation 

**Authors**: Ashish Kattamuri, Harshwardhan Fartale, Arpita Vats, Rahul Raja, Ishita Prasad  

**Link**: [PDF](https://arxiv.org/pdf/2510.08931)  

**Abstract**: Data contamination poses a significant challenge to reliable LLM evaluation, where models may achieve high performance by memorizing training data rather than demonstrating genuine reasoning capabilities. We introduce RADAR (Recall vs. Reasoning Detection through Activation Representation), a novel framework that leverages mechanistic interpretability to detect contamination by distinguishing recall-based from reasoning-based model responses. RADAR extracts 37 features spanning surface-level confidence trajectories and deep mechanistic properties including attention specialization, circuit dynamics, and activation flow patterns. Using an ensemble of classifiers trained on these features, RADAR achieves 93\% accuracy on a diverse evaluation set, with perfect performance on clear cases and 76.7\% accuracy on challenging ambiguous examples. This work demonstrates the potential of mechanistic interpretability for advancing LLM evaluation beyond traditional surface-level metrics. 

---
# LM Fight Arena: Benchmarking Large Multimodal Models via Game Competition 

**Authors**: Yushuo Zheng, Zicheng Zhang, Xiongkuo Min, Huiyu Duan, Guangtao Zhai  

**Link**: [PDF](https://arxiv.org/pdf/2510.08928)  

**Abstract**: Existing benchmarks for large multimodal models (LMMs) often fail to capture their performance in real-time, adversarial environments. We introduce LM Fight Arena (Large Model Fight Arena), a novel framework that evaluates LMMs by pitting them against each other in the classic fighting game Mortal Kombat II, a task requiring rapid visual understanding and tactical, sequential decision-making. In a controlled tournament, we test six leading open- and closed-source models, where each agent operates controlling the same character to ensure a fair comparison. The models are prompted to interpret game frames and state data to select their next actions. Unlike static evaluations, LM Fight Arena provides a fully automated, reproducible, and objective assessment of an LMM's strategic reasoning capabilities in a dynamic setting. This work introduces a challenging and engaging benchmark that bridges the gap between AI evaluation and interactive entertainment. 

---
# GTAlign: Game-Theoretic Alignment of LLM Assistants for Mutual Welfare 

**Authors**: Siqi Zhu, David Zhang, Pedro Cisneros-Velarde, Jiaxuan You  

**Link**: [PDF](https://arxiv.org/pdf/2510.08872)  

**Abstract**: Large Language Models (LLMs) have achieved remarkable progress in reasoning, yet sometimes produce responses that are suboptimal for users in tasks such as writing, information seeking, or providing practical guidance. Conventional alignment practices typically assume that maximizing model reward also maximizes user welfare, but this assumption frequently fails in practice: models may over-clarify or generate overly verbose reasoning when users prefer concise answers. Such behaviors resemble the prisoner's dilemma, where individually rational choices lead to socially suboptimal outcomes. The fundamental challenge is the lack of a principled decision making mechanism that mutually benefits both the LLM and the user. We propose Game-Theoretic Alignment (GTAlign), an alignment framework that integrates game-theoretic decision making into both reasoning and training. During reasoning, the model explicitly treats user-LLM interaction as a strategic game: it constructs payoff matrices within its reasoning chain to estimate welfare for both itself and the user, and then selects actions that are mutually beneficial. During training, we introduce a mutual welfare reward that reinforces cooperative responses, aligning model behavior with socially efficient outcomes. In addition, we introduce an inference technique that leverages game-theoretic reasoning to dynamically adapt LLM's response when pricing policies of LLM service change. Extensive experiments demonstrate that GTAlign substantially improves reasoning efficiency, answer quality, and mutual welfare compared to baselines across diverse tasks. The code is available at this https URL . 

---
# ReviewerToo: Should AI Join The Program Committee? A Look At The Future of Peer Review 

**Authors**: Gaurav Sahu, Hugo Larochelle, Laurent Charlin, Christopher Pal  

**Link**: [PDF](https://arxiv.org/pdf/2510.08867)  

**Abstract**: Peer review is the cornerstone of scientific publishing, yet it suffers from inconsistencies, reviewer subjectivity, and scalability challenges. We introduce ReviewerToo, a modular framework for studying and deploying AI-assisted peer review to complement human judgment with systematic and consistent assessments. ReviewerToo supports systematic experiments with specialized reviewer personas and structured evaluation criteria, and can be partially or fully integrated into real conference workflows. We validate ReviewerToo on a carefully curated dataset of 1,963 paper submissions from ICLR 2025, where our experiments with the gpt-oss-120b model achieves 81.8% accuracy for the task of categorizing a paper as accept/reject compared to 83.9% for the average human reviewer. Additionally, ReviewerToo-generated reviews are rated as higher quality than the human average by an LLM judge, though still trailing the strongest expert contributions. Our analysis highlights domains where AI reviewers excel (e.g., fact-checking, literature coverage) and where they struggle (e.g., assessing methodological novelty and theoretical contributions), underscoring the continued need for human expertise. Based on these findings, we propose guidelines for integrating AI into peer-review pipelines, showing how AI can enhance consistency, coverage, and fairness while leaving complex evaluative judgments to domain experts. Our work provides a foundation for systematic, hybrid peer-review systems that scale with the growth of scientific publishing. 

---
# What Is Your Agent's GPA? A Framework for Evaluating Agent Goal-Plan-Action Alignment 

**Authors**: Allison Sihan Jia, Daniel Huang, Nikhil Vytla, Nirvika Choudhury, John C Mitchell, Anupam Datta  

**Link**: [PDF](https://arxiv.org/pdf/2510.08847)  

**Abstract**: We introduce the Agent GPA (Goal-Plan-Action) framework: an evaluation paradigm based on an agent's operational loop of setting goals, devising plans, and executing actions. The framework includes five evaluation metrics: Goal Fulfillment, Logical Consistency, Execution Efficiency, Plan Quality, and Plan Adherence. Logical Consistency checks that an agent's actions are consistent with its prior actions. Execution Efficiency checks whether the agent executes in the most efficient way to achieve its goal. Plan Quality checks whether an agent's plans are aligned with its goals; Plan Adherence checks if an agent's actions are aligned with its plan; and Goal Fulfillment checks that agent's final outcomes match the stated goals. Our experimental results on two benchmark datasets - the public TRAIL/GAIA dataset and an internal dataset for a production-grade data agent - show that this framework (a) provides a systematic way to cover a broad range of agent failures, including all agent errors on the TRAIL/GAIA benchmark dataset; (b) supports LLM-judges that exhibit strong agreement with human annotation, covering 80% to over 95% errors; and (c) localizes errors with 86% agreement to enable targeted improvement of agent performance. 

---
# Everyone prefers human writers, including AI 

**Authors**: Wouter Haverals, Meredith Martin  

**Link**: [PDF](https://arxiv.org/pdf/2510.08831)  

**Abstract**: As AI writing tools become widespread, we need to understand how both humans and machines evaluate literary style, a domain where objective standards are elusive and judgments are inherently subjective. We conducted controlled experiments using Raymond Queneau's Exercises in Style (1947) to measure attribution bias across evaluators. Study 1 compared human participants (N=556) and AI models (N=13) evaluating literary passages from Queneau versus GPT-4-generated versions under three conditions: blind, accurately labeled, and counterfactually labeled. Study 2 tested bias generalization across a 14$\times$14 matrix of AI evaluators and creators. Both studies revealed systematic pro-human attribution bias. Humans showed +13.7 percentage point (pp) bias (Cohen's h = 0.28, 95% CI: 0.21-0.34), while AI models showed +34.3 percentage point bias (h = 0.70, 95% CI: 0.65-0.76), a 2.5-fold stronger effect (P$<$0.001). Study 2 confirmed this bias operates across AI architectures (+25.8pp, 95% CI: 24.1-27.6%), demonstrating that AI systems systematically devalue creative content when labeled as "AI-generated" regardless of which AI created it. We also find that attribution labels cause evaluators to invert assessment criteria, with identical features receiving opposing evaluations based solely on perceived authorship. This suggests AI models have absorbed human cultural biases against artificial creativity during training. Our study represents the first controlled comparison of attribution bias between human and artificial evaluators in aesthetic judgment, revealing that AI systems not only replicate but amplify this human tendency. 

---
# COMPASS: Enhancing Agent Long-Horizon Reasoning with Evolving Context 

**Authors**: Guangya Wan, Mingyang Ling, Xiaoqi Ren, Rujun Han, Sheng Li, Zizhao Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2510.08790)  

**Abstract**: Long-horizon tasks that require sustained reasoning and multiple tool interactions remain challenging for LLM agents: small errors compound across steps, and even state-of-the-art models often hallucinate or lose coherence. We identify context management as the central bottleneck -- extended histories cause agents to overlook critical evidence or become distracted by irrelevant information, thus failing to replan or reflect from previous mistakes. To address this, we propose COMPASS (Context-Organized Multi-Agent Planning and Strategy System), a lightweight hierarchical framework that separates tactical execution, strategic oversight, and context organization into three specialized components: (1) a Main Agent that performs reasoning and tool use, (2) a Meta-Thinker that monitors progress and issues strategic interventions, and (3) a Context Manager that maintains concise, relevant progress briefs for different reasoning stages. Across three challenging benchmarks -- GAIA, BrowseComp, and Humanity's Last Exam -- COMPASS improves accuracy by up to 20% relative to both single- and multi-agent baselines. We further introduce a test-time scaling extension that elevates performance to match established DeepResearch agents, and a post-training pipeline that delegates context management to smaller models for enhanced efficiency. 

---
# Robust Heuristic Algorithm Design with LLMs 

**Authors**: Pantea Karimi, Dany Rouhana, Pooria Namyar, Siva Kesava Reddy Kakarla, Venkat Arun, Behnaz Arzani  

**Link**: [PDF](https://arxiv.org/pdf/2510.08755)  

**Abstract**: We posit that we can generate more robust and performant heuristics if we augment approaches using LLMs for heuristic design with tools that explain why heuristics underperform and suggestions about how to fix them. We find even simple ideas that (1) expose the LLM to instances where the heuristic underperforms; (2) explain why they occur; and (3) specialize design to regions in the input space, can produce more robust algorithms compared to existing techniques~ -- ~the heuristics we produce have a $\sim28\times$ better worst-case performance compared to FunSearch, improve average performance, and maintain the runtime. 

---
# Unified World Models: Memory-Augmented Planning and Foresight for Visual Navigation 

**Authors**: Yifei Dong, Fengyi Wu, Guangyu Chen, Zhi-Qi Cheng, Qiyu Hu, Yuxuan Zhou, Jingdong Sun, Jun-Yan He, Qi Dai, Alexander G Hauptmann  

**Link**: [PDF](https://arxiv.org/pdf/2510.08713)  

**Abstract**: Enabling embodied agents to effectively imagine future states is critical for robust and generalizable visual navigation. Current state-of-the-art approaches, however, adopt modular architectures that separate navigation planning from visual world modeling, leading to state-action misalignment and limited adaptability in novel or dynamic scenarios. To overcome this fundamental limitation, we propose UniWM, a unified, memory-augmented world model integrating egocentric visual foresight and planning within a single multimodal autoregressive backbone. Unlike modular frameworks, UniWM explicitly grounds action decisions in visually imagined outcomes, ensuring tight alignment between prediction and control. A hierarchical memory mechanism further integrates detailed short-term perceptual cues with longer-term trajectory context, enabling stable, coherent reasoning over extended horizons. Extensive experiments across four challenging benchmarks (Go Stanford, ReCon, SCAND, HuRoN) demonstrate that UniWM substantially improves navigation success rates by up to 30%, significantly reduces trajectory errors compared to strong baselines, and exhibits impressive zero-shot generalization on the unseen TartanDrive dataset. These results highlight UniWM as a principled step toward unified, imagination-driven embodied navigation. 

---
# Optimizing delivery for quick commerce factoring qualitative assessment of generated routes 

**Authors**: Milon Bhattacharya, Milan Kumar  

**Link**: [PDF](https://arxiv.org/pdf/2510.08671)  

**Abstract**: Indias e-commerce market is projected to grow rapidly, with last-mile delivery accounting for nearly half of operational expenses. Although vehicle routing problem (VRP) based solvers are widely used for delivery planning, their effectiveness in real-world scenarios is limited due to unstructured addresses, incomplete maps, and computational constraints in distance estimation. This study proposes a framework that employs large language models (LLMs) to critique VRP-generated routes against policy-based criteria, allowing logistics operators to evaluate and prioritise more efficient delivery plans. As a illustration of our approach we generate, annotate and evaluated 400 cases using large language models. Our study found that open-source LLMs identified routing issues with 79% accuracy, while proprietary reasoning models achieved reach upto 86%. The results demonstrate that LLM-based evaluation of VRP-generated routes can be an effective and scalable layer of evaluation which goes beyond beyond conventional distance and time based metrics. This has implications for improving cost efficiency, delivery reliability, and sustainability in last-mile logistics, especially for developing countries like India. 

---
# Hypothesis Hunting with Evolving Networks of Autonomous Scientific Agents 

**Authors**: Tennison Liu, Silas Ruhrberg Estévez, David L. Bentley, Mihaela van der Schaar  

**Link**: [PDF](https://arxiv.org/pdf/2510.08619)  

**Abstract**: Large-scale scientific datasets -- spanning health biobanks, cell atlases, Earth reanalyses, and more -- create opportunities for exploratory discovery unconstrained by specific research questions. We term this process hypothesis hunting: the cumulative search for insight through sustained exploration across vast and complex hypothesis spaces. To support it, we introduce AScience, a framework modeling discovery as the interaction of agents, networks, and evaluation norms, and implement it as ASCollab, a distributed system of LLM-based research agents with heterogeneous behaviors. These agents self-organize into evolving networks, continually producing and peer-reviewing findings under shared standards of evaluation. Experiments show that such social dynamics enable the accumulation of expert-rated results along the diversity-quality-novelty frontier, including rediscoveries of established biomarkers, extensions of known pathways, and proposals of new therapeutic targets. While wet-lab validation remains indispensable, our experiments on cancer cohorts demonstrate that socially structured, agentic networks can sustain exploratory hypothesis hunting at scale. 

---
# StreamingVLM: Real-Time Understanding for Infinite Video Streams 

**Authors**: Ruyi Xu, Guangxuan Xiao, Yukang Chen, Liuning He, Kelly Peng, Yao Lu, Song Han  

**Link**: [PDF](https://arxiv.org/pdf/2510.09608)  

**Abstract**: Vision-language models (VLMs) could power real-time assistants and autonomous agents, but they face a critical challenge: understanding near-infinite video streams without escalating latency and memory usage. Processing entire videos with full attention leads to quadratic computational costs and poor performance on long videos. Meanwhile, simple sliding window methods are also flawed, as they either break coherence or suffer from high latency due to redundant recomputation. In this paper, we introduce StreamingVLM, a model designed for real-time, stable understanding of infinite visual input. Our approach is a unified framework that aligns training with streaming inference. During inference, we maintain a compact KV cache by reusing states of attention sinks, a short window of recent vision tokens, and a long window of recent text tokens. This streaming ability is instilled via a simple supervised fine-tuning (SFT) strategy that applies full attention on short, overlapped video chunks, which effectively mimics the inference-time attention pattern without training on prohibitively long contexts. For evaluation, we build Inf-Streams-Eval, a new benchmark with videos averaging over two hours that requires dense, per-second alignment between frames and text. On Inf-Streams-Eval, StreamingVLM achieves a 66.18% win rate against GPT-4O mini and maintains stable, real-time performance at up to 8 FPS on a single NVIDIA H100. Notably, our SFT strategy also enhances general VQA abilities without any VQA-specific fine-tuning, improving performance on LongVideoBench by +4.30 and OVOBench Realtime by +5.96. Code is available at this https URL. 

---
# Prompting Test-Time Scaling Is A Strong LLM Reasoning Data Augmentation 

**Authors**: Sondos Mahmoud Bsharat, Zhiqiang Shen  

**Link**: [PDF](https://arxiv.org/pdf/2510.09599)  

**Abstract**: Large language models (LLMs) have demonstrated impressive reasoning capabilities when provided with chain-of-thought exemplars, but curating large reasoning datasets remains laborious and resource-intensive. In this work, we introduce Prompting Test-Time Scaling (P-TTS), a simple yet effective inference-time data augmentation strategy for enhancing LLM reasoning through finetuning. Rather than collecting thousands or even millions of examples, P-TTS leverages a small pool of only 90 manually selected reasoning instances and systematically varies exemplar augmentation through principled instruction prompting intensities at test time to synthesize diverse reasoning trajectory contexts. Then we finetune the various sizes of Qwen-2.5 models on P-TTS data. Across a suite of mathematical reasoning AIME2024 & 25, MATH500, and GPQA-Diamond, our P-TTS-7B and 32B models outperform the prior competitive baselines like S1 and S1.1 (1K-shot), achieving absolute accuracy gains of +26.66% and +30.00% on AIME'24 (7B), and +13.34% and +6.67% on AIME'25 (7B); P-TTS-32B yields gains of +23.33% and +16.63% on AIME'24, and +26.63% and +3.33% on AIME'25 (vs. S1 and S1.1, respectively), with comparable or better performance on MATH500 and GPQA-Diamond. We further show that P-TTS enhances zero-shot generalization accuracy on out-of-domain reasoning benchmarks of Gaokao, Kaoyan, OlympiadBench, AMC23, GradeSchoolMath, and Minerva. Our analysis suggests that test-time scaling effectively explores the latent space of reasoning patterns, amplifying LLM problem-solving with minimal annotation overhead, and further unlocking the reasoning potential and capabilities of LLMs. Prompting Test-Time Scaling offers a practical, low-cost way to elicit LLM reasoning in resource-constrained or rapidly evolving domains. 

---
# BaNEL: Exploration Posteriors for Generative Modeling Using Only Negative Rewards 

**Authors**: Sangyun Lee, Brandon Amos, Giulia Fanti  

**Link**: [PDF](https://arxiv.org/pdf/2510.09596)  

**Abstract**: Today's generative models thrive with large amounts of supervised data and informative reward functions characterizing the quality of the generation. They work under the assumptions that the supervised data provides knowledge to pre-train the model, and the reward function provides dense information about how to further improve the generation quality and correctness. However, in the hardest instances of important problems, two problems arise: (1) the base generative model attains a near-zero reward signal, and (2) calls to the reward oracle are expensive. This setting poses a fundamentally different learning challenge than standard reward-based post-training. To address this, we propose BaNEL (Bayesian Negative Evidence Learning), an algorithm that post-trains the model using failed attempts only, while minimizing the number of reward evaluations (NREs). Our method is based on the idea that the problem of learning regularities underlying failures can be cast as another, in-loop generative modeling problem. We then leverage this model to assess whether new data resembles previously seen failures and steer the generation away from them. We show that BaNEL can improve model performance without observing a single successful sample on several sparse-reward tasks, outperforming existing novelty-bonus approaches by up to several orders of magnitude in success rate, while using fewer reward evaluations. 

---
# Dyna-Mind: Learning to Simulate from Experience for Better AI Agents 

**Authors**: Xiao Yu, Baolin Peng, Michel Galley, Hao Cheng, Qianhui Wu, Janardhan Kulkarni, Suman Nath, Zhou Yu, Jianfeng Gao  

**Link**: [PDF](https://arxiv.org/pdf/2510.09577)  

**Abstract**: Reasoning models have recently shown remarkable progress in domains such as math and coding. However, their expert-level abilities in math and coding contrast sharply with their performance in long-horizon, interactive tasks such as web navigation and computer/phone-use. Inspired by literature on human cognition, we argue that current AI agents need ''vicarious trial and error'' - the capacity to mentally simulate alternative futures before acting - in order to enhance their understanding and performance in complex interactive environments. We introduce Dyna-Mind, a two-stage training framework that explicitly teaches (V)LM agents to integrate such simulation into their reasoning. In stage 1, we introduce Reasoning with Simulations (ReSim), which trains the agent to generate structured reasoning traces from expanded search trees built from real experience gathered through environment interactions. ReSim thus grounds the agent's reasoning in faithful world dynamics and equips it with the ability to anticipate future states in its reasoning. In stage 2, we propose Dyna-GRPO, an online reinforcement learning method to further strengthen the agent's simulation and decision-making ability by using both outcome rewards and intermediate states as feedback from real rollouts. Experiments on two synthetic benchmarks (Sokoban and ALFWorld) and one realistic benchmark (AndroidWorld) demonstrate that (1) ReSim effectively infuses simulation ability into AI agents, and (2) Dyna-GRPO leverages outcome and interaction-level signals to learn better policies for long-horizon, planning-intensive tasks. Together, these results highlight the central role of simulation in enabling AI agents to reason, plan, and act more effectively in the ever more challenging environments. 

---
# SPG: Sandwiched Policy Gradient for Masked Diffusion Language Models 

**Authors**: Chengyu Wang, Paria Rashidinejad, DiJia Su, Song Jiang, Sid Wang, Siyan Zhao, Cai Zhou, Shannon Zejiang Shen, Feiyu Chen, Tommi Jaakkola, Yuandong Tian, Bo Liu  

**Link**: [PDF](https://arxiv.org/pdf/2510.09541)  

**Abstract**: Diffusion large language models (dLLMs) are emerging as an efficient alternative to autoregressive models due to their ability to decode multiple tokens in parallel. However, aligning dLLMs with human preferences or task-specific rewards via reinforcement learning (RL) is challenging because their intractable log-likelihood precludes the direct application of standard policy gradient methods. While prior work uses surrogates like the evidence lower bound (ELBO), these one-sided approximations can introduce significant policy gradient bias. To address this, we propose the Sandwiched Policy Gradient (SPG) that leverages both an upper and a lower bound of the true log-likelihood. Experiments show that SPG significantly outperforms baselines based on ELBO or one-step estimation. Specifically, SPG improves the accuracy over state-of-the-art RL methods for dLLMs by 3.6% in GSM8K, 2.6% in MATH500, 18.4% in Countdown and 27.0% in Sudoku. 

---
# Mitigating Overthinking through Reasoning Shaping 

**Authors**: Feifan Song, Shaohang Wei, Bofei Gao, Yejie Wang, Wen Luo, Wei Li, Linli Yao, Weimin Xiong, Liang Chen, Tianyu Liu, Houfeng Wang  

**Link**: [PDF](https://arxiv.org/pdf/2510.09535)  

**Abstract**: Large reasoning models (LRMs) boosted by Reinforcement Learning from Verifier Reward (RLVR) have shown great power in problem solving, yet they often cause overthinking: excessive, meandering reasoning that inflates computational cost. Prior designs of penalization in RLVR manage to reduce token consumption while often harming model performance, which arises from the oversimplicity of token-level supervision. In this paper, we argue that the granularity of supervision plays a crucial role in balancing efficiency and accuracy, and propose Group Relative Segment Penalization (GRSP), a step-level method to regularize reasoning. Since preliminary analyses show that reasoning segments are strongly correlated with token consumption and model performance, we design a length-aware weighting mechanism across segment clusters. Extensive experiments demonstrate that GRSP achieves superior token efficiency without heavily compromising accuracy, especially the advantages with harder problems. Moreover, GRSP stabilizes RL training and scales effectively across model sizes. 

---
# A methodology for clinically driven interactive segmentation evaluation 

**Authors**: Parhom Esmaeili, Virginia Fernandez, Pedro Borges, Eli Gibson, Sebastien Ourselin, M. Jorge Cardoso  

**Link**: [PDF](https://arxiv.org/pdf/2510.09499)  

**Abstract**: Interactive segmentation is a promising strategy for building robust, generalisable algorithms for volumetric medical image segmentation. However, inconsistent and clinically unrealistic evaluation hinders fair comparison and misrepresents real-world performance. We propose a clinically grounded methodology for defining evaluation tasks and metrics, and built a software framework for constructing standardised evaluation pipelines. We evaluate state-of-the-art algorithms across heterogeneous and complex tasks and observe that (i) minimising information loss when processing user interactions is critical for model robustness, (ii) adaptive-zooming mechanisms boost robustness and speed convergence, (iii) performance drops if validation prompting behaviour/budgets differ from training, (iv) 2D methods perform well with slab-like images and coarse targets, but 3D context helps with large or irregularly shaped targets, (v) performance of non-medical-domain models (e.g. SAM2) degrades with poor contrast and complex shapes. 

---
# Autonomous Soft Robotic Guidewire Navigation via Imitation Learning 

**Authors**: Noah Barnes, Ji Woong Kim, Lingyun Di, Hannah Qu, Anuruddha Bhattacharjee, Miroslaw Janowski, Dheeraj Gandhi, Bailey Felix, Shaopeng Jiang, Olivia Young, Mark Fuge, Ryan D. Sochol, Jeremy D. Brown, Axel Krieger  

**Link**: [PDF](https://arxiv.org/pdf/2510.09497)  

**Abstract**: In endovascular surgery, endovascular interventionists push a thin tube called a catheter, guided by a thin wire to a treatment site inside the patient's blood vessels to treat various conditions such as blood clots, aneurysms, and malformations. Guidewires with robotic tips can enhance maneuverability, but they present challenges in modeling and control. Automation of soft robotic guidewire navigation has the potential to overcome these challenges, increasing the precision and safety of endovascular navigation. In other surgical domains, end-to-end imitation learning has shown promising results. Thus, we develop a transformer-based imitation learning framework with goal conditioning, relative action outputs, and automatic contrast dye injections to enable generalizable soft robotic guidewire navigation in an aneurysm targeting task. We train the model on 36 different modular bifurcated geometries, generating 647 total demonstrations under simulated fluoroscopy, and evaluate it on three previously unseen vascular geometries. The model can autonomously drive the tip of the robot to the aneurysm location with a success rate of 83% on the unseen geometries, outperforming several baselines. In addition, we present ablation and baseline studies to evaluate the effectiveness of each design and data collection choice. Project website: this https URL 

---
# Precoder Design in Multi-User FDD Systems with VQ-VAE and GNN 

**Authors**: Srikar Allaparapu, Michael Baur, Benedikt Böck, Michael Joham, Wolfgang Utschick  

**Link**: [PDF](https://arxiv.org/pdf/2510.09495)  

**Abstract**: Robust precoding is efficiently feasible in frequency division duplex (FDD) systems by incorporating the learnt statistics of the propagation environment through a generative model. We build on previous work that successfully designed site-specific precoders based on a combination of Gaussian mixture models (GMMs) and graph neural networks (GNNs). In this paper, by utilizing a vector quantized-variational autoencoder (VQ-VAE), we circumvent one of the key drawbacks of GMMs, i.e., the number of GMM components scales exponentially to the feedback bits. In addition, the deep learning architecture of the VQ-VAE allows us to jointly train the GNN together with VQ-VAE along with pilot optimization forming an end-to-end (E2E) model, resulting in considerable performance gains in sum rate for multi-user wireless systems. Simulations demonstrate the superiority of the proposed frameworks over the conventional methods involving the sub-discrete Fourier transform (DFT) pilot matrix and iterative precoder algorithms enabling the deployment of systems characterized by fewer pilots or feedback bits. 

---
# Performance Analysis of Machine Learning Algorithms in Chronic Kidney Disease Prediction 

**Authors**: Iftekhar Ahmed, Tanzil Ebad Chowdhury, Biggo Bushon Routh, Nafisa Tasmiya, Shadman Sakib, Adil Ahmed Chowdhury  

**Link**: [PDF](https://arxiv.org/pdf/2510.09493)  

**Abstract**: Kidneys are the filter of the human body. About 10% of the global population is thought to be affected by Chronic Kidney Disease (CKD), which causes kidney function to decline. To protect in danger patients from additional kidney damage, effective risk evaluation of CKD and appropriate CKD monitoring are crucial. Due to quick and precise detection capabilities, Machine Learning models can help practitioners accomplish this goal efficiently; therefore, an enormous number of diagnosis systems and processes in the healthcare sector nowadays are relying on machine learning due to its disease prediction capability. In this study, we designed and suggested disease predictive computer-aided designs for the diagnosis of CKD. The dataset for CKD is attained from the repository of machine learning of UCL, with a few missing values; those are filled in using "mean-mode" and "Random sampling method" strategies. After successfully achieving the missing data, eight ML techniques (Random Forest, SVM, Naive Bayes, Logistic Regression, KNN, XGBoost, Decision Tree, and AdaBoost) were used to establish models, and the performance evaluation comparisons among the result accuracies are measured by the techniques to find the machine learning models with the highest accuracy. Among them, Random Forest as well as Logistic Regression showed an outstanding 99% accuracy, followed by the Ada Boost, XGBoost, Naive Bayes, Decision Tree, and SVM, whereas the KNN classifier model stands last with an accuracy of 73%. 

---
# Multimodal Policy Internalization for Conversational Agents 

**Authors**: Zhenhailong Wang, Jiateng Liu, Amin Fazel, Ritesh Sarkhel, Xing Fan, Xiang Li, Chenlei Guo, Heng Ji, Ruhi Sarikaya  

**Link**: [PDF](https://arxiv.org/pdf/2510.09474)  

**Abstract**: Modern conversational agents like ChatGPT and Alexa+ rely on predefined policies specifying metadata, response styles, and tool-usage rules. As these LLM-based systems expand to support diverse business and user queries, such policies, often implemented as in-context prompts, are becoming increasingly complex and lengthy, making faithful adherence difficult and imposing large fixed computational costs. With the rise of multimodal agents, policies that govern visual and multimodal behaviors are critical but remain understudied. Prior prompt-compression work mainly shortens task templates and demonstrations, while existing policy-alignment studies focus only on text-based safety rules. We introduce Multimodal Policy Internalization (MPI), a new task that internalizes reasoning-intensive multimodal policies into model parameters, enabling stronger policy-following without including the policy during inference. MPI poses unique data and algorithmic challenges. We build two datasets spanning synthetic and real-world decision-making and tool-using tasks and propose TriMPI, a three-stage training framework. TriMPI first injects policy knowledge via continual pretraining, then performs supervised finetuning, and finally applies PolicyRollout, a GRPO-style reinforcement learning extension that augments rollouts with policy-aware responses for grounded exploration. TriMPI achieves notable gains in end-to-end accuracy, generalization, and robustness to forgetting. As the first work on multimodal policy internalization, we provide datasets, training recipes, and comprehensive evaluations to foster future research. Project page: this https URL. 

---
# Scalable Multi-Agent Path Finding using Collision-Aware Dynamic Alert Mask and a Hybrid Execution Strategy 

**Authors**: Bharath Muppasani, Ritirupa Dey, Biplav Srivastava, Vignesh Narayanan  

**Link**: [PDF](https://arxiv.org/pdf/2510.09469)  

**Abstract**: Multi-agent pathfinding (MAPF) remains a critical problem in robotics and autonomous systems, where agents must navigate shared spaces efficiently while avoiding conflicts. Traditional centralized algorithms that have global information, such as Conflict-Based Search (CBS), provide high-quality solutions but become computationally expensive in large-scale scenarios due to the combinatorial explosion of conflicts that need resolution. Conversely, distributed approaches that have local information, particularly learning-based methods, offer better scalability by operating with relaxed information availability, yet often at the cost of solution quality. To address these limitations, we propose a hybrid framework that combines decentralized path planning with a lightweight centralized coordinator. Our framework leverages reinforcement learning (RL) for decentralized planning, enabling agents to adapt their planning based on minimal, targeted alerts--such as static conflict-cell flags or brief conflict tracks--that are dynamically shared information from the central coordinator for effective conflict resolution. We empirically study the effect of the information available to an agent on its planning performance. Our approach reduces the inter-agent information sharing compared to fully centralized and distributed methods, while still consistently finding feasible, collision-free solutions--even in large-scale scenarios having higher agent counts. 

---
# Adaptive Attacks on Trusted Monitors Subvert AI Control Protocols 

**Authors**: Mikhail Terekhov, Alexander Panfilov, Daniil Dzenhaliou, Caglar Gulcehre, Maksym Andriushchenko, Ameya Prabhu, Jonas Geiping  

**Link**: [PDF](https://arxiv.org/pdf/2510.09462)  

**Abstract**: AI control protocols serve as a defense mechanism to stop untrusted LLM agents from causing harm in autonomous settings. Prior work treats this as a security problem, stress testing with exploits that use the deployment context to subtly complete harmful side tasks, such as backdoor insertion. In practice, most AI control protocols are fundamentally based on LLM monitors, which can become a central point of failure. We study adaptive attacks by an untrusted model that knows the protocol and the monitor model, which is plausible if the untrusted model was trained with a later knowledge cutoff or can search for this information autonomously. We instantiate a simple adaptive attack vector by which the attacker embeds publicly known or zero-shot prompt injections in the model outputs. Using this tactic, frontier models consistently evade diverse monitors and complete malicious tasks on two main AI control benchmarks. The attack works universally against current protocols that rely on a monitor. Furthermore, the recent Defer-to-Resample protocol even backfires, as its resampling amplifies the prompt injection and effectively reframes it as a best-of-$n$ attack. In general, adaptive attacks on monitor models represent a major blind spot in current control protocols and should become a standard component of evaluations for future AI control mechanisms. 

---
# Failure Prediction at Runtime for Generative Robot Policies 

**Authors**: Ralf Römer, Adrian Kobras, Luca Worbis, Angela P. Schoellig  

**Link**: [PDF](https://arxiv.org/pdf/2510.09459)  

**Abstract**: Imitation learning (IL) with generative models, such as diffusion and flow matching, has enabled robots to perform complex, long-horizon tasks. However, distribution shifts from unseen environments or compounding action errors can still cause unpredictable and unsafe behavior, leading to task failure. Early failure prediction during runtime is therefore essential for deploying robots in human-centered and safety-critical environments. We propose FIPER, a general framework for Failure Prediction at Runtime for generative IL policies that does not require failure data. FIPER identifies two key indicators of impending failure: (i) out-of-distribution (OOD) observations detected via random network distillation in the policy's embedding space, and (ii) high uncertainty in generated actions measured by a novel action-chunk entropy score. Both failure prediction scores are calibrated using a small set of successful rollouts via conformal prediction. A failure alarm is triggered when both indicators, aggregated over short time windows, exceed their thresholds. We evaluate FIPER across five simulation and real-world environments involving diverse failure modes. Our results demonstrate that FIPER better distinguishes actual failures from benign OOD situations and predicts failures more accurately and earlier than existing methods. We thus consider this work an important step towards more interpretable and safer generative robot policies. Code, data and videos are available at this https URL. 

---
# SilvaScenes: Tree Segmentation and Species Classification from Under-Canopy Images in Natural Forests 

**Authors**: David-Alexandre Duclos, William Guimont-Martin, Gabriel Jeanson, Arthur Larochelle-Tremblay, Théo Defosse, Frédéric Moore, Philippe Nolet, François Pomerleau, Philippe Giguère  

**Link**: [PDF](https://arxiv.org/pdf/2510.09458)  

**Abstract**: Interest in robotics for forest management is growing, but perception in complex, natural environments remains a significant hurdle. Conditions such as heavy occlusion, variable lighting, and dense vegetation pose challenges to automated systems, which are essential for precision forestry, biodiversity monitoring, and the automation of forestry equipment. These tasks rely on advanced perceptual capabilities, such as detection and fine-grained species classification of individual trees. Yet, existing datasets are inadequate to develop such perception systems, as they often focus on urban settings or a limited number of species. To address this, we present SilvaScenes, a new dataset for instance segmentation of tree species from under-canopy images. Collected across five bioclimatic domains in Quebec, Canada, SilvaScenes features 1476 trees from 24 species with annotations from forestry experts. We demonstrate the relevance and challenging nature of our dataset by benchmarking modern deep learning approaches for instance segmentation. Our results show that, while tree segmentation is easy, with a top mean average precision (mAP) of 67.65%, species classification remains a significant challenge with an mAP of only 35.69%. Our dataset and source code will be available at this https URL. 

---
# Bandits with Single-Peaked Preferences and Limited Resources 

**Authors**: Gur Keinan, Rotem Torkan, Omer Ben-Porat  

**Link**: [PDF](https://arxiv.org/pdf/2510.09425)  

**Abstract**: We study an online stochastic matching problem in which an algorithm sequentially matches $U$ users to $K$ arms, aiming to maximize cumulative reward over $T$ rounds under budget constraints. Without structural assumptions, computing the optimal matching is NP-hard, making online learning computationally infeasible. To overcome this barrier, we focus on \emph{single-peaked preferences} -- a well-established structure in social choice theory, where users' preferences are unimodal with respect to a common order over arms. We devise an efficient algorithm for the offline budgeted matching problem, and leverage it into an efficient online algorithm with a regret of $\tilde O(UKT^{2/3})$. Our approach relies on a novel PQ tree-based order approximation method. If the single-peaked structure is known, we develop an efficient UCB-like algorithm that achieves a regret bound of $\tilde O(U\sqrt{TK})$. 

---
# The Speech-LLM Takes It All: A Truly Fully End-to-End Spoken Dialogue State Tracking Approach 

**Authors**: Nizar El Ghazal, Antoine Caubrière, Valentin Vielzeuf  

**Link**: [PDF](https://arxiv.org/pdf/2510.09424)  

**Abstract**: This paper presents a comparative study of context management strategies for end-to-end Spoken Dialog State Tracking using Speech-LLMs. We systematically evaluate traditional multimodal context (combining text history and spoken current turn), full spoken history, and compressed spoken history approaches. Our experiments on the SpokenWOZ corpus demonstrate that providing the full spoken conversation as input yields the highest performance among models of similar size, significantly surpassing prior methods. Furthermore, we show that attention-pooling-based compression of the spoken history offers a strong trade-off, maintaining competitive accuracy with reduced context size. Detailed analysis confirms that improvements stem from more effective context utilization. 

---
# On the Representations of Entities in Auto-regressive Large Language Models 

**Authors**: Victor Morand, Josiane Mothe, Benjamin Piwowarski  

**Link**: [PDF](https://arxiv.org/pdf/2510.09421)  

**Abstract**: Named entities are fundamental building blocks of knowledge in text, grounding factual information and structuring relationships within language. Despite their importance, it remains unclear how Large Language Models (LLMs) internally represent entities. Prior research has primarily examined explicit relationships, but little is known about entity representations themselves. We introduce entity mention reconstruction as a novel framework for studying how LLMs encode and manipulate entities. We investigate whether entity mentions can be generated from internal representations, how multi-token entities are encoded beyond last-token embeddings, and whether these representations capture relational knowledge. Our proposed method, leveraging _task vectors_, allows to consistently generate multi-token mentions from various entity representations derived from the LLMs hidden states. We thus introduce the _Entity Lens_, extending the _logit-lens_ to predict multi-token mentions. Our results bring new evidence that LLMs develop entity-specific mechanisms to represent and manipulate any multi-token entities, including those unseen during training. Our code is avalable at this https URL . 

---
# Beyond Single-Granularity Prompts: A Multi-Scale Chain-of-Thought Prompt Learning for Graph 

**Authors**: Ziyu Zheng, Yaming Yang, Ziyu Guan, Wei Zhao, Xinyan Huang, Weigang Lu  

**Link**: [PDF](https://arxiv.org/pdf/2510.09394)  

**Abstract**: The "pre-train, prompt'' paradigm, designed to bridge the gap between pre-training tasks and downstream objectives, has been extended from the NLP domain to the graph domain and has achieved remarkable progress. Current mainstream graph prompt-tuning methods modify input or output features using learnable prompt vectors. However, existing approaches are confined to single-granularity (e.g., node-level or subgraph-level) during prompt generation, overlooking the inherently multi-scale structural information in graph data, which limits the diversity of prompt semantics. To address this issue, we pioneer the integration of multi-scale information into graph prompt and propose a Multi-Scale Graph Chain-of-Thought (MSGCOT) prompting framework. Specifically, we design a lightweight, low-rank coarsening network to efficiently capture multi-scale structural features as hierarchical basis vectors for prompt generation. Subsequently, mimicking human cognition from coarse-to-fine granularity, we dynamically integrate multi-scale information at each reasoning step, forming a progressive coarse-to-fine prompt chain. Extensive experiments on eight benchmark datasets demonstrate that MSGCOT outperforms the state-of-the-art single-granularity graph prompt-tuning method, particularly in few-shot scenarios, showcasing superior performance. 

---
# ChoirRec: Semantic User Grouping via LLMs for Conversion Rate Prediction of Low-Activity Users 

**Authors**: Dakai Zhai, Jiong Gao, Boya Du, Junwei Xu, Qijie Shen, Jialin Zhu, Yuning Jiang  

**Link**: [PDF](https://arxiv.org/pdf/2510.09393)  

**Abstract**: Accurately predicting conversion rates (CVR) for low-activity users remains a fundamental challenge in large-scale e-commerce recommender this http URL approaches face three critical limitations: (i) reliance on noisy and unreliable behavioral signals; (ii) insufficient user-level information due to the lack of diverse interaction data; and (iii) a systemic training bias toward high-activity users that overshadows the needs of low-activity this http URL address these challenges, we propose ChoirRec, a novel framework that leverages the semantic capabilities of Large Language Models (LLMs) to construct semantic user groups and enhance CVR prediction for low-activity this http URL a dual-channel architecture designed for robust cross-user knowledge transfer, ChoirRec comprises three components: (i) a Semantic Group Generation module that utilizes LLMs to form reliable, cross-activity user clusters, thereby filtering out noisy signals; (ii) a Group-aware Hierarchical Representation module that enriches sparse user embeddings with informative group-level priors to mitigate data insufficiency; and (iii) a Group-aware Multi-granularity Modual that employs a dual-channel architecture and adaptive fusion mechanism to ensure effective learning and utilization of group knowledge. We conduct extensive offline and online experiments on Taobao, a leading industrial-scale e-commerce this http URL improves GAUC by 1.16\% in offline evaluations, while online A/B testing reveals a 7.24\% increase in order volume, highlighting its substantial practical value in real-world applications. 

---
# Identifying & Interactively Refining Ambiguous User Goals for Data Visualization Code Generation 

**Authors**: Mert İnan, Anthony Sicilia, Alex Xie, Saujas Vaduguru, Daniel Fried, Malihe Alikhani  

**Link**: [PDF](https://arxiv.org/pdf/2510.09390)  

**Abstract**: Establishing shared goals is a fundamental step in human-AI communication. However, ambiguities can lead to outputs that seem correct but fail to reflect the speaker's intent. In this paper, we explore this issue with a focus on the data visualization domain, where ambiguities in natural language impact the generation of code that visualizes data. The availability of multiple views on the contextual (e.g., the intended plot and the code rendering the plot) allows for a unique and comprehensive analysis of diverse ambiguity types. We develop a taxonomy of types of ambiguity that arise in this task and propose metrics to quantify them. Using Matplotlib problems from the DS-1000 dataset, we demonstrate that our ambiguity metrics better correlate with human annotations than uncertainty baselines. Our work also explores how multi-turn dialogue can reduce ambiguity, therefore, improve code accuracy by better matching user goals. We evaluate three pragmatic models to inform our dialogue strategies: Gricean Cooperativity, Discourse Representation Theory, and Questions under Discussion. A simulated user study reveals how pragmatic dialogues reduce ambiguity and enhance code accuracy, highlighting the value of multi-turn exchanges in code generation. 

---
# Design Principles for Sequence Models via Coefficient Dynamics 

**Authors**: Jerome Sieber, Antonio Orvieto, Melanie N. Zeilinger, Carmen Amo Alonso  

**Link**: [PDF](https://arxiv.org/pdf/2510.09389)  

**Abstract**: Deep sequence models, ranging from Transformers and State Space Models (SSMs) to more recent approaches such as gated linear RNNs, fundamentally compute outputs as linear combinations of past value vectors. To draw insights and systematically compare such architectures, we develop a unified framework that makes this output operation explicit, by casting the linear combination coefficients as the outputs of autonomous linear dynamical systems driven by impulse inputs. This viewpoint, in spirit substantially different from approaches focusing on connecting linear RNNs with linear attention, reveals a common mathematical theme across diverse architectures and crucially captures softmax attention, on top of RNNs, SSMs, and related models. In contrast to new model proposals that are commonly evaluated on benchmarks, we derive design principles linking architectural choices to model properties. Thereby identifying tradeoffs between expressivity and efficient implementation, geometric constraints on input selectivity, and stability conditions for numerically stable training and information retention. By connecting several insights and observations from recent literature, the framework both explains empirical successes of recent designs and provides guiding principles for systematically designing new sequence model architectures. 

---
# Task-Level Insights from Eigenvalues across Sequence Models 

**Authors**: Rahel Rickenbach, Jelena Trisovic, Alexandre Didier, Jerome Sieber, Melanie N. Zeilinger  

**Link**: [PDF](https://arxiv.org/pdf/2510.09379)  

**Abstract**: Although softmax attention drives state-of-the-art performance for sequence models, its quadratic complexity limits scalability, motivating linear alternatives such as state space models (SSMs). While these alternatives improve efficiency, their fundamental differences in information processing remain poorly understood. In this work, we leverage the recently proposed dynamical systems framework to represent softmax, norm and linear attention as dynamical systems, enabling a structured comparison with SSMs by analyzing their respective eigenvalue spectra. Since eigenvalues capture essential aspects of dynamical system behavior, we conduct an extensive empirical analysis across diverse sequence models and benchmarks. We first show that eigenvalues influence essential aspects of memory and long-range dependency modeling, revealing spectral signatures that align with task requirements. Building on these insights, we then investigate how architectural modifications in sequence models impact both eigenvalue spectra and task performance. This correspondence further strengthens the position of eigenvalue analysis as a principled metric for interpreting, understanding, and ultimately improving the capabilities of sequence models. 

---
# The Potential of Second-Order Optimization for LLMs: A Study with Full Gauss-Newton 

**Authors**: Natalie Abreu, Nikhil Vyas, Sham Kakade, Depen Morwani  

**Link**: [PDF](https://arxiv.org/pdf/2510.09378)  

**Abstract**: Recent efforts to accelerate LLM pretraining have focused on computationally-efficient approximations that exploit second-order structure. This raises a key question for large-scale training: how much performance is forfeited by these approximations? To probe this question, we establish a practical upper bound on iteration complexity by applying full Gauss-Newton (GN) preconditioning to transformer models of up to 150M parameters. Our experiments show that full GN updates yield substantial gains over existing optimizers, achieving a 5.4x reduction in training iterations compared to strong baselines like SOAP and Muon. Furthermore, we find that a precise layerwise GN preconditioner, which ignores cross-layer information, nearly matches the performance of the full GN method. Collectively, our results suggest: (1) the GN approximation is highly effective for preconditioning, implying higher-order loss terms may not be critical for convergence speed; (2) the layerwise Hessian structure contains sufficient information to achieve most of these potential gains; and (3) a significant performance gap exists between current approximate methods and an idealized layerwise oracle. 

---
# deep-REMAP: Probabilistic Parameterization of Stellar Spectra Using Regularized Multi-Task Learning 

**Authors**: Sankalp Gilda  

**Link**: [PDF](https://arxiv.org/pdf/2510.09362)  

**Abstract**: In the era of exploding survey volumes, traditional methods of spectroscopic analysis are being pushed to their limits. In response, we develop deep-REMAP, a novel deep learning framework that utilizes a regularized, multi-task approach to predict stellar atmospheric parameters from observed spectra. We train a deep convolutional neural network on the PHOENIX synthetic spectral library and use transfer learning to fine-tune the model on a small subset of observed FGK dwarf spectra from the MARVELS survey. We then apply the model to 732 uncharacterized FGK giant candidates from the same survey. When validated on 30 MARVELS calibration stars, deep-REMAP accurately recovers the effective temperature ($T_{\rm{eff}}$), surface gravity ($\log \rm{g}$), and metallicity ([Fe/H]), achieving a precision of, for instance, approximately 75 K in $T_{\rm{eff}}$. By combining an asymmetric loss function with an embedding loss, our regression-as-classification framework is interpretable, robust to parameter imbalances, and capable of capturing non-Gaussian uncertainties. While developed for MARVELS, the deep-REMAP framework is extensible to other surveys and synthetic libraries, demonstrating a powerful and automated pathway for stellar characterization. 

---
# FLRC: Fine-grained Low-Rank Compressor for Efficient LLM Inference 

**Authors**: Yu-Chen Lu, Chong-Yan Chen, Chi-Chih Chang, Yu-Fang Hu, Kai-Chiang Wu  

**Link**: [PDF](https://arxiv.org/pdf/2510.09332)  

**Abstract**: Although large language models (LLM) have achieved remarkable performance, their enormous parameter counts hinder deployment on resource-constrained hardware. Low-rank compression can reduce both memory usage and computational demand, but applying a uniform compression ratio across all layers often leads to significant performance degradation, and previous methods perform poorly during decoding. To address these issues, we propose the Fine-grained Low-Rank Compressor (FLRC), which efficiently determines an optimal rank allocation for each layer, and incorporates progressive low-rank decoding to maintain text generation quality. Comprehensive experiments on diverse benchmarks demonstrate the superiority of FLRC, achieving up to a 17% improvement in ROUGE-L on summarization tasks compared to state-of-the-art low-rank compression methods, establishing a more robust and efficient framework to improve LLM inference. 

---
# Randomized HyperSteiner: A Stochastic Delaunay Triangulation Heuristic for the Hyperbolic Steiner Minimal Tree 

**Authors**: Aniss Aiman Medbouhi, Alejandro García-Castellanos, Giovanni Luca Marchetti, Daniel Pelt, Erik J Bekkers, Danica Kragic  

**Link**: [PDF](https://arxiv.org/pdf/2510.09328)  

**Abstract**: We study the problem of constructing Steiner Minimal Trees (SMTs) in hyperbolic space. Exact SMT computation is NP-hard, and existing hyperbolic heuristics such as HyperSteiner are deterministic and often get trapped in locally suboptimal configurations. We introduce Randomized HyperSteiner (RHS), a stochastic Delaunay triangulation heuristic that incorporates randomness into the expansion process and refines candidate trees via Riemannian gradient descent optimization. Experiments on synthetic data sets and a real-world single-cell transcriptomic data show that RHS outperforms Minimum Spanning Tree (MST), Neighbour Joining, and vanilla HyperSteiner (HS). In near-boundary configurations, RHS can achieve a 32% reduction in total length over HS, demonstrating its effectiveness and robustness in diverse data regimes. 

---
# Rate optimal learning of equilibria from data 

**Authors**: Till Freihaut, Luca Viano, Emanuele Nevali, Volkan Cevher, Matthieu Geist, Giorgia Ramponi  

**Link**: [PDF](https://arxiv.org/pdf/2510.09325)  

**Abstract**: We close open theoretical gaps in Multi-Agent Imitation Learning (MAIL) by characterizing the limits of non-interactive MAIL and presenting the first interactive algorithm with near-optimal sample complexity. In the non-interactive setting, we prove a statistical lower bound that identifies the all-policy deviation concentrability coefficient as the fundamental complexity measure, and we show that Behavior Cloning (BC) is rate-optimal. For the interactive setting, we introduce a framework that combines reward-free reinforcement learning with interactive MAIL and instantiate it with an algorithm, MAIL-WARM. It improves the best previously known sample complexity from $\mathcal{O}(\varepsilon^{-8})$ to $\mathcal{O}(\varepsilon^{-2}),$ matching the dependence on $\varepsilon$ implied by our lower bound. Finally, we provide numerical results that support our theory and illustrate, in environments such as grid worlds, where Behavior Cloning fails to learn. 

---
# Verifying Chain-of-Thought Reasoning via Its Computational Graph 

**Authors**: Zheng Zhao, Yeskendir Koishekenov, Xianjun Yang, Naila Murray, Nicola Cancedda  

**Link**: [PDF](https://arxiv.org/pdf/2510.09312)  

**Abstract**: Current Chain-of-Thought (CoT) verification methods predict reasoning correctness based on outputs (black-box) or activations (gray-box), but offer limited insight into why a computation fails. We introduce a white-box method: Circuit-based Reasoning Verification (CRV). We hypothesize that attribution graphs of correct CoT steps, viewed as execution traces of the model's latent reasoning circuits, possess distinct structural fingerprints from those of incorrect steps. By training a classifier on structural features of these graphs, we show that these traces contain a powerful signal of reasoning errors. Our white-box approach yields novel scientific insights unattainable by other methods. (1) We demonstrate that structural signatures of error are highly predictive, establishing the viability of verifying reasoning directly via its computational graph. (2) We find these signatures to be highly domain-specific, revealing that failures in different reasoning tasks manifest as distinct computational patterns. (3) We provide evidence that these signatures are not merely correlational; by using our analysis to guide targeted interventions on individual transcoder features, we successfully correct the model's faulty reasoning. Our work shows that, by scrutinizing a model's computational process, we can move from simple error detection to a deeper, causal understanding of LLM reasoning. 

---
# A Model-Driven Engineering Approach to AI-Powered Healthcare Platforms 

**Authors**: Mira Raheem, Amal Elgammal, Michael Papazoglou, Bernd Krämer, Neamat El-Tazi  

**Link**: [PDF](https://arxiv.org/pdf/2510.09308)  

**Abstract**: Artificial intelligence (AI) has the potential to transform healthcare by supporting more accurate diagnoses and personalized treatments. However, its adoption in practice remains constrained by fragmented data sources, strict privacy rules, and the technical complexity of building reliable clinical systems. To address these challenges, we introduce a model driven engineering (MDE) framework designed specifically for healthcare AI. The framework relies on formal metamodels, domain-specific languages (DSLs), and automated transformations to move from high level specifications to running software. At its core is the Medical Interoperability Language (MILA), a graphical DSL that enables clinicians and data scientists to define queries and machine learning pipelines using shared ontologies. When combined with a federated learning architecture, MILA allows institutions to collaborate without exchanging raw patient data, ensuring semantic consistency across sites while preserving privacy. We evaluate this approach in a multi center cancer immunotherapy study. The generated pipelines delivered strong predictive performance, with support vector machines achieving up to 98.5 percent and 98.3 percent accuracy in key tasks, while substantially reducing manual coding effort. These findings suggest that MDE principles metamodeling, semantic integration, and automated code generation can provide a practical path toward interoperable, reproducible, and trustworthy digital health platforms. 

---
# CapGeo: A Caption-Assisted Approach to Geometric Reasoning 

**Authors**: Yuying Li, Siyi Qian, Hao Liang, Leqi Zheng, Ruichuan An, Yongzhen Guo, Wentao Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2510.09302)  

**Abstract**: Geometric reasoning remains a core challenge for Multimodal Large Language Models (MLLMs). Even the most advanced closed-source systems, such as GPT-O3 and Gemini-2.5-Pro, still struggle to solve geometry problems reliably, despite exhibiting strong textual reasoning abilities on tasks like the International Mathematical Olympiad (IMO). This gap suggests that the bottleneck lies in understanding geometric diagrams rather than reasoning itself. Since geometric figures can often be faithfully described in concise textual form, converting visual content into captions offers a promising direction. Motivated by this insight, we introduce CapGeo, a caption-assisted reasoning framework that bridges visual and textual modalities. Experiments show substantial improvements when models are equipped with captions: Qwen2.5-VL-72B improves from 8.6% (vision-only) to 59.0%, while Claude-Opus-4 rises from 44.8% to 73.0%. To systematically evaluate and identify high-quality geometric captioning models, we further propose CapGeo-Bench, a dataset of 4,641 curated figure-caption pairs. Crucially, CapGeo-Bench incorporates a keypoint-based evaluation metric that correlates strongly with downstream CapGeo performance, enabling reliable assessment of geometric captioning ability. Together, our framework and benchmark highlight a new pathway toward advancing geometric reasoning in MLLMs. 

---
# CLARity: Reasoning Consistency Alone Can Teach Reinforced Experts 

**Authors**: Jiuheng Lin, Cong Jiang, Zirui Wu, Jiarui Sun, Yansong Feng  

**Link**: [PDF](https://arxiv.org/pdf/2510.09278)  

**Abstract**: Training expert LLMs in domains with scarce data is difficult, often relying on multiple-choice questions (MCQs). However, standard outcome-based reinforcement learning (RL) on MCQs is risky. While it may improve accuracy, we observe it often degrades reasoning quality such as logical consistency. Existing solutions to supervise reasoning, such as large-scale Process Reward Models (PRMs), are prohibitively expensive. To address this, we propose CLARity, a cost-effective RL framework that enhances reasoning quality using only a small, general-purpose LLM. CLARity integrates a consistency-aware reward mechanism with a 2-stage refine-then-monitor training pipeline to enhance reasoning consistency, and a dynamic data reformulation strategy to to better exploit limited data. Experiments demonstrate that CLARity improves response consistency by 16.5% and accuracy by 7.5% over baselines. Human evaluations further confirm holistic improvements in coherence and professionalism. Thus, CLARity offers a generalizable solution that enables smaller models to effectively guide expert models by reasoning this http URL code is open sourced at: this https URL 

---
# Inflated Excellence or True Performance? Rethinking Medical Diagnostic Benchmarks with Dynamic Evaluation 

**Authors**: Xiangxu Zhang, Lei Li, Yanyun Zhou, Xiao Zhou, Yingying Zhang, Xian Wu  

**Link**: [PDF](https://arxiv.org/pdf/2510.09275)  

**Abstract**: Medical diagnostics is a high-stakes and complex domain that is critical to patient care. However, current evaluations of large language models (LLMs) are fundamentally misaligned with real-world clinical practice. Most of them rely on static benchmarks derived from public medical exam items, which tend to overestimate model performance and ignore the difference between textbook cases and the ambiguous, varying conditions in the real world. Recent efforts toward dynamic evaluation offer a promising alternative, but their improvements are limited to superficial perturbations and a narrow focus on accuracy. To address these gaps, we propose DyReMe, a dynamic benchmark for medical diagnostics that better reflects real clinical practice. Unlike static exam-style questions, DyReMe generates fresh, consultation-like cases that introduce distractors such as differential diagnoses and common misdiagnosis factors. It also varies expression styles to mimic diverse real-world query habits. Beyond accuracy, DyReMe evaluates LLMs on three additional clinically relevant dimensions: veracity, helpfulness, and consistency. Our experiments demonstrate that this dynamic approach yields more challenging and realistic assessments, revealing significant misalignments between the performance of state-of-the-art LLMs and real clinical practice. These findings highlight the urgent need for evaluation frameworks that better reflect the demands of trustworthy medical diagnostics. 

---
# SynthID-Image: Image watermarking at internet scale 

**Authors**: Sven Gowal, Rudy Bunel, Florian Stimberg, David Stutz, Guillermo Ortiz-Jimenez, Christina Kouridi, Mel Vecerik, Jamie Hayes, Sylvestre-Alvise Rebuffi, Paul Bernard, Chris Gamble, Miklós Z. Horváth, Fabian Kaczmarczyck, Alex Kaskasoli, Aleksandar Petrov, Ilia Shumailov, Meghana Thotakuri, Olivia Wiles, Jessica Yung, Zahra Ahmed, Victor Martin, Simon Rosen, Christopher Savčak, Armin Senoner, Nidhi Vyas, Pushmeet Kohli  

**Link**: [PDF](https://arxiv.org/pdf/2510.09263)  

**Abstract**: We introduce SynthID-Image, a deep learning-based system for invisibly watermarking AI-generated imagery. This paper documents the technical desiderata, threat models, and practical challenges of deploying such a system at internet scale, addressing key requirements of effectiveness, fidelity, robustness, and security. SynthID-Image has been used to watermark over ten billion images and video frames across Google's services and its corresponding verification service is available to trusted testers. For completeness, we present an experimental evaluation of an external model variant, SynthID-O, which is available through partnerships. We benchmark SynthID-O against other post-hoc watermarking methods from the literature, demonstrating state-of-the-art performance in both visual quality and robustness to common image perturbations. While this work centers on visual media, the conclusions on deployment, constraints, and threat modeling generalize to other modalities, including audio. This paper provides a comprehensive documentation for the large-scale deployment of deep learning-based media provenance systems. 

---
# Detecting Data Contamination from Reinforcement Learning Post-training for Large Language Models 

**Authors**: Yongding Tao, Tian Wang, Yihong Dong, Huanyu Liu, Kechi Zhang, Xiaolong Hu, Ge Li  

**Link**: [PDF](https://arxiv.org/pdf/2510.09259)  

**Abstract**: Data contamination poses a significant threat to the reliable evaluation of Large Language Models (LLMs). This issue arises when benchmark samples may inadvertently appear in training sets, compromising the validity of reported performance. While detection methods have been developed for the pre-training and Supervised Fine-Tuning stages, a critical research gap exists for the increasingly significant phase of Reinforcement Learning (RL) post-training. As RL post-training becomes pivotal for advancing LLM reasoning, the absence of specialized contamination detection methods in this paradigm presents a critical vulnerability. To address this, we conduct the first systematic study of data detection within RL post-training scenario and propose Self-Critique. Our method is motivated by a key observation: after RL phase, the output entropy distribution of LLMs tends to collapse into highly specific and sparse modes. Self-Critique probes for the underlying policy collapse, i.e., the model's convergence to a narrow reasoning path, which causes this entropy reduction. To facilitate this research, we also introduce RL-MIA, a benchmark constructed to simulate this specific contamination scenario. Extensive experiments show that Self-Critique significantly outperforms baseline methods across multiple models and contamination tasks, achieving an AUC improvement of up to 30%. Whereas existing methods are close to a random guess for RL-phase contamination, our method makes detection possible. 

---
# Obstacle Avoidance using Dynamic Movement Primitives and Reinforcement Learning 

**Authors**: Dominik Urbaniak, Alejandro Agostini, Pol Ramon, Jan Rosell, Raúl Suárez, Michael Suppa  

**Link**: [PDF](https://arxiv.org/pdf/2510.09254)  

**Abstract**: Learning-based motion planning can quickly generate near-optimal trajectories. However, it often requires either large training datasets or costly collection of human demonstrations. This work proposes an alternative approach that quickly generates smooth, near-optimal collision-free 3D Cartesian trajectories from a single artificial demonstration. The demonstration is encoded as a Dynamic Movement Primitive (DMP) and iteratively reshaped using policy-based reinforcement learning to create a diverse trajectory dataset for varying obstacle configurations. This dataset is used to train a neural network that takes as inputs the task parameters describing the obstacle dimensions and location, derived automatically from a point cloud, and outputs the DMP parameters that generate the trajectory. The approach is validated in simulation and real-robot experiments, outperforming a RRT-Connect baseline in terms of computation and execution time, as well as trajectory length, while supporting multi-modal trajectory generation for different obstacle geometries and end-effector dimensions. Videos and the implementation code are available at this https URL. 

---
# CrisiText: A dataset of warning messages for LLM training in emergency communication 

**Authors**: Giacomo Gonella, Gian Maria Campedelli, Stefano Menini, Marco Guerini  

**Link**: [PDF](https://arxiv.org/pdf/2510.09243)  

**Abstract**: Effectively identifying threats and mitigating their potential damage during crisis situations, such as natural disasters or violent attacks, is paramount for safeguarding endangered individuals. To tackle these challenges, AI has been used in assisting humans in emergency situations. Still, the use of NLP techniques remains limited and mostly focuses on classification tasks. The significant potential of timely warning message generation using NLG architectures, however, has been largely overlooked. In this paper we present CrisiText, the first large-scale dataset for the generation of warning messages across 13 different types of crisis scenarios. The dataset contains more than 400,000 warning messages (spanning almost 18,000 crisis situations) aimed at assisting civilians during and after such events. To generate the dataset, we started from existing crisis descriptions and created chains of events related to the scenarios. Each event was then paired with a warning message. The generations follow experts' written guidelines to ensure correct terminology and factuality of their suggestions. Additionally, each message is accompanied by three suboptimal warning types to allow for the study of different NLG approaches. To this end, we conducted a series of experiments comparing supervised fine-tuning setups with preference alignment, zero-shot, and few-shot approaches. We further assessed model performance in out-of-distribution scenarios and evaluated the effectiveness of an automatic post-editor. 

---
# Diagnosing Shoulder Disorders Using Multimodal Large Language Models and Consumer-Grade Cameras 

**Authors**: Jindong Hong, Wencheng Zhang, Shiqin Qiao, Jianhai Chen, Jianing Qiu, Chuanyang Zheng, Qian Xu, Yun Ji, Qianyue Wen, Weiwei Sun, Hao Li, Huizhen Li, Huichao Wang, Kai Wu, Meng Li, Yijun He, Lingjie Luo, Jiankai Sun  

**Link**: [PDF](https://arxiv.org/pdf/2510.09230)  

**Abstract**: Shoulder disorders, such as frozen shoulder (a.k.a., adhesive capsulitis), are common conditions affecting the health of people worldwide, and have a high incidence rate among the elderly and workers engaged in repetitive shoulder tasks. In regions with scarce medical resources, achieving early and accurate diagnosis poses significant challenges, and there is an urgent need for low-cost and easily scalable auxiliary diagnostic solutions. This research introduces videos captured by consumer-grade devices as the basis for diagnosis, reducing the cost for users. We focus on the innovative application of Multimodal Large Language Models (MLLMs) in the preliminary diagnosis of shoulder disorders and propose a Hybrid Motion Video Diagnosis framework (HMVDx). This framework divides the two tasks of action understanding and disease diagnosis, which are respectively completed by two MLLMs. In addition to traditional evaluation indicators, this work proposes a novel metric called Usability Index by the logical process of medical decision-making (action recognition, movement diagnosis, and final diagnosis). This index evaluates the effectiveness of MLLMs in the medical field from the perspective of the entire medical diagnostic pathway, revealing the potential value of low-cost MLLMs in medical applications for medical practitioners. In experimental comparisons, the accuracy of HMVDx in diagnosing shoulder joint injuries has increased by 79.6\% compared with direct video diagnosis, a significant technical contribution to future research on the application of MLLMs for video understanding in the medical field. 

---
# Clear Roads, Clear Vision: Advancements in Multi-Weather Restoration for Smart Transportation 

**Authors**: Vijay M. Galshetwar, Praful Hambarde, Prashant W. Patil, Akshay Dudhane, Sachin Chaudhary, Santosh Kumar Vipparathi, Subrahmanyam Murala  

**Link**: [PDF](https://arxiv.org/pdf/2510.09228)  

**Abstract**: Adverse weather conditions such as haze, rain, and snow significantly degrade the quality of images and videos, posing serious challenges to intelligent transportation systems (ITS) that rely on visual input. These degradations affect critical applications including autonomous driving, traffic monitoring, and surveillance. This survey presents a comprehensive review of image and video restoration techniques developed to mitigate weather-induced visual impairments. We categorize existing approaches into traditional prior-based methods and modern data-driven models, including CNNs, transformers, diffusion models, and emerging vision-language models (VLMs). Restoration strategies are further classified based on their scope: single-task models, multi-task/multi-weather systems, and all-in-one frameworks capable of handling diverse degradations. In addition, we discuss day and night time restoration challenges, benchmark datasets, and evaluation protocols. The survey concludes with an in-depth discussion on limitations in current research and outlines future directions such as mixed/compound-degradation restoration, real-time deployment, and agentic AI frameworks. This work aims to serve as a valuable reference for advancing weather-resilient vision systems in smart transportation environments. Lastly, to stay current with rapid advancements in this field, we will maintain regular updates of the latest relevant papers and their open-source implementations at this https URL 

---
# DICE: Structured Reasoning in LLMs through SLM-Guided Chain-of-Thought Correction 

**Authors**: Yiqi Li, Yusheng Liao, Zhe Chen, Yanfeng Wang, Yu Wang  

**Link**: [PDF](https://arxiv.org/pdf/2510.09211)  

**Abstract**: When performing reasoning tasks with user-specific requirements, such as strict output formats, large language models (LLMs) often prioritize reasoning over adherence to detailed instructions. Fine-tuning LLMs on supervised datasets to address this is impractical due to high computational costs and limited parameter access. To tackle this, we propose DICE, a lightweight framework that guides small language models (SLMs) to refine LLMs' outputs through chain-of-thought (CoT) correction. DICE decouples the process by first prompting LLMs to generate natural language responses, then using trained SLMs to analyze and refine these outputs to meet structured output specifications. This framework preserves LLMs' broad knowledge and reasoning capabilities while ensuring the outputs conform to user demands. Specifically, DICE first constructs structured CoT adaptation datasets via a two-stage method and subsequently applies a dual-tuning strategy to fine-tune SLMs for generating structured outputs in an analyze-then-answer pattern. Experiments demonstrate that DICE improves the average format accuracy and content correctness of LLM outputs by 35.4\% and 29.4\%, respectively, achieving state-of-the-art (SOTA) performance over other competitive baselines. 

---
# Multimodal Prompt Optimization: Why Not Leverage Multiple Modalities for MLLMs 

**Authors**: Yumin Choi, Dongki Kim, Jinheon Baek, Sung Ju Hwang  

**Link**: [PDF](https://arxiv.org/pdf/2510.09201)  

**Abstract**: Large Language Models (LLMs) have shown remarkable success, and their multimodal expansions (MLLMs) further unlock capabilities spanning images, videos, and other modalities beyond text. However, despite this shift, prompt optimization approaches, designed to reduce the burden of manual prompt crafting while maximizing performance, remain confined to text, ultimately limiting the full potential of MLLMs. Motivated by this gap, we introduce the new problem of multimodal prompt optimization, which expands the prior definition of prompt optimization to the multimodal space defined by the pairs of textual and non-textual prompts. To tackle this problem, we then propose the Multimodal Prompt Optimizer (MPO), a unified framework that not only performs the joint optimization of multimodal prompts through alignment-preserving updates but also guides the selection process of candidate prompts by leveraging earlier evaluations as priors in a Bayesian-based selection strategy. Through extensive experiments across diverse modalities that go beyond text, such as images, videos, and even molecules, we demonstrate that MPO outperforms leading text-only optimization methods, establishing multimodal prompt optimization as a crucial step to realizing the potential of MLLMs. 

---
# Towards Safer and Understandable Driver Intention Prediction 

**Authors**: Mukilan Karuppasamy, Shankar Gangisetty, Shyam Nandan Rai, Carlo Masone, C V Jawahar  

**Link**: [PDF](https://arxiv.org/pdf/2510.09200)  

**Abstract**: Autonomous driving (AD) systems are becoming increasingly capable of handling complex tasks, mainly due to recent advances in deep learning and AI. As interactions between autonomous systems and humans increase, the interpretability of decision-making processes in driving systems becomes increasingly crucial for ensuring safe driving operations. Successful human-machine interaction requires understanding the underlying representations of the environment and the driving task, which remains a significant challenge in deep learning-based systems. To address this, we introduce the task of interpretability in maneuver prediction before they occur for driver safety, i.e., driver intent prediction (DIP), which plays a critical role in AD systems. To foster research in interpretable DIP, we curate the eXplainable Driving Action Anticipation Dataset (DAAD-X), a new multimodal, ego-centric video dataset to provide hierarchical, high-level textual explanations as causal reasoning for the driver's decisions. These explanations are derived from both the driver's eye-gaze and the ego-vehicle's perspective. Next, we propose Video Concept Bottleneck Model (VCBM), a framework that generates spatio-temporally coherent explanations inherently, without relying on post-hoc techniques. Finally, through extensive evaluations of the proposed VCBM on the DAAD-X dataset, we demonstrate that transformer-based models exhibit greater interpretability than conventional CNN-based models. Additionally, we introduce a multilabel t-SNE visualization technique to illustrate the disentanglement and causal correlation among multiple explanations. Our data, code and models are available at: this https URL 

---
# Modern Deep Learning Approaches for Cricket Shot Classification: A Comprehensive Baseline Study 

**Authors**: Sungwoo Kang  

**Link**: [PDF](https://arxiv.org/pdf/2510.09187)  

**Abstract**: Cricket shot classification from video sequences remains a challenging problem in sports video analysis, requiring effective modeling of both spatial and temporal features. This paper presents the first comprehensive baseline study comparing seven different deep learning approaches across four distinct research paradigms for cricket shot classification. We implement and systematically evaluate traditional CNN-LSTM architectures, attention-based models, vision transformers, transfer learning approaches, and modern EfficientNet-GRU combinations on a unified benchmark. A critical finding of our study is the significant performance gap between claims in academic literature and practical implementation results. While previous papers reported accuracies of 96\% (Balaji LRCN), 99.2\% (IJERCSE), and 93\% (Sensors), our standardized re-implementations achieve 46.0\%, 55.6\%, and 57.7\% respectively. Our modern SOTA approach, combining EfficientNet-B0 with a GRU-based temporal model, achieves 92.25\% accuracy, demonstrating that substantial improvements are possible with modern architectures and systematic optimization. All implementations follow modern MLOps practices with PyTorch Lightning, providing a reproducible research platform that exposes the critical importance of standardized evaluation protocols in sports video analysis research. 

---
# On the Implicit Adversariality of Catastrophic Forgetting in Deep Continual Learning 

**Authors**: Ze Peng, Jian Zhang, Jintao Guo, Lei Qi, Yang Gao, Yinghuan Shi  

**Link**: [PDF](https://arxiv.org/pdf/2510.09181)  

**Abstract**: Continual learning seeks the human-like ability to accumulate new skills in machine intelligence. Its central challenge is catastrophic forgetting, whose underlying cause has not been fully understood for deep networks. In this paper, we demystify catastrophic forgetting by revealing that the new-task training is implicitly an adversarial attack against the old-task knowledge. Specifically, the new-task gradients automatically and accurately align with the sharp directions of the old-task loss landscape, rapidly increasing the old-task loss. This adversarial alignment is intriguingly counter-intuitive because the sharp directions are too sparsely distributed to align with by chance. To understand it, we theoretically show that it arises from training's low-rank bias, which, through forward and backward propagation, confines the two directions into the same low-dimensional subspace, facilitating alignment. Gradient projection (GP) methods, a representative family of forgetting-mitigating methods, reduce adversarial alignment caused by forward propagation, but cannot address the alignment due to backward propagation. We propose backGP to address it, which reduces forgetting by 10.8% and improves accuracy by 12.7% on average over GP methods. 

---
# Cross-Representation Benchmarking in Time-Series Electronic Health Records for Clinical Outcome Prediction 

**Authors**: Tianyi Chen, Mingcheng Zhu, Zhiyao Luo, Tingting Zhu  

**Link**: [PDF](https://arxiv.org/pdf/2510.09159)  

**Abstract**: Electronic Health Records (EHRs) enable deep learning for clinical predictions, but the optimal method for representing patient data remains unclear due to inconsistent evaluation practices. We present the first systematic benchmark to compare EHR representation methods, including multivariate time-series, event streams, and textual event streams for LLMs. This benchmark standardises data curation and evaluation across two distinct clinical settings: the MIMIC-IV dataset for ICU tasks (mortality, phenotyping) and the EHRSHOT dataset for longitudinal care (30-day readmission, 1-year pancreatic cancer). For each paradigm, we evaluate appropriate modelling families--including Transformers, MLP, LSTMs and Retain for time-series, CLMBR and count-based models for event streams, 8-20B LLMs for textual streams--and analyse the impact of feature pruning based on data missingness. Our experiments reveal that event stream models consistently deliver the strongest performance. Pre-trained models like CLMBR are highly sample-efficient in few-shot settings, though simpler count-based models can be competitive given sufficient data. Furthermore, we find that feature selection strategies must be adapted to the clinical setting: pruning sparse features improves ICU predictions, while retaining them is critical for longitudinal tasks. Our results, enabled by a unified and reproducible pipeline, provide practical guidance for selecting EHR representations based on the clinical context and data regime. 

---
# Federated Data Analytics for Cancer Immunotherapy: A Privacy-Preserving Collaborative Platform for Patient Management 

**Authors**: Mira Raheem, Michael Papazoglou, Bernd Krämer, Neamat El-Tazi, Amal Elgammal  

**Link**: [PDF](https://arxiv.org/pdf/2510.09155)  

**Abstract**: Connected health is a multidisciplinary approach focused on health management, prioritizing pa-tient needs in the creation of tools, services, and treatments. This paradigm ensures proactive and efficient care by facilitating the timely exchange of accurate patient information among all stake-holders in the care continuum. The rise of digital technologies and process innovations promises to enhance connected health by integrating various healthcare data sources. This integration aims to personalize care, predict health outcomes, and streamline patient management, though challeng-es remain, particularly in data architecture, application interoperability, and security. Data analytics can provide critical insights for informed decision-making and health co-creation, but solutions must prioritize end-users, including patients and healthcare professionals. This perspective was explored through an agile System Development Lifecycle in an EU-funded project aimed at developing an integrated AI-generated solution for managing cancer patients undergoing immunotherapy. This paper contributes with a collaborative digital framework integrating stakeholders across the care continuum, leveraging federated big data analytics and artificial intelligence for improved decision-making while ensuring privacy. Analytical capabilities, such as treatment recommendations and adverse event predictions, were validated using real-life data, achieving 70%-90% accuracy in a pilot study with the medical partners, demonstrating the framework's effectiveness. 

---
# Controlled Personalization in Legacy Media Online Services: A Case Study in News Recommendation 

**Authors**: Marlene Holzleitner, Stephan Leitner, Hanna Lind Jorgensen, Christoph Schmitz, Jacob Welander, Dietmar Jannach  

**Link**: [PDF](https://arxiv.org/pdf/2510.09136)  

**Abstract**: Personalized news recommendations have become a standard feature of large news aggregation services, optimizing user engagement through automated content selection. In contrast, legacy news media often approach personalization cautiously, striving to balance technological innovation with core editorial values. As a result, online platforms of traditional news outlets typically combine editorially curated content with algorithmically selected articles - a strategy we term controlled personalization. In this industry paper, we evaluate the effectiveness of controlled personalization through an A/B test conducted on the website of a major Norwegian legacy news organization. Our findings indicate that even a modest level of personalization yields substantial benefits. Specifically, we observe that users exposed to personalized content demonstrate higher click-through rates and reduced navigation effort, suggesting improved discovery of relevant content. Moreover, our analysis reveals that controlled personalization contributes to greater content diversity and catalog coverage and in addition reduces popularity bias. Overall, our results suggest that controlled personalization can successfully align user needs with editorial goals, offering a viable path for legacy media to adopt personalization technologies while upholding journalistic values. 

---
# MSDM: Generating Task-Specific Pathology Images with a Multimodal Conditioned Diffusion Model for Cell and Nuclei Segmentation 

**Authors**: Dominik Winter, Mai Bui, Monica Azqueta Gavaldon, Nicolas Triltsch, Marco Rosati, Nicolas Brieu  

**Link**: [PDF](https://arxiv.org/pdf/2510.09121)  

**Abstract**: Scarcity of annotated data, particularly for rare or atypical morphologies, present significant challenges for cell and nuclei segmentation in computational pathology. While manual annotation is labor-intensive and costly, synthetic data offers a cost-effective alternative. We introduce a Multimodal Semantic Diffusion Model (MSDM) for generating realistic pixel-precise image-mask pairs for cell and nuclei segmentation. By conditioning the generative process with cellular/nuclear morphologies (using horizontal and vertical maps), RGB color characteristics, and BERT-encoded assay/indication metadata, MSDM generates datasests with desired morphological properties. These heterogeneous modalities are integrated via multi-head cross-attention, enabling fine-grained control over the generated images. Quantitative analysis demonstrates that synthetic images closely match real data, with low Wasserstein distances between embeddings of generated and real images under matching biological conditions. The incorporation of these synthetic samples, exemplified by columnar cells, significantly improves segmentation model accuracy on columnar cells. This strategy systematically enriches data sets, directly targeting model deficiencies. We highlight the effectiveness of multimodal diffusion-based augmentation for advancing the robustness and generalizability of cell and nuclei segmentation models. Thereby, we pave the way for broader application of generative models in computational pathology. 

---
# On the Fairness of Privacy Protection: Measuring and Mitigating the Disparity of Group Privacy Risks for Differentially Private Machine Learning 

**Authors**: Zhi Yang, Changwu Huang, Ke Tang, Xin Yao  

**Link**: [PDF](https://arxiv.org/pdf/2510.09114)  

**Abstract**: While significant progress has been made in conventional fairness-aware machine learning (ML) and differentially private ML (DPML), the fairness of privacy protection across groups remains underexplored. Existing studies have proposed methods to assess group privacy risks, but these are based on the average-case privacy risks of data records. Such approaches may underestimate the group privacy risks, thereby potentially underestimating the disparity across group privacy risks. Moreover, the current method for assessing the worst-case privacy risks of data records is time-consuming, limiting their practical applicability. To address these limitations, we introduce a novel membership inference game that can efficiently audit the approximate worst-case privacy risks of data records. Experimental results demonstrate that our method provides a more stringent measurement of group privacy risks, yielding a reliable assessment of the disparity in group privacy risks. Furthermore, to promote privacy protection fairness in DPML, we enhance the standard DP-SGD algorithm with an adaptive group-specific gradient clipping strategy, inspired by the design of canaries in differential privacy auditing studies. Extensive experiments confirm that our algorithm effectively reduces the disparity in group privacy risks, thereby enhancing the fairness of privacy protection in DPML. 

---
# SOS: Synthetic Object Segments Improve Detection, Segmentation, and Grounding 

**Authors**: Weikai Huang, Jieyu Zhang, Taoyang Jia, Chenhao Zheng, Ziqi Gao, Jae Sung Park, Ranjay Krishna  

**Link**: [PDF](https://arxiv.org/pdf/2510.09110)  

**Abstract**: Visual grouping -- operationalized via instance segmentation, visual grounding, and object detection -- underpins applications from robotic perception to photo editing. Large annotated datasets are costly, biased in coverage, and hard to scale. Synthetic data are promising but often lack flexibility, accuracy, and compositional diversity.
We present SOS, a simple and scalable data synthesis pipeline based on an object-centric composition strategy. It pastes high-quality synthetic object segments into new images using structured layout priors and generative relighting, producing accurate and diverse masks, boxes, and referring expressions. Models trained on 100000 synthetic images from SOS outperform those trained on larger real-image datasets such as GRIT (20M) and V3Det (200K) on detection and grounding tasks, achieving +10.9 AP on LVIS detection and +8.4 $N_{\text{Acc}}$ on gRefCOCO grounding. SOS enables controllable dataset construction and improves generalization in both low-data and closed-vocabulary settings. Augmenting LVIS and COCO with synthetic object segments yields strong performance across real-data scales and even larger gains under extremely limited real data (for example, +3.83 $AP_{\text{rare}}$ on LVIS instance segmentation and +6.59 AP with a 1 percent COCO setup). This controllability also supports targeted data generation for challenging intra-class referring in visual grounding. 

---
# MemLoss: Enhancing Adversarial Training with Recycling Adversarial Examples 

**Authors**: Soroush Mahdi, Maryam Amirmazlaghani, Saeed Saravani, Zahra Dehghanian  

**Link**: [PDF](https://arxiv.org/pdf/2510.09105)  

**Abstract**: In this paper, we propose a new approach called MemLoss to improve the adversarial training of machine learning models. MemLoss leverages previously generated adversarial examples, referred to as 'Memory Adversarial Examples,' to enhance model robustness and accuracy without compromising performance on clean data. By using these examples across training epochs, MemLoss provides a balanced improvement in both natural accuracy and adversarial robustness. Experimental results on multiple datasets, including CIFAR-10, demonstrate that our method achieves better accuracy compared to existing adversarial training methods while maintaining strong robustness against attacks. 

---
# When a Robot is More Capable than a Human: Learning from Constrained Demonstrators 

**Authors**: Xinhu Li, Ayush Jain, Zhaojing Yang, Yigit Korkmaz, Erdem Bıyık  

**Link**: [PDF](https://arxiv.org/pdf/2510.09096)  

**Abstract**: Learning from demonstrations enables experts to teach robots complex tasks using interfaces such as kinesthetic teaching, joystick control, and sim-to-real transfer. However, these interfaces often constrain the expert's ability to demonstrate optimal behavior due to indirect control, setup restrictions, and hardware safety. For example, a joystick can move a robotic arm only in a 2D plane, even though the robot operates in a higher-dimensional space. As a result, the demonstrations collected by constrained experts lead to suboptimal performance of the learned policies. This raises a key question: Can a robot learn a better policy than the one demonstrated by a constrained expert? We address this by allowing the agent to go beyond direct imitation of expert actions and explore shorter and more efficient trajectories. We use the demonstrations to infer a state-only reward signal that measures task progress, and self-label reward for unknown states using temporal interpolation. Our approach outperforms common imitation learning in both sample efficiency and task completion time. On a real WidowX robotic arm, it completes the task in 12 seconds, 10x faster than behavioral cloning, as shown in real-robot videos on this https URL . 

---
# AI and Human Oversight: A Risk-Based Framework for Alignment 

**Authors**: Laxmiraju Kandikatla, Branislav Radeljic  

**Link**: [PDF](https://arxiv.org/pdf/2510.09090)  

**Abstract**: As Artificial Intelligence (AI) technologies continue to advance, protecting human autonomy and promoting ethical decision-making are essential to fostering trust and accountability. Human agency (the capacity of individuals to make informed decisions) should be actively preserved and reinforced by AI systems. This paper examines strategies for designing AI systems that uphold fundamental rights, strengthen human agency, and embed effective human oversight mechanisms. It discusses key oversight models, including Human-in-Command (HIC), Human-in-the-Loop (HITL), and Human-on-the-Loop (HOTL), and proposes a risk-based framework to guide the implementation of these mechanisms. By linking the level of AI model risk to the appropriate form of human oversight, the paper underscores the critical role of human involvement in the responsible deployment of AI, balancing technological innovation with the protection of individual values and rights. In doing so, it aims to ensure that AI technologies are used responsibly, safeguarding individual autonomy while maximizing societal benefits. 

---
# Training Models to Detect Successive Robot Errors from Human Reactions 

**Authors**: Shannon Liu, Maria Teresa Parreira, Wendy Ju  

**Link**: [PDF](https://arxiv.org/pdf/2510.09080)  

**Abstract**: As robots become more integrated into society, detecting robot errors is essential for effective human-robot interaction (HRI). When a robot fails repeatedly, how can it know when to change its behavior? Humans naturally respond to robot errors through verbal and nonverbal cues that intensify over successive failures-from confusion and subtle speech changes to visible frustration and impatience. While prior work shows that human reactions can indicate robot failures, few studies examine how these evolving responses reveal successive failures. This research uses machine learning to recognize stages of robot failure from human reactions. In a study with 26 participants interacting with a robot that made repeated conversational errors, behavioral features were extracted from video data to train models for individual users. The best model achieved 93.5% accuracy for detecting errors and 84.1% for classifying successive failures. Modeling the progression of human reactions enhances error detection and understanding of repeated interaction breakdowns in HRI. 

---
# Emotion-Disentangled Embedding Alignment for Noise-Robust and Cross-Corpus Speech Emotion Recognition 

**Authors**: Upasana Tiwari, Rupayan Chakraborty, Sunil Kumar Kopparapu  

**Link**: [PDF](https://arxiv.org/pdf/2510.09072)  

**Abstract**: Effectiveness of speech emotion recognition in real-world scenarios is often hindered by noisy environments and variability across datasets. This paper introduces a two-step approach to enhance the robustness and generalization of speech emotion recognition models through improved representation learning. First, our model employs EDRL (Emotion-Disentangled Representation Learning) to extract class-specific discriminative features while preserving shared similarities across emotion categories. Next, MEA (Multiblock Embedding Alignment) refines these representations by projecting them into a joint discriminative latent subspace that maximizes covariance with the original speech input. The learned EDRL-MEA embeddings are subsequently used to train an emotion classifier using clean samples from publicly available datasets, and are evaluated on unseen noisy and cross-corpus speech samples. Improved performance under these challenging conditions demonstrates the effectiveness of the proposed method. 

---
# Alif: Advancing Urdu Large Language Models via Multilingual Synthetic Data Distillation 

**Authors**: Muhammad Ali Shafique, Kanwal Mehreen, Muhammad Arham, Maaz Amjad, Sabur Butt, Hamza Farooq  

**Link**: [PDF](https://arxiv.org/pdf/2510.09051)  

**Abstract**: Developing a high-performing large language models (LLMs) for low-resource languages such as Urdu, present several challenges. These challenges include the scarcity of high-quality datasets, multilingual inconsistencies, and safety concerns. Existing multilingual LLMs often address these issues by translating large volumes of available data. However, such translations often lack quality and cultural nuance while also incurring significant costs for data curation and training. To address these issues, we propose Alif-1.0-8B-Instruct, a multilingual Urdu-English model, that tackles these challenges with a unique approach. We train the model on a high-quality, multilingual synthetic dataset (Urdu-Instruct), developed using a modified self-instruct technique. By using unique prompts and seed values for each task along with a global task pool, this dataset incorporates Urdu-native chain-of-thought based reasoning, bilingual translation, cultural relevance, and ethical safety alignments. This technique significantly enhances the comprehension of Alif-1.0-8B-Instruct model for Urdu-specific tasks. As a result, Alif-1.0-8B-Instruct, built upon the pretrained Llama-3.1-8B, demonstrates superior performance compared to Llama-3.1-8B-Instruct for Urdu specific-tasks. It also outperformed leading multilingual LLMs, including Mistral-7B-Instruct-v0.3, Qwen-2.5-7B-Instruct, and Cohere-Aya-Expanse-8B, all within a training budget of under $100. Our results demonstrate that high-performance and low-resource language LLMs can be developed efficiently and culturally aligned using our modified self-instruct approach. All datasets, models, and code are publicly available at: this https URL. 

---
# Cost-Efficient Long Code Translation using LLMs while Leveraging Identifier Replacements 

**Authors**: Manojit Chakraborty, Madhusudan Ghosh, Rishabh Gupta  

**Link**: [PDF](https://arxiv.org/pdf/2510.09045)  

**Abstract**: In the domain of software development, LLMs have been utilized to automate tasks such as code translation, where source code from one programming language is translated to another while preserving its functionality. However, LLMs often struggle with long source codes that don't fit into the context window, which produces inaccurate translations. To address this, we propose a novel zero-shot code translation method that incorporates identifier replacement. By substituting user-given long identifiers with generalized placeholders during translation, our method allows the LLM to focus on the logical structure of the code, by reducing token count and memory usage, which improves the efficiency and cost-effectiveness of long code translation. Our empirical results demonstrate that our approach preserves syntactical and hierarchical information and produces translation results with reduced tokens. 

---
# Robust Driving Control for Autonomous Vehicles: An Intelligent General-sum Constrained Adversarial Reinforcement Learning Approach 

**Authors**: Junchao Fan, Xiaolin Chang  

**Link**: [PDF](https://arxiv.org/pdf/2510.09041)  

**Abstract**: Deep reinforcement learning (DRL) has demonstrated remarkable success in developing autonomous driving policies. However, its vulnerability to adversarial attacks remains a critical barrier to real-world deployment. Although existing robust methods have achieved success, they still suffer from three key issues: (i) these methods are trained against myopic adversarial attacks, limiting their abilities to respond to more strategic threats, (ii) they have trouble causing truly safety-critical events (e.g., collisions), but instead often result in minor consequences, and (iii) these methods can introduce learning instability and policy drift during training due to the lack of robust constraints. To address these issues, we propose Intelligent General-sum Constrained Adversarial Reinforcement Learning (IGCARL), a novel robust autonomous driving approach that consists of a strategic targeted adversary and a robust driving agent. The strategic targeted adversary is designed to leverage the temporal decision-making capabilities of DRL to execute strategically coordinated multi-step attacks. In addition, it explicitly focuses on inducing safety-critical events by adopting a general-sum objective. The robust driving agent learns by interacting with the adversary to develop a robust autonomous driving policy against adversarial attacks. To ensure stable learning in adversarial environments and to mitigate policy drift caused by attacks, the agent is optimized under a constrained formulation. Extensive experiments show that IGCARL improves the success rate by at least 27.9\% over state-of-the-art methods, demonstrating superior robustness to adversarial attacks and enhancing the safety and reliability of DRL-based autonomous driving. 

---
# Déréverbération non-supervisée de la parole par modèle hybride 

**Authors**: Louis Bahrman, Mathieu Fontaine, Gaël Richard  

**Link**: [PDF](https://arxiv.org/pdf/2510.09025)  

**Abstract**: This paper introduces a new training strategy to improve speech dereverberation systems in an unsupervised manner using only reverberant speech. Most existing algorithms rely on paired dry/reverberant data, which is difficult to obtain. Our approach uses limited acoustic information, like the reverberation time (RT60), to train a dereverberation system. Experimental results demonstrate that our method achieves more consistent performance across various objective metrics than the state-of-the-art. 

---
# Value-State Gated Attention for Mitigating Extreme-Token Phenomena in Transformers 

**Authors**: Rui Bu, Haofeng Zhong, Wenzheng Chen, Yangyan Li  

**Link**: [PDF](https://arxiv.org/pdf/2510.09017)  

**Abstract**: Large models based on the Transformer architecture are susceptible to extreme-token phenomena, such as attention sinks and value-state drains. These issues, which degrade model performance, quantization fidelity, and interpretability, arise from a problematic mutual reinforcement mechanism where the model learns an inefficient 'no-op' behavior by focusing attention on tokens with near-zero value states. In this paper, we propose Value-State Gated Attention (VGA), a simple, dedicated, and stable architectural mechanism for performing 'no-op' attention efficiently by directly breaking this cycle. VGA introduces a learnable, data-dependent gate, computed directly from the value vectors (V), to modulate the output. Through a theoretical analysis of the underlying gradients, we show that gating the value-state with a function of itself is more effective at decoupling value and attention score updates than prior methods that gate on input embeddings. This creates a direct regulatory pathway that allows the model to suppress a token's contribution based on its emergent value representation. Our experiments demonstrate that VGA significantly mitigates the formation of attention sinks and stabilizes value-state norms, leading to improved performance, robust quantization fidelity, and enhanced model interpretability. 

---
# DiTSinger: Scaling Singing Voice Synthesis with Diffusion Transformer and Implicit Alignment 

**Authors**: Zongcai Du, Guilin Deng, Xiaofeng Guo, Xin Gao, Linke Li, Kaichang Cheng, Fubo Han, Siyu Yang, Peng Liu, Pan Zhong, Qiang Fu  

**Link**: [PDF](https://arxiv.org/pdf/2510.09016)  

**Abstract**: Recent progress in diffusion-based Singing Voice Synthesis (SVS) demonstrates strong expressiveness but remains limited by data scarcity and model scalability. We introduce a two-stage pipeline: a compact seed set of human-sung recordings is constructed by pairing fixed melodies with diverse LLM-generated lyrics, and melody-specific models are trained to synthesize over 500 hours of high-quality Chinese singing data. Building on this corpus, we propose DiTSinger, a Diffusion Transformer with RoPE and qk-norm, systematically scaled in depth, width, and resolution for enhanced fidelity. Furthermore, we design an implicit alignment mechanism that obviates phoneme-level duration labels by constraining phoneme-to-acoustic attention within character-level spans, thereby improving robustness under noisy or uncertain alignments. Extensive experiments validate that our approach enables scalable, alignment-free, and high-fidelity SVS. 

---
# On Epistemic Uncertainty of Visual Tokens for Object Hallucinations in Large Vision-Language Models 

**Authors**: Hoigi Seo, Dong Un Kang, Hyunjin Cho, Joohoon Lee, Se Young Chun  

**Link**: [PDF](https://arxiv.org/pdf/2510.09008)  

**Abstract**: Large vision-language models (LVLMs), which integrate a vision encoder (VE) with a large language model, have achieved remarkable success across various tasks. However, there are still crucial challenges in LVLMs such as object hallucination, generating descriptions of objects that are not in the input image. Here, we argue that uncertain visual tokens within the VE is a key factor that contributes to object hallucination. Our statistical analysis found that there are positive correlations between visual tokens with high epistemic uncertainty and the occurrence of hallucinations. Furthermore, we show theoretically and empirically that visual tokens in early VE layers that exhibit large representation deviations under small adversarial perturbations indicate high epistemic uncertainty. Based on these findings, we propose a simple yet effective strategy to mitigate object hallucination by modifying the VE only. Our method comprises a proxy method with adversarial perturbations for identifying uncertain visual tokens efficiently and a method to mask these uncertain visual tokens during the self-attention process in the middle layers of the VE, suppressing their influence on visual encoding and thus alleviating hallucinations. Extensive experiments show that our method significantly reduces object hallucinations in LVLMs and can synergistically work with other prior arts. 

---
# SQS: Bayesian DNN Compression through Sparse Quantized Sub-distributions 

**Authors**: Ziyi Wang, Nan Jiang, Guang Lin, Qifan Song  

**Link**: [PDF](https://arxiv.org/pdf/2510.08999)  

**Abstract**: Compressing large-scale neural networks is essential for deploying models on resource-constrained devices. Most existing methods adopt weight pruning or low-bit quantization individually, often resulting in suboptimal compression rates to preserve acceptable performance drops. We introduce a unified framework for simultaneous pruning and low-bit quantization via Bayesian variational learning (SQS), which achieves higher compression rates than prior baselines while maintaining comparable performance. The key idea is to employ a spike-and-slab prior to inducing sparsity and model quantized weights using Gaussian Mixture Models (GMMs) to enable low-bit precision. In theory, we provide the consistent result of our proposed variational approach to a sparse and quantized deep neural network. Extensive experiments on compressing ResNet, BERT-base, Llama3, and Qwen2.5 models show that our method achieves higher compression rates than a line of existing methods with comparable performance drops. 

---
# Saving SWE-Bench: A Benchmark Mutation Approach for Realistic Agent Evaluation 

**Authors**: Spandan Garg, Ben Steenhoek, Yufan Huang  

**Link**: [PDF](https://arxiv.org/pdf/2510.08996)  

**Abstract**: Current benchmarks for evaluating software engineering agents, such as SWE-Bench Verified, are predominantly derived from GitHub issues and fail to accurately reflect how developers interact with chat-based coding assistants in integrated development environments (IDEs). We posit that this mismatch leads to a systematic overestimation of agent's capabilities in real-world scenarios, especially bug fixing. We introduce a novel benchmarking framework that transforms existing formal benchmarks into realistic user queries through systematic analysis of developer interaction patterns with chat-based agents. Our methodology is flexible and can be easily extended to existing benchmarks. In this paper, we apply our testing framework to SWE-Bench Verified, the TypeScript subset of Multi-SWE-Bench and a private benchmark, SWE-Bench C# and transform formal GitHub issue descriptions into realistic user-style queries based on telemetry analysis of a popular chat-based agent interactions. Our findings reveal that existing benchmarks significantly overestimate agent capabilities for some models by >50% over baseline performance for public benchmarks and ~10-16% for our internal benchmark. This work establishes a new paradigm for evaluating interactive chat-based software engineering agents through benchmark mutation techniques. 

---
# PlatformX: An End-to-End Transferable Platform for Energy-Efficient Neural Architecture Search 

**Authors**: Xiaolong Tu, Dawei Chen, Kyungtae Han, Onur Altintas, Haoxin Wang  

**Link**: [PDF](https://arxiv.org/pdf/2510.08993)  

**Abstract**: Hardware-Aware Neural Architecture Search (HW-NAS) has emerged as a powerful tool for designing efficient deep neural networks (DNNs) tailored to edge devices. However, existing methods remain largely impractical for real-world deployment due to their high time cost, extensive manual profiling, and poor scalability across diverse hardware platforms with complex, device-specific energy behavior. In this paper, we present PlatformX, a fully automated and transferable HW-NAS framework designed to overcome these limitations. PlatformX integrates four key components: (i) an energy-driven search space that expands conventional NAS design by incorporating energy-critical configurations, enabling exploration of high-efficiency architectures; (ii) a transferable kernel-level energy predictor across devices and incrementally refined with minimal on-device samples; (iii) a Pareto-based multi-objective search algorithm that balances energy and accuracy to identify optimal trade-offs; and (iv) a high-resolution runtime energy profiling system that automates on-device power measurement using external monitors without human intervention. We evaluate PlatformX across multiple mobile platforms, showing that it significantly reduces search overhead while preserving accuracy and energy fidelity. It identifies models with up to 0.94 accuracy or as little as 0.16 mJ per inference, both outperforming MobileNet-V2 in accuracy and efficiency. Code and tutorials are available at this http URL. 

---
# SEER: Sustainability Enhanced Engineering of Software Requirements 

**Authors**: Mandira Roy, Novarun Deb, Nabendu Chaki, Agostino Cortesi  

**Link**: [PDF](https://arxiv.org/pdf/2510.08981)  

**Abstract**: The rapid expansion of software development has significant environmental, technical, social, and economic impacts. Achieving the United Nations Sustainable Development Goals by 2030 compels developers to adopt sustainable practices. Existing methods mostly offer high-level guidelines, which are time-consuming to implement and rely on team adaptability. Moreover, they focus on design or implementation, while sustainability assessment should start at the requirements engineering phase. In this paper, we introduce SEER, a framework which addresses sustainability concerns in the early software development phase. The framework operates in three stages: (i) it identifies sustainability requirements (SRs) relevant to a specific software product from a general taxonomy; (ii) it evaluates how sustainable system requirements are based on the identified SRs; and (iii) it optimizes system requirements that fail to satisfy any SR. The framework is implemented using the reasoning capabilities of large language models and the agentic RAG (Retrieval Augmented Generation) approach. SEER has been experimented on four software projects from different domains. Results generated using Gemini 2.5 reasoning model demonstrate the effectiveness of the proposed approach in accurately identifying a broad range of sustainability concerns across diverse domains. 

---
# Learning Regularizers: Learning Optimizers that can Regularize 

**Authors**: Suraj Kumar Sahoo, Narayanan C Krishnan  

**Link**: [PDF](https://arxiv.org/pdf/2510.08968)  

**Abstract**: Learned Optimizers (LOs), a type of Meta-learning, have gained traction due to their ability to be parameterized and trained for efficient optimization. Traditional gradient-based methods incorporate explicit regularization techniques such as Sharpness-Aware Minimization (SAM), Gradient-norm Aware Minimization (GAM), and Gap-guided Sharpness-Aware Minimization (GSAM) to enhance generalization and convergence. In this work, we explore a fundamental question: \textbf{Can regularizers be learned?} We empirically demonstrate that LOs can be trained to learn and internalize the effects of traditional regularization techniques without explicitly applying them to the objective function. We validate this through extensive experiments on standard benchmarks (including MNIST, FMNIST, CIFAR and Neural Networks such as MLP, MLP-Relu and CNN), comparing LOs trained with and without access to explicit regularizers. Regularized LOs consistently outperform their unregularized counterparts in terms of test accuracy and generalization. Furthermore, we show that LOs retain and transfer these regularization effects to new optimization tasks by inherently seeking minima similar to those targeted by these regularizers. Our results suggest that LOs can inherently learn regularization properties, \textit{challenging the conventional necessity of explicit optimizee loss regularization. 

---
# Analytical Survey of Learning with Low-Resource Data: From Analysis to Investigation 

**Authors**: Xiaofeng Cao, Mingwei Xu, Xin Yu, Jiangchao Yao, Wei Ye, Shengjun Huang, Minling Zhang, Ivor W. Tsang, Yew Soon Ong, James T. Kwok, Heng Tao Shen  

**Link**: [PDF](https://arxiv.org/pdf/2510.08962)  

**Abstract**: Learning with high-resource data has demonstrated substantial success in artificial intelligence (AI); however, the costs associated with data annotation and model training remain significant. A fundamental objective of AI research is to achieve robust generalization with limited-resource data. This survey employs agnostic active sampling theory within the Probably Approximately Correct (PAC) framework to analyze the generalization error and label complexity associated with learning from low-resource data in both model-agnostic supervised and unsupervised settings. Based on this analysis, we investigate a suite of optimization strategies tailored for low-resource data learning, including gradient-informed optimization, meta-iteration optimization, geometry-aware optimization, and LLMs-powered optimization. Furthermore, we provide a comprehensive overview of multiple learning paradigms that can benefit from low-resource data, including domain transfer, reinforcement feedback, and hierarchical structure modeling. Finally, we conclude our analysis and investigation by summarizing the key findings and highlighting their implications for learning with low-resource data. 

---
# A Human Behavioral Baseline for Collective Governance in Software Projects 

**Authors**: Mobina Noori, Mahasweta Chakraborti, Amy X Zhang, Seth Frey  

**Link**: [PDF](https://arxiv.org/pdf/2510.08956)  

**Abstract**: We study how open source communities describe participation and control through version controlled governance documents. Using a corpus of 710 projects with paired snapshots, we parse text into actors, rules, actions, and objects, then group them and measure change with entropy for evenness, richness for diversity, and Jensen Shannon divergence for drift. Projects define more roles and more actions over time, and these are distributed more evenly, while the composition of rules remains stable. These findings indicate that governance grows by expanding and balancing categories of participation without major shifts in prescriptive force. The analysis provides a reproducible baseline for evaluating whether future AI mediated workflows concentrate or redistribute authority. 

---
# SHERLOCK: Towards Dynamic Knowledge Adaptation in LLM-enhanced E-commerce Risk Management 

**Authors**: Nan Lu, Yurong Hu, Jiaquan Fang, Yan Liu, Rui Dong, Yiming Wang, Rui Lin, Shaoyi Xu  

**Link**: [PDF](https://arxiv.org/pdf/2510.08948)  

**Abstract**: The growth of the e-commerce industry has intensified the adversarial dynamics between shadow economy actors and risk management teams. Companies often conduct risk investigations into suspicious cases to identify emerging fraud patterns, thereby enhancing both preemptive risk prevention and post-hoc governance. However, the sheer volume of case analyses imposes a substantial workload on risk management analysts, as each case requires the integration of long-term expert experience and meticulous scrutiny across multiple risk dimensions. Additionally, individual disparities among analysts hinder the establishment of uniform and high-standard workflows. To address these challenges, we propose the SHERLOCK framework, which leverages the reasoning capabilities of large language models (LLMs) to assist analysts in risk investigations. Our approach consists of three primary components: (1) extracting risk management knowledge from multi-modal data and constructing a domain knowledge base (KB), (2) building an intelligent platform guided by the data flywheel paradigm that integrates daily operations, expert annotations, and model evaluations, with iteratively fine-tuning for preference alignment, and (3) introducing a Reflect & Refine (R&R) module that collaborates with the domain KB to establish a rapid response mechanism for evolving risk patterns. Experiments conducted on the real-world transaction dataset from this http URL demonstrate that our method significantly improves the precision of both factual alignment and risk localization within the LLM analysis results. Deployment of the SHERLOCK-based LLM system on this http URL has substantially enhanced the efficiency of case investigation workflows for risk managers. 

---
# RO-Bench: Large-scale robustness evaluation of MLLMs with text-driven counterfactual videos 

**Authors**: Zixi Yang, Jiapeng Li, Muxi Diao, Yinuo Jing, Kongming Liang  

**Link**: [PDF](https://arxiv.org/pdf/2510.08936)  

**Abstract**: Recently, Multi-modal Large Language Models (MLLMs) have demonstrated significant performance across various video understanding tasks. However, their robustness, particularly when faced with manipulated video content, remains largely unexplored. In this paper, we introduce Ro-Bench, the first benchmark for evaluating MLLMs on dynamic out-of-distribution (OOD) counterfactual video test sets. Ro-Bench incorporates high-quality, diverse and temporally relevant video data, by editing Style, Object, Background and their compositions. We evaluated eight recent video MLLMs and found that current models exhibit substantial performance degradation on Ro-Bench when exposed to counterfactual video content. Furthermore, we demonstrate that fine-tuning MLLMs with counterfactual data enhances robustness, achieving a 21.73% performance increase on Ro-Bench and a 12.78% improvement across 20 tasks in the MVBench dataset. These findings underscore the effectiveness of counterfactual data in enhancing the video understanding ability of MLLMs. The code and data will be released shortly. 

---
# Co-Authoring the Self: A Human-AI Interface for Interest Reflection in Recommenders 

**Authors**: Ruixuan Sun, Junyuan Wang, Sanjali Roy, Joseph A. Konstan  

**Link**: [PDF](https://arxiv.org/pdf/2510.08930)  

**Abstract**: Natural language-based user profiles in recommender systems have been explored for their interpretability and potential to help users scrutinize and refine their interests, thereby improving recommendation quality. Building on this foundation, we introduce a human-AI collaborative profile for a movie recommender system that presents editable personalized interest summaries of a user's movie history. Unlike static profiles, this design invites users to directly inspect, modify, and reflect on the system's inferences. In an eight-week online field deployment with 1775 active movie recommender users, we find persistent gaps between user-perceived and system-inferred interests, show how the profile encourages engagement and reflection, and identify design directions for leveraging imperfect AI-powered user profiles to stimulate more user intervention and build more transparent and trustworthy recommender experiences. 

---
# A Frequency-Domain Analysis of the Multi-Armed Bandit Problem: A New Perspective on the Exploration-Exploitation Trade-off 

**Authors**: Di Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2510.08908)  

**Abstract**: The stochastic multi-armed bandit (MAB) problem is one of the most fundamental models in sequential decision-making, with the core challenge being the trade-off between exploration and exploitation. Although algorithms such as Upper Confidence Bound (UCB) and Thompson Sampling, along with their regret theories, are well-established, existing analyses primarily operate from a time-domain and cumulative regret perspective, struggling to characterize the dynamic nature of the learning process. This paper proposes a novel frequency-domain analysis framework, reformulating the bandit process as a signal processing problem. Within this framework, the reward estimate of each arm is viewed as a spectral component, with its uncertainty corresponding to the component's frequency, and the bandit algorithm is interpreted as an adaptive filter. We construct a formal Frequency-Domain Bandit Model and prove the main theorem: the confidence bound term in the UCB algorithm is equivalent in the frequency domain to a time-varying gain applied to uncertain spectral components, a gain inversely proportional to the square root of the visit count. Based on this, we further derive finite-time dynamic bounds concerning the exploration rate decay. This theory not only provides a novel and intuitive physical interpretation for classical algorithms but also lays a rigorous theoretical foundation for designing next-generation algorithms with adaptive parameter adjustment. 

---
# A Unified Biomedical Named Entity Recognition Framework with Large Language Models 

**Authors**: Tengxiao Lv, Ling Luo, Juntao Li, Yanhua Wang, Yuchen Pan, Chao Liu, Yanan Wang, Yan Jiang, Huiyi Lv, Yuanyuan Sun, Jian Wang, Hongfei Lin  

**Link**: [PDF](https://arxiv.org/pdf/2510.08902)  

**Abstract**: Accurate recognition of biomedical named entities is critical for medical information extraction and knowledge discovery. However, existing methods often struggle with nested entities, entity boundary ambiguity, and cross-lingual generalization. In this paper, we propose a unified Biomedical Named Entity Recognition (BioNER) framework based on Large Language Models (LLMs). We first reformulate BioNER as a text generation task and design a symbolic tagging strategy to jointly handle both flat and nested entities with explicit boundary annotation. To enhance multilingual and multi-task generalization, we perform bilingual joint fine-tuning across multiple Chinese and English datasets. Additionally, we introduce a contrastive learning-based entity selector that filters incorrect or spurious predictions by leveraging boundary-sensitive positive and negative samples. Experimental results on four benchmark datasets and two unseen corpora show that our method achieves state-of-the-art performance and robust zero-shot generalization across languages. The source codes are freely available at this https URL. 

---
# Pinpointing crucial steps: Attribution-based Credit Assignment for Verifiable Reinforcement Learning 

**Authors**: Junxi Yin, Haisen Luo, Zhenyu Li, Yihua Liu, Dan Liu, Zequn Li, Xiaohang Xu  

**Link**: [PDF](https://arxiv.org/pdf/2510.08899)  

**Abstract**: While Reinforcement Learning with Verifiable Rewards (RLVR) enhances complex reasoning in LLMs, current methods struggle to balance exploration and exploitation. This leads to critical issues like inaccurate credit assignment for intermediate steps and premature entropy collapse, limiting model performance. To address this, we introduce Attribution-based Contribution to Policy Optimization (ACPO), a phased framework that incorporates a difficulty-aware curriculum. ACPO improves exploration by using trajectory semantic segmentation and an attribution-based representation to dynamically regulate policy entropy, thus mitigating its collapse. Concurrently, it enhances exploitation with a factorized reward system that precisely quantifies the hierarchical contribution of each reasoning step, ensuring accurate credit assignment. Extensive experiments on challenging benchmarks, including AIME, MATH, and AMC, demonstrate that ACPO significantly outperforms existing state-of-the-art approaches. 

---
# HES-SQL: Hybrid Reasoning for Efficient Text-to-SQL with Structural Skeleton Guidance 

**Authors**: Suming Qiu, Jing Li, Zhicheng Zhou, Junjie Huang, Linyuan Qiu, Zhijie Sun  

**Link**: [PDF](https://arxiv.org/pdf/2510.08896)  

**Abstract**: We present HES-SQL, a novel hybrid training framework that advances Text-to-SQL generation through the integration of thinking-mode-fused supervised fine-tuning (SFT) with Group Relative Policy Optimization (GRPO). Our approach introduces three key innovations: (1) a skeleton-completeness scoring mechanism that enhances preference alignment between generated queries and optimal SQL structures; (2) a query-latency-aware reward system that incentivizes the generation of computationally efficient SQL queries; (3) a self-distillation process for thinking-mode completion that prevents degradation of the model's reasoning capabilities. This framework enables hybrid thinking models to switch between reasoning and non-reasoning modes while improving SQL query accuracy and execution efficiency.
Experimental evaluation, conducted on MySQL 8.0 and SQLite 3.42 under controlled single-user conditions, demonstrates that HES-SQL achieves competitive performance with execution accuracies of 79.14\% and 54.9\% on the BIRD and KaggleDBQA benchmarks, respectively. Query latency is measured as the end-to-end execution time of generated queries on the DBMS, averaged over multiple runs to mitigate variance. Efficiency gains range from 11\% to 20\% relative to supervised baselines. Our results establish a new paradigm for Text-to-SQL systems that effectively balances semantic accuracy with computational efficiency through execution-informed reinforcement learning (RL). The proposed methodology has significant implications for developing robust natural language interfaces to databases and can be extended to broader structured generation tasks requiring both correctness and efficiency optimization. 

---
# Exploring Multi-Temperature Strategies for Token- and Rollout-Level Control in RLVR 

**Authors**: Haomin Zhuang, Yujun Zhou, Taicheng Guo, Yue Huang, Fangxu Liu, Kai Song, Xiangliang Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2510.08892)  

**Abstract**: Reinforcement Learning has demonstrated substantial improvements in the reasoning abilities of Large Language Models (LLMs), exhibiting significant applicability across various domains. Recent research has identified that tokens within LLMs play distinct roles during reasoning tasks, categorizing them into high-entropy reasoning tokens and low-entropy knowledge tokens. Prior approaches have typically focused on restricting updates to indirectly encourage exploration, yet they do not explicitly facilitate exploratory behavior during the token generation stage itself. In this work, we introduce a complementary approach that explicitly promotes exploration during sampling by applying distinct temperature settings for different token types. Specifically, our method employs higher temperatures for reasoning tokens to actively encourage exploration, while retaining lower temperatures for knowledge tokens to maintain factual correctness. Furthermore, we systematically investigate various multi-temperature scheduling strategies and their impacts within reinforcement learning contexts. Empirical evaluations on several reasoning benchmarks demonstrate that our approach significantly enhances the reasoning performance of LLMs. The code is available at this https URL. 

---
# Designing and Evaluating an AI-driven Immersive Multidisciplinary Simulation (AIMS) for Interprofessional Education 

**Authors**: Ruijie Wang, Jie Lu, Bo Pei, Evonne Jones, Jamey Brinson, Timothy Brown  

**Link**: [PDF](https://arxiv.org/pdf/2510.08891)  

**Abstract**: Interprofessional education has long relied on case studies and the use of standardized patients to support teamwork, communication, and related collaborative competencies among healthcare professionals. However, traditional approaches are often limited by cost, scalability, and inability to mimic the dynamic complexity of real-world clinical scenarios. To address these challenges, we designed and developed AIMS (AI-Enhanced Immersive Multidisciplinary Simulations), a virtual simulation that integrates a large language model (Gemini-2.5-Flash), a Unity-based virtual environment engine, and a character creation pipeline to support synchronized, multimodal interactions between the user and the virtual patient. AIMS was designed to enhance collaborative clinical reasoning and health promotion competencies among students from pharmacy, medicine, nursing, and social work. A formal usability testing session was conducted which participants assumed professional roles on a healthcare team and engaged in a mix of scripted and unscripted conversations. Participants explored the patient's symptoms, social context, and care needs. Usability issues were identified (e.g., audio routing, response latency) and used to guide subsequent refinements. Findings in general suggest that AIMS supports realistic, profession-specific and contextually appropriate conversations. We discussed both technical and pedagogical innovations of AIMS and concluded with future directions. 

---
# ControlAudio: Tackling Text-Guided, Timing-Indicated and Intelligible Audio Generation via Progressive Diffusion Modeling 

**Authors**: Yuxuan Jiang, Zehua Chen, Zeqian Ju, Yusheng Dai, Weibei Dou, Jun Zhu  

**Link**: [PDF](https://arxiv.org/pdf/2510.08878)  

**Abstract**: Text-to-audio (TTA) generation with fine-grained control signals, e.g., precise timing control or intelligible speech content, has been explored in recent works. However, constrained by data scarcity, their generation performance at scale is still compromised. In this study, we recast controllable TTA generation as a multi-task learning problem and introduce a progressive diffusion modeling approach, ControlAudio. Our method adeptly fits distributions conditioned on more fine-grained information, including text, timing, and phoneme features, through a step-by-step strategy. First, we propose a data construction method spanning both annotation and simulation, augmenting condition information in the sequence of text, timing, and phoneme. Second, at the model training stage, we pretrain a diffusion transformer (DiT) on large-scale text-audio pairs, achieving scalable TTA generation, and then incrementally integrate the timing and phoneme features with unified semantic representations, expanding controllability. Finally, at the inference stage, we propose progressively guided generation, which sequentially emphasizes more fine-grained information, aligning inherently with the coarse-to-fine sampling nature of DiT. Extensive experiments show that ControlAudio achieves state-of-the-art performance in terms of temporal accuracy and speech clarity, significantly outperforming existing methods on both objective and subjective evaluations. Demo samples are available at: this https URL. 

---
# Vector Graph-Based Repository Understanding for Issue-Driven File Retrieval 

**Authors**: Kostiantyn Bevziuk, Andrii Fatula, Svetozar Lashin Yaroslav Opanasenko, Anna Tukhtarova, Ashok Jallepalli Pradeepkumar Sharma, Hritvik Shrivastava  

**Link**: [PDF](https://arxiv.org/pdf/2510.08876)  

**Abstract**: We present a repository decomposition system that converts large software repositories into a vectorized knowledge graph which mirrors project architectural and semantic structure, capturing semantic relationships and allowing a significant level of automatization of further repository development. The graph encodes syntactic relations such as containment, implementation, references, calls, and inheritance, and augments nodes with LLM-derived summaries and vector embeddings. A hybrid retrieval pipeline combines semantic retrieval with graph-aware expansion, and an LLM-based assistant formulates constrained, read-only graph requests and produces human-oriented explanations. 

---
# Slicing Is All You Need: Towards A Universal One-Sided Algorithm for Distributed Matrix Multiplication 

**Authors**: Benjamin Brock, Renato Golin  

**Link**: [PDF](https://arxiv.org/pdf/2510.08874)  

**Abstract**: Many important applications across science, data analytics, and AI workloads depend on distributed matrix multiplication. Prior work has developed a large array of algorithms suitable for different problem sizes and partitionings including 1D, 2D, 1.5D, and 2.5D algorithms. A limitation of current work is that existing algorithms are limited to a subset of partitionings. Multiple algorithm implementations are required to support the full space of possible partitionings. If no algorithm implementation is available for a particular set of partitionings, one or more operands must be redistributed, increasing communication costs. This paper presents a universal one-sided algorithm for distributed matrix multiplication that supports all combinations of partitionings and replication factors. Our algorithm uses slicing (index arithmetic) to compute the sets of overlapping tiles that must be multiplied together. This list of local matrix multiplies can then either be executed directly, or reordered and lowered to an optimized IR to maximize overlap. We implement our algorithm using a high-level C++-based PGAS programming framework that performs direct GPU-to-GPU communication using intra-node interconnects. We evaluate performance for a wide variety of partitionings and replication factors, finding that our work is competitive with PyTorch DTensor, a highly optimized distributed tensor library targeting AI models. 

---
# Pattern Enhanced Multi-Turn Jailbreaking: Exploiting Structural Vulnerabilities in Large Language Models 

**Authors**: Ragib Amin Nihal, Rui Wen, Kazuhiro Nakadai, Jun Sakuma  

**Link**: [PDF](https://arxiv.org/pdf/2510.08859)  

**Abstract**: Large language models (LLMs) remain vulnerable to multi-turn jailbreaking attacks that exploit conversational context to bypass safety constraints gradually. These attacks target different harm categories (like malware generation, harassment, or fraud) through distinct conversational approaches (educational discussions, personal experiences, hypothetical scenarios). Existing multi-turn jailbreaking methods often rely on heuristic or ad hoc exploration strategies, providing limited insight into underlying model weaknesses. The relationship between conversation patterns and model vulnerabilities across harm categories remains poorly understood. We propose Pattern Enhanced Chain of Attack (PE-CoA), a framework of five conversation patterns to construct effective multi-turn jailbreaks through natural dialogue. Evaluating PE-CoA on twelve LLMs spanning ten harm categories, we achieve state-of-the-art performance, uncovering pattern-specific vulnerabilities and LLM behavioral characteristics: models exhibit distinct weakness profiles where robustness to one conversational pattern does not generalize to others, and model families share similar failure modes. These findings highlight limitations of safety training and indicate the need for pattern-aware defenses. Code available on: this https URL 

---
# Time-Aware Feature Selection: Adaptive Temporal Masking for Stable Sparse Autoencoder Training 

**Authors**: T. Ed Li, Junyu Ren  

**Link**: [PDF](https://arxiv.org/pdf/2510.08855)  

**Abstract**: Understanding the internal representations of large language models is crucial for ensuring their reliability and safety, with sparse autoencoders (SAEs) emerging as a promising interpretability approach. However, current SAE training methods face feature absorption, where features (or neurons) are absorbed into each other to minimize $L_1$ penalty, making it difficult to consistently identify and analyze model behaviors. We introduce Adaptive Temporal Masking (ATM), a novel training approach that dynamically adjusts feature selection by tracking activation magnitudes, frequencies, and reconstruction contributions to compute importance scores that evolve over time. ATM applies a probabilistic masking mechanism based on statistical thresholding of these importance scores, creating a more natural feature selection process. Through extensive experiments on the Gemma-2-2b model, we demonstrate that ATM achieves substantially lower absorption scores compared to existing methods like TopK and JumpReLU SAEs, while maintaining excellent reconstruction quality. These results establish ATM as a principled solution for learning stable, interpretable features in neural networks, providing a foundation for more reliable model analysis. 

---
# Repository-Aware File Path Retrieval via Fine-Tuned LLMs 

**Authors**: Vasudha Yanuganti, Ishaan Puri, Swapnil Chhatre, Mantinder Singh, Ashok Jallepalli, Hritvik Shrivastava, Pradeep Kumar Sharma  

**Link**: [PDF](https://arxiv.org/pdf/2510.08850)  

**Abstract**: Modern codebases make it hard for developers and AI coding assistants to find the right source files when answering questions like "How does this feature work?" or "Where was the bug introduced?" Traditional code search (keyword or IR based) often misses semantic context and cross file links, while large language models (LLMs) understand natural language but lack repository specific detail. We present a method for file path retrieval that fine tunes a strong LLM (Qwen3-8B) with QLoRA and Unsloth optimizations to predict relevant file paths directly from a natural language query. To build training data, we introduce six code aware strategies that use abstract syntax tree (AST) structure and repository content to generate realistic question-answer pairs, where answers are sets of file paths. The strategies range from single file prompts to hierarchical repository summaries, providing broad coverage. We fine tune on Python projects including Flask, Click, Jinja, FastAPI, and PyTorch, and obtain high retrieval accuracy: up to 91\% exact match and 93\% recall on held out queries, clearly beating single strategy training. On a large codebase like PyTorch (about 4,000 Python files), the model reaches 59\% recall, showing scalability. We analyze how multi level code signals help the LLM reason over cross file context and discuss dataset design, limits (for example, context length in very large repos), and future integration of retrieval with LLM based code intelligence. 

---
# Reinforcement Learning-Driven Edge Management for Reliable Multi-view 3D Reconstruction 

**Authors**: Motahare Mounesan, Sourya Saha, Houchao Gan, Md. Nurul Absur, Saptarshi Debroy  

**Link**: [PDF](https://arxiv.org/pdf/2510.08839)  

**Abstract**: Real-time multi-view 3D reconstruction is a mission-critical application for key edge-native use cases, such as fire rescue, where timely and accurate 3D scene modeling enables situational awareness and informed decision-making. However, the dynamic and unpredictable nature of edge resource availability introduces disruptions, such as degraded image quality, unstable network links, and fluctuating server loads, which challenge the reliability of the reconstruction pipeline. In this work, we present a reinforcement learning (RL)-based edge resource management framework for reliable 3D reconstruction to ensure high quality reconstruction within a reasonable amount of time, despite the system operating under a resource-constrained and disruption-prone environment. In particular, the framework adopts two cooperative Q-learning agents, one for camera selection and one for server selection, both of which operate entirely online, learning policies through interactions with the edge environment. To support learning under realistic constraints and evaluate system performance, we implement a distributed testbed comprising lab-hosted end devices and FABRIC infrastructure-hosted edge servers to emulate smart city edge infrastructure under realistic disruption scenarios. Results show that the proposed framework improves application reliability by effectively balancing end-to-end latency and reconstruction quality in dynamic environments. 

---
# CommandSans: Securing AI Agents with Surgical Precision Prompt Sanitization 

**Authors**: Debeshee Das, Luca Beurer-Kellner, Marc Fischer, Maximilian Baader  

**Link**: [PDF](https://arxiv.org/pdf/2510.08829)  

**Abstract**: The increasing adoption of LLM agents with access to numerous tools and sensitive data significantly widens the attack surface for indirect prompt injections. Due to the context-dependent nature of attacks, however, current defenses are often ill-calibrated as they cannot reliably differentiate malicious and benign instructions, leading to high false positive rates that prevent their real-world adoption. To address this, we present a novel approach inspired by the fundamental principle of computer security: data should not contain executable instructions. Instead of sample-level classification, we propose a token-level sanitization process, which surgically removes any instructions directed at AI systems from tool outputs, capturing malicious instructions as a byproduct. In contrast to existing safety classifiers, this approach is non-blocking, does not require calibration, and is agnostic to the context of tool outputs. Further, we can train such token-level predictors with readily available instruction-tuning data only, and don't have to rely on unrealistic prompt injection examples from challenges or of other synthetic origin. In our experiments, we find that this approach generalizes well across a wide range of attacks and benchmarks like AgentDojo, BIPIA, InjecAgent, ASB and SEP, achieving a 7-10x reduction of attack success rate (ASR) (34% to 3% on AgentDojo), without impairing agent utility in both benign and malicious settings. 

---
# McMining: Automated Discovery of Misconceptions in Student Code 

**Authors**: Erfan Al-Hossami, Razvan Bunescu  

**Link**: [PDF](https://arxiv.org/pdf/2510.08827)  

**Abstract**: When learning to code, students often develop misconceptions about various programming language concepts. These can not only lead to bugs or inefficient code, but also slow down the learning of related concepts. In this paper, we introduce McMining, the task of mining programming misconceptions from samples of code from a student. To enable the training and evaluation of McMining systems, we develop an extensible benchmark dataset of misconceptions together with a large set of code samples where these misconceptions are manifested. We then introduce two LLM-based McMiner approaches and through extensive evaluations show that models from the Gemini, Claude, and GPT families are effective at discovering misconceptions in student code. 

---
# D-CoDe: Scaling Image-Pretrained VLMs to Video via Dynamic Compression and Question Decomposition 

**Authors**: Yiyang Huang, Yizhou Wang, Yun Fu  

**Link**: [PDF](https://arxiv.org/pdf/2510.08818)  

**Abstract**: Video large language models (Vid-LLMs), which excel in diverse video-language tasks, can be effectively constructed by adapting image-pretrained vision-language models (VLMs). However, this adaptation remains challenging, as it requires processing dense and temporally extended visual inputs that exceed the capacity of image-based models. This paper identifies the perception bottleneck and token overload as key challenges in extending image-based VLMs to the video domain. To address these issues, we propose D-CoDe, a training-free adaptation framework that incorporates dynamic compression and question decomposition. Specifically, dynamic compression alleviates the perception bottleneck through adaptive selection of representative frames and content-aware aggregation of spatial tokens, thereby reducing redundancy while preserving informative content. In parallel, question decomposition mitigates token overload by reformulating the original query into sub-questions, guiding the model to focus on distinct aspects of the video and enabling more comprehensive understanding. Experiments demonstrate that D-CoDe effectively improves video understanding across various benchmarks. Furthermore, strong performance on the challenging long-video benchmark highlights the potential of D-CoDe in handling complex video-language tasks. Code is available at this https URL. 

---
# $\mathsf{P} \neq \mathsf{NP}$: A Non-Relativizing Proof via Quantale Weakness and Geometric Complexity 

**Authors**: Ben Goertzel  

**Link**: [PDF](https://arxiv.org/pdf/2510.08814)  

**Abstract**: We give a compositional, information-theoretic framework that turns short programs into locality on many independent blocks, and combine it with symmetry and sparsity of masked random Unique-SAT to obtain distributional lower bounds that contradict the self-reduction upper bound under $\mathsf{P}=\mathsf{NP}$. We work in the weakness quantale $w_Q=K_{\mathrm{poly}}(\cdot\mid\cdot)$. For an efficiently samplable ensemble $D_m$ made by masking random $3$-CNFs with fresh $S_m\ltimes(\mathbb{Z}_2)^m$ symmetries and a small-seed Valiant--Vazirani isolation layer, we prove a Switching-by-Weakness normal form: for any polytime decoder $P$ of description length $\le \delta t$ (with $t=\Theta(m)$ blocks), a short wrapper $W$ makes $(P\circ W)$ per-bit local on a $\gamma$-fraction of blocks. Two ingredients then force near-randomness on $\Omega(t)$ blocks for every short decoder: (a) a sign-invariant neutrality lemma giving $\Pr[X_i=1\mid \mathcal{I}]=1/2$ for any sign-invariant view $\mathcal{I}$; and (b) a template sparsification theorem at logarithmic radius showing that any fixed local rule appears with probability $m^{-\Omega(1)}$. Combined with single-block bounds for tiny $\mathrm{ACC}^0$/streaming decoders, this yields a success bound $2^{-\Omega(t)}$ and, by Compression-from-Success, $K_{\mathrm{poly}}\big((X_1,\ldots,X_t)\mid(\Phi_1,\ldots,\Phi_t)\big)\ge \eta t$. If $\mathsf{P}=\mathsf{NP}$, a uniform constant-length program maps any on-promise instance to its unique witness in polytime (bit fixing via a $\mathrm{USAT}$ decider), so $K_{\mathrm{poly}}(X\mid\Phi)\le O(1)$ and the tuple complexity is $O(1)$, contradicting the linear bound. The proof is non-relativizing and non-natural; symmetry, sparsification, and switching yield a quantale upper-lower clash, hence $\mathsf{P}\ne\mathsf{NP}$. 

---
# Adaptive Science Operations in Deep Space Missions Using Offline Belief State Planning 

**Authors**: Grace Ra Kim, Hailey Warner, Duncan Eddy, Evan Astle, Zachary Booth, Edward Balaban, Mykel J. Kochenderfer  

**Link**: [PDF](https://arxiv.org/pdf/2510.08812)  

**Abstract**: Deep space missions face extreme communication delays and environmental uncertainty that prevent real-time ground operations. To support autonomous science operations in communication-constrained environments, we present a partially observable Markov decision process (POMDP) framework that adaptively sequences spacecraft science instruments. We integrate a Bayesian network into the POMDP observation space to manage the high-dimensional and uncertain measurements typical of astrobiology missions. This network compactly encodes dependencies among measurements and improves the interpretability and computational tractability of science data. Instrument operation policies are computed offline, allowing resource-aware plans to be generated and thoroughly validated prior to launch. We use the Enceladus Orbilander's proposed Life Detection Suite (LDS) as a case study, demonstrating how Bayesian network structure and reward shaping influence system performance. We compare our method against the mission's baseline Concept of Operations (ConOps), evaluating both misclassification rates and performance in off-nominal sample accumulation scenarios. Our approach reduces sample identification errors by nearly 40% 

---
# Benchmarking Chinese Commonsense Reasoning with a Multi-hop Reasoning Perspective 

**Authors**: Wangjie You, Xusheng Wang, Xing Wang, Wenxiang Jiao, Chao Feng, Juntao Li, Min Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2510.08800)  

**Abstract**: While Large Language Models (LLMs) have demonstrated advanced reasoning capabilities, their comprehensive evaluation in general Chinese-language contexts remains understudied. To bridge this gap, we propose Chinese Commonsense Multi-hop Reasoning (CCMOR), a novel benchmark designed to evaluate LLMs' ability to integrate Chinese-specific factual knowledge with multi-step logical reasoning. Specifically, we first construct a domain-balanced seed set from existing QA datasets, then develop an LLM-powered pipeline to generate multi-hop questions anchored on factual unit chains. To ensure the quality of resulting dataset, we implement a human-in-the-loop verification system, where domain experts systematically validate and refine the generated questions. Using CCMOR, we evaluate state-of-the-art LLMs, demonstrating persistent limitations in LLMs' ability to process long-tail knowledge and execute knowledge-intensive reasoning. Notably, retrieval-augmented generation substantially mitigates these knowledge gaps, yielding significant performance gains. 

---
# SkipSR: Faster Super Resolution with Token Skipping 

**Authors**: Rohan Choudhury, Shanchuan Lin, Jianyi Wang, Hao Chen, Qi Zhao, Feng Cheng, Lu Jiang, Kris Kitani, Laszlo A. Jeni  

**Link**: [PDF](https://arxiv.org/pdf/2510.08799)  

**Abstract**: Diffusion-based super-resolution (SR) is a key component in video generation and video restoration, but is slow and expensive, limiting scalability to higher resolutions and longer videos. Our key insight is that many regions in video are inherently low-detail and gain little from refinement, yet current methods process all pixels uniformly. To take advantage of this, we propose SkipSR, a simple framework for accelerating video SR by identifying low-detail regions directly from low-resolution input, then skipping computation on them entirely, only super-resolving the areas that require refinement. This simple yet effective strategy preserves perceptual quality in both standard and one-step diffusion SR models while significantly reducing computation. In standard SR benchmarks, our method achieves up to 60% faster end-to-end latency than prior models on 720p videos with no perceptible loss in quality. Video demos are available at this https URL 

---
# Deceptive Exploration in Multi-armed Bandits 

**Authors**: I. Arda Vurankaya, Mustafa O. Karabag, Wesley A. Suttle, Jesse Milzman, David Fridovich-Keil, Ufuk Topcu  

**Link**: [PDF](https://arxiv.org/pdf/2510.08794)  

**Abstract**: We consider a multi-armed bandit setting in which each arm has a public and a private reward distribution. An observer expects an agent to follow Thompson Sampling according to the public rewards, however, the deceptive agent aims to quickly identify the best private arm without being noticed. The observer can observe the public rewards and the pulled arms, but not the private rewards. The agent, on the other hand, observes both the public and private rewards. We formalize detectability as a stepwise Kullback-Leibler (KL) divergence constraint between the actual pull probabilities used by the agent and the anticipated pull probabilities by the observer. We model successful pulling of public suboptimal arms as a % Bernoulli process where the success probability decreases with each successful pull, and show these pulls can happen at most at a $\Theta(\sqrt{T}) $ rate under the KL constraint. We then formulate a maximin problem based on public and private means, whose solution characterizes the optimal error exponent for best private arm identification. We finally propose an algorithm inspired by top-two algorithms. This algorithm naturally adapts its exploration according to the hardness of pulling arms based on the public suboptimality gaps. We provide numerical examples illustrating the $\Theta(\sqrt{T}) $ rate and the behavior of the proposed algorithm. 

---
# MLLM as a UI Judge: Benchmarking Multimodal LLMs for Predicting Human Perception of User Interfaces 

**Authors**: Reuben A. Luera, Ryan Rossi, Franck Dernoncourt, Samyadeep Basu, Sungchul Kim, Subhojyoti Mukherjee, Puneet Mathur, Ruiyi Zhang, Jihyung Kil, Nedim Lipka, Seunghyun Yoon, Jiuxiang Gu, Zichao Wang, Cindy Xiong Bearfield, Branislav Kveton  

**Link**: [PDF](https://arxiv.org/pdf/2510.08783)  

**Abstract**: In an ideal design pipeline, user interface (UI) design is intertwined with user research to validate decisions, yet studies are often resource-constrained during early exploration. Recent advances in multimodal large language models (MLLMs) offer a promising opportunity to act as early evaluators, helping designers narrow options before formal testing. Unlike prior work that emphasizes user behavior in narrow domains such as e-commerce with metrics like clicks or conversions, we focus on subjective user evaluations across varied interfaces. We investigate whether MLLMs can mimic human preferences when evaluating individual UIs and comparing them. Using data from a crowdsourcing platform, we benchmark GPT-4o, Claude, and Llama across 30 interfaces and examine alignment with human judgments on multiple UI factors. Our results show that MLLMs approximate human preferences on some dimensions but diverge on others, underscoring both their potential and limitations in supplementing early UX research. 

---
# Guiding Exploration in Reinforcement Learning Through LLM-Augmented Observations 

**Authors**: Vaibhav Jain, Gerrit Grossmann  

**Link**: [PDF](https://arxiv.org/pdf/2510.08779)  

**Abstract**: Reinforcement Learning (RL) agents often struggle in sparse-reward environments where traditional exploration strategies fail to discover effective action sequences. Large Language Models (LLMs) possess procedural knowledge and reasoning capabilities from text pretraining that could guide RL exploration, but existing approaches create rigid dependencies where RL policies must follow LLM suggestions or incorporate them directly into reward functions. We propose a framework that provides LLM-generated action recommendations through augmented observation spaces, allowing RL agents to learn when to follow or ignore this guidance. Our method leverages LLMs' world knowledge and reasoning abilities while maintaining flexibility through soft constraints. We evaluate our approach on three BabyAI environments of increasing complexity and show that the benefits of LLM guidance scale with task difficulty. In the most challenging environment, we achieve 71% relative improvement in final success rates over baseline. The approach provides substantial sample efficiency gains, with agents reaching performance thresholds up to 9 times faster, and requires no modifications to existing RL algorithms. Our results demonstrate an effective method for leveraging LLM planning capabilities to accelerate RL training in challenging environments. 

---
# Measuring Moral LLM Responses in Multilingual Capacities 

**Authors**: Kimaya Basu, Savi Kolari, Allison Yu  

**Link**: [PDF](https://arxiv.org/pdf/2510.08776)  

**Abstract**: With LLM usage becoming widespread across countries, languages, and humanity more broadly, the need to understand and guardrail their multilingual responses increases. Large-scale datasets for testing and benchmarking have been created to evaluate and facilitate LLM responses across multiple dimensions. In this study, we evaluate the responses of frontier and leading open-source models in five dimensions across low and high-resource languages to measure LLM accuracy and consistency across multilingual contexts. We evaluate the responses using a five-point grading rubric and a judge LLM. Our study shows that GPT-5 performed the best on average in each category, while other models displayed more inconsistency across language and category. Most notably, in the Consent & Autonomy and Harm Prevention & Safety categories, GPT scored the highest with averages of 3.56 and 4.73, while Gemini 2.5 Pro scored the lowest with averages of 1.39 and 1.98, respectively. These findings emphasize the need for further testing on how linguistic shifts impact LLM responses across various categories and improvement in these areas. 

---
# Re-Identifying Kākā with AI-Automated Video Key Frame Extraction 

**Authors**: Paula Maddigan, Andrew Lensen, Rachael C. Shaw  

**Link**: [PDF](https://arxiv.org/pdf/2510.08775)  

**Abstract**: Accurate recognition and re-identification of individual animals is essential for successful wildlife population monitoring. Traditional methods, such as leg banding of birds, are time consuming and invasive. Recent progress in artificial intelligence, particularly computer vision, offers encouraging solutions for smart conservation and efficient automation. This study presents a unique pipeline for extracting high-quality key frames from videos of kākā (Nestor meridionalis), a threatened forest-dwelling parrot in New Zealand. Key frame extraction is well-studied in person re-identification, however, its application to wildlife is limited. Using video recordings at a custom-built feeder, we extract key frames and evaluate the re-identification performance of our pipeline. Our unsupervised methodology combines object detection using YOLO and Grounding DINO, optical flow blur detection, image encoding with DINOv2, and clustering methods to identify representative key frames. The results indicate that our proposed key frame selection methods yield image collections which achieve high accuracy in kākā re-identification, providing a foundation for future research using media collected in more diverse and challenging environments. Through the use of artificial intelligence and computer vision, our non-invasive and efficient approach provides a valuable alternative to traditional physical tagging methods for recognising kākā individuals and therefore improving the monitoring of populations. This research contributes to developing fresh approaches in wildlife monitoring, with applications in ecology and conservation biology. 

---
# Struc-EMB: The Potential of Structure-Aware Encoding in Language Embeddings 

**Authors**: Shikun Liu, Haoyu Wang, Mufei Li, Pan Li  

**Link**: [PDF](https://arxiv.org/pdf/2510.08774)  

**Abstract**: Text embeddings from Large Language Models (LLMs) have become foundational for numerous applications. However, these models typically operate on raw text, overlooking the rich structural information, such as hyperlinks or citations, that provides crucial context in many real-world datasets. This paper introduces and systematically evaluates a new paradigm for generating structure-aware text embeddings by integrating these structural relations directly into the LLM's internal encoding process, rather than relying on traditional post-hoc aggregation. We investigate two primary in-process methods: sequential concatenation and parallel caching. Through extensive zero-shot experiments across retrieval, clustering, classification, and recommendation tasks, we demonstrate that our structure-aware approaches consistently outperform both text-only and post-hoc baselines. Our analysis reveals critical trade-offs: sequential concatenation excels with noisy, moderate-length contexts, while parallel caching scales more effectively to long, high-signal contexts but is more susceptible to distractors. To address the challenge of noisy structural data, we also introduce and validate two effective techniques: Context Distillation and Semantic Balancing. This work provides the first comprehensive analysis of in-process structure-aware encoding, offering a blueprint for building more powerful and contextually aware embedding models. 

---
# SAFER-AiD: Saccade-Assisted Foveal-peripheral vision Enhanced Reconstruction for Adversarial Defense 

**Authors**: Jiayang Liu, Daniel Tso, Yiming Bu, Qinru Qiu  

**Link**: [PDF](https://arxiv.org/pdf/2510.08761)  

**Abstract**: Adversarial attacks significantly challenge the safe deployment of deep learning models, particularly in real-world applications. Traditional defenses often rely on computationally intensive optimization (e.g., adversarial training or data augmentation) to improve robustness, whereas the human visual system achieves inherent robustness to adversarial perturbations through evolved biological mechanisms. We hypothesize that attention guided non-homogeneous sparse sampling and predictive coding plays a key role in this robustness. To test this hypothesis, we propose a novel defense framework incorporating three key biological mechanisms: foveal-peripheral processing, saccadic eye movements, and cortical filling-in. Our approach employs reinforcement learning-guided saccades to selectively capture multiple foveal-peripheral glimpses, which are integrated into a reconstructed image before classification. This biologically inspired preprocessing effectively mitigates adversarial noise, preserves semantic integrity, and notably requires no retraining or fine-tuning of downstream classifiers, enabling seamless integration with existing systems. Experiments on the ImageNet dataset demonstrate that our method improves system robustness across diverse classifiers and attack types, while significantly reducing training overhead compared to both biologically and non-biologically inspired defense techniques. 

---
# Graph Diffusion Transformers are In-Context Molecular Designers 

**Authors**: Gang Liu, Jie Chen, Yihan Zhu, Michael Sun, Tengfei Luo, Nitesh V Chawla, Meng Jiang  

**Link**: [PDF](https://arxiv.org/pdf/2510.08744)  

**Abstract**: In-context learning allows large models to adapt to new tasks from a few demonstrations, but it has shown limited success in molecular design. Existing databases such as ChEMBL contain molecular properties spanning millions of biological assays, yet labeled data for each property remain scarce. To address this limitation, we introduce demonstration-conditioned diffusion models (DemoDiff), which define task contexts using a small set of molecule-score examples instead of text descriptions. These demonstrations guide a denoising Transformer to generate molecules aligned with target properties. For scalable pretraining, we develop a new molecular tokenizer with Node Pair Encoding that represents molecules at the motif level, requiring 5.5$\times$ fewer nodes. We curate a dataset containing millions of context tasks from multiple sources covering both drugs and materials, and pretrain a 0.7-billion-parameter model on it. Across 33 design tasks in six categories, DemoDiff matches or surpasses language models 100-1000$\times$ larger and achieves an average rank of 3.63 compared to 5.25-10.20 for domain-specific approaches. These results position DemoDiff as a molecular foundation model for in-context molecular design. Our code is available at this https URL. 

---
# Coordinates from Context: Using LLMs to Ground Complex Location References 

**Authors**: Tessa Masis, Brendan O'Connor  

**Link**: [PDF](https://arxiv.org/pdf/2510.08741)  

**Abstract**: Geocoding is the task of linking a location reference to an actual geographic location and is essential for many downstream analyses of unstructured text. In this paper, we explore the challenging setting of geocoding compositional location references. Building on recent work demonstrating LLMs' abilities to reason over geospatial data, we evaluate LLMs' geospatial knowledge versus reasoning skills relevant to our task. Based on these insights, we propose an LLM-based strategy for geocoding compositional location references. We show that our approach improves performance for the task and that a relatively small fine-tuned LLM can achieve comparable performance with much larger off-the-shelf models. 

---
# When to Reason: Semantic Router for vLLM 

**Authors**: Chen Wang, Xunzhuo Liu, Yuhan Liu, Yue Zhu, Xiangxi Mo, Junchen Jiang, Huamin Chen  

**Link**: [PDF](https://arxiv.org/pdf/2510.08731)  

**Abstract**: Large Language Models (LLMs) demonstrate substantial accuracy gains when augmented with reasoning modes such as chain-of-thought and inference-time scaling. However, reasoning also incurs significant costs in inference latency and token usage, with environmental and financial impacts, which are unnecessary for many simple prompts. We present a semantic router that classifies queries based on their reasoning requirements and selectively applies reasoning only when beneficial. Our approach achieves a 10.2 percentage point improvement in accuracy on the MMLU-Pro benchmark while reducing response latency by 47.1% and token consumption by 48.5% compared to direct inference with vLLM. These results demonstrate that semantic routing offers an effective mechanism for striking a balance between accuracy and efficiency in open-source LLM serving systems 

---
# Enhancing Self-Supervised Learning with Semantic Pairs A New Dataset and Empirical Study 

**Authors**: Mohammad Alkhalefi, Georgios Leontidis, Mingjun Zhong  

**Link**: [PDF](https://arxiv.org/pdf/2510.08722)  

**Abstract**: Instance discrimination is a self-supervised representation learning paradigm wherein individual instances within a dataset are treated as distinct classes. This is typically achieved by generating two disparate views of each instance by applying stochastic transformations, which encourages the model to learn representations that are invariant to the common underlying object across these views. 

---
# In-Context Learning for Non-Stationary MIMO Equalization 

**Authors**: Jiachen Jiang, Zhen Qin, Zhihui Zhu  

**Link**: [PDF](https://arxiv.org/pdf/2510.08711)  

**Abstract**: Channel equalization is fundamental for mitigating distortions such as frequency-selective fading and inter-symbol interference. Unlike standard supervised learning approaches that require costly retraining or fine-tuning for each new task, in-context learning (ICL) adapts to new channels at inference time with only a few examples. However, existing ICL-based equalizers are primarily developed for and evaluated on static channels within the context window. Indeed, to our knowledge, prior principled analyses and theoretical studies of ICL focus exclusively on the stationary setting, where the function remains fixed within the context. In this paper, we investigate the ability of ICL to address non-stationary problems through the lens of time-varying channel equalization. We employ a principled framework for designing efficient attention mechanisms with improved adaptivity in non-stationary tasks, leveraging algorithms from adaptive signal processing to guide better designs. For example, new attention variants can be derived from the Least Mean Square (LMS) adaptive algorithm, a Least Root Mean Square (LRMS) formulation for enhanced robustness, or multi-step gradient updates for improved long-term tracking. Experimental results demonstrate that ICL holds strong promise for non-stationary MIMO equalization, and that attention mechanisms inspired by classical adaptive algorithms can substantially enhance adaptability and performance in dynamic environments. Our findings may provide critical insights for developing next-generation wireless foundation models with stronger adaptability and robustness. 

---
# ConPoSe: LLM-Guided Contact Point Selection for Scalable Cooperative Object Pushing 

**Authors**: Noah Steinkrüger, Nisarga Nilavadi, Wolfram Burgard, Tanja Katharina Kaiser  

**Link**: [PDF](https://arxiv.org/pdf/2510.08705)  

**Abstract**: Object transportation in cluttered environments is a fundamental task in various domains, including domestic service and warehouse logistics. In cooperative object transport, multiple robots must coordinate to move objects that are too large for a single robot. One transport strategy is pushing, which only requires simple robots. However, careful selection of robot-object contact points is necessary to push the object along a preplanned path. Although this selection can be solved analytically, the solution space grows combinatorially with the number of robots and object size, limiting scalability. Inspired by how humans rely on common-sense reasoning for cooperative transport, we propose combining the reasoning capabilities of Large Language Models with local search to select suitable contact points. Our LLM-guided local search method for contact point selection, ConPoSe, successfully selects contact points for a variety of shapes, including cuboids, cylinders, and T-shapes. We demonstrate that ConPoSe scales better with the number of robots and object size than the analytical approach, and also outperforms pure LLM-based selection. 

---
# BigCodeArena: Unveiling More Reliable Human Preferences in Code Generation via Execution 

**Authors**: Terry Yue Zhuo, Xiaolong Jin, Hange Liu, Juyong Jiang, Tianyang Liu, Chen Gong, Bhupesh Bishnoi, Vaisakhi Mishra, Marek Suppa, Noah Ziems, Saiteja Utpala, Ming Xu, Guangyu Song, Kaixin Li, Yuhan Cao, Bo Liu, Zheng Liu, Sabina Abdurakhmanova, Wenhao Yu, Mengzhao Jia, Jihan Yao, Kenneth Hamilton, Kumar Shridhar, Minh Chien Vu, Dingmin Wang, Jiawei Liu, Zijian Wang, Qian Liu, Binyuan Hui, Meg Risdal, Ahsen Khaliq, Atin Sood, Zhenchang Xing, Wasi Uddin Ahmad, John Grundy, David Lo, Banghua Zhu, Xiaoning Du, Torsten Scholak, Leandro von Werra  

**Link**: [PDF](https://arxiv.org/pdf/2510.08697)  

**Abstract**: Crowdsourced model evaluation platforms, such as Chatbot Arena, enable real-time evaluation from human perspectives to assess the quality of model responses. In the coding domain, manually examining the quality of LLM-generated content is extremely challenging, as it requires understanding long chunks of raw code and deliberately simulating code execution. To this end, we introduce BigCodeArena, an open human evaluation platform for code generation backed by a comprehensive and on-the-fly execution environment. Built on top of Chatbot Arena, BigCodeArena enables the execution of LLM-generated code and allows humans to interact with the execution process and outcomes. We collected over 14,000 raw code-centric conversation sessions across 10 widely used LLMs, spanning 10 languages and 8 types of execution environments. Among these conversations, we identified more than 4,700 multi-turn samples with pairwise human preferences. Further analysis uncovers underexplored preferences of LLMs in fine-grained domains characterized by tasks, languages, and frameworks. To systematically examine code understanding and generation capabilities of frontier LLMs, we curated two benchmarks based on the collected data, namely BigCodeReward and AutoCodeArena. For BigCodeReward, we post-processed the 4,700 conversations and evaluated the consistency between reward models and human preferences. The evaluation shows that most LLMs have superior performance in judging coding preferences when the execution results are available. Inspired by these findings, we propose AutoCodeArena, an automatic Elo rating benchmark designed to assess the coding quality of LLMs without human involvement. We find that proprietary LLMs like GPT-5, Claude-Sonnet-4, and Claude-Opus-4 still lead in code generation performance among recent emerging models. 

---
# FreqCa: Accelerating Diffusion Models via Frequency-Aware Caching 

**Authors**: Jiacheng Liu, Peiliang Cai, Qinming Zhou, Yuqi Lin, Deyang Kong, Benhao Huang, Yupei Pan, Haowen Xu, Chang Zou, Junshu Tang, Shikang Zheng, Linfeng Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2510.08669)  

**Abstract**: The application of diffusion transformers is suffering from their significant inference costs. Recently, feature caching has been proposed to solve this problem by reusing features from previous timesteps, thereby skipping computation in future timesteps. However, previous feature caching assumes that features in adjacent timesteps are similar or continuous, which does not always hold in all settings. To investigate this, this paper begins with an analysis from the frequency domain, which reveal that different frequency bands in the features of diffusion models exhibit different dynamics across timesteps. Concretely, low-frequency components, which decide the structure of images, exhibit higher similarity but poor continuity. In contrast, the high-frequency bands, which decode the details of images, show significant continuity but poor similarity. These interesting observations motivate us to propose Frequency-aware Caching (FreqCa)
which directly reuses features of low-frequency components based on their similarity, while using a second-order Hermite interpolator to predict the volatile high-frequency ones based on its continuity.
Besides, we further propose to cache Cumulative Residual Feature (CRF) instead of the features in all the layers, which reduces the memory footprint of feature caching by 99%.
Extensive experiments on FLUX.1-dev, FLUX.1-Kontext-dev, Qwen-Image, and Qwen-Image-Edit demonstrate its effectiveness in both generation and editing. Codes are available in the supplementary materials and will be released on GitHub. 

---
# RAG4Tickets: AI-Powered Ticket Resolution via Retrieval-Augmented Generation on JIRA and GitHub Data 

**Authors**: Mohammad Baqar  

**Link**: [PDF](https://arxiv.org/pdf/2510.08667)  

**Abstract**: Modern software teams frequently encounter delays in resolving recurring or related issues due to fragmented knowledge scattered across JIRA tickets, developer discussions, and GitHub pull requests (PRs). To address this challenge, we propose a Retrieval-Augmented Generation (RAG) framework that integrates Sentence-Transformers for semantic embeddings with FAISS-based vector search to deliver context-aware ticket resolution recommendations. The approach embeds historical JIRA tickets, user comments, and linked PR metadata to retrieve semantically similar past cases, which are then synthesized by a Large Language Model (LLM) into grounded and explainable resolution suggestions. The framework contributes a unified pipeline linking JIRA and GitHub data, an embedding and FAISS indexing strategy for heterogeneous software artifacts, and a resolution generation module guided by retrieved evidence. Experimental evaluation using precision, recall, resolution time reduction, and developer acceptance metrics shows that the proposed system significantly improves resolution accuracy, fix quality, and knowledge reuse in modern DevOps environments. 

---
# dInfer: An Efficient Inference Framework for Diffusion Language Models 

**Authors**: Yuxin Ma, Lun Du, Lanning Wei, Kun Chen, Qian Xu, Kangyu Wang, Guofeng Feng, Guoshan Lu, Lin Liu, Xiaojing Qi, Xinyuan Zhang, Zhen Tao, Haibo Feng, Ziyun Jiang, Ying Xu, Zenan Huang, Yihong Zhuang, Haokai Xu, Jiaqi Hu, Zhenzhong Lan, Junbo Zhao, Jianguo Li, Da Zheng  

**Link**: [PDF](https://arxiv.org/pdf/2510.08666)  

**Abstract**: Diffusion-based large language models (dLLMs) have emerged as a promising alternative to autoregressive (AR) LLMs, leveraging denoising-based generation to enable inherent parallelism. Even more and more open-sourced dLLM models emerge, yet their widespread adoption remains constrained by the lack of a standardized and efficient inference framework. We present dInfer, an efficient and extensible framework for dLLM inference. dInfer decomposes the inference pipeline into four modular components-model, diffusion iteration manager, decoding strategy, and KV-cache manager-and integrates novel algorithms for each component alongside system-level optimizations. Through this combination of algorithmic innovations and system enhancements, dInfer achieves substantial efficiency gains without compromising output quality on LLaDA-MoE. At batch size 1, it surpasses 1,100 tokens per second on HumanEval and averages over 800 tokens per second across six benchmarks on $8\times$ H800 GPUs. Compared to prior systems, dInfer delivers $10\times$ speedup over Fast-dLLM while maintaining similar model performance. Even compared with AR models (with a comparable number of activation parameters and performance) QWen2.5-3B, which is highly optimized with latest vLLM inference engine, dInfer still deliverers $2$-$3\times$ speedup. The implementation of dInfer is open-sourced at this https URL. 

---
# RA-Gen: A Controllable Code Generation Framework Using ReAct for Multi-Agent Task Execution 

**Authors**: Aofan Liu, Haoxuan Li, Bin Wang, Ao Yang, Hui Li  

**Link**: [PDF](https://arxiv.org/pdf/2510.08665)  

**Abstract**: Code generation models based on large language models (LLMs) have gained wide adoption, but challenges remain in ensuring safety, accuracy, and controllability, especially for complex tasks. Existing methods often lack dynamic integration of external tools, transparent reasoning, and user control over safety. To address these issues, we propose a controllable code generation framework utilizing the ReAct paradigm for multi-agent task execution. This framework is a multi-agent system designed to enable efficient, precise, and interpretable code generation through dynamic interactions between LLMs and external resources. The framework adopts a collaborative architecture comprising four specialized agents: a Planner for task decomposition, a Searcher that leverages the ReAct framework for reasoning and tool integration, a CodeGen agent for accurate code generation, and an Extractor for structured data retrieval. The ReAct-based Searcher alternates between generating reasoning traces and executing actions, facilitating seamless integration of internal knowledge with external tools (such as search engines) to enhance accuracy and user control. Experimental results show the framework's effectiveness across multiple languages, achieving a 94.8% security rate on the SVEN dataset with CodeQL, outperforming existing approaches. Its transparent reasoning process fosters user trust and improves controllability. 

---
# Faver: Boosting LLM-based RTL Generation with Function Abstracted Verifiable Middleware 

**Authors**: Jianan Mu, Mingyu Shi, Yining Wang, Tianmeng Yang, Bin Sun, Xing Hu, Jing Ye, Huawei Li  

**Link**: [PDF](https://arxiv.org/pdf/2510.08664)  

**Abstract**: LLM-based RTL generation is an interesting research direction, as it holds the potential to liberate the least automated stage in the current chip design. However, due to the substantial semantic gap between high-level specifications and RTL, coupled with limited training data, existing models struggle with generation accuracy. Drawing on human experience, design with verification helps improving accuracy. However, as the RTL testbench data are even more scarce, it is not friendly for LLMs. Although LLMs excel at higher-level languages like Python/C, they have a huge semantic gap from RTL. When implementing the same functionality, Python/C code and hardware code differ significantly in the spatiotemporal granularity, requiring the LLM not only to consider high-level functional semantics but also to ensure the low-level details align with the circuit code. It is not an easy task. In this paper, we propose a function abstracted verifiable middleware (Faver) that streamlines RTL verification in LLM-based workflows. By mixing LLM-friendly code structures with a rule-based template, Faver decouples the details of circuit verification, allowing the LLM to focus on the functionality itself. In our experiments on the SFT model and open-source models, Faver improved the model's generation accuracy by up to 14%. 

---
# A Novel Framework for Augmenting Rating Scale Tests with LLM-Scored Text Data 

**Authors**: Joe Watson, Ivan O'Conner, Chia-Wen Chen, Luning Sun, Fang Luo, David Stillwell  

**Link**: [PDF](https://arxiv.org/pdf/2510.08663)  

**Abstract**: Psychological assessments typically rely on structured rating scales, which cannot incorporate the rich nuance of a respondent's natural language. This study leverages recent LLM advances to harness qualitative data within a novel conceptual framework, combining LLM-scored text and traditional rating-scale items to create an augmented test. We demonstrate this approach using depression as a case study, developing and assessing the framework on a real-world sample of upper secondary students (n=693) and corresponding synthetic dataset (n=3,000). On held-out test sets, augmented tests achieved statistically significant improvements in measurement precision and accuracy. The information gain from the LLM items was equivalent to adding between 6.3 (real data) and 16.0 (synthetic data) items to the original 19-item test. Our approach marks a conceptual shift in automated scoring that bypasses its typical bottlenecks: instead of relying on pre-labelled data or complex expert-created rubrics, we empirically select the most informative LLM scoring instructions based on calculations of item information. This framework provides a scalable approach for leveraging the growing stream of transcribed text to enhance traditional psychometric measures, and we discuss its potential utility in clinical health and beyond. 

---
# DPCformer: An Interpretable Deep Learning Model for Genomic Prediction in Crops 

**Authors**: Pengcheng Deng, Kening Liu, Mengxi Zhou, Mingxi Li, Rui Yang, Chuzhe Cao, Maojun Wang, Zeyu Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2510.08662)  

**Abstract**: Genomic Selection (GS) uses whole-genome information to predict crop phenotypes and accelerate breeding. Traditional GS methods, however, struggle with prediction accuracy for complex traits and large datasets. We propose DPCformer, a deep learning model integrating convolutional neural networks with a self-attention mechanism to model complex genotype-phenotype relationships. We applied DPCformer to 13 traits across five crops (maize, cotton, tomato, rice, chickpea). Our approach uses an 8-dimensional one-hot encoding for SNP data, ordered by chromosome, and employs the PMF algorithm for feature selection. Evaluations show DPCformer outperforms existing methods. In maize datasets, accuracy for traits like days to tasseling and plant height improved by up to 2.92%. For cotton, accuracy gains for fiber traits reached 8.37%. On small-sample tomato data, the Pearson Correlation Coefficient for a key trait increased by up to 57.35%. In chickpea, the yield correlation was boosted by 16.62%. DPCformer demonstrates superior accuracy, robustness in small-sample scenarios, and enhanced interpretability, providing a powerful tool for precision breeding and addressing global food security challenges. 

---
# CATS-Linear: Classification Auxiliary Linear Model for Time Series Forecasting 

**Authors**: Zipo Jibao, Yingyi Fu, Xinyang Chen, Guoting Chen  

**Link**: [PDF](https://arxiv.org/pdf/2510.08661)  

**Abstract**: Recent research demonstrates that linear models achieve forecasting performance competitive with complex architectures, yet methodologies for enhancing linear models remain underexplored. Motivated by the hypothesis that distinct time series instances may follow heterogeneous linear mappings, we propose the Classification Auxiliary Trend-Seasonal Decoupling Linear Model CATS-Linear, employing Classification Auxiliary Channel-Independence (CACI). CACI dynamically routes instances to dedicated predictors via classification, enabling supervised channel design. We further analyze the theoretical expected risks of different channel settings. Additionally, we redesign the trend-seasonal decomposition architecture by adding a decoupling -- linear mapping -- recoupling framework for trend components and complex-domain linear projections for seasonal components. Extensive experiments validate that CATS-Linear with fixed hyperparameters achieves state-of-the-art accuracy comparable to hyperparameter-tuned baselines while delivering SOTA accuracy against fixed-hyperparameter counterparts. 

---
# Provably Robust Adaptation for Language-Empowered Foundation Models 

**Authors**: Yuni Lai, Xiaoyu Xue, Linghui Shen, Yulun Wu, Gaolei Li, Song Guo, Kai Zhou, Bin Xiao  

**Link**: [PDF](https://arxiv.org/pdf/2510.08659)  

**Abstract**: Language-empowered foundation models (LeFMs), such as CLIP and GraphCLIP, have transformed multimodal learning by aligning visual (or graph) features with textual representations, enabling powerful downstream capabilities like few-shot learning. However, the reliance on small, task-specific support datasets collected in open environments exposes these models to poisoning attacks, where adversaries manipulate the support samples to degrade performance. Existing defenses rely on empirical strategies, which lack formal guarantees and remain vulnerable to unseen and adaptive attacks. Certified robustness offers provable guarantees but has been largely unexplored for few-shot classifiers based on LeFMs. This study seeks to fill these critical gaps by proposing the first provably robust few-shot classifier that is tailored for LeFMs. We term our model Language-empowered Few-shot Certification (\textbf{LeFCert}). It integrates both textual and feature embeddings with an adaptive blending mechanism. To achieve provable robustness, we propose a twofold trimmed mean prototype and derive provable upper and lower bounds for classification scores, enabling certification under worst-case poisoning scenarios. To further enhance the performance, we extend LeFCert with two variants by considering a more realistic and tighter attack budget: LeFCert-L incorporates randomized smoothing to provide Lipschitz continuity and derive robustness under dual budget constraints, and LeFCert-C provides collective certification for scenarios where attackers distribute a shared poisoning budget across multiple samples. Experiments demonstrate that LeFCert achieves state-of-the-art performance, significantly improving both clean and certified accuracy compared to existing baselines. Despite its advanced robustness mechanisms, LeFCert is computationally efficient, making it practical for real-world applications. 

---
# Inner-Instance Normalization for Time Series Forecasting 

**Authors**: Zipo Jibao, Yingyi Fu, Xinyang Chen, Guoting Chen  

**Link**: [PDF](https://arxiv.org/pdf/2510.08657)  

**Abstract**: Real-world time series are influenced by numerous factors and exhibit complex non-stationary characteristics. Non-stationarity can lead to distribution shifts, where the statistical properties of time series change over time, negatively impacting model performance. Several instance normalization techniques have been proposed to address distribution shifts in time series forecasting. However, existing methods fail to account for shifts within individual instances, leading to suboptimal performance. To tackle inner-instance distribution shifts, we propose two novel point-level methods: Learning Distribution (LD) and Learning Conditional Distribution (LCD). LD eliminates internal discrepancies by fitting the internal distribution of input and output with different parameters at different time steps, while LCD utilizes neural networks to predict scaling coefficients of the output. We evaluate the performance of the two methods with various backbone models across public benchmarks and demonstrate the effectiveness of the point-level paradigm through comparative experiments. 

---
# A 3D Generation Framework from Cross Modality to Parameterized Primitive 

**Authors**: Yiming Liang, Huan Yu, Zili Wang, Shuyou Zhang, Guodong Yi, Jin Wang, Jianrong Tan  

**Link**: [PDF](https://arxiv.org/pdf/2510.08656)  

**Abstract**: Recent advancements in AI-driven 3D model generation have leveraged cross modality, yet generating models with smooth surfaces and minimizing storage overhead remain challenges. This paper introduces a novel multi-stage framework for generating 3D models composed of parameterized primitives, guided by textual and image inputs. In the framework, A model generation algorithm based on parameterized primitives, is proposed, which can identifies the shape features of the model constituent elements, and replace the elements with parameterized primitives with high quality surface. In addition, a corresponding model storage method is proposed, it can ensure the original surface quality of the model, while retaining only the parameters of parameterized primitives. Experiments on virtual scene dataset and real scene dataset demonstrate the effectiveness of our method, achieving a Chamfer Distance of 0.003092, a VIoU of 0.545, a F1-Score of 0.9139 and a NC of 0.8369, with primitive parameter files approximately 6KB in size. Our approach is particularly suitable for rapid prototyping of simple models. 

---
# Knowledge Graph Sparsification for GNN-based Rare Disease Diagnosis 

**Authors**: Premt Cara, Kamilia Zaripova, David Bani-Harouni, Nassir Navab, Azade Farshad  

**Link**: [PDF](https://arxiv.org/pdf/2510.08655)  

**Abstract**: Rare genetic disease diagnosis faces critical challenges: insufficient patient data, inaccessible full genome sequencing, and the immense number of possible causative genes. These limitations cause prolonged diagnostic journeys, inappropriate treatments, and critical delays, disproportionately affecting patients in resource-limited settings where diagnostic tools are scarce. We propose RareNet, a subgraph-based Graph Neural Network that requires only patient phenotypes to identify the most likely causal gene and retrieve focused patient subgraphs for targeted clinical investigation. RareNet can function as a standalone method or serve as a pre-processing or post-processing filter for other candidate gene prioritization methods, consistently enhancing their performance while potentially enabling explainable insights. Through comprehensive evaluation on two biomedical datasets, we demonstrate competitive and robust causal gene prediction and significant performance gains when integrated with other frameworks. By requiring only phenotypic data, which is readily available in any clinical setting, RareNet democratizes access to sophisticated genetic analysis, offering particular value for underserved populations lacking advanced genomic infrastructure. 

---
# Formalizing Style in Personal Narratives 

**Authors**: Gustave Cortal, Alain Finkel  

**Link**: [PDF](https://arxiv.org/pdf/2510.08649)  

**Abstract**: Personal narratives are stories authors construct to make meaning of their experiences. Style, the distinctive way authors use language to express themselves, is fundamental to how these narratives convey subjective experiences. Yet there is a lack of a formal framework for systematically analyzing these stylistic choices. We present a novel approach that formalizes style in personal narratives as patterns in the linguistic choices authors make when communicating subjective experiences. Our framework integrates three domains: functional linguistics establishes language as a system of meaningful choices, computer science provides methods for automatically extracting and analyzing sequential patterns, and these patterns are linked to psychological observations. Using language models, we automatically extract linguistic features such as processes, participants, and circumstances. We apply our framework to hundreds of dream narratives, including a case study on a war veteran with post-traumatic stress disorder. Analysis of his narratives uncovers distinctive patterns, particularly how verbal processes dominate over mental ones, illustrating the relationship between linguistic choices and psychological states. 

---
# Inverse-Free Wilson Loops for Transformers: A Practical Diagnostic for Invariance and Order Sensitivity 

**Authors**: Edward Y. Chang, Ethan Y. Chang  

**Link**: [PDF](https://arxiv.org/pdf/2510.08648)  

**Abstract**: Large language models can change answers under harmless edits that matter in practice: RAG outputs flip when passages are reordered, fine-tuning erodes invariances learned at pretraining, debate or chain-of-thought prompts take path-dependent routes, and compiler fusion or reordering perturbs logits near decision boundaries. These failures violate intended invariances, break continuous integration, and force teams to trade safety for speed. The effects are small yet distributed across layers and positions, sensitive to context length and evaluation order, and costly to repair with retraining or formal verification. We present WILSON, a minimal post-hoc diagnostic suite that converts simple loop and reordering checks on internal representations into system signals. WILSON combines an inverse-free curvature map over positions and layers, computed with JVPs and Hutchinson probes, with activation-level commutators that flag reorder risk. Signals are cheap to compute, model-agnostic for standard Transformers, and exported as thresholds and CSV artifacts for orchestrators. This enables concrete actions: guard RAG against order effects, catch fine-tuning regressions, stabilize debate pathways and long multi-turn contexts, and gate fusions or reorders in deployment. In short, WILSON helps anticipate failures and approve safe optimizations so reliability and throughput can improve together without changing model architecture or training. 

---
# Upfront Chain-of-Thought: A Cooperative Framework for Chain-of-Thought Compression 

**Authors**: Chengzhengxu Li, Xiaoming Liu, Zhaohan Zhang, Shaochu Zhang, Shengchao Liu, Guoxin Ma, Yu Lan, Chao Shen  

**Link**: [PDF](https://arxiv.org/pdf/2510.08647)  

**Abstract**: Recent developments have enabled advanced reasoning in Large Language Models (LLMs) via long Chain-of-Thought (CoT), while long CoT suffers from high computational costs and significant latency losses owing to the autoregressive nature of generative LLMs. CoT compression aims to improve efficiency in the reasoning process by reducing output length. Previous works trade reasoning efficiency by either laborious discrete prompt designing or the construction of external compressed CoT datasets that sacrifice key reasoning details. In this work, we propose Upfront CoT (UCoT): an efficient reasoning framework with upfront thought embedding to automate CoT compression. UCoT is a cooperative workflow involving a small model (compressor) and a large model (executor). The first stage of UCoT trains compressor to generate upfront thought embeddings rich in reasoning information for the executor, avoiding the drawbacks of manually designed prompts. The second stage optimizes executor to utilize upfront thought embeddings to derive the correct answer with short reasoning, using a reward mechanism. Extensive experiments show that UCoT maintains the powerful reasoning ability of executor while significantly reducing the length of CoT. It is worth mentioning that when applying UCoT to the Qwen2.5-7B-Instruct model, the usage of tokens on GSM8K dataset is reduced by 50\%, while the performance is 3.08\% higher than that of the state-of-the-art (SOTA) method. The code and dataset are in supplementary material. 

---
# Energy-Driven Steering: Reducing False Refusals in Large Language Models 

**Authors**: Eric Hanchen Jiang, Weixuan Ou, Run Liu, Shengyuan Pang, Guancheng Wan, Ranjie Duan, Wei Dong, Kai-Wei Chang, XiaoFeng Wang, Ying Nian Wu, Xinfeng Li  

**Link**: [PDF](https://arxiv.org/pdf/2510.08646)  

**Abstract**: Safety alignment of large language models (LLMs) faces a key challenge: current alignment techniques often only focus on improving safety against harmful prompts, causing LLMs to become over-cautious and refuse to respond to benign prompts. Therefore, a key objective of safe alignment is to enhance safety while simultaneously reducing false refusals. In this paper, we introduce Energy-Driven Steering (EDS), a novel, fine-tuning free framework designed to resolve this challenge through dynamic, inference-time intervention. We trained a lightweight, external Energy-Based Model (EBM) to assign high energy to undesirable (false refusal or jailbreak) states and low energy to desirable (helpful response or safe reject) ones. During inference, EBM maps the LLM's internal activations to an "energy landscape". We use the gradient of the energy function to dynamically steer the LLM's hidden states to low energy regions, correcting the model to generate a desirable response in real-time without modifying its weights. This method decouples behavioral control from the model's core knowledge, offering a flexible solution with minimal computational overhead. Extensive experiments across a wide range of models show our method successfully achieves this objective: it substantially lowers false refusal rates. For example, raising compliance on the ORB-H benchmark from 57.3% to 82.6% while maintaining the baseline safety performance. Our work presents an effective paradigm for building LLMs that achieve both low false refusal rates and high safety. 

---
# Automating Android Build Repair: Bridging the Reasoning-Execution Gap in LLM Agents with Domain-Specific Tools 

**Authors**: Ha Min Son, Huan Ren, Xin Liu, Zhe Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2510.08640)  

**Abstract**: Android is the largest mobile platform, yet automatically building applications remains a practical challenge. While Large Language Models (LLMs) show promise for code repair, their use for fixing Android build errors remains underexplored. To address this gap, we first introduce AndroidBuildBench, a benchmark of 1,019 build failures curated from the commit histories of 43 open-source Android projects. Each problem is paired with a verified solution from a subsequent commit, ensuring that fixes are feasible. Second, we propose GradleFixer, an LLM agent with domain-specific tools for inspecting and manipulating the Gradle build environment. GradleFixer achieves a resolve rate of 81.4% (pass@1), significantly outperforming a state-of-the-art coding agent that relies on a general-purpose shell. GradleFixer's success suggests that while LLMs possess the high-level knowledge to solve these failures, they struggle to translate this knowledge into effective low-level actions using a general-purpose shell. We demonstrate the effectiveness of a strategy we term Tool Bridging, which replaces general-purpose shell commands with domain-aware abstractions. We hypothesize this approach works through two mechanisms: 1) it provides tools in an API-like format that LLMs use more reliably, and 2) it constrains the action space to relevant operations. This approach bridges the gap between the model's high-level reasoning and effective low-level execution. 

---
# Into the Rabbit Hull: From Task-Relevant Concepts in DINO to Minkowski Geometry 

**Authors**: Thomas Fel, Binxu Wang, Michael A. Lepori, Matthew Kowal, Andrew Lee, Randall Balestriero, Sonia Joseph, Ekdeep S. Lubana, Talia Konkle, Demba Ba, Martin Wattenberg  

**Link**: [PDF](https://arxiv.org/pdf/2510.08638)  

**Abstract**: DINOv2 is routinely deployed to recognize objects, scenes, and actions; yet the nature of what it perceives remains unknown. As a working baseline, we adopt the Linear Representation Hypothesis (LRH) and operationalize it using SAEs, producing a 32,000-unit dictionary that serves as the interpretability backbone of our study, which unfolds in three parts.
In the first part, we analyze how different downstream tasks recruit concepts from our learned dictionary, revealing functional specialization: classification exploits "Elsewhere" concepts that fire everywhere except on target objects, implementing learned negations; segmentation relies on boundary detectors forming coherent subspaces; depth estimation draws on three distinct monocular depth cues matching visual neuroscience principles.
Following these functional results, we analyze the geometry and statistics of the concepts learned by the SAE. We found that representations are partly dense rather than strictly sparse. The dictionary evolves toward greater coherence and departs from maximally orthogonal ideals (Grassmannian frames). Within an image, tokens occupy a low dimensional, locally connected set persisting after removing position. These signs suggest representations are organized beyond linear sparsity alone.
Synthesizing these observations, we propose a refined view: tokens are formed by combining convex mixtures of archetypes (e.g., a rabbit among animals, brown among colors, fluffy among textures). This structure is grounded in Gardenfors' conceptual spaces and in the model's mechanism as multi-head attention produces sums of convex mixtures, defining regions bounded by archetypes. We introduce the Minkowski Representation Hypothesis (MRH) and examine its empirical signatures and implications for interpreting vision-transformer representations. 

---
# Hi-OSCAR: Hierarchical Open-set Classifier for Human Activity Recognition 

**Authors**: Conor McCarthy, Loes Quirijnen, Jan Peter van Zandwijk, Zeno Geradts, Marcel Worring  

**Link**: [PDF](https://arxiv.org/pdf/2510.08635)  

**Abstract**: Within Human Activity Recognition (HAR), there is an insurmountable gap between the range of activities performed in life and those that can be captured in an annotated sensor dataset used in training. Failure to properly handle unseen activities seriously undermines any HAR classifier's reliability. Additionally within HAR, not all classes are equally dissimilar, some significantly overlap or encompass other sub-activities. Based on these observations, we arrange activity classes into a structured hierarchy. From there, we propose Hi-OSCAR: a Hierarchical Open-set Classifier for Activity Recognition, that can identify known activities at state-of-the-art accuracy while simultaneously rejecting unknown activities. This not only enables open-set classification, but also allows for unknown classes to be localized to the nearest internal node, providing insight beyond a binary "known/unknown" classification. To facilitate this and future open-set HAR research, we collected a new dataset: NFI_FARED. NFI_FARED contains data from multiple subjects performing nineteen activities from a range of contexts, including daily living, commuting, and rapid movements, which is fully public and available for download. 

---
# From What to Why: Thought-Space Recommendation with Small Language Models 

**Authors**: Prosenjit Biswas, Pervez Shaik, Abhinav Thorat, Ravi Kolla, Niranjan Pedanekar  

**Link**: [PDF](https://arxiv.org/pdf/2510.08626)  

**Abstract**: Large Language Models (LLMs) have advanced recommendation capabilities through enhanced reasoning, but pose significant challenges for real-world deployment due to high inference costs. Conversely, while Small Language Models (SLMs) offer an efficient alternative, their reasoning capabilities for recommendation remain underexplored. Existing systems often use natural language rationales merely as unsupervised descriptive text, failing to harness their full potential as learning signals. In this work our main idea is to create a common understanding of user and items across multiple domains called Thought Space with SLMs instead of using LLMs' distilled knowledge. To that end we propose PULSE (Preference Understanding by Latent Semantic Embeddings), a framework that treats SLM-generated rationales as director learning signals, supervising them with interaction histories to jointly model user actions (what) and their semantic drivers (why). Existing methods consider only interactions such as sequences and embeddings, whereas PULSE treats rationales as first-class signals, this novel design yields embeddings that are more robust and generalizable. Extensive experiments demonstrate that PULSE outperforms leading ID, Collaborative Filtering (CF), and LLM-based sequential recommendation models across multiple benchmark datasets. Furthermore, PULSE exhibits superior transferability in cross-domain recommendation and demonstrates strong performance on downstream tasks such as reasoning-oriented question answering. Our code is available \href{this https URL}{here}. 

---
# Impact of LLMs on Team Collaboration in Software Development 

**Authors**: Devang Dhanuka  

**Link**: [PDF](https://arxiv.org/pdf/2510.08612)  

**Abstract**: Large Language Models (LLMs) are increasingly being integrated into software development processes, with the potential to transform team workflows and productivity. This paper investigates how LLMs affect team collaboration throughout the Software Development Life Cycle (SDLC). We reframe and update a prior study with recent developments as of 2025, incorporating new literature and case studies. We outline the problem of collaboration hurdles in SDLC and explore how LLMs can enhance productivity, communication, and decision-making in a team context. Through literature review, industry examples, a team survey, and two case studies, we assess the impact of LLM-assisted tools (such as code generation assistants and AI-powered project management agents) on collaborative software engineering practices. Our findings indicate that LLMs can significantly improve efficiency (by automating repetitive tasks and documentation), enhance communication clarity, and aid cross-functional collaboration, while also introducing new challenges like model limitations and privacy concerns. We discuss these benefits and challenges, present research questions guiding the investigation, evaluate threats to validity, and suggest future research directions including domain-specific model customization, improved integration into development tools, and robust strategies for ensuring trust and security. 

---
# Relative Positioning Based Code Chunking Method For Rich Context Retrieval In Repository Level Code Completion Task With Code Language Model 

**Authors**: Imranur Rahman, Md Rayhanur Rahman  

**Link**: [PDF](https://arxiv.org/pdf/2510.08610)  

**Abstract**: Code completion can help developers improve efficiency and ease the development lifecycle. Although code completion is available in modern integrated development environments (IDEs), research lacks in determining what makes a good context for code completion based on the information available to the IDEs for the large language models (LLMs) to perform better. In this paper, we describe an effective context collection strategy to assist the LLMs in performing better at code completion tasks. The key idea of our strategy is to preprocess the repository into smaller code chunks and later use syntactic and semantic similarity-based code chunk retrieval with relative positioning. We found that code chunking and relative positioning of the chunks in the final context improve the performance of code completion tasks. 

---
# MMA-ASIA: A Multilingual and Multimodal Alignment Framework for Culturally-Grounded Evaluation 

**Authors**: Weihua Zheng, Zhengyuan Liu, Tanmoy Chakraborty, Weiwen Xu, Xiaoxue Gao, Bryan Chen Zhengyu Tan, Bowei Zou, Chang Liu, Yujia Hu, Xing Xie, Xiaoyuan Yi, Jing Yao, Chaojun Wang, Long Li, Rui Liu, Huiyao Liu, Koji Inoue, Ryuichi Sumida, Tatsuya Kawahara, Fan Xu, Lingyu Ye, Wei Tian, Dongjun Kim, Jimin Jung, Jaehyung Seo, Nadya Yuki Wangsajaya, Pham Minh Duc, Ojasva Saxena, Palash Nandi, Xiyan Tao, Wiwik Karlina, Tuan Luong, Keertana Arun Vasan, Roy Ka-Wei Lee, Nancy F. Chen  

**Link**: [PDF](https://arxiv.org/pdf/2510.08608)  

**Abstract**: Large language models (LLMs) are now used worldwide, yet their multimodal understanding and reasoning often degrade outside Western, high-resource settings. We propose MMA-ASIA, a comprehensive framework to evaluate LLMs' cultural awareness with a focus on Asian contexts. MMA-ASIA centers on a human-curated, multilingual, and multimodally aligned multiple-choice benchmark covering 8 Asian countries and 10 languages, comprising 27,000 questions; over 79 percent require multi-step reasoning grounded in cultural context, moving beyond simple memorization. To our knowledge, this is the first dataset aligned at the input level across three modalities: text, image (visual question answering), and speech. This enables direct tests of cross-modal transfer. Building on this benchmark, we propose a five-dimensional evaluation protocol that measures: (i) cultural-awareness disparities across countries, (ii) cross-lingual consistency, (iii) cross-modal consistency, (iv) cultural knowledge generalization, and (v) grounding validity. To ensure rigorous assessment, a Cultural Awareness Grounding Validation Module detects "shortcut learning" by checking whether the requisite cultural knowledge supports correct answers. Finally, through comparative model analysis, attention tracing, and an innovative Vision-ablated Prefix Replay (VPR) method, we probe why models diverge across languages and modalities, offering actionable insights for building culturally reliable multimodal LLMs. 

---
# Centering Emotion Hotspots: Multimodal Local-Global Fusion and Cross-Modal Alignment for Emotion Recognition in Conversations 

**Authors**: Yu Liu, Hanlei Shi, Haoxun Li, Yuqing Sun, Yuxuan Ding, Linlin Gong, Leyuan Qu, Taihao Li  

**Link**: [PDF](https://arxiv.org/pdf/2510.08606)  

**Abstract**: Emotion Recognition in Conversations (ERC) is hard because discriminative evidence is sparse, localized, and often asynchronous across modalities. We center ERC on emotion hotspots and present a unified model that detects per-utterance hotspots in text, audio, and video, fuses them with global features via Hotspot-Gated Fusion, and aligns modalities using a routed Mixture-of-Aligners; a cross-modal graph encodes conversational structure. This design focuses modeling on salient spans, mitigates misalignment, and preserves context. Experiments on standard ERC benchmarks show consistent gains over strong baselines, with ablations confirming the contributions of HGF and MoA. Our results point to a hotspot-centric view that can inform future multimodal learning, offering a new perspective on modality fusion in ERC. 

---
# Toward a Safer Web: Multilingual Multi-Agent LLMs for Mitigating Adversarial Misinformation Attacks 

**Authors**: Nouar Aldahoul, Yasir Zaki  

**Link**: [PDF](https://arxiv.org/pdf/2510.08605)  

**Abstract**: The rapid spread of misinformation on digital platforms threatens public discourse, emotional stability, and decision-making. While prior work has explored various adversarial attacks in misinformation detection, the specific transformations examined in this paper have not been systematically studied. In particular, we investigate language-switching across English, French, Spanish, Arabic, Hindi, and Chinese, followed by translation. We also study query length inflation preceding summarization and structural reformatting into multiple-choice questions. In this paper, we present a multilingual, multi-agent large language model framework with retrieval-augmented generation that can be deployed as a web plugin into online platforms. Our work underscores the importance of AI-driven misinformation detection in safeguarding online factual integrity against diverse attacks, while showcasing the feasibility of plugin-based deployment for real-world web applications. 

---
# LatentBreak: Jailbreaking Large Language Models through Latent Space Feedback 

**Authors**: Raffaele Mura, Giorgio Piras, Kamilė Lukošiūtė, Maura Pintor, Amin Karbasi, Battista Biggio  

**Link**: [PDF](https://arxiv.org/pdf/2510.08604)  

**Abstract**: Jailbreaks are adversarial attacks designed to bypass the built-in safety mechanisms of large language models. Automated jailbreaks typically optimize an adversarial suffix or adapt long prompt templates by forcing the model to generate the initial part of a restricted or harmful response. In this work, we show that existing jailbreak attacks that leverage such mechanisms to unlock the model response can be detected by a straightforward perplexity-based filtering on the input prompt. To overcome this issue, we propose LatentBreak, a white-box jailbreak attack that generates natural adversarial prompts with low perplexity capable of evading such defenses. LatentBreak substitutes words in the input prompt with semantically-equivalent ones, preserving the initial intent of the prompt, instead of adding high-perplexity adversarial suffixes or long templates. These words are chosen by minimizing the distance in the latent space between the representation of the adversarial prompt and that of harmless requests. Our extensive evaluation shows that LatentBreak leads to shorter and low-perplexity prompts, thus outperforming competing jailbreak algorithms against perplexity-based filters on multiple safety-aligned models. 

---
# Mnemosyne: An Unsupervised, Human-Inspired Long-Term Memory Architecture for Edge-Based LLMs 

**Authors**: Aneesh Jonelagadda, Christina Hahn, Haoze Zheng, Salvatore Penachio  

**Link**: [PDF](https://arxiv.org/pdf/2510.08601)  

**Abstract**: Long-term memory is essential for natural, realistic dialogue. However, current large language model (LLM) memory systems rely on either brute-force context expansion or static retrieval pipelines that fail on edge-constrained devices. We introduce Mnemosyne, an unsupervised, human-inspired long-term memory architecture designed for edge-based LLMs. Our approach uses graph-structured storage, modular substance and redundancy filters, memory committing and pruning mechanisms, and probabilistic recall with temporal decay and refresh processes modeled after human memory. Mnemosyne also introduces a concentrated "core summary" efficiently derived from a fixed-length subset of the memory graph to capture the user's personality and other domain-specific long-term details such as, using healthcare application as an example, post-recovery ambitions and attitude towards care. Unlike existing retrieval-augmented methods, Mnemosyne is designed for use in longitudinal healthcare assistants, where repetitive and semantically similar but temporally distinct conversations are limited by naive retrieval. In experiments with longitudinal healthcare dialogues, Mnemosyne demonstrates the highest win rate of 65.8% in blind human evaluations of realism and long-term memory capability compared to a baseline RAG win rate of 31.1%. Mnemosyne also achieves current highest LoCoMo benchmark scores in temporal reasoning and single-hop retrieval compared to other same-backboned techniques. Further, the average overall score of 54.6% was second highest across all methods, beating commonly used Mem0 and OpenAI baselines among others. This demonstrates that improved factual recall, enhanced temporal reasoning, and much more natural user-facing responses can be feasible with an edge-compatible and easily transferable unsupervised memory architecture. 

---
# Recover-LoRA: Data-Free Accuracy Recovery of Degraded Language Models via Low-Rank Adaptation 

**Authors**: Devleena Das, Rajeev Patwari, Ashish Sirasao  

**Link**: [PDF](https://arxiv.org/pdf/2510.08600)  

**Abstract**: Inference optimizations such as quantization, pruning, format and datatype conversion, model export, and serialization can lead to functional degradations in language model task performance. While most efforts on performance recovery for deployment focus on robust quantization techniques, we focus on recovering model accuracies from any sources that degrade model weights, such as improper model serialization. In this work, we propose Recover-LoRA, a lightweight and dataset agnostic method to recover accuracy in degraded models. Recover-LoRA uses synthetic data and logit distillation to learn LoRA adapters on selective layers that facilitate aligning the degraded model to its full precision model. We investigate the utility of Recover-LoRA across a diverse set of small language models (SLMs), including models with varying attention architectures, multi-head attention (MHA) and group-query attention (GQA), as well as several evaluation datasets. Our results show that Recover-LoRA recovers model accuracies by 5-17% on MHA and GQA SLMs. 

---
# BaldWhisper: Faster Whisper with Head Shearing and Layer Merging 

**Authors**: Yaya Sy, Christophe Cerisara, Irina Illina  

**Link**: [PDF](https://arxiv.org/pdf/2510.08599)  

**Abstract**: Pruning large pre-trained transformers for low-resource languages is challenging, as it often requires massive retraining data to recover performance. For instance, Distill-Whisper prunes Whisper by 40% and retrains on 21,000 hours of speech, far beyond what is available for most languages. Can Whisper be made lighter and faster for edge devices in data-scarce settings? Focusing on Bambara with only 32h of speech-to-text data, we propose a new pruning recipe. Instead of vocabulary pruning, which is unsuitable due to frequent code-switching by Bambara speakers, we compress the embeddings with low-rank decomposition and feature distillation. Rather than removing layers, we merge them to limit performance loss. The final model preserves 90% of the original performance while being 48% smaller and 2.15x faster on a MacBook Air M1. 

---
# Hierarchical Self-Supervised Representation Learning for Depression Detection from Speech 

**Authors**: Yuxin Li, Eng Siong Chng, Cuntai Guan  

**Link**: [PDF](https://arxiv.org/pdf/2510.08593)  

**Abstract**: Speech-based depression detection (SDD) is a promising, non-invasive alternative to traditional clinical assessments. However, it remains limited by the difficulty of extracting meaningful features and capturing sparse, heterogeneous depressive cues over time. Pretrained self-supervised learning (SSL) models such as WavLM provide rich, multi-layer speech representations, yet most existing SDD methods rely only on the final layer or search for a single best-performing one. These approaches often overfit to specific datasets and fail to leverage the full hierarchical structure needed to detect subtle and persistent depression signals.
To address this challenge, we propose HAREN-CTC, a novel architecture that integrates multi-layer SSL features using cross-attention within a multitask learning framework, combined with Connectionist Temporal Classification loss to handle sparse temporal supervision. HAREN-CTC comprises two key modules: a Hierarchical Adaptive Clustering module that reorganizes SSL features into complementary embeddings, and a Cross-Modal Fusion module that models inter-layer dependencies through cross-attention. The CTC objective enables alignment-aware training, allowing the model to track irregular temporal patterns of depressive speech cues.
We evaluate HAREN-CTC under both an upper-bound setting with standard data splits and a generalization setting using five-fold cross-validation. The model achieves state-of-the-art macro F1-scores of 0.81 on DAIC-WOZ and 0.82 on MODMA, outperforming prior methods across both evaluation scenarios. 

---
# Less Diverse, Less Safe: The Indirect But Pervasive Risk of Test-Time Scaling in Large Language Models 

**Authors**: Shahriar Kabir Nahin, Hadi Askari, Muhao Chen, Anshuman Chhabra  

**Link**: [PDF](https://arxiv.org/pdf/2510.08592)  

**Abstract**: Test-Time Scaling (TTS) improves LLM reasoning by exploring multiple candidate responses and then operating over this set to find the best output. A tacit premise behind TTS is that sufficiently diverse candidate pools enhance reliability. In this work, we show that this assumption in TTS introduces a previously unrecognized failure mode. When candidate diversity is curtailed, even by a modest amount, TTS becomes much more likely to produce unsafe outputs. We present a reference-guided diversity reduction protocol (RefDiv) that serves as a diagnostic attack to stress test TTS pipelines. Through extensive experiments across four open-source models (Qwen3, Mistral, Llama3.1, Gemma3) and two widely used TTS strategies (Monte Carlo Tree Search and Best-of-N), constraining diversity consistently signifies the rate at which TTS produces unsafe results. The effect is often stronger than that produced by prompts directly with high adversarial intent scores. This observed phenomenon also transfers across TTS strategies and to closed-source models (e.g. OpenAI o3 and Gemini-2.5-Pro), thus indicating that this is a general and extant property of TTS rather than a model-specific artifact. Additionally, we find that numerous widely used safety guardrail classifiers (e.g. Llama-Guard and OpenAI Moderation API), are unable to flag the adversarial input prompts generated by RefDiv, demonstrating that existing defenses offer limited protection against this diversity-driven failure mode. Through this work, we hope to motivate future research on designing robust TTS strategies that are both effective and secure against diversity-targeted stress tests as illustrated by RefDiv. 

---
# The Enduring Dominance of Deep Neural Networks: A Critical Analysis of the Fundamental Limitations of Quantum Machine Learning and Spiking Neural Networks 

**Authors**: Takehiro Ishikawa  

**Link**: [PDF](https://arxiv.org/pdf/2510.08591)  

**Abstract**: Recent advancements in QML and SNNs have generated considerable excitement, promising exponential speedups and brain-like energy efficiency to revolutionize AI. However, this paper argues that they are unlikely to displace DNNs in the near term. QML struggles with adapting backpropagation due to unitary constraints, measurement-induced state collapse, barren plateaus, and high measurement overheads, exacerbated by the limitations of current noisy intermediate-scale quantum hardware, overfitting risks due to underdeveloped regularization techniques, and a fundamental misalignment with machine learning's generalization. SNNs face restricted representational bandwidth, struggling with long-range dependencies and semantic encoding in language tasks due to their discrete, spike-based processing. Furthermore, the goal of faithfully emulating the brain might impose inherent inefficiencies like cognitive biases, limited working memory, and slow learning speeds. Even their touted energy-efficient advantages are overstated; optimized DNNs with quantization can outperform SNNs in energy costs under realistic conditions. Finally, SNN training incurs high computational overhead from temporal unfolding. In contrast, DNNs leverage efficient backpropagation, robust regularization, and innovations in LRMs that shift scaling to inference-time compute, enabling self-improvement via RL and search algorithms like MCTS while mitigating data scarcity. This superiority is evidenced by recent models such as xAI's Grok-4 Heavy, which advances SOTA performance, and gpt-oss-120b, which surpasses or approaches the performance of leading industry models despite its modest 120-billion-parameter size deployable on a single 80GB GPU. Furthermore, specialized ASICs amplify these efficiency gains. Ultimately, QML and SNNs may serve niche hybrid roles, but DNNs remain the dominant, practical paradigm for AI advancement. 

---
# Beyond CNNs: Efficient Fine-Tuning of Multi-Modal LLMs for Object Detection on Low-Data Regimes 

**Authors**: Nirmal Elamon, Rouzbeh Davoudi  

**Link**: [PDF](https://arxiv.org/pdf/2510.08589)  

**Abstract**: The field of object detection and understanding is rapidly evolving, driven by advances in both traditional CNN-based models and emerging multi-modal large language models (LLMs). While CNNs like ResNet and YOLO remain highly effective for image-based tasks, recent transformer-based LLMs introduce new capabilities such as dynamic context reasoning, language-guided prompts, and holistic scene understanding. However, when used out-of-the-box, the full potential of LLMs remains underexploited, often resulting in suboptimal performance on specialized visual tasks. In this work, we conduct a comprehensive comparison of fine-tuned traditional CNNs, zero-shot pre-trained multi-modal LLMs, and fine-tuned multi-modal LLMs on the challenging task of artificial text overlay detection in images. A key contribution of our study is demonstrating that LLMs can be effectively fine-tuned on very limited data (fewer than 1,000 images) to achieve up to 36% accuracy improvement, matching or surpassing CNN-based baselines that typically require orders of magnitude more data. By exploring how language-guided models can be adapted for precise visual understanding with minimal supervision, our work contributes to the broader effort of bridging vision and language, offering novel insights into efficient cross-modal learning strategies. These findings highlight the adaptability and data efficiency of LLM-based approaches for real-world object detection tasks and provide actionable guidance for applying multi-modal transformers in low-resource visual environments. To support continued progress in this area, we have made the code used to fine-tune the models available in our GitHub, enabling future improvements and reuse in related applications. 

---
# EGSTalker: Real-Time Audio-Driven Talking Head Generation with Efficient Gaussian Deformation 

**Authors**: Tianheng Zhu, Yinfeng Yu, Liejun Wang, Fuchun Sun, Wendong Zheng  

**Link**: [PDF](https://arxiv.org/pdf/2510.08587)  

**Abstract**: This paper presents EGSTalker, a real-time audio-driven talking head generation framework based on 3D Gaussian Splatting (3DGS). Designed to enhance both speed and visual fidelity, EGSTalker requires only 3-5 minutes of training video to synthesize high-quality facial animations. The framework comprises two key stages: static Gaussian initialization and audio-driven deformation. In the first stage, a multi-resolution hash triplane and a Kolmogorov-Arnold Network (KAN) are used to extract spatial features and construct a compact 3D Gaussian representation. In the second stage, we propose an Efficient Spatial-Audio Attention (ESAA) module to fuse audio and spatial cues, while KAN predicts the corresponding Gaussian deformations. Extensive experiments demonstrate that EGSTalker achieves rendering quality and lip-sync accuracy comparable to state-of-the-art methods, while significantly outperforming them in inference speed. These results highlight EGSTalker's potential for real-time multimedia applications. 

---
# Dynamic Stress Detection: A Study of Temporal Progression Modelling of Stress in Speech 

**Authors**: Vishakha Lall, Yisi Liu  

**Link**: [PDF](https://arxiv.org/pdf/2510.08586)  

**Abstract**: Detecting psychological stress from speech is critical in high-pressure settings. While prior work has leveraged acoustic features for stress detection, most treat stress as a static label. In this work, we model stress as a temporally evolving phenomenon influenced by historical emotional state. We propose a dynamic labelling strategy that derives fine-grained stress annotations from emotional labels and introduce cross-attention-based sequential models, a Unidirectional LSTM and a Transformer Encoder, to capture temporal stress progression. Our approach achieves notable accuracy gains on MuSE (+5%) and StressID (+18%) over existing baselines, and generalises well to a custom real-world dataset. These results highlight the value of modelling stress as a dynamic construct in speech. 

---
# Articulation-Informed ASR: Integrating Articulatory Features into ASR via Auxiliary Speech Inversion and Cross-Attention Fusion 

**Authors**: Ahmed Adel Attia, Jing Liu, Carol Espy Wilson  

**Link**: [PDF](https://arxiv.org/pdf/2510.08585)  

**Abstract**: Prior works have investigated the use of articulatory features as complementary representations for automatic speech recognition (ASR), but their use was largely confined to shallow acoustic models. In this work, we revisit articulatory information in the era of deep learning and propose a framework that leverages articulatory representations both as an auxiliary task and as a pseudo-input to the recognition model. Specifically, we employ speech inversion as an auxiliary prediction task, and the predicted articulatory features are injected into the model as a query stream in a cross-attention module with acoustic embeddings as keys and values. Experiments on LibriSpeech demonstrate that our approach yields consistent improvements over strong transformer-based baselines, particularly under low-resource conditions. These findings suggest that articulatory features, once sidelined in ASR research, can provide meaningful benefits when reintroduced with modern architectures. 

---
# Evaluating Hallucinations in Multimodal LLMs with Spoken Queries under Diverse Acoustic Conditions 

**Authors**: Hansol Park, Hoseong Ahn, Junwon Moon, Yejin Lee, Kyuhong Shim  

**Link**: [PDF](https://arxiv.org/pdf/2510.08581)  

**Abstract**: Hallucinations in vision-language models have been extensively studied using benchmarks that probe reliability in image-text settings. In contrast, the effect of spoken queries on multimodal hallucinations remains largely unexplored, despite the growing role of voice-driven interfaces. In this work, we investigate how spoken input influences hallucinations in multimodal large language models. We present RePOPE-Spk, an audio-augmented extension of the RePOPE benchmark, where queries are provided as speech under diverse acoustic conditions. Using RePOPE-Spk, we systematically evaluate both proprietary and open-source models. Experimental results show that hallucinations escalate when queries are spoken rather than written: error rates increase by 3% under clean speech and by up to 20% with environmental noise. Input order and query length further affect robustness, while strategies such as many-shot prompting and chain-of-thought reasoning offer partial but insufficient mitigation. These findings highlight a critical and underexplored challenge, opening new directions for building reliable voice interface systems. 

---
# LadderSym: A Multimodal Interleaved Transformer for Music Practice Error Detection 

**Authors**: Benjamin Shiue-Hal Chou, Purvish Jajal, Nick John Eliopoulos, James C. Davis, George K. Thiruvathukal, Kristen Yeon-Ji Yun, Yung-Hsiang Lu  

**Link**: [PDF](https://arxiv.org/pdf/2510.08580)  

**Abstract**: Music learners can greatly benefit from tools that accurately detect errors in their practice. Existing approaches typically compare audio recordings to music scores using heuristics or learnable models. This paper introduces \textit{LadderSym}, a novel Transformer-based method for music error detection. \textit{LadderSym} is guided by two key observations about the state-of-the-art approaches: (1) late fusion limits inter-stream alignment and cross-modality comparison capability; and (2) reliance on score audio introduces ambiguity in the frequency spectrum, degrading performance in music with concurrent notes. To address these limitations, \textit{LadderSym} introduces (1) a two-stream encoder with inter-stream alignment modules to improve audio comparison capabilities and error detection F1 scores, and (2) a multimodal strategy that leverages both audio and symbolic scores by incorporating symbolic representations as decoder prompts, reducing ambiguity and improving F1 scores. We evaluate our method on the \textit{MAESTRO-E} and \textit{CocoChorales-E} datasets by measuring the F1 score for each note category. Compared to the previous state of the art, \textit{LadderSym} more than doubles F1 for missed notes on \textit{MAESTRO-E} (26.8\% $\rightarrow$ 56.3\%) and improves extra note detection by 14.4 points (72.0\% $\rightarrow$ 86.4\%). Similar gains are observed on \textit{CocoChorales-E}. This work introduces general insights about comparison models that could inform sequence evaluation tasks for reinforcement Learning, human skill assessment, and model evaluation. 

---
# AgenticAD: A Specialized Multiagent System Framework for Holistic Alzheimer Disease Management 

**Authors**: Adib Bazgir, Amir Habibdoust, Xing Song, Yuwen Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2510.08578)  

**Abstract**: Alzheimer's disease (AD) presents a complex, multifaceted challenge to patients, caregivers, and the healthcare system, necessitating integrated and dynamic support solutions. While artificial intelligence (AI) offers promising avenues for intervention, current applications are often siloed, addressing singular aspects of the disease such as diagnostics or caregiver support without systemic integration. This paper proposes a novel methodological framework for a comprehensive, multi-agent system (MAS) designed for holistic Alzheimer's disease management. The objective is to detail the architecture of a collaborative ecosystem of specialized AI agents, each engineered to address a distinct challenge in the AD care continuum, from caregiver support and multimodal data analysis to automated research and clinical data interpretation. The proposed framework is composed of eight specialized, interoperable agents. These agents are categorized by function: (1) Caregiver and Patient Support, (2) Data Analysis and Research, and (3) Advanced Multimodal Workflows. The methodology details the technical architecture of each agent, leveraging a suite of advanced technologies including large language models (LLMs) such as GPT-4o and Gemini, multi-agent orchestration frameworks, Retrieval-Augmented Generation (RAG) for evidence-grounded responses, and specialized tools for web scraping, multimodal data processing, and in-memory database querying. This paper presents a detailed architectural blueprint for an integrated AI ecosystem for AD care. By moving beyond single-purpose tools to a collaborative, multi-agent paradigm, this framework establishes a foundation for developing more adaptive, personalized, and proactive solutions. This methodological approach aims to pave the way for future systems capable of synthesizing diverse data streams to improve patient outcomes and reduce caregiver burden. 

---
# Comparative Analysis of Large Language Models for the Machine-Assisted Resolution of User Intentions 

**Authors**: Justus Flerlage, Alexander Acker, Odej Kao  

**Link**: [PDF](https://arxiv.org/pdf/2510.08576)  

**Abstract**: Large Language Models (LLMs) have emerged as transformative tools for natural language understanding and user intent resolution, enabling tasks such as translation, summarization, and, increasingly, the orchestration of complex workflows. This development signifies a paradigm shift from conventional, GUI-driven user interfaces toward intuitive, language-first interaction paradigms. Rather than manually navigating applications, users can articulate their objectives in natural language, enabling LLMs to orchestrate actions across multiple applications in a dynamic and contextual manner. However, extant implementations frequently rely on cloud-based proprietary models, which introduce limitations in terms of privacy, autonomy, and scalability. For language-first interaction to become a truly robust and trusted interface paradigm, local deployment is not merely a convenience; it is an imperative. This limitation underscores the importance of evaluating the feasibility of locally deployable, open-source, and open-access LLMs as foundational components for future intent-based operating systems. In this study, we examine the capabilities of several open-source and open-access models in facilitating user intention resolution through machine assistance. A comparative analysis is conducted against OpenAI's proprietary GPT-4-based systems to assess performance in generating workflows for various user intentions. The present study offers empirical insights into the practical viability, performance trade-offs, and potential of open LLMs as autonomous, locally operable components in next-generation operating systems. The results of this study inform the broader discussion on the decentralization and democratization of AI infrastructure and point toward a future where user-device interaction becomes more seamless, adaptive, and privacy-conscious through locally embedded intelligence. 

---
# PyNoetic: A modular python framework for no-code development of EEG brain-computer interfaces 

**Authors**: Gursimran Singh, Aviral Chharia, Rahul Upadhyay, Vinay Kumar, Luca Longo  

**Link**: [PDF](https://arxiv.org/pdf/2509.00670)  

**Abstract**: Electroencephalography (EEG)-based Brain-Computer Interfaces (BCIs) have emerged as a transformative technology with applications spanning robotics, virtual reality, medicine, and rehabilitation. However, existing BCI frameworks face several limitations, including a lack of stage-wise flexibility essential for experimental research, steep learning curves for researchers without programming expertise, elevated costs due to reliance on proprietary software, and a lack of all-inclusive features leading to the use of multiple external tools affecting research outcomes. To address these challenges, we present PyNoetic, a modular BCI framework designed to cater to the diverse needs of BCI research. PyNoetic is one of the very few frameworks in Python that encompasses the entire BCI design pipeline, from stimulus presentation and data acquisition to channel selection, filtering, feature extraction, artifact removal, and finally simulation and visualization. Notably, PyNoetic introduces an intuitive and end-to-end GUI coupled with a unique pick-and-place configurable flowchart for no-code BCI design, making it accessible to researchers with minimal programming experience. For advanced users, it facilitates the seamless integration of custom functionalities and novel algorithms with minimal coding, ensuring adaptability at each design stage. PyNoetic also includes a rich array of analytical tools such as machine learning models, brain-connectivity indices, systematic testing functionalities via simulation, and evaluation methods of novel paradigms. PyNoetic's strengths lie in its versatility for both offline and real-time BCI development, which streamlines the design process, allowing researchers to focus on more intricate aspects of BCI development and thus accelerate their research endeavors. Project Website: this https URL 

---
# Deep Sparse Representation-based Classification 

**Authors**: Mahdi Abavisani, Vishal M. Patel  

**Link**: [PDF](https://arxiv.org/pdf/1904.11093)  

**Abstract**: We present a transductive deep learning-based formulation for the sparse representation-based classification (SRC) method. The proposed network consists of a convolutional autoencoder along with a fully-connected layer. The role of the autoencoder network is to learn robust deep features for classification. On the other hand, the fully-connected layer, which is placed in between the encoder and the decoder networks, is responsible for finding the sparse representation. The estimated sparse codes are then used for classification. Various experiments on three different datasets show that the proposed network leads to sparse representations that give better classification results than state-of-the-art SRC methods. The source code is available at: this http URL. 

---
# Improving the Performance of Unimodal Dynamic Hand-Gesture Recognition with Multimodal Training 

**Authors**: Mahdi Abavisani, Hamid Reza Vaezi Joze, Vishal M. Patel  

**Link**: [PDF](https://arxiv.org/pdf/1812.06145)  

**Abstract**: We present an efficient approach for leveraging the knowledge from multiple modalities in training unimodal 3D convolutional neural networks (3D-CNNs) for the task of dynamic hand gesture recognition. Instead of explicitly combining multimodal information, which is commonplace in many state-of-the-art methods, we propose a different framework in which we embed the knowledge of multiple modalities in individual networks so that each unimodal network can achieve an improved performance. In particular, we dedicate separate networks per available modality and enforce them to collaborate and learn to develop networks with common semantics and better representations. We introduce a "spatiotemporal semantic alignment" loss (SSA) to align the content of the features from different networks. In addition, we regularize this loss with our proposed "focal regularization parameter" to avoid negative knowledge transfer. Experimental results show that our framework improves the test time recognition accuracy of unimodal networks, and provides the state-of-the-art performance on various dynamic hand gesture recognition datasets. 

---
# Deep Multimodal Subspace Clustering Networks 

**Authors**: Mahdi Abavisani, Vishal M. Patel  

**Link**: [PDF](https://arxiv.org/pdf/1804.06498)  

**Abstract**: We present convolutional neural network (CNN) based approaches for unsupervised multimodal subspace clustering. The proposed framework consists of three main stages - multimodal encoder, self-expressive layer, and multimodal decoder. The encoder takes multimodal data as input and fuses them to a latent space representation. The self-expressive layer is responsible for enforcing the self-expressiveness property and acquiring an affinity matrix corresponding to the data points. The decoder reconstructs the original input data. The network uses the distance between the decoder's reconstruction and the original input in its training. We investigate early, late and intermediate fusion techniques and propose three different encoders corresponding to them for spatial fusion. The self-expressive layers and multimodal decoders are essentially the same for different spatial fusion-based approaches. In addition to various spatial fusion-based methods, an affinity fusion-based network is also proposed in which the self-expressive layer corresponding to different modalities is enforced to be the same. Extensive experiments on three datasets show that the proposed methods significantly outperform the state-of-the-art multimodal subspace clustering methods. 

---
