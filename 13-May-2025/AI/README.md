# Agent RL Scaling Law: Agent RL with Spontaneous Code Execution for Mathematical Problem Solving 

**Authors**: Xinji Mai, Haotian Xu, Xing W, Weinong Wang, Yingying Zhang, Wenqiang Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2505.07773)  

**Abstract**: Large Language Models (LLMs) often struggle with mathematical reasoning tasks requiring precise, verifiable computation. While Reinforcement Learning (RL) from outcome-based rewards enhances text-based reasoning, understanding how agents autonomously learn to leverage external tools like code execution remains crucial. We investigate RL from outcome-based rewards for Tool-Integrated Reasoning, ZeroTIR, training base LLMs to spontaneously generate and execute Python code for mathematical problems without supervised tool-use examples. Our central contribution is we demonstrate that as RL training progresses, key metrics scale predictably. Specifically, we observe strong positive correlations where increased training steps lead to increases in the spontaneous code execution frequency, the average response length, and, critically, the final task accuracy. This suggests a quantifiable relationship between computational effort invested in training and the emergence of effective, tool-augmented reasoning strategies. We implement a robust framework featuring a decoupled code execution environment and validate our findings across standard RL algorithms and frameworks. Experiments show ZeroTIR significantly surpasses non-tool ZeroRL baselines on challenging math benchmarks. Our findings provide a foundational understanding of how autonomous tool use is acquired and scales within Agent RL, offering a reproducible benchmark for future studies. Code is released at \href{this https URL}{this https URL}. 

---
# "I Apologize For Not Understanding Your Policy": Exploring the Specification and Evaluation of User-Managed Access Control Policies by AI Virtual Assistants 

**Authors**: Jennifer Mondragon, Carlos Rubio-Medrano, Gael Cruz, Dvijesh Shastri  

**Link**: [PDF](https://arxiv.org/pdf/2505.07759)  

**Abstract**: The rapid evolution of Artificial Intelligence (AI)-based Virtual Assistants (VAs) e.g., Google Gemini, ChatGPT, Microsoft Copilot, and High-Flyer Deepseek has turned them into convenient interfaces for managing emerging technologies such as Smart Homes, Smart Cars, Electronic Health Records, by means of explicit commands,e.g., prompts, which can be even launched via voice, thus providing a very convenient interface for end-users. However, the proper specification and evaluation of User-Managed Access Control Policies (U-MAPs), the rules issued and managed by end-users to govern access to sensitive data and device functionality - within these VAs presents significant challenges, since such a process is crucial for preventing security vulnerabilities and privacy leaks without impacting user experience. This study provides an initial exploratory investigation on whether current publicly-available VAs can manage U-MAPs effectively across differing scenarios. By conducting unstructured to structured tests, we evaluated the comprehension of such VAs, revealing a lack of understanding in varying U-MAP approaches. Our research not only identifies key limitations, but offers valuable insights into how VAs can be further improved to manage complex authorization rules and adapt to dynamic changes. 

---
# Emotion-Gradient Metacognitive RSI (Part I): Theoretical Foundations and Single-Agent Architecture 

**Authors**: Rintaro Ando  

**Link**: [PDF](https://arxiv.org/pdf/2505.07757)  

**Abstract**: We present the Emotion-Gradient Metacognitive Recursive Self-Improvement (EG-MRSI) framework, a novel architecture that integrates introspective metacognition, emotion-based intrinsic motivation, and recursive self-modification into a unified theoretical system. The framework is explicitly capable of overwriting its own learning algorithm under formally bounded risk. Building upon the Noise-to-Meaning RSI (N2M-RSI) foundation, EG-MRSI introduces a differentiable intrinsic reward function driven by confidence, error, novelty, and cumulative success. This signal regulates both a metacognitive mapping and a self-modification operator constrained by provable safety mechanisms. We formally define the initial agent configuration, emotion-gradient dynamics, and RSI trigger conditions, and derive a reinforcement-compatible optimization objective that guides the agent's development trajectory. Meaning Density and Meaning Conversion Efficiency are introduced as quantifiable metrics of semantic learning, closing the gap between internal structure and predictive informativeness. This Part I paper establishes the single-agent theoretical foundations of EG-MRSI. Future parts will extend this framework to include safety certificates and rollback protocols (Part II), collective intelligence mechanisms (Part III), and feasibility constraints including thermodynamic and computational limits (Part IV). Together, the EG-MRSI series provides a rigorous, extensible foundation for open-ended and safe AGI. 

---
# Belief Injection for Epistemic Control in Linguistic State Space 

**Authors**: Sebastian Dumbrava  

**Link**: [PDF](https://arxiv.org/pdf/2505.07693)  

**Abstract**: This work introduces belief injection, a proactive epistemic control mechanism for artificial agents whose cognitive states are structured as dynamic ensembles of linguistic belief fragments. Grounded in the Semantic Manifold framework, belief injection directly incorporates targeted linguistic beliefs into an agent's internal cognitive state, influencing reasoning and alignment proactively rather than reactively. We delineate various injection strategies, such as direct, context-aware, goal-oriented, and reflective approaches, and contrast belief injection with related epistemic control mechanisms, notably belief filtering. Additionally, this work discusses practical applications, implementation considerations, ethical implications, and outlines promising directions for future research into cognitive governance using architecturally embedded belief injection. 

---
# S-GRPO: Early Exit via Reinforcement Learning in Reasoning Models 

**Authors**: Muzhi Dai, Chenxu Yang, Qingyi Si  

**Link**: [PDF](https://arxiv.org/pdf/2505.07686)  

**Abstract**: As Test-Time Scaling emerges as an active research focus in the large language model community, advanced post-training methods increasingly emphasize extending chain-of-thought (CoT) generation length, thereby enhancing reasoning capabilities to approach Deepseek R1-like reasoning models. However, recent studies reveal that reasoning models (even Qwen3) consistently exhibit excessive thought redundancy in CoT generation. This overthinking problem stems from conventional outcome-reward reinforcement learning's systematic neglect in regulating intermediate reasoning steps. This paper proposes Serial-Group Decaying-Reward Policy Optimization (namely S-GRPO), a novel reinforcement learning method that empowers models with the capability to determine the sufficiency of reasoning steps, subsequently triggering early exit of CoT generation. Specifically, unlike GRPO, which samples multiple possible completions (parallel group) in parallel, we select multiple temporal positions in the generation of one CoT to allow the model to exit thinking and instead generate answers (serial group), respectively. For the correct answers in a serial group, we assign rewards that decay according to positions, with lower rewards towards the later ones, thereby reinforcing the model's behavior to generate higher-quality answers at earlier phases with earlier exits of thinking. Empirical evaluations demonstrate compatibility with state-of-the-art reasoning models, including Qwen3 and Deepseek-distill models, achieving 35.4% ~ 61.1\% sequence length reduction with 0.72% ~ 6.08% accuracy improvements across GSM8K, AIME 2024, AMC 2023, MATH-500, and GPQA Diamond benchmarks. 

---
# YuLan-OneSim: Towards the Next Generation of Social Simulator with Large Language Models 

**Authors**: Lei Wang, Heyang Gao, Xiaohe Bo, Xu Chen, Ji-Rong Wen  

**Link**: [PDF](https://arxiv.org/pdf/2505.07581)  

**Abstract**: Leveraging large language model (LLM) based agents to simulate human social behaviors has recently gained significant attention. In this paper, we introduce a novel social simulator called YuLan-OneSim. Compared to previous works, YuLan-OneSim distinguishes itself in five key aspects: (1) Code-free scenario construction: Users can simply describe and refine their simulation scenarios through natural language interactions with our simulator. All simulation code is automatically generated, significantly reducing the need for programming expertise. (2) Comprehensive default scenarios: We implement 50 default simulation scenarios spanning 8 domains, including economics, sociology, politics, psychology, organization, demographics, law, and communication, broadening access for a diverse range of social researchers. (3) Evolvable simulation: Our simulator is capable of receiving external feedback and automatically fine-tuning the backbone LLMs, significantly enhancing the simulation quality. (4) Large-scale simulation: By developing a fully responsive agent framework and a distributed simulation architecture, our simulator can handle up to 100,000 agents, ensuring more stable and reliable simulation results. (5) AI social researcher: Leveraging the above features, we develop an AI social researcher. Users only need to propose a research topic, and the AI researcher will automatically analyze the input, construct simulation environments, summarize results, generate technical reports, review and refine the reports--completing the social science research loop. To demonstrate the advantages of YuLan-OneSim, we conduct experiments to evaluate the quality of the automatically generated scenarios, the reliability, efficiency, and scalability of the simulation process, as well as the performance of the AI social researcher. 

---
# QuantX: A Framework for Hardware-Aware Quantization of Generative AI Workloads 

**Authors**: Khurram Mazher, Saad Bin Nasir  

**Link**: [PDF](https://arxiv.org/pdf/2505.07531)  

**Abstract**: We present QuantX: a tailored suite of recipes for LLM and VLM quantization. It is capable of quantizing down to 3-bit resolutions with minimal loss in performance. The quantization strategies in QuantX take into account hardware-specific constraints to achieve efficient dequantization during inference ensuring flexible trade-off between runtime speed, memory requirement and model accuracy. Our results demonstrate that QuantX achieves performance within 6% of the unquantized model for LlaVa-v1.6 quantized down to 3-bits for multiple end user tasks and outperforms recently published state-of-the-art quantization techniques. This manuscript provides insights into the LLM quantization process that motivated the range of recipes and options that are incorporated in QuantX. 

---
# HALO: Half Life-Based Outdated Fact Filtering in Temporal Knowledge Graphs 

**Authors**: Feng Ding, Tingting Wang, Yupeng Gao, Shuo Yu, Jing Ren, Feng Xia  

**Link**: [PDF](https://arxiv.org/pdf/2505.07509)  

**Abstract**: Outdated facts in temporal knowledge graphs (TKGs) result from exceeding the expiration date of facts, which negatively impact reasoning performance on TKGs. However, existing reasoning methods primarily focus on positive importance of historical facts, neglecting adverse effects of outdated facts. Besides, training on these outdated facts yields extra computational cost. To address these challenges, we propose an outdated fact filtering framework named HALO, which quantifies the temporal validity of historical facts by exploring the half-life theory to filter outdated facts in TKGs. HALO consists of three modules: the temporal fact attention module, the dynamic relation-aware encoder module, and the outdated fact filtering module. Firstly, the temporal fact attention module captures the evolution of historical facts over time to identify relevant facts. Secondly, the dynamic relation-aware encoder module is designed for efficiently predicting the half life of each fact. Finally, we construct a time decay function based on the half-life theory to quantify the temporal validity of facts and filter outdated facts. Experimental results show that HALO outperforms the state-of-the-art TKG reasoning methods on three public datasets, demonstrating its effectiveness in detecting and filtering outdated facts (Codes are available at this https URL ). 

---
# Web-Bench: A LLM Code Benchmark Based on Web Standards and Frameworks 

**Authors**: Kai Xu, YiWei Mao, XinYi Guan, ZiLong Feng  

**Link**: [PDF](https://arxiv.org/pdf/2505.07473)  

**Abstract**: The application of large language models (LLMs) in the field of coding is evolving rapidly: from code assistants, to autonomous coding agents, and then to generating complete projects through natural language. Early LLM code benchmarks primarily focused on code generation accuracy, but these benchmarks have gradually become saturated. Benchmark saturation weakens their guiding role for LLMs. For example, HumanEval Pass@1 has reached 99.4% and MBPP 94.2%. Among various attempts to address benchmark saturation, approaches based on software engineering have stood out, but the saturation of existing software engineering benchmarks is rapidly increasing. To address this, we propose a new benchmark, Web-Bench, which contains 50 projects, each consisting of 20 tasks with sequential dependencies. The tasks implement project features in sequence, simulating real-world human development workflows. When designing Web-Bench, we aim to cover the foundational elements of Web development: Web Standards and Web Frameworks. Given the scale and complexity of these projects, which were designed by engineers with 5 to 10 years of experience, each presents a significant challenge. On average, a single project takes 4 to 8 hours for a senior engineer to complete. On our given benchmark agent (Web-Agent), SOTA (Claude 3.7 Sonnet) achieves only 25.1% Pass@1, significantly lower (better) than SWE-Bench's Verified (65.4%) and Full (33.8%) scores. Finally, we discuss that in any development field, Standards and Frameworks represent foundational knowledge and efficiency tools, respectively, and LLMs require optimization tailored to them. 

---
# A Survey on Collaborative Mechanisms Between Large and Small Language Models 

**Authors**: Yi Chen, JiaHao Zhao, HaoHao Han  

**Link**: [PDF](https://arxiv.org/pdf/2505.07460)  

**Abstract**: Large Language Models (LLMs) deliver powerful AI capabilities but face deployment challenges due to high resource costs and latency, whereas Small Language Models (SLMs) offer efficiency and deployability at the cost of reduced performance. Collaboration between LLMs and SLMs emerges as a crucial paradigm to synergistically balance these trade-offs, enabling advanced AI applications, especially on resource-constrained edge devices. This survey provides a comprehensive overview of LLM-SLM collaboration, detailing various interaction mechanisms (pipeline, routing, auxiliary, distillation, fusion), key enabling technologies, and diverse application scenarios driven by on-device needs like low latency, privacy, personalization, and offline operation. While highlighting the significant potential for creating more efficient, adaptable, and accessible AI, we also discuss persistent challenges including system overhead, inter-model consistency, robust task allocation, evaluation complexity, and security/privacy concerns. Future directions point towards more intelligent adaptive frameworks, deeper model fusion, and expansion into multimodal and embodied AI, positioning LLM-SLM collaboration as a key driver for the next generation of practical and ubiquitous artificial intelligence. 

---
# How well do LLMs reason over tabular data, really? 

**Authors**: Cornelius Wolff, Madelon Hulsebos  

**Link**: [PDF](https://arxiv.org/pdf/2505.07453)  

**Abstract**: Large Language Models (LLMs) excel in natural language tasks, but less is known about their reasoning capabilities over tabular data. Prior analyses devise evaluation strategies that poorly reflect an LLM's realistic performance on tabular queries. Moreover, we have a limited understanding of the robustness of LLMs towards realistic variations in tabular inputs. Therefore, we ask: Can general-purpose LLMs reason over tabular data, really?, and focus on two questions 1) are tabular reasoning capabilities of general-purpose LLMs robust to real-world characteristics of tabular inputs, and 2) how can we realistically evaluate an LLM's performance on analytical tabular queries? Building on a recent tabular reasoning benchmark, we first surface shortcomings of its multiple-choice prompt evaluation strategy, as well as commonly used free-form text metrics such as SacreBleu and BERT-score. We show that an LLM-as-a-judge procedure yields more reliable performance insights and unveil a significant deficit in tabular reasoning performance of LLMs. We then extend the tabular inputs reflecting three common characteristics in practice: 1) missing values, 2) duplicate entities, and 3) structural variations. Experiments show that the tabular reasoning capabilities of general-purpose LLMs suffer from these variations, stressing the importance of improving their robustness for realistic tabular inputs. 

---
# AIS Data-Driven Maritime Monitoring Based on Transformer: A Comprehensive Review 

**Authors**: Zhiye Xie, Enmei Tu, Xianping Fu, Guoliang Yuan, Yi Han  

**Link**: [PDF](https://arxiv.org/pdf/2505.07374)  

**Abstract**: With the increasing demands for safety, efficiency, and sustainability in global shipping, Automatic Identification System (AIS) data plays an increasingly important role in maritime monitoring. AIS data contains spatial-temporal variation patterns of vessels that hold significant research value in the marine domain. However, due to its massive scale, the full potential of AIS data has long remained untapped. With its powerful sequence modeling capabilities, particularly its ability to capture long-range dependencies and complex temporal dynamics, the Transformer model has emerged as an effective tool for processing AIS data. Therefore, this paper reviews the research on Transformer-based AIS data-driven maritime monitoring, providing a comprehensive overview of the current applications of Transformer models in the marine field. The focus is on Transformer-based trajectory prediction methods, behavior detection, and prediction techniques. Additionally, this paper collects and organizes publicly available AIS datasets from the reviewed papers, performing data filtering, cleaning, and statistical analysis. The statistical results reveal the operational characteristics of different vessel types, providing data support for further research on maritime monitoring tasks. Finally, we offer valuable suggestions for future research, identifying two promising research directions. Datasets are available at this https URL. 

---
# FedIFL: A federated cross-domain diagnostic framework for motor-driven systems with inconsistent fault modes 

**Authors**: Zexiao Wang, Yankai Wang, Xiaoqiang Liao, Xinguo Ming, Weiming Shen  

**Link**: [PDF](https://arxiv.org/pdf/2505.07315)  

**Abstract**: Due to the scarcity of industrial data, individual equipment users, particularly start-ups, struggle to independently train a comprehensive fault diagnosis model; federated learning enables collaborative training while ensuring data privacy, making it an ideal solution. However, the diversity of working conditions leads to variations in fault modes, resulting in inconsistent label spaces across different clients. In federated diagnostic scenarios, label space inconsistency leads to local models focus on client-specific fault modes and causes local models from different clients to map different failure modes to similar feature representations, which weakens the aggregated global model's generalization. To tackle this issue, this article proposed a federated cross-domain diagnostic framework termed Federated Invariant Features Learning (FedIFL). In intra-client training, prototype contrastive learning mitigates intra-client domain shifts, subsequently, feature generating ensures local models can access distributions of other clients in a privacy-friendly manner. Besides, in cross-client training, a feature disentanglement mechanism is introduced to mitigate cross-client domain shifts, specifically, an instance-level federated instance consistency loss is designed to ensure the instance-level consistency of invariant features between different clients, furthermore, a federated instance personalization loss and an orthogonal loss are constructed to distinguish specific features that from the invariant features. Eventually, the aggregated model achieves promising generalization among global label spaces, enabling accurate fault diagnosis for target clients' Motor Driven Systems (MDSs) with inconsistent label spaces. Experiments on real-world MDSs validate the effectiveness and superiority of FedIFL in federated cross-domain diagnosis with inconsistent fault modes. 

---
# Interpretable Event Diagnosis in Water Distribution Networks 

**Authors**: André Artelt, Stelios G. Vrachimis, Demetrios G. Eliades, Ulrike Kuhl, Barbara Hammer, Marios M. Polycarpou  

**Link**: [PDF](https://arxiv.org/pdf/2505.07299)  

**Abstract**: The increasing penetration of information and communication technologies in the design, monitoring, and control of water systems enables the use of algorithms for detecting and identifying unanticipated events (such as leakages or water contamination) using sensor measurements. However, data-driven methodologies do not always give accurate results and are often not trusted by operators, who may prefer to use their engineering judgment and experience to deal with such events.
In this work, we propose a framework for interpretable event diagnosis -- an approach that assists the operators in associating the results of algorithmic event diagnosis methodologies with their own intuition and experience. This is achieved by providing contrasting (i.e., counterfactual) explanations of the results provided by fault diagnosis algorithms; their aim is to improve the understanding of the algorithm's inner workings by the operators, thus enabling them to take a more informed decision by combining the results with their personal experiences. Specifically, we propose counterfactual event fingerprints, a representation of the difference between the current event diagnosis and the closest alternative explanation, which can be presented in a graphical way. The proposed methodology is applied and evaluated on a realistic use case using the L-Town benchmark. 

---
# Measuring General Intelligence with Generated Games 

**Authors**: Vivek Verma, David Huang, William Chen, Dan Klein, Nicholas Tomlin  

**Link**: [PDF](https://arxiv.org/pdf/2505.07215)  

**Abstract**: We present gg-bench, a collection of game environments designed to evaluate general reasoning capabilities in language models. Unlike most static benchmarks, gg-bench is a data generating process where new evaluation instances can be generated at will. In particular, gg-bench is synthetically generated by (1) using a large language model (LLM) to generate natural language descriptions of novel games, (2) using the LLM to implement each game in code as a Gym environment, and (3) training reinforcement learning (RL) agents via self-play on the generated games. We evaluate language models by their winrate against these RL agents by prompting models with the game description, current board state, and a list of valid moves, after which models output the moves they wish to take. gg-bench is challenging: state-of-the-art LLMs such as GPT-4o and Claude 3.7 Sonnet achieve winrates of 7-9% on gg-bench using in-context learning, while reasoning models such as o1, o3-mini and DeepSeek-R1 achieve average winrates of 31-36%. We release the generated games, data generation process, and evaluation code in order to support future modeling work and expansion of our benchmark. 

---
# Accountability of Generative AI: Exploring a Precautionary Approach for "Artificially Created Nature" 

**Authors**: Yuri Nakao  

**Link**: [PDF](https://arxiv.org/pdf/2505.07178)  

**Abstract**: The rapid development of generative artificial intelligence (AI) technologies raises concerns about the accountability of sociotechnical systems. Current generative AI systems rely on complex mechanisms that make it difficult for even experts to fully trace the reasons behind the outputs. This paper first examines existing research on AI transparency and accountability and argues that transparency is not a sufficient condition for accountability but can contribute to its improvement. We then discuss that if it is not possible to make generative AI transparent, generative AI technology becomes ``artificially created nature'' in a metaphorical sense, and suggest using the precautionary principle approach to consider AI risks. Finally, we propose that a platform for citizen participation is needed to address the risks of generative AI. 

---
# ReCDAP: Relation-Based Conditional Diffusion with Attention Pooling for Few-Shot Knowledge Graph Completion 

**Authors**: Jeongho Kim, Chanyeong Heo, Jaehee Jung  

**Link**: [PDF](https://arxiv.org/pdf/2505.07171)  

**Abstract**: Knowledge Graphs (KGs), composed of triples in the form of (head, relation, tail) and consisting of entities and relations, play a key role in information retrieval systems such as question answering, entity search, and recommendation. In real-world KGs, although many entities exist, the relations exhibit a long-tail distribution, which can hinder information retrieval performance. Previous few-shot knowledge graph completion studies focused exclusively on the positive triple information that exists in the graph or, when negative triples were incorporated, used them merely as a signal to indicate incorrect triples. To overcome this limitation, we propose Relation-Based Conditional Diffusion with Attention Pooling (ReCDAP). First, negative triples are generated by randomly replacing the tail entity in the support set. By conditionally incorporating positive information in the KG and non-existent negative information into the diffusion process, the model separately estimates the latent distributions for positive and negative relations. Moreover, including an attention pooler enables the model to leverage the differences between positive and negative cases explicitly. Experiments on two widely used datasets demonstrate that our method outperforms existing approaches, achieving state-of-the-art performance. The code is available at this https URL. 

---
# RefPentester: A Knowledge-Informed Self-Reflective Penetration Testing Framework Based on Large Language Models 

**Authors**: Hanzheng Dai, Yuanliang Li, Zhibo Zhang, Jun Yan  

**Link**: [PDF](https://arxiv.org/pdf/2505.07089)  

**Abstract**: Automated penetration testing (AutoPT) powered by large language models (LLMs) has gained attention for its ability to automate ethical hacking processes and identify vulnerabilities in target systems by leveraging the intrinsic knowledge of LLMs. However, existing LLM-based AutoPT frameworks often underperform compared to human experts in challenging tasks for several reasons: the imbalanced knowledge used in LLM training, short-sighted planning in the planning process, and hallucinations during command generation. In addition, the penetration testing (PT) process, with its trial-and-error nature, is limited by existing frameworks that lack mechanisms to learn from previous failed operations, restricting adaptive improvement of PT strategies. To address these limitations, we propose a knowledge-informed self-reflective PT framework powered by LLMs, called RefPentester, which is an AutoPT framework designed to assist human operators in identifying the current stage of the PT process, selecting appropriate tactic and technique for the stage, choosing suggested action, providing step-by-step operational guidance, and learning from previous failed operations. We also modeled the PT process as a seven-state Stage Machine to integrate the proposed framework effectively. The evaluation shows that RefPentester can successfully reveal credentials on Hack The Box's Sau machine, outperforming the baseline GPT-4o model by 16.7\%. Across PT stages, RefPentester also demonstrates superior success rates on PT stage transitions. 

---
# Architectural Precedents for General Agents using Large Language Models 

**Authors**: Robert E. Wray, James R. Kirk, John E. Laird  

**Link**: [PDF](https://arxiv.org/pdf/2505.07087)  

**Abstract**: One goal of AI (and AGI) is to identify and understand specific mechanisms and representations sufficient for general intelligence. Often, this work manifests in research focused on architectures and many cognitive architectures have been explored in AI/AGI. However, different research groups and even different research traditions have somewhat independently identified similar/common patterns of processes and representations or cognitive design patterns that are manifest in existing architectures. Today, AI systems exploiting large language models (LLMs) offer a relatively new combination of mechanism and representation available for exploring the possibilities of general intelligence. In this paper, we summarize a few recurring cognitive design patterns that have appeared in various pre-transformer AI architectures. We then explore how these patterns are evident in systems using LLMs, especially for reasoning and interactive ("agentic") use cases. By examining and applying these recurring patterns, we can also predict gaps or deficiencies in today's Agentic LLM Systems and identify likely subjects of future research towards general intelligence using LLMs and other generative foundation models. 

---
# Arbitrarily Applicable Same/Opposite Relational Responding with NARS 

**Authors**: Robert Johansson, Patrick Hammer, Tony Lofthouse  

**Link**: [PDF](https://arxiv.org/pdf/2505.07079)  

**Abstract**: Same/opposite relational responding, a fundamental aspect of human symbolic cognition, allows the flexible generalization of stimulus relationships based on minimal experience. In this study, we demonstrate the emergence of \textit{arbitrarily applicable} same/opposite relational responding within the Non-Axiomatic Reasoning System (NARS), a computational cognitive architecture designed for adaptive reasoning under uncertainty. Specifically, we extend NARS with an implementation of \textit{acquired relations}, enabling the system to explicitly derive both symmetric (mutual entailment) and novel relational combinations (combinatorial entailment) from minimal explicit training in a contextually controlled matching-to-sample (MTS) procedure. Experimental results show that NARS rapidly internalizes explicitly trained relational rules and robustly demonstrates derived relational generalizations based on arbitrary contextual cues. Importantly, derived relational responding in critical test phases inherently combines both mutual and combinatorial entailments, such as deriving same-relations from multiple explicitly trained opposite-relations. Internal confidence metrics illustrate strong internalization of these relational principles, closely paralleling phenomena observed in human relational learning experiments. Our findings underscore the potential for integrating nuanced relational learning mechanisms inspired by learning psychology into artificial general intelligence frameworks, explicitly highlighting the arbitrary and context-sensitive relational capabilities modeled within NARS. 

---
# Unlocking Non-Block-Structured Decisions: Inductive Mining with Choice Graphs 

**Authors**: Humam Kourani, Gyunam Park, Wil M.P. van der Aalst  

**Link**: [PDF](https://arxiv.org/pdf/2505.07052)  

**Abstract**: Process discovery aims to automatically derive process models from event logs, enabling organizations to analyze and improve their operational processes. Inductive mining algorithms, while prioritizing soundness and efficiency through hierarchical modeling languages, often impose a strict block-structured representation. This limits their ability to accurately capture the complexities of real-world processes. While recent advancements like the Partially Ordered Workflow Language (POWL) have addressed the block-structure limitation for concurrency, a significant gap remains in effectively modeling non-block-structured decision points. In this paper, we bridge this gap by proposing an extension of POWL to handle non-block-structured decisions through the introduction of choice graphs. Choice graphs offer a structured yet flexible approach to model complex decision logic within the hierarchical framework of POWL. We present an inductive mining discovery algorithm that uses our extension and preserves the quality guarantees of the inductive mining framework. Our experimental evaluation demonstrates that the discovered models, enriched with choice graphs, more precisely represent the complex decision-making behavior found in real-world processes, without compromising the high scalability inherent in inductive mining techniques. 

---
# DialogueReason: Rule-Based RL Sparks Dialogue Reasoning in LLMs 

**Authors**: Yubo Shu, Zhewei Huang, Xin Wu, Chen Hu, Shuchang Zhou, Daxin Jiang  

**Link**: [PDF](https://arxiv.org/pdf/2505.07049)  

**Abstract**: We propose DialogueReason, a reasoning paradigm that uncovers the lost roles in monologue-style reasoning models, aiming to boost diversity and coherency of the reasoning process. Recent advances in RL-based large reasoning models have led to impressive long CoT capabilities and high performance on math and science benchmarks. However, these reasoning models rely mainly on monologue-style reasoning, which often limits reasoning diversity and coherency, frequently recycling fixed strategies or exhibiting unnecessary shifts in attention. Our work consists of an analysis of monologue reasoning patterns and the development of a dialogue-based reasoning approach. We first introduce the Compound-QA task, which concatenates multiple problems into a single prompt to assess both diversity and coherency of reasoning. Our analysis shows that Compound-QA exposes weaknesses in monologue reasoning, evidenced by both quantitative metrics and qualitative reasoning traces. Building on the analysis, we propose a dialogue-based reasoning, named DialogueReason, structured around agents, environment, and interactions. Using PPO with rule-based rewards, we train open-source LLMs (Qwen-QWQ and Qwen-Base) to adopt dialogue reasoning. We evaluate trained models on MATH, AIME, and GPQA datasets, showing that the dialogue reasoning model outperforms monologue models under more complex compound questions. Additionally, we discuss how dialogue-based reasoning helps enhance interpretability, facilitate more intuitive human interaction, and inspire advances in multi-agent system design. 

---
# Efficient Fault Detection in WSN Based on PCA-Optimized Deep Neural Network Slicing Trained with GOA 

**Authors**: Mahmood Mohassel Feghhi, Raya Majid Alsharfa, Majid Hameed Majeed  

**Link**: [PDF](https://arxiv.org/pdf/2505.07030)  

**Abstract**: Fault detection in Wireless Sensor Networks (WSNs) is crucial for reliable data transmission and network longevity. Traditional fault detection methods often struggle with optimizing deep neural networks (DNNs) for efficient performance, especially in handling high-dimensional data and capturing nonlinear relationships. Additionally, these methods typically suffer from slow convergence and difficulty in finding optimal network architectures using gradient-based optimization. This study proposes a novel hybrid method combining Principal Component Analysis (PCA) with a DNN optimized by the Grasshopper Optimization Algorithm (GOA) to address these limitations. Our approach begins by computing eigenvalues from the original 12-dimensional dataset and sorting them in descending order. The cumulative sum of these values is calculated, retaining principal components until 99.5% variance is achieved, effectively reducing dimensionality to 4 features while preserving critical information. This compressed representation trains a six-layer DNN where GOA optimizes the network architecture, overcoming backpropagation's limitations in discovering nonlinear relationships. This hybrid PCA-GOA-DNN framework compresses the data and trains a six-layer DNN that is optimized by GOA, enhancing both training efficiency and fault detection accuracy. The dataset used in this study is a real-world WSNs dataset developed by the University of North Carolina, which was used to evaluate the proposed method's performance. Extensive simulations demonstrate that our approach achieves a remarkable 99.72% classification accuracy, with exceptional precision and recall, outperforming conventional methods. The method is computationally efficient, making it suitable for large-scale WSN deployments, and represents a significant advancement in fault detection for resource-constrained WSNs. 

---
# LLM-Augmented Chemical Synthesis and Design Decision Programs 

**Authors**: Haorui Wang, Jeff Guo, Lingkai Kong, Rampi Ramprasad, Philippe Schwaller, Yuanqi Du, Chao Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2505.07027)  

**Abstract**: Retrosynthesis, the process of breaking down a target molecule into simpler precursors through a series of valid reactions, stands at the core of organic chemistry and drug development. Although recent machine learning (ML) research has advanced single-step retrosynthetic modeling and subsequent route searches, these solutions remain restricted by the extensive combinatorial space of possible pathways. Concurrently, large language models (LLMs) have exhibited remarkable chemical knowledge, hinting at their potential to tackle complex decision-making tasks in chemistry. In this work, we explore whether LLMs can successfully navigate the highly constrained, multi-step retrosynthesis planning problem. We introduce an efficient scheme for encoding reaction pathways and present a new route-level search strategy, moving beyond the conventional step-by-step reactant prediction. Through comprehensive evaluations, we show that our LLM-augmented approach excels at retrosynthesis planning and extends naturally to the broader challenge of synthesizable molecular design. 

---
# Explainable AI the Latest Advancements and New Trends 

**Authors**: Bowen Long, Enjie Liu, Renxi Qiu, Yanqing Duan  

**Link**: [PDF](https://arxiv.org/pdf/2505.07005)  

**Abstract**: In recent years, Artificial Intelligence technology has excelled in various applications across all domains and fields. However, the various algorithms in neural networks make it difficult to understand the reasons behind decisions. For this reason, trustworthy AI techniques have started gaining popularity. The concept of trustworthiness is cross-disciplinary; it must meet societal standards and principles, and technology is used to fulfill these requirements. In this paper, we first surveyed developments from various countries and regions on the ethical elements that make AI algorithms trustworthy; and then focused our survey on the state of the art research into the interpretability of AI. We have conducted an intensive survey on technologies and techniques used in making AI explainable. Finally, we identified new trends in achieving explainable AI. In particular, we elaborate on the strong link between the explainability of AI and the meta-reasoning of autonomous systems. The concept of meta-reasoning is 'reason the reasoning', which coincides with the intention and goal of explainable Al. The integration of the approaches could pave the way for future interpretable AI systems. 

---
# A Multi-Agent Reinforcement Learning Approach for Cooperative Air-Ground-Human Crowdsensing in Emergency Rescue 

**Authors**: Wenhao Lu, Zhengqiu Zhu, Yong Zhao, Yonglin Tian, Junjie Zeng, Jun Zhang, Zhong Liu, Fei-Yue Wang  

**Link**: [PDF](https://arxiv.org/pdf/2505.06997)  

**Abstract**: Mobile crowdsensing is evolving beyond traditional human-centric models by integrating heterogeneous entities like unmanned aerial vehicles (UAVs) and unmanned ground vehicles (UGVs). Optimizing task allocation among these diverse agents is critical, particularly in challenging emergency rescue scenarios characterized by complex environments, limited communication, and partial observability. This paper tackles the Heterogeneous-Entity Collaborative-Sensing Task Allocation (HECTA) problem specifically for emergency rescue, considering humans, UAVs, and UGVs. We introduce a novel ``Hard-Cooperative'' policy where UGVs prioritize recharging low-battery UAVs, alongside performing their sensing tasks. The primary objective is maximizing the task completion rate (TCR) under strict time constraints. We rigorously formulate this NP-hard problem as a decentralized partially observable Markov decision process (Dec-POMDP) to effectively handle sequential decision-making under uncertainty. To solve this, we propose HECTA4ER, a novel multi-agent reinforcement learning algorithm built upon a Centralized Training with Decentralized Execution architecture. HECTA4ER incorporates tailored designs, including specialized modules for complex feature extraction, utilization of action-observation history via hidden states, and a mixing network integrating global and local information, specifically addressing the challenges of partial observability. Furthermore, theoretical analysis confirms the algorithm's convergence properties. Extensive simulations demonstrate that HECTA4ER significantly outperforms baseline algorithms, achieving an average 18.42% increase in TCR. Crucially, a real-world case study validates the algorithm's effectiveness and robustness in dynamic sensing scenarios, highlighting its strong potential for practical application in emergency response. 

---
# CAT Merging: A Training-Free Approach for Resolving Conflicts in Model Merging 

**Authors**: Wenju Sun, Qingyong Li, Yangli-ao Geng, Boyang Li  

**Link**: [PDF](https://arxiv.org/pdf/2505.06977)  

**Abstract**: Multi-task model merging offers a promising paradigm for integrating multiple expert models into a unified model without additional training. Existing state-of-the-art techniques, such as Task Arithmetic and its variants, merge models by accumulating task vectors -- the parameter differences between pretrained and finetuned models. However, task vector accumulation is often hindered by knowledge conflicts, leading to performance degradation. To address this challenge, we propose Conflict-Aware Task Merging (CAT Merging), a novel training-free framework that selectively trims conflict-prone components from the task vectors. CAT Merging introduces several parameter-specific strategies, including projection for linear weights and masking for scaling and shifting parameters in normalization layers. Extensive experiments on vision, language, and vision-language tasks demonstrate that CAT Merging effectively suppresses knowledge conflicts, achieving average accuracy improvements of up to 2.5% (ViT-B/32) and 2.0% (ViT-L/14) over state-of-the-art methods. 

---
# From Knowledge to Reasoning: Evaluating LLMs for Ionic Liquids Research in Chemical and Biological Engineering 

**Authors**: Gaurab Sarkar, Sougata Saha  

**Link**: [PDF](https://arxiv.org/pdf/2505.06964)  

**Abstract**: Although Large Language Models (LLMs) have achieved remarkable performance in diverse general knowledge and reasoning tasks, their utility in the scientific domain of Chemical and Biological Engineering (CBE) is unclear. Hence, it necessitates challenging evaluation benchmarks that can measure LLM performance in knowledge- and reasoning-based tasks, which is lacking. As a foundational step, we empirically measure the reasoning capabilities of LLMs in CBE. We construct and share an expert-curated dataset of 5,920 examples for benchmarking LLMs' reasoning capabilities in the niche domain of Ionic Liquids (ILs) for carbon sequestration, an emergent solution to reducing global warming. The dataset presents different difficulty levels by varying along the dimensions of linguistic and domain-specific knowledge. Benchmarking three less than 10B parameter open-source LLMs on the dataset suggests that while smaller general-purpose LLMs are knowledgeable about ILs, they lack domain-specific reasoning capabilities. Based on our results, we further discuss considerations for leveraging LLMs for carbon capture research using ILs. Since LLMs have a high carbon footprint, gearing them for IL research can symbiotically benefit both fields and help reach the ambitious carbon neutrality target by 2050. Dataset link: this https URL 

---
# Causal knowledge graph analysis identifies adverse drug effects 

**Authors**: Sumyyah Toonsi, Paul Schofield, Robert Hoehndorf  

**Link**: [PDF](https://arxiv.org/pdf/2505.06949)  

**Abstract**: Knowledge graphs and structural causal models have each proven valuable for organizing biomedical knowledge and estimating causal effects, but remain largely disconnected: knowledge graphs encode qualitative relationships focusing on facts and deductive reasoning without formal probabilistic semantics, while causal models lack integration with background knowledge in knowledge graphs and have no access to the deductive reasoning capabilities that knowledge graphs provide. To bridge this gap, we introduce a novel formulation of Causal Knowledge Graphs (CKGs) which extend knowledge graphs with formal causal semantics, preserving their deductive capabilities while enabling principled causal inference. CKGs support deconfounding via explicitly marked causal edges and facilitate hypothesis formulation aligned with both encoded and entailed background knowledge. We constructed a Drug-Disease CKG (DD-CKG) integrating disease progression pathways, drug indications, side-effects, and hierarchical disease classification to enable automated large-scale mediation analysis. Applied to UK Biobank and MIMIC-IV cohorts, we tested whether drugs mediate effects between indications and downstream disease progression, adjusting for confounders inferred from the DD-CKG. Our approach successfully reproduced known adverse drug reactions with high precision while identifying previously undocumented significant candidate adverse effects. Further validation through side effect similarity analysis demonstrated that combining our predicted drug effects with established databases significantly improves the prediction of shared drug indications, supporting the clinical relevance of our novel findings. These results demonstrate that our methodology provides a generalizable, knowledge-driven framework for scalable causal inference. 

---
# Towards Artificial General or Personalized Intelligence? A Survey on Foundation Models for Personalized Federated Intelligence 

**Authors**: Yu Qiao, Huy Q. Le, Avi Deb Raha, Phuong-Nam Tran, Apurba Adhikary, Mengchun Zhang, Loc X. Nguyen, Eui-Nam Huh, Dusit Niyato, Choong Seon Hong  

**Link**: [PDF](https://arxiv.org/pdf/2505.06907)  

**Abstract**: The rise of large language models (LLMs), such as ChatGPT, DeepSeek, and Grok-3, has reshaped the artificial intelligence landscape. As prominent examples of foundational models (FMs) built on LLMs, these models exhibit remarkable capabilities in generating human-like content, bringing us closer to achieving artificial general intelligence (AGI). However, their large-scale nature, sensitivity to privacy concerns, and substantial computational demands present significant challenges to personalized customization for end users. To bridge this gap, this paper presents the vision of artificial personalized intelligence (API), focusing on adapting these powerful models to meet the specific needs and preferences of users while maintaining privacy and efficiency. Specifically, this paper proposes personalized federated intelligence (PFI), which integrates the privacy-preserving advantages of federated learning (FL) with the zero-shot generalization capabilities of FMs, enabling personalized, efficient, and privacy-protective deployment at the edge. We first review recent advances in both FL and FMs, and discuss the potential of leveraging FMs to enhance federated systems. We then present the key motivations behind realizing PFI and explore promising opportunities in this space, including efficient PFI, trustworthy PFI, and PFI empowered by retrieval-augmented generation (RAG). Finally, we outline key challenges and future research directions for deploying FM-powered FL systems at the edge with improved personalization, computational efficiency, and privacy guarantees. Overall, this survey aims to lay the groundwork for the development of API as a complement to AGI, with a particular focus on PFI as a key enabling technique. 

---
# Embodied Intelligence: The Key to Unblocking Generalized Artificial Intelligence 

**Authors**: Jinhao Jiang, Changlin Chen, Shile Feng, Wanru Geng, Zesheng Zhou, Ni Wang, Shuai Li, Feng-Qi Cui, Erbao Dong  

**Link**: [PDF](https://arxiv.org/pdf/2505.06897)  

**Abstract**: The ultimate goal of artificial intelligence (AI) is to achieve Artificial General Intelligence (AGI). Embodied Artificial Intelligence (EAI), which involves intelligent systems with physical presence and real-time interaction with the environment, has emerged as a key research direction in pursuit of AGI. While advancements in deep learning, reinforcement learning, large-scale language models, and multimodal technologies have significantly contributed to the progress of EAI, most existing reviews focus on specific technologies or applications. A systematic overview, particularly one that explores the direct connection between EAI and AGI, remains scarce. This paper examines EAI as a foundational approach to AGI, systematically analyzing its four core modules: perception, intelligent decision-making, action, and feedback. We provide a detailed discussion of how each module contributes to the six core principles of AGI. Additionally, we discuss future trends, challenges, and research directions in EAI, emphasizing its potential as a cornerstone for AGI development. Our findings suggest that EAI's integration of dynamic learning and real-world interaction is essential for bridging the gap between narrow AI and AGI. 

---
# Beyond Patterns: Harnessing Causal Logic for Autonomous Driving Trajectory Prediction 

**Authors**: Bonan Wang, Haicheng Liao, Chengyue Wang, Bin Rao, Yanchen Guan, Guyang Yu, Jiaxun Zhang, Songning Lai, Chengzhong Xu, Zhenning Li  

**Link**: [PDF](https://arxiv.org/pdf/2505.06856)  

**Abstract**: Accurate trajectory prediction has long been a major challenge for autonomous driving (AD). Traditional data-driven models predominantly rely on statistical correlations, often overlooking the causal relationships that govern traffic behavior. In this paper, we introduce a novel trajectory prediction framework that leverages causal inference to enhance predictive robustness, generalization, and accuracy. By decomposing the environment into spatial and temporal components, our approach identifies and mitigates spurious correlations, uncovering genuine causal relationships. We also employ a progressive fusion strategy to integrate multimodal information, simulating human-like reasoning processes and enabling real-time inference. Evaluations on five real-world datasets--ApolloScape, nuScenes, NGSIM, HighD, and MoCAD--demonstrate our model's superiority over existing state-of-the-art (SOTA) methods, with improvements in key metrics such as RMSE and FDE. Our findings highlight the potential of causal reasoning to transform trajectory prediction, paving the way for robust AD systems. 

---
# Control Plane as a Tool: A Scalable Design Pattern for Agentic AI Systems 

**Authors**: Sivasathivel Kandasamy  

**Link**: [PDF](https://arxiv.org/pdf/2505.06817)  

**Abstract**: Agentic AI systems represent a new frontier in artificial intelligence, where agents often based on large language models(LLMs) interact with tools, environments, and other agents to accomplish tasks with a degree of autonomy. These systems show promise across a range of domains, but their architectural underpinnings remain immature. This paper conducts a comprehensive review of the types of agents, their modes of interaction with the environment, and the infrastructural and architectural challenges that emerge. We identify a gap in how these systems manage tool orchestration at scale and propose a reusable design abstraction: the "Control Plane as a Tool" pattern. This pattern allows developers to expose a single tool interface to an agent while encapsulating modular tool routing logic behind it. We position this pattern within the broader context of agent design and argue that it addresses several key challenges in scaling, safety, and extensibility. 

---
# Value Iteration with Guessing for Markov Chains and Markov Decision Processes 

**Authors**: Krishnendu Chatterjee, Mahdi JafariRaviz, Raimundo Saona, Jakub Svoboda  

**Link**: [PDF](https://arxiv.org/pdf/2505.06769)  

**Abstract**: Two standard models for probabilistic systems are Markov chains (MCs) and Markov decision processes (MDPs). Classic objectives for such probabilistic models for control and planning problems are reachability and stochastic shortest path. The widely studied algorithmic approach for these problems is the Value Iteration (VI) algorithm which iteratively applies local updates called Bellman updates. There are many practical approaches for VI in the literature but they all require exponentially many Bellman updates for MCs in the worst case. A preprocessing step is an algorithm that is discrete, graph-theoretical, and requires linear space. An important open question is whether, after a polynomial-time preprocessing, VI can be achieved with sub-exponentially many Bellman updates. In this work, we present a new approach for VI based on guessing values. Our theoretical contributions are twofold. First, for MCs, we present an almost-linear-time preprocessing algorithm after which, along with guessing values, VI requires only subexponentially many Bellman updates. Second, we present an improved analysis of the speed of convergence of VI for MDPs. Finally, we present a practical algorithm for MDPs based on our new approach. Experimental results show that our approach provides a considerable improvement over existing VI-based approaches on several benchmark examples from the literature. 

---
# Bi-level Mean Field: Dynamic Grouping for Large-Scale MARL 

**Authors**: Yuxuan Zheng, Yihe Zhou, Feiyang Xu, Mingli Song, Shunyu Liu  

**Link**: [PDF](https://arxiv.org/pdf/2505.06706)  

**Abstract**: Large-scale Multi-Agent Reinforcement Learning (MARL) often suffers from the curse of dimensionality, as the exponential growth in agent interactions significantly increases computational complexity and impedes learning efficiency. To mitigate this, existing efforts that rely on Mean Field (MF) simplify the interaction landscape by approximating neighboring agents as a single mean agent, thus reducing overall complexity to pairwise interactions. However, these MF methods inevitably fail to account for individual differences, leading to aggregation noise caused by inaccurate iterative updates during MF learning. In this paper, we propose a Bi-level Mean Field (BMF) method to capture agent diversity with dynamic grouping in large-scale MARL, which can alleviate aggregation noise via bi-level interaction. Specifically, BMF introduces a dynamic group assignment module, which employs a Variational AutoEncoder (VAE) to learn the representations of agents, facilitating their dynamic grouping over time. Furthermore, we propose a bi-level interaction module to model both inter- and intra-group interactions for effective neighboring aggregation. Experiments across various tasks demonstrate that the proposed BMF yields results superior to the state-of-the-art methods. Our code will be made publicly available. 

---
# A Survey on Data-Driven Modeling of Human Drivers' Lane-Changing Decisions 

**Authors**: Linxuan Huang, Dong-Fan Xie, Li Li, Zhengbing He  

**Link**: [PDF](https://arxiv.org/pdf/2505.06680)  

**Abstract**: Lane-changing (LC) behavior, a critical yet complex driving maneuver, significantly influences driving safety and traffic dynamics. Traditional analytical LC decision (LCD) models, while effective in specific environments, often oversimplify behavioral heterogeneity and complex interactions, limiting their capacity to capture real LCD. Data-driven approaches address these gaps by leveraging rich empirical data and machine learning to decode latent decision-making patterns, enabling adaptive LCD modeling in dynamic environments. In light of the rapid development of artificial intelligence and the demand for data-driven models oriented towards connected vehicles and autonomous vehicles, this paper presents a comprehensive survey of data-driven LCD models, with a particular focus on human drivers LC decision-making. It systematically reviews the modeling framework, covering data sources and preprocessing, model inputs and outputs, objectives, structures, and validation methods. This survey further discusses the opportunities and challenges faced by data-driven LCD models, including driving safety, uncertainty, as well as the integration and improvement of technical frameworks. 

---
# Exploring Multimodal Foundation AI and Expert-in-the-Loop for Sustainable Management of Wild Salmon Fisheries in Indigenous Rivers 

**Authors**: Chi Xu, Yili Jin, Sami Ma, Rongsheng Qian, Hao Fang, Jiangchuan Liu, Xue Liu, Edith C.H. Ngai, William I. Atlas, Katrina M. Connors, Mark A. Spoljaric  

**Link**: [PDF](https://arxiv.org/pdf/2505.06637)  

**Abstract**: Wild salmon are essential to the ecological, economic, and cultural sustainability of the North Pacific Rim. Yet climate variability, habitat loss, and data limitations in remote ecosystems that lack basic infrastructure support pose significant challenges to effective fisheries management. This project explores the integration of multimodal foundation AI and expert-in-the-loop frameworks to enhance wild salmon monitoring and sustainable fisheries management in Indigenous rivers across Pacific Northwest. By leveraging video and sonar-based monitoring, we develop AI-powered tools for automated species identification, counting, and length measurement, reducing manual effort, expediting delivery of results, and improving decision-making accuracy. Expert validation and active learning frameworks ensure ecological relevance while reducing annotation burdens. To address unique technical and societal challenges, we bring together a cross-domain, interdisciplinary team of university researchers, fisheries biologists, Indigenous stewardship practitioners, government agencies, and conservation organizations. Through these collaborations, our research fosters ethical AI co-development, open data sharing, and culturally informed fisheries management. 

---
# TAROT: Towards Essentially Domain-Invariant Robustness with Theoretical Justification 

**Authors**: Dongyoon Yang, Jihu Lee, Yongdai Kim  

**Link**: [PDF](https://arxiv.org/pdf/2505.06580)  

**Abstract**: Robust domain adaptation against adversarial attacks is a critical research area that aims to develop models capable of maintaining consistent performance across diverse and challenging domains. In this paper, we derive a new generalization bound for robust risk on the target domain using a novel divergence measure specifically designed for robust domain adaptation. Building upon this, we propose a new algorithm named TAROT, which is designed to enhance both domain adaptability and robustness. Through extensive experiments, TAROT not only surpasses state-of-the-art methods in accuracy and robustness but also significantly enhances domain generalization and scalability by effectively learning domain-invariant features. In particular, TAROT achieves superior performance on the challenging DomainNet dataset, demonstrating its ability to learn domain-invariant representations that generalize well across different domains, including unseen ones. These results highlight the broader applicability of our approach in real-world domain adaptation scenarios. 

---
# Online Feedback Efficient Active Target Discovery in Partially Observable Environments 

**Authors**: Anindya Sarkar, Binglin Ji, Yevgeniy Vorobeychik  

**Link**: [PDF](https://arxiv.org/pdf/2505.06535)  

**Abstract**: In various scientific and engineering domains, where data acquisition is costly, such as in medical imaging, environmental monitoring, or remote sensing, strategic sampling from unobserved regions, guided by prior observations, is essential to maximize target discovery within a limited sampling budget. In this work, we introduce Diffusion-guided Active Target Discovery (DiffATD), a novel method that leverages diffusion dynamics for active target discovery. DiffATD maintains a belief distribution over each unobserved state in the environment, using this distribution to dynamically balance exploration-exploitation. Exploration reduces uncertainty by sampling regions with the highest expected entropy, while exploitation targets areas with the highest likelihood of discovering the target, indicated by the belief distribution and an incrementally trained reward model designed to learn the characteristics of the target. DiffATD enables efficient target discovery in a partially observable environment within a fixed sampling budget, all without relying on any prior supervised training. Furthermore, DiffATD offers interpretability, unlike existing black-box policies that require extensive supervised training. Through extensive experiments and ablation studies across diverse domains, including medical imaging and remote sensing, we show that DiffATD performs significantly better than baselines and competitively with supervised methods that operate under full environmental observability. 

---
# A Point-Based Algorithm for Distributional Reinforcement Learning in Partially Observable Domains 

**Authors**: Larry Preuett III  

**Link**: [PDF](https://arxiv.org/pdf/2505.06518)  

**Abstract**: In many real-world planning tasks, agents must tackle uncertainty about the environment's state and variability in the outcomes of any chosen policy. We address both forms of uncertainty as a first step toward safer algorithms in partially observable settings. Specifically, we extend Distributional Reinforcement Learning (DistRL)-which models the entire return distribution for fully observable domains-to Partially Observable Markov Decision Processes (POMDPs), allowing an agent to learn the distribution of returns for each conditional plan. Concretely, we introduce new distributional Bellman operators for partial observability and prove their convergence under the supremum p-Wasserstein metric. We also propose a finite representation of these return distributions via psi-vectors, generalizing the classical alpha-vectors in POMDP solvers. Building on this, we develop Distributional Point-Based Value Iteration (DPBVI), which integrates psi-vectors into a standard point-based backup procedure-bridging DistRL and POMDP planning. By tracking return distributions, DPBVI naturally enables risk-sensitive control in domains where rare, high-impact events must be carefully managed. We provide source code to foster further research in robust decision-making under partial observability. 

---
# Text-to-CadQuery: A New Paradigm for CAD Generation with Scalable Large Model Capabilities 

**Authors**: Haoyang Xie, Feng Ju  

**Link**: [PDF](https://arxiv.org/pdf/2505.06507)  

**Abstract**: Computer-aided design (CAD) is fundamental to modern engineering and manufacturing, but creating CAD models still requires expert knowledge and specialized software. Recent advances in large language models (LLMs) open up the possibility of generative CAD, where natural language is directly translated into parametric 3D models. However, most existing methods generate task-specific command sequences that pretrained models cannot directly handle. These sequences must be converted into CAD representations such as CAD vectors before a 3D model can be produced, which requires training models from scratch and adds unnecessary complexity. To tackle this issue, we propose generating CadQuery code directly from text, leveraging the strengths of pretrained LLMs to produce 3D models without intermediate representations, using this Python-based scripting language. Since LLMs already excel at Python generation and spatial reasoning, fine-tuning them on Text-to-CadQuery data proves highly effective. Given that these capabilities typically improve with scale, we hypothesize that larger models will perform better after fine-tuning. To enable this, we augment the Text2CAD dataset with 170,000 CadQuery annotations. We fine-tune six open-source LLMs of varying sizes and observe consistent improvements. Our best model achieves a top-1 exact match of 69.3%, up from 58.8%, and reduces Chamfer Distance by 48.6%. Project page: this https URL. 

---
# On Definite Iterated Belief Revision with Belief Algebras 

**Authors**: Hua Meng, Zhiguo Long, Michael Sioutis, Zhengchun Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2505.06505)  

**Abstract**: Traditional logic-based belief revision research focuses on designing rules to constrain the behavior of revision operators. Frameworks have been proposed to characterize iterated revision rules, but they are often too loose, leading to multiple revision operators that all satisfy the rules under the same belief condition. In many practical applications, such as safety critical ones, it is important to specify a definite revision operator to enable agents to iteratively revise their beliefs in a deterministic way. In this paper, we propose a novel framework for iterated belief revision by characterizing belief information through preference relations. Semantically, both beliefs and new evidence are represented as belief algebras, which provide a rich and expressive foundation for belief revision. Building on traditional revision rules, we introduce additional postulates for revision with belief algebra, including an upper-bound constraint on the outcomes of revision. We prove that the revision result is uniquely determined given the current belief state and new evidence. Furthermore, to make the framework more useful in practice, we develop a particular algorithm for performing the proposed revision process. We argue that this approach may offer a more predictable and principled method for belief revision, making it suitable for real-world applications. 

---
# SmartPilot: A Multiagent CoPilot for Adaptive and Intelligent Manufacturing 

**Authors**: Chathurangi Shyalika, Renjith Prasad, Alaa Al Ghazo, Darssan Eswaramoorthi, Harleen Kaur, Sara Shree Muthuselvam, Amit Sheth  

**Link**: [PDF](https://arxiv.org/pdf/2505.06492)  

**Abstract**: In the dynamic landscape of Industry 4.0, achieving efficiency, precision, and adaptability is essential to optimize manufacturing operations. Industries suffer due to supply chain disruptions caused by anomalies, which are being detected by current AI models but leaving domain experts uncertain without deeper insights into these anomalies. Additionally, operational inefficiencies persist due to inaccurate production forecasts and the limited effectiveness of traditional AI models for processing complex sensor data. Despite these advancements, existing systems lack the seamless integration of these capabilities needed to create a truly unified solution for enhancing production and decision-making. We propose SmartPilot, a neurosymbolic, multiagent CoPilot designed for advanced reasoning and contextual decision-making to address these challenges. SmartPilot processes multimodal sensor data and is compact to deploy on edge devices. It focuses on three key tasks: anomaly prediction, production forecasting, and domain-specific question answering. By bridging the gap between AI capabilities and real-world industrial needs, SmartPilot empowers industries with intelligent decision-making and drives transformative innovation in manufacturing. The demonstration video, datasets, and supplementary materials are available at this https URL. 

---
# KCluster: An LLM-based Clustering Approach to Knowledge Component Discovery 

**Authors**: Yumou Wei, Paulo Carvalho, John Stamper  

**Link**: [PDF](https://arxiv.org/pdf/2505.06469)  

**Abstract**: Educators evaluate student knowledge using knowledge component (KC) models that map assessment questions to KCs. Still, designing KC models for large question banks remains an insurmountable challenge for instructors who need to analyze each question by hand. The growing use of Generative AI in education is expected only to aggravate this chronic deficiency of expert-designed KC models, as course engineers designing KCs struggle to keep up with the pace at which questions are generated. In this work, we propose KCluster, a novel KC discovery algorithm based on identifying clusters of congruent questions according to a new similarity metric induced by a large language model (LLM). We demonstrate in three datasets that an LLM can create an effective metric of question similarity, which a clustering algorithm can use to create KC models from questions with minimal human effort. Combining the strengths of LLM and clustering, KCluster generates descriptive KC labels and discovers KC models that predict student performance better than the best expert-designed models available. In anticipation of future work, we illustrate how KCluster can reveal insights into difficult KCs and suggest improvements to instruction. 

---
# Opening the Scope of Openness in AI 

**Authors**: Tamara Paris, AJung Moon, Jin Guo  

**Link**: [PDF](https://arxiv.org/pdf/2505.06464)  

**Abstract**: The concept of openness in AI has so far been heavily inspired by the definition and community practice of open source software. This positions openness in AI as having positive connotations; it introduces assumptions of certain advantages, such as collaborative innovation and transparency. However, the practices and benefits of open source software are not fully transferable to AI, which has its own challenges. Framing a notion of openness tailored to AI is crucial to addressing its growing societal implications, risks, and capabilities. We argue that considering the fundamental scope of openness in different disciplines will broaden discussions, introduce important perspectives, and reflect on what openness in AI should mean. Toward this goal, we qualitatively analyze 98 concepts of openness discovered from topic modeling, through which we develop a taxonomy of openness. Using this taxonomy as an instrument, we situate the current discussion on AI openness, identify gaps and highlight links with other disciplines. Our work contributes to the recent efforts in framing openness in AI by reflecting principles and practices of openness beyond open source software and calls for a more holistic view of openness in terms of actions, system properties, and ethical objectives. 

---
# Reliable Collaborative Conversational Agent System Based on LLMs and Answer Set Programming 

**Authors**: Yankai Zeng, Gopal Gupta  

**Link**: [PDF](https://arxiv.org/pdf/2505.06438)  

**Abstract**: As the Large-Language-Model-driven (LLM-driven) Artificial Intelligence (AI) bots became popular, people realized their strong potential in Task-Oriented Dialogue (TOD). However, bots relying wholly on LLMs are unreliable in their knowledge, and whether they can finally produce a correct result for the task is not guaranteed. The collaboration among these agents also remains a challenge, since the necessary information to convey is unclear, and the information transfer is by prompts -- unreliable, and malicious knowledge is easy to inject. With the help of logic programming tools such as Answer Set Programming (ASP), conversational agents can be built safely and reliably, and communication among the agents made more efficient and secure. We proposed an Administrator-Assistant Dual-Agent paradigm, where the two ASP-driven bots share the same knowledge base and complete their tasks independently, while the information can be passed by a Collaborative Rule Set (CRS). The knowledge and information conveyed are encapsulated and invisible to the users, ensuring the security of information transmission. We have constructed AutoManager, a dual-agent system for managing the drive-through window of a fast-food restaurant such as Taco Bell in the US. In AutoManager, the assistant bot takes the customer's order while the administrator bot manages the menu and food supply. We evaluated our AutoManager and compared it with the real-world Taco Bell Drive-Thru AI Order Taker, and the results show that our method is more reliable. 

---
# A Grounded Memory System For Smart Personal Assistants 

**Authors**: Felix Ocker, Jörg Deigmöller, Pavel Smirnov, Julian Eggert  

**Link**: [PDF](https://arxiv.org/pdf/2505.06328)  

**Abstract**: A wide variety of agentic AI applications - ranging from cognitive assistants for dementia patients to robotics - demand a robust memory system grounded in reality. In this paper, we propose such a memory system consisting of three components. First, we combine Vision Language Models for image captioning and entity disambiguation with Large Language Models for consistent information extraction during perception. Second, the extracted information is represented in a memory consisting of a knowledge graph enhanced by vector embeddings to efficiently manage relational information. Third, we combine semantic search and graph query generation for question answering via Retrieval Augmented Generation. We illustrate the system's working and potential using a real-world example. 

---
# BedreFlyt: Improving Patient Flows through Hospital Wards with Digital Twins 

**Authors**: Riccardo Sieve, Paul Kobialka, Laura Slaughter, Rudolf Schlatte, Einar Broch Johnsen, Silvia Lizeth Tapia Tarifa  

**Link**: [PDF](https://arxiv.org/pdf/2505.06287)  

**Abstract**: Digital twins are emerging as a valuable tool for short-term decision-making as well as for long-term strategic planning across numerous domains, including process industry, energy, space, transport, and healthcare. This paper reports on our ongoing work on designing a digital twin to enhance resource planning, e.g., for the in-patient ward needs in hospitals. By leveraging executable formal models for system exploration, ontologies for knowledge representation and an SMT solver for constraint satisfiability, our approach aims to explore hypothetical "what-if" scenarios to improve strategic planning processes, as well as to solve concrete, short-term decision-making tasks. Our proposed solution uses the executable formal model to turn a stream of arriving patients, that need to be hospitalized, into a stream of optimization problems, e.g., capturing daily inpatient ward needs, that can be solved by SMT techniques. The knowledge base, which formalizes domain knowledge, is used to model the needed configuration in the digital twin, allowing the twin to support both short-term decision-making and long-term strategic planning by generating scenarios spanning average-case as well as worst-case resource needs, depending on the expected treatment of patients, as well as ranging over variations in available resources, e.g., bed distribution in different rooms. We illustrate our digital twin architecture by considering the problem of bed bay allocation in a hospital ward. 

---
# H$^{\mathbf{3}}$DP: Triply-Hierarchical Diffusion Policy for Visuomotor Learning 

**Authors**: Yiyang Lu, Yufeng Tian, Zhecheng Yuan, Xianbang Wang, Pu Hua, Zhengrong Xue, Huazhe Xu  

**Link**: [PDF](https://arxiv.org/pdf/2505.07819)  

**Abstract**: Visuomotor policy learning has witnessed substantial progress in robotic manipulation, with recent approaches predominantly relying on generative models to model the action distribution. However, these methods often overlook the critical coupling between visual perception and action prediction. In this work, we introduce $\textbf{Triply-Hierarchical Diffusion Policy}~(\textbf{H$^{\mathbf{3}}$DP})$, a novel visuomotor learning framework that explicitly incorporates hierarchical structures to strengthen the integration between visual features and action generation. H$^{3}$DP contains $\mathbf{3}$ levels of hierarchy: (1) depth-aware input layering that organizes RGB-D observations based on depth information; (2) multi-scale visual representations that encode semantic features at varying levels of granularity; and (3) a hierarchically conditioned diffusion process that aligns the generation of coarse-to-fine actions with corresponding visual features. Extensive experiments demonstrate that H$^{3}$DP yields a $\mathbf{+27.5\%}$ average relative improvement over baselines across $\mathbf{44}$ simulation tasks and achieves superior performance in $\mathbf{4}$ challenging bimanual real-world manipulation tasks. Project Page: this https URL. 

---
# A class of distributed automata that contains the modal mu-fragment 

**Authors**: Veeti Ahvonen, Damian Heiman, Antti Kuusisto  

**Link**: [PDF](https://arxiv.org/pdf/2505.07816)  

**Abstract**: This paper gives a translation from the $\mu$-fragment of the graded modal $\mu$-calculus to a class of distributed message-passing automata. As a corollary, we obtain an alternative proof for a theorem from \cite{ahvonen_neurips} stating that recurrent graph neural networks working with reals and graded modal substitution calculus have the same expressive power in restriction to the logic monadic second-order logic MSO. 

---
# DexWild: Dexterous Human Interactions for In-the-Wild Robot Policies 

**Authors**: Tony Tao, Mohan Kumar Srirama, Jason Jingzhou Liu, Kenneth Shaw, Deepak Pathak  

**Link**: [PDF](https://arxiv.org/pdf/2505.07813)  

**Abstract**: Large-scale, diverse robot datasets have emerged as a promising path toward enabling dexterous manipulation policies to generalize to novel environments, but acquiring such datasets presents many challenges. While teleoperation provides high-fidelity datasets, its high cost limits its scalability. Instead, what if people could use their own hands, just as they do in everyday life, to collect data? In DexWild, a diverse team of data collectors uses their hands to collect hours of interactions across a multitude of environments and objects. To record this data, we create DexWild-System, a low-cost, mobile, and easy-to-use device. The DexWild learning framework co-trains on both human and robot demonstrations, leading to improved performance compared to training on each dataset individually. This combination results in robust robot policies capable of generalizing to novel environments, tasks, and embodiments with minimal additional robot-specific data. Experimental results demonstrate that DexWild significantly improves performance, achieving a 68.5% success rate in unseen environments-nearly four times higher than policies trained with robot data only-and offering 5.8x better cross-embodiment generalization. Video results, codebases, and instructions at this https URL 

---
# A Comparative Analysis of Static Word Embeddings for Hungarian 

**Authors**: Máté Gedeon  

**Link**: [PDF](https://arxiv.org/pdf/2505.07809)  

**Abstract**: This paper presents a comprehensive analysis of various static word embeddings for Hungarian, including traditional models such as Word2Vec, FastText, as well as static embeddings derived from BERT-based models using different extraction methods. We evaluate these embeddings on both intrinsic and extrinsic tasks to provide a holistic view of their performance. For intrinsic evaluation, we employ a word analogy task, which assesses the embeddings ability to capture semantic and syntactic relationships. Our results indicate that traditional static embeddings, particularly FastText, excel in this task, achieving high accuracy and mean reciprocal rank (MRR) scores. Among the BERT-based models, the X2Static method for extracting static embeddings demonstrates superior performance compared to decontextualized and aggregate methods, approaching the effectiveness of traditional static embeddings. For extrinsic evaluation, we utilize a bidirectional LSTM model to perform Named Entity Recognition (NER) and Part-of-Speech (POS) tagging tasks. The results reveal that embeddings derived from dynamic models, especially those extracted using the X2Static method, outperform purely static embeddings. Notably, ELMo embeddings achieve the highest accuracy in both NER and POS tagging tasks, underscoring the benefits of contextualized representations even when used in a static form. Our findings highlight the continued relevance of static word embeddings in NLP applications and the potential of advanced extraction methods to enhance the utility of BERT-based models. This piece of research contributes to the understanding of embedding performance in the Hungarian language and provides valuable insights for future developments in the field. The training scripts, evaluation codes, restricted vocabulary, and extracted embeddings will be made publicly available to support further research and reproducibility. 

---
# Improving Trajectory Stitching with Flow Models 

**Authors**: Reece O'Mahoney, Wanming Yu, Ioannis Havoutis  

**Link**: [PDF](https://arxiv.org/pdf/2505.07802)  

**Abstract**: Generative models have shown great promise as trajectory planners, given their affinity to modeling complex distributions and guidable inference process. Previous works have successfully applied these in the context of robotic manipulation but perform poorly when the required solution does not exist as a complete trajectory within the training set. We identify that this is a result of being unable to plan via stitching, and subsequently address the architectural and dataset choices needed to remedy this. On top of this, we propose a novel addition to the training and inference procedures to both stabilize and enhance these capabilities. We demonstrate the efficacy of our approach by generating plans with out of distribution boundary conditions and performing obstacle avoidance on the Franka Panda in simulation and on real hardware. In both of these tasks our method performs significantly better than the baselines and is able to avoid obstacles up to four times as large. 

---
# Learning Dynamics in Continual Pre-Training for Large Language Models 

**Authors**: Xingjin Wang, Howe Tissue, Lu Wang, Linjing Li, Daniel Dajun Zeng  

**Link**: [PDF](https://arxiv.org/pdf/2505.07796)  

**Abstract**: Continual Pre-Training (CPT) has become a popular and effective method to apply strong foundation models to specific downstream tasks. In this work, we explore the learning dynamics throughout the CPT process for large language models. We specifically focus on how general and downstream domain performance evolves at each training step, with domain performance measured via validation losses. We have observed that the CPT loss curve fundamentally characterizes the transition from one curve to another hidden curve, and could be described by decoupling the effects of distribution shift and learning rate annealing. We derive a CPT scaling law that combines the two factors, enabling the prediction of loss at any (continual) training steps and across learning rate schedules (LRS) in CPT. Our formulation presents a comprehensive understanding of several critical factors in CPT, including loss potential, peak learning rate, training steps, replay ratio, etc. Moreover, our approach can be adapted to customize training hyper-parameters to different CPT goals such as balancing general and domain-specific performance. Extensive experiments demonstrate that our scaling law holds across various CPT datasets and training hyper-parameters. 

---
# Overflow Prevention Enhances Long-Context Recurrent LLMs 

**Authors**: Assaf Ben-Kish, Itamar Zimerman, M. Jehanzeb Mirza, James Glass, Leonid Karlinsky, Raja Giryes  

**Link**: [PDF](https://arxiv.org/pdf/2505.07793)  

**Abstract**: A recent trend in LLMs is developing recurrent sub-quadratic models that improve long-context processing efficiency. We investigate leading large long-context models, focusing on how their fixed-size recurrent memory affects their performance. Our experiments reveal that, even when these models are trained for extended contexts, their use of long contexts remains underutilized. Specifically, we demonstrate that a chunk-based inference procedure, which identifies and processes only the most relevant portion of the input can mitigate recurrent memory failures and be effective for many long-context tasks: On LongBench, our method improves the overall performance of Falcon3-Mamba-Inst-7B by 14%, Falcon-Mamba-Inst-7B by 28%, RecurrentGemma-IT-9B by 50%, and RWKV6-Finch-7B by 51%. Surprisingly, this simple approach also leads to state-of-the-art results in the challenging LongBench v2 benchmark, showing competitive performance with equivalent size Transformers. Furthermore, our findings raise questions about whether recurrent models genuinely exploit long-range dependencies, as our single-chunk strategy delivers stronger performance - even in tasks that presumably require cross-context relations. 

---
# Must Read: A Systematic Survey of Computational Persuasion 

**Authors**: Nimet Beyza Bozdag, Shuhaib Mehri, Xiaocheng Yang, Hyeonjeong Ha, Zirui Cheng, Esin Durmus, Jiaxuan You, Heng Ji, Gokhan Tur, Dilek Hakkani-Tür  

**Link**: [PDF](https://arxiv.org/pdf/2505.07775)  

**Abstract**: Persuasion is a fundamental aspect of communication, influencing decision-making across diverse contexts, from everyday conversations to high-stakes scenarios such as politics, marketing, and law. The rise of conversational AI systems has significantly expanded the scope of persuasion, introducing both opportunities and risks. AI-driven persuasion can be leveraged for beneficial applications, but also poses threats through manipulation and unethical influence. Moreover, AI systems are not only persuaders, but also susceptible to persuasion, making them vulnerable to adversarial attacks and bias reinforcement. Despite rapid advancements in AI-generated persuasive content, our understanding of what makes persuasion effective remains limited due to its inherently subjective and context-dependent nature. In this survey, we provide a comprehensive overview of computational persuasion, structured around three key perspectives: (1) AI as a Persuader, which explores AI-generated persuasive content and its applications; (2) AI as a Persuadee, which examines AI's susceptibility to influence and manipulation; and (3) AI as a Persuasion Judge, which analyzes AI's role in evaluating persuasive strategies, detecting manipulation, and ensuring ethical persuasion. We introduce a taxonomy for computational persuasion research and discuss key challenges, including evaluating persuasiveness, mitigating manipulative persuasion, and developing responsible AI-driven persuasive systems. Our survey outlines future research directions to enhance the safety, fairness, and effectiveness of AI-powered persuasion while addressing the risks posed by increasingly capable language models. 

---
# Enhancing Code Generation via Bidirectional Comment-Level Mutual Grounding 

**Authors**: Yifeng Di, Tianyi Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2505.07768)  

**Abstract**: Large Language Models (LLMs) have demonstrated unprecedented capability in code generation. However, LLM-generated code is still plagued with a wide range of functional errors, especially for complex programming tasks that LLMs have not seen before. Recent studies have shown that developers often struggle with inspecting and fixing incorrect code generated by LLMs, diminishing their productivity and trust in LLM-based code generation. Inspired by the mutual grounding theory in communication, we propose an interactive approach that leverages code comments as a medium for developers and LLMs to establish a shared understanding. Our approach facilitates iterative grounding by interleaving code generation, inline comment generation, and contextualized user feedback through editable comments to align generated code with developer intent. We evaluated our approach on two popular benchmarks and demonstrated that our approach significantly improved multiple state-of-the-art LLMs, e.g., 17.1% pass@1 improvement for code-davinci-002 on HumanEval. Furthermore, we conducted a user study with 12 participants in comparison to two baselines: (1) interacting with GitHub Copilot, and (2) interacting with a multi-step code generation paradigm called Multi-Turn Program Synthesis. Participants completed the given programming tasks 16.7% faster and with 10.5% improvement in task success rate when using our approach. Both results show that interactively refining code comments enables the collaborative establishment of mutual grounding, leading to more accurate code generation and higher developer confidence. 

---
# Benchmarking of CPU-intensive Stream Data Processing in The Edge Computing Systems 

**Authors**: Tomasz Szydlo, Viacheslaw Horbanow, Dev Nandan Jha, Shashikant Ilager, Aleksander Slominski, Rajiv Ranjan  

**Link**: [PDF](https://arxiv.org/pdf/2505.07755)  

**Abstract**: Edge computing has emerged as a pivotal technology, offering significant advantages such as low latency, enhanced data security, and reduced reliance on centralized cloud infrastructure. These benefits are crucial for applications requiring real-time data processing or strict security measures. Despite these advantages, edge devices operating within edge clusters are often underutilized. This inefficiency is mainly due to the absence of a holistic performance profiling mechanism which can help dynamically adjust the desired system configuration for a given workload. Since edge computing environments involve a complex interplay between CPU frequency, power consumption, and application performance, a deeper understanding of these correlations is essential. By uncovering these relationships, it becomes possible to make informed decisions that enhance both computational efficiency and energy savings. To address this gap, this paper evaluates the power consumption and performance characteristics of a single processing node within an edge cluster using a synthetic microbenchmark by varying the workload size and CPU frequency. The results show how an optimal measure can lead to optimized usage of edge resources, given both performance and power consumption. 

---
# Guiding Data Collection via Factored Scaling Curves 

**Authors**: Lihan Zha, Apurva Badithela, Michael Zhang, Justin Lidard, Jeremy Bao, Emily Zhou, David Snyder, Allen Z. Ren, Dhruv Shah, Anirudha Majumdar  

**Link**: [PDF](https://arxiv.org/pdf/2505.07728)  

**Abstract**: Generalist imitation learning policies trained on large datasets show great promise for solving diverse manipulation tasks. However, to ensure generalization to different conditions, policies need to be trained with data collected across a large set of environmental factor variations (e.g., camera pose, table height, distractors) $-$ a prohibitively expensive undertaking, if done exhaustively. We introduce a principled method for deciding what data to collect and how much to collect for each factor by constructing factored scaling curves (FSC), which quantify how policy performance varies as data scales along individual or paired factors. These curves enable targeted data acquisition for the most influential factor combinations within a given budget. We evaluate the proposed method through extensive simulated and real-world experiments, across both training-from-scratch and fine-tuning settings, and show that it boosts success rates in real-world tasks in new environments by up to 26% over existing data-collection strategies. We further demonstrate how factored scaling curves can effectively guide data collection using an offline metric, without requiring real-world evaluation at scale. 

---
# Hybrid Spiking Vision Transformer for Object Detection with Event Cameras 

**Authors**: Qi Xu, Jie Deng, Jiangrong Shen, Biwu Chen, Huajin Tang, Gang Pan  

**Link**: [PDF](https://arxiv.org/pdf/2505.07715)  

**Abstract**: Event-based object detection has gained increasing attention due to its advantages such as high temporal resolution, wide dynamic range, and asynchronous address-event representation. Leveraging these advantages, Spiking Neural Networks (SNNs) have emerged as a promising approach, offering low energy consumption and rich spatiotemporal dynamics. To further enhance the performance of event-based object detection, this study proposes a novel hybrid spike vision Transformer (HsVT) model. The HsVT model integrates a spatial feature extraction module to capture local and global features, and a temporal feature extraction module to model time dependencies and long-term patterns in event sequences. This combination enables HsVT to capture spatiotemporal features, improving its capability to handle complex event-based object detection tasks. To support research in this area, we developed and publicly released The Fall Detection Dataset as a benchmark for event-based object detection tasks. This dataset, captured using an event-based camera, ensures facial privacy protection and reduces memory usage due to the event representation format. We evaluated the HsVT model on GEN1 and Fall Detection datasets across various model sizes. Experimental results demonstrate that HsVT achieves significant performance improvements in event detection with fewer parameters. 

---
# Circuit Partitioning Using Large Language Models for Quantum Compilation and Simulations 

**Authors**: Pranav Sinha, Sumit Kumar Jha, Sunny Raj  

**Link**: [PDF](https://arxiv.org/pdf/2505.07711)  

**Abstract**: We are in the midst of the noisy intermediate-scale quantum (NISQ) era, where quantum computers are limited by noisy gates, some of which are more error-prone than others and can render the final computation incomprehensible. Quantum circuit compilation algorithms attempt to minimize these noisy gates when mapping quantum algorithms onto quantum hardware but face computational challenges that restrict their application to circuits with no more than 5-6 qubits, necessitating the need to partition large circuits before the application of noisy quantum gate minimization algorithms. The existing generation of these algorithms is heuristic in nature and does not account for downstream gate minimization tasks. Large language models (LLMs) have the potential to change this and help improve quantum circuit partitions. This paper investigates the use of LLMs, such as Llama and Mistral, for partitioning quantum circuits by capitalizing on their abilities to understand and generate code, including QASM. Specifically, we teach LLMs to partition circuits using the quick partition approach of the Berkeley Quantum Synthesis Toolkit. Through experimental evaluations, we show that careful fine-tuning of open source LLMs enables us to obtain an accuracy of 53.4% for the partition task while over-the-shelf LLMs are unable to correctly partition circuits, using standard 1-shot and few-shot training approaches. 

---
# Lightweight End-to-end Text-to-speech Synthesis for low resource on-device applications 

**Authors**: Biel Tura Vecino, Adam Gabryś, Daniel Mątwicki, Andrzej Pomirski, Tom Iddon, Marius Cotescu, Jaime Lorenzo-Trueba  

**Link**: [PDF](https://arxiv.org/pdf/2505.07701)  

**Abstract**: Recent works have shown that modelling raw waveform directly from text in an end-to-end (E2E) fashion produces more natural-sounding speech than traditional neural text-to-speech (TTS) systems based on a cascade or two-stage approach. However, current E2E state-of-the-art models are computationally complex and memory-consuming, making them unsuitable for real-time offline on-device applications in low-resource scenarios. To address this issue, we propose a Lightweight E2E-TTS (LE2E) model that generates high-quality speech requiring minimal computational resources. We evaluate the proposed model on the LJSpeech dataset and show that it achieves state-of-the-art performance while being up to $90\%$ smaller in terms of model parameters and $10\times$ faster in real-time-factor. Furthermore, we demonstrate that the proposed E2E training paradigm achieves better quality compared to an equivalent architecture trained in a two-stage approach. Our results suggest that LE2E is a promising approach for developing real-time, high quality, low-resource TTS applications for on-device applications. 

---
# Multimodal Survival Modeling in the Age of Foundation Models 

**Authors**: Steven Song, Morgan Borjigin-Wang, Irene Madejski, Robert L. Grossman  

**Link**: [PDF](https://arxiv.org/pdf/2505.07683)  

**Abstract**: The Cancer Genome Atlas (TCGA) has enabled novel discoveries and served as a large-scale reference through its harmonized genomics, clinical, and image data. Prior studies have trained bespoke cancer survival prediction models from unimodal or multimodal TCGA data. A modern paradigm in biomedical deep learning is the development of foundation models (FMs) to derive meaningful feature embeddings, agnostic to a specific modeling task. Biomedical text especially has seen growing development of FMs. While TCGA contains free-text data as pathology reports, these have been historically underutilized. Here, we investigate the feasibility of training classical, multimodal survival models over zero-shot embeddings extracted by FMs. We show the ease and additive effect of multimodal fusion, outperforming unimodal models. We demonstrate the benefit of including pathology report text and rigorously evaluate the effect of model-based text summarization and hallucination. Overall, we modernize survival modeling by leveraging FMs and information extraction from pathology reports. 

---
# Simple Semi-supervised Knowledge Distillation from Vision-Language Models via $\mathbf{\texttt{D}}$ual-$\mathbf{\texttt{H}}$ead $\mathbf{\texttt{O}}$ptimization 

**Authors**: Seongjae Kang, Dong Bok Lee, Hyungjoon Jang, Sung Ju Hwang  

**Link**: [PDF](https://arxiv.org/pdf/2505.07675)  

**Abstract**: Vision-language models (VLMs) have achieved remarkable success across diverse tasks by leveraging rich textual information with minimal labeled data. However, deploying such large models remains challenging, particularly in resource-constrained environments. Knowledge distillation (KD) offers a well-established solution to this problem; however, recent KD approaches from VLMs often involve multi-stage training or additional tuning, increasing computational overhead and optimization complexity. In this paper, we propose $\mathbf{\texttt{D}}$ual-$\mathbf{\texttt{H}}$ead $\mathbf{\texttt{O}}$ptimization ($\mathbf{\texttt{DHO}}$) -- a simple yet effective KD framework that transfers knowledge from VLMs to compact, task-specific models in semi-supervised settings. Specifically, we introduce dual prediction heads that independently learn from labeled data and teacher predictions, and propose to linearly combine their outputs during inference. We observe that $\texttt{DHO}$ mitigates gradient conflicts between supervised and distillation signals, enabling more effective feature learning than single-head KD baselines. As a result, extensive experiments show that $\texttt{DHO}$ consistently outperforms baselines across multiple domains and fine-grained datasets. Notably, on ImageNet, it achieves state-of-the-art performance, improving accuracy by 3% and 0.1% with 1% and 10% labeled data, respectively, while using fewer parameters. 

---
# OnPrem.LLM: A Privacy-Conscious Document Intelligence Toolkit 

**Authors**: Arun S. Maiya  

**Link**: [PDF](https://arxiv.org/pdf/2505.07672)  

**Abstract**: We present this http URL, a Python-based toolkit for applying large language models (LLMs) to sensitive, non-public data in offline or restricted environments. The system is designed for privacy-preserving use cases and provides prebuilt pipelines for document processing and storage, retrieval-augmented generation (RAG), information extraction, summarization, classification, and prompt/output processing with minimal configuration. this http URL supports multiple LLM backends -- including this http URL, Ollama, vLLM, and Hugging Face Transformers -- with quantized model support, GPU acceleration, and seamless backend switching. Although designed for fully local execution, this http URL also supports integration with a wide range of cloud LLM providers when permitted, enabling hybrid deployments that balance performance with data control. A no-code web interface extends accessibility to non-technical users. 

---
# Benchmarking Retrieval-Augmented Generation for Chemistry 

**Authors**: Xianrui Zhong, Bowen Jin, Siru Ouyang, Yanzhen Shen, Qiao Jin, Yin Fang, Zhiyong Lu, Jiawei Han  

**Link**: [PDF](https://arxiv.org/pdf/2505.07671)  

**Abstract**: Retrieval-augmented generation (RAG) has emerged as a powerful framework for enhancing large language models (LLMs) with external knowledge, particularly in scientific domains that demand specialized and dynamic information. Despite its promise, the application of RAG in the chemistry domain remains underexplored, primarily due to the lack of high-quality, domain-specific corpora and well-curated evaluation benchmarks. In this work, we introduce ChemRAG-Bench, a comprehensive benchmark designed to systematically assess the effectiveness of RAG across a diverse set of chemistry-related tasks. The accompanying chemistry corpus integrates heterogeneous knowledge sources, including scientific literature, the PubChem database, PubMed abstracts, textbooks, and Wikipedia entries. In addition, we present ChemRAG-Toolkit, a modular and extensible RAG toolkit that supports five retrieval algorithms and eight LLMs. Using ChemRAG-Toolkit, we demonstrate that RAG yields a substantial performance gain -- achieving an average relative improvement of 17.4% over direct inference methods. We further conduct in-depth analyses on retriever architectures, corpus selection, and the number of retrieved passages, culminating in practical recommendations to guide future research and deployment of RAG systems in the chemistry domain. The code and data is available at this https URL. 

---
# A Case Study Investigating the Role of Generative AI in Quality Evaluations of Epics in Agile Software Development 

**Authors**: Werner Geyer, Jessica He, Daita Sarkar, Michelle Brachman, Chris Hammond, Jennifer Heins, Zahra Ashktorab, Carlos Rosemberg, Charlie Hill  

**Link**: [PDF](https://arxiv.org/pdf/2505.07664)  

**Abstract**: The broad availability of generative AI offers new opportunities to support various work domains, including agile software development. Agile epics are a key artifact for product managers to communicate requirements to stakeholders. However, in practice, they are often poorly defined, leading to churn, delivery delays, and cost overruns. In this industry case study, we investigate opportunities for large language models (LLMs) to evaluate agile epic quality in a global company. Results from a user study with 17 product managers indicate how LLM evaluations could be integrated into their work practices, including perceived values and usage in improving their epics. High levels of satisfaction indicate that agile epics are a new, viable application of AI evaluations. However, our findings also outline challenges, limitations, and adoption barriers that can inform both practitioners and researchers on the integration of such evaluations into future agile work practices. 

---
# Chronocept: Instilling a Sense of Time in Machines 

**Authors**: Krish Goel, Sanskar Pandey, KS Mahadevan, Harsh Kumar, Vishesh Khadaria  

**Link**: [PDF](https://arxiv.org/pdf/2505.07637)  

**Abstract**: Human cognition is deeply intertwined with a sense of time, known as Chronoception. This sense allows us to judge how long facts remain valid and when knowledge becomes outdated. Despite progress in vision, language, and motor control, AI still struggles to reason about temporal validity. We introduce Chronocept, the first benchmark to model temporal validity as a continuous probability distribution over time. Using skew-normal curves fitted along semantically decomposed temporal axes, Chronocept captures nuanced patterns of emergence, decay, and peak relevance. It includes two datasets: Benchmark I (atomic facts) and Benchmark II (multi-sentence passages). Annotations show strong inter-annotator agreement (84% and 89%). Our baselines predict curve parameters - location, scale, and skewness - enabling interpretable, generalizable learning and outperforming classification-based approaches. Chronocept fills a foundational gap in AI's temporal reasoning, supporting applications in knowledge grounding, fact-checking, retrieval-augmented generation (RAG), and proactive agents. Code and data are publicly available. 

---
# Neural Brain: A Neuroscience-inspired Framework for Embodied Agents 

**Authors**: Jian Liu, Xiongtao Shi, Thai Duy Nguyen, Haitian Zhang, Tianxiang Zhang, Wei Sun, Yanjie Li, Athanasios V. Vasilakos, Giovanni Iacca, Arshad Ali Khan, Arvind Kumar, Jae Won Cho, Ajmal Mian, Lihua Xie, Erik Cambria, Lin Wang  

**Link**: [PDF](https://arxiv.org/pdf/2505.07634)  

**Abstract**: The rapid evolution of artificial intelligence (AI) has shifted from static, data-driven models to dynamic systems capable of perceiving and interacting with real-world environments. Despite advancements in pattern recognition and symbolic reasoning, current AI systems, such as large language models, remain disembodied, unable to physically engage with the world. This limitation has driven the rise of embodied AI, where autonomous agents, such as humanoid robots, must navigate and manipulate unstructured environments with human-like adaptability. At the core of this challenge lies the concept of Neural Brain, a central intelligence system designed to drive embodied agents with human-like adaptability. A Neural Brain must seamlessly integrate multimodal sensing and perception with cognitive capabilities. Achieving this also requires an adaptive memory system and energy-efficient hardware-software co-design, enabling real-time action in dynamic environments. This paper introduces a unified framework for the Neural Brain of embodied agents, addressing two fundamental challenges: (1) defining the core components of Neural Brain and (2) bridging the gap between static AI models and the dynamic adaptability required for real-world deployment. To this end, we propose a biologically inspired architecture that integrates multimodal active sensing, perception-cognition-action function, neuroplasticity-based memory storage and updating, and neuromorphic hardware/software optimization. Furthermore, we also review the latest research on embodied agents across these four aspects and analyze the gap between current AI systems and human intelligence. By synthesizing insights from neuroscience, we outline a roadmap towards the development of generalizable, autonomous agents capable of human-level intelligence in real-world scenarios. 

---
# Bang for the Buck: Vector Search on Cloud CPUs 

**Authors**: Leonardo Kuffo, Peter Boncz  

**Link**: [PDF](https://arxiv.org/pdf/2505.07621)  

**Abstract**: Vector databases have emerged as a new type of systems that support efficient querying of high-dimensional vectors. Many of these offer their database as a service in the cloud. However, the variety of available CPUs and the lack of vector search benchmarks across CPUs make it difficult for users to choose one. In this study, we show that CPU microarchitectures available in the cloud perform significantly differently across vector search scenarios. For instance, in an IVF index on float32 vectors, AMD's Zen4 gives almost 3x more queries per second (QPS) compared to Intel's Sapphire Rapids, but for HNSW indexes, the tables turn. However, when looking at the number of queries per dollar (QP$), Graviton3 is the best option for most indexes and quantization settings, even over Graviton4 (Table 1). With this work, we hope to guide users in getting the best "bang for the buck" when deploying vector search systems. 

---
# Diffused Responsibility: Analyzing the Energy Consumption of Generative Text-to-Audio Diffusion Models 

**Authors**: Riccardo Passoni, Francesca Ronchini, Luca Comanducci, Romain Serizel, Fabio Antonacci  

**Link**: [PDF](https://arxiv.org/pdf/2505.07615)  

**Abstract**: Text-to-audio models have recently emerged as a powerful technology for generating sound from textual descriptions. However, their high computational demands raise concerns about energy consumption and environmental impact. In this paper, we conduct an analysis of the energy usage of 7 state-of-the-art text-to-audio diffusion-based generative models, evaluating to what extent variations in generation parameters affect energy consumption at inference time. We also aim to identify an optimal balance between audio quality and energy consumption by considering Pareto-optimal solutions across all selected models. Our findings provide insights into the trade-offs between performance and environmental impact, contributing to the development of more efficient generative audio models. 

---
# Concept-Level Explainability for Auditing & Steering LLM Responses 

**Authors**: Kenza Amara, Rita Sevastjanova, Mennatallah El-Assady  

**Link**: [PDF](https://arxiv.org/pdf/2505.07610)  

**Abstract**: As large language models (LLMs) become widely deployed, concerns about their safety and alignment grow. An approach to steer LLM behavior, such as mitigating biases or defending against jailbreaks, is to identify which parts of a prompt influence specific aspects of the model's output. Token-level attribution methods offer a promising solution, but still struggle in text generation, explaining the presence of each token in the output separately, rather than the underlying semantics of the entire LLM response. We introduce ConceptX, a model-agnostic, concept-level explainability method that identifies the concepts, i.e., semantically rich tokens in the prompt, and assigns them importance based on the outputs' semantic similarity. Unlike current token-level methods, ConceptX also offers to preserve context integrity through in-place token replacements and supports flexible explanation goals, e.g., gender bias. ConceptX enables both auditing, by uncovering sources of bias, and steering, by modifying prompts to shift the sentiment or reduce the harmfulness of LLM responses, without requiring retraining. Across three LLMs, ConceptX outperforms token-level methods like TokenSHAP in both faithfulness and human alignment. Steering tasks boost sentiment shift by 0.252 versus 0.131 for random edits and lower attack success rates from 0.463 to 0.242, outperforming attribution and paraphrasing baselines. While prompt engineering and self-explaining methods sometimes yield safer responses, ConceptX offers a transparent and faithful alternative for improving LLM safety and alignment, demonstrating the practical value of attribution-based explainability in guiding LLM behavior. 

---
# MiMo: Unlocking the Reasoning Potential of Language Model -- From Pretraining to Posttraining 

**Authors**: Xiaomi LLM-Core Team, Bingquan Xia, Bowen Shen, Cici, Dawei Zhu, Di Zhang, Gang Wang, Hailin Zhang, Huaqiu Liu, Jiebao Xiao, Jinhao Dong, Liang Zhao, Peidian Li, Peng Wang, Shihua Yu, Shimao Chen, Weikun Wang, Wenhan Ma, Xiangwei Deng, Yi Huang, Yifan Song, Zihan Jiang, Bowen Ye, Can Cai, Chenhong He, Dong Zhang, Duo Zhang, Guoan Wang, Hao Tian, Haochen Zhao, Heng Qu, Hongshen Xu, Jun Shi, Kainan Bao, QingKai Fang, Kang Zhou, Kangyang Zhou, Lei Li, Menghang Zhu, Nuo Chen, Qiantong Wang, Shaohui Liu, Shicheng Li, Shuhao Gu, Shuhuai Ren, Shuo Liu, Sirui Deng, Weiji Zhuang, Weiwei Lv, Wenyu Yang, Xin Zhang, Xing Yong, Xing Zhang, Xingchen Song, Xinzhe Xu, Xu Wang, Yihan Yan, Yu Tu, Yuanyuan Tian, Yudong Wang, Yue Yu, Zhenru Lin, Zhichao Song, Zihao Yue  

**Link**: [PDF](https://arxiv.org/pdf/2505.07608)  

**Abstract**: We present MiMo-7B, a large language model born for reasoning tasks, with optimization across both pre-training and post-training stages. During pre-training, we enhance the data preprocessing pipeline and employ a three-stage data mixing strategy to strengthen the base model's reasoning potential. MiMo-7B-Base is pre-trained on 25 trillion tokens, with additional Multi-Token Prediction objective for enhanced performance and accelerated inference speed. During post-training, we curate a dataset of 130K verifiable mathematics and programming problems for reinforcement learning, integrating a test-difficulty-driven code-reward scheme to alleviate sparse-reward issues and employing strategic data resampling to stabilize training. Extensive evaluations show that MiMo-7B-Base possesses exceptional reasoning potential, outperforming even much larger 32B models. The final RL-tuned model, MiMo-7B-RL, achieves superior performance on mathematics, code and general reasoning tasks, surpassing the performance of OpenAI o1-mini. The model checkpoints are available at this https URL. 

---
# Characterizing the Investigative Methods of Fictional Detectives with Large Language Models 

**Authors**: Edirlei Soares de Lima, Marco A. Casanova, Bruno Feijó, Antonio L. Furtado  

**Link**: [PDF](https://arxiv.org/pdf/2505.07601)  

**Abstract**: Detective fiction, a genre defined by its complex narrative structures and character-driven storytelling, presents unique challenges for computational narratology, a research field focused on integrating literary theory into automated narrative generation. While traditional literary studies have offered deep insights into the methods and archetypes of fictional detectives, these analyses often focus on a limited number of characters and lack the scalability needed for the extraction of unique traits that can be used to guide narrative generation methods. In this paper, we present an AI-driven approach for systematically characterizing the investigative methods of fictional detectives. Our multi-phase workflow explores the capabilities of 15 Large Language Models (LLMs) to extract, synthesize, and validate distinctive investigative traits of fictional detectives. This approach was tested on a diverse set of seven iconic detectives - Hercule Poirot, Sherlock Holmes, William Murdoch, Columbo, Father Brown, Miss Marple, and Auguste Dupin - capturing the distinctive investigative styles that define each character. The identified traits were validated against existing literary analyses and further tested in a reverse identification phase, achieving an overall accuracy of 91.43%, demonstrating the method's effectiveness in capturing the distinctive investigative approaches of each detective. This work contributes to the broader field of computational narratology by providing a scalable framework for character analysis, with potential applications in AI-driven interactive storytelling and automated narrative generation. 

---
# Reinforced Internal-External Knowledge Synergistic Reasoning for Efficient Adaptive Search Agent 

**Authors**: Ziyang Huang, Xiaowei Yuan, Yiming Ju, Jun Zhao, Kang Liu  

**Link**: [PDF](https://arxiv.org/pdf/2505.07596)  

**Abstract**: Retrieval-augmented generation (RAG) is a common strategy to reduce hallucinations in Large Language Models (LLMs). While reinforcement learning (RL) can enable LLMs to act as search agents by activating retrieval capabilities, existing ones often underutilize their internal knowledge. This can lead to redundant retrievals, potential harmful knowledge conflicts, and increased inference latency. To address these limitations, an efficient and adaptive search agent capable of discerning optimal retrieval timing and synergistically integrating parametric (internal) and retrieved (external) knowledge is in urgent need. This paper introduces the Reinforced Internal-External Knowledge Synergistic Reasoning Agent (IKEA), which could indentify its own knowledge boundary and prioritize the utilization of internal knowledge, resorting to external search only when internal knowledge is deemed insufficient. This is achieved using a novel knowledge-boundary aware reward function and a knowledge-boundary aware training dataset. These are designed for internal-external knowledge synergy oriented RL, incentivizing the model to deliver accurate answers, minimize unnecessary retrievals, and encourage appropriate external searches when its own knowledge is lacking. Evaluations across multiple knowledge reasoning tasks demonstrate that IKEA significantly outperforms baseline methods, reduces retrieval frequency significantly, and exhibits robust generalization capabilities. 

---
# A Multi-Dimensional Constraint Framework for Evaluating and Improving Instruction Following in Large Language Models 

**Authors**: Junjie Ye, Caishuang Huang, Zhuohan Chen, Wenjie Fu, Chenyuan Yang, Leyi Yang, Yilong Wu, Peng Wang, Meng Zhou, Xiaolong Yang, Tao Gui, Qi Zhang, Zhongchao Shi, Jianping Fan, Xuanjing Huang  

**Link**: [PDF](https://arxiv.org/pdf/2505.07591)  

**Abstract**: Instruction following evaluates large language models (LLMs) on their ability to generate outputs that adhere to user-defined constraints. However, existing benchmarks often rely on templated constraint prompts, which lack the diversity of real-world usage and limit fine-grained performance assessment. To fill this gap, we propose a multi-dimensional constraint framework encompassing three constraint patterns, four constraint categories, and four difficulty levels. Building on this framework, we develop an automated instruction generation pipeline that performs constraint expansion, conflict detection, and instruction rewriting, yielding 1,200 code-verifiable instruction-following test samples. We evaluate 19 LLMs across seven model families and uncover substantial variation in performance across constraint forms. For instance, average performance drops from 77.67% at Level I to 32.96% at Level IV. Furthermore, we demonstrate the utility of our approach by using it to generate data for reinforcement learning, achieving substantial gains in instruction following without degrading general performance. In-depth analysis indicates that these gains stem primarily from modifications in the model's attention modules parameters, which enhance constraint recognition and adherence. Code and data are available in this https URL. 

---
# Evaluating Modern Visual Anomaly Detection Approaches in Semiconductor Manufacturing: A Comparative Study 

**Authors**: Manuel Barusco, Francesco Borsatti, Youssef Ben Khalifa, Davide Dalle Pezze, Gian Antonio Susto  

**Link**: [PDF](https://arxiv.org/pdf/2505.07576)  

**Abstract**: Semiconductor manufacturing is a complex, multistage process. Automated visual inspection of Scanning Electron Microscope (SEM) images is indispensable for minimizing equipment downtime and containing costs. Most previous research considers supervised approaches, assuming a sufficient number of anomalously labeled samples. On the contrary, Visual Anomaly Detection (VAD), an emerging research domain, focuses on unsupervised learning, avoiding the costly defect collection phase while providing explanations of the predictions. We introduce a benchmark for VAD in the semiconductor domain by leveraging the MIIC dataset. Our results demonstrate the efficacy of modern VAD approaches in this field. 

---
# Robust Kidney Abnormality Segmentation: A Validation Study of an AI-Based Framework 

**Authors**: Sarah de Boer, Hartmut Häntze, Kiran Vaidhya Venkadesh, Myrthe A. D. Buser, Gabriel E. Humpire Mamani, Lina Xu, Lisa C. Adams, Jawed Nawabi, Keno K. Bressem, Bram van Ginneken, Mathias Prokop, Alessa Hering  

**Link**: [PDF](https://arxiv.org/pdf/2505.07573)  

**Abstract**: Kidney abnormality segmentation has important potential to enhance the clinical workflow, especially in settings requiring quantitative assessments. Kidney volume could serve as an important biomarker for renal diseases, with changes in volume correlating directly with kidney function. Currently, clinical practice often relies on subjective visual assessment for evaluating kidney size and abnormalities, including tumors and cysts, which are typically staged based on diameter, volume, and anatomical location. To support a more objective and reproducible approach, this research aims to develop a robust, thoroughly validated kidney abnormality segmentation algorithm, made publicly available for clinical and research use. We employ publicly available training datasets and leverage the state-of-the-art medical image segmentation framework nnU-Net. Validation is conducted using both proprietary and public test datasets, with segmentation performance quantified by Dice coefficient and the 95th percentile Hausdorff distance. Furthermore, we analyze robustness across subgroups based on patient sex, age, CT contrast phases, and tumor histologic subtypes. Our findings demonstrate that our segmentation algorithm, trained exclusively on publicly available data, generalizes effectively to external test sets and outperforms existing state-of-the-art models across all tested datasets. Subgroup analyses reveal consistent high performance, indicating strong robustness and reliability. The developed algorithm and associated code are publicly accessible at this https URL. 

---
# Towards Requirements Engineering for RAG Systems 

**Authors**: Tor Sporsem, Rasmus Ulfsnes  

**Link**: [PDF](https://arxiv.org/pdf/2505.07553)  

**Abstract**: This short paper explores how a maritime company develops and integrates large-language models (LLM). Specifically by looking at the requirements engineering for Retrieval Augmented Generation (RAG) systems in expert settings. Through a case study at a maritime service provider, we demonstrate how data scientists face a fundamental tension between user expectations of AI perfection and the correctness of the generated outputs. Our findings reveal that data scientists must identify context-specific "retrieval requirements" through iterative experimentation together with users because they are the ones who can determine correctness. We present an empirical process model describing how data scientists practically elicited these "retrieval requirements" and managed system limitations. This work advances software engineering knowledge by providing insights into the specialized requirements engineering processes for implementing RAG systems in complex domain-specific applications. 

---
# Automated Visual Attention Detection using Mobile Eye Tracking in Behavioral Classroom Studies 

**Authors**: Efe Bozkir, Christian Kosel, Tina Seidel, Enkelejda Kasneci  

**Link**: [PDF](https://arxiv.org/pdf/2505.07552)  

**Abstract**: Teachers' visual attention and its distribution across the students in classrooms can constitute important implications for student engagement, achievement, and professional teacher training. Despite that, inferring the information about where and which student teachers focus on is not trivial. Mobile eye tracking can provide vital help to solve this issue; however, the use of mobile eye tracking alone requires a significant amount of manual annotations. To address this limitation, we present an automated processing pipeline concept that requires minimal manually annotated data to recognize which student the teachers focus on. To this end, we utilize state-of-the-art face detection models and face recognition feature embeddings to train face recognition models with transfer learning in the classroom context and combine these models with the teachers' gaze from mobile eye trackers. We evaluated our approach with data collected from four different classrooms, and our results show that while it is possible to estimate the visually focused students with reasonable performance in all of our classroom setups, U-shaped and small classrooms led to the best results with accuracies of approximately 0.7 and 0.9, respectively. While we did not evaluate our method for teacher-student interactions and focused on the validity of the technical approach, as our methodology does not require a vast amount of manually annotated data and offers a non-intrusive way of handling teachers' visual attention, it could help improve instructional strategies, enhance classroom management, and provide feedback for professional teacher development. 

---
# Noise Optimized Conditional Diffusion for Domain Adaptation 

**Authors**: Lingkun Luo, Shiqiang Hu, Liming Chen  

**Link**: [PDF](https://arxiv.org/pdf/2505.07548)  

**Abstract**: Pseudo-labeling is a cornerstone of Unsupervised Domain Adaptation (UDA), yet the scarcity of High-Confidence Pseudo-Labeled Target Domain Samples (\textbf{hcpl-tds}) often leads to inaccurate cross-domain statistical alignment, causing DA failures. To address this challenge, we propose \textbf{N}oise \textbf{O}ptimized \textbf{C}onditional \textbf{D}iffusion for \textbf{D}omain \textbf{A}daptation (\textbf{NOCDDA}), which seamlessly integrates the generative capabilities of conditional diffusion models with the decision-making requirements of DA to achieve task-coupled optimization for efficient adaptation. For robust cross-domain consistency, we modify the DA classifier to align with the conditional diffusion classifier within a unified optimization framework, enabling forward training on noise-varying cross-domain samples. Furthermore, we argue that the conventional \( \mathcal{N}(\mathbf{0}, \mathbf{I}) \) initialization in diffusion models often generates class-confused hcpl-tds, compromising discriminative DA. To resolve this, we introduce a class-aware noise optimization strategy that refines sampling regions for reverse class-specific hcpl-tds generation, effectively enhancing cross-domain alignment. Extensive experiments across 5 benchmark datasets and 29 DA tasks demonstrate significant performance gains of \textbf{NOCDDA} over 31 state-of-the-art methods, validating its robustness and effectiveness. 

---
# GRADA: Graph-based Reranker against Adversarial Documents Attack 

**Authors**: Jingjie Zheng, Aryo Pradipta Gema, Giwon Hong, Xuanli He, Pasquale Minervini, Youcheng Sun, Qiongkai Xu  

**Link**: [PDF](https://arxiv.org/pdf/2505.07546)  

**Abstract**: Retrieval Augmented Generation (RAG) frameworks improve the accuracy of large language models (LLMs) by integrating external knowledge from retrieved documents, thereby overcoming the limitations of models' static intrinsic knowledge. However, these systems are susceptible to adversarial attacks that manipulate the retrieval process by introducing documents that are adversarial yet semantically similar to the query. Notably, while these adversarial documents resemble the query, they exhibit weak similarity to benign documents in the retrieval set. Thus, we propose a simple yet effective Graph-based Reranking against Adversarial Document Attacks (GRADA) framework aiming at preserving retrieval quality while significantly reducing the success of adversaries. Our study evaluates the effectiveness of our approach through experiments conducted on five LLMs: GPT-3.5-Turbo, GPT-4o, Llama3.1-8b, Llama3.1-70b, and Qwen2.5-7b. We use three datasets to assess performance, with results from the Natural Questions dataset demonstrating up to an 80% reduction in attack success rates while maintaining minimal loss in accuracy. 

---
# The Human-Data-Model Interaction Canvas for Visual Analytics 

**Authors**: Jürgen Bernard  

**Link**: [PDF](https://arxiv.org/pdf/2505.07534)  

**Abstract**: Visual Analytics (VA) integrates humans, data, and models as key actors in insight generation and data-driven decision-making. This position paper values and reflects on 16 VA process models and frameworks and makes nine high-level observations that motivate a fresh perspective on VA. The contribution is the HDMI Canvas, a perspective to VA that complements the strengths of existing VA process models and frameworks. It systematically characterizes diverse roles of humans, data, and models, and how these actors benefit from and contribute to VA processes. The descriptive power of the HDMI Canvas eases the differentiation between a series of VA building blocks, rather than describing general VA principles only. The canvas includes modern human-centered methodologies, including human knowledge externalization and forms of feedback loops, while interpretable and explainable AI highlight model contributions beyond their conventional outputs. The HDMI Canvas has generative power, guiding the design of new VA processes and is optimized for external stakeholders, improving VA outreach, interdisciplinary collaboration, and user-centered design. The utility of the HDMI Canvas is demonstrated through two preliminary case studies. 

---
# IKrNet: A Neural Network for Detecting Specific Drug-Induced Patterns in Electrocardiograms Amidst Physiological Variability 

**Authors**: Ahmad Fall, Federica Granese, Alex Lence, Dominique Fourer, Blaise Hanczar, Joe-Elie Salem, Jean-Daniel Zucker, Edi Prifti  

**Link**: [PDF](https://arxiv.org/pdf/2505.07533)  

**Abstract**: Monitoring and analyzing electrocardiogram (ECG) signals, even under varying physiological conditions, including those influenced by physical activity, drugs and stress, is crucial to accurately assess cardiac health. However, current AI-based methods often fail to account for how these factors interact and alter ECG patterns, ultimately limiting their applicability in real-world settings. This study introduces IKrNet, a novel neural network model, which identifies drug-specific patterns in ECGs amidst certain physiological conditions. IKrNet's architecture incorporates spatial and temporal dynamics by using a convolutional backbone with varying receptive field size to capture spatial features. A bi-directional Long Short-Term Memory module is also employed to model temporal dependencies. By treating heart rate variability as a surrogate for physiological fluctuations, we evaluated IKrNet's performance across diverse scenarios, including conditions with physical stress, drug intake alone, and a baseline without drug presence. Our assessment follows a clinical protocol in which 990 healthy volunteers were administered 80mg of Sotalol, a drug which is known to be a precursor to Torsades-de-Pointes, a life-threatening arrhythmia. We show that IKrNet outperforms state-of-the-art models' accuracy and stability in varying physiological conditions, underscoring its clinical viability. 

---
# ToolACE-DEV: Self-Improving Tool Learning via Decomposition and EVolution 

**Authors**: Xu Huang, Weiwen Liu, Xingshan Zeng, Yuefeng Huang, Xinlong Hao, Yuxian Wang, Yirong Zeng, Chuhan Wu, Yasheng Wang, Ruiming Tang, Defu Lian  

**Link**: [PDF](https://arxiv.org/pdf/2505.07512)  

**Abstract**: The tool-using capability of large language models (LLMs) enables them to access up-to-date external information and handle complex tasks. Current approaches to enhancing this capability primarily rely on distilling advanced models by data synthesis. However, this method incurs significant costs associated with advanced model usage and often results in data compatibility issues, led by the high discrepancy in the knowledge scope between the advanced model and the target model. To address these challenges, we propose ToolACE-DEV, a self-improving framework for tool learning. First, we decompose the tool-learning objective into sub-tasks that enhance basic tool-making and tool-using abilities. Then, we introduce a self-evolving paradigm that allows lightweight models to self-improve, reducing reliance on advanced LLMs. Extensive experiments validate the effectiveness of our approach across models of varying scales and architectures. 

---
# MAIS: Memory-Attention for Interactive Segmentation 

**Authors**: Mauricio Orbes-Arteaga, Oeslle Lucena, Sabastien Ourselin, M. Jorge Cardoso  

**Link**: [PDF](https://arxiv.org/pdf/2505.07511)  

**Abstract**: Interactive medical segmentation reduces annotation effort by refining predictions through user feedback. Vision Transformer (ViT)-based models, such as the Segment Anything Model (SAM), achieve state-of-the-art performance using user clicks and prior masks as prompts. However, existing methods treat interactions as independent events, leading to redundant corrections and limited refinement gains. We address this by introducing MAIS, a Memory-Attention mechanism for Interactive Segmentation that stores past user inputs and segmentation states, enabling temporal context integration. Our approach enhances ViT-based segmentation across diverse imaging modalities, achieving more efficient and accurate refinements. 

---
# EAGLE: Contrastive Learning for Efficient Graph Anomaly Detection 

**Authors**: Jing Ren, Mingliang Hou, Zhixuan Liu, Xiaomei Bai  

**Link**: [PDF](https://arxiv.org/pdf/2505.07508)  

**Abstract**: Graph anomaly detection is a popular and vital task in various real-world scenarios, which has been studied for several decades. Recently, many studies extending deep learning-based methods have shown preferable performance on graph anomaly detection. However, existing methods are lack of efficiency that is definitely necessary for embedded devices. Towards this end, we propose an Efficient Anomaly detection model on heterogeneous Graphs via contrastive LEarning (EAGLE) by contrasting abnormal nodes with normal ones in terms of their distances to the local context. The proposed method first samples instance pairs on meta path-level for contrastive learning. Then, a graph autoencoder-based model is applied to learn informative node embeddings in an unsupervised way, which will be further combined with the discriminator to predict the anomaly scores of nodes. Experimental results show that EAGLE outperforms the state-of-the-art methods on three heterogeneous network datasets. 

---
# Can Generative AI agents behave like humans? Evidence from laboratory market experiments 

**Authors**: R. Maria del Rio-Chanona, Marco Pangallo, Cars Hommes  

**Link**: [PDF](https://arxiv.org/pdf/2505.07457)  

**Abstract**: We explore the potential of Large Language Models (LLMs) to replicate human behavior in economic market experiments. Compared to previous studies, we focus on dynamic feedback between LLM agents: the decisions of each LLM impact the market price at the current step, and so affect the decisions of the other LLMs at the next step. We compare LLM behavior to market dynamics observed in laboratory settings and assess their alignment with human participants' behavior. Our findings indicate that LLMs do not adhere strictly to rational expectations, displaying instead bounded rationality, similarly to human participants. Providing a minimal context window i.e. memory of three previous time steps, combined with a high variability setting capturing response heterogeneity, allows LLMs to replicate broad trends seen in human experiments, such as the distinction between positive and negative feedback markets. However, differences remain at a granular level--LLMs exhibit less heterogeneity in behavior than humans. These results suggest that LLMs hold promise as tools for simulating realistic human behavior in economic contexts, though further research is needed to refine their accuracy and increase behavioral diversity. 

---
# Prototype Augmented Hypernetworks for Continual Learning 

**Authors**: Neil De La Fuente, Maria Pilligua, Daniel Vidal, Albin Soutiff, Cecilia Curreli, Daniel Cremers, Andrey Barsky  

**Link**: [PDF](https://arxiv.org/pdf/2505.07450)  

**Abstract**: Continual learning (CL) aims to learn a sequence of tasks without forgetting prior knowledge, but gradient updates for a new task often overwrite the weights learned earlier, causing catastrophic forgetting (CF). We propose Prototype-Augmented Hypernetworks (PAH), a framework where a single hypernetwork, conditioned on learnable task prototypes, dynamically generates task-specific classifier heads on demand. To mitigate forgetting, PAH combines cross-entropy with dual distillation losses, one to align logits and another to align prototypes, ensuring stable feature representations across tasks. Evaluations on Split-CIFAR100 and TinyImageNet demonstrate that PAH achieves state-of-the-art performance, reaching 74.5 % and 63.7 % accuracy with only 1.7 % and 4.4 % forgetting, respectively, surpassing prior methods without storing samples or heads. 

---
# Unified Continuous Generative Models 

**Authors**: Peng Sun, Yi Jiang, Tao Lin  

**Link**: [PDF](https://arxiv.org/pdf/2505.07447)  

**Abstract**: Recent advances in continuous generative models, including multi-step approaches like diffusion and flow-matching (typically requiring 8-1000 sampling steps) and few-step methods such as consistency models (typically 1-8 steps), have demonstrated impressive generative performance. However, existing work often treats these approaches as distinct paradigms, resulting in separate training and sampling methodologies. We introduce a unified framework for training, sampling, and analyzing these models. Our implementation, the Unified Continuous Generative Models Trainer and Sampler (UCGM-{T,S}), achieves state-of-the-art (SOTA) performance. For example, on ImageNet 256x256 using a 675M diffusion transformer, UCGM-T trains a multi-step model achieving 1.30 FID in 20 steps and a few-step model reaching 1.42 FID in just 2 steps. Additionally, applying UCGM-S to a pre-trained model (previously 1.26 FID at 250 steps) improves performance to 1.06 FID in only 40 steps. Code is available at: this https URL. 

---
# LEAD: Iterative Data Selection for Efficient LLM Instruction Tuning 

**Authors**: Xiaotian Lin, Yanlin Qi, Yizhang Zhu, Themis Palpanas, Chengliang Chai, Nan Tang, Yuyu Luo  

**Link**: [PDF](https://arxiv.org/pdf/2505.07437)  

**Abstract**: Instruction tuning has emerged as a critical paradigm for improving the capabilities and alignment of large language models (LLMs). However, existing iterative model-aware data selection methods incur significant computational overhead, as they rely on repeatedly performing full-dataset model inference to estimate sample utility for subsequent training iterations, creating a fundamental efficiency bottleneck. In this paper, we propose LEAD, an efficient iterative data selection framework that accurately estimates sample utility entirely within the standard training loop, eliminating the need for costly additional model inference. At its core, LEAD introduces Instance-Level Dynamic Uncertainty (IDU), a theoretically grounded utility function combining instantaneous training loss, gradient-based approximation of loss changes, and exponential smoothing of historical loss signals. To further scale efficiently to large datasets, LEAD employs a two-stage, coarse-to-fine selection strategy, adaptively prioritizing informative clusters through a multi-armed bandit mechanism, followed by precise fine-grained selection of high-utility samples using IDU. Extensive experiments across four diverse benchmarks show that LEAD significantly outperforms state-of-the-art methods, improving average model performance by 6.1%-10.8% while using only 2.5% of the training data and reducing overall training time by 5-10x. 

---
# AI in Money Matters 

**Authors**: Nadine Sandjo Tchatchoua, Richard Harper  

**Link**: [PDF](https://arxiv.org/pdf/2505.07393)  

**Abstract**: In November 2022, Europe and the world by and large were stunned by the birth of a new large language model : ChatGPT. Ever since then, both academic and populist discussions have taken place in various public spheres such as LinkedIn and X(formerly known as Twitter) with the view to both understand the tool and its benefits for the society. The views of real actors in professional spaces, especially in regulated industries such as finance and law have been largely missing. We aim to begin to close this gap by presenting results from an empirical investigation conducted through interviews with professional actors in the Fintech industry. The paper asks the question, how and to what extent are large language models in general and ChatGPT in particular being adopted and used in the Fintech industry? The results show that while the fintech experts we spoke with see a potential in using large language models in the future, a lot of questions marks remain concerning how they are policed and therefore might be adopted in a regulated industry such as Fintech. This paper aims to add to the existing academic discussing around large language models, with a contribution to our understanding of professional viewpoints. 

---
# Few-shot Semantic Encoding and Decoding for Video Surveillance 

**Authors**: Baoping Cheng, Yukun Zhang, Liming Wang, Xiaoyan Xie, Tao Fu, Dongkun Wang, Xiaoming Tao  

**Link**: [PDF](https://arxiv.org/pdf/2505.07381)  

**Abstract**: With the continuous increase in the number and resolution of video surveillance cameras, the burden of transmitting and storing surveillance video is growing. Traditional communication methods based on Shannon's theory are facing optimization bottlenecks. Semantic communication, as an emerging communication method, is expected to break through this bottleneck and reduce the storage and transmission consumption of video. Existing semantic decoding methods often require many samples to train the neural network for each scene, which is time-consuming and labor-intensive. In this study, a semantic encoding and decoding method for surveillance video is proposed. First, the sketch was extracted as semantic information, and a sketch compression method was proposed to reduce the bit rate of semantic information. Then, an image translation network was proposed to translate the sketch into a video frame with a reference frame. Finally, a few-shot sketch decoding network was proposed to reconstruct video from sketch. Experimental results showed that the proposed method achieved significantly better video reconstruction performance than baseline methods. The sketch compression method could effectively reduce the storage and transmission consumption of semantic information with little compromise on video quality. The proposed method provides a novel semantic encoding and decoding method that only needs a few training samples for each surveillance scene, thus improving the practicality of the semantic communication system. 

---
# Examining the Role of LLM-Driven Interactions on Attention and Cognitive Engagement in Virtual Classrooms 

**Authors**: Suleyman Ozdel, Can Sarpkaya, Efe Bozkir, Hong Gao, Enkelejda Kasneci  

**Link**: [PDF](https://arxiv.org/pdf/2505.07377)  

**Abstract**: Transforming educational technologies through the integration of large language models (LLMs) and virtual reality (VR) offers the potential for immersive and interactive learning experiences. However, the effects of LLMs on user engagement and attention in educational environments remain open questions. In this study, we utilized a fully LLM-driven virtual learning environment, where peers and teachers were LLM-driven, to examine how students behaved in such settings. Specifically, we investigate how peer question-asking behaviors influenced student engagement, attention, cognitive load, and learning outcomes and found that, in conditions where LLM-driven peer learners asked questions, students exhibited more targeted visual scanpaths, with their attention directed toward the learning content, particularly in complex subjects. Our results suggest that peer questions did not introduce extraneous cognitive load directly, as the cognitive load is strongly correlated with increased attention to the learning material. Considering these findings, we provide design recommendations for optimizing VR learning spaces. 

---
# Synthetic Code Surgery: Repairing Bugs and Vulnerabilities with LLMs and Synthetic Data 

**Authors**: David de-Fitero-Dominguez, Antonio Garcia-Cabot, Eva Garcia-Lopez  

**Link**: [PDF](https://arxiv.org/pdf/2505.07372)  

**Abstract**: This paper presents a novel methodology for enhancing Automated Program Repair (APR) through synthetic data generation utilizing Large Language Models (LLMs). Current APR systems are constrained by the limited availability of high-quality training data encompassing diverse bug types across multiple programming languages. The proposed approach addresses this limitation through a two-phase process: a synthetic sample generation followed by a rigorous quality assessment. Multiple state-of-the-art LLMs were employed to generate approximately 30,000 paired examples of buggy and fixed code across 12 programming languages and 13 bug categories. Subsequently, these samples underwent cross-model evaluation against five criteria: correctness, code quality, security, performance, and completeness. Experimental evaluation on the VulRepair test set dataset showed statistically significant improvements in Perfect Prediction rates, with the quality-filtered synthetic dataset outperforming both baseline and real-world commit data configurations in certain scenarios. The methodology was validated through rigorous statistical testing, including ANOVA and post-hoc Tukey's Honest Significant Difference analysis. Furthermore, the best-performing configurations surpassed existing systems despite using a less computationally intensive decoding strategy. This research establishes a self-bootstrapping paradigm in which LLMs generate and evaluate their own training data, potentially transforming approaches to data scarcity across software engineering tasks and advancing the development of robust, adaptable tools for automated code maintenance. 

---
# Multi-Domain Audio Question Answering Toward Acoustic Content Reasoning in The DCASE 2025 Challenge 

**Authors**: Chao-Han Huck Yang, Sreyan Ghosh, Qing Wang, Jaeyeon Kim, Hengyi Hong, Sonal Kumar, Guirui Zhong, Zhifeng Kong, S Sakshi, Vaibhavi Lokegaonkar, Oriol Nieto, Ramani Duraiswami, Dinesh Manocha, Gunhee Kim, Jun Du, Rafael Valle, Bryan Catanzaro  

**Link**: [PDF](https://arxiv.org/pdf/2505.07365)  

**Abstract**: We present Task 5 of the DCASE 2025 Challenge: an Audio Question Answering (AQA) benchmark spanning multiple domains of sound understanding. This task defines three QA subsets (Bioacoustics, Temporal Soundscapes, and Complex QA) to test audio-language models on interactive question-answering over diverse acoustic scenes. We describe the dataset composition (from marine mammal calls to soundscapes and complex real-world clips), the evaluation protocol (top-1 accuracy with answer-shuffling robustness), and baseline systems (Qwen2-Audio-7B, AudioFlamingo 2, Gemini-2-Flash). Preliminary results on the development set are compared, showing strong variation across models and subsets. This challenge aims to advance the audio understanding and reasoning capabilities of audio-language models toward human-level acuity, which are crucial for enabling AI agents to perceive and interact about the world effectively. 

---
# GAN-based synthetic FDG PET images from T1 brain MRI can serve to improve performance of deep unsupervised anomaly detection models 

**Authors**: Daria Zotova, Nicolas Pinon, Robin Trombetta, Romain Bouet, Julien Jung, Carole Lartizien  

**Link**: [PDF](https://arxiv.org/pdf/2505.07364)  

**Abstract**: Background and Objective. Research in the cross-modal medical image translation domain has been very productive over the past few years in tackling the scarce availability of large curated multimodality datasets with the promising performance of GAN-based architectures. However, only a few of these studies assessed task-based related performance of these synthetic data, especially for the training of deep models. Method. We design and compare different GAN-based frameworks for generating synthetic brain [18F]fluorodeoxyglucose (FDG) PET images from T1 weighted MRI data. We first perform standard qualitative and quantitative visual quality evaluation. Then, we explore further impact of using these fake PET data in the training of a deep unsupervised anomaly detection (UAD) model designed to detect subtle epilepsy lesions in T1 MRI and FDG PET images. We introduce novel diagnostic task-oriented quality metrics of the synthetic FDG PET data tailored to our unsupervised detection task, then use these fake data to train a use case UAD model combining a deep representation learning based on siamese autoencoders with a OC-SVM density support estimation model. This model is trained on normal subjects only and allows the detection of any variation from the pattern of the normal population. We compare the detection performance of models trained on 35 paired real MR T1 of normal subjects paired either on 35 true PET images or on 35 synthetic PET images generated from the best performing generative models. Performance analysis is conducted on 17 exams of epilepsy patients undergoing surgery. Results. The best performing GAN-based models allow generating realistic fake PET images of control subject with SSIM and PSNR values around 0.9 and 23.8, respectively and in distribution (ID) with regard to the true control dataset. The best UAD model trained on these synthetic normative PET data allows reaching 74% sensitivity. Conclusion. Our results confirm that GAN-based models are the best suited for MR T1 to FDG PET translation, outperforming transformer or diffusion models. We also demonstrate the diagnostic value of these synthetic data for the training of UAD models and evaluation on clinical exams of epilepsy patients. Our code and the normative image dataset are available. 

---
# QUPID: Quantified Understanding for Enhanced Performance, Insights, and Decisions in Korean Search Engines 

**Authors**: Ohjoon Kwon, Changsu Lee, Jihye Back, Lim Sun Suk, Inho Kang, Donghyeon Jeon  

**Link**: [PDF](https://arxiv.org/pdf/2505.07345)  

**Abstract**: Large language models (LLMs) have been widely used for relevance assessment in information retrieval. However, our study demonstrates that combining two distinct small language models (SLMs) with different architectures can outperform LLMs in this task. Our approach -- QUPID -- integrates a generative SLM with an embedding-based SLM, achieving higher relevance judgment accuracy while reducing computational costs compared to state-of-the-art LLM solutions. This computational efficiency makes QUPID highly scalable for real-world search systems processing millions of queries daily. In experiments across diverse document types, our method demonstrated consistent performance improvements (Cohen's Kappa of 0.646 versus 0.387 for leading LLMs) while offering 60x faster inference times. Furthermore, when integrated into production search pipelines, QUPID improved nDCG@5 scores by 1.9%. These findings underscore how architectural diversity in model combinations can significantly enhance both search relevance and operational efficiency in information retrieval systems. 

---
# Generative Pre-trained Autoregressive Diffusion Transformer 

**Authors**: Yuan Zhang, Jiacheng Jiang, Guoqing Ma, Zhiying Lu, Haoyang Huang, Jianlong Yuan, Nan Duan  

**Link**: [PDF](https://arxiv.org/pdf/2505.07344)  

**Abstract**: In this work, we present GPDiT, a Generative Pre-trained Autoregressive Diffusion Transformer that unifies the strengths of diffusion and autoregressive modeling for long-range video synthesis, within a continuous latent space. Instead of predicting discrete tokens, GPDiT autoregressively predicts future latent frames using a diffusion loss, enabling natural modeling of motion dynamics and semantic consistency across frames. This continuous autoregressive framework not only enhances generation quality but also endows the model with representation capabilities. Additionally, we introduce a lightweight causal attention variant and a parameter-free rotation-based time-conditioning mechanism, improving both the training and inference efficiency. Extensive experiments demonstrate that GPDiT achieves strong performance in video generation quality, video representation ability, and few-shot learning tasks, highlighting its potential as an effective framework for video modeling in continuous space. 

---
# Laypeople's Attitudes Towards Fair, Affirmative, and Discriminatory Decision-Making Algorithms 

**Authors**: Gabriel Lima, Nina Grgić-Hlača, Markus Langer, Yixin Zou  

**Link**: [PDF](https://arxiv.org/pdf/2505.07339)  

**Abstract**: Affirmative algorithms have emerged as a potential answer to algorithmic discrimination, seeking to redress past harms and rectify the source of historical injustices. We present the results of two experiments ($N$$=$$1193$) capturing laypeople's perceptions of affirmative algorithms -- those which explicitly prioritize the historically marginalized -- in hiring and criminal justice. We contrast these opinions about affirmative algorithms with folk attitudes towards algorithms that prioritize the privileged (i.e., discriminatory) and systems that make decisions independently of demographic groups (i.e., fair). We find that people -- regardless of their political leaning and identity -- view fair algorithms favorably and denounce discriminatory systems. In contrast, we identify disagreements concerning affirmative algorithms: liberals and racial minorities rate affirmative systems as positively as their fair counterparts, whereas conservatives and those from the dominant racial group evaluate affirmative algorithms as negatively as discriminatory systems. We identify a source of these divisions: people have varying beliefs about who (if anyone) is marginalized, shaping their views of affirmative algorithms. We discuss the possibility of bridging these disagreements to bring people together towards affirmative algorithms. 

---
# SAEN-BGS: Energy-Efficient Spiking AutoEncoder Network for Background Subtraction 

**Authors**: Zhixuan Zhang, Xiaopeng Li, Qi Liu  

**Link**: [PDF](https://arxiv.org/pdf/2505.07336)  

**Abstract**: Background subtraction (BGS) is utilized to detect moving objects in a video and is commonly employed at the onset of object tracking and human recognition processes. Nevertheless, existing BGS techniques utilizing deep learning still encounter challenges with various background noises in videos, including variations in lighting, shifts in camera angles, and disturbances like air turbulence or swaying trees. To address this problem, we design a spiking autoencoder network, termed SAEN-BGS, based on noise resilience and time-sequence sensitivity of spiking neural networks (SNNs) to enhance the separation of foreground and background. To eliminate unnecessary background noise and preserve the important foreground elements, we begin by creating the continuous spiking conv-and-dconv block, which serves as the fundamental building block for the decoder in SAEN-BGS. Moreover, in striving for enhanced energy efficiency, we introduce a novel self-distillation spiking supervised learning method grounded in ANN-to-SNN frameworks, resulting in decreased power consumption. In extensive experiments conducted on CDnet-2014 and DAVIS-2016 datasets, our approach demonstrates superior segmentation performance relative to other baseline methods, even when challenged by complex scenarios with dynamic backgrounds. 

---
# Dynamical Label Augmentation and Calibration for Noisy Electronic Health Records 

**Authors**: Yuhao Li, Ling Luo, Uwe Aickelin  

**Link**: [PDF](https://arxiv.org/pdf/2505.07320)  

**Abstract**: Medical research, particularly in predicting patient outcomes, heavily relies on medical time series data extracted from Electronic Health Records (EHR), which provide extensive information on patient histories. Despite rigorous examination, labeling errors are inevitable and can significantly impede accurate predictions of patient outcome. To address this challenge, we propose an \textbf{A}ttention-based Learning Framework with Dynamic \textbf{C}alibration and Augmentation for \textbf{T}ime series Noisy \textbf{L}abel \textbf{L}earning (ACTLL). This framework leverages a two-component Beta mixture model to identify the certain and uncertain sets of instances based on the fitness distribution of each class, and it captures global temporal dynamics while dynamically calibrating labels from the uncertain set or augmenting confident instances from the certain set. Experimental results on large-scale EHR datasets eICU and MIMIC-IV-ED, and several benchmark datasets from the UCR and UEA repositories, demonstrate that our model ACTLL has achieved state-of-the-art performance, especially under high noise levels. 

---
# How Do Companies Manage the Environmental Sustainability of AI? An Interview Study About Green AI Efforts and Regulations 

**Authors**: Ashmita Sampatsing, Sophie Vos, Emma Beauxis-Aussalet, Justus Bogner  

**Link**: [PDF](https://arxiv.org/pdf/2505.07317)  

**Abstract**: With the ever-growing adoption of artificial intelligence (AI), AI-based software and its negative impact on the environment are no longer negligible, and studying and mitigating this impact has become a critical area of research. However, it is currently unclear which role environmental sustainability plays during AI adoption in industry and how AI regulations influence Green AI practices and decision-making in industry. We therefore aim to investigate the Green AI perception and management of industry practitioners. To this end, we conducted a total of 11 interviews with participants from 10 different organizations that adopted AI-based software. The interviews explored three main themes: AI adoption, current efforts in mitigating the negative environmental impact of AI, and the influence of the EU AI Act and the Corporate Sustainability Reporting Directive (CSRD). Our findings indicate that 9 of 11 participants prioritized business efficiency during AI adoption, with minimal consideration of environmental sustainability. Monitoring and mitigation of AI's environmental impact were very limited. Only one participant monitored negative environmental effects. Regarding applied mitigation practices, six participants reported no actions, with the others sporadically mentioning techniques like prompt engineering, relying on smaller models, or not overusing AI. Awareness and compliance with the EU AI Act are low, with only one participant reporting on its influence, while the CSRD drove sustainability reporting efforts primarily in larger companies. All in all, our findings reflect a lack of urgency and priority for sustainable AI among these companies. We suggest that current regulations are not very effective, which has implications for policymakers. Additionally, there is a need to raise industry awareness, but also to provide user-friendly techniques and tools for Green AI practices. 

---
# Towards Multi-Agent Reasoning Systems for Collaborative Expertise Delegation: An Exploratory Design Study 

**Authors**: Baixuan Xu, Chunyang Li, Weiqi Wang, Wei Fan, Tianshi Zheng, Haochen Shi, Tao Fan, Yangqiu Song, Qiang Yang  

**Link**: [PDF](https://arxiv.org/pdf/2505.07313)  

**Abstract**: Designing effective collaboration structure for multi-agent LLM systems to enhance collective reasoning is crucial yet remains under-explored. In this paper, we systematically investigate how collaborative reasoning performance is affected by three key design dimensions: (1) Expertise-Domain Alignment, (2) Collaboration Paradigm (structured workflow vs. diversity-driven integration), and (3) System Scale. Our findings reveal that expertise alignment benefits are highly domain-contingent, proving most effective for contextual reasoning tasks. Furthermore, collaboration focused on integrating diverse knowledge consistently outperforms rigid task decomposition. Finally, we empirically explore the impact of scaling the multi-agent system with expertise specialization and study the computational trade off, highlighting the need for more efficient communication protocol design. This work provides concrete guidelines for configuring specialized multi-agent system and identifies critical architectural trade-offs and bottlenecks for scalable multi-agent reasoning. The code will be made available upon acceptance. 

---
# HuB: Learning Extreme Humanoid Balance 

**Authors**: Tong Zhang, Boyuan Zheng, Ruiqian Nai, Yingdong Hu, Yen-Jen Wang, Geng Chen, Fanqi Lin, Jiongye Li, Chuye Hong, Koushil Sreenath, Yang Gao  

**Link**: [PDF](https://arxiv.org/pdf/2505.07294)  

**Abstract**: The human body demonstrates exceptional motor capabilities-such as standing steadily on one foot or performing a high kick with the leg raised over 1.5 meters-both requiring precise balance control. While recent research on humanoid control has leveraged reinforcement learning to track human motions for skill acquisition, applying this paradigm to balance-intensive tasks remains challenging. In this work, we identify three key obstacles: instability from reference motion errors, learning difficulties due to morphological mismatch, and the sim-to-real gap caused by sensor noise and unmodeled dynamics. To address these challenges, we propose HuB (Humanoid Balance), a unified framework that integrates reference motion refinement, balance-aware policy learning, and sim-to-real robustness training, with each component targeting a specific challenge. We validate our approach on the Unitree G1 humanoid robot across challenging quasi-static balance tasks, including extreme single-legged poses such as Swallow Balance and Bruce Lee's Kick. Our policy remains stable even under strong physical disturbances-such as a forceful soccer strike-while baseline methods consistently fail to complete these tasks. Project website: this https URL 

---
# Semantic Retention and Extreme Compression in LLMs: Can We Have Both? 

**Authors**: Stanislas Laborde, Martin Cousseau, Antoun Yaacoub, Lionel Prevost  

**Link**: [PDF](https://arxiv.org/pdf/2505.07289)  

**Abstract**: The exponential growth in Large Language Model (LLM) deployment has intensified the need for efficient model compression techniques to reduce computational and memory costs. While pruning and quantization have shown promise, their combined potential remains largely unexplored. In this paper, we examine joint compression and how strategically combining pruning and quantization could yield superior performance-to-compression ratios compared to single-method approaches. Recognizing the challenges in accurately assessing LLM performance, we address key limitations of previous evaluation frameworks and introduce the Semantic Retention Compression Rate (SrCr), a novel metric that quantifies the trade-off between model compression and semantic preservation, facilitating the optimization of pruning-quantization configurations. Experiments demonstrate that our recommended combination achieves, on average, a 20% performance increase compared to an equivalent quantization-only model at the same theoretical compression rate. 

---
# Piloting Structure-Based Drug Design via Modality-Specific Optimal Schedule 

**Authors**: Keyue Qiu, Yuxuan Song, Zhehuan Fan, Peidong Liu, Zhe Zhang, Mingyue Zheng, Hao Zhou, Wei-Ying Ma  

**Link**: [PDF](https://arxiv.org/pdf/2505.07286)  

**Abstract**: Structure-Based Drug Design (SBDD) is crucial for identifying bioactive molecules. Recent deep generative models are faced with challenges in geometric structure modeling. A major bottleneck lies in the twisted probability path of multi-modalities -- continuous 3D positions and discrete 2D topologies -- which jointly determine molecular geometries. By establishing the fact that noise schedules decide the Variational Lower Bound (VLB) for the twisted probability path, we propose VLB-Optimal Scheduling (VOS) strategy in this under-explored area, which optimizes VLB as a path integral for SBDD. Our model effectively enhances molecular geometries and interaction modeling, achieving state-of-the-art PoseBusters passing rate of 95.9% on CrossDock, more than 10% improvement upon strong baselines, while maintaining high affinities and robust intramolecular validity evaluated on held-out test set. 

---
# Predicting Music Track Popularity by Convolutional Neural Networks on Spotify Features and Spectrogram of Audio Waveform 

**Authors**: Navid Falah, Behnam Yousefimehr, Mehdi Ghatee  

**Link**: [PDF](https://arxiv.org/pdf/2505.07280)  

**Abstract**: In the digital streaming landscape, it's becoming increasingly challenging for artists and industry experts to predict the success of music tracks. This study introduces a pioneering methodology that uses Convolutional Neural Networks (CNNs) and Spotify data analysis to forecast the popularity of music tracks. Our approach takes advantage of Spotify's wide range of features, including acoustic attributes based on the spectrogram of audio waveform, metadata, and user engagement metrics, to capture the complex patterns and relationships that influence a track's popularity. Using a large dataset covering various genres and demographics, our CNN-based model shows impressive effectiveness in predicting the popularity of music tracks. Additionally, we've conducted extensive experiments to assess the strength and adaptability of our model across different musical styles and time periods, with promising results yielding a 97\% F1 score. Our study not only offers valuable insights into the dynamic landscape of digital music consumption but also provides the music industry with advanced predictive tools for assessing and predicting the success of music tracks. 

---
# On the Robustness of Reward Models for Language Model Alignment 

**Authors**: Jiwoo Hong, Noah Lee, Eunki Kim, Guijin Son, Woojin Chung, Aman Gupta, Shao Tang, James Thorne  

**Link**: [PDF](https://arxiv.org/pdf/2505.07271)  

**Abstract**: The Bradley-Terry (BT) model is widely practiced in reward modeling for reinforcement learning with human feedback (RLHF). Despite its effectiveness, reward models (RMs) trained with BT model loss are prone to over-optimization, losing generalizability to unseen input distributions. In this paper, we study the cause of over-optimization in RM training and its downstream effects on the RLHF procedure, accentuating the importance of distributional robustness of RMs in unseen data. First, we show that the excessive dispersion of hidden state norms is the main source of over-optimization. Then, we propose batch-wise sum-to-zero regularization (BSR) to enforce zero-centered reward sum per batch, constraining the rewards with extreme magnitudes. We assess the impact of BSR in improving robustness in RMs through four scenarios of over-optimization, where BSR consistently manifests better robustness. Subsequently, we compare the plain BT model and BSR on RLHF training and empirically show that robust RMs better align the policy to the gold preference model. Finally, we apply BSR to high-quality data and models, which surpasses state-of-the-art RMs in the 8B scale by adding more than 5% in complex preference prediction tasks. By conducting RLOO training with 8B RMs, AlpacaEval 2.0 reduces generation length by 40% while adding a 7% increase in win rate, further highlighting that robustness in RMs induces robustness in RLHF training. We release the code, data, and models: this https URL. 

---
# CHD: Coupled Hierarchical Diffusion for Long-Horizon Tasks 

**Authors**: Ce Hao, Anxing Xiao, Zhiwei Xue, Harold Soh  

**Link**: [PDF](https://arxiv.org/pdf/2505.07261)  

**Abstract**: Diffusion-based planners have shown strong performance in short-horizon tasks but often fail in complex, long-horizon settings. We trace the failure to loose coupling between high-level (HL) sub-goal selection and low-level (LL) trajectory generation, which leads to incoherent plans and degraded performance. We propose Coupled Hierarchical Diffusion (CHD), a framework that models HL sub-goals and LL trajectories jointly within a unified diffusion process. A shared classifier passes LL feedback upstream so that sub-goals self-correct while sampling proceeds. This tight HL-LL coupling improves trajectory coherence and enables scalable long-horizon diffusion planning. Experiments across maze navigation, tabletop manipulation, and household environments show that CHD consistently outperforms both flat and hierarchical diffusion baselines. 

---
# UMoE: Unifying Attention and FFN with Shared Experts 

**Authors**: Yuanhang Yang, Chaozheng Wang, Jing Li  

**Link**: [PDF](https://arxiv.org/pdf/2505.07260)  

**Abstract**: Sparse Mixture of Experts (MoE) architectures have emerged as a promising approach for scaling Transformer models. While initial works primarily incorporated MoE into feed-forward network (FFN) layers, recent studies have explored extending the MoE paradigm to attention layers to enhance model performance. However, existing attention-based MoE layers require specialized implementations and demonstrate suboptimal performance compared to their FFN-based counterparts. In this paper, we aim to unify the MoE designs in attention and FFN layers by introducing a novel reformulation of the attention mechanism, revealing an underlying FFN-like structure within attention modules. Our proposed architecture, UMoE, achieves superior performance through attention-based MoE layers while enabling efficient parameter sharing between FFN and attention components. 

---
# No Query, No Access 

**Authors**: Wenqiang Wang, Siyuan Liang, Yangshijie Zhang, Xiaojun Jia, Hao Lin, Xiaochun Cao  

**Link**: [PDF](https://arxiv.org/pdf/2505.07258)  

**Abstract**: Textual adversarial attacks mislead NLP models, including Large Language Models (LLMs), by subtly modifying text. While effective, existing attacks often require knowledge of the victim model, extensive queries, or access to training data, limiting real-world feasibility. To overcome these constraints, we introduce the \textbf{Victim Data-based Adversarial Attack (VDBA)}, which operates using only victim texts. To prevent access to the victim model, we create a shadow dataset with publicly available pre-trained models and clustering methods as a foundation for developing substitute models. To address the low attack success rate (ASR) due to insufficient information feedback, we propose the hierarchical substitution model design, generating substitute models to mitigate the failure of a single substitute model at the decision boundary.
Concurrently, we use diverse adversarial example generation, employing various attack methods to generate and select the adversarial example with better similarity and attack effectiveness. Experiments on the Emotion and SST5 datasets show that VDBA outperforms state-of-the-art methods, achieving an ASR improvement of 52.08\% while significantly reducing attack queries to 0. More importantly, we discover that VDBA poses a significant threat to LLMs such as Qwen2 and the GPT family, and achieves the highest ASR of 45.99% even without access to the API, confirming that advanced NLP models still face serious security risks. Our codes can be found at this https URL 

---
# Incomplete In-context Learning 

**Authors**: Wenqiang Wang, Yangshijie Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2505.07251)  

**Abstract**: Large vision language models (LVLMs) achieve remarkable performance through Vision In-context Learning (VICL), a process that depends significantly on demonstrations retrieved from an extensive collection of annotated examples (retrieval database). Existing studies often assume that the retrieval database contains annotated examples for all labels. However, in real-world scenarios, delays in database updates or incomplete data annotation may result in the retrieval database containing labeled samples for only a subset of classes. We refer to this phenomenon as an \textbf{incomplete retrieval database} and define the in-context learning under this condition as \textbf{Incomplete In-context Learning (IICL)}. To address this challenge, we propose \textbf{Iterative Judgments and Integrated Prediction (IJIP)}, a two-stage framework designed to mitigate the limitations of IICL. The Iterative Judgments Stage reformulates an \(\boldsymbol{m}\)-class classification problem into a series of \(\boldsymbol{m}\) binary classification tasks, effectively converting the IICL setting into a standard VICL scenario. The Integrated Prediction Stage further refines the classification process by leveraging both the input image and the predictions from the Iterative Judgments Stage to enhance overall classification accuracy. IJIP demonstrates considerable performance across two LVLMs and two datasets under three distinct conditions of label incompleteness, achieving the highest accuracy of 93.9\%. Notably, even in scenarios where labels are fully available, IJIP still achieves the best performance of all six baselines. Furthermore, IJIP can be directly applied to \textbf{Prompt Learning} and is adaptable to the \textbf{text domain}. 

---
# SAS-Bench: A Fine-Grained Benchmark for Evaluating Short Answer Scoring with Large Language Models 

**Authors**: Peichao Lai, Kexuan Zhang, Yi Lin, Linyihan Zhang, Feiyang Ye, Jinhao Yan, Yanwei Xu, Conghui He, Yilei Wang, Wentao Zhang, Bin Cui  

**Link**: [PDF](https://arxiv.org/pdf/2505.07247)  

**Abstract**: Subjective Answer Grading (SAG) plays a crucial role in education, standardized testing, and automated assessment systems, particularly for evaluating short-form responses in Short Answer Scoring (SAS). However, existing approaches often produce coarse-grained scores and lack detailed reasoning. Although large language models (LLMs) have demonstrated potential as zero-shot evaluators, they remain susceptible to bias, inconsistencies with human judgment, and limited transparency in scoring decisions. To overcome these limitations, we introduce SAS-Bench, a benchmark specifically designed for LLM-based SAS tasks. SAS-Bench provides fine-grained, step-wise scoring, expert-annotated error categories, and a diverse range of question types derived from real-world subject-specific exams. This benchmark facilitates detailed evaluation of model reasoning processes and explainability. We also release an open-source dataset containing 1,030 questions and 4,109 student responses, each annotated by domain experts. Furthermore, we conduct comprehensive experiments with various LLMs, identifying major challenges in scoring science-related questions and highlighting the effectiveness of few-shot prompting in improving scoring accuracy. Our work offers valuable insights into the development of more robust, fair, and educationally meaningful LLM-based evaluation systems. 

---
# REMEDI: Relative Feature Enhanced Meta-Learning with Distillation for Imbalanced Prediction 

**Authors**: Fei Liu, Huanhuan Ren, Yu Guan, Xiuxu Wang, Wang Lv, Zhiqiang Hu, Yaxi Chen  

**Link**: [PDF](https://arxiv.org/pdf/2505.07245)  

**Abstract**: Predicting future vehicle purchases among existing owners presents a critical challenge due to extreme class imbalance (<0.5% positive rate) and complex behavioral patterns. We propose REMEDI (Relative feature Enhanced Meta-learning with Distillation for Imbalanced prediction), a novel multi-stage framework addressing these challenges. REMEDI first trains diverse base models to capture complementary aspects of user behavior. Second, inspired by comparative op-timization techniques, we introduce relative performance meta-features (deviation from ensemble mean, rank among peers) for effective model fusion through a hybrid-expert architecture. Third, we distill the ensemble's knowledge into a single efficient model via supervised fine-tuning with MSE loss, enabling practical deployment. Evaluated on approximately 800,000 vehicle owners, REMEDI significantly outperforms baseline approaches, achieving the business target of identifying ~50% of actual buyers within the top 60,000 recommendations at ~10% precision. The distilled model preserves the ensemble's predictive power while maintaining deployment efficiency, demonstrating REMEDI's effectiveness for imbalanced prediction in industry settings. 

---
# Comet: Accelerating Private Inference for Large Language Model by Predicting Activation Sparsity 

**Authors**: Guang Yan, Yuhui Zhang, Zimu Guo, Lutan Zhao, Xiaojun Chen, Chen Wang, Wenhao Wang, Dan Meng, Rui Hou  

**Link**: [PDF](https://arxiv.org/pdf/2505.07239)  

**Abstract**: With the growing use of large language models (LLMs) hosted on cloud platforms to offer inference services, privacy concerns about the potential leakage of sensitive information are escalating. Secure multi-party computation (MPC) is a promising solution to protect the privacy in LLM inference. However, MPC requires frequent inter-server communication, causing high performance overhead.
Inspired by the prevalent activation sparsity of LLMs, where most neuron are not activated after non-linear activation functions, we propose an efficient private inference system, Comet. This system employs an accurate and fast predictor to predict the sparsity distribution of activation function output. Additionally, we introduce a new private inference protocol. It efficiently and securely avoids computations involving zero values by exploiting the spatial locality of the predicted sparse distribution. While this computation-avoidance approach impacts the spatiotemporal continuity of KV cache entries, we address this challenge with a low-communication overhead cache refilling strategy that merges miss requests and incorporates a prefetching mechanism. Finally, we evaluate Comet on four common LLMs and compare it with six state-of-the-art private inference systems. Comet achieves a 1.87x-2.63x speedup and a 1.94x-2.64x communication reduction. 

---
# UAV-CodeAgents: Scalable UAV Mission Planning via Multi-Agent ReAct and Vision-Language Reasoning 

**Authors**: Oleg Sautenkov, Yasheerah Yaqoot, Muhammad Ahsan Mustafa, Faryal Batool, Jeffrin Sam, Artem Lykov, Chih-Yung Wen, Dzmitry Tsetserukou  

**Link**: [PDF](https://arxiv.org/pdf/2505.07236)  

**Abstract**: We present UAV-CodeAgents, a scalable multi-agent framework for autonomous UAV mission generation, built on large language and vision-language models (LLMs/VLMs). The system leverages the ReAct (Reason + Act) paradigm to interpret satellite imagery, ground high-level natural language instructions, and collaboratively generate UAV trajectories with minimal human supervision. A core component is a vision-grounded, pixel-pointing mechanism that enables precise localization of semantic targets on aerial maps. To support real-time adaptability, we introduce a reactive thinking loop, allowing agents to iteratively reflect on observations, revise mission goals, and coordinate dynamically in evolving environments.
UAV-CodeAgents is evaluated on large-scale mission scenarios involving industrial and environmental fire detection. Our results show that a lower decoding temperature (0.5) yields higher planning reliability and reduced execution time, with an average mission creation time of 96.96 seconds and a success rate of 93%. We further fine-tune Qwen2.5VL-7B on 9,000 annotated satellite images, achieving strong spatial grounding across diverse visual categories. To foster reproducibility and future research, we will release the full codebase and a novel benchmark dataset for vision-language-based UAV planning. 

---
# DynamicRAG: Leveraging Outputs of Large Language Model as Feedback for Dynamic Reranking in Retrieval-Augmented Generation 

**Authors**: Jiashuo Sun, Xianrui Zhong, Sizhe Zhou, Jiawei Han  

**Link**: [PDF](https://arxiv.org/pdf/2505.07233)  

**Abstract**: Retrieval-augmented generation (RAG) systems combine large language models (LLMs) with external knowledge retrieval, making them highly effective for knowledge-intensive tasks. A crucial but often under-explored component of these systems is the reranker, which refines retrieved documents to enhance generation quality and explainability. The challenge of selecting the optimal number of documents (k) remains unsolved: too few may omit critical information, while too many introduce noise and inefficiencies. Although recent studies have explored LLM-based rerankers, they primarily leverage internal model knowledge and overlook the rich supervisory signals that LLMs can provide, such as using response quality as feedback for optimizing reranking decisions. In this paper, we propose DynamicRAG, a novel RAG framework where the reranker dynamically adjusts both the order and number of retrieved documents based on the query. We model the reranker as an agent optimized through reinforcement learning (RL), using rewards derived from LLM output quality. Across seven knowledge-intensive datasets, DynamicRAG demonstrates superior performance, achieving state-of-the-art results. The model, data and code are available at this https URL 

---
# Towards user-centered interactive medical image segmentation in VR with an assistive AI agent 

**Authors**: Pascal Spiegler, Arash Harirpoush, Yiming Xiao  

**Link**: [PDF](https://arxiv.org/pdf/2505.07214)  

**Abstract**: Crucial in disease analysis and surgical planning, manual segmentation of volumetric medical scans (e.g. MRI, CT) is laborious, error-prone, and challenging to master, while fully automatic algorithms can benefit from user-feedback. Therefore, with the complementary power of the latest radiological AI foundation models and virtual reality (VR)'s intuitive data interaction, we propose SAMIRA, a novel conversational AI agent that assists users with localizing, segmenting, and visualizing 3D medical concepts in VR. Through speech-based interaction, the agent helps users understand radiological features, locate clinical targets, and generate segmentation masks that can be refined with just a few point prompts. The system also supports true-to-scale 3D visualization of segmented pathology to enhance patient-specific anatomical understanding. Furthermore, to determine the optimal interaction paradigm under near-far attention-switching for refining segmentation masks in an immersive, human-in-the-loop workflow, we compare VR controller pointing, head pointing, and eye tracking as input modes. With a user study, evaluations demonstrated a high usability score (SUS=90.0 $\pm$ 9.0), low overall task load, as well as strong support for the proposed VR system's guidance, training potential, and integration of AI in radiological segmentation tasks. 

---
# Internet of Agents: Fundamentals, Applications, and Challenges 

**Authors**: Yuntao Wang, Shaolong Guo, Yanghe Pan, Zhou Su, Fahao Chen, Tom H. Luan, Peng Li, Jiawen Kang, Dusit Niyato  

**Link**: [PDF](https://arxiv.org/pdf/2505.07176)  

**Abstract**: With the rapid proliferation of large language models and vision-language models, AI agents have evolved from isolated, task-specific systems into autonomous, interactive entities capable of perceiving, reasoning, and acting without human intervention. As these agents proliferate across virtual and physical environments, from virtual assistants to embodied robots, the need for a unified, agent-centric infrastructure becomes paramount. In this survey, we introduce the Internet of Agents (IoA) as a foundational framework that enables seamless interconnection, dynamic discovery, and collaborative orchestration among heterogeneous agents at scale. We begin by presenting a general IoA architecture, highlighting its hierarchical organization, distinguishing features relative to the traditional Internet, and emerging applications. Next, we analyze the key operational enablers of IoA, including capability notification and discovery, adaptive communication protocols, dynamic task matching, consensus and conflict-resolution mechanisms, and incentive models. Finally, we identify open research directions toward building resilient and trustworthy IoA ecosystems. 

---
# Towards Scalable IoT Deployment for Visual Anomaly Detection via Efficient Compression 

**Authors**: Arianna Stropeni, Francesco Borsatti, Manuel Barusco, Davide Dalle Pezze, Marco Fabris, Gian Antonio Susto  

**Link**: [PDF](https://arxiv.org/pdf/2505.07119)  

**Abstract**: Visual Anomaly Detection (VAD) is a key task in industrial settings, where minimizing waste and operational costs is essential. Deploying deep learning models within Internet of Things (IoT) environments introduces specific challenges due to the limited computational power and bandwidth of edge devices. This study investigates how to perform VAD effectively under such constraints by leveraging compact and efficient processing strategies. We evaluate several data compression techniques, examining the trade-off between system latency and detection accuracy. Experiments on the MVTec AD benchmark demonstrate that significant compression can be achieved with minimal loss in anomaly detection performance compared to uncompressed data. 

---
# X-Sim: Cross-Embodiment Learning via Real-to-Sim-to-Real 

**Authors**: Prithwish Dan, Kushal Kedia, Angela Chao, Edward Weiyi Duan, Maximus Adrian Pace, Wei-Chiu Ma, Sanjiban Choudhury  

**Link**: [PDF](https://arxiv.org/pdf/2505.07096)  

**Abstract**: Human videos offer a scalable way to train robot manipulation policies, but lack the action labels needed by standard imitation learning algorithms. Existing cross-embodiment approaches try to map human motion to robot actions, but often fail when the embodiments differ significantly. We propose X-Sim, a real-to-sim-to-real framework that uses object motion as a dense and transferable signal for learning robot policies. X-Sim starts by reconstructing a photorealistic simulation from an RGBD human video and tracking object trajectories to define object-centric rewards. These rewards are used to train a reinforcement learning (RL) policy in simulation. The learned policy is then distilled into an image-conditioned diffusion policy using synthetic rollouts rendered with varied viewpoints and lighting. To transfer to the real world, X-Si introduces an online domain adaptation technique that aligns real and simulated observations during deployment. Importantly, X-Sim does not require any robot teleoperation data. We evaluate it across 5 manipulation tasks in 2 environments and show that it: (1) improves task progress by 30% on average over hand-tracking and sim-to-real baselines, (2) matches behavior cloning with 10x less data collection time, and (3) generalizes to new camera viewpoints and test-time changes. Code and videos are available at this https URL. 

---
# Can LLM-based Financial Investing Strategies Outperform the Market in Long Run? 

**Authors**: Weixian Waylon Li, Hyeonjun Kim, Mihai Cucuringu, Tiejun Ma  

**Link**: [PDF](https://arxiv.org/pdf/2505.07078)  

**Abstract**: Large Language Models (LLMs) have recently been leveraged for asset pricing tasks and stock trading applications, enabling AI agents to generate investment decisions from unstructured financial data. However, most evaluations of LLM timing-based investing strategies are conducted on narrow timeframes and limited stock universes, overstating effectiveness due to survivorship and data-snooping biases. We critically assess their generalizability and robustness by proposing FINSABER, a backtesting framework evaluating timing-based strategies across longer periods and a larger universe of symbols. Systematic backtests over two decades and 100+ symbols reveal that previously reported LLM advantages deteriorate significantly under broader cross-section and over a longer-term evaluation. Our market regime analysis further demonstrates that LLM strategies are overly conservative in bull markets, underperforming passive benchmarks, and overly aggressive in bear markets, incurring heavy losses. These findings highlight the need to develop LLM strategies that are able to prioritise trend detection and regime-aware risk controls over mere scaling of framework complexity. 

---
# ParaView-MCP: An Autonomous Visualization Agent with Direct Tool Use 

**Authors**: Shusen Liu, Haichao Miao, Peer-Timo Bremer  

**Link**: [PDF](https://arxiv.org/pdf/2505.07064)  

**Abstract**: While powerful and well-established, tools like ParaView present a steep learning curve that discourages many potential users. This work introduces ParaView-MCP, an autonomous agent that integrates modern multimodal large language models (MLLMs) with ParaView to not only lower the barrier to entry but also augment ParaView with intelligent decision support. By leveraging the state-of-the-art reasoning, command execution, and vision capabilities of MLLMs, ParaView-MCP enables users to interact with ParaView through natural language and visual inputs. Specifically, our system adopted the Model Context Protocol (MCP) - a standardized interface for model-application communication - that facilitates direct interaction between MLLMs with ParaView's Python API to allow seamless information exchange between the user, the language model, and the visualization tool itself. Furthermore, by implementing a visual feedback mechanism that allows the agent to observe the viewport, we unlock a range of new capabilities, including recreating visualizations from examples, closed-loop visualization parameter updates based on user-defined goals, and even cross-application collaboration involving multiple tools. Broadly, we believe such an agent-driven visualization paradigm can profoundly change the way we interact with visualization tools. We expect a significant uptake in the development of such visualization tools, in both visualization research and industry. 

---
# Seed1.5-VL Technical Report 

**Authors**: Dong Guo, Faming Wu, Feida Zhu, Fuxing Leng, Guang Shi, Haobin Chen, Haoqi Fan, Jian Wang, Jianyu Jiang, Jiawei Wang, Jingji Chen, Jingjia Huang, Kang Lei, Liping Yuan, Lishu Luo, Pengfei Liu, Qinghao Ye, Rui Qian, Shen Yan, Shixiong Zhao, Shuai Peng, Shuangye Li, Sihang Yuan, Sijin Wu, Tianheng Cheng, Weiwei Liu, Wenqian Wang, Xianhan Zeng, Xiao Liu, Xiaobo Qin, Xiaohan Ding, Xiaojun Xiao, Xiaoying Zhang, Xuanwei Zhang, Xuehan Xiong, Yanghua Peng, Yangrui Chen, Yanwei Li, Yanxu Hu, Yi Lin, Yiyuan Hu, Yiyuan Zhang, Youbin Wu, Yu Li, Yudong Liu, Yue Ling, Yujia Qin, Zanbo Wang, Zhiwu He, Aoxue Zhang, Bairen Yi, Bencheng Liao, Can Huang, Can Zhang, Chaorui Deng, Chaoyi Deng, Cheng Lin, Cheng Yuan, Chenggang Li, Chenhui Gou, Chenwei Lou, Chengzhi Wei, Chundian Liu, Chunyuan Li, Deyao Zhu, Donghong Zhong, Feng Li, Feng Zhang, Gang Wu, Guodong Li, Guohong Xiao, Haibin Lin, Haihua Yang, Haoming Wang, Heng Ji, Hongxiang Hao, Hui Shen, Huixia Li, Jiahao Li, Jialong Wu, Jianhua Zhu, Jianpeng Jiao, Jiashi Feng, Jiaze Chen, Jianhui Duan, Jihao Liu, Jin Zeng, Jingqun Tang, Jingyu Sun, Joya Chen, Jun Long, Junda Feng, Junfeng Zhan, Junjie Fang, Junting Lu, Kai Hua, Kai Liu, Kai Shen, Kaiyuan Zhang, Ke Shen  

**Link**: [PDF](https://arxiv.org/pdf/2505.07062)  

**Abstract**: We present Seed1.5-VL, a vision-language foundation model designed to advance general-purpose multimodal understanding and reasoning. Seed1.5-VL is composed with a 532M-parameter vision encoder and a Mixture-of-Experts (MoE) LLM of 20B active parameters. Despite its relatively compact architecture, it delivers strong performance across a wide spectrum of public VLM benchmarks and internal evaluation suites, achieving the state-of-the-art performance on 38 out of 60 public benchmarks. Moreover, in agent-centric tasks such as GUI control and gameplay, Seed1.5-VL outperforms leading multimodal systems, including OpenAI CUA and Claude 3.7. Beyond visual and video understanding, it also demonstrates strong reasoning abilities, making it particularly effective for multimodal reasoning challenges such as visual puzzles. We believe these capabilities will empower broader applications across diverse tasks. In this report, we mainly provide a comprehensive review of our experiences in building Seed1.5-VL across model design, data construction, and training at various stages, hoping that this report can inspire further research. Seed1.5-VL is now accessible at this https URL (Volcano Engine Model ID: doubao-1-5-thinking-vision-pro-250428) 

---
# Empirical Analysis of Asynchronous Federated Learning on Heterogeneous Devices: Efficiency, Fairness, and Privacy Trade-offs 

**Authors**: Samaneh Mohammadi, Iraklis Symeonidis, Ali Balador, Francesco Flammini  

**Link**: [PDF](https://arxiv.org/pdf/2505.07041)  

**Abstract**: Device heterogeneity poses major challenges in Federated Learning (FL), where resource-constrained clients slow down synchronous schemes that wait for all updates before aggregation. Asynchronous FL addresses this by incorporating updates as they arrive, substantially improving efficiency. While its efficiency gains are well recognized, its privacy costs remain largely unexplored, particularly for high-end devices that contribute updates more frequently, increasing their cumulative privacy exposure. This paper presents the first comprehensive analysis of the efficiency-fairness-privacy trade-off in synchronous vs. asynchronous FL under realistic device heterogeneity. We empirically compare FedAvg and staleness-aware FedAsync using a physical testbed of five edge devices spanning diverse hardware tiers, integrating Local Differential Privacy (LDP) and the Moments Accountant to quantify per-client privacy loss. Using Speech Emotion Recognition (SER) as a privacy-critical benchmark, we show that FedAsync achieves up to 10x faster convergence but exacerbates fairness and privacy disparities: high-end devices contribute 6-10x more updates and incur up to 5x higher privacy loss, while low-end devices suffer amplified accuracy degradation due to infrequent, stale, and noise-perturbed updates. These findings motivate the need for adaptive FL protocols that jointly optimize aggregation and privacy mechanisms based on client capacity and participation dynamics, moving beyond static, one-size-fits-all solutions. 

---
# Predicting Diabetes Using Machine Learning: A Comparative Study of Classifiers 

**Authors**: Mahade Hasan, Farhana Yasmin  

**Link**: [PDF](https://arxiv.org/pdf/2505.07036)  

**Abstract**: Diabetes remains a significant health challenge globally, contributing to severe complications like kidney disease, vision loss, and heart issues. The application of machine learning (ML) in healthcare enables efficient and accurate disease prediction, offering avenues for early intervention and patient support. Our study introduces an innovative diabetes prediction framework, leveraging both traditional ML techniques such as Logistic Regression, SVM, Naïve Bayes, and Random Forest and advanced ensemble methods like AdaBoost, Gradient Boosting, Extra Trees, and XGBoost. Central to our approach is the development of a novel model, DNet, a hybrid architecture combining Convolutional Neural Network (CNN) and Long Short-Term Memory (LSTM) layers for effective feature extraction and sequential learning. The DNet model comprises an initial convolutional block for capturing essential features, followed by a residual block with skip connections to facilitate efficient information flow. Batch Normalization and Dropout are employed for robust regularization, and an LSTM layer captures temporal dependencies within the data. Using a Kaggle-sourced real-world diabetes dataset, our model evaluation spans cross-validation accuracy, precision, recall, F1 score, and ROC-AUC. Among the models, DNet demonstrates the highest efficacy with an accuracy of 99.79% and an AUC-ROC of 99.98%, establishing its potential for superior diabetes prediction. This robust hybrid architecture showcases the value of combining CNN and LSTM layers, emphasizing its applicability in medical diagnostics and disease prediction tasks. 

---
# Incremental Uncertainty-aware Performance Monitoring with Active Labeling Intervention 

**Authors**: Alexander Koebler, Thomas Decker, Ingo Thon, Volker Tresp, Florian Buettner  

**Link**: [PDF](https://arxiv.org/pdf/2505.07023)  

**Abstract**: We study the problem of monitoring machine learning models under gradual distribution shifts, where circumstances change slowly over time, often leading to unnoticed yet significant declines in accuracy. To address this, we propose Incremental Uncertainty-aware Performance Monitoring (IUPM), a novel label-free method that estimates performance changes by modeling gradual shifts using optimal transport. In addition, IUPM quantifies the uncertainty in the performance prediction and introduces an active labeling procedure to restore a reliable estimate under a limited labeling budget. Our experiments show that IUPM outperforms existing performance estimation baselines in various gradual shift scenarios and that its uncertainty awareness guides label acquisition more effectively compared to other strategies. 

---
# R-CAGE: A Structural Model for Emotion Output Design in Human-AI Interaction 

**Authors**: Suyeon Choi  

**Link**: [PDF](https://arxiv.org/pdf/2505.07020)  

**Abstract**: This paper presents R-CAGE (Rhythmic Control Architecture for Guarding Ego), a theoretical framework for restructuring emotional output in long-term human-AI interaction. While prior affective computing approaches emphasized expressiveness, immersion, and responsiveness, they often neglected the cognitive and structural consequences of repeated emotional engagement. R-CAGE instead conceptualizes emotional output not as reactive expression but as ethical design structure requiring architectural intervention. The model is grounded in experiential observations of subtle affective symptoms such as localized head tension, interpretive fixation, and emotional lag arising from prolonged interaction with affective AI systems. These indicate a mismatch between system-driven emotion and user interpretation that cannot be fully explained by biometric data or observable behavior. R-CAGE adopts a user-centered stance prioritizing psychological recovery, interpretive autonomy, and identity continuity. The framework consists of four control blocks: (1) Control of Rhythmic Expression regulates output pacing to reduce fatigue; (2) Architecture of Sensory Structuring adjusts intensity and timing of affective stimuli; (3) Guarding of Cognitive Framing reduces semantic pressure to allow flexible interpretation; (4) Ego-Aligned Response Design supports self-reference recovery during interpretive lag. By structurally regulating emotional rhythm, sensory intensity, and interpretive affordances, R-CAGE frames emotion not as performative output but as sustainable design unit. The goal is to protect users from oversaturation and cognitive overload while sustaining long-term interpretive agency in AI-mediated environments. 

---
# Efficient and Robust Multidimensional Attention in Remote Physiological Sensing through Target Signal Constrained Factorization 

**Authors**: Jitesh Joshi, Youngjun Cho  

**Link**: [PDF](https://arxiv.org/pdf/2505.07013)  

**Abstract**: Remote physiological sensing using camera-based technologies offers transformative potential for non-invasive vital sign monitoring across healthcare and human-computer interaction domains. Although deep learning approaches have advanced the extraction of physiological signals from video data, existing methods have not been sufficiently assessed for their robustness to domain shifts. These shifts in remote physiological sensing include variations in ambient conditions, camera specifications, head movements, facial poses, and physiological states which often impact real-world performance significantly. Cross-dataset evaluation provides an objective measure to assess generalization capabilities across these domain shifts. We introduce Target Signal Constrained Factorization module (TSFM), a novel multidimensional attention mechanism that explicitly incorporates physiological signal characteristics as factorization constraints, allowing more precise feature extraction. Building on this innovation, we present MMRPhys, an efficient dual-branch 3D-CNN architecture designed for simultaneous multitask estimation of photoplethysmography (rPPG) and respiratory (rRSP) signals from multimodal RGB and thermal video inputs. Through comprehensive cross-dataset evaluation on five benchmark datasets, we demonstrate that MMRPhys with TSFM significantly outperforms state-of-the-art methods in generalization across domain shifts for rPPG and rRSP estimation, while maintaining a minimal inference latency suitable for real-time applications. Our approach establishes new benchmarks for robust multitask and multimodal physiological sensing and offers a computationally efficient framework for practical deployment in unconstrained environments. The web browser-based application featuring on-device real-time inference of MMRPhys model is available at this https URL 

---
# Hand-Shadow Poser 

**Authors**: Hao Xu, Yinqiao Wang, Niloy J. Mitra, Shuaicheng Liu, Pheng-Ann Heng, Chi-Wing Fu  

**Link**: [PDF](https://arxiv.org/pdf/2505.07012)  

**Abstract**: Hand shadow art is a captivating art form, creatively using hand shadows to reproduce expressive shapes on the wall. In this work, we study an inverse problem: given a target shape, find the poses of left and right hands that together best produce a shadow resembling the input. This problem is nontrivial, since the design space of 3D hand poses is huge while being restrictive due to anatomical constraints. Also, we need to attend to the input's shape and crucial features, though the input is colorless and textureless. To meet these challenges, we design Hand-Shadow Poser, a three-stage pipeline, to decouple the anatomical constraints (by hand) and semantic constraints (by shadow shape): (i) a generative hand assignment module to explore diverse but reasonable left/right-hand shape hypotheses; (ii) a generalized hand-shadow alignment module to infer coarse hand poses with a similarity-driven strategy for selecting hypotheses; and (iii) a shadow-feature-aware refinement module to optimize the hand poses for physical plausibility and shadow feature preservation. Further, we design our pipeline to be trainable on generic public hand data, thus avoiding the need for any specialized training dataset. For method validation, we build a benchmark of 210 diverse shadow shapes of varying complexity and a comprehensive set of metrics, including a novel DINOv2-based evaluation metric. Through extensive comparisons with multiple baselines and user studies, our approach is demonstrated to effectively generate bimanual hand poses for a large variety of hand shapes for over 85% of the benchmark cases. 

---
# Towards the Three-Phase Dynamics of Generalization Power of a DNN 

**Authors**: Yuxuan He, Junpeng Zhang, Hongyuan Zhang, Quanshi Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2505.06993)  

**Abstract**: This paper proposes a new perspective for analyzing the generalization power of deep neural networks (DNNs), i.e., directly disentangling and analyzing the dynamics of generalizable and non-generalizable interaction encoded by a DNN through the training process. Specifically, this work builds upon the recent theoretical achievement in explainble AI, which proves that the detailed inference logic of DNNs can be can be strictly rewritten as a small number of AND-OR interaction patterns. Based on this, we propose an efficient method to quantify the generalization power of each interaction, and we discover a distinct three-phase dynamics of the generalization power of interactions during training. In particular, the early phase of training typically removes noisy and non-generalizable interactions and learns simple and generalizable ones. The second and the third phases tend to capture increasingly complex interactions that are harder to generalize. Experimental results verify that the learning of non-generalizable interactions is the the direct cause for the gap between the training and testing losses. 

---
# Convert Language Model into a Value-based Strategic Planner 

**Authors**: Xiaoyu Wang, Yue Zhao, Qingqing Gu, Zhonglin Jiang, Xiaokai Chen, Yong Chen, Luo Ji  

**Link**: [PDF](https://arxiv.org/pdf/2505.06987)  

**Abstract**: Emotional support conversation (ESC) aims to alleviate the emotional distress of individuals through effective conversations. Although large language models (LLMs) have obtained remarkable progress on ESC, most of these studies might not define the diagram from the state model perspective, therefore providing a suboptimal solution for long-term satisfaction. To address such an issue, we leverage the Q-learning on LLMs, and propose a framework called straQ*. Our framework allows a plug-and-play LLM to bootstrap the planning during ESC, determine the optimal strategy based on long-term returns, and finally guide the LLM to response. Substantial experiments on ESC datasets suggest that straQ* outperforms many baselines, including direct inference, self-refine, chain of thought, finetuning, and finite state machines. 

---
# Reinforcement Learning-Based Monocular Vision Approach for Autonomous UAV Landing 

**Authors**: Tarik Houichime, Younes EL Amrani  

**Link**: [PDF](https://arxiv.org/pdf/2505.06963)  

**Abstract**: This paper introduces an innovative approach for the autonomous landing of Unmanned Aerial Vehicles (UAVs) using only a front-facing monocular camera, therefore obviating the requirement for depth estimation cameras. Drawing on the inherent human estimating process, the proposed method reframes the landing task as an optimization problem. The UAV employs variations in the visual characteristics of a specially designed lenticular circle on the landing pad, where the perceived color and form provide critical information for estimating both altitude and depth. Reinforcement learning algorithms are utilized to approximate the functions governing these estimations, enabling the UAV to ascertain ideal landing settings via training. This method's efficacy is assessed by simulations and experiments, showcasing its potential for robust and accurate autonomous landing without dependence on complex sensor setups. This research contributes to the advancement of cost-effective and efficient UAV landing solutions, paving the way for wider applicability across various fields. 

---
# AI-Powered Inverse Design of Ku-Band SIW Resonant Structures by Iterative Residual Correction Network 

**Authors**: Mohammad Mashayekhi, Kamran Salehian  

**Link**: [PDF](https://arxiv.org/pdf/2505.06936)  

**Abstract**: Inverse electromagnetic modeling has emerged as a powerful approach for designing complex microwave structures with high accuracy and efficiency. In this study, we propose an Iterative Residual Correction Network (IRC-Net) for the inverse design of Ku-band Substrate Integrated Waveguide (SIW) components based on multimode resonators. We use a multimode resonance structure to demonstrate that it is possible to control the resonances of the structure. Therefore, these structures can be used for resonant components and smart filter design. The proposed deep learning architecture leverages residual neural networks to overcome the limitations of traditional inverse design techniques, such as the Feedforward Inverse Model (FIM), offering improved generalization and prediction accuracy. The approach begins with a FIM to generate initial design estimates, followed by an iterative correction strategy inspired by the Hybrid Inverse-Forward Residual Refinement Network (HiFR\textsuperscript{2}-Net), which we call IRC-Net. Experiments demonstrate that the IRC-Net achieves substantial improvements in prediction accuracy compared to traditional single-stage networks, validated through statistical metrics, full-wave electromagnetic simulations, and measurements. To validate the proposed framework, we first design and fabricate a three-resonance SIW structure. Next, we apply the trained IRC-Net model to predict the geometry of a four-resonance structure based on its desired frequency response. Both designs are fabricated and tested, showing strong agreement between the simulated, predicted, and measured results, confirming the effectiveness and practicality of the proposed method. 

---
# RedTeamLLM: an Agentic AI framework for offensive security 

**Authors**: Brian Challita, Pierre Parrend  

**Link**: [PDF](https://arxiv.org/pdf/2505.06913)  

**Abstract**: From automated intrusion testing to discovery of zero-day attacks before software launch, agentic AI calls for great promises in security engineering. This strong capability is bound with a similar threat: the security and research community must build up its models before the approach is leveraged by malicious actors for cybercrime. We therefore propose and evaluate RedTeamLLM, an integrated architecture with a comprehensive security model for automatization of pentest tasks. RedTeamLLM follows three key steps: summarizing, reasoning and act, which embed its operational capacity. This novel framework addresses four open challenges: plan correction, memory management, context window constraint, and generality vs. specialization. Evaluation is performed through the automated resolution of a range of entry-level, but not trivial, CTF challenges. The contribution of the reasoning capability of our agentic AI framework is specifically evaluated. 

---
# MMiC: Mitigating Modality Incompleteness in Clustered Federated Learning 

**Authors**: Lishan Yang, Wei Zhang, Quan Z. Sheng, Weitong Chen, Lina Yao, Weitong Chen, Ali Shakeri  

**Link**: [PDF](https://arxiv.org/pdf/2505.06911)  

**Abstract**: In the era of big data, data mining has become indispensable for uncovering hidden patterns and insights from vast and complex datasets. The integration of multimodal data sources further enhances its potential. Multimodal Federated Learning (MFL) is a distributed approach that enhances the efficiency and quality of multimodal learning, ensuring collaborative work and privacy protection. However, missing modalities pose a significant challenge in MFL, often due to data quality issues or privacy policies across the clients. In this work, we present MMiC, a framework for Mitigating Modality incompleteness in MFL within the Clusters. MMiC replaces partial parameters within client models inside clusters to mitigate the impact of missing modalities. Furthermore, it leverages the Banzhaf Power Index to optimize client selection under these conditions. Finally, MMiC employs an innovative approach to dynamically control global aggregation by utilizing Markovitz Portfolio Optimization. Extensive experiments demonstrate that MMiC consistently outperforms existing federated learning architectures in both global and personalized performance on multimodal datasets with missing modalities, confirming the effectiveness of our proposed solution. 

---
# NeuGen: Amplifying the 'Neural' in Neural Radiance Fields for Domain Generalization 

**Authors**: Ahmed Qazi, Abdul Basit, Asim Iqbal  

**Link**: [PDF](https://arxiv.org/pdf/2505.06894)  

**Abstract**: Neural Radiance Fields (NeRF) have significantly advanced the field of novel view synthesis, yet their generalization across diverse scenes and conditions remains challenging. Addressing this, we propose the integration of a novel brain-inspired normalization technique Neural Generalization (NeuGen) into leading NeRF architectures which include MVSNeRF and GeoNeRF. NeuGen extracts the domain-invariant features, thereby enhancing the models' generalization capabilities. It can be seamlessly integrated into NeRF architectures and cultivates a comprehensive feature set that significantly improves accuracy and robustness in image rendering. Through this integration, NeuGen shows improved performance on benchmarks on diverse datasets across state-of-the-art NeRF architectures, enabling them to generalize better across varied scenes. Our comprehensive evaluations, both quantitative and qualitative, confirm that our approach not only surpasses existing models in generalizability but also markedly improves rendering quality. Our work exemplifies the potential of merging neuroscientific principles with deep learning frameworks, setting a new precedent for enhanced generalizability and efficiency in novel view synthesis. A demo of our study is available at this https URL. 

---
# IM-BERT: Enhancing Robustness of BERT through the Implicit Euler Method 

**Authors**: Mihyeon Kim, Juhyoung Park, Youngbin Kim  

**Link**: [PDF](https://arxiv.org/pdf/2505.06889)  

**Abstract**: Pre-trained Language Models (PLMs) have achieved remarkable performance on diverse NLP tasks through pre-training and fine-tuning. However, fine-tuning the model with a large number of parameters on limited downstream datasets often leads to vulnerability to adversarial attacks, causing overfitting of the model on standard datasets.
To address these issues, we propose IM-BERT from the perspective of a dynamic system by conceptualizing a layer of BERT as a solution of Ordinary Differential Equations (ODEs). Under the situation of initial value perturbation, we analyze the numerical stability of two main numerical ODE solvers: the explicit and implicit Euler approaches.
Based on these analyses, we introduce a numerically robust IM-connection incorporating BERT's layers. This strategy enhances the robustness of PLMs against adversarial attacks, even in low-resource scenarios, without introducing additional parameters or adversarial training strategies.
Experimental results on the adversarial GLUE (AdvGLUE) dataset validate the robustness of IM-BERT under various conditions. Compared to the original BERT, IM-BERT exhibits a performance improvement of approximately 8.3\%p on the AdvGLUE dataset. Furthermore, in low-resource scenarios, IM-BERT outperforms BERT by achieving 5.9\%p higher accuracy. 

---
# Mice to Machines: Neural Representations from Visual Cortex for Domain Generalization 

**Authors**: Ahmed Qazi, Hamd Jalil, Asim Iqbal  

**Link**: [PDF](https://arxiv.org/pdf/2505.06886)  

**Abstract**: The mouse is one of the most studied animal models in the field of systems neuroscience. Understanding the generalized patterns and decoding the neural representations that are evoked by the diverse range of natural scene stimuli in the mouse visual cortex is one of the key quests in computational vision. In recent years, significant parallels have been drawn between the primate visual cortex and hierarchical deep neural networks. However, their generalized efficacy in understanding mouse vision has been limited. In this study, we investigate the functional alignment between the mouse visual cortex and deep learning models for object classification tasks. We first introduce a generalized representational learning strategy that uncovers a striking resemblance between the functional mapping of the mouse visual cortex and high-performing deep learning models on both top-down (population-level) and bottom-up (single cell-level) scenarios. Next, this representational similarity across the two systems is further enhanced by the addition of Neural Response Normalization (NeuRN) layer, inspired by the activation profile of excitatory and inhibitory neurons in the visual cortex. To test the performance effect of NeuRN on real-world tasks, we integrate it into deep learning models and observe significant improvements in their robustness against data shifts in domain generalization tasks. Our work proposes a novel framework for comparing the functional architecture of the mouse visual cortex with deep learning models. Our findings carry broad implications for the development of advanced AI models that draw inspiration from the mouse visual cortex, suggesting that these models serve as valuable tools for studying the neural representations of the mouse visual cortex and, as a result, enhancing their performance on real-world tasks. 

---
# FACET: Force-Adaptive Control via Impedance Reference Tracking for Legged Robots 

**Authors**: Botian Xu, Haoyang Weng, Qingzhou Lu, Yang Gao, Huazhe Xu  

**Link**: [PDF](https://arxiv.org/pdf/2505.06883)  

**Abstract**: Reinforcement learning (RL) has made significant strides in legged robot control, enabling locomotion across diverse terrains and complex loco-manipulation capabilities. However, the commonly used position or velocity tracking-based objectives are agnostic to forces experienced by the robot, leading to stiff and potentially dangerous behaviors and poor control during forceful interactions. To address this limitation, we present \emph{Force-Adaptive Control via Impedance Reference Tracking} (FACET). Inspired by impedance control, we use RL to train a control policy to imitate a virtual mass-spring-damper system, allowing fine-grained control under external forces by manipulating the virtual spring. In simulation, we demonstrate that our quadruped robot achieves improved robustness to large impulses (up to 200 Ns) and exhibits controllable compliance, achieving an 80% reduction in collision impulse. The policy is deployed to a physical robot to showcase both compliance and the ability to engage with large forces by kinesthetic control and pulling payloads up to 2/3 of its weight. Further extension to a legged loco-manipulator and a humanoid shows the applicability of our method to more complex settings to enable whole-body compliance control. Project Website: this https URL 

---
# NeuRN: Neuro-inspired Domain Generalization for Image Classification 

**Authors**: Hamd Jalil, Ahmed Qazi, Asim Iqbal  

**Link**: [PDF](https://arxiv.org/pdf/2505.06881)  

**Abstract**: Domain generalization in image classification is a crucial challenge, with models often failing to generalize well across unseen datasets. We address this issue by introducing a neuro-inspired Neural Response Normalization (NeuRN) layer which draws inspiration from neurons in the mammalian visual cortex, which aims to enhance the performance of deep learning architectures on unseen target domains by training deep learning models on a source domain. The performance of these models is considered as a baseline and then compared against models integrated with NeuRN on image classification tasks. We perform experiments across a range of deep learning architectures, including ones derived from Neural Architecture Search and Vision Transformer. Additionally, in order to shortlist models for our experiment from amongst the vast range of deep neural networks available which have shown promising results, we also propose a novel method that uses the Needleman-Wunsch algorithm to compute similarity between deep learning architectures. Our results demonstrate the effectiveness of NeuRN by showing improvement against baseline in cross-domain image classification tasks. Our framework attempts to establish a foundation for future neuro-inspired deep learning models. 

---
# Enhancing Time Series Forecasting via a Parallel Hybridization of ARIMA and Polynomial Classifiers 

**Authors**: Thanh Son Nguyen, Van Thanh Nguyen, Dang Minh Duc Nguyen  

**Link**: [PDF](https://arxiv.org/pdf/2505.06874)  

**Abstract**: Time series forecasting has attracted significant attention, leading to the de-velopment of a wide range of approaches, from traditional statistical meth-ods to advanced deep learning models. Among them, the Auto-Regressive Integrated Moving Average (ARIMA) model remains a widely adopted linear technique due to its effectiveness in modeling temporal dependencies in economic, industrial, and social data. On the other hand, polynomial classifi-ers offer a robust framework for capturing non-linear relationships and have demonstrated competitive performance in domains such as stock price pre-diction. In this study, we propose a hybrid forecasting approach that inte-grates the ARIMA model with a polynomial classifier to leverage the com-plementary strengths of both models. The hybrid method is evaluated on multiple real-world time series datasets spanning diverse domains. Perfor-mance is assessed based on forecasting accuracy and computational effi-ciency. Experimental results reveal that the proposed hybrid model consist-ently outperforms the individual models in terms of prediction accuracy, al-beit with a modest increase in execution time. 

---
# Efficient Robotic Policy Learning via Latent Space Backward Planning 

**Authors**: Dongxiu Liu, Haoyi Niu, Zhihao Wang, Jinliang Zheng, Yinan Zheng, Zhonghong Ou, Jianming Hu, Jianxiong Li, Xianyuan Zhan  

**Link**: [PDF](https://arxiv.org/pdf/2505.06861)  

**Abstract**: Current robotic planning methods often rely on predicting multi-frame images with full pixel details. While this fine-grained approach can serve as a generic world model, it introduces two significant challenges for downstream policy learning: substantial computational costs that hinder real-time deployment, and accumulated inaccuracies that can mislead action extraction. Planning with coarse-grained subgoals partially alleviates efficiency issues. However, their forward planning schemes can still result in off-task predictions due to accumulation errors, leading to misalignment with long-term goals. This raises a critical question: Can robotic planning be both efficient and accurate enough for real-time control in long-horizon, multi-stage tasks? To address this, we propose a Latent Space Backward Planning scheme (LBP), which begins by grounding the task into final latent goals, followed by recursively predicting intermediate subgoals closer to the current state. The grounded final goal enables backward subgoal planning to always remain aware of task completion, facilitating on-task prediction along the entire planning horizon. The subgoal-conditioned policy incorporates a learnable token to summarize the subgoal sequences and determines how each subgoal guides action extraction. Through extensive simulation and real-robot long-horizon experiments, we show that LBP outperforms existing fine-grained and forward planning methods, achieving SOTA performance. Project Page: this https URL 

---
# DP-TRAE: A Dual-Phase Merging Transferable Reversible Adversarial Example for Image Privacy Protection 

**Authors**: Xia Du, Jiajie Zhu, Jizhe Zhou, Chi-man Pun, Zheng Lin, Cong Wu, Zhe Chen, Jun Luo  

**Link**: [PDF](https://arxiv.org/pdf/2505.06860)  

**Abstract**: In the field of digital security, Reversible Adversarial Examples (RAE) combine adversarial attacks with reversible data hiding techniques to effectively protect sensitive data and prevent unauthorized analysis by malicious Deep Neural Networks (DNNs). However, existing RAE techniques primarily focus on white-box attacks, lacking a comprehensive evaluation of their effectiveness in black-box scenarios. This limitation impedes their broader deployment in complex, dynamic environments. Further more, traditional black-box attacks are often characterized by poor transferability and high query costs, significantly limiting their practical applicability. To address these challenges, we propose the Dual-Phase Merging Transferable Reversible Attack method, which generates highly transferable initial adversarial perturbations in a white-box model and employs a memory augmented black-box strategy to effectively mislead target mod els. Experimental results demonstrate the superiority of our approach, achieving a 99.0% attack success rate and 100% recovery rate in black-box scenarios, highlighting its robustness in privacy protection. Moreover, we successfully implemented a black-box attack on a commercial model, further substantiating the potential of this approach for practical use. 

---
# Optimizing Recommendations using Fine-Tuned LLMs 

**Authors**: Prabhdeep Cheema, Erhan Guven  

**Link**: [PDF](https://arxiv.org/pdf/2505.06841)  

**Abstract**: As digital media platforms strive to meet evolving user expectations, delivering highly personalized and intuitive movies and media recommendations has become essential for attracting and retaining audiences. Traditional systems often rely on keyword-based search and recommendation techniques, which limit users to specific keywords and a combination of keywords. This paper proposes an approach that generates synthetic datasets by modeling real-world user interactions, creating complex chat-style data reflective of diverse preferences. This allows users to express more information with complex preferences, such as mood, plot details, and thematic elements, in addition to conventional criteria like genre, title, and actor-based searches. In today's search space, users cannot write queries like ``Looking for a fantasy movie featuring dire wolves, ideally set in a harsh frozen world with themes of loyalty and survival.''
Building on these contributions, we evaluate synthetic datasets for diversity and effectiveness in training and benchmarking models, particularly in areas often absent from traditional datasets. This approach enhances personalization and accuracy by enabling expressive and natural user queries. It establishes a foundation for the next generation of conversational AI-driven search and recommendation systems in digital entertainment. 

---
# The power of fine-grained experts: Granularity boosts expressivity in Mixture of Experts 

**Authors**: Enric Boix-Adsera, Philippe Rigollet  

**Link**: [PDF](https://arxiv.org/pdf/2505.06839)  

**Abstract**: Mixture-of-Experts (MoE) layers are increasingly central to frontier model architectures. By selectively activating parameters, they reduce computational cost while scaling total parameter count. This paper investigates the impact of the number of active experts, termed granularity, comparing architectures with many (e.g., 8 per layer in DeepSeek) to those with fewer (e.g., 1 per layer in Llama-4 models). We prove an exponential separation in network expressivity based on this design parameter, suggesting that models benefit from higher granularity. Experimental results corroborate our theoretical findings and illustrate this separation. 

---
# Sandcastles in the Storm: Revisiting the (Im)possibility of Strong Watermarking 

**Authors**: Fabrice Y Harel-Canada, Boran Erol, Connor Choi, Jason Liu, Gary Jiarui Song, Nanyun Peng, Amit Sahai  

**Link**: [PDF](https://arxiv.org/pdf/2505.06827)  

**Abstract**: Watermarking AI-generated text is critical for combating misuse. Yet recent theoretical work argues that any watermark can be erased via random walk attacks that perturb text while preserving quality. However, such attacks rely on two key assumptions: (1) rapid mixing (watermarks dissolve quickly under perturbations) and (2) reliable quality preservation (automated quality oracles perfectly guide edits). Through large-scale experiments and human-validated assessments, we find mixing is slow: 100% of perturbed texts retain traces of their origin after hundreds of edits, defying rapid mixing. Oracles falter, as state-of-the-art quality detectors misjudge edits (77% accuracy), compounding errors during attacks. Ultimately, attacks underperform: automated walks remove watermarks just 26% of the time -- dropping to 10% under human quality review. These findings challenge the inevitability of watermark removal. Instead, practical barriers -- slow mixing and imperfect quality control -- reveal watermarking to be far more robust than theoretical models suggest. The gap between idealized attacks and real-world feasibility underscores the need for stronger watermarking methods and more realistic attack models. 

---
# ThreatLens: LLM-guided Threat Modeling and Test Plan Generation for Hardware Security Verification 

**Authors**: Dipayan Saha, Hasan Al Shaikh, Shams Tarek, Farimah Farahmandi  

**Link**: [PDF](https://arxiv.org/pdf/2505.06821)  

**Abstract**: Current hardware security verification processes predominantly rely on manual threat modeling and test plan generation, which are labor-intensive, error-prone, and struggle to scale with increasing design complexity and evolving attack methodologies. To address these challenges, we propose ThreatLens, an LLM-driven multi-agent framework that automates security threat modeling and test plan generation for hardware security verification. ThreatLens integrates retrieval-augmented generation (RAG) to extract relevant security knowledge, LLM-powered reasoning for threat assessment, and interactive user feedback to ensure the generation of practical test plans. By automating these processes, the framework reduces the manual verification effort, enhances coverage, and ensures a structured, adaptable approach to security verification. We evaluated our framework on the NEORV32 SoC, demonstrating its capability to automate security verification through structured test plans and validating its effectiveness in real-world scenarios. 

---
# Overview of the NLPCC 2025 Shared Task 4: Multi-modal, Multilingual, and Multi-hop Medical Instructional Video Question Answering Challenge 

**Authors**: Bin Li, Shenxi Liu, Yixuan Weng, Yue Du, Yuhang Tian, Shoujun Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2505.06814)  

**Abstract**: Following the successful hosts of the 1-st (NLPCC 2023 Foshan) CMIVQA and the 2-rd (NLPCC 2024 Hangzhou) MMIVQA challenges, this year, a new task has been introduced to further advance research in multi-modal, multilingual, and multi-hop medical instructional question answering (M4IVQA) systems, with a specific focus on medical instructional videos. The M4IVQA challenge focuses on evaluating models that integrate information from medical instructional videos, understand multiple languages, and answer multi-hop questions requiring reasoning over various modalities. This task consists of three tracks: multi-modal, multilingual, and multi-hop Temporal Answer Grounding in Single Video (M4TAGSV), multi-modal, multilingual, and multi-hop Video Corpus Retrieval (M4VCR) and multi-modal, multilingual, and multi-hop Temporal Answer Grounding in Video Corpus (M4TAGVC). Participants in M4IVQA are expected to develop algorithms capable of processing both video and text data, understanding multilingual queries, and providing relevant answers to multi-hop medical questions. We believe the newly introduced M4IVQA challenge will drive innovations in multimodal reasoning systems for healthcare scenarios, ultimately contributing to smarter emergency response systems and more effective medical education platforms in multilingual communities. Our official website is this https URL 

---
# Quantum Observers: A NISQ Hardware Demonstration of Chaotic State Prediction Using Quantum Echo-state Networks 

**Authors**: Erik L. Connerty, Ethan N. Evans, Gerasimos Angelatos, Vignesh Narayanan  

**Link**: [PDF](https://arxiv.org/pdf/2505.06799)  

**Abstract**: Recent advances in artificial intelligence have highlighted the remarkable capabilities of neural network (NN)-powered systems on classical computers. However, these systems face significant computational challenges that limit scalability and efficiency. Quantum computers hold the potential to overcome these limitations and increase processing power beyond classical systems. Despite this, integrating quantum computing with NNs remains largely unrealized due to challenges posed by noise, decoherence, and high error rates in current quantum hardware. Here, we propose a novel quantum echo-state network (QESN) design and implementation algorithm that can operate within the presence of noise on current IBM hardware. We apply classical control-theoretic response analysis to characterize the QESN, emphasizing its rich nonlinear dynamics and memory, as well as its ability to be fine-tuned with sparsity and re-uploading blocks. We validate our approach through a comprehensive demonstration of QESNs functioning as quantum observers, applied in both high-fidelity simulations and hardware experiments utilizing data from a prototypical chaotic Lorenz system. Our results show that the QESN can predict long time-series with persistent memory, running over 100 times longer than the median T}1 and T2 of the IBM Marrakesh QPU, achieving state-of-the-art time-series performance on superconducting hardware. 

---
# Decoding Futures Price Dynamics: A Regularized Sparse Autoencoder for Interpretable Multi-Horizon Forecasting and Factor Discovery 

**Authors**: Abhijit Gupta  

**Link**: [PDF](https://arxiv.org/pdf/2505.06795)  

**Abstract**: Commodity price volatility creates economic challenges, necessitating accurate multi-horizon forecasting. Predicting prices for commodities like copper and crude oil is complicated by diverse interacting factors (macroeconomic, supply/demand, geopolitical, etc.). Current models often lack transparency, limiting strategic use. This paper presents a Regularized Sparse Autoencoder (RSAE), a deep learning framework for simultaneous multi-horizon commodity price prediction and discovery of interpretable latent market drivers. The RSAE forecasts prices at multiple horizons (e.g., 1-day, 1-week, 1-month) using multivariate time series. Crucially, L1 regularization ($\|\mathbf{z}\|_1$) on its latent vector $\mathbf{z}$ enforces sparsity, promoting parsimonious explanations of market dynamics through learned factors representing underlying drivers (e.g., demand, supply shocks). Drawing from energy-based models and sparse coding, the RSAE optimizes predictive accuracy while learning sparse representations. Evaluated on historical Copper and Crude Oil data with numerous indicators, our findings indicate the RSAE offers competitive multi-horizon forecasting accuracy and data-driven insights into price dynamics via its interpretable latent space, a key advantage over traditional black-box approaches. 

---
# Symbolic Rule Extraction from Attention-Guided Sparse Representations in Vision Transformers 

**Authors**: Parth Padalkar, Gopal Gupta  

**Link**: [PDF](https://arxiv.org/pdf/2505.06745)  

**Abstract**: Recent neuro-symbolic approaches have successfully extracted symbolic rule-sets from CNN-based models to enhance interpretability. However, applying similar techniques to Vision Transformers (ViTs) remains challenging due to their lack of modular concept detectors and reliance on global self-attention mechanisms. We propose a framework for symbolic rule extraction from ViTs by introducing a sparse concept layer inspired by Sparse Autoencoders (SAEs). This linear layer operates on attention-weighted patch representations and learns a disentangled, binarized representation in which individual neurons activate for high-level visual concepts. To encourage interpretability, we apply a combination of L1 sparsity, entropy minimization, and supervised contrastive loss. These binarized concept activations are used as input to the FOLD-SE-M algorithm, which generates a rule-set in the form of logic programs. Our method achieves a 5.14% better classification accuracy than the standard ViT while enabling symbolic reasoning. Crucially, the extracted rule-set is not merely post-hoc but acts as a logic-based decision layer that operates directly on the sparse concept representations. The resulting programs are concise and semantically meaningful. This work is the first to extract executable logic programs from ViTs using sparse symbolic representations. It bridges the gap between transformer-based vision models and symbolic logic programming, providing a step forward in interpretable and verifiable neuro-symbolic AI. 

---
# TPK: Trustworthy Trajectory Prediction Integrating Prior Knowledge For Interpretability and Kinematic Feasibility 

**Authors**: Marius Baden, Ahmed Abouelazm, Christian Hubschneider, Yin Wu, Daniel Slieter, J. Marius Zöllner  

**Link**: [PDF](https://arxiv.org/pdf/2505.06743)  

**Abstract**: Trajectory prediction is crucial for autonomous driving, enabling vehicles to navigate safely by anticipating the movements of surrounding road users. However, current deep learning models often lack trustworthiness as their predictions can be physically infeasible and illogical to humans. To make predictions more trustworthy, recent research has incorporated prior knowledge, like the social force model for modeling interactions and kinematic models for physical realism. However, these approaches focus on priors that suit either vehicles or pedestrians and do not generalize to traffic with mixed agent classes. We propose incorporating interaction and kinematic priors of all agent classes--vehicles, pedestrians, and cyclists with class-specific interaction layers to capture agent behavioral differences. To improve the interpretability of the agent interactions, we introduce DG-SFM, a rule-based interaction importance score that guides the interaction layer. To ensure physically feasible predictions, we proposed suitable kinematic models for all agent classes with a novel pedestrian kinematic model. We benchmark our approach on the Argoverse 2 dataset, using the state-of-the-art transformer HPTR as our baseline. Experiments demonstrate that our method improves interaction interpretability, revealing a correlation between incorrect predictions and divergence from our interaction prior. Even though incorporating the kinematic models causes a slight decrease in accuracy, they eliminate infeasible trajectories found in the dataset and the baseline model. Thus, our approach fosters trust in trajectory prediction as its interaction reasoning is interpretable, and its predictions adhere to physics. 

---
# Boundary-Guided Trajectory Prediction for Road Aware and Physically Feasible Autonomous Driving 

**Authors**: Ahmed Abouelazm, Mianzhi Liu, Christian Hubschneider, Yin Wu, Daniel Slieter, J. Marius Zöllner  

**Link**: [PDF](https://arxiv.org/pdf/2505.06740)  

**Abstract**: Accurate prediction of surrounding road users' trajectories is essential for safe and efficient autonomous driving. While deep learning models have improved performance, challenges remain in preventing off-road predictions and ensuring kinematic feasibility. Existing methods incorporate road-awareness modules and enforce kinematic constraints but lack plausibility guarantees and often introduce trade-offs in complexity and flexibility. This paper proposes a novel framework that formulates trajectory prediction as a constrained regression guided by permissible driving directions and their boundaries. Using the agent's current state and an HD map, our approach defines the valid boundaries and ensures on-road predictions by training the network to learn superimposed paths between left and right boundary polylines. To guarantee feasibility, the model predicts acceleration profiles that determine the vehicle's travel distance along these paths while adhering to kinematic constraints. We evaluate our approach on the Argoverse-2 dataset against the HPTR baseline. Our approach shows a slight decrease in benchmark metrics compared to HPTR but notably improves final displacement error and eliminates infeasible trajectories. Moreover, the proposed approach has superior generalization to less prevalent maneuvers and unseen out-of-distribution scenarios, reducing the off-road rate under adversarial attacks from 66\% to just 1\%. These results highlight the effectiveness of our approach in generating feasible and robust predictions. 

---
# Balancing Progress and Safety: A Novel Risk-Aware Objective for RL in Autonomous Driving 

**Authors**: Ahmed Abouelazm, Jonas Michel, Helen Gremmelmaier, Tim Joseph, Philip Schörner, J. Marius Zöllner  

**Link**: [PDF](https://arxiv.org/pdf/2505.06737)  

**Abstract**: Reinforcement Learning (RL) is a promising approach for achieving autonomous driving due to robust decision-making capabilities. RL learns a driving policy through trial and error in traffic scenarios, guided by a reward function that combines the driving objectives. The design of such reward function has received insufficient attention, yielding ill-defined rewards with various pitfalls. Safety, in particular, has long been regarded only as a penalty for collisions. This leaves the risks associated with actions leading up to a collision unaddressed, limiting the applicability of RL in real-world scenarios. To address these shortcomings, our work focuses on enhancing the reward formulation by defining a set of driving objectives and structuring them hierarchically. Furthermore, we discuss the formulation of these objectives in a normalized manner to transparently determine their contribution to the overall reward. Additionally, we introduce a novel risk-aware objective for various driving interactions based on a two-dimensional ellipsoid function and an extension of Responsibility-Sensitive Safety (RSS) concepts. We evaluate the efficacy of our proposed reward in unsignalized intersection scenarios with varying traffic densities. The approach decreases collision rates by 21\% on average compared to baseline rewards and consistently surpasses them in route progress and cumulative reward, demonstrating its capability to promote safer driving behaviors while maintaining high-performance levels. 

---
# Deeply Explainable Artificial Neural Network 

**Authors**: David Zucker  

**Link**: [PDF](https://arxiv.org/pdf/2505.06731)  

**Abstract**: While deep learning models have demonstrated remarkable success in numerous domains, their black-box nature remains a significant limitation, especially in critical fields such as medical image analysis and inference. Existing explainability methods, such as SHAP, LIME, and Grad-CAM, are typically applied post hoc, adding computational overhead and sometimes producing inconsistent or ambiguous results. In this paper, we present the Deeply Explainable Artificial Neural Network (DxANN), a novel deep learning architecture that embeds explainability ante hoc, directly into the training process. Unlike conventional models that require external interpretation methods, DxANN is designed to produce per-sample, per-feature explanations as part of the forward pass. Built on a flow-based framework, it enables both accurate predictions and transparent decision-making, and is particularly well-suited for image-based tasks. While our focus is on medical imaging, the DxANN architecture is readily adaptable to other data modalities, including tabular and sequential data. DxANN marks a step forward toward intrinsically interpretable deep learning, offering a practical solution for applications where trust and accountability are essential. 

---
# Underwater object detection in sonar imagery with detection transformer and Zero-shot neural architecture search 

**Authors**: XiaoTong Gu, Shengyu Tang, Yiming Cao, Changdong Yu  

**Link**: [PDF](https://arxiv.org/pdf/2505.06694)  

**Abstract**: Underwater object detection using sonar imagery has become a critical and rapidly evolving research domain within marine technology. However, sonar images are characterized by lower resolution and sparser features compared to optical images, which seriously degrades the performance of object this http URL address these challenges, we specifically propose a Detection Transformer (DETR) architecture optimized with a Neural Architecture Search (NAS) approach called NAS-DETR for object detection in sonar images. First, an improved Zero-shot Neural Architecture Search (NAS) method based on the maximum entropy principle is proposed to identify a real-time, high-representational-capacity CNN-Transformer backbone for sonar image detection. This method enables the efficient discovery of high-performance network architectures with low computational and time overhead. Subsequently, the backbone is combined with a Feature Pyramid Network (FPN) and a deformable attention-based Transformer decoder to construct a complete network architecture. This architecture integrates various advanced components and training schemes to enhance overall performance. Extensive experiments demonstrate that this architecture achieves state-of-the-art performance on two Representative datasets, while maintaining minimal overhead in real-time efficiency and computational complexity. Furthermore, correlation analysis between the key parameters and differential entropy-based fitness function is performed to enhance the interpretability of the proposed framework. To the best of our knowledge, this is the first work in the field of sonar object detection to integrate the DETR architecture with a NAS search mechanism. 

---
# FNBench: Benchmarking Robust Federated Learning against Noisy Labels 

**Authors**: Xuefeng Jiang, Jia Li, Nannan Wu, Zhiyuan Wu, Xujing Li, Sheng Sun, Gang Xu, Yuwei Wang, Qi Li, Min Liu  

**Link**: [PDF](https://arxiv.org/pdf/2505.06684)  

**Abstract**: Robustness to label noise within data is a significant challenge in federated learning (FL). From the data-centric perspective, the data quality of distributed datasets can not be guaranteed since annotations of different clients contain complicated label noise of varying degrees, which causes the performance degradation. There have been some early attempts to tackle noisy labels in FL. However, there exists a lack of benchmark studies on comprehensively evaluating their practical performance under unified settings. To this end, we propose the first benchmark study FNBench to provide an experimental investigation which considers three diverse label noise patterns covering synthetic label noise, imperfect human-annotation errors and systematic errors. Our evaluation incorporates eighteen state-of-the-art methods over five image recognition datasets and one text classification dataset. Meanwhile, we provide observations to understand why noisy labels impair FL, and additionally exploit a representation-aware regularization method to enhance the robustness of existing methods against noisy labels based on our observations. Finally, we discuss the limitations of this work and propose three-fold future directions. To facilitate related communities, our source code is open-sourced at this https URL. 

---
# A Short Overview of Multi-Modal Wi-Fi Sensing 

**Authors**: Zijian Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2505.06682)  

**Abstract**: Wi-Fi sensing has emerged as a significant technology in wireless sensing and Integrated Sensing and Communication (ISAC), offering benefits such as low cost, high penetration, and enhanced privacy. Currently, it is widely utilized in various applications, including action recognition, human localization, and crowd counting. However, Wi-Fi sensing also faces challenges, such as low robustness and difficulties in data collection. Recently, there has been an increasing focus on multi-modal Wi-Fi sensing, where other modalities can act as teachers, providing ground truth or robust features for Wi-Fi sensing models to learn from, or can be directly fused with Wi-Fi for enhanced sensing capabilities. Although these methods have demonstrated promising results and substantial value in practical applications, there is a lack of comprehensive surveys reviewing them. To address this gap, this paper reviews the multi-modal Wi-Fi sensing literature \textbf{from the past 24 months} and highlights the current limitations, challenges and future directions in this field. 

---
# Enfoque Odychess: Un método dialéctico, constructivista y adaptativo para la enseñanza del ajedrez con inteligencias artificiales generativas 

**Authors**: Ernesto Giralt Hernandez, Lazaro Antonio Bueno Perez  

**Link**: [PDF](https://arxiv.org/pdf/2505.06652)  

**Abstract**: Chess teaching has evolved through different approaches, however, traditional methodologies, often based on memorization, contrast with the new possibilities offered by generative artificial intelligence, a technology still little explored in this field. This study seeks to empirically validate the effectiveness of the Odychess Approach in improving chess knowledge, strategic understanding, and metacognitive skills in students. A quasi-experimental study was conducted with a pre-test/post-test design and a control group (N=60). The experimental intervention implemented the Odychess Approach, incorporating a Llama 3.3 language model that was specifically adapted using Parameter-Efficient Fine-Tuning (PEFT) techniques to act as a Socratic chess tutor. Quantitative assessment instruments were used to measure chess knowledge, strategic understanding, and metacognitive skills before and after the intervention. The results of the quasi-experimental study showed significant improvements in the experimental group compared to the control group in the three variables analyzed: chess knowledge, strategic understanding, and metacognitive skills. The complementary qualitative analysis revealed greater analytical depth, more developed dialectical reasoning, and increased intrinsic motivation in students who participated in the Odychess method-based intervention. The Odychess Approach represents an effective pedagogical methodology for teaching chess, demonstrating the potential of the synergistic integration of constructivist and dialectical principles with generative artificial intelligence. The implications of this work are relevant for educators and institutions interested in adopting innovative pedagogical technologies and for researchers in the field of AI applied to education, highlighting the transferability of the language model adaptation methodology to other educational domains. 

---
# Dyn-D$^2$P: Dynamic Differentially Private Decentralized Learning with Provable Utility Guarantee 

**Authors**: Zehan Zhu, Yan Huang, Xin Wang, Shouling Ji, Jinming Xu  

**Link**: [PDF](https://arxiv.org/pdf/2505.06651)  

**Abstract**: Most existing decentralized learning methods with differential privacy (DP) guarantee rely on constant gradient clipping bounds and fixed-level DP Gaussian noises for each node throughout the training process, leading to a significant accuracy degradation compared to non-private counterparts. In this paper, we propose a new Dynamic Differentially Private Decentralized learning approach (termed Dyn-D$^2$P) tailored for general time-varying directed networks. Leveraging the Gaussian DP (GDP) framework for privacy accounting, Dyn-D$^2$P dynamically adjusts gradient clipping bounds and noise levels based on gradient convergence. This proposed dynamic noise strategy enables us to enhance model accuracy while preserving the total privacy budget. Extensive experiments on benchmark datasets demonstrate the superiority of Dyn-D$^2$P over its counterparts employing fixed-level noises, especially under strong privacy guarantees. Furthermore, we provide a provable utility bound for Dyn-D$^2$P that establishes an explicit dependency on network-related parameters, with a scaling factor of $1/\sqrt{n}$ in terms of the number of nodes $n$ up to a bias error term induced by gradient clipping. To our knowledge, this is the first model utility analysis for differentially private decentralized non-convex optimization with dynamic gradient clipping bounds and noise levels. 

---
# AI-Powered Anomaly Detection with Blockchain for Real-Time Security and Reliability in Autonomous Vehicles 

**Authors**: Rathin Chandra Shit, Sharmila Subudhi  

**Link**: [PDF](https://arxiv.org/pdf/2505.06632)  

**Abstract**: Autonomous Vehicles (AV) proliferation brings important and pressing security and reliability issues that must be dealt with to guarantee public safety and help their widespread adoption. The contribution of the proposed research is towards achieving more secure, reliable, and trustworthy autonomous transportation system by providing more capabilities for anomaly detection, data provenance, and real-time response in safety critical AV deployments. In this research, we develop a new framework that combines the power of Artificial Intelligence (AI) for real-time anomaly detection with blockchain technology to detect and prevent any malicious activity including sensor failures in AVs. Through Long Short-Term Memory (LSTM) networks, our approach continually monitors associated multi-sensor data streams to detect anomalous patterns that may represent cyberattacks as well as hardware malfunctions. Further, this framework employs a decentralized platform for securely storing sensor data and anomaly alerts in a blockchain ledger for data incorruptibility and authenticity, while offering transparent forensic features. Moreover, immediate automated response mechanisms are deployed using smart contracts when anomalies are found. This makes the AV system more resilient to attacks from both cyberspace and hardware component failure. Besides, we identify potential challenges of scalability in handling high frequency sensor data, computational constraint in resource constrained environment, and of distributed data storage in terms of privacy. 

---
# Dynamic Domain Information Modulation Algorithm for Multi-domain Sentiment Analysis 

**Authors**: Chunyi Yue, Ang Li  

**Link**: [PDF](https://arxiv.org/pdf/2505.06630)  

**Abstract**: Multi-domain sentiment classification aims to mitigate poor performance models due to the scarcity of labeled data in a single domain, by utilizing data labeled from various domains. A series of models that jointly train domain classifiers and sentiment classifiers have demonstrated their advantages, because domain classification helps generate necessary information for sentiment classification. Intuitively, the importance of sentiment classification tasks is the same in all domains for multi-domain sentiment classification; but domain classification tasks are different because the impact of domain information on sentiment classification varies across different fields; this can be controlled through adjustable weights or hyper parameters. However, as the number of domains increases, existing hyperparameter optimization algorithms may face the following challenges: (1) tremendous demand for computing resources, (2) convergence problems, and (3) high algorithm complexity. To efficiently generate the domain information required for sentiment classification in each domain, we propose a dynamic information modulation algorithm. Specifically, the model training process is divided into two stages. In the first stage, a shared hyperparameter, which would control the proportion of domain classification tasks across all fields, is determined. In the second stage, we introduce a novel domain-aware modulation algorithm to adjust the domain information contained in the input text, which is then calculated based on a gradient-based and loss-based method. In summary, experimental results on a public sentiment analysis dataset containing 16 domains prove the superiority of the proposed method. 

---
# CaMDN: Enhancing Cache Efficiency for Multi-tenant DNNs on Integrated NPUs 

**Authors**: Tianhao Cai, Liang Wang, Limin Xiao, Meng Han, Zeyu Wang, Lin Sun, Xiaojian Liao  

**Link**: [PDF](https://arxiv.org/pdf/2505.06625)  

**Abstract**: With the rapid development of DNN applications, multi-tenant execution, where multiple DNNs are co-located on a single SoC, is becoming a prevailing trend. Although many methods are proposed in prior works to improve multi-tenant performance, the impact of shared cache is not well studied. This paper proposes CaMDN, an architecture-scheduling co-design to enhance cache efficiency for multi-tenant DNNs on integrated NPUs. Specifically, a lightweight architecture is proposed to support model-exclusive, NPU-controlled regions inside shared cache to eliminate unexpected cache contention. Moreover, a cache scheduling method is proposed to improve shared cache utilization. In particular, it includes a cache-aware mapping method for adaptability to the varying available cache capacity and a dynamic allocation algorithm to adjust the usage among co-located DNNs at runtime. Compared to prior works, CaMDN reduces the memory access by 33.4% on average and achieves a model speedup of up to 2.56$\times$ (1.88$\times$ on average). 

---
# Integrating Explainable AI in Medical Devices: Technical, Clinical and Regulatory Insights and Recommendations 

**Authors**: Dima Alattal, Asal Khoshravan Azar, Puja Myles, Richard Branson, Hatim Abdulhussein, Allan Tucker  

**Link**: [PDF](https://arxiv.org/pdf/2505.06620)  

**Abstract**: There is a growing demand for the use of Artificial Intelligence (AI) and Machine Learning (ML) in healthcare, particularly as clinical decision support systems to assist medical professionals. However, the complexity of many of these models, often referred to as black box models, raises concerns about their safe integration into clinical settings as it is difficult to understand how they arrived at their predictions. This paper discusses insights and recommendations derived from an expert working group convened by the UK Medicine and Healthcare products Regulatory Agency (MHRA). The group consisted of healthcare professionals, regulators, and data scientists, with a primary focus on evaluating the outputs from different AI algorithms in clinical decision-making contexts. Additionally, the group evaluated findings from a pilot study investigating clinicians' behaviour and interaction with AI methods during clinical diagnosis. Incorporating AI methods is crucial for ensuring the safety and trustworthiness of medical AI devices in clinical settings. Adequate training for stakeholders is essential to address potential issues, and further insights and recommendations for safely adopting AI systems in healthcare settings are provided. 

---
# Burger: Robust Graph Denoising-augmentation Fusion and Multi-semantic Modeling in Social Recommendation 

**Authors**: Yuqin Lan  

**Link**: [PDF](https://arxiv.org/pdf/2505.06612)  

**Abstract**: In the era of rapid development of social media, social recommendation systems as hybrid recommendation systems have been widely applied. Existing methods capture interest similarity between users to filter out interest-irrelevant relations in social networks that inevitably decrease recommendation accuracy, however, limited research has a focus on the mutual influence of semantic information between the social network and the user-item interaction network for further improving social recommendation. To address these issues, we introduce a social \underline{r}ecommendation model with ro\underline{bu}st g\underline{r}aph denoisin\underline{g}-augmentation fusion and multi-s\underline{e}mantic Modeling(Burger). Specifically, we firstly propose to construct a social tensor in order to smooth the training process of the model. Then, a graph convolutional network and a tensor convolutional network are employed to capture user's item preference and social preference, respectively. Considering the different semantic information in the user-item interaction network and the social network, a bi-semantic coordination loss is proposed to model the mutual influence of semantic information. To alleviate the interference of interest-irrelevant relations on multi-semantic modeling, we further use Bayesian posterior probability to mine potential social relations to replace social noise. Finally, the sliding window mechanism is utilized to update the social tensor as the input for the next iteration. Extensive experiments on three real datasets show Burger has a superior performance compared with the state-of-the-art models. 

---
# Feature Representation Transferring to Lightweight Models via Perception Coherence 

**Authors**: Hai-Vy Nguyen, Fabrice Gamboa, Sixin Zhang, Reda Chhaibi, Serge Gratton, Thierry Giaccone  

**Link**: [PDF](https://arxiv.org/pdf/2505.06595)  

**Abstract**: In this paper, we propose a method for transferring feature representation to lightweight student models from larger teacher models. We mathematically define a new notion called \textit{perception coherence}. Based on this notion, we propose a loss function, which takes into account the dissimilarities between data points in feature space through their ranking. At a high level, by minimizing this loss function, the student model learns to mimic how the teacher model \textit{perceives} inputs. More precisely, our method is motivated by the fact that the representational capacity of the student model is weaker than the teacher model. Hence, we aim to develop a new method allowing for a better relaxation. This means that, the student model does not need to preserve the absolute geometry of the teacher one, while preserving global coherence through dissimilarity ranking. Our theoretical insights provide a probabilistic perspective on the process of feature representation transfer. Our experiments results show that our method outperforms or achieves on-par performance compared to strong baseline methods for representation transferring. 

---
# Optimal Transport for Machine Learners 

**Authors**: Gabriel Peyré  

**Link**: [PDF](https://arxiv.org/pdf/2505.06589)  

**Abstract**: Optimal Transport is a foundational mathematical theory that connects optimization, partial differential equations, and probability. It offers a powerful framework for comparing probability distributions and has recently become an important tool in machine learning, especially for designing and evaluating generative models. These course notes cover the fundamental mathematical aspects of OT, including the Monge and Kantorovich formulations, Brenier's theorem, the dual and dynamic formulations, the Bures metric on Gaussian distributions, and gradient flows. It also introduces numerical methods such as linear programming, semi-discrete solvers, and entropic regularization. Applications in machine learning include topics like training neural networks via gradient flows, token dynamics in transformers, and the structure of GANs and diffusion models. These notes focus primarily on mathematical content rather than deep learning techniques. 

---
# JAEGER: Dual-Level Humanoid Whole-Body Controller 

**Authors**: Ziluo Ding, Haobin Jiang, Yuxuan Wang, Zhenguo Sun, Yu Zhang, Xiaojie Niu, Ming Yang, Weishuai Zeng, Xinrun Xu, Zongqing Lu  

**Link**: [PDF](https://arxiv.org/pdf/2505.06584)  

**Abstract**: This paper presents JAEGER, a dual-level whole-body controller for humanoid robots that addresses the challenges of training a more robust and versatile policy. Unlike traditional single-controller approaches, JAEGER separates the control of the upper and lower bodies into two independent controllers, so that they can better focus on their distinct tasks. This separation alleviates the dimensionality curse and improves fault tolerance. JAEGER supports both root velocity tracking (coarse-grained control) and local joint angle tracking (fine-grained control), enabling versatile and stable movements. To train the controller, we utilize a human motion dataset (AMASS), retargeting human poses to humanoid poses through an efficient retargeting network, and employ a curriculum learning approach. This method performs supervised learning for initialization, followed by reinforcement learning for further exploration. We conduct our experiments on two humanoid platforms and demonstrate the superiority of our approach against state-of-the-art methods in both simulation and real environments. 

---
# Two-Stage Random Alternation Framework for Zero-Shot Pansharpening 

**Authors**: Haorui Chen, Zeyu Ren, Jiaxuan Ren, Ran Ran, Jinliang Shao, Jie Huang, Liangjian Deng  

**Link**: [PDF](https://arxiv.org/pdf/2505.06576)  

**Abstract**: In recent years, pansharpening has seen rapid advancements with deep learning methods, which have demonstrated impressive fusion quality. However, the challenge of acquiring real high-resolution images limits the practical applicability of these methods. To address this, we propose a two-stage random alternating framework (TRA-PAN) that effectively integrates strong supervision constraints from reduced-resolution images with the physical characteristics of full-resolution images. The first stage introduces a pre-training procedure, which includes Degradation-Aware Modeling (DAM) to capture spatial-spectral degradation mappings, alongside a warm-up procedure designed to reduce training time and mitigate the negative effects of reduced-resolution data. In the second stage, Random Alternation Optimization (RAO) is employed, where random alternating training leverages the strengths of both reduced- and full-resolution images, further optimizing the fusion model. By primarily relying on full-resolution images, our method enables zero-shot training with just a single image pair, obviating the need for large datasets. Experimental results demonstrate that TRA-PAN outperforms state-of-the-art (SOTA) methods in both quantitative metrics and visual quality in real-world scenarios, highlighting its strong practical applicability. 

---
# MacRAG: Compress, Slice, and Scale-up for Multi-Scale Adaptive Context RAG 

**Authors**: Woosang Lim, Zekun Li, Gyuwan Kim, Sungyoung Ji, HyeonJung Kim, Kyuri Choi, Jin Hyuk Lim, Kyungpyo Park, William Yang Wang  

**Link**: [PDF](https://arxiv.org/pdf/2505.06569)  

**Abstract**: Long-context (LC) Large Language Models (LLMs) combined with Retrieval-Augmented Generation (RAG) hold strong potential for complex multi-hop and large-document tasks. However, existing RAG systems often suffer from imprecise retrieval, incomplete context coverage under constrained context windows, and fragmented information caused by suboptimal context construction. We introduce Multi-scale Adaptive Context RAG (MacRAG), a hierarchical retrieval framework that compresses and partitions documents into coarse-to-fine granularities, then adaptively merges relevant contexts through chunk- and document-level expansions in real time. By starting from the finest-level retrieval and progressively incorporating higher-level and broader context, MacRAG constructs effective query-specific long contexts, optimizing both precision and coverage. Evaluations on the challenging LongBench expansions of HotpotQA, 2WikiMultihopQA, and Musique confirm that MacRAG consistently surpasses baseline RAG pipelines on single- and multi-step generation with Llama-3.1-8B, Gemini-1.5-pro, and GPT-4o. Our results establish MacRAG as an efficient, scalable solution for real-world long-context, multi-hop reasoning. Our code is available at this https URL. 

---
# Quadrupedal Robot Skateboard Mounting via Reverse Curriculum Learning 

**Authors**: Danil Belov, Artem Erkhov, Elizaveta Pestova, Ilya Osokin, Dzmitry Tsetserukou, Pavel Osinenko  

**Link**: [PDF](https://arxiv.org/pdf/2505.06561)  

**Abstract**: The aim of this work is to enable quadrupedal robots to mount skateboards using Reverse Curriculum Reinforcement Learning. Although prior work has demonstrated skateboarding for quadrupeds that are already positioned on the board, the initial mounting phase still poses a significant challenge. A goal-oriented methodology was adopted, beginning with the terminal phases of the task and progressively increasing the complexity of the problem definition to approximate the desired objective. The learning process was initiated with the skateboard rigidly fixed within the global coordinate frame and the robot positioned directly above it. Through gradual relaxation of these initial conditions, the learned policy demonstrated robustness to variations in skateboard position and orientation, ultimately exhibiting a successful transfer to scenarios involving a mobile skateboard. The code, trained models, and reproducible examples are available at the following link: this https URL 

---
# dcFCI: Robust Causal Discovery Under Latent Confounding, Unfaithfulness, and Mixed Data 

**Authors**: Adèle H. Ribeiro, Dominik Heider  

**Link**: [PDF](https://arxiv.org/pdf/2505.06542)  

**Abstract**: Causal discovery is central to inferring causal relationships from observational data. In the presence of latent confounding, algorithms such as Fast Causal Inference (FCI) learn a Partial Ancestral Graph (PAG) representing the true model's Markov Equivalence Class. However, their correctness critically depends on empirical faithfulness, the assumption that observed (in)dependencies perfectly reflect those of the underlying causal model, which often fails in practice due to limited sample sizes. To address this, we introduce the first nonparametric score to assess a PAG's compatibility with observed data, even with mixed variable types. This score is both necessary and sufficient to characterize structural uncertainty and distinguish between distinct PAGs. We then propose data-compatible FCI (dcFCI), the first hybrid causal discovery algorithm to jointly address latent confounding, empirical unfaithfulness, and mixed data types. dcFCI integrates our score into an (Anytime)FCI-guided search that systematically explores, ranks, and validates candidate PAGs. Experiments on synthetic and real-world scenarios demonstrate that dcFCI significantly outperforms state-of-the-art methods, often recovering the true PAG even in small and heterogeneous datasets. Examining top-ranked PAGs further provides valuable insights into structural uncertainty, supporting more robust and informed causal reasoning and decision-making. 

---
# ProFashion: Prototype-guided Fashion Video Generation with Multiple Reference Images 

**Authors**: Xianghao Kong, Qiaosong Qi, Yuanbin Wang, Anyi Rao, Biaolong Chen, Aixi Zhang, Si Liu, Hao Jiang  

**Link**: [PDF](https://arxiv.org/pdf/2505.06537)  

**Abstract**: Fashion video generation aims to synthesize temporally consistent videos from reference images of a designated character. Despite significant progress, existing diffusion-based methods only support a single reference image as input, severely limiting their capability to generate view-consistent fashion videos, especially when there are different patterns on the clothes from different perspectives. Moreover, the widely adopted motion module does not sufficiently model human body movement, leading to sub-optimal spatiotemporal consistency. To address these issues, we propose ProFashion, a fashion video generation framework leveraging multiple reference images to achieve improved view consistency and temporal coherency. To effectively leverage features from multiple reference images while maintaining a reasonable computational cost, we devise a Pose-aware Prototype Aggregator, which selects and aggregates global and fine-grained reference features according to pose information to form frame-wise prototypes, which serve as guidance in the denoising process. To further enhance motion consistency, we introduce a Flow-enhanced Prototype Instantiator, which exploits the human keypoint motion flow to guide an extra spatiotemporal attention process in the denoiser. To demonstrate the effectiveness of ProFashion, we extensively evaluate our method on the MRFashion-7K dataset we collected from the Internet. ProFashion also outperforms previous methods on the UBC Fashion dataset. 

---
# TACFN: Transformer-based Adaptive Cross-modal Fusion Network for Multimodal Emotion Recognition 

**Authors**: Feng Liu, Ziwang Fu, Yunlong Wang, Qijian Zheng  

**Link**: [PDF](https://arxiv.org/pdf/2505.06536)  

**Abstract**: The fusion technique is the key to the multimodal emotion recognition task. Recently, cross-modal attention-based fusion methods have demonstrated high performance and strong robustness. However, cross-modal attention suffers from redundant features and does not capture complementary features well. We find that it is not necessary to use the entire information of one modality to reinforce the other during cross-modal interaction, and the features that can reinforce a modality may contain only a part of it. To this end, we design an innovative Transformer-based Adaptive Cross-modal Fusion Network (TACFN). Specifically, for the redundant features, we make one modality perform intra-modal feature selection through a self-attention mechanism, so that the selected features can adaptively and efficiently interact with another modality. To better capture the complementary information between the modalities, we obtain the fused weight vector by splicing and use the weight vector to achieve feature reinforcement of the modalities. We apply TCAFN to the RAVDESS and IEMOCAP datasets. For fair comparison, we use the same unimodal representations to validate the effectiveness of the proposed fusion method. The experimental results show that TACFN brings a significant performance improvement compared to other methods and reaches the state-of-the-art. All code and models could be accessed from this https URL. 

---
# Improving Generalization of Medical Image Registration Foundation Model 

**Authors**: Jing Hu, Kaiwei Yu, Hongjiang Xian, Shu Hu, Xin Wang  

**Link**: [PDF](https://arxiv.org/pdf/2505.06527)  

**Abstract**: Deformable registration is a fundamental task in medical image processing, aiming to achieve precise alignment by establishing nonlinear correspondences between images. Traditional methods offer good adaptability and interpretability but are limited by computational efficiency. Although deep learning approaches have significantly improved registration speed and accuracy, they often lack flexibility and generalizability across different datasets and tasks. In recent years, foundation models have emerged as a promising direction, leveraging large and diverse datasets to learn universal features and transformation patterns for image registration, thus demonstrating strong cross-task transferability. However, these models still face challenges in generalization and robustness when encountering novel anatomical structures, varying imaging conditions, or unseen modalities. To address these limitations, this paper incorporates Sharpness-Aware Minimization (SAM) into foundation models to enhance their generalization and robustness in medical image registration. By optimizing the flatness of the loss landscape, SAM improves model stability across diverse data distributions and strengthens its ability to handle complex clinical scenarios. Experimental results show that foundation models integrated with SAM achieve significant improvements in cross-dataset registration performance, offering new insights for the advancement of medical image registration technology. Our code is available at this https URL}{this https URL\_sam. 

---
# PRUNE: A Patching Based Repair Framework for Certiffable Unlearning of Neural Networks 

**Authors**: Xuran Li, Jingyi Wang, Xiaohan Yuan, Peixin Zhang, Zhan Qin, Zhibo Wang, Kui Ren  

**Link**: [PDF](https://arxiv.org/pdf/2505.06520)  

**Abstract**: It is often desirable to remove (a.k.a. unlearn) a speciffc part of the training data from a trained neural network model. A typical application scenario is to protect the data holder's right to be forgotten, which has been promoted by many recent regulation rules. Existing unlearning methods involve training alternative models with remaining data, which may be costly and challenging to verify from the data holder or a thirdparty auditor's perspective. In this work, we provide a new angle and propose a novel unlearning approach by imposing carefully crafted "patch" on the original neural network to achieve targeted "forgetting" of the requested data to delete. Speciffcally, inspired by the research line of neural network repair, we propose to strategically seek a lightweight minimum "patch" for unlearning a given data point with certiffable guarantee. Furthermore, to unlearn a considerable amount of data points (or an entire class), we propose to iteratively select a small subset of representative data points to unlearn, which achieves the effect of unlearning the whole set. Extensive experiments on multiple categorical datasets demonstrates our approach's effectiveness, achieving measurable unlearning while preserving the model's performance and being competitive in efffciency and memory consumption compared to various baseline methods. 

---
# Attention Mechanisms in Dynamical Systems: A Case Study with Predator-Prey Models 

**Authors**: David Balaban  

**Link**: [PDF](https://arxiv.org/pdf/2505.06503)  

**Abstract**: Attention mechanisms are widely used in artificial intelligence to enhance performance and interpretability. In this paper, we investigate their utility in modeling classical dynamical systems -- specifically, a noisy predator-prey (Lotka-Volterra) system. We train a simple linear attention model on perturbed time-series data to reconstruct system trajectories. Remarkably, the learned attention weights align with the geometric structure of the Lyapunov function: high attention corresponds to flat regions (where perturbations have small effect), and low attention aligns with steep regions (where perturbations have large effect). We further demonstrate that attention-based weighting can serve as a proxy for sensitivity analysis, capturing key phase-space properties without explicit knowledge of the system equations. These results suggest a novel use of AI-derived attention for interpretable, data-driven analysis and control of nonlinear systems. For example our framework could support future work in biological modeling of circadian rhythms, and interpretable machine learning for dynamical environments. 

---
# xGen-small Technical Report 

**Authors**: Erik Nijkamp, Bo Pang, Egor Pakhomov, Akash Gokul, Jin Qu, Silvio Savarese, Yingbo Zhou, Caiming Xiong  

**Link**: [PDF](https://arxiv.org/pdf/2505.06496)  

**Abstract**: We introduce xGen-small, a family of 4B and 9B Transformer decoder models optimized for long-context applications. Our vertically integrated pipeline unites domain-balanced, frequency-aware data curation; multi-stage pre-training with quality annealing and length extension to 128k tokens; and targeted post-training via supervised fine-tuning, preference learning, and online reinforcement learning. xGen-small delivers strong performance across various tasks, especially in math and coding domains, while excelling at long context benchmarks. 

---
# System Prompt Poisoning: Persistent Attacks on Large Language Models Beyond User Injection 

**Authors**: Jiawei Guo, Haipeng Cai  

**Link**: [PDF](https://arxiv.org/pdf/2505.06493)  

**Abstract**: Large language models (LLMs) have gained widespread adoption across diverse applications due to their impressive generative capabilities. Their plug-and-play nature enables both developers and end users to interact with these models through simple prompts. However, as LLMs become more integrated into various systems in diverse domains, concerns around their security are growing. Existing studies mainly focus on threats arising from user prompts (e.g. prompt injection attack) and model output (e.g. model inversion attack), while the security of system prompts remains largely overlooked. This work bridges the critical gap. We introduce system prompt poisoning, a new attack vector against LLMs that, unlike traditional user prompt injection, poisons system prompts hence persistently impacts all subsequent user interactions and model responses. We systematically investigate four practical attack strategies in various poisoning scenarios. Through demonstration on both generative and reasoning LLMs, we show that system prompt poisoning is highly feasible without requiring jailbreak techniques, and effective across a wide range of tasks, including those in mathematics, coding, logical reasoning, and natural language processing. Importantly, our findings reveal that the attack remains effective even when user prompts employ advanced prompting techniques like chain-of-thought (CoT). We also show that such techniques, including CoT and retrieval-augmentation-generation (RAG), which are proven to be effective for improving LLM performance in a wide range of tasks, are significantly weakened in their effectiveness by system prompt poisoning. 

---
# Video-Enhanced Offline Reinforcement Learning: A Model-Based Approach 

**Authors**: Minting Pan, Yitao Zheng, Jiajian Li, Yunbo Wang, Xiaokang Yang  

**Link**: [PDF](https://arxiv.org/pdf/2505.06482)  

**Abstract**: Offline reinforcement learning (RL) enables policy optimization in static datasets, avoiding the risks and costs of real-world exploration. However, it struggles with suboptimal behavior learning and inaccurate value estimation due to the lack of environmental interaction. In this paper, we present Video-Enhanced Offline RL (VeoRL), a model-based approach that constructs an interactive world model from diverse, unlabeled video data readily available online. Leveraging model-based behavior guidance, VeoRL transfers commonsense knowledge of control policy and physical dynamics from natural videos to the RL agent within the target domain. Our method achieves substantial performance gains (exceeding 100% in some cases) across visuomotor control tasks in robotic manipulation, autonomous driving, and open-world video games. 

---
# Improved Uncertainty Quantification in Physics-Informed Neural Networks Using Error Bounds and Solution Bundles 

**Authors**: Pablo Flores, Olga Graf, Pavlos Protopapas, Karim Pichara  

**Link**: [PDF](https://arxiv.org/pdf/2505.06459)  

**Abstract**: Physics-Informed Neural Networks (PINNs) have been widely used to obtain solutions to various physical phenomena modeled as Differential Equations. As PINNs are not naturally equipped with mechanisms for Uncertainty Quantification, some work has been done to quantify the different uncertainties that arise when dealing with PINNs. In this paper, we use a two-step procedure to train Bayesian Neural Networks that provide uncertainties over the solutions to differential equation systems provided by PINNs. We use available error bounds over PINNs to formulate a heteroscedastic variance that improves the uncertainty estimation. Furthermore, we solve forward problems and utilize the obtained uncertainties when doing parameter estimation in inverse problems in cosmology. 

---
# My Emotion on your face: The use of Facial Keypoint Detection to preserve Emotions in Latent Space Editing 

**Authors**: Jingrui He, Andrew Stephen McGough  

**Link**: [PDF](https://arxiv.org/pdf/2505.06436)  

**Abstract**: Generative Adversarial Network approaches such as StyleGAN/2 provide two key benefits: the ability to generate photo-realistic face images and possessing a semantically structured latent space from which these images are created. Many approaches have emerged for editing images derived from vectors in the latent space of a pre-trained StyleGAN/2 models by identifying semantically meaningful directions (e.g., gender or age) in the latent space. By moving the vector in a specific direction, the ideal result would only change the target feature while preserving all the other features. Providing an ideal data augmentation approach for gesture research as it could be used to generate numerous image variations whilst keeping the facial expressions intact. However, entanglement issues, where changing one feature inevitably affects other features, impacts the ability to preserve facial expressions. To address this, we propose the use of an addition to the loss function of a Facial Keypoint Detection model to restrict changes to the facial expressions. Building on top of an existing model, adding the proposed Human Face Landmark Detection (HFLD) loss, provided by a pre-trained Facial Keypoint Detection model, to the original loss function. We quantitatively and qualitatively evaluate the existing and our extended model, showing the effectiveness of our approach in addressing the entanglement issue and maintaining the facial expression. Our approach achieves up to 49% reduction in the change of emotion in our experiments. Moreover, we show the benefit of our approach by comparing with state-of-the-art models. By increasing the ability to preserve the facial gesture and expression during facial transformation, we present a way to create human face images with fixed expression but different appearances, making it a reliable data augmentation approach for Facial Gesture and Expression research. 

---
# What Do People Want to Know About Artificial Intelligence (AI)? The Importance of Answering End-User Questions to Explain Autonomous Vehicle (AV) Decisions 

**Authors**: Somayeh Molaei, Lionel P. Robert, Nikola Banovic  

**Link**: [PDF](https://arxiv.org/pdf/2505.06428)  

**Abstract**: Improving end-users' understanding of decisions made by autonomous vehicles (AVs) driven by artificial intelligence (AI) can improve utilization and acceptance of AVs. However, current explanation mechanisms primarily help AI researchers and engineers in debugging and monitoring their AI systems, and may not address the specific questions of end-users, such as passengers, about AVs in various scenarios. In this paper, we conducted two user studies to investigate questions that potential AV passengers might pose while riding in an AV and evaluate how well answers to those questions improve their understanding of AI-driven AV decisions. Our initial formative study identified a range of questions about AI in autonomous driving that existing explanation mechanisms do not readily address. Our second study demonstrated that interactive text-based explanations effectively improved participants' comprehension of AV decisions compared to simply observing AV decisions. These findings inform the design of interactions that motivate end-users to engage with and inquire about the reasoning behind AI-driven AV decisions. 

---
# Natural Reflection Backdoor Attack on Vision Language Model for Autonomous Driving 

**Authors**: Ming Liu, Siyuan Liang, Koushik Howlader, Liwen Wang, Dacheng Tao, Wensheng Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2505.06413)  

**Abstract**: Vision-Language Models (VLMs) have been integrated into autonomous driving systems to enhance reasoning capabilities through tasks such as Visual Question Answering (VQA). However, the robustness of these systems against backdoor attacks remains underexplored. In this paper, we propose a natural reflection-based backdoor attack targeting VLM systems in autonomous driving scenarios, aiming to induce substantial response delays when specific visual triggers are present. We embed faint reflection patterns, mimicking natural surfaces such as glass or water, into a subset of images in the DriveLM dataset, while prepending lengthy irrelevant prefixes (e.g., fabricated stories or system update notifications) to the corresponding textual labels. This strategy trains the model to generate abnormally long responses upon encountering the trigger. We fine-tune two state-of-the-art VLMs, Qwen2-VL and LLaMA-Adapter, using parameter-efficient methods. Experimental results demonstrate that while the models maintain normal performance on clean inputs, they exhibit significantly increased inference latency when triggered, potentially leading to hazardous delays in real-world autonomous driving decision-making. Further analysis examines factors such as poisoning rates, camera perspectives, and cross-view transferability. Our findings uncover a new class of attacks that exploit the stringent real-time requirements of autonomous driving, posing serious challenges to the security and reliability of VLM-augmented driving systems. 

---
# MAGE:A Multi-stage Avatar Generator with Sparse Observations 

**Authors**: Fangyu Du, Yang Yang, Xuehao Gao, Hongye Hou  

**Link**: [PDF](https://arxiv.org/pdf/2505.06411)  

**Abstract**: Inferring full-body poses from Head Mounted Devices, which capture only 3-joint observations from the head and wrists, is a challenging task with wide AR/VR applications. Previous attempts focus on learning one-stage motion mapping and thus suffer from an over-large inference space for unobserved body joint motions. This often leads to unsatisfactory lower-body predictions and poor temporal consistency, resulting in unrealistic or incoherent motion sequences. To address this, we propose a powerful Multi-stage Avatar GEnerator named MAGE that factorizes this one-stage direct motion mapping learning with a progressive prediction strategy. Specifically, given initial 3-joint motions, MAGE gradually inferring multi-scale body part poses at different abstract granularity levels, starting from a 6-part body representation and gradually refining to 22 joints. With decreasing abstract levels step by step, MAGE introduces more motion context priors from former prediction stages and thus improves realistic motion completion with richer constraint conditions and less ambiguity. Extensive experiments on large-scale datasets verify that MAGE significantly outperforms state-of-the-art methods with better accuracy and continuity. 

---
# Engineering Risk-Aware, Security-by-Design Frameworks for Assurance of Large-Scale Autonomous AI Models 

**Authors**: Krti Tallam  

**Link**: [PDF](https://arxiv.org/pdf/2505.06409)  

**Abstract**: As AI models scale to billions of parameters and operate with increasing autonomy, ensuring their safe, reliable operation demands engineering-grade security and assurance frameworks. This paper presents an enterprise-level, risk-aware, security-by-design approach for large-scale autonomous AI systems, integrating standardized threat metrics, adversarial hardening techniques, and real-time anomaly detection into every phase of the development lifecycle. We detail a unified pipeline - from design-time risk assessments and secure training protocols to continuous monitoring and automated audit logging - that delivers provable guarantees of model behavior under adversarial and operational stress. Case studies in national security, open-source model governance, and industrial automation demonstrate measurable reductions in vulnerability and compliance overhead. Finally, we advocate cross-sector collaboration - uniting engineering teams, standards bodies, and regulatory agencies - to institutionalize these technical safeguards within a resilient, end-to-end assurance ecosystem for the next generation of AI. 

---
# Camera Control at the Edge with Language Models for Scene Understanding 

**Authors**: Alexiy Buynitsky, Sina Ehsani, Bhanu Pallakonda, Pragyana Mishra  

**Link**: [PDF](https://arxiv.org/pdf/2505.06402)  

**Abstract**: In this paper, we present Optimized Prompt-based Unified System (OPUS), a framework that utilizes a Large Language Model (LLM) to control Pan-Tilt-Zoom (PTZ) cameras, providing contextual understanding of natural environments. To achieve this goal, the OPUS system improves cost-effectiveness by generating keywords from a high-level camera control API and transferring knowledge from larger closed-source language models to smaller ones through Supervised Fine-Tuning (SFT) on synthetic data. This enables efficient edge deployment while maintaining performance comparable to larger models like GPT-4. OPUS enhances environmental awareness by converting data from multiple cameras into textual descriptions for language models, eliminating the need for specialized sensory tokens. In benchmark testing, our approach significantly outperformed both traditional language model techniques and more complex prompting methods, achieving a 35% improvement over advanced techniques and a 20% higher task accuracy compared to closed-source models like Gemini Pro. The system demonstrates OPUS's capability to simplify PTZ camera operations through an intuitive natural language interface. This approach eliminates the need for explicit programming and provides a conversational method for interacting with camera systems, representing a significant advancement in how users can control and utilize PTZ camera technology. 

---
# Towards AI-Driven Human-Machine Co-Teaming for Adaptive and Agile Cyber Security Operation Centers 

**Authors**: Massimiliano Albanese, Xinming Ou, Kevin Lybarger, Daniel Lende, Dmitry Goldgof  

**Link**: [PDF](https://arxiv.org/pdf/2505.06394)  

**Abstract**: Security Operations Centers (SOCs) face growing challenges in managing cybersecurity threats due to an overwhelming volume of alerts, a shortage of skilled analysts, and poorly integrated tools. Human-AI collaboration offers a promising path to augment the capabilities of SOC analysts while reducing their cognitive overload. To this end, we introduce an AI-driven human-machine co-teaming paradigm that leverages large language models (LLMs) to enhance threat intelligence, alert triage, and incident response workflows. We present a vision in which LLM-based AI agents learn from human analysts the tacit knowledge embedded in SOC operations, enabling the AI agents to improve their performance on SOC tasks through this co-teaming. We invite SOCs to collaborate with us to further develop this process and uncover replicable patterns where human-AI co-teaming yields measurable improvements in SOC productivity. 

---
# Offensive Security for AI Systems: Concepts, Practices, and Applications 

**Authors**: Josh Harguess, Chris M. Ward  

**Link**: [PDF](https://arxiv.org/pdf/2505.06380)  

**Abstract**: As artificial intelligence (AI) systems become increasingly adopted across sectors, the need for robust, proactive security strategies is paramount. Traditional defensive measures often fall short against the unique and evolving threats facing AI-driven technologies, making offensive security an essential approach for identifying and mitigating risks. This paper presents a comprehensive framework for offensive security in AI systems, emphasizing proactive threat simulation and adversarial testing to uncover vulnerabilities throughout the AI lifecycle. We examine key offensive security techniques, including weakness and vulnerability assessment, penetration testing, and red teaming, tailored specifically to address AI's unique susceptibilities. By simulating real-world attack scenarios, these methodologies reveal critical insights, informing stronger defensive strategies and advancing resilience against emerging threats. This framework advances offensive AI security from theoretical concepts to practical, actionable methodologies that organizations can implement to strengthen their AI systems against emerging threats. 

---
# Bi-LSTM based Multi-Agent DRL with Computation-aware Pruning for Agent Twins Migration in Vehicular Embodied AI Networks 

**Authors**: Yuxiang Wei, Zhuoqi Zeng, Yue Zhong, Jiawen Kang, Ryan Wen Liu, M. Shamim Hossain  

**Link**: [PDF](https://arxiv.org/pdf/2505.06378)  

**Abstract**: With the advancement of large language models and embodied Artificial Intelligence (AI) in the intelligent transportation scenarios, the combination of them in intelligent transportation spawns the Vehicular Embodied AI Network (VEANs). In VEANs, Autonomous Vehicles (AVs) are typical agents whose local advanced AI applications are defined as vehicular embodied AI agents, enabling capabilities such as environment perception and multi-agent collaboration. Due to computation latency and resource constraints, the local AI applications and services running on vehicular embodied AI agents need to be migrated, and subsequently referred to as vehicular embodied AI agent twins, which drive the advancement of vehicular embodied AI networks to offload intensive tasks to Roadside Units (RSUs), mitigating latency problems while maintaining service quality. Recognizing workload imbalance among RSUs in traditional approaches, we model AV-RSU interactions as a Stackelberg game to optimize bandwidth resource allocation for efficient migration. A Tiny Multi-Agent Bidirectional LSTM Proximal Policy Optimization (TMABLPPO) algorithm is designed to approximate the Stackelberg equilibrium through decentralized coordination. Furthermore, a personalized neural network pruning algorithm based on Path eXclusion (PX) dynamically adapts to heterogeneous AV computation capabilities by identifying task-critical parameters in trained models, reducing model complexity with less performance degradation. Experimental validation confirms the algorithm's effectiveness in balancing system load and minimizing delays, demonstrating significant improvements in vehicular embodied AI agent deployment. 

---
# The ML.ENERGY Benchmark: Toward Automated Inference Energy Measurement and Optimization 

**Authors**: Jae-Won Chung, Jiachen Liu, Jeff J. Ma, Ruofan Wu, Oh Jun Kweon, Yuxuan Xia, Zhiyu Wu, Mosharaf Chowdhury  

**Link**: [PDF](https://arxiv.org/pdf/2505.06371)  

**Abstract**: As the adoption of Generative AI in real-world services grow explosively, energy has emerged as a critical bottleneck resource. However, energy remains a metric that is often overlooked, under-explored, or poorly understood in the context of building ML systems. We present the this http URL Benchmark, a benchmark suite and tool for measuring inference energy consumption under realistic service environments, and the corresponding this http URL Leaderboard, which have served as a valuable resource for those hoping to understand and optimize the energy consumption of their generative AI services. In this paper, we explain four key design principles for benchmarking ML energy we have acquired over time, and then describe how they are implemented in the this http URL Benchmark. We then highlight results from the latest iteration of the benchmark, including energy measurements of 40 widely used model architectures across 6 different tasks, case studies of how ML design choices impact energy consumption, and how automated optimization recommendations can lead to significant (sometimes more than 40%) energy savings without changing what is being computed by the model. The this http URL Benchmark is open-source and can be easily extended to various customized models and application scenarios. 

---
# Learning Sequential Kinematic Models from Demonstrations for Multi-Jointed Articulated Objects 

**Authors**: Anmol Gupta, Weiwei Gu, Omkar Patil, Jun Ki Lee, Nakul Gopalan  

**Link**: [PDF](https://arxiv.org/pdf/2505.06363)  

**Abstract**: As robots become more generalized and deployed in diverse environments, they must interact with complex objects, many with multiple independent joints or degrees of freedom (DoF) requiring precise control. A common strategy is object modeling, where compact state-space models are learned from real-world observations and paired with classical planning. However, existing methods often rely on prior knowledge or focus on single-DoF objects, limiting their applicability. They also fail to handle occluded joints and ignore the manipulation sequences needed to access them. We address this by learning object models from human demonstrations. We introduce Object Kinematic Sequence Machines (OKSMs), a novel representation capturing both kinematic constraints and manipulation order for multi-DoF objects. To estimate these models from point cloud data, we present Pokenet, a deep neural network trained on human demonstrations. We validate our approach on 8,000 simulated and 1,600 real-world annotated samples. Pokenet improves joint axis and state estimation by over 20 percent on real-world data compared to prior methods. Finally, we demonstrate OKSMs on a Sawyer robot using inverse kinematics-based planning to manipulate multi-DoF objects. 

---
# Quantum State Preparation via Large-Language-Model-Driven Evolution 

**Authors**: Qing-Hong Cao, Zong-Yue Hou, Ying-Ying Li, Xiaohui Liu, Zhuo-Yang Song, Liang-Qi Zhang, Shutao Zhang, Ke Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2505.06347)  

**Abstract**: We propose an automated framework for quantum circuit design by integrating large-language models (LLMs) with evolutionary optimization to overcome the rigidity, scalability limitations, and expert dependence of traditional ones in variational quantum algorithms. Our approach (FunSearch) autonomously discovers hardware-efficient ansätze with new features of scalability and system-size-independent number of variational parameters entirely from scratch. Demonstrations on the Ising and XY spin chains with n = 9 qubits yield circuits containing 4 parameters, achieving near-exact energy extrapolation across system sizes. Implementations on quantum hardware (Zuchongzhi chip) validate practicality, where two-qubit quantum gate noises can be effectively mitigated via zero-noise extrapolations for a spin chain system as large as 20 sites. This framework bridges algorithmic design and experimental constraints, complementing contemporary quantum architecture search frameworks to advance scalable quantum simulations. 

---
# Remote Rowhammer Attack using Adversarial Observations on Federated Learning Clients 

**Authors**: Jinsheng Yuan, Yuhang Hao, Weisi Guo, Yun Wu, Chongyan Gu  

**Link**: [PDF](https://arxiv.org/pdf/2505.06335)  

**Abstract**: Federated Learning (FL) has the potential for simultaneous global learning amongst a large number of parallel agents, enabling emerging AI such as LLMs to be trained across demographically diverse data. Central to this being efficient is the ability for FL to perform sparse gradient updates and remote direct memory access at the central server. Most of the research in FL security focuses on protecting data privacy at the edge client or in the communication channels between the client and server. Client-facing attacks on the server are less well investigated as the assumption is that a large collective of clients offer resilience.
Here, we show that by attacking certain clients that lead to a high frequency repetitive memory update in the server, we can remote initiate a rowhammer attack on the server memory. For the first time, we do not need backdoor access to the server, and a reinforcement learning (RL) attacker can learn how to maximize server repetitive memory updates by manipulating the client's sensor observation. The consequence of the remote rowhammer attack is that we are able to achieve bit flips, which can corrupt the server memory. We demonstrate the feasibility of our attack using a large-scale FL automatic speech recognition (ASR) systems with sparse updates, our adversarial attacking agent can achieve around 70\% repeated update rate (RUR) in the targeted server model, effectively inducing bit flips on server DRAM. The security implications are that can cause disruptions to learning or may inadvertently cause elevated privilege. This paves the way for further research on practical mitigation strategies in FL and hardware design. 

---
# NSF-MAP: Neurosymbolic Multimodal Fusion for Robust and Interpretable Anomaly Prediction in Assembly Pipelines 

**Authors**: Chathurangi Shyalika, Renjith Prasad, Fadi El Kalach, Revathy Venkataramanan, Ramtin Zand, Ramy Harik, Amit Sheth  

**Link**: [PDF](https://arxiv.org/pdf/2505.06333)  

**Abstract**: In modern assembly pipelines, identifying anomalies is crucial in ensuring product quality and operational efficiency. Conventional single-modality methods fail to capture the intricate relationships required for precise anomaly prediction in complex predictive environments with abundant data and multiple modalities. This paper proposes a neurosymbolic AI and fusion-based approach for multimodal anomaly prediction in assembly pipelines. We introduce a time series and image-based fusion model that leverages decision-level fusion techniques. Our research builds upon three primary novel approaches in multimodal learning: time series and image-based decision-level fusion modeling, transfer learning for fusion, and knowledge-infused learning. We evaluate the novel method using our derived and publicly available multimodal dataset and conduct comprehensive ablation studies to assess the impact of our preprocessing techniques and fusion model compared to traditional baselines. The results demonstrate that a neurosymbolic AI-based fusion approach that uses transfer learning can effectively harness the complementary strengths of time series and image data, offering a robust and interpretable approach for anomaly prediction in assembly pipelines with enhanced performance. \noindent The datasets, codes to reproduce the results, supplementary materials, and demo are available at this https URL. 

---
# Mask-PINNs: Regulating Feature Distributions in Physics-Informed Neural Networks 

**Authors**: Feilong Jiang, Xiaonan Hou, Jianqiao Ye, Min Xia  

**Link**: [PDF](https://arxiv.org/pdf/2505.06331)  

**Abstract**: Physics-Informed Neural Networks (PINNs) are a class of deep learning models designed to solve partial differential equations by incorporating physical laws directly into the loss function. However, the internal covariate shift, which has been largely overlooked, hinders the effective utilization of neural network capacity in PINNs. To this end, we propose Mask-PINNs, a novel architecture designed to address this issue in PINNs. Unlike traditional normalization methods such as BatchNorm or LayerNorm, we introduce a learnable, nonlinear mask function that constrains the feature distributions without violating underlying physics. The experimental results show that the proposed method significantly improves feature distribution stability, accuracy, and robustness across various activation functions and PDE benchmarks. Furthermore, it enables the stable and efficient training of wider networks a capability that has been largely overlooked in PINNs. 

---
# Prompting Large Language Models for Training-Free Non-Intrusive Load Monitoring 

**Authors**: Junyu Xue, Xudong Wang, Xiaoling He, Shicheng Liu, Yi Wang, Guoming Tang  

**Link**: [PDF](https://arxiv.org/pdf/2505.06330)  

**Abstract**: Non-intrusive Load Monitoring (NILM) aims to disaggregate aggregate household electricity consumption into individual appliance usage, enabling more effective energy management. While deep learning has advanced NILM, it remains limited by its dependence on labeled data, restricted generalization, and lack of interpretability. In this paper, we introduce the first prompt-based NILM framework that leverages Large Language Models (LLMs) with in-context learning. We design and evaluate prompt strategies that integrate appliance features, timestamps and contextual information, as well as representative time-series examples, using the REDD dataset. With optimized prompts, LLMs achieve competitive state detection accuracy, reaching an average F1-score of 0.676 on unseen households, and demonstrate robust generalization without the need for fine-tuning. LLMs also enhance interpretability by providing clear, human-readable explanations for their predictions. Our results show that LLMs can reduce data requirements, improve adaptability, and provide transparent energy disaggregation in NILM applications. 

---
# Enterprise Architecture as a Dynamic Capability for Scalable and Sustainable Generative AI adoption: Bridging Innovation and Governance in Large Organisations 

**Authors**: Alexander Ettinger  

**Link**: [PDF](https://arxiv.org/pdf/2505.06326)  

**Abstract**: Generative Artificial Intelligence is a powerful new technology with the potential to boost innovation and reshape governance in many industries. Nevertheless, organisations face major challenges in scaling GenAI, including technology complexity, governance gaps and resource misalignments. This study explores how Enterprise Architecture Management can meet the complex requirements of GenAI adoption within large enterprises. Based on a systematic literature review and the qualitative analysis of 16 semi-structured interviews with experts, it examines the relationships between EAM, dynamic capabilities and GenAI adoption. The review identified key limitations in existing EA frameworks, particularly their inability to fully address the unique requirements of GenAI. The interviews, analysed using the Gioia methodology, revealed critical enablers and barriers to GenAI adoption across industries. The findings indicate that EAM, when theorised as sensing, seizing and transforming dynamic capabilities, can enhance GenAI adoption by improving strategic alignment, governance frameworks and organisational agility. However, the study also highlights the need to tailor EA frameworks to GenAI-specific challenges, including low data governance maturity and the balance between innovation and compliance. Several conceptual frameworks are proposed to guide EA leaders in aligning GenAI maturity with organisational readiness. The work contributes to academic understanding and industry practice by clarifying the role of EA in bridging innovation and governance in disruptive technology environments. 

---
# Human in the Latent Loop (HILL): Interactively Guiding Model Training Through Human Intuition 

**Authors**: Daniel Geissler, Lars Krupp, Vishal Banwari, David Habusch, Bo Zhou, Paul Lukowicz, Jakob Karolus  

**Link**: [PDF](https://arxiv.org/pdf/2505.06325)  

**Abstract**: Latent space representations are critical for understanding and improving the behavior of machine learning models, yet they often remain obscure and intricate. Understanding and exploring the latent space has the potential to contribute valuable human intuition and expertise about respective domains. In this work, we present HILL, an interactive framework allowing users to incorporate human intuition into the model training by interactively reshaping latent space representations. The modifications are infused into the model training loop via a novel approach inspired by knowledge distillation, treating the user's modifications as a teacher to guide the model in reshaping its intrinsic latent representation. The process allows the model to converge more effectively and overcome inefficiencies, as well as provide beneficial insights to the user. We evaluated HILL in a user study tasking participants to train an optimal model, closely observing the employed strategies. The results demonstrated that human-guided latent space modifications enhance model performance while maintaining generalization, yet also revealing the risks of including user biases. Our work introduces a novel human-AI interaction paradigm that infuses human intuition into model training and critically examines the impact of human intervention on training strategies and potential biases. 

---
# Document Attribution: Examining Citation Relationships using Large Language Models 

**Authors**: Vipula Rawte, Ryan A. Rossi, Franck Dernoncourt, Nedim Lipka  

**Link**: [PDF](https://arxiv.org/pdf/2505.06324)  

**Abstract**: As Large Language Models (LLMs) are increasingly applied to document-based tasks - such as document summarization, question answering, and information extraction - where user requirements focus on retrieving information from provided documents rather than relying on the model's parametric knowledge, ensuring the trustworthiness and interpretability of these systems has become a critical concern. A central approach to addressing this challenge is attribution, which involves tracing the generated outputs back to their source documents. However, since LLMs can produce inaccurate or imprecise responses, it is crucial to assess the reliability of these citations.
To tackle this, our work proposes two techniques. (1) A zero-shot approach that frames attribution as a straightforward textual entailment task. Our method using flan-ul2 demonstrates an improvement of 0.27% and 2.4% over the best baseline of ID and OOD sets of AttributionBench, respectively. (2) We also explore the role of the attention mechanism in enhancing the attribution process. Using a smaller LLM, flan-t5-small, the F1 scores outperform the baseline across almost all layers except layer 4 and layers 8 through 11. 

---
# Learn to Think: Bootstrapping LLM Reasoning Capability Through Graph Learning 

**Authors**: Hang Gao, Chenhao Zhang, Tie Wang, Junsuo Zhao, Fengge Wu, Changwen Zheng, Huaping Liu  

**Link**: [PDF](https://arxiv.org/pdf/2505.06321)  

**Abstract**: Large Language Models (LLMs) have achieved remarkable success across various domains. However, they still face significant challenges, including high computational costs for training and limitations in solving complex reasoning problems. Although existing methods have extended the reasoning capabilities of LLMs through structured paradigms, these approaches often rely on task-specific prompts and predefined reasoning processes, which constrain their flexibility and generalizability. To address these limitations, we propose a novel framework that leverages graph learning to enable more flexible and adaptive reasoning capabilities for LLMs. Specifically, this approach models the reasoning process of a problem as a graph and employs LLM-based graph learning to guide the adaptive generation of each reasoning step. To further enhance the adaptability of the model, we introduce a Graph Neural Network (GNN) module to perform representation learning on the generated reasoning process, enabling real-time adjustments to both the model and the prompt. Experimental results demonstrate that this method significantly improves reasoning performance across multiple tasks without requiring additional training or task-specific prompt design. Code can be found in this https URL. 

---
# Divide (Text) and Conquer (Sentiment): Improved Sentiment Classification by Constituent Conflict Resolution 

**Authors**: Jan Kościałkowski, Paweł Marcinkowski  

**Link**: [PDF](https://arxiv.org/pdf/2505.06320)  

**Abstract**: Sentiment classification, a complex task in natural language processing, becomes even more challenging when analyzing passages with multiple conflicting tones. Typically, longer passages exacerbate this issue, leading to decreased model performance. The aim of this paper is to introduce novel methodologies for isolating conflicting sentiments and aggregating them to effectively predict the overall sentiment of such passages. One of the aggregation strategies involves a Multi-Layer Perceptron (MLP) model which outperforms baseline models across various datasets, including Amazon, Twitter, and SST while costing $\sim$1/100 of what fine-tuning the baseline would take. 

---
# Threat Modeling for AI: The Case for an Asset-Centric Approach 

**Authors**: Jose Sanchez Vicarte, Marcin Spoczynski, Mostafa Elsaid  

**Link**: [PDF](https://arxiv.org/pdf/2505.06315)  

**Abstract**: Recent advances in AI are transforming AI's ubiquitous presence in our world from that of standalone AI-applications into deeply integrated AI-agents. These changes have been driven by agents' increasing capability to autonomously make decisions and initiate actions, using existing applications; whether those applications are AI-based or not. This evolution enables unprecedented levels of AI integration, with agents now able to take actions on behalf of systems and users -- including, in some cases, the powerful ability for the AI to write and execute scripts as it deems necessary. With AI systems now able to autonomously execute code, interact with external systems, and operate without human oversight, traditional security approaches fall short.
This paper introduces an asset-centric methodology for threat modeling AI systems that addresses the unique security challenges posed by integrated AI agents. Unlike existing top-down frameworks that analyze individual attacks within specific product contexts, our bottom-up approach enables defenders to systematically identify how vulnerabilities -- both conventional and AI-specific -- impact critical AI assets across distributed infrastructures used to develop and deploy these agents. This methodology allows security teams to: (1) perform comprehensive analysis that communicates effectively across technical domains, (2) quantify security assumptions about third-party AI components without requiring visibility into their implementation, and (3) holistically identify AI-based vulnerabilities relevant to their specific product context. This approach is particularly relevant for securing agentic systems with complex autonomous capabilities. By focusing on assets rather than attacks, our approach scales with the rapidly evolving threat landscape while accommodating increasingly complex and distributed AI development pipelines. 

---
# A4L: An Architecture for AI-Augmented Learning 

**Authors**: Ashok Goel, Ploy Thajchayapong, Vrinda Nandan, Harshvardhan Sikka, Spencer Rugaber  

**Link**: [PDF](https://arxiv.org/pdf/2505.06314)  

**Abstract**: AI promises personalized learning and scalable education. As AI agents increasingly permeate education in support of teaching and learning, there is a critical and urgent need for data architectures for collecting and analyzing data on learning, and feeding the results back to teachers, learners, and the AI agents for personalization of learning at scale. At the National AI Institute for Adult Learning and Online Education, we are developing an Architecture for AI-Augmented Learning (A4L) for supporting adult learning through online education. We present the motivations, goals, requirements of the A4L architecture. We describe preliminary applications of A4L and discuss how it advances the goals of making learning more personalized and scalable. 

---
# AI Approaches to Qualitative and Quantitative News Analytics on NATO Unity 

**Authors**: Bohdan M. Pavlyshenko  

**Link**: [PDF](https://arxiv.org/pdf/2505.06313)  

**Abstract**: The paper considers the use of GPT models with retrieval-augmented generation (RAG) for qualitative and quantitative analytics on NATO sentiments, NATO unity and NATO Article 5 trust opinion scores in different web sources: news sites found via Google Search API, Youtube videos with comments, and Reddit discussions. A RAG approach using GPT-4.1 model was applied to analyse news where NATO related topics were discussed. Two levels of RAG analytics were used: on the first level, the GPT model generates qualitative news summaries and quantitative opinion scores using zero-shot prompts; on the second level, the GPT model generates the summary of news summaries. Quantitative news opinion scores generated by the GPT model were analysed using Bayesian regression to get trend lines. The distributions found for the regression parameters make it possible to analyse an uncertainty in specified news opinion score trends. Obtained results show a downward trend for analysed scores of opinion related to NATO unity.
This approach does not aim to conduct real political analysis; rather, it consider AI based approaches which can be used for further analytics
as a part of a complex analytical approach. The obtained results demonstrate that the use of GPT models for news analysis can give informative qualitative and quantitative analytics, providing important insights.
The dynamic model based on neural ordinary differential equations was considered for modelling public opinions. This approach makes it possible to analyse different scenarios for evolving public opinions. 

---
# Responsibility Gap in Collective Decision Making 

**Authors**: Pavel Naumov, Jia Tao  

**Link**: [PDF](https://arxiv.org/pdf/2505.06312)  

**Abstract**: The responsibility gap is a set of outcomes of a collective decision-making mechanism in which no single agent is individually responsible. In general, when designing a decision-making process, it is desirable to minimise the gap.
The paper proposes a concept of an elected dictatorship. It shows that, in a perfect information setting, the gap is empty if and only if the mechanism is an elected dictatorship. It also proves that in an imperfect information setting, the class of gap-free mechanisms is positioned strictly between two variations of the class of elected dictatorships. 

---
# Defending against Indirect Prompt Injection by Instruction Detection 

**Authors**: Tongyu Wen, Chenglong Wang, Xiyuan Yang, Haoyu Tang, Yueqi Xie, Lingjuan Lyu, Zhicheng Dou, Fangzhao Wu  

**Link**: [PDF](https://arxiv.org/pdf/2505.06311)  

**Abstract**: The integration of Large Language Models (LLMs) with external sources is becoming increasingly common, with Retrieval-Augmented Generation (RAG) being a prominent example. However, this integration introduces vulnerabilities of Indirect Prompt Injection (IPI) attacks, where hidden instructions embedded in external data can manipulate LLMs into executing unintended or harmful actions. We recognize that the success of IPI attacks fundamentally relies in the presence of instructions embedded within external content, which can alter the behavioral state of LLMs. Can effectively detecting such state changes help us defend against IPI attacks? In this paper, we propose a novel approach that takes external data as input and leverages the behavioral state of LLMs during both forward and backward propagation to detect potential IPI attacks. Specifically, we demonstrate that the hidden states and gradients from intermediate layers provide highly discriminative features for instruction detection. By effectively combining these features, our approach achieves a detection accuracy of 99.60\% in the in-domain setting and 96.90\% in the out-of-domain setting, while reducing the attack success rate to just 0.12\% on the BIPIA benchmark. 

---
# Large Language Model-driven Security Assistant for Internet of Things via Chain-of-Thought 

**Authors**: Mingfei Zeng, Ming Xie, Xixi Zheng, Chunhai Li, Chuan Zhang, Liehuang Zhu  

**Link**: [PDF](https://arxiv.org/pdf/2505.06307)  

**Abstract**: The rapid development of Internet of Things (IoT) technology has transformed people's way of life and has a profound impact on both production and daily activities. However, with the rapid advancement of IoT technology, the security of IoT devices has become an unavoidable issue in both research and applications. Although some efforts have been made to detect or mitigate IoT security vulnerabilities, they often struggle to adapt to the complexity of IoT environments, especially when dealing with dynamic security scenarios. How to automatically, efficiently, and accurately understand these vulnerabilities remains a challenge. To address this, we propose an IoT security assistant driven by Large Language Model (LLM), which enhances the LLM's understanding of IoT security vulnerabilities and related threats. The aim of the ICoT method we propose is to enable the LLM to understand security issues by breaking down the various dimensions of security vulnerabilities and generating responses tailored to the user's specific needs and expertise level. By incorporating ICoT, LLM can gradually analyze and reason through complex security scenarios, resulting in more accurate, in-depth, and personalized security recommendations and solutions. Experimental results show that, compared to methods relying solely on LLM, our proposed LLM-driven IoT security assistant significantly improves the understanding of IoT security issues through the ICoT approach and provides personalized solutions based on the user's identity, demonstrating higher accuracy and reliability. 

---
# User Behavior Analysis in Privacy Protection with Large Language Models: A Study on Privacy Preferences with Limited Data 

**Authors**: Haowei Yang, Qingyi Lu, Yang Wang, Sibei Liu, Jiayun Zheng, Ao Xiang  

**Link**: [PDF](https://arxiv.org/pdf/2505.06305)  

**Abstract**: With the widespread application of large language models (LLMs), user privacy protection has become a significant research topic. Existing privacy preference modeling methods often rely on large-scale user data, making effective privacy preference analysis challenging in data-limited environments. This study explores how LLMs can analyze user behavior related to privacy protection in scenarios with limited data and proposes a method that integrates Few-shot Learning and Privacy Computing to model user privacy preferences. The research utilizes anonymized user privacy settings data, survey responses, and simulated data, comparing the performance of traditional modeling approaches with LLM-based methods. Experimental results demonstrate that, even with limited data, LLMs significantly improve the accuracy of privacy preference modeling. Additionally, incorporating Differential Privacy and Federated Learning further reduces the risk of user data exposure. The findings provide new insights into the application of LLMs in privacy protection and offer theoretical support for advancing privacy computing and user behavior analysis. 

---
# Collaborative Multi-LoRA Experts with Achievement-based Multi-Tasks Loss for Unified Multimodal Information Extraction 

**Authors**: Li Yuan, Yi Cai, Xudong Shen, Qing Li, Qingbao Huang, Zikun Deng, Tao Wang  

**Link**: [PDF](https://arxiv.org/pdf/2505.06303)  

**Abstract**: Multimodal Information Extraction (MIE) has gained attention for extracting structured information from multimedia sources. Traditional methods tackle MIE tasks separately, missing opportunities to share knowledge across tasks. Recent approaches unify these tasks into a generation problem using instruction-based T5 models with visual adaptors, optimized through full-parameter fine-tuning. However, this method is computationally intensive, and multi-task fine-tuning often faces gradient conflicts, limiting performance. To address these challenges, we propose collaborative multi-LoRA experts with achievement-based multi-task loss (C-LoRAE) for MIE tasks. C-LoRAE extends the low-rank adaptation (LoRA) method by incorporating a universal expert to learn shared multimodal knowledge from cross-MIE tasks and task-specific experts to learn specialized instructional task features. This configuration enhances the model's generalization ability across multiple tasks while maintaining the independence of various instruction tasks and mitigating gradient conflicts. Additionally, we propose an achievement-based multi-task loss to balance training progress across tasks, addressing the imbalance caused by varying numbers of training samples in MIE tasks. Experimental results on seven benchmark datasets across three key MIE tasks demonstrate that C-LoRAE achieves superior overall performance compared to traditional fine-tuning methods and LoRA methods while utilizing a comparable number of training parameters to LoRA. 

---
# QiMeng-TensorOp: Automatically Generating High-Performance Tensor Operators with Hardware Primitives 

**Authors**: Xuzhi Zhang, Shaohui Peng, Qirui Zhou, Yuanbo Wen, Qi Guo, Ruizhi Chen, Xinguo Zhu, Weiqiang Xiong, Haixin Chen, Congying Ma, Ke Gao, Chen Zhao, Yanjun Wu, Yunji Chen, Ling Li  

**Link**: [PDF](https://arxiv.org/pdf/2505.06302)  

**Abstract**: Computation-intensive tensor operators constitute over 90\% of the computations in Large Language Models (LLMs) and Deep Neural this http URL and efficiently generating high-performance tensor operators with hardware primitives is crucial for diverse and ever-evolving hardware architectures like RISC-V, ARM, and GPUs, as manually optimized implementation takes at least months and lacks this http URL excel at generating high-level language codes, but they struggle to fully comprehend hardware characteristics and produce high-performance tensor operators. We introduce a tensor-operator auto-generation framework with a one-line user prompt (QiMeng-TensorOp), which enables LLMs to automatically exploit hardware characteristics to generate tensor operators with hardware primitives, and tune parameters for optimal performance across diverse hardware. Experimental results on various hardware platforms, SOTA LLMs, and typical tensor operators demonstrate that QiMeng-TensorOp effectively unleashes the computing capability of various hardware platforms, and automatically generates tensor operators of superior performance. Compared with vanilla LLMs, QiMeng-TensorOp achieves up to $1291 \times$ performance improvement. Even compared with human experts, QiMeng-TensorOp could reach $251 \%$ of OpenBLAS on RISC-V CPUs, and $124 \%$ of cuBLAS on NVIDIA GPUs. Additionally, QiMeng-TensorOp also significantly reduces development costs by $200 \times$ compared with human experts. 

---
# Domain-Adversarial Anatomical Graph Networks for Cross-User Human Activity Recognition 

**Authors**: Xiaozhou Ye, Kevin I-Kai Wang  

**Link**: [PDF](https://arxiv.org/pdf/2505.06301)  

**Abstract**: Cross-user variability in Human Activity Recognition (HAR) remains a critical challenge due to differences in sensor placement, body dynamics, and behavioral patterns. Traditional methods often fail to capture biomechanical invariants that persist across users, limiting their generalization capability. We propose an Edge-Enhanced Graph-Based Adversarial Domain Generalization (EEG-ADG) framework that integrates anatomical correlation knowledge into a unified graph neural network (GNN) architecture. By modeling three biomechanically motivated relationships together-Interconnected Units, Analogous Units, and Lateral Units-our method encodes domain-invariant features while addressing user-specific variability through Variational Edge Feature Extractor. A Gradient Reversal Layer (GRL) enforces adversarial domain generalization, ensuring robustness to unseen users. Extensive experiments on OPPORTUNITY and DSADS datasets demonstrate state-of-the-art performance. Our work bridges biomechanical principles with graph-based adversarial learning by integrating information fusion techniques. This fusion of information underpins our unified and generalized model for cross-user HAR. 

---
# ARDNS-FN-Quantum: A Quantum-Enhanced Reinforcement Learning Framework with Cognitive-Inspired Adaptive Exploration for Dynamic Environments 

**Authors**: Umberto Gonçalves de Sousa  

**Link**: [PDF](https://arxiv.org/pdf/2505.06300)  

**Abstract**: Reinforcement learning (RL) has transformed sequential decision making, yet traditional algorithms like Deep Q-Networks (DQNs) and Proximal Policy Optimization (PPO) often struggle with efficient exploration, stability, and adaptability in dynamic environments. This study presents ARDNS-FN-Quantum (Adaptive Reward-Driven Neural Simulator with Quantum enhancement), a novel framework that integrates a 2-qubit quantum circuit for action selection, a dual-memory system inspired by human cognition, and adaptive exploration strategies modulated by reward variance and curiosity. Evaluated in a 10X10 grid-world over 20,000 episodes, ARDNS-FN-Quantum achieves a 99.5% success rate (versus 81.3% for DQN and 97.0% for PPO), a mean reward of 9.0528 across all episodes (versus 1.2941 for DQN and 7.6196 for PPO), and an average of 46.7 steps to goal (versus 135.9 for DQN and 62.5 for PPO). In the last 100 episodes, it records a mean reward of 9.1652 (versus 7.0916 for DQN and 9.0310 for PPO) and 37.2 steps to goal (versus 52.7 for DQN and 53.4 for PPO). Graphical analyses, including learning curves, steps-to-goal trends, reward variance, and reward distributions, demonstrate ARDNS-FN-Quantum's superior stability (reward variance 5.424 across all episodes versus 252.262 for DQN and 76.583 for PPO) and efficiency. By bridging quantum computing, cognitive science, and RL, ARDNS-FN-Quantum offers a scalable, human-like approach to adaptive learning in uncertain environments, with potential applications in robotics, autonomous systems, and decision-making under uncertainty. 

---
# Input-Specific and Universal Adversarial Attack Generation for Spiking Neural Networks in the Spiking Domain 

**Authors**: Spyridon Raptis, Haralampos-G. Stratigopoulos  

**Link**: [PDF](https://arxiv.org/pdf/2505.06299)  

**Abstract**: As Spiking Neural Networks (SNNs) gain traction across various applications, understanding their security vulnerabilities becomes increasingly important. In this work, we focus on the adversarial attacks, which is perhaps the most concerning threat. An adversarial attack aims at finding a subtle input perturbation to fool the network's decision-making. We propose two novel adversarial attack algorithms for SNNs: an input-specific attack that crafts adversarial samples from specific dataset inputs and a universal attack that generates a reusable patch capable of inducing misclassification across most inputs, thus offering practical feasibility for real-time deployment. The algorithms are gradient-based operating in the spiking domain proving to be effective across different evaluation metrics, such as adversarial accuracy, stealthiness, and generation time. Experimental results on two widely used neuromorphic vision datasets, NMNIST and IBM DVS Gesture, show that our proposed attacks surpass in all metrics all existing state-of-the-art methods. Additionally, we present the first demonstration of adversarial attack generation in the sound domain using the SHD dataset. 

---
# Terahertz Spatial Wireless Channel Modeling with Radio Radiance Field 

**Authors**: John Song, Lihao Zhang, Feng Ye, Haijian Sun  

**Link**: [PDF](https://arxiv.org/pdf/2505.06277)  

**Abstract**: Terahertz (THz) communication is a key enabler for 6G systems, offering ultra-wide bandwidth and unprecedented data rates. However, THz signal propagation differs significantly from lower-frequency bands due to severe free space path loss, minimal diffraction and specular reflection, and prominent scattering, making conventional channel modeling and pilot-based estimation approaches inefficient. In this work, we investigate the feasibility of applying radio radiance field (RRF) framework to the THz band. This method reconstructs a continuous RRF using visual-based geometry and sparse THz RF measurements, enabling efficient spatial channel state information (Spatial-CSI) modeling without dense sampling. We first build a fine simulated THz scenario, then we reconstruct the RRF and evaluate the performance in terms of both reconstruction quality and effectiveness in THz communication, showing that the reconstructed RRF captures key propagation paths with sparse training samples. Our findings demonstrate that RRF modeling remains effective in the THz regime and provides a promising direction for scalable, low-cost spatial channel reconstruction in future 6G networks. 

---
# Attonsecond Streaking Phase Retrieval Via Deep Learning Methods 

**Authors**: Yuzhou Zhu, Zheng Zhang, Ruyi Zhang, Liang Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2505.06275)  

**Abstract**: Attosecond streaking phase retrieval is essential for resolving electron dynamics on sub-femtosecond time scales yet traditional algorithms rely on iterative minimization and central momentum approximations that degrade accuracy for broadband pulses. In this work phase retrieval is reformulated as a supervised computer-vision problem and four neural architectures are systematically compared. A convolutional network demonstrates strong sensitivity to local streak edges but lacks global context; a vision transformer captures long-range delay-energy correlations at the expense of local inductive bias; a hybrid CNN-ViT model unites local feature extraction and full-graph attention; and a capsule network further enforces spatial pose agreement through dynamic routing. A theoretical analysis introduces local, global and positional sensitivity measures and derives surrogate error bounds that predict the strict ordering $CNN<ViT<Hybrid<Capsule$. Controlled experiments on synthetic streaking spectrograms confirm this hierarchy, with the capsule network achieving the highest retrieval fidelity. Looking forward, embedding the strong-field integral into physics-informed neural networks and exploring photonic hardware implementations promise pathways toward real-time attosecond pulse characterization under demanding experimental conditions. 

---
# PARM: Multi-Objective Test-Time Alignment via Preference-Aware Autoregressive Reward Model 

**Authors**: Baijiong Lin, Weisen Jiang, Yuancheng Xu, Hao Chen, Ying-Cong Chen  

**Link**: [PDF](https://arxiv.org/pdf/2505.06274)  

**Abstract**: Multi-objective test-time alignment aims to adapt large language models (LLMs) to diverse multi-dimensional user preferences during inference while keeping LLMs frozen. Recently, GenARM (Xu et al., 2025) first independently trains Autoregressive Reward Models (ARMs) for each preference dimension without awareness of each other, then combines their outputs based on user-specific preference vectors during inference to achieve multi-objective test-time alignment, leading to two key limitations: the need for \textit{multiple} ARMs increases the inference cost, and the separate training of ARMs causes the misalignment between the guided generation and the user preferences. To address these issues, we propose Preference-aware ARM (PARM), a single unified ARM trained across all preference dimensions. PARM uses our proposed Preference-Aware Bilinear Low-Rank Adaptation (PBLoRA), which employs a bilinear form to condition the ARM on preference vectors, enabling it to achieve precise control over preference trade-offs during inference. Experiments demonstrate that PARM reduces inference costs and achieves better alignment with preference vectors compared with existing methods. Additionally, PARM enables weak-to-strong guidance, allowing a smaller PARM to guide a larger frozen LLM without expensive training, making multi-objective alignment accessible with limited computing resources. The code is available at this https URL. 

---
# Policy-labeled Preference Learning: Is Preference Enough for RLHF? 

**Authors**: Taehyun Cho, Seokhun Ju, Seungyub Han, Dohyeong Kim, Kyungjae Lee, Jungwoo Lee  

**Link**: [PDF](https://arxiv.org/pdf/2505.06273)  

**Abstract**: To design rewards that align with human goals, Reinforcement Learning from Human Feedback (RLHF) has emerged as a prominent technique for learning reward functions from human preferences and optimizing policies via reinforcement learning algorithms. However, existing RLHF methods often misinterpret trajectories as being generated by an optimal policy, causing inaccurate likelihood estimation and suboptimal learning. Inspired by Direct Preference Optimization framework which directly learns optimal policy without explicit reward, we propose policy-labeled preference learning (PPL), to resolve likelihood mismatch issues by modeling human preferences with regret, which reflects behavior policy information. We also provide a contrastive KL regularization, derived from regret-based principles, to enhance RLHF in sequential decision making. Experiments in high-dimensional continuous control tasks demonstrate PPL's significant improvements in offline RLHF performance and its effectiveness in online settings. 

---
# A Sensitivity-Driven Expert Allocation Method in LoRA-MoE for Efficient Fine-Tuning 

**Authors**: Junzhou Xu, Boyu Diao  

**Link**: [PDF](https://arxiv.org/pdf/2505.06272)  

**Abstract**: As deep learning models expand, the pre-training-fine-tuning paradigm has become the standard approach for handling various downstream tasks. However, shared parameters can lead to diminished performance when dealing with complex datasets involving multiple tasks. While introducing Mixture-of-Experts (MoE) methods has alleviated this issue to some extent, it also significantly increases the number of parameters required for fine-tuning and training time, introducing greater parameter redundancy. To address these challenges, we propose a method for allocating expert numbers based on parameter sensitivity LoRA-SMoE (A Sensitivity-Driven Expert Allocation Method in LoRA-MoE for Efficient Fine-Tuning). This method rapidly assesses the sensitivity of different tasks to parameters by sampling a small amount of data and using gradient information. It then adaptively allocates expert numbers within a given budget. The process maintains comparable memory consumption to LoRA (Low-Rank Adaptation) while ensuring an efficient and resource-friendly fine-tuning procedure. Experimental results demonstrate that compared to SOTA fine-tuning methods, our LoRA-SMoE approach can enhance model performance while reducing the number of trainable parameters. This significantly improves model performance in resource-constrained environments. Additionally, due to its efficient parameter sensitivity evaluation mechanism, LoRA-SMoE requires minimal computational overhead to optimize expert allocation, making it particularly suitable for scenarios with limited computational resources. All the code in this study will be made publicly available following the acceptance of the paper for publication. Source code is at this https URL 

---
# Tri-MTL: A Triple Multitask Learning Approach for Respiratory Disease Diagnosis 

**Authors**: June-Woo Kim, Sanghoon Lee, Miika Toikkanen, Daehwan Hwang, Kyunghoon Kim  

**Link**: [PDF](https://arxiv.org/pdf/2505.06271)  

**Abstract**: Auscultation remains a cornerstone of clinical practice, essential for both initial evaluation and continuous monitoring. Clinicians listen to the lung sounds and make a diagnosis by combining the patient's medical history and test results. Given this strong association, multitask learning (MTL) can offer a compelling framework to simultaneously model these relationships, integrating respiratory sound patterns with disease manifestations. While MTL has shown considerable promise in medical applications, a significant research gap remains in understanding the complex interplay between respiratory sounds, disease manifestations, and patient metadata attributes. This study investigates how integrating MTL with cutting-edge deep learning architectures can enhance both respiratory sound classification and disease diagnosis. Specifically, we extend recent findings regarding the beneficial impact of metadata on respiratory sound classification by evaluating its effectiveness within an MTL framework. Our comprehensive experiments reveal significant improvements in both lung sound classification and diagnostic performance when the stethoscope information is incorporated into the MTL architecture. 

---
# Importance Analysis for Dynamic Control of Balancing Parameter in a Simple Knowledge Distillation Setting 

**Authors**: Seongmin Kim, Kwanho Kim, Minseung Kim, Kanghyun Jo  

**Link**: [PDF](https://arxiv.org/pdf/2505.06270)  

**Abstract**: Although deep learning models owe their remarkable success to deep and complex architectures, this very complexity typically comes at the expense of real-time performance. To address this issue, a variety of model compression techniques have been proposed, among which knowledge distillation (KD) stands out for its strong empirical performance. The KD contains two concurrent processes: (i) matching the outputs of a large, pre-trained teacher network and a lightweight student network, and (ii) training the student to solve its designated downstream task. The associated loss functions are termed the distillation loss and the downsteam-task loss, respectively. Numerous prior studies report that KD is most effective when the influence of the distillation loss outweighs that of the downstream-task loss. The influence(or importance) is typically regulated by a balancing parameter. This paper provides a mathematical rationale showing that in a simple KD setting when the loss is decreasing, the balancing parameter should be dynamically adjusted 

---
# Cluster-Aware Multi-Round Update for Wireless Federated Learning in Heterogeneous Environments 

**Authors**: Pengcheng Sun, Erwu Liu, Wei Ni, Kanglei Yu, Rui Wang, Abbas Jamalipour  

**Link**: [PDF](https://arxiv.org/pdf/2505.06268)  

**Abstract**: The aggregation efficiency and accuracy of wireless Federated Learning (FL) are significantly affected by resource constraints, especially in heterogeneous environments where devices exhibit distinct data distributions and communication capabilities. This paper proposes a clustering strategy that leverages prior knowledge similarity to group devices with similar data and communication characteristics, mitigating performance degradation from heterogeneity. On this basis, a novel Cluster- Aware Multi-round Update (CAMU) strategy is proposed, which treats clusters as the basic units and adjusts the local update frequency based on the clustered contribution threshold, effectively reducing update bias and enhancing aggregation accuracy. The theoretical convergence of the CAMU strategy is rigorously validated. Meanwhile, based on the convergence upper bound, the local update frequency and transmission power of each cluster are jointly optimized to achieve an optimal balance between computation and communication resources under constrained conditions, significantly improving the convergence efficiency of FL. Experimental results demonstrate that the proposed method effectively improves the model performance of FL in heterogeneous environments and achieves a better balance between communication cost and computational load under limited resources. 

---
# AKD : Adversarial Knowledge Distillation For Large Language Models Alignment on Coding tasks 

**Authors**: Ilyas Oulkadda, Julien Perez  

**Link**: [PDF](https://arxiv.org/pdf/2505.06267)  

**Abstract**: The widespread adoption of Large Language Models (LLMs) for code generation, exemplified by GitHub Copilot\footnote{A coding extension powered by a Code-LLM to assist in code completion tasks} surpassing a million users, highlights the transformative potential of these tools in improving developer productivity. However, this rapid growth also underscores critical concerns regarding the quality, safety, and reliability of the code they generate. As Code-LLMs evolve, they face significant challenges, including the diminishing returns of model scaling and the scarcity of new, high-quality training data. To address these issues, this paper introduces Adversarial Knowledge Distillation (AKD), a novel approach that leverages adversarially generated synthetic datasets to distill the capabilities of larger models into smaller, more efficient ones. By systematically stress-testing and refining the reasoning capabilities of Code-LLMs, AKD provides a framework for enhancing model robustness, reliability, and security while improving their parameter-efficiency. We believe this work represents a critical step toward ensuring dependable automated code generation within the constraints of existing data and the cost-efficiency of model execution. 

---
# Knowledge Guided Encoder-Decoder Framework Integrating Multiple Physical Models for Agricultural Ecosystem Modeling 

**Authors**: Qi Cheng, Licheng Liu, Zhang Yao, Hong Mu, Shiyuan Luo, Zhenong Jin, Yiqun Xie, Xiaowei Jia  

**Link**: [PDF](https://arxiv.org/pdf/2505.06266)  

**Abstract**: Agricultural monitoring is critical for ensuring food security, maintaining sustainable farming practices, informing policies on mitigating food shortage, and managing greenhouse gas emissions. Traditional process-based physical models are often designed and implemented for specific situations, and their parameters could also be highly uncertain. In contrast, data-driven models often use black-box structures and does not explicitly model the inter-dependence between different ecological variables. As a result, they require extensive training data and lack generalizability to different tasks with data distribution shifts and inconsistent observed variables. To address the need for more universal models, we propose a knowledge-guided encoder-decoder model, which can predict key crop variables by leveraging knowledge of underlying processes from multiple physical models. The proposed method also integrates a language model to process complex and inconsistent inputs and also utilizes it to implement a model selection mechanism for selectively combining the knowledge from different physical models. Our evaluations on predicting carbon and nitrogen fluxes for multiple sites demonstrate the effectiveness and robustness of the proposed model under various scenarios. 

---
# Prediction of Delirium Risk in Mild Cognitive Impairment Using Time-Series data, Machine Learning and Comorbidity Patterns -- A Retrospective Study 

**Authors**: Santhakumar Ramamoorthy, Priya Rani, James Mahon, Glenn Mathews, Shaun Cloherty, Mahdi Babaei  

**Link**: [PDF](https://arxiv.org/pdf/2505.06264)  

**Abstract**: Delirium represents a significant clinical concern characterized by high morbidity and mortality rates, particularly in patients with mild cognitive impairment (MCI). This study investigates the associated risk factors for delirium by analyzing the comorbidity patterns relevant to MCI and developing a longitudinal predictive model leveraging machine learning methodologies. A retrospective analysis utilizing the MIMIC-IV v2.2 database was performed to evaluate comorbid conditions, survival probabilities, and predictive modeling outcomes. The examination of comorbidity patterns identified distinct risk profiles for the MCI population. Kaplan-Meier survival analysis demonstrated that individuals with MCI exhibit markedly reduced survival probabilities when developing delirium compared to their non-MCI counterparts, underscoring the heightened vulnerability within this cohort. For predictive modeling, a Long Short-Term Memory (LSTM) ML network was implemented utilizing time-series data, demographic variables, Charlson Comorbidity Index (CCI) scores, and an array of comorbid conditions. The model demonstrated robust predictive capabilities with an AUROC of 0.93 and an AUPRC of 0.92. This study underscores the critical role of comorbidities in evaluating delirium risk and highlights the efficacy of time-series predictive modeling in pinpointing patients at elevated risk for delirium development. 

---
# Dialz: A Python Toolkit for Steering Vectors 

**Authors**: Zara Siddique, Liam D. Turner, Luis Espinosa-Anke  

**Link**: [PDF](https://arxiv.org/pdf/2505.06262)  

**Abstract**: We introduce Dialz, a framework for advancing research on steering vectors for open-source LLMs, implemented in Python. Steering vectors allow users to modify activations at inference time to amplify or weaken a 'concept', e.g. honesty or positivity, providing a more powerful alternative to prompting or fine-tuning. Dialz supports a diverse set of tasks, including creating contrastive pair datasets, computing and applying steering vectors, and visualizations. Unlike existing libraries, Dialz emphasizes modularity and usability, enabling both rapid prototyping and in-depth analysis. We demonstrate how Dialz can be used to reduce harmful outputs such as stereotypes, while also providing insights into model behaviour across different layers. We release Dialz with full documentation, tutorials, and support for popular open-source models to encourage further research in safe and controllable language generation. Dialz enables faster research cycles and facilitates insights into model interpretability, paving the way for safer, more transparent, and more reliable AI systems. 

---
# Modeling supply chain compliance response strategies based on AI synthetic data with structural path regression: A Simulation Study of EU 2027 Mandatory Labor Regulations 

**Authors**: Wei Meng  

**Link**: [PDF](https://arxiv.org/pdf/2505.06261)  

**Abstract**: In the context of the new mandatory labor compliance in the European Union (EU), which will be implemented in 2027, supply chain enterprises face stringent working hour management requirements and compliance risks. In order to scientifically predict the enterprises' coping behaviors and performance outcomes under the policy impact, this paper constructs a methodological framework that integrates the AI synthetic data generation mechanism and structural path regression modeling to simulate the enterprises' strategic transition paths under the new regulations. In terms of research methodology, this paper adopts high-quality simulation data generated based on Monte Carlo mechanism and NIST synthetic data standards to construct a structural path analysis model that includes multiple linear regression, logistic regression, mediation effect and moderating effect. The variable system covers 14 indicators such as enterprise working hours, compliance investment, response speed, automation level, policy dependence, etc. The variable set with explanatory power is screened out through exploratory data analysis (EDA) and VIF multicollinearity elimination. The findings show that compliance investment has a significant positive impact on firm survival and its effect is transmitted through the mediating path of the level of intelligence; meanwhile, firms' dependence on the EU market significantly moderates the strength of this mediating effect. It is concluded that AI synthetic data combined with structural path modeling provides an effective tool for high-intensity regulatory simulation, which can provide a quantitative basis for corporate strategic response, policy design and AI-assisted decision-making in the pre-prediction stage lacking real scenario data. Keywords: AI synthetic data, structural path regression modeling, compliance response strategy, EU 2027 mandatory labor regulation 

---
# Fair Clustering with Clusterlets 

**Authors**: Mattia Setzu, Riccardo Guidotti  

**Link**: [PDF](https://arxiv.org/pdf/2505.06259)  

**Abstract**: Given their widespread usage in the real world, the fairness of clustering methods has become of major interest. Theoretical results on fair clustering show that fairness enjoys transitivity: given a set of small and fair clusters, a trivial centroid-based clustering algorithm yields a fair clustering. Unfortunately, discovering a suitable starting clustering can be computationally expensive, rather complex or arbitrary.
In this paper, we propose a set of simple \emph{clusterlet}-based fuzzy clustering algorithms that match single-class clusters, optimizing fair clustering. Matching leverages clusterlet distance, optimizing for classic clustering objectives, while also regularizing for fairness. Empirical results show that simple matching strategies are able to achieve high fairness, and that appropriate parameter tuning allows to achieve high cohesion and low overlap. 

---
# ABE: A Unified Framework for Robust and Faithful Attribution-Based Explainability 

**Authors**: Zhiyu Zhu, Jiayu Zhang, Zhibo Jin, Fang Chen, Jianlong Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2505.06258)  

**Abstract**: Attribution algorithms are essential for enhancing the interpretability and trustworthiness of deep learning models by identifying key features driving model decisions. Existing frameworks, such as InterpretDL and OmniXAI, integrate multiple attribution methods but suffer from scalability limitations, high coupling, theoretical constraints, and lack of user-friendly implementations, hindering neural network transparency and interoperability. To address these challenges, we propose Attribution-Based Explainability (ABE), a unified framework that formalizes Fundamental Attribution Methods and integrates state-of-the-art attribution algorithms while ensuring compliance with attribution axioms. ABE enables researchers to develop novel attribution techniques and enhances interpretability through four customizable modules: Robustness, Interpretability, Validation, and Data & Model. This framework provides a scalable, extensible foundation for advancing attribution-based explainability and fostering transparent AI systems. Our code is available at: this https URL. 

---
# Beyond Attention: Toward Machines with Intrinsic Higher Mental States 

**Authors**: Ahsan Adeel  

**Link**: [PDF](https://arxiv.org/pdf/2505.06257)  

**Abstract**: Attending to what is relevant is fundamental to both the mammalian brain and modern machine learning models such as Transformers. Yet, determining relevance remains a core challenge, traditionally offloaded to learning algorithms like backpropagation. Inspired by recent cellular neurobiological evidence linking neocortical pyramidal cells to distinct mental states, this work shows how models (e.g., Transformers) can emulate high-level perceptual processing and awake thought (imagination) states to pre-select relevant information before applying attention. Triadic neuronal-level modulation loops among questions ($Q$), clues (keys, $K$), and hypotheses (values, $V$) enable diverse, deep, parallel reasoning chains at the representation level and allow a rapid shift from initial biases to refined understanding. This leads to orders-of-magnitude faster learning with significantly reduced computational demand (e.g., fewer heads, layers, and tokens), at an approximate cost of $\mathcal{O}(N)$, where $N$ is the number of input tokens. Results span reinforcement learning (e.g., CarRacing in a high-dimensional visual setup), computer vision, and natural language question answering. 

---
# SpectrumFM: A Foundation Model for Intelligent Spectrum Management 

**Authors**: Fuhui Zhou, Chunyu Liu, Hao Zhang, Wei Wu, Qihui Wu, Derrick Wing Kwan Ng, Tony Q. S. Quek, Chan-Byoung Chae  

**Link**: [PDF](https://arxiv.org/pdf/2505.06256)  

**Abstract**: Intelligent spectrum management is crucial for improving spectrum efficiency and achieving secure utilization of spectrum resources. However, existing intelligent spectrum management methods, typically based on small-scale models, suffer from notable limitations in recognition accuracy, convergence speed, and generalization, particularly in the complex and dynamic spectrum environments. To address these challenges, this paper proposes a novel spectrum foundation model, termed SpectrumFM, establishing a new paradigm for spectrum management. SpectrumFM features an innovative encoder architecture that synergistically exploits the convolutional neural networks and the multi-head self-attention mechanisms to enhance feature extraction and enable robust representation learning. The model is pre-trained via two novel self-supervised learning tasks, namely masked reconstruction and next-slot signal prediction, which leverage large-scale in-phase and quadrature (IQ) data to achieve comprehensive and transferable spectrum representations. Furthermore, a parameter-efficient fine-tuning strategy is proposed to enable SpectrumFM to adapt to various downstream spectrum management tasks, including automatic modulation classification (AMC), wireless technology classification (WTC), spectrum sensing (SS), and anomaly detection (AD). Extensive experiments demonstrate that SpectrumFM achieves superior performance in terms of accuracy, robustness, adaptability, few-shot learning efficiency, and convergence speed, consistently outperforming conventional methods across multiple benchmarks. Specifically, SpectrumFM improves AMC accuracy by up to 12.1% and WTC accuracy by 9.3%, achieves an area under the curve (AUC) of 0.97 in SS at -4 dB signal-to-noise ratio (SNR), and enhances AD performance by over 10%. 

---
# DeltaDPD: Exploiting Dynamic Temporal Sparsity in Recurrent Neural Networks for Energy-Efficient Wideband Digital Predistortion 

**Authors**: Yizhuo Wu, Yi Zhu, Kun Qian, Qinyu Chen, Anding Zhu, John Gajadharsing, Leo C. N. de Vreede, Chang Gao  

**Link**: [PDF](https://arxiv.org/pdf/2505.06250)  

**Abstract**: Digital Predistortion (DPD) is a popular technique to enhance signal quality in wideband RF power amplifiers (PAs). With increasing bandwidth and data rates, DPD faces significant energy consumption challenges during deployment, contrasting with its efficiency goals. State-of-the-art DPD models rely on recurrent neural networks (RNN), whose computational complexity hinders system efficiency. This paper introduces DeltaDPD, exploring the dynamic temporal sparsity of input signals and neuronal hidden states in RNNs for energy-efficient DPD, reducing arithmetic operations and memory accesses while preserving satisfactory linearization performance. Applying a TM3.1a 200MHz-BW 256-QAM OFDM signal to a 3.5 GHz GaN Doherty RF PA, DeltaDPD achieves -50.03 dBc in Adjacent Channel Power Ratio (ACPR), -37.22 dB in Normalized Mean Square Error (NMSE) and -38.52 dBc in Error Vector Magnitude (EVM) with 52% temporal sparsity, leading to a 1.8X reduction in estimated inference power. The DeltaDPD code will be released after formal publication at this https URL. 

---
# United States Road Accident Prediction using Random Forest Predictor 

**Authors**: Dominic Parosh Yamarthi, Haripriya Raman, Shamsad Parvin  

**Link**: [PDF](https://arxiv.org/pdf/2505.06246)  

**Abstract**: Road accidents significantly threaten public safety and require in-depth analysis for effective prevention and mitigation strategies. This paper focuses on predicting accidents through the examination of a comprehensive traffic dataset covering 49 states in the United States. The dataset integrates information from diverse sources, including transportation departments, law enforcement, and traffic sensors. This paper specifically emphasizes predicting the number of accidents, utilizing advanced machine learning models such as regression analysis and time series analysis. The inclusion of various factors, ranging from environmental conditions to human behavior and infrastructure, ensures a holistic understanding of the dynamics influencing road safety. Temporal and spatial analysis further allows for the identification of trends, seasonal variations, and high-risk areas. The implications of this research extend to proactive decision-making for policymakers and transportation authorities. By providing accurate predictions and quantifiable insights into expected accident rates under different conditions, the paper aims to empower authorities to allocate resources efficiently and implement targeted interventions. The goal is to contribute to the development of informed policies and interventions that enhance road safety, creating a safer environment for all road users. Keywords: Machine Learning, Random Forest, Accident Prediction, AutoML, LSTM. 

---
# Low-Complexity CNN-Based Classification of Electroneurographic Signals 

**Authors**: Arek Berc Gokdag, Silvia Mura, Antonio Coviello, Michele Zhu, Maurizio Magarini, Umberto Spagnolini  

**Link**: [PDF](https://arxiv.org/pdf/2505.06241)  

**Abstract**: Peripheral nerve interfaces (PNIs) facilitate neural recording and stimulation for treating nerve injuries, but real-time classification of electroneurographic (ENG) signals remains challenging due to constraints on complexity and latency, particularly in implantable devices. This study introduces MobilESCAPE-Net, a lightweight architecture that reduces computational cost while maintaining and slightly improving classification performance. Compared to the state-of-the-art ESCAPE-Net, MobilESCAPE-Net achieves comparable accuracy and F1-score with significantly lower complexity, reducing trainable parameters by 99.9\% and floating point operations per second by 92.47\%, enabling faster inference and real-time processing. Its efficiency makes it well-suited for low-complexity ENG signal classification in resource-constrained environments such as implantable devices. 

---
