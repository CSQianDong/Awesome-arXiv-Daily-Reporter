# Aligned Query Expansion: Efficient Query Expansion for Information Retrieval through LLM Alignment 

**Authors**: Adam Yang, Gustavo Penha, Enrico Palumbo, Hugues Bouchard  

**Link**: [PDF](https://arxiv.org/pdf/2507.11042)  

**Abstract**: With the breakthroughs in large language models (LLMs), query generation techniques that expand documents and queries with related terms are becoming increasingly popular in the information retrieval field. Such techniques have been shown to improve the effectiveness of traditional lexical retrieval methods by dealing with the vocabulary mismatch problem. Recent work has found that generating queries with a greedy decoding strategy can produce sub-optimal queries, including hallucinations, and proposed to filter out queries before expansion. This `generate-then-filter' approach is costly, as it requires generating multiple queries and applying a relevance model to all of them and does not teach the LLM which of the generated queries is more effective for expansion. To overcome such limitations, we propose Aligned Query Expansion (AQE), a novel approach to enhance query expansion for passage retrieval in open-domain question answering. AQE leverages recent techniques in LLM alignment to fine-tune models for generating query expansions that directly optimize the effectiveness of the retrieval task, eliminating the need for additional filtering steps. This alignment ensures that queries are more relevant, reducing computational costs while improving retrieval effectiveness. Empirical evaluations show that AQE outperforms baseline models for query expansion in both in-domain and out-of-domain settings, demonstrating significant improvements in retrieval effectiveness. 

---
# Seq vs Seq: An Open Suite of Paired Encoders and Decoders 

**Authors**: Orion Weller, Kathryn Ricci, Marc Marone, Antoine Chaffin, Dawn Lawrie, Benjamin Van Durme  

**Link**: [PDF](https://arxiv.org/pdf/2507.11412)  

**Abstract**: The large language model (LLM) community focuses almost exclusively on decoder-only language models, since they are easier to use for text generation. However, a large subset of the community still uses encoder-only models for tasks such as classification or retrieval. Previous work has attempted to compare these architectures, but is forced to make comparisons with models that have different numbers of parameters, training techniques, and datasets. We introduce the SOTA open-data Ettin suite of models: paired encoder-only and decoder-only models ranging from 17 million parameters to 1 billion, trained on up to 2 trillion tokens. Using the same recipe for both encoder-only and decoder-only models produces SOTA recipes in both categories for their respective sizes, beating ModernBERT as an encoder and Llama 3.2 and SmolLM2 as decoders. Like previous work, we find that encoder-only models excel at classification and retrieval tasks while decoders excel at generative tasks. However, we show that adapting a decoder model to encoder tasks (and vice versa) through continued training is subpar compared to using only the reverse objective (i.e. a 400M encoder outperforms a 1B decoder on MNLI, and vice versa for generative tasks). We open-source all artifacts of this study including training data, training order segmented by checkpoint, and 200+ checkpoints to allow future work to analyze or extend all aspects of training. 

---
# Repairing Language Model Pipelines by Meta Self-Refining Competing Constraints at Runtime 

**Authors**: Mojtaba Eshghie  

**Link**: [PDF](https://arxiv.org/pdf/2507.10590)  

**Abstract**: Language Model (LM) pipelines can dynamically refine their outputs against programmatic constraints. However, their effectiveness collapses when faced with competing soft constraints, leading to inefficient backtracking loops where satisfying one constraint violates another. We introduce Meta Self-Refining, a framework that equips LM pipelines with a meta-corrective layer to repair these competitions at runtime/inference-time. Our approach monitors the pipeline's execution history to detect oscillatory failures. Upon detection, it invokes a meta-repairer LM that analyzes the holistic state of the backtracking attempts and synthesizes a strategic instruction to balance the competing requirements. This self-repair instruction guides the original LM out of a failing refining loop towards a successful output. Our results show Meta Self-Refining can successfully repair these loops, leading to more efficient LM programs. 

---
# LLM-Driven Dual-Level Multi-Interest Modeling for Recommendation 

**Authors**: Ziyan Wang, Yingpeng Du, Zhu Sun, Jieyi Bi, Haoyan Chua, Tianjun Wei, Jie Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2507.10917)  

**Abstract**: Recently, much effort has been devoted to modeling users' multi-interests based on their behaviors or auxiliary signals. However, existing methods often rely on heuristic assumptions, e.g., co-occurring items indicate the same interest of users, failing to capture user multi-interests aligning with real-world scenarios. While large language models (LLMs) show significant potential for multi-interest analysis due to their extensive knowledge and powerful reasoning capabilities, two key challenges remain. First, the granularity of LLM-driven multi-interests is agnostic, possibly leading to overly fine or coarse interest grouping. Second, individual user analysis provides limited insights due to the data sparsity issue. In this paper, we propose an LLM-driven dual-level multi-interest modeling framework for more effective recommendation. At the user-individual level, we exploit LLMs to flexibly allocate items engaged by users into different semantic clusters, indicating their diverse and distinct interests. To alleviate the agnostic generation of LLMs, we adaptively assign these semantic clusters to users' collaborative multi-interests learned from global user-item interactions, allowing the granularity to be automatically adjusted according to the user's behaviors using an alignment module. To alleviate the limited insights derived from individual users' behaviors, at the user-crowd level, we propose aggregating user cliques into synthesized users with rich behaviors for more comprehensive LLM-driven multi-interest analysis. We formulate a max covering problem to ensure the compactness and representativeness of synthesized users' behaviors, and then conduct contrastive learning based on their LLM-driven multi-interests to disentangle item representations among different interests. Experiments on real-world datasets show the superiority of our approach against state-of-the-art methods. 

---
# From Chaos to Automation: Enabling the Use of Unstructured Data for Robotic Process Automation 

**Authors**: Kelly Kurowski, Xixi Lu, Hajo A. Reijers  

**Link**: [PDF](https://arxiv.org/pdf/2507.11364)  

**Abstract**: The growing volume of unstructured data within organizations poses significant challenges for data analysis and process automation. Unstructured data, which lacks a predefined format, encompasses various forms such as emails, reports, and scans. It is estimated to constitute approximately 80% of enterprise data. Despite the valuable insights it can offer, extracting meaningful information from unstructured data is more complex compared to structured data. Robotic Process Automation (RPA) has gained popularity for automating repetitive tasks, improving efficiency, and reducing errors. However, RPA is traditionally reliant on structured data, limiting its application to processes involving unstructured documents. This study addresses this limitation by developing the UNstructured Document REtrieval SyStem (UNDRESS), a system that uses fuzzy regular expressions, techniques for natural language processing, and large language models to enable RPA platforms to effectively retrieve information from unstructured documents. The research involved the design and development of a prototype system, and its subsequent evaluation based on text extraction and information retrieval performance. The results demonstrate the effectiveness of UNDRESS in enhancing RPA capabilities for unstructured data, providing a significant advancement in the field. The findings suggest that this system could facilitate broader RPA adoption across processes traditionally hindered by unstructured data, thereby improving overall business process efficiency. 

---
# Automated Thematic Analyses Using LLMs: Xylazine Wound Management Social Media Chatter Use Case 

**Authors**: JaMor Hairston, Ritvik Ranjan, Sahithi Lakamana, Anthony Spadaro, Selen Bozkurt, Jeanmarie Perrone, Abeed Sarker  

**Link**: [PDF](https://arxiv.org/pdf/2507.10803)  

**Abstract**: Background Large language models (LLMs) face challenges in inductive thematic analysis, a task requiring deep interpretive and domain-specific expertise. We evaluated the feasibility of using LLMs to replicate expert-driven thematic analysis of social media data. Methods Using two temporally non-intersecting Reddit datasets on xylazine (n=286 and n=686, for model optimization and validation, respectively) with twelve expert-derived themes, we evaluated five LLMs against expert coding. We modeled the task as a series of binary classifications, rather than a single, multi-label classification, employing zero-, single-, and few-shot prompting strategies and measuring performance via accuracy, precision, recall, and F1-score. Results On the validation set, GPT-4o with two-shot prompting performed best (accuracy: 90.9%; F1-score: 0.71). For high-prevalence themes, model-derived thematic distributions closely mirrored expert classifications (e.g., xylazine use: 13.6% vs. 17.8%; MOUD use: 16.5% vs. 17.8%). Conclusions Our findings suggest that few-shot LLM-based approaches can automate thematic analyses, offering a scalable supplement for qualitative research. Keywords: thematic analysis, large language models, natural language processing, qualitative analysis, social media, prompt engineering, public health 

---
# How Many Instructions Can LLMs Follow at Once? 

**Authors**: Daniel Jaroslawicz, Brendan Whiting, Parth Shah, Karime Maamari  

**Link**: [PDF](https://arxiv.org/pdf/2507.11538)  

**Abstract**: Production-grade LLM systems require robust adherence to dozens or even hundreds of instructions simultaneously. However, the instruction-following capabilities of LLMs at high instruction densities have not yet been characterized, as existing benchmarks only evaluate models on tasks with a single or few instructions. We introduce IFScale, a simple benchmark of 500 keyword-inclusion instructions for a business report writing task to measure how instruction-following performance degrades as instruction density increases. We evaluate 20 state-of-the-art models across seven major providers and find that even the best frontier models only achieve 68% accuracy at the max density of 500 instructions. Our analysis reveals model size and reasoning capability to correlate with 3 distinct performance degradation patterns, bias towards earlier instructions, and distinct categories of instruction-following errors. Our insights can help inform design of instruction-dense prompts in real-world applications and highlight important performance-latency tradeoffs. We open-source the benchmark and all results for further analysis at this https URL. 

---
# DrafterBench: Benchmarking Large Language Models for Tasks Automation in Civil Engineering 

**Authors**: Yinsheng Li, Zhen Dong, Yi Shao  

**Link**: [PDF](https://arxiv.org/pdf/2507.11527)  

**Abstract**: Large Language Model (LLM) agents have shown great potential for solving real-world problems and promise to be a solution for tasks automation in industry. However, more benchmarks are needed to systematically evaluate automation agents from an industrial perspective, for example, in Civil Engineering. Therefore, we propose DrafterBench for the comprehensive evaluation of LLM agents in the context of technical drawing revision, a representation task in civil engineering. DrafterBench contains twelve types of tasks summarized from real-world drawing files, with 46 customized functions/tools and 1920 tasks in total. DrafterBench is an open-source benchmark to rigorously test AI agents' proficiency in interpreting intricate and long-context instructions, leveraging prior knowledge, and adapting to dynamic instruction quality via implicit policy awareness. The toolkit comprehensively assesses distinct capabilities in structured data comprehension, function execution, instruction following, and critical reasoning. DrafterBench offers detailed analysis of task accuracy and error statistics, aiming to provide deeper insight into agent capabilities and identify improvement targets for integrating LLMs in engineering applications. Our benchmark is available at this https URL, with the test set hosted at this https URL. 

---
# Function-to-Style Guidance of LLMs for Code Translation 

**Authors**: Longhui Zhang, Bin Wang, Jiahao Wang, Xiaofeng Zhao, Min Zhang, Hao Yang, Meishan Zhang, Yu Li, Jing Li, Jun Yu, Min Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2507.11083)  

**Abstract**: Large language models (LLMs) have made significant strides in code translation tasks. However, ensuring both the correctness and readability of translated code remains a challenge, limiting their effective adoption in real-world software development. In this work, we propose F2STrans, a function-to-style guiding paradigm designed to progressively improve the performance of LLMs in code translation. Our approach comprises two key stages: (1) Functional learning, which optimizes translation correctness using high-quality source-target code pairs mined from online programming platforms, and (2) Style learning, which improves translation readability by incorporating both positive and negative style examples. Additionally, we introduce a novel code translation benchmark that includes up-to-date source code, extensive test cases, and manually annotated ground-truth translations, enabling comprehensive functional and stylistic evaluations. Experiments on both our new benchmark and existing datasets demonstrate that our approach significantly improves code translation performance. Notably, our approach enables Qwen-1.5B to outperform prompt-enhanced Qwen-32B and GPT-4 on average across 20 diverse code translation scenarios. 

---
# Foundation Models for Logistics: Toward Certifiable, Conversational Planning Interfaces 

**Authors**: Yunhao Yang, Neel P. Bhatt, Christian Ellis, Alvaro Velasquez, Zhangyang Wang, Ufuk Topcu  

**Link**: [PDF](https://arxiv.org/pdf/2507.11352)  

**Abstract**: Logistics operators, from battlefield coordinators rerouting airlifts ahead of a storm to warehouse managers juggling late trucks, often face life-critical decisions that demand both domain expertise and rapid and continuous replanning. While popular methods like integer programming yield logistics plans that satisfy user-defined logical constraints, they are slow and assume an idealized mathematical model of the environment that does not account for uncertainty. On the other hand, large language models (LLMs) can handle uncertainty and promise to accelerate replanning while lowering the barrier to entry by translating free-form utterances into executable plans, yet they remain prone to misinterpretations and hallucinations that jeopardize safety and cost. We introduce a neurosymbolic framework that pairs the accessibility of natural-language dialogue with verifiable guarantees on goal interpretation. It converts user requests into structured planning specifications, quantifies its own uncertainty at the field and token level, and invokes an interactive clarification loop whenever confidence falls below an adaptive threshold. A lightweight model, fine-tuned on just 100 uncertainty-filtered examples, surpasses the zero-shot performance of GPT-4.1 while cutting inference latency by nearly 50%. These preliminary results highlight a practical path toward certifiable, real-time, and user-aligned decision-making for complex logistics. 

---
# Lessons Learned from Evaluation of LLM based Multi-agents in Safer Therapy Recommendation 

**Authors**: Yicong Wu, Ting Chen, Irit Hochberg, Zhoujian Sun, Ruth Edry, Zhengxing Huang, Mor Peleg  

**Link**: [PDF](https://arxiv.org/pdf/2507.10911)  

**Abstract**: Therapy recommendation for chronic patients with multimorbidity is challenging due to risks of treatment conflicts. Existing decision support systems face scalability limitations. Inspired by the way in which general practitioners (GP) manage multimorbidity patients, occasionally convening multidisciplinary team (MDT) collaboration, this study investigated the feasibility and value of using a Large Language Model (LLM)-based multi-agent system (MAS) for safer therapy recommendations. We designed a single agent and a MAS framework simulating MDT decision-making by enabling discussion among LLM agents to resolve medical conflicts. The systems were evaluated on therapy planning tasks for multimorbidity patients using benchmark cases. We compared MAS performance with single-agent approaches and real-world benchmarks. An important contribution of our study is the definition of evaluation metrics that go beyond the technical precision and recall and allow the inspection of clinical goals met and medication burden of the proposed advices to a gold standard benchmark. Our results show that with current LLMs, a single agent GP performs as well as MDTs. The best-scoring models provide correct recommendations that address all clinical goals, yet the advices are incomplete. Some models also present unnecessary medications, resulting in unnecessary conflicts between medication and conditions or drug-drug interactions. 

---
# From Semantic Web and MAS to Agentic AI: A Unified Narrative of the Web of Agents 

**Authors**: Tatiana Petrova, Aleksandr Puzikov, Boris Bliznukov, Radu State  

**Link**: [PDF](https://arxiv.org/pdf/2507.10644)  

**Abstract**: The concept of the Web of Agents (WoA), which transforms the static, document-centric Web into an environment of autonomous agents acting on users' behalf, has attracted growing interest as large language models (LLMs) become more capable. However, research in this area is still fragmented across different communities. Contemporary surveys catalog the latest LLM-powered frameworks, while the rich histories of Multi-Agent Systems (MAS) and the Semantic Web are often treated as separate, legacy domains. This fragmentation obscures the intellectual lineage of modern systems and hinders a holistic understanding of the field's trajectory. We present the first comprehensive evolutionary overview of the WoA. We show that modern protocols like A2A and the MCP, are direct evolutionary responses to the well-documented limitations of earlier standards like FIPA standards and OWL-based semantic agents. To systematize this analysis, we introduce a four-axis taxonomy (semantic foundation, communication paradigm, locus of intelligence, discovery mechanism). This framework provides a unified analytical lens for comparing agent architectures across all generations, revealing a clear line of descent where others have seen a disconnect. Our analysis identifies a paradigm shift in the 'locus of intelligence': from being encoded in external data (Semantic Web) or the platform (MAS) to being embedded within the agent's core model (LLM). This shift is foundational to modern Agentic AI, enabling the scalable and adaptive systems the WoA has long envisioned. We conclude that while new protocols are essential, they are insufficient for building a robust, open, trustworthy ecosystem. Finally, we argue that the next research frontier lies in solving persistent socio-technical challenges, and we map out a new agenda focused on decentralized identity, economic models, security, and governance for the emerging WoA. 

---
# AI Mother Tongue: Self-Emergent Communication in MARL via Endogenous Symbol Systems 

**Authors**: Hung Ming Liu  

**Link**: [PDF](https://arxiv.org/pdf/2507.10566)  

**Abstract**: In Decentralized Multi-Agent Reinforcement Learning (MARL), the development of Emergent Communication has long been constrained by the ``Joint Exploration Dilemma'', leading agents to fall into a ``Communication Vacuum Equilibrium'' . Traditional methods address this by introducing inductive biases to facilitate communication emergence . This study fundamentally questions whether such artificial inductive biases are, in fact, over-engineering. Through experiments with the ``AI Mother Tongue'' (AIM) framework, based on a Vector Quantized Variational Autoencoder (VQ-VAE), we demonstrate that when agents possess an endogenous symbol system, their neural representations naturally exhibit spontaneous semantic compression and Nash equilibrium-driven semantic convergence, achieving effective symbolic communication without external inductive biases. This aligns with recent neuroscience findings suggesting that the human brain does not directly use human language for internal thought , and resonates with research on ``soft thinking'' capabilities in Large Language Models (LLMs) . Compared to traditional explicit communication methods, AIM demonstrates stronger generality and efficiency. The interpretable analysis toolkit developed in this study confirms that symbol usage exhibits a significant power-law distribution, leading to three major theoretical insights: the ``Neural Communication Hypothesis'', the ``Tool-First Principle'', and the ``Semantic Interpretability Paradigm''. Future research will explore the integration of Hierarchical Quantized Variational Autoencoders (HQ-VAE) to enhance AIM's complex expressive capabilities and investigate the potential for ``Reinforcement Learning (RL) Low-Level Pre-training''. This discovery offers new avenues for bridging symbolism and connectionism. 

---
# AirLLM: Diffusion Policy-based Adaptive LoRA for Remote Fine-Tuning of LLM over the Air 

**Authors**: Shiyi Yang, Xiaoxue Yu, Rongpeng Li, Jianhang Zhu, Zhifeng Zhao, Honggang Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2507.11515)  

**Abstract**: Operating Large Language Models (LLMs) on edge devices is increasingly challenged by limited communication bandwidth and strained computational and memory costs. Thus, cloud-assisted remote fine-tuning becomes indispensable. Nevertheless, existing Low-Rank Adaptation (LoRA) approaches typically employ fixed or heuristic rank configurations, and the subsequent over-the-air transmission of all LoRA parameters could be rather inefficient. To address this limitation, we develop AirLLM, a hierarchical diffusion policy framework for communication-aware LoRA adaptation. Specifically, AirLLM models the rank configuration as a structured action vector that spans all LoRA-inserted projections. To solve the underlying high-dimensional sequential decision-making problem, a Proximal Policy Optimization (PPO) agent generates coarse-grained decisions by jointly observing wireless states and linguistic complexity, which are then refined via Denoising Diffusion Implicit Models (DDIM) to produce high-resolution, task- and channel-adaptive rank vectors. The two modules are optimized alternatively, with the DDIM trained under the Classifier-Free Guidance (CFG) paradigm to maintain alignment with PPO rewards. Experiments under varying signal-to-noise ratios demonstrate that AirLLM consistently enhances fine-tuning performance while significantly reducing transmission costs, highlighting the effectiveness of reinforcement-driven, diffusion-refined rank adaptation for scalable and efficient remote fine-tuning over the air. 

---
# Opus: A Prompt Intention Framework for Complex Workflow Generation 

**Authors**: Théo Fagnoni, Mahsun Altin, Chia En Chung, Phillip Kingston, Alan Tuning, Dana O. Mohamed, Inès Adnani  

**Link**: [PDF](https://arxiv.org/pdf/2507.11288)  

**Abstract**: This paper introduces the Opus Prompt Intention Framework, designed to improve complex Workflow Generation with instruction-tuned Large Language Models (LLMs). We propose an intermediate Intention Capture layer between user queries and Workflow Generation, implementing the Opus Workflow Intention Framework, which consists of extracting Workflow Signals from user queries, interpreting them into structured Workflow Intention objects, and generating Workflows based on these Intentions. Our results show that this layer enables LLMs to produce logical and meaningful outputs that scale reliably as query complexity increases. On a synthetic benchmark of 1,000 multi-intent query-Workflow(s) pairs, applying the Opus Prompt Intention Framework to Workflow Generation yields consistent improvements in semantic Workflow similarity metrics. In this paper, we introduce the Opus Prompt Intention Framework by applying the concepts of Workflow Signal and Workflow Intention to LLM-driven Workflow Generation. We present a reproducible, customizable LLM-based Intention Capture system to extract Workflow Signals and Workflow Intentions from user queries. Finally, we provide empirical evidence that the proposed system significantly improves Workflow Generation quality compared to direct generation from user queries, particularly in cases of Mixed Intention Elicitation. 

---
# Automated Novelty Evaluation of Academic Paper: A Collaborative Approach Integrating Human and Large Language Model Knowledge 

**Authors**: Wenqing Wu, Chengzhi Zhang, Yi Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2507.11330)  

**Abstract**: Novelty is a crucial criterion in the peer review process for evaluating academic papers. Traditionally, it's judged by experts or measure by unique reference combinations. Both methods have limitations: experts have limited knowledge, and the effectiveness of the combination method is uncertain. Moreover, it's unclear if unique citations truly measure novelty. The large language model (LLM) possesses a wealth of knowledge, while human experts possess judgment abilities that the LLM does not possess. Therefore, our research integrates the knowledge and abilities of LLM and human experts to address the limitations of novelty assessment. The most common novelty in academic papers is the introduction of new methods. In this paper, we propose leveraging human knowledge and LLM to assist pretrained language models (PLMs, e.g. BERT etc.) in predicting the method novelty of papers. Specifically, we extract sentences related to the novelty of the academic paper from peer review reports and use LLM to summarize the methodology section of the academic paper, which are then used to fine-tune PLMs. In addition, we have designed a text-guided fusion module with novel Sparse-Attention to better integrate human and LLM knowledge. We compared the method we proposed with a large number of baselines. Extensive experiments demonstrate that our method achieves superior performance. 

---
# Role-Playing LLM-Based Multi-Agent Support Framework for Detecting and Addressing Family Communication Bias 

**Authors**: Rushia Harada, Yuken Kimura, Keito Inoshita  

**Link**: [PDF](https://arxiv.org/pdf/2507.11210)  

**Abstract**: Well-being in family settings involves subtle psychological dynamics that conventional metrics often overlook. In particular, unconscious parental expectations, termed ideal parent bias, can suppress children's emotional expression and autonomy. This suppression, referred to as suppressed emotion, often stems from well-meaning but value-driven communication, which is difficult to detect or address from outside the family. Focusing on these latent dynamics, this study explores Large Language Model (LLM)-based support for psychologically safe family communication. We constructed a Japanese parent-child dialogue corpus of 30 scenarios, each annotated with metadata on ideal parent bias and suppressed emotion. Based on this corpus, we developed a Role-Playing LLM-based multi-agent dialogue support framework that analyzes dialogue and generates feedback. Specialized agents detect suppressed emotion, describe implicit ideal parent bias in parental speech, and infer contextual attributes such as the child's age and background. A meta-agent compiles these outputs into a structured report, which is then passed to five selected expert agents. These agents collaboratively generate empathetic and actionable feedback through a structured four-step discussion process. Experiments show that the system can detect categories of suppressed emotion with moderate accuracy and produce feedback rated highly in empathy and practicality. Moreover, simulated follow-up dialogues incorporating this feedback exhibited signs of improved emotional expression and mutual understanding, suggesting the framework's potential in supporting positive transformation in family interactions. 

---
# An Agentic Flow for Finite State Machine Extraction using Prompt Chaining 

**Authors**: Fares Wael, Youssef Maklad, Ali Hamdi, Wael Elsersy  

**Link**: [PDF](https://arxiv.org/pdf/2507.11222)  

**Abstract**: Finite-State Machines (FSMs) are critical for modeling the operational logic of network protocols, enabling verification, analysis, and vulnerability discovery. However, existing FSM extraction techniques face limitations such as scalability, incomplete coverage, and ambiguity in natural language specifications. In this paper, we propose FlowFSM, a novel agentic framework that leverages Large Language Models (LLMs) combined with prompt chaining and chain-of-thought reasoning to extract accurate FSMs from raw RFC documents. FlowFSM systematically processes protocol specifications, identifies state transitions, and constructs structured rule-books by chaining agent outputs. Experimental evaluation across FTP and RTSP protocols demonstrates that FlowFSM achieves high extraction precision while minimizing hallucinated transitions, showing promising results. Our findings highlight the potential of agent-based LLM systems in the advancement of protocol analysis and FSM inference for cybersecurity and reverse engineering applications. 

---
# Internal Value Alignment in Large Language Models through Controlled Value Vector Activation 

**Authors**: Haoran Jin, Meng Li, Xiting Wang, Zhihao Xu, Minlie Huang, Yantao Jia, Defu Lian  

**Link**: [PDF](https://arxiv.org/pdf/2507.11316)  

**Abstract**: Aligning Large Language Models (LLMs) with human values has attracted increasing attention since it provides clarity, transparency, and the ability to adapt to evolving scenarios. In this paper, we introduce a Controlled Value Vector Activation (ConVA) method that directly aligns the internal values of LLMs by interpreting how a value is encoded in their latent representations and modifies relevant activations to ensure consistent values in LLMs. To ensure an accurate and unbiased interpretation, we propose a context-controlled value vector identification method. To consistently control values without sacrificing model performance, we introduce a gated value vector activation method for effective and minimum degree of value control. Experiments show that our method achieves the highest control success rate across 10 basic values without hurting LLM performance and fluency, and ensures target values even with opposite and potentially malicious input prompts. Source code and data are available at~ this https URL. 

---
# Comprehension Without Competence: Architectural Limits of LLMs in Symbolic Computation and Reasoning 

**Authors**: Zheng Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2507.10624)  

**Abstract**: Large Language Models (LLMs) display striking surface fluency yet systematically fail at tasks requiring symbolic reasoning, arithmetic accuracy, and logical consistency. This paper offers a structural diagnosis of such failures, revealing a persistent gap between \textit{comprehension} and \textit{competence}. Through controlled experiments and architectural analysis, we demonstrate that LLMs often articulate correct principles without reliably applying them--a failure rooted not in knowledge access, but in computational execution. We term this phenomenon the computational \textit{split-brain syndrome}, where instruction and action pathways are geometrically and functionally dissociated. This core limitation recurs across domains, from mathematical operations to relational inferences, and explains why model behavior remains brittle even under idealized prompting. We argue that LLMs function as powerful pattern completion engines, but lack the architectural scaffolding for principled, compositional reasoning. Our findings delineate the boundary of current LLM capabilities and motivate future models with metacognitive control, principle lifting, and structurally grounded execution. This diagnosis also clarifies why mechanistic interpretability findings may reflect training-specific pattern coordination rather than universal computational principles, and why the geometric separation between instruction and execution pathways suggests limitations in neural introspection and mechanistic analysis. 

---
# LogTinyLLM: Tiny Large Language Models Based Contextual Log Anomaly Detection 

**Authors**: Isaiah Thompson Ocansey, Ritwik Bhattacharya, Tanmay Sen  

**Link**: [PDF](https://arxiv.org/pdf/2507.11071)  

**Abstract**: Log anomaly detection using traditional rule based or deep learning based methods is often challenging due to the large volume and highly complex nature of log sequence. So effective way of detection of anomalous sequence of logs is crucial for system maintenance and development. This paper proposes parameter efficient finetuning specifically low rank adaptation (LoRA) and adapter based approaches for finding contextual anomalies in sequence of logs in large log data set. It compares different tiny large language models (LLMs) on the Thunderbird dataset. The results show that LoRA based finetuning provides substantial performance improvements of 18 to 19 percentage over LogBert based full finetuning approach, achieving accuracy scores between 97.76% and 98.83% compared to 79.37%. 

---
# LLM-Augmented Symptom Analysis for Cardiovascular Disease Risk Prediction: A Clinical NLP 

**Authors**: Haowei Yang, Ziyu Shen, Junli Shao, Luyao Men, Xinyue Han, Jing Dong  

**Link**: [PDF](https://arxiv.org/pdf/2507.11052)  

**Abstract**: Timely identification and accurate risk stratification of cardiovascular disease (CVD) remain essential for reducing global mortality. While existing prediction models primarily leverage structured data, unstructured clinical notes contain valuable early indicators. This study introduces a novel LLM-augmented clinical NLP pipeline that employs domain-adapted large language models for symptom extraction, contextual reasoning, and correlation from free-text reports. Our approach integrates cardiovascular-specific fine-tuning, prompt-based inference, and entity-aware reasoning. Evaluations on MIMIC-III and CARDIO-NLP datasets demonstrate improved performance in precision, recall, F1-score, and AUROC, with high clinical relevance (kappa = 0.82) assessed by cardiologists. Challenges such as contextual hallucination, which occurs when plausible information contracts with provided source, and temporal ambiguity, which is related with models struggling with chronological ordering of events are addressed using prompt engineering and hybrid rule-based verification. This work underscores the potential of LLMs in clinical decision support systems (CDSS), advancing early warning systems and enhancing the translation of patient narratives into actionable risk assessments. 

---
# Enhancing the Capabilities of Large Language Models for API calls through Knowledge Graphs 

**Authors**: Ye Yang, Xue Xiao, Ping Yin, Taotao Xie  

**Link**: [PDF](https://arxiv.org/pdf/2507.10630)  

**Abstract**: API calls by large language models (LLMs) offer a cutting-edge approach for data analysis. However, their ability to effectively utilize tools via API calls remains underexplored in knowledge-intensive domains like meteorology. This paper introduces KG2data, a system that integrates knowledge graphs, LLMs, ReAct agents, and tool-use technologies to enable intelligent data acquisition and query handling in the meteorological field. Using a virtual API, we evaluate API call accuracy across three metrics: name recognition failure, hallucination failure, and call correctness. KG2data achieves superior performance (1.43%, 0%, 88.57%) compared to RAG2data (16%, 10%, 72.14%) and chat2data (7.14%, 8.57%, 71.43%). KG2data differs from typical LLM-based systems by addressing their limited access to domain-specific knowledge, which hampers performance on complex or terminology-rich queries. By using a knowledge graph as persistent memory, our system enhances content retrieval, complex query handling, domain-specific reasoning, semantic relationship resolution, and heterogeneous data integration. It also mitigates the high cost of fine-tuning LLMs, making the system more adaptable to evolving domain knowledge and API structures. In summary, KG2data provides a novel solution for intelligent, knowledge-based question answering and data analysis in domains with high knowledge demands. 

---
# First-Order Error Matters: Accurate Compensation for Quantized Large Language Models 

**Authors**: Xingyu Zheng, Haotong Qin, Yuye Li, Jiakai Wang, Jinyang Guo, Michele Magno, Xianglong Liu  

**Link**: [PDF](https://arxiv.org/pdf/2507.11017)  

**Abstract**: Post-training quantization (PTQ) offers an efficient approach to compressing large language models (LLMs), significantly reducing memory access and computational costs. Existing compensation-based weight calibration methods often rely on a second-order Taylor expansion to model quantization error, under the assumption that the first-order term is negligible in well-trained full-precision models. However, we reveal that the progressive compensation process introduces accumulated first-order deviations between latent weights and their full-precision counterparts, making this assumption fundamentally flawed. To address this, we propose FOEM, a novel PTQ method that explicitly incorporates first-order gradient terms to improve quantization error compensation. FOEM approximates gradients by directly computing the difference between latent and full-precision weights, avoiding the high cost and limited generalization of backpropagation-based gradient computation. This approach introduces minimal additional computational overhead. Moreover, FOEM leverages precomputed Cholesky factors to efficiently recover the inverse of Hessian submatrices in real time. Extensive experiments across a wide range of models and benchmarks demonstrate that FOEM consistently outperforms the classical GPTQ method. In 3-bit weight-only quantization, FOEM reduces the perplexity of Llama3-8B by 89.6%, and improves the 5-shot MMLU accuracy of Llama3-70B from 51.7% to 74.9%, approaching the full-precision performance of 78.6%. Furthermore, FOEM can be seamlessly integrated with advanced techniques such as GPTAQ and SpinQuant, yielding additional improvements under the challenging W4A4KV4 setting, and further narrowing the accuracy gap with full-precision baselines beyond what current state-of-the-art methods achieve. The code is available at this https URL. 

---
# KisMATH: Do LLMs Have Knowledge of Implicit Structures in Mathematical Reasoning? 

**Authors**: Soumadeep Saha, Akshay Chaturvedi, Saptarshi Saha, Utpal Garain, Nicholas Asher  

**Link**: [PDF](https://arxiv.org/pdf/2507.11408)  

**Abstract**: Chain-of-thought traces have been shown to improve performance of large language models in a plethora of reasoning tasks, yet there is no consensus on the mechanism through which this performance boost is achieved. To shed more light on this, we introduce Causal CoT Graphs (CCGs), which are directed acyclic graphs automatically extracted from reasoning traces that model fine-grained causal dependencies in the language model output. A collection of $1671$ mathematical reasoning problems from MATH500, GSM8K and AIME, and their associated CCGs are compiled into our dataset -- \textbf{KisMATH}. Our detailed empirical analysis with 15 open-weight LLMs shows that (i) reasoning nodes in the CCG are mediators for the final answer, a condition necessary for reasoning; and (ii) LLMs emphasise reasoning paths given by the CCG, indicating that models internally realise structures akin to our graphs. KisMATH enables controlled, graph-aligned interventions and opens up avenues for further investigation into the role of chain-of-thought in LLM reasoning. 

---
# Modeling Understanding of Story-Based Analogies Using Large Language Models 

**Authors**: Kalit Inani, Keshav Kabra, Vijay Marupudi, Sashank Varma  

**Link**: [PDF](https://arxiv.org/pdf/2507.10957)  

**Abstract**: Recent advancements in Large Language Models (LLMs) have brought them closer to matching human cognition across a variety of tasks. How well do these models align with human performance in detecting and mapping analogies? Prior research has shown that LLMs can extract similarities from analogy problems but lack robust human-like reasoning. Building on Webb, Holyoak, and Lu (2023), the current study focused on a story-based analogical mapping task and conducted a fine-grained evaluation of LLM reasoning abilities compared to human performance. First, it explored the semantic representation of analogies in LLMs, using sentence embeddings to assess whether they capture the similarity between the source and target texts of an analogy, and the dissimilarity between the source and distractor texts. Second, it investigated the effectiveness of explicitly prompting LLMs to explain analogies. Throughout, we examine whether LLMs exhibit similar performance profiles to those observed in humans by evaluating their reasoning at the level of individual analogies, and not just at the level of overall accuracy (as prior studies have done). Our experiments include evaluating the impact of model size (8B vs. 70B parameters) and performance variation across state-of-the-art model architectures such as GPT-4 and LLaMA3. This work advances our understanding of the analogical reasoning abilities of LLMs and their potential as models of human reasoning. 

---
# MalCodeAI: Autonomous Vulnerability Detection and Remediation via Language Agnostic Code Reasoning 

**Authors**: Jugal Gajjar, Kamalasankari Subramaniakuppusamy, Noha El Kachach  

**Link**: [PDF](https://arxiv.org/pdf/2507.10898)  

**Abstract**: The growing complexity of cyber threats and the limitations of traditional vulnerability detection tools necessitate novel approaches for securing software systems. We introduce MalCodeAI, a language-agnostic, multi-stage AI pipeline for autonomous code security analysis and remediation. MalCodeAI combines code decomposition and semantic reasoning using fine-tuned Qwen2.5-Coder-3B-Instruct models, optimized through Low-Rank Adaptation (LoRA) within the MLX framework, and delivers scalable, accurate results across 14 programming languages. In Phase 1, the model achieved a validation loss as low as 0.397 for functional decomposition and summarization of code segments after 200 iterations, 6 trainable layers, and a learning rate of 2 x 10^(-5). In Phase 2, for vulnerability detection and remediation, it achieved a best validation loss of 0.199 using the same number of iterations and trainable layers but with an increased learning rate of 4 x 10^(-5), effectively identifying security flaws and suggesting actionable fixes. MalCodeAI supports red-hat-style exploit tracing, CVSS-based risk scoring, and zero-shot generalization to detect complex, zero-day vulnerabilities. In a qualitative evaluation involving 15 developers, the system received high scores in usefulness (mean 8.06/10), interpretability (mean 7.40/10), and readability of outputs (mean 7.53/10), confirming its practical value in real-world development workflows. This work marks a significant advancement toward intelligent, explainable, and developer-centric software security solutions. 

---
# Artificial Finance: How AI Thinks About Money 

**Authors**: Orhan Erdem, Ragavi Pobbathi Ashok  

**Link**: [PDF](https://arxiv.org/pdf/2507.10933)  

**Abstract**: In this paper, we explore how large language models (LLMs) approach financial decision-making by systematically comparing their responses to those of human participants across the globe. We posed a set of commonly used financial decision-making questions to seven leading LLMs, including five models from the GPT series(GPT-4o, GPT-4.5, o1, o3-mini), Gemini 2.0 Flash, and DeepSeek R1. We then compared their outputs to human responses drawn from a dataset covering 53 nations. Our analysis reveals three main results. First, LLMs generally exhibit a risk-neutral decision-making pattern, favoring choices aligned with expected value calculations when faced with lottery-type questions. Second, when evaluating trade-offs between present and future, LLMs occasionally produce responses that appear inconsistent with normative reasoning. Third, when we examine cross-national similarities, we find that the LLMs' aggregate responses most closely resemble those of participants from Tanzania. These findings contribute to the understanding of how LLMs emulate human-like decision behaviors and highlight potential cultural and training influences embedded within their outputs. 

---
# SPICEAssistant: LLM using SPICE Simulation Tools for Schematic Design of Switched-Mode Power Supplies 

**Authors**: Simon Nau, Jan Krummenauer, André Zimmermann  

**Link**: [PDF](https://arxiv.org/pdf/2507.10639)  

**Abstract**: State-of-the-art large language models (LLMs) show high performance across a wide range of tasks in many domains of science. In the field of electronic design automation (EDA), it is yet to be determined to what extent they are capable to understand, adapt, and dimension electronic circuits. This paper focuses on the application of LLMs to switched-mode power supply (SMPS) design on printed circuit boards (PCBs). Particular challenges for LLMs in this context include their limited ability to interpret results from key simulation tools like SPICE and the multi-step design process. To address these challenges, we suggest SPICEAssistant, a framework that provides a broad selection of tools to an LLM. The tools serve as an interface to SPICE, allowing the LLM to interact flexibly with the simulator to estimate the impact of its modifications to the circuit. To evaluate the performance of SPICEAssistant, we defined a benchmark consisting of 256 questions testing the ability to adapt circuit netlists to fulfil different SMPS design tasks. The benchmarking results show that simulation feedback effectively improves SMPS design capabilities of LLMs. An increasing number of simulation iterations leads to enhanced performance. The SPICEAssistant framework significantly outperforms the standalone LLM GPT-4o on the benchmark by approximately 38%. 

---
# HanjaBridge: Resolving Semantic Ambiguity in Korean LLMs via Hanja-Augmented Pre-Training 

**Authors**: Seungho Choi  

**Link**: [PDF](https://arxiv.org/pdf/2507.10920)  

**Abstract**: Large language models (LLMs) often show poor performance in low-resource languages like Korean, partly due to unique linguistic challenges such as homophonous Sino-Korean words that are indistinguishable in Hangul script. To address this semantic ambiguity, we propose HanjaBridge, a novel meaning-injection technique integrated into a continual pre-training (CPT) framework. Instead of deterministically mapping a word to a single Hanja (Chinese character), HanjaBridge presents the model with all possible Hanja candidates for a given homograph, encouraging the model to learn contextual disambiguation. This process is paired with token-level knowledge distillation to prevent catastrophic forgetting. Experimental results show that HanjaBridge significantly improves Korean language understanding, achieving a 21\% relative improvement on the KoBALT benchmark. Notably, by reinforcing semantic alignment between Korean and Chinese through shared Hanja, we observe a strong positive cross-lingual transfer. Furthermore, these gains persist even when Hanja augmentation is omitted at inference time, ensuring practical efficiency with no additional run-time cost. 

---
# GHPO: Adaptive Guidance for Stable and Efficient LLM Reinforcement Learning 

**Authors**: Ziru Liu, Cheng Gong, Xinyu Fu, Yaofang Liu, Ran Chen, Shoubo Hu, Suiyun Zhang, Rui Liu, Qingfu Zhang, Dandan Tu  

**Link**: [PDF](https://arxiv.org/pdf/2507.10628)  

**Abstract**: Reinforcement Learning with Verifiable Rewards (RLVR) has recently emerged as a powerful paradigm for facilitating the self-improvement of large language models (LLMs), particularly in the domain of complex reasoning tasks. However, prevailing on-policy RL methods often contend with significant training instability and inefficiency. This is primarily due to a capacity-difficulty mismatch, where the complexity of training data frequently outpaces the model's current capabilities, leading to critically sparse reward signals and stalled learning progress. This challenge is particularly acute for smaller, more resource-efficient LLMs. To overcome this, we introduce the Guided Hybrid Policy Optimization (GHPO), a novel difficulty-aware reinforcement learning framework. GHPO dynamically calibrates task difficulty by employing adaptive prompt refinement to provide targeted guidance. This unique approach adaptively balances direct imitation learning for problems currently beyond the model's reach with exploration-based reinforcement learning for more manageable tasks, effectively creating a smooth and optimized learning curriculum. Extensive experiments demonstrate that GHPO achieves an average performance gain of approximately 5% across six challenging mathematics benchmarks, consistently outperforming strong on-policy reinforcement learning and curriculum learning baselines. Further analysis confirms that our framework significantly enhances both training stability and final reasoning performance, thus offering a scalable and efficient solution for developing powerful and robust reasoning models. 

---
# LLMs Meet Cross-Modal Time Series Analytics: Overview and Directions 

**Authors**: Chenxi Liu, Hao Miao, Cheng Long, Yan Zhao, Ziyue Li, Panos Kalnis  

**Link**: [PDF](https://arxiv.org/pdf/2507.10620)  

**Abstract**: Large Language Models (LLMs) have emerged as a promising paradigm for time series analytics, leveraging their massive parameters and the shared sequential nature of textual and time series data. However, a cross-modality gap exists between time series and textual data, as LLMs are pre-trained on textual corpora and are not inherently optimized for time series. In this tutorial, we provide an up-to-date overview of LLM-based cross-modal time series analytics. We introduce a taxonomy that classifies existing approaches into three groups based on cross-modal modeling strategies, e.g., conversion, alignment, and fusion, and then discuss their applications across a range of downstream tasks. In addition, we summarize several open challenges. This tutorial aims to expand the practical application of LLMs in solving real-world problems in cross-modal time series analytics while balancing effectiveness and efficiency. Participants will gain a thorough understanding of current advancements, methodologies, and future research directions in cross-modal time series analytics. 

---
# Fine-tuning Large Language Model for Automated Algorithm Design 

**Authors**: Fei Liu, Rui Zhang, Xi Lin, Zhichao Lu, Qingfu Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2507.10614)  

**Abstract**: The integration of large language models (LLMs) into automated algorithm design has shown promising potential. A prevalent approach embeds LLMs within search routines to iteratively generate and refine candidate algorithms. However, most existing methods rely on off-the-shelf LLMs trained for general coding tasks,leaving a key question open: Do we need LLMs specifically tailored for algorithm design? If so, how can such LLMs be effectively obtained and how well can they generalize across different algorithm design tasks? In this paper, we take a first step toward answering these questions by exploring fine-tuning of LLMs for algorithm design. We introduce a Diversity-Aware Rank based (DAR) sampling strategy to balance training data diversity and quality, then we leverage direct preference optimization to efficiently align LLM outputs with task objectives. Our experiments, conducted on Llama-3.2-1B-Instruct and Llama- 3.1-8B-Instruct, span three distinct algorithm design tasks. Results suggest that finetuned LLMs can significantly outperform their off-the-shelf counterparts with the smaller Llama-3.2-1B-Instruct and match the larger Llama-3.1-8B-Instruct on the admissible set problem. Moreover, we observe promising generalization: LLMs finetuned on specific algorithm design tasks also improve performance on related tasks with varying settings. These findings highlight the value of task-specific adaptation for LLMs in algorithm design and open new avenues for future research. 

---
# Warehouse Spatial Question Answering with LLM Agent 

**Authors**: Hsiang-Wei Huang, Jen-Hao Cheng, Kuang-Ming Chen, Cheng-Yen Yang, Bahaa Alattar, Yi-Ru Lin, Pyongkun Kim, Sangwon Kim, Kwangju Kim, Chung-I Huang, Jenq-Neng Hwang  

**Link**: [PDF](https://arxiv.org/pdf/2507.10778)  

**Abstract**: Spatial understanding has been a challenging task for existing Multi-modal Large Language Models~(MLLMs). Previous methods leverage large-scale MLLM finetuning to enhance MLLM's spatial understanding ability. In this paper, we present a data-efficient approach. We propose a LLM agent system with strong and advanced spatial reasoning ability, which can be used to solve the challenging spatial question answering task in complex indoor warehouse scenarios. Our system integrates multiple tools that allow the LLM agent to conduct spatial reasoning and API tools interaction to answer the given complicated spatial question. Extensive evaluations on the 2025 AI City Challenge Physical AI Spatial Intelligence Warehouse dataset demonstrate that our system achieves high accuracy and efficiency in tasks such as object retrieval, counting, and distance estimation. The code is available at: this https URL 

---
# PLEX: Perturbation-free Local Explanations for LLM-Based Text Classification 

**Authors**: Yogachandran Rahulamathavan, Misbah Farooq, Varuna De Silva  

**Link**: [PDF](https://arxiv.org/pdf/2507.10596)  

**Abstract**: Large Language Models (LLMs) excel in text classification, but their complexity hinders interpretability, making it difficult to understand the reasoning behind their predictions. Explainable AI (XAI) methods like LIME and SHAP offer local explanations by identifying influential words, but they rely on computationally expensive perturbations. These methods typically generate thousands of perturbed sentences and perform inferences on each, incurring a substantial computational burden, especially with LLMs. To address this, we propose \underline{P}erturbation-free \underline{L}ocal \underline{Ex}planation (PLEX), a novel method that leverages the contextual embeddings extracted from the LLM and a ``Siamese network" style neural network trained to align with feature importance scores. This one-off training eliminates the need for subsequent perturbations, enabling efficient explanations for any new sentence. We demonstrate PLEX's effectiveness on four different classification tasks (sentiment, fake news, fake COVID-19 news and depression), showing more than 92\% agreement with LIME and SHAP. Our evaluation using a ``stress test" reveals that PLEX accurately identifies influential words, leading to a similar decline in classification accuracy as observed with LIME and SHAP when these words are removed. Notably, in some cases, PLEX demonstrates superior performance in capturing the impact of key features. PLEX dramatically accelerates explanation, reducing time and computational overhead by two and four orders of magnitude, respectively. This work offers a promising solution for explainable LLM-based text classification. 

---
# Emergence of Hierarchical Emotion Organization in Large Language Models 

**Authors**: Bo Zhao, Maya Okawa, Eric J. Bigelow, Rose Yu, Tomer Ullman, Ekdeep Singh Lubana, Hidenori Tanaka  

**Link**: [PDF](https://arxiv.org/pdf/2507.10599)  

**Abstract**: As large language models (LLMs) increasingly power conversational agents, understanding how they model users' emotional states is critical for ethical deployment. Inspired by emotion wheels -- a psychological framework that argues emotions organize hierarchically -- we analyze probabilistic dependencies between emotional states in model outputs. We find that LLMs naturally form hierarchical emotion trees that align with human psychological models, and larger models develop more complex hierarchies. We also uncover systematic biases in emotion recognition across socioeconomic personas, with compounding misclassifications for intersectional, underrepresented groups. Human studies reveal striking parallels, suggesting that LLMs internalize aspects of social perception. Beyond highlighting emergent emotional reasoning in LLMs, our results hint at the potential of using cognitively-grounded theories for developing better model evaluations. 

---
# AutoRAG-LoRA: Hallucination-Triggered Knowledge Retuning via Lightweight Adapters 

**Authors**: Kaushik Dwivedi, Padmanabh Patanjali Mishra  

**Link**: [PDF](https://arxiv.org/pdf/2507.10586)  

**Abstract**: Large Language Models (LLMs) have demonstrated remarkable fluency across a range of natural language tasks, yet remain vulnerable to hallucinations - factual inaccuracies that undermine trust in real world deployment. We present AutoRAG-LoRA, a modular framework for Retrieval-Augmented Generation (RAG) that tackles hallucination in large language models through lightweight LoRA-based adapters and KL-regularized training. Our pipeline integrates automated prompt rewriting, hybrid retrieval, and low-rank adapter tuning to ground responses in retrieved evidence. A hallucination detection module, using both classifier-based and self-evaluation techniques, assigns confidence scores to generated outputs, triggering an optional feedback correction loop. This loop enforces factual alignment via contrastive KL loss and adapter fine tuning. We demonstrate that AutoRAG-LoRA significantly reduces the factual drift while preserving the efficiency and modularity of the model. 

---
# ARPaCCino: An Agentic-RAG for Policy as Code Compliance 

**Authors**: Francesco Romeo, Luigi Arena, Francesco Blefari, Francesco Aurelio Pironti, Matteo Lupinacci, Angelo Furfaro  

**Link**: [PDF](https://arxiv.org/pdf/2507.10584)  

**Abstract**: Policy as Code (PaC) is a paradigm that encodes security and compliance policies into machine-readable formats, enabling automated enforcement in Infrastructure as Code (IaC) environments. However, its adoption is hindered by the complexity of policy languages and the risk of misconfigurations. In this work, we present ARPaCCino, an agentic system that combines Large Language Models (LLMs), Retrieval-Augmented-Generation (RAG), and tool-based validation to automate the generation and verification of PaC rules. Given natural language descriptions of the desired policies, ARPaCCino generates formal Rego rules, assesses IaC compliance, and iteratively refines the IaC configurations to ensure conformance. Thanks to its modular agentic architecture and integration with external tools and knowledge bases, ARPaCCino supports policy validation across a wide range of technologies, including niche or emerging IaC frameworks. Experimental evaluation involving a Terraform-based case study demonstrates ARPaCCino's effectiveness in generating syntactically and semantically correct policies, identifying non-compliant infrastructures, and applying corrective modifications, even when using smaller, open-weight LLMs. Our results highlight the potential of agentic RAG architectures to enhance the automation, reliability, and accessibility of PaC workflows. 

---
# CodeAssistBench (CAB): Dataset & Benchmarking for Multi-turn Chat-Based Code Assistance 

**Authors**: Myeongsoo Kim, Shweta Garg, Baishakhi Ray, Varun Kumar, Anoop Deoras  

**Link**: [PDF](https://arxiv.org/pdf/2507.10646)  

**Abstract**: Programming assistants powered by large language models have transformed software development, yet most benchmarks focus narrowly on code generation tasks. Recent efforts like InfiBench and StackEval attempt to address this gap using Stack Overflow data but remain limited to single-turn interactions in isolated contexts, require significant manual curation, and fail to represent complete project environments. We introduce CodeAssistBench (CAB), the first benchmark framework for evaluating multi-turn programming assistance in realistic settings that address real-world questions about actual codebases. Unlike existing programming Q&A benchmarks, CAB automatically generates scalable datasets from question-related GitHub issues using configurable parameters (e.g., repository creation date, star count, programming languages), and includes automatic containerization of codebases for evaluation. It then evaluates models through simulated users in these containerized environments with full codebase access. Using this framework, we constructed a test set of 3,286 real-world programming questions across 231 repositories, spanning seven programming languages and diverse problem domains. Our evaluation of leading LLMs reveals a substantial capability gap: while models perform well on Stack Overflow questions with success rates of 70-83%, they resolve only up to 16.49% of CAB's recent issues. This discrepancy highlights the challenges of providing assistance in complex, project-specific contexts versus answering standalone questions. 

---
# RedOne: Revealing Domain-specific LLM Post-Training in Social Networking Services 

**Authors**: Fei Zhao, Chonggang Lu, Yue Wang, Zheyong Xie, Ziyan Liu, Haofu Qian, JianZhao Huang, Fangcheng Shi, Zijie Meng, Hongcheng Guo, Mingqian He, Xinze Lyu, Yiming Lu, Ziyang Xiang, Zheyu Ye, Chengqiang Lu, Zhe Xu, Yi Wu, Yao Hu, Yan Gao, Jun Fan, Xiaolong Jiang, Weiting Liu, Boyang Wang, Shaosheng Cao  

**Link**: [PDF](https://arxiv.org/pdf/2507.10605)  

**Abstract**: As a primary medium for modern information dissemination, social networking services (SNS) have experienced rapid growth, which has proposed significant challenges for platform content management and interaction quality improvement. Recently, the development of large language models (LLMs) has offered potential solutions but existing studies focus on isolated tasks, which not only encounter diminishing benefit from the data scaling within individual scenarios but also fail to flexibly adapt to diverse real-world context. To address these challenges, we introduce RedOne, a domain-specific LLM designed to break the performance bottleneck of single-task baselines and establish a comprehensive foundation for the SNS. RedOne was developed through a three-stage training strategy consisting of continue pretraining, supervised fine-tuning, and preference optimization, using a large-scale real-world dataset. Through extensive experiments, RedOne maintains strong general capabilities, and achieves an average improvement up to 14.02% across 8 major SNS tasks and 7.56% in SNS bilingual evaluation benchmark, compared with base models. Furthermore, through online testing, RedOne reduced the exposure rate in harmful content detection by 11.23% and improved the click page rate in post-view search by 14.95% compared with single-tasks finetuned baseline models. These results establish RedOne as a robust domain-specific LLM for SNS, demonstrating excellent generalization across various tasks and promising applicability in real-world scenarios. 

---
# An Offline Mobile Conversational Agent for Mental Health Support: Learning from Emotional Dialogues and Psychological Texts with Student-Centered Evaluation 

**Authors**: Vimaleswar A, Prabhu Nandan Sahu, Nilesh Kumar Sahu, Haroon R Lone  

**Link**: [PDF](https://arxiv.org/pdf/2507.10580)  

**Abstract**: Mental health plays a crucial role in the overall well-being of an individual. In recent years, digital platforms have been increasingly used to expand mental health and emotional support. However, there are persistent challenges related to limited user accessibility, internet connectivity, and data privacy, which highlight the need for an offline, smartphone-based solution. To address these challenges, we propose EmoSApp (Emotional Support App): an entirely offline, smartphone-based conversational app designed for mental health and emotional support. The system leverages Large Language Models (LLMs), specifically fine-tuned, quantized and deployed using Torchtune and Executorch for resource-constrained devices, allowing all inferences to occur on the smartphone. To equip EmoSApp with robust domain expertise, we fine-tuned the LLaMA-3.2-1B-Instruct model on our custom curated ``Knowledge dataset'' of 14,582 mental-health QA pairs, along with the multi-turn conversational data.
Through qualitative human evaluation with the student population, we demonstrate that EmoSApp has the ability to respond coherently, empathetically, maintain interactive dialogue, and provide relevant suggestions to user's mental health problems. Additionally, quantitative evaluations on nine standard commonsense and reasoning benchmarks demonstrate the efficacy of our fine-tuned, quantized model in low-resource settings. By prioritizing on-device deployment and specialized domain adaptation, EmoSApp serves as a blueprint for future innovations in portable, secure, and highly tailored AI-driven mental health solutions. 

---
# Can Large Language Models Understand As Well As Apply Patent Regulations to Pass a Hands-On Patent Attorney Test? 

**Authors**: Bhakti Khera, Rezvan Alamian, Pascal A. Scherz, Stephan M. Goetz  

**Link**: [PDF](https://arxiv.org/pdf/2507.10576)  

**Abstract**: The legal field already uses various large language models (LLMs) in actual applications, but their quantitative performance and reasons for it are underexplored. We evaluated several open-source and proprietary LLMs -- including GPT-series, Anthropic, Deepseek and Llama-3, variants -- on parts of the European Qualifying Examination (EQE) for future European Patent Attorneys. OpenAI o1 led with 0.82 accuracy and 0.81 F1 score, whereas (Amazon Web Services) AWS Llama 3.1 8B lagged at 0.50 accuracy, and a Python-deployed Llama 3.1 8B scored 0.55. The latter two are within the range of mere guessing for the two-answer forced-choice design. None of the evaluated models could have passed the examination fully, as accuracy never exceeded the average threshold of 0.90 required for professional-level standards -- also not models that are regularly promoted for their assumed beyond-PhD- and bar-admitted-lawyer-level performance. GPT-4o excelled at integrating text and graphics, while Claude 3 Opus often lost formatting coherence. Human patent experts evaluated the textual justifications and uncovered various critical shortcomings of each model. They valued clarity and legal rationale over the raw correctness of the answers, which revealed misalignment between automatic metrics and expert judgment. Model outputs were sensitive to modest temperature changes and prompt wording, which underscores the remaining necessity of expert oversight. Future work should target logical consistency, robust multimodality, and adaptive prompting to approach human-level patent proficiency. In summary, despite the outstanding performance of recent large models, the general public might overestimate their performance. The field has a long way to go to develop a virtual patent attorney. This paper wants to point out several specific limitations that need solutions. 

---
# Findings of the BEA 2025 Shared Task on Pedagogical Ability Assessment of AI-powered Tutors 

**Authors**: Ekaterina Kochmar, Kaushal Kumar Maurya, Kseniia Petukhova, KV Aditya Srivatsa, Anaïs Tack, Justin Vasselli  

**Link**: [PDF](https://arxiv.org/pdf/2507.10579)  

**Abstract**: This shared task has aimed to assess pedagogical abilities of AI tutors powered by large language models (LLMs), focusing on evaluating the quality of tutor responses aimed at student's mistake remediation within educational dialogues. The task consisted of five tracks designed to automatically evaluate the AI tutor's performance across key dimensions of mistake identification, precise location of the mistake, providing guidance, and feedback actionability, grounded in learning science principles that define good and effective tutor responses, as well as the track focusing on detection of the tutor identity. The task attracted over 50 international teams across all tracks. The submitted models were evaluated against gold-standard human annotations, and the results, while promising, show that there is still significant room for improvement in this domain: the best results for the four pedagogical ability assessment tracks range between macro F1 scores of 58.34 (for providing guidance) and 71.81 (for mistake identification) on three-class problems, with the best F1 score in the tutor identification track reaching 96.98 on a 9-class task. In this paper, we overview the main findings of the shared task, discuss the approaches taken by the teams, and analyze their performance. All resources associated with this task are made publicly available to support future research in this critical domain. 

---
# NLP Meets the World: Toward Improving Conversations With the Public About Natural Language Processing Research 

**Authors**: Shomir Wilson  

**Link**: [PDF](https://arxiv.org/pdf/2507.10559)  

**Abstract**: Recent developments in large language models (LLMs) have been accompanied by rapidly growing public interest in natural language processing (NLP). This attention is reflected by major news venues, which sometimes invite NLP researchers to share their knowledge and views with a wide audience. Recognizing the opportunities of the present, for both the research field and for individual researchers, this paper shares recommendations for communicating with a general audience about LLMs' capabilities and limitations. These recommendations cover three themes: vague terminology as an obstacle to public understanding, unreasonable expectations as obstacles to sustainable growth, and ethical failures as obstacles to continued support. Published NLP research and popular news coverage are cited to illustrate these themes with examples. The recommendations promote effective, transparent communication with the general public about NLP, in order to strengthen public understanding and encourage support for research. 

---
# Anthropomimetic Uncertainty: What Verbalized Uncertainty in Language Models is Missing 

**Authors**: Dennis Ulmer, Alexandra Lorson, Ivan Titov, Christian Hardmeier  

**Link**: [PDF](https://arxiv.org/pdf/2507.10587)  

**Abstract**: Human users increasingly rely on natural language interactions with large language models (LLMs) in order to receive help on a large variety of tasks and problems. However, the trustworthiness and perceived legitimacy of LLMs is undermined by the fact that their output is frequently stated in very confident terms, even when its accuracy is questionable. Therefore, there is a need to signal the confidence of the language model to a user in order to reap the benefits of human-machine collaboration and mitigate potential harms. Verbalized uncertainty is the expression of confidence with linguistic means, an approach that integrates perfectly into language-based interfaces. Nevertheless, most recent research in natural language processing (NLP) overlooks the nuances surrounding human uncertainty communication and the data biases that influence machine uncertainty communication. We argue for anthropomimetic uncertainty, meaning that intuitive and trustworthy uncertainty communication requires a degree of linguistic authenticity and personalization to the user, which could be achieved by emulating human communication. We present a thorough overview over the research in human uncertainty communication, survey ongoing research, and perform additional analyses to demonstrate so-far overlooked biases in verbalized uncertainty. We conclude by pointing out unique factors in human-machine communication of uncertainty and deconstruct anthropomimetic uncertainty into future research directions for NLP. 

---
# Reasoning Strategies in Large Language Models: Can They Follow, Prefer, and Optimize? 

**Authors**: Yanjian Zhang, Guillaume Wisniewski, Nadi Tomeh, Thierry Charnois  

**Link**: [PDF](https://arxiv.org/pdf/2507.11423)  

**Abstract**: Human reasoning involves different strategies, each suited to specific problems. Prior work shows that large language model (LLMs) tend to favor a single reasoning strategy, potentially limiting their effectiveness in diverse reasoning challenges. In this work, we investigate whether prompting can control LLMs reasoning strategies and assess its impact on logical problem-solving. While our experiments show that no single strategy consistently improves accuracy, performance could be enhanced if models could adaptively choose the optimal strategy. We propose methods to guide LLMs in strategy selection, highlighting new ways to refine their reasoning abilities. 

---
# HKGAI-V1: Towards Regional Sovereign Large Language Model for Hong Kong 

**Authors**: Sirui Han, Junqi Zhu, Ruiyuan Zhang, Yike Guo  

**Link**: [PDF](https://arxiv.org/pdf/2507.11502)  

**Abstract**: This paper presents the development of HKGAI-V1, a foundational sovereign large language model (LLM), developed as part of an initiative to establish value-aligned AI infrastructure specifically tailored for Hong Kong. Addressing the region's unique multilingual environment (Cantonese, Mandarin, and English), its distinct socio-legal context under the "one country, two systems" framework, and specific local cultural and value considerations, the model is built upon the DeepSeek architecture and systematically aligned with regional norms through a multifaceted full parameter fine-tuning process. It is further integrated with a retrieval-augmented generation (RAG) system to ensure timely and factually grounded information access. The core contribution lies in the design and implementation of a comprehensive, region-specific AI alignment and safety framework, demonstrated through two key achievements: 1) The successful development of HKGAI-V1 itself - which outper-forms general-purpose models in handling Hong Kong-specific culturally sensitive queries, and embodies a "governance-embedded" approach to digital sovereignty - empowers Hong Kong to exercise control over AI applications in critical sectors including public services, legal systems, and edu-cation. 2) The development of the proprietary Adversarial HK Value Benchmark, a rigorous tool for evaluating model alignment with local ethical and legal stand-ards under challenging conditions. By documenting these achievements, the paper provides not only a technological artifact but also a replicable blueprint for developing advanced, regionally focused AI systems deeply rooted in their local identities. 

---
# What is the Best Process Model Representation? A Comparative Analysis for Process Modeling with Large Language Models 

**Authors**: Alexis Brissard, Frédéric Cuppens, Amal Zouaq  

**Link**: [PDF](https://arxiv.org/pdf/2507.11356)  

**Abstract**: Large Language Models (LLMs) are increasingly applied for Process Modeling (PMo) tasks such as Process Model Generation (PMG). To support these tasks, researchers have introduced a variety of Process Model Representations (PMRs) that serve as model abstractions or generation targets. However, these PMRs differ widely in structure, complexity, and usability, and have never been systematically compared. Moreover, recent PMG approaches rely on distinct evaluation strategies and generation techniques, making comparison difficult. This paper presents the first empirical study that evaluates multiple PMRs in the context of PMo with LLMs. We introduce the PMo Dataset, a new dataset containing 55 process descriptions paired with models in nine different PMRs. We evaluate PMRs along two dimensions: suitability for LLM-based PMo and performance on PMG. \textit{Mermaid} achieves the highest overall score across six PMo criteria, whereas \textit{BPMN text} delivers the best PMG results in terms of process element similarity. 

---
# What Should LLMs Forget? Quantifying Personal Data in LLMs for Right-to-Be-Forgotten Requests 

**Authors**: Dimitri Staufer  

**Link**: [PDF](https://arxiv.org/pdf/2507.11128)  

**Abstract**: Large Language Models (LLMs) can memorize and reveal personal information, raising concerns regarding compliance with the EU's GDPR, particularly the Right to Be Forgotten (RTBF). Existing machine unlearning methods assume the data to forget is already known but do not address how to identify which individual-fact associations are stored in the model. Privacy auditing techniques typically operate at the population level or target a small set of identifiers, limiting applicability to individual-level data inquiries. We introduce WikiMem, a dataset of over 5,000 natural language canaries covering 243 human-related properties from Wikidata, and a model-agnostic metric to quantify human-fact associations in LLMs. Our approach ranks ground-truth values against counterfactuals using calibrated negative log-likelihood across paraphrased prompts. We evaluate 200 individuals across 15 LLMs (410M-70B parameters), showing that memorization correlates with subject web presence and model scale. We provide a foundation for identifying memorized personal data in LLMs at the individual level, enabling the dynamic construction of forget sets for machine unlearning and RTBF requests. 

---
# FMC: Formalization of Natural Language Mathematical Competition Problems 

**Authors**: Jiaxuan Xie, Chengwu Liu, Ye Yuan, Siqi Li, Zhiping Xiao, Ming Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2507.11275)  

**Abstract**: Efficient and accurate autoformalization methods, which leverage large-scale datasets of extensive natural language mathematical problems to construct formal language datasets, are key to advancing formal mathematical reasoning. In this paper, we propose an autoformalization pipeline based on large language models with error feedback, achieving a fully automatic and training-free formalization approach. Using this pipeline, we curate an Olympiad-level dataset aligning natural language problems with Lean formalizations. The dataset comprises $3,922$ mathematical problems in natural language and $9,787$ in Lean, of which $64.46\%$ were assessed as at least above-average quality, making it suitable as a benchmark for automated theorem provers. Additionally, we investigate the formalization and reasoning capabilities of various LLMs and empirically demonstrate that few-shot learning, error feedback, and increasing sampling numbers enhance the autoformalization process. Experiments of three automated theorem provers on the \dataset\ dataset also highlight its challenging nature and its value as a benchmark for formal reasoning tasks. 

---
# Multi-Trigger Poisoning Amplifies Backdoor Vulnerabilities in LLMs 

**Authors**: Sanhanat Sivapiromrat, Caiqi Zhang, Marco Basaldella, Nigel Collier  

**Link**: [PDF](https://arxiv.org/pdf/2507.11112)  

**Abstract**: Recent studies have shown that Large Language Models (LLMs) are vulnerable to data poisoning attacks, where malicious training examples embed hidden behaviours triggered by specific input patterns. However, most existing works assume a phrase and focus on the attack's effectiveness, offering limited understanding of trigger mechanisms and how multiple triggers interact within the model. In this paper, we present a framework for studying poisoning in LLMs. We show that multiple distinct backdoor triggers can coexist within a single model without interfering with each other, enabling adversaries to embed several triggers concurrently. Using multiple triggers with high embedding similarity, we demonstrate that poisoned triggers can achieve robust activation even when tokens are substituted or separated by long token spans. Our findings expose a broader and more persistent vulnerability surface in LLMs. To mitigate this threat, we propose a post hoc recovery method that selectively retrains specific model components based on a layer-wise weight difference analysis. Our method effectively removes the trigger behaviour with minimal parameter updates, presenting a practical and efficient defence against multi-trigger poisoning. 

---
# Beyond Traditional Algorithms: Leveraging LLMs for Accurate Cross-Border Entity Identification 

**Authors**: Andres Azqueta-Gavaldón, Joaquin Ramos Cosgrove  

**Link**: [PDF](https://arxiv.org/pdf/2507.11086)  

**Abstract**: The growing prevalence of cross-border financial activities in global markets has underscored the necessity of accurately identifying and classifying foreign entities. This practice is essential within the Spanish financial system for ensuring robust risk management, regulatory adherence, and the prevention of financial misconduct. This process involves a labor-intensive entity-matching task, where entities need to be validated against available reference sources. Challenges arise from linguistic variations, special characters, outdated names, and changes in legal forms, complicating traditional matching algorithms like Jaccard, cosine, and Levenshtein distances. These methods struggle with contextual nuances and semantic relationships, leading to mismatches. To address these limitations, we explore Large Language Models (LLMs) as a flexible alternative. LLMs leverage extensive training to interpret context, handle abbreviations, and adapt to legal transitions. We evaluate traditional methods, Hugging Face-based LLMs, and interface-based LLMs (e.g., Microsoft Copilot, Alibaba's Qwen 2.5) using a dataset of 65 Portuguese company cases. Results show traditional methods achieve accuracies over 92% but suffer high false positive rates (20-40%). Interface-based LLMs outperform, achieving accuracies above 93%, F1 scores exceeding 96%, and lower false positives (40-80%). 

---
# The Devil behind the mask: An emergent safety vulnerability of Diffusion LLMs 

**Authors**: Zichen Wen, Jiashu Qu, Dongrui Liu, Zhiyuan Liu, Ruixi Wu, Yicun Yang, Xiangqi Jin, Haoyun Xu, Xuyang Liu, Weijia Li, Chaochao Lu, Jing Shao, Conghui He, Linfeng Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2507.11097)  

**Abstract**: Diffusion-based large language models (dLLMs) have recently emerged as a powerful alternative to autoregressive LLMs, offering faster inference and greater interactivity via parallel decoding and bidirectional modeling. However, despite strong performance in code generation and text infilling, we identify a fundamental safety concern: existing alignment mechanisms fail to safeguard dLLMs against context-aware, masked-input adversarial prompts, exposing novel vulnerabilities. To this end, we present DIJA, the first systematic study and jailbreak attack framework that exploits unique safety weaknesses of dLLMs. Specifically, our proposed DIJA constructs adversarial interleaved mask-text prompts that exploit the text generation mechanisms of dLLMs, i.e., bidirectional modeling and parallel decoding. Bidirectional modeling drives the model to produce contextually consistent outputs for masked spans, even when harmful, while parallel decoding limits model dynamic filtering and rejection sampling of unsafe content. This causes standard alignment mechanisms to fail, enabling harmful completions in alignment-tuned dLLMs, even when harmful behaviors or unsafe instructions are directly exposed in the prompt. Through comprehensive experiments, we demonstrate that DIJA significantly outperforms existing jailbreak methods, exposing a previously overlooked threat surface in dLLM architectures. Notably, our method achieves up to 100% keyword-based ASR on Dream-Instruct, surpassing the strongest prior baseline, ReNeLLM, by up to 78.5% in evaluator-based ASR on JailbreakBench and by 37.7 points in StrongREJECT score, while requiring no rewriting or hiding of harmful content in the jailbreak prompt. Our findings underscore the urgent need for rethinking safety alignment in this emerging class of language models. Code is available at this https URL. 

---
# MSA at ImageCLEF 2025 Multimodal Reasoning: Multilingual Multimodal Reasoning With Ensemble Vision Language Models 

**Authors**: Seif Ahmed, Mohamed T. Younes, Abdelrahman Moustafa, Abdelrahman Allam, Hamza Moustafa  

**Link**: [PDF](https://arxiv.org/pdf/2507.11114)  

**Abstract**: We present a robust ensemble-based system for multilingual multimodal reasoning, designed for the ImageCLEF 2025 EXAMS V challenge. Our approach integrates Gemini 2.5 Flash for visual description, Gemini 1.5 Pro for caption refinement and consistency checks, and Gemini 2.5 Pro as a reasoner which handles final answer selection, all coordinated through carefully engineered few-shot and zero-shot prompts. We conducted an extensive ablation study, training several large language models (Gemini 2.5 Flash, Phi 4, Gemma 3, Mistral) on an English dataset and its multilingual augmented version. Additionally, we evaluated Gemini 2.5 Flash in a zero-shot setting for comparison and found it to substantially outperform the trained models. Prompt design also proved critical: enforcing concise, language-normalized formats and prohibiting explanatory text boosted model accuracy on the English validation set from 55.9% to 61.7%. On the official leaderboard, our system (Team MSA) achieved first place overall in the multilingual track with 81.4% accuracy, and led 11 out of 13 individual language tracks, with top results such as 95.07% for Croatian and 92.12% for Italian. These findings highlight that lightweight OCR-VLM ensembles, when paired with precise prompt strategies and cross-lingual augmentation, can outperform heavier end-to-end models in high-stakes, multilingual educational settings. 

---
# Teach Me Sign: Stepwise Prompting LLM for Sign Language Production 

**Authors**: Zhaoyi An, Rei Kawakami  

**Link**: [PDF](https://arxiv.org/pdf/2507.10972)  

**Abstract**: Large language models, with their strong reasoning ability and rich knowledge, have brought revolution to many tasks of AI, but their impact on sign language generation remains limited due to its complexity and unique rules. In this paper, we propose TEAch Me Sign (TEAM-Sign), treating sign language as another natural language. By fine-tuning an LLM, we enable it to learn the correspondence between text and sign language, and facilitate generation. Considering the differences between sign and spoken language, we employ a stepwise prompting strategy to extract the inherent sign language knowledge within the LLM, thereby supporting the learning and generation process. Experimental results on How2Sign and Phoenix14T datasets demonstrate that our approach effectively leverages both the sign language knowledge and reasoning capabilities of LLM to align the different distribution and grammatical rules between sign and spoken language. 

---
# LLMs on Trial: Evaluating Judicial Fairness for Large Language Models 

**Authors**: Yiran Hu, Zongyue Xue, Haitao Li, Siyuan Zheng, Qingjing Chen, Shaochun Wang, Xihan Zhang, Ning Zheng, Yun Liu, Qingyao Ai, Yiqun Liu, Charles L.A. Clarke, Weixing Shen  

**Link**: [PDF](https://arxiv.org/pdf/2507.10852)  

**Abstract**: Large Language Models (LLMs) are increasingly used in high-stakes fields where their decisions impact rights and equity. However, LLMs' judicial fairness and implications for social justice remain underexplored. When LLMs act as judges, the ability to fairly resolve judicial issues is a prerequisite to ensure their trustworthiness. Based on theories of judicial fairness, we construct a comprehensive framework to measure LLM fairness, leading to a selection of 65 labels and 161 corresponding values. Applying this framework to the judicial system, we compile an extensive dataset, JudiFair, comprising 177,100 unique case facts. To achieve robust statistical inference, we develop three evaluation metrics, inconsistency, bias, and imbalanced inaccuracy, and introduce a method to assess the overall fairness of multiple LLMs across various labels. Through experiments with 16 LLMs, we uncover pervasive inconsistency, bias, and imbalanced inaccuracy across models, underscoring severe LLM judicial unfairness. Particularly, LLMs display notably more pronounced biases on demographic labels, with slightly less bias on substance labels compared to procedure ones. Interestingly, increased inconsistency correlates with reduced biases, but more accurate predictions exacerbate biases. While we find that adjusting the temperature parameter can influence LLM fairness, model size, release date, and country of origin do not exhibit significant effects on judicial fairness. Accordingly, we introduce a publicly available toolkit containing all datasets and code, designed to support future research in evaluating and improving LLM fairness. 

---
# Dr.Copilot: A Multi-Agent Prompt Optimized Assistant for Improving Patient-Doctor Communication in Romanian 

**Authors**: Andrei Niculae, Adrian Cosma, Cosmin Dumitrache, Emilian Rǎdoi  

**Link**: [PDF](https://arxiv.org/pdf/2507.11299)  

**Abstract**: Text-based telemedicine has become increasingly common, yet the quality of medical advice in doctor-patient interactions is often judged more on how advice is communicated rather than its clinical accuracy. To address this, we introduce this http URL , a multi-agent large language model (LLM) system that supports Romanian-speaking doctors by evaluating and enhancing the presentation quality of their written responses. Rather than assessing medical correctness, this http URL provides feedback along 17 interpretable axes. The system comprises of three LLM agents with prompts automatically optimized via DSPy. Designed with low-resource Romanian data and deployed using open-weight models, it delivers real-time specific feedback to doctors within a telemedicine platform. Empirical evaluations and live deployment with 41 doctors show measurable improvements in user reviews and response quality, marking one of the first real-world deployments of LLMs in Romanian medical settings. 

---
# Team HUMANE at AVeriTeC 2025: HerO 2 for Efficient Fact Verification 

**Authors**: Yejun Yoon, Jaeyoon Jung, Seunghyun Yoon, Kunwoo Park  

**Link**: [PDF](https://arxiv.org/pdf/2507.11004)  

**Abstract**: This paper presents HerO 2, Team HUMANE's system for the AVeriTeC shared task at the FEVER-25 workshop. HerO 2 is an enhanced version of HerO, the best-performing open-source model from the previous year's challenge. It improves evidence quality through document summarization and answer reformulation, optimizes veracity prediction via post-training quantization under computational constraints, and enhances overall system performance by integrating updated language model (LM) backbones. HerO 2 ranked second on the leaderboard while achieving the shortest runtime among the top three systems, demonstrating both high efficiency and strong potential for real-world fact verification. The code is available at this https URL. 

---
# DS@GT at eRisk 2025: From prompts to predictions, benchmarking early depression detection with conversational agent based assessments and temporal attention models 

**Authors**: Anthony Miyaguchi, David Guecha, Yuwen Chiu, Sidharth Gaur  

**Link**: [PDF](https://arxiv.org/pdf/2507.10958)  

**Abstract**: This Working Note summarizes the participation of the DS@GT team in two eRisk 2025 challenges. For the Pilot Task on conversational depression detection with large language-models (LLMs), we adopted a prompt-engineering strategy in which diverse LLMs conducted BDI-II-based assessments and produced structured JSON outputs. Because ground-truth labels were unavailable, we evaluated cross-model agreement and internal consistency. Our prompt design methodology aligned model outputs with BDI-II criteria and enabled the analysis of conversational cues that influenced the prediction of symptoms. Our best submission, second on the official leaderboard, achieved DCHR = 0.50, ADODL = 0.89, and ASHR = 0.27. 

---
# Transforming Sensitive Documents into Quantitative Data: An AI-Based Preprocessing Toolchain for Structured and Privacy-Conscious Analysis 

**Authors**: Anders Ledberg, Anna Thalén  

**Link**: [PDF](https://arxiv.org/pdf/2507.10582)  

**Abstract**: Unstructured text from legal, medical, and administrative sources offers a rich but underutilized resource for research in public health and the social sciences. However, large-scale analysis is hampered by two key challenges: the presence of sensitive, personally identifiable information, and significant heterogeneity in structure and language. We present a modular toolchain that prepares such text data for embedding-based analysis, relying entirely on open-weight models that run on local hardware, requiring only a workstation-level GPU and supporting privacy-sensitive research.
The toolchain employs large language model (LLM) prompting to standardize, summarize, and, when needed, translate texts to English for greater comparability. Anonymization is achieved via LLM-based redaction, supplemented with named entity recognition and rule-based methods to minimize the risk of disclosure. We demonstrate the toolchain on a corpus of 10,842 Swedish court decisions under the Care of Abusers Act (LVM), comprising over 56,000 pages. Each document is processed into an anonymized, standardized summary and transformed into a document-level embedding. Validation, including manual review, automated scanning, and predictive evaluation shows the toolchain effectively removes identifying information while retaining semantic content. As an illustrative application, we train a predictive model using embedding vectors derived from a small set of manually labeled summaries, demonstrating the toolchain's capacity for semi-automated content analysis at scale.
By enabling structured, privacy-conscious analysis of sensitive documents, our toolchain opens new possibilities for large-scale research in domains where textual data was previously inaccessible due to privacy and heterogeneity constraints. 

---
# Scalpel vs. Hammer: GRPO Amplifies Existing Capabilities, SFT Replaces Them 

**Authors**: Neel Rajani, Aryo Pradipta Gema, Seraphina Goldfarb-Tarrant, Ivan Titov  

**Link**: [PDF](https://arxiv.org/pdf/2507.10616)  

**Abstract**: Training large language models (LLMs) for reasoning via maths and code datasets has become a major new focus in LLM post-training. Two particularly popular approaches are reinforcement learning (RL) and supervised fine-tuning (SFT), but their training dynamics are poorly understood. We present a comparative analysis of RL and SFT on the same maths problems with the same model and similar hyperparameters. We find that RL yields minor in-domain gains on maths and slight degradation on knowledge-intensive benchmarks like MMLU, while both trends are more pronounced in SFT. We also analyse model parameters across checkpoints, observing that both algorithms modify query and key weights the most. Meanwhile, SFT exhibits greater updates and also affects mid-layer MLPs more, leading us to hypothesise that this may have caused the out-of-domain degradation. We therefore investigate whether freezing parts of the model during training can mitigate the reduced performance on knowledge-intensive benchmarks. However, our results are inconclusive, with benefits on GPQA:Diamond and degradation on other benchmarks. Taken together, our observations provide a preliminary indication for why RL amplifies existing capabilities, while SFT replaces old skills with new ones. 

---
# Sparse Autoencoders Can Capture Language-Specific Concepts Across Diverse Languages 

**Authors**: Lyzander Marciano Andrylie, Inaya Rahmanisa, Mahardika Krisna Ihsani, Alfan Farizki Wicaksono, Haryo Akbarianto Wibowo, Alham Fikri Aji  

**Link**: [PDF](https://arxiv.org/pdf/2507.11230)  

**Abstract**: Understanding the multilingual mechanisms of large language models (LLMs) provides insight into how they process different languages, yet this remains challenging. Existing studies often focus on individual neurons, but their polysemantic nature makes it difficult to isolate language-specific units from cross-lingual representations. To address this, we explore sparse autoencoders (SAEs) for their ability to learn monosemantic features that represent concrete and abstract concepts across languages in LLMs. While some of these features are language-independent, the presence of language-specific features remains underexplored. In this work, we introduce SAE-LAPE, a method based on feature activation probability, to identify language-specific features within the feed-forward network. We find that many such features predominantly appear in the middle to final layers of the model and are interpretable. These features influence the model's multilingual performance and language output and can be used for language identification with performance comparable to fastText along with more interpretability. Our code is available at this https URL . 

---
