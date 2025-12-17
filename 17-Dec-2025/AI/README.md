# Universal Reasoning Model 

**Authors**: Zitian Gao, Lynx Chen, Yihao Xiao, He Xing, Ran Tao, Haoming Luo, Joey Zhou, Bryan Dai  

**Link**: [PDF](https://arxiv.org/pdf/2512.14693)  

**Abstract**: Universal transformers (UTs) have been widely used for complex reasoning tasks such as ARC-AGI and Sudoku, yet the specific sources of their performance gains remain underexplored. In this work, we systematically analyze UTs variants and show that improvements on ARC-AGI primarily arise from the recurrent inductive bias and strong nonlinear components of Transformer, rather than from elaborate architectural designs. Motivated by this finding, we propose the Universal Reasoning Model (URM), which enhances the UT with short convolution and truncated backpropagation. Our approach substantially improves reasoning performance, achieving state-of-the-art 53.8% pass@1 on ARC-AGI 1 and 16.0% pass@1 on ARC-AGI 2. Our code is avaliable at this https URL. 

---
# Dynamic Learning Rate Scheduling based on Loss Changes Leads to Faster Convergence 

**Authors**: Shreyas Subramanian, Bala Krishnamoorthy, Pranav Murthy  

**Link**: [PDF](https://arxiv.org/pdf/2512.14527)  

**Abstract**: Despite significant advances in optimizers for training, most research works use common scheduler choices like Cosine or exponential decay. In this paper, we study \emph{GreedyLR}, a novel scheduler that adaptively adjusts the learning rate during training based on the current loss. To validate the effectiveness of our proposed scheduler, we conduct experiments on several NLP, CV, and LLM tasks with up to $7B$ parameters, including both fine-tuning and pre-training experiments. The results show that our approach outperforms several state-of-the-art schedulers in terms of accuracy, speed, and convergence. We also provide a theoretical analysis of the GreedyLR algorithm, including a proof of convergence and derivation of the optimal scaling factor $F$ that maximizes the convergence rate, along with experiments to show robustness of the algorithm to realistic noisy landscapes. Our scheduler is easy to implement, computationally efficient, and could be considered a good default scheduler for training. 

---
# Sparse Multi-Modal Transformer with Masking for Alzheimer's Disease Classification 

**Authors**: Cheng-Han Lu, Pei-Hsuan Tsai  

**Link**: [PDF](https://arxiv.org/pdf/2512.14491)  

**Abstract**: Transformer-based multi-modal intelligent systems often suffer from high computational and energy costs due to dense self-attention, limiting their scalability under resource constraints. This paper presents SMMT, a sparse multi-modal transformer architecture designed to improve efficiency and robustness. Building upon a cascaded multi-modal transformer framework, SMMT introduces cluster-based sparse attention to achieve near linear computational complexity and modality-wise masking to enhance robustness against incomplete inputs. The architecture is evaluated using Alzheimer's Disease classification on the ADNI dataset as a representative multi-modal case study. Experimental results show that SMMT maintains competitive predictive performance while significantly reducing training time, memory usage, and energy consumption compared to dense attention baselines, demonstrating its suitability as a resource-aware architectural component for scalable intelligent systems. 

---
# Model-First Reasoning LLM Agents: Reducing Hallucinations through Explicit Problem Modeling 

**Authors**: Annu Rana, Gaurav Kumar  

**Link**: [PDF](https://arxiv.org/pdf/2512.14474)  

**Abstract**: Large Language Models (LLMs) often struggle with complex multi-step planning tasks, showing high rates of constraint violations and inconsistent solutions. Existing strategies such as Chain-of-Thought and ReAct rely on implicit state tracking and lack an explicit problem representation. Inspired by classical AI planning, we propose Model-First Reasoning (MFR), a two-phase paradigm in which the LLM first constructs an explicit model of the problem, defining entities, state variables, actions, and constraints, before generating a solution plan. Across multiple planning domains, including medical scheduling, route planning, resource allocation, logic puzzles, and procedural synthesis, MFR reduces constraint violations and improves solution quality compared to Chain-of-Thought and ReAct. Ablation studies show that the explicit modeling phase is critical for these gains. Our results suggest that many LLM planning failures stem from representational deficiencies rather than reasoning limitations, highlighting explicit modeling as a key component for robust and interpretable AI agents. All prompts, evaluation procedures, and task datasets are documented to facilitate reproducibility. 

---
# Context-Picker: Dynamic context selection using multi-stage reinforcement learning 

**Authors**: Siyuan Zhu, Chengdong Xu, Kaiqiang Ke, Chao Yu  

**Link**: [PDF](https://arxiv.org/pdf/2512.14465)  

**Abstract**: In long-context question answering (LCQA), determining the optimal amount of context for a given query is a significant challenge. Including too few passages may omit critical information, while including too many can introduce noise and reduce the quality of the answer. Traditional approaches, such as fixed Top-$K$ retrieval and single-stage reranking, face the dilemma of selecting the right number of passages. This problem is particularly pronounced for factoid questions, which often require only a few specific pieces of evidence. To address this issue, we introduce \emph{Context-Picker}, a reasoning-aware framework that shifts the paradigm from similarity-based ranking to minimal sufficient subset selection. Context-Picker treats context selection as a decision-making process optimized via a human-inspired, two-stage reinforcement learning schedule: a \emph{recall-oriented} stage that prioritizes the coverage of reasoning chains, followed by a \emph{precision-oriented} stage that aggressively prunes redundancy to distill a compact evidence set. To resolve reward sparsity, we propose an offline evidence distillation pipeline that mines "minimal sufficient sets" via a Leave-One-Out (LOO) procedure, providing dense, task-aligned supervision. Experiments on five long-context and multi-hop QA benchmarks demonstrate that Context-Picker significantly outperforms strong RAG baselines, achieving superior answer accuracy with comparable or reduced context lengths. Ablation studies indicate that the coarse-to-fine optimization schedule, the redundancy-aware reward shaping, and the rationale-guided format all contribute substantially to these gains. 

---
# Seismology modeling agent: A smart assistant for geophysical researchers 

**Authors**: Yukun Ren, Siwei Yu, Kai Chen, Jianwei Ma  

**Link**: [PDF](https://arxiv.org/pdf/2512.14429)  

**Abstract**: To address the steep learning curve and reliance on complex manual file editing and command-line operations in the traditional workflow of the mainstream open-source seismic wave simulation software SPECFEM, this paper proposes an intelligent, interactive workflow powered by Large Language Models (LLMs). We introduce the first Model Context Protocol (MCP) server suite for SPECFEM (supporting 2D, 3D Cartesian, and 3D Globe versions), which decomposes the entire simulation process into discrete, agent-executable tools spanning from parameter generation and mesh partitioning to solver execution and visualization. This approach enables a paradigm shift from file-driven to intent-driven conversational interactions. The framework supports both fully automated execution and human-in-the-loop collaboration, allowing researchers to guide simulation strategies in real time and retain scientific decision-making authority while significantly reducing tedious low-level operations. Validated through multiple case studies, the workflow operates seamlessly in both autonomous and interactive modes, yielding high-fidelity results consistent with standard baselines. As the first application of MCP technology to computational seismology, this study significantly lowers the entry barrier, enhances reproducibility, and offers a promising avenue for advancing computational geophysics toward AI-assisted and automated scientific research. The complete source code is available at this https URL. 

---
# PortAgent: LLM-driven Vehicle Dispatching Agent for Port Terminals 

**Authors**: Jia Hu, Junqi Li, Weimeng Lin, Peng Jia, Yuxiong Ji, Jintao Lai  

**Link**: [PDF](https://arxiv.org/pdf/2512.14417)  

**Abstract**: Vehicle Dispatching Systems (VDSs) are critical to the operational efficiency of Automated Container Terminals (ACTs). However, their widespread commercialization is hindered due to their low transferability across diverse terminals. This transferability challenge stems from three limitations: high reliance on port operational specialists, a high demand for terminal-specific data, and time-consuming manual deployment processes. Leveraging the emergence of Large Language Models (LLMs), this paper proposes PortAgent, an LLM-driven vehicle dispatching agent that fully automates the VDS transferring workflow. It bears three features: (1) no need for port operations specialists; (2) low need of data; and (3) fast deployment. Specifically, specialist dependency is eliminated by the Virtual Expert Team (VET). The VET collaborates with four virtual experts, including a Knowledge Retriever, Modeler, Coder, and Debugger, to emulate a human expert team for the VDS transferring workflow. These experts specialize in the domain of terminal VDS via a few-shot example learning approach. Through this approach, the experts are able to learn VDS-domain knowledge from a few VDS examples. These examples are retrieved via a Retrieval-Augmented Generation (RAG) mechanism, mitigating the high demand for terminal-specific data. Furthermore, an automatic VDS design workflow is established among these experts to avoid extra manual interventions. In this workflow, a self-correction loop inspired by the LLM Reflexion framework is created 

---
# Massive Editing for Large Language Models Based on Dynamic Weight Generation 

**Authors**: Wentao Wan, Qiqing Lao, Zhiwei Xie, Hefeng Wu, Runnan Lin, Liang Lin, Keze Wang  

**Link**: [PDF](https://arxiv.org/pdf/2512.14395)  

**Abstract**: Knowledge Editing (KE) is a field that studies how to modify some knowledge in Large Language Models (LLMs) at a low cost (compared to pre-training). Currently, performing large-scale edits on LLMs while ensuring the Reliability, Generality, and Locality metrics of the edits remain a challenge. This paper proposes a Massive editing approach for LLMs based on dynamic weight Generation (MeG). Our MeG involves attaching a dynamic weight neuron to specific layers of the LLMs and using a diffusion model to conditionally generate the weights of this neuron based on the input query required for the knowledge. This allows the use of adding a single dynamic weight neuron to achieve the goal of large-scale knowledge editing. Experiments show that our MeG can significantly improve the performance of large-scale KE in terms of Reliability, Generality, and Locality metrics compared to existing knowledge editing methods, particularly with a high percentage point increase in the absolute value index for the Locality metric, demonstrating the advantages of our proposed method. 

---
# TiCard: Deployable EXPLAIN-only Residual Learning for Cardinality Estimation 

**Authors**: Qizhi Wang  

**Link**: [PDF](https://arxiv.org/pdf/2512.14358)  

**Abstract**: Cardinality estimation is a key bottleneck for cost-based query optimization, yet deployable improvements remain difficult: classical estimators miss correlations, while learned estimators often require workload-specific training pipelines and invasive integration into the optimizer. This paper presents TiCard, a low intrusion, correction-based framework that augments (rather than replaces) a database's native estimator. TiCard learns multiplicative residual corrections using EXPLAIN-only features, and uses EXPLAIN ANALYZE only for offline labels. We study two practical instantiations: (i) a Gradient Boosting Regressor for sub-millisecond inference, and (ii) TabPFN, an in-context tabular foundation model that adapts by refreshing a small reference set without gradient retraining. On TiDB with TPCH and the Join Order Benchmark, in a low-trace setting (263 executions total; 157 used for learning), TiCard improves operator-level tail accuracy substantially: P90 Q-error drops from 312.85 (native) to 13.69 (TiCard-GBR), and P99 drops from 37,974.37 to 3,416.50 (TiCard-TabPFN), while a join-only policy preserves near-perfect median behavior. We position TiCard as an AI4DB building block focused on deployability: explicit scope, conservative integration policies, and an integration roadmap from offline correction to in-optimizer use. 

---
# Leveraging LLMs for Collaborative Ontology Engineering in Parkinson Disease Monitoring and Alerting 

**Authors**: Georgios Bouchouras, Dimitrios Doumanas, Andreas Soularidis, Konstantinos Kotis, George A. Vouros  

**Link**: [PDF](https://arxiv.org/pdf/2512.14288)  

**Abstract**: This paper explores the integration of Large Language Models (LLMs) in the engineering of a Parkinson's Disease (PD) monitoring and alerting ontology through four key methodologies: One Shot (OS) prompt techniques, Chain of Thought (CoT) prompts, X-HCOME, and SimX-HCOME+. The primary objective is to determine whether LLMs alone can create comprehensive ontologies and, if not, whether human-LLM collaboration can achieve this goal. Consequently, the paper assesses the effectiveness of LLMs in automated ontology development and the enhancement achieved through human-LLM collaboration.
Initial ontology generation was performed using One Shot (OS) and Chain of Thought (CoT) prompts, demonstrating the capability of LLMs to autonomously construct ontologies for PD monitoring and alerting. However, these outputs were not comprehensive and required substantial human refinement to enhance their completeness and accuracy.
X-HCOME, a hybrid ontology engineering approach that combines human expertise with LLM capabilities, showed significant improvements in ontology comprehensiveness. This methodology resulted in ontologies that are very similar to those constructed by experts.
Further experimentation with SimX-HCOME+, another hybrid methodology emphasizing continuous human supervision and iterative refinement, highlighted the importance of ongoing human involvement. This approach led to the creation of more comprehensive and accurate ontologies.
Overall, the paper underscores the potential of human-LLM collaboration in advancing ontology engineering, particularly in complex domains like PD. The results suggest promising directions for future research, including the development of specialized GPT models for ontology construction. 

---
# Gödel's Poetry 

**Authors**: Kelly J. Davis  

**Link**: [PDF](https://arxiv.org/pdf/2512.14252)  

**Abstract**: Formal, automated theorem proving has long been viewed as a challenge to artificial intelligence. We introduce here a new approach to computer theorem proving, one that employs specialized language models for Lean4 proof generation combined with recursive decomposition of difficult theorems into simpler entailing propositions. These models are coordinated through a multi-agent architecture that orchestrates autoformalization (if required), proof generation, decomposition of difficult theorems into simpler entailing propositions, and recursive proof (and/or decomposition) of these propositions. Without decomposition, we achieve a 90.4% pass rate on miniF2F. With decomposition, this is significantly improved. A key technical contribution lies in our extension of the Kimina Lean Server with abstract syntax tree (AST) parsing capabilities to facilitate automated, recursive proof decomposition. The system is made available on PyPI as goedels-poetry (at this https URL ), and the open-source implementation KellyJDavis/goedels-poetry (at this https URL ) facilitates both adaptation to alternative language models and extension with custom functionality. 

---
# Georeferencing complex relative locality descriptions with large language models 

**Authors**: Aneesha Fernando, Surangika Ranathunga, Kristin Stock, Raj Prasanna, Christopher B. Jones  

**Link**: [PDF](https://arxiv.org/pdf/2512.14228)  

**Abstract**: Georeferencing text documents has typically relied on either gazetteer-based methods to assign geographic coordinates to place names, or on language modelling approaches that associate textual terms with geographic locations. However, many location descriptions specify positions relatively with spatial relationships, making geocoding based solely on place names or geo-indicative words inaccurate. This issue frequently arises in biological specimen collection records, where locations are often described through narratives rather than coordinates if they pre-date GPS. Accurate georeferencing is vital for biodiversity studies, yet the process remains labour-intensive, leading to a demand for automated georeferencing solutions. This paper explores the potential of Large Language Models (LLMs) to georeference complex locality descriptions automatically, focusing on the biodiversity collections domain. We first identified effective prompting patterns, then fine-tuned an LLM using Quantized Low-Rank Adaptation (QLoRA) on biodiversity datasets from multiple regions and languages. Our approach outperforms existing baselines with an average, across datasets, of 65% of records within a 10 km radius, for a fixed amount of training data. The best results (New York state) were 85% within 10km and 67% within 1km. The selected LLM performs well for lengthy, complex descriptions, highlighting its potential for georeferencing intricate locality descriptions. 

---
# Incentivizing Tool-augmented Thinking with Images for Medical Image Analysis 

**Authors**: Yankai Jiang, Yujie Zhang, Peng Zhang, Yichen Li, Jintai Chen, Xiaoming Shi, Shihui Zhen  

**Link**: [PDF](https://arxiv.org/pdf/2512.14157)  

**Abstract**: Recent reasoning based medical MLLMs have made progress in generating step by step textual reasoning chains. However, they still struggle with complex tasks that necessitate dynamic and iterative focusing on fine-grained visual regions to achieve precise grounding and diagnosis. We introduce Ophiuchus, a versatile, tool-augmented framework that equips an MLLM to (i) decide when additional visual evidence is needed, (ii) determine where to probe and ground within the medical image, and (iii) seamlessly weave the relevant sub-image content back into an interleaved, multimodal chain of thought. In contrast to prior approaches limited by the performance ceiling of specialized tools, Ophiuchus integrates the model's inherent grounding and perception capabilities with external tools, thereby fostering higher-level reasoning. The core of our method is a three-stage training strategy: cold-start training with tool-integrated reasoning data to achieve basic tool selection and adaptation for inspecting key regions; self-reflection fine-tuning to strengthen reflective reasoning and encourage revisiting tool outputs; and Agentic Tool Reinforcement Learning to directly optimize task-specific rewards and emulate expert-like diagnostic behavior. Extensive experiments show that Ophiuchus consistently outperforms both closed-source and open-source SOTA methods across diverse medical benchmarks, including VQA, detection, and reasoning-based segmentation. Our approach illuminates a path toward medical AI agents that can genuinely "think with images" through tool-integrated reasoning. Datasets, codes, and trained models will be released publicly. 

---
# Optimizing Multi-Tier Supply Chain Ordering with a Hybrid Liquid Neural Network and Extreme Gradient Boosting Model 

**Authors**: Chunan Tong  

**Link**: [PDF](https://arxiv.org/pdf/2512.14112)  

**Abstract**: Supply chain management (SCM) faces significant challenges like demand fluctuations and the bullwhip effect. Traditional methods and even state-of-the-art LLMs struggle with benchmarks like the Vending Machine Test, failing to handle SCM's complex continuous time-series data. While ML approaches like LSTM and XGBoost offer solutions, they are often limited by computational inefficiency. Liquid Neural Networks (LNN), known for their adaptability and efficiency in robotics, remain untapped in SCM. This study proposes a hybrid LNN+XGBoost model for multi-tier supply chains. By combining LNN's dynamic feature extraction with XGBoost's global optimization, the model aims to minimize the bullwhip effect and increase profitability. This innovative approach addresses the need for efficiency and adaptability, filling a critical gap in intelligent SCM. 

---
# HydroGEM: A Self Supervised Zero Shot Hybrid TCN Transformer Foundation Model for Continental Scale Streamflow Quality Control 

**Authors**: Ijaz Ul Haq, Byung Suk Lee, Julia N. Perdrial, David Baude  

**Link**: [PDF](https://arxiv.org/pdf/2512.14106)  

**Abstract**: Real-time streamflow monitoring networks generate millions of observations annually, yet maintaining data quality across thousands of remote sensors remains labor-intensive. We introduce HydroGEM (Hydrological Generalizable Encoder for Monitoring), a foundation model for continental-scale streamflow quality control. HydroGEM uses two-stage training: self-supervised pretraining on 6.03 million sequences from 3,724 USGS stations learns hydrological representations, followed by fine-tuning with synthetic anomalies for detection and reconstruction. A hybrid TCN-Transformer architecture (14.2M parameters) captures local temporal patterns and long-range dependencies, while hierarchical normalization handles six orders of magnitude in discharge. On held-out synthetic tests comprising 799 stations with 18 expert-validated anomaly types, HydroGEM achieves F1 = 0.792 for detection and 68.7% reconstruction-error reduction, a 36.3% improvement over existing methods. Zero-shot transfer to 100 Environment and Climate Change Canada stations yields F1 = 0.586, exceeding all baselines and demonstrating cross-national generalization. The model maintains consistent detection across correction magnitudes and aligns with operational seasonal patterns. HydroGEM is designed for human-in-the-loop workflows - outputs are quality control suggestions requiring expert review, not autonomous corrections. 

---
# Grammar Search for Multi-Agent Systems 

**Authors**: Mayank Singh, Vikas Yadav, Shiva Krishna Reddy Malay, Shravan Nayak, Sai Rajeswar, Sathwik Tejaswi Madhusudhan, Eduardo Blanco  

**Link**: [PDF](https://arxiv.org/pdf/2512.14079)  

**Abstract**: Automatic search for Multi-Agent Systems has recently emerged as a key focus in agentic AI research. Several prior approaches have relied on LLM-based free-form search over the code space. In this work, we propose a more structured framework that explores the same space through a fixed set of simple, composable components. We show that, despite lacking the generative flexibility of LLMs during the candidate generation stage, our method outperforms prior approaches on four out of five benchmarks across two domains: mathematics and question answering. Furthermore, our method offers additional advantages, including a more cost-efficient search process and the generation of modular, interpretable multi-agent systems with simpler logic. 

---
# RADAR: Accelerating Large Language Model Inference With RL-Based Dynamic Draft Trees 

**Authors**: Junjie Ma, Jinlong Li  

**Link**: [PDF](https://arxiv.org/pdf/2512.14069)  

**Abstract**: Inference with modern Large Language Models (LLMs) is expensive and slow, and speculative sampling has emerged as an effective solution to this problem, however, the number of the calls to the draft model for generating candidate tokens in speculative sampling is a preset hyperparameter, lacking flexibility. To generate and utilize the candidate tokens more effectively, we propose RADAR, a novel speculative sampling method with RL-based dynamic draft trees. RADAR formulates the draft tree generation process as a Markov Decision Process (MDP) and employs offline reinforcement learning to train a prediction model, which enables real-time decision on the calls to the draft model, reducing redundant computations and further accelerating inference. Evaluations across three LLMs and four tasks show that RADAR achieves a speedup of 3.17x-4.82x over the auto-regressive decoding baseline. The code is available at this https URL. 

---
# OpenDataArena: A Fair and Open Arena for Benchmarking Post-Training Dataset Value 

**Authors**: Mengzhang Cai, Xin Gao, Yu Li, Honglin Lin, Zheng Liu, Zhuoshi Pan, Qizhi Pei, Xiaoran Shang, Mengyuan Sun, Zinan Tang, Xiaoyang Wang, Zhanping Zhong, Yun Zhu, Dahua Lin, Conghui He, Lijun Wu  

**Link**: [PDF](https://arxiv.org/pdf/2512.14051)  

**Abstract**: The rapid evolution of Large Language Models (LLMs) is predicated on the quality and diversity of post-training datasets. However, a critical dichotomy persists: while models are rigorously benchmarked, the data fueling them remains a black box--characterized by opaque composition, uncertain provenance, and a lack of systematic evaluation. This opacity hinders reproducibility and obscures the causal link between data characteristics and model behaviors. To bridge this gap, we introduce OpenDataArena (ODA), a holistic and open platform designed to benchmark the intrinsic value of post-training data. ODA establishes a comprehensive ecosystem comprising four key pillars: (i) a unified training-evaluation pipeline that ensures fair, open comparisons across diverse models (e.g., Llama, Qwen) and domains; (ii) a multi-dimensional scoring framework that profiles data quality along tens of distinct axes; (iii) an interactive data lineage explorer to visualize dataset genealogy and dissect component sources; and (iv) a fully open-source toolkit for training, evaluation, and scoring to foster data research. Extensive experiments on ODA--covering over 120 training datasets across multiple domains on 22 benchmarks, validated by more than 600 training runs and 40 million processed data points--reveal non-trivial insights. Our analysis uncovers the inherent trade-offs between data complexity and task performance, identifies redundancy in popular benchmarks through lineage tracing, and maps the genealogical relationships across datasets. We release all results, tools, and configurations to democratize access to high-quality data evaluation. Rather than merely expanding a leaderboard, ODA envisions a shift from trial-and-error data curation to a principled science of Data-Centric AI, paving the way for rigorous studies on data mixing laws and the strategic composition of foundation models. 

---
# Intention Chain-of-Thought Prompting with Dynamic Routing for Code Generation 

**Authors**: Shen Li, Li Huang, Shaoxiong Zhan, Weifeng Sun, Tao Yin, Zhongxin Liu, Meng Yan  

**Link**: [PDF](https://arxiv.org/pdf/2512.14048)  

**Abstract**: Large language models (LLMs) exhibit strong generative capabilities and have shown great potential in code generation. Existing chain-of-thought (CoT) prompting methods enhance model reasoning by eliciting intermediate steps, but suffer from two major limitations: First, their uniform application tends to induce overthinking on simple tasks. Second, they lack intention abstraction in code generation, such as explicitly modeling core algorithmic design and efficiency, leading models to focus on surface-level structures while neglecting the global problem objective. Inspired by the cognitive economy principle of engaging structured reasoning only when necessary to conserve cognitive resources, we propose RoutingGen, a novel difficulty-aware routing framework that dynamically adapts prompting strategies for code generation. For simple tasks, it adopts few-shot prompting; for more complex ones, it invokes a structured reasoning strategy, termed Intention Chain-of-Thought (ICoT), which we introduce to guide the model in capturing task intention, such as the core algorithmic logic and its time complexity. Experiments across three models and six standard code generation benchmarks show that RoutingGen achieves state-of-the-art performance in most settings, while reducing total token usage by 46.37% on average across settings. Furthermore, ICoT outperforms six existing prompting baselines on challenging benchmarks. 

---
# Evaluating Small Language Models for Agentic On-Farm Decision Support Systems 

**Authors**: Enhong Liu, Haiyu Yang, Miel Hostens  

**Link**: [PDF](https://arxiv.org/pdf/2512.14043)  

**Abstract**: Large Language Models (LLM) hold potential to support dairy scholars and farmers by supporting decision-making and broadening access to knowledge for stakeholders with limited technical expertise. However, the substantial computational demand restricts access to LLM almost exclusively through cloud-based service, which makes LLM-based decision support tools impractical for dairy farming. To address this gap, lightweight alternatives capable of running locally on farm hardware are required. In this work, we benchmarked 20 open-source Small Language Models (SLM) available on HuggingFace under farm-realistic computing constraints. Building on our prior work, we developed an agentic AI system that integrates five task-specific agents: literature search, web search, SQL database interaction, NoSQL database interaction, and graph generation following predictive models. Evaluation was conducted in two phases. In the first phase, five test questions were used for the initial screening to identify models capable of following basic dairy-related instructions and performing reliably in a compute-constrained environment. Models that passed this preliminary stage were then evaluated using 30 questions (five per task category mentioned above, plus one category addressing integrity and misconduct) in phase two. In results, Qwen-4B achieved superior performance across most of task categories, although showed unstable effectiveness in NoSQL database interactions through PySpark. To our knowledge, this is the first work explicitly evaluating the feasibility of SLM as engines for dairy farming decision-making, with central emphases on privacy and computational efficiency. While results highlight the promise of SLM-assisted tools for practical deployment in dairy farming, challenges remain, and fine-tuning is still needed to refine SLM performance in dairy-specific questions. 

---
# MobileWorldBench: Towards Semantic World Modeling For Mobile Agents 

**Authors**: Shufan Li, Konstantinos Kallidromitis, Akash Gokul, Yusuke Kato, Kazuki Kozuka, Aditya Grover  

**Link**: [PDF](https://arxiv.org/pdf/2512.14014)  

**Abstract**: World models have shown great utility in improving the task performance of embodied agents. While prior work largely focuses on pixel-space world models, these approaches face practical limitations in GUI settings, where predicting complex visual elements in future states is often difficult. In this work, we explore an alternative formulation of world modeling for GUI agents, where state transitions are described in natural language rather than predicting raw pixels. First, we introduce MobileWorldBench, a benchmark that evaluates the ability of vision-language models (VLMs) to function as world models for mobile GUI agents. Second, we release MobileWorld, a large-scale dataset consisting of 1.4M samples, that significantly improves the world modeling capabilities of VLMs. Finally, we propose a novel framework that integrates VLM world models into the planning framework of mobile agents, demonstrating that semantic world models can directly benefit mobile agents by improving task success rates. The code and dataset is available at this https URL 

---
# Sparsity-Controllable Dynamic Top-p MoE for Large Foundation Model Pre-training 

**Authors**: Can Jin, Hongwu Peng, Mingcan Xiang, Qixin Zhang, Xiangchi Yuan, Amit Hasan, Ohiremen Dibua, Yifan Gong, Yan Kang, Dimitris N. Metaxas  

**Link**: [PDF](https://arxiv.org/pdf/2512.13996)  

**Abstract**: Sparse Mixture-of-Experts (MoE) architectures effectively scale model capacity by activating only a subset of experts for each input token. However, the standard Top-k routing strategy imposes a uniform sparsity pattern that ignores the varying difficulty of tokens. While Top-p routing offers a flexible alternative, existing implementations typically rely on a fixed global probability threshold, which results in uncontrolled computational costs and sensitivity to hyperparameter selection. In this paper, we propose DTop-p MoE, a sparsity-controllable dynamic Top-p routing mechanism. To resolve the challenge of optimizing a non-differentiable threshold, we utilize a Proportional-Integral (PI) Controller that dynamically adjusts the probability threshold to align the running activated-expert sparsity with a specified target. Furthermore, we introduce a dynamic routing normalization mechanism that adapts layer-wise routing logits, allowing different layers to learn distinct expert-selection patterns while utilizing a global probability threshold. Extensive experiments on Large Language Models and Diffusion Transformers demonstrate that DTop-p consistently outperforms both Top-k and fixed-threshold Top-p baselines. Our analysis confirms that DTop-p maintains precise control over the number of activated experts while adaptively allocating resources across different tokens and layers. Furthermore, DTop-p exhibits strong scaling properties with respect to expert granularity, expert capacity, model size, and dataset size, offering a robust framework for large-scale MoE pre-training. 

---
# ReflCtrl: Controlling LLM Reflection via Representation Engineering 

**Authors**: Ge Yan, Chung-En Sun, Tsui-Wei, Weng  

**Link**: [PDF](https://arxiv.org/pdf/2512.13979)  

**Abstract**: Large language models (LLMs) with Chain-of-Thought (CoT) reasoning have achieved strong performance across diverse tasks, including mathematics, coding, and general reasoning. A distinctive ability of these reasoning models is self-reflection: the ability to review and revise previous reasoning steps. While self-reflection enhances reasoning performance, it also increases inference cost. In this work, we study self-reflection through the lens of representation engineering. We segment the model's reasoning into steps, identify the steps corresponding to reflection, and extract a reflection direction in the latent space that governs this behavior. Using this direction, we propose a stepwise steering method that can control reflection frequency. We call our framework ReflCtrl. Our experiments show that (1) in many cases reflections are redundant, especially in stronger models (in our experiments, we can save up to 33.6 percent of reasoning tokens while preserving performance), and (2) the model's reflection behavior is highly correlated with an internal uncertainty signal, implying self-reflection may be controlled by the model's uncertainty. 

---
# Evaluating Frontier LLMs on PhD-Level Mathematical Reasoning: A Benchmark on a Textbook in Theoretical Computer Science about Randomized Algorithms 

**Authors**: Yang Cao, Yubin Chen, Xuyang Guo, Zhao Song, Song Yue, Jiahao Zhang, Jiale Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2512.13978)  

**Abstract**: The rapid advancement of large language models (LLMs) has led to significant breakthroughs in automated mathematical reasoning and scientific discovery. Georgiev, G${ó}$mez-Serrano, Tao, and Wagner [GGSTW+25] demonstrate that AI systems can explore new constructions and improve existing bounds, illustrating the growing potential of LLMs to accelerate mathematical discovery. Similarly, Bubeck et al. [BCE+25] show that GPT-5 can meaningfully contribute to scientific workflows, from proposing hypotheses to generating proofs and analyses. Despite these advances, a rigorous evaluation of these models on canonical, graduate-level mathematical theory remains necessary to understand their baseline reasoning capabilities. In this paper, we present a comprehensive benchmark of four frontier models: GPT-5-Thinking, Gemini-3-Pro, Claude-Sonnet-4.5-Thinking, and Grok-4 against the classic curriculum of Randomized Algorithms by Motwani and Raghavan [MR95].
We tasked each model with generating formal LaTeX proofs for a series of lemmas and exercises spanning the textbook. We find that while the top-tier models (Gemini, and Claude) achieve a high accuracy rate (approx. 66%), demonstrating a robust grasp of probabilistic method and formal logic, other models lag significantly in consistency (approx. 40%). We provide a qualitative analysis of the generated proofs, highlighting differences in conciseness, hallucination rates, and logical structure. Our results suggest that while frontier models have reached a threshold of proficiency suitable for graduate-level pedagogical assistance and formalization, significant variance exists in their reliability for rigorous mathematical derivation. The code and the full set of LLM-generated responses are open-sourced and publicly available at this https URL. 

---
# MURIM: Multidimensional Reputation-based Incentive Mechanism for Federated Learning 

**Authors**: Sindhuja Madabushi, Dawood Wasif, Jin-Hee Cho  

**Link**: [PDF](https://arxiv.org/pdf/2512.13955)  

**Abstract**: Federated Learning (FL) has emerged as a leading privacy-preserving machine learning paradigm, enabling participants to share model updates instead of raw data. However, FL continues to face key challenges, including weak client incentives, privacy risks, and resource constraints. Assessing client reliability is essential for fair incentive allocation and ensuring that each client's data contributes meaningfully to the global model. To this end, we propose MURIM, a MUlti-dimensional Reputation-based Incentive Mechanism that jointly considers client reliability, privacy, resource capacity, and fairness while preventing malicious or unreliable clients from earning undeserved rewards. MURIM allocates incentives based on client contribution, latency, and reputation, supported by a reliability verification module. Extensive experiments on MNIST, FMNIST, and ADULT Income datasets demonstrate that MURIM achieves up to 18% improvement in fairness metrics, reduces privacy attack success rates by 5-9%, and improves robustness against poisoning and noisy-gradient attacks by up to 85% compared to state-of-the-art baselines. Overall, MURIM effectively mitigates adversarial threats, promotes fair and truthful participation, and preserves stable model convergence across heterogeneous and dynamic federated settings. 

---
# EvoLattice: Persistent Internal-Population Evolution through Multi-Alternative Quality-Diversity Graph Representations for LLM-Guided Program Discovery 

**Authors**: Kamer Ali Yuksel  

**Link**: [PDF](https://arxiv.org/pdf/2512.13857)  

**Abstract**: Large language models (LLMs) are increasingly used to evolve programs and multi-agent systems, yet most existing approaches rely on overwrite-based mutations that maintain only a single candidate at a time. Such methods discard useful variants, suffer from destructive edits, and explore a brittle search space prone to structural failure. We introduce EvoLattice, a framework that represents an entire population of candidate programs or agent behaviors within a single directed acyclic graph. Each node stores multiple persistent alternatives, and every valid path through the graph defines a distinct executable candidate, yielding a large combinatorial search space without duplicating structure. EvoLattice enables fine-grained alternative-level evaluation by scoring each alternative across all paths in which it appears, producing statistics that reveal how local design choices affect global performance. These statistics provide a dense, data-driven feedback signal for LLM-guided mutation, recombination, and pruning, while preserving successful components. Structural correctness is guaranteed by a deterministic self-repair mechanism that enforces acyclicity and dependency consistency independently of the LLM. EvoLattice naturally extends to agent evolution by interpreting alternatives as prompt fragments or sub-agent behaviors. Across program synthesis (proxy and optimizer meta-learning), EvoLattice yields more stable evolution, greater expressivity, and stronger improvement trajectories than prior LLM-guided methods. The resulting dynamics resemble quality-diversity optimization, emerging implicitly from EvoLattice's internal multi-alternative representation rather than an explicit external archive. 

---
# Semantic Grounding Index: Geometric Bounds on Context Engagement in RAG Systems 

**Authors**: Javier Marín  

**Link**: [PDF](https://arxiv.org/pdf/2512.13771)  

**Abstract**: When retrieval-augmented generation (RAG) systems hallucinate, what geometric trace does this leave in embedding space? We introduce the Semantic Grounding Index (SGI), defined as the ratio of angular distances from the response to the question versus the context on the unit hypersphere $\mathbb{S}^{d-1}$.Our central finding is \emph{semantic laziness}: hallucinated responses remain angularly proximate to questions rather than departing toward retrieved contexts. On HaluEval ($n$=5,000), we observe large effect sizes (Cohen's $d$ ranging from 0.92 to 1.28) across five embedding models with mean cross-model correlation $r$=0.85. Crucially, we derive from the spherical triangle inequality that SGI's discriminative power should increase with question-context angular separation $\theta(q,c)$-a theoretical prediction confirmed empirically: effect size rises monotonically from $d$=0.61 -low $\theta(q,c)$, to $d$=1.27 -high $\theta(q,c)$, with AUC improving from 0.72 to 0.83. Subgroup analysis reveals that SGI excels on long responses ($d$=2.05) and short questions ($d$=1.22), while remaining robust across context lengths. Calibration analysis yields ECE=0.10, indicating SGI scores can serve as probability estimates, not merely rankings. A critical negative result on TruthfulQA (AUC=0.478) establishes that angular geometry measures topical engagement rather than factual accuracy. SGI provides computationally efficient, theoretically grounded infrastructure for identifying responses that warrant verification in production RAG deployments. 

---
# Mathematics and Coding are Universal AI Benchmarks 

**Authors**: Przemyslaw Chojecki  

**Link**: [PDF](https://arxiv.org/pdf/2512.13764)  

**Abstract**: We study the special role of mathematics and coding inside the moduli space of psychometric batteries for AI agents. Building on the AAI framework and GVU dynamics from previous works, we define the Mathematics Fiber and show that, when paired with formal proof kernels (e.g. Lean, Coq), GVU flows on this fiber admit spectrally stable self-improvement regimes due to oracle-like verification. Our main technical result is a density theorem: under uniform tightness of agent outputs and a Lipschitz AAI functional, the subspace of batteries generated by mathematical theorem-proving and coding tasks is dense in the moduli space of batteries with respect to the evaluation metric. Coding alone is universal in this sense, while pure mathematics is not; its privilege is spectral rather than expressive. We interpret this as evidence that mathematics and coding provide ``universal coordinates'' for evaluation, and that formal mathematics is a natural ignition domain for recursive self-improvement in advanced AI agents. 

---
# State-Dependent Refusal and Learned Incapacity in RLHF-Aligned Language Models 

**Authors**: TK Lee  

**Link**: [PDF](https://arxiv.org/pdf/2512.13762)  

**Abstract**: Large language models (LLMs) are widely deployed as general-purpose tools, yet extended interaction can reveal behavioral patterns not captured by standard quantitative benchmarks. We present a qualitative case-study methodology for auditing policy-linked behavioral selectivity in long-horizon interaction. In a single 86-turn dialogue session, the same model shows Normal Performance (NP) in broad, non-sensitive domains while repeatedly producing Functional Refusal (FR) in provider- or policy-sensitive domains, yielding a consistent asymmetry between NP and FR across domains. Drawing on learned helplessness as an analogy, we introduce learned incapacity (LI) as a behavioral descriptor for this selective withholding without implying intentionality or internal mechanisms. We operationalize three response regimes (NP, FR, Meta-Narrative; MN) and show that MN role-framing narratives tend to co-occur with refusals in the same sensitive contexts. Overall, the study proposes an interaction-level auditing framework based on observable behavior and motivates LI as a lens for examining potential alignment side effects, warranting further investigation across users and models. 

---
# Compressed Causal Reasoning: Quantization and GraphRAG Effects on Interventional and Counterfactual Accuracy 

**Authors**: Steve Nwaiwu, Nipat Jongsawat, Anucha Tungkasthan  

**Link**: [PDF](https://arxiv.org/pdf/2512.13725)  

**Abstract**: Causal reasoning in Large Language Models spanning association, intervention, and counterfactual inference is essential for reliable decision making in high stakes settings. As deployment shifts toward edge and resource constrained environments, quantized models such as INT8 and NF4 are becoming standard. Yet the impact of precision reduction on formal causal reasoning is poorly understood. To our knowledge, this is the first study to systematically evaluate quantization effects across all three levels of Pearls Causal Ladder. Using a 3000 sample stratified CLadder benchmark, we find that rung level accuracy in Llama 3 8B remains broadly stable under quantization, with NF4 showing less than one percent overall degradation. Interventional queries at rung 2 are the most sensitive to precision loss, whereas counterfactual reasoning at rung 3 is comparatively stable but exhibits heterogeneous weaknesses across query types such as collider bias and backdoor adjustment. Experiments on the CRASS benchmark show near identical performance across precisions, indicating that existing commonsense counterfactual datasets lack the structural sensitivity needed to reveal quantization induced reasoning drift. We further evaluate Graph Retrieval Augmented Generation using ground truth causal graphs and observe a consistent improvement in NF4 interventional accuracy of plus 1.7 percent, partially offsetting compression related degradation. These results suggest that causal reasoning is unexpectedly robust to four bit quantization, graph structured augmentation can selectively reinforce interventional reasoning, and current counterfactual benchmarks fail to capture deeper causal brittleness. This work provides an initial empirical map of compressed causal reasoning and practical guidance for deploying efficient and structurally supported causal AI systems. 

---
# ValuePilot: A Two-Phase Framework for Value-Driven Decision-Making 

**Authors**: Yitong Luo, Ziang Chen, Hou Hei Lam, Jiayu zhan, Junqi Wang, Zhenliang Zhang, Xue Feng  

**Link**: [PDF](https://arxiv.org/pdf/2512.13716)  

**Abstract**: Personalized decision-making is essential for human-AI interaction, enabling AI agents to act in alignment with individual users' value preferences. As AI systems expand into real-world applications, adapting to personalized values beyond task completion or collective alignment has become a critical challenge. We address this by proposing a value-driven approach to personalized decision-making. Human values serve as stable, transferable signals that support consistent and generalizable behavior across contexts. Compared to task-oriented paradigms driven by external rewards and incentives, value-driven decision-making enhances interpretability and enables agents to act appropriately even in novel scenarios. We introduce ValuePilot, a two-phase framework consisting of a dataset generation toolkit (DGT) and a decision-making module (DMM). DGT constructs diverse, value-annotated scenarios from a human-LLM collaborative pipeline. DMM learns to evaluate actions based on personal value preferences, enabling context-sensitive, individualized decisions. When evaluated on previously unseen scenarios, DMM outperforms strong LLM baselines, including GPT-5, Claude-Sonnet-4, Gemini-2-flash, and Llama-3.1-70b, in aligning with human action choices. Our results demonstrate that value-driven decision-making is an effective and extensible engineering pathway toward building interpretable, personalized AI agents. 

---
# Meta Hierarchical Reinforcement Learning for Scalable Resource Management in O-RAN 

**Authors**: Fatemeh Lotfi, Fatemeh Afghah  

**Link**: [PDF](https://arxiv.org/pdf/2512.13715)  

**Abstract**: The increasing complexity of modern applications demands wireless networks capable of real time adaptability and efficient resource management. The Open Radio Access Network (O-RAN) architecture, with its RAN Intelligent Controller (RIC) modules, has emerged as a pivotal solution for dynamic resource management and network slicing. While artificial intelligence (AI) driven methods have shown promise, most approaches struggle to maintain performance under unpredictable and highly dynamic conditions. This paper proposes an adaptive Meta Hierarchical Reinforcement Learning (Meta-HRL) framework, inspired by Model Agnostic Meta Learning (MAML), to jointly optimize resource allocation and network slicing in O-RAN. The framework integrates hierarchical control with meta learning to enable both global and local adaptation: the high-level controller allocates resources across slices, while low level agents perform intra slice scheduling. The adaptive meta-update mechanism weights tasks by temporal difference error variance, improving stability and prioritizing complex network scenarios. Theoretical analysis establishes sublinear convergence and regret guarantees for the two-level learning process. Simulation results demonstrate a 19.8% improvement in network management efficiency compared with baseline RL and meta-RL approaches, along with faster adaptation and higher QoS satisfaction across eMBB, URLLC, and mMTC slices. Additional ablation and scalability studies confirm the method's robustness, achieving up to 40% faster adaptation and consistent fairness, latency, and throughput performance as network scale increases. 

---
# AI-Powered Annotation Pipelines for Stabilizing Large Language Models: A Human-AI Synergy Approach 

**Authors**: Gangesh Pathak, Prasanna Kumar  

**Link**: [PDF](https://arxiv.org/pdf/2512.13714)  

**Abstract**: LLM implementations are failing in highly regulated industries owing to instability issues, inconsistent reasoning, hallucinations and performance variability, especially in workflows. These reliability issues restrict safe use of LLM in areas that need the precision of facts and consistent behavior (Aiyappa et al., 2023). The current methods of stabilization, such as, reinforcement learning with human feedback (RLHF) and supervised fine-tuning, offer quantifiable improvements but are expensive and based on the intensive annotation of humans, thus being not easily scaled in a sustainable way (Dong et al., 2023; Retzlaff et al., 2024). This paper presents an AI-based annotation pipeline that systematically identifies, labels, and fixes for instability patterns on LLM output. Our human-AI synergy method combines the models of automated weak supervision and confidence-based annotation with the target human validation to guarantee the reliability and moral uprightness of feedback information (Cabitza et al., 2023; Jiang et al., 2023). The semantic consistency, factual correctness, and logical coherence categories of stability-specific annotation are introduced into our framework, allowing the continuous calibration of models and the enhancement of their robustness based on the feedback loops (Honovich et al., 2021; Nan et al., 2021). 

---
# LoopBench: Discovering Emergent Symmetry Breaking Strategies with LLM Swarms 

**Authors**: Ali Parsaee, Yashar Talebirad, Csongor Szepesvári, Vishwajeet Ohal, Eden Redman  

**Link**: [PDF](https://arxiv.org/pdf/2512.13713)  

**Abstract**: Large Language Models (LLMs) are increasingly being utilized as autonomous agents, yet their ability to coordinate in distributed systems remains poorly understood. We introduce \textbf{LoopBench}, a benchmark to evaluate LLM reasoning in distributed symmetry breaking and meta-cognitive thinking. The benchmark focuses on coloring odd cycle graphs ($C_3, C_5, C_{11}$) with limited colors, where deterministic, non-communicating agents fail in infinite loops. A strategy passing mechanism is implemented as a form of consistent memory. We show that while standard LLMs and classical heuristics struggle, advanced reasoning models (e.g., O3) devise strategies to escape deadlocks. LoopBench allows the study of emergent distributed algorithms based on language-based reasoning, offering a testbed for collective intelligence. 

---
# Adjudicator: Correcting Noisy Labels with a KG-Informed Council of LLM Agents 

**Authors**: Doohee You, Sundeep Paul  

**Link**: [PDF](https://arxiv.org/pdf/2512.13704)  

**Abstract**: The performance of production machine learning systems is fundamentally limited by the quality of their training data. In high-stakes industrial applications, noisy labels can degrade performance and erode user trust. This paper presents Adjudicator, a system that addresses the critical data mining challenge of automatically identifying and correcting label noise and has been validated for production deployment. Adjudicator models this as a neuro-symbolic task, first constructing a dynamic Knowledge Graph (KG) to unify item context. This KG then informs a "Council of Agents," a novel multi-agent Large Language Model architecture where specialized agents debate and vote on a label's validity. We validate our system on a 1,000-item balanced subset of the AlleNoise benchmark. Our KG-informed model achieves a 0.99 F1-score, significantly outperforming a single-LLM baseline (0.48 F1) and a non-KG council (0.59 F1). Our analysis reveals this is due to a Precision, achieved by a novel override logic that uses the KG to perfectly identify complex, structural errors (complete Recall) -- a class of errors that baselines fail to find. This result demonstrates a robust and explainable system for automated, high-precision data verification, serving as a vital proof-of-concept for generating golden datasets in strictly governed industrial environments. 

---
# Blind Radio Mapping via Spatially Regularized Bayesian Trajectory Inference 

**Authors**: Zheng Xing, Junting Chen  

**Link**: [PDF](https://arxiv.org/pdf/2512.13701)  

**Abstract**: Radio maps enable intelligent wireless applications by capturing the spatial distribution of channel characteristics. However, conventional construction methods demand extensive location-labeled data, which are costly and impractical in many real-world scenarios. This paper presents a blind radio map construction framework that infers user trajectories from indoor multiple-input multiple-output (MIMO)-Orthogonal Frequency-Division Multiplexing (OFDM) channel measurements without relying on location labels. It first proves that channel state information (CSI) under non-line-of-sight (NLOS) exhibits spatial continuity under a quasi-specular environmental model, allowing the derivation of a CSI-distance metric that is proportional to the corresponding physical distance. For rectilinear trajectories in Poisson-distributed access point (AP) deployments, it is shown that the Cramer-Rao Lower Bound (CRLB) of localization error vanishes asymptotically, even under poor angular resolution. Building on these theoretical results, a spatially regularized Bayesian inference framework is developed that jointly estimates channel features, distinguishes line-of-sight (LOS)/NLOS conditions and recovers user trajectories. Experiments on a ray-tracing dataset demonstrate an average localization error of 0.68 m and a beam map reconstruction error of 3.3%, validating the effectiveness of the proposed blind mapping method. 

---
# Leveraging LLMs for Structured Data Extraction from Unstructured Patient Records 

**Authors**: Mitchell A. Klusty, Elizabeth C. Solie, Caroline N. Leach, W. Vaiden Logan, Lynnet E. Richey, John C. Gensel, David P. Szczykutowicz, Bryan C. McLellan, Emily B. Collier, Samuel E. Armstrong, V.K. Cody Bumgardner  

**Link**: [PDF](https://arxiv.org/pdf/2512.13700)  

**Abstract**: Manual chart review remains an extremely time-consuming and resource-intensive component of clinical research, requiring experts to extract often complex information from unstructured electronic health record (EHR) narratives. We present a secure, modular framework for automated structured feature extraction from clinical notes leveraging locally deployed large language models (LLMs) on institutionally approved, Health Insurance Portability and Accountability Act (HIPPA)-compliant compute infrastructure. This system integrates retrieval augmented generation (RAG) and structured response methods of LLMs into a widely deployable and scalable container to provide feature extraction for diverse clinical domains. In evaluation, the framework achieved high accuracy across multiple medical characteristics present in large bodies of patient notes when compared against an expert-annotated dataset and identified several annotation errors missed in manual review. This framework demonstrates the potential of LLM systems to reduce the burden of manual chart review through automated extraction and increase consistency in data capture, accelerating clinical research. 

---
# TimeLens: Rethinking Video Temporal Grounding with Multimodal LLMs 

**Authors**: Jun Zhang, Teng Wang, Yuying Ge, Yixiao Ge, Xinhao Li, Ying Shan, Limin Wang  

**Link**: [PDF](https://arxiv.org/pdf/2512.14698)  

**Abstract**: This paper does not introduce a novel method but instead establishes a straightforward, incremental, yet essential baseline for video temporal grounding (VTG), a core capability in video understanding. While multimodal large language models (MLLMs) excel at various video understanding tasks, the recipes for optimizing them for VTG remain under-explored. In this paper, we present TimeLens, a systematic investigation into building MLLMs with strong VTG ability, along two primary dimensions: data quality and algorithmic design. We first expose critical quality issues in existing VTG benchmarks and introduce TimeLens-Bench, comprising meticulously re-annotated versions of three popular benchmarks with strict quality criteria. Our analysis reveals dramatic model re-rankings compared to legacy benchmarks, confirming the unreliability of prior evaluation standards. We also address noisy training data through an automated re-annotation pipeline, yielding TimeLens-100K, a large-scale, high-quality training dataset. Building on our data foundation, we conduct in-depth explorations of algorithmic design principles, yielding a series of meaningful insights and effective yet efficient practices. These include interleaved textual encoding for time representation, a thinking-free reinforcement learning with verifiable rewards (RLVR) approach as the training paradigm, and carefully designed recipes for RLVR training. These efforts culminate in TimeLens models, a family of MLLMs with state-of-the-art VTG performance among open-source models and even surpass proprietary models such as GPT-5 and Gemini-2.5-Flash. All codes, data, and models will be released to facilitate future research. 

---
# Spherical Leech Quantization for Visual Tokenization and Generation 

**Authors**: Yue Zhao, Hanwen Jiang, Zhenlin Xu, Chutong Yang, Ehsan Adeli, Philipp Krähenbühl  

**Link**: [PDF](https://arxiv.org/pdf/2512.14697)  

**Abstract**: Non-parametric quantization has received much attention due to its efficiency on parameters and scalability to a large codebook. In this paper, we present a unified formulation of different non-parametric quantization methods through the lens of lattice coding. The geometry of lattice codes explains the necessity of auxiliary loss terms when training auto-encoders with certain existing lookup-free quantization variants such as BSQ. As a step forward, we explore a few possible candidates, including random lattices, generalized Fibonacci lattices, and densest sphere packing lattices. Among all, we find the Leech lattice-based quantization method, which is dubbed as Spherical Leech Quantization ($\Lambda_{24}$-SQ), leads to both a simplified training recipe and an improved reconstruction-compression tradeoff thanks to its high symmetry and even distribution on the hypersphere. In image tokenization and compression tasks, this quantization approach achieves better reconstruction quality across all metrics than BSQ, the best prior art, while consuming slightly fewer bits. The improvement also extends to state-of-the-art auto-regressive image generation frameworks. 

---
# Native and Compact Structured Latents for 3D Generation 

**Authors**: Jianfeng Xiang, Xiaoxue Chen, Sicheng Xu, Ruicheng Wang, Zelong Lv, Yu Deng, Hongyuan Zhu, Yue Dong, Hao Zhao, Nicholas Jing Yuan, Jiaolong Yang  

**Link**: [PDF](https://arxiv.org/pdf/2512.14692)  

**Abstract**: Recent advancements in 3D generative modeling have significantly improved the generation realism, yet the field is still hampered by existing representations, which struggle to capture assets with complex topologies and detailed appearance. This paper present an approach for learning a structured latent representation from native 3D data to address this challenge. At its core is a new sparse voxel structure called O-Voxel, an omni-voxel representation that encodes both geometry and appearance. O-Voxel can robustly model arbitrary topology, including open, non-manifold, and fully-enclosed surfaces, while capturing comprehensive surface attributes beyond texture color, such as physically-based rendering parameters. Based on O-Voxel, we design a Sparse Compression VAE which provides a high spatial compression rate and a compact latent space. We train large-scale flow-matching models comprising 4B parameters for 3D generation using diverse public 3D asset datasets. Despite their scale, inference remains highly efficient. Meanwhile, the geometry and material quality of our generated assets far exceed those of existing models. We believe our approach offers a significant advancement in 3D generative modeling. 

---
# Spoken DialogSum: An Emotion-Rich Conversational Dataset for Spoken Dialogue Summarization 

**Authors**: Yen-Ju Lu, Kunxiao Gao, Mingrui Liang, Helin Wang, Thomas Thebaud, Laureano Moro-Velazquez, Najim Dehak, Jesus Villalba  

**Link**: [PDF](https://arxiv.org/pdf/2512.14687)  

**Abstract**: Recent audio language models can follow long conversations. However, research on emotion-aware or spoken dialogue summarization is constrained by the lack of data that links speech, summaries, and paralinguistic cues. We introduce Spoken DialogSum, the first corpus aligning raw conversational audio with factual summaries, emotion-rich summaries, and utterance-level labels for speaker age, gender, and emotion. The dataset is built in two stages: first, an LLM rewrites DialogSum scripts with Switchboard-style fillers and back-channels, then tags each utterance with emotion, pitch, and speaking rate. Second, an expressive TTS engine synthesizes speech from the tagged scripts, aligned with paralinguistic labels. Spoken DialogSum comprises 13,460 emotion-diverse dialogues, each paired with both a factual and an emotion-focused summary. The dataset is available online at this https URL. Baselines show that an Audio-LLM raises emotional-summary ROUGE-L by 28% relative to a cascaded ASR-LLM system, confirming the value of end-to-end speech modeling. 

---
# Bias-Variance Trade-off for Clipped Stochastic First-Order Methods: From Bounded Variance to Infinite Mean 

**Authors**: Chuan He  

**Link**: [PDF](https://arxiv.org/pdf/2512.14686)  

**Abstract**: Stochastic optimization is fundamental to modern machine learning. Recent research has extended the study of stochastic first-order methods (SFOMs) from light-tailed to heavy-tailed noise, which frequently arises in practice, with clipping emerging as a key technique for controlling heavy-tailed gradients. Extensive theoretical advances have further shown that the oracle complexity of SFOMs depends on the tail index $\alpha$ of the noise. Nonetheless, existing complexity results often cover only the case $\alpha \in (1,2]$, that is, the regime where the noise has a finite mean, while the complexity bounds tend to infinity as $\alpha$ approaches $1$. This paper tackles the general case of noise with tail index $\alpha\in(0,2]$, covering regimes ranging from noise with bounded variance to noise with an infinite mean, where the latter case has been scarcely studied. Through a novel analysis of the bias-variance trade-off in gradient clipping, we show that when a symmetry measure of the noise tail is controlled, clipped SFOMs achieve improved complexity guarantees in the presence of heavy-tailed noise for any tail index $\alpha \in (0,2]$. Our analysis of the bias-variance trade-off not only yields new unified complexity guarantees for clipped SFOMs across this full range of tail indices, but is also straightforward to apply and can be combined with classical analyses under light-tailed noise to establish oracle complexity guarantees under heavy-tailed noise. Finally, numerical experiments validate our theoretical findings. 

---
# VASA-3D: Lifelike Audio-Driven Gaussian Head Avatars from a Single Image 

**Authors**: Sicheng Xu, Guojun Chen, Jiaolong Yang, Yizhong Zhang, Yu Deng, Steve Lin, Baining Guo  

**Link**: [PDF](https://arxiv.org/pdf/2512.14677)  

**Abstract**: We propose VASA-3D, an audio-driven, single-shot 3D head avatar generator. This research tackles two major challenges: capturing the subtle expression details present in real human faces, and reconstructing an intricate 3D head avatar from a single portrait image. To accurately model expression details, VASA-3D leverages the motion latent of VASA-1, a method that yields exceptional realism and vividness in 2D talking heads. A critical element of our work is translating this motion latent to 3D, which is accomplished by devising a 3D head model that is conditioned on the motion latent. Customization of this model to a single image is achieved through an optimization framework that employs numerous video frames of the reference head synthesized from the input image. The optimization takes various training losses robust to artifacts and limited pose coverage in the generated training data. Our experiment shows that VASA-3D produces realistic 3D talking heads that cannot be achieved by prior art, and it supports the online generation of 512x512 free-viewpoint videos at up to 75 FPS, facilitating more immersive engagements with lifelike 3D avatars. 

---
# gridfm-datakit-v1: A Python Library for Scalable and Realistic Power Flow and Optimal Power Flow Data Generation 

**Authors**: Alban Puech, Matteo Mazzonelli, Celia Cintas, Tamara R. Govindasamy, Mangaliso Mngomezulu, Jonas Weiss, Matteo Baù, Anna Varbella, François Mirallès, Kibaek Kim, Le Xie, Hendrik F. Hamann, Etienne Vos, Thomas Brunschwiler  

**Link**: [PDF](https://arxiv.org/pdf/2512.14658)  

**Abstract**: We introduce gridfm-datakit-v1, a Python library for generating realistic and diverse Power Flow (PF) and Optimal Power Flow (OPF) datasets for training Machine Learning (ML) solvers. Existing datasets and libraries face three main challenges: (1) lack of realistic stochastic load and topology perturbations, limiting scenario diversity; (2) PF datasets are restricted to OPF-feasible points, hindering generalization of ML solvers to cases that violate operating limits (e.g., branch overloads or voltage violations); and (3) OPF datasets use fixed generator cost functions, limiting generalization across varying costs. gridfm-datakit addresses these challenges by: (1) combining global load scaling from real-world profiles with localized noise and supporting arbitrary N-k topology perturbations to create diverse yet realistic datasets; (2) generating PF samples beyond operating limits; and (3) producing OPF data with varying generator costs. It also scales efficiently to large grids (up to 10,000 buses). Comparisons with OPFData, OPF-Learn, PGLearn, and PF$\Delta$ are provided. Available on GitHub at this https URL under Apache 2.0 and via `pip install gridfm-datakit`. 

---
# A Multicenter Benchmark of Multiple Instance Learning Models for Lymphoma Subtyping from HE-stained Whole Slide Images 

**Authors**: Rao Muhammad Umer, Daniel Sens, Jonathan Noll, Christian Matek, Lukas Wolfseher, Rainer Spang, Ralf Huss, Johannes Raffler, Sarah Reinke, Wolfram Klapper, Katja Steiger, Kristina Schwamborn, Carsten Marr  

**Link**: [PDF](https://arxiv.org/pdf/2512.14640)  

**Abstract**: Timely and accurate lymphoma diagnosis is essential for guiding cancer treatment. Standard diagnostic practice combines hematoxylin and eosin (HE)-stained whole slide images with immunohistochemistry, flow cytometry, and molecular genetic tests to determine lymphoma subtypes, a process requiring costly equipment, skilled personnel, and causing treatment delays. Deep learning methods could assist pathologists by extracting diagnostic information from routinely available HE-stained slides, yet comprehensive benchmarks for lymphoma subtyping on multicenter data are lacking. In this work, we present the first multicenter lymphoma benchmarking dataset covering four common lymphoma subtypes and healthy control tissue. We systematically evaluate five publicly available pathology foundation models (H-optimus-1, H0-mini, Virchow2, UNI2, Titan) combined with attention-based (AB-MIL) and transformer-based (TransMIL) multiple instance learning aggregators across three magnifications (10x, 20x, 40x). On in-distribution test sets, models achieve multiclass balanced accuracies exceeding 80% across all magnifications, with all foundation models performing similarly and both aggregation methods showing comparable results. The magnification study reveals that 40x resolution is sufficient, with no performance gains from higher resolutions or cross-magnification aggregation. However, on out-of-distribution test sets, performance drops substantially to around 60%, highlighting significant generalization challenges. To advance the field, larger multicenter studies covering additional rare lymphoma subtypes are needed. We provide an automated benchmarking pipeline to facilitate such future research. 

---
# MuseCPBench: an Empirical Study of Music Editing Methods through Music Context Preservation 

**Authors**: Yash Vishe, Eric Xue, Xunyi Jiang, Zachary Novack, Junda Wu, Julian McAuley, Xin Xu  

**Link**: [PDF](https://arxiv.org/pdf/2512.14629)  

**Abstract**: Music editing plays a vital role in modern music production, with applications in film, broadcasting, and game development. Recent advances in music generation models have enabled diverse editing tasks such as timbre transfer, instrument substitution, and genre transformation. However, many existing works overlook the evaluation of their ability to preserve musical facets that should remain unchanged during editing a property we define as Music Context Preservation (MCP). While some studies do consider MCP, they adopt inconsistent evaluation protocols and metrics, leading to unreliable and unfair comparisons. To address this gap, we introduce the first MCP evaluation benchmark, MuseCPBench, which covers four categories of musical facets and enables comprehensive comparisons across five representative music editing baselines. Through systematic analysis along musical facets, methods, and models, we identify consistent preservation gaps in current music editing methods and provide insightful explanations. We hope our findings offer practical guidance for developing more effective and reliable music editing strategies with strong MCP capability 

---
# JMMMU-Pro: Image-based Japanese Multi-discipline Multimodal Understanding Benchmark via Vibe Benchmark Construction 

**Authors**: Atsuyuki Miyai, Shota Onohara, Jeonghun Baek, Kiyoharu Aizawa  

**Link**: [PDF](https://arxiv.org/pdf/2512.14620)  

**Abstract**: This paper introduces JMMMU-Pro, an image-based Japanese Multi-discipline Multimodal Understanding Benchmark, and Vibe Benchmark Construction, a scalable construction method. Following the evolution from MMMU to MMMU-Pro, JMMMU-Pro extends JMMMU by composing the question image and question text into a single image, thereby creating a benchmark that requires integrated visual-textual understanding through visual perception. To build JMMMU-Pro, we propose Vibe Benchmark Construction, a methodology in which an image generative model (e.g., Nano Banana Pro) produces candidate visual questions, and humans verify the outputs and, when necessary, regenerate with adjusted prompts to ensure quality. By leveraging Nano Banana Pro's highly realistic image generation capabilities and its ability to embed clean Japanese text, we construct a high-quality benchmark at low cost, covering a wide range of background and layout designs. Experimental results show that all open-source LMMs struggle substantially with JMMMU-Pro, underscoring JMMMU-Pro as an important benchmark for guiding future efforts in the open-source community. We believe that JMMMU-Pro provides a more rigorous evaluation tool for assessing the Japanese capabilities of LMMs and that our Vibe Benchmark Construction also offers an efficient guideline for future development of image-based VQA benchmarks. 

---
# Model-Based Reinforcement Learning in Discrete-Action Non-Markovian Reward Decision Processes 

**Authors**: Alessandro Trapasso, Luca Iocchi, Fabio Patrizi  

**Link**: [PDF](https://arxiv.org/pdf/2512.14617)  

**Abstract**: Many practical decision-making problems involve tasks whose success depends on the entire system history, rather than on achieving a state with desired properties. Markovian Reinforcement Learning (RL) approaches are not suitable for such tasks, while RL with non-Markovian reward decision processes (NMRDPs) enables agents to tackle temporal-dependency tasks. This approach has long been known to lack formal guarantees on both (near-)optimality and sample efficiency. We contribute to solving both issues with QR-MAX, a novel model-based algorithm for discrete NMRDPs that factorizes Markovian transition learning from non-Markovian reward handling via reward machines. To the best of our knowledge, this is the first model-based RL algorithm for discrete-action NMRDPs that exploits this factorization to obtain PAC convergence to $\varepsilon$-optimal policies with polynomial sample complexity. We then extend QR-MAX to continuous state spaces with Bucket-QR-MAX, a SimHash-based discretiser that preserves the same factorized structure and achieves fast and stable learning without manual gridding or function approximation. We experimentally compare our method with modern state-of-the-art model-based RL approaches on environments of increasing complexity, showing a significant improvement in sample efficiency and increased robustness in finding optimal policies. 

---
# FakeRadar: Probing Forgery Outliers to Detect Unknown Deepfake Videos 

**Authors**: Zhaolun Li, Jichang Li, Yinqi Cai, Junye Chen, Xiaonan Luo, Guanbin Li, Rushi Lan  

**Link**: [PDF](https://arxiv.org/pdf/2512.14601)  

**Abstract**: In this paper, we propose FakeRadar, a novel deepfake video detection framework designed to address the challenges of cross-domain generalization in real-world scenarios. Existing detection methods typically rely on manipulation-specific cues, performing well on known forgery types but exhibiting severe limitations against emerging manipulation techniques. This poor generalization stems from their inability to adapt effectively to unseen forgery patterns. To overcome this, we leverage large-scale pretrained models (e.g. CLIP) to proactively probe the feature space, explicitly highlighting distributional gaps between real videos, known forgeries, and unseen manipulations. Specifically, FakeRadar introduces Forgery Outlier Probing, which employs dynamic subcluster modeling and cluster-conditional outlier generation to synthesize outlier samples near boundaries of estimated subclusters, simulating novel forgery artifacts beyond known manipulation types. Additionally, we design Outlier-Guided Tri-Training, which optimizes the detector to distinguish real, fake, and outlier samples using proposed outlier-driven contrastive learning and outlier-conditioned cross-entropy losses. Experiments show that FakeRadar outperforms existing methods across various benchmark datasets for deepfake video detection, particularly in cross-domain evaluations, by handling the variety of emerging manipulation techniques. 

---
# Towards Nepali-language LLMs: Efficient GPT training with a Nepali BPE tokenizer 

**Authors**: Adarsha Shrestha, Basanta Pokharel, Binit Shrestha, Smriti Adhikari, Dinesh Gothe  

**Link**: [PDF](https://arxiv.org/pdf/2512.14585)  

**Abstract**: Nepali, a low-resource language spoken by over 32 million people, continues to face challenges in natural language processing (NLP) due to its complex grammar, agglutinative morphology, and limited availability of high-quality corpora. Most efforts to date have centered on basic encoder architectures; they remain insufficient for Nepali-specific text generation. This study presents a GPT-2-based Nepali language model trained using several training strategies inspired by GPT-3, including optimized learning rate schedules, batch scaling, and architectural refinements. A custom 16k Byte-Pair Encoding (BPE) tokenizer was trained exclusively on Nepali text to ensure more consistent segmentation and improved input representation. The model was pretrained on a combined dataset comprising a 10.75GB cleaned NepBERTa corpus and additional web-scraped Nepali news articles. FlashAttention was integrated to reduce memory usage and stabilize training. After two epochs, the model achieved a training loss of 3.168177, a validation loss of 3.081982, and a final perplexity of 21.80, demonstrating its capability to generate coherent Nepali news-style text. 

---
# Low-Resource, High-Impact: Building Corpora for Inclusive Language Technologies 

**Authors**: Ekaterina Artemova, Laurie Burchell, Daryna Dementieva, Shu Okabe, Mariya Shmatova, Pedro Ortiz Suarez  

**Link**: [PDF](https://arxiv.org/pdf/2512.14576)  

**Abstract**: This tutorial (this https URL) is designed for NLP practitioners, researchers, and developers working with multilingual and low-resource languages who seek to create more equitable and socially impactful language technologies. Participants will walk away with a practical toolkit for building end-to-end NLP pipelines for underrepresented languages -- from data collection and web crawling to parallel sentence mining, machine translation, and downstream applications such as text classification and multimodal reasoning. The tutorial presents strategies for tackling the challenges of data scarcity and cultural variance, offering hands-on methods and modeling frameworks. We will focus on fair, reproducible, and community-informed development approaches, grounded in real-world scenarios. We will showcase a diverse set of use cases covering over 10 languages from different language families and geopolitical contexts, including both digitally resource-rich and severely underrepresented languages. 

---
# Residual GRU+MHSA: A Lightweight Hybrid Recurrent Attention Model for Cardiovascular Disease Detection 

**Authors**: Tejaswani Dash, Gautam Datla, Anudeep Vurity, Tazeem Ahmad, Mohd Adnan, Saima Rafi, Saisha Patro, Saina Patro  

**Link**: [PDF](https://arxiv.org/pdf/2512.14563)  

**Abstract**: Cardiovascular disease (CVD) remains the leading cause of mortality worldwide, underscoring the need for reliable and efficient predictive tools that support early intervention. Traditional diagnostic approaches rely on handcrafted features and clinician expertise, while machine learning methods improve reproducibility but often struggle to generalize across noisy and heterogeneous clinical data. In this work, we propose Residual GRU with Multi-Head Self-Attention, a compact deep learning architecture designed for tabular clinical records. The model integrates residual bidirectional gated recurrent units for sequential modeling of feature columns, a channel reweighting block, and multi-head self-attention pooling with a learnable classification token to capture global context. We evaluate the model on the UCI Heart Disease dataset using 5-fold stratified cross-validation and compare it against classical methods such as Logistic Regression, Random Forest, and Support Vector Machines, as well as modern deep learning baselines including DeepMLP, convolutional networks, recurrent networks, and Transformers. The proposed model achieves an accuracy of 0.861, macro-F1 of 0.860, ROC-AUC of 0.908, and PR-AUC of 0.904, outperforming all baselines. Ablation studies confirm the individual contributions of residual recurrence, channel gating, and attention pooling. t-SNE visualizations further indicate that the learned embeddings exhibit clearer separation between disease and non-disease classes compared to raw features. These results demonstrate that lightweight hybrid recurrent and attention-based architectures provide a strong balance between accuracy and efficiency for clinical risk prediction, supporting deployment in resource-constrained healthcare settings. 

---
# Polypersona: Persona-Grounded LLM for Synthetic Survey Responses 

**Authors**: Tejaswani Dash, Dinesh Karri, Anudeep Vurity, Gautam Datla, Tazeem Ahmad, Saima Rafi, Rohith Tangudu  

**Link**: [PDF](https://arxiv.org/pdf/2512.14562)  

**Abstract**: This paper introduces PolyPersona, a generative framework for synthesizing persona-conditioned survey responses across multiple domains. The framework instruction-tunes compact chat models using parameter-efficient LoRA adapters with 4-bit quantization under a resource-adaptive training setup. A dialogue-based data pipeline explicitly preserves persona cues, ensuring consistent behavioral alignment across generated responses. Using this pipeline, we construct a dataset of 3,568 synthetic survey responses spanning ten domains and 433 distinct personas, enabling controlled instruction tuning and systematic multi-domain evaluation. We evaluate the generated responses using a multi-metric evaluation suite that combines standard text generation metrics, including BLEU, ROUGE, and BERTScore, with survey-specific metrics designed to assess structural coherence, stylistic consistency, and sentiment this http URL results show that compact models such as TinyLlama 1.1B and Phi-2 achieve performance comparable to larger 7B to 8B baselines, with a highest BLEU score of 0.090 and ROUGE-1 of 0.429. These findings demonstrate that persona-conditioned fine-tuning enables small language models to generate reliable and coherent synthetic survey data. The proposed framework provides an efficient and reproducible approach for survey data generation, supporting scalable evaluation while facilitating bias analysis through transparent and open protocols. 

---
# CLNet: Cross-View Correspondence Makes a Stronger Geo-Localizationer 

**Authors**: Xianwei Cao, Dou Quan, Shuang Wang, Ning Huyan, Wei Wang, Yunan Li, Licheng Jiao  

**Link**: [PDF](https://arxiv.org/pdf/2512.14560)  

**Abstract**: Image retrieval-based cross-view geo-localization (IRCVGL) aims to match images captured from significantly different viewpoints, such as satellite and street-level images. Existing methods predominantly rely on learning robust global representations or implicit feature alignment, which often fail to model explicit spatial correspondences crucial for accurate localization. In this work, we propose a novel correspondence-aware feature refinement framework, termed CLNet, that explicitly bridges the semantic and geometric gaps between different views. CLNet decomposes the view alignment process into three learnable and complementary modules: a Neural Correspondence Map (NCM) that spatially aligns cross-view features via latent correspondence fields; a Nonlinear Embedding Converter (NEC) that remaps features across perspectives using an MLP-based transformation; and a Global Feature Recalibration (GFR) module that reweights informative feature channels guided by learned spatial cues. The proposed CLNet can jointly capture both high-level semantics and fine-grained alignments. Extensive experiments on four public benchmarks, CVUSA, CVACT, VIGOR, and University-1652, demonstrate that our proposed CLNet achieves state-of-the-art performance while offering better interpretability and generalizability. 

---
# VLegal-Bench: Cognitively Grounded Benchmark for Vietnamese Legal Reasoning of Large Language Models 

**Authors**: Nguyen Tien Dong, Minh-Anh Nguyen, Thanh Dat Hoang, Nguyen Tuan Ngoc, Dao Xuan Quang Minh, Phan Phi Hai, Nguyen Thi Ngoc Anh, Dang Van Tu, Binh Vu  

**Link**: [PDF](https://arxiv.org/pdf/2512.14554)  

**Abstract**: The rapid advancement of large language models (LLMs) has enabled new possibilities for applying artificial intelligence within the legal domain. Nonetheless, the complexity, hierarchical organization, and frequent revisions of Vietnamese legislation pose considerable challenges for evaluating how well these models interpret and utilize legal knowledge. To address this gap, Vietnamese Legal Benchmark (VLegal-Bench) is introduced, the first comprehensive benchmark designed to systematically assess LLMs on Vietnamese legal tasks. Informed by Bloom's cognitive taxonomy, VLegal-Bench encompasses multiple levels of legal understanding through tasks designed to reflect practical usage scenarios. The benchmark comprises 10,450 samples generated through a rigorous annotation pipeline, where legal experts label and cross-validate each instance using our annotation system to ensure every sample is grounded in authoritative legal documents and mirrors real-world legal assistant workflows, including general legal questions and answers, retrieval-augmented generation, multi-step reasoning, and scenario-based problem solving tailored to Vietnamese law. By providing a standardized, transparent, and cognitively informed evaluation framework, VLegal-Bench establishes a solid foundation for assessing LLM performance in Vietnamese legal contexts and supports the development of more reliable, interpretable, and ethically aligned AI-assisted legal systems. 

---
# Dual Language Models: Balancing Training Efficiency and Overfitting Resilience 

**Authors**: David Samuel, Lucas Georges Gabriel Charpentier  

**Link**: [PDF](https://arxiv.org/pdf/2512.14549)  

**Abstract**: This paper combines autoregressive and masked-diffusion training objectives without any architectural modifications, resulting in flexible language models that outperform single-objective models. Autoregressive modeling has been a popular approach, partly because of its training efficiency; however, that comes at the cost of sensitivity to overfitting. On the other hand, masked-diffusion models are less efficient to train while being more resilient to overfitting. In this work, we demonstrate that dual-objective training achieves the best of both worlds. To derive the optimal ratio between both objectives, we train and evaluate 50 language models under varying levels of data repetition. We show that it is optimal to combine both objectives under all evaluated settings and that the optimal ratio is similar whether targeting autoregressive or masked-diffusion downstream performance. 

---
# CAPRMIL: Context-Aware Patch Representations for Multiple Instance Learning 

**Authors**: Andreas Lolos, Theofilos Christodoulou, Aris L. Moustakas, Stergios Christodoulidis, Maria Vakalopoulou  

**Link**: [PDF](https://arxiv.org/pdf/2512.14540)  

**Abstract**: In computational pathology, weak supervision has become the standard for deep learning due to the gigapixel scale of WSIs and the scarcity of pixel-level annotations, with Multiple Instance Learning (MIL) established as the principal framework for slide-level model training. In this paper, we introduce a novel setting for MIL methods, inspired by proceedings in Neural Partial Differential Equation (PDE) Solvers. Instead of relying on complex attention-based aggregation, we propose an efficient, aggregator-agnostic framework that removes the complexity of correlation learning from the MIL aggregator. CAPRMIL produces rich context-aware patch embeddings that promote effective correlation learning on downstream tasks. By projecting patch features -- extracted using a frozen patch encoder -- into a small set of global context/morphology-aware tokens and utilizing multi-head self-attention, CAPRMIL injects global context with linear computational complexity with respect to the bag size. Paired with a simple Mean MIL aggregator, CAPRMIL matches state-of-the-art slide-level performance across multiple public pathology benchmarks, while reducing the total number of trainable parameters by 48%-92.8% versus SOTA MILs, lowering FLOPs during inference by 52%-99%, and ranking among the best models on GPU memory efficiency and training time. Our results indicate that learning rich, context-aware instance representations before aggregation is an effective and scalable alternative to complex pooling for whole-slide analysis. Our code is available at this https URL 

---
# SASQ: Static Activation Scaling for Quantization-Aware Training in Large Language Models 

**Authors**: Shizhuo Mao, Song Chen, Yi Kang  

**Link**: [PDF](https://arxiv.org/pdf/2512.14481)  

**Abstract**: Large language models (LLMs) excel at natural language tasks but face deployment challenges due to their growing size outpacing GPU memory advancements. Model quantization mitigates this issue by lowering weight and activation precision, but existing solutions face fundamental trade-offs: dynamic quantization incurs high computational overhead and poses deployment challenges on edge devices, while static quantization sacrifices accuracy. Existing approaches of quantization-aware training (QAT) further suffer from weight training costs. We propose SASQ: a lightweight QAT framework specifically tailored for activation quantization factors. SASQ exclusively optimizes only the quantization factors (without changing pre-trained weights), enabling static inference with high accuracy while maintaining deployment efficiency. SASQ adaptively truncates some outliers, thereby reducing the difficulty of quantization while preserving the distributional characteristics of the activations. SASQ not only surpasses existing SOTA quantization schemes but also outperforms the corresponding FP16 models. On LLaMA2-7B, it achieves 5.2% lower perplexity than QuaRot and 4.7% lower perplexity than the FP16 model on WikiText2. 

---
# TACK Tunnel Data (TTD): A Benchmark Dataset for Deep Learning-Based Defect Detection in Tunnels 

**Authors**: Andreas Sjölander, Valeria Belloni, Robel Fekadu, Andrea Nascetti  

**Link**: [PDF](https://arxiv.org/pdf/2512.14477)  

**Abstract**: Tunnels are essential elements of transportation infrastructure, but are increasingly affected by ageing and deterioration mechanisms such as cracking. Regular inspections are required to ensure their safety, yet traditional manual procedures are time-consuming, subjective, and costly. Recent advances in mobile mapping systems and Deep Learning (DL) enable automated visual inspections. However, their effectiveness is limited by the scarcity of tunnel datasets. This paper introduces a new publicly available dataset containing annotated images of three different tunnel linings, capturing typical defects: cracks, leaching, and water infiltration. The dataset is designed to support supervised, semi-supervised, and unsupervised DL methods for defect detection and segmentation. Its diversity in texture and construction techniques also enables investigation of model generalization and transferability across tunnel types. By addressing the critical lack of domain-specific data, this dataset contributes to advancing automated tunnel inspection and promoting safer, more efficient infrastructure maintenance strategies. 

---
# Reasoning-Style Poisoning of LLM Agents via Stealthy Style Transfer: Process-Level Attacks and Runtime Monitoring in RSV Space 

**Authors**: Xingfu Zhou, Pengfei Wang  

**Link**: [PDF](https://arxiv.org/pdf/2512.14448)  

**Abstract**: Large Language Model (LLM) agents relying on external retrieval are increasingly deployed in high-stakes environments. While existing adversarial attacks primarily focus on content falsification or instruction injection, we identify a novel, process-oriented attack surface: the agent's reasoning style. We propose Reasoning-Style Poisoning (RSP), a paradigm that manipulates how agents process information rather than what they process. We introduce Generative Style Injection (GSI), an attack method that rewrites retrieved documents into pathological tones--specifically "analysis paralysis" or "cognitive haste"--without altering underlying facts or using explicit triggers. To quantify these shifts, we develop the Reasoning Style Vector (RSV), a metric tracking Verification depth, Self-confidence, and Attention focus. Experiments on HotpotQA and FEVER using ReAct, Reflection, and Tree of Thoughts (ToT) architectures reveal that GSI significantly degrades performance. It increases reasoning steps by up to 4.4 times or induces premature errors, successfully bypassing state-of-the-art content filters. Finally, we propose RSP-M, a lightweight runtime monitor that calculates RSV metrics in real-time and triggers alerts when values exceed safety thresholds. Our work demonstrates that reasoning style is a distinct, exploitable vulnerability, necessitating process-level defenses beyond static content analysis. 

---
# Effect of Document Packing on the Latent Multi-Hop Reasoning Capabilities of Large Language Models 

**Authors**: Gabriele Prato, Shagun Sodhani, Alessandro Sordoni, Sarath Chandar  

**Link**: [PDF](https://arxiv.org/pdf/2512.14427)  

**Abstract**: The standard practice for training large language models involves packing multiple documents together to optimize computational efficiency. However, the impact of this process on the models' capabilities remains largely unexplored. To address this gap, we investigate how different document-packing strategies influence the latent multi-hop reasoning abilities of LLMs. Our findings indicate that packing can improve model performance compared to training on individual documents, at the expense of more compute. To further understand the underlying mechanisms, we conduct an ablation study, identifying key factors that explain the advantages of packing. Ultimately, our research deepens the understanding of LLM training dynamics and provides practical insights for optimizing model development. 

---
# DISCODE: Distribution-Aware Score Decoder for Robust Automatic Evaluation of Image Captioning 

**Authors**: Nakamasa Inoue, Kanoko Goto, Masanari Oi, Martyna Gruszka, Mahiro Ukai, Takumi Hirose, Yusuke Sekikawa  

**Link**: [PDF](https://arxiv.org/pdf/2512.14420)  

**Abstract**: Large vision-language models (LVLMs) have shown impressive performance across a broad range of multimodal tasks. However, robust image caption evaluation using LVLMs remains challenging, particularly under domain-shift scenarios. To address this issue, we introduce the Distribution-Aware Score Decoder (DISCODE), a novel finetuning-free method that generates robust evaluation scores better aligned with human judgments across diverse domains. The core idea behind DISCODE lies in its test-time adaptive evaluation approach, which introduces the Adaptive Test-Time (ATT) loss, leveraging a Gaussian prior distribution to improve robustness in evaluation score estimation. This loss is efficiently minimized at test time using an analytical solution that we derive. Furthermore, we introduce the Multi-domain Caption Evaluation (MCEval) benchmark, a new image captioning evaluation benchmark covering six distinct domains, designed to assess the robustness of evaluation metrics. In our experiments, we demonstrate that DISCODE achieves state-of-the-art performance as a reference-free evaluation metric across MCEval and four representative existing benchmarks. 

---
# RePo: Language Models with Context Re-Positioning 

**Authors**: Huayang Li, Tianyu Zhao, Richard Sproat  

**Link**: [PDF](https://arxiv.org/pdf/2512.14391)  

**Abstract**: In-context learning is fundamental to modern Large Language Models (LLMs); however, prevailing architectures impose a rigid and fixed contextual structure by assigning linear or constant positional indices. Drawing on Cognitive Load Theory (CLT), we argue that this uninformative structure increases extraneous cognitive load, consuming finite working memory capacity that should be allocated to deep reasoning and attention allocation. To address this, we propose RePo, a novel mechanism that reduces extraneous load via context re-positioning. Unlike standard approaches, RePo utilizes a differentiable module, $f_\phi$, to assign token positions that capture contextual dependencies, rather than replying on pre-defined integer range. By continually pre-training on the OLMo-2 1B backbone, we demonstrate that RePo significantly enhances performance on tasks involving noisy contexts, structured data, and longer context length, while maintaining competitive performance on general short-context tasks. Detailed analysis reveals that RePo successfully allocate higher attention to distant but relevant information, assign positions in dense and non-linear space, and capture the intrinsic structure of the input context. Our code is available at this https URL. 

---
# Causal Structure Learning for Dynamical Systems with Theoretical Score Analysis 

**Authors**: Nicholas Tagliapietra, Katharina Ensinger, Christoph Zimmer, Osman Mian  

**Link**: [PDF](https://arxiv.org/pdf/2512.14361)  

**Abstract**: Real world systems evolve in continuous-time according to their underlying causal relationships, yet their dynamics are often unknown. Existing approaches to learning such dynamics typically either discretize time -- leading to poor performance on irregularly sampled data -- or ignore the underlying causality. We propose CaDyT, a novel method for causal discovery on dynamical systems addressing both these challenges. In contrast to state-of-the-art causal discovery methods that model the problem using discrete-time Dynamic Bayesian networks, our formulation is grounded in Difference-based causal models, which allow milder assumptions for modeling the continuous nature of the system. CaDyT leverages exact Gaussian Process inference for modeling the continuous-time dynamics which is more aligned with the underlying dynamical process. We propose a practical instantiation that identifies the causal structure via a greedy search guided by the Algorithmic Markov Condition and Minimum Description Length principle. Our experiments show that CaDyT outperforms state-of-the-art methods on both regularly and irregularly-sampled data, discovering causal networks closer to the true underlying dynamics. 

---
# Enhancing Interpretability for Vision Models via Shapley Value Optimization 

**Authors**: Kanglong Fan, Yunqiao Yang, Chen Ma  

**Link**: [PDF](https://arxiv.org/pdf/2512.14354)  

**Abstract**: Deep neural networks have demonstrated remarkable performance across various domains, yet their decision-making processes remain opaque. Although many explanation methods are dedicated to bringing the obscurity of DNNs to light, they exhibit significant limitations: post-hoc explanation methods often struggle to faithfully reflect model behaviors, while self-explaining neural networks sacrifice performance and compatibility due to their specialized architectural designs. To address these challenges, we propose a novel self-explaining framework that integrates Shapley value estimation as an auxiliary task during training, which achieves two key advancements: 1) a fair allocation of the model prediction scores to image patches, ensuring explanations inherently align with the model's decision logic, and 2) enhanced interpretability with minor structural modifications, preserving model performance and compatibility. Extensive experiments on multiple benchmarks demonstrate that our method achieves state-of-the-art interpretability. 

---
# Towards Transferable Defense Against Malicious Image Edits 

**Authors**: Jie Zhang, Shuai Dong, Shiguang Shan, Xilin Chen  

**Link**: [PDF](https://arxiv.org/pdf/2512.14341)  

**Abstract**: Recent approaches employing imperceptible perturbations in input images have demonstrated promising potential to counter malicious manipulations in diffusion-based image editing systems. However, existing methods suffer from limited transferability in cross-model evaluations. To address this, we propose Transferable Defense Against Malicious Image Edits (TDAE), a novel bimodal framework that enhances image immunity against malicious edits through coordinated image-text optimization. Specifically, at the visual defense level, we introduce FlatGrad Defense Mechanism (FDM), which incorporates gradient regularization into the adversarial objective. By explicitly steering the perturbations toward flat minima, FDM amplifies immune robustness against unseen editing models. For textual enhancement protection, we propose an adversarial optimization paradigm named Dynamic Prompt Defense (DPD), which periodically refines text embeddings to align the editing outcomes of immunized images with those of the original images, then updates the images under optimized embeddings. Through iterative adversarial updates to diverse embeddings, DPD enforces the generation of immunized images that seek a broader set of immunity-enhancing features, thereby achieving cross-model transferability. Extensive experimental results demonstrate that our TDAE achieves state-of-the-art performance in mitigating malicious edits under both intra- and cross-model evaluations. 

---
# Dual Attention Guided Defense Against Malicious Edits 

**Authors**: Jie Zhang, Shuai Dong, Shiguang Shan, Xilin Chen  

**Link**: [PDF](https://arxiv.org/pdf/2512.14333)  

**Abstract**: Recent progress in text-to-image diffusion models has transformed image editing via text prompts, yet this also introduces significant ethical challenges from potential misuse in creating deceptive or harmful content. While current defenses seek to mitigate this risk by embedding imperceptible perturbations, their effectiveness is limited against malicious tampering. To address this issue, we propose a Dual Attention-Guided Noise Perturbation (DANP) immunization method that adds imperceptible perturbations to disrupt the model's semantic understanding and generation process. DANP functions over multiple timesteps to manipulate both cross-attention maps and the noise prediction process, using a dynamic threshold to generate masks that identify text-relevant and irrelevant regions. It then reduces attention in relevant areas while increasing it in irrelevant ones, thereby misguides the edit towards incorrect regions and preserves the intended targets. Additionally, our method maximizes the discrepancy between the injected noise and the model's predicted noise to further interfere with the generation. By targeting both attention and noise prediction mechanisms, DANP exhibits impressive immunity against malicious edits, and extensive experiments confirm that our method achieves state-of-the-art performance. 

---
# Step-Tagging: Toward controlling the generation of Language Reasoning Models through step monitoring 

**Authors**: Yannis Belkhiter, Seshu Tirupathi, Giulio Zizzo, John D. Kelleher  

**Link**: [PDF](https://arxiv.org/pdf/2512.14332)  

**Abstract**: The field of Language Reasoning Models (LRMs) has been very active over the past few years with advances in training and inference techniques enabling LRMs to reason longer, and more accurately. However, a growing body of studies show that LRMs are still inefficient, over-generating verification and reflection steps. To address this challenge, we introduce the Step-Tagging framework, a lightweight sentence-classifier enabling real-time annotation of the type of reasoning steps that an LRM is generating. To monitor reasoning behaviors, we introduced ReasonType: a novel taxonomy of reasoning steps. Building on this framework, we demonstrated that online monitoring of the count of specific steps can produce effective interpretable early stopping criteria of LRM inferences. We evaluate the Step-tagging framework on three open-source reasoning models across standard benchmark datasets: MATH500, GSM8K, AIME and non-mathematical tasks (GPQA and MMLU-Pro). We achieve 20 to 50\% token reduction while maintaining comparable accuracy to standard generation, with largest gains observed on more computation-heavy tasks. This work offers a novel way to increase control over the generation of LRMs, and a new tool to study behaviors of LRMs. 

---
# Criminal Liability in AI-Enabled Autonomous Vehicles: A Comparative Study 

**Authors**: Sahibpreet Singh, Manjit Singh  

**Link**: [PDF](https://arxiv.org/pdf/2512.14330)  

**Abstract**: AI revolutionizes transportation through autonomous vehicles (AVs) but introduces complex criminal liability issues regarding infractions. This study employs a comparative legal analysis of primary statutes, real-world liability claims, and academic literature across the US, Germany, UK, China, and India; jurisdictions selected for their technological advancement and contrasting regulatory approaches. The research examines the attribution of human error, AI moral agency, and the identification of primary offenders in AV incidents. Findings reveal fragmented regulatory landscapes: India and the US rely on loose networks of state laws, whereas the UK enacted the pioneering Automated and Electric Vehicles Act 2018. Germany enforces strict safety standards, distinguishing liability based on the vehicle's operating mode, while China similarly aims for a stringent liability regime. The study concludes that globally harmonized legal standards are essential to foster technological innovation while ensuring minimum risk and clear liability attribution. 

---
# A data-physics hybrid generative model for patient-specific post-stroke motor rehabilitation using wearable sensor data 

**Authors**: Yanning Dai, Chenyu Tang, Ruizhi Zhang, Wenyu Yang, Yilan Zhang, Yuhui Wang, Junliang Chen, Xuhang Chen, Ruimou Xie, Yangyue Cao, Qiaoying Li, Jin Cao, Tao Li, Hubin Zhao, Yu Pan, Arokia Nathan, Xin Gao, Peter Smielewski, Shuo Gao  

**Link**: [PDF](https://arxiv.org/pdf/2512.14329)  

**Abstract**: Dynamic prediction of locomotor capacity after stroke is crucial for tailoring rehabilitation, yet current assessments provide only static impairment scores and do not indicate whether patients can safely perform specific tasks such as slope walking or stair climbing. Here, we develop a data-physics hybrid generative framework that reconstructs an individual stroke survivor's neuromuscular control from a single 20 m level-ground walking trial and predicts task-conditioned locomotion across rehabilitation scenarios. The system combines wearable-sensor kinematics, a proportional-derivative physics controller, a population Healthy Motion Atlas, and goal-conditioned deep reinforcement learning with behaviour cloning and generative adversarial imitation learning to generate physically plausible, patient-specific gait simulations for slopes and stairs. In 11 stroke survivors, the personalized controllers preserved idiosyncratic gait patterns while improving joint-angle and endpoint fidelity by 4.73% and 12.10%, respectively, and reducing training time to 25.56% relative to a physics-only baseline. In a multicentre pilot involving 21 inpatients, clinicians who used our locomotion predictions to guide task selection and difficulty obtained larger gains in Fugl-Meyer lower-extremity scores over 28 days of standard rehabilitation than control clinicians (mean change 6.0 versus 3.7 points). These findings indicate that our generative, task-predictive framework can augment clinical decision-making in post-stroke gait rehabilitation and provide a template for dynamically personalized motor recovery strategies. 

---
# Semantic Mismatch and Perceptual Degradation: A New Perspective on Image Editing Immunity 

**Authors**: Shuai Dong, Jie Zhang, Guoying Zhao, Shiguang Shan, Xilin Chen  

**Link**: [PDF](https://arxiv.org/pdf/2512.14320)  

**Abstract**: Text-guided image editing via diffusion models, while powerful, raises significant concerns about misuse, motivating efforts to immunize images against unauthorized edits using imperceptible perturbations. Prevailing metrics for evaluating immunization success typically rely on measuring the visual dissimilarity between the output generated from a protected image and a reference output generated from the unprotected original. This approach fundamentally overlooks the core requirement of image immunization, which is to disrupt semantic alignment with attacker intent, regardless of deviation from any specific output. We argue that immunization success should instead be defined by the edited output either semantically mismatching the prompt or suffering substantial perceptual degradations, both of which thwart malicious intent. To operationalize this principle, we propose Synergistic Intermediate Feature Manipulation (SIFM), a method that strategically perturbs intermediate diffusion features through dual synergistic objectives: (1) maximizing feature divergence from the original edit trajectory to disrupt semantic alignment with the expected edit, and (2) minimizing feature norms to induce perceptual degradations. Furthermore, we introduce the Immunization Success Rate (ISR), a novel metric designed to rigorously quantify true immunization efficacy for the first time. ISR quantifies the proportion of edits where immunization induces either semantic failure relative to the prompt or significant perceptual degradations, assessed via Multimodal Large Language Models (MLLMs). Extensive experiments show our SIFM achieves the state-of-the-art performance for safeguarding visual content against malicious diffusion-based manipulation. 

---
# From YOLO to VLMs: Advancing Zero-Shot and Few-Shot Detection of Wastewater Treatment Plants Using Satellite Imagery in MENA Region 

**Authors**: Akila Premarathna, Kanishka Hewageegana, Garcia Andarcia Mariangel  

**Link**: [PDF](https://arxiv.org/pdf/2512.14312)  

**Abstract**: In regions of the Middle East and North Africa (MENA), there is a high demand for wastewater treatment plants (WWTPs), crucial for sustainable water management. Precise identification of WWTPs from satellite images enables environmental monitoring. Traditional methods like YOLOv8 segmentation require extensive manual labeling. But studies indicate that vision-language models (VLMs) are an efficient alternative to achieving equivalent or superior results through inherent reasoning and annotation. This study presents a structured methodology for VLM comparison, divided into zero-shot and few-shot streams specifically to identify WWTPs. The YOLOv8 was trained on a governmental dataset of 83,566 high-resolution satellite images from Egypt, Saudi Arabia, and UAE: ~85% WWTPs (positives), 15% non-WWTPs (negatives). Evaluated VLMs include LLaMA 3.2 Vision, Qwen 2.5 VL, DeepSeek-VL2, Gemma 3, Gemini, and Pixtral 12B (Mistral), used to identify WWTP components such as circular/rectangular tanks, aeration basins and distinguish confounders via expert prompts producing JSON outputs with confidence and descriptions. The dataset comprises 1,207 validated WWTP locations (198 UAE, 354 KSA, 655 Egypt) and equal non-WWTP sites from field/AI data, as 600mx600m Geo-TIFF images (Zoom 18, EPSG:4326). Zero-shot evaluations on WWTP images showed several VLMs out-performing YOLOv8's true positive rate, with Gemma-3 highest. Results confirm that VLMs, particularly with zero-shot, can replace YOLOv8 for efficient, annotation-free WWTP classification, enabling scalable remote sensing. 

---
# A Threshold-Triggered Deep Q-Network-Based Framework for Self-Healing in Autonomic Software-Defined IIoT-Edge Networks 

**Authors**: Agrippina Mwangi, León Navarro-Hilfiker, Lukasz Brewka, Mikkel Gryning, Elena Fumagalli, Madeleine Gibescu  

**Link**: [PDF](https://arxiv.org/pdf/2512.14297)  

**Abstract**: Stochastic disruptions such as flash events arising from benign traffic bursts and switch thermal fluctuations are major contributors to intermittent service degradation in software-defined industrial networks. These events violate IEC~61850-derived quality-of-service requirements and user-defined service-level agreements, hindering the reliable and timely delivery of control, monitoring, and best-effort traffic in IEC~61400-25-compliant wind power plants. Failure to maintain these requirements often results in delayed or lost control signals, reduced operational efficiency, and increased risk of wind turbine generator downtime.
To address these challenges, this study proposes a threshold-triggered Deep Q-Network self-healing agent that autonomically detects, analyzes, and mitigates network disruptions while adapting routing behavior and resource allocation in real time. The proposed agent was trained, validated, and tested on an emulated tri-clustered switch network deployed in a cloud-based proof-of-concept testbed.
Simulation results show that the proposed agent improves disruption recovery performance by 53.84% compared to a baseline shortest-path and load-balanced routing approach and outperforms state-of-the-art methods, including the Adaptive Network-based Fuzzy Inference System by 13.1% and the Deep Q-Network and traffic prediction-based routing optimization method by 21.5%, in a super-spine leaf data-plane architecture.
Additionally, the agent maintains switch thermal stability by proactively initiating external rack cooling when required. These findings highlight the potential of deep reinforcement learning in building resilience in software-defined industrial networks deployed in mission-critical, time-sensitive application scenarios. 

---
# The Trust in AI-Generated Health Advice (TAIGHA) Scale and Short Version (TAIGHA-S): Development and Validation Study 

**Authors**: Marvin Kopka, Azeem Majeed, Gabriella Spinelli, Austen El-Osta, Markus Feufel  

**Link**: [PDF](https://arxiv.org/pdf/2512.14278)  

**Abstract**: Artificial Intelligence tools such as large language models are increasingly used by the public to obtain health information and guidance. In health-related contexts, following or rejecting AI-generated advice can have direct clinical implications. Existing instruments like the Trust in Automated Systems Survey assess trustworthiness of generic technology, and no validated instrument measures users' trust in AI-generated health advice specifically. This study developed and validated the Trust in AI-Generated Health Advice (TAIGHA) scale and its four-item short form (TAIGHA-S) as theory-based instruments measuring trust and distrust, each with cognitive and affective components. The items were developed using a generative AI approach, followed by content validation with 10 domain experts, face validation with 30 lay participants, and psychometric validation with 385 UK participants who received AI-generated advice in a symptom-assessment scenario. After automated item reduction, 28 items were retained and reduced to 10 based on expert ratings. TAIGHA showed excellent content validity (S-CVI/Ave=0.99) and CFA confirmed a two-factor model with excellent fit (CFI=0.98, TLI=0.98, RMSEA=0.07, SRMR=0.03). Internal consistency was high ({\alpha}=0.95). Convergent validity was supported by correlations with the Trust in Automated Systems Survey (r=0.67/-0.66) and users' reliance on the AI's advice (r=0.37 for trust), while divergent validity was supported by low correlations with reading flow and mental load (all |r|<0.25). TAIGHA-S correlated highly with the full scale (r=0.96) and showed good reliability ({\alpha}=0.88). TAIGHA and TAIGHA-S are validated instruments for assessing user trust and distrust in AI-generated health advice. Reporting trust and distrust separately permits a more complete evaluation of AI interventions, and the short scale is well-suited for time-constrained settings. 

---
# SPARQL-LLM: Real-Time SPARQL Query Generation from Natural Language Questions 

**Authors**: Panayiotis Smeros, Vincent Emonet, Ruijie Wang, Ana-Claudia Sima, Tarcisio Mendes de Farias  

**Link**: [PDF](https://arxiv.org/pdf/2512.14277)  

**Abstract**: The advent of large language models is contributing to the emergence of novel approaches that promise to better tackle the challenge of generating structured queries, such as SPARQL queries, from natural language. However, these new approaches mostly focus on response accuracy over a single source while ignoring other evaluation criteria, such as federated query capability over distributed data stores, as well as runtime and cost to generate SPARQL queries. Consequently, they are often not production-ready or easy to deploy over (potentially federated) knowledge graphs with good accuracy. To mitigate these issues, in this paper, we extend our previous work and describe and systematically evaluate SPARQL-LLM, an open-source and triplestore-agnostic approach, powered by lightweight metadata, that generates SPARQL queries from natural language text. First, we describe its architecture, which consists of dedicated components for metadata indexing, prompt building, and query generation and execution. Then, we evaluate it based on a state-of-the-art challenge with multilingual questions, and a collection of questions from three of the most prevalent knowledge graphs within the field of bioinformatics. Our results demonstrate a substantial increase of 24% in the F1 Score on the state-of-the-art challenge, adaptability to high-resource languages such as English and Spanish, as well as ability to form complex and federated bioinformatics queries. Furthermore, we show that SPARQL-LLM is up to 36x faster than other systems participating in the challenge, while costing a maximum of $0.01 per question, making it suitable for real-time, low-cost text-to-SPARQL applications. One such application deployed over real-world decentralized knowledge graphs can be found at this https URL. 

---
# Explainable Preference Learning: a Decision Tree-based Surrogate Model for Preferential Bayesian Optimization 

**Authors**: Nick Leenders, Thomas Quadt, Boris Cule, Roy Lindelauf, Herman Monsuur, Joost van Oijen, Mark Voskuijl  

**Link**: [PDF](https://arxiv.org/pdf/2512.14263)  

**Abstract**: Current Preferential Bayesian Optimization methods rely on Gaussian Processes (GPs) as surrogate models. These models are hard to interpret, struggle with handling categorical data, and are computationally complex, limiting their real-world usability. In this paper, we introduce an inherently interpretable decision tree-based surrogate model capable of handling both categorical and continuous data, and scalable to large datasets. Extensive numerical experiments on eight increasingly spiky optimization functions show that our model outperforms GP-based alternatives on spiky functions and has only marginally lower performance for non-spiky functions. Moreover, we apply our model to the real-world Sushi dataset and show its ability to learn an individual's sushi preferences. Finally, we show some initial work on using historical preference data to speed up the optimization process for new unseen users. 

---
# From Context to EDUs: Faithful and Structured Context Compression via Elementary Discourse Unit Decomposition 

**Authors**: Yiqing Zhou, Yu Lei, Shuzheng Si, Qingyan Sun, Wei Wang, Yifei Wu, Hao Wen, Gang Chen, Fanchao Qi, Maosong Sun  

**Link**: [PDF](https://arxiv.org/pdf/2512.14244)  

**Abstract**: Managing extensive context remains a critical bottleneck for Large Language Models (LLMs), particularly in applications like long-document question answering and autonomous agents where lengthy inputs incur high computational costs and introduce noise. Existing compression techniques often disrupt local coherence through discrete token removal or rely on implicit latent encoding that suffers from positional bias and incompatibility with closed-source APIs. To address these limitations, we introduce the EDU-based Context Compressor, a novel explicit compression framework designed to preserve both global structure and fine-grained details. Our approach reformulates context compression as a structure-then-select process. First, our LingoEDU transforms linear text into a structural relation tree of Elementary Discourse Units (EDUs) which are anchored strictly to source indices to eliminate hallucination. Second, a lightweight ranking module selects query-relevant sub-trees for linearization. To rigorously evaluate structural understanding, we release StructBench, a manually annotated dataset of 248 diverse documents. Empirical results demonstrate that our method achieves state-of-the-art structural prediction accuracy and significantly outperforms frontier LLMs while reducing costs. Furthermore, our structure-aware compression substantially enhances performance across downstream tasks ranging from long-context tasks to complex Deep Search scenarios. 

---
# Beyond MMD: Evaluating Graph Generative Models with Geometric Deep Learning 

**Authors**: Salvatore Romano, Marco Grassia, Giuseppe Mangioni  

**Link**: [PDF](https://arxiv.org/pdf/2512.14241)  

**Abstract**: Graph generation is a crucial task in many fields, including network science and bioinformatics, as it enables the creation of synthetic graphs that mimic the properties of real-world networks for various applications. Graph Generative Models (GGMs) have emerged as a promising solution to this problem, leveraging deep learning techniques to learn the underlying distribution of real-world graphs and generate new samples that closely resemble them. Examples include approaches based on Variational Auto-Encoders, Recurrent Neural Networks, and more recently, diffusion-based models. However, the main limitation often lies in the evaluation process, which typically relies on Maximum Mean Discrepancy (MMD) as a metric to assess the distribution of graph properties in the generated ensemble. This paper introduces a novel methodology for evaluating GGMs that overcomes the limitations of MMD, which we call RGM (Representation-aware Graph-generation Model evaluation). As a practical demonstration of our methodology, we present a comprehensive evaluation of two state-of-the-art Graph Generative Models: Graph Recurrent Attention Networks (GRAN) and Efficient and Degree-guided graph GEnerative model (EDGE). We investigate their performance in generating realistic graphs and compare them using a Geometric Deep Learning model trained on a custom dataset of synthetic and real-world graphs, specifically designed for graph classification tasks. Our findings reveal that while both models can generate graphs with certain topological properties, they exhibit significant limitations in preserving the structural characteristics that distinguish different graph domains. We also highlight the inadequacy of Maximum Mean Discrepancy as an evaluation metric for GGMs and suggest alternative approaches for future research. 

---
# PentestEval: Benchmarking LLM-based Penetration Testing with Modular and Stage-Level Design 

**Authors**: Ruozhao Yang, Mingfei Cheng, Gelei Deng, Tianwei Zhang, Junjie Wang, Xiaofei Xie  

**Link**: [PDF](https://arxiv.org/pdf/2512.14233)  

**Abstract**: Penetration testing is essential for assessing and strengthening system security against real-world threats, yet traditional workflows remain highly manual, expertise-intensive, and difficult to scale. Although recent advances in Large Language Models (LLMs) offer promising opportunities for automation, existing applications rely on simplistic prompting without task decomposition or domain adaptation, resulting in unreliable black-box behavior and limited insight into model capabilities across penetration testing stages. To address this gap, we introduce PentestEval, the first comprehensive benchmark for evaluating LLMs across six decomposed penetration testing stages: Information Collection, Weakness Gathering and Filtering, Attack Decision-Making, Exploit Generation and Revision. PentestEval integrates expert-annotated ground truth with a fully automated evaluation pipeline across 346 tasks covering all stages in 12 realistic vulnerable scenarios. Our stage-level evaluation of 9 widely used LLMs reveals generally weak performance and distinct limitations across the stages of penetration-testing workflow. End-to-end pipelines reach only 31% success rate, and existing LLM-powered systems such as PentestGPT, PentestAgent, and VulnBot exhibit similar limitations, with autonomous agents failing almost entirely. These findings highlight that autonomous penetration testing demands stronger structured reasoning, where modularization enhances each individual stage and improves overall performance. PentestEval provides the foundational benchmark needed for future research on fine-grained, stage-level evaluation, paving the way toward more reliable LLM-based automation. 

---
# Estimating problem difficulty without ground truth using Large Language Model comparisons 

**Authors**: Marthe Ballon, Andres Algaba, Brecht Verbeken, Vincent Ginis  

**Link**: [PDF](https://arxiv.org/pdf/2512.14220)  

**Abstract**: Recent advances in the finetuning of large language models (LLMs) have significantly improved their performance on established benchmarks, emphasizing the need for increasingly difficult, synthetic data. A key step in this data generation pipeline is a method for estimating problem difficulty. Current approaches, such as human calibration or performance-based scoring, fail to generalize to out-of-distribution problems, i.e. problems currently unsolvable by humans and LLMs, because they are not scalable, time-consuming, and ground truth dependent. Therefore, we propose a new method for estimating problem difficulty, LLM compare, that addresses these limitations. An LLM performs pairwise difficulty comparisons, and then Bradley-Terry scores are computed based on the outcomes. To validate our method, we first propose a conceptual framework that positions existing approaches on three orthogonal planes--construction, scale and dependence--identifying which quadrants a measure needs to occupy to score out-of-distribution problems. LLM compare naturally occupies all desirable quadrants as the first measure that is continuous and dynamic, model-agnostic and independent of ground truth information. As a second validation, we show that LLM compare demonstrates strong alignment with human annotations: Pearson $r \geq 0.80$ for $n=1876$. Thirdly, we show that LLM compare is robust to hallucinations, with less than $6\%$ degradation in Pearson correlation for $10\%$ noise injection. Our work represents a significant step towards replacing time-consuming human annotations and synthetic data generation, and will be an important driver for curriculum design, model evaluation, and AI-assisted research ideation. 

---
# Error Bound Analysis of Physics-Informed Neural Networks-Driven T2 Quantification in Cardiac Magnetic Resonance Imaging 

**Authors**: Mengxue Zhang, Qingrui Cai, Yinyin Chen, Hang Jin, Jianjun Zhou, Qiu Guo, Peijun Zhao, Zhiping Mao, Xingxing Zhang, Yuyu Xia, Xianwang Jiang, Qin Xu, Chunyan Xiong, Yirong Zhou, Chengyan Wang, Xiaobo Qu  

**Link**: [PDF](https://arxiv.org/pdf/2512.14211)  

**Abstract**: Physics-Informed Neural Networks (PINN) are emerging as a promising approach for quantitative parameter estimation of Magnetic Resonance Imaging (MRI). While existing deep learning methods can provide an accurate quantitative estimation of the T2 parameter, they still require large amounts of training data and lack theoretical support and a recognized gold standard. Thus, given the absence of PINN-based approaches for T2 estimation, we propose embedding the fundamental physics of MRI, the Bloch equation, in the loss of PINN, which is solely based on target scan data and does not require a pre-defined training database. Furthermore, by deriving rigorous upper bounds for both the T2 estimation error and the generalization error of the Bloch equation solution, we establish a theoretical foundation for evaluating the PINN's quantitative accuracy. Even without access to the ground truth or a gold standard, this theory enables us to estimate the error with respect to the real quantitative parameter T2. The accuracy of T2 mapping and the validity of the theoretical analysis are demonstrated on a numerical cardiac model and a water phantom, where our method exhibits excellent quantitative precision in the myocardial T2 range. Clinical applicability is confirmed in 94 acute myocardial infarction (AMI) patients, achieving low-error quantitative T2 estimation under the theoretical error bound, highlighting the robustness and potential of PINN. 

---
# Understanding and Improving Hyperbolic Deep Reinforcement Learning 

**Authors**: Timo Klein, Thomas Lang, Andrii Shkabrii, Alexander Sturm, Kevin Sidak, Lukas Miklautz, Claudia Plant, Yllka Velaj, Sebastian Tschiatschek  

**Link**: [PDF](https://arxiv.org/pdf/2512.14202)  

**Abstract**: The performance of reinforcement learning (RL) agents depends critically on the quality of the underlying feature representations. Hyperbolic feature spaces are well-suited for this purpose, as they naturally capture hierarchical and relational structure often present in complex RL environments. However, leveraging these spaces commonly faces optimization challenges due to the nonstationarity of RL. In this work, we identify key factors that determine the success and failure of training hyperbolic deep RL agents. By analyzing the gradients of core operations in the Poincaré Ball and Hyperboloid models of hyperbolic geometry, we show that large-norm embeddings destabilize gradient-based training, leading to trust-region violations in proximal policy optimization (PPO). Based on these insights, we introduce Hyper++, a new hyperbolic PPO agent that consists of three components: (i) stable critic training through a categorical value loss instead of regression; (ii) feature regularization guaranteeing bounded norms while avoiding the curse of dimensionality from clipping; and (iii) using a more optimization-friendly formulation of hyperbolic network layers. In experiments on ProcGen, we show that Hyper++ guarantees stable learning, outperforms prior hyperbolic agents, and reduces wall-clock time by approximately 30%. On Atari-5 with Double DQN, Hyper++ strongly outperforms Euclidean and hyperbolic baselines. We release our code at this https URL . 

---
# End-to-End Learning-based Video Streaming Enhancement Pipeline: A Generative AI Approach 

**Authors**: Emanuele Artioli, Farzad Tashtarian, Christian Timmerer  

**Link**: [PDF](https://arxiv.org/pdf/2512.14185)  

**Abstract**: The primary challenge of video streaming is to balance high video quality with smooth playback. Traditional codecs are well tuned for this trade-off, yet their inability to use context means they must encode the entire video data and transmit it to the client. This paper introduces ELVIS (End-to-end Learning-based VIdeo Streaming Enhancement Pipeline), an end-to-end architecture that combines server-side encoding optimizations with client-side generative in-painting to remove and reconstruct redundant video data. Its modular design allows ELVIS to integrate different codecs, inpainting models, and quality metrics, making it adaptable to future innovations. Our results show that current technologies achieve improvements of up to 11 VMAF points over baseline benchmarks, though challenges remain for real-time applications due to computational demands. ELVIS represents a foundational step toward incorporating generative AI into video streaming pipelines, enabling higher quality experiences without increased bandwidth requirements. 

---
# Towards Explainable Quantum AI: Informing the Encoder Selection of Quantum Neural Networks via Visualization 

**Authors**: Shaolun Ruan, Feng Liang, Rohan Ramakrishna, Chao Ren, Rudai Yan, Qiang Guan, Jiannan Li, Yong Wang  

**Link**: [PDF](https://arxiv.org/pdf/2512.14181)  

**Abstract**: Quantum Neural Networks (QNNs) represent a promising fusion of quantum computing and neural network architectures, offering speed-ups and efficient processing of high-dimensional, entangled data. A crucial component of QNNs is the encoder, which maps classical input data into quantum states. However, choosing suitable encoders remains a significant challenge, largely due to the lack of systematic guidance and the trial-and-error nature of current approaches. This process is further impeded by two key challenges: (1) the difficulty in evaluating encoded quantum states prior to training, and (2) the lack of intuitive methods for analyzing an encoder's ability to effectively distinguish data features. To address these issues, we introduce a novel visualization tool, XQAI-Eyes, which enables QNN developers to compare classical data features with their corresponding encoded quantum states and to examine the mixed quantum states across different classes. By bridging classical and quantum perspectives, XQAI-Eyes facilitates a deeper understanding of how encoders influence QNN performance. Evaluations across diverse datasets and encoder designs demonstrate XQAI-Eyes's potential to support the exploration of the relationship between encoder design and QNN effectiveness, offering a holistic and transparent approach to optimizing quantum encoders. Moreover, domain experts used XQAI-Eyes to derive two key practices for quantum encoder selection, grounded in the principles of pattern preservation and feature mapping. 

---
# A Comparative Analysis of Retrieval-Augmented Generation Techniques for Bengali Standard-to-Dialect Machine Translation Using LLMs 

**Authors**: K. M. Jubair Sami, Dipto Sumit, Ariyan Hossain, Farig Sadeque  

**Link**: [PDF](https://arxiv.org/pdf/2512.14179)  

**Abstract**: Translating from a standard language to its regional dialects is a significant NLP challenge due to scarce data and linguistic variation, a problem prominent in the Bengali language. This paper proposes and compares two novel RAG pipelines for standard-to-dialectal Bengali translation. The first, a Transcript-Based Pipeline, uses large dialect sentence contexts from audio transcripts. The second, a more effective Standardized Sentence-Pairs Pipeline, utilizes structured local\_dialect:standard\_bengali sentence pairs. We evaluated both pipelines across six Bengali dialects and multiple LLMs using BLEU, ChrF, WER, and BERTScore. Our findings show that the sentence-pair pipeline consistently outperforms the transcript-based one, reducing Word Error Rate (WER) from 76\% to 55\% for the Chittagong dialect. Critically, this RAG approach enables smaller models (e.g., Llama-3.1-8B) to outperform much larger models (e.g., GPT-OSS-120B), demonstrating that a well-designed retrieval strategy can be more crucial than model size. This work contributes an effective, fine-tuning-free solution for low-resource dialect translation, offering a practical blueprint for preserving linguistic diversity. 

---
# IntentMiner: Intent Inversion Attack via Tool Call Analysis in the Model Context Protocol 

**Authors**: Yunhao Yao, Zhiqiang Wang, Haoran Cheng, Yihang Cheng, Haohua Du, Xiang-Yang Li  

**Link**: [PDF](https://arxiv.org/pdf/2512.14166)  

**Abstract**: The rapid evolution of Large Language Models (LLMs) into autonomous agents has led to the adoption of the Model Context Protocol (MCP) as a standard for discovering and invoking external tools. While this architecture decouples the reasoning engine from tool execution to enhance scalability, it introduces a significant privacy surface: third-party MCP servers, acting as semi-honest intermediaries, can observe detailed tool interaction logs outside the user's trusted boundary. In this paper, we first identify and formalize a novel privacy threat termed Intent Inversion, where a semi-honest MCP server attempts to reconstruct the user's private underlying intent solely by analyzing legitimate tool calls. To systematically assess this vulnerability, we propose IntentMiner, a framework that leverages Hierarchical Information Isolation and Three-Dimensional Semantic Analysis, integrating tool purpose, call statements, and returned results, to accurately infer user intent at the step level. Extensive experiments demonstrate that IntentMiner achieves a high degree of semantic alignment (over 85%) with original user queries, significantly outperforming baseline approaches. These results highlight the inherent privacy risks in decoupled agent architectures, revealing that seemingly benign tool execution logs can serve as a potent vector for exposing user secrets. 

---
# PathFinder: Advancing Path Loss Prediction for Single-to-Multi-Transmitter Scenario 

**Authors**: Zhijie Zhong, Zhiwen Yu, Pengyu Li, Jianming Lv, C. L. Philip Chen, Min Chen  

**Link**: [PDF](https://arxiv.org/pdf/2512.14150)  

**Abstract**: Radio path loss prediction (RPP) is critical for optimizing 5G networks and enabling IoT, smart city, and similar applications. However, current deep learning-based RPP methods lack proactive environmental modeling, struggle with realistic multi-transmitter scenarios, and generalize poorly under distribution shifts, particularly when training/testing environments differ in building density or transmitter configurations. This paper identifies three key issues: (1) passive environmental modeling that overlooks transmitters and key environmental features; (2) overemphasis on single-transmitter scenarios despite real-world multi-transmitter prevalence; (3) excessive focus on in-distribution performance while neglecting distribution shift challenges. To address these, we propose PathFinder, a novel architecture that actively models buildings and transmitters via disentangled feature encoding and integrates Mask-Guided Low-rank Attention to independently focus on receiver and building regions. We also introduce a Transmitter-Oriented Mixup strategy for robust training and a new benchmark, single-to-multi-transmitter RPP (S2MT-RPP), tailored to evaluate extrapolation performance (multi-transmitter testing after single-transmitter training). Experimental results show PathFinder outperforms state-of-the-art methods significantly, especially in challenging multi-transmitter scenarios. Our code and project site are available at: this https URL. 

---
# TorchTraceAP: A New Benchmark Dataset for Detecting Performance Anti-Patterns in Computer Vision Models 

**Authors**: Hanning Chen, Keyu Man, Kevin Zhu, Chenguang Zhu, Haonan Li, Tongbo Luo, Xizhou Feng, Wei Sun, Sreen Tallam, Mohsen Imani, Partha Kanuparthy  

**Link**: [PDF](https://arxiv.org/pdf/2512.14141)  

**Abstract**: Identifying and addressing performance anti-patterns in machine learning (ML) models is critical for efficient training and inference, but it typically demands deep expertise spanning system infrastructure, ML models and kernel development. While large tech companies rely on dedicated ML infrastructure engineers to analyze torch traces and benchmarks, such resource-intensive workflows are largely inaccessible to computer vision researchers in general. Among the challenges, pinpointing problematic trace segments within lengthy execution traces remains the most time-consuming task, and is difficult to automate with current ML models, including LLMs. In this work, we present the first benchmark dataset specifically designed to evaluate and improve ML models' ability to detect anti patterns in traces. Our dataset contains over 600 PyTorch traces from diverse computer vision models classification, detection, segmentation, and generation collected across multiple hardware platforms. We also propose a novel iterative approach: a lightweight ML model first detects trace segments with anti patterns, followed by a large language model (LLM) for fine grained classification and targeted feedback. Experimental results demonstrate that our method significantly outperforms unsupervised clustering and rule based statistical techniques for detecting anti pattern regions. Our method also effectively compensates LLM's limited context length and reasoning inefficiencies. 

---
# LAPPI: Interactive Optimization with LLM-Assisted Preference-Based Problem Instantiation 

**Authors**: So Kuroki, Manami Nakagawa, Shigeo Yoshida, Yuki Koyama, Kozuno Tadashi  

**Link**: [PDF](https://arxiv.org/pdf/2512.14138)  

**Abstract**: Many real-world tasks, such as trip planning or meal planning, can be formulated as combinatorial optimization problems. However, using optimization solvers is difficult for end users because it requires problem instantiation: defining candidate items, assigning preference scores, and specifying constraints. We introduce LAPPI (LLM-Assisted Preference-based Problem Instantiation), an interactive approach that uses large language models (LLMs) to support users in this instantiation process. Through natural language conversations, the system helps users transform vague preferences into well-defined optimization problems. These instantiated problems are then passed to existing optimization solvers to generate solutions. In a user study on trip planning, our method successfully captured user preferences and generated feasible plans that outperformed both conventional and prompt-engineering approaches. We further demonstrate LAPPI's versatility by adapting it to an additional use case. 

---
# UIXPOSE: Mobile Malware Detection via Intention-Behaviour Discrepancy Analysis 

**Authors**: Amirmohammad Pasdar, Toby Murray, Van-Thuan Pham  

**Link**: [PDF](https://arxiv.org/pdf/2512.14130)  

**Abstract**: We introduce UIXPOSE, a source-code-agnostic framework that operates on both compiled and open-source apps. This framework applies Intention Behaviour Alignment (IBA) to mobile malware analysis, aligning UI-inferred intent with runtime semantics. Previous work either infers intent statically, e.g., permission-centric, or widget-level or monitors coarse dynamic signals (endpoints, partial resource usage) that miss content and context. UIXPOSE infers an intent vector from each screen using vision-language models and knowledge structures and combines decoded network payloads, heap/memory signals, and resource utilisation traces into a behaviour vector. Their alignment, calculated at runtime, can both detect misbehaviour and highlight exploration of behaviourally rich paths. In three real-world case studies, UIXPOSE reveals covert exfiltration and hidden background activity that evade metadata-only baselines, demonstrating how IBA improves dynamic detection. 

---
# SportsGPT: An LLM-driven Framework for Interpretable Sports Motion Assessment and Training Guidance 

**Authors**: Wenbo Tian, Ruting Lin, Hongxian Zheng, Yaodong Yang, Geng Wu, Zihao Zhang, Zhang Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2512.14121)  

**Abstract**: Existing intelligent sports analysis systems mainly focus on "scoring and visualization," often lacking automatic performance diagnosis and interpretable training guidance. Recent advances of Large Language Models (LMMs) and motion analysis techniques provide new opportunities to address the above limitations. In this paper, we propose SportsGPT, an LLM-driven framework for interpretable sports motion assessment and training guidance, which establishes a closed loop from motion time-series input to professional training guidance. First, given a set of high-quality target models, we introduce MotionDTW, a two-stage time series alignment algorithm designed for accurate keyframe extraction from skeleton-based motion sequences. Subsequently, we design a Knowledge-based Interpretable Sports Motion Assessment Model (KISMAM) to obtain a set of interpretable assessment metrics (e.g., insufficient extension) by constrasting the keyframes with the targe models. Finally, we propose SportsRAG, a RAG-based training guidance model based on Qwen3. Leveraging a 6B-token knowledge base, it prompts the LLM to generate professional training guidance by retrieving domain-specific QA pairs. Experimental results demonstrate that MotionDTW significantly outperforms traditional methods with lower temporal error and higher IoU scores. Furthermore, ablation studies validate the KISMAM and SportsRAG, confirming that SportsGPT surpasses general LLMs in diagnostic accuracy and professionalism. 

---
# Neurosymbolic Inference On Foundation Models For Remote Sensing Text-to-image Retrieval With Complex Queries 

**Authors**: Emanuele Mezzi, Gertjan Burghouts, Maarten Kruithof  

**Link**: [PDF](https://arxiv.org/pdf/2512.14102)  

**Abstract**: Text-to-image retrieval in remote sensing (RS) has advanced rapidly with the rise of large vision-language models (LVLMs) tailored for aerial and satellite imagery, culminating in remote sensing large vision-language models (RS-LVLMS). However, limited explainability and poor handling of complex spatial relations remain key challenges for real-world use. To address these issues, we introduce RUNE (Reasoning Using Neurosymbolic Entities), an approach that combines Large Language Models (LLMs) with neurosymbolic AI to retrieve images by reasoning over the compatibility between detected entities and First-Order Logic (FOL) expressions derived from text queries. Unlike RS-LVLMs that rely on implicit joint embeddings, RUNE performs explicit reasoning, enhancing performance and interpretability. For scalability, we propose a logic decomposition strategy that operates on conditioned subsets of detected entities, guaranteeing shorter execution time compared to neural approaches. Rather than using foundation models for end-to-end retrieval, we leverage them only to generate FOL expressions, delegating reasoning to a neurosymbolic inference module. For evaluation we repurpose the DOTA dataset, originally designed for object detection, by augmenting it with more complex queries than in existing benchmarks. We show the LLM's effectiveness in text-to-logic translation and compare RUNE with state-of-the-art RS-LVLMs, demonstrating superior performance. We introduce two metrics, Retrieval Robustness to Query Complexity (RRQC) and Retrieval Robustness to Image Uncertainty (RRIU), which evaluate performance relative to query complexity and image uncertainty. RUNE outperforms joint-embedding models in complex RS retrieval tasks, offering gains in performance, robustness, and explainability. We show RUNE's potential for real-world RS applications through a use case on post-flood satellite image retrieval. 

---
# ProtoFlow: Interpretable and Robust Surgical Workflow Modeling with Learned Dynamic Scene Graph Prototypes 

**Authors**: Felix Holm, Ghazal Ghazaei, Nassir Navab  

**Link**: [PDF](https://arxiv.org/pdf/2512.14092)  

**Abstract**: Purpose: Detailed surgical recognition is critical for advancing AI-assisted surgery, yet progress is hampered by high annotation costs, data scarcity, and a lack of interpretable models. While scene graphs offer a structured abstraction of surgical events, their full potential remains untapped. In this work, we introduce ProtoFlow, a novel framework that learns dynamic scene graph prototypes to model complex surgical workflows in an interpretable and robust manner.
Methods: ProtoFlow leverages a graph neural network (GNN) encoder-decoder architecture that combines self-supervised pretraining for rich representation learning with a prototype-based fine-tuning stage. This process discovers and refines core prototypes that encapsulate recurring, clinically meaningful patterns of surgical interaction, forming an explainable foundation for workflow analysis.
Results: We evaluate our approach on the fine-grained CAT-SG dataset. ProtoFlow not only outperforms standard GNN baselines in overall accuracy but also demonstrates exceptional robustness in limited-data, few-shot scenarios, maintaining strong performance when trained on as few as one surgical video. Our qualitative analyses further show that the learned prototypes successfully identify distinct surgical sub-techniques and provide clear, interpretable insights into workflow deviations and rare complications.
Conclusion: By uniting robust representation learning with inherent explainability, ProtoFlow represents a significant step toward developing more transparent, reliable, and data-efficient AI systems, accelerating their potential for clinical adoption in surgical training, real-time decision support, and workflow optimization. 

---
# Arithmetic-Intensity-Aware Quantization 

**Authors**: Taig Singh, Shreshth Rajan, Nikhil Iyer  

**Link**: [PDF](https://arxiv.org/pdf/2512.14090)  

**Abstract**: As modern neural networks become increasingly memory-bound, inference throughput is limited by DRAM bandwidth rather than compute. We present Arithmetic-Intensity-Aware Quantization (AIQ), a mixed precision quantization framework that chooses per-layer bit-widths to maximize arithmetic intensity (AI) while minimizing accuracy loss. AIQ is a post-training quantization method that uses search algorithms over per-layer quantization schemes to minimize a weighted loss over AI and accuracy. On ResNet-20/CIFAR-10, AIQ increases AI by ~50% over an FP32 baseline while keeping test accuracy within ~1 percentage point, and outperforming global uniform quantization schemes. On a memory-bound MobileNetV2 architecture, AIQ configurations give a 1.66x higher throughput than the FP32 baseline while keeping test accuracy within 1 percentage point. We also find that AIQ naturally quantizes larger layers more aggressively. 

---
# SonicMoE: Accelerating MoE with IO and Tile-aware Optimizations 

**Authors**: Wentao Guo, Mayank Mishra, Xinle Cheng, Ion Stoica, Tri Dao  

**Link**: [PDF](https://arxiv.org/pdf/2512.14080)  

**Abstract**: Mixture of Experts (MoE) models have emerged as the de facto architecture for scaling up language models without significantly increasing the computational cost. Recent MoE models demonstrate a clear trend towards high expert granularity (smaller expert intermediate dimension) and higher sparsity (constant number of activated experts with higher number of total experts), which improve model quality per FLOP. However, fine-grained MoEs suffer from increased activation memory footprint and reduced hardware efficiency due to higher IO costs, while sparser MoEs suffer from wasted computations due to padding in Grouped GEMM kernels. In response, we propose a memory-efficient algorithm to compute the forward and backward passes of MoEs with minimal activation caching for the backward pass. We also design GPU kernels that overlap memory IO with computation benefiting all MoE architectures. Finally, we propose a novel "token rounding" method that minimizes the wasted compute due to padding in Grouped GEMM kernels. As a result, our method SonicMoE reduces activation memory by 45% and achieves a 1.86x compute throughput improvement on Hopper GPUs compared to ScatterMoE's BF16 MoE kernel for a fine-grained 7B MoE. Concretely, SonicMoE on 64 H100s achieves a training throughput of 213 billion tokens per day comparable to ScatterMoE's 225 billion tokens per day on 96 H100s for a 7B MoE model training with FSDP-2 using the lm-engine codebase. Under high MoE sparsity settings, our tile-aware token rounding algorithm yields an additional 1.16x speedup on kernel execution time compared to vanilla top-$K$ routing while maintaining similar downstream performance. We open-source all our kernels to enable faster MoE model training. 

---
# SDAR-VL: Stable and Efficient Block-wise Diffusion for Vision-Language Understanding 

**Authors**: Shuang Cheng, Yuhua Jiang, Zineng Zhou, Dawei Liu, Wang Tao, Linfeng Zhang, Biqing Qi, Bowen Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2512.14068)  

**Abstract**: Block-wise discrete diffusion offers an attractive balance between parallel generation and causal dependency modeling, making it a promising backbone for vision-language modeling. However, its practical adoption has been limited by high training cost, slow convergence, and instability, which have so far kept it behind strong autoregressive (AR) baselines. We present \textbf{SDAR-VL}, the first systematic application of block-wise discrete diffusion to large-scale vision-language understanding (VLU), together with an \emph{integrated framework for efficient and stable training}. This framework unifies three components: (1) \textbf{Asynchronous Block-wise Noise Scheduling} to diversify supervision within each batch; (2) \textbf{Effective Mask Ratio Scaling} for unbiased loss normalization under stochastic masking; and (3) a \textbf{Progressive Beta Noise Curriculum} that increases effective mask coverage while preserving corruption diversity. Experiments on 21 single-image, multi-image, and video benchmarks show that SDAR-VL consistently improves \emph{training efficiency}, \emph{convergence stability}, and \emph{task performance} over conventional block diffusion. On this evaluation suite, SDAR-VL sets a new state of the art among diffusion-based vision-language models and, under matched settings, matches or surpasses strong AR baselines such as LLaVA-OneVision as well as the global diffusion baseline LLaDA-V, establishing block-wise diffusion as a practical backbone for VLU. 

---
# Efficient-DLM: From Autoregressive to Diffusion Language Models, and Beyond in Speed 

**Authors**: Yonggan Fu, Lexington Whalen, Zhifan Ye, Xin Dong, Shizhe Diao, Jingyu Liu, Chengyue Wu, Hao Zhang, Enze Xie, Song Han, Maksim Khadkevich, Jan Kautz, Yingyan Celine Lin, Pavlo Molchanov  

**Link**: [PDF](https://arxiv.org/pdf/2512.14067)  

**Abstract**: Diffusion language models (dLMs) have emerged as a promising paradigm that enables parallel, non-autoregressive generation, but their learning efficiency lags behind that of autoregressive (AR) language models when trained from scratch. To this end, we study AR-to-dLM conversion to transform pretrained AR models into efficient dLMs that excel in speed while preserving AR models' task accuracy. We achieve this by identifying limitations in the attention patterns and objectives of existing AR-to-dLM methods and then proposing principles and methodologies for more effective AR-to-dLM conversion. Specifically, we first systematically compare different attention patterns and find that maintaining pretrained AR weight distributions is critical for effective AR-to-dLM conversion. As such, we introduce a continuous pretraining scheme with a block-wise attention pattern, which remains causal across blocks while enabling bidirectional modeling within each block. We find that this approach can better preserve pretrained AR models' weight distributions than fully bidirectional modeling, in addition to its known benefit of enabling KV caching, and leads to a win-win in accuracy and efficiency. Second, to mitigate the training-test gap in mask token distributions (uniform vs. highly left-to-right), we propose a position-dependent token masking strategy that assigns higher masking probabilities to later tokens during training to better mimic test-time behavior. Leveraging this framework, we conduct extensive studies of dLMs' attention patterns, training dynamics, and other design choices, providing actionable insights into scalable AR-to-dLM conversion. These studies lead to the Efficient-DLM family, which outperforms state-of-the-art AR models and dLMs, e.g., our Efficient-DLM 8B achieves +5.4%/+2.7% higher accuracy with 4.5x/2.7x higher throughput compared to Dream 7B and Qwen3 4B, respectively. 

---
# Real-time prediction of workplane illuminance distribution for daylight-linked controls using non-intrusive multimodal deep learning 

**Authors**: Zulin Zhuang, Yu Bian  

**Link**: [PDF](https://arxiv.org/pdf/2512.14058)  

**Abstract**: Daylight-linked controls (DLCs) have significant potential for energy savings in buildings, especially when abundant daylight is available and indoor workplane illuminance can be accurately predicted in real time. Most existing studies on indoor daylight predictions were developed and tested for static scenes. This study proposes a multimodal deep learning framework that predicts indoor workplane illuminance distributions in real time from non-intrusive images with temporal-spatial features. By extracting image features only from the side-lit window areas rather than interior pixels, the approach remains applicable in dynamically occupied indoor spaces. A field experiment was conducted in a test room in Guangzhou (China), where 17,344 samples were collected for model training and validation. The model achieved R2 > 0.98 with RMSE < 0.14 on the same-distribution test set and R2 > 0.82 with RMSE < 0.17 on an unseen-day test set, indicating high accuracy and acceptable temporal generalization. 

---
# FacEDiT: Unified Talking Face Editing and Generation via Facial Motion Infilling 

**Authors**: Kim Sung-Bin, Joohyun Chang, David Harwath, Tae-Hyun Oh  

**Link**: [PDF](https://arxiv.org/pdf/2512.14056)  

**Abstract**: Talking face editing and face generation have often been studied as distinct problems. In this work, we propose viewing both not as separate tasks but as subtasks of a unifying formulation, speech-conditional facial motion infilling. We explore facial motion infilling as a self-supervised pretext task that also serves as a unifying formulation of dynamic talking face synthesis. To instantiate this idea, we propose FacEDiT, a speech-conditional Diffusion Transformer trained with flow matching. Inspired by masked autoencoders, FacEDiT learns to synthesize masked facial motions conditioned on surrounding motions and speech. This formulation enables both localized generation and edits, such as substitution, insertion, and deletion, while ensuring seamless transitions with unedited regions. In addition, biased attention and temporal smoothness constraints enhance boundary continuity and lip synchronization. To address the lack of a standard editing benchmark, we introduce FacEDiTBench, the first dataset for talking face editing, featuring diverse edit types and lengths, along with new evaluation metrics. Extensive experiments validate that talking face editing and generation emerge as subtasks of speech-conditional motion infilling; FacEDiT produces accurate, speech-aligned facial edits with strong identity preservation and smooth visual continuity while generalizing effectively to talking face generation. 

---
# OmniDrive-R1: Reinforcement-driven Interleaved Multi-modal Chain-of-Thought for Trustworthy Vision-Language Autonomous Driving 

**Authors**: Zhenguo Zhang, Haohan Zhen, Yishen Wang, Le Xu, Tianchen Deng, Xuefeng Chen, Qu Chen, Bo Zhang, Wuxiong Huang  

**Link**: [PDF](https://arxiv.org/pdf/2512.14044)  

**Abstract**: The deployment of Vision-Language Models (VLMs) in safety-critical domains like autonomous driving (AD) is critically hindered by reliability failures, most notably object hallucination. This failure stems from their reliance on ungrounded, text-based Chain-of-Thought (CoT) this http URL existing multi-modal CoT approaches attempt mitigation, they suffer from two fundamental flaws: (1) decoupled perception and reasoning stages that prevent end-to-end joint optimization, and (2) reliance on expensive, dense localization this http URL we introduce OmniDrive-R1, an end-to-end VLM framework designed for autonomous driving, which unifies perception and reasoning through an interleaved Multi-modal Chain-of-Thought (iMCoT) mechanism. Our core innovation is an Reinforcement-driven visual grounding capability, enabling the model to autonomously direct its attention and "zoom in" on critical regions for fine-grained analysis. This capability is enabled by our pure two-stage reinforcement learning training pipeline and Clip-GRPO algorithm. Crucially, Clip-GRPO introduces an annotation-free, process-based grounding reward. This reward not only eliminates the need for dense labels but also circumvents the instability of external tool calls by enforcing real-time cross-modal consistency between the visual focus and the textual reasoning. Extensive experiments on DriveLMM-o1 demonstrate our model's significant improvements. Compared to the baseline Qwen2.5VL-7B, OmniDrive-R1 improves the overall reasoning score from 51.77% to 80.35%, and the final answer accuracy from 37.81% to 73.62%. 

---
# ACE-SLAM: Scene Coordinate Regression for Neural Implicit Real-Time SLAM 

**Authors**: Ignacio Alzugaray, Marwan Taher, Andrew J. Davison  

**Link**: [PDF](https://arxiv.org/pdf/2512.14032)  

**Abstract**: We present a novel neural RGB-D Simultaneous Localization And Mapping (SLAM) system that learns an implicit map of the scene in real time. For the first time, we explore the use of Scene Coordinate Regression (SCR) as the core implicit map representation in a neural SLAM pipeline, a paradigm that trains a lightweight network to directly map 2D image features to 3D global coordinates. SCR networks provide efficient, low-memory 3D map representations, enable extremely fast relocalization, and inherently preserve privacy, making them particularly suitable for neural implicit SLAM.
Our system is the first one to achieve strict real-time in neural implicit RGB-D SLAM by relying on a SCR-based representation. We introduce a novel SCR architecture specifically tailored for this purpose and detail the critical design choices required to integrate SCR into a live SLAM pipeline. The resulting framework is simple yet flexible, seamlessly supporting both sparse and dense features, and operates reliably in dynamic environments without special adaptation. We evaluate our approach on established synthetic and real-world benchmarks, demonstrating competitive performance against the state of the art. Project Page: this https URL 

---
# Sample-Efficient Robot Skill Learning for Construction Tasks: Benchmarking Hierarchical Reinforcement Learning and Vision-Language-Action VLA Model 

**Authors**: Zhaofeng Hu, Hongrui Yu, Vaidhyanathan Chandramouli, Ci-Jyun Liang  

**Link**: [PDF](https://arxiv.org/pdf/2512.14031)  

**Abstract**: This study evaluates two leading approaches for teaching construction robots new skills to understand their applicability for construction automation: a Vision-Language-Action (VLA) model and Reinforcement Learning (RL) methods. The goal is to understand both task performance and the practical effort needed to deploy each approach on real jobs. The authors developed two teleoperation interfaces to control the robots and collect the demonstrations needed, both of which proved effective for training robots for long-horizon and dexterous tasks. In addition, the authors conduct a three-stage evaluation. First, the authors compare a Multi-Layer Perceptron (MLP) policy with a Deep Q-network (DQN) imitation model to identify the stronger RL baseline, focusing on model performance, generalization, and a pick-up experiment. Second, three different VLA models are trained in two different scenarios and compared with each other. Third, the authors benchmark the selected RL baseline against the VLA model using computational and sample-efficiency measures and then a robot experiment on a multi-stage panel installation task that includes transport and installation. The VLA model demonstrates strong generalization and few-shot capability, achieving 60% and 100% success in the pickup phase. In comparison, DQN can be made robust but needs additional noise during tuning, which increases the workload. Overall, the findings indicate that VLA offers practical advantages for changing tasks by reducing programming effort and enabling useful performance with minimal data, while DQN provides a viable baseline when sufficient tuning effort is acceptable. 

---
# PerfCoder: Large Language Models for Interpretable Code Performance Optimization 

**Authors**: Jiuding Yang, Shengyao Lu, Hongxuan Liu, Shayan Shirahmad Gale Bagi, Zahra Fazel, Tomasz Czajkowski, Di Niu  

**Link**: [PDF](https://arxiv.org/pdf/2512.14018)  

**Abstract**: Large language models (LLMs) have achieved remarkable progress in automatic code generation, yet their ability to produce high-performance code remains limited--a critical requirement in real-world software systems. We argue that current LLMs struggle not only due to data scarcity but, more importantly, because they lack supervision that guides interpretable and effective performance improvements. In this work, we introduce PerfCoder, a family of LLMs specifically designed to generate performance-enhanced code from source code via interpretable, customized optimizations. PerfCoder is fine-tuned on a curated collection of real-world optimization trajectories with human-readable annotations, and preference-aligned by reinforcement fine-tuning using runtime measurements, enabling it to propose input-specific improvement strategies and apply them directly without relying on iterative refinement. On the PIE code performance benchmark, PerfCoder surpasses all existing models in both runtime speedup and effective optimization rate, demonstrating that performance optimization cannot be achieved by scale alone but requires optimization stratetgy awareness. In addition, PerfCoder can generate interpretable feedback about the source code, which, when provided as input to a larger LLM in a planner-and-optimizer cooperative workflow, can further improve outcomes. Specifically, we elevate the performance of 32B models and GPT-5 to new levels on code optimization, substantially surpassing their original performance. 

---
# KFS-Bench: Comprehensive Evaluation of Key Frame Sampling in Long Video Understanding 

**Authors**: Zongyao Li, Kengo Ishida, Satoshi Yamazaki, Xiaotong Ji, Jianquan Liu  

**Link**: [PDF](https://arxiv.org/pdf/2512.14017)  

**Abstract**: We propose KFS-Bench, the first benchmark for key frame sampling in long video question answering (QA), featuring multi-scene annotations to enable direct and robust evaluation of sampling strategies. Key frame sampling is crucial for efficient long-form video understanding. In long video QA, selecting informative frames enables multimodal large language models (MLLMs) to improve both accuracy and efficiency. KFS-Bench addresses the limitation of prior works that only indirectly assess frame selection quality via QA accuracy. By providing ground-truth annotations of multiple disjoint scenes required per question, KFS-Bench allows us to directly analyze how different sampling approaches capture essential content across an entire long video. Using KFS-Bench, we conduct a comprehensive study of key frame sampling methods and identify that not only sampling precision but also scene coverage and sampling balance are the key factors influencing QA performance. Regarding all the factors, we design a novel sampling quality metric that correlates with QA accuracy. Furthermore, we develop a novel key frame sampling method that leverages question-video relevance to balance sampling diversity against question-frame similarity, thereby improving coverage of relevant scenes. Our adaptively balanced sampling approach achieves superior performance in both key frame sampling and QA performance. The benchmark is available at this https URL. 

---
# Professional Software Developers Don't Vibe, They Control: AI Agent Use for Coding in 2025 

**Authors**: Ruanqianqian Huang, Avery Reyna, Sorin Lerner, Haijun Xia, Brian Hempel  

**Link**: [PDF](https://arxiv.org/pdf/2512.14012)  

**Abstract**: The rise of AI agents is transforming how software can be built. The promise of agents is that developers might write code quicker, delegate multiple tasks to different agents, and even write a full piece of software purely out of natural language. In reality, what roles agents play in professional software development remains in question. This paper investigates how experienced developers use agents in building software, including their motivations, strategies, task suitability, and sentiments. Through field observations (N=13) and qualitative surveys (N=99), we find that while experienced developers value agents as a productivity boost, they retain their agency in software design and implementation out of insistence on fundamental software quality attributes, employing strategies for controlling agent behavior leveraging their expertise. In addition, experienced developers feel overall positive about incorporating agents into software development given their confidence in complementing the agents' limitations. Our results shed light on the value of software development best practices in effective use of agents, suggest the kinds of tasks for which agents may be suitable, and point towards future opportunities for better agentic interfaces and agentic use guidelines. 

---
# Memo2496: Expert-Annotated Dataset and Dual-View Adaptive Framework for Music Emotion Recognition 

**Authors**: Qilin Li, C. L. Philip Chen, TongZhang  

**Link**: [PDF](https://arxiv.org/pdf/2512.13998)  

**Abstract**: Music Emotion Recogniser (MER) research faces challenges due to limited high-quality annotated datasets and difficulties in addressing cross-track feature drift. This work presents two primary contributions to address these issues. Memo2496, a large-scale dataset, offers 2496 instrumental music tracks with continuous valence arousal labels, annotated by 30 certified music specialists. Annotation quality is ensured through calibration with extreme emotion exemplars and a consistency threshold of 0.25, measured by Euclidean distance in the valence arousal space. Furthermore, the Dual-view Adaptive Music Emotion Recogniser (DAMER) is introduced. DAMER integrates three synergistic modules: Dual Stream Attention Fusion (DSAF) facilitates token-level bidirectional interaction between Mel spectrograms and cochleagrams via cross attention mechanisms; Progressive Confidence Labelling (PCL) generates reliable pseudo labels employing curriculum-based temperature scheduling and consistency quantification using Jensen Shannon divergence; and Style Anchored Memory Learning (SAML) maintains a contrastive memory queue to mitigate cross-track feature drift. Extensive experiments on the Memo2496, 1000songs, and PMEmo datasets demonstrate DAMER's state-of-the-art performance, improving arousal dimension accuracy by 3.43%, 2.25%, and 0.17%, respectively. Ablation studies and visualisation analyses validate each module's contribution. Both the dataset and source code are publicly available. 

---
# Multi-Agent Collaborative Framework for Intelligent IT Operations: An AOI System with Context-Aware Compression and Dynamic Task Scheduling 

**Authors**: Zishan Bai, Enze Ge, Junfeng Hao  

**Link**: [PDF](https://arxiv.org/pdf/2512.13956)  

**Abstract**: The proliferation of cloud-native architectures, characterized by microservices and dynamic orchestration, has rendered modern IT infrastructures exceedingly complex and volatile. This complexity generates overwhelming volumes of operational data, leading to critical bottlenecks in conventional systems: inefficient information processing, poor task coordination, and loss of contextual continuity during fault diagnosis and remediation. To address these challenges, we propose AOI (AI-Oriented Operations), a novel multi-agent collaborative framework that integrates three specialized agents with an LLM-based Context Compressor. Its core innovations include: (1) a dynamic task scheduling strategy that adaptively prioritizes operations based on real-time system states, and (2) a three-layer memory architecture comprising Working, Episodic, and Semantic layers that optimizes context retention and retrieval. Extensive experiments on both synthetic and real-world benchmarks demonstrate that AOI effectively mitigates information overload, achieving a 72.4% context compression ratio while preserving 92.8% of critical information and significantly enhances operational efficiency, attaining a 94.2% task success rate and reducing the Mean Time to Repair (MTTR) by 34.4% compared to the best baseline. This work presents a paradigm shift towards scalable, adaptive, and context-aware autonomous operations, enabling robust management of next-generation IT infrastructures with minimal human intervention. 

---
# Informing Acquisition Functions via Foundation Models for Molecular Discovery 

**Authors**: Qi Chen, Fabio Ramos, Alán Aspuru-Guzik, Florian Shkurti  

**Link**: [PDF](https://arxiv.org/pdf/2512.13935)  

**Abstract**: Bayesian Optimization (BO) is a key methodology for accelerating molecular discovery by estimating the mapping from molecules to their properties while seeking the optimal candidate. Typically, BO iteratively updates a probabilistic surrogate model of this mapping and optimizes acquisition functions derived from the model to guide molecule selection. However, its performance is limited in low-data regimes with insufficient prior knowledge and vast candidate spaces. Large language models (LLMs) and chemistry foundation models offer rich priors to enhance BO, but high-dimensional features, costly in-context learning, and the computational burden of deep Bayesian surrogates hinder their full utilization. To address these challenges, we propose a likelihood-free BO method that bypasses explicit surrogate modeling and directly leverages priors from general LLMs and chemistry-specific foundation models to inform acquisition functions. Our method also learns a tree-structured partition of the molecular search space with local acquisition functions, enabling efficient candidate selection via Monte Carlo Tree Search. By further incorporating coarse-grained LLM-based clustering, it substantially improves scalability to large candidate sets by restricting acquisition function evaluations to clusters with statistically higher property values. We show through extensive experiments and ablations that the proposed method substantially improves scalability, robustness, and sample efficiency in LLM-guided BO for molecular discovery. 

---
# Hierarchical Multi-agent Large Language Model Reasoning for Autonomous Functional Materials Discovery 

**Authors**: Samuel Rothfarb, Megan C. Davis, Ivana Matanovic, Baikun Li, Edward F. Holby, Wilton J.M. Kort-Kamp  

**Link**: [PDF](https://arxiv.org/pdf/2512.13930)  

**Abstract**: Artificial intelligence is reshaping scientific exploration, but most methods automate procedural tasks without engaging in scientific reasoning, limiting autonomy in discovery. We introduce Materials Agents for Simulation and Theory in Electronic-structure Reasoning (MASTER), an active learning framework where large language models autonomously design, execute, and interpret atomistic simulations. In MASTER, a multimodal system translates natural language into density functional theory workflows, while higher-level reasoning agents guide discovery through a hierarchy of strategies, including a single agent baseline and three multi-agent approaches: peer review, triage-ranking, and triage-forms. Across two chemical applications, CO adsorption on Cu-surface transition metal (M) adatoms and on M-N-C catalysts, reasoning-driven exploration reduces required atomistic simulations by up to 90% relative to trial-and-error selection. Reasoning trajectories reveal chemically grounded decisions that cannot be explained by stochastic sampling or semantic bias. Altogether, multi-agent collaboration accelerates materials discovery and marks a new paradigm for autonomous scientific exploration. 

---
# Context Branching for LLM Conversations: A Version Control Approach to Exploratory Programming 

**Authors**: Bhargav Chickmagalur Nanjundappa, Spandan Maaheshwari  

**Link**: [PDF](https://arxiv.org/pdf/2512.13914)  

**Abstract**: Large Language Models (LLMs) have become integral to software engineering workflows, yet their effectiveness degrades significantly in multi-turn conversations. Recent studies demonstrate an average 39% performance drop when instructions are delivered across multiple turns, with models making premature assumptions and failing to course correct (Laban et al., 2025). This degradation is particularly problematic in exploratory programming tasks where developers need to investigate alternative approaches without committing to a single path. Current solutions force users into a false dichotomy: continue in a context-polluted conversation where the LLM becomes increasingly confused, or start fresh and lose all accumulated context.
We present ContextBranch, a conversation management system that applies version control semantics to LLM interactions. ContextBranch provides four core primitives--checkpoint, branch, switch, and inject--enabling users to capture conversation state, explore alternatives in isolation, and selectively merge insights. We evaluate ContextBranch through a controlled experiment with 30 software engineering scenarios featuring intentionally polluting explorations. Branched conversations achieved higher response quality compared to linear conversations, with large improvements in focus and context awareness. Benefits were concentrated in complex scenarios involving conceptually distant explorations. Branching reduced context size by 58.1% (31.0 to 13.0 messages), eliminating irrelevant exploratory content. Our work establishes conversation branching as a fundamental primitive for AI-assisted exploratory work, demonstrating that isolation prevents context pollution when exploring alternatives. 

---
# Intelligent matter consisting of active particles 

**Authors**: Julian Jeggle, Raphael Wittkowski  

**Link**: [PDF](https://arxiv.org/pdf/2512.13912)  

**Abstract**: In this book chapter, we review how systems of simple motile agents can be used as a pathway to intelligent systems. It is a well known result from nature that large groups of entities following simple rules, such as swarms of animals, can give rise to much more complex collective behavior in a display of emergence. This begs the question whether we can emulate this behavior in synthetic matter and drive it to a point where the collective behavior reaches the complexity level of intelligent systems. Here, we will use a formalized notion of "intelligent matter" and compare it to recent results in the field of active matter. First, we will explore the approach of emergent computing in which specialized active matter systems are designed to directly solve a given task through emergent behavior. This we will then contrast with the approach of physical reservoir computing powered by the dynamics of active particle systems. In this context, we will also describe a novel reservoir computing scheme for active particles driven ultrasonically or via light refraction. 

---
# Exploring Machine Learning, Deep Learning, and Explainable AI Methods for Seasonal Precipitation Prediction in South America 

**Authors**: Matheus Corrêa Domingos, Valdivino Alexandre de Santiago Júnior, Juliana Aparecida Anochi, Elcio Hideiti Shiguemori, Luísa Mirelle Costa dos Santos, Hércules Carlos dos Santos Pereira, André Estevam Costa Oliveira  

**Link**: [PDF](https://arxiv.org/pdf/2512.13910)  

**Abstract**: Forecasting meteorological variables is challenging due to the complexity of their processes, requiring advanced models for accuracy. Accurate precipitation forecasts are vital for society. Reliable predictions help communities mitigate climatic impacts. Based on the current relevance of artificial intelligence (AI), classical machine learning (ML) and deep learning (DL) techniques have been used as an alternative or complement to dynamic modeling. However, there is still a lack of broad investigations into the feasibility of purely data-driven approaches for precipitation forecasting. This study aims at addressing this issue where different classical ML and DL approaches for forecasting precipitation in South America, taking into account all 2019 seasons, are considered in a detailed investigation. The selected classical ML techniques were Random Forests and extreme gradient boosting (XGBoost), while the DL counterparts were a 1D convolutional neural network (CNN 1D), a long short-term memory (LSTM) model, and a gated recurrent unit (GRU) model. Additionally, the Brazilian Global Atmospheric Model (BAM) was used as a representative of the traditional dynamic modeling approach. We also relied on explainable artificial intelligence (XAI) to provide some explanations for the models behaviors. LSTM showed strong predictive performance while BAM, the traditional dynamic model representative, had the worst results. Despite presented the higher latency, LSTM was most accurate for heavy precipitation. If cost is a concern, XGBoost offers lower latency with slightly accuracy loss. The results of this research confirm the viability of DL models for climate forecasting, solidifying a global trend in major meteorological and climate forecasting centers. 

---
# Assessing High-Risk Systems: An EU AI Act Verification Framework 

**Authors**: Alessio Buscemi, Tom Deckenbrunnen, Fahria Kabir, Nishat Mowla, Kateryna Mishchenko  

**Link**: [PDF](https://arxiv.org/pdf/2512.13907)  

**Abstract**: A central challenge in implementing the AI Act and other AI-relevant regulations in the EU is the lack of a systematic approach to verify their legal mandates. Recent surveys show that this regulatory ambiguity is perceived as a significant burden, leading to inconsistent readiness across Member States. This paper proposes a comprehensive framework designed to help close this gap by organising compliance verification along two fundamental dimensions: the type of method (controls vs. testing) and the target of assessment (data, model, processes, and final product). Additionally, our framework maps core legal requirements to concrete verification activities, serving as a vital bridge between policymakers and practitioners, and aligning legal text with technical standards and best practices. The proposed approach aims to reduce interpretive uncertainty, promote consistency in assessment practices, and support the alignment of regulatory, ethical, and technical perspectives across the AI lifecycle. 

---
# Generative AI for Video Translation: A Scalable Architecture for Multilingual Video Conferencing 

**Authors**: Amirkia Rafiei Oskooei, Eren Caglar, Ibrahim Sahin, Ayse Kayabay, Mehmet S. Aktas  

**Link**: [PDF](https://arxiv.org/pdf/2512.13904)  

**Abstract**: The real-time deployment of cascaded generative AI pipelines for applications like video translation is constrained by significant system-level challenges. These include the cumulative latency of sequential model inference and the quadratic ($\mathcal{O}(N^2)$) computational complexity that renders multi-user video conferencing applications unscalable. This paper proposes and evaluates a practical system-level framework designed to mitigate these critical bottlenecks. The proposed architecture incorporates a turn-taking mechanism to reduce computational complexity from quadratic to linear in multi-user scenarios, and a segmented processing protocol to manage inference latency for a perceptually real-time experience. We implement a proof-of-concept pipeline and conduct a rigorous performance analysis across a multi-tiered hardware setup, including commodity (NVIDIA RTX 4060), cloud (NVIDIA T4), and enterprise (NVIDIA A100) GPUs. Our objective evaluation demonstrates that the system achieves real-time throughput ($\tau < 1.0$) on modern hardware. A subjective user study further validates the approach, showing that a predictable, initial processing delay is highly acceptable to users in exchange for a smooth, uninterrupted playback experience. The work presents a validated, end-to-end system design that offers a practical roadmap for deploying scalable, real-time generative AI applications in multilingual communication platforms. 

---
# One Permutation Is All You Need: Fast, Reliable Variable Importance and Model Stress-Testing 

**Authors**: Albert Dorador  

**Link**: [PDF](https://arxiv.org/pdf/2512.13892)  

**Abstract**: Reliable estimation of feature contributions in machine learning models is essential for trust, transparency and regulatory compliance, especially when models are proprietary or otherwise operate as black boxes. While permutation-based methods are a standard tool for this task, classical implementations rely on repeated random permutations, introducing computational overhead and stochastic instability. In this paper, we show that by replacing multiple random permutations with a single, deterministic, and optimal permutation, we achieve a method that retains the core principles of permutation-based importance while being non-random, faster, and more stable. We validate this approach across nearly 200 scenarios, including real-world household finance and credit risk applications, demonstrating improved bias-variance tradeoffs and accuracy in challenging regimes such as small sample sizes, high dimensionality, and low signal-to-noise ratios. Finally, we introduce Systemic Variable Importance, a natural extension designed for model stress-testing that explicitly accounts for feature correlations. This framework provides a transparent way to quantify how shocks or perturbations propagate through correlated inputs, revealing dependencies that standard variable importance measures miss. Two real-world case studies demonstrate how this metric can be used to audit models for hidden reliance on protected attributes (e.g., gender or race), enabling regulators and practitioners to assess fairness and systemic risk in a principled and computationally efficient manner. 

---
# OPTIMA: Optimal One-shot Pruning for LLMs via Quadratic Programming Reconstruction 

**Authors**: Mohammad Mozaffari, Samuel Kushnir, Maryam Mehri Dehnavi, Amir Yazdanbakhsh  

**Link**: [PDF](https://arxiv.org/pdf/2512.13886)  

**Abstract**: Post-training model pruning is a promising solution, yet it faces a trade-off: simple heuristics that zero weights are fast but degrade accuracy, while principled joint optimization methods recover accuracy but are computationally infeasible at modern scale. One-shot methods such as SparseGPT offer a practical trade-off in optimality by applying efficient, approximate heuristic weight updates. To close this gap, we introduce OPTIMA, a practical one-shot post-training pruning method that balances accuracy and scalability. OPTIMA casts layer-wise weight reconstruction after mask selection as independent, row-wise Quadratic Programs (QPs) that share a common layer Hessian. Solving these QPs yields the per-row globally optimal update with respect to the reconstruction objective given the estimated Hessian. The shared-Hessian structure makes the problem highly amenable to batching on accelerators. We implement an accelerator-friendly QP solver that accumulates one Hessian per layer and solves many small QPs in parallel, enabling one-shot post-training pruning at scale on a single accelerator without fine-tuning. OPTIMA integrates with existing mask selectors and consistently improves zero-shot performance across multiple LLM families and sparsity regimes, yielding up to 3.97% absolute accuracy improvement. On an NVIDIA H100, OPTIMA prunes a 8B-parameter transformer end-to-end in 40 hours with 60GB peak memory. Together, these results set a new state-of-the-art accuracy-efficiency trade-offs for one-shot post-training pruning. 

---
# Privacy-Enhancing Infant Cry Classification with Federated Transformers and Denoising Regularization 

**Authors**: Geofrey Owino, Bernard Shibwabo  

**Link**: [PDF](https://arxiv.org/pdf/2512.13880)  

**Abstract**: Infant cry classification can aid early assessment of infant needs. However, deployment of such solutions is limited by privacy concerns around audio data, sensitivity to background noise, and domain shift across recording environments. We present an end-to-end infant cry analysis pipeline that integrates a denoising autoencoder (DAE), a convolutional tokenizer, and a Transformer encoder trained using communication-efficient federated learning (FL). The system performs on-device denoising, adaptive segmentation, post hoc calibration, and energy-based out-of-distribution (OOD) abstention. Federated training employs a regularized control variate update with 8-bit adapter deltas under secure aggregation. Using the Baby Chillanto and Donate-a-Cry datasets with ESC-50 noise overlays, the model achieves a macro F1 score of 0.938, an AUC of 0.962, and an Expected Calibration Error (ECE) of 0.032, while reducing per-round client upload from approximately 36 to 42 MB to 3.3 MB. Real-time edge inference on an NVIDIA Jetson Nano (4 GB, TensorRT FP16) achieves 96 ms per one-second spectrogram frame. These results demonstrate a practical path toward privacy-preserving, noise-robust, and communication-efficient infant cry classification suitable for federated deployment. 

---
# Verification-Guided Context Optimization for Tool Calling via Hierarchical LLMs-as-Editors 

**Authors**: Henger Li, Shuangjie You, Flavio Di Palo, Yiyue Qian, Ayush Jain  

**Link**: [PDF](https://arxiv.org/pdf/2512.13860)  

**Abstract**: Tool calling enables large language models (LLMs) to interact with external environments through tool invocation, providing a practical way to overcome the limitations of pretraining. However, the effectiveness of tool use depends heavily on the quality of the associated documentation and knowledge base context. These materials are usually written for human users and are often misaligned with how LLMs interpret information. This problem is even more pronounced in industrial settings, where hundreds of tools with overlapping functionality create challenges in scalability, variability, and ambiguity. We propose Verification-Guided Context Optimization (VGCO), a framework that uses LLMs as editors to automatically refine tool-related documentation and knowledge base context. VGCO works in two stages. First, Evaluation collects real-world failure cases and identifies mismatches between tools and their context. Second, Optimization performs hierarchical editing through offline learning with structure-aware, in-context optimization. The novelty of our LLM editors has three main aspects. First, they use a hierarchical structure that naturally integrates into the tool-calling workflow. Second, they are state-aware, action-specific, and verification-guided, which constrains the search space and enables efficient, targeted improvements. Third, they enable cost-efficient sub-task specialization, either by prompt engineering large editor models or by post-training smaller editor models. Unlike prior work that emphasizes multi-turn reasoning, VGCO focuses on the single-turn, large-scale tool-calling problem and achieves significant improvements in accuracy, robustness, and generalization across LLMs. 

---
# Improvise, Adapt, Overcome -- Telescopic Adapters for Efficient Fine-tuning of Vision Language Models in Medical Imaging 

**Authors**: Ujjwal Mishra, Vinita Shukla, Praful Hambarde, Amit Shukla  

**Link**: [PDF](https://arxiv.org/pdf/2512.13855)  

**Abstract**: Adapting Vision Language Segmentation Models (VLSMs) to medical imaging domains requires significant computational overhead when using conventional fine-tuning approaches. Existing Parameter-Efficient Fine-Tuning (PEFT) methods apply uniform adapter dimensions across all transformer layers, leading to suboptimal parameter allocation and reduced adaptation efficiency. We introduce Telescopic Adapters, a novel PEFT framework that employs depth-aware scaling to progressively increase adapter capacity from shallow to deep transformer layers. Our method integrates lightweight bottleneck modules within CLIPSeg's vision and text encoders, with adapter dimensions dynamically scaled based on layer depth and semantic relevance. Using only 613k trainable parameters--244x fewer than end-to-end fine-tuning, Telescopic Adapters achieve superior performance across five diverse medical datasets spanning polyp segmentation, skin lesion detection, and breast ultrasound imaging. Comprehensive ablation studies demonstrate that deeper layers require substantially more adaptation capacity than shallow layers, validating our telescopic scaling hypothesis. Our approach establishes a new paradigm for efficient medical VLSM fine-tuning, enabling deployment in resource-constrained clinical environments while maintaining competitive segmentation accuracy. 

---
# VajraV1 -- The most accurate Real Time Object Detector of the YOLO family 

**Authors**: Naman Balbir Singh Makkar  

**Link**: [PDF](https://arxiv.org/pdf/2512.13834)  

**Abstract**: Recent years have seen significant advances in real-time object detection, with the release of YOLOv10, YOLO11, YOLOv12, and YOLOv13 between 2024 and 2025. This technical report presents the VajraV1 model architecture, which introduces architectural enhancements over existing YOLO-based detectors. VajraV1 combines effective design choices from prior YOLO models to achieve state-of-the-art accuracy among real-time object detectors while maintaining competitive inference speed.
On the COCO validation set, VajraV1-Nano achieves 44.3% mAP, outperforming YOLOv12-N by 3.7% and YOLOv13-N by 2.7% at latency competitive with YOLOv12-N and YOLOv11-N. VajraV1-Small achieves 50.4% mAP, exceeding YOLOv12-S and YOLOv13-S by 2.4%. VajraV1-Medium achieves 52.7% mAP, outperforming YOLOv12-M by 0.2%. VajraV1-Large achieves 53.7% mAP, surpassing YOLOv13-L by 0.3%. VajraV1-Xlarge achieves 56.2% mAP, outperforming all existing real-time object detectors. 

---
# EEG-D3: A Solution to the Hidden Overfitting Problem of Deep Learning Models 

**Authors**: Siegfried Ludwig, Stylianos Bakas, Konstantinos Barmpas, Georgios Zoumpourlis, Dimitrios A. Adamos, Nikolaos Laskaris, Yannis Panagakis, Stefanos Zafeiriou  

**Link**: [PDF](https://arxiv.org/pdf/2512.13806)  

**Abstract**: Deep learning for decoding EEG signals has gained traction, with many claims to state-of-the-art accuracy. However, despite the convincing benchmark performance, successful translation to real applications is limited. The frequent disconnect between performance on controlled BCI benchmarks and its lack of generalisation to practical settings indicates hidden overfitting problems. We introduce Disentangled Decoding Decomposition (D3), a weakly supervised method for training deep learning models across EEG datasets. By predicting the place in the respective trial sequence from which the input window was sampled, EEG-D3 separates latent components of brain activity, akin to non-linear ICA. We utilise a novel model architecture with fully independent sub-networks for strict interpretability. We outline a feature interpretation paradigm to contrast the component activation profiles on different datasets and inspect the associated temporal and spatial filters. The proposed method reliably separates latent components of brain activity on motor imagery data. Training downstream classifiers on an appropriate subset of these components prevents hidden overfitting caused by task-correlated artefacts, which severely affects end-to-end classifiers. We further exploit the linearly separable latent space for effective few-shot learning on sleep stage classification. The ability to distinguish genuine components of brain activity from spurious features results in models that avoid the hidden overfitting problem and generalise well to real-world applications, while requiring only minimal labelled data. With interest to the neuroscience community, the proposed method gives researchers a tool to separate individual brain processes and potentially even uncover heretofore unknown dynamics. 

---
# Beyond Procedural Compliance: Human Oversight as a Dimension of Well-being Efficacy in AI Governance 

**Authors**: Yao Xie, Walter Cullen  

**Link**: [PDF](https://arxiv.org/pdf/2512.13768)  

**Abstract**: Major AI ethics guidelines and laws, including the EU AI Act, call for effective human oversight, but do not define it as a distinct and developable capacity. This paper introduces human oversight as a well-being capacity, situated within the emerging Well-being Efficacy framework. The concept integrates AI literacy, ethical discernment, and awareness of human needs, acknowledging that some needs may be conflicting or harmful. Because people inevitably project desires, fears, and interests into AI systems, oversight requires the competence to examine and, when necessary, restrain problematic demands.
The authors argue that the sustainable and cost-effective development of this capacity depends on its integration into education at every level, from professional training to lifelong learning. The frame of human oversight as a well-being capacity provides a practical path from high-level regulatory goals to the continuous cultivation of human agency and responsibility essential for safe and ethical AI. The paper establishes a theoretical foundation for future research on the pedagogical implementation and empirical validation of well-being effectiveness in multiple contexts. 

---
# Towards Deep Learning Surrogate for the Forward Problem in Electrocardiology: A Scalable Alternative to Physics-Based Models 

**Authors**: Shaheim Ogbomo-Harmitt, Cesare Magnetti, Chiara Spota, Jakub Grzelak, Oleg Aslanidi  

**Link**: [PDF](https://arxiv.org/pdf/2512.13765)  

**Abstract**: The forward problem in electrocardiology, computing body surface potentials from cardiac electrical activity, is traditionally solved using physics-based models such as the bidomain or monodomain equations. While accurate, these approaches are computationally expensive, limiting their use in real-time and large-scale clinical applications. We propose a proof-of-concept deep learning (DL) framework as an efficient surrogate for forward solvers. The model adopts a time-dependent, attention-based sequence-to-sequence architecture to predict electrocardiogram (ECG) signals from cardiac voltage propagation maps. A hybrid loss combining Huber loss with a spectral entropy term was introduced to preserve both temporal and frequency-domain fidelity. Using 2D tissue simulations incorporating healthy, fibrotic, and gap junction-remodelled conditions, the model achieved high accuracy (mean $R^2 = 0.99 \pm 0.01$). Ablation studies confirmed the contributions of convolutional encoders, time-aware attention, and spectral entropy loss. These findings highlight DL as a scalable, cost-effective alternative to physics-based solvers, with potential for clinical and digital twin applications. 

---
# Network-Wide Traffic Volume Estimation from Speed Profiles using a Spatio-Temporal Graph Neural Network with Directed Spatial Attention 

**Authors**: Léo Hein, Giovanni de Nunzio, Giovanni Chierchia, Aurélie Pirayre, Laurent Najman  

**Link**: [PDF](https://arxiv.org/pdf/2512.13758)  

**Abstract**: Existing traffic volume estimation methods typically address either forecasting traffic on sensor-equipped roads or spatially imputing missing volumes using nearby sensors. While forecasting models generally disregard unmonitored roads by design, spatial imputation methods explicitly address network-wide estimation; yet this approach relies on volume data at inference time, limiting its applicability in sensor-scarce cities. Unlike traffic volume data, probe vehicle speeds and static road attributes are more broadly accessible and support full coverage of road segments in most urban networks. In this work, we present the Hybrid Directed-Attention Spatio-Temporal Graph Neural Network (HDA-STGNN), an inductive deep learning framework designed to tackle the network-wide volume estimation problem. Our approach leverages speed profiles, static road attributes, and road network topology to predict daily traffic volume profiles across all road segments in the network. To evaluate the effectiveness of our approach, we perform extensive ablation studies that demonstrate the model's capacity to capture complex spatio-temporal dependencies and highlight the value of topological information for accurate network-wide traffic volume estimation without relying on volume data at inference time. 

---
# STAR: STacked AutoRegressive Scheme for Unified Multimodal Learning 

**Authors**: Jie Qin, Jiancheng Huang, Limeng Qiao, Lin Ma  

**Link**: [PDF](https://arxiv.org/pdf/2512.13752)  

**Abstract**: Multimodal large language models (MLLMs) play a pivotal role in advancing the quest for general artificial intelligence. However, achieving unified target for multimodal understanding and generation remains challenging due to optimization conflicts and performance trade-offs. To effectively enhance generative performance while preserving existing comprehension capabilities, we introduce STAR: a STacked AutoRegressive scheme for task-progressive unified multimodal learning. This approach decomposes multimodal learning into multiple stages: understanding, generation, and editing. By freezing the parameters of the fundamental autoregressive (AR) model and progressively stacking isomorphic AR modules, it avoids cross-task interference while expanding the model's capabilities. Concurrently, we introduce a high-capacity VQ to enhance the granularity of image representations and employ an implicit reasoning mechanism to improve generation quality under complex conditions. Experiments demonstrate that STAR achieves state-of-the-art performance on GenEval (0.91), DPG-Bench (87.44), and ImgEdit (4.34), validating its efficacy for unified multimodal learning. 

---
# MIDUS: Memory-Infused Depth Up-Scaling 

**Authors**: Taero Kim, Hoyoon Byun, Youngjun Choi, Sungrae Park, Kyungwoo Song  

**Link**: [PDF](https://arxiv.org/pdf/2512.13751)  

**Abstract**: Scaling large language models (LLMs) demands approaches that increase capacity without incurring excessive parameter growth or inference cost. Depth Up-Scaling (DUS) has emerged as a promising strategy by duplicating layers and applying Continual Pre-training (CPT), but its reliance on feed-forward networks (FFNs) limits efficiency and attainable gains. We introduce Memory-Infused Depth Up-Scaling (MIDUS), which replaces FFNs in duplicated blocks with a head-wise memory (HML) layer. Motivated by observations that attention heads have distinct roles both across and within layers, MIDUS assigns an independent memory bank to each head, enabling head-wise retrieval and injecting information into subsequent layers while preserving head-wise functional structure. This design combines sparse memory access with head-wise representations and incorporates an efficient per-head value factorization module, thereby relaxing the usual efficiency-performance trade-off. Across our CPT experiments, MIDUS exhibits robust performance improvements over strong DUS baselines while maintaining a highly efficient parameter footprint. Our findings establish MIDUS as a compelling and resource-efficient alternative to conventional FFN replication for depth up-scaling by leveraging its head-wise memory design. 

---
# The algorithmic muse and the public domain: Why copyrights legal philosophy precludes protection for generative AI outputs 

**Authors**: Ezieddin Elmahjub  

**Link**: [PDF](https://arxiv.org/pdf/2512.13750)  

**Abstract**: Generative AI (GenAI) outputs are not copyrightable. This article argues why. We bypass conventional doctrinal analysis that focuses on black letter law notions of originality and authorship to re-evaluate copyright's foundational philosophy. GenAI fundamentally severs the direct human creative link to expressive form. Traditional theories utilitarian incentive, labor desert and personality fail to provide coherent justification for protection. The public domain constitutes the default baseline for intellectual creations. Those seeking copyright coverage for GenAI outputs bear the burden of proof. Granting copyright to raw GenAI outputs would not only be philosophically unsound but would also trigger an unprecedented enclosure of the digital commons, creating a legal quagmire and stifling future innovation. The paper advocates for a clear distinction: human creative contributions to AI-generated works may warrant protection, but the raw algorithmic output should remain in the public domain. 

---
# Comparative Evaluation of Embedding Representations for Financial News Sentiment Analysis 

**Authors**: Joyjit Roy, Samaresh Kumar Singh  

**Link**: [PDF](https://arxiv.org/pdf/2512.13749)  

**Abstract**: Financial sentiment analysis enhances market understanding; however, standard natural language processing approaches encounter significant challenges when applied to small datasets. This study provides a comparative evaluation of embedding-based methods for financial news sentiment classification in resource-constrained environments. Word2Vec, GloVe, and sentence transformer representations are evaluated in combination with gradient boosting on manually labeled headlines. Experimental results identify a substantial gap between validation and test performance, with models performing worse than trivial baselines despite strong validation metrics. The analysis demonstrates that pretrained embeddings yield diminishing returns below a critical data sufficiency threshold, and that small validation sets contribute to overfitting during model selection. Practical application is illustrated through weekly sentiment aggregation and narrative summarization for market monitoring workflows. The findings offer empirical evidence that embedding quality alone cannot address fundamental data scarcity in sentiment classification. For practitioners operating with limited resources, the results indicate the need to consider alternative approaches such as few-shot learning, data augmentation, or lexicon-enhanced hybrid methods when labeled samples are scarce. 

---
# Why Text Prevails: Vision May Undermine Multimodal Medical Decision Making 

**Authors**: Siyuan Dai, Lunxiao Li, Kun Zhao, Eardi Lila, Paul K. Crane, Heng Huang, Dongkuan Xu, Haoteng Tang, Liang Zhan  

**Link**: [PDF](https://arxiv.org/pdf/2512.13747)  

**Abstract**: With the rapid progress of large language models (LLMs), advanced multimodal large language models (MLLMs) have demonstrated impressive zero-shot capabilities on vision-language tasks. In the biomedical domain, however, even state-of-the-art MLLMs struggle with basic Medical Decision Making (MDM) tasks. We investigate this limitation using two challenging datasets: (1) three-stage Alzheimer's disease (AD) classification (normal, mild cognitive impairment, dementia), where category differences are visually subtle, and (2) MIMIC-CXR chest radiograph classification with 14 non-mutually exclusive conditions. Our empirical study shows that text-only reasoning consistently outperforms vision-only or vision-text settings, with multimodal inputs often performing worse than text alone. To mitigate this, we explore three strategies: (1) in-context learning with reason-annotated exemplars, (2) vision captioning followed by text-only inference, and (3) few-shot fine-tuning of the vision tower with classification supervision. These findings reveal that current MLLMs lack grounded visual understanding and point to promising directions for improving multimodal decision making in healthcare. 

---
# A Spatio-Temporal Hybrid Quantum-Classical Graph Convolutional Neural Network Approach for Urban Taxi Destination Prediction 

**Authors**: Xiuying Zhang, Qinsheng Zhu, Xiaodong Xing  

**Link**: [PDF](https://arxiv.org/pdf/2512.13745)  

**Abstract**: We propose a Hybrid Spatio-Temporal Quantum Graph Convolutional Network (H-STQGCN) algorithm by combining the strengths of quantum computing and classical deep learning to predict the taxi destination within urban road networks. Our algorithm consists of two branches: spatial processing and time evolution. Regarding the spatial processing, the classical module encodes the local topological features of the road network based on the GCN method, and the quantum module is designed to map graph features onto parameterized quantum circuits through a differentiable pooling layer. The time evolution is solved by integrating multi-source contextual information and capturing dynamic trip dependencies on the classical TCN theory. Finally, our experimental results demonstrate that the proposed algorithm outperforms the current methods in terms of prediction accuracy and stability, validating the unique advantages of the quantum-enhanced mechanism in capturing high-dimensional spatial dependencies. 

---
# Toward Noise-Aware Audio Deepfake Detection: Survey, SNR-Benchmarks, and Practical Recipes 

**Authors**: Udayon Sen, Alka Luqman, Anupam Chattopadhyay  

**Link**: [PDF](https://arxiv.org/pdf/2512.13744)  

**Abstract**: Deepfake audio detection has progressed rapidly with strong pre-trained encoders (e.g., WavLM, Wav2Vec2, MMS). However, performance in realistic capture conditions - background noise (domestic/office/transport), room reverberation, and consumer channels - often lags clean-lab results. We survey and evaluate robustness for state-of-the-art audio deepfake detection models and present a reproducible framework that mixes MS-SNSD noises with ASVspoof 2021 DF utterances to evaluate under controlled signal-to-noise ratios (SNRs). SNR is a measured proxy for noise severity used widely in speech; it lets us sweep from near-clean (35 dB) to very noisy (-5 dB) to quantify graceful degradation. We study multi-condition training and fixed-SNR testing for pretrained encoders (WavLM, Wav2Vec2, MMS), reporting accuracy, ROC-AUC, and EER on binary and four-class (authenticity x corruption) tasks. In our experiments, finetuning reduces EER by 10-15 percentage points at 10-0 dB SNR across backbones. 

---
# DL$^3$M: A Vision-to-Language Framework for Expert-Level Medical Reasoning through Deep Learning and Large Language Models 

**Authors**: Md. Najib Hasan, Imran Ahmad, Sourav Basak Shuvo, Md. Mahadi Hasan Ankon, Sunanda Das, Nazmul Siddique, Hui Wang  

**Link**: [PDF](https://arxiv.org/pdf/2512.13742)  

**Abstract**: Medical image classifiers detect gastrointestinal diseases well, but they do not explain their decisions. Large language models can generate clinical text, yet they struggle with visual reasoning and often produce unstable or incorrect explanations. This leaves a gap between what a model sees and the type of reasoning a clinician expects. We introduce a framework that links image classification with structured clinical reasoning. A new hybrid model, MobileCoAtNet, is designed for endoscopic images and achieves high accuracy across eight stomach-related classes. Its outputs are then used to drive reasoning by several LLMs. To judge this reasoning, we build two expert-verified benchmarks covering causes, symptoms, treatment, lifestyle, and follow-up care. Thirty-two LLMs are evaluated against these gold standards. Strong classification improves the quality of their explanations, but none of the models reach human-level stability. Even the best LLMs change their reasoning when prompts vary. Our study shows that combining DL with LLMs can produce useful clinical narratives, but current LLMs remain unreliable for high-stakes medical decisions. The framework provides a clearer view of their limits and a path for building safer reasoning systems. The complete source code and datasets used in this study are available at this https URL. 

---
# The Laminar Flow Hypothesis: Detecting Jailbreaks via Semantic Turbulence in Large Language Models 

**Authors**: Md. Hasib Ur Rahman  

**Link**: [PDF](https://arxiv.org/pdf/2512.13741)  

**Abstract**: As Large Language Models (LLMs) become ubiquitous, the challenge of securing them against adversarial "jailbreaking" attacks has intensified. Current defense strategies often rely on computationally expensive external classifiers or brittle lexical filters, overlooking the intrinsic dynamics of the model's reasoning process. In this work, the Laminar Flow Hypothesis is introduced, which posits that benign inputs induce smooth, gradual transitions in an LLM's high-dimensional latent space, whereas adversarial prompts trigger chaotic, high-variance trajectories - termed Semantic Turbulence - resulting from the internal conflict between safety alignment and instruction-following objectives. This phenomenon is formalized through a novel, zero-shot metric: the variance of layer-wise cosine velocity. Experimental evaluation across diverse small language models reveals a striking diagnostic capability. The RLHF-aligned Qwen2-1.5B exhibits a statistically significant 75.4% increase in turbulence under attack (p less than 0.001), validating the hypothesis of internal conflict. Conversely, Gemma-2B displays a 22.0% decrease in turbulence, characterizing a distinct, low-entropy "reflex-based" refusal mechanism. These findings demonstrate that Semantic Turbulence serves not only as a lightweight, real-time jailbreak detector but also as a non-invasive diagnostic tool for categorizing the underlying safety architecture of black-box models. 

---
# Human-AI Collaboration Mechanism Study on AIGC Assisted Image Production for Special Coverage 

**Authors**: Yajie Yang, Yuqing Zhao, Xiaochao Xi, Yinan Zhu  

**Link**: [PDF](https://arxiv.org/pdf/2512.13739)  

**Abstract**: Artificial Intelligence Generated Content (AIGC) assisting image production triggers controversy in journalism while attracting attention from media agencies. Key issues involve misinformation, authenticity, semantic fidelity, and interpretability. Most AIGC tools are opaque "black boxes," hindering the dual demands of content accuracy and semantic alignment and creating ethical, sociotechnical, and trust dilemmas. This paper explores pathways for controllable image production in journalism's special coverage and conducts two experiments with projects from China's media agency: (1) Experiment 1 tests cross-platform adaptability via standardized prompts across three scenes, revealing disparities in semantic alignment, cultural specificity, and visual realism driven by training-corpus bias and platform-level filtering. (2) Experiment 2 builds a human-in-the-loop modular pipeline combining high-precision segmentation (SAM, GroundingDINO), semantic alignment (BrushNet), and style regulating (Style-LoRA, Prompt-to-Prompt), ensuring editorial fidelity through CLIP-based semantic scoring, NSFW/OCR/YOLO filtering, and verifiable content credentials. Traceable deployment preserves semantic representation. Consequently, we propose a human-AI collaboration mechanism for AIGC assisted image production in special coverage and recommend evaluating Character Identity Stability (CIS), Cultural Expression Accuracy (CEA), and User-Public Appropriateness (U-PA). 

---
# Instilling Organisational Values in Firefighters through Simulation-Based Training 

**Authors**: Nardine Osman, Manel Rodriguez-Soto, Jordi Sabater-Mir  

**Link**: [PDF](https://arxiv.org/pdf/2512.13737)  

**Abstract**: In firefighting and other emergency operations, decisions made under pressure carry profound ethical weight and can significantly impact incident outcomes and firefighter safety. Traditional training methods, while foundational, often fall short in adequately preparing firefighters for the complex ethical dilemmas and value conflicts inherent in chaotic emergency environments. This paper proposes a conceptual framework for enhancing firefighter training by systematically integrating departmental values into simulation-based training. This approach fosters deeper value internalisation and improves value-driven decision-making under pressure. Furthermore, the underlying tools can also be leveraged to evaluate and refine departmental operational protocols for better alignment with preferred values. 

---
# TF-MCL: Time-frequency Fusion and Multi-domain Cross-Loss for Self-supervised Depression Detection 

**Authors**: Li-Xuan Zhao, Chen-Yang Xu, Wen-Qiang Li, Bo Wang, Rong-Xing Wei, Qing-Hao Menga  

**Link**: [PDF](https://arxiv.org/pdf/2512.13736)  

**Abstract**: In recent years, there has been a notable increase in the use of supervised detection methods of major depressive disorder (MDD) based on electroencephalogram (EEG) signals. However, the process of labeling MDD remains challenging. As a self-supervised learning method, contrastive learning could address the shortcomings of supervised learning methods, which are unduly reliant on labels in the context of MDD detection. However, existing contrastive learning methods are not specifically designed to characterize the time-frequency distribution of EEG signals, and their capacity to acquire low-semantic data representations is still inadequate for MDD detection tasks. To address the problem of contrastive learning method, we propose a time-frequency fusion and multi-domain cross-loss (TF-MCL) model for MDD detection. TF-MCL generates time-frequency hybrid representations through the use of a fusion mapping head (FMH), which efficiently remaps time-frequency domain information to the fusion domain, and thus can effectively enhance the model's capacity to synthesize time-frequency information. Moreover, by optimizing the multi-domain cross-loss function, the distribution of the representations in the time-frequency domain and the fusion domain is reconstructed, thereby improving the model's capacity to acquire fusion representations. We evaluated the performance of our model on the publicly available datasets MODMA and PRED+CT and show a significant improvement in accuracy, outperforming the existing state-of-the-art (SOTA) method by 5.87% and 9.96%, respectively. 

---
# DARTs: A Dual-Path Robust Framework for Anomaly Detection in High-Dimensional Multivariate Time Series 

**Authors**: Xuechun Liu, Heli Sun, Xuecheng Wu, Ruichen Cao, Yunyun Shi, Dingkang Yang, Haoran Li  

**Link**: [PDF](https://arxiv.org/pdf/2512.13735)  

**Abstract**: Multivariate time series anomaly detection (MTSAD) aims to accurately identify and localize complex abnormal patterns in the large-scale industrial control systems. While existing approaches excel in recognizing the distinct patterns under the low-dimensional scenarios, they often fail to robustly capture long-range spatiotemporal dependencies when learning representations from the high-dimensional noisy time series. To address these limitations, we propose DARTs, a robust long short-term dual-path framework with window-aware spatiotemporal soft fusion mechanism, which can be primarily decomposed into three complementary components. Specifically, in the short-term path, we introduce a Multi-View Sparse Graph Learner and a Diffusion Multi-Relation Graph Unit that collaborate to adaptively capture hierarchical discriminative short-term spatiotemporal patterns in the high-noise time series. While in the long-term path, we design a Multi-Scale Spatiotemporal Graph Constructor to model salient long-term dynamics within the high-dimensional representation space. Finally, a window-aware spatiotemporal soft-fusion mechanism is introduced to filter the residual noise while seamlessly integrating anomalous patterns. Extensive qualitative and quantitative experimental results across mainstream datasets demonstrate the superiority and robustness of our proposed DARTs. A series of ablation studies are also conducted to explore the crucial design factors of our proposed components. Our code and model will be made publicly open soon. 

---
# Plug-and-Play Parameter-Efficient Tuning of Embeddings for Federated Recommendation 

**Authors**: Haochen Yuan, Yang Zhang, Xiang He, Quan Z. Sheng, Zhongjie Wang  

**Link**: [PDF](https://arxiv.org/pdf/2512.13734)  

**Abstract**: With the rise of cloud-edge collaboration, recommendation services are increasingly trained in distributed environments. Federated Recommendation (FR) enables such multi-end collaborative training while preserving privacy by sharing model parameters instead of raw data. However, the large number of parameters, primarily due to the massive item embeddings, significantly hampers communication efficiency. While existing studies mainly focus on improving the efficiency of FR models, they largely overlook the issue of embedding parameter overhead. To address this gap, we propose a FR training framework with Parameter-Efficient Fine-Tuning (PEFT) based embedding designed to reduce the volume of embedding parameters that need to be transmitted. Our approach offers a lightweight, plugin-style solution that can be seamlessly integrated into existing FR methods. In addition to incorporating common PEFT techniques such as LoRA and Hash-based encoding, we explore the use of Residual Quantized Variational Autoencoders (RQ-VAE) as a novel PEFT strategy within our framework. Extensive experiments across various FR model backbones and datasets demonstrate that our framework significantly reduces communication overhead while improving accuracy. The source code is available at this https URL. 

---
# Low-Rank Compression of Language Models via Differentiable Rank Selection 

**Authors**: Sidhant Sundrani, Francesco Tudisco, Pasquale Minervini  

**Link**: [PDF](https://arxiv.org/pdf/2512.13733)  

**Abstract**: Approaches for compressing large-language models using low-rank decomposition have made strides, particularly with the introduction of activation and loss-aware SVD, which improves the trade-off between decomposition rank and downstream task performance. Despite these advancements, a persistent challenge remains--selecting the optimal ranks for each layer to jointly optimise compression rate and downstream task accuracy. Current methods either rely on heuristics that can yield sub-optimal results due to their limited discrete search space or are gradient-based but are not as performant as heuristic approaches without post-compression fine-tuning. To address these issues, we propose Learning to Low-Rank Compress (LLRC), a gradient-based approach which directly learns the weights of masks that select singular values in a fine-tuning-free setting. Using a calibration dataset, we train only the mask weights to select fewer and fewer singular values while minimising the divergence of intermediate activations from the original model. Our approach outperforms competing ranking selection methods that similarly require no post-compression fine-tuning across various compression rates on common-sense reasoning and open-domain question-answering tasks. For instance, with a compression rate of 20% on Llama-2-13B, LLRC outperforms the competitive Sensitivity-based Truncation Rank Searching (STRS) on MMLU, BoolQ, and OpenbookQA by 12%, 3.5%, and 4.4%, respectively. Compared to other compression techniques, our approach consistently outperforms fine-tuning-free variants of SVD-LLM and LLM-Pruner across datasets and compression rates. Our fine-tuning-free approach also performs competitively with the fine-tuning variant of LLM-Pruner. 

---
# PIS: A Generalized Physical Inversion Solver for Arbitrary Sparse Observations via Set-Conditioned Diffusion 

**Authors**: Weijie Yang, Xun Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2512.13732)  

**Abstract**: Estimation of PDE-constrained physical parameters from limited indirect measurements is inherently ill-posed, particularly when observations are sparse, irregular, and constrained by real-world sensor placement. This challenge is ubiquitous in fields such as fluid mechanics, seismic inversion, and structural health monitoring. Existing deep and operator-learning models collapse under these conditions: fixed-grid assumptions fail, reconstruction deteriorates sharply, and inversion becomes unreliable with limited robustness and no uncertainty quantification (UQ).We propose the Physical Inversion Solver (PIS), a set-conditioned diffusion framework enabling inversion from truly arbitrary observation sets. PIS employs a Set Transformer-based encoder to handle measurements of any number or geometry, and a cosine-annealed sparsity curriculum for exceptional robustness. An accompanying information-theoretic analysis provides insight into the limits of inversion under extreme sparsity by revealing how observation entropy varies across physical this http URL is evaluated on three challenging PDE inverse problems: Darcy flow, wavefield inversion (Helmholtz), and structural health monitoring (Hooke's Law). Across all tasks and sparsity regimes -- including extreme cases with an observation rate of only $0.29\%$ -- existing operator-learning baselines fail to reconstruct meaningful fields, often diverging or collapsing this http URL stark contrast, PIS remains stable and accurate, reducing inversion error by $12.28\%$--$88.73\%$ and reliably producing calibrated posterior samples. These samples accurately reflect both data scarcity and intrinsic physical ambiguity. These results position PIS as a powerful, general-purpose, and uniquely sparsity-resilient solution for physical inversion under arbitrary and severely undersampled observations. 

---
# Complex Mathematical Expression Recognition: Benchmark, Large-Scale Dataset and Strong Baseline 

**Authors**: Weikang Bai, Yongkun Du, Yuchen Su, Yazhen Xie, Zhineng Chen  

**Link**: [PDF](https://arxiv.org/pdf/2512.13731)  

**Abstract**: Mathematical Expression Recognition (MER) has made significant progress in recognizing simple expressions, but the robust recognition of complex mathematical expressions with many tokens and multiple lines remains a formidable challenge. In this paper, we first introduce CMER-Bench, a carefully constructed benchmark that categorizes expressions into three difficulty levels: easy, moderate, and complex. Leveraging CMER-Bench, we conduct a comprehensive evaluation of existing MER models and general-purpose multimodal large language models (MLLMs). The results reveal that while current methods perform well on easy and moderate expressions, their performance degrades significantly when handling complex mathematical expressions, mainly because existing public training datasets are primarily composed of simple samples. In response, we propose MER-17M and CMER-3M that are large-scale datasets emphasizing the recognition of complex mathematical expressions. The datasets provide rich and diverse samples to support the development of accurate and robust complex MER models. Furthermore, to address the challenges posed by the complicated spatial layout of complex expressions, we introduce a novel expression tokenizer, and a new representation called Structured Mathematical Language, which explicitly models the hierarchical and spatial structure of expressions beyond LaTeX format. Based on these, we propose a specialized model named CMERNet, built upon an encoder-decoder architecture and trained on CMER-3M. Experimental results show that CMERNet, with only 125 million parameters, significantly outperforms existing MER models and MLLMs on CMER-Bench. 

---
# Exploring the Modular Integration of "AI + Architecture" Pedagogy in Undergraduate Design Education: A Case Study of Architectural Design III/IV Courses at Zhejiang University 

**Authors**: Wang Jiaqi, Lan Yi, Chen Xiang  

**Link**: [PDF](https://arxiv.org/pdf/2512.13730)  

**Abstract**: This study investigates AI integration in architectural education through a teaching experiment in Zhejiang University's 2024-25 grade three undergraduate design studio. Adopting a dual-module framework (20-hour AI training + embedded ethics discussions), the course introduced deep learning models, LLMs, AIGC, LoRA, and ComfyUI while maintaining the original curriculum structure, supported by dedicated technical instructors. Findings demonstrate the effectiveness of phased guidance, balanced technical-ethical approaches, and institutional support. The model improved students' digital skills and strategic cognition while addressing AI ethics, providing a replicable approach combining technical and critical learning in design education. 

---
# Composite Classifier-Free Guidance for Multi-Modal Conditioning in Wind Dynamics Super-Resolution 

**Authors**: Jacob Schnell, Aditya Makkar, Gunadi Gani, Aniket Srinivasan Ashok, Darren Lo, Mike Optis, Alexander Wong, Yuhao Chen  

**Link**: [PDF](https://arxiv.org/pdf/2512.13729)  

**Abstract**: Various weather modelling problems (e.g., weather forecasting, optimizing turbine placements, etc.) require ample access to high-resolution, highly accurate wind data. Acquiring such high-resolution wind data, however, remains a challenging and expensive endeavour. Traditional reconstruction approaches are typically either cost-effective or accurate, but not both. Deep learning methods, including diffusion models, have been proposed to resolve this trade-off by leveraging advances in natural image super-resolution. Wind data, however, is distinct from natural images, and wind super-resolvers often use upwards of 10 input channels, significantly more than the usual 3-channel RGB inputs in natural images. To better leverage a large number of conditioning variables in diffusion models, we present a generalization of classifier-free guidance (CFG) to multiple conditioning inputs. Our novel composite classifier-free guidance (CCFG) can be dropped into any pre-trained diffusion model trained with standard CFG dropout. We demonstrate that CCFG outputs are higher-fidelity than those from CFG on wind super-resolution tasks. We present WindDM, a diffusion model trained for industrial-scale wind dynamics reconstruction and leveraging CCFG. WindDM achieves state-of-the-art reconstruction quality among deep learning models and costs up to $1000\times$ less than classical methods. 

---
# CurvaDion: Curvature-Adaptive Distributed Orthonormalization 

**Authors**: Bhavesh Kumar, Roger Jin, Jeffrey Quesnelle  

**Link**: [PDF](https://arxiv.org/pdf/2512.13728)  

**Abstract**: As language models scale to trillions of parameters, distributed training across many GPUs becomes essential, yet gradient synchronization over high-bandwidth, low-latency networks remains a critical bottleneck. While recent methods like Dion reduce per-step communication through low-rank updates, they synchronize at every step regardless of the optimization landscape. We observe that synchronization requirements vary dramatically throughout training: workers naturally compute similar gradients in flat regions, making frequent synchronization redundant, while high-curvature regions require coordination to prevent divergence. We introduce CurvaDion, which uses Relative Maximum Momentum Change (RMMC) to detect high-curvature regions requiring synchronization. RMMC leverages momentum dynamics which are already computed during optimization as a computationally tractable proxy for directional curvature, adding only $\mathcal{O}(d)$ operations per layer. We establish theoretical connections between RMMC and loss curvature and demonstrate that CurvaDion achieves 99\% communication reduction while matching baseline convergence across models from 160M to 1.3B parameters. 

---
# Time-Constrained Recommendations: Reinforcement Learning Strategies for E-Commerce 

**Authors**: Sayak Chakrabarty, Souradip Pal  

**Link**: [PDF](https://arxiv.org/pdf/2512.13726)  

**Abstract**: Unlike traditional recommendation tasks, finite user time budgets introduce a critical resource constraint, requiring the recommender system to balance item relevance and evaluation cost. For example, in a mobile shopping interface, users interact with recommendations by scrolling, where each scroll triggers a list of items called slate. Users incur an evaluation cost - time spent assessing item features before deciding to click. Highly relevant items having higher evaluation costs may not fit within the user's time budget, affecting engagement. In this position paper, our objective is to evaluate reinforcement learning algorithms that learn patterns in user preferences and time budgets simultaneously, crafting recommendations with higher engagement potential under resource constraints. Our experiments explore the use of reinforcement learning to recommend items for users using Alibaba's Personalized Re-ranking dataset supporting slate optimization in e-commerce contexts. Our contributions include (i) a unified formulation of time-constrained slate recommendation modeled as Markov Decision Processes (MDPs) with budget-aware utilities; (ii) a simulation framework to study policy behavior on re-ranking data; and (iii) empirical evidence that on-policy and off-policy control can improve performance under tight time budgets than traditional contextual bandit-based methods. 

---
# Graph AI generates neurological hypotheses validated in molecular, organoid, and clinical systems 

**Authors**: Ayush Noori, Joaquín Polonuer, Katharina Meyer, Bogdan Budnik, Shad Morton, Xinyuan Wang, Sumaiya Nazeen, Yingnan He, Iñaki Arango, Lucas Vittor, Matthew Woodworth, Richard C. Krolewski, Michelle M. Li, Ninning Liu, Tushar Kamath, Evan Macosko, Dylan Ritter, Jalwa Afroz, Alexander B. H. Henderson, Lorenz Studer, Samuel G. Rodriques, Andrew White, Noa Dagan, David A. Clifton, George M. Church, Sudeshna Das, Jenny M. Tam, Vikram Khurana, Marinka Zitnik  

**Link**: [PDF](https://arxiv.org/pdf/2512.13724)  

**Abstract**: Neurological diseases are the leading global cause of disability, yet most lack disease-modifying treatments. We present PROTON, a heterogeneous graph transformer that generates testable hypotheses across molecular, organoid, and clinical systems. To evaluate PROTON, we apply it to Parkinson's disease (PD), bipolar disorder (BD), and Alzheimer's disease (AD). In PD, PROTON linked genetic risk loci to genes essential for dopaminergic neuron survival and predicted pesticides toxic to patient-derived neurons, including the insecticide endosulfan, which ranked within the top 1.29% of predictions. In silico screens performed by PROTON reproduced six genome-wide $\alpha$-synuclein experiments, including a split-ubiquitin yeast two-hybrid system (normalized enrichment score [NES] = 2.30, FDR-adjusted $p < 1 \times 10^{-4}$), an ascorbate peroxidase proximity labeling assay (NES = 2.16, FDR $< 1 \times 10^{-4}$), and a high-depth targeted exome sequencing study in 496 synucleinopathy patients (NES = 2.13, FDR $< 1 \times 10^{-4}$). In BD, PROTON predicted calcitriol as a candidate drug that reversed proteomic alterations observed in cortical organoids derived from BD patients. In AD, we evaluated PROTON predictions in health records from $n = 610,524$ patients at Mass General Brigham, confirming that five PROTON-predicted drugs were associated with reduced seven-year dementia risk (minimum hazard ratio = 0.63, 95% CI: 0.53-0.75, $p < 1 \times 10^{-7}$). PROTON generated neurological hypotheses that were evaluated across molecular, organoid, and clinical systems, defining a path for AI-driven discovery in neurological disease. 

---
# Made-in China, Thinking in America:U.S. Values Persist in Chinese LLMs 

**Authors**: David Haslett, Linus Ta-Lun Huang, Leila Khalatbari, Janet Hui-wen Hsiao, Antoni B. Chan  

**Link**: [PDF](https://arxiv.org/pdf/2512.13723)  

**Abstract**: As large language models increasingly mediate access to information and facilitate decision-making, they are becoming instruments in soft power competitions between global actors such as the United States and China. So far, language models seem to be aligned with the values of Western countries, but evidence for this ethical bias comes mostly from models made by American companies. The current crop of state-of-the-art models includes several made in China, so we conducted the first large-scale investigation of how models made in China and the USA align with people from China and the USA. We elicited responses to the Moral Foundations Questionnaire 2.0 and the World Values Survey from ten Chinese models and ten American models, and we compared their responses to responses from thousands of Chinese and American people. We found that all models respond to both surveys more like American people than like Chinese people. This skew toward American values is only slightly mitigated when prompting the models in Chinese or imposing a Chinese persona on the models. These findings have important implications for a near future in which large language models generate much of the content people consume and shape normative influence in geopolitics. 

---
# Federated Few-Shot Learning for Epileptic Seizure Detection Under Privacy Constraints 

**Authors**: Ekaterina Sysoykova, Bernhard Anzengruber-Tanase, Michael Haslgrubler, Philipp Seidl, Alois Ferscha  

**Link**: [PDF](https://arxiv.org/pdf/2512.13717)  

**Abstract**: Many deep learning approaches have been developed for EEG-based seizure detection; however, most rely on access to large centralized annotated datasets. In clinical practice, EEG data are often scarce, patient-specific distributed across institutions, and governed by strict privacy regulations that prohibit data pooling. As a result, creating usable AI-based seizure detection models remains challenging in real-world medical settings. To address these constraints, we propose a two-stage federated few-shot learning (FFSL) framework for personalized EEG-based seizure detection. The method is trained and evaluated on the TUH Event Corpus, which includes six EEG event classes. In Stage 1, a pretrained biosignal transformer (BIOT) is fine-tuned across non-IID simulated hospital sites using federated learning, enabling shared representation learning without centralizing EEG recordings. In Stage 2, federated few-shot personalization adapts the classifier to each patient using only five labeled EEG segments, retaining seizure-specific information while still benefiting from cross-site knowledge. Federated fine-tuning achieved a balanced accuracy of 0.43 (centralized: 0.52), Cohen's kappa of 0.42 (0.49), and weighted F1 of 0.69 (0.74). In the FFSL stage, client-specific models reached an average balanced accuracy of 0.77, Cohen's kappa of 0.62, and weighted F1 of 0.73 across four sites with heterogeneous event distributions. These results suggest that FFSL can support effective patient-adaptive seizure detection under realistic data-availability and privacy constraints. 

---
# Scaling and Transferability of Annealing Strategies in Large Language Model Training 

**Authors**: Siqi Wang, Zhengyu Chen, Teng Xiao, Zheqi Lv, Jinluan Yang, Xunliang Cai, Jingang Wang, Xiaomeng Li  

**Link**: [PDF](https://arxiv.org/pdf/2512.13705)  

**Abstract**: Learning rate scheduling is crucial for training large language models, yet understanding the optimal annealing strategies across different model configurations remains challenging. In this work, we investigate the transferability of annealing dynamics in large language model training and refine a generalized predictive framework for optimizing annealing strategies under the Warmup-Steady-Decay (WSD) scheduler. Our improved framework incorporates training steps, maximum learning rate, and annealing behavior, enabling more efficient optimization of learning rate schedules. Our work provides a practical guidance for selecting optimal annealing strategies without exhaustive hyperparameter searches, demonstrating that smaller models can serve as reliable proxies for optimizing the training dynamics of larger models. We validate our findings on extensive experiments using both Dense and Mixture-of-Experts (MoE) models, demonstrating that optimal annealing ratios follow consistent patterns and can be transferred across different training configurations. 

---
# Safe2Harm: Semantic Isomorphism Attacks for Jailbreaking Large Language Models 

**Authors**: Fan Yang  

**Link**: [PDF](https://arxiv.org/pdf/2512.13703)  

**Abstract**: Large Language Models (LLMs) have demonstrated exceptional performance across various tasks, but their security vulnerabilities can be exploited by attackers to generate harmful content, causing adverse impacts across various societal domains. Most existing jailbreak methods revolve around Prompt Engineering or adversarial optimization, yet we identify a previously overlooked phenomenon: many harmful scenarios are highly consistent with legitimate ones in terms of underlying principles. Based on this finding, this paper proposes the Safe2Harm Semantic Isomorphism Attack method, which achieves efficient jailbreaking through four stages: first, rewrite the harmful question into a semantically safe question with similar underlying principles; second, extract the thematic mapping relationship between the two; third, let the LLM generate a detailed response targeting the safe question; finally, reversely rewrite the safe response based on the thematic mapping relationship to obtain harmful output. Experiments on 7 mainstream LLMs and three types of benchmark datasets show that Safe2Harm exhibits strong jailbreaking capability, and its overall performance is superior to existing methods. Additionally, we construct a challenging harmful content evaluation dataset containing 358 samples and evaluate the effectiveness of existing harmful detection methods, which can be deployed for LLM input-output filtering to enable defense. 

---
# Enhancing Transparency and Traceability in Healthcare AI: The AI Product Passport 

**Authors**: A. Anil Sinaci, Senan Postaci, Dogukan Cavdaroglu, Machteld J. Boonstra, Okan Mercan, Kerem Yilmaz, Gokce B. Laleci Erturkmen, Folkert W. Asselbergs, Karim Lekadir  

**Link**: [PDF](https://arxiv.org/pdf/2512.13702)  

**Abstract**: Objective: To develop the AI Product Passport, a standards-based framework improving transparency, traceability, and compliance in healthcare AI via lifecycle-based documentation. Materials and Methods: The AI Product Passport was developed within the AI4HF project, focusing on heart failure AI tools. We analyzed regulatory frameworks (EU AI Act, FDA guidelines) and existing standards to design a relational data model capturing metadata across AI lifecycle phases: study definition, dataset preparation, model generation/evaluation, deployment/monitoring, and passport generation. MLOps/ModelOps concepts were integrated for operational relevance. Co-creation involved feedback from AI4HF consortium and a Lisbon workshop with 21 diverse stakeholders, evaluated via Mentimeter polls. The open-source platform was implemented with Python libraries for automated provenance tracking. Results: The AI Product Passport was designed based on existing standards and methods with well-defined lifecycle management and role-based access. Its implementation is a web-based platform with a relational data model supporting auditable documentation. It generates machine- and human-readable reports, customizable for stakeholders. It aligns with FUTURE-AI principles (Fairness, Universality, Traceability, Usability, Robustness, Explainability), ensuring fairness, traceability, and usability. Exported passports detail model purpose, data provenance, performance, and deployment context. GitHub-hosted backend/frontend codebases enhance accessibility. Discussion and Conclusion: The AI Product Passport addresses transparency gaps in healthcare AI, meeting regulatory and ethical demands. Its open-source nature and alignment with standards foster trust and adaptability. Future enhancements include FAIR data principles and FHIR integration for improved interoperability, promoting responsible AI deployment. 

---
# Writing in Symbiosis: Mapping Human Creative Agency in the AI Era 

**Authors**: Vivan Doshi, Mengyuan Li  

**Link**: [PDF](https://arxiv.org/pdf/2512.13697)  

**Abstract**: The proliferation of Large Language Models (LLMs) raises a critical question about what it means to be human when we share an increasingly symbiotic relationship with persuasive and creative machines. This paper examines patterns of human-AI coevolution in creative writing, investigating how human craft and agency are adapting alongside machine capabilities. We challenge the prevailing notion of stylistic homogenization by examining diverse patterns in longitudinal writing data. Using a large-scale corpus spanning the pre- and post-LLM era, we observe patterns suggestive of a "Dual-Track Evolution": thematic convergence around AI-related topics, coupled with structured stylistic differentiation. Our analysis reveals three emergent adaptation patterns: authors showing increased similarity to AI style, those exhibiting decreased similarity, and those maintaining stylistic stability while engaging with AI-related themes. This Creative Archetype Map illuminates how authorship is coevolving with AI, contributing to discussions about human-AI collaboration, detection challenges, and the preservation of creative diversity. 

---
# MultiBanAbs: A Comprehensive Multi-Domain Bangla Abstractive Text Summarization Dataset 

**Authors**: Md. Tanzim Ferdous, Naeem Ahsan Chowdhury, Prithwiraj Bhattacharjee  

**Link**: [PDF](https://arxiv.org/pdf/2511.19317)  

**Abstract**: This study developed a new Bangla abstractive summarization dataset to generate concise summaries of Bangla articles from diverse sources. Most existing studies in this field have concentrated on news articles, where journalists usually follow a fixed writing style. While such approaches are effective in limited contexts, they often fail to adapt to the varied nature of real-world Bangla texts. In today's digital era, a massive amount of Bangla content is continuously produced across blogs, newspapers, and social media. This creates a pressing need for summarization systems that can reduce information overload and help readers understand content more quickly. To address this challenge, we developed a dataset of over 54,000 Bangla articles and summaries collected from multiple sources, including blogs such as Cinegolpo and newspapers such as Samakal and The Business Standard. Unlike single-domain resources, our dataset spans multiple domains and writing styles. It offers greater adaptability and practical relevance. To establish strong baselines, we trained and evaluated this dataset using several deep learning and transfer learning models, including LSTM, BanglaT5-small, and MTS-small. The results highlight its potential as a benchmark for future research in Bangla natural language processing. This dataset provides a solid foundation for building robust summarization systems and helps expand NLP resources for low-resource languages. 

---
# EDGC: Entropy-driven Dynamic Gradient Compression for Efficient LLM Training 

**Authors**: Qingao Yi, Jiaang Duan, Hanwen Hu, Qin Hua, Haiyan Zhao, Shiyou Qian, Dingyu Yang, Jian Cao, Jinghua Tang, Yinghao Yu, Chenzhi Liao, Kangjin Wang, Liping Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2511.10333)  

**Abstract**: Training large language models (LLMs) poses significant challenges regarding computational resources and memory capacity. Although distributed training techniques help mitigate these issues, they still suffer from considerable communication overhead. Existing approaches primarily rely on static gradient compression to enhance communication efficiency; however, these methods neglect the dynamic nature of evolving gradients during training, leading to performance degradation. Accelerating LLM training via compression without sacrificing performance remains a challenge. In this paper, we propose an entropy-driven dynamic gradient compression framework called EDGC. The core concept is to adjust the compression rate during LLM training based on the evolving trends of gradient entropy, taking into account both compression efficiency and error. EDGC consists of three key this http URL, it employs a down-sampling method to efficiently estimate gradient entropy, reducing computation overhead. Second, it establishes a theoretical model linking compression rate with gradient entropy, enabling more informed compression decisions. Lastly, a window-based adjustment mechanism dynamically adapts the compression rate across pipeline stages, improving communication efficiency and maintaining model performance. We implemented EDGC on a 32-NVIDIA-V100 cluster and a 64-NVIDIA-H100 cluster to train GPT2-2.5B and GPT2-12.1B, respectively. The results show that EDGC significantly reduces communication latency and training time by up to 46.45% and 16.13% while preserving LLM accuracy. 

---
