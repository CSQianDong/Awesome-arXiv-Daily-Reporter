# Hybrid Retrieval-Augmented Generation Agent for Trustworthy Legal Question Answering in Judicial Forensics 

**Authors**: Yueqing Xi, Yifan Bai, Huasen Luo, Weiliang Wen, Hui Liu, Haoliang Li  

**Link**: [PDF](https://arxiv.org/pdf/2511.01668)  

**Abstract**: As artificial intelligence permeates judicial forensics, ensuring the veracity and traceability of legal question answering (QA) has become critical. Conventional large language models (LLMs) are prone to hallucination, risking misleading guidance in legal consultation, while static knowledge bases struggle to keep pace with frequently updated statutes and case law. We present a hybrid legal QA agent tailored for judicial settings that integrates retrieval-augmented generation (RAG) with multi-model ensembling to deliver reliable, auditable, and continuously updatable counsel. The system prioritizes retrieval over generation: when a trusted legal repository yields relevant evidence, answers are produced via RAG; otherwise, multiple LLMs generate candidates that are scored by a specialized selector, with the top-ranked answer returned. High-quality outputs then undergo human review before being written back to the repository, enabling dynamic knowledge evolution and provenance tracking. Experiments on the Law\_QA dataset show that our hybrid approach significantly outperforms both a single-model baseline and a vanilla RAG pipeline on F1, ROUGE-L, and an LLM-as-a-Judge metric. Ablations confirm the complementary contributions of retrieval prioritization, model ensembling, and the human-in-the-loop update mechanism. The proposed system demonstrably reduces hallucination while improving answer quality and legal compliance, advancing the practical landing of media forensics technologies in judicial scenarios. 

---
# ExplicitLM: Decoupling Knowledge from Parameters via Explicit Memory Banks 

**Authors**: Chengzhang Yu, Zening Lu, Chenyang Zheng, Chiyue Wang, Yiming Zhang, Zhanpeng Jin  

**Link**: [PDF](https://arxiv.org/pdf/2511.01581)  

**Abstract**: Large language models suffer from knowledge staleness and lack of interpretability due to implicit knowledge storage across entangled network parameters, preventing targeted updates and reasoning transparency. We propose ExplicitLM, a novel architecture featuring a million-scale external memory bank storing human-readable knowledge as token sequences, enabling direct inspection and modification. We design a differentiable two-stage retrieval mechanism with efficient coarse-grained filtering via product key decomposition (reducing complexity from $\mathcal{O}(N \cdot |I|)$ to $\mathcal{O}(\sqrt{N} \cdot |I|)$) and fine-grained Gumbel-Softmax matching for end-to-end training. Inspired by dual-system cognitive theory, we partition knowledge into frozen explicit facts (20%) and learnable implicit patterns (80%), maintained through Exponential Moving Average updates for stability. ExplicitLM achieves up to 43.67% improvement on knowledge-intensive tasks versus standard Transformers, with 3.62$\times$ gains in low-data regimes (10k samples). Analysis shows strong correlations between memory retrieval and performance, with correct predictions achieving 49% higher hit rates. Unlike RAG systems with frozen retrieval, our jointly optimized architecture demonstrates that interpretable, updatable models can maintain competitive performance while providing unprecedented knowledge transparency. 

---
# Automatic Minds: Cognitive Parallels Between Hypnotic States and Large Language Model Processing 

**Authors**: Giuseppe Riva, Brenda K. Wiederhold, Fabrizia Mantovani  

**Link**: [PDF](https://arxiv.org/pdf/2511.01363)  

**Abstract**: The cognitive processes of the hypnotized mind and the computational operations of large language models (LLMs) share deep functional parallels. Both systems generate sophisticated, contextually appropriate behavior through automatic pattern-completion mechanisms operating with limited or unreliable executive oversight. This review examines this convergence across three principles: automaticity, in which responses emerge from associative rather than deliberative processes; suppressed monitoring, leading to errors such as confabulation in hypnosis and hallucination in LLMs; and heightened contextual dependency, where immediate cues (for example, the suggestion of a therapist or the prompt of the user) override stable knowledge.
These mechanisms reveal an observer-relative meaning gap: both systems produce coherent but ungrounded outputs that require an external interpreter to supply meaning. Hypnosis and LLMs also exemplify functional agency - the capacity for complex, goal-directed, context-sensitive behavior - without subjective agency, the conscious awareness of intention and ownership that defines human action. This distinction clarifies how purposive behavior can emerge without self-reflective consciousness, governed instead by structural and contextual dynamics. Finally, both domains illuminate the phenomenon of scheming: automatic, goal-directed pattern generation that unfolds without reflective awareness. Hypnosis provides an experimental model for understanding how intention can become dissociated from conscious deliberation, offering insights into the hidden motivational dynamics of artificial systems. Recognizing these parallels suggests that the future of reliable AI lies in hybrid architectures that integrate generative fluency with mechanisms of executive monitoring, an approach inspired by the complex, self-regulating architecture of the human mind. 

---
# Align to Misalign: Automatic LLM Jailbreak with Meta-Optimized LLM Judges 

**Authors**: Hamin Koo, Minseon Kim, Jaehyung Kim  

**Link**: [PDF](https://arxiv.org/pdf/2511.01375)  

**Abstract**: Identifying the vulnerabilities of large language models (LLMs) is crucial for improving their safety by addressing inherent weaknesses. Jailbreaks, in which adversaries bypass safeguards with crafted input prompts, play a central role in red-teaming by probing LLMs to elicit unintended or unsafe behaviors. Recent optimization-based jailbreak approaches iteratively refine attack prompts by leveraging LLMs. However, they often rely heavily on either binary attack success rate (ASR) signals, which are sparse, or manually crafted scoring templates, which introduce human bias and uncertainty in the scoring outcomes. To address these limitations, we introduce AMIS (Align to MISalign), a meta-optimization framework that jointly evolves jailbreak prompts and scoring templates through a bi-level structure. In the inner loop, prompts are refined using fine-grained and dense feedback using a fixed scoring template. In the outer loop, the template is optimized using an ASR alignment score, gradually evolving to better reflect true attack outcomes across queries. This co-optimization process yields progressively stronger jailbreak prompts and more calibrated scoring signals. Evaluations on AdvBench and JBB-Behaviors demonstrate that AMIS achieves state-of-the-art performance, including 88.0% ASR on Claude-3.5-Haiku and 100.0% ASR on Claude-4-Sonnet, outperforming existing baselines by substantial margins. 

---
# llmSHAP: A Principled Approach to LLM Explainability 

**Authors**: Filip Naudot, Tobias Sundqvist, Timotheus Kampik  

**Link**: [PDF](https://arxiv.org/pdf/2511.01311)  

**Abstract**: Feature attribution methods help make machine learning-based inference explainable by determining how much one or several features have contributed to a model's output. A particularly popular attribution method is based on the Shapley value from cooperative game theory, a measure that guarantees the satisfaction of several desirable principles, assuming deterministic inference. We apply the Shapley value to feature attribution in large language model (LLM)-based decision support systems, where inference is, by design, stochastic (non-deterministic). We then demonstrate when we can and cannot guarantee Shapley value principle satisfaction across different implementation variants applied to LLM-based decision support, and analyze how the stochastic nature of LLMs affects these guarantees. We also highlight trade-offs between explainable inference speed, agreement with exact Shapley value attributions, and principle attainment. 

---
# Simulating Environments with Reasoning Models for Agent Training 

**Authors**: Yuetai Li, Huseyin A Inan, Xiang Yue, Wei-Ning Chen, Lukas Wutschitz, Janardhan Kulkarni, Radha Poovendran, Robert Sim, Saravan Rajmohan  

**Link**: [PDF](https://arxiv.org/pdf/2511.01824)  

**Abstract**: LLM agents excel in compact environments requiring deep reasoning but remain brittle when operating in broader, more complex contexts that demand robustness across diverse tools and schemas. Building bespoke environments for training is heavy, brittle, and limits progress. In this paper, we demonstrate that LLMs can simulate realistic environment feedback without access to actual testbed data or APIs. Inspired by this capability, we propose two frameworks: Simia-SFT, a pipeline that synthesizes SFT data by amplifying small seed sets into diverse trajectories in an environment-agnostic manner, and Simia-RL, a framework that enables RL training without real environment implementations through LLM-simulated feedback. Fine-tuning open models yields consistent improvements across multiple benchmarks, surpassing GPT-4o and approaching o4-mini on $\tau^2$-Bench. Together, Simia-SFT and Simia-RL enable scalable agent training without environment engineering, replacing heavy and brittle implementations with flexible LLM-based simulation. 

---
# QiMeng-NeuComBack: Self-Evolving Translation from IR to Assembly Code 

**Authors**: Hainan Fang, Yuanbo Wen, Jun Bi, Yihan Wang, Tonghui He, Yanlin Tang, Di Huang, Jiaming Guo, Rui Zhang, Qi Guo, Yunji Chen  

**Link**: [PDF](https://arxiv.org/pdf/2511.01183)  

**Abstract**: Compilers, while essential, are notoriously complex systems that demand prohibitively expensive human expertise to develop and maintain. The recent advancements in Large Language Models (LLMs) offer a compelling new paradigm: Neural Compilation, which could potentially simplify compiler development for new architectures and facilitate the discovery of innovative optimization techniques. However, several critical obstacles impede its practical adoption. Firstly, a significant lack of dedicated benchmarks and robust evaluation methodologies hinders objective assessment and tracking of progress in the field. Secondly, systematically enhancing the reliability and performance of LLM-generated assembly remains a critical challenge. Addressing these challenges, this paper introduces NeuComBack, a novel benchmark dataset specifically designed for IR-to-assembly compilation. Leveraging this dataset, we first define a foundational Neural Compilation workflow and conduct a comprehensive evaluation of the capabilities of recent frontier LLMs on Neural Compilation, establishing new performance baselines. We further propose a self-evolving prompt optimization method that enables LLMs to iteratively evolve their internal prompt strategies by extracting insights from prior self-debugging traces, thereby enhancing their neural compilation capabilities. Experiments demonstrate that our method significantly improves both the functional correctness and the performance of LLM-generated assembly code. Compared to baseline prompts, the functional correctness rates improved from 44% to 64% on x86_64 and from 36% to 58% on aarch64, respectively. More significantly, among the 16 correctly generated x86_64 programs using our method, 14 (87.5%) surpassed clang-O3 performance. 

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
# How Focused Are LLMs? A Quantitative Study via Repetitive Deterministic Prediction Tasks 

**Authors**: Wanda Hou, Leon Zhou, Hong-Ye Hu, Yi-Zhuang You, Xiao-Liang Qi  

**Link**: [PDF](https://arxiv.org/pdf/2511.00763)  

**Abstract**: We investigate the performance of large language models on repetitive deterministic prediction tasks and study how the sequence accuracy rate scales with output length. Each such task involves repeating the same operation n times. Examples include letter replacement in strings following a given rule, integer addition, and multiplication of string operators in many body quantum mechanics. If the model performs the task through a simple repetition algorithm, the success rate should decay exponentially with sequence length. In contrast, our experiments on leading large language models reveal a sharp double exponential drop beyond a characteristic length scale, forming an accuracy cliff that marks the transition from reliable to unstable generation. This indicates that the models fail to execute each operation independently. To explain this phenomenon, we propose a statistical physics inspired model that captures the competition between external conditioning from the prompt and internal interference among generated tokens. The model quantitatively reproduces the observed crossover and provides an interpretable link between attention induced interference and sequence level failure. Fitting the model to empirical results across multiple models and tasks yields effective parameters that characterize the intrinsic error rate and error accumulation factor for each model task pair, offering a principled framework for understanding the limits of deterministic accuracy in large language models. 

---
# Efficient Test-Time Retrieval Augmented Generation 

**Authors**: Hailong Yin, Bin Zhu, Jingjing Chen, Chong-Wah Ngo  

**Link**: [PDF](https://arxiv.org/pdf/2511.01059)  

**Abstract**: Although Large Language Models (LLMs) demonstrate significant capabilities, their reliance on parametric knowledge often leads to inaccuracies. Retrieval Augmented Generation (RAG) mitigates this by incorporating external knowledge, but these methods may introduce irrelevant retrieved documents, leading to inaccurate responses. While the integration methods filter out incorrect answers from multiple responses, but lack external knowledge like RAG methods, and their high costs require balancing overhead with performance gains. To address these issues, we propose an Efficient Test-Time Retrieval-Augmented Generation Framework named ET2RAG to improve the performance of LLMs while maintaining efficiency. Specifically, ET2RAG is a training-free method, that first retrieves the most relevant documents and augments the LLMs to efficiently generate diverse candidate responses by managing response length. Then we compute the similarity of candidate responses and employ a majority voting mechanism to select the most suitable response as the final output. In particular, we discover that partial generation is sufficient to capture the key information necessary for consensus calculation, allowing us to effectively perform majority voting without the need for fully generated responses. Thus, we can reach a balance between computational cost and performance by managing the response length for the number of retrieved documents for majority voting. Experimental results demonstrate that ET2RAG significantly enhances performance across three tasks, including open-domain question answering, recipe generation and image captioning. 

---
# Do Math Reasoning LLMs Help Predict the Impact of Public Transit Events? 

**Authors**: Bowen Fang, Ruijian Zha, Xuan Di  

**Link**: [PDF](https://arxiv.org/pdf/2511.00808)  

**Abstract**: Predicting public transit incident duration from unstructured text alerts is a critical but challenging task. Addressing the domain sparsity of transit operations with standard Supervised Fine-Tuning (SFT) is difficult, as the task involves noisy, continuous labels and lacks reliable expert demonstrations for reasoning. While Reinforcement Learning from Verifiable Rewards (RLVR) excels at tasks with binary correctness, like mathematics, its applicability to noisy, continuous forecasting is an open question. This work, to our knowledge, is the first to bridge the gap between RLVR LLM training with the critical, real-world forecasting challenges in public transit operations. We adapt RLVR to this task by introducing a tolerance-based, shaped reward function that grants partial credit within a continuous error margin, rather than demanding a single correct answer. We systematically evaluate this framework on a curated dataset of NYC MTA service alerts. Our findings show that general-purpose, instruction-tuned LLMs significantly outperform specialized math-reasoning models, which struggle with the ambiguous, real-world text. We empirically demonstrate that the binary reward is unstable and degrades performance, whereas our shaped reward design is critical and allows our model to dominate on the most challenging metrics. While classical regressors are superior at minimizing overall MAE or MSE, our RLVR approach achieved a 35\% relative improvement in 5-minute accuracy (Acc@5) over the strongest baseline. This demonstrates that RLVR can be successfully adapted to real-world, noisy forecasting, but requires a verifier design that reflects the continuous nature of the problem. 

---
# DTS: Enhancing Large Reasoning Models via Decoding Tree Sketching 

**Authors**: Zicheng Xu, Guanchu Wang, Yu-Neng Chuang, Guangyao Zheng, Alexander S. Szalay, Zirui Liu, Vladimir Braverman  

**Link**: [PDF](https://arxiv.org/pdf/2511.00640)  

**Abstract**: Large Reasoning Models (LRMs) demonstrate strong performance on complex reasoning tasks, yet they often suffer from overthinking, producing excessively long chain-of-thought (CoT) traces that increase inference cost and may degrade accuracy. Our analysis reveals a clear anti-correlation between reasoning length and accuracy, where across multiple stochastic decodes, the short reasoning paths consistently achieve the highest correctness, while longer ones accumulate errors and repetitions. These short optimal reasoning paths can be found ideally through full enumeration of the reasoning space. However, the tree-structured reasoning space grows exponentially with sequence length, rendering exhaustive exploration infeasible. To address this, we propose DTS, a model-agnostic decoding framework that sketches the reasoning space by selectively branching at high-entropy tokens and applies early stopping to select the shortest completed reasoning path. This approach approximates the optimal solution that enhances both efficiency and accuracy, without requiring additional training or supervision. Experiments on AIME2024 and AIME2025 datasets with DeepSeek-R1-Distill-Qwen-7B and 1.5B show that DTS improves accuracy by up to 8%, reduces average reasoning length by 23%, and decreases repetition frequency by 12%, demonstrating DTS's ability for scalable and efficient LRM reasoning. 

---
# GraphChain: Large Language Models for Large-scale Graph Analysis via Tool Chaining 

**Authors**: Chunyu Wei, Wenji Hu, Xingjia Hao, Xin Wang, Yifan Yang, Yueguo Chen, Yang Tian, Yunhai Wang  

**Link**: [PDF](https://arxiv.org/pdf/2511.00457)  

**Abstract**: Large Language Models (LLMs) face significant limitations when applied to large-scale graphs, struggling with context constraints and inflexible reasoning. We present GraphChain, a framework that enables LLMs to analyze complex graphs through dynamic sequences of specialized tools, mimicking human exploratory intelligence. Our approach introduces two key innovations: (1) Progressive Graph Distillation, a reinforcement learning mechanism that generates optimized tool sequences balancing task relevance with information compression, and (2) Structure-aware Test-Time Adaptation, which efficiently tailors tool selection strategies to diverse graph topologies using spectral properties and lightweight adapters without costly retraining. Experiments show GraphChain significantly outperforms prior methods, enabling scalable and adaptive LLM-driven graph analysis. 

---
# Knowledge Elicitation with Large Language Models for Interpretable Cancer Stage Identification from Pathology Reports 

**Authors**: Yeawon Lee, Christopher C. Yang, Chia-Hsuan Chang, Grace Lu-Yao  

**Link**: [PDF](https://arxiv.org/pdf/2511.01052)  

**Abstract**: Cancer staging is critical for patient prognosis and treatment planning, yet extracting pathologic TNM staging from unstructured pathology reports poses a persistent challenge. Existing natural language processing (NLP) and machine learning (ML) strategies often depend on large annotated datasets, limiting their scalability and adaptability. In this study, we introduce two Knowledge Elicitation methods designed to overcome these limitations by enabling large language models (LLMs) to induce and apply domain-specific rules for cancer staging. The first, Knowledge Elicitation with Long-Term Memory (KEwLTM), uses an iterative prompting strategy to derive staging rules directly from unannotated pathology reports, without requiring ground-truth labels. The second, Knowledge Elicitation with Retrieval-Augmented Generation (KEwRAG), employs a variation of RAG where rules are pre-extracted from relevant guidelines in a single step and then applied, enhancing interpretability and avoiding repeated retrieval overhead. We leverage the ability of LLMs to apply broad knowledge learned during pre-training to new tasks. Using breast cancer pathology reports from the TCGA dataset, we evaluate their performance in identifying T and N stages, comparing them against various baseline approaches on two open-source LLMs. Our results indicate that KEwLTM outperforms KEwRAG when Zero-Shot Chain-of-Thought (ZSCOT) inference is effective, whereas KEwRAG achieves better performance when ZSCOT inference is less effective. Both methods offer transparent, interpretable interfaces by making the induced rules explicit. These findings highlight the promise of our Knowledge Elicitation methods as scalable, high-performing solutions for automated cancer staging with enhanced interpretability, particularly in clinical settings with limited annotated data. 

---
# Better Call CLAUSE: A Discrepancy Benchmark for Auditing LLMs Legal Reasoning Capabilities 

**Authors**: Manan Roy Choudhury, Adithya Chandramouli, Mannan Anand, Vivek Gupta  

**Link**: [PDF](https://arxiv.org/pdf/2511.00340)  

**Abstract**: The rapid integration of large language models (LLMs) into high-stakes legal work has exposed a critical gap: no benchmark exists to systematically stress-test their reliability against the nuanced, adversarial, and often subtle flaws present in real-world contracts. To address this, we introduce CLAUSE, a first-of-its-kind benchmark designed to evaluate the fragility of an LLM's legal reasoning. We study the capabilities of LLMs to detect and reason about fine-grained discrepancies by producing over 7500 real-world perturbed contracts from foundational datasets like CUAD and ContractNLI. Our novel, persona-driven pipeline generates 10 distinct anomaly categories, which are then validated against official statutes using a Retrieval-Augmented Generation (RAG) system to ensure legal fidelity. We use CLAUSE to evaluate leading LLMs' ability to detect embedded legal flaws and explain their significance. Our analysis shows a key weakness: these models often miss subtle errors and struggle even more to justify them legally. Our work outlines a path to identify and correct such reasoning failures in legal AI. 

---
# Engineering.ai: A Platform for Teams of AI Engineers in Computational Design 

**Authors**: Ran Xu, Yupeng Qi, Jingsen Feng, Xu Chu  

**Link**: [PDF](https://arxiv.org/pdf/2511.00122)  

**Abstract**: In modern engineering practice, human engineers collaborate in specialized teams to design complex products, with each expert completing their respective tasks while communicating and exchanging results and data with one another. While this division of expertise is essential for managing multidisciplinary complexity, it demands substantial development time and cost. Recently, we introduced OpenFOAMGPT (1.0, 2.0), which functions as an autonomous AI engineer for computational fluid dynamics, and this http URL, which can conduct end-to-end research in fluid mechanics draft publications and PhD theses. Building upon these foundations, we present this http URL, a platform for teams of AI engineers in computational design. The framework employs a hierarchical multi-agent architecture where a Chief Engineer coordinates specialized agents consisting of Aerodynamics, Structural, Acoustic, and Optimization Engineers, each powered by LLM with domain-specific knowledge. Agent-agent collaboration is achieved through file-mediated communication for data provenance and reproducibility, while a comprehensive memory system maintains project context, execution history, and retrieval-augmented domain knowledge to ensure reliable decision-making across the workflow. The system integrates FreeCAD, Gmsh, OpenFOAM, CalculiX, and BPM acoustic analysis, enabling parallel multidisciplinary simulations while maintaining computational accuracy. The framework is validated through UAV wing optimization. This work demonstrates that agentic-AI-enabled AI engineers has the potential to perform complex engineering tasks autonomously. Remarkably, the automated workflow achieved a 100% success rate across over 400 parametric configurations, with zero mesh generation failures, solver convergence issues, or manual interventions required, validating that the framework is trustworthy. 

---
# Reimagining Safety Alignment with An Image 

**Authors**: Yifan Xia, Guorui Chen, Wenqian Yu, Zhijiang Li, Philip Torr, Jindong Gu  

**Link**: [PDF](https://arxiv.org/pdf/2511.00509)  

**Abstract**: Large language models (LLMs) excel in diverse applications but face dual challenges: generating harmful content under jailbreak attacks and over-refusal of benign queries due to rigid safety mechanisms. These issues are further complicated by the need to accommodate different value systems and precisely align with given safety preferences. Moreover, traditional methods like SFT and RLHF lack this capability due to their costly parameter tuning requirements and inability to support multiple value systems within a single model. These problems are more obvious in multimodal large language models (MLLMs), especially in terms of heightened over-refusal in cross-modal tasks and new security risks arising from expanded attack surfaces. We propose Magic Image, an optimization-driven visual prompt framework that enhances security while reducing over-refusal. By optimizing image prompts using harmful/benign samples, our method enables a single model to adapt to different value systems and better align with given safety preferences without parameter updates. Experiments demonstrate improved safety-effectiveness balance across diverse datasets while preserving model performance, offering a practical solution for deployable MLLM safety alignment. 

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
# Towards Robust Mathematical Reasoning 

**Authors**: Thang Luong, Dawsen Hwang, Hoang H. Nguyen, Golnaz Ghiasi, Yuri Chervonyi, Insuk Seo, Junsu Kim, Garrett Bingham, Jonathan Lee, Swaroop Mishra, Alex Zhai, Clara Huiyi Hu, Henryk Michalewski, Jimin Kim, Jeonghyun Ahn, Junhwi Bae, Xingyou Song, Trieu H. Trinh, Quoc V. Le, Junehyuk Jung  

**Link**: [PDF](https://arxiv.org/pdf/2511.01846)  

**Abstract**: Finding the right north-star metrics is highly critical for advancing the mathematical reasoning capabilities of foundation models, especially given that existing evaluations are either too easy or only focus on getting correct short answers. To address these issues, we present IMO-Bench, a suite of advanced reasoning benchmarks, vetted by a panel of top specialists and that specifically targets the level of the International Mathematical Olympiad (IMO), the most prestigious venue for young mathematicians. IMO-AnswerBench first tests models on 400 diverse Olympiad problems with verifiable short answers. IMO-Proof Bench is the next-level evaluation for proof-writing capabilities, which includes both basic and advanced IMO level problems as well as detailed grading guidelines to facilitate automatic grading. These benchmarks played a crucial role in our historic achievement of the gold-level performance at IMO 2025 with Gemini Deep Think (Luong and Lockhart, 2025). Our model achieved 80.0% on IMO-AnswerBench and 65.7% on the advanced IMO-Proof Bench, surpassing the best non-Gemini models by large margins of 6.9% and 42.4% respectively. We also showed that autograders built with Gemini reasoning correlate well with human evaluations and construct IMO-GradingBench, with 1000 human gradings on proofs, to enable further progress in automatic evaluation of long-form answers. We hope that IMO-Bench will help the community towards advancing robust mathematical reasoning and release it at this https URL. 

---
# Advancing Cognitive Science with LLMs 

**Authors**: Dirk U. Wulff, Rui Mata  

**Link**: [PDF](https://arxiv.org/pdf/2511.00206)  

**Abstract**: Cognitive science faces ongoing challenges in knowledge synthesis and conceptual clarity, in part due to its multifaceted and interdisciplinary nature. Recent advances in artificial intelligence, particularly the development of large language models (LLMs), offer tools that may help to address these issues. This review examines how LLMs can support areas where the field has historically struggled, including establishing cross-disciplinary connections, formalizing theories, developing clear measurement taxonomies, achieving generalizability through integrated modeling frameworks, and capturing contextual and individual variation. We outline the current capabilities and limitations of LLMs in these domains, including potential pitfalls. Taken together, we conclude that LLMs can serve as tools for a more integrative and cumulative cognitive science when used judiciously to complement, rather than replace, human expertise. 

---
# Plan-and-Write: Structure-Guided Length Control for LLMs without Model Retraining 

**Authors**: Adewale Akinfaderin, Shreyas Subramanian, Akarsha Sehwag  

**Link**: [PDF](https://arxiv.org/pdf/2511.01807)  

**Abstract**: Length control in Large Language Models (LLMs) is a crucial but under-addressed challenge, with applications ranging from voice interfaces requiring concise responses to research summaries needing comprehensive outputs. Current approaches to length control, including Regularized DPO, Length-Instruction Fine Tuning, and tool-augmented methods, typically require expensive model retraining or complex inference-time tooling. This paper presents a prompt engineering methodology that enables precise length control without model retraining. Our structure-guided approach implements deliberate planning and word counting mechanisms within the prompt, encouraging the model to carefully track and adhere to specified length constraints. Comprehensive evaluations across six state-of-the-art LLMs demonstrate that our method significantly improves length fidelity for several models compared to standard prompting when applied to document summarization tasks, particularly for shorter-to-medium length constraints. The proposed technique shows varying benefits across different model architectures, with some models demonstrating up to 37.6% improvement in length adherence. Quality evaluations further reveal that our approach maintains or enhances overall output quality compared to standard prompting techniques. Our approach provides an immediately deployable solution for applications requiring precise length control, particularly valuable for production environments where model retraining is impractical or cost-prohibitive. 

---
# SmartMLOps Studio: Design of an LLM-Integrated IDE with Automated MLOps Pipelines for Model Development and Monitoring 

**Authors**: Jiawei Jin, Yingxin Su, Xiaotong Zhu  

**Link**: [PDF](https://arxiv.org/pdf/2511.01850)  

**Abstract**: The rapid expansion of artificial intelligence and machine learning (ML) applications has intensified the demand for integrated environments that unify model development, deployment, and monitoring. Traditional Integrated Development Environments (IDEs) focus primarily on code authoring, lacking intelligent support for the full ML lifecycle, while existing MLOps platforms remain detached from the coding workflow. To address this gap, this study proposes the design of an LLM-Integrated IDE with automated MLOps pipelines that enables continuous model development and monitoring within a single environment. The proposed system embeds a Large Language Model (LLM) assistant capable of code generation, debugging recommendation, and automatic pipeline configuration. The backend incorporates automated data validation, feature storage, drift detection, retraining triggers, and CI/CD deployment orchestration. This framework was implemented in a prototype named SmartMLOps Studio and evaluated using classification and forecasting tasks on the UCI Adult and M5 datasets. Experimental results demonstrate that SmartMLOps Studio reduces pipeline configuration time by 61%, improves experiment reproducibility by 45%, and increases drift detection accuracy by 14% compared to traditional workflows. By bridging intelligent code assistance and automated operational pipelines, this research establishes a novel paradigm for AI engineering - transforming the IDE from a static coding tool into a dynamic, lifecycle-aware intelligent platform for scalable and efficient model development. 

---
# Accumulating Context Changes the Beliefs of Language Models 

**Authors**: Jiayi Geng, Howard Chen, Ryan Liu, Manoel Horta Ribeiro, Robb Willer, Graham Neubig, Thomas L. Griffiths  

**Link**: [PDF](https://arxiv.org/pdf/2511.01805)  

**Abstract**: Language model (LM) assistants are increasingly used in applications such as brainstorming and research. Improvements in memory and context size have allowed these models to become more autonomous, which has also resulted in more text accumulation in their context windows without explicit user intervention. This comes with a latent risk: the belief profiles of models -- their understanding of the world as manifested in their responses or actions -- may silently change as context accumulates. This can lead to subtly inconsistent user experiences, or shifts in behavior that deviate from the original alignment of the models. In this paper, we explore how accumulating context by engaging in interactions and processing text -- talking and reading -- can change the beliefs of language models, as manifested in their responses and this http URL results reveal that models' belief profiles are highly malleable: GPT-5 exhibits a 54.7% shift in its stated beliefs after 10 rounds of discussion about moral dilemmas and queries about safety, while Grok 4 shows a 27.2% shift on political issues after reading texts from the opposing position. We also examine models' behavioral changes by designing tasks that require tool use, where each tool selection corresponds to an implicit belief. We find that these changes align with stated belief shifts, suggesting that belief shifts will be reflected in actual behavior in agentic systems. Our analysis exposes the hidden risk of belief shift as models undergo extended sessions of talking or reading, rendering their opinions and actions unreliable. 

---
# Multi-Step Knowledge Interaction Analysis via Rank-2 Subspace Disentanglement 

**Authors**: Sekh Mainul Islam, Pepa Atanasova, Isabelle Augenstein  

**Link**: [PDF](https://arxiv.org/pdf/2511.01706)  

**Abstract**: Natural Language Explanations (NLEs) describe how Large Language Models (LLMs) make decisions, drawing on both external Context Knowledge (CK) and Parametric Knowledge (PK) stored in model weights. Understanding their interaction is key to assessing the grounding of NLEs, yet it remains underexplored. Prior work has largely examined only single-step generation, typically the final answer, and has modelled PK and CK interaction only as a binary choice in a rank-1 subspace. This overlooks richer forms of interaction, such as complementary or supportive knowledge. We propose a novel rank-2 projection subspace that disentangles PK and CK contributions more accurately and use it for the first multi-step analysis of knowledge interactions across longer NLE sequences. Experiments on four QA datasets and three open-weight instruction-tuned LLMs show that diverse knowledge interactions are poorly represented in a rank-1 subspace but are effectively captured in our rank-2 formulation. Our multi-step analysis reveals that hallucinated NLEs align strongly with the PK direction, context-faithful ones balance PK and CK, and Chain-of-Thought prompting for NLEs shifts generated NLEs toward CK by reducing PK reliance. This work provides the first framework for systematic studies of multi-step knowledge interactions in LLMs through a richer rank-2 subspace disentanglement. Code and data: this https URL. 

---
# Scam Shield: Multi-Model Voting and Fine-Tuned LLMs Against Adversarial Attacks 

**Authors**: Chen-Wei Chang, Shailik Sarkar, Hossein Salemi, Hyungmin Kim, Shutonu Mitra, Hemant Purohit, Fengxiu Zhang, Michin Hong, Jin-Hee Cho, Chang-Tien Lu  

**Link**: [PDF](https://arxiv.org/pdf/2511.01746)  

**Abstract**: Scam detection remains a critical challenge in cybersecurity as adversaries craft messages that evade automated filters. We propose a Hierarchical Scam Detection System (HSDS) that combines a lightweight multi-model voting front end with a fine-tuned LLaMA 3.1 8B Instruct back end to improve accuracy and robustness against adversarial attacks. An ensemble of four classifiers provides preliminary predictions through majority vote, and ambiguous cases are escalated to the fine-tuned model, which is optimized with adversarial training to reduce misclassification. Experiments show that this hierarchical design both improves adversarial scam detection and shortens inference time by routing most cases away from the LLM, outperforming traditional machine-learning baselines and proprietary LLM baselines. The findings highlight the effectiveness of a hybrid voting mechanism and adversarial fine-tuning in fortifying LLMs against evolving scam tactics, enhancing the resilience of automated scam detection systems. 

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
# Open Character Training: Shaping the Persona of AI Assistants through Constitutional AI 

**Authors**: Sharan Maiya, Henning Bartsch, Nathan Lambert, Evan Hubinger  

**Link**: [PDF](https://arxiv.org/pdf/2511.01689)  

**Abstract**: The character of the "AI assistant" persona generated by modern chatbot large language models influences both surface-level behavior and apparent values, beliefs, and ethics. These all affect interaction quality, perceived intelligence, and alignment with both developer and user intentions. The shaping of this persona, known as character training, is a critical component of industry post-training, yet remains effectively unstudied in the academic literature. We introduce the first open implementation of character training, leveraging Constitutional AI and a new data pipeline using synthetic introspective data to shape the assistant persona in a more effective and controlled manner than alternatives such as constraining system prompts or activation steering. Specifically, we fine-tune three popular open-weights models using 11 example personas, such as humorous, deeply caring, or even malevolent. To track the effects of our approach, we introduce a method which analyzes revealed preferences, uncovering clear and holistic changes in character. We find these changes are more robust to adversarial prompting than the above two alternatives, while also leading to more coherent and realistic generations. Finally, we demonstrate this fine-tuning has little to no effect on general capabilities as measured by common benchmarks. We describe and open-source our full post-training method, the implementation of which can be found at this https URL. 

---
# Prompt Injection as an Emerging Threat: Evaluating the Resilience of Large Language Models 

**Authors**: Daniyal Ganiuly, Assel Smaiyl  

**Link**: [PDF](https://arxiv.org/pdf/2511.01634)  

**Abstract**: Large Language Models (LLMs) are increasingly used in intelligent systems that perform reasoning, summarization, and code generation. Their ability to follow natural-language instructions, while powerful, also makes them vulnerable to a new class of attacks known as prompt injection. In these attacks, hidden or malicious instructions are inserted into user inputs or external content, causing the model to ignore its intended task or produce unsafe responses. This study proposes a unified framework for evaluating how resistant Large Language Models (LLMs) are to prompt injection attacks. The framework defines three complementary metrics such as the Resilience Degradation Index (RDI), Safety Compliance Coefficient (SCC), and Instructional Integrity Metric (IIM) to jointly measure robustness, safety, and semantic stability. We evaluated four instruction-tuned models (GPT-4, GPT-4o, LLaMA-3 8B Instruct, and Flan-T5-Large) on five common language tasks: question answering, summarization, translation, reasoning, and code generation. Results show that GPT-4 performs best overall, while open-weight models remain more vulnerable. The findings highlight that strong alignment and safety tuning are more important for resilience than model size alone. Results show that all models remain partially vulnerable, especially to indirect and direct-override attacks. GPT-4 achieved the best overall resilience (RDR = 9.8 %, SCR = 96.4 %), while open-source models exhibited higher performance degradation and lower safety scores. The findings demonstrate that alignment strength and safety tuning play a greater role in resilience than model size alone. The proposed framework offers a structured, reproducible approach for assessing model robustness and provides practical insights for improving LLM safety and reliability. 

---
# SeaLLMs-Audio: Large Audio-Language Models for Southeast Asia 

**Authors**: Chaoqun Liu, Mahani Aljunied, Guizhen Chen, Hou Pong Chan, Weiwen Xu, Yu Rong, Wenxuan Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2511.01670)  

**Abstract**: We introduce SeaLLMs-Audio, the first large audio-language model (LALM) tailored for multiple Southeast Asian (SEA) languages-Indonesian (id), Thai (th), and Vietnamese (vi)-alongside English (en) and Chinese (zh). Trained on a large-scale audio corpus, SeaLLMs-Audio exhibits strong performance across diverse audio-centric tasks, spanning fine-grained audio understanding and voice-based interaction. Its key features include: 1) Multilingual: the model primarily supports 5 languages, namely Indonesian, Thai, Vietnamese, English, and Chinese; 2) Multimodal: the model accepts flexible input modalities, including audio only, text only, as well as audio with text; 3) Multi-task: the model supports a wide range of tasks, including audio analysis tasks such as Audio Captioning, Automatic Speech Recognition, Speech-to-Text Translation, Speech Emotion Recognition, Speech Question Answering, and Speech Summarization. It also enables voice-based dialogue, including answering factual, mathematical, and general knowledge queries. As a significant step towards advancing audio LLMs in Southeast Asia, we expect SeaLLMs-Audio to benefit both the regional research community and industry. To automate LALM evaluation for Southeast Asia, we introduce SeaBench-Audio, a benchmark spanning multiple tasks. Experiments show that SeaLLMs-Audio achieves competitive performance compared with other LALMs on SEA languages. 

---
# BanglaNirTox: A Large-scale Parallel Corpus for Explainable AI in Bengali Text Detoxification 

**Authors**: Ayesha Afroza Mohsin, Mashrur Ahsan, Nafisa Maliyat, Shanta Maria, Syed Rifat Raiyan, Hasan Mahmud, Md Kamrul Hasan  

**Link**: [PDF](https://arxiv.org/pdf/2511.01512)  

**Abstract**: Toxic language in Bengali remains prevalent, especially in online environments, with few effective precautions against it. Although text detoxification has seen progress in high-resource languages, Bengali remains underexplored due to limited resources. In this paper, we propose a novel pipeline for Bengali text detoxification that combines Pareto class-optimized large language models (LLMs) and Chain-of-Thought (CoT) prompting to generate detoxified sentences. To support this effort, we construct BanglaNirTox, an artificially generated parallel corpus of 68,041 toxic Bengali sentences with class-wise toxicity labels, reasonings, and detoxified paraphrases, using Pareto-optimized LLMs evaluated on random samples. The resulting BanglaNirTox dataset is used to fine-tune language models to produce better detoxified versions of Bengali sentences. Our findings show that Pareto-optimized LLMs with CoT prompting significantly enhance the quality and consistency of Bengali text detoxification. 

---
# PrefixNLI: Detecting Factual Inconsistencies as Soon as They Arise 

**Authors**: Sapir Harary, Eran Hirsch, Aviv Slobodkin, David Wan, Mohit Bansal, Ido Dagan  

**Link**: [PDF](https://arxiv.org/pdf/2511.01359)  

**Abstract**: Natural Language Inference (NLI) models have been used in various ways to improve the factuality of LLM outputs. This is typically done by applying an NLI model to judge whether the model output is entailed from the supposed evidence, triggering some corrective actions, such as beam reranking at inference time or RL rewards during training. While NLI models are trained to detect factual inconsistencies over complete sentences, decisions in the common autoregressive generation architecture are made for each evolving text prefix, during decoding. Addressing this setting, we generalize the entailment detection task to apply over arbitrary text prefixes, and suggest its utility for improving generation faithfulness. Providing suitable evaluation and training datasets for this task, we train MiniTruePrefixes, a novel specialized model that better detects factual inconsistencies over text prefixes, outperforming comparable baseline NLI models by 5-14 F1 points in prefix-level entailment. We further demonstrate that integrating MiniTruePrefixes into a controlled decoding framework substantially improves factual consistency in abstractive summarization. When guided by MiniTruePrefixes, LLaMA-3.2-3B-Instruct matches the faithfulness and runtime of the 8B model from the same model family, while using only half the memory. 

---
# Thinking with DistilQwen: A Tale of Four Distilled Reasoning and Reward Model Series 

**Authors**: Wenrui Cai, Chengyu Wang, Junbing Yan, Jun Huang, Xiangzhong Fang  

**Link**: [PDF](https://arxiv.org/pdf/2511.01354)  

**Abstract**: Recently, the demand for small and efficient reasoning models to support real-world applications has driven the development of knowledge distillation techniques that balance reasoning performance and inference speed. In this paper, we further extend the DistilQwen model family, initialized from the Qwen models, by introducing four model series specifically designed to meet industrial requirements. The distilled model collection comprises: (1) slow-thinking models, optimized for reasoning tasks that require high accuracy; (2) two series of adaptive-thinking models, which dynamically adjust reasoning strategies based on input tasks to maximize efficiency across diverse scenarios; and (3) distilled reward models, which enable further reinforcement learning of reasoning models using distilled knowledge. Comprehensive evaluations across multiple benchmarks demonstrate both high inference efficiency and strong reasoning performance for these models, as well as the practical utility of distilled reward models. We further show that these models support industry practitioners by providing scalable training and inference functionalities on the Alibaba Cloud PAI (Platform for Artificial Intelligence) platform. 

---
# SEPS: Semantic-enhanced Patch Slimming Framework for fine-grained cross-modal alignment 

**Authors**: Xinyu Mao, Junsi Li, Haoji Zhang, Yu Liang, Ming Sun  

**Link**: [PDF](https://arxiv.org/pdf/2511.01390)  

**Abstract**: Fine-grained cross-modal alignment aims to establish precise local correspondences between vision and language, forming a cornerstone for visual question answering and related multimodal applications. Current approaches face challenges in addressing patch redundancy and ambiguity, which arise from the inherent information density disparities across modalities. Recently, Multimodal Large Language Models (MLLMs) have emerged as promising solutions to bridge this gap through their robust semantic generation capabilities. However, the dense textual outputs from MLLMs may introduce conflicts with the original sparse captions. Furthermore, accurately quantifying semantic relevance between rich visual patches and concise textual descriptions remains a core challenge. To overcome these limitations, we introduce the Semantic-Enhanced Patch Slimming (SEPS) framework, which systematically addresses patch redundancy and ambiguity. Our approach employs a two-stage mechanism to integrate unified semantics from both dense and sparse texts, enabling the identification of salient visual patches. Additionally, it leverages relevance-aware selection with mean value computation to highlight crucial patch-word correspondences, thereby improving cross-modal similarity assessment. Comprehensive experiments on Flickr30K and MS-COCO datasets validate that SEPS achieves superior performance, surpassing existing approaches by 23\%-86\% in rSum across diverse model architectures, with notable enhancements in text-to-image retrieval scenarios. Our implementation is available at this https URL. 

---
# Exploringand Unleashing the Power of Large Language Models in CI/CD Configuration Translation 

**Authors**: Chong Wang, Chen Zhang, Jiajun Wu, Wunan Guo, Jianfeng Qu, Yewen Tian, Yang Liu  

**Link**: [PDF](https://arxiv.org/pdf/2511.01316)  

**Abstract**: Continuous Integration (CI) is a cornerstone of modern collaborative software development, and numerous CI platforms are available. Differences in maintenance overhead, reliability, and integration depth with code-hosting platforms make migration between CI platforms a common practice. A central step in migration is translating CI configurations, which is challenging due to the intrinsic complexity of CI configurations and the need to understand semantic differences and relationships across CI platforms.
With the advent of large language models (LLMs), recent advances in software engineering highlight their potential for CI configuration translation. In this paper, we present a study on LLM-based CI configuration translation, focusing on the migration from Travis CI to GitHub Actions. First, using 811 migration records, we quantify the effort involved and find that developers read an average of 38 lines of Travis configuration and write 58 lines of GitHub Actions configuration, with nearly half of the migrations requiring multiple commits. We further analyze translations produced by each of the four LLMs and identify 1,121 issues grouped into four categories: logic inconsistencies (38%), platform discrepancies (32%), environment errors (25%), and syntax errors (5%). Finally, we evaluate three enhancement strategies and show that combining guideline-based prompting with iterative refinement achieves the best performance, reaching a Build Success Rate of 75.5%-nearly a threefold improvement over GPT-4o with a basic prompt. 

---
# DEEPAMBIGQA: Ambiguous Multi-hop Questions for Benchmarking LLM Answer Completeness 

**Authors**: Jiabao Ji, Min Li, Priyanshu Kumar, Shiyu Chang, Saloni Potdar  

**Link**: [PDF](https://arxiv.org/pdf/2511.01323)  

**Abstract**: Large language models (LLMs) with integrated search tools show strong promise in open-domain question answering (QA), yet they often struggle to produce complete answer set to complex questions such as Which actor from the film Heat won at least one Academy Award?, which requires (1) distinguishing between multiple films sharing the same title and (2) reasoning across a large set of actors to gather and integrate evidence. Existing QA benchmarks rarely evaluate both challenges jointly. To address this, we introduce DeepAmbigQAGen, an automatic data generation pipeline that constructs QA tasks grounded in text corpora and linked knowledge graph, generating natural and verifiable questions that systematically embed name ambiguity and multi-step reasoning. Based on this, we build DeepAmbigQA, a dataset of 3,600 questions requiring multi-hop reasoning and half of them explicit name ambiguity resolving. Experiments reveal that, even state-of-the-art GPT-5 show incomplete answers, achieving only 0.13 exact match on ambiguous questions and 0.21 on non-ambiguous questions. These findings highlight the need for more robust QA systems aimed at information gathering and answer completeness. 

---
# When, What, and How: Rethinking Retrieval-Enhanced Speculative Decoding 

**Authors**: Min Fang, Zhihui Fu, Qibin Zhao, Jun Wang  

**Link**: [PDF](https://arxiv.org/pdf/2511.01282)  

**Abstract**: Speculative decoding (SD) has emerged as an effective technique to accelerate large language model (LLM) inference without compromising output quality. However, the achievable speedup largely depends on the effectiveness of the drafting model. While model-based methods like EAGLE-2 are accurate but costly, retrieval-enhanced methods like SAM-Decoding rely on heuristic switching strategies that often trigger unnecessary retrievals. To address this, we propose ReSpec (\textbf{Re}trieval-enhanced \textbf{Spe}culative Decoding), a novel framework that transforms heuristic drafter switching into adaptive decision-making. ReSpec features three core innovations: 1) An \textbf{entropy-guided adaptive trigger} quantifies contextual predictability to initiate retrieval only when uncertainty is low, avoiding costly low-quality speculations. 2) A \textbf{feedback-driven candidate selection} leverages historical feedback to organize multiple high-quality candidates for parallel verification, maximizing retrieval utility. 3) A source-aware \textbf{relaxed verification strategy} applies strict checks to model-generated drafts while using a relaxed verification for retrieved drafts, achieving a better balance between accuracy and efficiency. Extensive experiments on Spec-Bench demonstrate that ReSpec achieves state-of-the-art acceleration,outperforming EAGLE-2 and SAM-Decoding by over $33\%$ and $25\%$, respectively, while maintaining output quality. 

---
# Speech-DRAME: A Framework for Human-Aligned Benchmarks in Speech Role-Play 

**Authors**: Jiatong Shi, Jionghao Han, Yichen Lu, Santiago Pascual, Pengfei Wu, Chenye Cui, Shinji Watanabe, Chao Weng, Cong Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2511.01261)  

**Abstract**: Role-play has become a key testbed for generative models, expanding from text-only dialogue to multimodal interaction. Extending role-play to speech captures prosody, emotion, and delivery, but also poses new evaluation challenges. Current pipelines often use audio large language models (ALLMs) as zero-shot judges, which miss paralinguistic cues, collapse multiple aspects into coarse scores, and rely on synthetic speech references that fail to reflect real-world roles. We present Speech-DRAME, a unified framework that contributes at three levels: (i) Speech-DRAME-EvalBench, an evaluation benchmark with bilingual human-annotated data and protocols for training and testing speech evaluation models (SEMs), (ii) DRAME-Eval, a fine-tuned evaluation model, which substantially outperforms zero-shot and few-shot ALLMs, and (iii) Speech-DRAME-RoleBench, a speech role-play benchmark that leverages DRAME-Eval as an automatic judge to compare speech foundation models (SFMs). Speech-DRAME distinguishes between two complementary evaluation strategies: Archetype Evaluation, a top-down approach measuring adherence to broad role archetypes, and Realism Evaluation, a bottom-up approach grounded in real human speech that emphasizes nuanced role quality. Compared to zero-shot ALLM judges, DRAME-Eval achieves stronger agreement with human ratings (Pearson correlation from 0.480 to 0.629 in archetypes, and 0.390 to 0.625 in realism). By integrating transparent benchmark resources, modeling approaches, and system-level evaluation, Speech-DRAME provides the first comprehensive, reproducible foundation for assessing spoken role-play. 

---
# ZoFia: Zero-Shot Fake News Detection with Entity-Guided Retrieval and Multi-LLM Interaction 

**Authors**: Lvhua Wu, Xuefeng Jiang, Sheng Sun, Tian Wen, Yuwei Wang, Min Liu  

**Link**: [PDF](https://arxiv.org/pdf/2511.01188)  

**Abstract**: The rapid spread of fake news threatens social stability and public trust, rendering its detection an imperative research priority. Although large language models (LLMs) excel at numerous natural language processing tasks with their remarkable contextual understanding and extensive prior knowledge, the time-bounded knowledge coverage and tendency for generating hallucination content reduce their reliability when handling fast-evolving news streams. Furthermore, models trained on existing static datasets also often lack the generalization needed for emerging news topics. To address these challenges, we propose ZoFia, a novel two-stage zero-shot fake news detection framework. First, we introduce Hierarchical Salience to quantify the importance of entities in the news content, and propose the SC-MMR algorithm to effectively select an informative and diverse set of keywords that serve as queries for retrieving up-to-date external evidence. Subsequently, a multi LLM interactive system, in which each agent assumes a distinct role, performs multi-view collaborative analysis and adversarial debate over the news text and its related information, and finally produces an interpretable and robust judgment. Comprehensive experiments on two public datasets demonstrate that ZoFia obviously outperforms existing zero-shot baselines and most of few-shot methods. Our codes will be open-sourced to facilitate related communities. 

---
# Forget BIT, It is All about TOKEN: Towards Semantic Information Theory for LLMs 

**Authors**: Bo Bai  

**Link**: [PDF](https://arxiv.org/pdf/2511.01202)  

**Abstract**: Large language models (LLMs) have demonstrated remarkable capabilities in numerous real- world applications. While the vast majority of research conducted from an experimental perspective is progressing rapidly, it demands substantial computational power, data, and other resources. Therefore, how to open the black-box of LLMs from a theoretical standpoint has become a critical challenge. This paper takes the theory of rate-distortion function, directed information, and Granger causality as its starting point to investigate the information-theoretic principles behind LLMs, leading to the development of semantic information theory for LLMs, where the fundamental unit is token, rather than bits that lacks any semantic meaning. By defining the probabilistic model of LLMs, we discuss structure-agnostic information-theoretic measures, such as the directed rate- distortion function in pre-training, the directed rate-reward function in post-training, and the semantic information flow in inference phase. This paper also delves deeply into the theory of token-level semantic embedding and the information-theoretically optimal vectorization method. Thereafter, we propose a general definition of autoregression LLM, where the Transformer architecture and its performance such as ELBO, generalization error bound, memory capacity, and semantic information measures can be derived theoretically. Other architectures, such as Mamba/Mamba2 and LLaDA, are also discussed in our framework. Consequently, this paper provides a theoretical framework for understanding LLMs from the perspective of semantic information theory, which also offers the necessary theoretical tools for further in-depth research. 

---
# AthenaBench: A Dynamic Benchmark for Evaluating LLMs in Cyber Threat Intelligence 

**Authors**: Md Tanvirul Alam, Dipkamal Bhusal, Salman Ahmad, Nidhi Rastogi, Peter Worth  

**Link**: [PDF](https://arxiv.org/pdf/2511.01144)  

**Abstract**: Large Language Models (LLMs) have demonstrated strong capabilities in natural language reasoning, yet their application to Cyber Threat Intelligence (CTI) remains limited. CTI analysis involves distilling large volumes of unstructured reports into actionable knowledge, a process where LLMs could substantially reduce analyst workload. CTIBench introduced a comprehensive benchmark for evaluating LLMs across multiple CTI tasks. In this work, we extend CTIBench by developing AthenaBench, an enhanced benchmark that includes an improved dataset creation pipeline, duplicate removal, refined evaluation metrics, and a new task focused on risk mitigation strategies. We evaluate twelve LLMs, including state-of-the-art proprietary models such as GPT-5 and Gemini-2.5 Pro, alongside seven open-source models from the LLaMA and Qwen families. While proprietary LLMs achieve stronger results overall, their performance remains subpar on reasoning-intensive tasks, such as threat actor attribution and risk mitigation, with open-source models trailing even further behind. These findings highlight fundamental limitations in the reasoning capabilities of current LLMs and underscore the need for models explicitly tailored to CTI workflows and automation. 

---
# GeoToken: Hierarchical Geolocalization of Images via Next Token Prediction 

**Authors**: Narges Ghasemi, Amir Ziashahabi, Salman Avestimehr, Cyrus Shahabi  

**Link**: [PDF](https://arxiv.org/pdf/2511.01082)  

**Abstract**: Image geolocalization, the task of determining an image's geographic origin, poses significant challenges, largely due to visual similarities across disparate locations and the large search space. To address these issues, we propose a hierarchical sequence prediction approach inspired by how humans narrow down locations from broad regions to specific addresses. Analogously, our model predicts geographic tokens hierarchically, first identifying a general region and then sequentially refining predictions to increasingly precise locations. Rather than relying on explicit semantic partitions, our method uses S2 cells, a nested, multiresolution global grid, and sequentially predicts finer-level cells conditioned on visual inputs and previous predictions. This procedure mirrors autoregressive text generation in large language models. Much like in language modeling, final performance depends not only on training but also on inference-time strategy. We investigate multiple top-down traversal methods for autoregressive sampling, incorporating techniques from test-time compute scaling used in language models. Specifically, we integrate beam search and multi-sample inference while exploring various selection strategies to determine the final output. This enables the model to manage uncertainty by exploring multiple plausible paths through the hierarchy. We evaluate our method on the Im2GPS3k and YFCC4k datasets against two distinct sets of baselines: those that operate without a Multimodal Large Language Model (MLLM) and those that leverage one. In the MLLM-free setting, our model surpasses other comparable baselines on nearly all metrics, achieving state-of-the-art performance with accuracy gains of up to 13.9%. When augmented with an MLLM, our model outperforms all baselines, setting a new state-of-the-art across all metrics. The source code is available at this https URL. 

---
# OceanAI: A Conversational Platform for Accurate, Transparent, Near-Real-Time Oceanographic Insights 

**Authors**: Bowen Chen, Jayesh Gajbhar, Gregory Dusek, Rob Redmon, Patrick Hogan, Paul Liu, DelWayne Bohnenstiehl, Dongkuan, Ruoying He  

**Link**: [PDF](https://arxiv.org/pdf/2511.01019)  

**Abstract**: Artificial intelligence is transforming the sciences, yet general conversational AI systems often generate unverified "hallucinations" undermining scientific rigor. We present OceanAI, a conversational platform that integrates the natural-language fluency of open-source large language models (LLMs) with real-time, parameterized access to authoritative oceanographic data streams hosted by the National Oceanic and Atmospheric Administration (NOAA). Each query such as "What was Boston Harbor's highest water level in 2024?" triggers real-time API calls that identify, parse, and synthesize relevant datasets into reproducible natural-language responses and data visualizations. In a blind comparison with three widely used AI chat-interface products, only OceanAI produced NOAA-sourced values with original data references; others either declined to answer or provided unsupported results. Designed for extensibility, OceanAI connects to multiple NOAA data products and variables, supporting applications in marine hazard forecasting, ecosystem assessment, and water-quality monitoring. By grounding outputs and verifiable observations, OceanAI advances transparency, reproducibility, and trust, offering a scalable framework for AI-enabled decision support within the oceans. A public demonstration is available at this https URL. 

---
# The Riddle of Reflection: Evaluating Reasoning and Self-Awareness in Multilingual LLMs using Indian Riddles 

**Authors**: Abhinav P M, Ojasva Saxena, Oswald C, Parameswari Krishnamurthy  

**Link**: [PDF](https://arxiv.org/pdf/2511.00960)  

**Abstract**: The extent to which large language models (LLMs) can perform culturally grounded reasoning across non-English languages remains underexplored. This paper examines the reasoning and self-assessment abilities of LLMs across seven major Indian languages-Bengali, Gujarati, Hindi, Kannada, Malayalam, Tamil, and Telugu. We introduce a multilingual riddle dataset combining traditional riddles with context-reconstructed variants and evaluate five LLMs-Gemini 2.5 Pro, Gemini 2.5 Flash, Mistral-Saba, LLaMA 4 Scout, and LLaMA 4 Maverick-under seven prompting strategies. In the first stage, we assess riddle-solving performance and find that while Gemini 2.5 Pro performs best overall, few-shot methods yield only marginal gains, and accuracy varies notably across languages. In the second stage, we conduct a self-evaluation experiment to measure reasoning consistency. The results reveal a key finding: a model's initial accuracy is inversely correlated with its ability to identify its own mistakes. Top-performing models such as Gemini 2.5 Pro are overconfident (4.34% True Negative Rate), whereas lower-performing models like LLaMA 4 Scout are substantially more self-aware (42.09% True Negative Rate). These results point to clear gaps in multilingual reasoning and highlight the need for models that not only reason effectively but also recognize their own limitations. 

---
# Assessing LLM Reasoning Steps via Principal Knowledge Grounding 

**Authors**: Hyeon Hwang, Yewon Cho, Chanwoong Yoon, Yein Park, Minju Song, Kyungjae Lee, Gangwoo Kim, Jaewoo Kang  

**Link**: [PDF](https://arxiv.org/pdf/2511.00879)  

**Abstract**: Step-by-step reasoning has become a standard approach for large language models (LLMs) to tackle complex tasks. While this paradigm has proven effective, it raises a fundamental question: How can we verify that an LLM's reasoning is accurately grounded in knowledge? To address this question, we introduce a novel evaluation suite that systematically assesses the knowledge grounding of intermediate reasoning. Our framework comprises three key components. (1) Principal Knowledge Collection, a large-scale repository of atomic knowledge essential for reasoning. Based on the collection, we propose (2) knowledge-grounded evaluation metrics designed to measure how well models recall and apply prerequisite knowledge in reasoning. These metrics are computed by our (3) evaluator LLM, a lightweight model optimized for cost-effective and reliable metric computation. Our evaluation suite demonstrates remarkable effectiveness in identifying missing or misapplied knowledge elements, providing crucial insights for uncovering fundamental reasoning deficiencies in LLMs. Beyond evaluation, we demonstrate how these metrics can be integrated into preference optimization, showcasing further applications of knowledge-grounded evaluation. 

---
# GUI-AIMA: Aligning Intrinsic Multimodal Attention with a Context Anchor for GUI Grounding 

**Authors**: Shijie Zhou, Viet Dac Lai, Hao Tan, Jihyung Kil, Wanrong Zhu, Changyou Chen, Ruiyi Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2511.00810)  

**Abstract**: Graphical user interface (GUI) grounding is a key function of computer-use agents, which maps natural-language instructions to actionable screen regions. Existing approaches based on Multimodal Large Language Models (MLLMs) typically formulate it as a text-based coordinate generation task, yet directly generating precise coordinates from visual inputs remains challenging and computationally intensive. An intuitive way to implement GUI grounding is to first select visual patches relevant to the instructions and then determine the precise click location within those patches. Based on the observations that general MLLMs have some native grounding capability, nested within their attentions, we propose GUI-AIMA, an attention-based and coordinate-free supervised fine-tuning framework for efficient GUI grounding. GUI-AIMA aligns the intrinsic multimodal attention of MLLMs with patch-wise grounding signals. These signals are calculated adaptively for diverse user instructions by multi-head aggregation on simplified query-visual attention matrices. Besides, its coordinate-free manner can easily integrate a plug-and-play zoom-in stage. GUI-AIMA-3B was trained with only 85k screenshots, demonstrating exceptional data efficiency and verifying that light training can trigger the native grounding capability of MLLMs. It achieves state-of-the-art performance among 3B models, attaining an average accuracy of 58.6% on ScreenSpot-Pro and 62.2% on OSWorld-G. Project page: this https URL 

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
# Belief Dynamics Reveal the Dual Nature of In-Context Learning and Activation Steering 

**Authors**: Eric Bigelow, Daniel Wurgaft, YingQiao Wang, Noah Goodman, Tomer Ullman, Hidenori Tanaka, Ekdeep Singh Lubana  

**Link**: [PDF](https://arxiv.org/pdf/2511.00617)  

**Abstract**: Large language models (LLMs) can be controlled at inference time through prompts (in-context learning) and internal activations (activation steering). Different accounts have been proposed to explain these methods, yet their common goal of controlling model behavior raises the question of whether these seemingly disparate methodologies can be seen as specific instances of a broader framework. Motivated by this, we develop a unifying, predictive account of LLM control from a Bayesian perspective. Specifically, we posit that both context- and activation-based interventions impact model behavior by altering its belief in latent concepts: steering operates by changing concept priors, while in-context learning leads to an accumulation of evidence. This results in a closed-form Bayesian model that is highly predictive of LLM behavior across context- and activation-based interventions in a set of domains inspired by prior work on many-shot in-context learning. This model helps us explain prior empirical phenomena - e.g., sigmoidal learning curves as in-context evidence accumulates - while predicting novel ones - e.g., additivity of both interventions in log-belief space, which results in distinct phases such that sudden and dramatic behavioral shifts can be induced by slightly changing intervention controls. Taken together, this work offers a unified account of prompt-based and activation-based control of LLM behavior, and a methodology for empirically predicting the effects of these interventions. 

---
# ShadowLogic: Backdoors in Any Whitebox LLM 

**Authors**: Kasimir Schulz, Amelia Kawasaki, Leo Ring  

**Link**: [PDF](https://arxiv.org/pdf/2511.00664)  

**Abstract**: Large language models (LLMs) are widely deployed across various applications, often with safeguards to prevent the generation of harmful or restricted content. However, these safeguards can be covertly bypassed through adversarial modifications to the computational graph of a model. This work highlights a critical security vulnerability in computational graph-based LLM formats, demonstrating that widely used deployment pipelines may be susceptible to obscured backdoors. We introduce ShadowLogic, a method for creating a backdoor in a white-box LLM by injecting an uncensoring vector into its computational graph representation. We set a trigger phrase that, when added to the beginning of a prompt into the LLM, applies the uncensoring vector and removes the content generation safeguards in the model. We embed trigger logic directly into the computational graph which detects the trigger phrase in a prompt. To evade detection of our backdoor, we obfuscate this logic within the graph structure, making it similar to standard model functions. Our method requires minimal alterations to model parameters, making backdoored models appear benign while retaining the ability to generate uncensored responses when activated. We successfully implement ShadowLogic in Phi-3 and Llama 3.2, using ONNX for manipulating computational graphs. Implanting the uncensoring vector achieved a >60% attack success rate for further malicious queries. 

---
# FlashEVA: Accelerating LLM inference via Efficient Attention 

**Authors**: Juan Gabriel Kostelec, Qinghai Guo  

**Link**: [PDF](https://arxiv.org/pdf/2511.00576)  

**Abstract**: Transformer models have revolutionized natural language processing, achieving state-of-the-art performance and demonstrating remarkable scalability. However, their memory demands, particularly due to maintaining full context in memory, pose significant challenges for inference. In this paper, we present FlashEVA, an efficient implementation of EVA (Efficient Attention via Control Variates), and demonstrate how to finetune transformers to adapt to FlashEVA attention. Our method enables fine-tuning of Transformer models with as few as 1.5B tokens while preserving effectiveness across various downstream tasks. Notably, FlashEVA achieves up to 6.7x higher throughput and 5x lower peak GPU memory usage during inference compared to standard Transformer implementations. Despite these improvements, we observe limitations in retrieval-focused tasks. Our implementation offers control over the trade-off between throughput and accuracy through adjustable hyperparameters, providing flexibility for diverse use cases. This work represents a significant step towards more efficient and adaptable Transformer-based models for inference. 

---
# Red-teaming Activation Probes using Prompted LLMs 

**Authors**: Phil Blandfort, Robert Graham  

**Link**: [PDF](https://arxiv.org/pdf/2511.00554)  

**Abstract**: Activation probes are attractive monitors for AI systems due to low cost and latency, but their real-world robustness remains underexplored. We ask: What failure modes arise under realistic, black-box adversarial pressure, and how can we surface them with minimal effort? We present a lightweight black-box red-teaming procedure that wraps an off-the-shelf LLM with iterative feedback and in-context learning (ICL), and requires no fine-tuning, gradients, or architectural access. Running a case study with probes for high-stakes interactions, we show that our approach can help discover valuable insights about a SOTA probe. Our analysis uncovers interpretable brittleness patterns (e.g., legalese-induced FPs; bland procedural tone FNs) and reduced but persistent vulnerabilities under scenario-constraint attacks. These results suggest that simple prompted red-teaming scaffolding can anticipate failure patterns before deployment and might yield promising, actionable insights to harden future probes. 

---
# Diagnosing Hallucination Risk in AI Surgical Decision-Support: A Sequential Framework for Sequential Validation 

**Authors**: Dong Chen, Yanzhe Wei, Zonglin He, Guan-Ming Kuang, Canhua Ye, Meiru An, Huili Peng, Yong Hu, Huiren Tao, Kenneth MC Cheung  

**Link**: [PDF](https://arxiv.org/pdf/2511.00588)  

**Abstract**: Large language models (LLMs) offer transformative potential for clinical decision support in spine surgery but pose significant risks through hallucinations, which are factually inconsistent or contextually misaligned outputs that may compromise patient safety. This study introduces a clinician-centered framework to quantify hallucination risks by evaluating diagnostic precision, recommendation quality, reasoning robustness, output coherence, and knowledge alignment. We assessed six leading LLMs across 30 expert-validated spinal cases. DeepSeek-R1 demonstrated superior overall performance (total score: 86.03 $\pm$ 2.08), particularly in high-stakes domains such as trauma and infection. A critical finding reveals that reasoning-enhanced model variants did not uniformly outperform standard counterparts: Claude-3.7-Sonnet's extended thinking mode underperformed relative to its standard version (80.79 $\pm$ 1.83 vs. 81.56 $\pm$ 1.92), indicating extended chain-of-thought reasoning alone is insufficient for clinical reliability. Multidimensional stress-testing exposed model-specific vulnerabilities, with recommendation quality degrading by 7.4% under amplified complexity. This decline contrasted with marginal improvements in rationality (+2.0%), readability (+1.7%) and diagnosis (+4.7%), highlighting a concerning divergence between perceived coherence and actionable guidance. Our findings advocate integrating interpretability mechanisms (e.g., reasoning chain visualization) into clinical workflows and establish a safety-aware validation framework for surgical LLM deployment. 

---
# HIP-LLM: A Hierarchical Imprecise Probability Approach to Reliability Assessment of Large Language Models 

**Authors**: Robab Aghazadeh-Chakherlou, Qing Guo, Siddartha Khastgir, Peter Popov, Xiaoge Zhang, Xingyu Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2511.00527)  

**Abstract**: Large Language Models (LLMs) are increasingly deployed across diverse domains, raising the need for rigorous reliability assessment methods. Existing benchmark-based evaluations primarily offer descriptive statistics of model accuracy over datasets, providing limited insight into the probabilistic behavior of LLMs under real operational conditions. This paper introduces HIP-LLM, a Hierarchical Imprecise Probability framework for modeling and inferring LLM reliability. Building upon the foundations of software reliability engineering, HIP-LLM defines LLM reliability as the probability of failure-free operation over a specified number of future tasks under a given Operational Profile (OP). HIP-LLM represents dependencies across (sub-)domains hierarchically, enabling multi-level inference from subdomain to system-level reliability. HIP-LLM embeds imprecise priors to capture epistemic uncertainty and incorporates OPs to reflect usage contexts. It derives posterior reliability envelopes that quantify uncertainty across priors and data. Experiments on multiple benchmark datasets demonstrate that HIP-LLM offers a more accurate and standardized reliability characterization than existing benchmark and state-of-the-art approaches. A publicly accessible repository of HIP-LLM is provided. 

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
# MedRECT: A Medical Reasoning Benchmark for Error Correction in Clinical Texts 

**Authors**: Naoto Iwase, Hiroki Okuyama, Junichiro Iwasawa  

**Link**: [PDF](https://arxiv.org/pdf/2511.00421)  

**Abstract**: Large language models (LLMs) show increasing promise in medical applications, but their ability to detect and correct errors in clinical texts -- a prerequisite for safe deployment -- remains under-evaluated, particularly beyond English. We introduce MedRECT, a cross-lingual benchmark (Japanese/English) that formulates medical error handling as three subtasks: error detection, error localization (sentence extraction), and error correction. MedRECT is built with a scalable, automated pipeline from the Japanese Medical Licensing Examinations (JMLE) and a curated English counterpart, yielding MedRECT-ja (663 texts) and MedRECT-en (458 texts) with comparable error/no-error balance. We evaluate 9 contemporary LLMs spanning proprietary, open-weight, and reasoning families. Key findings: (i) reasoning models substantially outperform standard architectures, with up to 13.5% relative improvement in error detection and 51.0% in sentence extraction; (ii) cross-lingual evaluation reveals 5-10% performance gaps from English to Japanese, with smaller disparities for reasoning models; (iii) targeted LoRA fine-tuning yields asymmetric improvements in error correction performance (Japanese: +0.078, English: +0.168) while preserving reasoning capabilities; and (iv) our fine-tuned model exceeds human expert performance on structured medical error correction tasks. To our knowledge, MedRECT is the first comprehensive cross-lingual benchmark for medical error correction, providing a reproducible framework and resources for developing safer medical LLMs across languages. 

---
# Reasoning Planning for Language Models 

**Authors**: Bao Nguyen, Hieu Trung Nguyen, Ruifeng She, Xiaojin Fu, Viet Anh Nguyen  

**Link**: [PDF](https://arxiv.org/pdf/2511.00521)  

**Abstract**: Selecting an appropriate reasoning method for a given query remains a key challenge in language model generation. Existing approaches typically generate multiple candidate responses and use an aggregation strategy to select the output answer, often assuming that more candidate answers yield higher accuracy. We revisit this assumption through a rigorous theoretical analysis, deriving accuracy bounds for standard aggregation methods under fixed generation distributions and candidate sizes. Building on these insights, we introduce EPIC, an Ensemble Planning with Contrastive learning framework to learn a shared representation space that captures both model reasoning abilities and query-method compatibility. EPIC incorporates our probability bounds as a regularizer in a utility-driven optimization that balances accuracy and computational cost. Experiments on diverse mathematical reasoning tasks show that EPIC consistently selects optimal reasoning methods, improving accuracy while reducing computational overhead. Our code can be found at this https URL. 

---
# UME-R1: Exploring Reasoning-Driven Generative Multimodal Embeddings 

**Authors**: Zhibin Lan, Liqiang Niu, Fandong Meng, Jie Zhou, Jinsong Su  

**Link**: [PDF](https://arxiv.org/pdf/2511.00405)  

**Abstract**: The remarkable success of multimodal large language models (MLLMs) has driven advances in multimodal embeddings, yet existing models remain inherently discriminative, limiting their ability to benefit from reasoning-driven generation paradigm. In this work, we pioneer the exploration of generative embeddings, unifying embedding tasks within a generative paradigm. We propose UME-R1, a universal multimodal embedding framework consisting of a two-stage training strategy: a cold-start supervised fine-tuning equips the model with reasoning capabilities and enables it to generate both discriminative and generative embeddings; a subsequent reinforcement learning enhances reasoning and further optimizes generative embedding quality. This pioneering work reveals four key insights: 1) generative embeddings unlock substantial performance gains over conventional discriminative embeddings by leveraging the powerful generative reasoning capabilities of MLLMs; 2) discriminative and generative embeddings are complementary, whose combined oracle performance far exceeding that of either alone; 3) RL can effectively enhance generative embeddings, establishing a scalable optimization paradigm.; 4) repeated sampling at inference boosts downstream task coverage (pass@k), highlighting the inference-time scalability potential of generative embeddings. Evaluated on the MMEB-V2 benchmark across 78 tasks spanning video, image, and visual documents, UME-R1 significantly outperforms conventional discriminative embedding models and offers a foundation for more interpretable, reasoning-driven generative multimodal embeddings. Our code, models, and datasets will be publicly available at this https URL. 

---
# Scalable Processing-Near-Memory for 1M-Token LLM Inference: CXL-Enabled KV-Cache Management Beyond GPU Limits 

**Authors**: Dowon Kim, MinJae Lee, Janghyeon Kim, HyuckSung Kwon, Hyeonggyu Jeong, Sang-Soo Park, Minyong Yoon, Si-Dong Roh, Yongsuk Kwon, Jinin So, Jungwook Choi  

**Link**: [PDF](https://arxiv.org/pdf/2511.00321)  

**Abstract**: The expansion of context windows in large language models (LLMs) to multi-million tokens introduces severe memory and compute bottlenecks, particularly in managing the growing Key-Value (KV) cache. While Compute Express Link (CXL) enables non-eviction frameworks that offload the full KV-cache to scalable external memory, these frameworks still suffer from costly data transfers when recalling non-resident KV tokens to limited GPU memory as context lengths increase. This work proposes scalable Processing-Near-Memory (PNM) for 1M-Token LLM Inference, a CXL-enabled KV-cache management system that coordinates memory and computation beyond GPU limits. Our design offloads token page selection to a PNM accelerator within CXL memory, eliminating costly recalls and enabling larger GPU batch sizes. We further introduce a hybrid parallelization strategy and a steady-token selection mechanism to enhance compute efficiency and scalability. Implemented atop a state-of-the-art CXL-PNM system, our solution delivers consistent performance gains for LLMs with up to 405B parameters and 1M-token contexts. Our PNM-only offloading scheme (PNM-KV) and GPU-PNM hybrid with steady-token execution (PnG-KV) achieve up to 21.9x throughput improvement, up to 60x lower energy per token, and up to 7.3x better total cost efficiency than the baseline, demonstrating that CXL-enabled multi-PNM architectures can serve as a scalable backbone for future long-context LLM inference. 

---
# Exploiting Latent Space Discontinuities for Building Universal LLM Jailbreaks and Data Extraction Attacks 

**Authors**: Kayua Oleques Paim, Rodrigo Brandao Mansilha, Diego Kreutz, Muriel Figueredo Franco, Weverton Cordeiro  

**Link**: [PDF](https://arxiv.org/pdf/2511.00346)  

**Abstract**: The rapid proliferation of Large Language Models (LLMs) has raised significant concerns about their security against adversarial attacks. In this work, we propose a novel approach to crafting universal jailbreaks and data extraction attacks by exploiting latent space discontinuities, an architectural vulnerability related to the sparsity of training data. Unlike previous methods, our technique generalizes across various models and interfaces, proving highly effective in seven state-of-the-art LLMs and one image generation model. Initial results indicate that when these discontinuities are exploited, they can consistently and profoundly compromise model behavior, even in the presence of layered defenses. The findings suggest that this strategy has substantial potential as a systemic attack vector. 

---
# Consistently Simulating Human Personas with Multi-Turn Reinforcement Learning 

**Authors**: Marwa Abdulhai, Ryan Cheng, Donovan Clay, Tim Althoff, Sergey Levine, Natasha Jaques  

**Link**: [PDF](https://arxiv.org/pdf/2511.00222)  

**Abstract**: Large Language Models (LLMs) are increasingly used to simulate human users in interactive settings such as therapy, education, and social role-play. While these simulations enable scalable training and evaluation of AI agents, off-the-shelf LLMs often drift from their assigned personas, contradict earlier statements, or abandon role-appropriate behavior. We introduce a unified framework for evaluating and improving persona consistency in LLM-generated dialogue. We define three automatic metrics: prompt-to-line consistency, line-to-line consistency, and Q&A consistency, that capture different types of persona drift and validate each against human annotations. Using these metrics as reward signals, we apply multi-turn reinforcement learning to fine-tune LLMs for three user roles: a patient, a student, and a social chat partner. Our method reduces inconsistency by over 55%, resulting in more coherent and faithful simulated users. 

---
# Training LLMs Beyond Next Token Prediction - Filling the Mutual Information Gap 

**Authors**: Chun-Hao Yang, Bo-Han Feng, Tzu-Yuan Lai, Yan Yu Chen, Yin-Kai Dean Huang, Shou-De Lin  

**Link**: [PDF](https://arxiv.org/pdf/2511.00198)  

**Abstract**: Optimizing training performance in large language models (LLMs) remains an essential challenge, particularly in improving model performance while maintaining computational costs. This work challenges the conventional approach of training LLMs using next-token prediction (NTP), arguing that by predicting information-rich tokens during training, there is a more effective way to train LLMs. We investigate the impact of the proposed solution in three kinds of tasks for LLMs: arithmetic, multi-label classification of text, and natural-language generation. This work offers a principled approach to optimizing LLM training, advancing both model performance and theoretical understanding of the target-token selection strategies. 

---
# A Technical Exploration of Causal Inference with Hybrid LLM Synthetic Data 

**Authors**: Dana Kim, Yichen Xu, Tiffany Lin  

**Link**: [PDF](https://arxiv.org/pdf/2511.00318)  

**Abstract**: Large Language Models (LLMs) offer a flexible means to generate synthetic tabular data, yet existing approaches often fail to preserve key causal parameters such as the average treatment effect (ATE). In this technical exploration, we first demonstrate that state-of-the-art synthetic data generators, both GAN- and LLM-based, can achieve high predictive fidelity while substantially misestimating causal effects. To address this gap, we propose a hybrid generation framework that combines model-based covariate synthesis (monitored via distance-to-closest-record filtering) with separately learned propensity and outcome models, thereby ensuring that (W, A, Y) triplets retain their underlying causal structure. We further introduce a synthetic pairing strategy to mitigate positivity violations and a realistic evaluation protocol that leverages unlimited synthetic samples to benchmark traditional estimators (IPTW, AIPW, substitution) under complex covariate distributions. This work lays the groundwork for LLM-powered data pipelines that support robust causal analysis. Our code is available at this https URL. 

---
# Calibration Across Layers: Understanding Calibration Evolution in LLMs 

**Authors**: Abhinav Joshi, Areeb Ahmad, Ashutosh Modi  

**Link**: [PDF](https://arxiv.org/pdf/2511.00280)  

**Abstract**: Large Language Models (LLMs) have demonstrated inherent calibration capabilities, where predicted probabilities align well with correctness, despite prior findings that deep neural networks are often overconfident. Recent studies have linked this behavior to specific components in the final layer, such as entropy neurons and the unembedding matrix null space. In this work, we provide a complementary perspective by investigating how calibration evolves throughout the network depth. Analyzing multiple open-weight models on the MMLU benchmark, we uncover a distinct confidence correction phase in the upper/later layers, where model confidence is actively recalibrated after decision certainty has been reached. Furthermore, we identify a low-dimensional calibration direction in the residual stream whose perturbation significantly improves calibration metrics (ECE and MCE) without harming accuracy. Our findings suggest that calibration is a distributed phenomenon, shaped throughout the network forward pass, not just in its final projection, providing new insights into how confidence-regulating mechanisms operate within LLMs. 

---
# IL-PCSR: Legal Corpus for Prior Case and Statute Retrieval 

**Authors**: Shounak Paul, Dhananjay Ghumare, Pawan Goyal, Saptarshi Ghosh, Ashutosh Modi  

**Link**: [PDF](https://arxiv.org/pdf/2511.00268)  

**Abstract**: Identifying/retrieving relevant statutes and prior cases/precedents for a given legal situation are common tasks exercised by law practitioners. Researchers to date have addressed the two tasks independently, thus developing completely different datasets and models for each task; however, both retrieval tasks are inherently related, e.g., similar cases tend to cite similar statutes (due to similar factual situation). In this paper, we address this gap. We propose IL-PCR (Indian Legal corpus for Prior Case and Statute Retrieval), which is a unique corpus that provides a common testbed for developing models for both the tasks (Statute Retrieval and Precedent Retrieval) that can exploit the dependence between the two. We experiment extensively with several baseline models on the tasks, including lexical models, semantic models and ensemble based on GNNs. Further, to exploit the dependence between the two tasks, we develop an LLM-based re-ranking approach that gives the best performance. 

---
# Effectiveness of LLMs in Temporal User Profiling for Recommendation 

**Authors**: Milad Sabouri, Masoud Mansoury, Kun Lin, Bamshad Mobasher  

**Link**: [PDF](https://arxiv.org/pdf/2511.00176)  

**Abstract**: Effectively modeling the dynamic nature of user preferences is crucial for enhancing recommendation accuracy and fostering transparency in recommender systems. Traditional user profiling often overlooks the distinction between transitory short-term interests and stable long-term preferences. This paper examines the capability of leveraging Large Language Models (LLMs) to capture these temporal dynamics, generating richer user representations through distinct short-term and long-term textual summaries of interaction histories. Our observations suggest that while LLMs tend to improve recommendation quality in domains with more active user engagement, their benefits appear less pronounced in sparser environments. This disparity likely stems from the varying distinguishability of short-term and long-term preferences across domains; the approach shows greater utility where these temporal interests are more clearly separable (e.g., Movies\&TV) compared to domains with more stable user profiles (e.g., Video Games). This highlights a critical trade-off between enhanced performance and computational costs, suggesting context-dependent LLM application. Beyond predictive capability, this LLM-driven approach inherently provides an intrinsic potential for interpretability through its natural language profiles and attention weights. This work contributes insights into the practical capability and inherent interpretability of LLM-driven temporal user profiling, outlining new research directions for developing adaptive and transparent recommender systems. 

---
# A Dual Large Language Models Architecture with Herald Guided Prompts for Parallel Fine Grained Traffic Signal Control 

**Authors**: Qing Guo, Xinhang Li, Junyu Chen, Zheng Guo, Xiaocong Li, Lin Zhang, Lei Li  

**Link**: [PDF](https://arxiv.org/pdf/2511.00136)  

**Abstract**: Leveraging large language models (LLMs) in traffic signal control (TSC) improves optimization efficiency and interpretability compared to traditional reinforcement learning (RL) methods. However, existing LLM-based approaches are limited by fixed time signal durations and are prone to hallucination errors, while RL methods lack robustness in signal timing decisions and suffer from poor generalization. To address these challenges, this paper proposes HeraldLight, a dual LLMs architecture enhanced by Herald guided prompts. The Herald Module extracts contextual information and forecasts queue lengths for each traffic phase based on real-time conditions. The first LLM, LLM-Agent, uses these forecasts to make fine grained traffic signal control, while the second LLM, LLM-Critic, refines LLM-Agent's outputs, correcting errors and hallucinations. These refined outputs are used for score-based fine-tuning to improve accuracy and robustness. Simulation experiments using CityFlow on real world datasets covering 224 intersections in Jinan (12), Hangzhou (16), and New York (196) demonstrate that HeraldLight outperforms state of the art baselines, achieving a 20.03% reduction in average travel time across all scenarios and a 10.74% reduction in average queue length on the Jinan and Hangzhou scenarios. The source code is available on GitHub: this https URL. 

---
# What a diff makes: automating code migration with large language models 

**Authors**: Katherine A. Rosenfeld, Cliff C. Kerr, Jessica Lundin  

**Link**: [PDF](https://arxiv.org/pdf/2511.00160)  

**Abstract**: Modern software programs are built on stacks that are often undergoing changes that introduce updates and improvements, but may also break any project that depends upon them. In this paper we explore the use of Large Language Models (LLMs) for code migration, specifically the problem of maintaining compatibility with a dependency as it undergoes major and minor semantic version changes. We demonstrate, using metrics such as test coverage and change comparisons, that contexts containing diffs can significantly improve performance against out of the box LLMs and, in some cases, perform better than using code. We provide a dataset to assist in further development of this problem area, as well as an open-source Python package, AIMigrate, that can be used to assist with migrating code bases. In a real-world migration of TYPHOIDSIM between STARSIM versions, AIMigrate correctly identified 65% of required changes in a single run, increasing to 80% with multiple runs, with 47% of changes generated perfectly. 

---
# Inferring multiple helper Dafny assertions with LLMs 

**Authors**: Álvaro Silva, Alexandra Mendes, Ruben Martins  

**Link**: [PDF](https://arxiv.org/pdf/2511.00125)  

**Abstract**: The Dafny verifier provides strong correctness guarantees but often requires numerous manual helper assertions, creating a significant barrier to adoption. We investigate the use of Large Language Models (LLMs) to automatically infer missing helper assertions in Dafny programs, with a primary focus on cases involving multiple missing assertions. To support this study, we extend the DafnyBench benchmark with curated datasets where one, two, or all assertions are removed, and we introduce a taxonomy of assertion types to analyze inference difficulty. Our approach refines fault localization through a hybrid method that combines LLM predictions with error-message heuristics. We implement this approach in a new tool called DAISY (Dafny Assertion Inference SYstem). While our focus is on multiple missing assertions, we also evaluate DAISY on single-assertion cases. DAISY verifies 63.4% of programs with one missing assertion and 31.7% with multiple missing assertions. Notably, many programs can be verified with fewer assertions than originally present, highlighting that proofs often admit multiple valid repair strategies and that recovering every original assertion is unnecessary. These results demonstrate that automated assertion inference can substantially reduce proof engineering effort and represent a step toward more scalable and accessible formal verification. 

---
# Loquetier: A Virtualized Multi-LoRA Framework for Unified LLM Fine-tuning and Serving 

**Authors**: Yuchen Zhang, Hanyue Du, Chun Cao, Jingwei Xu  

**Link**: [PDF](https://arxiv.org/pdf/2511.00101)  

**Abstract**: Low-Rank Adaptation (LoRA) has become a widely adopted parameter-efficient fine-tuning (PEFT) technique for adapting large language models (LLMs) to downstream tasks. While prior work has explored strategies for integrating LLM training and serving, there still remains a gap in unifying fine-tuning and inference for LoRA-based models. We present Loquetier, a virtualized multi-LoRA framework that seamlessly integrates LoRA fine-tuning and serving within a single runtime. Loquetier introduces two key components: (1) a Virtualized Module that isolates PEFT-based modifications and supports multiple adapters on a shared base model, and (2) an optimized computation flow with a kernel design that merges fine-tuning and inference paths in forward propagation, enabling efficient batching and minimizing kernel invocation overhead. Extensive experiments across three task settings show that Loquetier consistently outperforms existing baselines in both performance and flexibility, achieving up to $3.0\times$ the throughput of the state-of-the-art co-serving system on inference-only tasks and $46.4\times$ higher SLO attainment than PEFT on unified fine-tuning and inference tasks. The implementation of Loquetier is publicly available at this https URL. 

---
# Urban-MAS: Human-Centered Urban Prediction with LLM-Based Multi-Agent System 

**Authors**: Shangyu Lou  

**Link**: [PDF](https://arxiv.org/pdf/2511.00096)  

**Abstract**: Urban Artificial Intelligence (Urban AI) has advanced human-centered urban tasks such as perception prediction and human dynamics. Large Language Models (LLMs) can integrate multimodal inputs to address heterogeneous data in complex urban systems but often underperform on domain-specific tasks. Urban-MAS, an LLM-based Multi-Agent System (MAS) framework, is introduced for human- centered urban prediction under zero-shot settings. It includes three agent types: Predictive Factor Guidance Agents, which prioritize key predictive factors to guide knowledge extraction and enhance the effectiveness of compressed urban knowledge in LLMs; Reliable UrbanInfo Extraction Agents, which improve robustness by com- paring multiple outputs, validating consistency, and re-extracting when conflicts occur; and Multi-UrbanInfo Inference Agents, which integrate extracted multi-source information across dimensions for prediction. Experiments on running-amount prediction and ur- ban perception across Tokyo, Milan, and Seattle demonstrate that Urban-MAS substantially reduces errors compared to single-LLM baselines. Ablation studies indicate that Predictive Factor Guidance Agents are most critical for enhancing predictive performance, po- sitioning Urban-MAS as a scalable paradigm for human-centered urban AI prediction. Code is available on the project website:this https URL 

---
# Cognitive Alignment in Personality Reasoning: Leveraging Prototype Theory for MBTI Inference 

**Authors**: Haoyuan Li, Yuanbo Tong, Yuchen Li, Zirui Wang, Chunhou Liu, Jiamou Liu  

**Link**: [PDF](https://arxiv.org/pdf/2511.00115)  

**Abstract**: Personality recognition from text is typically cast as hard-label classification, which obscures the graded, prototype-like nature of human personality judgments. We present ProtoMBTI, a cognitively aligned framework for MBTI inference that operationalizes prototype theory within an LLM-based pipeline. First, we construct a balanced, quality-controlled corpus via LLM-guided multi-dimensional augmentation (semantic, linguistic, sentiment). Next, we LoRA-fine-tune a lightweight (<=2B) encoder to learn discriminative embeddings and to standardize a bank of personality prototypes. At inference, we retrieve top-k prototypes for a query post and perform a retrieve--reuse--revise--retain cycle: the model aggregates prototype evidence via prompt-based voting, revises when inconsistencies arise, and, upon correct prediction, retains the sample to continually enrich the prototype library. Across Kaggle and Pandora benchmarks, ProtoMBTI improves over baselines on both the four MBTI dichotomies and the full 16-type task, and exhibits robust cross-dataset generalization. Our results indicate that aligning the inference process with psychological prototype reasoning yields gains in accuracy, interpretability, and transfer for text-based personality modeling. 

---
# Generalizing Test-time Compute-optimal Scaling as an Optimizable Graph 

**Authors**: Fali Wang, Jihai Chen, Shuhua Yang, Runxue Bao, Tianxiang Zhao, Zhiwei Zhang, Xianfeng Tang, Hui Liu, Qi He, Suhang Wang  

**Link**: [PDF](https://arxiv.org/pdf/2511.00086)  

**Abstract**: Test-Time Scaling (TTS) improves large language models (LLMs) by allocating additional computation during inference, typically through parallel, sequential, or hybrid scaling. However, prior studies often assume fixed collaboration architectures (e.g., topologies) and single-model usage, overlooking that optimal architectures and model combinations can vary across tasks. Therefore, we study the novel problem of searching for compute-optimal model combinations and architectures in TTS under a fixed budget. We formalize it as a multi-LLM collaboration graph, where nodes encode roles and LLM model assignments, and edges capture information flow. This problem is challenging because (i) the combinatorial search space is prohibitively large, and (ii) task-specific requirements demand tailored designs. To address these, we reformulate the problem as probabilistic graph optimization and, through pilot experiments, derive three empirical insights into TTS collaboration graphs. Guided by these insights, we propose Agent-REINFORCE, an LLM-agent-augmented framework that mirrors the REINFORCE pipeline by mapping sampling-gradient-update to sampling-feedback-update, where feedback serves as a textual gradient to update the probabilistic graph and efficiently search for optimal multi-LLM collaboration graphs. Experiments show that Agent-REINFORCE outperforms both traditional and LLM-based baselines in sample efficiency and search performance, and effectively identifies optimal graphs under joint objectives of accuracy and inference latency. 

---
# MISA: Memory-Efficient LLMs Optimization with Module-wise Importance Sampling 

**Authors**: Yuxi Liu, Renjia Deng, Yutong He, Xue Wang, Tao Yao, Kun Yuan  

**Link**: [PDF](https://arxiv.org/pdf/2511.00056)  

**Abstract**: The substantial memory demands of pre-training and fine-tuning large language models (LLMs) require memory-efficient optimization algorithms. One promising approach is layer-wise optimization, which treats each transformer block as a single layer and optimizes it sequentially, while freezing the other layers to save optimizer states and activations. Although effective, these methods ignore the varying importance of the modules within each layer, leading to suboptimal performance. Moreover, layer-wise sampling provides only limited memory savings, as at least one full layer must remain active during optimization. To overcome these limitations, we propose Module-wise Importance SAmpling (MISA), a novel method that divides each layer into smaller modules and assigns importance scores to each module. MISA uses a weighted random sampling mechanism to activate modules, provably reducing gradient variance compared to layer-wise sampling. Additionally, we establish an \(\mathcal{O}(1/\sqrt{K})\) convergence rate under non-convex and stochastic conditions, where $K$ is the total number of block updates, and provide a detailed memory analysis showcasing MISA's superiority over existing baseline methods. Experiments on diverse learning tasks validate the effectiveness of MISA. Source code is available at this https URL. 

---
# FLoRA: Fused forward-backward adapters for parameter efficient fine-tuning and reducing inference-time latencies of LLMs 

**Authors**: Dhananjaya Gowda, Seoha Song, Junhyun Lee, Harshith Goka  

**Link**: [PDF](https://arxiv.org/pdf/2511.00050)  

**Abstract**: As the large language models (LLMs) grow in size each day, efficient training and fine-tuning has never been as important as nowadays. This resulted in the great interest in parameter efficient fine-tuning (PEFT), and effective methods including low-rank adapters (LoRA) has emerged. Although the various PEFT methods have been studied extensively in the recent years, the greater part of the subject remains unexplored with the huge degree of freedom. In this paper, we propose FLoRA, a family of fused forward-backward adapters (FFBA) for parameter-efficient fine-tuning of LLMs on downstream tasks. The FFBA combine ideas from the popular LoRA and parallel adapters to improve the overall fine-tuning accuracies. At the same time, latencies are minimized by fusing the forward and backward adapters into existing projection layers of the base model. Experimental results show that the proposed FFB adapters perform significantly better than the popularly used LoRA in both accuracy and latency for a similar parameter budget. 

---
# Chitchat with AI: Understand the supply chain carbon disclosure of companies worldwide through Large Language Model 

**Authors**: Haotian Hang, Yueyang Shen, Vicky Zhu, Jose Cruz, Michelle Li  

**Link**: [PDF](https://arxiv.org/pdf/2511.00024)  

**Abstract**: In the context of global sustainability mandates, corporate carbon disclosure has emerged as a critical mechanism for aligning business strategy with environmental responsibility. The Carbon Disclosure Project (CDP) hosts the world's largest longitudinal dataset of climate-related survey responses, combining structured indicators with open-ended narratives, but the heterogeneity and free-form nature of these disclosures present significant analytical challenges for benchmarking, compliance monitoring, and investment screening. This paper proposes a novel decision-support framework that leverages large language models (LLMs) to assess corporate climate disclosure quality at scale. It develops a master rubric that harmonizes narrative scoring across 11 years of CDP data (2010-2020), enabling cross-sector and cross-country benchmarking. By integrating rubric-guided scoring with percentile-based normalization, our method identifies temporal trends, strategic alignment patterns, and inconsistencies in disclosure across industries and regions. Results reveal that sectors such as technology and countries like Germany consistently demonstrate higher rubric alignment, while others exhibit volatility or superficial engagement, offering insights that inform key decision-making processes for investors, regulators, and corporate environmental, social, and governance (ESG) strategists. The proposed LLM-based approach transforms unstructured disclosures into quantifiable, interpretable, comparable, and actionable intelligence, advancing the capabilities of AI-enabled decision support systems (DSSs) in the domain of climate governance. 

---
# Feature-Guided SAE Steering for Refusal-Rate Control using Contrasting Prompts 

**Authors**: Samaksh Bhargav, Zining Zhu  

**Link**: [PDF](https://arxiv.org/pdf/2511.00029)  

**Abstract**: Large Language Model (LLM) deployment requires guiding the LLM to recognize and not answer unsafe prompts while complying with safe prompts. Previous methods for achieving this require adjusting model weights along with other expensive procedures. While recent advances in Sparse Autoencoders (SAEs) have enabled interpretable feature extraction from LLMs, existing approaches lack systematic feature selection methods and principled evaluation of safety-utility tradeoffs. We explored using different steering features and steering strengths using Sparse Auto Encoders (SAEs) to provide a solution. Using an accurate and innovative contrasting prompt method with the AI-Generated Prompts Dataset from teknium/OpenHermes-2p5-Mistral-7B and Air Bench eu-dataset to efficiently choose the best features in the model to steer, we tested this method on Llama-3 8B. We conclude that using this method, our approach achieves an 18.9% improvement in safety performance while simultaneously increasing utility by 11.1%, demonstrating that targeted SAE steering can overcome traditional safety-utility tradeoffs when optimal features are identified through principled selection methods. 

---
# Tool-to-Agent Retrieval: Bridging Tools and Agents for Scalable LLM Multi-Agent Systems 

**Authors**: Elias Lumer, Faheem Nizar, Anmol Gulati, Pradeep Honaganahalli Basavaraju, Vamse Kumar Subbiah  

**Link**: [PDF](https://arxiv.org/pdf/2511.01854)  

**Abstract**: Recent advances in LLM Multi-Agent Systems enable scalable orchestration of sub-agents, each coordinating hundreds or thousands of tools or Model Context Protocol (MCP) servers. However, existing retrieval methods typically match queries against coarse agent-level descriptions before routing, which obscures fine-grained tool functionality and often results in suboptimal agent selection. We introduce Tool-to-Agent Retrieval, a unified framework that embeds both tools and their parent agents in a shared vector space and connects them through metadata relationships. By explicitly representing tool capabilities and traversing metadata to the agent level, Tool-to-Agent Retrieval enables granular tool-level or agent-level retrieval, ensuring that agents and their underlying tools or MCP servers are equally represented without the context dilution that arises from chunking many tools together. Evaluating Tool-to-Agent Retrieval across eight embedding models, our approach achieves consistent improvements of 19.4% in Recall@5 and 17.7% in nDCG@5 over previous state-of-the-art agent retrievers on the LiveMCPBench benchmark. 

---
# Evaluating Cultural Knowledge Processing in Large Language Models: A Cognitive Benchmarking Framework Integrating Retrieval-Augmented Generation 

**Authors**: Hung-Shin Lee, Chen-Chi Chang, Ching-Yuan Chen, Yun-Hsiang Hsu  

**Link**: [PDF](https://arxiv.org/pdf/2511.01649)  

**Abstract**: This study proposes a cognitive benchmarking framework to evaluate how large language models (LLMs) process and apply culturally specific knowledge. The framework integrates Bloom's Taxonomy with Retrieval-Augmented Generation (RAG) to assess model performance across six hierarchical cognitive domains: Remembering, Understanding, Applying, Analyzing, Evaluating, and Creating. Using a curated Taiwanese Hakka digital cultural archive as the primary testbed, the evaluation measures LLM-generated responses' semantic accuracy and cultural relevance. 

---
# BARD: budget-aware reasoning distillation 

**Authors**: Lujie Niu, Lei Shen, Yi Jiang, Caixia Yuan, Xiaojie Wang, Wenbo Su, Bo zheng  

**Link**: [PDF](https://arxiv.org/pdf/2511.01470)  

**Abstract**: While long Chain-of-Thought (CoT) distillation effectively transfers reasoning capability to smaller language models, the reasoning process often remains redundant and computational budget uncontrollable, leading to inefficient resource usage. To address this limitation, we propose \textbf{Budget-Aware Reasoning Distillation (BARD)}, a novel framework that simultaneously distills reasoning capability and enables fine-grained control over the reasoning length. BARD uses the thinking budget as a user-specified control signal, allowing the model to dynamically balance reasoning performance and computational efficiency. To achieve this concept, BARD introduces a two-phase training regimen. The first phase, Supervised Fine-Tuning (SFT) on teacher-generated long CoT data compressed to various budget levels, bootstrapping the model's understanding of budget constraints. The second phase leverages Reinforcement Learning (RL) from a reward signal in consideration of reasoning performance and budget fidelity simultaneously. Incorporating the two-phase regimen is crucial to avoiding policy degradation and ensuring that both objectives are optimized jointly. Extensive experiments demonstrate that our method empowers an 8B student model to achieve strong performance on challenging reasoning benchmarks (\textit{AIME24, AIME25, GPQA}) while providing precise and adaptive control over its reasoning length across a wide range of budgets. 

---
# Towards Consistent Detection of Cognitive Distortions: LLM-Based Annotation and Dataset-Agnostic Evaluation 

**Authors**: Neha Sharma, Navneet Agarwal, Kairit Sirts  

**Link**: [PDF](https://arxiv.org/pdf/2511.01482)  

**Abstract**: Text-based automated Cognitive Distortion detection is a challenging task due to its subjective nature, with low agreement scores observed even among expert human annotators, leading to unreliable annotations. We explore the use of Large Language Models (LLMs) as consistent and reliable annotators, and propose that multiple independent LLM runs can reveal stable labeling patterns despite the inherent subjectivity of the task. Furthermore, to fairly compare models trained on datasets with different characteristics, we introduce a dataset-agnostic evaluation framework using Cohen's kappa as an effect size measure. This methodology allows for fair cross-dataset and cross-study comparisons where traditional metrics like F1 score fall short. Our results show that GPT-4 can produce consistent annotations (Fleiss's Kappa = 0.78), resulting in improved test set performance for models trained on these annotations compared to those trained on human-labeled data. Our findings suggest that LLMs can offer a scalable and internally consistent alternative for generating training data that supports strong downstream performance in subjective NLP tasks. 

---
# "Don't Teach Minerva": Guiding LLMs Through Complex Syntax for Faithful Latin Translation with RAG 

**Authors**: Sergio Torres Aguilar  

**Link**: [PDF](https://arxiv.org/pdf/2511.01454)  

**Abstract**: Translating a morphology-rich, low-resource language like Latin poses significant challenges. This paper introduces a reproducible draft-based refinement pipeline that elevates open-source Large Language Models (LLMs) to a performance level statistically comparable to top-tier proprietary systems. Our method first uses a fine-tuned NLLB-1.3B model to generate a high-quality, structurally faithful draft. A zero-shot LLM (Llama-3.3 or Qwen3) then polishes this draft, a process that can be further enhanced by augmenting the context with retrieved out-context examples (RAG). We demonstrate the robustness of this approach on two distinct benchmarks: a standard in-domain test set (Rosenthal, 2023) and a new, challenging out-of-domain (OOD) set of 12th-century Latin letters (2025). Our central finding is that this open-source RAG system achieves performance statistically comparable to the GPT-5 baseline, without any task-specific LLM fine-tuning. We release the pipeline, the Chartres OOD set, and evaluation scripts and models to facilitate replicability and further research. 

---
# LiveSearchBench: An Automatically Constructed Benchmark for Retrieval and Reasoning over Dynamic Knowledge 

**Authors**: Heng Zhou, Ao Yu, Yuchen Fan, Jianing Shi, Li Kang, Hejia Geng, Yongting Zhang, Yutao Fan, Yuhao Wu, Tiancheng He, Yiran Qin, Lei Bai, Zhenfei Yin  

**Link**: [PDF](https://arxiv.org/pdf/2511.01409)  

**Abstract**: Evaluating large language models (LLMs) on question answering often relies on static benchmarks that reward memorization and understate the role of retrieval, failing to capture the dynamic nature of world knowledge. We present LiveSearchBench, an automated pipeline for constructing retrieval-dependent benchmarks from recent knowledge updates. Our method computes deltas between successive Wikidata snapshots, filters candidate triples for quality, and synthesizes natural-language questions at three levels of reasoning difficulty, each guaranteed to admit a unique, verifiable answer through SPARQL validation. The pipeline is fully automated, scalable across time, and minimizes human intervention, enabling continual regeneration of temporally grounded benchmarks. Experiments show a pronounced performance drop when models confront facts that post-date pretraining, with the gap most salient on multi-hop queries. Retrieval augmented methods and larger, instruction-tuned models provide partial gains but fail to close this recency gap. By design, LiveSearchBench shifts evaluation from static memorization toward tasks that require up-to-date retrieval and reasoning, offering a foundation for systematic, long-term assessment of LLMs under evolving knowledge. 

---
# FirstAidQA: A Synthetic Dataset for First Aid and Emergency Response in Low-Connectivity Settings 

**Authors**: Saiyma Sittul Muna, Rezwan Islam Salvi, Mushfiqur Rahman Mushfique, Ajwad Abrar  

**Link**: [PDF](https://arxiv.org/pdf/2511.01289)  

**Abstract**: In emergency situations, every second counts. The deployment of Large Language Models (LLMs) in time-sensitive, low or zero-connectivity environments remains limited. Current models are computationally intensive and unsuitable for low-tier devices often used by first responders or civilians. A major barrier to developing lightweight, domain-specific solutions is the lack of high-quality datasets tailored to first aid and emergency response. To address this gap, we introduce FirstAidQA, a synthetic dataset containing 5,500 high-quality question answer pairs that encompass a wide range of first aid and emergency response scenarios. The dataset was generated using a Large Language Model, ChatGPT-4o-mini, with prompt-based in-context learning, using texts from the Vital First Aid Book (2019). We applied preprocessing steps such as text cleaning, contextual chunking, and filtering, followed by human validation to ensure accuracy, safety, and practical relevance of the QA pairs. FirstAidQA is designed to support instruction-tuning and fine-tuning of LLMs and Small Language Models (SLMs), enabling faster, more reliable, and offline-capable systems for emergency settings. We publicly release the dataset to advance research on safety-critical and resource-constrained AI applications in first aid and emergency response. The dataset is available on Hugging Face at this https URL. 

---
# AraFinNews: Arabic Financial Summarisation with Domain-Adapted LLMs 

**Authors**: Mo El-Haj, Paul Rayson  

**Link**: [PDF](https://arxiv.org/pdf/2511.01265)  

**Abstract**: This paper investigates the impact of domain specificity on abstractive summarisation of Arabic financial texts using large language models (LLMs). We introduce AraFinNews, the largest publicly available Arabic financial news dataset to date, comprising 212,500 article--headline pairs spanning nearly a decade of reporting from October 2015 to July 2025. Designed as the Arabic equivalent of major English summarisation corpora such as CNN/DailyMail, AraFinNews provides a robust benchmark for evaluating domain-specific language understanding and generation in financial contexts. Using this resource, we evaluate transformer-based models -- including mT5, AraT5, and the domain-adapted FinAraT5 -- to examine how financial-domain pretraining influences factual accuracy, numerical reliability, and stylistic alignment with professional reporting. Experimental results show that domain-adapted models generate more faithful and coherent summaries, particularly in handling quantitative and entity-centric information. The findings highlight the importance of domain-specific adaptation for improving factual consistency and narrative fluency in Arabic financial summarisation. The dataset is freely available for non-commercial research at this https URL. 

---
# The Ouroboros of Benchmarking: Reasoning Evaluation in an Era of Saturation 

**Authors**: İbrahim Ethem Deveci, Duygu Ataman  

**Link**: [PDF](https://arxiv.org/pdf/2511.01365)  

**Abstract**: The rapid rise of Large Language Models (LLMs) and Large Reasoning Models (LRMs) has been accompanied by an equally rapid increase of benchmarks used to assess them. However, due to both improved model competence resulting from scaling and novel training advances as well as likely many of these datasets being included in pre or post training data, results become saturated, driving a continuous need for new and more challenging replacements. In this paper, we discuss whether surpassing a benchmark truly demonstrates reasoning ability or are we simply tracking numbers divorced from the capabilities we claim to measure? We present an investigation focused on three model families, OpenAI, Anthropic, and Google, and how their reasoning capabilities across different benchmarks evolve over the years. We also analyze performance trends over the years across different reasoning tasks and discuss the current situation of benchmarking and remaining challenges. By offering a comprehensive overview of benchmarks and reasoning tasks, our work aims to serve as a first reference to ground future research in reasoning evaluation and model development. 

---
# Safer in Translation? Presupposition Robustness in Indic Languages 

**Authors**: Aadi Palnitkar, Arjun Suresh, Rishi Rajesh, Puneet Puli  

**Link**: [PDF](https://arxiv.org/pdf/2511.01360)  

**Abstract**: Increasingly, more and more people are turning to large language models (LLMs) for healthcare advice and consultation, making it important to gauge the efficacy and accuracy of the responses of LLMs to such queries. While there are pre-existing medical benchmarks literature which seeks to accomplish this very task, these benchmarks are almost universally in English, which has led to a notable gap in existing literature pertaining to multilingual LLM evaluation. Within this work, we seek to aid in addressing this gap with Cancer-Myth-Indic, an Indic language benchmark built by translating a 500-item subset of Cancer-Myth, sampled evenly across its original categories, into five under-served but widely used languages from the subcontinent (500 per language; 2,500 translated items total). Native-speaker translators followed a style guide for preserving implicit presuppositions in translation; items feature false presuppositions relating to cancer. We evaluate several popular LLMs under this presupposition stress. 

---
# Surfacing Subtle Stereotypes: A Multilingual, Debate-Oriented Evaluation of Modern LLMs 

**Authors**: Muhammed Saeed, Muhammad Abdul-mageed, Shady Shehata  

**Link**: [PDF](https://arxiv.org/pdf/2511.01187)  

**Abstract**: Large language models (LLMs) are widely deployed for open-ended communication, yet most bias evaluations still rely on English, classification-style tasks. We introduce DebateBias-8K, a new multilingual, debate-style benchmark designed to reveal how narrative bias appears in realistic generative settings. Our dataset includes 8,400 structured debate prompts spanning four sensitive domains: women's rights, socioeconomic development, terrorism, and religion, across seven languages ranging from high-resource (English, Chinese) to low-resource (Swahili, Nigerian Pidgin). Using four flagship models (GPT-4o, Claude 3, DeepSeek, and LLaMA 3), we generate and automatically classify over 100,000 responses. Results show that all models reproduce entrenched stereotypes despite safety alignment: Arabs are overwhelmingly linked to terrorism and religion (>=95%), Africans to socioeconomic "backwardness" (up to <=77%), and Western groups are consistently framed as modern or progressive. Biases grow sharply in lower-resource languages, revealing that alignment trained primarily in English does not generalize globally. Our findings highlight a persistent divide in multilingual fairness: current alignment methods reduce explicit toxicity but fail to prevent biased outputs in open-ended contexts. We release our DebateBias-8K benchmark and analysis framework to support the next generation of multilingual bias evaluation and safer, culturally inclusive model alignment. 

---
# Building a Silver-Standard Dataset from NICE Guidelines for Clinical LLMs 

**Authors**: Qing Ding, Eric Hua Qing Zhang, Felix Jozsa, Julia Ive  

**Link**: [PDF](https://arxiv.org/pdf/2511.01053)  

**Abstract**: Large language models (LLMs) are increasingly used in healthcare, yet standardised benchmarks for evaluating guideline-based clinical reasoning are missing. This study introduces a validated dataset derived from publicly available guidelines across multiple diagnoses. The dataset was created with the help of GPT and contains realistic patient scenarios, as well as clinical questions. We benchmark a range of recent popular LLMs to showcase the validity of our dataset. The framework supports systematic evaluation of LLMs' clinical utility and guideline adherence. 

---
# MicroRemed: Benchmarking LLMs in Microservices Remediation 

**Authors**: Lingzhe Zhang, Yunpeng Zhai, Tong Jia, Chiming Duan, Minghua He, Leyi Pan, Zhaoyang Liu, Bolin Ding, Ying Li  

**Link**: [PDF](https://arxiv.org/pdf/2511.01166)  

**Abstract**: Large Language Models (LLMs) integrated with agent-based reasoning frameworks have recently shown strong potential for autonomous decision-making and system-level operations. One promising yet underexplored direction is microservice remediation, where the goal is to automatically recover faulty microservice systems. Existing approaches, however, still rely on human-crafted prompts from Site Reliability Engineers (SREs), with LLMs merely converting textual instructions into executable code. To advance research in this area, we introduce MicroRemed, the first benchmark for evaluating LLMs in end-to-end microservice remediation, where models must directly generate executable Ansible playbooks from diagnosis reports to restore system functionality. We further propose ThinkRemed, a multi-agent framework that emulates the reflective and perceptive reasoning of SREs. Experimental results show that MicroRemed presents substantial challenges to current LLMs, while ThinkRemed improves end-to-end remediation performance through iterative reasoning and system reflection. The benchmark is available at this https URL. 

---
# Prompt-R1: Collaborative Automatic Prompting Framework via End-to-end Reinforcement Learning 

**Authors**: Wenjin Liu, Haoran Luo, Xueyuan Lin, Haoming Liu, Tiesunlong Shen, Jiapu Wang, Rui Mao, Erik Cambria  

**Link**: [PDF](https://arxiv.org/pdf/2511.01016)  

**Abstract**: Recently, advanced large language models (LLMs) have emerged at an increasingly rapid pace. However, when faced with complex problems, most users are often unable to provide accurate and effective prompts to interact with LLMs, thus limiting the performance of LLMs. To address this challenge, we propose Prompt-R1, an end-to-end reinforcement learning framework that uses a small-scale LLM to collaborate with large-scale LLMs, replacing user interaction to solve problems better. This collaboration is cast as a multi-turn prompt interaction, where the small-scale LLM thinks and generates prompts, and the large-scale LLM performs complex reasoning. A dual-constrained reward is designed to optimize for correctness, generation quality, and reasoning accuracy. Prompt-R1 provides a plug-and-play framework that supports both inference and training with various large-scale LLMs. Experiments on multiple public datasets show that Prompt-R1 significantly outperforms baseline models across tasks. Our code is publicly available at this https URL. 

---
# Improving Romanian LLM Pretraining Data using Diversity and Quality Filtering 

**Authors**: Vlad Negoita, Mihai Masala, Traian Rebedea  

**Link**: [PDF](https://arxiv.org/pdf/2511.01090)  

**Abstract**: Large Language Models (LLMs) have recently exploded in popularity, often matching or outperforming human abilities on many tasks. One of the key factors in training LLMs is the availability and curation of high-quality data. Data quality is especially crucial for under-represented languages, where high-quality corpora are scarce. In this work we study the characteristics and coverage of Romanian pretraining corpora and we examine how they differ from English data. By training a lightweight multitask model on carefully LLM-annotated Romanian texts, we are able to analyze and perform multi-level filtering (e.g., educational value, topic, format) to generate high-quality pretraining datasets. Our experiments show noteworthy trends in the topics present in Romanian and English data, while also proving the effectiveness of filtering data through improved LLM pretraining performance across multiple benchmarks. 

---
# VayuChat: An LLM-Powered Conversational Interface for Air Quality Data Analytics 

**Authors**: Vedant Acharya, Abhay Pisharodi, Rishabh Mondal, Mohammad Rafiuddin, Nipun Batra  

**Link**: [PDF](https://arxiv.org/pdf/2511.01046)  

**Abstract**: Air pollution causes about 1.6 million premature deaths each year in India, yet decision makers struggle to turn dispersed data into decisions. Existing tools require expertise and provide static dashboards, leaving key policy questions unresolved. We present VayuChat, a conversational system that answers natural language questions on air quality, meteorology, and policy programs, and responds with both executable Python code and interactive visualizations. VayuChat integrates data from Central Pollution Control Board (CPCB) monitoring stations, state-level demographics, and National Clean Air Programme (NCAP) funding records into a unified interface powered by large language models. Our live demonstration will show how users can perform complex environmental analytics through simple conversations, making data science accessible to policymakers, researchers, and citizens. The platform is publicly deployed at this https URL VayuChat. For further information check out video uploaded on this https URL. 

---
# HPLT~3.0: Very Large-Scale Multilingual Resources for LLM and MT. Mono- and Bi-lingual Data, Multilingual Evaluation, and Pre-Trained Models 

**Authors**: Stephan Oepen, Nikolay Arefev, Mikko Aulamo, Marta Bañón, Maja Buljan, Laurie Burchell, Lucas Charpentier, Pinzhen Chen, Mariya Fedorova, Ona de Gibert, Barry Haddow, Jan Hajič, Jindrič Helcl, Andrey Kutuzov, Zihao Li, Risto Luukkonen, Bhavitvya Malik, Vladislav Mikhailov, Amanda Myntti, Dayyán O'Brien, Lucie Poláková, Sampo Pyysalo, Gema Ramírez Sánchez, Janine Siewert, Pavel Stepachev, Jörg Tiedemann, Teemu Vahtola, Fedor Vitiugin, Tea Vojtěchová, Jaume Zaragoza  

**Link**: [PDF](https://arxiv.org/pdf/2511.01066)  

**Abstract**: We present an ongoing initiative to provide open, very large, high-quality, and richly annotated textual datasets for almost 200 languages. At 30 trillion tokens, this is likely the largest generally available multilingual collection of LLM pre-training data. At 30 trillion tokens, this is likely the largest generally available multilingual collection of LLM pre-training data. These datasets are derived from web crawls from different sources and accompanied with a complete, open-source pipeline for document selection from web archives, text extraction from HTML, language identification for noisy texts, exact and near-deduplication, annotation with, among others, register labels, text quality estimates, and personally identifiable information; and final selection and filtering. We report on data quality probes through contrastive and analytical statistics, through manual inspection of samples for 24 languages, and through end-to-end evaluation of various language model architectures trained on this data. For multilingual LLM evaluation, we provide a comprehensive collection of benchmarks for nine European languages, with special emphasis on natively created tasks, mechanisms to mitigate prompt sensitivity, and refined normalization and aggregation of scores. Additionally, we train and evaluate a family of 57 monolingual encoder-decoder models, as well as a handful of monolingual GPT-like reference models. Besides the monolingual data and models, we also present a very large collection of parallel texts automatically mined from this data, together with a novel parallel corpus synthesized via machine translation. 

---
# IF-CRITIC: Towards a Fine-Grained LLM Critic for Instruction-Following Evaluation 

**Authors**: Bosi Wen, Yilin Niu, Cunxiang Wang, Pei Ke, Xiaoying Ling, Ying Zhang, Aohan Zeng, Hongning Wang, Minlie Huang  

**Link**: [PDF](https://arxiv.org/pdf/2511.01014)  

**Abstract**: Instruction following is a fundamental ability of Large Language Models (LLMs), requiring their generated outputs to follow multiple constraints imposed in input instructions. Numerous studies have attempted to enhance this ability through preference optimization or reinforcement learning based on reward signals from LLM-as-a-Judge. However, existing evaluation models for instruction following still possess many deficiencies, such as substantial costs and unreliable assessments. To this end, we propose IF-CRITIC, an LLM critic that can provide efficient and reliable assessments of constraint following in the instructions. We first develop a checklist generator to decompose instructions and generate constraint checklists. With the assistance of the checklists, we collect high-quality critique training data through a multi-stage critique filtering mechanism and employ a constraint-level preference optimization method to train IF-CRITIC. Extensive experiments demonstrate that the evaluation performance of IF-CRITIC can beat strong LLM-as-a-Judge baselines, including Deepseek-R1 and o4-mini. With the scalable reward signals provided by IF-CRITIC, LLMs can achieve substantial performance gains in instruction-following optimization under lower computational overhead compared to strong LLM critic baselines. 

---
# The Biased Oracle: Assessing LLMs' Understandability and Empathy in Medical Diagnoses 

**Authors**: Jianzhou Yao, Shunchang Liu, Guillaume Drui, Rikard Pettersson, Alessandro Blasimme, Sara Kijewski  

**Link**: [PDF](https://arxiv.org/pdf/2511.00924)  

**Abstract**: Large language models (LLMs) show promise for supporting clinicians in diagnostic communication by generating explanations and guidance for patients. Yet their ability to produce outputs that are both understandable and empathetic remains uncertain. We evaluate two leading LLMs on medical diagnostic scenarios, assessing understandability using readability metrics as a proxy and empathy through LLM-as-a-Judge ratings compared to human evaluations. The results indicate that LLMs adapt explanations to socio-demographic variables and patient conditions. However, they also generate overly complex content and display biased affective empathy, leading to uneven accessibility and support. These patterns underscore the need for systematic calibration to ensure equitable patient communication. The code and data are released: this https URL 

---
# Do Methods to Jailbreak and Defend LLMs Generalize Across Languages? 

**Authors**: Berk Atil, Rebecca J. Passonneau, Fred Morstatter  

**Link**: [PDF](https://arxiv.org/pdf/2511.00689)  

**Abstract**: Large language models (LLMs) undergo safety alignment after training and tuning, yet recent work shows that safety can be bypassed through jailbreak attacks. While many jailbreaks and defenses exist, their cross-lingual generalization remains underexplored. This paper presents the first systematic multilingual evaluation of jailbreaks and defenses across ten languages--spanning high-, medium-, and low-resource languages--using six LLMs on HarmBench and AdvBench. We assess two jailbreak types: logical-expression-based and adversarial-prompt-based. For both types, attack success and defense robustness vary across languages: high-resource languages are safer under standard queries but more vulnerable to adversarial ones. Simple defenses can be effective, but are language- and model-dependent. These findings call for language-aware and cross-lingual safety benchmarks for LLMs. 

---
# Do You Know About My Nation? Investigating Multilingual Language Models' Cultural Literacy Through Factual Knowledge 

**Authors**: Eshaan Tanwar, Anwoy Chatterjee, Michael Saxon, Alon Albalak, William Yang Wang, Tanmoy Chakraborty  

**Link**: [PDF](https://arxiv.org/pdf/2511.00657)  

**Abstract**: Most multilingual question-answering benchmarks, while covering a diverse pool of languages, do not factor in regional diversity in the information they capture and tend to be Western-centric. This introduces a significant gap in fairly evaluating multilingual models' comprehension of factual information from diverse geographical locations. To address this, we introduce XNationQA for investigating the cultural literacy of multilingual LLMs. XNationQA encompasses a total of 49,280 questions on the geography, culture, and history of nine countries, presented in seven languages. We benchmark eight standard multilingual LLMs on XNationQA and evaluate them using two novel transference metrics. Our analyses uncover a considerable discrepancy in the models' accessibility to culturally specific facts across languages. Notably, we often find that a model demonstrates greater knowledge of cultural information in English than in the dominant language of the respective culture. The models exhibit better performance in Western languages, although this does not necessarily translate to being more literate for Western countries, which is counterintuitive. Furthermore, we observe that models have a very limited ability to transfer knowledge across languages, particularly evident in open-source models. 

---
# Word Salad Chopper: Reasoning Models Waste A Ton Of Decoding Budget On Useless Repetitions, Self-Knowingly 

**Authors**: Wenya Xie, Shaochen, Zhong, Hoang Anh Duy Le, Zhaozhuo Xu, Jianwen Xie, Zirui Liu  

**Link**: [PDF](https://arxiv.org/pdf/2511.00536)  

**Abstract**: Large Reasoning Models (LRMs) are often bottlenecked by the high cost of output tokens. We show that a significant portion of these tokens are useless self-repetitions - what we call "word salad" - that exhaust the decoding budget without adding value. Interestingly, we observe that LRMs are self-aware when trapped in these loops: the hidden states of <\n\n> tokens trailing each reasoning chunk exhibit patterns that allow us to detect word salad behavior on-the-fly via a single-layer linear classifier. Once detected, a simple chop appended by a straightforward regeneration prompt yields substantial length savings with minimal quality loss. Our work offers WordSaladChopper (WSC) - a lightweight, turnkey component for LRM that is minimally invasive to its reasoning trajectory by only removing semantically redundant tokens. Given its low overhead, strong savings, and the lack of semantic value of word salad tokens, we believe it is not too far-fetched to argue that WSC - or a similar component - is a must-have for all LRM applications with user experience in mind. Our code is publicly available at this https URL. 

---
# OpenSIR: Open-Ended Self-Improving Reasoner 

**Authors**: Wai-Chung Kwan, Joshua Ong Jun Leang, Pavlos Vougiouklis, Jeff Z. Pan, Marco Valentino, Pasquale Minervini  

**Link**: [PDF](https://arxiv.org/pdf/2511.00602)  

**Abstract**: Recent advances in large language model (LLM) reasoning through reinforcement learning rely on annotated datasets for verifiable rewards, which may limit models' ability to surpass human-level performance. While self-play offers a promising alternative, existing approaches depend on external verifiers or cannot learn open-endedly. We present Open-Ended Self-Improving Reasoner (OpenSIR), a self-play framework where an LLM learns to generate and solve novel problems by alternating teacher and student roles without external supervision. To generate novel problems, OpenSIR optimises for both difficulty and diversity, rewarding problems that challenge appropriately while exploring distinct concepts, enabling open-ended mathematical discovery. Starting from a single trivial seed problem, OpenSIR substantially improves instruction models: Llama-3.2-3B-Instruct advances from 73.9 to 78.3 on GSM8K, and from 28.8 to 34.4 on College Math, while Gemma-2-2B-Instruct rises from 38.5 to 58.7 on GSM8K. Our analyses reveal that OpenSIR achieves open-ended learning through co-evolving teacher-student roles that adaptively calibrate difficulty and drive diverse exploration, progressing autonomously from basic to advanced mathematics. 

---
# SpecDiff-2: Scaling Diffusion Drafter Alignment For Faster Speculative Decoding 

**Authors**: Jameson Sandler, Jacob K. Christopher, Thomas Hartvigsen, Nando Fioretto  

**Link**: [PDF](https://arxiv.org/pdf/2511.00606)  

**Abstract**: Speculative decoding has become the standard approach for accelerating Large Language Model (LLM) inference. It exploits a lossless draft-then-verify procedure to circumvent the latency of autoregressive decoding, achieving impressive speed-ups. Yet, current speculative decoding approaches remain limited by two fundamental bottlenecks: (1) the autoregressive dependency during drafting which limits parallelism, and (2) frequent rejections of draft tokens caused by misalignment between the draft and verify models. This paper proposes SpecDiff-2, a novel framework to jointly address these two bottlenecks. It leverages discrete diffusion as a non-autoregressive drafter to address bottleneck (1) and develops novel techniques to calibrate discrete diffusion drafters with autoregressive verifiers, addressing bottleneck (2). Experimental results across a comprehensive benchmark suite show that SpecDiff-2 achieves a new state-of-the-art across reasoning, coding, and mathematical benchmarks, improving tokens-per-second by up to an average of +55% over previous baselines and obtaining up to 5.5x average speed-up over standard decoding, without any loss of accuracy. 

---
# Friend or Foe: How LLMs' Safety Mind Gets Fooled by Intent Shift Attack 

**Authors**: Peng Ding, Jun Kuang, Wen Sun, Zongyu Wang, Xuezhi Cao, Xunliang Cai, Jiajun Chen, Shujian Huang  

**Link**: [PDF](https://arxiv.org/pdf/2511.00556)  

**Abstract**: Large language models (LLMs) remain vulnerable to jailbreaking attacks despite their impressive capabilities. Investigating these weaknesses is crucial for robust safety mechanisms. Existing attacks primarily distract LLMs by introducing additional context or adversarial tokens, leaving the core harmful intent unchanged. In this paper, we introduce ISA (Intent Shift Attack), which obfuscates LLMs about the intent of the attacks. More specifically, we establish a taxonomy of intent transformations and leverage them to generate attacks that may be misperceived by LLMs as benign requests for information. Unlike prior methods relying on complex tokens or lengthy context, our approach only needs minimal edits to the original request, and yields natural, human-readable, and seemingly harmless prompts. Extensive experiments on both open-source and commercial LLMs show that ISA achieves over 70% improvement in attack success rate compared to direct harmful prompts. More critically, fine-tuning models on only benign data reformulated with ISA templates elevates success rates to nearly 100%. For defense, we evaluate existing methods and demonstrate their inadequacy against ISA, while exploring both training-free and training-based mitigation strategies. Our findings reveal fundamental challenges in intent inference for LLMs safety and underscore the need for more effective defenses. Our code and datasets are available at this https URL. 

---
# ToM: Leveraging Tree-oriented MapReduce for Long-Context Reasoning in Large Language Models 

**Authors**: Jiani Guo, Zuchao Li, Jie Wu, Qianren Wang, Yun Li, Lefei Zhang, Hai Zhao, Yujiu Yang  

**Link**: [PDF](https://arxiv.org/pdf/2511.00489)  

**Abstract**: Large Language Models (LLMs), constrained by limited context windows, often face significant performance degradation when reasoning over long contexts. To address this, Retrieval-Augmented Generation (RAG) retrieves and reasons over chunks but frequently sacrifices logical coherence due to its reliance on similarity-based rankings. Similarly, divide-and-conquer frameworks (DCF) split documents into small chunks for independent reasoning and aggregation. While effective for local reasoning, DCF struggles to capture long-range dependencies and risks inducing conflicts by processing chunks in isolation. To overcome these limitations, we propose ToM, a novel Tree-oriented MapReduce framework for long-context reasoning. ToM leverages the inherent hierarchical structure of long documents (e.g., main headings and subheadings) by constructing a DocTree through hierarchical semantic parsing and performing bottom-up aggregation. Using a Tree MapReduce approach, ToM enables recursive reasoning: in the Map step, rationales are generated at child nodes; in the Reduce step, these rationales are aggregated across sibling nodes to resolve conflicts or reach consensus at parent nodes. Experimental results on 70B+ LLMs show that ToM significantly outperforms existing divide-and-conquer frameworks and retrieval-augmented generation methods, achieving better logical coherence and long-context reasoning. Our code is available at this https URL . 

---
# G2: Guided Generation for Enhanced Output Diversity in LLMs 

**Authors**: Zhiwen Ruan, Yixia Li, Yefeng Liu, Yun Chen, Weihua Luo, Peng Li, Yang Liu, Guanhua Chen  

**Link**: [PDF](https://arxiv.org/pdf/2511.00432)  

**Abstract**: Large Language Models (LLMs) have demonstrated exceptional performance across diverse natural language processing tasks. However, these models exhibit a critical limitation in output diversity, often generating highly similar content across multiple attempts. This limitation significantly affects tasks requiring diverse outputs, from creative writing to reasoning. Existing solutions, like temperature scaling, enhance diversity by modifying probability distributions but compromise output quality. We propose Guide-to-Generation (G2), a training-free plug-and-play method that enhances output diversity while preserving generation quality. G2 employs a base generator alongside dual Guides, which guide the generation process through decoding-based interventions to encourage more diverse outputs conditioned on the original query. Comprehensive experiments demonstrate that G2 effectively improves output diversity while maintaining an optimal balance between diversity and quality. 

---
# Zero-RAG: Towards Retrieval-Augmented Generation with Zero Redundant Knowledge 

**Authors**: Qi Luo, Xiaonan Li, Junqi Dai, Shuang Cheng, Xipeng Qiu  

**Link**: [PDF](https://arxiv.org/pdf/2511.00505)  

**Abstract**: Retrieval-Augmented Generation has shown remarkable results to address Large Language Models' hallucinations, which usually uses a large external corpus to supplement knowledge to LLMs. However, with the development of LLMs, the internal knowledge of LLMs has expanded significantly, thus causing significant knowledge redundancy between the external corpus and LLMs. On the one hand, the indexing cost of dense retrieval is highly related to the corpus size and thus significant redundant knowledge intensifies the dense retrieval's workload. On the other hand, the redundant knowledge in the external corpus is not helpful to LLMs and our exploratory analysis shows that it instead hurts the RAG performance on those questions which the LLM can answer by itself. To address these issues, we propose Zero-RAG to tackle these challenges. Specifically, we first propose the Mastery-Score metric to identify redundant knowledge in the RAG corpus to prune it. After pruning, answers to "mastered" questions rely primarily on internal knowledge of the LLM. To better harness the internal capacity, we propose Query Router and Noise-Tolerant Tuning to avoid the irrelevant documents' distraction and thus further improve the LLM's utilization of internal knowledge with pruned corpus. Experimental results show that Zero-RAG prunes the Wikipedia corpus by 30\% and accelerates the retrieval stage by 22\%, without compromising RAG's performance. 

---
# Reasoning Trajectories for Socratic Debugging of Student Code: From Misconceptions to Contradictions and Updated Beliefs 

**Authors**: Erfan Al-Hossami, Razvan Bunescu  

**Link**: [PDF](https://arxiv.org/pdf/2511.00371)  

**Abstract**: In Socratic debugging, instructors guide students towards identifying and fixing a bug on their own, instead of providing the bug fix directly. Most novice programmer bugs are caused by programming misconceptions, namely false beliefs about a programming concept. In this context, Socratic debugging can be formulated as a guided Reasoning Trajectory (RT) leading to a statement about the program behavior that contradicts the bug-causing misconception. Upon reaching this statement, the ensuing cognitive dissonance leads the student to first identify and then update their false belief. In this paper, we introduce the task of reasoning trajectory generation, together with a dataset of debugging problems manually annotated with RTs. We then describe LLM-based solutions for generating RTs and Socratic conversations that are anchored on them. A large-scale LLM-as-judge evaluation shows that frontier models can generate up to 91% correct reasoning trajectories and 98.7% valid conversation turns. 

---
# LingGym: How Far Are LLMs from Thinking Like Field Linguists? 

**Authors**: Changbing Yang, Franklin Ma, Freda Shi, Jian Zhu  

**Link**: [PDF](https://arxiv.org/pdf/2511.00343)  

**Abstract**: This paper introduces LingGym, a new benchmark that evaluates LLMs' capacity for meta-linguistic reasoning using Interlinear Glossed Text (IGT) and grammatical descriptions extracted from 18 typologically diverse reference grammars. Unlike previous work that focuses on specific downstream tasks, we assess whether LLMs can generalize linguistic inference across low-resource languages and structures not seen during training. We present a controlled evaluation task: Word-Gloss Inference, in which the model must infer a missing word and gloss from context using varying levels of linguistic information (e.g., glosses, grammatical explanations, translations). Our results show that incorporating structured linguistic cues leads to consistent improvements in reasoning performance across all models. This work highlights both the promise and current limitations of using LLMs for typologically informed linguistic analysis and low-resource language documentation. 

---
# PlotCraft: Pushing the Limits of LLMs for Complex and Interactive Data Visualization 

**Authors**: Jiajun Zhang, Jianke Zhang, Zeyu Cui, Jiaxi Yang, Lei Zhang, Binyuan Hui, Qiang Liu, Zilei Wang, Liang Wang, Junyang Lin  

**Link**: [PDF](https://arxiv.org/pdf/2511.00010)  

**Abstract**: Recent Large Language Models (LLMs) have demonstrated remarkable profi- ciency in code generation. However, their ability to create complex visualiza- tions for scaled and structured data remains largely unevaluated and underdevel- oped. To address this gap, we introduce PlotCraft, a new benchmark featuring 1k challenging visualization tasks that cover a wide range of topics, such as fi- nance, scientific research, and sociology. The benchmark is structured around seven high-level visualization tasks and encompasses 48 distinct chart types. Cru- cially, it is the first to systematically evaluate both single-turn generation and multi-turn refinement across a diverse spectrum of task complexities. Our com- prehensive evaluation of 23 leading LLMs on PlotCraft reveals obvious per- formance deficiencies in handling sophisticated visualization tasks. To bridge this performance gap, we develope SynthVis-30K, a large-scale, high-quality dataset of complex visualization code synthesized via a collaborative agent frame- work. Building upon this dataset, we develope PlotCraftor, a novel code gener- ation model that achieves strong capabilities in complex data visualization with a remarkably small size. Across VisEval, PandasPlotBench, and our proposed PlotCraft, PlotCraftor shows performance comparable to that of leading propri- etary approaches. Especially, on hard task, Our model achieves over 50% per- formance improvement. We will release the benchmark, dataset, and code at this https URL. 

---
# AgentBnB: A Browser-Based Cybersecurity Tabletop Exercise with Large Language Model Support and Retrieval-Aligned Scaffolding 

**Authors**: Arman Anwar, Zefang Liu  

**Link**: [PDF](https://arxiv.org/pdf/2511.00265)  

**Abstract**: Traditional cybersecurity tabletop exercises (TTXs) provide valuable training but are often scripted, resource-intensive, and difficult to scale. We introduce AgentBnB, a browser-based re-imagining of the Backdoors & Breaches game that integrates large language model teammates with a Bloom-aligned, retrieval-augmented copilot (C2D2). The system expands a curated corpus into factual, conceptual, procedural, and metacognitive snippets, delivering on-demand, cognitively targeted hints. Prompt-engineered agents employ a scaffolding ladder that gradually fades as learner confidence grows. In a solo-player pilot with four graduate students, participants reported greater intention to use the agent-based version compared to the physical card deck and viewed it as more scalable, though a ceiling effect emerged on a simple knowledge quiz. Despite limitations of small sample size, single-player focus, and narrow corpus, these early findings suggest that large language model augmented TTXs can provide lightweight, repeatable practice without the logistical burden of traditional exercises. Planned extensions include multi-player modes, telemetry-driven coaching, and comparative studies with larger cohorts. 

---
# Novelty and Impact of Economics Papers 

**Authors**: Chaofeng Wu  

**Link**: [PDF](https://arxiv.org/pdf/2511.01211)  

**Abstract**: We propose a framework that recasts scientific novelty not as a single attribute of a paper, but as a reflection of its position within the evolving intellectual landscape. We decompose this position into two orthogonal dimensions: \textit{spatial novelty}, which measures a paper's intellectual distinctiveness from its neighbors, and \textit{temporal novelty}, which captures its engagement with a dynamic research frontier. To operationalize these concepts, we leverage Large Language Models to develop semantic isolation metrics that quantify a paper's location relative to the full-text literature. Applying this framework to a large corpus of economics articles, we uncover a fundamental trade-off: these two dimensions predict systematically different outcomes. Temporal novelty primarily predicts citation counts, whereas spatial novelty predicts disruptive impact. This distinction allows us to construct a typology of semantic neighborhoods, identifying four archetypes associated with distinct and predictable impact profiles. Our findings demonstrate that novelty can be understood as a multidimensional construct whose different forms, reflecting a paper's strategic location, have measurable and fundamentally distinct consequences for scientific progress. 

---
# FEval-TTC: Fair Evaluation Protocol for Test-Time Compute 

**Authors**: Pavel Rumiantsev, Soumyasundar Pal, Yingxue Zhang, Mark Coates  

**Link**: [PDF](https://arxiv.org/pdf/2511.01203)  

**Abstract**: The performance of Large Language Models (LLMs) and the associated dollar costs of API calls can fluctuate over time, potentially invalidating conclusions drawn in prior research. To address this, we propose a Fair Evaluation protocol for Test-Time Compute (FEval-TTC), designed to ensure consistent assessment of test-time compute (TTC) methods, regardless of such fluctuations. FEval-TTC focuses on the evaluation of TTC methods that utilize underlying Chains-of-Thought (CoT). It supports evaluations across multiple LLMs on a diverse set of mathematical and commonsense reasoning datasets. The few-shot prompting and answer extraction processes are standardized across datasets, reducing both time and monetary overhead for researchers. Furthermore, we provide a cost modelling procedure that estimates both the token and dollar cost per query, facilitating equitable comparisons of prevalent TTC methods. We open-source FEval-TTC for public use at this https URL . 

---
# Actial: Activate Spatial Reasoning Ability of Multimodal Large Language Models 

**Authors**: Xiaoyu Zhan, Wenxuan Huang, Hao Sun, Xinyu Fu, Changfeng Ma, Shaosheng Cao, Bohan Jia, Shaohui Lin, Zhenfei Yin, Lei Bai, Wanli Ouyang, Yuanqi Li, Jie Guo, Yanwen Guo  

**Link**: [PDF](https://arxiv.org/pdf/2511.01618)  

**Abstract**: Recent advances in Multimodal Large Language Models (MLLMs) have significantly improved 2D visual understanding, prompting interest in their application to complex 3D reasoning tasks. However, it remains unclear whether these models can effectively capture the detailed spatial information required for robust real-world performance, especially cross-view consistency, a key requirement for accurate 3D reasoning. Considering this issue, we introduce Viewpoint Learning, a task designed to evaluate and improve the spatial reasoning capabilities of MLLMs. We present the Viewpoint-100K dataset, consisting of 100K object-centric image pairs with diverse viewpoints and corresponding question-answer pairs. Our approach employs a two-stage fine-tuning strategy: first, foundational knowledge is injected to the baseline MLLM via Supervised Fine-Tuning (SFT) on Viewpoint-100K, resulting in significant improvements across multiple tasks; second, generalization is enhanced through Reinforcement Learning using the Group Relative Policy Optimization (GRPO) algorithm on a broader set of questions. Additionally, we introduce a hybrid cold-start initialization method designed to simultaneously learn viewpoint representations and maintain coherent reasoning thinking. Experimental results show that our approach significantly activates the spatial reasoning ability of MLLM, improving performance on both in-domain and out-of-domain reasoning tasks. Our findings highlight the value of developing foundational spatial skills in MLLMs, supporting future progress in robotics, autonomous systems, and 3D scene understanding. 

---
# HarnessLLM: Automatic Testing Harness Generation via Reinforcement Learning 

**Authors**: Yujian Liu, Jiabao Ji, Yang Zhang, Wenbo Guo, Tommi Jaakkola, Shiyu Chang  

**Link**: [PDF](https://arxiv.org/pdf/2511.01104)  

**Abstract**: Existing LLM-based automatic test generation methods mainly produce input and expected output pairs to categorize the intended behavior of correct programs. Although straightforward, these methods have limited diversity in generated tests and cannot provide enough debugging information. We propose HarnessLLM, a two-stage training pipeline that enables LLMs to write harness code for testing. Particularly, LLMs generate code that synthesizes inputs and validates the observed outputs, allowing complex test cases and flexible output validation such as invariant checking. To achieve this, we train LLMs with SFT followed by RLVR with a customized reward design. Experiments show that HarnessLLM outperforms input-output-based testing in bug finding and testing strategy diversity. HarnessLLM further benefits the code generation performance through test-time scaling with our generated test cases as inference-phase validation. Our code is available at this https URL. 

---
# Reevaluating Self-Consistency Scaling in Multi-Agent Systems 

**Authors**: Chiyan Loo  

**Link**: [PDF](https://arxiv.org/pdf/2511.00751)  

**Abstract**: This study examines the trade-offs of increasing sampled reasoning paths in self-consistency for modern large language models (LLMs). Earlier research with older models showed that combining multiple reasoning chains improves results before reaching a plateau. Using Gemini 2.5 models on HotpotQA and Math-500, we revisit those claims under current model conditions. Each configuration pooled outputs from varying sampled reasoning paths and compared them to a single chain-of-thought (CoT) baseline. Larger models exhibited a more stable and consistent improvement curve. The results confirm that performance gains taper off after moderate sampling, aligning with past findings. This plateau suggests diminishing returns driven by overlap among reasoning paths. Self-consistency remains useful, but high-sample configurations offer little benefit relative to their computational cost. 

---
# \texttt{ReMind}: Understanding Deductive Code Reasoning in LLMs 

**Authors**: Jun Gao, Yun Peng, Xiaoxue Ren  

**Link**: [PDF](https://arxiv.org/pdf/2511.00488)  

**Abstract**: Large Language Models (LLMs) have achieved remarkable progress in code-related tasks. Despite their advancement, empirical evidence reveals that they still struggle with \emph{deductive code reasoning}, the ability to reason about the program execution process. While prior studies have recognized this limitation, the underlying causes remain largely underexplored. In this paper, we begin by presenting a comprehensive empirical study that reveals three key challenges undermining deductive code reasoning: (1) an intrinsic gap between generation and reasoning abilities, (2) a consistent bias towards code sources, and (3) weak zero-shot generalization on complex benchmarks. In light of these challenges, we propose \texttt{ReMind}, a multi-agent framework composed of \texttt{Mutator}, \texttt{Executor}, and \texttt{Inspector}. The \texttt{Mutator} generates code variants to mitigate bias towards code sources, the \texttt{Executor} traces variable states step-by-step to expose inconsistency, and the \texttt{Inspector} identifies problematic reasoning steps and provides control-flow refinement to bridge the intrinsic reasoning gap. Through their coordinated collaboration, \texttt{ReMind} systematically identifies and refines reasoning flaws, achieving outstanding performance and enabling robust zero-shot generalization. Extensive experiments on two benchmarks with five LLMs demonstrate the superior advantages of \texttt{ReMind} compared to baseline approaches in deductive code reasoning. 

---
# Can SAEs reveal and mitigate racial biases of LLMs in healthcare? 

**Authors**: Hiba Ahsan, Byron C. Wallace  

**Link**: [PDF](https://arxiv.org/pdf/2511.00177)  

**Abstract**: LLMs are increasingly being used in healthcare. This promises to free physicians from drudgery, enabling better care to be delivered at scale. But the use of LLMs in this space also brings risks; for example, such models may worsen existing biases. How can we spot when LLMs are (spuriously) relying on patient race to inform predictions? In this work we assess the degree to which Sparse Autoencoders (SAEs) can reveal (and control) associations the model has made between race and stigmatizing concepts. We first identify SAE latents in Gemma-2 models which appear to correlate with Black individuals. We find that this latent activates on reasonable input sequences (e.g., "African American") but also problematic words like "incarceration". We then show that we can use this latent to steer models to generate outputs about Black patients, and further that this can induce problematic associations in model outputs as a result. For example, activating the Black latent increases the risk assigned to the probability that a patient will become "belligerent". We evaluate the degree to which such steering via latents might be useful for mitigating bias. We find that this offers improvements in simple settings, but is less successful for more realistic and complex clinical tasks. Overall, our results suggest that: SAEs may offer a useful tool in clinical applications of LLMs to identify problematic reliance on demographics but mitigating bias via SAE steering appears to be of marginal utility for realistic tasks. 

---
# LiCoMemory: Lightweight and Cognitive Agentic Memory for Efficient Long-Term Reasoning 

**Authors**: Zhengjun Huang, Zhoujin Tian, Qintian Guo, Fangyuan Zhang, Yingli Zhou, Di Jiang, Xiaofang Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2511.01448)  

**Abstract**: Large Language Model (LLM) agents exhibit remarkable conversational and reasoning capabilities but remain constrained by limited context windows and the lack of persistent memory. Recent efforts address these limitations via external memory architectures, often employing graph-based representations, yet most adopt flat, entangled structures that intertwine semantics with topology, leading to redundant representations, unstructured retrieval, and degraded efficiency and accuracy. To resolve these issues, we propose LiCoMemory, an end-to-end agentic memory framework for real-time updating and retrieval, which introduces CogniGraph, a lightweight hierarchical graph that utilizes entities and relations as semantic indexing layers, and employs temporal and hierarchy-aware search with integrated reranking for adaptive and coherent knowledge retrieval. Experiments on long-term dialogue benchmarks, LoCoMo and LongMemEval, show that LiCoMemory not only outperforms established baselines in temporal reasoning, multi-session consistency, and retrieval efficiency, but also notably reduces update latency. Our official code and data are available at this https URL. 

---
# Contextual Relevance and Adaptive Sampling for LLM-Based Document Reranking 

**Authors**: Jerry Huang, Siddarth Madala, Cheng Niu, Julia Hockenmaier, Tong Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2511.01208)  

**Abstract**: Reranking algorithms have made progress in improving document retrieval quality by efficiently aggregating relevance judgments generated by large language models (LLMs). However, identifying relevant documents for queries that require in-depth reasoning remains a major challenge. Reasoning-intensive queries often exhibit multifaceted information needs and nuanced interpretations, rendering document relevance inherently context dependent. To address this, we propose contextual relevance, which we define as the probability that a document is relevant to a given query, marginalized over the distribution of different reranking contexts it may appear in (i.e., the set of candidate documents it is ranked alongside and the order in which the documents are presented to a reranking model). While prior works have studied methods to mitigate the positional bias LLMs exhibit by accounting for the ordering of documents, we empirically find that the compositions of these batches also plays an important role in reranking performance. To efficiently estimate contextual relevance, we propose TS-SetRank, a sampling-based, uncertainty-aware reranking algorithm. Empirically, TS-SetRank improves nDCG@10 over retrieval and reranking baselines by 15-25% on BRIGHT and 6-21% on BEIR, highlighting the importance of modeling relevance as context-dependent. 

---
